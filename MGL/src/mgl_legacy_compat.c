/*
 * mgl_legacy_compat.c
 * MGL
 *
 * Implementation of the Legacy GLSL Compatibility Subsystem.
 *
 * See mgl_legacy_compat.h for the architectural rationale.  This module
 * performs source-level translation of pre-GLSL-3.30 constructs before the
 * source reaches the frontend (which parses core-profile GLSL 4.50).
 *
 * Reference: GLSLangSpec.1.10.pdf ~ GLSLangSpec.1.50.pdf
 */

#include "mgl_legacy_compat.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/* === Constants === */

/* GL default gl_MaxTextureCoords is 8; array sized to this upper bound. */
#define MGL_LEGACY_MAX_TEX_COORDS 8
/* GL default gl_MaxDrawBuffers is 8; array sized to this upper bound. */
#define MGL_LEGACY_MAX_DRAW_BUFFERS 8

/* === Internal helpers === */

static int is_ident_char(int c)
{
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '_';
}

/* ---- Comment/string-aware scanner ---- */

typedef enum {
    SCAN_NORMAL,
    SCAN_LINE_COMMENT,
    SCAN_BLOCK_COMMENT,
    SCAN_STRING
} scan_state_t;

static scan_state_t scan_step(const char **pp, scan_state_t state)
{
    const char *p = *pp;
    switch (state) {
        case SCAN_NORMAL:
            if (p[0] == '/' && p[1] == '/') {
                *pp = p + 2;
                return SCAN_LINE_COMMENT;
            }
            if (p[0] == '/' && p[1] == '*') {
                *pp = p + 2;
                return SCAN_BLOCK_COMMENT;
            }
            if (p[0] == '"') {
                *pp = p + 1;
                return SCAN_STRING;
            }
            *pp = p + 1;
            return SCAN_NORMAL;

        case SCAN_LINE_COMMENT:
            if (*p == '\n') {
                *pp = p + 1;
                return SCAN_NORMAL;
            }
            *pp = p + 1;
            return SCAN_LINE_COMMENT;

        case SCAN_BLOCK_COMMENT:
            if (p[0] == '*' && p[1] == '/') {
                *pp = p + 2;
                return SCAN_NORMAL;
            }
            *pp = p + 1;
            return SCAN_BLOCK_COMMENT;

        case SCAN_STRING:
            if (p[0] == '\\' && p[1] != '\0') {
                *pp = p + 2;
                return SCAN_STRING;
            }
            if (*p == '"') {
                *pp = p + 1;
                return SCAN_NORMAL;
            }
            *pp = p + 1;
            return SCAN_STRING;
    }
    *pp = p + 1;
    return SCAN_NORMAL;
}

static bool code_uses_identifier(const char *src, const char *name)
{
    if (!src || !name) return false;
    size_t name_len = strlen(name);
    if (name_len == 0) return false;

    scan_state_t state = SCAN_NORMAL;
    const char *p = src;

    while (*p) {
        if (state == SCAN_NORMAL) {
            if (strncmp(p, name, name_len) == 0) {
                int before = (p == src) ? 0 : (unsigned char)p[-1];
                int after = (unsigned char)p[name_len];
                if (!is_ident_char(before) &&
                    !is_ident_char(after) &&
                    before != '.') {
                    return true;
                }
            }
        }
        state = scan_step(&p, state);
    }
    return false;
}

static bool code_contains(const char *src, const char *needle)
{
    if (!src || !needle) return false;
    size_t needle_len = strlen(needle);
    if (needle_len == 0) return false;

    scan_state_t state = SCAN_NORMAL;
    const char *p = src;

    while (*p) {
        if (state == SCAN_NORMAL) {
            if (strncmp(p, needle, needle_len) == 0) {
                return true;
            }
        }
        state = scan_step(&p, state);
    }
    return false;
}

/* ---- Identifier-aware replacement ---- */

static void replace_identifier(char *src, size_t src_capacity,
                               const char *needle, const char *replacement)
{
    if (!src || !needle || !replacement || src_capacity == 0) return;

    size_t needle_len = strlen(needle);
    size_t replacement_len = strlen(replacement);
    if (needle_len == 0) return;

    long diff = (long)replacement_len - (long)needle_len;
    char *cursor = src;

    while ((cursor = strstr(cursor, needle)) != NULL) {
        int before = (cursor == src) ? 0 : (unsigned char)cursor[-1];
        int after = (unsigned char)cursor[needle_len];
        if (is_ident_char(before) || is_ident_char(after) || before == '.') {
            cursor += needle_len;
            continue;
        }

        if (diff > 0) {
            size_t tail_len = strlen(cursor + needle_len);
            size_t used = (size_t)(cursor - src) + needle_len + tail_len + 1;
            if (used + (size_t)diff > src_capacity) {
                cursor += needle_len;
                continue;
            }
            memmove(cursor + replacement_len,
                    cursor + needle_len,
                    tail_len + 1);
        } else if (diff < 0) {
            size_t tail_len = strlen(cursor + needle_len);
            memmove(cursor + replacement_len,
                    cursor + needle_len,
                    tail_len + 1);
        }

        memcpy(cursor, replacement, replacement_len);
        cursor += replacement_len;
    }
}

/* ---- Declaration injection ---- */

static void inject_after_version(char *src, size_t src_capacity,
                                 const char *text)
{
    if (!src || !text || src_capacity == 0) return;

    size_t text_len = strlen(text);
    if (text_len == 0) return;

    size_t src_len = strlen(src);
    if (src_len + text_len + 1 > src_capacity) {
        fprintf(stderr,
                "[MGL] legacy_compat: injection skipped (buffer full, need %zu)\n",
                src_len + text_len + 1);
        return;
    }

    char *version_line = strstr(src, "#version");
    char *insert_point;
    if (version_line) {
        char *newline = strchr(version_line, '\n');
        if (newline) {
            insert_point = newline + 1;
        } else {
            insert_point = version_line + strlen(version_line);
        }
    } else {
        insert_point = src;
    }

    while (*insert_point) {
        char *line_start = insert_point;
        char *p = line_start;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') {
            char *nl = strchr(line_start, '\n');
            if (nl) {
                insert_point = nl + 1;
            } else {
                insert_point = line_start + strlen(line_start);
                break;
            }
        } else {
            break;
        }
    }

    size_t tail_len = strlen(insert_point);
    memmove(insert_point + text_len, insert_point, tail_len + 1);
    memcpy(insert_point, text, text_len);
}

/* === Legacy texture function table (GLSL 1.10 §8.8) ===
 *
 * Each entry maps a legacy texture function name to its GLSL 1.30+
 * equivalent.  Order matters: longer names first so that texture2DProj
 * is replaced before texture2D (identifier-aware replace makes this
 * technically unnecessary, but we keep the ordering for clarity).
 *
 * Shadow functions (texture1DShadow, texture2DShadow, texture1DProjShadow,
 * texture2DProjShadow) are intentionally NOT in this table — see header
 * comment for rationale. */
typedef struct {
    const char *legacy;
    const char *modern;
} texture_fn_map_t;

static const texture_fn_map_t s_texture_fn_map[] = {
    /* Proj variants — must come before non-Proj (longer name first) */
    {"texture1DProj",   "textureProj"},
    {"texture2DProj",   "textureProj"},
    {"texture3DProj",   "textureProj"},
    {"texture2DRect",   "texture"},     /* GL_ARB_texture_rectangle */
    /* Non-Proj variants */
    {"texture1D",       "texture"},
    {"texture2D",       "texture"},
    {"texture3D",       "texture"},
    {"textureCube",     "texture"},
    {NULL, NULL}
};

/* === Legacy builtin variable table (GLSL 1.10 §7.1, §7.2) ===
 *
 * Each entry describes a builtin variable removed in core profile 3.30.
 * Fields:
 *   legacy_name - the gl_* identifier to detect and rename
 *   vs_name     - replacement name in vertex shader (NULL = not used in VS)
 *   fs_name     - replacement name in fragment shader (NULL = not used in FS)
 *   type        - GLSL type string (e.g. "vec4")
 *   vs_dir      - direction in VS: "in" (attribute), "out" (varying), NULL
 *   fs_dir      - direction in FS: "in" (varying), "out" (output), NULL
 *
 * Note on gl_Color / gl_SecondaryColor:
 *   In VS these are attribute inputs; in FS they are varying inputs that
 *   correspond to VS's gl_FrontColor / gl_FrontSecondaryColor respectively
 *   (selected by gl_FrontFacing in fixed-function GL).  We rename FS's
 *   gl_Color to _mglFrontColor so it links with VS's gl_FrontColor output.
 *   Back-face selection is not emulated (Phase 3 concern). */
typedef struct {
    const char *legacy_name;
    const char *vs_name;
    const char *fs_name;
    const char *type;
    const char *vs_dir;   /* "in", "out", or NULL */
    const char *fs_dir;   /* "in", "out", or NULL */
} legacy_builtin_t;

static const legacy_builtin_t s_builtins[] = {
    /* --- VS attribute inputs (§7.1) --- */
    {"gl_Normal",            "_mglNormal",            NULL,                       "vec3",  "in",  NULL},
    {"gl_Color",             "_mglColor",             "_mglFrontColor",           "vec4",  "in",  "in"},
    {"gl_SecondaryColor",    "_mglSecondaryColor",    "_mglFrontSecondaryColor",  "vec4",  "in",  "in"},
    {"gl_FogCoord",          "_mglFogCoord",          NULL,                       "float", "in",  NULL},

    /* --- VS varying outputs (§7.1) --- */
    {"gl_FrontColor",          "_mglFrontColor",          NULL, "vec4",  "out", NULL},
    {"gl_BackColor",           "_mglBackColor",           NULL, "vec4",  "out", NULL},
    {"gl_FrontSecondaryColor", "_mglFrontSecondaryColor", NULL, "vec4",  "out", NULL},
    {"gl_BackSecondaryColor",  "_mglBackSecondaryColor",  NULL, "vec4",  "out", NULL},
    {"gl_ClipVertex",          "_mglClipVertex",          NULL, "vec4",  "out", NULL},
    {"gl_FogFragCoord",        "_mglFogFragCoord",        "_mglFogFragCoord", "float", "out", "in"},

    {NULL, NULL, NULL, NULL, NULL, NULL}
};

/* Legacy matrix built-in uniforms (§7.4).  These are injected with their
 * ORIGINAL gl_ names (the AIR frontend accepts gl_-prefixed user-declared
 * uniforms), so the GL-side uniform contract is unchanged: applications keep
 * resolving e.g. "gl_ModelViewProjectionMatrix" and setting it directly. */
typedef struct legacy_matrix_t {
    const char *name;       /* builtin uniform name (kept verbatim) */
    const char *type;       /* mat3 / mat4 */
    int         array_size; /* 0 = scalar, >0 = array of this size */
} legacy_matrix_t;

static const legacy_matrix_t s_legacy_matrices[] = {
    { "gl_ModelViewMatrix",                     "mat4", 0 },
    { "gl_ProjectionMatrix",                    "mat4", 0 },
    { "gl_ModelViewProjectionMatrix",           "mat4", 0 },
    { "gl_TextureMatrix",                       "mat4", MGL_LEGACY_MAX_TEX_COORDS },
    { "gl_NormalMatrix",                        "mat3", 0 },
    { "gl_ModelViewMatrixInverse",              "mat4", 0 },
    { "gl_ModelViewMatrixTranspose",            "mat4", 0 },
    { "gl_ModelViewMatrixInverseTranspose",     "mat4", 0 },
    { "gl_ProjectionMatrixInverse",             "mat4", 0 },
    { "gl_ProjectionMatrixTranspose",           "mat4", 0 },
    { "gl_ProjectionMatrixInverseTranspose",    "mat4", 0 },
    { "gl_ModelViewProjectionMatrixInverse",    "mat4", 0 },
    { "gl_ModelViewProjectionMatrixTranspose",  "mat4", 0 },
    { "gl_ModelViewProjectionMatrixInverseTranspose", "mat4", 0 },
    { "gl_TextureMatrixInverse",                "mat4", MGL_LEGACY_MAX_TEX_COORDS },
    { "gl_TextureMatrixTranspose",              "mat4", MGL_LEGACY_MAX_TEX_COORDS },
    { "gl_TextureMatrixInverseTranspose",       "mat4", MGL_LEGACY_MAX_TEX_COORDS },
    { NULL, NULL, 0 }
};

/* gl_MultiTexCoord0..7 generated dynamically (8 entries). */
#define MGL_NUM_MULTITEXCOORD 8

/* === Public API === */

void mgl_legacy_detect(const char *src, mgl_legacy_features_t *features)
{
    if (!src || !features) return;

    memset(features, 0, sizeof(*features));

    /* Keywords */
    features->has_attribute = code_uses_identifier(src, "attribute");
    features->has_varying   = code_uses_identifier(src, "varying");

    /* Fragment outputs */
    features->has_gl_FragColor = code_uses_identifier(src, "gl_FragColor");
    features->has_gl_TexCoord  = code_uses_identifier(src, "gl_TexCoord");
    features->has_gl_FragData  = code_uses_identifier(src, "gl_FragData");

    /* Legacy texture functions */
    for (int i = 0; s_texture_fn_map[i].legacy; i++) {
        if (code_uses_identifier(src, s_texture_fn_map[i].legacy)) {
            /* Map table entry to features field.  We could use a more
             * elegant mapping, but a simple chain is clear enough. */
            const char *name = s_texture_fn_map[i].legacy;
            if      (strcmp(name, "texture1D")     == 0) features->has_texture1D     = GL_TRUE;
            else if (strcmp(name, "texture1DProj") == 0) features->has_texture1DProj = GL_TRUE;
            else if (strcmp(name, "texture2D")     == 0) features->has_texture2D     = GL_TRUE;
            else if (strcmp(name, "texture2DProj") == 0) features->has_texture2DProj = GL_TRUE;
            else if (strcmp(name, "texture3D")     == 0) features->has_texture3D     = GL_TRUE;
            else if (strcmp(name, "texture3DProj") == 0) features->has_texture3DProj = GL_TRUE;
            else if (strcmp(name, "textureCube")   == 0) features->has_textureCube   = GL_TRUE;
            else if (strcmp(name, "texture2DRect") == 0) features->has_texture2DRect = GL_TRUE;
        }
    }

    /* Legacy builtin variables */
    for (int i = 0; s_builtins[i].legacy_name; i++) {
        if (code_uses_identifier(src, s_builtins[i].legacy_name)) {
            features->has_legacy_builtins = GL_TRUE;
            break;
        }
    }
    /* gl_Vertex (implicit legacy position attribute) */
    if (!features->has_legacy_builtins) {
        if (code_uses_identifier(src, "gl_Vertex")) {
            features->has_legacy_builtins = GL_TRUE;
        }
    }
    /* gl_MultiTexCoord0..7 */
    if (!features->has_legacy_builtins) {
        for (int i = 0; i < MGL_NUM_MULTITEXCOORD; i++) {
            char name[32];
            snprintf(name, sizeof(name), "gl_MultiTexCoord%d", i);
            if (code_uses_identifier(src, name)) {
                features->has_legacy_builtins = GL_TRUE;
                break;
            }
        }
    }

    /* Legacy matrix built-in uniforms (§7.4) */
    for (int i = 0; s_legacy_matrices[i].name; i++) {
        if (code_uses_identifier(src, s_legacy_matrices[i].name)) {
            features->has_legacy_matrices = GL_TRUE;
            break;
        }
    }

    /* ftransform() */
    features->has_ftransform = code_contains(src, "ftransform()");

    /* Aggregate */
    features->needs_translation =
        features->has_attribute        ||
        features->has_varying          ||
        features->has_gl_FragColor     ||
        features->has_gl_TexCoord      ||
        features->has_gl_FragData      ||
        features->has_texture1D        ||
        features->has_texture1DProj    ||
        features->has_texture2D        ||
        features->has_texture2DProj    ||
        features->has_texture3D        ||
        features->has_texture3DProj    ||
        features->has_textureCube      ||
        features->has_texture2DRect    ||
        features->has_legacy_builtins  ||
        features->has_legacy_matrices ||
        features->has_ftransform;
}

int mgl_translate_legacy_glsl(char *src,
                              size_t src_capacity,
                              GLuint shader_type,
                              int version,
                              const mgl_legacy_features_t *features)
{
    if (!src || src_capacity == 0) return -1;

    if (version >= 330) return 0;

    /* Perform detection if caller didn't supply features. */
    mgl_legacy_features_t local_features;
    if (!features) {
        mgl_legacy_detect(src, &local_features);
        features = &local_features;
    }

    if (!features->needs_translation) return 0;

    int modified = 0;
    bool is_vertex = (shader_type == GL_VERTEX_SHADER);
    bool is_fragment = (shader_type == GL_FRAGMENT_SHADER);

    /* --- Step 1: Keyword rewrites (attribute / varying) --- */

    if (features->has_attribute) {
        replace_identifier(src, src_capacity, "attribute", "in");
        modified = 1;
    }

    if (features->has_varying) {
        const char *replacement = is_vertex ? "out" : "in";
        replace_identifier(src, src_capacity, "varying", replacement);
        modified = 1;
    }

    /* --- Step 2: Legacy texture function rewrites (§8.8) --- */

    for (int i = 0; s_texture_fn_map[i].legacy; i++) {
        if (code_uses_identifier(src, s_texture_fn_map[i].legacy)) {
            replace_identifier(src, src_capacity,
                               s_texture_fn_map[i].legacy,
                               s_texture_fn_map[i].modern);
            modified = 1;
        }
    }

    /* --- Step 3: Builtin variable rewrites (§7.1, §7.2) --- */

    /* gl_FragColor / gl_TexCoord / gl_FragData (fragment-specific) */
    if (features->has_gl_FragColor && is_fragment) {
        replace_identifier(src, src_capacity,
                           "gl_FragColor", "_mglFragColor");
        modified = 1;
    }

    if (features->has_gl_TexCoord) {
        replace_identifier(src, src_capacity,
                           "gl_TexCoord", "_mglTexCoord");
        modified = 1;
    }

    if (features->has_gl_FragData && is_fragment) {
        replace_identifier(src, src_capacity,
                           "gl_FragData", "_mglFragData");
        modified = 1;
    }

    /* Other builtin variables (table-driven) */
    if (features->has_legacy_builtins) {
        for (int i = 0; s_builtins[i].legacy_name; i++) {
            const legacy_builtin_t *b = &s_builtins[i];
            if (!code_uses_identifier(src, b->legacy_name)) continue;

            /* Pick the stage-appropriate replacement name. */
            const char *new_name = is_vertex ? b->vs_name : b->fs_name;
            if (!new_name) {
                /* This builtin is not valid in the current stage.
                 * Rename anyway to avoid the frontend rejecting the gl_ name,
                 * using the available name (vs_name or fs_name). */
                new_name = b->vs_name ? b->vs_name : b->fs_name;
                if (!new_name) continue;
            }
            replace_identifier(src, src_capacity,
                               b->legacy_name, new_name);
            modified = 1;
        }

        /* gl_MultiTexCoord0..7 */
        for (int i = 0; i < MGL_NUM_MULTITEXCOORD; i++) {
            char legacy[32], modern[32];
            snprintf(legacy, sizeof(legacy), "gl_MultiTexCoord%d", i);
            snprintf(modern, sizeof(modern), "_mglMultiTexCoord%d", i);
            if (code_uses_identifier(src, legacy)) {
                replace_identifier(src, src_capacity, legacy, modern);
                modified = 1;
            }
        }
    }

    /* --- Step 4: Inject declarations for renamed builtins --- */

    char preamble[2048];
    size_t off = 0;

    off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
        "/* MGL legacy GLSL translation: renamed builtins declared as\n"
        " * user variables so the frontend can parse them. */\n");

    /* gl_TexCoord declaration (array, both VS out and FS in) */
    if (features->has_gl_TexCoord) {
        const char *dir = is_vertex ? "out" : "in";
        off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
            "%s vec4 _mglTexCoord[%d];\n", dir, MGL_LEGACY_MAX_TEX_COORDS);
    }

    /* Legacy matrix built-in uniforms (§7.4): injected verbatim so the
     * GL-side uniform names survive (app keeps glGetUniformLocation on the
     * original gl_ names). */
    if (features->has_legacy_matrices) {
        for (int i = 0; s_legacy_matrices[i].name; i++) {
            const legacy_matrix_t *m = &s_legacy_matrices[i];
            if (!code_uses_identifier(src, m->name)) continue;
            if (m->array_size > 0) {
                off += (size_t)snprintf(preamble + off,
                    sizeof(preamble) - off,
                    "uniform %s %s[%d];\n",
                    m->type, m->name, m->array_size);
            } else {
                off += (size_t)snprintf(preamble + off,
                    sizeof(preamble) - off,
                    "uniform %s %s;\n", m->type, m->name);
            }
        }
    }

    /* gl_FragColor / gl_FragData (fragment outputs, mutually exclusive) */
    if (features->has_gl_FragData && is_fragment) {
        off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
            "layout(location = 0) out vec4 _mglFragData[%d];\n",
            MGL_LEGACY_MAX_DRAW_BUFFERS);
    } else if (features->has_gl_FragColor && is_fragment) {
        off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
            "layout(location = 0) out vec4 _mglFragColor;\n");
    }

    /* Other builtin variable declarations (table-driven) */
    if (features->has_legacy_builtins) {
        for (int i = 0; s_builtins[i].legacy_name; i++) {
            const legacy_builtin_t *b = &s_builtins[i];
            if (!code_uses_identifier(src, b->legacy_name)) {
                /* Already renamed — check if the renamed name is present. */
                const char *vs = b->vs_name;
                const char *fs = b->fs_name;
                const char *check = is_vertex ? vs : fs;
                if (!check || !strstr(src, check)) continue;
                /* Fall through to inject declaration for renamed var. */
            }

            const char *dir  = is_vertex ? b->vs_dir : b->fs_dir;
            const char *name = is_vertex ? b->vs_name : b->fs_name;
            if (!dir || !name) {
                /* Not applicable to this stage.  But the identifier may
                 * still have been renamed using the other stage's name;
                 * skip declaration injection in that case. */
                continue;
            }
            off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
                "%s %s %s;\n", dir, b->type, name);
        }

        /* gl_Vertex: the legacy implicit position attribute, injected with
         * layout(location = 0) (the legacy fixed-function attribute slot) so
         * the app can bind vertex data at the conventional location 0; the
         * name is kept verbatim (the frontend accepts gl_-prefixed user
         * declarations) and the GL attribute contract survives. */
        if (is_vertex && code_uses_identifier(src, "gl_Vertex")) {
            if (!strstr(src, "layout(location = 0) in vec4 gl_Vertex;")) {
                off += (size_t)snprintf(preamble + off,
                    sizeof(preamble) - off,
                    "layout(location = 0) in vec4 gl_Vertex;\n");
            }
        }

        /* gl_MultiTexCoord0..7 declarations (VS attribute inputs only) */
        if (is_vertex) {
            for (int i = 0; i < MGL_NUM_MULTITEXCOORD; i++) {
                char modern[32];
                snprintf(modern, sizeof(modern), "_mglMultiTexCoord%d", i);
                /* Check if the original or renamed name is present. */
                char legacy[32];
                snprintf(legacy, sizeof(legacy), "gl_MultiTexCoord%d", i);
                if (code_uses_identifier(src, legacy) ||
                    strstr(src, modern)) {
                    off += (size_t)snprintf(preamble + off,
                        sizeof(preamble) - off,
                        "in vec4 %s;\n", modern);
                }
            }
        }
    }

    if (off > 0 && off < sizeof(preamble)) {
        inject_after_version(src, src_capacity, preamble);
        modified = 1;
    }

    if (modified) {
        fprintf(stderr,
                "[MGL] Legacy GLSL %d (%s): translated (attr=%d vary=%d "
                "fragColor=%d texCoord=%d fragData=%d texFn=%d builtins=%d matrices=%d)\n",
                version,
                is_vertex ? "VS" : (is_fragment ? "FS" : "other"),
                features->has_attribute,
                features->has_varying,
                features->has_gl_FragColor,
                features->has_gl_TexCoord,
                features->has_gl_FragData,
                features->has_texture1D || features->has_texture1DProj ||
                features->has_texture2D || features->has_texture2DProj ||
                features->has_texture3D || features->has_texture3DProj ||
                features->has_textureCube || features->has_texture2DRect,
                features->has_legacy_builtins,
                features->has_legacy_matrices);
    }

    return modified;
}
