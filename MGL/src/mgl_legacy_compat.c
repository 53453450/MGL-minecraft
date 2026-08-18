/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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

static bool code_has_const_decl(const char *src, const char *name)
{
    /* True if the source already declares "const <type> <name>" (any
     * value); used to source-guard builtin-constant injection so a shader
     * that supplies its own constant (e.g. for testing) is left alone. */
    if (!src || !name) return false;
    char needle[96];
    snprintf(needle, sizeof(needle), "const int %s", name);
    size_t needle_len = strlen(needle);
    scan_state_t state = SCAN_NORMAL;
    const char *p = src;
    while (*p) {
        if (state == SCAN_NORMAL) {
            if (strncmp(p, needle, needle_len) == 0) {
                int after = (unsigned char)p[needle_len];
                if (!is_ident_char(after)) return true;
            }
        }
        state = scan_step(&p, state);
    }
    return false;
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

/* Rename the first `void main(...)` in the source to `void <replacement>`.
 * Comment/string-aware; tolerant of whitespace between tokens and of the
 * `(void)` parameter spelling.  Returns 1 on success, 0 when no main is
 * found or the buffer would overflow. */
static int rename_first_main(char *src, size_t src_capacity,
                             const char *replacement)
{
    if (!src) return 0;
    const char *needle = "void";
    size_t needle_len = strlen(needle);
    scan_state_t state = SCAN_NORMAL;
    char *p = src;
    while (*p) {
        if (state == SCAN_NORMAL) {
            if (strncmp(p, needle, needle_len) == 0) {
                /* token boundary before "void" */
                int before = (p == src) ? 0 : (unsigned char)p[-1];
                if (!is_ident_char(before)) {
                    /* optional whitespace, then "main" */
                    const char *q = p + needle_len;
                    while (*q == ' ' || *q == '\t' || *q == '\r' ||
                           *q == '\n') q++;
                    if (strncmp(q, "main", 4) == 0 &&
                        !is_ident_char((unsigned char)q[4])) {
                        const char *r = q + 4;
                        while (*r == ' ' || *r == '\t' || *r == '\r' ||
                               *r == '\n') r++;
                        if (*r == '(') {
                            size_t repl_len = strlen(replacement);
                            size_t name_len = (size_t)(q - p) + 4;
                            size_t used = strlen(p);
                            if ((size_t)((p - src) + used +
                                         (repl_len - name_len)) + 1 >
                                src_capacity) {
                                return 0;
                            }
                            memmove(p + repl_len, p + name_len,
                                    used - name_len + 1);
                            memcpy(p, replacement, repl_len);
                            return 1;
                        }
                    }
                }
            }
        }
        const char *cp = p;
        state = scan_step(&cp, state);
        p = (char *)cp;
    }
    return 0;
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

static void replace_literal(char *src, size_t src_capacity,
                            const char *needle, const char *replacement)
{
    /* Literal substring replacement (no identifier-boundary check) — for
     * rewriting bracketed forms such as "_mglFragData[0]". */
    if (!src || !needle || !replacement || src_capacity == 0) return;

    size_t needle_len = strlen(needle);
    size_t replacement_len = strlen(replacement);
    if (needle_len == 0) return;

    long diff = (long)replacement_len - (long)needle_len;
    char *cursor = src;

    while ((cursor = strstr(cursor, needle)) != NULL) {
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
            memmove(cursor + replacement_len,
                    cursor + needle_len,
                    strlen(cursor + needle_len) + 1);
        }
        memcpy(cursor, replacement, replacement_len);
        cursor += replacement_len;
    }
}

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
 *   Back-face selection is not emulated. */
typedef struct {
    const char *legacy_name;
    const char *vs_name;
    const char *fs_name;
    const char *type;
    const char *vs_dir;   /* "in", "out", or NULL */
    const char *fs_dir;   /* "in", "out", or NULL */
    int vs_location;      /* fixed-function attribute slot for VS attribute
                           * inputs (gl_Vertex=0, gl_Normal=2, gl_Color=3,
                           * gl_SecondaryColor=4, gl_FogCoord=5);
                           * -1 = linker-assigned */
} legacy_builtin_t;

static const legacy_builtin_t s_builtins[] = {
    /* --- VS attribute inputs (§7.1) --- */
    {"gl_Normal",            "_mglNormal",            NULL,                       "vec3",  "in",  NULL,  2},
    {"gl_Color",             "_mglColor",             "_mglFrontColor",           "vec4",  "in",  "in",  3},
    {"gl_SecondaryColor",    "_mglSecondaryColor",    "_mglFrontSecondaryColor",  "vec4",  "in",  "in",  4},
    {"gl_FogCoord",          "_mglFogCoord",          NULL,                       "float", "in",  NULL,  5},

    /* --- VS varying outputs (§7.1) --- */
    {"gl_FrontColor",          "_mglFrontColor",          NULL, "vec4",  "out", NULL, -1},
    {"gl_BackColor",           "_mglBackColor",           NULL, "vec4",  "out", NULL, -1},
    {"gl_FrontSecondaryColor", "_mglFrontSecondaryColor", NULL, "vec4",  "out", NULL, -1},
    {"gl_BackSecondaryColor",  "_mglBackSecondaryColor",  NULL, "vec4",  "out", NULL, -1},
    {"gl_ClipVertex",          "_mglClipVertex",          NULL, "vec4",  "out", NULL, -1},
    {"gl_FogFragCoord",        "_mglFogFragCoord",        "_mglFogFragCoord", "float", "out", "in", -1},

    {NULL, NULL, NULL, NULL, NULL, NULL, -1}
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

/* GLSL 1.10 built-in compile-time constants (§7.4): injected as const
 * declarations with their ORIGINAL gl_ names (the AIR frontend folds
 * global const initializers) so legacy shaders that size loops/arrays
 * from them keep working.  Values are MGL's actual limits where the
 * engine has a matching cap, else the GLSL 1.10 spec minimum. */
typedef struct legacy_const_t {
    const char *name;  /* builtin constant name (kept verbatim) */
    int         value;
} legacy_const_t;

static const legacy_const_t s_legacy_constants[] = {
    { "gl_MaxLights",                 8 },
    { "gl_MaxClipPlanes",             6 },
    { "gl_MaxTextureUnits",           8 },
    { "gl_MaxTextureCoords",          MGL_LEGACY_MAX_TEX_COORDS },
    { "gl_MaxVertexAttribs",          16 },
    { "gl_MaxVertexUniformComponents", 512 },
    { "gl_MaxVaryingFloats",          32 },
    { "gl_MaxVertexTextureImageUnits", 8 },
    { "gl_MaxCombinedTextureImageUnits", 8 },
    { "gl_MaxTextureImageUnits",      8 },
    { "gl_MaxFragmentUniformComponents", 512 },
    { "gl_MaxDrawBuffers",            MGL_LEGACY_MAX_DRAW_BUFFERS },
    { NULL, 0 }
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
    /* gl_ClipVertex: the legacy eye-space clip-vertex output (VS). */
    features->has_gl_ClipVertex = code_uses_identifier(src, "gl_ClipVertex");
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
    /* gl_FragData[0]-only rewrite state (see Step 3); consumed by the
     * fragment-output declaration injection in Step 4. */
    bool fragdataOnlyIndex0 = false;
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

    /* --- Step 2.5: ftransform() expansion (§7.4) --- */
    /* ftransform() is the fixed-function vertex transform
     * gl_ModelViewProjectionMatrix * gl_Vertex.  Both names are injected by
     * Step 4 (matrix table + gl_Vertex layout-0 declaration), so expanding
     * the call makes ftransform-only shaders compile and render. */
    if (features->has_ftransform && is_vertex) {
        const char *call = "ftransform()";
        const char *expansion = "gl_ModelViewProjectionMatrix * gl_Vertex";
        size_t call_len = strlen(call);
        size_t exp_len = strlen(expansion);
        char *p = src;
        while ((p = strstr(p, call)) != NULL) {
            /* Verify this is a true ftransform() token (identifier
             * boundary on the left) and not e.g. a comment/string
             * (the detector's code_contains has the same leniency;
             * matching its semantics is sufficient here). */
            if (p == src || !is_ident_char((unsigned char)p[-1])) {
                long diff = (long)exp_len - (long)call_len;
                size_t used = strlen(p);
                if ((size_t)((p - src) + used + diff) + 1 > src_capacity) {
                    break;
                }
                memmove(p + exp_len, p + call_len, used - call_len + 1);
                memcpy(p, expansion, exp_len);
                p += exp_len;
                modified = 1;
                continue;
            }
            p += call_len;
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

        /* gl_FragData[0]-only shaders (the common legacy pattern, incl.
         * "#define gl_FragColor gl_FragData[0]" ports): rewrite index-0
         * writes to the scalar color output so the single color-attachment
         * path works end to end.  Any non-zero or dynamic index keeps the
         * array form — the AIR backend cannot codegen array fragment
         * outputs (index > 0 needs real MRT, a later concern). */
        fragdataOnlyIndex0 = true;
        {
            const char *p = src;
            while ((p = strstr(p, "_mglFragData[")) != NULL) {
                const char *idx = p + 13; /* strlen("_mglFragData[") */
                if (*idx != '0' || idx[1] != ']') {
                    fragdataOnlyIndex0 = false;
                    break;
                }
                p = idx + 2;
            }
        }
        if (fragdataOnlyIndex0) {
            replace_literal(src, src_capacity,
                            "_mglFragData[0]", "_mglFragColor");
            modified = 1;
        }
    }

    /* Other builtin variables (table-driven) */
    if (features->has_legacy_builtins) {
        for (int i = 0; s_builtins[i].legacy_name; i++) {
            const legacy_builtin_t *b = &s_builtins[i];
            if (!code_uses_identifier(src, b->legacy_name)) continue;

            /* Fixed-function attribute inputs (VS) keep their ORIGINAL
             * gl_ names, like gl_Vertex: the explicit
             * layout(location = N) declaration below carries the name, so
             * applications can keep calling glGetAttribLocation with the
             * legacy name and bind at the conventional slot. */
            if (is_vertex && b->vs_location >= 0) continue;

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

    /* GLSL 1.10 built-in constants (§7.4): injected verbatim (const int)
     * so the frontend folds them at use sites; source-guarded against
     * shaders that already declare them. */
    for (int i = 0; s_legacy_constants[i].name; i++) {
        const legacy_const_t *c = &s_legacy_constants[i];
        if (!code_uses_identifier(src, c->name)) continue;
        if (code_has_const_decl(src, c->name)) continue;
        off += (size_t)snprintf(preamble + off,
            sizeof(preamble) - off,
            "const int %s = %d;\n", c->name, c->value);
    }

    /* gl_Vertex: the legacy implicit position attribute, injected with
     * layout(location = 0) (the legacy fixed-function attribute slot) so
     * the app can bind vertex data at the conventional location 0; the
     * name is kept verbatim (the frontend accepts gl_-prefixed user
     * declarations) and the GL attribute contract survives.  Source-guarded
     * (ftransform() expansion may have introduced it after detection). */
    if (is_vertex && code_uses_identifier(src, "gl_Vertex")) {
        if (!strstr(src, "layout(location = 0) in vec4 gl_Vertex;")) {
            off += (size_t)snprintf(preamble + off,
                sizeof(preamble) - off,
                "layout(location = 0) in vec4 gl_Vertex;\n");
        }
    }

    /* gl_FragColor / gl_FragData (fragment outputs, mutually exclusive) */
    if (features->has_gl_FragData && is_fragment) {
        if (fragdataOnlyIndex0) {
            /* All gl_FragData writes were at index 0 and were rewritten to
             * the scalar _mglFragColor in Step 3. */
            off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
                "layout(location = 0) out vec4 _mglFragColor;\n");
        } else {
            off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
                "layout(location = 0) out vec4 _mglFragData[%d];\n",
                MGL_LEGACY_MAX_DRAW_BUFFERS);
        }
    } else if (features->has_gl_FragColor && is_fragment) {
        off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
            "layout(location = 0) out vec4 _mglFragColor;\n");
    }

    /* Other builtin variable declarations (table-driven) */
    if (features->has_legacy_builtins) {
        for (int i = 0; s_builtins[i].legacy_name; i++) {
            const legacy_builtin_t *b = &s_builtins[i];
            if (!code_uses_identifier(src, b->legacy_name)) {
                /* Already renamed — check if the renamed name is present.
                 * Mirrors Step 3's fallback: a stage-inapplicable builtin
                 * is renamed with the available vs_name/fs_name. */
                const char *vs = b->vs_name;
                const char *fs = b->fs_name;
                const char *check = is_vertex ? vs : fs;
                if (!check) check = vs ? vs : fs;
                if (!check || !strstr(src, check)) continue;
                /* Fall through to inject declaration for renamed var. */
            }

            const char *dir  = is_vertex ? b->vs_dir : b->fs_dir;
            const char *name = is_vertex ? b->vs_name : b->fs_name;
            if (!dir || !name) {
                /* Stage-inapplicable builtin renamed with the other
                 * stage's name (Step 3 fallback): declare it as a varying
                 * in this stage so the linkage resolves — e.g. the FS
                 * gl_BackColor input in two-sided lighting
                 * (gl_FrontFacing ? gl_Color : gl_BackColor) links to the
                 * VS gl_BackColor output of the same name. */
                const char *fb = b->vs_name ? b->vs_name : b->fs_name;
                if (fb && strstr(src, fb)) {
                    char decl[160];
                    snprintf(decl, sizeof(decl), "%s %s %s;\n",
                             is_vertex ? "out" : "in", b->type, fb);
                    if (!strstr(preamble, decl))
                        off += (size_t)snprintf(preamble + off,
                            sizeof(preamble) - off, "%s", decl);
                }
                continue;
            }
            if (is_vertex && b->vs_location >= 0) {
                /* Fixed-function attribute slots so legacy apps bind data
                 * at the conventional locations (gl_Vertex=0, gl_Normal=2,
                 * gl_Color=3, gl_SecondaryColor=4, gl_FogCoord=5) without
                 * querying renamed names.  The ORIGINAL gl_ name is kept
                 * (Step 3 skipped the rename), matching gl_Vertex. */
                off += (size_t)snprintf(preamble + off, sizeof(preamble) - off,
                    "layout(location = %d) %s %s %s;\n",
                    b->vs_location, dir, b->type, b->legacy_name);
            } else {
                char decl[160];
                snprintf(decl, sizeof(decl), "%s %s %s;\n", dir, b->type, name);
                /* Dedup: several table entries can share a renamed name in
                 * this stage (e.g. FS gl_Color and gl_FrontColor both
                 * become _mglFrontColor); never declare it twice. */
                if (!strstr(preamble, decl))
                    off += (size_t)snprintf(preamble + off,
                        sizeof(preamble) - off, "%s", decl);
            }
        }

        /* gl_MultiTexCoord0..7 declarations (VS attribute inputs only).
         * Legacy fixed-function slots: texcoord i binds at attribute
         * location 8 + i, so legacy apps can bind texture data at the
         * conventional slot without querying the (renamed) attribute. */
        if (is_vertex) {
            for (int i = 0; i < MGL_NUM_MULTITEXCOORD; i++) {
                char modern[32];
                snprintf(modern, sizeof(modern), "_mglMultiTexCoord%d", i);
                /* Check if the original or renamed name is present. */
                char legacy[32];
                snprintf(legacy, sizeof(legacy), "gl_MultiTexCoord%d", i);
                if (code_uses_identifier(src, legacy) ||
                    strstr(src, modern)) {
                    char located[64];
                    snprintf(located, sizeof(located),
                             "layout(location = %d) in vec4 %s;\n",
                             8 + i, modern);
                    if (!strstr(src, located)) {
                        off += (size_t)snprintf(preamble + off,
                            sizeof(preamble) - off,
                            "%s", located);
                    }
                }
            }
        }
    }

    if (off > 0 && off < sizeof(preamble)) {
        inject_after_version(src, src_capacity, preamble);
        modified = 1;
    }

    /* --- Step 5: legacy clip-plane derivation wrapper (VS only) ---
     * gl_ClipVertex is the legacy eye-space clip vertex; the fixed-function
     * runtime derives per-plane clip distances from the GL clip-plane state
     * (_mglClipPlane/_mglClipPlaneEnabled, refreshed by MGL per draw).
     * The user's main is renamed and a wrapper main computes
     * gl_ClipDistance[i] = mix(1, dot(plane_i, clipVertex), enabled_i)
     * after it, so disabled planes (or zero planes) never clip. */
    if (is_vertex && features->has_gl_ClipVertex &&
        !strstr(src, "_mglLegacyUserMain")) {
        if (rename_first_main(src, src_capacity, "void _mglLegacyUserMain")) {
            static const char s_clip_wrapper[] =
                "\n"
                "uniform vec4 _mglClipPlane[8];\n"
                "uniform float _mglClipPlaneEnabled[8];\n"
                "void main() {\n"
                "    _mglLegacyUserMain();\n"
                "    gl_ClipDistance[0] = mix(1.0, dot(_mglClipPlane[0], _mglClipVertex), _mglClipPlaneEnabled[0]);\n"
                "    gl_ClipDistance[1] = mix(1.0, dot(_mglClipPlane[1], _mglClipVertex), _mglClipPlaneEnabled[1]);\n"
                "    gl_ClipDistance[2] = mix(1.0, dot(_mglClipPlane[2], _mglClipVertex), _mglClipPlaneEnabled[2]);\n"
                "    gl_ClipDistance[3] = mix(1.0, dot(_mglClipPlane[3], _mglClipVertex), _mglClipPlaneEnabled[3]);\n"
                "    gl_ClipDistance[4] = mix(1.0, dot(_mglClipPlane[4], _mglClipVertex), _mglClipPlaneEnabled[4]);\n"
                "    gl_ClipDistance[5] = mix(1.0, dot(_mglClipPlane[5], _mglClipVertex), _mglClipPlaneEnabled[5]);\n"
                "    gl_ClipDistance[6] = mix(1.0, dot(_mglClipPlane[6], _mglClipVertex), _mglClipPlaneEnabled[6]);\n"
                "    gl_ClipDistance[7] = mix(1.0, dot(_mglClipPlane[7], _mglClipVertex), _mglClipPlaneEnabled[7]);\n"
                "}\n";
            size_t used = strlen(src);
            if (used + sizeof(s_clip_wrapper) <= src_capacity) {
                memcpy(src + used, s_clip_wrapper,
                       sizeof(s_clip_wrapper));
                modified = 1;
            }
        }
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
