/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * shaders.c
 * MGL
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>

#include "shaders.h"
#include "glm_context.h"

const glslang_resource_t* glslang_default_resource(void);

#define MGL_MAX_UBO_BINDINGS 512

typedef struct {
    char *name;
    int binding;
} MGLUBOBindingEntry;

static MGLUBOBindingEntry s_ubo_binding_entries[MGL_MAX_UBO_BINDINGS];
static int s_ubo_binding_count = 0;

static int mgl_is_identifier_char(int c);

static int mgl_get_or_assign_ubo_binding(const char *block_name)
{
    if (!block_name || !block_name[0]) {
        return 0;
    }

    for (int i = 0; i < s_ubo_binding_count; i++) {
        if (s_ubo_binding_entries[i].name && strcmp(s_ubo_binding_entries[i].name, block_name) == 0) {
            return s_ubo_binding_entries[i].binding;
        }
    }

    if (s_ubo_binding_count >= MGL_MAX_UBO_BINDINGS) {
        /* Fallback: deterministic but bounded binding index */
        unsigned hash = 2166136261u;
        for (const char *p = block_name; *p; p++) {
            hash ^= (unsigned char)*p;
            hash *= 16777619u;
        }
        return (int)(hash % 256u);
    }

    s_ubo_binding_entries[s_ubo_binding_count].name = strdup(block_name);
    s_ubo_binding_entries[s_ubo_binding_count].binding = s_ubo_binding_count;
    s_ubo_binding_count++;
    return s_ubo_binding_entries[s_ubo_binding_count - 1].binding;
}

/* Parse a GLSL layout qualifier list (the content between "layout(" and ")")
 * and normalize it for SPIR-V / Metal compatibility.
 *
 * - Replaces "packed" and "shared" with "std140" (SPIR-V only supports std140 / std430).
 * - Strips leading / trailing whitespace from individual qualifiers.
 * - Returns the number of characters needed to represent the normalized qualifier
 *   list (not including a NUL terminator).
 */
static size_t mgl_normalize_layout_qualifiers(char *dst, size_t dst_capacity,
                                               const char *qualifier_src, size_t qualifier_len,
                                               int binding)
{
    char work[256];
    char binding_str[32];
    int binding_written = 0;

    if (qualifier_len >= sizeof(work)) {
        return 0;
    }
    memcpy(work, qualifier_src, qualifier_len);
    work[qualifier_len] = '\0';

    /* Build normalized qualifier list in dst. GLSL layout qualifiers are
     * comma-separated; qualifiers such as "binding = 0" may contain spaces.
     */
    size_t out = 0;
    const char *cursor = work;
    int first = 1;

    while (*cursor) {
        /* Skip whitespace and commas between qualifiers. */
        while (*cursor && (isspace((unsigned char)*cursor) || *cursor == ',')) {
            cursor++;
        }
        if (!*cursor) {
            break;
        }

        /* Find the end of this comma-delimited qualifier. */
        const char *start = cursor;
        while (*cursor && *cursor != ',') {
            cursor++;
        }
        const char *end = cursor;
        while (end > start && isspace((unsigned char)end[-1])) {
            end--;
        }
        size_t tok_len = (size_t)(end - start);
        if (tok_len == 0) {
            continue;
        }

        /* Skip "packed" and "shared" — replace them with "std140" later. */
        if ((tok_len == 6 && memcmp(start, "packed", 6) == 0) ||
            (tok_len == 6 && memcmp(start, "shared", 6) == 0)) {
            continue;
        }

        /* Preserve any existing binding qualifier and its value. */
        if (tok_len >= 7 && memcmp(start, "binding", 7) == 0) {
            binding_written = 1;
            if (!first && out < dst_capacity) {
                dst[out++] = ',';
                dst[out++] = ' ';
            }
            if (out + tok_len < dst_capacity) {
                memcpy(dst + out, start, tok_len);
                out += tok_len;
            }
            first = 0;
            continue;
        }

        if (!first && out + 2 < dst_capacity) {
            dst[out++] = ',';
            dst[out++] = ' ';
        }

        if (out + tok_len < dst_capacity) {
            memcpy(dst + out, start, tok_len);
            out += tok_len;
        }
        first = 0;
    }

    /* Always emit "std140" as the packing qualifier. */
    if (!first && out + 2 < dst_capacity) {
        dst[out++] = ',';
        dst[out++] = ' ';
    }
    if (out + 6 < dst_capacity) {
        memcpy(dst + out, "std140", 6);
        out += 6;
    }

    /* Append our binding unless the source already had one. */
    if (!binding_written && binding >= 0) {
        int needed = snprintf(binding_str, sizeof(binding_str), ", binding = %d", binding);
        if (needed > 0 && out + (size_t)needed < dst_capacity) {
            memcpy(dst + out, binding_str, (size_t)needed);
            out += (size_t)needed;
        }
    }

    return out;
}

/* Scan GLSL source for "layout(...) uniform BlockName" declarations and ensure
 * each one carries a binding = N qualifier required for SPIR-V / Metal.
 * Unsupported packing modes (packed, shared) are normalized to std140.
 * Returns true if any bindings were injected (requiring version upgrade).
 */
static bool mgl_patch_uniform_block_bindings(char *src, size_t src_capacity)
{
    char *cursor = src;
    bool patched = false;

    if (!src || src_capacity == 0) {
        return false;
    }

    while (*cursor) {
        /* 1. Find "layout(". */
        char *layout = strstr(cursor, "layout(");
        if (!layout) {
            break;
        }

        /* 2. Find matching ")". */
        char *paren_open = layout + 7; /* skip "layout(" */
        char *paren_close = strchr(paren_open, ')');
        if (!paren_close) {
            cursor = layout + 7;
            continue;
        }

        /* 3. Skip whitespace after ")" and check for "uniform". */
        char *after_paren = paren_close + 1;
        while (*after_paren && isspace((unsigned char)*after_paren)) {
            after_paren++;
        }
        if (strncmp(after_paren, "uniform", 7) != 0) {
            cursor = paren_close + 1;
            continue;
        }

        /* 4. Extract block name. */
        char *name_start = after_paren + 7; /* skip "uniform" */
        while (*name_start && isspace((unsigned char)*name_start)) {
            name_start++;
        }
        if (!isalpha((unsigned char)*name_start) && *name_start != '_') {
            cursor = paren_close + 1;
            continue;
        }

        char block_name[128];
        size_t bn = 0;
        char *p = name_start;
        while ((*p == '_' || isalnum((unsigned char)*p)) && bn + 1 < sizeof(block_name)) {
            block_name[bn++] = *p++;
        }
        block_name[bn] = '\0';
        if (bn == 0) {
            cursor = paren_close + 1;
            continue;
        }

        char *after_name = p;
        while (*after_name && isspace((unsigned char)*after_name)) {
            after_name++;
        }
        if (*after_name != '{') {
            cursor = paren_close + 1;
            continue;
        }

        int binding = mgl_get_or_assign_ubo_binding(block_name);

        /* 5. Normalize the layout qualifier. */
        char normalized[256];
        size_t qualifier_len = (size_t)(paren_close - paren_open);
        size_t norm_len = mgl_normalize_layout_qualifiers(
            normalized, sizeof(normalized),
            paren_open, qualifier_len, binding);
        if (norm_len == 0 || norm_len >= sizeof(normalized)) {
            cursor = paren_close + 1;
            continue;
        }

        /* 6. Build the replacement: "layout(NORM) uniform " */
        char replacement[320];
        int repl_total = snprintf(replacement, sizeof(replacement),
                                  "layout(%.*s) uniform ",
                                  (int)norm_len, normalized);
        if (repl_total < 0 || (size_t)repl_total >= sizeof(replacement)) {
            cursor = paren_close + 1;
            continue;
        }
        size_t repl_len = (size_t)repl_total;

        /* 7. Compute old span length and perform the replacement. */
        size_t old_len = (size_t)(after_paren + 7 - layout); /* "layout(...) uniform " */

        if (repl_len <= old_len) {
            /* New text is shorter or equal: in-place overwrite + space-pad. */
            memcpy(layout, replacement, repl_len);
            if (repl_len < old_len) {
                memset(layout + repl_len, ' ', old_len - repl_len);
            }
            cursor = layout + old_len;
            patched = true;
        } else {
            /* New text is longer: shift the tail right. */
            size_t tail_len = strlen(layout + old_len);
            size_t used = strlen(src);
            size_t grow = repl_len - old_len;
            if (used + grow + 1 >= src_capacity) {
                /* No room to grow safely; skip this block. */
                cursor = layout + old_len;
                continue;
            }
            memmove(layout + repl_len, layout + old_len, tail_len + 1);
            memcpy(layout, replacement, repl_len);
            cursor = layout + repl_len;
            patched = true;
        }
    }

    cursor = src;
    while (*cursor) {
        char *uniform_kw = strstr(cursor, "uniform");
        if (!uniform_kw) {
            break;
        }

        int before = uniform_kw == src ? 0 : uniform_kw[-1];
        int after = uniform_kw[7];
        if (mgl_is_identifier_char(before) || mgl_is_identifier_char(after)) {
            cursor = uniform_kw + 7;
            continue;
        }

        char *prev = uniform_kw;
        while (prev > src && isspace((unsigned char)prev[-1])) {
            prev--;
        }
        if (prev > src && prev[-1] == ')') {
            cursor = uniform_kw + 7;
            continue;
        }

        char *name_start = uniform_kw + 7;
        while (*name_start && isspace((unsigned char)*name_start)) {
            name_start++;
        }
        if (!isalpha((unsigned char)*name_start) && *name_start != '_') {
            cursor = uniform_kw + 7;
            continue;
        }

        char block_name[128];
        size_t bn = 0;
        char *p = name_start;
        while ((*p == '_' || isalnum((unsigned char)*p)) && bn + 1 < sizeof(block_name)) {
            block_name[bn++] = *p++;
        }
        block_name[bn] = '\0';

        char *after_name = p;
        while (*after_name && isspace((unsigned char)*after_name)) {
            after_name++;
        }
        if (*after_name != '{') {
            cursor = uniform_kw + 7;
            continue;
        }

        int binding = mgl_get_or_assign_ubo_binding(block_name);
        char replacement[256];
        int repl_total = snprintf(replacement, sizeof(replacement),
                                  "layout(std140, binding = %d) uniform ",
                                  binding);
        if (repl_total < 0 || (size_t)repl_total >= sizeof(replacement)) {
            cursor = uniform_kw + 7;
            continue;
        }

        size_t repl_len = (size_t)repl_total;
        size_t old_len = (size_t)(name_start - uniform_kw);
        size_t tail_len = strlen(uniform_kw + old_len);
        size_t used = strlen(src);
        size_t grow = repl_len - old_len;
        if (repl_len < old_len || used + grow + 1 >= src_capacity) {
            cursor = uniform_kw + old_len;
            continue;
        }

        memmove(uniform_kw + repl_len, uniform_kw + old_len, tail_len + 1);
        memcpy(uniform_kw, replacement, repl_len);
        cursor = uniform_kw + repl_len;
        patched = true;
    }

    return patched;
}

static void mgl_ensure_420pack_extension(char *src, size_t src_capacity)
{
    const char *ext_line = "#extension GL_ARB_shading_language_420pack : require\n";
    size_t ext_len = strlen(ext_line);
    char *version_line;
    char *newline;
    size_t used;

    if (!src || src_capacity == 0) {
        return;
    }
    if (!strstr(src, "binding = ")) {
        return;
    }
    if (strstr(src, "GL_ARB_shading_language_420pack")) {
        return;
    }

    version_line = strstr(src, "#version");
    if (!version_line) {
        return;
    }
    newline = strchr(version_line, '\n');
    if (!newline) {
        return;
    }

    used = strlen(src);
    if (used + ext_len + 1 >= src_capacity) {
        return;
    }

    memmove(newline + 1 + ext_len, newline + 1, strlen(newline + 1) + 1);
    memcpy(newline + 1, ext_line, ext_len);
}

static void mgl_upgrade_version_for_bindings(char *src)
{
    char *version_line;
    int version = 0;
    char profile[32] = {0};
    char replacement[64];
    char *newline;
    size_t old_len;
    size_t new_len;

    if (!src) {
        return;
    }
    if (!strstr(src, "binding = ")) {
        return;
    }

    version_line = strstr(src, "#version");
    if (!version_line) {
        return;
    }

    if (sscanf(version_line, "#version %d %31s", &version, profile) < 1) {
        return;
    }
    if (version >= 420) {
        return;
    }

    newline = strchr(version_line, '\n');
    if (!newline) {
        return;
    }

    snprintf(replacement, sizeof(replacement), "#version 420 core");
    old_len = (size_t)(newline - version_line);
    new_len = strlen(replacement);

    if (new_len <= old_len) {
        memset(version_line, ' ', old_len);
        memcpy(version_line, replacement, new_len);
    } else {
        size_t rest = strlen(newline);
        memmove(version_line + new_len, newline, rest + 1);
        memcpy(version_line, replacement, new_len);
    }
}

static int mgl_is_identifier_char(int c)
{
    return c == '_' || isalnum((unsigned char)c);
}

static void mgl_replace_glsl_identifier_with_shorter(char *src,
                                                     const char *needle,
                                                     const char *replacement)
{
    size_t needle_len;
    size_t replacement_len;
    char *cursor;

    if (!src || !needle || !replacement) {
        return;
    }

    needle_len = strlen(needle);
    replacement_len = strlen(replacement);
    if (needle_len == 0 || replacement_len > needle_len) {
        return;
    }

    cursor = src;
    while ((cursor = strstr(cursor, needle)) != NULL) {
        int before = cursor == src ? 0 : cursor[-1];
        int after = cursor[needle_len];
        if (mgl_is_identifier_char(before) || mgl_is_identifier_char(after)) {
            cursor += needle_len;
            continue;
        }

        memcpy(cursor, replacement, replacement_len);
        if (replacement_len < needle_len) {
            memmove(cursor + replacement_len,
                    cursor + needle_len,
                    strlen(cursor + needle_len) + 1);
        }
        cursor += replacement_len;
    }
}

static void mgl_downgrade_derivative_control_intrinsics(char *src)
{
    if (!src || (!strstr(src, "Fine") && !strstr(src, "Coarse"))) {
        return;
    }

    mgl_replace_glsl_identifier_with_shorter(src, "dFdxFine", "dFdx");
    mgl_replace_glsl_identifier_with_shorter(src, "dFdyFine", "dFdy");
    mgl_replace_glsl_identifier_with_shorter(src, "fwidthFine", "fwidth");
    mgl_replace_glsl_identifier_with_shorter(src, "dFdxCoarse", "dFdx");
    mgl_replace_glsl_identifier_with_shorter(src, "dFdyCoarse", "dFdy");
    mgl_replace_glsl_identifier_with_shorter(src, "fwidthCoarse", "fwidth");
}

const char *getShaderTypeStr(GLuint type)
{
    static const char *types[] = {"VERTEX_SHADER", "FRAGMENT_SHADER",
        "GEOMETRY_SHADER", "TESS_CONTROL_SHADER", "TESS_EVALUATION_SHADER",
        "COMPUTE_SHADER", "MAX_SHADER_TYPES", NULL};

    if (type >= _MAX_SHADER_TYPES)
        return "UNKNOWN_SHADER";

    return types[type];
};

GLuint glShaderTypeToGLMType(GLuint type)
{
    switch(type) {
        case GL_VERTEX_SHADER: return _VERTEX_SHADER;
        case GL_FRAGMENT_SHADER: return _FRAGMENT_SHADER;
        case GL_GEOMETRY_SHADER: return _GEOMETRY_SHADER;
        case GL_TESS_CONTROL_SHADER: return _TESS_CONTROL_SHADER;
        case GL_TESS_EVALUATION_SHADER: return _TESS_EVALUATION_SHADER;
        case GL_COMPUTE_SHADER: return _COMPUTE_SHADER;
        default:
            // CRITICAL FIX: Handle unknown shader types gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Unknown shader type 0x%x, defaulting to vertex shader\n", type);
            return _VERTEX_SHADER;
    }
}

glslang_stage_t getGLSLStage(GLuint type)
{
    switch(type) {
        case GL_VERTEX_SHADER: return GLSLANG_STAGE_VERTEX;
        case GL_FRAGMENT_SHADER: return GLSLANG_STAGE_FRAGMENT;
        case GL_GEOMETRY_SHADER: return GLSLANG_STAGE_GEOMETRY;
        case GL_TESS_CONTROL_SHADER: return GLSLANG_STAGE_TESSCONTROL;
        case GL_TESS_EVALUATION_SHADER: return GLSLANG_STAGE_TESSEVALUATION;
        case GL_COMPUTE_SHADER: return GLSLANG_STAGE_COMPUTE;
        default:
            // CRITICAL FIX: Handle unknown shader types gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Unknown GLSL shader type 0x%x, defaulting to vertex\n", type);
            return GLSLANG_STAGE_VERTEX;
    }

    return 0;
}

void initGLSLInput(GLMContext ctx, GLuint type, const char *src, glslang_input_t *input)
{
    input->language = GLSLANG_SOURCE_GLSL;
    input->stage = getGLSLStage(type);
    input->client = GLSLANG_CLIENT_OPENGL;
    input->target_language = GLSLANG_TARGET_SPV;
    input->target_language_version = GLSLANG_TARGET_SPV_1_0;

    /* Detect and upgrade GLSL version from source
     * GLSL 1.40 (OpenGL 3.1) shaders from virglrenderer need upgrading to 3.30
     * for glslang's SPIR-V compatibility with desktop OpenGL
     *
     * Default to 330 (minimum for SPIR-V) instead of 460 to be more permissive
     */
    int glsl_version = 330; /* Default to GLSL 3.30 - minimum for SPIR-V */
    int original_version = 330;
    const char *version_str = strstr(src, "#version");
    if (version_str) {
        int scanned_version;
        if (sscanf(version_str, "#version %d", &scanned_version) == 1) {
            original_version = scanned_version;
            glsl_version = scanned_version;
            /* Upgrade legacy GLSL versions to 330 minimum for SPIR-V */
            if (glsl_version < 330) {
                glsl_version = 330;
            }
        }
    }

    /* Set client_version to match GLSL version for SPIR-V targeting
     * This prevents "forced to be (450, core)" error when using GLSL 330 shaders
     * Must be set AFTER version detection above
     *
     * Note: glslang only exposes GLSLANG_TARGET_OPENGL_450, so we use that
     * as the SPIR-V target for all modern GLSL versions
     */
    if (glsl_version < 330) {
        /* Legacy GLSL - still target 450 for SPIR-V but shader will be upgraded */
        input->client_version = GLSLANG_TARGET_OPENGL_450;
    } else if (glsl_version == 330) {
        /* GLSL 3.30 shaders - target OpenGL 3.30 for SPIR-V */
        input->client_version = 330;  /* Use numeric value directly */
    } else {
        /* GLSL 4.00+ - target OpenGL 4.50 for SPIR-V */
        input->client_version = GLSLANG_TARGET_OPENGL_450;
    }

    /* Build a mutable source copy for compatibility rewrites. */
    static char *modified_src = NULL;
    static size_t modified_src_size = 0;
    size_t src_len = strlen(src);
    if (src_len + 2048 > modified_src_size) {
        modified_src_size = src_len + 2048;
        free(modified_src);
        modified_src = (char *)malloc(modified_src_size);
    }

    if (!modified_src) {
        fprintf(stderr, "[MGL] ERROR: Failed to allocate modified_src\n");
        input->code = src;
    } else {
        strcpy(modified_src, src);

        if (original_version < 330) {
            fprintf(stderr, "[MGL] Upgrading GLSL shader from version %d to %d\n",
                    original_version, glsl_version);

            /* Find and replace #version line */
            char *version_line = strstr(modified_src, "#version");
            if (!version_line) {
                fprintf(stderr, "[MGL] WARNING: #version not found in source\n");
            } else {
                char *newline = strchr(version_line, '\n');
                if (!newline) {
                    fprintf(stderr, "[MGL] WARNING: newline not found after #version\n");
                } else {
                    char version_buf[64];
                    snprintf(version_buf, sizeof(version_buf), "#version %d core", glsl_version);
                    size_t old_len = (size_t)(newline - version_line);
                    size_t new_len = strlen(version_buf);

                    if (new_len <= old_len) {
                        memset(version_line, ' ', old_len);
                        memcpy(version_line, version_buf, new_len);
                    } else {
                        size_t rest_of_src = strlen(newline);
                        memmove(version_line + new_len, newline, rest_of_src + 1);
                        memcpy(version_line, version_buf, new_len);
                    }
                }
            }
        }

        /* Inject explicit UBO bindings for desktop GLSL sources that omit them.
         * Newer glslang/SPIR-V paths may require these at parse time. */
        bool ubo_patched = mgl_patch_uniform_block_bindings(modified_src, modified_src_size);
        if (ubo_patched) {
            mgl_upgrade_version_for_bindings(modified_src);
            mgl_ensure_420pack_extension(modified_src, modified_src_size);
        }
        mgl_downgrade_derivative_control_intrinsics(modified_src);

        if (strstr(modified_src, "#version 420") != NULL && glsl_version < 420) {
            glsl_version = 420;
        }

        input->code = modified_src;
    }

    input->default_version = glsl_version;
    input->default_profile = GLSLANG_CORE_PROFILE;
    /* Use relaxed OpenGL-style validation at shader compile stage.
     * Program-level link/map_io will assign/validate resource interfaces.
     * This avoids forcing explicit layout(binding=...) in vanilla MC GLSL 330.
     */
    input->messages = GLSLANG_MSG_RELAXED_ERRORS_BIT;
    input->resource = glslang_default_resource();

    input->force_default_version_and_profile = 0;
}

Shader *newShader(GLMContext ctx, GLenum type, GLuint shader)
{
    Shader *ptr;
    char shader_type_name[128];

    ptr = (Shader *)malloc(sizeof(Shader));
    // CRITICAL SECURITY FIX: Check malloc result instead of using assert()
    if (!ptr) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate memory for shader\n");
        STATE(error) = GL_OUT_OF_MEMORY;
        return NULL;
    }

    bzero(ptr, sizeof(Shader));

    ptr->name = shader;
    ptr->type = type;
    ptr->glm_type = glShaderTypeToGLMType(type);

    snprintf(shader_type_name, sizeof(shader_type_name), "%s_%d", getShaderTypeStr(ptr->glm_type), shader);
    ptr->mtl_shader_type_name = strdup(shader_type_name);

    return ptr;
}

Shader *getShader(GLMContext ctx, GLenum type, GLuint shader)
{
    Shader *ptr;

    ptr = (Shader *)searchHashTable(&STATE(shader_table), shader);

    if (!ptr)
    {
        ptr = newShader(ctx, type, shader);

        insertHashElement(&STATE(shader_table), shader, ptr);
    }

    return ptr;
}

int isShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;

    ptr = (Shader *)searchHashTable(&STATE(shader_table), shader);

    if (ptr && !ptr->delete_status)
        return 1;

    return 0;
}

Shader *findShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;

    ptr = (Shader *)searchHashTable(&STATE(shader_table), shader);

    return ptr;
}

GLuint mglCreateShader(GLMContext ctx, GLenum type)
{
    GLuint shader;

    switch(type)
    {
        case GL_VERTEX_SHADER:
        case GL_FRAGMENT_SHADER:
        case GL_GEOMETRY_SHADER:
        case GL_COMPUTE_SHADER:
        case GL_TESS_CONTROL_SHADER:
        case GL_TESS_EVALUATION_SHADER:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    shader = getNewName(&STATE(shader_table));

    getShader(ctx, type, shader);

    return shader;
}

void mglFreeShader(GLMContext ctx, Shader *ptr)
{
    if (ptr->compiled_glsl_shader)
    {
        glslang_shader_delete(ptr->compiled_glsl_shader);
    }

    if (ptr->mtl_data.function)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.function);
    }
    if (ptr->mtl_data.library)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.library);
    }
    if (ptr->mtl_data.zero_to_one_function)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.zero_to_one_function);
    }
    if (ptr->mtl_data.zero_to_one_library)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.zero_to_one_library);
    }
    if (ptr->mtl_data.upper_left_function)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.upper_left_function);
    }
    if (ptr->mtl_data.upper_left_library)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.upper_left_library);
    }
    if (ptr->mtl_data.upper_left_zero_to_one_function)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.upper_left_zero_to_one_function);
    }
    if (ptr->mtl_data.upper_left_zero_to_one_library)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data.upper_left_zero_to_one_library);
    }

    free((void *)ptr->mtl_shader_type_name);
    free((void *)ptr->src);
    if (ptr->log) free(ptr->log);

    free(ptr);
}

void mglDeleteShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;

    /* OpenGL spec: A value of 0 for shader will be silently ignored. */
    if (shader == 0) {
        return;
    }

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    ptr->delete_status = GL_TRUE;

    if (ptr->refcount == 0)
    {
        deleteHashElement(&STATE(shader_table), shader);
        mglFreeShader(ctx, ptr);
    }
}

GLboolean mglIsShader(GLMContext ctx, GLuint shader)
{
    return isShader(ctx, shader);
}

void mglShaderSource(GLMContext ctx, GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length)
{
    size_t len;
    GLchar *src;
    Shader *ptr;

    ERROR_CHECK_RETURN(shader != 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    if (count>1)
    {
        // compute storage requirement
        len = 0;
        if (!length) {
            for(int i=0; i<count; i++)
            {
                len += strlen(string[i]);
            }
        }
        else {
            for(int i=0; i<count; i++)
            {
                len += length[i];
            }
        }   
        ERROR_CHECK_RETURN(len, GL_INVALID_VALUE);

        // allocate storage
        src = (GLchar *)malloc(len+1); // +1 for NULL
        ERROR_CHECK_RETURN(src, GL_OUT_OF_MEMORY);

        if (!length) {        
            // string[i] are null-terminated
            *src = 0;
            for(int i=0; i<count; ++i)
            {
                strlcat(src, string[i], len+1);
            }
            if (strlen(src) != (size_t)len) {
                fprintf(stderr,
                        "MGL WARNING: shader source length mismatch expected=%zu actual=%zu\n",
                        (size_t)len,
                        strlen(src));
            }
        } else {
            // CRITICAL SECURITY FIX: Prevent buffer overflow in shader source concatenation
            // string[i] may not be null-terminated - we must validate bounds carefully
            size_t cum_len = 0;
            for(int i=0; i<count; ++i)
            {
                // CRITICAL: Check if adding this string would exceed buffer bounds
                if (cum_len + length[i] > (size_t)len) {
                    // SECURITY: Truncate safely instead of overflowing buffer
                    fprintf(stderr, "MGL SECURITY ERROR: Shader source concatenation would overflow buffer, truncating safely\n");
                    // Copy only what fits
                    size_t safe_copy_len = ((size_t)len > cum_len) ? ((size_t)len - cum_len) : 0;
                    if (safe_copy_len > 0) {
                        strncpy(&src[cum_len], string[i], safe_copy_len);
                    }
                    cum_len = len; // Force termination at end
                    break;
                }

                // CRITICAL: Validate source pointer and length before copy
                if (!string[i]) {
                    fprintf(stderr, "MGL SECURITY ERROR: NULL string pointer in shader source concatenation\n");
                    continue; // Skip this string
                }

                strncpy(&src[cum_len], string[i], length[i]);
                cum_len += length[i];
            }
            // CRITICAL: Ensure null termination regardless of truncation
            src[cum_len < (size_t)len ? cum_len : (size_t)len] = '\0';
        }
    }
    else
    {
        ERROR_CHECK_RETURN(string, GL_INVALID_VALUE);

        src = strdup(*string);
        len = strlen(src);

        ERROR_CHECK_RETURN(len, GL_INVALID_VALUE);
    }

    ptr->src_len = len;
    ptr->src = src;
    ptr->dirty_bits |= DIRTY_SHADER;
}

void mglCompileShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;
    glslang_input_t glsl_input;
    glslang_shader_t *glsl_shader;
    int err;

    ERROR_CHECK_RETURN(shader != 0, GL_INVALID_VALUE);

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    initGLSLInput(ctx, ptr->type, ptr->src, &glsl_input);

    glsl_shader = glslang_shader_create(&glsl_input);
    if (glsl_shader == NULL)
    {
        // CRITICAL FIX: Handle shader creation failure gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Failed to create GLSL shader for type 0x%x\n", ptr->type);

        // Set error state for the shader - only set log message
        if (!ptr->log) {
            ptr->log = strdup("GLSL shader creation failed - insufficient memory or unsupported shader type");
        }
        return;
    }

    if (ptr->log)
    {
        free(ptr->log);
        ptr->log = NULL;
    }

    /* Use OpenGL semantics (not Vulkan rules) and auto-map bindings/locations
     * so Minecraft GLSL 330 shaders without explicit layout(binding=...) work.
     */
    /* IMPORTANT: do not auto-map bindings per-shader here.
     * Per-shader auto binding assignment can diverge between VS/FS and then fail
     * at program link with "Layout binding qualifier must match".
     * We resolve bindings/locations at program level via glslang_program_map_io().
     */
    int options = GLSLANG_SHADER_AUTO_MAP_LOCATIONS;

    /* Detect if this is a legacy GLSL shader that needs location auto-assignment */
    int shader_version = 330; /* Default */
    const char *version_str = strstr(ptr->src, "#version");
    if (version_str) {
        sscanf(version_str, "#version %d", &shader_version);
    }

    if (shader_version < 330) {
        fprintf(stderr, "[MGL] Enabling compatibility mode for legacy GLSL %d shader\n", shader_version);
    }
    glslang_shader_set_options(glsl_shader, options);

    err = glslang_shader_preprocess(glsl_shader, &glsl_input);
    if (!err)
    {
        // PROPER FIX: Enhanced error logging with proper formatting
        const char *preprocessed = glslang_shader_get_preprocessed_code(glsl_shader);
        const char *info_log = glslang_shader_get_info_log(glsl_shader);
        const char *debug_log = glslang_shader_get_info_debug_log(glsl_shader);

        fprintf(stderr, "MGL SHADER ERROR: glslang_shader_preprocess failed with error: %d\n", err);
        fprintf(stderr, "MGL SHADER ERROR: Shader type: %s\n", getShaderTypeStr(ptr->glm_type));
        fprintf(stderr, "MGL SHADER ERROR: Preprocessed code:\n%s\n", preprocessed ? preprocessed : "(null)");
        fprintf(stderr, "MGL SHADER ERROR: Info log:\n%s\n", info_log ? info_log : "(null)");
        fprintf(stderr, "MGL SHADER ERROR: Debug log:\n%s\n", debug_log ? debug_log : "(null)");

        size_t len;

        len = 1024;
        len += strlen(glslang_shader_get_preprocessed_code(glsl_shader));
        len += strlen(glslang_shader_get_info_log(glsl_shader));
        len += strlen(glslang_shader_get_info_debug_log(glsl_shader));

        ptr->log = (char *)malloc(len);

        ptr->log[0] = 0;

        snprintf(ptr->log, len,
                "glslang_shader_preprocess failed err: %d\n"
                "glslang_shader_get_preprocessed_code:\n%s\n"
                "glslang_shader_get_info_log:%s\n"
                "glslang_shader_get_info_debug_log:\n%s\n",
                err,
                glslang_shader_get_preprocessed_code(glsl_shader),
                glslang_shader_get_info_log(glsl_shader),
                glslang_shader_get_info_debug_log(glsl_shader));

        return;
    }

    err = glslang_shader_parse(glsl_shader, &glsl_input);
    if (!err)
    {
        // PROPER FIX: Enhanced parse error logging
        const char *preprocessed = glslang_shader_get_preprocessed_code(glsl_shader);
        const char *info_log = glslang_shader_get_info_log(glsl_shader);
        const char *debug_log = glslang_shader_get_info_debug_log(glsl_shader);

        fprintf(stderr, "MGL SHADER ERROR: glslang_shader_parse failed with error: %d\n", err);
        fprintf(stderr, "MGL SHADER ERROR: Shader type: %s\n", getShaderTypeStr(ptr->glm_type));
        fprintf(stderr, "MGL SHADER ERROR: Preprocessed code:\n%s\n", preprocessed ? preprocessed : "(null)");
        fprintf(stderr, "MGL SHADER ERROR: Info log:\n%s\n", info_log ? info_log : "(null)");
        fprintf(stderr, "MGL SHADER ERROR: Debug log:\n%s\n", debug_log ? debug_log : "(null)");

        size_t len;

        len = 1024;
        len += strlen(glslang_shader_get_preprocessed_code(glsl_shader));
        len += strlen(glslang_shader_get_info_log(glsl_shader));
        len += strlen(glslang_shader_get_info_debug_log(glsl_shader));

        ptr->log = (char *)malloc(len);

        ptr->log[0] = 0;

        snprintf(ptr->log, len,
                "glslang_shader_preprocess failed err: %d\n"
                "glslang_shader_get_preprocessed_code:\n%s\n"
                "glslang_shader_get_info_log:%s\n"
                "glslang_shader_get_info_debug_log:\n%s\n",
                err,
                glslang_shader_get_preprocessed_code(glsl_shader),
                glslang_shader_get_info_log(glsl_shader),
                glslang_shader_get_info_debug_log(glsl_shader));

        return;
    }

    if (ptr->compiled_glsl_shader) {
        ptr->dirty_bits |= DIRTY_SHADER;
    }

    ptr->compiled_glsl_shader = glsl_shader;
}

void mglGetShaderiv(GLMContext ctx, GLuint shader, GLenum pname, GLint *params)
{
    Shader *ptr;

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    switch(pname)
    {
        case GL_SHADER_TYPE:
            switch(ptr->glm_type)
            {
                case _VERTEX_SHADER: *params = GL_VERTEX_SHADER; break;
                case _FRAGMENT_SHADER: *params = GL_FRAGMENT_SHADER; break;
                case _GEOMETRY_SHADER: *params = GL_GEOMETRY_SHADER; break;
                case _COMPUTE_SHADER: *params = GL_COMPUTE_SHADER; break;
                case _TESS_CONTROL_SHADER: *params = GL_TESS_CONTROL_SHADER; break;
                case _TESS_EVALUATION_SHADER: *params = GL_TESS_EVALUATION_SHADER; break;
                default:
                    // CRITICAL FIX: Handle unknown shader types gracefully instead of crashing
                    fprintf(stderr, "MGL ERROR: Unknown internal shader type %d, defaulting to vertex\n", ptr->glm_type);
                    *params = GL_VERTEX_SHADER;
            }
            break;

        case GL_DELETE_STATUS:
            *params = GL_FALSE;
            break;

        case GL_COMPILE_STATUS:
            if (ptr->log)
            {
                *params = GL_FALSE;
            }
            else
            {
                *params = GL_TRUE;
            }
            break;

        case GL_INFO_LOG_LENGTH:
            *params = ptr->log ? (GLint)strlen(ptr->log) : 0;
            break;

        case GL_SHADER_SOURCE_LENGTH:
            *params = (GLint)ptr->src_len;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglGetShaderInfoLog(GLMContext ctx, GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    Shader *ptr;

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    if (ptr->log)
    {
        if (length)
        {
            *length = (GLsizei)strlen(ptr->log);
        }

        if (infoLog)
        {
            if (bufSize >= strlen(ptr->log))
            {
                memcpy(infoLog, ptr->log, strlen(ptr->log));
            }
        }
    }
}

void mglGetShaderSource(GLMContext ctx, GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source)
{
    Shader *ptr;

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    if (ptr->src)
    {
        if (length)
        {
            *length = (GLsizei)ptr->src_len;
        }

        if (source)
        {
            if (bufSize >= (GLsizei)ptr->src_len)
            {
                memcpy(source, ptr->src, ptr->src_len);
            }
        }
    }

}
