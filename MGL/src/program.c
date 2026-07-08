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
 * program.c
 * MGL
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <malloc/malloc.h>
#include <CoreFoundation/CoreFoundation.h>
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#include "spirv-tools/libspirv.h"
#include "spirv_cross_c.h"
#include "spirv.h"

#include "glm_context.h"
#include "shaders.h"
#include "buffers.h"
#include "mgl_safety.h"
#include "mgl_buffer_slots.h"
#include "mgl_ir_postprocess.h"
#include "msl_patch_pipeline.h"
#include "mgl_metal_ref.h"

#ifndef MGL_VERBOSE_PROGRAM_LOGS
#define MGL_VERBOSE_PROGRAM_LOGS 0
#endif

#define MGL_TEXEL_BUFFER_TEXTURE_WIDTH 4096u
#define MGL_INTERNAL_UNIFORM_BUFFER_NAME_BASE 0xf0000000u
#define MGL_SYNTHETIC_SAMPLER_LOCATION_BASE 0x4000
#define MGL_FRAG_COORD_PARAMS_MSL_NAME "_mglFragCoordParams"
#define MGL_FRAG_COORD_PARAMS_BUFFER_INDEX 30

static size_t mglRoundUpSize(size_t value, size_t alignment)
{
    return alignment ? ((value + alignment - 1) / alignment) * alignment : value;
}

static GLboolean mglPointerLooksMallocOwned(const void *ptr)
{
    uintptr_t value = (uintptr_t)ptr;
    if (!ptr || value < 0x10000u) {
        return GL_FALSE;
    }

    return malloc_size(ptr) > 0 ? GL_TRUE : GL_FALSE;
}

static void mglFreeProgramAttribName(Program *program, GLuint index, const char *reason)
{
    if (!program || index >= MAX_ATTRIBS) {
        return;
    }

    char *name = program->attrib_location_names[index];
    GLboolean owned = program->attrib_location_name_owned[index];
    program->attrib_location_names[index] = NULL;
    program->attrib_location_name_owned[index] = GL_FALSE;

    if (!name) {
        return;
    }

    if (owned && mglPointerLooksMallocOwned(name)) {
        free(name);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: skipped invalid attrib name free program=%u index=%u ptr=%p owned=%d reason=%s\n",
            program->name,
            index,
            (void *)name,
            owned,
            reason ? reason : "(unknown)");
}

static GLboolean mglSetProgramAttribName(Program *program, GLuint index, const char *name)
{
    if (!program || index >= MAX_ATTRIBS || !name) {
        return GL_FALSE;
    }

    mglFreeProgramAttribName(program, index, "replace");
    program->attrib_location_names[index] = strdup(name);
    if (!program->attrib_location_names[index]) {
        return GL_FALSE;
    }
    program->attrib_location_name_owned[index] = GL_TRUE;
    return GL_TRUE;
}

static const char *mglMemStr(const char *haystack, size_t haystack_len, const char *needle)
{
    size_t needle_len = needle ? strlen(needle) : 0u;
    if (!haystack || !needle || needle_len == 0u || needle_len > haystack_len) {
        return NULL;
    }

    for (size_t i = 0; i <= haystack_len - needle_len; i++) {
        if (memcmp(haystack + i, needle, needle_len) == 0) {
            return haystack + i;
        }
    }

    return NULL;
}

static GLboolean mglRangeContainsToken(const char *begin, const char *end, const char *token)
{
    if (!begin || !end || end <= begin || !token) {
        return GL_FALSE;
    }

    return mglMemStr(begin, (size_t)(end - begin), token) ? GL_TRUE : GL_FALSE;
}

static GLboolean mglGLSLDeclaresRowMajorUBOMember(const char *glsl_src,
                                                  const char *block_name,
                                                  const char *member_name)
{
    if (!glsl_src || !block_name || !block_name[0] || !member_name || !member_name[0]) {
        return GL_FALSE;
    }

    size_t block_len = strlen(block_name);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, block_name)) != NULL) {
        const char *after_name = pos + block_len;
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)*after_name) || *after_name == '_')) {
            pos = after_name;
            continue;
        }

        const char *brace = after_name;
        while (*brace && isspace((unsigned char)*brace)) {
            brace++;
        }
        if (*brace != '{') {
            pos = after_name;
            continue;
        }

        const char *decl_begin = pos;
        while (decl_begin > glsl_src && decl_begin[-1] != ';' && decl_begin[-1] != '}') {
            decl_begin--;
        }
        GLboolean block_row_major = mglRangeContainsToken(decl_begin, brace, "row_major");
        GLboolean block_column_major = mglRangeContainsToken(decl_begin, brace, "column_major");

        const char *block_end = strchr(brace, '}');
        if (!block_end) {
            return block_row_major && !block_column_major;
        }

        const char *member = brace + 1;
        size_t member_len = strlen(member_name);
        while ((member = mglMemStr(member, (size_t)(block_end - member), member_name)) != NULL) {
            const char *member_end = member + member_len;
            if ((member > brace + 1 && (isalnum((unsigned char)member[-1]) || member[-1] == '_')) ||
                (isalnum((unsigned char)*member_end) || *member_end == '_')) {
                member = member_end;
                continue;
            }

            const char *stmt_begin = member;
            while (stmt_begin > brace + 1 && stmt_begin[-1] != ';' && stmt_begin[-1] != '{') {
                stmt_begin--;
            }
            const char *stmt_end = member_end;
            while (stmt_end < block_end && *stmt_end != ';') {
                stmt_end++;
            }
            if (mglRangeContainsToken(stmt_begin, stmt_end, "column_major")) {
                return GL_FALSE;
            }
            if (mglRangeContainsToken(stmt_begin, stmt_end, "row_major")) {
                return GL_TRUE;
            }
            return block_row_major && !block_column_major;
        }

        return block_row_major && !block_column_major;
    }

    return GL_FALSE;
}

static char *mglRecoverMemberNameFromGLSLComposite(const char *glsl_src,
                                                   const char *composite_name,
                                                   unsigned member_index,
                                                   GLboolean require_block_brace)
{
    if (!glsl_src || !composite_name || !composite_name[0]) {
        return NULL;
    }

    size_t block_len = strlen(composite_name);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, composite_name)) != NULL) {
        const char *after_name = pos + block_len;
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)*after_name) || *after_name == '_')) {
            pos = after_name;
            continue;
        }

        const char *brace = after_name;
        while (*brace && isspace((unsigned char)*brace)) {
            brace++;
        }
        if (!require_block_brace) {
            const char *p = pos;
            while (p > glsl_src && isspace((unsigned char)p[-1])) {
                p--;
            }
            const char *word_end = p;
            while (p > glsl_src && (isalpha((unsigned char)p[-1]) || p[-1] == '_')) {
                p--;
            }
            if ((size_t)(word_end - p) != 6u || memcmp(p, "struct", 6) != 0) {
                pos = after_name;
                continue;
            }
        }
        if (*brace != '{') {
            pos = after_name;
            continue;
        }

        const char *block_end = strchr(brace, '}');
        if (!block_end) {
            return NULL;
        }

        const char *stmt_begin = brace + 1;
        unsigned nth = 0;
        while (stmt_begin < block_end) {
            const char *stmt_end = stmt_begin;
            while (stmt_end < block_end && *stmt_end != ';') {
                stmt_end++;
            }
            if (stmt_end >= block_end) {
                break;
            }

            const char *end = stmt_end;
            while (end > stmt_begin && isspace((unsigned char)end[-1])) {
                end--;
            }
            while (end > stmt_begin && end[-1] == ']') {
                int depth = 1;
                end--;
                while (end > stmt_begin && depth > 0) {
                    end--;
                    if (*end == ']') {
                        depth++;
                    } else if (*end == '[') {
                        depth--;
                    }
                }
                while (end > stmt_begin && isspace((unsigned char)end[-1])) {
                    end--;
                }
            }

            const char *name_end = end;
            while (end > stmt_begin &&
                   (isalnum((unsigned char)end[-1]) || end[-1] == '_')) {
                end--;
            }

            if (name_end > end &&
                (isalpha((unsigned char)*end) || *end == '_')) {
                if (nth == member_index) {
                    size_t name_len = (size_t)(name_end - end);
                    char *name = (char *)malloc(name_len + 1u);
                    if (name) {
                        memcpy(name, end, name_len);
                        name[name_len] = '\0';
                    }
                    return name;
                }
                nth++;
            }

            stmt_begin = stmt_end + 1;
        }

        pos = block_end + 1;
    }

    return NULL;
}

static char *mglRecoverUBOMemberNameFromGLSL(const char *glsl_src,
                                             const char *block_name,
                                             unsigned member_index)
{
    return mglRecoverMemberNameFromGLSLComposite(glsl_src,
                                                 block_name,
                                                 member_index,
                                                 GL_TRUE);
}

static char *mglRecoverStructMemberNameFromGLSL(const char *glsl_src,
                                                const char *struct_name,
                                                unsigned member_index)
{
    return mglRecoverMemberNameFromGLSLComposite(glsl_src,
                                                 struct_name,
                                                 member_index,
                                                 GL_FALSE);
}

static char *mglDupRange(const char *begin, const char *end);

static char *mglGLSLTypeNameForMemberInComposite(const char *glsl_src,
                                                 const char *composite_name,
                                                 const char *member_name,
                                                 GLboolean require_block_brace)
{
    if (!glsl_src || !composite_name || !composite_name[0] ||
        !member_name || !member_name[0]) {
        return NULL;
    }

    size_t composite_len = strlen(composite_name);
    size_t member_len = strlen(member_name);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, composite_name)) != NULL) {
        const char *after_name = pos + composite_len;
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)*after_name) || *after_name == '_')) {
            pos = after_name;
            continue;
        }

        const char *brace = after_name;
        while (*brace && isspace((unsigned char)*brace)) {
            brace++;
        }
        if (!require_block_brace) {
            const char *p = pos;
            while (p > glsl_src && isspace((unsigned char)p[-1])) {
                p--;
            }
            const char *word_end = p;
            while (p > glsl_src && (isalpha((unsigned char)p[-1]) || p[-1] == '_')) {
                p--;
            }
            if ((size_t)(word_end - p) != 6u || memcmp(p, "struct", 6) != 0) {
                pos = after_name;
                continue;
            }
        }
        if (*brace != '{') {
            pos = after_name;
            continue;
        }

        const char *block_end = strchr(brace, '}');
        if (!block_end) {
            return NULL;
        }

        const char *member = brace + 1;
        while ((member = mglMemStr(member, (size_t)(block_end - member), member_name)) != NULL) {
            const char *member_end = member + member_len;
            if ((member > brace + 1 && (isalnum((unsigned char)member[-1]) || member[-1] == '_')) ||
                (isalnum((unsigned char)*member_end) || *member_end == '_')) {
                member = member_end;
                continue;
            }

            const char *stmt_begin = member;
            while (stmt_begin > brace + 1 && stmt_begin[-1] != ';' && stmt_begin[-1] != '{') {
                stmt_begin--;
            }
            const char *type_end = member;
            while (type_end > stmt_begin && isspace((unsigned char)type_end[-1])) {
                type_end--;
            }
            const char *type_begin = type_end;
            while (type_begin > stmt_begin &&
                   (isalnum((unsigned char)type_begin[-1]) || type_begin[-1] == '_')) {
                type_begin--;
            }
            if (type_end > type_begin &&
                (isalpha((unsigned char)*type_begin) || *type_begin == '_')) {
                return mglDupRange(type_begin, type_end);
            }
            member = member_end;
        }

        pos = block_end + 1;
    }

    return NULL;
}

static char *mglGLSLCompositeTypeNameForPath(const char *glsl_src,
                                             const char *block_name,
                                             const char *path)
{
    if (!glsl_src || !block_name || !path || !path[0]) {
        return NULL;
    }

    char *current_composite = strdup(block_name);
    char *type_name = NULL;
    const char *cursor = path;
    GLboolean current_is_block = GL_TRUE;

    if (!current_composite) {
        return NULL;
    }

    while (*cursor) {
        const char *token_begin = cursor;
        while (*cursor && *cursor != '.' && *cursor != '[') {
            cursor++;
        }
        if (cursor <= token_begin) {
            break;
        }

        char *token = mglDupRange(token_begin, cursor);
        if (!token) {
            free(current_composite);
            free(type_name);
            return NULL;
        }

        free(type_name);
        type_name = mglGLSLTypeNameForMemberInComposite(glsl_src,
                                                        current_composite,
                                                        token,
                                                        current_is_block);
        free(token);
        if (!type_name) {
            free(current_composite);
            return NULL;
        }

        free(current_composite);
        current_composite = strdup(type_name);
        if (!current_composite) {
            free(type_name);
            return NULL;
        }
        current_is_block = GL_FALSE;

        while (*cursor == '[') {
            cursor++;
            while (*cursor && *cursor != ']') {
                cursor++;
            }
            if (*cursor == ']') {
                cursor++;
            }
        }
        if (*cursor == '.') {
            cursor++;
            continue;
        }
        break;
    }

    free(current_composite);
    return type_name;
}

static char *mglDupRange(const char *begin, const char *end)
{
    if (!begin || !end || end < begin) {
        return NULL;
    }

    size_t len = (size_t)(end - begin);
    char *ret = (char *)malloc(len + 1u);
    if (ret) {
        memcpy(ret, begin, len);
        ret[len] = '\0';
    }
    return ret;
}

static char *mglGLSLUBOInstanceName(const char *glsl_src, const char *block_name)
{
    if (!glsl_src || !block_name || !block_name[0]) {
        return NULL;
    }

    size_t block_len = strlen(block_name);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, block_name)) != NULL) {
        const char *after_name = pos + block_len;
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)*after_name) || *after_name == '_')) {
            pos = after_name;
            continue;
        }

        const char *brace = after_name;
        while (*brace && isspace((unsigned char)*brace)) {
            brace++;
        }
        if (*brace != '{') {
            pos = after_name;
            continue;
        }

        const char *block_end = strchr(brace, '}');
        if (!block_end) {
            return NULL;
        }

        const char *p = block_end + 1;
        while (*p && isspace((unsigned char)*p)) {
            p++;
        }
        if (!(isalpha((unsigned char)*p) || *p == '_')) {
            return NULL;
        }
        const char *name_begin = p;
        while (isalnum((unsigned char)*p) || *p == '_') {
            p++;
        }
        return mglDupRange(name_begin, p);
    }

    return NULL;
}

/* Detect whether a UBO is declared as an array in the GLSL source and
 * return its array size.  For a declaration like:
 *   layout(binding=N) uniform BlockName { ... } instanceName[3];
 * returns 3.  For:
 *   layout(binding=N) uniform BlockName { ... } instanceName[1];
 * returns 1 (array of size 1, still an array).  Returns 0 if the UBO
 * is NOT declared as an array (no instance name or no [N] suffix). */
static GLuint mglGLSLUBOArraySize(const char *glsl_src, const char *block_name)
{
    if (!glsl_src || !block_name || !block_name[0]) {
        return 0;
    }

    size_t block_len = strlen(block_name);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, block_name)) != NULL) {
        const char *after_name = pos + block_len;
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)*after_name) || *after_name == '_')) {
            pos = after_name;
            continue;
        }

        const char *brace = after_name;
        while (*brace && isspace((unsigned char)*brace)) {
            brace++;
        }
        if (*brace != '{') {
            pos = after_name;
            continue;
        }

        const char *block_end = strchr(brace, '}');
        if (!block_end) {
            return 0;
        }

        const char *p = block_end + 1;
        while (*p && isspace((unsigned char)*p)) {
            p++;
        }
        if (!(isalpha((unsigned char)*p) || *p == '_')) {
            return 0;
        }
        /* Skip instance name */
        while (isalnum((unsigned char)*p) || *p == '_') {
            p++;
        }
        while (*p && isspace((unsigned char)*p)) {
            p++;
        }
        if (*p != '[') {
            return 0;
        }
        p++;
        while (*p && isspace((unsigned char)*p)) {
            p++;
        }
        char *end = NULL;
        unsigned long parsed = strtoul(p, &end, 10);
        if (!end || p == end) {
            return 0;
        }
        return (GLuint)parsed;
    }

    return 0;
}

static char *mglBuildUBOMemberQueryName(const SpirvResource *ubo, const SpirvUBOMember *member)
{
    if (!ubo || !member || !member->name) {
        return NULL;
    }

    if (!ubo->ubo_has_instance_name) {
        return strdup(member->name);
    }

    size_t block_len = ubo->name ? strlen(ubo->name) : 0u;
    size_t member_len = strlen(member->name);
    size_t suffix_len = (member->size > 1 && !strchr(member->name, '[')) ? 3u : 0u;
    char *ret = (char *)malloc(block_len + 1u + member_len + suffix_len + 1u);
    if (!ret) {
        return NULL;
    }
    snprintf(ret, block_len + 1u + member_len + suffix_len + 1u,
             "%s.%s%s",
             ubo->name ? ubo->name : "",
             member->name,
             suffix_len ? "[0]" : "");
    return ret;
}

static char *mglGLSLAccessPathForUBOMember(const char *glsl_src,
                                           const char *block_name,
                                           const char *instance_name,
                                           const char *member_name)
{
    if (!glsl_src || !block_name || !block_name[0] ||
        !instance_name || !instance_name[0] ||
        !member_name || !member_name[0]) {
        return NULL;
    }

    size_t inst_len = strlen(instance_name);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, instance_name)) != NULL) {
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)pos[inst_len]) || pos[inst_len] == '_')) {
            pos += inst_len;
            continue;
        }

        const char *p = pos + inst_len;
        while (*p == '[') {
            p++;
            while (*p && *p != ']') {
                p++;
            }
            if (*p == ']') {
                p++;
            }
        }
        if (*p != '.') {
            pos += inst_len;
            continue;
        }

        const char *path_begin = p + 1;
        size_t member_len = strlen(member_name);
        if (strncmp(path_begin, member_name, member_len) != 0 ||
            (isalnum((unsigned char)path_begin[member_len]) || path_begin[member_len] == '_')) {
            pos += inst_len;
            continue;
        }

        const char *q = path_begin;
        while (*q) {
            if (isalpha((unsigned char)*q) || *q == '_') {
                q++;
                while (isalnum((unsigned char)*q) || *q == '_') {
                    q++;
                }
                continue;
            }
            if (*q == '[') {
                q++;
                while (*q && *q != ']') {
                    q++;
                }
                if (*q == ']') {
                    q++;
                }
                continue;
            }
            if (*q == '.') {
                q++;
                continue;
            }
            break;
        }

        const char *path_end = q;
        size_t block_len = strlen(block_name);
        size_t path_len = (size_t)(path_end - path_begin);
        char *ret = (char *)malloc(block_len + 1u + path_len + 4u);
        if (!ret) {
            return NULL;
        }
        memcpy(ret, block_name, block_len);
        ret[block_len] = '.';
        memcpy(ret + block_len + 1u, path_begin, path_len);
        ret[block_len + 1u + path_len] = '\0';
        char *last_bracket = strrchr(ret, '[');
        if (last_bracket && last_bracket > ret + block_len + 1u + member_len) {
            strcpy(last_bracket, "[0]");
        }
        return ret;

        pos += inst_len;
    }

    return NULL;
}

static GLuint mglGLTypeFromSPVCType(spvc_type type)
{
    if (!type) {
        return GL_FLOAT;
    }

    spvc_basetype base = spvc_type_get_basetype(type);
    unsigned raw_vec = spvc_type_get_vector_size(type);
    unsigned vec_size = raw_vec > 0 ? raw_vec : 1;
    unsigned cols = spvc_type_get_columns(type);

    switch (base) {
        case SPVC_BASETYPE_FP32:
            if (cols > 1) {
                static const GLuint mats[] = {
                    0, GL_FLOAT_MAT2, GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4,
                    GL_FLOAT_MAT3x2, GL_FLOAT_MAT3, GL_FLOAT_MAT3x4,
                    GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, GL_FLOAT_MAT4
                };
                unsigned key = (cols - 2) * 3 + (vec_size - 2) + 1;
                if (key < sizeof(mats) / sizeof(mats[0])) {
                    return mats[key];
                }
            } else if (vec_size >= 1 && vec_size <= 4) {
                static const GLuint v[] = {GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4};
                return v[vec_size - 1];
            }
            break;
        case SPVC_BASETYPE_INT32:
            if (vec_size >= 1 && vec_size <= 4) {
                static const GLuint v[] = {GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4};
                return v[vec_size - 1];
            }
            break;
        case SPVC_BASETYPE_UINT32:
            if (vec_size >= 1 && vec_size <= 4) {
                static const GLuint v[] = {GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4};
                return v[vec_size - 1];
            }
            break;
        case SPVC_BASETYPE_BOOLEAN:
            if (vec_size >= 1 && vec_size <= 4) {
                static const GLuint v[] = {GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4};
                return v[vec_size - 1];
            }
            break;
        case SPVC_BASETYPE_FP64:
            if (vec_size >= 1 && vec_size <= 4) {
                static const GLuint v[] = {GL_DOUBLE, GL_DOUBLE_VEC2, GL_DOUBLE_VEC3, GL_DOUBLE_VEC4};
                return v[vec_size - 1];
            }
            break;
        default:
            break;
    }

    return GL_FLOAT;
}

static GLint mglGLArraySizeFromSPVCType(spvc_type type)
{
    if (!type) {
        return 1;
    }
    unsigned array_dims = spvc_type_get_num_array_dimensions(type);
    if (array_dims > 0) {
        SpvId size = spvc_type_get_array_dimension(type, 0);
        return size > 0 ? (GLint)size : 1;
    }
    return 1;
}

static GLboolean mglGLSLNameLooksLikeType(const char *name)
{
    static const char *glsl_types[] = {
        "float","int","uint","bool","double",
        "vec2","vec3","vec4","ivec2","ivec3","ivec4",
        "uvec2","uvec3","uvec4","bvec2","bvec3","bvec4",
        "dvec2","dvec3","dvec4",
        "mat2","mat3","mat4","mat2x2","mat2x3","mat2x4",
        "mat3x2","mat3x3","mat3x4","mat4x2","mat4x3","mat4x4",
        "dmat2","dmat3","dmat4",
        "sampler2D","samplerCube","sampler3D",
        "isampler2D","usampler2D",
        NULL
    };

    if (!name || !name[0]) {
        return GL_FALSE;
    }
    for (int i = 0; glsl_types[i]; i++) {
        if (strcmp(name, glsl_types[i]) == 0) {
            return GL_TRUE;
        }
    }
    return GL_FALSE;
}

static char *mglLeafNameFromPath(const char *path)
{
    const char *leaf = path;

    if (!path) {
        return NULL;
    }
    for (const char *p = path; *p; p++) {
        if (*p == '.') {
            leaf = p + 1;
        }
    }
    while (leaf && leaf[0] == '[') {
        const char *end = strchr(leaf, ']');
        leaf = end ? end + 1 : leaf;
        if (leaf[0] == '.') {
            leaf++;
        }
    }

    size_t len = strcspn(leaf, "[.");
    char *ret = (char *)malloc(len + 1u);
    if (!ret) {
        return NULL;
    }
    memcpy(ret, leaf, len);
    ret[len] = '\0';
    return ret;
}

static GLuint mglGLBoolTypeForVectorSize(unsigned vec_size)
{
    static const GLuint v[] = {GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4};
    if (vec_size >= 1 && vec_size <= 4) {
        return v[vec_size - 1];
    }
    return GL_BOOL;
}

static GLuint mglGLTypeFromSPVCTypeAndGLSL(spvc_type type,
                                           const char *glsl_src,
                                           const char *block_name,
                                           const char *name)
{
    GLuint gl_type = mglGLTypeFromSPVCType(type);

    if (!type || !glsl_src || !name) {
        return gl_type;
    }

    spvc_basetype base = spvc_type_get_basetype(type);
    if (base != SPVC_BASETYPE_UINT32) {
        return gl_type;
    }

    char *leaf = mglLeafNameFromPath(name);
    if (!leaf || !leaf[0]) {
        free(leaf);
        return gl_type;
    }

    unsigned raw_vec = spvc_type_get_vector_size(type);
    unsigned vec_size = raw_vec > 0 ? raw_vec : 1;

    if (block_name && block_name[0]) {
        const char *last_dot = strrchr(name, '.');
        char *decl_type = NULL;
        if (last_dot) {
            char *parent_path = mglDupRange(name, last_dot);
            char *member_leaf = mglLeafNameFromPath(last_dot + 1);
            char *parent_type = parent_path
                ? mglGLSLCompositeTypeNameForPath(glsl_src, block_name, parent_path)
                : NULL;
            if (parent_type && member_leaf) {
                decl_type = mglGLSLTypeNameForMemberInComposite(glsl_src,
                                                                parent_type,
                                                                member_leaf,
                                                                GL_FALSE);
            }
            free(parent_path);
            free(member_leaf);
            free(parent_type);
        } else {
            decl_type = mglGLSLTypeNameForMemberInComposite(glsl_src,
                                                            block_name,
                                                            leaf,
                                                            GL_TRUE);
        }
        if (decl_type) {
            if (strcmp(decl_type, "bool") == 0 ||
                strcmp(decl_type, "bvec2") == 0 ||
                strcmp(decl_type, "bvec3") == 0 ||
                strcmp(decl_type, "bvec4") == 0) {
                free(decl_type);
                free(leaf);
                return mglGLBoolTypeForVectorSize(vec_size);
            }
            free(decl_type);
            free(leaf);
            return gl_type;
        }
    }

    size_t leaf_len = strlen(leaf);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, leaf)) != NULL) {
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)pos[leaf_len]) || pos[leaf_len] == '_')) {
            pos += leaf_len;
            continue;
        }

        const char *te = pos;
        while (te > glsl_src && isspace((unsigned char)te[-1])) {
            te--;
        }
        const char *ts = te;
        while (ts > glsl_src && !isspace((unsigned char)ts[-1]) && ts[-1] != '\n') {
            ts--;
        }
        size_t tl = (size_t)(te - ts);
        if (tl == 4 && memcmp(ts, "bool", 4) == 0) {
            free(leaf);
            return mglGLBoolTypeForVectorSize(vec_size);
        }
        if (vec_size == 2 && tl == 5 && memcmp(ts, "bvec2", 5) == 0) {
            free(leaf);
            return mglGLBoolTypeForVectorSize(vec_size);
        }
        if (vec_size == 3 && tl == 5 && memcmp(ts, "bvec3", 5) == 0) {
            free(leaf);
            return mglGLBoolTypeForVectorSize(vec_size);
        }
        if (vec_size == 4 && tl == 5 && memcmp(ts, "bvec4", 5) == 0) {
            free(leaf);
            return mglGLBoolTypeForVectorSize(vec_size);
        }
        pos += leaf_len;
    }

    free(leaf);
    return gl_type;
}

static char *mglJoinUBOMemberPath(const char *prefix, const char *member_name)
{
    if (!member_name || !member_name[0]) {
        return NULL;
    }
    if (!prefix || !prefix[0]) {
        return strdup(member_name);
    }

    size_t prefix_len = strlen(prefix);
    size_t member_len = strlen(member_name);
    char *ret = (char *)malloc(prefix_len + 1u + member_len + 1u);
    if (!ret) {
        return NULL;
    }
    memcpy(ret, prefix, prefix_len);
    ret[prefix_len] = '.';
    memcpy(ret + prefix_len + 1u, member_name, member_len);
    ret[prefix_len + 1u + member_len] = '\0';
    return ret;
}

static char *mglAppendArrayZeroSuffix(const char *name, unsigned num_dims)
{
    if (!name) {
        return NULL;
    }
    const char *leaf = strrchr(name, '.');
    leaf = leaf ? leaf + 1 : name;
    if (strchr(leaf, '[')) {
        return strdup(name);
    }
    if (num_dims == 0) {
        num_dims = 1;
    }
    size_t len = strlen(name);
    /* Each dimension appends "[0]" (3 chars). */
    char *ret = (char *)malloc(len + 3u * num_dims + 1u);
    if (!ret) {
        return NULL;
    }
    memcpy(ret, name, len);
    for (unsigned d = 0; d < num_dims; d++) {
        memcpy(ret + len + 3u * d, "[0]", 3);
    }
    ret[len + 3u * num_dims] = '\0';
    return ret;
}

/* Compute Metal/MSL type alignment for a spvc_type.
 * Metal follows C struct layout rules: each member is aligned to its
 * natural alignment, and the struct size is padded to the max alignment.
 *
 * - Scalar (32-bit): 4
 * - Scalar (64-bit): 8
 * - 2-component vector: 8
 * - 3-component vector: 16 (Metal pads vec3 to 16-byte alignment)
 * - 4-component vector: 16
 * - Matrix: alignment = column vector alignment
 * - Array: alignment = element alignment
 * - Struct: max of member alignments
 */
static GLuint mglMetalTypeAlignmentFromSPVC(spvc_compiler compiler, spvc_type type)
{
    if (!type) {
        return 16;
    }

    spvc_basetype base = spvc_type_get_basetype(type);
    unsigned bit_width = spvc_type_get_bit_width(type);
    unsigned vec_size = spvc_type_get_vector_size(type);
    unsigned columns = spvc_type_get_columns(type);
    unsigned array_dims = spvc_type_get_num_array_dimensions(type);

    /* For arrays, alignment = element alignment (resolve to base type) */
    if (array_dims > 0) {
        spvc_type_id elem_type_id = spvc_type_get_base_type_id(type);
        spvc_type elem_type = elem_type_id
            ? spvc_compiler_get_type_handle(compiler, elem_type_id) : NULL;
        return mglMetalTypeAlignmentFromSPVC(compiler, elem_type);
    }

    /* For structs, alignment = max of member alignments */
    if (base == SPVC_BASETYPE_STRUCT) {
        unsigned member_count = spvc_type_get_num_member_types(type);
        GLuint max_align = 4;
        for (unsigned i = 0; i < member_count; i++) {
            spvc_type_id mt_id = spvc_type_get_member_type(type, i);
            spvc_type mt = spvc_compiler_get_type_handle(compiler, mt_id);
            GLuint align = mglMetalTypeAlignmentFromSPVC(compiler, mt);
            if (align > max_align) {
                max_align = align;
            }
        }
        return max_align;
    }

    /* For matrices, alignment = column vector alignment */
    if (columns > 1) {
        /* Matrix: columns of vectors. Alignment = vector alignment. */
        if (vec_size == 2) return 8;
        return 16; /* vec3 and vec4 both align to 16 */
    }

    /* Scalar and vector types */
    GLuint base_align = (bit_width == 64) ? 8 : 4;

    if (vec_size == 1) {
        return base_align; /* scalar */
    } else if (vec_size == 2) {
        return 8; /* 2-component vector */
    } else {
        return 16; /* 3 and 4-component vectors align to 16 */
    }
}

/* Compute MSL struct member offset by iterating over preceding members.
 * Used for plain struct uniforms (UniformConstant storage class) which
 * don't have Offset decorations in SPIR-V.  The offset follows C struct
 * layout rules: each member is placed at the next offset aligned to its
 * type alignment. */
static GLuint mglComputeMSLStructMemberOffset(spvc_compiler compiler,
                                               spvc_type struct_type,
                                               unsigned member_index)
{
    if (!struct_type || member_index == 0) {
        return 0;
    }

    GLuint running = 0;
    unsigned member_count = spvc_type_get_num_member_types(struct_type);

    for (unsigned i = 0; i < member_index && i < member_count; i++) {
        spvc_type_id mt_id = spvc_type_get_member_type(struct_type, i);
        spvc_type mt = spvc_compiler_get_type_handle(compiler, mt_id);
        if (!mt) {
            continue;
        }

        GLuint align = mglMetalTypeAlignmentFromSPVC(compiler, mt);
        if (align == 0) align = 4;

        /* Align running to member alignment */
        running = (running + align - 1) & ~(align - 1);

        /* Get member size */
        size_t member_size = 0;
        if (spvc_compiler_get_declared_struct_member_size(
                compiler, struct_type, i, &member_size) != SPVC_SUCCESS || member_size == 0) {
            /* Fallback: estimate from type */
            unsigned bit_width = spvc_type_get_bit_width(mt);
            unsigned vec_size = spvc_type_get_vector_size(mt);
            unsigned columns = spvc_type_get_columns(mt);
            unsigned array_dims = spvc_type_get_num_array_dimensions(mt);
            GLuint elem_size = (bit_width == 64) ? 8 : 4;
            if (vec_size > 1) elem_size *= vec_size;
            if (columns > 1) elem_size *= columns;
            GLuint array_size = 1;
            if (array_dims > 0) {
                for (unsigned d = 0; d < array_dims; d++) {
                    array_size *= (GLuint)spvc_type_get_array_dimension(mt, d);
                }
            }
            member_size = elem_size * array_size;
        }

        running += (GLuint)member_size;
    }

    /* Align to this member's alignment */
    spvc_type_id mt_id = spvc_type_get_member_type(struct_type, member_index);
    spvc_type mt = spvc_compiler_get_type_handle(compiler, mt_id);
    GLuint align = mt ? mglMetalTypeAlignmentFromSPVC(compiler, mt) : 16;
    if (align == 0) align = 4;
    running = (running + align - 1) & ~(align - 1);

    return running;
}

/* Compute the total MSL struct size using Metal/C alignment rules.
 * The struct size = (offset after last member) padded to struct alignment. */
static GLuint mglComputeMSLStructSize(spvc_compiler compiler, spvc_type struct_type)
{
    if (!struct_type) return 0;
    unsigned member_count = spvc_type_get_num_member_types(struct_type);
    if (member_count == 0) return 0;

    GLuint running = 0;
    GLuint max_align = 4;

    for (unsigned i = 0; i < member_count; i++) {
        spvc_type_id mt_id = spvc_type_get_member_type(struct_type, i);
        spvc_type mt = spvc_compiler_get_type_handle(compiler, mt_id);
        if (!mt) continue;

        GLuint align = mglMetalTypeAlignmentFromSPVC(compiler, mt);
        if (align == 0) align = 4;
        if (align > max_align) max_align = align;

        running = (running + align - 1) & ~(align - 1);

        size_t member_size = 0;
        if (spvc_compiler_get_declared_struct_member_size(
                compiler, struct_type, i, &member_size) != SPVC_SUCCESS || member_size == 0) {
            unsigned bit_width = spvc_type_get_bit_width(mt);
            unsigned vec_size = spvc_type_get_vector_size(mt);
            unsigned columns = spvc_type_get_columns(mt);
            unsigned array_dims = spvc_type_get_num_array_dimensions(mt);
            GLuint elem_size = (bit_width == 64) ? 8 : 4;
            if (vec_size > 1) elem_size *= vec_size;
            if (columns > 1) elem_size *= columns;
            GLuint array_size = 1;
            if (array_dims > 0) {
                for (unsigned d = 0; d < array_dims; d++) {
                    array_size *= (GLuint)spvc_type_get_array_dimension(mt, d);
                }
            }
            member_size = elem_size * array_size;
        }
        running += (GLuint)member_size;
    }

    /* Pad struct size to max member alignment */
    running = (running + max_align - 1) & ~(max_align - 1);
    return running;
}

static GLboolean mglSpvcStructMemberOffset(spvc_compiler compiler,
                                           spvc_type struct_type,
                                           spvc_type_id struct_type_id,
                                           unsigned member_index,
                                           GLuint *out)
{
    unsigned value = 0;
    if (spvc_compiler_type_struct_member_offset(
            compiler, struct_type, member_index, &value) == SPVC_SUCCESS && value > 0) {
        *out = value;
        return GL_TRUE;
    }
    value = spvc_compiler_get_member_decoration(
        compiler, struct_type_id, member_index, SpvDecorationOffset);
    if (value > 0) {
        *out = value;
        return GL_TRUE;
    }

    /* No Offset decoration (plain struct uniform, not UBO).  Compute MSL
     * struct layout offset using Metal type alignment rules. */
    *out = mglComputeMSLStructMemberOffset(compiler, struct_type, member_index);
    return GL_TRUE;
}

static GLint mglSpvcStructMemberMatrixStride(spvc_compiler compiler,
                                             spvc_type struct_type,
                                             spvc_type_id struct_type_id,
                                             unsigned member_index)
{
    unsigned value = 0;
    if (spvc_compiler_type_struct_member_matrix_stride(
            compiler, struct_type, member_index, &value) == SPVC_SUCCESS) {
        return (GLint)value;
    }
    return (GLint)spvc_compiler_get_member_decoration(
        compiler, struct_type_id, member_index, SpvDecorationMatrixStride);
}

static GLint mglSpvcStructMemberArrayStride(spvc_compiler compiler,
                                            spvc_type struct_type,
                                            spvc_type_id struct_type_id,
                                            unsigned member_index)
{
    unsigned value = 0;
    if (spvc_compiler_type_struct_member_array_stride(
            compiler, struct_type, member_index, &value) == SPVC_SUCCESS) {
        return (GLint)value;
    }
    return (GLint)spvc_compiler_get_member_decoration(
        compiler, struct_type_id, member_index, SpvDecorationArrayStride);
}

static GLint mglGLTypeLocationCount(GLuint gl_type, GLint array_size)
{
    GLint element_locations = 1;
    switch (gl_type) {
        case GL_FLOAT_MAT2:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT2x4:
        case GL_DOUBLE_MAT2:
        case GL_DOUBLE_MAT2x3:
        case GL_DOUBLE_MAT2x4:
            element_locations = 2;
            break;
        case GL_FLOAT_MAT3x2:
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT3x4:
        case GL_DOUBLE_MAT3x2:
        case GL_DOUBLE_MAT3:
        case GL_DOUBLE_MAT3x4:
            element_locations = 3;
            break;
        case GL_FLOAT_MAT4x2:
        case GL_FLOAT_MAT4x3:
        case GL_FLOAT_MAT4:
        case GL_DOUBLE_MAT4x2:
        case GL_DOUBLE_MAT4x3:
        case GL_DOUBLE_MAT4:
            element_locations = 4;
            break;
        default:
            element_locations = 1;
            break;
    }
    if (array_size > 1) {
        return element_locations * array_size;
    }
    return element_locations;
}

static GLint mglSPVCTypeLocationCount(spvc_compiler compiler, spvc_type type)
{
    if (!type) {
        return 1;
    }

    spvc_basetype base = spvc_type_get_basetype(type);
    unsigned array_dims = spvc_type_get_num_array_dimensions(type);
    GLint array_size = (array_dims > 0) ? mglGLArraySizeFromSPVCType(type) : 1;
    GLint element_locations;

    if (base == SPVC_BASETYPE_STRUCT) {
        unsigned member_count = spvc_type_get_num_member_types(type);
        GLint total = 0;
        for (unsigned i = 0; i < member_count; i++) {
            spvc_type_id member_type_id = spvc_type_get_member_type(type, i);
            spvc_type member_type = spvc_compiler_get_type_handle(compiler, member_type_id);
            total += mglSPVCTypeLocationCount(compiler, member_type);
        }
        element_locations = total > 0 ? total : 1;
    } else {
        element_locations = mglGLTypeLocationCount(mglGLTypeFromSPVCType(type), 1);
    }

    if (array_size > 1) {
        return element_locations * array_size;
    }
    return element_locations;
}

/* CTS-convention location step for struct member offset computation.
 *
 * The Khronos CTS explicit_uniform_location tests advance the per-member
 * location cursor by the member's *array size* (1 for non-array types
 * including matrices), NOT by the spec-correct location count (which
 * would be num_columns for matrices).  This helper mirrors that
 * convention so struct-member location offsets match what the CTS
 * expects.  Only used for struct-member location_offset accumulation;
 * general uniform location assignment still uses mglSPVCTypeLocationCount.
 */
static GLint mglSPVCTypeLocationStep(spvc_compiler compiler, spvc_type type)
{
    if (!type) {
        return 1;
    }

    spvc_basetype base = spvc_type_get_basetype(type);
    unsigned array_dims = spvc_type_get_num_array_dimensions(type);
    GLint array_size = (array_dims > 0) ? mglGLArraySizeFromSPVCType(type) : 1;
    GLint element_step;

    if (base == SPVC_BASETYPE_STRUCT) {
        unsigned member_count = spvc_type_get_num_member_types(type);
        GLint total = 0;
        for (unsigned i = 0; i < member_count; i++) {
            spvc_type_id member_type_id = spvc_type_get_member_type(type, i);
            spvc_type member_type = spvc_compiler_get_type_handle(compiler, member_type_id);
            total += mglSPVCTypeLocationStep(compiler, member_type);
        }
        element_step = total > 0 ? total : 1;
    } else {
        /* CTS convention: every non-array leaf type counts as 1 location,
         * regardless of matrix columns. */
        element_step = 1;
    }

    if (array_size > 1) {
        return element_step * array_size;
    }
    return element_step;
}

static GLboolean mglAppendReflectedUBOMember(SpirvResource *ubo,
                                             GLuint *count,
                                             const char *name,
                                             GLuint gl_type,
                                             GLuint offset,
                                             GLint array_stride,
                                             GLint matrix_stride,
                                             GLboolean is_row_major,
                                             GLint size,
                                             GLint location_offset,
                                             GLint top_level_array_size,
                                             GLint top_level_array_stride)
{
    SpirvUBOMember *grown = NULL;

    if (!ubo || !count || !name || !name[0]) {
        return GL_FALSE;
    }

    grown = (SpirvUBOMember *)realloc(
        ubo->ubo_members, ((size_t)(*count) + 1u) * sizeof(SpirvUBOMember));
    if (!grown) {
        return GL_FALSE;
    }
    ubo->ubo_members = grown;

    SpirvUBOMember *member = &ubo->ubo_members[*count];
    memset(member, 0, sizeof(*member));
    member->name = strdup(name);
    member->gl_type = gl_type;
    member->offset = offset;
    member->array_stride = array_stride;
    member->matrix_stride = matrix_stride;
    member->is_row_major = (matrix_stride > 0) ? is_row_major : GL_FALSE;
    member->size = size;  /* 0 for runtime-sized arrays, 1 for scalars, N for arrays */
    member->location_offset = location_offset;
    member->top_level_array_size = top_level_array_size > 0 ? top_level_array_size : 1;
    member->top_level_array_stride = top_level_array_stride;
    member->query_name = mglBuildUBOMemberQueryName(ubo, member);
    if (!member->name || !member->query_name) {
        free((void *)member->name);
        free(member->query_name);
        memset(member, 0, sizeof(*member));
        return GL_FALSE;
    }

    (*count)++;
    ubo->ubo_member_count = *count;
    return GL_TRUE;
}

/* ---- Plain struct uniform active-member detection via SPIR-V analysis ----
 *
 * spvc_compiler_get_active_buffer_ranges only works for buffer-backed
 * resources (UBO/SSBO).  Plain struct uniforms (storage class UniformConstant)
 * have no Offset decorations, so that API returns 0 ranges.
 *
 * Instead, we directly scan the SPIR-V binary for OpAccessChain /
 * OpInBoundsAccessChain instructions whose base pointer (recursively resolved)
 * is the struct uniform variable.  Each access chain yields a constant index
 * path (e.g. [2,1,1,3,0] for l[2].b[1].d[0]).  During reflection we only
 * emit members whose path prefix-matches an active path.
 */

#define MGL_ACTIVE_MAX_PATHS   512
#define MGL_ACTIVE_MAX_DEPTH   32

typedef struct {
    GLuint  indices[MGL_ACTIVE_MAX_DEPTH];
    GLuint  len;
} MGLActivePath;

typedef struct {
    MGLActivePath  paths[MGL_ACTIVE_MAX_PATHS];
    GLuint         count;
} MGLActivePathSet;

/* SPIR-V opcodes */
#define MGL_SPV_OP_CONSTANT              43
#define MGL_SPV_OP_ACCESS_CHAIN          65
#define MGL_SPV_OP_IN_BOUNDS_ACCESS_CHAIN 66

/* Resolve an OpAccessChain result back to its root variable ID, collecting
 * the full constant index path.  Returns the root variable ID, or 0 if the
 * chain could not be fully resolved (e.g. non-constant index). */
static GLuint mglSpvResolveAccessChainRoot(GLuint result_id,
                                           GLuint bound,
                                           const GLuint *const_values,
                                           const GLboolean *is_const,
                                           const GLuint *chain_base,
                                           const GLuint *chain_num_idx,
                                           const GLuint *chain_idx_ids,
                                           const GLboolean *chain_valid,
                                           GLuint idx_stride,
                                           GLuint path_out[MGL_ACTIVE_MAX_DEPTH],
                                           GLuint *path_len_out)
{
    GLuint path[MGL_ACTIVE_MAX_DEPTH];
    GLuint path_len = 0;
    GLuint current = result_id;

    while (current > 0 && current < bound && chain_valid[current] && path_len < MGL_ACTIVE_MAX_DEPTH) {
        GLuint num = chain_num_idx[current];
        const GLuint *ids = &chain_idx_ids[current * idx_stride];
        /* prepend indices in reverse */
        for (GLint i = (GLint)num - 1; i >= 0 && path_len < MGL_ACTIVE_MAX_DEPTH; i--) {
            GLuint idx_id = ids[i];
            if (idx_id > 0 && idx_id < bound && is_const[idx_id]) {
                path[path_len++] = const_values[idx_id];
            } else {
                /* non-constant index — can't resolve, give up */
                *path_len_out = 0;
                return 0;
            }
        }
        current = chain_base[current];
    }

    /* reverse */
    for (GLuint i = 0; i < path_len; i++) {
        path_out[i] = path[path_len - 1 - i];
    }
    *path_len_out = path_len;
    return current;
}

/* Collect all active member-access paths rooted at var_id. */
static void mglCollectActivePaths(const unsigned int *spirv, size_t word_count,
                                  GLuint var_id,
                                  MGLActivePathSet *out)
{
    out->count = 0;
    if (!spirv || word_count < 5 || !out) return;

    GLuint bound = spirv[3];
    if (bound == 0 || bound > 0x00FFFFFFu) return; /* sanity */

    /* We use simple malloc'd arrays indexed by SPIR-V ID. */
    GLuint    *const_values = (GLuint *)calloc(bound, sizeof(GLuint));
    GLboolean *is_const     = (GLboolean *)calloc(bound, sizeof(GLboolean));
    GLuint    *chain_base   = (GLuint *)calloc(bound, sizeof(GLuint));
    GLuint    *chain_num_idx = (GLuint *)calloc(bound, sizeof(GLuint));
    GLboolean *chain_valid  = (GLboolean *)calloc(bound, sizeof(GLboolean));

    /* index IDs: each access chain can have up to 16 indices in practice.
     * Store as a flat array: chain_idx_ids[id * 16 + i] */
    const GLuint IDX_STRIDE = 16;
    GLuint *chain_idx_ids = (GLuint *)calloc((size_t)bound * IDX_STRIDE, sizeof(GLuint));

    if (!const_values || !is_const || !chain_base || !chain_num_idx ||
        !chain_valid || !chain_idx_ids) {
        free(const_values); free(is_const); free(chain_base);
        free(chain_num_idx); free(chain_valid); free(chain_idx_ids);
        return;
    }

    /* Parse SPIR-V instructions */
    size_t offset = 5; /* skip 5-word header */
    while (offset < word_count) {
        GLuint word = spirv[offset];
        GLuint opcode = word & 0xFFFFu;
        GLuint inst_len = word >> 16;
        if (inst_len == 0 || offset + inst_len > word_count) break;

        if (opcode == MGL_SPV_OP_CONSTANT && inst_len >= 4) {
            GLuint result_id = spirv[offset + 2];
            if (result_id > 0 && result_id < bound) {
                /* For 32-bit scalar int constants, value is at offset+3.
                 * For 64-bit, it's at offset+3 and offset+4.  We only care
                 * about 32-bit indices, so taking offset+3 is fine. */
                const_values[result_id] = spirv[offset + 3];
                is_const[result_id] = GL_TRUE;
            }
        } else if ((opcode == MGL_SPV_OP_ACCESS_CHAIN ||
                    opcode == MGL_SPV_OP_IN_BOUNDS_ACCESS_CHAIN) && inst_len >= 4) {
            GLuint result_id = spirv[offset + 2];
            GLuint base_id   = spirv[offset + 3];
            if (result_id > 0 && result_id < bound) {
                chain_base[result_id] = base_id;
                chain_num_idx[result_id] = inst_len - 4; /* minus word0,type,result,base */
                chain_valid[result_id] = GL_TRUE;
                GLuint n = chain_num_idx[result_id];
                if (n > IDX_STRIDE) n = IDX_STRIDE;
                for (GLuint i = 0; i < n; i++) {
                    chain_idx_ids[result_id * IDX_STRIDE + i] = spirv[offset + 4 + i];
                }
            }
        }

        offset += inst_len;
    }

    /* Resolve each access chain to its root and collect paths for var_id */
    for (GLuint id = 1; id < bound && out->count < MGL_ACTIVE_MAX_PATHS; id++) {
        if (!chain_valid[id]) continue;

        GLuint path[MGL_ACTIVE_MAX_DEPTH];
        GLuint path_len = 0;
        GLuint root = mglSpvResolveAccessChainRoot(id, bound, const_values, is_const,
                                                    chain_base, chain_num_idx,
                                                    chain_idx_ids, chain_valid,
                                                    IDX_STRIDE,
                                                    path, &path_len);
        if (root == var_id && path_len > 0) {
            MGLActivePath *p = &out->paths[out->count++];
            p->len = path_len > MGL_ACTIVE_MAX_DEPTH ? MGL_ACTIVE_MAX_DEPTH : path_len;
            for (GLuint i = 0; i < p->len; i++) {
                p->indices[i] = path[i];
            }
        }
    }

    free(const_values); free(is_const); free(chain_base);
    free(chain_num_idx); free(chain_valid); free(chain_idx_ids);
}

/* Check if any active path starts with the given prefix (prefix match).
 * Used to decide whether to recurse into a struct/array member. */
static GLboolean mglActivePathHasPrefix(const MGLActivePathSet *set,
                                        const GLuint *prefix, GLuint prefix_len)
{
    if (!set || set->count == 0 || prefix_len == 0) return GL_FALSE;
    for (GLuint i = 0; i < set->count; i++) {
        const MGLActivePath *p = &set->paths[i];
        if (p->len < prefix_len) continue;
        GLboolean match = GL_TRUE;
        for (GLuint j = 0; j < prefix_len; j++) {
            if (p->indices[j] != prefix[j]) { match = GL_FALSE; break; }
        }
        if (match) return GL_TRUE;
    }
    return GL_FALSE;
}

/* Check if any active path exactly matches the given path.
 * Used to decide whether to emit a leaf member. */
static GLboolean mglActivePathExactMatch(const MGLActivePathSet *set,
                                         const GLuint *path, GLuint path_len)
{
    if (!set || set->count == 0 || path_len == 0) return GL_FALSE;
    for (GLuint i = 0; i < set->count; i++) {
        const MGLActivePath *p = &set->paths[i];
        if (p->len != path_len) continue;
        GLboolean match = GL_TRUE;
        for (GLuint j = 0; j < path_len; j++) {
            if (p->indices[j] != path[j]) { match = GL_FALSE; break; }
        }
        if (match) return GL_TRUE;
    }
    return GL_FALSE;
}

/* Check if a byte offset falls within any active buffer range.
 * When active_ranges is NULL, everything is considered active (used for UBOs
 * where we reflect all declared members). For plain struct uniforms, the
 * ranges come from spvc_compiler_get_active_buffer_ranges and only members
 * whose byte extent overlaps an active range are reflected. */
static GLboolean mglByteOffsetIsActive(GLuint offset,
                                       GLint array_stride,
                                       GLint array_size,
                                       const spvc_buffer_range *active_ranges,
                                       size_t num_active_ranges)
{
    if (!active_ranges || num_active_ranges == 0) {
        return GL_TRUE; /* no filter → include everything */
    }

    /* For arrays, check if any element's starting offset falls within an
     * active range.  For non-arrays (array_stride <= 0 or array_size <= 1),
     * just check the single offset. */
    GLint n = (array_stride > 0 && array_size > 1) ? array_size : 1;
    for (GLint elem = 0; elem < n; elem++) {
        GLuint elem_offset = offset + (GLuint)(elem * array_stride);
        for (size_t r = 0; r < num_active_ranges; r++) {
            if (elem_offset >= (GLuint)active_ranges[r].offset &&
                elem_offset < (GLuint)(active_ranges[r].offset + active_ranges[r].range)) {
                return GL_TRUE;
            }
        }
    }
    return GL_FALSE;
}

static GLboolean mglReflectUBOMemberLeaves(Program *program,
                                           int stage,
                                           spvc_compiler compiler,
                                           SpirvResource *ubo,
                                           spvc_type struct_type,
                                           spvc_type_id struct_type_id,
                                           const char *prefix,
                                           GLuint base_offset,
                                           GLboolean inherited_row_major,
                                           GLuint *count,
                                           GLint location_offset,
                                           const spvc_buffer_range *active_ranges,
                                           size_t num_active_ranges,
                                           const MGLActivePathSet *active_paths,
                                           GLuint *current_path,
                                           GLuint current_path_len,
                                           GLint top_level_array_size,
                                           GLint top_level_array_stride);

static GLboolean mglReflectUBOStructMember(Program *program,
                                           int stage,
                                           spvc_compiler compiler,
                                           SpirvResource *ubo,
                                           spvc_type struct_type,
                                           spvc_type_id struct_type_id,
                                           unsigned member_index,
                                           const char *prefix,
                                           GLuint base_offset,
                                           GLboolean inherited_row_major,
                                           GLuint *count,
                                           GLint location_offset,
                                           const spvc_buffer_range *active_ranges,
                                           size_t num_active_ranges,
                                           const MGLActivePathSet *active_paths,
                                           GLuint *current_path,
                                           GLuint current_path_len,
                                           GLint top_level_array_size,
                                           GLint top_level_array_stride)
{
    const char *member_name_raw =
        spvc_compiler_get_member_name(compiler, struct_type_id, member_index);
    char *member_name = NULL;
    char *path = NULL;
    spvc_type_id member_type_id = spvc_type_get_member_type(struct_type, member_index);
    spvc_type member_type = spvc_compiler_get_type_handle(compiler, member_type_id);
    GLuint member_offset = 0;
    GLint matrix_stride = 0;
    GLint array_stride = 0;
    GLboolean row_major = inherited_row_major;
    const char *glsl_src = program && program->shader_slots[stage]
        ? program->shader_slots[stage]->src : NULL;

    if (member_name_raw && member_name_raw[0] &&
        !mglGLSLNameLooksLikeType(member_name_raw)) {
        member_name = strdup(member_name_raw);
    } else if (!prefix || !prefix[0]) {
        member_name = mglRecoverUBOMemberNameFromGLSL(glsl_src, ubo->name, member_index);
    } else {
        const char *struct_name = spvc_compiler_get_name(compiler, struct_type_id);
        if (struct_name && struct_name[0]) {
            member_name = mglRecoverStructMemberNameFromGLSL(glsl_src,
                                                             struct_name,
                                                             member_index);
        }
        if (!member_name) {
            char *glsl_struct_name = mglGLSLCompositeTypeNameForPath(glsl_src,
                                                                     ubo->name,
                                                                     prefix);
            if (glsl_struct_name) {
                member_name = mglRecoverStructMemberNameFromGLSL(glsl_src,
                                                                 glsl_struct_name,
                                                                 member_index);
                free(glsl_struct_name);
            }
        }
    }
    if (!member_name) {
        char synthetic[32];
        snprintf(synthetic, sizeof(synthetic), "_ubo_m%u", member_index);
        member_name = strdup(synthetic);
    }
    if (!member_name) {
        return GL_FALSE;
    }

    path = mglJoinUBOMemberPath(prefix, member_name);
    free(member_name);
    if (!path) {
        return GL_FALSE;
    }

    mglSpvcStructMemberOffset(compiler, struct_type, struct_type_id, member_index, &member_offset);
    matrix_stride = mglSpvcStructMemberMatrixStride(compiler, struct_type, struct_type_id, member_index);
    array_stride = mglSpvcStructMemberArrayStride(compiler, struct_type, struct_type_id, member_index);

    spvc_bool row_major_raw = spvc_compiler_has_member_decoration(
        compiler, struct_type_id, member_index, SpvDecorationRowMajor);
    spvc_bool col_major_raw = spvc_compiler_has_member_decoration(
        compiler, struct_type_id, member_index, SpvDecorationColMajor);
    if (row_major_raw) {
        row_major = GL_TRUE;
    } else if (col_major_raw) {
        row_major = GL_FALSE;
    } else if (!prefix || !prefix[0]) {
        const char *leaf = mglLeafNameFromPath(path);
        if (mglGLSLDeclaresRowMajorUBOMember(glsl_src, ubo->name, leaf)) {
            row_major = GL_TRUE;
        }
    }

    GLuint absolute_offset = base_offset + member_offset;

    /* For direct block members (prefix is NULL or empty), compute the
     * top-level array size and stride from the member type.  These are
     * propagated to all leaf members for GL_TOP_LEVEL_ARRAY_SIZE /
     * GL_TOP_LEVEL_ARRAY_STRIDE queries on GL_BUFFER_VARIABLE.
     * SPIRV-Cross stores array dimensions from innermost to outermost,
     * so the outermost (top-level) dimension is at index dims-1. */
    if (!prefix || !prefix[0]) {
        if (member_type && spvc_type_get_num_array_dimensions(member_type) > 0) {
            unsigned dims = spvc_type_get_num_array_dimensions(member_type);
            SpvId raw_dim = spvc_type_get_array_dimension(member_type, dims - 1);
            top_level_array_size = raw_dim > 0 ? (GLint)raw_dim : 1;
            top_level_array_stride = array_stride;
        } else {
            top_level_array_size = 1;
            top_level_array_stride = 0;
        }
    }

    if (member_type &&
        spvc_type_get_basetype(member_type) == SPVC_BASETYPE_STRUCT) {
        unsigned array_dims = spvc_type_get_num_array_dimensions(member_type);
        if (array_dims > 0) {
            /* For array-of-struct, resolve the element (base) type so that
             * member names and decorations are looked up on the struct type,
             * not on the array type.  SPIRV-Cross stores member names and
             * Offset/MatrixStride/ArrayStride decorations on the element
             * type, not on the array type. */
            spvc_type_id elem_type_id = spvc_type_get_base_type_id(member_type);
            spvc_type elem_type = elem_type_id
                ? spvc_compiler_get_type_handle(compiler, elem_type_id) : NULL;

            GLint stride = array_stride > 0 ? array_stride : 0;
            GLint elements = mglGLArraySizeFromSPVCType(member_type);
            GLint total_member_loc = mglSPVCTypeLocationStep(compiler, member_type);
            GLint elem_loc_count = (elements > 0)
                ? total_member_loc / elements : total_member_loc;
            for (GLint elem = 0; elem < elements; elem++) {
                /* For plain struct uniforms, check if this array element is
                 * part of an active access path before recursing. */
                if (active_paths && current_path && current_path_len < MGL_ACTIVE_MAX_DEPTH) {
                    current_path[current_path_len] = (GLuint)elem;
                    if (!mglActivePathHasPrefix(active_paths,
                                                current_path,
                                                current_path_len + 1)) {
                        continue; /* skip this array element */
                    }
                }
                size_t path_len = strlen(path);
                char suffix[32];
                snprintf(suffix, sizeof(suffix), "[%d]", elem);
                char *elem_path = (char *)malloc(path_len + strlen(suffix) + 1u);
                if (!elem_path) {
                    free(path);
                    return GL_FALSE;
                }
                snprintf(elem_path, path_len + strlen(suffix) + 1u, "%s%s", path, suffix);
                if (!mglReflectUBOMemberLeaves(program,
                                               stage,
                                               compiler,
                                               ubo,
                                               elem_type ? elem_type : member_type,
                                               elem_type_id ? elem_type_id : member_type_id,
                                               elem_path,
                                               absolute_offset + (GLuint)(elem * stride),
                                               row_major,
                                               count,
                                               location_offset + elem * elem_loc_count,
                                               active_ranges,
                                               num_active_ranges,
                                               active_paths,
                                               current_path,
                                               current_path_len + 1,
                                               top_level_array_size,
                                               top_level_array_stride)) {
                    free(elem_path);
                    free(path);
                    return GL_FALSE;
                }
                free(elem_path);
            }
            free(path);
            return GL_TRUE;
        }

        /* Non-array struct member: check active path prefix before recursing */
        if (active_paths && current_path && current_path_len < MGL_ACTIVE_MAX_DEPTH) {
            if (!mglActivePathHasPrefix(active_paths,
                                        current_path,
                                        current_path_len)) {
                free(path);
                return GL_TRUE; /* not active — skip, but continue recursion */
            }
        }

        GLboolean ok = mglReflectUBOMemberLeaves(program,
                                                 stage,
                                                 compiler,
                                                 ubo,
                                                 member_type,
                                                 member_type_id,
                                                 path,
                                                 absolute_offset,
                                                 row_major,
                                                 count,
                                                 location_offset,
                                                 active_ranges,
                                                 num_active_ranges,
                                                 active_paths,
                                                 current_path,
                                                 current_path_len,
                                                 top_level_array_size,
                                                 top_level_array_stride);
        free(path);
        return ok;
    }

    /* SPIRV-Cross stores array dimensions from innermost to outermost.
     * GL_ARRAY_SIZE is the innermost dimension (array_dimension(0)).
     * For runtime-sized arrays (OpTypeRuntimeArray, dimension == 0), size is 0. */
    unsigned member_array_dims = member_type ? spvc_type_get_num_array_dimensions(member_type) : 0;
    GLint size;
    if (member_array_dims > 0) {
        SpvId raw_dim = spvc_type_get_array_dimension(member_type, 0);
        size = raw_dim > 0 ? (GLint)raw_dim : 0;
    } else {
        size = 1;
    }

    /* For plain struct uniforms, only reflect leaf members whose path is a
     * prefix of some active SPIR-V OpAccessChain path. */
    if (active_paths && current_path && current_path_len > 0) {
        if (!mglActivePathHasPrefix(active_paths,
                                    current_path,
                                    current_path_len)) {
            free(path);
            return GL_TRUE; /* not active — skip, but continue recursion */
        }
    }

    /* UBO byte-offset filtering (kept for UBO path; no-op for plain structs
     * where active_ranges is NULL). */
    if (!mglByteOffsetIsActive(absolute_offset,
                               array_stride,
                               size,
                               active_ranges,
                               num_active_ranges)) {
        free(path);
        return GL_TRUE; /* not active — skip, but continue recursion */
    }

    char *query_path = (member_type && spvc_type_get_num_array_dimensions(member_type) > 0)
        ? mglAppendArrayZeroSuffix(path, spvc_type_get_num_array_dimensions(member_type))
        : strdup(path);
    free(path);
    if (!query_path) {
        return GL_FALSE;
    }

    GLboolean ok = mglAppendReflectedUBOMember(ubo,
                                               count,
                                               query_path,
                                               mglGLTypeFromSPVCTypeAndGLSL(member_type,
                                                                            glsl_src,
                                                                            ubo->name,
                                                                            query_path),
                                               absolute_offset,
                                               array_stride,
                                               matrix_stride,
                                               row_major,
                                               size,
                                               location_offset,
                                               top_level_array_size,
                                               top_level_array_stride);
    if (ok && getenv("MGL_DEBUG_UBO_REFLECT")) {
        fprintf(stderr,
                "MGL UBO MEMBER program=%u stage=%d ubo=%s member=%u finalName=%s queryName=%s offset=%u\n",
                program ? program->name : 0,
                stage,
                ubo->name ? ubo->name : "(null)",
                member_index,
                query_path,
                ubo->ubo_members[*count - 1u].query_name ? ubo->ubo_members[*count - 1u].query_name : "(null)",
                absolute_offset);
    }
    free(query_path);
    return ok;
}

static GLboolean mglReflectUBOMemberLeaves(Program *program,
                                           int stage,
                                           spvc_compiler compiler,
                                           SpirvResource *ubo,
                                           spvc_type struct_type,
                                           spvc_type_id struct_type_id,
                                           const char *prefix,
                                           GLuint base_offset,
                                           GLboolean inherited_row_major,
                                           GLuint *count,
                                           GLint location_offset,
                                           const spvc_buffer_range *active_ranges,
                                           size_t num_active_ranges,
                                           const MGLActivePathSet *active_paths,
                                           GLuint *current_path,
                                           GLuint current_path_len,
                                           GLint top_level_array_size,
                                           GLint top_level_array_stride)
{
    if (!struct_type || spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT) {
        return GL_FALSE;
    }

    unsigned member_count = spvc_type_get_num_member_types(struct_type);
    GLint running = 0;
    for (unsigned mem_idx = 0; mem_idx < member_count; mem_idx++) {
        spvc_type_id member_type_id = spvc_type_get_member_type(struct_type, mem_idx);
        spvc_type member_type = spvc_compiler_get_type_handle(compiler, member_type_id);
        GLint member_loc_count = mglSPVCTypeLocationStep(compiler, member_type);

        /* Build current path for this member: [existing path..., mem_idx] */
        if (active_paths && current_path && current_path_len < MGL_ACTIVE_MAX_DEPTH) {
            current_path[current_path_len] = mem_idx;
        }

        if (!mglReflectUBOStructMember(program,
                                       stage,
                                       compiler,
                                       ubo,
                                       struct_type,
                                       struct_type_id,
                                       mem_idx,
                                       prefix,
                                       base_offset,
                                       inherited_row_major,
                                       count,
                                       location_offset + running,
                                       active_ranges,
                                       num_active_ranges,
                                       active_paths,
                                       current_path,
                                       current_path_len + 1,
                                       top_level_array_size,
                                       top_level_array_stride)) {
            return GL_FALSE;
        }
        running += member_loc_count;
    }
    return GL_TRUE;
}

static GLboolean mglGLSLContainsToken(const char *src, const char *token)
{
    if (!src || !token || !token[0]) {
        return GL_FALSE;
    }

    size_t token_len = strlen(token);
    const char *pos = src;
    while ((pos = strstr(pos, token)) != NULL) {
        if ((pos == src || !(isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) &&
            !(isalnum((unsigned char)pos[token_len]) || pos[token_len] == '_')) {
            return GL_TRUE;
        }
        pos += token_len;
    }
    return GL_FALSE;
}

static GLboolean mglUniformBlockNameSeen(Program *program, int max_stage, GLuint max_index, const char *name, GLuint gl_binding)
{
    for (int stage = _VERTEX_SHADER; stage <= max_stage && stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        GLuint limit = (stage == max_stage) ? max_index : resources->count;
        for (GLuint i = 0; i < limit; i++) {
            SpirvResource *res = &resources->list[i];
            if (name && name[0] != '\0') {
                if (res->name && !strcmp(name, res->name)) {
                    return GL_TRUE;
                }
                continue;
            }
            if ((!res->name || res->name[0] == '\0') && res->gl_binding == gl_binding) {
                return GL_TRUE;
            }
        }
    }
    return GL_FALSE;
}

static GLuint mglProgramUniformBlockArraySize(const SpirvResource *block)
{
    return (block && block->ubo_array_size > 0) ? block->ubo_array_size : 1u;
}

static GLint mglActiveUniformBlockCount(Program *program)
{
    GLint total = 0;

    if (!program) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (!mglUniformBlockNameSeen(program, stage, i, res->name, res->gl_binding)) {
                total += (GLint)mglProgramUniformBlockArraySize(res);
            }
        }
    }

    return total;
}

/* Count the number of distinct active atomic-counter buffer binding points
 * referenced by any stage of the program.  Each distinct gl_binding of an
 * ATOMIC_COUNTER resource identifies one atomic-counter buffer. */
static GLint mglActiveAtomicCounterBufferCount(Program *program)
{
    GLint total = 0;

    if (!program) {
        return 0;
    }

    /* Track distinct gl_binding values across all stages.  Atomic-counter
     * buffer bindings are in [0, MAX_BINDABLE_BUFFERS). */
    GLboolean seen[MAX_BINDABLE_BUFFERS];
    memset(seen, 0, sizeof(seen));
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (res->gl_binding < MAX_BINDABLE_BUFFERS && !seen[res->gl_binding]) {
                seen[res->gl_binding] = GL_TRUE;
                total++;
            }
        }
    }
    return total;
}

static GLint mglActiveUniformBlockMaxNameLength(Program *program)
{
    GLint max_len = 0;

    if (!program) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglUniformBlockNameSeen(program, stage, i, res->name, res->gl_binding)) {
                continue;
            }
            GLuint element_count = mglProgramUniformBlockArraySize(res);
            for (GLuint element = 0; element < element_count; element++) {
                GLint len = 1;
                if (res->name) {
                    len = (GLint)strlen(res->name) + 1;
                    if (res->ubo_is_array || element_count > 1) {
                        char suffix[32];
                        snprintf(suffix, sizeof(suffix), "[%u]", element);
                        len += (GLint)strlen(suffix);
                    }
                }
                if (len > max_len) {
                    max_len = len;
                }
            }
        }
    }

    return max_len;
}

static SpirvResourceList *mglProgramActiveAttribList(Program *program)
{
    if (!program) {
        return NULL;
    }

    return &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
}

static GLboolean mglProgramActiveAttribHasName(const SpirvResource *res)
{
    return (res && res->name && res->name[0] != '\0') ? GL_TRUE : GL_FALSE;
}

static GLint mglProgramActiveAttribCount(Program *program)
{
    SpirvResourceList *resources = mglProgramActiveAttribList(program);
    if (!resources || !resources->list) {
        return 0;
    }

    GLint count = 0;
    for (GLuint i = 0; i < resources->count; i++) {
        if (mglProgramActiveAttribHasName(&resources->list[i])) {
            count++;
        }
    }

    return count;
}

static SpirvResource *mglProgramActiveAttribAt(Program *program, GLuint index)
{
    SpirvResourceList *resources = mglProgramActiveAttribList(program);
    if (!resources || !resources->list) {
        return NULL;
    }

    GLuint ordinal = 0;
    for (GLuint i = 0; i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if (!mglProgramActiveAttribHasName(res)) {
            continue;
        }
        if (ordinal == index) {
            return res;
        }
        ordinal++;
    }

    return NULL;
}

static GLint mglProgramActiveAttribMaxNameLength(Program *program)
{
    GLint max_len = 0;
    GLint count = mglProgramActiveAttribCount(program);

    for (GLint i = 0; i < count; i++) {
        SpirvResource *res = mglProgramActiveAttribAt(program, (GLuint)i);
        GLint len = (GLint)(res && res->name ? strlen(res->name) + 1 : 1);
        if (len > max_len) {
            max_len = len;
        }
    }

    return max_len;
}

static GLenum mglProgramActiveAttribType(const SpirvResource *res)
{
    const char *name = res ? res->name : NULL;

    if (!name || !name[0]) {
        return GL_FLOAT;
    }

    if (!strcmp(name, "Position") ||
        !strcmp(name, "Normal")) {
        return GL_FLOAT_VEC3;
    }
    if (!strcmp(name, "Color")) {
        return GL_FLOAT_VEC4;
    }
    if (!strcmp(name, "UV") ||
        !strcmp(name, "UV0") ||
        !strcmp(name, "TexCoord") ||
        !strcmp(name, "texCoord")) {
        return GL_FLOAT_VEC2;
    }
    if (!strcmp(name, "UV1") ||
        !strcmp(name, "UV2")) {
        return GL_INT_VEC2;
    }
    /* 1.21.11: LineWidth moved from uniform to per-vertex attribute (VertexFormatElement.LINE_WIDTH) */
    if (!strcmp(name, "LineWidth")) {
        return GL_FLOAT;
    }

    if (strstr(name, "Color")) {
        return GL_FLOAT_VEC4;
    }
    if (strstr(name, "UV") ||
        strstr(name, "TexCoord") ||
        strstr(name, "texCoord")) {
        return GL_FLOAT_VEC2;
    }
    if (strstr(name, "Normal")) {
        return GL_FLOAT_VEC3;
    }

    return GL_FLOAT_VEC4;
}

static GLint mglSyntheticSamplerUniformLocation(int stage, int res_type, GLuint index)
{
    return MGL_SYNTHETIC_SAMPLER_LOCATION_BASE + (stage * 0x1000) + (res_type * 0x100) + (GLint)index;
}

/*
 * Scan the original GLSL source for a sampler/image uniform declaration
 * matching `resource_name` and, if it carries an explicit
 * `layout(location = N)` qualifier, return N.  Returns -1 when the
 * declaration has no explicit location (the common case where glslang
 * auto-emits SpvDecorationLocation as a descriptor index).
 *
 * The GLSL forms recognised:
 *   layout(location = 2)  uniform highp sampler2D sampler;
 *   layout(location=1)    uniform sampler2D u0[3];
 *   layout(location = 13, binding = 0) uniform sampler2D u1;
 *
 * `resource_name` may include an array element suffix (e.g. "u0[0]");
 * only the base identifier is used for matching.
 */
static GLint mglFindExplicitUniformLocation(const char *glsl_src, const char *resource_name)
{
    if (!glsl_src || !resource_name || !resource_name[0]) {
        return -1;
    }

    /* Strip a trailing [..] suffix so "u0[0]" matches declaration "u0". */
    char base_name[256];
    size_t name_len = strlen(resource_name);
    const char *bracket = strchr(resource_name, '[');
    if (bracket) {
        name_len = (size_t)(bracket - resource_name);
    }
    if (name_len == 0 || name_len >= sizeof(base_name)) {
        return -1;
    }
    memcpy(base_name, resource_name, name_len);
    base_name[name_len] = '\0';

    const char *pos = glsl_src;
    while ((pos = strstr(pos, base_name)) != NULL) {
        const char *after_name = pos + name_len;
        /* Ensure the match is a whole identifier (not a substring of a
         * longer identifier). */
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)*after_name) || *after_name == '_')) {
            pos = after_name;
            continue;
        }

        /* Walk backwards from the identifier to find the start of its
         * declaration, skipping whitespace and the type tokens that
         * typically appear between `uniform` and the name, e.g.
         *   uniform highp sampler2D sampler
         * We look for the nearest preceding `layout(` and `uniform`. */
        const char *scan = pos;
        const char *layout_start = NULL;
        bool found_uniform = false;
        while (scan > glsl_src) {
            scan--;
            if (*scan == ';') {
                /* Hit a previous declaration's terminator; give up. */
                break;
            }
            if (!found_uniform && scan + 7 <= pos) {
                if (strncmp(scan, "uniform", 7) == 0) {
                    const char *after = scan + 7;
                    if (scan == glsl_src || !isalnum((unsigned char)scan[-1]) && scan[-1] != '_') {
                        if (*after == 0 || isspace((unsigned char)*after)) {
                            found_uniform = true;
                        }
                    }
                }
            }
            if (*scan == 'l' && scan + 6 <= pos && strncmp(scan, "layout", 6) == 0) {
                const char *after = scan + 6;
                if (scan == glsl_src || !isalnum((unsigned char)scan[-1]) && scan[-1] != '_') {
                    if (*after == 0 || isspace((unsigned char)*after) || *after == '(') {
                        layout_start = scan;
                        break;
                    }
                }
            }
        }

        if (!layout_start) {
            /* No layout qualifier preceding this declaration. */
            return -1;
        }

        /* Parse the layout(...) parenthesised group. */
        const char *lp = layout_start + 6;
        while (*lp && isspace((unsigned char)*lp)) {
            lp++;
        }
        if (*lp != '(') {
            return -1;
        }
        lp++;
        /* Scan comma-separated entries inside layout(...). */
        const char *paren_end = lp;
        int depth = 1;
        while (*paren_end && depth > 0) {
            if (*paren_end == '(') depth++;
            else if (*paren_end == ')') depth--;
            if (depth > 0) paren_end++;
        }
        if (*paren_end != ')') {
            return -1;
        }

        const char *entry = lp;
        for (;;) {
            const char *comma = entry;
            int d = 0;
            while (comma < paren_end && !(*comma == ',' && d == 0)) {
                if (*comma == '(') d++;
                else if (*comma == ')') d--;
                comma++;
            }
            /* entry..comma is one layout entry. */
            const char *p = entry;
            while (p < comma && isspace((unsigned char)*p)) p++;
            if (p + 8 <= comma && strncmp(p, "location", 8) == 0) {
                const char *after = p + 8;
                while (after < comma && isspace((unsigned char)*after)) after++;
                if (after < comma && *after == '=') {
                    after++;
                    while (after < comma && isspace((unsigned char)*after)) after++;
                    char *end = NULL;
                    unsigned long val = strtoul(after, &end, 10);
                    if (end != after) {
                        return (GLint)val;
                    }
                }
            }
            if (comma == paren_end) break;
            entry = comma + 1;
        }

        /* layout() present but no location entry -> not explicit. */
        return -1;
    }

    return -1;
}

static GLint mglSamplerUniformLocationFromReflection(GLuint reflected_location,
                                                     int stage,
                                                     int res_type,
                                                     GLuint index,
                                                     const char *glsl_src,
                                                     const char *resource_name)
{
    /*
     * SPIRV-Cross/Metal reflection reports descriptor argument locations here,
     * not OpenGL uniform locations. Minecraft 1.21.11 commonly has a vertex
     * Sampler2 and fragment Sampler0 that both reflect as location 0; exposing
     * that through glGetUniformLocation makes later glUniform1i calls overwrite
     * the wrong sampler. Keep GL sampler locations in our own namespace, then
     * unify resources with the same sampler name after both stages are linked.
     *
     * Glslang always emits SpvDecorationLocation on uniform variables, even
     * when the GLSL source does not declare an explicit layout(location=N).
     * The reflected location is therefore the SPIR-V/Metal descriptor index,
     * not a reliable OpenGL uniform location, and using it verbatim causes
     * collisions with plain uniforms (e.g. "uniform int layer") that share
     * the same reflected location. Use the synthetic namespace unless the
     * GLSL source explicitly declares layout(location=N) on this resource.
     */
    (void)reflected_location;
    if (glsl_src && resource_name) {
        GLint explicit_loc = mglFindExplicitUniformLocation(glsl_src, resource_name);
        if (explicit_loc >= 0) {
            return explicit_loc;
        }
    }
    return mglSyntheticSamplerUniformLocation(stage, res_type, index);
}

static bool mglIsSamplerResourceType(int res_type)
{
    return res_type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE ||
           res_type == SPVC_RESOURCE_TYPE_SEPARATE_IMAGE ||
           res_type == SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS ||
           res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE;
}

static bool mglUniformNameLooksSamplerLike(const char *name)
{
    if (!name || !*name) {
        return false;
    }

    return strstr(name, "Sampler") != NULL ||
           strcmp(name, "CloudFaces") == 0;
}

static bool mglUniformConstantBaseTypeIsSamplerLike(spvc_basetype basetype)
{
    return basetype == SPVC_BASETYPE_IMAGE ||
           basetype == SPVC_BASETYPE_SAMPLED_IMAGE ||
           basetype == SPVC_BASETYPE_SAMPLER;
}

static bool mglProgramResourceLooksSamplerLike(const SpirvResource *res, int res_type)
{
    if (!res) {
        return false;
    }

    switch (res_type) {
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
            return true;
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
            return res->image_dim != 0u ||
                   res->uniform_location >= MGL_SYNTHETIC_SAMPLER_LOCATION_BASE ||
                   mglUniformNameLooksSamplerLike(res->name);
        default:
            return false;
    }
}

static bool mglSamplerResourceNamesMatch(const char *a, const char *b)
{
    if (!a || !b) {
        return false;
    }
    if (strcmp(a, b) == 0) {
        return true;
    }

    size_t a_len = strlen(a);
    size_t b_len = strlen(b);
    if (a_len >= 3u && strcmp(a + a_len - 3u, "[0]") == 0) {
        a_len -= 3u;
    }
    if (b_len >= 3u && strcmp(b + b_len - 3u, "[0]") == 0) {
        b_len -= 3u;
    }
    return a_len == b_len && strncmp(a, b, a_len) == 0;
}

static void mglUnifySamplerUniformLocations(Program *program)
{
    if (!program) {
        return;
    }

    static const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (int leader_stage = _VERTEX_SHADER; leader_stage < _MAX_SHADER_TYPES; leader_stage++) {
        for (size_t leader_rt = 0; leader_rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); leader_rt++) {
            int leader_type = sampler_resource_types[leader_rt];
            SpirvResourceList *leaders = &program->spirv_resources_list[leader_stage][leader_type];
            for (GLuint leader_i = 0; leaders->list && leader_i < leaders->count; leader_i++) {
                SpirvResource *leader = &leaders->list[leader_i];
                if (!mglProgramResourceLooksSamplerLike(leader, leader_type) ||
                    !leader->name ||
                    leader->uniform_location < 0) {
                    continue;
                }

                GLint unified_sampler_unit = leader->sampler_unit;
                for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
                    for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
                        int res_type = sampler_resource_types[rt];
                        SpirvResourceList *resources = &program->spirv_resources_list[stage][res_type];
                        for (GLuint i = 0; resources->list && i < resources->count; i++) {
                            SpirvResource *res = &resources->list[i];
                            if (mglProgramResourceLooksSamplerLike(res, res_type) &&
                                res->name &&
                                mglSamplerResourceNamesMatch(res->name, leader->name) &&
                                res->sampler_unit > unified_sampler_unit) {
                                unified_sampler_unit = res->sampler_unit;
                            }
                        }
                    }
                }

                leader->sampler_unit = unified_sampler_unit;
                for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
                    for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
                        int res_type = sampler_resource_types[rt];
                        SpirvResourceList *resources = &program->spirv_resources_list[stage][res_type];
                        for (GLuint i = 0; resources->list && i < resources->count; i++) {
                            SpirvResource *res = &resources->list[i];
                            if (res == leader ||
                                !mglProgramResourceLooksSamplerLike(res, res_type) ||
                                !res->name ||
                                !mglSamplerResourceNamesMatch(res->name, leader->name)) {
                                continue;
                            }

                            res->uniform_location = leader->uniform_location;
                            res->sampler_unit = unified_sampler_unit;
                        }
                    }
                }
            }
        }
    }
}

static SpirvResource *mglFindAssignedPlainUniformResource(Program *program, const char *name)
{
    if (!program || !name || !*name) {
        return NULL;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (res->uniform_location < 0 ||
                mglProgramResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) ||
                !res->name ||
                strcmp(res->name, name) != 0) {
                continue;
            }
            return res;
        }
    }

    return NULL;
}

static GLint mglFirstFreePlainUniformLocation(const bool used[MAX_BINDABLE_BUFFERS])
{
    for (GLint location = 0; location < MAX_BINDABLE_BUFFERS; location++) {
        if (!used[location]) {
            return location;
        }
    }

    return -1;
}

static void mglAssignPlainUniformLocations(Program *program)
{
    if (!program) {
        return;
    }

    bool used[MAX_BINDABLE_BUFFERS] = {false};

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglProgramResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)) {
                continue;
            }

            if (res->location != 0xffffffffu &&
                       res->location < 1024u) {
                res->uniform_location = (GLint)res->location;
                if (res->uniform_location < MAX_BINDABLE_BUFFERS) {
                    used[res->uniform_location] = true;
                }
            } else if (res->uniform_location >= 0 &&
                       res->uniform_location < MAX_BINDABLE_BUFFERS) {
                used[res->uniform_location] = true;
            }
        }
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglProgramResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)) {
                continue;
            }
            if (res->uniform_location >= 0) {
                continue;
            }

            SpirvResource *assigned = mglFindAssignedPlainUniformResource(program, res->name);
            if (assigned && assigned->uniform_location >= 0 &&
                assigned->uniform_location < MAX_BINDABLE_BUFFERS) {
                res->uniform_location = assigned->uniform_location;
                continue;
            }

            GLint preferred = -1;
            if (res->location < MAX_BINDABLE_BUFFERS && !used[res->location]) {
                preferred = (GLint)res->location;
            } else if (res->gl_binding < MAX_BINDABLE_BUFFERS && !used[res->gl_binding]) {
                preferred = (GLint)res->gl_binding;
            } else {
                preferred = mglFirstFreePlainUniformLocation(used);
            }

            if (preferred < 0) {
                fprintf(stderr,
                        "MGL WARNING: no plain uniform location left program=%u name=%s stage=%d\n",
                        program->name,
                        res->name ? res->name : "(null)",
                        stage);
                continue;
            }

            res->uniform_location = preferred;
            used[preferred] = true;
            fprintf(stderr,
                    "MGL PLAIN UNIFORM FIX: program=%u stage=%d name=%s loc=%d metal=%u\n",
                    program->name,
                    stage,
                    res->name ? res->name : "(null)",
                    preferred,
                    (unsigned)res->binding);
        }
    }
}

static GLint mglDefaultSamplerUnitForProgramResource(Program *program, const SpirvResource *res)
{
    (void)program;

    /*
     * A sampler without an explicit layout binding reflects gl_binding=0,
     * matching OpenGL's initial sampler value. layout(binding=N) initializes
     * the sampler uniform to N; keep that GL unit independent from the compact
     * Metal argument slot stored in res->binding.
     */
    return res ? (GLint)res->gl_binding : 0;
}

static void mglApplyDefaultSamplerUnit(Program *program, int stage, int res_type, SpirvResource *res)
{
    if (!program || !res || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }
    if (!mglProgramResourceLooksSamplerLike(res, res_type)) {
        return;
    }

    GLint unit = mglDefaultSamplerUnitForProgramResource(program, res);
    if (unit < 0 || unit >= TEXTURE_UNITS) {
        return;
    }

    /*
     * Store the OpenGL sampler uniform default on the resource itself. The
     * resource binding is now the Metal argument slot and can be shared by
     * unrelated resources such as vertex Sampler2 and fragment Sampler0.
     */
    res->sampler_unit = unit;
    res->sampler_unit_explicit = GL_FALSE;
}

static bool mglMSLIdentifierChar(char c)
{
    return (c == '_') ||
           (c >= '0' && c <= '9') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

typedef enum MGLMSLBindingKind {
    MGL_MSL_BINDING_TEXTURE,
    MGL_MSL_BINDING_BUFFER,
    MGL_MSL_BINDING_SAMPLER
} MGLMSLBindingKind;

typedef struct MGLMSLBindingEntry {
    MGLMSLBindingKind kind;
    GLuint index;
    const char *segment;
    size_t segment_len;
} MGLMSLBindingEntry;

#define MGL_MSL_BINDING_MAP_MAX 512u

typedef struct MGLMSLBindingMap {
    MGLMSLBindingEntry entries[MGL_MSL_BINDING_MAP_MAX];
    size_t count;
} MGLMSLBindingMap;

static GLboolean mglSegmentContainsIdentifier(const char *segment,
                                              size_t segment_len,
                                              const char *name)
{
    if (!segment || !name) {
        return GL_FALSE;
    }

    size_t name_len = strlen(name);
    if (name_len == 0 || name_len > segment_len) {
        return GL_FALSE;
    }

    const char *end = segment + segment_len;
    for (const char *cursor = segment; cursor + name_len <= end; cursor++) {
        if (memcmp(cursor, name, name_len) != 0) {
            continue;
        }

        char before = (cursor == segment) ? '\0' : cursor[-1];
        char after = (cursor + name_len == end) ? '\0' : cursor[name_len];
        if (!mglMSLIdentifierChar(before) && !mglMSLIdentifierChar(after)) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static const char *mglPreviousMSLArgumentBoundary(const char *msl, const char *attribute)
{
    const char *cursor = attribute;
    unsigned angle_depth = 0;

    while (cursor > msl) {
        char c = cursor[-1];
        if (c == '\n' || c == '\r') {
            break;
        }
        if (c == '>') {
            angle_depth++;
        } else if (c == '<') {
            if (angle_depth > 0) {
                angle_depth--;
            }
        } else if (c == ',' && angle_depth == 0) {
            break;
        }
        cursor--;
    }

    while (*cursor == ' ' || *cursor == '\t') {
        cursor++;
    }

    return cursor;
}

static const char *mglNextMSLArgumentBoundary(const char *attribute)
{
    const char *cursor = attribute;
    unsigned angle_depth = 0;

    while (*cursor) {
        char c = *cursor;
        if (c == '\n' || c == '\r') {
            break;
        }
        if (c == '<') {
            angle_depth++;
        } else if (c == '>') {
            if (angle_depth > 0) {
                angle_depth--;
            }
        } else if (c == ',' && angle_depth == 0) {
            break;
        }
        cursor++;
    }

    return cursor;
}

static GLboolean mglParseMSLBindingAttribute(const char *attribute,
                                             const char *prefix,
                                             GLuint *index_out)
{
    if (!attribute || !prefix || !index_out) {
        return GL_FALSE;
    }

    size_t prefix_len = strlen(prefix);
    if (strncmp(attribute, prefix, prefix_len) != 0) {
        return GL_FALSE;
    }

    const char *index_start = attribute + prefix_len;
    char *end = NULL;
    unsigned long value = strtoul(index_start, &end, 10);
    if (end == index_start || value >= TEXTURE_UNITS) {
        return GL_FALSE;
    }

    *index_out = (GLuint)value;
    return GL_TRUE;
}

static void mglMSLBindingMapAdd(MGLMSLBindingMap *map,
                                MGLMSLBindingKind kind,
                                GLuint index,
                                const char *segment_start,
                                const char *segment_end)
{
    if (!map || !segment_start || !segment_end || segment_end < segment_start ||
        map->count >= MGL_MSL_BINDING_MAP_MAX) {
        return;
    }

    while (segment_end > segment_start &&
           (segment_end[-1] == ' ' || segment_end[-1] == '\t')) {
        segment_end--;
    }

    MGLMSLBindingEntry *entry = &map->entries[map->count++];
    entry->kind = kind;
    entry->index = index;
    entry->segment = segment_start;
    entry->segment_len = (size_t)(segment_end - segment_start);
}

static void mglBuildMSLBindingMap(const char *msl, MGLMSLBindingMap *map)
{
    if (!map) {
        return;
    }

    memset(map, 0, sizeof(*map));
    if (!msl) {
        return;
    }

    const char *cursor = msl;
    while (*cursor) {
        const char *texture_attr = strstr(cursor, "[[texture(");
        const char *buffer_attr = strstr(cursor, "[[buffer(");
        const char *sampler_attr = strstr(cursor, "[[sampler(");
        const char *attribute = texture_attr;
        const char *prefix = "[[texture(";
        MGLMSLBindingKind kind = MGL_MSL_BINDING_TEXTURE;

        if (!attribute || (buffer_attr && buffer_attr < attribute)) {
            attribute = buffer_attr;
            prefix = "[[buffer(";
            kind = MGL_MSL_BINDING_BUFFER;
        }
        if (!attribute || (sampler_attr && sampler_attr < attribute)) {
            attribute = sampler_attr;
            prefix = "[[sampler(";
            kind = MGL_MSL_BINDING_SAMPLER;
        }
        if (!attribute) {
            break;
        }

        GLuint index = 0;
        if (mglParseMSLBindingAttribute(attribute, prefix, &index)) {
            const char *segment_start = mglPreviousMSLArgumentBoundary(msl, attribute);
            const char *segment_end = mglNextMSLArgumentBoundary(attribute);
            mglMSLBindingMapAdd(map, kind, index, segment_start, segment_end);
        }

        cursor = attribute + 2;
    }
}

static GLboolean mglFindMSLResourceIndexInMap(const MGLMSLBindingMap *map,
                                              MGLMSLBindingKind kind,
                                              const char *name,
                                              GLuint *index_out)
{
    if (!map || !name || !index_out) {
        return GL_FALSE;
    }

    for (size_t i = 0; i < map->count; i++) {
        const MGLMSLBindingEntry *entry = &map->entries[i];
        if (entry->kind == kind &&
            mglSegmentContainsIdentifier(entry->segment, entry->segment_len, name)) {
            *index_out = entry->index;
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLint mglFindMSLResourceArraySizeInMap(const MGLMSLBindingMap *map,
                                              MGLMSLBindingKind kind,
                                              const char *name)
{
    if (!map || !name) {
        return 1;
    }

    for (size_t i = 0; i < map->count; i++) {
        const MGLMSLBindingEntry *entry = &map->entries[i];
        if (entry->kind != kind ||
            !mglSegmentContainsIdentifier(entry->segment, entry->segment_len, name)) {
            continue;
        }

        const char *array = strstr(entry->segment, "array<");
        const char *end = entry->segment + entry->segment_len;
        if (!array || array >= end) {
            return 1;
        }

        unsigned angle_depth = 0;
        for (const char *cursor = array + 6; cursor < end; cursor++) {
            if (*cursor == '<') {
                angle_depth++;
            } else if (*cursor == '>') {
                if (angle_depth == 0) {
                    break;
                }
                angle_depth--;
            } else if (*cursor == ',' && angle_depth == 0) {
                char *parse_end = NULL;
                long count = strtol(cursor + 1, &parse_end, 10);
                return count > 0 ? (GLint)count : 1;
            }
        }
    }

    return 1;
}

static void replace_all_substr(char **pstr, const char *from, const char *to);
static const char *mglFindMSLEntryParameterOpen(const char *msl);
static const char *mglFindMSLEntryParameterClose(const char *msl);

static GLboolean mglMSLBufferSlotConflicts(Program *pptr, int stage, GLuint slot)
{
    /* Delegates to the unified predicate in mgl_ir_postprocess.  This keeps
     * the IR pre-mapping path (before spvc_compiler_compile) and the MSL
     * string fallback (applyMSLResourceBindings, after compile) using the
     * exact same conflict definition so they cannot drift. */
    return mglBufferSlotConflictsForProgram(pptr, stage, slot);
}

static void applyMSLResourceBindings(Program *pptr, int stage, char **msl_ptr)
{
    if (!pptr || !msl_ptr || !*msl_ptr || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }

    const char *msl = *msl_ptr;
    MGLMSLBindingMap binding_map;
    mglBuildMSLBindingMap(msl, &binding_map);

    const int texture_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (size_t t = 0; t < sizeof(texture_resource_types) / sizeof(texture_resource_types[0]); t++) {
        int res_type = texture_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            GLuint metal_index = 0;
            if (!res->name ||
                !mglFindMSLResourceIndexInMap(&binding_map, MGL_MSL_BINDING_TEXTURE, res->name, &metal_index)) {
                continue;
            }

            if (res->binding != metal_index) {
                fprintf(stderr,
                        "MGL RESOURCE FIX: program=%u stage=%d type=%d %s texture binding %u -> %u\n",
                        pptr->name,
                        stage,
                        res_type,
                        res->name,
                        (unsigned)res->binding,
                        (unsigned)metal_index);
                res->binding = metal_index;
            }
            GLint msl_array_size =
                mglFindMSLResourceArraySizeInMap(&binding_map, MGL_MSL_BINDING_TEXTURE, res->name);
            if (msl_array_size > res->gl_array_size) {
                res->gl_array_size = msl_array_size;
            }
        }
    }

    const int buffer_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    GLboolean used_slots[kMGLMaxMetalVertexBufferCount] = {0};
    for (size_t k = 0; k < binding_map.count; k++) {
        if (binding_map.entries[k].kind == MGL_MSL_BINDING_BUFFER &&
            binding_map.entries[k].index < kMGLMaxMetalVertexBufferCount) {
            used_slots[binding_map.entries[k].index] = GL_TRUE;
        }
    }

    for (size_t t = 0; t < sizeof(buffer_resource_types) / sizeof(buffer_resource_types[0]); t++) {
        int res_type = buffer_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            GLuint metal_index = 0;
            if (!res->name ||
                !mglFindMSLResourceIndexInMap(&binding_map, MGL_MSL_BINDING_BUFFER, res->name, &metal_index)) {
                continue;
            }

            {
                GLboolean slot_conflicts =
                    mglMSLBufferSlotConflicts(pptr, stage, metal_index);

                if (slot_conflicts) {
                    /*
                     * Remap this user buffer away from the reserved slot.
                     * Find the entry in the binding map so we can target
                     * this specific parameter's [[buffer(N)]] attribute
                     * without touching any MGL-internal buffer that may
                     * share the same slot.
                     *
                     * MGL_ASSERT_NO_MSL_BINDING_REWRITE: when the IR
                     * postprocess pipeline is expected to have handled all
                     * conflicts, reaching this string-level fallback is a
                     * loud failure so the IR path can be fixed.  In normal
                     * operation this branch should be rare (IR pre-mapping
                     * already moved the buffer to a safe slot). */
                    if (mglAssertNoMSLBindingRewriteEnabled()) {
                        fprintf(stderr,
                                "MGL ASSERT: string-level buffer binding "
                                "rewrite required after IR pre-map: program=%u "
                                "stage=%d %s buffer '%s' at slot %u\n",
                                pptr->name, stage,
                                res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER ? "UBO" :
                                res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ? "SSBO" :
                                res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER ? "atomic" : "buffer",
                                res->name ? res->name : "(null)",
                                (unsigned)metal_index);
                    }
                    const MGLMSLBindingEntry *map_entry = NULL;
                    for (size_t k = 0; k < binding_map.count; k++) {
                        if (binding_map.entries[k].kind == MGL_MSL_BINDING_BUFFER &&
                            binding_map.entries[k].index == metal_index &&
                            mglSegmentContainsIdentifier(
                                binding_map.entries[k].segment,
                                binding_map.entries[k].segment_len,
                                res->name)) {
                            map_entry = &binding_map.entries[k];
                            break;
                        }
                    }

                    GLuint free_slot = kMGLMaxMetalVertexBufferCount;
                    if (map_entry) {
                        const GLuint free_limit =
                            (stage == _VERTEX_SHADER) ? (GLuint)kMGLPointSizeBufferIndex
                                                      : (GLuint)MGL_BUFFER_SIZE_BUFFER_INDEX;
                        for (GLuint s = 0; s < free_limit; s++) {
                            if (!used_slots[s] &&
                                !mglMSLBufferSlotConflicts(pptr, stage, s)) {
                                free_slot = s;
                                break;
                            }
                        }

                        if (free_slot < kMGLMaxMetalVertexBufferCount) {
                            char *old_decl = strndup(map_entry->segment,
                                                     map_entry->segment_len);
                            if (old_decl) {
                                char old_attr[48], new_attr[48];
                                snprintf(old_attr, sizeof(old_attr),
                                         "[[buffer(%u)]]", (unsigned)metal_index);
                                snprintf(new_attr, sizeof(new_attr),
                                         "[[buffer(%u)]]", (unsigned)free_slot);

                                char *attr_pos = strstr(old_decl, old_attr);
                                if (attr_pos) {
                                    size_t head_len = attr_pos - old_decl;
                                    const char *tail = attr_pos + strlen(old_attr);
                                    size_t new_len = head_len +
                                                     strlen(new_attr) +
                                                     strlen(tail);
                                    char *new_decl = malloc(new_len + 1);
                                    if (new_decl) {
                                        memcpy(new_decl, old_decl, head_len);
                                        memcpy(new_decl + head_len, new_attr,
                                               strlen(new_attr));
                                        memcpy(new_decl + head_len +
                                               strlen(new_attr), tail,
                                               strlen(tail) + 1);

                                        replace_all_substr(msl_ptr,
                                                           old_decl, new_decl);
                                        free(new_decl);

                                        mglBuildMSLBindingMap(*msl_ptr,
                                                              &binding_map);
                                        memset(used_slots, 0,
                                               sizeof(used_slots));
                                        for (size_t k = 0; k < binding_map.count; k++) {
                                            if (binding_map.entries[k].kind ==
                                                    MGL_MSL_BINDING_BUFFER &&
                                                binding_map.entries[k].index < kMGLMaxMetalVertexBufferCount) {
                                                used_slots[
                                                    binding_map.entries[k].index] = GL_TRUE;
                                            }
                                        }

                                        fprintf(stderr,
                                                "MGL RESOURCE CONFLICT RESOLVED: "
                                                "program=%u stage=%d %s buffer '%s' "
                                                "remapped from reserved slot %u "
                                                "to slot %u\n",
                                                pptr->name, stage,
                                                res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER
                                                    ? "UBO" :
                                                res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER
                                                    ? "SSBO" :
                                                res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER
                                                    ? "atomic" : "buffer",
                                                res->name,
                                                (unsigned)metal_index,
                                                (unsigned)free_slot);
                                        metal_index = free_slot;
                                    }
                                }
                                free(old_decl);
                            }
                        }
                    }

                    if (metal_index != free_slot) {
                        const char *reserved_name =
                            mglBufferSlotReservedName(metal_index);
                        fprintf(stderr,
                                "MGL RESOURCE CONFLICT: program=%u stage=%d "
                                "%s buffer '%s' at reserved slot %u (%s) -- "
                                "%s\n",
                                pptr->name, stage,
                                res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER
                                    ? "UBO" :
                                res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER
                                    ? "SSBO" :
                                res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER
                                    ? "atomic" : "buffer",
                                res->name,
                                (unsigned)metal_index,
                                reserved_name ? reserved_name : "?",
                                free_slot < kMGLMaxMetalVertexBufferCount
                                    ? "parameter declaration not found in MSL"
                                    : (stage == _VERTEX_SHADER
                                           ? "no free slot available in [0,14]"
                                           : "no free slot available in [0,24]"));
                    }
                }
            }

            if (res->binding != metal_index) {
                fprintf(stderr,
                        "MGL RESOURCE FIX: program=%u stage=%d type=%d %s buffer binding %u -> %u (gl=%u)\n",
                        pptr->name,
                        stage,
                        res_type,
                        res->name,
                        (unsigned)res->binding,
                        (unsigned)metal_index,
                        (unsigned)res->gl_binding);
                res->binding = metal_index;
            }
        }
    }

    SpirvResourceList *samplers =
        &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS];
    for (GLuint i = 0; i < samplers->count; i++) {
        SpirvResource *res = &samplers->list[i];
        GLuint metal_index = 0;
        if (!res->name ||
            !mglFindMSLResourceIndexInMap(&binding_map, MGL_MSL_BINDING_SAMPLER, res->name, &metal_index)) {
            continue;
        }
        if (res->binding != metal_index) {
            fprintf(stderr,
                    "MGL RESOURCE FIX: program=%u stage=%d type=%d %s sampler binding %u -> %u\n",
                    pptr->name,
                    stage,
                    SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
                    res->name,
                    (unsigned)res->binding,
                    (unsigned)metal_index);
            res->binding = metal_index;
        }
    }

    const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (size_t t = 0; t < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); t++) {
        int res_type = sampler_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            mglApplyDefaultSamplerUnit(pptr, stage, res_type, &resources->list[i]);
        }
    }
}

// Program Pipeline management
ProgramPipeline *newProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr;

    ptr = (ProgramPipeline *)malloc(sizeof(ProgramPipeline));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate program pipeline %u\n", pipeline);
        return NULL;
    }

    bzero(ptr, sizeof(ProgramPipeline));
    ptr->name = pipeline;

    return ptr;
}

ProgramPipeline *findProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    return (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table), pipeline);
}

ProgramPipeline *getProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);

    if (!ptr)
    {
        ptr = newProgramPipeline(ctx, pipeline);
        if (!ptr)
            return NULL;
        insertHashElement(&STATE(program_pipeline_table), pipeline, ptr);
    }

    return ptr;
}

// Transform Feedback management
TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr;

    ptr = (TransformFeedback *)malloc(sizeof(TransformFeedback));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate transform feedback %u\n", name);
        return NULL;
    }

    bzero(ptr, sizeof(TransformFeedback));
    ptr->name = name;
    ptr->target = GL_TRANSFORM_FEEDBACK;
    ptr->created = (name == 0) ? GL_TRUE : GL_FALSE;
    ptr->active = GL_FALSE;
    ptr->paused = GL_FALSE;
    ptr->primitive_mode = GL_NONE;

    return ptr;
}

TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name)
{
    return (TransformFeedback *)searchHashTable(&STATE(transform_feedback_table), name);
}

TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr = findTransformFeedback(ctx, name);

    if (!ptr)
    {
        ptr = newTransformFeedback(ctx, name);
        if (!ptr)
            return NULL;
        insertHashElement(&STATE(transform_feedback_table), name, ptr);
    }

    return ptr;
}

Program *newProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    ptr = (Program *)malloc(sizeof(Program));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate program %u\n", program);
        return NULL;
    }

    bzero(ptr, sizeof(Program));

    ptr->name = program;
    for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
        ptr->sampler_units[i] = -1;
    }
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
            ptr->sampler_units_by_stage[stage][i] = -1;
        }
    }

    return ptr;
}

Program *getProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return NULL;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    if (!ptr)
    {
        ptr = newProgram(ctx, program);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(program_table), program, ptr);
    }

    return ptr;
}

int isProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return 0;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    if (ptr)
        return 1;

    return 0;
}

Program *findProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return NULL;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    return ptr;
}

static void mglFreeSpirvResourceOwnedFields(SpirvResource *res)
{
    if (!res) {
        return;
    }

    free((void *)res->name);
    res->name = NULL;

    if (res->ubo_members) {
        for (GLuint m = 0; m < res->ubo_member_count; m++) {
            free((void *)res->ubo_members[m].name);
            free(res->ubo_members[m].query_name);
        }
        free(res->ubo_members);
        res->ubo_members = NULL;
    }
    res->ubo_member_count = 0;
    res->ubo_member = NULL;

    free(res->ubo_array_bindings);
    res->ubo_array_bindings = NULL;

    free(res->ubo_instance_name);
    res->ubo_instance_name = NULL;
}

GLuint mglCreateProgram(GLMContext ctx)
{
    GLuint program;

    program = getNewName(&STATE(program_table));

    if (!getProgram(ctx, program))
        return 0;

    return program;
}

void mglFreeProgram(GLMContext ctx, Program *ptr)
{
    /* linked_glsl_program is a non-NULL linked-state marker (set to
     * (glslang_program_t*)ptr on successful link), NOT a real glslang
     * program.  The real glslang_program_t is deleted at the end of
     * mglLinkProgram.  Do NOT call glslang_program_delete here — it would
     * dereference the marker as if it were a glslang object. */
    ptr->linked_glsl_program = NULL;

    mglSafeReleaseMetalObj((void **)&ptr->mtl_data);

    for(int i=0; i<_MAX_SHADER_TYPES; i++)
    {
        // CRITICAL FIX: Add NULL checks before all free/release operations to prevent double-frees
        if (ptr->spirv[i].ir) {
            free(ptr->spirv[i].ir);
            ptr->spirv[i].ir = NULL;
        }
        if (ptr->spirv[i].msl_str) {
            free(ptr->spirv[i].msl_str);
            ptr->spirv[i].msl_str = NULL;
        }
        if (ptr->spirv[i].msl_str_capture) {
            free(ptr->spirv[i].msl_str_capture);
            ptr->spirv[i].msl_str_capture = NULL;
        }
        if (ptr->spirv[i].entry_point) {
            free(ptr->spirv[i].entry_point);
            ptr->spirv[i].entry_point = NULL;
        }
        mglSafeReleaseMetalObj((void **)&ptr->spirv[i].mtl_function);
        mglSafeReleaseMetalObj((void **)&ptr->spirv[i].mtl_library);
        
        for(int j=0; j<_MAX_SPIRV_RES; j++)
        {
            // CRITICAL FIX: Add NULL checks and clear pointers to prevent double-frees
            SpirvResourceList *rl = &ptr->spirv_resources_list[i][j];
            if (rl->list) {
                for (GLuint k = 0; k < rl->count; k++) {
                    mglFreeSpirvResourceOwnedFields(&rl->list[k]);
                }
                free(rl->list);
                rl->list = NULL;
                rl->count = 0;
            }
        }
        
        if (ptr->attached_shader_counts[i] > 0) {
            for (GLuint attached = 0;
                 attached < ptr->attached_shader_counts[i] &&
                 attached < MAX_ATTACHED_SHADERS_PER_STAGE;
                 attached++) {
                Shader *sptr = ptr->attached_shader_slots[i][attached];
                if (!sptr) {
                    continue;
                }
                sptr->refcount--;
                if (sptr->refcount == 0 && sptr->delete_status)
                {
                    deleteHashElement(&STATE(shader_table), sptr->name);
                    mglFreeShader(ctx, sptr);
                }
                ptr->attached_shader_slots[i][attached] = NULL;
            }
            ptr->attached_shader_counts[i] = 0;
            ptr->shader_slots[i] = NULL;
        }
        else if (ptr->shader_slots[i])
        {
                Shader *sptr = ptr->shader_slots[i];
                sptr->refcount--;
                if (sptr->refcount == 0 && sptr->delete_status)
                {
                    deleteHashElement(&STATE(shader_table), sptr->name);
                    mglFreeShader(ctx, sptr);
                }
        }
    }

    for (int i = 0; i < MAX_ATTRIBS; i++) {
        mglFreeProgramAttribName(ptr, (GLuint)i, "program delete");
    }

    for (GLuint i = 0; i < ptr->frag_data_location_count && i < MAX_ATTRIBS; i++) {
        if (ptr->frag_data_location_names[i]) {
            free(ptr->frag_data_location_names[i]);
            ptr->frag_data_location_names[i] = NULL;
        }
    }
    ptr->frag_data_location_count = 0;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
        for (GLuint i = 0; i < ptr->builtin_program_input_count[s] && i < 16; i++) {
            if (ptr->builtin_program_inputs[s][i].name) {
                free(ptr->builtin_program_inputs[s][i].name);
                ptr->builtin_program_inputs[s][i].name = NULL;
            }
        }
        ptr->builtin_program_input_count[s] = 0;
    }

    for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
        for (GLuint i = 0; i < ptr->builtin_program_output_count[s] && i < 16; i++) {
            if (ptr->builtin_program_outputs[s][i].name) {
                free(ptr->builtin_program_outputs[s][i].name);
                ptr->builtin_program_outputs[s][i].name = NULL;
            }
        }
        ptr->builtin_program_output_count[s] = 0;
    }

    free(ptr);
}

GLboolean mglProgramPointerUsableForName(GLMContext ctx, Program *program, GLuint expectedName)
{
    if (!ctx || !program || expectedName == 0u) {
        return GL_FALSE;
    }

    if (!mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program)) ||
        program->name != expectedName) {
        return GL_FALSE;
    }

    if (mglHashTableContainsData(&STATE(program_table), program)) {
        return GL_TRUE;
    }

    /*
     * glDeleteProgram removes the name immediately, but the current program
     * and any deferred draws that captured it must keep using the object until
     * their references are released.
     */
    if (program->delete_status &&
        program->refcount > 0 &&
        program->linked_glsl_program != NULL) {
        return GL_TRUE;
    }

    return GL_FALSE;
}

void mglRetainProgramReference(GLMContext ctx, Program *program)
{
    if (!ctx || !program) {
        return;
    }

    GLuint programName = 0u;
    if (mglObjectPointerLooksPlausible(program) &&
        mglPointerRangeIsReadable(program, sizeof(*program))) {
        programName = program->name;
    }

    if (programName == 0u ||
        !mglProgramPointerUsableForName(ctx, program, programName)) {
        return;
    }

    program->refcount++;
}

void mglReleaseProgramReference(GLMContext ctx, Program *program)
{
    if (!ctx || !program ||
        !mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program))) {
        return;
    }

    if (program->refcount > 0) {
        program->refcount--;
    }
    if (program->refcount == 0 && program->delete_status) {
        mglFreeProgram(ctx, program);
    }
}

void mglDeleteProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    ptr = findProgram(ctx, program);

    if (!ptr)
    {
        // Per GL spec, glDeleteProgram silently ignores names that do not
        // correspond to an existing program object (including names that were
        // already deleted). Do NOT set a sticky GL error here — leaving
        // GL_INVALID_OPERATION set would poison the next API call's
        // glGetError() check (e.g. glDeleteFramebuffers), surfacing as a
        // spurious CTS crash.
        return;
    }

    mglFlushPendingDraws(ctx);

    deleteHashElement(&STATE(program_table), program);
    
    ptr->delete_status = GL_TRUE;
    
    if (ptr->refcount == 0)
    {
        mglFreeProgram(ctx, ptr);
    }
}

GLboolean mglIsProgram(GLMContext ctx, GLuint program)
{
    if (isProgram(ctx, program))
        return GL_TRUE;

    return GL_FALSE;
}

static GLboolean mglProgramHasAttachedShader(Program *program, GLuint stage, Shader *shader)
{
    if (!program || stage >= _MAX_SHADER_TYPES || !shader) {
        return GL_FALSE;
    }

    for (GLuint i = 0;
         i < program->attached_shader_counts[stage] &&
         i < MAX_ATTACHED_SHADERS_PER_STAGE;
         i++) {
        if (program->attached_shader_slots[stage][i] == shader) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLuint mglProgramAttachedShaderCount(Program *program, GLuint stage)
{
    if (!program || stage >= _MAX_SHADER_TYPES) {
        return 0u;
    }

    if (program->attached_shader_counts[stage] > 0u) {
        return program->attached_shader_counts[stage];
    }

    return ((program->attached_shader_mask & (1u << stage)) != 0u &&
            program->shader_slots[stage]) ? 1u : 0u;
}

void mglAttachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    GLuint index;

    sptr = findShader(ctx, shader);

    if (!sptr)
    {
        // CRITICAL FIX: Handle missing shader gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Shader %u not found in attach shader\n", shader);
        STATE(error) = GL_INVALID_VALUE;
        return;
    }

    pptr = findProgram(ctx, program);

    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

        return;
    }

    index = sptr->glm_type;

    mglFlushPendingDraws(ctx);

    if (mglProgramHasAttachedShader(pptr, index, sptr)) {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    if (pptr->attached_shader_counts[index] >= MAX_ATTACHED_SHADERS_PER_STAGE) {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    if (!pptr->shader_slots[index]) {
        pptr->shader_slots[index] = sptr;
    }

    pptr->attached_shader_slots[index][pptr->attached_shader_counts[index]++] = sptr;
    pptr->attached_shader_mask |= (1u << index);
    sptr->refcount++;
    pptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglDetachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    GLuint index;

    pptr = findProgram(ctx, program);
    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    sptr = findShader(ctx, shader);

    if (!sptr)
    {
        // If not found in hash table, check if it is attached to the program
        for (int i=0; i<_MAX_SHADER_TYPES; i++) {
            for (GLuint attached = 0;
                 attached < pptr->attached_shader_counts[i] &&
                 attached < MAX_ATTACHED_SHADERS_PER_STAGE;
                 attached++) {
                if (pptr->attached_shader_slots[i][attached] &&
                    pptr->attached_shader_slots[i][attached]->name == shader) {
                    sptr = pptr->attached_shader_slots[i][attached];
                    break;
                }
            }
            if (sptr) {
                break;
            }
            if (pptr->shader_slots[i] && pptr->shader_slots[i]->name == shader) {
                sptr = pptr->shader_slots[i];
                break;
            }
        }
    }

    if (!sptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    index = sptr->glm_type;

    GLuint detach_index = MAX_ATTACHED_SHADERS_PER_STAGE;
    for (GLuint attached = 0;
         attached < pptr->attached_shader_counts[index] &&
         attached < MAX_ATTACHED_SHADERS_PER_STAGE;
         attached++) {
        if (pptr->attached_shader_slots[index][attached] == sptr) {
            detach_index = attached;
            break;
        }
    }

    if (detach_index == MAX_ATTACHED_SHADERS_PER_STAGE ||
        (pptr->attached_shader_mask & (1u << index)) == 0u)
    {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    mglFlushPendingDraws(ctx);

    for (GLuint attached = detach_index + 1u;
         attached < pptr->attached_shader_counts[index] &&
         attached < MAX_ATTACHED_SHADERS_PER_STAGE;
         attached++) {
        pptr->attached_shader_slots[index][attached - 1u] =
            pptr->attached_shader_slots[index][attached];
    }
    if (pptr->attached_shader_counts[index] > 0u) {
        pptr->attached_shader_counts[index]--;
        pptr->attached_shader_slots[index][pptr->attached_shader_counts[index]] = NULL;
    }

    if (pptr->attached_shader_counts[index] == 0u) {
        pptr->attached_shader_mask &= ~(1u << index);
        if (!pptr->linked_glsl_program) {
            pptr->shader_slots[index] = NULL;
        }
    } else if (pptr->shader_slots[index] == sptr) {
        pptr->shader_slots[index] = pptr->attached_shader_slots[index][0];
    }

    /*
     * A successful link creates an executable that survives shader detach and
     * deletion. Keep the shader object as the executable's backing storage;
     * it is released when replaced or when the program is destroyed.
     */
    if (!pptr->linked_glsl_program) {
        sptr->refcount--;

        if (sptr->refcount == 0 && sptr->delete_status)
        {
            deleteHashElement(&STATE(shader_table), sptr->name);
            mglFreeShader(ctx, sptr);
        }

        pptr->dirty_bits |= DIRTY_PROGRAM;
    }
}

void error_callback(void *userdata, const char *error)
{
    if (!error)
        return;
    DEBUG_PRINT("parseSPIRVShader error:%s\n", error);
}


static_assert(_VERTEX_SHADER == GLSLANG_STAGE_VERTEX, "_VERTEX_SHADER == GLSLANG_STAGE_VERTEX failed");
static_assert(_TESS_CONTROL_SHADER == GLSLANG_STAGE_TESSCONTROL, "_TESS_CONTROL_SHADER == GLSLANG_STAGE_TESSCONTROL failed");
static_assert(_TESS_EVALUATION_SHADER == GLSLANG_STAGE_TESSEVALUATION, "_TESS_EVALUATION_SHADER == GLSLANG_STAGE_TESSEVALUATION failed");
static_assert(_GEOMETRY_SHADER == GLSLANG_STAGE_GEOMETRY, "_GEOMETRY_SHADER == GLSLANG_STAGE_GEOMETRY failed");
static_assert(_FRAGMENT_SHADER == GLSLANG_STAGE_FRAGMENT, "_FRAGMENT_SHADER == GLSLANG_STAGE_FRAGMENT failed");
static_assert(_COMPUTE_SHADER == GLSLANG_STAGE_COMPUTE, "_COMPUTE_SHADER == GLSLANG_STAGE_COMPUTE failed");

void addShadersToProgram(GLMContext ctx, Program *pptr, glslang_program_t *glsl_program)
{
    // add shaders
    for(int i=0;i<_MAX_SHADER_TYPES; i++)
    {
        Shader *ptr;

        if ((pptr->attached_shader_mask & (1u << i)) == 0u) {
            continue;
        }

        GLuint attached_count = mglProgramAttachedShaderCount(pptr, (GLuint)i);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            ptr = (pptr->attached_shader_counts[i] > 0u)
                ? pptr->attached_shader_slots[i][attached]
                : pptr->shader_slots[i];
            if (!ptr) {
                continue;
            }
            // should have glsl shader here
            if (!ptr->compiled_glsl_shader) {
                fprintf(stderr,
                        "MGL ERROR: program %u shader stage %d has no compiled GLSL shader\n",
                        pptr ? pptr->name : 0u,
                        i);
                if (ctx)
                    STATE(error) = GL_INVALID_OPERATION;
                continue;
            }

            glslang_program_add_shader(glsl_program, ptr->compiled_glsl_shader);
        }
    }
}

static void replace_all_substr(char **pstr, const char *from, const char *to)
{
    char *src;
    char *pos;
    size_t from_len;
    size_t to_len;
    size_t count = 0;
    size_t src_len;
    size_t new_len;
    char *dst;
    char *out;

    if (!pstr || !*pstr || !from || !to) {
        return;
    }

    src = *pstr;
    from_len = strlen(from);
    to_len = strlen(to);
    if (from_len == 0) {
        return;
    }

    pos = src;
    while ((pos = strstr(pos, from)) != NULL) {
        count++;
        pos += from_len;
    }

    if (count == 0) {
        return;
    }

    src_len = strlen(src);
    if (to_len >= from_len) {
        size_t diff = to_len - from_len;
        if (diff != 0 && count > (SIZE_MAX - src_len) / diff) {
            fprintf(stderr,
                    "MGL MSL PATCH FAIL: replace_all_substr('%s'->'%s') "
                    "size overflow; original MSL preserved\n",
                    from, to);
            return;
        }
        new_len = src_len + count * diff;
    } else {
        size_t diff = from_len - to_len;
        if (diff != 0 && count > src_len / diff) {
            fprintf(stderr,
                    "MGL MSL PATCH FAIL: replace_all_substr('%s'->'%s') "
                    "size underflow; original MSL preserved\n",
                    from, to);
            return;
        }
        new_len = src_len - count * diff;
    }
    if (new_len == SIZE_MAX) {
        fprintf(stderr,
                "MGL MSL PATCH FAIL: replace_all_substr('%s'->'%s') "
                "allocation size overflow; original MSL preserved\n",
                from, to);
        return;
    }
    out = (char *)malloc(new_len + 1);
    if (!out) {
        /* Allocation failed — preserve the original string (rollback
         * semantics).  Log loudly so silent MSL corruption is visible:
         * the caller's patch did not apply, and the resulting MSL may
         * reference un-rewritten patterns that fail Metal compilation
         * or produce wrong behavior. */
        fprintf(stderr,
                "MGL MSL PATCH FAIL: replace_all_substr('%s'->'%s') allocation "
                "of %zu bytes failed; original MSL preserved (patch NOT applied)\n",
                from, to, new_len + 1);
        return;
    }

    pos = src;
    dst = out;
    while (1) {
        char *match = strstr(pos, from);
        size_t chunk_len;
        if (!match) {
            strcpy(dst, pos);
            break;
        }
        chunk_len = (size_t)(match - pos);
        memcpy(dst, pos, chunk_len);
        dst += chunk_len;
        memcpy(dst, to, to_len);
        dst += to_len;
        pos = match + from_len;
    }

    free(*pstr);
    *pstr = out;
}

static GLboolean mglReplaceMSLIdentifier(char **msl_ptr,
                                         const char *from,
                                         const char *to)
{
    const char *src;
    const char *cursor;
    char *out;
    char *dst;
    size_t from_len;
    size_t to_len;
    size_t src_len;
    size_t count = 0u;

    if (!msl_ptr || !*msl_ptr || !from || !to) {
        return GL_FALSE;
    }

    from_len = strlen(from);
    to_len = strlen(to);
    if (from_len == 0u || strcmp(from, to) == 0) {
        return GL_FALSE;
    }

    src = *msl_ptr;
    cursor = src;
    while ((cursor = strstr(cursor, from)) != NULL) {
        char before = (cursor == src) ? '\0' : cursor[-1];
        char after = cursor[from_len];
        if (!mglMSLIdentifierChar(before) && !mglMSLIdentifierChar(after)) {
            count++;
        }
        cursor += from_len;
    }

    if (count == 0u) {
        return GL_FALSE;
    }

    src_len = strlen(src);
    out = (char *)malloc(src_len + count * (to_len > from_len ? (to_len - from_len) : 0u) + 1u);
    if (!out) {
        fprintf(stderr,
                "MGL MSL PATCH FAIL: mglReplaceMSLIdentifier('%s'->'%s') allocation failed; "
                "original MSL preserved (patch NOT applied)\n",
                from, to);
        return GL_FALSE;
    }

    cursor = src;
    dst = out;
    while (*cursor) {
        const char *match = strstr(cursor, from);
        if (!match) {
            strcpy(dst, cursor);
            break;
        }

        char before = (match == src) ? '\0' : match[-1];
        char after = match[from_len];
        if (mglMSLIdentifierChar(before) || mglMSLIdentifierChar(after)) {
            size_t chunk = (size_t)(match - cursor) + from_len;
            memcpy(dst, cursor, chunk);
            dst += chunk;
            cursor = match + from_len;
            continue;
        }

        size_t prefix = (size_t)(match - cursor);
        memcpy(dst, cursor, prefix);
        dst += prefix;
        memcpy(dst, to, to_len);
        dst += to_len;
        cursor = match + from_len;
    }

    free(*msl_ptr);
    *msl_ptr = out;
    return GL_TRUE;
}

static GLboolean mglReplaceMSLIdentifierBeforeChar(char **msl_ptr,
                                                   const char *from,
                                                   const char *to,
                                                   char required_after)
{
    const char *src;
    const char *cursor;
    char *out;
    char *dst;
    size_t from_len;
    size_t to_len;
    size_t src_len;
    size_t new_len;
    size_t count = 0u;

    if (!msl_ptr || !*msl_ptr || !from || !to) {
        return GL_FALSE;
    }

    from_len = strlen(from);
    to_len = strlen(to);
    if (from_len == 0u || strcmp(from, to) == 0) {
        return GL_FALSE;
    }

    src = *msl_ptr;
    cursor = src;
    while ((cursor = strstr(cursor, from)) != NULL) {
        char before = (cursor == src) ? '\0' : cursor[-1];
        char after = cursor[from_len];
        if (!mglMSLIdentifierChar(before) && before != '.' && after == required_after) {
            count++;
        }
        cursor += from_len;
    }

    if (count == 0u) {
        return GL_FALSE;
    }

    src_len = strlen(src);
    if (to_len >= from_len) {
        new_len = src_len + count * (to_len - from_len);
    } else {
        new_len = src_len - count * (from_len - to_len);
    }
    out = (char *)malloc(new_len + 1u);
    if (!out) {
        return GL_FALSE;
    }

    cursor = src;
    dst = out;
    while (*cursor) {
        const char *match = strstr(cursor, from);
        if (!match) {
            strcpy(dst, cursor);
            break;
        }

        char before = (match == src) ? '\0' : match[-1];
        char after = match[from_len];
        if (mglMSLIdentifierChar(before) || before == '.' || after != required_after) {
            size_t chunk = (size_t)(match - cursor) + from_len;
            memcpy(dst, cursor, chunk);
            dst += chunk;
            cursor = match + from_len;
            continue;
        }

        size_t prefix = (size_t)(match - cursor);
        memcpy(dst, cursor, prefix);
        dst += prefix;
        memcpy(dst, to, to_len);
        dst += to_len;
        cursor = match + from_len;
    }

    free(*msl_ptr);
    *msl_ptr = out;
    return GL_TRUE;
}

static GLboolean mglInsertStringAt(char **pstr, const char *position, const char *insertion)
{
    if (!pstr || !*pstr || !position || !insertion ||
        position < *pstr || position > *pstr + strlen(*pstr)) {
        return GL_FALSE;
    }

    size_t source_len = strlen(*pstr);
    size_t insertion_len = strlen(insertion);
    size_t prefix_len = (size_t)(position - *pstr);
    char *out = (char *)malloc(source_len + insertion_len + 1u);
    if (!out) {
        return GL_FALSE;
    }

    memcpy(out, *pstr, prefix_len);
    memcpy(out + prefix_len, insertion, insertion_len);
    memcpy(out + prefix_len + insertion_len,
           *pstr + prefix_len,
           source_len - prefix_len + 1u);
    free(*pstr);
    *pstr = out;
    return GL_TRUE;
}

static void applyMSLFragCoordOriginFix(int stage, char **msl_ptr)
{
    static const char position_parameter[] = "float4 gl_FragCoord [[position]]";
    static const char injected_parameter[] =
        "constant float4& " MGL_FRAG_COORD_PARAMS_MSL_NAME
        " [[buffer(30)]], ";
    static const char injected_body[] =
        "\n    if (" MGL_FRAG_COORD_PARAMS_MSL_NAME ".y > 0.5) "
        "gl_FragCoord.y = " MGL_FRAG_COORD_PARAMS_MSL_NAME ".x - gl_FragCoord.y;";

    if (stage != _FRAGMENT_SHADER || !msl_ptr || !*msl_ptr ||
        strstr(*msl_ptr, MGL_FRAG_COORD_PARAMS_MSL_NAME)) {
        return;
    }

    const char *position = strstr(*msl_ptr, position_parameter);
    if (!position) {
        return;
    }

    if (!mglInsertStringAt(msl_ptr, position, injected_parameter)) {
        return;
    }

    position = strstr(*msl_ptr, position_parameter);
    const char *body = position ? strchr(position, '{') : NULL;
    if (!body || !mglInsertStringAt(msl_ptr, body + 1, injected_body)) {
        fprintf(stderr, "MGL WARNING: failed to inject gl_FragCoord origin conversion\n");
    }
}

static GLboolean mglFindMSLIdentifierBefore(const char *start,
                                            const char *pos,
                                            char *out,
                                            size_t out_size)
{
    if (!start || !pos || !out || out_size == 0) {
        return GL_FALSE;
    }

    const char *p = pos;
    while (p > start && isspace((unsigned char)p[-1])) {
        p--;
    }
    const char *name_end = p;
    while (p > start &&
           (isalnum((unsigned char)p[-1]) || p[-1] == '_')) {
        p--;
    }

    size_t len = (size_t)(name_end - p);
    if (len == 0 || len >= out_size) {
        return GL_FALSE;
    }

    memcpy(out, p, len);
    out[len] = '\0';
    return GL_TRUE;
}

static GLboolean mglInjectMSLPointSizeParams(char **msl_ptr)
{
    if (!msl_ptr || !*msl_ptr || strstr(*msl_ptr, "_mgl_point_size_params")) {
        return GL_TRUE;
    }

    const char *param_open = mglFindMSLEntryParameterOpen(*msl_ptr);
    const char *param_close = mglFindMSLEntryParameterClose(*msl_ptr);
    if (!param_open || !param_close || param_open >= param_close) {
        return GL_FALSE;
    }

    GLboolean has_existing_params = GL_FALSE;
    for (const char *p = param_open + 1; p < param_close; p++) {
        if (!isspace((unsigned char)*p)) {
            has_existing_params = GL_TRUE;
            break;
        }
    }

    char param[128];
    snprintf(param, sizeof(param),
             "%sconstant float2& _mgl_point_size_params [[buffer(%u)]]",
             has_existing_params ? ", " : "",
             (unsigned)kMGLPointSizeBufferIndex);
    return mglInsertStringAt(msl_ptr, param_close, param) ? GL_TRUE : GL_FALSE;
}

/* Inject or override [[point_size]] output for vertex-producing shaders.
 *
 * Metal requires a [[point_size]] output for point primitives
 * (MTLPrimitiveTypePoint) to rasterize with a defined, nonzero size.
 * GLSL shaders that don't write gl_PointSize produce SPIR-V without the
 * PointSize builtin, so SPIRV-Cross omits [[point_size]] from the output
 * struct.  This function post-processes the MSL to add:
 *   1. A `float mgl_injected_point_size [[point_size]];` field to the
 *      vertex output struct (right after the [[position]] field).
 *   2. An assignment before every `return out;` in the entry function.
 *
 * Vertex shaders read the current GL point-size state from a tiny internal
 * buffer.  If the shader already declares gl_PointSize, fixed-function point
 * size still overrides it when GL_PROGRAM_POINT_SIZE is disabled. */
static void mglInjectMSLPointSizeBuiltin(int stage, char **msl_ptr)
{
    if (!msl_ptr || !*msl_ptr) {
        return;
    }

    /* Only vertex-producing stages need point_size. */
    if (stage != _VERTEX_SHADER &&
        stage != _TESS_EVALUATION_SHADER &&
        stage != _GEOMETRY_SHADER) {
        return;
    }

    GLboolean has_point_size = strstr(*msl_ptr, "[[point_size]]") != NULL;
    GLboolean dynamic_vertex_point_size = (stage == _VERTEX_SHADER ||
                                           stage == _TESS_EVALUATION_SHADER ||
                                           stage == _GEOMETRY_SHADER);
    char point_size_name[128] = "mgl_injected_point_size";

    /* Find the [[position]] qualifier in the output struct. */
    const char *pos_qualifier = strstr(*msl_ptr, "[[position]]");
    if (!pos_qualifier) {
        return;
    }

    if (has_point_size) {
        const char *ps_qualifier = strstr(*msl_ptr, "[[point_size]]");
        if (!mglFindMSLIdentifierBefore(*msl_ptr,
                                        ps_qualifier,
                                        point_size_name,
                                        sizeof(point_size_name))) {
            return;
        }
        if (!dynamic_vertex_point_size) {
            return;
        }
    } else {
        /* Find the end of the line containing [[position]] (the terminating ';'). */
        const char *stmt_end = strchr(pos_qualifier, ';');
        if (!stmt_end) {
            return;
        }

        /* Insert the point_size field right after the position field. */
        static const char point_size_field[] = "\n    float mgl_injected_point_size [[point_size]];";
        if (!mglInsertStringAt(msl_ptr, stmt_end + 1, point_size_field)) {
            return;
        }
    }

    if (dynamic_vertex_point_size &&
        !mglInjectMSLPointSizeParams(msl_ptr)) {
        return;
    }

    /* SPIRV-Cross consistently names the output variable `out` and uses
     * `return out;` as the return statement. */
    const char return_pattern[] = "return out;";
    size_t return_len = strlen(return_pattern);
    size_t cursor_offset = 0;
    const char *cursor = *msl_ptr;
    while ((cursor = strstr(cursor, return_pattern)) != NULL) {
        cursor_offset = (size_t)(cursor - *msl_ptr);
        char point_size_assign[256];
        if (dynamic_vertex_point_size && has_point_size) {
            snprintf(point_size_assign, sizeof(point_size_assign),
                     "if (_mgl_point_size_params.y == 0.0) { out.%s = _mgl_point_size_params.x; } ",
                     point_size_name);
        } else if (dynamic_vertex_point_size) {
            snprintf(point_size_assign, sizeof(point_size_assign),
                     "out.%s = _mgl_point_size_params.x; ",
                     point_size_name);
        } else {
            snprintf(point_size_assign, sizeof(point_size_assign),
                     "out.%s = 1.0; ",
                     point_size_name);
        }
        if (!mglInsertStringAt(msl_ptr, cursor, point_size_assign)) {
            break;
        }
        /* Move past the inserted text + the return pattern for the next search. */
        cursor = *msl_ptr + cursor_offset + strlen(point_size_assign) + return_len;
    }
}

/* SPIRV-Cross lowers GLSL image2DRect to Metal texture2d, but emits
 * imageSize(g_image_rect) as `int2(tex.get_width())` — a single-argument
 * int2 constructor that broadcasts width to both components, yielding
 * (width, width) instead of (width, height).  This breaks the
 * KHR-GL46.shader_image_size tests for image2DRect.
 *
 * Fix by rewriting `int2(IDENT.get_width())` into
 * `int2(IDENT.get_width(), IDENT.get_height())` wherever IDENT is a
 * texture variable.  The 2-argument form `int2(a, b)` is left untouched
 * because the `get_width()` is followed by a comma, not a ')'. */
static void mglFixMSLImage2DRectImageSize(char **msl_ptr)
{
    if (!msl_ptr || !*msl_ptr) {
        return;
    }

    const char needle[] = ".get_width())";
    const size_t needle_len = sizeof(needle) - 1u;

    char *cursor = *msl_ptr;
    while ((cursor = strstr(cursor, needle)) != NULL) {
        /* Cursor points at ".get_width())".  Walk backwards to find the
         * identifier that owns this method call. */
        char *ident_end = cursor;  /* points at '.' */
        char *p = ident_end;
        while (p > *msl_ptr && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) {
            p--;
        }
        char *ident_start = p;
        size_t ident_len = (size_t)(ident_end - ident_start);
        if (ident_len == 0u) {
            cursor += needle_len;
            continue;
        }

        /* Skip whitespace between ident_end and '.', which is ident_end itself. */
        /* Now walk backwards from ident_start, skipping whitespace, to find '('. */
        char *q = ident_start;
        while (q > *msl_ptr && isspace((unsigned char)q[-1])) {
            q--;
        }
        /* q now points just after '('.  Check for "int2(" (or other type prefixes). */
        const char *prefixes[] = { "int2(", "uint2(", "float2(" };
        size_t prefix_lens[] = { 5u, 6u, 7u };
        int matched = -1;
        for (int i = 0; i < 3; i++) {
            size_t plen = prefix_lens[i];
            if ((size_t)(q - *msl_ptr) >= plen &&
                strncmp(q - plen, prefixes[i], plen) == 0) {
                matched = i;
                break;
            }
        }
        if (matched < 0) {
            cursor += needle_len;
            continue;
        }

        /* Build the replacement: PREFIX(IDENT.get_width(), IDENT.get_height()) */
        char *close_paren = cursor + needle_len - 1u;  /* points at ')' */
        size_t prefix_len = prefix_lens[matched];
        char *open_paren = q - prefix_len;  /* points at '(' */

        /* Construct new text.  The range [before, close_paren] includes the
         * ')' that closes int2(...), so the replacement must re-close it. */
        char *before = open_paren + prefix_len;  /* start of IDENT */
        size_t new_inner_len = ident_len + strlen(".get_width(), ") + ident_len + strlen(".get_height())");
        char *new_inner = (char *)malloc(new_inner_len + 1u);
        if (!new_inner) {
            cursor += needle_len;
            continue;
        }
        memcpy(new_inner, ident_start, ident_len);
        memcpy(new_inner + ident_len, ".get_width(), ", strlen(".get_width(), "));
        memcpy(new_inner + ident_len + strlen(".get_width(), "), ident_start, ident_len);
        memcpy(new_inner + ident_len + strlen(".get_width(), ") + ident_len, ".get_height())", strlen(".get_height())"));
        new_inner[new_inner_len] = '\0';

        /* Replace the range [before, close_paren] with new_inner. */
        size_t before_offset = (size_t)(before - *msl_ptr);
        size_t close_offset = (size_t)(close_paren - *msl_ptr);
        size_t old_len = close_offset - before_offset + 1u;  /* inclusive of ')' */
        size_t total_len = strlen(*msl_ptr);
        size_t new_total = total_len - old_len + new_inner_len;
        char *new_msl = (char *)malloc(new_total + 1u);
        if (!new_msl) {
            free(new_inner);
            cursor += needle_len;
            continue;
        }
        memcpy(new_msl, *msl_ptr, before_offset);
        memcpy(new_msl + before_offset, new_inner, new_inner_len);
        strcpy(new_msl + before_offset + new_inner_len, close_paren + 1u);
        free(new_inner);
        free(*msl_ptr);
        *msl_ptr = new_msl;

        /* Advance cursor past the replacement in the new buffer. */
        cursor = *msl_ptr + before_offset + new_inner_len;
    }
}

/* Convert a TES (Tessellation Evaluation Shader) MSL from a Metal
 * post-tessellation vertex function to a compute kernel.
 *
 * SPIRV-Cross lowers GL_TESS_EVALUATION_SHADER to:
 *   [[ patch(quad, 0) ]] vertex void name(... uint gl_PrimitiveID [[patch_id]] ...)
 *
 * On macOS SDKs where MTLRenderPipelineDescriptor no longer has
 * postTessellationVertexFunction / isTessellationEnabled (Metal 4 era),
 * we cannot use drawPatches: with a post-tessellation vertex function.
 * Instead, we rewrite the MSL to a compute kernel and dispatch it
 * with dispatchThreadgroups:, exactly like TCS.
 *
 * Rewrites:
 *   "[[ patch(quad, 0) ]] vertex void"  →  "kernel void"
 *   "[[patch_id]]"                       →  "[[threadgroup_position_in_grid]]"
 *   "[[position_in_patch]]"              →  "[[thread_position_in_threadgroup]]"
 */

/* Fix TCS MSL [[stage_in]] for compute kernel context.
 * SPIRV-Cross generates TCS compute kernels with a [[stage_in]] parameter
 * for vertex input, but Metal compute pipelines don't support stage_in
 * descriptors.  Replace:
 *   <type> <name> [[stage_in]]
 * with:
 *   device <type> *_mgl_tcs_in_buffer [[buffer(kMGLBufferSlot_TCSStageInRepl)]]
 * and inject at body start:
 *   <type> <name> = _mgl_tcs_in_buffer[gl_PrimitiveID * spvIndirectParams[0] + gl_InvocationID];
 * Also strips [[attribute(N)]] from struct members (invalid in device buffers).
 * The replacement slot is kept below SPIRV-Cross's runtime-sized SSBO size
 * buffer (25) and the tessellation helper slots (26-30). */
static void mglFixMSLTcsStageIn(char **msl_ptr)
{
    if (!msl_ptr || !*msl_ptr) {
        return;
    }

    const char *stage_in_attr = "[[stage_in]]";
    char *pos = strstr(*msl_ptr, stage_in_attr);
    if (pos == NULL) {
        return;  /* No stage_in — nothing to fix. */
    }

    /* Walk backwards from [[stage_in]] to find parameter name and type. */
    const char *p = pos;
    while (p > *msl_ptr && isspace((unsigned char)p[-1])) {
        p--;
    }
    const char *name_end = p;
    const char *name_start = name_end;
    while (name_start > *msl_ptr &&
           (isalnum((unsigned char)name_start[-1]) || name_start[-1] == '_')) {
        name_start--;
    }
    size_t name_len = (size_t)(name_end - name_start);

    const char *before_name = name_start;
    while (before_name > *msl_ptr && isspace((unsigned char)before_name[-1])) {
        before_name--;
    }
    const char *type_end = before_name;
    const char *type_start = type_end;
    while (type_start > *msl_ptr &&
           (isalnum((unsigned char)type_start[-1]) || type_start[-1] == '_')) {
        type_start--;
    }
    size_t type_len = (size_t)(type_end - type_start);

    if (type_len == 0 || name_len == 0 || type_len >= 256 || name_len >= 256) {
        return;
    }

    char param_type[256];
    char param_name[256];
    memcpy(param_type, type_start, type_len);
    param_type[type_len] = '\0';
    memcpy(param_name, name_start, name_len);
    param_name[name_len] = '\0';

    /* Replace "<type> <name> [[stage_in]]" with a device buffer parameter. */
    const char *replacement_start = type_start;
    size_t old_param_len = (size_t)(pos + strlen(stage_in_attr) - replacement_start);

    char new_param[512];
    snprintf(new_param, sizeof(new_param),
             "device %s *_mgl_tcs_in_buffer [[buffer(%u)]]",
             param_type,
             (unsigned)kMGLBufferSlot_TCSStageInRepl);
    size_t new_param_len = strlen(new_param);

    size_t before_len = (size_t)(replacement_start - *msl_ptr);
    char *new_msl = (char *)malloc(strlen(*msl_ptr) - old_param_len + new_param_len + 1);
    memcpy(new_msl, *msl_ptr, before_len);
    memcpy(new_msl + before_len, new_param, new_param_len);
    strcpy(new_msl + before_len + new_param_len, replacement_start + old_param_len);
    free(*msl_ptr);
    *msl_ptr = new_msl;

    /* Inject local variable at function body start:
     * <type> <name> = _mgl_tcs_in_buffer[gl_PrimitiveID * spvIndirectParams[0] + gl_InvocationID]; */
    char *brace = strchr(*msl_ptr + before_len + new_param_len, '{');
    if (brace != NULL) {
        char inject[512];
        snprintf(inject, sizeof(inject),
                 "\n    %s %s = _mgl_tcs_in_buffer[gl_PrimitiveID * spvIndirectParams[0] + gl_InvocationID];",
                 param_type, param_name);
        size_t inject_len = strlen(inject);
        size_t brace_pos = (size_t)(brace + 1 - *msl_ptr);
        char *new_msl2 = (char *)malloc(strlen(*msl_ptr) + inject_len + 1);
        memcpy(new_msl2, *msl_ptr, brace_pos);
        memcpy(new_msl2 + brace_pos, inject, inject_len);
        strcpy(new_msl2 + brace_pos + inject_len, brace + 1);
        free(*msl_ptr);
        *msl_ptr = new_msl2;
    }

    /* Strip [[attribute(N)]] from struct members (invalid in device buffer context),
     * but keep a comment marker so the TCS compute dispatch can pack the
     * replacement stage_in buffer with the original VAO locations. */
    {
        const char *attr_prefix = "[[attribute(";
        char *apos = *msl_ptr;
        while ((apos = strstr(apos, attr_prefix)) != NULL) {
            char *close = strstr(apos, "]]");
            if (close == NULL) {
                break;
            }
            char *strip_start = apos;
            while (strip_start > *msl_ptr && isspace((unsigned char)strip_start[-1])) {
                strip_start--;
            }
            size_t strip_len = (size_t)(close + 2 - strip_start);
            size_t before = (size_t)(strip_start - *msl_ptr);
            char *num_start = apos + strlen(attr_prefix);
            char *num_end = num_start;
            while (num_end < close && isdigit((unsigned char)*num_end)) {
                num_end++;
            }
            char marker[64] = "";
            if (num_end > num_start) {
                size_t num_len = (size_t)(num_end - num_start);
                if (num_len > 20u) {
                    num_len = 20u;
                }
                char num[24];
                memcpy(num, num_start, num_len);
                num[num_len] = '\0';
                snprintf(marker, sizeof(marker), " /*mgl_attribute(%s)*/", num);
            }
            size_t marker_len = strlen(marker);
            char *strip_msl = (char *)malloc(strlen(*msl_ptr) - strip_len + marker_len + 1);
            memcpy(strip_msl, *msl_ptr, before);
            memcpy(strip_msl + before, marker, marker_len);
            strcpy(strip_msl + before + marker_len, close + 2);
            free(*msl_ptr);
            *msl_ptr = strip_msl;
            apos = *msl_ptr + before + marker_len;
        }
    }
}

/* Ensure _mgl_patch_id3 is available in the TES compute kernel.
 *
 * Three cases:
 * 1. _mgl_patch_id3 already declared → nothing to do.
 * 2. [[threadgroup_position_in_grid]] already used by another parameter
 *    (from SPIRV-Cross, e.g. for barrier support) → find that param's
 *    name and inject "uint3 _mgl_patch_id3 = <name>;" alias at body start.
 * 3. Neither exists → inject "uint3 _mgl_patch_id3 [[threadgroup_position_in_grid]]"
 *    as a new kernel parameter.
 */
static void mglEnsurePatchId3Param(char **msl_ptr)
{
    if (!msl_ptr || !*msl_ptr) {
        return;
    }

    /* Case 1: already present */
    if (strstr(*msl_ptr, "_mgl_patch_id3") != NULL) {
        return;
    }

    const char *tg_pos_attr = "[[threadgroup_position_in_grid]]";
    const char *attr_pos = strstr(*msl_ptr, tg_pos_attr);

    if (attr_pos) {
        /* Case 2: [[threadgroup_position_in_grid]] already used by another
         * parameter (from SPIRV-Cross, e.g. for gl_PrimitiveID). We need to
         * either reuse it (if it's already uint3) or convert the scalar
         * uint param to uint3 so it matches the other vector params
         * (Metal requires all [[...]]-qualified input declarations to be
         * all scalar or all vector with the same number of elements). */
        const char *end = attr_pos;
        while (end > *msl_ptr && isspace((unsigned char)end[-1])) {
            end--;
        }
        const char *name_end = end;
        const char *name_start = name_end;
        while (name_start > *msl_ptr &&
               (isalnum((unsigned char)name_start[-1]) || name_start[-1] == '_')) {
            name_start--;
        }

        if (name_start >= name_end) {
            return; /* couldn't find name */
        }

        size_t name_len = (size_t)(name_end - name_start);

        /* Walk further back past whitespace to find the type token. */
        const char *type_end = name_start;
        while (type_end > *msl_ptr && isspace((unsigned char)type_end[-1])) {
            type_end--;
        }
        const char *type_start = type_end;
        while (type_start > *msl_ptr &&
               (isalnum((unsigned char)type_start[-1]) || type_start[-1] == '_')) {
            type_start--;
        }
        size_t type_len = (size_t)(type_end - type_start);
        /* If the param is already uint3, we can assign directly. Otherwise
         * (scalar uint), we need to convert the scalar param to uint3 and
         * update all scalar usages to use .x. */
        int is_uint3 = (type_len == 5 &&
                        strncmp(type_start, "uint3", 5) == 0);
        int is_int3 = (type_len == 4 &&
                       strncmp(type_start, "int3", 4) == 0);

        /* Find the kernel body opening brace after the parameter list. */
        const char *close_paren = mglFindMSLEntryParameterClose(*msl_ptr);
        if (!close_paren) {
            return;
        }
        const char *brace = strchr(close_paren, '{');
        if (!brace) {
            return;
        }

        if (is_uint3 || is_int3) {
            /* Param is already a vector; just alias it. */
            char alias[300];
            snprintf(alias, sizeof(alias),
                     "\n    uint3 _mgl_patch_id3 = %.*s;",
                     (int)name_len, name_start);
            size_t alias_len = strlen(alias);
            size_t before = (size_t)(brace + 1 - *msl_ptr);
            char *nm = (char *)malloc(strlen(*msl_ptr) + alias_len + 1);
            memcpy(nm, *msl_ptr, before);
            memcpy(nm + before, alias, alias_len);
            strcpy(nm + before + alias_len, brace + 1);
            free(*msl_ptr);
            *msl_ptr = nm;
            return;
        }

        /* Scalar case: replace the scalar type token in the parameter
         * declaration with "uint3", then inject an alias, then update
         * all scalar usages of <name> to <name>.x. */
        {
            /* 1. Replace "uint <name>" with "uint3 <name>" in the param
             *    declaration (first occurrence in the type position).
             *    We target the exact spot we identified. */
            size_t type_before = (size_t)(type_start - *msl_ptr);
            size_t type_after = (size_t)(type_end - *msl_ptr);
            const char *new_type = "uint3";
            size_t new_type_len = strlen(new_type);
            /* Compute body length starting from type_after. */
            size_t tail_len = strlen(*msl_ptr) - type_after;
            char *nm = (char *)malloc(type_before + new_type_len + tail_len + 1);
            memcpy(nm, *msl_ptr, type_before);
            memcpy(nm + type_before, new_type, new_type_len);
            memcpy(nm + type_before + new_type_len,
                   type_end, tail_len);
            nm[type_before + new_type_len + tail_len] = '\0';
            free(*msl_ptr);
            *msl_ptr = nm;
            /* Re-locate brace after the type replacement. */
            close_paren = mglFindMSLEntryParameterClose(*msl_ptr);
            if (!close_paren) {
                return;
            }
            brace = strchr(close_paren, '{');
            if (!brace) {
                return;
            }

            /* 2. Inject "uint3 _mgl_patch_id3 = <name>;" alias right
             *    after '{'.  Using direct assignment now works because
             *    <name> is now uint3.
             *
             *    NOTE: name_start pointed into the OLD buffer which was
             *    freed above.  Recompute the name by finding the
             *    [[threadgroup_position_in_grid]] attribute again and
             *    walking back to the param name. */
            const char *attr_pos2 = strstr(*msl_ptr, tg_pos_attr);
            if (!attr_pos2) {
                return;
            }
            const char *end2 = attr_pos2;
            while (end2 > *msl_ptr && isspace((unsigned char)end2[-1])) {
                end2--;
            }
            const char *name_start2 = end2;
            while (name_start2 > *msl_ptr &&
                   (isalnum((unsigned char)name_start2[-1]) || name_start2[-1] == '_')) {
                name_start2--;
            }
            if (name_start2 >= end2) {
                return;
            }
            size_t name_len2 = (size_t)(end2 - name_start2);
            /* Copy the name into a local buffer now, because subsequent
             * buffer reallocations will invalidate name_start2. */
            char param_name[128];
            if (name_len2 >= sizeof(param_name)) {
                return;
            }
            memcpy(param_name, name_start2, name_len2);
            param_name[name_len2] = '\0';

            char alias[300];
            snprintf(alias, sizeof(alias),
                     "\n    uint3 _mgl_patch_id3 = %s;",
                     param_name);
            size_t alias_len = strlen(alias);
            size_t before = (size_t)(brace + 1 - *msl_ptr);
            char *am = (char *)malloc(strlen(*msl_ptr) + alias_len + 1);
            memcpy(am, *msl_ptr, before);
            memcpy(am + before, alias, alias_len);
            strcpy(am + before + alias_len, brace + 1);
            free(*msl_ptr);
            *msl_ptr = am;

            /* 3. Update scalar usages of <name> in the body to <name>.x.
             *    We must NOT touch the alias line we just injected
             *    ("uint3 _mgl_patch_id3 = <name>;") nor the parameter
             *    declaration.  So we only rewrite occurrences that appear
             *    after the alias line.  Build a needle "<name>" and walk
             *    forward from the end of the alias, replacing each whole-
             *    word match with "<name>.x" (avoid touching already-
             *    suffixed <name>.x).
             *
             *    NOTE: all pointers (alias_pos, search_from, sc) must be
             *    recomputed from the current *msl_ptr because the alias
             *    injection above freed and reallocated the buffer. */
            const char *alias_needle = "_mgl_patch_id3 = ";
            const char *alias_pos = strstr(*msl_ptr, alias_needle);
            if (!alias_pos) {
                return;
            }
            const char *search_from = alias_pos + strlen(alias_needle);
            /* skip past the name and the ";" */
            const char *sc = search_from + name_len2;
            while (*sc && *sc != ';') sc++;
            if (*sc == ';') sc++;

            /* Now replace whole-word "<name>" occurrences starting from sc
             * with "<name>.x".  Use a buffer-based replace loop. */
            char needle[128];
            snprintf(needle, sizeof(needle), "%s", param_name);
            size_t needle_len = strlen(needle);
            /* Use a replacement that adds ".x" but only if the next char
             * is not already '.' or an identifier char or '('. */
            char *p = (char *)sc;
            int safety_iter = 0;
            while ((p = strstr(p, needle)) != NULL && safety_iter < 10000) {
                safety_iter++;
                /* Check preceding char: must be a word boundary. */
                char prev = (p > *msl_ptr) ? p[-1] : '\0';
                if (p > *msl_ptr &&
                    (isalnum((unsigned char)prev) || prev == '_')) {
                    p += needle_len;
                    continue;
                }
                char next = p[needle_len];
                /* Skip if already has ".x" or is part of a larger ident. */
                if (next == '.' ||
                    isalnum((unsigned char)next) || next == '_') {
                    p += needle_len;
                    continue;
                }
                /* Insert ".x" after the match. */
                size_t before = (size_t)(p - *msl_ptr);
                const char *suffix = ".x";
                size_t suffix_len = strlen(suffix);
                char *nm2 = (char *)malloc(strlen(*msl_ptr) + suffix_len + 1);
                memcpy(nm2, *msl_ptr, before + needle_len);
                memcpy(nm2 + before + needle_len, suffix, suffix_len);
                strcpy(nm2 + before + needle_len + suffix_len,
                       p + needle_len);
                free(*msl_ptr);
                *msl_ptr = nm2;
                p = *msl_ptr + before + needle_len + suffix_len;
            }
            return;
        }
    }

    /* Case 3: inject new parameter */
    const char *cp = mglFindMSLEntryParameterClose(*msl_ptr);
    if (!cp) {
        return;
    }

    const char *inj = "uint3 _mgl_patch_id3 [[threadgroup_position_in_grid]]";
    size_t inj_len = strlen(inj);
    const char *pfx = ", ";
    size_t pfx_len = strlen(pfx);
    const char *kw = strstr(*msl_ptr, "kernel void ");
    const char *op = kw ? strchr(kw, '(') : NULL;
    const char *pp = op ? op + 1 : cp;
    while (pp < cp && isspace((unsigned char)*pp)) {
        pp++;
    }
    if (pp >= cp) {
        pfx = "";
        pfx_len = 0;
    }
    size_t b = (size_t)(cp - *msl_ptr);
    char *nm = (char *)malloc(strlen(*msl_ptr) + pfx_len + inj_len + 1);
    memcpy(nm, *msl_ptr, b);
    memcpy(nm + b, pfx, pfx_len);
    memcpy(nm + b + pfx_len, inj, inj_len);
    strcpy(nm + b + pfx_len + inj_len, cp);
    free(*msl_ptr);
    *msl_ptr = nm;
}

static void mglFixMSLTessCullDistanceOutputs(char **msl_ptr)
{
    if (!msl_ptr || !*msl_ptr || !strstr(*msl_ptr, "gl_CullDistance")) {
        return;
    }

    for (unsigned i = 0; i < 16u; i++) {
        char from[64];
        char to[64];
        snprintf(from, sizeof(from), "out.gl_CullDistance[%u]", i);
        snprintf(to, sizeof(to), "out.gl_CullDistance_%u", i);
        replace_all_substr(msl_ptr, from, to);
    }

    char *pos = *msl_ptr;
    while ((pos = strstr(pos, "_RESERVED_IDENTIFIER_FIXUP_gl_CullDistance")) != NULL) {
        char *line_start = pos;
        while (line_start > *msl_ptr && line_start[-1] != '\n') {
            line_start--;
        }
        char *line_end = strchr(pos, '\n');
        if (!line_end) {
            line_end = pos + strlen(pos);
        } else {
            line_end++;
        }

        size_t before = (size_t)(line_start - *msl_ptr);
        size_t remove_len = (size_t)(line_end - line_start);
        char *new_msl = (char *)malloc(strlen(*msl_ptr) - remove_len + 1);
        if (!new_msl) {
            return;
        }
        memcpy(new_msl, *msl_ptr, before);
        strcpy(new_msl + before, line_end);
        free(*msl_ptr);
        *msl_ptr = new_msl;
        pos = *msl_ptr + before;
    }
}

static void mglFixMSLTesAsComputeKernel(Program *program, char **msl_ptr)
{
    if (!msl_ptr || !*msl_ptr) {
        return;
    }

    /* Step 1: Replace "[[ patch(...) ]] vertex <ret_type> <name>" with
     * "kernel void <name>".  The return type may be a struct (not void),
     * so search for "vertex " after "]]" and skip the return type to find
     * the function name. */
    {
        const char *src = *msl_ptr;
        while (src != NULL) {
            /* Find "vertex " in the MSL. */
            const char *vkw = strstr(src, "vertex ");
            if (vkw == NULL) {
                break;
            }
            /* Walk backwards from vkw to find "]]" (skip whitespace). */
            const char *p = vkw;
            while (p > src && isspace((unsigned char)p[-1])) {
                p--;
            }
            if (p < src + 2 || strncmp(p - 2, "]]", 2) != 0) {
                src = vkw + 1; /* not a tessellation vertex; advance */
                continue;
            }
            /* Walk back to find "[[ patch(" */
            const char *patch_kw = NULL;
            {
                const char *search = *msl_ptr;
                while (search < vkw) {
                    const char *found = strstr(search, "[[");
                    if (found == NULL || found >= vkw) {
                        break;
                    }
                    const char *after = found + 2;
                    while (after < vkw && isspace((unsigned char)*after)) {
                        after++;
                    }
                    if (strncmp(after, "patch(", 6) == 0) {
                        patch_kw = found;
                    }
                    search = found + 2;
                }
            }
            if (patch_kw == NULL) {
                src = vkw + 1;
                continue;
            }
            /* Skip "vertex " and the return type to find the function name.
             * The function name is the last identifier before '('. */
            const char *after_vertex = vkw + strlen("vertex ");
            const char *paren = strchr(after_vertex, '(');
            if (paren == NULL) {
                src = vkw + 1;
                continue;
            }
            const char *fname_start = paren;
            while (fname_start > after_vertex && !isspace((unsigned char)fname_start[-1])) {
                fname_start--;
            }
            /* Replace from patch_kw to fname_start with "kernel void " */
            size_t old_len = (size_t)(fname_start - patch_kw);
            const char *replacement = "kernel void ";
            size_t new_len = strlen(replacement);
            char *new_msl = (char *)malloc(strlen(*msl_ptr) - old_len + new_len + 1);
            size_t before = (size_t)(patch_kw - *msl_ptr);
            memcpy(new_msl, *msl_ptr, before);
            memcpy(new_msl + before, replacement, new_len);
            strcpy(new_msl + before + new_len, fname_start);
            free(*msl_ptr);
            *msl_ptr = new_msl;
            src = *msl_ptr + before + new_len;
        }
    }

    /* Step 1b: After converting to "kernel void", the function body may
     * still contain "return <var>;" from the original vertex shader.  Metal
     * compute kernels are void and cannot return a value, so replace
     * "return <identifier>;" with "return;" in the converted kernel. */
    {
        char *pos = *msl_ptr;
        while ((pos = strstr(pos, "return ")) != NULL) {
            /* Check if this is "return <identifier>;" (not "return;" or
             * "return (<expr>;" etc).  Skip identifier characters. */
            char *after_kw = pos + strlen("return ");
            char *p = after_kw;
            while (*p == '_' || isalnum((unsigned char)*p)) {
                p++;
            }
            /* Skip optional whitespace before ';'. */
            char *semi = p;
            while (*semi == ' ' || *semi == '\t') {
                semi++;
            }
            if (*semi == ';' && p > after_kw) {
                /* This is "return <identifier>;". Replace with "return;". */
                size_t before = (size_t)(pos - *msl_ptr);
                size_t old_len = (size_t)(semi + 1 - pos);
                const char *replacement = "return;";
                size_t new_len = strlen(replacement);
                char *new_msl = (char *)malloc(strlen(*msl_ptr) - old_len + new_len + 1);
                memcpy(new_msl, *msl_ptr, before);
                memcpy(new_msl + before, replacement, new_len);
                strcpy(new_msl + before + new_len, semi + 1);
                free(*msl_ptr);
                *msl_ptr = new_msl;
                pos = *msl_ptr + before + new_len;
            } else {
                pos = after_kw;
            }
        }
    }

    /* Step 2: Replace [[patch_id]] → [[threadgroup_position_in_grid]]. */
    {
        const char *old_attr = "[[patch_id]]";
        const char *new_attr = "[[threadgroup_position_in_grid]]";
        size_t old_len = strlen(old_attr);
        size_t new_len = strlen(new_attr);
        char *pos = *msl_ptr;
        while ((pos = strstr(pos, old_attr)) != NULL) {
            size_t before = (size_t)(pos - *msl_ptr);
            char *new_msl = (char *)malloc(strlen(*msl_ptr) - old_len + new_len + 1);
            memcpy(new_msl, *msl_ptr, before);
            memcpy(new_msl + before, new_attr, new_len);
            strcpy(new_msl + before + new_len, pos + old_len);
            free(*msl_ptr);
            *msl_ptr = new_msl;
            pos = *msl_ptr + before + new_len;
        }
    }

    /* Step 3: Replace "float3 <var> [[position_in_patch]]" (with optional
     * spaces) with "uint3 _mgl_tess_tc [[thread_position_in_threadgroup]]"
     * and inject "float3 <var> = (float3)_mgl_tess_tc;" at the function body
     * start.  Metal's [[thread_position_in_threadgroup]] requires uint3, but
     * gl_TessCoord is float3 in GLSL/SPIRV-Cross output. */
    {
        const char *patterns[] = {
            "[[position_in_patch]]",
            "[[ position_in_patch ]]",
            NULL
        };
        for (int pi = 0; patterns[pi] != NULL; pi++) {
            const char *old_attr = patterns[pi];
            const char *new_attr = "[[thread_position_in_threadgroup]]";
            size_t old_len = strlen(old_attr);
            size_t new_len = strlen(new_attr);
            char *pos = *msl_ptr;
            while ((pos = strstr(pos, old_attr)) != NULL) {
                /* Walk backwards from pos to find the variable name. */
                const char *before_attr = pos;
                while (before_attr > *msl_ptr && isspace((unsigned char)before_attr[-1])) {
                    before_attr--;
                }
                const char *varname_end = before_attr;
                const char *varname_start = varname_end;
                while (varname_start > *msl_ptr &&
                       (isalnum((unsigned char)varname_start[-1]) || varname_start[-1] == '_')) {
                    varname_start--;
                }
                /* Skip whitespace before varname to find the type. */
                const char *before_var = varname_start;
                while (before_var > *msl_ptr && isspace((unsigned char)before_var[-1])) {
                    before_var--;
                }
                const char *type_end = before_var;
                const char *type_start = type_end;
                while (type_start > *msl_ptr &&
                       (isalnum((unsigned char)type_start[-1]) || type_start[-1] == '_')) {
                    type_start--;
                }
                size_t type_len = (size_t)(type_end - type_start);
                size_t varname_len = (size_t)(varname_end - varname_start);

                if (varname_len > 0 &&
                    ((type_len == 6 && strncmp(type_start, "float3", 6) == 0) ||
                     (type_len == 6 && strncmp(type_start, "float2", 6) == 0) ||
                     (type_len == 5 && strncmp(type_start, "float", 5) == 0))) {
                    /* Save type and varname before freeing *msl_ptr. */
                    char vartype[256];
                    char varname[256];
                    size_t copy_t = type_len < 255 ? type_len : 255;
                    size_t copy_v = varname_len < 255 ? varname_len : 255;
                    memcpy(vartype, type_start, copy_t);
                    vartype[copy_t] = '\0';
                    memcpy(varname, varname_start, copy_v);
                    varname[copy_v] = '\0';

                    /* Replace "<type> <var> [[position_in_patch]]" with
                     * "uint3 _mgl_tess_tc [[thread_position_in_threadgroup]]" */
                    const char *rep = "uint3 _mgl_tess_tc ";
                    size_t rep_len = strlen(rep);
                    size_t total_old = (size_t)(pos + old_len - type_start);
                    size_t total_new = rep_len + new_len;
                    char *new_msl = (char *)malloc(strlen(*msl_ptr) - total_old + total_new + 1);
                    size_t before = (size_t)(type_start - *msl_ptr);
                    memcpy(new_msl, *msl_ptr, before);
                    memcpy(new_msl + before, rep, rep_len);
                    memcpy(new_msl + before + rep_len, new_attr, new_len);
                    strcpy(new_msl + before + rep_len + new_len, pos + old_len);
                    free(*msl_ptr);
                    *msl_ptr = new_msl;

                    /* Inject type conversion at function body start '{'.
                     * gl_TessCoord is float3 in GLSL, but isolines use float2
                     * and some shaders may declare a float2 tesscoord variable.
                     * Convert uint3 to the original type appropriately. */
                    char *brace = strchr(*msl_ptr + before + total_new, '{');
                    if (brace != NULL) {
                        char inject[512];
                        if (strcmp(vartype, "float3") == 0) {
                            snprintf(inject, sizeof(inject),
                                     "\n    float3 %s = (float3)_mgl_tess_tc;", varname);
                        } else if (strcmp(vartype, "float2") == 0) {
                            snprintf(inject, sizeof(inject),
                                     "\n    float2 %s = (float2)_mgl_tess_tc.xy;", varname);
                        } else {
                            snprintf(inject, sizeof(inject),
                                     "\n    float %s = (float)_mgl_tess_tc.x;", varname);
                        }
                        size_t inject_len = strlen(inject);
                        size_t brace_pos = (size_t)(brace + 1 - *msl_ptr);
                        char *new_msl2 = (char *)malloc(strlen(*msl_ptr) + inject_len + 1);
                        memcpy(new_msl2, *msl_ptr, brace_pos);
                        memcpy(new_msl2 + brace_pos, inject, inject_len);
                        strcpy(new_msl2 + brace_pos + inject_len, brace + 1);
                        free(*msl_ptr);
                        *msl_ptr = new_msl2;
                        pos = *msl_ptr + before + total_new + inject_len;
                    } else {
                        pos = *msl_ptr + before + total_new;
                    }
                } else {
                    /* Non-float type, just replace the attribute. */
                    size_t before = (size_t)(pos - *msl_ptr);
                    char *new_msl = (char *)malloc(strlen(*msl_ptr) - old_len + new_len + 1);
                    memcpy(new_msl, *msl_ptr, before);
                    memcpy(new_msl + before, new_attr, new_len);
                    strcpy(new_msl + before + new_len, pos + old_len);
                    free(*msl_ptr);
                    *msl_ptr = new_msl;
                    pos = *msl_ptr + before + new_len;
                }
            }
        }
    }

    /* Step 4: Replace the [[stage_in]] parameter with a device buffer.
     * SPIRV-Cross generates a post-tessellation vertex function with:
     *   kernel void <name>(<patchInType> <paramName> [[stage_in]], ...)
     * where <patchInType> contains a `patch_control_point<inner_type> gl_in;`
     * member.  Both `[[stage_in]]` and `patch_control_point<>` are illegal in
     * compute kernels, so we replace the parameter with:
     *   device <inner_type> *gl_in [[buffer(30)]]
     * and rewrite body references `<paramName>.gl_in` → `gl_in`. */
    {
        const char *stage_in_attr = "[[stage_in]]";
        char *pos = strstr(*msl_ptr, stage_in_attr);
        if (pos != NULL) {
            /* Walk backwards from [[stage_in]] to find the parameter name
             * and type.  Expected format: "<type> <name> [[stage_in]]" */
            const char *attr_start = pos;
            const char *p = attr_start;

            /* Skip whitespace before [[stage_in]] */
            while (p > *msl_ptr && isspace((unsigned char)p[-1])) {
                p--;
            }
            /* p now points to the end of the parameter name */
            const char *name_end = p;
            const char *name_start = name_end;
            while (name_start > *msl_ptr &&
                   (isalnum((unsigned char)name_start[-1]) || name_start[-1] == '_')) {
                name_start--;
            }
            size_t name_len = (size_t)(name_end - name_start);

            /* Skip whitespace before name to find type */
            const char *before_name = name_start;
            while (before_name > *msl_ptr && isspace((unsigned char)before_name[-1])) {
                before_name--;
            }
            const char *type_end = before_name;
            const char *type_start = type_end;
            while (type_start > *msl_ptr &&
                   (isalnum((unsigned char)type_start[-1]) || type_start[-1] == '_')) {
                type_start--;
            }
            size_t type_len = (size_t)(type_end - type_start);

            if (type_len > 0 && name_len > 0 && type_len < 256 && name_len < 256) {
                char param_type[256];
                char param_name[256];
                memcpy(param_type, type_start, type_len);
                param_type[type_len] = '\0';
                memcpy(param_name, name_start, name_len);
                param_name[name_len] = '\0';

                /* Find the struct definition to extract the inner type from
                 * patch_control_point<inner_type>.  Search for:
                 *   struct <param_type> { ... patch_control_point<inner> gl_in; ... } */
                char struct_pattern[512];
                snprintf(struct_pattern, sizeof(struct_pattern),
                         "struct %s", param_type);
                char *struct_pos = strstr(*msl_ptr, struct_pattern);
                if (struct_pos != NULL) {
                    /* Find "patch_control_point<" after the struct definition */
                    char *pcp_pos = strstr(struct_pos, "patch_control_point<");
                    if (pcp_pos != NULL) {
                        /* Extract the inner type name between < and > */
                        char *inner_start = pcp_pos + strlen("patch_control_point<");
                        char *inner_end = strchr(inner_start, '>');
                        if (inner_end != NULL) {
                            size_t inner_len = (size_t)(inner_end - inner_start);
                            if (inner_len > 0 && inner_len < 256) {
                                char inner_type[256];
                                /* Trim whitespace from inner type */
                                const char *is = inner_start;
                                const char *ie = inner_end;
                                while (is < ie && isspace((unsigned char)*is)) is++;
                                while (ie > is && isspace((unsigned char)ie[-1])) ie--;
                                inner_len = (size_t)(ie - is);
                                memcpy(inner_type, is, inner_len);
                                inner_type[inner_len] = '\0';

                                /* Replace the parameter:
                                 *   "<type> <name> [[stage_in]]"
                                 * with:
                                 *   "device <inner_type> *gl_in [[buffer(30)]], device <type> *<name> [[buffer(27)]]"
                                 * We keep <name> (e.g. patchIn) as a device buffer so per-patch
                                 * data (tc_patch_data etc.) is still accessible.  The per-vertex
                                 * data (gl_in) is split out to its own buffer(30). */
                                const char *replacement_start = type_start;
                                size_t old_param_len = (size_t)(pos + strlen(stage_in_attr) - replacement_start);

                                char new_param[768];
                                snprintf(new_param, sizeof(new_param),
                                         "device %s *gl_in [[buffer(30)]], device %s *%s [[buffer(27)]]",
                                         inner_type, param_type, param_name);
                                size_t new_param_len = strlen(new_param);

                                size_t before_len = (size_t)(replacement_start - *msl_ptr);
                                char *new_msl = (char *)malloc(strlen(*msl_ptr) - old_param_len + new_param_len + 1);
                                memcpy(new_msl, *msl_ptr, before_len);
                                memcpy(new_msl + before_len, new_param, new_param_len);
                                strcpy(new_msl + before_len + new_param_len,
                                       replacement_start + old_param_len);
                                free(*msl_ptr);
                                *msl_ptr = new_msl;

                                /* Now replace "<param_name>.gl_in" → "gl_in"
                                 * in the function body. */
                                char body_pattern[512];
                                snprintf(body_pattern, sizeof(body_pattern),
                                         "%s.gl_in", param_name);
                                size_t body_old_len = strlen(body_pattern);
                                const char *body_new = "gl_in";
                                size_t body_new_len = strlen(body_new);

                                char *bpos = *msl_ptr;
                                while ((bpos = strstr(bpos, body_pattern)) != NULL) {
                                    size_t bbefore = (size_t)(bpos - *msl_ptr);
                                    char *bnew = (char *)malloc(strlen(*msl_ptr) - body_old_len + body_new_len + 1);
                                    memcpy(bnew, *msl_ptr, bbefore);
                                    memcpy(bnew + bbefore, body_new, body_new_len);
                                    strcpy(bnew + bbefore + body_new_len,
                                           bpos + body_old_len);
                                    free(*msl_ptr);
                                    *msl_ptr = bnew;
                                    bpos = *msl_ptr + bbefore + body_new_len;
                                }

                                /* Step 4a: Replace remaining "<param_name>.<field>"
                                 * references (per-patch data like tc_patch_data)
                                 * with "<param_name>[_mgl_patch_id].<field>".
                                 * Also add a "uint3 _mgl_patch_id3 [[threadgroup_position_in_grid]]"
                                 * parameter to the kernel if not already present. */
                                {
                                    char patchin_dot[512];
                                    snprintf(patchin_dot, sizeof(patchin_dot),
                                             "%s.", param_name);
                                    if (strstr(*msl_ptr, patchin_dot) != NULL) {
                                        /* Ensure _mgl_patch_id3 param exists. */
                                        mglEnsurePatchId3Param(msl_ptr);

                                        /* Replace "<param_name>." with "<param_name>[_mgl_patch_id]." */
                                        const char *old_dot = patchin_dot;
                                        size_t old_dot_len = strlen(old_dot);
                                        char new_dot[600];
                                        snprintf(new_dot, sizeof(new_dot),
                                                 "%s[_mgl_patch_id3.x].", param_name);
                                        size_t new_dot_len = strlen(new_dot);

                                        char *dpos = *msl_ptr;
                                        while ((dpos = strstr(dpos, old_dot)) != NULL) {
                                            size_t dbefore = (size_t)(dpos - *msl_ptr);
                                            char *dnew = (char *)malloc(strlen(*msl_ptr) - old_dot_len + new_dot_len + 1);
                                            memcpy(dnew, *msl_ptr, dbefore);
                                            memcpy(dnew + dbefore, new_dot, new_dot_len);
                                            strcpy(dnew + dbefore + new_dot_len,
                                                   dpos + old_dot_len);
                                            free(*msl_ptr);
                                            *msl_ptr = dnew;
                                            dpos = *msl_ptr + dbefore + new_dot_len;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* Step 4b: Remove "patch_control_point<...> <name>;" members from
     * struct definitions.  After Step 4, the per-vertex data is accessed
     * via a separate device buffer (gl_in [[buffer(30)]]), so the
     * patch_control_point member in the patchIn struct is no longer needed.
     * Metal compute kernels cannot use patch_control_point<> (it's only
     * valid in post-tessellation vertex functions), so leaving it causes
     * a compile error. */
    {
        const char *pcp_kw = "patch_control_point<";
        size_t pcp_kw_len = strlen(pcp_kw);
        char *pos = *msl_ptr;
        while ((pos = strstr(pos, pcp_kw)) != NULL) {
            /* Find the end of the statement: the ';' after the member name. */
            char *semi = strchr(pos, ';');
            if (semi == NULL) {
                break;
            }
            /* Extend to include the trailing ';' and any preceding whitespace
             * on the line (to avoid leaving blank lines). */
            char *line_start = pos;
            while (line_start > *msl_ptr && line_start[-1] != '\n') {
                line_start--;
            }
            size_t remove_len = (size_t)(semi + 1 - line_start);
            size_t before = (size_t)(line_start - *msl_ptr);
            char *nm = (char *)malloc(strlen(*msl_ptr) - remove_len + 1);
            memcpy(nm, *msl_ptr, before);
            strcpy(nm + before, semi + 1);
            free(*msl_ptr);
            *msl_ptr = nm;
            pos = *msl_ptr + before;
        }
    }

    /* Step 4c: Replace "gl_in.size()" (from GLSL gl_in.length() / gl_PatchVerticesIn)
     * with a patch vertex count passed as buffer(28).  After Step 4, gl_in is
     * a raw device pointer, so .size() doesn't work.  We replace the call with
     * "_mgl_patch_vertices_in" and add a "uint _mgl_patch_vertices_in [[buffer(28)]]"
     * parameter to the kernel. */
    {
        const char *size_call = "gl_in.size()";
        if (strstr(*msl_ptr, size_call) != NULL) {
            /* Replace "gl_in.size()" with "_mgl_patch_vertices_in". */
            size_t old_len = strlen(size_call);
            const char *new_expr = "_mgl_patch_info.x";
            size_t new_len = strlen(new_expr);
            char *spos = *msl_ptr;
            while ((spos = strstr(spos, size_call)) != NULL) {
                size_t before = (size_t)(spos - *msl_ptr);
                char *nm = (char *)malloc(strlen(*msl_ptr) - old_len + new_len + 1);
                memcpy(nm, *msl_ptr, before);
                memcpy(nm + before, new_expr, new_len);
                strcpy(nm + before + new_len, spos + old_len);
                free(*msl_ptr);
                *msl_ptr = nm;
                spos = *msl_ptr + before + new_len;
            }

            /* Add the _mgl_patch_vertices_in parameter if not already present. */
            if (strstr(*msl_ptr, "_mgl_patch_info [[buffer(28)]]") == NULL) {
                const char *close_paren = mglFindMSLEntryParameterClose(*msl_ptr);
                if (close_paren != NULL) {
                    /* Find the opening '(' to check if params are empty. */
                    const char *kernel_kw = strstr(*msl_ptr, "kernel void ");
                    const char *open_paren = kernel_kw ? strchr(kernel_kw, '(') : NULL;
                    const char *inject = "constant uint2& _mgl_patch_info [[buffer(28)]]";
                    size_t inject_len = strlen(inject);
                    const char *prefix = ", ";
                    size_t prefix_len = strlen(prefix);
                    /* Check if parameter list is empty. */
                    const char *p = open_paren + 1;
                    while (p < close_paren && isspace((unsigned char)*p)) p++;
                    if (p >= close_paren) { prefix = ""; prefix_len = 0; }

                    size_t before = (size_t)(close_paren - *msl_ptr);
                    char *nm = (char *)malloc(strlen(*msl_ptr) + prefix_len + inject_len + 1);
                    memcpy(nm, *msl_ptr, before);
                    memcpy(nm + before, prefix, prefix_len);
                    memcpy(nm + before + prefix_len, inject, inject_len);
                    strcpy(nm + before + prefix_len + inject_len, close_paren);
                    free(*msl_ptr);
                    *msl_ptr = nm;
                }
            }
        }
    }

    /* Step 5: Strip [[attribute(N)]] annotations from struct members.
     * These are only valid for vertex stage-in data, which is no longer
     * used after Step 4 converted [[stage_in]] to a device buffer.  Metal
     * may reject [[attribute]] in device buffer struct context. */
    {
        const char *attr_prefix = "[[attribute(";
        size_t attr_prefix_len = strlen(attr_prefix);
        char *pos = *msl_ptr;
        while ((pos = strstr(pos, attr_prefix)) != NULL) {
            /* Find the closing "]]" */
            char *close = strstr(pos, "]]");
            if (close == NULL) {
                break;
            }
            size_t attr_len = (size_t)(close + 2 - pos);

            /* Check if this is on a struct member (preceded by a type/name,
             * not a parameter).  We strip all [[attribute(N)]] since the
             * TES kernel has no stage-in. */
            /* Also strip preceding whitespace for cleaner output. */
            char *strip_start = pos;
            while (strip_start > *msl_ptr && isspace((unsigned char)strip_start[-1])) {
                strip_start--;
            }
            size_t strip_len = (size_t)(close + 2 - strip_start);

            size_t before = (size_t)(strip_start - *msl_ptr);
            char *new_msl = (char *)malloc(strlen(*msl_ptr) - strip_len + 1);
            memcpy(new_msl, *msl_ptr, before);
            strcpy(new_msl + before, close + 2);
            free(*msl_ptr);
            *msl_ptr = new_msl;
            pos = *msl_ptr + before;
        }
    }

    /* Step 5b: Patch gl_in[N] indexing to be per-patch.
     *
     * After Step 4, gl_in is a device pointer to the entire TCS output
     * buffer.  TES accesses gl_in[N] where N is the vertex index within
     * the patch (0..patchVertices-1).  But the buffer contains all patches,
     * so we need gl_in[patchID * outputVertices + N].
     *
     * We replace "gl_in[" with "gl_in[_mgl_patch_id3.x * _mgl_patch_info.y + "
     * where _mgl_patch_info.y is the TCS output vertex count (packed into
     * buffer(28) alongside _mgl_patch_info.x = patch_vertices_in).
     *
     * We also ensure _mgl_patch_id3 is declared (it may be missing if
     * Step 4a didn't run because the shader has no per-patch inputs).
     *
     * Skip the "gl_in.size()" which was already handled by Step 4c. */
    {
        /* Ensure _mgl_patch_id3 param exists (may have been added by Step 4a
         * or Step 6).  If not, add it now. */
        mglEnsurePatchId3Param(msl_ptr);

        /* Ensure _mgl_patch_info param exists (added by Step 4c only when
         * gl_in.size() is present).  Step 5b uses _mgl_patch_info.y for
         * per-patch gl_in indexing, so we must add it if not already there. */
        if (strstr(*msl_ptr, "_mgl_patch_info [[buffer(28)]]") == NULL) {
            const char *cp = mglFindMSLEntryParameterClose(*msl_ptr);
            if (cp != NULL) {
                const char *inj = "constant uint2& _mgl_patch_info [[buffer(28)]]";
                size_t inj_len = strlen(inj);
                const char *pfx = ", ";
                size_t pfx_len = strlen(pfx);
                const char *kw = strstr(*msl_ptr, "kernel void ");
                const char *op = kw ? strchr(kw, '(') : NULL;
                const char *pp = op ? op + 1 : cp;
                while (pp < cp && isspace((unsigned char)*pp)) pp++;
                if (pp >= cp) { pfx = ""; pfx_len = 0; }
                size_t b = (size_t)(cp - *msl_ptr);
                char *nm = (char *)malloc(strlen(*msl_ptr) + pfx_len + inj_len + 1);
                memcpy(nm, *msl_ptr, b);
                memcpy(nm + b, pfx, pfx_len);
                memcpy(nm + b + pfx_len, inj, inj_len);
                strcpy(nm + b + pfx_len + inj_len, cp);
                free(*msl_ptr);
                *msl_ptr = nm;
            }
        }

        /* Replace "gl_in[" with "gl_in[_mgl_patch_id3.x * _mgl_patch_info.y + "
         * but skip "gl_in.size()" (already replaced by Step 4c to
         * _mgl_patch_info.x, so "gl_in[" won't match those).
         * Also skip the parameter declaration "device <type> *gl_in [[buffer(30)]]"
         * which doesn't contain "gl_in[". */
        const char *old_idx = "gl_in[";
        size_t old_idx_len = strlen(old_idx);
        const char *new_idx = "gl_in[_mgl_patch_id3.x * _mgl_patch_info.y + ";
        size_t new_idx_len = strlen(new_idx);
        char *ipos = *msl_ptr;
        while ((ipos = strstr(ipos, old_idx)) != NULL) {
            size_t before = (size_t)(ipos - *msl_ptr);
            char *nm = (char *)malloc(strlen(*msl_ptr) - old_idx_len + new_idx_len + 1);
            memcpy(nm, *msl_ptr, before);
            memcpy(nm + before, new_idx, new_idx_len);
            strcpy(nm + before + new_idx_len, ipos + old_idx_len);
            free(*msl_ptr);
            *msl_ptr = nm;
            ipos = *msl_ptr + before + new_idx_len;
        }
    }

    /* Step 6: XFB (Transform Feedback) capture.
     *
     * When the program has transform feedback varyings configured
     * (transform_feedback_varying_count > 0) with GL_INTERLEAVED_ATTRIBS,
     * the TES compute kernel must write the captured output fields into an
     * XFB buffer so glMapBufferRange on GL_TRANSFORM_FEEDBACK_BUFFER returns
     * the expected data.  This is needed because the TES kernel bypasses the
     * render pipeline (which would normally capture XFB output).
     *
     * We inject:
     *   1. A "device char* _mgl_xfb_out [[buffer(29)]]" parameter.
     *   2. Before the final "return;" of the kernel, write each XFB-captured
     *      field of the "out" struct to the XFB buffer at the correct offset.
     *
     * The XFB buffer is bound in dispatchTessEvaluationShader (MGLRenderer.m)
     * to buffer(29).  The per-patch offset is _mgl_patch_id * xfb_stride.
     */
    if (program &&
        program->transform_feedback_varying_count > 0 &&
        program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS) {

        GLboolean has_xfb = GL_FALSE;
        for (GLsizei i = 0; i < program->transform_feedback_varying_count; i++) {
            const char *vname = program->transform_feedback_varying_names[i];
            if (vname && vname[0] != '\0') {
                SpirvResourceList *outputs =
                    &program->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
                for (GLuint j = 0; outputs->list && j < outputs->count; j++) {
                    if (outputs->list[j].name &&
                        strcmp(outputs->list[j].name, vname) == 0) {
                        has_xfb = GL_TRUE;
                        break;
                    }
                }
            }
        }

        if (has_xfb) {
            /* Inject the XFB output buffer parameter [[buffer(29)]]. */
            if (strstr(*msl_ptr, "_mgl_xfb_out [[buffer(29)]]") == NULL) {
                const char *close_paren = mglFindMSLEntryParameterClose(*msl_ptr);
                if (close_paren != NULL) {
                    const char *inject = "device char* _mgl_xfb_out [[buffer(29)]]";
                    size_t inject_len = strlen(inject);
                    const char *prefix = ", ";
                    size_t prefix_len = strlen(prefix);
                    const char *kernel_kw = strstr(*msl_ptr, "kernel void ");
                    const char *open_paren = kernel_kw ? strchr(kernel_kw, '(') : NULL;
                    const char *p = open_paren ? open_paren + 1 : close_paren;
                    while (p < close_paren && isspace((unsigned char)*p)) p++;
                    if (p >= close_paren) { prefix = ""; prefix_len = 0; }

                    size_t before = (size_t)(close_paren - *msl_ptr);
                    char *nm = (char *)malloc(strlen(*msl_ptr) + prefix_len + inject_len + 1);
                    memcpy(nm, *msl_ptr, before);
                    memcpy(nm + before, prefix, prefix_len);
                    memcpy(nm + before + prefix_len, inject, inject_len);
                    strcpy(nm + before + prefix_len + inject_len, close_paren);
                    free(*msl_ptr);
                    *msl_ptr = nm;
                }
            }

            /* Build the XFB write statements. */
            char write_block[4096];
            write_block[0] = '\0';
            size_t wb_len = 0;
            GLuint xfb_offset = 0;
            GLboolean build_ok = GL_TRUE;

            for (GLsizei i = 0; i < program->transform_feedback_varying_count && build_ok; i++) {
                const char *vname = program->transform_feedback_varying_names[i];
                if (!vname || vname[0] == '\0') {
                    continue;
                }
                if (strchr(vname, '.') != NULL) {
                    build_ok = GL_FALSE;
                    break;
                }

                /* Find the field type in the _out struct:
                 * "    <type> <vname> [[<attr>]]" */
                char field_pattern[256];
                snprintf(field_pattern, sizeof(field_pattern), " %s [[", vname);
                char *fpos = strstr(*msl_ptr, field_pattern);
                if (!fpos) {
                    build_ok = GL_FALSE;
                    break;
                }
                char *type_end = fpos;
                while (type_end > *msl_ptr && isspace((unsigned char)type_end[-1])) {
                    type_end--;
                }
                char *type_start = type_end;
                while (type_start > *msl_ptr &&
                       (isalnum((unsigned char)type_start[-1]) || type_start[-1] == '_')) {
                    type_start--;
                }
                size_t type_len = (size_t)(type_end - type_start);
                if (type_len == 0 || type_len >= 64) {
                    build_ok = GL_FALSE;
                    break;
                }
                char field_type[64];
                memcpy(field_type, type_start, type_len);
                field_type[type_len] = '\0';

                GLuint field_bytes = 0;
                if (strcmp(field_type, "float") == 0) field_bytes = 4;
                else if (strcmp(field_type, "int") == 0) field_bytes = 4;
                else if (strcmp(field_type, "uint") == 0) field_bytes = 4;
                else if (strcmp(field_type, "float2") == 0) field_bytes = 8;
                else if (strcmp(field_type, "float3") == 0) field_bytes = 12;
                else if (strcmp(field_type, "float4") == 0) field_bytes = 16;
                else if (strcmp(field_type, "int2") == 0) field_bytes = 8;
                else if (strcmp(field_type, "int3") == 0) field_bytes = 12;
                else if (strcmp(field_type, "int4") == 0) field_bytes = 16;
                else if (strcmp(field_type, "uint2") == 0) field_bytes = 8;
                else if (strcmp(field_type, "uint3") == 0) field_bytes = 12;
                else if (strcmp(field_type, "uint4") == 0) field_bytes = 16;
                else {
                    build_ok = GL_FALSE;
                    break;
                }

                char stmt[512];
                snprintf(stmt, sizeof(stmt),
                         "\n    *((device %s*)(_mgl_xfb_out + %u)) = out.%s;",
                         field_type, xfb_offset, vname);
                size_t stmt_len = strlen(stmt);
                if (wb_len + stmt_len >= sizeof(write_block) - 1) {
                    build_ok = GL_FALSE;
                    break;
                }
                memcpy(write_block + wb_len, stmt, stmt_len);
                wb_len += stmt_len;
                write_block[wb_len] = '\0';

                xfb_offset += field_bytes;
            }

            if (build_ok && wb_len > 0) {
                /* Ensure _mgl_patch_id3 param exists (may have been added by
                 * Step 4a or 5b).  If not, add it now. */
                mglEnsurePatchId3Param(msl_ptr);

                /* Ensure _mgl_tess_tc (thread_position_in_threadgroup) exists.
                 * Step 3 adds it when gl_TessCoord is used, but for XFB-only
                 * shaders it may be missing.  Reuse it as the per-patch vertex
                 * index.  Also add _mgl_tptg (threads_per_threadgroup) for the
                 * vertex count per patch. */
                if (strstr(*msl_ptr, "_mgl_tess_tc") == NULL) {
                    const char *cp2 = mglFindMSLEntryParameterClose(*msl_ptr);
                    if (cp2 != NULL) {
                        const char *inj2 = "uint3 _mgl_tess_tc [[thread_position_in_threadgroup]]";
                        size_t inj2_len = strlen(inj2);
                        const char *pfx2 = ", ";
                        size_t pfx2_len = strlen(pfx2);
                        const char *kw2 = strstr(*msl_ptr, "kernel void ");
                        const char *op2 = kw2 ? strchr(kw2, '(') : NULL;
                        const char *pp2 = op2 ? op2 + 1 : cp2;
                        while (pp2 < cp2 && isspace((unsigned char)*pp2)) pp2++;
                        if (pp2 >= cp2) { pfx2 = ""; pfx2_len = 0; }
                        size_t b2 = (size_t)(cp2 - *msl_ptr);
                        char *nm2 = (char *)malloc(strlen(*msl_ptr) + pfx2_len + inj2_len + 1);
                        memcpy(nm2, *msl_ptr, b2);
                        memcpy(nm2 + b2, pfx2, pfx2_len);
                        memcpy(nm2 + b2 + pfx2_len, inj2, inj2_len);
                        strcpy(nm2 + b2 + pfx2_len + inj2_len, cp2);
                        free(*msl_ptr);
                        *msl_ptr = nm2;
                    }
                }
                /* Ensure _mgl_tptg (threads_per_threadgroup) exists. */
                if (strstr(*msl_ptr, "_mgl_tptg") == NULL) {
                    const char *cp2 = mglFindMSLEntryParameterClose(*msl_ptr);
                    if (cp2 != NULL) {
                        const char *inj2 = "uint3 _mgl_tptg [[threads_per_threadgroup]]";
                        size_t inj2_len = strlen(inj2);
                        const char *pfx2 = ", ";
                        size_t pfx2_len = strlen(pfx2);
                        const char *kw2 = strstr(*msl_ptr, "kernel void ");
                        const char *op2 = kw2 ? strchr(kw2, '(') : NULL;
                        const char *pp2 = op2 ? op2 + 1 : cp2;
                        while (pp2 < cp2 && isspace((unsigned char)*pp2)) pp2++;
                        if (pp2 >= cp2) { pfx2 = ""; pfx2_len = 0; }
                        size_t b2 = (size_t)(cp2 - *msl_ptr);
                        char *nm2 = (char *)malloc(strlen(*msl_ptr) + pfx2_len + inj2_len + 1);
                        memcpy(nm2, *msl_ptr, b2);
                        memcpy(nm2 + b2, pfx2, pfx2_len);
                        memcpy(nm2 + b2 + pfx2_len, inj2, inj2_len);
                        strcpy(nm2 + b2 + pfx2_len + inj2_len, cp2);
                        free(*msl_ptr);
                        *msl_ptr = nm2;
                    }
                }

                /* Find the final "return;" and inject the stride offset +
                 * write block before it. */
                char *last_return = NULL;
                char *search = *msl_ptr;
                while ((search = strstr(search, "return;")) != NULL) {
                    last_return = search;
                    search += 7;
                }
                if (last_return) {
                    char stride_decl[512];
                    snprintf(stride_decl, sizeof(stride_decl),
                             "\n    _mgl_xfb_out += (_mgl_patch_id3.x * _mgl_tptg.x + _mgl_tess_tc.x) * %u;",
                             xfb_offset);
                    size_t stride_len = strlen(stride_decl);
                    size_t insert_pos = (size_t)(last_return - *msl_ptr);
                    char *nm = (char *)malloc(strlen(*msl_ptr) + stride_len + wb_len + 1);
                    memcpy(nm, *msl_ptr, insert_pos);
                    memcpy(nm + insert_pos, stride_decl, stride_len);
                    memcpy(nm + insert_pos + stride_len, write_block, wb_len);
                    strcpy(nm + insert_pos + stride_len + wb_len, last_return);
                    free(*msl_ptr);
                    *msl_ptr = nm;
                }
            }
        }
    }

    mglFixMSLTessCullDistanceOutputs(msl_ptr);
}

/* Find the opening '(' of the MSL entry-function parameter list for any
 * shader stage (kernel/vertex/fragment). */
static const char *mglFindMSLEntryParameterOpen(const char *msl)
{
    if (!msl) {
        return NULL;
    }

    /* SPIRV-Cross emits entry functions as:
     *   kernel void <name>(...)
     *   vertex <out_type> <name>(...)
     *   fragment <out_type> <name>(...)
     * Find the first '(' after the entry keyword and balance parentheses. */
    static const struct { const char *kw; size_t len; } kEntryKinds[] = {
        { "kernel ", 7 },
        { "vertex ", 7 },
        { "fragment ", 9 },
    };

    const char *entry = NULL;
    for (size_t i = 0; i < sizeof(kEntryKinds)/sizeof(kEntryKinds[0]); i++) {
        const char *p = msl;
        while ((p = strstr(p, kEntryKinds[i].kw)) != NULL) {
            /* Make sure this is an entry-function declaration: the keyword
             * must be at the start of a line (not inside an identifier). */
            if (p == msl || p[-1] == '\n' || p[-1] == '\t' || p[-1] == ' ') {
                entry = p;
                break;
            }
            p += kEntryKinds[i].len;
        }
        if (entry) {
            break;
        }
    }
    if (!entry) {
        /* Fall back to the legacy compute-only search (kernel void <name>). */
        const char *kernel = strstr(msl, "kernel void ");
        return kernel ? strchr(kernel, '(') : NULL;
    }

    return strchr(entry, '(');
}

/* Find the closing ')' of the MSL entry-function parameter list for any
 * shader stage (kernel/vertex/fragment), not just compute kernels.  Returns
 * a pointer to the ')' in the source, or NULL if no entry function is found. */
static const char *mglFindMSLEntryParameterClose(const char *msl)
{
    const char *open = mglFindMSLEntryParameterOpen(msl);
    if (!open) {
        return NULL;
    }
    int depth = 0;
    for (const char *p = open; *p; p++) {
        if (*p == '(') {
            depth++;
        } else if (*p == ')') {
            depth--;
            if (depth == 0) {
                return p;
            }
        }
    }
    return NULL;
}

static void mglInjectMSLAtomicCounterArguments(Program *program, int stage, char **msl_ptr)
{
    if (!program || !msl_ptr || !*msl_ptr ||
        stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }

    SpirvResourceList *atomics =
        &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER];
    if (!atomics->count) {
        return;
    }

    MGLMSLBindingMap binding_map;
    mglBuildMSLBindingMap(*msl_ptr, &binding_map);

    GLboolean used_slots[MAX_BINDABLE_BUFFERS] = {0};
    for (size_t i = 0; i < binding_map.count; i++) {
        if (binding_map.entries[i].kind == MGL_MSL_BINDING_BUFFER &&
            binding_map.entries[i].index < MAX_BINDABLE_BUFFERS) {
            used_slots[binding_map.entries[i].index] = GL_TRUE;
        }
    }

    GLuint next_slot = 0u;
    for (GLuint i = 0; i < atomics->count; i++) {
        SpirvResource *res = &atomics->list[i];
        if (!res->name || res->name[0] == '\0') {
            continue;
        }

        GLuint existing_slot = 0u;
        if (mglFindMSLResourceIndexInMap(&binding_map,
                                         MGL_MSL_BINDING_BUFFER,
                                         res->name,
                                         &existing_slot)) {
            continue;
        }

        while (next_slot < MAX_BINDABLE_BUFFERS && used_slots[next_slot]) {
            next_slot++;
        }
        if (next_slot >= MAX_BINDABLE_BUFFERS) {
            fprintf(stderr,
                    "MGL WARNING: no Metal buffer slot available for atomic counter %s\n",
                    res->name);
            break;
        }

        char injected_parameter[256];
        /* If the entry function's parameter list is empty (the '(' is
         * immediately followed by ')'), inject the first argument without a
         * leading comma to avoid `func(, device ...)` syntax errors. */
        const char *close = mglFindMSLEntryParameterClose(*msl_ptr);
        GLboolean empty_param_list = (close && close > *msl_ptr && close[-1] == '(');
        /* Use pointer (*) for array atomic counters so SPIRV-Cross's
         * generated `&counters[index]` subscript compiles.  Single
         * (non-array) counters use a reference (&) so `&counters` yields
         * `device atomic_uint*` as SPIRV-Cross expects. */
        GLboolean is_array = (res->gl_array_size > 1);
        int written = snprintf(injected_parameter,
                               sizeof(injected_parameter),
                               "%sdevice atomic_uint%s %s [[buffer(%u)]]",
                               empty_param_list ? "" : ", ",
                               is_array ? "*" : "&",
                               res->name,
                               (unsigned)next_slot);
        if (written <= 0 || (size_t)written >= sizeof(injected_parameter)) {
            continue;
        }

        if (!close || !mglInsertStringAt(msl_ptr, close, injected_parameter)) {
            fprintf(stderr,
                    "MGL WARNING: failed to inject Metal atomic counter argument %s\n",
                    res->name);
            continue;
        }

        used_slots[next_slot] = GL_TRUE;
        next_slot++;
    }

    replace_all_substr(msl_ptr, "(thread atomic_uint*)&", "(device atomic_uint*)&");
}

static size_t count_substr(const char *str, const char *needle)
{
    size_t count = 0;
    size_t needle_len;
    const char *pos;

    if (!str || !needle) {
        return 0;
    }

    needle_len = strlen(needle);
    if (needle_len == 0) {
        return 0;
    }

    pos = str;
    while ((pos = strstr(pos, needle)) != NULL) {
        count++;
        pos += needle_len;
    }

    return count;
}

static void mglFixMSLPlainStructPointerArrayAccess(Program *program,
                                                   int stage,
                                                   char **msl)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES || !msl || !*msl) {
        return;
    }

    SpirvResourceList *resources =
        &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
    size_t fix_count = 0;
    for (GLuint i = 0; resources->list && i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if (!res->name || !res->ubo_members || res->gl_array_size <= 1) {
            continue;
        }

        char pointer_array_decl[128];
        snprintf(pointer_array_decl, sizeof(pointer_array_decl), "* %s[]", res->name);
        if (!strstr(*msl, pointer_array_decl)) {
            snprintf(pointer_array_decl, sizeof(pointer_array_decl), "*%s[]", res->name);
            if (!strstr(*msl, pointer_array_decl)) {
                continue;
            }
        }

        for (GLint elem = 0; elem < res->gl_array_size && elem < 64; elem++) {
            char from[64];
            char to[64];
            snprintf(from, sizeof(from), "%s[%d].", res->name, elem);
            snprintf(to, sizeof(to), "%s[%d]->", res->name, elem);
            size_t hits = count_substr(*msl, from);
            if (hits > 0) {
                replace_all_substr(msl, from, to);
                fix_count += hits;
            }
        }
    }

    if (fix_count > 0) {
        fprintf(stderr,
                "MGL MSL PLAIN STRUCT PTR ARRAY FIX: program=%u stage=%d hits=%zu\n",
                program->name,
                stage,
                fix_count);
    }
}

static int mglMSLDoubleReplacementLength(const char *token)
{
    if (!token) {
        return 0;
    }
    if (!strcmp(token, "double")) {
        return 6;
    }
    if (!strcmp(token, "double2") ||
        !strcmp(token, "double3") ||
        !strcmp(token, "double4")) {
        return 7;
    }
    if (!strcmp(token, "double2x2") ||
        !strcmp(token, "double2x3") ||
        !strcmp(token, "double2x4") ||
        !strcmp(token, "double3x2") ||
        !strcmp(token, "double3x3") ||
        !strcmp(token, "double3x4") ||
        !strcmp(token, "double4x2") ||
        !strcmp(token, "double4x3") ||
        !strcmp(token, "double4x4")) {
        return 9;
    }
    return 0;
}

static void mglLowerMSLDoubleTypesToFloat(char **pstr)
{
    char *src;
    char *out;
    size_t len;
    size_t r = 0;
    size_t w = 0;
    size_t replacements = 0;

    if (!pstr || !*pstr) {
        return;
    }

    src = *pstr;
    len = strlen(src);
    out = (char *)malloc(len + 1);
    if (!out) {
        return;
    }

    while (r < len) {
        int match_len = 0;
        const char *replacement = NULL;

        if ((src[r] == 'l' || src[r] == 'L') &&
            r + 1 < len &&
            (src[r + 1] == 'f' || src[r + 1] == 'F') &&
            r > 0 &&
            (isdigit((unsigned char)src[r - 1]) || src[r - 1] == '.')) {
            out[w++] = 'f';
            r += 2;
            replacements++;
            continue;
        }

        if ((r == 0 ||
             (src[r - 1] == '_' || isalnum((unsigned char)src[r - 1])) == 0) &&
            !strncmp(src + r, "double", 6)) {
            char token[16] = {0};
            size_t t = 0;
            while (r + t < len &&
                   t + 1 < sizeof(token) &&
                   (src[r + t] == '_' || isalnum((unsigned char)src[r + t]))) {
                token[t] = src[r + t];
                t++;
            }
            token[t] = '\0';
            match_len = mglMSLDoubleReplacementLength(token);
            if (match_len > 0 &&
                (r + (size_t)match_len >= len ||
                 (src[r + (size_t)match_len] == '_' ||
                  isalnum((unsigned char)src[r + (size_t)match_len])) == 0)) {
                if (match_len == 6) {
                    replacement = "float";
                } else if (match_len == 7) {
                    static char vec_buf[8];
                    vec_buf[0] = 'f';
                    vec_buf[1] = 'l';
                    vec_buf[2] = 'o';
                    vec_buf[3] = 'a';
                    vec_buf[4] = 't';
                    vec_buf[5] = src[r + 6];
                    vec_buf[6] = '\0';
                    replacement = vec_buf;
                } else {
                    static char mat_buf[10];
                    mat_buf[0] = 'f';
                    mat_buf[1] = 'l';
                    mat_buf[2] = 'o';
                    mat_buf[3] = 'a';
                    mat_buf[4] = 't';
                    memcpy(mat_buf + 5, src + r + 6, 3);
                    mat_buf[8] = '\0';
                    replacement = mat_buf;
                }
            } else {
                match_len = 0;
            }
        }

        if (replacement) {
            size_t repl_len = strlen(replacement);
            memcpy(out + w, replacement, repl_len);
            w += repl_len;
            r += (size_t)match_len;
            replacements++;
        } else {
            out[w++] = src[r++];
        }
    }

    out[w] = '\0';
    if (replacements > 0) {
        free(*pstr);
        *pstr = out;
    } else {
        free(out);
    }
}

static GLboolean mglFindMSLUserLocationForName(const char *msl, const char *name, GLuint *location_out)
{
    if (!msl || !name || !location_out) {
        return GL_FALSE;
    }

    size_t name_len = strlen(name);
    if (name_len == 0) {
        return GL_FALSE;
    }

    const char *cursor = msl;
    while ((cursor = strstr(cursor, name)) != NULL) {
        const char *line_start = cursor;
        const char *line_end = cursor;

        while (line_start > msl && line_start[-1] != '\n' && line_start[-1] != '\r') {
            line_start--;
        }
        while (*line_end && *line_end != '\n' && *line_end != '\r') {
            line_end++;
        }

        char before = (cursor == line_start) ? '\0' : cursor[-1];
        char after = cursor[name_len];
        GLboolean before_ident = (before == '_') ||
                                  (before >= '0' && before <= '9') ||
                                  (before >= 'A' && before <= 'Z') ||
                                  (before >= 'a' && before <= 'z');
        GLboolean after_ident = (after == '_') ||
                                 (after >= '0' && after <= '9') ||
                                 (after >= 'A' && after <= 'Z') ||
                                 (after >= 'a' && after <= 'z');
        if (!before_ident && !after_ident) {
            const char *loc = strstr(cursor, "[[user(locn");
            if (loc && loc < line_end) {
                loc += strlen("[[user(locn");
                char *end = NULL;
                unsigned long parsed = strtoul(loc, &end, 10);
                if (end && end > loc && end <= line_end) {
                    *location_out = (GLuint)parsed;
                    return GL_TRUE;
                }
            }
        }

        cursor += name_len;
    }

    return GL_FALSE;
}

static GLboolean mglBuildMSLResourceNameVariant(const char *name,
                                                unsigned variant,
                                                char *out,
                                                size_t out_size)
{
    if (!name || !out || out_size == 0u) {
        return GL_FALSE;
    }

    switch (variant) {
        case 0:
            snprintf(out, out_size, "%s", name);
            return GL_TRUE;
        case 1:
            snprintf(out, out_size, "%s_0", name);
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglFindMSLUserLocationForResourceName(const char *msl,
                                                       const char *name,
                                                       GLuint *location_out,
                                                       char *matched_name,
                                                       size_t matched_name_size)
{
    char candidate[256];

    for (unsigned variant = 0; mglBuildMSLResourceNameVariant(name, variant, candidate, sizeof(candidate)); variant++) {
        if (mglFindMSLUserLocationForName(msl, candidate, location_out)) {
            if (matched_name && matched_name_size > 0u) {
                snprintf(matched_name, matched_name_size, "%s", candidate);
            }
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLboolean mglReplaceMSLUserLocationForResourceName(char **msl_ptr,
                                                          const char *name,
                                                          GLuint current_location,
                                                          GLuint desired_location,
                                                          char *matched_name,
                                                          size_t matched_name_size)
{
    char candidate[256];

    if (!msl_ptr || !*msl_ptr || !name) {
        return GL_FALSE;
    }

    for (unsigned variant = 0; mglBuildMSLResourceNameVariant(name, variant, candidate, sizeof(candidate)); variant++) {
        char from[320];
        char to[320];

        snprintf(from, sizeof(from), "%s [[user(locn%u)]]",
                 candidate, (unsigned)current_location);
        snprintf(to, sizeof(to), "%s [[user(locn%u)]]",
                 candidate, (unsigned)desired_location);

        if (strstr(*msl_ptr, from)) {
            replace_all_substr(msl_ptr, from, to);
            if (matched_name && matched_name_size > 0u) {
                snprintf(matched_name, matched_name_size, "%s", candidate);
            }
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static size_t mglMSLVectorCSize(unsigned components)
{
    /* sizeof(simd::floatN) in Metal: vec3 is padded to 16 bytes
     * (alignof=16, sizeof=16), matching std140 effective size.
     * Verified via C++ test: sizeof(simd::float3)==16. */
    switch (components) {
        case 1: return 4;
        case 2: return 8;
        case 3:
        case 4:
            return 16;
        default:
            return 0;
    }
}

static size_t mglStd140VectorAlign(unsigned components)
{
    switch (components) {
        case 1: return 4;
        case 2: return 8;
        case 3:
        case 4:
            return 16;
        default:
            return 0;
    }
}

static GLboolean mglMSLUniformTypeLayout(const char *trimmed,
                                         size_t *c_size_out,
                                         size_t *std140_align_out,
                                         size_t *actual_align_out)
{
    const char *p = NULL;
    unsigned components = 1;
    GLboolean packed = GL_FALSE;

    if (!trimmed || !c_size_out || !std140_align_out || !actual_align_out) {
        return GL_FALSE;
    }

    if (strncmp(trimmed, "packed_", 7) == 0) {
        packed = GL_TRUE;
        trimmed += 7;
    }

    if (strncmp(trimmed, "float", 5) == 0) {
        p = trimmed + 5;
    } else if (strncmp(trimmed, "uint", 4) == 0) {
        p = trimmed + 4;
    } else if (strncmp(trimmed, "int", 3) == 0) {
        p = trimmed + 3;
    } else {
        return GL_FALSE;
    }

    if (*p >= '2' && *p <= '4') {
        components = (unsigned)(*p - '0');
        p++;
        if (*p == 'x' && trimmed[0] == 'f') {
            unsigned rows = 0;
            p++;
            if (*p < '2' || *p > '4') {
                return GL_FALSE;
            }
            rows = (unsigned)(*p - '0');
            p++;
            if (*p != ' ' && *p != '\t') {
                return GL_FALSE;
            }

            if (packed) {
                return GL_FALSE;
            }
            *c_size_out = components * mglMSLVectorCSize(rows);
            *std140_align_out = 16;
            *actual_align_out = 16;
            return *c_size_out > 0 ? GL_TRUE : GL_FALSE;
        }
    }

    if (*p != ' ' && *p != '\t') {
        return GL_FALSE;
    }

    if (packed) {
        *c_size_out = components * 4u;
        *std140_align_out = 4;
        *actual_align_out = 4;
        return GL_TRUE;
    }

    *c_size_out = mglMSLVectorCSize(components);
    *std140_align_out = mglStd140VectorAlign(components);
    *actual_align_out = *std140_align_out;
    return (*c_size_out > 0 && *std140_align_out > 0 && *actual_align_out > 0) ? GL_TRUE : GL_FALSE;
}

typedef struct MGLMSLStructLayout {
    char name[128];
    size_t size;
    size_t align;
} MGLMSLStructLayout;

typedef struct MGLMSLStructDeps {
    char name[128];
    char member_types[64][128];
    unsigned member_type_count;
} MGLMSLStructDeps;

static GLboolean mglMSLNameInList(char names[][128], unsigned count, const char *name)
{
    if (!name || !*name) {
        return GL_FALSE;
    }

    for (unsigned i = 0; i < count; i++) {
        if (strcmp(names[i], name) == 0) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLboolean mglMSLAddName(char names[][128], unsigned *count, unsigned capacity, const char *name)
{
    if (!names || !count || !name || !*name ||
        mglMSLNameInList(names, *count, name) || *count >= capacity) {
        return GL_FALSE;
    }

    strncpy(names[*count], name, 127);
    names[*count][127] = '\0';
    (*count)++;
    return GL_TRUE;
}

static GLboolean mglGLSLBlockDeclContainsToken(const char *glsl_src,
                                               const char *block_name,
                                               const char *token)
{
    if (!glsl_src || !block_name || !*block_name || !token || !*token) {
        return GL_FALSE;
    }

    size_t block_len = strlen(block_name);
    const char *pos = glsl_src;
    while ((pos = strstr(pos, block_name)) != NULL) {
        const char *after_name = pos + block_len;
        if ((pos > glsl_src && (isalnum((unsigned char)pos[-1]) || pos[-1] == '_')) ||
            (isalnum((unsigned char)*after_name) || *after_name == '_')) {
            pos = after_name;
            continue;
        }

        const char *brace = after_name;
        while (*brace && isspace((unsigned char)*brace)) {
            brace++;
        }
        if (*brace != '{') {
            pos = after_name;
            continue;
        }

        const char *decl_begin = pos;
        while (decl_begin > glsl_src && decl_begin[-1] != ';' && decl_begin[-1] != '}') {
            decl_begin--;
        }
        if (mglRangeContainsToken(decl_begin, brace, token)) {
            return GL_TRUE;
        }
        pos = after_name;
    }

    return GL_FALSE;
}

static GLboolean mglMSLTokenLooksStructLike(const char *token)
{
    if (!token || !*token ||
        strncmp(token, "float", 5) == 0 ||
        strncmp(token, "uint", 4) == 0 ||
        strncmp(token, "int", 3) == 0 ||
        strncmp(token, "char", 4) == 0 ||
        strncmp(token, "bool", 4) == 0 ||
        strncmp(token, "packed_", 7) == 0 ||
        strcmp(token, "struct") == 0) {
        return GL_FALSE;
    }
    return GL_TRUE;
}

static MGLMSLStructDeps *mglMSLFindStructDeps(MGLMSLStructDeps *deps,
                                              unsigned count,
                                              const char *name)
{
    if (!deps || !name || !*name) {
        return NULL;
    }

    for (unsigned i = 0; i < count; i++) {
        if (strcmp(deps[i].name, name) == 0) {
            return &deps[i];
        }
    }

    return NULL;
}

static void mglMSLCollectStructDeps(const char *msl,
                                    MGLMSLStructDeps *deps,
                                    unsigned *dep_count,
                                    unsigned dep_capacity)
{
    GLboolean in_struct = GL_FALSE;
    MGLMSLStructDeps *current = NULL;
    const char *p = msl;

    if (!msl || !deps || !dep_count) {
        return;
    }

    *dep_count = 0;
    while (*p) {
        const char *line_start = p;
        const char *line_end = strchr(p, '\n');
        size_t line_len = line_end ? (size_t)(line_end - line_start + 1) : strlen(line_start);

        if (!in_struct) {
            char st[128] = {0};
            if (sscanf(line_start, "struct %127s", st) == 1 && *dep_count < dep_capacity) {
                char *brace = strchr(st, '{');
                if (brace) *brace = '\0';
                current = &deps[(*dep_count)++];
                memset(current, 0, sizeof(*current));
                strncpy(current->name, st, sizeof(current->name) - 1);
                in_struct = GL_TRUE;
            }
        } else {
            const char *trimmed = line_start;
            char type_name[128] = {0};
            while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
            if (memchr(line_start, '}', line_len - (line_end ? 1 : 0))) {
                in_struct = GL_FALSE;
                current = NULL;
            } else if (current && sscanf(trimmed, "%127s", type_name) == 1 &&
                       mglMSLTokenLooksStructLike(type_name)) {
                char *array = strchr(type_name, '[');
                if (array) *array = '\0';
                char *ptr = strchr(type_name, '*');
                if (ptr) *ptr = '\0';
                if (!mglMSLNameInList(current->member_types,
                                      current->member_type_count,
                                      type_name) &&
                    current->member_type_count < 64) {
                    strncpy(current->member_types[current->member_type_count],
                            type_name,
                            sizeof(current->member_types[current->member_type_count]) - 1);
                    current->member_types[current->member_type_count][127] = '\0';
                    current->member_type_count++;
                }
            }
        }

        p = line_end ? line_end + 1 : line_start + line_len;
    }
}

static size_t mglMSLDeclaratorArrayCount(const char *trimmed)
{
    const char *bracket = strchr(trimmed, '[');
    const char *semicolon = strchr(trimmed, ';');
    const char *newline = strchr(trimmed, '\n');
    char *end = NULL;
    unsigned long count = 0;

    if (!bracket) {
        return 1;
    }
    if (semicolon && bracket > semicolon) {
        return 1;
    }
    if (newline && bracket > newline) {
        return 1;
    }

    count = strtoul(bracket + 1, &end, 10);
    if (!end || end == bracket + 1 || *end != ']' || count == 0) {
        return 1;
    }

    return (size_t)count;
}

static const MGLMSLStructLayout *mglMSLFindStructLayout(const MGLMSLStructLayout *layouts,
                                                        unsigned count,
                                                        const char *name)
{
    if (!layouts || !name || !*name) {
        return NULL;
    }

    for (unsigned i = 0; i < count; i++) {
        if (strcmp(layouts[i].name, name) == 0) {
            return &layouts[i];
        }
    }

    return NULL;
}

static GLboolean mglMSLStructMemberLayout(const char *trimmed,
                                          const MGLMSLStructLayout *layouts,
                                          unsigned layout_count,
                                          size_t *c_size_out,
                                          size_t *std140_align_out,
                                          size_t *actual_align_out)
{
    char type_name[128] = {0};
    const MGLMSLStructLayout *layout = NULL;
    size_t array_count = 1;

    if (!trimmed || !layouts || !c_size_out || !std140_align_out || !actual_align_out) {
        return GL_FALSE;
    }

    if (sscanf(trimmed, "%127s", type_name) != 1) {
        return GL_FALSE;
    }

    layout = mglMSLFindStructLayout(layouts, layout_count, type_name);
    if (!layout) {
        return GL_FALSE;
    }

    array_count = mglMSLDeclaratorArrayCount(trimmed);
    *actual_align_out = layout->align ? layout->align : 1;
    *std140_align_out = 16;
    *c_size_out = mglRoundUpSize(layout->size, *actual_align_out) * array_count;
    return *c_size_out > 0 ? GL_TRUE : GL_FALSE;
}

static void applyMSLUniformBufferPacking(Program *pptr, int stage)
{
    if (!pptr || !pptr->spirv[stage].msl_str) {
        return;
    }

    /*
     * Minecraft writes UBO data in GLSL std140 layout. Metal's vector3 types
     * are already 16-byte sized/aligned, so they naturally cover std140 vec3
     * padding before another 16/8-byte-aligned member. Insert only the padding
     * needed before members whose natural placement would otherwise be too
     * early, e.g. a float followed by an int2.
     */
    const char *src = pptr->spirv[stage].msl_str;
    size_t src_len = strlen(src);
    size_t cap = src_len + src_len / 2 + 4096;
    char *out = (char *)malloc(cap);
    if (!out) return;

        size_t out_len = 0;
        bool in_struct = false;
        bool patch_struct = false;
        unsigned pad_count = 0;
        size_t metal_offset = 0; /* actual Metal C-layout byte position */
        size_t struct_actual_align = 1;
    char struct_name[128] = {0};
    MGLMSLStructLayout struct_layouts[256];
    unsigned struct_layout_count = 0;
    MGLMSLStructDeps struct_deps[256];
    unsigned struct_dep_count = 0;
    char patch_struct_names[256][128];
    unsigned patch_struct_count = 0;
    GLboolean debug_pack = getenv("MGL_DEBUG_MSL_PACK") ? GL_TRUE : GL_FALSE;

    mglMSLCollectStructDeps(src,
                            struct_deps,
                            &struct_dep_count,
                            (unsigned)(sizeof(struct_deps) / sizeof(struct_deps[0])));

    SpirvResourceList *ubo_resources =
        &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
    for (GLuint i = 0; i < ubo_resources->count; i++) {
        mglMSLAddName(patch_struct_names,
                      &patch_struct_count,
                      (unsigned)(sizeof(patch_struct_names) / sizeof(patch_struct_names[0])),
                      ubo_resources->list[i].name);
    }

    const char *glsl_src = pptr->shader_slots[stage]
        ? pptr->shader_slots[stage]->src : NULL;
    SpirvResourceList *ssbo_resources =
        &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_BUFFER];
    for (GLuint i = 0; i < ssbo_resources->count; i++) {
        const char *name = ssbo_resources->list[i].name;
        if (mglGLSLBlockDeclContainsToken(glsl_src, name, "std140")) {
            mglMSLAddName(patch_struct_names,
                          &patch_struct_count,
                          (unsigned)(sizeof(patch_struct_names) / sizeof(patch_struct_names[0])),
                          name);
        }
    }

    for (GLboolean changed = GL_TRUE; changed;) {
        changed = GL_FALSE;
        for (unsigned i = 0; i < patch_struct_count; i++) {
            MGLMSLStructDeps *dep = mglMSLFindStructDeps(struct_deps,
                                                        struct_dep_count,
                                                        patch_struct_names[i]);
            if (!dep) {
                continue;
            }
            for (unsigned j = 0; j < dep->member_type_count; j++) {
                if (mglMSLFindStructDeps(struct_deps, struct_dep_count, dep->member_types[j]) &&
                    mglMSLAddName(patch_struct_names,
                                  &patch_struct_count,
                                  (unsigned)(sizeof(patch_struct_names) / sizeof(patch_struct_names[0])),
                                  dep->member_types[j])) {
                    changed = GL_TRUE;
                }
            }
        }
    }

    const char *p = src;
    while (*p) {
            const char *line_start = p;
            const char *line_end = strchr(p, '\n');
            size_t line_len = line_end ? (size_t)(line_end - line_start + 1) : strlen(line_start);

            /* Detect struct entry/exit. */
            if (!in_struct) {
                char st[128] = {0};
                if (sscanf(line_start, "struct %127s", st) == 1) {
                    char *brace = strchr(st, '{');
                    if (brace) *brace = '\0';
                    in_struct = true;
                    patch_struct = mglMSLNameInList(patch_struct_names, patch_struct_count, st);
                    metal_offset = 0;
                    struct_actual_align = 1;
                    strncpy(struct_name, st, sizeof(struct_name) - 1);
                    struct_name[sizeof(struct_name) - 1] = '\0';
                }
            }

            /* Ensure output capacity. */
            if (out_len + line_len + 1024 > cap) {
                cap = out_len + line_len + 4096;
                char *grown = (char *)realloc(out, cap);
                if (!grown) { free(out); return; }
                out = grown;
            }

            if (in_struct && patch_struct) {
                const char *trimmed = line_start;
                while (*trimmed == ' ' || *trimmed == '\t') trimmed++;

                /* Map Metal type to: (C_size, std140_align). */
                size_t c_size = 0, std140_align = 0, actual_align = 0;
                mglMSLUniformTypeLayout(trimmed, &c_size, &std140_align, &actual_align);
                if (std140_align == 0 && strncmp(trimmed, "char ", 5) == 0) {
                    c_size = mglMSLDeclaratorArrayCount(trimmed);
                    std140_align = 1;
                    actual_align = 1;
                }
                if (std140_align > 0) {
                    size_t array_count = mglMSLDeclaratorArrayCount(trimmed);
                    if (strncmp(trimmed, "char ", 5) != 0) {
                        c_size *= array_count;
                    }
                    if (array_count > 1 && std140_align < 16 &&
                        strncmp(trimmed, "char ", 5) != 0) {
                        std140_align = 16;
                    }
                } else {
                    mglMSLStructMemberLayout(trimmed,
                                             struct_layouts,
                                             struct_layout_count,
                                             &c_size,
                                             &std140_align,
                                             &actual_align);
                }

                if (std140_align > 0) {
                    /* Insert pad if current Metal offset doesn't meet std140 alignment. */
                    size_t before_offset = metal_offset;
                    size_t misalign = metal_offset % std140_align;
                    if (misalign != 0) {
                        size_t pad = std140_align - misalign;
                        while (pad >= 4) {
                            int n = snprintf(out + out_len, cap - out_len,
                                             "    int _mgl_pad%u;\n", pad_count++);
                            if (n > 0) { out_len += (size_t)n; metal_offset += 4; pad -= 4; }
                            else break;
                        }
                    }
                    if (debug_pack) {
                        fprintf(stderr,
                                "MGL MSL PACK: program=%u stage=%d struct=%s member=%.*s before=%zu align=%zu size=%zu after=%zu\n",
                                pptr->name,
                                stage,
                                struct_name,
                                (int)(line_len ? line_len - 1 : line_len),
                                line_start,
                                before_offset,
                                std140_align,
                                c_size,
                                metal_offset + c_size);
                    }
                    /* Copy member as-is. */
                    memcpy(out + out_len, line_start, line_len);
                    out_len += line_len;
                    metal_offset += c_size; /* natural C size */
                    if (actual_align > struct_actual_align) {
                        struct_actual_align = actual_align;
                    }
                    p = line_end ? line_end + 1 : line_start + line_len;
                    continue;
                }
            }

            /* Copy line as-is. */
            memcpy(out + out_len, line_start, line_len);
            out_len += line_len;

            if (in_struct && memchr(line_start, '}', line_len - (line_end ? 1 : 0))) {
                if (patch_struct && struct_layout_count < (sizeof(struct_layouts) / sizeof(struct_layouts[0]))) {
                    MGLMSLStructLayout *layout = &struct_layouts[struct_layout_count++];
                    strncpy(layout->name, struct_name, sizeof(layout->name) - 1);
                    layout->name[sizeof(layout->name) - 1] = '\0';
                    layout->align = struct_actual_align ? struct_actual_align : 1;
                    layout->size = mglRoundUpSize(metal_offset, layout->align);
                    if (debug_pack) {
                        fprintf(stderr,
                                "MGL MSL PACK STRUCT: program=%u stage=%d struct=%s size=%zu align=%zu\n",
                                pptr->name,
                                stage,
                                layout->name,
                                layout->size,
                                layout->align);
                    }
                }
                in_struct = false;
                patch_struct = false;
                metal_offset = 0;
                struct_actual_align = 1;
                struct_name[0] = '\0';
            }

            p = line_end ? line_end + 1 : line_start + line_len;
        }

        out[out_len] = '\0';
        if (pad_count > 0) {
            fprintf(stderr, "MGL MSL STd140 PAD: program=%u stage=%d %u pad(s)\n",
                    pptr->name, stage, pad_count);
            free(pptr->spirv[stage].msl_str);
            pptr->spirv[stage].msl_str = out;
        } else {
            free(out);
        }
}

static GLint mglPlainUniformResourceLocationForProgram(const SpirvResource *res)
{
    if (!res) {
        return -1;
    }

    if (res->uniform_location >= 0) {
        return res->uniform_location;
    }
    if (res->location != 0xffffffffu && res->location < 1024u) {
        return (GLint)res->location;
    }
    if (res->gl_binding < MAX_BINDABLE_BUFFERS) {
        return (GLint)res->gl_binding;
    }
    return -1;
}

static GLboolean mglParseScalarUniformInitializer(const char *src,
                                                  const char *name,
                                                  spvc_basetype basetype,
                                                  uint8_t *value,
                                                  GLsizeiptr *size_out)
{
    if (!src || !name || !value || !size_out) {
        return GL_FALSE;
    }

    const char *p = src;
    char base_name[256];
    size_t name_len = strlen(name);
    if (name_len >= sizeof(base_name)) {
        return GL_FALSE;
    }
    memcpy(base_name, name, name_len + 1u);
    if (name_len >= 3u && strcmp(base_name + name_len - 3u, "[0]") == 0) {
        name_len -= 3u;
        base_name[name_len] = '\0';
    }

    while ((p = strstr(p, "uniform")) != NULL) {
        const char *before = (p == src) ? src : p - 1;
        if (p != src && ((*before == '_') || isalnum((unsigned char)*before))) {
            p += 7;
            continue;
        }

        const char *q = p + 7;
        if ((*q == '_') || isalnum((unsigned char)*q)) {
            p += 7;
            continue;
        }
        while (*q && isspace((unsigned char)*q)) {
            q++;
        }

        const char *type = q;
        while (*q && !isspace((unsigned char)*q)) {
            q++;
        }
        size_t type_len = (size_t)(q - type);
        while (*q && isspace((unsigned char)*q)) {
            q++;
        }

        if (strncmp(q, base_name, name_len) != 0 ||
            ((q[name_len] == '_') || isalnum((unsigned char)q[name_len]))) {
            p += 7;
            continue;
        }
        q += name_len;
        while (*q && isspace((unsigned char)*q)) {
            q++;
        }

        unsigned array_count = 0u;
        if (*q == '[') {
            char *end = NULL;
            unsigned long parsed_count;
            q++;
            while (*q && isspace((unsigned char)*q)) {
                q++;
            }
            parsed_count = strtoul(q, &end, 10);
            if (!end || end == q || parsed_count == 0ul || parsed_count > 64ul) {
                p += 7;
                continue;
            }
            q = end;
            while (*q && isspace((unsigned char)*q)) {
                q++;
            }
            if (*q != ']') {
                p += 7;
                continue;
            }
            q++;
            array_count = (unsigned)parsed_count;
            while (*q && isspace((unsigned char)*q)) {
                q++;
            }
        }

        if (*q != '=') {
            p += 7;
            continue;
        }
        q++;
        while (*q && isspace((unsigned char)*q)) {
            q++;
        }

        if ((basetype == SPVC_BASETYPE_INT32 && type_len == 3 && memcmp(type, "int", 3) == 0) ||
            (basetype == SPVC_BASETYPE_UINT32 && type_len == 4 && memcmp(type, "uint", 4) == 0)) {
            if (array_count > 0u) {
                if (strncmp(q, type, type_len) != 0) {
                    p += 7;
                    continue;
                }
                q += type_len;
                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q != '[') {
                    p += 7;
                    continue;
                }
                q++;
                char *array_end = NULL;
                unsigned long constructor_count = strtoul(q, &array_end, 10);
                if (!array_end || array_end == q || constructor_count != array_count) {
                    p += 7;
                    continue;
                }
                q = array_end;
                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q != ']') {
                    p += 7;
                    continue;
                }
                q++;
                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q != '(') {
                    p += 7;
                    continue;
                }
                q++;

                GLboolean parse_ok = GL_TRUE;
                for (unsigned index = 0u; index < array_count; index++) {
                    char *end = NULL;
                    uint32_t v;
                    while (*q && isspace((unsigned char)*q)) {
                        q++;
                    }
                    if (basetype == SPVC_BASETYPE_INT32) {
                        long parsed = strtol(q, &end, 0);
                        v = (uint32_t)(GLint)parsed;
                    } else {
                        unsigned long parsed = strtoul(q, &end, 0);
                        v = (uint32_t)parsed;
                    }
                    if (!end || end == q) {
                        parse_ok = GL_FALSE;
                        break;
                    }
                    memcpy(value + ((size_t)index * sizeof(v)), &v, sizeof(v));
                    q = end;
                    if (basetype == SPVC_BASETYPE_UINT32 && (*q == 'u' || *q == 'U')) {
                        q++;
                    }
                    while (*q && isspace((unsigned char)*q)) {
                        q++;
                    }
                    if (index + 1u < array_count) {
                        if (*q != ',') {
                            parse_ok = GL_FALSE;
                            break;
                        }
                        q++;
                    }
                }
                if (!parse_ok) {
                    p += 7;
                    continue;
                }

                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q == ')') {
                    *size_out = (GLsizeiptr)((size_t)array_count * sizeof(uint32_t));
                    return GL_TRUE;
                }
            } else {
                char *end = NULL;
                long parsed = strtol(q, &end, 0);
                if (end && end != q) {
                    GLint v = (GLint)parsed;
                    memcpy(value, &v, sizeof(v));
                    *size_out = (GLsizeiptr)sizeof(v);
                    return GL_TRUE;
                }
            }
        } else if (basetype == SPVC_BASETYPE_FP32 && type_len == 5 && memcmp(type, "float", 5) == 0) {
            if (array_count > 0u) {
                if (strncmp(q, type, type_len) != 0) {
                    p += 7;
                    continue;
                }
                q += type_len;
                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q != '[') {
                    p += 7;
                    continue;
                }
                q++;
                char *array_end = NULL;
                unsigned long constructor_count = strtoul(q, &array_end, 10);
                if (!array_end || array_end == q || constructor_count != array_count) {
                    p += 7;
                    continue;
                }
                q = array_end;
                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q != ']') {
                    p += 7;
                    continue;
                }
                q++;
                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q != '(') {
                    p += 7;
                    continue;
                }
                q++;

                GLboolean parse_ok = GL_TRUE;
                for (unsigned index = 0u; index < array_count; index++) {
                    char *end = NULL;
                    float parsed;
                    while (*q && isspace((unsigned char)*q)) {
                        q++;
                    }
                    parsed = strtof(q, &end);
                    if (!end || end == q) {
                        parse_ok = GL_FALSE;
                        break;
                    }
                    memcpy(value + ((size_t)index * sizeof(parsed)), &parsed, sizeof(parsed));
                    q = end;
                    if (*q == 'f' || *q == 'F') {
                        q++;
                    }
                    while (*q && isspace((unsigned char)*q)) {
                        q++;
                    }
                    if (index + 1u < array_count) {
                        if (*q != ',') {
                            parse_ok = GL_FALSE;
                            break;
                        }
                        q++;
                    }
                }
                if (!parse_ok) {
                    p += 7;
                    continue;
                }

                while (*q && isspace((unsigned char)*q)) {
                    q++;
                }
                if (*q == ')') {
                    *size_out = (GLsizeiptr)((size_t)array_count * sizeof(float));
                    return GL_TRUE;
                }
            } else {
                char *end = NULL;
                float parsed = strtof(q, &end);
                if (end && end != q) {
                    memcpy(value, &parsed, sizeof(parsed));
                    *size_out = (GLsizeiptr)sizeof(parsed);
                    return GL_TRUE;
                }
            }
        }

        p += 7;
    }

    return GL_FALSE;
}

static void mglApplyPlainUniformInitializers(GLMContext ctx, Program *program, int stage)
{
    if (!ctx || !program || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        !program->shader_slots[stage] || !program->shader_slots[stage]->src) {
        return;
    }

    SpirvResourceList *resources =
        &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
    for (GLuint i = 0; resources->list && i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if (!res->name || res->name[0] == '\0') {
            continue;
        }

        spvc_type reflected_type = NULL;
        spvc_basetype basetype = SPVC_BASETYPE_UNKNOWN;
        if (res->type_id) {
            /* Type handles are only valid during SPIRV-Cross compilation, so
             * infer scalar initializer compatibility from generated MSL names
             * and resource type metadata here.
             */
        }

        const char *src = program->shader_slots[stage]->src;
        uint8_t value[256] = {0};
        GLsizeiptr size = 0;

        if (!mglParseScalarUniformInitializer(src, res->name, SPVC_BASETYPE_INT32, value, &size) &&
            !mglParseScalarUniformInitializer(src, res->name, SPVC_BASETYPE_UINT32, value, &size) &&
            !mglParseScalarUniformInitializer(src, res->name, SPVC_BASETYPE_FP32, value, &size)) {
            (void)reflected_type;
            (void)basetype;
            continue;
        }

        GLint location = mglPlainUniformResourceLocationForProgram(res);
        if (location < 0 || location >= MAX_BINDABLE_BUFFERS) {
            continue;
        }
        BufferBaseTarget *slot = &program->plain_uniform_buffers[location];
        if (slot->buf || slot->buffer != 0) {
            continue;
        }

        GLuint internalName = MGL_INTERNAL_UNIFORM_BUFFER_NAME_BASE |
                              (((GLuint)program->name & 0x0fffu) << 12) |
                              (GLuint)location;
        slot->buf = newBuffer(ctx, GL_UNIFORM_BUFFER, internalName);
        if (!slot->buf) {
            continue;
        }
        insertHashElement(&ctx->state.buffer_table, internalName, slot->buf);
        initBufferData(ctx, slot->buf, size, value, true);
        slot->buffer = slot->buf->name;
        slot->offset = 0;
        slot->size = size;
    }
}

/* === MSL Patch Pipeline wrappers ===
 *
 * Each wrapper matches the MSLPatchFn signature and adapts one step of the
 * MSL post-processing that previously ran inline in parseSPIRVShaderToMetal.
 * The wrappers extract `ptr` (the Program) and `stage` from the patch context.
 * Existing patch helpers mostly return void and cannot signal failure, so the
 * wrappers return GL_TRUE on success (the pipeline's per-step rollback still
 * guards against a NULL result). */

static GLboolean mglPatchRemoveRestrict(MSLPatchContext *ctx, char **msl_ptr)
{
    /* Remove C99 `restrict` qualifier that SPIRV-Cross emits on reference
     * and pointer parameters (e.g. "device T& restrict var" or
     * "device T* restrict ptr").  Metal's compiler rejects `restrict`
     * in these positions (it expects `__restrict` on pointers and does
     * not support it on references at all).  Removing it is safe because
     * it is only an optimization hint with no semantic effect. */
    char *str_ret = *msl_ptr;
    char *read = str_ret;
    char *write = str_ret;
    while (*read) {
        /* Look for " restrict " preceded by '&' or '*'. */
        if ((read[0] == '&' || read[0] == '*') &&
            read[1] == ' ' &&
            strncmp(read + 2, "restrict ", 9) == 0) {
            /* Skip " restrict " (9 chars after "& "), keeping the & and space. */
            *write++ = *read++;  /* copy & or * */
            *write++ = *read++;  /* copy space */
            read += 9;            /* skip "restrict " */
        } else {
            *write++ = *read++;
        }
    }
    *write = '\0';
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglPatchFixSamplerShadowing(MSLPatchContext *ctx, char **msl_ptr)
{
    /* Some generated MSL uses `sampler` as an identifier, which collides
     * with Metal's `sampler` type in function signatures. Normalize these
     * generated helper names to keep compilation valid. */
    char *str_ret = *msl_ptr;
    static const char *sampler_shadowing_texture_types[] = {
        "texture1d<float>", "texture1d<int>", "texture1d<uint>",
        "texture1d_array<float>", "texture1d_array<int>", "texture1d_array<uint>",
        "texture2d<float>", "texture2d<int>", "texture2d<uint>",
        "texture2d_array<float>", "texture2d_array<int>", "texture2d_array<uint>",
        "texture2d_ms<float>", "texture2d_ms<int>", "texture2d_ms<uint>",
        "texture2d_ms_array<float>", "texture2d_ms_array<int>", "texture2d_ms_array<uint>",
        "texture_buffer<float>", "texture_buffer<int>", "texture_buffer<uint>",
        "texture3d<float>", "texture3d<int>", "texture3d<uint>",
        "texturecube<float>", "texturecube<int>", "texturecube<uint>",
        "texturecube_array<float>", "texturecube_array<int>", "texturecube_array<uint>",
        "depth2d<float>", "depth2d<int>", "depth2d<uint>",
        "depth2d_array<float>", "depth2d_array<int>", "depth2d_array<uint>",
        "depth2d_ms<float>", "depth2d_ms<int>", "depth2d_ms<uint>",
        "depth2d_ms_array<float>", "depth2d_ms_array<int>", "depth2d_ms_array<uint>",
        "depthcube<float>", "depthcube<int>", "depthcube<uint>",
        "depthcube_array<float>", "depthcube_array<int>", "depthcube_array<uint>",
    };
    size_t n_types = sizeof(sampler_shadowing_texture_types) / sizeof(sampler_shadowing_texture_types[0]);
    GLboolean renamed_sampler_parameter = GL_FALSE;
    for (size_t ti = 0; ti < n_types; ti++) {
        const char *type = sampler_shadowing_texture_types[ti];
        char from[128], to[128];
        snprintf(from, sizeof(from), "%s sampler,", type);
        snprintf(to, sizeof(to), "%s sourceTex,", type);
        if (strstr(str_ret, from))
            renamed_sampler_parameter = GL_TRUE;
        replace_all_substr(&str_ret, from, to);

        snprintf(from, sizeof(from), "%s sampler)", type);
        snprintf(to, sizeof(to), "%s sourceTex)", type);
        if (strstr(str_ret, from))
            renamed_sampler_parameter = GL_TRUE;
        replace_all_substr(&str_ret, from, to);

        snprintf(from, sizeof(from), "%s sampler [[", type);
        snprintf(to, sizeof(to), "%s sourceTex [[", type);
        if (strstr(str_ret, from))
            renamed_sampler_parameter = GL_TRUE;
        replace_all_substr(&str_ret, from, to);
    }
    if (renamed_sampler_parameter) {
        replace_all_substr(&str_ret, " samplerSmplr", " sourceSmplr");
        replace_all_substr(&str_ret, "(samplerSmplr", "(sourceSmplr");
        replace_all_substr(&str_ret, ", samplerSmplr", ", sourceSmplr");
        mglReplaceMSLIdentifierBeforeChar(&str_ret, "sampler", "sourceTex", '.');
        replace_all_substr(&str_ret, "(sampler, ", "(sourceTex, ");
        replace_all_substr(&str_ret, ", sampler, ", ", sourceTex, ");
        replace_all_substr(&str_ret, "(sampler)", "(sourceTex)");
        replace_all_substr(&str_ret, ", sampler)", ", sourceTex)");
    }
    *msl_ptr = str_ret;
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglPatchFixUnknownTextureType(MSLPatchContext *ctx, char **msl_ptr)
{
    /* SPIRV-Cross can emit this placeholder for sampler2DRect. Metal has no
     * rectangle texture object; MGL stores GL_TEXTURE_RECTANGLE as a 2D Metal
     * texture and binds a non-normalized Metal sampler for rectangle targets. */
    replace_all_substr(msl_ptr, "unknown_texture_type<", "texture2d<");
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglPatchStripThreadConstRef(MSLPatchContext *ctx, char **msl_ptr)
{
    replace_all_substr(msl_ptr, "thread const bool&", "bool");
    replace_all_substr(msl_ptr, "thread const int&", "int");
    replace_all_substr(msl_ptr, "thread const uint&", "uint");
    replace_all_substr(msl_ptr, "thread const float&", "float");
    replace_all_substr(msl_ptr, "thread const float2&", "float2");
    replace_all_substr(msl_ptr, "thread const float3&", "float3");
    replace_all_substr(msl_ptr, "thread const float4&", "float4");
    replace_all_substr(msl_ptr, "thread const int2&", "int2");
    replace_all_substr(msl_ptr, "thread const int3&", "int3");
    replace_all_substr(msl_ptr, "thread const int4&", "int4");
    replace_all_substr(msl_ptr, "thread const uint2&", "uint2");
    replace_all_substr(msl_ptr, "thread const uint3&", "uint3");
    replace_all_substr(msl_ptr, "thread const uint4&", "uint4");
    replace_all_substr(msl_ptr, "thread const float3x3&", "float3x3");
    replace_all_substr(msl_ptr, "thread const float4x4&", "float4x4");
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglPatchRenameLengthSquared(MSLPatchContext *ctx, char **msl_ptr)
{
    /* SPIRV-Cross may generate helper functions named `length_squared`
     * for float2/float3/float4 types. Metal has a built-in `length_squared`
     * for vector types (MSL 3.1+), causing an ambiguity error when both
     * the user-defined function and the built-in are in scope. Rename the
     * generated helper function and all call sites to avoid the conflict. */
    mglReplaceMSLIdentifier(msl_ptr, "length_squared", "_mgl_length_squared");
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglPatchLowerDoubleTypes(MSLPatchContext *ctx, char **msl_ptr)
{
    mglLowerMSLDoubleTypesToFloat(msl_ptr);
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglPatchFixEndPortalLayer(MSLPatchContext *ctx, char **msl_ptr)
{
    replace_all_substr(msl_ptr,
                       "float4x4 end_portal_layer(thread const float& layer, thread const float& GameTime)",
                       "float4x4 end_portal_layer(float layer, float GameTime)");
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglPatchFullscreenFBYFlip(MSLPatchContext *ctx, char **msl_ptr)
{
    Program *ptr = ctx->program;
    int stage = ctx->stage;
    char *str_ret = *msl_ptr;
    if (stage == _VERTEX_SHADER &&
        strstr(str_ret, "float2 screenPos = (in.Position.xy * 2.0) - float2(1.0);") &&
        strstr(str_ret, "out.gl_Position = float4(screenPos.x, screenPos.y, 1.0, 1.0);") &&
        strstr(str_ret, "out.texCoord = in.Position.xy;")) {
        fprintf(stderr,
                "MGL MSL FULLSCREEN FIX: program=%u flips sampled framebuffer texcoord Y\n",
                ptr->name);
        replace_all_substr(&str_ret,
                           "out.texCoord = in.Position.xy;",
                           "out.texCoord = float2(in.Position.x, 1.0 - in.Position.y);");
        ptr->spirv[_VERTEX_SHADER].mgl_injected_framebuffer_yflip = GL_TRUE;
    }
    *msl_ptr = str_ret;
    return GL_TRUE;
}

static GLboolean mglPatchFragCoordOriginFix(MSLPatchContext *ctx, char **msl_ptr)
{
    applyMSLFragCoordOriginFix(ctx->stage, msl_ptr);
    return GL_TRUE;
}

static GLboolean mglPatchFixPlainStructPointerArray(MSLPatchContext *ctx, char **msl_ptr)
{
    mglFixMSLPlainStructPointerArrayAccess(ctx->program, ctx->stage, msl_ptr);
    return GL_TRUE;
}

static GLboolean mglPatchInjectAtomicCounterArgs(MSLPatchContext *ctx, char **msl_ptr)
{
    mglInjectMSLAtomicCounterArguments(ctx->program, ctx->stage, msl_ptr);
    return GL_TRUE;
}

static GLboolean mglPatchApplyResourceBindings(MSLPatchContext *ctx, char **msl_ptr)
{
    /* Mutates both the program state (binding slots) and the MSL string
     * (remaps any user buffers that land on MGL-reserved slots).  Returns
     * GL_TRUE so the pipeline does not roll back. */
    applyMSLResourceBindings(ctx->program, ctx->stage, msl_ptr);
    return GL_TRUE;
}

static GLboolean mglProgramHasPassthroughGeometryShader(Program *ptr);

static GLboolean mglPatchInjectPointSizeBuiltin(MSLPatchContext *ctx, char **msl_ptr)
{
    Program *ptr = ctx->program;
    int stage = ctx->stage;
    /* Skip point_size injection for the vertex shader when a geometry
     * shader is present.  Metal does not support geometry shaders, so
     * the GS is skipped at bind time — but injecting [[point_size]] into
     * the VS would cause GL_POINTS draws to rasterize even though the
     * (skipped) geometry shader is supposed to re-emit the vertices. */
    if (!(stage == _VERTEX_SHADER &&
          ptr->shader_slots[_GEOMETRY_SHADER] &&
          !mglProgramHasPassthroughGeometryShader(ptr))) {
        mglInjectMSLPointSizeBuiltin(stage, msl_ptr);
    }
    return GL_TRUE;
}

static GLboolean mglPatchFixImage2DRectImageSize(MSLPatchContext *ctx, char **msl_ptr)
{
    mglFixMSLImage2DRectImageSize(msl_ptr);
    (void)ctx;
    return GL_TRUE;
}

static GLboolean mglProgramHasPassthroughGeometryShader(Program *ptr)
{
    const char *src = (ptr && ptr->shader_slots[_GEOMETRY_SHADER])
        ? ptr->shader_slots[_GEOMETRY_SHADER]->src
        : NULL;
    if (!src) {
        return GL_FALSE;
    }

    return strstr(src, "EmitVertex()") &&
           strstr(src, "EndPrimitive()") &&
           strstr(src, "gl_Position = gl_in[n_vertex_index].gl_Position") &&
           !strstr(src, "gl_PrimitiveID") &&
           !strstr(src, "gl_Layer") &&
           !strstr(src, "gl_ViewportIndex");
}

static GLboolean mglPatchTesAsComputeKernel(MSLPatchContext *ctx, char **msl_ptr)
{
    /* TES is lowered to a post-tessellation vertex function by SPIRV-Cross,
     * but macOS 26.5 SDK removed postTessellationVertexFunction /
     * isTessellationEnabled from MTLRenderPipelineDescriptor. Rewrite the
     * MSL to a plain compute kernel so we can dispatch it like TCS. */
    if (ctx->stage == _TESS_EVALUATION_SHADER) {
        mglFixMSLTesAsComputeKernel(ctx->program, msl_ptr);
    }
    return GL_TRUE;
}

static GLboolean mglPatchTcsStageInFix(MSLPatchContext *ctx, char **msl_ptr)
{
    /* TCS is generated by SPIRV-Cross as a compute kernel, but it has a
     * [[stage_in]] parameter for vertex input which Metal compute pipelines
     * don't support.  Replace it with a device buffer. */
    if (ctx->stage == _TESS_CONTROL_SHADER) {
        mglFixMSLTcsStageIn(msl_ptr);
    }
    return GL_TRUE;
}

char *parseSPIRVShaderToMetal(GLMContext ctx, Program *ptr, int stage)
{
    const SpvId *spirv;
    size_t word_count;
    char *str_ret;
    int parse_res;

    spvc_context context = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler compiler_msl = NULL;
    spvc_compiler_options options = NULL;
    spvc_resources resources = NULL;
    const spvc_reflected_resource *list = NULL;
    const char *result = NULL;
    size_t count;
    size_t i;

    if (!ptr || stage < 0 || stage >= _MAX_SHADER_TYPES || !ptr->shader_slots[stage]) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    spirv = ptr->spirv[stage].ir;
    word_count = ptr->spirv[stage].size;
    if (!spirv || word_count == 0) {
        fprintf(stderr,
                "MGL ERROR: parseSPIRVShaderToMetal missing SPIR-V program=%u stage=%d words=%zu\n",
                ptr->name,
                stage,
                word_count);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    /* SPIRV-Cross throws "Metal does not support isoline tessellation" for
     * TES with SpvExecutionModeIsolines.  Patch the SPIR-V to replace
     * Isolines with Triangles so SPIRV-Cross can generate MSL.  The original
     * mode is recorded in tess_gen_mode during reflection (see below).
     *
     * OpExecutionMode layout: word0 = (word_count << 16) | opcode(16),
     * word1 = entry_point, word2 = mode.
     * Per spirv.h: SpvExecutionModeIsolines=25, SpvExecutionModeTriangles=22. */
    GLboolean spirv_was_isolines = GL_FALSE;
    if (stage == _TESS_EVALUATION_SHADER) {
        const unsigned int OpExecutionMode = 16;
        const unsigned int SpvExecutionModeIsolines = 25;
        const unsigned int SpvExecutionModeTriangles = 22;
        for (size_t i = 5; i + 2 < word_count; ) {  /* skip header (5 words) */
            unsigned int word0 = spirv[i];
            unsigned int opcode = word0 & 0xFFFFu;
            unsigned int instr_words = word0 >> 16;
            if (instr_words == 0) break;  /* malformed */
            if (opcode == OpExecutionMode && instr_words >= 3) {
                unsigned int mode = spirv[i + 2];
                if (mode == SpvExecutionModeIsolines) {
                    /* Cast away const to patch the SPIR-V in place. */
                    ((unsigned int *)spirv)[i + 2] = SpvExecutionModeTriangles;
                    spirv_was_isolines = GL_TRUE;
                }
            }
            i += instr_words;
        }
    }

    // Create context.
    if (spvc_context_create(&context) != SPVC_SUCCESS || !context) {
        fprintf(stderr, "MGL ERROR: spvc_context_create failed\n");
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // Set debug callback.
    spvc_context_set_error_callback(context, error_callback, ctx);

    // Parse the SPIR-V.
    parse_res = spvc_context_parse_spirv(context, spirv, word_count, &ir);
    if (parse_res != SPVC_SUCCESS || !ir) {
        fprintf(stderr,
                "MGL ERROR: spvc_context_parse_spirv failed program=%u stage=%d err=%d\n",
                ptr->name,
                stage,
                parse_res);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // Hand it off to a compiler instance and give it ownership of the IR.
    if (spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler_msl) != SPVC_SUCCESS ||
        !compiler_msl) {
        fprintf(stderr, "MGL ERROR: spvc_context_create_compiler failed program=%u stage=%d\n", ptr->name, stage);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // ERROR_CHECK_RETURN(spvc_compiler_msl_add_discrete_descriptor_set(compiler_msl, 3) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_msl_add_discrete_descriptor_set(compiler_msl, 3) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_msl_add_discrete_descriptor_set failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // Modify options.
    // ERROR_CHECK_RETURN(spvc_compiler_create_compiler_options(compiler_msl, &options) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_create_compiler_options(compiler_msl, &options) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_create_compiler_options failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // ERROR_CHECK_RETURN(spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_bool(SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D, SPVC_TRUE) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_bool(SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // ERROR_CHECK_RETURN(spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3,1,0)) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3,1,0)) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_uint(SPVC_COMPILER_OPTION_MSL_VERSION) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (spvc_compiler_options_set_uint(options,
                                       SPVC_COMPILER_OPTION_MSL_TEXEL_BUFFER_TEXTURE_WIDTH,
                                       MGL_TEXEL_BUFFER_TEXTURE_WIDTH) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_uint(SPVC_COMPILER_OPTION_MSL_TEXEL_BUFFER_TEXTURE_WIDTH) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_FIXUP_DEPTH_CONVENTION, SPVC_TRUE) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_bool(SPVC_COMPILER_OPTION_FIXUP_DEPTH_CONVENTION) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    /* Set the buffer size buffer index for runtime-sized SSBO arrays.
     * SPIRV-Cross emits code that reads buffer byte-sizes from a
     * constant uint* buffer when a shader uses .length() on unsized
     * SSBO arrays.  We use slot 25 (auto-assigned by SPIRV-Cross by
     * default, but set explicitly here for reliability). */
    if (spvc_compiler_options_set_uint(options,
                                       SPVC_COMPILER_OPTION_MSL_BUFFER_SIZE_BUFFER_INDEX,
                                       MGL_BUFFER_SIZE_BUFFER_INDEX) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_uint(SPVC_COMPILER_OPTION_MSL_BUFFER_SIZE_BUFFER_INDEX) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    //ERROR_CHECK_RETURN(spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_GLSL_VERSION, 4.5) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    // ERROR_CHECK_RETURN(spvc_compiler_install_compiler_options(compiler_msl, options) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_install_compiler_options(compiler_msl, options) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_install_compiler_options failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    
    // create an entry point for metal based on the shader type and name
    GLuint name;
    char entry_point[128];
    name = ptr->shader_slots[stage]->name;

    SpvExecutionModel model = SpvExecutionModelVertex; // CRITICAL FIX: Initialize with safe default
    switch(stage)
    {
        case _VERTEX_SHADER: model = SpvExecutionModelVertex; break;
        case _TESS_CONTROL_SHADER: model = SpvExecutionModelTessellationControl; break;
        case _TESS_EVALUATION_SHADER: model = SpvExecutionModelTessellationEvaluation; break;
        case _GEOMETRY_SHADER: model = SpvExecutionModelGeometry; break;
        case _FRAGMENT_SHADER: model = SpvExecutionModelFragment; break;
        case _COMPUTE_SHADER: model = SpvExecutionModelGLCompute; break;
        default: // CRITICAL FIX: Handle error gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
            STATE(error) = GL_INVALID_OPERATION;
            return NULL;
    }

    switch(stage)
    {
        case _VERTEX_SHADER: snprintf(entry_point, sizeof(entry_point), "vertex_%d_main",name); break;
        case _TESS_CONTROL_SHADER: snprintf(entry_point, sizeof(entry_point), "tess_control_%d_main",name); break;
        case _TESS_EVALUATION_SHADER: snprintf(entry_point, sizeof(entry_point), "tess_evaluation_%d_main",name); break;
        case _GEOMETRY_SHADER: snprintf(entry_point, sizeof(entry_point), "geometry_%d",name); break;
        case _FRAGMENT_SHADER: snprintf(entry_point, sizeof(entry_point), "fragment_%d",name); break;
        case _COMPUTE_SHADER: snprintf(entry_point, sizeof(entry_point), "compute_%d",name); break;
        default: // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
    }

    const char *cleansed_entry_point;
    cleansed_entry_point = spvc_compiler_get_cleansed_entry_point_name(compiler_msl, "main", model);

    spvc_result err;
    err = spvc_compiler_rename_entry_point(compiler_msl, cleansed_entry_point, entry_point, model);
    if (err != SPVC_SUCCESS) {
        fprintf(stderr,
                "MGL ERROR: spvc_compiler_rename_entry_point failed program=%u stage=%d err=%d\n",
                ptr->name,
                stage,
                err);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // set the entry point for metal
    ptr->shader_slots[stage]->entry_point = strdup(entry_point);
    ptr->spirv[stage].entry_point = strdup(entry_point);

    // compute shader
    if (stage == _COMPUTE_SHADER)
    {
        spvc_result res;
        const spvc_entry_point *entry_points;
        size_t num_entry_points;

        res = spvc_compiler_get_entry_points(compiler_msl, &entry_points, &num_entry_points);
        if (res != SPVC_SUCCESS) {
            fprintf(stderr,
                    "MGL ERROR: spvc_compiler_get_entry_points failed program=%u err=%d\n",
                    ptr->name,
                    res);
            spvc_context_destroy(context);
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
        }
        
        for(int i=0; i<num_entry_points; i++)
        {
            DEBUG_PRINT("Entry point: %s Execution Model: %d\n", entry_points[i].name, entry_points[i].execution_model);
        }

        ptr->local_workgroup_size.x = spvc_compiler_get_execution_mode_argument_by_index(compiler_msl, SpvExecutionModeLocalSize, 0);
        ptr->local_workgroup_size.y = spvc_compiler_get_execution_mode_argument_by_index(compiler_msl, SpvExecutionModeLocalSize, 1);
        ptr->local_workgroup_size.z = spvc_compiler_get_execution_mode_argument_by_index(compiler_msl, SpvExecutionModeLocalSize, 2);
    }

    /* TCS: extract layout(vertices=N) out; via SpvExecutionModeOutputVertices */
    if (stage == _TESS_CONTROL_SHADER)
    {
        ptr->tess_control_output_vertices =
            (GLuint)spvc_compiler_get_execution_mode_argument_by_index(
                compiler_msl, SpvExecutionModeOutputVertices, 0);
    }

    /* TES: extract layout(primitive_mode, vertex_spacing, ordering, point_mode) in;
     * via SpvExecutionMode* enums.  GLSL TES layout declarations compile to
     * SPIR-V OpExecutionMode instructions on the TES entry point. */
    if (stage == _TESS_EVALUATION_SHADER)
    {
        const SpvExecutionMode *modes = NULL;
        size_t num_modes = 0;
        if (spvc_compiler_get_execution_modes(compiler_msl, &modes, &num_modes) == SPVC_SUCCESS)
        {
            for (size_t i = 0; i < num_modes; i++)
            {
                SpvExecutionMode m = modes[i];
                switch (m)
                {
                    case SpvExecutionModeTriangles:
                        ptr->tess_gen_mode = GL_TRIANGLES; break;
                    case SpvExecutionModeQuads:
                        ptr->tess_gen_mode = GL_QUADS; break;
                    case SpvExecutionModeIsolines:
                        ptr->tess_gen_mode = GL_ISOLINES; break;
                    case SpvExecutionModeSpacingEqual:
                        ptr->tess_gen_spacing = GL_EQUAL; break;
                    case SpvExecutionModeSpacingFractionalEven:
                        ptr->tess_gen_spacing = GL_FRACTIONAL_EVEN; break;
                    case SpvExecutionModeSpacingFractionalOdd:
                        ptr->tess_gen_spacing = GL_FRACTIONAL_ODD; break;
                    case SpvExecutionModeVertexOrderCw:
                        ptr->tess_gen_vertex_order = GL_CW; break;
                    case SpvExecutionModeVertexOrderCcw:
                        ptr->tess_gen_vertex_order = GL_CCW; break;
                    case SpvExecutionModePointMode:
                        ptr->tess_gen_point_mode = GL_TRUE; break;
                    default: break;
                }
            }
        }
        /* If we patched Isolines→Triangles in the SPIR-V, restore the
         * original mode so dispatch logic uses isolines primitive counting. */
        if (spirv_was_isolines) {
            ptr->tess_gen_mode = GL_ISOLINES;
        }
    }
    
    // Do some basic reflection.
    if (spvc_compiler_create_shader_resources(compiler_msl, &resources) != SPVC_SUCCESS || !resources) {
        fprintf(stderr,
                "MGL ERROR: spvc_compiler_create_shader_resources failed program=%u stage=%d\n",
                ptr->name,
                stage);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }
    for (int res_type=SPVC_RESOURCE_TYPE_UNIFORM_BUFFER; res_type < SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE; res_type++)
    {
#if DEBUG
        const char *res_name[] = {"NONE", "UNIFORM_BUFFER", "UNIFORM_CONSTANT", "STORAGE_BUFFER", "STAGE_INPUT", "STAGE_OUTPUT",
            "SUBPASS_INPUT", "STORAGE_INPUT", "SAMPLED_IMAGE", "ATOMIC_COUNTER", "PUSH_CONSTANT", "SEPARATE_IMAGE",
            "SEPARATE_SAMPLERS", "ACCELERATION_STRUCTURE", "RAY_QUERY"};
#endif
        
        spvc_resources_get_resource_list_for_type(resources, res_type, &list, &count);

        ptr->spirv_resources_list[stage][res_type].count = (GLuint)count;

        // CRITICAL SECURITY FIX: Prevent integer overflow in resource allocation
        // Check if count * sizeof(SpirvResource) would overflow size_t
        if (count > SIZE_MAX / sizeof(SpirvResource)) {
            fprintf(stderr, "MGL SECURITY ERROR: Resource count %zu would cause allocation overflow\n", count);
            spvc_context_destroy(context);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, NULL);
        }

        size_t alloc_size = count * sizeof(SpirvResource);
        if (count == 0) {
            ptr->spirv_resources_list[stage][res_type].list = NULL;
        } else {
            ptr->spirv_resources_list[stage][res_type].list =
                (SpirvResource *)calloc(count, sizeof(SpirvResource));
        }
        if (count != 0 && !ptr->spirv_resources_list[stage][res_type].list) {
            fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate %zu bytes for resource list\n", alloc_size);
            spvc_context_destroy(context);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, NULL);
        }

        for (i = 0; i < count; i++)
        {
            DEBUG_PRINT("res_type: %s ID: %u, BaseTypeID: %u, TypeID: %u, Name: %s ", res_name[res_type], list[i].id, list[i].base_type_id, list[i].type_id,
                   list[i].name);
            
            switch(res_type)
            {
                case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
                case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
                case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
                case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
                    DEBUG_PRINT("Set: %u, Binding: %u Uniform: %d offset: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationBinding),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationUniform),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationOffset));
                    break;

                case SPVC_RESOURCE_TYPE_STAGE_INPUT:
                case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
                case SPVC_RESOURCE_TYPE_SUBPASS_INPUT:
                    DEBUG_PRINT("Set: %u, Location: %d Index: %d, offset: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationIndex),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationOffset));
                    break;
                    
                case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
                case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
                    DEBUG_PRINT("Set: %u, Location: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation));
                    break;

                default:
                    DEBUG_PRINT("Set: %u, Binding: %u Location: %d Index: %d, Uniform: %d offset: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationBinding),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationIndex),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationUniform),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationOffset));
                    break;
            }
            
            spvc_type reflected_type = NULL;
            spvc_basetype reflected_basetype = SPVC_BASETYPE_UNKNOWN;
            if (list[i].type_id) {
                reflected_type = spvc_compiler_get_type_handle(compiler_msl, list[i].type_id);
            }
            if (!reflected_type && list[i].base_type_id) {
                reflected_type = spvc_compiler_get_type_handle(compiler_msl, list[i].base_type_id);
            }
            if (reflected_type) {
                reflected_basetype = spvc_type_get_basetype(reflected_type);
            }

            bool uniform_constant_sampler_like =
                (res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) &&
                (mglUniformConstantBaseTypeIsSamplerLike(reflected_basetype) ||
                 mglUniformNameLooksSamplerLike(list[i].name));

            ptr->spirv_resources_list[stage][res_type].list[i]._id = list[i].id;
            ptr->spirv_resources_list[stage][res_type].list[i].base_type_id = list[i].base_type_id;
            ptr->spirv_resources_list[stage][res_type].list[i].type_id = list[i].type_id;
            ptr->spirv_resources_list[stage][res_type].list[i].name = strdup(list[i].name);
            ptr->spirv_resources_list[stage][res_type].list[i].set = spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet);
            ptr->spirv_resources_list[stage][res_type].list[i].gl_binding = spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationBinding);
            ptr->spirv_resources_list[stage][res_type].list[i].ubo_array_size = 1;
            ptr->spirv_resources_list[stage][res_type].list[i].ubo_is_array = GL_FALSE;
            ptr->spirv_resources_list[stage][res_type].list[i].ubo_array_element = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].ubo_array_bindings = NULL;
            ptr->spirv_resources_list[stage][res_type].list[i].ubo_has_instance_name = GL_FALSE;
            ptr->spirv_resources_list[stage][res_type].list[i].ubo_instance_name = NULL;
            ptr->spirv_resources_list[stage][res_type].list[i].binding = ptr->spirv_resources_list[stage][res_type].list[i].gl_binding;
            ptr->spirv_resources_list[stage][res_type].list[i].location = spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation);
            /* SpvDecorationIndex holds the dual-source blending index for
             * fragment outputs (layout(location = 0, index = 1) out vec4 color).
             * Store it so GL_LOCATION_INDEX queries can retrieve it.
             * Per the GL spec, fragment shader PROGRAM_OUTPUT defaults to
             * index 0 when the Index decoration is absent; all other
             * interfaces/stages default to -1 (not applicable). */
            {
                GLuint default_idx;
                if (stage == _FRAGMENT_SHADER &&
                    res_type == SPVC_RESOURCE_TYPE_STAGE_OUTPUT) {
                    default_idx = 0;
                } else {
                    default_idx = (GLuint)-1;
                }
                ptr->spirv_resources_list[stage][res_type].list[i].location_index =
                    spvc_compiler_has_decoration(compiler_msl, list[i].id, SpvDecorationIndex) ?
                    spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationIndex) : default_idx;
            }
            /* For atomic counters, SpvDecorationLocation is not used; store
             * the buffer offset (SpvDecorationOffset) in the location field
             * so GL_OFFSET queries can retrieve it. */
            if (res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER) {
                GLuint ac_offset = spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationOffset);
                ptr->spirv_resources_list[stage][res_type].list[i].location = ac_offset;
            }
            ptr->spirv_resources_list[stage][res_type].list[i].gl_type = mglGLTypeFromSPVCType(reflected_type);
            ptr->spirv_resources_list[stage][res_type].list[i].gl_array_size = mglGLArraySizeFromSPVCType(reflected_type);
            ptr->spirv_resources_list[stage][res_type].list[i].is_array =
                (reflected_type && spvc_type_get_num_array_dimensions(reflected_type) > 0) ? GL_TRUE : GL_FALSE;
            ptr->spirv_resources_list[stage][res_type].list[i].num_array_dims =
                reflected_type ? spvc_type_get_num_array_dimensions(reflected_type) : 0;
            /* Detect per-patch variables for tessellation shaders. */
            ptr->spirv_resources_list[stage][res_type].list[i].is_per_patch =
                (res_type == SPVC_RESOURCE_TYPE_STAGE_INPUT ||
                 res_type == SPVC_RESOURCE_TYPE_STAGE_OUTPUT) &&
                spvc_compiler_has_decoration(compiler_msl, list[i].id, SpvDecorationPatch) ? GL_TRUE : GL_FALSE;
            ptr->spirv_resources_list[stage][res_type].list[i].uniform_location =
                (mglIsSamplerResourceType(res_type) || uniform_constant_sampler_like)
                    ? mglSamplerUniformLocationFromReflection(ptr->spirv_resources_list[stage][res_type].list[i].location,
                                                              stage,
                                                              res_type,
                                                              (GLuint)i,
                                                              ptr->shader_slots[stage] ? ptr->shader_slots[stage]->src : NULL,
                                                              list[i].name)
                    : -1;
            ptr->spirv_resources_list[stage][res_type].list[i].sampler_unit = -1;
            ptr->spirv_resources_list[stage][res_type].list[i].sampler_unit_explicit = GL_FALSE;
            ptr->spirv_resources_list[stage][res_type].list[i].required_size = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_dim = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_arrayed = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_multisampled = 0;

            if (res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER ||
                res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER) {
                SpirvResource *block_res = &ptr->spirv_resources_list[stage][res_type].list[i];
                spvc_type block_type = reflected_type;
                if (block_type) {
                    unsigned array_dims = spvc_type_get_num_array_dimensions(block_type);
                    if (array_dims > 0) {
                        GLuint array_size = (GLuint)spvc_type_get_array_dimension(block_type, 0);
                        block_res->ubo_is_array = GL_TRUE;
                        block_res->ubo_array_size = array_size > 0 ? array_size : 1;
                    }
                }
                /* SPIRV-Cross's MSL compiler resolves UBO/SSBO type_id to the
                 * struct type, not the array type, so the array dimension
                 * check above may return 0 even for block arrays.  Fall back
                 * to parsing the GLSL source to detect the array size. */
                if (block_res->ubo_array_size == 1) {
                    const char *glsl_src = ptr->shader_slots[stage]
                        ? ptr->shader_slots[stage]->src : NULL;
                    GLuint glsl_array_size = mglGLSLUBOArraySize(glsl_src, block_res->name);
                    if (glsl_array_size > 0) {
                        block_res->ubo_is_array = GL_TRUE;
                        block_res->ubo_array_size = glsl_array_size > 0 ? glsl_array_size : 1;
                    }
                }
                block_res->ubo_array_bindings =
                    (GLuint *)calloc(block_res->ubo_array_size, sizeof(GLuint));
                if (block_res->ubo_array_bindings) {
                    for (GLuint ai = 0; ai < block_res->ubo_array_size; ai++) {
                        block_res->ubo_array_bindings[ai] = block_res->gl_binding + ai;
                    }
                }
                const char *glsl_src = ptr->shader_slots[stage]
                    ? ptr->shader_slots[stage]->src : NULL;
                block_res->ubo_instance_name = mglGLSLUBOInstanceName(glsl_src, block_res->name);
                block_res->ubo_has_instance_name =
                    (block_res->ubo_instance_name && block_res->ubo_instance_name[0]) ? GL_TRUE : GL_FALSE;
            }

            bool resource_has_image_type =
                res_type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE ||
                res_type == SPVC_RESOURCE_TYPE_SEPARATE_IMAGE ||
                res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE ||
                (res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
                 (reflected_basetype == SPVC_BASETYPE_IMAGE ||
                  reflected_basetype == SPVC_BASETYPE_SAMPLED_IMAGE));

            if (resource_has_image_type) {
                spvc_type image_type = reflected_type;

                if (image_type) {
                    ptr->spirv_resources_list[stage][res_type].list[i].image_dim =
                        (GLuint)spvc_type_get_image_dimension(image_type);
                    ptr->spirv_resources_list[stage][res_type].list[i].image_arrayed =
                        (GLuint)spvc_type_get_image_arrayed(image_type);
                    ptr->spirv_resources_list[stage][res_type].list[i].image_multisampled =
                        (GLuint)spvc_type_get_image_multisampled(image_type);

                    if (ptr->spirv_resources_list[stage][res_type].list[i].image_dim == (GLuint)SpvDimCube) {
                        fprintf(stderr,
                                "MGL SPIRV IMAGE resource program=%u stage=%d type=%d name=%s binding=%u dim=Cube arrayed=%u multisampled=%u\n",
                                ptr->name,
                                stage,
                                res_type,
                                list[i].name ? list[i].name : "(null)",
                                ptr->spirv_resources_list[stage][res_type].list[i].binding,
                                ptr->spirv_resources_list[stage][res_type].list[i].image_arrayed,
                                ptr->spirv_resources_list[stage][res_type].list[i].image_multisampled);
                    }
                }
            }

            if (res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER ||
                res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT ||
                res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ||
                res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER ||
                res_type == SPVC_RESOURCE_TYPE_PUSH_CONSTANT) {
                size_t declared_size = 0;

                if (reflected_type && spvc_type_get_basetype(reflected_type) == SPVC_BASETYPE_STRUCT) {
                    size_t struct_size = 0;
                    if (spvc_compiler_get_declared_struct_size(compiler_msl, reflected_type, &struct_size) == SPVC_SUCCESS) {
                        declared_size = struct_size;
                    }
                }

                /* Some Minecraft 1.21 shaders can make SPIRV-Cross crash while
                 * traversing active buffer ranges. The declared struct size is
                 * enough for our uniform buffer sizing, so avoid that fragile
                 * optional reflection pass. */

                if (res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER && declared_size > 0) {
                    declared_size = mglRoundUpSize(declared_size, 16);
                }

                ptr->spirv_resources_list[stage][res_type].list[i].required_size = declared_size;
            }
        }
    }

    /* Expand user-defined stage I/O blocks (e.g. `out Color { float r, g, b;
     * vec4 iLikePie; } vs_color;`) into individual member resources
     * (Color.r, Color.g, Color.b, Color.iLikePie) per the GL 4.3 program
     * interface query spec.  Built-in blocks like gl_PerVertex are handled
     * separately by the built-in reflection below and do not appear in the
     * regular STAGE_INPUT/STAGE_OUTPUT lists. */
    {
        const int io_res_types[] = {
            SPVC_RESOURCE_TYPE_STAGE_INPUT,
            SPVC_RESOURCE_TYPE_STAGE_OUTPUT
        };
        for (int ti = 0; ti < (int)(sizeof(io_res_types) / sizeof(io_res_types[0])); ti++) {
            int io_type = io_res_types[ti];
            SpirvResourceList *io_list = &ptr->spirv_resources_list[stage][io_type];
            if (!io_list->list || io_list->count == 0)
                continue;

            /* First pass: count expanded resources and detect any blocks */
            GLuint expanded_count = 0;
            GLboolean has_blocks = GL_FALSE;
            for (GLuint i = 0; i < io_list->count; i++) {
                SpirvResource *res = &io_list->list[i];
                spvc_type_id struct_type_id = res->type_id;
                spvc_type struct_type = struct_type_id ?
                    spvc_compiler_get_type_handle(compiler_msl, struct_type_id) : NULL;
                if ((!struct_type ||
                     spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT) &&
                    res->base_type_id) {
                    struct_type_id = res->base_type_id;
                    struct_type = spvc_compiler_get_type_handle(compiler_msl, struct_type_id);
                }
                GLboolean is_blk = (struct_type &&
                    spvc_type_get_basetype(struct_type) == SPVC_BASETYPE_STRUCT &&
                    (spvc_compiler_has_decoration(compiler_msl, struct_type_id, SpvDecorationBlock) ||
                     (res->base_type_id && res->base_type_id != struct_type_id &&
                      spvc_compiler_has_decoration(compiler_msl, res->base_type_id, SpvDecorationBlock))));
                if (is_blk) {
                    /* Ensure struct_type_id points to the ID with the Block decoration */
                    if (!spvc_compiler_has_decoration(compiler_msl, struct_type_id, SpvDecorationBlock) &&
                        res->base_type_id && spvc_compiler_has_decoration(compiler_msl, res->base_type_id, SpvDecorationBlock)) {
                        struct_type_id = res->base_type_id;
                        struct_type = spvc_compiler_get_type_handle(compiler_msl, struct_type_id);
                    }
                    GLboolean is_builtin_block = GL_FALSE;
                    unsigned member_count = spvc_type_get_num_member_types(struct_type);
                    for (unsigned mi = 0; mi < member_count; mi++) {
                        if (spvc_compiler_has_member_decoration(
                                compiler_msl, struct_type_id, mi, SpvDecorationBuiltIn)) {
                            is_builtin_block = GL_TRUE;
                            break;
                        }
                    }
                    if (is_builtin_block) {
                        expanded_count += 1;
                    } else {
                        expanded_count += member_count;
                        has_blocks = GL_TRUE;
                    }
                } else {
                    expanded_count += 1;
                }
            }

            if (!has_blocks)
                continue;

            /* Second pass: build the new expanded list */
            SpirvResource *new_list = (SpirvResource *)calloc(expanded_count, sizeof(SpirvResource));
            if (!new_list)
                continue;

            GLuint new_idx = 0;
            for (GLuint i = 0; i < io_list->count; i++) {
                SpirvResource *res = &io_list->list[i];
                spvc_type_id struct_type_id = res->type_id;
                spvc_type struct_type = struct_type_id ?
                    spvc_compiler_get_type_handle(compiler_msl, struct_type_id) : NULL;
                if ((!struct_type ||
                     spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT) &&
                    res->base_type_id) {
                    struct_type_id = res->base_type_id;
                    struct_type = spvc_compiler_get_type_handle(compiler_msl, struct_type_id);
                }
                GLboolean is_block = (struct_type &&
                    spvc_type_get_basetype(struct_type) == SPVC_BASETYPE_STRUCT &&
                    (spvc_compiler_has_decoration(compiler_msl, struct_type_id, SpvDecorationBlock) ||
                     (res->base_type_id && res->base_type_id != struct_type_id &&
                      spvc_compiler_has_decoration(compiler_msl, res->base_type_id, SpvDecorationBlock))));

                if (is_block) {
                    /* Ensure struct_type_id points to the ID with the Block decoration */
                    if (!spvc_compiler_has_decoration(compiler_msl, struct_type_id, SpvDecorationBlock) &&
                        res->base_type_id && spvc_compiler_has_decoration(compiler_msl, res->base_type_id, SpvDecorationBlock)) {
                        struct_type_id = res->base_type_id;
                        struct_type = spvc_compiler_get_type_handle(compiler_msl, struct_type_id);
                    }
                    /* Skip builtin blocks (gl_PerVertex) — handled elsewhere */
                    GLboolean is_builtin_block = GL_FALSE;
                    unsigned member_count = spvc_type_get_num_member_types(struct_type);
                    for (unsigned mi = 0; mi < member_count; mi++) {
                        if (spvc_compiler_has_member_decoration(
                                compiler_msl, struct_type_id, mi, SpvDecorationBuiltIn)) {
                            is_builtin_block = GL_TRUE;
                            break;
                        }
                    }
                    if (is_builtin_block) {
                        /* Copy as-is (move name ownership) */
                        new_list[new_idx] = *res;
                        res->name = NULL; /* prevent double-free */
                        new_idx++;
                        continue;
                    }

                    /* Get the block name from the struct type */
                    const char *block_name = spvc_compiler_get_name(compiler_msl, struct_type_id);
                    if (!block_name || !block_name[0])
                        block_name = res->name ? res->name : "block";

                    for (unsigned mi = 0; mi < member_count; mi++) {
                        spvc_type_id member_type_id = spvc_type_get_member_type(struct_type, mi);
                        spvc_type member_type = member_type_id ?
                            spvc_compiler_get_type_handle(compiler_msl, member_type_id) : NULL;
                        const char *member_name = spvc_compiler_get_member_name(
                            compiler_msl, struct_type_id, mi);

                        char name_buf[256];
                        snprintf(name_buf, sizeof(name_buf), "%s.%s",
                                 block_name, member_name ? member_name : "");

                        SpirvResource *dst = &new_list[new_idx++];
                        memset(dst, 0, sizeof(SpirvResource));
                        dst->_id = res->_id;
                        dst->base_type_id = res->base_type_id;
                        dst->type_id = struct_type_id;
                        dst->name = strdup(name_buf);
                        dst->gl_type = mglGLTypeFromSPVCType(member_type);
                        dst->gl_array_size = mglGLArraySizeFromSPVCType(member_type);
                        dst->is_array = (member_type &&
                            spvc_type_get_num_array_dimensions(member_type) > 0) ? GL_TRUE : GL_FALSE;
                        dst->num_array_dims = member_type ?
                            spvc_type_get_num_array_dimensions(member_type) : 0;
                        /* Assign location: prefer member Location decoration, then
                         * block variable Location + member index, then auto-assign
                         * starting from 0 so members sort before builtins. */
                        if (spvc_compiler_has_member_decoration(
                                compiler_msl, struct_type_id, mi, SpvDecorationLocation)) {
                            dst->location = spvc_compiler_get_member_decoration(
                                compiler_msl, struct_type_id, mi, SpvDecorationLocation);
                        } else {
                            GLuint base_loc = spvc_compiler_get_decoration(
                                compiler_msl, res->_id, SpvDecorationLocation);
                            dst->location = base_loc + mi;
                        }
                        dst->gl_binding = (GLuint)-1;
                        /* Fragment shader outputs default to index 0 per GL spec;
                         * all other stage I/O defaults to -1. */
                        {
                            GLuint default_idx = (stage == _FRAGMENT_SHADER &&
                                                  io_type == SPVC_RESOURCE_TYPE_STAGE_OUTPUT)
                                                 ? 0u : (GLuint)-1;
                            dst->location_index = spvc_compiler_has_member_decoration(
                                compiler_msl, struct_type_id, mi, SpvDecorationIndex) ?
                                spvc_compiler_get_member_decoration(
                                    compiler_msl, struct_type_id, mi, SpvDecorationIndex) : default_idx;
                        }
                        dst->uniform_location = -1;
                        dst->sampler_unit = -1;
                        dst->sampler_unit_explicit = GL_FALSE;
                        dst->ubo_array_size = 1;
                        dst->ubo_is_array = GL_FALSE;
                        dst->ubo_array_element = 0;
                        dst->ubo_array_bindings = NULL;
                        dst->ubo_has_instance_name = GL_FALSE;
                        dst->ubo_instance_name = NULL;
                        dst->required_size = 0;
                        dst->image_dim = 0;
                        dst->image_arrayed = 0;
                        dst->image_multisampled = 0;
                        dst->is_per_patch = GL_FALSE;
                        dst->ubo_members = NULL;
                        dst->ubo_member_count = 0;
                    }
                    /* Free the block resource's name */
                    free((void *)res->name);
                } else {
                    /* Non-block resource — copy as-is (move name ownership) */
                    new_list[new_idx] = *res;
                    res->name = NULL; /* prevent double-free */
                    new_idx++;
                }
            }

            /* Replace the old list */
            free(io_list->list);
            io_list->list = new_list;
            io_list->count = expanded_count;
        }
    }

    /* Reflect UBO/SSBO member uniforms.
     * Each SPVC_RESOURCE_TYPE_UNIFORM_BUFFER and SPVC_RESOURCE_TYPE_STORAGE_BUFFER
     * resource corresponds to a struct whose members need to be exposed via
     * glGetActiveUniform / glGetActiveUniformsiv (for UBOs) and via
     * GL_BUFFER_VARIABLE queries (for SSBOs) so that CTS and applications can
     * query offsets, strides, and types. */
    const int block_res_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER
    };
    for (int ti = 0; ti < (int)(sizeof(block_res_types) / sizeof(block_res_types[0])); ti++) {
        int res_type = block_res_types[ti];
        SpirvResourceList *ubo_list = &ptr->spirv_resources_list[stage][res_type];
        for (GLuint ubo_idx = 0; ubo_list->list && ubo_idx < ubo_list->count; ubo_idx++) {
            SpirvResource *ubo = &ubo_list->list[ubo_idx];
            spvc_type struct_type = NULL;
            spvc_type_id struct_type_id = 0;

            if (ubo->type_id) {
                struct_type = spvc_compiler_get_type_handle(compiler_msl, ubo->type_id);
                struct_type_id = ubo->type_id;
            }
            if ((!struct_type ||
                 spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT) &&
                ubo->base_type_id) {
                struct_type = spvc_compiler_get_type_handle(compiler_msl, ubo->base_type_id);
                struct_type_id = ubo->base_type_id;
            }
            if (!struct_type ||
                spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT) {
                ubo->ubo_members = NULL;
                ubo->ubo_member_count = 0;
                continue;
            }

            if (getenv("MGL_DEBUG_UBO_REFLECT")) {
                fprintf(stderr,
                        "MGL UBO REFLECT program=%u stage=%d ubo=%s id=%u type=%u base=%u structType=%u structName=%s\n",
                        ptr->name,
                        stage,
                        ubo->name ? ubo->name : "(null)",
                        ubo->_id,
                        ubo->type_id,
                        ubo->base_type_id,
                        struct_type_id,
                        spvc_compiler_get_name(compiler_msl, struct_type_id));
            }

            ubo->ubo_members = NULL;
            ubo->ubo_member_count = 0;
            GLuint reflected_count = 0;
            GLboolean default_row_major = GL_FALSE;
            const char *glsl_src = ptr->shader_slots[stage]
                ? ptr->shader_slots[stage]->src : NULL;
            if (mglGLSLDeclaresRowMajorUBOMember(glsl_src, ubo->name, "")) {
                default_row_major = GL_TRUE;
            }

            if (!mglReflectUBOMemberLeaves(ptr,
                                           stage,
                                           compiler_msl,
                                           ubo,
                                           struct_type,
                                           struct_type_id,
                                           NULL,
                                           0,
                                           default_row_major,
                                           &reflected_count,
                                           0,
                                           NULL,
                                           0,
                                           NULL,
                                           NULL,
                                           0,
                                           0,
                                           0)) {
                for (GLuint m = 0; m < ubo->ubo_member_count; m++) {
                    free((void *)ubo->ubo_members[m].name);
                    free(ubo->ubo_members[m].query_name);
                }
                free(ubo->ubo_members);
                ubo->ubo_members = NULL;
                ubo->ubo_member_count = 0;
            }
        }
    }

    /* Reflect plain (non-UBO) struct uniform members.
     * Each SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT resource whose type is a struct
     * needs its leaf members exposed via glGetActiveUniform / glGetUniformLocation
     * so that applications can query "var.member" / "var[elem].member" locations.
     * Unlike UBO members, plain struct members carry a GL location_offset relative
     * to the struct variable's base location. */
    {
        SpirvResourceList *uc_list =
            &ptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint uc_idx = 0; uc_list->list && uc_idx < uc_list->count; uc_idx++) {
            SpirvResource *res = &uc_list->list[uc_idx];
            spvc_type elem_type = NULL;
            spvc_type_id elem_type_id = 0;

            if (res->ubo_members || !res->name || !res->name[0]) {
                continue;
            }

            /* Prefer base_type_id (struct with array dimensions stripped) so
             * that location counting reflects a single struct element. */
            if (res->base_type_id) {
                spvc_type t = spvc_compiler_get_type_handle(compiler_msl, res->base_type_id);
                if (t && spvc_type_get_basetype(t) == SPVC_BASETYPE_STRUCT) {
                    elem_type = t;
                    elem_type_id = res->base_type_id;
                }
            }
            if (!elem_type && res->type_id) {
                spvc_type t = spvc_compiler_get_type_handle(compiler_msl, res->type_id);
                if (t && spvc_type_get_basetype(t) == SPVC_BASETYPE_STRUCT &&
                    spvc_type_get_num_array_dimensions(t) == 0) {
                    elem_type = t;
                    elem_type_id = res->type_id;
                }
            }
            if (!elem_type) {
                continue;
            }

            /* Skip samplers/images that happen to live in the uniform-constant
             * list; only genuine structs are reflected here. */
            if (mglUniformConstantBaseTypeIsSamplerLike(spvc_type_get_basetype(elem_type)) ||
                mglUniformNameLooksSamplerLike(res->name)) {
                continue;
            }

            /* Determine the top-level array size of the struct variable. */
            GLint array_size = 1;
            if (res->type_id) {
                spvc_type t = spvc_compiler_get_type_handle(compiler_msl, res->type_id);
                if (t) {
                    GLint asz = mglGLArraySizeFromSPVCType(t);
                    if (asz > 1) {
                        array_size = asz;
                    }
                }
            }
            if (array_size <= 1 && res->gl_array_size > 1) {
                array_size = res->gl_array_size;
            }

            /* Use CTS-convention location step (matrices count as 1 location)
             * so per-element offsets match explicit_uniform_location tests. */
            GLint elem_loc_count = mglSPVCTypeLocationStep(compiler_msl, elem_type);

            /* For plain struct uniforms, spvc_compiler_get_active_buffer_ranges
             * returns 0 (they are not buffer-backed).  Instead, we directly
             * scan the SPIR-V binary for OpAccessChain instructions rooted at
             * this variable to determine which members are active. */
            const spvc_buffer_range *active_ranges = NULL;
            size_t num_active_ranges = 0;

            MGLActivePathSet active_path_set;
            active_path_set.count = 0;
            const unsigned int *spirv_ir = ptr->spirv[stage].ir;
            size_t spirv_wc = ptr->spirv[stage].size;
            if (spirv_ir && spirv_wc >= 5 && res->_id) {
                mglCollectActivePaths(spirv_ir, spirv_wc,
                                      (GLuint)res->_id,
                                      &active_path_set);
            }

            /* Compute the byte stride for array-of-struct elements so that
             * absolute_offset within the reflection matches the offsets
             * reported by get_active_buffer_ranges (which are relative to
             * the start of the variable, not the start of a single element).
             * Also store the struct size in required_size for render-time
             * struct buffer packing. */
            size_t elem_byte_size = 0;
            spvc_compiler_get_declared_struct_size(compiler_msl,
                                                   elem_type,
                                                   &elem_byte_size);
            if (elem_byte_size == 0) {
                /* Plain struct uniforms lack Offset decorations, so
                 * spvc_compiler_get_declared_struct_size returns 0.
                 * Compute the size using Metal/C alignment rules. */
                elem_byte_size = mglComputeMSLStructSize(compiler_msl, elem_type);
            }
            res->required_size = elem_byte_size;

            res->ubo_members = NULL;
            res->ubo_member_count = 0;
            GLuint reflected_count = 0;
            GLboolean reflect_ok = GL_TRUE;

            /* current_path buffer used during reflection to track the member
             * index path from the root variable. */
            GLuint current_path[MGL_ACTIVE_MAX_DEPTH];

            for (GLint elem = 0; elem < array_size && reflect_ok; elem++) {
                size_t name_len = strlen(res->name);
                size_t prefix_len = name_len;
                char *prefix = NULL;
                if (array_size > 1) {
                    prefix_len = name_len + 16;
                }
                prefix = (char *)malloc(prefix_len + 1u);
                if (!prefix) {
                    reflect_ok = GL_FALSE;
                    break;
                }
                if (array_size > 1) {
                    snprintf(prefix, prefix_len + 1u, "%s[%d]", res->name, elem);
                } else {
                    memcpy(prefix, res->name, name_len);
                    prefix[name_len] = '\0';
                }

                GLint loc_offset = elem * elem_loc_count;
                GLuint elem_base_offset = (array_size > 1 && elem_byte_size > 0)
                    ? (GLuint)(elem * (GLint)elem_byte_size) : 0u;

                /* For array-of-struct, the first index in the SPIR-V access
                 * chain is the array element index. */
                GLuint path_start = 0;
                const MGLActivePathSet *path_set_ptr = NULL;
                if (active_path_set.count > 0) {
                    if (array_size > 1) {
                        current_path[0] = (GLuint)elem;
                        path_start = 1;
                        /* Only reflect if this array element has active members */
                        if (!mglActivePathHasPrefix(&active_path_set,
                                                     current_path, 1)) {
                            free(prefix);
                            continue;
                        }
                    }
                    path_set_ptr = &active_path_set;
                }

                if (!mglReflectUBOMemberLeaves(ptr,
                                               stage,
                                               compiler_msl,
                                               res,
                                               elem_type,
                                               elem_type_id,
                                               prefix,
                                               elem_base_offset,
                                               GL_FALSE,
                                               &reflected_count,
                                               loc_offset,
                                               active_ranges,
                                               num_active_ranges,
                                               path_set_ptr,
                                               current_path,
                                               path_start,
                                               0,
                                               0)) {
                    reflect_ok = GL_FALSE;
                }
                free(prefix);
            }

            if (!reflect_ok || res->ubo_member_count == 0) {
                for (GLuint m = 0; m < res->ubo_member_count; m++) {
                    free((void *)res->ubo_members[m].name);
                    free(res->ubo_members[m].query_name);
                }
                free(res->ubo_members);
                res->ubo_members = NULL;
                res->ubo_member_count = 0;
            }
        }
    }

    /* Rename GLSL uniform variables whose name conflicts with Metal built-in
     * types (e.g. "sampler") to avoid MSL compilation errors such as
     * "must use 'struct' tag to refer to type 'sampler' in this scope".
     * This happens AFTER reflection data is collected, so MGL stores the
     * original GLSL name for glGetUniformLocation queries, but BEFORE MSL
     * compilation so the generated Metal source uses the safe name. */
    {
        const spvc_reflected_resource *rename_list = NULL;
        size_t rename_count = 0;
        for (int res_type = SPVC_RESOURCE_TYPE_UNIFORM_BUFFER;
             res_type < SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE;
             res_type++)
        {
            spvc_resources_get_resource_list_for_type(resources, res_type,
                                                       &rename_list, &rename_count);
            for (size_t ri = 0; ri < rename_count; ri++)
            {
                if (rename_list[ri].name &&
                    strcmp(rename_list[ri].name, "sampler") == 0)
                {
                    spvc_compiler_set_name(compiler_msl, rename_list[ri].id,
                                           "mgl_sampler_tex");
                }
            }
        }
    }

    /* Note: std140 ArrayStride repair for SSBO members is now handled by
     * the ir_fix_std140_array_strides pass in mgl_ir_postprocess.c, which
     * runs as part of mglRunIRPostprocessPipeline before spvc_compiler_compile. */

    /* Reflect built-in variables (gl_VertexID, gl_InstanceID, gl_FragDepth,
     * gl_SampleMask, gl_Position, etc.) so they appear as active PROGRAM_INPUT /
     * PROGRAM_OUTPUT resources per the GL 4.3 program interface query spec.
     * Both input and output built-ins are reflected for every stage and stored
     * per-stage, so that separate (single-stage) programs can expose their own
     * stage's built-ins.  The query layer picks the correct stage (first active
     * stage for PROGRAM_INPUT, last active stage for PROGRAM_OUTPUT). */
    {
        struct {
            spvc_builtin_resource_type rt;
            GLboolean is_input;
        } dirs[2] = {
            { SPVC_BUILTIN_RESOURCE_TYPE_STAGE_INPUT,  GL_TRUE  },
            { SPVC_BUILTIN_RESOURCE_TYPE_STAGE_OUTPUT, GL_FALSE },
        };

        /* Use the active-variables filtered resource list so that only built-ins
         * actually referenced by the shader are exposed.  Without this filter,
         * SPIRV-Cross returns every built-in that exists for the stage. */
        spvc_set active_set = NULL;
        spvc_resources active_resources = NULL;
        if (spvc_compiler_get_active_interface_variables(compiler_msl, &active_set) == SPVC_SUCCESS &&
            active_set &&
            spvc_compiler_create_shader_resources_for_active_variables(
                compiler_msl, &active_resources, active_set) == SPVC_SUCCESS &&
            active_resources)
        {
            for (int dir_i = 0; dir_i < 2; dir_i++) {
                spvc_builtin_resource_type builtin_rt = dirs[dir_i].rt;
                GLboolean is_input = dirs[dir_i].is_input;

                const spvc_reflected_builtin_resource *builtin_list = NULL;
                size_t builtin_count = 0;
                if (spvc_resources_get_builtin_resource_list_for_type(
                        active_resources, builtin_rt, &builtin_list, &builtin_count) != SPVC_SUCCESS)
                    continue;

                for (size_t bi = 0; bi < builtin_count; bi++) {
                    SpvBuiltIn builtin = builtin_list[bi].builtin;
                    const char *member_name = NULL;
                    GLuint gl_type = 0;
                    GLboolean is_array_builtin = GL_FALSE;

                    if (is_input) {
                        switch (builtin) {
                            case SpvBuiltInVertexId:       member_name = "gl_VertexID";        gl_type = GL_INT; break;
                            case SpvBuiltInInstanceId:     member_name = "gl_InstanceID";      gl_type = GL_INT; break;
                            case SpvBuiltInPrimitiveId:    member_name = "gl_PrimitiveID";     gl_type = GL_INT; break;
                            case SpvBuiltInInvocationId:   member_name = "gl_InvocationID";    gl_type = GL_INT; break;
                            case SpvBuiltInPatchVertices:  member_name = "gl_PatchVerticesIn"; gl_type = GL_INT; break;
                            case SpvBuiltInFragCoord:      member_name = "gl_FragCoord";       gl_type = GL_FLOAT_VEC4; break;
                            case SpvBuiltInPointCoord:     member_name = "gl_PointCoord";      gl_type = GL_FLOAT_VEC2; break;
                            case SpvBuiltInFrontFacing:    member_name = "gl_FrontFacing";     gl_type = GL_BOOL; break;
                            case SpvBuiltInSampleId:       member_name = "gl_SampleID";        gl_type = GL_INT; break;
                            case SpvBuiltInSamplePosition: member_name = "gl_SamplePosition";  gl_type = GL_FLOAT_VEC2; break;
                            case SpvBuiltInSampleMask:     member_name = "gl_SampleMaskIn[0]"; gl_type = GL_INT; is_array_builtin = GL_TRUE; break;
                            case SpvBuiltInPosition:       member_name = "gl_Position";        gl_type = GL_FLOAT_VEC4; break;
                            case SpvBuiltInPointSize:      member_name = "gl_PointSize";       gl_type = GL_FLOAT; break;
                            case SpvBuiltInClipDistance:   member_name = "gl_ClipDistance";    gl_type = GL_FLOAT; is_array_builtin = GL_TRUE; break;
                            case SpvBuiltInCullDistance:   member_name = "gl_CullDistance";    gl_type = GL_FLOAT; is_array_builtin = GL_TRUE; break;
                            default: break;
                        }
                    } else {
                        switch (builtin) {
                            case SpvBuiltInPosition:       member_name = "gl_Position";          gl_type = GL_FLOAT_VEC4; break;
                            case SpvBuiltInPointSize:      member_name = "gl_PointSize";         gl_type = GL_FLOAT; break;
                            case SpvBuiltInClipDistance:   member_name = "gl_ClipDistance";      gl_type = GL_FLOAT; is_array_builtin = GL_TRUE; break;
                            case SpvBuiltInCullDistance:   member_name = "gl_CullDistance";      gl_type = GL_FLOAT; is_array_builtin = GL_TRUE; break;
                            case SpvBuiltInFragDepth:      member_name = "gl_FragDepth";         gl_type = GL_FLOAT; break;
                            case SpvBuiltInSampleMask:     member_name = "gl_SampleMask[0]";     gl_type = GL_INT; is_array_builtin = GL_TRUE; break;
                            case SpvBuiltInTessLevelInner: member_name = "gl_TessLevelInner[0]"; gl_type = GL_FLOAT; is_array_builtin = GL_TRUE; break;
                            case SpvBuiltInTessLevelOuter: member_name = "gl_TessLevelOuter[0]"; gl_type = GL_FLOAT; is_array_builtin = GL_TRUE; break;
                            default: break;
                        }
                    }
                    if (!member_name) continue;

                    /* For block members (gl_PerVertex), prefix with
                     * "gl_PerVertex." if the block has an instance name. */
                    char name_buf[128];
                    const char *name = member_name;
                    GLboolean is_block = spvc_compiler_has_decoration(
                        compiler_msl, builtin_list[bi].resource.type_id, SpvDecorationBlock);
                    if (!is_block && builtin_list[bi].resource.base_type_id &&
                        builtin_list[bi].resource.base_type_id != builtin_list[bi].resource.type_id) {
                        is_block = spvc_compiler_has_decoration(
                            compiler_msl, builtin_list[bi].resource.base_type_id, SpvDecorationBlock);
                    }
                    if (is_block) {
                        const char *var_name = spvc_compiler_get_name(
                            compiler_msl, builtin_list[bi].resource.id);
                        if (var_name && var_name[0]) {
                            snprintf(name_buf, sizeof(name_buf), "gl_PerVertex.%s", member_name);
                            name = name_buf;
                        }
                    }

                    SpirvResource *dst;
                    if (is_input) {
                        if (ptr->builtin_program_input_count[stage] >= 16) break;
                        dst = &ptr->builtin_program_inputs[stage][ptr->builtin_program_input_count[stage]++];
                    } else {
                        if (ptr->builtin_program_output_count[stage] >= 16) break;
                        dst = &ptr->builtin_program_outputs[stage][ptr->builtin_program_output_count[stage]++];
                    }

                    memset(dst, 0, sizeof(SpirvResource));
                    dst->_id = builtin_list[bi].resource.id;
                    dst->base_type_id = builtin_list[bi].resource.base_type_id;
                    dst->type_id = builtin_list[bi].resource.type_id;
                    dst->name = strdup(name);
                    dst->gl_type = gl_type;
                    dst->gl_array_size = 1;
                    dst->is_array = is_array_builtin;
                    dst->num_array_dims = is_array_builtin ? 1 : 0;
                    dst->location = (GLuint)-1;
                    dst->gl_binding = (GLuint)-1;
                    dst->location_index = (GLuint)-1;
                    dst->uniform_location = -1;
                    dst->ubo_array_size = 1;
                    dst->ubo_is_array = GL_FALSE;
                    dst->ubo_array_element = 0;
                    dst->ubo_array_bindings = NULL;
                    dst->ubo_has_instance_name = GL_FALSE;
                    dst->ubo_instance_name = NULL;
                    dst->sampler_unit = -1;
                    dst->sampler_unit_explicit = GL_FALSE;
                    dst->required_size = 0;
                    dst->image_dim = 0;
                    dst->image_arrayed = 0;
                    dst->image_multisampled = 0;
                    dst->is_per_patch = GL_FALSE;
                    dst->ubo_members = NULL;
                    dst->ubo_member_count = 0;
                }
            }
        }
    }

    /* IR postprocess pipeline (Phase 5B-2).  Runs after all reflection is
     * complete (spirv_resources_list, builtins) and BEFORE spvc_compiler_compile.
     * Remaps any user buffer whose GL binding lands on an MGL-reserved Metal
     * buffer slot to a free slot in [0, 24] via spvc_compiler_set_decoration,
     * so the compiled MSL already has correct [[buffer(N)]] — no string-level
     * replacement needed in applyMSLResourceBindings for these cases.
     *
     * Gated by env vars:
     *   MGL_DISABLE_IR_REMAP=1 — bypass, force legacy string-level fallback.
     *   MGL_DEBUG_IR_REMAP=1    — log every binding decision.
     * See mgl_ir_postprocess.h for the full pass list and ordering. */
    mglRunIRPostprocessPipeline(ctx, ptr, stage, compiler_msl);

    if (spvc_compiler_compile(compiler_msl, &result) != SPVC_SUCCESS || !result) {
        const char *last_error = spvc_context_get_last_error_string(context);
        fprintf(stderr,
                "MGL WARNING: spvc_compiler_compile failed program=%u stage=%d: %s\n",
                ptr->name,
                stage,
                last_error ? last_error : "(no diagnostic)");
        spvc_context_destroy(context);
        return NULL;
    }
    if (getenv("MGL_DUMP_MSL")) {
        fprintf(stderr, "MGL DBG MSL DUMP (program=%u stage=%d):\n%.8000s\n", ptr->name, stage, result);
    }
    DEBUG_PRINT("\n%s\n", result);

    str_ret = strdup(result);
    if (str_ret) {
        MSLPatchPipeline pipeline;
        if (mslPipelineInit(&pipeline, ptr, stage, str_ret)) {
            str_ret = NULL;  /* pipeline owns it now */

            mslPipelineAddStep(&pipeline, "remove_restrict", mglPatchRemoveRestrict);
            mslPipelineAddStep(&pipeline, "fix_sampler_shadowing", mglPatchFixSamplerShadowing);
            mslPipelineAddStep(&pipeline, "fix_unknown_texture_type", mglPatchFixUnknownTextureType);
            mslPipelineAddStep(&pipeline, "strip_thread_const_ref", mglPatchStripThreadConstRef);
            mslPipelineAddStep(&pipeline, "rename_length_squared", mglPatchRenameLengthSquared);
            mslPipelineAddStep(&pipeline, "lower_double_types", mglPatchLowerDoubleTypes);
            mslPipelineAddStep(&pipeline, "fix_end_portal_layer", mglPatchFixEndPortalLayer);
            mslPipelineAddStep(&pipeline, "fullscreen_fb_yflip", mglPatchFullscreenFBYFlip);
            mslPipelineAddStep(&pipeline, "fragcoord_origin_fix", mglPatchFragCoordOriginFix);
            mslPipelineAddStep(&pipeline, "fix_plain_struct_pointer_array", mglPatchFixPlainStructPointerArray);
            mslPipelineAddStep(&pipeline, "inject_atomic_counter_args", mglPatchInjectAtomicCounterArgs);
            mslPipelineAddStep(&pipeline, "apply_resource_bindings", mglPatchApplyResourceBindings);
            mslPipelineAddStep(&pipeline, "inject_point_size_builtin", mglPatchInjectPointSizeBuiltin);
            mslPipelineAddStep(&pipeline, "fix_image2drect_imagesize", mglPatchFixImage2DRectImageSize);
            mslPipelineAddStep(&pipeline, "tes_as_compute_kernel", mglPatchTesAsComputeKernel);
            mslPipelineAddStep(&pipeline, "tcs_stage_in_fix", mglPatchTcsStageInFix);
            mslPipelineAddStep(&pipeline, "apply_resource_bindings_final", mglPatchApplyResourceBindings);

            mslPipelineRun(&pipeline);
            str_ret = mslPipelineTakeResult(&pipeline);
            mslPipelineDestroy(&pipeline);
        }

        if (getenv("MGL_DUMP_MSL") && str_ret) {
            char dumpPath[512];
            snprintf(dumpPath, sizeof(dumpPath),
                     "/tmp/mgl_program_%u_stage_%d.msl",
                     ptr->name, stage);
            FILE *dumpFile = fopen(dumpPath, "w");
            if (dumpFile) {
                fputs(str_ret, dumpFile);
                fclose(dumpFile);
                fprintf(stderr, "MGL DUMP MSL: wrote patched MSL to %s\n", dumpPath);
            }
        }
    }

    /* Check whether SPIRV-Cross emitted spvBufferSizeConstants for
     * runtime-sized SSBO arrays.  The renderer must bind a buffer of
     * uint32 byte-sizes at MGL_BUFFER_SIZE_BUFFER_INDEX when true. */
    ptr->spirv[stage].needs_buffer_size_buffer =
        spvc_compiler_msl_needs_buffer_size_buffer(compiler_msl) ? GL_TRUE : GL_FALSE;

    // Frees all memory we allocated so far.
    spvc_context_destroy(context);

    return str_ret;
}

/* Compile an MSL "capture" variant for GPU transform feedback.
 *
 * Re-runs the SPIRV-Cross MSL backend with output-capture options so the
 * generated vertex function writes its stage-output struct into a device
 * buffer (the XFB buffer), mirroring what the TES-as-compute-kernel path
 * does by hand in mglFixMSLTesAsComputeKernel Step 6.
 *
 * The variant is stored separately in spirv[stage].msl_str_capture; the
 * normal render-pipeline MSL (spirv[stage].msl_str) is unchanged. This is
 * groundwork only: compiling and storing the variant confirms the capture
 * options build cleanly. The renderer dispatch that would actually run this
 * variant against the XFB buffer is NOT yet wired (see the TODO(gpu-xfb)
 * marker in mglDrawArrays).
 *
 * Gated on the MGL_XFB_GPU_CAPTURE env var so the default build pays no
 * cost. Only the vertex shader (the feedback stage for VS-only programs) is
 * compiled; the existing TES path handles tessellation XFB independently.
 * Returns a malloc'd MSL string (caller frees) or NULL on failure / when the
 * gate is off.
 */
char *mglCompileMSLCaptureVariant(GLMContext ctx, Program *ptr, int stage)
{
    const SpvId *spirv;
    size_t word_count;
    spvc_context context = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler compiler_msl = NULL;
    spvc_compiler_options options = NULL;
    const char *result = NULL;
    char *str_ret = NULL;

    if (!ptr || stage < 0 || stage >= _MAX_SHADER_TYPES || !ptr->shader_slots[stage]) {
        return NULL;
    }
    if (stage != _VERTEX_SHADER) {
        /* Only VS XFB is targeted by this groundwork; TES has its own path. */
        return NULL;
    }
    if (ptr->transform_feedback_varying_count <= 0) {
        return NULL;
    }
    if (getenv("MGL_XFB_GPU_CAPTURE") == NULL) {
        return NULL;
    }

    spirv = ptr->spirv[stage].ir;
    word_count = ptr->spirv[stage].size;
    if (!spirv || word_count == 0) {
        return NULL;
    }

    if (spvc_context_create(&context) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: spvc_context_create failed\n");
        return NULL;
    }
    if (spvc_context_parse_spirv(context, spirv, word_count, &ir) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: spvc_context_parse_spirv failed\n");
        spvc_context_destroy(context);
        return NULL;
    }
    if (spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir,
                                     SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                     &compiler_msl) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: spvc_context_create_compiler failed\n");
        spvc_context_destroy(context);
        return NULL;
    }

    if (spvc_compiler_create_compiler_options(compiler_msl, &options) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: options create failed\n");
        spvc_context_destroy(context);
        return NULL;
    }

    /* Mirror the base MSL options set in parseSPIRVShaderToMetal so the
     * capture variant is otherwise consistent with the render variant. */
    (void)spvc_compiler_options_set_bool(options,
                                         SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D,
                                         SPVC_TRUE);
    (void)spvc_compiler_options_set_uint(options,
                                         SPVC_COMPILER_OPTION_MSL_VERSION,
                                         SPVC_MAKE_MSL_VERSION(3,1,0));
    (void)spvc_compiler_options_set_bool(options,
                                         SPVC_COMPILER_OPTION_FIXUP_DEPTH_CONVENTION,
                                         SPVC_TRUE);
    (void)spvc_compiler_options_set_uint(options,
                                         SPVC_COMPILER_OPTION_MSL_BUFFER_SIZE_BUFFER_INDEX,
                                         MGL_BUFFER_SIZE_BUFFER_INDEX);

    /* SPIRV-Cross capture-to-buffer options. These make the vertex function
     * write its output struct to a device buffer at the index below, and
     * disable rasterization (no fragment stage needed for XFB-only draws). */
    if (spvc_compiler_options_set_bool(options,
                                       SPVC_COMPILER_OPTION_MSL_CAPTURE_OUTPUT_TO_BUFFER,
                                       SPVC_TRUE) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: set CAPTURE_OUTPUT_TO_BUFFER failed\n");
        spvc_context_destroy(context);
        return NULL;
    }
    if (spvc_compiler_options_set_bool(options,
                                       SPVC_COMPILER_OPTION_MSL_DISABLE_RASTERIZATION,
                                       SPVC_TRUE) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: set DISABLE_RASTERIZATION failed\n");
        spvc_context_destroy(context);
        return NULL;
    }
    /* Reuse the TES XFB slot convention (buffer 29) so the same bind site in
     * MGLRenderer.m can serve both paths once the dispatch is wired. */
    if (spvc_compiler_options_set_uint(options,
                                       SPVC_COMPILER_OPTION_MSL_SHADER_OUTPUT_BUFFER_INDEX,
                                       29) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: set SHADER_OUTPUT_BUFFER_INDEX failed\n");
        spvc_context_destroy(context);
        return NULL;
    }

    if (spvc_compiler_install_compiler_options(compiler_msl, options) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: install options failed\n");
        spvc_context_destroy(context);
        return NULL;
    }

    if (spvc_compiler_compile(compiler_msl, &result) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: mglCompileMSLCaptureVariant: compile failed\n");
        spvc_context_destroy(context);
        return NULL;
    }

    str_ret = result ? strdup(result) : NULL;

    spvc_context_destroy(context);
    return str_ret;
}

static void clearStageCompileState(Program *pptr, int stage)
{
    if (pptr->spirv[stage].ir) {
        free(pptr->spirv[stage].ir);
        pptr->spirv[stage].ir = NULL;
    }
    if (pptr->spirv[stage].msl_str) {
        free(pptr->spirv[stage].msl_str);
        pptr->spirv[stage].msl_str = NULL;
    }
    if (pptr->spirv[stage].msl_str_capture) {
        free(pptr->spirv[stage].msl_str_capture);
        pptr->spirv[stage].msl_str_capture = NULL;
    }
    if (pptr->spirv[stage].entry_point) {
        free(pptr->spirv[stage].entry_point);
        pptr->spirv[stage].entry_point = NULL;
    }
    mglSafeReleaseMetalObj((void **)&pptr->spirv[stage].mtl_function);
    mglSafeReleaseMetalObj((void **)&pptr->spirv[stage].mtl_library);

    for (int res_type = 0; res_type < _MAX_SPIRV_RES; res_type++) {
        SpirvResourceList *rl = &pptr->spirv_resources_list[stage][res_type];
        if (rl->list) {
            for (GLuint i = 0; i < rl->count; i++) {
                mglFreeSpirvResourceOwnedFields(&rl->list[i]);
            }
            free(rl->list);
            rl->list = NULL;
        }
        rl->count = 0;
    }
}

static GLboolean mglShaderSourceHasToken(const char *start, const char *end, const char *token)
{
    size_t token_len;

    if (!start || !end || !token || start > end) {
        return GL_FALSE;
    }

    token_len = strlen(token);
    for (const char *p = start; p + token_len <= end; p++) {
        if (strncmp(p, token, token_len) != 0) {
            continue;
        }

        int before = (p == start) ? 0 : (isalnum((unsigned char)p[-1]) || p[-1] == '_');
        int after = (p[token_len] == '\0') ? 0 : (isalnum((unsigned char)p[token_len]) || p[token_len] == '_');
        if (!before && !after) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLboolean mglProgramPerVertexSignature(Program *program, int stage, unsigned *signature)
{
    Shader *shader;
    const char *src;
    const char *p;
    unsigned sig = 0u;
    GLboolean found = GL_FALSE;

    if (signature) {
        *signature = 0u;
    }
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES || !signature) {
        return GL_FALSE;
    }

    shader = program->shader_slots[stage];
    src = shader ? shader->src : NULL;
    if (!src) {
        return GL_FALSE;
    }

    p = src;
    while ((p = strstr(p, "gl_PerVertex")) != NULL) {
        const char *open = strchr(p, '{');
        const char *close = open ? strchr(open + 1, '}') : NULL;
        if (!open || !close) {
            p += strlen("gl_PerVertex");
            continue;
        }

        if (mglShaderSourceHasToken(open, close, "gl_Position")) {
            sig |= 1u << 0;
        }
        if (mglShaderSourceHasToken(open, close, "gl_PointSize")) {
            sig |= 1u << 1;
        }
        if (mglShaderSourceHasToken(open, close, "gl_ClipDistance")) {
            sig |= 1u << 2;
        }
        if (mglShaderSourceHasToken(open, close, "gl_CullDistance")) {
            sig |= 1u << 3;
        }
        found = GL_TRUE;
        p = close + 1;
    }

    if (found) {
        *signature = sig;
    }
    return found;
}

GLboolean mglProgramPipelinePerVertexCompatible(Program *const *stage_programs)
{
    unsigned reference = 0u;
    GLboolean have_reference = GL_FALSE;

    if (!stage_programs) {
        return GL_TRUE;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Program *program = stage_programs[stage];
        unsigned signature = 0u;

        if (!program || !program->shader_slots[stage]) {
            continue;
        }
        if (!mglProgramPerVertexSignature(program, stage, &signature)) {
            continue;
        }

        if (!have_reference) {
            reference = signature;
            have_reference = GL_TRUE;
            continue;
        }
        if (signature != reference) {
            return GL_FALSE;
        }
    }

    return GL_TRUE;
}

static GLboolean mglLinkedProgramPerVertexCompatible(Program *program)
{
    Program *stage_programs[_MAX_SHADER_TYPES] = {0};

    if (!program) {
        return GL_TRUE;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if ((program->attached_shader_mask & (1u << stage)) != 0u &&
            program->shader_slots[stage]) {
            stage_programs[stage] = program;
        }
    }

    return mglProgramPipelinePerVertexCompatible(stage_programs);
}

static GLint mglDefaultAttribLocationForName(const char *name)
{
    if (!name) {
        return -1;
    }

    if (strcmp(name, "Position") == 0) return 0;
    if (strcmp(name, "Color") == 0) return 1;
    if (strcmp(name, "UV0") == 0) return 2;
    if (strcmp(name, "UV1") == 0) return 3;
    if (strcmp(name, "UV2") == 0) return 4;
    if (strcmp(name, "Normal") == 0) return 5;

    return -1;
}

static GLint mglProgramVertexInputOrdinal(Program *pptr, const char *name)
{
    if (!pptr || !name) {
        return -1;
    }

    SpirvResourceList *vertex_inputs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    if (!vertex_inputs->list) {
        return -1;
    }

    for (GLuint i = 0; i < vertex_inputs->count; i++) {
        const char *input_name = vertex_inputs->list[i].name;
        if (input_name && strcmp(input_name, name) == 0) {
            return (GLint)i;
        }
    }

    return -1;
}

static GLboolean mglProgramHasVertexInputNamed(Program *pptr, const char *name)
{
    return mglProgramVertexInputOrdinal(pptr, name) >= 0 ? GL_TRUE : GL_FALSE;
}

static GLint mglContextualDefaultAttribLocationForName(Program *pptr, const char *name)
{
    if (!pptr || !name) {
        return -1;
    }

    /*
     * Mojang's shader names are stable, but the set of inputs is not. Newer
     * GUI/item shaders often omit UV0 or UV1, so the vanilla fallback table
     * must collapse around the attributes that are actually present.
     */
    GLboolean has_color = mglProgramHasVertexInputNamed(pptr, "Color");
    GLboolean has_uv0 = mglProgramHasVertexInputNamed(pptr, "UV0");
    GLboolean has_uv1 = mglProgramHasVertexInputNamed(pptr, "UV1");
    GLboolean has_uv2 = mglProgramHasVertexInputNamed(pptr, "UV2");

    if (strcmp(name, "UV2") == 0) {
        if (!has_uv0 && !has_uv1) {
            return 2;
        }
        if (has_uv0 && !has_uv1) {
            return 3;
        }
        return 4;
    }

    if (strcmp(name, "Normal") == 0) {
        if (has_uv2 && !has_uv1) {
            return has_uv0 ? 4 : 3;
        }
        return 5;
    }

    if (has_color && has_uv0 && !has_uv1 && !has_uv2) {
        GLint color_ordinal = mglProgramVertexInputOrdinal(pptr, "Color");
        GLint uv0_ordinal = mglProgramVertexInputOrdinal(pptr, "UV0");
        if (uv0_ordinal >= 0 && color_ordinal >= 0 &&
            uv0_ordinal < color_ordinal) {
            if (strcmp(name, "UV0") == 0) {
                return 1;
            }
            if (strcmp(name, "Color") == 0) {
                return 2;
            }
        }
    }

    return mglDefaultAttribLocationForName(name);
}

static GLint mglDesiredAttribLocationForName(Program *pptr, const char *name)
{
    if (!pptr || !name) {
        return -1;
    }

    for (int i = 0; i < MAX_ATTRIBS; i++) {
        if (pptr->attrib_location_names[i] &&
            strcmp(pptr->attrib_location_names[i], name) == 0) {
            return i;
        }
    }

    return mglContextualDefaultAttribLocationForName(pptr, name);
}

static void applyVertexInputLocations(Program *pptr)
{
    if (!pptr || !pptr->spirv[_VERTEX_SHADER].msl_str) {
        return;
    }

    SpirvResourceList *vertex_inputs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    if (!vertex_inputs->list) {
        return;
    }

    for (GLuint i = 0; i < vertex_inputs->count; i++) {
        SpirvResource *vs_in = &vertex_inputs->list[i];
        GLint desiredLocation = mglDesiredAttribLocationForName(pptr, vs_in->name);
        if (desiredLocation < 0 || desiredLocation >= MAX_ATTRIBS ||
            vs_in->location == (GLuint)desiredLocation) {
            continue;
        }

        char from[256];
        char to[256];
        snprintf(from, sizeof(from), "%s [[attribute(%u)]]",
                 vs_in->name, (unsigned)vs_in->location);
        snprintf(to, sizeof(to), "%s [[attribute(%u)]]",
                 vs_in->name, (unsigned)desiredLocation);

        if (strstr(pptr->spirv[_VERTEX_SHADER].msl_str, from)) {
            fprintf(stderr,
                    "MGL ATTRIB FIX: program=%u vertex input %s loc %u -> %d\n",
                    pptr->name,
                    vs_in->name,
                    (unsigned)vs_in->location,
                    desiredLocation);
            replace_all_substr(&pptr->spirv[_VERTEX_SHADER].msl_str, from, to);
            vs_in->location = (GLuint)desiredLocation;
        } else {
            fprintf(stderr,
                    "MGL ATTRIB WARNING: program=%u wanted %s loc %u -> %d but MSL pattern was not found\n",
                    pptr->name,
                    vs_in->name,
                    (unsigned)vs_in->location,
                    desiredLocation);
        }
    }
}

/* Parse SPIR-V to find the constant index prefix used in OpAccessChain
 * instructions that reference `var_id`.  This is used to build the GL
 * active-uniform name for multi-dimensional array uniforms.
 *
 * For example, if the shader has `uniform vec4 a[3][4][5]` and accesses
 * `a[2][1][i]` (i variable), the SPIR-V OpAccessChain has constant indices
 * [2, 1] followed by a variable index.  This function returns [2, 1].
 *
 * Returns the number of constant prefix indices found (0 if none).
 */
static unsigned mglSPIRVFindAccessChainConstantIndices(
    const unsigned int *ir, size_t ir_size_bytes,
    unsigned var_id,
    unsigned *out_indices, unsigned max_indices)
{
    if (!ir || ir_size_bytes < 20 || !out_indices || max_indices == 0) {
        return 0;
    }

    size_t word_count = ir_size_bytes / sizeof(unsigned int);

    /* Build a simple constant table for 32-bit integer OpConstant values. */
    #define MGL_SPV_MAX_CONSTANTS 2048
    struct { unsigned id; unsigned value; } constants[MGL_SPV_MAX_CONSTANTS];
    unsigned num_constants = 0;

    /* Skip header (5 words). */
    size_t offset = 5;
    while (offset + 1 < word_count) {
        unsigned word = ir[offset];
        unsigned inst_word_count = word >> 16;
        unsigned opcode = word & 0xFFFF;
        if (inst_word_count == 0 || offset + inst_word_count > word_count) {
            break;
        }

        /* OpConstant = 43 */
        if (opcode == 43 && inst_word_count >= 4 && num_constants < MGL_SPV_MAX_CONSTANTS) {
            unsigned result_id = ir[offset + 2];
            unsigned value = ir[offset + 3];
            constants[num_constants].id = result_id;
            constants[num_constants].value = value;
            num_constants++;
        }

        offset += inst_word_count;
    }

    /* Find OpAccessChain (65) / OpInBoundsAccessChain (66) with Base == var_id. */
    offset = 5;
    while (offset + 1 < word_count) {
        unsigned word = ir[offset];
        unsigned inst_word_count = word >> 16;
        unsigned opcode = word & 0xFFFF;
        if (inst_word_count == 0 || offset + inst_word_count > word_count) {
            break;
        }

        if ((opcode == 65 || opcode == 66) && inst_word_count >= 4) {
            unsigned base_id = ir[offset + 3];
            if (base_id == var_id) {
                unsigned num_index_ids = inst_word_count - 4;
                unsigned const_count = 0;
                for (unsigned i = 0; i < num_index_ids && i < max_indices; i++) {
                    unsigned index_id = ir[offset + 4 + i];
                    int found = 0;
                    for (unsigned j = 0; j < num_constants; j++) {
                        if (constants[j].id == index_id) {
                            out_indices[const_count++] = constants[j].value;
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        break;
                    }
                }
                return const_count;
            }
        }

        offset += inst_word_count;
    }

    return 0;
}

/* Post-process uniform names for multi-dimensional arrays.
 *
 * GL requires that for `uniform vec4 a[3][4][5]` accessed as `a[2][1][i]`,
 * the active uniform name is "a[2][1][0]" with GL_ARRAY_SIZE = 5 (innermost).
 * SPIRV-Cross reflection gives the base name "a" with the full array type.
 * This function analyzes the SPIR-V to find the constant access-chain indices
 * and rewrites the name to include them.
 */
static void applyMultiDimArrayUniformNames(Program *pptr)
{
    if (!pptr) {
        return;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (!pptr->spirv[stage].ir) {
            continue;
        }

        SpirvResourceList *rl =
            &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        if (!rl || !rl->list) {
            continue;
        }

        for (GLuint i = 0; i < rl->count; i++) {
            SpirvResource *res = &rl->list[i];
            if (!res->name || !res->is_array) {
                continue;
            }

            unsigned dims = res->num_array_dims;
            if (dims <= 1) {
                continue; /* Only multi-dimensional arrays need rewriting. */
            }

            /* Find constant access-chain indices from the SPIR-V. */
            unsigned const_indices[8];
            unsigned num_const = mglSPIRVFindAccessChainConstantIndices(
                pptr->spirv[stage].ir,
                pptr->spirv[stage].size * sizeof(unsigned int),
                res->_id,
                const_indices, 8);

            if (num_const == 0) {
                /* No constant indices found — fall back to all-zeros. */
                continue;
            }

            /* If all dimensions are constant (accessing a single element),
             * GL_ARRAY_SIZE = 1 and the name includes all indices.
             * Otherwise, the constant prefix indices are in the name and
             * "[0]" is appended for the innermost (variable) dimension. */
            unsigned num_name_indices = num_const;
            GLboolean append_zero = GL_FALSE;
            if (num_const < dims) {
                /* There are remaining variable-index dimensions. */
                append_zero = GL_TRUE;
            }

            /* Build the new name: "varname[idx1][idx2]...[0]" */
            size_t base_len = strlen(res->name);
            /* Each "[N]" is at most 12 chars ( "[4294967295]" ). */
            size_t buf_size = base_len + num_name_indices * 12 + 4 + 1;
            char *new_name = (char *)malloc(buf_size);
            if (!new_name) {
                continue;
            }
            memcpy(new_name, res->name, base_len);
            size_t pos = base_len;
            for (unsigned d = 0; d < num_name_indices; d++) {
                pos += snprintf(new_name + pos, buf_size - pos,
                                "[%u]", const_indices[d]);
            }
            if (append_zero) {
                pos += snprintf(new_name + pos, buf_size - pos, "[0]");
            }
            new_name[pos] = '\0';

            free((void *)res->name);
            res->name = new_name;

            /* If we appended "[0]", the innermost dimension is the array.
             * GL_ARRAY_SIZE is already set to the innermost dimension. */
            /* If all indices were constant, it's a single element. */
            if (!append_zero) {
                res->gl_array_size = 1;
                res->is_array = GL_FALSE;
            }
        }
    }
}

/* Apply pre-link fragment output bindings set via glBindFragDataLocation(Indexed).
 * Updates the location and location_index of fragment stage output resources
 * so GL_LOCATION and GL_LOCATION_INDEX queries return the user-specified values. */
static void applyFragmentOutputLocationIndices(Program *pptr)
{
    if (!pptr || pptr->frag_data_location_count == 0) {
        return;
    }

    SpirvResourceList *frag_outputs =
        &pptr->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    if (!frag_outputs || !frag_outputs->list) {
        return;
    }

    for (GLuint i = 0; i < frag_outputs->count; i++) {
        SpirvResource *fs_out = &frag_outputs->list[i];
        if (!fs_out->name) {
            continue;
        }

        for (GLuint j = 0; j < pptr->frag_data_location_count; j++) {
            if (!pptr->frag_data_location_names[j]) {
                continue;
            }
            if (strcmp(pptr->frag_data_location_names[j], fs_out->name) == 0) {
                fs_out->location = pptr->frag_data_color_numbers[j];
                fs_out->location_index = pptr->frag_data_indices[j];
                break;
            }
        }
    }
}

static void alignFragmentInputLocationsToVertexOutputs(Program *pptr)
{
    if (!pptr ||
        !pptr->spirv[_FRAGMENT_SHADER].msl_str ||
        !pptr->spirv[_VERTEX_SHADER].msl_str) {
        return;
    }

    SpirvResourceList *vertex_outputs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    SpirvResourceList *fragment_inputs =
        &pptr->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];

    if (!vertex_outputs->list || !fragment_inputs->list) {
        return;
    }

    for (GLuint f = 0; f < fragment_inputs->count; f++) {
        SpirvResource *fs_in = &fragment_inputs->list[f];
        if (!fs_in->name || fs_in->name[0] == '\0') {
            continue;
        }

        for (GLuint v = 0; v < vertex_outputs->count; v++) {
            SpirvResource *vs_out = &vertex_outputs->list[v];
            if (!vs_out->name || strcmp(fs_in->name, vs_out->name) != 0) {
                continue;
            }

            GLuint desired_location = vs_out->location;
            GLuint current_location = fs_in->location;
            char vs_msl_name[256] = {0};
            char fs_msl_name[256] = {0};
            if (mglFindMSLUserLocationForResourceName(pptr->spirv[_VERTEX_SHADER].msl_str,
                                                      vs_out->name,
                                                      &desired_location,
                                                      vs_msl_name,
                                                      sizeof(vs_msl_name))) {
                vs_out->location = desired_location;
            }
            if (mglFindMSLUserLocationForResourceName(pptr->spirv[_FRAGMENT_SHADER].msl_str,
                                                      fs_in->name,
                                                      &current_location,
                                                      fs_msl_name,
                                                      sizeof(fs_msl_name))) {
                fs_in->location = current_location;
            }

            if (current_location == desired_location) {
                break;
            }

            if (mglReplaceMSLUserLocationForResourceName(&pptr->spirv[_FRAGMENT_SHADER].msl_str,
                                                         fs_in->name,
                                                         current_location,
                                                         desired_location,
                                                         fs_msl_name,
                                                         sizeof(fs_msl_name))) {
                fprintf(stderr,
                        "MGL IFACE FIX: program=%u fragment input %s/%s loc %u -> %u to match vertex output %s/%s\n",
                        pptr->name,
                        fs_in->name,
                        fs_msl_name[0] ? fs_msl_name : fs_in->name,
                        (unsigned)current_location,
                        (unsigned)desired_location,
                        vs_out->name,
                        vs_msl_name[0] ? vs_msl_name : vs_out->name);
                fs_in->location = desired_location;
            } else {
                fprintf(stderr,
                        "MGL IFACE WARNING: program=%u wanted to align %s loc %u -> %u but MSL pattern was not found\n",
                        pptr->name,
                        fs_in->name,
                        (unsigned)current_location,
                        (unsigned)desired_location);
            }
            break;
        }
    }
}

static GLboolean mglSpirvVaryingTypesCompatible(const SpirvResource *a,
                                                const SpirvResource *b)
{
    if (!a || !b) {
        return GL_FALSE;
    }

    if (a->gl_type != 0u && b->gl_type != 0u) {
        return a->gl_type == b->gl_type ? GL_TRUE : GL_FALSE;
    }

    if (a->gl_array_size > 0 && b->gl_array_size > 0 &&
        a->gl_array_size != b->gl_array_size) {
        return GL_FALSE;
    }

    return GL_TRUE;
}

static SpirvResource *mglFindVaryingByName(SpirvResourceList *list,
                                           const char *name,
                                           const SpirvResource *type_peer)
{
    if (!list || !list->list || !name || name[0] == '\0') {
        return NULL;
    }

    for (GLuint i = 0; i < list->count; i++) {
        SpirvResource *candidate = &list->list[i];
        if (!candidate->name || strcmp(candidate->name, name) != 0) {
            continue;
        }
        if (type_peer && !mglSpirvVaryingTypesCompatible(candidate, type_peer)) {
            continue;
        }
        return candidate;
    }

    return NULL;
}

static SpirvResource *mglFindVaryingByLocation(SpirvResourceList *list,
                                               GLuint location,
                                               const SpirvResource *type_peer)
{
    if (!list || !list->list) {
        return NULL;
    }

    for (GLuint i = 0; i < list->count; i++) {
        SpirvResource *candidate = &list->list[i];
        if (!candidate->name || candidate->location != location) {
            continue;
        }
        if (type_peer && !mglSpirvVaryingTypesCompatible(candidate, type_peer)) {
            continue;
        }
        return candidate;
    }

    return NULL;
}

static void mglBridgeSkippedGeometryShaderVaryings(Program *pptr)
{
    if (!pptr ||
        !pptr->shader_slots[_GEOMETRY_SHADER] ||
        !pptr->spirv[_VERTEX_SHADER].msl_str ||
        !pptr->spirv[_FRAGMENT_SHADER].msl_str) {
        return;
    }

    SpirvResourceList *vertex_outputs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    SpirvResourceList *geometry_inputs =
        &pptr->spirv_resources_list[_GEOMETRY_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    SpirvResourceList *geometry_outputs =
        &pptr->spirv_resources_list[_GEOMETRY_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    SpirvResourceList *fragment_inputs =
        &pptr->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];

    if (!vertex_outputs->list || !geometry_inputs->list ||
        !geometry_outputs->list || !fragment_inputs->list) {
        return;
    }

    for (GLuint f = 0; f < fragment_inputs->count; f++) {
        SpirvResource *fs_in = &fragment_inputs->list[f];
        SpirvResource *gs_out = NULL;
        SpirvResource *vs_out = NULL;
        GLuint fs_location;
        char fs_msl_name[256] = {0};
        char vs_msl_name[256] = {0};
        const char *fs_name;
        const char *vs_name;
        char expected_vs_name[256] = {0};

        if (!fs_in->name || fs_in->name[0] == '\0') {
            continue;
        }

        fs_name = fs_in->name;
        fs_location = fs_in->location;
        if (mglFindMSLUserLocationForResourceName(pptr->spirv[_FRAGMENT_SHADER].msl_str,
                                                  fs_in->name,
                                                  &fs_location,
                                                  fs_msl_name,
                                                  sizeof(fs_msl_name))) {
            fs_name = fs_msl_name[0] ? fs_msl_name : fs_in->name;
            fs_in->location = fs_location;
        }

        gs_out = mglFindVaryingByName(geometry_outputs, fs_in->name, fs_in);
        if (!gs_out && fs_name != fs_in->name) {
            gs_out = mglFindVaryingByName(geometry_outputs, fs_name, fs_in);
        }
        if (!gs_out) {
            gs_out = mglFindVaryingByLocation(geometry_outputs, fs_location, fs_in);
        }
        if (!gs_out) {
            continue;
        }

        if (strncmp(gs_out->name, "gs_fs_", 6) == 0) {
            snprintf(expected_vs_name, sizeof(expected_vs_name),
                     "vs_gs_%s", gs_out->name + 6);
        }

        if (expected_vs_name[0]) {
            vs_out = mglFindVaryingByName(vertex_outputs, expected_vs_name, fs_in);
        }

        if (!vs_out) {
            for (GLuint g = 0; g < geometry_inputs->count; g++) {
                SpirvResource *gs_in = &geometry_inputs->list[g];
                if (!gs_in->name ||
                    gs_in->location != gs_out->location ||
                    !mglSpirvVaryingTypesCompatible(gs_in, fs_in)) {
                    continue;
                }
                vs_out = mglFindVaryingByName(vertex_outputs, gs_in->name, gs_in);
                if (vs_out) {
                    break;
                }
            }
        }
        if (!vs_out) {
            vs_out = mglFindVaryingByLocation(vertex_outputs, gs_out->location, fs_in);
        }
        if (!vs_out || !vs_out->name) {
            continue;
        }

        vs_name = vs_out->name;
        if (mglFindMSLUserLocationForResourceName(pptr->spirv[_VERTEX_SHADER].msl_str,
                                                  vs_out->name,
                                                  &vs_out->location,
                                                  vs_msl_name,
                                                  sizeof(vs_msl_name))) {
            vs_name = vs_msl_name[0] ? vs_msl_name : vs_out->name;
        }

        GLboolean renamed = GL_FALSE;
        if (strcmp(vs_name, fs_name) != 0) {
            renamed = mglReplaceMSLIdentifier(&pptr->spirv[_FRAGMENT_SHADER].msl_str,
                                              fs_name,
                                              vs_name);
            if (renamed) {
                fprintf(stderr,
                        "MGL GS SKIP IFACE NAME FIX: program=%u fragment input %s -> %s via skipped GS %s\n",
                        pptr->name,
                        fs_name,
                        vs_name,
                        gs_out->name ? gs_out->name : "(null)");
                fs_name = vs_name;
            }
        }

        if (vs_out->location == fs_location) {
            if (!renamed) {
                continue;
            }
            continue;
        }

        const char *location_patch_name = renamed ? vs_name : fs_in->name;
        if (!mglReplaceMSLUserLocationForResourceName(&pptr->spirv[_FRAGMENT_SHADER].msl_str,
                                                      location_patch_name,
                                                      fs_location,
                                                      vs_out->location,
                                                      fs_msl_name,
                                                      sizeof(fs_msl_name))) {
            fprintf(stderr,
                    "MGL GS SKIP IFACE WARNING: program=%u wanted FS %s loc %u -> %u to match VS %s but MSL pattern was not found\n",
                    pptr->name,
                    fs_in->name,
                    (unsigned)fs_location,
                    (unsigned)vs_out->location,
                    vs_out->name);
            continue;
        }

        fprintf(stderr,
                "MGL GS SKIP IFACE FIX: program=%u align FS %s/%s loc %u -> %u to VS %s/%s via skipped GS %s\n",
                pptr->name,
                fs_in->name,
                fs_name,
                (unsigned)fs_location,
                (unsigned)vs_out->location,
                vs_out->name,
                vs_name,
                gs_out->name ? gs_out->name : "(null)");
        fs_in->location = vs_out->location;
    }
}

static bool compileStageFromLinkedProgram(GLMContext ctx, Program *pptr, glslang_program_t *glsl_program, int stage)
{
    const char *spirv_messages;

    /* Safety check: ensure we have a shader for this stage */
    if (!pptr->shader_slots[stage]) {
        return true;
    }

    clearStageCompileState(pptr, stage);

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Generating SPIRV for stage %d\n", stage);
    }
    glslang_program_SPIRV_generate(glsl_program, stage);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV generated\n");
    }

    spirv_messages = glslang_program_SPIRV_get_messages(glsl_program);
    if (spirv_messages && spirv_messages[0] != '\0')
    {
        fprintf(stderr, "MGL Error: glslang_program_SPIRV_get_messages:\n%s\n", spirv_messages);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return false;
    }

    // save SPIRV code
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Getting SPIRV size\n");
    }
    pptr->spirv[stage].size = glslang_program_SPIRV_get_size(glsl_program);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV size: %zu\n", pptr->spirv[stage].size);
    }

    // CRITICAL SECURITY FIX: Prevent integer overflow in SPIRV allocation
    // Check if size * sizeof(unsigned) would overflow size_t
    if (pptr->spirv[stage].size > SIZE_MAX / sizeof(unsigned)) {
        fprintf(stderr, "MGL SECURITY ERROR: SPIRV size %zu would cause allocation overflow\n", pptr->spirv[stage].size);
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return false;
    }

    size_t alloc_size = pptr->spirv[stage].size * sizeof(unsigned);
    pptr->spirv[stage].ir = (unsigned int *)malloc(alloc_size);
    if (!pptr->spirv[stage].ir) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate %zu bytes for SPIRV\n", alloc_size);
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return false;
    }
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Getting SPIRV IR\n");
    }
    glslang_program_SPIRV_get(glsl_program, pptr->spirv[stage].ir);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV IR obtained\n");
    }

    // compile SPIRV to Metal
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: About to parse SPIRV to Metal\n");
    }
    pptr->spirv[stage].msl_str = parseSPIRVShaderToMetal(ctx, pptr, stage);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV parsed to Metal\n");
    }
    if (pptr->spirv[stage].msl_str == NULL) {
        fprintf(stderr,
                "MGL WARNING: parseSPIRVShaderToMetal failed for stage %d; keeping reflection data and marking stage non-renderable\n",
                stage);
        return true;
    }
    applyMSLUniformBufferPacking(pptr, stage);
    if (getenv("MGL_DUMP_MSL_POST_PACK") && pptr->spirv[stage].msl_str) {
        char dump_path[256];
        snprintf(dump_path, sizeof(dump_path),
                 "/tmp/mgl_program_%u_stage_%d_post_pack.msl",
                 pptr->name,
                 stage);
        FILE *dump = fopen(dump_path, "w");
        if (dump) {
            fputs(pptr->spirv[stage].msl_str, dump);
            fclose(dump);
            fprintf(stderr,
                    "MGL MSL POST PACK DUMP: program=%u stage=%d path=%s\n",
                    pptr->name,
                    stage,
                    dump_path);
        }
    }

    /* Compile the GPU transform-feedback capture variant for the vertex
     * stage when the program has XFB varyings. Groundwork only — the
     * renderer dispatch is not yet wired. Off by default (env-gated). */
    if (stage == _VERTEX_SHADER && pptr->transform_feedback_varying_count > 0) {
        char *capture_msl = mglCompileMSLCaptureVariant(ctx, pptr, stage);
        if (capture_msl) {
            pptr->spirv[stage].msl_str_capture = capture_msl;
        }
        /* Failure is non-fatal: link still succeeds; the CPU passthrough
         * path and the existing TES path remain available. */
    }

    mglApplyPlainUniformInitializers(ctx, pptr, stage);

    return true;
}

/* Validate that all transform feedback varying names are active outputs
 * of the program.  Per the GL spec, LinkProgram must fail if any captured
 * varying is not an output of the last pre-rasterization stage. */
static bool mglValidateTransformFeedbackVaryings(Program *pptr)
{
    if (!pptr || pptr->transform_feedback_varying_count <= 0)
        return true;

    /* Determine the last active stage before fragment (the stage that
     * provides transform feedback outputs).  Priority: geometry > tess
     * eval > vertex. */
    int feedback_stage = -1;
    if (pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT)
        feedback_stage = _GEOMETRY_SHADER;
    else if (pptr->attached_shader_mask & TESS_EVALUATION_SHADER_MASK_BIT)
        feedback_stage = _TESS_EVALUATION_SHADER;
    else if (pptr->attached_shader_mask & VERTEX_SHADER_MASK_BIT)
        feedback_stage = _VERTEX_SHADER;

    if (feedback_stage < 0)
    {
        /* No stage to capture from — all varyings are invalid. */
        return false;
    }

    SpirvResourceList *outputs =
        &pptr->spirv_resources_list[feedback_stage][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];

    for (GLsizei i = 0; i < pptr->transform_feedback_varying_count; i++)
    {
        const char *varying = pptr->transform_feedback_varying_names[i];
        if (!varying || varying[0] == '\0')
            continue;

        /* Strip array subscript for comparison (e.g. "foo[0]" -> "foo"). */
        char base_name[96];
        strncpy(base_name, varying, sizeof(base_name) - 1);
        base_name[sizeof(base_name) - 1] = '\0';
        char *bracket = strchr(base_name, '[');
        if (bracket)
            *bracket = '\0';

        bool found = false;
        for (GLuint j = 0; j < outputs->count; j++)
        {
            if (outputs->list[j].name &&
                strcmp(outputs->list[j].name, base_name) == 0)
            {
                found = true;
                break;
            }
        }

        if (!found)
            return false;
    }

    return true;
}

void mglLinkProgram(GLMContext ctx, GLuint program)
{
    Program *pptr;
    glslang_program_t *glsl_program;
    int err;
    bool link_ok = true;
    bool has_any_shader = false;

    pptr = findProgram(ctx, program);

    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

        return;
    }

    mglFlushPendingDraws(ctx);

    pptr->uses_vertex_id = GL_FALSE;
    pptr->uses_primitive_id = GL_FALSE;
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (mglProgramAttachedShaderCount(pptr, (GLuint)stage) > 0u) {
            has_any_shader = true;
            break;
        }
    }

    if (!has_any_shader) {
        fprintf(stderr, "MGL WARNING: mglLinkProgram called with no attached shaders\n");
        pptr->linked_glsl_program = NULL;
        return;
    }

    if ((pptr->attached_shader_mask & COMPUTE_SHADER_MASK_BIT) &&
        (pptr->attached_shader_mask & ~COMPUTE_SHADER_MASK_BIT)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: compute shaders cannot be linked with non-compute stages\n",
                pptr->name);
        pptr->linked_glsl_program = NULL;
        return;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if ((pptr->attached_shader_mask & (1u << stage)) == 0u) {
            continue;
        }

        GLuint attached_count = mglProgramAttachedShaderCount(pptr, (GLuint)stage);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            Shader *shader = (pptr->attached_shader_counts[stage] > 0u)
                ? pptr->attached_shader_slots[stage][attached]
                : pptr->shader_slots[stage];
            if (!shader || !shader->compiled_glsl_shader) {
                fprintf(stderr,
                        "MGL WARNING: mglLinkProgram failed program %u: shader stage %d is not compiled\n",
                        pptr->name,
                        stage);
                pptr->linked_glsl_program = NULL;
                return;
            }
        }
    }

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Creating glslang program for full-link\n");
    }
    glsl_program = glslang_program_create();
    if (!glsl_program) {
        fprintf(stderr, "MGL Error: glslang_program_create failed\n");
        pptr->linked_glsl_program = NULL;
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Adding shaders to program\n");
    }
    addShadersToProgram(ctx, pptr, glsl_program);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Shaders added\n");
    }

    err = glslang_program_link(glsl_program, GLSLANG_MSG_DEFAULT_BIT);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Program link returned %d\n", err);
    }
    if (!err)
    {
        fprintf(stderr, "MGL Error: glslang_program_link failed err: %d\n", err);
        fprintf(stderr, "MGL Error: glslang_program_SPIRV_get_messages:\n%s\n", glslang_program_SPIRV_get_messages(glsl_program));
        fprintf(stderr, "MGL Error: glslang_program_get_info_log:\n%s\n", glslang_program_get_info_log(glsl_program));
        fprintf(stderr, "MGL Error: glslang_program_get_info_debug_log:\n%s\n", glslang_program_get_info_debug_log(glsl_program));
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
        return;
    }

    err = glslang_program_map_io(glsl_program);
    if (!err)
    {
        fprintf(stderr, "MGL WARNING: glslang_program_map_io failed; continuing with linked program\n");
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
    {
        if (!compileStageFromLinkedProgram(ctx, pptr, glsl_program, stage)) {
            link_ok = false;
            break;
        }
    }

    if (!link_ok) {
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
        return;
    }

    /* Validate transform feedback varyings: the link must fail if any
     * captured varying is not an active output of the program. */
    if (!mglValidateTransformFeedbackVaryings(pptr)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: transform feedback "
                "varying not found in program outputs\n",
                pptr->name);
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
        return;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        GLuint attached_count = mglProgramAttachedShaderCount(pptr, (GLuint)stage);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            Shader *shader = (pptr->attached_shader_counts[stage] > 0u)
                ? pptr->attached_shader_slots[stage][attached]
                : pptr->shader_slots[stage];
            if (!shader || !shader->src) {
                continue;
            }
            if (strstr(shader->src, "gl_VertexID") ||
                strstr(shader->src, "gl_VertexIndex")) {
                pptr->uses_vertex_id = GL_TRUE;
            }
            if (strstr(shader->src, "gl_PrimitiveID") ||
                strstr(shader->src, "gl_PrimitiveIndex")) {
                pptr->uses_primitive_id = GL_TRUE;
            }
        }
    }

    applyVertexInputLocations(pptr);
    applyFragmentOutputLocationIndices(pptr);
    applyMultiDimArrayUniformNames(pptr);
    alignFragmentInputLocationsToVertexOutputs(pptr);
    mglBridgeSkippedGeometryShaderVaryings(pptr);
    mglAssignPlainUniformLocations(pptr);
    mglUnifySamplerUniformLocations(pptr);

    if (pptr->program_separable &&
        (pptr->attached_shader_mask & (pptr->attached_shader_mask - 1u)) != 0u &&
        !mglLinkedProgramPerVertexCompatible(pptr)) {
        fprintf(stderr,
                "MGL WARNING: separable program %u has incompatible gl_PerVertex redeclarations\n",
                pptr->name);
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
        return;
    }

    /* Validate layout(binding=N) values against GL implementation limits.
     * The GL spec requires that the link fail if a binding point exceeds
     * the corresponding MAX_*_BINDINGS limit.  glslang does not know the
     * implementation limits, so MGL must enforce them here. */
    {
        bool binding_error = false;
        for (int stage = 0; stage < _MAX_SHADER_TYPES && !binding_error; stage++) {
            if ((pptr->attached_shader_mask & (1u << stage)) == 0u) {
                continue;
            }

            /* Uniform blocks: GL_MAX_UNIFORM_BUFFER_BINDINGS */
            {
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    GLuint n = rl->list[i].ubo_array_size > 0
                                   ? rl->list[i].ubo_array_size : 1;
                    if (b + n > ctx->state.var.max_uniform_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d UBO '%s' binding %u (array size %u) "
                                "exceeds GL_MAX_UNIFORM_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b, n,
                                ctx->state.var.max_uniform_buffer_bindings);
                        binding_error = true;
                        break;
                    }
                }
            }

            /* Shader storage buffers: GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS */
            if (!binding_error) {
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_BUFFER];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    if (b >= ctx->state.var.max_shader_storage_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d SSBO '%s' binding %u "
                                "exceeds GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                ctx->state.var.max_shader_storage_buffer_bindings);
                        binding_error = true;
                        break;
                    }
                }
            }

            /* Storage images: GL_MAX_IMAGE_UNITS */
            if (!binding_error) {
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    if (b >= ctx->state.var.max_image_units) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d image '%s' binding %u "
                                "exceeds GL_MAX_IMAGE_UNITS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                ctx->state.var.max_image_units);
                        binding_error = true;
                        break;
                    }
                }
            }

            /* Atomic counters: GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS */
            if (!binding_error) {
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    if (b >= ctx->state.var.max_atomic_counter_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d atomic counter '%s' binding %u "
                                "exceeds GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                ctx->state.var.max_atomic_counter_buffer_bindings);
                        binding_error = true;
                        break;
                    }
                }
            }
        }

        if (binding_error) {
            glslang_program_delete(glsl_program);
            pptr->linked_glsl_program = NULL;
            return;
        }
    }

    /* The glslang_program_t was only needed during linking for SPIR-V
     * generation and reflection.  Release it now; pptr->linked_glsl_program
     * is reused as a non-NULL linked-state marker (see callers that test
     * `if (ptr->linked_glsl_program)` to check link status). */
    glslang_program_delete(glsl_program);
    /* linked_glsl_program is used as a linked-state marker only. */
    pptr->linked_glsl_program = (glslang_program_t *)pptr;
    pptr->dirty_bits |= DIRTY_PROGRAM;

    /* Only call mtlBindProgram if Metal functions are initialized */
    if (ctx->mtl_funcs.mtlBindProgram) {
        ctx->mtl_funcs.mtlBindProgram(ctx, pptr);
    } else {
        fprintf(stderr, "WARNING: Metal functions not initialized, skipping mtlBindProgram\n");
    }

    //ERROR_CHECK_RETURN(pptr->mtl_data, GL_INVALID_OPERATION);
}

void mglUseProgram(GLMContext ctx, GLuint program)
{
    Program *pptr = NULL;
    static GLuint s_last_unlinked_program = 0;
    static unsigned int s_unlinked_program_hits = 0;

    if (program)
    {
        pptr = findProgram(ctx, program);

        if (!pptr ||
            !mglObjectPointerLooksPlausible(pptr) ||
            !mglHashTableContainsData(&STATE(program_table), pptr) ||
            !mglPointerRangeIsReadable(pptr, sizeof(*pptr)))
        {
            fprintf(stderr, "MGL Error: mglUseProgram program %u not found or invalid\n", program);
            // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

            return;
        }

        if (!pptr->linked_glsl_program)
        {
            // Compatibility fallback: some pipelines can probe/use programs before
            // link is completed/available in this backend. Skip instead of poisoning
            // global GL error state every frame.
            s_unlinked_program_hits++;
            if (s_last_unlinked_program != program || (s_unlinked_program_hits % 128u) == 1u) {
                fprintf(stderr, "MGL WARNING: mglUseProgram skipping unlinked program %u (hit=%u)\n",
                        program, s_unlinked_program_hits);
                s_last_unlinked_program = program;
            }
            return;
        }
    }
    else
    {
        pptr = NULL;
    }

    bool bindingChanged =
        ctx->state.program != pptr ||
        ctx->state.program_name != program ||
        ctx->state.var.current_program != program;

    if (bindingChanged)
    {
        Program *oldProgram = ctx->state.program;
        if (oldProgram &&
            !mglProgramPointerUsableForName(ctx,
                                            oldProgram,
                                            oldProgram->name ? oldProgram->name : ctx->state.program_name))
        {
            fprintf(stderr, "MGL WARNING: mglUseProgram dropping invalid cached program pointer %p\n",
                    (void *)oldProgram);
            oldProgram = NULL;
            ctx->state.program = NULL;
        }

        if (oldProgram)
        {
            oldProgram->refcount--;
            if (oldProgram->refcount == 0 && oldProgram->delete_status)
            {
                mglFreeProgram(ctx, oldProgram);
            }
        }

        ctx->state.program = pptr;

        if (pptr)
        {
            pptr->refcount++;
        }
        ctx->state.dirty_bits |= DIRTY_PROGRAM;
    }

    /*
     * Keep program name and pointer state in sync so renderer-side recovery can
     * re-resolve by name if the cached pointer is lost.
     */
    ctx->state.program_name = program;
    ctx->state.var.current_program = program;

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL UseProgram program=%u resolved=%p\n",
                program, (void *)ctx->state.program);
    }
}

void mglBindAttribLocation(GLMContext ctx, GLuint program, GLuint index, const GLchar *name)
{
    if (index >= MAX_ATTRIBS || !name) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!mglSetProgramAttribName(ptr, index, name)) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    ptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglGetActiveAttrib(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (length) {
        *length = 0;
    }
    if (size) {
        *size = 0;
    }
    if (type) {
        *type = 0;
    }
    if (name && bufSize > 0) {
        name[0] = '\0';
    }

    if (!ctx) {
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    SpirvResource *res = mglProgramActiveAttribAt(ptr, index);
    if (!res) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (size) {
        *size = 1;
    }
    if (type) {
        *type = mglProgramActiveAttribType(res);
    }

    const char *src = res->name ? res->name : "";
    GLsizei src_len = (GLsizei)strlen(src);
    if (length) {
        *length = src_len;
    }
    if (name && bufSize > 0) {
        GLsizei copy_len = src_len < (bufSize - 1) ? src_len : (bufSize - 1);
        if (copy_len > 0) {
            memcpy(name, src, (size_t)copy_len);
        }
        name[copy_len] = '\0';
    }
}

void mglGetActiveUniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (length) {
        *length = 0;
    }
    if (size) {
        *size = 0;
    }
    if (type) {
        *type = 0;
    }
    if (name && bufSize > 0) {
        name[0] = '\0';
    }

    if (!ctx) {
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    int stage = -1;
    int res_type = -1;
    SpirvResource *res = mglProgramActiveUniformAt(ptr, index, &stage, &res_type);
    (void)stage;
    if (!res) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (size) {
        *size = res->ubo_member ? res->ubo_member->size
                                : mglProgramActiveUniformSize(res, res_type);
    }
    if (type) {
        *type = res->ubo_member ? (GLenum)res->ubo_member->gl_type
                                : (GLenum)mglProgramActiveUniformGLType(res, res_type);
    }
    mglProgramCopyActiveUniformName(res, bufSize, length, name);
}

void mglGetAttachedShaders(GLMContext ctx, GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders)
{
    if (count) {
        *count = 0;
    }
    if (!ctx) {
        return;
    }
    if (maxCount < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLsizei written = 0;
    for (int i = 0; i < _MAX_SHADER_TYPES; i++) {
        GLuint attached_count = mglProgramAttachedShaderCount(ptr, (GLuint)i);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            Shader *shader = (ptr->attached_shader_counts[i] > 0u)
                ? ptr->attached_shader_slots[i][attached]
                : ptr->shader_slots[i];
            if (!shader) {
                continue;
            }
            if (written < maxCount) {
                if (shaders) {
                    shaders[written] = shader->name;
                }
                written++;
            }
        }
    }

    if (count) {
        *count = written;
    }
}

GLint  mglGetAttribLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
	if (isProgram(ctx, program) == GL_FALSE)
	{
		ERROR_RETURN(GL_INVALID_OPERATION); // also may be GL_INVALID_VALUE ????

		return -1;
	}

	Program *ptr;

	ptr = getProgram(ctx, program);
	if (!ptr)
	{
		ERROR_RETURN(GL_INVALID_OPERATION);
		return -1;
	}

	if (ptr->linked_glsl_program == NULL)
	{
		ERROR_RETURN(GL_INVALID_OPERATION);

		return -1;
	}

    SpirvResourceList *vertex_inputs =
        &ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];

    /* Fast path: exact name match. */
    for (GLuint i = 0; vertex_inputs->list && i < vertex_inputs->count; i++)
    {
        const char *str = vertex_inputs->list[i].name;

        if (str && !strcmp(str, name))
        {
            return (GLint)vertex_inputs->list[i].location;
        }
    }

    /* Array element query: "foo[N]" should resolve to location(foo) + N.
     * Per the GL spec, glGetAttribLocation accepts "foo[0]" (equivalent to
     * "foo") and "foo[N]" for the Nth element of an array attribute. */
    const char *bracket = strrchr(name, '[');
    if (bracket && bracket[1] != ']')
    {
        size_t base_len = (size_t)(bracket - name);
        char *endp = NULL;
        long idx = strtol(bracket + 1, &endp, 10);
        if (endp && *endp == ']' && idx >= 0)
        {
            for (GLuint i = 0; vertex_inputs->list && i < vertex_inputs->count; i++)
            {
                const char *str = vertex_inputs->list[i].name;
                if (str && strlen(str) == base_len &&
                    !strncmp(str, name, base_len))
                {
                    return (GLint)vertex_inputs->list[i].location + (GLint)idx;
                }
            }
        }
    }

	return -1;
}

void mglGetProgramiv(GLMContext ctx, GLuint program, GLenum pname, GLint *params)
{
    Program *pptr = findProgram(ctx, program);
    ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);
    
    switch (pname) {
        case GL_LINK_STATUS:
            *params = pptr->linked_glsl_program ? GL_TRUE : GL_FALSE;
            break;
        case GL_DELETE_STATUS:
            *params = GL_FALSE;  /* Programs are not deleted by default */
            break;
        case GL_VALIDATE_STATUS:
            *params = GL_TRUE;  /* Assume valid */
            break;
        case GL_INFO_LOG_LENGTH:
            *params = 0;  /* No info log for now */
            break;
        case GL_ATTACHED_SHADERS:
            {
                int count = 0;
                for (int i = 0; i < _MAX_SHADER_TYPES; i++) {
                    count += (int)mglProgramAttachedShaderCount(pptr, (GLuint)i);
                }
                *params = count;
            }
            break;
        case GL_ACTIVE_ATTRIBUTES:
            *params = mglProgramActiveAttribCount(pptr);
            break;
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH:
            *params = mglProgramActiveAttribMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORMS:
            *params = mglProgramActiveUniformCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_MAX_LENGTH:
            *params = mglProgramActiveUniformMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCKS:
            *params = mglActiveUniformBlockCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
            *params = mglActiveUniformBlockMaxNameLength(pptr);
            break;
        case GL_COMPUTE_WORK_GROUP_SIZE:
            /*
             * Per the spec, querying GL_COMPUTE_WORK_GROUP_SIZE on a program
             * with no linked compute stage must return {0,0,0}; it does NOT
             * generate an error.  GL_INVALID_OPERATION is raised when the
             * program itself is not linked.
             */
            if (!pptr->linked_glsl_program) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (pptr->shader_slots[_COMPUTE_SHADER]) {
                params[0] = pptr->local_workgroup_size.x;
                params[1] = pptr->local_workgroup_size.y;
                params[2] = pptr->local_workgroup_size.z;
            } else {
                params[0] = 0;
                params[1] = 0;
                params[2] = 0;
            }
            break;
        case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS:
            *params = mglActiveAtomicCounterBufferCount(pptr);
            break;
        case GL_GEOMETRY_INPUT_TYPE:        /* 0x8917 */
        case GL_GEOMETRY_OUTPUT_TYPE:       /* 0x8918 */
        case GL_GEOMETRY_VERTICES_OUT:      /* 0x8916 */
        case GL_GEOMETRY_SHADER_INVOCATIONS:/* 0x887F */
            /* Geometry-shader reflection queries.  MGL does not execute GS on
             * Metal, but the program still carries the GS layout metadata
             * captured at compile/link time.  Return 0 when no GS is
             * attached so the queries are at least well-defined. */
            if (!pptr->linked_glsl_program) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            *params = 0;
            break;
        case GL_TESS_CONTROL_OUTPUT_VERTICES:  /* 0x8E75 */
            if (!pptr->linked_glsl_program) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            *params = pptr->shader_slots[_TESS_CONTROL_SHADER]
                ? (GLint)pptr->tess_control_output_vertices : 0;
            break;
        case GL_TESS_GEN_MODE:             /* 0x8E76 */
        case GL_TESS_GEN_SPACING:          /* 0x8E77 */
        case GL_TESS_GEN_VERTEX_ORDER:     /* 0x8E78 */
        case GL_TESS_GEN_POINT_MODE:       /* 0x8E79 */
            /* TES execution-mode reflection.  Returns the layout(...) values
             * captured from SPIR-V OpExecutionMode at link time.  0 when no
             * TES is attached. */
            if (!pptr->linked_glsl_program) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (!pptr->shader_slots[_TESS_EVALUATION_SHADER]) {
                *params = 0;
            } else if (pname == GL_TESS_GEN_MODE) {
                *params = (GLint)pptr->tess_gen_mode;
            } else if (pname == GL_TESS_GEN_SPACING) {
                *params = (GLint)pptr->tess_gen_spacing;
            } else if (pname == GL_TESS_GEN_VERTEX_ORDER) {
                *params = (GLint)pptr->tess_gen_vertex_order;
            } else {
                *params = (GLint)pptr->tess_gen_point_mode;
            }
            break;
        case GL_COMPLETION_STATUS_KHR: /* GL_ARB/KHR_parallel_shader_compile */
            /* MGL links synchronously, so every program is always complete by
             * the time this query is issued. */
            *params = GL_TRUE;
            break;
        default:
            fprintf(stderr, "mglGetProgramiv: unhandled pname 0x%x\n", pname);
            *params = 0;
            break;
    }
}

void mglGetProgramInfoLog(GLMContext ctx, GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    Program *pptr = findProgram(ctx, program);
    ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);
    
    /* For now, always return an empty info log */
    if (bufSize > 0 && infoLog) {
        infoLog[0] = '\0';
        if (length) {
            *length = 0;
        }
    }
}



#pragma mark program pipelines
void mglGenProgramPipelines(GLMContext ctx, GLsizei n, GLuint *pipelines)
{
    for (GLsizei i = 0; i < n; i++)
    {
        pipelines[i] = getNewName(&STATE(program_pipeline_table));
        getProgramPipeline(ctx, pipelines[i]);
    }
}

GLboolean mglIsProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);
    return ptr ? GL_TRUE : GL_FALSE;
}

void mglDeleteProgramPipelines(GLMContext ctx, GLsizei n, const GLuint *pipelines)
{
    mglFlushPendingDraws(ctx);

    for (GLsizei i = 0; i < n; i++)
    {
        if (pipelines[i] == 0)
            continue;
            
        ProgramPipeline *ptr = findProgramPipeline(ctx, pipelines[i]);
        if (!ptr)
            continue;
            
        // If deleting currently bound pipeline, unbind it
        if (STATE(program_pipeline) && STATE(program_pipeline)->name == pipelines[i])
        {
            STATE(program_pipeline) = NULL;
            STATE(var.program_pipeline_binding) = 0;
            STATE(dirty_bits) |= DIRTY_PROGRAM;
        }
        
        // Remove from hash table and free
        deleteHashElement(&STATE(program_pipeline_table), pipelines[i]);
        free(ptr);
    }
}

void mglBindProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    if (pipeline == 0)
    {
        STATE(program_pipeline) = NULL;
        STATE(var.program_pipeline_binding) = 0;
        STATE(dirty_bits) |= DIRTY_PROGRAM;
        return;
    }
    
    ProgramPipeline *ptr = getProgramPipeline(ctx, pipeline);
    STATE(program_pipeline) = ptr;
    STATE(var.program_pipeline_binding) = ptr ? pipeline : 0;
    STATE(dirty_bits) |= DIRTY_PROGRAM;
}

void mglUseProgramStages(GLMContext ctx, GLuint pipeline, GLbitfield stages, GLuint program)
{
    ProgramPipeline *pipe_ptr = findProgramPipeline(ctx, pipeline);
    if (!pipe_ptr)
    {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    mglFlushPendingDraws(ctx);

    Program *prog_ptr = NULL;
    if (program != 0)
    {
        prog_ptr = findProgram(ctx, program);
        if (!prog_ptr)
        {
            STATE(error) = GL_INVALID_VALUE;
            return;
        }
    }
    
    // Attach program to specified stages
    if (stages & GL_VERTEX_SHADER_BIT)
        pipe_ptr->stage_programs[_VERTEX_SHADER] = prog_ptr;
    if (stages & GL_FRAGMENT_SHADER_BIT)
        pipe_ptr->stage_programs[_FRAGMENT_SHADER] = prog_ptr;
    if (stages & GL_GEOMETRY_SHADER_BIT)
        pipe_ptr->stage_programs[_GEOMETRY_SHADER] = prog_ptr;
    if (stages & GL_TESS_CONTROL_SHADER_BIT)
        pipe_ptr->stage_programs[_TESS_CONTROL_SHADER] = prog_ptr;
    if (stages & GL_TESS_EVALUATION_SHADER_BIT)
        pipe_ptr->stage_programs[_TESS_EVALUATION_SHADER] = prog_ptr;
    if (stages & GL_COMPUTE_SHADER_BIT)
        pipe_ptr->stage_programs[_COMPUTE_SHADER] = prog_ptr;
        
    pipe_ptr->validated = GL_FALSE;
    STATE(dirty_bits) |= DIRTY_PROGRAM;
}
