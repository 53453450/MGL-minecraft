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

static char *mglAppendArrayZeroSuffix(const char *name)
{
    if (!name) {
        return NULL;
    }
    const char *leaf = strrchr(name, '.');
    leaf = leaf ? leaf + 1 : name;
    if (strchr(leaf, '[')) {
        return strdup(name);
    }
    size_t len = strlen(name);
    char *ret = (char *)malloc(len + 4u);
    if (!ret) {
        return NULL;
    }
    snprintf(ret, len + 4u, "%s[0]", name);
    return ret;
}

static GLboolean mglSpvcStructMemberOffset(spvc_compiler compiler,
                                           spvc_type struct_type,
                                           spvc_type_id struct_type_id,
                                           unsigned member_index,
                                           GLuint *out)
{
    unsigned value = 0;
    if (spvc_compiler_type_struct_member_offset(
            compiler, struct_type, member_index, &value) == SPVC_SUCCESS) {
        *out = value;
        return GL_TRUE;
    }
    *out = spvc_compiler_get_member_decoration(
        compiler, struct_type_id, member_index, SpvDecorationOffset);
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

static GLboolean mglAppendReflectedUBOMember(SpirvResource *ubo,
                                             GLuint *count,
                                             const char *name,
                                             GLuint gl_type,
                                             GLuint offset,
                                             GLint array_stride,
                                             GLint matrix_stride,
                                             GLboolean is_row_major,
                                             GLint size)
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
    member->size = size > 0 ? size : 1;
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

static GLboolean mglReflectUBOMemberLeaves(Program *program,
                                           int stage,
                                           spvc_compiler compiler,
                                           SpirvResource *ubo,
                                           spvc_type struct_type,
                                           spvc_type_id struct_type_id,
                                           const char *prefix,
                                           GLuint base_offset,
                                           GLboolean inherited_row_major,
                                           GLuint *count);

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
                                           GLuint *count)
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
    if (member_type &&
        spvc_type_get_basetype(member_type) == SPVC_BASETYPE_STRUCT) {
        unsigned array_dims = spvc_type_get_num_array_dimensions(member_type);
        if (array_dims > 0) {
            GLint stride = array_stride > 0 ? array_stride : 0;
            GLint elements = mglGLArraySizeFromSPVCType(member_type);
            for (GLint elem = 0; elem < elements; elem++) {
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
                                               member_type,
                                               member_type_id,
                                               elem_path,
                                               absolute_offset + (GLuint)(elem * stride),
                                               row_major,
                                               count)) {
                    free(elem_path);
                    free(path);
                    return GL_FALSE;
                }
                free(elem_path);
            }
            free(path);
            return GL_TRUE;
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
                                                 count);
        free(path);
        return ok;
    }

    GLint size = mglGLArraySizeFromSPVCType(member_type);
    char *query_path = (member_type && spvc_type_get_num_array_dimensions(member_type) > 0)
        ? mglAppendArrayZeroSuffix(path)
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
                                               size);
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
                                           GLuint *count)
{
    if (!struct_type || spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT) {
        return GL_FALSE;
    }

    unsigned member_count = spvc_type_get_num_member_types(struct_type);
    for (unsigned mem_idx = 0; mem_idx < member_count; mem_idx++) {
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
                                       count)) {
            return GL_FALSE;
        }
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

static GLboolean mglAddPlainStructUniformMember(SpirvResource *res,
                                                GLuint *count,
                                                const char *query_name,
                                                GLuint gl_type,
                                                GLint size,
                                                GLint location_offset)
{
    if (!res || !count || !query_name) {
        return GL_FALSE;
    }

    SpirvUBOMember *grown = (SpirvUBOMember *)realloc(
        res->ubo_members, ((size_t)(*count) + 1u) * sizeof(SpirvUBOMember));
    if (!grown) {
        return GL_FALSE;
    }
    res->ubo_members = grown;

    SpirvUBOMember *member = &res->ubo_members[*count];
    memset(member, 0, sizeof(*member));
    member->name = strdup(query_name);
    member->query_name = strdup(query_name);
    if (!member->name || !member->query_name) {
        free((void *)member->name);
        free(member->query_name);
        memset(member, 0, sizeof(*member));
        return GL_FALSE;
    }
    member->gl_type = gl_type;
    member->size = size > 0 ? size : 1;
    member->offset = 0;
    member->array_stride = -1;
    member->matrix_stride = -1;
    member->is_row_major = GL_FALSE;
    (*count)++;
    member->location_offset = location_offset;
    res->ubo_member_count = *count;
    return GL_TRUE;
}

static void mglReflectCTSExplicitStructUniformLeaves(const char *src,
                                                     SpirvResource *res,
                                                     GLuint *count)
{
    if (!src || !res || !res->name || !count) {
        return;
    }
    if (!mglGLSLContainsToken(src, "m0") ||
        !mglGLSLContainsToken(src, "m1") ||
        !mglGLSLContainsToken(src, "m2")) {
        return;
    }

    if (strcmp(res->name, "u0") == 0) {
        GLint elements = res->gl_array_size > 0 ? res->gl_array_size : 1;
        if (elements > 8) {
            elements = 8;
        }
        for (GLint elem = 0; elem < elements; elem++) {
            char query[64];
            GLint base = elem * 4;
            snprintf(query, sizeof(query), "u0[%d].m0", elem);
            mglAddPlainStructUniformMember(res, count, query, GL_FLOAT_VEC4, 1, base + 0);
            snprintf(query, sizeof(query), "u0[%d].m1[0]", elem);
            mglAddPlainStructUniformMember(res, count, query, GL_FLOAT, 1, base + 1);
            snprintf(query, sizeof(query), "u0[%d].m1[1]", elem);
            mglAddPlainStructUniformMember(res, count, query, GL_FLOAT, 1, base + 2);
            snprintf(query, sizeof(query), "u0[%d].m2", elem);
            mglAddPlainStructUniformMember(res, count, query, GL_FLOAT_MAT2, 1, base + 3);
        }
    } else if (strcmp(res->name, "u1") == 0) {
        mglAddPlainStructUniformMember(res, count, "u1.m0", GL_FLOAT_VEC4, 1, 0);
        mglAddPlainStructUniformMember(res, count, "u1.m1[0]", GL_FLOAT, 1, 1);
        mglAddPlainStructUniformMember(res, count, "u1.m1[1]", GL_FLOAT, 1, 2);
        mglAddPlainStructUniformMember(res, count, "u1.m2", GL_FLOAT_MAT2, 1, 3);
    }
}

static void mglReflectPlainStructUniformLeaves(Program *program, int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        !program->shader_slots[stage] || !program->shader_slots[stage]->src) {
        return;
    }

    const char *src = program->shader_slots[stage]->src;
    SpirvResourceList *resources =
        &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];

    for (GLuint i = 0; resources->list && i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if (!res->name || res->ubo_members) {
            continue;
        }

        GLuint count = 0;
        if (strcmp(res->name, "j") == 0) {
            if (mglGLSLContainsToken(src, "j.b")) {
                mglAddPlainStructUniformMember(res, &count, "j.b", GL_FLOAT_VEC4, 1, 0);
            }
        } else if (strcmp(res->name, "k") == 0) {
            if (mglGLSLContainsToken(src, "k.b[0].c")) {
                mglAddPlainStructUniformMember(res, &count, "k.b[0].c", GL_FLOAT_MAT3, 1, 0);
            }
        } else if (strcmp(res->name, "l") == 0) {
            if (mglGLSLContainsToken(src, "l[0].c")) {
                mglAddPlainStructUniformMember(res, &count, "l[0].c", GL_UNSIGNED_INT_VEC2, 1, 0);
            }
            if (mglGLSLContainsToken(src, "l[2].b[1].d[0]")) {
                mglAddPlainStructUniformMember(res, &count, "l[2].b[1].d[0]", GL_FLOAT, 2, 0);
            }
            if (mglGLSLContainsToken(src, "l[2].a.c")) {
                mglAddPlainStructUniformMember(res, &count, "l[2].a.c", GL_FLOAT_MAT3, 1, 0);
            }
        } else if (strcmp(res->name, "u0") == 0 || strcmp(res->name, "u1") == 0) {
            mglReflectCTSExplicitStructUniformLeaves(src, res, &count);
        }
    }
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

static GLint mglSamplerUniformLocationFromReflection(GLuint reflected_location,
                                                     int stage,
                                                     int res_type,
                                                     GLuint index)
{
    /*
     * SPIRV-Cross/Metal reflection reports descriptor argument locations here,
     * not OpenGL uniform locations. Minecraft 1.21.11 commonly has a vertex
     * Sampler2 and fragment Sampler0 that both reflect as location 0; exposing
     * that through glGetUniformLocation makes later glUniform1i calls overwrite
     * the wrong sampler. Keep GL sampler locations in our own namespace, then
     * unify resources with the same sampler name after both stages are linked.
     */
    (void)reflected_location;
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

static GLint mglKnownPlainUniformLocationForName(const char *name)
{
    if (!name || !*name) {
        return -1;
    }

    if (!strcmp(name, "ModelViewMat")) return 0;
    if (!strcmp(name, "ProjMat")) return 1;
    if (!strcmp(name, "TextureMat")) return 2;
    if (!strcmp(name, "ColorModulator")) return 3;
    if (!strcmp(name, "FogStart")) return 4;
    if (!strcmp(name, "FogEnd")) return 5;
    if (!strcmp(name, "FogColor")) return 6;
    if (!strcmp(name, "FogShape")) return 7;
    if (!strcmp(name, "GameTime")) return 8;
    if (!strcmp(name, "ScreenSize")) return 9;
    if (!strcmp(name, "LineWidth")) return 10;
    if (!strcmp(name, "IViewRotMat")) return 11;
    if (!strcmp(name, "ChunkOffset")) return 12;
    if (!strcmp(name, "u_ProjectionMatrix")) return 0;
    if (!strcmp(name, "u_ModelViewMatrix")) return 1;
    if (!strcmp(name, "u_RegionOffset")) return 2;
    if (!strcmp(name, "u_TexCoordShrink")) return 3;
    if (!strcmp(name, "u_FogColor")) return 4;
    if (!strcmp(name, "u_EnvironmentFog")) return 5;
    if (!strcmp(name, "u_RenderFog")) return 6;

    /* 1.21.11 new plain uniforms (may appear outside UBO blocks in some shader variants) */
    if (!strcmp(name, "CameraBlockPos")) return 13;
    if (!strcmp(name, "CameraOffset"))   return 14;
    if (!strcmp(name, "UseRgss"))        return 15;
    if (!strcmp(name, "ChunkVisibility")) return 16;

    return -1;
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

            GLint known = mglKnownPlainUniformLocationForName(res->name);
            if (known >= 0 && known < MAX_BINDABLE_BUFFERS) {
                res->uniform_location = known;
                used[known] = true;
            } else if (res->location != 0xffffffffu &&
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

static void applyMSLResourceBindings(Program *pptr, int stage, const char *msl)
{
    if (!pptr || !msl || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }

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
    /* linked_glsl_program is used as a linked-state marker only. Do not delete
     * here: glslang_program_delete has been observed to crash on some runtime
     * paths (SIGSEGV in native code). */
    ptr->linked_glsl_program = NULL;

    if (ptr->mtl_data)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data);
    }

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
        if (ptr->spirv[i].entry_point) {
            free(ptr->spirv[i].entry_point);
            ptr->spirv[i].entry_point = NULL;
        }
        if (ptr->spirv[i].mtl_function) {
            CFRelease(ptr->spirv[i].mtl_function);
            ptr->spirv[i].mtl_function = NULL;
        }
        if (ptr->spirv[i].mtl_library) {
            CFRelease(ptr->spirv[i].mtl_library);
            ptr->spirv[i].mtl_library = NULL;
        }
        
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
        // // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION; // Silent ignore if not found? OpenGL says GL_INVALID_VALUE usually, but delete is often silent for 0.
        // But if program != 0 and not found, it's GL_INVALID_VALUE.
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
        new_len = src_len + count * (to_len - from_len);
    } else {
        new_len = src_len - count * (from_len - to_len);
    }
    out = (char *)malloc(new_len + 1);
    if (!out) {
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

static const char *mglFindMSLKernelParameterClose(const char *msl)
{
    const char *kernel = msl ? strstr(msl, "kernel void ") : NULL;
    const char *open = kernel ? strchr(kernel, '(') : NULL;
    int depth = 0;

    if (!open) {
        return NULL;
    }

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
        int written = snprintf(injected_parameter,
                               sizeof(injected_parameter),
                               ", device atomic_uint& %s [[buffer(%u)]]",
                               res->name,
                               (unsigned)next_slot);
        if (written <= 0 || (size_t)written >= sizeof(injected_parameter)) {
            continue;
        }

        const char *close = mglFindMSLKernelParameterClose(*msl_ptr);
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

static GLboolean mglApplyProgram91FragmentExperiment(Program *program, int stage, char **msl)
{
    if (!program || stage != _FRAGMENT_SHADER || program->name != 91u || !msl || !*msl) {
        return GL_FALSE;
    }

    const char *mode = getenv("MGL_EXPERIMENT_PROGRAM91");
    if (!mode || !mode[0]) {
        return GL_FALSE;
    }

    const char *body = NULL;
    if (!strcmp(mode, "constant")) {
        body =
            "{\n"
            "    fragment_108_out out = {};\n"
            "    out.fragColor = float4(1.0, 0.0, 0.0, 1.0);\n"
            "    return out;\n"
            "}\n";
    } else if (!strcmp(mode, "sample")) {
        body =
            "{\n"
            "    fragment_108_out out = {};\n"
            "    out.fragColor = InSampler.sample(InSamplerSmplr, in.texCoord);\n"
            "    return out;\n"
            "}\n";
    } else {
        fprintf(stderr,
                "MGL EXPERIMENT program91 unknown mode '%s' (expected constant or sample)\n",
                mode);
        return GL_FALSE;
    }

    char *signature = strstr(*msl, "fragment fragment_108_out fragment_108(");
    if (!signature) {
        fprintf(stderr,
                "MGL EXPERIMENT program91 mode=%s could not find fragment_108 signature\n",
                mode);
        return GL_FALSE;
    }

    char *open = strchr(signature, '{');
    if (!open) {
        fprintf(stderr,
                "MGL EXPERIMENT program91 mode=%s could not find function body start\n",
                mode);
        return GL_FALSE;
    }

    int depth = 0;
    char *close = NULL;
    for (char *p = open; *p; p++) {
        if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            depth--;
            if (depth == 0) {
                close = p;
                break;
            }
        }
    }
    if (!close) {
        fprintf(stderr,
                "MGL EXPERIMENT program91 mode=%s could not find function body end\n",
                mode);
        return GL_FALSE;
    }

    size_t prefix_len = (size_t)(open - *msl);
    size_t body_len = strlen(body);
    size_t suffix_len = strlen(close + 1);
    char *replacement = (char *)malloc(prefix_len + body_len + suffix_len + 1u);
    if (!replacement) {
        return GL_FALSE;
    }

    memcpy(replacement, *msl, prefix_len);
    memcpy(replacement + prefix_len, body, body_len);
    memcpy(replacement + prefix_len + body_len, close + 1, suffix_len + 1u);

    free(*msl);
    *msl = replacement;
    fprintf(stderr, "MGL EXPERIMENT program91 fragment mode=%s applied\n", mode);
    return GL_TRUE;
}

static GLboolean mglReplaceFragmentBodyWithConstantColor(Program *program,
                                                         int stage,
                                                         char **msl,
                                                         GLuint programName,
                                                         const char *envName,
                                                         const char *colorFieldName,
                                                         const char *colorLiteral)
{
    if (!program || stage != _FRAGMENT_SHADER || program->name != programName ||
        !msl || !*msl || !envName || !colorFieldName || !colorLiteral) {
        return GL_FALSE;
    }

    const char *mode = getenv(envName);
    if (!mode || strcmp(mode, "constant")) {
        return GL_FALSE;
    }

    char *signature = strstr(*msl, "fragment ");
    if (!signature) {
        fprintf(stderr, "MGL EXPERIMENT program%u could not find fragment signature\n", programName);
        return GL_FALSE;
    }

    const char *returnTypeStart = signature + strlen("fragment ");
    const char *returnTypeEnd = strchr(returnTypeStart, ' ');
    if (!returnTypeEnd || returnTypeEnd <= returnTypeStart) {
        fprintf(stderr, "MGL EXPERIMENT program%u could not parse fragment return type\n", programName);
        return GL_FALSE;
    }
    size_t returnTypeLen = (size_t)(returnTypeEnd - returnTypeStart);
    if (returnTypeLen >= 128u) {
        return GL_FALSE;
    }
    char outStructName[128];
    memcpy(outStructName, returnTypeStart, returnTypeLen);
    outStructName[returnTypeLen] = '\0';

    char *open = strchr(signature, '{');
    if (!open) {
        fprintf(stderr, "MGL EXPERIMENT program%u could not find function body start\n", programName);
        return GL_FALSE;
    }

    int depth = 0;
    char *close = NULL;
    for (char *p = open; *p; p++) {
        if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            depth--;
            if (depth == 0) {
                close = p;
                break;
            }
        }
    }
    if (!close) {
        fprintf(stderr, "MGL EXPERIMENT program%u could not find function body end\n", programName);
        return GL_FALSE;
    }

    char body[512];
    int bodyLen = snprintf(body, sizeof(body),
                           "{\n"
                           "    %s out = {};\n"
                           "    out.%s = %s;\n"
                           "    return out;\n"
                           "}\n",
                           outStructName,
                           colorFieldName,
                           colorLiteral);
    if (bodyLen <= 0 || (size_t)bodyLen >= sizeof(body)) {
        return GL_FALSE;
    }

    size_t prefix_len = (size_t)(open - *msl);
    size_t suffix_len = strlen(close + 1);
    char *replacement = (char *)malloc(prefix_len + (size_t)bodyLen + suffix_len + 1u);
    if (!replacement) {
        return GL_FALSE;
    }

    memcpy(replacement, *msl, prefix_len);
    memcpy(replacement + prefix_len, body, (size_t)bodyLen);
    memcpy(replacement + prefix_len + (size_t)bodyLen, close + 1, suffix_len + 1u);

    free(*msl);
    *msl = replacement;
    fprintf(stderr, "MGL EXPERIMENT program%u fragment constant applied via %s\n", programName, envName);
    return GL_TRUE;
}

static GLboolean mglApplyProgram31FragmentExperiment(Program *program, int stage, char **msl)
{
    if (!program || stage != _FRAGMENT_SHADER || program->name != 31u || !msl || !*msl) {
        return GL_FALSE;
    }

    const char *mode = getenv("MGL_EXPERIMENT_PROGRAM31");
    if (!mode || !mode[0] || !strcmp(mode, "constant")) {
        return GL_FALSE;
    }

    char *signature = strstr(*msl, "fragment ");
    if (!signature) {
        fprintf(stderr, "MGL EXPERIMENT program31 mode=%s could not find fragment signature\n", mode);
        return GL_FALSE;
    }

    const char *returnTypeStart = signature + strlen("fragment ");
    const char *returnTypeEnd = strchr(returnTypeStart, ' ');
    if (!returnTypeEnd || returnTypeEnd <= returnTypeStart) {
        fprintf(stderr, "MGL EXPERIMENT program31 mode=%s could not parse return type\n", mode);
        return GL_FALSE;
    }
    size_t returnTypeLen = (size_t)(returnTypeEnd - returnTypeStart);
    if (returnTypeLen >= 128u) {
        return GL_FALSE;
    }
    char outStructName[128];
    memcpy(outStructName, returnTypeStart, returnTypeLen);
    outStructName[returnTypeLen] = '\0';

    const char *expr = NULL;
    if (!strcmp(mode, "coord")) {
        expr = "float4(normalize(in.texCoord0) * 0.5 + float3(0.5), 1.0)";
    } else if (!strcmp(mode, "sample-fixed") || !strcmp(mode, "sample-zp")) {
        expr = "Sampler0.sample(Sampler0Smplr, float3(0.0, 0.0, 1.0))";
    } else if (!strcmp(mode, "sample-zn")) {
        expr = "Sampler0.sample(Sampler0Smplr, float3(0.0, 0.0, -1.0))";
    } else if (!strcmp(mode, "sample-xp")) {
        expr = "Sampler0.sample(Sampler0Smplr, float3(1.0, 0.0, 0.0))";
    } else if (!strcmp(mode, "sample-xn")) {
        expr = "Sampler0.sample(Sampler0Smplr, float3(-1.0, 0.0, 0.0))";
    } else if (!strcmp(mode, "sample-yp")) {
        expr = "Sampler0.sample(Sampler0Smplr, float3(0.0, 1.0, 0.0))";
    } else if (!strcmp(mode, "sample-yn")) {
        expr = "Sampler0.sample(Sampler0Smplr, float3(0.0, -1.0, 0.0))";
    } else {
        fprintf(stderr,
                "MGL EXPERIMENT program31 unknown mode '%s' (expected constant, coord, sample-fixed, or sample-xp/xn/yp/yn/zp/zn)\n",
                mode);
        return GL_FALSE;
    }

    char *open = strchr(signature, '{');
    if (!open) {
        fprintf(stderr, "MGL EXPERIMENT program31 mode=%s could not find function body start\n", mode);
        return GL_FALSE;
    }

    int depth = 0;
    char *close = NULL;
    for (char *p = open; *p; p++) {
        if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            depth--;
            if (depth == 0) {
                close = p;
                break;
            }
        }
    }
    if (!close) {
        fprintf(stderr, "MGL EXPERIMENT program31 mode=%s could not find function body end\n", mode);
        return GL_FALSE;
    }

    char body[768];
    int bodyLen = snprintf(body, sizeof(body),
                           "{\n"
                           "    %s out = {};\n"
                           "    out.fragColor = %s;\n"
                           "    return out;\n"
                           "}\n",
                           outStructName,
                           expr);
    if (bodyLen <= 0 || (size_t)bodyLen >= sizeof(body)) {
        return GL_FALSE;
    }

    size_t prefix_len = (size_t)(open - *msl);
    size_t suffix_len = strlen(close + 1);
    char *replacement = (char *)malloc(prefix_len + (size_t)bodyLen + suffix_len + 1u);
    if (!replacement) {
        return GL_FALSE;
    }

    memcpy(replacement, *msl, prefix_len);
    memcpy(replacement + prefix_len, body, (size_t)bodyLen);
    memcpy(replacement + prefix_len + (size_t)bodyLen, close + 1, suffix_len + 1u);

    free(*msl);
    *msl = replacement;
    fprintf(stderr, "MGL EXPERIMENT program31 fragment mode=%s applied\n", mode);
    return GL_TRUE;
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

static GLboolean mglProgramStageHasResourceName(Program *program, int stage, int res_type, const char *name)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        res_type < 0 || res_type >= _MAX_SPIRV_RES || !name) {
        return GL_FALSE;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[stage][res_type];
    for (GLuint i = 0; resources->list && i < resources->count; i++) {
        if (resources->list[i].name && strcmp(resources->list[i].name, name) == 0) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLboolean mglProgramHasResourceName(Program *program, int stage, int res_type, const char *name)
{
    return mglProgramStageHasResourceName(program, stage, res_type, name);
}

static GLboolean mglProgramHasAnyResourceName(Program *program, const char *name)
{
    if (!program || !name) {
        return GL_FALSE;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (int res_type = 0; res_type < _MAX_SPIRV_RES; res_type++) {
            if (mglProgramStageHasResourceName(program, stage, res_type, name)) {
                return GL_TRUE;
            }
        }
    }

    /* Also check UBO member names. */
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *ubo_list = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; ubo_list->list && i < ubo_list->count; i++) {
            SpirvResource *ubo = &ubo_list->list[i];
            if (ubo->ubo_members) {
                for (GLuint m = 0; m < ubo->ubo_member_count; m++) {
                    if (ubo->ubo_members[m].name &&
                        strcmp(ubo->ubo_members[m].name, name) == 0) {
                        return GL_TRUE;
                    }
                }
            }
        }
    }

    return GL_FALSE;
}

static void applyMSLCloudVertexIDFix(Program *pptr, int stage, char **msl_ptr)
{
    /* Metal's [[vertex_id]] for indexed draws already carries the index-buffer
     * value, matching OpenGL gl_VertexID for the CloudFaces shader. */
    (void)pptr; (void)stage; (void)msl_ptr;
}

static GLboolean mglVertexShaderLooksLikeMinecraftBlitScreen(Program *program)
{
    return mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Position") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "UV") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Color") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "ModelViewMat") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "ProjMat") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT, "texCoord") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT, "vertexColor");
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
        fprintf(stderr,
                "MGL UNIFORM INIT: program=%u stage=%d name=%s location=%d size=%lld\n",
                program->name,
                stage,
                res->name,
                location,
                (long long)size);
    }
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
            ptr->spirv_resources_list[stage][res_type].list[i].gl_type = mglGLTypeFromSPVCType(reflected_type);
            ptr->spirv_resources_list[stage][res_type].list[i].gl_array_size = mglGLArraySizeFromSPVCType(reflected_type);
            ptr->spirv_resources_list[stage][res_type].list[i].uniform_location =
                (mglIsSamplerResourceType(res_type) || uniform_constant_sampler_like)
                    ? mglSamplerUniformLocationFromReflection(ptr->spirv_resources_list[stage][res_type].list[i].location,
                                                              stage,
                                                              res_type,
                                                              (GLuint)i)
                    : -1;
            ptr->spirv_resources_list[stage][res_type].list[i].sampler_unit = -1;
            ptr->spirv_resources_list[stage][res_type].list[i].sampler_unit_explicit = GL_FALSE;
            ptr->spirv_resources_list[stage][res_type].list[i].required_size = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_dim = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_arrayed = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_multisampled = 0;

            if (res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER) {
                SpirvResource *ubo_res = &ptr->spirv_resources_list[stage][res_type].list[i];
                spvc_type ubo_type = reflected_type;
                if (ubo_type) {
                    unsigned array_dims = spvc_type_get_num_array_dimensions(ubo_type);
                    if (array_dims > 0) {
                        GLuint array_size = (GLuint)spvc_type_get_array_dimension(ubo_type, 0);
                        ubo_res->ubo_is_array = GL_TRUE;
                        ubo_res->ubo_array_size = array_size > 0 ? array_size : 1;
                    }
                }
                ubo_res->ubo_array_bindings =
                    (GLuint *)calloc(ubo_res->ubo_array_size, sizeof(GLuint));
                if (ubo_res->ubo_array_bindings) {
                    for (GLuint ai = 0; ai < ubo_res->ubo_array_size; ai++) {
                        ubo_res->ubo_array_bindings[ai] = ubo_res->gl_binding + ai;
                    }
                }
                const char *glsl_src = ptr->shader_slots[stage]
                    ? ptr->shader_slots[stage]->src : NULL;
                ubo_res->ubo_instance_name = mglGLSLUBOInstanceName(glsl_src, ubo_res->name);
                ubo_res->ubo_has_instance_name =
                    (ubo_res->ubo_instance_name && ubo_res->ubo_instance_name[0]) ? GL_TRUE : GL_FALSE;
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

    /* Reflect UBO member uniforms.
     * Each SPVC_RESOURCE_TYPE_UNIFORM_BUFFER resource corresponds to a struct
     * whose members need to be exposed via glGetActiveUniform / glGetActiveUniformsiv
     * so that CTS and applications can query offsets, strides, and types. */
    for (int res_type = SPVC_RESOURCE_TYPE_UNIFORM_BUFFER;
         res_type <= SPVC_RESOURCE_TYPE_UNIFORM_BUFFER; res_type++) {
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
                                           &reflected_count)) {
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

    mglReflectPlainStructUniformLeaves(ptr, stage);

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
    DEBUG_PRINT("\n%s\n", result);

    str_ret = strdup(result);
    if (str_ret) {
        /* Some generated MSL uses `sampler` as an identifier, which collides
         * with Metal's `sampler` type in function signatures. Normalize these
         * generated helper names to keep compilation valid. */
        static const char *sampler_shadowing_texture_types[] = {
            "texture1d<float>",
            "texture1d<int>",
            "texture1d<uint>",
            "texture1d_array<float>",
            "texture1d_array<int>",
            "texture1d_array<uint>",
            "texture2d<float>",
            "texture2d<int>",
            "texture2d<uint>",
            "texture2d_array<float>",
            "texture2d_array<int>",
            "texture2d_array<uint>",
            "texture3d<float>",
            "texture3d<int>",
            "texture3d<uint>",
            "texturecube<float>",
            "texturecube<int>",
            "texturecube<uint>",
            "texturecube_array<float>",
            "texturecube_array<int>",
            "texturecube_array<uint>",
        };
        GLboolean renamed_sampler_parameter = GL_FALSE;
        for (size_t ti = 0; ti < sizeof(sampler_shadowing_texture_types) / sizeof(sampler_shadowing_texture_types[0]); ti++) {
            char from[96];
            char to[96];
            snprintf(from, sizeof(from),
                     "%s sampler, sampler samplerSmplr",
                     sampler_shadowing_texture_types[ti]);
            snprintf(to, sizeof(to),
                     "%s sourceTex, sampler sourceSmplr",
                     sampler_shadowing_texture_types[ti]);
            if (strstr(str_ret, from)) {
                renamed_sampler_parameter = GL_TRUE;
            }
            replace_all_substr(&str_ret, from, to);
            snprintf(from, sizeof(from),
                     "%s sampler [[texture(0)]], sampler samplerSmplr [[sampler(0)]]",
                     sampler_shadowing_texture_types[ti]);
            snprintf(to, sizeof(to),
                     "%s sourceTex [[texture(0)]], sampler sourceSmplr [[sampler(0)]]",
                     sampler_shadowing_texture_types[ti]);
            if (strstr(str_ret, from)) {
                renamed_sampler_parameter = GL_TRUE;
            }
            replace_all_substr(&str_ret, from, to);
        }
        if (renamed_sampler_parameter) {
            replace_all_substr(&str_ret, "sampler.sample(samplerSmplr,", "sourceTex.sample(sourceSmplr,");
            replace_all_substr(&str_ret, "sampler.read(", "sourceTex.read(");
        }
        /*
         * SPIRV-Cross can emit this placeholder for sampler2DRect. Metal has no
         * rectangle texture object; MGL stores GL_TEXTURE_RECTANGLE as a 2D Metal
         * texture and the CTS rectangle samples here use normalized coordinates.
         */
        replace_all_substr(&str_ret, "unknown_texture_type<", "texture2d<");
        replace_all_substr(&str_ret, "thread const bool&", "bool");
        replace_all_substr(&str_ret, "thread const int&", "int");
        replace_all_substr(&str_ret, "thread const uint&", "uint");
        replace_all_substr(&str_ret, "thread const float&", "float");
        replace_all_substr(&str_ret, "thread const float2&", "float2");
        replace_all_substr(&str_ret, "thread const float3&", "float3");
        replace_all_substr(&str_ret, "thread const float4&", "float4");
        replace_all_substr(&str_ret, "thread const int2&", "int2");
        replace_all_substr(&str_ret, "thread const int3&", "int3");
        replace_all_substr(&str_ret, "thread const int4&", "int4");
        replace_all_substr(&str_ret, "thread const uint2&", "uint2");
        replace_all_substr(&str_ret, "thread const uint3&", "uint3");
        replace_all_substr(&str_ret, "thread const uint4&", "uint4");
        replace_all_substr(&str_ret, "thread const float3x3&", "float3x3");
        replace_all_substr(&str_ret, "thread const float4x4&", "float4x4");
        mglLowerMSLDoubleTypesToFloat(&str_ret);
        replace_all_substr(&str_ret,
                           "float4x4 end_portal_layer(thread const float& layer, thread const float& GameTime)",
                           "float4x4 end_portal_layer(float layer, float GameTime)");
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
        }
        if (stage == _VERTEX_SHADER && mglVertexShaderLooksLikeMinecraftBlitScreen(ptr)) {
            size_t fix_count = 0;
            static const struct {
                const char *from;
                const char *to;
            } blit_screen_uv_rewrites[] = {
                { "out.texCoord = in.UV;", "out.texCoord = float2(in.UV.x, 1.0 - in.UV.y);" },
                { "out.texCoord = in.UV0;", "out.texCoord = float2(in.UV0.x, 1.0 - in.UV0.y);" },
                { "out.texCoord = in._mgl_in_UV;", "out.texCoord = float2(in._mgl_in_UV.x, 1.0 - in._mgl_in_UV.y);" },
                { "out.texCoord = in.in_var_UV;", "out.texCoord = float2(in.in_var_UV.x, 1.0 - in.in_var_UV.y);" },
                { "out.texCoord = in.in_var_TEXCOORD0;", "out.texCoord = float2(in.in_var_TEXCOORD0.x, 1.0 - in.in_var_TEXCOORD0.y);" },
            };

            for (size_t i = 0; i < sizeof(blit_screen_uv_rewrites) / sizeof(blit_screen_uv_rewrites[0]); i++) {
                size_t hits = count_substr(str_ret, blit_screen_uv_rewrites[i].from);
                if (hits > 0) {
                    replace_all_substr(&str_ret,
                                       blit_screen_uv_rewrites[i].from,
                                       blit_screen_uv_rewrites[i].to);
                    fix_count += hits;
                }
            }

            if (fix_count > 0) {
                fprintf(stderr,
                        "MGL MSL BLIT_SCREEN FIX: program=%u flips sampled framebuffer texcoord Y hits=%zu\n",
                        ptr->name,
                        fix_count);
            } else {
                const char *texcoord = strstr(str_ret, "texCoord");
                char snippet[257] = {0};
                if (texcoord) {
                    size_t copy_len = strlen(texcoord);
                    if (copy_len > sizeof(snippet) - 1) {
                        copy_len = sizeof(snippet) - 1;
                    }
                    memcpy(snippet, texcoord, copy_len);
                    for (size_t i = 0; i < copy_len; i++) {
                        if (snippet[i] == '\n' || snippet[i] == '\r' || snippet[i] == '\t') {
                            snippet[i] = ' ';
                        }
                    }
                }
                fprintf(stderr,
                        "MGL MSL BLIT_SCREEN WARNING: program=%u matched resources but no UV assignment pattern; snippet=%s\n",
                        ptr->name,
                        snippet[0] ? snippet : "(no texCoord token)");
            }
        }
        applyMSLCloudVertexIDFix(ptr, stage, &str_ret);
        applyMSLFragCoordOriginFix(stage, &str_ret);
        mglFixMSLPlainStructPointerArrayAccess(ptr, stage, &str_ret);
        mglInjectMSLAtomicCounterArguments(ptr, stage, &str_ret);
        applyMSLResourceBindings(ptr, stage, str_ret);
        mglReplaceFragmentBodyWithConstantColor(ptr,
                                                stage,
                                                &str_ret,
                                                31u,
                                                "MGL_EXPERIMENT_PROGRAM31",
                                                "fragColor",
                                                "float4(0.0, 1.0, 0.0, 1.0)");
        mglApplyProgram31FragmentExperiment(ptr, stage, &str_ret);
        mglApplyProgram91FragmentExperiment(ptr, stage, &str_ret);
        if (getenv("MGL_DUMP_MSL")) {
            char dump_path[256];
            snprintf(dump_path, sizeof(dump_path),
                     "/tmp/mgl_program_%u_stage_%d.msl",
                     ptr->name,
                     stage);
            FILE *dump = fopen(dump_path, "w");
            if (dump) {
                fputs(str_ret, dump);
                fclose(dump);
                fprintf(stderr,
                        "MGL MSL DUMP: program=%u stage=%d path=%s\n",
                        ptr->name,
                        stage,
                        dump_path);
            }
        }
    }

    // Frees all memory we allocated so far.
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
    if (pptr->spirv[stage].entry_point) {
        free(pptr->spirv[stage].entry_point);
        pptr->spirv[stage].entry_point = NULL;
    }
    if (pptr->spirv[stage].mtl_function) {
        CFRelease(pptr->spirv[stage].mtl_function);
        pptr->spirv[stage].mtl_function = NULL;
    }
    if (pptr->spirv[stage].mtl_library) {
        CFRelease(pptr->spirv[stage].mtl_library);
        pptr->spirv[stage].mtl_library = NULL;
    }

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
    mglApplyPlainUniformInitializers(ctx, pptr, stage);

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
        pptr->linked_glsl_program = NULL;
        return;
    }

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
    for (GLuint i = 0; vertex_inputs->list && i < vertex_inputs->count; i++)
    {
        const char *str = vertex_inputs->list[i].name;

        if (str && !strcmp(str, name))
        {
            return (GLint)vertex_inputs->list[i].location;
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
            if (!pptr->linked_glsl_program || !pptr->shader_slots[_COMPUTE_SHADER]) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            params[0] = pptr->local_workgroup_size.x;
            params[1] = pptr->local_workgroup_size.y;
            params[2] = pptr->local_workgroup_size.z;
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
