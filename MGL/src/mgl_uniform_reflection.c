/*
 * mgl_uniform_reflection.c
 * MGL
 *
 * Uniform/Attribute Reflection Subsystem (Category A).
 *
 * Implementation of pure reflection helpers extracted from program.c.
 * See mgl_uniform_reflection.h for details.
 */

#include "mgl_uniform_reflection.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#define MGL_SYNTHETIC_SAMPLER_LOCATION_BASE 0x4000

/* SPIR-V opcodes */
#define MGL_SPV_OP_CONSTANT              43
#define MGL_SPV_OP_ACCESS_CHAIN          65
#define MGL_SPV_OP_IN_BOUNDS_ACCESS_CHAIN 66

/* ---- Group A.1: String/GLSL Parsing Helpers ---- */

const char *mglMemStr(const char *haystack, size_t haystack_len, const char *needle)
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

GLboolean mglRangeContainsToken(const char *begin, const char *end, const char *token)
{
    if (!begin || !end || end <= begin || !token) {
        return GL_FALSE;
    }

    return mglMemStr(begin, (size_t)(end - begin), token) ? GL_TRUE : GL_FALSE;
}

GLboolean mglGLSLDeclaresRowMajorUBOMember(const char *glsl_src,
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

char *mglRecoverMemberNameFromGLSLComposite(const char *glsl_src,
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

char *mglRecoverUBOMemberNameFromGLSL(const char *glsl_src,
                                             const char *block_name,
                                             unsigned member_index)
{
    return mglRecoverMemberNameFromGLSLComposite(glsl_src,
                                                 block_name,
                                                 member_index,
                                                 GL_TRUE);
}

char *mglRecoverStructMemberNameFromGLSL(const char *glsl_src,
                                                const char *struct_name,
                                                unsigned member_index)
{
    return mglRecoverMemberNameFromGLSLComposite(glsl_src,
                                                 struct_name,
                                                 member_index,
                                                 GL_FALSE);
}

char *mglGLSLTypeNameForMemberInComposite(const char *glsl_src,
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

char *mglGLSLCompositeTypeNameForPath(const char *glsl_src,
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

char *mglDupRange(const char *begin, const char *end)
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

char *mglGLSLUBOInstanceName(const char *glsl_src, const char *block_name)
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
GLuint mglGLSLUBOArraySize(const char *glsl_src, const char *block_name)
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

char *mglBuildUBOMemberQueryName(const SpirvResource *ubo, const SpirvUBOMember *member)
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

char *mglGLSLAccessPathForUBOMember(const char *glsl_src,
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

/* ---- Group A.2: SPIRV-Cross Type/Location/Size Query Helpers ---- */

GLuint mglGLTypeFromSPVCType(spvc_type type)
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

GLint mglGLArraySizeFromSPVCType(spvc_type type)
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

GLboolean mglGLSLNameLooksLikeType(const char *name)
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

char *mglLeafNameFromPath(const char *path)
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

GLuint mglGLBoolTypeForVectorSize(unsigned vec_size)
{
    static const GLuint v[] = {GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4};
    if (vec_size >= 1 && vec_size <= 4) {
        return v[vec_size - 1];
    }
    return GL_BOOL;
}

GLuint mglGLTypeFromSPVCTypeAndGLSL(spvc_type type,
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

char *mglJoinUBOMemberPath(const char *prefix, const char *member_name)
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

char *mglAppendArrayZeroSuffix(const char *name, unsigned num_dims)
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
GLuint mglMetalTypeAlignmentFromSPVC(spvc_compiler compiler, spvc_type type)
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
GLuint mglComputeMSLStructMemberOffset(spvc_compiler compiler,
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
GLuint mglComputeMSLStructSize(spvc_compiler compiler, spvc_type struct_type)
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

GLboolean mglSpvcStructMemberOffset(spvc_compiler compiler,
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

GLint mglSpvcStructMemberMatrixStride(spvc_compiler compiler,
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

GLint mglSpvcStructMemberArrayStride(spvc_compiler compiler,
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

GLint mglGLTypeLocationCount(GLuint gl_type, GLint array_size)
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

GLint mglSPVCTypeLocationCount(spvc_compiler compiler, spvc_type type)
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
GLint mglSPVCTypeLocationStep(spvc_compiler compiler, spvc_type type)
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

/* ---- Group A.3: UBO Member Reflection ---- */

GLboolean mglAppendReflectedUBOMember(SpirvResource *ubo,
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

/* ---- Group A.4: Active Path / SPIR-V Binary Analysis ---- */

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

/* Resolve an OpAccessChain result back to its root variable ID, collecting
 * the full constant index path.  Returns the root variable ID, or 0 if the
 * chain could not be fully resolved (e.g. non-constant index). */
GLuint mglSpvResolveAccessChainRoot(GLuint result_id,
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
void mglCollectActivePaths(const unsigned int *spirv, size_t word_count,
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
GLboolean mglActivePathHasPrefix(const MGLActivePathSet *set,
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
GLboolean mglActivePathExactMatch(const MGLActivePathSet *set,
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
GLboolean mglByteOffsetIsActive(GLuint offset,
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

GLboolean mglReflectUBOStructMember(Program *program,
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

GLboolean mglReflectUBOMemberLeaves(Program *program,
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

GLboolean mglGLSLContainsToken(const char *src, const char *token)
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

/* ---- Group A.5: Active Uniform/Block/Attrib Query Functions ---- */

GLboolean mglUniformBlockNameSeen(Program *program, int max_stage, GLuint max_index, const char *name, GLuint gl_binding)
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

GLuint mglProgramUniformBlockArraySize(const SpirvResource *block)
{
    return (block && block->ubo_array_size > 0) ? block->ubo_array_size : 1u;
}

GLint mglActiveUniformBlockCount(Program *program)
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
GLint mglActiveAtomicCounterBufferCount(Program *program)
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

GLint mglActiveUniformBlockMaxNameLength(Program *program)
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

SpirvResourceList *mglProgramActiveAttribList(Program *program)
{
    if (!program) {
        return NULL;
    }

    return &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
}

GLboolean mglProgramActiveAttribHasName(const SpirvResource *res)
{
    return (res && res->name && res->name[0] != '\0') ? GL_TRUE : GL_FALSE;
}

GLint mglProgramActiveAttribCount(Program *program)
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

SpirvResource *mglProgramActiveAttribAt(Program *program, GLuint index)
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

GLint mglProgramActiveAttribMaxNameLength(Program *program)
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

GLenum mglProgramActiveAttribType(const SpirvResource *res)
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

/* ---- Group A.6: Sampler Uniform Location Unification ---- */

GLint mglSyntheticSamplerUniformLocation(int stage, int res_type, GLuint index)
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
GLint mglFindExplicitUniformLocation(const char *glsl_src, const char *resource_name)
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

GLint mglSamplerUniformLocationFromReflection(GLuint reflected_location,
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

bool mglIsSamplerResourceType(int res_type)
{
    return res_type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE ||
           res_type == SPVC_RESOURCE_TYPE_SEPARATE_IMAGE ||
           res_type == SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS ||
           res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE;
}

bool mglUniformNameLooksSamplerLike(const char *name)
{
    if (!name || !*name) {
        return false;
    }

    return strstr(name, "Sampler") != NULL ||
           strcmp(name, "CloudFaces") == 0;
}

bool mglUniformConstantBaseTypeIsSamplerLike(spvc_basetype basetype)
{
    return basetype == SPVC_BASETYPE_IMAGE ||
           basetype == SPVC_BASETYPE_SAMPLED_IMAGE ||
           basetype == SPVC_BASETYPE_SAMPLER;
}

bool mglProgramResourceLooksSamplerLike(const SpirvResource *res, int res_type)
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

bool mglSamplerResourceNamesMatch(const char *a, const char *b)
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

void mglUnifySamplerUniformLocations(Program *program)
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

/* ---- Group A.7: Plain Uniform Location Assignment ---- */

SpirvResource *mglFindAssignedPlainUniformResource(Program *program, const char *name)
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

GLint mglFirstFreePlainUniformLocation(const bool used[MAX_BINDABLE_BUFFERS])
{
    for (GLint location = 0; location < MAX_BINDABLE_BUFFERS; location++) {
        if (!used[location]) {
            return location;
        }
    }

    return -1;
}

void mglAssignPlainUniformLocations(Program *program)
{
    if (!program) {
        return;
    }

    bool used[MAX_BINDABLE_BUFFERS] = {false};
    /* Track which uniform NAME claimed each location. GL uses a single uniform
     * location namespace across the whole linked program, but SPIR-V numbers
     * default-block uniforms per stage (each stage from 0). Honoring a
     * per-stage location directly (as this pass used to) let a fragment-stage
     * uniform collide with a different vertex-stage uniform on the same
     * location, so both indexed the same plain_uniform_buffers[loc] slot and
     * clobbered each other. Record the claimant name so we can tell a genuine
     * cross-stage SHARED uniform (same name — keep the shared location) apart
     * from an accidental collision (different name — defer to pass 2, which
     * assigns a free location). */
    const char *used_by[MAX_BINDABLE_BUFFERS] = {NULL};

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglProgramResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)) {
                continue;
            }

            if (res->location != 0xffffffffu &&
                       res->location < 1024u &&
                       res->location < MAX_BINDABLE_BUFFERS) {
                GLint candidate = (GLint)res->location;
                bool sameName = used_by[candidate] && res->name &&
                                strcmp(used_by[candidate], res->name) == 0;
                if (!used[candidate] || sameName) {
                    /* Free slot, or the same uniform shared across stages. */
                    res->uniform_location = candidate;
                    used[candidate] = true;
                    if (res->name) {
                        used_by[candidate] = res->name;
                    }
                } else {
                    /* Collision with a different uniform in another stage:
                     * leave uniform_location = -1 so pass 2 relocates it to a
                     * free slot instead of aliasing this one. */
                    res->uniform_location = -1;
                }
            } else if (res->location != 0xffffffffu && res->location < 1024u) {
                /* location >= MAX_BINDABLE_BUFFERS: cannot index used[]; keep
                 * prior behavior of honoring it verbatim. */
                res->uniform_location = (GLint)res->location;
            } else if (res->uniform_location >= 0 &&
                       res->uniform_location < MAX_BINDABLE_BUFFERS) {
                used[res->uniform_location] = true;
                if (res->name) {
                    used_by[res->uniform_location] = res->name;
                }
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

void mglAssignAggregateMemberLocations(Program *program)
{
    if (!program) {
        return;
    }
    /* Member name -> location, shared across stages. */
    char **names = NULL;
    GLint *locs = NULL;
    uint32_t name_count = 0;
    GLint next_loc = 0;
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (res->ubo_members && res->ubo_member_count > 0) {
                res->uniform_location = 0;
                for (GLuint m = 0; m < res->ubo_member_count; m++) {
                    SpirvUBOMember *mem = &res->ubo_members[m];
                    const char *name = mem->name ? mem->name : "";
                    GLint loc = -1;
                    for (uint32_t k = 0; k < name_count; k++) {
                        if (strcmp(names[k], name) == 0) {
                            loc = locs[k];
                            break;
                        }
                    }
                    if (loc < 0) {
                        loc = next_loc++;
                        char **nn = (char **)realloc(
                            names, (name_count + 1) * sizeof(char *));
                        GLint *nl = (GLint *)realloc(
                            locs, (name_count + 1) * sizeof(GLint));
                        if (!nn || !nl) {
                            free(nn ? nn : names);
                            free(nl ? nl : locs);
                            return;
                        }
                        names = nn;
                        locs = nl;
                        names[name_count] = strdup(name);
                        locs[name_count] = loc;
                        name_count++;
                    }
                    mem->location_offset = loc;
                }
            }
        }
    }
    for (uint32_t k = 0; k < name_count; k++) {
        free(names[k]);
    }
    free(names);
    free(locs);
}

GLint mglDefaultSamplerUnitForProgramResource(Program *program, const SpirvResource *res)
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

void mglApplyDefaultSamplerUnit(Program *program, int stage, int res_type, SpirvResource *res)
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

/* ---- Group A.8: Misc Reflection Utilities ---- */

void mglFreeSpirvResourceOwnedFields(SpirvResource *res)
{
    if (!res) {
        return;
    }

    free((void *)res->name);
    res->name = NULL;

    free(res->msl_name);
    res->msl_name = NULL;
    for (GLuint i = 0; res->msl_argument_names &&
                       i < res->msl_argument_count; i++) {
        free(res->msl_argument_names[i]);
    }
    free(res->msl_argument_names);
    res->msl_argument_names = NULL;
    res->msl_argument_count = 0u;
    free(res->msl_combined_sampler_name);
    res->msl_combined_sampler_name = NULL;
    res->msl_combined_sampler_binding = (GLuint)-1;
    res->msl_active = GL_FALSE;
    res->msl_has_combined_sampler = GL_FALSE;
    res->msl_binding_kind = MGL_MSL_BINDING_NONE;

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

GLint mglPlainUniformResourceLocationForProgram(const SpirvResource *res)
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

GLboolean mglParseScalarUniformInitializer(const char *src,
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

unsigned mglSPIRVFindAccessChainConstantIndices(
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
