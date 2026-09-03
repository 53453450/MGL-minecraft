/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_reflect.c
 * MGL
 *
 * Reflection export from the self-hosted GLSL frontend's MGLIRModule.
 * Produces the MGLShaderResource tables the GL query layer and the per-draw
 * binding paths consume, replacing the historical reflection step (which
 * delegated to the old SPIRV-Cross reflection step).
 */

#include "mgl_air_reflect.h"

#include <GL/glcorearb.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_uniform_reflection.h"
#include "glm_limits.h" /* MAX_ATTRIBS: attrib_names contract size */
#include "mgl_types_buffer.h" /* MAX_BINDABLE_BUFFERS */

GLuint mglAirGLTypeFromIR(const MGLIRType *t)
{
    if (!t) {
        return GL_INVALID_ENUM;
    }
    if (t->kind == MGLIR_TYPE_ARRAY) {
        return mglAirGLTypeFromIR(t->elem_type);
    }
    switch (t->kind) {
    case MGLIR_TYPE_SCALAR:
        switch (t->scalar) {
        case MGLIR_SCALAR_FLOAT: return GL_FLOAT;
        case MGLIR_SCALAR_INT:   return GL_INT;
        case MGLIR_SCALAR_UINT:  return GL_UNSIGNED_INT;
        case MGLIR_SCALAR_BOOL:  return GL_BOOL;
        default:                 return GL_FLOAT;
        }
    case MGLIR_TYPE_VECTOR: {
        GLuint base = (t->scalar == MGLIR_SCALAR_INT)   ? GL_INT_VEC2
                    : (t->scalar == MGLIR_SCALAR_UINT)  ? GL_UNSIGNED_INT_VEC2
                    : (t->scalar == MGLIR_SCALAR_BOOL)  ? GL_BOOL_VEC2
                                                        : GL_FLOAT_VEC2;
        return (GLuint)(base + t->cols - 2);
    }
    case MGLIR_TYPE_MATRIX: {
        /* Square matrices enumerate contiguously; matCxR non-square types
         * have their own enum block (GL 4.6 §22.4 / Table 22.2). */
        if (t->cols == t->rows) {
            return (GLuint)(GL_FLOAT_MAT2 + (t->cols - 2));
        }
        if (t->cols >= 2 && t->cols <= 4 && t->rows >= 2 && t->rows <= 4) {
            switch (t->cols * 10 + t->rows) {
            case 23: return GL_FLOAT_MAT2x3;
            case 24: return GL_FLOAT_MAT2x4;
            case 32: return GL_FLOAT_MAT3x2;
            case 34: return GL_FLOAT_MAT3x4;
            case 42: return GL_FLOAT_MAT4x2;
            case 43: return GL_FLOAT_MAT4x3;
            default: break;
            }
        }
        return (GLuint)(GL_FLOAT_MAT2 + (t->cols - 2));
    }
    case MGLIR_TYPE_ATOMIC_COUNTER:
        return GL_UNSIGNED_INT_ATOMIC_COUNTER;
    case MGLIR_TYPE_SAMPLER: {
        const GLboolean is_int = t->tex_storage == MGLIR_SCALAR_INT;
        const GLboolean is_uint = t->tex_storage == MGLIR_SCALAR_UINT;
        if (t->tex_depth && !is_int && !is_uint) {
            switch (t->tex_kind) {
            case MGLIR_TEX_1D:         return GL_SAMPLER_1D_SHADOW;
            case MGLIR_TEX_2D:         return GL_SAMPLER_2D_SHADOW;
            case MGLIR_TEX_CUBE:       return GL_SAMPLER_CUBE_SHADOW;
            case MGLIR_TEX_1D_ARRAY:   return GL_SAMPLER_1D_ARRAY_SHADOW;
            case MGLIR_TEX_2D_ARRAY:   return GL_SAMPLER_2D_ARRAY_SHADOW;
            case MGLIR_TEX_CUBE_ARRAY: return GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW;
            default:                   return GL_SAMPLER_2D_SHADOW;
            }
        }
#define MGL_SAMPLER_TYPE(_FLOAT, _INT, _UINT) \
        (is_int ? (_INT) : (is_uint ? (_UINT) : (_FLOAT)))
        switch (t->tex_kind) {
        case MGLIR_TEX_1D:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_1D, GL_INT_SAMPLER_1D,
                                    GL_UNSIGNED_INT_SAMPLER_1D);
        case MGLIR_TEX_2D:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_2D, GL_INT_SAMPLER_2D,
                                    GL_UNSIGNED_INT_SAMPLER_2D);
        case MGLIR_TEX_3D:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_3D, GL_INT_SAMPLER_3D,
                                    GL_UNSIGNED_INT_SAMPLER_3D);
        case MGLIR_TEX_CUBE:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_CUBE, GL_INT_SAMPLER_CUBE,
                                    GL_UNSIGNED_INT_SAMPLER_CUBE);
        case MGLIR_TEX_2D_RECT:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_2D_RECT,
                                    GL_INT_SAMPLER_2D_RECT,
                                    GL_UNSIGNED_INT_SAMPLER_2D_RECT);
        case MGLIR_TEX_1D_ARRAY:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_1D_ARRAY,
                                    GL_INT_SAMPLER_1D_ARRAY,
                                    GL_UNSIGNED_INT_SAMPLER_1D_ARRAY);
        case MGLIR_TEX_2D_ARRAY:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_2D_ARRAY,
                                    GL_INT_SAMPLER_2D_ARRAY,
                                    GL_UNSIGNED_INT_SAMPLER_2D_ARRAY);
        case MGLIR_TEX_CUBE_ARRAY:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_CUBE_MAP_ARRAY,
                                    GL_INT_SAMPLER_CUBE_MAP_ARRAY,
                                    GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY);
        case MGLIR_TEX_2D_MS:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_2D_MULTISAMPLE,
                                    GL_INT_SAMPLER_2D_MULTISAMPLE,
                                    GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE);
        case MGLIR_TEX_2D_MS_ARRAY:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_2D_MULTISAMPLE_ARRAY,
                                    GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY,
                                    GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY);
        case MGLIR_TEX_BUFFER:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_BUFFER, GL_INT_SAMPLER_BUFFER,
                                    GL_UNSIGNED_INT_SAMPLER_BUFFER);
        default:
            return MGL_SAMPLER_TYPE(GL_SAMPLER_2D, GL_INT_SAMPLER_2D,
                                    GL_UNSIGNED_INT_SAMPLER_2D);
        }
#undef MGL_SAMPLER_TYPE
    }
    case MGLIR_TYPE_IMAGE: {
        const GLboolean is_int = t->tex_storage == MGLIR_SCALAR_INT;
        const GLboolean is_uint = t->tex_storage == MGLIR_SCALAR_UINT;
#define MGL_IMAGE_TYPE(_FLOAT, _INT, _UINT) \
        (is_int ? (_INT) : (is_uint ? (_UINT) : (_FLOAT)))
        switch (t->tex_kind) {
        case MGLIR_TEX_1D:
            return MGL_IMAGE_TYPE(GL_IMAGE_1D, GL_INT_IMAGE_1D,
                                  GL_UNSIGNED_INT_IMAGE_1D);
        case MGLIR_TEX_2D:
            return MGL_IMAGE_TYPE(GL_IMAGE_2D, GL_INT_IMAGE_2D,
                                  GL_UNSIGNED_INT_IMAGE_2D);
        case MGLIR_TEX_3D:
            return MGL_IMAGE_TYPE(GL_IMAGE_3D, GL_INT_IMAGE_3D,
                                  GL_UNSIGNED_INT_IMAGE_3D);
        case MGLIR_TEX_CUBE:
            return MGL_IMAGE_TYPE(GL_IMAGE_CUBE, GL_INT_IMAGE_CUBE,
                                  GL_UNSIGNED_INT_IMAGE_CUBE);
        case MGLIR_TEX_2D_RECT:
            return MGL_IMAGE_TYPE(GL_IMAGE_2D_RECT, GL_INT_IMAGE_2D_RECT,
                                  GL_UNSIGNED_INT_IMAGE_2D_RECT);
        case MGLIR_TEX_1D_ARRAY:
            return MGL_IMAGE_TYPE(GL_IMAGE_1D_ARRAY, GL_INT_IMAGE_1D_ARRAY,
                                  GL_UNSIGNED_INT_IMAGE_1D_ARRAY);
        case MGLIR_TEX_2D_ARRAY:
            return MGL_IMAGE_TYPE(GL_IMAGE_2D_ARRAY, GL_INT_IMAGE_2D_ARRAY,
                                  GL_UNSIGNED_INT_IMAGE_2D_ARRAY);
        case MGLIR_TEX_CUBE_ARRAY:
            return MGL_IMAGE_TYPE(GL_IMAGE_CUBE_MAP_ARRAY,
                                  GL_INT_IMAGE_CUBE_MAP_ARRAY,
                                  GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY);
        case MGLIR_TEX_2D_MS:
            return MGL_IMAGE_TYPE(GL_IMAGE_2D_MULTISAMPLE,
                                  GL_INT_IMAGE_2D_MULTISAMPLE,
                                  GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE);
        case MGLIR_TEX_2D_MS_ARRAY:
            return MGL_IMAGE_TYPE(GL_IMAGE_2D_MULTISAMPLE_ARRAY,
                                  GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY,
                                  GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY);
        case MGLIR_TEX_BUFFER:
            return MGL_IMAGE_TYPE(GL_IMAGE_BUFFER, GL_INT_IMAGE_BUFFER,
                                  GL_UNSIGNED_INT_IMAGE_BUFFER);
        default:
            return MGL_IMAGE_TYPE(GL_IMAGE_2D, GL_INT_IMAGE_2D,
                                  GL_UNSIGNED_INT_IMAGE_2D);
        }
#undef MGL_IMAGE_TYPE
    }
    default:
        /* Struct-typed block members are not flattened into leaf uniforms
         * yet; report a benign valid type so the GL enumeration never
         * yields an invalid enum (CTS log printing dereferences the type
         * name table). */
        return GL_FLOAT;
    }
}

GLint mglAirGLArraySizeFromIR(const MGLIRType *t)
{
    if (!t) {
        return 0;
    }
    if (t->kind == MGLIR_TYPE_ARRAY) {
        return (GLint)t->array_size;   /* 0 = runtime array */
    }
    return 1;
}

static const MGLIRType *air_uniform_block_type(const MGLIRType *type)
{
    if (type && type->kind == MGLIR_TYPE_ARRAY) {
        type = type->elem_type;
    }
    return type && type->kind == MGLIR_TYPE_STRUCT && type->member_count > 0
        ? type : NULL;
}

static GLuint air_uniform_block_element_count(const MGLIRType *type)
{
    /* Length-1 instance arrays still need one Metal buffer slot and
     * `name[0].member` codegen against element slots. */
    return type && type->kind == MGLIR_TYPE_ARRAY && type->array_size > 0u
        ? type->array_size : 1u;
}

/* GL names interface blocks by the block name, not the instance: for
 * `uniform Colors { ... } uni_colors;` GetProgramResourceIndex(GL_UNIFORM_BLOCK)
 * must find "Colors".  The reflected symbol is named after the instance, so
 * rename the resource to the block name and keep the instance aside for
 * consumers that need to qualify member access. */
/* GL 4.6 §7.3.1.1 block-member flattening: nested struct fields become
 * dotted-path leaves (`nest.a`), arrays of structs enumerate every
 * element (`arr[0].x`, `arr[1].x`, ...), and arrays of scalars/vectors/
 * matrices stay one entry named `arr[0]` with size = element count.
 * Offsets are absolute within the block and come from the sema layout
 * pass, which caches member_offsets on every nested struct type. */
static int air_block_flatten(const MGLIRType *st, uint32_t base_off,
                             const char *prefix,
                             SpirvUBOMember **out, uint32_t *count,
                             uint32_t *cap)
{
    if (!st || st->kind != MGLIR_TYPE_STRUCT) {
        return -1;
    }
    for (uint32_t i = 0; i < st->member_count; i++) {
        const MGLIRType *mt = st->members[i];
        const char *mn = st->member_names[i];
        uint32_t off = base_off + (st->member_offsets ? st->member_offsets[i]
                                                      : 0u);
        char path[192];
        snprintf(path, sizeof(path), "%s%s%s", prefix, prefix[0] ? "." : "",
                 mn ? mn : "?");

        if (mt->kind == MGLIR_TYPE_STRUCT) {
            if (air_block_flatten(mt, off, path, out, count, cap) != 0) {
                return -1;
            }
            continue;
        }
        if (mt->kind == MGLIR_TYPE_ARRAY && mt->elem_type &&
            mt->elem_type->kind == MGLIR_TYPE_STRUCT) {
            uint32_t n = mt->array_size ? mt->array_size : 1u;
            uint32_t stride = mt->layout.array_stride > 0
                                  ? (uint32_t)mt->layout.array_stride
                                  : 0u;
            for (uint32_t el = 0; el < n; el++) {
                char epath[208];
                snprintf(epath, sizeof(epath), "%s[%u]", path, el);
                if (air_block_flatten(mt->elem_type, off + el * stride,
                                      epath, out, count, cap) != 0) {
                    return -1;
                }
            }
            continue;
        }

        /* Leaf: scalar/vector/matrix or an array of those.  Array leaves
         * are one entry named with a "[0]" postfix at every path level
         * (GL 4.6 §7.3.1.1). */
        if (*count == *cap) {
            uint32_t ncap = *cap ? *cap * 2 : 8;
            SpirvUBOMember *nl =
                (SpirvUBOMember *)realloc(*out, ncap * sizeof(SpirvUBOMember));
            if (!nl) {
                return -1;
            }
            *out = nl;
            *cap = ncap;
        }
        const MGLIRType *lt = (mt->kind == MGLIR_TYPE_ARRAY) ? mt->elem_type
                                                             : mt;
        SpirvUBOMember *u = &(*out)[(*count)++];
        memset(u, 0, sizeof(*u));
        if (mt->kind == MGLIR_TYPE_ARRAY) {
            char apath[208];
            snprintf(apath, sizeof(apath), "%s[0]", path);
            u->name = strdup(apath);
            u->query_name = strdup(apath);
            u->size = mglAirGLArraySizeFromIR(mt);
            u->array_stride = (GLint)mt->layout.array_stride;
        } else {
            u->name = strdup(path);
            u->query_name = strdup(path);
            u->size = 1;
            u->array_stride = 0;
        }
        u->gl_type = mglAirGLTypeFromIR(lt);
        u->offset = off;
        u->matrix_stride = (lt && lt->kind == MGLIR_TYPE_MATRIX)
                               ? (GLint)lt->layout.matrix_stride
                               : 0;
        u->is_row_major = (lt && lt->kind == MGLIR_TYPE_MATRIX && lt->row_major)
                              ? GL_TRUE
                              : GL_FALSE;
        u->location_offset = -1;
        u->top_level_array_size = u->size;
        u->top_level_array_stride = u->array_stride;
    }
    return 0;
}

static void apply_block_interface_name(MGLShaderResource *res,
                                       const MGLIRType *type,
                                       const char *instance_name)
{
    const MGLIRType *block_type = air_uniform_block_type(type);
    if (!block_type || !block_type->name || !block_type->name[0] ||
        !res->name || strcmp(res->name, block_type->name) == 0) {
        return;
    }
    char *renamed = strdup(block_type->name);
    if (!renamed) {
        return;
    }
    free((void *)res->name);
    res->name = renamed;
    if (instance_name && instance_name[0]) {
        res->ubo_instance_name = strdup(instance_name);
        res->ubo_has_instance_name = res->ubo_instance_name ? GL_TRUE : GL_FALSE;
    }
    /* GL 4.6 §7.3.1.1: when a block is declared with an instance name, the
     * uniforms inside are reported as "<blockName>.<member>"; anonymous
     * blocks report the bare member name.  air_block_flatten bakes the
     * "[0]" postfix into leaf paths, so only the block prefix is added
     * here. */
    if (res->ubo_members && res->ubo_member_count > 0) {
        for (uint32_t m = 0; m < res->ubo_member_count; m++) {
            SpirvUBOMember *u = &res->ubo_members[m];
            if (!u->name) {
                continue;
            }
            size_t bn = strlen(block_type->name);
            size_t mn = strlen(u->name);
            char *qn = (char *)malloc(bn + 1 + mn + 1);
            if (!qn) {
                continue;
            }
            memcpy(qn, block_type->name, bn);
            qn[bn] = '.';
            memcpy(qn + bn + 1, u->name, mn + 1);
            free((void *)u->query_name);
            u->query_name = qn;
        }
    }
}

static void push_resource(MGLShaderResourceList *list, const MGLIRSymbol *s,
                          const MGLIRType *type, GLuint location,
                          GLuint binding, int stage)
{
    MGLShaderResource r;
    memset(&r, 0, sizeof(r));
    r.name = strdup(s->name);
    r.location = location;
    r.gl_binding = binding;
    r.binding = binding;
    /* Resource reflection keeps the top-level array type for array size and
     * binding expansion, but sampler/image metadata must come from its
     * element type.  mglAirGLTypeFromIR(array) has no sampler case and would
     * otherwise silently report GL_FLOAT, causing integer texture arrays to
     * bind float fallback textures. */
    const MGLIRType *value_type = type;
    while (value_type && value_type->kind == MGLIR_TYPE_ARRAY)
        value_type = value_type->elem_type;
    r.gl_type = mglAirGLTypeFromIR(value_type ? value_type : type);
    r.gl_array_size = mglAirGLArraySizeFromIR(type);
    r.is_array = (type->kind == MGLIR_TYPE_ARRAY) ? GL_TRUE : GL_FALSE;
    r.is_per_patch = (s->qualifiers & MGL_AST_Q_PATCH) ? GL_TRUE : GL_FALSE;
    r.block_member = s->block_name ? GL_TRUE : GL_FALSE;
    r.stream = (s->stream >= 0) ? s->stream : 0;
    r.num_array_dims = (type->kind == MGLIR_TYPE_ARRAY) ? 1u : 0u;
    r.uniform_location = -1;
    r.sampler_unit = 0;
    r.sampler_unit_explicit = GL_FALSE;
    if (value_type && (value_type->kind == MGLIR_TYPE_SAMPLER ||
                       value_type->kind == MGLIR_TYPE_IMAGE)) {
        switch (value_type->tex_kind) {
        case MGLIR_TEX_1D:
        case MGLIR_TEX_1D_ARRAY:
            r.image_dim = MGL_IMAGE_DIM_1D;
            break;
        case MGLIR_TEX_3D:
            r.image_dim = MGL_IMAGE_DIM_3D;
            break;
        case MGLIR_TEX_CUBE:
        case MGLIR_TEX_CUBE_ARRAY:
            r.image_dim = MGL_IMAGE_DIM_CUBE;
            break;
        case MGLIR_TEX_2D_RECT:
            r.image_dim = MGL_IMAGE_DIM_RECT;
            break;
        case MGLIR_TEX_BUFFER:
            r.image_dim = MGL_IMAGE_DIM_BUFFER;
            break;
        case MGLIR_TEX_SUBPASS:
        case MGLIR_TEX_SUBPASS_MS:
            r.image_dim = MGL_IMAGE_DIM_SUBPASS_DATA;
            break;
        default:
            r.image_dim = MGL_IMAGE_DIM_2D;
            break;
        }
        r.image_arrayed =
            value_type->tex_kind == MGLIR_TEX_1D_ARRAY ||
            value_type->tex_kind == MGLIR_TEX_2D_ARRAY ||
            value_type->tex_kind == MGLIR_TEX_CUBE_ARRAY ||
            value_type->tex_kind == MGLIR_TEX_2D_MS_ARRAY;
        r.image_multisampled =
            value_type->tex_kind == MGLIR_TEX_2D_MS ||
            value_type->tex_kind == MGLIR_TEX_2D_MS_ARRAY ||
            value_type->tex_kind == MGLIR_TEX_SUBPASS_MS;
        r.texture_data_kind = value_type->tex_depth
            ? MGL_SHADER_TEXTURE_DATA_DEPTH
            : value_type->tex_storage == MGLIR_SCALAR_INT
                ? MGL_SHADER_TEXTURE_DATA_SINT
                : value_type->tex_storage == MGLIR_SCALAR_UINT
                    ? MGL_SHADER_TEXTURE_DATA_UINT
                    : MGL_SHADER_TEXTURE_DATA_FLOAT;
    }
    (void)stage;

    if (type->kind == MGLIR_TYPE_STRUCT && type->member_count > 0) {
        /* Semantic analysis computes interface-block layout before reflection.
         * Preserve the full block size so the renderer can isolate an indexed
         * range that is shorter than the shader's statically addressable data.
         * Without this, a short writable SSBO is bound directly and a later
         * CPU shadow upload can overwrite GPU-written bytes. */
        if (type->layout_valid) {
            r.required_size = type->layout.size;
        }
        r.ubo_member_count = 0;
        {
            SpirvUBOMember *leaves = NULL;
            uint32_t leaf_count = 0, leaf_cap = 0;
            if (air_block_flatten(type, 0u, "", &leaves, &leaf_count,
                                  &leaf_cap) == 0 &&
                leaf_count > 0) {
                r.ubo_members = leaves;
                r.ubo_member_count = leaf_count;
            } else {
                for (uint32_t m = 0; m < leaf_count; m++) {
                    free((void *)leaves[m].name);
                    free((void *)leaves[m].query_name);
                }
                free(leaves);
            }
        }
    }

    MGLShaderResource *nl = (MGLShaderResource *)realloc(
        list->list, (list->count + 1) * sizeof(MGLShaderResource));
    if (!nl) {
        return;
    }
    list->list = nl;
    list->list[list->count++] = r;
}

static void destroy_list(MGLShaderResourceList *list)
{
    for (GLuint i = 0; i < list->count; i++) {
        MGLShaderResource *r = &list->list[i];
        free((void *)r->name);
        free(r->ubo_instance_name);
        if (r->ubo_members) {
            for (GLuint m = 0; m < r->ubo_member_count; m++) {
                free((void *)r->ubo_members[m].name);
                free(r->ubo_members[m].query_name);
            }
            free(r->ubo_members);
        }
        free(r->ubo_array_bindings);
    }
    free(list->list);
    list->list = NULL;
    list->count = 0;
}

static uint32_t air_reflect_attrib_location(const char *name,
                                             const char *const *attrib_names)
{
    if (name && attrib_names) {
        for (int i = 0; i < MAX_ATTRIBS; i++) {
            if (attrib_names[i] && strcmp(attrib_names[i], name) == 0) {
                return (uint32_t)i;
            }
        }
    }
    if (name) {
        static const struct { const char *n; uint32_t l; } def[] = {
            {"Position", 0}, {"Color", 1}, {"UV0", 2},
            {"UV1", 3}, {"UV2", 4}, {"Normal", 5},
        };
        for (size_t k = 0; k < sizeof(def) / sizeof(def[0]); k++) {
            if (strcmp(def[k].n, name) == 0) {
                return def[k].l;
            }
        }
    }
    return UINT32_MAX;
}

int mglAirReflectModule(const MGLIRModule *mod, int stage,
                        const char *const *attrib_names,
                        MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES],
                        char *err, size_t errCap)
{
    if (!mod || !lists) {
        if (err && errCap) snprintf(err, errCap, "bad args");
        return -1;
    }
    /* Plain (non-sampler) uniforms are packed by the AIR backend into a
     * single std140 struct buffer, so the exporter mirrors that: members
     * are collected here and emitted as one struct-packed resource the
     * renderer's STRUCT_PACKED path consumes. */
    MGLShaderResource agg;
    memset(&agg, 0, sizeof(agg));
    MGLIRType **agg_types = NULL;
    const char **agg_names = NULL;
    uint32_t agg_count = 0;
    uint32_t agg_size = 0;

    /* The AIR backend assigns buffer argument locations independently of
     * any legacy SPIR-V binding decoration: vertex stages start SSBOs at
     * hasBuffer + attrCount (plain-uniform pack first, then attributes),
     * UBOs after the SSBOs; fragment stages start both from 0.  The
     * exporter must mirror that so the renderer's Metal slots match the
     * air.location_index values the metallib actually declares. */
    int isVS = (stage == MGL_STAGE_VERTEX);
    uint32_t user_buffer_base =
        (stage == MGL_STAGE_TESS_EVALUATION) ? 1u : 0u;
    uint32_t attrCount = 0, ssboCount = 0, uboSlotCount = 0, acCount = 0, hasPlain = 0;
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        const MGLIRSymbol *s = mod->symbols[i];
        /* gl_-prefixed symbols are backend builtins (stage I/O like
         * gl_Position / gl_in) and are skipped — EXCEPT:
         *  - uniform-qualified ones: the legacy-GLSL frontend injects the
         *    fixed-function matrix uniforms (gl_ModelViewProjectionMatrix
         *    etc.) verbatim as regular uniforms, and
         *  - explicitly-located ones (location != UINT32_MAX): the legacy
         *    frontend injects gl_Vertex with layout(location = 0) so the
         *    implicit position attribute is bindable at the legacy slot.
         * Both must flow through reflection so the GL uniform/attribute
         * contract and the attrCount-driven slot math stay consistent with
         * the metallib the AIR backend emits (which counts every
         * VarSym::ATTR, gl_-prefixed or not). */
        if (s->is_function ||
            (s->name && strncmp(s->name, "gl_", 3) == 0 &&
             !(s->qualifiers & MGL_AST_Q_UNIFORM) &&
             s->location == UINT32_MAX)) {
            continue;
        }
        uint32_t q = s->qualifiers;
        const MGLIRType *t = s->type;
        const MGLIRType *base_t = t;
        while (base_t && base_t->kind == MGLIR_TYPE_ARRAY)
            base_t = base_t->elem_type;
        if (isVS && (q & MGL_AST_Q_IN)) {
            attrCount++;
        } else if ((q & MGL_AST_Q_UNIFORM) &&
                   base_t && base_t->kind == MGLIR_TYPE_ATOMIC_COUNTER) {
            acCount++;
        } else if ((q & MGL_AST_Q_BUFFER) && !s->block_name) {
            /* Flattened anonymous SSBO members share the owning block's
             * Metal slot; only the block itself advances ssboCount. */
            ssboCount++;
        } else if ((q & MGL_AST_Q_UNIFORM) && !s->block_name &&
                   air_uniform_block_type(t)) {
            uboSlotCount += air_uniform_block_element_count(t);
        } else if ((q & MGL_AST_Q_UNIFORM) &&
                   base_t && base_t->kind != MGLIR_TYPE_SAMPLER &&
                   base_t && base_t->kind != MGLIR_TYPE_IMAGE &&
                   base_t->kind != MGLIR_TYPE_ATOMIC_COUNTER && !s->block_name &&
                   !air_uniform_block_type(t)) {
            hasPlain = 1;
        }
    }
    uint32_t ssbo_binding = user_buffer_base +
        (isVS ? (hasPlain + attrCount)
              : ((stage == MGL_STAGE_COMPUTE ||
                  stage == MGL_STAGE_TESS_EVALUATION ||
                  stage == MGL_STAGE_GEOMETRY) ? hasPlain : 0));
    uint32_t ubo_binding = ssbo_binding + ssboCount;
    uint32_t gl_ubo_binding = 0;
    uint32_t ac_binding = ubo_binding + uboSlotCount;
    if (acCount > 0u && ac_binding + acCount > MAX_BINDABLE_BUFFERS) {
        if (err && errCap) {
            snprintf(err, errCap,
                     "atomic counter Metal slots exceed MAX_BINDABLE_BUFFERS");
        }
        mglAirReflectDestroy(lists);
        return -1;
    }

    /* Sampler / image Metal texture slots must match AIR metadata order
     * (sampled textures first, then storage images) — not GLSL declaration
     * order.  CTS shaders often declare `image2D` before `sampler2D`; a
     * single declaration-order walk swapped those slots and left
     * texelFetch reading the image texture (black). */
    uint32_t texture_binding = 0;
    uint32_t sampler_binding = 0;
    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t i = 0; i < mod->symbol_count; i++) {
            const MGLIRSymbol *s = mod->symbols[i];
            if (!s || s->is_function || !(s->qualifiers & MGL_AST_Q_UNIFORM))
                continue;
            if (s->name && strncmp(s->name, "gl_", 3) == 0 &&
                s->location == UINT32_MAX)
                continue;
            const MGLIRType *t = s->type;
            const MGLIRType *base_t = t;
            while (base_t && base_t->kind == MGLIR_TYPE_ARRAY)
                base_t = base_t->elem_type;
            if (pass == 0) {
                if (!base_t || base_t->kind != MGLIR_TYPE_SAMPLER)
                    continue;
                GLuint location =
                    s->location != UINT32_MAX ? s->location : UINT32_MAX;
                push_resource(&lists[_SAMPLED_IMAGE_RES], s, t, location,
                              texture_binding, stage);
                MGLShaderResource *last =
                    &lists[_SAMPLED_IMAGE_RES]
                         .list[lists[_SAMPLED_IMAGE_RES].count - 1];
                if (s->binding != UINT32_MAX) {
                    last->gl_binding = s->binding;
                    /* layout(binding=N) sets the sampler uniform's initial
                     * texture-unit value (queried via GetUniformiv).  Array
                     * elements take N, N+1, … from sampler_unit + ordinal. */
                    last->sampler_unit = (GLint)s->binding;
                }
                last->resource_active = GL_TRUE;
                last->has_combined_sampler = GL_TRUE;
                last->combined_sampler_binding = sampler_binding;
                last->uniform_location =
                    (s->location != UINT32_MAX)
                        ? (GLint)s->location
                        : mglSyntheticSamplerUniformLocation(
                              stage, _SAMPLED_IMAGE_RES, sampler_binding);
                GLuint elements = mglAirGLArraySizeFromIR(t);
                if (elements < 1u) elements = 1u;
                texture_binding += elements;
                sampler_binding += elements;
            } else {
                if (!base_t || base_t->kind != MGLIR_TYPE_IMAGE)
                    continue;
                GLuint location =
                    s->location != UINT32_MAX ? s->location : UINT32_MAX;
                push_resource(&lists[_STORAGE_IMAGE_RES], s, t, location,
                              texture_binding, stage);
                MGLShaderResource *last =
                    &lists[_STORAGE_IMAGE_RES]
                         .list[lists[_STORAGE_IMAGE_RES].count - 1];
                last->sampler_unit = -1;
                if (s->binding != UINT32_MAX) {
                    last->gl_binding = s->binding;
                    /* Same as samplers: layout(binding=N) is the image-unit
                     * initial value for GetUniformiv / array expansion. */
                    last->sampler_unit = (GLint)s->binding;
                }
                last->uniform_location =
                    (s->location != UINT32_MAX)
                        ? (GLint)s->location
                        : mglSyntheticSamplerUniformLocation(
                              stage, _STORAGE_IMAGE_RES, texture_binding);
                GLuint elements = mglAirGLArraySizeFromIR(t);
                if (elements < 1u) elements = 1u;
                texture_binding += elements;
            }
        }
    }
    /* Extra auto-location stride consumed by interface-block array
     * members (one location per element, see the Q_IN branch below). */
    uint32_t gs_input_span_pad = 0;
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        const MGLIRSymbol *s = mod->symbols[i];
        /* gl_-prefixed symbols are backend builtins (stage I/O like
         * gl_Position / gl_in) and are skipped — EXCEPT:
         *  - uniform-qualified ones: the legacy-GLSL frontend injects the
         *    fixed-function matrix uniforms (gl_ModelViewProjectionMatrix
         *    etc.) verbatim as regular uniforms, and
         *  - explicitly-located ones (location != UINT32_MAX): the legacy
         *    frontend injects gl_Vertex with layout(location = 0) so the
         *    implicit position attribute is bindable at the legacy slot.
         * Both must flow through reflection so the GL uniform/attribute
         * contract and the attrCount-driven slot math stay consistent with
         * the metallib the AIR backend emits (which counts every
         * VarSym::ATTR, gl_-prefixed or not). */
        if (s->is_function ||
            (s->name && strncmp(s->name, "gl_", 3) == 0 &&
             !(s->qualifiers & MGL_AST_Q_UNIFORM) &&
             s->location == UINT32_MAX)) {
            continue;
        }
        const MGLIRType *t = s->type;
        const MGLIRType *base_t = t;
        while (base_t && base_t->kind == MGLIR_TYPE_ARRAY)
            base_t = base_t->elem_type;
        uint32_t q = s->qualifiers;
        GLuint location = s->location != UINT32_MAX ? s->location : UINT32_MAX;

        /* Interface-block instances flatten into per-member varying
         * symbols (each with block_name set); the struct-typed instance
         * symbol itself is not an interface resource. */
        if (!s->block_name && (q & (MGL_AST_Q_IN | MGL_AST_Q_OUT)) &&
            !(q & (MGL_AST_Q_UNIFORM | MGL_AST_Q_BUFFER)) &&
            (t->kind == MGLIR_TYPE_STRUCT ||
             (t->kind == MGLIR_TYPE_ARRAY && t->elem_type &&
              t->elem_type->kind == MGLIR_TYPE_STRUCT))) {
            continue;
        }

        if (q & MGL_AST_Q_UNIFORM) {
            /* Samplers / images already pushed above (AIR slot order). */
            if (base_t && (base_t->kind == MGLIR_TYPE_SAMPLER ||
                           base_t->kind == MGLIR_TYPE_IMAGE))
                continue;
            if (base_t && base_t->kind == MGLIR_TYPE_ATOMIC_COUNTER) {
                push_resource(&lists[_ATOMIC_COUNTER_RES], s, t, location,
                              ac_binding++, stage);
                MGLShaderResource *last =
                    &lists[_ATOMIC_COUNTER_RES].list[
                        lists[_ATOMIC_COUNTER_RES].count - 1];
                if (s->binding != UINT32_MAX) {
                    last->gl_binding = s->binding;
                } else {
                    last->gl_binding = 0;
                }
                last->location = s->offset != UINT32_MAX ? s->offset : 0u;
                {
                    GLuint elements = mglAirGLArraySizeFromIR(t);
                    if (elements < 1u) elements = 1u;
                    last->required_size = elements * (GLuint)sizeof(GLuint);
                }
                continue;
            }
            if (s->block_name) {
                continue;   /* block member: covered by the block resource */
            }
            const MGLIRType *block_type = air_uniform_block_type(t);
            if (block_type) {
                /* A block instance array is one GL block per element, backed
                 * by consecutive Metal buffer arguments.  Keep one reflected
                 * resource with per-element binding metadata so the common
                 * buffer mapper expands it at draw time. */
                GLuint block_count = air_uniform_block_element_count(t);
                push_resource(&lists[_UNIFORM_BUFFER_RES], s, block_type,
                              location, ubo_binding, stage);
                MGLShaderResource *last =
                    &lists[_UNIFORM_BUFFER_RES].list[
                        lists[_UNIFORM_BUFFER_RES].count - 1];
                apply_block_interface_name(last, t, s->name);
                last->ubo_array_size = block_count;
                last->ubo_is_array =
                    (t->kind == MGLIR_TYPE_ARRAY && t->array_size > 0u)
                        ? GL_TRUE : GL_FALSE;
                if (last->ubo_is_array) {
                    last->ubo_array_bindings = (GLuint *)calloc(
                        block_count, sizeof(*last->ubo_array_bindings));
                }
                GLuint gl_block_binding = s->binding != UINT32_MAX
                    ? s->binding : gl_ubo_binding;
                last->gl_binding = gl_block_binding;
                if (last->ubo_array_bindings) {
                    for (GLuint element = 0; element < block_count; element++) {
                        last->ubo_array_bindings[element] =
                            gl_block_binding + element;
                    }
                }
                gl_ubo_binding += block_count;
                ubo_binding += block_count;
                continue;
            }
            /* Plain uniform: collect into the packed aggregate. */
            MGLIRType **nt = (MGLIRType **)realloc(
                agg_types, (agg_count + 1) * sizeof(MGLIRType *));
            const char **nn = (const char **)realloc(
                agg_names, (agg_count + 1) * sizeof(const char *));
            if (!nt || !nn) {
                if (err && errCap) snprintf(err, errCap, "out of memory");
                mglAirReflectDestroy(lists);
                return -1;
            }
            agg_types = nt;
            agg_names = nn;
            agg_types[agg_count] = (MGLIRType *)t;
            agg_names[agg_count] = s->name;
            agg_count++;
            continue;
        }
        if ((q & MGL_AST_Q_BUFFER) && !s->block_name) {
            /* Mirror UBO instance arrays: one reflected resource with a
             * per-element binding table so consecutive GL binding points
             * expand to consecutive Metal buffer arguments. */
            const MGLIRType *block_type = air_uniform_block_type(t);
            GLuint block_count = air_uniform_block_element_count(t);
            const MGLIRType *res_type = block_type ? block_type : t;
            push_resource(&lists[_STORAGE_BUFFER_RES], s, res_type,
                          location, ssbo_binding, stage);
            MGLShaderResource *ssbo_last =
                &lists[_STORAGE_BUFFER_RES].list[
                    lists[_STORAGE_BUFFER_RES].count - 1];
            apply_block_interface_name(ssbo_last, t, s->name);
            ssbo_last->ubo_array_size = block_count;
            ssbo_last->ubo_is_array =
                (t->kind == MGLIR_TYPE_ARRAY && t->array_size > 0u)
                    ? GL_TRUE : GL_FALSE;
            if (ssbo_last->ubo_is_array) {
                ssbo_last->ubo_array_bindings = (GLuint *)calloc(
                    block_count, sizeof(*ssbo_last->ubo_array_bindings));
            }
            GLuint gl_block_binding = s->binding != UINT32_MAX
                ? s->binding : ssbo_binding;
            ssbo_last->gl_binding = gl_block_binding;
            if (ssbo_last->ubo_array_bindings) {
                for (GLuint element = 0; element < block_count; element++) {
                    ssbo_last->ubo_array_bindings[element] =
                        gl_block_binding + element;
                }
            }
            ssbo_binding += block_count;
        } else if (q & MGL_AST_Q_IN) {
            /* Desired location: explicit bindings, stable names, then
             * declaration order (CLI-style auto-mapped locations). */
            uint32_t want = air_reflect_attrib_location(s->name,
                                                        attrib_names);
            if (want != UINT32_MAX) {
                location = want;
            } else if (location == UINT32_MAX) {
                location = lists[_STAGE_INPUT_RES].count +
                           gs_input_span_pad;
            }
            push_resource(&lists[_STAGE_INPUT_RES], s, t, location, 0,
                          stage);
            /* Array / matrix attributes occupy one location per element or
             * column so glGetAttribLocation("a[i]") == base+i stays aligned
             * with VAO binds and the AIR vertex_input location_index
             * sequence (GL 4.6 §4.4.1). */
            if (t->kind == MGLIR_TYPE_ARRAY && t->array_size > 1u) {
                gs_input_span_pad += t->array_size - 1u;
            } else if (t->kind == MGLIR_TYPE_MATRIX && t->cols > 1u) {
                gs_input_span_pad += t->cols - 1u;
            }
        } else if (q & MGL_AST_Q_OUT) {
            if (location == UINT32_MAX) {
                location = lists[_STAGE_OUTPUT_RES].count;
            }
            push_resource(&lists[_STAGE_OUTPUT_RES], s, t, location, 0,
                          stage);
        }
    }

    if (agg_count > 0) {
        /* std140 layout, mirroring the AIR backend's collectUniforms. */
        uint32_t off = 0;
        agg.ubo_members = (SpirvUBOMember *)calloc(
            agg_count, sizeof(SpirvUBOMember));
        if (!agg.ubo_members) {
            free(agg_types);
            free(agg_names);
            mglAirReflectDestroy(lists);
            if (err && errCap) snprintf(err, errCap, "out of memory");
            return -1;
        }
        agg.ubo_member_count = agg_count;
        for (uint32_t m = 0; m < agg_count; m++) {
            uint32_t size = 0;
            if (mglIRComputeLayout(agg_types[m], MGLIR_LAYOUT_STD140, &size) != 0) {
                size = 4;
            }
            off = (off + agg_types[m]->layout.alignment - 1) &
                  ~(agg_types[m]->layout.alignment - 1);
            SpirvUBOMember *u = &agg.ubo_members[m];
            u->name = strdup(agg_names[m]);
            u->query_name = strdup(agg_names[m]);
            u->gl_type = mglAirGLTypeFromIR(agg_types[m]);
            u->offset = off;
            u->array_stride = (agg_types[m]->kind == MGLIR_TYPE_ARRAY)
                                  ? (GLint)agg_types[m]->layout.array_stride
                                  : -1;
            u->matrix_stride = (agg_types[m]->kind == MGLIR_TYPE_MATRIX)
                                   ? (GLint)agg_types[m]->layout.matrix_stride
                                   : -1;
            u->is_row_major = GL_FALSE;
            u->size = mglAirGLArraySizeFromIR(agg_types[m]);
            u->location_offset = (GLint)m;
            u->top_level_array_size = u->size;
            u->top_level_array_stride = u->array_stride;
            off += size;
        }
        agg_size = off;
        char agg_name[64];
        snprintf(agg_name, sizeof(agg_name), "air_uniforms_s%d", stage);
        agg.name = strdup(agg_name);
        agg.ubo_member_count = agg_count;
        agg.required_size = agg_size;
        agg.uniform_location = -1;   /* assigned by the link pass per stage */
        agg.location = UINT32_MAX;   /* let the link pass assign locations */
        agg.gl_binding = 0;
        agg.binding = user_buffer_base;
        MGLShaderResourceList *l = &lists[_UNIFORM_CONSTANT_RES];
        MGLShaderResource *nl = (MGLShaderResource *)realloc(
            l->list, (l->count + 1) * sizeof(MGLShaderResource));
        if (nl) {
            l->list = nl;
            l->list[l->count++] = agg;
        }
        free(agg_types);
        free(agg_names);
    }
    return 0;
}

void mglAirReflectDestroy(MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES])
{
    if (!lists) {
        return;
    }
    for (int i = 0; i < MGL_MAX_SHADER_RESOURCES; i++) {
        destroy_list(&lists[i]);
    }
}
