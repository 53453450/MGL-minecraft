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
    case MGLIR_TYPE_MATRIX:
        return (GLuint)(GL_FLOAT_MAT2 + (t->cols - 2));
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
        return GL_INVALID_ENUM;
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
    r.gl_type = mglAirGLTypeFromIR(type);
    r.gl_array_size = mglAirGLArraySizeFromIR(type);
    r.is_array = (type->kind == MGLIR_TYPE_ARRAY) ? GL_TRUE : GL_FALSE;
    r.is_per_patch = (s->qualifiers & MGL_AST_Q_PATCH) ? GL_TRUE : GL_FALSE;
    r.stream = (s->stream >= 0) ? s->stream : 0;
    r.num_array_dims = (type->kind == MGLIR_TYPE_ARRAY) ? 1u : 0u;
    r.uniform_location = -1;
    r.sampler_unit = 0;
    r.sampler_unit_explicit = GL_FALSE;
    if (type->kind == MGLIR_TYPE_SAMPLER ||
        type->kind == MGLIR_TYPE_IMAGE) {
        switch (type->tex_kind) {
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
            type->tex_kind == MGLIR_TEX_1D_ARRAY ||
            type->tex_kind == MGLIR_TEX_2D_ARRAY ||
            type->tex_kind == MGLIR_TEX_CUBE_ARRAY ||
            type->tex_kind == MGLIR_TEX_2D_MS_ARRAY;
        r.image_multisampled =
            type->tex_kind == MGLIR_TEX_2D_MS ||
            type->tex_kind == MGLIR_TEX_2D_MS_ARRAY ||
            type->tex_kind == MGLIR_TEX_SUBPASS_MS;
        r.texture_data_kind = type->tex_depth
            ? MGL_SHADER_TEXTURE_DATA_DEPTH
            : type->tex_storage == MGLIR_SCALAR_INT
                ? MGL_SHADER_TEXTURE_DATA_SINT
                : type->tex_storage == MGLIR_SCALAR_UINT
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
        r.ubo_member_count = type->member_count;
        r.ubo_members = (SpirvUBOMember *)calloc(
            type->member_count, sizeof(SpirvUBOMember));
        for (uint32_t m = 0; m < type->member_count; m++) {
            const MGLIRType *mt = type->members[m];
            SpirvUBOMember *u = &r.ubo_members[m];
            u->name = strdup(type->member_names[m]);
            u->query_name = strdup(type->member_names[m]);
            u->gl_type = mglAirGLTypeFromIR(mt);
            u->offset = type->member_offsets ? type->member_offsets[m] : 0;
            u->array_stride = (mt->kind == MGLIR_TYPE_ARRAY)
                                  ? (GLint)mt->layout.array_stride
                                  : -1;
            u->matrix_stride = (mt->kind == MGLIR_TYPE_MATRIX)
                                   ? (GLint)mt->layout.matrix_stride
                                   : -1;
            u->is_row_major = GL_FALSE;
            u->size = mglAirGLArraySizeFromIR(mt);
            u->location_offset = -1;
            u->top_level_array_size = u->size;
            u->top_level_array_stride = u->array_stride;
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
    uint32_t attrCount = 0, ssboCount = 0, hasPlain = 0;
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        const MGLIRSymbol *s = mod->symbols[i];
        if (s->is_function ||
            (s->name && strncmp(s->name, "gl_", 3) == 0)) {
            continue;
        }
        uint32_t q = s->qualifiers;
        const MGLIRType *t = s->type;
        if (isVS && (q & MGL_AST_Q_IN)) {
            attrCount++;
        } else if (q & MGL_AST_Q_BUFFER) {
            ssboCount++;
        } else if ((q & MGL_AST_Q_UNIFORM) &&
                   t->kind != MGLIR_TYPE_SAMPLER &&
                   t->kind != MGLIR_TYPE_IMAGE && !s->block_name &&
                   !(t->kind == MGLIR_TYPE_STRUCT && t->member_count > 0)) {
            hasPlain = 1;
        }
    }
    uint32_t ssbo_binding = user_buffer_base +
        (isVS ? (hasPlain + attrCount)
              : ((stage == MGL_STAGE_COMPUTE ||
                  stage == MGL_STAGE_TESS_EVALUATION) ? hasPlain : 0));
    uint32_t ubo_binding = ssbo_binding + ssboCount;

    /* Sampler bindings increment per sampler, matching the AIR metadata
     * texture location indices. */
    uint32_t texture_binding = 0;
    uint32_t sampler_binding = 0;
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        const MGLIRSymbol *s = mod->symbols[i];
        if (s->is_function ||
            (s->name && strncmp(s->name, "gl_", 3) == 0)) {
            continue;
        }
        const MGLIRType *t = s->type;
        uint32_t q = s->qualifiers;
        GLuint location = s->location != UINT32_MAX ? s->location : UINT32_MAX;

        if (q & MGL_AST_Q_UNIFORM) {
            if (t->kind == MGLIR_TYPE_SAMPLER) {
                push_resource(&lists[_SAMPLED_IMAGE_RES], s, t, location,
                              texture_binding, stage);
                MGLShaderResource *last =
                    &lists[_SAMPLED_IMAGE_RES].list[
                        lists[_SAMPLED_IMAGE_RES].count - 1];
                if (s->binding != UINT32_MAX) {
                    last->gl_binding = s->binding;
                }
                last->resource_active = GL_TRUE;
                last->has_combined_sampler = GL_TRUE;
                last->combined_sampler_binding = sampler_binding;
                /* Sampler GL uniform locations live in the synthetic
                 * namespace (mirrors the SPIRV-Cross-era path in
                 * mglSamplerUniformLocationFromReflection) unless the GLSL
                 * declares an explicit layout(location=N); otherwise MC's
                 * glGetUniformLocation/glUniform1i sampler-unit setup
                 * cannot target this resource. */
                last->uniform_location =
                    (s->location != UINT32_MAX)
                        ? (GLint)s->location
                        : mglSyntheticSamplerUniformLocation(
                              stage, _SAMPLED_IMAGE_RES, sampler_binding);
                texture_binding++;
                sampler_binding++;
                continue;
            }
            if (t->kind == MGLIR_TYPE_IMAGE) {
                push_resource(&lists[_STORAGE_IMAGE_RES], s, t, location,
                              texture_binding, stage);
                MGLShaderResource *last =
                    &lists[_STORAGE_IMAGE_RES].list[
                        lists[_STORAGE_IMAGE_RES].count - 1];
                last->sampler_unit = -1;
                if (s->binding != UINT32_MAX) {
                    last->gl_binding = s->binding;
                }
                texture_binding++;
                continue;
            }
            if (s->block_name) {
                continue;   /* block member: covered by the block resource */
            }
            if (t->kind == MGLIR_TYPE_STRUCT && t->member_count > 0) {
                /* Uniform block: independent resource with members. */
                push_resource(&lists[_UNIFORM_BUFFER_RES], s, t, location,
                              ubo_binding++, stage);
                if (s->binding != UINT32_MAX) {
                    lists[_UNIFORM_BUFFER_RES].list[
                        lists[_UNIFORM_BUFFER_RES].count - 1].gl_binding =
                        s->binding;
                }
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
        if (q & MGL_AST_Q_BUFFER) {
            push_resource(&lists[_STORAGE_BUFFER_RES], s, t, location,
                          ssbo_binding++, stage);
            if (s->binding != UINT32_MAX) {
                lists[_STORAGE_BUFFER_RES].list[
                    lists[_STORAGE_BUFFER_RES].count - 1].gl_binding = s->binding;
            }
        } else if (q & MGL_AST_Q_IN) {
            /* Desired location: explicit bindings, stable names, then
             * declaration order (CLI-style auto-mapped locations). */
            uint32_t want = air_reflect_attrib_location(s->name,
                                                        attrib_names);
            if (want != UINT32_MAX) {
                location = want;
            } else if (location == UINT32_MAX) {
                location = lists[_STAGE_INPUT_RES].count;
            }
            push_resource(&lists[_STAGE_INPUT_RES], s, t, location, 0,
                          stage);
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
