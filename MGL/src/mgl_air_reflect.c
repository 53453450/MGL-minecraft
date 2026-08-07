/*
 * mgl_air_reflect.c
 * MGL
 *
 * Reflection export from the self-hosted GLSL frontend's MGLIRModule.
 * Produces the SpirvResource tables the GL query layer and the per-draw
 * binding paths consume, replacing the SPIRV-Cross reflection step.
 */

#include "mgl_air_reflect.h"

#include <GL/glcorearb.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    case MGLIR_TYPE_SAMPLER:
        switch (t->tex_kind) {
        case MGLIR_TEX_2D:   return GL_SAMPLER_2D;
        case MGLIR_TEX_3D:   return GL_SAMPLER_3D;
        case MGLIR_TEX_CUBE: return GL_SAMPLER_CUBE;
        default:             return GL_SAMPLER_2D;
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

static void push_resource(SpirvResourceList *list, const MGLIRSymbol *s,
                          const MGLIRType *type, GLuint location,
                          GLuint binding, int stage)
{
    SpirvResource r;
    memset(&r, 0, sizeof(r));
    r.name = strdup(s->name);
    r.msl_name = strdup(s->name);
    r.location = location;
    r.gl_binding = binding;
    r.binding = binding;
    r.gl_type = mglAirGLTypeFromIR(type);
    r.gl_array_size = mglAirGLArraySizeFromIR(type);
    r.is_array = (type->kind == MGLIR_TYPE_ARRAY) ? GL_TRUE : GL_FALSE;
    r.num_array_dims = (type->kind == MGLIR_TYPE_ARRAY) ? 1u : 0u;
    r.uniform_location = -1;
    r.sampler_unit = 0;
    r.sampler_unit_explicit = GL_FALSE;
    (void)stage;

    if (type->kind == MGLIR_TYPE_STRUCT && type->member_count > 0) {
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

    SpirvResource *nl = (SpirvResource *)realloc(
        list->list, (list->count + 1) * sizeof(SpirvResource));
    if (!nl) {
        return;
    }
    list->list = nl;
    list->list[list->count++] = r;
}

static void destroy_list(SpirvResourceList *list)
{
    for (GLuint i = 0; i < list->count; i++) {
        SpirvResource *r = &list->list[i];
        free((void *)r->name);
        free(r->msl_name);
        if (r->msl_argument_names) {
            for (GLuint a = 0; a < r->msl_argument_count; a++) {
                free(r->msl_argument_names[a]);
            }
            free(r->msl_argument_names);
        }
        free(r->msl_combined_sampler_name);
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

int mglAirReflectModule(const MGLIRModule *mod, int stage,
                        SpirvResourceList lists[_MAX_SPIRV_RES],
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
    SpirvResource agg;
    memset(&agg, 0, sizeof(agg));
    MGLIRType **agg_types = NULL;
    const char **agg_names = NULL;
    uint32_t agg_count = 0;
    uint32_t agg_size = 0;

    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        const MGLIRSymbol *s = mod->symbols[i];
        if (s->is_function) {
            continue;
        }
        const MGLIRType *t = s->type;
        uint32_t q = s->qualifiers;
        GLuint location = s->location != UINT32_MAX ? s->location : UINT32_MAX;
        GLuint binding = s->binding != UINT32_MAX ? s->binding : 0;

        if (q & MGL_AST_Q_UNIFORM) {
            if (t->kind == MGLIR_TYPE_SAMPLER) {
                push_resource(&lists[_SEPARATE_IMAGE_RES], s, t, location,
                              binding, stage);
                continue;
            }
            if (s->block_name) {
                continue;   /* block member: covered by the block resource */
            }
            if (t->kind == MGLIR_TYPE_STRUCT && t->member_count > 0) {
                /* Uniform block: independent resource with members. */
                push_resource(&lists[_UNIFORM_BUFFER_RES], s, t, location,
                              binding, stage);
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
                          binding, stage);
        } else if (q & MGL_AST_Q_IN) {
            push_resource(&lists[_STAGE_INPUT_RES], s, t, location, binding,
                          stage);
        } else if (q & MGL_AST_Q_OUT) {
            push_resource(&lists[_STAGE_OUTPUT_RES], s, t, location, binding,
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
        agg.name = strdup("air_uniforms");
        agg.msl_name = strdup("air_uniforms");
        agg.ubo_member_count = agg_count;
        agg.required_size = agg_size;
        agg.uniform_location = 0;
        agg.location = UINT32_MAX;   /* let the link pass assign locations */
        agg.gl_binding = 0;
        agg.binding = 0;
        SpirvResourceList *l = &lists[_UNIFORM_CONSTANT_RES];
        SpirvResource *nl = (SpirvResource *)realloc(
            l->list, (l->count + 1) * sizeof(SpirvResource));
        if (nl) {
            l->list = nl;
            l->list[l->count++] = agg;
        }
        free(agg_types);
        free(agg_names);
    }
    return 0;
}

void mglAirReflectDestroy(SpirvResourceList lists[_MAX_SPIRV_RES])
{
    if (!lists) {
        return;
    }
    for (int i = 0; i < _MAX_SPIRV_RES; i++) {
        destroy_list(&lists[i]);
    }
}
