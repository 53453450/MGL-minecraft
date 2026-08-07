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
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        const MGLIRSymbol *s = mod->symbols[i];
        if (s->is_function) {
            continue;
        }
        const MGLIRType *t = s->type;
        uint32_t q = s->qualifiers;
        GLuint location = s->location != UINT32_MAX ? s->location : 0;
        GLuint binding = s->binding != UINT32_MAX ? s->binding : 0;
        SpirvResourceList *dst = NULL;

        if (q & MGL_AST_Q_UNIFORM) {
            if (t->kind == MGLIR_TYPE_SAMPLER) {
                dst = &lists[_SEPARATE_IMAGE_RES];
            } else {
                dst = &lists[_UNIFORM_CONSTANT_RES];
            }
        } else if (q & MGL_AST_Q_BUFFER) {
            dst = &lists[_STORAGE_BUFFER_RES];
        } else if (q & MGL_AST_Q_IN) {
            dst = &lists[_STAGE_INPUT_RES];
        } else if (q & MGL_AST_Q_OUT) {
            dst = &lists[_STAGE_OUTPUT_RES];
        } else {
            continue;   /* locals and unsupported kinds are not reflected */
        }
        push_resource(dst, s, t, location, binding, stage);
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
