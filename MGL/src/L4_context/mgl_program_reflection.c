/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_program_reflection.h"
#include "mgl.h"
#include "mgl_glsl_ast.h" /* MGLTranslationUnit / MGLDecl: gl_PerVertex redeclarations */
#include "mgl_metal_ref.h"
#include "mgl_uniform_reflection.h"
#include "mgl_program_backend.h"
#include "mgl_render_api_program.h"

_Static_assert(sizeof(((struct MGLProgramBackend *)0)->modules) <=
                   (size_t)MGL_RENDER_PROGRAM_MODULE_BACKEND_BLOB,
               "MGL_RENDER_PROGRAM_MODULE_BACKEND_BLOB too small");
_Static_assert(MGL_PROGRAM_BACKEND_STAGES == _MAX_SHADER_TYPES,
               "program backend stage count must match shader types");

void mglShaderModuleBackendClear(struct MGLShaderModuleBackend *module)
{
    if (!module) {
        return;
    }
    free(module->metallib_bytes);
    free(module->metallib_tess_capture_bytes);
    free(module->metallib_cull_capture_bytes);
    free(module->metallib_bytes_tes_compute);
    mglSafeReleaseMetalObj(&module->mtl_function);
    mglSafeReleaseMetalObj(&module->mtl_function_compute);
    mglSafeReleaseMetalObj(&module->mtl_library);
    mglSafeReleaseMetalObj(&module->mtl_tess_capture_function);
    mglSafeReleaseMetalObj(&module->mtl_tess_capture_library);
    mglSafeReleaseMetalObj(&module->mtl_cull_capture_function);
    mglSafeReleaseMetalObj(&module->mtl_cull_capture_library);
    memset(module, 0, sizeof(*module));
}

void mglProgramBackendRelease(Program *program)
{
    if (!program || !program->backend) {
        return;
    }
    for (int stage = 0; stage < MGL_PROGRAM_BACKEND_STAGES; stage++) {
        mglShaderModuleBackendClear(&program->backend->modules[stage]);
    }
    free(program->backend);
    program->backend = NULL;
}

void clearStageCompileState(Program *program, int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }

    MGLShaderModule *compiled = &program->modules[stage];
    free(compiled->entry_point);
    compiled->entry_point = NULL;
    if (program->backend) {
        mglShaderModuleBackendClear(&program->backend->modules[stage]);
    }

    for (int type = 0; type < MGL_MAX_SHADER_RESOURCES; type++) {
        MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            mglFreeMGLShaderResourceOwnedFields(&list->list[i]);
        }
        free(list->list);
        list->list = NULL;
        list->count = 0;
    }
}

GLboolean mglProgramPerVertexSignature(Program *program, int stage,
                                       unsigned *signature)
{
    if (signature) {
        *signature = 0;
    }
    if (!program || !signature || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return GL_FALSE;
    }

    /* A gl_PerVertex redeclaration is a normal block declaration in the parsed
     * TU (type name "gl_PerVertex", members = the redeclared builtins), so the
     * signature is read from the declaration instead of scanning the source
     * text for the block and its member tokens: a commented-out or similarly
     * named struct is no longer mistaken for a redeclaration, and a stage
     * without a TU simply has no signature.  The instance name is irrelevant -
     * `out gl_PerVertex { ... } vs_out;` declares the same interface. */
    Shader *shader = program->shader_slots[stage];
    MGLTranslationUnit *tu = shader ? shader->frontend_tu : NULL;
    if (!tu) {
        return GL_FALSE;
    }

    unsigned result = 0;
    GLboolean found = GL_FALSE;
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        for (MGLDecl *decl = tu->decls[i]; decl; decl = decl->next_declarator) {
            if (!decl->type || !decl->type->name ||
                strcmp(decl->type->name, "gl_PerVertex") != 0) {
                continue;
            }
            for (uint32_t m = 0; m < decl->struct_member_count; m++) {
                const MGLDecl *member = decl->struct_members[m];
                const char *name = member ? member->name : NULL;
                if (!name) {
                    continue;
                }
                if (strcmp(name, "gl_Position") == 0) {
                    result |= 1u << 0;
                } else if (strcmp(name, "gl_PointSize") == 0) {
                    result |= 1u << 1;
                } else if (strcmp(name, "gl_ClipDistance") == 0) {
                    result |= 1u << 2;
                } else if (strcmp(name, "gl_CullDistance") == 0) {
                    result |= 1u << 3;
                }
            }
            found = GL_TRUE;
        }
    }

    if (found) {
        *signature = result;
    }
    return found;
}

GLboolean mglProgramPipelinePerVertexCompatible(
    Program *const *stage_programs)
{
    unsigned reference = 0;
    GLboolean have_reference = GL_FALSE;
    if (!stage_programs) {
        return GL_TRUE;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Program *program = stage_programs[stage];
        unsigned signature = 0;
        if (!program || !program->shader_slots[stage] ||
            !mglProgramPerVertexSignature(program, stage, &signature)) {
            continue;
        }
        if (!have_reference) {
            reference = signature;
            have_reference = GL_TRUE;
        } else if (signature != reference) {
            return GL_FALSE;
        }
    }
    return GL_TRUE;
}

GLboolean mglProgramPipelineHasOrphanedLinkedStages(
    Program *const *stage_programs)
{
    if (!stage_programs) {
        return GL_FALSE;
    }
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Program *program = stage_programs[stage];
        if (!program) {
            continue;
        }
        for (int other = 0; other < _MAX_SHADER_TYPES; other++) {
            if (program->shader_slots[other] &&
                stage_programs[other] != program) {
                return GL_TRUE;
            }
        }
    }
    return GL_FALSE;
}

GLboolean mglLinkedProgramPerVertexCompatible(Program *program)
{
    Program *stages[_MAX_SHADER_TYPES] = {0};
    if (!program) {
        return GL_TRUE;
    }
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if ((program->attached_shader_mask & (1u << stage)) &&
            program->shader_slots[stage]) {
            stages[stage] = program;
        }
    }
    return mglProgramPipelinePerVertexCompatible(stages);
}

void applyMultiDimArrayUniformNames(Program *program)
{
    (void)program;
}

void applyFragmentOutputLocationIndices(Program *program)
{
    if (!program || program->frag_data_location_count == 0) return;
    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_OUTPUT_RES];
    for (GLuint i = 0; i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (!output->name) continue;
        for (GLuint j = 0; j < program->frag_data_location_count; j++) {
            if (program->frag_data_location_names[j] &&
                strcmp(program->frag_data_location_names[j], output->name) == 0) {
                output->location = program->frag_data_color_numbers[j];
                output->location_index = program->frag_data_indices[j];
                break;
            }
        }
    }
}

GLboolean mglProgramVaryingTypesCompatible(const MGLShaderResource *a,
                                           const MGLShaderResource *b)
{
    if (!a || !b) return GL_FALSE;
    if (a->gl_type && b->gl_type && a->gl_type != b->gl_type) {
        return GL_FALSE;
    }
    if (a->gl_array_size > 0 && b->gl_array_size > 0 &&
        a->gl_array_size != b->gl_array_size) {
        return GL_FALSE;
    }
    return GL_TRUE;
}

MGLShaderResource *mglFindVaryingByName(MGLShaderResourceList *list,
                                    const char *name,
                                    const MGLShaderResource *type_peer)
{
    if (!list || !name) return NULL;
    for (GLuint i = 0; i < list->count; i++) {
        MGLShaderResource *candidate = &list->list[i];
        if (candidate->name && strcmp(candidate->name, name) == 0 &&
            (!type_peer || mglProgramVaryingTypesCompatible(candidate,
                                                            type_peer))) {
            return candidate;
        }
    }
    return NULL;
}

MGLShaderResource *mglFindVaryingByLocation(MGLShaderResourceList *list,
                                        GLuint location,
                                        const MGLShaderResource *type_peer)
{
    if (!list) return NULL;
    for (GLuint i = 0; i < list->count; i++) {
        MGLShaderResource *candidate = &list->list[i];
        if (candidate->location == location &&
            (!type_peer || mglProgramVaryingTypesCompatible(candidate,
                                                            type_peer))) {
            return candidate;
        }
    }
    return NULL;
}

static void mglAlignInputsToOutputs(MGLShaderResourceList *outputs,
                                    MGLShaderResourceList *inputs)
{
    if (!outputs || !inputs) return;
    for (GLuint i = 0; i < inputs->count; i++) {
        MGLShaderResource *input = &inputs->list[i];
        MGLShaderResource *output = mglFindVaryingByName(outputs, input->name,
                                                     input);
        if (!output) {
            output = mglFindVaryingByLocation(outputs, input->location, input);
        }
        if (output) {
            input->location = output->location;
        }
    }
}

void alignFragmentInputLocationsToVertexOutputs(Program *program)
{
    if (!program) return;
    mglAlignInputsToOutputs(
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES],
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES]);
}

void mglBridgeSkippedGeometryShaderVaryings(Program *program)
{
    if (!program || !program->shader_slots[_GEOMETRY_SHADER]) return;
    mglAlignInputsToOutputs(
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES],
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES]);
}

void mglRenderProgramReleaseBackend(Program *program)
{
    mglProgramBackendRelease(program);
}

uint64_t mglRenderProgramPipelineCacheInstanceId(const Program *program)
{
    return mglProgramBackendView(program)->pipeline_cache_instance_id;
}

void mglRenderProgramAssignPipelineCacheInstanceId(Program *program,
                                                   uint64_t instance_id)
{
    mglProgramBackendEnsure(program)->pipeline_cache_instance_id = instance_id;
}

void mglRenderProgramBumpPipelineCacheGeneration(Program *program)
{
    mglProgramBackendEnsure(program)->pipeline_cache_generation++;
}

void mglRenderProgramCaptureModuleBackends(Program *program, void *blob,
                                           size_t blob_bytes)
{
    size_t need = sizeof(((struct MGLProgramBackend *)0)->modules);
    if (!blob || blob_bytes < need) {
        return;
    }
    memset(blob, 0, blob_bytes);
    if (program && program->backend) {
        memcpy(blob, program->backend->modules, need);
        memset(program->backend->modules, 0, need);
    }
}

void mglRenderProgramClearModuleBackends(Program *program)
{
    if (program && program->backend) {
        memset(program->backend->modules, 0,
               sizeof(program->backend->modules));
    }
}

void mglRenderProgramRestoreModuleBackends(Program *program, const void *blob,
                                           size_t blob_bytes)
{
    size_t need = sizeof(((struct MGLProgramBackend *)0)->modules);
    if (!program || !blob || blob_bytes < need) {
        return;
    }
    memcpy(mglProgramBackendEnsure(program)->modules, blob, need);
}

void mglRenderProgramClearCapturedModuleBackend(void *blob, size_t blob_bytes,
                                                int stage)
{
    size_t one = sizeof(struct MGLShaderModuleBackend);
    size_t need = one * (size_t)MGL_PROGRAM_BACKEND_STAGES;
    if (!blob || blob_bytes < need || stage < 0 ||
        stage >= MGL_PROGRAM_BACKEND_STAGES) {
        return;
    }
    mglShaderModuleBackendClear(
        (struct MGLShaderModuleBackend *)((unsigned char *)blob +
                                          (size_t)stage * one));
}

const unsigned char *mglRenderProgramModuleMetallibBytes(const Program *program,
                                                         int stage)
{
    return mglProgramModuleView(program, stage)->metallib_bytes;
}

size_t mglRenderProgramModuleMetallibSize(const Program *program, int stage)
{
    return mglProgramModuleView(program, stage)->metallib_size;
}

void mglRenderProgramSetModuleMetallib(Program *program, int stage,
                                       unsigned char *bytes, size_t size)
{
    struct MGLShaderModuleBackend *mod = mglProgramModuleBackend(program, stage);
    mod->metallib_bytes = bytes;
    mod->metallib_size = size;
}

void mglRenderProgramSetModuleTessCaptureMetallib(Program *program, int stage,
                                                  unsigned char *bytes,
                                                  size_t size)
{
    struct MGLShaderModuleBackend *mod = mglProgramModuleBackend(program, stage);
    mod->metallib_tess_capture_bytes = bytes;
    mod->metallib_tess_capture_size = size;
}

void mglRenderProgramSetModuleCullCaptureMetallib(Program *program, int stage,
                                                  unsigned char *bytes,
                                                  size_t size)
{
    struct MGLShaderModuleBackend *mod = mglProgramModuleBackend(program, stage);
    mod->metallib_cull_capture_bytes = bytes;
    mod->metallib_cull_capture_size = size;
}

void mglRenderProgramSetModuleTesComputeMetallib(Program *program, int stage,
                                                 unsigned char *bytes,
                                                 size_t size)
{
    struct MGLShaderModuleBackend *mod = mglProgramModuleBackend(program, stage);
    mod->metallib_bytes_tes_compute = bytes;
    mod->metallib_size_tes_compute = size;
}

void mglRenderProgramSetModulePrimitiveIdVaryingMetallib(
    Program *program, int stage, unsigned char *bytes, size_t size)
{
    struct MGLShaderModuleBackend *mod = mglProgramModuleBackend(program, stage);
    mod->metallib_tess_capture_bytes = bytes;
    mod->metallib_tess_capture_size = size;
}

void mglRenderProgramSetModuleNeedsRuntimeArraySizeBuffer(Program *program,
                                                          int stage, int needs)
{
    mglProgramModuleBackend(program, stage)->needs_runtime_array_size_buffer =
        needs ? 1u : 0u;
}

/* Evaluate a GLSL constant initializer into up to 16 x 32-bit words.
 * Covers scalar / small-vector literals and constructors used as uniform
 * defaults (CTS `uniform uint id = 2;`). Returns word count, or 0. */
static uint32_t mglEvalConstUniformInit(const MGLExpr *e, uint32_t words[16],
                                        uint32_t *out_base)
{
    if (!e || !words || !out_base) {
        return 0u;
    }
    if (e->kind == MGL_EXPR_LITERAL) {
        *out_base = e->u.literal.base;
        switch (e->u.literal.base) {
        case MGL_AST_TYPE_BOOL:
        case MGL_AST_TYPE_INT: {
            GLint v = (GLint)e->u.literal.value;
            memcpy(words, &v, sizeof(v));
            return 1u;
        }
        case MGL_AST_TYPE_UINT: {
            GLuint v = (GLuint)e->u.literal.value;
            memcpy(words, &v, sizeof(v));
            return 1u;
        }
        case MGL_AST_TYPE_FLOAT: {
            GLfloat v = (GLfloat)e->u.literal.value;
            memcpy(words, &v, sizeof(v));
            return 1u;
        }
        default:
            return 0u;
        }
    }
    if (e->kind == MGL_EXPR_UNARY && e->u.unary.op == MGL_OP_SUB &&
        e->u.unary.operand) {
        uint32_t base = 0u;
        uint32_t n = mglEvalConstUniformInit(e->u.unary.operand, words, &base);
        if (n != 1u) {
            return 0u;
        }
        *out_base = base;
        if (base == MGL_AST_TYPE_FLOAT) {
            GLfloat v;
            memcpy(&v, words, sizeof(v));
            v = -v;
            memcpy(words, &v, sizeof(v));
            return 1u;
        }
        if (base == MGL_AST_TYPE_INT || base == MGL_AST_TYPE_BOOL) {
            GLint v;
            memcpy(&v, words, sizeof(v));
            v = -v;
            memcpy(words, &v, sizeof(v));
            return 1u;
        }
        return 0u;
    }
    if (e->kind == MGL_EXPR_CALL && e->u.call.name && e->u.call.arg_count > 0u &&
        e->u.call.arg_count <= 16u) {
        /* Scalar/vector constructors and array ctors: int(x), uint[8](...),
         * int[](...). */
        const char *name = e->u.call.name;
        uint32_t expect_base = 0u;
        uint32_t expect_comps = 1u;
        const int is_arr = e->u.call.is_array_ctor;
        if (strcmp(name, "int") == 0) {
            expect_base = MGL_AST_TYPE_INT;
            expect_comps = is_arr ? e->u.call.arg_count : 1u;
        } else if (strcmp(name, "uint") == 0) {
            expect_base = MGL_AST_TYPE_UINT;
            expect_comps = is_arr ? e->u.call.arg_count : 1u;
        } else if (strcmp(name, "float") == 0) {
            expect_base = MGL_AST_TYPE_FLOAT;
            expect_comps = is_arr ? e->u.call.arg_count : 1u;
        } else if (strcmp(name, "bool") == 0) {
            expect_base = MGL_AST_TYPE_BOOL;
            expect_comps = is_arr ? e->u.call.arg_count : 1u;
        } else if (strcmp(name, "ivec2") == 0 || strcmp(name, "ivec3") == 0 ||
                   strcmp(name, "ivec4") == 0) {
            expect_base = MGL_AST_TYPE_INT;
            expect_comps = (uint32_t)(name[4] - '0');
        } else if (strcmp(name, "uvec2") == 0 || strcmp(name, "uvec3") == 0 ||
                   strcmp(name, "uvec4") == 0) {
            expect_base = MGL_AST_TYPE_UINT;
            expect_comps = (uint32_t)(name[4] - '0');
        } else if (strcmp(name, "vec2") == 0 || strcmp(name, "vec3") == 0 ||
                   strcmp(name, "vec4") == 0) {
            expect_base = MGL_AST_TYPE_FLOAT;
            expect_comps = (uint32_t)(name[3] - '0');
        } else if (strncmp(name, "mat", 3) == 0 && name[3] >= '2' &&
                   name[3] <= '4') {
            expect_base = MGL_AST_TYPE_FLOAT;
            if (name[4] == '\0') {
                uint32_t d = (uint32_t)(name[3] - '0');
                expect_comps = d * d;
            } else if (name[4] == 'x' && name[5] >= '2' && name[5] <= '4' &&
                       name[6] == '\0') {
                expect_comps = (uint32_t)(name[3] - '0') *
                               (uint32_t)(name[5] - '0');
            } else {
                return 0u;
            }
        } else {
            return 0u;
        }
        if (is_arr) {
            expect_comps = e->u.call.arg_count;
        }
        if (expect_comps == 0u || expect_comps > 16u) {
            return 0u;
        }
        *out_base = expect_base;
        uint32_t woff = 0u;
        const int splat = !is_arr && e->u.call.arg_count == 1u &&
            expect_comps > 1u && strncmp(name, "mat", 3) != 0;
        for (uint32_t ai = 0u; ai < e->u.call.arg_count; ai++) {
            uint32_t ab = 0u;
            uint32_t tmp[16];
            uint32_t na = mglEvalConstUniformInit(e->u.call.args[ai], tmp, &ab);
            if (na == 0u) {
                return 0u;
            }
            if (splat) {
                if (na != 1u) {
                    return 0u;
                }
                na = expect_comps;
                for (uint32_t k = 1u; k < na; k++) {
                    tmp[k] = tmp[0];
                }
            }
            if (woff + na > expect_comps) {
                return 0u;
            }
            for (uint32_t k = 0u; k < na; k++) {
                uint32_t src = tmp[splat ? 0u : k];
                if (expect_base == MGL_AST_TYPE_FLOAT &&
                    ab != MGL_AST_TYPE_FLOAT) {
                    GLfloat fv = 0.0f;
                    if (ab == MGL_AST_TYPE_INT || ab == MGL_AST_TYPE_BOOL) {
                        GLint iv;
                        memcpy(&iv, &src, sizeof(iv));
                        fv = (GLfloat)iv;
                    } else if (ab == MGL_AST_TYPE_UINT) {
                        GLuint uv;
                        memcpy(&uv, &src, sizeof(uv));
                        fv = (GLfloat)uv;
                    } else {
                        return 0u;
                    }
                    memcpy(&words[woff], &fv, sizeof(fv));
                } else if (expect_base != MGL_AST_TYPE_FLOAT &&
                           ab == MGL_AST_TYPE_FLOAT) {
                    GLfloat fv;
                    memcpy(&fv, &src, sizeof(fv));
                    if (expect_base == MGL_AST_TYPE_UINT) {
                        GLuint uv = (GLuint)fv;
                        memcpy(&words[woff], &uv, sizeof(uv));
                    } else {
                        GLint iv = (GLint)fv;
                        memcpy(&words[woff], &iv, sizeof(iv));
                    }
                } else {
                    words[woff] = src;
                }
                if (expect_base == MGL_AST_TYPE_BOOL) {
                    GLint bv;
                    memcpy(&bv, &words[woff], sizeof(bv));
                    bv = bv ? 1 : 0;
                    memcpy(&words[woff], &bv, sizeof(bv));
                }
                woff++;
            }
            if (splat) {
                break;
            }
        }
        if (woff != expect_comps) {
            return 0u;
        }
        return expect_comps;
    }
    return 0u;
}

/* Seed plain-uniform CPU slots from GLSL default initializers so the first
 * draw sees defaults without SSA-folding them (which would ignore glUniform*). */
void mglSeedUniformInitializers(GLMContext ctx, Program *pptr)
{
    if (!ctx || !pptr) {
        return;
    }
    Program *prev_prog = STATE(program);
    GLuint prev_name = STATE(program_name);
    STATE(program) = pptr;
    STATE(program_name) = pptr->name;

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Shader *shader = pptr->shader_slots[stage];
        if (!shader || !shader->src) {
            continue;
        }
        MGLTranslationUnit *tu = shader->frontend_tu;
        if (!tu) {
            continue;
        }
        for (uint32_t di = 0; di < tu->decl_count; di++) {
            for (MGLDecl *d = tu->decls[di]; d; d = d->next_declarator) {
                if (!d->name || !d->init || d->body) {
                    continue;
                }
                if (!(d->qualifiers & MGL_AST_Q_UNIFORM) ||
                    (d->qualifiers & MGL_AST_Q_BUFFER) ||
                    (d->qualifiers & MGL_AST_Q_CONST)) {
                    continue;
                }
                /* Skip interface blocks / opaque types. */
                if (d->struct_members ||
                    (d->type && (d->type->base == MGL_AST_TYPE_SAMPLER ||
                                 d->type->base == MGL_AST_TYPE_IMAGE ||
                                 d->type->base == MGL_AST_TYPE_ATOMIC_UINT ||
                                 d->type->base == MGL_AST_TYPE_STRUCT))) {
                    continue;
                }
                GLint loc = mglGetUniformLocation(ctx, pptr->name, d->name);
                if (loc < 0) {
                    continue;
                }
                uint32_t words_local[16];
                uint32_t *words = words_local;
                uint32_t *words_heap = NULL;
                uint32_t base = 0u;
                uint32_t n = 0u;
                const int is_init_list = (d->init->kind == MGL_EXPR_INIT_LIST);
                const int is_arr_ctor =
                    (d->init->kind == MGL_EXPR_CALL &&
                     d->init->u.call.is_array_ctor);
                if (is_init_list || is_arr_ctor) {
                    uint32_t argc = is_init_list
                        ? d->init->u.init_list.arg_count
                        : d->init->u.call.arg_count;
                    MGLExpr **args = is_init_list
                        ? d->init->u.init_list.args
                        : d->init->u.call.args;
                    if (argc == 0u || !args) {
                        continue;
                    }
                    uint32_t elem_comps = 1u;
                    if (d->type && d->type->vec_size > 1)
                        elem_comps = (uint32_t)d->type->vec_size;
                    uint32_t cap = argc * elem_comps;
                    if (cap < argc)
                        cap = argc;
                    if (cap > 16u) {
                        words_heap = (uint32_t *)calloc(cap, sizeof(uint32_t));
                        if (!words_heap) {
                            continue;
                        }
                        words = words_heap;
                    }
                    int ok = 1;
                    uint32_t want = (d->type && d->type->base)
                        ? d->type->base : 0u;
                    uint32_t woff = 0u;
                    for (uint32_t ei = 0u; ei < argc; ei++) {
                        uint32_t tmp[16];
                        uint32_t ab = 0u;
                        uint32_t nw = mglEvalConstUniformInit(args[ei], tmp,
                                                              &ab);
                        if (nw == 0u || woff + nw > cap) {
                            ok = 0;
                            break;
                        }
                        if (ei == 0u) {
                            base = want ? want : ab;
                        }
                        for (uint32_t k = 0u; k < nw; k++) {
                            if (want == MGL_AST_TYPE_FLOAT &&
                                ab != MGL_AST_TYPE_FLOAT) {
                                GLfloat fv = 0.0f;
                                if (ab == MGL_AST_TYPE_INT ||
                                    ab == MGL_AST_TYPE_BOOL) {
                                    GLint iv;
                                    memcpy(&iv, tmp + k, sizeof(iv));
                                    fv = (GLfloat)iv;
                                } else if (ab == MGL_AST_TYPE_UINT) {
                                    GLuint uv;
                                    memcpy(&uv, tmp + k, sizeof(uv));
                                    fv = (GLfloat)uv;
                                }
                                memcpy(&words[woff], &fv, sizeof(fv));
                            } else {
                                words[woff] = tmp[k];
                            }
                            woff++;
                        }
                    }
                    if (!ok) {
                        free(words_heap);
                        continue;
                    }
                    n = woff;
                    if (want) {
                        base = want;
                    }
                } else {
                    n = mglEvalConstUniformInit(d->init, words, &base);
                }
                if (n == 0u) {
                    static int s_seedFailLogged;
                    if (!s_seedFailLogged) {
                        fprintf(stderr,
                                "MGL WARNING: uniform initializer seeding "
                                "failed (first: '%s' program=%u); defaults "
                                "may be zero\n",
                                d->name, (unsigned)pptr->name);
                        s_seedFailLogged = 1;
                    }
                    free(words_heap);
                    continue;
                }
                const int is_array = (d->array_count > 0u) ||
                    (d->init && d->init->kind == MGL_EXPR_INIT_LIST) ||
                    (d->init && d->init->kind == MGL_EXPR_CALL &&
                     d->init->u.call.is_array_ctor);
                switch (base) {
                case MGL_AST_TYPE_BOOL:
                case MGL_AST_TYPE_INT:
                    if (is_array || n > 4u) {
                        mglUniform1iv(ctx, loc, (GLsizei)n, (const GLint *)words);
                    } else if (n == 1u) {
                        GLint v;
                        memcpy(&v, words, sizeof(v));
                        mglUniform1i(ctx, loc, v);
                    } else if (n == 2u) {
                        mglUniform2iv(ctx, loc, 1, (const GLint *)words);
                    } else if (n == 3u) {
                        mglUniform3iv(ctx, loc, 1, (const GLint *)words);
                    } else if (n == 4u) {
                        mglUniform4iv(ctx, loc, 1, (const GLint *)words);
                    }
                    break;
                case MGL_AST_TYPE_UINT:
                    if (is_array || n > 4u) {
                        mglUniform1uiv(ctx, loc, (GLsizei)n,
                                       (const GLuint *)words);
                    } else if (n == 1u) {
                        GLuint v;
                        memcpy(&v, words, sizeof(v));
                        mglUniform1ui(ctx, loc, v);
                    } else if (n == 2u) {
                        mglUniform2uiv(ctx, loc, 1, (const GLuint *)words);
                    } else if (n == 3u) {
                        mglUniform3uiv(ctx, loc, 1, (const GLuint *)words);
                    } else if (n == 4u) {
                        mglUniform4uiv(ctx, loc, 1, (const GLuint *)words);
                    }
                    break;
                case MGL_AST_TYPE_FLOAT:
                    if (is_array || n > 4u) {
                        mglUniform1fv(ctx, loc, (GLsizei)n,
                                      (const GLfloat *)words);
                    } else if (n == 1u) {
                        GLfloat v;
                        memcpy(&v, words, sizeof(v));
                        mglUniform1f(ctx, loc, v);
                    } else if (n == 2u) {
                        mglUniform2fv(ctx, loc, 1, (const GLfloat *)words);
                    } else if (n == 3u) {
                        mglUniform3fv(ctx, loc, 1, (const GLfloat *)words);
                    } else if (n == 4u) {
                        mglUniform4fv(ctx, loc, 1, (const GLfloat *)words);
                    }
                    break;
                default:
                    break;
                }
                free(words_heap);
            }
        }
    }

    STATE(program) = prev_prog;
    STATE(program_name) = prev_name;
}
