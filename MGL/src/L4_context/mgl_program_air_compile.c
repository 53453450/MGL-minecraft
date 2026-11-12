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
#include <limits.h>
#include <stdatomic.h>
#include <ctype.h>
#include <malloc/malloc.h>
#include <CoreFoundation/CoreFoundation.h>
#include "mgl_shader_abi.h"
#include "mgl.h"

#include "glm_context.h"
#include "shaders.h"
#include "buffers.h"
#include "mgl_safety.h"
#include "mgl_buffer_slots.h"
#include "mgl_uniform_reflection.h"
#include "mgl_program_reflection.h"
#include "mgl_program_resource.h"
#include "mgl_sampler_compat.h"
#include "mgl_buffer_plan.h"
#include "mgl_shader_resource.h"
#include "mgl_compile_artifact.h"
#include "mgl_compile_plan.h"
#include "mgl_air_reflect.h"
#include "mgl_render_api_binding.h"
#include "mgl_env_flag.h"
#include "mgl_render_api_program.h"

#include "mgl_trace_log.h"
#include <stdbool.h>


/* Protos for helpers defined in mgl_program_link.c */
Program *findProgram(GLMContext ctx, GLuint program); GLuint mglProgramAttachedShaderCount(Program *program, GLuint stage); GLboolean mglProgramHasAttachedShader(Program *program, GLuint stage, Shader *shader); GLuint mglTransformFeedbackTypeComponents(GLuint type); bool mglTransformFeedbackBaseName(const char *name, char out[96]); int mglCompileCaptureVariant(const char *src, const char *const *attrib_names, uint32_t flags, unsigned char **bytes, size_t *size, char *err, size_t err_cap); const char *mglSkipGLSLVersionDirective(const char *s); char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
GLuint mglProgramAttachedShaderCount(Program *program, GLuint stage); GLboolean mglProgramHasAttachedShader(Program *program, GLuint stage, Shader *shader); GLuint mglTransformFeedbackTypeComponents(GLuint type); bool mglTransformFeedbackBaseName(const char *name, char out[96]); int mglCompileCaptureVariant(const char *src, const char *const *attrib_names, uint32_t flags, unsigned char **bytes, size_t *size, char *err, size_t err_cap); const char *mglSkipGLSLVersionDirective(const char *s); char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
GLboolean mglProgramHasAttachedShader(Program *program, GLuint stage, Shader *shader); GLuint mglTransformFeedbackTypeComponents(GLuint type); bool mglTransformFeedbackBaseName(const char *name, char out[96]); int mglCompileCaptureVariant(const char *src, const char *const *attrib_names, uint32_t flags, unsigned char **bytes, size_t *size, char *err, size_t err_cap); const char *mglSkipGLSLVersionDirective(const char *s); char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
GLuint mglTransformFeedbackTypeComponents(GLuint type); bool mglTransformFeedbackBaseName(const char *name, char out[96]); int mglCompileCaptureVariant(const char *src, const char *const *attrib_names, uint32_t flags, unsigned char **bytes, size_t *size, char *err, size_t err_cap); const char *mglSkipGLSLVersionDirective(const char *s); char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
bool mglTransformFeedbackBaseName(const char *name, char out[96]); int mglCompileCaptureVariant(const char *src, const char *const *attrib_names, uint32_t flags, unsigned char **bytes, size_t *size, char *err, size_t err_cap); const char *mglSkipGLSLVersionDirective(const char *s); char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
int mglCompileCaptureVariant(const char *src, const char *const *attrib_names, uint32_t flags, unsigned char **bytes, size_t *size, char *err, size_t err_cap); const char *mglSkipGLSLVersionDirective(const char *s); char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
const char *mglSkipGLSLVersionDirective(const char *s); char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
char *mglConcatSameStageSources(Program *pptr, int stage); void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
void mglCopyExplicitBinding(MGLShaderResource *dst, const MGLShaderResource *src); void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;
void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count, const char *name, GLuint gl_type, GLint array_size, GLboolean is_array); /* A16: keep the last successful executable until a new link commits. */ typedef struct MGLSavedLinkExecutable;

int mglAirCompileStage(GLMContext ctx, Program *pptr, int stage)
{
    Shader *shader = pptr->shader_slots[stage];
    if (!shader || !shader->src) {
        return 1;
    }
    /* Map the legacy stage numbering (VS=0, TCS=1, TES=2, GS=3, FS=4,
     * CS=5) to the AIR ABI stages (VS=0, FS=1, CS=2, TCS/TES/GS M3). */
    int air_stage;
    switch (stage) {
    case _VERTEX_SHADER:        air_stage = MGL_STAGE_VERTEX; break;
    case _FRAGMENT_SHADER:      air_stage = MGL_STAGE_FRAGMENT; break;
    case _COMPUTE_SHADER:       air_stage = MGL_STAGE_COMPUTE; break;
    case _TESS_CONTROL_SHADER:  air_stage = MGL_STAGE_TESS_CONTROL; break;
    case _TESS_EVALUATION_SHADER: air_stage = MGL_STAGE_TESS_EVALUATION; break;
    case _GEOMETRY_SHADER:      air_stage = MGL_STAGE_GEOMETRY; break;
    default:
        return 1;   /* unsupported stage: skip (link continues) */
    }
    clearStageCompileState(pptr, stage);
    GLuint attached_count = mglProgramAttachedShaderCount(pptr, (GLuint)stage);
    char *merged_src = NULL;
    const char *compile_src = shader->src;
    if (attached_count > 1u) {
        merged_src = mglConcatSameStageSources(pptr, stage);
        if (!merged_src) {
            return 0;
        }
        compile_src = merged_src;
    }
    unsigned char *bytes = NULL;
    size_t size = 0;
    char err[512] = {0};
    /* Snapshot the attribute bindings: the source strings are owned by
     * this program and could be released between stage compiles. */
    const char *attrib_snapshot[MAX_ATTRIBS] = {NULL};
    for (int ai = 0; ai < MAX_ATTRIBS; ai++) {
        if (pptr->attrib_location_names[ai]) {
            attrib_snapshot[ai] = strdup(pptr->attrib_location_names[ai]);
        }
    }
    /* GLSL 4.0 subroutines: capture the frontend's tables for this stage so
     * glGetSubroutine* answers from the program instead of re-parsing.  The
     * substring test keeps the extra parse off the common path. */
    memset(&pptr->subroutine_tables[stage], 0,
           sizeof(pptr->subroutine_tables[stage]));
    if (compile_src && strstr(compile_src, "subroutine") != NULL) {
        MGLAIRStageInfo sub_info = {0};
        char sub_err[256] = {0};
        if (mglAirReflectGLSLStageInfo(compile_src, air_stage, &sub_info,
                                       sub_err, sizeof sub_err) == 0) {
            pptr->subroutine_tables[stage] = sub_info.subroutines;
            /* GL 4.6 §7.10: subroutine-uniform locations are unique within a
             * stage and must be below GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS.
             * Both are link errors; without these checks the program linked
             * and the second selection silently aliased the first. */
            if (!mglSubroutineValidateTable(
                    &pptr->subroutine_tables[stage],
                    1024u,      /* GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS */
                    sub_err, sizeof sub_err)) {
                fprintf(stderr, "MGL DIAG subroutine table validate failed "
                        "program=%u stage=%d err=[%s]\n",
                        (unsigned)pptr->name, stage, sub_err);
                free(merged_src);
                return 0;
            }
        } else {
            fprintf(stderr, "MGL DIAG subroutine reflect failed program=%u "
                    "stage=%d err=[%s]\n", (unsigned)pptr->name, stage, sub_err);
        }
    }
    MGLAIRStageInfo stage_info = {0};
    if (stage == _GEOMETRY_SHADER &&
        mglAirReflectGLSLStageInfo(shader->src, air_stage, &stage_info,
                                   err, sizeof err) == 0) {
        pptr->geometry_input_type = stage_info.geometry_input_type;
        pptr->geometry_output_type = stage_info.geometry_output_type;
        pptr->geometry_vertices_out = stage_info.geometry_vertices_out;
        pptr->geometry_max_vertices_specified =
            stage_info.geometry_max_vertices_specified ? GL_TRUE : GL_FALSE;
        pptr->geometry_invocations = stage_info.geometry_invocations;
        pptr->geometry_stream_count = stage_info.gs_stream_count;
        for (uint32_t si = 0; si < 4u; si++) {
            pptr->geometry_stream_varying_count[si] =
                stage_info.gs_stream_varying_count[si];
            pptr->geometry_stream_xfb_stride[si] =
                stage_info.gs_stream_xfb_stride[si];
        }
    }
    if (stage == _TESS_EVALUATION_SHADER) {
        /* The TES metallib TESS tag must carry the control points per
         * patch (4*cpc + patchKind); Metal uses that compile-time value to
         * compute the per-patch control-point offset, so a 0 here makes
         * every patch read record 0.  Prefer the TCS output vertices when
         * present, else the GL default patch size (3). */
        stage_info.tess_patch_vertices =
            pptr->tess_control_output_vertices > 0u
                ? pptr->tess_control_output_vertices
                : 3u;
    }
    MGLCompilePlan plan;
    mglCompilePlanInit(&plan, compile_src, air_stage, attrib_snapshot);
    plan.tess_patch_vertices = stage_info.tess_patch_vertices;
    if (attached_count > 1u) {
        plan.air_flags |= MGL_AIR_COMPILE_STAGE_LINK_MERGE;
    }
    mglCompilePlanApplyRasterVariant(
        &plan, stage, pptr->shader_slots[_GEOMETRY_SHADER] != NULL,
        pptr->program_separable != GL_FALSE,
        pptr->tess_eval_compute != GL_FALSE,
        pptr->tess_eval_render_vertex != GL_FALSE,
        ctx && ctx->active_state &&
            STATE(var).clip_depth_mode == GL_ZERO_TO_ONE,
        pptr->clip_distance_count);
    /* Native post-tessellation feeds FS directly: it cannot insert a GS
     * between TES and FS, and cannot feed transform feedback.  Force the
     * compute expansion path for triangles/quads (same as isolines /
     * point_mode) whenever either follows. */
    if (stage == _TESS_EVALUATION_SHADER &&
        (pptr->transform_feedback_varying_count > 0 ||
         pptr->shader_slots[_GEOMETRY_SHADER])) {
        plan.air_flags |= MGL_AIR_COMPILE_FORCE_TES_COMPUTE;
    }
    /* Isolines / point_mode without XFB or a GS rasterize the expanded
     * stream directly through a render vertex function.  The default-on
     * MGL_TES_VERTEX_RENDER gate lets the draw path fall back to the
     * compute expansion for risk containment and A/B comparison.
     *
     * This must be decidable BEFORE the TES is compiled: the flag is an
     * input to the TES codegen, while pptr->tess_eval_render_vertex is an
     * output of it (set from stage_info below).  Gating on the field here
     * made the flag always absent on the pass that produced the function,
     * so the TES fell back to the compute kernel while the draw path still
     * routed to EXEC_TES_VERTEX and fed that kernel to a render PSO.
     * primitive mode / point_mode are re-checked by the backend from the
     * parsed TU (mgl_air_backend.cpp isTESVertex), so the two conditions
     * that are only known post-parse stay owned by codegen. */
    if (stage == _TESS_EVALUATION_SHADER &&
        pptr->transform_feedback_varying_count == 0 &&
        !pptr->shader_slots[_GEOMETRY_SHADER] &&
        mgl_env_flag_enabled_default_on("MGL_TES_VERTEX_RENDER")) {
        plan.air_flags |= MGL_AIR_COMPILE_TES_VERTEX;
    }
    /* TES-compute and TES-vertex (isolines / point_mode) both tag outs as
     * mgl_loc_N — same ABI as the GS passthrough.  Without this flag the FS
     * keeps name tags and Metal rejects the pipeline
     * (point_rendering: result_color).  Stages compile VS→…→TES→GS→FS, so
     * both tess_eval_compute and tess_eval_render_vertex are already known
     * when the FS is compiled.
     *
     * Separable FS/VS programs link alone, so the monolithic GS/TES probe
     * above never fires — yet SSO draws still rasterize through the GS
     * passthrough (or TES-vertex) which emits mgl_loc_N.  Tag separable
     * VS outs and FS ins the same way so Metal can pair them
     * (shader_subroutine.subroutines_with_separate_shader_objects). */
    /* When a GS is present, FS mgl_loc_N tags must follow GS output
     * locations (passthrough VS).  Remap by name at compile time.
     * TES-compute / TES-vertex (no GS) remap FS inputs to TES outs the same
     * way.  Same for TES←TCS: declaration-order locations diverge when the
     * TES omits some TCS per-vertex outs (barrier_guarded_read_calls).
     * TCS←VS: CTS shaders often declare TCS inputs in a different order
     * than VS outs (vsColor before vsPosition); pair by name (GL §7.4.1). */
    const MGLShaderResourceList *iface_peers = NULL;
    if (stage == _FRAGMENT_SHADER && pptr->shader_slots[_GEOMETRY_SHADER]) {
        iface_peers =
            &pptr->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES];
    } else if (stage == _FRAGMENT_SHADER &&
               (pptr->tess_eval_compute || pptr->tess_eval_render_vertex) &&
               pptr->shader_slots[_TESS_EVALUATION_SHADER]) {
        iface_peers =
            &pptr->shader_resources_list[_TESS_EVALUATION_SHADER]
                                       [_STAGE_OUTPUT_RES];
    } else if (stage == _TESS_EVALUATION_SHADER &&
               pptr->shader_slots[_TESS_CONTROL_SHADER]) {
        iface_peers =
            &pptr->shader_resources_list[_TESS_CONTROL_SHADER][_STAGE_OUTPUT_RES];
    } else if (stage == _TESS_CONTROL_SHADER &&
               pptr->shader_slots[_VERTEX_SHADER]) {
        iface_peers =
            &pptr->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES];
    }
    plan.iface_peers = iface_peers;
    int air_rc;
    if (mglCompileArtifactCanReuseAtLink(shader->cached_artifact, plan.stage,
                                         plan.air_flags, plan.iface_peers,
                                         plan.attrib_names)) {
        /* R2: adopt glCompileShader CompileArtifact — skip second parse/codegen. */
        MGLCompileArtifact *cached = shader->cached_artifact;
        bytes = cached->metallib_bytes;
        size = cached->metallib_size;
        cached->metallib_bytes = NULL;
        cached->metallib_size = 0;
        memcpy(pptr->shader_resources_list[stage], cached->resources,
               sizeof(cached->resources));
        memset(cached->resources, 0, sizeof(cached->resources));
        stage_info = cached->stage_info;
        cached->complete = 0;
        mglCompileArtifactFree(shader->cached_artifact);
        shader->cached_artifact = NULL;
        mglCompileFrontendNoteReuse();
        air_rc = 0;
    } else {
        /* Every stage, including variant flags/peers, publishes through
         * CompileArtifact.complete so link never binds a half-built executable. */
        MGLCompileArtifact art;
        mglCompileArtifactInit(&art);
        air_rc = mglCompileArtifactFromPlan(&plan, &art, err, sizeof err);
        if (air_rc == 0 && art.complete) {
            bytes = art.metallib_bytes;
            size = art.metallib_size;
            art.metallib_bytes = NULL;
            art.metallib_size = 0;
            memcpy(pptr->shader_resources_list[stage], art.resources,
                   sizeof(art.resources));
            memset(art.resources, 0, sizeof(art.resources));
            stage_info = art.stage_info;
            mglShaderReplaceFrontendTU(shader, art.tu);
            art.tu = NULL;
            mglCompileArtifactDestroy(&art);
            if (shader->frontend_valid &&
                shader->frontend_stage == air_stage &&
                plan.air_flags == 0u && iface_peers == NULL) {
                mglCompileFrontendNoteReuse();
            }
        } else {
            mglCompileArtifactDestroy(&art);
            air_rc = -1;
        }
    }
    if (air_rc == 0 && stage_info.needs_sparse_residency) {
        mglRenderProgramSetModuleNeedsRuntimeArraySizeBuffer(
            pptr, stage, stage_info.needs_runtime_array_size_buffer ? 1 : 0);
        GLuint sparse_slot = 0u;
        if (!mglBufferSlotFindSparseResidency(
                pptr, stage, plan.sparse_residency_buffer_index,
                &sparse_slot)) {
            snprintf(err, sizeof err,
                     "no free Metal buffer slot for sparse residency table");
            air_rc = -1;
        } else if (sparse_slot != plan.sparse_residency_buffer_index) {
            MGLCompileArtifact *relocated = mglCompileArtifactCreate();
            plan.sparse_residency_buffer_index = sparse_slot;
            if (!relocated ||
                mglCompileArtifactFromPlan(&plan, relocated, err,
                                           sizeof err) != 0 ||
                !relocated->complete) {
                mglCompileArtifactFree(relocated);
                if (err[0] == '\0') {
                    snprintf(err, sizeof err,
                             "failed to relocate sparse residency table");
                }
                air_rc = -1;
            } else {
                mglShaderFree(bytes);
                bytes = relocated->metallib_bytes;
                size = relocated->metallib_size;
                relocated->metallib_bytes = NULL;
                relocated->metallib_size = 0;
                mglAirReflectDestroy(pptr->shader_resources_list[stage]);
                memset(pptr->shader_resources_list[stage], 0,
                       sizeof(pptr->shader_resources_list[stage]));
                memcpy(pptr->shader_resources_list[stage], relocated->resources,
                       sizeof(relocated->resources));
                memset(relocated->resources, 0, sizeof(relocated->resources));
                stage_info = relocated->stage_info;
                if (relocated->tu) {
                    mglShaderReplaceFrontendTU(shader, relocated->tu);
                    relocated->tu = NULL;
                }
                mglCompileArtifactFree(relocated);
            }
        }
    }
    if (getenv("MGL_DUMP_SRC") && compile_src) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/mgl_src_p%u_s%d.glsl",
                 (unsigned)pptr->name, stage);
        FILE *fp = fopen(path, "wb");
        if (fp) { fwrite(compile_src, 1, strlen(compile_src), fp); fclose(fp); }
    }
    if (air_rc != 0) {
        mglShaderFree(bytes);
        bytes = NULL;
        for (int ai = 0; ai < MAX_ATTRIBS; ai++) {
            free((void *)attrib_snapshot[ai]);
        }
        fprintf(stderr,
                "MGL WARNING: AIR compile failed program %u stage %d: %s\n",
                pptr->name, stage, err);
        /* MGL_DUMP_FAILED_SRC=1 leaves the exact source that failed next to the
         * message; reproducing a codegen error offline needs the text, and the
         * CTS side only has the test's builder. */
        if (getenv("MGL_DUMP_FAILED_SRC") && compile_src) {
            char path[256];
            snprintf(path, sizeof(path), "/tmp/mgl_failed_p%u_s%d.glsl",
                     (unsigned)pptr->name, stage);
            FILE *fp = fopen(path, "wb");
            if (fp) {
                fwrite(compile_src, 1, strlen(compile_src), fp);
                fclose(fp);
                fprintf(stderr, "MGL WARNING: failed source written to %s\n", path);
            }
        }
        free(merged_src);
        return 0;
    }
    /* Publish this stage's exact builtin usage (the reflected resource lists
     * filter every gl_ builtin out, so callers used to scan the source). */
    pptr->air_builtin_mask[stage] = stage_info.builtin_mask;

    /* Capture variants must see the same bindAttribLocation map as the
     * reflected VS, or sparse locations (CTS enable_disable) misalign
     * [[attribute(N)]] with the vertex descriptor. */
    if (stage == _VERTEX_SHADER) {
        pptr->uses_cull_distance = stage_info.uses_cull_distance
            ? GL_TRUE : GL_FALSE;
        pptr->cull_distance_count = stage_info.cull_distance_count;
        if (stage_info.clip_distance_count > pptr->clip_distance_count)
            pptr->clip_distance_count = stage_info.clip_distance_count;
        pptr->ir_uses_cull_distance = pptr->uses_cull_distance;
        unsigned char *capture_bytes = NULL;
        size_t capture_size = 0;
        char capture_err[512] = {0};
        if (mglCompileCaptureVariant(
                shader->src, attrib_snapshot, MGL_AIR_COMPILE_TESS_CAPTURE,
                &capture_bytes, &capture_size, capture_err,
                sizeof capture_err) == 0) {
            mglRenderProgramSetModuleTessCaptureMetallib(pptr, stage,
                                                         capture_bytes,
                                                         capture_size);
            if (getenv("MGL_DUMP_AIR")) {
                FILE *f = fopen("/tmp/poison_capvar.air", "wb");
                if (f) {
                    fwrite(capture_bytes, 1, capture_size, f);
                    fclose(f);
                    FILE *fsrc =
                        fopen("/tmp/poison_capvar_src.glsl", "wb");
                    if (fsrc) {
                        fwrite(shader->src, 1, strlen(shader->src), fsrc);
                        fclose(fsrc);
                    }
                    fprintf(stderr,
                            "MGL DUMP: capvar.air %zu bytes\n",
                            capture_size);
                }
            }
        } else {
            fprintf(stderr,
                    "MGL WARNING: AIR tess VS capture compile failed "
                    "program %u: %s\n",
                    pptr->name, capture_err);
        }
        if (stage_info.uses_cull_distance) {
            unsigned char *cull_capture_bytes = NULL;
            size_t cull_capture_size = 0;
            char cull_capture_err[512] = {0};
            if (mglCompileCaptureVariant(
                    shader->src, attrib_snapshot, MGL_AIR_COMPILE_CULL_CAPTURE,
                    &cull_capture_bytes, &cull_capture_size, cull_capture_err,
                    sizeof cull_capture_err) == 0) {
                mglRenderProgramSetModuleCullCaptureMetallib(
                    pptr, stage, cull_capture_bytes, cull_capture_size);
                if (getenv("MGL_DUMP_AIR")) {
                    FILE *f = fopen("/tmp/cullcap.air", "wb");
                    if (f) {
                        fwrite(cull_capture_bytes, 1, cull_capture_size, f);
                        fclose(f);
                    }
                }
            } else {
                fprintf(stderr,
                        "MGL WARNING: AIR cull-distance capture compile failed "
                        "program %u: %s\n",
                        pptr->name, cull_capture_err);
            }
        }
    }
    /* TES render-vertex / compute dual emit: when the TES is an isolines or
     * point_mode program eligible for the render-vertex path, also emit the
     * compute expansion kernel so indexed draws (glDrawElements /
     * glMultiDrawElements) can fall back to the verified compute path, which
     * carries the gather ABI the render-vertex ABI lacks.  metallib_bytes
     * above stays the render-vertex function; the kernel lives in
     * metallib_bytes_tes_compute.  attrib_snapshot is still valid here
     * (freed just below). */
    if (stage == _TESS_EVALUATION_SHADER &&
        pptr->transform_feedback_varying_count == 0 &&
        !pptr->shader_slots[_GEOMETRY_SHADER] &&
        mgl_env_flag_enabled_default_on("MGL_TES_VERTEX_RENDER")) {
        MGLCompilePlan compute_plan = plan;
        compute_plan.air_flags &= (uint32_t)(~MGL_AIR_COMPILE_TES_VERTEX);
        MGLCompileArtifact compute_art;
        mglCompileArtifactInit(&compute_art);
        char compute_err[512] = {0};
        int compute_rc = mglCompileArtifactFromPlan(
            &compute_plan, &compute_art, compute_err, sizeof compute_err);
        if (compute_rc == 0 && compute_art.complete) {
            mglRenderProgramSetModuleTesComputeMetallib(
                pptr, stage, compute_art.metallib_bytes,
                compute_art.metallib_size);
            compute_art.metallib_bytes = NULL;
            compute_art.metallib_size = 0;
            mglCompileArtifactDestroy(&compute_art);
        } else {
            mglCompileArtifactDestroy(&compute_art);
            fprintf(stderr,
                    "MGL WARNING: TES compute-kernel emit failed program %u: %s\n",
                    pptr->name, compute_err);
        }
    }

    if (stage == _FRAGMENT_SHADER &&
        (plan.air_flags & MGL_AIR_COMPILE_DIRECT_PRIMITIVE_ID) != 0u &&
        (stage_info.builtin_mask & MGL_AIR_BUILTIN_PRIMITIVE_ID) != 0u) {
        /* A separable FS can later share a pipeline with a GS or a TES
         * expansion.  Direct draws use the native primitive-id metallib above;
         * expanded draws still need the location-32 float carrier. */
        MGLCompilePlan varying_plan = plan;
        varying_plan.air_flags &=
            (uint32_t)(~MGL_AIR_COMPILE_DIRECT_PRIMITIVE_ID);
        MGLCompileArtifact varying_art;
        mglCompileArtifactInit(&varying_art);
        char varying_err[512] = {0};
        int varying_rc = mglCompileArtifactFromPlan(
            &varying_plan, &varying_art, varying_err, sizeof varying_err);
        if (varying_rc == 0 && varying_art.complete) {
            mglRenderProgramSetModulePrimitiveIdVaryingMetallib(
                pptr, stage, varying_art.metallib_bytes,
                varying_art.metallib_size);
            varying_art.metallib_bytes = NULL;
            varying_art.metallib_size = 0;
            mglCompileArtifactDestroy(&varying_art);
        } else {
            mglCompileArtifactDestroy(&varying_art);
            fprintf(stderr,
                    "MGL WARNING: FS primitive-id carrier emit failed "
                    "program %u: %s\n",
                    pptr->name, varying_err);
        }
    }

    for (int ai = 0; ai < MAX_ATTRIBS; ai++) {
        free((void *)attrib_snapshot[ai]);
    }
    if (stage == _COMPUTE_SHADER) {
        pptr->local_workgroup_size.x = stage_info.compute_local_size_x
                                           ? stage_info.compute_local_size_x
                                           : 1u;
        pptr->local_workgroup_size.y = stage_info.compute_local_size_y
                                           ? stage_info.compute_local_size_y
                                           : 1u;
        pptr->local_workgroup_size.z = stage_info.compute_local_size_z
                                           ? stage_info.compute_local_size_z
                                           : 1u;
        pptr->shared_memory_bytes = stage_info.shared_memory_bytes;
    }
    mglRenderProgramSetModuleMetallib(pptr, stage, bytes, size);
    /* Second metallib with software sampler reduction for VS/FS that
     * actually sample. Bound only when a MIN/MAX reduction sampler is
     * active this draw; skip emit when the stage has no samplers so
     * ordinary links do not pay an extra frontend parse. */
    if ((stage == _VERTEX_SHADER || stage == _FRAGMENT_SHADER) &&
        mglProgramStageHasSamplers(pptr, stage)) {
        MGLCompilePlan red_plan = plan;
        red_plan.air_flags |= MGL_AIR_COMPILE_SOFTWARE_SAMPLER_REDUCTION;
        MGLCompileArtifact red_art;
        mglCompileArtifactInit(&red_art);
        char red_err[512];
        red_err[0] = '\0';
        if (mglCompileArtifactFromPlan(&red_plan, &red_art, red_err,
                                       sizeof red_err) == 0 &&
            red_art.complete) {
            mglRenderProgramSetModuleSamplerReductionMetallib(
                pptr, stage, red_art.metallib_bytes, red_art.metallib_size);
            red_art.metallib_bytes = NULL;
            red_art.metallib_size = 0;
            mglCompileArtifactDestroy(&red_art);
        } else {
            mglCompileArtifactDestroy(&red_art);
            fprintf(stderr,
                    "MGL WARNING: sampler-reduction metallib emit failed "
                    "program %u stage %d: %s\n",
                    pptr->name, (int)stage, red_err);
        }
    }
    {
        /* MGL_DUMP_SHADER_DIR=<dir>: keep every linked stage's source,
         * metallib and its LLVM IR as prog<name>_stage<N>.{glsl,metallib,ll},
         * so a pipeline the log names by program number can be reproduced. */
        const char *dump_dir = getenv("MGL_DUMP_SHADER_DIR");
        if (dump_dir && dump_dir[0]) {
            char path[1024];
            snprintf(path, sizeof path, "%s/prog%u_stage%d.glsl", dump_dir,
                     (unsigned)pptr->name, (int)stage);
            FILE *f = fopen(path, "wb");
            if (f) {
                fwrite(shader->src, 1, strlen(shader->src), f);
                fclose(f);
            }
            snprintf(path, sizeof path, "%s/prog%u_stage%d.ll", dump_dir,
                     (unsigned)pptr->name, (int)stage);
            char *ir_text = mglAirMetallibToIRText(bytes, size);
            if (ir_text) {
                f = fopen(path, "w");
                if (f) {
                    fputs(ir_text, f);
                    fclose(f);
                }
                mglShaderFree(ir_text);
            }
            snprintf(path, sizeof path, "%s/prog%u_attribs.txt", dump_dir,
                     (unsigned)pptr->name);
            f = fopen(path, "w");
            if (f) {
                for (int ai = 0; ai < MAX_ATTRIBS; ai++) {
                    if (pptr->attrib_location_names[ai]) {
                        fprintf(f, "%d %s\n", ai,
                                pptr->attrib_location_names[ai]);
                    }
                }
                fclose(f);
            }
            snprintf(path, sizeof path, "%s/prog%u_stage%d.metallib",
                     dump_dir, (unsigned)pptr->name, (int)stage);
            f = fopen(path, "wb");
            if (f) {
                fwrite(bytes, 1, size, f);
                fclose(f);
            }
        }
    }
    if (getenv("MGL_DUMP_AIR") && stage == _GEOMETRY_SHADER) {
        FILE *f = fopen("/tmp/poison_gs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: gs.air %zu bytes\n", size);
        }
    }
    if (getenv("MGL_DUMP_AIR") && stage == _VERTEX_SHADER) {
        FILE *f = fopen("/tmp/poison_vs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: vs.air %zu bytes\n", size);
        }
    }
    if (getenv("MGL_DUMP_AIR") && stage == _FRAGMENT_SHADER) {
        FILE *f = fopen("/tmp/poison_fs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: fs.air %zu bytes\n", size);
        }
    }
    if (getenv("MGL_DUMP_AIR") && stage == _COMPUTE_SHADER) {
        FILE *f = fopen("/tmp/poison_cs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: cs.air %zu bytes\n", size);
        }
    }
    mglRenderProgramSetModuleNeedsRuntimeArraySizeBuffer(
        pptr, stage,
        stage_info.needs_runtime_array_size_buffer ? 1 : 0);
    mglRenderProgramSetModuleNeedsSparseResidency(
        pptr, stage, stage_info.needs_sparse_residency ? 1 : 0,
        stage_info.sparse_residency_buffer_index);
    if (pptr->modules[stage].entry_point) {
        free(pptr->modules[stage].entry_point);
    }
    pptr->modules[stage].entry_point = strdup("main");
    if (stage == _TESS_CONTROL_SHADER) {
        pptr->tess_control_output_vertices =
            stage_info.tess_control_output_vertices;
    } else if (stage == _TESS_EVALUATION_SHADER) {
        pptr->tess_gen_mode = stage_info.tess_gen_mode;
        pptr->tess_gen_spacing = stage_info.tess_gen_spacing;
        pptr->tess_gen_vertex_order = stage_info.tess_gen_vertex_order;
        pptr->tess_gen_point_mode =
            stage_info.tess_gen_point_mode ? GL_TRUE : GL_FALSE;
        pptr->tess_gen_mode_specified =
            stage_info.tess_gen_mode_specified ? GL_TRUE : GL_FALSE;
        /* Split the TES execution mode.  The Metal tessellator has no
         * isolines patch type or point output topology, so those compile to
         * the AIR compute expansion — or, when nothing forces a compute
         * record (no XFB, no following GS), to a render vertex function that
         * rasterizes the expanded point/line stream directly.  uses_tess_level
         * stays on the compute path: it needs the exact float32 factors the
         * compute ABI reads, and triangles/quads fill has its own native
         * post-tessellation route. */
        /* Must agree with the MGL_AIR_COMPILE_TES_VERTEX decision above,
         * including the env gate: the draw path picks EXEC_TES_VERTEX from
         * this field, and only a TES compiled with that flag produces the
         * render vertex function it needs.  With the gate off the field
         * clears, tess_eval_compute stays set, and the draw falls back to
         * the compute expansion instead of MGL_TESS_EXEC_UNSUPPORTED. */
        pptr->tess_eval_render_vertex =
            ((pptr->tess_gen_mode == GL_ISOLINES ||
              pptr->tess_gen_point_mode) &&
             pptr->transform_feedback_varying_count == 0 &&
             !pptr->shader_slots[_GEOMETRY_SHADER] &&
             mgl_env_flag_enabled_default_on("MGL_TES_VERTEX_RENDER"))
                ? GL_TRUE : GL_FALSE;
        /* Both can be set for isolines / point_mode under the render-vertex
         * gate: the TES metallib then carries the render-vertex function
         * (metallib_bytes) and the compute expansion kernel
         * (metallib_bytes_tes_compute).  Non-indexed draws use the vertex
         * path; indexed draws fall back to the compute kernel, which owns
         * the gather ABI.  For triangles/quads / XFB / GS / uses_tess_level
         * only tess_eval_compute is set and the kernel lives in
         * metallib_bytes as before. */
        pptr->tess_eval_compute =
            ((pptr->tess_gen_mode == GL_ISOLINES ||
              pptr->tess_gen_point_mode ||
              pptr->transform_feedback_varying_count > 0 ||
              pptr->shader_slots[_GEOMETRY_SHADER] ||
              stage_info.uses_tess_level))
                ? GL_TRUE : GL_FALSE;
        pptr->tess_uses_cull_distance =
            stage_info.uses_cull_distance ? GL_TRUE : GL_FALSE;
        pptr->tess_cull_distance_count = stage_info.cull_distance_count;
    } else if (stage == _GEOMETRY_SHADER) {
        pptr->geometry_input_type = stage_info.geometry_input_type;
        pptr->geometry_output_type = stage_info.geometry_output_type;
        pptr->geometry_vertices_out = stage_info.geometry_vertices_out;
        pptr->geometry_max_vertices_specified =
            stage_info.geometry_max_vertices_specified ? GL_TRUE : GL_FALSE;
        pptr->geometry_invocations = stage_info.geometry_invocations;
        pptr->geometry_stream_count = stage_info.gs_stream_count;
        for (uint32_t si = 0; si < 4u; si++) {
            pptr->geometry_stream_varying_count[si] =
                stage_info.gs_stream_varying_count[si];
            pptr->geometry_stream_xfb_stride[si] =
                stage_info.gs_stream_xfb_stride[si];
        }
    }
    free(merged_src);
    return 1;
}

