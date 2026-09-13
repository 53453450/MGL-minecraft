/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_dispatch.c — the TCS dispatch entry moved out of
 * MGLRenderer+Tessellation.m (P0-1, log 126).
 *
 *   -dispatchTessControlShader:program:contract:  -> mglTessDispatchControlShader
 *   -newTCSStageInBufferForContext:...            -> mglTessDispatchNewTCSStageInBuffer (static)
 *
 * Translation rules are the ones the previous cuts settled: `self` becomes the
 * renderer handle, the pass-manager state and the tessellation record arrive
 * through MGLRendererStateAreas, MGL_STATE() becomes the C twin, "[self ...]"
 * becomes the C entry of mgl_renderer_ports.h and the NSMutableArray of
 * temporaries becomes the mglRendererTemporaries handle.
 *
 * Two ownership details are the whole risk of this cut:
 *   - the compute pipeline handle is a +1 that the ARC local
 *     `(__bridge_transfer id)tcsPipeline` released at every exit; the C twin
 *     releases it at the single `done:` label instead;
 *   - the temporaries set is created +1 and released at the same label, after
 *     the plan has been encoded (mglRenderExecuteComputeExecutionPlan encodes
 *     synchronously, so the command buffer holds the buffers from then on).
 * The copy-back list is cleared at that label too: the method cleared it on
 * every path, and the clear (backend list reset + memset) is idempotent.
 */

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_tess_dispatch.h"
#include "mgl_tess_stage_bind.h"         /* stage-binding + texture plan (log 125) */
#include "mgl_tess_texture.h"            /* mglTessEnsureTextureMetalData */
#include "mgl_tess_compute_ops.h"        /* mglTessBindPointSizeParamsToComputeEncoder */
#include "mgl_renderer_ports.h"          /* state areas, command buffer, processBuffer */
#include "mgl_renderer_backend.h"        /* tcs output / patch-out / capture slots */
#include "mgl_render.h"                  /* buffer creation, copy encodings, execute */
#include "mgl_compute_pipeline_cache.h"  /* mglGetOrCreateProgramComputePipeline */
#include "mgl_draw_tess.h"               /* the tess plan predicates */
#include "mgl_air_tess_abi.h"            /* mglRenderFillDefaultTessFactorBuffer */
#include "mgl_vertex_attrib_query.h"     /* mglRendererGetValidatedVAO */
#include "mgl_vertex_attrib_binding.h"   /* mglRendererResolveVertexAttribBinding */
#include "mgl_index_buffer.h"            /* mglPrimitiveRestartIndexForType */
#include "mgl_size_constants.h"          /* kMGLMaxBufferSlots */
#include "mgl_thread_affinity.h"         /* MGL_ASSERT_GL_THREAD */
#include "glm_limits.h"                  /* MAX_ATTRIBS, TEXTURE_UNITS */

/* Declared next to their definitions in Objective-C headers a .c file cannot
 * include; repeated here the way mgl_renderer_ports.c repeats its prototypes.
 * mglRendererBuildCurrentVertexAttribBytes returns NSUInteger there, which is
 * the same 64-bit unsigned type as size_t on every target MGL builds for. */
extern Buffer *getElementBuffer(GLMContext ctx);
extern size_t mglRendererBuildCurrentVertexAttribBytes(GLMContext ctx,
                                                       GLuint attribute,
                                                       const VertexAttrib *attrib,
                                                       uint8_t bytes[16]);

/* MGL_STATE() from MGLRenderer_Private.h, in C (the same twin as
 * mgl_compute_bind.c). */
static GLMState *mglTessDispatchState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

static void *mglTessDispatchCommandBufferOwner(
    const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentCommandBufferOwner : NULL;
}

static void *mglTessDispatchRenderEncoderOwner(
    const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentRenderEncoderOwner : NULL;
}

/* === File-local twins of the Objective-C statics ==========================
 * MGLRenderer+Tessellation.m still owns the `id`-typed statics for the two AIR
 * TES methods that have not moved yet; these are the C twins over the same
 * mglRender* entries. */

/* The .m's MGL_TESS_RESOURCE_STORAGE_SHARED: MTLResourceStorageModeShared. */
#define MGL_TESS_DISPATCH_STORAGE_SHARED 0u

/* +1 buffer, or NULL. */
static void *mglTessDispatchCreateBuffer(size_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((uint64_t)length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

/* +1 buffer, or NULL. */
static void *mglTessDispatchCreateBufferWithBytes(const void *bytes,
                                                  size_t length,
                                                  uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL, &buffer) ==
            0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

static void *mglTessDispatchBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents(buffer, &contents, &length) == 0
               ? contents
               : NULL;
}

/* The .m's mglRendererReadableBufferBytes. */
static const uint8_t *mglTessDispatchReadableBufferBytes(Buffer *buffer)
{
    if (!buffer) {
        return NULL;
    }
    if (buffer->data.buffer_data &&
        mglRenderCPUPointerUsable((const void *)buffer->data.buffer_data)) {
        return (const uint8_t *)(uintptr_t)buffer->data.buffer_data;
    }
    if (buffer->data.mtl_data) {
        return (const uint8_t *)mglTessDispatchBufferContents(
            buffer->data.mtl_data);
    }
    return NULL;
}

/* Was -newTCSStageInBufferForContext:program:first:count:indexType:indices:
 * baseVertex:baseInstance:patchVertices:patchCount:outStride:.  Returns the +1
 * stage_in buffer, or NULL. */
static void *mglTessDispatchNewTCSStageInBuffer(
    void *renderer, GLMContext draw_ctx, Program *tcs_program, GLint first,
    GLsizei count, GLenum index_type, const void *indices, GLint base_vertex,
    GLuint base_instance, GLuint patch_vertices, GLuint patch_count,
    size_t *out_stride)
{
    MGL_ASSERT_GL_THREAD();
    if (out_stride) {
        *out_stride = 0u;
    }
    if (!renderer || !draw_ctx || !tcs_program || count <= 0) {
        return NULL;
    }

    if (!tcs_program->modules[_TESS_CONTROL_SHADER].metallib_bytes) {
        return NULL;
    }

    MGLTessTCSStageInPlan stage_plan = {0};
    if (!mglTessPlanTCSStageIn(patch_vertices, patch_count, count,
                               &stage_plan)) {
        return NULL;
    }
    size_t tcs_in_stride = (size_t)stage_plan.stride;
    size_t member_count = stage_plan.member_count;
    MGLTessStageInMember members[MAX_ATTRIBS];
    memset(members, 0, sizeof(members));
    members[0] = stage_plan.members[0];
    size_t tcs_in_vertices = (size_t)stage_plan.vertices;

    VertexArray *vao = mglRendererGetValidatedVAO(draw_ctx, "tcs.stage_in");
    if (!vao) {
        return NULL;
    }

    const uint8_t *index_bytes = NULL;
    size_t index_offset = (size_t)(uintptr_t)indices;
    uint32_t restart_index = 0u;
    int primitive_restart = 0;
    if (index_type != 0u) {
        Buffer *ebo = getElementBuffer(draw_ctx);
        if (!ebo || !mglRendererProcessBuffer(renderer, ebo)) {
            fprintf(stderr,
                    "MGL TESS WARNING: TCS indexed stage_in has no readable "
                    "element buffer\n");
            return NULL;
        }
        const uint8_t *ebo_bytes = mglTessDispatchReadableBufferBytes(ebo);
        MGLTessIndexedStageInPlan index_plan = {0};
        if (!ebo_bytes ||
            !mglTessPlanIndexedStageIn((uint32_t)index_type,
                                       (uint64_t)index_offset, (int32_t)count,
                                       ebo->size, &index_plan) ||
            index_plan.status != MGL_TESS_INDEXED_STAGE_IN_OK) {
            fprintf(stderr,
                    "MGL TESS WARNING: TCS indexed stage_in element range OOB "
                    "offset=%lu size=%lld",
                    (unsigned long)index_offset, (long long)ebo->size);
            return NULL;
        }
        index_bytes = ebo_bytes + index_offset;
        primitive_restart = mglPrimitiveRestartIndexForType(
            draw_ctx, index_type, &restart_index);
    }

    size_t tcs_in_size = (size_t)stage_plan.bytes;
    void *stage_in_buffer = mglTessDispatchCreateBuffer(
        tcs_in_size, MGL_TESS_DISPATCH_STORAGE_SHARED);
    void *stage_in_contents = mglTessDispatchBufferContents(stage_in_buffer);
    if (!stage_in_contents) {
        return NULL;
    }
    if (!mglTessInitStageInDefaults(stage_in_contents, tcs_in_vertices,
                                    tcs_in_stride)) {
        return NULL;
    }

    if (mglTessTCSStageInEmptyOK((uint32_t)member_count)) {
        if (out_stride) {
            *out_stride = tcs_in_stride;
        }
        return stage_in_buffer;
    }

    MGLTessStageInAttribSrc srcs[MAX_ATTRIBS];
    memset(srcs, 0, sizeof(srcs));
    for (size_t m = 0; m < member_count; m++) {
        const MGLTessStageInMember *member = &members[m];
        if (!mglTessStageInAttribInRange(member->attribute)) {
            continue;
        }
        const VertexAttrib *attrib = &vao->attrib[member->attribute];
        MGLResolvedVertexAttribBinding resolved = {0};
        bool has_binding = mglRendererResolveVertexAttribBinding(
            draw_ctx, vao, member->attribute, "tcs.stage_in", &resolved);
        bool use_current_value = mglTessStageInUseCurrentValue(
                                     vao->enabled_attribs, member->attribute,
                                     has_binding ? 1 : 0) != 0;
        srcs[m].type = attrib->type;
        srcs[m].attrib_size = attrib->size;
        srcs[m].normalized = attrib->normalized;
        if (use_current_value) {
            srcs[m].use_current = 1u;
            if (mglRendererBuildCurrentVertexAttribBytes(
                    draw_ctx, member->attribute, attrib, srcs[m].current) > 0u) {
                srcs[m].current_valid = 1u;
            }
        } else if (has_binding) {
            Buffer *vbo = resolved.buffer;
            if (vbo && mglRendererProcessBuffer(renderer, vbo)) {
                srcs[m].bytes = mglTessDispatchReadableBufferBytes(vbo);
                srcs[m].stride = resolved.stride;
                srcs[m].divisor = resolved.divisor;
                srcs[m].binding_offset = resolved.binding_offset;
                srcs[m].relativeoffset = resolved.relativeoffset;
                srcs[m].buffer_size = mglRenderBufferSizeOrZero(vbo->size);
            }
        }
    }
    if (!mglTessPackStageInRecords(
            stage_in_contents, tcs_in_vertices, tcs_in_stride, first, count,
            index_bytes, index_type, primitive_restart != 0, restart_index,
            base_vertex, base_instance, members, (uint32_t)member_count, srcs)) {
        return NULL;
    }

    if (out_stride) {
        *out_stride = tcs_in_stride;
    }
    return stage_in_buffer;
}

/* Was -dispatchTessControlShader:program:contract:. */
bool mglTessDispatchControlShader(void *renderer, GLMContext glm_ctx,
                                  Program *tcs_program,
                                  const MGLAIRTessDrawContract *contract)
{
    MGLStageBindingCopyBackList stage_copy_backs = {0};
    void *tcs_pipeline = NULL;
    void *temporaries = NULL;
    bool ok = false;

    if (!renderer || !tcs_program || !glm_ctx || !contract) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    Shader *tcs_shader = tcs_program->shader_slots[_TESS_CONTROL_SHADER];
    if (!mglTessStageHasCompiledFunction(
            tcs_shader ? 1 : 0,
            tcs_program->modules[_TESS_CONTROL_SHADER].mtl_function ? 1 : 0)) {
        fprintf(stderr,
                "MGL TESS WARNING: TCS program %u has no compiled function",
                tcs_program->name);
        return false;
    }

    /* Create compute pipeline state for TCS kernel.  The ready handle is a +1
     * the ARC local owned; here the `done:` label releases it. */
    void *tcs_pipeline_handle = NULL;
    char tcs_pipeline_error[512] = {0};
    int tcs_pipeline_result = mglGetOrCreateProgramComputePipeline(
        tcs_program, _TESS_CONTROL_SHADER, &tcs_pipeline_handle,
        tcs_pipeline_error, sizeof(tcs_pipeline_error));
    if (mglTessComputePipelineReady(tcs_pipeline_result,
                                    tcs_pipeline_handle ? 1 : 0)) {
        tcs_pipeline = tcs_pipeline_handle;
    }
    if (!tcs_pipeline) {
        fprintf(stderr,
                "MGL TESS ERROR: failed to create TCS compute pipeline for "
                "program %u: %s",
                tcs_program->name,
                tcs_pipeline_error[0] ? tcs_pipeline_error : "unknown error");
        return false;
    }

    temporaries = mglRendererTemporariesCreate();

    /* PASS 1: Pre-resolve all Metal textures that the TCS kernel needs.
     * This must happen BEFORE we open a compute encoder, because lazy
     * Metal texture creation (bindMTLTexture:) may open its own blit
     * encoder on the command buffer, and Metal forbids two encoders
     * on the same command buffer simultaneously.  End any active render
     * encoder first for the same reason. */
    if (mglTessMustEndRenderBeforeCompute(mglRenderEncoderOwnerHasCurrent(
            mglTessDispatchRenderEncoderOwner(&areas)))) {
        mglRendererEndRenderEncodingPort(renderer);
    }

    /* Ensure a writable command buffer exists.  The GL_PATCHES path returns
     * before processGLState() (which normally creates the command buffer),
     * and prior operations (glBufferData, glEndQuery, etc.) may have
     * committed the previous command buffer. */
    MGLRenderCommandBufferState command_state = {0};
    const int has_command_state = mglRenderCommandBufferOwnerHasState(
        mglTessDispatchCommandBufferOwner(&areas), &command_state);
    if (mglTessCommandBufferNeedsNew(has_command_state, command_state.status)) {
        /* -newCommandBuffer was METAL_LOCK + -newCommandBufferLocked +
         * METAL_UNLOCK; the lock is the GL-thread assertion. */
        MGL_ASSERT_GL_THREAD();
        if (!mglRendererNewCommandBufferLockedPort(renderer)) {
            fprintf(stderr,
                    "MGL TESS ERROR: failed to create command buffer for TCS "
                    "dispatch\n");
            goto done;
        }
    }

    MGLTessTextureBind tcs_texture_binds[TEXTURE_UNITS * 2u];
    const uint32_t tcs_texture_bind_count = mglTessCollectTextureBinds(
        glm_ctx, tcs_program, _TESS_CONTROL_SHADER, tcs_texture_binds,
        (uint32_t)(sizeof(tcs_texture_binds) / sizeof(tcs_texture_binds[0])));
    if (!mglTessEnsureTextureMetalData(renderer, tcs_texture_binds,
                                       tcs_texture_bind_count, glm_ctx)) {
        goto done;
    }

    MGLTessStageBufferBindingList stage_buffer_bindings = {0};
    if (!mglTessPrepareStageBufferBindings(renderer, &stage_buffer_bindings,
                                           _TESS_CONTROL_SHADER,
                                           &stage_copy_backs)) {
        goto done;
    }

    MGLRenderComputeExecutionPlan execution_plan = {0};
    execution_plan.pipeline = tcs_pipeline;

    if (!mglTessPlanTextureBinds(renderer, tcs_texture_binds,
                                 tcs_texture_bind_count, glm_ctx,
                                 &execution_plan, temporaries)) {
        goto done;
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TCS. */
    if (!mglTessBindPreparedStageBufferBindings(&stage_buffer_bindings, NULL,
                                                &execution_plan, temporaries)) {
        goto done;
    }
    mglTessBindPointSizeParamsToComputeEncoder(renderer, tcs_program,
                                               _TESS_CONTROL_SHADER,
                                               &execution_plan, temporaries);

    MGLTessTCSCoreLayout tcs_layout;
    if (!mglTessComputeTCSCoreLayout(tcs_program, contract, &tcs_layout)) {
        goto done;
    }
    areas.tessellation->tcsOutputStride = tcs_layout.output_stride;
    areas.tessellation->tcsOutVertices = tcs_layout.tcs_out_vertices;
    const GLuint patch_vertices = tcs_layout.patch_vertices;
    const GLuint instance_count = tcs_layout.instance_count;
    const GLuint patch_count = tcs_layout.patch_count;

    void *tcs_output_buffer = mglTessDispatchCreateBuffer(
        (size_t)tcs_layout.output_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
    (void)mglRendererBackendSetTcsOutputBuffer(areas.backend, tcs_output_buffer);
    void *tcs_output_contents =
        mglTessDispatchBufferContents(tcs_output_buffer);
    if (!tcs_output_contents) {
        goto done;
    }
    memset(tcs_output_contents, 0, (size_t)tcs_layout.output_bytes);
    areas.tessellation->tcsOutputOffset = 0u;
    mglRendererTemporariesAdd(temporaries, tcs_output_buffer);

    void *tcs_patch_out_buffer = mglTessDispatchCreateBuffer(
        (size_t)tcs_layout.patch_out_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
    (void)mglRendererBackendSetTcsPatchOutBuffer(areas.backend,
                                                 tcs_patch_out_buffer);
    void *tcs_patch_out_contents =
        mglTessDispatchBufferContents(tcs_patch_out_buffer);
    if (!tcs_patch_out_contents) {
        goto done;
    }
    memset(tcs_patch_out_contents, 0, (size_t)tcs_layout.patch_out_bytes);
    mglRendererTemporariesAdd(temporaries, tcs_patch_out_buffer);

    GLuint indirect_params[2] = {0u, 0u};
    mglTessFillTCSIndirectParams(patch_vertices, instance_count,
                                 indirect_params);
    void *indirect_buffer = mglTessDispatchCreateBufferWithBytes(
        indirect_params, sizeof(indirect_params),
        MGL_TESS_DISPATCH_STORAGE_SHARED);
    if (!indirect_buffer) {
        goto done;
    }
    mglRendererTemporariesAdd(temporaries, indirect_buffer);

    void *tess_factor_buffer = mglTessDispatchCreateBuffer(
        (size_t)tcs_layout.factor_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
    void *tess_factor_contents =
        mglTessDispatchBufferContents(tess_factor_buffer);
    if (!tess_factor_contents) {
        goto done;
    }
    /* GL 4.6 §11.2.2: TCS-unwritten tess levels take PATCH_DEFAULT_*. */
    if (mglRenderFillDefaultTessFactorBuffer(
            tess_factor_contents, tcs_layout.factor_bytes,
            mglTessDispatchState(&areas)->var.patch_default_outer_level,
            mglTessDispatchState(&areas)->var.patch_default_inner_level,
            tcs_layout.patch_count) != 0) {
        goto done;
    }
    mglRendererTemporariesAdd(temporaries, tess_factor_buffer);

    size_t tcs_in_stride = 0u;
    void *tcs_stage_in_buffer =
        mglRendererBackendGetTessVertexCaptureBuffer(areas.backend);
    size_t tcs_stage_in_offset =
        areas.tessellation->tessVertexCaptureOffset;
    MGLTessTCSStageInSourcePlan stage_in_source = {0};
    mglTessPlanTCSStageInSource(tcs_stage_in_buffer ? 1 : 0, tcs_program,
                                &stage_in_source);
    if (stage_in_source.kind == MGL_TESS_TCS_STAGE_IN_CAPTURE) {
        tcs_in_stride = (size_t)stage_in_source.stride;
        mglRendererTemporariesAdd(temporaries, tcs_stage_in_buffer);
    } else {
        tcs_stage_in_buffer = mglTessDispatchNewTCSStageInBuffer(
            renderer, glm_ctx, tcs_program, contract->first,
            (GLsizei)contract->vertex_count, contract->index_type,
            (const void *)(uintptr_t)contract->index_source,
            contract->base_vertex, contract->base_instance, patch_vertices,
            patch_count, &tcs_in_stride);
        tcs_stage_in_offset = 0u;
        if (tcs_stage_in_buffer) {
            mglRendererTemporariesAdd(temporaries, tcs_stage_in_buffer);
        }
    }
    if (!tcs_stage_in_buffer) {
        fprintf(stderr,
                "MGL TESS WARNING: failed to pack TCS stage_in buffer for "
                "program %u",
                tcs_program ? (unsigned)tcs_program->name : 0u);
        goto done;
    }
    if (!mglTessAppendTCSCoreBindings(
            &execution_plan, tcs_output_buffer, tcs_patch_out_buffer,
            indirect_buffer, tess_factor_buffer, tcs_stage_in_buffer,
            (uint64_t)tcs_stage_in_offset, &tcs_layout)) {
        goto done;
    }

    {
        MGLRenderCopyBackEntry copy_back_entries[kMGLMaxBufferSlots] = {0};
        uint32_t copy_back_entry_count = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)stage_copy_backs.slots,
            kMGLMaxBufferSlots, copy_back_entries, kMGLMaxBufferSlots);
        execution_plan.barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
        MGLRenderComputeExecutionResult execution_result = {0};
        char execution_error[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                mglTessDispatchCommandBufferOwner(&areas),
                areas.gpu_recovery_command_owner
                    ? *areas.gpu_recovery_command_owner
                    : NULL,
                &execution_plan, copy_back_entries, copy_back_entry_count, 1u,
                &execution_result, execution_error,
                sizeof(execution_error)) != 0) {
            if (execution_result.transaction.device_reset_requested) {
                atomic_store_explicit(&areas.core->deviceResetRequested, true,
                                      memory_order_release);
            }
            fprintf(stderr, "MGL TESS ERROR: C++ TCS execution failed: %s",
                    execution_error[0] ? execution_error : "unknown error");
            goto done;
        }
    }

    /* Save tess factor buffer for TES patch-draw path. */
    (void)mglRendererBackendSetCurrentTessFactorBuffer(areas.backend,
                                                       tess_factor_buffer);
    ok = true;

done:
    mglRendererClearStageBindingCopyBacksPort(renderer, &stage_copy_backs);
    if (temporaries) {
        mglRendererTemporariesRelease(temporaries);
    }
    CFRelease((CFTypeRef)tcs_pipeline);
    return ok;
}
