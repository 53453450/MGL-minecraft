/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_compute_dispatch.c — the dispatch orchestration of
 * MGLRenderer+Compute.m moved here (P0-1, log 111).  The translation is the
 * established one: ctx -> areas.ctx, MGL_STATE(ctx) -> the twin below (with the
 * caller's context where the method used MGL_STATE(glm_ctx)),
 * _renderPassManager->state -> areas.command, _gpuRecovery.commandRecoveryOwner
 * -> *areas.gpu_recovery_command_owner, _batching.currentCommandBufferHasWork
 * -> areas.batching->currentCommandBufferHasWork, _deviceResetRequested ->
 * areas.core->deviceResetRequested, "[self ...]" -> the C entries of
 * mgl_renderer_ports.h, NSLog -> fprintf on the same sink.
 *
 * The methods also assigned `ctx = glm_ctx;` (the renderer's context ivar)
 * before doing their work; the shell keeps that assignment as
 * mglPlatformShellSetContext() so the port-wrapped methods downstream still see
 * the same renderer context they did before.
 */

#include <stdio.h>
#include "mgl_render_pass_manager_ops.h" /* mglRenderPassNewCommandBufferLocked */
#include <string.h>
#include <stdatomic.h>

#include "mgl_render_pass_sync_ops.h"
#include "mgl_compute_dispatch.h"
#include "mgl_compute_bind.h"
#include "mgl_stage_copy_back.h"
#include "mgl_renderer_ports.h"     /* state areas + host entries */
#include "mgl_renderer_backend.h"   /* mglRendererProcessBuffer, context lookup */
#include "mgl_texture_bind.h"       /* mglRendererBindMTLTexture */
#include "mgl_compute_pipeline_cache.h" /* mglGetOrCreateProgramComputePipeline */
#include "mgl_binding_stage.h"      /* MGLStageBindingCopyBackList */
#include "mgl_buffer_slots.h"       /* kMGLMaxBufferSlots */
#include "mgl_types_texture.h"      /* ImageUnit */
#include "mgl_types_state.h"        /* mglMarkRendererDirtyBits */
#include "glm_limits.h"             /* TEXTURE_UNITS */

/* MGL_STATE() from MGLRenderer_Private.h, in C. */
static GLMState *mglComputeDispatchState(const MGLRendererStateAreas *areas,
                                         GLMContext context)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return context ? context->active_state : NULL;
}

bool mglComputeProcess(void *renderer, void *encoder,
                       MGLStageBindingCopyBackList *copy_backs,
                       MGLRenderComputeExecutionPlan *plan, void *temporaries)
{
    /* from https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Compute-Ctx/Compute-Ctx.html */
    if (!encoder && !plan) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: processCompute called with NULL encoder\n");
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglComputeDispatchState(&areas, ctx);

    Program *program = mglResolveProgramForStageFromState(ctx, _COMPUTE_SHADER);
    if (!program) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: glDispatchCompute with no current program\n");
        mglDispatchError(ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return false;
    }

    if (program->dirty_bits)
    {
        if (!mglRenderPassBindMTLProgram(renderer, program)) {
            fprintf(stderr,
                    "MGL COMPUTE ERROR: failed to bind compute program %u\n",
                    program->name);
            return false;
        }
    }

    Shader *computeShader = program->shader_slots[_COMPUTE_SHADER];
    if (!computeShader) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: current program %u has no compute shader\n",
                program->name);
        mglDispatchError(ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return false;
    }

    void *func = program->modules[_COMPUTE_SHADER].mtl_function;
    if (!func) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: compute shader for program %u has no Metal function\n",
                program->name);
        return false;
    }

    void *computePipelineHandle = NULL;
    char computePipelineError[2048] = {0};
    int computePipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _COMPUTE_SHADER, &computePipelineHandle,
        computePipelineError, sizeof(computePipelineError));
    void *computePipelineState =
        computePipelineResult == 0 && computePipelineHandle
            ? computePipelineHandle
            : NULL;
    if (!computePipelineState) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: failed to create compute pipeline for program %u: %s\n",
                program->name,
                computePipelineError[0] ? computePipelineError : "unknown error");
        return false;
    }

    if (plan) {
        plan->pipeline = computePipelineState;
        if (temporaries) {
            mglRendererTemporariesAdd(temporaries, computePipelineState);
        }
    } else {
        (void)mglRenderSetComputePipelineState(encoder, computePipelineState);
    }

    RETURN_FALSE_ON_FAILURE(mglComputeBindBuffersToEncoder(
        renderer, _COMPUTE_SHADER, encoder, copy_backs, plan, temporaries));

    RETURN_FALSE_ON_FAILURE(mglComputeBindTexturesToEncoder(
        renderer, _COMPUTE_SHADER, encoder, plan, temporaries));

    /* setSamplerState:atIndex: / setSamplerStates:... / setThreadgroupMemoryLength:atIndex:
     * are issued by the binding layers above; nothing left to encode here. */

    state->dirty_bits = 0;

    return true;
}

bool mglComputeRunDispatchOrchestrationLocked(
    void *renderer, GLMContext glm_ctx, uint32_t dispatch_kind, uint32_t groups_x,
    uint32_t groups_y, uint32_t groups_z, void *indirect_buffer,
    size_t indirect_offset, const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMState *state = mglComputeDispatchState(&areas, glm_ctx);
    void *command_buffer_owner =
        areas.command ? areas.command->currentCommandBufferOwner : NULL;
    void *recovery_owner = areas.gpu_recovery_command_owner
                               ? *areas.gpu_recovery_command_owner
                               : NULL;

    /* end encoding on current render encoder */
    mglRendererEndRenderEncodingLocked(renderer);

    if (!mglRenderPassEnsureWritableCommandBufferLocked(renderer, reason)) {
        return false;
    }

    for (size_t unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *imageTexture = state->image_units[unit].tex;
        if (imageTexture) {
            if (!mglRendererBindMTLTexture(renderer, imageTexture)) {
                return false;
            }
        }

        Texture *sampledTexture = state->active_textures[unit];
        if (sampledTexture) {
            if (!mglRendererBindMTLTexture(renderer, sampledTexture)) {
                return false;
            }
        }
    }

    MGLStageBindingCopyBackList copyBacks = {0};
    const bool useExecutionPlan = true;
    MGLRenderComputeExecutionPlan executionPlan = {0};
    void *executionTemporaries = useExecutionPlan
        ? mglRendererTemporariesCreate() : NULL;
    void *computeCommandEncoder = NULL;
    if (!useExecutionPlan) {
        computeCommandEncoder = mglRenderCreateComputeEncoderBorrowed(
            command_buffer_owner);
        if (!computeCommandEncoder) {
            fprintf(stderr,
                    "MGL ERROR: Failed to create compute command encoder for %s\n",
                    reason ? reason : "dispatch");
            return false;
        }
    }

    if (!mglComputeProcess(renderer, computeCommandEncoder, &copyBacks,
                           useExecutionPlan ? &executionPlan : NULL,
                           executionTemporaries)) {
        if (computeCommandEncoder) {
            (void)mglRenderEndComputeEncoder(computeCommandEncoder);
        }
        mglClearStageBindingCopyBacks(renderer, &copyBacks);
        mglRendererTemporariesRelease(executionTemporaries);
        return false;
    }

    Program *ptr = mglResolveProgramForStageFromState(glm_ctx, _COMPUTE_SHADER);
    if (!ptr) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: %s with no current compute program after binding\n",
                reason ? reason : "glDispatchCompute");
        if (computeCommandEncoder) {
            (void)mglRenderEndComputeEncoder(computeCommandEncoder);
        }
        mglClearStageBindingCopyBacks(renderer, &copyBacks);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        mglRendererTemporariesRelease(executionTemporaries);
        return false;
    }

    bool hasCopyBackEntries = false;
    if (useExecutionPlan) {
        /* Copy-back resources are consumed by a following blit/CPU-visible
         * transaction. Request an explicit buffer barrier at the end of the
         * compute encoder so the plan carries the visibility requirement
         * alongside dispatch and binding state. */
        for (size_t slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            if (copyBacks.slots[slot].length != 0) {
                hasCopyBackEntries = true;
                break;
            }
        }
        executionPlan.barrier_scope = hasCopyBackEntries
            ? MGL_RENDER_COMPUTE_BARRIER_BUFFERS
            : MGL_RENDER_COMPUTE_BARRIER_NONE;
        executionPlan.dispatch = (MGLRenderComputePlan){
            .dispatch_kind = dispatch_kind,
            .groups_x = groups_x,
            .groups_y = groups_y,
            .groups_z = groups_z,
            .local_x = ptr->local_workgroup_size.x,
            .local_y = ptr->local_workgroup_size.y,
            .local_z = ptr->local_workgroup_size.z,
            .indirect_buffer = indirect_buffer,
            .indirect_offset = indirect_offset,
        };
        MGLRenderCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)copyBacks.slots, kMGLMaxBufferSlots,
            copyBackEntries, kMGLMaxBufferSlots);
        MGLRenderComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                command_buffer_owner,
                recovery_owner,
                &executionPlan,
                copyBackEntries,
                copyBackEntryCount,
                0u,
                &executionResult,
                executionError,
                sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&areas.core->deviceResetRequested, true,
                                      memory_order_release);
            }
            fprintf(stderr,
                    "MGL COMPUTE ERROR: C++ %s execution transaction failed: %s\n",
                    reason ? reason : "dispatch",
                    executionError[0] ? executionError : "unknown error");
            mglClearStageBindingCopyBacks(renderer, &copyBacks);
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
            mglRendererTemporariesRelease(executionTemporaries);
            return false;
        }
        mglClearStageBindingCopyBacks(renderer, &copyBacks);
    } else {
        MGLRenderThreadgroupSize tg = {0};
        mglRenderThreadgroupSize(
            ptr->local_workgroup_size.x, ptr->local_workgroup_size.y,
            ptr->local_workgroup_size.z, &tg);
        if (dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
            (void)mglRenderDispatchCompute(computeCommandEncoder, groups_x,
                                           groups_y, groups_z, tg.x, tg.y, tg.z);
        } else {
            (void)mglRenderDispatchComputeIndirect(computeCommandEncoder,
                                                   indirect_buffer,
                                                   (uint64_t)indirect_offset,
                                                   tg.x, tg.y, tg.z);
        }
    }

    if (computeCommandEncoder) {
        (void)mglRenderEndComputeEncoder(computeCommandEncoder);
    }
    /* Without this, a dispatch with no copy-backs stays in the current
     * command buffer and flushCommandBufferLocked's empty-CB skip drops it:
     * glFinish then never executes the compute writes (SSBO stores vanish). */
    if (areas.batching) {
        areas.batching->currentCommandBufferHasWork = 1;
    }

    if (!useExecutionPlan &&
        !mglFlushStageBindingCopyBacks(renderer, &copyBacks, 0)) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: failed to copy isolated writable buffer prefixes after %s\n",
                reason ? reason : "dispatch");
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
        mglRendererTemporariesRelease(executionTemporaries);
        return false;
    }
    if (useExecutionPlan && hasCopyBackEntries &&
        !mglRenderPassNewCommandBufferLocked(renderer)) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: failed to install post-compute command buffer after %s\n",
                reason ? reason : "dispatch");
        mglRendererTemporariesRelease(executionTemporaries);
        return false;
    }

    /* The plan held every temporary alive while it was encoded and dispatched;
     * the Objective-C version dropped its NSMutableArray here. */
    mglRendererTemporariesRelease(executionTemporaries);

    mglMarkRendererDirtyBits(
        glm_ctx->active_state,
        DIRTY_STATE | DIRTY_FBO | DIRTY_PROGRAM | DIRTY_VAO |
        DIRTY_RENDER_STATE | DIRTY_TEX_BINDING | DIRTY_TEX |
        DIRTY_TEX_PARAM | DIRTY_SAMPLER | DIRTY_ALPHA_STATE |
        DIRTY_BUFFER | DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE);
    return true;
}

void mglComputeMtlDispatchLocked(void *renderer, GLMContext glm_ctx,
                                 uint32_t groups_x, uint32_t groups_y,
                                 uint32_t groups_z)
{
    if (!glm_ctx) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: mtlDispatchCompute called with NULL context\n");
        return;
    }

    mglPlatformShellSetContext(renderer, glm_ctx);

    if (groups_x == 0 || groups_y == 0 || groups_z == 0) {
        fprintf(stderr,
                "MGL COMPUTE TRACE: glDispatchCompute zero-sized dispatch %ux%ux%u skipped\n",
                groups_x, groups_y, groups_z);
        return;
    }

    if (!mglComputeRunDispatchOrchestrationLocked(
            renderer, glm_ctx, MGL_RENDER_COMPUTE_DISPATCH_DIRECT,
            groups_x, groups_y, groups_z, NULL, 0, "glDispatchCompute")) {
        return;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMState *state = mglComputeDispatchState(&areas, glm_ctx);
    for (size_t unit = 0; unit < TEXTURE_UNITS; unit++) {
        ImageUnit *imageUnit = &state->image_units[unit];
        Texture *imageTexture = imageUnit->tex;
        if (!imageTexture ||
            !mglRenderImageAccessWritable((uint32_t)imageUnit->access)) {
            continue;
        }
        imageTexture->metal_data_authoritative = (GLboolean)mglRenderGLBoolean(1);
        if (imageTexture->faces[0].levels &&
            imageUnit->level >= 0 &&
            imageUnit->level < (GLint)imageTexture->num_levels) {
            imageTexture->faces[0].levels[imageUnit->level].metal_data_authoritative =
                (GLboolean)mglRenderGLBoolean(1);
        }
    }
}

void mglComputeMtlDispatchIndirectLocked(void *renderer, GLMContext glm_ctx,
                                         intptr_t indirect)
{
    if (!glm_ctx) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: mtlDispatchComputeIndirect called with NULL context\n");
        return;
    }

    mglPlatformShellSetContext(renderer, glm_ctx);

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMState *state = mglComputeDispatchState(&areas, glm_ctx);

    Buffer *glIndirectBuffer = state->buffers[_DISPATCH_INDIRECT_BUFFER];
    if (state->var.dispatch_indirect_buffer_binding == 0 || !glIndirectBuffer) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: glDispatchComputeIndirect with no GL_DISPATCH_INDIRECT_BUFFER bound\n");
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    if (indirect < 0) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
        return;
    }

    if (!mglRendererProcessBuffer(renderer, glIndirectBuffer)) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: failed to process dispatch indirect buffer %u\n",
                glIndirectBuffer ? glIndirectBuffer->name : 0u);
        return;
    }

    void *indirectBuffer = glIndirectBuffer->data.mtl_data;
    if (!indirectBuffer) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: dispatch indirect buffer %u has no Metal backing\n",
                glIndirectBuffer ? glIndirectBuffer->name : 0u);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    size_t indirectOffset = (size_t)indirect;
    size_t indirectArgBytes = 3u * sizeof(uint32_t);
    MGLRenderBufferInfo indirectBufferInfo = {0};
    if (mglRenderGetBufferInfo(indirectBuffer, &indirectBufferInfo) != 0 ||
        indirectOffset > indirectBufferInfo.length ||
        indirectArgBytes > (indirectBufferInfo.length - indirectOffset)) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: dispatch indirect range exceeds Metal buffer buffer=%u off=%lu bytes=%lu len=%lu\n",
                glIndirectBuffer ? glIndirectBuffer->name : 0u,
                (unsigned long)indirectOffset,
                (unsigned long)indirectArgBytes,
                (unsigned long)indirectBufferInfo.length);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (!mglComputeRunDispatchOrchestrationLocked(
            renderer, glm_ctx, MGL_RENDER_COMPUTE_DISPATCH_INDIRECT,
            0, 0, 0, indirectBuffer, indirectOffset,
            "glDispatchComputeIndirect")) {
        return;
    }
}
