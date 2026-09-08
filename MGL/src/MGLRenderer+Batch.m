/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Batch.m
// Batch scheduling and execution methods extracted from MGLRenderer+Draw.m.
// These methods do not depend on any file-scope static functions in Draw.m.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#import "mgl_sampler_compat.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_batch_path.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_restore.h"

static BOOL mglBatchHasActiveEncoder(void *owner)
{
    return mglRenderEncoderOwnerHasCurrent(owner) != 0;
}

static void mglBatchDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger indexCount,
    uint64_t indexType,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)indexType,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}


static MGLBatchStateKeyView mglBatchStateKeyViewFromKey(const MGLStateKey *key)
{
    MGLBatchStateKeyView v;
    memset(&v, 0, sizeof(v));
    if (!key) {
        return v;
    }
    v.program_name = key->program_name;
    v.program_pipeline_name = key->program_pipeline_name;
    v.vertex_program_name = key->vertex_program_name;
    v.fragment_program_name = key->fragment_program_name;
    v.vao_name = key->vao_name;
    v.vertex_layout_hash = key->vertex_layout_hash;
    v.texture_hash = key->texture_hash;
    v.render_state_hash = key->render_state_hash;
    v.uniform_buffer_hash = key->uniform_buffer_hash;
    v.caps_flags = key->caps_flags;
    v.scissor_enabled = key->scissor_enabled;
    v.primitive_type = key->primitive_type;
    memcpy(v.viewport, key->viewport, sizeof(v.viewport));
    memcpy(v.scissor, key->scissor, sizeof(v.scissor));
    return v;
}

@implementation MGLRenderer (Batch)

- (void)recordArrayDrawSubmittedMode:(GLenum)mode vertexCount:(uint64_t)vertexCount
{
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglTraceNowSeconds());
    MGL_FRAME_STORE(g_mglLastDrawArraysProgram, mglCurrentRenderProgramKey(ctx));
    MGL_FRAME_STORE(g_mglLastDrawArraysMode, mode);
    MGL_FRAME_STORE(g_mglLastDrawArraysCount,
                    (vertexCount > (uint64_t)INT_MAX) ? INT_MAX : (GLsizei)vertexCount);
    MGL_FRAME_INC(g_mglDrawArraysSinceSwap);
    MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap, vertexCount);
    [self markCurrentFramebufferDrawAttachmentsWritten];
}

- (void)recordElementDrawSubmittedMode:(GLenum)mode indexCount:(uint64_t)indexCount
{
    MGL_FRAME_STORE(g_mglLastDrawElementsSeconds, mglTraceNowSeconds());
    MGL_FRAME_STORE(g_mglLastDrawElementsProgram, mglCurrentRenderProgramKey(ctx));
    MGL_FRAME_STORE(g_mglLastDrawElementsMode, mode);
    MGL_FRAME_STORE(g_mglLastDrawElementsCount,
                    (indexCount > (uint64_t)INT_MAX) ? INT_MAX : (GLsizei)indexCount);
    MGL_FRAME_INC(g_mglDrawElementsSinceSwap);
    MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap, indexCount);
    [self markCurrentFramebufferDrawAttachmentsWritten];
}

- (bool)bindActiveTexturesToMTL
{
    // search through active_texture_mask for enabled bits
    // 128 bits long.. do it on 4 parts
    for(int i=0; i<4; i++)
    {
        unsigned mask = MGL_STATE(ctx)->active_texture_mask[i];

        if (mask)
        {
            for(int bitpos=0; bitpos<32; bitpos++)
            {
                if (mask & (0x1 << bitpos))
                {
                    Texture *tex;
                    int unit = i * 32 + bitpos;

                    tex = MGL_STATE(ctx)->active_textures[unit];
                    if (!tex)
                    {
                        // Stale active texture mask bit; clear it and continue.
                        MGL_STATE(ctx)->active_texture_mask[i] &= ~(0x1u << bitpos);
                        mglInvalidateStateHashCachesForDirtyBits(ctx->active_state,
                                                                DIRTY_TEX_BINDING);
                        continue;
                    }

                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);
                }

                // early out
                if ((mask >> (bitpos + 1)) == 0)
                    break;
            }
        }
    }

    return true;
}

- (void)invalidateLastBoundState
{
    mglRenderBindingInvalidate(_bindingStateOwner);
}

/* DUAL-PROXY INVARIANT HELPERS: see MGLRenderer_Private.h.
 *
 * These centralize all writes to _core.activeState and ctx->active_state so
 * that the invariant ("MGL_STATE(ctx) and MGL_STATE(ctx)->ctx) return the same
 * GLMState") cannot be broken by a caller forgetting to update one side.
 *
 * Valid invariant configurations:
 *   (A) _activeState == NULL  -> MGL_STATE falls through to ctx->active_state
 *                                (the "deactivated" / default mode)
 *   (B) _activeState != NULL  -> _activeState MUST equal ctx->active_state
 *                                (the "activated" mode used during batch replay)
 *
 * Configuration (A) is the teardown target; (B) is the batch-replay target. */
/* Configuration (A) is the teardown target; (B) is batch-replay via
 * replay_state (ARCHITECTURE_AUDIT R3). */
- (void)mglActivateReplayStateForContext:(GLMContext)glm_ctx
{
    memcpy(&glm_ctx->replay_state, &glm_ctx->state, sizeof(glm_ctx->replay_state));
    glm_ctx->active_state = &glm_ctx->replay_state;
    _core.activeState = &glm_ctx->replay_state;
}

- (void)mglRestoreLiveActiveStateForContext:(GLMContext)glm_ctx
{
    /* Configuration (A): ctx->active_state points to live embedded state,
     * _activeState is NULL so MGL_STATE() falls through. */
    glm_ctx->active_state = &glm_ctx->state;
    _core.activeState = NULL;
}

- (void)mglAssertDualProxyInSyncForContext:(GLMContext)glm_ctx
{
    /* Invariant checkpoint.  NSCAssert is compiled out in release builds,
     * so this is zero-cost in shipping binaries.  In debug builds it catches
     * desync at the earliest observation point (function entry/exit) instead
     * of letting it manifest as wrong binds/dirty bits later. */
    NSCAssert(_core.activeState == NULL || _core.activeState == glm_ctx->active_state,
              @"DUAL-PROXY DESYNC: _activeState != ctx->active_state — "
              @"MGL_STATE() and STATE() would read different GLMState objects");
}

- (void)recordLastBoundVertexBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    mglRenderBindingRecordVertexBuffer(
        _bindingStateOwner, (__bridge void *)buffer, offset, (uint32_t)index);
}

- (void)recordLastBoundFragmentBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    mglRenderBindingRecordFragmentBuffer(
        _bindingStateOwner, (__bridge void *)buffer, offset, (uint32_t)index);
}

- (void)invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index
{
    mglRenderBindingInvalidateVertexBuffer(
        _bindingStateOwner, (uint32_t)index);
}

- (void)invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index
{
    mglRenderBindingInvalidateFragmentBuffer(
        _bindingStateOwner, (uint32_t)index);
}

- (void)setViewportIfNeeded:(MGLViewportValue)viewport
{
    void *owner = _renderPassManager.state->currentRenderEncoderOwner;
    mglRenderBindingSetViewportForOwner(
        _bindingStateOwner, owner, viewport.origin_x, viewport.origin_y,
        viewport.width, viewport.height, viewport.znear, viewport.zfar);
}

- (void)setScissorRectIfNeeded:(MGLScissorRectValue)rect
{
    void *owner = _renderPassManager.state->currentRenderEncoderOwner;
    mglRenderBindingSetScissorForOwner(
        _bindingStateOwner, owner, rect.x, rect.y, rect.width, rect.height);
}

- (void)setTriangleFillModeIfNeeded:(uint32_t)mode
{
    void *owner = _renderPassManager.state->currentRenderEncoderOwner;
    mglRenderBindingSetTriangleFillForOwner(
        _bindingStateOwner, owner, (uint32_t)mode);
}

- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx
                           alreadyDone:(const MGLResourceSyncWork *)done
{
    GLMState *state = MGL_STATE(glm_ctx);
    if (!done || !done->mappedBuffers) {
        RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    }
    if (!done || !done->updatedBaseLists) {
        RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->vertex_buffer_map_list]);
        RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->fragment_buffer_map_list]);
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner,
    };
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindBufferSizeConstantsForRenderEncoder]);
    if (!done || !done->boundActiveTextures) {
        RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);
    }
    RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-active-texture-bind"]);
    if (![self bindTexturesToCurrentRenderEncoder:&encCtx]) {
        RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-sampled-texture-bind"]);
        RETURN_FALSE_ON_FAILURE([self bindTexturesToCurrentRenderEncoder:&encCtx]);
    }
    return true;
}

- (void)restoreStateFromKey:(const MGLStateKey *)key context:(GLMContext)glm_ctx
{
    /* Program */
    mglRestoreProgramPipelinePair(glm_ctx,
                                  key->program_name,
                                  key->program_pipeline_name);

    /* VAO */
    uint32_t vaoName = key->vao_name;
    if (vaoName != (MGL_STATE(glm_ctx)->vao ? MGL_STATE(glm_ctx)->vao->name : 0)) {
        VertexArray *vaoInst = NULL;
        if (vaoName != 0) {
            vaoInst = (VertexArray *)searchHashTable(&MGL_STATE(glm_ctx)->vao_table, vaoName);
        }
        MGL_STATE(glm_ctx)->vao = vaoInst;
    }

    /* FBO */
    uint32_t batchFBO = key->fbo_name;
    uint32_t currentFBO = MGL_STATE(glm_ctx)->framebuffer ? MGL_STATE(glm_ctx)->framebuffer->name : 0;
    if (batchFBO != currentFBO) {
        Framebuffer *fbo = NULL;
        if (batchFBO != 0) {
            fbo = (Framebuffer *)searchHashTable(&MGL_STATE(glm_ctx)->framebuffer_table, batchFBO);
        }
        MGL_STATE(glm_ctx)->framebuffer = fbo;
    }
    mglRendererSyncFramebufferBindingNames(glm_ctx);

    /* Viewport */
    MGL_STATE(glm_ctx)->viewport[0] = key->viewport[0];
    MGL_STATE(glm_ctx)->viewport[1] = key->viewport[1];
    MGL_STATE(glm_ctx)->viewport[2] = key->viewport[2];
    MGL_STATE(glm_ctx)->viewport[3] = key->viewport[3];

    /* Scissor */
    if (key->scissor_enabled) {
        MGL_STATE(glm_ctx)->caps.scissor_test = true;
        MGL_STATE(glm_ctx)->var.scissor_box[0] = key->scissor[0];
        MGL_STATE(glm_ctx)->var.scissor_box[1] = key->scissor[1];
        MGL_STATE(glm_ctx)->var.scissor_box[2] = key->scissor[2];
        MGL_STATE(glm_ctx)->var.scissor_box[3] = key->scissor[3];
    } else {
        MGL_STATE(glm_ctx)->caps.scissor_test = false;
    }
}

- (void)flushDrawBuffer:(GLMContext)glm_ctx
{
    /* Unlocked entry point: acquire METAL_LOCK and delegate to Locked variant.
     * Locked callers (mtlSwapBuffersLocked:, flushCommandBufferLocked:) call
     * flushDrawBufferLocked: directly to avoid recursive lock re-entry. */
    METAL_LOCK();
    @try {
        [self flushDrawBufferLocked:glm_ctx];
    } @finally {
        METAL_UNLOCK();
    }
}

void mglRendererFlushDrawBuffer(GLMContext glm_ctx)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        @autoreleasepool {
            @try {
                [renderer flushDrawBuffer:glm_ctx];
            } @catch (NSException *exception) {
                NSLog(@"MGL ERROR: callback flushDrawBuffer exception: %@", exception);
            }
        }
    }
    mglRendererBackendEnd(&_backend_lease);
}

- (void)flushDrawBufferLocked:(GLMContext)glm_ctx
{
    ctx = glm_ctx;

    /* DUAL-PROXY INVARIANT checkpoint: entering flushDrawBuffer.  All
     * subsequent batch replay / teardown paths assume the proxies start in
     * sync.  NSCAssert compiled out in release. */
    [self mglAssertDualProxyInSyncForContext:glm_ctx];

    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cb->batch_count == 0) {
        return;
    }

    /* Draws are about to be encoded into the current CB. */
    _currentCBHasWork = YES;

    static uint64_t s_flushDrawBufferLogCount = 0;
    uint64_t flushHit = ++s_flushDrawBufferLogCount;
    BOOL traceFlush = kMGLDiagnosticStateLogs &&
                      (flushHit <= 16ull || (flushHit % 512ull) == 0ull ||
                       cb->total_commands >= 128ull);
    uint32_t mdiBatchCount = 0;
    uint32_t mdiCommandCount = 0;
    uint32_t icbBatchCount = 0;
    uint32_t icbCommandCount = 0;
    uint32_t directBatchCount = 0;
    uint32_t directCommandCount = 0;
    uint32_t streamMergedBatchCount = 0;
    uint32_t streamMergedCommandCount = 0;
    uint32_t skippedCommandCount = 0;

    GLMState savedState;
    memcpy(&savedState, glm_ctx->active_state, sizeof(savedState));
    GLenum savedError = savedState.error;
    /* R3: replay into ctx->replay_state so restoreStateForBatch does not
     * overwrite live GL state.  Teardown retargets active_state to live. */
    [self mglActivateReplayStateForContext:glm_ctx];
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    GLenum replayError = (GLenum)mglRenderErrorNone();

    @try {

    /* Same-key restore skip: consecutive sequential batches that share an
     * MGLStateKey can reuse the already-bound encoder state without another
     * ~83KB GLMState memcpy + full processGLState.  Collision residual is
     * identical to batch merge (memcmp of hashed key fields).
     * Hold a stack copy of lastKey — do not keep pointers into batch array
     * past teardown. */
    MGLStateKey lastKey;
    BOOL lastKeyValid = NO;
    BOOL lastExecuteOk = NO;
    /* The previous batch was stream-merged: its transient vertex storage is
     * still reflected in active_state, so the next batch must not skip its
     * restore and must at least re-run the VAO/buffer domains — but lastKey
     * stays valid so the other delta domains keep narrowing. */
    BOOL lastWasStreamBatch = NO;
    memset(&lastKey, 0, sizeof(lastKey));

    for (uint32_t b = 0; b < cb->batch_count; b++) {
        @autoreleasepool {
            MGLDrawBatch *batch = &cb->batches[b];
            if (batch->command_count == 0)
                continue;

            BOOL wantAbsoluteVertexOffsets =
                batch->has_dynamic_vertex_bindings ? YES : NO;

            {
            /* A3: same-key skip decision in mgl_batch_same_key_skip_decision. */
            MGLBatchSameKeySkipIn skipIn = {
                .skip_enabled = _batching.skipSameKeyRestoreEnabled ? 1u : 0u,
                .last_key_valid = lastKeyValid ? 1u : 0u,
                .last_execute_ok = lastExecuteOk ? 1u : 0u,
                .last_was_stream = lastWasStreamBatch ? 1u : 0u,
                .has_encoder =
                    mglRenderEncoderOwnerHasCurrent(
                        _renderPassManager.state->currentRenderEncoderOwner)
                        ? 1u
                        : 0u,
                .bind_valid = mglBindingStateIsValid(_bindingStateOwner) ? 1u : 0u,
                .keys_equal = mglStateKeysEqual(&batch->key, &lastKey) ? 1u : 0u,
                .absolute_offsets_match =
                    (wantAbsoluteVertexOffsets ==
                     _batching.absoluteVertexBindingOffsets)
                        ? 1u
                        : 0u,
                .pass_matches =
                    [self currentRenderPassMatchesCurrentFramebuffer] ? 1u : 0u,
            };
            const int skipDec = mgl_batch_same_key_skip_decision(&skipIn);
            BOOL canSkipRestore = (skipDec == MGL_BATCH_SAME_KEY_SKIP);
            if (skipDec == MGL_BATCH_SAME_KEY_FAIL_NO_ENCODER) {
                MGL_PERF_INC(g_mglSkipFailNoEncoderSinceSwap);
            } else if (skipDec == MGL_BATCH_SAME_KEY_FAIL_BIND) {
                MGL_PERF_INC(g_mglSkipFailBindInvalidSinceSwap);
            } else if (skipDec == MGL_BATCH_SAME_KEY_FAIL_KEY) {
                MGL_PERF_INC(g_mglSkipFailKeyDifferSinceSwap);
            } else if (skipDec == MGL_BATCH_SAME_KEY_FAIL_PASS) {
                MGL_PERF_INC(g_mglSkipFailPassMismatchSinceSwap);
            }

            if (!_batching.skipSameKeyRestoreEnabled &&
                       lastKeyValid &&
                       lastExecuteOk &&
                       mglStateKeysEqual(&batch->key, &lastKey) &&
                       mglEnvFlagEnabled("MGL_SKIP_SAME_KEY_ORACLE")) {
                /* Oracle: measure skip opportunity without changing behavior. */
                MGL_PERF_INC(g_mglSameKeyOracleWouldSkipSinceSwap);
            }

            if (canSkipRestore) {
                /* DUAL-PROXY INVARIANT: both _activeState (ObjC ivar) and
                 * glm_ctx->active_state (C pointer) must point to the same GLMState.
                 * After skipping restore, they both still point to ctx->state from
                 * the previous batch, which is correct. Verify the invariant holds. */
                if (_activeState != glm_ctx->active_state) {
                    /* Defensive: sync _activeState to match ctx->active_state if they
                     * diverged (shouldn't happen, but fail gracefully). */
                    _activeState = glm_ctx->active_state;
                }
                MGL_STATE(glm_ctx)->dirty_bits = 0;
                MGL_PERF_INC(g_mglSameKeyRestoreSkipsSinceSwap);
            } else {
                /* Per-batch Metal vertex-buffer contract: dynamic BindVertexBuffer
                 * overrides store absolute VERTEX_BINDING_OFFSET and rebind via
                 * setVertexBuffer:offset:.  The descriptor must therefore bake only
                 * relativeoffset for those batches (see generateVertexDescriptorState).
                 * Set before restore so DIRTY_VAO rebuilds the matching descriptor. */
                GLuint absoluteContractDirty = 0u;
                if (wantAbsoluteVertexOffsets !=
                    _batching.absoluteVertexBindingOffsets) {
                    absoluteContractDirty = (DIRTY_VAO | DIRTY_BUFFER);
                }
                _batching.absoluteVertexBindingOffsets = wantAbsoluteVertexOffsets;
                [self restoreStateForBatch:batch
                                   context:glm_ctx
                                savedState:&savedState
                                   prevKey:(lastKeyValid ? &lastKey : NULL)
                           forcedDirtyBits:((lastWasStreamBatch
                                            ? (DIRTY_VAO | DIRTY_BUFFER) : 0u) |
                                           absoluteContractDirty)];
            }

            if (![self checkBatchShouldExecute:batch
                                       context:glm_ctx
                                       flushId:flushHit
                                    batchIndex:b
                                   replayError:&replayError
                               skippedCommands:&skippedCommandCount]) {
                lastExecuteOk = NO;
                continue;
            }

            lastExecuteOk = YES;
            lastKey = batch->key;
            lastKeyValid = YES;
            lastWasStreamBatch = NO;
            MGL_PERF_INC(g_mglBatchesReplayedSinceSwap);

            MGLBatchPath scheduledPath = [self scheduleDrawBatch:batch context:glm_ctx];
            MGLEncodeContext encCtx = {
                .render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner,
            };
            switch (scheduledPath) {
                case MGL_BATCH_PATH_STREAM_MERGE:
                    streamMergedBatchCount++;
                    streamMergedCommandCount += batch->command_count;
                    MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap);
                    MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap,
                                 batch->command_count);
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_STREAM_MERGE"];
                    [self issueStreamMergedBatch:batch context:glm_ctx encodeContext:&encCtx];
                    /* Stream batches bind transient vertex storage that is not
                     * represented by MGLStateKey, so the next batch must do a
                     * real restore even when the GL keys are equal.  The
                     * transient binds went through the recorded bindingSync
                     * path (cache stays truthful) and only the VAO/buffer
                     * domains are polluted — keep lastKey for delta narrowing
                     * and force those domains on the next restore instead of
                     * dropping the key entirely. */
                    lastWasStreamBatch = YES;
                    break;
                case MGL_BATCH_PATH_MDI:
                    mdiBatchCount++;
                    mdiCommandCount += batch->command_count;
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_MDI"];
                    [self issueMDIBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
                case MGL_BATCH_PATH_ICB:
                    icbBatchCount++;
                    icbCommandCount += batch->command_count;
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_ICB"];
                    [self issueIndirectCommandBufferBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
                default:
                    directBatchCount++;
                    directCommandCount += batch->command_count;
                    MGL_PERF_INC(g_mglBatchesDirectSinceSwap);
                    MGL_PERF_ADD(g_mglDrawDirectSinceSwap, batch->command_count);
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_DIRECT"];
                    [self issueDirectBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
            }

            [self recordBatchCommandStats:batch context:glm_ctx];
        } /* sequentialBatch */
        }
    }
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglTraceNowSeconds());
    if (traceFlush || skippedCommandCount > 0 ||
        !mglRenderErrorIsNone((uint32_t)replayError)) {
        mglTraceLogNSString(@"MGL TRACE flushDrawBuffer hit=%llu batches=%u totalCommands=%u arrays=%u elements=%u streamMergedBatches=%u streamMergedCommands=%u mdiBatches=%u mdiCommands=%u icbBatches=%u icbCommands=%u directBatches=%u directCommands=%u skippedCommands=%u",
              (unsigned long long)flushHit,
              cb->batch_count, cb->total_commands,
              cb->array_cmd_count, cb->element_cmd_count,
              streamMergedBatchCount, streamMergedCommandCount,
              mdiBatchCount, mdiCommandCount,
              icbBatchCount, icbCommandCount,
              directBatchCount, directCommandCount,
              skippedCommandCount);
    }
    } @finally {
        [_renderPassManager setTraceReplayFlushId:0 batchIndex:0];
        [self teardownBatchReplayForContext:glm_ctx savedState:&savedState
                                savedError:savedError replayError:replayError];
    }
}

- (MGLBatchPath)scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    /* O2.1: path decision in pure C (mgl_batch_select_path). ObjC only
     * materializes context-derived flags + OS ICB gate. */
    MGLBatchSelectInputs in = {0};
    if (!batch) {
        return (MGLBatchPath)mgl_batch_select_path(&in);
    }
    in.command_count = batch->command_count;
    in.sampler_snapshots_mixed = batch->sampler_snapshots_mixed ? 1u : 0u;
    in.stream_merged = batch->stream_merged ? 1u : 0u;
    in.has_dynamic_uniform_bindings =
        batch->has_dynamic_uniform_bindings ? 1u : 0u;
    in.has_dynamic_vertex_bindings =
        batch->has_dynamic_vertex_bindings ? 1u : 0u;
    in.has_dynamic_texture_bindings =
        batch->has_dynamic_texture_bindings ? 1u : 0u;
    in.mdi_compatible = batch->mdi_compatible ? 1u : 0u;
    in.uses_elements = batch->uses_elements ? 1u : 0u;
    in.primitive_type = batch->key.primitive_type;
    /* O2.4: unified ICB env (ENABLE_ICB / legacy BATCH|PIPELINES). */
    {
        MGLBatchIcbConfig icb = mgl_batch_icb_config();
        in.enable_icb = icb.enable;
        in.disable_icb = icb.disable;
    }
    in.disable_mdi = mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1u : 0u;
    if (@available(macOS 10.14, *)) {
        in.icb_os_supported = 1u;
    }
    Program *vertexProgram =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    if (vertexProgram && vertexProgram->uses_cull_distance) {
        in.uses_cull_distance = 1u;
    }
    if (batch->command_count > 0u) {
        in.polygon_mode_point =
            mglPolygonModePointForDrawMode(glm_ctx, batch->commands[0].mode)
                ? 1u
                : 0u;
        if (batch->uses_elements) {
            uint32_t dummy = 0u;
            in.primitive_restart =
                mglPrimitiveRestartIndexForType(
                    glm_ctx, batch->commands[0].indexType, &dummy)
                    ? 1u
                    : 0u;
        }
    }
    return (MGLBatchPath)mgl_batch_select_path(&in);
}

- (void)restoreStateForBatch:(MGLDrawBatch *)batch
                     context:(GLMContext)glm_ctx
                  savedState:(const GLMState *)savedState
{
    [self restoreStateForBatch:batch context:glm_ctx savedState:savedState
                       prevKey:NULL forcedDirtyBits:0u];
}

- (void)restoreStateForBatch:(MGLDrawBatch *)batch
                     context:(GLMContext)glm_ctx
                  savedState:(const GLMState *)savedState
                     prevKey:(const MGLStateKey *)prevKey
             forcedDirtyBits:(GLuint)forcedDirtyBits
{
    MGL_SIGNPOST_BEGIN(RestoreStateForBatch);
    /* DUAL-PROXY INVARIANT checkpoint: entering batch replay state restore.
     * Caller keeps ctx->active_state on live state until remaining ctx->state
     * readers are migrated to active_state (R3).  Sync _activeState at end. */
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    if (batch->state_snapshot) {
        /* Selective restore: only copy hot fields (~51KB vs 82KB full).
         * Cold fields (HashTables + unused buffer_base types) are restored
         * from savedState below. */
        mglCopyHotStateFields(glm_ctx->active_state,
                              (const GLMState *)batch->state_snapshot);
        MGL_PERF_INC(g_mglReplayMemcpyCountSinceSwap);
        /* The snapshot shallow-copies the 10 embedded HashTables in GLMState.
         * Each HashTable owns a dynamically-allocated keys/states array that
         * may have been reallocated since the snapshot was taken, making the
         * snapshot's copies stale (use-after-free risk).  Preserve the live
         * HashTables from savedState so lookups during replay remain valid. */
        mgl_batch_replay_copy_object_hash_tables(MGL_STATE(glm_ctx), savedState);
        /* The 11 cold buffer_base types need no restore: the hot copy above
         * skips them and nothing in replay writes them, so active_state still
         * holds the pre-flush live values (== savedState). */
        mglRestoreProgramPipelinePair(glm_ctx, MGL_STATE(glm_ctx)->program_name,
                                     MGL_STATE(glm_ctx)->var.program_pipeline_binding);
    } else {
        [self restoreStateFromKey:&batch->key context:glm_ctx];
    }
    /* Activate snapshot-based state access for sync functions.
     * _activeState points to ctx->state (which now holds the snapshot data). */
    _activeState = glm_ctx->active_state;
    MGL_STATE(glm_ctx)->dirty_bits = 0;

    static const GLuint kMGLFullReplayDirtyBits =
        (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_RENDER_STATE |
         DIRTY_TEX_BINDING | DIRTY_TEX | DIRTY_TEX_PARAM |
         DIRTY_SAMPLER | DIRTY_ALPHA_STATE | DIRTY_BUFFER |
         DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE);

    GLuint replayDirtyBits = kMGLFullReplayDirtyBits;
    BOOL prevKeyValid = (prevKey != NULL);
    BOOL canDelta = _batching.dirtyKeyDeltaEnabled &&
                    prevKeyValid &&
                    mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 0 &&
                    mglBindingStateIsValid(_bindingStateOwner);

    if (canDelta) {
        /* A3: dirty-key domain narrowing in mgl_batch_compute_key_delta_dirty_bits. */
        const MGLBatchDirtyDomainMasks masks = {
            .program = (DIRTY_PROGRAM | DIRTY_BUFFER_BASE_STATE | DIRTY_BUFFER),
            .vao = (DIRTY_VAO | DIRTY_BUFFER),
            .texture = (DIRTY_TEX | DIRTY_TEX_BINDING | DIRTY_TEX_PARAM |
                        DIRTY_SAMPLER | DIRTY_IMAGE_UNIT_STATE),
            .render_state = (DIRTY_RENDER_STATE | DIRTY_ALPHA_STATE),
        };
        MGLBatchStateKeyView prevView = mglBatchStateKeyViewFromKey(prevKey);
        MGLBatchStateKeyView curView = mglBatchStateKeyViewFromKey(&batch->key);
        MGLBatchDirtyDeltaFlags dflags;
        replayDirtyBits = mgl_batch_compute_key_delta_dirty_bits(
            1, &prevView, &curView, kMGLFullReplayDirtyBits, &masks, &dflags);
        if (dflags.domain_program) {
            MGL_PERF_INC(g_mglDeltaDomainProgramSinceSwap);
        }
        if (dflags.domain_vao) {
            MGL_PERF_INC(g_mglDeltaDomainVAOSinceSwap);
        }
        if (dflags.domain_texture) {
            MGL_PERF_INC(g_mglDeltaDomainTextureSinceSwap);
        }
        if (dflags.domain_render_state_ubo_only) {
            MGL_PERF_INC(g_mglDeltaDomainRenderStateUboOnlySinceSwap);
        } else if (dflags.domain_render_state) {
            MGL_PERF_INC(g_mglDeltaDomainRenderStateSinceSwap);
        }
        if (dflags.narrowed) {
            MGL_PERF_INC(g_mglDirtyKeyDeltaNarrowSinceSwap);
        }
    }
    replayDirtyBits |= forcedDirtyBits;

    Framebuffer *replayFBO = MGL_STATE(glm_ctx)->framebuffer;
    if ((replayFBO && (replayFBO->dirty_bits & DIRTY_FBO_BINDING)) ||
        (prevKeyValid && prevKey->fbo_name != batch->key.fbo_name) ||
        (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 0 &&
         ![self currentRenderPassMatchesCurrentFramebuffer])) {
        replayDirtyBits |= DIRTY_FBO;
    }
    /* Empty encoder cannot delta-bind — force full domains. */
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 0 || !mglBindingStateIsValid(_bindingStateOwner)) {
        replayDirtyBits = kMGLFullReplayDirtyBits |
                          ((replayDirtyBits & DIRTY_FBO) ? DIRTY_FBO : 0);
        if ((replayFBO && (replayFBO->dirty_bits & DIRTY_FBO_BINDING)) ||
            (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 0 &&
             ![self currentRenderPassMatchesCurrentFramebuffer])) {
            replayDirtyBits |= DIRTY_FBO;
        } else if (prevKeyValid && prevKey->fbo_name != batch->key.fbo_name) {
            replayDirtyBits |= DIRTY_FBO;
        }
    }
    mglMarkRendererDirtyBits(glm_ctx->active_state, replayDirtyBits);
    MGL_SIGNPOST_END(RestoreStateForBatch);
}

- (void)teardownBatchReplayForContext:(GLMContext)glm_ctx
                           savedState:(const GLMState *)savedState
                           savedError:(GLenum)savedError
                          replayError:(GLenum)replayError
{
    /* DUAL-PROXY INVARIANT checkpoint: entering batch replay teardown. */
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    /* R3: when flush redirected into replay_state, live GLMState was not
     * overwritten by restoreStateForBatch.  Sync HashTable struct fields that
     * may have grown through the shared array storage, then skip the full
     * savedState memcpy onto live. */
    const BOOL usedReplayWorkspace =
        (glm_ctx->active_state == &glm_ctx->replay_state);
    if (usedReplayWorkspace) {
        mgl_batch_replay_sync_hash_tables_from_replay(&glm_ctx->state,
                                                      &glm_ctx->replay_state);
    }
    [self mglRestoreLiveActiveStateForContext:glm_ctx];
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    _batching.absoluteVertexBindingOffsets = NO;
    mglResetCommandBufferForContext(glm_ctx, &glm_ctx->draw_command_buffer);
    if (_batching.arenaSnapshotEnabled) {
        mglResetBatchArena(&_batching.batchArena);
    }
    if (!usedReplayWorkspace) {
        memcpy(glm_ctx->active_state, savedState, sizeof(GLMState));
    }
    /* savedState carries the independent hash flags latched by live mutations.
     * Clear only renderer-consumed legacy bits; deriving flags from those bits
     * would force an unnecessary hash recompute after every non-empty flush. */
    mglClearStateDirtyBitsPreservingHashInvalidation(glm_ctx->active_state);
    mglRestoreProgramPipelinePair(glm_ctx, MGL_STATE(glm_ctx)->program_name,
                                  MGL_STATE(glm_ctx)->var.program_pipeline_binding);
    if (mglRenderErrorIsNone((uint32_t)savedError) &&
        !mglRenderErrorIsNone((uint32_t)replayError)) {
        MGL_STATE(glm_ctx)->error = replayError;
    }
    (void)savedState;
}

- (void)mglTraceSkipBatchCommands:(MGLDrawBatch *)batch
                          context:(GLMContext)glm_ctx
                          flushId:(uint64_t)flushId
                       batchIndex:(uint32_t)batchIndex
                            phase:(const char *)phase
                           reason:(const char *)reason
                  skippedCommands:(uint32_t *)skippedCommands
{
    [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                batchIndex:batchIndex phase:phase];
    for (uint32_t i = 0; i < batch->command_count; i++) {
        [self traceReplayCommand:batch command:&batch->commands[i]
                         context:glm_ctx flushId:flushId
                      batchIndex:batchIndex commandIndex:i
                           phase:"SKIP" reason:reason];
    }
    *skippedCommands += batch->command_count;
    MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
}

- (BOOL)checkBatchShouldExecute:(MGLDrawBatch *)batch
                        context:(GLMContext)glm_ctx
                        flushId:(uint64_t)flushId
                     batchIndex:(uint32_t)batchIndex
                    replayError:(GLenum *)replayError
                skippedCommands:(uint32_t *)skippedCommands
{
    [_renderPassManager setTraceReplayFlushId:flushId batchIndex:batchIndex];
    [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                batchIndex:batchIndex phase:"RESTORE"];

    if (![self prepareRenderPassIfFBOChanged:batch context:glm_ctx replayError:replayError]) {
        [self mglTraceSkipBatchCommands:batch context:glm_ctx flushId:flushId
                             batchIndex:batchIndex phase:"SKIP_FBO_ROTATION"
                                 reason:"fbo_rotation"
                        skippedCommands:skippedCommands];
        return NO;
    }

    if ([self processGLState:true] == false) {
        if (!mglRenderErrorIsNone((uint32_t)MGL_STATE(glm_ctx)->error)) {
            *replayError = MGL_STATE(glm_ctx)->error;
        }
        [self mglTraceSkipBatchCommands:batch context:glm_ctx flushId:flushId
                             batchIndex:batchIndex phase:"SKIP_PROCESS_STATE"
                                 reason:"processGLState"
                        skippedCommands:skippedCommands];
        return NO;
    }

    /* A stable sampler snapshot is batch state, not per-draw state. Apply it
     * once after texture binding so stream-merge, MDI and ICB paths remain
     * available. Only genuinely mixed batches rebind per command. */
    MGLEncodeContext samplerEncCtx = {
        .render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner,
    };
    if (!batch->sampler_snapshots_mixed &&
        batch->sampler_snapshot_id != MGL_INVALID_SAMPLER_SNAPSHOT_ID &&
        ![self applySamplerSnapshotForCommand:&batch->commands[0]
                                      context:glm_ctx
                                encodeContext:&samplerEncCtx]) {
        [self mglTraceSkipBatchCommands:batch context:glm_ctx flushId:flushId
                             batchIndex:batchIndex phase:"SKIP_SAMPLER_SNAPSHOT"
                                 reason:"sampler_snapshot"
                        skippedCommands:skippedCommands];
        return NO;
    }

    [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                batchIndex:batchIndex phase:"READY"];

    if ([self currentDrawRasterizationIsEmpty]) {
        [self mglTraceSkipBatchCommands:batch context:glm_ctx flushId:flushId
                             batchIndex:batchIndex phase:"SKIP_EMPTY_RASTER"
                                 reason:"empty_rasterization"
                        skippedCommands:skippedCommands];
        return NO;
    }

    GLenum mode = batch->commands[0].mode;
    if ([self currentDrawModeIsFullyCulled:mode]) {
        [self mglTraceSkipBatchCommands:batch context:glm_ctx flushId:flushId
                             batchIndex:batchIndex phase:"SKIP_FULLY_CULLED"
                                 reason:"front_and_back_culled"
                        skippedCommands:skippedCommands];
        return NO;
    }

    [self applyPolygonOffsetForDrawMode:mode];
    return YES;
}

- (void)recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        if (mgl_batch_replay_cmd_is_array_draw((uint32_t)cmd->type)) {
            MGL_FRAME_INC(g_mglDrawArraysSinceSwap);
            MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap,
                          (uint64_t)(cmd->count > 0 ? cmd->count : 0));
        } else if (mglDrawCommandUsesElements(cmd)) {
            MGL_FRAME_INC(g_mglDrawElementsSinceSwap);
            MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap,
                          (uint64_t)(cmd->count > 0 ? cmd->count : 0));
        }
    }
    [self markCurrentFramebufferDrawAttachmentsWritten];
    (void)glm_ctx;
}

- (void)issueStreamMergedBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                 encodeContext:(const MGLEncodeContext *)encCtx
{
    /* A3: stream path plan in mgl_batch_replay_stream_path. */
    const int streamPath = mgl_batch_replay_stream_path(
        batch, mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1 : 0);
    if (streamPath == MGL_BATCH_STREAM_EMPTY) {
        if (batch && batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"SKIP"
                              reason:mgl_batch_replay_stream_path_reason(streamPath)];
        }
        return;
    }
    if (streamPath == MGL_BATCH_STREAM_BAD_PRIM) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:mgl_batch_replay_stream_path_reason(streamPath)];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }
    if (streamPath == MGL_BATCH_STREAM_TRY_MDI) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"ISSUE"
                              reason:mgl_batch_replay_stream_path_reason(streamPath)];
        }
        if ([self issueStreamMergedMDIBatch:batch context:glm_ctx encodeContext:encCtx]) {
            return;
        }
    }

    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;
    if (!indexBuffer || ![self processBuffer:indexBuffer]) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_index_buffer"];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    id mtlIndexBuffer = (__bridge id)(indexBuffer->data.mtl_data);
    if (!mtlIndexBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_no_mtl_index"];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    MGLDrawCommand *firstCmd = &batch->commands[0];
    uint32_t primType = (uint32_t)batch->key.primitive_type;

    mglBatchDrawIndexedPrimitives(
        encCtx->render_encoder_owner, primType,
        (NSUInteger)batch->stream_index_count,
        MGL_DRAW_INDEX_UINT32, mtlIndexBuffer, 0, 1, 0,
        firstCmd->baseInstance);
    [self traceReplayCommand:batch
                     command:firstCmd
                     context:glm_ctx
                     flushId:_renderPassManager.state->traceReplayFlushId
                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                commandIndex:0
                       phase:"SUBMIT"
                      reason:"stream_direct_merged"];
}


@end
