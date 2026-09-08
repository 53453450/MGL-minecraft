/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3: flush/restore/check/stream/schedule encode split from Batch.m
 * (cluster metric). Same (Batch) category; plans in mgl_batch_restore /
 * mgl_batch_issue / mgl_batch_path. Not mgl_render.cpp / metal_port /
 * replay_trace shell growth.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_batch_path.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_restore.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"

#include <string.h>

@implementation MGLRenderer (Batch)

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
    MGLBatchFlushPathStats pathStats;
    memset(&pathStats, 0, sizeof(pathStats));
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
                const GLuint absoluteContractDirty =
                    mgl_batch_restore_absolute_contract_dirty(
                        wantAbsoluteVertexOffsets ? 1 : 0,
                        _batching.absoluteVertexBindingOffsets ? 1 : 0,
                        (DIRTY_VAO | DIRTY_BUFFER));
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
            mgl_batch_flush_accum_path(&pathStats, (int)scheduledPath,
                                       batch->command_count);
            const char *issuePhase = mgl_batch_flush_path_phase((int)scheduledPath);
            [self traceReplayBatch:batch context:glm_ctx flushId:flushHit
                        batchIndex:b phase:issuePhase];
            switch (scheduledPath) {
                case MGL_BATCH_PATH_STREAM_MERGE:
                    MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap);
                    MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap,
                                 batch->command_count);
                    [self issueStreamMergedBatch:batch context:glm_ctx encodeContext:&encCtx];
                    /* Stream transient VAO/buffer pollution: keep lastKey for
                     * delta narrowing; force VAO/buffer on next restore. */
                    lastWasStreamBatch = YES;
                    break;
                case MGL_BATCH_PATH_MDI:
                    [self issueMDIBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
                case MGL_BATCH_PATH_ICB:
                    [self issueIndirectCommandBufferBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
                default:
                    MGL_PERF_INC(g_mglBatchesDirectSinceSwap);
                    MGL_PERF_ADD(g_mglDrawDirectSinceSwap, batch->command_count);
                    [self issueDirectBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
            }

            [self recordBatchCommandStats:batch context:glm_ctx];
        } /* sequentialBatch */
        }
    }
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglTraceNowSeconds());
    if (mgl_batch_flush_should_trace_log(
            flushHit, cb->total_commands, kMGLDiagnosticStateLogs ? 1 : 0,
            skippedCommandCount,
            mglRenderErrorIsNone((uint32_t)replayError) ? 0 : 1)) {
        mglTraceLogNSString(@"MGL TRACE flushDrawBuffer hit=%llu batches=%u totalCommands=%u arrays=%u elements=%u streamMergedBatches=%u streamMergedCommands=%u mdiBatches=%u mdiCommands=%u icbBatches=%u icbCommands=%u directBatches=%u directCommands=%u skippedCommands=%u",
              (unsigned long long)flushHit,
              cb->batch_count, cb->total_commands,
              cb->array_cmd_count, cb->element_cmd_count,
              pathStats.stream_batches, pathStats.stream_commands,
              pathStats.mdi_batches, pathStats.mdi_commands,
              pathStats.icb_batches, pathStats.icb_commands,
              pathStats.direct_batches, pathStats.direct_commands,
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
    /* A3: batch flags → mgl_batch_fill_select_inputs_*; path in C. */
    MGLBatchSelectInputs in = {0};
    if (!batch) {
        return (MGLBatchPath)mgl_batch_select_path(&in);
    }
    mgl_batch_fill_select_inputs_from_batch_flags(
        batch->command_count, batch->sampler_snapshots_mixed ? 1 : 0,
        batch->stream_merged ? 1 : 0,
        batch->has_dynamic_uniform_bindings ? 1 : 0,
        batch->has_dynamic_vertex_bindings ? 1 : 0,
        batch->has_dynamic_texture_bindings ? 1 : 0,
        batch->mdi_compatible ? 1 : 0, batch->uses_elements ? 1 : 0,
        batch->key.primitive_type, &in);
    MGLBatchIcbConfig icb = mgl_batch_icb_config();
    in.enable_icb = icb.enable;
    in.disable_icb = icb.disable;
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

    const GLuint kMGLFullReplayDirtyBits = mgl_batch_restore_full_dirty_bits();
    GLuint replayDirtyBits = kMGLFullReplayDirtyBits;
    BOOL prevKeyValid = (prevKey != NULL);
    BOOL canDelta = mgl_batch_restore_can_delta(
                        _batching.dirtyKeyDeltaEnabled ? 1 : 0,
                        prevKeyValid ? 1 : 0,
                        mglRenderEncoderOwnerHasCurrent(
                            _renderPassManager.state->currentRenderEncoderOwner),
                        mglBindingStateIsValid(_bindingStateOwner) ? 1 : 0)
                        ? YES
                        : NO;

    if (canDelta) {
        /* A3: dirty-key domain narrowing in mgl_batch_compute_key_delta_dirty_bits. */
        MGLBatchDirtyDomainMasks masks;
        mgl_batch_restore_default_domain_masks(&masks);
        MGLBatchStateKeyView prevView;
        MGLBatchStateKeyView curView;
        mgl_batch_state_key_view_from_key(prevKey, &prevView);
        mgl_batch_state_key_view_from_key(&batch->key, &curView);
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

    /* A3: FBO dirty fold in mgl_batch_restore_fold_fbo_dirty. */
    Framebuffer *replayFBO = MGL_STATE(glm_ctx)->framebuffer;
    const MGLBatchRestoreFboIn fboIn = {
        .fbo_binding_dirty =
            (replayFBO && (replayFBO->dirty_bits & DIRTY_FBO_BINDING)) ? 1u : 0u,
        .prev_fbo_differs =
            (prevKeyValid && prevKey->fbo_name != batch->key.fbo_name) ? 1u : 0u,
        .has_encoder =
            mglRenderEncoderOwnerHasCurrent(
                _renderPassManager.state->currentRenderEncoderOwner)
                ? 1u
                : 0u,
        .bind_valid = mglBindingStateIsValid(_bindingStateOwner) ? 1u : 0u,
        .pass_matches =
            [self currentRenderPassMatchesCurrentFramebuffer] ? 1u : 0u,
    };
    replayDirtyBits = mgl_batch_restore_fold_fbo_dirty(
        replayDirtyBits, kMGLFullReplayDirtyBits, DIRTY_FBO, &fboIn);
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
    if (mgl_batch_issue_should_apply_stable_sampler(
            batch->sampler_snapshots_mixed ? 1 : 0,
            batch->sampler_snapshot_id,
            MGL_INVALID_SAMPLER_SNAPSHOT_ID) &&
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
    const int processOk = indexBuffer ? ([self processBuffer:indexBuffer] ? 1 : 0) : 0;
    id mtlIndexBuffer =
        (indexBuffer && indexBuffer->data.mtl_data)
            ? (__bridge id)(indexBuffer->data.mtl_data)
            : nil;
    const int indexReady = mgl_batch_issue_stream_index_ready(
        indexBuffer ? 1 : 0, processOk, mtlIndexBuffer ? 1 : 0);
    if (indexReady != MGL_BATCH_STREAM_INDEX_OK) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:mgl_batch_issue_stream_index_reason(indexReady)];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    MGLDrawCommand *firstCmd = &batch->commands[0];
    uint32_t primType = (uint32_t)batch->key.primitive_type;

    (void)mgl_batch_mtl_draw_indexed(
        encCtx->render_encoder_owner, primType,
        (uint64_t)batch->stream_index_count, MGL_DRAW_INDEX_UINT32,
        (__bridge void *)mtlIndexBuffer, 0, 1, 0,
        (uint64_t)firstCmd->baseInstance);
    [self traceReplayCommand:batch
                     command:firstCmd
                     context:glm_ctx
                     flushId:_renderPassManager.state->traceReplayFlushId
                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                commandIndex:0
                       phase:"SUBMIT"
                      reason:mgl_batch_issue_stream_index_reason(indexReady)];
}


@end
