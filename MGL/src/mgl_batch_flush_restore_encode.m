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
    [self mglActivateReplayStateForContext:glm_ctx];
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    GLenum replayError = (GLenum)mglRenderErrorNone();

    @try {

    MGLStateKey lastKey;
    BOOL lastKeyValid = NO;
    BOOL lastExecuteOk = NO;
    BOOL lastWasStreamBatch = NO;  /* force VAO/buffer dirty after stream */
    memset(&lastKey, 0, sizeof(lastKey));

    for (uint32_t b = 0; b < cb->batch_count; b++) {
        @autoreleasepool {
            MGLDrawBatch *batch = &cb->batches[b];
            if (batch->command_count == 0)
                continue;

            BOOL wantAbsoluteVertexOffsets =
                batch->has_dynamic_vertex_bindings ? YES : NO;

            {
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
            mgl_batch_mtl_restore_note_skip_fail_perf(skipDec);
            if (mgl_batch_restore_oracle_would_skip(
                    _batching.skipSameKeyRestoreEnabled ? 1 : 0,
                    lastKeyValid ? 1 : 0, lastExecuteOk ? 1 : 0,
                    mglStateKeysEqual(&batch->key, &lastKey) ? 1 : 0) &&
                mglEnvFlagEnabled("MGL_SKIP_SAME_KEY_ORACLE")) {
                MGL_PERF_INC(g_mglSameKeyOracleWouldSkipSinceSwap);
            }

            if (canSkipRestore) {
                if (_activeState != glm_ctx->active_state) {
                    _activeState = glm_ctx->active_state;
                }
                MGL_STATE(glm_ctx)->dirty_bits = 0;
                MGL_PERF_INC(g_mglSameKeyRestoreSkipsSinceSwap);
            } else {
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
            const int pathPerf =
                mgl_batch_flush_scheduled_path_perf_kind((int)scheduledPath);
            if (pathPerf == MGL_BATCH_FLUSH_PERF_STREAM) {
                MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap);
                MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap, batch->command_count);
                [self issueStreamMergedBatch:batch context:glm_ctx encodeContext:&encCtx];
                lastWasStreamBatch = YES;
            } else if (scheduledPath == MGL_BATCH_PATH_MDI) {
                [self issueMDIBatch:batch context:glm_ctx encodeContext:&encCtx];
            } else if (scheduledPath == MGL_BATCH_PATH_ICB) {
                [self issueIndirectCommandBufferBatch:batch context:glm_ctx encodeContext:&encCtx];
            } else {
                MGL_PERF_INC(g_mglBatchesDirectSinceSwap);
                MGL_PERF_ADD(g_mglDrawDirectSinceSwap, batch->command_count);
                [self issueDirectBatch:batch context:glm_ctx encodeContext:&encCtx];
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
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    if (batch->state_snapshot) {
        mglCopyHotStateFields(glm_ctx->active_state,
                              (const GLMState *)batch->state_snapshot);
        MGL_PERF_INC(g_mglReplayMemcpyCountSinceSwap);
        mgl_batch_replay_copy_object_hash_tables(MGL_STATE(glm_ctx), savedState);
        mglRestoreProgramPipelinePair(glm_ctx, MGL_STATE(glm_ctx)->program_name,
                                     MGL_STATE(glm_ctx)->var.program_pipeline_binding);
    } else {
        [self restoreStateFromKey:&batch->key context:glm_ctx];
    }
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

    MGLBatchDirtyDeltaFlags dflags;
    memset(&dflags, 0, sizeof(dflags));
    if (canDelta) {
        replayDirtyBits = mgl_batch_mtl_restore_plan_delta_dirty(
            1, prevKey, &batch->key, kMGLFullReplayDirtyBits, &dflags);
        mgl_batch_mtl_restore_note_delta_perf(&dflags);
    }
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
    replayDirtyBits = mgl_batch_restore_finish_dirty(
        replayDirtyBits, forcedDirtyBits, kMGLFullReplayDirtyBits, DIRTY_FBO,
        &fboIn);
    mglMarkRendererDirtyBits(glm_ctx->active_state, replayDirtyBits);
    MGL_SIGNPOST_END(RestoreStateForBatch);
}

- (void)teardownBatchReplayForContext:(GLMContext)glm_ctx
                           savedState:(const GLMState *)savedState
                           savedError:(GLenum)savedError
                          replayError:(GLenum)replayError
{
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
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

- (BOOL)mglCheckSkip:(MGLDrawBatch *)batch
             context:(GLMContext)glm_ctx
             flushId:(uint64_t)flushId
          batchIndex:(uint32_t)batchIndex
               phase:(const char *)phase
              reason:(const char *)reason
     skippedCommands:(uint32_t *)skippedCommands
{
    [self mglTraceSkipBatchCommands:batch context:glm_ctx flushId:flushId
                         batchIndex:batchIndex phase:phase reason:reason
                    skippedCommands:skippedCommands];
    return NO;
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

    if (![self prepareRenderPassIfFBOChanged:batch context:glm_ctx
                                 replayError:replayError]) {
        return [self mglCheckSkip:batch context:glm_ctx flushId:flushId
                       batchIndex:batchIndex phase:"SKIP_FBO_ROTATION"
                           reason:"fbo_rotation"
                  skippedCommands:skippedCommands];
    }
    if ([self processGLState:true] == false) {
        if (!mglRenderErrorIsNone((uint32_t)MGL_STATE(glm_ctx)->error)) {
            *replayError = MGL_STATE(glm_ctx)->error;
        }
        return [self mglCheckSkip:batch context:glm_ctx flushId:flushId
                       batchIndex:batchIndex phase:"SKIP_PROCESS_STATE"
                           reason:"processGLState"
                  skippedCommands:skippedCommands];
    }
    MGLEncodeContext samplerEncCtx = {
        .render_encoder_owner =
            _renderPassManager.state->currentRenderEncoderOwner,
    };
    if (mgl_batch_issue_should_apply_stable_sampler(
            batch->sampler_snapshots_mixed ? 1 : 0, batch->sampler_snapshot_id,
            MGL_INVALID_SAMPLER_SNAPSHOT_ID) &&
        ![self applySamplerSnapshotForCommand:&batch->commands[0]
                                      context:glm_ctx
                                encodeContext:&samplerEncCtx]) {
        return [self mglCheckSkip:batch context:glm_ctx flushId:flushId
                       batchIndex:batchIndex phase:"SKIP_SAMPLER_SNAPSHOT"
                           reason:"sampler_snapshot"
                  skippedCommands:skippedCommands];
    }
    [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                batchIndex:batchIndex phase:"READY"];
    if ([self currentDrawRasterizationIsEmpty]) {
        return [self mglCheckSkip:batch context:glm_ctx flushId:flushId
                       batchIndex:batchIndex phase:"SKIP_EMPTY_RASTER"
                           reason:"empty_rasterization"
                  skippedCommands:skippedCommands];
    }
    GLenum mode = batch->commands[0].mode;
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return [self mglCheckSkip:batch context:glm_ctx flushId:flushId
                       batchIndex:batchIndex phase:"SKIP_FULLY_CULLED"
                           reason:"front_and_back_culled"
                  skippedCommands:skippedCommands];
    }
    [self applyPolygonOffsetForDrawMode:mode];
    return YES;
}

- (void)recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    MGLBatchCmdFrameStats st;
    mgl_batch_flush_accum_cmd_frame_stats(batch, &st);
    MGL_FRAME_ADD(g_mglDrawArraysSinceSwap, st.array_draws);
    MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap, st.array_vertices);
    MGL_FRAME_ADD(g_mglDrawElementsSinceSwap, st.element_draws);
    MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap, st.element_indices);
    [self markCurrentFramebufferDrawAttachmentsWritten];
    (void)glm_ctx;
}

- (void)mglTraceStreamCmd0:(MGLDrawBatch *)batch
                   context:(GLMContext)glm_ctx
                     phase:(const char *)phase
                    reason:(const char *)reason
{
    if (!batch || batch->command_count == 0) {
        return;
    }
    [self traceReplayCommand:batch
                     command:&batch->commands[0]
                     context:glm_ctx
                     flushId:_renderPassManager.state->traceReplayFlushId
                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                commandIndex:0
                       phase:phase
                      reason:reason];
}

- (void)issueStreamMergedBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                 encodeContext:(const MGLEncodeContext *)encCtx
{
    const int streamPath = mgl_batch_replay_stream_path(
        batch, mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1 : 0);
    if (streamPath == MGL_BATCH_STREAM_EMPTY) {
        [self mglTraceStreamCmd0:batch context:glm_ctx phase:"SKIP"
                          reason:mgl_batch_replay_stream_path_reason(streamPath)];
        return;
    }
    if (streamPath == MGL_BATCH_STREAM_BAD_PRIM) {
        [self mglTraceStreamCmd0:batch context:glm_ctx phase:"FALLBACK"
                          reason:mgl_batch_replay_stream_path_reason(streamPath)];
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }
    if (streamPath == MGL_BATCH_STREAM_TRY_MDI) {
        [self mglTraceStreamCmd0:batch context:glm_ctx phase:"ISSUE"
                          reason:mgl_batch_replay_stream_path_reason(streamPath)];
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
        [self mglTraceStreamCmd0:batch context:glm_ctx phase:"FALLBACK"
                          reason:mgl_batch_issue_stream_index_reason(indexReady)];
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    MGLDrawCommand *firstCmd = &batch->commands[0];
    (void)mgl_batch_mtl_draw_indexed(
        encCtx->render_encoder_owner, (uint32_t)batch->key.primitive_type,
        (uint64_t)batch->stream_index_count, MGL_DRAW_INDEX_UINT32,
        (__bridge void *)mtlIndexBuffer, 0, 1, 0,
        (uint64_t)firstCmd->baseInstance);
    [self mglTraceStreamCmd0:batch context:glm_ctx phase:"SUBMIT"
                      reason:mgl_batch_issue_stream_index_reason(indexReady)];
}




@end
