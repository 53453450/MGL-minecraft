/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: flush/restore/check/stream/schedule encode (Batch cluster).
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+BatchPorts_Private.h"
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

typedef struct {
    __unsafe_unretained MGLRenderer *r; GLMContext ctx; const GLMState *saved;
    uint64_t hit; GLenum *err; uint32_t *skipped; MGLStateKey key;
    MGLBatchFlushLoopState *st; MGLEncodeContext enc;
} FCtx;
static MGLDrawBatch *FB(FCtx *c, uint32_t b) { return &c->ctx->draw_command_buffer.batches[b]; }
static uint32_t fCount(void *v) { return ((FCtx *)v)->ctx->draw_command_buffer.batch_count; }
static uint32_t fCmds(void *v, uint32_t b) { return FB(v, b)->command_count; }
static void fSkipIn(void *v, uint32_t b, MGLBatchSameKeySkipIn *in, int *wa)
{
    FCtx *c = v; MGLDrawBatch *batch = FB(c, b);
    int want = batch->has_dynamic_vertex_bindings ? 1 : 0; if (wa) *wa = want;
    in->has_encoder = mglRenderEncoderOwnerHasCurrent(
        mglRendererRenderPassManager(c->r).state->currentRenderEncoderOwner) ? 1u : 0u;
    in->bind_valid = mglBindingStateIsValid(c->r->_bindingStateOwner) ? 1u : 0u;
    in->keys_equal = mglStateKeysEqual(&batch->key, &c->key) ? 1u : 0u;
    in->absolute_offsets_match =
        (want == (c->r->_batching.absoluteVertexBindingOffsets ? 1 : 0)) ? 1u : 0u;
    in->pass_matches = [c->r currentRenderPassMatchesCurrentFramebuffer] ? 1u : 0u;
}
static void fNote(void *v, int d) { (void)v; mgl_batch_mtl_restore_note_skip_fail_perf(d); }
static int fOracleEq(void *v, uint32_t b) { return mglStateKeysEqual(&FB(v, b)->key, &((FCtx *)v)->key); }
static void fOracle(void *v) { (void)v; MGL_PERF_INC(g_mglSameKeyOracleWouldSkipSinceSwap); }
static void fApplySkip(void *v, uint32_t b)
{ (void)b; FCtx *c = v; if (c->r->_core.activeState != c->ctx->active_state)
      c->r->_core.activeState = c->ctx->active_state;
  c->ctx->active_state->dirty_bits = 0; MGL_PERF_INC(g_mglSameKeyRestoreSkipsSinceSwap); }
static void fSetAbs(void *v, int w) { ((FCtx *)v)->r->_batching.absoluteVertexBindingOffsets = w ? YES : NO; }
static void fRestore(void *v, uint32_t b, uint32_t forced)
{ FCtx *c = v; [c->r restoreStateForBatch:FB(c, b) context:c->ctx savedState:c->saved
      prevKey:(c->st->last_key_valid ? &c->key : NULL) forcedDirtyBits:forced]; }
static int fCheck(void *v, uint32_t b)
{ FCtx *c = v; return [c->r checkBatchShouldExecute:FB(c, b) context:c->ctx flushId:c->hit
      batchIndex:b replayError:c->err skippedCommands:c->skipped] ? 1 : 0; }
static void fMark(void *v, uint32_t b)
{ ((FCtx *)v)->key = FB(v, b)->key; MGL_PERF_INC(g_mglBatchesReplayedSinceSwap); }
static int fSched(void *v, uint32_t b)
{ return (int)[((FCtx *)v)->r scheduleDrawBatch:FB(v, b) context:((FCtx *)v)->ctx]; }
static void fTrace(void *v, uint32_t b, const char *ph)
{ FCtx *c = v; [c->r traceReplayBatch:FB(c, b) context:c->ctx flushId:c->hit batchIndex:b phase:ph]; }
static void fPerfS(void *v, uint32_t n)
{ (void)v; MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap); MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap, n); }
static void fPerfD(void *v, uint32_t n)
{ (void)v; MGL_PERF_INC(g_mglBatchesDirectSinceSwap); MGL_PERF_ADD(g_mglDrawDirectSinceSwap, n); }
static void fEnc(FCtx *c)
{ c->enc.render_encoder_owner = mglRendererRenderPassManager(c->r).state->currentRenderEncoderOwner; }
static void fIssS(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); [c->r issueStreamMergedBatch:FB(c, b) context:c->ctx encodeContext:&c->enc]; }
static void fIssM(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); [c->r issueMDIBatch:FB(c, b) context:c->ctx encodeContext:&c->enc]; }
static void fIssI(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); [c->r issueIndirectCommandBufferBatch:FB(c, b) context:c->ctx encodeContext:&c->enc]; }
static void fIssD(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); [c->r issueDirectBatch:FB(c, b) context:c->ctx encodeContext:&c->enc]; }
static void fRec(void *v, uint32_t b)
{ [((FCtx *)v)->r recordBatchCommandStats:FB(v, b) context:((FCtx *)v)->ctx]; }

typedef struct {
    __unsafe_unretained MGLRenderer *r; MGLDrawBatch *batch; GLMContext ctx;
    uint64_t hit; uint32_t bi; GLenum *err; uint32_t *skipped; GLenum mode;
} CCtx;
static void cBegin(void *v)
{ CCtx *c = v; [mglRendererRenderPassManager(c->r) setTraceReplayFlushId:c->hit batchIndex:c->bi];
  [c->r traceReplayBatch:c->batch context:c->ctx flushId:c->hit batchIndex:c->bi phase:"RESTORE"]; }
static int cFbo(void *v)
{ CCtx *c = v; return [c->r prepareRenderPassIfFBOChanged:c->batch context:c->ctx
      replayError:c->err] ? 1 : 0; }
static int cProc(void *v) { return [((CCtx *)v)->r processGLState:true] ? 1 : 0; }
static void cErr(void *v)
{ CCtx *c = v; if (!mglRenderErrorIsNone((uint32_t)c->ctx->active_state->error))
      *c->err = c->ctx->active_state->error; }
static int cShouldS(void *v)
{ CCtx *c = v; return mgl_batch_issue_should_apply_stable_sampler(
      c->batch->sampler_snapshots_mixed ? 1 : 0, c->batch->sampler_snapshot_id,
      MGL_INVALID_SAMPLER_SNAPSHOT_ID); }
static int cApplyS(void *v)
{ CCtx *c = v; MGLEncodeContext e = {.render_encoder_owner =
      mglRendererRenderPassManager(c->r).state->currentRenderEncoderOwner};
  return [c->r applySamplerSnapshotForCommand:&c->batch->commands[0] context:c->ctx
      encodeContext:&e] ? 1 : 0; }
static void cReady(void *v)
{ CCtx *c = v; [c->r traceReplayBatch:c->batch context:c->ctx flushId:c->hit batchIndex:c->bi
      phase:"READY"]; }
static int cEmpty(void *v) { return [((CCtx *)v)->r currentDrawRasterizationIsEmpty] ? 1 : 0; }
static int cCull(void *v)
{ CCtx *c = v; c->mode = c->batch->commands[0].mode;
  return [c->r currentDrawModeIsFullyCulled:c->mode] ? 1 : 0; }
static void cPoly(void *v) { [((CCtx *)v)->r applyPolygonOffsetForDrawMode:((CCtx *)v)->mode]; }
static int cSkip(void *v, const char *ph, const char *rs)
{ CCtx *c = v; return [c->r mglTraceSkipBatchCommands:c->batch context:c->ctx flushId:c->hit
      batchIndex:c->bi phase:ph reason:rs skippedCommands:c->skipped] ? 1 : 0; }

typedef struct {
    __unsafe_unretained MGLRenderer *r; MGLDrawBatch *batch; GLMContext ctx;
    const MGLEncodeContext *enc;
} SCtx;
static void sTr0(void *v, const char *ph, const char *rs)
{ SCtx *c = v; [c->r mglTraceStreamCmd0:c->batch context:c->ctx phase:ph reason:rs]; }
static int sMdi(void *v)
{ SCtx *c = v; return [c->r issueStreamMergedMDIBatch:c->batch context:c->ctx
      encodeContext:c->enc] ? 1 : 0; }
static void sDir(void *v)
{ SCtx *c = v; [c->r issueDirectBatch:c->batch context:c->ctx encodeContext:c->enc]; }
static int sIdx(void *v, void **mtl)
{
    SCtx *c = v; Buffer *ib = (Buffer *)c->batch->stream_index_buffer;
    int pok = ib ? ([c->r processBuffer:ib] ? 1 : 0) : 0;
    id m = (ib && ib->data.mtl_data) ? (__bridge id)ib->data.mtl_data : nil;
    int ready = mgl_batch_issue_stream_index_ready(ib ? 1 : 0, pok, m ? 1 : 0);
    if (ready != MGL_BATCH_STREAM_INDEX_OK) {
        [c->r mglTraceStreamCmd0:c->batch context:c->ctx phase:"FALLBACK"
                          reason:mgl_batch_issue_stream_index_reason(ready)];
        if (mtl) *mtl = NULL; return 0;
    }
    if (mtl) *mtl = (__bridge void *)m; return 1;
}
static void sDraw(void *v, void *mtl)
{
    SCtx *c = v; MGLDrawCommand *cmd = &c->batch->commands[0];
    (void)mgl_batch_mtl_draw_indexed(c->enc->render_encoder_owner,
        (uint32_t)c->batch->key.primitive_type, (uint64_t)c->batch->stream_index_count,
        MGL_DRAW_INDEX_UINT32, mtl, 0, 1, 0, (uint64_t)cmd->baseInstance);
    [c->r mglTraceStreamCmd0:c->batch context:c->ctx phase:"SUBMIT"
                      reason:mgl_batch_issue_stream_index_reason(MGL_BATCH_STREAM_INDEX_OK)];
}

typedef struct {
    __unsafe_unretained MGLRenderer *r; MGLDrawBatch *batch; GLMContext ctx;
    uint64_t hit; uint32_t bi; const char *reason;
} SkipCtx;
static void skipTraceCmd(void *v, uint32_t i)
{ SkipCtx *c = v; [c->r traceReplayCommand:c->batch command:&c->batch->commands[i]
      context:c->ctx flushId:c->hit batchIndex:c->bi commandIndex:i phase:"SKIP" reason:c->reason]; }



- (void)flushDrawBufferLocked:(GLMContext)glm_ctx
{
    ctx = glm_ctx;
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cb->batch_count == 0) return;
    _currentCBHasWork = YES;
    static uint64_t s_hit = 0; uint64_t hit = ++s_hit; uint32_t skipped = 0;
    GLMState saved; memcpy(&saved, glm_ctx->active_state, sizeof(saved));
    GLenum savedError = saved.error;
    [self mglActivateReplayStateForContext:glm_ctx];
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    GLenum replayError = (GLenum)mglRenderErrorNone();
    @try {
        MGLBatchFlushLoopState st; memset(&st, 0, sizeof(st));
        FCtx fc = {.r = self, .ctx = glm_ctx, .saved = &saved, .hit = hit,
                   .err = &replayError, .skipped = &skipped, .st = &st};
        memset(&fc.key, 0, sizeof(fc.key));
        MGLBatchFlushLoopOps ops = {
            .ctx = &fc, .batch_count = fCount, .command_count = fCmds,
            .fill_skip_in = fSkipIn, .note_skip_perf = fNote,
            .oracle_keys_equal = fOracleEq, .on_oracle_would_skip = fOracle,
            .apply_same_key_skip = fApplySkip, .set_absolute_offsets = fSetAbs,
            .restore = fRestore, .check_execute = fCheck, .mark_execute_ok = fMark,
            .schedule = fSched, .trace_phase = fTrace, .perf_stream = fPerfS,
            .perf_direct = fPerfD, .issue_stream = fIssS, .issue_mdi = fIssM,
            .issue_icb = fIssI, .issue_direct = fIssD, .record_stats = fRec,
            .vao_buffer_dirty_mask = (DIRTY_VAO | DIRTY_BUFFER),
            .skip_enabled = _batching.skipSameKeyRestoreEnabled ? 1u : 0u,
            .oracle_env_enabled = mglEnvFlagEnabled("MGL_SKIP_SAME_KEY_ORACLE") ? 1u : 0u,
        };
        mgl_batch_flush_run_batches(&st, &ops);
        MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglTraceNowSeconds());
        if (mgl_batch_flush_should_trace_log(
                hit, cb->total_commands, kMGLDiagnosticStateLogs ? 1 : 0, skipped,
                mglRenderErrorIsNone((uint32_t)replayError) ? 0 : 1)) {
            mglTraceLogNSString(
                @"MGL TRACE flushDrawBuffer hit=%llu batches=%u totalCommands=%u "
                @"arrays=%u elements=%u streamMergedBatches=%u streamMergedCommands=%u "
                @"mdiBatches=%u mdiCommands=%u icbBatches=%u icbCommands=%u "
                @"directBatches=%u directCommands=%u skippedCommands=%u",
                (unsigned long long)hit, cb->batch_count, cb->total_commands,
                cb->array_cmd_count, cb->element_cmd_count,
                st.path_stats.stream_batches, st.path_stats.stream_commands,
                st.path_stats.mdi_batches, st.path_stats.mdi_commands,
                st.path_stats.icb_batches, st.path_stats.icb_commands,
                st.path_stats.direct_batches, st.path_stats.direct_commands, skipped);
        }
    } @finally {
        [_renderPassManager setTraceReplayFlushId:0 batchIndex:0];
        [self teardownBatchReplayForContext:glm_ctx savedState:&saved
                                savedError:savedError replayError:replayError];
    }
}

- (MGLBatchPath)scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    MGLBatchSelectInputs in = {0};
    if (!batch) return (MGLBatchPath)mgl_batch_select_path(&in);
    mgl_batch_fill_select_inputs_from_batch_flags(
        batch->command_count, batch->sampler_snapshots_mixed ? 1 : 0,
        batch->stream_merged ? 1 : 0, batch->has_dynamic_uniform_bindings ? 1 : 0,
        batch->has_dynamic_vertex_bindings ? 1 : 0,
        batch->has_dynamic_texture_bindings ? 1 : 0, batch->mdi_compatible ? 1 : 0,
        batch->uses_elements ? 1 : 0, batch->key.primitive_type, &in);
    MGLBatchIcbConfig icb = mgl_batch_icb_config();
    in.enable_icb = icb.enable; in.disable_icb = icb.disable;
    in.disable_mdi = mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1u : 0u;
    if (@available(macOS 10.14, *)) in.icb_os_supported = 1u;
    Program *vertexProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    if (vertexProgram && vertexProgram->uses_cull_distance) in.uses_cull_distance = 1u;
    if (batch->command_count > 0u) {
        in.polygon_mode_point =
            mglPolygonModePointForDrawMode(glm_ctx, batch->commands[0].mode) ? 1u : 0u;
        if (batch->uses_elements) {
            uint32_t dummy = 0u;
            in.primitive_restart =
                mglPrimitiveRestartIndexForType(glm_ctx, batch->commands[0].indexType, &dummy)
                    ? 1u : 0u;
        }
    }
    return (MGLBatchPath)mgl_batch_select_path(&in);
}

- (void)restoreStateForBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                  savedState:(const GLMState *)savedState
{
    [self restoreStateForBatch:batch context:glm_ctx savedState:savedState prevKey:NULL
               forcedDirtyBits:0u];
}

- (void)restoreStateForBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                  savedState:(const GLMState *)savedState prevKey:(const MGLStateKey *)prevKey
             forcedDirtyBits:(GLuint)forcedDirtyBits
{
    MGL_SIGNPOST_BEGIN(RestoreStateForBatch);
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    if (batch->state_snapshot) {
        mglCopyHotStateFields(glm_ctx->active_state, (const GLMState *)batch->state_snapshot);
        MGL_PERF_INC(g_mglReplayMemcpyCountSinceSwap);
        mgl_batch_replay_copy_object_hash_tables(glm_ctx->active_state, savedState);
        mglRestoreProgramPipelinePair(glm_ctx, glm_ctx->active_state->program_name,
                                      glm_ctx->active_state->var.program_pipeline_binding);
    } else {
        [self restoreStateFromKey:&batch->key context:glm_ctx];
    }
    _activeState = glm_ctx->active_state; glm_ctx->active_state->dirty_bits = 0;
    const GLuint kFull = mgl_batch_restore_full_dirty_bits();
    GLuint replayDirtyBits = kFull;
    BOOL prevKeyValid = (prevKey != NULL);
    BOOL canDelta = mgl_batch_restore_can_delta(
                        _batching.dirtyKeyDeltaEnabled ? 1 : 0, prevKeyValid ? 1 : 0,
                        mglRenderEncoderOwnerHasCurrent(
                            _renderPassManager.state->currentRenderEncoderOwner),
                        mglBindingStateIsValid(_bindingStateOwner) ? 1 : 0) ? YES : NO;
    MGLBatchDirtyDeltaFlags dflags; memset(&dflags, 0, sizeof(dflags));
    if (canDelta) {
        replayDirtyBits = mgl_batch_mtl_restore_plan_delta_dirty(
            1, prevKey, &batch->key, kFull, &dflags);
        mgl_batch_mtl_restore_note_delta_perf(&dflags);
    }
    Framebuffer *replayFBO = glm_ctx->active_state->framebuffer;
    const MGLBatchRestoreFboIn fboIn = {
        .fbo_binding_dirty =
            (replayFBO && (replayFBO->dirty_bits & DIRTY_FBO_BINDING)) ? 1u : 0u,
        .prev_fbo_differs =
            (prevKeyValid && prevKey->fbo_name != batch->key.fbo_name) ? 1u : 0u,
        .has_encoder = mglRenderEncoderOwnerHasCurrent(
                           _renderPassManager.state->currentRenderEncoderOwner) ? 1u : 0u,
        .bind_valid = mglBindingStateIsValid(_bindingStateOwner) ? 1u : 0u,
        .pass_matches = [self currentRenderPassMatchesCurrentFramebuffer] ? 1u : 0u,
    };
    replayDirtyBits = mgl_batch_restore_finish_dirty(replayDirtyBits, forcedDirtyBits, kFull,
                                                     DIRTY_FBO, &fboIn);
    mglMarkRendererDirtyBits(glm_ctx->active_state, replayDirtyBits);
    MGL_SIGNPOST_END(RestoreStateForBatch);
}

- (void)teardownBatchReplayForContext:(GLMContext)glm_ctx
                           savedState:(const GLMState *)savedState
                           savedError:(GLenum)savedError replayError:(GLenum)replayError
{
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    const BOOL usedReplayWorkspace = (glm_ctx->active_state == &glm_ctx->replay_state);
    if (usedReplayWorkspace)
        mgl_batch_replay_sync_hash_tables_from_replay(&glm_ctx->state, &glm_ctx->replay_state);
    [self mglRestoreLiveActiveStateForContext:glm_ctx];
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    _batching.absoluteVertexBindingOffsets = NO;
    mglResetCommandBufferForContext(glm_ctx, &glm_ctx->draw_command_buffer);
    if (_batching.arenaSnapshotEnabled) mglResetBatchArena(&_batching.batchArena);
    if (!usedReplayWorkspace) memcpy(glm_ctx->active_state, savedState, sizeof(GLMState));
    mglClearStateDirtyBitsPreservingHashInvalidation(glm_ctx->active_state);
    mglRestoreProgramPipelinePair(glm_ctx, glm_ctx->active_state->program_name,
                                  glm_ctx->active_state->var.program_pipeline_binding);
    if (mglRenderErrorIsNone((uint32_t)savedError) &&
        !mglRenderErrorIsNone((uint32_t)replayError))
        glm_ctx->active_state->error = replayError;
    (void)savedState;
}

- (BOOL)mglTraceSkipBatchCommands:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                          flushId:(uint64_t)flushId batchIndex:(uint32_t)batchIndex
                            phase:(const char *)phase reason:(const char *)reason
                  skippedCommands:(uint32_t *)skippedCommands
{
    [self traceReplayBatch:batch context:glm_ctx flushId:flushId batchIndex:batchIndex
                     phase:phase];
    SkipCtx sc = {.r = self, .batch = batch, .ctx = glm_ctx, .hit = flushId, .bi = batchIndex,
                  .reason = reason};
    mgl_batch_flush_trace_skip_commands(batch->command_count, skipTraceCmd, &sc,
                                        skippedCommands);
    MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
    return NO;
}

- (BOOL)checkBatchShouldExecute:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                        flushId:(uint64_t)flushId batchIndex:(uint32_t)batchIndex
                    replayError:(GLenum *)replayError
                skippedCommands:(uint32_t *)skippedCommands
{
    CCtx c = {.r = self, .batch = batch, .ctx = glm_ctx, .hit = flushId, .bi = batchIndex,
              .err = replayError, .skipped = skippedCommands,
              .mode = batch && batch->command_count ? batch->commands[0].mode : 0};
    MGLBatchCheckExecOps ops = {
        .ctx = &c, .begin_trace = cBegin, .prepare_fbo = cFbo, .process_gl_state = cProc,
        .capture_error_if_any = cErr, .should_apply_sampler = cShouldS,
        .apply_sampler = cApplyS, .trace_ready = cReady, .empty_raster = cEmpty,
        .fully_culled = cCull, .apply_polygon_offset = cPoly, .trace_skip = cSkip,
    };
    return mgl_batch_check_should_execute(&ops) ? YES : NO;
}

- (void)recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    MGLBatchCmdFrameStats st; mgl_batch_flush_accum_cmd_frame_stats(batch, &st);
    MGL_FRAME_ADD(g_mglDrawArraysSinceSwap, st.array_draws);
    MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap, st.array_vertices);
    MGL_FRAME_ADD(g_mglDrawElementsSinceSwap, st.element_draws);
    MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap, st.element_indices);
    [self markCurrentFramebufferDrawAttachmentsWritten];
    (void)glm_ctx;
}

- (void)mglTraceStreamCmd0:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                     phase:(const char *)phase reason:(const char *)reason
{
    if (!batch || batch->command_count == 0) return;
    [self traceReplayCommand:batch command:&batch->commands[0] context:glm_ctx
                     flushId:_renderPassManager.state->traceReplayFlushId
                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                commandIndex:0 phase:phase reason:reason];
}

- (void)issueStreamMergedBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                 encodeContext:(const MGLEncodeContext *)encCtx
{
    SCtx c = {.r = self, .batch = batch, .ctx = glm_ctx, .enc = encCtx};
    MGLBatchStreamMergedOps ops = {
        .ctx = &c, .trace_cmd0 = sTr0, .try_stream_mdi = sMdi, .issue_direct = sDir,
        .resolve_stream_index = sIdx, .draw_stream_indexed = sDraw,
    };
    mgl_batch_issue_stream_merged(batch, mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1 : 0, &ops);
}

@end
