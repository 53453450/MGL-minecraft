/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: flush/restore/check/stream/schedule encode (Batch cluster).
 *
 * Formerly mgl_batch_flush_restore_encode.m.  The loops and ops callbacks are
 * C now; the renderer operations they drive go through mgl_renderer_ports.h,
 * and the one ObjC-only frame (the @try/@finally that must tear the replay
 * workspace down even when a draw raises) stays in the shim as
 * mglRendererFlushDrawBufferLockedPort.
 */
#include "mgl_renderer_ports.h"   /* C port surface (T4) */
#include "mgl_draw_issue.h"       /* mglDrawHost{BindContext,RasterizationIsEmpty,...} */
#include "mgl_batch_restore.h"
#include "mgl_batch_rt_mark.h"
#include "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_batch_path.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"
#include <string.h>

typedef struct {
    void *r; GLMContext ctx; const GLMState *saved;
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
        mglRendererCurrentRenderEncoderOwnerPort(c->r)) ? 1u : 0u;
    in->bind_valid = mglRendererBindingStateIsValidPort(c->r) ? 1u : 0u;
    in->keys_equal = mglStateKeysEqual(&batch->key, &c->key) ? 1u : 0u;
    const MGLBatchingState *bs = mglRendererBatchingStatePort(c->r);
    in->absolute_offsets_match =
        (want == (bs && bs->absoluteVertexBindingOffsets ? 1 : 0)) ? 1u : 0u;
    in->pass_matches = mglRendererCurrentRenderPassMatchesFramebufferPort(c->r) ? 1u : 0u;
}
static void fNote(void *v, int d) { (void)v; mgl_batch_mtl_restore_note_skip_fail_perf(d); }
static int fOracleEq(void *v, uint32_t b) { return mglStateKeysEqual(&FB(v, b)->key, &((FCtx *)v)->key); }
static void fOracle(void *v) { (void)v; MGL_PERF_INC(g_mglSameKeyOracleWouldSkipSinceSwap); }
static void fApplySkip(void *v, uint32_t b)
{ (void)b; FCtx *c = v; mglRendererSetActiveStatePort(c->r, c->ctx);
  c->ctx->active_state->dirty_bits = 0; MGL_PERF_INC(g_mglSameKeyRestoreSkipsSinceSwap); }
static void fSetAbs(void *v, int w)
{ MGLBatchingState *bs = mglRendererBatchingStatePort(((FCtx *)v)->r);
  if (bs) bs->absoluteVertexBindingOffsets = w ? 1u : 0u; }
static void fRestore(void *v, uint32_t b, uint32_t forced)
{ FCtx *c = v; mglBatchRestoreStateForBatch(c->r, FB(c, b), c->ctx, c->saved,
      (c->st->last_key_valid ? &c->key : NULL), forced); }
static int fCheck(void *v, uint32_t b)
{ FCtx *c = v; return mglBatchCheckShouldExecute(c->r, FB(c, b), c->ctx, c->hit, b,
      c->err, c->skipped); }
static void fMark(void *v, uint32_t b)
{ ((FCtx *)v)->key = FB(v, b)->key; MGL_PERF_INC(g_mglBatchesReplayedSinceSwap); }
static int fSched(void *v, uint32_t b)
{ return (int)mglBatchScheduleDrawBatch(((FCtx *)v)->r, FB(v, b), ((FCtx *)v)->ctx); }
static void fTrace(void *v, uint32_t b, const char *ph)
{ FCtx *c = v; mglBatchTraceReplayBatch(c->r, FB(c, b), c->ctx, c->hit, b, ph); }
static void fPerfS(void *v, uint32_t n)
{ (void)v; MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap); MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap, n); }
static void fPerfD(void *v, uint32_t n)
{ (void)v; MGL_PERF_INC(g_mglBatchesDirectSinceSwap); MGL_PERF_ADD(g_mglDrawDirectSinceSwap, n); }
static void fEnc(FCtx *c)
{ c->enc.render_encoder_owner = mglRendererCurrentRenderEncoderOwnerPort(c->r); }
static void fIssS(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); mglBatchIssueStreamMergedBatch(c->r, FB(c, b), c->ctx, &c->enc); }
static void fIssM(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); mglBatchIssueMDIBatch(c->r, FB(c, b), c->ctx, &c->enc); }
static void fIssI(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); mglBatchIssueIndirectCommandBufferBatch(c->r, FB(c, b), c->ctx, &c->enc); }
static void fIssD(void *v, uint32_t b)
{ FCtx *c = v; fEnc(c); mglBatchIssueDirectBatch(c->r, FB(c, b), c->ctx, &c->enc); }
static void fRec(void *v, uint32_t b)
{ mglBatchRecordCommandStats(((FCtx *)v)->r, FB(v, b), ((FCtx *)v)->ctx); }

typedef struct {
    void *r; MGLDrawBatch *batch; GLMContext ctx;
    uint64_t hit; uint32_t bi; GLenum *err; uint32_t *skipped; GLenum mode;
} CCtx;
static void cBegin(void *v)
{ CCtx *c = v; mglRendererTraceReplaySetPort(c->r, c->hit, c->bi);
  mglBatchTraceReplayBatch(c->r, c->batch, c->ctx, c->hit, c->bi, "RESTORE"); }
static int cFbo(void *v)
{ CCtx *c = v; return mglRendererPrepareRenderPassIfFBOChangedPort(c->r, c->batch, c->ctx,
      c->err); }
static int cProc(void *v) { return mglRendererProcessGLStatePort(((CCtx *)v)->r, 1); }
static void cErr(void *v)
{ CCtx *c = v; if (!mglRenderErrorIsNone((uint32_t)c->ctx->active_state->error))
      *c->err = c->ctx->active_state->error; }
static int cShouldS(void *v)
{ CCtx *c = v; return mgl_batch_issue_should_apply_stable_sampler(
      c->batch->sampler_snapshots_mixed ? 1 : 0, c->batch->sampler_snapshot_id,
      MGL_INVALID_SAMPLER_SNAPSHOT_ID); }
static int cApplyS(void *v)
{ CCtx *c = v; MGLEncodeContext e = {.render_encoder_owner =
      mglRendererCurrentRenderEncoderOwnerPort(c->r)};
  return mglBatchApplySamplerSnapshot(c->r, &c->batch->commands[0], c->ctx, &e) ? 1 : 0; }
static void cReady(void *v)
{ CCtx *c = v; mglBatchTraceReplayBatch(c->r, c->batch, c->ctx, c->hit, c->bi, "READY"); }
static int cEmpty(void *v) { return mglDrawHostRasterizationIsEmpty(((CCtx *)v)->r) ? 1 : 0; }
static int cCull(void *v)
{ CCtx *c = v; c->mode = c->batch->commands[0].mode;
  return mglDrawHostModeFullyCulled(c->r, c->mode) ? 1 : 0; }
static void cPoly(void *v)
{ CCtx *c = v; mglDrawHostApplyPolygonOffset(c->r, c->mode); }
static int cSkip(void *v, const char *ph, const char *rs)
{ CCtx *c = v; return mglBatchTraceSkipCommands(c->r, c->batch, c->ctx, c->hit, c->bi,
      ph, rs, c->skipped); }

typedef struct {
    void *r; MGLDrawBatch *batch; GLMContext ctx;
    const MGLEncodeContext *enc;
} SCtx;
static void sTr0(void *v, const char *ph, const char *rs)
{ SCtx *c = v; mglBatchTraceStreamCmd0(c->r, c->batch, c->ctx, ph, rs); }
static int sMdi(void *v)
{ SCtx *c = v; return mglBatchIssueStreamMergedMDIBatch(c->r, c->batch, c->ctx, c->enc) ? 1 : 0; }
static void sDir(void *v)
{ SCtx *c = v; mglBatchIssueDirectBatch(c->r, c->batch, c->ctx, c->enc); }
static int sIdx(void *v, void **mtl)
{
    SCtx *c = v; Buffer *ib = (Buffer *)c->batch->stream_index_buffer;
    int pok = ib ? mglRendererProcessBufferPort(c->r, ib) : 0;
    void *m = ib ? ib->data.mtl_data : NULL;
    int ready = mgl_batch_issue_stream_index_ready(ib ? 1 : 0, pok, m ? 1 : 0);
    if (ready != MGL_BATCH_STREAM_INDEX_OK) {
        mglBatchTraceStreamCmd0(c->r, c->batch, c->ctx, "FALLBACK",
                               mgl_batch_issue_stream_index_reason(ready));
        if (mtl) *mtl = NULL; return 0;
    }
    if (mtl) *mtl = m; return 1;
}
static void sDraw(void *v, void *mtl)
{
    SCtx *c = v; MGLDrawCommand *cmd = &c->batch->commands[0];
    (void)mgl_batch_mtl_draw_indexed(c->enc->render_encoder_owner,
        (uint32_t)c->batch->key.primitive_type, (uint64_t)c->batch->stream_index_count,
        MGL_DRAW_INDEX_UINT32, mtl, 0, 1, 0, (uint64_t)cmd->baseInstance);
    mglBatchTraceStreamCmd0(c->r, c->batch, c->ctx, "SUBMIT",
                            mgl_batch_issue_stream_index_reason(MGL_BATCH_STREAM_INDEX_OK));
}

typedef struct {
    void *r; MGLDrawBatch *batch; GLMContext ctx;
    uint64_t hit; uint32_t bi; const char *reason;
} SkipCtx;
static void skipTraceCmd(void *v, uint32_t i)
{ SkipCtx *c = v; mglBatchTraceReplayCommand(c->r, c->batch,
      &c->batch->commands[i], c->ctx, c->hit, c->bi, i, "SKIP", c->reason); }


/* Body of the former flushDrawBufferLocked: up to (not including) the @try --
 * bind the context, take the batch snapshot and switch to the replay workspace.
 * Returns 0 when there is nothing to replay. */
int mglBatchFlushBegin(void *renderer, GLMContext glm_ctx, MGLBatchFlushPass *pass)
{
    if (!renderer || !glm_ctx || !pass) return 0;
    if (!mglDrawHostBindContext(renderer, glm_ctx)) return 0;
    mglRendererAssertDualProxyPort(renderer, glm_ctx);
    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cb->batch_count == 0) return 0;
    mglRendererSetCurrentCBHasWorkPort(renderer, 1);
    static uint64_t s_hit = 0;
    pass->hit = ++s_hit;
    pass->skipped = 0;
    memcpy(&pass->saved, glm_ctx->active_state, sizeof(pass->saved));
    pass->saved_error = pass->saved.error;
    mglRendererActivateReplayStatePort(renderer, glm_ctx);
    mglRendererAssertDualProxyPort(renderer, glm_ctx);
    pass->replay_error = (GLenum)mglRenderErrorNone();
    return 1;
}

/* The @try body: run the flush loop and log the per-flush summary. */
void mglBatchFlushRunBatches(void *renderer, GLMContext glm_ctx, MGLBatchFlushPass *pass)
{
    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    uint64_t hit = pass->hit;
    uint32_t skipped = pass->skipped;
    MGLBatchingState *bs = mglRendererBatchingStatePort(renderer);
    MGLBatchFlushLoopState st; memset(&st, 0, sizeof(st));
    FCtx fc = {.r = renderer, .ctx = glm_ctx, .saved = &pass->saved, .hit = hit,
               .err = &pass->replay_error, .skipped = &skipped, .st = &st};
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
        .skip_enabled = bs && bs->skipSameKeyRestoreEnabled ? 1u : 0u,
        .oracle_env_enabled = mgl_env_flag_enabled("MGL_SKIP_SAME_KEY_ORACLE") ? 1u : 0u,
    };
    mgl_batch_flush_run_batches(&st, &ops);
    pass->skipped = skipped;
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglTraceNowSeconds());
    if (mgl_batch_flush_should_trace_log(
            hit, cb->total_commands, kMGLDiagnosticStateLogs ? 1 : 0, skipped,
            mglRenderErrorIsNone((uint32_t)pass->replay_error) ? 0 : 1)) {
        mglTraceLog(
            "MGL TRACE flushDrawBuffer hit=%llu batches=%u totalCommands=%u "
            "arrays=%u elements=%u streamMergedBatches=%u streamMergedCommands=%u "
            "mdiBatches=%u mdiCommands=%u icbBatches=%u icbCommands=%u "
            "directBatches=%u directCommands=%u skippedCommands=%u",
            (unsigned long long)hit, cb->batch_count, cb->total_commands,
            cb->array_cmd_count, cb->element_cmd_count,
            st.path_stats.stream_batches, st.path_stats.stream_commands,
            st.path_stats.mdi_batches, st.path_stats.mdi_commands,
            st.path_stats.icb_batches, st.path_stats.icb_commands,
            st.path_stats.direct_batches, st.path_stats.direct_commands, skipped);
    }
}

MGLBatchPath mglBatchScheduleDrawBatch(void *renderer, MGLDrawBatch *batch,
                                      GLMContext glm_ctx)
{
    (void)renderer;
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
    in.disable_mdi = mgl_env_flag_enabled("MGL_DISABLE_MDI") ? 1u : 0u;
    if (__builtin_available(macOS 10.14, *)) in.icb_os_supported = 1u;
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

void mglBatchRestoreStateForBatch(void *renderer, MGLDrawBatch *batch, GLMContext glm_ctx,
                                 const GLMState *savedState,
                                 const MGLStateKey *prevKey, GLuint forcedDirtyBits)
{
    if (!renderer || !batch || !glm_ctx) return;
    MGLBatchingState *bs = mglRendererBatchingStatePort(renderer);
    MGL_SIGNPOST_BEGIN(RestoreStateForBatch);
    mglRendererAssertDualProxyPort(renderer, glm_ctx);
    if (batch->state_snapshot) {
        mglCopyHotStateFields(glm_ctx->active_state, (const GLMState *)batch->state_snapshot);
        MGL_PERF_INC(g_mglReplayMemcpyCountSinceSwap);
        mgl_batch_replay_copy_object_hash_tables(glm_ctx->active_state, savedState);
        mglRestoreProgramPipelinePair(glm_ctx, glm_ctx->active_state->program_name,
                                      glm_ctx->active_state->var.program_pipeline_binding);
    } else {
        mglBatchRestoreStateFromKey(&batch->key, glm_ctx);
    }
    mglRendererSetActiveStatePort(renderer, glm_ctx);
    glm_ctx->active_state->dirty_bits = 0;
    const GLuint kFull = mgl_batch_restore_full_dirty_bits();
    GLuint replayDirtyBits = kFull;
    int prevKeyValid = (prevKey != NULL);
    int canDelta = mgl_batch_restore_can_delta(
                       bs && bs->dirtyKeyDeltaEnabled ? 1 : 0, prevKeyValid,
                       mglRenderEncoderOwnerHasCurrent(
                           mglRendererCurrentRenderEncoderOwnerPort(renderer)),
                       mglRendererBindingStateIsValidPort(renderer))
                       ? 1 : 0;
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
                           mglRendererCurrentRenderEncoderOwnerPort(renderer)) ? 1u : 0u,
        .bind_valid = mglRendererBindingStateIsValidPort(renderer) ? 1u : 0u,
        .pass_matches = mglRendererCurrentRenderPassMatchesFramebufferPort(renderer) ? 1u : 0u,
    };
    replayDirtyBits = mgl_batch_restore_finish_dirty(replayDirtyBits, forcedDirtyBits, kFull,
                                                     DIRTY_FBO, &fboIn);
    mglMarkRendererDirtyBits(glm_ctx->active_state, replayDirtyBits);
    MGL_SIGNPOST_END(RestoreStateForBatch);
}

void mglBatchTeardownReplay(void *renderer, GLMContext glm_ctx, MGLBatchFlushPass *pass)
{
    MGLBatchingState *bs = mglRendererBatchingStatePort(renderer);
    mglRendererAssertDualProxyPort(renderer, glm_ctx);
    const int usedReplayWorkspace = (glm_ctx->active_state == &glm_ctx->replay_state);
    if (usedReplayWorkspace)
        mgl_batch_replay_sync_hash_tables_from_replay(&glm_ctx->state, &glm_ctx->replay_state);
    mglRendererRestoreLiveActiveStatePort(renderer, glm_ctx);
    mglRendererAssertDualProxyPort(renderer, glm_ctx);
    if (bs) bs->absoluteVertexBindingOffsets = 0u;
    mglResetCommandBufferForContext(glm_ctx, &glm_ctx->draw_command_buffer);
    if (bs && bs->arenaSnapshotEnabled) mglResetBatchArena(&bs->batchArena);
    if (!usedReplayWorkspace) memcpy(glm_ctx->active_state, &pass->saved, sizeof(GLMState));
    mglClearStateDirtyBitsPreservingHashInvalidation(glm_ctx->active_state);
    mglRestoreProgramPipelinePair(glm_ctx, glm_ctx->active_state->program_name,
                                  glm_ctx->active_state->var.program_pipeline_binding);
    if (mglRenderErrorIsNone((uint32_t)pass->saved_error) &&
        !mglRenderErrorIsNone((uint32_t)pass->replay_error))
        glm_ctx->active_state->error = pass->replay_error;
}

int mglBatchTraceSkipCommands(void *renderer, MGLDrawBatch *batch, GLMContext glm_ctx,
                             uint64_t flushId, uint32_t batchIndex, const char *phase,
                             const char *reason, uint32_t *skippedCommands)
{
    mglBatchTraceReplayBatch(renderer, batch, glm_ctx, flushId, batchIndex, phase);
    SkipCtx sc = {.r = renderer, .batch = batch, .ctx = glm_ctx, .hit = flushId,
                  .bi = batchIndex, .reason = reason};
    mgl_batch_flush_trace_skip_commands(batch->command_count, skipTraceCmd, &sc,
                                        skippedCommands);
    MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
    return 0;
}

int mglBatchCheckShouldExecute(void *renderer, MGLDrawBatch *batch, GLMContext glm_ctx,
                              uint64_t flushId, uint32_t batchIndex,
                              GLenum *replayError, uint32_t *skippedCommands)
{
    CCtx c = {.r = renderer, .batch = batch, .ctx = glm_ctx, .hit = flushId,
              .bi = batchIndex, .err = replayError, .skipped = skippedCommands,
              .mode = batch && batch->command_count ? batch->commands[0].mode : 0};
    MGLBatchCheckExecOps ops = {
        .ctx = &c, .begin_trace = cBegin, .prepare_fbo = cFbo, .process_gl_state = cProc,
        .capture_error_if_any = cErr, .should_apply_sampler = cShouldS,
        .apply_sampler = cApplyS, .trace_ready = cReady, .empty_raster = cEmpty,
        .fully_culled = cCull, .apply_polygon_offset = cPoly, .trace_skip = cSkip,
    };
    return mgl_batch_check_should_execute(&ops) ? 1 : 0;
}

void mglBatchRecordCommandStats(void *renderer, MGLDrawBatch *batch, GLMContext glm_ctx)
{
    MGLBatchCmdFrameStats st; mgl_batch_flush_accum_cmd_frame_stats(batch, &st);
    MGL_FRAME_ADD(g_mglDrawArraysSinceSwap, st.array_draws);
    MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap, st.array_vertices);
    MGL_FRAME_ADD(g_mglDrawElementsSinceSwap, st.element_draws);
    MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap, st.element_indices);
    mglBatchRtMarkCurrentFramebufferDrawAttachments(renderer, glm_ctx);
}

void mglBatchTraceStreamCmd0(void *renderer, MGLDrawBatch *batch, GLMContext glm_ctx,
                            const char *phase, const char *reason)
{
    if (!batch || batch->command_count == 0) return;
    mglBatchTraceReplayCommand(renderer, batch, &batch->commands[0], glm_ctx,
                              mglRendererBatchTraceFlushIdPort(renderer),
                              mglRendererBatchTraceBatchIndexPort(renderer),
                              0, phase, reason);
}

void mglBatchIssueStreamMergedBatch(void *renderer, MGLDrawBatch *batch, GLMContext glm_ctx,
                                   const MGLEncodeContext *encCtx)
{
    SCtx c = {.r = renderer, .batch = batch, .ctx = glm_ctx, .enc = encCtx};
    MGLBatchStreamMergedOps ops = {
        .ctx = &c, .trace_cmd0 = sTr0, .try_stream_mdi = sMdi, .issue_direct = sDir,
        .resolve_stream_index = sIdx, .draw_stream_indexed = sDraw,
    };
    mgl_batch_issue_stream_merged(batch, mgl_env_flag_enabled("MGL_DISABLE_MDI") ? 1 : 0, &ops);
}
