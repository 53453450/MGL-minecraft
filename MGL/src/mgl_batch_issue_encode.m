/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: MDI/direct issue — loops in mgl_batch_mtl_issue_mdi_batch / issue_direct_batch.
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+BatchPorts_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"
#include "mgl_batch_encode_shared.h"

@implementation MGLRenderer (Draw)

typedef struct {
    MGLRenderer *r; MGLDrawBatch *batch; GLMContext ctx;
    MGLEncodeContext *enc; const MGLEncodeContext *cenc;
} MGLIssueEncCtx;

static BOOL mglBatchHasEnc(const MGLEncodeContext *e)
{ return e && mglRenderEncoderOwnerHasCurrent(e->render_encoder_owner) != 0; }
static void mglIssueTrace(void *v, uint32_t i, const char *phase, const char *reason)
{
    MGLIssueEncCtx *c = v;
    if (!c->batch || i >= c->batch->command_count) return;
    [c->r traceReplayCommand:c->batch command:&c->batch->commands[i] context:c->ctx
                     flushId:mglRendererRenderPassManager(c->r).state->traceReplayFlushId
                  batchIndex:mglRendererRenderPassManager(c->r).state->traceReplayBatchIndex
                commandIndex:i phase:phase reason:reason];
}
static void mglMdiDirect(void *v)
{ MGLIssueEncCtx *c = v; [c->r issueDirectBatch:c->batch context:c->ctx encodeContext:c->cenc]; }
static void *mglMdiScratch(void *v, uint64_t len, uint64_t *off)
{
    NSUInteger o = 0;
    id b = [((MGLIssueEncCtx *)v)->r mdiArgumentScratchBufferWithLength:(NSUInteger)len offset:&o];
    if (off) *off = (uint64_t)o; return (__bridge void *)b;
}
static int mglMdiMap(void *v, void *buf, uint64_t off, uint64_t need, void **out)
{ (void)v; return mgl_batch_encode_map_scratch(buf, off, need, out); }
static int mglMdiResolve(void *v, uint32_t i, uint32_t gl_itype, void **mtl,
                         uint64_t *ioff, uint32_t *mtype)
{
    MGLIssueEncCtx *c = v; MGLDrawCommand *cmd = &c->batch->commands[i];
    Buffer *glBuf = NULL; id idxBuf = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"mdiBatch" context:c->ctx
                                     glBuffer:&glBuf mtlBuffer:&idxBuf])
        return 0;
    NSUInteger drawOff = ioff ? (NSUInteger)*ioff : cmd->indexBufferOffset;
    uint64_t drawType = mglIndexTypeForGLType((GLenum)gl_itype);
    id prepared = mglPreparedElementIndexBuffer((__bridge id)mglRendererBackendGetDevice(mglRendererBackend(c->r)), glBuf, idxBuf, (GLenum)gl_itype,
                                                &drawOff, &drawType);
    if (ioff) *ioff = (uint64_t)drawOff; if (mtype) *mtype = (uint32_t)drawType;
    if (mtl) *mtl = (__bridge void *)prepared; return 1;
}
static void mglDirRefresh(void *v)
{ MGLIssueEncCtx *c = v; c->enc->render_encoder_owner =
      mglRendererRenderPassManager(c->r).state->currentRenderEncoderOwner; }
static int mglDirSimple(void *v)
{ MGLIssueEncCtx *c = v; return [c->r tryReplaySimpleBatch:c->batch context:c->ctx
      encodeContext:c->enc] ? 1 : 0; }
static uint32_t mglDirCount(void *v) { return ((MGLIssueEncCtx *)v)->batch->command_count; }
static void mglDirFill(void *v, uint32_t i, MGLBatchDirectCmdView *out)
{
    MGLDrawCommand *cmd = &((MGLIssueEncCtx *)v)->batch->commands[i];
    out->type = (uint32_t)cmd->type; out->mode = (uint32_t)cmd->mode;
    out->count = cmd->count; out->first = cmd->first;
    out->instance_count = cmd->instanceCount; out->base_instance = cmd->baseInstance;
}
static int mglDirCullUse(void *v)
{ Program *p = mglResolveProgramForStageFromState(((MGLIssueEncCtx *)v)->ctx, _VERTEX_SHADER);
  return (p && p->uses_cull_distance) ? 1 : 0; }
static uint32_t mglDirPrim(void *v)
{ return (uint32_t)((MGLIssueEncCtx *)v)->batch->key.primitive_type; }
static int mglDirSnapMix(void *v)
{ return ((MGLIssueEncCtx *)v)->batch->sampler_snapshots_mixed ? 1 : 0; }
static int mglDirDynTex(void *v)
{ return ((MGLIssueEncCtx *)v)->batch->has_dynamic_texture_bindings ? 1 : 0; }
static int mglDirCullCap(void *v, uint32_t i, int cullPath)
{
    MGLIssueEncCtx *c = v; MGLDrawCommand *cmd = &c->batch->commands[i];
    if (cullPath == MGL_BATCH_CULL_CAPTURE_ARRAYS)
        return [c->r captureAIRCullDistancesForArrayDraw:c->ctx first:cmd->first count:cmd->count
            instanceCount:cmd->instanceCount baseInstance:cmd->baseInstance] ? 1 : 0;
    if (cullPath != MGL_BATCH_CULL_CAPTURE_ELEMENTS) return 0;
    Buffer *eb = NULL; id meb = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"cullDistanceCapture" context:c->ctx
                                     glBuffer:&eb mtlBuffer:&meb])
        return 0;
    const uint8_t *src = mglElementIndexSourceForDraw(eb, meb, cmd->indexType,
                                                      cmd->indexBufferOffset, cmd->count);
    return [c->r captureAIRCullDistancesForElementDraw:c->ctx indexBytes:src
        indexType:cmd->indexType count:cmd->count baseVertex:cmd->baseVertex
        instanceCount:cmd->instanceCount baseInstance:cmd->baseInstance] ? 1 : 0;
}
static int mglDirAfterCull(void *v)
{ MGLIssueEncCtx *c = v; return ([c->r processGLState:true] &&
      mglRenderEncoderOwnerHasCurrent(mglRendererRenderPassManager(c->r).state->currentRenderEncoderOwner))
      ? 1 : 0; }
static int mglDirDyn(void *v, uint32_t i)
{ MGLIssueEncCtx *c = v; return [c->r applyDynamicBindingsForCommand:&c->batch->commands[i]
      context:c->ctx encodeContext:c->enc] ? 1 : 0; }
static int mglDirSamp(void *v, uint32_t i)
{ MGLIssueEncCtx *c = v; return [c->r applySamplerSnapshotForCommand:&c->batch->commands[i]
      context:c->ctx encodeContext:c->enc] ? 1 : 0; }
static int mglDirPolyPt(void *v, uint32_t mode)
{ return mglPolygonModePointForDrawMode(((MGLIssueEncCtx *)v)->ctx, (GLenum)mode) ? 1 : 0; }
static void mglDirSkip(void *v, uint32_t i, const char *reason) { mglIssueTrace(v, i, "SKIP", reason); }

static int mglDirTryCullArr(void *v, uint32_t i, uint32_t mode, int32_t first, int32_t count,
                            int32_t ic, uint32_t bi)
{
    (void)i; MGLIssueEncCtx *c = v;
    if (!mglBatchHasEnc(c->enc)) return 0;
    Program *p = mglResolveProgramForStageFromState(c->ctx, _VERTEX_SHADER);
    if (!p || !p->uses_cull_distance) return 0;
    return mglEncodeCullDistanceArraySplitForRenderEncoderOwner(
        c->enc->render_encoder_owner, (__bridge id)mglRendererBackendGetDevice(mglRendererBackend(c->r)), (GLenum)mode, first, count, (size_t)ic,
        (size_t)bi, (__bridge void *)c->r, c->enc, mglRendererBindCullDistanceEmu) ? 1 : 0;
}
static void mglDirBindCullEmu(void *v, uint32_t mode, int32_t first)
{
    MGLIssueEncCtx *c = v;
    Program *p = mglResolveProgramForStageFromState(c->ctx, _VERTEX_SHADER);
    if (p && p->uses_cull_distance)
        [c->r bindCullDistanceEmulationBuffers:(GLenum)mode firstVertex:(GLuint)first
                               explicitVertices:NULL explicitVertexCount:0u encodeContext:c->enc];
}
static int mglDirEncArr(void *v, uint32_t mode, int32_t first, int32_t count, int32_t ic,
                        uint32_t bi)
{
    MGLIssueEncCtx *c = v;
    return mglEncodeDrawArraysForRenderEncoderOwner(
        c->enc->render_encoder_owner, c->ctx, (__bridge id)mglRendererBackendGetDevice(mglRendererBackend(c->r)), (GLenum)mode, first, count,
        (size_t)ic, (size_t)bi, "batch") ? 1 : 0;
}
static int mglDirPrepEl(void *v, uint32_t i, MGLBatchDirectElementPrep *out)
{
    MGLIssueEncCtx *c = v; MGLDrawCommand *cmd = &c->batch->commands[i];
    Buffer *glBuf = NULL; id idxBuf = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"directBatch" context:c->ctx
                                     glBuffer:&glBuf mtlBuffer:&idxBuf])
        return 0;
    NSUInteger idxOffset = cmd->indexBufferOffset;
    uint64_t mtlIdxType = mglIndexTypeForGLType(cmd->indexType);
    out->gl_buffer = glBuf; out->mtl_buffer = (__bridge void *)idxBuf;
    out->index_offset = (uint64_t)idxOffset; out->gl_index_type = (uint32_t)cmd->indexType;
    out->mtl_index_type = (uint32_t)mtlIdxType; out->base_vertex = cmd->baseVertex;
    out->base_instance = cmd->baseInstance;
    out->cull_index_bytes = mglElementIndexSourceForDraw(glBuf, idxBuf, cmd->indexType,
                                                         idxOffset, cmd->count);
    return 1;
}
static int mglDirPolyLine(void *v, uint32_t mode)
{ return mglPolygonModeLineForDrawMode(((MGLIssueEncCtx *)v)->ctx, (GLenum)mode) ? 1 : 0; }
static int mglDirTryCullEl(void *v, uint32_t i, uint32_t mode, const MGLBatchDirectElementPrep *prep,
                           int32_t count, int32_t ic, int poly_line)
{
    (void)i; MGLIssueEncCtx *c = v;
    return [c->r encodeCullDistanceElementDraw:(GLenum)mode indexBytes:prep->cull_index_bytes
        indexType:(GLenum)prep->gl_index_type count:count baseVertex:prep->base_vertex
        instanceCount:ic baseInstance:prep->base_instance polygonLineMode:poly_line
        encodeContext:c->enc] ? 1 : 0;
}
static int mglDirEncEl(void *v, uint32_t mode, const MGLBatchDirectElementPrep *prep,
                       int32_t count, int32_t ic)
{
    MGLIssueEncCtx *c = v;
    return mglEncodeDrawElementsForRenderEncoderOwner(
        c->enc->render_encoder_owner, c->ctx, (__bridge id)mglRendererBackendGetDevice(mglRendererBackend(c->r)), (Buffer *)prep->gl_buffer,
        (__bridge id)prep->mtl_buffer, (GLenum)mode, (GLenum)prep->gl_index_type,
        (NSUInteger)prep->index_offset, count, ic, prep->base_vertex, prep->base_instance,
        "directBatch") ? 1 : 0;
}


- (void)issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
        encodeContext:(const MGLEncodeContext *)encCtx
{
    MGLIssueEncCtx ctx = {.r = self, .batch = batch, .ctx = glm_ctx, .cenc = encCtx};
    MGLBatchMdiIssueOps ops = {
        .ctx = &ctx, .on_trace = mglIssueTrace, .issue_direct = mglMdiDirect,
        .alloc_scratch = mglMdiScratch, .map_scratch = mglMdiMap, .resolve_index = mglMdiResolve,
    };
    mgl_batch_mtl_issue_mdi_batch(batch, mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1 : 0,
                                  encCtx ? encCtx->render_encoder_owner : NULL, &ops);
}

- (void)issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
           encodeContext:(const MGLEncodeContext *)encCtx
{
    MGLEncodeContext live = *encCtx;
    live.render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner;
    MGLIssueEncCtx ctx = {.r = self, .batch = batch, .ctx = glm_ctx, .enc = &live};
    MGLBatchDirectIssueOps ops = {
        .ctx = &ctx, .refresh_encoder = mglDirRefresh, .try_simple_replay = mglDirSimple,
        .command_count = mglDirCount, .fill_cmd = mglDirFill, .uses_cull_distance = mglDirCullUse,
        .batch_primitive_type = mglDirPrim, .snapshots_mixed = mglDirSnapMix,
        .has_dyn_texture = mglDirDynTex, .cull_capture = mglDirCullCap,
        .after_cull_ok = mglDirAfterCull, .apply_dyn_bindings = mglDirDyn,
        .apply_cmd_sampler = mglDirSamp, .polygon_mode_point = mglDirPolyPt,
        .trace_skip = mglDirSkip,
        .array_submit = {.ctx = &ctx, .try_cull_array_split = mglDirTryCullArr,
                         .bind_cull_emu_arrays = mglDirBindCullEmu,
                         .encode_arrays = mglDirEncArr, .on_trace = mglIssueTrace},
        .element_submit = {.ctx = &ctx, .prepare_element = mglDirPrepEl,
                           .polygon_mode_line = mglDirPolyLine,
                           .try_cull_element_split = mglDirTryCullEl,
                           .encode_elements = mglDirEncEl, .on_trace = mglIssueTrace},
    };
    mgl_batch_issue_direct_batch(&ops);
}

@end
