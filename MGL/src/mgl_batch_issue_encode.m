/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: MDI/direct issue — loops in mgl_batch_mtl_issue_mdi_batch / issue_direct_batch.
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"
#include "mgl_batch_encode_shared.h"

typedef struct {
    MGLRenderer *r;
    MGLDrawBatch *batch;
    GLMContext ctx;
    MGLEncodeContext *enc;
    const MGLEncodeContext *cenc;
} MGLIssueEncCtx;

static BOOL mglBatchHasEnc(const MGLEncodeContext *e)
{
    return e && mglRenderEncoderOwnerHasCurrent(e->render_encoder_owner) != 0;
}

static void mglIssueTrace(void *v, uint32_t i, const char *phase,
                          const char *reason)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    if (!c->batch || i >= c->batch->command_count) return;
    [c->r traceReplayCommand:c->batch command:&c->batch->commands[i]
                     context:c->ctx
                     flushId:c->r->_renderPassManager.state->traceReplayFlushId
                  batchIndex:c->r->_renderPassManager.state->traceReplayBatchIndex
                commandIndex:i phase:phase reason:reason];
}

static void mglMdiDirect(void *v)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    [c->r issueDirectBatch:c->batch context:c->ctx encodeContext:c->cenc];
}
static void *mglMdiScratch(void *v, uint64_t len, uint64_t *off)
{
    NSUInteger o = 0;
    id b = [((MGLIssueEncCtx *)v)->r mdiArgumentScratchBufferWithLength:(NSUInteger)len
                                                                 offset:&o];
    if (off) *off = (uint64_t)o;
    return (__bridge void *)b;
}
static int mglMdiMap(void *v, void *buf, uint64_t off, uint64_t need, void **out)
{
    (void)v;
    return mgl_batch_encode_map_scratch(buf, off, need, out);
}
static int mglMdiResolve(void *v, uint32_t i, uint32_t gl_itype, void **mtl,
                         uint64_t *ioff, uint32_t *mtype)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    MGLDrawCommand *cmd = &c->batch->commands[i];
    Buffer *glBuf = NULL;
    id idxBuf = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"mdiBatch" context:c->ctx
                                     glBuffer:&glBuf mtlBuffer:&idxBuf])
        return 0;
    NSUInteger drawOff = ioff ? (NSUInteger)*ioff : cmd->indexBufferOffset;
    uint64_t drawType = mglIndexTypeForGLType((GLenum)gl_itype);
    id prepared = mglPreparedElementIndexBuffer(
        c->r->_device, glBuf, idxBuf, (GLenum)gl_itype, &drawOff, &drawType);
    if (ioff) *ioff = (uint64_t)drawOff;
    if (mtype) *mtype = (uint32_t)drawType;
    if (mtl) *mtl = (__bridge void *)prepared;
    return 1;
}

static void mglDirRefresh(void *v)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    c->enc->render_encoder_owner =
        c->r->_renderPassManager.state->currentRenderEncoderOwner;
}
static int mglDirSimple(void *v)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    return [c->r tryReplaySimpleBatch:c->batch context:c->ctx
                        encodeContext:c->enc] ? 1 : 0;
}
static uint32_t mglDirCount(void *v)
{
    return ((MGLIssueEncCtx *)v)->batch->command_count;
}
static void mglDirFill(void *v, uint32_t i, MGLBatchDirectCmdView *out)
{
    MGLDrawCommand *cmd = &((MGLIssueEncCtx *)v)->batch->commands[i];
    out->type = (uint32_t)cmd->type;
    out->mode = (uint32_t)cmd->mode;
    out->count = cmd->count;
    out->instance_count = cmd->instanceCount;
    out->base_instance = cmd->baseInstance;
}
static int mglDirCullUse(void *v)
{
    Program *p = mglResolveProgramForStageFromState(((MGLIssueEncCtx *)v)->ctx,
                                                    _VERTEX_SHADER);
    return (p && p->uses_cull_distance) ? 1 : 0;
}
static uint32_t mglDirPrim(void *v)
{
    return (uint32_t)((MGLIssueEncCtx *)v)->batch->key.primitive_type;
}
static int mglDirSnapMix(void *v)
{
    return ((MGLIssueEncCtx *)v)->batch->sampler_snapshots_mixed ? 1 : 0;
}
static int mglDirDynTex(void *v)
{
    return ((MGLIssueEncCtx *)v)->batch->has_dynamic_texture_bindings ? 1 : 0;
}
static int mglDirCullCap(void *v, uint32_t i, int cullPath)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    MGLDrawCommand *cmd = &c->batch->commands[i];
    if (cullPath == MGL_BATCH_CULL_CAPTURE_ARRAYS) {
        return [c->r captureAIRCullDistancesForArrayDraw:c->ctx first:cmd->first
                                                   count:cmd->count
                                           instanceCount:cmd->instanceCount
                                            baseInstance:cmd->baseInstance]
                   ? 1 : 0;
    }
    if (cullPath != MGL_BATCH_CULL_CAPTURE_ELEMENTS) return 0;
    Buffer *eb = NULL;
    id meb = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"cullDistanceCapture"
                                      context:c->ctx glBuffer:&eb mtlBuffer:&meb])
        return 0;
    const uint8_t *src = mglElementIndexSourceForDraw(
        eb, meb, cmd->indexType, cmd->indexBufferOffset, cmd->count);
    return [c->r captureAIRCullDistancesForElementDraw:c->ctx indexBytes:src
                                             indexType:cmd->indexType
                                                 count:cmd->count
                                            baseVertex:cmd->baseVertex
                                         instanceCount:cmd->instanceCount
                                          baseInstance:cmd->baseInstance]
               ? 1 : 0;
}
static int mglDirAfterCull(void *v)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    return ([c->r processGLState:true] &&
            mglRenderEncoderOwnerHasCurrent(
                c->r->_renderPassManager.state->currentRenderEncoderOwner) != 0)
               ? 1 : 0;
}
static int mglDirDyn(void *v, uint32_t i)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    return [c->r applyDynamicBindingsForCommand:&c->batch->commands[i]
                                        context:c->ctx encodeContext:c->enc]
               ? 1 : 0;
}
static int mglDirSamp(void *v, uint32_t i)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    return [c->r applySamplerSnapshotForCommand:&c->batch->commands[i]
                                        context:c->ctx encodeContext:c->enc]
               ? 1 : 0;
}
static int mglDirPolyPt(void *v, uint32_t mode)
{
    return mglPolygonModePointForDrawMode(((MGLIssueEncCtx *)v)->ctx, (GLenum)mode)
               ? 1 : 0;
}
static void mglDirSkip(void *v, uint32_t i, const char *reason)
{
    mglIssueTrace(v, i, "SKIP", reason);
}

static BOOL mglDirCullArrayDraw(MGLIssueEncCtx *c, GLenum mode, GLint first,
                                GLsizei count, GLsizei ic, GLuint bi)
{
    Program *p = mglResolveProgramForStageFromState(c->ctx, _VERTEX_SHADER);
    if (!p || !p->uses_cull_distance ||
        !mglBatchHasEnc(c->enc)) {
        return NO;
    }
    return mglEncodeCullDistanceArraySplitForRenderEncoderOwner(
        c->enc->render_encoder_owner, c->r->_device, mode, first, count,
        (size_t)ic, (size_t)bi, (__bridge void *)c->r, c->enc,
        mglRendererBindCullDistanceEmu);
}

static void mglDirSubmitArrays(void *v, uint32_t i, uint32_t mode, int32_t count,
                               int32_t ic, uint32_t bi, int poly,
                               const char *reason, const char *cullReason)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    MGLDrawCommand *cmd = &c->batch->commands[i];
    if (!poly &&
        mglDirCullArrayDraw(c, (GLenum)mode, cmd->first, count, ic, bi)) {
        mglIssueTrace(v, i, "SUBMIT", cullReason);
        return;
    }
    if (!poly) {
        Program *p = mglResolveProgramForStageFromState(c->ctx, _VERTEX_SHADER);
        if (p && p->uses_cull_distance) {
            [c->r bindCullDistanceEmulationBuffers:(GLenum)mode
                                        firstVertex:(GLuint)cmd->first
                                   explicitVertices:NULL
                                 explicitVertexCount:0u
                                      encodeContext:c->enc];
        }
    }
    const bool ok = mglEncodeDrawArraysForRenderEncoderOwner(
        c->enc->render_encoder_owner, c->ctx, c->r->_device, (GLenum)mode,
        cmd->first, count, (size_t)ic, (size_t)bi, "batch");
    mglIssueTrace(v, i, ok ? "SUBMIT" : "SKIP", reason);
}

static void mglDirSubmitElements(void *v, uint32_t i, uint32_t mode,
                                 int32_t count, int32_t ic, int poly)
{
    MGLIssueEncCtx *c = (MGLIssueEncCtx *)v;
    MGLDrawCommand *cmd = &c->batch->commands[i];
    Buffer *glBuf = NULL;
    id idxBuf = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"directBatch"
                                      context:c->ctx glBuffer:&glBuf
                                    mtlBuffer:&idxBuf]) {
        mglIssueTrace(v, i, "SKIP", "direct_resolve_element");
        return;
    }
    NSUInteger idxOffset = cmd->indexBufferOffset;
    uint64_t mtlIdxType = mglIndexTypeForGLType(cmd->indexType);
    if ((GLuint)mtlIdxType == 0xFFFFFFFF) {
        mglIssueTrace(v, i, "SKIP", "direct_index_type");
        return;
    }
    const uint8_t *cullSrc = mglElementIndexSourceForDraw(
        glBuf, idxBuf, cmd->indexType, idxOffset, count);
    if (!poly &&
        [c->r encodeCullDistanceElementDraw:(GLenum)mode
                                  indexBytes:cullSrc
                                   indexType:cmd->indexType
                                       count:count
                                  baseVertex:cmd->baseVertex
                               instanceCount:ic
                                baseInstance:cmd->baseInstance
                             polygonLineMode:mglPolygonModeLineForDrawMode(
                                                 c->ctx, (GLenum)mode)
                               encodeContext:c->enc]) {
        mglIssueTrace(v, i, "SUBMIT", "direct_elements_cull_distance_split");
        return;
    }
    const bool encoded = mglEncodeDrawElementsForRenderEncoderOwner(
        c->enc->render_encoder_owner, c->ctx, c->r->_device, glBuf, idxBuf,
        (GLenum)mode, cmd->indexType, idxOffset, count, ic, cmd->baseVertex,
        cmd->baseInstance, "directBatch");
    mglIssueTrace(v, i, encoded ? "SUBMIT" : "SKIP",
                  encoded ? "direct_elements" : "direct_elements_encode_failed");
}

@implementation MGLRenderer (Draw)

- (void)issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                encodeContext:(const MGLEncodeContext *)encCtx
{
    MGLIssueEncCtx ctx = {.r = self, .batch = batch, .ctx = glm_ctx, .cenc = encCtx};
    MGLBatchMdiIssueOps ops = {
        .ctx = &ctx, .on_trace = mglIssueTrace, .issue_direct = mglMdiDirect,
        .alloc_scratch = mglMdiScratch, .map_scratch = mglMdiMap,
        .resolve_index = mglMdiResolve,
    };
    mgl_batch_mtl_issue_mdi_batch(batch,
                                  mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1 : 0,
                                  encCtx ? encCtx->render_encoder_owner : NULL,
                                  &ops);
}

- (void)issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
             encodeContext:(const MGLEncodeContext *)encCtx
{
    MGLEncodeContext live = *encCtx;
    live.render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    MGLIssueEncCtx ctx = {.r = self, .batch = batch, .ctx = glm_ctx, .enc = &live};
    MGLBatchDirectIssueOps ops = {
        .ctx = &ctx,
        .refresh_encoder = mglDirRefresh,
        .try_simple_replay = mglDirSimple,
        .command_count = mglDirCount,
        .fill_cmd = mglDirFill,
        .uses_cull_distance = mglDirCullUse,
        .batch_primitive_type = mglDirPrim,
        .snapshots_mixed = mglDirSnapMix,
        .has_dyn_texture = mglDirDynTex,
        .cull_capture = mglDirCullCap,
        .after_cull_ok = mglDirAfterCull,
        .apply_dyn_bindings = mglDirDyn,
        .apply_cmd_sampler = mglDirSamp,
        .polygon_mode_point = mglDirPolyPt,
        .trace_skip = mglDirSkip,
        .submit_arrays = mglDirSubmitArrays,
        .submit_elements = mglDirSubmitElements,
    };
    mgl_batch_issue_direct_batch(&ops);
}

@end
