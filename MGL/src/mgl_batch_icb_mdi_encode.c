/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: ICB + stream-MDI — whole loops via mgl_batch_mtl_issue_*_batch.
 *
 * Formerly mgl_batch_icb_mdi_encode.m.  The two issue entry points are C
 * drivers now; the one piece that has to stay ObjC -- the @try/@catch around
 * the Metal indirect-command-buffer allocation -- lives behind
 * mglRendererCreateIndirectCommandBufferPort in mgl_renderer_port_shim.m.
 */
#include "mgl_renderer_ports.h"   /* C port surface (T4) */
#include "mgl_draw_issue.h"       /* mglDrawHostDevice */
#include "mgl_types_program.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_batch_path.h"
#include "mgl_batch_replay.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"
#include "mgl_batch_rt_mark.h"    /* mglBatchTraceReplayCommand */
#include "mgl_batch_encode_shared.h"
#include <CoreFoundation/CoreFoundation.h>

typedef struct {
    void *r;
    MGLDrawBatch *batch;
    GLMContext ctx;
    const MGLEncodeContext *enc;
    void *sticky_icb; /* +1 reference held for the C driver's lifetime */
} MGLIcbMdiCtx;

static void mglIcbTrace(void *v, uint32_t i, const char *phase, const char *reason)
{
    MGLIcbMdiCtx *c = (MGLIcbMdiCtx *)v;
    if (!c->batch || i >= c->batch->command_count) return;
    mglBatchTraceReplayCommand(c->r, c->batch, &c->batch->commands[i],
                               c->ctx,
                               mglRendererBatchTraceFlushIdPort(c->r),
                               mglRendererBatchTraceBatchIndexPort(c->r),
                               i, phase, reason);
}

static void *mglIcbScratch(void *v, uint64_t len, uint64_t *off)
{
    /* The ops table passes ITS ctx (MGLIcbMdiCtx*), not the renderer handle --
     * the port takes the renderer, so unwrap here. */
    return mglRendererMdiScratchBufferPort(((MGLIcbMdiCtx *)v)->r, len, off);
}

static int mglIcbMap(void *v, void *buf, uint64_t off, uint64_t need, void **out)
{
    (void)v;
    return mgl_batch_encode_map_scratch(buf, off, need, out);
}

static int mglIcbResolve(void *v, uint32_t i, uint32_t gl_itype, void **mtl,
                         uint64_t *ioff, uint32_t *mtype)
{
    MGLIcbMdiCtx *c = (MGLIcbMdiCtx *)v;
    MGLDrawCommand *cmd = &c->batch->commands[i];
    Buffer *glBuf = NULL;
    void *idxBuf = NULL;
    if (!mglRendererResolveElementBufferPort(c->r, cmd, "icbBatch", c->ctx,
                                             &glBuf, &idxBuf))
        return 0;
    size_t drawOff = ioff ? (size_t)*ioff : (size_t)cmd->indexBufferOffset;
    uint64_t drawType = mglRenderMTLIndexTypeForGLType((uint32_t)gl_itype);
    void *prepared = mglPreparedElementIndexBuffer(
        mglDrawHostDevice(c->r), glBuf, idxBuf, (GLenum)gl_itype, &drawOff, &drawType);
    if (ioff) *ioff = (uint64_t)drawOff;
    if (mtype) *mtype = (uint32_t)drawType;
    if (mtl) *mtl = prepared;
    return 1;
}

static void *mglIcbCreate(void *v, int indexed, uint64_t count)
{
    MGLIcbMdiCtx *c = (MGLIcbMdiCtx *)v;
    int failed = 0;
    void *icb = mglRendererCreateIndirectCommandBufferPort(c->r, indexed, count,
                                                           &failed);
    if (failed) {
        mglIcbTrace(v, 0, "FALLBACK", "icb_create_exception");
        return NULL;
    }
    if (!icb) {
        mglIcbTrace(v, 0, "FALLBACK", "icb_create_nil");
        return NULL;
    }
    if (c->sticky_icb) {
        CFRelease(c->sticky_icb);
        c->sticky_icb = NULL;
    }
    c->sticky_icb = icb;
    return c->sticky_icb;
}

static void *mglStreamIdx(void *v)
{
    MGLIcbMdiCtx *c = (MGLIcbMdiCtx *)v;
    Buffer *indexBuffer = (Buffer *)c->batch->stream_index_buffer;
    if (!indexBuffer || !mglRendererProcessBufferPort(c->r, indexBuffer)) {
        mglIcbTrace(v, 0, "FALLBACK", "stream_mdi_index_buffer");
        return NULL;
    }
    void *mtl = indexBuffer->data.mtl_data;
    if (!mtl) {
        mglIcbTrace(v, 0, "FALLBACK", "stream_mdi_no_mtl_index");
        return NULL;
    }
    return mtl;
}


int mglBatchIssueStreamMergedMDIBatch(void *renderer, MGLDrawBatch *batch,
                                      GLMContext glm_ctx,
                                      const MGLEncodeContext *encCtx)
{
    MGLIcbMdiCtx ctx = {.r = renderer, .batch = batch, .ctx = glm_ctx, .enc = encCtx};
    MGLBatchStreamMdiIssueOps ops = {
        .ctx = &ctx,
        .on_trace = mglIcbTrace,
        .resolve_stream_index = mglStreamIdx,
        .alloc_scratch = mglIcbScratch,
        .map_scratch = mglIcbMap,
    };
    return mgl_batch_mtl_issue_stream_mdi_batch(
               batch, mgl_env_flag_enabled("MGL_DISABLE_MDI"),
               encCtx ? encCtx->render_encoder_owner : NULL, &ops)
               ? 1
               : 0;
}

int mglBatchIssueIndirectCommandBufferBatch(void *renderer, MGLDrawBatch *batch,
                                            GLMContext glm_ctx,
                                            const MGLEncodeContext *encCtx)
{
    MGLBatchIcbConfig icbCfg = mgl_batch_icb_config();
    MGLIcbMdiCtx ctx = {.r = renderer, .batch = batch, .ctx = glm_ctx, .enc = encCtx,
                       .sticky_icb = NULL};
    MGLBatchIcbIssueOps ops = {
        .ctx = &ctx,
        .on_trace = mglIcbTrace,
        .create_icb = mglIcbCreate,
        .resolve_index = mglIcbResolve,
    };
    int os_ok = 0;
    if (__builtin_available(macOS 10.14, *)) {
        os_ok = 1;
    }
    const int ok = mgl_batch_mtl_issue_icb_batch(
        batch, mglDrawHostDevice(renderer) ? 1 : 0,
        (encCtx &&
         mglRenderEncoderOwnerHasCurrent(encCtx->render_encoder_owner))
            ? 1
            : 0,
        (int)icbCfg.enable, (int)icbCfg.disable, os_ok,
        encCtx ? encCtx->render_encoder_owner : NULL, &ops);
    if (ctx.sticky_icb) {
        CFRelease(ctx.sticky_icb);
        ctx.sticky_icb = NULL;
    }
    return ok ? 1 : 0;
}
