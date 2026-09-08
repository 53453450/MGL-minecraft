/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: ICB + stream-MDI — whole loops via mgl_batch_mtl_issue_*_batch.
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_batch_path.h"
#include "mgl_batch_replay.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"
#include "mgl_batch_encode_shared.h"
#include <CoreFoundation/CoreFoundation.h>

typedef struct {
    MGLRenderer *r;
    MGLDrawBatch *batch;
    GLMContext ctx;
    const MGLEncodeContext *enc;
    void *sticky_icb; /* objc_retain for C++ driver lifetime */
} MGLIcbMdiCtx;

static void mglIcbTrace(void *v, uint32_t i, const char *phase, const char *reason)
{
    MGLIcbMdiCtx *c = (MGLIcbMdiCtx *)v;
    if (!c->batch || i >= c->batch->command_count) return;
    [c->r traceReplayCommand:c->batch command:&c->batch->commands[i]
                     context:c->ctx
                     flushId:c->r->_renderPassManager.state->traceReplayFlushId
                  batchIndex:c->r->_renderPassManager.state->traceReplayBatchIndex
                commandIndex:i phase:phase reason:reason];
}

static void *mglIcbScratch(void *v, uint64_t len, uint64_t *off)
{
    NSUInteger o = 0;
    id b = [((MGLIcbMdiCtx *)v)->r mdiArgumentScratchBufferWithLength:(NSUInteger)len
                                                               offset:&o];
    if (off) *off = (uint64_t)o;
    return (__bridge void *)b;
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
    id idxBuf = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"icbBatch" context:c->ctx
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

static void *mglIcbCreate(void *v, int indexed, uint64_t count)
{
    MGLIcbMdiCtx *c = (MGLIcbMdiCtx *)v;
    id icb = nil;
    @try {
        icb = (__bridge_transfer id)mgl_batch_mtl_create_icb(indexed, count);
    } @catch (NSException *ex) {
        static uint64_t s_hit = 0;
        uint64_t hit = ++s_hit;
        if (hit <= 8ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: ICB creation failed, falling back: %@", ex);
        }
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
    c->sticky_icb = (__bridge_retained void *)icb;
    return c->sticky_icb;
}

static void *mglStreamIdx(void *v)
{
    MGLIcbMdiCtx *c = (MGLIcbMdiCtx *)v;
    Buffer *indexBuffer = (Buffer *)c->batch->stream_index_buffer;
    if (!indexBuffer || ![c->r processBuffer:indexBuffer]) {
        mglIcbTrace(v, 0, "FALLBACK", "stream_mdi_index_buffer");
        return NULL;
    }
    id mtl = (__bridge id)(indexBuffer->data.mtl_data);
    if (!mtl) {
        mglIcbTrace(v, 0, "FALLBACK", "stream_mdi_no_mtl_index");
        return NULL;
    }
    return (__bridge void *)mtl;
}

@implementation MGLRenderer (Batch)

- (BOOL)issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                    encodeContext:(const MGLEncodeContext *)encCtx
{
    MGLIcbMdiCtx ctx = {.r = self, .batch = batch, .ctx = glm_ctx, .enc = encCtx};
    MGLBatchStreamMdiIssueOps ops = {
        .ctx = &ctx,
        .on_trace = mglIcbTrace,
        .resolve_stream_index = mglStreamIdx,
        .alloc_scratch = mglIcbScratch,
        .map_scratch = mglIcbMap,
    };
    return mgl_batch_mtl_issue_stream_mdi_batch(
               batch, mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1 : 0,
               encCtx ? encCtx->render_encoder_owner : NULL, &ops)
               ? YES
               : NO;
}

- (BOOL)issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx
{
    MGLBatchIcbConfig icbCfg = mgl_batch_icb_config();
    MGLIcbMdiCtx ctx = {.r = self, .batch = batch, .ctx = glm_ctx, .enc = encCtx,
                       .sticky_icb = NULL};
    MGLBatchIcbIssueOps ops = {
        .ctx = &ctx,
        .on_trace = mglIcbTrace,
        .create_icb = mglIcbCreate,
        .resolve_index = mglIcbResolve,
    };
    int os_ok = 0;
    if (@available(macOS 10.14, *)) {
        os_ok = 1;
    }
    const int ok = mgl_batch_mtl_issue_icb_batch(
        batch, _device ? 1 : 0,
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
    return ok ? YES : NO;
}

- (id)mdiArgumentScratchBufferWithLength:(NSUInteger)length
                                  offset:(NSUInteger *)offsetOut
{
    return (__bridge id)[_renderPassManager
        mdiArgumentScratchBufferWithDevice:(__bridge void *)_device
                                     length:length
                                     offset:offsetOut];
}

@end
