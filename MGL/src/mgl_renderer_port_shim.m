/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_renderer_port_shim.m — the ObjC side of the C port surface (T4).
 *
 * Every function here is a three-line wrapper: it casts the void *renderer
 * handle back to MGLRenderer and calls the entry point the batch path needs.
 * They exist so C translation units (mgl_batch_issue_encode.c today, its
 * neighbours next) do not have to be Objective-C to drive the renderer, and so
 * that the remaining ObjC surface lives in ONE file instead of being spread
 * over the TUs that will themselves be converted.
 *
 * When an implementation file is converted, its wrapper moves into it (as
 * mglRendererAttachmentTextureFor did) and this shim shrinks -- the goal is for
 * it to disappear, or to be the single platform-shell TU the zero-ObjC target
 * allows.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+BatchPorts_Private.h"
#import "MGLRenderer+RenderPass_Private.h"
#import "MGLRenderer+Binding_Private.h"
#include "mgl_renderer_ports.h"
#include "mgl_batch_restore.h"
#include "mgl_texture_sampler.h"

#include <string.h>   /* mglBatchFlushBegin/RunBatches/TeardownReplay */
#include "mgl_renderer_backend.h"
#include "mgl_batch_mtl_encode.h"  /* mgl_batch_mtl_create_icb */


void *mglRendererCreateIndirectCommandBufferPort(void *renderer, int indexed,
                                                 uint64_t count,
                                                 int *failed_out)
{
    (void)renderer;
    if (failed_out) {
        *failed_out = 0;
    }
    /* The @try/@catch is the reason this one stays ObjC for now: Metal raises
     * when an indirect command buffer cannot be allocated, and that has to
     * become a NULL result the replay path can fall back from. */
    @try {
        return mgl_batch_mtl_create_icb(indexed, count);
    } @catch (NSException *ex) {
        static uint64_t s_hit = 0;
        uint64_t hit = ++s_hit;
        if (hit <= 8ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: ICB creation failed, falling back: %@", ex);
        }
        if (failed_out) {
            *failed_out = 1;
        }
        return NULL;
    }
}


int mglRendererProcessGLStatePort(void *renderer, int draw_command)
{
    return [(__bridge MGLRenderer *)renderer processGLState:draw_command ? true : false]
               ? 1
               : 0;
}



int mglRendererMapBuffersToMTLPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r mapBuffersToMTL]) ? 1 : 0;
}

int mglRendererBindVertexBuffersToCurrentRenderEncoderPort(void *renderer,
                                                           const void *encode_context)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r bindVertexBuffersToCurrentRenderEncoder:
                       (const MGLEncodeContext *)encode_context])
               ? 1
               : 0;
}

int mglRendererBindFragmentBuffersToCurrentRenderEncoderPort(void *renderer,
                                                             const void *encode_context)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r bindFragmentBuffersToCurrentRenderEncoder:
                       (const MGLEncodeContext *)encode_context])
               ? 1
               : 0;
}

int mglRendererBindTexturesToCurrentRenderEncoderPort(void *renderer,
                                                      const void *encode_context)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r bindTexturesToCurrentRenderEncoder:
                       (const MGLEncodeContext *)encode_context])
               ? 1
               : 0;
}

int mglRendererRestoreRenderEncoderAfterTextureUploadPort(void *renderer,
                                                          const char *label)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r restoreRenderEncoderAfterTextureUploadForDraw:label]) ? 1 : 0;
}


int mglRendererBindMTLTexturePort(void *renderer, Texture *texture)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && texture && [r bindMTLTexture:texture]) ? 1 : 0;
}


/* === Batch replay shell (former MGLRenderer+Batch.m) =====================
 * These members are pure renderer plumbing: the dual-proxy invariant, the
 * replay-workspace switch, the lock/exception frame around a flush and the
 * outer exception guard of the C entry point.  They are the ObjC-only part of
 * that file, so they live here; its loops moved to C. */
@implementation MGLRenderer (BatchZeroShell)

/* Locked variant of the flush: the caller holds METAL_LOCK.  The body (and its
 * own @try/@finally around the replay teardown) lives in C. */
- (void)flushDrawBuffer:(GLMContext)glm_ctx
{
    METAL_LOCK();
    mglRendererFlushDrawBufferLockedPort((__bridge void *)self, glm_ctx);
    METAL_UNLOCK();
}

@end

/* C entry point: lease the backend, then flush under an autorelease pool with a
 * last-resort exception guard so a throwing draw never escapes into C. */
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

/* === Batch flush / replay-workspace ports =============================== */

void mglRendererStateAreasPort(void *renderer, MGLRendererStateAreas *areas_out)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!areas_out) {
        return;
    }
    memset(areas_out, 0, sizeof(*areas_out));
    if (!r) {
        return;
    }
    areas_out->core = &r->_core;
    areas_out->backend = r->_backend;
    areas_out->ctx = r->ctx;
    areas_out->batching = &r->_batching;
    /* The manager exposes a const pointer; the record itself is mutable and
     * the flush driver writes the trace-replay identity through it. */
    areas_out->command = (MGLCommandState *)[mglRendererRenderPassManager(r) state];
    areas_out->pipeline_cache = [r->_pipelineCache state];
    areas_out->binding_state_owner = &r->_bindingStateOwner;
    areas_out->fragment_trace_bindings = &r->_resourceFallback.fragmentTextureTraceBindings[0];
}

int mglRendererEnsureWritableCommandBufferPort(void *renderer,
                                               const char *reason)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r ensureWritableCommandBuffer:reason]) ? 1 : 0;
}

int mglRendererCurrentRenderPassMatchesFramebufferPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r currentRenderPassMatchesCurrentFramebuffer]) ? 1 : 0;
}

int mglRendererPrepareRenderPassIfFBOChangedPort(void *renderer, void *batch,
                                                 GLMContext ctx, GLenum *replay_error)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r prepareRenderPassIfFBOChanged:(MGLDrawBatch *)batch
                                          context:ctx
                                      replayError:replay_error])
               ? 1
               : 0;
}

/* The @try/@finally frame the C flush driver cannot express: the teardown in
 * the @finally has to run even when a draw raises. */
void mglRendererFlushDrawBufferLockedPort(void *renderer, GLMContext glm_ctx)
{
    MGLBatchFlushPass pass;
    if (!mglBatchFlushBegin(renderer, glm_ctx, &pass)) {
        return;
    }
    @try {
        mglBatchFlushRunBatches(renderer, glm_ctx, &pass);
    } @finally {
        MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
        if (areas.command) {
            areas.command->traceReplayFlushId = 0u;
            areas.command->traceReplayBatchIndex = 0u;
        }
        mglBatchTeardownReplay(renderer, glm_ctx, &pass);
    }
}
