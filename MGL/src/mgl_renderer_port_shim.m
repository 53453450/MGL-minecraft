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

void *mglRendererMdiScratchBufferPort(void *renderer, uint64_t length,
                                      uint64_t *offset_out)
{
    /* Body of the former -[MGLRenderer mdiArgumentScratchBufferWithLength:
     * offset:]; the render pass manager owns the ring buffer. */
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r) {
        return NULL;
    }
    NSUInteger offset = 0u;
    id buffer = (__bridge id)[mglRendererRenderPassManager(r)
        mdiArgumentScratchBufferWithDevice:mglRendererBackendGetDevice(r->_backend)
                                    length:(NSUInteger)length
                                    offset:&offset];
    if (offset_out) {
        *offset_out = (uint64_t)offset;
    }
    return (__bridge void *)buffer;
}

int mglRendererProcessBufferPort(void *renderer, void *buffer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && buffer && [r processBuffer:(Buffer *)buffer]) ? 1 : 0;
}

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

int mglRendererResolveElementBufferPort(void *renderer, const void *command,
                                        const char *label, GLMContext ctx,
                                        Buffer **gl_buffer_out,
                                        void **mtl_buffer_out)
{
    Buffer *gl_buffer = NULL;
    id mtl_buffer = nil;
    if (![(__bridge MGLRenderer *)renderer
            resolveElementBufferForCommand:(const MGLDrawCommand *)command
                                    label:label
                                  context:ctx
                                 glBuffer:&gl_buffer
                                mtlBuffer:&mtl_buffer]) {
        return 0;
    }
    if (gl_buffer_out) {
        *gl_buffer_out = gl_buffer;
    }
    if (mtl_buffer_out) {
        *mtl_buffer_out = (__bridge void *)mtl_buffer;
    }
    return 1;
}

int mglRendererCaptureCullArrayPort(void *renderer, GLMContext ctx, int32_t first,
                                    int32_t count, int32_t instance_count,
                                    uint32_t base_instance)
{
    return [(__bridge MGLRenderer *)renderer
               captureAIRCullDistancesForArrayDraw:ctx
                                             first:(GLint)first
                                             count:(GLsizei)count
                                     instanceCount:(GLsizei)instance_count
                                      baseInstance:(GLuint)base_instance]
               ? 1
               : 0;
}

int mglRendererCaptureCullElementPort(void *renderer, GLMContext ctx,
                                      const uint8_t *index_bytes,
                                      uint32_t index_type, int32_t count,
                                      int32_t base_vertex,
                                      int32_t instance_count,
                                      uint32_t base_instance)
{
    return [(__bridge MGLRenderer *)renderer
               captureAIRCullDistancesForElementDraw:ctx
                                          indexBytes:index_bytes
                                           indexType:(GLenum)index_type
                                               count:(GLsizei)count
                                          baseVertex:(GLint)base_vertex
                                       instanceCount:(GLsizei)instance_count
                                        baseInstance:(GLuint)base_instance]
               ? 1
               : 0;
}

int mglRendererProcessGLStatePort(void *renderer, int draw_command)
{
    return [(__bridge MGLRenderer *)renderer processGLState:draw_command ? true : false]
               ? 1
               : 0;
}

int mglRendererUpdateDirtyBaseBufferListPort(void *renderer, void *upload)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && upload && [r updateDirtyBaseBufferList:(BufferMapList *)upload]) ? 1 : 0;
}

void mglRendererBindMTLBufferPort(void *renderer, void *buffer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r && buffer) {
        [r bindMTLBuffer:(Buffer *)buffer];
    }
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

void *mglRendererSamplerStateForSnapshotKeyPort(void *renderer, const void *key)
{
    /* Body of the former -[MGLRenderer samplerStateForSnapshotKey:].  The
     * return is unretained, as it was there: the backend snapshot cache holds
     * the state it hands back. */
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r || !key) {
        return NULL;
    }
    void *cachedState = NULL;
    int cacheResult = mglRendererBackendGetSamplerSnapshotState(
        r->_backend, (const MGLSamplerSnapshotKey *)key, &cachedState);
    if (cacheResult == 1) {
        return cachedState;
    }
    if (cacheResult < 0) {
        return NULL;
    }
    TextureParameter params;
    mgl_batch_replay_fill_sampler_params((const MGLSamplerSnapshotKey *)key, &params);
    /* +1 from the C sampler creation; the backend cache below takes ownership
     * through the Put call, so release our reference again. */
    void *state = mglTextureCreateSamplerForTexParam(
        &params, ((const MGLSamplerSnapshotKey *)key)->target);
    if (!state) {
        return NULL;
    }
    if (mglRendererBackendPutSamplerSnapshotState(
            r->_backend, (const MGLSamplerSnapshotKey *)key, state) != 0) {
        mglReleaseMetalObjNoNull(state);
        return NULL;
    }
    mglReleaseMetalObjNoNull(state);   /* the backend snapshot cache retains it */
    return state;
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

int mglRendererBindingStateIsValidPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && mglBindingStateIsValid(r->_bindingStateOwner)) ? 1 : 0;
}

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
    areas_out->command = [mglRendererRenderPassManager(r) state];
    areas_out->pipeline_cache = [r->_pipelineCache state];
    areas_out->binding_state_owner = &r->_bindingStateOwner;
    areas_out->fragment_trace_bindings = &r->_resourceFallback.fragmentTextureTraceBindings[0];
}

const MGLCommandState *mglRendererCommandStatePort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? [mglRendererRenderPassManager(r) state] : NULL;
}

MGLBatchingState *mglRendererBatchingStatePort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? &r->_batching : NULL;
}

void mglRendererTraceReplaySetPort(void *renderer, uint64_t flush_id,
                                   uint32_t batch_index)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [mglRendererRenderPassManager(r) setTraceReplayFlushId:flush_id
                                                    batchIndex:batch_index];
    }
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
        mglRendererTraceReplaySetPort(renderer, 0u, 0u);
        mglBatchTeardownReplay(renderer, glm_ctx, &pass);
    }
}
