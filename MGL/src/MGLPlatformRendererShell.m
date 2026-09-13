/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#import "MGLPlatformRendererShell.h"
#import <Metal/Metal.h>
#include "mgl_render.h"

#include <stdio.h>
#include <string.h>
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+BatchPorts_Private.h"
#import "MGLRenderer+RenderPass_Private.h"
#import "MGLRenderer+Binding_Private.h"
#include "mgl_renderer_ports.h"
#include "mgl_batch_restore.h"
#include "mgl_texture_sampler.h"
#include "mgl_renderer_backend.h"
#include "mgl_batch_mtl_encode.h"  /* mgl_batch_mtl_create_icb */

#include <string.h>   /* mglBatchFlushBegin/RunBatches/TeardownReplay */

@implementation MGLPlatformRendererShell

- (instancetype)initWithView:(NSView *)view
{
    self = [super init];
    if (self) {
        _view = view;
        /* Match CAMetalLayer default (display sync on) until glfwSwapInterval. */
        _swapInterval = 1;
    }
    return self;
}


- (int)mglSwapInterval
{
    return _swapInterval;
}

- (BOOL)mglShouldSkipPresentForUnlockedSwap
{
    /* interval==0 + hidden/occluded window: skip CA present so the drawable
     * pool is not paced by the display. Visible windows still present with
     * displaySyncEnabled=NO. Override with MGL_UNLOCKED_SKIP_PRESENT=0/1. */
    static int envMode = -2; /* -2 unset, -1 auto, 0 force off, 1 force on */
    if (envMode == -2) {
        const char *v = getenv("MGL_UNLOCKED_SKIP_PRESENT");
        if (v && v[0] == '0' && v[1] == '\0') {
            envMode = 0;
        } else if (v && v[0] == '1' && v[1] == '\0') {
            envMode = 1;
        } else {
            envMode = -1;
        }
    }
    if (envMode == 0) {
        return NO;
    }
    if (envMode == 1) {
        return YES;
    }
    NSWindow *window = self.view.window;
    if (!window) {
        return YES;
    }
    if (!window.isVisible) {
        return YES;
    }
    if ((window.occlusionState & NSWindowOcclusionStateVisible) == 0) {
        return YES;
    }
    return NO;
}

- (id)mglCreateSystemDefaultDevice
{
    return MTLCreateSystemDefaultDevice();
}

- (BOOL)mglConfigureMetalLayerWithDevice:(id)device
                    requestedPixelFormat:(uint32_t)requestedPixelFormat
                     actualPixelFormat:(uint32_t *)actualPixelFormat
{
    if (!device) return NO;

    const uint32_t fallbackPixelFormat = 80u;
    uint32_t pixelFormat = mglRenderMetalLayerPixelFormatIsSupported(
        requestedPixelFormat) ? requestedPixelFormat : fallbackPixelFormat;
    CAMetalLayer *layer = [[CAMetalLayer alloc] init];
    if (!layer) return NO;

    layer.device = device;
    @try {
        layer.pixelFormat = (MTLPixelFormat)pixelFormat;
    } @catch (NSException *exception) {
        NSLog(@"MGL CAMetalLayer invalid pixelFormat=%u requested=%u exception=%@; falling back to BGRA8Unorm",
              pixelFormat, requestedPixelFormat, exception);
        pixelFormat = fallbackPixelFormat;
        layer.pixelFormat = (MTLPixelFormat)pixelFormat;
    }
    layer.opaque = YES;
    layer.framebufferOnly = NO;
    layer.allowsNextDrawableTimeout = YES;
    layer.magnificationFilter = kCAFilterNearest;
    layer.presentsWithTransaction = NO;
    layer.displaySyncEnabled = (_swapInterval > 0);
    self.layer = layer;

    if (self.view.layer) {
        [self.view.layer addSublayer:layer];
    } else {
        self.view.layer = layer;
    }
    if (actualPixelFormat) *actualPixelFormat = pixelFormat;
    return YES;
}

- (void)mglDetachMetalLayer
{
    self.drawable = nil;
    [self.layer removeFromSuperlayer];
    self.layer = nil;
}

- (id)mglCaptureDescriptorForDevice:(id)device
                         outputPath:(NSString *)outputPath
{
    if (!device || outputPath.length == 0) return nil;
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:outputPath];
    descriptor.captureObject = device;
    return descriptor;
}

- (BOOL)mglStartCaptureWithDescriptor:(id)descriptor
                                error:(NSError **)error
{
    if (!descriptor) return NO;
    return [MTLCaptureManager.sharedCaptureManager
        startCaptureWithDescriptor:(MTLCaptureDescriptor *)descriptor
        error:error];
}

- (void)mglStopCapture
{
    [MTLCaptureManager.sharedCaptureManager stopCapture];
}

- (id)mglNextDrawable
{
    self.drawable = [self.layer nextDrawable];
    return self.drawable;
}

- (id)mglDrawableTexture
{
    return self.drawable.texture;
}


- (BOOL)mglHasMetalLayer
{
    return self.layer != nil;
}

- (CGSize)mglMetalLayerDrawableSize
{
    return self.layer ? self.layer.drawableSize : CGSizeZero;
}

- (CGRect)mglMetalLayerFrame
{
    return self.layer ? self.layer.frame : CGRectZero;
}

- (void)mglSetMetalLayerFrame:(CGRect)frame contentsScale:(CGFloat)scale
{
    if (!self.layer) return;
    self.layer.frame = frame;
    self.layer.contentsScale = scale;
}

- (void)mglSetMetalLayerDrawableSize:(CGSize)size
{
    if (self.layer) self.layer.drawableSize = size;
}

void *mglPlatformRendererShellTextureForDrawable(void *drawable)
{
    if (!drawable) return NULL;
    id<CAMetalDrawable> metalDrawable = (__bridge id<CAMetalDrawable>)drawable;
    return (__bridge void *)metalDrawable.texture;
}

- (int)performOperation:(MGLPlatformRendererShellOperation)operation
                context:(void *)context
                 result:(MGLPlatformRendererShellResult *)result
{
    if (result) memset(result, 0, sizeof(*result));
    if (!operation) return -1;
    @try {
        int status = operation(context);
        if (result) result->status = status;
        return status;
    } @catch (NSException *exception) {
        if (result) {
            result->status = -1;
            snprintf(result->exception_name, sizeof(result->exception_name),
                     "%s", exception.name.UTF8String ?: "NSException");
            snprintf(result->exception_reason, sizeof(result->exception_reason),
                     "%s", exception.reason.UTF8String ?: "unknown");
        }
        return -1;
    }
}

@end


#ifndef MGL_PLATFORM_SHELL_SMOKE
/* The C++ smoke harness compiles this file standalone to check that the shell
 * keeps building as Objective-C++; the port wrappers below need the
 * batch/replay half of the library, so they are compiled out there. */
/* === renderer port shim (merged from MGLPlatformRendererShell.m, T5) =========
 * The Objective-C surface C talks to now lives in this single platform TU:
 * the shell above plus the port wrappers below.  The port count is unchanged
 * (13) - this is the T5 consolidation into one shell translation unit, not a
 * port reduction. */
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

/* C entry point for the pipeline cache's blend setter: the cache object comes
 * from the state areas and the message stays in this Objective-C TU, so C never
 * needs a port for it. */
/* C entry point for the AGX queue recreation: the method lives in
 * MGLRenderer.m where the queue ivar is visible. */
void *mglPlatformShellMetalDevice(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? [r mglMetalDevicePointer] : NULL;
}

int mglPlatformShellMetalObjectsPresent(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? [r mglMetalObjectsPresent] : 0;
}

int mglPlatformShellRecreateCommandQueue(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? [r mglRecreateCommandQueue] : 0;
}

/* C entry point for the pipeline cache's cache reset (same shape as the blend
 * setter: the cache object travels in the state areas). */
int mglPipelineCacheResetCaches(void *pipeline_cache_object)
{
    MGLPipelineCache *cache = (__bridge MGLPipelineCache *)pipeline_cache_object;
    if (!cache) {
        return 0;
    }
    [cache resetCaches];
    return 1;
}

/* Runs a C body with the Objective-C exception guard the renderer's cleanup
 * paths always had.  Kept in the shell TU because @try/@catch has no C form. */
/* Like mglPlatformShellGuardedCall, but with a context argument and an
 * always-run finally - the C twin of @try/@catch/@finally. */
int mglPlatformShellGuardedCallCtx(void *renderer, const char *what,
                                   int (*body)(void *, void *), void *ctx,
                                   void (*finally_fn)(void *, void *))
{
    @try {
        return body ? body(renderer, ctx) : 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "MGL ERROR: Exception during %s: %s\n",
                what ? what : "operation",
                exception.description ? exception.description.UTF8String : "?");
        return 0;
    } @finally {
        if (finally_fn) {
            finally_fn(renderer, ctx);
        }
    }
}

int mglPlatformShellGuardedCall(void *renderer, const char *what,
                                int (*body)(void *))
{
    if (!body) {
        return 0;
    }
    @try {
        return body(renderer);
    } @catch (NSException *exception) {
        fprintf(stderr, "MGL ERROR: Exception during %s: %s\n",
                what ? what : "operation",
                exception.description ? exception.description.UTF8String : "?");
        return 0;
    }
}

int mglPlatformShellPipelineCacheSetBlend(void *pipeline_cache_object,
                                          uint32_t index,
                                          const MGLRenderPipelineBlendState *blend)
{
    MGLPipelineCache *cache = (__bridge MGLPipelineCache *)pipeline_cache_object;
    if (!cache || !blend || index >= MAX_COLOR_ATTACHMENTS) {
        return 0;
    }
    [cache setBlendFactorsForAttachment:(NSUInteger)index
                           srcRgbFactor:blend->source_rgb_factor
                         srcAlphaFactor:blend->source_alpha_factor
                           dstRgbFactor:blend->destination_rgb_factor
                         dstAlphaFactor:blend->destination_alpha_factor
                           rgbOperation:blend->rgb_operation
                         alphaOperation:blend->alpha_operation
                              colorMask:blend->color_write_mask];
    return 1;
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
    /* The manager exposes a const pointer; the record itself is mutable and
     * the flush driver writes the trace-replay identity through it. */
    areas_out->command = (MGLCommandState *)[mglRendererRenderPassManager(r) state];
    areas_out->pipeline_cache = [r->_pipelineCache state];
    areas_out->binding_state_owner = &r->_bindingStateOwner;
    areas_out->pipeline_cache_object = (__bridge void *)r->_pipelineCache;
    areas_out->gpu_recovery_command_owner = &r->_gpuRecovery.commandRecoveryOwner;
    areas_out->pipeline_cache_set_blend = mglPlatformShellPipelineCacheSetBlend;
    areas_out->tess_native_tes_active = (int32_t)r->_tessellation.nativeTESActive;
    areas_out->tess_native_tes_program = (void *)r->_tessellation.nativeTESProgram;
    areas_out->tess_tcs_output_stride = (uint32_t)r->_tessellation.tcsOutputStride;
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


#endif /* MGL_PLATFORM_SHELL_SMOKE */