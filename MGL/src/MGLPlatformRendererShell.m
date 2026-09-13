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
#import "MGLRenderer+Texture_Private.h"  /* texture upload ports */
#import "MGLRenderer+Binding_Private.h"
#include "mgl_renderer_ports.h"
#include "mgl_batch_restore.h"
#include "mgl_texture_sampler.h"
#include "mgl_texture_bind.h"     /* mglRendererBindMTLTexture (was -bindMTLTextureLocked:) */
#include "mgl_binding_state_ops.h"  /* mglBindingInvalidateLastBoundState */
#include "mgl_compute_dispatch.h"     /* compute dispatch entries */
#import "MGLRenderer+Lifecycle_Private.h"  /* renderer construction/teardown */
#include "mgl.h"
#include "draw_command.h"
#include "mgl_frame_activity.h"  /* MGL_PERF_INC/ADD (pipeline cache counters) */
#include "mgl_air_loader.h"      /* MGLRenderPipelineDescriptorState */
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


/* === draw / tessellation host entries (phase 2) ========================= */
void mglRendererFlushCommandBufferPort(void *renderer, int finish)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [r flushCommandBuffer:finish ? true : false];
    }
}

int mglRendererEnsureRasterEncoderForDrawPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r ensureRasterEncoderForDraw]) ? 1 : 0;
}

int mglRendererPrepareEmulatedIndirectCPUReadPort(void *renderer,
                                                  GLMContext draw_ctx,
                                                  const char *label)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r prepareEmulatedIndirectCPURead:draw_ctx label:label]) ? 1 : 0;
}

int mglRendererEnsureAIRGeometryPassthroughPort(void *renderer,
                                                Program *program,
                                                uint32_t output_primitive)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r ensureAIRGeometryPassthroughFunctionForProgram:program
                                                  outputPrimitive:output_primitive])
               ? 1
               : 0;
}

int mglRendererDispatchTessControlShaderPort(
    void *renderer, GLMContext glm_ctx, Program *program,
    const struct MGLAIRTessDrawContract *contract)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r dispatchTessControlShader:glm_ctx
                                      program:program
                                     contract:contract])
               ? 1
               : 0;
}

int mglRendererDispatchAIRTessEvalComputePort(
    void *renderer, GLMContext glm_ctx, Program *program,
    const struct MGLAIRTessDrawContract *contract, uint32_t patch_count,
    int32_t instance_count, uint32_t base_instance)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r dispatchAIRTessEvalCompute:glm_ctx
                                       program:program
                                      contract:contract
                                    patchCount:patch_count
                                 instanceCount:(GLsizei)instance_count
                                  baseInstance:base_instance])
               ? 1
               : 0;
}

int mglRendererDispatchAIRTessEvalVertexRenderPort(
    void *renderer, GLMContext glm_ctx, Program *program,
    const struct MGLAIRTessDrawContract *contract, uint32_t patch_count,
    int32_t instance_count, uint32_t base_instance)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r dispatchAIRTessEvalVertexRender:glm_ctx
                                            program:program
                                           contract:contract
                                         patchCount:patch_count
                                      instanceCount:(GLsizei)instance_count
                                       baseInstance:base_instance])
               ? 1
               : 0;
}

int mglRendererBindStorageImagesForVertexProgramPort(void *renderer,
                                                     Program *vertex_program,
                                                     Program *fragment_program)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r bindStorageImagesForVertexProgram:vertex_program
                                      fragmentProgram:fragment_program])
               ? 1
               : 0;
}

/* GPU capture: the capture session lives on the shell object, which owns the
 * MTLCaptureManager descriptor/start/stop calls. */
void mglPlatformShellGpuCaptureStart(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    /* MGLRenderer carries the capture methods (they come from the platform
     * shell class), and the renderer owns the backend ivar holding the device. */
    if (!r || !getenv("MGL_GPU_CAPTURE")) {
        return;
    }
    id desc = [r mglCaptureDescriptorForDevice:(__bridge id)mglRendererBackendGetDevice(r->_backend)
                                    outputPath:[NSString stringWithUTF8String:getenv("MGL_GPU_CAPTURE")]];
    NSError *capErr = nil;
    if (desc && [r mglStartCaptureWithDescriptor:desc error:&capErr]) {
        NSLog(@"MGL GPU capture started -> %s", getenv("MGL_GPU_CAPTURE"));
    } else {
        NSLog(@"MGL GPU capture start failed: %@", capErr.localizedDescription);
    }
}

void mglPlatformShellGpuCaptureStop(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [r mglStopCapture];
    }
}

/* === compute / tessellation host entries =================================
 * Thin forwards for the stages that are still Objective-C; see the ownership
 * notes next to their declarations in mgl_renderer_ports.h. */
void mglPlatformShellSetContext(void *renderer, GLMContext glm_ctx)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [r mglSetActiveContext:glm_ctx];
    }
}

int mglRendererBindMTLProgramPort(void *renderer, Program *program)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && program && [r bindMTLProgram:program]) ? 1 : 0;
}

void mglRendererEndRenderEncodingPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [r endRenderEncoding];
    }
}

int mglRendererNewCommandBufferLockedPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r newCommandBufferLocked]) ? 1 : 0;
}

int mglRendererProcessGLStateLockedPort(void *renderer, int draw_command)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r processGLStateLocked:draw_command ? true : false]) ? 1 : 0;
}

void mglRendererClearStageBindingCopyBacksPort(void *renderer, void *copy_backs)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [r clearStageBindingCopyBacks:(MGLStageBindingCopyBackList *)copy_backs];
    }
}

void mglRendererClearStageBindingCopyBackPort(void *renderer, void *copy_backs,
                                              uint64_t index)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [r clearStageBindingCopyBack:(MGLStageBindingCopyBackList *)copy_backs
                             atIndex:(NSUInteger)index];
    }
}

int mglRendererRecordStageBindingCopyBackPort(
    void *renderer, void *copy_backs, uint64_t index, void *temporary,
    void *destination, Buffer *destination_buffer, uint64_t destination_offset,
    uint64_t length)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r recordStageBindingCopyBack:(MGLStageBindingCopyBackList *)copy_backs
                                       atIndex:(NSUInteger)index
                                     temporary:(__bridge id)temporary
                                   destination:(__bridge id)destination
                             destinationBuffer:destination_buffer
                            destinationOffset:(NSUInteger)destination_offset
                                        length:(NSUInteger)length])
               ? 1
               : 0;
}

int mglRendererFlushStageBindingCopyBacksPort(void *renderer, void *copy_backs,
                                              int require_cpu_visibility)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && [r flushStageBindingCopyBacks:(MGLStageBindingCopyBackList *)copy_backs
                          requireCPUVisibility:require_cpu_visibility ? YES : NO])
               ? 1
               : 0;
}

void *mglRendererIsolatedStageBindingBufferPort(void *renderer,
                                                const BufferMap *map,
                                                void *source,
                                                uint64_t required_length)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r) {
        return NULL;
    }
    /* The method returns an autoreleased +0 object; the C caller owns its ref. */
    id isolated = [r isolatedStageBindingBufferForMap:map
                                               source:(__bridge id)source
                                       requiredLength:(NSUInteger)required_length];
    return (void *)CFBridgingRetain(isolated);
}

void *mglRendererMaterializeSampledSamplerPort(
    void *renderer, Texture *texture, uint32_t texture_unit,
    void *default_sampler, int force_default, uint32_t sampler_target,
    uint32_t program_name, uint32_t spirv_binding, const char *stage,
    void *texture_handle)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r) {
        return NULL;
    }
    /* Borrowed sampler: the renderer keeps it (GL sampler object, texture
     * parameters or the default sampler). */
    return (__bridge void *)[r materializeSampledSamplerForTexture:texture
                                                       textureUnit:texture_unit
                                                   defaultSampler:(__bridge id)default_sampler
                                                     forceDefault:force_default ? YES : NO
                                                    samplerTarget:sampler_target
                                                      programName:program_name
                                                     spirvBinding:spirv_binding
                                                            stage:stage
                                                          texture:(__bridge id)texture_handle];
}

void *mglRendererTemporariesCreate(void)
{
    return (void *)CFBridgingRetain([NSMutableArray array]);
}

void mglRendererTemporariesAdd(void *temporaries, void *object)
{
    if (!temporaries || !object) {
        return;
    }
    [(__bridge NSMutableArray *)temporaries addObject:(__bridge id)object];
}

void mglRendererTemporariesRelease(void *temporaries)
{
    if (temporaries) {
        CFBridgingRelease(temporaries);
    }
}

/* The former mglRendererBindMTLTexturePort is gone: the body is the C function
 * mglRendererBindMTLTexture (mgl_texture_bind.h), and so is
 * mglRendererMapBuffersToMTLPort: mglRendererMapBuffersToMTL (mgl_buffer_map.h)
 * replaced it. */


/* === Texture materialization ports (mglRendererBindMTLTexture) ===========
 * The remaining Objective-C half of the texture bind: creation (both paths),
 * the two CPU-data uploads and the default sampler.  MGLRenderer+Texture.m owns
 * the bodies; the CREATE ports hand the +1 back to C through CFBridgingRetain. */

void *mglRendererCreateMTLTextureFromGLTexturePort(void *renderer, Texture *tex)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r || !tex) {
        return NULL;
    }
    return (void *)CFBridgingRetain([r createMTLTextureFromGLTexture:tex]);
}

void *mglRendererCreateFallbackMTLTexturePort(void *renderer, Texture *tex)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r || !tex) {
        return NULL;
    }
    return (void *)CFBridgingRetain([r createFallbackMTLTexture:tex]);
}

int mglRendererUploadFullCPUTextureDataPort(void *renderer, Texture *tex,
                                            void *texture,
                                            const char *reason)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r || !tex) {
        return 0;
    }
    return [r uploadFullCPUTextureDataIntoTexture:tex
                                            metal:(__bridge id)texture
                                           reason:reason]
               ? 1
               : 0;
}

int mglRendererUploadDirtyCPUTextureDataPort(void *renderer, Texture *tex,
                                             void *texture,
                                             uint32_t pixel_format,
                                             uint32_t num_faces,
                                             uint32_t upload_level_count,
                                             int is_array,
                                             int texture1d_backed_by_2d,
                                             int texture1d_array_backed_by_2d_array,
                                             uint32_t tex_type,
                                             int *out_all_levels_uploaded)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r || !tex) {
        return 0;
    }
    BOOL allLevelsUploaded = NO;
    BOOL uploaded = [r
        uploadDirtyCPUTextureData:tex
                            metal:(__bridge id)texture
                      pixelFormat:pixel_format
                        numFaces:(uint)num_faces
                uploadLevelCount:(GLuint)upload_level_count
                         isArray:(BOOL)(is_array != 0)
              texture1DBackedBy2D:(BOOL)(texture1d_backed_by_2d != 0)
        texture1DArrayBackedBy2DArray:(BOOL)(texture1d_array_backed_by_2d_array != 0)
                         texType:tex_type
            outAllLevelsUploaded:&allLevelsUploaded];
    if (out_all_levels_uploaded) {
        *out_all_levels_uploaded = allLevelsUploaded ? 1 : 0;
    }
    return uploaded ? 1 : 0;
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
int mglPlatformShellMSSampleInLoop(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? [r mglMSSampleInLoop] : 0;
}

void mglPlatformShellSetMSSampleState(void *renderer, int in_loop,
                                      int32_t forced, int32_t offset)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (r) {
        [r mglSetMSSampleState:in_loop forced:forced offset:offset];
    }
}

int mglPlatformShellNewCommandBuffer(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? [r mglEnsureNewCommandBuffer] : 0;
}

void *mglPlatformShellDrawable(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? [r mglDrawablePointer] : NULL;
}

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
    areas_out->command = r->_renderPassManager->state;
    areas_out->pipeline_cache = [r->_pipelineCache state];
    areas_out->binding_state_owner = &r->_bindingStateOwner;
    areas_out->pipeline_cache_object = (__bridge void *)r->_pipelineCache;
    areas_out->gpu_recovery_command_owner = &r->_gpuRecovery.commandRecoveryOwner;
    areas_out->pipeline_cache_set_blend = mglPlatformShellPipelineCacheSetBlend;
    areas_out->tess_native_tes_active = (int32_t)r->_tessellation.nativeTESActive;
    areas_out->tessellation = &r->_tessellation;
    areas_out->geometry = &r->_geometry;
    areas_out->tess_native_tes_program = (void *)r->_tessellation.nativeTESProgram;
    areas_out->tess_tcs_output_stride = (uint32_t)r->_tessellation.tcsOutputStride;
    areas_out->tess_cull_capture_first_instance =
        (uint32_t)r->_tessellation.cullDistanceCaptureFirstInstance;
    areas_out->tess_cull_capture_instance_stride =
        (uint32_t)r->_tessellation.cullDistanceCaptureInstanceStride;
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




/* === compute dispatch entries (T5 merge from MGLRenderer+Compute.m) ======
 * The orchestration is the C functions of mgl_compute_dispatch.h; what stays
 * here is the lease/lock frame and the renderer lookup, which need the
 * MGLRenderer type. */
void mglRendererDispatchCompute(GLMContext glm_ctx,
                                unsigned int groups_x,
                                unsigned int groups_y,
                                unsigned int groups_z)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        METAL_LOCK();
        mglComputeMtlDispatchLocked((__bridge void *)renderer, glm_ctx,
                                    groups_x, groups_y, groups_z);
        METAL_UNLOCK();
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDispatchComputeIndirect(GLMContext glm_ctx,
                                        intptr_t indirect)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        METAL_LOCK();
        mglComputeMtlDispatchIndirectLocked((__bridge void *)renderer, glm_ctx,
                                            indirect);
        METAL_UNLOCK();
    }
    mglRendererBackendEnd(&_backend_lease);
}

/* === Texture binding (T5 merge from MGLRenderer+Binding.m) ================
 * The bind body is the C function mglRendererBindMTLTexture (mgl_texture_bind.h);
 * what is left of the category is the lock/thread-assert frame the Objective-C
 * call sites use and the GL entry point above it. */
@implementation MGLRenderer (BindingShell)

- (bool)bindMTLTexture:(Texture *)tex
{
    METAL_LOCK();
    const bool result = mglRendererBindMTLTexture((__bridge void *)self, tex);
    METAL_UNLOCK();
    return result;
}

@end

void mglRendererBindTexture(GLMContext glm_ctx,
                            Texture *texture)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx && texture) {
        (void)[renderer bindMTLTexture:texture];
    }
    mglRendererBackendEnd(&_backend_lease);
}

/* === Renderer lifecycle (T5 merge from MGLRenderer+Lifecycle.m) ===========
 * Construction, the view/window observers and teardown are Cocoa API: KVO on
 * the view, NSWindow notifications, the CALayer/drawable bring-up and the
 * MTLDevice/backend bootstrap.  They are the platform half of the renderer, so
 * they live in the single Objective-C TU; everything they drive is C.
 * See docs section 0.29 for the block table and the shell's line ceiling. */

/* KVO context shared by the observer registration in
 * createMGLRendererAndBindToContext:view: and observeValueForKeyPath:. */
static void *s_kvoViewGeometryContext = &s_kvoViewGeometryContext;

@interface MGLRenderer (LifecycleBackendBoundary)
- (void)mglBackendWillDestroy:(MGLRendererBackendHandle *)backend;
@end

@implementation MGLRenderer (Lifecycle)

#pragma mark C interface to context functions

void mglRendererPlatformBackendWillDestroy(
    void *platform_shell,
    MGLRendererBackendHandle *backend)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)platform_shell;
    [renderer mglBackendWillDestroy:backend];
}

- (id) initMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    if (!window || !glm_ctx) {
        NSLog(@"MGL ERROR: renderer initialization requires a window and GLMContext");
        return nil;
    }
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    if (!renderer) {
        NSLog(@"MGL ERROR: failed to allocate renderer");
        return nil;
    }

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    if (!view) {
        NSLog(@"MGL ERROR: failed to allocate renderer view");
        return nil;
    }

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return renderer;
}

- (id) createMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    if (!window || !glm_ctx) {
        NSLog(@"MGL ERROR: renderer creation requires a window and GLMContext");
        return nil;
    }
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    if (!renderer) {
        NSLog(@"MGL ERROR: failed to allocate renderer");
        return nil;
    }

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    if (!view) {
        NSLog(@"MGL ERROR: failed to allocate renderer view");
        return nil;
    }

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return renderer;
}


void* CppCreateMGLRendererFromContextAndBindToWindow (void *glm_ctx, void *window)
{
    if (!window || !glm_ctx) {
        NSLog(@"MGL ERROR: renderer creation requires a window and GLMContext");
        return NULL;
    }
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    if (!renderer) {
        NSLog(@"MGL ERROR: failed to allocate renderer");
        return NULL;
    }
    NSWindow * w = (__bridge NSWindow *)(window); // just a plain bridge as the autorelease pool will try to release this and crash on exit
    if (!w) {
        NSLog(@"MGL ERROR: invalid window handle");
        return NULL;
    }
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    if (!view) {
        NSLog(@"MGL ERROR: failed to allocate renderer view");
        return NULL;
    }
    [view setWantsLayer:YES];
    //assert(w.contentView);
    //[w.contentView addSubview:view];
    [w setContentView:view];
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    if (![renderer mglRendererIsReady]) {
        NSLog(@"MGL ERROR: renderer initialization failed closed");
        return NULL;
    }
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The context retains the renderer through platform_renderer_shell.
    // The caller must NOT CFRelease/free the returned pointer, and must keep
    // glm_ctx alive while using the returned pointer.
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererHeadless (void *glm_ctx)
{
    if (!glm_ctx) {
        NSLog(@"MGL ERROR: headless renderer creation requires a GLMContext");
        return NULL;
    }
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    if (!renderer) {
        NSLog(@"MGL ERROR: failed to allocate headless renderer");
        return NULL;
    }

    // Create a dummy NSView for headless rendering
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    if (!view) {
        NSLog(@"MGL ERROR: failed to allocate headless renderer view");
        return NULL;
    }
    [view setWantsLayer:YES];

    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    if (![renderer mglRendererIsReady]) {
        NSLog(@"MGL ERROR: headless renderer initialization failed closed");
        return NULL;
    }
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The context retains the renderer through platform_renderer_shell.
    // The caller must NOT CFRelease/free the returned pointer, and must keep
    // glm_ctx alive while using the returned pointer.
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererAndBindToContext (void *glm_ctx)
{
    // Compatibility export used by reference libMGL.dylib.
    // Falls back to headless binding when no Cocoa window is supplied.
    return CppCreateMGLRendererHeadless(glm_ctx);
}

- (void) createMGLRendererAndBindToContext: (GLMContext) glm_ctx view: (NSView *) view
{
    mglClaimGLThread();            /* idempotent; records the init thread as the GL thread */
    ctx = glm_ctx;
    _backend = NULL;
    self.view = view;
    self.layer = nil;
    if (!self.view) {
        NSLog(@"MGL ERROR: failed to bind platform renderer view");
        return;
    }
    _renderPassManager = mglPassManagerCreate();
    mglPassManagerSetRuntimeContext(_renderPassManager, glm_ctx);

    /* start the DontCare frame generation at 1 so it never matches a
     * texture's zero-initialized mtl_rt_frame_generation stamp until that
     * texture is actually written this frame. */
    mglPassManagerSetDontCareFrameGeneration(_renderPassManager, 1u);

    BOOL psoDedupEnabled = mglEnvFlagEnabledDefaultOn("MGL_PSO_DEDUP");
    BOOL depthStencilCacheEnabled = mglEnvFlagEnabledDefaultOn("MGL_DS_CACHE");
    BOOL binaryArchiveEnabled = mglEnvFlagEnabledDefaultOn("MGL_BINARY_ARCHIVE");
    _pipelineCache = [[MGLPipelineCache alloc]
        initWithPSODedupEnabled:psoDedupEnabled
      depthStencilCacheEnabled:depthStencilCacheEnabled
           binaryArchiveEnabled:binaryArchiveEnabled];
    /* Snapshot arena: batch snapshot/commands from bump allocator. */
    _batching.arenaSnapshotEnabled = mglEnvFlagEnabledDefaultOn("MGL_ARENA_SNAPSHOT");
    if (_batching.arenaSnapshotEnabled) {
        if (mglInitBatchArena(&_batching.batchArena, 4u * 1024u * 1024u)) {
            ctx->batch_arena = &_batching.batchArena;
            NSLog(@"MGL INFO: Snapshot arena enabled (initial chunk capacity %zu bytes)",
                  _batching.batchArena.initial_capacity);
        } else {
            _batching.arenaSnapshotEnabled = NO;
            NSLog(@"MGL WARNING: Snapshot arena malloc failed; falling back to per-batch malloc");
        }
    }
    _batching.skipSameKeyRestoreEnabled = mglEnvFlagEnabledDefaultOn("MGL_SKIP_SAME_KEY_RESTORE");
    _batching.dirtyKeyDeltaEnabled = mglEnvFlagEnabledDefaultOn("MGL_DIRTY_KEY_DELTA");
    /* Initialize last-bound render encoder dedup state to a clean slate.
     * The C++ binding state's valid bit starts false so the first bind on the first encoder is
     * never incorrectly skipped. */
    mglBindingInvalidateLastBoundState((__bridge void *)self);
    NSLog(@"MGL INFO: AGX GPU error tracking initialized");
    NSLog(@"MGL INFO: perf gates pso_dedup=%d ds_cache=%d arena=%d "
          "same_key_restore=%d dirty_key_delta=%d (set VAR=0 to disable)",
          _pipelineCache.state->psoDedupEnabled ? 1 : 0,
          _pipelineCache.state->dsCacheEnabled ? 1 : 0,
          _batching.arenaSnapshotEnabled ? 1 : 0,
          _batching.skipSameKeyRestoreEnabled ? 1 : 0,
          _batching.dirtyKeyDeltaEnabled ? 1 : 0);

    if (glm_ctx->renderer_backend) {
        mglRendererBackendDestroy(
            (MGLRendererBackendHandle **)&glm_ctx->renderer_backend);
    }
    if (glm_ctx->platform_renderer_shell) {
        CFRelease(glm_ctx->platform_renderer_shell);
        glm_ctx->platform_renderer_shell = NULL;
    }

    // VIRTUALIZED AGX DETECTION: Create Metal device with virtualization safety
    NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating Metal device with virtualization detection");

    // Create the Metal device
    id device = [self mglCreateSystemDefaultDevice];
    if (!device) {
        NSLog(@"MGL ERROR: Metal device not found - this is required for Apple Silicon");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, AGX GPU error tracking
        //        command recovery owner, _pipeline*Format/
        //        _pipelineCache.state->pipelineProgramName and
        //        _pipelineCache.state->pipelineStateCache.
        //   NIL: _device, _commandQueue, _view.
        // Continuing is pointless without a Metal device — every subsequent
        // operation depends on it.
        return; // Exit early rather than continuing with nil device
    }

    NSLog(@"MGL INFO: Metal device created: %@", device);

    MGLRendererBackendCreateInfo backendInfo = {
        .objc_device = (__bridge void *)device,
        .context = glm_ctx,
        .binding_slot_count = TEXTURE_UNITS,
        .query_capacity = 256u,
    };
    if (mglRendererBackendCreate(&backendInfo, &_backend) != 0) {
        NSLog(@"MGL ERROR: failed to create Metal-cpp renderer backend");
        return;
    }
    glm_ctx->renderer_backend = _backend;
    glm_ctx->platform_renderer_shell = (void *)CFBridgingRetain(self);
    MGLRendererBackendLease initLease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &initLease) != 0) {
        NSLog(@"MGL ERROR: failed to acquire backend lease during init");
        return;
    }
    _bindingStateOwner = mglRendererBackendLeaseGetOwner(
        &initLease, MGL_RENDERER_BACKEND_OWNER_BINDING);
    _queryStateOwner = mglRendererBackendLeaseGetOwner(
        &initLease, MGL_RENDERER_BACKEND_OWNER_QUERY);
    _gpuRecovery.commandRecoveryOwner = mglRendererBackendLeaseGetOwner(
        &initLease, MGL_RENDERER_BACKEND_OWNER_RECOVERY);
    NSLog(@"MGL INFO: Metal-cpp renderer backend ready (%p)", _backend);
    mglRenderAttachRuntimeOwners(
        glm_ctx,
        _renderPassManager->state->currentCommandBufferOwner,
        _renderPassManager->state->currentRenderEncoderOwner,
        _renderPassManager->state->renderPassStateOwner);
    _pipelineCache.device = _device;

    /* Initialize AGX Capability Layer (centralized device detection +
     * capability queries + driver bug markers).  Replaces scattered
     * `containsString:@"AGX"` checks and hardcoded constants. */
    MGLCapabilityInit(&_capability, (__bridge void *)_device);

    // PROPER AGX VIRTUALIZATION DETECTION: Maintain Metal functionality with virtualization compatibility
    BOOL isVirtualized = _capability.isVirtualized;
    char deviceName[128];
    (void)mglRenderGetDeviceIdentity(
        (__bridge const void *)_device, NULL,
        deviceName, sizeof(deviceName));

    // DETECTION: Check if running in QEMU virtualization but keep Metal enabled
    if (isVirtualized) {
        isVirtualized = YES;
        NSLog(@"MGL INFO: AGX device detected - enabling virtualization compatibility mode: %s", deviceName);
        NSLog(@"MGL INFO: Metal functionality will be maintained with AGX virtualization safety measures");
    }

    // Create command queue with virtualization-safe settings
    if (isVirtualized) {
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Enabling virtualization-safe command queue settings");
    }

    uint32_t maxCommandBuffers = isVirtualized
        ? (uint32_t)MGLCapabilityMaxConcurrentCommandBuffers(&_capability)
        : 0u;
    void *commandQueue = NULL;
    (void)mglRendererBackendResetCommandQueue(
        _backend, maxCommandBuffers, &commandQueue);
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Failed to create Metal command queue");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, AGX GPU error tracking
        //        fields, _pipeline*Format/_pipelineCache.state->pipelineProgramName,
        //        _pipelineCache.state->pipelineStateCache, _device,
        //        MTL4 compiler (if available), _capability.
        //   NIL: _commandQueue, _view.
        // Continuing is pointless without a command queue — no encoding or
        // submission is possible.
        mglRendererBackendEnd(&initLease);
        return;
    }

    NSLog(@"MGL INFO: Metal command queue created successfully");

    /* Load or create Binary Archive for PSO compile acceleration.
     * Gated by MGL_BINARY_ARCHIVE (default ON; =0 disables).
     * The archive is stored in the user's Caches directory and persists
     * compiled PSO binaries across launches, reducing cold-start PSO
     * compile time from ~10s to ~2s on subsequent launches. */
    if (_pipelineCache.binaryArchiveEnabled) {
        if (@available(macOS 11.0, *)) {
            [_pipelineCache loadBinaryArchive];
        } else {
            [_pipelineCache disableBinaryArchive];
        }
    }

    _view = view;

    // PROPER FIX: Create Metal layer with AGX-safe settings in the platform shell.
    NSLog(@"MGL INFO: PROPER FIX - Creating Metal layer with AGX-safe settings");

    uint32_t requestedPixelFormat = ctx ? ctx->pixel_format.mtl_pixel_format : 0u;
    uint32_t pf = 0u;
    if (![self mglConfigureMetalLayerWithDevice:_device
                          requestedPixelFormat:requestedPixelFormat
                           actualPixelFormat:&pf]) {
        NSLog(@"MGL ERROR: Failed to create Metal layer");
        mglRendererBackendEnd(&initLease);
        return;
    }

    if (ctx && ctx->pixel_format.mtl_pixel_format != (GLuint)pf) {
        NSLog(@"MGL CAMetalLayer sync default framebuffer metal format glFormat=0x%x glType=0x%x oldMtl=%u newMtl=%lu",
              ctx->pixel_format.format,
              ctx->pixel_format.type,
              ctx->pixel_format.mtl_pixel_format,
              (unsigned long)pf);
        ctx->pixel_format.mtl_pixel_format = (GLuint)pf;
    }
    NSLog(@"MGL CAMetalLayer pixelFormat=%lu requested=%lu glFormat=0x%x glType=0x%x",
          (unsigned long)pf,
          (unsigned long)requestedPixelFormat,
          ctx ? ctx->pixel_format.format : 0u,
          ctx ? ctx->pixel_format.type : 0u);
    /* Initial geometry: the renderer is created on the main thread (AppKit
     * window setup), so read the view geometry synchronously here.  Later
     * changes arrive via KVO → mglMainThreadSyncViewGeometry. */
    if (NSThread.isMainThread) {
        [self mglMainThreadSyncViewGeometry];
    } else {
        (void)[self mglApplyPendingDrawableSize];
    }

    /* Observe view geometry changes so the GL thread never needs to touch
     * NSView/NSWindow/NSScreen.  KVO fires on the main thread (bounds is only
     * mutated there), publishing an atomic drawable-size snapshot.  The
     * "window" keyPath is observed as well so resize/backing notifications
     * can be attached lazily once the view joins a window. */
    [_view addObserver:self
            forKeyPath:@"bounds"
               options:0
               context:s_kvoViewGeometryContext];
    [_view addObserver:self
            forKeyPath:@"window"
               options:NSKeyValueObservingOptionInitial
               context:s_kvoViewGeometryContext];

    mglDrawBuffer(glm_ctx, (GLenum)mglRenderDefaultFrontBuffer());

    // Create initial command buffer for AGX safety
    @try {
        mglPassManagerInstallNewCommandBufferFromQueue(_renderPassManager, (__bridge void *)_commandQueue);
        MGLRenderCommandBufferState commandState = {0};
        if (!mglRenderCommandBufferOwnerHasState(
                _renderPassManager->state->currentCommandBufferOwner,
                &commandState)) {
            NSLog(@"MGL ERROR: Failed to create initial Metal command buffer");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating initial Metal command buffer: %@", exception);
    }
    
    // PROACTIVE TEXTURE CREATION: Create essential textures to break sync loop
    NSLog(@"MGL INFO: PROACTIVE - Creating essential textures to prevent magenta screen");
    [self createProactiveTextures];

    // GPU capture setup is exposed by MGLPlatformRendererShell when needed.
    mglRendererBackendEnd(&initLease);
}

- (BOOL)mglRendererIsReady
{
    if (!ctx) {
        return NO;
    }
    MGLRendererBackendLease lease = {};
    if (mglRendererBackendBeginContext(ctx, &lease) != 0) {
        return NO;
    }
    BOOL ready =
        _backend && _device &&
        mglRendererBackendIsReady(_backend) == 1 &&
        _commandQueueOwner && _commandQueue && _layer && _renderPassManager;
    if (ready) {
        MGLRenderCommandBufferState commandState = {0};
        ready = mglRenderCommandBufferOwnerHasState(
            _renderPassManager->state->currentCommandBufferOwner,
            &commandState);
    }
    mglRendererBackendEnd(&lease);
    return ready;
}

- (void)mglBackendWillDestroy:(MGLRendererBackendHandle *)backend
{
    if (_backend != backend) return;
    _backend = NULL;
    _bindingStateOwner = NULL;
    _queryStateOwner = NULL;
    _gpuRecovery.commandRecoveryOwner = NULL;
}

/* Publish view geometry to the GL thread as an atomic snapshot.  Main thread
 * only — this is the sole place NSView/NSWindow/NSScreen are read, so the
 * render thread never touches AppKit.  The GL thread consumes the snapshot via
 * mglApplyPendingDrawableSize and sets CAMetalLayer.drawableSize. */
- (void)mglMainThreadSyncViewGeometry
{
    NSAssert(NSThread.isMainThread, @"AppKit geometry must be read on main thread");
    if (!_view || ![self mglHasMetalLayer]) {
        return;
    }

    NSRect bounds = [_view bounds];
    if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0) {
        bounds = [_view frame];
        bounds.origin = NSZeroPoint;
    }

    NSRect backingBounds = [_view convertRectToBacking:bounds];
    CGFloat scale = 1.0;
    if (bounds.size.width > 0.0 && backingBounds.size.width > 0.0) {
        scale = backingBounds.size.width / bounds.size.width;
    } else {
        NSWindow *window = [_view window];
        if (window) {
            scale = [window backingScaleFactor];
        } else if ([NSScreen mainScreen]) {
            scale = [[NSScreen mainScreen] backingScaleFactor];
        }
        if (scale <= 0.0) {
            scale = 1.0;
        }
        backingBounds = NSMakeRect(0.0, 0.0, bounds.size.width * scale, bounds.size.height * scale);
    }

    [self mglSetMetalLayerFrame:bounds contentsScale:scale];

    uint32_t pw = (uint32_t)MAX(1.0, backingBounds.size.width + 0.5);
    uint32_t ph = (uint32_t)MAX(1.0, backingBounds.size.height + 0.5);
    atomic_store_explicit(&_pendingDrawableW, pw, memory_order_relaxed);
    atomic_store_explicit(&_pendingDrawableH, ph, memory_order_relaxed);
    atomic_store_explicit(&_drawableSizeDirty, true, memory_order_release);
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context
{
    if (context == s_kvoViewGeometryContext) {
        if ([keyPath isEqualToString:@"window"]) {
            [self mglUpdateWindowNotificationObserver];
        }
        [self mglMainThreadSyncViewGeometry];
        return;
    }
    [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
}

/* Attach/detach window observation as the view's window changes.  The window
 * is not known when the renderer is created, so this is wired lazily. */
- (void)mglUpdateWindowNotificationObserver
{
    NSWindow *window = _view.window;
    if (window == _observedWindow) {
        return;
    }
    if (_observedWindow) {
        [[NSNotificationCenter defaultCenter] removeObserver:self
                                                        name:NSWindowDidResizeNotification
                                                      object:_observedWindow];
        [[NSNotificationCenter defaultCenter] removeObserver:self
                                                        name:NSWindowDidChangeBackingPropertiesNotification
                                                      object:_observedWindow];
    }
    _observedWindow = window;
    if (window) {
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(mglWindowGeometryChanged:)
                                                     name:NSWindowDidResizeNotification
                                                   object:window];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(mglWindowGeometryChanged:)
                                                     name:NSWindowDidChangeBackingPropertiesNotification
                                                   object:window];
    }
}

- (void)mglWindowGeometryChanged:(NSNotification *)notification
{
    (void)notification;
    [self mglMainThreadSyncViewGeometry];
}

// PROACTIVE TEXTURE CREATION: Create essential textures during initialization to break sync loop
- (void)createProactiveTextures
{
    NSLog(@"MGL PROACTIVE: Starting essential texture creation");

    @try {
        if (mglRendererBackendCreateProactiveTexture(_backend) == 0) {
            NSLog(@"MGL PROACTIVE SUCCESS: Created 256x256 gradient texture (prevents magenta screen)");
        } else {
            NSLog(@"MGL PROACTIVE ERROR: Could not create proactive texture");
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL PROACTIVE ERROR: Exception creating proactive textures: %@", exception.reason);
    }

    NSLog(@"MGL PROACTIVE: Essential texture creation completed");
}

// CRITICAL FIX: Proper resource cleanup to prevent memory leaks and crashes
- (void)dealloc
{
    NSLog(@"MGL INFO: MGLRenderer dealloc - cleaning up Metal resources");

    @try {
        /* Remove the geometry observers before any view/state teardown. */
        if (_view) {
            [_view removeObserver:self forKeyPath:@"bounds" context:s_kvoViewGeometryContext];
            [_view removeObserver:self forKeyPath:@"window" context:s_kvoViewGeometryContext];
        }
        /* Detach window notifications without the lazy re-wiring path. */
        if (_observedWindow) {
            [[NSNotificationCenter defaultCenter] removeObserver:self
                                                            name:NSWindowDidResizeNotification
                                                          object:_observedWindow];
            [[NSNotificationCenter defaultCenter] removeObserver:self
                                                            name:NSWindowDidChangeBackingPropertiesNotification
                                                          object:_observedWindow];
            _observedWindow = nil;
        }

        // Stop any ongoing capture
        [self mglStopCapture];

        // End any active rendering
        [self endRenderEncoding];

        /* Drop strong references held by the last-bound dedup cache before
         * releasing the underlying Metal resources below. */
        mglBindingInvalidateLastBoundState((__bridge void *)self);
        // Cleanup command buffer and encoder
        MGLRenderCommandBufferState commandState = {0};
        if (mglRenderCommandBufferOwnerHasState(
                _renderPassManager->state->currentCommandBufferOwner,
                &commandState)) {
            NSLog(@"MGL INFO: Releasing current command buffer");
            mglPassManagerDiscardCurrentCommandBuffer(_renderPassManager);
        }

        if (mglRenderEncoderOwnerHasCurrent(
                _renderPassManager->state->currentRenderEncoderOwner) == 1) {
            NSLog(@"MGL INFO: Releasing current render encoder");
            mglPassManagerClearCurrentRenderEncoder(_renderPassManager);
        }

        MGLRendererBackendShutdownResult shutdownResult = {0};
        if (_backend &&
            mglRendererBackendShutdown(_backend, &shutdownResult) != 0) {
            NSLog(@"MGL ERROR: renderer backend shutdown wait failed code=%lld",
                  shutdownResult.last_submission_error_code);
        }

        mglPassManagerSetRuntimeContext(_renderPassManager, NULL);
        mglPassManagerDestroy(_renderPassManager);
        _renderPassManager = NULL;

        mglRenderDetachRuntimeOwners(ctx);

        if (_pipelineCache) {
            if (_pipelineCache.state->pipelineState) {
                NSLog(@"MGL INFO: Releasing pipeline state");
            }
            [_pipelineCache saveBinaryArchive];
            [_pipelineCache shutdown];
            _pipelineCache = nil;
        }
        if (_backend && mglRendererBackendIsDestroying(_backend) != 1) {
            if (ctx && ctx->renderer_backend == _backend) {
                mglRendererBackendDestroy(
                    (MGLRendererBackendHandle **)&ctx->renderer_backend);
            } else {
                mglRendererBackendDestroy(&_backend);
            }
        }
        _backend = NULL;
        _bindingStateOwner = NULL;
        _queryStateOwner = NULL;
        _gpuRecovery.commandRecoveryOwner = NULL;

        // Cleanup drawable and layer
        if (_drawable) {
            NSLog(@"MGL INFO: Releasing drawable");
            _drawable = nil;
        }

        if ([self mglHasMetalLayer]) {
            NSLog(@"MGL INFO: Removing and releasing layer");
            [self mglDetachMetalLayer];
        }

        /* Task 4: Release all address-stable snapshot arena chunks. */
        mglDestroyBatchArena(&_batching.batchArena);
        _view = nil;

    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during dealloc cleanup: %@", exception);
    }

    NSLog(@"MGL INFO: MGLRenderer dealloc completed");
}

@end

/* === MGLPipelineCache (T5 merge from MGLPipelineCache.m) ====================
 * The pipeline cache is a platform object: its state record is already
 * C (MGLPipelineCacheState), its owner is a C++ handle and the rest is
 * Foundation archive-path plumbing (NSBundle/NSFileManager/NSURL).  It therefore
 * belongs in the single Objective-C TU; the class interface stays in
 * MGLPipelineCache.h.
 *
 * Deferred (see docs section 0.28): turning the class itself into a C handle.
 * It has only 15 message-send sites left, but its archive path is Foundation
 * code whose observable behaviour the A/B oracle deliberately filters out
 * ("BINARY ARCHIVE" lines), so that conversion wants its own oracle first. */

@interface MGLPipelineCache ()
- (BOOL)ensureOwnerCreated;
- (BOOL)ensureOwner;
@end

/* v5 excludes either kind of incomplete render pipeline and isolates both
 * sanitizer builds and archive producers. The producer boundary prevents the
 * temporary A/B implementations from sharing mutable state; the archive-aware
 * PSO creation path below separately prevents repeated adds on cache hits. */
#if __has_feature(address_sanitizer)
static NSString * const kMGLPipelineArchiveBuildSchema = @"v5-asan";
#elif __has_feature(thread_sanitizer)
static NSString * const kMGLPipelineArchiveBuildSchema = @"v5-tsan";
#else
static NSString * const kMGLPipelineArchiveBuildSchema = @"v5";
#endif

static NSString *MGLSafeArchivePathComponent(NSString *value)
{
    if (value.length == 0) return @"unknown";
    NSCharacterSet *unsafe = [[NSCharacterSet alphanumericCharacterSet] invertedSet];
    return [[value componentsSeparatedByCharactersInSet:unsafe] componentsJoinedByString:@"_"];
}

@implementation MGLPipelineCache

- (instancetype)initWithPSODedupEnabled:(BOOL)psoDedupEnabled
                depthStencilCacheEnabled:(BOOL)depthStencilCacheEnabled
                     binaryArchiveEnabled:(BOOL)binaryArchiveEnabled
{
    self = [super init];
    if (!self) return nil;

    _state.pipelineColor0Format = 0u;
    _state.pipelineDepthFormat = 0u;
    _state.pipelineStencilFormat = 0u;
    _state.psoDedupEnabled = psoDedupEnabled;
    _state.dsCacheEnabled = depthStencilCacheEnabled;
    _binaryArchiveRequested = binaryArchiveEnabled;
    return self;
}

- (const MGLPipelineCacheState *)state
{
    return &_state;
}

- (BOOL)ensureOwnerCreated
{
    if (_owner) return YES;
    if (!_cacheDevice) return NO;
    if (mglRenderCreatePipelineCacheOwner(
            _state.psoDedupEnabled ? 1 : 0,
            _state.dsCacheEnabled ? 1 : 0,
            _binaryArchiveRequested ? 1 : 0,
            &_owner) != 0 || !_owner) {
        _owner = NULL;
        return NO;
    }

    MGLRenderPipelineActiveState active = {
        .pipeline_state = _state.pipelineState,
        .vertex_function = _state.pipelineVertexFunction,
        .fragment_function = _state.pipelineFragmentFunction,
        .color0_format = (uint32_t)_state.pipelineColor0Format,
        .depth_format = (uint32_t)_state.pipelineDepthFormat,
        .stencil_format = (uint32_t)_state.pipelineStencilFormat,
        .program_name = _state.pipelineProgramName,
    };
    mglRenderActivatePipelineState(_owner, &active);
    return YES;
}

- (BOOL)ensureOwner
{
    return [self ensureOwnerCreated];
}

- (BOOL)isBinaryArchiveEnabled
{
    int enabled = _binaryArchiveRequested ? 1 : 0;
    if (_owner) {
        mglRenderGetPipelineBinaryArchiveState(
            _owner, &enabled, NULL);
    }
    return enabled != 0;
}

- (id)device
{
    return (__bridge id)_cacheDevice;
}

- (void)setDevice:(id)device
{
    void *opaqueDevice = (__bridge void *)device;
    if (_cacheDevice != opaqueDevice) {
        mglRenderDestroyPipelineCacheOwner(&_owner);
    }
    _cacheDevice = opaqueDevice;
    if (_cacheDevice) [self ensureOwnerCreated];
}

- (id)depthStencilStateForValueState:
    (const MGLRenderDepthStencilDescriptorState *)descriptorState
{
    if (!descriptorState || !_cacheDevice || ![self ensureOwner]) return nil;
    void *statePtr = NULL;
    if (_state.dsCacheEnabled) {
        int created = 0;
        if (mglRenderGetOrCreateDepthStencilState(
                _owner, descriptorState, &statePtr, &created) == 0 &&
            statePtr) {
            if (created) MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
            return (__bridge id)statePtr;
        }
        return nil;
    }
    if (mglRenderCreateDepthStencilStateFromState(
            descriptorState, &statePtr) == 0 && statePtr) {
        MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
        return (__bridge_transfer id)statePtr;
    }
    return nil;
}

- (BOOL)lookupPipelineForWords:(const uint64_t *)words
                      pipeline:(id *)pipelineOut
                vertexFunction:(id *)vertexFunctionOut
              fragmentFunction:(id *)fragmentFunctionOut
{
    if (pipelineOut) *pipelineOut = nil;
    if (vertexFunctionOut) *vertexFunctionOut = nil;
    if (fragmentFunctionOut) *fragmentFunctionOut = nil;
    if (!words || !pipelineOut || !vertexFunctionOut ||
        !fragmentFunctionOut) {
        return NO;
    }
    if (![self ensureOwner]) return NO;
    MGLRenderPipelineActiveState cached = {0};
    if (mglRenderLookupPipeline(_owner, words, &cached) != 1 ||
        !cached.pipeline_state) {
        return NO;
    }
    *pipelineOut = (__bridge id)cached.pipeline_state;
    *vertexFunctionOut = (__bridge id)cached.vertex_function;
    *fragmentFunctionOut = (__bridge id)cached.fragment_function;
    return YES;
}

- (NSUInteger)storePipeline:(id)pipeline
              vertexFunction:(id)vertexFunction
            fragmentFunction:(id)fragmentFunction
                    forWords:(const uint64_t *)words
{
    if (!pipeline || !words) return 0;
    if (![self ensureOwner]) return 0;
    MGLRenderPipelineActiveState state = {
        .pipeline_state = (__bridge void *)pipeline,
        .vertex_function = (__bridge void *)vertexFunction,
        .fragment_function = (__bridge void *)fragmentFunction,
    };
    uint32_t removed = 0;
    if (mglRenderStorePipeline(
            _owner, words, &state, &removed) != 0) {
        return 0;
    }
    MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, removed);
    return (NSUInteger)removed;
}

- (BOOL)pipelineDescriptorStateForWords:(const uint64_t *)words
                                  state:(MGLRenderPipelineDescriptorState *)stateOut
{
    if (!words || !stateOut) return NO;
    return [self ensureOwner] &&
        mglRenderLookupPipelineDescriptorState(
            _owner, words, stateOut) == 1;
}

- (void)storePipelineDescriptorState:(const MGLRenderPipelineDescriptorState *)state
                            forWords:(const uint64_t *)words
{
    if (!state || !words) return;
    if (![self ensureOwner]) return;
    mglRenderStorePipelineDescriptorState(_owner, words, state);
}

- (BOOL)blendStateForAttachment:(NSUInteger)index
                            out:(MGLRenderPipelineBlendState *)outState
{
    if (index >= MAX_COLOR_ATTACHMENTS || !outState) return NO;
    return [self ensureOwner] &&
        mglRenderGetPipelineBlendState(
            _owner, (uint32_t)index, outState) == 0;
}

- (NSURL *)binaryArchiveURL
{
    NSArray *caches = NSSearchPathForDirectoriesInDomains(NSCachesDirectory,
                                                          NSUserDomainMask, YES);
    NSString *baseDir = caches.firstObject ?: NSTemporaryDirectory();
    NSString *bundleID = NSBundle.mainBundle.bundleIdentifier;
    if (bundleID.length == 0) bundleID = NSProcessInfo.processInfo.processName;
    NSString *mglDir = [[baseDir stringByAppendingPathComponent:@"MGL"]
                        stringByAppendingPathComponent:MGLSafeArchivePathComponent(bundleID)];
    NSFileManager *fileManager = NSFileManager.defaultManager;
    if (![fileManager fileExistsAtPath:mglDir]) {
        [fileManager createDirectoryAtPath:mglDir
               withIntermediateDirectories:YES
                                attributes:nil
                                     error:NULL];
    }

    uint64_t registryID = 0;
    char deviceName[256] = {0};
    (void)mglRenderGetDeviceIdentity(_cacheDevice,
                                        &registryID, deviceName,
                                        sizeof(deviceName));
    NSString *deviceNameString = deviceName[0]
        ? [NSString stringWithUTF8String:deviceName]
        : @"unknown";
    NSString *deviceID = registryID != 0
        ? [NSString stringWithFormat:@"%016llx", (unsigned long long)registryID]
        : MGLSafeArchivePathComponent(deviceNameString);
    NSString *schema = [NSString stringWithFormat:@"%@-cpp",
                        kMGLPipelineArchiveBuildSchema];
    NSString *filename = [NSString stringWithFormat:@"pipeline-%@-%@.binaryarchive",
                          schema, deviceID];
    return [NSURL fileURLWithPath:[mglDir stringByAppendingPathComponent:filename]];
}

- (void)loadBinaryArchive
{
    if (!self.binaryArchiveEnabled || !_cacheDevice ||
        ![self ensureOwnerCreated]) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSString *archiveKey = archiveURL.path;
    NSFileManager *fileManager = NSFileManager.defaultManager;
    BOOL archiveExists = [fileManager fileExistsAtPath:archiveKey];
    int reused = 0;
    char message[512] = {0};
    int result = mglRenderLoadPipelineBinaryArchive(
        _owner, archiveKey.UTF8String, (__bridge void *)archiveURL,
        archiveExists ? 1 : 0, &reused, message, sizeof(message));
    if (result != 0 && archiveExists) {
        NSError *removeError = nil;
        if (![fileManager removeItemAtURL:archiveURL error:&removeError]) {
            NSLog(@"MGL BINARY ARCHIVE: failed to remove incompatible archive: %@",
                  removeError.localizedDescription);
        }
        NSLog(@"MGL BINARY ARCHIVE: rebuilding incompatible archive: %s",
              message[0] ? message : "unknown error");
        archiveExists = NO;
        message[0] = '\0';
        result = mglRenderLoadPipelineBinaryArchive(
            _owner, archiveKey.UTF8String, (__bridge void *)archiveURL,
            0, &reused, message, sizeof(message));
    }
    if (result == 0) {
        NSLog(@"MGL BINARY ARCHIVE: %@ %@",
              reused ? @"reused" : (archiveExists ? @"loaded" : @"created"),
              archiveURL.lastPathComponent);
    } else {
        NSLog(@"MGL BINARY ARCHIVE: unavailable, PSO compile will continue without it: %s",
              message[0] ? message : "unknown error");
    }
}

- (void)saveBinaryArchive
{
    int present = 0;
    if (!_owner ||
        mglRenderGetPipelineBinaryArchiveState(
            _owner, NULL, &present) != 0 || !present) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSString *archiveKey = archiveURL.path;
    NSError *removeError = nil;
    char message[512] = {0};
    BOOL ok = mglRenderSerializePipelineBinaryArchive(
        _owner, (__bridge void *)archiveURL,
        message, sizeof(message)) == 0;
    BOOL discarded = NO;
    if (!ok) {
        NSFileManager *fileManager = NSFileManager.defaultManager;
        discarded = ![fileManager fileExistsAtPath:archiveKey] ||
            [fileManager removeItemAtURL:archiveURL error:&removeError];
        mglRenderDiscardPipelineBinaryArchive(
            _owner, archiveKey.UTF8String);
    }
    if (ok) {
        NSLog(@"MGL BINARY ARCHIVE: saved to %@", archiveURL.lastPathComponent);
    } else {
        NSString *description = message[0]
            ? [NSString stringWithUTF8String:message] : @"unknown error";
        if (discarded) {
            NSLog(@"MGL BINARY ARCHIVE: discarded unserializable archive: %@",
                  description);
        } else {
            NSLog(@"MGL BINARY ARCHIVE: serialize failed: %@; removal failed: %@",
                  description,
                  removeError.localizedDescription);
        }
    }
}

- (int)createRenderPipelineFromState:
    (const MGLRenderPipelineDescriptorState *)state
    vertexFunction:(void *)vertexFunction
    fragmentFunction:(void *)fragmentFunction
    pipelineOut:(void **)pipelineOut
    errorMessage:(char *)errorMessage
    errorCapacity:(size_t)errorCapacity
{
    if (![self ensureOwnerCreated]) return -1;
    return mglRenderCreateRenderPipelineFromStateWithArchiveOwner(
        _owner, vertexFunction, fragmentFunction, state,
        pipelineOut, errorMessage, errorCapacity);
}

- (void)invalidatePipelineState
{
    if ([self ensureOwner]) {
        mglRenderInvalidatePipelineActiveState(_owner);
    }
    _state.pipelineState = NULL;
    _state.pipelineColor0Format = 0u;
    _state.pipelineDepthFormat = 0u;
    _state.pipelineStencilFormat = 0u;
    _state.pipelineProgramName = 0u;
    _state.pipelineVertexFunction = NULL;
    _state.pipelineFragmentFunction = NULL;
}

- (void)setPipelineState:(id)pipelineState
{
    if ([self ensureOwner]) {
        mglRenderSetPipelineActiveObject(
            _owner, (__bridge void *)pipelineState);
    }
    _state.pipelineState = (__bridge void *)pipelineState;
}

- (void)activatePipelineState:(id)pipelineState
                 color0Format:(uint32_t)color0Format
                  depthFormat:(uint32_t)depthFormat
                stencilFormat:(uint32_t)stencilFormat
                  programName:(GLuint)programName
               vertexFunction:(id)vertexFunction
             fragmentFunction:(id)fragmentFunction
{
    if ([self ensureOwner]) {
        MGLRenderPipelineActiveState active = {
            .pipeline_state = (__bridge void *)pipelineState,
            .vertex_function = (__bridge void *)vertexFunction,
            .fragment_function = (__bridge void *)fragmentFunction,
            .color0_format = (uint32_t)color0Format,
            .depth_format = (uint32_t)depthFormat,
            .stencil_format = (uint32_t)stencilFormat,
            .program_name = programName,
        };
        mglRenderActivatePipelineState(_owner, &active);
    }
    _state.pipelineState = (__bridge void *)pipelineState;
    _state.pipelineColor0Format = (uint64_t)color0Format;
    _state.pipelineDepthFormat = (uint64_t)depthFormat;
    _state.pipelineStencilFormat = (uint64_t)stencilFormat;
    _state.pipelineProgramName = programName;
    _state.pipelineVertexFunction = (__bridge void *)vertexFunction;
    _state.pipelineFragmentFunction = (__bridge void *)fragmentFunction;
}

- (void)setBlendFactorsForAttachment:(NSUInteger)index
                        srcRgbFactor:(uint32_t)srcRgbFactor
                      srcAlphaFactor:(uint32_t)srcAlphaFactor
                        dstRgbFactor:(uint32_t)dstRgbFactor
                      dstAlphaFactor:(uint32_t)dstAlphaFactor
                        rgbOperation:(uint32_t)rgbOperation
                      alphaOperation:(uint32_t)alphaOperation
                           colorMask:(uint32_t)colorMask
{
    if (index >= MAX_COLOR_ATTACHMENTS) return;
    if ([self ensureOwner]) {
        MGLRenderPipelineBlendState blend = {
            .source_rgb_factor = (uint32_t)srcRgbFactor,
            .destination_rgb_factor = (uint32_t)dstRgbFactor,
            .source_alpha_factor = (uint32_t)srcAlphaFactor,
            .destination_alpha_factor = (uint32_t)dstAlphaFactor,
            .rgb_operation = (uint32_t)rgbOperation,
            .alpha_operation = (uint32_t)alphaOperation,
            .color_write_mask = (uint32_t)colorMask,
        };
        mglRenderSetPipelineBlendState(
            _owner, (uint32_t)index, &blend);
    }
}

- (void)disableBinaryArchive
{
    _binaryArchiveRequested = NO;
    if ([self ensureOwnerCreated]) {
        mglRenderDisablePipelineBinaryArchive(_owner);
    }
}

- (void)resetCaches
{
    mglRenderResetPipelineCacheOwner(_owner);
    _state.pipelineState = NULL;
    _state.pipelineVertexFunction = NULL;
    _state.pipelineFragmentFunction = NULL;
}

- (void)shutdown
{
    [self resetCaches];
    _cacheDevice = NULL;
    mglRenderDestroyPipelineCacheOwner(&_owner);
}

- (void)dealloc
{
    mglRenderDestroyPipelineCacheOwner(&_owner);
}

@end

#endif /* MGL_PLATFORM_SHELL_SMOKE */