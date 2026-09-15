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
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_pass_sync_ops.h"
#include "mgl_renderer_host.h"
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
#include "mgl_aux_assets.h"
#include "mgl_shader_resource.h"  /* mglShaderCompileGLSL, mglRenderCreateAuxFunctions */
#include "mgl_air_loader.h"      /* MGLRenderPipelineDescriptorState */
#include "mgl_pipeline_cache_path.h"  /* archive path (log 203) */
#include "mgl_platform_shell_internal.h"  /* pending-size apply (log 206) */
#include "mgl_renderer_backend.h"
#include "mgl_batch_mtl_encode.h"  /* mgl_batch_mtl_create_icb */
#import <objc/message.h>          /* objc_msgSend (runtime-created classes) */

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
/* === renderer port shim (moved out, log 206) =============================
 * The Objective-C surface C talks to - the indirect-command-buffer entry, the
 * rasterizer-discard stub factory, the layer metrics/drawable ports, the GPU
 * capture pair, the context/temporaries helpers and the swap-path queries -
 * is now C++ in MGL/src/mgl_platform_shell.cpp.  What is left below is the
 * part that still needs Objective-C: the two shell classes and the ports that
 * send them messages. */

/* The former mglRendererBindMTLTexturePort is gone: the body is the C function
 * mglRendererBindMTLTexture (mgl_texture_bind.h), and so is
 * mglRendererMapBuffersToMTLPort: mglRendererMapBuffersToMTL (mgl_buffer_map.h)
 * replaced it. */


/* === renderer port shim (moved out, log 206 / log 207) ==================
 * The Objective-C surface C talks to is C++ in MGL/src/mgl_platform_shell.cpp
 * now: the ports themselves, the pipeline-cache bridges, the exception-guarded
 * call frames, the state-area assembly, the compute dispatch entries - and the
 * two category members that were only a lock frame (`flushDrawBuffer:`,
 * `bindMTLTexture:`), which are registered on the class at load time there
 * instead of being compiled here.
 *
 * What is left in this file is the platform shell itself: the two classes and
 * the lifecycle block below. */

/* === renderer lifecycle (moved out, log 208) =============================
 * Construction, the view/window observers (KVO + NSWindow notifications) and
 * teardown are C++ in MGL/src/mgl_platform_shell.cpp now: the methods are
 * registered on MGLRenderer at load time from there, and the C entry points
 * (CppCreateMGLRenderer*, mglRendererPlatformBackendWillDestroy) are defined
 * there too.  The window notification names are the framework's exported
 * constants, linked rather than copied.
 *
 * What is left in this file is the shell class and the (still empty) MGLRenderer
 * implementation that carries the class extension's @package ivars. */

/* === The MGLRenderer class itself (log 202) ================================
 * The class implementation had to come with the class extension: the @package
 * ivars are declared in the MGLRenderer () extension inside
 * MGLRenderer_Private.h, and an extension's ivars only materialise in the TU
 * that also holds the @implementation.  Every method has already moved to C or
 * into this shell, so the implementation is deliberately empty -- deleting
 * MGLRenderer.m without this block left the class undefined and the linker
 * reported "_OBJC_CLASS_$_MGLRenderer referenced from MGLPlatformRendererShell". */
@implementation MGLRenderer
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


/* === MGLPipelineCache: the class moved to mgl_pipeline_cache_class.cpp =====
 * It is registered with the Objective-C runtime at load time now (no .m), so the
 * extension declaration and the build schema constants live there too.  Callers
 * keep sending it messages, which a runtime-registered class supports. */

#endif /* MGL_PLATFORM_SHELL_SMOKE */