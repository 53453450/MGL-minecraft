/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_platform_shell.cpp - the platform shell, rewritten as C++ so that MGL/
 * holds no .m file at all (P0-1, T5 option (a); log 206).
 *
 * The shell's C ports and, as the conversion advances, the two classes behind
 * them move here from MGLPlatformRendererShell.m.  Message sends become
 * objc_msgSend through mglSend<> (mgl_objc_bridge.h), ivars are reached through
 * the mirror structs in mgl_renderer_ivars.h, and the exception boundary uses
 * the preprocessor hook in mgl_objc_exception_bridge.cpp.
 *
 * The port half stays behind MGL_PLATFORM_SHELL_SMOKE, exactly as it did in the
 * .m: the smoke gate compiles this file standalone, and those ports need the
 * batch/replay half of the library.
 */

#include "mgl_objc_bridge.h"
#include "mgl_batch_public.h"
#include "mgl_platform_shell_result.h"   /* MGLPlatformRendererShellResult */
#include "mgl_renderer_ivars.h"

#include <CoreGraphics/CoreGraphics.h>
#include <dispatch/dispatch.h>

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "draw_command.h"             /* mglInitBatchArena / mglDestroyBatchArena */
#include "glm_limits.h"               /* MAX_COLOR_ATTACHMENTS */
#include "mgl_binding_state_ops.h"    /* mglBindingInvalidateLastBoundState */
#include "mgl.h"                      /* mglDrawBuffer */
#include "mgl_capability.h"           /* MGLCapabilityInit */
#include "mgl_render_pass_manager_ops.h" /* mglRendererEndRenderEncodingLocked */
#include "mgl_air_loader.h"           /* MGLRenderPipelineDescriptorState */
#include "mgl_aux_assets.h"
#include "mgl_compute_dispatch.h"     /* mglComputeMtlDispatch*Locked */
#include "mgl_render.h"               /* MGLRenderPipelineBlendState */
#include "mgl_render_pass_manager.h"  /* r->_renderPassManager->state */
#include "mgl_pipeline_cache_state.h" /* MGLPipelineCacheState */
#include "mgl_renderer_backend.h"
#include "mgl_platform_shell_internal.h"
#include "mgl_renderer_host.h"        /* mglRendererEnsureNewCommandBuffer */
#include "mgl_renderer_ports.h"
#include "mgl_shader_abi.h"
#include "mgl_shader_resource.h"
#include "mgl_texture_bind.h"          /* mglRendererBindMTLTexture */
#include "mgl_thread_affinity.h"
#include "mgl_context_host_ops.h"      /* GLFW-facing C ABI (H2 ops table) */


/* ==========================================================================
 * MGLPlatformRendererShell, registered with the Objective-C runtime.
 *
 * This is the last piece of the platform layer: the class used to be the .m
 * this file replaces.  Both it and MGLRenderer are created with
 * objc_allocateClassPair at load time (log 210), which is why
 * MGL/src/MGLPlatformRendererShell.m could be deleted - and why consumers look
 * the classes up by name instead of referencing a class symbol (log 209).
 *
 * This block is deliberately outside MGL_PLATFORM_SHELL_SMOKE: the smoke gate
 * builds it standalone to check that the shell keeps building and behaving.
 * ========================================================================== */

/* Metal's device factory and CoreAnimation's filter constant are C symbols;
 * their Objective-C headers cannot be included from C++. */
extern "C" MGLObjectId MTLCreateSystemDefaultDevice(void);
extern "C" MGLObjectId const kCAFilterNearest;

/* MTLCaptureDestinationGPUTraceDocument (Metal) and
 * NSWindowOcclusionStateVisible (AppKit). */
enum { kMGLMTLCaptureDestinationGPUTraceDocument = 2 };
enum { kMGLNSWindowOcclusionStateVisible = 2u };

/* The registered classes: a C implementation of a method cannot recover the
 * class it was installed on from the receiver (object_getClass returns the most
 * derived class), and [super ...] needs exactly that - measured as an infinite
 * self-call in dealloc before these were kept (log 210). */
static Class s_mglShellClass = Nil;
static Class s_mglRendererClass = Nil;


static SEL s_shellSelInit = NULL;
static SEL s_shellSelView = NULL;
static SEL s_shellSelSetView = NULL;
static SEL s_shellSelLayer = NULL;
static SEL s_shellSelSetLayer = NULL;
static SEL s_shellSelDrawable = NULL;
static SEL s_shellSelSetDrawable = NULL;
static SEL s_shellSelWindow = NULL;
static SEL s_shellSelIsVisible = NULL;
static SEL s_shellSelOcclusionState = NULL;
static SEL s_shellSelIsMainThread = NULL;
static SEL s_shellSelSetDevice = NULL;
static SEL s_shellSelSetPixelFormat = NULL;
static SEL s_shellSelSetOpaque = NULL;
static SEL s_shellSelSetFramebufferOnly = NULL;
static SEL s_shellSelSetAllowsTimeout = NULL;
static SEL s_shellSelSetMagnification = NULL;
static SEL s_shellSelSetPresentsWithTransaction = NULL;
static SEL s_shellSelSetDisplaySync = NULL;
static SEL s_shellSelDrawableSize = NULL;
static SEL s_shellSelSetDrawableSize = NULL;
static SEL s_shellSelSetContentsScale = NULL;
static SEL s_shellSelFrameRect = NULL;
static SEL s_shellSelSetFrame = NULL;
static SEL s_shellSelRemoveFromSuperlayer = NULL;
static SEL s_shellSelAddSublayer = NULL;
static SEL s_shellSelNextDrawable = NULL;
static SEL s_shellSelTexture = NULL;
static SEL s_shellSelLength = NULL;
static SEL s_shellSelFileURLWithPath = NULL;
static SEL s_shellSelSetDestination = NULL;
static SEL s_shellSelSetOutputURL = NULL;
static SEL s_shellSelSetCaptureObject = NULL;
static SEL s_shellSelSharedCaptureManager = NULL;
static SEL s_shellSelStartCapture = NULL;
static SEL s_shellSelStopCapture = NULL;
static SEL s_shellSelAlloc = NULL;

/* ARC's strong property store: retain the new value, release the old one. */
static void mglShellStoreObject(void **slot, MGLObjectId value)
{
    if (value) {
        (void)mglSend<MGLObjectId>(value, sel_registerName("retain"));
    }
    void *previous = *slot;
    *slot = (void *)value;
    if (previous) {
        mglReleaseObject((MGLObjectId)previous);
    }
}

static MGLObjectId mglShellViewOf(MGLObjectId self)
{
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    return ivars ? (MGLObjectId)ivars->_view : NULL;
}

static MGLObjectId mglShellLayerOf(MGLObjectId self)
{
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    return ivars ? (MGLObjectId)ivars->_layer : NULL;
}

static MGLObjectId mglShellDrawableOf(MGLObjectId self)
{
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    return ivars ? (MGLObjectId)ivars->_drawable : NULL;
}

/* - (instancetype)initWithView:(NSView *)view */
static MGLObjectId mglShellInitWithView(MGLObjectId self, SEL cmd, MGLObjectId view)
{
    (void)cmd;
    struct objc_super super = {
        self, s_mglShellClass ? class_getSuperclass(s_mglShellClass)
                              : class_getSuperclass(object_getClass(self))
    };
    self = ((MGLObjectId(*)(struct objc_super *, SEL))objc_msgSendSuper)(
        &super, MGL_SEL(s_shellSelInit, "init"));
    if (!self) {
        return NULL;
    }
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    if (ivars) {
        mglShellStoreObject(&ivars->_view, view);
        /* Match CAMetalLayer default (display sync on) until glfwSwapInterval. */
        ivars->_swapInterval = 1;
    }
    return self;
}

static MGLObjectId mglShellGetView(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    return mglShellViewOf(self);
}

static void mglShellSetView(MGLObjectId self, SEL cmd, MGLObjectId value)
{
    (void)cmd;
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    if (ivars) {
        mglShellStoreObject(&ivars->_view, value);
    }
}

static MGLObjectId mglShellGetLayer(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    return mglShellLayerOf(self);
}

static void mglShellSetLayer(MGLObjectId self, SEL cmd, MGLObjectId value)
{
    (void)cmd;
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    if (ivars) {
        mglShellStoreObject(&ivars->_layer, value);
    }
}

static MGLObjectId mglShellGetDrawable(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    return mglShellDrawableOf(self);
}

static void mglShellSetDrawable(MGLObjectId self, SEL cmd, MGLObjectId value)
{
    (void)cmd;
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    if (ivars) {
        mglShellStoreObject(&ivars->_drawable, value);
    }
}

static int mglShellSwapInterval(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    return ivars ? ivars->_swapInterval : 0;
}

static signed char mglShellShouldSkipPresent(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
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
        return 0;
    }
    if (envMode == 1) {
        return 1;
    }
    MGLObjectId view = mglShellViewOf(self);
    MGLObjectId window =
        view ? mglSend<MGLObjectId>(view, MGL_SEL(s_shellSelWindow, "window")) : NULL;
    if (!window) {
        return 1;
    }
    if (!mglSend<signed char>(window, MGL_SEL(s_shellSelIsVisible, "isVisible"))) {
        return 1;
    }
    if ((mglSend<unsigned long>(window, MGL_SEL(s_shellSelOcclusionState,
                                                "occlusionState")) &
         (unsigned long)kMGLNSWindowOcclusionStateVisible) == 0) {
        return 1;
    }
    return 0;
}

static MGLObjectId mglShellCreateSystemDefaultDevice(MGLObjectId self, SEL cmd)
{
    (void)self;
    (void)cmd;
    /* MTLCreateSystemDefaultDevice returns +1; the method returns it the way
     * ARC did, autoreleased. */
    return mglAutoreleaseObject(MTLCreateSystemDefaultDevice());
}

static signed char mglShellConfigureMetalLayer(MGLObjectId self, SEL cmd,
                                              MGLObjectId device,
                                              uint32_t requestedPixelFormat,
                                              uint32_t *actualPixelFormat)
{
    (void)cmd;
    if (!device) return 0;

    const uint32_t fallbackPixelFormat = 80u;
    uint32_t pixelFormat = mglRenderMetalLayerPixelFormatIsSupported(
                               requestedPixelFormat)
                               ? requestedPixelFormat
                               : fallbackPixelFormat;
    MGLObjectId layerClass = (MGLObjectId)objc_getClass("CAMetalLayer");
    MGLObjectId layer =
        layerClass ? mglSend<MGLObjectId>(mglSend<MGLObjectId>(
                                              layerClass,
                                              MGL_SEL(s_shellSelAlloc, "alloc")),
                                          MGL_SEL(s_shellSelInit, "init"))
                   : NULL;
    if (!layer) return 0;

    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetDevice, "setDevice:"), device);
    try {
        (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetPixelFormat, "setPixelFormat:"),
                            pixelFormat);
    } catch (...) {
        fprintf(stderr,
                "MGL CAMetalLayer invalid pixelFormat=%u requested=%u exception=%s; falling back to BGRA8Unorm\n",
                pixelFormat, requestedPixelFormat,
                mglCaughtExceptionDescription(mglTakeCaughtException()));
        pixelFormat = fallbackPixelFormat;
        (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetPixelFormat, "setPixelFormat:"),
                            pixelFormat);
    }
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetOpaque, "setOpaque:"), (signed char)1);
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetFramebufferOnly,
                                       "setFramebufferOnly:"),
                        (signed char)0);
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetAllowsTimeout,
                                       "setAllowsNextDrawableTimeout:"),
                        (signed char)1);
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetMagnification,
                                       "setMagnificationFilter:"),
                        kCAFilterNearest);
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetPresentsWithTransaction,
                                       "setPresentsWithTransaction:"),
                        (signed char)0);
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetDisplaySync,
                                       "setDisplaySyncEnabled:"),
                        (signed char)((ivars && ivars->_swapInterval > 0) ? 1 : 0));
    mglShellSetLayer(self, MGL_SEL(s_shellSelSetLayer, "setLayer:"), layer);

    MGLObjectId view = mglShellViewOf(self);
    MGLObjectId viewLayer =
        view ? mglSend<MGLObjectId>(view, MGL_SEL(s_shellSelLayer, "layer")) : NULL;
    if (viewLayer) {
        (void)mglSend<void>(viewLayer, MGL_SEL(s_shellSelAddSublayer, "addSublayer:"),
                            layer);
    } else if (view) {
        (void)mglSend<void>(view, MGL_SEL(s_shellSelSetLayer, "setLayer:"), layer);
    }
    /* The class owns the layer through its property; ARC's local +1 is gone. */
    mglReleaseObject(layer);
    if (actualPixelFormat) *actualPixelFormat = pixelFormat;
    return 1;
}

static void mglShellDetachMetalLayer(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    mglShellSetDrawable(self, MGL_SEL(s_shellSelSetDrawable, "setDrawable:"),
                        (MGLObjectId)NULL);
    MGLObjectId layer = mglShellLayerOf(self);
    if (layer) {
        (void)mglSend<void>(layer, MGL_SEL(s_shellSelRemoveFromSuperlayer,
                                           "removeFromSuperlayer"));
    }
    mglShellSetLayer(self, MGL_SEL(s_shellSelSetLayer, "setLayer:"),
                     (MGLObjectId)NULL);
}

static MGLObjectId mglShellCaptureDescriptor(MGLObjectId self, SEL cmd,
                                            MGLObjectId device,
                                            MGLObjectId outputPath)
{
    (void)self;
    (void)cmd;
    const unsigned long length =
        outputPath ? mglSend<unsigned long>(outputPath, MGL_SEL(s_shellSelLength, "length"))
                   : 0ul;
    if (!device || length == 0) return NULL;
    MGLObjectId descriptorClass = (MGLObjectId)objc_getClass("MTLCaptureDescriptor");
    MGLObjectId descriptor =
        descriptorClass
            ? mglSend<MGLObjectId>(mglSend<MGLObjectId>(
                                       descriptorClass,
                                       MGL_SEL(s_shellSelAlloc, "alloc")),
                                   MGL_SEL(s_shellSelInit, "init"))
            : NULL;
    if (!descriptor) return NULL;
    (void)mglSend<void>(descriptor, MGL_SEL(s_shellSelSetDestination, "setDestination:"),
                        (unsigned long)kMGLMTLCaptureDestinationGPUTraceDocument);
    MGLObjectId urlClass = (MGLObjectId)objc_getClass("NSURL");
    MGLObjectId url =
        urlClass ? mglSend<MGLObjectId>(urlClass,
                                        MGL_SEL(s_shellSelFileURLWithPath,
                                                "fileURLWithPath:"),
                                        outputPath)
                 : NULL;
    (void)mglSend<void>(descriptor, MGL_SEL(s_shellSelSetOutputURL, "setOutputURL:"), url);
    (void)mglSend<void>(descriptor, MGL_SEL(s_shellSelSetCaptureObject,
                                            "setCaptureObject:"),
                        device);
    return mglAutoreleaseObject(descriptor);
}

static signed char mglShellStartCapture(MGLObjectId self, SEL cmd, MGLObjectId descriptor,
                                       MGLObjectId *error)
{
    (void)self;
    (void)cmd;
    if (!descriptor) return 0;
    MGLObjectId managerClass = (MGLObjectId)objc_getClass("MTLCaptureManager");
    MGLObjectId manager =
        managerClass ? mglSend<MGLObjectId>(managerClass,
                                            MGL_SEL(s_shellSelSharedCaptureManager,
                                                    "sharedCaptureManager"))
                     : NULL;
    if (!manager) return 0;
    return mglSend<signed char>(manager, MGL_SEL(s_shellSelStartCapture,
                                                 "startCaptureWithDescriptor:error:"),
                                descriptor, error);
}

static void mglShellStopCapture(MGLObjectId self, SEL cmd)
{
    (void)self;
    (void)cmd;
    MGLObjectId managerClass = (MGLObjectId)objc_getClass("MTLCaptureManager");
    MGLObjectId manager =
        managerClass ? mglSend<MGLObjectId>(managerClass,
                                            MGL_SEL(s_shellSelSharedCaptureManager,
                                                    "sharedCaptureManager"))
                     : NULL;
    if (manager) {
        (void)mglSend<void>(manager, MGL_SEL(s_shellSelStopCapture, "stopCapture"));
    }
}

static MGLObjectId mglShellNextDrawable(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLObjectId layer = mglShellLayerOf(self);
    MGLObjectId drawable =
        layer ? mglSend<MGLObjectId>(layer, MGL_SEL(s_shellSelNextDrawable,
                                                    "nextDrawable"))
              : NULL;
    mglShellSetDrawable(self, MGL_SEL(s_shellSelSetDrawable, "setDrawable:"), drawable);
    return mglShellDrawableOf(self);
}

static MGLObjectId mglShellDrawableTexture(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLObjectId drawable = mglShellDrawableOf(self);
    /* Borrowed from the drawable: it must NOT be autoreleased here, because the
     * drawable owns the only reference (rule 68). */
    return drawable ? mglSend<MGLObjectId>(drawable, MGL_SEL(s_shellSelTexture, "texture"))
                    : NULL;
}

static signed char mglShellHasMetalLayer(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    return mglShellLayerOf(self) ? 1 : 0;
}

static CGSize mglShellLayerDrawableSize(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLObjectId layer = mglShellLayerOf(self);
    return layer ? mglSend<CGSize>(layer, MGL_SEL(s_shellSelDrawableSize,
                                                  "drawableSize"))
                 : CGSizeMake(0.0, 0.0);
}

static CGRect mglShellLayerFrame(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLObjectId layer = mglShellLayerOf(self);
    return layer ? mglSend<CGRect>(layer, MGL_SEL(s_shellSelFrameRect, "frame"))
                 : CGRectMake(0.0, 0.0, 0.0, 0.0);
}

static void mglShellSetLayerFrame(MGLObjectId self, SEL cmd, CGRect frame,
                                  double scale)
{
    (void)cmd;
    MGLObjectId layer = mglShellLayerOf(self);
    if (!layer) return;
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetFrame, "setFrame:"), frame);
    (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetContentsScale,
                                       "setContentsScale:"),
                        scale);
}

static void mglShellSetLayerDrawableSize(MGLObjectId self, SEL cmd, CGSize size)
{
    (void)cmd;
    MGLObjectId layer = mglShellLayerOf(self);
    if (layer) {
        (void)mglSend<void>(layer, MGL_SEL(s_shellSelSetDrawableSize,
                                           "setDrawableSize:"),
                            size);
    }
}

/* - (int)performOperation:(MGLPlatformRendererShellOperation)operation
 *                 context:(void *)context
 *                  result:(MGLPlatformRendererShellResult *)result */
static int mglShellPerformOperation(MGLObjectId self, SEL cmd,
                                    MGLPlatformRendererShellOperation operation,
                                    void *context,
                                    MGLPlatformRendererShellResult *result)
{
    (void)self;
    (void)cmd;
    if (result) memset(result, 0, sizeof(*result));
    if (!operation) return -1;
    try {
        int status = operation(context);
        if (result) result->status = status;
        return status;
    } catch (...) {
        MGLObjectId exception = mglTakeCaughtException();
        if (result) {
            result->status = -1;
            mglFillCaughtException(exception, result->exception_name,
                                   sizeof(result->exception_name),
                                   result->exception_reason,
                                   sizeof(result->exception_reason));
        }
        return -1;
    }
}

/* The C bridge the renderer diagnostics use. */
void *mglPlatformRendererShellTextureForDrawable(void *drawable)
{
    if (!drawable) return NULL;
    MGLObjectId texture = mglSend<MGLObjectId>((MGLObjectId)drawable,
                                               MGL_SEL(s_shellSelTexture, "texture"));
    return (void *)texture;
}

/* The class and its ivars, in the order the @implementation used to have them:
 * _swapInterval is the declared ivar, the other three come from the properties.
 * class_addIvar takes size and alignment explicitly, which is what lets the
 * @package records below mirror the old compiler layout exactly. */
struct MGLIvarSpec {
    const char *name;
    size_t size;
    uint8_t alignment;   /* log2 of the alignment: class_addIvar's convention */
};

/* class_addIvar takes the alignment as a power of two, not the alignment
 * itself: passing alignof(T) made every ivar 256-byte aligned and broke the
 * mirror-struct view of the @package block (measured, log 210). */
static constexpr uint8_t mglIvarAlignLog2(size_t alignment)
{
    uint8_t shift = 0;
    while (((size_t)1 << shift) < alignment) {
        shift++;
    }
    return shift;
}

static const MGLIvarSpec kMGLShellIvars[] = {
    { "_swapInterval", sizeof(int), mglIvarAlignLog2(alignof(int)) },
    { "_view", sizeof(void *), mglIvarAlignLog2(alignof(void *)) },
    { "_layer", sizeof(void *), mglIvarAlignLog2(alignof(void *)) },
    { "_drawable", sizeof(void *), mglIvarAlignLog2(alignof(void *)) },
};

/* The sixteen @package ivars of `@interface MGLRenderer ()` in
 * MGLRenderer_Private.h, in order, with the exact sizes and alignments the
 * mirror struct MGLRendererIvars is compiled with. */
static const MGLIvarSpec kMGLRendererIvars[] = {
    { "ctx", sizeof(((MGLRendererIvars *)0)->ctx),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->ctx))) },
    { "_backend", sizeof(((MGLRendererIvars *)0)->_backend),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_backend))) },
    { "_observedWindow", sizeof(((MGLRendererIvars *)0)->_observedWindow),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_observedWindow))) },
    { "_core", sizeof(((MGLRendererIvars *)0)->_core),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_core))) },
    { "_gpuRecovery", sizeof(((MGLRendererIvars *)0)->_gpuRecovery),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_gpuRecovery))) },
    { "_pipelineCache", sizeof(((MGLRendererIvars *)0)->_pipelineCache),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_pipelineCache))) },
    { "_queryStateOwner", sizeof(((MGLRendererIvars *)0)->_queryStateOwner),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_queryStateOwner))) },
    { "_renderPassManager", sizeof(((MGLRendererIvars *)0)->_renderPassManager),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_renderPassManager))) },
    { "_resourceFallback", sizeof(((MGLRendererIvars *)0)->_resourceFallback),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_resourceFallback))) },
    { "_bindingStateOwner", sizeof(((MGLRendererIvars *)0)->_bindingStateOwner),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_bindingStateOwner))) },
    { "_tessellation", sizeof(((MGLRendererIvars *)0)->_tessellation),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_tessellation))) },
    { "_geometry", sizeof(((MGLRendererIvars *)0)->_geometry),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_geometry))) },
    { "_mglForcedMSSampleId", sizeof(((MGLRendererIvars *)0)->_mglForcedMSSampleId),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_mglForcedMSSampleId))) },
    { "_mglMSSamplePlaneOffset",
      sizeof(((MGLRendererIvars *)0)->_mglMSSamplePlaneOffset),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_mglMSSamplePlaneOffset))) },
    { "_mglInMSSampleDrawLoop",
      sizeof(((MGLRendererIvars *)0)->_mglInMSSampleDrawLoop),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_mglInMSSampleDrawLoop))) },
    { "_batching", sizeof(((MGLRendererIvars *)0)->_batching),
      mglIvarAlignLog2(alignof(decltype(((MGLRendererIvars *)0)->_batching))) },
};


/* - (void)dealloc: NSObject's dealloc does not release this class's strong
 * ivars (a runtime-registered class has no ARC destructor), so the shell does it
 * and then hands the object to the superclass, the way ARC's dealloc did. */
static void mglPlatformShellDealloc(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars(self);
    if (ivars) {
        mglShellStoreObject(&ivars->_drawable, NULL);
        mglShellStoreObject(&ivars->_layer, NULL);
        mglShellStoreObject(&ivars->_view, NULL);
    }
    /* Search starts at NSObject: s_mglShellClass is the class this method is
     * installed on, so its superclass is where the shell's dealloc continues. */
    struct objc_super super = {
        self, s_mglShellClass ? class_getSuperclass(s_mglShellClass)
                              : class_getSuperclass(object_getClass(self))
    };
    ((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(
        &super, sel_registerName("dealloc"));
}

/* Rule 64: nothing calls this - the constructor attribute is the only entry
 * point.  Priority 101 runs it before the other constructors in this file (the
 * category and lifecycle method installers, which need the class to exist) and
 * before anything outside the library can look the classes up. */
__attribute__((constructor(101)))
static void mglInstallPlatformShellClasses(void)
{
    Class shellClass = objc_allocateClassPair(objc_getClass("NSObject"),
                                              "MGLPlatformRendererShell", 0);
    if (!shellClass) {
        fprintf(stderr, "MGL ERROR: could not register MGLPlatformRendererShell\n");
        return;
    }
    for (size_t i = 0; i < sizeof(kMGLShellIvars) / sizeof(kMGLShellIvars[0]); i++) {
        class_addIvar(shellClass, kMGLShellIvars[i].name, kMGLShellIvars[i].size,
                      kMGLShellIvars[i].alignment, "?");
    }
    class_addMethod(shellClass, sel_registerName("initWithView:"),
                    (IMP)mglShellInitWithView, "@@:@");
    class_addMethod(shellClass, sel_registerName("view"), (IMP)mglShellGetView, "@@:");
    class_addMethod(shellClass, sel_registerName("setView:"), (IMP)mglShellSetView,
                    "v@:@");
    class_addMethod(shellClass, sel_registerName("layer"), (IMP)mglShellGetLayer, "@@:");
    class_addMethod(shellClass, sel_registerName("setLayer:"), (IMP)mglShellSetLayer,
                    "v@:@");
    class_addMethod(shellClass, sel_registerName("drawable"), (IMP)mglShellGetDrawable,
                    "@@:");
    class_addMethod(shellClass, sel_registerName("setDrawable:"),
                    (IMP)mglShellSetDrawable, "v@:@");
    class_addMethod(shellClass, sel_registerName("mglSwapInterval"),
                    (IMP)mglShellSwapInterval, "i@:");
    class_addMethod(shellClass, sel_registerName("mglShouldSkipPresentForUnlockedSwap"),
                    (IMP)mglShellShouldSkipPresent, "c@:");
    class_addMethod(shellClass, sel_registerName("mglCreateSystemDefaultDevice"),
                    (IMP)mglShellCreateSystemDefaultDevice, "@@:");
    class_addMethod(shellClass,
                    sel_registerName("mglConfigureMetalLayerWithDevice:"
                                     "requestedPixelFormat:actualPixelFormat:"),
                    (IMP)mglShellConfigureMetalLayer, "c@:^vI^I");
    class_addMethod(shellClass, sel_registerName("mglDetachMetalLayer"),
                    (IMP)mglShellDetachMetalLayer, "v@:");
    class_addMethod(shellClass,
                    sel_registerName("mglCaptureDescriptorForDevice:outputPath:"),
                    (IMP)mglShellCaptureDescriptor, "@@:@@");
    class_addMethod(shellClass,
                    sel_registerName("mglStartCaptureWithDescriptor:error:"),
                    (IMP)mglShellStartCapture, "c@:@^@");
    class_addMethod(shellClass, sel_registerName("mglStopCapture"),
                    (IMP)mglShellStopCapture, "v@:");
    class_addMethod(shellClass, sel_registerName("mglNextDrawable"),
                    (IMP)mglShellNextDrawable, "@@:");
    class_addMethod(shellClass, sel_registerName("mglDrawableTexture"),
                    (IMP)mglShellDrawableTexture, "@@:");
    class_addMethod(shellClass, sel_registerName("mglHasMetalLayer"),
                    (IMP)mglShellHasMetalLayer, "c@:");
    class_addMethod(shellClass, sel_registerName("mglMetalLayerDrawableSize"),
                    (IMP)mglShellLayerDrawableSize, "{CGSize=dd}@:");
    class_addMethod(shellClass, sel_registerName("mglMetalLayerFrame"),
                    (IMP)mglShellLayerFrame, "{CGRect={CGPoint=dd}{CGSize=dd}}@:");
    class_addMethod(shellClass,
                    sel_registerName("mglSetMetalLayerFrame:contentsScale:"),
                    (IMP)mglShellSetLayerFrame,
                    "v@:{CGRect={CGPoint=dd}{CGSize=dd}}d");
    class_addMethod(shellClass, sel_registerName("mglSetMetalLayerDrawableSize:"),
                    (IMP)mglShellSetLayerDrawableSize, "v@:{CGSize=dd}");
    class_addMethod(shellClass, sel_registerName("performOperation:context:result:"),
                    (IMP)mglShellPerformOperation, "i@:^?^v^v");
    class_addMethod(shellClass, sel_registerName("dealloc"),
                    (IMP)mglPlatformShellDealloc, "v@:");
    objc_registerClassPair(shellClass);
    s_mglShellClass = shellClass;

    Class rendererClass =
        objc_allocateClassPair(shellClass, "MGLRenderer", 0);
    if (!rendererClass) {
        fprintf(stderr, "MGL ERROR: could not register MGLRenderer\n");
        return;
    }
    for (size_t i = 0; i < sizeof(kMGLRendererIvars) / sizeof(kMGLRendererIvars[0]);
         i++) {
        class_addIvar(rendererClass, kMGLRendererIvars[i].name,
                      kMGLRendererIvars[i].size, kMGLRendererIvars[i].alignment,
                      "?");
    }
    /* The methods themselves are installed by the two constructors below (the
     * category pair and the lifecycle block): they run at default priority, that
     * is, after this one, and find both classes in the runtime. */
    objc_registerClassPair(rendererClass);
    s_mglRendererClass = rendererClass;
}

#ifndef MGL_PLATFORM_SHELL_SMOKE

/* Every function in this block is a C port: the definitions must keep C
 * linkage even when no header declares them, or the C callers would look for
 * the unmangled name and fail to link. */
extern "C" {

/* Cached selectors ([...] -> objc_msgSend needs a SEL, not a literal). */
static SEL s_selHasMetalLayer = NULL;
static SEL s_selMetalLayerDrawableSize = NULL;
static SEL s_selMetalLayerFrame = NULL;
static SEL s_selSetMetalLayerDrawableSize = NULL;
static SEL s_selNextDrawable = NULL;
static SEL s_selDrawableTexture = NULL;
static SEL s_selDrawable = NULL;
static SEL s_selSetDrawable = NULL;
static SEL s_selShouldSkipPresent = NULL;
static SEL s_selCaptureDescriptor = NULL;
static SEL s_selStartCapture = NULL;
static SEL s_selStopCapture = NULL;
static SEL s_selLocalizedDescription = NULL;
static SEL s_selArray = NULL;
static SEL s_selAddObject = NULL;
static SEL s_selFlushDrawBuffer = NULL;
static SEL s_selBindMTLTexture = NULL;
static SEL s_selSwapInterval = NULL;
static SEL s_selSetDisplaySyncEnabled = NULL;
static SEL s_selState = NULL;
static SEL s_selLayer = NULL;
static SEL s_selReason = NULL;
static SEL s_selDescription = NULL;
static SEL s_selInvalidatePipelineState = NULL;
static SEL s_selPipelineDescriptorStateForWords = NULL;
static SEL s_selCreateRenderPipelineFromState = NULL;
static SEL s_selStorePipeline = NULL;
static SEL s_selStorePipelineDescriptorState = NULL;
static SEL s_selDepthStencilStateForValueState = NULL;
static SEL s_selLookupPipelineForWords = NULL;
static SEL s_selActivatePipelineState = NULL;
static SEL s_selResetCaches = NULL;
static SEL s_selBlendStateForAttachment = NULL;
static SEL s_selSetBlendFactorsForAttachment = NULL;

/* METAL_LOCK/METAL_UNLOCK are renderer-private macros in MGLRenderer_Private.h
 * that only assert the GL thread; the C twin is the same (see
 * mgl_draw_metal_port.c). */
#define METAL_LOCK()   do { MGL_ASSERT_GL_THREAD(); } while (0)
#define METAL_UNLOCK() do { } while (0)

/* === renderer port shim ==================================================
 * The Objective-C surface C talks to lives in this TU.  Every function below
 * was an Objective-C method or an Objective-C function in
 * MGLPlatformRendererShell.m; its body is unchanged except for the translation
 * described in mgl_objc_bridge.h. */

void *mglRendererCreateIndirectCommandBuffer(void *renderer, int indexed,
                                                 uint64_t count,
                                                 int *failed_out)
{
    (void)renderer;
    if (failed_out) {
        *failed_out = 0;
    }
    /* The @try/@catch is the reason this one was still ObjC: Metal raises when
     * an indirect command buffer cannot be allocated, and that has to become a
     * NULL result the replay path can fall back from.  A C++ catch(...) catches
     * the NSException just the same (rule 67); the object comes back from the
     * exception bridge so the log line keeps its `%@`. */
    try {
        return mgl_batch_mtl_create_icb(indexed, count);
    } catch (...) {
        static uint64_t s_hit = 0;
        uint64_t hit = ++s_hit;
        if (hit <= 8ull || (hit % 256ull) == 0ull) {
            fprintf(stderr, "MGL WARNING: ICB creation failed, falling back: %s\n",
                    mglCaughtExceptionDescription(mglTakeCaughtException()));
        }
        if (failed_out) {
            *failed_out = 1;
        }
        return NULL;
    }
}

/* === draw / tessellation host entries (phase 2) ========================= */
/* The rasterizer-discard stub fragment function (moved out of
 * MGLRenderer+RenderPass.m so that file could be deleted, log 193).  The
 * dispatch_once + aux-asset/self-hosted-GLSL compile path was Objective-C by
 * construction; dispatch_once and blocks are plain C/C++ here, and the only
 * Objective-C left was the object bookkeeping (rules 67). */
typedef enum MGLStubFSValueClass {
    MGLStubFSFloat = 0,
    MGLStubFSInt,
    MGLStubFSUint,
} MGLStubFSValueClass;

static MGLObjectId mglRasterizerDiscardStubFragmentFunctionForClass(
    MGLStubFSValueClass valueClass)
{
    static MGLObjectId s_fs[MGLStubFSUint + 1] = { NULL, NULL, NULL };
    static dispatch_once_t once[MGLStubFSUint + 1];

    dispatch_once(&once[valueClass], ^{
        void *fs = NULL;
        char err[256] = {0};
        if (valueClass == MGLStubFSFloat) {
            /* Precompiled aux asset (no runtime source compile). */
            const MGLAuxShaderAsset *safe =
                mglAuxShaderAssetFind("safe_fallback");
            void *vs = NULL;
            if (!safe || !safe->data || safe->size == 0 ||
                mglRenderCreateAuxFunctions(
                    safe->data, safe->size, safe->hash,
                    "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
                    &vs, &fs, err, sizeof(err)) != 0 || !fs) {
                fprintf(stderr, "MGL ERROR: discard stub FS unavailable: %s\n",
                        err[0] ? err : "asset missing");
                if (vs) {
                    mglBridgingRelease(vs);
                }
                return;
            }
            mglBridgingRelease(vs);
        } else {
            /* Integer-format targets reject a float4 output, and no
             * precompiled integer stub asset ships in the aux table.
             * Compile the integer zero stub at runtime through the
             * self-hosted GLSL->AIR backend (the same path real programs
             * take; its fragment output carries the correct
             * air.render_target int/uint type). */
            static const char *s_stubSource[MGLStubFSUint + 1] = {
                NULL,
                "#version 330\n"
                "out ivec4 mgl_stub_color_int;\n"
                "void main() { mgl_stub_color_int = ivec4(0); }\n",
                "#version 330\n"
                "out uvec4 mgl_stub_color_uint;\n"
                "void main() { mgl_stub_color_uint = uvec4(0u); }\n",
            };
            unsigned char *bytes = NULL;
            size_t size = 0;
            if (mglShaderCompileGLSL(
                    s_stubSource[valueClass], MGL_STAGE_FRAGMENT,
                    &bytes, &size, err, sizeof(err)) != 0 || !bytes) {
                fprintf(stderr, "MGL ERROR: stub FS compile failed: %s\n",
                        err[0] ? err : "unknown");
                return;
            }
            /* mglRenderCreateAuxFunctions supports fragment-only blobs by
             * accepting a NULL vertex entry, but the vertex output argument
             * itself is still required so the API can publish both results.
             * Passing NULL here made every integer render target fail with
             * "bad args" before the stub function was even looked up. */
            void *unusedVertex = NULL;
            if (mglRenderCreateAuxFunctions(
                    bytes, size, 0u, NULL, "main",
                    &unusedVertex, &fs, err, sizeof(err)) != 0 || !fs) {
                fprintf(stderr, "MGL ERROR: stub FS function load failed: %s\n",
                        err[0] ? err : "unknown");
                if (unusedVertex) {
                    mglBridgingRelease(unusedVertex);
                }
                free(bytes);
                return;
            }
            if (unusedVertex) {
                mglBridgingRelease(unusedVertex);
            }
            free(bytes);
        }
        /* The create call hands back a +1; the cache below is what holds it
         * (the ARC original spelled this `__bridge_transfer id`). */
        s_fs[valueClass] = (MGLObjectId)fs;
    });
    /* Borrowed on purpose.  The create call above handed back one reference and
     * this cache is what owns it, so the result must not go through ARC's
     * return convention: an autorelease here would hand the pool a release of
     * the cache's *only* reference, the object would die at the next drain and
     * the following call would message a deallocated instance.  That is not
     * theory - the first version of this file did exactly that and
     * `air_cull_distance` died in the stub path (NSZombieEnabled:
     * "-[_MTLFunctionInternal autorelease]: message sent to deallocated
     * instance", log 206).  Both C callers only borrow the pointer
     * (mgl_pso_build_ops.c, mgl_render_pass_manager_ops.c). */
    return s_fs[valueClass];
}

/* C-callable bridge for the C pipeline-descriptor host. */
void *mglRenderPassDiscardStubFragmentFunction(uint32_t valueClass)
{
    return (void *)mglRasterizerDiscardStubFragmentFunctionForClass(
        (MGLStubFSValueClass)valueClass);
}

int mglRendererLayerMetrics(void *renderer,
                                MGLRendererLayerMetricsValue *metrics_out)
{
    MGLObjectId r = (MGLObjectId)renderer;
    if (!r) {
        return 0;
    }
    const signed char hasLayer =
        mglSend<signed char>(r, MGL_SEL(s_selHasMetalLayer, "mglHasMetalLayer"));
    if (metrics_out) {
        CGSize drawableSize =
            mglSend<CGSize>(r, MGL_SEL(s_selMetalLayerDrawableSize,
                                       "mglMetalLayerDrawableSize"));
        CGRect frame =
            mglSend<CGRect>(r, MGL_SEL(s_selMetalLayerFrame, "mglMetalLayerFrame"));
        metrics_out->drawable_width = (double)drawableSize.width;
        metrics_out->drawable_height = (double)drawableSize.height;
        metrics_out->frame_width = (double)frame.size.width;
        metrics_out->frame_height = (double)frame.size.height;
    }
    return hasLayer ? 1 : 0;
}

void mglRendererNextDrawable(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    if (r) {
        /* -mglNextDrawable assigns self.drawable itself (which is what
         * `_drawable = [self mglNextDrawable]` did). */
        (void)mglSend<MGLObjectId>(r, MGL_SEL(s_selNextDrawable, "mglNextDrawable"));
    }
}

void *mglRendererDrawableTexture(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    return r ? (void *)mglSend<MGLObjectId>(
                   r, MGL_SEL(s_selDrawableTexture, "mglDrawableTexture"))
             : NULL;
}

/* mglPlatformShellApplyPendingDrawableSizeCGSize lives further down this TU
 * and is shared with the shell's Objective-C half (mgl_platform_shell_internal.h). */

int mglRendererEnsureLayerDrawableSizeAtLeastWidth(void *renderer,
                                                       size_t required_width,
                                                       size_t required_height,
                                                       const char *reason)
{
    /* -mglEnsureLayerDrawableSizeAtLeastWidth:height:reason: moved here for the
     * same reason as the pending-size apply above (log 201). */
    MGLObjectId r = (MGLObjectId)renderer;
    if (!r ||
        !mglSend<signed char>(r, MGL_SEL(s_selHasMetalLayer, "mglHasMetalLayer")) ||
        required_width == 0 || required_height == 0) {
        return 0;
    }

    CGSize viewDrawableSize =
        mglPlatformShellApplyPendingDrawableSizeCGSize((void *)r);
    uint64_t targetWidth = required_width;
    uint64_t viewW = (uint64_t)(viewDrawableSize.width > 1.0 ? viewDrawableSize.width : 1.0);
    if (viewW > targetWidth) targetWidth = viewW;
    uint64_t targetHeight = required_height;
    uint64_t viewH = (uint64_t)(viewDrawableSize.height > 1.0 ? viewDrawableSize.height : 1.0);
    if (viewH > targetHeight) targetHeight = viewH;
    CGSize oldDrawableSize =
        mglSend<CGSize>(r, MGL_SEL(s_selMetalLayerDrawableSize,
                                   "mglMetalLayerDrawableSize"));

    if ((uint64_t)oldDrawableSize.width == targetWidth &&
        (uint64_t)oldDrawableSize.height == targetHeight) {
        return 0;
    }

    (void)mglSend<void>(r, MGL_SEL(s_selSetMetalLayerDrawableSize,
                                   "mglSetMetalLayerDrawableSize:"),
                        CGSizeMake((CGFloat)targetWidth, (CGFloat)targetHeight));
    if (mglSend<MGLObjectId>(r, MGL_SEL(s_selDrawable, "drawable"))) {
        (void)mglSend<void>(r, MGL_SEL(s_selSetDrawable, "setDrawable:"),
                            (MGLObjectId)NULL);
    }

    static uint64_t s_forcedDrawableResizeCount = 0;
    uint64_t hit = ++s_forcedDrawableResizeCount;
    if (hit <= 32ull || (hit % 120ull) == 0ull) {
        fprintf(stderr,
                "MGL SIZE force drawable reason=%s hit=%llu required=%lux%lu viewSync=%.0fx%.0f old=%.0fx%.0f new=%lux%lu\n",
                reason ? reason : "unknown", (unsigned long long)hit,
                (unsigned long)required_width, (unsigned long)required_height,
                viewDrawableSize.width, viewDrawableSize.height,
                oldDrawableSize.width, oldDrawableSize.height,
                (unsigned long)targetWidth, (unsigned long)targetHeight);
    }

    return 1;
}

/* GPU capture: the capture session lives on the shell object, which owns the
 * MTLCaptureManager descriptor/start/stop calls. */
void mglPlatformShellGpuCaptureStart(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    /* MGLRenderer carries the capture methods (they come from the platform
     * shell class), and the renderer owns the backend ivar holding the device. */
    if (!r || !getenv("MGL_GPU_CAPTURE")) {
        return;
    }
    MGLRendererIvars *ivars = mglRendererIvars(r);
    MGLObjectId desc =
        mglSend<MGLObjectId>(r, MGL_SEL(s_selCaptureDescriptor,
                                        "mglCaptureDescriptorForDevice:outputPath:"),
                             (MGLObjectId)mglRendererBackendGetDevice(
                                 ivars ? ivars->_backend : NULL),
                             mglNewUTF8String(getenv("MGL_GPU_CAPTURE")));
    MGLObjectId capErr = NULL;
    if (desc && mglSend<signed char>(r, MGL_SEL(s_selStartCapture,
                                                "mglStartCaptureWithDescriptor:error:"),
                                     desc, &capErr)) {
        fprintf(stderr, "MGL GPU capture started -> %s\n", getenv("MGL_GPU_CAPTURE"));
    } else {
        fprintf(stderr, "MGL GPU capture start failed: %s\n",
                mglObjectDescriptionUTF8(
                    mglSend<MGLObjectId>(capErr, MGL_SEL(s_selLocalizedDescription,
                                                         "localizedDescription"))));
    }
}

void mglPlatformShellGpuCaptureStop(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    if (r) {
        (void)mglSend<void>(r, MGL_SEL(s_selStopCapture, "mglStopCapture"));
    }
}

/* === compute / tessellation host entries =================================
 * Thin forwards for the stages that are still Objective-C; see the ownership
 * notes next to their declarations in mgl_renderer_ports.h. */
void mglPlatformShellSetContext(void *renderer, GLMContext glm_ctx)
{
    MGLRendererIvars *ivars = mglRendererIvars((MGLObjectId)renderer);
    if (ivars) {
        /* -mglSetActiveContext: was one assignment to this @package ivar; the
         * shell does it directly so the method can go (log 201). */
        ivars->ctx = glm_ctx;
    }
}

void *mglRendererTemporariesCreate(void)
{
    MGLObjectId array =
        mglSend<MGLObjectId>((MGLObjectId)objc_getClass("NSMutableArray"),
                             MGL_SEL(s_selArray, "array"));
    return mglBridgingRetain(array);
}

void mglRendererTemporariesAdd(void *temporaries, void *object)
{
    if (!temporaries || !object) {
        return;
    }
    (void)mglSend<void>((MGLObjectId)temporaries,
                        MGL_SEL(s_selAddObject, "addObject:"), (MGLObjectId)object);
}

void mglRendererTemporariesRelease(void *temporaries)
{
    mglBridgingRelease(temporaries);
}

/* === swap-path queries and effects that must run on the shell (log 180) == */
int mglPlatformShellShouldSkipPresentForUnlockedSwap(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    return r ? (mglSend<signed char>(r, MGL_SEL(s_selShouldSkipPresent,
                                                "mglShouldSkipPresentForUnlockedSwap"))
                   ? 1
                   : 0)
             : 0;
}

/* -mglApplyPendingDrawableSize moved here (log 201): it only touches the
 * core-state atomics and the layer helpers this TU already owns. */
CGSize mglPlatformShellApplyPendingDrawableSizeCGSize(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    MGL_ASSERT_GL_THREAD();
    MGLRendererIvars *ivars = mglRendererIvars(r);
    if (!ivars) {
        return CGSizeMake(0.0, 0.0);
    }
    if (atomic_exchange_explicit(&ivars->_core.drawableSizeDirty, false,
                                 memory_order_acquire)) {
        uint32_t w = atomic_load_explicit(&ivars->_core.pendingDrawableW,
                                          memory_order_relaxed);
        uint32_t h = atomic_load_explicit(&ivars->_core.pendingDrawableH,
                                          memory_order_relaxed);
        CGSize s = CGSizeMake((CGFloat)(w > 1u ? w : 1u),
                              (CGFloat)(h > 1u ? h : 1u));
        (void)mglSend<void>(r, MGL_SEL(s_selSetMetalLayerDrawableSize,
                                       "mglSetMetalLayerDrawableSize:"), s);
        return s;
    }
    return mglSend<CGSize>(r, MGL_SEL(s_selMetalLayerDrawableSize,
                                      "mglMetalLayerDrawableSize"));
}

MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    MGLSizeValue size = {0};
    if (!r) {
        return size;
    }
    CGSize applied = mglPlatformShellApplyPendingDrawableSizeCGSize(renderer);
    size.width = (uint64_t)applied.width;
    size.height = (uint64_t)applied.height;
    return size;
}

void *mglPlatformShellDrawablePointer(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    return r ? (void *)mglSend<MGLObjectId>(r, MGL_SEL(s_selDrawable, "drawable"))
             : NULL;
}

/* The drawable is a property on this class, so C can only clear it here. */
void mglPlatformShellSetDrawable(void *renderer, void *drawable)
{
    MGLObjectId r = (MGLObjectId)renderer;
    if (r) {
        (void)mglSend<void>(r, MGL_SEL(s_selSetDrawable, "setDrawable:"),
                            (MGLObjectId)drawable);
    }
}


/* === batch replay shell (former MGLRenderer (BatchZeroShell)) ============
 * The two category members below are the lock/exception frame around a flush
 * and the lock frame around a texture bind.  Their bodies are plain C now and
 * are registered on the class at load time (class_addMethod), exactly as the
 * compiler's category used to do it, so every existing call site keeps working
 * whether it sends the message or calls the C entry point. */

static void mglShellFlushDrawBuffer(MGLObjectId self, SEL cmd, GLMContext glm_ctx)
{
    (void)cmd;
    METAL_LOCK();
    mglRendererFlushDrawBufferLocked((void *)self, glm_ctx);
    METAL_UNLOCK();
}

static bool mglShellBindMTLTexture(MGLObjectId self, SEL cmd, Texture *tex)
{
    (void)cmd;
    METAL_LOCK();
    const bool result = mglRendererBindMTLTexture((void *)self, tex);
    METAL_UNLOCK();
    return result;
}

/* Rule 64: nothing calls this - the constructor attribute is the only entry
 * point.  The class is compiler-generated while the shell still is ObjC, so
 * these are added to it rather than to a class we registered ourselves. */
__attribute__((constructor))
static void mglInstallShellCategoryMethods(void)
{
    Class rendererClass = objc_getClass("MGLRenderer");
    if (!rendererClass) {
        return;
    }
    /* The type encodings are metadata (dispatch goes through objc_msgSend with
     * an explicit cast); they follow the loose convention already used for the
     * runtime-registered pipeline cache. */
    class_addMethod(rendererClass, sel_registerName("flushDrawBuffer:"),
                    (IMP)mglShellFlushDrawBuffer, "v@:^{GLMContextRec_t}");
    class_addMethod(rendererClass, sel_registerName("bindMTLTexture:"),
                    (IMP)mglShellBindMTLTexture, "c@:^?");
}

/* C entry point: lease the backend, then flush under an autorelease pool with a
 * last-resort exception guard so a throwing draw never escapes into C. */
void mglRendererFlushDrawBuffer(GLMContext glm_ctx)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    MGLObjectId renderer =
        (MGLObjectId)(glm_ctx ? glm_ctx->platform_renderer_shell : NULL);
    if (renderer && glm_ctx) {
        MGLScopedAutoreleasePool pool;
        try {
            (void)mglSend<void>(renderer,
                                MGL_SEL(s_selFlushDrawBuffer, "flushDrawBuffer:"),
                                glm_ctx);
        } catch (...) {
            fprintf(stderr, "MGL ERROR: callback flushDrawBuffer exception: %s\n",
                    mglCaughtExceptionDescription(mglTakeCaughtException()));
        }
    }
    mglRendererBackendEnd(&backend_lease);
}

/* === batch flush / replay-workspace ports =============================== */

int mglPlatformShellMSSampleInLoop(void *renderer)
{
    /* The ivar is @package, so the shell reads it directly instead of keeping a
     * one-line Objective-C method in MGLRenderer.m alive for it (log 201). */
    MGLRendererIvars *ivars = mglRendererIvars((MGLObjectId)renderer);
    return ivars ? (ivars->_mglInMSSampleDrawLoop ? 1 : 0) : 0;
}

/* The plane offset is the second half of the emulated-MS-sample loop state and
 * has no Objective-C getter; C reads the ivar through this forwarder. */
int mglPlatformShellMSSamplePlaneOffset(void *renderer)
{
    MGLRendererIvars *ivars = mglRendererIvars((MGLObjectId)renderer);
    return ivars ? (int)ivars->_mglMSSamplePlaneOffset : 0;
}

void mglPlatformShellSetMSSampleState(void *renderer, int in_loop,
                                      int32_t forced, int32_t offset)
{
    MGLRendererIvars *ivars = mglRendererIvars((MGLObjectId)renderer);
    if (ivars) {
        ivars->_mglInMSSampleDrawLoop = in_loop ? 1 : 0;
        ivars->_mglForcedMSSampleId = forced;
        ivars->_mglMSSamplePlaneOffset = offset;
    }
}

int mglPlatformShellNewCommandBuffer(void *renderer)
{
    return renderer ? mglRendererEnsureNewCommandBuffer(renderer) : 0;
}

void *mglPlatformShellDrawable(void *renderer)
{
    MGLObjectId r = (MGLObjectId)renderer;
    return r ? (void *)mglSend<MGLObjectId>(r, MGL_SEL(s_selDrawable, "drawable"))
             : NULL;
}

void *mglPlatformShellMetalDevice(void *renderer)
{
    MGLRendererIvars *ivars = mglRendererIvars((MGLObjectId)renderer);
    return ivars ? mglRendererBackendGetDevice(ivars->_backend) : NULL;
}

int mglPlatformShellMetalObjectsPresent(void *renderer)
{
    MGLRendererIvars *ivars = mglRendererIvars((MGLObjectId)renderer);
    if (!ivars) return 0;
    return (mglRendererBackendGetDevice(ivars->_backend) &&
            mglRendererBackendGetCommandQueue(ivars->_backend))
               ? 1
               : 0;
}

int mglPlatformShellRecreateCommandQueue(void *renderer)
{
    MGLRendererIvars *ivars = mglRendererIvars((MGLObjectId)renderer);
    if (!ivars || !ivars->_backend) {
        return 0;
    }
    void *commandQueue = NULL;
    (void)mglRendererBackendResetCommandQueue(ivars->_backend, 0u, &commandQueue);
    return mglRendererBackendGetCommandQueue(ivars->_backend) != NULL ? 1 : 0;
}

/* C entry point for the pipeline cache's cache reset (the cache object travels
 * in the state areas; the class is runtime-created, so it is reached as an id -
 * log 205). */
int mglPipelineCacheResetCaches(void *pipeline_cache_object)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache) {
        return 0;
    }
    (void)mglSend<void>(cache, MGL_SEL(s_selResetCaches, "resetCaches"));
    return 1;
}

/* === exception-guarded C bodies =========================================
 * @try/@catch has no C form, but a C++ catch(...) catches the NSException the
 * same way (rule 67); the object comes back from the exception bridge. */

int mglPlatformShellGuardedCallCtx(void *renderer, const char *what,
                                   int (*body)(void *, void *), void *ctx,
                                   void (*finally_fn)(void *, void *))
{
    /* @finally, so it also runs on the return below.  The original's `?: "?"`
     * fallback only fired for a nil exception, which the recorder rules out. */
    auto finally_body = mglScopeExit([&] {
        if (finally_fn) {
            finally_fn(renderer, ctx);
        }
    });
    try {
        return body ? body(renderer, ctx) : 0;
    } catch (...) {
        MGLObjectId exception = mglTakeCaughtException();
        /* `exception.description ? exception.description.UTF8String : "?"` */
        const char *description = exception
            ? mglUTF8String(mglSend<MGLObjectId>(
                  exception, MGL_SEL(s_selDescription, "description")))
            : NULL;
        fprintf(stderr, "MGL ERROR: Exception during %s: %s\n",
                what ? what : "operation", description ? description : "?");
        return 0;
    }
}

int mglPlatformShellGuardedCallCtxReason(void *renderer, const char *what,
                                         int (*body)(void *, void *), void *ctx,
                                         char *reason_out,
                                         size_t reason_capacity)
{
    (void)what;
    (void)renderer;
    (void)ctx;
    if (reason_out && reason_capacity > 0) {
        reason_out[0] = '\0';
    }
    try {
        return body ? body(renderer, ctx) : 0;
    } catch (...) {
        if (reason_out && reason_capacity > 0) {
            MGLObjectId exception = mglTakeCaughtException();
            const char *reason = exception
                ? mglUTF8String(mglSend<MGLObjectId>(
                      exception, MGL_SEL(s_selReason, "reason")))
                : NULL;
            snprintf(reason_out, reason_capacity, "%s",
                     reason ? reason : "(null)");
        }
        return 0;
    }
}

int mglPlatformShellGuardedCall(void *renderer, const char *what,
                                int (*body)(void *))
{
    if (!body) {
        return 0;
    }
    try {
        return body(renderer);
    } catch (...) {
        MGLObjectId exception = mglTakeCaughtException();
        /* `exception.description ? exception.description.UTF8String : "?"` */
        const char *description = exception
            ? mglUTF8String(mglSend<MGLObjectId>(
                  exception, MGL_SEL(s_selDescription, "description")))
            : NULL;
        fprintf(stderr, "MGL ERROR: Exception during %s: %s\n",
                what ? what : "operation", description ? description : "?");
        return 0;
    }
}

/* === pipeline-cache bridges =============================================
 * The cache object is a runtime-created class, so C never names its class
 * symbol (log 205): every bridge takes it as void * and sends by id. */

/* Clears the cache's active pipeline state (log 178). */
void mglPlatformShellPipelineCacheInvalidate(void *pipeline_cache_object)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (cache) {
        (void)mglSend<void>(cache, MGL_SEL(s_selInvalidatePipelineState,
                                           "invalidatePipelineState"));
    }
}

/* Pipeline-cache value-state bridges for the C PSO build path (log 187). */
int mglPlatformShellPipelineCacheDescriptorStateForWords(
    void *pipeline_cache_object, const uint64_t *words,
    MGLRenderPipelineDescriptorState *state_out)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache || !words || !state_out) {
        return 0;
    }
    return mglSend<signed char>(
               cache,
               MGL_SEL(s_selPipelineDescriptorStateForWords,
                       "pipelineDescriptorStateForWords:state:"),
               words, state_out)
               ? 1
               : 0;
}

int mglPlatformShellPipelineCacheCreatePSO(
    void *pipeline_cache_object, const MGLRenderPipelineDescriptorState *state,
    void *vertex_function, void *fragment_function, void **pipeline_out,
    char *error_message, size_t error_capacity)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache || !state || !pipeline_out) {
        return -1;
    }
    return mglSend<int>(cache,
                        MGL_SEL(s_selCreateRenderPipelineFromState,
                                "createRenderPipelineFromState:"
                                "vertexFunction:fragmentFunction:"
                                "pipelineOut:errorMessage:errorCapacity:"),
                        state, vertex_function, fragment_function, pipeline_out,
                        error_message, error_capacity);
}

void mglPlatformShellPipelineCacheStorePipeline(
    void *pipeline_cache_object, void *pipeline, void *vertex_function,
    void *fragment_function, const uint64_t *words)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache || !words) {
        return;
    }
    (void)mglSend<void>(cache,
                        MGL_SEL(s_selStorePipeline,
                                "storePipeline:vertexFunction:"
                                "fragmentFunction:forWords:"),
                        (MGLObjectId)pipeline, (MGLObjectId)vertex_function,
                        (MGLObjectId)fragment_function, words);
}

void mglPlatformShellPipelineCacheStoreDescriptorState(
    void *pipeline_cache_object, const MGLRenderPipelineDescriptorState *state,
    const uint64_t *words)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache || !state || !words) {
        return;
    }
    (void)mglSend<void>(cache,
                        MGL_SEL(s_selStorePipelineDescriptorState,
                                "storePipelineDescriptorState:forWords:"),
                        state, words);
}

void mglPlatformShellPipelineCacheDepthStencilStateForValueState(
    void *pipeline_cache_object,
    const struct MGLRenderDepthStencilDescriptorState_t *state, void **out)
{
    if (out) *out = NULL;
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache || !state) {
        return;
    }
    MGLObjectId dsState =
        mglSend<MGLObjectId>(cache,
                             MGL_SEL(s_selDepthStencilStateForValueState,
                                     "depthStencilStateForValueState:"),
                             state);
    if (out) *out = (void *)dsState;
}

int mglPlatformShellPipelineCacheLookupPipeline(
    void *pipeline_cache_object, const uint64_t *words, void **pipeline_out,
    void **vertex_function_out, void **fragment_function_out)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache) {
        return 0;
    }
    MGLObjectId pipeline = NULL;
    MGLObjectId vertexFunction = NULL;
    MGLObjectId fragmentFunction = NULL;
    const signed char found =
        mglSend<signed char>(cache,
                             MGL_SEL(s_selLookupPipelineForWords,
                                     "lookupPipelineForWords:pipeline:"
                                     "vertexFunction:fragmentFunction:"),
                             words, &pipeline, &vertexFunction,
                             &fragmentFunction);
    if (pipeline_out) *pipeline_out = (void *)pipeline;
    if (vertex_function_out) *vertex_function_out = (void *)vertexFunction;
    if (fragment_function_out) {
        *fragment_function_out = (void *)fragmentFunction;
    }
    return found ? 1 : 0;
}

void mglPlatformShellPipelineCacheActivate(
    void *pipeline_cache_object, void *pipeline, uint32_t color0_format,
    uint32_t depth_format, uint32_t stencil_format, uint32_t program_name,
    void *vertex_function, void *fragment_function)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache) {
        return;
    }
    (void)mglSend<void>(cache,
                        MGL_SEL(s_selActivatePipelineState,
                                "activatePipelineState:color0Format:"
                                "depthFormat:stencilFormat:programName:"
                                "vertexFunction:fragmentFunction:"),
                        (MGLObjectId)pipeline, color0_format, depth_format,
                        stencil_format, program_name,
                        (MGLObjectId)vertex_function,
                        (MGLObjectId)fragment_function);
}

/* Runs a C body inside an autorelease pool (log 186): the render-encoder C
 * entry kept the .m's pool so autoreleased temporaries still drain there. */
int mglPlatformShellAutoreleasePoolCall(void *renderer,
                                        int (*body)(void *))
{
    if (!body) {
        return 0;
    }
    int result = 0;
    {
        MGLScopedAutoreleasePool pool;
        result = body(renderer);
    }
    return result;
}

/* The cache's C++ owner holds the blend record; C reads it through this
 * forwarder, the counterpart of mglPlatformShellPipelineCacheSetBlend. */
int mglPlatformShellPipelineCacheBlendState(void *pipeline_cache_object,
                                           uint32_t index,
                                           MGLRenderPipelineBlendState *blend)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache || !blend || index >= MAX_COLOR_ATTACHMENTS) {
        return 0;
    }
    return mglSend<signed char>(cache,
                                MGL_SEL(s_selBlendStateForAttachment,
                                        "blendStateForAttachment:out:"),
                                (unsigned long)index, blend)
               ? 1
               : 0;
}

int mglPlatformShellPipelineCacheSetBlend(void *pipeline_cache_object,
                                          uint32_t index,
                                          const MGLRenderPipelineBlendState *blend)
{
    MGLObjectId cache = (MGLObjectId)pipeline_cache_object;
    if (!cache || !blend || index >= MAX_COLOR_ATTACHMENTS) {
        return 0;
    }
    (void)mglSend<void>(cache,
                        MGL_SEL(s_selSetBlendFactorsForAttachment,
                                "setBlendFactorsForAttachment:srcRgbFactor:"
                                "srcAlphaFactor:dstRgbFactor:dstAlphaFactor:"
                                "rgbOperation:alphaOperation:colorMask:"),
                        (unsigned long)index, blend->source_rgb_factor,
                        blend->source_alpha_factor,
                        blend->destination_rgb_factor,
                        blend->destination_alpha_factor, blend->rgb_operation,
                        blend->alpha_operation, blend->color_write_mask);
    return 1;
}

/* === renderer state areas =============================================== */

void mglRendererFillStateAreas(void *renderer, MGLRendererStateAreas *areas_out)
{
    MGLObjectId r = (MGLObjectId)renderer;
    MGLRendererIvars *r_ivars;
    if (!areas_out) {
        return;
    }
    memset(areas_out, 0, sizeof(*areas_out));
    r_ivars = mglRendererIvars(r);
    if (!r_ivars) {
        return;
    }
    areas_out->core = &r_ivars->_core;
    areas_out->backend = r_ivars->_backend;
    areas_out->render_pass_manager = r_ivars->_renderPassManager;
    areas_out->ctx = r_ivars->ctx;
    areas_out->batching = &r_ivars->_batching;
    /* The manager exposes a const pointer; the record itself is mutable and
     * the flush driver writes the trace-replay identity through it. */
    areas_out->command =
        ((MGLRenderPassManager *)r_ivars->_renderPassManager)->state;
    areas_out->pipeline_cache =
        (const MGLPipelineCacheState *)mglSend<MGLObjectId>(
            (MGLObjectId)r_ivars->_pipelineCache, MGL_SEL(s_selState, "state"));
    areas_out->binding_state_owner = &r_ivars->_bindingStateOwner;
    areas_out->pipeline_cache_object = r_ivars->_pipelineCache;
    areas_out->gpu_recovery_command_owner =
        &r_ivars->_gpuRecovery.commandRecoveryOwner;
    areas_out->pipeline_cache_set_blend = mglPlatformShellPipelineCacheSetBlend;
    areas_out->pipeline_cache_blend_state =
        mglPlatformShellPipelineCacheBlendState;
    areas_out->pipeline_cache_invalidate =
        mglPlatformShellPipelineCacheInvalidate;
    areas_out->gpu_recovery = &r_ivars->_gpuRecovery;
    areas_out->pipeline_cache_descriptor_state_for_words =
        mglPlatformShellPipelineCacheDescriptorStateForWords;
    areas_out->pipeline_cache_create_pso = mglPlatformShellPipelineCacheCreatePSO;
    areas_out->pipeline_cache_store_pipeline =
        mglPlatformShellPipelineCacheStorePipeline;
    areas_out->pipeline_cache_store_descriptor_state =
        mglPlatformShellPipelineCacheStoreDescriptorState;
    areas_out->pipeline_cache_lookup_pipeline =
        mglPlatformShellPipelineCacheLookupPipeline;
    areas_out->pipeline_cache_depth_stencil_state_for_value_state =
        mglPlatformShellPipelineCacheDepthStencilStateForValueState;
    areas_out->pipeline_cache_activate = mglPlatformShellPipelineCacheActivate;
    areas_out->tess_native_tes_active =
        (int32_t)r_ivars->_tessellation.nativeTESActive;
    areas_out->tessellation = &r_ivars->_tessellation;
    areas_out->geometry = &r_ivars->_geometry;
    areas_out->tess_native_tes_program =
        (void *)r_ivars->_tessellation.nativeTESProgram;
    areas_out->tess_tcs_output_stride =
        (uint32_t)r_ivars->_tessellation.tcsOutputStride;
    areas_out->drawable = (void *)mglSend<MGLObjectId>(
        r, MGL_SEL(s_selDrawable, "drawable"));
    areas_out->query_state_owner = r_ivars->_queryStateOwner;
    areas_out->gpu_interface_mismatch_blocked_program =
        (uint32_t)r_ivars->_gpuRecovery.interfaceMismatchBlockedProgram;
    areas_out->gpu_interface_mismatch_blocked_until =
        (double)r_ivars->_gpuRecovery.interfaceMismatchBlockedUntil;
    areas_out->mssample_forced_id = (int32_t)r_ivars->_mglForcedMSSampleId;
    areas_out->layer =
        (void *)mglSend<MGLObjectId>(r, MGL_SEL(s_selLayer, "layer"));
    areas_out->swap_interval =
        r ? (int32_t)mglSend<int>(r, MGL_SEL(s_selSwapInterval, "mglSwapInterval"))
          : 0;
    areas_out->tess_cull_capture_first_instance =
        (uint32_t)r_ivars->_tessellation.cullDistanceCaptureFirstInstance;
    areas_out->tess_cull_capture_instance_stride =
        (uint32_t)r_ivars->_tessellation.cullDistanceCaptureInstanceStride;
    areas_out->fragment_trace_bindings =
        &r_ivars->_resourceFallback.fragmentTextureTraceBindings[0];
}

/* The @try/@finally frame the C flush driver cannot express: the teardown has
 * to run even when a draw raises - and, because there is no @catch, the
 * exception must then keep going.  An RAII guard gives both halves. */
void mglRendererFlushDrawBufferLocked(void *renderer, GLMContext glm_ctx)
{
    MGLBatchFlushPass pass;
    if (!mglBatchFlushBegin(renderer, glm_ctx, &pass)) {
        return;
    }
    auto teardown = mglScopeExit([&] {
        MGLRendererStateAreas areas;
        mglRendererFillStateAreas(renderer, &areas);
        if (areas.command) {
            areas.command->traceReplayFlushId = 0u;
            areas.command->traceReplayBatchIndex = 0u;
        }
        mglBatchTeardownReplay(renderer, glm_ctx, &pass);
    });
    mglBatchFlushRunBatches(renderer, glm_ctx, &pass);
}

/* === compute dispatch entries (T5 merge from MGLRenderer+Compute.m) ======
 * The orchestration is the C functions of mgl_compute_dispatch.h; what stays
 * here is the lease/lock frame and the renderer lookup. */
void mglRendererDispatchCompute(GLMContext glm_ctx,
                                unsigned int groups_x,
                                unsigned int groups_y,
                                unsigned int groups_z)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    MGLObjectId renderer =
        (MGLObjectId)(glm_ctx ? glm_ctx->platform_renderer_shell : NULL);
    if (renderer && glm_ctx) {
        METAL_LOCK();
        mglComputeMtlDispatchLocked((void *)renderer, glm_ctx,
                                    groups_x, groups_y, groups_z);
        METAL_UNLOCK();
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererDispatchComputeIndirect(GLMContext glm_ctx,
                                        intptr_t indirect)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    MGLObjectId renderer =
        (MGLObjectId)(glm_ctx ? glm_ctx->platform_renderer_shell : NULL);
    if (renderer && glm_ctx) {
        METAL_LOCK();
        mglComputeMtlDispatchIndirectLocked((void *)renderer, glm_ctx, indirect);
        METAL_UNLOCK();
    }
    mglRendererBackendEnd(&backend_lease);
}

/* === texture binding (T5 merge from MGLRenderer+Binding.m) ==============
 * The bind body is the C function mglRendererBindMTLTexture
 * (mgl_texture_bind.h); what is left of the category is the lock frame, which
 * is registered on the class at load time above. */
void mglRendererBindTexture(GLMContext glm_ctx,
                            Texture *texture)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    MGLObjectId renderer =
        (MGLObjectId)(glm_ctx ? glm_ctx->platform_renderer_shell : NULL);
    if (renderer && glm_ctx && texture) {
        (void)mglSend<bool>(renderer,
                            MGL_SEL(s_selBindMTLTexture, "bindMTLTexture:"),
                            texture);
    }
    mglRendererBackendEnd(&backend_lease);
}


/* === renderer lifecycle (T5 merge from MGLRenderer+Lifecycle.m) ==========
 * Construction, the view/window observers and teardown are Cocoa API: KVO on
 * the view, NSWindow notifications, the CALayer/drawable bring-up and the
 * MTLDevice/backend bootstrap.  This was the renderer's last Objective-C: it is
 * C++ now, with AppKit reached as id handles through objc_msgSend, the @package
 * ivars through mglRendererIvars(), and the window notification names taken from
 * the framework's own exported constants - linked, not copied, so the observer
 * names are the very objects NSNotificationCenter publishes. */

/* mgl_renderer_entries.c; only the Objective-C private header declares it, so a
 * C translation unit adds its own declaration (the pattern uniforms.c uses). */
extern "C" int mglEnvFlagEnabledDefaultOn(const char *name);

extern "C" {
/* NSString * const, exported by AppKit (see NSWindow.h). */
extern MGLObjectId const NSWindowDidResizeNotification;
extern MGLObjectId const NSWindowDidChangeBackingPropertiesNotification;
}

/* NSKeyValueObservingOptionInitial */
enum { kMGLKVOOptionInitial = 0x1 };

/* KVO context shared by the observer registration and the callback. */
static void *s_kvoViewGeometryContext = &s_kvoViewGeometryContext;

static SEL s_selAlloc = NULL;
static SEL s_selInit = NULL;
static SEL s_selInitWithFrame = NULL;
static SEL s_selSetWantsLayer = NULL;
static SEL s_selSetContentView = NULL;
static SEL s_selSetLayer = NULL;
static SEL s_selInitCache = NULL;
static SEL s_selView = NULL;
static SEL s_selSetView = NULL;
static SEL s_selBounds = NULL;
static SEL s_selFrameRect = NULL;
static SEL s_selConvertRectToBacking = NULL;
static SEL s_selWindow = NULL;
static SEL s_selBackingScaleFactor = NULL;
static SEL s_selMainScreen = NULL;
static SEL s_selIsMainThread = NULL;
static SEL s_selIsEqualToString = NULL;
static SEL s_selAddObserverForKeyPath = NULL;
static SEL s_selRemoveObserverForKeyPath = NULL;
static SEL s_selDefaultCenter = NULL;
static SEL s_selAddObserverSelectorName = NULL;
static SEL s_selRemoveObserverName = NULL;
static SEL s_selObserveValueForKeyPath = NULL;
static SEL s_selCreateAndBind = NULL;
static SEL s_selMglRendererIsReady = NULL;
static SEL s_selMglMainThreadSync = NULL;
static SEL s_selMglUpdateWindowObserver = NULL;
static SEL s_selMglWindowGeometryChanged = NULL;
static SEL s_selCreateProactiveTextures = NULL;
static SEL s_selMglBackendWillDestroy = NULL;
static SEL s_selMglCreateSystemDefaultDevice = NULL;
static SEL s_selMglConfigureMetalLayer = NULL;
static SEL s_selMglSetMetalLayerFrame = NULL;
static SEL s_selMglDetachMetalLayer = NULL;
static SEL s_selIsBinaryArchiveEnabled = NULL;
static SEL s_selLoadBinaryArchive = NULL;
static SEL s_selDisableBinaryArchive = NULL;
static SEL s_selSaveBinaryArchive = NULL;
static SEL s_selShutdown = NULL;
static SEL s_selSetDevice = NULL;
static SEL s_selOldInitRenderer = NULL;
static SEL s_selOldCreateRenderer = NULL;

static double mglMaxDouble(double a, double b)
{
    return a > b ? a : b;   /* Foundation's MAX() */
}

/* [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)] - +1, so the
 * caller releases once it has handed the view to the window (ARC's local). */
static MGLObjectId mglCreateRendererView(void)
{
    MGLObjectId viewClass = (MGLObjectId)objc_getClass("NSView");
    if (!viewClass) {
        return NULL;
    }
    MGLObjectId view = mglSend<MGLObjectId>(viewClass, MGL_SEL(s_selAlloc, "alloc"));
    if (!view) {
        return NULL;
    }
    return mglSend<MGLObjectId>(
        view, MGL_SEL(s_selInitWithFrame, "initWithFrame:"),
        CGRectMake(100.0, 100.0, 100.0, 100.0));
}

/* [[MGLRenderer alloc] init] - +1. */
static MGLObjectId mglCreateRendererObject(void)
{
    MGLObjectId rendererClass = (MGLObjectId)objc_getClass("MGLRenderer");
    if (!rendererClass) {
        return NULL;
    }
    MGLObjectId renderer =
        mglSend<MGLObjectId>(rendererClass, MGL_SEL(s_selAlloc, "alloc"));
    return renderer
               ? mglSend<MGLObjectId>(renderer, MGL_SEL(s_selInit, "init"))
               : NULL;
}

static const MGLPipelineCacheState *mglRendererCacheState(MGLRendererIvars *ivars)
{
    MGLObjectId cache = ivars ? (MGLObjectId)ivars->_pipelineCache : NULL;
    return cache ? (const MGLPipelineCacheState *)mglSend<MGLObjectId>(
                       cache, MGL_SEL(s_selState, "state"))
                 : NULL;
}

static MGLObjectId mglRendererCommandQueue(MGLRendererIvars *ivars)
{
    return (ivars && ivars->_backend)
               ? (MGLObjectId)mglRendererBackendGetCommandQueue(ivars->_backend)
               : NULL;
}

static MGLObjectId mglRendererDevice(MGLRendererIvars *ivars)
{
    return (ivars && ivars->_backend)
               ? (MGLObjectId)mglRendererBackendGetDevice(ivars->_backend)
               : NULL;
}

static void *mglRendererCommandQueueOwner(MGLRendererIvars *ivars)
{
    return (ivars && ivars->_backend)
               ? mglRendererBackendGetOwner(
                     ivars->_backend, MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE)
               : NULL;
}

/* - (void) createMGLRendererAndBindToContext: (GLMContext) glm_ctx view: (NSView *) view
 */
static void mglShellCreateAndBind(MGLObjectId self, SEL cmd, GLMContext glm_ctx,
                                  MGLObjectId view)
{
    (void)cmd;
    MGLRendererIvars *ivars = mglRendererIvars(self);
    if (!ivars) {
        return;
    }
    mglClaimGLThread();            /* idempotent; records the init thread as the GL thread */
    ivars->ctx = glm_ctx;
    ivars->_backend = NULL;
    (void)mglSend<void>(self, MGL_SEL(s_selSetView, "setView:"), view);
    (void)mglSend<void>(self, MGL_SEL(s_selSetLayer, "setLayer:"),
                        (MGLObjectId)NULL);
    if (!mglSend<MGLObjectId>(self, MGL_SEL(s_selView, "view"))) {
        fprintf(stderr, "MGL ERROR: failed to bind platform renderer view\n");
        return;
    }
    ivars->_renderPassManager = mglPassManagerCreate();
    mglPassManagerSetRuntimeContext((MGLRenderPassManager *)ivars->_renderPassManager,
                                    glm_ctx);

    /* start the DontCare frame generation at 1 so it never matches a
     * texture's zero-initialized mtl_rt_frame_generation stamp until that
     * texture is actually written this frame. */
    mglPassManagerSetDontCareFrameGeneration(
        (MGLRenderPassManager *)ivars->_renderPassManager, 1u);

    const signed char psoDedupEnabled =
        mglEnvFlagEnabledDefaultOn("MGL_PSO_DEDUP") ? 1 : 0;
    const signed char depthStencilCacheEnabled =
        mglEnvFlagEnabledDefaultOn("MGL_DS_CACHE") ? 1 : 0;
    const signed char binaryArchiveEnabled =
        mglEnvFlagEnabledDefaultOn("MGL_BINARY_ARCHIVE") ? 1 : 0;
    /* The cache class is registered by the runtime (log 205), so it is created
     * through objc_getClass + objc_msgSend rather than with a class symbol. */
    {
        MGLObjectId cacheClass = (MGLObjectId)objc_getClass("MGLPipelineCache");
        MGLObjectId cache = NULL;
        if (cacheClass) {
            MGLObjectId allocated =
                mglSend<MGLObjectId>(cacheClass, MGL_SEL(s_selAlloc, "alloc"));
            if (allocated) {
                cache = mglSend<MGLObjectId>(
                    allocated,
                    MGL_SEL(s_selInitCache,
                            "initWithPSODedupEnabled:"
                            "depthStencilCacheEnabled:"
                            "binaryArchiveEnabled:"),
                    psoDedupEnabled, depthStencilCacheEnabled,
                    binaryArchiveEnabled);
            }
        }
        /* Two hops: id -> void* (unretained) -> the ivar's class type.  The
         * class symbol itself is never referenced, which is the point. */
        ivars->_pipelineCache = (void *)cache;
    }
    /* Snapshot arena: batch snapshot/commands from bump allocator. */
    ivars->_batching.arenaSnapshotEnabled =
        mglEnvFlagEnabledDefaultOn("MGL_ARENA_SNAPSHOT") ? 1 : 0;
    if (ivars->_batching.arenaSnapshotEnabled) {
        if (mglInitBatchArena(&ivars->_batching.batchArena, 4u * 1024u * 1024u)) {
            ivars->ctx->batch_arena = &ivars->_batching.batchArena;
            fprintf(stderr,
                    "MGL INFO: Snapshot arena enabled (initial chunk capacity %zu bytes)\n",
                    ivars->_batching.batchArena.initial_capacity);
        } else {
            ivars->_batching.arenaSnapshotEnabled = 0;
            fprintf(stderr, "MGL WARNING: Snapshot arena malloc failed; falling back to per-batch malloc\n");
        }
    }
    ivars->_batching.skipSameKeyRestoreEnabled =
        mglEnvFlagEnabledDefaultOn("MGL_SKIP_SAME_KEY_RESTORE") ? 1 : 0;
    ivars->_batching.dirtyKeyDeltaEnabled =
        mglEnvFlagEnabledDefaultOn("MGL_DIRTY_KEY_DELTA") ? 1 : 0;
    /* Initialize last-bound render encoder dedup state to a clean slate.
     * The C++ binding state's valid bit starts false so the first bind on the first encoder is
     * never incorrectly skipped. */
    mglBindingInvalidateLastBoundState((void *)self);
    fprintf(stderr, "MGL INFO: AGX GPU error tracking initialized\n");
    {
        const MGLPipelineCacheState *cacheState = mglRendererCacheState(ivars);
        fprintf(stderr,
                "MGL INFO: perf gates pso_dedup=%d ds_cache=%d arena=%d "
                "same_key_restore=%d dirty_key_delta=%d (set VAR=0 to disable)\n",
                (cacheState && cacheState->psoDedupEnabled) ? 1 : 0,
                (cacheState && cacheState->dsCacheEnabled) ? 1 : 0,
                ivars->_batching.arenaSnapshotEnabled ? 1 : 0,
                ivars->_batching.skipSameKeyRestoreEnabled ? 1 : 0,
                ivars->_batching.dirtyKeyDeltaEnabled ? 1 : 0);
    }

    if (glm_ctx->renderer_backend) {
        mglRendererBackendDestroy(
            (MGLRendererBackendHandle **)&glm_ctx->renderer_backend);
    }
    if (glm_ctx->platform_renderer_shell) {
        CFRelease(glm_ctx->platform_renderer_shell);
        glm_ctx->platform_renderer_shell = NULL;
    }

    /* VIRTUALIZED AGX DETECTION: Create Metal device with virtualization safety */
    fprintf(stderr, "MGL INFO: VIRTUALIZED AGX - Creating Metal device with virtualization detection\n");

    MGLObjectId device =
        mglSend<MGLObjectId>(self, MGL_SEL(s_selMglCreateSystemDefaultDevice,
                                           "mglCreateSystemDefaultDevice"));
    if (!device) {
        fprintf(stderr, "MGL ERROR: Metal device not found - this is required for Apple Silicon\n");
        /* Intentional early return on critical Metal initialization failure.
         * The renderer is left in a PARTIALLY INITIALIZED state:
         *   SET: ctx, AGX GPU error tracking
         *        command recovery owner, _pipeline*Format/
         *        _pipelineCache.state->pipelineProgramName and
         *        _pipelineCache.state->pipelineStateCache.
         *   NIL: _device, _commandQueue, _view.
         * Continuing is pointless without a Metal device - every subsequent
         * operation depends on it. */
        return;
    }

    fprintf(stderr, "MGL INFO: Metal device created: %s\n",
            mglObjectDescriptionUTF8(device));

    MGLRendererBackendCreateInfo backendInfo = {
        .objc_device = (void *)device,
        .context = glm_ctx,
        .binding_slot_count = TEXTURE_UNITS,
        .query_capacity = 256u,
    };
    if (mglRendererBackendCreate(&backendInfo, &ivars->_backend) != 0) {
        fprintf(stderr, "MGL ERROR: failed to create Metal-cpp renderer backend\n");
        return;
    }
    glm_ctx->renderer_backend = ivars->_backend;
    glm_ctx->platform_renderer_shell = mglBridgingRetain(self);
    MGLRendererBackendLease initLease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &initLease) != 0) {
        fprintf(stderr, "MGL ERROR: failed to acquire backend lease during init\n");
        return;
    }
    ivars->_bindingStateOwner = mglRendererBackendLeaseGetOwner(
        &initLease, MGL_RENDERER_BACKEND_OWNER_BINDING);
    ivars->_queryStateOwner = mglRendererBackendLeaseGetOwner(
        &initLease, MGL_RENDERER_BACKEND_OWNER_QUERY);
    ivars->_gpuRecovery.commandRecoveryOwner = mglRendererBackendLeaseGetOwner(
        &initLease, MGL_RENDERER_BACKEND_OWNER_RECOVERY);
    fprintf(stderr, "MGL INFO: Metal-cpp renderer backend ready (%p)\n",
            (void *)ivars->_backend);
    {
        MGLRenderPassManager *passManager =
            (MGLRenderPassManager *)ivars->_renderPassManager;
        mglRenderAttachRuntimeOwners(
            glm_ctx,
            passManager->state->currentCommandBufferOwner,
            passManager->state->currentRenderEncoderOwner,
            passManager->state->renderPassStateOwner);
    }
    {
        MGLObjectId cache = (MGLObjectId)ivars->_pipelineCache;
        if (cache) {
            (void)mglSend<void>(cache, MGL_SEL(s_selSetDevice, "setDevice:"), device);
        }
    }

    /* Initialize AGX Capability Layer (centralized device detection +
     * capability queries + driver bug markers).  Replaces scattered
     * `containsString:@"AGX"` checks and hardcoded constants. */
    MGLCapabilityInit(&ivars->_core.capability, (void *)device);

    /* PROPER AGX VIRTUALIZATION DETECTION: Maintain Metal functionality with virtualization compatibility */
    signed char isVirtualized = ivars->_core.capability.isVirtualized;
    char deviceName[128];
    (void)mglRenderGetDeviceIdentity((const void *)device, NULL,
                                     deviceName, sizeof(deviceName));

    /* DETECTION: Check if running in QEMU virtualization but keep Metal enabled */
    if (isVirtualized) {
        isVirtualized = 1;
        fprintf(stderr, "MGL INFO: AGX device detected - enabling virtualization compatibility mode: %s\n",
                deviceName);
        fprintf(stderr, "MGL INFO: Metal functionality will be maintained with AGX virtualization safety measures\n");
    }

    /* Create command queue with virtualization-safe settings */
    if (isVirtualized) {
        fprintf(stderr, "MGL INFO: VIRTUALIZED AGX - Enabling virtualization-safe command queue settings\n");
    }

    uint32_t maxCommandBuffers = isVirtualized
        ? (uint32_t)MGLCapabilityMaxConcurrentCommandBuffers(&ivars->_core.capability)
        : 0u;
    void *commandQueue = NULL;
    (void)mglRendererBackendResetCommandQueue(ivars->_backend, maxCommandBuffers,
                                              &commandQueue);
    if (!mglRendererCommandQueue(ivars)) {
        fprintf(stderr, "MGL ERROR: Failed to create Metal command queue\n");
        /* Intentional early return on critical Metal initialization failure.
         * The renderer is left in a PARTIALLY INITIALIZED state:
         *   SET: ctx, AGX GPU error tracking
         *        fields, _pipeline*Format/_pipelineCache.state->pipelineProgramName,
         *        _pipelineCache.state->pipelineStateCache, _device,
         *        MTL4 compiler (if available), _capability.
         *   NIL: _commandQueue, _view.
         * Continuing is pointless without a command queue - no encoding or
         * submission is possible. */
        mglRendererBackendEnd(&initLease);
        return;
    }

    fprintf(stderr, "MGL INFO: Metal command queue created successfully\n");

    /* Load or create Binary Archive for PSO compile acceleration.
     * Gated by MGL_BINARY_ARCHIVE (default ON; =0 disables).
     * The archive is stored in the user's Caches directory and persists
     * compiled PSO binaries across launches, reducing cold-start PSO
     * compile time from ~10s to ~2s on subsequent launches. */
    {
        MGLObjectId cache = (MGLObjectId)ivars->_pipelineCache;
        if (cache &&
            mglSend<signed char>(cache, MGL_SEL(s_selIsBinaryArchiveEnabled,
                                                "isBinaryArchiveEnabled"))) {
            if (__builtin_available(macos 11.0, *)) {   /* @available(macOS 11.0, *) */
                (void)mglSend<void>(cache, MGL_SEL(s_selLoadBinaryArchive,
                                                   "loadBinaryArchive"));
            } else {
                (void)mglSend<void>(cache, MGL_SEL(s_selDisableBinaryArchive,
                                                   "disableBinaryArchive"));
            }
        }
    }

    (void)mglSend<void>(self, MGL_SEL(s_selSetView, "setView:"), view);

    /* PROPER FIX: Create Metal layer with AGX-safe settings in the platform shell. */
    fprintf(stderr, "MGL INFO: PROPER FIX - Creating Metal layer with AGX-safe settings\n");

    uint32_t requestedPixelFormat =
        ivars->ctx ? ivars->ctx->pixel_format.mtl_pixel_format : 0u;
    uint32_t pf = 0u;
    if (!mglSend<signed char>(self,
                              MGL_SEL(s_selMglConfigureMetalLayer,
                                      "mglConfigureMetalLayerWithDevice:"
                                      "requestedPixelFormat:actualPixelFormat:"),
                              device, requestedPixelFormat, &pf)) {
        fprintf(stderr, "MGL ERROR: Failed to create Metal layer\n");
        mglRendererBackendEnd(&initLease);
        return;
    }

    if (ivars->ctx &&
        ivars->ctx->pixel_format.mtl_pixel_format != (GLuint)pf) {
        fprintf(stderr,
                "MGL CAMetalLayer sync default framebuffer metal format glFormat=0x%x glType=0x%x oldMtl=%u newMtl=%lu\n",
                ivars->ctx->pixel_format.format,
                ivars->ctx->pixel_format.type,
                ivars->ctx->pixel_format.mtl_pixel_format,
                (unsigned long)pf);
        ivars->ctx->pixel_format.mtl_pixel_format = (GLuint)pf;
    }
    fprintf(stderr,
            "MGL CAMetalLayer pixelFormat=%lu requested=%lu glFormat=0x%x glType=0x%x\n",
            (unsigned long)pf, (unsigned long)requestedPixelFormat,
            ivars->ctx ? ivars->ctx->pixel_format.format : 0u,
            ivars->ctx ? ivars->ctx->pixel_format.type : 0u);
    /* Initial geometry: the renderer is created on the main thread (AppKit
     * window setup), so read the view geometry synchronously here.  Later
     * changes arrive via KVO -> mglMainThreadSyncViewGeometry. */
    if (mglSend<signed char>((MGLObjectId)objc_getClass("NSThread"),
                             MGL_SEL(s_selIsMainThread, "isMainThread"))) {
        (void)mglSend<void>(self, MGL_SEL(s_selMglMainThreadSync,
                                          "mglMainThreadSyncViewGeometry"));
    } else {
        (void)mglPlatformShellApplyPendingDrawableSizeCGSize((void *)self);
    }

    /* Observe view geometry changes so the GL thread never needs to touch
     * NSView/NSWindow/NSScreen.  KVO fires on the main thread (bounds is only
     * mutated there), publishing an atomic drawable-size snapshot.  The
     * "window" keyPath is observed as well so resize/backing notifications
     * can be attached lazily once the view joins a window. */
    (void)mglSend<void>(view,
                        MGL_SEL(s_selAddObserverForKeyPath,
                                "addObserver:forKeyPath:options:context:"),
                        self, mglNewUTF8String("bounds"), (unsigned long)0,
                        s_kvoViewGeometryContext);
    (void)mglSend<void>(view,
                        MGL_SEL(s_selAddObserverForKeyPath,
                                "addObserver:forKeyPath:options:context:"),
                        self, mglNewUTF8String("window"),
                        (unsigned long)kMGLKVOOptionInitial,
                        s_kvoViewGeometryContext);

    mglDrawBuffer(glm_ctx, (GLenum)mglRenderDefaultFrontBuffer());

    /* Create initial command buffer for AGX safety */
    try {
        mglPassManagerInstallNewCommandBufferFromQueue(
            (MGLRenderPassManager *)ivars->_renderPassManager,
            (void *)mglRendererCommandQueue(ivars));
        MGLRenderCommandBufferState commandState = {0};
        if (!mglRenderCommandBufferOwnerHasState(
                ((MGLRenderPassManager *)ivars->_renderPassManager)
                    ->state->currentCommandBufferOwner,
                &commandState)) {
            fprintf(stderr, "MGL ERROR: Failed to create initial Metal command buffer\n");
        }
    } catch (...) {
        fprintf(stderr,
                "MGL ERROR: Exception creating initial Metal command buffer: %s\n",
                mglCaughtExceptionDescription(mglTakeCaughtException()));
    }

    /* PROACTIVE TEXTURE CREATION: Create essential textures to break sync loop */
    fprintf(stderr, "MGL INFO: PROACTIVE - Creating essential textures to prevent magenta screen\n");
    (void)mglSend<void>(self, MGL_SEL(s_selCreateProactiveTextures,
                                      "createProactiveTextures"));

    /* GPU capture setup is exposed by MGLPlatformRendererShell when needed. */
    mglRendererBackendEnd(&initLease);
}

/* - (BOOL)mglRendererIsReady */
static signed char mglShellRendererIsReady(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLRendererIvars *ivars = mglRendererIvars(self);
    if (!ivars || !ivars->ctx) {
        return 0;
    }
    MGLRendererBackendLease lease = {};
    if (mglRendererBackendBeginContext(ivars->ctx, &lease) != 0) {
        return 0;
    }
    const MGLObjectId device = mglRendererDevice(ivars);
    const MGLObjectId commandQueue = mglRendererCommandQueue(ivars);
    const void *commandQueueOwner = mglRendererCommandQueueOwner(ivars);
    signed char ready =
        (ivars->_backend && device &&
         mglRendererBackendIsReady(ivars->_backend) == 1 && commandQueueOwner &&
         commandQueue &&
         mglSend<MGLObjectId>(self, MGL_SEL(s_selLayer, "layer")) &&
         ivars->_renderPassManager)
            ? 1
            : 0;
    if (ready) {
        MGLRenderCommandBufferState commandState = {0};
        ready = mglRenderCommandBufferOwnerHasState(
                    ((MGLRenderPassManager *)ivars->_renderPassManager)
                        ->state->currentCommandBufferOwner,
                    &commandState)
                    ? 1
                    : 0;
    }
    mglRendererBackendEnd(&lease);
    return ready;
}

/* - (void)mglBackendWillDestroy:(MGLRendererBackendHandle *)backend */
static void mglShellBackendWillDestroy(MGLObjectId self, SEL cmd,
                                       MGLRendererBackendHandle *backend)
{
    (void)cmd;
    MGLRendererIvars *ivars = mglRendererIvars(self);
    if (!ivars || ivars->_backend != backend) return;
    ivars->_backend = NULL;
    ivars->_bindingStateOwner = NULL;
    ivars->_queryStateOwner = NULL;
    ivars->_gpuRecovery.commandRecoveryOwner = NULL;
}

/* Publish view geometry to the GL thread as an atomic snapshot.  Main thread
 * only - this is the sole place NSView/NSWindow/NSScreen are read, so the
 * render thread never touches AppKit.  The GL thread consumes the snapshot via
 * mglApplyPendingDrawableSize and sets CAMetalLayer.drawableSize. */
static void mglShellMainThreadSyncViewGeometry(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLRendererIvars *ivars = mglRendererIvars(self);
    if (!ivars) {
        return;
    }
    /* NSAssert(NSThread.isMainThread, @"AppKit geometry must be read on main
     * thread").  A failing assert is a programming error either way; report it
     * and return rather than raise, so a mis-threaded call cannot take the
     * renderer down with it. */
    if (!mglSend<signed char>((MGLObjectId)objc_getClass("NSThread"),
                              MGL_SEL(s_selIsMainThread, "isMainThread"))) {
        fprintf(stderr, "MGL ERROR: AppKit geometry must be read on main thread\n");
        return;
    }
    MGLObjectId view = mglSend<MGLObjectId>(self, MGL_SEL(s_selView, "view"));
    if (!view ||
        !mglSend<signed char>(self, MGL_SEL(s_selHasMetalLayer,
                                            "mglHasMetalLayer"))) {
        return;
    }

    CGRect bounds = mglSend<CGRect>(view, MGL_SEL(s_selBounds, "bounds"));
    if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0) {
        bounds = mglSend<CGRect>(view, MGL_SEL(s_selFrameRect, "frame"));
        bounds.origin = CGPointMake(0.0, 0.0);   /* NSZeroPoint */
    }

    CGRect backingBounds =
        mglSend<CGRect>(view, MGL_SEL(s_selConvertRectToBacking,
                                      "convertRectToBacking:"),
                        bounds);
    double scale = 1.0;
    if (bounds.size.width > 0.0 && backingBounds.size.width > 0.0) {
        scale = backingBounds.size.width / bounds.size.width;
    } else {
        MGLObjectId window = mglSend<MGLObjectId>(view, MGL_SEL(s_selWindow, "window"));
        if (window) {
            scale = mglSend<double>(window, MGL_SEL(s_selBackingScaleFactor,
                                                    "backingScaleFactor"));
        } else {
            MGLObjectId screenClass = (MGLObjectId)objc_getClass("NSScreen");
            MGLObjectId screen =
                screenClass ? mglSend<MGLObjectId>(screenClass,
                                                   MGL_SEL(s_selMainScreen, "mainScreen"))
                            : NULL;
            if (screen) {
                scale = mglSend<double>(screen, MGL_SEL(s_selBackingScaleFactor,
                                                        "backingScaleFactor"));
            }
        }
        if (scale <= 0.0) {
            scale = 1.0;
        }
        backingBounds = CGRectMake(0.0, 0.0, bounds.size.width * scale,
                                   bounds.size.height * scale);   /* NSMakeRect */
    }

    (void)mglSend<void>(self,
                        MGL_SEL(s_selMglSetMetalLayerFrame,
                                "mglSetMetalLayerFrame:contentsScale:"),
                        bounds, scale);

    uint32_t pw = (uint32_t)mglMaxDouble(1.0, backingBounds.size.width + 0.5);
    uint32_t ph = (uint32_t)mglMaxDouble(1.0, backingBounds.size.height + 0.5);
    atomic_store_explicit(&ivars->_core.pendingDrawableW, pw, memory_order_relaxed);
    atomic_store_explicit(&ivars->_core.pendingDrawableH, ph, memory_order_relaxed);
    atomic_store_explicit(&ivars->_core.drawableSizeDirty, true, memory_order_release);
}

/* - (void)observeValueForKeyPath:ofObject:change:context: */
static void mglShellObserveValueForKeyPath(MGLObjectId self, SEL cmd,
                                           MGLObjectId keyPath, MGLObjectId object,
                                           MGLObjectId change, void *context)
{
    (void)cmd;
    (void)object;
    (void)change;
    if (context == s_kvoViewGeometryContext) {
        if (keyPath &&
            mglSend<signed char>(keyPath,
                                 MGL_SEL(s_selIsEqualToString, "isEqualToString:"),
                                 mglNewUTF8String("window"))) {
            (void)mglSend<void>(self, MGL_SEL(s_selMglUpdateWindowObserver,
                                              "mglUpdateWindowNotificationObserver"));
        }
        (void)mglSend<void>(self, MGL_SEL(s_selMglMainThreadSync,
                                          "mglMainThreadSyncViewGeometry"));
        return;
    }
    struct objc_super super = { self, class_getSuperclass(object_getClass(self)) };
    ((void (*)(struct objc_super *, SEL, MGLObjectId, MGLObjectId, MGLObjectId,
               void *))objc_msgSendSuper)(
        &super,
        MGL_SEL(s_selObserveValueForKeyPath,
                "observeValueForKeyPath:ofObject:change:context:"),
        keyPath, object, change, context);
}

/* Attach/detach window observation as the view's window changes.  The window
 * is not known when the renderer is created, so this is wired lazily. */
static void mglShellUpdateWindowNotificationObserver(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLRendererIvars *ivars = mglRendererIvars(self);
    if (!ivars) {
        return;
    }
    MGLObjectId view = mglSend<MGLObjectId>(self, MGL_SEL(s_selView, "view"));
    MGLObjectId window =
        view ? mglSend<MGLObjectId>(view, MGL_SEL(s_selWindow, "window")) : NULL;
    /* The ivar is __weak; read it through the weak table so a dead window reads
     * as nil instead of dangling. */
    MGLObjectId observed = objc_loadWeak((MGLObjectId *)&ivars->_observedWindow);
    if (window == observed) {
        return;
    }
    MGLObjectId center =
        mglSend<MGLObjectId>((MGLObjectId)objc_getClass("NSNotificationCenter"),
                             MGL_SEL(s_selDefaultCenter, "defaultCenter"));
    if (observed) {
        (void)mglSend<void>(center,
                            MGL_SEL(s_selRemoveObserverName,
                                    "removeObserver:name:object:"),
                            self, NSWindowDidResizeNotification, observed);
        (void)mglSend<void>(center,
                            MGL_SEL(s_selRemoveObserverName,
                                    "removeObserver:name:object:"),
                            self, NSWindowDidChangeBackingPropertiesNotification,
                            observed);
    }
    objc_storeWeak((MGLObjectId *)&ivars->_observedWindow, window);
    if (window) {
        (void)mglSend<void>(center,
                            MGL_SEL(s_selAddObserverSelectorName,
                                    "addObserver:selector:name:object:"),
                            self,
                            MGL_SEL(s_selMglWindowGeometryChanged,
                                    "mglWindowGeometryChanged:"),
                            NSWindowDidResizeNotification, window);
        (void)mglSend<void>(center,
                            MGL_SEL(s_selAddObserverSelectorName,
                                    "addObserver:selector:name:object:"),
                            self,
                            MGL_SEL(s_selMglWindowGeometryChanged,
                                    "mglWindowGeometryChanged:"),
                            NSWindowDidChangeBackingPropertiesNotification,
                            window);
    }
}

/* - (void)mglWindowGeometryChanged:(NSNotification *)notification */
static void mglShellWindowGeometryChanged(MGLObjectId self, SEL cmd,
                                          MGLObjectId notification)
{
    (void)cmd;
    (void)notification;
    (void)mglSend<void>(self, MGL_SEL(s_selMglMainThreadSync,
                                      "mglMainThreadSyncViewGeometry"));
}

/* PROACTIVE TEXTURE CREATION: Create essential textures during initialization
 * to break sync loop. */
static void mglShellCreateProactiveTextures(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLRendererIvars *ivars = mglRendererIvars(self);
    fprintf(stderr, "MGL PROACTIVE: Starting essential texture creation\n");

    try {
        if (mglRendererBackendCreateProactiveTexture(
                ivars ? ivars->_backend : NULL) == 0) {
            fprintf(stderr, "MGL PROACTIVE SUCCESS: Created 256x256 gradient texture (prevents magenta screen)\n");
        } else {
            fprintf(stderr, "MGL PROACTIVE ERROR: Could not create proactive texture\n");
        }
    } catch (...) {
        MGLObjectId exception = mglTakeCaughtException();
        const char *reason = exception
            ? mglUTF8String(mglSend<MGLObjectId>(
                  exception, MGL_SEL(s_selReason, "reason")))
            : NULL;
        fprintf(stderr,
                "MGL PROACTIVE ERROR: Exception creating proactive textures: %s\n",
                reason ? reason : "(null)");
    }

    fprintf(stderr, "MGL PROACTIVE: Essential texture creation completed\n");
}

/* CRITICAL FIX: Proper resource cleanup to prevent memory leaks and crashes.
 * A C dealloc must hand the object to the superclass at the end: the ARC
 * original had that call inserted by the compiler. */
static void mglShellDealloc(MGLObjectId self, SEL cmd)
{
    (void)cmd;
    MGLRendererIvars *ivars = mglRendererIvars(self);
    fprintf(stderr, "MGL INFO: MGLRenderer dealloc - cleaning up Metal resources\n");

    try {
        /* Remove the geometry observers before any view/state teardown. */
        MGLObjectId view = mglSend<MGLObjectId>(self, MGL_SEL(s_selView, "view"));
        if (view) {
            (void)mglSend<void>(view,
                                MGL_SEL(s_selRemoveObserverForKeyPath,
                                        "removeObserver:forKeyPath:context:"),
                                self, mglNewUTF8String("bounds"),
                                s_kvoViewGeometryContext);
            (void)mglSend<void>(view,
                                MGL_SEL(s_selRemoveObserverForKeyPath,
                                        "removeObserver:forKeyPath:context:"),
                                self, mglNewUTF8String("window"),
                                s_kvoViewGeometryContext);
        }
        /* Detach window notifications without the lazy re-wiring path. */
        MGLObjectId observed =
            ivars ? objc_loadWeak((MGLObjectId *)&ivars->_observedWindow) : NULL;
        if (observed) {
            MGLObjectId center = mglSend<MGLObjectId>(
                (MGLObjectId)objc_getClass("NSNotificationCenter"),
                MGL_SEL(s_selDefaultCenter, "defaultCenter"));
            (void)mglSend<void>(center,
                                MGL_SEL(s_selRemoveObserverName,
                                        "removeObserver:name:object:"),
                                self, NSWindowDidResizeNotification, observed);
            (void)mglSend<void>(center,
                                MGL_SEL(s_selRemoveObserverName,
                                        "removeObserver:name:object:"),
                                self,
                                NSWindowDidChangeBackingPropertiesNotification,
                                observed);
            objc_storeWeak((MGLObjectId *)&ivars->_observedWindow, NULL);
        }

        /* Stop any ongoing capture */
        (void)mglSend<void>(self, MGL_SEL(s_selStopCapture, "mglStopCapture"));

        /* End any active rendering */
        mglRendererEndRenderEncodingLocked((void *)self);

        /* Drop strong references held by the last-bound dedup cache before
         * releasing the underlying Metal resources below. */
        mglBindingInvalidateLastBoundState((void *)self);
        /* Cleanup command buffer and encoder */
        if (ivars && ivars->_renderPassManager) {
            MGLRenderPassManager *passManager =
                (MGLRenderPassManager *)ivars->_renderPassManager;
            MGLRenderCommandBufferState commandState = {0};
            if (mglRenderCommandBufferOwnerHasState(
                    passManager->state->currentCommandBufferOwner,
                    &commandState)) {
                fprintf(stderr, "MGL INFO: Releasing current command buffer\n");
                mglPassManagerDiscardCurrentCommandBuffer(passManager);
            }

            if (mglRenderEncoderOwnerHasCurrent(
                    passManager->state->currentRenderEncoderOwner) == 1) {
                fprintf(stderr, "MGL INFO: Releasing current render encoder\n");
                mglPassManagerClearCurrentRenderEncoder(passManager);
            }

            MGLRendererBackendShutdownResult shutdownResult = {0};
            if (ivars->_backend &&
                mglRendererBackendShutdown(ivars->_backend, &shutdownResult) != 0) {
                fprintf(stderr,
                        "MGL ERROR: renderer backend shutdown wait failed code=%lld\n",
                        shutdownResult.last_submission_error_code);
            }

            mglPassManagerSetRuntimeContext(passManager, NULL);
            mglPassManagerDestroy(passManager);
            ivars->_renderPassManager = NULL;
        }

        if (ivars) {
            mglRenderDetachRuntimeOwners(ivars->ctx);
        }

        MGLObjectId cache = ivars ? (MGLObjectId)ivars->_pipelineCache : NULL;
        if (cache) {
            const MGLPipelineCacheState *cacheState = mglRendererCacheState(ivars);
            if (cacheState && cacheState->pipelineState) {
                fprintf(stderr, "MGL INFO: Releasing pipeline state\n");
            }
            (void)mglSend<void>(cache, MGL_SEL(s_selSaveBinaryArchive,
                                               "saveBinaryArchive"));
            (void)mglSend<void>(cache, MGL_SEL(s_selShutdown, "shutdown"));
            /* ARC's `_pipelineCache = nil` releases the strong ivar. */
            mglReleaseObject(cache);
            ivars->_pipelineCache = NULL;
        }
        if (ivars && ivars->_backend &&
            mglRendererBackendIsDestroying(ivars->_backend) != 1) {
            if (ivars->ctx && ivars->ctx->renderer_backend == ivars->_backend) {
                mglRendererBackendDestroy(
                    (MGLRendererBackendHandle **)&ivars->ctx->renderer_backend);
            } else {
                mglRendererBackendDestroy(&ivars->_backend);
            }
        }
        if (ivars) {
            ivars->_backend = NULL;
            ivars->_bindingStateOwner = NULL;
            ivars->_queryStateOwner = NULL;
            ivars->_gpuRecovery.commandRecoveryOwner = NULL;
        }

        /* Cleanup drawable and layer */
        if (mglSend<MGLObjectId>(self, MGL_SEL(s_selDrawable, "drawable"))) {
            fprintf(stderr, "MGL INFO: Releasing drawable\n");
            (void)mglSend<void>(self, MGL_SEL(s_selSetDrawable, "setDrawable:"),
                                (MGLObjectId)NULL);
        }

        if (mglSend<signed char>(self, MGL_SEL(s_selHasMetalLayer,
                                               "mglHasMetalLayer"))) {
            fprintf(stderr, "MGL INFO: Removing and releasing layer\n");
            (void)mglSend<void>(self, MGL_SEL(s_selMglDetachMetalLayer,
                                              "mglDetachMetalLayer"));
        }

        /* Task 4: Release all address-stable snapshot arena chunks. */
        if (ivars) {
            mglDestroyBatchArena(&ivars->_batching.batchArena);
        }
        (void)mglSend<void>(self, MGL_SEL(s_selSetView, "setView:"),
                            (MGLObjectId)NULL);

    } catch (...) {
        fprintf(stderr, "MGL ERROR: Exception during dealloc cleanup: %s\n",
                mglCaughtExceptionDescription(mglTakeCaughtException()));
    }

    fprintf(stderr, "MGL INFO: MGLRenderer dealloc completed\n");

    struct objc_super super = { self, class_getSuperclass(object_getClass(self)) };
    ((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(
        &super, sel_registerName("dealloc"));
}

/* The two legacy creators the compatibility header declares.  Both are
 * init/create-family, so the renderer comes back +1. */
static MGLObjectId mglShellLegacyCreate(MGLObjectId self, SEL cmd, void *glm_ctx,
                                       MGLObjectId window, int initializing)
{
    (void)cmd;
    (void)self;
    if (!window || !glm_ctx) {
        fprintf(stderr,
                "MGL ERROR: renderer %s requires a window and GLMContext\n",
                initializing ? "initialization" : "creation");
        return NULL;
    }
    MGLObjectId renderer = mglCreateRendererObject();
    if (!renderer) {
        fprintf(stderr, "MGL ERROR: failed to allocate renderer\n");
        return NULL;
    }
    MGLObjectId view = mglCreateRendererView();
    if (!view) {
        fprintf(stderr, "MGL ERROR: failed to allocate renderer view\n");
        return NULL;
    }
    (void)mglSend<void>(view, MGL_SEL(s_selSetWantsLayer, "setWantsLayer:"),
                        (signed char)1);
    (void)mglSend<void>(window, MGL_SEL(s_selSetContentView, "setContentView:"), view);
    (void)mglSend<void>(renderer,
                        MGL_SEL(s_selCreateAndBind,
                                "createMGLRendererAndBindToContext:view:"),
                        glm_ctx, view);
    mglReleaseObject(view);
    return renderer;
}

static MGLObjectId mglShellInitRenderer(MGLObjectId self, SEL cmd, void *glm_ctx,
                                        MGLObjectId window)
{
    return mglShellLegacyCreate(self, cmd, glm_ctx, window, 1);
}

static MGLObjectId mglShellCreateRenderer(MGLObjectId self, SEL cmd, void *glm_ctx,
                                          MGLObjectId window)
{
    return mglShellLegacyCreate(self, cmd, glm_ctx, window, 0);
}

/* Rule 64: nothing calls this - the constructor attribute is the only entry
 * point.  The class is compiler-generated while the shell still is ObjC, so
 * these are added to it rather than to a class we registered ourselves. */
__attribute__((constructor))
static void mglInstallLifecycleMethods(void)
{
    Class rendererClass = objc_getClass("MGLRenderer");
    if (!rendererClass) {
        return;
    }
    class_addMethod(rendererClass,
                    sel_registerName("createMGLRendererAndBindToContext:view:"),
                    (IMP)mglShellCreateAndBind, "v@:^{GLMContextRec_t}@");
    class_addMethod(rendererClass, sel_registerName("mglRendererIsReady"),
                    (IMP)mglShellRendererIsReady, "c@:");
    class_addMethod(rendererClass, sel_registerName("mglBackendWillDestroy:"),
                    (IMP)mglShellBackendWillDestroy, "v@:^?");
    class_addMethod(rendererClass, sel_registerName("mglMainThreadSyncViewGeometry"),
                    (IMP)mglShellMainThreadSyncViewGeometry, "v@:");
    class_addMethod(rendererClass,
                    sel_registerName("observeValueForKeyPath:ofObject:change:context:"),
                    (IMP)mglShellObserveValueForKeyPath, "v@:@@@^v");
    class_addMethod(rendererClass,
                    sel_registerName("mglUpdateWindowNotificationObserver"),
                    (IMP)mglShellUpdateWindowNotificationObserver, "v@:");
    class_addMethod(rendererClass, sel_registerName("mglWindowGeometryChanged:"),
                    (IMP)mglShellWindowGeometryChanged, "v@:@");
    class_addMethod(rendererClass, sel_registerName("createProactiveTextures"),
                    (IMP)mglShellCreateProactiveTextures, "v@:");
    class_addMethod(rendererClass, sel_registerName("dealloc"),
                    (IMP)mglShellDealloc, "v@:");
    class_addMethod(rendererClass,
                    sel_registerName("initMGLRendererFromContext:andBindToWindow:"),
                    (IMP)mglShellInitRenderer, "@@:^v@");
    class_addMethod(rendererClass,
                    sel_registerName("createMGLRendererFromContext:andBindToWindow:"),
                    (IMP)mglShellCreateRenderer, "@@:^v@");
}

/* === the C entry points the rest of the library and GLFW use ============ */

void mglRendererPlatformBackendWillDestroy(
    void *platform_shell,
    MGLRendererBackendHandle *backend)
{
    if (!platform_shell) {
        return;
    }
    (void)mglSend<void>((MGLObjectId)platform_shell,
                        MGL_SEL(s_selMglBackendWillDestroy,
                                "mglBackendWillDestroy:"),
                        backend);
}

void *CppCreateMGLRendererFromContextAndBindToWindow(void *glm_ctx, void *window)
{
    if (!window || !glm_ctx) {
        fprintf(stderr,
                "MGL ERROR: renderer creation requires a window and GLMContext\n");
        return NULL;
    }
    MGLObjectId renderer = mglCreateRendererObject();
    if (!renderer) {
        fprintf(stderr, "MGL ERROR: failed to allocate renderer\n");
        return NULL;
    }
    /* just a plain bridge as the autorelease pool will try to release this and
     * crash on exit */
    MGLObjectId w = (MGLObjectId)window;
    if (!w) {
        fprintf(stderr, "MGL ERROR: invalid window handle\n");
        mglReleaseObject(renderer);
        return NULL;
    }
    MGLObjectId view = mglCreateRendererView();
    if (!view) {
        fprintf(stderr, "MGL ERROR: failed to allocate renderer view\n");
        mglReleaseObject(renderer);
        return NULL;
    }
    (void)mglSend<void>(view, MGL_SEL(s_selSetWantsLayer, "setWantsLayer:"),
                        (signed char)1);
    (void)mglSend<void>(w, MGL_SEL(s_selSetContentView, "setContentView:"), view);
    (void)mglSend<void>(renderer,
                        MGL_SEL(s_selCreateAndBind,
                                "createMGLRendererAndBindToContext:view:"),
                        glm_ctx, view);
    mglReleaseObject(view);
    if (!mglSend<signed char>(renderer, MGL_SEL(s_selMglRendererIsReady,
                                                "mglRendererIsReady"))) {
        fprintf(stderr, "MGL ERROR: renderer initialization failed closed\n");
        mglReleaseObject(renderer);
        return NULL;
    }
    /* Ownership: the returned pointer is NON-OWNING (borrowed).
     * The context retains the renderer through platform_renderer_shell.
     * The caller must NOT CFRelease/free the returned pointer, and must keep
     * glm_ctx alive while using the returned pointer.  The local +1 is dropped
     * here, exactly where ARC dropped it. */
    mglReleaseObject(renderer);
    return (void *)renderer;
}

void *CppCreateMGLRendererHeadless(void *glm_ctx)
{
    if (!glm_ctx) {
        fprintf(stderr,
                "MGL ERROR: headless renderer creation requires a GLMContext\n");
        return NULL;
    }
    MGLObjectId renderer = mglCreateRendererObject();
    if (!renderer) {
        fprintf(stderr, "MGL ERROR: failed to allocate headless renderer\n");
        return NULL;
    }

    /* Create a dummy NSView for headless rendering */
    MGLObjectId view = mglCreateRendererView();
    if (!view) {
        fprintf(stderr, "MGL ERROR: failed to allocate headless renderer view\n");
        mglReleaseObject(renderer);
        return NULL;
    }
    (void)mglSend<void>(view, MGL_SEL(s_selSetWantsLayer, "setWantsLayer:"),
                        (signed char)1);

    (void)mglSend<void>(renderer,
                        MGL_SEL(s_selCreateAndBind,
                                "createMGLRendererAndBindToContext:view:"),
                        glm_ctx, view);
    mglReleaseObject(view);
    if (!mglSend<signed char>(renderer, MGL_SEL(s_selMglRendererIsReady,
                                                "mglRendererIsReady"))) {
        fprintf(stderr,
                "MGL ERROR: headless renderer initialization failed closed\n");
        mglReleaseObject(renderer);
        return NULL;
    }
    /* Ownership: the returned pointer is NON-OWNING (borrowed); see above. */
    mglReleaseObject(renderer);
    return (void *)renderer;
}

void *CppCreateMGLRendererAndBindToContext(void *glm_ctx)
{
    /* Compatibility export used by reference libMGL.dylib.
     * Falls back to headless binding when no Cocoa window is supplied. */
    return CppCreateMGLRendererHeadless(glm_ctx);
}

/* === GLFW context-host ops (H2, docs/GLFW_MGL_INVOCATION_PLAN.md §3.3) ====
 * The GLFW fork used to drive MGLRenderer through objc_msgSend, so deleting
 * a selector on this side could only fail at run time
 * (-[MGLRenderer mglSetSwapInterval:]: unrecognized selector, log 208-⑥).
 * These entries replace every remaining GLFW→MGL message send with plain C
 * function pointers: a removed or renamed entry is now a compile-time event
 * for a rebuilt consumer, and the version/size fields let a stale consumer
 * degrade instead of crash. */

static void *mglContextHostCreateAndBind(void *glm_ctx, void *view)
{
    if (!glm_ctx || !view) {
        fprintf(stderr,
                "MGL ERROR: context-host create needs a GLMContext and a view\n");
        return NULL;
    }
    MGLScopedAutoreleasePool pool;
    MGLObjectId renderer = mglCreateRendererObject();
    if (!renderer) {
        fprintf(stderr, "MGL ERROR: failed to allocate renderer\n");
        return NULL;
    }
    mglShellCreateAndBind(renderer, NULL, (GLMContext)glm_ctx,
                          (MGLObjectId)view);
    /* +1 ownership transfers to the consumer; release_owner balances it. */
    return (void *)renderer;
}

static int mglContextHostRendererIsReady(void *owner)
{
    if (!owner) {
        return 0;
    }
    MGLScopedAutoreleasePool pool;
    return (int)mglShellRendererIsReady((MGLObjectId)owner, NULL);
}

static void mglContextHostSetSwapInterval(void *owner, int interval)
{
    MGLPlatformShellIvars *ivars = mglPlatformShellIvars((MGLObjectId)owner);
    if (!ivars) {
        return;
    }
    /* The deleted setter's exact contract: negative values clamp to 0, the
     * ivar records the request, and the layer follows for vsync pacing. */
    if (interval < 0) {
        interval = 0;
    }
    ivars->_swapInterval = interval;
    MGLObjectId layer = (MGLObjectId)ivars->_layer;
    if (layer) {
        MGLScopedAutoreleasePool pool;
        mglSend<void>(layer,
                      MGL_SEL(s_selSetDisplaySyncEnabled,
                              "setDisplaySyncEnabled:"),
                      (signed char)(interval > 0 ? 1 : 0));
    }
}

static void mglContextHostReleaseOwner(void *owner)
{
    mglReleaseObject((MGLObjectId)owner);
}

static const MGLContextHostOps g_mglContextHostOps = {
    MGL_CONTEXT_HOST_OPS_VERSION,
    (uint32_t)sizeof(MGLContextHostOps),
    mglContextHostCreateAndBind,
    mglContextHostRendererIsReady,
    mglContextHostSetSwapInterval,
    mglContextHostReleaseOwner,
};

const MGLContextHostOps *mglContextHostOps(void)
{
    return &g_mglContextHostOps;
}

} /* extern "C" */

#endif /* MGL_PLATFORM_SHELL_SMOKE */
