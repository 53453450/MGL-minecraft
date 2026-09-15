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
#include "mgl_renderer_ivars.h"

#include <CoreGraphics/CoreGraphics.h>
#include <dispatch/dispatch.h>

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm_limits.h"              /* MAX_COLOR_ATTACHMENTS */
#include "mgl_air_loader.h"           /* MGLRenderPipelineDescriptorState */
#include "mgl_aux_assets.h"
#include "mgl_batch_mtl_encode.h"
#include "mgl_batch_restore.h"        /* mglBatchFlushBegin/RunBatches/Teardown */
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
        return CGSizeZero;
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

} /* extern "C" */

#endif /* MGL_PLATFORM_SHELL_SMOKE */
