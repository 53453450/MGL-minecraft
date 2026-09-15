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

#include "mgl_aux_assets.h"
#include "mgl_batch_mtl_encode.h"
#include "mgl_render.h"
#include "mgl_renderer_backend.h"
#include "mgl_platform_shell_internal.h"
#include "mgl_renderer_ports.h"
#include "mgl_shader_abi.h"
#include "mgl_shader_resource.h"
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

} /* extern "C" */

#endif /* MGL_PLATFORM_SHELL_SMOKE */
