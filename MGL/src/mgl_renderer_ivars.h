/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_ivars.h - the renderer's @package ivars, seen from C++
 * (P0-1, T5 option (a); log 206).
 *
 * Once the shell is C++ there is no TU left that can include
 * MGLRenderer_Private.h (it pulls in AppKit and the Objective-C class
 * extension), so the converted code reaches the ivars through a mirror struct
 * and a runtime-resolved base offset instead of a compiler-computed one:
 *
 *     r->ctx                       -> mglRendererIvars(r)->ctx
 *     r->_core.pendingDrawableW    -> mglRendererIvars(r)->_core.pendingDrawableW
 *     self->_view                  -> mglPlatformShellIvars(self)->_view
 *
 * The mirror must keep the field order, names and types of MGLRenderer_Private.h
 * exactly: the offsets it reads are the real object's.  The private header also
 * aliases several of these with macros (`_pendingDrawableW` is
 * `_core.pendingDrawableW`, `_view` is `self.view`, and so on) - the converted
 * code spells the target out rather than carrying the macro along.
 *
 * When the class itself is registered with the runtime (T6) the same order is
 * recreated with class_addIvar, so this view keeps working unchanged.
 */

#ifndef MGL_RENDERER_IVARS_H
#define MGL_RENDERER_IVARS_H

#include "mgl_objc_bridge.h"

#include "glm_context.h"
#include "mgl_batching_state.h"
#include "mgl_gpu_recovery_state.h"
#include "mgl_renderer_backend.h"
#include "mgl_renderer_core_state.h"
#include "mgl_resource_fallback_state.h"
#include "mgl_tessellation_state.h"

/* @interface MGLRenderer () { @package ... } - sixteen ivars, in order. */
struct MGLRendererIvars {
    GLMContext ctx;                     /* the name the GLM macros expect */
    MGLRendererBackendHandle *_backend;
    void *_observedWindow;              /* __weak NSWindow * */
    MGLRendererCoreState _core;
    MGLGPURecoveryState _gpuRecovery;
    void *_pipelineCache;               /* MGLPipelineCache * */
    void *_queryStateOwner;
    void *_renderPassManager;           /* MGLRenderPassManager * */
    MGLResourceFallbackState _resourceFallback;
    void *_bindingStateOwner;
    MGLTessellationState _tessellation;
    MGLGeometryState _geometry;
    GLint _mglForcedMSSampleId;
    GLint _mglMSSamplePlaneOffset;
    signed char _mglInMSSampleDrawLoop; /* BOOL */
    MGLBatchingState _batching;
};

/* MGLPlatformRendererShell: one declared ivar plus the three synthesized by its
 * properties (NSView *view, CAMetalLayer *layer, id<CAMetalDrawable> drawable).
 * Only the order below is arbitrary - nothing reads these by offset except this
 * struct, and the runtime registration (T1) declares them in this order. */
struct MGLPlatformShellIvars {
    int _swapInterval;
    void *_view;                        /* NSView * */
    void *_layer;                       /* CAMetalLayer * */
    void *_drawable;                    /* id<CAMetalDrawable> */
};

static inline MGLRendererIvars *mglRendererIvars(MGLObjectId renderer)
{
    static ptrdiff_t s_offset = PTRDIFF_MIN;
    if (!renderer) {
        return NULL;
    }
    ptrdiff_t offset = MGL_IVAR(s_offset, object_getClass(renderer), "ctx");
    return offset >= 0 ? (MGLRendererIvars *)((char *)renderer + offset) : NULL;
}

static inline MGLPlatformShellIvars *mglPlatformShellIvars(MGLObjectId shell)
{
    static ptrdiff_t s_offset = PTRDIFF_MIN;
    if (!shell) {
        return NULL;
    }
    ptrdiff_t offset = MGL_IVAR(s_offset, object_getClass(shell), "_swapInterval");
    return offset >= 0 ? (MGLPlatformShellIvars *)((char *)shell + offset) : NULL;
}

#endif /* MGL_RENDERER_IVARS_H */
