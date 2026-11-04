/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * L0 platform shell ivar layout (Phase 8). Sole owner of renderer/shell
 * instance fields for the C++ shell; MGLRenderer_Private.h has been removed.
 *
 *     r->ctx                    -> mglRendererIvars(r)->ctx
 *     r->_core.pendingDrawableW -> mglRendererIvars(r)->_core.pendingDrawableW
 *     self->_view               -> mglPlatformShellIvars(self)->_view
 *
 * Offsets are resolved at runtime via class_addIvar registration order.
 */

#ifndef MGL_PLATFORM_SHELL_IVARS_H
#define MGL_PLATFORM_SHELL_IVARS_H

#include "mgl_objc_bridge.h"

#include "glm_context.h"
#include "mgl_batching_state.h"
#include "mgl_gpu_recovery_state.h"
#include "mgl_renderer_backend.h"
#include "mgl_renderer_core_state.h"
#include "mgl_resource_fallback_state.h"
#include "mgl_tessellation_state.h"

struct MGLRendererIvars {
    GLMContext ctx;
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

#endif /* MGL_PLATFORM_SHELL_IVARS_H */
