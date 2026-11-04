/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_core_state.h — renderer core record + active-state helpers.
 *
 * ARCH-003 / Phase 2: the active GLMState pointer lives only on
 * `ctx->active_state`. Activate/Restore are thin wrappers around
 * `mglContextSetActiveState`.
 */

#ifndef MGL_RENDERER_CORE_STATE_H
#define MGL_RENDERER_CORE_STATE_H

#include "glm_context.h"
#include "mgl_capability.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Drawable indices of the core state's per-drawable size table (moved here from
 * the Objective-C MGLRenderer_State.h). */
enum {
    _FRONT,
    _BACK,
    _FRONT_LEFT,
    _FRONT_RIGHT,
    _BACK_LEFT,
    _BACK_RIGHT,
    _MAX_DRAW_BUFFERS
};

typedef struct MGLDrawable_t {
    GLuint width;
    GLuint height;
} MGLDrawable;

typedef struct MGLRendererCoreState_t {
    MGLCapability capability;
    MGLDrawable drawBuffers[_MAX_DRAW_BUFFERS];
    uint8_t defaultDrawableWrittenSinceLastSwap;
    /* Last primitive mode the draw path recorded (was the MGLRenderer
     * _lastDrawPrimitiveMode ivar; the C draw host port writes it). */
    uint32_t lastDrawPrimitiveMode;
    /* Lock-free hand-off channels.  Written by the completion-handler thread
     * / main queue, drained (and resynchronized) on the GL thread. */
    _Atomic bool deviceResetRequested;
    _Atomic uint32_t pendingDrawableW;
    _Atomic uint32_t pendingDrawableH;
    _Atomic bool drawableSizeDirty;
} MGLRendererCoreState;

/* Point ctx->active_state at a flush-owned replay workspace.
 * `workspace` must already hold a valid copy of the pre-flush live state
 * (typically &pass->saved).  Passing NULL falls back to ctx->replay_state
 * and copies live into it — retained for callers that have no pass buffer. */
void mglCoreActivateReplayState(MGLRendererCoreState *core, GLMContext ctx,
                                GLMState *workspace);

/* Point ctx->active_state back at the live state (batch replay teardown). */
void mglCoreRestoreLiveActiveState(MGLRendererCoreState *core, GLMContext ctx);

/* Debug checkpoint: ctx->active_state non-NULL (ARCH-003). */
void mglCoreAssertDualProxy(const MGLRendererCoreState *core, GLMContext ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_CORE_STATE_H */
