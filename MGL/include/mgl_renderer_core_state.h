/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_core_state.h — the renderer's core state and the dual-proxy
 * invariant, C-visible.
 *
 * MGLRendererCoreState (the active state pointer, the capability snapshot, the
 * per-drawable sizes and the lock-free hand-off channels) used to be defined in
 * the Objective-C MGLRenderer_State.h, so the C flush driver reached it through
 * four wrapper ports and the dual-proxy invariant lived in Objective-C methods.
 * The record and the three invariant helpers are C now; the renderer still owns
 * the record as its ivar and the state-areas port hands out its address.
 *
 * DUAL-PROXY INVARIANT: `activeState` here and `ctx->active_state` are two
 * proxies for the same logical GLMState and must always agree (or this one is
 * NULL, meaning "the live state").  mglCoreAssertDualProxy checks it.
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
    GLMState *activeState;
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

/* Point both proxies at the replay workspace (batch flush begin). */
void mglCoreActivateReplayState(MGLRendererCoreState *core, GLMContext ctx);

/* Point both proxies back at the live state (batch replay teardown). */
void mglCoreRestoreLiveActiveState(MGLRendererCoreState *core, GLMContext ctx);

/* Debug checkpoint of the dual-proxy invariant. */
void mglCoreAssertDualProxy(const MGLRendererCoreState *core, GLMContext ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_CORE_STATE_H */
