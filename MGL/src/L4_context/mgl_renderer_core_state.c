/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_core_state.c — active-state redirect helpers (ARCH-003).
 *
 * Formerly dual-proxy writers for `_core.activeState` + `ctx->active_state`.
 * Phase 2: only `mglContextSetActiveState` mutates the pointer.
 */

#include "mgl_renderer_core_state.h"
#include "mgl_encode_context.h"

#include <assert.h>
#include <string.h>

void mglCoreActivateReplayState(MGLRendererCoreState *core, GLMContext ctx,
                                GLMState *workspace)
{
    (void)core;
    if (!ctx) {
        return;
    }
    /* T11-1: prefer a caller-owned workspace (pass->saved) so flush pays one
     * full GLMState copy, not two.  The ctx->replay_state fallback keeps the
     * helpers usable without a flush pass. */
    if (!workspace) {
        workspace = &ctx->replay_state;
        memcpy(workspace, &ctx->state, sizeof(*workspace));
    }
    mglContextSetActiveState(ctx, workspace);
}

void mglCoreRestoreLiveActiveState(MGLRendererCoreState *core, GLMContext ctx)
{
    (void)core;
    if (!ctx) {
        return;
    }
    mglContextSetActiveState(ctx, &ctx->state);
}

void mglCoreAssertDualProxy(const MGLRendererCoreState *core, GLMContext ctx)
{
    (void)core;
    mglContextAssertActiveState(ctx);
}

void mglEncodeContextRequireReplayState(const MGLEncodeContext *enc,
                                        GLMContext ctx)
{
    assert(enc && "T0-1: encode context required");
    assert(enc->state && "T0-1: encode context missing explicit GLMState *");
    assert(ctx && enc->state == (void *)ctx->active_state &&
           "T0-1: encode state must be the active replay workspace");
}
