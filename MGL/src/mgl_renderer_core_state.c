/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_core_state.c — the dual-proxy invariant, in C.
 *
 * Formerly -[MGLRenderer mglActivateReplayStateForContext:],
 * -[MGLRenderer mglRestoreLiveActiveStateForContext:] and
 * -[MGLRenderer mglAssertDualProxyInSyncForContext:], which the C flush driver
 * reached through shim ports.  Plain C: two pointer writes each and a memcpy of
 * the replay workspace.
 */

#include "mgl_renderer_core_state.h"
#include "mgl_encode_context.h"

#include <assert.h>
#include <string.h>

void mglCoreActivateReplayState(MGLRendererCoreState *core, GLMContext ctx,
                                GLMState *workspace)
{
    if (!core || !ctx) {
        return;
    }
    /* T11-1: prefer a caller-owned workspace (pass->saved) so flush pays one
     * full GLMState copy, not two.  The ctx->replay_state fallback keeps the
     * dual-proxy helpers usable without a flush pass. */
    if (!workspace) {
        workspace = &ctx->replay_state;
        memcpy(workspace, &ctx->state, sizeof(*workspace));
    }
    ctx->active_state = workspace;
    core->activeState = workspace;
}

void mglCoreRestoreLiveActiveState(MGLRendererCoreState *core, GLMContext ctx)
{
    if (!ctx) {
        return;
    }
    ctx->active_state = &ctx->state;
    if (core) {
        core->activeState = NULL;
    }
}

void mglCoreAssertDualProxy(const MGLRendererCoreState *core, GLMContext ctx)
{
    (void)core;
    (void)ctx;
    assert((!core || !ctx || core->activeState == NULL ||
            core->activeState == ctx->active_state) &&
           "DUAL-PROXY DESYNC: _activeState != ctx->active_state - "
           "MGL_STATE() and STATE() would read different GLMState objects");
}

void mglEncodeContextRequireReplayState(const MGLEncodeContext *enc,
                                        GLMContext ctx)
{
    assert(enc && "T0-1: encode context required");
    assert(enc->state && "T0-1: encode context missing explicit GLMState *");
    assert(ctx && enc->state == (void *)ctx->active_state &&
           "T0-1: encode state must be the active replay workspace");
}
