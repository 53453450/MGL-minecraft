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

#include <assert.h>
#include <string.h>

void mglCoreActivateReplayState(MGLRendererCoreState *core, GLMContext ctx)
{
    if (!core || !ctx) {
        return;
    }
    memcpy(&ctx->replay_state, &ctx->state, sizeof(ctx->replay_state));
    ctx->active_state = &ctx->replay_state;
    core->activeState = &ctx->replay_state;
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
