/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_manager_ops.c — C twins of the MGLRenderPassManager methods
 * that only touch the command state.  The manager keeps calling its own
 * methods; C callers use these.
 */

#include "mgl_render_pass_manager_ops.h"
#include "mgl_renderer_ports.h"

/* Local twin of the manager's file-static helper of the same name. */
static void mglRenderPassManagerSyncRuntimeOwners(MGLCommandState *state)
{
    GLMContext context = state ? state->runtimeContext : NULL;
    if (!context) {
        return;
    }
    mglRenderAttachRuntimeOwners(context, state->currentCommandBufferOwner,
                                 state->currentRenderEncoderOwner,
                                 state->renderPassStateOwner);
}

void mglRenderPassManagerEndCurrentRenderEncoder(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || !cs->currentRenderEncoderOwner ||
        mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }
    (void)mglRenderEndRenderEncoderOwner(cs->currentRenderEncoderOwner);
}

void mglRenderPassManagerClearCurrentRenderEncoder(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    /* encoder ended - invalidate FBO match cache. */
    mglRenderClearFboMatchCache(cs->renderPassIdentityOwner);
    mglRenderDestroyRenderEncoderOwner(&cs->currentRenderEncoderOwner);
    mglRenderPassManagerSyncRuntimeOwners(cs);
}

void mglRenderPassManagerDiscardCurrentCommandBuffer(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    mglRenderDiscardCommandBufferOwnerCurrent(cs->currentCommandBufferOwner);
    mglRenderDestroyMDIScratchOwner(&cs->mdiArgsScratchOwner);
    mglRenderPassManagerSyncRuntimeOwners(cs);
}
