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
#include "mgl_binding_state_ops.h"      /* mglBindingInvalidateLastBoundState */
#include "mgl_trace_strategy.h"         /* mglClearFragmentTraceBindingsForRenderer */
#include "mgl_blit_sampled_copy.h"      /* GLSampled copies refresh */

#include <stdio.h>

/* Defined in MGLRenderer.m; declared in the Objective-C
 * MGLRenderer+RenderPass_Private.h. */
extern void mglLogRenderPassLifecycle(const char *tag, uint64_t call,
                                      GLMContext ctx, void *commandBufferOwner,
                                      void *renderEncoderOwner,
                                      void *renderPassStateOwner, void *drawable,
                                      Framebuffer *renderPassFramebuffer,
                                      GLuint renderPassFramebufferName,
                                      GLenum renderPassDrawBuffer,
                                      GLsizei renderPassDrawBufferCount);

/* Shell-provided drawable accessor (private ivar) and guarded call. */
extern void *mglPlatformShellDrawable(void *renderer);
extern int mglPlatformShellGuardedCall(void *renderer, const char *what,
                                       int (*body)(void *));

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

int mglRenderPassManagerCommitCommandBufferTransaction(
    void *renderer, void *commandBuffer, void *recoveryOwner,
    int waitForCompletion, MGLRenderCommandBufferTransaction *result)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return -1;
    }
    int transactionResult = mglRenderCommitCommandBufferTransaction(
        cs->currentCommandBufferOwner, &cs->detachedCommandBufferSubmission,
        commandBuffer, recoveryOwner, waitForCompletion ? 1u : 0u, result);
    mglRenderPassManagerSyncRuntimeOwners(cs);
    return transactionResult;
}

void mglRenderPassManagerReleaseDetachedCommandBufferIfOwned(
    void *renderer, void *commandBuffer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || !cs->detachedCommandBufferSubmission) {
        return;
    }
    if (commandBuffer &&
        mglRenderCommandBufferSubmissionMatchesBuffer(
            cs->detachedCommandBufferSubmission, commandBuffer) != 1) {
        return;
    }
    mglRenderDestroyCommandBufferSubmission(&cs->detachedCommandBufferSubmission);
}

/* Local twins of the manager's file-static helpers (same bodies). */
static void mglRenderPassManagerSyncIdentityView(
    MGLCommandState *commandState, const MGLRenderPassIdentityState *identity)
{
    commandState->renderPassFramebuffer = (Framebuffer *)identity->framebuffer;
    commandState->renderPassFramebufferName = identity->framebuffer_name;
    commandState->renderPassDrawBuffer = identity->draw_buffer;
    commandState->renderPassDrawBufferCount = (GLsizei)identity->draw_buffer_count;
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; ++index) {
        commandState->renderPassDrawBuffers[index] = identity->draw_buffers[index];
    }
}

static void mglRenderPassManagerStoreIdentity(
    MGLCommandState *commandState, const MGLRenderPassIdentityState *identity)
{
    if (!commandState->renderPassIdentityOwner &&
        mglRenderCreateRenderPassIdentityOwner(
            &commandState->renderPassIdentityOwner) != 0) {
        commandState->renderPassIdentityOwner = NULL;
    }
    if (commandState->renderPassIdentityOwner &&
        mglRenderUpdateRenderPassIdentity(commandState->renderPassIdentityOwner,
                                          identity) != 0) {
        mglRenderDestroyRenderPassIdentityOwner(
            &commandState->renderPassIdentityOwner);
    }
    mglRenderPassManagerSyncIdentityView(commandState, identity);
}

void mglRenderPassManagerClearRenderPassIdentity(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    /* render pass ended - invalidate FBO match cache. */
    mglRenderClearFboMatchCache(cs->renderPassIdentityOwner);
    MGLRenderPassIdentityState identity = {0};
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        identity.draw_buffers[index] = (GLenum)mglRenderEmptyDrawBuffer();
    }
    mglRenderPassManagerStoreIdentity(cs, &identity);
}

/* Former @try body of -[MGLRenderer endRenderEncodingLocked]. */
static int mglRendererEndRenderEncodingGuardedBody(void *renderer)
{
    mglRenderPassManagerEndCurrentRenderEncoder(renderer);
    mglRenderPassManagerClearCurrentRenderEncoder(renderer);
    mglClearFragmentTraceBindingsForRenderer(renderer, "end_render_encoding");
    mglRenderPassManagerClearRenderPassIdentity(renderer);
    return 1;
}

/* Body of -[MGLRenderer endRenderEncodingLocked] (P0-1).  The @try/@catch is the
 * shell guard; when it fails the same three cleanups run as the old @catch. */
void mglRendererEndRenderEncodingLocked(void *renderer)
{
    mglBindingInvalidateLastBoundState(renderer);

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs ||
        mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }

    /* An active render encoder means work was encoded into the current command
     * buffer, so the flush must not skip the commit. */
    if (areas.batching) {
        areas.batching->currentCommandBufferHasWork = 1u;
    }

    Framebuffer *endedFramebuffer = cs->renderPassFramebuffer;

    static uint64_t s_renderPassEndLogCount = 0;
    uint64_t hit = ++s_renderPassEndLogCount;
    if (hit <= 128ull || (hit % 1024ull) == 0ull) {
        mglLogRenderPassLifecycle("end", hit, areas.ctx,
                                  cs->currentCommandBufferOwner,
                                  cs->currentRenderEncoderOwner,
                                  cs->renderPassStateOwner,
                                  mglPlatformShellDrawable(renderer),
                                  cs->renderPassFramebuffer,
                                  cs->renderPassFramebufferName,
                                  cs->renderPassDrawBuffer,
                                  cs->renderPassDrawBufferCount);
    }

    if (!mglPlatformShellGuardedCall(renderer, "end render encoding",
                                     mglRendererEndRenderEncodingGuardedBody)) {
        fprintf(stderr,
                "MGL ERROR: Exception ending render encoder - ignoring\n");
        mglRenderPassManagerClearCurrentRenderEncoder(renderer);
        mglClearFragmentTraceBindingsForRenderer(renderer,
                                                 "end_render_encoding_exception");
        mglRenderPassManagerClearRenderPassIdentity(renderer);
    }

    /* A later batch may sample this render target before the command buffer is
     * submitted, so refresh its GL-visible copy immediately. */
    if (endedFramebuffer) {
        mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer(
            renderer, endedFramebuffer, "end_render_pass");
    }
}
