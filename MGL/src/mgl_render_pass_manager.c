/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_manager.c — the MGLRenderPassManager class moved here
 * (P0-1, log 106).  Every method was a forward to the mglRender* facade with
 * `_state` substituted for `manager->state`, so the translation is mechanical:
 * `[self clearFboMatchCache]` -> mglPassManagerClearFboMatchCache(manager), and
 * the three file statics of MGLRenderPassManager.m moved with it.
 */

#include <stdlib.h>
#include <string.h>

#include "mgl_render_pass_manager.h"
#include "mgl_draw_buffer.h"   /* mglMetalDrawBufferCount / At, mglRenderEmptyDrawBuffer */
#include "mgl_render.h"

/* The C++ CommandBufferOwner is the only command-buffer creation path. */
static void *mglPassManagerCreateCommandBuffer(void **owner, void *command_queue)
{
    if (!owner || !command_queue) {
        return NULL;
    }
    void *command_buffer = NULL;
    int result = *owner
        ? mglRenderResetCommandBufferOwner(*owner, command_queue, &command_buffer)
        : mglRenderCreateCommandBufferOwner(command_queue, owner, &command_buffer);
    return result == 0 && command_buffer ? command_buffer : NULL;
}

static void *mglPassManagerMakeRenderEncoder(
    void *command_buffer_owner, const MGLRenderPassState *state)
{
    if (!command_buffer_owner || !state) {
        return NULL;
    }
    void *encoder = NULL;
    return mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
               command_buffer_owner, state, &encoder) == 0 && encoder
        ? encoder
        : NULL;
}

static void mglPassManagerSyncRuntimeOwners(MGLCommandState *state)
{
    GLMContext context = state ? state->runtimeContext : NULL;
    if (!context) {
        return;
    }
    mglRenderAttachRuntimeOwners(context,
                                 state->currentCommandBufferOwner,
                                 state->currentRenderEncoderOwner,
                                 state->renderPassStateOwner);
}

static void mglPassManagerSyncIdentityView(
    MGLCommandState *commandState,
    const MGLRenderPassIdentityState *identity)
{
    commandState->renderPassFramebuffer = (Framebuffer *)identity->framebuffer;
    commandState->renderPassFramebufferName = identity->framebuffer_name;
    commandState->renderPassDrawBuffer = identity->draw_buffer;
    commandState->renderPassDrawBufferCount = (GLsizei)identity->draw_buffer_count;
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; ++index) {
        commandState->renderPassDrawBuffers[index] = identity->draw_buffers[index];
    }
}

static void mglPassManagerStoreIdentity(
    MGLCommandState *commandState,
    const MGLRenderPassIdentityState *identity)
{
    if (!commandState->renderPassIdentityOwner &&
        mglRenderCreateRenderPassIdentityOwner(
            &commandState->renderPassIdentityOwner) != 0) {
        commandState->renderPassIdentityOwner = NULL;
    }
    if (commandState->renderPassIdentityOwner &&
        mglRenderUpdateRenderPassIdentity(
            commandState->renderPassIdentityOwner, identity) != 0) {
        mglRenderDestroyRenderPassIdentityOwner(
            &commandState->renderPassIdentityOwner);
    }
    mglPassManagerSyncIdentityView(commandState, identity);
}

MGLRenderPassManager *mglPassManagerCreate(void)
{
    MGLRenderPassManager *manager =
        (MGLRenderPassManager *)calloc(1, sizeof(*manager));
    if (!manager) {
        return NULL;
    }
    manager->state = &manager->command_state;
    mglPassManagerClearRenderPassIdentity(manager);
    return manager;
}

void mglPassManagerDestroy(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    mglPassManagerShutdown(manager);
    free(manager);
}

void mglPassManagerSetRuntimeContext(MGLRenderPassManager *manager,
                                     GLMContext context)
{
    if (!manager) {
        return;
    }
    MGLCommandState *state = manager->state;
    if (state->runtimeContext && state->runtimeContext != context) {
        mglRenderDetachRuntimeOwners(state->runtimeContext);
    }
    state->runtimeContext = context;
    mglPassManagerSyncRuntimeOwners(state);
}

void mglPassManagerUpdateRenderPassIdentityForContext(
    MGLRenderPassManager *manager, GLMContext context)
{
    if (!manager) {
        return;
    }
    /* render pass identity changed — invalidate FBO match cache. */
    mglPassManagerClearFboMatchCache(manager);
    GLMState *activeState = context ? context->active_state : NULL;
    MGLRenderPassIdentityState identity = {0};
    identity.framebuffer = activeState ? activeState->framebuffer : NULL;
    identity.framebuffer_name = identity.framebuffer
        ? ((Framebuffer *)identity.framebuffer)->name
        : 0u;
    identity.draw_buffer = activeState ? activeState->draw_buffer : 0u;
    identity.draw_buffer_count = context
        ? (uint32_t)mglMetalDrawBufferCount(context)
        : 0u;
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        identity.draw_buffers[index] =
            context && index < identity.draw_buffer_count
                ? mglMetalDrawBufferAt(context, (GLuint)index)
                : (GLenum)mglRenderEmptyDrawBuffer();
    }
    mglPassManagerStoreIdentity(manager->state, &identity);
}

void mglPassManagerClearRenderPassIdentity(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    /* render pass ended — invalidate FBO match cache. */
    mglPassManagerClearFboMatchCache(manager);
    MGLRenderPassIdentityState identity = {0};
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        identity.draw_buffers[index] = (GLenum)mglRenderEmptyDrawBuffer();
    }
    mglPassManagerStoreIdentity(manager->state, &identity);
}

void *mglPassManagerInstallNewCommandBufferFromQueue(
    MGLRenderPassManager *manager, void *command_queue)
{
    if (!manager) {
        return NULL;
    }
    MGLCommandState *state = manager->state;
    void *command_buffer = NULL;
    if (command_queue) {
        command_buffer = mglPassManagerCreateCommandBuffer(
            &state->currentCommandBufferOwner, command_queue);
    }
    if (!command_buffer) {
        mglRenderDestroyCommandBufferOwner(&state->currentCommandBufferOwner);
    }
    mglPassManagerResetMDIScratch(manager);
    mglPassManagerSyncRuntimeOwners(state);
    return command_buffer;
}

void *mglPassManagerDetachCurrentCommandBufferForSubmission(
    MGLRenderPassManager *manager)
{
    if (!manager) {
        return NULL;
    }
    MGLCommandState *state = manager->state;
    if (!state->currentCommandBufferOwner) {
        return NULL;
    }
    mglRenderDestroyCommandBufferSubmission(
        &state->detachedCommandBufferSubmission);
    void *detachedBuffer = NULL;
    if (mglRenderTakeCommandBufferSubmission(
            state->currentCommandBufferOwner,
            &state->detachedCommandBufferSubmission,
            &detachedBuffer) != 0 || !detachedBuffer) {
        return NULL;
    }
    mglPassManagerSyncRuntimeOwners(state);
    return detachedBuffer;
}

void mglPassManagerDiscardCurrentCommandBuffer(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    MGLCommandState *state = manager->state;
    mglRenderDiscardCommandBufferOwnerCurrent(state->currentCommandBufferOwner);
    mglPassManagerResetMDIScratch(manager);
    mglPassManagerSyncRuntimeOwners(state);
}

int mglPassManagerCommitCommandBufferTransaction(
    MGLRenderPassManager *manager, void *command_buffer,
    MGLCommandBufferRecoveryOwner *recovery_owner,
    int wait_for_completion, MGLRenderCommandBufferTransaction *result)
{
    if (!manager) {
        return -1;
    }
    MGLCommandState *state = manager->state;
    int transactionResult = mglRenderCommitCommandBufferTransaction(
        state->currentCommandBufferOwner,
        &state->detachedCommandBufferSubmission,
        command_buffer,
        recovery_owner,
        wait_for_completion ? 1u : 0u,
        result);
    mglPassManagerSyncRuntimeOwners(state);
    return transactionResult;
}

int mglPassManagerHasLastSubmittedCommandBuffer(MGLRenderPassManager *manager)
{
    if (!manager) {
        return 0;
    }
    return mglRenderCommandBufferOwnerHasLastSubmitted(
               manager->state->currentCommandBufferOwner) == 1;
}

int mglPassManagerWaitForLastSubmittedCommandBuffer(
    MGLRenderPassManager *manager, MGLRenderCommandBufferState *state)
{
    if (!manager) {
        return -1;
    }
    return mglRenderWaitCommandBufferOwnerLastSubmitted(
        manager->state->currentCommandBufferOwner, state);
}

void *mglPassManagerConsumeTransactionCreatedCurrentCommandBuffer(
    MGLRenderPassManager *manager)
{
    if (!manager) {
        return NULL;
    }
    MGLCommandState *state = manager->state;
    void *command_buffer = NULL;
    if (mglRenderCommandBufferOwnerConsumeTransactionCurrent(
            state->currentCommandBufferOwner, &command_buffer) != 1 ||
        !command_buffer) {
        return NULL;
    }
    mglPassManagerResetMDIScratch(manager);
    mglPassManagerSyncRuntimeOwners(state);
    return command_buffer;
}

void mglPassManagerReleaseDetachedCommandBufferIfOwned(
    MGLRenderPassManager *manager, void *command_buffer)
{
    if (!manager) {
        return;
    }
    MGLCommandState *state = manager->state;
    /* ownership guard via the C++ submission. */
    if (!state->detachedCommandBufferSubmission ||
        (command_buffer &&
         mglRenderCommandBufferSubmissionMatchesBuffer(
             state->detachedCommandBufferSubmission,
             command_buffer) != 1)) {
        return;
    }
    mglRenderDestroyCommandBufferSubmission(
        &state->detachedCommandBufferSubmission);
}

void mglPassManagerClearCurrentCommandBufferSyncListEntries(
    MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    /* entries are never dereferenced — Sync objects are owned by the GL sync
     * lifecycle. */
    MGLCommandState *state = manager->state;
    if (!state->currentCommandBufferOwner) {
        return;
    }
    mglRenderCommandBufferOwnerClearSyncs(state->currentCommandBufferOwner);
}

void *mglPassManagerDetachPendingEventWithSyncName(MGLRenderPassManager *manager,
                                                   uint32_t *sync_name_out)
{
    if (!manager) {
        return NULL;
    }
    /* transfers the owner's reference. */
    GLsizei syncName = 0;
    void *event = NULL;
    mglRenderPendingEventDetach(manager->state->pendingEventOwner, &syncName,
                                &event);
    if (sync_name_out) {
        *sync_name_out = (uint32_t)syncName;
    }
    if (!event) {
        return NULL;
    }
    return event;
}

void mglPassManagerInstallRenderEncoder(MGLRenderPassManager *manager,
                                        void *render_encoder)
{
    if (!manager) {
        return;
    }
    /* the C++ RenderEncoderOwner is the single source on BOTH gates. */
    /* new encoder — invalidate FBO match cache. */
    mglPassManagerClearFboMatchCache(manager);
    MGLCommandState *state = manager->state;
    if (render_encoder) {
        int result = state->currentRenderEncoderOwner
            ? mglRenderResetRenderEncoderOwner(state->currentRenderEncoderOwner,
                                               render_encoder)
            : mglRenderCreateRenderEncoderOwner(render_encoder,
                                                &state->currentRenderEncoderOwner);
        if (result != 0) {
            mglRenderDestroyRenderEncoderOwner(&state->currentRenderEncoderOwner);
        }
    } else {
        mglRenderDestroyRenderEncoderOwner(&state->currentRenderEncoderOwner);
    }
    mglPassManagerSyncRuntimeOwners(state);
}

void *mglPassManagerCreateRenderEncoder(MGLRenderPassManager *manager)
{
    if (!manager) {
        return NULL;
    }
    MGLCommandState *state = manager->state;
    if (!state->currentCommandBufferOwner || !state->renderPassStateOwner) {
        return NULL;
    }
    MGLRenderPassState renderPassState = {0};
    if (mglRenderGetRenderPassStateOwner(state->renderPassStateOwner,
                                         &renderPassState) != 0) {
        return NULL;
    }
    return mglPassManagerMakeRenderEncoder(state->currentCommandBufferOwner,
                                            &renderPassState);
}

void mglPassManagerEndCurrentRenderEncoder(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    MGLCommandState *state = manager->state;
    if (!state->currentRenderEncoderOwner ||
        mglRenderEncoderOwnerHasCurrent(state->currentRenderEncoderOwner) != 1) {
        return;
    }
    (void)mglRenderEndRenderEncoderOwner(state->currentRenderEncoderOwner);
}

void mglPassManagerClearCurrentRenderEncoder(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    /* encoder ended — invalidate FBO match cache. */
    mglPassManagerClearFboMatchCache(manager);
    MGLCommandState *state = manager->state;
    mglRenderDestroyRenderEncoderOwner(&state->currentRenderEncoderOwner);
    mglPassManagerSyncRuntimeOwners(state);
}

void mglPassManagerEndCommandBufferCommit(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    mglRenderCommandBufferOwnerEndCommit(
        manager->state->currentCommandBufferOwner);
}

void mglPassManagerResetMDIScratch(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    mglRenderDestroyMDIScratchOwner(&manager->state->mdiArgsScratchOwner);
}

void mglPassManagerInstallNewRenderPassDescriptor(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    /* new descriptor — invalidate FBO match cache. */
    mglPassManagerClearFboMatchCache(manager);
    MGLCommandState *state = manager->state;
    mglRenderDestroyRenderPassStateOwner(&state->renderPassStateOwner);
    if (mglRenderCreateDefaultRenderPassStateOwner(
            &state->renderPassStateOwner) != 0) {
        state->renderPassStateOwner = NULL;
    }
    mglPassManagerSyncRuntimeOwners(state);
}

void mglPassManagerSetFboMatchCacheResult(MGLRenderPassManager *manager,
                                          int result, uint32_t fbo_name,
                                          uint64_t generation)
{
    if (!manager) {
        return;
    }
    MGLCommandState *state = manager->state;
    if (state->renderPassIdentityOwner && fbo_name != 0u) {
        MGLRenderFboMatchCacheState cache = {
            .fbo_name = fbo_name,
            .generation = generation,
            .result = result,
        };
        mglRenderSetFboMatchCache(state->renderPassIdentityOwner, &cache);
    }
}

void mglPassManagerClearFboMatchCache(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    mglRenderClearFboMatchCache(manager->state->renderPassIdentityOwner);
}

void mglPassManagerSetCurrentDrawUsesRTSampledCopy(MGLRenderPassManager *manager,
                                                   int uses_rt_sampled_copy)
{
    if (!manager) {
        return;
    }
    manager->state->currentDrawUsesRTSampledCopy = uses_rt_sampled_copy;
}

void mglPassManagerSetDontCareFrameGeneration(MGLRenderPassManager *manager,
                                              uint32_t generation)
{
    if (!manager) {
        return;
    }
    manager->state->dontCareFrameGeneration = generation;
}

void mglPassManagerIncrementDontCareFrameGenerationWithWrap(
    MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    if (++manager->state->dontCareFrameGeneration == 0u) {
        /* skip 0 (texture stamp init) and the wrap sentinel. */
        manager->state->dontCareFrameGeneration = 2u;
    }
}

void mglPassManagerShutdown(MGLRenderPassManager *manager)
{
    if (!manager) {
        return;
    }
    MGLCommandState *state = manager->state;
    mglRenderDestroyRenderPassStateOwner(&state->renderPassStateOwner);
    mglPassManagerSyncRuntimeOwners(state);
    mglPassManagerClearCurrentRenderEncoder(manager);
    mglPassManagerDiscardCurrentCommandBuffer(manager);
    mglRenderDestroyMDIScratchOwner(&state->mdiArgsScratchOwner);
    mglPassManagerReleaseDetachedCommandBufferIfOwned(manager, NULL);
    mglPassManagerEndCommandBufferCommit(manager);
    mglRenderDestroyCommandBufferOwner(&state->currentCommandBufferOwner);
    mglPassManagerClearRenderPassIdentity(manager);
    mglRenderDestroyRenderPassIdentityOwner(&state->renderPassIdentityOwner);

    /* sync tracking list lives inside the C++ owner; its destructor frees it. */
    mglRenderDestroyPendingEventOwner(&state->pendingEventOwner);
    state->currentDrawUsesRTSampledCopy = 0;
}
