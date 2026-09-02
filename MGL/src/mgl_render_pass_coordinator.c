/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * C command/render-pass coordinator — logic migrated from MGLRenderPassManager.m
 */

#include "mgl_render_pass_coordinator.h"

#include <string.h>

#include "mgl_draw_buffer.h"

static void mglCmdSyncRuntimeOwners(MGLCommandState *state)
{
    GLMContext context = state ? state->runtimeContext : NULL;
    if (!context) return;
    mglRenderAttachRuntimeOwners(
        context,
        state->currentCommandBufferOwner,
        state->currentRenderEncoderOwner,
        state->renderPassStateOwner);
}

static void mglCmdSyncIdentityView(MGLCommandState *commandState,
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

static void mglCmdStoreIdentity(MGLCommandState *commandState,
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
    mglCmdSyncIdentityView(commandState, identity);
}

static void *mglCmdCreateCommandBuffer(MGLCommandState *state, void *commandQueue)
{
    if (!state || !commandQueue) return NULL;
    void *commandBuffer = NULL;
    int result = state->currentCommandBufferOwner
        ? mglRenderResetCommandBufferOwner(
              state->currentCommandBufferOwner, commandQueue, &commandBuffer)
        : mglRenderCreateCommandBufferOwner(
              commandQueue, &state->currentCommandBufferOwner, &commandBuffer);
    return (result == 0 && commandBuffer) ? commandBuffer : NULL;
}

static void *mglCmdCreateRenderEncoderFromState(void *commandBufferOwner,
                                                const MGLRenderPassState *state)
{
    if (!commandBufferOwner || !state) return NULL;
    void *encoder = NULL;
    return mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
               commandBufferOwner, state, &encoder) == 0 && encoder
        ? encoder
        : NULL;
}

void mglCmdInit(MGLCommandState *state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    mglCmdClearRenderPassIdentity(state);
}

void mglCmdSetRuntimeContext(MGLCommandState *state, GLMContext context)
{
    if (!state) return;
    if (state->runtimeContext && state->runtimeContext != context) {
        mglRenderDetachRuntimeOwners(state->runtimeContext);
    }
    state->runtimeContext = context;
    mglCmdSyncRuntimeOwners(state);
}

void mglCmdUpdateRenderPassIdentityForContext(MGLCommandState *state,
                                              GLMContext context)
{
    if (!state) return;
    mglCmdClearFboMatchCache(state);
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
                : GL_NONE;
    }
    mglCmdStoreIdentity(state, &identity);
}

void mglCmdClearRenderPassIdentity(MGLCommandState *state)
{
    if (!state) return;
    mglCmdClearFboMatchCache(state);
    MGLRenderPassIdentityState identity = {0};
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        identity.draw_buffers[index] = GL_NONE;
    }
    mglCmdStoreIdentity(state, &identity);
}

void *mglCmdInstallNewCommandBufferFromQueue(MGLCommandState *state,
                                             void *commandQueue)
{
    if (!state) return NULL;
    void *commandBuffer = mglCmdCreateCommandBuffer(state, commandQueue);
    if (!commandBuffer) {
        mglRenderDestroyCommandBufferOwner(&state->currentCommandBufferOwner);
    }
    mglCmdResetMdiScratch(state);
    mglCmdSyncRuntimeOwners(state);
    return commandBuffer;
}

void *mglCmdDetachCurrentCommandBufferForSubmission(MGLCommandState *state)
{
    if (!state || !state->currentCommandBufferOwner) return NULL;
    mglRenderDestroyCommandBufferSubmission(&state->detachedCommandBufferSubmission);
    void *detachedBuffer = NULL;
    if (mglRenderTakeCommandBufferSubmission(
            state->currentCommandBufferOwner,
            &state->detachedCommandBufferSubmission,
            &detachedBuffer) != 0 || !detachedBuffer) {
        return NULL;
    }
    mglCmdSyncRuntimeOwners(state);
    return detachedBuffer;
}

void mglCmdDiscardCurrentCommandBuffer(MGLCommandState *state)
{
    if (!state) return;
    mglRenderDiscardCommandBufferOwnerCurrent(state->currentCommandBufferOwner);
    mglCmdResetMdiScratch(state);
    mglCmdSyncRuntimeOwners(state);
}

bool mglCmdCommitDetachedCommandBufferIfOwned(MGLCommandState *state,
                                              void *commandBuffer)
{
    if (!state || !commandBuffer || !state->detachedCommandBufferSubmission ||
        mglRenderCommandBufferSubmissionMatchesBuffer(
            state->detachedCommandBufferSubmission,
            commandBuffer) != 1) {
        return false;
    }
    if (mglRenderCommitCommandBufferSubmission(
            &state->detachedCommandBufferSubmission) != 0) {
        return false;
    }
    return true;
}

int mglCmdCommitCommandBufferTransaction(
    MGLCommandState *state, void *commandBuffer, void *recoveryOwner,
    bool waitForCompletion, MGLRenderCommandBufferTransaction *result)
{
    if (!state) return -1;
    int transactionResult = mglRenderCommitCommandBufferTransaction(
        state->currentCommandBufferOwner,
        &state->detachedCommandBufferSubmission,
        commandBuffer,
        recoveryOwner,
        waitForCompletion ? 1u : 0u,
        result);
    mglCmdSyncRuntimeOwners(state);
    return transactionResult;
}

bool mglCmdHasLastSubmittedCommandBuffer(const MGLCommandState *state)
{
    return state &&
           mglRenderCommandBufferOwnerHasLastSubmitted(
               state->currentCommandBufferOwner) == 1;
}

int mglCmdWaitForLastSubmittedCommandBuffer(MGLCommandState *state,
                                            MGLRenderCommandBufferState *out)
{
    if (!state) return -1;
    return mglRenderWaitCommandBufferOwnerLastSubmitted(
        state->currentCommandBufferOwner, out);
}

void *mglCmdConsumeTransactionCreatedCurrentCommandBuffer(MGLCommandState *state)
{
    if (!state) return NULL;
    void *commandBuffer = NULL;
    if (mglRenderCommandBufferOwnerConsumeTransactionCurrent(
            state->currentCommandBufferOwner, &commandBuffer) != 1 ||
        !commandBuffer) {
        return NULL;
    }
    mglCmdResetMdiScratch(state);
    mglCmdSyncRuntimeOwners(state);
    return commandBuffer;
}

void mglCmdReleaseDetachedCommandBufferIfOwned(MGLCommandState *state,
                                             void *commandBuffer)
{
    if (!state || !state->detachedCommandBufferSubmission) return;
    if (commandBuffer &&
        mglRenderCommandBufferSubmissionMatchesBuffer(
            state->detachedCommandBufferSubmission,
            commandBuffer) != 1) {
        return;
    }
    mglRenderDestroyCommandBufferSubmission(&state->detachedCommandBufferSubmission);
}

bool mglCmdAppendSyncToCurrentCommandBuffer(MGLCommandState *state, Sync *sync)
{
    if (!sync) return false;
    if (!state || !state->currentCommandBufferOwner) return true;
    return mglRenderCommandBufferOwnerAppendSync(
               state->currentCommandBufferOwner, sync) == 0;
}

void mglCmdClearCurrentCommandBufferSyncListEntries(MGLCommandState *state)
{
    if (!state || !state->currentCommandBufferOwner) return;
    mglRenderCommandBufferOwnerClearSyncs(state->currentCommandBufferOwner);
}

void *mglCmdPreparePendingEventWithDevice(MGLCommandState *state,
                                          void *device, GLsizei syncName)
{
    (void)device;
    if (!state) return NULL;
    if (!state->pendingEventOwner &&
        mglRenderCreatePendingEventOwner(&state->pendingEventOwner) != 0) {
        state->pendingEventOwner = NULL;
        return NULL;
    }
    void *event = NULL;
    if (mglRenderPendingEventPrepare(
            state->pendingEventOwner, syncName, &event) != 0 || !event) {
        return NULL;
    }
    return event;
}

void *mglCmdDetachPendingEventWithSyncName(MGLCommandState *state,
                                           GLuint *syncNameOut)
{
    if (!state) return NULL;
    GLsizei syncName = 0;
    void *event = NULL;
    mglRenderPendingEventDetach(state->pendingEventOwner, &syncName, &event);
    if (syncNameOut) *syncNameOut = (GLuint)syncName;
    return event;
}

void mglCmdClearPendingEvent(MGLCommandState *state)
{
    if (state && state->pendingEventOwner) {
        mglRenderPendingEventClear(state->pendingEventOwner);
    }
}

void mglCmdInstallRenderEncoder(MGLCommandState *state, void *renderEncoder)
{
    if (!state) return;
    mglCmdClearFboMatchCache(state);
    if (renderEncoder) {
        int result = state->currentRenderEncoderOwner
            ? mglRenderResetRenderEncoderOwner(
                  state->currentRenderEncoderOwner, renderEncoder)
            : mglRenderCreateRenderEncoderOwner(
                  renderEncoder, &state->currentRenderEncoderOwner);
        if (result != 0) {
            mglRenderDestroyRenderEncoderOwner(&state->currentRenderEncoderOwner);
        }
    } else {
        mglRenderDestroyRenderEncoderOwner(&state->currentRenderEncoderOwner);
    }
    mglCmdSyncRuntimeOwners(state);
}

void *mglCmdCreateRenderEncoder(MGLCommandState *state)
{
    if (!state || !state->currentCommandBufferOwner ||
        !state->renderPassStateOwner) {
        return NULL;
    }
    MGLRenderPassState renderPassState = {0};
    if (mglRenderGetRenderPassStateOwner(
            state->renderPassStateOwner, &renderPassState) != 0) {
        return NULL;
    }
    return mglCmdCreateRenderEncoderFromState(
        state->currentCommandBufferOwner, &renderPassState);
}

void mglCmdEndCurrentRenderEncoder(MGLCommandState *state)
{
    if (!state || !state->currentRenderEncoderOwner ||
        mglRenderEncoderOwnerHasCurrent(state->currentRenderEncoderOwner) != 1) {
        return;
    }
    (void)mglRenderEndRenderEncoderOwner(state->currentRenderEncoderOwner);
}

void mglCmdClearCurrentRenderEncoder(MGLCommandState *state)
{
    if (!state) return;
    mglCmdClearFboMatchCache(state);
    mglRenderDestroyRenderEncoderOwner(&state->currentRenderEncoderOwner);
    mglCmdSyncRuntimeOwners(state);
}

bool mglCmdBeginCommandBufferCommit(MGLCommandState *state)
{
    return state &&
           mglRenderCommandBufferOwnerBeginCommit(
               state->currentCommandBufferOwner) == 1;
}

void mglCmdEndCommandBufferCommit(MGLCommandState *state)
{
    if (state) {
        mglRenderCommandBufferOwnerEndCommit(state->currentCommandBufferOwner);
    }
}

void *mglCmdMdiArgumentScratchBufferWithDevice(MGLCommandState *state,
                                               void *device, size_t length,
                                               size_t *offsetOut)
{
    if (offsetOut) *offsetOut = 0;
    MGLRenderCommandBufferState commandBufferState = {0};
    if (!state || !device || length == 0 ||
        mglRenderGetCommandBufferOwnerState(
            state->currentCommandBufferOwner, &commandBufferState) != 0) {
        return NULL;
    }
    if (!state->mdiArgsScratchOwner &&
        mglRenderCreateMDIScratchOwner(&state->mdiArgsScratchOwner) != 0) {
        return NULL;
    }
    void *buffer = NULL;
    uint64_t offset = 0;
    uint64_t capacity = 0;
    if (mglRenderAllocateMDIScratch(
            state->mdiArgsScratchOwner, (uint64_t)length, 256u,
            &buffer, &offset, &capacity) != 0 || !buffer) {
        return NULL;
    }
    if (offsetOut) *offsetOut = (size_t)offset;
    return buffer;
}

void mglCmdResetMdiScratch(MGLCommandState *state)
{
    if (state) {
        mglRenderDestroyMDIScratchOwner(&state->mdiArgsScratchOwner);
    }
}

void mglCmdInstallNewRenderPassDescriptor(MGLCommandState *state)
{
    if (!state) return;
    mglCmdClearFboMatchCache(state);
    mglRenderDestroyRenderPassStateOwner(&state->renderPassStateOwner);
    if (mglRenderCreateDefaultRenderPassStateOwner(
            &state->renderPassStateOwner) != 0) {
        state->renderPassStateOwner = NULL;
    }
    mglCmdSyncRuntimeOwners(state);
}

void mglCmdSetFboMatchCacheResult(MGLCommandState *state, bool result,
                                  GLuint fboName, uint64_t generation)
{
    if (!state || !state->renderPassIdentityOwner || fboName == 0u) return;
    MGLRenderFboMatchCacheState cache = {
        .fbo_name = fboName,
        .generation = generation,
        .result = result,
    };
    mglRenderSetFboMatchCache(state->renderPassIdentityOwner, &cache);
}

void mglCmdClearFboMatchCache(MGLCommandState *state)
{
    if (state) {
        mglRenderClearFboMatchCache(state->renderPassIdentityOwner);
    }
}

void mglCmdSetTraceReplayFlushId(MGLCommandState *state, uint64_t flushId,
                                 uint32_t batchIndex)
{
    if (!state) return;
    state->traceReplayFlushId = flushId;
    state->traceReplayBatchIndex = batchIndex;
}

void mglCmdSetCurrentDrawUsesRTSampledCopy(MGLCommandState *state,
                                           bool usesRTSampledCopy)
{
    if (state) state->currentDrawUsesRTSampledCopy = usesRTSampledCopy;
}

void mglCmdSetDontCareFrameGeneration(MGLCommandState *state, GLuint generation)
{
    if (state) state->dontCareFrameGeneration = generation;
}

void mglCmdIncrementDontCareFrameGenerationWithWrap(MGLCommandState *state)
{
    if (!state) return;
    if (++state->dontCareFrameGeneration == 0u) {
        state->dontCareFrameGeneration = 2u;
    }
}

void mglCmdShutdown(MGLCommandState *state)
{
    if (!state) return;
    mglRenderDestroyRenderPassStateOwner(&state->renderPassStateOwner);
    mglCmdSyncRuntimeOwners(state);
    mglCmdClearCurrentRenderEncoder(state);
    mglCmdDiscardCurrentCommandBuffer(state);
    mglCmdResetMdiScratch(state);
    mglCmdReleaseDetachedCommandBufferIfOwned(state, NULL);
    mglCmdEndCommandBufferCommit(state);
    mglRenderDestroyCommandBufferOwner(&state->currentCommandBufferOwner);
    mglCmdClearRenderPassIdentity(state);
    mglRenderDestroyRenderPassIdentityOwner(&state->renderPassIdentityOwner);
    mglRenderDestroyPendingEventOwner(&state->pendingEventOwner);
    state->currentDrawUsesRTSampledCopy = false;
}

int mglCmdGetRenderPassIdentity(const MGLCommandState *state,
                                MGLRenderPassIdentityState *identity_out)
{
    if (!state || !identity_out) return -1;
    if (state->renderPassIdentityOwner &&
        mglRenderGetRenderPassIdentity(state->renderPassIdentityOwner,
                                       identity_out) == 0) {
        return 0;
    }
    return -1;
}

int mglCmdGetRenderPassPersistentState(const MGLCommandState *state,
                                       MGLRenderPassState *state_out)
{
    if (!state || !state_out || !state->renderPassStateOwner) return -1;
    return mglRenderGetRenderPassStateOwner(state->renderPassStateOwner,
                                            state_out);
}

int mglCmdProbeFboMatchCache(const MGLCommandState *state, GLuint fbo_name,
                             uint64_t generation, bool *result_out)
{
    if (!state || !result_out || fbo_name == 0u ||
        !state->renderPassIdentityOwner) {
        return 0;
    }
    MGLRenderFboMatchCacheState cache = {0};
    if (mglRenderGetFboMatchCache(state->renderPassIdentityOwner, &cache) != 0 ||
        cache.fbo_name != fbo_name || cache.generation != generation) {
        return 0;
    }
    *result_out = cache.result != 0;
    return 1;
}
