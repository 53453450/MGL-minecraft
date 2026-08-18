/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#import "MGLRenderPassManager.h"

#import "mgl_draw_buffer.h"
#include "mgl_env_flag.h"

static BOOL mglRenderPassManagerCommandBufferState(
    void *owner, MGLRenderCommandBufferState *stateOut)
{
    if (stateOut) memset(stateOut, 0, sizeof(*stateOut));
    return owner && stateOut &&
           mglRenderGetCommandBufferOwnerState(owner, stateOut) == 0;
}

static id mglRenderPassManagerCreateCommandBuffer(
    void **owner, id commandQueue)
{
    if (!owner || !commandQueue) return nil;
    void *commandBuffer = NULL;
    int result = *owner
        ? mglRenderResetCommandBufferOwner(
              *owner, (__bridge void *)commandQueue, &commandBuffer)
        : mglRenderCreateCommandBufferOwner(
              (__bridge void *)commandQueue, owner, &commandBuffer);
    return result == 0 && commandBuffer
        ? (__bridge id)commandBuffer
        : nil;
}

static id
mglRenderPassManagerCreateRenderEncoder(
    void *commandBufferOwner, const MGLRenderPassState *state)
{
    if (!commandBufferOwner || !state) return nil;
    void *encoder = NULL;
    return mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
               commandBufferOwner, state, &encoder) == 0 && encoder
        ? (__bridge id)encoder
        : nil;
}

static void mglRenderPassManagerSyncRuntimeOwners(MGLCommandState *state)
{
    GLMContext context = state ? state->runtimeContext : NULL;
    if (!context) return;
    mglRenderAttachRuntimeOwners(
        context,
        state->currentCommandBufferOwner,
        state->currentRenderEncoderOwner,
        state->renderPassStateOwner);
}

static void mglRenderPassManagerSyncIdentityView(
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

static void mglRenderPassManagerStoreIdentity(
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
    mglRenderPassManagerSyncIdentityView(commandState, identity);
}

@implementation MGLRenderPassManager

- (instancetype)init
{
    self = [super init];
    if (self) {
        [self clearRenderPassIdentity];
    }
    return self;
}

- (const MGLCommandState *)state
{
    return &_state;
}

- (void)setRuntimeContext:(GLMContext)context
{
    if (_state.runtimeContext && _state.runtimeContext != context) {
        mglRenderDetachRuntimeOwners(_state.runtimeContext);
    }
    _state.runtimeContext = context;
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (void)updateRenderPassIdentityForContext:(GLMContext)context
{
    /* render pass identity changed — invalidate FBO match cache. */
    [self clearFboMatchCache];
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
    mglRenderPassManagerStoreIdentity(&_state, &identity);
}

- (void)clearRenderPassIdentity
{
    /* render pass ended — invalidate FBO match cache. */
    [self clearFboMatchCache];
    MGLRenderPassIdentityState identity = {0};
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        identity.draw_buffers[index] = GL_NONE;
    }
    mglRenderPassManagerStoreIdentity(&_state, &identity);
}

- (void *)installNewCommandBufferFromQueue:(void *)commandQueue
{
    /* The C++ CommandBufferOwner is the only command-buffer creation path. */
    id commandBuffer = nil;
    if (commandQueue) {
        commandBuffer = mglRenderPassManagerCreateCommandBuffer(
            &_state.currentCommandBufferOwner,
            (__bridge id)commandQueue);
    }
    if (!commandBuffer) {
        mglRenderDestroyCommandBufferOwner(
            &_state.currentCommandBufferOwner);
    }
    [self resetMDIScratch];
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return (__bridge void *)commandBuffer;
}

- (void *)detachCurrentCommandBufferForSubmission
{
    if (!_state.currentCommandBufferOwner) return nil;
    mglRenderDestroyCommandBufferSubmission(
        &_state.detachedCommandBufferSubmission);
    void *detachedBuffer = NULL;
    if (mglRenderTakeCommandBufferSubmission(
            _state.currentCommandBufferOwner,
            &_state.detachedCommandBufferSubmission,
            &detachedBuffer) != 0 || !detachedBuffer) {
        return nil;
    }
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return detachedBuffer;
}

- (void)discardCurrentCommandBuffer
{
    mglRenderDiscardCommandBufferOwnerCurrent(
        _state.currentCommandBufferOwner);
    [self resetMDIScratch];
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (BOOL)commitDetachedCommandBufferIfOwned:(void *)commandBuffer
{
    /* ownership guard via the C++ submission. */
    if (!commandBuffer || !_state.detachedCommandBufferSubmission ||
        mglRenderCommandBufferSubmissionMatchesBuffer(
            _state.detachedCommandBufferSubmission,
            commandBuffer) != 1) {
        return NO;
    }
    if (mglRenderCommitCommandBufferSubmission(
            &_state.detachedCommandBufferSubmission) != 0) {
        return NO;
    }
    return YES;
}

- (int)commitCommandBufferTransaction:(void *)commandBuffer
                         recoveryOwner:(void *)recoveryOwner
                     waitForCompletion:(BOOL)waitForCompletion
                                result:(MGLRenderCommandBufferTransaction *)result
{
    int transactionResult = mglRenderCommitCommandBufferTransaction(
        _state.currentCommandBufferOwner,
        &_state.detachedCommandBufferSubmission,
        commandBuffer,
        recoveryOwner,
        waitForCompletion ? 1u : 0u,
        result);
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return transactionResult;
}

- (BOOL)hasLastSubmittedCommandBuffer
{
    return mglRenderCommandBufferOwnerHasLastSubmitted(
        _state.currentCommandBufferOwner) == 1;
}

- (int)waitForLastSubmittedCommandBuffer:(MGLRenderCommandBufferState *)state
{
    return mglRenderWaitCommandBufferOwnerLastSubmitted(
        _state.currentCommandBufferOwner, state);
}

- (void *)consumeTransactionCreatedCurrentCommandBuffer
{
    void *commandBuffer = NULL;
    if (mglRenderCommandBufferOwnerConsumeTransactionCurrent(
            _state.currentCommandBufferOwner, &commandBuffer) != 1 ||
        !commandBuffer) {
        return nil;
    }
    [self resetMDIScratch];
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return commandBuffer;
}

- (void)releaseDetachedCommandBufferIfOwned:(void *)commandBuffer
{
    /* ownership guard via the C++ submission. */
    if (!_state.detachedCommandBufferSubmission ||
        (commandBuffer &&
         mglRenderCommandBufferSubmissionMatchesBuffer(
             _state.detachedCommandBufferSubmission,
             commandBuffer) != 1)) {
        return;
    }
    mglRenderDestroyCommandBufferSubmission(
        &_state.detachedCommandBufferSubmission);
}

- (BOOL)appendSyncToCurrentCommandBuffer:(Sync *)sync
{
    /* the tracking list now lives inside the C++
     * command-buffer owner; this method is a thin adapter.  The list is
     * advisory only (never read by the wait paths), so the gate-off path
     * without an owner reports success as before. */
    if (!sync) {
        return NO;
    }
    if (!_state.currentCommandBufferOwner) {
        return YES;
    }
    return mglRenderCommandBufferOwnerAppendSync(
               _state.currentCommandBufferOwner, sync) == 0;
}

- (void)clearCurrentCommandBufferSyncListEntries
{
    /* entries are never dereferenced — Sync objects are
     * owned by the GL sync lifecycle. */
    if (!_state.currentCommandBufferOwner) {
        return;
    }
    mglRenderCommandBufferOwnerClearSyncs(_state.currentCommandBufferOwner);
}

- (void *)preparePendingEventWithDevice:(__unused void *)device
                                     syncName:(GLsizei)syncName
{
    /* the pending event slot lives inside the C++
     * PendingEventOwner; this method is a thin adapter. */
    if (!_state.pendingEventOwner &&
        mglRenderCreatePendingEventOwner(&_state.pendingEventOwner) != 0) {
        _state.pendingEventOwner = NULL;
        return nil;
    }
    void *event = NULL;
    if (mglRenderPendingEventPrepare(
            _state.pendingEventOwner, syncName, &event) != 0 || !event) {
        return nil;
    }
    return event;
}

- (void *)detachPendingEventWithSyncName:(GLuint *)syncNameOut
{
    /* transfers the owner's reference via __bridge_transfer. */
    GLsizei syncName = 0;
    void *event = NULL;
    mglRenderPendingEventDetach(
        _state.pendingEventOwner, &syncName, &event);
    if (syncNameOut) {
        *syncNameOut = (GLuint)syncName;
    }
    if (!event) {
        return nil;
    }
    return event;
}

- (void)clearPendingEvent
{
    /* discard the pending event; the owner stays. */
    if (_state.pendingEventOwner) {
        mglRenderPendingEventClear(_state.pendingEventOwner);
    }
}

- (void)installRenderEncoder:(void *)renderEncoder
{
    /* the C++ RenderEncoderOwner is the single source on
     * BOTH gates — the ObjC mirror is gone; reads go through the getter. */
    /* new encoder — invalidate FBO match cache. */
    [self clearFboMatchCache];
    if (renderEncoder) {
        int result = _state.currentRenderEncoderOwner
            ? mglRenderResetRenderEncoderOwner(
                  _state.currentRenderEncoderOwner,
                  renderEncoder)
            : mglRenderCreateRenderEncoderOwner(
                  renderEncoder,
                  &_state.currentRenderEncoderOwner);
        if (result != 0) {
            mglRenderDestroyRenderEncoderOwner(
                &_state.currentRenderEncoderOwner);
        }
    } else {
        mglRenderDestroyRenderEncoderOwner(
            &_state.currentRenderEncoderOwner);
    }
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (void *)createRenderEncoder
{
    if (!_state.currentCommandBufferOwner || !_state.renderPassStateOwner) {
        return nil;
    }
    MGLRenderPassState renderPassState = {0};
    if (mglRenderGetRenderPassStateOwner(
            _state.renderPassStateOwner, &renderPassState) != 0) {
        return nil;
    }
    id encoder = mglRenderPassManagerCreateRenderEncoder(
        _state.currentCommandBufferOwner, &renderPassState);
    return (__bridge void *)encoder;
}

- (void)endCurrentRenderEncoder
{
    if (!_state.currentRenderEncoderOwner ||
        mglRenderEncoderOwnerHasCurrent(
            _state.currentRenderEncoderOwner) != 1) return;
    (void)mglRenderEndRenderEncoderOwner(
        _state.currentRenderEncoderOwner);
}

- (void)clearCurrentRenderEncoder
{
    /* encoder ended — invalidate FBO match cache. */
    [self clearFboMatchCache];
    mglRenderDestroyRenderEncoderOwner(
        &_state.currentRenderEncoderOwner);
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (BOOL)beginCommandBufferCommit
{
    return mglRenderCommandBufferOwnerBeginCommit(
               _state.currentCommandBufferOwner) == 1;
}

- (void)endCommandBufferCommit
{
    mglRenderCommandBufferOwnerEndCommit(
        _state.currentCommandBufferOwner);
}

- (void *)mdiArgumentScratchBufferWithDevice:(void *)device
                                              length:(NSUInteger)length
                                              offset:(NSUInteger *)offsetOut
{
    if (offsetOut) {
        *offsetOut = 0;
    }
    MGLRenderCommandBufferState commandBufferState = {0};
    if (!device ||
        !mglRenderPassManagerCommandBufferState(
            _state.currentCommandBufferOwner, &commandBufferState) ||
        length == 0) {
        return nil;
    }

    /* both gates share the C++ MDIScratchOwner — the ObjC
     * gate-off allocator and the mirror fields are gone.  The returned buffer
     * is a borrowed reference (the owner keeps it alive and may swap it on
     * growth, same lifetime contract as the old mirror). */
    if (!_state.mdiArgsScratchOwner &&
        mglRenderCreateMDIScratchOwner(&_state.mdiArgsScratchOwner) != 0) {
        return nil;
    }
    void *buffer = NULL;
    uint64_t offset = 0;
    uint64_t capacity = 0;
    if (mglRenderAllocateMDIScratch(
            _state.mdiArgsScratchOwner, (uint64_t)length, 256u,
            &buffer, &offset, &capacity) != 0 || !buffer ||
        offset > NSUIntegerMax) {
        return nil;
    }
    if (offsetOut) *offsetOut = (NSUInteger)offset;
    return buffer;
}

- (void)resetMDIScratch
{
    mglRenderDestroyMDIScratchOwner(&_state.mdiArgsScratchOwner);
}

- (void)installNewRenderPassDescriptor
{
    /* new descriptor — invalidate FBO match cache. */
    [self clearFboMatchCache];
    mglRenderDestroyRenderPassStateOwner(
        &_state.renderPassStateOwner);
    if (mglRenderCreateDefaultRenderPassStateOwner(
            &_state.renderPassStateOwner) != 0) {
        _state.renderPassStateOwner = NULL;
    }
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (void)setFboMatchCacheResult:(BOOL)result
                       fboName:(GLuint)fboName
                     generation:(uint64_t)generation
{
    if (_state.renderPassIdentityOwner && fboName != 0u) {
        MGLRenderFboMatchCacheState cache = {
            .fbo_name = fboName,
            .generation = generation,
            .result = result,
        };
        mglRenderSetFboMatchCache(
            _state.renderPassIdentityOwner, &cache);
    }
}

- (void)clearFboMatchCache
{
    mglRenderClearFboMatchCache(_state.renderPassIdentityOwner);
}

- (void)setTraceReplayFlushId:(uint64_t)flushId batchIndex:(uint32_t)batchIndex
{
    _state.traceReplayFlushId = flushId;
    _state.traceReplayBatchIndex = batchIndex;
}

- (void)setCurrentDrawUsesRTSampledCopy:(BOOL)usesRTSampledCopy
{
    _state.currentDrawUsesRTSampledCopy = usesRTSampledCopy;
}

- (void)setDontCareFrameGeneration:(GLuint)generation
{
    _state.dontCareFrameGeneration = generation;
}

- (void)incrementDontCareFrameGenerationWithWrap
{
    if (++_state.dontCareFrameGeneration == 0u) {
        _state.dontCareFrameGeneration = 2u;  /* skip 0 (texture stamp init) and wrap sentinel */
    }
}

- (void)shutdown
{
    mglRenderDestroyRenderPassStateOwner(
        &_state.renderPassStateOwner);
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    [self clearCurrentRenderEncoder];
    [self discardCurrentCommandBuffer];
    mglRenderDestroyMDIScratchOwner(&_state.mdiArgsScratchOwner);
    [self releaseDetachedCommandBufferIfOwned:nil];
    [self endCommandBufferCommit];
    mglRenderDestroyCommandBufferOwner(
        &_state.currentCommandBufferOwner);
    [self clearRenderPassIdentity];
    mglRenderDestroyRenderPassIdentityOwner(
        &_state.renderPassIdentityOwner);

    /* sync tracking list lives inside the C++ owner;
     * the owner destructor frees it. */
    mglRenderDestroyPendingEventOwner(&_state.pendingEventOwner);
    _state.currentDrawUsesRTSampledCopy = NO;
}

@end
