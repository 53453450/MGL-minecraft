#import "MGLRenderPassManager.h"

#import "mgl_draw_buffer.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp_objc.h"

static void mglRenderPassManagerSyncRuntimeOwners(MGLCommandState *state)
{
    GLMContext context = state ? state->runtimeContext : NULL;
    if (!context) return;
    mglRenderCppAttachRuntimeOwners(
        context,
        state->currentCommandBufferOwner,
        state->currentRenderEncoderOwner,
        state->renderPassStateOwner);
}

static void mglRenderPassManagerSyncIdentityView(
    MGLCommandState *commandState,
    const MGLRenderCppRenderPassIdentityState *identity)
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
    const MGLRenderCppRenderPassIdentityState *identity)
{
    if (!commandState->renderPassIdentityOwner &&
        mglRenderCppCreateRenderPassIdentityOwner(
            &commandState->renderPassIdentityOwner) != 0) {
        commandState->renderPassIdentityOwner = NULL;
    }
    if (commandState->renderPassIdentityOwner &&
        mglRenderCppUpdateRenderPassIdentity(
            commandState->renderPassIdentityOwner, identity) != 0) {
        mglRenderCppDestroyRenderPassIdentityOwner(
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
        mglRenderCppDetachRuntimeOwners(_state.runtimeContext);
    }
    _state.runtimeContext = context;
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (void)updateRenderPassIdentityForContext:(GLMContext)context
{
    /* render pass identity changed — invalidate FBO match cache. */
    [self clearFboMatchCache];
    GLMState *activeState = context ? context->active_state : NULL;
    MGLRenderCppRenderPassIdentityState identity = {0};
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
    MGLRenderCppRenderPassIdentityState identity = {0};
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        identity.draw_buffers[index] = GL_NONE;
    }
    mglRenderPassManagerStoreIdentity(&_state, &identity);
}

- (MGLMetalCommandBufferRef)installNewCommandBufferFromQueue:(MGLMetalCommandQueueRef)commandQueue
{
    /* The C++ CommandBufferOwner is the only command-buffer creation path. */
    MGLMetalCommandBufferRef commandBuffer = nil;
    if (commandQueue) {
        commandBuffer = mglRenderCppCreateOrResetCommandBufferOwner(
            &_state.currentCommandBufferOwner, commandQueue);
    }
    if (!commandBuffer) {
        mglRenderCppDestroyCommandBufferOwner(
            &_state.currentCommandBufferOwner);
    }
    [self resetMDIScratch];
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return commandBuffer;
}

- (MGLMetalCommandBufferRef)detachCurrentCommandBufferForSubmission
{
    if (!_state.currentCommandBufferOwner) return nil;
    mglRenderCppDestroyCommandBufferSubmission(
        &_state.detachedCommandBufferSubmission);
    void *detachedBuffer = NULL;
    if (mglRenderCppTakeCommandBufferSubmission(
            _state.currentCommandBufferOwner,
            &_state.detachedCommandBufferSubmission,
            &detachedBuffer) != 0 || !detachedBuffer) {
        return nil;
    }
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return (__bridge MGLMetalCommandBufferRef)detachedBuffer;
}

- (void)discardCurrentCommandBuffer
{
    mglRenderCppDiscardCommandBufferOwnerCurrent(
        _state.currentCommandBufferOwner);
    [self resetMDIScratch];
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (BOOL)commitDetachedCommandBufferIfOwned:(MGLMetalCommandBufferRef)commandBuffer
{
    /* P4.5 (item 1141): ownership guard via the C++ submission. */
    if (!commandBuffer || !_state.detachedCommandBufferSubmission ||
        mglRenderCppCommandBufferSubmissionMatchesBuffer(
            _state.detachedCommandBufferSubmission,
            (__bridge void *)commandBuffer) != 1) {
        return NO;
    }
    if (mglRenderCppCommitCommandBufferSubmission(
            &_state.detachedCommandBufferSubmission) != 0) {
        return NO;
    }
    return YES;
}

- (int)commitCommandBufferTransaction:(MGLMetalCommandBufferRef)commandBuffer
                         recoveryOwner:(void *)recoveryOwner
                     waitForCompletion:(BOOL)waitForCompletion
                                result:(MGLRenderCppCommandBufferTransaction *)result
{
    int transactionResult = mglRenderCppCommitCommandBufferTransaction(
        _state.currentCommandBufferOwner,
        &_state.detachedCommandBufferSubmission,
        (__bridge void *)commandBuffer,
        recoveryOwner,
        waitForCompletion ? 1u : 0u,
        result);
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return transactionResult;
}

- (BOOL)hasLastSubmittedCommandBuffer
{
    return mglRenderCppCommandBufferOwnerHasLastSubmitted(
        _state.currentCommandBufferOwner) == 1;
}

- (int)waitForLastSubmittedCommandBuffer:(MGLRenderCppCommandBufferState *)state
{
    return mglRenderCppWaitCommandBufferOwnerLastSubmitted(
        _state.currentCommandBufferOwner, state);
}

- (MGLMetalCommandBufferRef)consumeTransactionCreatedCurrentCommandBuffer
{
    void *commandBuffer = NULL;
    if (mglRenderCppCommandBufferOwnerConsumeTransactionCurrent(
            _state.currentCommandBufferOwner, &commandBuffer) != 1 ||
        !commandBuffer) {
        return nil;
    }
    [self resetMDIScratch];
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    return (__bridge MGLMetalCommandBufferRef)commandBuffer;
}

- (void)releaseDetachedCommandBufferIfOwned:(MGLMetalCommandBufferRef)commandBuffer
{
    /* P4.5 (item 1141): ownership guard via the C++ submission. */
    if (!_state.detachedCommandBufferSubmission ||
        (commandBuffer &&
         mglRenderCppCommandBufferSubmissionMatchesBuffer(
             _state.detachedCommandBufferSubmission,
             (__bridge void *)commandBuffer) != 1)) {
        return;
    }
    mglRenderCppDestroyCommandBufferSubmission(
        &_state.detachedCommandBufferSubmission);
}

- (BOOL)appendSyncToCurrentCommandBuffer:(Sync *)sync
{
    /* P4.5 (item 1141): the tracking list now lives inside the C++
     * command-buffer owner; this method is a thin adapter.  The list is
     * advisory only (never read by the wait paths), so the gate-off path
     * without an owner reports success as before. */
    if (!sync) {
        return NO;
    }
    if (!_state.currentCommandBufferOwner) {
        return YES;
    }
    return mglRenderCppCommandBufferOwnerAppendSync(
               _state.currentCommandBufferOwner, sync) == 0;
}

- (void)clearCurrentCommandBufferSyncListEntries
{
    /* P4.5 (item 1141): entries are never dereferenced — Sync objects are
     * owned by the GL sync lifecycle. */
    if (!_state.currentCommandBufferOwner) {
        return;
    }
    mglRenderCppCommandBufferOwnerClearSyncs(_state.currentCommandBufferOwner);
}

- (MGLMetalEventRef)preparePendingEventWithDevice:(__unused MGLMetalDeviceRef)device
                                     syncName:(GLsizei)syncName
{
    /* P4.5 (item 1141): the pending event slot lives inside the C++
     * PendingEventOwner; this method is a thin adapter. */
    if (!_state.pendingEventOwner &&
        mglRenderCppCreatePendingEventOwner(&_state.pendingEventOwner) != 0) {
        _state.pendingEventOwner = NULL;
        return nil;
    }
    void *event = NULL;
    if (mglRenderCppPendingEventPrepare(
            _state.pendingEventOwner, syncName, &event) != 0 || !event) {
        return nil;
    }
    return (__bridge MGLMetalEventRef)event;
}

- (MGLMetalEventRef)detachPendingEventWithSyncName:(GLuint *)syncNameOut
{
    /* P4.5 (item 1141): transfers the owner's reference via __bridge_transfer. */
    GLsizei syncName = 0;
    void *event = NULL;
    mglRenderCppPendingEventDetach(
        _state.pendingEventOwner, &syncName, &event);
    if (syncNameOut) {
        *syncNameOut = (GLuint)syncName;
    }
    if (!event) {
        return nil;
    }
    return (__bridge_transfer MGLMetalEventRef)event;
}

- (void)clearPendingEvent
{
    /* P4.5 (item 1141): discard the pending event; the owner stays. */
    if (_state.pendingEventOwner) {
        mglRenderCppPendingEventClear(_state.pendingEventOwner);
    }
}

- (void)installRenderEncoder:(MGLMetalRenderCommandEncoderRef)renderEncoder
{
    /* P4.5 (item 1141): the C++ RenderEncoderOwner is the single source on
     * BOTH gates — the ObjC mirror is gone; reads go through the getter. */
    /* new encoder — invalidate FBO match cache. */
    [self clearFboMatchCache];
    if (renderEncoder) {
        int result = _state.currentRenderEncoderOwner
            ? mglRenderCppResetRenderEncoderOwner(
                  _state.currentRenderEncoderOwner,
                  (__bridge void *)renderEncoder)
            : mglRenderCppCreateRenderEncoderOwner(
                  (__bridge void *)renderEncoder,
                  &_state.currentRenderEncoderOwner);
        if (result != 0) {
            mglRenderCppDestroyRenderEncoderOwner(
                &_state.currentRenderEncoderOwner);
        }
    } else {
        mglRenderCppDestroyRenderEncoderOwner(
            &_state.currentRenderEncoderOwner);
    }
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (MGLMetalRenderCommandEncoderRef)createRenderEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    (void)descriptor;
    if (!_state.currentCommandBufferOwner || !_state.renderPassStateOwner) {
        return nil;
    }
    MGLRenderCppRenderPassState renderPassState = {0};
    if (mglRenderCppGetRenderPassStateOwner(
            _state.renderPassStateOwner, &renderPassState) != 0) {
        return nil;
    }
    return mglRenderCreateRenderEncoderForCommandBufferOwner(
        _state.currentCommandBufferOwner, &renderPassState);
}

- (void)endCurrentRenderEncoder
{
    if (!_state.currentRenderEncoderOwner ||
        mglRenderCppRenderEncoderOwnerHasCurrent(
            _state.currentRenderEncoderOwner) != 1) return;
    (void)mglRenderCppEndRenderEncoderOwner(
        _state.currentRenderEncoderOwner);
}

- (void)clearCurrentRenderEncoder
{
    /* encoder ended — invalidate FBO match cache. */
    [self clearFboMatchCache];
    mglRenderCppDestroyRenderEncoderOwner(
        &_state.currentRenderEncoderOwner);
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (BOOL)beginCommandBufferCommit
{
    return mglRenderCppCommandBufferOwnerBeginCommit(
               _state.currentCommandBufferOwner) == 1;
}

- (void)endCommandBufferCommit
{
    mglRenderCppCommandBufferOwnerEndCommit(
        _state.currentCommandBufferOwner);
}

- (MGLMetalBufferRef)mdiArgumentScratchBufferWithDevice:(MGLMetalDeviceRef)device
                                              length:(NSUInteger)length
                                              offset:(NSUInteger *)offsetOut
{
    if (offsetOut) {
        *offsetOut = 0;
    }
    MGLRenderCppCommandBufferState commandBufferState = {0};
    if (!device ||
        !mglRenderCommandBufferOwnerState(
            _state.currentCommandBufferOwner, &commandBufferState) ||
        length == 0) {
        return nil;
    }

    /* P4.5 (item 1155): both gates share the C++ MDIScratchOwner — the ObjC
     * gate-off allocator and the mirror fields are gone.  The returned buffer
     * is a borrowed reference (the owner keeps it alive and may swap it on
     * growth, same lifetime contract as the old mirror). */
    if (!_state.mdiArgsScratchOwner &&
        mglRenderCppCreateMDIScratchOwner(&_state.mdiArgsScratchOwner) != 0) {
        return nil;
    }
    void *buffer = NULL;
    uint64_t offset = 0;
    uint64_t capacity = 0;
    if (mglRenderCppAllocateMDIScratch(
            _state.mdiArgsScratchOwner, (uint64_t)length, 256u,
            &buffer, &offset, &capacity) != 0 || !buffer ||
        offset > NSUIntegerMax) {
        return nil;
    }
    if (offsetOut) *offsetOut = (NSUInteger)offset;
    return (__bridge MGLMetalBufferRef)buffer;
}

- (void)resetMDIScratch
{
    mglRenderCppDestroyMDIScratchOwner(&_state.mdiArgsScratchOwner);
}

- (void)installNewRenderPassDescriptor
{
    /* new descriptor — invalidate FBO match cache. */
    [self clearFboMatchCache];
    mglRenderCppDestroyRenderPassStateOwner(
        &_state.renderPassStateOwner);
    if (mglRenderCppCreateDefaultRenderPassStateOwner(
            &_state.renderPassStateOwner) != 0) {
        _state.renderPassStateOwner = NULL;
    }
    _state.renderPassDescriptor = nil;
    mglRenderPassManagerSyncRuntimeOwners(&_state);
}

- (void)setFboMatchCacheResult:(BOOL)result
                       fboName:(GLuint)fboName
                     generation:(uint64_t)generation
{
    if (_state.renderPassIdentityOwner && fboName != 0u) {
        MGLRenderCppFboMatchCacheState cache = {
            .fbo_name = fboName,
            .generation = generation,
            .result = result,
        };
        mglRenderCppSetFboMatchCache(
            _state.renderPassIdentityOwner, &cache);
    }
}

- (void)clearFboMatchCache
{
    mglRenderCppClearFboMatchCache(_state.renderPassIdentityOwner);
}

- (void)setTraceReplayFlushId:(uint64_t)flushId batchIndex:(uint32_t)batchIndex
{
    _state.traceReplayFlushId = flushId;
    _state.traceReplayBatchIndex = batchIndex;
}

- (void)setTransientDepthTexture:(nullable MGLMetalTextureRef)texture
                           width:(NSUInteger)width
                          height:(NSUInteger)height
{
    _state.transientDepthTexture = texture;
    _state.transientDepthTextureWidth = width;
    _state.transientDepthTextureHeight = height;
}

- (void)setFallbackRenderTargetTexture:(nullable MGLMetalTextureRef)texture
{
    _state.fallbackRenderTargetTexture = texture;
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
    _state.renderPassDescriptor = nil;
    mglRenderCppDestroyRenderPassStateOwner(
        &_state.renderPassStateOwner);
    mglRenderPassManagerSyncRuntimeOwners(&_state);
    [self clearCurrentRenderEncoder];
    [self discardCurrentCommandBuffer];
    mglRenderCppDestroyMDIScratchOwner(&_state.mdiArgsScratchOwner);
    [self releaseDetachedCommandBufferIfOwned:nil];
    [self endCommandBufferCommit];
    mglRenderCppDestroyCommandBufferOwner(
        &_state.currentCommandBufferOwner);
    [self clearRenderPassIdentity];
    mglRenderCppDestroyRenderPassIdentityOwner(
        &_state.renderPassIdentityOwner);

    /* P4.5 (item 1141): sync tracking list lives inside the C++ owner;
     * the owner destructor frees it. */
    _state.fallbackRenderTargetTexture = nil;
    _state.transientDepthTexture = nil;
    mglRenderCppDestroyPendingEventOwner(&_state.pendingEventOwner);
    _state.currentDrawUsesRTSampledCopy = NO;
}

@end
