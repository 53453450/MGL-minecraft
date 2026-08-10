#import "MGLRenderPassManager.h"

#import "mgl_draw_buffer.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp_objc.h"

static BOOL mglRenderPassManagerUsesMetalCpp(void)
{
    return mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
           mglRenderCppGetDevice() != NULL;
}

static id<MTLEvent> mglRenderPassManagerCreateEvent(id<MTLDevice> device)
{
    if (mglRenderPassManagerUsesMetalCpp()) {
        void *event = NULL;
        if (mglRenderCppCreateEvent(&event) == 0 && event) {
            return (__bridge_transfer id<MTLEvent>)event;
        }
    }
    return [device newEvent];
}

static id<MTLBuffer> mglRenderPassManagerCreateBuffer(
    id<MTLDevice> device,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglRenderPassManagerUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
            buffer) {
            return (__bridge_transfer id<MTLBuffer>)buffer;
        }
    }
    return [device newBufferWithLength:length options:options];
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
    if (mglRenderPassManagerUsesMetalCpp()) {
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
    } else {
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

- (id<MTLCommandBuffer>)installNewCommandBufferFromQueue:(id<MTLCommandQueue>)commandQueue
{
    id<MTLCommandBuffer> commandBuffer = nil;
    if (mglRenderPassManagerUsesMetalCpp() && commandQueue) {
        commandBuffer = mglRenderCppCreateOrResetCommandBufferOwner(
            &_state.currentCommandBufferOwner, commandQueue);
    }
    if (!commandBuffer) {
        mglRenderCppDestroyCommandBufferOwner(
            &_state.currentCommandBufferOwner);
        commandBuffer = commandQueue ? [commandQueue commandBuffer] : nil;
    }
    _state.currentCommandBuffer = commandBuffer;
    [self resetMDIScratch];
    return _state.currentCommandBuffer;
}

- (id<MTLCommandBuffer>)detachCurrentCommandBufferForSubmission
{
    id<MTLCommandBuffer> commandBuffer = _state.currentCommandBuffer;
    if (commandBuffer && _state.currentCommandBufferOwner) {
        mglRenderCppDestroyCommandBufferSubmission(
            &_state.detachedCommandBufferSubmission);
        _state.detachedCommandBuffer = NULL;
        void *detachedBuffer = NULL;
        if (mglRenderCppTakeCommandBufferSubmission(
                _state.currentCommandBufferOwner,
                &_state.detachedCommandBufferSubmission,
                &detachedBuffer) == 0 && detachedBuffer) {
            commandBuffer = (__bridge id<MTLCommandBuffer>)detachedBuffer;
            _state.detachedCommandBuffer = detachedBuffer;
        } else {
            /* Preserve the old ARC handoff if allocation of the submission
             * handle fails. commandBuffer is already a strong local. */
            mglRenderCppDiscardCommandBufferOwnerCurrent(
                _state.currentCommandBufferOwner);
        }
    }
    _state.currentCommandBuffer = nil;
    return commandBuffer;
}

- (void)discardCurrentCommandBuffer
{
    _state.currentCommandBuffer = nil;
    mglRenderCppDiscardCommandBufferOwnerCurrent(
        _state.currentCommandBufferOwner);
    [self resetMDIScratch];
}

- (BOOL)commitDetachedCommandBufferIfOwned:(id<MTLCommandBuffer>)commandBuffer
{
    if (!commandBuffer || !_state.detachedCommandBufferSubmission ||
        _state.detachedCommandBuffer != (__bridge void *)commandBuffer) {
        return NO;
    }
    if (mglRenderCppCommitCommandBufferSubmission(
            &_state.detachedCommandBufferSubmission) != 0) {
        return NO;
    }
    _state.detachedCommandBuffer = NULL;
    return YES;
}

- (void)releaseDetachedCommandBufferIfOwned:(id<MTLCommandBuffer>)commandBuffer
{
    if (!_state.detachedCommandBufferSubmission ||
        (commandBuffer &&
         _state.detachedCommandBuffer != (__bridge void *)commandBuffer)) {
        return;
    }
    mglRenderCppDestroyCommandBufferSubmission(
        &_state.detachedCommandBufferSubmission);
    _state.detachedCommandBuffer = NULL;
}

- (BOOL)appendSyncToCurrentCommandBuffer:(Sync *)sync
{
    if (!sync) {
        return NO;
    }

    SyncList *syncList = _state.currentCommandBufferSyncList;
    if (!syncList) {
        syncList = (SyncList *)malloc(sizeof(SyncList));
        if (!syncList) {
            NSLog(@"MGL SECURITY ERROR: Failed to allocate SyncList");
            return NO;
        }
        syncList->count = 0;
        syncList->size = 8;
        syncList->list = (Sync **)malloc(sizeof(Sync *) * syncList->size);
        if (!syncList->list) {
            NSLog(@"MGL SECURITY ERROR: Failed to allocate SyncList array");
            free(syncList);
            return NO;
        }
        _state.currentCommandBufferSyncList = syncList;
    }

    if (syncList->count >= syncList->size) {
        size_t currentSize = (size_t)syncList->size;
        if (currentSize > SIZE_MAX / 2 / sizeof(Sync *)) {
            NSLog(@"MGL SECURITY ERROR: SyncList size would overflow, preventing expansion");
            return NO;
        }

        size_t newSize = currentSize * 2;
        Sync **newList = (Sync **)realloc(syncList->list, sizeof(Sync *) * newSize);
        if (!newList) {
            NSLog(@"MGL SECURITY ERROR: Failed to reallocate SyncList array");
            return NO;
        }
        syncList->size = (GLuint)newSize;
        syncList->list = newList;
    }

    syncList->list[syncList->count++] = sync;
    return YES;
}

- (void)clearCurrentCommandBufferSyncListEntries
{
    SyncList *syncList = _state.currentCommandBufferSyncList;
    if (!syncList) {
        return;
    }

    GLuint count = syncList->count;
    GLuint size = syncList->size;
    if (!syncList->list || size == 0) {
        NSLog(@"MGL WARNING: Sync list storage invalid (list=%p size=%u), resetting",
              syncList->list, size);
        syncList->count = 0;
        return;
    }

    if (count > size) {
        NSLog(@"MGL WARNING: Sync list count overflow (count=%u size=%u), clamping",
              count, size);
        count = size;
    }
    for (GLuint index = 0; index < count; index++) {
        syncList->list[index] = NULL;
    }
    syncList->count = 0;
}

- (id<MTLEvent>)preparePendingEventWithDevice:(id<MTLDevice>)device
                                     syncName:(GLsizei)syncName
{
    if (!_state.currentEvent) {
        _state.currentEvent = mglRenderPassManagerCreateEvent(device);
    }
    if (!_state.currentEvent) {
        return nil;
    }
    _state.currentSyncName = syncName;
    return _state.currentEvent;
}

- (id<MTLEvent>)detachPendingEventWithSyncName:(GLuint *)syncNameOut
{
    id<MTLEvent> event = _state.currentEvent;
    if (syncNameOut) {
        *syncNameOut = (GLuint)_state.currentSyncName;
    }
    _state.currentEvent = nil;
    _state.currentSyncName = 0;
    return event;
}

- (void)clearPendingEvent
{
    _state.currentEvent = nil;
    _state.currentSyncName = 0;
}

- (void)installRenderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder
{
    /* new encoder — invalidate FBO match cache. */
    [self clearFboMatchCache];
    if (mglRenderPassManagerUsesMetalCpp() && renderEncoder) {
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
    _state.currentRenderEncoder = renderEncoder;
}

- (id<MTLRenderCommandEncoder>)createRenderEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    id<MTLRenderCommandEncoder> renderEncoder = nil;
    if (mglRenderPassManagerUsesMetalCpp() &&
        _state.currentCommandBuffer && _state.renderPassStateOwner &&
        (!descriptor || descriptor == _state.renderPassDescriptor)) {
        void *encoderCPP = NULL;
        if (mglRenderCppCreateRenderEncoderFromStateOwner(
                (__bridge void *)_state.currentCommandBuffer,
                _state.renderPassStateOwner, &encoderCPP) == 0 &&
            encoderCPP) {
            renderEncoder = (__bridge id<MTLRenderCommandEncoder>)encoderCPP;
        }
    }
    if (!renderEncoder && _state.currentCommandBuffer && descriptor) {
        renderEncoder = [_state.currentCommandBuffer
            renderCommandEncoderWithDescriptor:descriptor];
    }
    return renderEncoder;
}

- (void)endCurrentRenderEncoder
{
    if (!_state.currentRenderEncoder) return;
    if (_state.currentRenderEncoderOwner &&
        mglRenderCppEndRenderEncoderOwner(
            _state.currentRenderEncoderOwner) == 0) {
        return;
    }
    if (mglRenderPassManagerUsesMetalCpp() &&
        mglRenderCppEndRenderEncoder(
            (__bridge void *)_state.currentRenderEncoder) == 0) {
        return;
    }
    [_state.currentRenderEncoder endEncoding];
}

- (void)clearCurrentRenderEncoder
{
    /* encoder ended — invalidate FBO match cache. */
    [self clearFboMatchCache];
    _state.currentRenderEncoder = nil;
    mglRenderCppDestroyRenderEncoderOwner(
        &_state.currentRenderEncoderOwner);
}

- (BOOL)beginCommandBufferCommit
{
    if (_state.isCommittingCommandBuffer) {
        return NO;
    }
    _state.isCommittingCommandBuffer = YES;
    return YES;
}

- (void)endCommandBufferCommit
{
    _state.isCommittingCommandBuffer = NO;
}

- (id<MTLBuffer>)mdiArgumentScratchBufferWithDevice:(id<MTLDevice>)device
                                              length:(NSUInteger)length
                                              offset:(NSUInteger *)offsetOut
{
    if (offsetOut) {
        *offsetOut = 0;
    }
    if (!device || !_state.currentCommandBuffer || length == 0) {
        return nil;
    }

    if (mglRenderPassManagerUsesMetalCpp()) {
        if (!_state.mdiArgsScratchOwner &&
            mglRenderCppCreateMDIScratchOwner(
                &_state.mdiArgsScratchOwner) != 0) {
            return nil;
        }
        void *buffer = NULL;
        uint64_t offset = 0;
        uint64_t capacity = 0;
        if (mglRenderCppAllocateMDIScratch(
                _state.mdiArgsScratchOwner, (uint64_t)length, 256u,
                &buffer, &offset, &capacity) != 0 || !buffer ||
            offset > NSUIntegerMax || capacity > NSUIntegerMax) {
            return nil;
        }
        _state.mdiArgsScratchBuffer = (__bridge id<MTLBuffer>)buffer;
        _state.mdiArgsScratchOffset = (NSUInteger)(offset + length);
        _state.mdiArgsScratchCapacity = (NSUInteger)capacity;
        if (offsetOut) *offsetOut = (NSUInteger)offset;
        return _state.mdiArgsScratchBuffer;
    }

    const NSUInteger alignment = 256u;
    NSUInteger alignedOffset =
        (_state.mdiArgsScratchOffset + (alignment - 1u)) & ~(alignment - 1u);
    if (alignedOffset < _state.mdiArgsScratchOffset ||
        length > NSUIntegerMax - alignedOffset) {
        return nil;
    }

    NSUInteger requiredBytes = alignedOffset + length;
    if (!_state.mdiArgsScratchBuffer || requiredBytes > _state.mdiArgsScratchCapacity) {
        NSUInteger newCapacity = _state.mdiArgsScratchCapacity
            ? _state.mdiArgsScratchCapacity * 2u
            : 64u * 1024u;
        if (newCapacity < length) {
            newCapacity = length;
        }
        if (newCapacity < requiredBytes) {
            newCapacity = requiredBytes;
        }
        if (newCapacity < _state.mdiArgsScratchCapacity) {
            return nil;
        }

        id<MTLBuffer> newBuffer = mglRenderPassManagerCreateBuffer(
            device, newCapacity, MTLResourceStorageModeShared);
        if (!newBuffer) {
            return nil;
        }
        _state.mdiArgsScratchBuffer = newBuffer;
        _state.mdiArgsScratchCapacity = newCapacity;
        alignedOffset = 0;
        requiredBytes = length;
    }

    _state.mdiArgsScratchOffset = requiredBytes;
    if (offsetOut) {
        *offsetOut = alignedOffset;
    }
    return _state.mdiArgsScratchBuffer;
}

- (void)resetMDIScratch
{
    mglRenderCppDestroyMDIScratchOwner(&_state.mdiArgsScratchOwner);
    _state.mdiArgsScratchBuffer = nil;
    _state.mdiArgsScratchCapacity = 0;
    _state.mdiArgsScratchOffset = 0;
}

- (void)installNewRenderPassDescriptor
{
    /* new descriptor — invalidate FBO match cache. */
    [self clearFboMatchCache];
    mglRenderCppDestroyRenderPassStateOwner(
        &_state.renderPassStateOwner);
    MTLRenderPassDescriptor *descriptor =
        [MTLRenderPassDescriptor renderPassDescriptor];
    _state.renderPassDescriptor = descriptor;
    if (mglRenderPassManagerUsesMetalCpp() && descriptor) {
        if (mglRenderCppCreateDefaultRenderPassStateOwner(
                &_state.renderPassStateOwner) != 0) {
            _state.renderPassStateOwner = NULL;
        }
    }
}

- (void)setFboMatchCacheResult:(BOOL)result
                       fboName:(GLuint)fboName
                     generation:(uint64_t)generation
{
    _state.lastFboMatchFboName = fboName;
    _state.lastFboMatchFboGeneration = generation;
    _state.lastFboMatchResult = result;
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
    _state.lastFboMatchFboName = 0u;
    mglRenderCppClearFboMatchCache(_state.renderPassIdentityOwner);
}

- (void)setTraceReplayFlushId:(uint64_t)flushId batchIndex:(uint32_t)batchIndex
{
    _state.traceReplayFlushId = flushId;
    _state.traceReplayBatchIndex = batchIndex;
}

- (void)setTransientDepthTexture:(nullable id<MTLTexture>)texture
                           width:(NSUInteger)width
                          height:(NSUInteger)height
{
    _state.transientDepthTexture = texture;
    _state.transientDepthTextureWidth = width;
    _state.transientDepthTextureHeight = height;
}

- (void)setFallbackRenderTargetTexture:(nullable id<MTLTexture>)texture
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
    [self clearCurrentRenderEncoder];
    [self discardCurrentCommandBuffer];
    mglRenderCppDestroyMDIScratchOwner(&_state.mdiArgsScratchOwner);
    [self releaseDetachedCommandBufferIfOwned:nil];
    mglRenderCppDestroyCommandBufferOwner(
        &_state.currentCommandBufferOwner);
    [self clearRenderPassIdentity];
    mglRenderCppDestroyRenderPassIdentityOwner(
        &_state.renderPassIdentityOwner);

    if (_state.currentCommandBufferSyncList) {
        free(_state.currentCommandBufferSyncList->list);
        free(_state.currentCommandBufferSyncList);
        _state.currentCommandBufferSyncList = NULL;
    }

    _state.fallbackRenderTargetTexture = nil;
    _state.transientDepthTexture = nil;
    [self clearPendingEvent];
    _state.currentDrawUsesRTSampledCopy = NO;
    [self endCommandBufferCommit];
}

@end
