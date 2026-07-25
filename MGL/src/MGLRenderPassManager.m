#import "MGLRenderPassManager.h"

#import "mgl_draw_buffer.h"

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
    _state.renderPassFramebuffer = activeState ? activeState->framebuffer : NULL;
    _state.renderPassFramebufferName = _state.renderPassFramebuffer
        ? _state.renderPassFramebuffer->name
        : 0u;
    _state.renderPassDrawBuffer = activeState ? activeState->draw_buffer : 0u;
    _state.renderPassDrawBufferCount = context ? mglMetalDrawBufferCount(context) : 0;
    for (int index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        _state.renderPassDrawBuffers[index] =
            context && index < _state.renderPassDrawBufferCount
                ? mglMetalDrawBufferAt(context, (GLuint)index)
                : GL_NONE;
    }
}

- (void)clearRenderPassIdentity
{
    /* render pass ended — invalidate FBO match cache. */
    [self clearFboMatchCache];
    _state.renderPassFramebuffer = NULL;
    _state.renderPassFramebufferName = 0u;
    _state.renderPassDrawBuffer = 0u;
    _state.renderPassDrawBufferCount = 0;
    for (int index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        _state.renderPassDrawBuffers[index] = GL_NONE;
    }
}

- (id<MTLCommandBuffer>)installNewCommandBufferFromQueue:(id<MTLCommandQueue>)commandQueue
{
    _state.currentCommandBuffer = commandQueue ? [commandQueue commandBuffer] : nil;
    [self resetMDIScratch];
    return _state.currentCommandBuffer;
}

- (id<MTLCommandBuffer>)detachCurrentCommandBufferForSubmission
{
    id<MTLCommandBuffer> commandBuffer = _state.currentCommandBuffer;
    _state.currentCommandBuffer = nil;
    return commandBuffer;
}

- (void)discardCurrentCommandBuffer
{
    _state.currentCommandBuffer = nil;
    [self resetMDIScratch];
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
        _state.currentEvent = [device newEvent];
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
    _state.currentRenderEncoder = renderEncoder;
}

- (void)clearCurrentRenderEncoder
{
    /* encoder ended — invalidate FBO match cache. */
    [self clearFboMatchCache];
    _state.currentRenderEncoder = nil;
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

        id<MTLBuffer> newBuffer = [device newBufferWithLength:newCapacity
                                                      options:MTLResourceStorageModeShared];
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
    _state.mdiArgsScratchBuffer = nil;
    _state.mdiArgsScratchCapacity = 0;
    _state.mdiArgsScratchOffset = 0;
}

- (void)installNewRenderPassDescriptor
{
    /* new descriptor — invalidate FBO match cache. */
    [self clearFboMatchCache];
    _state.renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
}

- (void)setFboMatchCacheResult:(BOOL)result
                       fboName:(GLuint)fboName
                     generation:(uint64_t)generation
{
    _state.lastFboMatchFboName = fboName;
    _state.lastFboMatchFboGeneration = generation;
    _state.lastFboMatchResult = result;
}

- (void)clearFboMatchCache
{
    _state.lastFboMatchFboName = 0u;
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
    [self clearCurrentRenderEncoder];
    [self discardCurrentCommandBuffer];
    [self clearRenderPassIdentity];

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
