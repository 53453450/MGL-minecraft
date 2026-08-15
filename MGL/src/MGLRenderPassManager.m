#import "MGLRenderPassManager.h"

#import "mgl_draw_buffer.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp_objc.h"

static BOOL mglRenderPassManagerUsesMetalCpp(void)
{
    return mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
           mglRenderCppGetDevice() != NULL;
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

- (id<MTLEvent>)preparePendingEventWithDevice:(__unused id<MTLDevice>)device
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
    return (__bridge id<MTLEvent>)event;
}

- (id<MTLEvent>)detachPendingEventWithSyncName:(GLuint *)syncNameOut
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
    return (__bridge_transfer id<MTLEvent>)event;
}

- (void)clearPendingEvent
{
    /* P4.5 (item 1141): discard the pending event; the owner stays. */
    if (_state.pendingEventOwner) {
        mglRenderCppPendingEventClear(_state.pendingEventOwner);
    }
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
    if (mglRenderPassManagerUsesMetalCpp()) {
        /* P4.1f: gate-on 下 C++ render pass state owner 是唯一权威 —— 不再
         * 创建 ObjC MTLRenderPassDescriptor（encoder 从 state owner 创建）。
         * renderPassDescriptor 保持 nil；owner-first 读取 helper 在 owner
         * 可用时不触达镜像。 */
        if (mglRenderCppCreateDefaultRenderPassStateOwner(
                &_state.renderPassStateOwner) != 0) {
            _state.renderPassStateOwner = NULL;
        }
        _state.renderPassDescriptor = nil;
        return;
    }
    MTLRenderPassDescriptor *descriptor =
        [MTLRenderPassDescriptor renderPassDescriptor];
    _state.renderPassDescriptor = descriptor;
}

- (void)setFboMatchCacheResult:(BOOL)result
                       fboName:(GLuint)fboName
                     generation:(uint64_t)generation
{
    /* P4.1f: gate-on 下 C++ identity owner 是 FBO-match 缓存的唯一权威，
     * 镜像不再写（gate-off 的 A/B 基线仍用镜像）。 */
    if (mglRenderPassManagerUsesMetalCpp()) {
        if (_state.renderPassIdentityOwner && fboName != 0u) {
            MGLRenderCppFboMatchCacheState cache = {
                .fbo_name = fboName,
                .generation = generation,
                .result = result,
            };
            mglRenderCppSetFboMatchCache(
                _state.renderPassIdentityOwner, &cache);
        }
        return;
    }
    _state.lastFboMatchFboName = fboName;
    _state.lastFboMatchFboGeneration = generation;
    _state.lastFboMatchResult = result;
}

- (void)clearFboMatchCache
{
    if (mglRenderPassManagerUsesMetalCpp()) {
        mglRenderCppClearFboMatchCache(_state.renderPassIdentityOwner);
        return;
    }
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

    /* P4.5 (item 1141): sync tracking list lives inside the C++ owner;
     * the owner destructor frees it. */
    _state.fallbackRenderTargetTexture = nil;
    _state.transientDepthTexture = nil;
    mglRenderCppDestroyPendingEventOwner(&_state.pendingEventOwner);
    _state.currentDrawUsesRTSampledCopy = NO;
    [self endCommandBufferCommit];
}

@end
