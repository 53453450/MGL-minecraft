#ifndef MGLRenderPassManager_h
#define MGLRenderPassManager_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "glm_context.h"

typedef struct SyncList_t {
    GLuint count;
    GLuint size;
    Sync * _Nullable * _Nullable list;
} SyncList;

typedef struct MGLCommandState_t {
    MTLRenderPassDescriptor *__strong _Nullable renderPassDescriptor;
    Framebuffer *_Nullable renderPassFramebuffer;
    GLuint renderPassFramebufferName;
    GLenum renderPassDrawBuffer;
    GLsizei renderPassDrawBufferCount;
    GLenum renderPassDrawBuffers[MAX_COLOR_ATTACHMENTS];
    uint64_t traceReplayFlushId;
    uint32_t traceReplayBatchIndex;
    GLuint dontCareFrameGeneration;
    id<MTLCommandBuffer> __strong _Nullable currentCommandBuffer;
    SyncList *_Nullable currentCommandBufferSyncList;
    id<MTLBuffer> __strong _Nullable mdiArgsScratchBuffer;
    NSUInteger mdiArgsScratchCapacity;
    NSUInteger mdiArgsScratchOffset;
    id<MTLRenderCommandEncoder> __strong _Nullable currentRenderEncoder;
    BOOL parallelEncodeActive;
    id<MTLTexture> __strong _Nullable fallbackRenderTargetTexture;
    id<MTLTexture> __strong _Nullable transientDepthTexture;
    NSUInteger transientDepthTextureWidth;
    NSUInteger transientDepthTextureHeight;
    BOOL currentDrawUsesRTSampledCopy;
    GLuint blitOperationComplete;
    id<MTLEvent> __strong _Nullable currentEvent;
    GLsizei currentSyncName;
    BOOL isCommittingCommandBuffer;
} MGLCommandState;

NS_ASSUME_NONNULL_BEGIN

@interface MGLRenderPassManager : NSObject {
@private
    MGLCommandState _state;
}

@property(nonatomic, readonly) MGLCommandState *state;

- (void)updateRenderPassIdentityForContext:(GLMContext)context;
- (void)clearRenderPassIdentity;
- (nullable id<MTLCommandBuffer>)installNewCommandBufferFromQueue:(nullable id<MTLCommandQueue>)commandQueue;
- (nullable id<MTLCommandBuffer>)detachCurrentCommandBufferForSubmission;
- (void)discardCurrentCommandBuffer;
- (BOOL)appendSyncToCurrentCommandBuffer:(Sync *)sync;
- (void)clearCurrentCommandBufferSyncListEntries;
- (nullable id<MTLEvent>)preparePendingEventWithDevice:(id<MTLDevice>)device
                                             syncName:(GLsizei)syncName;
- (nullable id<MTLEvent>)detachPendingEventWithSyncName:(nullable GLuint *)syncNameOut;
- (void)clearPendingEvent;
- (void)installRenderEncoder:(nullable id<MTLRenderCommandEncoder>)renderEncoder;
- (void)clearCurrentRenderEncoder;
- (void)beginParallelEncoding;
- (void)endParallelEncoding;
- (BOOL)isParallelEncodingActive;
- (BOOL)beginCommandBufferCommit;
- (void)endCommandBufferCommit;
- (nullable id<MTLBuffer>)mdiArgumentScratchBufferWithDevice:(id<MTLDevice>)device
                                                      length:(NSUInteger)length
                                                      offset:(nullable NSUInteger *)offsetOut;
- (void)resetMDIScratch;
- (void)shutdown;

@end

NS_ASSUME_NONNULL_END

#endif /* MGLRenderPassManager_h */
