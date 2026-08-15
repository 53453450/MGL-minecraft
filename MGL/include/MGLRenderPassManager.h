#ifndef MGLRenderPassManager_h
#define MGLRenderPassManager_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "glm_context.h"
#include "mgl_render_cpp_objc.h"

typedef struct MGLCommandState_t {
    void *_Nullable renderPassIdentityOwner;
    void *_Nullable renderPassStateOwner;
    MTLRenderPassDescriptor *__strong _Nullable renderPassDescriptor;
    Framebuffer *_Nullable renderPassFramebuffer;
    GLuint renderPassFramebufferName;
    GLenum renderPassDrawBuffer;
    GLsizei renderPassDrawBufferCount;
    GLenum renderPassDrawBuffers[MAX_COLOR_ATTACHMENTS];
    uint64_t traceReplayFlushId;
    uint32_t traceReplayBatchIndex;
    GLuint dontCareFrameGeneration;
    void *_Nullable currentCommandBufferOwner;
    void *_Nullable detachedCommandBufferSubmission;
    void *_Nullable mdiArgsScratchOwner;
    void *_Nullable currentRenderEncoderOwner;
    MGLMetalTextureRef __strong _Nullable fallbackRenderTargetTexture;
    MGLMetalTextureRef __strong _Nullable transientDepthTexture;
    NSUInteger transientDepthTextureWidth;
    NSUInteger transientDepthTextureHeight;
    BOOL currentDrawUsesRTSampledCopy;
    GLuint blitOperationComplete;
    void *_Nullable pendingEventOwner;
    BOOL isCommittingCommandBuffer;
    /* Cache for currentRenderPassMatchesCurrentFramebuffer.
     * lastFboMatchFboName == 0 means "invalid cache, recompute".
     * Valid only for non-default FBOs (fbo != NULL && fboName != 0);
     * the default-framebuffer path is never cached because its inputs
     * (drawable, depth/stencil caps, _drawBuffers) change independently
     * of fbo_attachment_generation.
     * Invalidated on encoder install/clear, descriptor install, and
     * render-pass identity update/clear — all signals that the render
     * pass configuration may have changed. */
    GLuint lastFboMatchFboName;
    uint64_t lastFboMatchFboGeneration;
    BOOL lastFboMatchResult;
} MGLCommandState;

NS_ASSUME_NONNULL_BEGIN

@interface MGLRenderPassManager : NSObject {
@private
    MGLCommandState _state;
}

@property(nonatomic, readonly) const MGLCommandState *state;

- (void)updateRenderPassIdentityForContext:(GLMContext)context;
- (void)clearRenderPassIdentity;
- (nullable MGLMetalCommandBufferRef)installNewCommandBufferFromQueue:(nullable MGLMetalCommandQueueRef)commandQueue;
- (nullable MGLMetalCommandBufferRef)detachCurrentCommandBufferForSubmission;
- (void)discardCurrentCommandBuffer;
- (BOOL)commitDetachedCommandBufferIfOwned:(nullable MGLMetalCommandBufferRef)commandBuffer;
- (void)releaseDetachedCommandBufferIfOwned:(nullable MGLMetalCommandBufferRef)commandBuffer;
- (BOOL)appendSyncToCurrentCommandBuffer:(Sync *)sync;
- (void)clearCurrentCommandBufferSyncListEntries;
- (nullable MGLMetalEventRef)preparePendingEventWithDevice:(MGLMetalDeviceRef)device
                                             syncName:(GLsizei)syncName;
- (nullable MGLMetalEventRef)detachPendingEventWithSyncName:(nullable GLuint *)syncNameOut;
- (void)clearPendingEvent;
- (void)installRenderEncoder:(nullable MGLMetalRenderCommandEncoderRef)renderEncoder;
- (nullable MGLMetalRenderCommandEncoderRef)createRenderEncoderWithDescriptor:(nullable MTLRenderPassDescriptor *)descriptor;
- (void)endCurrentRenderEncoder;
- (void)clearCurrentRenderEncoder;
- (BOOL)beginCommandBufferCommit;
- (void)endCommandBufferCommit;
- (nullable MGLMetalBufferRef)mdiArgumentScratchBufferWithDevice:(MGLMetalDeviceRef)device
                                                      length:(NSUInteger)length
                                                      offset:(nullable NSUInteger *)offsetOut;
- (void)resetMDIScratch;
- (void)installNewRenderPassDescriptor;
/* Store/clear the FBO-match cache used by
 * currentRenderPassMatchesCurrentFramebuffer.  Pass fboName=0 to
 * invalidate (equivalent to clearFboMatchCache). */
- (void)setFboMatchCacheResult:(BOOL)result
                       fboName:(GLuint)fboName
                     generation:(uint64_t)generation;
- (void)clearFboMatchCache;
- (void)setTraceReplayFlushId:(uint64_t)flushId batchIndex:(uint32_t)batchIndex;
- (void)setTransientDepthTexture:(nullable MGLMetalTextureRef)texture
                           width:(NSUInteger)width
                          height:(NSUInteger)height;
- (void)setFallbackRenderTargetTexture:(nullable MGLMetalTextureRef)texture;
- (void)setCurrentDrawUsesRTSampledCopy:(BOOL)usesRTSampledCopy;
- (void)setDontCareFrameGeneration:(GLuint)generation;
- (void)incrementDontCareFrameGenerationWithWrap;
- (void)shutdown;

@end

NS_ASSUME_NONNULL_END

#endif /* MGLRenderPassManager_h */
