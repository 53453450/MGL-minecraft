/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#ifndef MGLRenderPassManager_h
#define MGLRenderPassManager_h

#import <Foundation/Foundation.h>

#include "glm_context.h"
#include "mgl_render_cpp.h"

typedef struct MGLCommandState_t {
    void *_Nullable renderPassIdentityOwner;
    void *_Nullable renderPassStateOwner;
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
    BOOL currentDrawUsesRTSampledCopy;
    void *_Nullable pendingEventOwner;
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
    GLMContext _Nullable runtimeContext;
} MGLCommandState;

NS_ASSUME_NONNULL_BEGIN

@interface MGLRenderPassManager : NSObject {
@private
    MGLCommandState _state;
}

@property(nonatomic, readonly) const MGLCommandState *state;

- (void)setRuntimeContext:(nullable GLMContext)context;
- (void)updateRenderPassIdentityForContext:(GLMContext)context;
- (void)clearRenderPassIdentity;
- (void * _Nullable)installNewCommandBufferFromQueue:(void * _Nullable)commandQueue;
- (void * _Nullable)detachCurrentCommandBufferForSubmission;
- (void)discardCurrentCommandBuffer;
- (BOOL)commitDetachedCommandBufferIfOwned:(void * _Nullable)commandBuffer;
- (int)commitCommandBufferTransaction:(void * _Nullable)commandBuffer
                         recoveryOwner:(nullable void *)recoveryOwner
                     waitForCompletion:(BOOL)waitForCompletion
                                result:(MGLRenderCppCommandBufferTransaction *)result;
- (BOOL)hasLastSubmittedCommandBuffer;
- (int)waitForLastSubmittedCommandBuffer:(MGLRenderCppCommandBufferState *)state;
- (void * _Nullable)consumeTransactionCreatedCurrentCommandBuffer;
- (void)releaseDetachedCommandBufferIfOwned:(void * _Nullable)commandBuffer;
- (BOOL)appendSyncToCurrentCommandBuffer:(Sync *)sync;
- (void)clearCurrentCommandBufferSyncListEntries;
- (void * _Nullable)preparePendingEventWithDevice:(void * _Nullable)device
                                             syncName:(GLsizei)syncName;
- (void * _Nullable)detachPendingEventWithSyncName:(nullable GLuint *)syncNameOut;
- (void)clearPendingEvent;
- (void)installRenderEncoder:(void * _Nullable)renderEncoder;
- (void * _Nullable)createRenderEncoder;
- (void)endCurrentRenderEncoder;
- (void)clearCurrentRenderEncoder;
- (BOOL)beginCommandBufferCommit;
- (void)endCommandBufferCommit;
- (void * _Nullable)mdiArgumentScratchBufferWithDevice:(void * _Nullable)device
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
- (void)setCurrentDrawUsesRTSampledCopy:(BOOL)usesRTSampledCopy;
- (void)setDontCareFrameGeneration:(GLuint)generation;
- (void)incrementDontCareFrameGenerationWithWrap;
- (void)shutdown;

@end

NS_ASSUME_NONNULL_END

#endif /* MGLRenderPassManager_h */
