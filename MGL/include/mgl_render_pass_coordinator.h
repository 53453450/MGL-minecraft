/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * C command/render-pass coordinator — replaces MGLRenderPassManager ObjC shell.
 */

#ifndef MGL_RENDER_PASS_COORDINATOR_H
#define MGL_RENDER_PASS_COORDINATOR_H

#include <stdbool.h>
#include <stdint.h>

#include "glm_context.h"
#include "mgl_render.h"

typedef struct MGLCommandState_t {
    void *renderPassIdentityOwner;
    void *renderPassStateOwner;
    Framebuffer *renderPassFramebuffer;
    GLuint renderPassFramebufferName;
    GLenum renderPassDrawBuffer;
    GLsizei renderPassDrawBufferCount;
    GLenum renderPassDrawBuffers[MAX_COLOR_ATTACHMENTS];
    uint64_t traceReplayFlushId;
    uint32_t traceReplayBatchIndex;
    GLuint dontCareFrameGeneration;
    void *currentCommandBufferOwner;
    void *detachedCommandBufferSubmission;
    void *mdiArgsScratchOwner;
    void *currentRenderEncoderOwner;
    bool currentDrawUsesRTSampledCopy;
    void *pendingEventOwner;
    GLuint lastFboMatchFboName;
    uint64_t lastFboMatchFboGeneration;
    bool lastFboMatchResult;
    GLMContext runtimeContext;
} MGLCommandState;

#ifdef __cplusplus
extern "C" {
#endif

void mglCmdInit(MGLCommandState *state);
void mglCmdSetRuntimeContext(MGLCommandState *state, GLMContext context);
void mglCmdUpdateRenderPassIdentityForContext(MGLCommandState *state,
                                             GLMContext context);
void mglCmdClearRenderPassIdentity(MGLCommandState *state);
void *mglCmdInstallNewCommandBufferFromQueue(MGLCommandState *state,
                                             void *commandQueue);
void *mglCmdDetachCurrentCommandBufferForSubmission(MGLCommandState *state);
void mglCmdDiscardCurrentCommandBuffer(MGLCommandState *state);
bool mglCmdCommitDetachedCommandBufferIfOwned(MGLCommandState *state,
                                             void *commandBuffer);
int mglCmdCommitCommandBufferTransaction(
    MGLCommandState *state, void *commandBuffer, void *recoveryOwner,
    bool waitForCompletion, MGLRenderCommandBufferTransaction *result);
bool mglCmdHasLastSubmittedCommandBuffer(const MGLCommandState *state);
int mglCmdWaitForLastSubmittedCommandBuffer(MGLCommandState *state,
                                            MGLRenderCommandBufferState *out);
void *mglCmdConsumeTransactionCreatedCurrentCommandBuffer(MGLCommandState *state);
void mglCmdReleaseDetachedCommandBufferIfOwned(MGLCommandState *state,
                                               void *commandBuffer);
bool mglCmdAppendSyncToCurrentCommandBuffer(MGLCommandState *state, Sync *sync);
void mglCmdClearCurrentCommandBufferSyncListEntries(MGLCommandState *state);
void *mglCmdPreparePendingEventWithDevice(MGLCommandState *state,
                                          void *device, GLsizei syncName);
void *mglCmdDetachPendingEventWithSyncName(MGLCommandState *state,
                                           GLuint *syncNameOut);
void mglCmdClearPendingEvent(MGLCommandState *state);
void mglCmdInstallRenderEncoder(MGLCommandState *state, void *renderEncoder);
void *mglCmdCreateRenderEncoder(MGLCommandState *state);
void mglCmdEndCurrentRenderEncoder(MGLCommandState *state);
void mglCmdClearCurrentRenderEncoder(MGLCommandState *state);
bool mglCmdBeginCommandBufferCommit(MGLCommandState *state);
void mglCmdEndCommandBufferCommit(MGLCommandState *state);
void *mglCmdMdiArgumentScratchBufferWithDevice(MGLCommandState *state,
                                               void *device, size_t length,
                                               size_t *offsetOut);
void mglCmdResetMdiScratch(MGLCommandState *state);
void mglCmdInstallNewRenderPassDescriptor(MGLCommandState *state);
void mglCmdSetFboMatchCacheResult(MGLCommandState *state, bool result,
                                  GLuint fboName, uint64_t generation);
void mglCmdClearFboMatchCache(MGLCommandState *state);
void mglCmdSetTraceReplayFlushId(MGLCommandState *state, uint64_t flushId,
                                 uint32_t batchIndex);
void mglCmdSetCurrentDrawUsesRTSampledCopy(MGLCommandState *state,
                                           bool usesRTSampledCopy);
void mglCmdSetDontCareFrameGeneration(MGLCommandState *state, GLuint generation);
void mglCmdIncrementDontCareFrameGenerationWithWrap(MGLCommandState *state);
void mglCmdShutdown(MGLCommandState *state);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_PASS_COORDINATOR_H */
