/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_manager.h — the render-pass manager, formerly the
 * Objective-C class MGLRenderPassManager (P0-1, log 106).
 *
 * The class was one ivar (`MGLCommandState`) plus 28 methods that all forwarded
 * to the mglRender* facade, so it becomes a C struct with the same methods as
 * functions.  `state` stays a POINTER member so the ~270 existing
 * `manager.state->field` reads in the Objective-C files keep their shape
 * (`manager->state->field`), and so passing the state to the C helpers that take
 * `const MGLCommandState *` needs no `&`.
 */

#ifndef MGL_RENDER_PASS_MANAGER_H
#define MGL_RENDER_PASS_MANAGER_H

#include "glm_context.h"
#include "mgl_command_state.h"   /* MGLCommandState */
#include "mgl_render.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLRenderPassManager_t {
    /* The command state every C driver reads; points at command_state below. */
    MGLCommandState *state;
    MGLCommandState command_state;
} MGLRenderPassManager;

/* Allocates the manager and clears its render-pass identity.  NULL when the
 * allocation fails; destroy it with mglPassManagerDestroy (which shuts it
 * down first). */
MGLRenderPassManager *mglPassManagerCreate(void);
void mglPassManagerDestroy(MGLRenderPassManager *manager);

void mglPassManagerSetRuntimeContext(MGLRenderPassManager *manager,
                                     GLMContext context);
void mglPassManagerUpdateRenderPassIdentityForContext(
    MGLRenderPassManager *manager, GLMContext context);
void mglPassManagerClearRenderPassIdentity(MGLRenderPassManager *manager);
void *mglPassManagerInstallNewCommandBufferFromQueue(
    MGLRenderPassManager *manager, void *command_queue);
void *mglPassManagerDetachCurrentCommandBufferForSubmission(
    MGLRenderPassManager *manager);
void mglPassManagerDiscardCurrentCommandBuffer(MGLRenderPassManager *manager);
int mglPassManagerCommitCommandBufferTransaction(MGLRenderPassManager *manager, void *command_buffer, MGLCommandBufferRecoveryOwner *recovery_owner, int wait_for_completion, MGLRenderCommandBufferTransaction *result);
int mglPassManagerHasLastSubmittedCommandBuffer(MGLRenderPassManager *manager);
int mglPassManagerWaitForLastSubmittedCommandBuffer(
    MGLRenderPassManager *manager, MGLRenderCommandBufferState *state);
void *mglPassManagerConsumeTransactionCreatedCurrentCommandBuffer(
    MGLRenderPassManager *manager);
void mglPassManagerReleaseDetachedCommandBufferIfOwned(
    MGLRenderPassManager *manager, void *command_buffer);
void mglPassManagerClearCurrentCommandBufferSyncListEntries(
    MGLRenderPassManager *manager);
void *mglPassManagerDetachPendingEventWithSyncName(MGLRenderPassManager *manager,
                                                   uint32_t *sync_name_out);
void mglPassManagerInstallRenderEncoder(MGLRenderPassManager *manager,
                                        void *render_encoder);
void *mglPassManagerCreateRenderEncoder(MGLRenderPassManager *manager);
void mglPassManagerEndCurrentRenderEncoder(MGLRenderPassManager *manager);
void mglPassManagerClearCurrentRenderEncoder(MGLRenderPassManager *manager);
void mglPassManagerEndCommandBufferCommit(MGLRenderPassManager *manager);
void mglPassManagerResetMDIScratch(MGLRenderPassManager *manager);
void mglPassManagerInstallNewRenderPassDescriptor(MGLRenderPassManager *manager);
void mglPassManagerSetFboMatchCacheResult(MGLRenderPassManager *manager,
                                          int result, uint32_t fbo_name,
                                          uint64_t generation);
void mglPassManagerClearFboMatchCache(MGLRenderPassManager *manager);
void mglPassManagerSetCurrentDrawUsesRTSampledCopy(MGLRenderPassManager *manager,
                                                   int uses_rt_sampled_copy);
void mglPassManagerSetDontCareFrameGeneration(MGLRenderPassManager *manager,
                                              uint32_t generation);
void mglPassManagerIncrementDontCareFrameGenerationWithWrap(
    MGLRenderPassManager *manager);
void mglPassManagerShutdown(MGLRenderPassManager *manager);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_PASS_MANAGER_H */
