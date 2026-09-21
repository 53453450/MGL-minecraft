/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_QUERY_H
#define MGL_RENDER_API_QUERY_H

/* Declarations for the query slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Renderer initialization state as a C ABI value, never a borrowed object. */
int mglRenderIsInitialized(void);

/* Query device capabilities through Metal-cpp and return a pure value-state
 * snapshot. The device pointer is borrowed for the duration of the call. */
int mglRenderQueryCapability(void *device,
                                MGLRenderCapabilityState *state_out);

void mglRenderGetSync(GLMContext glm_ctx, Sync *sync);

unsigned int mglRenderGetSyncStatus(GLMContext glm_ctx, Sync *sync);

uint64_t mglRenderGetGPUTimestamp(GLMContext glm_ctx);

void mglRenderBeginTimerQueryCallback(GLMContext glm_ctx);

uint64_t mglRenderEndTimerQueryCallback(GLMContext glm_ctx);

void mglRenderBeginSampleQueryCallback(GLMContext glm_ctx,
                                          unsigned int target);

uint64_t mglRenderEndSampleQueryCallback(GLMContext glm_ctx);

int mglRenderIsSmallRGBA8(uint32_t width, uint32_t height, uint32_t internalformat);

int mglRenderIsValidGLCompareFunction(uint32_t func);

int mglRenderIsValidGLBlendEquation(uint32_t op);

int mglRenderIsValidGLBlendFactor(uint32_t factor);

/* Return stable device identity data for platform-neutral cache naming. */
int mglRenderGetDeviceIdentity(const void *device,
                                  uint64_t *registry_id_out,
                                  char *name_out,
                                  size_t name_capacity);

int mglRenderGetOrCreateDepthStencilState(MGLPipelineCacheOwner *owner, const MGLRenderDepthStencilDescriptorState *descriptor, void **depth_stencil_state_out, int *created_out);

int mglRenderCreateQueryStateOwner(uint32_t visibility_slot_count, MGLQueryStateOwner **owner_out);

int mglRenderBeginSampleQuery(MGLQueryStateOwner *owner, uint32_t counting, const char *buffer_label, void **visibility_buffer_out);

void mglRenderEndSampleQuery(MGLQueryStateOwner *owner);

int mglRenderIsSampleQueryActive(MGLQueryStateOwner *owner, uint32_t *active_out);

int mglRenderAcquireSampleQuerySlot(MGLQueryStateOwner *owner, uint32_t *mode_out, uint64_t *offset_out);

int mglRenderGetSampleQueryResult(MGLQueryStateOwner *owner, uint64_t *result_out);

int mglRenderBeginTimerQuery(MGLQueryStateOwner *owner);

int mglRenderEndTimerQuery(MGLQueryStateOwner *owner, uint64_t *elapsed_out);

void mglRenderDestroyQueryStateOwner(MGLQueryStateOwner **owner);

/* Validate and encode a complete compute plan, then (when copy-backs or CPU
 * visibility require a boundary) encode the copy-back blit and perform the
 * owner submit/wait transaction before synchronizing GL CPU prefixes. */
int mglRenderExecuteComputeExecutionPlan(MGLCommandBufferOwner *command_buffer_owner, MGLCommandBufferRecoveryOwner *recovery_owner, const MGLRenderComputeExecutionPlan *plan, const MGLRenderCopyBackEntry *copy_backs, uint32_t copy_back_count, uint32_t require_cpu_visibility, MGLRenderComputeExecutionResult *result, char *err, size_t errcap);

int mglRenderIsEmulatedMSColorTexture(uint32_t target, int32_t samples);

int mglRenderPlanDirtyDomains(uint32_t dirty_bits, int draw_command,
                              int has_pipeline, int fbo_binding_dirty,
                              MGLDirtyDomainPlan *out);

int mglRenderGetFboMatchCache(MGLRenderPassIdentityOwner *owner, MGLRenderFboMatchCacheState *cache_out);

int mglRenderGetRenderTargetSizeOwner(MGLRenderPassStateOwner *owner, uint64_t *width_out, uint64_t *height_out);

#ifdef __cplusplus
}
#endif

#endif
