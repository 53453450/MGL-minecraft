/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Incomplete types for mgl_render_api_*.h when included without the facade.
 * Full layouts stay in mgl_render.h / mgl_render_internal.h.
 */

#ifndef MGL_RENDER_FWD_H
#define MGL_RENDER_FWD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

typedef struct Buffer_t Buffer;
typedef struct Texture_t Texture;
typedef struct TextureLevel_t TextureLevel;
typedef struct TextureParameter_t TextureParameter;
typedef struct Program_t Program;
typedef struct __GLsync Sync;

typedef struct MGLBindingState MGLBindingState;
typedef struct MGLCommandBufferOwner MGLCommandBufferOwner;
typedef struct MGLCommandBufferRecoveryOwner MGLCommandBufferRecoveryOwner;
typedef struct MGLCommandQueueOwner MGLCommandQueueOwner;
typedef struct MGLCullDistanceIndexPlan MGLCullDistanceIndexPlan;
typedef struct MGLMDIScratchOwner MGLMDIScratchOwner;
typedef struct MGLPendingEventOwner MGLPendingEventOwner;
typedef struct MGLPipelineCacheOwner MGLPipelineCacheOwner;
typedef struct MGLQueryStateOwner MGLQueryStateOwner;
typedef struct MGLRenderEncoderOwner MGLRenderEncoderOwner;
typedef struct MGLRenderPassIdentityOwner MGLRenderPassIdentityOwner;
typedef struct MGLRenderPassStateOwner MGLRenderPassStateOwner;
typedef struct MGLTextureStagingOwner MGLTextureStagingOwner;

typedef struct MGLMetalAttachmentSubresource_t MGLMetalAttachmentSubresource;
typedef struct MGLCullDistanceEmuParams_t MGLCullDistanceEmuParams;
typedef struct MGLDirtyDomainPlan MGLDirtyDomainPlan;
typedef struct MGLRenderAttribFetchPlan MGLRenderAttribFetchPlan;
typedef struct MGLRenderBindingSnapshot_t MGLRenderBindingSnapshot;
typedef struct MGLRenderBindingStats MGLRenderBindingStats;
typedef struct MGLRenderBlitFramebufferPlan_t MGLRenderBlitFramebufferPlan;
typedef struct MGLRenderBlitScissorRect_t MGLRenderBlitScissorRect;
typedef struct MGLRenderBufferCopyEntry_t MGLRenderBufferCopyEntry;
typedef struct MGLRenderBufferInfo_t MGLRenderBufferInfo;
typedef struct MGLRenderCapabilityState_t MGLRenderCapabilityState;
typedef struct MGLRenderCommandBufferCommitDecision_t
    MGLRenderCommandBufferCommitDecision;
typedef struct MGLRenderCommandBufferCompletionDecision_t
    MGLRenderCommandBufferCompletionDecision;
typedef struct MGLRenderCommandBufferCompletionResult_t
    MGLRenderCommandBufferCompletionResult;
typedef struct MGLRenderCommandBufferState_t MGLRenderCommandBufferState;
typedef struct MGLRenderCommandBufferTransaction_t
    MGLRenderCommandBufferTransaction;
typedef struct MGLRenderCommandRecoverySkipDecision_t
    MGLRenderCommandRecoverySkipDecision;
typedef struct MGLRenderCommandRecoverySnapshot_t
    MGLRenderCommandRecoverySnapshot;
typedef struct MGLRenderCommandRecoverySuccess_t
    MGLRenderCommandRecoverySuccess;
typedef struct MGLRenderComputeBindingSnapshot_t
    MGLRenderComputeBindingSnapshot;
typedef struct MGLRenderComputeDispatchSetup_t MGLRenderComputeDispatchSetup;
typedef struct MGLRenderComputeExecutionPlan_t MGLRenderComputeExecutionPlan;
typedef struct MGLRenderComputeExecutionResult_t
    MGLRenderComputeExecutionResult;
typedef struct MGLRenderComputePlan_t MGLRenderComputePlan;
typedef struct MGLRenderCopyBackEntry_t MGLRenderCopyBackEntry;
typedef struct MGLRenderCullDistanceAttribPort
    MGLRenderCullDistanceAttribPort;
typedef struct MGLRenderCullDistanceLayout_t MGLRenderCullDistanceLayout;
typedef struct MGLRenderCullDistancePrimitive_t
    MGLRenderCullDistancePrimitive;
typedef struct MGLRenderDepthStencilDescriptorState_t
    MGLRenderDepthStencilDescriptorState;
typedef struct MGLRenderDrawPlan_t MGLRenderDrawPlan;
typedef struct MGLRenderFboMatchCacheState_t MGLRenderFboMatchCacheState;
typedef struct MGLRenderLevelUploadPrep_t MGLRenderLevelUploadPrep;
typedef struct MGLRenderPassAttachmentState_t MGLRenderPassAttachmentState;
typedef struct MGLRenderPassIdentityState_t MGLRenderPassIdentityState;
typedef struct MGLRenderPassState_t MGLRenderPassState;
typedef struct MGLRenderPipelineActiveState_t MGLRenderPipelineActiveState;
typedef struct MGLRenderPipelineBlendState_t MGLRenderPipelineBlendState;
typedef struct MGLRenderPipelineDescriptorState
    MGLRenderPipelineDescriptorState;
typedef struct MGLRenderReadTextureRegionClip_t
    MGLRenderReadTextureRegionClip;
typedef struct MGLRenderResourceBindingSnapshot_t
    MGLRenderResourceBindingSnapshot;
typedef struct MGLRenderScaledBlitUVs_t MGLRenderScaledBlitUVs;
typedef struct MGLRenderTextureDescriptorState_t
    MGLRenderTextureDescriptorState;
typedef struct MGLRenderTextureInfo_t MGLRenderTextureInfo;
typedef struct MGLRenderTextureTargetPlan_t MGLRenderTextureTargetPlan;
typedef struct MGLRenderVertexAttribResolve_t MGLRenderVertexAttribResolve;
typedef struct MGLRenderVertexConversion_t MGLRenderVertexConversion;

typedef void (*MGLRenderCommandBufferCompletion)(
    void *context, const MGLRenderCommandBufferState *state);
typedef void (*MGLRenderDestroyContext)(void *context);

#define MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN 0u
#define MGL_RENDER_TEXTURE_DATA_KIND_FLOAT 1u
#define MGL_RENDER_TEXTURE_DATA_KIND_SINT 2u
#define MGL_RENDER_TEXTURE_DATA_KIND_UINT 3u
#define MGL_RENDER_TEXTURE_DATA_KIND_DEPTH 4u

#endif /* MGL_RENDER_FWD_H */
