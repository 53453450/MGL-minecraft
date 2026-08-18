/*
 * mgl_capability.m
 * MGL
 *
 * Implementation of the AGX Capability Layer.
 */

#include "mgl_capability.h"
#include "mgl_render_cpp.h"
#import <string.h>

void MGLCapabilityInit(MGLCapability *cap, void *deviceRef)
{
    if (!cap) return;
    memset(cap, 0, sizeof(*cap));
    cap->device = deviceRef;
    MGLRenderCppCapabilityState state = {0};
    if (mglRenderCppQueryCapability(deviceRef, &state) != 0) return;
    cap->family = (MGLGPUFamily)state.family;
    cap->isVirtualized = state.is_virtualized != 0;
    cap->supports8xMSAA = state.supports8x_msaa != 0;
    cap->maxSampleCount = state.max_sample_count;
    cap->maxTextureDimensions = state.max_texture_dimensions;
    cap->bug_3dGetBytesSliceOOB = state.bug_3d_getbytes_slice_oob != 0;
    cap->bug_3dReplaceRegionNonZeroOrigin =
        state.bug_3d_replace_region_nonzero_origin != 0;
    cap->bug_3dCopyFromBufferSliceOOB =
        state.bug_3d_copy_from_buffer_slice_oob != 0;
    cap->bug_asyncShaderCompileInVM = state.bug_async_shader_compile_in_vm != 0;
    cap->bug_mslPipelineRejection = state.bug_msl_pipeline_rejection != 0;
    cap->commandBufferRecoveryLimit = state.command_buffer_recovery_limit;
    cap->maxConcurrentCommandBuffers = state.max_concurrent_command_buffers;
    cap->textureAlignmentBytes = state.texture_alignment_bytes;
    cap->conservativeCPUCacheMode = state.conservative_cpu_cache_mode != 0;
}

bool MGLCapabilitySupportsSampleCount(MGLCapability *cap, uint64_t samples)
{
    if (!cap || samples <= 1) return true;
    return samples <= cap->maxSampleCount;
}

uint64_t MGLCapabilityClampSampleCount(MGLCapability *cap, uint64_t requested)
{
    if (!cap) return 1;
    if (requested <= 1) return 1;
    static const uint64_t candidates[] = { 32u, 16u, 8u, 4u, 2u };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i)
        if (candidates[i] <= requested && candidates[i] <= cap->maxSampleCount)
            return candidates[i];
    return 1;
}

uint64_t MGLCapabilityTextureAlignment(MGLCapability *cap)
{
    return cap ? cap->textureAlignmentBytes : 256;
}

bool MGLCapabilityUseConservativeCPUCache(MGLCapability *cap)
{
    return cap ? cap->conservativeCPUCacheMode : false;
}

uint64_t MGLCapabilityMaxConcurrentCommandBuffers(MGLCapability *cap)
{
    return cap ? cap->maxConcurrentCommandBuffers : 64;
}

bool MGLCapabilityHasBug(MGLCapability *cap, const char *bugName)
{
    if (!cap || !bugName) return false;

    if (strcmp(bugName, MGL_BUG_3D_GETBYTES_SLICE_OOB) == 0) {
        return cap->bug_3dGetBytesSliceOOB;
    }
    if (strcmp(bugName, MGL_BUG_3D_REPLACE_REGION_NONZERO_ORIGIN) == 0) {
        return cap->bug_3dReplaceRegionNonZeroOrigin;
    }
    if (strcmp(bugName, MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB) == 0) {
        return cap->bug_3dCopyFromBufferSliceOOB;
    }
    if (strcmp(bugName, MGL_BUG_ASYNC_SHADER_COMPILE_IN_VM) == 0) {
        return cap->bug_asyncShaderCompileInVM;
    }
    if (strcmp(bugName, MGL_BUG_MSL_PIPELINE_REJECTION) == 0) {
        return cap->bug_mslPipelineRejection;
    }
    return false;
}
