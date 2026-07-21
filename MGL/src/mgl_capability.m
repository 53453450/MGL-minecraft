/*
 * mgl_capability.m
 * MGL
 *
 * Implementation of the AGX Capability Layer.
 */

#import "mgl_capability.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <string.h>

void MGLCapabilityInit(MGLCapability *cap, id<MTLDevice> device)
{
    memset(cap, 0, sizeof(*cap));
    cap->device = device;

    NSString *name = [device name] ?: @"Unknown Metal Device";
    BOOL supportsAppleFamily = NO;
    if (@available(macOS 10.15, *)) {
        supportsAppleFamily = [device supportsFamily:MTLGPUFamilyApple1];
    }

    if ([name containsString:@"AGX"]) {
        /* Current AGX detection also implies virtualization in the MGL test
         * environment.  On bare-metal Apple Silicon the device name is
         * typically "Apple M1"/"Apple M2"/...  When running under QEMU /
         * virtualization the name contains "AGX". */
        cap->family = MGL_GPU_FAMILY_VIRTUALIZED;
        cap->isVirtualized = YES;
        NSLog(@"MGL CAP: AGX virtualized device detected: %@", name);
    } else if (supportsAppleFamily || [name hasPrefix:@"Apple "]) {
        cap->family = MGL_GPU_FAMILY_AGX;
        cap->isVirtualized = NO;
        NSLog(@"MGL CAP: Apple Silicon device detected: %@", name);
    } else {
        cap->family = MGL_GPU_FAMILY_OTHER;
        NSLog(@"MGL CAP: Non-AGX device detected: %@", name);
    }

    /* === Capability queries === */
    cap->maxSampleCount = 1;
    static const NSUInteger sampleCounts[] = { 32u, 16u, 8u, 4u, 2u };
    for (NSUInteger i = 0; i < sizeof(sampleCounts) / sizeof(sampleCounts[0]); ++i) {
        NSUInteger sampleCount = sampleCounts[i];
        if ([device supportsTextureSampleCount:sampleCount]) {
            cap->maxSampleCount = sampleCount;
            break;
        }
    }
    cap->supports8xMSAA = (cap->maxSampleCount >= 8);

    /* === Driver bug markers ===
     *
     * AGX (both virtualized and bare-metal Apple Silicon) shares the same
     * driver bug set.  If a future macOS version fixes any of these, the
     * fix is a one-line change here. */
    if (cap->family == MGL_GPU_FAMILY_VIRTUALIZED || cap->family == MGL_GPU_FAMILY_AGX) {
        cap->bug_3dGetBytesSliceOOB = YES;
        cap->bug_3dReplaceRegionNonZeroOrigin = YES;
        cap->bug_3dCopyFromBufferSliceOOB = YES;
        cap->bug_mslPipelineRejection = YES;

        /* Async shader compile crash is specific to virtualization. */
        cap->bug_asyncShaderCompileInVM = cap->isVirtualized;

        cap->textureAlignmentBytes = 256;
        cap->conservativeCPUCacheMode = YES;
        cap->maxConcurrentCommandBuffers = cap->isVirtualized ? 16 : 64;
        cap->commandBufferRecoveryLimit = 4096;
    } else {
        /* Conservative defaults for non-AGX. */
        cap->textureAlignmentBytes = 256;
        cap->conservativeCPUCacheMode = NO;
        cap->maxConcurrentCommandBuffers = 64;
        cap->commandBufferRecoveryLimit = 4096;
    }
}

bool MGLCapabilitySupportsSampleCount(MGLCapability *cap, NSUInteger samples)
{
    if (!cap || samples <= 1) return true;
    return samples <= cap->maxSampleCount &&
        [cap->device supportsTextureSampleCount:samples];
}

NSUInteger MGLCapabilityClampSampleCount(MGLCapability *cap, NSUInteger requested)
{
    if (!cap) return 1;
    if (requested <= 1) return 1;

    static const NSUInteger candidates[] = { 32u, 16u, 8u, 4u, 2u };

    for (NSUInteger i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        NSUInteger candidate = candidates[i];
        if (candidate <= requested && [cap->device supportsTextureSampleCount:candidate]) {
            return candidate;
        }
    }

    return 1u;
}

NSUInteger MGLCapabilityTextureAlignment(MGLCapability *cap)
{
    return cap ? cap->textureAlignmentBytes : 256;
}

bool MGLCapabilityUseConservativeCPUCache(MGLCapability *cap)
{
    return cap ? cap->conservativeCPUCacheMode : false;
}

NSUInteger MGLCapabilityMaxConcurrentCommandBuffers(MGLCapability *cap)
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
