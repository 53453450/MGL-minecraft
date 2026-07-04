/*
 * mgl_capability.h
 * MGL
 *
 * AGX Capability Layer: centralized device detection, capability queries,
 * and driver bug markers.  This layer decouples OpenGL spec compliance from
 * Apple GPU / AGX driver peculiarities so that the rest of MGL can query
 * capabilities through a single semantic API instead of scattering
 * `containsString:@"AGX"` checks and hardcoded constants.
 */

#ifndef MGL_CAPABILITY_H
#define MGL_CAPABILITY_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

typedef enum {
    MGL_GPU_FAMILY_UNKNOWN = 0,
    MGL_GPU_FAMILY_AGX,           /* Apple Silicon (M1/M2/M3/M4...) */
    MGL_GPU_FAMILY_VIRTUALIZED,   /* AGX in QEMU / virtualization */
    MGL_GPU_FAMILY_OTHER,         /* Intel / AMD on macOS */
} MGLGPUFamily;

/* Semantic driver bug names.  Keep in sync with the implementation's
 * MGLCapabilityHasBug string comparisons. */
#define MGL_BUG_3D_GETBYTES_SLICE_OOB           "3d_getbytes_slice_oob"
#define MGL_BUG_3D_REPLACE_REGION_NONZERO_ORIGIN "3d_replace_region_nonzero_origin"
#define MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB   "3d_copy_from_buffer_slice_oob"
#define MGL_BUG_ASYNC_SHADER_COMPILE_IN_VM      "async_shader_compile_in_vm"
#define MGL_BUG_MSL_PIPELINE_REJECTION          "msl_pipeline_rejection"

typedef struct MGLCapability_t {
#ifdef __OBJC__
    id<MTLDevice>  __strong device;
#endif
    MGLGPUFamily   family;
    bool           isVirtualized;

    /* === Capability queries (lazy-cached at init) === */
    bool           supports8xMSAA;
    NSUInteger     maxSampleCount;
    NSUInteger     maxTextureDimensions;

    /* === Driver bug markers (semantic) === */
    bool           bug_3dGetBytesSliceOOB;
    bool           bug_3dReplaceRegionNonZeroOrigin;
    bool           bug_3dCopyFromBufferSliceOOB;
    bool           bug_asyncShaderCompileInVM;
    bool           bug_mslPipelineRejection;

    /* === Robustness config === */
    NSUInteger     commandBufferRecoveryLimit;
    NSUInteger     maxConcurrentCommandBuffers;
    NSUInteger     textureAlignmentBytes;
    bool           conservativeCPUCacheMode;
} MGLCapability;

/* Initialize capability from a Metal device.  Must be called once after the
 * device is created.  Subsequent queries read cached fields without touching
 * the Metal API. */
#ifdef __OBJC__
void MGLCapabilityInit(MGLCapability *cap, id<MTLDevice> device);
#endif

/* === Capability query API === */
bool       MGLCapabilitySupportsSampleCount(MGLCapability *cap, NSUInteger samples);
NSUInteger MGLCapabilityClampSampleCount(MGLCapability *cap, NSUInteger requested);
NSUInteger MGLCapabilityTextureAlignment(MGLCapability *cap);
bool       MGLCapabilityUseConservativeCPUCache(MGLCapability *cap);
NSUInteger MGLCapabilityMaxConcurrentCommandBuffers(MGLCapability *cap);

/* === Driver bug query API === */
bool       MGLCapabilityHasBug(MGLCapability *cap, const char *bugName);

#endif /* MGL_CAPABILITY_H */
