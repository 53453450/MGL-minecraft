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
    /* Borrowed from the renderer backend, which owns the Metal device for
     * the full lifetime of this value-state cache. */
    void          *device;
    MGLGPUFamily   family;
    bool           isVirtualized;

    /* === Capability queries (lazy-cached at init) === */
    bool           supports8xMSAA;
    uint64_t       maxSampleCount;
    uint64_t       maxTextureDimensions;

    /* === Driver bug markers (semantic) === */
    bool           bug_3dGetBytesSliceOOB;
    bool           bug_3dReplaceRegionNonZeroOrigin;
    bool           bug_3dCopyFromBufferSliceOOB;
    bool           bug_asyncShaderCompileInVM;
    bool           bug_mslPipelineRejection;

    /* === Robustness config === */
    uint64_t       commandBufferRecoveryLimit;
    uint64_t       maxConcurrentCommandBuffers;
    uint64_t       textureAlignmentBytes;
    bool           conservativeCPUCacheMode;
} MGLCapability;

/* Initialize capability from a backend-owned Metal device.  Must be called
 * once after backend creation; the borrowed device pointer remains valid
 * until backend shutdown. */
void MGLCapabilityInit(MGLCapability *cap, void *deviceRef);

/* === Capability query API === */
bool       MGLCapabilitySupportsSampleCount(MGLCapability *cap, uint64_t samples);
uint64_t   MGLCapabilityClampSampleCount(MGLCapability *cap, uint64_t requested);
uint64_t   MGLCapabilityTextureAlignment(MGLCapability *cap);
bool       MGLCapabilityUseConservativeCPUCache(MGLCapability *cap);
uint64_t   MGLCapabilityMaxConcurrentCommandBuffers(MGLCapability *cap);

/* === Driver bug query API === */
bool       MGLCapabilityHasBug(MGLCapability *cap, const char *bugName);

#endif /* MGL_CAPABILITY_H */
