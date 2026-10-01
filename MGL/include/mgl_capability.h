/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * MGLCapabilityHasBug string comparisons.
 *
 * AGX 3D copies need no driver-bug workaround: Metal honours the documented
 * destinationSlice/destinationOrigin and getBytes contracts for 3D textures
 * (see docs/AGX_COPY3D_DRIVER_BUG_RECHECK_2026-09-20.md). */
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
#ifdef __cplusplus
extern "C" {
#endif

void MGLCapabilityInit(MGLCapability *cap, void *deviceRef);

/* === Capability query API === */
bool       MGLCapabilitySupportsSampleCount(MGLCapability *cap, uint64_t samples);
uint64_t   MGLCapabilityClampSampleCount(MGLCapability *cap, uint64_t requested);
uint64_t   MGLCapabilityTextureAlignment(MGLCapability *cap);
bool       MGLCapabilityUseConservativeCPUCache(MGLCapability *cap);
uint64_t   MGLCapabilityMaxConcurrentCommandBuffers(MGLCapability *cap);

/* === Driver bug query API === */
bool       MGLCapabilityHasBug(MGLCapability *cap, const char *bugName);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MGL_CAPABILITY_H */
