/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Bridge MGLGPURecoveryState (ObjC renderer) ↔ MGLPipelineRecoveryState (pure C).
 */

#ifndef MGL_RENDERER_GPU_RECOVERY_BRIDGE_H
#define MGL_RENDERER_GPU_RECOVERY_BRIDGE_H

#include "mgl_pipeline_recovery.h"

struct MGLGPURecoveryState_t;
typedef struct MGLGPURecoveryState_t MGLGPURecoveryState;

#ifdef __cplusplus
extern "C" {
#endif

MGLPipelineRecoveryState mglPipelineRecoveryViewFromGPU(
    const MGLGPURecoveryState *gpu);
void mglPipelineRecoveryApplyToGPU(MGLGPURecoveryState *gpu,
                                   const MGLPipelineRecoveryState *view);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_GPU_RECOVERY_BRIDGE_H */
