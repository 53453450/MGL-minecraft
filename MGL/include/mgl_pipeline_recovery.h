/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Pipeline GPU-recovery breaker decisions (pure C, gtest-friendly).
 */

#ifndef MGL_PIPELINE_RECOVERY_H
#define MGL_PIPELINE_RECOVERY_H

#include <stdbool.h>
#include <stdint.h>

typedef struct MGLPipelineRecoveryState_t {
    double pipeline_retry_after;
    double interface_mismatch_retry_after;
    double program_mismatch_retry_after;
    uint32_t interface_mismatch_program_name;
    uint32_t interface_mismatch_color0_format;
    uint32_t interface_mismatch_depth_format;
    uint32_t interface_mismatch_stencil_format;
    uint32_t interface_mismatch_streak;
    uint32_t program_mismatch_program_name;
    uint32_t program_mismatch_streak;
    uint32_t interface_mismatch_blocked_program;
    double interface_mismatch_blocked_until;
    uint32_t interface_mismatch_blocked_streak;
} MGLPipelineRecoveryState;

#ifdef __cplusplus
extern "C" {
#endif

bool mglPipelineRecoveryShouldAbortForProgramMismatch(
    const MGLPipelineRecoveryState *recovery, double now,
    uint32_t program_name, const void *existing_pipeline_state);

bool mglPipelineRecoveryEvaluatePipelineRetry(
    MGLPipelineRecoveryState *recovery, double now, uint32_t program_name,
    const void *existing_pipeline_state, bool *skip_pipeline_build_out);

bool mglPipelineRecoveryShouldAbortForInterfaceMismatch(
    const MGLPipelineRecoveryState *recovery, double now,
    uint32_t program_name, uint32_t color0_format, uint32_t depth_format,
    uint32_t stencil_format);

void mglPipelineRecoveryOnCacheHit(MGLPipelineRecoveryState *recovery,
                                   uint32_t program_name,
                                   uint32_t invalid_pixel_format);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PIPELINE_RECOVERY_H */
