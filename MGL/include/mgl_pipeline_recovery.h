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

typedef struct MGLPipelineRecoveryReuseInput_t {
    const void *previous_pipeline_state;
    uint32_t current_program_name;
    uint32_t cached_program_name;
    const void *cached_vertex_function;
    const void *cached_fragment_function;
    const void *vertex_function;
    const void *fragment_function;
    uint32_t cached_color0_format;
    uint32_t cached_depth_format;
    uint32_t cached_stencil_format;
    uint32_t built_color0_format;
    uint32_t built_depth_format;
    uint32_t built_stencil_format;
    uint32_t invalid_pixel_format;
} MGLPipelineRecoveryReuseInput;

bool mglPipelineRecoveryCanReusePreviousOnInterfaceMismatch(
    const MGLPipelineRecoveryReuseInput *input);

void mglPipelineRecoveryRecordReuseOnInterfaceMismatch(
    MGLPipelineRecoveryState *recovery, double now, uint32_t program_name,
    uint32_t color0_format, uint32_t depth_format, uint32_t stencil_format);

typedef struct MGLPipelineRecoveryMismatchDelays_t {
    double interface_retry_delay;
    double program_retry_delay;
    double quarantine_delay;
    bool log_interface_throttle;
    bool log_program_breaker;
    bool log_quarantine;
} MGLPipelineRecoveryMismatchDelays;

void mglPipelineRecoveryRecordInterfaceMismatchFailure(
    MGLPipelineRecoveryState *recovery, double now, uint32_t program_name,
    uint32_t color0_format, uint32_t depth_format, uint32_t stencil_format,
    MGLPipelineRecoveryMismatchDelays *delays_out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PIPELINE_RECOVERY_H */
