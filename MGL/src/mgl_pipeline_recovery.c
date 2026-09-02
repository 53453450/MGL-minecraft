/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_pipeline_recovery.h"

bool mglPipelineRecoveryShouldAbortForProgramMismatch(
    const MGLPipelineRecoveryState *recovery, double now,
    uint32_t program_name, const void *existing_pipeline_state)
{
    if (!recovery || !existing_pipeline_state || program_name == 0u) {
        return false;
    }
    return (program_name == recovery->program_mismatch_program_name &&
            now < recovery->program_mismatch_retry_after);
}

bool mglPipelineRecoveryEvaluatePipelineRetry(
    MGLPipelineRecoveryState *recovery, double now, uint32_t program_name,
    const void *existing_pipeline_state, bool *skip_pipeline_build_out)
{
    if (skip_pipeline_build_out) {
        *skip_pipeline_build_out = false;
    }
    if (!recovery || now >= recovery->pipeline_retry_after) {
        return false;
    }

    bool retry_applies =
        (program_name != 0u &&
         (program_name == recovery->interface_mismatch_program_name ||
          program_name == recovery->program_mismatch_program_name ||
          program_name == recovery->interface_mismatch_blocked_program));

    if (retry_applies) {
        if (existing_pipeline_state) {
            if (skip_pipeline_build_out) {
                *skip_pipeline_build_out = true;
            }
            return true;
        }
        recovery->pipeline_retry_after = 0.0;
        recovery->program_mismatch_retry_after = 0.0;
        recovery->interface_mismatch_retry_after = 0.0;
        return false;
    }

    return false;
}

bool mglPipelineRecoveryShouldAbortForInterfaceMismatch(
    const MGLPipelineRecoveryState *recovery, double now,
    uint32_t program_name, uint32_t color0_format, uint32_t depth_format,
    uint32_t stencil_format)
{
    if (!recovery || now >= recovery->interface_mismatch_retry_after) {
        return false;
    }
    return (program_name == recovery->interface_mismatch_program_name &&
            color0_format == recovery->interface_mismatch_color0_format &&
            depth_format == recovery->interface_mismatch_depth_format &&
            stencil_format == recovery->interface_mismatch_stencil_format);
}

void mglPipelineRecoveryOnCacheHit(MGLPipelineRecoveryState *recovery,
                                   uint32_t program_name,
                                   uint32_t invalid_pixel_format)
{
    if (!recovery) {
        return;
    }

    recovery->interface_mismatch_streak = 0u;
    recovery->interface_mismatch_program_name = 0u;
    recovery->interface_mismatch_color0_format = invalid_pixel_format;
    recovery->interface_mismatch_depth_format = invalid_pixel_format;
    recovery->interface_mismatch_stencil_format = invalid_pixel_format;
    recovery->interface_mismatch_retry_after = 0.0;

    if (recovery->program_mismatch_program_name == program_name) {
        recovery->program_mismatch_program_name = 0u;
        recovery->program_mismatch_retry_after = 0.0;
        recovery->program_mismatch_streak = 0u;
    }
    if (recovery->interface_mismatch_blocked_program == program_name) {
        recovery->interface_mismatch_blocked_program = 0u;
        recovery->interface_mismatch_blocked_until = 0.0;
        recovery->interface_mismatch_blocked_streak = 0u;
    }
}
