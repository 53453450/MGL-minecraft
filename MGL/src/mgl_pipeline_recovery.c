/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_pipeline_recovery.h"

#include <limits.h>

static bool mglPipelineRecoveryAttachmentFormatCompatible(uint32_t cached,
                                                          uint32_t built,
                                                          uint32_t invalid)
{
    return (cached == invalid || built == invalid || cached == built);
}

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

bool mglPipelineRecoveryCanReusePreviousOnInterfaceMismatch(
    const MGLPipelineRecoveryReuseInput *input)
{
    if (!input || !input->previous_pipeline_state ||
        input->cached_program_name == 0u ||
        input->cached_program_name != input->current_program_name ||
        input->cached_vertex_function != input->vertex_function ||
        input->cached_fragment_function != input->fragment_function) {
        return false;
    }
    return mglPipelineRecoveryAttachmentFormatCompatible(
               input->cached_color0_format, input->built_color0_format,
               input->invalid_pixel_format) &&
           mglPipelineRecoveryAttachmentFormatCompatible(
               input->cached_depth_format, input->built_depth_format,
               input->invalid_pixel_format) &&
           mglPipelineRecoveryAttachmentFormatCompatible(
               input->cached_stencil_format, input->built_stencil_format,
               input->invalid_pixel_format);
}

void mglPipelineRecoveryRecordReuseOnInterfaceMismatch(
    MGLPipelineRecoveryState *recovery, double now, uint32_t program_name,
    uint32_t color0_format, uint32_t depth_format, uint32_t stencil_format)
{
    if (!recovery) {
        return;
    }
    recovery->interface_mismatch_program_name = program_name;
    recovery->interface_mismatch_color0_format = color0_format;
    recovery->interface_mismatch_depth_format = depth_format;
    recovery->interface_mismatch_stencil_format = stencil_format;
    recovery->interface_mismatch_streak = 1u;
    recovery->interface_mismatch_retry_after = now + 0.10;
    recovery->pipeline_retry_after = recovery->interface_mismatch_retry_after;
}

void mglPipelineRecoveryRecordInterfaceMismatchFailure(
    MGLPipelineRecoveryState *recovery, double now, uint32_t program_name,
    uint32_t color0_format, uint32_t depth_format, uint32_t stencil_format,
    MGLPipelineRecoveryMismatchDelays *delays_out)
{
    if (delays_out) {
        *delays_out = (MGLPipelineRecoveryMismatchDelays){0};
    }
    if (!recovery) {
        return;
    }

    bool same_mismatch_signature =
        (program_name == recovery->interface_mismatch_program_name &&
         color0_format == recovery->interface_mismatch_color0_format &&
         depth_format == recovery->interface_mismatch_depth_format &&
         stencil_format == recovery->interface_mismatch_stencil_format);
    if (same_mismatch_signature) {
        if (recovery->interface_mismatch_streak < UINT32_MAX) {
            recovery->interface_mismatch_streak++;
        }
    } else {
        recovery->interface_mismatch_streak = 1u;
        recovery->interface_mismatch_program_name = program_name;
        recovery->interface_mismatch_color0_format = color0_format;
        recovery->interface_mismatch_depth_format = depth_format;
        recovery->interface_mismatch_stencil_format = stencil_format;
    }

    uint32_t capped_shift = (recovery->interface_mismatch_streak > 5u)
                                ? 4u
                                : (recovery->interface_mismatch_streak - 1u);
    double retry_delay = 0.10 * (double)(1u << capped_shift);
    if (retry_delay > 2.0) {
        retry_delay = 2.0;
    }
    recovery->interface_mismatch_retry_after = now + retry_delay;

    if (recovery->program_mismatch_program_name == program_name) {
        if (recovery->program_mismatch_streak < UINT32_MAX) {
            recovery->program_mismatch_streak++;
        }
    } else {
        recovery->program_mismatch_program_name = program_name;
        recovery->program_mismatch_streak = 1u;
    }
    uint32_t program_shift =
        (recovery->program_mismatch_streak > 6u)
            ? 6u
            : (recovery->program_mismatch_streak - 1u);
    double program_delay = 0.25 * (double)(1u << program_shift);
    if (program_delay > 20.0) {
        program_delay = 20.0;
    }
    recovery->program_mismatch_retry_after = now + program_delay;

    if (recovery->interface_mismatch_blocked_program == program_name) {
        if (recovery->interface_mismatch_blocked_streak < UINT32_MAX) {
            recovery->interface_mismatch_blocked_streak++;
        }
    } else {
        recovery->interface_mismatch_blocked_program = program_name;
        recovery->interface_mismatch_blocked_streak = 1u;
    }
    double quarantine_delay = retry_delay * 8.0;
    if (quarantine_delay < 1.00) {
        quarantine_delay = 1.00;
    }
    if (quarantine_delay > 15.00) {
        quarantine_delay = 15.00;
    }
    recovery->interface_mismatch_blocked_until = now + quarantine_delay;
    recovery->pipeline_retry_after =
        (recovery->interface_mismatch_blocked_until >
         recovery->interface_mismatch_retry_after)
            ? recovery->interface_mismatch_blocked_until
            : recovery->interface_mismatch_retry_after;

    if (delays_out) {
        delays_out->interface_retry_delay = retry_delay;
        delays_out->program_retry_delay = program_delay;
        delays_out->quarantine_delay = quarantine_delay;
        delays_out->log_interface_throttle =
            (recovery->interface_mismatch_streak <= 5u ||
             (recovery->interface_mismatch_streak % 200u) == 0u);
        delays_out->log_program_breaker =
            (recovery->program_mismatch_streak <= 8u ||
             (recovery->program_mismatch_streak % 64u) == 0u);
        delays_out->log_quarantine =
            (recovery->interface_mismatch_blocked_streak <= 6u ||
             (recovery->interface_mismatch_blocked_streak % 64u) == 0u);
    }
}
