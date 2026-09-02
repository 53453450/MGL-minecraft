/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#import "MGLRenderer_State.h"
#include "mgl_renderer_gpu_recovery_bridge.h"

MGLPipelineRecoveryState mglPipelineRecoveryViewFromGPU(
    const MGLGPURecoveryState *gpu)
{
    MGLPipelineRecoveryState view = {0};
    if (!gpu) {
        return view;
    }
    view.pipeline_retry_after = gpu->pipelineRetryAfter;
    view.interface_mismatch_retry_after = gpu->interfaceMismatchRetryAfter;
    view.program_mismatch_retry_after = gpu->programMismatchRetryAfter;
    view.interface_mismatch_program_name = gpu->interfaceMismatchProgramName;
    view.interface_mismatch_color0_format = gpu->interfaceMismatchColor0Format;
    view.interface_mismatch_depth_format = gpu->interfaceMismatchDepthFormat;
    view.interface_mismatch_stencil_format = gpu->interfaceMismatchStencilFormat;
    view.interface_mismatch_streak = gpu->interfaceMismatchStreak;
    view.program_mismatch_program_name = gpu->programMismatchProgramName;
    view.program_mismatch_streak = gpu->programMismatchStreak;
    view.interface_mismatch_blocked_program = gpu->interfaceMismatchBlockedProgram;
    view.interface_mismatch_blocked_until = gpu->interfaceMismatchBlockedUntil;
    view.interface_mismatch_blocked_streak = gpu->interfaceMismatchBlockedStreak;
    return view;
}

void mglPipelineRecoveryApplyToGPU(MGLGPURecoveryState *gpu,
                                   const MGLPipelineRecoveryState *view)
{
    if (!gpu || !view) {
        return;
    }
    gpu->pipelineRetryAfter = view->pipeline_retry_after;
    gpu->interfaceMismatchRetryAfter = view->interface_mismatch_retry_after;
    gpu->programMismatchRetryAfter = view->program_mismatch_retry_after;
    gpu->interfaceMismatchProgramName = view->interface_mismatch_program_name;
    gpu->interfaceMismatchColor0Format = view->interface_mismatch_color0_format;
    gpu->interfaceMismatchDepthFormat = view->interface_mismatch_depth_format;
    gpu->interfaceMismatchStencilFormat = view->interface_mismatch_stencil_format;
    gpu->interfaceMismatchStreak = view->interface_mismatch_streak;
    gpu->programMismatchProgramName = view->program_mismatch_program_name;
    gpu->programMismatchStreak = view->program_mismatch_streak;
    gpu->interfaceMismatchBlockedProgram = view->interface_mismatch_blocked_program;
    gpu->interfaceMismatchBlockedUntil = view->interface_mismatch_blocked_until;
    gpu->interfaceMismatchBlockedStreak = view->interface_mismatch_blocked_streak;
}
