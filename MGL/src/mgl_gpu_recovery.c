/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_gpu_recovery.c — bodies of -[MGLRenderer clearTextureCache] and
 * -getOptimalAlignmentForPixelFormat:, which never needed Objective-C. */

#include "mgl_gpu_recovery.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_renderer_ports.h"
#include "mgl_render.h"     /* mglRenderCommandRecovery* */

#include <time.h>

#include <stdio.h>

void mglRendererClearTextureCache(void)
{
    /* PROPER FIX: Intelligent texture cache cleanup.  Texture binding cache
     * cleanup would need the renderer's instance state; this is the hook that
     * would take it. */
    fprintf(stderr, "MGL INFO: Clearing texture cache to free memory\n");
}

uint64_t mglRendererOptimalAlignmentForPixelFormat(uint32_t format)
{
    (void)format;
    /* aligned_alloc requires an alignment compatible with platform pointer
     * alignment.  A conservative 64-byte value avoids EINVAL on macOS/arm64 and
     * is safe for texture rows. */
    return 64;
}

/* Wall clock in UNIX seconds, matching [[NSDate date] timeIntervalSince1970]
 * that the Objective-C versions used for the recovery timestamps. */
static double mglGpuRecoveryNowSeconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

void mglRendererRecordGPUError(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.gpu_recovery_command_owner
                      ? *areas.gpu_recovery_command_owner
                      : NULL;
    MGLRenderCommandRecoverySnapshot state = {0};
    if (mglRenderCommandRecoveryRecordError(owner, mglGpuRecoveryNowSeconds(),
                                            &state) == 0) {
        fprintf(stderr, "MGL AGX: Recorded GPU error (%llu consecutive)\n",
                (unsigned long long)state.consecutive_errors);
    }
}

void mglRendererRecordGPUSuccess(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.gpu_recovery_command_owner
                      ? *areas.gpu_recovery_command_owner
                      : NULL;
    MGLRenderCommandRecoverySuccess result = {0};
    if (mglRenderCommandRecoveryRecordSuccess(owner, mglGpuRecoveryNowSeconds(),
                                              &result) == 0 &&
        result.sustained_recovery) {
        fprintf(stderr,
                "MGL AGX: Sustained GPU recovery (%llu successes), resetting error count (was %llu)\n",
                (unsigned long long)result.recovered_successes,
                (unsigned long long)result.previous_errors);
    }
}

void mglRendererClearProblematicGPUState(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;

    fprintf(stderr, "MGL AGX: Clearing problematic GPU state for recovery\n");

    MGLRenderCommandBufferState currentState = {0};
    if (cs && mglRenderCommandBufferOwnerHasState(cs->currentCommandBufferOwner,
                                                   &currentState)) {
        mglRenderPassManagerDiscardCurrentCommandBuffer(renderer);
    }
    /* Don't recreate the command queue immediately - let it rest: the AGX driver
     * needs time to recover from the error state. */
}

int mglRendererShouldSkipGPUOperations(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.gpu_recovery_command_owner
                      ? *areas.gpu_recovery_command_owner
                      : NULL;

    MGLRenderCommandRecoverySkipDecision decision = {0};
    if (mglRenderCommandRecoveryShouldSkip(owner, mglGpuRecoveryNowSeconds(),
                                           &decision) != 0) {
        return 0;
    }
    if (decision.recovery_timed_out && decision.previous_errors > 0) {
        fprintf(stderr,
                "MGL AGX: Recovery timeout - attempting GPU operations (had %llu errors)\n",
                (unsigned long long)decision.previous_errors);
    }
    if (decision.entered_recovery_mode) {
        fprintf(stderr,
                "MGL AGX: Entering recovery mode after %llu consecutive errors\n",
                (unsigned long long)decision.state.consecutive_errors);
        mglRendererClearProblematicGPUState(renderer);
    }
    return decision.should_skip != 0;
}
