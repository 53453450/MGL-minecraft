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
#include "mgl_sync.h"
#include "mgl_thread_affinity.h"   /* MGL_ASSERT_GL_THREAD */       /* MGL_COMMAND_BUFFER_STATUS_COMMITTED */
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

/* Body of the former -[MGLRenderer cleanupCommandBuffer]; the @try/@catch that
 * wrapped it is provided by mglPlatformShellGuardedCall(). */
int mglRendererCleanupCommandBufferBody(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return 0;
    }

    MGLRenderCommandBufferState currentState = {0};
    if (mglRenderCommandBufferOwnerHasState(cs->currentCommandBufferOwner,
                                            &currentState)) {
        if (currentState.status == MGL_COMMAND_BUFFER_STATUS_COMMITTED) {
            /* Do not block indefinitely here; cleanup can be invoked on the
             * render thread.  Command buffers retain resources until
             * completion, so dropping the reference is safe. */
            if (0) {   /* kMGLVerboseFrameLoopLogs is NO (MGLRenderer+RenderPass_Private.h) */
                fprintf(stderr,
                        "MGL INFO: cleanupCommandBuffer skipping blocking wait for committed command buffer\n");
            }
        }
        mglRenderPassManagerDiscardCurrentCommandBuffer(renderer);
    }

    if (mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) == 1) {
        mglRenderPassManagerEndCurrentRenderEncoder(renderer);
        mglRenderPassManagerClearCurrentRenderEncoder(renderer);
    }
    return 1;
}

void mglRendererResetMetalState(void *renderer)
{
    MGL_ASSERT_GL_THREAD();
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    fprintf(stderr, "MGL INFO: Performing full Metal state reset for AGX recovery\n");

    /* Runs on the GL calling thread (frame-boundary drain in mtlSwapBuffers or
     * GL-layer error paths), so this is not a cross-thread reset. */
    (void)mglPlatformShellGuardedCall(renderer, "command buffer cleanup",
                                      mglRendererCleanupCommandBufferBody);

    fprintf(stderr,
            "MGL AGX RECOVERY: Recreating command queue to clear GPU error state\n");
    if (!mglPlatformShellRecreateCommandQueue(renderer)) {
        fprintf(stderr,
                "MGL CRITICAL: Failed to recreate command queue during AGX recovery\n");
    } else {
        fprintf(stderr, "MGL AGX RECOVERY: Command queue successfully recreated\n");
    }

    (void)mglPipelineCacheResetCaches(areas.pipeline_cache_object);

    mglRendererClearTextureCache();

    fprintf(stderr, "MGL INFO: AGX Metal state reset completed\n");
}

/* Body of the former -[MGLRenderer validateMetalObjects]; the @try/@catch is the
 * shell guard, the device/queue probes are shell forwards, and the wall clock
 * matches [[NSDate date] timeIntervalSince1970] as before. */
static int mglRendererValidateMetalObjectsBody(void *renderer)
{
    if (!mglPlatformShellMetalObjectsPresent(renderer)) {
        fprintf(stderr,
                "MGL ERROR: Metal device or command queue is nil during validation\n");
        return 0;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;

    /* GPU ERROR THROTTLING: track recent failures to prevent error cascades. */
    static uint64_t consecutiveGpuErrors = 0;
    static double lastErrorTime = 0.0;
    static const double throttleWindow = 2.0;   /* 2 second throttle window */
    static const uint64_t maxErrorsPerWindow = 3;

    MGLRenderCommandBufferState currentState = {0};
    int hasCurrentCommandBuffer =
        cs && mglRenderCommandBufferOwnerHasState(cs->currentCommandBufferOwner,
                                                  &currentState);
    if (hasCurrentCommandBuffer && currentState.has_error) {
        double currentTime = mglGpuRecoveryNowSeconds();
        if (currentTime - lastErrorTime < throttleWindow) {
            consecutiveGpuErrors++;
            fprintf(stderr,
                    "MGL GPU THROTTLING: %llu consecutive GPU errors detected\n",
                    (unsigned long long)consecutiveGpuErrors);
            if (consecutiveGpuErrors > maxErrorsPerWindow) {
                fprintf(stderr,
                        "MGL CRITICAL: GPU error threshold exceeded - throttling operations for %.1f seconds\n",
                        throttleWindow);
                mglRendererResetMetalState(renderer);
                if (currentTime - lastErrorTime > throttleWindow) {
                    consecutiveGpuErrors = 0;
                } else {
                    return 0;   /* skip this operation to prevent more errors */
                }
            }
        } else {
            consecutiveGpuErrors = 1;
            lastErrorTime = currentTime;
        }
    }

    /* Device registry ID changes indicate virtualization issues. */
    if (__builtin_available(macOS 11.0, *)) {
        void *device = mglPlatformShellMetalDevice(renderer);
        if (device) {
            uint64_t registryID = 0;
            (void)mglRenderGetDeviceIdentity(device, &registryID, NULL, 0);
            if (registryID == 0) {
                fprintf(stderr,
                        "MGL WARNING: Detected virtualized Metal environment - enabling safety mode\n");
            }
        }
    }
    return 1;
}

int mglRendererValidateMetalObjects(void *renderer)
{
    return mglPlatformShellGuardedCall(renderer, "Metal object validation",
                                       mglRendererValidateMetalObjectsBody);
}
