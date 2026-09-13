/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_gpu_recovery.h — the GPU-recovery helpers that carry no Objective-C
 * dependency, moved out of MGLRenderer+GPURecovery.m (P0-1). */

#ifndef MGL_GPU_RECOVERY_H
#define MGL_GPU_RECOVERY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Texture-cache cleanup hook (diagnostic today; kept as the single place the
 * renderer would hand its texture cache back). */
void mglRendererClearTextureCache(void);

/* Row alignment handed to aligned_alloc for texture uploads: a conservative
 * 64 bytes, which is valid on macOS/arm64. */
uint64_t mglRendererOptimalAlignmentForPixelFormat(uint32_t format);

/* Command-recovery bookkeeping driven from the texture path. */
void mglRendererRecordGPUError(void *renderer);
void mglRendererRecordGPUSuccess(void *renderer);

/* GPU-operation gating: clears problematic resources and reports whether GPU
 * work should be skipped while the driver recovers. */
void mglRendererClearProblematicGPUState(void *renderer);
int mglRendererShouldSkipGPUOperations(void *renderer);

/* Body (no exception guard) of the command-buffer cleanup; call it through
 * mglPlatformShellGuardedCall() to keep the historical @try/@catch. */
int mglRendererCleanupCommandBufferBody(void *renderer);

/* Runs a C body with the Objective-C @try/@catch the cleanup paths always had
 * (implemented in the shell TU).  Returns the body's result, or 0 when it threw. */
int mglPlatformShellGuardedCall(void *renderer, const char *what,
                                int (*body)(void *));

#ifdef __cplusplus
}
#endif

#endif /* MGL_GPU_RECOVERY_H */
