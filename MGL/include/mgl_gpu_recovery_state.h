/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_gpu_recovery_state.h - the renderer's GPU-recovery record, C-visible.
 *
 * Moved out of the Objective-C MGLRenderer_State.h for the same reason as the
 * renderer core state: the PSO build path writes its interface-mismatch and
 * quarantine fields, and that path is C now.  The renderer keeps the record as
 * an ivar and the state areas hand out its address.
 */

#ifndef MGL_GPU_RECOVERY_STATE_H
#define MGL_GPU_RECOVERY_STATE_H

#include "glcorearb.h"

#if defined(__APPLE__)
#include <CoreFoundation/CFBase.h>
#endif
/* The renderer records timestamps with CFTimeInterval (a double). */
#ifndef CFTimeInterval
typedef double CFTimeInterval;
#endif
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLGPURecoveryState_t {
    void *commandRecoveryOwner;
    GLuint interfaceMismatchBlockedProgram;
    CFTimeInterval interfaceMismatchBlockedUntil;
    uint32_t interfaceMismatchBlockedStreak;
    CFTimeInterval pipelineRetryAfter;
    CFTimeInterval interfaceMismatchRetryAfter;
    GLuint interfaceMismatchProgramName;
    uint32_t interfaceMismatchColor0Format;
    uint32_t interfaceMismatchDepthFormat;
    uint32_t interfaceMismatchStencilFormat;
    uint32_t interfaceMismatchStreak;
    GLuint programMismatchProgramName;
    CFTimeInterval programMismatchRetryAfter;
    uint32_t programMismatchStreak;
} MGLGPURecoveryState;

#ifdef __cplusplus
}
#endif

#endif /* MGL_GPU_RECOVERY_STATE_H */
