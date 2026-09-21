/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_platform_shell_result.h - the platform shell's C surface, C-visible.
 *
 * Moved out of the Objective-C MGLPlatformRendererShell.h so that the shell's
 * C++ translation unit can include it (the class itself is registered with the
 * Objective-C runtime now; log 210).  MGLPlatformRendererShell.h includes this
 * header, so the Objective-C spelling of these names keeps working.
 */

#ifndef MGL_PLATFORM_SHELL_RESULT_H
#define MGL_PLATFORM_SHELL_RESULT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The exception boundary's result record: status plus the name and reason of a
 * raised NSException, for the C caller. */
typedef struct MGLPlatformRendererShellResult {
    int32_t status;
    char exception_name[128];
    char exception_reason[512];
} MGLPlatformRendererShellResult;

typedef int (*MGLPlatformRendererShellOperation)(void *context);

/* Platform-only drawable bridge used by renderer diagnostics. */
void *mglPlatformRendererShellTextureForDrawable(void *drawable);

/* C entry points that construct the platform renderer shell. Pure C callers
 * (glm_context.c) use these without including the ObjC MGLRenderer.h. */
void *CppCreateMGLRendererFromContextAndBindToWindow(void *glm_ctx,
                                                     void *window);
void *CppCreateMGLRendererHeadless(void *glm_ctx);
void *CppCreateMGLRendererAndBindToContext(void *glm_ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PLATFORM_SHELL_RESULT_H */
