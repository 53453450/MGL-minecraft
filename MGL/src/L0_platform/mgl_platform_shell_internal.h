/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_platform_shell_internal.h - the platform shell's own C-callable helpers
 * (P0-1, T5 option (a); log 206).
 *
 * The shell's ports moved to mgl_platform_shell.cpp while the shell classes are
 * still Objective-C, so the helpers both halves share need a C declaration.
 * Only the shell implementation uses these; nothing outside the platform layer
 * should.
 */

#ifndef MGL_PLATFORM_SHELL_INTERNAL_H
#define MGL_PLATFORM_SHELL_INTERNAL_H

#include <CoreGraphics/CGGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -mglApplyPendingDrawableSize: publish the atomically recorded view geometry
 * to the Metal layer, or read the layer's current drawable size when nothing
 * is pending.  Must run on the GL thread. */
CGSize mglPlatformShellApplyPendingDrawableSizeCGSize(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PLATFORM_SHELL_INTERNAL_H */
