/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_ms_sample_loop.h — emulated-MSample per-sample draw loop (P0-1). */

#ifndef MGL_MS_SAMPLE_LOOP_H
#define MGL_MS_SAMPLE_LOOP_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runs `draw_once(draw_ctx)` once per emulated sample plane.  Returns 1 when the
 * loop ran (the caller must not issue the plain draw), 0 when it did not apply. */
int mglRendererRunEmulatedMSSampleDrawLoopIfNeeded(
    void *renderer, GLMContext ctx, void (*draw_once)(void *), void *draw_ctx);

/* Copies plane 0 to every other plane when per-sample draws did not fill them. */
void mglRendererBroadcastEmulatedMSSamplePlanesAfterDrawIfNeeded(void *renderer,
                                                                 GLMContext ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_MS_SAMPLE_LOOP_H */
