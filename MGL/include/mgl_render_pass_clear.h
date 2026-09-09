/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_render_pass_clear.h — O3.1 clear-value planning (pure C, no Metal).
 *
 * Deliberately a SEPARATE header from mgl_render_pass_plan.h: this one needs
 * the full MGLRenderPassState definition from mgl_render.h, while
 * mgl_render_pass_plan.h is pulled in by glm_context.h and therefore must NOT
 * include mgl_render.h (that would create an include cycle: mgl_render.h ->
 * glm_context.h -> mgl_render_pass_plan.h -> mgl_render.h).
 *
 * The function declared here is implemented in mgl_render_pass_plan.c so it
 * stays with the rest of the render-pass plan layer.
 */

#ifndef MGL_RENDER_PASS_CLEAR_H
#define MGL_RENDER_PASS_CLEAR_H

#include <stdint.h>
#include "mgl_render.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve the clear color (RGBA) / depth / stencil for a single attachment of
 * an already-fetched persistent MGLRenderPassState.
 *
 * Sinks the decision formerly inline in MGLRenderer+RenderPass.m's
 * mglRenderPassClearValuesFor.  Returns 1 on success; 0 for a NULL state, an
 * out-of-range color index (>= MGL_RENDER_MAX_COLOR_ATTACHMENTS), or an
 * unrecognized attachment kind.  Output pointers may be NULL.
 *
 * Taking the state struct directly (instead of MGLCommandState *) keeps this
 * free of the renderPassStateOwner and makes it directly unit-testable — this
 * is what the clear-value regression harness (test-render-pass-clear-plan)
 * drives.
 */
int mglRenderPassPlanClearValues(const MGLRenderPassState *state,
                                 uint32_t attachmentKind,
                                 uint32_t colorIndex,
                                 double clearColorOut[4],
                                 double *clearDepthOut,
                                 uint32_t *clearStencilOut);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_PASS_CLEAR_H */
