/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_draw_support.h — the draw-support predicates and the polygon-offset
 * application moved out of MGLRenderer+DrawSupport.m (P0-1).
 *
 * Their bodies were already C: the rasterization-empty decision, the
 * polygon-offset decision, the fully-culled predicate and the buffer resolve
 * all call the mglRender and mglRenderBinding helpers; only the Objective-C state
 * access (`ctx`, `_renderPassManager.state`, `_bindingStateOwner`) kept them in
 * a category, and that state now arrives through MGLRendererStateAreas.
 */

#ifndef MGL_DRAW_SUPPORT_H
#define MGL_DRAW_SUPPORT_H

#include "glm_context.h"
#include "mgl_types_buffer.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Indirect (draw-indirect) buffer for the draw about to be encoded: fills the
 * GL buffer and, when it has one, the Metal buffer.  0 when there is none. */
int mglDrawResolveIndirectBuffer(void *renderer, const char *label,
                                 GLMContext ctx, Buffer **gl_out,
                                 void **mtl_out);

/* True when the current viewport/scissor and render-pass attachments leave no
 * rasterizable pixels. */
int mglDrawRasterizationIsEmpty(void *renderer);

/* Apply the polygon-offset decision for `mode` (PolygonMode repair, triangle
 * fill mode, depth bias) to the live binding state. */
void mglDrawApplyPolygonOffset(void *renderer, uint32_t mode);

/* True when `mode`'s primitives are all culled by the current cull state. */
int mglDrawModeIsFullyCulled(void *renderer, uint32_t mode);

/* True when the current fragment program needs per-sample MS values. */
int mglDrawFragmentNeedsPerSampleMSValues(GLMContext ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_SUPPORT_H */
