/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_stage_encode_drivers.h — the vertex/fragment stage binding drivers moved
 * out of MGLRenderer+BindingState.m (P0-1, log 131).
 *
 *   -bindVertexBuffersToCurrentRenderEncoder:  -> mglStageEncodeBindVertexBuffers
 *   -bindVertexAttributesFromVAO:…             -> mglStageEncodeBindVertexAttributes
 *   -bindPointSizeParamsIfNeeded:…             -> mglStageEncodeBindPointSizeParams
 *   -bindFragmentBuffersToCurrentRenderEncoder:-> mglStageEncodeBindFragmentBuffers
 *   -finalizeStageBufferPresentMask:…          -> mglStageEncodeFinalizePresentMask
 *
 * The vertex and fragment drivers replace the shell ports
 * mglRendererBindVertexBuffersToCurrentRenderEncoderPort and
 * mglRendererBindFragmentBuffersToCurrentRenderEncoderPort (retired in the same
 * cut), so the C draw host calls these entries directly.
 */

#ifndef MGL_STAGE_ENCODE_DRIVERS_H
#define MGL_STAGE_ENCODE_DRIVERS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"         /* GLMContext */
#include "mgl_encode_context.h"  /* MGLEncodeContext */
#include "mgl_render.h"          /* MGLRenderBindingSnapshot */
#include "mgl_types_buffer.h"    /* MAX_BINDABLE_BUFFERS */
#include "mgl_types_program.h"   /* Program */

#ifdef __cplusplus
extern "C" {
#endif

/* Vertex stage: mapped buffers, VAO attributes, fallback fill and the
 * present-mask finalize. */
bool mglStageEncodeBindVertexBuffers(void *renderer,
                                     const MGLEncodeContext *enc_ctx);

/* Fragment stage: mapped buffers, fallback fill and the present-mask finalize. */
bool mglStageEncodeBindFragmentBuffers(void *renderer,
                                       const MGLEncodeContext *enc_ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STAGE_ENCODE_DRIVERS_H */
