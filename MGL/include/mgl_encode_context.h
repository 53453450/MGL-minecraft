/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_encode_context.h — the encode target the draw / batch paths pass down.
 *
 * The record used to be declared in MGLRenderer+Draw_Private.h, which C
 * translation units cannot include.  The C drivers of the batch plans
 * (mgl_batch_issue_encode.c, and its neighbours as they follow) take the same
 * pointer, so the definition lives here in a C-safe header and the Objective-C
 * header includes it instead of repeating it.
 */

#ifndef MGL_ENCODE_CONTEXT_H
#define MGL_ENCODE_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

/* Encode target passed explicitly to issue and bind methods.
 * T0-1: `state` is GLMState * for the flush replay workspace (same object
 * activate pointed both dual-proxy slots at).  Stored as void * so this
 * header does not pull glm_context.h. */
typedef struct MGLEncodeContext {
    void *render_encoder_owner;
    void *state; /* GLMState * */
} MGLEncodeContext;

/* Issue entry: enc->state must be non-NULL and equal ctx->active_state. */
void mglEncodeContextRequireReplayState(const MGLEncodeContext *enc,
                                        GLMContext ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_ENCODE_CONTEXT_H */
