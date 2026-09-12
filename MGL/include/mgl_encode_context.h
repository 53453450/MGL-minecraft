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

/* Encode target passed explicitly to issue and bind methods. */
typedef struct MGLEncodeContext {
    void *render_encoder_owner;
} MGLEncodeContext;

#ifdef __cplusplus
}
#endif

#endif /* MGL_ENCODE_CONTEXT_H */
