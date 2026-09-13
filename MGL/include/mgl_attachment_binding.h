/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_attachment_binding.h — FBO attachment -> Metal texture binding, moved out
 * of MGLRenderer+VertexLayout.m (P0-1).
 *
 * Both entry points were already C apart from their Objective-C shells: the
 * attachment texture comes from mglRendererAttachmentTextureFor() (the C port of
 * -framebufferAttachmentTexture:) and the Metal push goes through
 * mglRendererBindMTLTexturePort(), so no new Objective-C surface is needed.
 */

#ifndef MGL_ATTACHMENT_BINDING_H
#define MGL_ATTACHMENT_BINDING_H

#include "glm_context.h"
#include "mgl_types_framebuffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bind one attachment's texture (marks it a render target for draw buffers).
 * Returns 0 when the Metal bind failed.  A missing texture is not an error. */
int mglRendererBindFramebufferTexture(void *renderer, FBOAttachment *attachment,
                                      int is_draw_buffer);

/* Bind every attachment of the currently bound framebuffer.  Returns 0 on a
 * NULL/invalid context or framebuffer, or when one attachment fails to bind. */
int mglRendererBindFramebufferAttachmentTextures(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_ATTACHMENT_BINDING_H */
