/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_readback_clear.h — the pending-clear application that readback
 * and blit need before touching a texture.
 *
 * Formerly four MGLRenderer+Texture.m methods (mglApplyPendingFBODepthClear
 * ForReadback:, mglApplyPendingFBOColorClearForReadback:,
 * mglApplyPendingDefaultDepthClearToTexture:,
 * mglApplyPendingDefaultColorClearToTexture:).  They were already plain C; they
 * read the command buffer owner through the C-visible command state.
 */

#ifndef MGL_TEXTURE_READBACK_CLEAR_H
#define MGL_TEXTURE_READBACK_CLEAR_H

#include "glm_context.h"
#include "mgl_types_framebuffer.h"
#include "mgl_types_texture.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Apply the attachment's pending depth clear (and mark the level written). */
void mglTextureApplyPendingFBODepthClearForReadback(void *renderer,
                                                    Framebuffer *fbo,
                                                    FBOAttachment *attachment,
                                                    Texture *texture_obj,
                                                    void *mtl_texture);

/* Same for colour; attachmentEnum is kept for call-site symmetry. */
void mglTextureApplyPendingFBOColorClearForReadback(void *renderer,
                                                    Framebuffer *fbo,
                                                    FBOAttachment *attachment,
                                                    Texture *texture_obj,
                                                    void *mtl_texture,
                                                    uint32_t attachment_enum);

/* Apply the default framebuffer's pending depth clear. */
void mglTextureApplyPendingDefaultDepthClear(void *renderer, void *mtl_texture);

/* Apply the default framebuffer's pending colour clear. */
void mglTextureApplyPendingDefaultColorClear(void *renderer, void *mtl_texture);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_READBACK_CLEAR_H */
