/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Narrow attachment→Texture lookup (ARCH-010 / Phase 5).
 * Prefer this over mgl_renderer_ports.h when only FBO attachment textures
 * are needed.
 */

#ifndef MGL_RENDERER_ATTACHMENT_H
#define MGL_RENDERER_ATTACHMENT_H

#include "glm_context.h"
#include "mgl_types_framebuffer.h"
#include "mgl_types_texture.h"

#ifdef __cplusplus
extern "C" {
#endif

Texture *mglRendererAttachmentTextureFor(GLMContext ctx, FBOAttachment *att);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_ATTACHMENT_H */
