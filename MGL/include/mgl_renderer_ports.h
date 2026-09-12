/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_renderer_ports.h — C-callable renderer ports (ObjC-zeroing T4).
 *
 * The renderer owns state that C code needs to read: the texture behind an FBO
 * attachment and the render-pass state owner.  Both used to be reachable only
 * through Objective-C messages; these ports keep the accessors in one place so
 * C modules stop needing a category to be linked next to them.
 *
 * The renderer handle is a `void *` because C has no MGLRenderer type; the
 * implementation casts it back.
 */

#ifndef MGL_RENDERER_PORTS_H
#define MGL_RENDERER_PORTS_H

#include "glm_context.h"           /* GLMContext */
#include "mgl_types_framebuffer.h" /* FBOAttachment */
#include "mgl_types_texture.h"     /* Texture */

#ifdef __cplusplus
extern "C" {
#endif

/* Texture an attachment renders into: the renderbuffer's texture, or the
 * texture object (looked up and cached on first use).  NULL when the
 * attachment carries none -- the C port of
 * -[MGLRenderer framebufferAttachmentTexture:]. */
Texture *mglRendererAttachmentTextureFor(GLMContext ctx, FBOAttachment *att);

/* Render pass state owner of a renderer handle, or NULL when there is none. */
void *mglRendererRenderPassStateOwnerPort(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_PORTS_H */
