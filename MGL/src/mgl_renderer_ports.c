/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_renderer_ports.c — C ports of renderer accessors (ObjC-zeroing T4).
 *
 * mglRendererAttachmentTextureFor() is the exact behaviour of the Objective-C
 * -[MGLRenderer framebufferAttachmentTexture:] it replaced: a renderbuffer
 * attachment resolves through rbo->tex, a texture attachment through the cached
 * buf.tex or a findTexture() lookup that is then cached, and both a NULL
 * attachment and an attachment without a texture are reported on stderr.
 * -[MGLRenderer framebufferAttachmentTexture:] forwards here, so the ~27 ObjC
 * call sites keep their behaviour.
 */

#include "mgl_renderer_ports.h"

#include <stdio.h>

/* Defined in textures.c (C); declared here like framebuffers.c does. */
extern Texture *findTexture(GLMContext ctx, GLuint texture);

Texture *mglRendererAttachmentTextureFor(GLMContext ctx, FBOAttachment *att)
{
    Texture *tex = NULL;

    if (!att) {
        fprintf(stderr,
                "MGL ERROR: framebufferAttachmentTexture called with NULL attachment\n");
        return NULL;
    }

    if (mglRenderTargetIsRenderbuffer((uint32_t)att->textarget)) {
        if (att->buf.rbo) {
            tex = att->buf.rbo->tex;
        }
    } else {
        tex = att->buf.tex;
        if (!tex && att->texture != 0 && ctx) {
            tex = findTexture(ctx, att->texture);
            if (tex) {
                att->buf.tex = tex;
            }
        }
    }
    if (!tex) {
        fprintf(stderr,
                "MGL WARN: framebuffer attachment has no texture (target=0x%x)\n",
                (unsigned)att->textarget);
    }

    return tex;
}
