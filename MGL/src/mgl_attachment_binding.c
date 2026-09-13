/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_attachment_binding.c — bodies of -[MGLRenderer bindFramebufferTexture:
 * isDrawBuffer:] and -bindFramebufferAttachmentTextures, moved out of
 * MGLRenderer+VertexLayout.m.
 *
 * The context and the framebuffer come from the state areas, the attachment
 * texture from the C port of -framebufferAttachmentTexture:, and the Metal push
 * from the existing bind port; the NSLog diagnostics became fprintf on the same
 * stderr sink with the same fields.
 */

#include "mgl_attachment_binding.h"
#include "mgl_renderer_ports.h"   /* state areas, attachment texture */
#include "mgl_texture_bind.h"   /* mglRendererBindMTLTexture */

#include <stdint.h>
#include <stdio.h>

int mglRendererBindFramebufferTexture(void *renderer, FBOAttachment *attachment,
                                      int is_draw_buffer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    Texture *tex = mglRendererAttachmentTextureFor(areas.ctx, attachment);
    if (!tex) {
        /* Incomplete/missing attachment. Do not crash. */
        return 1;
    }

    if (is_draw_buffer) {
        tex->is_render_target = true;
    }

    if (!mglRendererBindMTLTexture(renderer, tex)) {
        return 0;
    }

    return 1;
}

int mglRendererBindFramebufferAttachmentTextures(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;

    /* MEMORY SAFETY: Validate context and framebuffer.  Only the pointer lower
     * bound is checked (high addresses are valid on macOS/arm64). */
    if (!ctx) {
        fprintf(stderr,
                "MGL ERROR: NULL context detected in bindFramebufferAttachmentTextures\n");
        return 0;
    }
    uintptr_t ctx_addr = (uintptr_t)ctx;
    if (ctx_addr < 0x1000) {
        fprintf(stderr,
                "MGL ERROR: Invalid context pointer detected in bindFramebufferAttachmentTextures: 0x%lx\n",
                (unsigned long)ctx_addr);
        return 0;
    }

    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!fbo) {
        fprintf(stderr,
                "MGL ERROR: NULL framebuffer detected in bindFramebufferAttachmentTextures\n");
        return 0;
    }
    uintptr_t fbo_addr = (uintptr_t)fbo;
    if (fbo_addr < 0x1000) {
        fprintf(stderr,
                "MGL ERROR: Invalid framebuffer pointer detected in bindFramebufferAttachmentTextures: 0x%lx\n",
                (unsigned long)fbo_addr);
        return 0;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (fbo->color_attachments[i].texture) {
            int isDrawBuffer = 1;
            if (mglRenderTargetIsRenderbuffer(
                    (uint32_t)fbo->color_attachments[i].textarget) &&
                fbo->color_attachments[i].buf.rbo) {
                isDrawBuffer = fbo->color_attachments[i].buf.rbo->is_draw_buffer;
            }

            if (!mglRendererBindFramebufferTexture(
                    renderer, &fbo->color_attachments[i], isDrawBuffer)) {
                DEBUG_PRINT("Failed Framebuffer Attachment\n");
                return 0;
            }
        }

        /* early out */
        if ((fbo->color_attachment_bitfield >> (i + 1)) == 0) {
            break;
        }
    }

    if (fbo->depth.texture) {
        if (!mglRendererBindFramebufferTexture(renderer, &fbo->depth, 1)) {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return 0;
        }
    }

    if (fbo->stencil.texture) {
        if (!mglRendererBindFramebufferTexture(renderer, &fbo->stencil, 1)) {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return 0;
        }
    }

    return 1;
}
