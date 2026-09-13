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
#include "mgl_render.h"           /* command-buffer snapshot, MDI scratch owner */
#include "mgl_renderer_backend.h" /* mglRendererBackendGetDevice */

#include <stdint.h>
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

/* The manager's command state, reached through the state areas.  Replaces the
 * former mglRendererCommandStatePort wrapper (the areas already carry the
 * pointer, so the shim does not need an entry point of its own). */
const MGLCommandState *mglRendererCommandStateFor(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.command;
}

/* Body of the former -[MGLRenderPassManager mdiArgumentScratchBufferWithDevice:
 * length:offset:].  The arena itself is C++ (mglRenderAllocateMDIScratch) and
 * the owner pointer is a field of the command state, so no Objective-C message
 * is involved: the returned buffer is borrowed, exactly as before. */
void *mglRendererMdiScratchBuffer(void *renderer, uint64_t length,
                                  uint64_t *offset_out)
{
    if (offset_out) {
        *offset_out = 0;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || length == 0 ||
        !mglRendererBackendGetDevice(areas.backend)) {
        return NULL;
    }

    MGLRenderCommandBufferState commandBufferState = {0};
    if (!mglRenderCommandBufferOwnerHasState(cs->currentCommandBufferOwner,
                                             &commandBufferState)) {
        return NULL;
    }

    if (!cs->mdiArgsScratchOwner &&
        mglRenderCreateMDIScratchOwner(&cs->mdiArgsScratchOwner) != 0) {
        return NULL;
    }
    void *buffer = NULL;
    uint64_t offset = 0;
    uint64_t capacity = 0;
    if (mglRenderAllocateMDIScratch(cs->mdiArgsScratchOwner, length, 256u,
                                    &buffer, &offset, &capacity) != 0 ||
        !buffer) {
        return NULL;
    }
    if (offset_out) {
        *offset_out = offset;
    }
    return buffer;
}
