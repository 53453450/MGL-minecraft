/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_readback_clear.c — pending clears applied before readback/blit,
 * moved out of MGLRenderer+Texture.m (P0-1).  Only the diagnostics changed
 * sink (NSLog -> stderr).
 */

#include "mgl_texture_readback_clear.h"
#include "mgl_renderer_ports.h"   /* command state */
#include "mgl_render.h"
#include "mgl_texture_compat.h"  /* mglMarkTextureLevelRenderTargetWrittenImpl */
#include "mgl_sync.h"          /* mglMetalAttachmentSubresourceForAttachment */

#include <stdio.h>

void mglTextureApplyPendingFBODepthClearForReadback(void *renderer,
                                                    Framebuffer *fbo,
                                                    FBOAttachment *attachment,
                                                    Texture *texture_obj,
                                                    void *mtl_texture)
{
    if (!fbo || !attachment || !mtl_texture ||
        !mglRenderClearMaskHasDepth((uint32_t)attachment->clear_bitmask)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    if (mglRenderEncodeDepthClearForCommandBufferOwner(
            mglRendererCommandStateFor(renderer)->currentCommandBufferOwner,
            mtl_texture, subresource.level,
            subresource.slice, subresource.depthPlane,
            attachment->clear_color[0]) == 0) {
        attachment->clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearDepth(
                (uint32_t)attachment->clear_bitmask);
        mglMarkTextureLevelRenderTargetWrittenImpl(texture_obj, attachment->level,
                                                       __func__, __LINE__);
    } else {
        fprintf(stderr, "MGL WARNING: C++ readPixels depth clear failed fbo=%u\n",
                (unsigned)fbo->name);
    }
}

void mglTextureApplyPendingFBOColorClearForReadback(void *renderer,
                                                    Framebuffer *fbo,
                                                    FBOAttachment *attachment,
                                                    Texture *texture_obj,
                                                    void *mtl_texture,
                                                    uint32_t attachment_enum)
{
    (void)attachment_enum;
    if (!fbo || !attachment || !mtl_texture ||
        !mglRenderClearMaskHasColor((uint32_t)attachment->clear_bitmask)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    if (mglRenderEncodeColorClearForCommandBufferOwner(
            mglRendererCommandStateFor(renderer)->currentCommandBufferOwner,
            mtl_texture, subresource.level,
            subresource.slice, subresource.depthPlane,
            attachment->clear_color[0], attachment->clear_color[1],
            attachment->clear_color[2], attachment->clear_color[3]) == 0) {
        attachment->clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearColor(
                (uint32_t)attachment->clear_bitmask);
        mglMarkTextureLevelRenderTargetWrittenImpl(texture_obj, attachment->level,
                                                       __func__, __LINE__);
        return;
    }
    fprintf(stderr, "MGL WARNING: C++ readPixels FBO color clear failed fbo=%u\n",
            (unsigned)fbo->name);
}

void mglTextureApplyPendingDefaultDepthClear(void *renderer, void *mtl_texture)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    if (!ctx || !mtl_texture ||
        !mglRenderClearMaskHasDepth((uint32_t)ctx->active_state->default_fbo_clear_bitmask)) {
        return;
    }

    if (mglRenderEncodeDepthClearForCommandBufferOwner(
            mglRendererCommandStateFor(renderer)->currentCommandBufferOwner,
            mtl_texture, 0, 0, 0,
            ctx->active_state->var.depth_clear_value) == 0) {
        ctx->active_state->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearDepth(
                (uint32_t)ctx->active_state->default_fbo_clear_bitmask);
    } else {
        fprintf(stderr, "MGL WARNING: C++ default depth clear failed\n");
    }
}

void mglTextureApplyPendingDefaultColorClear(void *renderer, void *mtl_texture)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    if (!ctx || !mtl_texture ||
        !mglRenderClearMaskHasColor((uint32_t)ctx->active_state->default_fbo_clear_bitmask)) {
        return;
    }

    if (mglRenderEncodeColorClearForCommandBufferOwner(
            mglRendererCommandStateFor(renderer)->currentCommandBufferOwner,
            mtl_texture, 0, 0, 0,
            ctx->active_state->default_clear_color[0],
            ctx->active_state->default_clear_color[1],
            ctx->active_state->default_clear_color[2],
            ctx->active_state->default_clear_color[3]) == 0) {
        ctx->active_state->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearColor(
                (uint32_t)ctx->active_state->default_fbo_clear_bitmask);
        return;
    }
    fprintf(stderr, "MGL WARNING: C++ default framebuffer color clear failed\n");
}
