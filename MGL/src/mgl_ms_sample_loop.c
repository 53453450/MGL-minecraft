/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* Bodies of -[MGLRenderer runEmulatedMSSampleDrawLoopIfNeeded:drawOnce:] and
 * -broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded: (P0-1).  The block became a
 * function pointer plus a context, and the three _mgl* ivars travel through the
 * shell forwarders. */

#include "mgl_ms_sample_loop.h"
#include "mgl_renderer_ports.h"
#include "mgl_render_pass_manager_ops.h"  /* mglRendererEndRenderEncodingLocked */
#include "mgl_draw_support.h"          /* mglDrawEmulatedMSColor0Texture, … */
#include "mgl_render.h"
#include "mgl_metal_ref.h"
#include "mgl_sync.h"                  /* mglMetalAttachmentSubresourceForAttachment */

#include <stdint.h>

extern int mglPlatformShellMSSampleInLoop(void *renderer);
extern void mglPlatformShellSetMSSampleState(void *renderer, int in_loop,
                                             int32_t forced, int32_t offset);
extern int mglPlatformShellNewCommandBuffer(void *renderer);

int mglRendererRunEmulatedMSSampleDrawLoopIfNeeded(
    void *renderer, GLMContext ctx, void (*draw_once)(void *), void *draw_ctx)
{
    if (mglPlatformShellMSSampleInLoop(renderer) || !draw_once) {
        return 0;
    }
    Texture *tex = mglDrawEmulatedMSColor0Texture(ctx);
    if (!tex) {
        return 0;
    }
    if (!mglDrawFragmentNeedsPerSampleMSValues(ctx)) {
        return 0;
    }

    const GLint samples = tex->samples > 1 ? (GLint)tex->samples : 1;
    mglPlatformShellSetMSSampleState(renderer, 1, 0, 0);
    for (GLint s = 0; s < samples; s++) {
        mglPlatformShellSetMSSampleState(renderer, 1, s, s);
        mglRendererEndRenderEncodingLocked(renderer);
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_FBO);
        Framebuffer *fbo = ctx->active_state->framebuffer;
        if (fbo) {
            fbo->dirty_bits |= DIRTY_FBO_BINDING;
        }
        draw_once(draw_ctx);
    }
    mglPlatformShellSetMSSampleState(renderer, 0, 0, 0);
    return 1;
}

void mglRendererBroadcastEmulatedMSSamplePlanesAfterDrawIfNeeded(void *renderer,
                                                                 GLMContext ctx)
{
    if (mglPlatformShellMSSampleInLoop(renderer)) {
        return;
    }
    Texture *tex = mglDrawEmulatedMSColor0Texture(ctx);
    if (!tex || !tex->mtl_data) {
        return;
    }
    if (mglDrawFragmentNeedsPerSampleMSValues(ctx)) {
        /* Per-sample draws already filled each plane. */
        return;
    }

    const uint64_t samples = tex->samples > 1u ? (uint64_t)tex->samples : 1u;
    if (samples <= 1u) {
        return;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || !ctx->active_state->framebuffer) {
        return;
    }
    MGLMetalAttachmentSubresource sub = mglMetalAttachmentSubresourceForAttachment(
        &ctx->active_state->framebuffer->color_attachments[0]);
    const uint64_t baseSlice = sub.slice;
    const uint64_t level = sub.level;
    MGLRenderTextureInfo info = {0};
    (void)mglRenderGetTextureInfo(tex->mtl_data, &info);
    if (info.width == 0u || info.height == 0u) {
        return;
    }
    if (baseSlice + samples > info.array_length) {
        return;
    }

    mglRendererEndRenderEncodingLocked(renderer);
    if (!cs->currentCommandBufferOwner &&
        !mglPlatformShellNewCommandBuffer(renderer)) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(cs->currentCommandBufferOwner);
    if (!blit) {
        return;
    }
    for (uint64_t s = 1u; s < samples; s++) {
        (void)mglRenderBlitCopyTexture(
            blit, tex->mtl_data, baseSlice, level, 0u, 0u, 0u, info.width,
            info.height, 1u, tex->mtl_data, baseSlice + s, level, 0u, 0u, 0u);
    }
    (void)mglRenderEndBlitEncoder(blit);
}
