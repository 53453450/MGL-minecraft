/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+DrawStageHost.m
// A1 / O1.4 residual2: thin ObjC glue (kept; not empty). Capture/validate/cull
// encode → C++ (mgl_draw_tess / mgl_draw_cull / mgl_draw_issue). GS Metal
// expansion HostOps → mgl_draw_metal_port.m → mgl_draw_gs_metal.cpp. This
// category keeps bindCull VAO ports + MS sample loop + one-line wrappers.
// Do not open new thick ObjC categories here.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+DrawSupportUtil.h"
#include "mgl_draw_cull.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_binding_state_ops.h"
#include "mgl_draw_support.h"  /* fragment-needs-per-sample-MS */
#include "mgl_draw_issue.h"
#include "mgl_draw_tess.h"
#include "mgl_draw_encode.h"
#include "mgl_index_buffer.h"
#include "mgl_render.h"
#include "glm_limits.h"
#include "mgl_buffer_query.h"

@implementation MGLRenderer (DrawStageHost)











- (BOOL)runEmulatedMSSampleDrawLoopIfNeeded:(GLMContext)glm_ctx
                                   drawOnce:(void (^)(void))drawOnce
{
    if (_mglInMSSampleDrawLoop || !drawOnce) return NO;
    Texture *tex = mglDrawEmulatedMSColor0Texture(glm_ctx);
    if (!tex) return NO;
    if (!mglDrawFragmentNeedsPerSampleMSValues(glm_ctx)) return NO;

    const GLint samples = (GLint)MAX(tex->samples, 1);
    _mglInMSSampleDrawLoop = YES;
    for (GLint s = 0; s < samples; s++) {
        _mglForcedMSSampleId = s;
        _mglMSSamplePlaneOffset = s;
        mglRendererEndRenderEncodingLocked((__bridge void *)self);
        mglMarkStateDirtyBits(MGL_STATE(glm_ctx), DIRTY_FBO);
        Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
        if (fbo) {
            fbo->dirty_bits |= DIRTY_FBO_BINDING;
        }
        drawOnce();
    }
    _mglForcedMSSampleId = 0;
    _mglMSSamplePlaneOffset = 0;
    _mglInMSSampleDrawLoop = NO;
    return YES;
}

- (void)broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:(GLMContext)glm_ctx
{
    if (_mglInMSSampleDrawLoop) return;
    Texture *tex = mglDrawEmulatedMSColor0Texture(glm_ctx);
    if (!tex || !tex->mtl_data) return;
    if (mglDrawFragmentNeedsPerSampleMSValues(glm_ctx)) {
        /* Per-sample draws already filled each plane. */
        return;
    }

    const NSUInteger samples = MAX((NSUInteger)tex->samples, 1u);
    if (samples <= 1u) return;

    MGLMetalAttachmentSubresource sub =
        mglMetalAttachmentSubresourceForAttachment(
            &MGL_STATE(glm_ctx)->framebuffer->color_attachments[0]);
    const NSUInteger baseSlice = sub.slice;
    const NSUInteger level = sub.level;
    MGLRenderTextureInfo info = {0};
    (void)mglRenderGetTextureInfo(tex->mtl_data, &info);
    if (info.width == 0u || info.height == 0u) return;
    if (baseSlice + samples > info.array_length) return;

    mglRendererEndRenderEncodingLocked((__bridge void *)self);
    if (!_renderPassManager.state->currentCommandBufferOwner &&
        ![self newCommandBufferLocked]) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(
        _renderPassManager.state->currentCommandBufferOwner);
    if (!blit) return;
    for (NSUInteger s = 1u; s < samples; s++) {
        (void)mglRenderBlitCopyTexture(
            blit, tex->mtl_data, baseSlice, level, 0u, 0u, 0u,
            info.width, info.height, 1u,
            tex->mtl_data, baseSlice + s, level, 0u, 0u, 0u);
    }
    (void)mglRenderEndBlitEncoder(blit);
}



@end
