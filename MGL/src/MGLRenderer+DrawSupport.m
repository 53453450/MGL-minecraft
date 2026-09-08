/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+DrawSupport.m
// O1.6: resolve / raster / polygon-offset / ensure-encoder only
// (<400 LOC). id ports → mgl_draw_metal_port.m; stage host elsewhere.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+DrawSupportUtil.h"
#include "mgl_draw_mode.h"
#include "mgl_draw_encode.h"

@implementation MGLRenderer (Draw)

- (BOOL)resolveElementBufferForDraw:(const char *)label
                            context:(GLMContext)drawCtx
                           glBuffer:(Buffer **)glBufferOut
                          mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = getElementBuffer(drawCtx);
    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBufferForCommand:(const MGLDrawCommand *)cmd
                                  label:(const char *)label
                                context:(GLMContext)drawCtx
                               glBuffer:(Buffer **)glBufferOut
                              mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = NULL;
    if (cmd && cmd->element_buffer_name) {
        gl_element_buffer = mglRendererGetValidatedBuffer(drawCtx,
                                                          mglDrawCommandElementBuffer(drawCtx, cmd),
                                                          label ? label : "deferred indexed draw",
                                                          0);
        if (!gl_element_buffer) {
            return NO;
        }
    } else {
        gl_element_buffer = getElementBuffer(drawCtx);
    }

    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBuffer:(Buffer *)gl_element_buffer
                       label:(const char *)label
                     context:(GLMContext)drawCtx
                    glBuffer:(Buffer **)glBufferOut
                   mtlBuffer:(id *)mtlBufferOut
{
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no element array buffer is bound", label ? label : "indexed draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        }
        return NO;
    }

    if ([self processBuffer:gl_element_buffer] == false) {
        return NO;
    }

    id indexBuffer = (__bridge id)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: %s skipped because element buffer %u has no Metal buffer",
              label ? label : "indexed draw",
              gl_element_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_element_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indexBuffer;
    }
    return YES;
}

- (BOOL)resolveIndirectBufferForDraw:(const char *)label
                             context:(GLMContext)drawCtx
                            glBuffer:(Buffer **)glBufferOut
                           mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(drawCtx);
    if (!gl_indirect_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no draw indirect buffer is bound", label ? label : "indirect draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        }
        return NO;
    }

    if ([self processBuffer:gl_indirect_buffer] == false) {
        return NO;
    }

    id indirectBuffer = (__bridge id)(gl_indirect_buffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer",
              label ? label : "indirect draw",
              gl_indirect_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_indirect_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indirectBuffer;
    }
    return YES;
}

- (BOOL)prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label
{
    if (!drawCtx) {
        NSLog(@"MGL WARNING: %s skipped because context is NULL",
              label ? label : "indirect emulation");
        return NO;
    }

    /* The C draw-indirect frontends already flush pending command buffers before
     * dispatching into these Metal entry points. If processGLState has just
     * rebuilt a render encoder, keep it; a second flush can discard the fresh
     * pass and make state restoration fail for CPU-emulated indirect modes. */
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        NSLog(@"MGL WARNING: %s skipped because CPU-read synchronization left no render encoder",
              label ? label : "indirect emulation");
        return NO;
    }
    return YES;
}

- (BOOL)currentDrawRasterizationIsEmpty
{
    if (!ctx) {
        return NO;
    }

    GLint vx = MGL_STATE(ctx)->viewport[0];
    GLint vy = MGL_STATE(ctx)->viewport[1];
    GLint vw = MGL_STATE(ctx)->viewport[2];
    GLint vh = MGL_STATE(ctx)->viewport[3];

    NSUInteger passWidth = 0;
    NSUInteger passHeight = 0;
    mglRenderGetRenderTargetSizeOwner(
        _renderPassManager.state->renderPassStateOwner,
        (uint64_t *)&passWidth, (uint64_t *)&passHeight);
    if (passWidth == 0 || passHeight == 0) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id color = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i);
            if (color) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(color);
                passWidth = info.width;
                passHeight = info.height;
                break;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id depth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            if (depth) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(depth);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id stencil = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            if (stencil) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(stencil);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
    }


    return mglRenderRasterizationIsEmpty(
               vx, vy, vw, vh,
               (uint32_t)passWidth, (uint32_t)passHeight,
               MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
               MGL_STATE(ctx)->var.scissor_box[0],
               MGL_STATE(ctx)->var.scissor_box[1],
               MGL_STATE(ctx)->var.scissor_box[2],
               MGL_STATE(ctx)->var.scissor_box[3]) != 0;
}

- (void)applyPolygonOffsetForDrawMode:(GLenum)mode
{
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return;
    }


    MGLRenderPolygonOffsetDecision decision = {0};
    mglRenderPolygonOffsetDecision(
        (uint32_t)mode,
        ctx ? 1 : 0,
        mglDrawModeProducesPolygons(mode) ? 1 : 0,
        (uint32_t)(ctx ? MGL_STATE(ctx)->var.polygon_mode : 0u),
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_point) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_line) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_fill) ? 1 : 0,
        &decision);
    uint32_t triangleFillMode = decision.triangle_fill_mode ? 1u : 0u;
    if (decision.needs_polygon_mode_repair) {
        uint32_t repaired = mglRenderPolygonModeOrFill(
            (uint32_t)MGL_STATE(ctx)->var.polygon_mode);
        mglLogRenderStateRepair("polygon_mode", MGL_STATE(ctx)->var.polygon_mode,
                                (GLenum)repaired);
        MGL_STATE(ctx)->var.polygon_mode = (GLenum)repaired;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];

    BOOL enableDepthBias = decision.enable_depth_bias != 0;

    if (enableDepthBias) {
        float _bias = MGL_STATE(ctx)->var.polygon_offset_units;
        float _slope = MGL_STATE(ctx)->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    } else {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            0.0f, 0.0f, 0.0f);
    }
}

- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode
{
    return ctx &&
           mglRenderDrawModeFullyCulled(
               MGL_STATE(ctx)->caps.cull_face ? 1 : 0,
               (uint32_t)MGL_STATE(ctx)->var.cull_face_mode,
               mglDrawModeProducesPolygons(mode) ? 1 : 0) != 0;
}

- (BOOL)ensureRasterEncoderForDraw
{
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }
    [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_DRAW];
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return NO;
    }
    if (!_pipelineCache.state->pipelineState) {
        return NO;
    }

    uint32_t rpColor0Format = 0u;
    uint32_t rpDepthFormat = 0u;
    uint32_t rpStencilFormat = 0u;
    MGLRenderPassAttachmentState colorAttachment = {0};
    MGLRenderPassAttachmentState depthAttachment = {0};
    MGLRenderPassAttachmentState stencilAttachment = {0};
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, &colorAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, &depthAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, &stencilAttachment);
    id rpColor0 = (__bridge id)colorAttachment.texture;
    id rpDepth = (__bridge id)depthAttachment.texture;
    id rpStencil = (__bridge id)stencilAttachment.texture;
    MGLRenderTextureInfo textureInfo = {0};
    if (rpColor0 && mglRenderGetTextureInfo(
            (__bridge void *)rpColor0, &textureInfo) == 0) {
        rpColor0Format = textureInfo.pixel_format;
    }
    if (rpDepth && mglRenderGetTextureInfo(
            (__bridge void *)rpDepth, &textureInfo) == 0) {
        rpDepthFormat = textureInfo.pixel_format;
    }
    if (rpStencil && mglRenderGetTextureInfo(
            (__bridge void *)rpStencil, &textureInfo) == 0) {
        rpStencilFormat = textureInfo.pixel_format;
    }

    const BOOL colorMismatch =
        (_pipelineCache.state->pipelineColor0Format != 0u &&
         rpColor0Format != 0u &&
         _pipelineCache.state->pipelineColor0Format != rpColor0Format);
    const BOOL depthMismatch =
        (_pipelineCache.state->pipelineDepthFormat != rpDepthFormat);
    const BOOL stencilMismatch =
        (_pipelineCache.state->pipelineStencilFormat != rpStencilFormat);
    if (colorMismatch || depthMismatch || stencilMismatch) {
        return NO;
    }
    if (mglRenderSetRenderPipelineStateForOwner(
            _renderPassManager.state->currentRenderEncoderOwner,
            _pipelineCache.state->pipelineState) != 0) {
        return NO;
    }
    mglRenderBindingSetPipelineState(_bindingStateOwner,
                                     _pipelineCache.state->pipelineState);
    MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
    return YES;
}

@end
