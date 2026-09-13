/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_draw_support.c — bodies of -[MGLRenderer resolveIndirectBufferForDraw:],
 * -currentDrawRasterizationIsEmpty, -applyPolygonOffsetForDrawMode: and
 * -currentDrawModeIsFullyCulled:.
 *
 * Every step was already C (mglRenderRasterizationIsEmpty,
 * mglRenderPolygonOffsetDecision, mglRenderDrawModeFullyCulled,
 * mglRenderBinding*ForOwner, mglRendererProcessBuffer); the Objective-C state
 * they read arrives through the state areas now.  The NSLog diagnostics became
 * fprintf on the same stderr sink with the same fields.
 */

#include "mgl_draw_support.h"
#include "mgl_draw_issue.h"
#include "mgl_draw_tess.h"    /* mglResolveProgramForStageFromState */   /* mglDrawHostRunVertexCaptureIndexed */
#include "mgl_renderer_ports.h"   /* state areas, mglRendererProcessBuffer */
#include "mgl_render.h"
#include "mgl_state_compat.h"     /* mglLogRenderStateRepair */
#include "error.h"                /* mglDispatchError */

#include <stdio.h>

/* Defined in MGLRenderer.m; declared next to the definition in the
 * Objective-C MGLRenderer+Draw_Private.h, which C cannot include. */
extern Buffer *getIndirectBuffer(GLMContext ctx);

int mglDrawResolveIndirectBuffer(void *renderer, const char *label,
                                 GLMContext ctx, Buffer **gl_out,
                                 void **mtl_out)
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(ctx);
    if (!gl_indirect_buffer) {
        fprintf(stderr,
                "MGL WARNING: %s skipped because no draw indirect buffer is bound\n",
                label ? label : "indirect draw");
        if (ctx) {
            mglDispatchError(ctx, label ? label : "resolveIndirectBufferForDraw",
                             (GLenum)mglRenderErrorInvalidOperation());
        }
        return 0;
    }

    if (!mglRendererProcessBuffer(renderer, gl_indirect_buffer)) {
        return 0;
    }

    void *indirectBuffer = gl_indirect_buffer->data.mtl_data;
    if (!indirectBuffer) {
        fprintf(stderr,
                "MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer\n",
                label ? label : "indirect draw",
                gl_indirect_buffer->name);
        return 0;
    }

    if (gl_out) {
        *gl_out = gl_indirect_buffer;
    }
    if (mtl_out) {
        *mtl_out = indirectBuffer;
    }
    return 1;
}

int mglDrawRasterizationIsEmpty(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *cs = areas.command;
    if (!ctx || !cs) {
        return 0;
    }

    GLint vx = ctx->active_state->viewport[0];
    GLint vy = ctx->active_state->viewport[1];
    GLint vw = ctx->active_state->viewport[2];
    GLint vh = ctx->active_state->viewport[3];

    uint64_t passWidth = 0;
    uint64_t passHeight = 0;
    mglRenderGetRenderTargetSizeOwner(cs->renderPassStateOwner, &passWidth,
                                      &passHeight);
    if (passWidth == 0 || passHeight == 0) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            void *color = mglRenderGetRenderPassAttachmentTextureOwner(
                cs->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i);
            if (color) {
                MGLRenderTextureInfo info = {0};
                if (mglRenderGetTextureInfo(color, &info) != 0) {
                    continue;
                }
                passWidth = info.width;
                passHeight = info.height;
                break;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            void *depth = mglRenderGetRenderPassAttachmentTextureOwner(
                cs->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            if (depth) {
                MGLRenderTextureInfo info = {0};
                if (mglRenderGetTextureInfo(depth, &info) == 0) {
                    passWidth = info.width;
                    passHeight = info.height;
                }
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            void *stencil = mglRenderGetRenderPassAttachmentTextureOwner(
                cs->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            if (stencil) {
                MGLRenderTextureInfo info = {0};
                if (mglRenderGetTextureInfo(stencil, &info) == 0) {
                    passWidth = info.width;
                    passHeight = info.height;
                }
            }
        }
    }

    return mglRenderRasterizationIsEmpty(
               vx, vy, vw, vh,
               (uint32_t)passWidth, (uint32_t)passHeight,
               ctx->active_state->caps.scissor_test ? 1 : 0,
               ctx->active_state->var.scissor_box[0],
               ctx->active_state->var.scissor_box[1],
               ctx->active_state->var.scissor_box[2],
               ctx->active_state->var.scissor_box[3]) != 0;
}

void mglDrawApplyPolygonOffset(void *renderer, uint32_t mode)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *cs = areas.command;
    if (!ctx || !cs) {
        return;
    }
    if (mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }

    MGLRenderPolygonOffsetDecision decision = {0};
    mglRenderPolygonOffsetDecision(
        mode,
        ctx ? 1 : 0,
        mglRenderDrawModeProducesPolygons((uint64_t)mode) ? 1 : 0,
        (uint32_t)ctx->active_state->var.polygon_mode,
        ctx->active_state->caps.polygon_offset_point ? 1 : 0,
        ctx->active_state->caps.polygon_offset_line ? 1 : 0,
        ctx->active_state->caps.polygon_offset_fill ? 1 : 0,
        &decision);
    uint32_t triangleFillMode = decision.triangle_fill_mode ? 1u : 0u;
    if (decision.needs_polygon_mode_repair) {
        uint32_t repaired =
            mglRenderPolygonModeOrFill((uint32_t)ctx->active_state->var.polygon_mode);
        mglLogRenderStateRepair("polygon_mode", ctx->active_state->var.polygon_mode,
                                (GLenum)repaired);
        ctx->active_state->var.polygon_mode = (GLenum)repaired;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
    }

    /* -[MGLRenderer setTriangleFillModeIfNeeded:] is a two-line forward to the
     * owner-aware binding call. */
    if (areas.binding_state_owner) {
        mglRenderBindingSetTriangleFillForOwner(*areas.binding_state_owner,
                                                cs->currentRenderEncoderOwner,
                                                triangleFillMode);
    }

    int enableDepthBias = decision.enable_depth_bias != 0;

    if (enableDepthBias) {
        float bias = ctx->active_state->var.polygon_offset_units;
        float slope = ctx->active_state->var.polygon_offset_factor;
        float clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            areas.binding_state_owner ? *areas.binding_state_owner : NULL,
            cs->currentRenderEncoderOwner, bias, clamp, slope);
    } else {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            areas.binding_state_owner ? *areas.binding_state_owner : NULL,
            cs->currentRenderEncoderOwner, 0.0f, 0.0f, 0.0f);
    }
}

int mglDrawModeIsFullyCulled(void *renderer, uint32_t mode)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    if (!ctx) {
        return 0;
    }
    return mglRenderDrawModeFullyCulled(
               ctx->active_state->caps.cull_face ? 1 : 0,
               (uint32_t)ctx->active_state->var.cull_face_mode,
               mglRenderDrawModeProducesPolygons((uint64_t)mode) ? 1 : 0) != 0;
}

/* Body of the former -[MGLRenderer fragmentNeedsPerSampleMSValuesForContext:]. */
int mglDrawFragmentNeedsPerSampleMSValues(GLMContext ctx)
{
    Program *fp = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    if (!fp) {
        return 0;
    }
    return mglRenderFragmentNeedsPerSampleMSValues(fp) != 0;
}
