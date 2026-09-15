/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_vertex_layout.c — bodies of -[MGLRenderer generateVertexDescriptorState:]
 * and -updateBlendStateCache, moved out of MGLRenderer+VertexLayout.m.
 *
 * The GL state comes from the state areas (areas.ctx->active_state), the
 * tessellation inputs and the pipeline-cache object from the extra areas fields
 * the shell fills, and the blend upload goes through the cache's own setter
 * behind the function pointer in the areas - so no new port is involved.
 */

#include "mgl_vertex_layout.h"
#include "mgl_renderer_ports.h"
#include "mgl_render.h"
#include "mgl_state_compat.h"   /* mglLogRenderStateRepair */
#include "mgl_draw_tess.h"

#include <stdio.h>

/* Defined in the C++/C back end; declared in the Objective-C
 * MGLRenderer+VertexLayout_Private.h, which C cannot include. */
extern bool mglRenderGenerateVertexDescriptorState(
    GLMContext ctx, MGLRenderPipelineDescriptorState *state, int nativeTESActive,
    const Program *nativeTESProgram, uint32_t tcsOutputStride,
    int absoluteVertexBindingOffsets, const char *where);

int mglRendererGenerateVertexDescriptorState(
    void *renderer, MGLRenderPipelineDescriptorState *state)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    return mglRenderGenerateVertexDescriptorState(
               areas.ctx,
               state,
               (int)areas.tess_native_tes_active,
               (const Program *)areas.tess_native_tes_program,
               areas.tess_tcs_output_stride,
               (int)(areas.batching ? areas.batching->absoluteVertexBindingOffsets : 0u),
               "generateVertexDescriptorState:") ? 1 : 0;
}

void mglRendererUpdateBlendStateCache(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMState *st = areas.ctx ? areas.ctx->active_state : NULL;
    if (!st) {
        return;
    }

    bool repairedState = false;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        uint32_t srcRgb = (uint32_t)st->var.blend_src_rgb[i];
        if (mglRenderRepairBlendSrcFactor(&srcRgb)) {
            mglLogRenderStateRepair("blend_src_rgb", st->var.blend_src_rgb[i], (GLenum)srcRgb);
            st->var.blend_src_rgb[i] = (GLenum)srcRgb;
            repairedState = true;
        }
        uint32_t srcAlpha = (uint32_t)st->var.blend_src_alpha[i];
        if (mglRenderRepairBlendSrcFactor(&srcAlpha)) {
            mglLogRenderStateRepair("blend_src_alpha", st->var.blend_src_alpha[i], (GLenum)srcAlpha);
            st->var.blend_src_alpha[i] = (GLenum)srcAlpha;
            repairedState = true;
        }
        uint32_t dstRgb = (uint32_t)st->var.blend_dst_rgb[i];
        if (mglRenderRepairBlendDstFactor(&dstRgb)) {
            mglLogRenderStateRepair("blend_dst_rgb", st->var.blend_dst_rgb[i], (GLenum)dstRgb);
            st->var.blend_dst_rgb[i] = (GLenum)dstRgb;
            repairedState = true;
        }
        uint32_t dstAlpha = (uint32_t)st->var.blend_dst_alpha[i];
        if (mglRenderRepairBlendDstFactor(&dstAlpha)) {
            mglLogRenderStateRepair("blend_dst_alpha", st->var.blend_dst_alpha[i], (GLenum)dstAlpha);
            st->var.blend_dst_alpha[i] = (GLenum)dstAlpha;
            repairedState = true;
        }
        uint32_t eqRgb = (uint32_t)st->var.blend_equation_rgb[i];
        if (mglRenderRepairBlendEquation(&eqRgb)) {
            mglLogRenderStateRepair("blend_equation_rgb", st->var.blend_equation_rgb[i], (GLenum)eqRgb);
            st->var.blend_equation_rgb[i] = (GLenum)eqRgb;
            repairedState = true;
        }
        uint32_t eqAlpha = (uint32_t)st->var.blend_equation_alpha[i];
        if (mglRenderRepairBlendEquation(&eqAlpha)) {
            mglLogRenderStateRepair("blend_equation_alpha", st->var.blend_equation_alpha[i], (GLenum)eqAlpha);
            st->var.blend_equation_alpha[i] = (GLenum)eqAlpha;
            repairedState = true;
        }

        uint32_t colorMask_i = mglRenderColorWriteMaskFromChannels(
            st->caps.use_color_mask[i] ? 1 : 0,
            st->var.color_writemask[i][0] ? 1 : 0,
            st->var.color_writemask[i][1] ? 1 : 0,
            st->var.color_writemask[i][2] ? 1 : 0,
            st->var.color_writemask[i][3] ? 1 : 0);
        colorMask_i = mglRenderForceDefaultFBOAlphaWrite(
            i, st->framebuffer ? 1 : 0, colorMask_i);
        uint32_t srcRgbF = 0u, srcAlphaF = 0u, dstRgbF = 0u, dstAlphaF = 0u;
        uint32_t rgbOp = 0u, alphaOp = 0u;
        (void)mglRenderBlendFactorFromGL((uint32_t)st->var.blend_src_rgb[i], &srcRgbF);
        (void)mglRenderBlendFactorFromGL((uint32_t)st->var.blend_src_alpha[i], &srcAlphaF);
        (void)mglRenderBlendFactorFromGL((uint32_t)st->var.blend_dst_rgb[i], &dstRgbF);
        (void)mglRenderBlendFactorFromGL((uint32_t)st->var.blend_dst_alpha[i], &dstAlphaF);
        (void)mglRenderBlendOperationFromGL((uint32_t)st->var.blend_equation_rgb[i], &rgbOp);
        (void)mglRenderBlendOperationFromGL((uint32_t)st->var.blend_equation_alpha[i], &alphaOp);

        MGLRenderPipelineBlendState blend = {
            .source_rgb_factor = srcRgbF,
            .destination_rgb_factor = dstRgbF,
            .source_alpha_factor = srcAlphaF,
            .destination_alpha_factor = dstAlphaF,
            .rgb_operation = rgbOp,
            .alpha_operation = alphaOp,
            .color_write_mask = colorMask_i,
        };
        if (areas.pipeline_cache_set_blend) {
            (void)areas.pipeline_cache_set_blend(areas.pipeline_cache_object,
                                                 (uint32_t)i, &blend);
        }
    }
    if (repairedState) {
        mglMarkStateDirtyBits(areas.ctx->active_state,
                              DIRTY_RENDER_STATE | DIRTY_ALPHA_STATE);
    }
}
