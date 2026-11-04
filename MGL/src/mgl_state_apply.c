/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_state_apply.c — dirty-domain apply shared by Pass / Flush (ARCH-016).
 */

#include "mgl_state_apply.h"

#include "mgl_attachment_binding.h"
#include "mgl_buffer_map.h"
#include "mgl_encode_context.h"
#include "mgl_frame_activity.h"
#include "mgl_render_api_query.h"
#include "mgl_render_encoder_ops.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_pass_sync_ops.h"
#include "mgl_renderer_binding_ops.h"
#include "mgl_renderer_ports.h"
#include "mgl_stage_encode_drivers.h"
#include "mgl_trace_log.h"

#include "mgl_pso_build_ops.h"

#include <stdint.h>

extern Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx,
                                                      const char *where);

static GLMState *mglSaStateView(const MGLRendererStateView *view)
{
    return view->ctx ? view->ctx->active_state : NULL;
}

bool mglStateApplyDirtyDomains(void *renderer, int draw_command,
                               MGLResourceSyncWork *work)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    int fboBindingDirty = 0;
    if ((mglSaStateView(&areas)->dirty_bits & (DIRTY_STATE | DIRTY_FBO)) ==
        (DIRTY_STATE | DIRTY_FBO)) {
        Framebuffer *framebuffer =
            mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO");
        if (framebuffer && (framebuffer->dirty_bits & DIRTY_FBO_BINDING)) {
            fboBindingDirty = 1;
        }
    }
    MGLDirtyDomainPlan plan = {0};
    if (mglRenderPlanDirtyDomains(
            mglSaStateView(&areas)->dirty_bits, draw_command ? 1 : 0,
            areas.pipeline_cache->pipelineState != NULL ? 1 : 0, fboBindingDirty,
            &plan) != 0) {
        return false;
    }

    bool deferredBufferMapForPipelineBuild = plan.defer_buffer_map;
    if (plan.has_dirty)
    {
        if (mglSaStateView(&areas)->dirty_bits &
            (DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER |
             DIRTY_PROGRAM | DIRTY_IMAGE_UNIT_STATE)) {
            mglRendererBindingMarkSampledNeedsBind(renderer);
        }
        if (plan.sync_render_pass)
        {
            RETURN_FALSE_ON_FAILURE(
                mglRenderPassSyncRenderPassStateForContext(renderer, ctx));
        }

        if (plan.bind_fbo_attachments)
        {
            RETURN_FALSE_ON_FAILURE(
                mglRendererBindFramebufferAttachmentTextures(renderer));
            Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(
                ctx, "processGLState.dirtyStateFBO.afterBind");
            if (framebuffer) {
                framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
            }
        }

        if (mglSaStateView(&areas)->dirty_bits & DIRTY_STATE)
        {
            mglSaStateView(&areas)->dirty_bits &= ~DIRTY_STATE;
        }

        if (plan.remap_buffers)
        {
            if (plan.defer_buffer_map) {
                static uint64_t s_deferredMapCount = 0;
                s_deferredMapCount++;
                if (s_deferredMapCount <= 16 || (s_deferredMapCount % 1000ull) == 0ull) {
                    mglTraceLog(
                        "MGL DRAW SKIP: pipelineState is nil (deferring buffer mapping, occurrence=%llu)",
                        (unsigned long long)s_deferredMapCount);
                }
            } else {
                RETURN_FALSE_ON_FAILURE(mglRendererMapBuffersToMTL(renderer));
                if (work) work->mappedBuffers = true;
            }

            mglSaStateView(&areas)->dirty_bits &= ~DIRTY_BUFFER_BASE_STATE;
        }

        if (plan.bind_textures)
        {
            RETURN_FALSE_ON_FAILURE(
                mglRendererBindActiveTexturesToMTL(renderer, ctx));
            if (work) work->boundActiveTextures = true;

            mglSaStateView(&areas)->dirty_bits &=
                ~(DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER);
        }

        if (plan.vao_path)
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(
                renderer, &mglSaStateView(&areas)->vertex_buffer_map_list));
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(
                renderer, &mglSaStateView(&areas)->fragment_buffer_map_list));
            if (work) work->updatedBaseLists = true;

            if (mglRenderEncoderOwnerHasCurrent(
                    areas.command->currentRenderEncoderOwner) != 1) {
                RETURN_FALSE_ON_FAILURE(
                    mglRenderPassNewRenderEncoderLockedWithReason(
                        renderer, MGL_ENC_REASON_VAO));
            }

            mglRenderPassUpdateCurrentRenderEncoder(renderer);

            mglSaStateView(&areas)->dirty_bits &= ~DIRTY_RENDER_STATE;
        }
        else if (plan.buffer_path)
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(
                renderer, &mglSaStateView(&areas)->vertex_buffer_map_list));
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(
                renderer, &mglSaStateView(&areas)->fragment_buffer_map_list));
            if (work) work->updatedBaseLists = true;

            mglSaStateView(&areas)->dirty_bits &= ~DIRTY_BUFFER;
        }
        else if (plan.render_state_path)
        {
            if (mglRenderEncoderOwnerHasCurrent(
                    areas.command->currentRenderEncoderOwner) != 1)
            {
                RETURN_FALSE_ON_FAILURE(
                    mglRenderPassNewRenderEncoderLockedWithReason(
                        renderer, MGL_ENC_REASON_RS));
            }

            mglRenderPassUpdateCurrentRenderEncoder(renderer);

            mglSaStateView(&areas)->dirty_bits &= ~DIRTY_RENDER_STATE;
        }

        if (plan.sync_pipeline)
        {
            RETURN_FALSE_ON_FAILURE(mglRenderPassSyncPipelineState(
                renderer, deferredBufferMapForPipelineBuild));
        }

        mglSaStateView(&areas)->dirty_bits = 0;
    }
    else
    {
        MGLEncodeContext encCtx = {
            .render_encoder_owner = areas.command->currentRenderEncoderOwner,
        };

        if (mglRendererCheckForDirtyBufferData(
                renderer, &mglSaStateView(&areas)->vertex_buffer_map_list))
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(
                renderer, &mglSaStateView(&areas)->vertex_buffer_map_list));

            RETURN_FALSE_ON_FAILURE(
                mglStageEncodeBindVertexBuffers(renderer, &encCtx));
        }

        if (mglRendererCheckForDirtyBufferData(
                renderer, &mglSaStateView(&areas)->fragment_buffer_map_list))
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(
                renderer, &mglSaStateView(&areas)->fragment_buffer_map_list));

            RETURN_FALSE_ON_FAILURE(
                mglStageEncodeBindFragmentBuffers(renderer, &encCtx));
        }
    }
    return true;
}
