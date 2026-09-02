/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * GL→Metal state synchronization orchestration (C++ home).
 * Phase 2: dirty-domain dispatch migrated from processDirtyStateDomains.
 */

#include "mgl_renderer_sync.h"

#include "mgl_render.h"
#include "mgl_types_state.h"

extern "C" bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command);
extern "C" uint32_t mglRenderClassifyDirtySyncDomains(uint32_t dirty_bits);

static bool mglSyncOp(const MGLRendererSyncOps *ops,
                      bool (*fn)(void *, GLMContext),
                      GLMContext context)
{
    return ops && ops->renderer && fn && fn(ops->renderer, context);
}

extern "C" int mglRenderProcessDirtyStateDomains(
    GLMContext context, uint32_t domain_mask, int draw_command,
    const MGLCommandState *command_state, MGLResourceSyncWork *work,
    const MGLRendererSyncOps *ops)
{
    if (!context || !context->active_state || !ops) {
        return 0;
    }

    GLMState *state = context->active_state;
    if (!state->dirty_bits) {
        if ((domain_mask & MGL_SYNC_DOMAIN_ALL) == 0) {
            return 1;
        }
        return mglSyncOp(ops, ops->sync_incidental_buffer_data, context) ? 1
                                                                       : 0;
    }

    bool deferred_buffer_map = false;

    if ((state->dirty_bits & DIRTY_FBO) &&
        (domain_mask & MGL_SYNC_DOMAIN_FBO)) {
        if (!mglSyncOp(ops, ops->sync_render_pass_for_fbo, context)) {
            return 0;
        }
    }

    if ((state->dirty_bits & DIRTY_STATE) &&
        (domain_mask & MGL_SYNC_DOMAIN_STATE)) {
        if ((state->dirty_bits & DIRTY_FBO) &&
            ops->bind_framebuffer_attachments_in_state_block &&
            !ops->bind_framebuffer_attachments_in_state_block(ops->renderer,
                                                              context)) {
            return 0;
        }
        state->dirty_bits &= ~DIRTY_STATE;
    }

    if ((state->dirty_bits &
         (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_BUFFER_BASE_STATE)) &&
        (domain_mask & MGL_SYNC_DOMAIN_PROGRAM_VAO)) {
        if (draw_command && ops->should_defer_buffer_map &&
            ops->should_defer_buffer_map(ops->renderer, context,
                                         draw_command)) {
            deferred_buffer_map = true;
        } else if (!mglSyncOp(ops, ops->map_buffers, context)) {
            return 0;
        } else if (work) {
            work->mappedBuffers = true;
        }
        state->dirty_bits &= ~DIRTY_BUFFER_BASE_STATE;
    }

    if ((state->dirty_bits &
         (DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER)) &&
        (domain_mask & MGL_SYNC_DOMAIN_TEX)) {
        if (!mglSyncOp(ops, ops->bind_active_textures, context)) {
            return 0;
        }
        if (work) {
            work->boundActiveTextures = true;
        }
        state->dirty_bits &=
            ~(DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER);
    }

    if ((state->dirty_bits & DIRTY_VAO) && (domain_mask & MGL_SYNC_DOMAIN_VAO)) {
        if (!mglSyncOp(ops, ops->update_base_buffer_lists, context)) {
            return 0;
        }
        if (work) {
            work->updatedBaseLists = true;
        }
        if (command_state &&
            mglRenderEncoderOwnerHasCurrent(
                command_state->currentRenderEncoderOwner) != 1) {
            if (!ops->ensure_render_encoder ||
                !ops->ensure_render_encoder(ops->renderer, context,
                                            MGL_ENC_REASON_VAO)) {
                return 0;
            }
        }
        if (!mglSyncOp(ops, ops->update_render_encoder, context)) {
            return 0;
        }
        state->dirty_bits &= ~DIRTY_RENDER_STATE;
    } else if ((state->dirty_bits & DIRTY_BUFFER) &&
               (domain_mask & MGL_SYNC_DOMAIN_BUFFER)) {
        if (!mglSyncOp(ops, ops->update_base_buffer_lists, context)) {
            return 0;
        }
        if (work) {
            work->updatedBaseLists = true;
        }
        state->dirty_bits &= ~DIRTY_BUFFER;
    } else if ((state->dirty_bits & DIRTY_RENDER_STATE) &&
               (domain_mask & MGL_SYNC_DOMAIN_RENDER_STATE)) {
        if (command_state &&
            mglRenderEncoderOwnerHasCurrent(
                command_state->currentRenderEncoderOwner) != 1) {
            if (!ops->ensure_render_encoder ||
                !ops->ensure_render_encoder(ops->renderer, context,
                                            MGL_ENC_REASON_RS)) {
                return 0;
            }
        }
        if (!mglSyncOp(ops, ops->update_render_encoder, context)) {
            return 0;
        }
        state->dirty_bits &= ~DIRTY_RENDER_STATE;
    }

    if ((state->dirty_bits &
         (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_ALPHA_STATE |
          DIRTY_RENDER_STATE)) &&
        (domain_mask & MGL_SYNC_DOMAIN_PIPELINE)) {
        if (!ops->sync_pipeline ||
            !ops->sync_pipeline(ops->renderer, context,
                                deferred_buffer_map ? 1 : 0)) {
            return 0;
        }
    }

    state->dirty_bits = 0;
    return 1;
}

extern "C" int mglRenderProcessGLState(GLMContext context, int draw_command)
{
    if (!context) {
        return 0;
    }
    return mglRendererObjCProcessGLState(context, draw_command != 0) ? 1 : 0;
}
