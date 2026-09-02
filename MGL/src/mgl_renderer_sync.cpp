/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * GL→Metal state synchronization orchestration (C++ home).
 * Phase 2: dirty-domain dispatch migrated from processDirtyStateDomains.
 */

#include "mgl_renderer_sync.h"

#include "mgl_renderer_binding.h"
#include "mgl_render.h"
#include "mgl_types_state.h"
#include "mgl_types_framebuffer.h"

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

extern "C" int mglRenderSyncRenderPassForFbo(
    GLMContext context, const MGLCommandState *command_state,
    const MGLRenderPassSyncOps *ops)
{
    if (!context || !context->active_state || !ops) {
        return 0;
    }

    GLMState *state = context->active_state;
    Framebuffer *framebuffer = ops->get_validated_framebuffer
        ? ops->get_validated_framebuffer(ops->renderer, context,
                                         "processGLState.dirtyFBO")
        : state->framebuffer;
    const bool binding_dirty =
        framebuffer && (framebuffer->dirty_bits & DIRTY_FBO_BINDING);

    if (command_state &&
        mglRenderEncoderOwnerHasCurrent(
            command_state->currentRenderEncoderOwner) == 1 &&
        !binding_dirty && ops->render_pass_matches_framebuffer &&
        ops->render_pass_matches_framebuffer(ops->renderer, context)) {
        state->dirty_bits &= ~DIRTY_FBO;
        return 1;
    }

    if (framebuffer && binding_dirty) {
        if (!ops->bind_framebuffer_attachment_textures ||
            !ops->bind_framebuffer_attachment_textures(ops->renderer,
                                                       context)) {
            return 0;
        }
        framebuffer = ops->get_validated_framebuffer
            ? ops->get_validated_framebuffer(
                  ops->renderer, context,
                  "processGLState.dirtyFBO.afterBind")
            : state->framebuffer;
        if (framebuffer) {
            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
        }
    }

    if (!ops->rotate_render_encoder_for_fbo ||
        !ops->rotate_render_encoder_for_fbo(ops->renderer, context)) {
        return 0;
    }
    return 1;
}

extern "C" int mglRenderProcessGLStatePreamble(
    GLMContext context, MGLCommandState *command_state, int draw_command,
    uint64_t process_call, int trace_process,
    const MGLProcessGLStatePreambleOps *ops)
{
    if (!context || !context->active_state || !ops) {
        return MGL_PREAMBLE_FAIL;
    }

    GLMState *state = context->active_state;

    if (ops->ensure_metal_objects_ready &&
        !ops->ensure_metal_objects_ready(ops->renderer)) {
        return MGL_PREAMBLE_FAIL;
    }

    if (draw_command && command_state && ops->on_draw_command_begin) {
        ops->on_draw_command_begin(ops->renderer, context, command_state);
    }

    if (!draw_command && ops->end_render_pass_non_draw) {
        ops->end_render_pass_non_draw(ops->renderer, process_call);
    }

    if (state->vao == NULL) {
        if (draw_command) {
            if (ops->reject_draw_without_vao) {
                ops->reject_draw_without_vao(ops->renderer, context);
            }
            return MGL_PREAMBLE_FAIL;
        }
        if (ops->handle_null_vao_path) {
            return ops->handle_null_vao_path(ops->renderer, context,
                                             draw_command);
        }
        return MGL_PREAMBLE_DONE_OK;
    }

    if (!draw_command) {
        return MGL_PREAMBLE_DONE_OK;
    }

    if (ops->check_program_quarantine &&
        !ops->check_program_quarantine(ops->renderer, context)) {
        return MGL_PREAMBLE_FAIL;
    }

    if (command_state) {
        MGLRenderCommandBufferState process_command_state = {0};
        const int process_has_command =
            mglRenderGetCommandBufferOwnerState(
                command_state->currentCommandBufferOwner,
                &process_command_state) == 0;
        if (process_has_command &&
            mglRenderEncoderOwnerHasCurrent(
                command_state->currentRenderEncoderOwner) != 1) {
            const uint32_t pre_status =
                (uint32_t)process_command_state.status;
            if (pre_status >= MGLCommandBufferStatusCommitted) {
                if (!ops->rotate_finalized_command_buffer ||
                    !ops->rotate_finalized_command_buffer(
                        ops->renderer, context, trace_process)) {
                    return MGL_PREAMBLE_FAIL;
                }
            }
        } else if (!process_has_command) {
            if (!ops->create_initial_command_buffer ||
                !ops->create_initial_command_buffer(ops->renderer, context,
                                                  trace_process)) {
                return MGL_PREAMBLE_FAIL;
            }
        }
    }

    return MGL_PREAMBLE_CONTINUE;
}

extern "C" int mglRenderProcessGLStateTail(
    GLMContext context, const MGLCommandState *command_state,
    int draw_command, int trace_process,
    MGLResourceSyncWork *resource_sync_work,
    const MGLProcessGLStateTailOps *ops)
{
    if (!context || !ops) {
        return 0;
    }

    if (command_state &&
        mglRenderEncoderOwnerHasCurrent(
            command_state->currentRenderEncoderOwner) != 1) {
        if (!ops->recover_nil_render_encoder ||
            !ops->recover_nil_render_encoder(ops->renderer, context)) {
            return 0;
        }
    }

    if (draw_command) {
        if (!ops->prepare_draw_pass ||
            !ops->prepare_draw_pass(ops->renderer, context)) {
            return 0;
        }
        if (ops->log_draw_pipeline_lookup) {
            ops->log_draw_pipeline_lookup(ops->renderer, context);
        }
    }

    if (!ops->ensure_pipeline_ready ||
        !ops->ensure_pipeline_ready(ops->renderer, context, trace_process)) {
        return 0;
    }
    if (!ops->validate_render_pass ||
        !ops->validate_render_pass(ops->renderer, context, trace_process)) {
        return 0;
    }
    if (!ops->bind_pipeline ||
        !ops->bind_pipeline(ops->renderer, context, trace_process)) {
        return 0;
    }
    if (!mglRenderSyncResourceBindings(context, resource_sync_work)) {
        return 0;
    }
    if (draw_command && ops->apply_post_bind_draw_state &&
        !ops->apply_post_bind_draw_state(ops->renderer, context)) {
        return 0;
    }
    return 1;
}

extern "C" int mglRenderProcessGLState(GLMContext context, int draw_command)
{
    if (!context) {
        return 0;
    }
    return mglRendererObjCProcessGLState(context, draw_command != 0) ? 1 : 0;
}
