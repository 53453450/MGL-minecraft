/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_render_pass_plan.h"
#include "mgl_render_pass_clear.h"   /* MGLRenderPassState + O3.1 clear-value plan */

#include <string.h>

MGLProcessGLStateClass mglRenderClassifyProcessGLState(int has_ctx,
                                                       int draw_command,
                                                       int has_vao,
                                                       int dirty_state)
{
    if (!has_ctx) {
        return MGL_PROCESS_GL_ABORT;
    }
    if (!has_vao) {
        if (draw_command) {
            return MGL_PROCESS_GL_ABORT;
        }
        if (dirty_state) {
            return MGL_PROCESS_GL_NO_VAO_CLEAR;
        }
        return MGL_PROCESS_GL_NON_DRAW;
    }
    if (!draw_command) {
        return MGL_PROCESS_GL_NON_DRAW;
    }
    return MGL_PROCESS_GL_CONTINUE;
}

int mglRenderProcessGLState(const MGLProcessGLStateInputs *in,
                            MGLProcessGLStatePlan *out)
{
    if (!out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (!in || !in->has_ctx || !in->ctx_ptr_sane) {
        out->result = MGL_PGL_RESULT_ABORT;
        out->process_class = MGL_PROCESS_GL_ABORT;
        return 0;
    }
    if (!in->device_ok || !in->queue_ok) {
        out->result = MGL_PGL_RESULT_ABORT;
        out->process_class = MGL_PROCESS_GL_ABORT;
        return 0;
    }

    if (in->draw_command) {
        out->clear_rt_sampled_copy = 1u;
    }

    out->process_class = mglRenderClassifyProcessGLState(
        1, in->draw_command ? 1 : 0, in->has_vao ? 1 : 0,
        in->dirty_state ? 1 : 0);

    if (out->process_class == MGL_PROCESS_GL_ABORT) {
        out->result = MGL_PGL_RESULT_ABORT;
        return 0;
    }
    if (out->process_class == MGL_PROCESS_GL_NON_DRAW) {
        out->result = MGL_PGL_RESULT_EARLY_OK;
        if (!in->draw_command) {
            out->non_draw_end_pass_if_fbo_changed = 1u;
        }
        return 0;
    }
    if (out->process_class == MGL_PROCESS_GL_NO_VAO_CLEAR) {
        out->result = MGL_PGL_RESULT_EARLY_OK;
        out->no_vao_clear_path = 1u;
        if (!in->draw_command) {
            out->non_draw_end_pass_if_fbo_changed = 1u;
        }
        return 0;
    }

    if (in->quarantine_blocks_draw) {
        out->result = MGL_PGL_RESULT_ABORT;
        return 0;
    }

    if (in->has_command_buffer && !in->encoder_has_current &&
        in->command_buffer_status >= (uint32_t)MGL_PGL_CB_STATUS_COMMITTED) {
        out->rotate_finalized_command_buffer = 1u;
    } else if (!in->has_command_buffer) {
        out->create_initial_command_buffer = 1u;
    }

    out->process_dirty_domains = 1u;
    out->result = MGL_PGL_RESULT_CONTINUE;
    return 0;
}

int mglRenderProcessGLStateAfterDirty(const MGLProcessGLStateAfterInputs *in,
                                      MGLProcessGLStateAfterPlan *out)
{
    if (!out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (!in) {
        out->result = MGL_PGL_RESULT_ABORT;
        return 0;
    }

    if (!in->encoder_has_current) {
        out->recover_nil_encoder = 1u;
    }
    if (in->draw_command) {
        out->ensure_pass_matches_fbo = 1u;
    }
    if (!in->has_pipeline_state) {
        out->fail_nil_pipeline = 1u;
        out->result = MGL_PGL_RESULT_ABORT;
        return 0;
    }

    out->validate_attachments = 1u;
    out->set_pipeline = 1u;
    out->sync_resources = 1u;
    if (in->frag_needs_fragcoord || in->frag_needs_sample) {
        out->bind_frag_coord_slot = 1u;
    }
    if (in->frag_needs_lod_bias) {
        out->bind_lod_bias_slot = 1u;
    }
    if (in->draw_command && in->fragment_trace_uses_rt_sampled_copy) {
        out->maybe_mark_rt_sampled_copy = 1u;
    }
    out->result = MGL_PGL_RESULT_CONTINUE;
    return 0;
}

/* O3.1: attachment-kind classification (moved out of mgl_render.cpp).
 * COLOR -> 1, DEPTH -> 2, STENCIL -> 3, anything else -> 0. */
int mglRenderPassAttachmentClass(uint32_t kind)
{
    if (kind == (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR) {
        return 1;
    }
    if (kind == (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH) {
        return 2;
    }
    if (kind == (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL) {
        return 3;
    }
    return 0;
}

int mglRenderPassColorAttachmentIndexValid(uint32_t color_index,
                                           uint32_t max_color)
{
    return color_index < max_color ? 1 : 0;
}

/* O3.1: pure clear-value resolution for one attachment.
 * Line-for-line port of the body that used to live inline in
 * MGLRenderer+RenderPass.m's mglRenderPassClearValuesFor.  The caller (ObjC)
 * still owns fetching the persistent MGLRenderPassState, so this stays free of
 * MGLCommandState / renderPassStateOwner and is directly unit-testable. */
int mglRenderPassPlanClearValues(const MGLRenderPassState *state,
                                 uint32_t attachmentKind,
                                 uint32_t colorIndex,
                                 double clearColorOut[4],
                                 double *clearDepthOut,
                                 uint32_t *clearStencilOut)
{
    if (!state) {
        return 0;
    }
    switch (mglRenderPassAttachmentClass(attachmentKind)) {
        case 1: {
            if (!mglRenderPassColorAttachmentIndexValid(
                    colorIndex,
                    (uint32_t)MGL_RENDER_MAX_COLOR_ATTACHMENTS)) {
                return 0;
            }
            const MGLRenderPassColorState *color = &state->color[colorIndex];
            if (clearColorOut) {
                clearColorOut[0] = color->clear_red;
                clearColorOut[1] = color->clear_green;
                clearColorOut[2] = color->clear_blue;
                clearColorOut[3] = color->clear_alpha;
            }
            return 1;
        }
        case 2:
            if (clearDepthOut) *clearDepthOut = state->depth.clear_depth;
            return 1;
        case 3:
            if (clearStencilOut) *clearStencilOut = state->stencil.clear_stencil;
            return 1;
        default:
            return 0;
    }
}

/* ---- O3.1: load / store actions, stale clear bits, attachment match ---- */

int mglRenderPassPlanLoadStore(const MGLRenderPassLoadStoreInput *in,
                               MGLRenderPassLoadStorePlan *out) {
    if (!in || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->load_action = MGLLoadActionLoad;

    if (in->attachment_kind == MGL_RP_ATTACHMENT_COLOR) {
        if (!in->attachment_present) {
            /* Nothing to clear into: keep whatever the pass state carried. */
            return 0;
        }
        if (in->has_clear_pending) {
            out->load_action = MGLLoadActionClear;
            out->set_store_action = 1;
            out->store_action = MGLStoreActionStore;
            return 0;
        }
        if (in->dontcare_enabled && in->texture_present &&
            in->first_use_this_frame && !in->blend_enabled) {
            out->load_action = MGLLoadActionDontCare;
        }
        return 0;
    }

    /* Depth and stencil: clear wins, otherwise load and keep the contents by
     * storing when there is an attachment texture to store into. */
    if (in->has_clear_pending) {
        out->load_action = MGLLoadActionClear;
        out->set_store_action = 1;
        out->store_action = MGLStoreActionStore;
        return 0;
    }
    out->load_action = MGLLoadActionLoad;
    if (in->texture_present) {
        out->set_store_action = 1;
        out->store_action = MGLStoreActionStore;
    }
    return 0;
}

int mglRenderPassDropsStaleColorClear(uint32_t clear_mask,
                                      uint32_t attached_bitfield,
                                      uint32_t attachment_index) {
    if (attachment_index >= 32u) {
        return 0;
    }
    const uint32_t bit = 1u << attachment_index;
    return (clear_mask & bit) != 0u && (attached_bitfield & bit) == 0u ? 1 : 0;
}

int mglRenderPassAttachmentsMatch(const MGLRenderPassAttachmentMatchInput *in) {
    if (!in || !in->identity_ok) {
        return 0;
    }
    for (uint32_t i = 0; i < in->slot_count; i++) {
        const MGLRenderPassSlotMatch *slot = &in->slots[i];
        if (!slot->compare) {
            continue;
        }
        if (slot->actual != slot->expected) {
            return 0;
        }
    }
    if (in->actual_depth != in->expected_depth ||
        in->actual_stencil != in->expected_stencil) {
        return 0;
    }
    /* A required attachment that is not there cannot be drawn into. */
    if (in->depth_required && !in->expected_depth) {
        return 0;
    }
    if (in->stencil_required && !in->expected_stencil) {
        return 0;
    }
    return 1;
}
