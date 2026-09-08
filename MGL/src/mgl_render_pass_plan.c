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
