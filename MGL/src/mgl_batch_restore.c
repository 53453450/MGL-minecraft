/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_batch_restore.h"

#include <string.h>

int mgl_batch_same_key_skip_decision(const MGLBatchSameKeySkipIn *in)
{
    if (!in || !in->skip_enabled) {
        return MGL_BATCH_SAME_KEY_NO_SKIP;
    }
    if (!in->last_key_valid || !in->last_execute_ok || in->last_was_stream) {
        return MGL_BATCH_SAME_KEY_NO_SKIP;
    }
    if (in->has_encoder && in->bind_valid && in->keys_equal &&
        in->absolute_offsets_match && in->pass_matches) {
        return MGL_BATCH_SAME_KEY_SKIP;
    }
    if (!in->has_encoder) {
        return MGL_BATCH_SAME_KEY_FAIL_NO_ENCODER;
    }
    if (!in->bind_valid) {
        return MGL_BATCH_SAME_KEY_FAIL_BIND;
    }
    if (!in->keys_equal) {
        return MGL_BATCH_SAME_KEY_FAIL_KEY;
    }
    return MGL_BATCH_SAME_KEY_FAIL_PASS;
}

static int view_viewport_equal(const MGLBatchStateKeyView *a,
                               const MGLBatchStateKeyView *b)
{
    return memcmp(a->viewport, b->viewport, sizeof(a->viewport)) == 0 &&
           memcmp(a->scissor, b->scissor, sizeof(a->scissor)) == 0;
}

uint32_t mgl_batch_compute_key_delta_dirty_bits(
    int can_delta, const MGLBatchStateKeyView *prev,
    const MGLBatchStateKeyView *cur, uint32_t full_bits,
    const MGLBatchDirtyDomainMasks *masks, MGLBatchDirtyDeltaFlags *flags_out)
{
    if (flags_out) {
        memset(flags_out, 0, sizeof(*flags_out));
    }
    if (!can_delta || !prev || !cur || !masks) {
        return full_bits;
    }

    uint32_t bits = 0u;
    if (prev->program_name != cur->program_name ||
        prev->program_pipeline_name != cur->program_pipeline_name ||
        prev->vertex_program_name != cur->vertex_program_name ||
        prev->fragment_program_name != cur->fragment_program_name) {
        bits |= masks->program;
        if (flags_out) {
            flags_out->domain_program = 1u;
        }
    }
    if (prev->vao_name != cur->vao_name ||
        prev->vertex_layout_hash != cur->vertex_layout_hash) {
        bits |= masks->vao;
        if (flags_out) {
            flags_out->domain_vao = 1u;
        }
    }
    if (prev->texture_hash != cur->texture_hash) {
        bits |= masks->texture;
        if (flags_out) {
            flags_out->domain_texture = 1u;
        }
    }
    if (prev->render_state_hash != cur->render_state_hash ||
        prev->caps_flags != cur->caps_flags ||
        prev->scissor_enabled != cur->scissor_enabled ||
        prev->primitive_type != cur->primitive_type ||
        !view_viewport_equal(prev, cur)) {
        bits |= masks->render_state;
        if (flags_out) {
            const int ubo_only =
                ((prev->render_state_hash ^ prev->uniform_buffer_hash) ==
                 (cur->render_state_hash ^ cur->uniform_buffer_hash)) &&
                prev->caps_flags == cur->caps_flags &&
                prev->scissor_enabled == cur->scissor_enabled &&
                prev->primitive_type == cur->primitive_type &&
                view_viewport_equal(prev, cur);
            if (ubo_only) {
                flags_out->domain_render_state_ubo_only = 1u;
            } else {
                flags_out->domain_render_state = 1u;
            }
        }
    }
    if (flags_out && bits != full_bits) {
        flags_out->narrowed = 1u;
    }
    return bits;
}


uint32_t mgl_batch_restore_fold_fbo_dirty(uint32_t replay_dirty_bits,
                                         uint32_t full_bits,
                                         uint32_t dirty_fbo_mask,
                                         const MGLBatchRestoreFboIn *in)
{
    if (!in) {
        return replay_dirty_bits;
    }
    const int need_fbo =
        in->fbo_binding_dirty || in->prev_fbo_differs ||
        (in->has_encoder && !in->pass_matches);
    if (need_fbo) {
        replay_dirty_bits |= dirty_fbo_mask;
    }
    if (!in->has_encoder || !in->bind_valid) {
        replay_dirty_bits =
            full_bits | ((replay_dirty_bits & dirty_fbo_mask) ? dirty_fbo_mask
                                                             : 0u);
        if (in->fbo_binding_dirty ||
            (in->has_encoder && !in->pass_matches) ||
            in->prev_fbo_differs) {
            replay_dirty_bits |= dirty_fbo_mask;
        }
    }
    return replay_dirty_bits;
}

uint32_t mgl_batch_restore_absolute_contract_dirty(
    int want_absolute, int current_absolute, uint32_t vao_buffer_mask)
{
    return (want_absolute != current_absolute) ? vao_buffer_mask : 0u;
}
