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


/* Keep in sync with mgl_types_state.h dirty* enum / DIRTY_* masks. */
enum {
    MGL_BATCH_DIRTY_VAO = 1u << 0,
    MGL_BATCH_DIRTY_BUFFER = 1u << 2,
    MGL_BATCH_DIRTY_TEX = 1u << 3,
    MGL_BATCH_DIRTY_TEX_PARAM = 1u << 4,
    MGL_BATCH_DIRTY_TEX_BINDING = 1u << 5,
    MGL_BATCH_DIRTY_SAMPLER = 1u << 6,
    MGL_BATCH_DIRTY_PROGRAM = 1u << 8,
    MGL_BATCH_DIRTY_RENDER_STATE = 1u << 11,
    MGL_BATCH_DIRTY_ALPHA_STATE = 1u << 12,
    MGL_BATCH_DIRTY_IMAGE_UNIT = 1u << 13,
    MGL_BATCH_DIRTY_BUFFER_BASE = 1u << 14
};

uint32_t mgl_batch_restore_full_dirty_bits(void)
{
    return (MGL_BATCH_DIRTY_PROGRAM | MGL_BATCH_DIRTY_VAO |
            MGL_BATCH_DIRTY_RENDER_STATE | MGL_BATCH_DIRTY_TEX_BINDING |
            MGL_BATCH_DIRTY_TEX | MGL_BATCH_DIRTY_TEX_PARAM |
            MGL_BATCH_DIRTY_SAMPLER | MGL_BATCH_DIRTY_ALPHA_STATE |
            MGL_BATCH_DIRTY_BUFFER | MGL_BATCH_DIRTY_BUFFER_BASE |
            MGL_BATCH_DIRTY_IMAGE_UNIT);
}

void mgl_batch_restore_default_domain_masks(MGLBatchDirtyDomainMasks *out)
{
    if (!out) {
        return;
    }
    out->program = (MGL_BATCH_DIRTY_PROGRAM | MGL_BATCH_DIRTY_BUFFER_BASE |
                    MGL_BATCH_DIRTY_BUFFER);
    out->vao = (MGL_BATCH_DIRTY_VAO | MGL_BATCH_DIRTY_BUFFER);
    out->texture =
        (MGL_BATCH_DIRTY_TEX | MGL_BATCH_DIRTY_TEX_BINDING |
         MGL_BATCH_DIRTY_TEX_PARAM | MGL_BATCH_DIRTY_SAMPLER |
         MGL_BATCH_DIRTY_IMAGE_UNIT);
    out->render_state =
        (MGL_BATCH_DIRTY_RENDER_STATE | MGL_BATCH_DIRTY_ALPHA_STATE);
}

int mgl_batch_restore_can_delta(int dirty_key_delta_enabled, int prev_key_valid,
                                int has_encoder, int bind_valid)
{
    return dirty_key_delta_enabled && prev_key_valid && has_encoder &&
           bind_valid;
}

int mgl_batch_restore_oracle_would_skip(int skip_enabled, int last_key_valid,
                                        int last_execute_ok, int keys_equal)
{
    return !skip_enabled && last_key_valid && last_execute_ok && keys_equal;
}

uint32_t mgl_batch_restore_finish_dirty(uint32_t delta_bits, uint32_t forced_bits,
                                        uint32_t full_bits,
                                        uint32_t dirty_fbo_mask,
                                        const MGLBatchRestoreFboIn *fbo)
{
    return mgl_batch_restore_fold_fbo_dirty(delta_bits | forced_bits, full_bits,
                                            dirty_fbo_mask, fbo);
}

uint32_t mgl_batch_restore_plan_delta_dirty(
    int can_delta, const MGLBatchStateKeyView *prev,
    const MGLBatchStateKeyView *cur, uint32_t full_bits,
    MGLBatchDirtyDeltaFlags *flags_out)
{
    MGLBatchDirtyDomainMasks masks;
    mgl_batch_restore_default_domain_masks(&masks);
    return mgl_batch_compute_key_delta_dirty_bits(can_delta, prev, cur,
                                                  full_bits, &masks, flags_out);
}

void mgl_batch_restore_apply_from_key(const MGLBatchRestoreFromKeyOps *ops)
{
    if (!ops) {
        return;
    }
    if (ops->restore_program) {
        ops->restore_program(ops->ctx, ops->program_name,
                             ops->program_pipeline_name);
    }
    if (ops->set_vao) {
        ops->set_vao(ops->ctx, ops->vao_name);
    }
    if (ops->set_fbo) {
        ops->set_fbo(ops->ctx, ops->fbo_name);
    }
    if (ops->sync_fbo_names) {
        ops->sync_fbo_names(ops->ctx);
    }
    if (ops->apply_viewport_scissor) {
        ops->apply_viewport_scissor(ops->ctx, ops->viewport,
                                    ops->scissor_enabled ? 1 : 0, ops->scissor);
    }
}

void mgl_batch_restore_run_for_batch(const MGLBatchRestoreForBatchOps *ops)
{
    if (!ops) {
        return;
    }
    if (ops->has_snapshot) {
        if (ops->apply_snapshot) {
            ops->apply_snapshot(ops->ctx);
        }
    } else if (ops->apply_from_key) {
        ops->apply_from_key(ops->ctx);
    }
    if (ops->after_apply) {
        ops->after_apply(ops->ctx);
    }
    const uint32_t full = mgl_batch_restore_full_dirty_bits();
    uint32_t replay = full;
    MGLBatchDirtyDeltaFlags flags;
    memset(&flags, 0, sizeof(flags));
    const int can = ops->can_delta && ops->can_delta(ops->ctx);
    if (can && ops->plan_delta_dirty) {
        replay = ops->plan_delta_dirty(ops->ctx, full, &flags);
        if (ops->note_delta_perf) {
            ops->note_delta_perf(ops->ctx, &flags);
        }
    }
    MGLBatchRestoreFboIn fbo;
    memset(&fbo, 0, sizeof(fbo));
    if (ops->fill_fbo) {
        ops->fill_fbo(ops->ctx, &fbo);
    }
    replay = mgl_batch_restore_finish_dirty(replay, ops->forced_bits, full,
                                            ops->dirty_fbo_mask, &fbo);
    if (ops->mark_dirty) {
        ops->mark_dirty(ops->ctx, replay);
    }
}

void mgl_batch_teardown_run(const MGLBatchTeardownOps *ops)
{
    if (!ops) {
        return;
    }
    if (ops->used_replay_workspace && ops->sync_hash_from_replay) {
        ops->sync_hash_from_replay(ops->ctx);
    }
    if (ops->restore_live_active) {
        ops->restore_live_active(ops->ctx);
    }
    if (ops->clear_absolute_offsets) {
        ops->clear_absolute_offsets(ops->ctx);
    }
    if (ops->reset_command_buffer) {
        ops->reset_command_buffer(ops->ctx);
    }
    if (ops->arena_snapshot_enabled && ops->reset_arena) {
        ops->reset_arena(ops->ctx);
    }
    if (!ops->used_replay_workspace && ops->restore_saved_state) {
        ops->restore_saved_state(ops->ctx);
    }
    if (ops->clear_dirty_preserve_hash) {
        ops->clear_dirty_preserve_hash(ops->ctx);
    }
    if (ops->restore_program_pair) {
        ops->restore_program_pair(ops->ctx);
    }
    if (ops->propagate_replay_error) {
        ops->propagate_replay_error(ops->ctx);
    }
}

