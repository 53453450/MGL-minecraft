/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_batch_issue.h"

#include "mgl_batch_path.h"
#include "mgl_batch_restore.h"

#include <limits.h>
#include <string.h>

int mgl_batch_issue_stream_mdi_gate(const MGLBatchStreamMdiGateIn *in,
                                    size_t *needed_bytes)
{
    if (needed_bytes) {
        *needed_bytes = 0;
    }
    if (!in || !in->stream_merged || in->command_count == 0 ||
        in->stream_index_count == 0 || !in->has_encoder) {
        return MGL_BATCH_STREAM_MDI_FAIL_EMPTY;
    }
    if (in->disable_mdi) {
        return MGL_BATCH_STREAM_MDI_FAIL_DISABLED;
    }
    if (in->primitive_type == 0xFFu) {
        return MGL_BATCH_STREAM_MDI_FAIL_BAD_PRIM;
    }
    if (in->arg_size == 0 ||
        in->command_count > (UINT32_MAX / in->arg_size)) {
        return MGL_BATCH_STREAM_MDI_FAIL_OVERFLOW;
    }
    if (needed_bytes) {
        *needed_bytes = in->arg_size * (size_t)in->command_count;
    }
    return MGL_BATCH_STREAM_MDI_OK;
}

const char *mgl_batch_issue_stream_mdi_gate_reason(int gate)
{
    switch (gate) {
    case MGL_BATCH_STREAM_MDI_FAIL_EMPTY:
        return "stream_mdi_empty";
    case MGL_BATCH_STREAM_MDI_FAIL_DISABLED:
        return "stream_mdi_disabled";
    case MGL_BATCH_STREAM_MDI_FAIL_BAD_PRIM:
        return "stream_mdi_bad_prim";
    case MGL_BATCH_STREAM_MDI_FAIL_OVERFLOW:
        return "stream_mdi_args_overflow";
    default:
        return "stream_mdi_ok";
    }
}

void mgl_batch_issue_direct_arrays_params(uint32_t cmd_type,
                                          int32_t instance_count,
                                          uint32_t base_instance,
                                          int32_t *out_instance_count,
                                          uint32_t *out_base_instance,
                                          const char **out_reason,
                                          const char **out_cull_reason)
{
    int32_t ic = instance_count;
    uint32_t bi = 0u;
    const char *reason = "direct_arrays";
    const char *cull = "direct_arrays_cull_distance_split";
    if (cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS) {
        ic = 1;
        bi = 0u;
        reason = "direct_arrays";
        cull = "direct_arrays_cull_distance_split";
    } else if (cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED) {
        bi = 0u;
        reason = "direct_arrays_instanced";
        cull = "direct_arrays_instanced_cull_distance_split";
    } else if (cmd_type ==
               MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE) {
        bi = base_instance;
        reason = "direct_arrays_base_instance";
        cull = "direct_arrays_base_instance_cull_distance_split";
    }
    if (out_instance_count) {
        *out_instance_count = ic;
    }
    if (out_base_instance) {
        *out_base_instance = bi;
    }
    if (out_reason) {
        *out_reason = reason;
    }
    if (out_cull_reason) {
        *out_cull_reason = cull;
    }
}

int mgl_batch_issue_dyn_cmd_has_bindings(uint8_t vertex_count,
                                         uint8_t uniform_count,
                                         uint8_t texture_count)
{
    return vertex_count != 0 || uniform_count != 0 || texture_count != 0;
}

int mgl_batch_issue_dyn_needs_mapper_fallback(int vertex_ok, int uniform_ok)
{
    return !(vertex_ok && uniform_ok);
}

int mgl_batch_issue_cull_capture_path(int uses_cull_distance,
                                      uint32_t cmd_type)
{
    if (!uses_cull_distance) {
        return MGL_BATCH_CULL_CAPTURE_NONE;
    }
    if (cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS ||
        cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED ||
        cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE) {
        return MGL_BATCH_CULL_CAPTURE_ARRAYS;
    }
    return MGL_BATCH_CULL_CAPTURE_ELEMENTS;
}


int mgl_batch_issue_stream_index_ready(int has_index_buffer, int process_ok,
                                       int has_mtl_index)
{
    if (!has_index_buffer || !process_ok) {
        return MGL_BATCH_STREAM_INDEX_NO_BUFFER;
    }
    if (!has_mtl_index) {
        return MGL_BATCH_STREAM_INDEX_NO_MTL;
    }
    return MGL_BATCH_STREAM_INDEX_OK;
}

const char *mgl_batch_issue_stream_index_reason(int ready)
{
    switch (ready) {
    case MGL_BATCH_STREAM_INDEX_NO_BUFFER:
        return "stream_index_buffer";
    case MGL_BATCH_STREAM_INDEX_NO_MTL:
        return "stream_no_mtl_index";
    default:
        return "stream_direct_merged";
    }
}

int mgl_batch_issue_should_apply_stable_sampler(int snapshots_mixed,
                                                uint32_t snapshot_id,
                                                uint32_t invalid_id)
{
    return !snapshots_mixed && snapshot_id != invalid_id;
}

void mgl_batch_flush_accum_path(MGLBatchFlushPathStats *stats, int path,
                                uint32_t command_count)
{
    if (!stats) {
        return;
    }
    switch (path) {
    case MGL_BATCH_SELECT_STREAM_MERGE:
        stats->stream_batches++;
        stats->stream_commands += command_count;
        break;
    case MGL_BATCH_SELECT_MDI:
        stats->mdi_batches++;
        stats->mdi_commands += command_count;
        break;
    case MGL_BATCH_SELECT_ICB:
        stats->icb_batches++;
        stats->icb_commands += command_count;
        break;
    default:
        stats->direct_batches++;
        stats->direct_commands += command_count;
        break;
    }
}

const char *mgl_batch_flush_path_phase(int path)
{
    switch (path) {
    case MGL_BATCH_SELECT_STREAM_MERGE:
        return "ISSUE_STREAM_MERGE";
    case MGL_BATCH_SELECT_MDI:
        return "ISSUE_MDI";
    case MGL_BATCH_SELECT_ICB:
        return "ISSUE_ICB";
    default:
        return "ISSUE_DIRECT";
    }
}

int mgl_batch_flush_should_trace_log(uint64_t hit, uint32_t total_commands,
                                     int diag_enabled, uint32_t skipped_commands,
                                     int replay_error_nonzero)
{
    if (skipped_commands > 0u || replay_error_nonzero) {
        return 1;
    }
    if (!diag_enabled) {
        return 0;
    }
    return hit <= 16ull || (hit % 512ull) == 0ull || total_commands >= 128ull;
}

int mgl_batch_issue_scratch_range_ok(uint64_t offset, uint64_t needed,
                                     uint64_t length)
{
    if (offset > length) {
        return 0;
    }
    return needed <= (length - offset);
}

int mgl_batch_issue_should_apply_cmd_sampler(int snapshots_mixed,
                                             int has_dynamic_texture_bindings)
{
    return snapshots_mixed || has_dynamic_texture_bindings;
}

void mgl_batch_issue_icb_array_draw_params(uint32_t first, uint32_t count,
                                           uint32_t instance_count,
                                           uint32_t base_instance,
                                           MGLBatchIcbArrayDrawParams *out)
{
    if (!out) {
        return;
    }
    out->vertex_start = first;
    out->vertex_count = count;
    out->instance_count = instance_count;
    out->base_instance = base_instance;
}

uint32_t mgl_batch_issue_icb_command_types(int indexed)
{
    return indexed ? 2u : 1u;
}

void mgl_batch_issue_cmd_stat_delta(uint32_t cmd_type, int32_t count,
                                    int uses_elements,
                                    MGLBatchCmdStatDelta *out)
{
    if (!out) {
        return;
    }
    out->array_draws = 0u;
    out->array_vertices = 0ull;
    out->element_draws = 0u;
    out->element_indices = 0ull;
    const uint64_t n = (count > 0) ? (uint64_t)count : 0ull;
    /* Array cmd_type values match MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS*. */
    if (cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS ||
        cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED ||
        cmd_type == MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE) {
        out->array_draws = 1u;
        out->array_vertices = n;
        return;
    }
    if (uses_elements) {
        out->element_draws = 1u;
        out->element_indices = n;
    }
}

int mgl_batch_flush_scheduled_path_perf_kind(int path)
{
    switch (path) {
    case MGL_BATCH_SELECT_STREAM_MERGE:
        return MGL_BATCH_FLUSH_PERF_STREAM;
    case MGL_BATCH_SELECT_MDI:
    case MGL_BATCH_SELECT_ICB:
        return MGL_BATCH_FLUSH_PERF_NONE;
    default:
        return MGL_BATCH_FLUSH_PERF_DIRECT;
    }
}


int mgl_batch_issue_apply_dyn_bindings(uint8_t vertex_count, uint8_t uniform_count,
                                       uint8_t texture_count,
                                       const MGLBatchDynApplyOps *ops)
{
    if (!ops) {
        return 0;
    }
    if (!mgl_batch_issue_dyn_cmd_has_bindings(vertex_count, uniform_count,
                                             texture_count)) {
        return 1;
    }
    if (ops->refresh_owner) {
        ops->refresh_owner(ops->ctx);
    }
    if (!ops->has_encoder || !ops->has_encoder(ops->ctx)) {
        return 0;
    }
    if (vertex_count > 0u) {
        if (!ops->build_dyn_vao || !ops->build_dyn_vao(ops->ctx)) {
            return 0;
        }
    }
    if (uniform_count > 0u) {
        if (!ops->apply_ubo || !ops->apply_ubo(ops->ctx)) {
            return 0;
        }
    }
    int tex_ok = 1;
    if (texture_count > 0u) {
        if (!ops->apply_tex || !ops->apply_tex(ops->ctx)) {
            return 0;
        }
        tex_ok = ops->bind_tex_direct && ops->bind_tex_direct(ops->ctx);
        if (ops->refresh_owner) {
            ops->refresh_owner(ops->ctx);
        }
        if (!tex_ok) {
            tex_ok = ops->bind_tex_mapper && ops->bind_tex_mapper(ops->ctx);
            if (!tex_ok && ops->restore_after_tex_upload) {
                if (ops->refresh_owner) {
                    ops->refresh_owner(ops->ctx);
                }
                tex_ok = ops->restore_after_tex_upload(ops->ctx);
                if (ops->refresh_owner) {
                    ops->refresh_owner(ops->ctx);
                }
                if (tex_ok) {
                    tex_ok =
                        ops->bind_tex_mapper && ops->bind_tex_mapper(ops->ctx);
                }
            }
            if (ops->refresh_owner) {
                ops->refresh_owner(ops->ctx);
            }
        }
        if (ops->refresh_owner) {
            ops->refresh_owner(ops->ctx);
        }
    }
    if (!tex_ok) {
        return 0;
    }
    if (ops->refresh_owner) {
        ops->refresh_owner(ops->ctx);
    }
    const int vertex_ok =
        vertex_count == 0u ||
        (ops->bind_vertex_direct && ops->bind_vertex_direct(ops->ctx));
    if (ops->refresh_owner) {
        ops->refresh_owner(ops->ctx);
    }
    const int uniform_ok =
        uniform_count == 0u ||
        (ops->bind_uniform_direct && ops->bind_uniform_direct(ops->ctx));
    if (ops->refresh_owner) {
        ops->refresh_owner(ops->ctx);
    }
    if (!mgl_batch_issue_dyn_needs_mapper_fallback(vertex_ok, uniform_ok)) {
        return 1;
    }
    return ops->mapper_fallback && ops->mapper_fallback(ops->ctx);
}

void mgl_batch_flush_run_batches(MGLBatchFlushLoopState *st,
                                 const MGLBatchFlushLoopOps *ops)
{
    if (!st || !ops || !ops->batch_count || !ops->command_count) {
        return;
    }
    void *ctx = ops->ctx;
    const uint32_t n = ops->batch_count(ctx);
    for (uint32_t b = 0; b < n; b++) {
        const uint32_t cmd_count = ops->command_count(ctx, b);
        if (cmd_count == 0u) {
            continue;
        }

        MGLBatchSameKeySkipIn skip_in;
        memset(&skip_in, 0, sizeof(skip_in));
        int want_abs = 0;
        if (ops->fill_skip_in) {
            ops->fill_skip_in(ctx, b, &skip_in, &want_abs);
        }
        skip_in.skip_enabled = ops->skip_enabled ? 1u : 0u;
        skip_in.last_key_valid = st->last_key_valid;
        skip_in.last_execute_ok = st->last_execute_ok;
        skip_in.last_was_stream = st->last_was_stream;

        const int skip_dec = mgl_batch_same_key_skip_decision(&skip_in);
        if (ops->note_skip_perf) {
            ops->note_skip_perf(ctx, skip_dec);
        }
        if (ops->oracle_env_enabled && ops->on_oracle_would_skip &&
            mgl_batch_restore_oracle_would_skip(
                ops->skip_enabled ? 1 : 0, st->last_key_valid ? 1 : 0,
                st->last_execute_ok ? 1 : 0,
                (ops->oracle_keys_equal && ops->oracle_keys_equal(ctx, b))
                    ? 1
                    : 0)) {
            ops->on_oracle_would_skip(ctx);
        }

        if (skip_dec == MGL_BATCH_SAME_KEY_SKIP) {
            if (ops->apply_same_key_skip) {
                ops->apply_same_key_skip(ctx, b);
            }
        } else {
            /* absolute_offsets_match was computed before set; mismatch → dirty. */
            uint32_t forced =
                st->last_was_stream ? ops->vao_buffer_dirty_mask : 0u;
            if (!skip_in.absolute_offsets_match) {
                forced |= ops->vao_buffer_dirty_mask;
            }
            if (ops->set_absolute_offsets) {
                ops->set_absolute_offsets(ctx, want_abs);
            }
            if (ops->restore) {
                ops->restore(ctx, b, forced);
            }
        }

        if (!ops->check_execute || !ops->check_execute(ctx, b)) {
            st->last_execute_ok = 0u;
            continue;
        }

        st->last_execute_ok = 1u;
        st->last_key_valid = 1u;
        st->last_was_stream = 0u;
        if (ops->mark_execute_ok) {
            ops->mark_execute_ok(ctx, b);
        }

        const int path = ops->schedule ? ops->schedule(ctx, b)
                                       : MGL_BATCH_SELECT_DIRECT;
        mgl_batch_flush_accum_path(&st->path_stats, path, cmd_count);
        const char *phase = mgl_batch_flush_path_phase(path);
        if (ops->trace_phase) {
            ops->trace_phase(ctx, b, phase);
        }

        const int path_perf = mgl_batch_flush_scheduled_path_perf_kind(path);
        if (path_perf == MGL_BATCH_FLUSH_PERF_STREAM) {
            if (ops->perf_stream) {
                ops->perf_stream(ctx, cmd_count);
            }
            if (ops->issue_stream) {
                ops->issue_stream(ctx, b);
            }
            st->last_was_stream = 1u;
        } else if (path == MGL_BATCH_SELECT_MDI) {
            if (ops->issue_mdi) {
                ops->issue_mdi(ctx, b);
            }
        } else if (path == MGL_BATCH_SELECT_ICB) {
            if (ops->issue_icb) {
                ops->issue_icb(ctx, b);
            }
        } else {
            if (ops->perf_direct) {
                ops->perf_direct(ctx, cmd_count);
            }
            if (ops->issue_direct) {
                ops->issue_direct(ctx, b);
            }
        }

        if (ops->record_stats) {
            ops->record_stats(ctx, b);
        }
    }
}

int mgl_batch_check_should_execute(const MGLBatchCheckExecOps *ops)
{
    if (!ops) {
        return 0;
    }
    if (ops->begin_trace) {
        ops->begin_trace(ops->ctx);
    }
    if (ops->prepare_fbo && !ops->prepare_fbo(ops->ctx)) {
        return ops->trace_skip
                   ? ops->trace_skip(ops->ctx, "SKIP_FBO_ROTATION",
                                     "fbo_rotation")
                   : 0;
    }
    if (ops->process_gl_state && !ops->process_gl_state(ops->ctx)) {
        if (ops->capture_error_if_any) {
            ops->capture_error_if_any(ops->ctx);
        }
        return ops->trace_skip
                   ? ops->trace_skip(ops->ctx, "SKIP_PROCESS_STATE",
                                     "processGLState")
                   : 0;
    }
    if (ops->should_apply_sampler && ops->should_apply_sampler(ops->ctx)) {
        if (!ops->apply_sampler || !ops->apply_sampler(ops->ctx)) {
            return ops->trace_skip
                       ? ops->trace_skip(ops->ctx, "SKIP_SAMPLER_SNAPSHOT",
                                         "sampler_snapshot")
                       : 0;
        }
    }
    if (ops->trace_ready) {
        ops->trace_ready(ops->ctx);
    }
    if (ops->empty_raster && ops->empty_raster(ops->ctx)) {
        return ops->trace_skip
                   ? ops->trace_skip(ops->ctx, "SKIP_EMPTY_RASTER",
                                     "empty_rasterization")
                   : 0;
    }
    if (ops->fully_culled && ops->fully_culled(ops->ctx)) {
        return ops->trace_skip
                   ? ops->trace_skip(ops->ctx, "SKIP_FULLY_CULLED",
                                     "front_and_back_culled")
                   : 0;
    }
    if (ops->apply_polygon_offset) {
        ops->apply_polygon_offset(ops->ctx);
    }
    return 1;
}
