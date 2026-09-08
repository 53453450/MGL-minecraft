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

#include <limits.h>

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
