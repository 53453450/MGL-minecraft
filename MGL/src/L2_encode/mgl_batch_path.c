/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_batch_path.h"

#include "mgl_env_flag.h"

int mgl_batch_select_path(const MGLBatchSelectInputs *in)
{
    if (!in || in->command_count == 0u) {
        return MGL_BATCH_SELECT_DIRECT;
    }

    if (in->sampler_snapshots_mixed) {
        return MGL_BATCH_SELECT_DIRECT;
    }

    if (in->uses_cull_distance) {
        /* CullDistance capture + per-primitive expansion need issueDirectBatch. */
        return MGL_BATCH_SELECT_DIRECT;
    }

    if (in->stream_merged) {
        return MGL_BATCH_SELECT_STREAM_MERGE;
    }

    if (!in->has_dynamic_uniform_bindings && !in->has_dynamic_vertex_bindings &&
        !in->has_dynamic_texture_bindings && !in->sampler_snapshots_mixed &&
        in->enable_icb && !in->disable_icb && in->icb_os_supported &&
        in->primitive_type != 0xFFu) {
        return MGL_BATCH_SELECT_ICB;
    }

    if (!in->disable_mdi && in->mdi_compatible &&
        in->command_count >= (uint32_t)MGL_BATCH_SELECT_MDI_MIN_COMMANDS &&
        !in->polygon_mode_point) {
        if (!(in->uses_elements && in->primitive_restart)) {
            return MGL_BATCH_SELECT_MDI;
        }
    }

    return MGL_BATCH_SELECT_DIRECT;
}

MGLBatchIcbConfig mgl_batch_icb_config(void)
{
    MGLBatchIcbConfig cfg = {0};
    /* Unified name or either legacy enable → enable. */
    if (mgl_env_flag_enabled("MGL_ENABLE_ICB") ||
        mgl_env_flag_enabled("MGL_ENABLE_ICB_BATCH") ||
        mgl_env_flag_enabled("MGL_ENABLE_ICB_PIPELINES")) {
        cfg.enable = 1u;
    }
    if (mgl_env_flag_enabled("MGL_DISABLE_ICB") ||
        mgl_env_flag_enabled("MGL_DISABLE_ICB_BATCH")) {
        cfg.disable = 1u;
    }
    return cfg;
}

int mgl_batch_icb_support_indirect_command_buffers(void)
{
    MGLBatchIcbConfig cfg = mgl_batch_icb_config();
    return (cfg.enable && !cfg.disable) ? 1 : 0;
}

void mgl_batch_fill_select_inputs_from_batch_flags(
    uint32_t command_count, int sampler_snapshots_mixed, int stream_merged,
    int has_dynamic_uniform_bindings, int has_dynamic_vertex_bindings,
    int has_dynamic_texture_bindings, int mdi_compatible, int uses_elements,
    uint8_t primitive_type, MGLBatchSelectInputs *out)
{
    if (!out) {
        return;
    }
    *out = (MGLBatchSelectInputs){0};
    out->command_count = command_count;
    out->sampler_snapshots_mixed = sampler_snapshots_mixed ? 1u : 0u;
    out->stream_merged = stream_merged ? 1u : 0u;
    out->has_dynamic_uniform_bindings =
        has_dynamic_uniform_bindings ? 1u : 0u;
    out->has_dynamic_vertex_bindings = has_dynamic_vertex_bindings ? 1u : 0u;
    out->has_dynamic_texture_bindings =
        has_dynamic_texture_bindings ? 1u : 0u;
    out->mdi_compatible = mdi_compatible ? 1u : 0u;
    out->uses_elements = uses_elements ? 1u : 0u;
    out->primitive_type = primitive_type;
}
