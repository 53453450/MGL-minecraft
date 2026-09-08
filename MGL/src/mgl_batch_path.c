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
