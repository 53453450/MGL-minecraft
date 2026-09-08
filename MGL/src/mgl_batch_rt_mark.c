/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_batch_rt_mark.h"

int mgl_batch_rt_attachment_active(uint32_t bitfield, uint32_t index,
                                   uint32_t max_attachments)
{
    if (index >= max_attachments) {
        return 0;
    }
    return ((bitfield >> index) & 1u) != 0u;
}

int mgl_batch_rt_yflip_authority(int has_injected_yflip,
                                 int yflip_sampler_explicit,
                                 int has_in_sampler_named,
                                 int has_diffuse_sampler_named)
{
    if (!has_injected_yflip || !yflip_sampler_explicit) {
        return 0;
    }
    if (has_in_sampler_named || has_diffuse_sampler_named) {
        return 0;
    }
    return 1;
}

int mgl_batch_rt_should_trace_write_mark(uint64_t hit)
{
    return hit <= 128ull || (hit % 256ull) == 0ull;
}

int mgl_batch_rt_should_cross_mark(int already_marked, int attachment_active)
{
    return !already_marked && attachment_active;
}

int mgl_batch_rt_should_diag_attachment0(uint32_t attachment_index,
                                         int trace_enabled, int can_use_copy)
{
    return attachment_index == 0u && trace_enabled && can_use_copy;
}

void mgl_batch_trace_fs_slot_flags(const MGLBatchTraceFsSlot *slots, uint32_t n,
                                   int *has_rt_out, int *used_copy_out)
{
    int has_rt = 0;
    int used_copy = 0;
    if (slots) {
        for (uint32_t i = 0; i < n; i++) {
            if (slots[i].rt_write_version != 0u) {
                has_rt = 1;
            }
            if (slots[i].used_sampled_copy) {
                used_copy = 1;
            }
        }
    }
    if (has_rt_out) {
        *has_rt_out = has_rt;
    }
    if (used_copy_out) {
        *used_copy_out = used_copy;
    }
}
