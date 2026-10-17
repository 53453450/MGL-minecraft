/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_buffer_layout.c -- see mgl_buffer_layout.h.
 */

#include "mgl_buffer_layout.h"

#include <string.h>

static int layout_add(uint32_t a, uint32_t b, uint32_t *out)
{
    if (a > UINT32_MAX - b) {
        return -1;
    }
    *out = a + b;
    return 0;
}

/* The tessellation evaluation stage keeps slot 0 for the contract buffer. */
static uint32_t layout_user_base(int stage)
{
    return stage == MGL_STAGE_TESS_EVALUATION ? 1u : 0u;
}

static int layout_stage_has_pack(int stage)
{
    return stage == MGL_STAGE_VERTEX || stage == MGL_STAGE_FRAGMENT ||
           stage == MGL_STAGE_COMPUTE || stage == MGL_STAGE_TESS_CONTROL ||
           stage == MGL_STAGE_TESS_EVALUATION || stage == MGL_STAGE_GEOMETRY;
}

int mglBufferSlotLayoutBuild(const MGLBufferSlotCounts *in,
                             MGLBufferSlotLayout *out)
{
    if (!out) {
        return MGL_BUFFER_LAYOUT_BAD_ARGS;
    }
    memset(out, 0, sizeof(*out));
    if (!in) {
        return MGL_BUFFER_LAYOUT_BAD_ARGS;
    }

    const uint32_t base = layout_user_base(in->stage);
    const uint32_t plain =
        in->has_plain_pack && layout_stage_has_pack(in->stage) ? 1u : 0u;
    const uint32_t attrs = in->stage == MGL_STAGE_VERTEX ? in->attr_slots : 0u;

    MGLBufferSlotLayout layout;
    memset(&layout, 0, sizeof(layout));
    layout.plain_base = base;
    layout.plain_slots = plain;
    if (layout_add(base, plain, &layout.ssbo_base) != 0 ||
        layout_add(layout.ssbo_base, attrs, &layout.ssbo_base) != 0 ||
        layout_add(layout.ssbo_base, in->ssbo_slots, &layout.ubo_base) != 0 ||
        layout_add(layout.ubo_base, in->ubo_slots,
                   &layout.atomic_counter_base) != 0 ||
        layout_add(layout.atomic_counter_base, in->atomic_counter_slots,
                   &layout.end) != 0) {
        return MGL_BUFFER_LAYOUT_OVERFLOW;
    }
    if (in->atomic_counter_slots > 0u && layout.end > in->atomic_counter_limit) {
        return MGL_BUFFER_LAYOUT_ATOMIC_EXCEEDS;
    }
    *out = layout;
    return MGL_BUFFER_LAYOUT_OK;
}
