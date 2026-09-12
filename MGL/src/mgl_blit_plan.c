/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_blit_plan.c — O4.4 depth/stencil blit gates (pure C).
 *
 * The three gates below were read off MGLRenderer+Blit.m's
 * blitFramebufferDepthStencil: while it still owned them; the harness in
 * test_legacy_compat/test_blit_plan.c pins each boundary.
 */

#include "mgl_blit_plan.h"

#include <string.h>

static int32_t blit_min_i32(int32_t a, int32_t b) { return a < b ? a : b; }
static int32_t blit_max_i32(int32_t a, int32_t b) { return a > b ? a : b; }

void mglBlitFillDSTextureInput(MGLBlitDSInput *in, uint32_t read_format,
                               uint32_t draw_format, uint32_t read_samples,
                               uint32_t draw_samples, uint32_t read_type,
                               uint32_t draw_type, uint32_t read_width,
                               uint32_t read_height, uint32_t draw_width,
                               uint32_t draw_height, int read_format_packed_ds) {
    if (!in) {
        return;
    }
    memset(in, 0, sizeof(*in));
    in->read_format = read_format;
    in->draw_format = draw_format;
    in->read_samples = read_samples;
    in->draw_samples = draw_samples;
    in->read_type = read_type;
    in->draw_type = draw_type;
    in->read_width = read_width;
    in->read_height = read_height;
    in->draw_width = draw_width;
    in->draw_height = draw_height;
    in->read_format_packed_ds = read_format_packed_ds ? 1 : 0;
}

void mglBlitFillDSSubresourceInput(MGLBlitDSInput *in, uint32_t read_level,
                                   uint32_t read_slice,
                                   uint32_t read_depth_plane,
                                   uint32_t draw_level, uint32_t draw_slice,
                                   uint32_t draw_depth_plane) {
    if (!in) {
        return;
    }
    in->read_level = read_level;
    in->read_slice = read_slice;
    in->read_depth_plane = read_depth_plane;
    in->draw_level = draw_level;
    in->draw_slice = draw_slice;
    in->draw_depth_plane = draw_depth_plane;
}

void mglBlitFillDSRectInput(MGLBlitDSInput *in, int32_t src_x0, int32_t src_y0,
                            int32_t src_x1, int32_t src_y1, int32_t dst_x0,
                            int32_t dst_y0, int32_t dst_x1, int32_t dst_y1,
                            int scissor_enabled, int32_t scissor_x,
                            int32_t scissor_y, int32_t scissor_width,
                            int32_t scissor_height) {
    if (!in) {
        return;
    }
    in->src_x0 = src_x0;
    in->src_y0 = src_y0;
    in->src_x1 = src_x1;
    in->src_y1 = src_y1;
    in->dst_x0 = dst_x0;
    in->dst_y0 = dst_y0;
    in->dst_x1 = dst_x1;
    in->dst_y1 = dst_y1;
    in->scissor_enabled = scissor_enabled ? 1 : 0;
    in->scissor_x = scissor_x;
    in->scissor_y = scissor_y;
    in->scissor_width = scissor_width;
    in->scissor_height = scissor_height;
}

void mglBlitFillDSMaskInput(MGLBlitDSInput *in, int has_depth, int has_stencil,
                            int filter_is_nearest) {
    if (!in) {
        return;
    }
    in->has_depth = has_depth ? 1 : 0;
    in->has_stencil = has_stencil ? 1 : 0;
    in->filter_is_nearest = filter_is_nearest ? 1 : 0;
}

int mglBlitPlanDepthStencil(const MGLBlitDSInput *in, MGLBlitDSPlan *out) {
    if (!in || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));

    const int32_t src_w = in->src_x1 - in->src_x0;
    const int32_t src_h = in->src_y1 - in->src_y0;
    const int32_t dst_w = in->dst_x1 - in->dst_x0;
    const int32_t dst_h = in->dst_y1 - in->dst_y0;
    const int same_format = in->read_format == in->draw_format;
    const int scaled = (src_w != dst_w) || (src_h != dst_h);

    /* ---- MSAA resolve: multi-sampled source, single-sampled destination,
     * level / depth plane 0 on both sides, full unscaled rectangle.  The array
     * slice may differ (the resolve copies slice to slice). */
    if (same_format && in->read_samples > 1u && in->draw_samples <= 1u &&
        in->read_level == 0u && in->draw_level == 0u &&
        in->read_depth_plane == 0u && in->draw_depth_plane == 0u &&
        in->src_x0 == 0 && in->src_y0 == 0 && in->dst_x0 == 0 &&
        in->dst_y0 == 0 && !scaled && src_w > 0 && src_h > 0 &&
        (uint32_t)src_w <= in->read_width &&
        (uint32_t)src_h <= in->read_height &&
        (uint32_t)dst_w <= in->draw_width &&
        (uint32_t)dst_h <= in->draw_height) {
        out->msaa_resolve = 1;
        out->resolve_depth = in->has_depth ? 1 : 0;
        out->resolve_stencil =
            (in->has_stencil && in->read_format_packed_ds) ? 1 : 0;
    }

    /* ---- 1:1 copy: both single-sampled, same format, unscaled, positive
     * source size.  The GL scissor clips the destination, which moves the
     * source origin with it. */
    if (same_format && in->read_samples == 1u && in->draw_samples == 1u &&
        src_w > 0 && src_h > 0 && !scaled) {
        out->same_size_copy = 1;
        int32_t dst_x0 = in->dst_x0;
        int32_t dst_y0 = in->dst_y0;
        int32_t dst_x1 = in->dst_x1;
        int32_t dst_y1 = in->dst_y1;
        if (in->scissor_enabled) {
            dst_x0 = blit_max_i32(dst_x0, in->scissor_x);
            dst_y0 = blit_max_i32(dst_y0, in->scissor_y);
            dst_x1 = blit_min_i32(dst_x1, in->scissor_x + in->scissor_width);
            dst_y1 = blit_min_i32(dst_y1, in->scissor_y + in->scissor_height);
        }
        const int32_t copy_w = dst_x1 - dst_x0;
        const int32_t copy_h = dst_y1 - dst_y0;
        out->copy_dst_x0 = dst_x0;
        out->copy_dst_y0 = dst_y0;
        out->copy_dst_x1 = dst_x1;
        out->copy_dst_y1 = dst_y1;
        out->copy_src_x0 = in->src_x0 + (dst_x0 - in->dst_x0);
        out->copy_src_y0 = in->src_y0 + (dst_y0 - in->dst_y0);
        out->copy_valid = copy_w > 0 && copy_h > 0 &&
                          out->copy_src_x0 >= 0 && out->copy_src_y0 >= 0 &&
                          out->copy_src_x0 + copy_w <= (int32_t)in->read_width &&
                          out->copy_src_y0 + copy_h <= (int32_t)in->read_height &&
                          dst_x0 >= 0 && dst_y0 >= 0 &&
                          dst_x1 <= (int32_t)in->draw_width &&
                          dst_y1 <= (int32_t)in->draw_height;
    }

    /* ---- Scaled render: single-sampled both sides, scaled, GL_NEAREST (the
     * GL spec forbids filtering for depth), 2D textures at level / slice /
     * depth plane 0. */
    if (same_format && in->read_samples == 1u && in->draw_samples == 1u &&
        src_w > 0 && src_h > 0 && scaled && in->filter_is_nearest &&
        in->read_level == 0u && in->read_slice == 0u &&
        in->read_depth_plane == 0u && in->read_type == MGLTextureType2D &&
        in->draw_level == 0u && in->draw_slice == 0u &&
        in->draw_depth_plane == 0u && in->draw_type == MGLTextureType2D) {
        out->scaled_render = 1;
    }

    return 0;
}
