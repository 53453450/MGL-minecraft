/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_blit_plan.h — O4.4: depth/stencil blit gate planning (pure C, no Metal).
 *
 * glBlitFramebuffer with a depth and/or stencil mask takes one of three paths:
 * an MSAA resolve through a render pass, a 1:1 copy through a blit encoder, or
 * a scaled draw through a depth-writing pipeline.  Which paths the current
 * rectangle and textures allow, the scissor-clipped copy rectangle and the
 * resolve arms are decided here; the ObjC side only materializes the encoders.
 */

#ifndef MGL_BLIT_PLAN_H
#define MGL_BLIT_PLAN_H

#include <stdint.h>

#include "mgl_render_values.h" /* MGLTextureType* */

#ifdef __cplusplus
extern "C" {
#endif

/* Facts the three depth/stencil gates are evaluated against. */
typedef struct MGLBlitDSInput {
    /* textures */
    uint32_t read_format;
    uint32_t draw_format;
    uint32_t read_samples;
    uint32_t draw_samples;
    uint32_t read_type;
    uint32_t draw_type;
    uint32_t read_width;
    uint32_t read_height;
    uint32_t draw_width;
    uint32_t draw_height;
    int read_format_packed_ds; /* pixel format carries depth + stencil */
    /* subresources */
    uint32_t read_level;
    uint32_t read_slice;
    uint32_t read_depth_plane;
    uint32_t draw_level;
    uint32_t draw_slice;
    uint32_t draw_depth_plane;
    /* rectangles, in GL (bottom-up) coordinates */
    int32_t src_x0;
    int32_t src_y0;
    int32_t src_x1;
    int32_t src_y1;
    int32_t dst_x0;
    int32_t dst_y0;
    int32_t dst_x1;
    int32_t dst_y1;
    /* scissor box, applied to the 1:1 copy only */
    int scissor_enabled;
    int32_t scissor_x;
    int32_t scissor_y;
    int32_t scissor_width;
    int32_t scissor_height;
    /* mask and filter */
    int has_depth;
    int has_stencil;
    int filter_is_nearest;
} MGLBlitDSInput;

typedef struct MGLBlitDSPlan {
    /* MSAA resolve: read is multisampled, draw is not, same format, level and
     * depth plane 0 on both sides, unscaled, origin 0, inside both textures. */
    int msaa_resolve;
    int resolve_depth;   /* the mask asked for depth */
    int resolve_stencil; /* stencil asked for, and the source is packed DS */
    /* 1:1 copy: both single-sampled, same format, unscaled, positive source
     * size.  copy_valid says the scissor-clipped rectangle is inside both
     * textures; the rectangles are meaningful either way. */
    int same_size_copy;
    int copy_valid;
    int32_t copy_dst_x0;
    int32_t copy_dst_y0;
    int32_t copy_dst_x1;
    int32_t copy_dst_y1;
    int32_t copy_src_x0;
    int32_t copy_src_y0;
    /* scaled render: single-sampled both sides, same format, scaled,
     * GL_NEAREST, 2D textures with level / slice / depth plane 0. */
    int scaled_render;
} MGLBlitDSPlan;

/* Grouped fills, mirroring the other plan modules: the texture fill clears the
 * input and must come first, the rest set their own fields. */
void mglBlitFillDSTextureInput(MGLBlitDSInput *in, uint32_t read_format,
                               uint32_t draw_format, uint32_t read_samples,
                               uint32_t draw_samples, uint32_t read_type,
                               uint32_t draw_type, uint32_t read_width,
                               uint32_t read_height, uint32_t draw_width,
                               uint32_t draw_height, int read_format_packed_ds);

void mglBlitFillDSSubresourceInput(MGLBlitDSInput *in, uint32_t read_level,
                                   uint32_t read_slice,
                                   uint32_t read_depth_plane,
                                   uint32_t draw_level, uint32_t draw_slice,
                                   uint32_t draw_depth_plane);

void mglBlitFillDSRectInput(MGLBlitDSInput *in, int32_t src_x0, int32_t src_y0,
                            int32_t src_x1, int32_t src_y1, int32_t dst_x0,
                            int32_t dst_y0, int32_t dst_x1, int32_t dst_y1,
                            int scissor_enabled, int32_t scissor_x,
                            int32_t scissor_y, int32_t scissor_width,
                            int32_t scissor_height);

void mglBlitFillDSMaskInput(MGLBlitDSInput *in, int has_depth, int has_stencil,
                            int filter_is_nearest);

/* Fills every gate; returns 0 on success, -1 on bad arguments. */
int mglBlitPlanDepthStencil(const MGLBlitDSInput *in, MGLBlitDSPlan *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BLIT_PLAN_H */
