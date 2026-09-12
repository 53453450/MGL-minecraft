/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Depth/stencil blit gate harness (O4.4).
 *
 * Golden for mglBlitPlanDepthStencil(): the three paths a glBlitFramebuffer
 * with a depth/stencil mask can take, plus the scissor-clipped copy rectangle
 * and the resolve arms.  The expectations were read off
 * MGLRenderer+Blit.m's blitFramebufferDepthStencil: before the gates moved
 * into the plan:
 *
 *   MSAA resolve  same format, read samples > 1, draw samples <= 1, level and
 *                 depth plane 0 on both sides, origin 0, unscaled, inside both
 *                 textures -- a differing array slice is allowed;
 *   same-size     same format, both single sampled, unscaled, positive source
 *                 size; the GL scissor clips the destination and drags the
 *                 source origin with it, and the clipped rectangle has to be
 *                 inside both textures to be usable;
 *   scaled        same format, both single sampled, scaled, GL_NEAREST, 2D
 *                 textures with level / slice / depth plane 0.
 *
 * Build: see `make test-blit-plan`.
 */

#include "mgl_blit_plan.h"

#include <stdio.h>
#include <string.h>

static int tests_run;
static int tests_passed;

#define CHECK(cond, label)                                       \
    do {                                                         \
        tests_run++;                                             \
        if (cond) {                                              \
            tests_passed++;                                      \
            printf("  [PASS] %s\n", (label));                    \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

/* A 64x64 single-sampled 2D source blitting the full rectangle to a matching
 * destination: the plain same-size copy case. */
static MGLBlitDSInput base_input(void)
{
    MGLBlitDSInput in;
    memset(&in, 0, sizeof(in));
    in.read_format = 100u;
    in.draw_format = 100u;
    in.read_samples = 1u;
    in.draw_samples = 1u;
    in.read_type = MGLTextureType2D;
    in.draw_type = MGLTextureType2D;
    in.read_width = 64u;
    in.read_height = 64u;
    in.draw_width = 64u;
    in.draw_height = 64u;
    in.src_x1 = 64;
    in.src_y1 = 64;
    in.dst_x1 = 64;
    in.dst_y1 = 64;
    in.has_depth = 1;
    in.filter_is_nearest = 1;
    return in;
}

static void test_same_size_copy(void)
{
    MGLBlitDSPlan plan;
    MGLBlitDSInput in = base_input();

    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "copy plan ok");
    CHECK(plan.same_size_copy == 1 && plan.copy_valid == 1,
          "an unscaled full copy is usable");
    CHECK(plan.copy_dst_x0 == 0 && plan.copy_dst_y0 == 0 &&
              plan.copy_dst_x1 == 64 && plan.copy_dst_y1 == 64,
          "the copy rectangle is the destination");
    CHECK(plan.copy_src_x0 == 0 && plan.copy_src_y0 == 0,
          "an unclipped copy keeps the source origin");
    CHECK(plan.msaa_resolve == 0 && plan.scaled_render == 0,
          "a single-sampled unscaled pair is neither resolve nor scaled");

    /* A sub-rectangle copy: the source origin follows the destination. */
    in = base_input();
    in.src_x0 = 8;
    in.src_y0 = 4;
    in.src_x1 = 24;
    in.src_y1 = 20;
    in.dst_x0 = 16;
    in.dst_y0 = 32;
    in.dst_x1 = 32;
    in.dst_y1 = 48;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "sub-rect plan ok");
    CHECK(plan.copy_dst_x0 == 16 && plan.copy_dst_y0 == 32 &&
              plan.copy_src_x0 == 8 && plan.copy_src_y0 == 4,
          "the copy keeps origin and source");

    /* The scissor clips the destination and moves the source with it. */
    in = base_input();
    in.dst_x1 = 64;
    in.dst_y1 = 64;
    in.scissor_enabled = 1;
    in.scissor_x = 10;
    in.scissor_y = 20;
    in.scissor_width = 30;
    in.scissor_height = 24;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "scissor plan ok");
    CHECK(plan.copy_dst_x0 == 10 && plan.copy_dst_y0 == 20 &&
              plan.copy_dst_x1 == 40 && plan.copy_dst_y1 == 44,
          "the scissor clips the destination");
    CHECK(plan.copy_src_x0 == 10 && plan.copy_src_y0 == 20,
          "the clipped destination drags the source");
    CHECK(plan.copy_valid == 1, "a clipped copy inside both is usable");

    /* A source rectangle that starts before the texture makes the copy
     * unusable even though the sizes still line up. */
    in = base_input();
    in.src_x0 = -8;   /* src width 24 to keep it unscaled */
    in.src_x1 = 16;
    in.dst_x1 = 24;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "bad source plan ok");
    CHECK(plan.same_size_copy == 1 && plan.copy_valid == 0,
          "a copy starting before the source texture is refused");
    CHECK(plan.copy_src_x0 == -8, "the refused copy still reports its origin");

    /* An empty rectangle is not a copy. */
    in = base_input();
    in.dst_x1 = in.dst_x0;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "empty plan ok");
    CHECK(plan.copy_valid == 0, "an empty scissor result is refused");

    /* A different format takes none of the paths. */
    in = base_input();
    in.draw_format = 101u;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "format plan ok");
    CHECK(plan.same_size_copy == 0 && plan.msaa_resolve == 0 &&
              plan.scaled_render == 0,
          "a format mismatch takes no path");
}

static void test_msaa_resolve(void)
{
    MGLBlitDSPlan plan;
    MGLBlitDSInput in = base_input();
    in.read_samples = 4u;

    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "resolve plan ok");
    CHECK(plan.msaa_resolve == 1, "a multisampled source resolves");
    CHECK(plan.resolve_depth == 1, "the depth arm follows the mask");
    CHECK(plan.resolve_stencil == 0,
          "a depth-only source has no stencil arm");
    CHECK(plan.same_size_copy == 0 && plan.scaled_render == 0,
          "a multisampled source is not copied or scaled");

    /* Packed depth+stencil with both mask bits resolves both. */
    in.has_stencil = 1;
    in.read_format_packed_ds = 1;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "packed plan ok");
    CHECK(plan.resolve_depth == 1 && plan.resolve_stencil == 1,
          "a packed source resolves depth and stencil");

    /* Stencil without a packed source has no arm of its own. */
    in.has_depth = 0;
    in.has_stencil = 1;
    in.read_format_packed_ds = 0;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "stencil plan ok");
    CHECK(plan.msaa_resolve == 1 && plan.resolve_depth == 0 &&
              plan.resolve_stencil == 0,
          "an unpacked stencil has no resolve arm");

    /* A differing array slice is allowed through the resolve. */
    in = base_input();
    in.read_samples = 4u;
    in.read_slice = 2u;
    in.draw_slice = 3u;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "slice plan ok");
    CHECK(plan.msaa_resolve == 1, "the slice may differ");

    /* Level and depth plane may not. */
    in = base_input();
    in.read_samples = 4u;
    in.read_level = 1u;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "level plan ok");
    CHECK(plan.msaa_resolve == 0, "a non-zero level blocks the resolve");
    in = base_input();
    in.read_samples = 4u;
    in.draw_depth_plane = 1u;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "plane plan ok");
    CHECK(plan.msaa_resolve == 0, "a non-zero depth plane blocks the resolve");

    /* Origin and scaling may not differ. */
    in = base_input();
    in.read_samples = 4u;
    in.src_x0 = 1;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "origin plan ok");
    CHECK(plan.msaa_resolve == 0, "a non-zero source origin blocks the resolve");
    in = base_input();
    in.read_samples = 4u;
    in.dst_x1 = 32;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "scaled resolve plan ok");
    CHECK(plan.msaa_resolve == 0, "a scaled resolve is refused");

    /* The rectangle has to fit inside both textures. */
    in = base_input();
    in.read_samples = 4u;
    in.read_width = 32u;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "oversized plan ok");
    CHECK(plan.msaa_resolve == 0, "a rectangle past the source blocks it");
}

static void test_scaled_render(void)
{
    MGLBlitDSPlan plan;
    MGLBlitDSInput in = base_input();
    in.dst_x1 = 128;
    in.dst_y1 = 128;

    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "scaled plan ok");
    CHECK(plan.scaled_render == 1, "a scaled nearest 2D pair renders");
    CHECK(plan.same_size_copy == 0 && plan.msaa_resolve == 0,
          "a scaled pair is not copied or resolved");

    in.filter_is_nearest = 0;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "linear plan ok");
    CHECK(plan.scaled_render == 0, "linear filtering blocks the scaled path");

    in = base_input();
    in.dst_y1 = 128;
    in.read_type = MGLTextureType2DArray;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "array plan ok");
    CHECK(plan.scaled_render == 0, "a non-2D source blocks the scaled path");

    in = base_input();
    in.dst_y1 = 128;
    in.draw_slice = 1u;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "draw slice plan ok");
    CHECK(plan.scaled_render == 0, "a non-zero draw slice blocks the scaled path");

    in = base_input();
    in.dst_x1 = 128;
    in.read_samples = 2u;
    in.draw_samples = 2u;
    CHECK(mglBlitPlanDepthStencil(&in, &plan) == 0, "ms scaled plan ok");
    CHECK(plan.scaled_render == 0, "multisampled input blocks the scaled path");

    /* The grouped fills are what the caller uses: the texture fill clears the
     * input, the rest set their own groups. */
    {
        MGLBlitDSInput filled;
        mglBlitFillDSTextureInput(&filled, 100u, 100u, 1u, 1u, MGLTextureType2D,
                                  MGLTextureType2D, 64u, 64u, 64u, 64u, 1);
        mglBlitFillDSSubresourceInput(&filled, 2u, 3u, 4u, 5u, 6u, 7u);
        mglBlitFillDSRectInput(&filled, 1, 2, 33, 34, 5, 6, 37, 38, 1, 7, 8, 9,
                               10);
        mglBlitFillDSMaskInput(&filled, 1, 1, 0);
        CHECK(filled.read_format == 100u && filled.draw_width == 64u &&
                  filled.read_format_packed_ds == 1,
              "fill textures");
        CHECK(filled.read_level == 2u && filled.read_slice == 3u &&
                  filled.draw_depth_plane == 7u,
              "fill subresources");
        CHECK(filled.src_x0 == 1 && filled.dst_x1 == 37 &&
                  filled.scissor_enabled == 1 && filled.scissor_width == 9,
              "fill rect and scissor");
        CHECK(filled.has_depth == 1 && filled.has_stencil == 1 &&
                  filled.filter_is_nearest == 0,
              "fill mask and filter");
        MGLBlitDSPlan filledPlan;
        CHECK(mglBlitPlanDepthStencil(&filled, &filledPlan) == 0 &&
                  filledPlan.scaled_render == 0,
              "a filled scaled-linear input takes no scaled path");
    }
    mglBlitFillDSTextureInput(NULL, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0);
    mglBlitFillDSSubresourceInput(NULL, 0u, 0u, 0u, 0u, 0u, 0u);
    mglBlitFillDSRectInput(NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    mglBlitFillDSMaskInput(NULL, 0, 0, 0);
    CHECK(1, "the fills tolerate a NULL input");

    /* Degenerate inputs. */
    CHECK(mglBlitPlanDepthStencil(NULL, &plan) == -1, "NULL input is refused");
    in = base_input();
    CHECK(mglBlitPlanDepthStencil(&in, NULL) == -1, "NULL output is refused");
}

int main(void)
{
    printf("depth/stencil blit gate harness (O4.4)\n");
    test_same_size_copy();
    test_msaa_resolve();
    test_scaled_render();
    printf("test_blit_plan: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
