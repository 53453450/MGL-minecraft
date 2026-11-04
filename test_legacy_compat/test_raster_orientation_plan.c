/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Raster orientation plan harness (FBO row-order redesign, Phase 1).
 *
 * With a window-system target the plan must reproduce what the encoder state
 * code computed inline before it existed: mglRenderMetalScissorY,
 * mglRenderMetalViewportY + mglRenderApplyClipOriginToViewport and the
 * "invert the winding for UPPER_LEFT" rule.  The sweep checks that against
 * those functions, which are kept as the oracle.  The GL-order cases pin the
 * new rules.
 *
 * Build: see `make test-raster-orientation-plan`.
 */

#include "mgl_raster_orientation_plan.h"

#include <stdio.h>
#include <string.h>

#include "mgl_pso_format_class.h"
#include "mgl_program_resource.h"

/* The format-class file reads a program's builtin mask in code this harness
 * does not reach. */
uint32_t mglProgramStageBuiltinMask(const Program *program, int stage)
{
    (void)program;
    (void)stage;
    return 0u;
}

static int tests_run;
static int tests_passed;

#define CHECK(cond, label)                                       \
    do {                                                         \
        tests_run++;                                             \
        if (cond) {                                              \
            tests_passed++;                                      \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

static MGLRasterOrientationPlan plan_for(uint32_t origin, int gl_order)
{
    MGLRasterOrientationInput in = {origin, gl_order};
    MGLRasterOrientationPlan plan;
    memset(&plan, 0xAB, sizeof(plan));
    mglRasterOrientationPlanBuild(&in, &plan);
    return plan;
}

static void test_window_target_matches_inline_math(void)
{
    static const uint32_t origins[2] = {GL_LOWER_LEFT, GL_UPPER_LEFT};
    int agree = 1;
    int cases = 0;
    for (int o = 0; o < 2; o++) {
        const MGLRasterOrientationPlan plan = plan_for(origins[o], 0);
        for (uint32_t pass_h = 1; pass_h <= 70; pass_h += 3)
        for (int32_t y = -4; y <= 80; y += 3)
        for (int32_t h = 0; h <= 80; h += 7) {
            cases++;
            if (mglRasterPlanScissorY(&plan, y, h, pass_h) !=
                mglRenderMetalScissorY(y, h, pass_h, origins[o])) {
                agree = 0;
            }
            double py = (double)y, ph = (double)h;
            double oy = mglRenderMetalViewportY((double)y, (double)h, pass_h);
            double oh = (double)h;
            mglRenderApplyClipOriginToViewport(&oy, &oh, origins[o]);
            mglRasterPlanViewportY(&plan, &py, &ph, pass_h);
            if (py != oy || ph != oh) {
                agree = 0;
            }
        }
        /* The winding rule: invert exactly for UPPER_LEFT. */
        if ((plan.invert_winding != 0) != (origins[o] == GL_UPPER_LEFT)) {
            agree = 0;
        }
        /* Today's gl_FragCoord flip: exactly the lower-left clip origin. */
        if ((plan.fragcoord_from_top != 0) != (origins[o] == GL_LOWER_LEFT)) {
            agree = 0;
        }
    }
    printf("  swept %d window-target inputs\n", cases);
    CHECK(agree, "a window-system target reproduces the inline scissor, "
                 "viewport and winding math");
}

static void test_gl_order_target_rules(void)
{
    MGLRasterOrientationPlan p = plan_for(GL_LOWER_LEFT, 1);
    CHECK(!p.scissor_from_top && !p.viewport_from_top &&
              p.negative_viewport_height && p.invert_winding &&
              !p.fragcoord_from_top,
          "GL-order target, lower-left: GL y direct, viewport negated, "
          "winding inverted, gl_FragCoord.y is the row");
    CHECK(mglRasterPlanScissorY(&p, 5, 10, 64) == 5,
          "GL-order scissor y is GL's y");
    double y = 5.0, h = 10.0;
    mglRasterPlanViewportY(&p, &y, &h, 64);
    CHECK(y == 15.0 && h == -10.0,
          "GL-order viewport: origin at the far edge, height negated");

    p = plan_for(GL_UPPER_LEFT, 1);
    CHECK(!p.scissor_from_top && !p.viewport_from_top &&
              !p.negative_viewport_height && !p.invert_winding &&
              !p.fragcoord_from_top,
          "GL-order target, upper-left: the two inversions cancel and Metal's "
          "native mapping applies");
    y = 5.0;
    h = 10.0;
    mglRasterPlanViewportY(&p, &y, &h, 64);
    CHECK(y == 5.0 && h == 10.0,
          "GL-order upper-left viewport is GL's rectangle, positive height");

    p = plan_for(GL_LOWER_LEFT, 0);
    CHECK(p.scissor_from_top && p.viewport_from_top &&
              !p.negative_viewport_height && !p.invert_winding &&
              p.fragcoord_from_top,
          "window target, lower-left: the default");
}

static void test_edges(void)
{
    MGLRasterOrientationPlan p = plan_for(GL_LOWER_LEFT, 0);
    CHECK(mglRasterPlanScissorY(&p, 60, 10, 64) == 0,
          "a scissor past the top clamps at row 0");
    double y = 60.0, h = 10.0;
    mglRasterPlanViewportY(&p, &y, &h, 64);
    CHECK(y == 0.0 && h == 10.0, "a viewport past the top clamps at row 0");
    y = 3.0;
    h = 0.0;
    p = plan_for(GL_UPPER_LEFT, 0);
    mglRasterPlanViewportY(&p, &y, &h, 64);
    CHECK(h == 0.0, "a zero-height viewport is not negated");

    MGLRasterOrientationPlan out;
    MGLRasterOrientationInput in = {GL_LOWER_LEFT, 0};
    CHECK(mglRasterOrientationPlanBuild(NULL, &out) == -1 &&
              mglRasterOrientationPlanBuild(&in, NULL) == -1,
          "NULL arguments are refused");
    mglRasterPlanViewportY(NULL, &y, &h, 64);
    CHECK(mglRasterPlanScissorY(NULL, 7, 3, 64) == 7,
          "a NULL plan leaves scissor y alone");
}

int main(void)
{
    printf("raster orientation plan harness (FBO row-order redesign, phase 1)\n");
    test_window_target_matches_inline_math();
    test_gl_order_target_rules();
    test_edges();
    printf("test_raster_orientation_plan: %d/%d passed\n", tests_passed,
           tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
