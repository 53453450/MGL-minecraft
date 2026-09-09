/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Clear-value regression protection harness for O3.1
 * (mglRenderPassPlanClearValues).  No Metal / no ObjC required.
 *
 * This is the behavioural oracle that gates moving the
 * load / store / clear / attachment-match decisions out of
 * MGLRenderer+RenderPass.m into the C plan layer.  It drives the plan function
 * directly with hand-populated MGLRenderPassState goldens, so it does NOT
 * depend on the full regression suite (which is currently truncated by the
 * known upstream SIGSEGV at [16/93]).
 */

#include "mgl_render_pass_clear.h"

#include <stdio.h>
#include <string.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static MGLRenderPassState base_state(void)
{
    MGLRenderPassState s;
    memset(&s, 0, sizeof(s));
    s.color[0].clear_red = 0.10;
    s.color[0].clear_green = 0.20;
    s.color[0].clear_blue = 0.30;
    s.color[0].clear_alpha = 0.40;
    s.color[3].clear_red = 1.00;
    s.color[3].clear_green = 0.90;
    s.color[3].clear_blue = 0.80;
    s.color[3].clear_alpha = 0.70;
    s.depth.clear_depth = 0.5;
    s.stencil.clear_stencil = 7u;
    return s;
}

static void test_color_clear_values(void)
{
    MGLRenderPassState s = base_state();
    double rgba[4] = {-1.0, -1.0, -1.0, -1.0};

    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0u,
               rgba, NULL, NULL) == 1,
           "color attachment 0 resolves");
    expect(rgba[0] == 0.10 && rgba[1] == 0.20 && rgba[2] == 0.30 &&
               rgba[3] == 0.40,
           "color attachment 0 RGBA");

    double rgba3[4] = {-1.0, -1.0, -1.0, -1.0};
    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 3u,
               rgba3, NULL, NULL) == 1,
           "color attachment 3 resolves");
    expect(rgba3[0] == 1.00 && rgba3[1] == 0.90 && rgba3[2] == 0.80 &&
               rgba3[3] == 0.70,
           "color attachment 3 RGBA");

    /* Untouched slots stay zero (state was memset). */
    double rgba1[4] = {-1.0, -1.0, -1.0, -1.0};
    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 1u,
               rgba1, NULL, NULL) == 1,
           "color attachment 1 resolves (zeroed)");
    expect(rgba1[0] == 0.0 && rgba1[1] == 0.0 && rgba1[2] == 0.0 &&
               rgba1[3] == 0.0,
           "color attachment 1 is zero");
}

static void test_depth_stencil_clear_values(void)
{
    MGLRenderPassState s = base_state();
    double depth = -1.0;
    uint32_t stencil = 99u;

    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u,
               NULL, &depth, NULL) == 1,
           "depth resolves");
    expect(depth == 0.5, "depth clear value");

    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u,
               NULL, NULL, &stencil) == 1,
           "stencil resolves");
    expect(stencil == 7u, "stencil clear value");
}

static void test_rejects(void)
{
    MGLRenderPassState s = base_state();
    double rgba[4];
    memset(rgba, 0, sizeof(rgba));

    /* Out-of-range color index must be rejected (MAX_COLOR_ATTACHMENTS == 8). */
    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 8u,
               rgba, NULL, NULL) == 0,
           "color index 8 rejected (== MAX)");
    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 100u,
               rgba, NULL, NULL) == 0,
           "color index 100 rejected");

    /* Unrecognized attachment kind must be rejected. */
    expect(mglRenderPassPlanClearValues(&s, 42u, 0u, rgba, NULL, NULL) == 0,
           "unknown attachment kind rejected");

    /* NULL state must be rejected. */
    expect(mglRenderPassPlanClearValues(
               NULL, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0u,
               rgba, NULL, NULL) == 0,
           "NULL state rejected");
}

static void test_null_out_params_tolerated(void)
{
    MGLRenderPassState s = base_state();

    /* Output pointers are optional: the plan must still report success. */
    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0u,
               NULL, NULL, NULL) == 1,
           "color with NULL out params still succeeds");
    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u,
               NULL, NULL, NULL) == 1,
           "depth with NULL out params still succeeds");
    expect(mglRenderPassPlanClearValues(
               &s, (uint32_t)MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u,
               NULL, NULL, NULL) == 1,
           "stencil with NULL out params still succeeds");
}

int main(void)
{
    test_color_clear_values();
    test_depth_stencil_clear_values();
    test_rejects();
    test_null_out_params_tolerated();

    if (g_fails) {
        fprintf(stderr, "clear-value harness: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("clear-value harness: all clear-value cases passed\n");
    return 0;
}
