/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Sampler plan harness (docs/AUDIT_LAYERING_2026-10-17.md §8.7).
 *
 * Locks the GL -> MGLSamplerPlan translation that used to live inside
 * mglRenderCreateSamplerForGL: the mip choice packed into the minification
 * filter, the five wrap modes, the compare mode/function gate, the rectangle
 * target's non-normalized coordinates, and the two aggregate flags the
 * encoder uses (border colour chosen / force clamp-to-edge).
 *
 * Build: see `make test-sampler-plan`.
 */

#include <stddef.h>

#include "glm_params.h"

/* mgl_types_texture.h expects the project's context typedef to be in scope;
 * defining it here keeps the harness from dragging in glm_context.h. */
typedef struct GLMContextRec_t *GLMContext;

#include "mgl_types_texture.h"
#include "mgl_sampler_plan.h"

#include <stdio.h>
#include <string.h>

static int tests_run;
static int tests_failed;

static void expect(int ok, const char *what)
{
    tests_run++;
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        tests_failed++;
    }
}

static TextureParameter default_params(void)
{
    TextureParameter p;
    memset(&p, 0, sizeof(p));
    p.min_filter = GL_NEAREST;
    p.mag_filter = GL_NEAREST;
    p.wrap_s = GL_REPEAT;
    p.wrap_t = GL_REPEAT;
    p.wrap_r = GL_REPEAT;
    p.compare_mode = GL_NONE;
    p.compare_func = GL_LEQUAL;
    p.min_lod = -0.5f;
    p.max_lod = 2000.0f;
    p.max_anisotropy = 8.0f;
    return p;
}

static void test_filters(void)
{
    static const struct {
        uint32_t gl_min;
        uint8_t filter;
        uint8_t mip;
    } cases[] = {
        {GL_NEAREST, MGL_SAMPLER_FILTER_NEAREST, MGL_SAMPLER_MIP_NONE},
        {GL_LINEAR, MGL_SAMPLER_FILTER_LINEAR, MGL_SAMPLER_MIP_NONE},
        {GL_NEAREST_MIPMAP_NEAREST, MGL_SAMPLER_FILTER_NEAREST, MGL_SAMPLER_MIP_NEAREST},
        {GL_LINEAR_MIPMAP_NEAREST, MGL_SAMPLER_FILTER_LINEAR, MGL_SAMPLER_MIP_NEAREST},
        {GL_NEAREST_MIPMAP_LINEAR, MGL_SAMPLER_FILTER_NEAREST, MGL_SAMPLER_MIP_LINEAR},
        {GL_LINEAR_MIPMAP_LINEAR, MGL_SAMPLER_FILTER_LINEAR, MGL_SAMPLER_MIP_LINEAR},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        TextureParameter p = default_params();
        MGLSamplerPlan plan;
        p.min_filter = cases[i].gl_min;
        expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == 0, "min filter builds");
        expect(plan.min_filter == cases[i].filter, "min filter mapped");
        expect(plan.mip_filter == cases[i].mip, "mip filter split out");
    }

    TextureParameter p = default_params();
    MGLSamplerPlan plan;
    p.min_filter = 0x1234;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == -1, "bad min filter rejected");
    p = default_params();
    p.mag_filter = 0x1234;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == -1, "bad mag filter rejected");
}

static void test_wrap(void)
{
    static const struct {
        uint32_t gl;
        uint8_t mgl;
    } cases[] = {
        {GL_REPEAT, MGL_SAMPLER_WRAP_REPEAT},
        {GL_MIRRORED_REPEAT, MGL_SAMPLER_WRAP_MIRRORED_REPEAT},
        {GL_CLAMP_TO_EDGE, MGL_SAMPLER_WRAP_CLAMP_TO_EDGE},
        {GL_CLAMP_TO_BORDER, MGL_SAMPLER_WRAP_CLAMP_TO_BORDER},
        {GL_MIRROR_CLAMP_TO_EDGE, MGL_SAMPLER_WRAP_MIRROR_CLAMP_TO_EDGE},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        TextureParameter p = default_params();
        MGLSamplerPlan plan;
        p.wrap_s = cases[i].gl;
        expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == 0, "wrap builds");
        expect(plan.wrap_s == cases[i].mgl, "wrap s mapped");
        expect(plan.wrap_t == MGL_SAMPLER_WRAP_REPEAT, "wrap t independent");
    }

    TextureParameter p = default_params();
    MGLSamplerPlan plan;
    p.wrap_r = 0x1234;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == -1, "bad wrap rejected");
}

static void test_compare(void)
{
    TextureParameter p = default_params();
    MGLSamplerPlan plan;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == 0, "no compare builds");
    expect(plan.compare == MGL_SAMPLER_COMPARE_NONE, "no compare");

    p.compare_mode = GL_COMPARE_REF_TO_TEXTURE;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == 0, "compare builds");
    expect(plan.compare == MGL_SAMPLER_COMPARE_LEQUAL, "compare func mapped");
    /* The encoder casts (compare - 1) to MTL::CompareFunction; verify the
     * order contract for the two ends. */
    expect(MGL_SAMPLER_COMPARE_NEVER - 1 == 0, "never is metal 0");
    expect(MGL_SAMPLER_COMPARE_ALWAYS - 1 == 7, "always is metal 7");

    p.compare_func = 0x1234;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == -1, "bad compare func rejected");

    p = default_params();
    p.compare_mode = GL_LINEAR;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == -1, "bad compare mode rejected");
}

static void test_target_and_flags(void)
{
    TextureParameter p = default_params();
    MGLSamplerPlan plan;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_RECTANGLE, &plan) == 0, "rectangle builds");
    expect(plan.normalized_coordinates == 0, "rectangle is not normalized");
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == 0, "2d builds");
    expect(plan.normalized_coordinates == 1, "2d is normalized");

    p = default_params();
    p.wrap_s = GL_CLAMP_TO_EDGE;
    p.wrap_t = GL_CLAMP_TO_EDGE;
    p.wrap_r = GL_CLAMP_TO_EDGE;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_RECTANGLE, &plan) == 0, "rect edge builds");
    expect(plan.all_wrap_clamp_to_edge == 1, "all edge");
    expect(plan.any_wrap_border == 0, "no border wrap");

    p.wrap_t = GL_CLAMP_TO_BORDER;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == 0, "border builds");
    expect(plan.any_wrap_border == 1, "border wrap flagged");
    expect(plan.all_wrap_clamp_to_edge == 0, "not all edge");

    p.border_color[0] = 0.25f;
    p.border_color[3] = 1.0f;
    p.min_lod = -2.0f;
    p.max_lod = 100.0f;
    p.max_anisotropy = 4.0f;
    expect(mglSamplerPlanBuild(&p, GL_TEXTURE_2D, &plan) == 0, "values build");
    expect(plan.border_color[0] == 0.25f && plan.border_color[3] == 1.0f, "border copied");
    expect(plan.min_lod == -2.0f && plan.max_lod == 100.0f, "lod copied");
    expect(plan.max_anisotropy == 4.0f, "aniso copied");
}

int main(void)
{
    test_filters();
    test_wrap();
    test_compare();
    test_target_and_flags();

    if (tests_run == 0) {
        fprintf(stderr, "test_sampler_plan: no tests ran\n");
        return 1;
    }
    printf("test_sampler_plan: %d checks, %d failed\n", tests_run,
           tests_failed);
    return tests_failed ? 1 : 0;
}
