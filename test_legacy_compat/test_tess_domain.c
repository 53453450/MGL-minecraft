/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Golden vectors for TessFactorNormalize + TessDomainGenerator.
 * Spec: OpenGL 4.6 §11.2.2.1 / §11.2.2.2 / §11.2.2.3.
 */

#include "mgl_tess_domain.h"

#include "glcorearb.h"

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static int feq(float a, float b)
{
    return fabsf(a - b) < 1e-5f;
}

static MGLTessFactorInput make_in(uint32_t mode, uint32_t spacing,
                                  uint32_t winding, uint32_t point_mode,
                                  float o0, float o1, float o2, float o3,
                                  float i0, float i1)
{
    MGLTessFactorInput in;
    memset(&in, 0, sizeof(in));
    in.gen_mode = mode;
    in.spacing = spacing;
    in.winding = winding;
    in.point_mode = point_mode;
    in.outer[0] = o0;
    in.outer[1] = o1;
    in.outer[2] = o2;
    in.outer[3] = o3;
    in.inner[0] = i0;
    in.inner[1] = i1;
    return in;
}

static void test_round_spacing(void)
{
    expect(mglTessRoundLevelForSpacing(GL_EQUAL, 1u) == 1u, "equal 1");
    expect(mglTessRoundLevelForSpacing(GL_EQUAL, 3u) == 3u, "equal 3");
    expect(mglTessRoundLevelForSpacing(GL_FRACTIONAL_ODD, 1u) == 1u, "FO 1");
    expect(mglTessRoundLevelForSpacing(GL_FRACTIONAL_ODD, 2u) == 3u, "FO 2→3");
    expect(mglTessRoundLevelForSpacing(GL_FRACTIONAL_EVEN, 1u) == 2u, "FE 1→2");
    expect(mglTessRoundLevelForSpacing(GL_FRACTIONAL_EVEN, 3u) == 4u, "FE 3→4");
}

static void test_inner_eps(void)
{
    MGLTessNormalizedFactors n;
    MGLTessFactorInput in;

    /* All levels 1: degenerate — no 1+ε bump. */
    in = make_in(GL_QUADS, GL_FRACTIONAL_ODD, GL_CCW, 0, 1, 1, 1, 1, 1, 1);
    mglTessNormalizeFactors(&in, &n);
    expect(!n.discard && n.all_levels_one, "all-one flag");
    expect(n.inner_eff[0] == 1u && n.inner_eff[1] == 1u, "all-one FO inner=1");

    /* Inner clamped to 1, outers >1 → 1+ε → FO rounds to 3. */
    in = make_in(GL_QUADS, GL_FRACTIONAL_ODD, GL_CCW, 0, 4, 4, 4, 4, 1, 1);
    mglTessNormalizeFactors(&in, &n);
    expect(!n.discard && !n.all_levels_one, "bump case not all-one");
    expect(n.inner_eff[0] == 3u && n.inner_eff[1] == 3u,
           "§11.2.2.1 FO inner 1+ε → 3");

    in = make_in(GL_QUADS, GL_EQUAL, GL_CCW, 0, 4, 4, 4, 4, 1, 1);
    mglTessNormalizeFactors(&in, &n);
    expect(n.inner_eff[0] == 2u && n.inner_eff[1] == 2u,
           "§11.2.2.1 equal inner 1+ε → 2");

    in = make_in(GL_TRIANGLES, GL_FRACTIONAL_ODD, GL_CCW, 0, 3, 3, 3, 1, 1, 1);
    mglTessNormalizeFactors(&in, &n);
    expect(n.inner_eff[0] == 3u, "tri FO inner 1+ε → 3");
}

static void test_discard(void)
{
    MGLTessNormalizedFactors n;
    MGLTessFactorInput in =
        make_in(GL_QUADS, GL_EQUAL, GL_CCW, 0, 0, 1, 1, 1, 1, 1);
    mglTessNormalizeFactors(&in, &n);
    expect(n.discard, "outer 0 discards");
    expect(mglTessDomainVertexCount(&in) == 0u, "discard → 0 verts");
}

/* Point mode emits boundary vertices and every interior mesh vertex. */
static void test_point_mode_counts(void)
{
    MGLTessFactorInput in;

    in = make_in(GL_QUADS, GL_FRACTIONAL_ODD, GL_CCW, 1, 1, 1, 1, 1, 3, 3);
    expect(mglTessDomainVertexCount(&in) == 8u, "FO quad: 4 boundary + 4 interior");

    in = make_in(GL_QUADS, GL_FRACTIONAL_EVEN, GL_CCW, 1, 1, 1, 1, 1, 3, 3);
    expect(mglTessDomainVertexCount(&in) == 17u, "FE quad: 8 boundary + 9 interior");

    in = make_in(GL_TRIANGLES, GL_FRACTIONAL_ODD, GL_CCW, 1, 1, 1, 1, 1, 3, 1);
    expect(mglTessDomainVertexCount(&in) == 6u, "FO tri: 3 boundary + 3 interior");
}

static void test_equal_quad_grid(void)
{
    /* Equal spacing, all levels 3: 9 cells cover the unit square.
     * Fractional spacing does not imply equal spacing after rounding. */
    MGLTessFactorInput in =
        make_in(GL_QUADS, GL_EQUAL, GL_CCW, 0, 3, 3, 3, 3, 3, 3);
    MGLTessCoord coords[64];
    uint32_t n;
    uint32_t i;
    int saw_u0 = 0, saw_u1 = 0, saw_v_third = 0;

    expect(mglTessDomainVertexCount(&in) == 54u,
           "equal quad grid has 54 triangle vertices");
    n = mglTessGenerateDomain(&in, coords, 64);
    expect(n == 54u, "generate fills 54");

    for (i = 0; i < n; i++) {
        if (feq(coords[i].u, 0.f))
            saw_u0 = 1;
        if (feq(coords[i].u, 1.f))
            saw_u1 = 1;
        if (feq(coords[i].v, 1.f / 3.f))
            saw_v_third = 1;
    }
    expect(saw_u0 && saw_u1 && saw_v_third,
           "equal level-3 grid boundary and interior coordinates");

    for (int winding = 0; winding < 2; winding++) {
        float area = 0.f;
        in.winding = winding ? GL_CW : GL_CCW;
        n = mglTessGenerateDomain(&in, coords, 64);
        for (i = 0; i < n; i += 3) {
            const MGLTessCoord a = coords[i], b = coords[i + 1], c = coords[i + 2];
            const float det = (b.u - a.u) * (c.v - a.v) -
                              (b.v - a.v) * (c.u - a.u);
            expect(winding ? det < 0.f : det > 0.f, "quad winding");
            area += fabsf(det) * 0.5f;
        }
        expect(feq(area, 1.f), "quad triangles cover unit area");
    }

    /* Level-1 all-one: 6 verts, corners only. */
    in = make_in(GL_QUADS, GL_EQUAL, GL_CCW, 0, 1, 1, 1, 1, 1, 1);
    expect(mglTessDomainVertexCount(&in) == 6u, "level-1 quad → 6");
    {
        MGLTessCoord c;
        expect(mglTessDomainCoordAt(&in, 0, &c) == 0 && feq(c.u, 0.f) &&
                   feq(c.v, 0.f),
               "level-1 first vert (0,0)");
    }
}

static void test_factor_boundaries(void)
{
    static const uint32_t modes[] = {GL_TRIANGLES, GL_QUADS, GL_ISOLINES};
    static const uint32_t spacings[] = {GL_EQUAL, GL_FRACTIONAL_ODD, GL_FRACTIONAL_EVEN};
    static const uint32_t expected[] = {64u, 63u, 64u};
    for (unsigned m = 0; m < 3; m++) {
        for (unsigned s = 0; s < 3; s++) {
            MGLTessNormalizedFactors n;
            MGLTessFactorInput in = make_in(modes[m], spacings[s], GL_CCW, 0,
                INFINITY, FLT_MAX, 1000.f, 1000.f, INFINITY, FLT_MAX);
            mglTessNormalizeFactors(&in, &n);
            expect(!n.discard, "positive large levels do not discard");
            expect(n.outer_eff[0] == (modes[m] == GL_ISOLINES ? 64u : expected[s]),
                   "outer level clamped before integer conversion");
            expect(n.outer_eff[1] == expected[s], "spacing maximum");
            in.outer[0] = NAN;
            mglTessNormalizeFactors(&in, &n);
            expect(n.discard, "relevant outer NaN discards");
            in.outer[0] = -INFINITY;
            mglTessNormalizeFactors(&in, &n);
            expect(n.discard, "negative outer infinity discards");
        }
    }
    MGLTessNormalizedFactors n;
    MGLTessFactorInput in = make_in(GL_ISOLINES, GL_FRACTIONAL_ODD, GL_CCW, 0,
        2.5f, 2.5f, NAN, -INFINITY, NAN, FLT_MAX);
    mglTessNormalizeFactors(&in, &n);
    expect(!n.discard && n.outer_eff[0] == 3 && n.outer_eff[1] == 3,
           "isolines ignore unused factors");
    in = make_in(GL_QUADS, GL_EQUAL, GL_CCW, 0, 1, 1, 1, 1, NAN, -INFINITY);
    mglTessNormalizeFactors(&in, &n);
    expect(!n.discard && n.inner_eff[0] == 1 && n.inner_eff[1] == 1,
           "inner NaN lower-bound QoI and negative infinity clamp");
}

static void test_isolines(void)
{
    MGLTessFactorInput in =
        make_in(GL_ISOLINES, GL_EQUAL, GL_CCW, 0, 1, 2, 1, 1, 1, 1);
    expect(mglTessDomainVertexCount(&in) == 4u, "isolines n=1 m=2 → 4");
    in.point_mode = 1;
    expect(mglTessDomainVertexCount(&in) == 3u, "isolines point m=2 → 3");

    /* outer[1] under FO: ceil 2 → round to 3 (§11.2.2.2). */
    in = make_in(GL_ISOLINES, GL_FRACTIONAL_ODD, GL_CCW, 0, 1, 2, 1, 1, 1, 1);
    expect(mglTessDomainVertexCount(&in) == 6u, "isolines FO m=2→3 → 6");
}

static void test_fractional_edge_goldens(void)
{
    const uint32_t spacings[] = {GL_EQUAL, GL_FRACTIONAL_ODD, GL_FRACTIONAL_EVEN};
    const float expected[3][5] = {
        {0, 1.f / 3.f, 2.f / 3.f, 1, 0},
        {0, 0.3f, 0.7f, 1, 0},
        {0, 0.1f, 0.5f, 0.9f, 1},
    };
    const uint32_t counts[] = {4, 4, 5};
    for (unsigned s = 0; s < 3; s++) {
        MGLTessFactorInput in = make_in(GL_ISOLINES, spacings[s], GL_CCW, 1,
            1, 2.5f, 0, 0, 0, 0);
        MGLTessCoord points[5];
        expect(mglTessGenerateDomain(&in, points, 5) == counts[s], "fractional edge point count");
        for (unsigned i = 0; i < counts[s]; i++)
            expect(feq(points[i].u, expected[s][i]) && feq(points[i].v, 0),
                   "fractional short segments use unrounded factor");
    }
}

static void test_triangle_ring_goldens(void)
{
    const uint32_t points[] = {3, 7, 12, 19, 27};
    const uint32_t triangles[] = {1, 6, 13, 24, 37};
    for (unsigned level = 1; level <= 5; level++) {
        MGLTessFactorInput in = make_in(GL_TRIANGLES, GL_EQUAL, GL_CCW, 1,
            level, level, level, 1, level, 1);
        expect(mglTessDomainVertexCount(&in) == points[level - 1], "triangular ring vertex count");
        in.point_mode = 0;
        expect(mglTessDomainVertexCount(&in) == 3u * triangles[level - 1], "triangular ring primitive count");
    }
    MGLTessFactorInput in = make_in(GL_TRIANGLES, GL_EQUAL, GL_CCW, 1,
        4, 4, 4, 1, 4, 1);
    MGLTessCoord coords[19];
    expect(mglTessGenerateDomain(&in, coords, 19) == 19, "tri level 4 generation");
    int saw_corner = 0, saw_midpoint = 0, saw_center = 0;
    for (unsigned i = 0; i < 19; i++) {
        saw_corner |= feq(coords[i].u, 1.f/6) && feq(coords[i].v, 1.f/6);
        saw_midpoint |= feq(coords[i].u, 5.f/12) && feq(coords[i].v, 1.f/6);
        saw_center |= feq(coords[i].u, 1.f/3) && feq(coords[i].v, 1.f/3);
    }
    expect(saw_corner && saw_midpoint && saw_center, "equilateral perpendicular projection golden");
}

static void test_domain_invariants(void)
{
    const uint32_t modes[] = {GL_QUADS, GL_TRIANGLES};
    const uint32_t spacings[] = {GL_EQUAL, GL_FRACTIONAL_ODD, GL_FRACTIONAL_EVEN};
    const float levels[] = {1.f, 2.f, 2.5f, 4.f, 5.5f, 64.f};
    for (unsigned m = 0; m < 2; m++)
    for (unsigned s = 0; s < 3; s++)
    for (unsigned l = 0; l < 6; l++)
    for (unsigned cw = 0; cw < 2; cw++) {
        MGLTessFactorInput in = make_in(modes[m], spacings[s], cw ? GL_CW : GL_CCW, 0,
            2, 3, 5, 7, levels[l], levels[l]);
        const uint32_t count = mglTessDomainVertexCount(&in);
        MGLTessCoord *coords = (MGLTessCoord *)malloc(count * sizeof(*coords));
        expect(coords != NULL, "allocate domain golden buffer");
        if (!coords) return;
        expect(count % 3u == 0 && mglTessGenerateDomain(&in, coords, count) == count,
               "whole triangle stream");
        double area = 0;
        for (uint32_t i = 0; i < count; i++) {
            expect(coords[i].u >= 0 && coords[i].u <= 1 && coords[i].v >= 0 && coords[i].v <= 1,
                   "coordinates inside unit domain");
            if (modes[m] == GL_TRIANGLES)
                expect(coords[i].w >= -1e-7f && feq(coords[i].u + coords[i].v + coords[i].w, 1),
                       "nonnegative barycentric coordinates sum to one");
        }
        for (uint32_t i = 0; i < count; i += 3) {
            const MGLTessCoord a = coords[i], b = coords[i + 1], c = coords[i + 2];
            const double det = ((double)b.u-a.u)*((double)c.v-a.v) -
                               ((double)b.v-a.v)*((double)c.u-a.u);
            expect(cw ? det <= 1e-8 : det >= -1e-8, "domain winding is uniform");
            area += fabs(det) * 0.5;
        }
        expect(fabs(area - (modes[m] == GL_QUADS ? 1.0 : 0.5)) < 1e-5,
               "triangles cover domain area exactly once");
        free(coords);
    }
}

int main(void)
{
    test_round_spacing();
    test_inner_eps();
    test_discard();
    test_point_mode_counts();
    test_equal_quad_grid();
    test_factor_boundaries();
    test_isolines();
    test_fractional_edge_goldens();
    test_triangle_ring_goldens();
    test_domain_invariants();
    if (g_fails) {
        fprintf(stderr, "test_tess_domain: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("test_tess_domain: ok\n");
    return 0;
}
