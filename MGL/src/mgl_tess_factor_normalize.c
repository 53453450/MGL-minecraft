/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Tessellation factor normalize: discard, ceil clamp, §11.2.2.1 1+ε,
 * §11.2.2 spacing round.  Derived from GL 4.6; covered by test_tess_domain.
 */

#include "mgl_tess_domain.h"

#include "glcorearb.h"

#include <math.h>

uint32_t mglTessRoundLevelForSpacing(uint32_t spacing, uint32_t ceil_level)
{
    /* GL_MAX_TESS_GEN_LEVEL is 64.  Fractional modes clamp before rounding
     * per GL 4.6 §11.2.2 / EXT_tessellation_shader. */
    const uint32_t max_level = 64u;
    if (spacing == GL_FRACTIONAL_EVEN) {
        if (ceil_level < 2u)
            ceil_level = 2u;
        if (ceil_level > max_level)
            ceil_level = max_level;
        {
            const uint32_t r =
                (ceil_level & 1u) ? ceil_level + 1u : ceil_level;
            return r > 2u ? r : 2u;
        }
    }
    if (spacing == GL_FRACTIONAL_ODD) {
        if (ceil_level < 1u)
            ceil_level = 1u;
        if (ceil_level > max_level - 1u)
            ceil_level = max_level - 1u;
        return (ceil_level & 1u) ? ceil_level : ceil_level + 1u;
    }
    if (ceil_level < 1u)
        ceil_level = 1u;
    if (ceil_level > max_level)
        ceil_level = max_level;
    return ceil_level;
}

static uint32_t tess_ceil_level1(float v)
{
    /* Inner NaN has no prescribed coordinate result; choose the lower bound
     * deterministically. Relevant outer NaNs have already discarded the patch.
     * Clamp before integer conversion, including positive infinity. */
    if (!(v >= 1.0f))
        return 1u;
    if (!isfinite(v) || v >= 64.0f)
        return 64u;
    return (uint32_t)ceilf(v);
}

static float tess_clamp_level(float v, uint32_t spacing)
{
    const float lower = spacing == GL_FRACTIONAL_EVEN ? 2.f : 1.f;
    const float upper = spacing == GL_FRACTIONAL_ODD ? 63.f : 64.f;
    if (!(v >= lower)) return lower;
    return v > upper ? upper : v;
}

static uint32_t tess_round_inner(uint32_t spacing, uint32_t ceil_inner,
                                 int all_levels_one)
{
    /* GL 4.6 §11.2.2.1: clamped inner == 1 and not the all-levels-1
     * degenerate patch → treat as 1+ε before spacing round
     * (equal → 2, fractional_odd → 3). */
    if (ceil_inner == 1u && !all_levels_one)
        ceil_inner = 2u;
    return mglTessRoundLevelForSpacing(spacing, ceil_inner);
}

static int tess_discard(uint32_t gen_mode, const float *edge)
{
    /* GL 4.6 §11.2.2: only outer levels ≤ 0 (or NaN) discard. Inner must not. */
    if (!edge)
        return 1;
    switch (gen_mode) {
    case GL_ISOLINES:
        return edge[0] <= 0.0f || edge[1] <= 0.0f || isnan(edge[0]) ||
               isnan(edge[1]);
    case GL_QUADS:
        return edge[0] <= 0.0f || edge[1] <= 0.0f || edge[2] <= 0.0f ||
               edge[3] <= 0.0f || isnan(edge[0]) || isnan(edge[1]) ||
               isnan(edge[2]) || isnan(edge[3]);
    default: /* GL_TRIANGLES */
        return edge[0] <= 0.0f || edge[1] <= 0.0f || edge[2] <= 0.0f ||
               isnan(edge[0]) || isnan(edge[1]) || isnan(edge[2]);
    }
}

void mglTessNormalizeFactors(const MGLTessFactorInput *in,
                             MGLTessNormalizedFactors *out)
{
    uint32_t i;
    if (!out)
        return;
    out->discard = 1;
    out->all_levels_one = 0;
    for (i = 0; i < 4u; i++) {
        out->outer_ceil[i] = 1u;
        out->outer_eff[i] = 1u;
        out->outer_clamped[i] = 1.f;
    }
    out->inner_ceil[0] = out->inner_ceil[1] = 1u;
    out->inner_eff[0] = out->inner_eff[1] = 1u;
    out->inner_clamped[0] = out->inner_clamped[1] = 1.f;
    if (!in)
        return;

    if (tess_discard(in->gen_mode, in->outer))
        return;
    out->discard = 0;

    for (i = 0; i < 4u; i++) {
        out->outer_ceil[i] = tess_ceil_level1(in->outer[i]);
        out->outer_clamped[i] = tess_clamp_level(in->outer[i],
            in->gen_mode == GL_ISOLINES && i == 0 ? GL_EQUAL : in->spacing);
    }
    out->inner_ceil[0] = tess_ceil_level1(in->inner[0]);
    out->inner_ceil[1] = tess_ceil_level1(in->inner[1]);
    for (i = 0; i < 2u; i++)
        out->inner_clamped[i] = tess_clamp_level(in->inner[i], in->spacing);

    if (in->gen_mode == GL_ISOLINES) {
        /* §11.2.2.3: outer[0] always equal_spacing; outer[1] uses TES spacing. */
        out->all_levels_one = 0;
        out->outer_eff[0] =
            mglTessRoundLevelForSpacing(GL_EQUAL, out->outer_ceil[0]);
        out->outer_eff[1] =
            mglTessRoundLevelForSpacing(in->spacing, out->outer_ceil[1]);
        out->outer_eff[2] = out->outer_eff[3] = 1u;
        out->inner_eff[0] = out->inner_eff[1] = 1u;
        return;
    }

    if (in->gen_mode == GL_QUADS) {
        out->all_levels_one =
            (out->inner_ceil[0] == 1u && out->inner_ceil[1] == 1u &&
             out->outer_ceil[0] == 1u && out->outer_ceil[1] == 1u &&
             out->outer_ceil[2] == 1u && out->outer_ceil[3] == 1u);
        for (i = 0; i < 4u; i++) {
            out->outer_eff[i] =
                mglTessRoundLevelForSpacing(in->spacing, out->outer_ceil[i]);
        }
        out->inner_eff[0] = tess_round_inner(in->spacing, out->inner_ceil[0],
                                             out->all_levels_one);
        out->inner_eff[1] = tess_round_inner(in->spacing, out->inner_ceil[1],
                                             out->all_levels_one);
        for (i = 0; i < 2u; i++) {
            if (!out->all_levels_one && out->inner_clamped[i] == 1.f)
                out->inner_clamped[i] = nextafterf(1.f, 2.f);
        }
        return;
    }

    /* GL_TRIANGLES */
    out->all_levels_one =
        (out->inner_ceil[0] == 1u && out->outer_ceil[0] == 1u &&
         out->outer_ceil[1] == 1u && out->outer_ceil[2] == 1u);
    for (i = 0; i < 3u; i++) {
        out->outer_eff[i] =
            mglTessRoundLevelForSpacing(in->spacing, out->outer_ceil[i]);
    }
    out->outer_eff[3] = 1u;
    out->inner_eff[0] =
        tess_round_inner(in->spacing, out->inner_ceil[0], out->all_levels_one);
    out->inner_eff[1] = 1u;
    if (!out->all_levels_one && out->inner_clamped[0] == 1.f)
        out->inner_clamped[0] = nextafterf(1.f, 2.f);
}
