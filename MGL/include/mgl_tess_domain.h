/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Tessellation domain engine: factor normalize + domain generation.
 * Pure C, no Metal types.  GL enum values for mode/spacing/winding.
 *
 * Spec anchors: OpenGL 4.6 §11.2.2.1 (inner 1+ε), §11.2.2.2 (clamp/round,
 * patch discard), §11.2.2.3 (isolines outer[0] equal_spacing).
 */

#ifndef MGL_TESS_DOMAIN_H
#define MGL_TESS_DOMAIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLTessCoord {
    float u;
    float v;
    float w; /* triangles barycentric; 0 for quads/isolines */
} MGLTessCoord;

typedef struct MGLTessFactorInput {
    float outer[4];
    float inner[2];
    uint32_t gen_mode;   /* GL_TRIANGLES / GL_QUADS / GL_ISOLINES */
    uint32_t spacing;    /* GL_EQUAL / GL_FRACTIONAL_ODD / GL_FRACTIONAL_EVEN */
    uint32_t winding;    /* GL_CW / GL_CCW (domain winding) */
    uint32_t point_mode; /* 0 or 1 */
} MGLTessFactorInput;

typedef struct MGLTessNormalizedFactors {
    int discard;
    int all_levels_one;
    uint32_t outer_ceil[4];
    uint32_t inner_ceil[2];
    uint32_t outer_eff[4]; /* after spacing round (§11.2.2.2) */
    uint32_t inner_eff[2]; /* after 1+ε then spacing round (§11.2.2.1) */
    float outer_clamped[4]; /* Unrounded factors retain fractional spacing. */
    float inner_clamped[2];
} MGLTessNormalizedFactors;

/* §11.2.2.2: clamp then round subdivision count. */
uint32_t mglTessRoundLevelForSpacing(uint32_t spacing, uint32_t ceil_level);

/* Discard → ceil → 1+ε on inner → spacing round. */
void mglTessNormalizeFactors(const MGLTessFactorInput *in,
                             MGLTessNormalizedFactors *out);

/* Expanded primitive stream (unique points in point mode); 0 if discarded. */
uint32_t mglTessDomainVertexCount(const MGLTessFactorInput *in);

/* TessCoord for invocation index in [0, vertex_count). */
int mglTessDomainCoordAt(const MGLTessFactorInput *in, uint32_t index,
                         MGLTessCoord *out);

/* Fill complete coords[0..count); returns 0 on bad args/discard/short capacity. */
uint32_t mglTessGenerateDomain(const MGLTessFactorInput *in,
                               MGLTessCoord *coords, uint32_t cap);
uint32_t mglTessGenerateDomainStrided(const MGLTessFactorInput *in,
    void *coords, uint32_t cap, uint32_t stride);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TESS_DOMAIN_H */
