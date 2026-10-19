/* SPDX-License-Identifier: LGPL-3.0-only */
/*
 * mgl_sampler_plan.h
 * MGL
 *
 * Sampler plan: GL sampler state resolved to MGL-owned values, so the Metal
 * encoder maps numbers to Metal enums without reading GL enums itself
 * (docs/AUDIT_LAYERING_2026-10-17.md §5.1: L3 must not branch on GL_*).
 *
 * Pure data in, pure data out; no Metal, no renderer state.
 */
#ifndef MGL_SAMPLER_PLAN_H
#define MGL_SAMPLER_PLAN_H

#include <stdint.h>

/* TextureParameter lives in mgl_types_texture.h, which assumes the GL types
 * and size_t are already in scope; forward-declare it here so this header
 * stays self-contained. */
typedef struct TextureParameter_t TextureParameter;

#ifdef __cplusplus
extern "C" {
#endif

/* Filters map 1:1 onto Metal's nearest/linear.  The mip choice is split out
 * because GL encodes it inside the minification filter. */
typedef enum MGLSamplerFilter {
    MGL_SAMPLER_FILTER_NEAREST = 0,
    MGL_SAMPLER_FILTER_LINEAR = 1
} MGLSamplerFilter;

typedef enum MGLSamplerMipFilter {
    MGL_SAMPLER_MIP_NONE = 0,
    MGL_SAMPLER_MIP_NEAREST = 1,
    MGL_SAMPLER_MIP_LINEAR = 2
} MGLSamplerMipFilter;

typedef enum MGLSamplerWrap {
    MGL_SAMPLER_WRAP_REPEAT = 0,
    MGL_SAMPLER_WRAP_MIRRORED_REPEAT = 1,
    MGL_SAMPLER_WRAP_CLAMP_TO_EDGE = 2,
    MGL_SAMPLER_WRAP_CLAMP_TO_BORDER = 3,
    MGL_SAMPLER_WRAP_MIRROR_CLAMP_TO_EDGE = 4
} MGLSamplerWrap;

/* Compare function values are ordered like MTL::CompareFunction (Never = 0 ..
 * Always = 7) shifted by one, so the encoder can cast (compare - 1).  Keep
 * this order if Metal's enum ever changes. */
typedef enum MGLSamplerCompare {
    MGL_SAMPLER_COMPARE_NONE = 0,
    MGL_SAMPLER_COMPARE_NEVER = 1,
    MGL_SAMPLER_COMPARE_LESS = 2,
    MGL_SAMPLER_COMPARE_EQUAL = 3,
    MGL_SAMPLER_COMPARE_LEQUAL = 4,
    MGL_SAMPLER_COMPARE_GREATER = 5,
    MGL_SAMPLER_COMPARE_NOTEQUAL = 6,
    MGL_SAMPLER_COMPARE_GEQUAL = 7,
    MGL_SAMPLER_COMPARE_ALWAYS = 8
} MGLSamplerCompare;

typedef struct MGLSamplerPlan {
    uint8_t min_filter;   /* MGLSamplerFilter */
    uint8_t mip_filter;   /* MGLSamplerMipFilter */
    uint8_t mag_filter;   /* MGLSamplerFilter */
    uint8_t wrap_s;       /* MGLSamplerWrap */
    uint8_t wrap_t;       /* MGLSamplerWrap */
    uint8_t wrap_r;       /* MGLSamplerWrap */
    uint8_t compare;      /* MGLSamplerCompare */
    uint8_t normalized_coordinates;   /* 0 for GL_TEXTURE_RECTANGLE */
    uint8_t any_wrap_border;          /* some wrap is CLAMP_TO_BORDER */
    uint8_t all_wrap_clamp_to_edge;   /* s/t/r all CLAMP_TO_EDGE */
    float border_color[4];
    float min_lod;
    float max_lod;
    float max_anisotropy;
} MGLSamplerPlan;

/* Resolve GL sampler parameters into the plan.  Returns 0 on success, -1 when
 * a filter, wrap mode, compare mode or compare function is not a valid GL
 * value for the sampler (the caller reports the error).
 *
 * `target` is only used to decide whether coordinates are normalized
 * (GL_TEXTURE_RECTANGLE is the one non-normalized sampler target). */
int mglSamplerPlanBuild(const TextureParameter *params, uint32_t target,
                        MGLSamplerPlan *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SAMPLER_PLAN_H */
