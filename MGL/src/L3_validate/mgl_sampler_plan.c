/* SPDX-License-Identifier: LGPL-3.0-only */
/*
 * mgl_sampler_plan.c
 * MGL
 *
 * GL sampler state -> MGLSamplerPlan (see mgl_sampler_plan.h).
 */

#include "glm_context.h"
#include "mgl_sampler_plan.h"

static int mglSamplerWrapFromGL(uint32_t value, uint8_t *out)
{
    switch (value) {
        case GL_REPEAT: *out = MGL_SAMPLER_WRAP_REPEAT; return 0;
        case GL_MIRRORED_REPEAT: *out = MGL_SAMPLER_WRAP_MIRRORED_REPEAT; return 0;
        case GL_CLAMP_TO_EDGE: *out = MGL_SAMPLER_WRAP_CLAMP_TO_EDGE; return 0;
        case GL_CLAMP_TO_BORDER: *out = MGL_SAMPLER_WRAP_CLAMP_TO_BORDER; return 0;
        case GL_MIRROR_CLAMP_TO_EDGE:
            *out = MGL_SAMPLER_WRAP_MIRROR_CLAMP_TO_EDGE;
            return 0;
        default:
            return -1;
    }
}

static int mglSamplerCompareFromGL(uint32_t value, uint8_t *out)
{
    switch (value) {
        case GL_NEVER: *out = MGL_SAMPLER_COMPARE_NEVER; return 0;
        case GL_LESS: *out = MGL_SAMPLER_COMPARE_LESS; return 0;
        case GL_EQUAL: *out = MGL_SAMPLER_COMPARE_EQUAL; return 0;
        case GL_LEQUAL: *out = MGL_SAMPLER_COMPARE_LEQUAL; return 0;
        case GL_GREATER: *out = MGL_SAMPLER_COMPARE_GREATER; return 0;
        case GL_NOTEQUAL: *out = MGL_SAMPLER_COMPARE_NOTEQUAL; return 0;
        case GL_GEQUAL: *out = MGL_SAMPLER_COMPARE_GEQUAL; return 0;
        case GL_ALWAYS: *out = MGL_SAMPLER_COMPARE_ALWAYS; return 0;
        default:
            return -1;
    }
}

int mglSamplerPlanBuild(const TextureParameter *params, uint32_t target,
                        MGLSamplerPlan *out)
{
    if (!params || !out) {
        return -1;
    }

    *out = (MGLSamplerPlan){0};
    out->mag_filter = MGL_SAMPLER_FILTER_NEAREST;
    out->mip_filter = MGL_SAMPLER_MIP_NONE;
    out->compare = MGL_SAMPLER_COMPARE_NONE;

    /* GL packs the mip filter into the minification filter. */
    switch (params->min_filter) {
        case GL_NEAREST:
            out->min_filter = MGL_SAMPLER_FILTER_NEAREST;
            break;
        case GL_LINEAR:
            out->min_filter = MGL_SAMPLER_FILTER_LINEAR;
            break;
        case GL_NEAREST_MIPMAP_NEAREST:
            out->min_filter = MGL_SAMPLER_FILTER_NEAREST;
            out->mip_filter = MGL_SAMPLER_MIP_NEAREST;
            break;
        case GL_LINEAR_MIPMAP_NEAREST:
            out->min_filter = MGL_SAMPLER_FILTER_LINEAR;
            out->mip_filter = MGL_SAMPLER_MIP_NEAREST;
            break;
        case GL_NEAREST_MIPMAP_LINEAR:
            out->min_filter = MGL_SAMPLER_FILTER_NEAREST;
            out->mip_filter = MGL_SAMPLER_MIP_LINEAR;
            break;
        case GL_LINEAR_MIPMAP_LINEAR:
            out->min_filter = MGL_SAMPLER_FILTER_LINEAR;
            out->mip_filter = MGL_SAMPLER_MIP_LINEAR;
            break;
        default:
            return -1;
    }

    switch (params->mag_filter) {
        case GL_NEAREST:
            out->mag_filter = MGL_SAMPLER_FILTER_NEAREST;
            break;
        case GL_LINEAR:
            out->mag_filter = MGL_SAMPLER_FILTER_LINEAR;
            break;
        default:
            return -1;
    }

    if (mglSamplerWrapFromGL(params->wrap_s, &out->wrap_s) < 0 ||
        mglSamplerWrapFromGL(params->wrap_t, &out->wrap_t) < 0 ||
        mglSamplerWrapFromGL(params->wrap_r, &out->wrap_r) < 0) {
        return -1;
    }

    if (params->compare_mode == GL_COMPARE_REF_TO_TEXTURE) {
        if (mglSamplerCompareFromGL(params->compare_func, &out->compare) < 0) {
            return -1;
        }
    } else if (params->compare_mode != GL_NONE) {
        return -1;
    }

    out->normalized_coordinates = target != GL_TEXTURE_RECTANGLE ? 1u : 0u;
    out->any_wrap_border =
        (out->wrap_s == MGL_SAMPLER_WRAP_CLAMP_TO_BORDER ||
         out->wrap_t == MGL_SAMPLER_WRAP_CLAMP_TO_BORDER ||
         out->wrap_r == MGL_SAMPLER_WRAP_CLAMP_TO_BORDER)
            ? 1u
            : 0u;
    out->all_wrap_clamp_to_edge =
        (out->wrap_s == MGL_SAMPLER_WRAP_CLAMP_TO_EDGE &&
         out->wrap_t == MGL_SAMPLER_WRAP_CLAMP_TO_EDGE &&
         out->wrap_r == MGL_SAMPLER_WRAP_CLAMP_TO_EDGE)
            ? 1u
            : 0u;

    for (int i = 0; i < 4; i++) {
        out->border_color[i] = params->border_color[i];
    }
    out->min_lod = params->min_lod;
    out->max_lod = params->max_lod;
    out->max_anisotropy = params->max_anisotropy;
    out->reduction_mode = params->reduction_mode == GL_MIN ? 1u
                          : params->reduction_mode == GL_MAX ? 2u : 0u;
    return 0;
}
