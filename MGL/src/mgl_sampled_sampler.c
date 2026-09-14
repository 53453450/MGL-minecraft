/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_sampled_sampler.c — mglSampledSamplerMaterialize (P0-1, log 151).
 *
 * Mechanical translation of -materializeSampledSamplerForTexture:…: `self`
 * becomes the renderer handle, `ctx` comes from the state areas (the .m read
 * the `ctx` ivar), and the two `(__bridge id)` reads plus the
 * `CFBridgingRetain` stay borrowed/retained exactly as ARC did — the return
 * value is a borrowed handle, like the method's +0 `id`.
 */

#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <string.h>

#include "mgl_sampled_sampler.h"
#include "mgl_binding_texture.h" /* sampler materialize plan + logging */
#include "mgl_texture_sampler.h" /* mglTextureCreateSamplerForTexParam */
#include "mgl_renderer_ports.h"  /* state areas */
#include "mgl_render.h"          /* mglRenderGetTextureInfo */
#include "mgl_metal_ref.h"       /* mglSafeReleaseMetalObj */
#include "mgl_types_state.h"     /* Sampler */
#include "mgl_sampled_fallback.h" /* mglSampledFallbackTextureForExpectedType */
#include "mgl_texture_compat.h"  /* pixel-format/kind compatibility */
#include "mgl_trace_strategy.h" /* mglWriteProgramMSLDump */

/* MGL_STATE() from MGLRenderer_Private.h, in C (same twin as mgl_tess_dispatch.c). */
static GLMState *mglSsState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

static uint64_t mglSsTextureWidth(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info.width;
}

static uint64_t mglSsTextureHeight(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info.height;
}

static uint64_t mglSsTextureMipmapLevelCount(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info.mipmap_level_count;
}

void *mglSampledSamplerMaterialize(void *renderer, Texture *ptr,
                                   GLuint texture_unit, void *default_sampler,
                                   int force_default, GLuint sampler_target,
                                   GLuint program_name, GLuint spirv_binding,
                                   const char *stage, void *texture)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;

    Sampler *gl_sampler = (texture_unit < TEXTURE_UNITS)
                              ? mglSsState(&areas)->texture_samplers[texture_unit]
                              : NULL;
    MGLSamplerMaterializeInput in = {0};
    mglBindingTextureFillSamplerMaterializeInput(
        &in, force_default ? 1 : 0, texture_unit < TEXTURE_UNITS ? 1 : 0,
        gl_sampler ? 1 : 0, gl_sampler && gl_sampler->dirty_bits ? 1 : 0,
        gl_sampler && gl_sampler->mtl_data ? 1 : 0,
        ptr && ptr->params.mtl_data ? 1 : 0,
        (stage && stage[0] == 'v') ? 1 : 0);
    MGLSamplerMaterializePlan plan = {0};
    if (mglBindingTexturePlanSamplerMaterialize(&in, &plan) != 0) {
        return default_sampler;
    }
    if (plan.action == MGL_SM_ACTION_USE_DEFAULT) {
        return default_sampler;
    }
    void *sampler = default_sampler;
    const TextureParameter *params = NULL;
    GLuint sampler_name = 0u;
    if (plan.action == MGL_SM_ACTION_USE_GL_SAMPLER && gl_sampler) {
        if (plan.recreate_gl_sampler_mtl) {
            if (gl_sampler->mtl_data) {
                mglSafeReleaseMetalObj((void **)&gl_sampler->mtl_data);
            }
            GLuint target = sampler_target
                                ? sampler_target
                                : (ptr ? ptr->target : GL_TEXTURE_2D);
            /* CFBridgingRetain(X) == (__bridge_retained CFTypeRef)(X). */
            gl_sampler->mtl_data = (void *)CFRetain((CFTypeRef)
                mglTextureCreateSamplerForTexParam(&gl_sampler->params, target));
        }
        if (plan.clear_gl_sampler_dirty) {
            gl_sampler->dirty_bits = 0;
        }
        sampler = gl_sampler->mtl_data;
        params = &gl_sampler->params;
        sampler_name = gl_sampler->name;
    } else if (plan.action == MGL_SM_ACTION_USE_TEX_PARAMS && ptr) {
        sampler = ptr->params.mtl_data;
        params = &ptr->params;
    } else {
        return default_sampler;
    }
    if (params && mglTraceLogIsEnabled()) {
        mglBindingLogSamplerResolve(
            mglBindingTextureSamplerStageTag(stage), program_name, spirv_binding,
            texture_unit, plan.source_tag ? plan.source_tag : "?", sampler_name,
            params->min_filter, params->mag_filter, params->wrap_s, params->wrap_t,
            params->min_lod, params->max_lod, ptr ? ptr->name : 0u,
            ptr ? ptr->params.base_level : 0u, ptr ? ptr->params.max_level : 0u,
            ptr ? ptr->width : 0u, ptr ? ptr->height : 0u,
            texture ? mglSsTextureWidth(texture) : 0u,
            texture ? mglSsTextureHeight(texture) : 0u,
            texture ? mglSsTextureMipmapLevelCount(texture) : 0u);
    }
    return sampler;
}

static uint32_t mglSsTextureType(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info.texture_type;
}

static uint32_t mglSsTexturePixelFormat(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info.pixel_format;
}

void *mglSampledCompatFallbackPlan(void *renderer, Texture *ptr, void *texture,
                                   uint32_t expected_type, uint32_t expected_kind,
                                   const char *stage, GLuint program_name,
                                   GLuint spirv_binding, void *sample_program,
                                   int *used_fallback_out)
{
    MGLSampledTextureBindInput cin = {0};
    mglBindingTextureFillSampledCompatInput(
        &cin, texture ? 1 : 0, texture ? mglSsTextureType(texture) : 0u,
        expected_type,
        !texture || mglTexturePixelFormatCompatibleWithExpectedDataKind(
                        mglSsTexturePixelFormat(texture), expected_kind)
            ? 1
            : 0);
    MGLSampledTextureBindPlan cplan = {0};
    if (mglBindingTexturePlanSampled(&cin, &cplan) != 0 ||
        (cplan.action != MGL_ST_ACTION_TYPE_FALLBACK &&
         cplan.action != MGL_ST_ACTION_KIND_FALLBACK)) {
        return texture;
    }
    static uint64_t s_compat_mismatch_log_count = 0;
    if (mglBindingTextureRateLogHit(&s_compat_mismatch_log_count, 32ull, 512ull)) {
        mglBindingLogTexCompatMismatch(
            cplan.action == MGL_ST_ACTION_TYPE_FALLBACK ? "TYPE" : "DATA", stage,
            spirv_binding, program_name, ptr ? ptr->name : 0u, cin.mtl_type,
            expected_type, s_compat_mismatch_log_count);
    }
    if (sample_program) {
        char dump_reason[128];
        snprintf(dump_reason, sizeof(dump_reason),
                 "tex-%s-mismatch-%s-binding-%u",
                 cplan.action == MGL_ST_ACTION_TYPE_FALLBACK ? "type" : "data",
                 stage ? stage : "x", spirv_binding);
        mglWriteProgramMSLDump(sample_program, dump_reason);
    }
    texture = mglSampledFallbackTextureForExpectedType(renderer, expected_type,
                                                       expected_kind);
    if (used_fallback_out) {
        *used_fallback_out = 1;
    }
    return texture;
}
