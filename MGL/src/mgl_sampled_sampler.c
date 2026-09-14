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
#include "mgl_coordinate.h"      /* MGLYFlipDecision / mglDecideYFlipForSampledRT */
#include "mgl_rt_sync.h"         /* mglTextureCanUseGLSampledRenderTargetCopy */
#include "mgl_trace_strategy.h"  /* mglTraceRTYFlipDiagnosticsEnabled */
#include "mgl_trace_log.h"       /* mglTraceLogIsEnabled */
#include "mgl_blit_drivers.h"    /* mglBlitFreshGLSampledRenderTargetCopyForSampling */
#include "mgl_types_program.h"   /* Program */
#include "mgl_renderer_backend.h" /* mglRendererGetProgramBinding* */
#include "mgl_shader_resource.h" /* mglMetalCombinedSamplerSlotForElement */
#include "mgl_binding_policy.h"  /* mglRenderTextureBindingStageForShader */
#include "mgl_texture_bind.h"    /* mglRendererBindMTLTexture */
#include "mgl_texture_binding_resolve.h" /* mglTextureForSampledResourceForStage */
#include "mgl_trace_strategy.h" /* mglWriteProgramMSLDump */

/* The .m's file-local sampler slot ceiling (MGLRenderer+BindingState.m). */
enum { kMaxFragmentSamplerSlots = 16 };

/* MGLRenderer_Private.h declares this BOOL (signed char on macOS). */
extern signed char mglEnvFlagEnabled(const char *name);

/* Twins of the MGLRenderer+Draw_Private.h statics (rule 7). */
static int mglSsTraceRTYFlipDiagnosticsEnabled(void)
{
    return mglTraceLogIsEnabled() && mglEnvFlagEnabled("MGL_TRACE_RT_YFLIP");
}

static const char *mglSsYFlipDecisionName(MGLYFlipDecision decision)
{
    switch (decision) {
        case MGL_YFLIP_USE_ORIGINAL:
            return "original";
        case MGL_YFLIP_USE_SAMPLED_COPY:
            return "sampled-copy";
        case MGL_YFLIP_USE_ORIGINAL_AND_INJECT:
            return "original-inject";
        default:
            return "unknown";
    }
}

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

/* Twin of the MGLRenderer+Blit_Private.h static inline (see mgl_blit_drivers.c
 * for the blit-side copy of this helper). */
static int mglSsGLSampledCopyContentFresh(const Texture *tex)
{
    return tex != NULL && tex->mtl_gl_sampled_data != NULL &&
           tex->mtl_gl_sampled_write_version ==
               tex->mtl_render_target_write_version &&
           tex->mtl_gl_sampled_dirty_mip_mask == 0u;
}

bool mglSampledRenderTargetCopyPlan(
    void *renderer, Texture *ptr, void **texture_ptr, Program *sample_program,
    uint32_t expected_type, uint32_t expected_kind, int used_type_fallback,
    const char *stage, GLuint program_name, GLuint spirv_binding,
    GLuint texture_unit, const char *sampled_name, int *used_sampled_copy_out,
    void **direct_texture_for_trace, void **sampled_copy_for_trace)
{
    if (!texture_ptr || used_type_fallback || !ptr || !ptr->is_render_target) {
        return true;
    }
    void *texture = *texture_ptr;
    MGLYFlipDecision yflip = mglDecideYFlipForSampledRT(ptr, sample_program);
    if (mglSsTraceRTYFlipDiagnosticsEnabled()) {
        mglBindingLogRTYFlipDecision(
            stage, program_name, sampled_name, spirv_binding, texture_unit,
            ptr->name, mglTraceTextureLabel(ptr), mglSsYFlipDecisionName(yflip),
            (int)yflip, ptr->mtl_render_yflip_authority,
            ptr->mtl_render_target_write_version,
            ptr->mtl_gl_sampled_write_version, ptr->mtl_gl_sampled_data ? 1 : 0,
            mglProgramHasExistingFramebufferSampleYFlip(sample_program) ? 1 : 0);
    }

    void *sampled_copy = ptr->mtl_gl_sampled_data;
    MGLSampledTextureBindInput in = {0};
    mglBindingTextureFillSampledRTInput(
        &in, used_type_fallback ? 1 : 0, 1, (int)yflip,
        ptr->mtl_gl_sampled_data ? 1 : 0, mglSsGLSampledCopyContentFresh(ptr) ? 1 : 0,
        mglTextureCanUseGLSampledRenderTargetCopy(ptr) ? 1 : 0,
        (stage && stage[0] == 'f') ? 1 : 0,
        sampled_copy && (expected_type == 0 ||
                         mglSsTextureType(sampled_copy) == expected_type)
            ? 1
            : 0,
        sampled_copy &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglSsTexturePixelFormat(sampled_copy), expected_kind)
            ? 1
            : 0);

    MGLSampledTextureBindPlan plan = {0};
    if (mglBindingTexturePlanSampled(&in, &plan) != 0) {
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_USE_COPY && sampled_copy) {
        if (direct_texture_for_trace) {
            *direct_texture_for_trace = texture;
        }
        if (sampled_copy_for_trace) {
            *sampled_copy_for_trace = sampled_copy;
        }
        if (mglTraceLogIsEnabled()) {
            MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_BIND, .stage = stage,
                            .program = program_name, .name = sampled_name,
                            .binding = spirv_binding, .unit = texture_unit,
                            .tex = ptr->name, .label = mglTraceTextureLabel(ptr),
                            .original = (const void *)texture,
                            .copy = (const void *)sampled_copy);
        }
        void *chosen = sampled_copy;
        if (plan.apply_base_level_view) {
            chosen = mglSampledTextureViewForBaseLevel(ptr, sampled_copy);
        }
        *texture_ptr = chosen;
        if (used_sampled_copy_out) {
            *used_sampled_copy_out = 1;
        }
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_REPAIR) {
        void *repaired_copy = mglBlitFreshGLSampledRenderTargetCopyForSampling(
            renderer, ptr, texture, stage, program_name, spirv_binding,
            texture_unit, expected_type, expected_kind);
        if (!repaired_copy) {
            return true;
        }
        in.repaired_available = 1;
        in.repaired_fresh = mglSsGLSampledCopyContentFresh(ptr) ? 1 : 0;
        if (mglBindingTexturePlanSampled(&in, &plan) != 0) {
            return true;
        }
        if (plan.action == MGL_ST_ACTION_RT_RETRY) {
            return false;
        }
        if (plan.action == MGL_ST_ACTION_RT_USE_COPY) {
            void *chosen = repaired_copy;
            if (plan.apply_base_level_view) {
                chosen = mglSampledTextureViewForBaseLevel(ptr, repaired_copy);
            }
            *texture_ptr = chosen;
            if (used_sampled_copy_out) {
                *used_sampled_copy_out = 1;
            }
        }
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_GATE_MISS && mglTraceLogIsEnabled()) {
        MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_GATE_MISS, .stage = stage,
                        .program = program_name, .name = sampled_name,
                        .binding = spirv_binding, .unit = texture_unit,
                        .tex = ptr->name, .label = mglTraceTextureLabel(ptr),
                        .is_rt = 1,
                        .has_copy = ptr->mtl_gl_sampled_data ? 1 : 0,
                        .can_use = in.can_use_rt_copy,
                        .expected_type = expected_type);
    } else if (plan.action == MGL_ST_ACTION_RT_ORIGINAL) {
        static uint64_t s_rt_sample_copy_skip_existing_flip_log_count = 0;
        if (mglTraceLogIsEnabled() &&
            mglBindingTextureRateLogHit(
                &s_rt_sample_copy_skip_existing_flip_log_count, 32ull, 512ull)) {
            MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_SKIP_YFLIP,
                            .hit = s_rt_sample_copy_skip_existing_flip_log_count,
                            .stage = stage, .program = program_name,
                            .name = sampled_name, .binding = spirv_binding,
                            .tex = ptr ? ptr->name : 0u,
                            .decision_name = mglSsYFlipDecisionName(yflip),
                            .decision = (int)yflip);
        }
        if (plan.apply_base_level_view && texture) {
            *texture_ptr = mglSampledTextureViewForBaseLevel(ptr, texture);
        }
    }
    return true;
}

/* Twins of the MGLRenderer+Draw_Private.h resource-binding statics (rule 7). */
static int mglSsCollectResourceBinding(MGLRenderResourceBindingSnapshot *snapshot,
                                       uint32_t stage, uint32_t kind,
                                       void *resource, uint32_t index)
{
    if (!snapshot || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        kind > MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return 0;
    }
    uint32_t *count = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                          ? &snapshot->vertex_op_count
                          : &snapshot->fragment_op_count;
    MGLRenderResourceBindingOp *ops = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                                          ? snapshot->vertex_ops
                                          : snapshot->fragment_ops;
    if (*count >= MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        return 0;
    }
    ops[(*count)++] = (MGLRenderResourceBindingOp){
        .kind = kind,
        .index = index,
        .resource = resource,
    };
    return 1;
}

static int mglSsQueueResourceBinding(int collect, void *binding_state_owner,
                                     void *render_encoder_owner,
                                     MGLRenderResourceBindingSnapshot *snapshot,
                                     uint32_t stage, uint32_t kind,
                                     void *resource, uint32_t index)
{
    if (collect) {
        return mglSsCollectResourceBinding(snapshot, stage, kind, resource,
                                           index);
    }
    if (kind == MGL_RENDER_RESOURCE_BINDING_TEXTURE) {
        return mglRenderBindingSetTextureForOwner(binding_state_owner,
                                                  render_encoder_owner,
                                                  resource, stage, index) >= 0;
    }
    if (kind == MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return mglRenderBindingSetSamplerForOwner(binding_state_owner,
                                                  render_encoder_owner,
                                                  resource, stage, index) >= 0;
    }
    return 0;
}

static int mglSsFlushResourceBindings(void *binding_state_owner,
                                      void *render_encoder_owner,
                                      MGLRenderResourceBindingSnapshot *snapshot)
{
    if (!snapshot || (snapshot->vertex_op_count == 0 &&
                      snapshot->fragment_op_count == 0)) {
        return 1;
    }
    if (mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
            binding_state_owner, render_encoder_owner, snapshot, NULL, 0) != 0) {
        return 0;
    }
    *snapshot = (MGLRenderResourceBindingSnapshot){0};
    return 1;
}

bool mglSampledBindSeparateSamplersAndArrayTextures(
    void *renderer, Program *vertex_program, Program *fragment_program,
    GLuint fragment_program_name, GLuint vertex_program_name,
    void *default_sampler, uint64_t bind_call, int trace_bind,
    GLuint *separate_sampler_count, GLuint *bound_separate_samplers)
{
    (void)vertex_program_name;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    void *binding_state_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    void *render_encoder_owner =
        areas.command ? areas.command->currentRenderEncoderOwner : NULL;

    const int use_resource_snapshot = 1;
    MGLRenderResourceBindingSnapshot resource_snapshot = {0};

    *separate_sampler_count = mglRendererGetProgramBindingCount(
        ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES);
    *bound_separate_samplers = 0;
    for (GLuint i = 0; i < *separate_sampler_count; i++) {
        GLuint spirv_binding = mglRendererGetProgramBinding(
            ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES, (int)i);
        GLuint gl_binding = mglRendererGetProgramGLBinding(
            ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES, (int)i);
        if (!mglBindingTextureSeparateSamplerInRange(spirv_binding, gl_binding,
                                                     TEXTURE_UNITS)) {
            continue;
        }
        Program *sample_program = fragment_program;
        MGLShaderResource *sampler_resource = NULL;
        if (sample_program &&
            i < sample_program->shader_resources_list[_FRAGMENT_SHADER]
                    [_SEPARATE_SAMPLERS_RES]
                        .count) {
            sampler_resource =
                &sample_program->shader_resources_list[_FRAGMENT_SHADER]
                     [_SEPARATE_SAMPLERS_RES]
                         .list[i];
        }
        GLuint texture_unit = mglTextureUnitForSampledResource(
            sampler_resource,
            mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER),
            spirv_binding, _FRAGMENT_SHADER);

        void *sampler = mglSampledSamplerMaterialize(
            renderer, NULL, texture_unit, default_sampler, 0,
            (GLuint)mglRenderSamplerObjectTarget(), fragment_program_name,
            spirv_binding, "fragment", NULL);
        if (sampler && spirv_binding < kMaxFragmentSamplerSlots) {
            if (!mglSsQueueResourceBinding(
                    use_resource_snapshot, binding_state_owner,
                    render_encoder_owner, &resource_snapshot,
                    MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER, sampler,
                    spirv_binding)) {
                return false;
            }
            (*bound_separate_samplers)++;
        }

        if (trace_bind && i < 6) {
            mglTraceLog("texbind.separateSampler call=%llu idx=%u binding=%u "
                        "unit=%u sampler=%p",
                        (unsigned long long)bind_call, (unsigned)i,
                        (unsigned)spirv_binding, (unsigned)texture_unit,
                        sampler);
        }
    }

    Program *array_programs[] = {vertex_program, fragment_program};
    int array_stages[] = {areas.tess_native_tes_active ? _TESS_EVALUATION_SHADER
                                                       : _VERTEX_SHADER,
                          _FRAGMENT_SHADER};
    for (size_t program_index = 0; program_index < 2; program_index++) {
        Program *array_program = array_programs[program_index];
        int array_stage = array_stages[program_index];
        if (!array_program) {
            continue;
        }

        MGLShaderResourceList *array_resources =
            &array_program->shader_resources_list[array_stage][_SAMPLED_IMAGE_RES];
        for (GLuint resource_index = 0;
             array_resources->list && resource_index < array_resources->count;
             resource_index++) {
            MGLShaderResource *resource = &array_resources->list[resource_index];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            uint32_t expected_type = (uint32_t)mglRendererGetProgramExpectedTextureType(
                ctx, array_stage, _SAMPLED_IMAGE_RES, (int)resource_index);
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metal_slot = resource->binding + (GLuint)element;
                GLuint sampler_slot = mglMetalCombinedSamplerSlotForElement(
                    resource, (GLuint)element);
                if (!mglBindingTextureArrayElementSlotOk(metal_slot,
                                                         TEXTURE_UNITS)) {
                    break;
                }

                GLuint texture_unit = mglTextureUnitForSampledResource(
                    NULL, mglResolveProgramForStageFromState(ctx, array_stage),
                    metal_slot, array_stage);
                Texture *array_texture = mglTextureForSampledResourceForStage(
                    ctx, NULL, metal_slot, array_stage, expected_type);
                void *metal_texture = NULL;
                void *metal_sampler = default_sampler;
                if (array_texture &&
                    mglRendererBindMTLTexture(renderer, array_texture)) {
                    metal_texture = array_texture->mtl_data;
                    metal_sampler = mglSampledSamplerMaterialize(
                        renderer, array_texture, texture_unit, default_sampler, 0,
                        array_texture->target, array_program->name, metal_slot,
                        "vertex", metal_texture);
                }
                if (!metal_texture) {
                    metal_texture = mglSampledFallbackTextureForExpectedType(
                        renderer, expected_type, MGLTextureDataKindFloat);
                }

                uint32_t bind_stage =
                    mglRenderTextureBindingStageForShader(array_stage);
                if (!mglSsQueueResourceBinding(
                        use_resource_snapshot, binding_state_owner,
                        render_encoder_owner, &resource_snapshot, bind_stage,
                        MGL_RENDER_RESOURCE_BINDING_TEXTURE, metal_texture,
                        metal_slot)) {
                    return false;
                }
                if (mglBindingTextureShouldBindCombinedSampler(
                        resource->has_combined_sampler ? 1 : 0,
                        metal_sampler ? 1 : 0, sampler_slot,
                        kMaxFragmentSamplerSlots)) {
                    if (!mglSsQueueResourceBinding(
                            use_resource_snapshot, binding_state_owner,
                            render_encoder_owner, &resource_snapshot,
                            bind_stage, MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                            metal_sampler, sampler_slot)) {
                        return false;
                    }
                }
            }
        }
    }
    if (use_resource_snapshot &&
        !mglSsFlushResourceBindings(binding_state_owner, render_encoder_owner,
                                    &resource_snapshot)) {
        return false;
    }
    return true;
}
