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
#include "mgl_safety.h"          /* mglObjectPointerLooksPlausible / range check */
#include "mgl_types_state.h"      /* MGLState / history depth */
#include "mgl_trace_strategy.h"  /* focused/trace-file binding log gates */
#include "mgl_texture_debug.h"   /* mglTraceTextureName */
#include "mgl_byte_hash.h"       /* mglTraceHashBytes */
#include "mgl_focus_program.h"   /* mglIsFocusedLoadingProgram */
#include "mgl_state_log.h"       /* mglMipDiagEnabled */
#include "pixel_utils.h"          /* mtlFormatForGLInternalFormat */
#include "mgl_encode_context.h"   /* MGLEncodeContext */
#include "mgl_storage_image_bind.h" /* mglBindingStateBindStorageImagesForVertexProgram */
#include "mgl_trace_strategy.h" /* mglWriteProgramMSLDump */

/* The .m's file-local invalid-pixel-format sentinel (MGLRenderer+BindingState.m). */
enum { MGL_BINDING_PIXEL_FORMAT_INVALID = 0u };

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
                    (areas.command ? areas.command->currentRenderEncoderOwner : NULL), &resource_snapshot,
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
                        (areas.command ? areas.command->currentRenderEncoderOwner : NULL), &resource_snapshot, bind_stage,
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
                            (areas.command ? areas.command->currentRenderEncoderOwner : NULL), &resource_snapshot,
                            bind_stage, MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                            metal_sampler, sampler_slot)) {
                        return false;
                    }
                }
            }
        }
    }
    if (use_resource_snapshot &&
        !mglSsFlushResourceBindings(binding_state_owner, (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
                                    &resource_snapshot)) {
        return false;
    }
    return true;
}

/* === depth-texture recovery for fragment sampling (P0-1, log 154) ======== */

/* The ObjC-private-header C symbols this path needs, restated for this TU.
 * BOOL is signed char on macOS, so BOOL returns/out-params are declared as
 * such (not int) to keep the ABI exact. */
extern signed char mglRendererGLSampledCopyLooksUsable(
    Texture *tex, uint32_t expected_type, MGLTextureDataKind expected_kind,
    signed char allow_previous_write_version, void **copy_out,
    signed char *used_previous_write_version_out);
extern signed char mglRendererTextureLooksLikeSampledColor2D(
    GLMContext glctx, Texture *tex);
extern signed char mglRendererTextureLooksRecoverableSampled2D(
    GLMContext glctx, Texture *tex, uint32_t expected_type,
    MGLTextureDataKind expected_kind);
extern Texture *mglFindFramebufferColorTexturePairedWithDepth(
    GLMContext glctx, Texture *depth_texture, GLuint *fbo_name_out);
extern signed char mglCurrentDrawFramebufferUsesColorTexture(
    GLMContext glctx, Texture *texture, GLuint expected_fbo_name,
    size_t *attachment_index_out);

/* Twin of the +BindingState.m static (NSUInteger* is size_t* on macOS). */
static signed char mglSsRenderPassUsesColorTexture(void *owner, void *texture,
                                                   size_t *attachment_index_out)
{
    uint32_t attachment_index = MAX_COLOR_ATTACHMENTS;
    const signed char found =
        (signed char)mglRenderPassUsesColorTextureOwner(owner, texture,
                                                        &attachment_index);
    if (attachment_index_out) {
        *attachment_index_out = attachment_index;
    }
    return found;
}

/* The .m's MGL_ABORT_TBIND_IF_ENCODER_CLOSED().  The owner is re-read at every
 * use, like the macro did (rule: a cached render-encoder owner goes stale). */
#define MGL_SS_ABORT_TBIND_IF_ENCODER_CLOSED()                                 \
    do {                                                                       \
        if (mglRenderEncoderOwnerHasCurrent(                                   \
                areas.command ? areas.command->currentRenderEncoderOwner       \
                              : NULL) == 0) {                                  \
            if (ctx) {                                                         \
                mglMarkRendererDirtyBits(ctx->active_state,                    \
                                         (DIRTY_TEX | DIRTY_TEX_BINDING |      \
                                          DIRTY_RENDER_STATE));                \
            }                                                                  \
            return false;                                                      \
        }                                                                      \
    } while (0)

bool mglSampledRecoverFragmentDepthTexture(
    void *renderer, Texture **ptr_ptr, void **texture_ptr,
    const char *sampled_name, GLuint spirv_binding, GLuint texture_unit,
    uint32_t expected_type, uint32_t expected_kind,
    GLuint fragment_program_name, int *suppress_missing_ptr,
    int *used_fallback_ptr)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;

    Texture *ptr = *ptr_ptr;
    void *texture = *texture_ptr;
    int suppress_missing = *suppress_missing_ptr;
    int used_fallback = *used_fallback_ptr;

    RETURN_FALSE_ON_FAILURE(mglRendererBindMTLTexture(renderer, ptr));
    MGL_SS_ABORT_TBIND_IF_ENCODER_CLOSED();
    if (ptr->mtl_data) {
        texture = ptr->mtl_data;
        /* Defer RT base-level views until Y-flip (avoids MRT view races). */
        if (!ptr->is_render_target) {
            texture = mglSampledTextureViewForBaseLevel(ptr, texture);
        }
    }

    TextureLevel *depth_sample_level0 = mglTraceTextureBaseLevel(ptr);
    MGLDepthRecoverInput gin = {0};
    mglBindingTextureFillDepthRecoverGateInput(
        &gin, texture ? 1 : 0,
        mglBindingTextureSampledNameIsInSampler(sampled_name),
        texture && mglMetalPixelFormatIsDepthOrStencil(
                       mglSsTexturePixelFormat(texture)),
        ptr && ptr->is_render_target ? 1 : 0,
        depth_sample_level0 && depth_sample_level0->ever_written,
        depth_sample_level0 && depth_sample_level0->has_initialized_data);
    MGLDepthRecoverPlan gplan = {0};
    if (mglBindingTexturePlanDepthRecover(&gin, &gplan) != 0 ||
        gplan.action == MGL_DR_ACTION_KEEP) {
        goto done;
    }

    if (gplan.action == MGL_DR_ACTION_ENTER_INSAMPLER) {
        GLuint paired_fbo_name = 0u;
        Texture *paired_color = mglFindFramebufferColorTexturePairedWithDepth(
            ctx, ptr, &paired_fbo_name);
        Texture *recover_texture = NULL;
        void *recover_mtl = NULL;
        const char *recover_reason = "none";
        int recovered_from_sampled_copy = 0;
        int recovered_from_previous_version = 0;
        size_t recover_att = MAX_COLOR_ATTACHMENTS;
        size_t cur_att = MAX_COLOR_ATTACHMENTS;
        int paired_cur = mglCurrentDrawFramebufferUsesColorTexture(
            ctx, paired_color, paired_fbo_name, &cur_att);
        void *paired_mtl = NULL;
        if (paired_color) {
            RETURN_FALSE_ON_FAILURE(
                mglRendererBindMTLTexture(renderer, paired_color));
            MGL_SS_ABORT_TBIND_IF_ENCODER_CLOSED();
            paired_mtl = paired_color->mtl_data;
            if (!paired_cur && paired_mtl) {
                paired_cur = mglSsRenderPassUsesColorTexture(
                    areas.command ? areas.command->renderPassStateOwner : NULL,
                    paired_mtl, &cur_att);
            }
        }
        MGLDepthRecoverInput iin = {0};
        mglBindingTextureFillDepthRecoverInSamplerInput(
            &iin, paired_color ? 1 : 0, paired_cur ? 1 : 0, paired_mtl ? 1 : 0,
            paired_mtl && mglMetalPixelFormatIsDepthOrStencil(
                              mglSsTexturePixelFormat(paired_mtl)),
            texture_unit < TEXTURE_UNITS ? 1 : 0);
        MGLDepthRecoverPlan iplan = {0};
        (void)mglBindingTexturePlanDepthRecover(&iin, &iplan);

        if (iplan.action == MGL_DR_ACTION_PROBE_PAIRED_COPY) {
            static uint64_t s_hist_sup = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_hist_sup)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_HIST_SUPPRESSED,
                                .hit = s_hist_sup,
                                .program = fragment_program_name,
                                .binding = spirv_binding, .unit = texture_unit,
                                .fbo = paired_fbo_name, .color_att = cur_att,
                                .depth_tex = ptr ? ptr->name : 0u,
                                .paired_color = paired_color ? paired_color->name
                                                             : 0u);
            }
            void *paired_copy = NULL;
            signed char used_prev = 0;
            int usable =
                paired_color &&
                mglRendererGLSampledCopyLooksUsable(
                    paired_color, expected_type,
                    (MGLTextureDataKind)expected_kind, 1, &paired_copy,
                    &used_prev);
            MGLDepthRecoverInput cin = {0};
            mglBindingTextureFillDepthRecoverCopyInput(&cin, usable ? 1 : 0);
            MGLDepthRecoverPlan cplan = {0};
            (void)mglBindingTexturePlanDepthRecover(&cin, &cplan);
            if (cplan.action == MGL_DR_ACTION_USE_RECOVER) {
                recover_texture = paired_color;
                recover_mtl = paired_copy;
                recover_reason =
                    cplan.reason_tag ? cplan.reason_tag : "paired-current-copy";
                recovered_from_sampled_copy = 1;
                recovered_from_previous_version = used_prev ? 1 : 0;
                recover_att = cur_att;
            } else {
                static uint64_t s_no_copy = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_no_copy)) {
                    MGL_EMIT_DR_LOG(
                        .kind = MGL_DR_LOG_NO_COPY, .hit = s_no_copy,
                        .program = fragment_program_name,
                        .binding = spirv_binding, .unit = texture_unit,
                        .fbo = paired_fbo_name, .color_att = cur_att,
                        .depth_tex = ptr ? ptr->name : 0u,
                        .color_tex = paired_color ? paired_color->name : 0u,
                        .depth_fmt = mglSsTexturePixelFormat(texture),
                        .sampled_ver = paired_color
                                           ? paired_color
                                                 ->mtl_gl_sampled_write_version
                                           : 0u,
                        .rt_ver = paired_color
                                      ? paired_color
                                            ->mtl_render_target_write_version
                                      : 0u);
                }
                texture = NULL;
                suppress_missing = 1;
            }
        } else if (iplan.action == MGL_DR_ACTION_USE_PAIRED_DIRECT) {
            static uint64_t s_rec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_rec)) {
                MGL_EMIT_DR_LOG(
                    .kind = MGL_DR_LOG_PAIRED_DIRECT, .hit = s_rec,
                    .program = fragment_program_name,
                    .binding = spirv_binding, .unit = texture_unit,
                    .fbo = paired_fbo_name, .depth_tex = ptr ? ptr->name : 0u,
                    .color_tex = paired_color->name,
                    .depth_fmt = mglSsTexturePixelFormat(texture),
                    .color_fmt = mglSsTexturePixelFormat(paired_mtl),
                    .w = mglSsTextureWidth(paired_mtl),
                    .h = mglSsTextureHeight(paired_mtl));
            }
            ptr = paired_color;
            texture = paired_mtl;
        } else if (iplan.action == MGL_DR_ACTION_SCAN_HISTORY) {
            for (GLuint hi = 0; hi < MGL_RECENT_SAMPLED_2D_HISTORY; hi++) {
                Texture *cand =
                    mglSsState(&areas)->recent_sampled_2d_textures[texture_unit]
                                                              [hi];
                if (!cand || cand == ptr || cand == paired_color ||
                    !mglRendererTextureLooksLikeSampledColor2D(ctx, cand)) {
                    continue;
                }
                void *cand_mtl = cand->mtl_data;
                size_t cand_att = MAX_COLOR_ATTACHMENTS;
                int cand_cur = mglCurrentDrawFramebufferUsesColorTexture(
                                   ctx, cand, 0u, &cand_att) ||
                               mglSsRenderPassUsesColorTexture(
                                   areas.command
                                       ? areas.command->renderPassStateOwner
                                       : NULL,
                                   cand_mtl, &cand_att);
                if (!cand_cur && (!cand->mtl_data || cand->dirty_bits)) {
                    RETURN_FALSE_ON_FAILURE(
                        mglRendererBindMTLTexture(renderer, cand));
                    MGL_SS_ABORT_TBIND_IF_ENCODER_CLOSED();
                    cand_mtl = cand->mtl_data;
                    cand_att = MAX_COLOR_ATTACHMENTS;
                    cand_cur = mglCurrentDrawFramebufferUsesColorTexture(
                                   ctx, cand, 0u, &cand_att) ||
                               mglSsRenderPassUsesColorTexture(
                                   areas.command
                                       ? areas.command->renderPassStateOwner
                                       : NULL,
                                   cand_mtl, &cand_att);
                }
                void *cand_copy = NULL;
                signed char used_prev = 0;
                int copy_ok =
                    cand->is_render_target &&
                    mglRendererGLSampledCopyLooksUsable(
                        cand, expected_type, (MGLTextureDataKind)expected_kind,
                        cand_cur ? 1 : 0, &cand_copy, &used_prev);
                MGLDepthRecoverInput hin = {0};
                mglBindingTextureFillDepthRecoverHistoryInput(
                    &hin, 1, cand->is_render_target ? 1 : 0, cand_cur ? 1 : 0,
                    cand_mtl ? 1 : 0, copy_ok ? 1 : 0,
                    cand_mtl && mglMetalPixelFormatIsDepthOrStencil(
                                    mglSsTexturePixelFormat(cand_mtl)),
                    !cand_mtl || expected_type == 0 ||
                        mglSsTextureType(cand_mtl) == expected_type,
                    !cand_mtl ||
                        mglTexturePixelFormatCompatibleWithExpectedDataKind(
                            mglSsTexturePixelFormat(cand_mtl), expected_kind));
                MGLDepthRecoverPlan hplan = {0};
                (void)mglBindingTexturePlanDepthRecover(&hin, &hplan);
                if (hplan.action == MGL_DR_ACTION_HISTORY_USE_COPY) {
                    recover_texture = cand;
                    recover_mtl = cand_copy;
                    recover_reason =
                        hplan.reason_tag ? hplan.reason_tag : "history-copy";
                    recovered_from_sampled_copy = 1;
                    recovered_from_previous_version = used_prev ? 1 : 0;
                    recover_att = cand_att;
                    break;
                }
                if (hplan.action == MGL_DR_ACTION_HISTORY_USE_DIRECT) {
                    recover_texture = cand;
                    recover_mtl = cand_mtl;
                    recover_reason =
                        hplan.reason_tag ? hplan.reason_tag : "history-direct";
                    recover_att = cand_att;
                    break;
                }
            }
        } else if (iplan.action == MGL_DR_ACTION_LOG_UNPAIRED) {
            static uint64_t s_unp = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_unp)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_UNPAIRED, .hit = s_unp,
                                .program = fragment_program_name,
                                .binding = spirv_binding, .unit = texture_unit,
                                .depth_tex = ptr ? ptr->name : 0u,
                                .depth_fmt = mglSsTexturePixelFormat(texture),
                                .w = mglSsTextureWidth(texture),
                                .h = mglSsTextureHeight(texture));
            }
        }

        if (recover_texture && recover_mtl) {
            static uint64_t s_hist_rec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_hist_rec)) {
                MGL_EMIT_DR_LOG(
                    .kind = MGL_DR_LOG_HISTORY_RECOVERY, .hit = s_hist_rec,
                    .reason = recover_reason, .program = fragment_program_name,
                    .binding = spirv_binding, .unit = texture_unit,
                    .fbo = paired_fbo_name, .color_att = recover_att,
                    .depth_tex = ptr ? ptr->name : 0u,
                    .recover_tex = recover_texture ? recover_texture->name : 0u,
                    .depth_fmt = mglSsTexturePixelFormat(texture),
                    .recover_fmt = mglSsTexturePixelFormat(recover_mtl),
                    .w = mglSsTextureWidth(recover_mtl),
                    .h = mglSsTextureHeight(recover_mtl),
                    .copy = recovered_from_sampled_copy ? 1 : 0,
                    .prev_ver = recovered_from_previous_version ? 1 : 0,
                    .sampled_ver = recover_texture
                                       ? recover_texture
                                             ->mtl_gl_sampled_write_version
                                       : 0u,
                    .rt_ver = recover_texture
                                  ? recover_texture
                                        ->mtl_render_target_write_version
                                  : 0u,
                    .paired_color = paired_color ? paired_color->name : 0u,
                    .paired_current = paired_cur ? 1 : 0);
            }
            ptr = recover_texture;
            texture = recover_mtl;
        }
        goto done;
    }

    /* ENTER_RT — plan@C rt_sub 0/1/2 + thin bind/fallback ports. */
    {
        Texture *unit_active = texture_unit < TEXTURE_UNITS
                                   ? mglSsState(&areas)->active_textures[texture_unit]
                                   : NULL;
        Texture *unit_2d =
            texture_unit < TEXTURE_UNITS
                ? mglSsState(&areas)
                      ->texture_units[texture_unit]
                      .textures[_TEXTURE_2D]
                : NULL;
        Texture *last_2d = texture_unit < TEXTURE_UNITS
                               ? mglSsState(&areas)
                                     ->last_sampled_2d_textures[texture_unit]
                               : NULL;
        Texture *recover_texture = NULL;
        const char *recover_reason = "none";
        GLuint recover_fbo_name = 0u;
        Texture *paired_color = mglFindFramebufferColorTexturePairedWithDepth(
            ctx, ptr, &recover_fbo_name);
        size_t draw_att = MAX_COLOR_ATTACHMENTS;
        if (paired_color) {
            RETURN_FALSE_ON_FAILURE(
                mglRendererBindMTLTexture(renderer, paired_color));
            MGL_SS_ABORT_TBIND_IF_ENCODER_CLOSED();
            void *paired_mtl = paired_color->mtl_data;
            int paired_cur = mglSsRenderPassUsesColorTexture(
                areas.command ? areas.command->renderPassStateOwner : NULL,
                paired_mtl, &draw_att);
            MGLDepthRecoverInput rin = {0};
            mglBindingTextureFillDepthRecoverRTInput(
                &rin, 0, 1, paired_mtl ? 1 : 0, paired_cur ? 1 : 0,
                paired_mtl && mglMetalPixelFormatIsDepthOrStencil(
                                  mglSsTexturePixelFormat(paired_mtl)),
                !paired_mtl || expected_type == 0 ||
                    mglSsTextureType(paired_mtl) == expected_type,
                !paired_mtl ||
                    mglTexturePixelFormatCompatibleWithExpectedDataKind(
                        mglSsTexturePixelFormat(paired_mtl), expected_kind),
                0, 0, 0, 0);
            MGLDepthRecoverPlan rplan = {0};
            (void)mglBindingTexturePlanDepthRecover(&rin, &rplan);
            if (rplan.action == MGL_DR_ACTION_RT_USE_PAIRED) {
                recover_texture = paired_color;
                recover_reason =
                    rplan.reason_tag ? rplan.reason_tag : "paired-color";
            } else if (rplan.action == MGL_DR_ACTION_RT_SKIP_CURRENT) {
                static uint64_t s_skip = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_skip)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_SKIP, .hit = s_skip,
                                    .program = fragment_program_name,
                                    .name = sampled_name,
                                    .binding = spirv_binding,
                                    .unit = texture_unit, .fbo = recover_fbo_name,
                                    .depth_tex = ptr ? ptr->name : 0u,
                                    .color_tex = paired_color ? paired_color->name
                                                              : 0u);
                }
            }
        }
        int still_depth = texture && mglMetalPixelFormatIsDepthOrStencil(
                                         mglSsTexturePixelFormat(texture));
        MGLDepthRecoverInput r1 = {0};
        mglBindingTextureFillDepthRecoverRTInput(
            &r1, 1, 0, 0, 0, 0, 0, 0, recover_texture ? 1 : 0,
            (!recover_texture &&
             mglRendererTextureLooksRecoverableSampled2D(
                 ctx, last_2d, expected_type,
                 (MGLTextureDataKind)expected_kind))
                ? 1
                : 0,
            still_depth ? 1 : 0, 0);
        MGLDepthRecoverPlan p1 = {0};
        (void)mglBindingTexturePlanDepthRecover(&r1, &p1);
        if (p1.action == MGL_DR_ACTION_RT_SUPPRESS_LAST2D) {
            static uint64_t s_sup = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_sup)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_SUPPRESS_LAST2D,
                                .hit = s_sup, .program = fragment_program_name,
                                .name = sampled_name, .binding = spirv_binding,
                                .unit = texture_unit,
                                .depth_tex = ptr ? ptr->name : 0u,
                                .last2d = last_2d->name);
            }
            mglBindingTextureFillDepthRecoverRTInput(
                &r1, 1, 0, 0, 0, 0, 0, 0, recover_texture ? 1 : 0, 0,
                still_depth ? 1 : 0, 0);
            (void)mglBindingTexturePlanDepthRecover(&r1, &p1);
        }
        if (p1.action == MGL_DR_ACTION_RT_APPLY && recover_texture) {
            RETURN_FALSE_ON_FAILURE(
                mglRendererBindMTLTexture(renderer, recover_texture));
            MGL_SS_ABORT_TBIND_IF_ENCODER_CLOSED();
            void *recover_mtl = recover_texture->mtl_data;
            int recover_ok =
                recover_mtl &&
                !mglMetalPixelFormatIsDepthOrStencil(
                    mglSsTexturePixelFormat(recover_mtl)) &&
                (expected_type == 0 ||
                 mglSsTextureType(recover_mtl) == expected_type) &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglSsTexturePixelFormat(recover_mtl), expected_kind);
            MGLDepthRecoverInput r2 = {0};
            mglBindingTextureFillDepthRecoverRTInput(
                &r2, 2, 0, 0, 0, 0, 0, 0, 0, 0, still_depth ? 1 : 0,
                recover_ok ? 1 : 0);
            MGLDepthRecoverPlan p2 = {0};
            (void)mglBindingTexturePlanDepthRecover(&r2, &p2);
            if (p2.action == MGL_DR_ACTION_USE_RECOVER) {
                Framebuffer *current_fbo =
                    ctx ? mglSsState(&areas)->framebuffer : NULL;
                GLuint color_tex_name = 0u;
                GLuint depth_tex_name = 0u;
                if (current_fbo &&
                    mglObjectPointerLooksPlausible(current_fbo) &&
                    mglPointerRangeIsReadable(current_fbo, sizeof(*current_fbo))) {
                    color_tex_name = current_fbo->color_attachments[0].texture;
                    depth_tex_name = current_fbo->depth.texture;
                }
                static uint64_t s_rt = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_rt)) {
                    MGL_EMIT_DR_LOG(
                        .kind = MGL_DR_LOG_RT_RECOVER, .hit = s_rt,
                        .reason = recover_reason,
                        .program = fragment_program_name, .name = sampled_name,
                        .binding = spirv_binding, .unit = texture_unit,
                        .depth_tex = ptr ? ptr->name : 0u,
                        .recover_tex = recover_texture->name,
                        .depth_fmt = mglSsTexturePixelFormat(texture),
                        .recover_fmt = mglSsTexturePixelFormat(recover_mtl),
                        .w = mglSsTextureWidth(texture),
                        .h = mglSsTextureHeight(texture),
                        .level = depth_sample_level0,
                        .ever = depth_sample_level0
                                    ? depth_sample_level0->ever_written
                                    : 0u,
                        .init = depth_sample_level0
                                    ? depth_sample_level0->has_initialized_data
                                    : 0u,
                        .unit_active = mglTraceTextureName(unit_active),
                        .unit_tex2d = mglTraceTextureName(unit_2d),
                        .unit_last2d = mglTraceTextureName(last_2d),
                        .recover_fbo = recover_fbo_name,
                        .current_fbo = current_fbo ? current_fbo->name : 0u,
                        .color_tex = color_tex_name,
                        .fbo_depth_tex = depth_tex_name);
                }
                ptr = recover_texture;
                texture = recover_mtl;
                still_depth = 0;
            }
        }
        if (still_depth ||
            (texture && mglMetalPixelFormatIsDepthOrStencil(
                            mglSsTexturePixelFormat(texture)))) {
            void *fallback_texture = mglSampledFallbackTextureForExpectedType(
                renderer, expected_type, expected_kind);
            if (fallback_texture) {
                static uint64_t s_fb = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_fb)) {
                    MGL_EMIT_DR_LOG(
                        .kind = MGL_DR_LOG_RT_FALLBACK, .hit = s_fb,
                        .program = fragment_program_name, .name = sampled_name,
                        .binding = spirv_binding, .unit = texture_unit,
                        .depth_tex = ptr ? ptr->name : 0u,
                        .depth_fmt = mglSsTexturePixelFormat(texture),
                        .w = mglSsTextureWidth(texture),
                        .h = mglSsTextureHeight(texture),
                        .level = depth_sample_level0,
                        .ever = depth_sample_level0
                                    ? depth_sample_level0->ever_written
                                    : 0u,
                        .init = depth_sample_level0
                                    ? depth_sample_level0->has_initialized_data
                                    : 0u,
                        .unit_active = mglTraceTextureName(unit_active),
                        .unit_tex2d = mglTraceTextureName(unit_2d),
                        .unit_last2d = mglTraceTextureName(last_2d));
                }
                texture = fallback_texture;
                used_fallback = 1;
            }
        }
    }

done:
    *ptr_ptr = ptr;
    *texture_ptr = texture;
    *suppress_missing_ptr = suppress_missing;
    *used_fallback_ptr = used_fallback;
    return true;
}

/* === sampled diagnostic ports (P0-1, log 155) ============================ */


/* -emitSampledDiagPortsForProgram:stage:stageIsFragment:sampledName:
 *  spirvBinding:textureUnit:sampledResource:ptr:texture:sampler:usedFallback:
 *  expectedType:lookupType:bindCall:programName:vertexProgramName:
 *  fragmentProgramName:usedSampledCopyTrace:directTextureForTrace:
 *  sampledCopyForTrace:focusedCounter:traceFileCounter: */
void mglSampledEmitDiagPorts(
    void *renderer, Program *program, const char *stage, int stage_is_fragment,
    const char *sampled_name, GLuint spirv_binding, GLuint texture_unit,
    MGLShaderResource *sampled_resource, Texture *ptr, void *texture,
    void *sampler, int used_fallback, uint32_t expected_type,
    uint32_t lookup_type, uint64_t bind_call, GLuint program_name,
    GLuint vertex_program_name, GLuint fragment_program_name,
    int used_sampled_copy_trace, void *direct_texture_for_trace,
    void *sampled_copy_for_trace, uint64_t *focused_counter,
    uint64_t *trace_file_counter)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglSsState(&areas);

    TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
    int expected_index = (int)mglRenderTextureIndexForMetalType(
        (lookup_type ? lookup_type : expected_type));
    Texture *unit_active = NULL;
    Texture *unit_expected = NULL;
    Texture *unit_2d = NULL;
    Texture *unit_cube = NULL;
    if (texture_unit < TEXTURE_UNITS) {
        unit_active = state->active_textures[texture_unit];
        unit_2d = state->texture_units[texture_unit].textures[_TEXTURE_2D];
        unit_cube = state->texture_units[texture_unit].textures[_TEXTURE_CUBE_MAP];
        if (expected_index >= 0 && expected_index < _MAX_TEXTURE_TYPES) {
            unit_expected = state->texture_units[texture_unit]
                                .textures[expected_index];
        }
    }
    MGLSampledDiagEmitInput ein = {0};
    mglBindingTextureFillSampledDiagEmitCore(
        &ein, stage, program_name, vertex_program_name, fragment_program_name,
        sampled_name, spirv_binding, texture_unit,
        sampled_resource ? (int)sampled_resource->sampler_unit : -1,
        (sampled_resource && sampled_resource->sampler_unit_explicit) ? 1 : 0,
        ptr ? ptr->name : 0u, ptr ? ptr->target : 0u,
        (used_fallback || (stage_is_fragment && ptr && ptr->name == 13u)) ? 1 : 0,
        expected_type, lookup_type, expected_index,
        mglTraceTextureName(unit_active), mglTraceTextureName(unit_expected),
        mglTraceTextureName(unit_2d), mglTraceTextureName(unit_cube),
        texture ? mglSsTextureType(texture) : 0,
        texture ? mglSsTextureWidth(texture) : 0,
        texture ? mglSsTextureHeight(texture) : 0,
        texture ? mglSsTexturePixelFormat(texture)
                : MGL_BINDING_PIXEL_FORMAT_INVALID,
        level0 ? level0->width : 0u, level0 ? level0->height : 0u,
        level0 ? level0->depth : 0u, level0 ? level0->data_size : 0u,
        level0 ? level0->ever_written : 0u,
        level0 ? level0->has_initialized_data : 0u,
        level0 ? level0->suspicious_zero_upload : 0u,
        level0 ? level0->last_init_source : 0u,
        level0 ? level0->last_upload_size : 0u,
        level0 ? level0->last_src_hash : 0ull,
        (level0 && level0->data && level0->data_size > 0)
            ? mglTraceHashBytes((const void *)(uintptr_t)level0->data,
                                level0->data_size)
            : 0ull,
        ptr ? ptr->name : 0u, stage_is_fragment ? 1 : 0,
        stage_is_fragment && ptr && mglTextureCanUseGLSampledRenderTargetCopy(ptr)
            ? 1
            : 0,
        stage_is_fragment && mglIsFocusedLoadingProgram(program_name) &&
                (bind_call <= 2048ull || ((bind_call % 512ull) == 0ull))
            ? 1
            : 0,
        !stage_is_fragment &&
                ((program && program->name == 34u) ||
                 (!program && program_name == 34u))
            ? 1
            : 0,
        ptr && mglRenderTextureTargetIsBuffer((uint32_t)ptr->target) ? 1 : 0,
        level0 && level0->suspicious_zero_upload, level0 && !level0->ever_written,
        level0 && !level0->has_initialized_data, texture ? 1 : 0, bind_call,
        used_sampled_copy_trace ? 1 : 0,
        ctx && state->framebuffer ? state->framebuffer->name : 0u,
        areas.command ? areas.command->renderPassFramebufferName : 0u,
        texture_unit < TEXTURE_UNITS
            ? mglTraceTextureName(
                  state->texture_units[texture_unit]
                      .textures[_TEXTURE_BUFFER_TARGET])
            : 0u);
    ein.mtl = texture;
    ein.ptr = ptr;
    ein.sampler = sampler;
    ein.l0_src = level0 ? (const void *)(uintptr_t)level0->last_src_ptr : NULL;
    ein.do_focused = mglProgramNeedsBindingTrace(program) &&
                     mglShouldLogFocusedBinding(focused_counter);
    ein.do_trace_file =
        mglProgramNeedsTraceLog(program) &&
        mglShouldLogTraceFileBindingForProgram(program, trace_file_counter);
    ein.direct_for_trace = direct_texture_for_trace;
    ein.copy_for_trace = sampled_copy_for_trace;
    ein.rt_label = mglTraceTextureLabel(ptr);
    ein.rp_color = mglRenderGetRenderPassAttachmentTextureOwner(
        areas.command ? areas.command->renderPassStateOwner : NULL,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    ein.rp_depth = mglRenderGetRenderPassAttachmentTextureOwner(
        areas.command ? areas.command->renderPassStateOwner : NULL,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    MGLSampledDiagEmitResult eres = {0};
    mglBindingTextureEmitSampledDiagPorts(&ein, &eres);
    if (eres.want_readback && texture && level0) {
        mglRendererTraceSampledTextureReadbackPort(
            renderer, texture, ptr, level0, program_name, spirv_binding,
            stage_is_fragment ? "fragment" : "vertex",
            eres.readback_reason ? eres.readback_reason : "",
            eres.readback_hit);
    }
}

/* === sampled texture binding for one stage (P0-1, log 156) =============== */

/* Twin of the +BindingState.m static. */
static MGLShaderResource *mglSsResourceAtOrdinal(Program *program, int stage,
                                                 int res_type, GLuint ordinal,
                                                 GLuint *element_out)
{
    if (element_out) {
        *element_out = 0u;
    }
    if (!program || stage < 0 || res_type < 0) {
        return NULL;
    }
    MGLShaderResourceList *list = &program->shader_resources_list[stage][res_type];
    GLuint rem = ordinal;
    for (GLuint ri = 0; ri < list->count; ri++) {
        GLuint elements = mglRenderShaderResourceElementCount(
            (uint32_t)list->list[ri].gl_array_size);
        if (rem < elements) {
            if (element_out) {
                *element_out = rem;
            }
            return &list->list[ri];
        }
        rem -= elements;
    }
    return NULL;
}

/* -bindSampledTexturesForStage:isFragmentStage:program:programName:
 *  vertexProgramName:fragmentProgramName:defaultSampler:bindCall:traceBind:
 *  boundCount:fallbackCount:nilCount:samplerCount:sampledCount: */
bool mglSampledBindTexturesForStage(
    void *renderer, int shader_stage, int is_fragment, Program *program,
    GLuint program_name, GLuint vertex_program_name,
    GLuint fragment_program_name, void *default_sampler, uint64_t bind_call,
    int trace_bind, GLuint *bound_count, GLuint *fallback_count,
    GLuint *nil_count, GLuint *sampler_count, GLuint *sampled_count_out)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    void *binding_state_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    GLuint bound = bound_count ? *bound_count : 0u;
    GLuint fallback = fallback_count ? *fallback_count : 0u;
    GLuint nil_tex = nil_count ? *nil_count : 0u;
    GLuint bound_samplers = sampler_count ? *sampler_count : 0u;
    const int use_resource_snapshot = 1;
    MGLRenderResourceBindingSnapshot resource_snapshot = {0};
    const uint32_t metal_stage = is_fragment ? MGL_RENDER_BINDING_STAGE_FRAGMENT
                                           : MGL_RENDER_BINDING_STAGE_VERTEX;
    const char *stage_tag = is_fragment ? "fragment" : "vertex";
    GLuint count = mglRendererGetProgramBindingCount(ctx, shader_stage,
                                                     _SAMPLED_IMAGE_RES);
    if (sampled_count_out) {
        *sampled_count_out = count;
    }
    static uint64_t s_focused_sampled_logs[2] = {0, 0};
    static uint64_t s_trace_file_sampled_logs[2] = {0, 0};

    for (GLuint i = 0; i < count; i++) {
        Program *sample_program = program;
        MGLShaderResource *sampled_resource = mglSsResourceAtOrdinal(
            sample_program, shader_stage, _SAMPLED_IMAGE_RES, i, NULL);
        const char *sampled_name = sampled_resource ? sampled_resource->name : "";
        GLuint spirv_binding = sampled_resource
            ? (GLuint)mglRendererGetProgramBinding(ctx, shader_stage,
                                                   _SAMPLED_IMAGE_RES, (int32_t)i)
            : 0u;
        GLuint glBinding = sampled_resource
            ? (GLuint)mglRendererGetProgramGLBinding(ctx, shader_stage,
                                                     _SAMPLED_IMAGE_RES, (int32_t)i)
            : 0u;
        MGLSampledTextureBindInput sin = {0};
        mglBindingTextureFillSampledGateInput(
            &sin, spirv_binding, glBinding, TEXTURE_UNITS,
            /* No skip recipe: the SPIRV-era resource-skip heuristics are
             * gone, so a sampler-like resource always reaches the plan. */
            0,
            sampled_resource ? 1 : 0);
        MGLSampledTextureBindPlan splan = {0};
        if (mglBindingTexturePlanSampled(&sin, &splan) != 0 ||
            splan.action == MGL_ST_ACTION_SKIP) {
            continue;
        }
        GLuint texture_unit = mglTextureUnitForSampledResource(sampled_resource, sample_program, spirv_binding, shader_stage);
        uint32_t expected_type = (uint32_t)mglExpectedTextureTypeForResource(
            sample_program, shader_stage, sampled_resource);
        uint32_t lookup_type =
            (uint32_t)mglDeclaredTextureTypeFromResource(sampled_resource);
        MGLTextureDataKind expected_kind =
            (MGLTextureDataKind)mglExpectedTextureDataKindForResource(
                sample_program, shader_stage, sampled_resource);
        Texture *ptr = mglTextureForSampledResource(
            ctx, sampled_resource, spirv_binding, shader_stage,
            (lookup_type ? lookup_type : expected_type), texture_unit);
        void *texture = NULL;
        void *sampler = is_fragment ? NULL : default_sampler;
        void *direct_texture_for_trace = NULL;
        void *sampled_copy_for_trace = NULL;
        int used_fallback = 0;
        int suppress_missing = 0;
        int used_sampled_copy = 0;

        if (ptr) {
            if (is_fragment) {
                {
                    /* `&ptr` is a Texture** (a C struct pointer) so it travels
                     * as-is; the `id *` and `BOOL *` out-params go through a
                     * void* / int temporary (rule 4). */
                    void *texture_raw = texture;
                    int suppress_missing_raw = suppress_missing ? 1 : 0;
                    int used_fallback_raw = used_fallback ? 1 : 0;
                    bool recovered = mglSampledRecoverFragmentDepthTexture(
                        renderer, &ptr, &texture_raw, sampled_name,
                        spirv_binding, texture_unit, expected_type,
                        (uint32_t)expected_kind, fragment_program_name,
                        &suppress_missing_raw, &used_fallback_raw);
                    texture = texture_raw;
                    suppress_missing = suppress_missing_raw ? 1 : 0;
                    used_fallback = used_fallback_raw ? 1 : 0;
                    if (!recovered) {
                        return false;
                    }
                }
                {
                    /* The .m's `id *` out-params travel as void* temporaries
                     * (rule 4: an `id __strong` address cannot become void**)
                     * and the BOOL out-param takes an int temporary. */
                    void *texture_raw = texture;
                    void *direct_trace_raw = direct_texture_for_trace;
                    void *copy_trace_raw = sampled_copy_for_trace;
                    int used_copy_raw = used_sampled_copy ? 1 : 0;
                    bool planned = mglSampledRenderTargetCopyPlan(
                        renderer, ptr, &texture_raw, sample_program,
                        expected_type, (uint32_t)expected_kind,
                        used_fallback ? 1 : 0, stage_tag, program_name,
                        spirv_binding, texture_unit, sampled_name, &used_copy_raw,
                        &direct_trace_raw, &copy_trace_raw);
                    texture = texture_raw;
                    direct_texture_for_trace = direct_trace_raw;
                    sampled_copy_for_trace = copy_trace_raw;
                    used_sampled_copy = used_copy_raw ? 1 : 0;
                    if (!planned) {
                        return false;
                    }
                }
                {
                    /* BOOL out-params take an int temporary in C (rule 4). */
                    int used_fallback_raw = used_fallback ? 1 : 0;
                    texture = mglSampledCompatFallbackPlan(
                        renderer, ptr, texture,
                        expected_type, (uint32_t)expected_kind, stage_tag,
                        program_name, spirv_binding, sample_program,
                        &used_fallback_raw);
                    used_fallback = used_fallback_raw ? 1 : 0;
                }
                if (used_fallback) {
                    used_sampled_copy = 0;
                }
                sampler = mglSampledSamplerMaterialize(
                    renderer, ptr, texture_unit,
                    sampler, 0, ptr ? ptr->target : 0u,
                    program_name, spirv_binding, stage_tag,
                    texture);
                if (mglMipDiagEnabled() && texture_unit < TEXTURE_UNITS) {
                    Sampler *gl_sampler =
                        mglSsState(&areas)->texture_samplers[texture_unit];
                    const TextureParameter *effective =
                        gl_sampler ? &gl_sampler->params : &ptr->params;
                    static uint64_t s_frag_sampler_state[TEXTURE_UNITS];
                    (void)mglBindingTextureEmitMipDiagFragIfChanged(
                        &s_frag_sampler_state[texture_unit],
                        mglBindingTextureMipDiagSignature(
                            ptr->name, effective->min_filter,
                            effective->mag_filter, ptr->params.base_level,
                            ptr->params.max_level,
                            texture ? mglSsTextureMipmapLevelCount(texture)
                                    : 0u,
                            (uint64_t)(uintptr_t)texture, used_sampled_copy ? 1 : 0,
                            ptr->mtl_gl_sampled_levels,
                            ptr->mtl_gl_sampled_dirty_mip_mask,
                            ptr->mtl_gl_sampled_write_version !=
                                ptr->mtl_render_target_write_version),
                        texture_unit, spirv_binding, fragment_program_name, ptr->name,
                        gl_sampler ? "gl_sampler" : "texParams",
                        effective->min_filter, effective->mag_filter,
                        effective->min_lod, effective->max_lod,
                        effective->max_anisotropy, ptr->params.base_level,
                        ptr->params.max_level, ptr->num_levels,
                        texture ? mglSsTextureMipmapLevelCount(texture)
                                : 0u,
                        texture ? mglSsTextureWidth(texture) : 0u,
                        texture ? mglSsTextureHeight(texture) : 0u,
                        texture,
                        ptr->is_render_target ? 1 : 0, used_sampled_copy ? 1 : 0,
                        ptr->mtl_gl_sampled_levels,
                        ptr->mtl_gl_sampled_dirty_mip_mask,
                        ptr->mtl_render_target_write_version,
                        ptr->mtl_gl_sampled_write_version);
                }
            } else {
                RETURN_FALSE_ON_FAILURE(mglRendererBindMTLTexture(renderer, ptr));
                MGL_SS_ABORT_TBIND_IF_ENCODER_CLOSED();
                if (ptr->mtl_data) {
                    texture = (ptr->mtl_data);
                    texture = mglSampledTextureViewForBaseLevel(
                        ptr, texture);
                }
                {
                    /* BOOL out-params take an int temporary in C (rule 4). */
                    int used_fallback_raw = used_fallback ? 1 : 0;
                    texture = mglSampledCompatFallbackPlan(
                        renderer, ptr, texture,
                        expected_type, (uint32_t)expected_kind, stage_tag,
                        program_name, spirv_binding, sample_program,
                        &used_fallback_raw);
                    used_fallback = used_fallback_raw ? 1 : 0;
                }
                sampler = mglSampledSamplerMaterialize(
                    renderer, ptr, texture_unit,
                    default_sampler, 0, ptr ? ptr->target : 0u,
                    program_name, spirv_binding, stage_tag,
                    texture);
                {
                    void *texture_raw = texture;
                    bool planned = mglSampledRenderTargetCopyPlan(
                        renderer, ptr, &texture_raw, sample_program,
                        expected_type, (uint32_t)expected_kind,
                        used_fallback ? 1 : 0, stage_tag, program_name,
                        spirv_binding, texture_unit, sampled_name, NULL, NULL,
                        NULL);
                    texture = texture_raw;
                    if (!planned) {
                        return false;
                    }
                }
            }
        }

        GLuint sampler_binding =
            sampled_resource && sampled_resource->has_combined_sampler
                ? mglMetalCombinedSamplerSlot(sampled_resource)
                : spirv_binding;
        mglBindingTextureFillSampledFinalInput(
            &sin, texture ? 1 : 0, suppress_missing ? 1 : 0, used_fallback ? 1 : 0,
            sampled_resource && sampled_resource->has_combined_sampler ? 1 : 0,
            sampler_binding, (uint32_t)kMaxFragmentSamplerSlots, sampler ? 1 : 0,
            is_fragment ? mglBindingTextureForceDefaultSampler(
                             used_fallback ? 1 : 0,
                             expected_kind == MGLTextureDataKindDepth ? 1 : 0)
                       : 0);

        if (!texture && !suppress_missing) {
            texture = mglSampledFallbackTextureForExpectedType(renderer, expected_type, expected_kind);
            if (texture) {
                sin.has_bound_texture = 1;
                fallback++;
                if (is_fragment) {
                    used_fallback = 1;
                    used_sampled_copy = 0;
                    mglFocusLoadingProgram(fragment_program_name, "sample-fallback",
                                           bind_call);
                    static uint64_t s_frag_fb_log = 0;
                    if (mglBindingTextureRateLogHit(&s_frag_fb_log, 32ull, 512ull)) {
                        mglBindingLogTexFallbackEx(
                            s_frag_fb_log, spirv_binding, fragment_program_name,
                            ptr ? ptr->name : 0u, 0, NULL, 0u);
                    }
                }
            }
        } else if (is_fragment && !texture && suppress_missing) {
            static uint64_t s_frag_fb_sup = 0;
            if (mglBindingTextureRateLogHit(&s_frag_fb_sup, 64ull, 512ull)) {
                mglBindingLogTexFallbackEx(
                    s_frag_fb_sup, spirv_binding, fragment_program_name,
                    ptr ? ptr->name : 0u, 1, sampled_name, texture_unit);
            }
        }

        if (is_fragment) {
            MGLSamplerMaterializeInput fin = {0};
            mglBindingTextureFillSamplerMaterializeInput(
                &fin, sin.force_default_sampler, 0, 0, 0, 0, 0, 1);
            MGLSamplerMaterializePlan fplan = {0};
            (void)mglBindingTexturePlanSamplerMaterialize(&fin, &fplan);
            if (fplan.action == MGL_SM_ACTION_USE_DEFAULT || !sampler) {
                sampler = default_sampler;
            }
            sin.has_sampler = sampler ? 1 : 0;
            sin.used_type_fallback = used_fallback ? 1 : 0;
            sin.force_default_sampler = mglBindingTextureForceDefaultSampler(
                used_fallback ? 1 : 0,
                expected_kind == MGLTextureDataKindDepth ? 1 : 0);
        }

        if (mglBindingTexturePlanSampled(&sin, &splan) != 0) {
            continue;
        }
        if (!is_fragment && splan.action != MGL_ST_ACTION_QUEUE) {
            continue;
        }

        if ((is_fragment || splan.queue_texture) &&
            !mglSsQueueResourceBinding(
                use_resource_snapshot, binding_state_owner,
                (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
                &resource_snapshot, metal_stage,
                MGL_RENDER_RESOURCE_BINDING_TEXTURE, texture,
                is_fragment ? spirv_binding : splan.texture_slot)) {
            return false;
        }
        if (!is_fragment && splan.queue_sampler &&
            !mglSsQueueResourceBinding(
                use_resource_snapshot, binding_state_owner,
                (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
                &resource_snapshot, metal_stage,
                MGL_RENDER_RESOURCE_BINDING_SAMPLER, sampler,
                splan.sampler_slot)) {
            return false;
        }

        GLuint sample_program_name =
            sample_program ? sample_program->name : program_name;
        if (is_fragment && spirv_binding < TEXTURE_UNITS) {
            mglBindingTextureWriteFragTrace(
                &areas.fragment_trace_bindings[spirv_binding],
                ptr ? ptr->name : 0u, texture_unit, spirv_binding, sample_program_name,
                ptr ? ptr->mtl_render_target_write_version : 0u,
                ptr ? ptr->mtl_gl_sampled_write_version : 0u, ptr,
                texture,
                (direct_texture_for_trace ? direct_texture_for_trace
                                                       : texture),
                sampled_copy_for_trace,
                texture ? mglSsTextureWidth(texture) : 0u,
                texture ? mglSsTextureHeight(texture) : 0u,
                texture ? mglSsTexturePixelFormat(texture)
                        : MGL_BINDING_PIXEL_FORMAT_INVALID,
                texture ? mglSsTextureType(texture) : 0u,
                used_sampled_copy ? 1 : 0, used_fallback ? 1 : 0);
        }

        mglSampledEmitDiagPorts(
            renderer, sample_program, stage_tag, is_fragment ? 1 : 0,
            sampled_name, spirv_binding, texture_unit, sampled_resource, ptr,
            texture, sampler,
            used_fallback ? 1 : 0, expected_type, lookup_type, bind_call,
            sample_program_name, vertex_program_name,
            is_fragment ? fragment_program_name : 0u, used_sampled_copy ? 1 : 0,
            direct_texture_for_trace,
            sampled_copy_for_trace,
            &s_focused_sampled_logs[is_fragment ? 1 : 0],
            &s_trace_file_sampled_logs[is_fragment ? 1 : 0]);

        if (is_fragment) {
            switch (mglBindingTextureSampledMarkKind(texture ? 1 : 0,
                                                     used_fallback ? 1 : 0)) {
            case MGL_ST_MARK_BOUND:
                bound++;
                break;
            case MGL_ST_MARK_FALLBACK:
                /* Keep nil_tex as original GL failure count; Metal gets fallback. */
                nil_tex++;
                break;
            default:
                nil_tex++;
                break;
            }
            if (sampler &&
                (!sampled_resource || sampled_resource->has_combined_sampler) &&
                sampler_binding < kMaxFragmentSamplerSlots) {
                if (!mglSsQueueResourceBinding(
                        use_resource_snapshot, binding_state_owner,
                        (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
                        &resource_snapshot, metal_stage,
                        MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                        sampler, sampler_binding)) {
                    return false;
                }
                bound_samplers++;
            }
            if (trace_bind && i < 6) {
                TextureLevel *level0 =
                    (ptr && ptr->faces[0].levels) ? &ptr->faces[0].levels[0]
                                                  : NULL;
                uint32_t cpuFirst = 0u;
                int cpuOk = level0 && level0->data && level0->data_size >= 4 &&
                            (uintptr_t)level0->data >= 0x1000ull;
                if (cpuOk) {
                    memcpy(&cpuFirst, (const void *)level0->data, 4);
                }
                mglTraceLog(
                    "texbind.sampled call=%llu idx=%u binding=%u glTex=%u "
                    "target=0x%x internal=0x%x l0=%ux%ux%u l0bytes=%lu "
                    "l0first=0x%08x(valid=%d) source=%u upload=%lu src=%p "
                    "hash=0x%016llx ever=%u full=%u zero=%u mtl=%p "
                    "size=%lux%lu sampler=%p fallback=%d",
                    (unsigned long long)bind_call, (unsigned)i,
                    (unsigned)spirv_binding, ptr ? (unsigned)ptr->name : 0u,
                    ptr ? (unsigned)ptr->target : 0u,
                    ptr ? (unsigned)ptr->internalformat : 0u,
                    level0 ? (unsigned)level0->width : 0u,
                    level0 ? (unsigned)level0->height : 0u,
                    level0 ? (unsigned)level0->depth : 0u,
                    (unsigned long)(level0 ? level0->data_size : 0u),
                    (unsigned)cpuFirst, cpuOk ? 1 : 0,
                    (unsigned)(level0 ? level0->last_init_source : 0u),
                    (unsigned long)(level0 ? level0->last_upload_size : 0u),
                    (void *)(level0 ? level0->last_src_ptr : NULL),
                    (unsigned long long)(level0 ? level0->last_src_hash : 0ull),
                    (unsigned)(level0 ? level0->ever_written : 0u),
                    (unsigned)(level0 ? level0->has_initialized_data : 0u),
                    (unsigned)(level0 ? level0->suspicious_zero_upload : 0u),
                    texture,
                    (unsigned long)(texture ? mglSsTextureWidth(texture)
                                            : 0),
                    (unsigned long)(texture ? mglSsTextureHeight(texture)
                                            : 0),
                    sampler, used_fallback ? 1 : 0);
            }
        } else if (texture) {
            bound++;
            if (used_fallback) {
                fallback++;
            }
        }
    }

    if (use_resource_snapshot &&
        !mglSsFlushResourceBindings(
            binding_state_owner,
            (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
            &resource_snapshot)) {
        return false;
    }
    if (bound_count) {
        *bound_count = bound;
    }
    if (fallback_count) {
        *fallback_count = fallback;
    }
    if (nil_count) {
        *nil_count = nil_tex;
    }
    if (sampler_count) {
        *sampler_count = bound_samplers;
    }
    return true;
    return true;
}

/* === whole-encoder texture binding + storage-image views (P0-1, log 157) == */

/* Twins of the remaining +BindingState.m statics. */
static MGLRenderTextureInfo mglSsTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

static uint64_t mglSsTextureArrayLength(void *texture)
{
    return mglSsTextureInfo(texture).array_length;
}

static void *mglSsCacheImageUnitView(ImageUnit *iu, void *fallback, void *view)
{
    if (!view) {
        return fallback;
    }
    if (iu->mtl_image_view) {
        mglRenderReleaseMetalObject(iu->mtl_image_view);
        iu->mtl_image_view = NULL;
    }
    iu->mtl_image_view = view; /* +1 from newTextureView */
    return iu->mtl_image_view;
}

/* BindImageTexture <format> -> PixelFormatView (CTS advanced-cast). */
static uint32_t mglSsImageBindPixelFormat(const ImageUnit *iu,
                                          uint32_t native_format)
{
    if (!iu) {
        return native_format;
    }
    const uint32_t bind_format = mtlFormatForGLInternalFormat(iu->internalformat);
    return mglRenderImageBindPixelFormat(iu->internalformat, native_format,
                                         bind_format);
}

static int mglSsHasActiveEncoder(const MGLEncodeContext *enc_ctx)
{
    if (!enc_ctx) {
        return 0;
    }
    return mglRenderEncoderOwnerHasCurrent(enc_ctx->render_encoder_owner) != 0;
}

/* MGLRenderer+Draw_Private.h's static inline. */
static int mglSsShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return 0;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

/* Storage-image view: non-layered slice + format/mip PixelFormatView (CTS).
 * Keeps its exported name (mgl_renderer_backend.h). */
void *mglRendererStorageImageTexture(void *base_texture, ImageUnit *iu)
{
    void *texture = base_texture;
    if (!texture || !iu) {
        return base_texture;
    }
    if (iu->mtl_image_view) {
        return iu->mtl_image_view;
    }
    const MGLRenderTextureInfo info = mglSsTextureInfo(texture);
    if (info.width == 0u) {
        return base_texture;
    }
    const size_t level = (size_t)iu->level;
    /* CTS incomplete_textures: mip past mipmapLevelCount → unbound. */
    if (!mglRenderImageLevelInRange((uint32_t)level,
                                    (uint32_t)info.mipmap_level_count)) {
        return NULL;
    }
    const uint32_t srcType = info.texture_type;
    const uint32_t bindFormat =
        mglSsImageBindPixelFormat(iu, info.pixel_format);
    const GLenum glTarget = iu->tex ? iu->tex->target : (GLenum)0;
    const int isMsTarget = mglRenderImageTargetIsMultisample((uint32_t)glTarget);
    uint32_t dstType = 0u;
    if (mglRenderImageNeedsNonLayeredSlice(iu->layered ? 1 : 0, isMsTarget,
                                           srcType, &dstType)) {
            void *view = NULL;
            int rc = mglRenderCreateTextureViewRange(
                    base_texture, bindFormat, dstType,
                    level, 1u, (uint64_t)iu->layer, 1u,
                    0, 0, 0, 0, 0, &view);
            if (rc == 0 && view) {
                return mglSsCacheImageUnitView(
                    iu, texture, view);
            }
    }

    if (mglRenderImageNeedsFormatOrMipView(level, bindFormat,
                                           info.pixel_format)) {
        size_t slice_count = (size_t)mglRenderImageViewSliceCount(
            srcType, mglSsTextureArrayLength(texture));
        void *view = NULL;
        if (mglRenderCreateTextureViewRange(
                base_texture, bindFormat,
                info.texture_type, level, 1u, 0u, slice_count,
                0, 0, 0, 0, 0, &view) == 0 && view) {
            return mglSsCacheImageUnitView(
                iu, texture, view);
        }
    }
    return base_texture;
}

/* -bindTexturesToCurrentRenderEncoder: */
bool mglBindTexturesToCurrentRenderEncoder(void *renderer,
                                           const MGLEncodeContext *enc_ctx)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    void *binding_state_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    static uint64_t s_bindTexturesCallCount = 0;
    uint64_t bindCall = ++s_bindTexturesCallCount;
    bool traceBind = mglSsShouldTraceCall(bindCall);
    GLuint vertexSampledCount = 0;
    GLuint vertexBoundTextures = 0;
    GLuint vertexFallbackTextures = 0;
    GLuint boundSampledTextures = 0;
    GLuint nilSampledTextures = 0;
    GLuint fallbackSampledTextures = 0;
    GLuint boundSampledSamplers = 0;
    Program *vertexProgram = NULL;
    Program *fragmentProgram = NULL;
    GLuint vertexProgramName = 0u;
    GLuint fragmentProgramName = 0u;
    const int use_resource_snapshot = 1;
    MGLRenderResourceBindingSnapshot resource_snapshot = {0};

    if (!mglSsHasActiveEncoder(enc_ctx)) {
        // No active render encoder yet (or it was rotated). Texture/sampler binding
        // can be deferred until the next encoder is created.
        return true;
    }

    /* Per-draw sampler snapshot for replay / RT-copy cull; clear stale slots. */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             "bind_textures_begin",
                                             areas.fragment_trace_bindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             areas.pipeline_cache->pipelineProgramName);

        memset(areas.fragment_trace_bindings, 0,
               sizeof(areas.fragment_trace_bindings));
    } else {
        mglClearFragmentTextureTraceFunctionalFlags(
            areas.fragment_trace_bindings, TEXTURE_UNITS);
    }

    const int vertexResourceStage = areas.tess_native_tes_active
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    vertexProgram = areas.tess_native_tes_active
        ? areas.tess_native_tes_program
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    vertexProgramName = vertexProgram ? vertexProgram->name : mglCurrentRenderProgramKey(ctx);
    fragmentProgramName = fragmentProgram ? fragmentProgram->name : mglCurrentRenderProgramKey(ctx);

    void *default_sampler = mglTextureFallbackSamplerState(renderer);
    if (default_sampler) {
        if (vertexProgram) {
            (void)mglProgramSamplesTextureUnit(vertexProgram, 0);
        }
        if (fragmentProgram && fragmentProgram != vertexProgram) {
            (void)mglProgramSamplesTextureUnit(fragmentProgram, 0);
        }
        MGLSamplerWarmupPlan warm = {0};
        mglBindingTexturePlanSamplerWarmup(
            1, vertexProgram ? 1 : 0, fragmentProgram ? 1 : 0,
            vertexProgram ? vertexProgram->sampled_texture_unit_mask : NULL,
            (fragmentProgram && fragmentProgram != vertexProgram)
                ? fragmentProgram->sampled_texture_unit_mask
                : NULL,
            (uint32_t)TEXTURE_UNITS, (uint32_t)kMaxFragmentSamplerSlots, &warm);
        for (uint32_t s = 0; s < warm.warmup_count; s++) {
            if (warm.mode == MGL_SW_MODE_MASK &&
                !mglBindingTextureSamplerWarmupSlotActive(warm.mask, s)) {
                continue;
            }
            if (!mglSsQueueResourceBinding(
                    use_resource_snapshot, binding_state_owner,
                    (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
                    &resource_snapshot, MGL_RENDER_BINDING_STAGE_VERTEX,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    default_sampler, s) ||
                !mglSsQueueResourceBinding(
                    use_resource_snapshot, binding_state_owner,
                    (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
                    &resource_snapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    default_sampler, s)) {
                return false;
            }
        }
    }

    if (use_resource_snapshot &&
        !mglSsFlushResourceBindings(
            binding_state_owner,
            (areas.command ? areas.command->currentRenderEncoderOwner : NULL),
            &resource_snapshot)) {
        return false;
    }

    GLuint sampledCount = 0;
    GLuint separateSamplerCount = 0;
    GLuint boundSeparateSamplers = 0;

    /* Bind VS+FS sampled images (Metal validates every active stage). */
    if (!mglSampledBindTexturesForStage(
            renderer, vertexResourceStage, 0, vertexProgram,
            vertexProgramName, vertexProgramName, 0u,
            default_sampler, bindCall, traceBind ? 1 : 0,
            &vertexBoundTextures, &vertexFallbackTextures, NULL, NULL,
            &vertexSampledCount)) {
        return false;
    }
    if (!mglSampledBindTexturesForStage(
            renderer, _FRAGMENT_SHADER, 1, fragmentProgram,
            fragmentProgramName, vertexProgramName, fragmentProgramName,
            default_sampler, bindCall, traceBind ? 1 : 0,
            &boundSampledTextures, &fallbackSampledTextures,
            &nilSampledTextures, &boundSampledSamplers, &sampledCount)) {
        return false;
    }

    /* The storage-image driver is C now (log 130). */
    if (!mglBindingStateBindStorageImagesForVertexProgram(
            renderer, vertexProgram, fragmentProgram)) {
        return false;
    }

    if (!mglSampledBindSeparateSamplersAndArrayTextures(
            renderer, vertexProgram, fragmentProgram,
            fragmentProgramName, vertexProgramName,
            default_sampler, bindCall, traceBind ? 1 : 0,
            &separateSamplerCount, &boundSeparateSamplers)) {
        return false;
    }

    int interesting = (sampledCount > 0 && boundSampledTextures == 0) ||
                       fallbackSampledTextures > 0 || vertexFallbackTextures > 0;
    static uint64_t s_interestingTextureSummaryCount = 0;
    if (traceBind ||
        (interesting && mglBindingTextureRateLogHit(
                            &s_interestingTextureSummaryCount, 64ull, 512ull))) {
        mglTraceLog(
            "texbind.summary call=%llu program=%u vertexSampled=%u "
            "vertexBoundTex=%u vertexFallback=%u sampled=%u boundTex=%u "
            "nilTex=%u fallbackTex=%u sampledSamplers=%u separateSamplers=%u "
            "boundSeparate=%u",
            (unsigned long long)bindCall,
            (unsigned)mglCurrentRenderProgramKey(ctx),
            (unsigned)vertexSampledCount, (unsigned)vertexBoundTextures,
            (unsigned)vertexFallbackTextures, (unsigned)sampledCount,
            (unsigned)boundSampledTextures, (unsigned)nilSampledTextures,
            (unsigned)fallbackSampledTextures, (unsigned)boundSampledSamplers,
            (unsigned)separateSamplerCount, (unsigned)boundSeparateSamplers);
    }

    return true;
}
