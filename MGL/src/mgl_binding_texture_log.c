/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Rate-limited BindingState logging ports for O3.3. Freeze/shrink only —
 * do not spawn another log TU; do not grow this shell. Prefer merged
 * kind/POD entry points (DepthRecover / RTSampleCopy / TexFallbackEx).
 */

#include "mgl_binding_texture.h"
#include <stdio.h>
#include <stdbool.h>
#include "mgl_trace_log.h"

void mglBindingLogTBINDFocused(
    const char *stage, uint32_t program, const char *resource,
    uint32_t metal_slot, uint32_t sampler_unit, uint32_t gl_tex, uint32_t target,
    const void *mtl, uint64_t mtl_type, uint64_t w, uint64_t h,
    uint32_t l0w, uint32_t l0h, uint32_t ever, uint32_t init, uint32_t source)
{
    fprintf(stderr, "MGL TBIND focused stage=%s program=%u resource=%s metalTextureSlot=%u samplerUnit=%u glTex=%u target=0x%x mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
          stage ? stage : "?", (unsigned)program, resource ? resource : "",
          (unsigned)metal_slot, (unsigned)sampler_unit, (unsigned)gl_tex,
          (unsigned)target, mtl, (unsigned long)mtl_type, (unsigned long)w,
          (unsigned long)h, (unsigned)l0w, (unsigned)l0h, (unsigned)ever,
          (unsigned)init, (unsigned)source);
}

void mglBindingLogTBINDTraceFile(
    const char *stage, uint32_t program, const char *resource,
    uint32_t metal_slot, uint32_t sampler_unit, int res_unit, int explicit_unit,
    uint32_t gl_tex, uint32_t target, int fallback, uint64_t expected_type,
    uint64_t lookup_type, int expected_index, uint32_t unit_active,
    uint32_t unit_expected, uint32_t unit_2d, uint32_t unit_cube,
    const void *mtl, uint64_t mtl_type, uint64_t w, uint64_t h, uint32_t l0w,
    uint32_t l0h, uint32_t ever, uint32_t init, uint32_t source)
{
    mglTraceLog(
        "TBIND stage=%s program=%u resource=%s metalTextureSlot=%u samplerUnit=%u "
        "resUnit=%d explicit=%d glTex=%u target=0x%x fallback=%d expectedType=%lu "
        "lookupType=%lu expectedIndex=%d unit(active=%u expected=%u tex2D=%u cube=%u) "
        "mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
        stage ? stage : "?", (unsigned)program, resource ? resource : "",
        (unsigned)metal_slot, (unsigned)sampler_unit, res_unit, explicit_unit,
        (unsigned)gl_tex, (unsigned)target, fallback,
        (unsigned long)expected_type, (unsigned long)lookup_type, expected_index,
        (unsigned)unit_active, (unsigned)unit_expected, (unsigned)unit_2d,
        (unsigned)unit_cube, mtl, (unsigned long)mtl_type, (unsigned long)w,
        (unsigned long)h, (unsigned)l0w, (unsigned)l0h, (unsigned)ever,
        (unsigned)init, (unsigned)source);
}

void mglBindingLogSampleDetail(
    uint64_t bind_call, uint64_t hit, const char *stage, uint32_t program,
    const char *name, uint32_t binding, uint32_t unit, uint64_t expected_type,
    int expected_index, uint32_t ptr_tex, const void *ptr, uint32_t target,
    int fallback, const void *mtl, uint64_t mtl_type, uint64_t mtl_w,
    uint64_t mtl_h, uint32_t unit_active, uint32_t unit_expected,
    uint32_t unit_2d, uint32_t unit_cube, uint32_t l0w, uint32_t l0h,
    uint32_t l0d, uint64_t bytes, uint32_t ever, uint32_t full, uint32_t zero,
    uint32_t source, uint64_t upload, const void *src, uint64_t hash,
    uint64_t data_hash)
{
    mglTraceLog(
        "MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=%s program=%u "
        "name=%s binding=%u unit=%u expectedType=%lu expectedIndex=%d ptrTex=%u "
        "ptr=%p target=0x%x fallback=%d mtlTex=%p mtlType=%lu mtlSize=%lux%lu "
        "unit(active=%u expected=%u tex2D=%u cube=%u) l0=%ux%ux%u bytes=%lu "
        "init(ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx "
        "dataHash=0x%016llx)",
        (unsigned long long)bind_call, (unsigned long long)hit,
        stage ? stage : "?", (unsigned)program, name ? name : "",
        (unsigned)binding, (unsigned)unit, (unsigned long)expected_type,
        expected_index, (unsigned)ptr_tex, ptr, (unsigned)target, fallback, mtl,
        (unsigned long)mtl_type, (unsigned long)mtl_w, (unsigned long)mtl_h,
        (unsigned)unit_active, (unsigned)unit_expected, (unsigned)unit_2d,
        (unsigned)unit_cube, (unsigned)l0w, (unsigned)l0h, (unsigned)l0d,
        (unsigned long)bytes, (unsigned)ever, (unsigned)full, (unsigned)zero,
        (unsigned)source, (unsigned long)upload, src,
        (unsigned long long)hash, (unsigned long long)data_hash);
}

void mglBindingLogTexCompatMismatch(
    const char *kind, const char *stage, uint32_t binding, uint32_t program,
    uint32_t gl_tex, uint64_t mtl_type, uint64_t expected, uint64_t hit)
{
    fprintf(stderr, "MGL TEX %s MISMATCH %s binding=%u program=%u glTex=%u mtlType=%lu expected=%lu hit=%llu",
          kind ? kind : "?", stage ? stage : "?", (unsigned)binding,
          (unsigned)program, (unsigned)gl_tex, (unsigned long)mtl_type,
          (unsigned long)expected, (unsigned long long)hit);
}

void mglBindingLogTexBufferBind(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t ptr_tex, uint32_t active, uint32_t buffer_slot,
    uint64_t expected_type, uint64_t lookup_type, const void *mtl,
    uint64_t mtl_type, uint64_t w, uint64_t h, uint64_t format,
    const void *sampler)
{
    fprintf(stderr, "MGL TEXBUFFER BIND vertex hit=%llu program=%u binding=%u unit=%u ptrTex=%u active=%u bufferSlot=%u expectedType=%lu lookupType=%lu mtlTex=%p mtlType=%lu size=%lux%lu format=%lu sampler=%p",
          (unsigned long long)hit, (unsigned)program, (unsigned)binding,
          (unsigned)unit, (unsigned)ptr_tex, (unsigned)active,
          (unsigned)buffer_slot, (unsigned long)expected_type,
          (unsigned long)lookup_type, mtl, (unsigned long)mtl_type,
          (unsigned long)w, (unsigned long)h, (unsigned long)format, sampler);
}

void mglBindingLogRTYFlipDecision(
    const char *stage, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t tex, const char *label, const char *decision_name,
    int decision, uint32_t authority, uint32_t rt_ver, uint32_t copy_ver,
    int has_copy, int sample_yflip)
{
    mglTraceLog(
        "RT_YFLIP_DECISION stage=%s program=%u name=%s binding=%u unit=%u tex=%u "
        "label=\"%s\" decision=%s(%d) authority=0x%x rtVer=%u copyVer=%u hasCopy=%d "
        "sampleYFlip=%d",
        stage ? stage : "?", (unsigned)program, name ? name : "",
        (unsigned)binding, (unsigned)unit, (unsigned)tex, label ? label : "",
        decision_name ? decision_name : "?", decision, (unsigned)authority,
        (unsigned)rt_ver, (unsigned)copy_ver, has_copy, sample_yflip);
}

void mglBindingLogRTSampleCopy(const MGLBindingRTCopyLog *log)
{
    if (!log) {
        return;
    }
    switch (log->kind) {
    case MGL_RT_LOG_BIND:
        mglTraceLog(
            "RT_SAMPLE_COPY_BIND stage=%s program=%u name=%s binding=%u unit=%u tex=%u "
            "label=\"%s\" original=%p copy=%p",
            log->stage ? log->stage : "?", (unsigned)log->program,
            log->name ? log->name : "", (unsigned)log->binding, (unsigned)log->unit,
            (unsigned)log->tex, log->label ? log->label : "", log->original,
            log->copy);
        break;
    case MGL_RT_LOG_GATE_MISS:
        mglTraceLog(
            "RT_SAMPLE_COPY_GATE_MISS stage=%s program=%u name=%s binding=%u unit=%u "
            "tex=%u label=\"%s\" isRT=%d hasCopy=%d canUse=%d expectedType=%lu",
            log->stage ? log->stage : "?", (unsigned)log->program,
            log->name ? log->name : "", (unsigned)log->binding, (unsigned)log->unit,
            (unsigned)log->tex, log->label ? log->label : "", log->is_rt,
            log->has_copy, log->can_use, (unsigned long)log->expected_type);
        break;
    case MGL_RT_LOG_SKIP_YFLIP:
        mglTraceLog(
            "RT_SAMPLE_COPY_SKIP_EXISTING_YFLIP hit=%llu stage=%s program=%u name=%s "
            "binding=%u tex=%u decision=%s(%d)",
            (unsigned long long)log->hit, log->stage ? log->stage : "?",
            (unsigned)log->program, log->name ? log->name : "",
            (unsigned)log->binding, (unsigned)log->tex,
            log->decision_name ? log->decision_name : "?", log->decision);
        break;
    default:
        break;
    }
}

void mglBindingLogRTSampleCopySample(
    uint64_t hit, uint64_t bind_call, uint32_t program, uint32_t vs,
    uint32_t fs, const char *name, uint32_t binding, uint32_t unit,
    uint32_t rt_tex, const char *label, int fallback, int use_copy,
    const void *ptr, const void *mtl, const void *direct, const void *copy,
    uint64_t fmt, uint64_t type, uint64_t w, uint64_t h, uint32_t draw_fbo,
    uint32_t rp_fbo, const void *rp_color, const void *rp_depth)
{
    mglTraceLog(
        "RT_SAMPLE_COPY_SAMPLE hit=%llu bindCall=%llu program=%u vs=%u fs=%u "
        "name=%s binding=%u unit=%u rtTex=%u label=\"%s\" fallback=%d useCopy=%d "
        "ptr=%p mtl=%p direct=%p copy=%p fmt=%lu type=%lu size=%lux%lu drawFbo=%u "
        "rpFbo=%u rpColor=%p rpDepth=%p",
        (unsigned long long)hit, (unsigned long long)bind_call, (unsigned)program,
        (unsigned)vs, (unsigned)fs, name ? name : "", (unsigned)binding,
        (unsigned)unit, (unsigned)rt_tex, label ? label : "", fallback, use_copy,
        ptr, mtl, direct, copy, (unsigned long)fmt, (unsigned long)type,
        (unsigned long)w, (unsigned long)h, (unsigned)draw_fbo, (unsigned)rp_fbo,
        rp_color, rp_depth);
}

void mglBindingLogSamplerResolve(
    const char *stage_tag, uint32_t program, uint32_t binding, uint32_t unit,
    const char *source, uint32_t sampler_name, uint32_t min_filter,
    uint32_t mag_filter, uint32_t wrap_s, uint32_t wrap_t, double min_lod,
    double max_lod, uint32_t gl_tex, uint32_t base, uint32_t max_level,
    uint32_t tex_w, uint32_t tex_h, uint64_t bound_w, uint64_t bound_h,
    uint64_t bound_levels)
{
    mglTraceLogExternal(
        "%s_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=%s samplerName=%u "
        "minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f "
        "glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
        stage_tag ? stage_tag : "?", (unsigned)program, (unsigned)binding,
        (unsigned)unit, source ? source : "?", (unsigned)sampler_name,
        (unsigned)min_filter, (unsigned)mag_filter, (unsigned)wrap_s,
        (unsigned)wrap_t, min_lod, max_lod, (unsigned)gl_tex, (unsigned)base,
        (unsigned)max_level, (unsigned)tex_w, (unsigned)tex_h,
        (unsigned long)bound_w, (unsigned long)bound_h,
        (unsigned long)bound_levels);
}

void mglBindingLogMipDiagFrag(
    uint32_t unit, uint32_t binding, uint32_t program, uint32_t gl_tex,
    const char *source, uint32_t min_filter, uint32_t mag_filter, double min_lod,
    double max_lod, double aniso, uint32_t base, uint32_t max_level,
    uint32_t gl_levels, uint64_t mtl_levels, uint64_t mtl_w, uint64_t mtl_h,
    const void *mtl, int render_target, int via_copy, uint32_t copy_levels,
    uint32_t dirty_mips, uint32_t rt_ver, uint32_t copy_ver)
{
    fprintf(stderr, "MGL MIP_DIAG frag unit=%u binding=%u program=%u glTex=%u "
          "source=%s minFilter=0x%x magFilter=0x%x minLod=%.1f maxLod=%.1f aniso=%.1f "
          "base=%u max=%u glLevels=%u mtlLevels=%lu mtlW=%lu mtlH=%lu mtlTex=%p "
          "renderTarget=%d viaCopy=%d copyLevels=%u dirtyMips=0x%x rtVer=%u copyVer=%u",
          (unsigned)unit, (unsigned)binding, (unsigned)program, (unsigned)gl_tex,
          source ? source : "?", (unsigned)min_filter, (unsigned)mag_filter,
          min_lod, max_lod, aniso, (unsigned)base, (unsigned)max_level,
          (unsigned)gl_levels, (unsigned long)mtl_levels, (unsigned long)mtl_w,
          (unsigned long)mtl_h, mtl, render_target, via_copy,
          (unsigned)copy_levels, (unsigned)dirty_mips, (unsigned)rt_ver,
          (unsigned)copy_ver);
}

void mglBindingLogTexFallbackEx(uint64_t hit, uint32_t binding, uint32_t program,
                                uint32_t gl_tex, int suppressed, const char *name,
                                uint32_t unit)
{
    if (suppressed) {
        fprintf(stderr, "MGL TEX FALLBACK SUPPRESSED fragment sampled binding=%u program=%u name=%s glTex=%u unit=%u reason=insampler-current-target-no-copy hit=%llu",
              (unsigned)binding, (unsigned)program, name ? name : "",
              (unsigned)gl_tex, (unsigned)unit, (unsigned long long)hit);
    } else {
        fprintf(stderr, "MGL TEX FALLBACK fragment sampled binding=%u program=%u glTex=%u hit=%llu",
              (unsigned)binding, (unsigned)program, (unsigned)gl_tex,
              (unsigned long long)hit);
    }
}

void mglBindingLogDepthRecover(const MGLBindingDepthLog *log)
{
    if (!log || log->kind == MGL_DR_LOG_NONE) {
        return;
    }
    switch (log->kind) {
    case MGL_DR_LOG_HIST_SUPPRESSED:
        fprintf(stderr, "MGL INSAMPLER DEPTH HISTORY SCAN SUPPRESSED hit=%llu program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u pairedColor=%u currentDrawTarget=1",
              (unsigned long long)log->hit, (unsigned)log->program,
              (unsigned)log->binding, (unsigned)log->unit, (unsigned)log->fbo,
              (unsigned long)log->color_att, (unsigned)log->depth_tex,
              (unsigned)log->paired_color);
        break;
    case MGL_DR_LOG_NO_COPY:
        fprintf(stderr, "MGL INSAMPLER DEPTH CURRENT TARGET false COPY hit=%llu program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u colorTex=%u depthFmt=%lu sampledVersion=%u rtVersion=%u",
              (unsigned long long)log->hit, (unsigned)log->program,
              (unsigned)log->binding, (unsigned)log->unit, (unsigned)log->fbo,
              (unsigned long)log->color_att, (unsigned)log->depth_tex,
              (unsigned)log->color_tex, (unsigned long)log->depth_fmt,
              (unsigned)log->sampled_ver, (unsigned)log->rt_ver);
        break;
    case MGL_DR_LOG_PAIRED_DIRECT:
        fprintf(stderr, "MGL INSAMPLER DEPTH RECOVERY hit=%llu program=%u binding=%u unit=%u fbo=%u depthTex=%u colorTex=%u depthFmt=%lu colorFmt=%lu size=%lux%lu",
              (unsigned long long)log->hit, (unsigned)log->program,
              (unsigned)log->binding, (unsigned)log->unit, (unsigned)log->fbo,
              (unsigned)log->depth_tex, (unsigned)log->color_tex,
              (unsigned long)log->depth_fmt, (unsigned long)log->color_fmt,
              (unsigned long)log->w, (unsigned long)log->h);
        break;
    case MGL_DR_LOG_UNPAIRED:
        fprintf(stderr, "MGL INSAMPLER DEPTH UNPAIRED hit=%llu program=%u binding=%u unit=%u depthTex=%u fmt=%lu size=%lux%lu",
              (unsigned long long)log->hit, (unsigned)log->program,
              (unsigned)log->binding, (unsigned)log->unit,
              (unsigned)log->depth_tex, (unsigned long)log->depth_fmt,
              (unsigned long)log->w, (unsigned long)log->h);
        break;
    case MGL_DR_LOG_HISTORY_RECOVERY:
        fprintf(stderr, "MGL INSAMPLER DEPTH RECOVERY hit=%llu reason=%s program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u recoverTex=%u depthFmt=%lu recoverFmt=%lu size=%lux%lu copy=%d prevVersion=%d sampledVersion=%u rtVersion=%u pairedColor=%u pairedCurrent=%d",
              (unsigned long long)log->hit, log->reason ? log->reason : "none",
              (unsigned)log->program, (unsigned)log->binding, (unsigned)log->unit,
              (unsigned)log->fbo, (unsigned long)log->color_att,
              (unsigned)log->depth_tex, (unsigned)log->recover_tex,
              (unsigned long)log->depth_fmt, (unsigned long)log->recover_fmt,
              (unsigned long)log->w, (unsigned long)log->h, log->copy,
              log->prev_ver, (unsigned)log->sampled_ver, (unsigned)log->rt_ver,
              (unsigned)log->paired_color, log->paired_current);
        break;
    case MGL_DR_LOG_RT_SKIP:
        fprintf(stderr, "MGL SAMPLED DEPTH RT RECOVER SKIP current-draw-target hit=%llu program=%u name=%s binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u colorTex=%u",
              (unsigned long long)log->hit, (unsigned)log->program,
              log->name ? log->name : "", (unsigned)log->binding,
              (unsigned)log->unit, (unsigned)log->fbo,
              (unsigned long)log->color_att, (unsigned)log->depth_tex,
              (unsigned)log->color_tex);
        break;
    case MGL_DR_LOG_RT_SUPPRESS_LAST2D:
        fprintf(stderr, "MGL SAMPLED DEPTH RT RECOVER SUPPRESS last-sampled-2d hit=%llu program=%u name=%s binding=%u unit=%u depthTex=%u last2D=%u",
              (unsigned long long)log->hit, (unsigned)log->program,
              log->name ? log->name : "", (unsigned)log->binding,
              (unsigned)log->unit, (unsigned)log->depth_tex,
              (unsigned)log->last2d);
        break;
    case MGL_DR_LOG_RT_RECOVER:
        fprintf(stderr, "MGL SAMPLED DEPTH RT RECOVER hit=%llu reason=%s program=%u name=%s binding=%u unit=%u depthTex=%u recoverTex=%u fmt=%lu recoverFmt=%lu size=%lux%lu level=%p ever=%u init=%u unit(active=%u tex2D=%u last2D=%u) recoverFbo=%u currentFbo=%u colorTex=%u fboDepthTex=%u",
              (unsigned long long)log->hit, log->reason ? log->reason : "none",
              (unsigned)log->program, log->name ? log->name : "",
              (unsigned)log->binding, (unsigned)log->unit,
              (unsigned)log->depth_tex, (unsigned)log->recover_tex,
              (unsigned long)log->depth_fmt, (unsigned long)log->recover_fmt,
              (unsigned long)log->w, (unsigned long)log->h, log->level,
              (unsigned)log->ever, (unsigned)log->init,
              (unsigned)log->unit_active, (unsigned)log->unit_tex2d,
              (unsigned)log->unit_last2d, (unsigned)log->recover_fbo,
              (unsigned)log->current_fbo, (unsigned)log->color_tex,
              (unsigned)log->fbo_depth_tex);
        break;
    case MGL_DR_LOG_RT_FALLBACK:
        fprintf(stderr, "MGL SAMPLED DEPTH RT FALLBACK hit=%llu program=%u name=%s binding=%u unit=%u depthTex=%u fmt=%lu size=%lux%lu level=%p ever=%u init=%u unit(active=%u tex2D=%u last2D=%u)",
              (unsigned long long)log->hit, (unsigned)log->program,
              log->name ? log->name : "", (unsigned)log->binding,
              (unsigned)log->unit, (unsigned)log->depth_tex,
              (unsigned long)log->depth_fmt, (unsigned long)log->w,
              (unsigned long)log->h, log->level, (unsigned)log->ever,
              (unsigned)log->init, (unsigned)log->unit_active,
              (unsigned)log->unit_tex2d, (unsigned)log->unit_last2d);
        break;
    default:
        break;
    }
}
