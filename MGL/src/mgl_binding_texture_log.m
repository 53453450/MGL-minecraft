/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Rate-limited depth-recover NSLog ports for O3.3 BindingState.
 * Keeps +BindingState.m = plan@C + thin apply; do not grow +Binding.m.
 */

#import <Foundation/Foundation.h>
#include "mgl_binding_texture.h"

void mglBindingLogInSamplerDepthHistorySuppressed(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t fbo, uint64_t color_att, uint32_t depth_tex, uint32_t paired_color)
{
    NSLog(@"MGL INSAMPLER DEPTH HISTORY SCAN SUPPRESSED hit=%llu program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u pairedColor=%u currentDrawTarget=1",
          (unsigned long long)hit, (unsigned)program, (unsigned)binding, (unsigned)unit,
          (unsigned)fbo, (unsigned long)color_att, (unsigned)depth_tex, (unsigned)paired_color);
}

void mglBindingLogInSamplerDepthNoCopy(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t fbo, uint64_t color_att, uint32_t depth_tex, uint32_t color_tex,
    uint64_t depth_fmt, uint32_t sampled_ver, uint32_t rt_ver)
{
    NSLog(@"MGL INSAMPLER DEPTH CURRENT TARGET NO COPY hit=%llu program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u colorTex=%u depthFmt=%lu sampledVersion=%u rtVersion=%u",
          (unsigned long long)hit, (unsigned)program, (unsigned)binding, (unsigned)unit,
          (unsigned)fbo, (unsigned long)color_att, (unsigned)depth_tex, (unsigned)color_tex,
          (unsigned long)depth_fmt, (unsigned)sampled_ver, (unsigned)rt_ver);
}

void mglBindingLogInSamplerDepthPairedDirect(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t fbo, uint32_t depth_tex, uint32_t color_tex, uint64_t depth_fmt,
    uint64_t color_fmt, uint64_t w, uint64_t h)
{
    NSLog(@"MGL INSAMPLER DEPTH RECOVERY hit=%llu program=%u binding=%u unit=%u fbo=%u depthTex=%u colorTex=%u depthFmt=%lu colorFmt=%lu size=%lux%lu",
          (unsigned long long)hit, (unsigned)program, (unsigned)binding, (unsigned)unit,
          (unsigned)fbo, (unsigned)depth_tex, (unsigned)color_tex,
          (unsigned long)depth_fmt, (unsigned long)color_fmt,
          (unsigned long)w, (unsigned long)h);
}

void mglBindingLogInSamplerDepthUnpaired(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t depth_tex, uint64_t fmt, uint64_t w, uint64_t h)
{
    NSLog(@"MGL INSAMPLER DEPTH UNPAIRED hit=%llu program=%u binding=%u unit=%u depthTex=%u fmt=%lu size=%lux%lu",
          (unsigned long long)hit, (unsigned)program, (unsigned)binding, (unsigned)unit,
          (unsigned)depth_tex, (unsigned long)fmt, (unsigned long)w, (unsigned long)h);
}

void mglBindingLogInSamplerDepthHistoryRecovery(
    uint64_t hit, const char *reason, uint32_t program, uint32_t binding,
    uint32_t unit, uint32_t fbo, uint64_t color_att, uint32_t depth_tex,
    uint32_t recover_tex, uint64_t depth_fmt, uint64_t recover_fmt,
    uint64_t w, uint64_t h, int copy, int prev_ver, uint32_t sampled_ver,
    uint32_t rt_ver, uint32_t paired_color, int paired_current)
{
    NSLog(@"MGL INSAMPLER DEPTH RECOVERY hit=%llu reason=%s program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u recoverTex=%u depthFmt=%lu recoverFmt=%lu size=%lux%lu copy=%d prevVersion=%d sampledVersion=%u rtVersion=%u pairedColor=%u pairedCurrent=%d",
          (unsigned long long)hit, reason ? reason : "none", (unsigned)program,
          (unsigned)binding, (unsigned)unit, (unsigned)fbo, (unsigned long)color_att,
          (unsigned)depth_tex, (unsigned)recover_tex, (unsigned long)depth_fmt,
          (unsigned long)recover_fmt, (unsigned long)w, (unsigned long)h, copy, prev_ver,
          (unsigned)sampled_ver, (unsigned)rt_ver, (unsigned)paired_color, paired_current);
}

void mglBindingLogSampledDepthRtSkip(
    uint64_t hit, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t fbo, uint64_t color_att, uint32_t depth_tex,
    uint32_t color_tex)
{
    NSLog(@"MGL SAMPLED DEPTH RT RECOVER SKIP current-draw-target hit=%llu program=%u name=%s binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u colorTex=%u",
          (unsigned long long)hit, (unsigned)program, name ? name : "", (unsigned)binding,
          (unsigned)unit, (unsigned)fbo, (unsigned long)color_att, (unsigned)depth_tex,
          (unsigned)color_tex);
}

void mglBindingLogSampledDepthRtSuppressLast2D(
    uint64_t hit, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t depth_tex, uint32_t last2d)
{
    NSLog(@"MGL SAMPLED DEPTH RT RECOVER SUPPRESS last-sampled-2d hit=%llu program=%u name=%s binding=%u unit=%u depthTex=%u last2D=%u",
          (unsigned long long)hit, (unsigned)program, name ? name : "", (unsigned)binding,
          (unsigned)unit, (unsigned)depth_tex, (unsigned)last2d);
}

void mglBindingLogSampledDepthRtRecover(
    uint64_t hit, const char *reason, uint32_t program, const char *name,
    uint32_t binding, uint32_t unit, uint32_t depth_tex, uint32_t recover_tex,
    uint64_t fmt, uint64_t recover_fmt, uint64_t w, uint64_t h,
    const void *level, uint32_t ever, uint32_t init, uint32_t unit_active,
    uint32_t unit_tex2d, uint32_t unit_last2d, uint32_t recover_fbo,
    uint32_t current_fbo, uint32_t color_tex, uint32_t fbo_depth_tex)
{
    NSLog(@"MGL SAMPLED DEPTH RT RECOVER hit=%llu reason=%s program=%u name=%s binding=%u unit=%u depthTex=%u recoverTex=%u fmt=%lu recoverFmt=%lu size=%lux%lu level=%p ever=%u init=%u unit(active=%u tex2D=%u last2D=%u) recoverFbo=%u currentFbo=%u colorTex=%u fboDepthTex=%u",
          (unsigned long long)hit, reason ? reason : "none", (unsigned)program,
          name ? name : "", (unsigned)binding, (unsigned)unit, (unsigned)depth_tex,
          (unsigned)recover_tex, (unsigned long)fmt, (unsigned long)recover_fmt,
          (unsigned long)w, (unsigned long)h, level, (unsigned)ever, (unsigned)init,
          (unsigned)unit_active, (unsigned)unit_tex2d, (unsigned)unit_last2d,
          (unsigned)recover_fbo, (unsigned)current_fbo, (unsigned)color_tex,
          (unsigned)fbo_depth_tex);
}

void mglBindingLogSampledDepthRtFallback(
    uint64_t hit, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t depth_tex, uint64_t fmt, uint64_t w, uint64_t h,
    const void *level, uint32_t ever, uint32_t init, uint32_t unit_active,
    uint32_t unit_tex2d, uint32_t unit_last2d)
{
    NSLog(@"MGL SAMPLED DEPTH RT FALLBACK hit=%llu program=%u name=%s binding=%u unit=%u depthTex=%u fmt=%lu size=%lux%lu level=%p ever=%u init=%u unit(active=%u tex2D=%u last2D=%u)",
          (unsigned long long)hit, (unsigned)program, name ? name : "", (unsigned)binding,
          (unsigned)unit, (unsigned)depth_tex, (unsigned long)fmt, (unsigned long)w,
          (unsigned long)h, level, (unsigned)ever, (unsigned)init, (unsigned)unit_active,
          (unsigned)unit_tex2d, (unsigned)unit_last2d);
}
