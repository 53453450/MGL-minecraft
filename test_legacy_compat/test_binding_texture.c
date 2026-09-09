/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_binding_texture + attrib plan (O3.3). No Metal required.
 */

#include "mgl_binding_stage.h"
#include "mgl_binding_texture.h"

#include <stdio.h>
#include <string.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static void test_image_helpers(void)
{
    expect(mglRenderImageTargetIsMultisample(0x9100) == 1, "ms 2d");
    expect(mglRenderImageLevelInRange(0, 4) == 1, "level ok");
    expect(mglRenderImageLevelInRange(4, 4) == 0, "level oor");
    expect(mglRenderImageBindPixelFormat(0, 10, 20) == 10u, "no ifmt");
    expect(mglRenderImageBindPixelFormat(1, 10, 20) == 20u, "mapped");
    uint32_t dst = 0;
    expect(mglRenderImageNeedsNonLayeredSlice(0, 0, 3 /*2DArray*/, &dst) == 1,
           "slice 2darray");
    expect(dst == 2u /*2D*/, "dst 2d");
    expect(mglRenderImageViewSliceCount(5 /*cube*/, 2) == 12u, "cube slices");
}

static void test_storage_plan(void)
{
    MGLStorageImageBindInput in;
    MGLStorageImageBindPlan plan;
    memset(&in, 0, sizeof(in));
    in.pass = MGL_SI_PASS_ENSURE;
    in.skip_resource = 1;
    expect(mglBindingTexturePlanStorageImage(&in, &plan) == 0, "plan ok");
    expect(plan.action == MGL_SI_ACTION_SKIP, "skip resource");

    memset(&in, 0, sizeof(in));
    in.pass = MGL_SI_PASS_ENSURE;
    in.has_resource = 1;
    in.resource_binding = 2;
    in.element = 1;
    in.use_resource_unit = 1;
    in.sampler_unit = 3;
    in.max_units = 32;
    expect(mglBindingTexturePlanStorageImage(&in, &plan) == 0, "ensure");
    expect(plan.action == MGL_SI_ACTION_ENSURE_TEX, "ensure tex");
    expect(plan.metal_slot == 3u, "metal slot binding+elem");
    expect(plan.gl_unit == 4u, "gl unit sampler+elem");

    in.pass = MGL_SI_PASS_BIND;
    expect(mglBindingTexturePlanStorageImage(&in, &plan) == 0, "bind");
    expect(plan.action == MGL_SI_ACTION_BIND_TEX, "bind tex");
}

static void test_sampled_plan(void)
{
    MGLSampledTextureBindInput in;
    MGLSampledTextureBindPlan plan;
    memset(&in, 0, sizeof(in));
    in.phase = MGL_ST_PHASE_GATE;
    in.spirv_binding = 100;
    in.gl_binding = 0;
    in.max_units = 32;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "gate");
    expect(plan.action == MGL_ST_ACTION_SKIP, "oor skip");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_ST_PHASE_COMPAT;
    in.has_mtl_texture = 1;
    in.mtl_type = 2;
    in.expected_type = 5;
    in.format_kind_ok = 1;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "compat");
    expect(plan.action == MGL_ST_ACTION_TYPE_FALLBACK, "type fb");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_ST_PHASE_RT;
    in.is_render_target = 1;
    in.yflip = 1; /* SAMPLED_COPY */
    in.has_sampled_copy = 1;
    in.copy_fresh = 1;
    in.can_use_rt_copy = 1;
    in.copy_type_ok = 1;
    in.copy_kind_ok = 1;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "rt");
    expect(plan.action == MGL_ST_ACTION_RT_USE_COPY, "use copy");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_ST_PHASE_FINAL;
    in.has_bound_texture = 1;
    in.has_resource = 1;
    in.has_combined_sampler = 1;
    in.has_sampler = 1;
    in.sampler_binding = 2;
    in.max_sampler_slots = 16;
    in.spirv_binding = 2;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "final");
    expect(plan.action == MGL_ST_ACTION_QUEUE, "queue");
    expect(plan.queue_texture == 1 && plan.queue_sampler == 1, "queue both");
}

static void test_attrib_plan(void)
{
    MGLAttribBindInput in;
    MGLAttribBindPlan plan;
    memset(&in, 0, sizeof(in));
    in.program_uses_attrib = 1;
    in.uses_current_value = 1;
    in.mapped_index = 3;
    in.max_metal_slots = 31;
    expect(mglBindingStagePlanAttribEntry(&in, &plan) == 0, "attr");
    expect(plan.action == MGL_ATTR_ACTION_CURRENT, "current");

    memset(&in, 0, sizeof(in));
    in.program_uses_attrib = 1;
    in.has_attrib_binding = 1;
    in.mapped_index = 1;
    in.max_metal_slots = 31;
    in.offsets_valid = 1;
    in.conversion_kind = 2;
    expect(mglBindingStagePlanAttribEntry(&in, &plan) == 0, "conv");
    expect(plan.action == MGL_ATTR_ACTION_CONVERT, "convert");

    memset(&in, 0, sizeof(in));
    in.program_uses_attrib = 1;
    in.has_attrib_binding = 1;
    in.mapped_index = 1;
    in.max_metal_slots = 31;
    in.offsets_valid = 1;
    in.phase = MGL_ATTR_PHASE_POST_MTL;
    in.has_mtl_data = 1;
    in.mtl_usable = 1;
    in.binding_offset = 16;
    in.metal_len = 1024;
    expect(mglBindingStagePlanAttribEntry(&in, &plan) == 0, "bind");
    expect(plan.action == MGL_ATTR_ACTION_BIND, "bind buf");
    expect(plan.metal_bind_offset == 0u, "relative mode @0");

    expect(mglRenderIntegerAttribDstIsInt(0x8B53) == 1, "ivec2");
    expect(mglRenderVertexMetalBindOffset(1, 64) == 64u, "absolute");
}

static void test_warmup_gates(void)
{
    uint32_t mask[4] = {0x2u, 0, 0, 0};
    expect(mglBindingTextureSamplerWarmupSlotActive(mask, 1) == 1, "bit1");
    expect(mglBindingTextureSamplerMaskEmpty(mask) == 0, "not empty");
    expect(mglBindingTextureSeparateSamplerInRange(1, 2, 32) == 1, "sep ok");
    expect(mglBindingTextureShouldBindCombinedSampler(1, 1, 3, 16) == 1,
           "combined");
}

static void test_depth_recover_plan(void)
{
    expect(mglBindingTextureSampledNameIsInSampler("InSampler") == 1, "in name");
    expect(mglBindingTextureSampledNameIsInSampler("Diffuse") == 0, "not in");
    uint64_t ctr = 0;
    expect(mglBindingTextureDepthRecoverLogHit(&ctr) == 1, "log first");
    expect(ctr == 1ull, "ctr1");

    MGLDepthRecoverInput in;
    MGLDepthRecoverPlan plan;
    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_GATE;
    in.has_texture = 1;
    in.is_depth_or_stencil = 1;
    in.is_insampler = 1;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "gate in");
    expect(plan.action == MGL_DR_ACTION_ENTER_INSAMPLER, "enter in");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_GATE;
    in.has_texture = 1;
    in.is_depth_or_stencil = 1;
    in.is_render_target = 1;
    in.level0_ever_written = 0;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "gate rt");
    expect(plan.action == MGL_DR_ACTION_ENTER_RT, "enter rt");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_INSAMPLER;
    in.paired_is_current_draw = 1;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "in cur");
    expect(plan.action == MGL_DR_ACTION_PROBE_PAIRED_COPY, "probe copy");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_COPY;
    in.paired_copy_usable = 0;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "no copy");
    expect(plan.action == MGL_DR_ACTION_NIL_SUPPRESS, "nil suppress");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_INSAMPLER;
    in.has_paired_color = 1;
    in.has_paired_mtl = 1;
    in.paired_is_depth_or_stencil = 0;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "paired");
    expect(plan.action == MGL_DR_ACTION_USE_PAIRED_DIRECT, "paired direct");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_HISTORY;
    in.candidate_valid = 1;
    in.candidate_is_rt = 1;
    in.candidate_copy_usable = 1;
    in.candidate_is_current_draw = 1;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "hist");
    expect(plan.action == MGL_DR_ACTION_HISTORY_USE_COPY, "hist copy");
    expect(plan.reason_tag && strcmp(plan.reason_tag, "history-current-copy") == 0,
           "hist tag");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_HISTORY;
    in.candidate_valid = 1;
    in.candidate_has_mtl = 1;
    in.candidate_type_ok = 1;
    in.candidate_kind_ok = 1;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "hist dir");
    expect(plan.action == MGL_DR_ACTION_HISTORY_USE_DIRECT, "hist direct");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_DR_PHASE_RT;
    in.rt_sub = 0;
    in.has_paired_color = 1;
    in.has_paired_mtl = 1;
    in.candidate_type_ok = 1;
    in.candidate_kind_ok = 1;
    expect(mglBindingTexturePlanDepthRecover(&in, &plan) == 0, "rt paired");
    expect(plan.action == MGL_DR_ACTION_RT_USE_PAIRED, "rt use paired");
    expect(plan.reason_tag && strcmp(plan.reason_tag, "paired-color") == 0,
           "rt tag");
}


static void test_sampler_materialize_plan(void)
{
    MGLSamplerMaterializeInput in;
    MGLSamplerMaterializePlan plan;
    memset(&in, 0, sizeof(in));
    in.force_default = 1;
    expect(mglBindingTexturePlanSamplerMaterialize(&in, &plan) == 0, "sm force");
    expect(plan.action == MGL_SM_ACTION_USE_DEFAULT, "sm default");

    memset(&in, 0, sizeof(in));
    in.unit_in_range = 1;
    in.has_gl_sampler = 1;
    in.gl_sampler_dirty = 1;
    in.has_gl_sampler_mtl = 1;
    expect(mglBindingTexturePlanSamplerMaterialize(&in, &plan) == 0, "sm gl");
    expect(plan.action == MGL_SM_ACTION_USE_GL_SAMPLER, "sm gl act");
    expect(plan.recreate_gl_sampler_mtl == 1, "sm recreate");

    memset(&in, 0, sizeof(in));
    in.require_tex_params_mtl = 1;
    in.has_tex_params_mtl = 1;
    expect(mglBindingTexturePlanSamplerMaterialize(&in, &plan) == 0, "sm tex");
    expect(plan.action == MGL_SM_ACTION_USE_TEX_PARAMS, "sm tex act");

    memset(&in, 0, sizeof(in));
    in.require_tex_params_mtl = 0; /* fragment style */
    expect(mglBindingTexturePlanSamplerMaterialize(&in, &plan) == 0, "sm frag");
    expect(plan.action == MGL_SM_ACTION_USE_TEX_PARAMS, "sm frag tex");
}

static void test_sampled_diag_and_rt_ports(void)
{
    MGLSampledTextureBindInput in;
    MGLSampledTextureBindPlan plan;
    memset(&in, 0, sizeof(in));
    in.phase = MGL_ST_PHASE_RT;
    in.is_render_target = 1;
    in.yflip = 0; /* ORIGINAL */
    in.want_base_level_on_original = 1;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "rt orig");
    expect(plan.action == MGL_ST_ACTION_RT_ORIGINAL, "rt orig act");
    expect(plan.apply_base_level_view == 1, "base view");

    memset(&in, 0, sizeof(in));
    in.phase = MGL_ST_PHASE_RT;
    in.is_render_target = 1;
    in.yflip = 1;
    in.has_sampled_copy = 1;
    in.copy_fresh = 1;
    in.can_use_rt_copy = 1;
    in.copy_type_ok = 1;
    in.copy_kind_ok = 1;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "rt copy");
    expect(plan.apply_base_level_view == 1, "copy view");

    MGLSampledDiagGateInput din;
    MGLSampledDiagGatePlan dplan;
    memset(&din, 0, sizeof(din));
    din.stage_is_fragment = 1;
    din.used_fallback = 1;
    expect(mglBindingTexturePlanSampledDiag(&din, &dplan) == 0, "diag");
    expect(dplan.log_detail == 1, "diag detail");

    uint64_t ctr = 0;
    expect(mglBindingTextureRateLogHit(&ctr, 1ull, 10ull) == 1, "rate1");
    expect(mglBindingTextureMipDiagMix(1ull, 2ull) != 0ull, "mix");
}


static void test_sampled_final_helpers(void)
{
    expect(mglBindingTextureForceDefaultSampler(1, 1) == 1, "force depth fb");
    expect(mglBindingTextureForceDefaultSampler(1, 0) == 0, "no force color");
    expect(mglBindingTextureForceDefaultSampler(0, 1) == 0, "no force clean");

    MGLSampledTextureBindInput in;
    memset(&in, 0, sizeof(in));
    in.spirv_binding = 3u;
    in.has_resource = 1;
    mglBindingTextureFillSampledFinalInput(&in, 1, 0, 0, 1, 5u, 16u, 1, 0);
    expect(in.phase == MGL_ST_PHASE_FINAL, "final phase");
    expect(in.has_bound_texture == 1, "final bound");
    expect(in.sampler_binding == 5u, "final samp slot");
    expect(in.max_sampler_slots == 16u, "final max");
    expect(in.has_combined_sampler == 1, "final combined");

    MGLSampledTextureBindPlan plan;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "final plan");
    expect(plan.action == MGL_ST_ACTION_QUEUE, "final queue");
    expect(plan.queue_texture == 1, "final q tex");
    expect(plan.queue_sampler == 1, "final q samp");

    mglBindingTextureFillSampledFinalInput(&in, 0, 1, 0, 0, 0u, 16u, 0, 0);
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "suppress plan");
    expect(plan.action == MGL_ST_ACTION_SUPPRESS_FALLBACK, "suppress act");
}


static void test_sampled_gate_compat_fill(void)
{
    MGLSampledTextureBindInput in;
    mglBindingTextureFillSampledGateInput(&in, 3u, 1u, 32u, 0, 1);
    expect(in.phase == MGL_ST_PHASE_GATE, "gate phase");
    expect(in.spirv_binding == 3u && in.has_resource == 1, "gate fields");
    MGLSampledTextureBindPlan plan;
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "gate plan");
    expect(plan.action == MGL_ST_ACTION_PROCEED, "gate proceed");

    mglBindingTextureFillSampledCompatInput(&in, 1, 2u, 2u, 1);
    expect(in.phase == MGL_ST_PHASE_COMPAT, "compat phase");
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "compat plan");
    expect(plan.action == MGL_ST_ACTION_PROCEED, "compat ok");

    mglBindingTextureFillSampledCompatInput(&in, 1, 1u, 2u, 1);
    expect(mglBindingTexturePlanSampled(&in, &plan) == 0, "compat type");
    expect(plan.action == MGL_ST_ACTION_TYPE_FALLBACK, "type fb");
}

static void test_apply_masks(void)
{
    MGLSamplerWarmupPlan warm;
    uint32_t vmask[4] = {0x3u, 0, 0, 0};
    uint32_t fmask[4] = {0x4u, 0, 0, 0};
    mglBindingTexturePlanSamplerWarmup(1, 1, 1, vmask, fmask, 32, 16, &warm);
    expect(warm.mode == MGL_SW_MODE_MASK, "warm mask mode");
    expect(warm.mask[0] == 0x7u, "warm or mask");
    expect(warm.warmup_count == 16u, "warm count capped");
    expect(mglBindingTextureSamplerWarmupSlotActive(warm.mask, 0) == 1, "slot0");
    expect(mglBindingTextureSamplerWarmupSlotActive(warm.mask, 3) == 0, "slot3");

    mglBindingTexturePlanSamplerWarmup(1, 0, 0, NULL, NULL, 32, 16, &warm);
    expect(warm.mode == MGL_SW_MODE_ALL, "warm all no prog");

    mglBindingTexturePlanSamplerWarmup(0, 1, 1, vmask, fmask, 32, 16, &warm);
    expect(warm.mode == MGL_SW_MODE_NONE, "warm none");

    expect(mglBindingTextureSampledMarkKind(1, 0) == MGL_ST_MARK_BOUND, "mark bound");
    expect(mglBindingTextureSampledMarkKind(1, 1) == MGL_ST_MARK_FALLBACK, "mark fb");
    expect(mglBindingTextureSampledMarkKind(0, 0) == MGL_ST_MARK_NIL, "mark nil");

    uint8_t present[8] = {1, 0, 1, 1, 0, 0, 0, 1};
    expect(mglBindingStageBuildPresentMask(present, 8) == 0x8Du, "present mask");
    expect(mglBindingStageCountPresent(present, 8) == 4u, "present count");
}


/* Stubs for log ports referenced by EmitSampledDiag / EmitMipDiag (texture_log.m). */
void mglBindingLogTBINDFocused(const char *a, uint32_t b, const char *c, uint32_t d,
    uint32_t e, uint32_t f, uint32_t g, const void *h, uint64_t i, uint64_t j,
    uint64_t k, uint32_t l, uint32_t m, uint32_t n, uint32_t o, uint32_t p)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p; }
void mglBindingLogTBINDTraceFile(const char *a, uint32_t b, const char *c, uint32_t d,
    uint32_t e, int f, int g, uint32_t h, uint32_t i, int j, uint64_t k, uint64_t l,
    int m, uint32_t n, uint32_t o, uint32_t p, uint32_t q, const void *r, uint64_t s,
    uint64_t t, uint64_t u, uint32_t v, uint32_t w, uint32_t x, uint32_t y, uint32_t z)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p;(void)q;(void)r;(void)s;(void)t;(void)u;(void)v;(void)w;(void)x;(void)y;(void)z; }
void mglBindingLogSampleDetail(uint64_t a, uint64_t b, const char *c, uint32_t d,
    const char *e, uint32_t f, uint32_t g, uint64_t h, int i, uint32_t j, const void *k,
    uint32_t l, int m, const void *n, uint64_t o, uint64_t p, uint64_t q, uint32_t r,
    uint32_t s, uint32_t t, uint32_t u, uint32_t v, uint32_t w, uint32_t x, uint64_t y,
    uint32_t z, uint32_t aa, uint32_t ab, uint32_t ac, uint64_t ad, const void *ae,
    uint64_t af, uint64_t ag)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p;(void)q;(void)r;(void)s;(void)t;(void)u;(void)v;(void)w;(void)x;(void)y;(void)z;(void)aa;(void)ab;(void)ac;(void)ad;(void)ae;(void)af;(void)ag; }
void mglBindingLogTexBufferBind(uint64_t a, uint32_t b, uint32_t c, uint32_t d,
    uint32_t e, uint32_t f, uint32_t g, uint64_t h, uint64_t i, const void *j,
    uint64_t k, uint64_t l, uint64_t m, uint64_t n, const void *o)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o; }
void mglBindingLogRTSampleCopySample(uint64_t a, uint64_t b, uint32_t c, uint32_t d,
    uint32_t e, const char *f, uint32_t g, uint32_t h, uint32_t i, const char *j,
    int k, int l, const void *m, const void *n, const void *o, const void *p,
    uint64_t q, uint64_t r, uint64_t s, uint64_t t, uint32_t u, uint32_t v,
    const void *w, const void *x)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p;(void)q;(void)r;(void)s;(void)t;(void)u;(void)v;(void)w;(void)x; }
void mglBindingLogMipDiagFrag(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
    const char *e, uint32_t f, uint32_t g, double h, double i, double j, uint32_t k,
    uint32_t l, uint32_t m, uint64_t n, uint64_t o, uint64_t p, const void *q, int r,
    int s, uint32_t t, uint32_t u, uint32_t v, uint32_t w)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p;(void)q;(void)r;(void)s;(void)t;(void)u;(void)v;(void)w; }

static void test_o33_fill_emit_ports(void)
{
    MGLDepthRecoverInput din;
    MGLDepthRecoverPlan dplan;
    mglBindingTextureFillDepthRecoverGateInput(&din, 1, 1, 1, 0, 1, 1);
    expect(din.phase == MGL_DR_PHASE_GATE && din.is_insampler == 1, "dr gate fill");
    expect(mglBindingTexturePlanDepthRecover(&din, &dplan) == 0, "dr gate plan");
    expect(dplan.action == MGL_DR_ACTION_ENTER_INSAMPLER, "dr enter in");

    mglBindingTextureFillDepthRecoverRTInput(&din, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0);
    expect(din.phase == MGL_DR_PHASE_RT && din.rt_sub == 0, "dr rt0 fill");
    expect(mglBindingTexturePlanDepthRecover(&din, &dplan) == 0, "dr rt0 plan");
    expect(dplan.action == MGL_DR_ACTION_RT_USE_PAIRED, "dr rt paired");

    mglBindingTextureFillDepthRecoverRTInput(&din, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
    expect(mglBindingTexturePlanDepthRecover(&din, &dplan) == 0, "dr rt1");
    expect(dplan.action == MGL_DR_ACTION_RT_APPLY, "dr rt apply");

    mglBindingTextureFillDepthRecoverRTInput(&din, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
    expect(mglBindingTexturePlanDepthRecover(&din, &dplan) == 0, "dr rt2");
    expect(dplan.action == MGL_DR_ACTION_USE_RECOVER, "dr rt recover");

    MGLSamplerMaterializeInput sm;
    mglBindingTextureFillSamplerMaterializeInput(&sm, 0, 1, 1, 0, 1, 0, 0);
    expect(sm.has_gl_sampler == 1 && sm.unit_in_range == 1, "sm fill");

    MGLSampledTextureBindInput rt;
    mglBindingTextureFillSampledRTInput(&rt, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    expect(rt.phase == MGL_ST_PHASE_RT && rt.copy_fresh == 1, "rt fill");

    MGLStorageImageBindInput si;
    mglBindingTextureFillStorageImageInput(&si, MGL_SI_PASS_BIND, 0, 1, 2, 1, 9,
                                           1, 0, 0, 3, 4, 5, 32);
    expect(si.pass == MGL_SI_PASS_BIND && si.element == 1u, "si fill");

    uint8_t frag[128];
    memset(frag, 0xab, sizeof(frag));
    mglBindingTextureWriteFragTrace(frag, 7, 3, 2, 11, 100, 200, (void *)1,
                                    (void *)2, (void *)3, (void *)4, 64, 32, 9, 2,
                                    1, 0);
    expect(frag[0] != 0xab, "frag trace wrote");

    MGLSampledDiagEmitInput ein;
    memset(&ein, 0, sizeof(ein));
    mglBindingTextureFillSampledDiagEmitCore(
        &ein, "fragment", 1, 2, 3, "InSampler", 0, 1, -1, 0, 13, 0xde1, 1, 2, 2,
        0, 0, 0, 0, 0, 2, 16, 16, 0, 4, 4, 1, 16, 1, 1, 0, 0, 0, 0, 0, 13, 1, 0, 0,
        0, 0, 0, 1, 0, 1, 10, 0, 0, 0, 0);
    expect(ein.stage_is_fragment == 1 && ein.gl_tex == 13u, "diag core");
    MGLSampledDiagEmitResult eres;
    mglBindingTextureEmitSampledDiagPorts(&ein, &eres);
    expect(eres.want_readback == 0 || eres.want_readback == 1, "diag emit");

    uint64_t mip_state = 0;
    expect(mglBindingTextureEmitMipDiagFragIfChanged(
               &mip_state, 0xabcull, 0, 0, 1, 2, "tex", 1, 1, 0.0, 1.0, 1.0, 0, 0,
               1, 1, 16, 16, NULL, 0, 0, 0, 0, 0, 0) == 1,
           "mip first");
    expect(mglBindingTextureEmitMipDiagFragIfChanged(
               &mip_state, 0xabcull, 0, 0, 1, 2, "tex", 1, 1, 0.0, 1.0, 1.0, 0, 0,
               1, 1, 16, 16, NULL, 0, 0, 0, 0, 0, 0) == 0,
           "mip same");
}

int main(void)
{
    test_image_helpers();
    test_storage_plan();
    test_sampled_plan();
    test_attrib_plan();
    test_warmup_gates();
    test_depth_recover_plan();
    test_sampler_materialize_plan();
    test_sampled_diag_and_rt_ports();
    test_sampled_final_helpers();
    test_sampled_gate_compat_fill();
    test_apply_masks();
    test_o33_fill_emit_ports();
    if (g_fails) {
        fprintf(stderr, "%d failure(s)\n", g_fails);
        return 1;
    }
    puts("test_binding_texture: ok");
    return 0;
}
