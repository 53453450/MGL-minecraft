/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_binding_texture.h
 *
 * O3.3 residual — sampled-texture / storage-image / depth-recover BindingState
 * plans + image-view helpers. Pure C plans; optional ObjC log helpers in
 * mgl_binding_texture_log.m. No Metal-cpp, no renderer instance.
 * Texture-type integers match mgl_render_values.h (MGLTextureType*).
 * Y-flip decision integers match mgl_coordinate.h (MGL_YFLIP_*).
 *
 * ObjC +BindingState stays plan@C + thin set*Texture / queue ports.
 * Do not sink into mgl_render.cpp. Do not grow +Binding.m.
 */

#ifndef MGL_BINDING_TEXTURE_H
#define MGL_BINDING_TEXTURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Image-view helpers (ex-mgl_render.cpp) ---- */

uint32_t mglRenderImageBindPixelFormat(uint32_t internalformat,
                                       uint32_t native_format,
                                       uint32_t mapped_bind_format);
int mglRenderImageTargetIsMultisample(uint32_t gl_target);
int mglRenderImageLevelInRange(uint32_t level, uint32_t mipmap_count);
int mglRenderImageNeedsNonLayeredSlice(int layered, int is_ms, uint32_t src_type,
                                       uint32_t *dst_type_out);
int mglRenderImageNeedsFormatOrMipView(uint32_t level, uint32_t bind_format,
                                       uint32_t native_format);
uint64_t mglRenderImageViewSliceCount(uint32_t src_type, uint64_t array_length);

/* ---- Storage-image slot/unit plan ---- */

enum {
    MGL_SI_PASS_ENSURE = 0, /* pass-1: ensure MTLTexture for gl unit */
    MGL_SI_PASS_BIND = 1    /* pass-2: queue encoder texture slot */
};

enum {
    MGL_SI_ACTION_SKIP = 0,
    MGL_SI_ACTION_ENSURE_TEX,
    MGL_SI_ACTION_BIND_TEX
};

typedef struct MGLStorageImageBindInput {
    int pass; /* MGL_SI_PASS_* */
    int skip_resource;
    int has_resource;
    uint32_t resource_binding;
    uint32_t element;
    uint32_t fallback_metal_slot;
    int use_resource_unit; /* explicit_by_slot || resource */
    int explicit_by_slot;
    uint32_t explicit_unit;
    int32_t sampler_unit;
    uint32_t resource_gl_binding;
    uint32_t fallback_gl_binding;
    uint32_t max_units;
    int has_tex;
} MGLStorageImageBindInput;

typedef struct MGLStorageImageBindPlan {
    uint32_t action; /* MGL_SI_ACTION_* */
    uint32_t metal_slot;
    uint32_t gl_unit;
} MGLStorageImageBindPlan;

int mglBindingTexturePlanStorageImage(const MGLStorageImageBindInput *in,
                                      MGLStorageImageBindPlan *out);

/* ---- Sampled-texture bind plan (vertex/fragment shared spine) ---- */

enum {
    MGL_ST_PHASE_GATE = 0,   /* OOR / skip before materialize */
    MGL_ST_PHASE_COMPAT = 1, /* type / data-kind after MTL texture */
    MGL_ST_PHASE_RT = 2,     /* render-target Y-flip path */
    MGL_ST_PHASE_FINAL = 3   /* nil fallback + queue texture/sampler */
};

enum {
    MGL_ST_ACTION_SKIP = 0,
    MGL_ST_ACTION_PROCEED,          /* materialize / continue phase */
    MGL_ST_ACTION_TYPE_FALLBACK,    /* replace with expected-type fallback */
    MGL_ST_ACTION_KIND_FALLBACK,    /* replace with expected-kind fallback */
    MGL_ST_ACTION_RT_USE_COPY,      /* bind fresh sampled copy */
    MGL_ST_ACTION_RT_REPAIR,        /* ObjC rebuilds copy; may retry */
    MGL_ST_ACTION_RT_RETRY,         /* encoder rotated; caller returns false */
    MGL_ST_ACTION_RT_ORIGINAL,      /* keep original (or inject path) */
    MGL_ST_ACTION_RT_GATE_MISS,     /* copy wanted but unavailable */
    MGL_ST_ACTION_NIL_FALLBACK,     /* missing texture → fallback */
    MGL_ST_ACTION_SUPPRESS_FALLBACK,/* keep nil (InSampler depth gate) */
    MGL_ST_ACTION_QUEUE             /* queue texture (+ optional sampler) */
};

enum {
    MGL_ST_REASON_OK = 0,
    MGL_ST_REASON_OOR,
    MGL_ST_REASON_SKIP_RESOURCE,
    MGL_ST_REASON_TYPE_MISMATCH,
    MGL_ST_REASON_KIND_MISMATCH,
    MGL_ST_REASON_RT_COPY,
    MGL_ST_REASON_RT_REPAIR,
    MGL_ST_REASON_RT_RETRY,
    MGL_ST_REASON_RT_ORIGINAL,
    MGL_ST_REASON_RT_GATE,
    MGL_ST_REASON_NIL,
    MGL_ST_REASON_SUPPRESS,
    MGL_ST_REASON_QUEUE
};

typedef struct MGLSampledTextureBindInput {
    int phase; /* MGL_ST_PHASE_* */
    uint32_t spirv_binding;
    uint32_t gl_binding;
    uint32_t max_units;
    int skip_resource;
    int has_resource; /* shader resource resolved */
    /* COMPAT */
    int has_mtl_texture;
    uint32_t mtl_type;
    uint32_t expected_type;
    int format_kind_ok; /* pixel format compatible with expected kind */
    /* RT */
    int used_type_fallback;
    int is_render_target;
    int yflip; /* MGL_YFLIP_* */
    int has_sampled_copy;
    int copy_fresh;
    int can_use_rt_copy;
    int copy_type_ok;
    int copy_kind_ok;
    int repaired_available;
    int repaired_fresh; /* after repair: content fresh → RETRY else USE_COPY */
    /* FINAL */
    int has_bound_texture;
    int suppress_missing_fallback;
    int has_combined_sampler;
    uint32_t sampler_binding;
    uint32_t max_sampler_slots;
    int has_sampler;
    int force_default_sampler; /* depth fallback → default sampler */
} MGLSampledTextureBindInput;

typedef struct MGLSampledTextureBindPlan {
    uint32_t action; /* MGL_ST_ACTION_* */
    uint32_t reason; /* MGL_ST_REASON_* */
    int queue_texture;
    int queue_sampler;
    uint32_t texture_slot;
    uint32_t sampler_slot;
    int mark_bound;
    int mark_fallback;
    int mark_nil;
} MGLSampledTextureBindPlan;

int mglBindingTexturePlanSampled(const MGLSampledTextureBindInput *in,
                                 MGLSampledTextureBindPlan *out);

/* Sampler warmup: 1 if slot bit set in 128-bit mask (4×uint32). */
int mglBindingTextureSamplerWarmupSlotActive(const uint32_t mask[4],
                                             uint32_t slot);
int mglBindingTextureSamplerMaskEmpty(const uint32_t mask[4]);

/* Separate-sampler / array-element gates. */
int mglBindingTextureSeparateSamplerInRange(uint32_t spirv, uint32_t gl,
                                            uint32_t max_units);
int mglBindingTextureArrayElementSlotOk(uint32_t metal_slot, uint32_t max_units);
int mglBindingTextureShouldBindCombinedSampler(int has_combined, int has_sampler,
                                               uint32_t sampler_slot,
                                               uint32_t max_sampler_slots);

/* ---- Depth-recover plan (InSampler + uninitialized depth RT) ---- */

enum {
    MGL_DR_PHASE_GATE = 0,     /* enter InSampler / RT / keep */
    MGL_DR_PHASE_INSAMPLER = 1,/* after paired-color facts */
    MGL_DR_PHASE_COPY = 2,     /* after paired sampled-copy probe */
    MGL_DR_PHASE_HISTORY = 3,  /* one recent_sampled_2d candidate */
    MGL_DR_PHASE_RT = 4        /* non-InSampler depth RT recover */
};

enum {
    MGL_DR_ACTION_KEEP = 0,
    MGL_DR_ACTION_ENTER_INSAMPLER,
    MGL_DR_ACTION_ENTER_RT,
    MGL_DR_ACTION_PROBE_PAIRED_COPY, /* ObjC probes paired GL sampled copy */
    MGL_DR_ACTION_USE_RECOVER,       /* apply recover ptr/mtl set by caller */
    MGL_DR_ACTION_NIL_SUPPRESS,      /* texture=nil + suppress missing fb */
    MGL_DR_ACTION_USE_PAIRED_DIRECT, /* ptr=pairedColor, texture=pairedMTL */
    MGL_DR_ACTION_SCAN_HISTORY,
    MGL_DR_ACTION_HISTORY_PROBE,     /* bind candidate then re-enter HISTORY */
    MGL_DR_ACTION_HISTORY_USE_COPY,
    MGL_DR_ACTION_HISTORY_USE_DIRECT,
    MGL_DR_ACTION_HISTORY_CONTINUE,
    MGL_DR_ACTION_LOG_UNPAIRED,
    MGL_DR_ACTION_RT_USE_PAIRED,     /* recoverTexture = pairedColor */
    MGL_DR_ACTION_RT_SKIP_CURRENT,   /* paired is current draw target */
    MGL_DR_ACTION_RT_CONTINUE,       /* try last2D / apply / fallback */
    MGL_DR_ACTION_RT_SUPPRESS_LAST2D,
    MGL_DR_ACTION_RT_APPLY,          /* bind recoverTexture + check MTL */
    MGL_DR_ACTION_RT_FALLBACK        /* type/kind fallback sampled tex */
};

enum {
    MGL_DR_REASON_OK = 0,
    MGL_DR_REASON_NOT_DEPTH,
    MGL_DR_REASON_PAIRED_CURRENT,
    MGL_DR_REASON_PAIRED_COPY,
    MGL_DR_REASON_PAIRED_NO_COPY,
    MGL_DR_REASON_PAIRED_DIRECT,
    MGL_DR_REASON_HISTORY,
    MGL_DR_REASON_UNPAIRED,
    MGL_DR_REASON_RT_UNINIT,
    MGL_DR_REASON_RT_PAIRED,
    MGL_DR_REASON_RT_CURRENT,
    MGL_DR_REASON_RT_LAST2D,
    MGL_DR_REASON_RT_FALLBACK,
    MGL_DR_REASON_KEEP
};

typedef struct MGLDepthRecoverInput {
    int phase; /* MGL_DR_PHASE_* */
    /* GATE */
    int has_texture;
    int is_insampler;
    int is_depth_or_stencil;
    int is_render_target;
    int level0_ever_written;
    int level0_has_init;
    /* INSAMPLER / COPY */
    int has_paired_color;
    int paired_is_current_draw;
    int has_paired_mtl;
    int paired_is_depth_or_stencil;
    int paired_copy_usable;
    int unit_in_range;
    /* HISTORY candidate */
    int candidate_valid;
    int candidate_is_rt;
    int candidate_is_current_draw;
    int candidate_needs_bind;
    int candidate_has_mtl;
    int candidate_copy_usable;
    int candidate_is_depth_or_stencil;
    int candidate_type_ok;
    int candidate_kind_ok;
    /* RT */
    int has_recover;
    int recover_mtl_ok;
    int still_depth_or_stencil;
    int last2d_recoverable;
    int rt_sub; /* 0=after paired, 1=after last2d, 2=after apply check */
} MGLDepthRecoverInput;

typedef struct MGLDepthRecoverPlan {
    uint32_t action; /* MGL_DR_ACTION_* */
    uint32_t reason; /* MGL_DR_REASON_* */
    /* Static tag for logs; never heap. May be NULL. */
    const char *reason_tag;
} MGLDepthRecoverPlan;

int mglBindingTextureSampledNameIsInSampler(const char *sampled_name);
int mglBindingTextureDepthRecoverLogHit(uint64_t *counter);
int mglBindingTexturePlanDepthRecover(const MGLDepthRecoverInput *in,
                                      MGLDepthRecoverPlan *out);

/* Rate-limited NSLog helpers (implemented in mgl_binding_texture_log.m). */
void mglBindingLogInSamplerDepthHistorySuppressed(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t fbo, uint64_t color_att, uint32_t depth_tex, uint32_t paired_color);
void mglBindingLogInSamplerDepthNoCopy(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t fbo, uint64_t color_att, uint32_t depth_tex, uint32_t color_tex,
    uint64_t depth_fmt, uint32_t sampled_ver, uint32_t rt_ver);
void mglBindingLogInSamplerDepthPairedDirect(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t fbo, uint32_t depth_tex, uint32_t color_tex, uint64_t depth_fmt,
    uint64_t color_fmt, uint64_t w, uint64_t h);
void mglBindingLogInSamplerDepthUnpaired(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t depth_tex, uint64_t fmt, uint64_t w, uint64_t h);
void mglBindingLogInSamplerDepthHistoryRecovery(
    uint64_t hit, const char *reason, uint32_t program, uint32_t binding,
    uint32_t unit, uint32_t fbo, uint64_t color_att, uint32_t depth_tex,
    uint32_t recover_tex, uint64_t depth_fmt, uint64_t recover_fmt,
    uint64_t w, uint64_t h, int copy, int prev_ver, uint32_t sampled_ver,
    uint32_t rt_ver, uint32_t paired_color, int paired_current);
void mglBindingLogSampledDepthRtSkip(
    uint64_t hit, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t fbo, uint64_t color_att, uint32_t depth_tex,
    uint32_t color_tex);
void mglBindingLogSampledDepthRtSuppressLast2D(
    uint64_t hit, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t depth_tex, uint32_t last2d);
void mglBindingLogSampledDepthRtRecover(
    uint64_t hit, const char *reason, uint32_t program, const char *name,
    uint32_t binding, uint32_t unit, uint32_t depth_tex, uint32_t recover_tex,
    uint64_t fmt, uint64_t recover_fmt, uint64_t w, uint64_t h,
    const void *level, uint32_t ever, uint32_t init, uint32_t unit_active,
    uint32_t unit_tex2d, uint32_t unit_last2d, uint32_t recover_fbo,
    uint32_t current_fbo, uint32_t color_tex, uint32_t fbo_depth_tex);
void mglBindingLogSampledDepthRtFallback(
    uint64_t hit, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t depth_tex, uint64_t fmt, uint64_t w, uint64_t h,
    const void *level, uint32_t ever, uint32_t init, uint32_t unit_active,
    uint32_t unit_tex2d, uint32_t unit_last2d);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BINDING_TEXTURE_H */
