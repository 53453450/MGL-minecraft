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
 * O3.3 residual — sampled-texture / storage-image / depth-recover /
 * Y-flip RT / sampler-materialize / apply-masks BindingState plans +
 * image-view helpers. Pure C plans; optional ObjC log helpers in
 * mgl_binding_texture_log.m (freeze/shrink only — never grow). No Metal-cpp,
 * no renderer instance.
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
    /* Fragment ORIGINAL path wants base-level view on live RT. */
    int want_base_level_on_original;
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
    /* RT apply ports (ObjC thin set*). */
    int apply_base_level_view; /* viewForBaseLevel on chosen MTL texture */
    int force_default_sampler; /* FINAL: depth-fallback → default sampler */
} MGLSampledTextureBindPlan;

int mglBindingTexturePlanSampled(const MGLSampledTextureBindInput *in,
                                 MGLSampledTextureBindPlan *out);

/* FINAL apply helpers (V/F shared sampled spine). */
int mglBindingTextureForceDefaultSampler(int used_fallback,
                                         int expected_kind_is_depth);
void mglBindingTextureFillSampledFinalInput(
    MGLSampledTextureBindInput *in, int has_bound_texture, int suppress_missing,
    int used_type_fallback, int has_combined_sampler, uint32_t sampler_binding,
    uint32_t max_sampler_slots, int has_sampler, int force_default_sampler);


/* Sampler warmup: 1 if slot bit set in 128-bit mask (4×uint32). */
int mglBindingTextureSamplerWarmupSlotActive(const uint32_t mask[4],
                                             uint32_t slot);
int mglBindingTextureSamplerMaskEmpty(const uint32_t mask[4]);

/* Warmup apply-mask: OR V/F sampled_texture_unit_mask → mode + count. */
enum {
    MGL_SW_MODE_NONE = 0, /* no default sampler to warm */
    MGL_SW_MODE_MASK = 1, /* warm only slots set in mask */
    MGL_SW_MODE_ALL = 2   /* no program or empty mask → warm all */
};

typedef struct MGLSamplerWarmupPlan {
    uint32_t mode; /* MGL_SW_MODE_* */
    uint32_t warmup_count;
    uint32_t mask[4];
} MGLSamplerWarmupPlan;

/* vertex_mask/fragment_mask may be NULL (treated as empty). */
void mglBindingTexturePlanSamplerWarmup(
    int has_default_sampler, int has_vertex_program, int has_fragment_program,
    const uint32_t vertex_mask[4], const uint32_t fragment_mask[4],
    uint32_t max_units, uint32_t max_sampler_slots, MGLSamplerWarmupPlan *out);

/* Sampled bind mark apply-mask (FINAL counters). */
enum {
    MGL_ST_MARK_NONE = 0,
    MGL_ST_MARK_BOUND = 1,
    MGL_ST_MARK_FALLBACK = 2,
    MGL_ST_MARK_NIL = 3
};
uint32_t mglBindingTextureSampledMarkKind(int has_texture, int used_fallback);

/* Separate-sampler / array-element gates. */
int mglBindingTextureSeparateSamplerInRange(uint32_t spirv, uint32_t gl,
                                            uint32_t max_units);
int mglBindingTextureArrayElementSlotOk(uint32_t metal_slot, uint32_t max_units);
int mglBindingTextureShouldBindCombinedSampler(int has_combined, int has_sampler,
                                               uint32_t sampler_slot,
                                               uint32_t max_sampler_slots);

/* ---- Sampler materialize plan (V/F shared) ---- */

enum {
    MGL_SM_ACTION_KEEP = 0,          /* leave sampler as caller default/nil */
    MGL_SM_ACTION_USE_GL_SAMPLER,    /* unit-bound Sampler object */
    MGL_SM_ACTION_USE_TEX_PARAMS,    /* TextureParameter.mtl_data */
    MGL_SM_ACTION_USE_DEFAULT        /* force renderer default sampler */
};

enum {
    MGL_SM_REASON_OK = 0,
    MGL_SM_REASON_FORCE_DEFAULT,
    MGL_SM_REASON_GL_SAMPLER,
    MGL_SM_REASON_TEX_PARAMS,
    MGL_SM_REASON_KEEP
};

typedef struct MGLSamplerMaterializeInput {
    int force_default;           /* depth type/kind fallback */
    int unit_in_range;
    int has_gl_sampler;
    int gl_sampler_dirty;
    int has_gl_sampler_mtl;
    int has_tex_params_mtl;
    /* Vertex requires tex-params MTL; fragment assigns even if NULL. */
    int require_tex_params_mtl;
} MGLSamplerMaterializeInput;

typedef struct MGLSamplerMaterializePlan {
    uint32_t action; /* MGL_SM_ACTION_* */
    uint32_t reason; /* MGL_SM_REASON_* */
    int recreate_gl_sampler_mtl; /* release dirty + createMTLSampler */
    int clear_gl_sampler_dirty;
    /* Static tag for traces; never heap. */
    const char *source_tag;
} MGLSamplerMaterializePlan;

int mglBindingTexturePlanSamplerMaterialize(const MGLSamplerMaterializeInput *in,
                                            MGLSamplerMaterializePlan *out);

/* ---- Sampled bind diagnostic gates (TBIND / sample-detail) ---- */

enum {
    MGL_SD_ACTION_SKIP = 0,
    MGL_SD_ACTION_LOG_DETAIL,
    MGL_SD_ACTION_LOG_GUI_RT,   /* fragment atlas RT path */
    MGL_SD_ACTION_LOG_READBACK  /* suspicious level readback */
};

typedef struct MGLSampledDiagGateInput {
    int stage_is_fragment;
    int used_fallback;
    int is_gui_rt_copy_eligible;
    int focused_loading_window; /* fragment focus program + bindCall window */
    int vertex_focus_program;   /* historical program==34 */
    int level0_suspicious_zero;
    int level0_never_written;
    int level0_uninit;
    int has_bound_texture;
    int is_texel_buffer;
} MGLSampledDiagGateInput;

typedef struct MGLSampledDiagGatePlan {
    uint32_t action; /* primary */
    int log_detail;
    int log_gui_rt;
    int log_readback;
    int log_texel_buffer;
} MGLSampledDiagGatePlan;

int mglBindingTexturePlanSampledDiag(const MGLSampledDiagGateInput *in,
                                     MGLSampledDiagGatePlan *out);


/* Generic rate-limit helper (early N hits, then every period). */
int mglBindingTextureRateLogHit(uint64_t *counter, uint64_t early,
                                uint64_t period);

/* FNV-offset mix for MIP_DIAG sampler-state change detection. */
uint64_t mglBindingTextureMipDiagMix(uint64_t sig, uint64_t value);
uint64_t mglBindingTextureMipDiagSignature(
    uint32_t tex_name, uint32_t min_filter, uint32_t mag_filter,
    uint32_t base_level, uint32_t max_level, uint64_t mtl_levels,
    uint64_t mtl_ptr_bits, int via_copy, uint32_t sampled_levels,
    uint32_t dirty_mip_mask, int version_mismatch);


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

/* Rate-limited NSLog helpers (implemented in mgl_binding_texture_log.m).
 * Extend this shell only — do not spawn another log TU. */

void mglBindingLogTBINDFocused(
    const char *stage, uint32_t program, const char *resource,
    uint32_t metal_slot, uint32_t sampler_unit, uint32_t gl_tex, uint32_t target,
    const void *mtl, uint64_t mtl_type, uint64_t w, uint64_t h,
    uint32_t l0w, uint32_t l0h, uint32_t ever, uint32_t init, uint32_t source);

void mglBindingLogTBINDTraceFile(
    const char *stage, uint32_t program, const char *resource,
    uint32_t metal_slot, uint32_t sampler_unit, int res_unit, int explicit_unit,
    uint32_t gl_tex, uint32_t target, int fallback, uint64_t expected_type,
    uint64_t lookup_type, int expected_index, uint32_t unit_active,
    uint32_t unit_expected, uint32_t unit_2d, uint32_t unit_cube,
    const void *mtl, uint64_t mtl_type, uint64_t w, uint64_t h, uint32_t l0w,
    uint32_t l0h, uint32_t ever, uint32_t init, uint32_t source);

void mglBindingLogSampleDetail(
    uint64_t bind_call, uint64_t hit, const char *stage, uint32_t program,
    const char *name, uint32_t binding, uint32_t unit, uint64_t expected_type,
    int expected_index, uint32_t ptr_tex, const void *ptr, uint32_t target,
    int fallback, const void *mtl, uint64_t mtl_type, uint64_t mtl_w,
    uint64_t mtl_h, uint32_t unit_active, uint32_t unit_expected,
    uint32_t unit_2d, uint32_t unit_cube, uint32_t l0w, uint32_t l0h,
    uint32_t l0d, uint64_t bytes, uint32_t ever, uint32_t full, uint32_t zero,
    uint32_t source, uint64_t upload, const void *src, uint64_t hash,
    uint64_t data_hash);

void mglBindingLogTexCompatMismatch(
    const char *kind, const char *stage, uint32_t binding, uint32_t program,
    uint32_t gl_tex, uint64_t mtl_type, uint64_t expected, uint64_t hit);
void mglBindingLogTexBufferBind(
    uint64_t hit, uint32_t program, uint32_t binding, uint32_t unit,
    uint32_t ptr_tex, uint32_t active, uint32_t buffer_slot,
    uint64_t expected_type, uint64_t lookup_type, const void *mtl,
    uint64_t mtl_type, uint64_t w, uint64_t h, uint64_t format,
    const void *sampler);
void mglBindingLogRTYFlipDecision(
    const char *stage, uint32_t program, const char *name, uint32_t binding,
    uint32_t unit, uint32_t tex, const char *label, const char *decision_name,
    int decision, uint32_t authority, uint32_t rt_ver, uint32_t copy_ver,
    int has_copy, int sample_yflip);
void mglBindingLogRTSampleCopySample(
    uint64_t hit, uint64_t bind_call, uint32_t program, uint32_t vs,
    uint32_t fs, const char *name, uint32_t binding, uint32_t unit,
    uint32_t rt_tex, const char *label, int fallback, int use_copy,
    const void *ptr, const void *mtl, const void *direct, const void *copy,
    uint64_t fmt, uint64_t type, uint64_t w, uint64_t h, uint32_t draw_fbo,
    uint32_t rp_fbo, const void *rp_color, const void *rp_depth);

void mglBindingLogSamplerResolve(
    const char *stage_tag, uint32_t program, uint32_t binding, uint32_t unit,
    const char *source, uint32_t sampler_name, uint32_t min_filter,
    uint32_t mag_filter, uint32_t wrap_s, uint32_t wrap_t, double min_lod,
    double max_lod, uint32_t gl_tex, uint32_t base, uint32_t max_level,
    uint32_t tex_w, uint32_t tex_h, uint64_t bound_w, uint64_t bound_h,
    uint64_t bound_levels);
void mglBindingLogMipDiagFrag(
    uint32_t unit, uint32_t binding, uint32_t program, uint32_t gl_tex,
    const char *source, uint32_t min_filter, uint32_t mag_filter, double min_lod,
    double max_lod, double aniso, uint32_t base, uint32_t max_level,
    uint32_t gl_levels, uint64_t mtl_levels, uint64_t mtl_w, uint64_t mtl_h,
    const void *mtl, int render_target, int via_copy, uint32_t copy_levels,
    uint32_t dirty_mips, uint32_t rt_ver, uint32_t copy_ver);


/* Depth-recover / RT-sample-copy / fallback log ports (merged — shrink shell).
 * Prefer these over the old per-message wrappers (removed). */

enum {
    MGL_DR_LOG_NONE = 0,
    MGL_DR_LOG_HIST_SUPPRESSED,
    MGL_DR_LOG_NO_COPY,
    MGL_DR_LOG_PAIRED_DIRECT,
    MGL_DR_LOG_UNPAIRED,
    MGL_DR_LOG_HISTORY_RECOVERY,
    MGL_DR_LOG_RT_SKIP,
    MGL_DR_LOG_RT_SUPPRESS_LAST2D,
    MGL_DR_LOG_RT_RECOVER,
    MGL_DR_LOG_RT_FALLBACK
};

typedef struct MGLBindingDepthLog {
    uint32_t kind; /* MGL_DR_LOG_* */
    uint64_t hit;
    uint32_t program;
    uint32_t binding;
    uint32_t unit;
    uint32_t fbo;
    uint64_t color_att;
    uint32_t depth_tex;
    uint32_t color_tex;
    uint32_t recover_tex;
    uint32_t paired_color;
    uint32_t last2d;
    uint64_t depth_fmt;
    uint64_t color_fmt;
    uint64_t recover_fmt;
    uint64_t w;
    uint64_t h;
    uint32_t sampled_ver;
    uint32_t rt_ver;
    int copy;
    int prev_ver;
    int paired_current;
    const char *reason;
    const char *name;
    const void *level;
    uint32_t ever;
    uint32_t init;
    uint32_t unit_active;
    uint32_t unit_tex2d;
    uint32_t unit_last2d;
    uint32_t recover_fbo;
    uint32_t current_fbo;
    uint32_t fbo_depth_tex;
} MGLBindingDepthLog;

void mglBindingLogDepthRecover(const MGLBindingDepthLog *log);

enum {
    MGL_RT_LOG_BIND = 1,
    MGL_RT_LOG_GATE_MISS = 2,
    MGL_RT_LOG_SKIP_YFLIP = 3
};

typedef struct MGLBindingRTCopyLog {
    uint32_t kind; /* MGL_RT_LOG_* */
    uint64_t hit;
    const char *stage;
    uint32_t program;
    const char *name;
    uint32_t binding;
    uint32_t unit;
    uint32_t tex;
    const char *label;
    const void *original;
    const void *copy;
    int is_rt;
    int has_copy;
    int can_use;
    uint64_t expected_type;
    const char *decision_name;
    int decision;
} MGLBindingRTCopyLog;

void mglBindingLogRTSampleCopy(const MGLBindingRTCopyLog *log);

/* Fallback: suppressed!=0 → SUPPRESSED path. */
void mglBindingLogTexFallbackEx(uint64_t hit, uint32_t binding, uint32_t program,
                                uint32_t gl_tex, int suppressed, const char *name,
                                uint32_t unit);


/* O3.3: ObjC BindingState emit wrappers for merged depth/RT log PODs. */
#ifndef MGL_EMIT_DR_LOG
#define MGL_EMIT_DR_LOG(...) \
    do { \
        MGLBindingDepthLog _mgl_dr_log = {__VA_ARGS__}; \
        mglBindingLogDepthRecover(&_mgl_dr_log); \
    } while (0)
#endif
#ifndef MGL_EMIT_RT_LOG
#define MGL_EMIT_RT_LOG(...) \
    do { \
        MGLBindingRTCopyLog _mgl_rt_log = {__VA_ARGS__}; \
        mglBindingLogRTSampleCopy(&_mgl_rt_log); \
    } while (0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* MGL_BINDING_TEXTURE_H */
