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
 * O3.3 residual — sampled-texture / storage-image BindingState plans +
 * image-view helpers. Pure C; no Metal-cpp, no renderer instance.
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

#ifdef __cplusplus
}
#endif

#endif /* MGL_BINDING_TEXTURE_H */
