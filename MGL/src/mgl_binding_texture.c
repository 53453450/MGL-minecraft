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
 * mgl_binding_texture.c — O3.3 residual sampled/storage /
 * Y-flip RT / sampler-materialize plans.
 * Pure C; do not grow +Binding.m / mgl_render.cpp.
 */

#include "mgl_binding_texture.h"
#include "mgl_binding_policy.h"
#include "mgl_render_values.h"

#include <string.h>

/* GL enum ABI (avoid GL headers). */
enum {
    MGL_BT_GL_TEXTURE_2D_MULTISAMPLE = 0x9100,
    MGL_BT_GL_TEXTURE_2D_MULTISAMPLE_ARRAY = 0x9102
};

/* mgl_coordinate.h MGL_YFLIP_* */
enum {
    MGL_BT_YFLIP_ORIGINAL = 0,
    MGL_BT_YFLIP_SAMPLED_COPY = 1,
    MGL_BT_YFLIP_ORIGINAL_AND_INJECT = 2
};

uint32_t mglRenderImageBindPixelFormat(uint32_t internalformat,
                                       uint32_t native_format,
                                       uint32_t mapped_bind_format) {
    if (internalformat == 0u) {
        return native_format;
    }
    if (mapped_bind_format == 0u) {
        return native_format;
    }
    return mapped_bind_format;
}

int mglRenderImageTargetIsMultisample(uint32_t gl_target) {
    return gl_target == MGL_BT_GL_TEXTURE_2D_MULTISAMPLE ||
                   gl_target == MGL_BT_GL_TEXTURE_2D_MULTISAMPLE_ARRAY
               ? 1
               : 0;
}

int mglRenderImageLevelInRange(uint32_t level, uint32_t mipmap_count) {
    return level < mipmap_count ? 1 : 0;
}

int mglRenderImageNeedsNonLayeredSlice(int layered, int is_ms, uint32_t src_type,
                                       uint32_t *dst_type_out) {
    if (layered || is_ms) {
        return 0;
    }
    uint32_t dst = 0u;
    if (src_type == (uint32_t)MGLTextureType2DArray ||
        src_type == (uint32_t)MGLTextureType3D ||
        src_type == (uint32_t)MGLTextureTypeCube ||
        src_type == (uint32_t)MGLTextureTypeCubeArray) {
        dst = (uint32_t)MGLTextureType2D;
    } else if (src_type == (uint32_t)MGLTextureType1DArray) {
        dst = (uint32_t)MGLTextureType1D;
    } else {
        return 0;
    }
    if (dst_type_out) {
        *dst_type_out = dst;
    }
    return 1;
}

int mglRenderImageNeedsFormatOrMipView(uint32_t level, uint32_t bind_format,
                                       uint32_t native_format) {
    return level > 0u || bind_format != native_format ? 1 : 0;
}

uint64_t mglRenderImageViewSliceCount(uint32_t src_type, uint64_t array_length) {
    uint64_t slices = array_length;
    if (src_type == (uint32_t)MGLTextureTypeCube ||
        src_type == (uint32_t)MGLTextureTypeCubeArray) {
        slices = array_length * 6u;
    }
    return slices < 1u ? 1u : slices;
}

int mglBindingTexturePlanStorageImage(const MGLStorageImageBindInput *in,
                                      MGLStorageImageBindPlan *out) {
    if (!in || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (in->skip_resource) {
        out->action = MGL_SI_ACTION_SKIP;
        return 0;
    }
    out->metal_slot = mglRenderResourceMetalSlot(
        in->has_resource, in->resource_binding, in->element,
        in->fallback_metal_slot);
    if (in->use_resource_unit) {
        out->gl_unit = mglRenderImageUnitFromResource(
            in->explicit_by_slot, in->explicit_unit, in->sampler_unit,
            in->resource_gl_binding, in->element);
    } else {
        out->gl_unit = in->fallback_gl_binding;
    }

    if (in->pass == MGL_SI_PASS_ENSURE) {
        if (!mglRenderImageUnitsInRange(0u, out->gl_unit, in->max_units)) {
            out->action = MGL_SI_ACTION_SKIP;
            return 0;
        }
        /* ObjC no-ops ENSURE when image unit has no texture. */
        out->action = MGL_SI_ACTION_ENSURE_TEX;
        return 0;
    }

    /* BIND pass */
    if (!mglRenderImageUnitsInRange(out->metal_slot, out->gl_unit,
                                    in->max_units)) {
        out->action = MGL_SI_ACTION_SKIP;
        return 0;
    }
    out->action = MGL_SI_ACTION_BIND_TEX;
    return 0;
}

static void mglBindingTextureSampledClear(MGLSampledTextureBindPlan *out) {
    memset(out, 0, sizeof(*out));
}

int mglBindingTexturePlanSampled(const MGLSampledTextureBindInput *in,
                                 MGLSampledTextureBindPlan *out) {
    if (!in || !out) {
        return -1;
    }
    mglBindingTextureSampledClear(out);
    out->texture_slot = in->program_binding;
    out->sampler_slot = in->sampler_binding;

    if (in->phase == MGL_ST_PHASE_GATE) {
        if (mglRenderMetalBindingPastUnits(in->program_binding, in->max_units) ||
            mglRenderMetalBindingPastUnits(in->gl_binding, in->max_units)) {
            out->action = MGL_ST_ACTION_SKIP;
            out->reason = MGL_ST_REASON_OOR;
            return 0;
        }
        if (in->skip_resource) {
            out->action = MGL_ST_ACTION_SKIP;
            out->reason = MGL_ST_REASON_SKIP_RESOURCE;
            return 0;
        }
        out->action = MGL_ST_ACTION_PROCEED;
        out->reason = MGL_ST_REASON_OK;
        return 0;
    }

    if (in->phase == MGL_ST_PHASE_COMPAT) {
        if (in->has_mtl_texture && in->expected_type != 0u &&
            in->mtl_type != in->expected_type) {
            out->action = MGL_ST_ACTION_TYPE_FALLBACK;
            out->reason = MGL_ST_REASON_TYPE_MISMATCH;
            return 0;
        }
        if (in->has_mtl_texture && !in->format_kind_ok) {
            out->action = MGL_ST_ACTION_KIND_FALLBACK;
            out->reason = MGL_ST_REASON_KIND_MISMATCH;
            return 0;
        }
        out->action = MGL_ST_ACTION_PROCEED;
        out->reason = MGL_ST_REASON_OK;
        return 0;
    }

    if (in->phase == MGL_ST_PHASE_RT) {
        if (in->used_type_fallback || !in->is_render_target) {
            out->action = MGL_ST_ACTION_PROCEED;
            out->reason = MGL_ST_REASON_OK;
            return 0;
        }
        if (in->yflip != MGL_BT_YFLIP_SAMPLED_COPY) {
            out->action = MGL_ST_ACTION_RT_ORIGINAL;
            out->reason = MGL_ST_REASON_RT_ORIGINAL;
            out->apply_base_level_view = in->want_base_level_on_original ? 1 : 0;
            return 0;
        }
        /* Prefer an already-fresh usable copy. */
        if (in->has_sampled_copy && in->copy_fresh && in->can_use_rt_copy &&
            in->copy_type_ok && in->copy_kind_ok) {
            out->action = MGL_ST_ACTION_RT_USE_COPY;
            out->reason = MGL_ST_REASON_RT_COPY;
            out->apply_base_level_view = 1;
            return 0;
        }
        /* Repair attempt results (ObjC filled after freshGLSampled*). */
        if (in->repaired_available) {
            if (!in->repaired_fresh) {
                out->action = MGL_ST_ACTION_RT_USE_COPY;
                out->reason = MGL_ST_REASON_RT_REPAIR;
                out->apply_base_level_view = 1;
                return 0;
            }
            out->action = MGL_ST_ACTION_RT_RETRY;
            out->reason = MGL_ST_REASON_RT_RETRY;
            return 0;
        }
        if (in->can_use_rt_copy) {
            out->action = MGL_ST_ACTION_RT_REPAIR;
            out->reason = MGL_ST_REASON_RT_REPAIR;
            return 0;
        }
        out->action = MGL_ST_ACTION_RT_GATE_MISS;
        out->reason = MGL_ST_REASON_RT_GATE;
        return 0;
    }

    /* FINAL */
    if (!in->has_bound_texture) {
        out->action = MGL_ST_ACTION_NIL_FALLBACK;
        out->reason = MGL_ST_REASON_NIL;
        out->mark_fallback = 1;
        /* After ObjC applies fallback, caller re-enters FINAL with texture. */
        return 0;
    }

    out->action = MGL_ST_ACTION_QUEUE;
    out->reason = MGL_ST_REASON_QUEUE;
    out->queue_texture = 1;
    out->mark_bound = 1;
    if (in->used_type_fallback) {
        out->mark_fallback = 1;
    }
    if (in->force_default_sampler) {
        out->force_default_sampler = 1;
    }
    /* Historical: (!resource || has_combined) && sampler && slot < max */
    if (in->has_sampler && in->sampler_binding < in->max_sampler_slots &&
        (!in->has_resource || in->has_combined_sampler)) {
        out->queue_sampler = 1;
    }
    return 0;
}


int mglBindingTextureForceDefaultSampler(int used_fallback,
                                         int expected_kind_is_depth) {
    return (used_fallback && expected_kind_is_depth) ? 1 : 0;
}

void mglBindingTextureFillSampledFinalInput(
    MGLSampledTextureBindInput *in, int has_bound_texture, int used_type_fallback,
    int has_combined_sampler, uint32_t sampler_binding,
    uint32_t max_sampler_slots, int has_sampler, int force_default_sampler) {
    if (!in) {
        return;
    }
    in->phase = MGL_ST_PHASE_FINAL;
    in->has_bound_texture = has_bound_texture ? 1 : 0;
    in->used_type_fallback = used_type_fallback ? 1 : 0;
    in->has_combined_sampler = has_combined_sampler ? 1 : 0;
    in->sampler_binding = sampler_binding;
    in->max_sampler_slots = max_sampler_slots;
    in->has_sampler = has_sampler ? 1 : 0;
    in->force_default_sampler = force_default_sampler ? 1 : 0;
}


void mglBindingTextureFillSampledGateInput(
    MGLSampledTextureBindInput *in, uint32_t program_binding, uint32_t gl_binding,
    uint32_t max_units, int skip_resource, int has_resource)
{
    if (!in) {
        return;
    }
    memset(in, 0, sizeof(*in));
    in->phase = MGL_ST_PHASE_GATE;
    in->program_binding = program_binding;
    in->gl_binding = gl_binding;
    in->max_units = max_units;
    in->skip_resource = skip_resource ? 1 : 0;
    in->has_resource = has_resource ? 1 : 0;
}

void mglBindingTextureFillSampledCompatInput(
    MGLSampledTextureBindInput *in, int has_mtl_texture, uint32_t mtl_type,
    uint32_t expected_type, int format_kind_ok)
{
    if (!in) {
        return;
    }
    memset(in, 0, sizeof(*in));
    in->phase = MGL_ST_PHASE_COMPAT;
    in->has_mtl_texture = has_mtl_texture ? 1 : 0;
    in->mtl_type = mtl_type;
    in->expected_type = expected_type;
    in->format_kind_ok = format_kind_ok ? 1 : 0;
}

int mglBindingTextureSamplerWarmupSlotActive(const uint32_t mask[4],
                                             uint32_t slot) {
    if (!mask) {
        return 0;
    }
    return (mask[slot >> 5] & (1u << (slot & 31u))) != 0u ? 1 : 0;
}

int mglBindingTextureSamplerMaskEmpty(const uint32_t mask[4]) {
    if (!mask) {
        return 1;
    }
    return (mask[0] | mask[1] | mask[2] | mask[3]) == 0u ? 1 : 0;
}

void mglBindingTexturePlanSamplerWarmup(
    int has_default_sampler, int has_vertex_program, int has_fragment_program,
    const uint32_t vertex_mask[4], const uint32_t fragment_mask[4],
    uint32_t max_units, uint32_t max_sampler_slots, MGLSamplerWarmupPlan *out)
{
    uint32_t i;
    if (!out) {
        return;
    }
    out->mode = MGL_SW_MODE_NONE;
    out->warmup_count = 0u;
    out->mask[0] = out->mask[1] = out->mask[2] = out->mask[3] = 0u;
    if (!has_default_sampler) {
        return;
    }
    if (vertex_mask) {
        for (i = 0; i < 4u; i++) {
            out->mask[i] |= vertex_mask[i];
        }
    }
    if (fragment_mask) {
        for (i = 0; i < 4u; i++) {
            out->mask[i] |= fragment_mask[i];
        }
    }
    out->warmup_count = max_units;
    if (out->warmup_count > max_sampler_slots) {
        out->warmup_count = max_sampler_slots;
    }
    {
        int has_prog = has_vertex_program || has_fragment_program;
        int empty = mglBindingTextureSamplerMaskEmpty(out->mask);
        if (has_prog && !empty) {
            out->mode = MGL_SW_MODE_MASK;
        } else {
            out->mode = MGL_SW_MODE_ALL;
        }
    }
}

uint32_t mglBindingTextureSampledMarkKind(int has_texture, int used_fallback)
{
    if (has_texture && !used_fallback) {
        return MGL_ST_MARK_BOUND;
    }
    if (used_fallback) {
        return MGL_ST_MARK_FALLBACK;
    }
    if (!has_texture) {
        return MGL_ST_MARK_NIL;
    }
    return MGL_ST_MARK_NIL;
}

int mglBindingTextureSeparateSamplerInRange(uint32_t spirv, uint32_t gl,
                                            uint32_t max_units) {
    return !mglRenderMetalBindingPastUnits(spirv, max_units) &&
                   !mglRenderMetalBindingPastUnits(gl, max_units)
               ? 1
               : 0;
}

int mglBindingTextureArrayElementSlotOk(uint32_t metal_slot, uint32_t max_units) {
    return !mglRenderMetalBindingPastUnits(metal_slot, max_units) ? 1 : 0;
}

int mglBindingTextureShouldBindCombinedSampler(int has_combined, int has_sampler,
                                               uint32_t sampler_slot,
                                               uint32_t max_sampler_slots) {
    return has_combined && has_sampler && sampler_slot < max_sampler_slots ? 1
                                                                           : 0;
}

int mglBindingTexturePlanSamplerMaterialize(const MGLSamplerMaterializeInput *in,
                                            MGLSamplerMaterializePlan *out) {
    if (!in || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->source_tag = NULL;

    if (in->force_default) {
        out->action = MGL_SM_ACTION_USE_DEFAULT;
        out->reason = MGL_SM_REASON_FORCE_DEFAULT;
        out->source_tag = "default";
        return 0;
    }
    if (in->unit_in_range && in->has_gl_sampler) {
        out->action = MGL_SM_ACTION_USE_GL_SAMPLER;
        out->reason = MGL_SM_REASON_GL_SAMPLER;
        out->source_tag = "glSampler";
        if (in->gl_sampler_dirty && in->has_gl_sampler_mtl) {
            out->recreate_gl_sampler_mtl = 1;
        } else if (!in->has_gl_sampler_mtl) {
            out->recreate_gl_sampler_mtl = 1;
        }
        out->clear_gl_sampler_dirty = 1;
        return 0;
    }
    if (in->require_tex_params_mtl) {
        if (!in->has_tex_params_mtl) {
            out->action = MGL_SM_ACTION_KEEP;
            out->reason = MGL_SM_REASON_KEEP;
            return 0;
        }
        out->action = MGL_SM_ACTION_USE_TEX_PARAMS;
        out->reason = MGL_SM_REASON_TEX_PARAMS;
        out->source_tag = "texParamsFallback";
        return 0;
    }
    /* Fragment: assign tex-params MTL even when NULL. */
    out->action = MGL_SM_ACTION_USE_TEX_PARAMS;
    out->reason = MGL_SM_REASON_TEX_PARAMS;
    out->source_tag = "texParamsFallback";
    return 0;
}

int mglBindingTextureRateLogHit(uint64_t *counter, uint64_t early,
                                uint64_t period) {
    if (!counter) {
        return 0;
    }
    uint64_t hit = ++(*counter);
    if (early == 0ull) {
        early = 1ull;
    }
    if (period == 0ull) {
        return hit <= early ? 1 : 0;
    }
    return hit <= early || (hit % period) == 0ull ? 1 : 0;
}

uint64_t mglBindingTextureMipDiagMix(uint64_t sig, uint64_t value) {
    sig ^= value;
    sig *= 1099511628211ULL;
    return sig;
}

uint64_t mglBindingTextureMipDiagSignature(
    uint32_t tex_name, uint32_t min_filter, uint32_t mag_filter,
    uint32_t base_level, uint32_t max_level, uint64_t mtl_levels,
    uint64_t mtl_ptr_bits, int via_copy, uint32_t sampled_levels,
    uint32_t dirty_mip_mask, int version_mismatch) {
    uint64_t signature = 1469598103934665603ULL;
    signature = mglBindingTextureMipDiagMix(signature, tex_name);
    signature = mglBindingTextureMipDiagMix(signature, min_filter);
    signature = mglBindingTextureMipDiagMix(signature, mag_filter);
    signature = mglBindingTextureMipDiagMix(signature, base_level);
    signature = mglBindingTextureMipDiagMix(signature, max_level);
    signature = mglBindingTextureMipDiagMix(signature, mtl_levels);
    signature = mglBindingTextureMipDiagMix(signature, mtl_ptr_bits);
    signature = mglBindingTextureMipDiagMix(signature, via_copy ? 1u : 0u);
    signature = mglBindingTextureMipDiagMix(signature, sampled_levels);
    signature = mglBindingTextureMipDiagMix(signature, dirty_mip_mask);
    signature = mglBindingTextureMipDiagMix(signature, version_mismatch ? 1u : 0u);
    return signature;
}

int mglBindingTexturePlanSampledDiag(const MGLSampledDiagGateInput *in,
                                     MGLSampledDiagGatePlan *out) {
    if (!in || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (in->is_texel_buffer) {
        out->log_texel_buffer = 1;
        out->action = MGL_SD_ACTION_LOG_DETAIL;
        return 0;
    }
    int level_bad = in->level0_suspicious_zero || in->level0_never_written ||
                    in->level0_uninit;
    if (in->stage_is_fragment) {
        int suspicious = in->used_fallback || in->is_gui_rt_copy_eligible ||
                         in->focused_loading_window || level_bad;
        /* historical also flags glTex name==13; caller ORs into used_fallback
         * or focused window before calling. */
        if (!suspicious) {
            out->action = MGL_SD_ACTION_SKIP;
            return 0;
        }
        out->log_detail = 1;
        out->action = MGL_SD_ACTION_LOG_DETAIL;
        if (in->is_gui_rt_copy_eligible) {
            out->log_gui_rt = 1;
            out->action = MGL_SD_ACTION_LOG_GUI_RT;
        }
        if (in->has_bound_texture && level_bad) {
            out->log_readback = 1;
        }
        return 0;
    }
    /* vertex: focus program 34 or bad level0 */
    if (in->vertex_focus_program || level_bad) {
        out->log_detail = 1;
        out->action = MGL_SD_ACTION_LOG_DETAIL;
        return 0;
    }
    out->action = MGL_SD_ACTION_SKIP;
    return 0;
}

void mglBindingTextureFillSamplerMaterializeInput(
    MGLSamplerMaterializeInput *in, int force_default, int unit_in_range,
    int has_gl_sampler, int gl_sampler_dirty, int has_gl_sampler_mtl,
    int has_tex_params_mtl, int require_tex_params_mtl)
{
    if (!in) {
        return;
    }
    memset(in, 0, sizeof(*in));
    in->force_default = force_default ? 1 : 0;
    in->unit_in_range = unit_in_range ? 1 : 0;
    in->has_gl_sampler = has_gl_sampler ? 1 : 0;
    in->gl_sampler_dirty = gl_sampler_dirty ? 1 : 0;
    in->has_gl_sampler_mtl = has_gl_sampler_mtl ? 1 : 0;
    in->has_tex_params_mtl = has_tex_params_mtl ? 1 : 0;
    in->require_tex_params_mtl = require_tex_params_mtl ? 1 : 0;
}

void mglBindingTextureFillSampledRTInput(
    MGLSampledTextureBindInput *in, int used_type_fallback, int is_render_target,
    int yflip, int has_sampled_copy, int copy_fresh, int can_use_rt_copy,
    int want_base_level_on_original, int copy_type_ok, int copy_kind_ok)
{
    if (!in) {
        return;
    }
    memset(in, 0, sizeof(*in));
    in->phase = MGL_ST_PHASE_RT;
    in->used_type_fallback = used_type_fallback ? 1 : 0;
    in->is_render_target = is_render_target ? 1 : 0;
    in->yflip = yflip;
    in->has_sampled_copy = has_sampled_copy ? 1 : 0;
    in->copy_fresh = copy_fresh ? 1 : 0;
    in->can_use_rt_copy = can_use_rt_copy ? 1 : 0;
    in->want_base_level_on_original = want_base_level_on_original ? 1 : 0;
    in->copy_type_ok = copy_type_ok ? 1 : 0;
    in->copy_kind_ok = copy_kind_ok ? 1 : 0;
}

void mglBindingTextureFillStorageImageInput(
    MGLStorageImageBindInput *in, int pass, int skip_resource, int has_resource,
    uint32_t resource_binding, uint32_t element, uint32_t fallback_metal_slot,
    int use_resource_unit, int explicit_by_slot, uint32_t explicit_unit,
    int32_t sampler_unit, uint32_t resource_gl_binding,
    uint32_t fallback_gl_binding, uint32_t max_units)
{
    if (!in) {
        return;
    }
    memset(in, 0, sizeof(*in));
    in->pass = pass;
    in->skip_resource = skip_resource ? 1 : 0;
    in->has_resource = has_resource ? 1 : 0;
    in->resource_binding = resource_binding;
    in->element = element;
    in->fallback_metal_slot = fallback_metal_slot;
    in->use_resource_unit = use_resource_unit ? 1 : 0;
    in->explicit_by_slot = explicit_by_slot ? 1 : 0;
    in->explicit_unit = explicit_unit;
    in->sampler_unit = sampler_unit;
    in->resource_gl_binding = resource_gl_binding;
    in->fallback_gl_binding = fallback_gl_binding;
    in->max_units = max_units;
}

void mglBindingTextureFillSampledDiagGateInput(
    MGLSampledDiagGateInput *in, int stage_is_fragment, int used_fallback,
    int is_gui_rt_copy_eligible, int focused_loading_window,
    int vertex_focus_program, int level0_suspicious_zero,
    int level0_never_written, int level0_uninit, int has_bound_texture,
    int is_texel_buffer)
{
    if (!in) {
        return;
    }
    memset(in, 0, sizeof(*in));
    in->stage_is_fragment = stage_is_fragment ? 1 : 0;
    in->used_fallback = used_fallback ? 1 : 0;
    in->is_gui_rt_copy_eligible = is_gui_rt_copy_eligible ? 1 : 0;
    in->focused_loading_window = focused_loading_window ? 1 : 0;
    in->vertex_focus_program = vertex_focus_program ? 1 : 0;
    in->level0_suspicious_zero = level0_suspicious_zero ? 1 : 0;
    in->level0_never_written = level0_never_written ? 1 : 0;
    in->level0_uninit = level0_uninit ? 1 : 0;
    in->has_bound_texture = has_bound_texture ? 1 : 0;
    in->is_texel_buffer = is_texel_buffer ? 1 : 0;
}

/* LP64 layout mirror of MGLFragmentTextureTraceBinding. */
typedef struct MGLBindingFragTraceLayout {
    uint32_t gl_texture_name;
    uint32_t sampler_unit;
    uint32_t metal_binding;
    uint32_t program_name;
    uint32_t rt_write_version;
    uint32_t sampled_write_version;
    void *gl_texture_ptr;
    void *mtl_texture_ptr;
    void *direct_mtl_texture_ptr;
    void *sampled_copy_ptr;
    uint64_t width;
    uint64_t height;
    uint64_t pixel_format;
    uint64_t texture_type;
    uint8_t used_sampled_copy;
    uint8_t used_fallback;
} MGLBindingFragTraceLayout;

void mglBindingTextureWriteFragTrace(
    void *out, uint32_t gl_texture_name, uint32_t sampler_unit,
    uint32_t metal_binding, uint32_t program_name, uint32_t rt_write_version,
    uint32_t sampled_write_version, void *gl_texture_ptr, void *mtl_texture_ptr,
    void *direct_mtl_texture_ptr, void *sampled_copy_ptr, uint64_t width,
    uint64_t height, uint64_t pixel_format, uint64_t texture_type,
    int used_sampled_copy, int used_fallback)
{
    MGLBindingFragTraceLayout layout;
    if (!out) {
        return;
    }
    memset(&layout, 0, sizeof(layout));
    layout.gl_texture_name = gl_texture_name;
    layout.sampler_unit = sampler_unit;
    layout.metal_binding = metal_binding;
    layout.program_name = program_name;
    layout.rt_write_version = rt_write_version;
    layout.sampled_write_version = sampled_write_version;
    layout.gl_texture_ptr = gl_texture_ptr;
    layout.mtl_texture_ptr = mtl_texture_ptr;
    layout.direct_mtl_texture_ptr = direct_mtl_texture_ptr;
    layout.sampled_copy_ptr = sampled_copy_ptr;
    layout.width = width;
    layout.height = height;
    layout.pixel_format = pixel_format;
    layout.texture_type = texture_type;
    layout.used_sampled_copy = used_sampled_copy ? 1u : 0u;
    layout.used_fallback = used_fallback ? 1u : 0u;
    memcpy(out, &layout, sizeof(layout));
}


void mglBindingTextureFillSampledDiagEmitCore(
    MGLSampledDiagEmitInput *in, const char *stage, uint32_t program_name,
    uint32_t vertex_program_name, uint32_t fragment_program_name,
    const char *sampled_name, uint32_t program_binding, uint32_t texture_unit,
    int res_unit, int explicit_unit, uint32_t gl_tex, uint32_t target,
    int used_fallback, uint64_t expected_type, uint64_t lookup_type,
    int expected_index, uint32_t unit_active, uint32_t unit_expected,
    uint32_t unit_2d, uint32_t unit_cube, uint64_t mtl_type, uint64_t mtl_w,
    uint64_t mtl_h, uint64_t mtl_format, uint32_t l0w, uint32_t l0h, uint32_t l0d,
    uint64_t l0_bytes, uint32_t l0_ever, uint32_t l0_full, uint32_t l0_zero,
    uint32_t l0_source, uint64_t l0_upload, uint64_t l0_hash, uint64_t l0_data_hash,
    uint32_t ptr_tex, int stage_is_fragment, int is_gui_rt_copy_eligible,
    int focused_loading_window, int vertex_focus_program, int is_texel_buffer,
    int level0_suspicious_zero, int level0_never_written, int level0_uninit,
    int has_bound_texture, uint64_t bind_call, int used_sampled_copy_trace,
    uint32_t draw_fbo, uint32_t rp_fbo, uint32_t unit_buffer_tex)
{
    if (!in) {
        return;
    }
    in->stage = stage;
    in->program_name = program_name;
    in->vertex_program_name = vertex_program_name;
    in->fragment_program_name = fragment_program_name;
    in->sampled_name = sampled_name;
    in->program_binding = program_binding;
    in->texture_unit = texture_unit;
    in->res_unit = res_unit;
    in->explicit_unit = explicit_unit ? 1 : 0;
    in->gl_tex = gl_tex;
    in->target = target;
    in->used_fallback = used_fallback ? 1 : 0;
    in->expected_type = expected_type;
    in->lookup_type = lookup_type;
    in->expected_index = expected_index;
    in->unit_active = unit_active;
    in->unit_expected = unit_expected;
    in->unit_2d = unit_2d;
    in->unit_cube = unit_cube;
    in->mtl_type = mtl_type;
    in->mtl_w = mtl_w;
    in->mtl_h = mtl_h;
    in->mtl_format = mtl_format;
    in->l0w = l0w;
    in->l0h = l0h;
    in->l0d = l0d;
    in->l0_bytes = l0_bytes;
    in->l0_ever = l0_ever;
    in->l0_full = l0_full;
    in->l0_zero = l0_zero;
    in->l0_source = l0_source;
    in->l0_upload = l0_upload;
    in->l0_hash = l0_hash;
    in->l0_data_hash = l0_data_hash;
    in->ptr_tex = ptr_tex;
    in->stage_is_fragment = stage_is_fragment ? 1 : 0;
    in->is_gui_rt_copy_eligible = is_gui_rt_copy_eligible ? 1 : 0;
    in->focused_loading_window = focused_loading_window ? 1 : 0;
    in->vertex_focus_program = vertex_focus_program ? 1 : 0;
    in->is_texel_buffer = is_texel_buffer ? 1 : 0;
    in->level0_suspicious_zero = level0_suspicious_zero ? 1 : 0;
    in->level0_never_written = level0_never_written ? 1 : 0;
    in->level0_uninit = level0_uninit ? 1 : 0;
    in->has_bound_texture = has_bound_texture ? 1 : 0;
    in->bind_call = bind_call;
    in->used_sampled_copy_trace = used_sampled_copy_trace ? 1 : 0;
    in->draw_fbo = draw_fbo;
    in->rp_fbo = rp_fbo;
    in->unit_buffer_tex = unit_buffer_tex;
}


static int mglBindingTextureMipDiagStateChanged(uint64_t *cache,
                                                 uint64_t signature)
{
    if (!cache) {
        return 0;
    }
    if (*cache == signature) {
        return 0;
    }
    *cache = signature;
    return 1;
}

int mglBindingTextureEmitMipDiagFragIfChanged(
    uint64_t *state_slot, uint64_t signature, uint32_t unit, uint32_t binding,
    uint32_t program, uint32_t gl_tex, const char *source, uint32_t min_filter,
    uint32_t mag_filter, double min_lod, double max_lod, double aniso,
    uint32_t base, uint32_t max_level, uint32_t gl_levels, uint64_t mtl_levels,
    uint64_t mtl_w, uint64_t mtl_h, const void *mtl, int render_target,
    int via_copy, uint32_t copy_levels, uint32_t dirty_mips, uint32_t rt_ver,
    uint32_t copy_ver)
{
    if (!mglBindingTextureMipDiagStateChanged(state_slot, signature)) {
        return 0;
    }
    mglBindingLogMipDiagFrag(unit, binding, program, gl_tex, source, min_filter,
                             mag_filter, min_lod, max_lod, aniso, base, max_level,
                             gl_levels, mtl_levels, mtl_w, mtl_h, mtl,
                             render_target, via_copy, copy_levels, dirty_mips,
                             rt_ver, copy_ver);
    return 1;
}

void mglBindingTextureEmitSampledDiagPorts(
    const MGLSampledDiagEmitInput *in, MGLSampledDiagEmitResult *out)
{
    static uint64_t s_texelBufferBindLogs = 0;
    static uint64_t s_sampleDetailLogCount[2] = {0, 0};
    static uint64_t s_guiRTSampleLogCount = 0;
    static uint64_t s_sampleReadbackCount = 0;
    MGLSampledDiagGateInput din;
    MGLSampledDiagGatePlan dplan = {0};

    if (out) {
        out->want_readback = 0;
        out->readback_reason = NULL;
        out->readback_hit = 0;
    }
    if (!in) {
        return;
    }

    if (in->do_focused) {
        mglBindingLogTBINDFocused(
            in->stage, in->program_name, in->sampled_name, in->program_binding,
            in->texture_unit, in->gl_tex, in->target, in->mtl, in->mtl_type,
            in->mtl_w, in->mtl_h, in->l0w, in->l0h, in->l0_ever, in->l0_full,
            in->l0_source);
    }
    if (in->do_trace_file) {
        mglBindingLogTBINDTraceFile(
            in->stage, in->program_name, in->sampled_name, in->program_binding,
            in->texture_unit, in->res_unit, in->explicit_unit, in->gl_tex,
            in->target, in->used_fallback, in->expected_type, in->lookup_type,
            in->expected_index, in->unit_active, in->unit_expected, in->unit_2d,
            in->unit_cube, in->mtl, in->mtl_type, in->mtl_w, in->mtl_h, in->l0w,
            in->l0h, in->l0_ever, in->l0_full, in->l0_source);
    }

    mglBindingTextureFillSampledDiagGateInput(
        &din, in->stage_is_fragment, in->used_fallback,
        in->is_gui_rt_copy_eligible, in->focused_loading_window,
        in->vertex_focus_program, in->level0_suspicious_zero,
        in->level0_never_written, in->level0_uninit, in->has_bound_texture,
        in->is_texel_buffer);
    (void)mglBindingTexturePlanSampledDiag(&din, &dplan);

    if (dplan.log_texel_buffer &&
        mglBindingTextureRateLogHit(&s_texelBufferBindLogs, 8ull, 2048ull)) {
        mglBindingLogTexBufferBind(
            s_texelBufferBindLogs, in->program_name, in->program_binding,
            in->texture_unit, in->ptr_tex, in->unit_active, in->unit_buffer_tex,
            in->expected_type, in->lookup_type, in->mtl, in->mtl_type, in->mtl_w,
            in->mtl_h, in->mtl_format, in->sampler);
    }
    if (dplan.log_detail) {
        uint64_t *ctr =
            &s_sampleDetailLogCount[in->stage_is_fragment ? 1 : 0];
        uint64_t early = in->stage_is_fragment ? 256ull : 128ull;
        if (mglBindingTextureRateLogHit(ctr, early, 512ull)) {
            mglBindingLogSampleDetail(
                in->bind_call, *ctr, in->stage, in->program_name,
                in->sampled_name, in->program_binding, in->texture_unit,
                in->expected_type, in->expected_index, in->ptr_tex, in->ptr,
                in->target, in->used_fallback, in->mtl, in->mtl_type, in->mtl_w,
                in->mtl_h, in->unit_active, in->unit_expected, in->unit_2d,
                in->unit_cube, in->l0w, in->l0h, in->l0d, in->l0_bytes,
                in->l0_ever, in->l0_full, in->l0_zero, in->l0_source,
                in->l0_upload, in->l0_src, in->l0_hash, in->l0_data_hash);
        }
    }
    if (dplan.log_gui_rt &&
        mglBindingTextureRateLogHit(&s_guiRTSampleLogCount, 128ull, 256ull)) {
        mglBindingLogRTSampleCopySample(
            s_guiRTSampleLogCount, in->bind_call, in->program_name,
            in->vertex_program_name, in->fragment_program_name,
            in->sampled_name, in->program_binding, in->texture_unit, in->ptr_tex,
            in->rt_label, in->used_fallback, in->used_sampled_copy_trace,
            in->ptr, in->mtl, in->direct_for_trace, in->copy_for_trace,
            in->mtl_format, in->mtl_type, in->mtl_w, in->mtl_h, in->draw_fbo,
            in->rp_fbo, in->rp_color, in->rp_depth);
    }
    if (dplan.log_readback && in->mtl && out) {
        if (mglBindingTextureRateLogHit(&s_sampleReadbackCount, 32ull,
                                        512ull)) {
            out->want_readback = 1;
            out->readback_hit = s_sampleReadbackCount;
            out->readback_reason =
                in->l0_zero ? "zero-level"
                            : (!in->l0_ever ? "never-written" : "not-initialized");
        }
    }
}

const char *mglBindingTextureSamplerStageTag(const char *stage) {
    if (stage && (stage[0] == 'v' || stage[0] == 'V')) {
        return "VERT";
    }
    if (stage && (stage[0] == 'c' || stage[0] == 'C')) {
        return "COMP";
    }
    return "FRAG";
}
