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
 * mgl_binding_texture.c — O3.3 residual sampled/storage BindingState plans.
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
    out->texture_slot = in->spirv_binding;
    out->sampler_slot = in->sampler_binding;

    if (in->phase == MGL_ST_PHASE_GATE) {
        if (mglRenderMetalBindingPastUnits(in->spirv_binding, in->max_units) ||
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
            return 0;
        }
        /* Prefer an already-fresh usable copy. */
        if (in->has_sampled_copy && in->copy_fresh && in->can_use_rt_copy &&
            in->copy_type_ok && in->copy_kind_ok) {
            out->action = MGL_ST_ACTION_RT_USE_COPY;
            out->reason = MGL_ST_REASON_RT_COPY;
            return 0;
        }
        /* Repair attempt results (ObjC filled after freshGLSampled*). */
        if (in->repaired_available) {
            if (!in->repaired_fresh) {
                out->action = MGL_ST_ACTION_RT_USE_COPY;
                out->reason = MGL_ST_REASON_RT_REPAIR;
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
        if (in->suppress_missing_fallback) {
            out->action = MGL_ST_ACTION_SUPPRESS_FALLBACK;
            out->reason = MGL_ST_REASON_SUPPRESS;
            out->mark_nil = 1;
            return 0;
        }
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
    /* Historical: (!resource || has_combined) && sampler && slot < max */
    if (in->has_sampler && in->sampler_binding < in->max_sampler_slots &&
        (!in->has_resource || in->has_combined_sampler)) {
        out->queue_sampler = 1;
    }
    (void)in->force_default_sampler;
    return 0;
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
