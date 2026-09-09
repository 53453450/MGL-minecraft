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
 * mgl_binding_stage.c — O3.3 residual stage-buffer bind plan + helpers.
 * UBO / SSBO / plain-uniform / atomic bind decisions for +BindingState.
 * Pure C; resource-type numeric ABI. Do not grow +Binding.m / mgl_render.cpp.
 */

#include "mgl_binding_stage.h"

#include <string.h>

/* mgl_types_program.h resource ABI (avoid heavy includes). */
enum {
    MGL_SB_RES_UNIFORM_BUFFER = 1,
    MGL_SB_RES_UNIFORM_CONSTANT = 2,
    MGL_SB_RES_STORAGE_BUFFER = 3,
    MGL_SB_RES_ATOMIC_COUNTER = 9
};

int mglRenderUseInlineFragmentBytes(int is_base_binding, int64_t size) {
    return !is_base_binding && size < 4096 ? 1 : 0;
}

int mglRenderCPUPointerLooksTagged(const void *p) {
    return p && (uintptr_t)p < 0x100000000ULL ? 1 : 0;
}

int mglRenderMetalDataPointerUsable(const void *p) {
    return p && (uintptr_t)p >= 0x10000u ? 1 : 0;
}

int mglRenderNeedsIsolatedStageBinding(int has_buffer, int64_t offset,
                                       uint64_t metal_len, uint64_t available,
                                       uint32_t required) {
    return !has_buffer || offset < 0 || (uint64_t)offset >= metal_len ||
                   available < required
               ? 1
               : 0;
}

int mglRenderAllowIsolateGPUWriteTarget(int gpu_write_target,
                                        int allow_when_gpu) {
    return !gpu_write_target || allow_when_gpu ? 1 : 0;
}

int mglRenderBindOffsetInBuffer(int64_t offset, int64_t size) {
    if (offset < 0 || size <= 0) {
        return 0;
    }
    return (uint64_t)offset < (uint64_t)size ? 1 : 0;
}

uint32_t mglRenderRequiredBindingBytesForMap(int resource_type,
                                             uint32_t reflected,
                                             int64_t visible,
                                             uint32_t min_stage) {
    if (resource_type == MGL_SB_RES_UNIFORM_BUFFER && reflected > 0u) {
        if (visible > 0 && (uint64_t)visible < (uint64_t)reflected) {
            return (uint32_t)visible;
        }
        return reflected;
    }
    return reflected > min_stage ? reflected : min_stage;
}

int mglRenderUseUniformConstantInline(int is_base, int resource_type,
                                      int has_cpu, int64_t offset,
                                      uint32_t required, uint32_t scratch) {
    return is_base && resource_type == MGL_SB_RES_UNIFORM_CONSTANT && has_cpu &&
                   offset == 0 && required <= scratch
               ? 1
               : 0;
}

int mglRenderIsolateUBOPrefersCPUShadow(uint32_t resource_type, int has_buf,
                                        int has_cpu, int64_t offset) {
    return resource_type == (uint32_t)MGL_SB_RES_UNIFORM_BUFFER && has_buf &&
                   has_cpu && offset >= 0
               ? 1
               : 0;
}

int mglRenderIsolateUBOUsesFullStore(uint32_t resource_type) {
    return resource_type == (uint32_t)MGL_SB_RES_UNIFORM_BUFFER ? 1 : 0;
}

uint64_t mglRenderIsolateCopyLength(uint64_t src_bytes, uint64_t required) {
    return src_bytes > required ? required : src_bytes;
}

int mglRenderWritableStorageNeedsGPUAuthoritative(int resource_type) {
    return resource_type == MGL_SB_RES_STORAGE_BUFFER ||
                   resource_type == MGL_SB_RES_ATOMIC_COUNTER
               ? 1
               : 0;
}

uint32_t mglBindingStageFallbackResourceTypes(uint32_t *types_out,
                                              uint32_t cap) {
    static const uint32_t kTypes[MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT] = {
        MGL_SB_RES_UNIFORM_BUFFER, MGL_SB_RES_UNIFORM_CONSTANT,
        MGL_SB_RES_STORAGE_BUFFER, MGL_SB_RES_ATOMIC_COUNTER};
    uint32_t n = MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT;
    if (!types_out || cap < n) {
        return n;
    }
    memcpy(types_out, kTypes, sizeof(kTypes));
    return n;
}

int mglBindingStageResolveSlot(int is_base_binding, int has_metal_binding,
                               int32_t metal_binding_index,
                               int32_t gl_binding_index, uint32_t max_slots,
                               uint32_t *out_slot) {
    if (!is_base_binding) {
        return 0;
    }
    int32_t slot = has_metal_binding ? metal_binding_index : gl_binding_index;
    if (slot < 0 || (uint32_t)slot >= max_slots) {
        return 0;
    }
    if (out_slot) {
        *out_slot = (uint32_t)slot;
    }
    return 1;
}

int mglBindingStageFallbackNeedsBind(int any_present_at_slot,
                                     int has_fallback_buffer) {
    return !any_present_at_slot && has_fallback_buffer ? 1 : 0;
}

static void mglBindingStagePlanClear(MGLStageBufferBindPlan *out) {
    memset(out, 0, sizeof(*out));
}

int mglBindingStagePlanMapEntry(const MGLStageBufferBindInput *in,
                                MGLStageBufferBindPlan *out) {
    if (!in || !out) {
        return -1;
    }
    mglBindingStagePlanClear(out);

    /* Vertex stage-buffer pass is base/resource only; attribs bind elsewhere. */
    if (!in->is_fragment && !in->is_base_binding) {
        out->action = MGL_SB_ACTION_SKIP;
        out->reason = MGL_SB_REASON_NOT_BASE;
        return 0;
    }

    uint32_t slot = 0u;
    if (in->is_base_binding) {
        if (!mglBindingStageResolveSlot(1, in->has_metal_binding,
                                        in->metal_binding_index,
                                        in->gl_binding_index, in->max_metal_slots,
                                        &slot)) {
            out->action = MGL_SB_ACTION_SKIP;
            out->reason = MGL_SB_REASON_SLOT_OOR;
            return 0;
        }
    } else {
        /* Fragment non-base: client index is the Metal slot. */
        if (in->gl_binding_index < 0 ||
            (uint32_t)in->gl_binding_index >= in->max_metal_slots) {
            out->action = MGL_SB_ACTION_SKIP;
            out->reason = MGL_SB_REASON_SLOT_OOR;
            return 0;
        }
        slot = (uint32_t)in->gl_binding_index;
    }
    out->metal_slot = slot;
    out->gl_binding =
        in->gl_binding_index >= 0 ? (uint32_t)in->gl_binding_index : 0u;
    out->bind_offset = in->offset >= 0 ? (uint64_t)in->offset : 0u;
    out->reflected_bytes = in->reflected_required;

    if (!in->is_fragment && in->attrib_slot_reserved) {
        out->action = MGL_SB_ACTION_SKIP;
        out->reason = MGL_SB_REASON_ATTRIB_RESERVED;
        return 0;
    }

    if (in->is_base_binding && in->gl_binding_index >= 0 &&
        (uint32_t)in->gl_binding_index < in->max_gl_bindings) {
        out->mark_base_present = 1;
    }

    if (!in->has_buffer) {
        out->action = MGL_SB_ACTION_CLEAR;
        out->reason = MGL_SB_REASON_NULL_BUFFER;
        return 0;
    }

    if (in->offset < 0) {
        out->action = MGL_SB_ACTION_CLEAR;
        out->reason = MGL_SB_REASON_BAD_OFFSET;
        return 0;
    }

    if (in->buffer_size < 0) {
        /* Vertex clears the slot; fragment historically only skipped. */
        if (in->is_fragment) {
            out->action = MGL_SB_ACTION_SKIP;
        } else {
            out->action = MGL_SB_ACTION_CLEAR;
        }
        out->reason = MGL_SB_REASON_BAD_SIZE;
        return 0;
    }

    uint32_t required = in->min_stage_bytes;
    if (in->is_base_binding && in->gl_binding_index >= 0 &&
        (uint32_t)in->gl_binding_index < in->max_gl_bindings) {
        required = mglRenderRequiredBindingBytesForMap(
            (int)in->resource_type, in->reflected_required,
            in->visible_range, in->min_stage_bytes);
    }
    out->required_bytes = required;

    /* ---- Fragment non-base-style small inline (historical FS path) ----
     * UseInlineFragmentBytes is !is_base && size<4096; base bindings never
     * take this arm. Kept for ABI completeness if callers pass is_base=0. */
    if (in->is_fragment &&
        mglRenderUseInlineFragmentBytes(in->is_base_binding, in->buffer_size)) {
        if (in->has_cpu_data && in->buffer_size > 0) {
            if (mglRenderCPUPointerLooksTagged(in->cpu_ptr)) {
                out->action = MGL_SB_ACTION_CLEAR;
                out->reason = MGL_SB_REASON_INLINE_FS_TAGGED;
                return 0;
            }
            if (!mglRenderBindOffsetInBuffer(in->offset, in->buffer_size)) {
                out->action = MGL_SB_ACTION_CLEAR;
                out->reason = MGL_SB_REASON_INLINE_FS_BAD_OFF;
                return 0;
            }
            out->action = MGL_SB_ACTION_INLINE_BYTES;
            out->reason = MGL_SB_REASON_INLINE_FS_SMALL;
            out->inline_src_offset = (uint64_t)in->offset;
            out->inline_length =
                (uint32_t)((uint64_t)in->buffer_size - (uint64_t)in->offset);
            out->inline_visible = out->inline_length;
            out->mark_any_present = 1;
            out->invalidate_last_bound = 1;
            out->clear_cpu_dirty = 1;
            return 0;
        }
        if (in->has_mtl_data) {
            if (mglRenderCPUPointerLooksTagged(in->mtl_ptr)) {
                out->action = MGL_SB_ACTION_CLEAR;
                out->reason = MGL_SB_REASON_INLINE_FS_TAGGED;
                return 0;
            }
            if (in->phase == MGL_SB_PHASE_PRE_MTL && in->metal_len == 0u) {
                out->action = MGL_SB_ACTION_NEED_MTL;
                out->reason = MGL_SB_REASON_NEED_ENSURE;
                out->use_mtl_as_inline_src = 1;
                return 0;
            }
            if (in->metal_len == 0u ||
                (uint64_t)in->offset >= in->metal_len) {
                out->action = MGL_SB_ACTION_CLEAR;
                out->reason = MGL_SB_REASON_INLINE_FS_BAD_OFF;
                return 0;
            }
            out->metal_len = (uint32_t)in->metal_len;
            out->available_bytes = (uint32_t)in->visible_mtl;
            if (in->binding_state_valid && in->buffer_matches) {
                out->action = MGL_SB_ACTION_SKIP_MATCHED;
                out->reason = MGL_SB_REASON_MATCHED;
                out->mark_any_present = 1;
                out->clear_cpu_dirty = 1;
                return 0;
            }
            out->action = MGL_SB_ACTION_BIND_BUFFER;
            out->reason = MGL_SB_REASON_INLINE_FS_MTL;
            out->use_mtl_as_inline_src = 1;
            out->mark_any_present = 1;
            out->clear_cpu_dirty = 1;
            return 0;
        }
        out->action = MGL_SB_ACTION_CLEAR;
        out->reason = MGL_SB_REASON_INLINE_FS_EMPTY;
        return 0;
    }

    /* ---- Uniform-constant set*Bytes (both stages) ---- */
    if (mglRenderUseUniformConstantInline(
            in->is_base_binding ? 1 : 0, (int)in->resource_type,
            in->has_cpu_data ? 1 : 0, in->offset, required, in->scratch_cap)) {
        uint32_t visible = (uint32_t)in->visible_cpu;
        uint32_t inline_len = visible > required ? visible : required;
        if (visible > 0u && inline_len <= in->scratch_cap) {
            out->action = MGL_SB_ACTION_INLINE_BYTES;
            out->reason = MGL_SB_REASON_INLINE_UC;
            out->inline_length = inline_len;
            out->inline_visible = visible;
            out->inline_src_offset = 0u;
            out->mark_any_present = 1;
            out->invalidate_last_bound = 1;
            out->clear_cpu_dirty_if_no_mtl = 1;
            return 0;
        }
    }

    /* ---- PRE_MTL: ask ObjC to ensure backing, then re-plan ---- */
    if (in->phase == MGL_SB_PHASE_PRE_MTL) {
        out->action = MGL_SB_ACTION_NEED_MTL;
        out->reason = MGL_SB_REASON_NEED_ENSURE;
        return 0;
    }

    /* ---- POST_MTL: isolate or bind ---- */
    int has_usable_mtl = in->has_mtl_data && in->mtl_usable ? 1 : 0;
    out->metal_len = (uint32_t)in->metal_len;
    out->available_bytes = (uint32_t)in->visible_mtl;

    if (mglRenderNeedsIsolatedStageBinding(
            has_usable_mtl, in->offset, in->metal_len, in->visible_mtl,
            required) &&
        mglRenderAllowIsolateGPUWriteTarget(in->gpu_write_target ? 1 : 0,
                                            in->allow_isolate_when_gpu ? 1
                                                                       : 0)) {
        out->action = MGL_SB_ACTION_ISOLATE;
        out->reason = MGL_SB_REASON_ISOLATE;
        out->needs_flush_snapshot = 1;
        out->needs_copy_back =
            in->allow_isolate_when_gpu &&
                    mglRenderWritableStorageNeedsGPUAuthoritative(
                        (int)in->resource_type) &&
                    has_usable_mtl && in->visible_mtl > 0u
                ? 1
                : 0;
        out->mark_any_present = 1;
        out->bind_offset = 0u; /* isolated always @0 */
        return 0;
    }

    if (!has_usable_mtl) {
        out->action = MGL_SB_ACTION_CLEAR;
        out->reason = MGL_SB_REASON_NULL_BUFFER;
        return 0;
    }

    if (in->binding_state_valid && in->buffer_matches) {
        out->action = MGL_SB_ACTION_SKIP_MATCHED;
        out->reason = MGL_SB_REASON_MATCHED;
        out->mark_any_present = 1;
        return 0;
    }

    out->action = MGL_SB_ACTION_BIND_BUFFER;
    out->reason = MGL_SB_REASON_BIND;
    out->mark_any_present = 1;
    return 0;
}

/* ---- Attrib helpers + plan (O3.3 attrib BindingState) ---- */

enum {
    MGL_SB_GL_INT = 0x1404,
    MGL_SB_GL_INT_VEC2 = 0x8B53,
    MGL_SB_GL_INT_VEC3 = 0x8B54,
    MGL_SB_GL_INT_VEC4 = 0x8B55,
    MGL_SB_ATTRIB_CONV_NONE = 0,
    MGL_SB_ATTRIB_SPAN_OVERFLOW = -2
};

int mglRenderIntegerAttribDstIsInt(uint32_t shader_gl_type) {
    return shader_gl_type == MGL_SB_GL_INT ||
                   shader_gl_type == MGL_SB_GL_INT_VEC2 ||
                   shader_gl_type == MGL_SB_GL_INT_VEC3 ||
                   shader_gl_type == MGL_SB_GL_INT_VEC4
               ? 1
               : 0;
}

int mglRenderSkipAlreadyBoundUnconverted(int conversion_kind,
                                         int already_present) {
    return conversion_kind == MGL_SB_ATTRIB_CONV_NONE && already_present ? 1
                                                                        : 0;
}

int mglRenderAttribNeedsConversionBind(int conversion_kind) {
    return conversion_kind != MGL_SB_ATTRIB_CONV_NONE ? 1 : 0;
}

int mglRenderAttribWrittenRangeTracked(int64_t written_min,
                                       int64_t written_max) {
    return written_min >= 0 && written_max >= 0 ? 1 : 0;
}

int mglRenderAttribOutsideWrittenRange(int64_t attr_off, int64_t attr_end,
                                       int64_t written_min,
                                       int64_t written_max) {
    return attr_off < written_min || attr_end > written_max ? 1 : 0;
}

uint64_t mglRenderVertexMetalBindOffset(int absolute_mode,
                                        uint64_t binding_offset) {
    return absolute_mode ? binding_offset : 0u;
}

int mglRenderBindingOffsetInMetal(uint64_t offset, uint64_t metal_len) {
    return offset < metal_len ? 1 : 0;
}

int mglBindingStagePlanAttribEntry(const MGLAttribBindInput *in,
                                   MGLAttribBindPlan *out) {
    if (!in || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (!in->program_uses_attrib) {
        out->action = MGL_ATTR_ACTION_SKIP;
        out->reason = MGL_ATTR_REASON_UNUSED;
        return 0;
    }
    if (!in->uses_current_value && !in->has_attrib_binding) {
        out->action = MGL_ATTR_ACTION_SKIP;
        out->reason = MGL_ATTR_REASON_NO_BINDING;
        return 0;
    }
    if (in->mapped_index < 0 ||
        (uint32_t)in->mapped_index >= in->max_metal_slots) {
        out->action = MGL_ATTR_ACTION_SKIP;
        out->reason = MGL_ATTR_REASON_BAD_MAP;
        return 0;
    }
    out->metal_slot = (uint32_t)in->mapped_index;

    if (in->uses_current_value) {
        out->action = MGL_ATTR_ACTION_CURRENT;
        out->reason = MGL_ATTR_REASON_CURRENT;
        out->metal_bind_offset = 0u;
        out->mark_present = 1;
        return 0;
    }

    if (!in->has_attrib_binding) {
        out->action = MGL_ATTR_ACTION_SKIP;
        out->reason = MGL_ATTR_REASON_NO_BINDING;
        return 0;
    }

    if (!in->offsets_valid) {
        out->action = MGL_ATTR_ACTION_BLOCK;
        out->reason = MGL_ATTR_REASON_BAD_OFFSET;
        return 0;
    }
    if (in->span_status == MGL_SB_ATTRIB_SPAN_OVERFLOW) {
        out->action = MGL_ATTR_ACTION_BLOCK;
        out->reason = MGL_ATTR_REASON_SPAN_OVERFLOW;
        return 0;
    }

    if (mglRenderSkipAlreadyBoundUnconverted(in->conversion_kind,
                                             in->already_present)) {
        out->action = MGL_ATTR_ACTION_SKIP_ALREADY;
        out->reason = MGL_ATTR_REASON_ALREADY;
        return 0;
    }

    if (mglRenderAttribNeedsConversionBind(in->conversion_kind)) {
        out->action = MGL_ATTR_ACTION_CONVERT;
        out->reason = MGL_ATTR_REASON_CONVERT;
        out->metal_bind_offset = 0u;
        out->mark_present = 1;
        return 0;
    }

    if (in->phase == MGL_ATTR_PHASE_SELECT) {
        out->action = MGL_ATTR_ACTION_NEED_MTL;
        out->reason = MGL_ATTR_REASON_NEED_ENSURE;
        return 0;
    }

    if (!in->has_mtl_data || !in->mtl_usable) {
        out->action = MGL_ATTR_ACTION_SKIP;
        out->reason = MGL_ATTR_REASON_BAD_MTL;
        return 0;
    }

    out->metal_bind_offset = mglRenderVertexMetalBindOffset(
        in->absolute_vertex_offsets, in->binding_offset);
    if (!mglRenderBindingOffsetInMetal(in->binding_offset, in->metal_len)) {
        /* Historical: check binding_offset against metal_len (not bind offset). */
        out->action = MGL_ATTR_ACTION_SKIP;
        out->reason = MGL_ATTR_REASON_BAD_MTL;
        return 0;
    }

    if (in->binding_state_valid && in->buffer_matches) {
        out->action = MGL_ATTR_ACTION_SKIP_MATCHED;
        out->reason = MGL_ATTR_REASON_MATCHED;
        out->mark_present = 1;
        return 0;
    }

    out->action = MGL_ATTR_ACTION_BIND;
    out->reason = MGL_ATTR_REASON_BIND;
    out->mark_present = 1;
    return 0;
}
