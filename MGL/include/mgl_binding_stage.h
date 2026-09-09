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
 * mgl_binding_stage.h
 *
 * O3.3 residual — stage buffer (UBO / SSBO / plain-uniform / atomic) +
 * vertex-attrib bind plan + helpers for +BindingState. Pure C; no Metal-cpp,
 * no renderer instance. Resource-type integers use mgl_types_program ABI
 * (_UNIFORM_BUFFER_RES=1, _UNIFORM_CONSTANT_RES=2, _STORAGE_BUFFER_RES=3,
 * _ATOMIC_COUNTER_RES=9). Attrib conversion kinds match mgl_render.h
 * (MGL_ATTRIB_CONV_*).
 *
 * ObjC +BindingState stays a thin set*Buffer / set*Bytes / set*Texture port.
 * Do not sink these helpers into mgl_render.cpp.
 * Do not grow +Binding.m into a thick shell.
 *
 * Callers historically went through mgl_render.h; that header includes
 * this one so BindingState / Compute / MGLRenderer keep call sites.
 */

#ifndef MGL_BINDING_STAGE_H
#define MGL_BINDING_STAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Stage-bind helpers (ex-mgl_render.cpp; same mglRender* names) ---- */

int mglRenderUseInlineFragmentBytes(int is_base_binding, int64_t size);
int mglRenderCPUPointerLooksTagged(const void *p);
int mglRenderMetalDataPointerUsable(const void *p);
int mglRenderNeedsIsolatedStageBinding(int has_buffer, int64_t offset,
                                       uint64_t metal_len, uint64_t available,
                                       uint32_t required);
int mglRenderAllowIsolateGPUWriteTarget(int gpu_write_target,
                                        int allow_when_gpu);
int mglRenderBindOffsetInBuffer(int64_t offset, int64_t size);
uint32_t mglRenderRequiredBindingBytesForMap(int resource_type,
                                             uint32_t reflected,
                                             int64_t visible,
                                             uint32_t min_stage);
int mglRenderUseUniformConstantInline(int is_base, int resource_type,
                                      int has_cpu, int64_t offset,
                                      uint32_t required, uint32_t scratch);
int mglRenderIsolateUBOPrefersCPUShadow(uint32_t resource_type, int has_buf,
                                        int has_cpu, int64_t offset);
int mglRenderIsolateUBOUsesFullStore(uint32_t resource_type);
uint64_t mglRenderIsolateCopyLength(uint64_t src_bytes, uint64_t required);
int mglRenderWritableStorageNeedsGPUAuthoritative(int resource_type);

/* ---- Fallback resource-type table (UBO / constant / SSBO / atomic) ---- */

enum { MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT = 4 };

/* Fills types[4] with {_UNIFORM_BUFFER_RES, _UNIFORM_CONSTANT_RES,
 * _STORAGE_BUFFER_RES, _ATOMIC_COUNTER_RES}. Returns count (4). */
uint32_t mglBindingStageFallbackResourceTypes(uint32_t *types_out,
                                              uint32_t cap);

/* ---- Map-entry bind plan (vertex / fragment stage buffers) ---- */

enum {
    MGL_SB_PHASE_PRE_MTL = 0,  /* before ensure MTL / dirty upload */
    MGL_SB_PHASE_POST_MTL = 1  /* metal_len / visible_mtl known */
};

enum {
    MGL_SB_ACTION_SKIP = 0,       /* continue loop, no encode */
    MGL_SB_ACTION_CLEAR,          /* set*Buffer nil + clear binding state */
    MGL_SB_ACTION_INLINE_BYTES,   /* set*Bytes (uniform-constant or FS small) */
    MGL_SB_ACTION_NEED_MTL,       /* ensure MTL then re-plan POST_MTL */
    MGL_SB_ACTION_ISOLATE,        /* isolatedStageBindingBuffer + bind@0 */
    MGL_SB_ACTION_BIND_BUFFER,    /* set*Buffer at bind_offset */
    MGL_SB_ACTION_SKIP_MATCHED    /* already bound; perf skip only */
};

enum {
    MGL_SB_REASON_OK = 0,
    MGL_SB_REASON_NOT_BASE,
    MGL_SB_REASON_SLOT_OOR,
    MGL_SB_REASON_ATTRIB_RESERVED,
    MGL_SB_REASON_NULL_BUFFER,
    MGL_SB_REASON_BAD_OFFSET,
    MGL_SB_REASON_BAD_SIZE,
    MGL_SB_REASON_INLINE_UC,
    MGL_SB_REASON_INLINE_FS_SMALL,
    MGL_SB_REASON_INLINE_FS_TAGGED,
    MGL_SB_REASON_INLINE_FS_BAD_OFF,
    MGL_SB_REASON_INLINE_FS_MTL,
    MGL_SB_REASON_INLINE_FS_EMPTY,
    MGL_SB_REASON_NEED_ENSURE,
    MGL_SB_REASON_ISOLATE,
    MGL_SB_REASON_BIND,
    MGL_SB_REASON_MATCHED
};

typedef struct MGLStageBufferBindInput {
    int phase; /* MGL_SB_PHASE_* */
    int is_fragment;
    int is_base_binding;
    int has_metal_binding;
    int32_t metal_binding_index;
    int32_t gl_binding_index;
    uint32_t resource_type;
    int64_t offset;
    int64_t buffer_size;     /* GL Buffer::size */
    int has_buffer;          /* validated Buffer* non-NULL */
    int has_cpu_data;
    int has_mtl_data;
    int mtl_usable; /* stage-specific: VS MetalDataPointerUsable / FS !tagged */
    const void *cpu_ptr;     /* Buffer::data.buffer_data (may be tagged) */
    const void *mtl_ptr;     /* Buffer::data.mtl_data */
    int cpu_dirty;
    int gpu_write_target;
    int allow_isolate_when_gpu; /* native TES active */
    int attrib_slot_reserved;   /* vertex: slot claimed by attrib */
    uint32_t max_metal_slots;
    uint32_t max_gl_bindings; /* MAX_BINDABLE_BUFFERS for base present */
    uint32_t reflected_required;
    uint32_t min_stage_bytes; /* kMGLMinimumStageBindingSize */
    uint32_t scratch_cap;     /* kMGLStageBindingStackScratchSize */
    uint64_t visible_cpu;     /* mglBufferMapVisibleBackingBytes(cpu) */
    int64_t visible_range;    /* mglBufferMapVisibleSize for required bytes */
    uint64_t metal_len;       /* 0 if unknown / not ensured */
    uint64_t visible_mtl;     /* visible from metal backing */
    int binding_state_valid;
    int buffer_matches; /* last-bound matches candidate (POST only) */
} MGLStageBufferBindInput;

typedef struct MGLStageBufferBindPlan {
    uint32_t action; /* MGL_SB_ACTION_* */
    uint32_t reason; /* MGL_SB_REASON_* */
    uint32_t metal_slot;
    uint32_t gl_binding;
    uint64_t bind_offset;
    uint32_t required_bytes;
    uint32_t reflected_bytes;
    uint32_t available_bytes;
    uint32_t metal_len;
    uint32_t inline_length;
    uint32_t inline_visible;
    uint64_t inline_src_offset; /* fragment small: offset into cpu_ptr */
    int mark_base_present;
    int mark_any_present;
    int clear_cpu_dirty;          /* clear DIRTY_BUFFER_DATA after encode */
    int clear_cpu_dirty_if_no_mtl; /* only if !has_mtl_data */
    int needs_flush_snapshot;     /* isolate lifetime */
    int needs_copy_back;          /* TES writable SSBO/atomic */
    int invalidate_last_bound;    /* after set*Bytes */
    int use_mtl_as_inline_src;    /* fragment small MTL fallback path */
} MGLStageBufferBindPlan;

/* Plan one mapped stage-buffer entry. Returns 0 on success (plan filled). */
int mglBindingStagePlanMapEntry(const MGLStageBufferBindInput *in,
                                MGLStageBufferBindPlan *out);

/* Resolve metal slot for a base map entry; 1 if in range. */
int mglBindingStageResolveSlot(int is_base_binding, int has_metal_binding,
                               int32_t metal_binding_index,
                               int32_t gl_binding_index, uint32_t max_slots,
                               uint32_t *out_slot);

/* Fallback slot bind: 1 if missing any_present and fallback buffer exists. */
int mglBindingStageFallbackNeedsBind(int any_present_at_slot,
                                     int has_fallback_buffer);


/* ---- Attrib bind helpers (ex-mgl_render.cpp) ---- */

int mglRenderIntegerAttribDstIsInt(uint32_t shader_gl_type);
int mglRenderSkipAlreadyBoundUnconverted(int conversion_kind, int already_present);
int mglRenderAttribNeedsConversionBind(int conversion_kind);
int mglRenderAttribWrittenRangeTracked(int64_t written_min, int64_t written_max);
int mglRenderAttribOutsideWrittenRange(int64_t attr_off, int64_t attr_end,
                                       int64_t written_min, int64_t written_max);
uint64_t mglRenderVertexMetalBindOffset(int absolute_mode,
                                        uint64_t binding_offset);
int mglRenderBindingOffsetInMetal(uint64_t offset, uint64_t metal_len);

/* ---- Vertex-attrib bind plan ---- */

enum {
    MGL_ATTR_PHASE_SELECT = 0, /* before MTL ensure */
    MGL_ATTR_PHASE_POST_MTL = 1
};

enum {
    MGL_ATTR_ACTION_SKIP = 0,
    MGL_ATTR_ACTION_CURRENT,       /* packed current-value pool */
    MGL_ATTR_ACTION_BLOCK,         /* abort draw */
    MGL_ATTR_ACTION_SKIP_ALREADY,  /* unconverted + already present */
    MGL_ATTR_ACTION_CONVERT,       /* converted stream @0 */
    MGL_ATTR_ACTION_NEED_MTL,      /* ensure MTL then POST */
    MGL_ATTR_ACTION_BIND,          /* setVertexBuffer */
    MGL_ATTR_ACTION_SKIP_MATCHED   /* already bound */
};

enum {
    MGL_ATTR_REASON_OK = 0,
    MGL_ATTR_REASON_UNUSED,
    MGL_ATTR_REASON_NO_BINDING,
    MGL_ATTR_REASON_BAD_MAP,
    MGL_ATTR_REASON_CURRENT,
    MGL_ATTR_REASON_BAD_OFFSET,
    MGL_ATTR_REASON_SPAN_OVERFLOW,
    MGL_ATTR_REASON_ALREADY,
    MGL_ATTR_REASON_CONVERT,
    MGL_ATTR_REASON_NEED_ENSURE,
    MGL_ATTR_REASON_BAD_MTL,
    MGL_ATTR_REASON_BIND,
    MGL_ATTR_REASON_MATCHED
};

typedef struct MGLAttribBindInput {
    int phase;
    int program_uses_attrib;
    int uses_current_value;
    int has_attrib_binding;
    int32_t mapped_index;
    uint32_t max_metal_slots;
    int offsets_valid;          /* mglRenderAttribOffsetsValid */
    int span_status;            /* MGL_ATTRIB_SPAN_* (0=ok, -2=overflow) */
    int conversion_kind;        /* MGL_ATTRIB_CONV_* */
    int already_present;
    int has_mtl_data;
    int mtl_usable;
    uint64_t binding_offset;
    uint64_t metal_len;
    int absolute_vertex_offsets;
    int binding_state_valid;
    int buffer_matches; /* last-bound matches candidate @ metal_bind_offset */
} MGLAttribBindInput;

typedef struct MGLAttribBindPlan {
    uint32_t action;
    uint32_t reason;
    uint32_t metal_slot;
    uint64_t metal_bind_offset;
    int mark_present;
} MGLAttribBindPlan;

int mglBindingStagePlanAttribEntry(const MGLAttribBindInput *in,
                                   MGLAttribBindPlan *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BINDING_STAGE_H */
