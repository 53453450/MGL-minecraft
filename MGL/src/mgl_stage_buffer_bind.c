/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_stage_buffer_bind.c — the per-stage buffer binding drivers moved out of
 * MGLRenderer+BindingState.m (P0-1, log 129).
 *
 * The method-level macros of the Objective-C header became static functions
 * over one small context struct (the same translation mgl_compute_bind.c used):
 *
 *   MGL_BIND_SNAP_FLUSH / _COLLECT_BUFFER / _COLLECT_BYTES  -> mglSbSnap*
 *   MGL_BIND_STAGE_EMIT_BUFFER / _EMIT_BYTES / _UPDATE /
 *   _PERF_SKIP / _CLEAR_BINDING                             -> mglSbStage*
 *   the two per-method macro families MGL_SMB_* / MGL_SFB_* -> the same
 *                                                              mglSbStage* calls
 *
 * `self`, `ctx`, `_backend`, `_bindingStateOwner` and `_tessellation` arrive
 * through MGLRendererStateAreas; `(__bridge id)` becomes the plain handle.
 *
 * OWNERSHIP (the rule of log 128): every `id` local of the methods was an ARC
 * retain.  The only such local here is `isolated`, and the port that replaces
 * -isolatedStageBindingBufferForMap:… hands back a **+1** (the method returned
 * +0), so the twin releases it once the copy-back recorder and the binding emit
 * have taken their own references.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_stage_buffer_bind.h"
#include "mgl_stage_copy_back.h"
#include "mgl_renderer_ports.h"     /* state areas, isolated buffer, copy-back */
#include "mgl_renderer_backend.h"   /* program binding sizes/counts, fallback buffer */
#include "mgl_render.h"             /* snapshot encode, slot + map predicates */
#include "mgl_binding_stage.h"      /* the stage-bind plan API */
#include "mgl_binding_state_ops.h"  /* invalidate-last-bound helpers */
#include "mgl_shader_resource.h"    /* client binding / metal slot per element */
#include "mgl_buffer_slots.h"       /* kMGLMaxMetalVertexBufferCount, … */
#include "mgl_buffer_map.h"         /* mglNoteBufferEncoded */
#include "mgl_texture_bind.h"       /* mglRendererBindMTLBuffer */
#include "mgl_metal_ref.h"          /* mglSafeReleaseMetalObj */
#include "mgl_frame_activity.h"     /* MGL_PERF_INC + the bind counters */

/* Declared next to its definition in the Objective-C MGLRenderer+Draw_Private.h
 * (which a .c file cannot include); repeated here the way mgl_renderer_ports.c
 * repeats its prototypes.  [self isolatedStageBindingBufferForMap:…] is the
 * port of mgl_renderer_ports.h instead. */
extern Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate,
                                             const char *where, size_t slot);

/* The two static inline predicates of MGLRenderer_Private.h, in C. */
static int mglSbBindingStateIsValid(void *owner)
{
    uint32_t valid = 0;
    return owner && mglRenderBindingGetValid(owner, &valid) == 0 && valid;
}

static int mglSbBindingStateBufferMatches(void *owner, uint32_t stage,
                                          void *buffer, uint64_t offset,
                                          uint32_t index)
{
    void *current = NULL;
    uint64_t current_offset = 0;
    return owner && mglRenderBindingGetBuffer(owner, stage, index, &current,
                                              &current_offset) == 0 &&
           current == buffer && current_offset == offset;
}

/* -[MGLRenderer …] bound-verbose logging: the Objective-C header's
 * kMGLVerboseBindLogs macro is getenv("MGL_VERBOSE_BIND") != NULL. */
static int mglSbVerboseBindLogs(void)
{
    return getenv("MGL_VERBOSE_BIND") != NULL;
}

/* The .m's mglBindingStateBufferLength. */
static uint64_t mglSbBufferLength(void *buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(buffer, &info) == 0 ? info.length
                                                                : 0u;
}

/* The two constants the Objective-C header defines (kMGLMinimumStageBindingSize
 * is the C header mgl_buffer_slots.h). */
#define kMGLStageBindingStackScratchSize 4096u
#define kMGLDefaultStageFallbackBufferSize 4096u

/* === the macro twins ====================================================== */

static void mglSbSnapFlush(MGLRenderBindingSnapshot *snap, int frag,
                           size_t *scratch_used,
                           const MGLEncodeContext *enc_ctx)
{
    uint32_t *count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
    if (*count > 0) {
        mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
            enc_ctx->render_encoder_owner, snap, NULL, 0);
        *snap = (MGLRenderBindingSnapshot){0};
        *scratch_used = 0;
    }
}

static void mglSbSnapCollectBuffer(MGLRenderBindingSnapshot *snap, int frag,
                                   size_t *scratch_used,
                                   const MGLEncodeContext *enc_ctx,
                                   uint32_t slot, const void *buf_ptr,
                                   uint64_t offset)
{
    uint32_t *count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
    MGLRenderBindingOp *ops =
        frag ? snap->fragment_ops : snap->vertex_ops;
    if (*count >= MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
        mglSbSnapFlush(snap, frag, scratch_used, enc_ctx);
        count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
        ops = frag ? snap->fragment_ops : snap->vertex_ops;
    }
    ops[(*count)++] = (MGLRenderBindingOp){0u, (uint32_t)slot, offset,
                                           (void *)buf_ptr, NULL, 0u};
}

static void mglSbSnapCollectBytes(MGLRenderBindingSnapshot *snap, int frag,
                                  uint8_t *scratch, size_t *scratch_used,
                                  size_t scratch_cap,
                                  const MGLEncodeContext *enc_ctx,
                                  uint32_t slot, const void *src, size_t len)
{
    uint32_t *count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
    MGLRenderBindingOp *ops =
        frag ? snap->fragment_ops : snap->vertex_ops;
    if (*scratch_used + len > scratch_cap ||
        *count >= MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
        mglSbSnapFlush(snap, frag, scratch_used, enc_ctx);
        count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
        ops = frag ? snap->fragment_ops : snap->vertex_ops;
    }
    uint8_t *dst = scratch + *scratch_used;
    memcpy(dst, src, len);
    *scratch_used += len;
    ops[(*count)++] =
        (MGLRenderBindingOp){1u, (uint32_t)slot, 0, NULL, dst, (uint32_t)len};
}

static void mglSbStageEmitBuffer(const MGLEncodeContext *enc_ctx, int use_snap,
                                 int is_fragment, int frag,
                                 MGLRenderBindingSnapshot *snap,
                                 size_t *scratch_used, uint32_t slot,
                                 const void *buf_ptr, uint64_t offset)
{
    if (use_snap) {
        mglSbSnapCollectBuffer(snap, frag, scratch_used, enc_ctx, slot, buf_ptr,
                               offset);
    } else if (is_fragment) {
        (void)mglRenderSetRenderBufferForOwner(
            enc_ctx->render_encoder_owner, (void *)buf_ptr, offset,
            MGL_RENDER_BINDING_STAGE_FRAGMENT, slot);
    } else {
        (void)mglRenderSetRenderBufferForOwner(
            enc_ctx->render_encoder_owner, (void *)buf_ptr, offset,
            MGL_RENDER_BINDING_STAGE_VERTEX, slot);
    }
}

static void mglSbStageEmitBytes(const MGLEncodeContext *enc_ctx, int use_snap,
                                int is_fragment, int frag,
                                MGLRenderBindingSnapshot *snap,
                                uint8_t *scratch, size_t *scratch_used,
                                size_t scratch_cap, uint32_t slot,
                                const void *src, size_t len)
{
    if (use_snap) {
        mglSbSnapCollectBytes(snap, frag, scratch, scratch_used, scratch_cap,
                              enc_ctx, slot, src, len);
    } else if (is_fragment) {
        (void)mglRenderSetRenderBytesForOwner(
            enc_ctx->render_encoder_owner, src, len,
            MGL_RENDER_BINDING_STAGE_FRAGMENT, slot);
    } else {
        (void)mglRenderSetRenderBytesForOwner(
            enc_ctx->render_encoder_owner, src, len,
            MGL_RENDER_BINDING_STAGE_VERTEX, slot);
    }
}

/* MGL_BIND_STAGE_UPDATE: `owner` is the binding-state owner slot value. */
static void mglSbStageUpdate(int frag, void *owner, void *buf,
                             uint64_t offset, uint32_t slot)
{
    if (frag) {
        mglRenderBindingUpdateFragmentBuffer(owner, buf, offset, slot);
        MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
    } else {
        mglRenderBindingUpdateVertexBuffer(owner, buf, offset, slot);
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    }
}

static void mglSbStagePerfSkip(int frag)
{
    if (frag) {
        MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
    } else {
        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
    }
}

static void mglSbStageClearBinding(int frag, void *owner, uint32_t slot)
{
    if (frag) {
        mglRenderBindingClearFragmentBuffer(owner, slot);
    } else {
        mglRenderBindingClearVertexBuffer(owner, slot);
    }
}

/* === -bindStageBufferMapEntriesForStage:… ================================= */

bool mglBindingStateBindStageBufferMapEntries(
    void *renderer, int shader_stage, int is_fragment, BufferMapList *map_list,
    bool *any_binding_present, bool *base_binding_present,
    const bool *attrib_binding_reserved, const MGLEncodeContext *enc_ctx,
    MGLRenderBindingSnapshot *binding_snapshot, uint8_t *byte_scratch,
    size_t *byte_scratch_used, size_t byte_scratch_capacity, int use_snapshot,
    uint32_t max_metal_slots, int allow_isolate_when_gpu,
    int needs_copy_back_on_isolate)
{
    if (!renderer || !map_list || !any_binding_present || !base_binding_present ||
        !enc_ctx || !byte_scratch || !byte_scratch_used) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    void *binding_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;

    const int frag = is_fragment ? 1 : 0;
    const int use_snap = use_snapshot && binding_snapshot != NULL;
    const uint32_t metal_stage = is_fragment
                                     ? MGL_RENDER_BINDING_STAGE_FRAGMENT
                                     : MGL_RENDER_BINDING_STAGE_VERTEX;
    MGLRenderBindingSnapshot *snap = binding_snapshot;
    size_t scratch_used = *byte_scratch_used;

    GLuint map_count = (GLuint)mglBindingStageClampMapCount(
        (uint32_t)map_list->count, (uint32_t)MAX_MAPPED_BUFFERS, NULL);

    for (GLuint i = 0; i < map_count; i++) {
        BufferMap *map = &map_list->buffers[i];
        if (is_fragment && mglSbVerboseBindLogs()) {
            fprintf(stderr,
                    "MGL FBIND slot=%u candidate=%p mask=0x%x baseIndex=%u "
                    "offset=%lld\n",
                    i, (void *)map->buf, map->attribute_mask,
                    map->buffer_base_index, (long long)map->offset);
        }
        Buffer *ptr = mglRendererGetValidatedBuffer(
            areas.ctx, map->buf, "bindStageBufferMapEntriesForStage", (size_t)i);
        GLintptr offset = map->offset;
        bool is_base_binding =
            mglRenderBufferMapIsBaseBinding(map->attribute_mask) != 0;
        GLuint gl_binding_index = map->buffer_base_index;
        int64_t metal_resolved =
            map->has_metal_binding
                ? (int64_t)map->metal_binding_index
                : (int64_t)mglRendererGetProgramMetalBufferIndexForStage(
                      areas.ctx, shader_stage, gl_binding_index);

        size_t reflected_required_bytes = 0;
        if (is_base_binding && gl_binding_index < MAX_BINDABLE_BUFFERS) {
            reflected_required_bytes =
                map->has_metal_binding
                    ? mglRendererGetProgramBindingRequiredSize(
                          areas.ctx, shader_stage, (int)map->resource_type,
                          (int)map->resource_index)
                    : mglRendererGetProgramBindingRequiredSizeForStage(
                          areas.ctx, shader_stage, gl_binding_index);
        }
        uint64_t visible_cpu =
            ptr ? (uint64_t)mglBufferMapVisibleBackingBytes(
                      map, ptr->data.buffer_size)
                : 0u;

        MGLStageBufferBindInput bin = {0};
        mglBindingStageFillMapEntryInput(
            &bin, frag, MGL_SB_PHASE_PRE_MTL, is_base_binding ? 1 : 0,
            map->has_metal_binding ? 1 : 0, (int32_t)map->metal_binding_index,
            (int32_t)map->buffer_base_index, (uint32_t)map->resource_type,
            map->offset, ptr ? ptr->size : -1, ptr ? 1 : 0,
            ptr && ptr->data.buffer_data ? 1 : 0,
            ptr && ptr->data.mtl_data ? 1 : 0,
            ptr ? (const void *)(uintptr_t)ptr->data.buffer_data : NULL,
            ptr ? ptr->data.mtl_data : NULL,
            ptr && mglRenderBufferHasCPUDirty(ptr->data.dirty_bits) ? 1 : 0,
            ptr && ptr->gpu_write_target ? 1 : 0, allow_isolate_when_gpu ? 1 : 0,
            0, max_metal_slots, (uint32_t)MAX_BINDABLE_BUFFERS,
            (uint32_t)reflected_required_bytes,
            (uint32_t)kMGLMinimumStageBindingSize,
            (uint32_t)kMGLStageBindingStackScratchSize, visible_cpu,
            mglBufferMapVisibleSize(map));
        if (!is_fragment && is_base_binding &&
            mglRenderBufferSlotInRange((int32_t)metal_resolved,
                                       max_metal_slots) &&
            attrib_binding_reserved &&
            attrib_binding_reserved[(size_t)metal_resolved]) {
            bin.attrib_slot_reserved = 1;
        }
        if (is_base_binding) {
            bin.has_metal_binding = 1;
            bin.metal_binding_index = (int32_t)metal_resolved;
        }

        MGLStageBufferBindPlan plan = {0};
        if (mglBindingStagePlanMapEntry(&bin, &plan) != 0) {
            continue;
        }
        uint32_t binding_index = plan.metal_slot;
        if (plan.mark_base_present && gl_binding_index < MAX_BINDABLE_BUFFERS) {
            base_binding_present[gl_binding_index] = true;
        }

        if (plan.action == MGL_SB_ACTION_SKIP) {
            continue;
        }
        if (plan.action == MGL_SB_ACTION_CLEAR) {
            if (is_fragment && !ptr) {
                map->buf = NULL;
            }
            mglSbStageEmitBuffer(enc_ctx, use_snap, is_fragment, frag, snap,
                                 &scratch_used, binding_index, NULL, 0);
            mglSbStageClearBinding(frag, binding_owner, binding_index);
            continue;
        }
        if (plan.action == MGL_SB_ACTION_INLINE_BYTES) {
            uint8_t padded[kMGLStageBindingStackScratchSize];
            const void *inline_bytes = mglBindingStageInlineBytesSrc(
                padded, (uint32_t)sizeof(padded),
                (const void *)((const uint8_t *)bin.cpu_ptr +
                               plan.inline_src_offset),
                plan.inline_visible, plan.inline_length);
            mglSbStageEmitBytes(enc_ctx, use_snap, is_fragment, frag, snap,
                                byte_scratch, &scratch_used,
                                byte_scratch_capacity, binding_index,
                                inline_bytes, plan.inline_length);
            if (use_snap) {
                *byte_scratch_used = scratch_used;
            }
            if (plan.invalidate_last_bound) {
                if (is_fragment) {
                    mglBindingInvalidateLastBoundFragmentBufferAtIndex(
                        renderer, binding_index);
                } else {
                    mglBindingInvalidateLastBoundVertexBufferAtIndex(
                        renderer, binding_index);
                }
            }
            if (plan.mark_any_present) {
                any_binding_present[binding_index] = true;
            }
            if (plan.clear_cpu_dirty_if_no_mtl && ptr && !ptr->data.mtl_data) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            if (plan.clear_cpu_dirty && ptr) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            continue;
        }

        if (plan.action != MGL_SB_ACTION_NEED_MTL) {
            continue;
        }

        if (!(is_fragment && plan.use_mtl_as_inline_src)) {
            if (!ptr->data.mtl_data) {
                mglRendererBindMTLBuffer(renderer, ptr);
            } else if (mglRenderBufferHasCPUDirty(ptr->data.dirty_bits)) {
                (void)mglRendererUpdateDirtyBuffer(renderer, ptr);
            }
        }

        int mtl_usable = mglBindingStagePostMtlUsable(
            frag, ptr->data.mtl_data, plan.use_mtl_as_inline_src ? 1 : 0);
        void *buffer = mtl_usable ? ptr->data.mtl_data : NULL;
        size_t metal_len = buffer ? (size_t)mglSbBufferLength(buffer) : 0u;
        size_t available_bytes =
            buffer ? mglBufferMapVisibleBackingBytes(map, metal_len) : 0u;
        mglBindingStageFillMapEntryPostMtl(
            &bin, ptr->data.mtl_data ? 1 : 0, ptr->data.mtl_data, mtl_usable,
            (uint64_t)metal_len, (uint64_t)available_bytes,
            mglSbBindingStateIsValid(binding_owner) ? 1 : 0,
            buffer && mglSbBindingStateBufferMatches(
                          binding_owner, metal_stage, buffer, (uint64_t)offset,
                          (uint32_t)plan.metal_slot)
                ? 1
                : 0);
        if (mglBindingStagePlanMapEntry(&bin, &plan) != 0) {
            continue;
        }
        binding_index = plan.metal_slot;

        if (plan.action == MGL_SB_ACTION_CLEAR) {
            mglSbStageEmitBuffer(enc_ctx, use_snap, is_fragment, frag, snap,
                                 &scratch_used, binding_index, NULL, 0);
            mglSbStageClearBinding(frag, binding_owner, binding_index);
            if (plan.clear_cpu_dirty && ptr) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            continue;
        }
        if (plan.action == MGL_SB_ACTION_ISOLATE) {
            /* +1 out of the port; the recorder and the emit take their own
             * references, so this one is released below (log 128 rule). */
            void *isolated = mglBufferIsolatedStageBinding(
                renderer, map, buffer, plan.required_bytes);
            if (!isolated) {
                fprintf(stderr,
                        "MGL WARNING: %s failed to isolate undersized buffer=%u "
                        "slot=%lu required=%u available=%u\n",
                        is_fragment ? "FBIND" : "VBIND", ptr->name,
                        (unsigned long)binding_index, plan.required_bytes,
                        plan.available_bytes);
                mglSbStageEmitBuffer(enc_ctx, use_snap, is_fragment, frag, snap,
                                     &scratch_used, binding_index, NULL, 0);
                mglSbStageClearBinding(frag, binding_owner, binding_index);
                continue;
            }
            if (needs_copy_back_on_isolate && plan.needs_copy_back && buffer &&
                plan.available_bytes > 0 &&
                !mglRecordStageBindingCopyBack(
                    renderer, &areas.tessellation->nativeTESCopyBacks,
                    binding_index, isolated, buffer, ptr, (uint64_t)offset,
                    plan.available_bytes)) {
                mglSafeReleaseMetalObj(&isolated);
                if (use_snap) {
                    *byte_scratch_used = scratch_used;
                }
                return false;
            }
            mglSbStageEmitBuffer(enc_ctx, use_snap, is_fragment, frag, snap,
                                 &scratch_used, binding_index, isolated, 0);
            /* The port hands back +1 and the Objective-C method leaned on the
             * autorelease pool to keep it alive until the snapshot was
             * encoded; C has no pool, and mgl_compute_bind.c resolves the same
             * situation by flushing while the buffer is still alive.  Encoding
             * earlier is order-preserving (same ops, same encoder). */
            if (use_snap) {
                mglSbSnapFlush(snap, frag, &scratch_used, enc_ctx);
                *byte_scratch_used = scratch_used;
            }
            mglSbStageUpdate(frag, binding_owner, isolated, 0, binding_index);
            mglSafeReleaseMetalObj(&isolated);
            any_binding_present[binding_index] = true;
            continue;
        }
        if (plan.action == MGL_SB_ACTION_SKIP_MATCHED) {
            mglSbStagePerfSkip(frag);
            any_binding_present[binding_index] = true;
            if (plan.clear_cpu_dirty && ptr) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            continue;
        }
        if (plan.action == MGL_SB_ACTION_BIND_BUFFER) {
            mglSbStageEmitBuffer(enc_ctx, use_snap, is_fragment, frag, snap,
                                 &scratch_used, binding_index, buffer,
                                 (uint64_t)plan.bind_offset);
            mglSbStageUpdate(frag, binding_owner, buffer,
                             (uint64_t)plan.bind_offset, binding_index);
            if (!is_fragment) {
                mglNoteBufferEncoded(ptr);
            }
            any_binding_present[binding_index] = true;
            if (plan.clear_cpu_dirty && ptr) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            continue;
        }
    }

    if (use_snap) {
        *byte_scratch_used = scratch_used;
    }
    return true;
}

/* === -bindStageFallbackBuffersForStage:… ================================== */

void mglBindingStateBindStageFallbackBuffers(
    void *renderer, int shader_stage, int is_fragment, Program *active_program,
    bool *any_binding_present, bool *base_binding_present,
    const MGLEncodeContext *enc_ctx, MGLRenderBindingSnapshot *binding_snapshot,
    int use_snapshot, uint32_t max_metal_slots, int enable_all_slot_fill)
{
    if (!renderer || !enc_ctx) {
        return;
    }

    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    void *binding_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;

    MGLRenderBindingSnapshot *snap = binding_snapshot;
    const int use_snap = use_snapshot && snap != NULL;
    const int frag = is_fragment ? 1 : 0;
    size_t scratch_dummy = 0;
    const uint32_t metal_stage = is_fragment
                                     ? MGL_RENDER_BINDING_STAGE_FRAGMENT
                                     : MGL_RENDER_BINDING_STAGE_VERTEX;

    void *fallback_binding_buffer = mglRendererBackendGetFallbackBindingBuffer(
        areas.backend, kMGLDefaultStageFallbackBufferSize);

    uint32_t resource_types[MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT];
    uint32_t resource_type_count = mglBindingStageFallbackResourceTypes(
        resource_types, MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT);
    for (uint32_t t = 0; t < resource_type_count; t++) {
        int resource_type = (int)resource_types[t];
        int count = mglRendererGetProgramBindingCount(areas.ctx, shader_stage,
                                                     resource_type);
        Program *program = active_program;
        for (int i = 0; i < count; i++) {
            if (!program || resource_type < 0 ||
                resource_type >= MGL_MAX_SHADER_RESOURCES ||
                i >= (int)program->shader_resources_list[shader_stage]
                             [resource_type]
                                 .count) {
                continue;
            }
            MGLShaderResource *resource =
                &program->shader_resources_list[shader_stage][resource_type]
                     .list[i];
            GLuint element_count =
                mglStageBufferResourceElementCount(resource_type, resource);
            for (GLuint element = 0; element < element_count; element++) {
                GLuint client_binding = mglClientBufferBindingForResourceElement(
                    resource_type, resource, element);
                if (client_binding >= MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                int64_t metal_binding = (int64_t)mglMetalResourceSlotForElement(
                    resource, element);
                if (metal_binding < 0 ||
                    metal_binding >= (int64_t)max_metal_slots) {
                    continue;
                }
                size_t slot = (size_t)metal_binding;
                int matches =
                    mglSbBindingStateIsValid(binding_owner) &&
                    mglSbBindingStateBufferMatches(binding_owner, metal_stage,
                                                   fallback_binding_buffer, 0,
                                                   (uint32_t)slot);
                uint32_t action = mglBindingStagePlanFallbackSlot(
                    any_binding_present[slot] ? 1 : 0,
                    fallback_binding_buffer ? 1 : 0,
                    mglSbBindingStateIsValid(binding_owner) ? 1 : 0,
                    matches ? 1 : 0);
                if (action == MGL_FB_SLOT_SKIP) {
                    continue;
                }
                if (action == MGL_FB_SLOT_EMIT) {
                    mglSbStageEmitBuffer(enc_ctx, use_snap, is_fragment, frag,
                                         snap, &scratch_dummy, (uint32_t)slot,
                                         fallback_binding_buffer, 0);
                    mglSbStageUpdate(frag, binding_owner,
                                     (void *)fallback_binding_buffer, 0,
                                     (uint32_t)slot);
                } else {
                    mglSbStagePerfSkip(frag);
                }
                base_binding_present[client_binding] = true;
                any_binding_present[slot] = true;
            }
        }
    }

    if (enable_all_slot_fill && fallback_binding_buffer) {
        for (size_t s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            int matches =
                mglSbBindingStateIsValid(binding_owner) &&
                mglSbBindingStateBufferMatches(binding_owner, metal_stage,
                                               fallback_binding_buffer, 0,
                                               (uint32_t)s);
            uint32_t action = mglBindingStagePlanFallbackSlot(
                any_binding_present[s] ? 1 : 0, 1,
                mglSbBindingStateIsValid(binding_owner) ? 1 : 0, matches ? 1 : 0);
            if (action == MGL_FB_SLOT_SKIP) {
                continue;
            }
            if (action == MGL_FB_SLOT_EMIT) {
                mglSbStageEmitBuffer(enc_ctx, use_snap, is_fragment, frag, snap,
                                     &scratch_dummy, (uint32_t)s,
                                     fallback_binding_buffer, 0);
                mglSbStageUpdate(frag, binding_owner,
                                 (void *)fallback_binding_buffer, 0,
                                 (uint32_t)s);
            } else {
                mglSbStagePerfSkip(frag);
            }
            any_binding_present[s] = true;
        }
    }

    if (use_snap) {
        mglSbSnapFlush(snap, frag, &scratch_dummy, enc_ctx);
    }
}
