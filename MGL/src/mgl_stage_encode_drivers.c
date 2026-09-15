/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_stage_encode_drivers.c — the vertex/fragment stage binding drivers moved
 * out of MGLRenderer+BindingState.m (P0-1, log 131).
 *
 * The five methods became four entries plus file-scope state:
 *
 *   -bindVertexBuffersToCurrentRenderEncoder:   -> mglStageEncodeBindVertexBuffers
 *   -bindVertexAttributesFromVAO:…              -> mglStageEncodeBindVertexAttributes (static)
 *   -bindPointSizeParamsIfNeeded:…              -> mglStageEncodeBindPointSizeParams (static)
 *   -bindFragmentBuffersToCurrentRenderEncoder: -> mglStageEncodeBindFragmentBuffers
 *   -finalizeStageBufferPresentMask:…           -> mglStageEncodeFinalizePresentMask (static)
 *
 * Translation rules (the ones the earlier cuts settled): `self` becomes the
 * renderer handle, `ctx` is `areas.ctx`, `_backend`/`_device` come from
 * `areas.backend`, `_batching` from `areas.batching`, `_bindingStateOwner` from
 * `*areas.binding_state_owner`, the per-method macros become the mglSe*
 * functions, `NSLog` becomes fprintf on the same sink, `__FUNCTION__` keeps the
 * exact selector string the Objective-C method passed to the validators (the
 * stderr text is part of the A/B oracle), and NSMutableData becomes a calloc'd
 * zero-filled block (same bytes, same length, freed at the same point).
 *
 * OWNERSHIP (log 128 rule): the only `id` locals here are the packed-current
 * attribute buffer and the converted attribute buffer.  The former is kept
 * alive by the backend cache (the setter takes it) exactly as before; the
 * latter is released through mglBufferReleaseConvertedVertexBuffer the way the
 * method did, and every borrowed Metal handle is used within the iteration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_stage_encode_drivers.h"
#include "mgl_stage_buffer_bind.h"   /* the drivers converted in log 129 */
#include "mgl_renderer_ports.h"      /* state areas */
#include "mgl_renderer_backend.h"    /* packed current-attrib cache, device */
#include "mgl_render.h"              /* snapshot encode, binding-state record */
#include "mgl_binding_stage.h"       /* the attrib plan API */
#include "mgl_binding_policy.h"      /* kMGLMaxMetalVertexBufferCount helpers */
#include "mgl_binding_texture.h"     /* mglBindingTextureRateLogHit */
#include "mgl_buffer_map.h"          /* converted-vertex-buffer facade */
#include "mgl_vertex_attrib_query.h" /* program/attrib predicates */
#include "mgl_vertex_attrib_binding.h" /* mglRendererResolveVertexAttribBinding */
#include "mgl_vertex_format.h"       /* mglVertexFormatName */
#include "mgl_trace_strategy.h"      /* mglProgramNeedsTraceLog */
#include "mgl_trace_log.h"           /* mglTraceLog, mglTraceClockNS, kMGLDiagnosticStateLogs */
#include "mgl_texture_bind.h"        /* mglRendererBindMTLBuffer */
#include "mgl_env_flag.h"            /* mgl_env_flag_enabled */
#include "mgl_binding_state_ops.h"   /* invalidate-last-bound helpers */
#include "mgl_frame_activity.h"      /* MGL_PERF_INC + the bind counters */
#include "glm_limits.h"              /* MAX_ATTRIBS, MAX_MAPPED_BUFFERS */
#include "mgl_size_constants.h"      /* kMGLMaxBufferSlots */
#include "mgl_buffer_slots.h"        /* kMGLPointSizeBufferIndex */
#include "mgl_types_state.h"         /* mglMarkRendererDirtyBits (tbind gate) */

/* Declared next to their definitions in Objective-C headers a .c file cannot
 * include; repeated here the way mgl_renderer_ports.c repeats its prototypes.
 * mglBindingStateIsValid / …BufferMatches are
 * the one-line predicates of MGLRenderer_Private.h, repeated as mglSe*
 * functions below. */
extern Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);
extern void mglLogLoopHeartbeat(const char *tag, uint64_t call_count,
                                double now_seconds, double *last_call_seconds,
                                uint64_t *last_call_count,
                                double warn_gap_seconds);

/* Declared next to its definition in the Objective-C MGLRenderer+Draw_Private.h. */
extern size_t mglRendererBuildCurrentVertexAttribBytes(GLMContext ctx,
                                                       GLuint attribute,
                                                       const VertexAttrib *attrib,
                                                       uint8_t bytes[16]);

/* The Objective-C header's constants and one-line predicates, in C. */
#define MGL_BINDING_RESOURCE_STORAGE_SHARED 0u
#define kMGLEnableVertexAllSlotFallback 1

static bool mglSeShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

/* MGL_STATE() from MGLRenderer_Private.h, in C. */
static GLMState *mglSeState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* kMGLVerboseBindLogs: getenv("MGL_VERBOSE_BIND") != NULL. */
static int mglSeVerboseBindLogs(void)
{
    return getenv("MGL_VERBOSE_BIND") != NULL;
}

static int mglSeBindingStateTextureSlotCount(void *owner)
{
    uint64_t mask[2] = {0, 0};
    if (!owner || mglRenderBindingGetTextureSlotMask(owner, mask) != 0) {
        return 0;
    }
    return __builtin_popcountll(mask[0]) + __builtin_popcountll(mask[1]);
}

static int mglSeBindingStateIsValid(void *owner)
{
    uint32_t valid = 0;
    return owner && mglRenderBindingGetValid(owner, &valid) == 0 && valid;
}

static int mglSeBindingStateBufferMatches(void *owner, uint32_t stage,
                                          void *buffer, uint64_t offset,
                                          uint32_t index)
{
    void *current = NULL;
    uint64_t current_offset = 0;
    return owner && mglRenderBindingGetBuffer(owner, stage, index, &current,
                                              &current_offset) == 0 &&
           current == buffer && current_offset == offset;
}

/* The .m's static predicates. */
static int mglSeHasActiveEncoder(const MGLEncodeContext *enc_ctx)
{
    return enc_ctx && enc_ctx->render_encoder_owner != NULL ? 1 : 0;
}

static uint64_t mglSeBufferLength(void *buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(buffer, &info) == 0 ? info.length
                                                                : 0u;
}

/* +1 buffer, or NULL (the .m's mglBindingStateCreateBufferWithBytes). */
static void *mglSeCreateBufferWithBytes(void *device, const void *bytes,
                                       size_t length, uint64_t options)
{
    void *buffer = NULL;
    (void)device;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL, &buffer) ==
            0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

/* The .m's kMGLCurrentAttribRepeatCount / ValueBytes / PoolStride (macros of
 * MGLRenderer_Private.h). */
#define kMGLSeCurrentAttribRepeatCount 4096u
#define kMGLSeCurrentAttribValueBytes 16u
#define kMGLSeCurrentAttribPoolStride \
    ((uint32_t)kMGLSeCurrentAttribRepeatCount * kMGLSeCurrentAttribValueBytes)

/* === the macro twins ====================================================== */

static void mglSeSnapFlush(MGLRenderBindingSnapshot *snap, int frag,
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

static void mglSeSnapCollectBuffer(MGLRenderBindingSnapshot *snap, int frag,
                                   size_t *scratch_used,
                                   const MGLEncodeContext *enc_ctx,
                                   uint32_t slot, const void *buf_ptr,
                                   uint64_t offset)
{
    uint32_t *count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
    MGLRenderBindingOp *ops = frag ? snap->fragment_ops : snap->vertex_ops;
    if (*count >= MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
        mglSeSnapFlush(snap, frag, scratch_used, enc_ctx);
        count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
        ops = frag ? snap->fragment_ops : snap->vertex_ops;
    }
    ops[(*count)++] = (MGLRenderBindingOp){0u, (uint32_t)slot, offset,
                                           (void *)buf_ptr, NULL, 0u};
}

static void mglSeSnapCollectBytes(MGLRenderBindingSnapshot *snap, int frag,
                                  uint8_t *scratch, size_t *scratch_used,
                                  size_t scratch_cap,
                                  const MGLEncodeContext *enc_ctx,
                                  uint32_t slot, const void *src, size_t len)
{
    uint32_t *count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
    MGLRenderBindingOp *ops = frag ? snap->fragment_ops : snap->vertex_ops;
    if (*scratch_used + len > scratch_cap ||
        *count >= MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
        mglSeSnapFlush(snap, frag, scratch_used, enc_ctx);
        count = frag ? &snap->fragment_op_count : &snap->vertex_op_count;
        ops = frag ? snap->fragment_ops : snap->vertex_ops;
    }
    uint8_t *dst = scratch + *scratch_used;
    memcpy(dst, src, len);
    *scratch_used += len;
    ops[(*count)++] =
        (MGLRenderBindingOp){1u, (uint32_t)slot, 0, NULL, dst, (uint32_t)len};
}

/* The .m's mglBindingStateEmitAttribBuffer (vertex stage only). */
static int mglSeEmitAttribBuffer(void *binding_state_owner,
                                 const MGLEncodeContext *enc_ctx,
                                 MGLRenderBindingSnapshot *snapshot,
                                 int use_snapshot, size_t *scratch_used,
                                 size_t slot, void *buffer, size_t offset,
                                 bool *any_binding_present)
{
    int valid = mglSeBindingStateIsValid(binding_state_owner) ? 1 : 0;
    int matches =
        valid && mglSeBindingStateBufferMatches(binding_state_owner,
                                                MGL_RENDER_BINDING_STAGE_VERTEX,
                                                buffer, offset, (uint32_t)slot)
            ? 1
            : 0;
    int emitted = 0;
    if (mglBindingStageAttribNeedsEmit(valid, matches)) {
        if (use_snapshot && snapshot) {
            mglSeSnapCollectBuffer(snapshot, 0, scratch_used, enc_ctx,
                                   (uint32_t)slot, buffer, offset);
        } else {
            (void)mglRenderSetRenderBufferForOwner(
                enc_ctx->render_encoder_owner, buffer, offset,
                MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)slot);
        }
        mglRenderBindingUpdateVertexBuffer(binding_state_owner, buffer, offset,
                                           (uint32_t)slot);
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        emitted = 1;
    } else {
        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
    }
    if (any_binding_present) {
        any_binding_present[slot] = true;
    }
    return emitted;
}

/* The .m's MGL_VPS_EMIT_BYTES macro (vertex stage). */
static void mglSeEmitVertexBytes(const MGLEncodeContext *enc_ctx,
                                 MGLRenderBindingSnapshot *snapshot,
                                 int use_snapshot, uint8_t *byte_scratch,
                                 size_t *byte_scratch_used,
                                 size_t byte_scratch_capacity, uint32_t slot,
                                 const void *src, size_t len)
{
    if (use_snapshot) {
        mglSeSnapCollectBytes(snapshot, 0, byte_scratch, byte_scratch_used,
                              byte_scratch_capacity, enc_ctx, slot, src, len);
    } else {
        (void)mglRenderSetRenderBytesForOwner(
            enc_ctx->render_encoder_owner, src, len,
            MGL_RENDER_BINDING_STAGE_VERTEX, slot);
    }
}

/* -finalizeStageBufferPresentMask:… */
static void mglSeFinalizePresentMask(
    void *renderer, int is_fragment, const bool *any_binding_present,
    const bool *base_binding_present, const bool *attrib_binding_reserved,
    uint64_t bind_call, uint32_t map_count, uint64_t start_clock)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    void *binding_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;

    uint32_t mask = mglBindingStageBuildPresentMask(
        (const uint8_t *)any_binding_present, (uint32_t)kMGLMaxBufferSlots);
    if (is_fragment) {
        mglRenderBindingOrFragmentBufferMask(binding_owner, mask);
    } else {
        mglRenderBindingOrVertexBufferMask(binding_owner, mask);
    }
    if (mgl_env_flag_enabled("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_trace[2] = {0, 0};
        if ((++s_trace[is_fragment ? 1 : 0] % 500) == 1) {
            uint32_t active = mglBindingStageCountPresent(
                (const uint8_t *)any_binding_present,
                (uint32_t)kMGLMaxBufferSlots);
            if (is_fragment) {
                fprintf(stderr,
                        "MGL SPARSE FBIND: fbuf=0x%x(%u/31) texSlots=%d/128\n",
                        mask, active,
                        mglSeBindingStateTextureSlotCount(binding_owner));
            } else {
                fprintf(stderr, "MGL SPARSE VBIND: mask=0x%x active=%u/31\n",
                        mask, active);
            }
        }
    }
    if (kMGLDiagnosticStateLogs && mglSeShouldTraceCall(bind_call)) {
        uint32_t slot_cap = is_fragment ? (uint32_t)MAX_BINDABLE_BUFFERS
                                        : (uint32_t)kMGLMaxMetalVertexBufferCount;
        uint32_t bound = mglBindingStageCountPresent(
            (const uint8_t *)any_binding_present, slot_cap);
        uint32_t base = mglBindingStageCountPresent(
            (const uint8_t *)base_binding_present,
            (uint32_t)MAX_BINDABLE_BUFFERS);
        double us = (mglTraceClockNS() - start_clock) / 1000.0;
        if (is_fragment) {
            mglTraceLog("fbind.end call=%llu mapCount=%u boundSlots=%u "
                        "baseSlots=%u elapsed=%.1fus",
                        (unsigned long long)bind_call, (unsigned)map_count,
                        bound, base, us);
        } else {
            uint32_t reserved =
                attrib_binding_reserved
                    ? mglBindingStageCountPresent(
                          (const uint8_t *)attrib_binding_reserved,
                          (uint32_t)kMGLMaxMetalVertexBufferCount)
                    : 0u;
            mglTraceLog("vbind.end call=%llu mapCount=%u boundSlots=%u "
                        "reservedSlots=%u baseSlots=%u elapsed=%.1fus",
                        (unsigned long long)bind_call, (unsigned)map_count,
                        bound, reserved, base, us);
        }
    }
}

/* -bindPointSizeParamsIfNeeded:… */
static void mglSeBindPointSizeParams(void *renderer, bool *any_binding_present,
                                     const MGLEncodeContext *enc_ctx,
                                     MGLRenderBindingSnapshot *binding_snapshot,
                                     uint8_t *byte_scratch,
                                     size_t *byte_scratch_used,
                                     size_t byte_scratch_capacity,
                                     int use_snapshot)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);

    int needs_point_size_params = 0;
    MGLRenderBindingSnapshot *snap = binding_snapshot;
    uint8_t *scratch = byte_scratch;
    size_t *scratch_used = byte_scratch_used;
    const int use_snap = use_snapshot && snap != NULL && scratch != NULL &&
                         scratch_used != NULL;
    const int point_size_stages[] = {_VERTEX_SHADER, _TESS_EVALUATION_SHADER,
                                     _GEOMETRY_SHADER};
    for (size_t ps = 0;
         ps < sizeof(point_size_stages) / sizeof(point_size_stages[0]); ps++) {
        Program *point_program = mglResolveProgramForStageFromState(
            areas.ctx, point_size_stages[ps]);
        if (!point_program) {
            continue;
        }
        if (point_program->uses_point_size_params) {
            needs_point_size_params = 1;
            break;
        }
    }
    if (needs_point_size_params) {
        float point_size_params[2] = {
            areas.ctx && mglSeState(&areas)->var.point_size > 0.0f
                ? mglSeState(&areas)->var.point_size
                : 1.0f,
            areas.ctx && mglSeState(&areas)->caps.program_point_size ? 1.0f
                                                                    : 0.0f};
        mglSeEmitVertexBytes(enc_ctx, snap, use_snap, scratch, scratch_used,
                             byte_scratch_capacity,
                             kMGLPointSizeBufferIndex, point_size_params,
                             sizeof(point_size_params));
        mglBindingInvalidateLastBoundVertexBufferAtIndex(
            renderer, kMGLPointSizeBufferIndex);
        any_binding_present[kMGLPointSizeBufferIndex] = true;
    }

    if (use_snap) {
        mglSeSnapFlush(snap, 0, scratch_used, enc_ctx);
    }
}

/* === -bindVertexAttributesFromVAO:… ====================================== */

static bool mglStageEncodeBindVertexAttributes(
    void *renderer, VertexArray *vao, Program *active_program,
    bool attribs_enabled_by_app, int *attrib_binding_index,
    bool *any_binding_present, const MGLEncodeContext *enc_ctx,
    MGLRenderBindingSnapshot *binding_snapshot, int use_snapshot)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    void *binding_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;

    size_t binding_index = 0;
    MGLRenderBindingSnapshot *snapshot = binding_snapshot;
    const int use_snap = use_snapshot && snapshot != NULL;
    size_t scratch_dummy = 0;

    /* Same mapping as generateVertexDescriptorState; plan@C + setVertexBuffer. */
    GLuint max_attribs = MAX_ATTRIBS;
    for (GLuint attrib = 0; attrib < max_attribs; attrib++) {
        int uses_current_value =
            mglRendererVertexAttribUsesCurrentValue(vao, attrib) ? 1 : 0;
        MGLResolvedVertexAttribBinding resolved = {0};
        bool has_attrib_binding = mglRendererResolveVertexAttribBinding(
            areas.ctx, vao, attrib,
            "-[MGLRenderer(Draw) bindVertexAttributesFromVAO:activeProgram:"
            "attribsEnabledByApp:attribBindingIndex:anyBindingPresent:"
            "encodeContext:bindingSnapshot:useSnapshot:]",
            &resolved);
        int mapped_index =
            (attrib < MAX_ATTRIBS) ? attrib_binding_index[attrib] : -1;
        uint32_t planned_format = 0u;
        int effective_normalized = 0;
        int conversion_kind = MGL_ATTRIB_CONV_NONE;

        int offsets_valid = 0, span_status = 0, already_present = 0;
        uint64_t binding_offset = 0;
        if (has_attrib_binding && resolved.buffer) {
            offsets_valid =
                mglRenderAttribOffsetsValid(resolved.binding_offset,
                                            resolved.relativeoffset)
                    ? 1
                    : 0;
            int64_t attr_offset = 0, attr_span = 0, attr_end = 0;
            span_status = mglRenderPlanVertexAttribSpan(
                (int64_t)resolved.binding_offset,
                (int64_t)resolved.relativeoffset, (uint32_t)resolved.attrib->type,
                (uint32_t)resolved.attrib->size, &attr_offset, &attr_span,
                &attr_end);
            if (mglSeVerboseBindLogs() &&
                mglRenderAttribWrittenRangeTracked(resolved.buffer->written_min,
                                                   resolved.buffer->written_max) &&
                mglRenderAttribOutsideWrittenRange(
                    attr_offset, attr_end, resolved.buffer->written_min,
                    resolved.buffer->written_max)) {
                static uint64_t s_vbind_written_range_warning_count = 0;
                if (mglBindingTextureRateLogHit(
                        &s_vbind_written_range_warning_count, 16ull, 4096ull)) {
                    fprintf(stderr,
                            "MGL VBIND WARNING draw: attrib=%u buffer=%u "
                            "attrRange=[%lld,%lld) outside written [%lld,%lld) "
                            "type=0x%x size=%u hit=%llu\n",
                            attrib, resolved.buffer->name, (long long)attr_offset,
                            (long long)attr_end,
                            (long long)resolved.buffer->written_min,
                            (long long)resolved.buffer->written_max,
                            (unsigned)resolved.attrib->type,
                            (unsigned)resolved.attrib->size,
                            (unsigned long long)s_vbind_written_range_warning_count);
                }
            }
            MGLShaderResource *attr_res =
                mglRendererProgramVertexAttribResource(active_program, attrib);
            int needs_conversion = 0;
            mglRenderPlanVertexAttribFormat(
                (uint32_t)resolved.attrib->type, (uint32_t)resolved.attrib->size,
                resolved.attrib->integer ? 1 : 0,
                resolved.attrib->normalized ? 1 : 0,
                mglRendererVertexAttribIsColorInput(active_program, attrib) ? 1
                                                                           : 0,
                attr_res ? (uint32_t)attr_res->gl_type : 0u, &planned_format,
                &needs_conversion, &effective_normalized, &conversion_kind);
            (void)needs_conversion;
            already_present =
                (mapped_index >= 0 &&
                 mapped_index < (int)kMGLMaxMetalVertexBufferCount &&
                 any_binding_present[mapped_index])
                    ? 1
                    : 0;
            binding_offset = (uint64_t)resolved.binding_offset;
        }
        MGLAttribBindInput ain = {0};
        mglBindingStageFillAttribSelectInput(
            &ain,
            mglRendererProgramUsesVertexAttrib(active_program, attrib) ? 1 : 0,
            uses_current_value, has_attrib_binding ? 1 : 0, mapped_index,
            (uint32_t)kMGLMaxMetalVertexBufferCount, offsets_valid, span_status,
            conversion_kind, already_present, binding_offset,
            areas.batching->absoluteVertexBindingOffsets ? 1 : 0);

        MGLAttribBindPlan plan = {0};
        if (mglBindingStagePlanAttribEntry(&ain, &plan) != 0) {
            continue;
        }
        if (plan.action == MGL_ATTR_ACTION_SKIP ||
            plan.action == MGL_ATTR_ACTION_SKIP_ALREADY) {
            if (plan.reason == MGL_ATTR_REASON_BAD_MAP) {
                fprintf(stderr,
                        "MGL ERROR: VBIND attrib=%u unresolved mapping=%d\n",
                        attrib, mapped_index);
            } else if (plan.reason == MGL_ATTR_REASON_NO_BINDING &&
                       !uses_current_value && !has_attrib_binding) {
                /* disabled attrib */
            } else if (plan.reason == MGL_ATTR_REASON_NO_BINDING) {
                fprintf(stderr,
                        "MGL VBIND skip attrib=%u: enabled but buffer is "
                        "invalid\n",
                        attrib);
            }
            continue;
        }
        if (plan.action == MGL_ATTR_ACTION_BLOCK) {
            if (plan.reason == MGL_ATTR_REASON_BAD_OFFSET) {
                fprintf(stderr,
                        "MGL VBIND BLOCK draw: attrib=%u buffer=%u negative "
                        "bindingOffset=%lld relativeOffset=%lld\n",
                        attrib, resolved.buffer->name,
                        (long long)resolved.binding_offset,
                        (long long)resolved.relativeoffset);
            } else {
                fprintf(stderr,
                        "MGL VBIND BLOCK draw: attrib=%u buffer=%u attr span "
                        "overflow (type=0x%x size=%u)\n",
                        attrib, resolved.buffer->name,
                        (unsigned)resolved.attrib->type,
                        (unsigned)resolved.attrib->size);
            }
            if (use_snap) {
                mglSeSnapFlush(snapshot, 0, &scratch_dummy, enc_ctx);
            }
            return false;
        }

        binding_index = (size_t)plan.metal_slot;

        if (plan.action == MGL_ATTR_ACTION_CURRENT) {
            uint8_t pool_values[MAX_ATTRIBS][16];
            memset(pool_values, 0, sizeof(pool_values));
            for (GLuint a = 0; a < (GLuint)MAX_ATTRIBS; a++) {
                uint8_t tmp[16] = {0};
                size_t built = mglRendererBuildCurrentVertexAttribBytes(
                    areas.ctx, a, &vao->attrib[a], tmp);
                if (built == 0u || built > 16u) {
                    continue;
                }
                memcpy(pool_values[a], tmp, 16u);
            }
            void *current_attrib_buffer =
                mglRendererBackendGetPackedCurrentAttribBuffer(
                    areas.backend, pool_values, (uint32_t)sizeof(pool_values),
                    kMGLSeCurrentAttribRepeatCount);
            if (current_attrib_buffer == NULL) {
                size_t pool_bytes =
                    (size_t)MAX_ATTRIBS * kMGLSeCurrentAttribPoolStride;
                /* The method used a zero-filled NSMutableData of this length. */
                uint8_t *pool = (uint8_t *)calloc(1u, pool_bytes);
                if (!pool) {
                    fprintf(stderr,
                            "MGL VBIND skip attrib=%u: packed current pool "
                            "alloc\n",
                            attrib);
                    continue;
                }
                mglRenderPackCurrentAttribPool(
                    (const uint8_t *)pool_values, (uint32_t)MAX_ATTRIBS, pool,
                    (uint64_t)pool_bytes, kMGLSeCurrentAttribRepeatCount,
                    kMGLSeCurrentAttribValueBytes);
                current_attrib_buffer = mglSeCreateBufferWithBytes(
                    mglRendererBackendGetDevice(areas.backend), pool, pool_bytes,
                    MGL_BINDING_RESOURCE_STORAGE_SHARED);
                free(pool);
                if (!current_attrib_buffer ||
                    mglRendererBackendSetPackedCurrentAttribBuffer(
                        areas.backend, pool_values,
                        (uint32_t)sizeof(pool_values),
                        kMGLSeCurrentAttribRepeatCount,
                        current_attrib_buffer) != 0) {
                    fprintf(stderr,
                            "MGL VBIND skip attrib=%u: packed current MTL "
                            "cache\n",
                            attrib);
                    continue;
                }
            }
            mglSeEmitAttribBuffer(binding_owner, enc_ctx, snapshot, use_snap,
                                  &scratch_dummy, binding_index,
                                  current_attrib_buffer, 0, any_binding_present);
            static uint64_t s_trace_file_current_attrib_bind_logs = 0;
            if (mglProgramNeedsTraceLog(active_program) &&
                mglShouldLogTraceFileBindingForProgram(
                    active_program, &s_trace_file_current_attrib_bind_logs)) {
                MGLShaderResource *resource =
                    mglRendererProgramVertexAttribResource(active_program,
                                                           attrib);
                mglTraceLog(
                    "VATTR_BIND_CURRENT_PACKED program=%u attrib=%u resource=%s "
                    "loc=%u metalSlot=%lu poolOffset=%lu "
                    "valueF=(%.6f,%.6f,%.6f,%.6f)",
                    active_program ? (unsigned)active_program->name : 0u,
                    (unsigned)attrib,
                    resource && resource->name ? resource->name : "(unknown)",
                    resource ? (unsigned)resource->location : 0xffffffffu,
                    (unsigned long)binding_index,
                    (unsigned long)((size_t)attrib *
                                    kMGLSeCurrentAttribPoolStride),
                    mglSeState(&areas)->current_vertex_attrib[attrib].f[0],
                    mglSeState(&areas)->current_vertex_attrib[attrib].f[1],
                    mglSeState(&areas)->current_vertex_attrib[attrib].f[2],
                    mglSeState(&areas)->current_vertex_attrib[attrib].f[3]);
            }
            continue;
        }

        Buffer *attrib_buffer = resolved.buffer;
        const VertexAttrib *attrib_state = resolved.attrib;
        if (mglSeVerboseBindLogs()) {
            fprintf(stderr,
                    "MGL VBIND attrib map attrib=%u -> index=%lu buffer=%u "
                    "bindingOffset=%lld table=%d\n",
                    attrib, (unsigned long)binding_index,
                    (unsigned)attrib_buffer->name,
                    (long long)resolved.binding_offset,
                    resolved.uses_binding_table ? 1 : 0);
        }

        if (plan.action == MGL_ATTR_ACTION_CONVERT) {
            MGLShaderResource *conv_res =
                mglRendererProgramVertexAttribResource(active_program, attrib);
            int integer_conv_dst_is_int =
                mglRenderIntegerAttribDstIsInt(conv_res ? conv_res->gl_type : 0u)
                    ? 1
                    : 0;
            size_t converted_stride = 0;
            void *converted_buffer =
                mglBufferCreateConvertedVertexBufferForAttribKind(
                    renderer, ain.conversion_kind, attrib_buffer, &resolved,
                    attrib_state->size, attrib_state->type,
                    attrib_state->normalized, integer_conv_dst_is_int,
                    &converted_stride);
            if (!converted_buffer) {
                fprintf(stderr,
                        "MGL VBIND skip attrib=%u buffer=%u: failed to convert "
                        "vertex attrib kind=%d type=0x%x\n",
                        attrib, attrib_buffer->name, ain.conversion_kind,
                        (unsigned)attrib_state->type);
                continue;
            }
            (void)converted_stride;
            if (mglSeEmitAttribBuffer(binding_owner, enc_ctx, snapshot,
                                      use_snap, &scratch_dummy, binding_index,
                                      converted_buffer, 0,
                                      any_binding_present)) {
                if (use_snap) {
                    mglSeSnapFlush(snapshot, 0, &scratch_dummy, enc_ctx);
                }
            }
            /* The conversion facade hands back a +1 retain; release it here the
             * way ARC released the strong local (the encoder retains the
             * buffer for the lifetime of the encoding). */
            mglBufferReleaseConvertedVertexBuffer(converted_buffer);
            continue;
        }

        /* NEED_MTL → ensure → POST plan */
        if (!attrib_buffer->data.mtl_data) {
            mglRendererBindMTLBuffer(renderer, attrib_buffer);
        }
        if (!attrib_buffer->data.mtl_data) {
            fprintf(stderr,
                    "MGL VBIND skip attrib=%u buffer=%u: no Metal backing\n",
                    attrib, attrib_buffer->name);
            continue;
        }
        if (!mglRenderMetalDataPointerUsable(attrib_buffer->data.mtl_data)) {
            fprintf(stderr,
                    "MGL VBIND skip attrib=%u buffer=%u: suspicious mtl_data=%p\n",
                    attrib, attrib_buffer->name, attrib_buffer->data.mtl_data);
            continue;
        }
        void *attrib_metal_buffer = attrib_buffer->data.mtl_data;
        if (!attrib_metal_buffer) {
            fprintf(stderr,
                    "MGL VBIND skip attrib=%u buffer=%u: Metal bridge failed\n",
                    attrib, attrib_buffer->name);
            continue;
        }
        size_t attrib_binding_offset = (size_t)resolved.binding_offset;
        uint64_t provisional_off = mglRenderVertexMetalBindOffset(
            ain.absolute_vertex_offsets, ain.binding_offset);
        mglBindingStageFillAttribPostMtlInput(
            &ain, 1, 1, (uint64_t)mglSeBufferLength(attrib_metal_buffer),
            mglSeBindingStateIsValid(binding_owner) ? 1 : 0,
            mglSeBindingStateBufferMatches(
                binding_owner, MGL_RENDER_BINDING_STAGE_VERTEX,
                attrib_metal_buffer, provisional_off, (uint32_t)binding_index)
                ? 1
                : 0);
        if (mglBindingStagePlanAttribEntry(&ain, &plan) != 0) {
            continue;
        }
        if (plan.action == MGL_ATTR_ACTION_SKIP) {
            if (plan.reason == MGL_ATTR_REASON_BAD_MTL) {
                fprintf(stderr,
                        "MGL VBIND skip attrib=%u buffer=%u: bindingOffset=%lu "
                        ">= metalLen=%lu\n",
                        attrib, attrib_buffer->name,
                        (unsigned long)attrib_binding_offset,
                        (unsigned long)ain.metal_len);
            }
            continue;
        }
        size_t metal_bind_offset = (size_t)plan.metal_bind_offset;
        if (plan.action == MGL_ATTR_ACTION_SKIP_MATCHED) {
            MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            any_binding_present[binding_index] = true;
            continue;
        }
        mglSeEmitAttribBuffer(binding_owner, enc_ctx, snapshot, use_snap,
                              &scratch_dummy, binding_index, attrib_metal_buffer,
                              metal_bind_offset, any_binding_present);
        mglNoteBufferEncoded(attrib_buffer);
        static uint64_t s_trace_file_vertex_attrib_bind_logs = 0;
        if (mglProgramNeedsTraceLog(active_program) &&
            mglShouldLogTraceFileBindingForProgram(
                active_program, &s_trace_file_vertex_attrib_bind_logs)) {
            MGLShaderResource *resource =
                mglRendererProgramVertexAttribResource(active_program, attrib);
            GLboolean effective_normalized_log = effective_normalized != 0;
            uint32_t format = mglRenderAttribFormatOrFallback(
                planned_format, (uint32_t)attrib_state->type,
                (uint32_t)attrib_state->size,
                effective_normalized_log ? 1 : 0);
            mglTraceLog(
                "VATTR_BIND program=%u attrib=%u resource=%s loc=%u "
                "metalSlot=%lu glBuffer=%u bindingIndex=%u bindingOffset=%lu "
                "relOffset=%lld stride=%u size=%u type=0x%x normalized=%u/%u "
                "divisor=%u table=%d metalLen=%lu format=%lu(%s)",
                active_program ? (unsigned)active_program->name : 0u,
                (unsigned)attrib,
                resource && resource->name ? resource->name : "(unknown)",
                resource ? (unsigned)resource->location : 0xffffffffu,
                (unsigned long)binding_index, (unsigned)attrib_buffer->name,
                (unsigned)resolved.binding_index,
                (unsigned long)attrib_binding_offset,
                (long long)resolved.relativeoffset, (unsigned)resolved.stride,
                (unsigned)attrib_state->size, (unsigned)attrib_state->type,
                (unsigned)attrib_state->normalized,
                (unsigned)effective_normalized, (unsigned)resolved.divisor,
                resolved.uses_binding_table ? 1 : 0, (unsigned long)ain.metal_len,
                (unsigned long)format, mglVertexFormatName(format));
        }
        if (mglSeVerboseBindLogs()) {
            fprintf(stderr,
                    "MGL SET VERTEX ATTRIB BUFFER index=%lu glName=%u offset=%lu "
                    "avail=%lu attrib=%u stride=%u rel=0x%llx mtl=%p\n",
                    (unsigned long)binding_index, attrib_buffer->name,
                    (unsigned long)attrib_binding_offset,
                    (unsigned long)ain.metal_len, attrib,
                    (unsigned)resolved.stride,
                    (unsigned long long)(uintptr_t)resolved.relativeoffset,
                    attrib_buffer->data.mtl_data);
        }
    }

    if (use_snap) {
        mglSeSnapFlush(snapshot, 0, &scratch_dummy, enc_ctx);
    }
    return true;
}

/* === -bindVertexBuffersToCurrentRenderEncoder: ============================ */

bool mglStageEncodeBindVertexBuffers(void *renderer,
                                     const MGLEncodeContext *enc_ctx)
{
    static uint64_t s_vbind_call_count = 0;
    static double s_vbind_last_call_time = 0.0;
    static uint64_t s_vbind_last_call_count = 0;
    uint64_t vbind_call = ++s_vbind_call_count;
    double vbind_start_seconds = mglTraceNowSeconds();
    uint64_t vbind_start_ns = mglTraceClockNS();
    mglLogLoopHeartbeat("vbind.loop", vbind_call, vbind_start_seconds,
                        &s_vbind_last_call_time, &s_vbind_last_call_count, 0.25);

    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    void *binding_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;

    bool any_binding_present[MAX_MAPPED_BUFFERS] = {false};
    bool base_binding_present[MAX_BINDABLE_BUFFERS] = {false};
    bool attrib_binding_reserved[MAX_MAPPED_BUFFERS] = {false};
    int attrib_binding_index[MAX_ATTRIBS];
    Program *active_program = NULL;
    VertexArray *vao = NULL;
    GLuint map_count = 0;

    if (mglSeVerboseBindLogs()) {
        fprintf(stderr, "MGL VBIND begin ctx=%p vao=%p owner=%p\n",
                (void *)areas.ctx,
                areas.ctx ? (void *)mglSeState(&areas)->vao : NULL,
                enc_ctx ? enc_ctx->render_encoder_owner : NULL);
    }

    if (!areas.ctx || !mglSeHasActiveEncoder(enc_ctx)) {
        fprintf(stderr, "MGL VBIND skip: encoder/ctx nil\n");
        return false;
    }

    vao = mglRendererGetValidatedVAO(
        areas.ctx, "-[MGLRenderer(Draw) bindVertexBuffersToCurrentRenderEncoder:]");
    if (!vao) {
        fprintf(stderr, "MGL VBIND skip: vao nil/invalid\n");
        return false;
    }
    active_program = areas.tessellation->nativeTESActive
                         ? areas.tessellation->nativeTESProgram
                         : mglResolveProgramForStageFromState(areas.ctx,
                                                              _VERTEX_SHADER);
    const int vertex_stage = areas.tessellation->nativeTESActive
                                 ? _TESS_EVALUATION_SHADER
                                 : _VERTEX_SHADER;

    const int use_vertex_binding_snapshot = 1;
    MGLRenderBindingSnapshot vbind_snapshot = {0};
    uint8_t vbind_byte_scratch[4096];
    size_t vbind_byte_scratch_used = 0;

    if (mglSeVerboseBindLogs()) {
        fprintf(stderr, "MGL VBIND vao=%p magic=0x%x\n", (void *)vao, vao->magic);
    }
    {
        int overflow = 0;
        map_count = (GLuint)mglBindingStageClampMapCount(
            (uint32_t)mglSeState(&areas)->vertex_buffer_map_list.count,
            (uint32_t)MAX_MAPPED_BUFFERS, &overflow);
        if (overflow) {
            static uint64_t s_vbind_map_count_overflow = 0;
            uint64_t hit = ++s_vbind_map_count_overflow;
            if (hit <= 16ull || (hit % 4096ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: VBIND mapCount exceeds "
                        "MAX_MAPPED_BUFFERS=%d, clamping (hit=%llu)\n",
                        MAX_MAPPED_BUFFERS, (unsigned long long)hit);
            }
        }
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        attrib_binding_index[i] = -1;
    }

    /* Reserve attrib slots before base/resource bindings. */
    bool attribs_enabled_by_app = (vao->enabled_attribs != 0u);
    GLuint reserve_max_attribs = MAX_ATTRIBS;
    for (GLuint attrib = 0; attrib < reserve_max_attribs; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(active_program, attrib)) {
            continue;
        }

        int mapped_index = mglRendererGetVertexBufferIndexWithAttributeSet(
            renderer, (int)attrib);
        if (mapped_index < 0 ||
            mapped_index >= (int)kMGLMaxMetalVertexBufferCount) {
            fprintf(stderr,
                    "MGL ERROR: VBIND reserve attrib=%u unresolved mapping=%d\n",
                    attrib, mapped_index);
            continue;
        }

        attrib_binding_index[attrib] = mapped_index;
        attrib_binding_reserved[mapped_index] = true;
    }

    if (mglSeVerboseBindLogs()) {
        for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
            int enabled = attribs_enabled_by_app &&
                          ((vao->enabled_attribs >> i) & 0x1u) != 0;
            MGLResolvedVertexAttribBinding resolved = {0};
            Buffer *attrib_buffer =
                mglRendererResolveVertexAttribBinding(
                    areas.ctx, vao, i,
                    "-[MGLRenderer(Draw) bindVertexBuffersToCurrentRenderEncoder:]",
                    &resolved)
                    ? resolved.buffer
                    : NULL;
            fprintf(stderr,
                    "MGL VBIND attrib=%u en=%d buf=%u off=%lld rel=0x%llx "
                    "stride=%u size=%u type=0x%x norm=%u div=%u bind=%u "
                    "table=%d mtl=%p ever=%u written=[%lld,%lld)\n",
                    i, enabled ? 1 : 0, attrib_buffer ? attrib_buffer->name : 0u,
                    (long long)(attrib_buffer ? resolved.binding_offset
                                              : vao->attrib[i].binding_offset),
                    (unsigned long long)(uintptr_t)vao->attrib[i].relativeoffset,
                    (unsigned)(attrib_buffer ? resolved.stride
                                             : vao->attrib[i].stride),
                    (unsigned)vao->attrib[i].size, (unsigned)vao->attrib[i].type,
                    (unsigned)vao->attrib[i].normalized,
                    (unsigned)(attrib_buffer ? resolved.divisor
                                             : vao->attrib[i].divisor),
                    (unsigned)vao->attrib[i].buffer_bindingindex,
                    attrib_buffer && resolved.uses_binding_table ? 1 : 0,
                    attrib_buffer ? attrib_buffer->data.mtl_data : NULL,
                    attrib_buffer ? (unsigned)attrib_buffer->ever_written : 0u,
                    attrib_buffer ? (long long)attrib_buffer->written_min : 0ll,
                    attrib_buffer ? (long long)attrib_buffer->written_max : 0ll);
        }
    }

    /* The stage-buffer binding drivers are C now (log 129). */
    if (!mglBindingStateBindStageBufferMapEntries(
            renderer, vertex_stage, 0,
            &mglSeState(&areas)->vertex_buffer_map_list, any_binding_present,
            base_binding_present, attrib_binding_reserved, enc_ctx,
            &vbind_snapshot, vbind_byte_scratch, &vbind_byte_scratch_used,
            sizeof(vbind_byte_scratch), use_vertex_binding_snapshot,
            (uint32_t)kMGLMaxMetalVertexBufferCount,
            areas.tessellation->nativeTESActive, 1)) {
        return false;
    }

    if (use_vertex_binding_snapshot && vbind_snapshot.vertex_op_count > 0) {
        mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
            enc_ctx->render_encoder_owner, &vbind_snapshot, NULL, 0);
        vbind_snapshot = (MGLRenderBindingSnapshot){0};
        vbind_byte_scratch_used = 0;
    }

    if (!mglStageEncodeBindVertexAttributes(
            renderer, vao, active_program, attribs_enabled_by_app,
            attrib_binding_index, any_binding_present, enc_ctx, &vbind_snapshot,
            use_vertex_binding_snapshot)) {
        return false;
    }

    if (map_count > 0) {
        mglBindingStateBindStageFallbackBuffers(
            renderer, vertex_stage, 0, active_program, any_binding_present,
            base_binding_present, enc_ctx, &vbind_snapshot,
            use_vertex_binding_snapshot, (uint32_t)kMGLMaxMetalVertexBufferCount,
            kMGLEnableVertexAllSlotFallback);
    }

    mglSeBindPointSizeParams(renderer, any_binding_present, enc_ctx,
                             &vbind_snapshot, vbind_byte_scratch,
                             &vbind_byte_scratch_used,
                             sizeof(vbind_byte_scratch),
                             use_vertex_binding_snapshot);

    /* Finalize present-mask after VAO / fallback / point-size updates. */
    mglSeFinalizePresentMask(renderer, 0, any_binding_present,
                             base_binding_present, attrib_binding_reserved,
                             vbind_call, map_count, vbind_start_ns);

    mglRenderBindingSetValid(binding_owner, 1);
    return true;
}

/* === -bindFragmentBuffersToCurrentRenderEncoder: ========================== */

bool mglStageEncodeBindFragmentBuffers(void *renderer,
                                       const MGLEncodeContext *enc_ctx)
{
    static uint64_t s_fbind_call_count = 0;
    static double s_fbind_last_call_time = 0.0;
    static uint64_t s_fbind_last_call_count = 0;
    uint64_t fbind_call = ++s_fbind_call_count;
    double fbind_start_seconds = mglTraceNowSeconds();
    uint64_t fbind_start_ns = mglTraceClockNS();
    mglLogLoopHeartbeat("fbind.loop", fbind_call, fbind_start_seconds,
                        &s_fbind_last_call_time, &s_fbind_last_call_count, 0.25);

    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    void *binding_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;

    GLuint map_count = 0;
    bool any_binding_present[MAX_BINDABLE_BUFFERS] = {false};
    bool base_binding_present[MAX_BINDABLE_BUFFERS] = {false};
    Program *active_program = NULL;

    if (mglSeVerboseBindLogs()) {
        fprintf(stderr, "MGL FBIND begin ctx=%p owner=%p\n", (void *)areas.ctx,
                enc_ctx ? enc_ctx->render_encoder_owner : NULL);
    }

    if (!areas.ctx || !mglSeHasActiveEncoder(enc_ctx)) {
        fprintf(stderr, "MGL FBIND skip: ctx/encoder nil\n");
        return false;
    }
    active_program =
        mglResolveProgramForStageFromState(areas.ctx, _FRAGMENT_SHADER);

    const int use_binding_snapshot = 1;
    MGLRenderBindingSnapshot snapshot = {0};
    uint8_t fbind_byte_scratch[4096];
    size_t fbind_byte_scratch_used = 0;

    {
        int overflow = 0;
        map_count = (GLuint)mglBindingStageClampMapCount(
            (uint32_t)mglSeState(&areas)->fragment_buffer_map_list.count,
            (uint32_t)MAX_MAPPED_BUFFERS, &overflow);
        if (overflow) {
            static uint64_t s_fbind_map_count_overflow = 0;
            uint64_t hit = ++s_fbind_map_count_overflow;
            if (hit <= 16ull || (hit % 4096ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: FBIND mapCount exceeds "
                        "MAX_MAPPED_BUFFERS=%d, clamping (hit=%llu)\n",
                        MAX_MAPPED_BUFFERS, (unsigned long long)hit);
            }
        }
    }

    if (!mglBindingStateBindStageBufferMapEntries(
            renderer, _FRAGMENT_SHADER, 1,
            &mglSeState(&areas)->fragment_buffer_map_list, any_binding_present,
            base_binding_present, NULL, enc_ctx, &snapshot, fbind_byte_scratch,
            &fbind_byte_scratch_used, sizeof(fbind_byte_scratch),
            use_binding_snapshot, (uint32_t)MAX_BINDABLE_BUFFERS, 0, 0)) {
        return false;
    }

    if (use_binding_snapshot && snapshot.fragment_op_count > 0) {
        mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
            enc_ctx->render_encoder_owner, &snapshot, NULL, 0);
        snapshot = (MGLRenderBindingSnapshot){0};
        fbind_byte_scratch_used = 0;
    }

    if (map_count > 0) {
        mglBindingStateBindStageFallbackBuffers(
            renderer, _FRAGMENT_SHADER, 1, active_program, any_binding_present,
            base_binding_present, enc_ctx, &snapshot, use_binding_snapshot,
            MAX_BINDABLE_BUFFERS, 1);
    }

    mglSeFinalizePresentMask(renderer, 1, any_binding_present,
                             base_binding_present, NULL, fbind_call, map_count,
                             fbind_start_ns);

    mglRenderBindingSetValid(binding_owner, 1);
    return true;
}
