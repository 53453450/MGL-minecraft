/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Split from draw_command.c (Phase 7 soft god-TU target).
 */
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdatomic.h>

#include "glm_context.h"
#include "draw_command.h"
#include "mgl_byte_hash.h"
#include "mgl_safety.h"
#include "mgl_frame_activity.h"
#include "mgl_trace_log.h"
#include "mgl_sampler_compat.h"
#include "mgl_batch_hazard.h"
#include "mgl_env_flag.h"
#include "mgl_render_api_buffer.h"
#include "mgl_buffer_backend.h"
#include "mgl_draw_command_shared.h"

extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern void mglSparseScrubBoundImages(GLMContext ctx);

bool mglBatchInitMDICompatible(const MGLDrawBatch *batch, const MGLDrawCommand *cmd)
{
    if (batch->key.primitive_type == 0xFF) return false;

    MGLDrawCommandType cmd_type = cmd->type;
    bool cmd_uses_elements = (cmd_type != MGL_CMD_DRAW_ARRAYS &&
                             cmd_type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
                             cmd_type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE);
    if (cmd_uses_elements != batch->uses_elements) return false;

    /* Emulated modes can't use MDI */
    GLenum mode = cmd->mode;
    if (mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP) return false;

    return true;
}

/* Called on each command append to maintain mdi_compatible incrementally.
 * Short-circuits when the batch is already incompatible.  Only checks
 * per-command properties:
 *   - draw mode (GL_TRIANGLE_FAN / GL_LINE_LOOP emulated by MDI-incompatible
 *     primitives)
 *   - indexType match against the first command (only meaningful for the
 *     2nd and later commands; the 1st command's self-comparison is skipped
 *     since commands[0] is the just-stored command)
 * Batch-level invariants (primitive type, element usage) were validated at
 * creation by mglBatchInitMDICompatible and are not re-checked here. */
void mglBatchUpdateMDIOnAppend(MGLDrawBatch *batch, const MGLDrawCommand *cmd)
{
    if (!batch->mdi_compatible) return;

    GLenum mode = cmd->mode;
    if (mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP) {
        batch->mdi_compatible = false;
        return;
    }

    if (batch->uses_elements && batch->command_count > 1u) {
        if (batch->commands[0].indexType != cmd->indexType) {
            batch->mdi_compatible = false;
        }
    }
}

void mglUpdateBatchSamplerSnapshotState(MGLDrawBatch *batch,
                                               uint16_t snapshot_id)
{
    if (!batch) return;

    if (batch->command_count == 0u) {
        batch->sampler_snapshot_id = snapshot_id;
    } else if (batch->sampler_snapshot_id != snapshot_id) {
        batch->sampler_snapshots_mixed = true;
        batch->mdi_compatible = false;
    }

    if (snapshot_id != MGL_INVALID_SAMPLER_SNAPSHOT_ID) {
        batch->has_sampler_snapshots = true;
    }
}

static size_t mglCommandIndexSize(GLenum type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE:  return 1;
        case GL_UNSIGNED_SHORT: return 2;
        case GL_UNSIGNED_INT:   return 4;
        default:                return 0;
    }
}

static size_t mglCommandAttribComponentSize(GLenum type)
{
    switch (type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            return 4;
        case GL_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

static size_t mglCommandAttribElementBytes(GLenum type, GLuint size)
{
    switch (type) {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            return 4;
        default: {
            size_t component = mglCommandAttribComponentSize(type);
            if (component == 0 || size == 0 || size > SIZE_MAX / component) {
                return 0;
            }
            return component * (size_t)size;
        }
    }
}

typedef struct {
    VertexAttrib *attrib;
    Buffer       *buffer;
    GLintptr      binding_offset;
    GLuint        stride;
    GLuint        divisor;
    GLintptr      relativeoffset;
    GLuint        binding_index;
    bool          uses_binding_table;
} MGLCommandResolvedAttrib;

static bool mglCommandResolveVertexAttrib(VertexArray *vao,
                                          GLuint attribute,
                                          MGLCommandResolvedAttrib *out)
{
    if (!vao || attribute >= MAX_ATTRIBS || !out) {
        return false;
    }

    VertexAttrib *attrib = &vao->attrib[attribute];
    Buffer *buffer = attrib->buffer;
    GLintptr bindingOffset = attrib->binding_offset;
    GLuint stride = attrib->stride;
    GLuint divisor = attrib->divisor;
    GLuint bindingIndex = attrib->buffer_bindingindex;
    bool usesBindingTable = false;

    if (bindingIndex < MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        BufferBinding *binding = &vao->bindings[bindingIndex];
        if (binding->buffer) {
            buffer = binding->buffer;
            bindingOffset = binding->offset;
            stride = (binding->stride > 0) ? (GLuint)binding->stride : attrib->stride;
            divisor = binding->divisor;
            usesBindingTable = true;
        }
    }

    if (!buffer) {
        return false;
    }

    out->attrib = attrib;
    out->buffer = buffer;
    out->binding_offset = bindingOffset;
    out->stride = stride;
    out->divisor = divisor;
    out->relativeoffset = attrib->relativeoffset;
    out->binding_index = bindingIndex;
    out->uses_binding_table = usesBindingTable;
    return true;
}

static bool mglCommandPrimitiveRestartIndex(GLMContext ctx, GLenum type, uint64_t *restart)
{
    if (!ctx || !restart ||
        (!ctx->active_state->caps.primitive_restart && !ctx->active_state->caps.primitive_restart_fixed_index)) {
        return false;
    }

    if (ctx->active_state->caps.primitive_restart_fixed_index) {
        switch (type) {
            case GL_UNSIGNED_BYTE:  *restart = 0xFFu; return true;
            case GL_UNSIGNED_SHORT: *restart = 0xFFFFu; return true;
            case GL_UNSIGNED_INT:   *restart = 0xFFFFFFFFu; return true;
            default: return false;
        }
    }

    *restart = (uint64_t)ctx->active_state->var.primitive_restart_index;
    return true;
}

static bool mglCommandReadIndexValue(const uint8_t *data, GLenum type, GLsizei i, uint64_t *value)
{
    if (!data || !value) return false;

    switch (type) {
        case GL_UNSIGNED_BYTE:
            *value = data[i];
            return true;
        case GL_UNSIGNED_SHORT: {
            uint16_t v;
            memcpy(&v, data + ((size_t)i * sizeof(v)), sizeof(v));
            *value = v;
            return true;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v;
            memcpy(&v, data + ((size_t)i * sizeof(v)), sizeof(v));
            *value = v;
            return true;
        }
        default:
            return false;
    }
}

/* Cached element-index scan (min/max) shared by the hazard vertex-range walk
 * and the stream-merge candidate walk.  One O(count) pass fills both the
 * restart-excluding range and the all-index range; repeated draws of the same
 * (buffer, offset, count, type, restart) with no intervening content_version
 * bump reuse it.  Buffers whose bytes can move without a GL call (mapped,
 * persistently mappable, GPU-write targets) are scanned live every time. */
typedef struct MGLCommandElementIndexScan {
    uint64_t min_index;       /* excluding restart indices */
    uint64_t max_index;
    uint64_t all_min_index;   /* min over every index */
    uint64_t all_max_index;   /* max over every index */
    bool live_valid;          /* at least one non-restart index exists */
} MGLCommandElementIndexScan;

static bool mglElementIndexScanCacheable(const Buffer *elementBuffer)
{
    return elementBuffer && !elementBuffer->mapped &&
           !elementBuffer->gpu_write_target &&
           !(elementBuffer->access_flags & GL_MAP_PERSISTENT_BIT);
}

static bool mglScanCommandElementIndices(Buffer *elementBuffer,
                                         GLenum indexType,
                                         uint64_t indexOffset,
                                         GLsizei count,
                                         bool hasRestart,
                                         uint64_t restart,
                                         MGLCommandElementIndexScan *out)
{
    if (!elementBuffer || !elementBuffer->data.buffer_data || count <= 0 || !out) {
        return false;
    }
    size_t indexSize = mglCommandIndexSize(indexType);
    if (indexSize == 0) return false;
    if ((uint64_t)count > UINT64_MAX / (uint64_t)indexSize) return false;
    uint64_t indexBytes = (uint64_t)count * (uint64_t)indexSize;
    if (indexBytes > UINT64_MAX - indexOffset ||
        indexOffset + indexBytes > (uint64_t)elementBuffer->size) {
        return false;
    }

    const uint32_t restart32 = hasRestart ? (uint32_t)restart : 0u;
    if (mglElementIndexScanCacheable(elementBuffer) &&
        elementBuffer->scan_cache_valid &&
        elementBuffer->scan_cache_content_version == elementBuffer->content_version &&
        elementBuffer->scan_cache_offset == indexOffset &&
        elementBuffer->scan_cache_count == (uint32_t)count &&
        elementBuffer->scan_cache_type == indexType &&
        elementBuffer->scan_cache_restart_enabled == (hasRestart ? 1 : 0) &&
        elementBuffer->scan_cache_restart_index == restart32) {
        out->min_index = elementBuffer->scan_cache_min_index;
        out->max_index = elementBuffer->scan_cache_max_index;
        out->all_min_index = elementBuffer->scan_cache_all_min_index;
        out->all_max_index = elementBuffer->scan_cache_all_max_index;
        out->live_valid = elementBuffer->scan_cache_live_valid != 0;
        return true;
    }

    const uint8_t *data =
        (const uint8_t *)(uintptr_t)elementBuffer->data.buffer_data + indexOffset;
    uint64_t minIndex = UINT64_MAX;
    uint64_t maxIndex = 0;
    uint64_t allMinIndex = UINT64_MAX;
    uint64_t allMaxIndex = 0;
    for (GLsizei i = 0; i < count; i++) {
        uint64_t index = 0;
        if (!mglCommandReadIndexValue(data, indexType, i, &index)) {
            return false;
        }
        if (index < allMinIndex) allMinIndex = index;
        if (index > allMaxIndex) allMaxIndex = index;
        if (!hasRestart || index != restart) {
            if (index < minIndex) minIndex = index;
            if (index > maxIndex) maxIndex = index;
        }
    }

    const bool liveValid = (minIndex != UINT64_MAX);
    out->min_index = liveValid ? minIndex : 0;
    out->max_index = maxIndex;
    out->all_min_index = allMinIndex;
    out->all_max_index = allMaxIndex;
    out->live_valid = liveValid;

    if (mglElementIndexScanCacheable(elementBuffer)) {
        elementBuffer->scan_cache_min_index = (uint32_t)out->min_index;
        elementBuffer->scan_cache_max_index = (uint32_t)out->max_index;
        elementBuffer->scan_cache_all_min_index = (uint32_t)allMinIndex;
        elementBuffer->scan_cache_all_max_index = (uint32_t)allMaxIndex;
        elementBuffer->scan_cache_content_version = elementBuffer->content_version;
        elementBuffer->scan_cache_offset = indexOffset;
        elementBuffer->scan_cache_count = (uint32_t)count;
        elementBuffer->scan_cache_type = indexType;
        elementBuffer->scan_cache_restart_index = restart32;
        elementBuffer->scan_cache_restart_enabled = hasRestart ? 1 : 0;
        elementBuffer->scan_cache_valid = 1;
        elementBuffer->scan_cache_live_valid = liveValid ? 1 : 0;
    }
    return true;
}

static bool mglCommandApplyBaseVertexToIndexRange(const MGLDrawCommand *cmd,
                                                  uint64_t minIndex,
                                                  uint64_t maxIndex,
                                                  uint64_t *minVertex,
                                                  uint64_t *maxVertex)
{
    if (!cmd || !minVertex || !maxVertex) return false;
    if (cmd->baseVertex < 0) {
        uint64_t negBase = (uint64_t)(-((int64_t)cmd->baseVertex));
        *minVertex = (minIndex > negBase) ? (minIndex - negBase) : 0;
        *maxVertex = (maxIndex > negBase) ? (maxIndex - negBase) : 0;
        return true;
    }
    uint64_t base = (uint64_t)cmd->baseVertex;
    if (minIndex > UINT64_MAX - base || maxIndex > UINT64_MAX - base) {
        return false;
    }
    *minVertex = minIndex + base;
    *maxVertex = maxIndex + base;
    return true;
}

static bool mglCommandComputeElementVertexRange(GLMContext ctx,
                                                const MGLDrawCommand *cmd,
                                                uint64_t *minVertex,
                                                uint64_t *maxVertex)
{
    if (!ctx || !cmd || !minVertex || !maxVertex || cmd->count <= 0) return false;

    /* AUDIT-034: DrawRangeElements declares the live index span — use it
     * instead of scanning the element buffer (indices outside are undefined). */
    if (cmd->has_index_range) {
        if (cmd->range_end < cmd->range_start) return false;
        return mglCommandApplyBaseVertexToIndexRange(
            cmd, (uint64_t)cmd->range_start, (uint64_t)cmd->range_end,
            minVertex, maxVertex);
    }

    Buffer *elementBuffer = mglDrawCommandElementBuffer(ctx, cmd);
    size_t indexSize = mglCommandIndexSize(cmd->indexType);
    if (!elementBuffer || !elementBuffer->data.buffer_data || indexSize == 0) return false;
    if (cmd->indexBufferOffset > (GLuint)elementBuffer->size) return false;
    if ((uint64_t)cmd->count > (UINT64_MAX / (uint64_t)indexSize)) return false;

    uint64_t byteCount = (uint64_t)cmd->count * (uint64_t)indexSize;
    uint64_t offset = (uint64_t)cmd->indexBufferOffset;
    if (byteCount > UINT64_MAX - offset ||
        offset + byteCount > (uint64_t)elementBuffer->size) {
        return false;
    }

    uint64_t restart = 0;
    bool hasRestart = mglCommandPrimitiveRestartIndex(ctx, cmd->indexType, &restart);

    MGLCommandElementIndexScan scan;
    if (!mglScanCommandElementIndices(elementBuffer, cmd->indexType, offset,
                                      cmd->count, hasRestart, restart, &scan)) {
        return false;
    }
    if (!scan.live_valid) {
        return false;
    }

    return mglCommandApplyBaseVertexToIndexRange(
        cmd, scan.min_index, scan.max_index, minVertex, maxVertex);
}

static bool mglCommandComputeArrayVertexRange(const MGLDrawCommand *cmd,
                                              uint64_t *minVertex,
                                              uint64_t *maxVertex)
{
    if (!cmd || !minVertex || !maxVertex || cmd->first < 0 || cmd->count <= 0) {
        return false;
    }

    uint64_t first = (uint64_t)cmd->first;
    uint64_t count = (uint64_t)cmd->count;
    if (count == 0 || first > UINT64_MAX - (count - 1u)) {
        return false;
    }

    *minVertex = first;
    *maxVertex = first + count - 1u;
    return true;
}

static void mglTrackPendingAttribRead(GLMContext ctx,
                                      const MGLCommandResolvedAttrib *resolved,
                                      uint64_t minVertex,
                                      uint64_t maxVertex,
                                      GLsizei instanceCount,
                                      GLuint baseInstance)
{
    if (!ctx || !resolved || !resolved->attrib || !resolved->buffer) return;

    VertexAttrib *attrib = resolved->attrib;
    size_t elemBytes = mglCommandAttribElementBytes(attrib->type, attrib->size);
    uint64_t stride = resolved->stride > 0 ? (uint64_t)resolved->stride : (uint64_t)elemBytes;
    if (elemBytes == 0 || stride == 0 ||
        resolved->binding_offset < 0 || resolved->relativeoffset < 0) {
        mglTrackPendingReadWholeBuffer(ctx, resolved->buffer);
        return;
    }

    uint64_t firstIndex = minVertex;
    uint64_t lastIndex = maxVertex;
    if (resolved->divisor != 0) {
        uint64_t instances = instanceCount > 0 ? (uint64_t)instanceCount : 1u;
        uint64_t base = (uint64_t)baseInstance;
        firstIndex = base / (uint64_t)resolved->divisor;
        lastIndex = (base + instances - 1u) / (uint64_t)resolved->divisor;
    }

    uint64_t baseOffset = (uint64_t)resolved->binding_offset + (uint64_t)resolved->relativeoffset;
    if (baseOffset > UINT64_MAX - ((uint64_t)elemBytes) ||
        firstIndex > UINT64_MAX / stride ||
        lastIndex > UINT64_MAX / stride) {
        mglTrackPendingReadWholeBuffer(ctx, resolved->buffer);
        return;
    }

    uint64_t start = baseOffset + firstIndex * stride;
    uint64_t end = baseOffset + lastIndex * stride + (uint64_t)elemBytes;
    if (end <= start) {
        mglTrackPendingReadWholeBuffer(ctx, resolved->buffer);
        return;
    }

    mglTrackPendingReadRange(ctx, resolved->buffer, start, end);
}

void mglTrackPendingBaseBufferReads(GLMContext ctx)
{
    if (!ctx) return;

    uint64_t activeCount = 0;

    const int trackedTargets[] = {
        _UNIFORM_BUFFER,
        _UNIFORM_CONSTANT,
        _SHADER_STORAGE_BUFFER,
        _ATOMIC_COUNTER_BUFFER,
        _TEXTURE_BUFFER
    };

    /* Use the per-target active_mask bitmap to skip the ~84-slot linear
     * scan.  Typical Minecraft draws have only a handful of active bindings,
     * so visiting only set bits turns ~420 empty-slot checks into ~5-10
     * iterations.  plain_uniform_buffers (below) stays a linear scan
     * because it is a bare BufferBaseTarget[] without a BufferBase wrapper
     * and has few active slots in practice. */
    for (size_t t = 0; t < sizeof(trackedTargets) / sizeof(trackedTargets[0]); t++) {
        int target = trackedTargets[t];
        BufferBase *base = &ctx->active_state->buffer_base[target];
        for (uint32_t w = 0; w < MGL_BUFFER_BASE_ACTIVE_WORDS; w++) {
            uint64_t bits = base->active_mask[w];
            while (bits) {
                int i = __builtin_ctzll(bits);
                bits &= bits - 1;
                BufferBaseTarget *binding = &base->buffers[(w << 6) + i];
                if (!binding->buf) continue;
                activeCount++;
                if (binding->size > 0 && binding->offset >= 0) {
                    mglTrackPendingReadBytes(ctx,
                                             binding->buf,
                                             (uint64_t)binding->offset,
                                             (uint64_t)binding->size);
                } else {
                    mglTrackPendingReadWholeBuffer(ctx, binding->buf);
                }
            }
        }
    }

    Program *program = ctx->active_state->program;
    if (program) {
        /* plain_uniform_active_mask bitmap scan replaces the 84-slot linear
         * scan; only slots with buf != NULL are visited. */
        for (int w = 0; w < (int)MGL_PLAIN_UNIFORM_ACTIVE_WORDS; w++) {
            uint64_t bits = program->plain_uniform_active_mask[w];
            while (bits) {
                int i = (int)((w * 64) + __builtin_ctzll(bits));
                bits &= bits - 1;
                BufferBaseTarget *binding = &program->plain_uniform_buffers[i];
                if (!binding->buf) continue;
                activeCount++;
                if (binding->size > 0 && binding->offset >= 0) {
                    mglTrackPendingReadBytes(ctx,
                                             binding->buf,
                                             (uint64_t)binding->offset,
                                             (uint64_t)binding->size);
                } else {
                    mglTrackPendingReadWholeBuffer(ctx, binding->buf);
                }
            }
        }
    }

    MGL_PERF_ADD(g_mglHazardActiveBindingsSinceSwap, activeCount);
}

#define MGL_STREAM_MERGE_MAX_SOURCE_BYTES (64u * 1024u)
#define MGL_STREAM_MERGE_MAX_BATCH_BYTES  (4u * 1024u * 1024u)

/* MGLStreamMergeCandidate: mgl_draw_command_shared.h */

static size_t mglAlignUpSize(size_t value, size_t alignment)
{
    if (alignment <= 1u) return value;
    size_t rem = value % alignment;
    if (rem == 0u) return value;
    if (value > SIZE_MAX - (alignment - rem)) return SIZE_MAX;
    return value + (alignment - rem);
}

static Buffer *mglNewTransientBatchBuffer(GLenum target)
{
    Buffer *buffer = (Buffer *)calloc(1, sizeof(Buffer));
    if (!buffer) return NULL;

    buffer->name = 0;
    buffer->target = target;
    buffer->size = 0;
    buffer->written_min = -1;
    buffer->written_max = -1;
    buffer->transient_batch_buffer = GL_TRUE;
    buffer->data.dirty_bits = DIRTY_BUFFER_ADDR | DIRTY_BUFFER_DATA;
    return buffer;
}

/* === transient scratch pool ============================================ */

/* A per-draw batch stream churns one vertex + one index scratch per batch;
 * without pooling every batch pays a calloc, a CPU backing realloc and — at
 * flush — a fresh Metal buffer (a mach trap per draw).  Pool the Buffer
 * objects and their Metal stores; contents are rewritten per batch.  A
 * pooled buffer may only be reused once the commit that last read it has
 * completed, which the generation tag fences. */
typedef struct MGLTransientPoolNode {
    struct MGLTransientPoolNode *next;
    Buffer *buffer;
    uint64_t generation;   /* reading commit's generation at release time */
} MGLTransientPoolNode;

#define MGL_TRANSIENT_POOL_MAX 64

static int mglTransientPoolSlot(GLenum target)
{
    return (target == GL_ELEMENT_ARRAY_BUFFER) ? 1 : 0;
}

static Buffer *mglTakeTransientPoolNode(GLMContext ctx, int slot,
                                        MGLTransientPoolNode **link,
                                        MGLTransientPoolNode *node,
                                        GLenum target)
{
    Buffer *buffer = node->buffer;
    *link = node->next;
    ctx->transient_pool_count[slot]--;
    free(node);

    /* Restore pristine first-use fields; the CPU backing and the Metal
     * store (with their larger capacities) are retained.  A busy store is
     * safe: UpdateDirty CoW-snapshots it before the CPU rewrite lands. */
    buffer->name = 0;
    buffer->target = target;
    buffer->size = 0;
    buffer->written_min = -1;
    buffer->written_max = -1;
    buffer->transient_batch_buffer = GL_TRUE;
    buffer->data.dirty_bits = 0;
    buffer->has_initialized_data = GL_FALSE;
    buffer->ever_written = GL_FALSE;
    buffer->last_init_source = kInitBufferDataCopy;
    buffer->last_write_offset = 0;
    buffer->last_write_size = 0;
    buffer->last_write_src_ptr = NULL;
    buffer->last_write_src_hash = 0;
    /* Derived state must not survive into the new use: the index scan
     * cache is keyed on content_version, and a recycled buffer keeps both
     * unless they are bumped here. */
    buffer->content_version++;
    buffer->scan_cache_valid = 0;
    buffer->scan_cache_live_valid = 0;
    return buffer;
}

Buffer *mglAcquireTransientBatchBuffer(GLMContext ctx, GLenum target)
{
    int slot = mglTransientPoolSlot(target);
    MGLTransientPoolNode **head =
        ctx ? (MGLTransientPoolNode **)&ctx->transient_pool[slot] : NULL;

    if (head && *head) {
        const uint64_t completed = mglRenderCompletedBufferGeneration();
        MGLTransientPoolNode **link = head;
        MGLTransientPoolNode *node = *head;
        while (node) {
            if (node->generation <= completed) {
                return mglTakeTransientPoolNode(ctx, slot, link, node, target);
            }
            link = &node->next;
            node = node->next;
        }

        /* No idle slot: reuse the oldest busy shell.  SnapshotSharedDirty
         * CoW-protects the in-flight Metal store; without this, each
         * in-flight batch beyond the pool depth paid CreateBufferWithBytes. */
        node = *head;
        return mglTakeTransientPoolNode(ctx, slot, head, node, target);
    }

    return mglNewTransientBatchBuffer(target);
}

void mglRecycleTransientBatchBuffer(GLMContext ctx, Buffer *buffer)
{
    int slot = buffer ? mglTransientPoolSlot(buffer->target) : 0;
    if (!ctx || !buffer ||
        ctx->transient_pool_count[slot] >= MGL_TRANSIENT_POOL_MAX) {
        mglDestroyTransientBuffer(ctx, buffer);
        return;
    }

    MGLTransientPoolNode *node = (MGLTransientPoolNode *)malloc(sizeof(*node));
    if (!node) {
        mglDestroyTransientBuffer(ctx, buffer);
        return;
    }
    node->buffer = buffer;
    /* The stamp names the accumulating command buffer; its completion (paired
     * in mglRegisterBufferGenerationCompletion) is what makes the slot
     * reusable, so the GPU is done reading before the pool hands it out. */
    node->generation = mglRenderCurrentBufferGeneration();
    node->next = (MGLTransientPoolNode *)ctx->transient_pool[slot];
    ctx->transient_pool[slot] = node;
    ctx->transient_pool_count[slot]++;
}

void mglDrainTransientPool(GLMContext ctx)
{
    for (int slot = 0; slot < 2; slot++) {
        MGLTransientPoolNode *node = (MGLTransientPoolNode *)ctx->transient_pool[slot];
        while (node) {
            MGLTransientPoolNode *next = node->next;
            mglDestroyTransientBuffer(ctx, node->buffer);
            free(node);
            node = next;
        }
        ctx->transient_pool[slot] = NULL;
        ctx->transient_pool_count[slot] = 0;
    }
}

void mglDrainTransientPoolForContext(GLMContext ctx)
{
    if (!ctx) return;
    mglDrainTransientPool(ctx);
}

static bool mglEnsureTransientBufferCapacity(Buffer *buffer, size_t needed)
{
    if (!buffer) return false;

    /* A pooled scratch keeps the previous batch's Metal store, whose length
     * is the size it was last created at.  A larger batch must drop it so
     * the flush recreates it at the new size — the upload path clamps to
     * the Metal length and would otherwise silently truncate. */
    if (needed > 0 && mglBufferMtl(buffer)) {
        MGLRenderBufferInfo info = {0};
        if (mglRenderGetBufferInfo(mglBufferMtl(buffer), &info) == 0 &&
            (size_t)info.length < needed) {
            mglRenderReleaseBufferMetalData(NULL, buffer);
        }
    }

    if (needed <= buffer->data.buffer_size) return true;

    size_t newCapacity = buffer->data.buffer_size ? buffer->data.buffer_size : 4096u;
    while (newCapacity < needed) {
        if (newCapacity > SIZE_MAX / 2u) {
            newCapacity = needed;
            break;
        }
        newCapacity *= 2u;
    }

    void *oldData = (void *)(uintptr_t)buffer->data.buffer_data;
    void *newData = realloc(oldData, newCapacity);
    if (!newData) return false;

    buffer->data.buffer_data = (vm_address_t)(uintptr_t)newData;
    buffer->data.buffer_size = newCapacity;
    buffer->data.dirty_bits |= DIRTY_BUFFER_ADDR | DIRTY_BUFFER_DATA;
    return true;
}

static void mglMarkTransientBufferWritten(Buffer *buffer, size_t bytes)
{
    if (!buffer) return;

    buffer->size = (GLsizeiptr)bytes;
    /* Contents always change; ADDR only when there is no Metal store (or
     * EnsureCapacity already dropped one that was too small).  Re-setting
     * ADDR on a live store forced UpdateDirty down the BindBufferStorage
     * path even though the existing MTLBuffer was reusable. */
    buffer->data.dirty_bits |= DIRTY_BUFFER_DATA;
    if (!mglBufferMtl(buffer)) {
        buffer->data.dirty_bits |= DIRTY_BUFFER_ADDR;
    }
    buffer->has_initialized_data = GL_TRUE;
    buffer->ever_written = GL_TRUE;
    buffer->written_min = 0;
    buffer->written_max = (GLintptr)bytes;
    buffer->last_init_source = kInitBufferDataCopy;
    buffer->last_write_offset = 0;
    buffer->last_write_size = (GLsizeiptr)bytes;
    buffer->last_write_src_ptr = (const void *)(uintptr_t)buffer->data.buffer_data;
    buffer->last_write_src_hash = 0;
}

static uint64_t mglStreamHashAttrib(uint64_t hash,
                                    const MGLCommandResolvedAttrib *resolved,
                                    GLuint index)
{
    if (!resolved || !resolved->attrib) return hash;

    const VertexAttrib *attrib = resolved->attrib;
    hash ^= mglRotateLeft64((uint64_t)index, 3);
    hash ^= mglRotateLeft64((uint64_t)attrib->size, 7);
    hash ^= mglRotateLeft64((uint64_t)attrib->type, 11);
    hash ^= mglRotateLeft64((uint64_t)attrib->normalized, 13);
    hash ^= mglRotateLeft64((uint64_t)attrib->integer, 17);
    hash ^= mglRotateLeft64((uint64_t)attrib->long_attribute, 19);
    hash ^= mglRotateLeft64((uint64_t)resolved->stride, 23);
    hash ^= mglRotateLeft64((uint64_t)resolved->relativeoffset, 29);
    hash ^= mglRotateLeft64((uint64_t)resolved->binding_index, 31);
    hash ^= mglRotateLeft64((uint64_t)resolved->binding_offset, 41);
    hash ^= mglRotateLeft64((uint64_t)resolved->divisor, 47);
    return hash;
}

/* ---- Stream-merge incompatible vertex layout registry ----
 *
 * Certain vertex layouts are known to produce corrupt rendering when
 * stream-merged (draw call batching that concatenates vertex/index data
 * from multiple draws into a single transient buffer).  The two layouts
 * below were identified from Minecraft's vertex formats:
 *
 *   - Weather particles: blending + depth-write-off, stride 28, 4 attribs.
 *     Merging reorders draw calls relative to the blend equation, producing
 *     visible particle sorting artifacts.
 *
 *   - Chunk terrain: stride 32, 5 attribs.  Merging concatenates terrain
 *     slices from different chunks; the per-draw index base assumptions in
 *     the terrain shader's gl_VertexID usage break, producing garbled
 *     geometry.
 *
 * Set env var MGL_DISABLE_STREAM_MERGE_EXCLUSIONS=1 to bypass these
 * exclusions (useful for testing whether MGL improvements have made them
 * unnecessary).  Set MGL_DEBUG_STREAM_MERGE=1 to log each exclusion hit.
 */

typedef struct {
    const char *name;
    size_t vertexStride;
    uint32_t attribMask;      /* low bits that must ALL be set */
    uint32_t attribCount;
    struct {
        GLenum type;
        GLuint size;
        GLuint normalized;
        GLuint integer;
        GLintptr relativeoffset;
    } attribs[8];
} MGLStreamMergeExcludedLayout;

static const MGLStreamMergeExcludedLayout kStreamMergeExcludedLayouts[] = {
    {
        .name = "weather_particle",
        .vertexStride = 28u,
        .attribMask = 0xfu,
        .attribCount = 4,
        .attribs = {
            /* pos:   vec3 float @ 0  */
            { GL_FLOAT,         3, GL_FALSE, GL_FALSE, 0  },
            /* uv:    vec2 float @ 12 */
            { GL_FLOAT,         2, GL_FALSE, GL_FALSE, 12 },
            /* color: vec4 ubyte normalized @ 20 */
            { GL_UNSIGNED_BYTE, 4, GL_TRUE,  GL_FALSE, 20 },
            /* light: vec2 short integer @ 24 */
            { GL_SHORT,         2, GL_FALSE, GL_TRUE,  24 },
        },
    },
    {
        .name = "chunk_terrain",
        .vertexStride = 32u,
        .attribMask = 0x1fu,
        .attribCount = 5,
        .attribs = {
            /* pos:    vec3 float @ 0  */
            { GL_FLOAT,         3, GL_FALSE, GL_FALSE, 0  },
            /* color:  vec4 ubyte normalized @ 12 */
            { GL_UNSIGNED_BYTE, 4, GL_TRUE,  GL_FALSE, 12 },
            /* uv:     vec2 float @ 16 */
            { GL_FLOAT,         2, GL_FALSE, GL_FALSE, 16 },
            /* light:  vec2 short integer @ 24 */
            { GL_SHORT,         2, GL_FALSE, GL_TRUE,  24 },
            /* normal: vec3 byte normalized @ 28 */
            { GL_BYTE,          3, GL_TRUE,  GL_FALSE, 28 },
        },
    },
};

static bool mglAttribMatchesExclusion(const VertexAttrib *attr,
                                      const MGLStreamMergeExcludedLayout *layout,
                                      GLuint index)
{
    if (!attr || !layout || index >= layout->attribCount) return false;
    const typeof(layout->attribs[0]) *e = &layout->attribs[index];
    return attr->type == e->type &&
           attr->size == e->size &&
           attr->normalized == e->normalized &&
           attr->integer == e->integer &&
           attr->relativeoffset == e->relativeoffset;
}

static bool mglVAOMatchesExcludedLayout(const VertexArray *vao,
                                        uint32_t attribMask,
                                        size_t vertexStride,
                                        const MGLStreamMergeExcludedLayout **outLayout)
{
    if (outLayout) *outLayout = NULL;
    if (!vao) return false;

    const size_t count =
        sizeof(kStreamMergeExcludedLayouts) / sizeof(kStreamMergeExcludedLayouts[0]);
    for (size_t i = 0; i < count; i++) {
        const MGLStreamMergeExcludedLayout *layout = &kStreamMergeExcludedLayouts[i];
        if (vertexStride != layout->vertexStride) continue;
        if ((attribMask & layout->attribMask) != layout->attribMask) continue;

        bool match = true;
        for (GLuint a = 0; a < layout->attribCount; a++) {
            if (!mglAttribMatchesExclusion(&vao->attrib[a], layout, a)) {
                match = false;
                break;
            }
        }
        if (match) {
            if (outLayout) *outLayout = layout;
            return true;
        }
    }
    return false;
}

/* Weather-particle exclusion additionally requires blend-on + depth-write-off
 * to avoid false-positives on other stride-28 layouts.  Chunk-terrain needs
 * no extra state gating. */
static bool mglStreamMergeExclusionStateOK(GLMContext ctx,
                                           const MGLStreamMergeExcludedLayout *layout,
                                           const Buffer *vertexBuffer)
{
    if (!layout) return false;

    if (strcmp(layout->name, "weather_particle") == 0) {
        if (!ctx || !vertexBuffer) return false;
        if (vertexBuffer->usage != GL_DYNAMIC_DRAW) return false;
        if (vertexBuffer->size <= 0) return false;
        bool blendEnabled = (ctx->active_state->caps.blend == GL_TRUE);
        if (!blendEnabled) {
            GLuint maxDrawBuffers = ctx->active_state->var.max_draw_buffers;
            if (maxDrawBuffers > MAX_COLOR_ATTACHMENTS) maxDrawBuffers = MAX_COLOR_ATTACHMENTS;
            for (GLuint i = 0; i < maxDrawBuffers; i++) {
                if (ctx->active_state->caps.blendi[i] == GL_TRUE) {
                    blendEnabled = true;
                    break;
                }
            }
        }
        if (!blendEnabled || ctx->active_state->var.depth_writemask == GL_TRUE) return false;
        return true;
    }

    /* chunk_terrain — no extra state gating */
    return true;
}

static bool mglProgramUsesStreamMergeUnsafeBuiltin(const Program *program)
{
    return program &&
           (program->uses_vertex_id == GL_TRUE ||
            program->uses_primitive_id == GL_TRUE);
}

static bool mglCurrentProgramUsesStreamMergeUnsafeBuiltin(GLMContext ctx)
{
    if (!ctx) return true;

    if (ctx->active_state->program_name != 0u) {
        return mglProgramUsesStreamMergeUnsafeBuiltin(ctx->active_state->program);
    }

    ProgramPipeline *pipeline = ctx->active_state->program_pipeline;
    if (!pipeline && ctx->active_state->var.program_pipeline_binding != 0u) {
        pipeline = (ProgramPipeline *)searchHashTable(&ctx->active_state->program_pipeline_table,
                                                      ctx->active_state->var.program_pipeline_binding);
    }
    if (!pipeline) {
        return false;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (mglProgramUsesStreamMergeUnsafeBuiltin(pipeline->stage_programs[stage])) {
            return true;
        }
    }

    return false;
}

bool mglPrepareStreamMergeCandidate(GLMContext ctx,
                                           const MGLDrawCommand *cmd,
                                           bool uses_elements,
                                           MGLStreamMergeCandidate *out)
{
    if (!ctx || !cmd || !out) return false;
    {
        static bool sStreamMergeDisabled = false;
        static bool sStreamMergeDisableChecked = false;
        if (!sStreamMergeDisableChecked) {
            sStreamMergeDisabled = (getenv("MGL_DISABLE_STREAM_MERGE") != NULL);
            sStreamMergeDisableChecked = true;
        }
        if (sStreamMergeDisabled) return false;
    }
    if (cmd->mode != GL_TRIANGLES || cmd->count <= 0) return false;
    if (cmd->instanceCount != 1 || cmd->baseInstance != 0) return false;
    if (ctx->active_state->var.polygon_mode != GL_FILL) return false;
    if (mglCurrentProgramUsesStreamMergeUnsafeBuiltin(ctx)) {
        MGL_PERF_INC(g_mglMergeRejectUnsafeBuiltinSinceSwap);
        return false;
    }

    VertexArray *vao = ctx->active_state->vao;
    Buffer *indexBuffer = mglDrawCommandElementBuffer(ctx, cmd);
    size_t indexSize = uses_elements ? mglCommandIndexSize(cmd->indexType) : 0u;
    uint64_t indexOffset = (uint64_t)cmd->indexBufferOffset;
    const uint8_t *indices = NULL;
    if (!vao) {
        return false;
    }
    if (uses_elements) {
        uint64_t restart = 0;
        if (mglCommandPrimitiveRestartIndex(ctx, cmd->indexType, &restart)) {
            return false;
        }
        if (!indexBuffer || indexBuffer->mapped ||
            !indexBuffer->data.buffer_data || indexSize == 0u) {
            return false;
        }
        if (cmd->indexBufferOffset > (GLuint)indexBuffer->size) {
            return false;
        }
        if ((uint64_t)cmd->count > UINT64_MAX / (uint64_t)indexSize) {
            return false;
        }

        uint64_t indexBytes = (uint64_t)cmd->count * (uint64_t)indexSize;
        if (indexBytes > UINT64_MAX - indexOffset ||
            indexOffset + indexBytes > (uint64_t)indexBuffer->size) {
            return false;
        }
        indices = (const uint8_t *)(uintptr_t)indexBuffer->data.buffer_data + indexOffset;
    } else if (cmd->first < 0 ||
               (uint64_t)cmd->first > UINT64_MAX - ((uint64_t)cmd->count - 1u)) {
        return false;
    }

    GLuint maxAttribs = MAX_ATTRIBS;

    bool explicitAttribs = (vao->enabled_attribs != 0u);
    Buffer *vertexBuffer = NULL;
    size_t vertexStride = 0;
    GLintptr bindingOffset = -1;
    size_t maxAttribEnd = 0;
    uint32_t attribMask = 0;
    uint64_t layoutHash = 0x9e3779b97f4a7c15ULL;

    for (GLuint i = 0; i < maxAttribs; i++) {
        bool useAttrib = explicitAttribs
            ? ((vao->enabled_attribs & (1u << i)) != 0u)
            : (vao->attrib[i].buffer != NULL);
        if (!useAttrib) {
            if (explicitAttribs && (vao->enabled_attribs >> (i + 1u)) == 0u) break;
            continue;
        }

        MGLCommandResolvedAttrib resolved = {0};
        if (!mglCommandResolveVertexAttrib(vao, i, &resolved) ||
            resolved.divisor != 0u ||
            resolved.binding_offset < 0 || resolved.relativeoffset < 0) {
            return false;
        }

        VertexAttrib *attrib = resolved.attrib;
        size_t elemBytes = mglCommandAttribElementBytes(attrib->type, attrib->size);
        size_t stride = resolved.stride > 0 ? (size_t)resolved.stride : elemBytes;
        if (elemBytes == 0u || stride == 0u) return false;
        if ((size_t)resolved.relativeoffset > SIZE_MAX - elemBytes) return false;
        size_t attribEnd = (size_t)resolved.relativeoffset + elemBytes;
        if (attribEnd > stride) return false;

        if (!vertexBuffer) {
            vertexBuffer = resolved.buffer;
            vertexStride = stride;
            bindingOffset = resolved.binding_offset;
        } else if (vertexBuffer != resolved.buffer ||
                   vertexStride != stride ||
                   bindingOffset != resolved.binding_offset) {
            return false;
        }

        attribMask |= (1u << i);
        if (attribEnd > maxAttribEnd) maxAttribEnd = attribEnd;
        layoutHash = mglStreamHashAttrib(layoutHash, &resolved, i);
    }

    if (!vertexBuffer || vertexBuffer->mapped ||
        attribMask == 0u || !vertexBuffer->data.buffer_data ||
        vertexBuffer->size <= 0) {
        return false;
    }
    /* Stream merge copies the CPU shadow into a transient VBO.  Buffers that
     * shaders may have written (SSBO/XFB/atomic) can hold newer bytes only in
     * Metal — merging from the shadow drops those stores (CTS
     * advanced-write-geometry phase2). */
    if (vertexBuffer->gpu_write_target) {
        return false;
    }
    if (bindingOffset < 0 || vertexStride == 0u ||
        ((size_t)bindingOffset % vertexStride) != 0u ||
        ((size_t)vertexBuffer->size % vertexStride) != 0u) {
        return false;
    }
    /* Check the draw's actual vertex data footprint, not the source buffer
     * size.  Large buffers (e.g., Minecraft's GUI vertex buffer) are fine as
     * long as the per-draw referenced vertices fit within the transient
     * batch budget.  The 4MB batch limit (checked later in append) prevents
     * unbounded transient buffer growth. */
    if (vertexStride > 0 &&
        (uint64_t)cmd->count * (uint64_t)vertexStride > (uint64_t)MGL_STREAM_MERGE_MAX_SOURCE_BYTES) {
        return false;
    }
    /* Exclude known-incompatible vertex layouts from stream merging.
     * See kStreamMergeExcludedLayouts above for rationale.  Env var
     * MGL_DISABLE_STREAM_MERGE_EXCLUSIONS=1 bypasses for testing. */
    {
        static bool sExclusionsDisabled = false;
        static bool sExclusionsChecked = false;
        if (!sExclusionsChecked) {
            sExclusionsDisabled = (getenv("MGL_DISABLE_STREAM_MERGE_EXCLUSIONS") != NULL);
            sExclusionsChecked = true;
        }
        if (!sExclusionsDisabled) {
            const MGLStreamMergeExcludedLayout *excluded = NULL;
            if (mglVAOMatchesExcludedLayout(vao, attribMask, vertexStride, &excluded)) {
                if (mglStreamMergeExclusionStateOK(ctx, excluded, vertexBuffer)) {
                    static bool sDebugLog = false;
                    static bool sDebugChecked = false;
                    if (!sDebugChecked) {
                        sDebugLog = (getenv("MGL_DEBUG_STREAM_MERGE") != NULL);
                        sDebugChecked = true;
                    }
                    if (sDebugLog && excluded) {
                        fprintf(stderr,
                                "MGL DEBUG: stream-merge excluded layout '%s' "
                                "(stride=%zu attribMask=0x%x)\n",
                                excluded->name, vertexStride, attribMask);
                    }
                    MGL_PERF_INC(g_mglMergeRejectExcludedLayoutSinceSwap);
                    return false;
                } else {
                    MGL_PERF_INC(g_mglMergeRejectBufferHazardSinceSwap);
                }
            }
        }
    }

    uint64_t minSourceIndex = UINT64_MAX;
    uint64_t maxSourceIndex = 0u;
    if (uses_elements) {
        /* Restart is disabled on this path (gated above), so the cached
         * all-index range carries the same information the per-index loop
         * used to recompute on every draw: bounds failure is monotonic in
         * the source index, so checking the range endpoints is equivalent
         * to checking every index. */
        MGLCommandElementIndexScan scan;
        if (!mglScanCommandElementIndices(indexBuffer, cmd->indexType, indexOffset,
                                          cmd->count, false, 0, &scan)) {
            return false;
        }
        int64_t minSource = (int64_t)scan.all_min_index + (int64_t)cmd->baseVertex;
        int64_t maxSource = (int64_t)scan.all_max_index + (int64_t)cmd->baseVertex;
        if (minSource < 0) return false;
        uint64_t minSourceU = (uint64_t)minSource;
        uint64_t maxSourceU = (uint64_t)maxSource;
        if (maxSourceU > (UINT64_MAX - (uint64_t)bindingOffset) / (uint64_t)vertexStride) {
            return false;
        }
        uint64_t maxVertexByte = (uint64_t)bindingOffset +
                                 (maxSourceU * (uint64_t)vertexStride);
        if (maxVertexByte > UINT64_MAX - (uint64_t)maxAttribEnd ||
            maxVertexByte + (uint64_t)maxAttribEnd > (uint64_t)vertexBuffer->size) {
            return false;
        }
        minSourceIndex = minSourceU;
        maxSourceIndex = maxSourceU;
    } else {
        /* Non-indexed draws reference a linear vertex sequence; the source
         * index range is closed-form, so skip the per-vertex scan. */
        uint64_t lastSourceIndex = (uint64_t)cmd->first + (uint64_t)cmd->count - 1u;
        if (lastSourceIndex > (UINT64_MAX - (uint64_t)bindingOffset) / (uint64_t)vertexStride) {
            return false;
        }
        uint64_t vertexByte = (uint64_t)bindingOffset +
                              (lastSourceIndex * (uint64_t)vertexStride);
        if (vertexByte > UINT64_MAX - (uint64_t)maxAttribEnd ||
            vertexByte + (uint64_t)maxAttribEnd > (uint64_t)vertexBuffer->size) {
            return false;
        }
        minSourceIndex = (uint64_t)cmd->first;
        maxSourceIndex = lastSourceIndex;
    }
    if (minSourceIndex == UINT64_MAX ||
        minSourceIndex > maxSourceIndex ||
        minSourceIndex > (UINT64_MAX - (uint64_t)bindingOffset) / (uint64_t)vertexStride ||
        maxSourceIndex > (UINT64_MAX - (uint64_t)bindingOffset) / (uint64_t)vertexStride) {
        return false;
    }

    uint64_t sourceStart = (uint64_t)bindingOffset +
                           minSourceIndex * (uint64_t)vertexStride;
    uint64_t sourceEnd = (uint64_t)bindingOffset +
                         maxSourceIndex * (uint64_t)vertexStride;
    if (sourceEnd > UINT64_MAX - (uint64_t)maxAttribEnd ||
        sourceEnd + (uint64_t)maxAttribEnd > (uint64_t)vertexBuffer->size ||
        sourceStart > sourceEnd + (uint64_t)maxAttribEnd ||
        sourceEnd + (uint64_t)maxAttribEnd - sourceStart > SIZE_MAX) {
        return false;
    }
    size_t sourceBytes = (size_t)(sourceEnd + (uint64_t)maxAttribEnd - sourceStart);

    memset(out, 0, sizeof(*out));
    out->vao = vao;
    out->vertex_buffer = vertexBuffer;
    out->index_buffer = indexBuffer;
    out->index_bytes = indices;
    out->index_size = indexSize;
    out->uses_elements = uses_elements;
    out->vertex_bytes = sourceBytes;
    out->vertex_source_offset = (size_t)sourceStart;
    out->vertex_stride = vertexStride;
    out->max_attrib_end = maxAttribEnd;
    out->source_first_index = minSourceIndex;
    out->source_last_index = maxSourceIndex;
    out->binding_offset = bindingOffset;
    out->attrib_mask = attribMask;
    out->layout_hash = layoutHash ^
                       mglRotateLeft64((uint64_t)vertexStride, 37) ^
                       mglRotateLeft64((uint64_t)bindingOffset, 43);
    return true;
}

bool mglInitializeStreamMergedBatch(GLMContext ctx,
                                           MGLDrawBatch *batch,
                                           const MGLStreamMergeCandidate *candidate)
{
    MGL_SIGNPOST_BEGIN(InitStreamMergedBatch);
    if (!ctx || !batch || !candidate || !candidate->vao) {
        MGL_SIGNPOST_END(InitStreamMergedBatch);
        return false;
    }

    MGLBatchArena *arena = ctx->batch_arena;
    if (arena && arena->enabled) {
        batch->state_snapshot = arenaAlloc(arena, sizeof(GLMState));
        batch->vao_snapshot = arenaAlloc(arena, sizeof(VertexArray));
        batch->arena_managed = true;
    } else {
        batch->state_snapshot = malloc(sizeof(GLMState));
        batch->vao_snapshot = malloc(sizeof(VertexArray));
        batch->arena_managed = false;
    }
    batch->stream_vertex_buffer = mglAcquireTransientBatchBuffer(ctx, GL_ARRAY_BUFFER);
    batch->stream_index_buffer = mglAcquireTransientBatchBuffer(ctx, GL_ELEMENT_ARRAY_BUFFER);
    if (!batch->state_snapshot || !batch->vao_snapshot ||
        !batch->stream_vertex_buffer || !batch->stream_index_buffer) {
        MGL_SIGNPOST_END(InitStreamMergedBatch);
        return false;
    }

    mglCopyHotStateFields((GLMState *)batch->state_snapshot, ctx->active_state);
    memcpy(batch->vao_snapshot, candidate->vao, sizeof(VertexArray));

    VertexArray *vao = (VertexArray *)batch->vao_snapshot;
    Buffer *vertexBuffer = (Buffer *)batch->stream_vertex_buffer;
    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;

    vao->transient_batch_vao = GL_TRUE;
    /* vao_snapshot is a shallow copy of the live VAO — do NOT release the
     * Metal object through a callback; the live VAO still owns it.  Only
     * nullify the pointer so batch replay doesn't reference the live VAO's
     * Metal data. */
    vao->dirty_bits |= DIRTY_VAO_ATTRIB | DIRTY_VAO_BUFFER_BASE;
    vao->element_array.buffer = indexBuffer;

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if ((candidate->attrib_mask & (1u << i)) == 0u) continue;
        GLuint binding = vao->attrib[i].buffer_bindingindex;
        vao->attrib[i].buffer = vertexBuffer;
        vao->attrib[i].binding_offset = 0;
        if (binding < MGL_MAX_VERTEX_ATTRIB_BINDINGS &&
            vao->bindings[binding].buffer == candidate->vertex_buffer) {
            vao->bindings[binding].buffer = vertexBuffer;
            vao->bindings[binding].offset = 0;
        }
    }

    GLMState *snapshot = (GLMState *)batch->state_snapshot;
    snapshot->vao = vao;
    snapshot->buffers[_ARRAY_BUFFER] = vertexBuffer;
    snapshot->buffers[_ELEMENT_ARRAY_BUFFER] = indexBuffer;
    snapshot->var.array_buffer_binding = 0;
    snapshot->var.element_array_buffer_binding = 0;
    snapshot->dirty_bits |= DIRTY_VAO | DIRTY_BUFFER;

    batch->stream_merged = true;
    batch->stream_layout_hash = candidate->layout_hash;
    batch->stream_vertex_stride = candidate->vertex_stride;
    batch->mdi_compatible = true;
    /* Record-time merge comparisons need the draw's real bindings; the
     * transient vao_snapshot carries the merged scratch buffers instead. */
    batch->source_vao = candidate->vao;

    MGL_PERF_INC(g_mglSnapshotAllocationCountSinceSwap);
    /* Report the bytes actually written (hot state copy + VAO freeze), not
     * the allocation size: this counter is the snapshot-cost metric the
     * state-machine migration tracks (STATE_MACHINE_REVIEW 4.3).  The
     * allocation itself remains sizeof(GLMState)+sizeof(VertexArray) per
     * batch regardless. */
    MGL_PERF_ADD(g_mglSnapshotBytesAllocatedSinceSwap,
                 mglSnapshotHotStateBytes() +
                 (batch->vao_snapshot ? sizeof(VertexArray) : 0));

    mglRetainBatchProgramReferences(ctx, batch);
    mglRetainBatchBufferReferences(batch);
    MGL_SIGNPOST_END(InitStreamMergedBatch);
    return true;
}

bool mglAppendStreamMergedData(MGLDrawBatch *batch,
                                      const MGLStreamMergeCandidate *candidate,
                                      const MGLDrawCommand *srcCmd,
                                      MGLDrawCommand *storedCmd)
{
    if (!batch || !candidate || !srcCmd || !storedCmd ||
        !batch->stream_vertex_buffer || !batch->stream_index_buffer) {
        return false;
    }

    Buffer *vertexBuffer = (Buffer *)batch->stream_vertex_buffer;
    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;

    size_t vertexOffset = mglAlignUpSize(batch->stream_vertex_bytes, candidate->vertex_stride);
    if (vertexOffset == SIZE_MAX ||
        vertexOffset > MGL_STREAM_MERGE_MAX_BATCH_BYTES ||
        candidate->vertex_bytes > MGL_STREAM_MERGE_MAX_BATCH_BYTES - vertexOffset) {
        return false;
    }
    size_t newVertexBytes = vertexOffset + candidate->vertex_bytes;

    uint64_t vertexBase = (uint64_t)vertexOffset / (uint64_t)candidate->vertex_stride;

    if ((uint64_t)srcCmd->count > (UINT64_MAX / sizeof(uint32_t)) ||
        (size_t)srcCmd->count > (MGL_STREAM_MERGE_MAX_BATCH_BYTES - batch->stream_index_bytes) / sizeof(uint32_t)) {
        return false;
    }
    size_t indexWriteOffset = batch->stream_index_bytes;
    size_t indexBytesToAppend = (size_t)srcCmd->count * sizeof(uint32_t);
    size_t newIndexBytes = indexWriteOffset + indexBytesToAppend;

    /* mglPrepareStreamMergeCandidate already walked these same indices in
     * this record call (readability, sign, min/max), so the per-index
     * re-validation reduces to bounding the largest merged index. */
    if (candidate->source_last_index < candidate->source_first_index) return false;
    uint64_t maxMergedIndex = vertexBase +
        (candidate->source_last_index - candidate->source_first_index);
    if (maxMergedIndex > UINT32_MAX) return false;

    if (!mglEnsureTransientBufferCapacity(vertexBuffer, newVertexBytes) ||
        !mglEnsureTransientBufferCapacity(indexBuffer, newIndexBytes)) {
        return false;
    }

    uint8_t *vertexDst = (uint8_t *)(uintptr_t)vertexBuffer->data.buffer_data;
    if (vertexOffset > batch->stream_vertex_bytes) {
        memset(vertexDst + batch->stream_vertex_bytes, 0, vertexOffset - batch->stream_vertex_bytes);
    }
    memcpy(vertexDst + vertexOffset,
           (const uint8_t *)(uintptr_t)candidate->vertex_buffer->data.buffer_data +
               candidate->vertex_source_offset,
           candidate->vertex_bytes);

    uint32_t *indexDst = (uint32_t *)((uint8_t *)(uintptr_t)indexBuffer->data.buffer_data + indexWriteOffset);
    for (GLsizei i = 0; i < srcCmd->count; i++) {
        uint64_t rawIndex = (uint64_t)srcCmd->first + (uint64_t)i;
        if (candidate->uses_elements) {
            (void)mglCommandReadIndexValue(candidate->index_bytes, srcCmd->indexType, i, &rawIndex);
        }
        int64_t sourceIndex = candidate->uses_elements
            ? (int64_t)rawIndex + (int64_t)srcCmd->baseVertex
            : (int64_t)rawIndex;
        indexDst[i] = (uint32_t)(vertexBase +
                                 ((uint64_t)sourceIndex - candidate->source_first_index));
    }

    batch->stream_vertex_bytes = newVertexBytes;
    batch->stream_index_bytes = newIndexBytes;
    batch->stream_index_count += (size_t)srcCmd->count;
    mglMarkTransientBufferWritten(vertexBuffer, batch->stream_vertex_bytes);
    mglMarkTransientBufferWritten(indexBuffer, batch->stream_index_bytes);

    uint16_t sampler_snapshot_id = storedCmd->sampler_snapshot_id;
    *storedCmd = *srcCmd;
    storedCmd->sampler_snapshot_id = sampler_snapshot_id;
    storedCmd->indexType = GL_UNSIGNED_INT;
    storedCmd->indexBufferOffset = (GLuint)indexWriteOffset;
    storedCmd->element_buffer_name = indexBuffer ? indexBuffer->name : 0u;
    storedCmd->baseVertex = 0;
    storedCmd->baseInstance = 0;
    return true;
}

void mglTrackPendingDrawBufferReads(GLMContext ctx,
                                           const MGLDrawCommand *cmd,
                                           bool uses_elements)
{
    if (!ctx || !cmd) return;

    if (uses_elements && cmd->element_buffer_name && cmd->count > 0) {
        Buffer *elementBuffer = mglDrawCommandElementBuffer(ctx, cmd);
        if (!elementBuffer)
            return;
        size_t indexSize = mglCommandIndexSize(cmd->indexType);
        if (indexSize > 0) {
            uint64_t indexBytes = (uint64_t)cmd->count * (uint64_t)indexSize;
            mglTrackPendingReadBytes(ctx,
                                     elementBuffer,
                                     (uint64_t)cmd->indexBufferOffset,
                                     indexBytes);
        } else {
            mglTrackPendingReadWholeBuffer(ctx, elementBuffer);
        }
    }

    uint64_t minVertex = 0;
    uint64_t maxVertex = 0;
    bool rangeKnown = uses_elements
        ? mglCommandComputeElementVertexRange(ctx, cmd, &minVertex, &maxVertex)
        : mglCommandComputeArrayVertexRange(cmd, &minVertex, &maxVertex);

    VertexArray *vao = ctx->active_state->vao;
    if (vao) {
        GLuint maxAttribs = MAX_ATTRIBS;

        bool explicitAttribs = (vao->enabled_attribs != 0u);
        for (GLuint i = 0; i < maxAttribs; i++) {
            bool useAttrib = explicitAttribs
                ? ((vao->enabled_attribs & (1u << i)) != 0u)
                : (mglCommandResolveVertexAttrib(vao, i, &(MGLCommandResolvedAttrib){0}));
            if (!useAttrib) {
                if (explicitAttribs && (vao->enabled_attribs >> (i + 1u)) == 0u) break;
                continue;
            }

            MGLCommandResolvedAttrib resolved = {0};
            if (!mglCommandResolveVertexAttrib(vao, i, &resolved)) continue;

            if (rangeKnown) {
                mglTrackPendingAttribRead(ctx,
                                          &resolved,
                                          minVertex,
                                          maxVertex,
                                          cmd->instanceCount,
                                          cmd->baseInstance);
            } else {
                mglTrackPendingReadWholeBuffer(ctx, resolved.buffer);
            }
        }
    }

    mglTrackPendingBaseBufferReads(ctx);
}

