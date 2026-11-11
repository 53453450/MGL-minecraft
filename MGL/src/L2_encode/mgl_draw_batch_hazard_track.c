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

static uint32_t mglEnvUInt32Clamped(const char *name,
                                    uint32_t defaultValue,
                                    uint32_t minValue,
                                    uint32_t maxValue)
{
    const char *env = name ? getenv(name) : NULL;
    if (!env || env[0] == '\0') {
        return defaultValue;
    }

    char *end = NULL;
    unsigned long parsed = strtoul(env, &end, 10);
    if (end == env || parsed == 0ul) {
        return defaultValue;
    }

    if (parsed < (unsigned long)minValue) {
        return minValue;
    }
    if (parsed > (unsigned long)maxValue) {
        return maxValue;
    }
    return (uint32_t)parsed;
}

uint32_t mglRuntimeMaxDrawsPerBatch(void)
{
    static uint32_t s_value = 0u;
    if (s_value == 0u) {
        s_value = mglEnvUInt32Clamped("MGL_BATCH_MAX_DRAWS",
                                      MGL_MAX_DRAWS_PER_BATCH,
                                      1u,
                                      MGL_MAX_DRAWS_PER_BATCH);
    }
    return s_value;
}

uint32_t mglRuntimeMaxBatchCount(void)
{
    static uint32_t s_value = 0u;
    if (s_value == 0u) {
        s_value = mglEnvUInt32Clamped("MGL_BATCH_MAX_COUNT",
                                      MGL_MAX_BATCHES,
                                      1u,
                                      MGL_MAX_BATCHES);
    }
    return s_value;
}

/* MGL_BIND_NO_FLUSH: default ON; =0/false/no/off disables. Cached process-wide. */
int mglBindNoFlushEnabled(void)
{
    static _Atomic int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_acquire);
    if (v < 0) {
        const char *value = getenv("MGL_BIND_NO_FLUSH");
        if (!value || value[0] == '\0') {
            v = 1;
        } else if (strcmp(value, "0") == 0 ||
                   strcasecmp(value, "false") == 0 ||
                   strcasecmp(value, "no") == 0 ||
                   strcasecmp(value, "off") == 0) {
            v = 0;
        } else {
            v = 1;
        }
        atomic_store_explicit(&cached, v, memory_order_release);
    }
    return v != 0;
}

int mglSamplerSnapshotEnabled(void)
{
    static _Atomic int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_acquire);
    if (v < 0) {
        const char *value = getenv("MGL_DRAW_SAMPLER_SNAPSHOT");
        if (!value || value[0] == '\0') {
            v = 1;
        } else if (strcmp(value, "0") == 0 ||
                   strcasecmp(value, "false") == 0 ||
                   strcasecmp(value, "no") == 0 ||
                   strcasecmp(value, "off") == 0) {
            v = 0;
        } else {
            v = 1;
        }
        atomic_store_explicit(&cached, v, memory_order_release);
    }
    return v != 0;
}

static bool mglSamplerSnapshotSupportsParameter(GLenum pname)
{
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:
        case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
        case GL_TEXTURE_WRAP_R:
        case GL_TEXTURE_MIN_LOD:
        case GL_TEXTURE_MAX_LOD:
        case GL_TEXTURE_COMPARE_MODE:
        case GL_TEXTURE_COMPARE_FUNC:
        case GL_TEXTURE_BORDER_COLOR:
        case GL_TEXTURE_MAX_ANISOTROPY:
            return true;
        default:
            return false;
    }
}

bool mglSamplerSnapshotCanDeferParameter(GLMContext ctx, GLenum pname)
{
    return ctx && ctx->draw_defer_enabled &&
           mglSamplerSnapshotEnabled() &&
           mglSamplerSnapshotSupportsParameter(pname) &&
           !ctx->draw_command_buffer.sampler_snapshot_incomplete;
}

static void mglNormalizeMutationRange(int64_t offset, int64_t size, uint64_t *start, uint64_t *end)
{
    if (!start || !end) return;

    if (offset < 0 || size < 0) {
        *start = 0;
        *end = UINT64_MAX;
        return;
    }

    *start = (uint64_t)offset;
    if (size == 0) {
        *end = *start;
        return;
    }

    if ((uint64_t)size > UINT64_MAX - *start) {
        *end = UINT64_MAX;
    } else {
        *end = *start + (uint64_t)size;
    }
}

static inline uint32_t mglBufferRangeBucket(const void *buffer)
{
    uintptr_t h = (uintptr_t)buffer;
    h >>= 4u;
    h ^= h >> 17u;
    h *= UINT64_C(0x9e3779b97f4a7c15);
    h ^= h >> 29u;
    return (uint32_t)h & MGL_BUFFER_RANGE_BUCKET_MASK;
}


/* O2.2: hazard overflow policy lives in mgl_batch_hazard.*; ObjC never chooses. */
static int mglHazardHasPendingDraws(const MGLCommandBuffer *cb)
{
    return cb && cb->batch_count > 0u && cb->total_commands > 0u;
}

static MGLHazardOverflowPolicy mglHazardPolicyNow(void)
{
    /* PERF-002: default flush-and-continue so overflow does not sticky-degrade
     * every subsequent hazard query. Opt into legacy sticky via
     * MGL_HAZARD_OVERFLOW_STICKY=1. MGL_HAZARD_OVERFLOW_FLUSH_CONTINUE remains
     * accepted as an explicit (now redundant) enable. */
    if (mgl_env_flag_enabled("MGL_HAZARD_OVERFLOW_STICKY")) {
        return MGL_HAZARD_OVERFLOW_STICKY;
    }
    (void)mgl_env_flag_enabled("MGL_HAZARD_OVERFLOW_FLUSH_CONTINUE");
    return MGL_HAZARD_OVERFLOW_FLUSH_AND_CONTINUE;
}

/* Capacity exhausted: sticky latches overflow (return 0); flush-and-continue
 * flushes pending draws (tables reset) and returns 1 so the caller can insert. */
static void mglHazardClearOverflowLatchFlags(MGLCommandBuffer *cb)
{
    if (!cb) return;
    cb->buffer_read_range_overflow = false;
    cb->texture_read_overflow = false;
    cb->texture_write_overflow = false;
    cb->pending_vao_ref_overflow = false;
    cb->pending_vao_name_overflow = false;
}

static int mglHazardAllowInsertAfterCapacityFull(GLMContext ctx,
                                                 MGLCommandBuffer *cb)
{
    MGLHazardTrackAction action = mgl_batch_hazard_on_capacity_full(
        mglHazardPolicyNow(), mglHazardHasPendingDraws(cb));
    if (action == MGL_HAZARD_TRACK_FLUSH_THEN_RETRY) {
        /* PERF-015: flush-and-continue resets hazard tables (and overflow
         * latches) via mglResetCommandBufferForContext; clear latches first so
         * a partial flush path never leaves query_degraded sticky. Sticky
         * degradation remains only with MGL_HAZARD_OVERFLOW_STICKY=1 or
         * capacity-full with zero pending batches. */
        mglHazardClearOverflowLatchFlags(cb);
        mglFlushCommandBuffer(ctx);
        return 1;
    }
    return 0;
}

/* Grow heap-backed buffer_read_ranges / next up to MGL_MAX_PENDING_BUFFER_RANGES. */
static int mglEnsureBufferReadRangeCapacity(MGLCommandBuffer *cb, uint32_t need)
{
    if (!cb || need == 0) return 0;
    if (need <= cb->buffer_read_range_capacity) return 1;
    if (need > MGL_MAX_PENDING_BUFFER_RANGES) return 0;

    uint32_t new_cap = cb->buffer_read_range_capacity
                           ? cb->buffer_read_range_capacity * 2u
                           : 64u;
    if (new_cap < need) new_cap = need;
    if (new_cap > MGL_MAX_PENDING_BUFFER_RANGES)
        new_cap = MGL_MAX_PENDING_BUFFER_RANGES;

    MGLBufferReadRange *ranges = (MGLBufferReadRange *)realloc(
        cb->buffer_read_ranges, (size_t)new_cap * sizeof(*ranges));
    uint32_t *next = (uint32_t *)realloc(
        cb->buffer_read_range_next, (size_t)new_cap * sizeof(*next));
    if (!ranges || !next) {
        /* realloc may have moved one side; keep whatever pointers we still own. */
        if (ranges) cb->buffer_read_ranges = ranges;
        if (next) cb->buffer_read_range_next = next;
        return 0;
    }
    if (new_cap > cb->buffer_read_range_capacity) {
        memset(next + cb->buffer_read_range_capacity, 0,
               (size_t)(new_cap - cb->buffer_read_range_capacity) * sizeof(*next));
    }
    cb->buffer_read_ranges = ranges;
    cb->buffer_read_range_next = next;
    cb->buffer_read_range_capacity = new_cap;
    return 1;
}

static int mglEnsureTextureWriteCapacity(MGLCommandBuffer *cb, uint32_t need)
{
    if (!cb || need == 0) return 0;
    if (need <= cb->texture_write_capacity) return 1;
    if (need > MGL_MAX_PENDING_TEXTURE_WRITES) return 0;
    uint32_t new_cap = cb->texture_write_capacity ? cb->texture_write_capacity * 2u : 32u;
    if (new_cap < need) new_cap = need;
    if (new_cap > MGL_MAX_PENDING_TEXTURE_WRITES) new_cap = MGL_MAX_PENDING_TEXTURE_WRITES;
    void **objs = (void **)realloc(cb->texture_write_objects,
                                   (size_t)new_cap * sizeof(*objs));
    if (!objs) return 0;
    cb->texture_write_objects = objs;
    cb->texture_write_capacity = new_cap;
    return 1;
}

static int mglEnsureTextureReadCapacity(MGLCommandBuffer *cb, uint32_t need)
{
    if (!cb || need == 0) return 0;
    if (need <= cb->texture_read_capacity) return 1;
    if (need > MGL_MAX_PENDING_TEXTURE_READS) return 0;
    uint32_t new_cap = cb->texture_read_capacity ? cb->texture_read_capacity * 2u : 64u;
    if (new_cap < need) new_cap = need;
    if (new_cap > MGL_MAX_PENDING_TEXTURE_READS) new_cap = MGL_MAX_PENDING_TEXTURE_READS;
    void **objs = (void **)realloc(cb->texture_read_objects,
                                   (size_t)new_cap * sizeof(*objs));
    if (!objs) return 0;
    cb->texture_read_objects = objs;
    cb->texture_read_capacity = new_cap;
    return 1;
}

static int mglEnsureSamplerKeyCapacity(MGLCommandBuffer *cb, uint32_t need)
{
    if (!cb || need == 0) return 0;
    if (need <= cb->sampler_snapshot_key_capacity) return 1;
    if (need > MGL_MAX_SAMPLER_SNAPSHOT_KEYS) return 0;
    uint32_t new_cap = cb->sampler_snapshot_key_capacity
                           ? (uint32_t)cb->sampler_snapshot_key_capacity * 2u
                           : 16u;
    if (new_cap < need) new_cap = need;
    if (new_cap > MGL_MAX_SAMPLER_SNAPSHOT_KEYS) new_cap = MGL_MAX_SAMPLER_SNAPSHOT_KEYS;
    MGLSamplerSnapshotKey *keys = (MGLSamplerSnapshotKey *)realloc(
        cb->sampler_snapshot_keys, (size_t)new_cap * sizeof(*keys));
    if (!keys) return 0;
    cb->sampler_snapshot_keys = keys;
    cb->sampler_snapshot_key_capacity = (uint16_t)new_cap;
    return 1;
}

static int mglEnsureSamplerSetCapacity(MGLCommandBuffer *cb, uint32_t need)
{
    if (!cb || need == 0) return 0;
    if (need <= cb->sampler_snapshot_set_capacity) return 1;
    if (need > MGL_MAX_SAMPLER_SNAPSHOT_SETS) return 0;
    uint32_t new_cap = cb->sampler_snapshot_set_capacity
                           ? (uint32_t)cb->sampler_snapshot_set_capacity * 2u
                           : 16u;
    if (new_cap < need) new_cap = need;
    if (new_cap > MGL_MAX_SAMPLER_SNAPSHOT_SETS) new_cap = MGL_MAX_SAMPLER_SNAPSHOT_SETS;
    MGLSamplerSnapshotSet *sets = (MGLSamplerSnapshotSet *)realloc(
        cb->sampler_snapshot_sets, (size_t)new_cap * sizeof(*sets));
    if (!sets) return 0;
    cb->sampler_snapshot_sets = sets;
    cb->sampler_snapshot_set_capacity = (uint16_t)new_cap;
    return 1;
}

void mglTrackPendingReadRange(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t end)
{
    if (!ctx || !buffer || end <= start) return;

    /* Walk only this buffer's bucket chain instead of every recorded range —
     * the previous full scan made a flush window with N ranges cost O(N)
     * per insert (O(N²) total). */
    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    uint32_t bucket = mglBufferRangeBucket(buffer);
    if (cb->buffer_read_range_next) {
        for (uint32_t link = cb->buffer_read_range_bucket[bucket]; link;
             link = cb->buffer_read_range_next[link - 1u]) {
            MGLBufferReadRange *range = &cb->buffer_read_ranges[link - 1u];
            if (range->buffer == buffer &&
                mglRangesOverlap(range->start, range->end, start, end)) {
                if (start < range->start) range->start = start;
                if (end > range->end) range->end = end;
                return;
            }
        }
    }

    if (cb->buffer_read_range_count >= MGL_MAX_PENDING_BUFFER_RANGES) {
        if (!mglHazardAllowInsertAfterCapacityFull(ctx, cb) ||
            cb->buffer_read_range_count >= MGL_MAX_PENDING_BUFFER_RANGES) {
            cb->buffer_read_range_overflow = true;
            return;
        }
    }

    if (!mglEnsureBufferReadRangeCapacity(cb, cb->buffer_read_range_count + 1u)) {
        cb->buffer_read_range_overflow = true;
        return;
    }

    uint32_t index = cb->buffer_read_range_count++;
    MGLBufferReadRange *range = &cb->buffer_read_ranges[index];
    range->buffer = buffer;
    range->start = start;
    range->end = end;
    cb->buffer_read_range_next[index] = cb->buffer_read_range_bucket[bucket];
    cb->buffer_read_range_bucket[bucket] = index + 1u;

    mglNoteOccupancyPeak(&s_peak_buffer_read_ranges, cb->buffer_read_range_count);
    MGL_PERF_INC(g_mglHazardRangeCountSinceSwap);
}

void mglTrackPendingReadBytes(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t size)
{
    uint64_t end = (size > UINT64_MAX - start) ? UINT64_MAX : start + size;
    mglTrackPendingReadRange(ctx, buffer, start, end);
}

void mglTrackPendingReadWholeBuffer(GLMContext ctx, Buffer *buffer)
{
    if (!buffer) return;
    uint64_t end = buffer->size > 0 ? (uint64_t)buffer->size : UINT64_MAX;
    mglTrackPendingReadRange(ctx, buffer, 0, end);
}

static Texture *mglPendingDrawAttachmentTexture(FBOAttachment *attachment)
{
    if (!attachment) return NULL;

    if (attachment->textarget == GL_RENDERBUFFER) {
        return (attachment->buf.rbo && attachment->buf.rbo->tex) ? attachment->buf.rbo->tex : NULL;
    }

    return attachment->buf.tex;
}

static bool mglDrawBufferToColorAttachmentIndex(GLMContext ctx, GLenum buffer, GLuint *outIndex)
{
    if (!ctx || !outIndex) return false;
    if (buffer < GL_COLOR_ATTACHMENT0) return false;

    GLuint index = (GLuint)(buffer - GL_COLOR_ATTACHMENT0);
    if (index >= ctx->active_state->max_color_attachments || index >= MAX_COLOR_ATTACHMENTS) {
        return false;
    }

    *outIndex = index;
    return true;
}

static void mglTrackPendingTextureWrite(GLMContext ctx, Texture *texture);

static void mglTrackPendingFramebufferDepthStencilWrites(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx || !fbo) return;

    mglTrackPendingTextureWrite(ctx, mglPendingDrawAttachmentTexture(&fbo->depth));
    if (fbo->stencil.buf.tex != fbo->depth.buf.tex ||
        fbo->stencil.buf.rbo != fbo->depth.buf.rbo ||
        fbo->stencil.textarget != fbo->depth.textarget) {
        mglTrackPendingTextureWrite(ctx, mglPendingDrawAttachmentTexture(&fbo->stencil));
    }
}

static void mglTrackPendingTextureWrite(GLMContext ctx, Texture *texture)
{
    if (!ctx || !texture) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;

    /* O(1) dedup via hash-set index instead of O(n) linear scan. */
    uint32_t slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_WRITE_INDEX_MASK);
    if (cb->texture_write_objects) {
        for (;;) {
            uint32_t entry = cb->texture_write_index[slot];
            if (entry == 0) break;
            if (cb->texture_write_objects[entry - 1] == texture) return;
            slot = (slot + 1) & MGL_TEX_WRITE_INDEX_MASK;
        }
    } else {
        while (cb->texture_write_index[slot] != 0)
            slot = (slot + 1) & MGL_TEX_WRITE_INDEX_MASK;
    }

    if (cb->texture_write_count >= MGL_MAX_PENDING_TEXTURE_WRITES) {
        if (!mglHazardAllowInsertAfterCapacityFull(ctx, cb) ||
            cb->texture_write_count >= MGL_MAX_PENDING_TEXTURE_WRITES) {
            cb->texture_write_overflow = true;
            return;
        }
        slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_WRITE_INDEX_MASK);
        for (;;) {
            uint32_t entry = cb->texture_write_index[slot];
            if (entry == 0) break;
            if (cb->texture_write_objects &&
                cb->texture_write_objects[entry - 1] == texture) return;
            slot = (slot + 1) & MGL_TEX_WRITE_INDEX_MASK;
        }
    }

    if (!mglEnsureTextureWriteCapacity(cb, cb->texture_write_count + 1u)) {
        cb->texture_write_overflow = true;
        return;
    }

    uint32_t idx = cb->texture_write_count++;
    cb->texture_write_objects[idx] = texture;
    cb->texture_write_index[slot] = idx + 1;
    mglNoteOccupancyPeak(&s_peak_texture_writes, cb->texture_write_count);
}

static void mglTrackPendingTextureRead(GLMContext ctx, Texture *texture)
{
    if (!ctx || !texture) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;

    uint32_t slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_READ_INDEX_MASK);
    if (cb->texture_read_objects) {
        for (;;) {
            uint32_t entry = cb->texture_read_index[slot];
            if (entry == 0) break;
            if (cb->texture_read_objects[entry - 1] == texture) return;
            slot = (slot + 1) & MGL_TEX_READ_INDEX_MASK;
        }
    } else {
        while (cb->texture_read_index[slot] != 0)
            slot = (slot + 1) & MGL_TEX_READ_INDEX_MASK;
    }

    if (cb->texture_read_count >= MGL_MAX_PENDING_TEXTURE_READS) {
        if (!mglHazardAllowInsertAfterCapacityFull(ctx, cb) ||
            cb->texture_read_count >= MGL_MAX_PENDING_TEXTURE_READS) {
            cb->texture_read_overflow = true;
            return;
        }
        slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_READ_INDEX_MASK);
        for (;;) {
            uint32_t entry = cb->texture_read_index[slot];
            if (entry == 0) break;
            if (cb->texture_read_objects &&
                cb->texture_read_objects[entry - 1] == texture) return;
            slot = (slot + 1) & MGL_TEX_READ_INDEX_MASK;
        }
    }

    if (!mglEnsureTextureReadCapacity(cb, cb->texture_read_count + 1u)) {
        cb->texture_read_overflow = true;
        return;
    }

    uint32_t idx = cb->texture_read_count++;
    cb->texture_read_objects[idx] = texture;
    cb->texture_read_index[slot] = idx + 1;
    mglNoteOccupancyPeak(&s_peak_texture_reads, cb->texture_read_count);
}

/* Forward declaration: defined below, used by the program-aware hazard
 * guards in both mglTrackPendingSampledTextureReads and
 * mglFlushPendingDrawsForActiveTextures. */
static bool mglStateSamplesTextureUnit(GLMContext ctx, GLuint unit);

void mglTrackPendingSampledTextureReads(GLMContext ctx)
{
    if (!ctx) return;

    unsigned *mask = ctx->active_state->active_texture_mask;
    for (int w = 0; w < 4; w++) {
        unsigned bits = mask[w];
        while (bits) {
            int bit = __builtin_ctz(bits);
            bits &= bits - 1;
            int unit = (w * 32) + bit;
            if (unit >= TEXTURE_UNITS) {
                continue;
            }

            /* Program-aware guard: only track textures as "read" on units
             * the current program actually samples.  Without this, a texture
             * left bound on an unsampled unit (e.g. FBO color attachment
             * after glTexImage2D) would be falsely tracked as read, causing
             * mglFlushPendingDrawsBeforeFramebufferTextureWrites to flush
             * on the next draw. */
            if (!mglStateSamplesTextureUnit(ctx, (GLuint)unit)) {
                continue;
            }

            Texture *active = ctx->active_state->active_textures[unit];
            if (active) {
                mglTrackPendingTextureRead(ctx, active);
            }

            TextureUnit *textureUnit = &ctx->active_state->texture_units[unit];
            for (int target = 0; target < _MAX_TEXTURE_TYPES; target++) {
                Texture *bound = textureUnit->textures[target];
                if (bound) {
                    mglTrackPendingTextureRead(ctx, bound);
                }
            }
        }
    }
}

void mglTrackPendingFramebufferTextureWrites(GLMContext ctx)
{
    if (!ctx) return;

    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!fbo) return;

    GLsizei drawBufferCount = ctx->active_state->draw_buffer_count;
    if (drawBufferCount <= 0 || drawBufferCount > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        drawBufferCount = 1;
    }

    for (GLsizei slot = 0; slot < drawBufferCount; slot++) {
        GLenum drawBuffer = ctx->active_state->draw_buffers[slot];
        if (drawBuffer == GL_NONE) {
            continue;
        }

        GLuint attachmentIndex = 0u;
        if (!mglDrawBufferToColorAttachmentIndex(ctx, drawBuffer, &attachmentIndex)) {
            continue;
        }
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        mglTrackPendingTextureWrite(ctx,
                                    mglPendingDrawAttachmentTexture(&fbo->color_attachments[attachmentIndex]));
    }

    mglTrackPendingFramebufferDepthStencilWrites(ctx, fbo);
}

/*
 * mglFlushPendingDrawsBeforeFramebufferTextureWrites — flush on post-framebuffer-attachment
 * write/read hazards
 *
 * Trigger: flush when a pending draw read the texture backing any color/depth/stencil
 * attachment of the currently bound FBO.
 * Guarantee: prevents WAR (write-after-read) hazards — an earlier draw sampled an
 * attachment texture that a new draw will now write; ensure the sampling reads the
 * earlier draw's output, not the new draw's overwrite.
 * Overflow degradation: when texture_read_overflow is set, degrade to an unconditional
 * full flush.
 * Extra behavior: emits a trace log on hit (first 64 hits, then every 512th).
 */
void mglFlushPendingDrawsBeforeFramebufferTextureWrites(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return;
    if (cb->texture_read_count == 0 && !cb->texture_read_overflow) return;
    if (mgl_batch_hazard_query_degraded(cb->texture_read_overflow)) {
        mglFlushCommandBuffer(ctx);
        return;
    }

    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!fbo) return;

    GLsizei drawBufferCount = ctx->active_state->draw_buffer_count;
    if (drawBufferCount <= 0 || drawBufferCount > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        drawBufferCount = 1;
    }

    for (GLsizei slot = 0; slot < drawBufferCount; slot++) {
        GLenum drawBuffer = ctx->active_state->draw_buffers[slot];
        if (drawBuffer == GL_NONE) {
            continue;
        }

        GLuint attachmentIndex = 0u;
        if (!mglDrawBufferToColorAttachmentIndex(ctx, drawBuffer, &attachmentIndex)) {
            continue;
        }
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        Texture *texture =
            mglPendingDrawAttachmentTexture(&fbo->color_attachments[attachmentIndex]);
        if (texture && mglPendingDrawsReadTexture(ctx, texture)) {
            static uint64_t s_framebufferWriteHazardFlushCount = 0;
            uint64_t hit = ++s_framebufferWriteHazardFlushCount;
            if (mglTraceLogIsEnabled() &&
                (hit <= 64ull || (hit % 512ull) == 0ull)) {
                mglTraceLogExternal("MGL TRACE pending framebuffer write hazard flush hit=%llu tex=%u attachment=%u batches=%u commands=%u",
                                    (unsigned long long)hit,
                                    texture ? texture->name : 0u,
                                    (unsigned)attachmentIndex,
                                    ctx->draw_command_buffer.batch_count,
                                    ctx->draw_command_buffer.total_commands);
            }
            mglFlushCommandBuffer(ctx);
            return;
        }
    }

    const struct {
        const char *name;
        FBOAttachment *attachment;
    } depthStencilAttachments[] = {
        { "depth", &fbo->depth },
        { "stencil", &fbo->stencil }
    };
    for (size_t i = 0; i < sizeof(depthStencilAttachments) / sizeof(depthStencilAttachments[0]); i++) {
        Texture *texture = mglPendingDrawAttachmentTexture(depthStencilAttachments[i].attachment);
        if (texture && mglPendingDrawsReadTexture(ctx, texture)) {
            static uint64_t s_framebufferDepthStencilWriteHazardFlushCount = 0;
            uint64_t hit = ++s_framebufferDepthStencilWriteHazardFlushCount;
            if (mglTraceLogIsEnabled() &&
                (hit <= 64ull || (hit % 512ull) == 0ull)) {
                mglTraceLogExternal("MGL TRACE pending framebuffer %s write hazard flush hit=%llu tex=%u batches=%u commands=%u",
                                    depthStencilAttachments[i].name,
                                    (unsigned long long)hit,
                                    texture ? texture->name : 0u,
                                    ctx->draw_command_buffer.batch_count,
                                    ctx->draw_command_buffer.total_commands);
            }
            mglFlushCommandBuffer(ctx);
            return;
        }
    }
}

/*
 * mglPendingDrawsReadBufferRange — buffer-range read hazard query
 *
 * Trigger: returns true when a pending draw reads a buffer range overlapping
 * [offset, offset+size).
 * Guarantee: provides the per-range hazard detection a CPU-side write performs
 * before overwriting a buffer, so an encoded draw's reads are not broken.
 * Overflow degradation: when buffer_read_range_overflow is set, return true for
 * any buffer/range, which degrades the subsequent flush to an unconditional
 * full flush.
 */
bool mglPendingDrawsReadBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size)
{
    if (!ctx || !buffer || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->buffer_read_range_count == 0 && !cb->buffer_read_range_overflow) return false;
    if (mgl_batch_hazard_query_degraded(cb->buffer_read_range_overflow)) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        return true;
    }

    uint64_t start = 0;
    uint64_t end = 0;
    mglNormalizeMutationRange(offset, size, &start, &end);

    uint32_t bucket = mglBufferRangeBucket(buffer);
    if (!cb->buffer_read_ranges || !cb->buffer_read_range_next) return false;
    for (uint32_t link = cb->buffer_read_range_bucket[bucket]; link;
         link = cb->buffer_read_range_next[link - 1u]) {
        MGLBufferReadRange *range = &cb->buffer_read_ranges[link - 1u];
        if (range->buffer == buffer && mglRangesOverlap(range->start, range->end, start, end)) {
            return true;
        }
    }

    return false;
}

/*
 * mglPendingDrawsWriteTexture — texture write hazard query
 *
 * Trigger: returns true when a pending draw wrote the given texture.
 * Guarantee: detects RAW (read-after-write) hazards, ensuring later sampling
 * of the texture does not read a stale write.
 * Overflow degradation: when texture_write_overflow is set, return true for
 * any texture.
 */
bool mglPendingDrawsWriteTexture(GLMContext ctx, void *texture)
{
    if (!ctx || !texture || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->texture_write_count == 0 && !cb->texture_write_overflow) return false;
    if (mgl_batch_hazard_query_degraded(cb->texture_write_overflow)) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        return true;
    }

    /* O(1) membership via hash-set index instead of O(n) scan.
     * Query path is side-effect free: if the table is degenerate (every
     * slot occupied but no match), we conservatively return true without
     * setting texture_write_overflow.  The insert path already sets the
     * sticky overflow flag when capacity is exhausted, so the next insert
     * will latch overflow and trigger the conservative flush on subsequent
     * queries.  Avoiding writes here keeps queries pure and predictable. */
    uint32_t slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_WRITE_INDEX_MASK);
    uint32_t start_slot = slot;
    if (!cb->texture_write_objects) return false;
    do {
        uint32_t entry = cb->texture_write_index[slot];
        if (entry == 0) return false;                /* empty — not present */
        if (cb->texture_write_objects[entry - 1] == texture) return true;
        slot = (slot + 1) & MGL_TEX_WRITE_INDEX_MASK; /* probe */
    } while (slot != start_slot);
    /* Table fully probed with no empty slot and no match: degenerate.
     * Conservatively report a hit; the next insert will set overflow. */
    return true;
}

/*
 * mglPendingDrawsReadTexture — texture read hazard query
 *
 * Trigger: returns true when a pending draw read (sampled) the given texture.
 * Guarantee: detects WAR (write-after-read) hazards, ensuring a later write to
 * the texture does not break an encoded draw's sampling.
 * Overflow degradation: when texture_read_overflow is set, return true for any
 * texture.
 */
bool mglPendingDrawsReadTexture(GLMContext ctx, void *texture)
{
    if (!ctx || !texture || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->texture_read_count == 0 && !cb->texture_read_overflow) return false;
    if (mgl_batch_hazard_query_degraded(cb->texture_read_overflow)) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        return true;
    }

    /* O(1) membership via hash-set index instead of O(n) scan.
     * Query path is side-effect free (see mglPendingDrawsWriteTexture). */
    uint32_t slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_READ_INDEX_MASK);
    uint32_t start_slot = slot;
    if (!cb->texture_read_objects) return false;
    do {
        uint32_t entry = cb->texture_read_index[slot];
        if (entry == 0) return false;                /* empty — not present */
        if (cb->texture_read_objects[entry - 1] == texture) return true;
        slot = (slot + 1) & MGL_TEX_READ_INDEX_MASK; /* probe */
    } while (slot != start_slot);
    /* Degenerate table: conservatively report a hit; next insert latches overflow. */
    return true;
}

/*
 * mglFlushPendingDrawsForBuffer — whole-buffer hazard flush
 *
 * Trigger: flush when a pending draw read the whole given buffer (equivalent to
 * a [0, INT64_MAX) range query hit).
 * Guarantee: ensures a later CPU-side write/redefinition of the buffer does not
 * break an encoded draw's reads.
 * Overflow degradation: when buffer_read_range_overflow is set, degrade to an
 * unconditional full flush.
 */
void mglFlushPendingDrawsForBuffer(GLMContext ctx, void *buffer)
{
    if (mglPendingDrawsReadBufferRange(ctx, buffer, 0, -1)) {
        MGL_PERF_INC(g_mglFlushReasonBufferRangeSinceSwap);
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglFlushPendingDrawsForBufferRange — range-based hazard-detection flush
 *
 * Trigger: flush when a pending draw reads a buffer range overlapping
 * [offset, offset+size).
 * Guarantee: ensures a later CPU write to that buffer range does not break an
 * encoded draw's reads.
 * Overflow degradation: when buffer_read_range_overflow is set, degrade to an
 * unconditional full flush.
 */
void mglFlushPendingDrawsForBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size)
{
    if (mglPendingDrawsReadBufferRange(ctx, buffer, offset, size)) {
        MGL_PERF_INC(g_mglFlushReasonBufferRangeSinceSwap);
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglPendingDrawsReferenceVertexArray — VAO reference hazard query
 *
 * Trigger: returns true when a pending draw references the given VAO (matched
 * by source_vao / state_snapshot.vao / vao_name).
 * Guarantee: provides the hazard detection performed before VAO state changes,
 * ensuring later VAO mutations do not affect encoded draws.
 * Overflow degradation: stream-merged batches own a private VAO snapshot and
 * copied transient vertex/index data, so later VAO mutations cannot affect
 * their encoded draws; such batches are never registered.  When the reverse
 * index overflows, degrade to an unconditional hit (same as texture hazards).
 */
static bool mglPendingDrawsReferenceVertexArray(GLMContext ctx, VertexArray *vao)
{
    if (!ctx || !vao || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;

    if (cb->pending_vao_ref_count == 0 && cb->pending_vao_name_count == 0 &&
        !cb->pending_vao_ref_overflow && !cb->pending_vao_name_overflow) {
        return false;
    }

    if (mgl_batch_hazard_query_degraded(cb->pending_vao_ref_overflow) ||
        mgl_batch_hazard_query_degraded(cb->pending_vao_name_overflow)) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        return true;
    }

    /* O(1) pointer membership (source_vao and snapshot->vao). */
    {
        uint32_t slot = (uint32_t)(((uintptr_t)vao >> 4) & MGL_VAO_REF_INDEX_MASK);
        uint32_t start_slot = slot;
        do {
            uint32_t entry = cb->pending_vao_ref_index[slot];
            if (entry == 0) break;
            if (cb->pending_vao_refs[entry - 1] == vao) return true;
            slot = (slot + 1) & MGL_VAO_REF_INDEX_MASK;
        } while (slot != start_slot);
        if (slot == start_slot && cb->pending_vao_ref_index[slot] != 0) {
            /* Degenerate full table with no match — conservative hit. */
            return true;
        }
    }

    /* O(1) name membership (no-snapshot key.vao_name fallback). */
    if (vao->name != 0) {
        uint32_t slot = (vao->name * 2654435761u) & MGL_VAO_REF_INDEX_MASK;
        uint32_t start_slot = slot;
        do {
            uint32_t entry = cb->pending_vao_name_index[slot];
            if (entry == 0) return false;
            if (cb->pending_vao_names[entry - 1] == vao->name) return true;
            slot = (slot + 1) & MGL_VAO_REF_INDEX_MASK;
        } while (slot != start_slot);
        return true;
    }

    return false;
}

/*
 * mglFlushPendingDrawsForVertexArray — VAO hazard flush
 *
 * Trigger: flush when a pending draw references the given VAO.
 * Guarantee: ensures later mutations of the VAO's state/vertex attributes do
 * not change the behavior of encoded draws.
 * Overflow degradation: stream-merged batches own a private VAO snapshot and
 * are unaffected; all other matching batches trigger a flush.
 */
void mglFlushPendingDrawsForVertexArray(GLMContext ctx, void *vao)
{
    if (mglPendingDrawsReferenceVertexArray(ctx, (VertexArray *)vao)) {
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglFlushPendingDrawsForTexture — texture read/write hazard flush
 *
 * Trigger: flush when a pending draw wrote or read the given texture.
 * Guarantee: ensures any later operation on the texture (a write breaking
 * sampling, or a read seeing stale writes) cannot break synchronization.
 * Overflow degradation: degrade to an unconditional full flush when either
 * texture_write_overflow or texture_read_overflow is set.
 */
void mglFlushPendingDrawsForTexture(GLMContext ctx, void *texture)
{
    if (mglPendingDrawsWriteTexture(ctx, texture) ||
        mglPendingDrawsReadTexture(ctx, texture)) {
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglFlushPendingDrawsBeforeTextureWrite — read/write hazard flush before a
 * CPU-side texture write
 *
 * Trigger: flush when a pending draw read or wrote the given texture; used
 * before CPU-side texture upload/update.
 * Guarantee: ensures the CPU-side write cannot break an encoded draw's sampling
 * (WAR) nor race with an encoded draw's write (WAW).
 * Overflow degradation: degrade to an unconditional full flush when either
 * texture_read_overflow or texture_write_overflow is set.
 * Extra behavior: emits a trace log on hit (first 64 hits, then every 512th).
 */
void mglFlushPendingDrawsBeforeTextureWrite(GLMContext ctx, void *texture)
{
    if (!ctx || !texture || !ctx->draw_defer_enabled) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return;
    if (cb->texture_read_count == 0 && cb->texture_write_count == 0 &&
        !cb->texture_read_overflow && !cb->texture_write_overflow) {
        return;
    }

    if (mglPendingDrawsReadTexture(ctx, texture) ||
        mglPendingDrawsWriteTexture(ctx, texture)) {
        static uint64_t s_textureWriteHazardFlushCount = 0;
        uint64_t hit = ++s_textureWriteHazardFlushCount;
        if (mglTraceLogIsEnabled() &&
            (hit <= 64ull || (hit % 512ull) == 0ull)) {
            Texture *tex = (Texture *)texture;
            mglTraceLogExternal("MGL TRACE pending texture write hazard flush hit=%llu tex=%u batches=%u commands=%u",
                                (unsigned long long)hit,
                                tex ? tex->name : 0u,
                                ctx ? ctx->draw_command_buffer.batch_count : 0u,
                                ctx ? ctx->draw_command_buffer.total_commands : 0u);
        }
        MGL_PERF_INC(g_mglFlushReasonTexWriteSinceSwap);
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglBuildActiveSampledTextureUnitMask — merge the current program's or each
 * pipeline stage program's sampled_texture_unit_mask into the cached bitmap on
 * GLMState.
 *
 * Semantics match the legacy mglStateSamplesTextureUnit implementation:
 *  - monolithic program bound: take the program's mask
 *  - pipeline bound (program_name == 0): OR the stage programs' masks
 *  - pipeline with no stage program: conservative all-ones (preserves the
 *    "return true" behavior)
 *  - no program and no pipeline: conservative all-ones
 *
 * Cached in GLMState::active_sampled_texture_unit_mask, invalidated by
 * DIRTY_PROGRAM.  Once built, every per-unit query degrades to a single bit
 * test.
 */
static void mglBuildActiveSampledTextureUnitMask(GLMContext ctx)
{
    if (!ctx || !ctx->active_state) return;
    GLMState *state = ctx->active_state;

    /* Default: conservative (all units sampled).  Overwritten below when
     * a usable program or pipeline with at least one stage program is found. */
    state->active_sampled_texture_unit_mask[0] = 0xFFFFFFFFu;
    state->active_sampled_texture_unit_mask[1] = 0xFFFFFFFFu;
    state->active_sampled_texture_unit_mask[2] = 0xFFFFFFFFu;
    state->active_sampled_texture_unit_mask[3] = 0xFFFFFFFFu;

    Program *program = state->program;
    if (program) {
        /* mglProgramSamplesTextureUnit(program, 0) triggers lazy build of
         * the program-level mask if not yet valid. */
        (void)mglProgramSamplesTextureUnit(program, 0);
        for (int i = 0; i < 4; i++) {
            state->active_sampled_texture_unit_mask[i] =
                program->sampled_texture_unit_mask[i];
        }
        state->active_sampled_texture_unit_mask_valid = 1u;
        return;
    }

    /* Pipeline path only applies when GL_CURRENT_PROGRAM is 0. */
    if (state->program_name != 0u) {
        /* No monolithic program but name is set: treat as no program bound
         * (will fall through to conservative all-ones above). */
        state->active_sampled_texture_unit_mask_valid = 1u;
        return;
    }

    ProgramPipeline *pipeline = state->program_pipeline;
    if (!pipeline) {
        /* No program and no pipeline: conservative. */
        state->active_sampled_texture_unit_mask_valid = 1u;
        return;
    }

    /* OR every stage program's mask.  Track whether any stage is present so
     * an empty pipeline remains conservative (matches legacy semantics). */
    bool hasAnyStage = false;
    uint32_t merged[4] = {0, 0, 0, 0};
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Program *stageProg = pipeline->stage_programs[stage];
        if (!stageProg) continue;
        hasAnyStage = true;
        (void)mglProgramSamplesTextureUnit(stageProg, 0);
        for (int i = 0; i < 4; i++) {
            merged[i] |= stageProg->sampled_texture_unit_mask[i];
        }
    }

    if (hasAnyStage) {
        for (int i = 0; i < 4; i++) {
            state->active_sampled_texture_unit_mask[i] = merged[i];
        }
    }
    /* else: empty pipeline -> keep all-ones conservative. */
    state->active_sampled_texture_unit_mask_valid = 1u;
}

const uint32_t *mglGetActiveSampledTextureUnitMask(GLMContext ctx)
{
    if (!ctx || !ctx->active_state) return NULL;
    GLMState *state = ctx->active_state;
    if (!state->active_sampled_texture_unit_mask_valid) {
        mglBuildActiveSampledTextureUnitMask(ctx);
    }
    return state->active_sampled_texture_unit_mask;
}

/*
 * mglStateSamplesTextureUnit — whether the currently active program/pipeline
 * actually samples texture unit `unit`
 *
 * O(1) bit test against the merged mask cached on GLMState; the mask is
 * invalidated by DIRTY_PROGRAM and built lazily on first query.  With no
 * program and no pipeline bound, degrades to a conservative true (mask
 * all-ones, preserving the legacy flush behavior).
 */
static bool mglStateSamplesTextureUnit(GLMContext ctx, GLuint unit)
{
    if (!ctx) return true;
    if (unit >= TEXTURE_UNITS) return false;

    const uint32_t *mask = mglGetActiveSampledTextureUnitMask(ctx);
    if (!mask) return true;
    return (mask[unit >> 5] & (1u << (unit & 31u))) != 0u;
}

static Program *mglSamplerSnapshotProgramForStage(GLMContext ctx, int stage)
{
    if (!ctx || stage < 0 || stage >= _MAX_SHADER_TYPES) return NULL;
    if (ctx->active_state->program) return ctx->active_state->program;
    if (!ctx->active_state->program_pipeline) return NULL;
    return ctx->active_state->program_pipeline->stage_programs[stage];
}

static int mglSamplerSnapshotTargetIndex(const MGLShaderResource *resource)
{
    if (!resource) return -1;

    switch ((MGLImageDimension)resource->image_dim) {
        case MGL_IMAGE_DIM_1D:
            return resource->image_arrayed ? _TEXTURE_1D_ARRAY : _TEXTURE_1D;
        case MGL_IMAGE_DIM_2D:
            if (resource->image_multisampled) {
                return resource->image_arrayed
                    ? _TEXTURE_2D_MULTISAMPLE_ARRAY
                    : _TEXTURE_2D_MULTISAMPLE;
            }
            return resource->image_arrayed ? _TEXTURE_2D_ARRAY : _TEXTURE_2D;
        case MGL_IMAGE_DIM_3D:
            return _TEXTURE_3D;
        case MGL_IMAGE_DIM_CUBE:
            return resource->image_arrayed
                ? _TEXTURE_CUBE_MAP_ARRAY
                : _TEXTURE_CUBE_MAP;
        case MGL_IMAGE_DIM_RECT:
            return _TEXTURE_RECTANGLE;
        case MGL_IMAGE_DIM_BUFFER:
            return _TEXTURE_BUFFER_TARGET;
        default:
            return -1;
    }
}

static void mglBuildSamplerSnapshotKey(const TextureParameter *params,
                                       GLenum target,
                                       MGLSamplerSnapshotKey *key)
{
    memset(key, 0, sizeof(*key));
    key->target = target;
    key->min_filter = params->min_filter;
    key->mag_filter = params->mag_filter;
    key->wrap_s = params->wrap_s;
    key->wrap_t = params->wrap_t;
    key->wrap_r = params->wrap_r;
    key->compare_mode = params->compare_mode;
    key->compare_func = params->compare_func;
    key->max_anisotropy = params->max_anisotropy;
    key->min_lod = params->min_lod;
    key->max_lod = params->max_lod;
    memcpy(key->border_color, params->border_color, sizeof(key->border_color));
}

static uint64_t mglSamplerSnapshotHashBytes(const void *data, size_t size)
{
    return mglHashBytesFNV1a(data, size);
}

static bool mglInternSamplerSnapshotKey(MGLCommandBuffer *cb,
                                        const MGLSamplerSnapshotKey *key,
                                        uint16_t *index_out)
{
    if (!cb || !key || !index_out) return false;

    const uint32_t mask = MGL_SAMPLER_SNAPSHOT_KEY_INDEX_SIZE - 1u;
    uint32_t slot = (uint32_t)mglSamplerSnapshotHashBytes(key, sizeof(*key)) & mask;
    for (uint32_t probe = 0; probe < MGL_SAMPLER_SNAPSHOT_KEY_INDEX_SIZE;
         probe++, slot = (slot + 1u) & mask) {
        uint16_t encoded = cb->sampler_snapshot_key_index[slot];
        if (encoded == 0u) break;
        uint16_t index = encoded - 1u;
        if (index < cb->sampler_snapshot_key_count && cb->sampler_snapshot_keys &&
            memcmp(&cb->sampler_snapshot_keys[index], key, sizeof(*key)) == 0) {
            *index_out = index;
            return true;
        }
    }
    if (cb->sampler_snapshot_key_count >= MGL_MAX_SAMPLER_SNAPSHOT_KEYS) {
        return false;
    }
    if (!mglEnsureSamplerKeyCapacity(cb, (uint32_t)cb->sampler_snapshot_key_count + 1u)) {
        return false;
    }

    uint16_t index = cb->sampler_snapshot_key_count++;
    cb->sampler_snapshot_keys[index] = *key;
    slot = (uint32_t)mglSamplerSnapshotHashBytes(key, sizeof(*key)) & mask;
    while (cb->sampler_snapshot_key_index[slot] != 0u) {
        slot = (slot + 1u) & mask;
    }
    cb->sampler_snapshot_key_index[slot] = index + 1u;
    *index_out = index;
    mglNoteOccupancyPeak(&s_peak_sampler_snapshot_keys, cb->sampler_snapshot_key_count);
    return true;
}

static bool mglAppendSamplerSnapshotEntry(MGLCommandBuffer *cb,
                                          MGLSamplerSnapshotSet *set,
                                          int stage,
                                          GLuint metal_slot,
                                          GLuint texture_unit,
                                          int target_index,
                                          const TextureParameter *params,
                                          GLenum target)
{
    if (!cb || !set ||
        (stage != _VERTEX_SHADER && stage != _FRAGMENT_SHADER) ||
        metal_slot >= 16u || texture_unit >= TEXTURE_UNITS ||
        target_index < 0 || target_index >= _MAX_TEXTURE_TYPES) {
        return false;
    }

    uint16_t key_index = MGL_FALLBACK_SAMPLER_KEY_INDEX;
    if (params) {
        MGLSamplerSnapshotKey key;
        mglBuildSamplerSnapshotKey(params, target, &key);
        if (!mglInternSamplerSnapshotKey(cb, &key, &key_index)) return false;
    }

    MGLSamplerSnapshotEntry entry = {
        .key_index = key_index,
        .stage = (uint8_t)stage,
        .metal_slot = (uint8_t)metal_slot,
        .texture_unit = (uint8_t)texture_unit,
        .target_index = (uint8_t)target_index,
    };
    for (uint8_t i = 0; i < set->count; i++) {
        if (memcmp(&set->entries[i], &entry, sizeof(entry)) == 0) return true;
    }
    if (set->count >= MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES) return false;
    set->entries[set->count++] = entry;
    return true;
}

static bool mglCaptureProgramSamplerSnapshots(GLMContext ctx,
                                              MGLCommandBuffer *cb,
                                              Program *program,
                                              int stage,
                                              MGLSamplerSnapshotSet *set)
{
    if (!program) return true;

    MGLShaderResourceList *separate =
        &program->shader_resources_list[stage][_SEPARATE_SAMPLERS_RES];
    for (GLuint i = 0; separate->list && i < separate->count; i++) {
        if (separate->list[i].resource_active) return false;
    }

    MGLShaderResourceList *sampled =
        &program->shader_resources_list[stage][_SAMPLED_IMAGE_RES];
    for (GLuint i = 0; sampled->list && i < sampled->count; i++) {
        MGLShaderResource *resource = &sampled->list[i];
        if (!resource->resource_active || !resource->has_combined_sampler) continue;
        if (resource->is_array || resource->gl_array_size > 1 ||
            resource->combined_sampler_binding == (GLuint)-1) {
            return false;
        }

        GLint unit = mglResolveSamplerResourceUnit(program,
                                                   resource,
                                                   stage,
                                                   _SAMPLED_IMAGE_RES);
        int target_index = mglSamplerSnapshotTargetIndex(resource);
        if (unit < 0 || unit >= TEXTURE_UNITS || target_index < 0) return false;

        Texture *texture =
            ctx->active_state->texture_units[unit].textures[target_index];
        if (!texture) {
            if (!mglAppendSamplerSnapshotEntry(cb,
                                               set,
                                               stage,
                                               resource->combined_sampler_binding,
                                               (GLuint)unit,
                                               target_index,
                                               NULL,
                                               0u)) {
                return false;
            }
            continue;
        }
        if (texture->index != (GLuint)target_index) return false;

        Sampler *sampler = ctx->active_state->texture_samplers[unit];
        const TextureParameter *params = sampler ? &sampler->params : &texture->params;
        if (!mglAppendSamplerSnapshotEntry(cb,
                                           set,
                                           stage,
                                           resource->combined_sampler_binding,
                                           (GLuint)unit,
                                           target_index,
                                           params,
                                           texture->target)) {
            return false;
        }
    }
    return true;
}

static bool mglProgramHasUnsupportedStageSamplers(Program *program, int stage)
{
    if (!program) return false;
    const int types[] = {
        _SAMPLED_IMAGE_RES,
        _SEPARATE_SAMPLERS_RES,
    };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        MGLShaderResourceList *list = &program->shader_resources_list[stage][types[t]];
        for (GLuint i = 0; list->list && i < list->count; i++) {
            if (list->list[i].resource_active) return true;
        }
    }
    return false;
}

bool mglCaptureSamplerSnapshot(GLMContext ctx, uint16_t *snapshot_id)
{
    if (!ctx || !snapshot_id) return false;
    *snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    MGLSamplerSnapshotSet candidate;
    memset(&candidate, 0, sizeof(candidate));

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Program *program = mglSamplerSnapshotProgramForStage(ctx, stage);
        if (stage == _VERTEX_SHADER || stage == _FRAGMENT_SHADER) {
            if (!mglCaptureProgramSamplerSnapshots(ctx, cb, program, stage, &candidate)) {
                return false;
            }
        } else if (stage != _COMPUTE_SHADER &&
                   mglProgramHasUnsupportedStageSamplers(program, stage)) {
            return false;
        }
    }

    if (candidate.count == 0) return true;
    const uint32_t mask = MGL_SAMPLER_SNAPSHOT_SET_INDEX_SIZE - 1u;
    uint32_t slot =
        (uint32_t)mglSamplerSnapshotHashBytes(&candidate, sizeof(candidate)) & mask;
    for (uint32_t probe = 0; probe < MGL_SAMPLER_SNAPSHOT_SET_INDEX_SIZE;
         probe++, slot = (slot + 1u) & mask) {
        uint16_t encoded = cb->sampler_snapshot_set_index[slot];
        if (encoded == 0u) break;
        uint16_t index = encoded - 1u;
        if (index < cb->sampler_snapshot_set_count && cb->sampler_snapshot_sets &&
            memcmp(&cb->sampler_snapshot_sets[index],
                   &candidate,
                   sizeof(candidate)) == 0) {
            *snapshot_id = index;
            return true;
        }
    }
    if (cb->sampler_snapshot_set_count >= MGL_MAX_SAMPLER_SNAPSHOT_SETS) {
        return false;
    }
    if (!mglEnsureSamplerSetCapacity(cb, (uint32_t)cb->sampler_snapshot_set_count + 1u)) {
        return false;
    }

    uint16_t index = cb->sampler_snapshot_set_count++;
    cb->sampler_snapshot_sets[index] = candidate;
    slot = (uint32_t)mglSamplerSnapshotHashBytes(&candidate, sizeof(candidate)) & mask;
    while (cb->sampler_snapshot_set_index[slot] != 0u) {
        slot = (slot + 1u) & mask;
    }
    cb->sampler_snapshot_set_index[slot] = index + 1u;
    *snapshot_id = index;
    mglNoteOccupancyPeak(&s_peak_sampler_snapshot_sets, cb->sampler_snapshot_set_count);
    return true;
}

/*
 * mglFlushPendingDrawsForActiveTextures — write-after-read hazard flush for
 * active texture units
 *
 * Trigger: flush when a pending draw wrote a texture bound to any currently
 * active texture unit; called before each draw.
 * Guarantee: prevents WAR hazards — an earlier draw wrote a texture that the
 * current draw reads as a sampler.
 * Program-aware: flushes only when the current program actually samples the
 * unit, avoiding false-positive hazard flushes caused by leftover FBO color
 * attachment texture bindings.
 * Overflow degradation: degrade to an unconditional full flush when
 * texture_write_overflow is set.
 */
void mglFlushPendingDrawsForActiveTextures(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return;
    if (cb->texture_write_count == 0 && !cb->texture_write_overflow) return;
    if (mgl_batch_hazard_query_degraded(cb->texture_write_overflow)) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        MGL_PERF_INC(g_mglFlushReasonActiveTexWarSinceSwap);
        mglFlushCommandBuffer(ctx);
        return;
    }

    unsigned *mask = ctx->active_state->active_texture_mask;
    for (int w = 0; w < 4; w++) {
        unsigned bits = mask[w];
        while (bits) {
            int bit = __builtin_ctz(bits);
            bits &= bits - 1;
            int unit = (w * 32) + bit;
            if (unit >= TEXTURE_UNITS) {
                continue;
            }

            /* Program-aware guard: skip units the current program/pipeline
             * never samples.  This eliminates false-positive WAR flushes
             * where a texture (e.g. FBO color attachment left bound after
             * glTexImage2D) sits on a unit no sampler reads. */
            if (!mglStateSamplesTextureUnit(ctx, (GLuint)unit)) {
                continue;
            }

            Texture *active = ctx->active_state->active_textures[unit];
            if (active && mglPendingDrawsWriteTexture(ctx, active)) {
                MGL_PERF_INC(g_mglFlushReasonActiveTexWarSinceSwap);
                mglFlushCommandBuffer(ctx);
                return;
            }

            TextureUnit *textureUnit = &ctx->active_state->texture_units[unit];
            for (int target = 0; target < _MAX_TEXTURE_TYPES; target++) {
                Texture *bound = textureUnit->textures[target];
                if (bound && mglPendingDrawsWriteTexture(ctx, bound)) {
                    MGL_PERF_INC(g_mglFlushReasonActiveTexWarSinceSwap);
                    mglFlushCommandBuffer(ctx);
                    return;
                }
            }
        }
    }
}

/* Mix pointer with GL name, matching mglHashBufferPointerName. A recycled
 * malloc address alone must not keep two different textures/samplers equal. */
