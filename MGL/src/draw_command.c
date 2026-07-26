/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * draw_command.c
 * MGL
 *
 */

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdatomic.h>

#include "glm_context.h"
#include "draw_command.h"
#include "mgl_safety.h"
#include "mgl_frame_activity.h"
#include "mgl_trace_log.h"
#include "mgl_sampler_compat.h"
#include "spirv_cross_c.h"

/* === Task 4: Snapshot Arena (bump allocator) === */

#define MGL_ARENA_INITIAL_CAPACITY  (4u * 1024u * 1024u)  /* 4 MB */

struct MGLBatchArenaChunk {
    struct MGLBatchArenaChunk *next;
    size_t                     offset;
    size_t                     capacity;
    unsigned char              data[];
};

static MGLBatchArenaChunk *mglNewBatchArenaChunk(size_t capacity)
{
    if (capacity == 0 || capacity > SIZE_MAX - sizeof(MGLBatchArenaChunk)) {
        return NULL;
    }

    MGLBatchArenaChunk *chunk = malloc(sizeof(*chunk) + capacity);
    if (!chunk) {
        return NULL;
    }

    chunk->next = NULL;
    chunk->offset = 0;
    chunk->capacity = capacity;
    return chunk;
}

bool mglInitBatchArena(MGLBatchArena *arena, size_t initial_capacity)
{
    if (!arena) {
        return false;
    }

    memset(arena, 0, sizeof(*arena));
    size_t capacity = initial_capacity ? initial_capacity : MGL_ARENA_INITIAL_CAPACITY;
    MGLBatchArenaChunk *chunk = mglNewBatchArenaChunk(capacity);
    if (!chunk) {
        return false;
    }

    arena->head = chunk;
    arena->current = chunk;
    arena->initial_capacity = capacity;
    arena->enabled = 1;
    return true;
}

static void *arenaAlloc(MGLBatchArena *arena, size_t size)
{
    if (!arena || !arena->enabled || size == 0) {
        return NULL;
    }

    /* Align to 16 bytes */
    if (size > SIZE_MAX - 15u) {
        return NULL;
    }
    size_t aligned = (size + 15u) & ~(size_t)15u;

    MGLBatchArenaChunk *chunk = arena->current;
    if (!chunk || aligned > chunk->capacity - chunk->offset) {
        MGLBatchArenaChunk *next = chunk ? chunk->next : arena->head;
        while (next && aligned > next->capacity - next->offset) {
            next = next->next;
        }
        if (next) {
            chunk = next;
        } else {
            size_t capacity = chunk ? chunk->capacity : arena->initial_capacity;
            if (capacity == 0) {
                capacity = MGL_ARENA_INITIAL_CAPACITY;
            }
            while (capacity < aligned) {
                if (capacity > SIZE_MAX / 2u) {
                    capacity = aligned;
                    break;
                }
                capacity *= 2u;
            }
            next = mglNewBatchArenaChunk(capacity);
            if (!next) {
                return NULL;
            }
            if (chunk) {
                MGLBatchArenaChunk *tail = chunk;
                while (tail->next) {
                    tail = tail->next;
                }
                tail->next = next;
            } else if (!arena->head) {
                arena->head = next;
            }
            chunk = next;
        }
        arena->current = chunk;
    }

    void *ptr = chunk->data + chunk->offset;
    chunk->offset += aligned;
    return ptr;
}

void mglResetBatchArena(MGLBatchArena *arena)
{
    if (!arena) return;
    for (MGLBatchArenaChunk *chunk = arena->head; chunk; chunk = chunk->next) {
        chunk->offset = 0;
    }
    arena->current = arena->head;
}

void mglDestroyBatchArena(MGLBatchArena *arena)
{
    if (!arena) return;

    MGLBatchArenaChunk *chunk = arena->head;
    while (chunk) {
        MGLBatchArenaChunk *next = chunk->next;
        free(chunk);
        chunk = next;
    }
    memset(arena, 0, sizeof(*arena));
}

void mglInitCommandBuffer(MGLCommandBuffer *cb)
{
    if (!cb) return;
    memset(cb, 0, sizeof(*cb));
}

static void mglDestroyTransientBuffer(GLMContext ctx, Buffer *buffer)
{
    if (!buffer) return;

    if (ctx && ctx->mtl_funcs.release_buffer_metal_data)
        ctx->mtl_funcs.release_buffer_metal_data(ctx, buffer);
    if (buffer->data.buffer_data) {
        free((void *)(uintptr_t)buffer->data.buffer_data);
        buffer->data.buffer_data = 0;
        buffer->data.buffer_size = 0;
    }

    free(buffer);
}

/* Buffer_base types that batch snapshots must retain/release.
 * Only the 5 "hot" types are in the snapshot (mglCopyHotStateFields Region 4);
 * the 11 cold types stay live in active_state through replay (never written).
 * Retaining cold types would dereference stale arena pointers → SIGSEGV.
 * MUST stay in sync with mglCopyHotStateFields Region 4. */
static const int kMGLSnapshotBufferBaseTypes[] = {
#define MGL_SNAPSHOT_LIST_HOT(_t_) _t_,
    MGL_SNAPSHOT_HOT_BUFFER_BASE_TYPES(MGL_SNAPSHOT_LIST_HOT)
#undef MGL_SNAPSHOT_LIST_HOT
};
static const size_t kMGLSnapshotBufferBaseTypeCount =
    sizeof(kMGLSnapshotBufferBaseTypes) / sizeof(kMGLSnapshotBufferBaseTypes[0]);

static void mglRetainBatchBufferReferences(MGLDrawBatch *batch)
{
    GLMState *snap = (GLMState *)batch->state_snapshot;
    if (!snap) return;
    for (size_t t = 0; t < kMGLSnapshotBufferBaseTypeCount; t++) {
        BufferBaseTarget *slots = snap->buffer_base[kMGLSnapshotBufferBaseTypes[t]].buffers;
        for (size_t i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            if (slots[i].buf) mglRetainBufferReference(slots[i].buf);
        }
    }
}

/* Must be called BEFORE state_snapshot is freed, since it reads buf pointers
 * from the snapshot. */
static void mglReleaseBatchBufferReferences(GLMContext ctx, MGLDrawBatch *batch)
{
    GLMState *snap = (GLMState *)batch->state_snapshot;
    if (!snap) return;
    for (size_t t = 0; t < kMGLSnapshotBufferBaseTypeCount; t++) {
        BufferBaseTarget *slots = snap->buffer_base[kMGLSnapshotBufferBaseTypes[t]].buffers;
        for (size_t i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            if (slots[i].buf) mglReleaseBufferReference(ctx, slots[i].buf);
        }
    }
}

static void mglReleaseBatch(GLMContext ctx, MGLDrawBatch *batch)
{
    if (!batch) return;

    /* release buffer references BEFORE freeing state_snapshot, since
     * we need to read buf pointers from the snapshot. */
    mglReleaseBatchBufferReferences(ctx, batch);

    /* Arena-managed allocations (commands, state_snapshot, vao_snapshot) are
     * freed collectively via arena reset (mglResetBatchArena), not
     * individually.  Only free them on the non-arena path. */
    if (!batch->arena_managed) {
        if (batch->commands) {
            free(batch->commands);
            batch->commands = NULL;
        }
        if (batch->state_snapshot) {
            free(batch->state_snapshot);
            batch->state_snapshot = NULL;
        }
        if (batch->vao_snapshot) {
            free(batch->vao_snapshot);
            batch->vao_snapshot = NULL;
        }
    } else {
        batch->commands = NULL;
        batch->state_snapshot = NULL;
        batch->vao_snapshot = NULL;
    }
    batch->source_vao = NULL;
    if (batch->retained_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_program);
        batch->retained_program = NULL;
    }
    if (batch->retained_vertex_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_vertex_program);
        batch->retained_vertex_program = NULL;
    }
    if (batch->retained_fragment_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_fragment_program);
        batch->retained_fragment_program = NULL;
    }
    if (batch->stream_vertex_buffer) {
        mglDestroyTransientBuffer(ctx, (Buffer *)batch->stream_vertex_buffer);
        batch->stream_vertex_buffer = NULL;
    }
    if (batch->stream_index_buffer) {
        mglDestroyTransientBuffer(ctx, (Buffer *)batch->stream_index_buffer);
        batch->stream_index_buffer = NULL;
    }
}

static Program *mglRetainBatchProgram(GLMContext ctx, MGLDrawBatch *batch, Program *program, GLuint expectedName, void **slot)
{
    if (!ctx || !batch || !program || !slot) {
        return NULL;
    }

    if (expectedName == 0u) {
        /* Name unknown: probe before dereferencing to read it. */
        if (!mglObjectPointerLooksPlausible(program) ||
            !mglPointerRangeIsReadable(program, sizeof(*program))) {
            return NULL;
        }
        expectedName = program->name;
    }
    if (!mglProgramPointerUsableForName(ctx, program, expectedName)) {
        return NULL;
    }

    /* UsableForName just proved the pointer live and matching; retain
     * directly rather than re-validating via mglRetainProgramReference. */
    program->refcount++;
    *slot = program;
    return program;
}

/* Retain the programs batch replay will dereference: the monolithic program
 * (a glUseProgram binding covers all stages), or — when a program pipeline is
 * bound instead — its vertex and fragment stage programs.  These three slots
 * are the COMPLETE set replay consumes; replay only resolves _VERTEX_SHADER
 * and _FRAGMENT_SHADER.  If replay is ever extended to dereference a pipeline's
 * geometry/tess/compute stage program, that stage MUST be retained here too, or
 * replay will touch a possibly-freed program (use-after-free). */
static void mglRetainBatchProgramReferences(GLMContext ctx, MGLDrawBatch *batch)
{
    if (!ctx || !batch) {
        return;
    }

    (void)mglRetainBatchProgram(ctx,
                                batch,
                                ctx->active_state->program,
                                ctx->active_state->program_name,
                                &batch->retained_program);

    if (ctx->active_state->program_name != 0u || !ctx->active_state->program_pipeline) {
        return;
    }

    ProgramPipeline *pipeline = ctx->active_state->program_pipeline;
    if (!mglObjectPointerLooksPlausible(pipeline) ||
        !mglPointerRangeIsReadable(pipeline, sizeof(*pipeline))) {
        return;
    }

    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_VERTEX_SHADER],
                                pipeline->stage_programs[_VERTEX_SHADER]
                                    ? pipeline->stage_programs[_VERTEX_SHADER]->name
                                    : 0u,
                                &batch->retained_vertex_program);
    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_FRAGMENT_SHADER],
                                pipeline->stage_programs[_FRAGMENT_SHADER]
                                    ? pipeline->stage_programs[_FRAGMENT_SHADER]->name
                                    : 0u,
                                &batch->retained_fragment_program);
}

static bool mglInitializeBatchStateSnapshot(GLMContext ctx, MGLDrawBatch *batch)
{
    MGL_SIGNPOST_BEGIN(InitBatchSnapshot);
    if (!ctx || !batch) {
        MGL_SIGNPOST_END(InitBatchSnapshot);
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
    if (!batch->state_snapshot || !batch->vao_snapshot) {
        MGL_SIGNPOST_END(InitBatchSnapshot);
        return false;
    }
    /* Selective snapshot: only copy hot fields (~51KB vs 82KB full).
     * Cold fields (HashTables + unused buffer_base types) are skipped;
     * they are restored from savedState at replay time. */
    mglCopyHotStateFields((GLMState *)batch->state_snapshot, ctx->active_state);
    batch->source_vao = ctx->active_state->vao;
    if (ctx->active_state->vao) {
        memcpy(batch->vao_snapshot, ctx->active_state->vao, sizeof(VertexArray));
        ((VertexArray *)batch->vao_snapshot)->transient_batch_vao = GL_TRUE;
        ((GLMState *)batch->state_snapshot)->vao = (VertexArray *)batch->vao_snapshot;
    } else {
        if (!batch->arena_managed) {
            free(batch->vao_snapshot);
        }
        batch->vao_snapshot = NULL;
    }

    MGL_PERF_INC(g_mglSnapshotAllocationCountSinceSwap);
    MGL_PERF_ADD(g_mglSnapshotBytesAllocatedSinceSwap,
                 sizeof(GLMState) + sizeof(VertexArray));

    mglRetainBatchProgramReferences(ctx, batch);
    mglRetainBatchBufferReferences(batch);
    MGL_SIGNPOST_END(InitBatchSnapshot);
    return true;
}

void mglResetCommandBufferForContext(GLMContext ctx, MGLCommandBuffer *cb)
{
    if (!cb) return;

    for (uint32_t i = 0; i < cb->batch_count; i++) {
        mglReleaseBatch(ctx, &cb->batches[i]);
    }

    memset(cb, 0, sizeof(*cb));
}

void mglResetCommandBuffer(MGLCommandBuffer *cb)
{
    mglResetCommandBufferForContext(NULL, cb);
}

static inline uint64_t mglRotateLeft64(uint64_t x, int n)
{
    n &= 63;
    if (n == 0) return x;
    return (x << n) | (x >> (64 - n));
}

/* SIMD-accelerated hash for 16+ byte data.
 * Uses ARM NEON to process 16 bytes per iteration (~2x speedup on large data).
 * Falls back to scalar for small data (<16 bytes). */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

static uint64_t mglHashBytes64_SIMD(const void *data, size_t size, uint64_t seed)
{
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = seed ^ 0xcbf29ce484222325ULL;

    if (!bytes) return hash;

    /* SIMD path: process 16-byte chunks with NEON */
    if (size >= 16) {
        const uint8_t *ptr = bytes;
        size_t chunks = size / 16;

        /* Load FNV prime as vector for parallel multiply */
        const uint64_t fnv_prime = 0x100000001b3ULL;
        uint64x2_t hash_vec = vdupq_n_u64(hash);
        uint64x2_t prime_vec = vdupq_n_u64(fnv_prime);

        for (size_t c = 0; c < chunks; c++) {
            /* Load 16 bytes as 2x uint64 */
            uint8x16_t chunk = vld1q_u8(ptr);
            uint64x2_t chunk64 = vreinterpretq_u64_u8(chunk);

            /* FNV-1a: hash ^= data, hash *= prime (done in parallel for 2x uint64) */
            hash_vec = veorq_u64(hash_vec, chunk64);

            /* Multiply by FNV prime (64-bit multiply on NEON)
             * Note: ARM64 has vmulq_u64 but we need to handle carries properly
             * for hash quality. Use scalar multiply but vectorize XOR. */
            uint64_t h0 = vgetq_lane_u64(hash_vec, 0) * fnv_prime;
            uint64_t h1 = vgetq_lane_u64(hash_vec, 1) * fnv_prime;
            hash_vec = vsetq_lane_u64(h0, hash_vec, 0);
            hash_vec = vsetq_lane_u64(h1, hash_vec, 1);

            ptr += 16;
        }

        /* Combine the two hash lanes with XOR */
        hash = vgetq_lane_u64(hash_vec, 0) ^ vgetq_lane_u64(hash_vec, 1);

        /* Process remaining bytes with scalar code */
        size_t processed = chunks * 16;
        bytes += processed;
        size -= processed;
    }

    /* Scalar path for remaining bytes (or all bytes if size < 16) */
    size_t i = 0;
    while (i + 8 <= size) {
        uint64_t word;
        memcpy(&word, bytes + i, 8);
        hash ^= word;
        hash *= 0x100000001b3ULL;
        i += 8;
    }
    if (i + 4 <= size) {
        uint32_t word;
        memcpy(&word, bytes + i, 4);
        hash ^= (uint64_t)word;
        hash *= 0x100000001b3ULL;
        i += 4;
    }
    while (i < size) {
        hash ^= (uint64_t)bytes[i];
        hash *= 0x100000001b3ULL;
        i++;
    }
    return hash;
}
#endif

static uint64_t mglHashBytes64(const void *data, size_t size, uint64_t seed)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    /* Use SIMD version on ARM with NEON support */
    return mglHashBytes64_SIMD(data, size, seed);
#else
    /* Fallback: original chunked FNV-1a implementation */
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = seed ^ 0xcbf29ce484222325ULL;

    if (!bytes) return hash;

    /* chunked FNV-1a — process 8-byte words, then a 4-byte word, then
     * the byte tail.  Same FNV prime/mix as the byte loop but ~8× fewer
     * multiply iterations on aligned-ish data.  memcpy avoids UB on
     * unaligned access and is optimized to a single load on arm64. */
    size_t i = 0;
    while (i + 8 <= size) {
        uint64_t word;
        memcpy(&word, bytes + i, 8);
        hash ^= word;
        hash *= 0x100000001b3ULL;
        i += 8;
    }
    if (i + 4 <= size) {
        uint32_t word;
        memcpy(&word, bytes + i, 4);
        hash ^= (uint64_t)word;
        hash *= 0x100000001b3ULL;
        i += 4;
    }
    while (i < size) {
        hash ^= (uint64_t)bytes[i];
        hash *= 0x100000001b3ULL;
        i++;
    }
    return hash;
#endif
}

static inline bool mglRangesOverlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
    return a0 < b1 && b0 < a1;
}

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

static uint32_t mglRuntimeMaxDrawsPerBatch(void)
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

static uint32_t mglRuntimeMaxBatchCount(void)
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

static void mglTrackPendingReadRange(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t end)
{
    if (!ctx || !buffer || end <= start) return;

    /* Walk only this buffer's bucket chain instead of every recorded range —
     * the previous full scan made a flush window with N ranges cost O(N)
     * per insert (O(N²) total). */
    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    uint32_t bucket = mglBufferRangeBucket(buffer);
    for (uint32_t link = cb->buffer_read_range_bucket[bucket]; link;
         link = cb->buffer_read_range_next[link - 1u]) {
        MGLBufferReadRange *range = &cb->buffer_read_ranges[link - 1u];
        if (range->buffer == buffer && mglRangesOverlap(range->start, range->end, start, end)) {
            if (start < range->start) range->start = start;
            if (end > range->end) range->end = end;
            return;
        }
    }

    if (cb->buffer_read_range_count >= MGL_MAX_PENDING_BUFFER_RANGES) {
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

    MGL_PERF_INC(g_mglHazardRangeCountSinceSwap);
}

static void mglTrackPendingReadBytes(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t size)
{
    uint64_t end = (size > UINT64_MAX - start) ? UINT64_MAX : start + size;
    mglTrackPendingReadRange(ctx, buffer, start, end);
}

static void mglTrackPendingReadWholeBuffer(GLMContext ctx, Buffer *buffer)
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
    for (;;) {
        uint32_t entry = cb->texture_write_index[slot];
        if (entry == 0) break;                       /* empty — not present */
        if (cb->texture_write_objects[entry - 1] == texture) return; /* dup */
        slot = (slot + 1) & MGL_TEX_WRITE_INDEX_MASK; /* probe */
    }

    if (cb->texture_write_count >= MGL_MAX_PENDING_TEXTURE_WRITES) {
        cb->texture_write_overflow = true;
        return;
    }

    uint32_t idx = cb->texture_write_count++;
    cb->texture_write_objects[idx] = texture;
    /* Insert into the empty slot found by the probe above. */
    cb->texture_write_index[slot] = idx + 1;
}

static void mglTrackPendingTextureRead(GLMContext ctx, Texture *texture)
{
    if (!ctx || !texture) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;

    /* O(1) dedup via hash-set index. */
    uint32_t slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_READ_INDEX_MASK);
    for (;;) {
        uint32_t entry = cb->texture_read_index[slot];
        if (entry == 0) break;
        if (cb->texture_read_objects[entry - 1] == texture) return;
        slot = (slot + 1) & MGL_TEX_READ_INDEX_MASK;
    }

    if (cb->texture_read_count >= MGL_MAX_PENDING_TEXTURE_READS) {
        cb->texture_read_overflow = true;
        return;
    }

    uint32_t idx = cb->texture_read_count++;
    cb->texture_read_objects[idx] = texture;
    cb->texture_read_index[slot] = idx + 1;
}

/* Forward declaration: defined below, used by the program-aware hazard
 * guards in both mglTrackPendingSampledTextureReads and
 * mglFlushPendingDrawsForActiveTextures. */
static bool mglStateSamplesTextureUnit(GLMContext ctx, GLuint unit);

static void mglTrackPendingSampledTextureReads(GLMContext ctx)
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

static void mglTrackPendingFramebufferTextureWrites(GLMContext ctx)
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
 * mglFlushPendingDrawsBeforeFramebufferTextureWrites — 帧缓冲附件写后读危害刷新
 *
 * 触发条件：待处理 draw 读取了当前绑定 FBO 的颜色/深度/模板附件所对应 texture 时 flush。
 * 保证语义：防止写后读（WAR）危害——前序 draw 采样了某附件 texture，而新 draw 又将写入该附件；
 *           确保采样读到的是前序 draw 完成后的内容，而非新 draw 覆盖后的内容。
 * 溢出退化：当 texture_read_overflow 置位时退化为无条件全刷新。
 * 附加行为：命中时输出 trace 日志（前 64 次及之后每 512 次一次）。
 */
static void mglFlushPendingDrawsBeforeFramebufferTextureWrites(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return;
    if (cb->texture_read_count == 0 && !cb->texture_read_overflow) return;
    if (cb->texture_read_overflow) {
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
 * mglPendingDrawsReadBufferRange — 缓冲区范围读危害查询
 *
 * 触发条件：当待处理 draw 读取的缓冲区范围与 [offset, offset+size) 存在重叠时返回 true。
 * 保证语义：为 CPU 端写入缓冲区前的按范围危害检测提供基础，避免破坏已编码 draw 的读取。
 * 溢出退化：当 buffer_read_range_overflow 置位时退化为对任意 buffer/range 均返回 true，
 *           使随后的 flush 调用退化为无条件全刷新。
 */
bool mglPendingDrawsReadBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size)
{
    if (!ctx || !buffer || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->buffer_read_range_count == 0 && !cb->buffer_read_range_overflow) return false;
    if (cb->buffer_read_range_overflow) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        return true;
    }

    uint64_t start = 0;
    uint64_t end = 0;
    mglNormalizeMutationRange(offset, size, &start, &end);

    uint32_t bucket = mglBufferRangeBucket(buffer);
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
 * mglPendingDrawsWriteTexture — 纹理写危害查询
 *
 * 触发条件：当待处理 draw 写入了给定 texture 时返回 true。
 * 保证语义：检测 RAW/写后读危害，确保后续对该纹理的采样不会读到陈旧的写入。
 * 溢出退化：当 texture_write_overflow 置位时退化为对任意 texture 均返回 true。
 */
bool mglPendingDrawsWriteTexture(GLMContext ctx, void *texture)
{
    if (!ctx || !texture || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->texture_write_count == 0 && !cb->texture_write_overflow) return false;
    if (cb->texture_write_overflow) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        return true;
    }

    /* O(1) membership via hash-set index instead of O(n) scan. */
    uint32_t slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_WRITE_INDEX_MASK);
    uint32_t start_slot = slot;
    uint32_t probes = 0;
    for (;;) {
        uint32_t entry = cb->texture_write_index[slot];
        if (entry == 0) return false;                /* empty — not present */
        if (cb->texture_write_objects[entry - 1] == texture) return true;
        slot = (slot + 1) & MGL_TEX_WRITE_INDEX_MASK; /* probe */
        if (++probes > (MGL_TEX_WRITE_INDEX_MASK + 1) || slot == start_slot) {
            /* Probed entire table without finding empty slot or match — treat as overflow */
            cb->texture_write_overflow = true;
            return true;
        }
    }
}

/*
 * mglPendingDrawsReadTexture — 纹理读危害查询
 *
 * 触发条件：当待处理 draw 读取（采样）了给定 texture 时返回 true。
 * 保证语义：检测 WAR/读后写危害，确保随后对该纹理的写入不会破坏已编码 draw 的采样。
 * 溢出退化：当 texture_read_overflow 置位时退化为对任意 texture 均返回 true。
 */
bool mglPendingDrawsReadTexture(GLMContext ctx, void *texture)
{
    if (!ctx || !texture || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->texture_read_count == 0 && !cb->texture_read_overflow) return false;
    if (cb->texture_read_overflow) {
        MGL_PERF_INC(g_mglHazardOverflowFlushesSinceSwap);
        return true;
    }

    /* O(1) membership via hash-set index instead of O(n) scan. */
    uint32_t slot = (uint32_t)(((uintptr_t)texture >> 4) & MGL_TEX_READ_INDEX_MASK);
    uint32_t start_slot = slot;
    uint32_t probes = 0;
    for (;;) {
        uint32_t entry = cb->texture_read_index[slot];
        if (entry == 0) return false;                /* empty — not present */
        if (cb->texture_read_objects[entry - 1] == texture) return true;
        slot = (slot + 1) & MGL_TEX_READ_INDEX_MASK; /* probe */
        if (++probes > (MGL_TEX_READ_INDEX_MASK + 1) || slot == start_slot) {
            /* Probed entire table without finding empty slot or match — treat as overflow */
            cb->texture_read_overflow = true;
            return true;
        }
    }
}

/*
 * mglFlushPendingDrawsForBuffer — 整缓冲区危害刷新
 *
 * 触发条件：待处理 draw 读取了指定的整个 buffer（等价于 [0, INT64_MAX) 范围查询命中）时 flush。
 * 保证语义：确保随后对该 buffer 的 CPU 端写入/重定义不会破坏已编码 draw 的读取。
 * 溢出退化：当 buffer_read_range_overflow 置位时退化为无条件全刷新。
 */
void mglFlushPendingDrawsForBuffer(GLMContext ctx, void *buffer)
{
    if (mglPendingDrawsReadBufferRange(ctx, buffer, 0, -1)) {
        MGL_PERF_INC(g_mglFlushReasonBufferRangeSinceSwap);
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglFlushPendingDrawsForBufferRange — 按范围危害检测刷新
 *
 * 触发条件：待处理 draw 读取的缓冲区范围与 [offset, offset+size) 重叠时 flush。
 * 保证语义：确保随后对该缓冲区范围的 CPU 写入不会破坏已编码 draw 的读取。
 * 溢出退化：当 buffer_read_range_overflow 置位时，退化为无条件全刷新。
 */
void mglFlushPendingDrawsForBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size)
{
    if (mglPendingDrawsReadBufferRange(ctx, buffer, offset, size)) {
        MGL_PERF_INC(g_mglFlushReasonBufferRangeSinceSwap);
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglPendingDrawsReferenceVertexArray — VAO 引用危害查询
 *
 * 触发条件：待处理 draw 引用了给定 VAO（按 source_vao / state_snapshot.vao / vao_name 匹配）时返回 true。
 * 保证语义：为 VAO 状态变更前的危害检测提供基础，确保后续 VAO 修改不会影响已编码 draw。
 * 溢出退化：stream-merged 的 batch 持有私有 VAO 快照与已拷贝的瞬时顶点/索引数据，
 *           后续 VAO 修改不会影响其已编码 draw，故直接跳过此类 batch。
 */
static bool mglPendingDrawsReferenceVertexArray(GLMContext ctx, VertexArray *vao)
{
    if (!ctx || !vao || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;

    for (uint32_t i = 0; i < cb->batch_count; i++) {
        MGLDrawBatch *batch = &cb->batches[i];
        if (batch->command_count == 0) continue;

        /*
         * Stream-merged batches own a private VAO snapshot and copied transient
         * vertex/index data, so later mutations of the application's VAO cannot
         * change the already recorded draw.
         */
        if (batch->stream_merged) continue;

        if (batch->state_snapshot) {
            GLMState *snapshot = (GLMState *)batch->state_snapshot;
            if (batch->source_vao == vao || snapshot->vao == vao) {
                return true;
            }
            continue;
        }

        if (batch->key.vao_name == vao->name) {
            return true;
        }
    }

    return false;
}

/*
 * mglFlushPendingDrawsForVertexArray — VAO 危害刷新
 *
 * 触发条件：待处理 draw 引用了指定 VAO 时 flush。
 * 保证语义：确保随后对该 VAO 的状态/顶点属性修改不会改变已编码 draw 的行为。
 * 溢出退化：stream-merged batch 持有私有 VAO 快照，不受影响；其余 batch 命中即 flush。
 */
void mglFlushPendingDrawsForVertexArray(GLMContext ctx, void *vao)
{
    if (mglPendingDrawsReferenceVertexArray(ctx, (VertexArray *)vao)) {
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglFlushPendingDrawsForTexture — 纹理读/写危害刷新
 *
 * 触发条件：待处理 draw 写入或读取了指定 texture 时 flush。
 * 保证语义：确保随后对该 texture 的任何操作（写破坏采样，或读看到陈旧写入）不会破坏同步。
 * 溢出退化：当 texture_write_overflow 或 texture_read_overflow 任一置位时退化为无条件全刷新。
 */
void mglFlushPendingDrawsForTexture(GLMContext ctx, void *texture)
{
    if (mglPendingDrawsWriteTexture(ctx, texture) ||
        mglPendingDrawsReadTexture(ctx, texture)) {
        mglFlushCommandBuffer(ctx);
    }
}

/*
 * mglFlushPendingDrawsBeforeTextureWrite — CPU 端纹理写入前的读/写危害刷新
 *
 * 触发条件：待处理 draw 读取或写入了指定 texture 时 flush；用于 CPU 端纹理上传/更新前。
 * 保证语义：确保随后对 texture 的 CPU 端写入不会破坏已编码 draw 的采样（WAR），
 *           也避免与已编码 draw 的写入发生冲突（WAW）。
 * 溢出退化：当 texture_read_overflow 或 texture_write_overflow 任一置位时退化为无条件全刷新。
 * 附加行为：命中时输出 trace 日志（前 64 次及之后每 512 次一次）。
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
 * mglStateSamplesTextureUnit — 当前活跃 program 是否实际采样纹理单元 unit
 *
 * 遍历当前 monolithic program 或 pipeline 各 stage program，调用
 * mglProgramSamplesTextureUnit 判断是否有 sampler 资源解析到该 unit。
 * 无 program 且无 pipeline 时退化为保守返回 true（保留旧刷新行为）。
 */
static bool mglStateSamplesTextureUnit(GLMContext ctx, GLuint unit)
{
    if (!ctx) return true;

    Program *program = ctx->active_state->program;
    if (program && mglProgramSamplesTextureUnit(program, unit)) {
        return true;
    }

    /* Pipeline with separate stage programs. */
    if (!program && ctx->active_state->program_pipeline) {
        ProgramPipeline *pipeline = ctx->active_state->program_pipeline;
        bool hasAnyStage = false;
        for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
            Program *stageProg = pipeline->stage_programs[stage];
            if (!stageProg) continue;
            hasAnyStage = true;
            if (mglProgramSamplesTextureUnit(stageProg, unit)) {
                return true;
            }
        }
        /* Pipeline has stage programs but none sample this unit -> safe.
         * Empty pipeline -> conservative. */
        return !hasAnyStage;
    }

    /* No program and no pipeline: conservative. */
    if (!program) return true;

    return false;
}

static Program *mglSamplerSnapshotProgramForStage(GLMContext ctx, int stage)
{
    if (!ctx || stage < 0 || stage >= _MAX_SHADER_TYPES) return NULL;
    if (ctx->active_state->program) return ctx->active_state->program;
    if (!ctx->active_state->program_pipeline) return NULL;
    return ctx->active_state->program_pipeline->stage_programs[stage];
}

static int mglSamplerSnapshotTargetIndex(const SpirvResource *resource)
{
    if (!resource) return -1;

    switch ((SpvDim)resource->image_dim) {
        case SpvDim1D:
            return resource->image_arrayed ? _TEXTURE_1D_ARRAY : _TEXTURE_1D;
        case SpvDim2D:
            if (resource->image_multisampled) {
                return resource->image_arrayed
                    ? _TEXTURE_2D_MULTISAMPLE_ARRAY
                    : _TEXTURE_2D_MULTISAMPLE;
            }
            return resource->image_arrayed ? _TEXTURE_2D_ARRAY : _TEXTURE_2D;
        case SpvDim3D:
            return _TEXTURE_3D;
        case SpvDimCube:
            return resource->image_arrayed
                ? _TEXTURE_CUBE_MAP_ARRAY
                : _TEXTURE_CUBE_MAP;
        case SpvDimRect:
            return _TEXTURE_RECTANGLE;
        case SpvDimBuffer:
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
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
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
        if (index < cb->sampler_snapshot_key_count &&
            memcmp(&cb->sampler_snapshot_keys[index], key, sizeof(*key)) == 0) {
            *index_out = index;
            return true;
        }
    }
    if (cb->sampler_snapshot_key_count >= MGL_MAX_SAMPLER_SNAPSHOT_KEYS) {
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

    SpirvResourceList *separate =
        &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS];
    for (GLuint i = 0; separate->list && i < separate->count; i++) {
        if (separate->list[i].msl_active) return false;
    }

    SpirvResourceList *sampled =
        &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; sampled->list && i < sampled->count; i++) {
        SpirvResource *resource = &sampled->list[i];
        if (!resource->msl_active || !resource->msl_has_combined_sampler) continue;
        if (resource->is_array || resource->gl_array_size > 1 ||
            resource->msl_combined_sampler_binding == (GLuint)-1) {
            return false;
        }

        GLint unit = mglResolveSamplerResourceUnit(program,
                                                   resource,
                                                   stage,
                                                   SPVC_RESOURCE_TYPE_SAMPLED_IMAGE);
        int target_index = mglSamplerSnapshotTargetIndex(resource);
        if (unit < 0 || unit >= TEXTURE_UNITS || target_index < 0) return false;

        Texture *texture =
            ctx->active_state->texture_units[unit].textures[target_index];
        if (!texture) {
            if (!mglAppendSamplerSnapshotEntry(cb,
                                               set,
                                               stage,
                                               resource->msl_combined_sampler_binding,
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
                                           resource->msl_combined_sampler_binding,
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
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
    };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        SpirvResourceList *list = &program->spirv_resources_list[stage][types[t]];
        for (GLuint i = 0; list->list && i < list->count; i++) {
            if (list->list[i].msl_active) return true;
        }
    }
    return false;
}

static bool mglCaptureSamplerSnapshot(GLMContext ctx, uint16_t *snapshot_id)
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
        if (index < cb->sampler_snapshot_set_count &&
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

    uint16_t index = cb->sampler_snapshot_set_count++;
    cb->sampler_snapshot_sets[index] = candidate;
    slot = (uint32_t)mglSamplerSnapshotHashBytes(&candidate, sizeof(candidate)) & mask;
    while (cb->sampler_snapshot_set_index[slot] != 0u) {
        slot = (slot + 1u) & mask;
    }
    cb->sampler_snapshot_set_index[slot] = index + 1u;
    *snapshot_id = index;
    return true;
}

/*
 * mglFlushPendingDrawsForActiveTextures — 活跃纹理单元写后读危害刷新
 *
 * 触发条件：待处理 draw 写入了当前任何活跃纹理单元所绑定的 texture 时 flush；每次 draw 前调用。
 * 保证语义：防止写后读（WAR）危害——前序 draw 写入了某 texture，而当前 draw 又将其作为采样器读取。
 * 程序感知：仅当当前 program 实际采样该 unit 时才 flush，避免 FBO 颜色附件纹理残留绑定
 *           导致的误报刷新（false-positive hazard）。
 * 溢出退化：当 texture_write_overflow 置位时退化为无条件全刷新。
 */
void mglFlushPendingDrawsForActiveTextures(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return;
    if (cb->texture_write_count == 0 && !cb->texture_write_overflow) return;
    if (cb->texture_write_overflow) {
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

static uint64_t mglComputeTextureHash(GLMContext ctx)
{
    uint64_t hash = 0;
    unsigned *mask = ctx->active_state->active_texture_mask;
    for (int w = 0; w < 4; w++) {
        unsigned bits = mask[w];
        while (bits) {
            int i = __builtin_ctz(bits);
            bits &= bits - 1;
            int unit = w * 32 + i;
            if (unit < TEXTURE_UNITS) {
                Texture *tex = ctx->active_state->active_textures[unit];
                uint64_t tex_ptr = tex ? (uint64_t)(uintptr_t)tex : 0;
                hash ^= mglRotateLeft64(tex_ptr, unit & 63);

                TextureUnit *tex_unit = &ctx->active_state->texture_units[unit];
                for (int t = 0; t < _MAX_TEXTURE_TYPES; t++) {
                    Texture *typed_tex = tex_unit->textures[t];
                    uint64_t typed_ptr = typed_tex ? (uint64_t)(uintptr_t)typed_tex : 0;
                    hash ^= mglRotateLeft64(typed_ptr + (uint64_t)(t + 1), (unit + t) & 63);
                }

                Sampler *sampler = ctx->active_state->texture_samplers[unit];
                uint64_t sampler_ptr = sampler ? (uint64_t)(uintptr_t)sampler : 0;
                hash ^= mglRotateLeft64(sampler_ptr, (unit + 17) & 63);
            }
        }
    }
    return hash;
}

static void mglHashBufferBaseBinding(uint64_t *hash,
                                     const BufferBaseTarget *binding,
                                     uint64_t salt)
{
    if (!hash || !binding) return;

    uint64_t binding_ptr = binding->buf ? (uint64_t)(uintptr_t)binding->buf : 0;
    *hash ^= mglRotateLeft64(binding_ptr ^ ((uint64_t)binding->buffer << 1), salt & 63);
    *hash ^= mglRotateLeft64((uint64_t)binding->offset, (salt + 17u) & 63);
    *hash ^= mglRotateLeft64((uint64_t)binding->size, (salt + 37u) & 63);
}

/* Hash one buffer_base target.  Empty slots are fully zeroed at
 * unbind/delete time, so their contribution is 0 and iterating only
 * active_mask bits is bit-identical to scanning all 84 slots (the
 * DEBUG sampled assert below verifies this). */
static void mglHashBufferBaseTarget(uint64_t *hash,
                                    const BufferBase *base,
                                    int target,
                                    bool use_mask)
{
    if (use_mask) {
        for (unsigned w = 0; w < MGL_BUFFER_BASE_ACTIVE_WORDS; w++) {
            uint64_t mask = base->active_mask[w];
            while (mask) {
                int i = (int)(w * 64u) + __builtin_ctzll(mask);
                mask &= mask - 1;
                mglHashBufferBaseBinding(hash,
                                         &base->buffers[i],
                                         ((uint64_t)(target + 1) * 131u) + (uint64_t)i);
            }
        }
    } else {
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            mglHashBufferBaseBinding(hash,
                                     &base->buffers[i],
                                     ((uint64_t)(target + 1) * 131u) + (uint64_t)i);
        }
    }
}

static uint64_t mglComputeDrawBufferBindingHashScan(GLMContext ctx, bool use_mask)
{
    uint64_t hash = 0;

    /* Keep this list aligned with the buffer_base slots captured by
     * mglCopyHotStateFields for deferred batch replay. */
    const int draw_buffer_targets[] = {
        _UNIFORM_BUFFER,
        _UNIFORM_CONSTANT,
        _SHADER_STORAGE_BUFFER,
        _ATOMIC_COUNTER_BUFFER,
        _TRANSFORM_FEEDBACK_BUFFER
    };

    for (size_t t = 0; t < sizeof(draw_buffer_targets) / sizeof(draw_buffer_targets[0]); t++) {
        int target = draw_buffer_targets[t];
        mglHashBufferBaseTarget(&hash,
                                &ctx->active_state->buffer_base[target],
                                target,
                                use_mask);
    }

    Program *program = ctx->active_state->program;
    if (program) {
        /* plain_uniform_buffers has no active_mask; keep the linear scan. */
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            mglHashBufferBaseBinding(&hash,
                                     &program->plain_uniform_buffers[i],
                                     0x700u + (uint64_t)i);
        }
    }

    return hash;
}

static uint64_t mglComputeDrawBufferBindingHash(GLMContext ctx)
{
    uint64_t hash = mglComputeDrawBufferBindingHashScan(ctx, true);
#ifdef DEBUG
    static uint32_t s_maskCheck = 0;
    if ((++s_maskCheck & 0xFFu) == 0u) {
        assert(hash == mglComputeDrawBufferBindingHashScan(ctx, false));
    }
#endif
    return hash;
}

static uint64_t mglComputeUniformBufferBindingHash(GLMContext ctx)
{
    uint64_t hash = 0;
    mglHashBufferBaseTarget(&hash,
                            &ctx->active_state->buffer_base[_UNIFORM_BUFFER],
                            _UNIFORM_BUFFER,
                            true);
    return hash;
}

static void mglHashBufferPointerName(uint64_t *hash, const Buffer *buffer, uint64_t salt)
{
    if (!hash) return;

    uint64_t ptr = buffer ? (uint64_t)(uintptr_t)buffer : 0u;
    uint64_t name = buffer ? (uint64_t)buffer->name : 0u;
    *hash ^= mglRotateLeft64(ptr ^ (name << 1), salt & 63);
}

static void mglHashCurrentVertexAttrib(uint64_t *hash,
                                       const CurrentVertexAttrib *current,
                                       uint64_t salt)
{
    if (!hash || !current) return;

    *hash ^= mglHashBytes64(current->f, sizeof(current->f), 0x63766166u + salt);
    *hash ^= mglHashBytes64(current->i, sizeof(current->i), 0x63766169u + salt);
    *hash ^= mglHashBytes64(current->u, sizeof(current->u), 0x63766175u + salt);
    *hash ^= mglHashBytes64(current->d, sizeof(current->d), 0x63766164u + salt);
    *hash ^= mglRotateLeft64((uint64_t)current->type, (salt + 1u) & 63);
    *hash ^= mglRotateLeft64((uint64_t)current->integer, (salt + 2u) & 63);
    *hash ^= mglRotateLeft64((uint64_t)current->long_attribute, (salt + 3u) & 63);
}

static void mglHashElementArrayState(uint64_t *hash, const VertexArray *vao)
{
    if (!hash || !vao) return;

    mglHashBufferPointerName(hash, vao->element_array.buffer, 0x900u);
    *hash ^= mglRotateLeft64((uint64_t)vao->element_array.type, 11);
    *hash ^= mglRotateLeft64((uint64_t)vao->element_array.size, 13);
    *hash ^= mglRotateLeft64((uint64_t)(uintptr_t)vao->element_array.ptr, 17);
}

static uint64_t mglComputeVertexArrayStateHash(GLMContext ctx, bool uses_elements)
{
    VertexArray *vao = ctx ? ctx->active_state->vao : NULL;
    if (!vao) return 0u;

    uint64_t hash = 0x56414f5354415445ULL;
    hash ^= mglRotateLeft64((uint64_t)vao->name, 3);
    hash ^= mglRotateLeft64((uint64_t)vao->enabled_attribs, 7);

    GLuint maxAttribs = MAX_ATTRIBS;
    for (GLuint i = 0; i < maxAttribs; i++) {
        const VertexAttrib *attrib = &vao->attrib[i];
        uint64_t salt = 0x100u + (uint64_t)i * 17u;
        mglHashBufferPointerName(&hash, attrib->buffer, salt);
        hash ^= mglRotateLeft64((uint64_t)attrib->size, (salt + 1u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->type, (salt + 2u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->normalized, (salt + 3u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->integer, (salt + 4u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->long_attribute, (salt + 5u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->stride, (salt + 6u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->divisor, (salt + 7u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->relativeoffset, (salt + 8u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->binding_offset, (salt + 9u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->buffer_bindingindex, (salt + 10u) & 63);
        if ((vao->enabled_attribs & (1u << i)) == 0u) {
            mglHashCurrentVertexAttrib(&hash,
                                       &ctx->active_state->current_vertex_attrib[i],
                                       0x300u + (uint64_t)i * 23u);
        }
    }

    for (GLuint i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++) {
        const BufferBinding *binding = &vao->bindings[i];
        uint64_t salt = 0x500u + (uint64_t)i * 19u;
        mglHashBufferPointerName(&hash, binding->buffer, salt);
        hash ^= mglRotateLeft64((uint64_t)binding->offset, (salt + 1u) & 63);
        hash ^= mglRotateLeft64((uint64_t)binding->stride, (salt + 2u) & 63);
        hash ^= mglRotateLeft64((uint64_t)binding->divisor, (salt + 3u) & 63);
    }

    if (uses_elements)
        mglHashElementArrayState(&hash, vao);

    return hash;
}

static uint64_t mglComputeRenderStateHash(GLMContext ctx)
{
    uint64_t hash = mglHashBytes64(&ctx->active_state->caps,
                                   sizeof(ctx->active_state->caps),
                                   0xfeedfacecafebeefULL);
    GLMParams *var = &ctx->active_state->var;

    hash ^= mglHashBytes64(ctx->active_state->viewport, sizeof(ctx->active_state->viewport), 0x101u);
    hash ^= mglRotateLeft64((uint64_t)ctx->active_state->draw_buffer, 7);
    hash ^= mglRotateLeft64((uint64_t)ctx->active_state->draw_buffer_count, 13);
    hash ^= mglHashBytes64(ctx->active_state->draw_buffers, sizeof(ctx->active_state->draw_buffers), 0x102u);

    hash ^= mglHashBytes64(&var->point_size, sizeof(var->point_size), 0x201u);
    hash ^= mglHashBytes64(&var->line_width, sizeof(var->line_width), 0x202u);
    hash ^= mglRotateLeft64((uint64_t)var->polygon_mode, 5);
    hash ^= mglRotateLeft64((uint64_t)var->cull_face_mode, 11);
    hash ^= mglRotateLeft64((uint64_t)var->front_face, 17);
    hash ^= mglHashBytes64(var->depth_range, sizeof(var->depth_range), 0x203u);
    hash ^= var->depth_writemask ? 0xAAAA5555ULL : 0ULL;
    hash ^= mglRotateLeft64((uint64_t)var->depth_func, 8);

    hash ^= mglRotateLeft64((uint64_t)var->logic_op, 9);
    hash ^= mglRotateLeft64((uint64_t)var->logic_op_mode, 10);
    hash ^= mglHashBytes64(var->color_writemask, sizeof(var->color_writemask), 0x204u);
    hash ^= mglHashBytes64(var->blend_color, sizeof(var->blend_color), 0x205u);
    hash ^= mglHashBytes64(var->blend_dst_rgb, sizeof(var->blend_dst_rgb), 0x206u);
    hash ^= mglHashBytes64(var->blend_src_rgb, sizeof(var->blend_src_rgb), 0x207u);
    hash ^= mglHashBytes64(var->blend_dst_alpha, sizeof(var->blend_dst_alpha), 0x208u);
    hash ^= mglHashBytes64(var->blend_src_alpha, sizeof(var->blend_src_alpha), 0x209u);
    hash ^= mglHashBytes64(var->blend_equation_rgb, sizeof(var->blend_equation_rgb), 0x20au);
    hash ^= mglHashBytes64(var->blend_equation_alpha, sizeof(var->blend_equation_alpha), 0x20bu);

    hash ^= mglRotateLeft64((uint64_t)var->stencil_func, 24);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_value_mask, 25);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_fail, 26);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_pass_depth_fail, 27);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_pass_depth_pass, 28);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_ref, 29);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_writemask, 30);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_func, 31);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_fail, 32);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_pass_depth_fail, 33);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_pass_depth_pass, 34);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_ref, 35);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_value_mask, 36);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_writemask, 37);

    hash ^= mglHashBytes64(&var->polygon_offset_units, sizeof(var->polygon_offset_units), 0x20cu);
    hash ^= mglHashBytes64(&var->polygon_offset_factor, sizeof(var->polygon_offset_factor), 0x20du);
    hash ^= mglHashBytes64(&var->sample_coverage_value, sizeof(var->sample_coverage_value), 0x20eu);
    hash ^= mglRotateLeft64((uint64_t)var->sample_coverage_invert, 38);
    hash ^= mglRotateLeft64((uint64_t)var->primitive_restart_index, 39);
    hash ^= mglRotateLeft64((uint64_t)var->clip_origin, 40);
    hash ^= mglRotateLeft64((uint64_t)var->clip_depth_mode, 41);
    hash ^= mglRotateLeft64((uint64_t)var->provoking_vertex, 42);

    return hash;
}

static uint8_t mglModeToPrimitiveType(GLenum mode)
{
    switch (mode) {
        case GL_POINTS:         return 0;  /* MTLPrimitiveTypePoint */
        case GL_LINES:          return 1;  /* MTLPrimitiveTypeLine */
        case GL_LINE_STRIP:     return 2;  /* MTLPrimitiveTypeLineStrip */
        case GL_TRIANGLES:      return 3;  /* MTLPrimitiveTypeTriangle */
        case GL_TRIANGLE_STRIP: return 4;  /* MTLPrimitiveTypeTriangleStrip */
        /* Metal has no native support for these modes; they are emulated
         * in the draw dispatch path (MGLRenderer.m). Return 0xFF so the
         * batch system routes them through the per-draw emulation path
         * instead of multi-draw-indirect. */
        case GL_TRIANGLE_FAN:
        case GL_LINE_LOOP:
        case GL_PATCHES:
        case GL_QUADS:
        case GL_LINES_ADJACENCY:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_TRIANGLES_ADJACENCY:
        case GL_TRIANGLE_STRIP_ADJACENCY:
            return 0xFF;
        default:
            return 0xFF;
    }
}

static uint16_t mglComputeCapsFlags(GLMContext ctx)
{
    uint16_t flags = 0;
    GLMCaps *caps = &ctx->active_state->caps;

    if (caps->cull_face)        flags |= (1u << 0);
    if (caps->depth_test)       flags |= (1u << 1);
    if (caps->stencil_test)     flags |= (1u << 2);
    if (caps->blend)            flags |= (1u << 3);
    if (caps->scissor_test)     flags |= (1u << 4);
    if (caps->polygon_offset_fill)  flags |= (1u << 5);
    if (caps->polygon_offset_line)  flags |= (1u << 6);
    if (caps->polygon_offset_point) flags |= (1u << 7);
    if (ctx->active_state->var.cull_face_mode == GL_FRONT_AND_BACK) flags |= (1u << 8);

    return flags;
}

void mglComputeStateKey(GLMContext ctx, GLenum mode, bool uses_elements, MGLStateKey *out)
{
    MGL_SIGNPOST_BEGIN(ComputeStateKey);
    if (!ctx || !out) {
        MGL_SIGNPOST_END(ComputeStateKey);
        return;
    }

    /* Every other field is written unconditionally below.  These are only
     * written conditionally (program names when a pipeline is bound without a
     * direct program; scissor[] only when scissor is enabled), so zero them
     * explicitly to keep memcmp-based equality correct without a full-struct
     * memset. */
    out->vertex_program_name = 0;
    out->fragment_program_name = 0;
    out->scissor[0] = 0;
    out->scissor[1] = 0;
    out->scissor[2] = 0;
    out->scissor[3] = 0;

    out->program_name = ctx->active_state->program_name;
    out->program_pipeline_name = ctx->active_state->var.program_pipeline_binding;
    if (out->program_pipeline_name == 0 && ctx->active_state->program_pipeline) {
        out->program_pipeline_name = ctx->active_state->program_pipeline->name;
    }
    if (ctx->active_state->program_name == 0 && out->program_pipeline_name != 0) {
        ProgramPipeline *pipeline = ctx->active_state->program_pipeline;
        if (!pipeline || pipeline->name != out->program_pipeline_name) {
            pipeline = (ProgramPipeline *)searchHashTable(&ctx->active_state->program_pipeline_table,
                                                          out->program_pipeline_name);
        }
        out->vertex_program_name = pipeline && pipeline->stage_programs[_VERTEX_SHADER]
            ? pipeline->stage_programs[_VERTEX_SHADER]->name
            : 0u;
        out->fragment_program_name = pipeline && pipeline->stage_programs[_FRAGMENT_SHADER]
            ? pipeline->stage_programs[_FRAGMENT_SHADER]->name
            : 0u;
    }
    out->vao_name = ctx->active_state->vao ? ctx->active_state->vao->name : 0;
    out->fbo_name = ctx->active_state->framebuffer ? ctx->active_state->framebuffer->name : 0;

    for (int i = 0; i < 4; i++) {
        out->viewport[i] = (int16_t)ctx->active_state->viewport[i];
    }

    out->scissor_enabled = ctx->active_state->caps.scissor_test ? 1 : 0;
    if (out->scissor_enabled) {
        for (int i = 0; i < 4; i++) {
            out->scissor[i] = (int16_t)ctx->active_state->var.scissor_box[i];
        }
    }

    out->primitive_type = mglModeToPrimitiveType(mode);
    out->caps_flags = mglComputeCapsFlags(ctx);

    /* Phase 1 #1: Dirty-Flag Hash Optimization
     * Only recompute hashes when dirty flags are set (texture bindings changed,
     * VAO changed, render state changed). This reduces redundant hash computation
     * from ~1200 → ~360 per frame when state is stable across draw calls.
     *
     * NOTE: cached hashes live in GLMState (ctx->active_state), NOT in the
     * MGLStateKey output struct.  mglStateKeysEqual() uses memcmp on the full
     * MGLStateKey, so the key must contain only per-draw-computed values —
     * any cached/padding fields there would be uninitialized garbage and
     * break batch-merge comparisons.
     *
     * NOTE: `mode` is a per-draw parameter, not persistent GL state, so it is
     * NOT part of the cached render-state hash.  It is XORed in on every call
     * so that different draw modes produce different keys. */
    static int dirty_hash_enabled = -1;
    if (dirty_hash_enabled < 0) {
        const char *env = getenv("MGL_OPT_DIRTY_HASH");
        dirty_hash_enabled = (env == NULL || atoi(env) != 0) ? 1 : 0;
    }

    if (dirty_hash_enabled) {
        /* Legacy dirty bits remain set until Metal consumes them. Hash-cache
         * invalidation is tracked separately so stable draws do not recompute
         * the same hashes while renderer state is still pending. */
        if (ctx->active_state->texture_dirty) {
            ctx->active_state->cached_texture_hash = mglComputeTextureHash(ctx);
            ctx->active_state->texture_dirty = 0;
        }
        out->texture_hash = ctx->active_state->cached_texture_hash;

        if (ctx->active_state->vertex_layout_dirty) {
            /* Cache only the draw-arrays-independent portion.  Element-array
             * state is added below for indexed draws so changing draw kind
             * cannot reuse a hash computed with the opposite uses_elements
             * value. */
            ctx->active_state->cached_vertex_layout_hash =
                mglComputeVertexArrayStateHash(ctx, false);
            ctx->active_state->vertex_layout_dirty = 0;
        }
        out->vertex_layout_hash = ctx->active_state->cached_vertex_layout_hash;
        if (uses_elements)
            mglHashElementArrayState(&out->vertex_layout_hash, ctx->active_state->vao);

        if (ctx->active_state->render_state_dirty) {
            ctx->active_state->cached_render_state_hash =
                mglComputeRenderStateHash(ctx) ^
                mglComputeDrawBufferBindingHash(ctx);
            ctx->active_state->render_state_dirty = 0;
        }
        out->render_state_hash = ctx->active_state->cached_render_state_hash ^
                                 mglRotateLeft64((uint64_t)mode, 21);

        /* uniform_buffer_hash is cached like the other hash domains instead
         * of being recomputed every draw.  Its inputs (buffer_base[_UNIFORM_
         * BUFFER]) are a strict subset of render_state_hash inputs, so it
         * shares the same dirty bits (DIRTY_BUFFER_BASE_STATE |
         * DIRTY_PROGRAM).  Kept as a separate field because
         * mglStateKeysEqualIgnoringUniformRanges XORs it back out of
         * render_state_hash during batch merge comparisons. */
        if (ctx->active_state->uniform_buffer_dirty) {
            ctx->active_state->cached_uniform_buffer_hash =
                mglComputeUniformBufferBindingHash(ctx);
            ctx->active_state->uniform_buffer_dirty = 0;
        }
        out->uniform_buffer_hash = ctx->active_state->cached_uniform_buffer_hash;
    } else {
        /* Legacy path: unconditional hash computation */
        out->texture_hash = mglComputeTextureHash(ctx);
        out->vertex_layout_hash = mglComputeVertexArrayStateHash(ctx, uses_elements);
        out->render_state_hash = mglComputeRenderStateHash(ctx) ^
                                 mglComputeDrawBufferBindingHash(ctx) ^
                                 mglRotateLeft64((uint64_t)mode, 21);
        out->uniform_buffer_hash = mglComputeUniformBufferBindingHash(ctx);
    }

    MGL_SIGNPOST_END(ComputeStateKey);
}

bool mglStateKeysEqual(const MGLStateKey *a, const MGLStateKey *b)
{
    if (!a || !b) return false;
    return memcmp(a, b, sizeof(MGLStateKey)) == 0;
}

static bool mglStateKeysEqualIgnoringUniformRanges(const MGLStateKey *a,
                                                   const MGLStateKey *b)
{
    if (!a || !b) return false;

    /* render_state_hash includes the UBO hash for legacy key compatibility.
     * XOR it back out before comparing the non-UBO render/buffer state. */
    if ((a->render_state_hash ^ a->uniform_buffer_hash) !=
        (b->render_state_hash ^ b->uniform_buffer_hash)) {
        return false;
    }

    /* Compare fields directly instead of copying both keys to the stack,
     * zeroing the two range-hash fields, and running a full-struct memcmp.
     * The range hashes are intentionally excluded from this comparison. */
    return a->program_name == b->program_name &&
           a->program_pipeline_name == b->program_pipeline_name &&
           a->vertex_program_name == b->vertex_program_name &&
           a->fragment_program_name == b->fragment_program_name &&
           a->vao_name == b->vao_name &&
           a->fbo_name == b->fbo_name &&
           a->viewport[0] == b->viewport[0] &&
           a->viewport[1] == b->viewport[1] &&
           a->viewport[2] == b->viewport[2] &&
           a->viewport[3] == b->viewport[3] &&
           a->scissor[0] == b->scissor[0] &&
           a->scissor[1] == b->scissor[1] &&
           a->scissor[2] == b->scissor[2] &&
           a->scissor[3] == b->scissor[3] &&
           a->scissor_enabled == b->scissor_enabled &&
           a->primitive_type == b->primitive_type &&
           a->caps_flags == b->caps_flags &&
           a->texture_hash == b->texture_hash &&
           a->vertex_layout_hash == b->vertex_layout_hash;
}

static bool mglCaptureDynamicUniformRanges(GLMContext ctx,
                                           const MGLDrawBatch *batch,
                                           MGLDrawCommand *cmd,
                                           bool capture_all_bound_ranges)
{
    if (!ctx || !batch || !batch->state_snapshot || !cmd) return false;

    const GLMState *snapshot = (const GLMState *)batch->state_snapshot;
    const BufferBase *base_ubo = &snapshot->buffer_base[_UNIFORM_BUFFER];
    const BufferBase *current_ubo = &ctx->active_state->buffer_base[_UNIFORM_BUFFER];

    cmd->dynamic_uniform_binding_count = 0;

    /* The active_mask bitmap mirrors buf != NULL (maintained by
     * mglBufferBaseSetActive / ClearActive at bind/unbind/delete time), so:
     *   - Bits set only in `current` (added) or only in `base` (removed)
     *     mean an object was bound/unbound since snapshot — fall back to
     *     the conservative hazard/flush path by returning false.
     *   - Bits set in both are the only slots we need to inspect for
     *     offset/size overrides.
     *   - Bits set in neither are skipped: both buf are NULL, and the
     *     bind/unbind contract keeps buffer names 0 when buf is NULL. */
    uint64_t base_mask[2]  = { base_ubo->active_mask[0],  base_ubo->active_mask[1] };
    uint64_t curr_mask[2]  = { current_ubo->active_mask[0], current_ubo->active_mask[1] };

    /* Object additions or removals since snapshot => cannot capture
     * dynamic overrides, fall back to hazard path. */
    if ((base_mask[0] ^ curr_mask[0]) != 0u ||
        (base_mask[1] ^ curr_mask[1]) != 0u) {
        return false;
    }

    /* Early bound-count check (replaces the per-slot ++bound_count). */
    uint16_t bound_count = (uint16_t)(__builtin_popcountll(curr_mask[0]) +
                                      __builtin_popcountll(curr_mask[1]));
    if (bound_count > MGL_MAX_DYNAMIC_UNIFORM_BINDINGS) {
        return false;
    }

    /* Iterate only the active slots. */
    for (uint32_t w = 0; w < MGL_BUFFER_BASE_ACTIVE_WORDS; w++) {
        uint64_t bits = curr_mask[w];
        while (bits) {
            int i = __builtin_ctzll(bits);
            bits &= bits - 1;
            uint16_t slot = (uint16_t)((w << 6) + i);

            const BufferBaseTarget *base = &base_ubo->buffers[slot];
            const BufferBaseTarget *current = &current_ubo->buffers[slot];

            /* Both slots are active per the mask check above, but verify
             * the underlying pointers/names match (cheap pointer compare). */
            if (base->buffer != current->buffer || base->buf != current->buf) {
                return false;
            }
            if (!capture_all_bound_ranges &&
                base->offset == current->offset && base->size == current->size) {
                continue;
            }
            if (cmd->dynamic_uniform_binding_count >= MGL_MAX_DYNAMIC_UNIFORM_BINDINGS) {
                return false;
            }

            MGLDynamicUniformBinding *override =
                &cmd->dynamic_uniform_bindings[cmd->dynamic_uniform_binding_count++];
            override->binding_index = slot;
            override->offset = current->offset;
            override->size = current->size;
        }
    }
    return cmd->dynamic_uniform_binding_count > 0;
}

static const Buffer *mglVertexArrayEffectiveBuffer(const VertexArray *vao,
                                                   GLuint attrib_index)
{
    if (!vao || attrib_index >= MAX_ATTRIBS) return NULL;
    const VertexAttrib *attrib = &vao->attrib[attrib_index];
    if (attrib->buffer_bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        const BufferBinding *binding =
            &vao->bindings[attrib->buffer_bindingindex];
        if (binding->buffer) return binding->buffer;
    }
    return attrib->buffer;
}

static bool mglVertexAttribLayoutsEqual(const VertexAttrib *a,
                                        const VertexAttrib *b)
{
    return a && b &&
           a->size == b->size &&
           a->type == b->type &&
           a->normalized == b->normalized &&
           a->integer == b->integer &&
           a->long_attribute == b->long_attribute &&
           a->stride == b->stride &&
           a->divisor == b->divisor &&
           a->relativeoffset == b->relativeoffset &&
           a->binding_offset == b->binding_offset &&
           a->buffer_bindingindex == b->buffer_bindingindex;
}

static bool mglVertexArraysHaveCompatibleBindings(GLMContext ctx,
                                                  const GLMState *snapshot,
                                                  const VertexArray *base,
                                                  const VertexArray *current)
{
    if (!ctx || !snapshot || !base || !current ||
        base->enabled_attribs == 0u ||
        base->enabled_attribs != current->enabled_attribs) {
        return false;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if (!mglVertexAttribLayoutsEqual(&base->attrib[i], &current->attrib[i])) {
            return false;
        }
        if ((base->enabled_attribs & (1u << i)) == 0u &&
            memcmp(&snapshot->current_vertex_attrib[i],
                   &ctx->active_state->current_vertex_attrib[i],
                   sizeof(CurrentVertexAttrib)) != 0) {
            return false;
        }
    }

    for (GLuint i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++) {
        const BufferBinding *a = &base->bindings[i];
        const BufferBinding *b = &current->bindings[i];
        if (a->stride != b->stride || a->divisor != b->divisor) {
            return false;
        }
    }

    /* Metal slot assignment groups attributes that share one VBO.  Preserve
     * that equivalence relation even though the actual buffer objects differ
     * between base and current.
     *
     * Cache the effective buffer per attribute once, then give each attribute
     * a canonical partition label: the smallest enabled index sharing the same
     * buffer.  Two partitions are equal iff the canonical labels match
     * element-wise. */
    const Buffer *base_bufs[MAX_ATTRIBS];
    const Buffer *current_bufs[MAX_ATTRIBS];
    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        base_bufs[i] = NULL;
        current_bufs[i] = NULL;
        if ((base->enabled_attribs & (1u << i)) == 0u) continue;
        base_bufs[i] = mglVertexArrayEffectiveBuffer(base, i);
        current_bufs[i] = mglVertexArrayEffectiveBuffer(current, i);
        if (!base_bufs[i] || !current_bufs[i]) return false;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if ((base->enabled_attribs & (1u << i)) == 0u) continue;
        /* Find smallest enabled j < i with the same buffer on each side. */
        GLuint base_lbl = i;
        GLuint current_lbl = i;
        for (GLuint j = 0; j < i; j++) {
            if ((base->enabled_attribs & (1u << j)) == 0u) continue;
            if (base_lbl == i && base_bufs[j] == base_bufs[i]) {
                base_lbl = j;
            }
            if (current_lbl == i && current_bufs[j] == current_bufs[i]) {
                current_lbl = j;
            }
            if (base_lbl != i && current_lbl != i) break;
        }
        if (base_lbl != current_lbl) return false;
    }
    return true;
}

static bool mglCaptureDynamicVertexBindings(GLMContext ctx,
                                            const MGLDrawBatch *batch,
                                            MGLDrawCommand *cmd)
{
    if (!ctx || !batch || !batch->vao_snapshot || !cmd ||
        !ctx->active_state->vao) {
        return false;
    }

    const VertexArray *base = (const VertexArray *)batch->vao_snapshot;
    const VertexArray *current = ctx->active_state->vao;
    const GLMState *snapshot = (const GLMState *)batch->state_snapshot;
    if (!mglVertexArraysHaveCompatibleBindings(ctx, snapshot, base, current)) {
        return false;
    }

    cmd->dynamic_vertex_binding_count = 0;
    uint16_t captured_mask = 0u;
    for (GLuint attrib_index = 0; attrib_index < MAX_ATTRIBS; attrib_index++) {
        if ((base->enabled_attribs & (1u << attrib_index)) == 0u) continue;

        const VertexAttrib *base_attrib = &base->attrib[attrib_index];
        const VertexAttrib *current_attrib = &current->attrib[attrib_index];
        GLuint binding_index = base_attrib->buffer_bindingindex;
        if (binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS ||
            binding_index != current_attrib->buffer_bindingindex) {
            return false;
        }
        uint16_t binding_bit = (uint16_t)(1u << binding_index);
        if ((captured_mask & binding_bit) != 0u) continue;

        const BufferBinding *base_binding = &base->bindings[binding_index];
        const BufferBinding *current_binding = &current->bindings[binding_index];
        const Buffer *base_buffer = base_binding->buffer
            ? base_binding->buffer : base_attrib->buffer;
        Buffer *current_buffer = current_binding->buffer
            ? current_binding->buffer : current_attrib->buffer;
        GLintptr base_offset = base_binding->buffer
            ? base_binding->offset : base_attrib->binding_offset;
        GLintptr current_offset = current_binding->buffer
            ? current_binding->offset : current_attrib->binding_offset;
        if (!base_buffer || !current_buffer || current_offset < base_offset) {
            return false;
        }
        uint64_t delta = (uint64_t)(current_offset - base_offset);
        if (delta > UINT32_MAX ||
            cmd->dynamic_vertex_binding_count >= MGL_MAX_DYNAMIC_VERTEX_BINDINGS) {
            return false;
        }

        MGLDynamicVertexBinding *override =
            &cmd->dynamic_vertex_bindings[cmd->dynamic_vertex_binding_count++];
        override->buffer = current_buffer;
        override->offset_delta = (uint32_t)delta;
        override->binding_index = (uint8_t)binding_index;
        memset(override->reserved, 0, sizeof(override->reserved));
        captured_mask |= binding_bit;
    }

    /* Attributes sharing one Metal slot must move by the same delta. */
    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if ((base->enabled_attribs & (1u << i)) == 0u) continue;
        for (GLuint j = i + 1; j < MAX_ATTRIBS; j++) {
            if ((base->enabled_attribs & (1u << j)) == 0u ||
                mglVertexArrayEffectiveBuffer(base, i) !=
                    mglVertexArrayEffectiveBuffer(base, j)) {
                continue;
            }
            uint8_t bi = (uint8_t)base->attrib[i].buffer_bindingindex;
            uint8_t bj = (uint8_t)base->attrib[j].buffer_bindingindex;
            uint32_t di = 0u, dj = 0u;
            for (uint8_t k = 0; k < cmd->dynamic_vertex_binding_count; k++) {
                const MGLDynamicVertexBinding *binding =
                    &cmd->dynamic_vertex_bindings[k];
                if (binding->binding_index == bi) di = binding->offset_delta;
                if (binding->binding_index == bj) dj = binding->offset_delta;
            }
            if (di != dj) return false;
        }
    }
    /* Indexed commands already carry an immutable elementBuffer pointer.
     * Treat an EBO-only VAO hash change as a capturable per-draw binding so
     * A -> B -> A index-buffer switches can remain in one direct batch. */
    bool element_binding_changed =
        cmd->elementBuffer && cmd->elementBuffer != base->element_array.buffer;
    return cmd->dynamic_vertex_binding_count > 0 || element_binding_changed;
}

static bool mglStateKeysEqualIgnoringDynamicBindings(const MGLStateKey *a,
                                                      const MGLStateKey *b)
{
    if (!a || !b ||
        (a->render_state_hash ^ a->uniform_buffer_hash) !=
        (b->render_state_hash ^ b->uniform_buffer_hash)) {
        return false;
    }

    MGLStateKey a_without_vao = *a;
    MGLStateKey b_without_vao = *b;
    a_without_vao.vao_name = 0;
    b_without_vao.vao_name = 0;
    a_without_vao.vertex_layout_hash = 0;
    b_without_vao.vertex_layout_hash = 0;
    a_without_vao.texture_hash = 0;
    b_without_vao.texture_hash = 0;
    a_without_vao.render_state_hash = 0;
    b_without_vao.render_state_hash = 0;
    a_without_vao.uniform_buffer_hash = 0;
    b_without_vao.uniform_buffer_hash = 0;
    return mglStateKeysEqual(&a_without_vao, &b_without_vao);
}

static bool mglCaptureDynamicTextureBindings(GLMContext ctx,
                                             const MGLDrawBatch *batch,
                                             MGLDrawCommand *cmd)
{
    if (!ctx || !batch || !cmd || !batch->state_snapshot) return false;

    const GLMState *snapshot = (const GLMState *)batch->state_snapshot;
    cmd->dynamic_texture_binding_count = 0;

    for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
        if (!mglStateSamplesTextureUnit(ctx, unit)) continue;

        bool snapshot_active =
            (snapshot->active_texture_mask[unit / 32u] & (1u << (unit % 32u))) != 0u;
        bool current_active =
            (ctx->active_state->active_texture_mask[unit / 32u] & (1u << (unit % 32u))) != 0u;
        if (snapshot_active != current_active ||
            snapshot->texture_samplers[unit] !=
                ctx->active_state->texture_samplers[unit]) {
            return false;
        }
        if (!current_active) continue;

        bool found_active = ctx->active_state->active_textures[unit] == NULL;
        for (GLuint target = 0; target < _MAX_TEXTURE_TYPES; target++) {
            Texture *base = snapshot->texture_units[unit].textures[target];
            Texture *current = ctx->active_state->texture_units[unit].textures[target];
            if ((base == NULL) != (current == NULL)) return false;
            if (!current) continue;
            if (current->index != target || (base && base->index != target) ||
                current->target != base->target ||
                cmd->dynamic_texture_binding_count >=
                    MGL_MAX_DYNAMIC_TEXTURE_BINDINGS) {
                return false;
            }

            MGLDynamicTextureBinding *binding =
                &cmd->dynamic_texture_bindings[
                    cmd->dynamic_texture_binding_count++];
            binding->unit = (uint8_t)unit;
            binding->target_index = (uint8_t)target;
            binding->is_active =
                ctx->active_state->active_textures[unit] == current ? 1u : 0u;
            binding->texture = current;
            found_active = found_active || binding->is_active != 0u;
        }
        if (!found_active) return false;
    }

    return cmd->dynamic_texture_binding_count > 0;
}

static bool mglCaptureDynamicDrawBindings(GLMContext ctx,
                                          const MGLDrawBatch *batch,
                                          const MGLStateKey *key,
                                          MGLDrawCommand *cmd)
{
    if (!ctx || !batch || !key || !cmd || !batch->state_snapshot) {
        return false;
    }

    bool capture_vao = batch->has_dynamic_vertex_bindings ||
        key->vao_name != batch->key.vao_name ||
        key->vertex_layout_hash != batch->key.vertex_layout_hash;
    if (capture_vao) {
        if (!mglCaptureDynamicVertexBindings(ctx, batch, cmd)) return false;
    }

    bool capture_uniforms = batch->has_dynamic_uniform_bindings ||
        key->uniform_buffer_hash != batch->key.uniform_buffer_hash;
    if (capture_uniforms &&
        !mglCaptureDynamicUniformRanges(ctx, batch, cmd,
                                        batch->has_dynamic_uniform_bindings)) {
        return false;
    }
    bool capture_textures = batch->has_dynamic_texture_bindings ||
        key->texture_hash != batch->key.texture_hash;
    if (capture_textures &&
        !mglCaptureDynamicTextureBindings(ctx, batch, cmd)) {
        return false;
    }
    return true;
}

static bool mglBatchIsMDICompatible(const MGLDrawBatch *batch, const MGLDrawCommand *cmd)
{
    uint8_t prim_type = batch->key.primitive_type;
    if (prim_type == 0xFF) return false;

    MGLDrawCommandType cmd_type = cmd->type;
    bool cmd_uses_elements = (cmd_type != MGL_CMD_DRAW_ARRAYS &&
                              cmd_type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
                              cmd_type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE);

    if (cmd_uses_elements != batch->uses_elements) return false;
    if (batch->uses_elements &&
        batch->command_count > 0 &&
        batch->commands &&
        batch->commands[0].indexType != cmd->indexType) {
        return false;
    }

    /* Emulated modes can't use MDI */
    GLenum mode = cmd->mode;
    if (mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP) return false;

    return true;
}

static void mglUpdateBatchSamplerSnapshotState(MGLDrawBatch *batch,
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

static bool mglCommandComputeElementVertexRange(GLMContext ctx,
                                                const MGLDrawCommand *cmd,
                                                uint64_t *minVertex,
                                                uint64_t *maxVertex)
{
    if (!ctx || !cmd || !minVertex || !maxVertex || cmd->count <= 0) return false;

    Buffer *elementBuffer = (Buffer *)cmd->elementBuffer;
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

    const uint8_t *data = (const uint8_t *)(uintptr_t)elementBuffer->data.buffer_data + offset;
    uint64_t restart = 0;
    bool hasRestart = mglCommandPrimitiveRestartIndex(ctx, cmd->indexType, &restart);

    uint64_t minIndex = UINT64_MAX;
    uint64_t maxIndex = 0;
    for (GLsizei i = 0; i < cmd->count; i++) {
        uint64_t index = 0;
        if (!mglCommandReadIndexValue(data, cmd->indexType, i, &index)) {
            return false;
        }
        if (hasRestart && index == restart) {
            continue;
        }
        if (index < minIndex) minIndex = index;
        if (index > maxIndex) maxIndex = index;
    }

    if (minIndex == UINT64_MAX) {
        return false;
    }

    if (cmd->baseVertex < 0) {
        uint64_t negBase = (uint64_t)(-((int64_t)cmd->baseVertex));
        *minVertex = (minIndex > negBase) ? (minIndex - negBase) : 0;
        *maxVertex = (maxIndex > negBase) ? (maxIndex - negBase) : 0;
    } else {
        uint64_t base = (uint64_t)cmd->baseVertex;
        if (minIndex > UINT64_MAX - base || maxIndex > UINT64_MAX - base) {
            return false;
        }
        *minVertex = minIndex + base;
        *maxVertex = maxIndex + base;
    }

    return true;
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

static void mglTrackPendingBaseBufferReads(GLMContext ctx)
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
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
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

    MGL_PERF_ADD(g_mglHazardActiveBindingsSinceSwap, activeCount);
}

#define MGL_STREAM_MERGE_MAX_SOURCE_BYTES (64u * 1024u)
#define MGL_STREAM_MERGE_MAX_BATCH_BYTES  (4u * 1024u * 1024u)

typedef struct {
    VertexArray *vao;
    Buffer      *vertex_buffer;
    Buffer      *index_buffer;
    const uint8_t *index_bytes;
    size_t       index_size;
    bool         uses_elements;
    size_t       vertex_bytes;
    size_t       vertex_source_offset;
    size_t       vertex_stride;
    size_t       max_attrib_end;
    uint64_t     source_first_index;
    GLintptr     binding_offset;
    uint32_t     attrib_mask;
    uint64_t     layout_hash;
} MGLStreamMergeCandidate;

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

static bool mglEnsureTransientBufferCapacity(Buffer *buffer, size_t needed)
{
    if (!buffer) return false;
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
    buffer->data.dirty_bits |= DIRTY_BUFFER_ADDR | DIRTY_BUFFER_DATA;
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

static bool mglPrepareStreamMergeCandidate(GLMContext ctx,
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
    Buffer *indexBuffer = (Buffer *)cmd->elementBuffer;
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
    for (GLsizei i = 0; i < cmd->count; i++) {
        uint64_t rawIndex = (uint64_t)cmd->first + (uint64_t)i;
        if (uses_elements &&
            !mglCommandReadIndexValue(indices, cmd->indexType, i, &rawIndex)) {
            return false;
        }

        int64_t sourceIndex = uses_elements
            ? (int64_t)rawIndex + (int64_t)cmd->baseVertex
            : (int64_t)rawIndex;
        if (sourceIndex < 0) return false;
        uint64_t sourceIndexU = (uint64_t)sourceIndex;
        if (sourceIndexU > (UINT64_MAX - (uint64_t)bindingOffset) / (uint64_t)vertexStride) {
            return false;
        }

        uint64_t vertexByte = (uint64_t)bindingOffset +
                              (sourceIndexU * (uint64_t)vertexStride);
        if (vertexByte > UINT64_MAX - (uint64_t)maxAttribEnd ||
            vertexByte + (uint64_t)maxAttribEnd > (uint64_t)vertexBuffer->size) {
            return false;
        }
        if (sourceIndexU < minSourceIndex) minSourceIndex = sourceIndexU;
        if (sourceIndexU > maxSourceIndex) maxSourceIndex = sourceIndexU;
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
    out->binding_offset = bindingOffset;
    out->attrib_mask = attribMask;
    out->layout_hash = layoutHash ^
                       mglRotateLeft64((uint64_t)vertexStride, 37) ^
                       mglRotateLeft64((uint64_t)bindingOffset, 43);
    return true;
}

static bool mglInitializeStreamMergedBatch(GLMContext ctx,
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
    batch->stream_vertex_buffer = mglNewTransientBatchBuffer(GL_ARRAY_BUFFER);
    batch->stream_index_buffer = mglNewTransientBatchBuffer(GL_ELEMENT_ARRAY_BUFFER);
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
    vao->mtl_data = NULL;
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

    MGL_PERF_INC(g_mglSnapshotAllocationCountSinceSwap);
    MGL_PERF_ADD(g_mglSnapshotBytesAllocatedSinceSwap,
                 sizeof(GLMState) + sizeof(VertexArray));

    mglRetainBatchProgramReferences(ctx, batch);
    mglRetainBatchBufferReferences(batch);
    MGL_SIGNPOST_END(InitStreamMergedBatch);
    return true;
}

static bool mglAppendStreamMergedData(MGLDrawBatch *batch,
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

    for (GLsizei i = 0; i < srcCmd->count; i++) {
        uint64_t rawIndex = (uint64_t)srcCmd->first + (uint64_t)i;
        if (candidate->uses_elements &&
            !mglCommandReadIndexValue(candidate->index_bytes, srcCmd->indexType, i, &rawIndex)) {
            return false;
        }
        int64_t sourceIndex = candidate->uses_elements
            ? (int64_t)rawIndex + (int64_t)srcCmd->baseVertex
            : (int64_t)rawIndex;
        if (sourceIndex < 0) return false;
        uint64_t sourceIndexU = (uint64_t)sourceIndex;
        if (sourceIndexU < candidate->source_first_index) return false;
        uint64_t mergedIndex = vertexBase + (sourceIndexU - candidate->source_first_index);
        if (mergedIndex > UINT32_MAX) return false;
    }

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
    storedCmd->elementBuffer = indexBuffer;
    storedCmd->baseVertex = 0;
    storedCmd->baseInstance = 0;
    return true;
}

static void mglTrackPendingDrawBufferReads(GLMContext ctx,
                                           const MGLDrawCommand *cmd,
                                           bool uses_elements)
{
    if (!ctx || !cmd) return;

    if (uses_elements && cmd->elementBuffer && cmd->count > 0) {
        size_t indexSize = mglCommandIndexSize(cmd->indexType);
        if (indexSize > 0) {
            uint64_t indexBytes = (uint64_t)cmd->count * (uint64_t)indexSize;
            mglTrackPendingReadBytes(ctx,
                                     (Buffer *)cmd->elementBuffer,
                                     (uint64_t)cmd->indexBufferOffset,
                                     indexBytes);
        } else {
            mglTrackPendingReadWholeBuffer(ctx, (Buffer *)cmd->elementBuffer);
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

void mglRecordDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd)
{
    MGL_SIGNPOST_BEGIN(RecordDrawCommand);
    if (!ctx || !cmd) {
        MGL_SIGNPOST_END(RecordDrawCommand);
        return;
    }

    /* DrawCommand Recorder: keep the GL entry points shallow and capture a
     * validated draw into the deferred command buffer.  The Batch Builder and
     * Batch Queue stages below decide whether the command joins a direct batch
     * or a stream-merge batch. */
    mglFlushPendingDrawsForActiveTextures(ctx);
    mglFlushPendingDrawsBeforeFramebufferTextureWrites(ctx);

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    uint32_t maxDrawsPerBatch = mglRuntimeMaxDrawsPerBatch();
    uint32_t maxBatchCount = mglRuntimeMaxBatchCount();
    bool cmd_uses_elements =
        (cmd->type != MGL_CMD_DRAW_ARRAYS &&
         cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
         cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE);

    /* A short snapshot ID is only valid for the current command buffer.
     * Avoid a later capacity flush invalidating an ID captured below. */
    if (cb->batch_count >= maxBatchCount) {
        MGL_PERF_INC(g_mglFlushReasonCapacitySinceSwap);
        mglFlushCommandBuffer(ctx);
    }

    MGLDrawCommand stored_cmd = *cmd;
    stored_cmd.sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
    if (mglSamplerSnapshotEnabled() && !cb->sampler_snapshot_incomplete) {
        bool captured =
            mglCaptureSamplerSnapshot(ctx, &stored_cmd.sampler_snapshot_id);
        if (!captured && cb->total_commands > 0) {
            mglFlushCommandBuffer(ctx);
            captured =
                mglCaptureSamplerSnapshot(ctx, &stored_cmd.sampler_snapshot_id);
        }
        if (!captured) {
            cb->sampler_snapshot_incomplete = true;
            stored_cmd.sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
        }
    }

    MGLStateKey key;
    mglComputeStateKey(ctx, cmd->mode, cmd_uses_elements, &key);

    MGLStreamMergeCandidate streamCandidate;
    bool can_stream_merge =
        mglPrepareStreamMergeCandidate(ctx, cmd, cmd_uses_elements, &streamCandidate);
    if (can_stream_merge && cb->batch_count > 0) {
        MGLDrawBatch *last = &cb->batches[cb->batch_count - 1];
        if (mglStateKeysEqual(&last->key, &key) &&
            (last->sampler_snapshots_mixed ||
             last->sampler_snapshot_id != stored_cmd.sampler_snapshot_id)) {
            /* Once sampler state changes, keep subsequent draws in one mixed
             * direct batch instead of consuming a stream batch per snapshot. */
            can_stream_merge = false;
        }
    }
    /*
     * Batch reuse is decoupled from stream-merge.  When keys match, non-
     * stream-merged draws can share a batch — each draw keeps its own
     * MGLDrawCommand (first/count/baseVertex) but reuses the shared
     * state_snapshot.  Hazard detection (mglFlushPendingDrawsForBuffer/
     * VertexArray/Texture) flushes the batch before any state change that
     * would invalidate the snapshot, so reuse is safe.
     */
    bool can_reuse_batch = true;

    /* Find matching batch (check last first for spatial locality).
     *
     * Batch matching uses memcmp on the full MGLStateKey struct (via
     * mglStateKeysEqual).  The key contains hash fields for texture
     * bindings, render state (blend/depth/stencil/etc.), and vertex layout.
     * A hash collision would cause two different states to be falsely merged
     * into one batch, resulting in the second draw using the first draw's
     * render or vertex state during replay.
     *
     * Collision probability: for N=128 batches, P(collision) ≈ N²/2^65
     * ≈ 2^-51, which is negligible.  The hash incorporates pointer
     * addresses (for textures) and ~50 state fields (for render state),
     * making intentional collision infeasible.  If correctness is ever
     * questioned, enable MGL_DEBUG_STREAM_MERGE to log batch merges. */
    MGLDrawBatch *batch = NULL;
    if (can_reuse_batch && cb->batch_count > 0) {
        MGLDrawBatch *last = &cb->batches[cb->batch_count - 1];
        bool keys_match = mglStateKeysEqual(&last->key, &key);
        if (keys_match &&
            last->uses_elements == cmd_uses_elements &&
            last->stream_merged == can_stream_merge &&
            (!can_stream_merge ||
             (last->stream_layout_hash == streamCandidate.layout_hash &&
              !last->sampler_snapshots_mixed &&
              last->sampler_snapshot_id == stored_cmd.sampler_snapshot_id)) &&
            last->command_count < maxDrawsPerBatch &&
            (!last->has_dynamic_uniform_bindings ||
             mglCaptureDynamicUniformRanges(ctx, last, &stored_cmd, true)) &&
            (!last->has_dynamic_texture_bindings ||
             mglCaptureDynamicTextureBindings(ctx, last, &stored_cmd)) &&
            (!last->has_dynamic_vertex_bindings ||
             mglCaptureDynamicVertexBindings(ctx, last, &stored_cmd))) {
            batch = last;
        } else if (!keys_match && !can_stream_merge && !last->stream_merged &&
                   last->uses_elements == cmd_uses_elements &&
                   last->command_count < maxDrawsPerBatch &&
                   mglStateKeysEqualIgnoringUniformRanges(&last->key, &key) &&
                   mglCaptureDynamicUniformRanges(
                       ctx, last, &stored_cmd,
                       last->has_dynamic_uniform_bindings) &&
                   (!last->has_dynamic_texture_bindings ||
                    mglCaptureDynamicTextureBindings(ctx, last, &stored_cmd)) &&
                   (!last->has_dynamic_vertex_bindings ||
                    mglCaptureDynamicVertexBindings(ctx, last, &stored_cmd))) {
            batch = last;
            batch->has_dynamic_uniform_bindings = true;
            batch->mdi_compatible = false;
        } else if (!keys_match && mglBindNoFlushEnabled() &&
                   !can_stream_merge && !last->stream_merged &&
                   last->uses_elements == cmd_uses_elements &&
                   last->command_count < maxDrawsPerBatch &&
                   mglStateKeysEqualIgnoringDynamicBindings(&last->key, &key) &&
                   mglCaptureDynamicDrawBindings(ctx, last, &key, &stored_cmd)) {
            batch = last;
            if (stored_cmd.dynamic_vertex_binding_count > 0) {
                batch->has_dynamic_vertex_bindings = true;
            }
            if (stored_cmd.dynamic_uniform_binding_count > 0) {
                batch->has_dynamic_uniform_bindings = true;
            }
            if (stored_cmd.dynamic_texture_binding_count > 0) {
                batch->has_dynamic_texture_bindings = true;
            }
            batch->mdi_compatible = false;
        } else if (!keys_match) {
            MGL_PERF_INC(g_mglMergeRejectStateDiffersSinceSwap);
        } else if (last->command_count >= maxDrawsPerBatch) {
            MGL_PERF_INC(g_mglMergeRejectAppendFailedSinceSwap);
        }
    }
    if (!batch) {
        if (cb->batch_count >= maxBatchCount) {
            MGL_PERF_INC(g_mglFlushReasonCapacitySinceSwap);
            mglFlushCommandBuffer(ctx);
            if (cb->batch_count >= maxBatchCount) {
                fprintf(stderr, "MGL Error: mglAppendDrawCommand: batch buffer full after flush\n");
                MGL_SIGNPOST_END(RecordDrawCommand);
                return;
            }
        }
        batch = &cb->batches[cb->batch_count];
        memset(batch, 0, sizeof(*batch));
        batch->sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
        batch->key = key;
        batch->uses_elements = cmd_uses_elements;
        /* The frontend command is zero-initialized and does not own the
         * command-buffer-local snapshot ID.  Check the normalized copy so a
         * draw without a sampler snapshot is not mistaken for snapshot 0. */
        batch->mdi_compatible = mglBatchIsMDICompatible(batch, &stored_cmd);

        if (can_stream_merge) {
            if (!mglInitializeStreamMergedBatch(ctx, batch, &streamCandidate)) {
                static unsigned long long s_streamMergeInitFailCount = 0;
                unsigned long long smInitHit = ++s_streamMergeInitFailCount;
                if (smInitHit <= 64ull || (smInitHit % 512ull) == 0ull) {
                    fprintf(stderr, "MGL Warning: stream merged batch init failed; falling back to normal deferred draw (hit=%llu)\n",
                            smInitHit);
                }
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                batch->sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
                batch->key = key;
                batch->mdi_compatible = false;
                batch->uses_elements = cmd_uses_elements;
                can_stream_merge = false;
            }
        }

        if (!can_stream_merge) {
            if (!mglInitializeBatchStateSnapshot(ctx, batch)) {
                fprintf(stderr, "MGL Error: mglAppendDrawCommand: state snapshot alloc failed\n");
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                MGL_SIGNPOST_END(RecordDrawCommand);
                return;
            }
        }

        cb->batch_count++;
    }

    if (can_stream_merge) {
        if (!mglAppendStreamMergedData(batch, &streamCandidate, cmd, &stored_cmd)) {
            MGL_PERF_INC(g_mglMergeRejectAppendFailedSinceSwap);
            static unsigned long long s_streamMergeAppendFailCount = 0;
            unsigned long long smAppendHit = ++s_streamMergeAppendFailCount;
            if (smAppendHit <= 64ull || (smAppendHit % 512ull) == 0ull) {
                fprintf(stderr, "MGL Warning: stream merged append failed; falling back to normal deferred draw (hit=%llu)\n",
                        smAppendHit);
            }
            if (batch->command_count == 0 &&
                cb->batch_count > 0 &&
                batch == &cb->batches[cb->batch_count - 1]) {
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                cb->batch_count--;
            }
            can_stream_merge = false;
            batch = NULL;
            if (!batch) {
                if (cb->batch_count >= maxBatchCount) {
                    MGL_PERF_INC(g_mglFlushReasonCapacitySinceSwap);
                    mglFlushCommandBuffer(ctx);
                    if (cb->batch_count >= maxBatchCount) {
                        fprintf(stderr, "MGL Error: mglAppendDrawCommand: fallback batch buffer full after flush\n");
                        MGL_SIGNPOST_END(RecordDrawCommand);
                        return;
                    }
                }
                batch = &cb->batches[cb->batch_count];
                memset(batch, 0, sizeof(*batch));
                batch->sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
                batch->key = key;
                batch->mdi_compatible = false;
                batch->uses_elements = cmd_uses_elements;
                if (!mglInitializeBatchStateSnapshot(ctx, batch)) {
                    fprintf(stderr, "MGL Error: mglAppendDrawCommand: fallback state snapshot alloc failed\n");
                    mglReleaseBatch(ctx, batch);
                    memset(batch, 0, sizeof(*batch));
                    MGL_SIGNPOST_END(RecordDrawCommand);
                    return;
                }
                cb->batch_count++;
            }
            uint16_t sampler_snapshot_id = stored_cmd.sampler_snapshot_id;
            stored_cmd = *cmd;
            stored_cmd.sampler_snapshot_id = sampler_snapshot_id;
        }
    }

    if (batch->command_count >= batch->command_capacity) {
        uint32_t newCapacity = batch->command_capacity ? batch->command_capacity * 2u : 64u;
        if (newCapacity < batch->command_capacity || newCapacity > maxDrawsPerBatch) {
            newCapacity = maxDrawsPerBatch;
        }
        if (newCapacity <= batch->command_count) {
            fprintf(stderr, "MGL Error: mglAppendDrawCommand: command capacity exhausted\n");
            MGL_SIGNPOST_END(RecordDrawCommand);
            return;
        }
        size_t newBytes = (size_t)newCapacity * sizeof(MGLDrawCommand);
        MGLDrawCommand *new_cmds;
        if (batch->arena_managed && ctx->batch_arena && ctx->batch_arena->enabled) {
            /* Arena path: allocate new block and copy; old block is
             * reclaimed on arena reset (no individual free). */
            new_cmds = (MGLDrawCommand *)arenaAlloc(ctx->batch_arena, newBytes);
            if (new_cmds && batch->commands && batch->command_count > 0) {
                memcpy(new_cmds, batch->commands,
                       (size_t)batch->command_count * sizeof(MGLDrawCommand));
            }
        } else {
            new_cmds = (MGLDrawCommand *)realloc(batch->commands, newBytes);
        }
        if (!new_cmds) {
            fprintf(stderr, "MGL Error: mglAppendDrawCommand: realloc failed\n");
            if (batch->command_count == 0 &&
                cb->batch_count > 0 &&
                batch == &cb->batches[cb->batch_count - 1]) {
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                cb->batch_count--;
            }
            MGL_SIGNPOST_END(RecordDrawCommand);
            return;
        }
        batch->commands = new_cmds;
        batch->command_capacity = newCapacity;
    }
    mglUpdateBatchSamplerSnapshotState(batch, stored_cmd.sampler_snapshot_id);
    batch->commands[batch->command_count] = stored_cmd;
    batch->command_count++;
    cb->total_commands++;

    if (!mglBatchIsMDICompatible(batch, &stored_cmd)) {
        batch->mdi_compatible = false;
    }

    if (batch->uses_elements) {
        cb->element_cmd_count++;
    } else {
        cb->array_cmd_count++;
    }

    if (can_stream_merge) {
        mglTrackPendingBaseBufferReads(ctx);
    } else {
        mglTrackPendingDrawBufferReads(ctx, cmd, cmd_uses_elements);
    }
    mglTrackPendingSampledTextureReads(ctx);
    mglTrackPendingFramebufferTextureWrites(ctx);
    MGL_SIGNPOST_END(RecordDrawCommand);
}

void mglAppendDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd)
{
    mglRecordDrawCommand(ctx, cmd);
}

/*
 * mglFlushCommandBuffer — 最低级 flush，将延迟 draw 缓冲区提交给 Metal
 *
 * 触发条件：当 batch_count > 0 时，通过 ctx->mtl_funcs.mtlFlushDrawBuffer(ctx) 将已编码的
 *           延迟 draw 提交到 Metal 后端执行。
 * 保证语义：所有危害检测 flush 函数的最终汇聚点；调用后所有 per-CB read/write 跟踪数组被清空。
 * 溢出退化：无（本函数不做危害检测，仅负责提交）。
 */
void mglFlushCommandBuffer(GLMContext ctx)
{
    if (!ctx) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0) return;

    if (ctx->mtl_funcs.mtlFlushDrawBuffer) {
        MGL_PERF_INC(g_mglFlushTotalSinceSwap);
        ctx->mtl_funcs.mtlFlushDrawBuffer(ctx);
    }
}

/*
 * mglFlushPendingDraws — 无条件全刷新（无危害检测）
 *
 * 触发条件：当 draw_defer_enabled 时无条件调用 mglFlushCommandBuffer；不做任何危害检测。
 * 保证语义：用于任何可能广泛使待处理 draw 失效的状态变更（如广泛的 GL 状态重置），
 *           保守地保证所有已编码 draw 提交完毕后再进行后续操作。
 * 溢出退化：本函数本身就是全刷新，等价于溢出退化后的最终行为。
 */
void mglFlushPendingDraws(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;
    mglFlushCommandBuffer(ctx);
}

const char *mglDrawCommandTypeName(MGLDrawCommandType type)
{
    switch (type) {
        case MGL_CMD_DRAW_ARRAYS: return "draw_arrays";
        case MGL_CMD_DRAW_ELEMENTS: return "draw_elements";
        case MGL_CMD_DRAW_ARRAYS_INSTANCED: return "draw_arrays_instanced";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED: return "draw_elements_instanced";
        case MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX: return "draw_elements_base_vertex";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX: return "draw_elements_instanced_base_vertex";
        case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE: return "draw_arrays_instanced_base_instance";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE: return "draw_elements_instanced_base_instance";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE: return "draw_elements_instanced_base_vertex_base_instance";
        default: return "unknown";
    }
}

bool mglDrawCommandUsesElements(const MGLDrawCommand *cmd)
{
    if (!cmd) {
        return false;
    }

    switch (cmd->type) {
        case MGL_CMD_DRAW_ARRAYS:
        case MGL_CMD_DRAW_ARRAYS_INSTANCED:
        case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
            return false;
        default:
            return true;
    }
}

