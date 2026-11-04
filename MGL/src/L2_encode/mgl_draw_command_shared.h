/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * Cross-TU helpers for draw_command Phase-7 split.
 */
#ifndef MGL_DRAW_COMMAND_SHARED_H
#define MGL_DRAW_COMMAND_SHARED_H

#include "draw_command.h"
#include "glm_context.h"
#include "mgl_byte_hash.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

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

static inline uint64_t mglHashBytes64_SIMD(const void *data, size_t size, uint64_t seed)
{
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = seed ^ 0xcbf29ce484222325ULL;

    if (!bytes) return hash;

    /* SIMD path: process 16-byte chunks with NEON.
     * Threshold is 32 bytes (2+ chunks): for 16-byte inputs the scalar
     * 8-byte loop is faster because the NEON path's vectorized XOR is
     * offset by 2× lane extraction + scalar multiply (vmulq_u64 carries
     * 128-bit product, unsuitable for FNV-1a).  Small inputs dominate
     * mglComputeRenderStateHash (~15 calls of 4-32 bytes) and
     * mglHashCurrentVertexAttrib (4 calls of 4-16 bytes per attrib). */
    if (size >= 32) {
        const uint8_t *ptr = bytes;
        size_t chunks = size / 16;

        /* Load FNV prime as vector for parallel multiply */
        const uint64_t fnv_prime = 0x100000001b3ULL;
        uint64x2_t hash_vec = vdupq_n_u64(hash);

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

static inline uint64_t mglHashBytes64(const void *data, size_t size, uint64_t seed)
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

typedef struct MGLStreamMergeCandidate {
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
    uint64_t     source_last_index;
    GLintptr     binding_offset;
    uint32_t     attrib_mask;
    uint64_t     layout_hash;
} MGLStreamMergeCandidate;

static inline void mglNoteOccupancyPeak(uint32_t *peak, uint32_t value)
{
    if (peak && value > *peak) {
        *peak = value;
    }
}

extern uint32_t s_peak_buffer_read_ranges;
extern uint32_t s_peak_texture_writes;
extern uint32_t s_peak_texture_reads;
extern uint32_t s_peak_sampler_snapshot_keys;
extern uint32_t s_peak_sampler_snapshot_sets;
extern uint32_t s_peak_pending_vao_refs;
extern uint32_t s_peak_pending_vao_names;

void *arenaAlloc(MGLBatchArena *arena, size_t size);
Buffer *mglAcquireTransientBatchBuffer(GLMContext ctx, GLenum target);
void mglRecycleTransientBatchBuffer(GLMContext ctx, Buffer *buffer);
void mglDrainTransientPool(GLMContext ctx);
void mglDestroyTransientBuffer(GLMContext ctx, Buffer *buffer);

bool mglInitializeBatchStateSnapshot(GLMContext ctx, MGLDrawBatch *batch);
bool mglInitializeOrShareBatchStateSnapshot(GLMContext ctx, MGLCommandBuffer *cb,
                                            MGLDrawBatch *batch);
void mglReleaseBatch(GLMContext ctx, MGLDrawBatch *batch);
void mglRetainBatchBufferReferences(MGLDrawBatch *batch);
void mglRetainBatchProgramReferences(GLMContext ctx, MGLDrawBatch *batch);

uint32_t mglRuntimeMaxDrawsPerBatch(void);
uint32_t mglRuntimeMaxBatchCount(void);
const uint32_t *mglGetActiveSampledTextureUnitMask(GLMContext ctx);
void mglTrackPendingFramebufferTextureWrites(GLMContext ctx);
void mglTrackPendingSampledTextureReads(GLMContext ctx);
void mglFlushPendingDrawsBeforeFramebufferTextureWrites(GLMContext ctx);
bool mglCaptureSamplerSnapshot(GLMContext ctx, uint16_t *snapshot_id);
void mglTrackPendingReadRange(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t end);
void mglTrackPendingReadBytes(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t size);
void mglTrackPendingReadWholeBuffer(GLMContext ctx, Buffer *buffer);

bool mglCaptureDynamicDrawBindings(GLMContext ctx, const MGLDrawBatch *batch,
                                   const MGLStateKey *key, MGLDrawCommand *cmd);
bool mglCaptureDynamicTextureBindings(GLMContext ctx, const MGLDrawBatch *batch,
                                      MGLDrawCommand *cmd);
bool mglCaptureDynamicUniformRanges(GLMContext ctx, const MGLDrawBatch *batch,
                                    MGLDrawCommand *cmd,
                                    bool capture_all_bound_ranges);
bool mglCaptureDynamicVertexBindings(GLMContext ctx, const MGLDrawBatch *batch,
                                     MGLDrawCommand *cmd);
bool mglStateKeysEqualIgnoringDynamicBindings(const MGLStateKey *a,
                                              const MGLStateKey *b);
bool mglStateKeysEqualIgnoringUniformRanges(const MGLStateKey *a,
                                            const MGLStateKey *b);
bool mglStateKeysEqualIgnoringUniformAndElementRanges(
    const MGLStateKey *a, const MGLStateKey *b, const VertexArray *vao_a,
    const VertexArray *vao_b, const CurrentVertexAttrib *current_attribs_a,
    const CurrentVertexAttrib *current_attribs_b, bool uses_elements);
bool mglStateKeysEqualIgnoringUniformElementAndTextures(
    const MGLStateKey *a, const MGLStateKey *b, const VertexArray *vao_a,
    const VertexArray *vao_b, const CurrentVertexAttrib *current_attribs_a,
    const CurrentVertexAttrib *current_attribs_b, bool uses_elements);

bool mglBatchInitMDICompatible(const MGLDrawBatch *batch, const MGLDrawCommand *cmd);
void mglBatchUpdateMDIOnAppend(MGLDrawBatch *batch, const MGLDrawCommand *cmd);
void mglUpdateBatchSamplerSnapshotState(MGLDrawBatch *batch, uint16_t snapshot_id);
bool mglPrepareStreamMergeCandidate(GLMContext ctx, const MGLDrawCommand *cmd,
                                    bool uses_elements,
                                    MGLStreamMergeCandidate *out);
bool mglInitializeStreamMergedBatch(GLMContext ctx, MGLDrawBatch *batch,
                                    const MGLStreamMergeCandidate *candidate);
bool mglAppendStreamMergedData(MGLDrawBatch *batch,
                               const MGLStreamMergeCandidate *candidate,
                               const MGLDrawCommand *srcCmd,
                               MGLDrawCommand *storedCmd);
void mglTrackPendingBaseBufferReads(GLMContext ctx);
void mglTrackPendingDrawBufferReads(GLMContext ctx, const MGLDrawCommand *cmd,
                                    bool uses_elements);

#endif
