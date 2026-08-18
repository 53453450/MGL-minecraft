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
 * mgl_index_buffer.h
 * MGL
 *
 * Index Buffer Builder Subsystem.
 *
 * Metal does not support GL primitive types GL_TRIANGLE_FAN, GL_LINE_LOOP,
 * GL_QUADS, or GL_UNSIGNED_BYTE index buffers.  This module builds
 * Metal-compatible UInt32 index buffers that expand those GL primitives into
 * triangle lists / line lists, and provides the buffer-source readers used by
 * the draw-element path to locate CPU-readable index bytes.
 *
 * All functions are pure (no self/ivar dependency).  The 5 hot-path helpers
 * are static inline in the header; the 17 buffer builders / source readers
 * are extern in the .m file.
 *
 * Dependencies:
 *   - glcorearb.h (GL enums, GLuint/GLsizei)
 *   - glm_context.h (Buffer, GLMContext)
 *   - mgl_vertex_format.h (mglGLIndexElementSize, mglReadGLIndexValue)
 *   - opaque Metal handles owned by mgl_render.cpp
 */

#ifndef MGL_INDEX_BUFFER_H
#define MGL_INDEX_BUFFER_H

#include "glcorearb.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifndef __OBJC__
typedef size_t NSUInteger;
#ifndef NSUIntegerMax
#define NSUIntegerMax SIZE_MAX
#endif
#endif

#include "glm_context.h"
#include "mgl_vertex_format.h"

#ifdef __OBJC__
typedef id MGLIndexMetalHandle;
#else
typedef void *MGLIndexMetalHandle;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decl: the pure scan lives in mgl_render.cpp (both gates); the
 * inline helper below is a thin gate-agnostic shim so every caller — the
 * element draw validation in Draw.m and the DrawSupport cull-distance path —
 * exercises the same C++ implementation. */
int mglRenderScanIndexRangeIgnoringRestart(
    const uint8_t *bytes, uint32_t elem_width, uint32_t count,
    int restart_enabled, uint32_t restart_index,
    uint32_t *out_min, uint32_t *out_max, int *out_valid);

int mglRenderPrimitiveRestartFixedIndex(uint64_t gl_index_type, uint32_t *out);

int mglRenderComputePreparedIndexByteOffset(uint64_t gl_index_type,
                                               uint64_t gl_byte_offset,
                                               uint64_t *out_prepared_offset);

uint64_t mglRenderQuadTriangleIndexCount(uint64_t source_vertex_count);

int mglRenderComputeIndexByteOffset(uint64_t base_byte_offset,
                                       uint64_t first_element,
                                       uint64_t index_stride,
                                       uint64_t *out_byte_offset);

/* === Inline helpers (hot-path) === */

/* Scans `count` indices of `indexBytes` (type `indexType`), skipping entries
 * equal to `restartIndex` when `primitiveRestartEnabled` is true, and returns
 * the min/max index values via out params.  Returns false on invalid args. */
static inline bool mglScanIndexRangeIgnoringRestart(const uint8_t *indexBytes,
                                                    GLenum indexType,
                                                    GLsizei count,
                                                    bool primitiveRestartEnabled,
                                                    uint32_t restartIndex,
                                                    uint32_t *outMin,
                                                    uint32_t *outMax)
{
    if (!indexBytes || !outMin || !outMax) {
        return false;
    }
    const uint32_t elemWidth = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t lo = 0u, hi = 0u;
    int valid = 0;
    if (mglRenderScanIndexRangeIgnoringRestart(
            indexBytes, elemWidth, (uint32_t)(count > 0 ? count : 0),
            primitiveRestartEnabled ? 1 : 0, restartIndex,
            &lo, &hi, &valid) != 0 || !valid) {
        return false;
    }
    *outMin = lo;
    *outMax = hi;
    return true;
}

/* Resolves the primitive-restart index value for `indexType` based on the
 * context's caps (primitive_restart / primitive_restart_fixed_index).
 * Returns false (and leaves *outRestartIndex untouched) if restart is not
 * enabled. */
static inline bool mglPrimitiveRestartIndexForType(GLMContext ctx,
                                                   GLenum indexType,
                                                   uint32_t *outRestartIndex)
{
    if (!ctx || (!ctx->active_state->caps.primitive_restart && !ctx->active_state->caps.primitive_restart_fixed_index)) {
        return false;
    }

    uint32_t restartIndex = 0u;
    if (ctx->active_state->caps.primitive_restart_fixed_index) {
        if (mglRenderPrimitiveRestartFixedIndex((uint64_t)indexType,
                                                   &restartIndex) != 1) {
            return false;
        }
    } else {
        restartIndex = ctx->active_state->var.primitive_restart_index;
    }

    if (outRestartIndex) {
        *outRestartIndex = restartIndex;
    }
    return true;
}

/* Computes the total triangle index count for `sourceIndexCount` vertices
 * arranged as quads (4 vertices per quad → 2 triangles → 6 indices).
 * Returns 0 on overflow. */
static inline NSUInteger mglQuadTriangleIndexCount(NSUInteger sourceIndexCount)
{
    uint64_t c = mglRenderQuadTriangleIndexCount((uint64_t)sourceIndexCount);
    if (c > (uint64_t)NSUIntegerMax) {
        return 0u;
    }
    return (NSUInteger)c;
}

/* Computes baseByteOffset + firstElement * indexStride with overflow checks.
 * Pure arithmetic; the logic lives in mgl_render.cpp and this inline is a
 * thin delegating shim. */
static inline bool mglComputeIndexByteOffset(NSUInteger baseByteOffset,
                                             NSUInteger firstElement,
                                             NSUInteger indexStride,
                                             NSUInteger *outByteOffset)
{
    if (!outByteOffset) {
        return false;
    }
    uint64_t out = 0u;
    int r = mglRenderComputeIndexByteOffset(
        (uint64_t)baseByteOffset, (uint64_t)firstElement, (uint64_t)indexStride, &out);
    if (r != 0) {
        return false;
    }
    *outByteOffset =
        (NSUInteger)(out > (uint64_t)NSUIntegerMax ? (uint64_t)NSUIntegerMax : out);
    return true;
}

/* Computes the prepared (Metal-side) byte offset for a GL element buffer.
 * GL_UNSIGNED_BYTE indices are expanded to UInt16, so the byte offset is
 * doubled; other index types pass through unchanged.  Pure arithmetic; the
 * logic lives in mgl_render.cpp and this inline is a thin delegating
 * shim. */
static inline bool mglComputePreparedIndexByteOffset(GLenum glIndexType,
                                                    NSUInteger glByteOffset,
                                                    NSUInteger *outPreparedByteOffset)
{
    uint64_t out = 0u;
    uint64_t type = glIndexType;
    uint64_t off = (uint64_t)glByteOffset;
    int ok = mglRenderComputePreparedIndexByteOffset(type, off, &out);
    if (ok != 0 || !outPreparedByteOffset) {
        return false;
    }
    *outPreparedByteOffset =
        (NSUInteger)(out > (uint64_t)NSUIntegerMax ? (uint64_t)NSUIntegerMax : out);
    return true;
}

/* === Metal index buffer builders (extern) ===
 *
 * Each builder allocates a new MTLBuffer with UInt32 indices that expand the
 * GL primitive into Metal-compatible triangle/line lists.  Returns nil on
 * failure.  *outIndexCount receives the number of indices in the new buffer
 * (0 on failure). */

/* Array (non-indexed) variants — generate sequential vertex indices. */
MGLIndexMetalHandle mglNewTriangleFanArrayIndexBuffer(MGLIndexMetalHandle device,
                                                       size_t vertexCount,
                                                       size_t *outIndexCount);

MGLIndexMetalHandle mglNewLineLoopArrayIndexBuffer(MGLIndexMetalHandle device,
                                                    size_t firstVertex,
                                                    size_t vertexCount,
                                                    size_t *outIndexCount);

MGLIndexMetalHandle mglNewTriangleStripArrayIndexBuffer(MGLIndexMetalHandle device,
                                                         size_t vertexCount,
                                                         size_t *outIndexCount);

MGLIndexMetalHandle mglNewQuadArrayIndexBuffer(MGLIndexMetalHandle device,
                                                size_t vertexCount,
                                                size_t *outIndexCount);

MGLIndexMetalHandle mglNewQuadArrayLineIndexBuffer(MGLIndexMetalHandle device,
                                                    size_t vertexCount,
                                                    size_t *outIndexCount);

/* Element (indexed) variants — read source indices and expand. */
MGLIndexMetalHandle mglNewTriangleFanElementIndexBuffer(MGLIndexMetalHandle device,
                                                         const uint8_t *sourceIndexBytes,
                                                         GLenum sourceIndexType,
                                                         size_t sourceIndexCount,
                                                         size_t *outIndexCount);

MGLIndexMetalHandle mglNewTriangleStripElementIndexBuffer(MGLIndexMetalHandle device,
                                                           const uint8_t *sourceIndexBytes,
                                                           GLenum sourceIndexType,
                                                           size_t sourceIndexCount,
                                                           size_t *outIndexCount);

MGLIndexMetalHandle mglNewLineLoopElementIndexBuffer(MGLIndexMetalHandle device,
                                                      const uint8_t *sourceIndexBytes,
                                                      GLenum sourceIndexType,
                                                      size_t sourceIndexCount,
                                                      size_t *outIndexCount);

MGLIndexMetalHandle mglNewQuadElementIndexBuffer(MGLIndexMetalHandle device,
                                                  const uint8_t *sourceIndexBytes,
                                                  GLenum sourceIndexType,
                                                  size_t sourceIndexCount,
                                                  size_t *outIndexCount);

MGLIndexMetalHandle mglNewQuadElementLineIndexBuffer(MGLIndexMetalHandle device,
                                                      const uint8_t *sourceIndexBytes,
                                                      GLenum sourceIndexType,
                                                      size_t sourceIndexCount,
                                                      size_t *outIndexCount);

/* Expands GL_UNSIGNED_BYTE indices to Metal-compatible UInt16. */
MGLIndexMetalHandle mglNewUInt16IndexBufferFromUInt8(MGLIndexMetalHandle device,
                                                      const uint8_t *sourceIndexBytes,
                                                      size_t sourceIndexCount);

/* === Buffer source readers ===
 *
 * Locate CPU-readable bytes for a GL element buffer, with Metal buffer
 * fallback.  Used by the draw-element path and the UInt8→UInt16 expansion. */

const uint8_t *mglReadableBufferBytes(Buffer *glBuffer,
                                      MGLIndexMetalHandle metalBuffer,
                                      size_t *outSourceByteCount);

const uint8_t *mglElementIndexSourceBytes(Buffer *glElementBuffer,
                                          MGLIndexMetalHandle metalElementBuffer,
                                          size_t *outSourceByteCount);

const uint8_t *mglElementIndexSourceForDraw(Buffer *glElementBuffer,
                                            MGLIndexMetalHandle metalElementBuffer,
                                            GLenum glIndexType,
                                            size_t indexOffset,
                                            GLsizei indexCount);

bool mglReadBufferBytes(Buffer *glBuffer,
                        MGLIndexMetalHandle metalBuffer,
                        size_t byteOffset,
                        void *dst,
                        size_t byteCount,
                        const char *label);

/* Prepares the element index buffer for a draw call: if the GL index type is
 * GL_UNSIGNED_BYTE, expands to a new UInt16 MTLBuffer and adjusts the offset
 * + MetalIndexType accordingly.  Otherwise returns the original metalElementBuffer
 * unchanged.  Returns nil on expansion failure. */
MGLIndexMetalHandle mglPreparedElementIndexBuffer(MGLIndexMetalHandle device,
                                                   Buffer *glElementBuffer,
                                                   MGLIndexMetalHandle metalElementBuffer,
                                                   GLenum glIndexType,
                                                   size_t *ioIndexBufferOffset,
                                                   uint64_t *outMetalIndexType);

/* Mark the snapshot-pool slot holding buf's current Metal backing as
 * encoded in the current frame, so it is not recycled until that frame's GPU
 * work completes.  Defined in MGLRenderer+Buffer.m. */
void mglNoteBufferEncoded(Buffer *buf);

#ifdef __cplusplus
}
#endif

#endif /* MGL_INDEX_BUFFER_H */
