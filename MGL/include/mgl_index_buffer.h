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
 *   - Metal.framework (id<MTLDevice>, id<MTLBuffer>, MTLIndexType) — under __OBJC__
 */

#ifndef MGL_INDEX_BUFFER_H
#define MGL_INDEX_BUFFER_H

#include "glcorearb.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#include "glm_context.h"
#include "mgl_vertex_format.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    if (!indexBytes || count <= 0 || !outMin || !outMax) {
        return false;
    }

    uint32_t minIndex = UINT32_MAX;
    uint32_t maxIndex = 0u;
    for (GLsizei i = 0; i < count; i++) {
        uint32_t idxValue = mglReadGLIndexValue(indexBytes, indexType, (NSUInteger)i);
        if (primitiveRestartEnabled && idxValue == restartIndex) {
            continue;
        }
        if (idxValue < minIndex) {
            minIndex = idxValue;
        }
        if (idxValue > maxIndex) {
            maxIndex = idxValue;
        }
    }

    if (minIndex > maxIndex) {
        return false;
    }

    *outMin = minIndex;
    *outMax = maxIndex;
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
        switch (indexType) {
            case GL_UNSIGNED_BYTE:
                restartIndex = 0xffu;
                break;
            case GL_UNSIGNED_SHORT:
                restartIndex = 0xffffu;
                break;
            case GL_UNSIGNED_INT:
                restartIndex = 0xffffffffu;
                break;
            default:
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
    NSUInteger quadCount = sourceIndexCount / 4u;
    if (quadCount > (NSUIntegerMax / 6u)) {
        return 0u;
    }
    return quadCount * 6u;
}

/* Computes baseByteOffset + firstElement * indexStride with overflow checks. */
static inline bool mglComputeIndexByteOffset(NSUInteger baseByteOffset,
                                             NSUInteger firstElement,
                                             NSUInteger indexStride,
                                             NSUInteger *outByteOffset)
{
    if (!outByteOffset || indexStride == 0u) {
        return false;
    }
    if (firstElement > (NSUIntegerMax / indexStride)) {
        return false;
    }

    NSUInteger relativeByteOffset = firstElement * indexStride;
    if (baseByteOffset > (NSUIntegerMax - relativeByteOffset)) {
        return false;
    }

    *outByteOffset = baseByteOffset + relativeByteOffset;
    return true;
}

/* Computes the prepared (Metal-side) byte offset for a GL element buffer.
 * GL_UNSIGNED_BYTE indices are expanded to UInt16, so the byte offset is
 * doubled; other index types pass through unchanged. */
static inline bool mglComputePreparedIndexByteOffset(GLenum glIndexType,
                                                    NSUInteger glByteOffset,
                                                    NSUInteger *outPreparedByteOffset)
{
    if (!outPreparedByteOffset) {
        return false;
    }
    if (glIndexType == GL_UNSIGNED_BYTE) {
        if (glByteOffset > (NSUIntegerMax / sizeof(uint16_t))) {
            return false;
        }
        *outPreparedByteOffset = glByteOffset * sizeof(uint16_t);
        return true;
    }

    *outPreparedByteOffset = glByteOffset;
    return true;
}

/* === Metal index buffer builders (extern) ===
 *
 * Each builder allocates a new MTLBuffer with UInt32 indices that expand the
 * GL primitive into Metal-compatible triangle/line lists.  Returns nil on
 * failure.  *outIndexCount receives the number of indices in the new buffer
 * (0 on failure). */

#ifdef __OBJC__

/* Array (non-indexed) variants — generate sequential vertex indices. */
id<MTLBuffer> mglNewTriangleFanArrayIndexBuffer(id<MTLDevice> device,
                                                 NSUInteger vertexCount,
                                                 NSUInteger *outIndexCount);

id<MTLBuffer> mglNewLineLoopArrayIndexBuffer(id<MTLDevice> device,
                                              NSUInteger firstVertex,
                                              NSUInteger vertexCount,
                                              NSUInteger *outIndexCount);

id<MTLBuffer> mglNewTriangleStripArrayIndexBuffer(id<MTLDevice> device,
                                                   NSUInteger vertexCount,
                                                   NSUInteger *outIndexCount);

id<MTLBuffer> mglNewQuadArrayIndexBuffer(id<MTLDevice> device,
                                          NSUInteger vertexCount,
                                          NSUInteger *outIndexCount);

id<MTLBuffer> mglNewQuadArrayLineIndexBuffer(id<MTLDevice> device,
                                              NSUInteger vertexCount,
                                              NSUInteger *outIndexCount);

/* Element (indexed) variants — read source indices and expand. */
id<MTLBuffer> mglNewTriangleFanElementIndexBuffer(id<MTLDevice> device,
                                                   const uint8_t *sourceIndexBytes,
                                                   GLenum sourceIndexType,
                                                   NSUInteger sourceIndexCount,
                                                   NSUInteger *outIndexCount);

id<MTLBuffer> mglNewTriangleStripElementIndexBuffer(id<MTLDevice> device,
                                                     const uint8_t *sourceIndexBytes,
                                                     GLenum sourceIndexType,
                                                     NSUInteger sourceIndexCount,
                                                     NSUInteger *outIndexCount);

id<MTLBuffer> mglNewLineLoopElementIndexBuffer(id<MTLDevice> device,
                                                const uint8_t *sourceIndexBytes,
                                                GLenum sourceIndexType,
                                                NSUInteger sourceIndexCount,
                                                NSUInteger *outIndexCount);

id<MTLBuffer> mglNewQuadElementIndexBuffer(id<MTLDevice> device,
                                            const uint8_t *sourceIndexBytes,
                                            GLenum sourceIndexType,
                                            NSUInteger sourceIndexCount,
                                            NSUInteger *outIndexCount);

id<MTLBuffer> mglNewQuadElementLineIndexBuffer(id<MTLDevice> device,
                                                const uint8_t *sourceIndexBytes,
                                                GLenum sourceIndexType,
                                                NSUInteger sourceIndexCount,
                                                NSUInteger *outIndexCount);

/* Expands GL_UNSIGNED_BYTE indices to Metal-compatible UInt16. */
id<MTLBuffer> mglNewUInt16IndexBufferFromUInt8(id<MTLDevice> device,
                                                const uint8_t *sourceIndexBytes,
                                                NSUInteger sourceIndexCount);

/* === Buffer source readers ===
 *
 * Locate CPU-readable bytes for a GL element buffer, with Metal buffer
 * fallback.  Used by the draw-element path and the UInt8→UInt16 expansion. */

const uint8_t *mglReadableBufferBytes(Buffer *glBuffer,
                                      id<MTLBuffer> metalBuffer,
                                      NSUInteger *outSourceByteCount);

const uint8_t *mglElementIndexSourceBytes(Buffer *glElementBuffer,
                                          id<MTLBuffer> metalElementBuffer,
                                          NSUInteger *outSourceByteCount);

const uint8_t *mglElementIndexSourceForDraw(Buffer *glElementBuffer,
                                            id<MTLBuffer> metalElementBuffer,
                                            GLenum glIndexType,
                                            NSUInteger indexOffset,
                                            GLsizei indexCount);

BOOL mglReadBufferBytes(Buffer *glBuffer,
                        id<MTLBuffer> metalBuffer,
                        NSUInteger byteOffset,
                        void *dst,
                        NSUInteger byteCount,
                        const char *label);

/* Prepares the element index buffer for a draw call: if the GL index type is
 * GL_UNSIGNED_BYTE, expands to a new UInt16 MTLBuffer and adjusts the offset
 * + MetalIndexType accordingly.  Otherwise returns the original metalElementBuffer
 * unchanged.  Returns nil on expansion failure. */
id<MTLBuffer> mglPreparedElementIndexBuffer(id<MTLDevice> device,
                                             Buffer *glElementBuffer,
                                             id<MTLBuffer> metalElementBuffer,
                                             GLenum glIndexType,
                                             NSUInteger *ioIndexBufferOffset,
                                             MTLIndexType *outMetalIndexType);

/* P3: mark the snapshot-pool slot holding buf's current Metal backing as
 * encoded in the current frame, so it is not recycled until that frame's GPU
 * work completes.  Defined in MGLRenderer+Buffer.m. */
void mglNoteBufferEncoded(Buffer *buf);

#endif /* __OBJC__ */

#ifdef __cplusplus
}
#endif

#endif /* MGL_INDEX_BUFFER_H */
