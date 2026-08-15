/*
 * mgl_index_buffer.m
 * MGL
 *
 * Implementation of the Index Buffer Builder Subsystem.
 * See mgl_index_buffer.h for the API contract.
 */

#import "mgl_index_buffer.h"
#import "mgl_safety.h"   /* mglPointerRangeIsReadable — in MGL/src/ */

#include <stdlib.h>
#include <string.h>
#include <os/lock.h>
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"
#include "mgl_render_cpp_objc.h" /* P4: ref typedefs */

static MGLMetalBufferRef mglIndexCreateBuffer(MGLMetalDeviceRef device,
                                          NSUInteger length)
{
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() != NULL) {
        void *buffer = NULL;
        if (mglRenderCppCreateBuffer(
                length, MTLResourceStorageModeShared, NULL, &buffer) == 0 &&
            buffer) {
            return (__bridge_transfer MGLMetalBufferRef)buffer;
        }
    }
    return [device newBufferWithLength:length
                               options:MTLResourceStorageModeShared];
}

/* Allocate the MTLBuffer up front and write indices directly into its
 * .contents pointer, avoiding the calloc + newBufferWithBytes + free
 * triple-step (one fewer malloc/free/memcpy per call) on the per-draw
 * primitive-emulation path (triangle-fan / line-loop / quad expansion). */
static MGLMetalBufferRef mglNewUninitializedIndexBuffer(MGLMetalDeviceRef device,
                                                    NSUInteger byteCount,
                                                    void **outContents)
{
    if (!device || byteCount == 0u || !outContents) {
        return nil;
    }
    MGLMetalBufferRef buffer = mglIndexCreateBuffer(device, byteCount);
    if (!buffer) {
        return nil;
    }
    *outContents = buffer.contents;
    return buffer;
}

/* Persistent index buffer cache for array-variant primitive emulation.
 * Array-variant indices are 0-based, monotonic, and prefix-compatible:
 * a buffer built for a larger vertexCount contains the exact prefix for
 * any smaller vertexCount.  Cache the largest buffer per primitive type;
 * smaller requests reuse it with a reduced indexCount — no allocation
 * or O(N) fill on cache hit.  MC's GUI/font/particle paths issue hundreds
 * of GL_QUADS draws per frame, all hitting this cache after warmup. */
static os_unfair_lock s_arrayIndexCacheLock = OS_UNFAIR_LOCK_INIT;
static MGLMetalBufferRef s_cachedFanArrayBuffer = nil;
static NSUInteger     s_cachedFanArrayVertexCount = 0;
static MGLMetalBufferRef s_cachedStripArrayBuffer = nil;
static NSUInteger     s_cachedStripArrayVertexCount = 0;
static MGLMetalBufferRef s_cachedQuadArrayBuffer = nil;
static NSUInteger     s_cachedQuadArrayVertexCount = 0;
static MGLMetalBufferRef s_cachedQuadLineArrayBuffer = nil;
static NSUInteger     s_cachedQuadLineArrayVertexCount = 0;

/* LINE_LOOP array-variant indices are firstVertex-relative (firstVertex+i),
 * so they are NOT prefix-compatible across different (firstVertex,count)
 * ranges.  Repeat draws of the same loop geometry (same GL_LINE_LOOP mesh
 * drawn every frame) hit a small fixed-slot cache keyed on (firstVertex,
 * vertexCount); missing keys allocate a fresh buffer. */
#define MGL_LINE_LOOP_ARRAY_CACHE_SLOTS 4u
static struct {
    NSUInteger   firstVertex;
    NSUInteger   vertexCount;
    MGLMetalBufferRef buffer;
} s_lineLoopArrayCache[MGL_LINE_LOOP_ARRAY_CACHE_SLOTS];

MGLMetalBufferRef mglNewTriangleFanArrayIndexBuffer(MGLMetalDeviceRef device,
                                                NSUInteger vertexCount,
                                                NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || vertexCount < 3u) {
        return nil;
    }

    NSUInteger triangleCount = vertexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    if (s_cachedFanArrayBuffer && vertexCount <= s_cachedFanArrayVertexCount) {
        MGLMetalBufferRef cached = s_cachedFanArrayBuffer;
        os_unfair_lock_unlock(&s_arrayIndexCacheLock);
        if (outIndexCount) {
            *outIndexCount = indexCount;
        }
        return cached;
    }
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    uint32_t *expanded = NULL;
    uint64_t indexCount64 = 0u;
    if (mglRenderCppExpandTriangleFanArrayIndices(
            (uint32_t)vertexCount, &expanded, &indexCount64) != 0) {
        return nil;
    }
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, (NSUInteger)indexCount64 * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)indexCount64 * sizeof(uint32_t));
    free(expanded);

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    s_cachedFanArrayBuffer = buffer;
    s_cachedFanArrayVertexCount = vertexCount;
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    if (outIndexCount) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewLineLoopArrayIndexBuffer(MGLMetalDeviceRef device,
                                             NSUInteger firstVertex,
                                             NSUInteger vertexCount,
                                             NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || vertexCount < 2u) {
        return nil;
    }
    if (vertexCount > (NSUIntegerMax - 1u) ||
        firstVertex > ((NSUInteger)UINT32_MAX) ||
        vertexCount > (((NSUInteger)UINT32_MAX + 1u) - firstVertex)) {
        return nil;
    }

    NSUInteger indexCount = vertexCount + 1u;

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    for (NSUInteger slot = 0u; slot < MGL_LINE_LOOP_ARRAY_CACHE_SLOTS; slot++) {
        MGLMetalBufferRef cached = s_lineLoopArrayCache[slot].buffer;
        if (cached &&
            s_lineLoopArrayCache[slot].firstVertex == firstVertex &&
            s_lineLoopArrayCache[slot].vertexCount == vertexCount) {
            os_unfair_lock_unlock(&s_arrayIndexCacheLock);
            if (outIndexCount) {
                *outIndexCount = indexCount;
            }
            return cached;
        }
    }
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    uint32_t *expanded = NULL;
    uint64_t indexCount64 = 0u;
    if (mglRenderCppExpandLineLoopArrayIndices(
            (uint32_t)firstVertex, (uint32_t)vertexCount,
            &expanded, &indexCount64) != 0) {
        return nil;
    }
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, (NSUInteger)indexCount64 * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)indexCount64 * sizeof(uint32_t));
    free(expanded);

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    for (NSUInteger slot = 0u; slot < MGL_LINE_LOOP_ARRAY_CACHE_SLOTS; slot++) {
        if (s_lineLoopArrayCache[slot].buffer == nil) {
            s_lineLoopArrayCache[slot].firstVertex = firstVertex;
            s_lineLoopArrayCache[slot].vertexCount = vertexCount;
            s_lineLoopArrayCache[slot].buffer = buffer;
            break;
        }
    }
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    if (outIndexCount) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewTriangleStripArrayIndexBuffer(MGLMetalDeviceRef device,
                                                  NSUInteger vertexCount,
                                                  NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || vertexCount < 3u) {
        return nil;
    }

    NSUInteger triangleCount = vertexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    if (s_cachedStripArrayBuffer && vertexCount <= s_cachedStripArrayVertexCount) {
        MGLMetalBufferRef cached = s_cachedStripArrayBuffer;
        os_unfair_lock_unlock(&s_arrayIndexCacheLock);
        if (outIndexCount) {
            *outIndexCount = indexCount;
        }
        return cached;
    }
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(device,
                                                          indexCount * sizeof(uint32_t),
                                                          (void **)&indices);
    if (!buffer) {
        return nil;
    }

    uint32_t *expanded = NULL;
    uint64_t indexCount64 = 0u;
    if (mglRenderCppExpandTriangleStripArrayIndices(
            (uint32_t)vertexCount, &expanded, &indexCount64) != 0) {
        return nil;
    }
    /* NOTE: indices was already allocated by the strip builder above. */
    memcpy(indices, expanded, (size_t)indexCount64 * sizeof(uint32_t));
    free(expanded);

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    s_cachedStripArrayBuffer = buffer;
    s_cachedStripArrayVertexCount = vertexCount;
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    if (outIndexCount) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewTriangleFanElementIndexBuffer(MGLMetalDeviceRef device,
                                                  const uint8_t *sourceIndexBytes,
                                                  GLenum sourceIndexType,
                                                  NSUInteger sourceIndexCount,
                                                  NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || !sourceIndexBytes || sourceIndexCount < 3u) {
        return nil;
    }

    /* P4.5 (item 1141/887): 三角形扇形元素展开（中心 + 线性子索引三元组）
     * 在 C++（mglRenderCppExpandTriangleFanIndices，两门共用）。 */
    const uint32_t elemWidth = sourceIndexType == GL_UNSIGNED_BYTE ? 1u
        : sourceIndexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t *expanded = NULL;
    uint64_t indexCount = 0u;
    if (mglRenderCppExpandTriangleFanIndices(
            sourceIndexBytes, elemWidth, (uint32_t)sourceIndexCount,
            &expanded, &indexCount) != 0 || indexCount == 0u) {
        if (expanded) free(expanded);
        return nil;
    }
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, (NSUInteger)indexCount * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)indexCount * sizeof(uint32_t));
    free(expanded);
    if (outIndexCount) {
        *outIndexCount = (NSUInteger)indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewTriangleStripElementIndexBuffer(MGLMetalDeviceRef device,
                                                    const uint8_t *sourceIndexBytes,
                                                    GLenum sourceIndexType,
                                                    NSUInteger sourceIndexCount,
                                                    NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || !sourceIndexBytes || sourceIndexCount < 3u) {
        return nil;
    }

    const uint32_t elemWidth = sourceIndexType == GL_UNSIGNED_BYTE ? 1u
        : sourceIndexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t *expanded = NULL;
    uint64_t indexCount = 0u;
    if (mglRenderCppExpandTriangleStripIndices(
            sourceIndexBytes, elemWidth, (uint32_t)sourceIndexCount,
            &expanded, &indexCount) != 0 || indexCount == 0u) {
        if (expanded) free(expanded);
        return nil;
    }
    NSUInteger byteCount = (NSUInteger)indexCount * sizeof(uint32_t);
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, byteCount, (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)byteCount);
    free(expanded);
    if (outIndexCount) {
        *outIndexCount = (NSUInteger)indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewLineLoopElementIndexBuffer(MGLMetalDeviceRef device,
                                               const uint8_t *sourceIndexBytes,
                                               GLenum sourceIndexType,
                                               NSUInteger sourceIndexCount,
                                               NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || !sourceIndexBytes || sourceIndexCount < 2u) {
        return nil;
    }
    const uint32_t elemWidth = sourceIndexType == GL_UNSIGNED_BYTE ? 1u
        : sourceIndexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t *expanded = NULL;
    uint64_t indexCount = 0u;
    /* P4.5 (item 1141/887): 条带/线环元素展开在 C++
     * （mglRenderCppExpandTriangleStripIndices / ExpandLineLoopIndices）。 */
    if (mglRenderCppExpandLineLoopIndices(
            sourceIndexBytes, elemWidth, (uint32_t)sourceIndexCount,
            &expanded, &indexCount) != 0 || indexCount == 0u) {
        if (expanded) free(expanded);
        return nil;
    }
    NSUInteger loopIndexCount = (NSUInteger)indexCount;
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, loopIndexCount * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)loopIndexCount * sizeof(uint32_t));
    free(expanded);
    if (outIndexCount) {
        *outIndexCount = loopIndexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewQuadArrayIndexBuffer(MGLMetalDeviceRef device,
                                         NSUInteger vertexCount,
                                         NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    NSUInteger indexCount = mglQuadTriangleIndexCount(vertexCount);
    if (!device || indexCount == 0u) {
        return nil;
    }
    if (indexCount > (NSUIntegerMax / sizeof(uint32_t))) {
        return nil;
    }

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    if (s_cachedQuadArrayBuffer && vertexCount <= s_cachedQuadArrayVertexCount) {
        MGLMetalBufferRef cached = s_cachedQuadArrayBuffer;
        os_unfair_lock_unlock(&s_arrayIndexCacheLock);
        if (outIndexCount) {
            *outIndexCount = indexCount;
        }
        return cached;
    }
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    const NSUInteger quadCount = vertexCount / 4u;
    uint32_t *expanded = NULL;
    uint64_t indexCount64 = 0u;
    if (mglRenderCppExpandQuadArrayIndices(
            (uint32_t)quadCount, &expanded, &indexCount64) != 0) {
        return nil;
    }
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, (NSUInteger)indexCount64 * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)indexCount64 * sizeof(uint32_t));
    free(expanded);

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    s_cachedQuadArrayBuffer = buffer;
    s_cachedQuadArrayVertexCount = vertexCount;
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    if (outIndexCount) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewQuadElementIndexBuffer(MGLMetalDeviceRef device,
                                           const uint8_t *sourceIndexBytes,
                                           GLenum sourceIndexType,
                                           NSUInteger sourceIndexCount,
                                           NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    NSUInteger indexCount = mglQuadTriangleIndexCount(sourceIndexCount);
    if (!device || !sourceIndexBytes || indexCount == 0u) {
        return nil;
    }
    if (indexCount > (NSUIntegerMax / sizeof(uint32_t))) {
        return nil;
    }

    const NSUInteger quadCount = sourceIndexCount / 4u;
    const uint32_t elemWidth = sourceIndexType == GL_UNSIGNED_BYTE ? 1u
        : sourceIndexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t *expanded = NULL;
    uint64_t indexCount64 = 0u;
    if (mglRenderCppExpandQuadElementIndices(
            sourceIndexBytes, elemWidth, (uint32_t)quadCount,
            &expanded, &indexCount64) != 0) {
        return nil;
    }
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, (NSUInteger)indexCount64 * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)indexCount64 * sizeof(uint32_t));
    free(expanded);

    if (outIndexCount) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewQuadArrayLineIndexBuffer(MGLMetalDeviceRef device,
                                             NSUInteger vertexCount,
                                             NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    NSUInteger quadCount = vertexCount / 4u;
    if (!device || quadCount == 0u || quadCount > (NSUIntegerMax / (8u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = quadCount * 8u;

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    if (s_cachedQuadLineArrayBuffer && vertexCount <= s_cachedQuadLineArrayVertexCount) {
        MGLMetalBufferRef cached = s_cachedQuadLineArrayBuffer;
        os_unfair_lock_unlock(&s_arrayIndexCacheLock);
        if (outIndexCount) {
            *outIndexCount = indexCount;
        }
        return cached;
    }
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    uint32_t *expanded = NULL;
    uint64_t indexCount64 = 0u;
    if (mglRenderCppExpandQuadArrayLineIndices(
            (uint32_t)quadCount, &expanded, &indexCount64) != 0) {
        return nil;
    }
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, (NSUInteger)indexCount64 * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)indexCount64 * sizeof(uint32_t));
    free(expanded);

    os_unfair_lock_lock(&s_arrayIndexCacheLock);
    s_cachedQuadLineArrayBuffer = buffer;
    s_cachedQuadLineArrayVertexCount = vertexCount;
    os_unfair_lock_unlock(&s_arrayIndexCacheLock);

    if (outIndexCount) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

MGLMetalBufferRef mglNewQuadElementLineIndexBuffer(MGLMetalDeviceRef device,
                                               const uint8_t *sourceIndexBytes,
                                               GLenum sourceIndexType,
                                               NSUInteger sourceIndexCount,
                                               NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    NSUInteger quadCount = sourceIndexCount / 4u;
    if (!device || !sourceIndexBytes || quadCount == 0u || quadCount > (NSUIntegerMax / (8u * sizeof(uint32_t)))) {
        return nil;
    }

    const uint32_t elemWidth = sourceIndexType == GL_UNSIGNED_BYTE ? 1u
        : sourceIndexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t *expanded = NULL;
    uint64_t indexCount64 = 0u;
    if (mglRenderCppExpandQuadElementLineIndices(
            sourceIndexBytes, elemWidth, (uint32_t)quadCount,
            &expanded, &indexCount64) != 0) {
        return nil;
    }
    uint32_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(
        device, (NSUInteger)indexCount64 * sizeof(uint32_t), (void **)&indices);
    if (!buffer) {
        free(expanded);
        return nil;
    }
    memcpy(indices, expanded, (size_t)indexCount64 * sizeof(uint32_t));
    free(expanded);

    if (outIndexCount) {
        *outIndexCount = (NSUInteger)indexCount64;
    }

    return buffer;
}

MGLMetalBufferRef mglNewUInt16IndexBufferFromUInt8(MGLMetalDeviceRef device,
                                               const uint8_t *sourceIndexBytes,
                                               NSUInteger sourceIndexCount)
{
    if (!device || !sourceIndexBytes) {
        return nil;
    }
    if (sourceIndexCount == 0u) {
        return nil;
    }
    if (sourceIndexCount > (NSUIntegerMax / sizeof(uint16_t))) {
        return nil;
    }

    uint16_t *indices = NULL;
    MGLMetalBufferRef buffer = mglNewUninitializedIndexBuffer(device,
                                                          sourceIndexCount * sizeof(uint16_t),
                                                          (void **)&indices);
    if (!buffer) {
        return nil;
    }
    for (NSUInteger i = 0; i < sourceIndexCount; i++) {
        indices[i] = (uint16_t)sourceIndexBytes[i];
    }

    return buffer;
}

const uint8_t *mglReadableBufferBytes(Buffer *glBuffer,
                                      MGLMetalBufferRef metalBuffer,
                                      NSUInteger *outSourceByteCount)
{
    if (outSourceByteCount) {
        *outSourceByteCount = 0u;
    }

    if (glBuffer && glBuffer->data.buffer_data) {
        NSUInteger glByteCount = 0u;
        if (glBuffer->data.buffer_size > 0u) {
            glByteCount = (NSUInteger)glBuffer->data.buffer_size;
        } else if (glBuffer->size > 0) {
            glByteCount = (NSUInteger)glBuffer->size;
        }
        const void *glBytes = (const void *)(uintptr_t)glBuffer->data.buffer_data;
        if (glByteCount > 0u &&
            mglPointerRangeIsReadable(glBytes, glByteCount)) {
            if (outSourceByteCount) {
                *outSourceByteCount = glByteCount;
            }
            return (const uint8_t *)glBytes;
        }
    }

    if (metalBuffer && metalBuffer.contents && metalBuffer.length > 0u) {
        if (outSourceByteCount) {
            *outSourceByteCount = metalBuffer.length;
        }
        return (const uint8_t *)metalBuffer.contents;
    }

    return NULL;
}

const uint8_t *mglElementIndexSourceBytes(Buffer *glElementBuffer,
                                          MGLMetalBufferRef metalElementBuffer,
                                          NSUInteger *outSourceByteCount)
{
    return mglReadableBufferBytes(glElementBuffer,
                                  metalElementBuffer,
                                  outSourceByteCount);
}

const uint8_t *mglElementIndexSourceForDraw(Buffer *glElementBuffer,
                                            MGLMetalBufferRef metalElementBuffer,
                                            GLenum glIndexType,
                                            NSUInteger indexOffset,
                                            GLsizei indexCount)
{
    NSUInteger sourceByteCount = 0u;
    const uint8_t *sourceBytes = mglElementIndexSourceBytes(glElementBuffer,
                                                            metalElementBuffer,
                                                            &sourceByteCount);
    NSUInteger indexStride = mglGLIndexElementSize(glIndexType);
    if (!sourceBytes || sourceByteCount == 0u || indexStride == 0u || indexCount <= 0) {
        return NULL;
    }
    if ((NSUInteger)indexCount > (NSUIntegerMax / indexStride)) {
        return NULL;
    }

    NSUInteger neededBytes = (NSUInteger)indexCount * indexStride;
    if (indexOffset > sourceByteCount || (sourceByteCount - indexOffset) < neededBytes) {
        return NULL;
    }

    return sourceBytes + indexOffset;
}

BOOL mglReadBufferBytes(Buffer *glBuffer,
                        MGLMetalBufferRef metalBuffer,
                        NSUInteger byteOffset,
                        void *dst,
                        NSUInteger byteCount,
                        const char *label)
{
    if (!dst || byteCount == 0u) {
        return NO;
    }

    NSUInteger sourceByteCount = 0u;
    const uint8_t *sourceBytes = mglReadableBufferBytes(glBuffer,
                                                        metalBuffer,
                                                        &sourceByteCount);
    if (!sourceBytes || byteOffset > sourceByteCount ||
        (sourceByteCount - byteOffset) < byteCount) {
        NSLog(@"MGL WARNING: %s CPU read unavailable buffer=%u offset=%lu bytes=%lu source=%p sourceBytes=%lu mtl=%p",
              label ? label : "buffer",
              glBuffer ? glBuffer->name : 0u,
              (unsigned long)byteOffset,
              (unsigned long)byteCount,
              sourceBytes,
              (unsigned long)sourceByteCount,
              metalBuffer);
        return NO;
    }

    memcpy(dst, sourceBytes + byteOffset, byteCount);
    return YES;
}

MGLMetalBufferRef mglPreparedElementIndexBuffer(MGLMetalDeviceRef device,
                                            Buffer *glElementBuffer,
                                            MGLMetalBufferRef metalElementBuffer,
                                            GLenum glIndexType,
                                            NSUInteger *ioIndexBufferOffset,
                                            MTLIndexType *outMetalIndexType)
{
    if (outMetalIndexType) {
        switch (glIndexType) {
            case GL_UNSIGNED_BYTE:
            case GL_UNSIGNED_SHORT:
                *outMetalIndexType = MTLIndexTypeUInt16;
                break;
            case GL_UNSIGNED_INT:
                *outMetalIndexType = MTLIndexTypeUInt32;
                break;
            default:
                *outMetalIndexType = (MTLIndexType)0xFFFFFFFF;
                break;
        }
    }
    if (glIndexType != GL_UNSIGNED_BYTE) {
        /* The EBO's own Metal backing is encoded directly by the caller: pin
         * its snapshot-pool slot for the current frame (P3). */
        mglNoteBufferEncoded(glElementBuffer);
        return metalElementBuffer;
    }

    NSUInteger sourceOffset = ioIndexBufferOffset ? *ioIndexBufferOffset : 0u;
    NSUInteger sourceByteCount = 0u;
    const uint8_t *sourceBytes = mglElementIndexSourceBytes(glElementBuffer,
                                                            metalElementBuffer,
                                                            &sourceByteCount);

    if (!sourceBytes || sourceByteCount == 0u) {
        NSLog(@"MGL WARNING: unable to expand GL_UNSIGNED_BYTE element buffer gl=%u source=%p bytes=%lu",
              glElementBuffer ? glElementBuffer->name : 0u,
              sourceBytes,
              (unsigned long)sourceByteCount);
        return nil;
    }

    /* determine cache eligibility.
     * Cache is safe only when:
     * - glElementBuffer has CPU-side backing (buffer_data): source bytes are
     *   GL-tracked and last_write_src_hash reflects their contents.
     * - Buffer is not persistent-mapped: app cannot modify contents without
     *   a GL call that updates last_write_src_hash. */
    BOOL cache_eligible = (glElementBuffer != NULL &&
                           glElementBuffer->data.buffer_data != 0 &&
                           !(glElementBuffer->storage_flags & GL_MAP_PERSISTENT_BIT));

    /* check cached UInt16 expansion */
    if (cache_eligible &&
        glElementBuffer->mtl_uint16_expanded_data != NULL &&
        glElementBuffer->mtl_uint16_expanded_src_hash != 0ull &&
        glElementBuffer->mtl_uint16_expanded_src_hash == glElementBuffer->last_write_src_hash &&
        glElementBuffer->mtl_uint16_expanded_byte_count == sourceByteCount) {
        /* Cache hit — return the cached expanded buffer */
        MGLMetalBufferRef cached = (__bridge MGLMetalBufferRef)glElementBuffer->mtl_uint16_expanded_data;
        if (ioIndexBufferOffset) {
            if (sourceOffset > (NSUIntegerMax / sizeof(uint16_t))) {
                return nil;
            }
            *ioIndexBufferOffset = sourceOffset * sizeof(uint16_t);
        }
        if (outMetalIndexType) {
            *outMetalIndexType = MTLIndexTypeUInt16;
        }
        return cached;
    }

    MGLMetalBufferRef expanded = mglNewUInt16IndexBufferFromUInt8(device, sourceBytes, sourceByteCount);
    if (!expanded) {
        NSLog(@"MGL WARNING: failed to allocate expanded UInt16 element buffer for GL_UNSIGNED_BYTE gl=%u bytes=%lu",
              glElementBuffer ? glElementBuffer->name : 0u,
              (unsigned long)sourceByteCount);
        return nil;
    }

    /* store expanded buffer in cache for subsequent draws */
    if (cache_eligible) {
        if (glElementBuffer->mtl_uint16_expanded_data) {
            CFRelease(glElementBuffer->mtl_uint16_expanded_data);
        }
        CFRetain((__bridge CFTypeRef)expanded);
        glElementBuffer->mtl_uint16_expanded_data = (__bridge void *)expanded;
        glElementBuffer->mtl_uint16_expanded_src_hash = glElementBuffer->last_write_src_hash;
        glElementBuffer->mtl_uint16_expanded_byte_count = sourceByteCount;
    }

    if (ioIndexBufferOffset) {
        if (sourceOffset > (NSUIntegerMax / sizeof(uint16_t))) {
            return nil;
        }
        *ioIndexBufferOffset = sourceOffset * sizeof(uint16_t);
    }
    if (outMetalIndexType) {
        *outMetalIndexType = MTLIndexTypeUInt16;
    }
    return expanded;
}
