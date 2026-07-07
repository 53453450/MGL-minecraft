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

id<MTLBuffer> mglNewTriangleFanArrayIndexBuffer(id<MTLDevice> device,
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
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = 0u;
        indices[(tri * 3u) + 1u] = (uint32_t)(tri + 1u);
        indices[(tri * 3u) + 2u] = (uint32_t)(tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewLineLoopArrayIndexBuffer(id<MTLDevice> device,
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
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger i = 0; i < vertexCount; i++) {
        if (i > UINT32_MAX) {
            free(indices);
            return nil;
        }
        indices[i] = (uint32_t)(firstVertex + i);
    }
    indices[vertexCount] = (uint32_t)firstVertex;

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewTriangleStripArrayIndexBuffer(id<MTLDevice> device,
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
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = (uint32_t)tri;
        indices[(tri * 3u) + 1u] = (uint32_t)(tri + 1u);
        indices[(tri * 3u) + 2u] = (uint32_t)(tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewTriangleFanElementIndexBuffer(id<MTLDevice> device,
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

    NSUInteger triangleCount = sourceIndexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    uint32_t center = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, 0u);
    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = center;
        indices[(tri * 3u) + 1u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 1u);
        indices[(tri * 3u) + 2u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewTriangleStripElementIndexBuffer(id<MTLDevice> device,
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

    NSUInteger triangleCount = sourceIndexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri);
        indices[(tri * 3u) + 1u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 1u);
        indices[(tri * 3u) + 2u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewLineLoopElementIndexBuffer(id<MTLDevice> device,
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
    if (sourceIndexCount > (NSUIntegerMax - 1u)) {
        return nil;
    }

    NSUInteger indexCount = sourceIndexCount + 1u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger i = 0; i < sourceIndexCount; i++) {
        indices[i] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, i);
    }
    indices[sourceIndexCount] = indices[0];

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewQuadArrayIndexBuffer(id<MTLDevice> device,
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

    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    NSUInteger quadCount = vertexCount / 4u;
    for (NSUInteger quad = 0; quad < quadCount; quad++) {
        NSUInteger src = quad * 4u;
        NSUInteger dst = quad * 6u;
        if ((src + 3u) > UINT32_MAX) {
            free(indices);
            return nil;
        }
        indices[dst + 0u] = (uint32_t)(src + 0u);
        indices[dst + 1u] = (uint32_t)(src + 1u);
        indices[dst + 2u] = (uint32_t)(src + 2u);
        indices[dst + 3u] = (uint32_t)(src + 0u);
        indices[dst + 4u] = (uint32_t)(src + 2u);
        indices[dst + 5u] = (uint32_t)(src + 3u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewQuadElementIndexBuffer(id<MTLDevice> device,
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

    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    NSUInteger quadCount = sourceIndexCount / 4u;
    for (NSUInteger quad = 0; quad < quadCount; quad++) {
        NSUInteger src = quad * 4u;
        NSUInteger dst = quad * 6u;
        uint32_t i0 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 0u);
        uint32_t i1 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 1u);
        uint32_t i2 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 2u);
        uint32_t i3 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 3u);
        indices[dst + 0u] = i0;
        indices[dst + 1u] = i1;
        indices[dst + 2u] = i2;
        indices[dst + 3u] = i0;
        indices[dst + 4u] = i2;
        indices[dst + 5u] = i3;
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewQuadArrayLineIndexBuffer(id<MTLDevice> device,
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
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger quad = 0; quad < quadCount; quad++) {
        NSUInteger src = quad * 4u;
        NSUInteger dst = quad * 8u;
        if ((src + 3u) > UINT32_MAX) {
            free(indices);
            return nil;
        }
        indices[dst + 0u] = (uint32_t)(src + 0u);
        indices[dst + 1u] = (uint32_t)(src + 1u);
        indices[dst + 2u] = (uint32_t)(src + 1u);
        indices[dst + 3u] = (uint32_t)(src + 2u);
        indices[dst + 4u] = (uint32_t)(src + 2u);
        indices[dst + 5u] = (uint32_t)(src + 3u);
        indices[dst + 6u] = (uint32_t)(src + 3u);
        indices[dst + 7u] = (uint32_t)(src + 0u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewQuadElementLineIndexBuffer(id<MTLDevice> device,
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

    NSUInteger indexCount = quadCount * 8u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger quad = 0; quad < quadCount; quad++) {
        NSUInteger src = quad * 4u;
        NSUInteger dst = quad * 8u;
        uint32_t i0 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 0u);
        uint32_t i1 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 1u);
        uint32_t i2 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 2u);
        uint32_t i3 = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, src + 3u);
        indices[dst + 0u] = i0;
        indices[dst + 1u] = i1;
        indices[dst + 2u] = i1;
        indices[dst + 3u] = i2;
        indices[dst + 4u] = i2;
        indices[dst + 5u] = i3;
        indices[dst + 6u] = i3;
        indices[dst + 7u] = i0;
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

id<MTLBuffer> mglNewUInt16IndexBufferFromUInt8(id<MTLDevice> device,
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

    uint16_t *indices = (uint16_t *)calloc(sourceIndexCount, sizeof(uint16_t));
    if (!indices) {
        return nil;
    }
    for (NSUInteger i = 0; i < sourceIndexCount; i++) {
        indices[i] = (uint16_t)sourceIndexBytes[i];
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(sourceIndexCount * sizeof(uint16_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);
    return buffer;
}

const uint8_t *mglReadableBufferBytes(Buffer *glBuffer,
                                      id<MTLBuffer> metalBuffer,
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
                                          id<MTLBuffer> metalElementBuffer,
                                          NSUInteger *outSourceByteCount)
{
    return mglReadableBufferBytes(glElementBuffer,
                                  metalElementBuffer,
                                  outSourceByteCount);
}

const uint8_t *mglElementIndexSourceForDraw(Buffer *glElementBuffer,
                                            id<MTLBuffer> metalElementBuffer,
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
                        id<MTLBuffer> metalBuffer,
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

id<MTLBuffer> mglPreparedElementIndexBuffer(id<MTLDevice> device,
                                            Buffer *glElementBuffer,
                                            id<MTLBuffer> metalElementBuffer,
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

    id<MTLBuffer> expanded = mglNewUInt16IndexBufferFromUInt8(device, sourceBytes, sourceByteCount);
    if (!expanded) {
        NSLog(@"MGL WARNING: failed to allocate expanded UInt16 element buffer for GL_UNSIGNED_BYTE gl=%u bytes=%lu",
              glElementBuffer ? glElementBuffer->name : 0u,
              (unsigned long)sourceByteCount);
        return nil;
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
