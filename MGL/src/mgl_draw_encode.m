/*
 * mgl_draw_encode.m
 * MGL
 *
 * Implementation of the Draw Encode Subsystem.
 * See mgl_draw_encode.h for the API contract.
 */

#include "mgl_draw_encode.h"

#include <stdlib.h>
#include <stdio.h>
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"

static void mglDrawEncodePrimitives(void *renderEncoderOwner,
                                    uint32_t primitiveType,
                                    size_t vertexStart,
                                    size_t vertexCount,
                                    size_t instanceCount,
                                    size_t baseInstance)
{
    const MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    if (!renderEncoderOwner) return;
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static void mglDrawEncodeIndexed(void *renderEncoderOwner,
                                 uint32_t primitiveType,
                                 size_t indexCount,
                                 uint32_t indexType,
                                 MGLDrawMetalHandle indexBuffer,
                                 size_t indexBufferOffset,
                                 size_t instanceCount,
                                 int64_t baseVertex,
                                 size_t baseInstance)
{
    const MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)indexType,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        };
    if (!renderEncoderOwner) return;
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static bool mglEncodeArrayLineLoopTarget(void *renderEncoderOwner,
                            GLMContext drawCtx,
                            MGLDrawMetalHandle device,
                            GLsizei count,
                            GLint firstVertex,
                            size_t instanceCount,
                            size_t baseInstance,
                            const char *label)
{
    if (count < 2) {
        return true;
    }
    if (firstVertex < 0) {
        fprintf(stderr, "MGL WARNING: %s line loop array emulation invalid first=%d",
              label ? label : "draw",
              (int)firstVertex);
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_VALUE);
        }
        return false;
    }

    size_t loopIndexCount = 0u;
    MGLDrawMetalHandle loopIndexBuffer = mglNewLineLoopArrayIndexBuffer(device,
                                                                   (size_t)firstVertex,
                                                                   (size_t)count,
                                                                   &loopIndexCount);
    if (!loopIndexBuffer || loopIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s line loop array emulation failed count=%d first=%d",
              label ? label : "draw",
              (int)count,
              (int)firstVertex);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
                         MGL_DRAW_PRIMITIVE_LINE_STRIP, loopIndexCount,
                         MGL_DRAW_INDEX_UINT32, loopIndexBuffer, 0,
                         instanceCount, 0, baseInstance);
    return true;
}

static bool mglEncodeArrayTriangleFanTarget(void *renderEncoderOwner,
                                      MGLDrawMetalHandle device,
                                      GLsizei count,
                                      GLint baseVertex,
                                      size_t instanceCount,
                                      size_t baseInstance,
                                      const char *label)
{
    if (count < 3) {
        return true;
    }

    size_t fanIndexCount = 0u;
    MGLDrawMetalHandle fanIndexBuffer = mglNewTriangleFanArrayIndexBuffer(device,
                                                                     (size_t)count,
                                                                     &fanIndexCount);
    if (!fanIndexBuffer || fanIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s triangle fan array emulation failed count=%d baseVertex=%d",
              label ? label : "draw",
              (int)count,
              (int)baseVertex);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
                         MGL_DRAW_PRIMITIVE_TRIANGLE, fanIndexCount,
                         MGL_DRAW_INDEX_UINT32, fanIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return true;
}

static bool mglEncodeElementLineLoopTarget(void *renderEncoderOwner,
                                     MGLDrawMetalHandle device,
                                     Buffer *glElementBuffer,
                                     MGLDrawMetalHandle metalElementBuffer,
                                     GLenum glIndexType,
                                     size_t indexOffset,
                                     GLsizei count,
                                     size_t instanceCount,
                                     int64_t baseVertex,
                                     size_t baseInstance,
                                     const char *label)
{
    if (count < 2) {
        return true;
    }

    const uint8_t *loopSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                             metalElementBuffer,
                                                             glIndexType,
                                                             indexOffset,
                                                             count);
    size_t loopIndexCount = 0u;
    MGLDrawMetalHandle loopIndexBuffer = mglNewLineLoopElementIndexBuffer(device,
                                                                     loopSource,
                                                                     glIndexType,
                                                                     (size_t)count,
                                                                     &loopIndexCount);
    if (!loopIndexBuffer || loopIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s line loop element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              loopSource);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
                         MGL_DRAW_PRIMITIVE_LINE_STRIP, loopIndexCount,
                         MGL_DRAW_INDEX_UINT32, loopIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return true;
}

static bool mglEncodeElementTriangleFanTarget(void *renderEncoderOwner,
                                        MGLDrawMetalHandle device,
                                        Buffer *glElementBuffer,
                                        MGLDrawMetalHandle metalElementBuffer,
                                        GLenum glIndexType,
                                        size_t indexOffset,
                                        GLsizei count,
                                        size_t instanceCount,
                                        int64_t baseVertex,
                                        size_t baseInstance,
                                        const char *label)
{
    if (count < 3) {
        return true;
    }

    const uint8_t *fanSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                            metalElementBuffer,
                                                            glIndexType,
                                                            indexOffset,
                                                            count);
    size_t fanIndexCount = 0u;
    MGLDrawMetalHandle fanIndexBuffer = mglNewTriangleFanElementIndexBuffer(device,
                                                                       fanSource,
                                                                       glIndexType,
                                                                       (size_t)count,
                                                                       &fanIndexCount);
    if (!fanIndexBuffer || fanIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s triangle fan element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              fanSource);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
                         MGL_DRAW_PRIMITIVE_TRIANGLE, fanIndexCount,
                         MGL_DRAW_INDEX_UINT32, fanIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return true;
}

static bool mglEncodeArrayQuadsTarget(void *renderEncoderOwner,
                                MGLDrawMetalHandle device,
                                GLsizei count,
                                GLint baseVertex,
                                size_t instanceCount,
                                size_t baseInstance,
                                bool lineMode,
                                const char *label)
{
    if (count < 4) {
        return true;
    }

    size_t quadIndexCount = 0u;
    MGLDrawMetalHandle quadIndexBuffer = lineMode
        ? mglNewQuadArrayLineIndexBuffer(device, (size_t)count, &quadIndexCount)
        : mglNewQuadArrayIndexBuffer(device, (size_t)count, &quadIndexCount);
    if (!quadIndexBuffer || quadIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s quad array emulation failed count=%d baseVertex=%d",
              label ? label : "draw",
              (int)count,
              (int)baseVertex);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
        lineMode ? MGL_DRAW_PRIMITIVE_LINE : MGL_DRAW_PRIMITIVE_TRIANGLE,
        quadIndexCount, MGL_DRAW_INDEX_UINT32, quadIndexBuffer, 0,
        instanceCount, baseVertex, baseInstance);
    return true;
}

static bool mglEncodeElementQuadsTarget(void *renderEncoderOwner,
                                  MGLDrawMetalHandle device,
                                  Buffer *glElementBuffer,
                                  MGLDrawMetalHandle metalElementBuffer,
                                  GLenum glIndexType,
                                  size_t indexOffset,
                                  GLsizei count,
                                  size_t instanceCount,
                                  int64_t baseVertex,
                                  size_t baseInstance,
                                  bool lineMode,
                                  const char *label)
{
    if (count < 4) {
        return true;
    }

    const uint8_t *quadSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                             metalElementBuffer,
                                                             glIndexType,
                                                             indexOffset,
                                                             count);
    size_t quadIndexCount = 0u;
    MGLDrawMetalHandle quadIndexBuffer = lineMode
        ? mglNewQuadElementLineIndexBuffer(device, quadSource, glIndexType, (size_t)count, &quadIndexCount)
        : mglNewQuadElementIndexBuffer(device, quadSource, glIndexType, (size_t)count, &quadIndexCount);
    if (!quadIndexBuffer || quadIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s quad element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              quadSource);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
        lineMode ? MGL_DRAW_PRIMITIVE_LINE : MGL_DRAW_PRIMITIVE_TRIANGLE,
        quadIndexCount, MGL_DRAW_INDEX_UINT32, quadIndexBuffer, 0,
        instanceCount, baseVertex, baseInstance);
    return true;
}

static bool mglEncodeArrayPolygonPointTarget(void *renderEncoderOwner,
                                       MGLDrawMetalHandle device,
                                       GLenum mode,
                                       GLint first,
                                       GLsizei count,
                                       size_t instanceCount,
                                       size_t baseInstance,
                                       const char *label)
{
    if (count < 3) {
        return true;
    }
    if (mode == GL_QUADS && count < 4) {
        return true;
    }

    if (mode == GL_TRIANGLES) {
        size_t drawableCount = ((size_t)count / 3u) * 3u;
        if (drawableCount == 0u) {
            return true;
        }
        mglDrawEncodePrimitives(renderEncoderOwner,
                                MGL_DRAW_PRIMITIVE_POINT, first,
                                drawableCount, instanceCount, baseInstance);
        return true;
    }

    size_t pointIndexCount = 0u;
    MGLDrawMetalHandle pointIndexBuffer = (MGLDrawMetalHandle)0;
    if (mode == GL_TRIANGLE_FAN) {
        pointIndexBuffer = mglNewTriangleFanArrayIndexBuffer(device,
                                                             (size_t)count,
                                                             &pointIndexCount);
    } else if (mode == GL_TRIANGLE_STRIP) {
        pointIndexBuffer = mglNewTriangleStripArrayIndexBuffer(device,
                                                               (size_t)count,
                                                               &pointIndexCount);
    } else if (mode == GL_QUADS) {
        pointIndexBuffer = mglNewQuadArrayIndexBuffer(device,
                                                      (size_t)count,
                                                      &pointIndexCount);
    } else {
        return false;
    }

    if (!pointIndexBuffer || pointIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s polygon point array emulation failed mode=0x%x count=%d first=%d",
              label ? label : "draw",
              (unsigned)mode,
              (int)count,
              (int)first);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
                         MGL_DRAW_PRIMITIVE_POINT, pointIndexCount,
                         MGL_DRAW_INDEX_UINT32, pointIndexBuffer, 0,
                         instanceCount, first, baseInstance);
    return true;
}

static bool mglEncodeElementPolygonPointTarget(void *renderEncoderOwner,
                                         MGLDrawMetalHandle device,
                                         Buffer *glElementBuffer,
                                         MGLDrawMetalHandle metalElementBuffer,
                                         GLenum mode,
                                         GLenum glIndexType,
                                         uint32_t metalIndexType,
                                         size_t indexOffset,
                                         GLsizei count,
                                         size_t instanceCount,
                                         int64_t baseVertex,
                                         size_t baseInstance,
                                         const char *label)
{
    if (count < 3) {
        return true;
    }
    if (mode == GL_QUADS && count < 4) {
        return true;
    }

    if (mode == GL_TRIANGLES) {
        size_t drawableIndexCount = ((size_t)count / 3u) * 3u;
        if (drawableIndexCount == 0u) {
            return true;
        }

        size_t drawIndexOffset = indexOffset;
        uint64_t drawIndexType = metalIndexType;
        MGLDrawMetalHandle drawIndexBuffer = mglPreparedElementIndexBuffer(device,
                                                                      glElementBuffer,
                                                                      metalElementBuffer,
                                                                      glIndexType,
                                                                      &drawIndexOffset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer) {
            return false;
        }

        mglDrawEncodeIndexed(renderEncoderOwner,
                             MGL_DRAW_PRIMITIVE_POINT,
                             drawableIndexCount, drawIndexType,
                             drawIndexBuffer, drawIndexOffset,
                             instanceCount, baseVertex, baseInstance);
        return true;
    }

    const uint8_t *source = mglElementIndexSourceForDraw(glElementBuffer,
                                                         metalElementBuffer,
                                                         glIndexType,
                                                         indexOffset,
                                                         count);
    size_t pointIndexCount = 0u;
    MGLDrawMetalHandle pointIndexBuffer = (MGLDrawMetalHandle)0;
    if (mode == GL_TRIANGLE_FAN) {
        pointIndexBuffer = mglNewTriangleFanElementIndexBuffer(device,
                                                               source,
                                                               glIndexType,
                                                               (size_t)count,
                                                               &pointIndexCount);
    } else if (mode == GL_TRIANGLE_STRIP) {
        pointIndexBuffer = mglNewTriangleStripElementIndexBuffer(device,
                                                                 source,
                                                                 glIndexType,
                                                                 (size_t)count,
                                                                 &pointIndexCount);
    } else if (mode == GL_QUADS) {
        pointIndexBuffer = mglNewQuadElementIndexBuffer(device,
                                                        source,
                                                        glIndexType,
                                                        (size_t)count,
                                                        &pointIndexCount);
    } else {
        return false;
    }

    if (!pointIndexBuffer || pointIndexCount == 0u) {
        fprintf(stderr, "MGL WARNING: %s polygon point element emulation failed mode=0x%x ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              (unsigned)mode,
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              source);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
                         MGL_DRAW_PRIMITIVE_POINT, pointIndexCount,
                         MGL_DRAW_INDEX_UINT32, pointIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return true;
}

static bool mglEncodeRestartSegmentTarget(void *renderEncoderOwner,
                                    MGLDrawMetalHandle device,
                                    Buffer *glElementBuffer,
                                    MGLDrawMetalHandle metalElementBuffer,
                                    MGLDrawMetalHandle preparedIndexBuffer,
                                    GLenum mode,
                                    uint32_t primitiveType,
                                    GLenum glIndexType,
                                    uint32_t preparedIndexType,
                                    size_t baseIndexByteOffset,
                                    size_t segmentStart,
                                    size_t segmentIndexCount,
                                    size_t instanceCount,
                                    int64_t baseVertex,
                                    size_t baseInstance,
                                    bool lineMode,
                                    const char *label)
{
    if (!mglPrimitiveModeHasDrawableSegment(mode, segmentIndexCount)) {
        return true;
    }

    size_t segmentGLByteOffset = 0u;
    size_t indexStride = mglGLIndexElementSize(glIndexType);
    if (!mglComputeIndexByteOffset(baseIndexByteOffset,
                                   segmentStart,
                                   indexStride,
                                   &segmentGLByteOffset)) {
        fprintf(stderr, "MGL WARNING: %s primitive restart segment offset overflow base=%lu start=%lu stride=%lu count=%lu",
              label ? label : "draw",
              (unsigned long)baseIndexByteOffset,
              (unsigned long)segmentStart,
              (unsigned long)indexStride,
              (unsigned long)segmentIndexCount);
        return false;
    }

    if (primitiveType == MGL_DRAW_PRIMITIVE_POINT &&
        (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN || mode == GL_QUADS)) {
        return mglEncodeElementPolygonPointTarget(renderEncoderOwner,
                                            device,
                                            glElementBuffer,
                                            metalElementBuffer,
                                            mode,
                                            glIndexType,
                                            preparedIndexType,
                                            segmentGLByteOffset,
                                            (GLsizei)segmentIndexCount,
                                            instanceCount,
                                            baseVertex,
                                            baseInstance,
                                            label);
    }

    if (mode == GL_TRIANGLE_FAN) {
        return mglEncodeElementTriangleFanTarget(renderEncoderOwner,
                                           device,
                                           glElementBuffer,
                                           metalElementBuffer,
                                           glIndexType,
                                           segmentGLByteOffset,
                                           (GLsizei)segmentIndexCount,
                                           instanceCount,
                                           baseVertex,
                                           baseInstance,
                                           label);
    }

    if (mode == GL_LINE_LOOP) {
        return mglEncodeElementLineLoopTarget(renderEncoderOwner,
                                        device,
                                        glElementBuffer,
                                        metalElementBuffer,
                                        glIndexType,
                                        segmentGLByteOffset,
                                        (GLsizei)segmentIndexCount,
                                        instanceCount,
                                        baseVertex,
                                        baseInstance,
                                        label);
    }

    if (mode == GL_QUADS) {
        return mglEncodeElementQuadsTarget(renderEncoderOwner,
                                     device,
                                     glElementBuffer,
                                     metalElementBuffer,
                                     glIndexType,
                                     segmentGLByteOffset,
                                     (GLsizei)segmentIndexCount,
                                     instanceCount,
                                     baseVertex,
                                     baseInstance,
                                     lineMode,
                                     label);
    }

    size_t preparedByteOffset = 0u;
    if (!mglComputePreparedIndexByteOffset(glIndexType,
                                           segmentGLByteOffset,
                                           &preparedByteOffset)) {
        fprintf(stderr, "MGL WARNING: %s primitive restart prepared offset overflow glType=0x%x byteOffset=%lu",
              label ? label : "draw",
              (unsigned)glIndexType,
              (unsigned long)segmentGLByteOffset);
        return false;
    }

    mglDrawEncodeIndexed(renderEncoderOwner,
                         primitiveType, segmentIndexCount,
                         preparedIndexType, preparedIndexBuffer,
                         preparedByteOffset, instanceCount, baseVertex,
                         baseInstance);
    return true;
}

static MGLPrimitiveRestartEncodeResult mglEncodePrimitiveRestartedElementDrawTarget(void *renderEncoderOwner,
                                                                              MGLDrawMetalHandle device,
                                                                              GLMContext ctx,
                                                                              Buffer *glElementBuffer,
                                                                              MGLDrawMetalHandle metalElementBuffer,
                                                                              GLenum mode,
                                                                              uint32_t primitiveType,
                                                                              GLenum glIndexType,
                                                                              uint32_t metalIndexType,
                                                                              size_t indexOffset,
                                                                              GLsizei count,
                                                                              size_t instanceCount,
                                                                              int64_t baseVertex,
                                                                              size_t baseInstance,
                                                                              const char *label)
{
    uint32_t restartIndex = 0u;
    if (!mglPrimitiveRestartIndexForType(ctx, glIndexType, &restartIndex)) {
        return MGLPrimitiveRestartEncodeNotNeeded;
    }
    if (count <= 0) {
        return MGLPrimitiveRestartEncodeHandled;
    }

    const uint8_t *source = mglElementIndexSourceForDraw(glElementBuffer,
                                                         metalElementBuffer,
                                                         glIndexType,
                                                         indexOffset,
                                                         count);
    if (!source) {
        fprintf(stderr, "MGL WARNING: %s primitive restart enabled but index bytes are not CPU-readable ebo=%u count=%d type=0x%x offset=%lu; skipping draw to avoid treating restart as a vertex",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned)glIndexType,
              (unsigned long)indexOffset);
        return MGLPrimitiveRestartEncodeFailed;
    }

    /* Single type-specialized scan replaces the original two-pass
     * approach (detect + segment).  Type-specialized pointer access
     * eliminates the per-element switch+memcpy overhead of
     * mglReadGLIndexValue, and collecting restart positions in a stack
     * array avoids re-scanning the index buffer during segment encoding.
     * The 256-entry array covers virtually all real draws; the rare
     * overflow case falls back to a type-specialized re-scan. */
    size_t restartPositions[256];
    size_t restartPositionCount = 0;  /* total count, may exceed 256 */
    bool sawRestart = false;

    switch (glIndexType) {
        case GL_UNSIGNED_BYTE: {
            const uint8_t *typedSrc = (const uint8_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == (uint8_t)restartIndex) {
                    sawRestart = true;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (size_t)i;
                    restartPositionCount++;
                }
            }
            break;
        }
        case GL_UNSIGNED_SHORT: {
            const uint16_t *typedSrc = (const uint16_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == (uint16_t)restartIndex) {
                    sawRestart = true;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (size_t)i;
                    restartPositionCount++;
                }
            }
            break;
        }
        case GL_UNSIGNED_INT: {
            const uint32_t *typedSrc = (const uint32_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == restartIndex) {
                    sawRestart = true;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (size_t)i;
                    restartPositionCount++;
                }
            }
            break;
        }
        default:
            break;
    }
    if (!sawRestart) {
        return MGLPrimitiveRestartEncodeNotNeeded;
    }

    bool emulatedMode = (mode == GL_TRIANGLE_FAN ||
                         mode == GL_LINE_LOOP ||
                         mode == GL_QUADS ||
                         (primitiveType == MGL_DRAW_PRIMITIVE_POINT &&
                          (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP)));
    MGLDrawMetalHandle preparedIndexBuffer = metalElementBuffer;
    uint64_t preparedIndexType = metalIndexType;
    if (!emulatedMode) {
        preparedIndexBuffer = mglPreparedElementIndexBuffer(device,
                                                            glElementBuffer,
                                                            metalElementBuffer,
                                                            glIndexType,
                                                            NULL,
                                                            &preparedIndexType);
        if (!preparedIndexBuffer) {
            return MGLPrimitiveRestartEncodeFailed;
        }
    }

    size_t segmentStart = 0u;
    bool encodedAllSegments = true;

    if (restartPositionCount <= 256) {
        /* Common path: use collected positions — no re-scan needed. */
        for (size_t rp = 0; rp < restartPositionCount; rp++) {
            size_t restartAt = restartPositions[rp];
            size_t segmentCount = restartAt - segmentStart;
            if (!mglEncodeRestartSegmentTarget(renderEncoderOwner,
                                         device,
                                         glElementBuffer,
                                         metalElementBuffer,
                                         preparedIndexBuffer,
                                         mode,
                                         primitiveType,
                                         glIndexType,
                                         preparedIndexType,
                                         indexOffset,
                                         segmentStart,
                                         segmentCount,
                                         instanceCount,
                                         baseVertex,
                                         baseInstance,
                                         mglPolygonModeLineForDrawMode(ctx, mode),
                                         label)) {
                encodedAllSegments = false;
                break;
            }
            segmentStart = restartAt + 1u;
        }
    } else {
        /* Fallback: too many restarts for stack array, type-specialized re-scan. */
        switch (glIndexType) {
            case GL_UNSIGNED_BYTE: {
                const uint8_t *typedSrc = (const uint8_t *)source;
                for (GLsizei i = 0; i < count && encodedAllSegments; i++) {
                    if (typedSrc[i] != (uint8_t)restartIndex) continue;
                    size_t segmentCount = (size_t)i - segmentStart;
                    if (!mglEncodeRestartSegmentTarget(renderEncoderOwner,
                                                 device, glElementBuffer, metalElementBuffer,
                                                 preparedIndexBuffer, mode, primitiveType,
                                                 glIndexType, preparedIndexType, indexOffset,
                                                 segmentStart, segmentCount, instanceCount,
                                                 baseVertex, baseInstance,
                                                 mglPolygonModeLineForDrawMode(ctx, mode), label)) {
                        encodedAllSegments = false;
                        break;
                    }
                    segmentStart = (size_t)i + 1u;
                }
                break;
            }
            case GL_UNSIGNED_SHORT: {
                const uint16_t *typedSrc = (const uint16_t *)source;
                for (GLsizei i = 0; i < count && encodedAllSegments; i++) {
                    if (typedSrc[i] != (uint16_t)restartIndex) continue;
                    size_t segmentCount = (size_t)i - segmentStart;
                    if (!mglEncodeRestartSegmentTarget(renderEncoderOwner,
                                                 device, glElementBuffer, metalElementBuffer,
                                                 preparedIndexBuffer, mode, primitiveType,
                                                 glIndexType, preparedIndexType, indexOffset,
                                                 segmentStart, segmentCount, instanceCount,
                                                 baseVertex, baseInstance,
                                                 mglPolygonModeLineForDrawMode(ctx, mode), label)) {
                        encodedAllSegments = false;
                        break;
                    }
                    segmentStart = (size_t)i + 1u;
                }
                break;
            }
            case GL_UNSIGNED_INT: {
                const uint32_t *typedSrc = (const uint32_t *)source;
                for (GLsizei i = 0; i < count && encodedAllSegments; i++) {
                    if (typedSrc[i] != restartIndex) continue;
                    size_t segmentCount = (size_t)i - segmentStart;
                    if (!mglEncodeRestartSegmentTarget(renderEncoderOwner,
                                                 device, glElementBuffer, metalElementBuffer,
                                                 preparedIndexBuffer, mode, primitiveType,
                                                 glIndexType, preparedIndexType, indexOffset,
                                                 segmentStart, segmentCount, instanceCount,
                                                 baseVertex, baseInstance,
                                                 mglPolygonModeLineForDrawMode(ctx, mode), label)) {
                        encodedAllSegments = false;
                        break;
                    }
                    segmentStart = (size_t)i + 1u;
                }
                break;
            }
            default:
                break;
        }
    }

    if (encodedAllSegments) {
        size_t trailingCount = (size_t)count - segmentStart;
        encodedAllSegments = mglEncodeRestartSegmentTarget(renderEncoderOwner,
                                                     device,
                                                     glElementBuffer,
                                                     metalElementBuffer,
                                                     preparedIndexBuffer,
                                                     mode,
                                                     primitiveType,
                                                     glIndexType,
                                                     preparedIndexType,
                                                     indexOffset,
                                                     segmentStart,
                                                     trailingCount,
                                                     instanceCount,
                                                     baseVertex,
                                                     baseInstance,
                                                     mglPolygonModeLineForDrawMode(ctx, mode),
                                                     label);
    }

    return encodedAllSegments ? MGLPrimitiveRestartEncodeHandled : MGLPrimitiveRestartEncodeFailed;
}

bool mglEncodeArrayLineLoopForRenderEncoderOwner(
    void *renderEncoderOwner,
    GLMContext drawCtx, MGLDrawMetalHandle device, GLsizei count,
    GLint firstVertex, size_t instanceCount, size_t baseInstance,
    const char *label)
{
    return mglEncodeArrayLineLoopTarget(renderEncoderOwner, drawCtx, device, count, firstVertex,
        instanceCount, baseInstance, label);
}

bool mglEncodeArrayTriangleFanForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLsizei count, GLint baseVertex,
    size_t instanceCount, size_t baseInstance, const char *label)
{
    return mglEncodeArrayTriangleFanTarget(renderEncoderOwner, device, count, baseVertex, instanceCount,
        baseInstance, label);
}

bool mglEncodeElementLineLoopForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum glIndexType,
    size_t indexOffset, GLsizei count, size_t instanceCount,
    int64_t baseVertex, size_t baseInstance, const char *label)
{
    return mglEncodeElementLineLoopTarget(renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, glIndexType, indexOffset, count, instanceCount,
        baseVertex, baseInstance, label);
}

bool mglEncodeElementTriangleFanForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum glIndexType,
    size_t indexOffset, GLsizei count, size_t instanceCount,
    int64_t baseVertex, size_t baseInstance, const char *label)
{
    return mglEncodeElementTriangleFanTarget(renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, glIndexType, indexOffset, count, instanceCount,
        baseVertex, baseInstance, label);
}

bool mglEncodeArrayQuadsForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLsizei count, GLint baseVertex,
    size_t instanceCount, size_t baseInstance, bool lineMode,
    const char *label)
{
    return mglEncodeArrayQuadsTarget(renderEncoderOwner, device, count, baseVertex, instanceCount,
        baseInstance, lineMode, label);
}

bool mglEncodeElementQuadsForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum glIndexType,
    size_t indexOffset, GLsizei count, size_t instanceCount,
    int64_t baseVertex, size_t baseInstance, bool lineMode,
    const char *label)
{
    return mglEncodeElementQuadsTarget(renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, glIndexType, indexOffset, count, instanceCount,
        baseVertex, baseInstance, lineMode, label);
}

bool mglEncodeArrayPolygonPointForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLenum mode, GLint first, GLsizei count,
    size_t instanceCount, size_t baseInstance, const char *label)
{
    return mglEncodeArrayPolygonPointTarget(renderEncoderOwner, device, mode, first, count,
        instanceCount, baseInstance, label);
}

bool mglEncodeElementPolygonPointForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum mode, GLenum glIndexType,
    uint32_t metalIndexType, size_t indexOffset, GLsizei count,
    size_t instanceCount, int64_t baseVertex,
    size_t baseInstance, const char *label)
{
    return mglEncodeElementPolygonPointTarget(renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, mode, glIndexType, metalIndexType, indexOffset,
        count, instanceCount, baseVertex, baseInstance, label);
}

MGLPrimitiveRestartEncodeResult
mglEncodePrimitiveRestartedElementDrawForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLMContext ctx, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum mode,
    uint32_t primitiveType, GLenum glIndexType,
    uint32_t metalIndexType, size_t indexOffset, GLsizei count,
    size_t instanceCount, int64_t baseVertex,
    size_t baseInstance, const char *label)
{
    return mglEncodePrimitiveRestartedElementDrawTarget(renderEncoderOwner, device, ctx, glElementBuffer,
        metalElementBuffer, mode, primitiveType, glIndexType, metalIndexType,
        indexOffset, count, instanceCount, baseVertex, baseInstance, label);
}

bool mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(GLMContext ctx,
                                                                  GLenum glIndexType,
                                                                  const char *label)
{
    uint32_t restartIndex = 0u;
    if (!mglPrimitiveRestartIndexForType(ctx, glIndexType, &restartIndex)) {
        return false;
    }

    static uint64_t s_indirectRestartSkipCount = 0;
    s_indirectRestartSkipCount++;
    if (s_indirectRestartSkipCount <= 8u || (s_indirectRestartSkipCount % 1000u) == 0u) {
        fprintf(stderr, "MGL WARNING: %s primitive restart with indirect indexed draw is not emulated yet type=0x%x restart=%u occurrence=%llu; skipping draw",
              label ? label : "drawElementsIndirect",
              (unsigned)glIndexType,
              (unsigned)restartIndex,
              (unsigned long long)s_indirectRestartSkipCount);
    }
    return true;
}

bool mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(GLMContext ctx,
                                                               GLenum mode,
                                                               const char *label)
{
    if (!mglPolygonModePointForDrawMode(ctx, mode)) {
        return false;
    }

    static uint64_t s_indirectPolygonPointSkipCount = 0;
    s_indirectPolygonPointSkipCount++;
    if (s_indirectPolygonPointSkipCount <= 8u || (s_indirectPolygonPointSkipCount % 1000u) == 0u) {
        fprintf(stderr, "MGL WARNING: %s GL_POLYGON_MODE=GL_POINT requires triangle expansion for indirect draw mode=0x%x occurrence=%llu; skipping draw",
              label ? label : "drawIndirect",
              (unsigned)mode,
              (unsigned long long)s_indirectPolygonPointSkipCount);
    }
    return true;
}
