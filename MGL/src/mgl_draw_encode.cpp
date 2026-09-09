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
#include "mgl_render.h"

static void mglDrawEncodePrimitives(void *renderEncoderOwner,
                                    uint32_t primitiveType,
                                    size_t vertexStart,
                                    size_t vertexCount,
                                    size_t instanceCount,
                                    size_t baseInstance)
{
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    if (!renderEncoderOwner) return;
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
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
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)indexType,
            .index_buffer = (void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .base_vertex = baseVertex,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    if (!renderEncoderOwner) return;
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
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
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
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
    if (mglRenderQuadsCountTooSmall((uint32_t)mode, (int32_t)count)) {
        return true;
    }

    if (mglRenderDrawModeIsTriangles((uint32_t)mode)) {
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
    if (mglRenderEmulateTriangleFan((uint32_t)mode, 0)) {
        pointIndexBuffer = mglNewTriangleFanArrayIndexBuffer(device,
                                                             (size_t)count,
                                                             &pointIndexCount);
    } else if (mglRenderDrawModeIsTriangleStrip((uint32_t)mode)) {
        pointIndexBuffer = mglNewTriangleStripArrayIndexBuffer(device,
                                                               (size_t)count,
                                                               &pointIndexCount);
    } else if (mglRenderEmulateQuads((uint32_t)mode, 0)) {
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
    if (mglRenderQuadsCountTooSmall((uint32_t)mode, (int32_t)count)) {
        return true;
    }

    if (mglRenderDrawModeIsTriangles((uint32_t)mode)) {
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
    if (mglRenderEmulateTriangleFan((uint32_t)mode, 0)) {
        pointIndexBuffer = mglNewTriangleFanElementIndexBuffer(device,
                                                               source,
                                                               glIndexType,
                                                               (size_t)count,
                                                               &pointIndexCount);
    } else if (mglRenderDrawModeIsTriangleStrip((uint32_t)mode)) {
        pointIndexBuffer = mglNewTriangleStripElementIndexBuffer(device,
                                                                 source,
                                                                 glIndexType,
                                                                 (size_t)count,
                                                                 &pointIndexCount);
    } else if (mglRenderEmulateQuads((uint32_t)mode, 0)) {
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
        mglRenderPolygonPointEmulateMode((uint32_t)mode)) {
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

    if (mglRenderEmulateTriangleFan((uint32_t)mode, 0)) {
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

    if (mglRenderEmulateLineLoop((uint32_t)mode)) {
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

    if (mglRenderEmulateQuads((uint32_t)mode, 0)) {
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

    if (mglRenderIndexTypeIsU8((uint32_t)glIndexType)) {
            const uint8_t *typedSrc = (const uint8_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == (uint8_t)restartIndex) {
                    sawRestart = true;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (size_t)i;
                    restartPositionCount++;
                }
            }
    } else if (mglRenderIndexTypeIsU16((uint32_t)glIndexType)) {
            const uint16_t *typedSrc = (const uint16_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == (uint16_t)restartIndex) {
                    sawRestart = true;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (size_t)i;
                    restartPositionCount++;
                }
            }
    } else if (mglRenderIndexTypeIsU32((uint32_t)glIndexType)) {
            const uint32_t *typedSrc = (const uint32_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == restartIndex) {
                    sawRestart = true;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (size_t)i;
                    restartPositionCount++;
                }
            }
    }
    if (!sawRestart) {
        return MGLPrimitiveRestartEncodeNotNeeded;
    }

    bool emulatedMode = (mglRenderDrawModeNeedsEmulate((uint32_t)mode) ||
                         (primitiveType == MGL_DRAW_PRIMITIVE_POINT &&
                          (mglRenderDrawModeIsTriangles((uint32_t)mode) ||
                           mglRenderDrawModeIsTriangleStrip((uint32_t)mode))));
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
        if (mglRenderIndexTypeIsU8((uint32_t)glIndexType)) {
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
        } else if (mglRenderIndexTypeIsU16((uint32_t)glIndexType)) {
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
        } else if (mglRenderIndexTypeIsU32((uint32_t)glIndexType)) {
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

bool mglEncodeDrawArraysForRenderEncoderOwner(
    void *renderEncoderOwner,
    GLMContext ctx,
    MGLDrawMetalHandle device,
    GLenum mode,
    GLint first,
    GLsizei count,
    size_t instanceCount,
    size_t baseInstance,
    const char *label)
{
    if (!renderEncoderOwner) {
        return false;
    }
    if (mglPolygonModePointForDrawMode(ctx, mode)) {
        return mglEncodeArrayPolygonPointForRenderEncoderOwner(
            renderEncoderOwner, device, mode, first, count,
            instanceCount, baseInstance, label);
    }
    if (mglRenderEmulateTriangleFan((uint32_t)mode, 0)) {
        return mglEncodeArrayTriangleFanForRenderEncoderOwner(
            renderEncoderOwner, device, count, first,
            instanceCount, baseInstance, label);
    }
    if (mglRenderEmulateLineLoop((uint32_t)mode)) {
        return mglEncodeArrayLineLoopForRenderEncoderOwner(
            renderEncoderOwner, ctx, device, count, first,
            instanceCount, baseInstance, label);
    }
    if (mglRenderEmulateQuads((uint32_t)mode, 0)) {
        return mglEncodeArrayQuadsForRenderEncoderOwner(
            renderEncoderOwner, device, count, first,
            instanceCount, baseInstance,
            mglPolygonModeLineForDrawMode(ctx, mode), label);
    }
    const uint32_t primitiveType =
        mglRenderMTLPrimitiveTypeForGLMode((uint32_t)mode);
    if (primitiveType == 0xFFFFFFFFu) {
        fprintf(stderr,
                "MGL WARNING: %s unsupported primitive mode=0x%x, skipping draw",
                label ? label : "drawArrays",
                (unsigned)mode);
        return false;
    }
    mglDrawEncodePrimitives(renderEncoderOwner, primitiveType,
                            (size_t)first, (size_t)count,
                            instanceCount, baseInstance);
    return true;
}

bool mglEncodeDrawElementsForRenderEncoderOwner(
    void *renderEncoderOwner,
    GLMContext ctx,
    MGLDrawMetalHandle device,
    Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer,
    GLenum mode,
    GLenum glIndexType,
    size_t indexOffset,
    GLsizei count,
    size_t instanceCount,
    int64_t baseVertex,
    size_t baseInstance,
    const char *label)
{
    if (!renderEncoderOwner || !glElementBuffer || !metalElementBuffer) {
        return false;
    }

    const uint32_t metalIndexType =
        mglRenderMTLIndexTypeForGLType((uint32_t)glIndexType);
    if (metalIndexType == 0xFFFFFFFFu) {
        fprintf(stderr,
                "MGL WARNING: %s unsupported index type=0x%x, skipping draw",
                label ? label : "drawElements",
                (unsigned)glIndexType);
        return false;
    }

    const bool polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    uint32_t primitiveType;
    if (polygonModePoint) {
        primitiveType = MGL_DRAW_PRIMITIVE_POINT;
    } else if (mglRenderEmulateTriangleFan((uint32_t)mode, 0)) {
        primitiveType = MGL_DRAW_PRIMITIVE_TRIANGLE;
    } else if (mglRenderEmulateLineLoop((uint32_t)mode)) {
        primitiveType = MGL_DRAW_PRIMITIVE_LINE_STRIP;
    } else if (mglRenderEmulateQuads((uint32_t)mode, 0)) {
        primitiveType = MGL_DRAW_PRIMITIVE_TRIANGLE;
    } else {
        primitiveType = mglRenderMTLPrimitiveTypeForGLMode((uint32_t)mode);
    }
    if (primitiveType == 0xFFFFFFFFu) {
        fprintf(stderr,
                "MGL WARNING: %s unsupported primitive mode=0x%x, skipping draw",
                label ? label : "drawElements",
                (unsigned)mode);
        return false;
    }

    const MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDrawForRenderEncoderOwner(
            renderEncoderOwner, device, ctx, glElementBuffer,
            metalElementBuffer, mode, primitiveType, glIndexType,
            metalIndexType, indexOffset, count, instanceCount,
            baseVertex, baseInstance, label);
    if (restartResult == MGLPrimitiveRestartEncodeFailed) {
        return false;
    }
    if (restartResult == MGLPrimitiveRestartEncodeHandled) {
        return true;
    }

    if (polygonModePoint) {
        return mglEncodeElementPolygonPointForRenderEncoderOwner(
            renderEncoderOwner, device, glElementBuffer, metalElementBuffer,
            mode, glIndexType, metalIndexType, indexOffset, count,
            instanceCount, baseVertex, baseInstance, label);
    }
    if (mglRenderEmulateTriangleFan((uint32_t)mode, 0)) {
        return mglEncodeElementTriangleFanForRenderEncoderOwner(
            renderEncoderOwner, device, glElementBuffer, metalElementBuffer,
            glIndexType, indexOffset, count, instanceCount,
            baseVertex, baseInstance, label);
    }
    if (mglRenderEmulateLineLoop((uint32_t)mode)) {
        return mglEncodeElementLineLoopForRenderEncoderOwner(
            renderEncoderOwner, device, glElementBuffer, metalElementBuffer,
            glIndexType, indexOffset, count, instanceCount,
            baseVertex, baseInstance, label);
    }
    if (mglRenderEmulateQuads((uint32_t)mode, 0)) {
        return mglEncodeElementQuadsForRenderEncoderOwner(
            renderEncoderOwner, device, glElementBuffer, metalElementBuffer,
            glIndexType, indexOffset, count, instanceCount,
            baseVertex, baseInstance,
            mglPolygonModeLineForDrawMode(ctx, mode), label);
    }

    size_t drawIndexOffset = indexOffset;
    uint64_t drawIndexType = (uint64_t)metalIndexType;
    MGLDrawMetalHandle drawIndexBuffer = mglPreparedElementIndexBuffer(
        device, glElementBuffer, metalElementBuffer, glIndexType,
        &drawIndexOffset, &drawIndexType);
    if (!drawIndexBuffer) {
        return false;
    }
    mglDrawEncodeIndexed(renderEncoderOwner, primitiveType, (size_t)count,
                         (uint32_t)drawIndexType, drawIndexBuffer,
                         drawIndexOffset, instanceCount, baseVertex,
                         baseInstance);
    return true;
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

bool mglEncodeDrawArraysIndirectForRenderEncoderOwner(
    void *renderEncoderOwner,
    GLMContext ctx,
    GLenum mode,
    MGLDrawMetalHandle indirectBuffer,
    size_t indirectOffset,
    const char *label)
{
    (void)label;
    if (!renderEncoderOwner || !indirectBuffer) {
        return false;
    }
    const uint32_t primitiveType =
        mglPolygonModePointForDrawMode(ctx, mode)
            ? (uint32_t)MGL_DRAW_PRIMITIVE_POINT
            : mglRenderMTLPrimitiveTypeForGLMode((uint32_t)mode);
    if (primitiveType == 0xFFFFFFFFu) {
        fprintf(stderr,
                "MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call\n",
                (unsigned)mode);
        return false;
    }
    const MGLRenderDrawPlan plan = {
        .kind = MGL_RENDER_DRAW_ARRAY_INDIRECT,
        .primitive_type = primitiveType,
        .indirect_buffer = (void *)indirectBuffer,
        .indirect_buffer_offset = indirectOffset,
    };
    return mglRenderEncodeDrawForRenderEncoderOwner(
               renderEncoderOwner, &plan, NULL, 0) == 0;
}

bool mglEncodeDrawElementsIndirectForRenderEncoderOwner(
    void *renderEncoderOwner,
    GLMContext ctx,
    MGLDrawMetalHandle device,
    Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer,
    GLenum mode,
    GLenum glIndexType,
    MGLDrawMetalHandle indirectBuffer,
    size_t indirectOffset,
    const char *label)
{
    (void)label;
    if (!renderEncoderOwner || !indirectBuffer) {
        return false;
    }
    uint32_t metalIndexType = mglRenderMTLIndexTypeForGLType((uint32_t)glIndexType);
    if (metalIndexType == 0xFFFFFFFFu) {
        fprintf(stderr,
                "MGL WARNING: Unsupported index type=0x%x, skipping draw call\n",
                (unsigned)glIndexType);
        return false;
    }
    const uint32_t primitiveType =
        mglPolygonModePointForDrawMode(ctx, mode)
            ? (uint32_t)MGL_DRAW_PRIMITIVE_POINT
            : mglRenderMTLPrimitiveTypeForGLMode((uint32_t)mode);
    if (primitiveType == 0xFFFFFFFFu) {
        fprintf(stderr,
                "MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call\n",
                (unsigned)mode);
        return false;
    }
    size_t indexOffset = 0u;
    uint64_t drawIndexType = metalIndexType;
    MGLDrawMetalHandle prepared = mglPreparedElementIndexBuffer(
        device, glElementBuffer, metalElementBuffer, glIndexType, &indexOffset,
        &drawIndexType);
    if (!prepared) {
        return false;
    }
    const MGLRenderDrawPlan plan = {
        .kind = MGL_RENDER_DRAW_INDEXED_INDIRECT,
        .primitive_type = primitiveType,
        .index_type = (uint32_t)drawIndexType,
        .index_buffer = (void *)prepared,
        .index_buffer_offset = indexOffset,
        .indirect_buffer = (void *)indirectBuffer,
        .indirect_buffer_offset = indirectOffset,
    };
    return mglRenderEncodeDrawForRenderEncoderOwner(
               renderEncoderOwner, &plan, NULL, 0) == 0;
}


bool mglEncodeCullDistanceArraySplitForRenderEncoderOwner(
    void *renderEncoderOwner, MGLDrawMetalHandle device, GLenum mode,
    GLint first, GLsizei count, size_t instanceCount, size_t baseInstance,
    void *bind_renderer, const void *bind_encode_context,
    MGLCullDistanceBindFn bind)
{
    if (!renderEncoderOwner || count <= 0 || instanceCount == 0u) {
        return false;
    }
    void *planOwner = NULL;
    void *indexBufferHandle = NULL;
    uint64_t primitiveCount = 0u;
    const int rc = mglRenderCreateCullDistanceArrayPlan(
        (void *)device, (uint32_t)mode, first, (uint64_t)count,
        &planOwner, &indexBufferHandle, &primitiveCount);
    if (rc == 1) {
        return false;
    }
    if (rc != 0 || !planOwner) {
        return true;
    }
    MGLDrawMetalHandle indexBuffer = indexBufferHandle;
    for (uint64_t i = 0u; i < primitiveCount; i++) {
        MGLRenderCullDistancePrimitive prim = {0};
        if (mglRenderGetCullDistanceIndexPrimitive(planOwner, i, &prim) != 0) {
            break;
        }
        if (bind) {
            if (prim.index_count == 0u) {
                bind(bind_renderer, bind_encode_context, mode,
                     prim.vertices[0], NULL, 0u);
            } else {
                bind(bind_renderer, bind_encode_context, mode, (GLuint)first,
                     prim.vertices, prim.vertex_count);
            }
        }
        if (prim.index_count == 0u) {
            mglDrawEncodePrimitives(renderEncoderOwner, prim.primitive_type,
                                    (size_t)prim.vertices[0], 2u, instanceCount,
                                    baseInstance);
        } else if (indexBuffer) {
            mglDrawEncodeIndexed(renderEncoderOwner, prim.primitive_type,
                                 (size_t)prim.index_count,
                                 MGL_DRAW_INDEX_UINT32, indexBuffer,
                                 (size_t)prim.index_buffer_offset,
                                 instanceCount, 0, baseInstance);
        }
    }
    mglRenderDestroyCullDistanceIndexPlan(&planOwner);
    return true;
}
