/*
 * mgl_draw_encode.m
 * MGL
 *
 * Implementation of the Draw Encode Subsystem.
 * See mgl_draw_encode.h for the API contract.
 */

#import "mgl_draw_encode.h"

#include <stdlib.h>
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"
#include "mgl_render_cpp_objc.h" /* transitional Metal ref typedefs */

static void mglDrawEncodePrimitives(MGLMetalRenderCommandEncoderRef encoder,
                                    void *renderEncoderOwner,
                                    MTLPrimitiveType primitiveType,
                                    NSUInteger vertexStart,
                                    NSUInteger vertexCount,
                                    NSUInteger instanceCount,
                                    NSUInteger baseInstance)
{
    const MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    if (renderEncoderOwner &&
        mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice()) {
        (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
            renderEncoderOwner, &plan, NULL, 0);
    } else {
        (void)mglRenderCppEncodeDraw((__bridge void *)encoder, &plan, NULL, 0);
    }
}

static void mglDrawEncodeIndexed(MGLMetalRenderCommandEncoderRef encoder,
                                 void *renderEncoderOwner,
                                 MTLPrimitiveType primitiveType,
                                 NSUInteger indexCount,
                                 MTLIndexType indexType,
                                 MGLMetalBufferRef indexBuffer,
                                 NSUInteger indexBufferOffset,
                                 NSUInteger instanceCount,
                                 NSInteger baseVertex,
                                 NSUInteger baseInstance)
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
    if (renderEncoderOwner &&
        mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice()) {
        (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
            renderEncoderOwner, &plan, NULL, 0);
    } else {
        (void)mglRenderCppEncodeDraw((__bridge void *)encoder, &plan, NULL, 0);
    }
}

static BOOL mglEncodeArrayLineLoopTarget(MGLMetalRenderCommandEncoderRef encoder,
                            void *renderEncoderOwner,
                            GLMContext drawCtx,
                            MGLMetalDeviceRef device,
                            GLsizei count,
                            GLint firstVertex,
                            NSUInteger instanceCount,
                            NSUInteger baseInstance,
                            const char *label)
{
    if (count < 2) {
        return YES;
    }
    if (firstVertex < 0) {
        NSLog(@"MGL WARNING: %s line loop array emulation invalid first=%d",
              label ? label : "draw",
              (int)firstVertex);
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_VALUE);
        }
        return NO;
    }

    NSUInteger loopIndexCount = 0u;
    MGLMetalBufferRef loopIndexBuffer = mglNewLineLoopArrayIndexBuffer(device,
                                                                   (NSUInteger)firstVertex,
                                                                   (NSUInteger)count,
                                                                   &loopIndexCount);
    if (!loopIndexBuffer || loopIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s line loop array emulation failed count=%d first=%d",
              label ? label : "draw",
              (int)count,
              (int)firstVertex);
        return NO;
    }

    mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                         MTLPrimitiveTypeLineStrip, loopIndexCount,
                         MTLIndexTypeUInt32, loopIndexBuffer, 0,
                         instanceCount, 0, baseInstance);
    return YES;
}

static BOOL mglEncodeArrayTriangleFanTarget(MGLMetalRenderCommandEncoderRef encoder,
                                      void *renderEncoderOwner,
                                      MGLMetalDeviceRef device,
                                      GLsizei count,
                                      GLint baseVertex,
                                      NSUInteger instanceCount,
                                      NSUInteger baseInstance,
                                      const char *label)
{
    if (count < 3) {
        return YES;
    }

    NSUInteger fanIndexCount = 0u;
    MGLMetalBufferRef fanIndexBuffer = mglNewTriangleFanArrayIndexBuffer(device,
                                                                     (NSUInteger)count,
                                                                     &fanIndexCount);
    if (!fanIndexBuffer || fanIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s triangle fan array emulation failed count=%d baseVertex=%d",
              label ? label : "draw",
              (int)count,
              (int)baseVertex);
        return NO;
    }

    mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                         MTLPrimitiveTypeTriangle, fanIndexCount,
                         MTLIndexTypeUInt32, fanIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return YES;
}

static BOOL mglEncodeElementLineLoopTarget(MGLMetalRenderCommandEncoderRef encoder,
                                     void *renderEncoderOwner,
                                     MGLMetalDeviceRef device,
                                     Buffer *glElementBuffer,
                                     MGLMetalBufferRef metalElementBuffer,
                                     GLenum glIndexType,
                                     NSUInteger indexOffset,
                                     GLsizei count,
                                     NSUInteger instanceCount,
                                     NSInteger baseVertex,
                                     NSUInteger baseInstance,
                                     const char *label)
{
    if (count < 2) {
        return YES;
    }

    const uint8_t *loopSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                             metalElementBuffer,
                                                             glIndexType,
                                                             indexOffset,
                                                             count);
    NSUInteger loopIndexCount = 0u;
    MGLMetalBufferRef loopIndexBuffer = mglNewLineLoopElementIndexBuffer(device,
                                                                     loopSource,
                                                                     glIndexType,
                                                                     (NSUInteger)count,
                                                                     &loopIndexCount);
    if (!loopIndexBuffer || loopIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s line loop element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              loopSource);
        return NO;
    }

    mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                         MTLPrimitiveTypeLineStrip, loopIndexCount,
                         MTLIndexTypeUInt32, loopIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return YES;
}

static BOOL mglEncodeElementTriangleFanTarget(MGLMetalRenderCommandEncoderRef encoder,
                                        void *renderEncoderOwner,
                                        MGLMetalDeviceRef device,
                                        Buffer *glElementBuffer,
                                        MGLMetalBufferRef metalElementBuffer,
                                        GLenum glIndexType,
                                        NSUInteger indexOffset,
                                        GLsizei count,
                                        NSUInteger instanceCount,
                                        NSInteger baseVertex,
                                        NSUInteger baseInstance,
                                        const char *label)
{
    if (count < 3) {
        return YES;
    }

    const uint8_t *fanSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                            metalElementBuffer,
                                                            glIndexType,
                                                            indexOffset,
                                                            count);
    NSUInteger fanIndexCount = 0u;
    MGLMetalBufferRef fanIndexBuffer = mglNewTriangleFanElementIndexBuffer(device,
                                                                       fanSource,
                                                                       glIndexType,
                                                                       (NSUInteger)count,
                                                                       &fanIndexCount);
    if (!fanIndexBuffer || fanIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s triangle fan element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              fanSource);
        return NO;
    }

    mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                         MTLPrimitiveTypeTriangle, fanIndexCount,
                         MTLIndexTypeUInt32, fanIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return YES;
}

static BOOL mglEncodeArrayQuadsTarget(MGLMetalRenderCommandEncoderRef encoder,
                                void *renderEncoderOwner,
                                MGLMetalDeviceRef device,
                                GLsizei count,
                                GLint baseVertex,
                                NSUInteger instanceCount,
                                NSUInteger baseInstance,
                                BOOL lineMode,
                                const char *label)
{
    if (count < 4) {
        return YES;
    }

    NSUInteger quadIndexCount = 0u;
    MGLMetalBufferRef quadIndexBuffer = lineMode
        ? mglNewQuadArrayLineIndexBuffer(device, (NSUInteger)count, &quadIndexCount)
        : mglNewQuadArrayIndexBuffer(device, (NSUInteger)count, &quadIndexCount);
    if (!quadIndexBuffer || quadIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s quad array emulation failed count=%d baseVertex=%d",
              label ? label : "draw",
              (int)count,
              (int)baseVertex);
        return NO;
    }

    mglDrawEncodeIndexed(
        encoder, renderEncoderOwner,
        lineMode ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle,
        quadIndexCount, MTLIndexTypeUInt32, quadIndexBuffer, 0,
        instanceCount, baseVertex, baseInstance);
    return YES;
}

static BOOL mglEncodeElementQuadsTarget(MGLMetalRenderCommandEncoderRef encoder,
                                  void *renderEncoderOwner,
                                  MGLMetalDeviceRef device,
                                  Buffer *glElementBuffer,
                                  MGLMetalBufferRef metalElementBuffer,
                                  GLenum glIndexType,
                                  NSUInteger indexOffset,
                                  GLsizei count,
                                  NSUInteger instanceCount,
                                  NSInteger baseVertex,
                                  NSUInteger baseInstance,
                                  BOOL lineMode,
                                  const char *label)
{
    if (count < 4) {
        return YES;
    }

    const uint8_t *quadSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                             metalElementBuffer,
                                                             glIndexType,
                                                             indexOffset,
                                                             count);
    NSUInteger quadIndexCount = 0u;
    MGLMetalBufferRef quadIndexBuffer = lineMode
        ? mglNewQuadElementLineIndexBuffer(device, quadSource, glIndexType, (NSUInteger)count, &quadIndexCount)
        : mglNewQuadElementIndexBuffer(device, quadSource, glIndexType, (NSUInteger)count, &quadIndexCount);
    if (!quadIndexBuffer || quadIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s quad element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              quadSource);
        return NO;
    }

    mglDrawEncodeIndexed(
        encoder, renderEncoderOwner,
        lineMode ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle,
        quadIndexCount, MTLIndexTypeUInt32, quadIndexBuffer, 0,
        instanceCount, baseVertex, baseInstance);
    return YES;
}

static BOOL mglEncodeArrayPolygonPointTarget(MGLMetalRenderCommandEncoderRef encoder,
                                       void *renderEncoderOwner,
                                       MGLMetalDeviceRef device,
                                       GLenum mode,
                                       GLint first,
                                       GLsizei count,
                                       NSUInteger instanceCount,
                                       NSUInteger baseInstance,
                                       const char *label)
{
    if (count < 3) {
        return YES;
    }
    if (mode == GL_QUADS && count < 4) {
        return YES;
    }

    if (mode == GL_TRIANGLES) {
        NSUInteger drawableCount = ((NSUInteger)count / 3u) * 3u;
        if (drawableCount == 0u) {
            return YES;
        }
        mglDrawEncodePrimitives(encoder, renderEncoderOwner,
                                MTLPrimitiveTypePoint, first,
                                drawableCount, instanceCount, baseInstance);
        return YES;
    }

    NSUInteger pointIndexCount = 0u;
    MGLMetalBufferRef pointIndexBuffer = nil;
    if (mode == GL_TRIANGLE_FAN) {
        pointIndexBuffer = mglNewTriangleFanArrayIndexBuffer(device,
                                                             (NSUInteger)count,
                                                             &pointIndexCount);
    } else if (mode == GL_TRIANGLE_STRIP) {
        pointIndexBuffer = mglNewTriangleStripArrayIndexBuffer(device,
                                                               (NSUInteger)count,
                                                               &pointIndexCount);
    } else if (mode == GL_QUADS) {
        pointIndexBuffer = mglNewQuadArrayIndexBuffer(device,
                                                      (NSUInteger)count,
                                                      &pointIndexCount);
    } else {
        return NO;
    }

    if (!pointIndexBuffer || pointIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s polygon point array emulation failed mode=0x%x count=%d first=%d",
              label ? label : "draw",
              (unsigned)mode,
              (int)count,
              (int)first);
        return NO;
    }

    mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                         MTLPrimitiveTypePoint, pointIndexCount,
                         MTLIndexTypeUInt32, pointIndexBuffer, 0,
                         instanceCount, first, baseInstance);
    return YES;
}

static BOOL mglEncodeElementPolygonPointTarget(MGLMetalRenderCommandEncoderRef encoder,
                                         void *renderEncoderOwner,
                                         MGLMetalDeviceRef device,
                                         Buffer *glElementBuffer,
                                         MGLMetalBufferRef metalElementBuffer,
                                         GLenum mode,
                                         GLenum glIndexType,
                                         MTLIndexType metalIndexType,
                                         NSUInteger indexOffset,
                                         GLsizei count,
                                         NSUInteger instanceCount,
                                         NSInteger baseVertex,
                                         NSUInteger baseInstance,
                                         const char *label)
{
    if (count < 3) {
        return YES;
    }
    if (mode == GL_QUADS && count < 4) {
        return YES;
    }

    if (mode == GL_TRIANGLES) {
        NSUInteger drawableIndexCount = ((NSUInteger)count / 3u) * 3u;
        if (drawableIndexCount == 0u) {
            return YES;
        }

        NSUInteger drawIndexOffset = indexOffset;
        MTLIndexType drawIndexType = metalIndexType;
        MGLMetalBufferRef drawIndexBuffer = mglPreparedElementIndexBuffer(device,
                                                                      glElementBuffer,
                                                                      metalElementBuffer,
                                                                      glIndexType,
                                                                      &drawIndexOffset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer) {
            return NO;
        }

        mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                             MTLPrimitiveTypePoint,
                             drawableIndexCount, drawIndexType,
                             drawIndexBuffer, drawIndexOffset,
                             instanceCount, baseVertex, baseInstance);
        return YES;
    }

    const uint8_t *source = mglElementIndexSourceForDraw(glElementBuffer,
                                                         metalElementBuffer,
                                                         glIndexType,
                                                         indexOffset,
                                                         count);
    NSUInteger pointIndexCount = 0u;
    MGLMetalBufferRef pointIndexBuffer = nil;
    if (mode == GL_TRIANGLE_FAN) {
        pointIndexBuffer = mglNewTriangleFanElementIndexBuffer(device,
                                                               source,
                                                               glIndexType,
                                                               (NSUInteger)count,
                                                               &pointIndexCount);
    } else if (mode == GL_TRIANGLE_STRIP) {
        pointIndexBuffer = mglNewTriangleStripElementIndexBuffer(device,
                                                                 source,
                                                                 glIndexType,
                                                                 (NSUInteger)count,
                                                                 &pointIndexCount);
    } else if (mode == GL_QUADS) {
        pointIndexBuffer = mglNewQuadElementIndexBuffer(device,
                                                        source,
                                                        glIndexType,
                                                        (NSUInteger)count,
                                                        &pointIndexCount);
    } else {
        return NO;
    }

    if (!pointIndexBuffer || pointIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s polygon point element emulation failed mode=0x%x ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              (unsigned)mode,
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              source);
        return NO;
    }

    mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                         MTLPrimitiveTypePoint, pointIndexCount,
                         MTLIndexTypeUInt32, pointIndexBuffer, 0,
                         instanceCount, baseVertex, baseInstance);
    return YES;
}

static BOOL mglEncodeRestartSegmentTarget(MGLMetalRenderCommandEncoderRef encoder,
                                    void *renderEncoderOwner,
                                    MGLMetalDeviceRef device,
                                    Buffer *glElementBuffer,
                                    MGLMetalBufferRef metalElementBuffer,
                                    MGLMetalBufferRef preparedIndexBuffer,
                                    GLenum mode,
                                    MTLPrimitiveType primitiveType,
                                    GLenum glIndexType,
                                    MTLIndexType preparedIndexType,
                                    NSUInteger baseIndexByteOffset,
                                    NSUInteger segmentStart,
                                    NSUInteger segmentIndexCount,
                                    NSUInteger instanceCount,
                                    NSInteger baseVertex,
                                    NSUInteger baseInstance,
                                    BOOL lineMode,
                                    const char *label)
{
    if (!mglPrimitiveModeHasDrawableSegment(mode, segmentIndexCount)) {
        return YES;
    }

    NSUInteger segmentGLByteOffset = 0u;
    NSUInteger indexStride = mglGLIndexElementSize(glIndexType);
    if (!mglComputeIndexByteOffset(baseIndexByteOffset,
                                   segmentStart,
                                   indexStride,
                                   &segmentGLByteOffset)) {
        NSLog(@"MGL WARNING: %s primitive restart segment offset overflow base=%lu start=%lu stride=%lu count=%lu",
              label ? label : "draw",
              (unsigned long)baseIndexByteOffset,
              (unsigned long)segmentStart,
              (unsigned long)indexStride,
              (unsigned long)segmentIndexCount);
        return NO;
    }

    if (primitiveType == MTLPrimitiveTypePoint &&
        (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN || mode == GL_QUADS)) {
        return mglEncodeElementPolygonPointTarget(encoder,
                                            renderEncoderOwner,
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
        return mglEncodeElementTriangleFanTarget(encoder,
                                           renderEncoderOwner,
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
        return mglEncodeElementLineLoopTarget(encoder,
                                        renderEncoderOwner,
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
        return mglEncodeElementQuadsTarget(encoder,
                                     renderEncoderOwner,
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

    NSUInteger preparedByteOffset = 0u;
    if (!mglComputePreparedIndexByteOffset(glIndexType,
                                           segmentGLByteOffset,
                                           &preparedByteOffset)) {
        NSLog(@"MGL WARNING: %s primitive restart prepared offset overflow glType=0x%x byteOffset=%lu",
              label ? label : "draw",
              (unsigned)glIndexType,
              (unsigned long)segmentGLByteOffset);
        return NO;
    }

    mglDrawEncodeIndexed(encoder, renderEncoderOwner,
                         primitiveType, segmentIndexCount,
                         preparedIndexType, preparedIndexBuffer,
                         preparedByteOffset, instanceCount, baseVertex,
                         baseInstance);
    return YES;
}

static MGLPrimitiveRestartEncodeResult mglEncodePrimitiveRestartedElementDrawTarget(MGLMetalRenderCommandEncoderRef encoder,
                                                                              void *renderEncoderOwner,
                                                                              MGLMetalDeviceRef device,
                                                                              GLMContext ctx,
                                                                              Buffer *glElementBuffer,
                                                                              MGLMetalBufferRef metalElementBuffer,
                                                                              GLenum mode,
                                                                              MTLPrimitiveType primitiveType,
                                                                              GLenum glIndexType,
                                                                              MTLIndexType metalIndexType,
                                                                              NSUInteger indexOffset,
                                                                              GLsizei count,
                                                                              NSUInteger instanceCount,
                                                                              NSInteger baseVertex,
                                                                              NSUInteger baseInstance,
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
        NSLog(@"MGL WARNING: %s primitive restart enabled but index bytes are not CPU-readable ebo=%u count=%d type=0x%x offset=%lu; skipping draw to avoid treating restart as a vertex",
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
    NSUInteger restartPositions[256];
    NSUInteger restartPositionCount = 0;  /* total count, may exceed 256 */
    BOOL sawRestart = NO;

    switch (glIndexType) {
        case GL_UNSIGNED_BYTE: {
            const uint8_t *typedSrc = (const uint8_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == (uint8_t)restartIndex) {
                    sawRestart = YES;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (NSUInteger)i;
                    restartPositionCount++;
                }
            }
            break;
        }
        case GL_UNSIGNED_SHORT: {
            const uint16_t *typedSrc = (const uint16_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == (uint16_t)restartIndex) {
                    sawRestart = YES;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (NSUInteger)i;
                    restartPositionCount++;
                }
            }
            break;
        }
        case GL_UNSIGNED_INT: {
            const uint32_t *typedSrc = (const uint32_t *)source;
            for (GLsizei i = 0; i < count; i++) {
                if (typedSrc[i] == restartIndex) {
                    sawRestart = YES;
                    if (restartPositionCount < 256)
                        restartPositions[restartPositionCount] = (NSUInteger)i;
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

    BOOL emulatedMode = (mode == GL_TRIANGLE_FAN ||
                         mode == GL_LINE_LOOP ||
                         mode == GL_QUADS ||
                         (primitiveType == MTLPrimitiveTypePoint &&
                          (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP)));
    MGLMetalBufferRef preparedIndexBuffer = metalElementBuffer;
    MTLIndexType preparedIndexType = metalIndexType;
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

    NSUInteger segmentStart = 0u;
    BOOL encodedAllSegments = YES;

    if (restartPositionCount <= 256) {
        /* Common path: use collected positions — no re-scan needed. */
        for (NSUInteger rp = 0; rp < restartPositionCount; rp++) {
            NSUInteger restartAt = restartPositions[rp];
            NSUInteger segmentCount = restartAt - segmentStart;
            if (!mglEncodeRestartSegmentTarget(encoder,
                                         renderEncoderOwner,
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
                encodedAllSegments = NO;
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
                    NSUInteger segmentCount = (NSUInteger)i - segmentStart;
                    if (!mglEncodeRestartSegmentTarget(encoder,
                                                 renderEncoderOwner,
                                                 device, glElementBuffer, metalElementBuffer,
                                                 preparedIndexBuffer, mode, primitiveType,
                                                 glIndexType, preparedIndexType, indexOffset,
                                                 segmentStart, segmentCount, instanceCount,
                                                 baseVertex, baseInstance,
                                                 mglPolygonModeLineForDrawMode(ctx, mode), label)) {
                        encodedAllSegments = NO;
                        break;
                    }
                    segmentStart = (NSUInteger)i + 1u;
                }
                break;
            }
            case GL_UNSIGNED_SHORT: {
                const uint16_t *typedSrc = (const uint16_t *)source;
                for (GLsizei i = 0; i < count && encodedAllSegments; i++) {
                    if (typedSrc[i] != (uint16_t)restartIndex) continue;
                    NSUInteger segmentCount = (NSUInteger)i - segmentStart;
                    if (!mglEncodeRestartSegmentTarget(encoder,
                                                 renderEncoderOwner,
                                                 device, glElementBuffer, metalElementBuffer,
                                                 preparedIndexBuffer, mode, primitiveType,
                                                 glIndexType, preparedIndexType, indexOffset,
                                                 segmentStart, segmentCount, instanceCount,
                                                 baseVertex, baseInstance,
                                                 mglPolygonModeLineForDrawMode(ctx, mode), label)) {
                        encodedAllSegments = NO;
                        break;
                    }
                    segmentStart = (NSUInteger)i + 1u;
                }
                break;
            }
            case GL_UNSIGNED_INT: {
                const uint32_t *typedSrc = (const uint32_t *)source;
                for (GLsizei i = 0; i < count && encodedAllSegments; i++) {
                    if (typedSrc[i] != restartIndex) continue;
                    NSUInteger segmentCount = (NSUInteger)i - segmentStart;
                    if (!mglEncodeRestartSegmentTarget(encoder,
                                                 renderEncoderOwner,
                                                 device, glElementBuffer, metalElementBuffer,
                                                 preparedIndexBuffer, mode, primitiveType,
                                                 glIndexType, preparedIndexType, indexOffset,
                                                 segmentStart, segmentCount, instanceCount,
                                                 baseVertex, baseInstance,
                                                 mglPolygonModeLineForDrawMode(ctx, mode), label)) {
                        encodedAllSegments = NO;
                        break;
                    }
                    segmentStart = (NSUInteger)i + 1u;
                }
                break;
            }
            default:
                break;
        }
    }

    if (encodedAllSegments) {
        NSUInteger trailingCount = (NSUInteger)count - segmentStart;
        encodedAllSegments = mglEncodeRestartSegmentTarget(encoder,
                                                     renderEncoderOwner,
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

BOOL mglEncodeArrayLineLoop(MGLMetalRenderCommandEncoderRef encoder,
                            GLMContext drawCtx, MGLMetalDeviceRef device,
                            GLsizei count, GLint firstVertex,
                            NSUInteger instanceCount, NSUInteger baseInstance,
                            const char *label)
{
    return mglEncodeArrayLineLoopTarget(
        encoder, NULL, drawCtx, device, count, firstVertex, instanceCount,
        baseInstance, label);
}

BOOL mglEncodeArrayLineLoopForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    GLMContext drawCtx, MGLMetalDeviceRef device, GLsizei count,
    GLint firstVertex, NSUInteger instanceCount, NSUInteger baseInstance,
    const char *label)
{
    return mglEncodeArrayLineLoopTarget(
        encoder, renderEncoderOwner, drawCtx, device, count, firstVertex,
        instanceCount, baseInstance, label);
}

BOOL mglEncodeArrayTriangleFan(MGLMetalRenderCommandEncoderRef encoder,
                               MGLMetalDeviceRef device, GLsizei count,
                               GLint baseVertex, NSUInteger instanceCount,
                               NSUInteger baseInstance, const char *label)
{
    return mglEncodeArrayTriangleFanTarget(
        encoder, NULL, device, count, baseVertex, instanceCount, baseInstance,
        label);
}

BOOL mglEncodeArrayTriangleFanForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, GLsizei count, GLint baseVertex,
    NSUInteger instanceCount, NSUInteger baseInstance, const char *label)
{
    return mglEncodeArrayTriangleFanTarget(
        encoder, renderEncoderOwner, device, count, baseVertex, instanceCount,
        baseInstance, label);
}

BOOL mglEncodeElementLineLoop(MGLMetalRenderCommandEncoderRef encoder,
                              MGLMetalDeviceRef device,
                              Buffer *glElementBuffer,
                              MGLMetalBufferRef metalElementBuffer,
                              GLenum glIndexType, NSUInteger indexOffset,
                              GLsizei count, NSUInteger instanceCount,
                              NSInteger baseVertex, NSUInteger baseInstance,
                              const char *label)
{
    return mglEncodeElementLineLoopTarget(
        encoder, NULL, device, glElementBuffer, metalElementBuffer,
        glIndexType, indexOffset, count, instanceCount, baseVertex,
        baseInstance, label);
}

BOOL mglEncodeElementLineLoopForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, Buffer *glElementBuffer,
    MGLMetalBufferRef metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, const char *label)
{
    return mglEncodeElementLineLoopTarget(
        encoder, renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, glIndexType, indexOffset, count, instanceCount,
        baseVertex, baseInstance, label);
}

BOOL mglEncodeElementTriangleFan(MGLMetalRenderCommandEncoderRef encoder,
                                 MGLMetalDeviceRef device,
                                 Buffer *glElementBuffer,
                                 MGLMetalBufferRef metalElementBuffer,
                                 GLenum glIndexType, NSUInteger indexOffset,
                                 GLsizei count, NSUInteger instanceCount,
                                 NSInteger baseVertex, NSUInteger baseInstance,
                                 const char *label)
{
    return mglEncodeElementTriangleFanTarget(
        encoder, NULL, device, glElementBuffer, metalElementBuffer,
        glIndexType, indexOffset, count, instanceCount, baseVertex,
        baseInstance, label);
}

BOOL mglEncodeElementTriangleFanForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, Buffer *glElementBuffer,
    MGLMetalBufferRef metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, const char *label)
{
    return mglEncodeElementTriangleFanTarget(
        encoder, renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, glIndexType, indexOffset, count, instanceCount,
        baseVertex, baseInstance, label);
}

BOOL mglEncodeArrayQuads(MGLMetalRenderCommandEncoderRef encoder,
                         MGLMetalDeviceRef device, GLsizei count,
                         GLint baseVertex, NSUInteger instanceCount,
                         NSUInteger baseInstance, BOOL lineMode,
                         const char *label)
{
    return mglEncodeArrayQuadsTarget(
        encoder, NULL, device, count, baseVertex, instanceCount, baseInstance,
        lineMode, label);
}

BOOL mglEncodeArrayQuadsForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, GLsizei count, GLint baseVertex,
    NSUInteger instanceCount, NSUInteger baseInstance, BOOL lineMode,
    const char *label)
{
    return mglEncodeArrayQuadsTarget(
        encoder, renderEncoderOwner, device, count, baseVertex, instanceCount,
        baseInstance, lineMode, label);
}

BOOL mglEncodeElementQuads(MGLMetalRenderCommandEncoderRef encoder,
                           MGLMetalDeviceRef device, Buffer *glElementBuffer,
                           MGLMetalBufferRef metalElementBuffer,
                           GLenum glIndexType, NSUInteger indexOffset,
                           GLsizei count, NSUInteger instanceCount,
                           NSInteger baseVertex, NSUInteger baseInstance,
                           BOOL lineMode, const char *label)
{
    return mglEncodeElementQuadsTarget(
        encoder, NULL, device, glElementBuffer, metalElementBuffer,
        glIndexType, indexOffset, count, instanceCount, baseVertex,
        baseInstance, lineMode, label);
}

BOOL mglEncodeElementQuadsForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, Buffer *glElementBuffer,
    MGLMetalBufferRef metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, BOOL lineMode,
    const char *label)
{
    return mglEncodeElementQuadsTarget(
        encoder, renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, glIndexType, indexOffset, count, instanceCount,
        baseVertex, baseInstance, lineMode, label);
}

BOOL mglEncodeArrayPolygonPoint(MGLMetalRenderCommandEncoderRef encoder,
                                MGLMetalDeviceRef device, GLenum mode,
                                GLint first, GLsizei count,
                                NSUInteger instanceCount,
                                NSUInteger baseInstance, const char *label)
{
    return mglEncodeArrayPolygonPointTarget(
        encoder, NULL, device, mode, first, count, instanceCount, baseInstance,
        label);
}

BOOL mglEncodeArrayPolygonPointForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, GLenum mode, GLint first, GLsizei count,
    NSUInteger instanceCount, NSUInteger baseInstance, const char *label)
{
    return mglEncodeArrayPolygonPointTarget(
        encoder, renderEncoderOwner, device, mode, first, count,
        instanceCount, baseInstance, label);
}

BOOL mglEncodeElementPolygonPoint(MGLMetalRenderCommandEncoderRef encoder,
                                  MGLMetalDeviceRef device,
                                  Buffer *glElementBuffer,
                                  MGLMetalBufferRef metalElementBuffer,
                                  GLenum mode, GLenum glIndexType,
                                  MTLIndexType metalIndexType,
                                  NSUInteger indexOffset, GLsizei count,
                                  NSUInteger instanceCount,
                                  NSInteger baseVertex,
                                  NSUInteger baseInstance, const char *label)
{
    return mglEncodeElementPolygonPointTarget(
        encoder, NULL, device, glElementBuffer, metalElementBuffer, mode,
        glIndexType, metalIndexType, indexOffset, count, instanceCount,
        baseVertex, baseInstance, label);
}

BOOL mglEncodeElementPolygonPointForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, Buffer *glElementBuffer,
    MGLMetalBufferRef metalElementBuffer, GLenum mode, GLenum glIndexType,
    MTLIndexType metalIndexType, NSUInteger indexOffset, GLsizei count,
    NSUInteger instanceCount, NSInteger baseVertex,
    NSUInteger baseInstance, const char *label)
{
    return mglEncodeElementPolygonPointTarget(
        encoder, renderEncoderOwner, device, glElementBuffer,
        metalElementBuffer, mode, glIndexType, metalIndexType, indexOffset,
        count, instanceCount, baseVertex, baseInstance, label);
}

BOOL mglEncodeRestartSegment(MGLMetalRenderCommandEncoderRef encoder,
                             MGLMetalDeviceRef device,
                             Buffer *glElementBuffer,
                             MGLMetalBufferRef metalElementBuffer,
                             MGLMetalBufferRef preparedIndexBuffer,
                             GLenum mode, MTLPrimitiveType primitiveType,
                             GLenum glIndexType,
                             MTLIndexType preparedIndexType,
                             NSUInteger baseIndexByteOffset,
                             NSUInteger segmentStart,
                             NSUInteger segmentIndexCount,
                             NSUInteger instanceCount, NSInteger baseVertex,
                             NSUInteger baseInstance, BOOL lineMode,
                             const char *label)
{
    return mglEncodeRestartSegmentTarget(
        encoder, NULL, device, glElementBuffer, metalElementBuffer,
        preparedIndexBuffer, mode, primitiveType, glIndexType,
        preparedIndexType, baseIndexByteOffset, segmentStart,
        segmentIndexCount, instanceCount, baseVertex, baseInstance, lineMode,
        label);
}

MGLPrimitiveRestartEncodeResult mglEncodePrimitiveRestartedElementDraw(
    MGLMetalRenderCommandEncoderRef encoder, MGLMetalDeviceRef device,
    GLMContext ctx, Buffer *glElementBuffer,
    MGLMetalBufferRef metalElementBuffer, GLenum mode,
    MTLPrimitiveType primitiveType, GLenum glIndexType,
    MTLIndexType metalIndexType, NSUInteger indexOffset, GLsizei count,
    NSUInteger instanceCount, NSInteger baseVertex,
    NSUInteger baseInstance, const char *label)
{
    return mglEncodePrimitiveRestartedElementDrawTarget(
        encoder, NULL, device, ctx, glElementBuffer, metalElementBuffer, mode,
        primitiveType, glIndexType, metalIndexType, indexOffset, count,
        instanceCount, baseVertex, baseInstance, label);
}

MGLPrimitiveRestartEncodeResult
mglEncodePrimitiveRestartedElementDrawForRenderEncoderOwner(
    MGLMetalRenderCommandEncoderRef encoder, void *renderEncoderOwner,
    MGLMetalDeviceRef device, GLMContext ctx, Buffer *glElementBuffer,
    MGLMetalBufferRef metalElementBuffer, GLenum mode,
    MTLPrimitiveType primitiveType, GLenum glIndexType,
    MTLIndexType metalIndexType, NSUInteger indexOffset, GLsizei count,
    NSUInteger instanceCount, NSInteger baseVertex,
    NSUInteger baseInstance, const char *label)
{
    return mglEncodePrimitiveRestartedElementDrawTarget(
        encoder, renderEncoderOwner, device, ctx, glElementBuffer,
        metalElementBuffer, mode, primitiveType, glIndexType, metalIndexType,
        indexOffset, count, instanceCount, baseVertex, baseInstance, label);
}

BOOL mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(GLMContext ctx,
                                                                  GLenum glIndexType,
                                                                  const char *label)
{
    uint32_t restartIndex = 0u;
    if (!mglPrimitiveRestartIndexForType(ctx, glIndexType, &restartIndex)) {
        return NO;
    }

    static uint64_t s_indirectRestartSkipCount = 0;
    s_indirectRestartSkipCount++;
    if (s_indirectRestartSkipCount <= 8u || (s_indirectRestartSkipCount % 1000u) == 0u) {
        NSLog(@"MGL WARNING: %s primitive restart with indirect indexed draw is not emulated yet type=0x%x restart=%u occurrence=%llu; skipping draw",
              label ? label : "drawElementsIndirect",
              (unsigned)glIndexType,
              (unsigned)restartIndex,
              (unsigned long long)s_indirectRestartSkipCount);
    }
    return YES;
}

BOOL mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(GLMContext ctx,
                                                               GLenum mode,
                                                               const char *label)
{
    if (!mglPolygonModePointForDrawMode(ctx, mode)) {
        return NO;
    }

    static uint64_t s_indirectPolygonPointSkipCount = 0;
    s_indirectPolygonPointSkipCount++;
    if (s_indirectPolygonPointSkipCount <= 8u || (s_indirectPolygonPointSkipCount % 1000u) == 0u) {
        NSLog(@"MGL WARNING: %s GL_POLYGON_MODE=GL_POINT requires triangle expansion for indirect draw mode=0x%x occurrence=%llu; skipping draw",
              label ? label : "drawIndirect",
              (unsigned)mode,
              (unsigned long long)s_indirectPolygonPointSkipCount);
    }
    return YES;
}
