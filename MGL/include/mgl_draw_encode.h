/*
 * mgl_draw_encode.h
 * MGL
 *
 * Draw Encode Subsystem.
 *
 * GL primitive-mode emulation encoders that translate non-Metal primitive
 * types (GL_TRIANGLE_FAN, GL_LINE_LOOP, GL_QUADS) and primitive-restart
 * draws into indexed Metal draws (triangle lists / line lists / point lists).
 *
 * All functions take encoder/device/ctx as parameters (no self/ivar access).
 * They depend on mgl_index_buffer.h (index buffer builders) and
 * mgl_draw_mode.h (polygon-mode classification).
 *
 * Dependencies: Metal.framework (id<MTLDevice>, id<MTLBuffer>,
 * id<MTLRenderCommandEncoder>, MTLPrimitiveType) + glm_context.h
 * (GLMContext, mglDispatchError) + mgl_index_buffer.h + mgl_draw_mode.h.
 */

#ifndef MGL_DRAW_ENCODE_H
#define MGL_DRAW_ENCODE_H

#include "glcorearb.h"

#include <stdbool.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#include "glm_context.h"
#include "mgl_index_buffer.h"
#include "mgl_draw_mode.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Result of primitive-restart encoding. */
typedef enum MGLPrimitiveRestartEncodeResult {
    MGLPrimitiveRestartEncodeNotNeeded = 0,
    MGLPrimitiveRestartEncodeHandled = 1,
    MGLPrimitiveRestartEncodeFailed = 2,
} MGLPrimitiveRestartEncodeResult;

#ifdef __OBJC__

/* === Array (non-indexed) emulators === */

BOOL mglEncodeArrayLineLoop(id<MTLRenderCommandEncoder> encoder,
                            GLMContext drawCtx,
                            id<MTLDevice> device,
                            GLsizei count,
                            GLint firstVertex,
                            NSUInteger instanceCount,
                            NSUInteger baseInstance,
                            const char *label);

BOOL mglEncodeArrayTriangleFan(id<MTLRenderCommandEncoder> encoder,
                                id<MTLDevice> device,
                                GLsizei count,
                                GLint baseVertex,
                                NSUInteger instanceCount,
                                NSUInteger baseInstance,
                                const char *label);

BOOL mglEncodeArrayQuads(id<MTLRenderCommandEncoder> encoder,
                         id<MTLDevice> device,
                         GLsizei count,
                         GLint baseVertex,
                         NSUInteger instanceCount,
                         NSUInteger baseInstance,
                         BOOL lineMode,
                         const char *label);

BOOL mglEncodeArrayPolygonPoint(id<MTLRenderCommandEncoder> encoder,
                                id<MTLDevice> device,
                                GLenum mode,
                                GLint first,
                                GLsizei count,
                                NSUInteger instanceCount,
                                NSUInteger baseInstance,
                                const char *label);

/* === Element (indexed) emulators === */

BOOL mglEncodeElementLineLoop(id<MTLRenderCommandEncoder> encoder,
                              id<MTLDevice> device,
                              Buffer *glElementBuffer,
                              id<MTLBuffer> metalElementBuffer,
                              GLenum glIndexType,
                              NSUInteger indexOffset,
                              GLsizei count,
                              NSUInteger instanceCount,
                              NSInteger baseVertex,
                              NSUInteger baseInstance,
                              const char *label);

BOOL mglEncodeElementTriangleFan(id<MTLRenderCommandEncoder> encoder,
                                  id<MTLDevice> device,
                                  Buffer *glElementBuffer,
                                  id<MTLBuffer> metalElementBuffer,
                                  GLenum glIndexType,
                                  NSUInteger indexOffset,
                                  GLsizei count,
                                  NSUInteger instanceCount,
                                  NSInteger baseVertex,
                                  NSUInteger baseInstance,
                                  const char *label);

BOOL mglEncodeElementQuads(id<MTLRenderCommandEncoder> encoder,
                           id<MTLDevice> device,
                           Buffer *glElementBuffer,
                           id<MTLBuffer> metalElementBuffer,
                           GLenum glIndexType,
                           NSUInteger indexOffset,
                           GLsizei count,
                           NSUInteger instanceCount,
                           NSInteger baseVertex,
                           NSUInteger baseInstance,
                           BOOL lineMode,
                           const char *label);

BOOL mglEncodeElementPolygonPoint(id<MTLRenderCommandEncoder> encoder,
                                  id<MTLDevice> device,
                                  Buffer *glElementBuffer,
                                  id<MTLBuffer> metalElementBuffer,
                                  GLenum mode,
                                  GLenum glIndexType,
                                  MTLIndexType metalIndexType,
                                  NSUInteger indexOffset,
                                  GLsizei count,
                                  NSUInteger instanceCount,
                                  NSInteger baseVertex,
                                  NSUInteger baseInstance,
                                  const char *label);

/* === Primitive restart === */

BOOL mglEncodeRestartSegment(id<MTLRenderCommandEncoder> encoder,
                             id<MTLDevice> device,
                             Buffer *glElementBuffer,
                             id<MTLBuffer> metalElementBuffer,
                             id<MTLBuffer> preparedIndexBuffer,
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
                             const char *label);

MGLPrimitiveRestartEncodeResult mglEncodePrimitiveRestartedElementDraw(id<MTLRenderCommandEncoder> encoder,
                                                                       id<MTLDevice> device,
                                                                       GLMContext ctx,
                                                                       Buffer *glElementBuffer,
                                                                       id<MTLBuffer> metalElementBuffer,
                                                                       GLenum mode,
                                                                       MTLPrimitiveType primitiveType,
                                                                       GLenum glIndexType,
                                                                       MTLIndexType metalIndexType,
                                                                       NSUInteger indexOffset,
                                                                       GLsizei count,
                                                                       NSUInteger instanceCount,
                                                                       NSInteger baseVertex,
                                                                       NSUInteger baseInstance,
                                                                       const char *label);

/* Owner-aware variants used by the gate-on renderer path. The encoder argument
 * is retained only for the gate-off adapter; C++ resolves the active encoder
 * from renderEncoderOwner when MGL_USE_METALCPP is enabled. */
BOOL mglEncodeArrayLineLoopForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    GLMContext drawCtx, id<MTLDevice> device, GLsizei count,
    GLint firstVertex, NSUInteger instanceCount, NSUInteger baseInstance,
    const char *label);
BOOL mglEncodeArrayTriangleFanForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, GLsizei count, GLint baseVertex,
    NSUInteger instanceCount, NSUInteger baseInstance, const char *label);
BOOL mglEncodeArrayQuadsForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, GLsizei count, GLint baseVertex,
    NSUInteger instanceCount, NSUInteger baseInstance, BOOL lineMode,
    const char *label);
BOOL mglEncodeArrayPolygonPointForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, GLenum mode, GLint first, GLsizei count,
    NSUInteger instanceCount, NSUInteger baseInstance, const char *label);
BOOL mglEncodeElementLineLoopForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, const char *label);
BOOL mglEncodeElementTriangleFanForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, const char *label);
BOOL mglEncodeElementQuadsForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, BOOL lineMode,
    const char *label);
BOOL mglEncodeElementPolygonPointForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum mode, GLenum glIndexType,
    MTLIndexType metalIndexType, NSUInteger indexOffset, GLsizei count,
    NSUInteger instanceCount, NSInteger baseVertex,
    NSUInteger baseInstance, const char *label);
MGLPrimitiveRestartEncodeResult
mglEncodePrimitiveRestartedElementDrawForRenderEncoderOwner(
    id<MTLRenderCommandEncoder> encoder, void *renderEncoderOwner,
    id<MTLDevice> device, GLMContext ctx, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum mode,
    MTLPrimitiveType primitiveType, GLenum glIndexType,
    MTLIndexType metalIndexType, NSUInteger indexOffset, GLsizei count,
    NSUInteger instanceCount, NSInteger baseVertex,
    NSUInteger baseInstance, const char *label);

/* === Indirect-draw skip checks === */

BOOL mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(GLMContext ctx,
                                                            GLenum glIndexType,
                                                            const char *label);

BOOL mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(GLMContext ctx,
                                                         GLenum mode,
                                                         const char *label);

#endif /* __OBJC__ */

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_ENCODE_H */
