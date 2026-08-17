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

/* Owner-aware primitive emulation. C++ resolves the active encoder from
 * renderEncoderOwner; no borrowed render encoder crosses this API. */
BOOL mglEncodeArrayLineLoopForRenderEncoderOwner(
    void *renderEncoderOwner,
    GLMContext drawCtx, id<MTLDevice> device, GLsizei count,
    GLint firstVertex, NSUInteger instanceCount, NSUInteger baseInstance,
    const char *label);
BOOL mglEncodeArrayTriangleFanForRenderEncoderOwner(
    void *renderEncoderOwner,
    id<MTLDevice> device, GLsizei count, GLint baseVertex,
    NSUInteger instanceCount, NSUInteger baseInstance, const char *label);
BOOL mglEncodeArrayQuadsForRenderEncoderOwner(
    void *renderEncoderOwner,
    id<MTLDevice> device, GLsizei count, GLint baseVertex,
    NSUInteger instanceCount, NSUInteger baseInstance, BOOL lineMode,
    const char *label);
BOOL mglEncodeArrayPolygonPointForRenderEncoderOwner(
    void *renderEncoderOwner,
    id<MTLDevice> device, GLenum mode, GLint first, GLsizei count,
    NSUInteger instanceCount, NSUInteger baseInstance, const char *label);
BOOL mglEncodeElementLineLoopForRenderEncoderOwner(
    void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, const char *label);
BOOL mglEncodeElementTriangleFanForRenderEncoderOwner(
    void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, const char *label);
BOOL mglEncodeElementQuadsForRenderEncoderOwner(
    void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum glIndexType,
    NSUInteger indexOffset, GLsizei count, NSUInteger instanceCount,
    NSInteger baseVertex, NSUInteger baseInstance, BOOL lineMode,
    const char *label);
BOOL mglEncodeElementPolygonPointForRenderEncoderOwner(
    void *renderEncoderOwner,
    id<MTLDevice> device, Buffer *glElementBuffer,
    id<MTLBuffer> metalElementBuffer, GLenum mode, GLenum glIndexType,
    MTLIndexType metalIndexType, NSUInteger indexOffset, GLsizei count,
    NSUInteger instanceCount, NSInteger baseVertex,
    NSUInteger baseInstance, const char *label);
MGLPrimitiveRestartEncodeResult
mglEncodePrimitiveRestartedElementDrawForRenderEncoderOwner(
    void *renderEncoderOwner,
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
