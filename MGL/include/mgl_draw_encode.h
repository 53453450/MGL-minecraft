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
 * Dependencies: opaque Metal handles and numeric Metal enum values + glm_context.h
 * (GLMContext, mglDispatchError) + mgl_index_buffer.h + mgl_draw_mode.h.
 */

#ifndef MGL_DRAW_ENCODE_H
#define MGL_DRAW_ENCODE_H

#include "glcorearb.h"

#include <stdbool.h>

#include "glm_context.h"
#include "mgl_index_buffer.h"
#include "mgl_draw_mode.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __OBJC__
typedef id MGLDrawMetalHandle;
#else
typedef void *MGLDrawMetalHandle;
#endif

enum {
    MGL_DRAW_PRIMITIVE_POINT = 0,
    MGL_DRAW_PRIMITIVE_LINE = 1,
    MGL_DRAW_PRIMITIVE_LINE_STRIP = 2,
    MGL_DRAW_PRIMITIVE_TRIANGLE = 3,
    MGL_DRAW_PRIMITIVE_TRIANGLE_STRIP = 4,
};

enum {
    MGL_DRAW_INDEX_UINT16 = 0,
    MGL_DRAW_INDEX_UINT32 = 1,
};

/* Metal indirect-draw buffer layout expressed as pure C value-state. */
typedef struct MGLDrawPrimitivesIndirectArguments_t {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t vertexStart;
    uint32_t baseInstance;
} MGLDrawPrimitivesIndirectArguments;

typedef struct MGLDrawIndexedPrimitivesIndirectArguments_t {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t indexStart;
    int32_t baseVertex;
    uint32_t baseInstance;
} MGLDrawIndexedPrimitivesIndirectArguments;

/* Result of primitive-restart encoding. */
typedef enum MGLPrimitiveRestartEncodeResult {
    MGLPrimitiveRestartEncodeNotNeeded = 0,
    MGLPrimitiveRestartEncodeHandled = 1,
    MGLPrimitiveRestartEncodeFailed = 2,
} MGLPrimitiveRestartEncodeResult;

/* Owner-aware primitive emulation. C++ resolves the active encoder from
 * renderEncoderOwner; no borrowed render encoder crosses this API. */
bool mglEncodeArrayLineLoopForRenderEncoderOwner(
    void *renderEncoderOwner,
    GLMContext drawCtx, MGLDrawMetalHandle device, GLsizei count,
    GLint firstVertex, size_t instanceCount, size_t baseInstance,
    const char *label);
bool mglEncodeArrayTriangleFanForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLsizei count, GLint baseVertex,
    size_t instanceCount, size_t baseInstance, const char *label);
bool mglEncodeArrayQuadsForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLsizei count, GLint baseVertex,
    size_t instanceCount, size_t baseInstance, bool lineMode,
    const char *label);
bool mglEncodeArrayPolygonPointForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLenum mode, GLint first, GLsizei count,
    size_t instanceCount, size_t baseInstance, const char *label);
bool mglEncodeElementLineLoopForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum glIndexType,
    size_t indexOffset, GLsizei count, size_t instanceCount,
    int64_t baseVertex, size_t baseInstance, const char *label);
bool mglEncodeElementTriangleFanForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum glIndexType,
    size_t indexOffset, GLsizei count, size_t instanceCount,
    int64_t baseVertex, size_t baseInstance, const char *label);
bool mglEncodeElementQuadsForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum glIndexType,
    size_t indexOffset, GLsizei count, size_t instanceCount,
    int64_t baseVertex, size_t baseInstance, bool lineMode,
    const char *label);
bool mglEncodeElementPolygonPointForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum mode, GLenum glIndexType,
    uint32_t metalIndexType, size_t indexOffset, GLsizei count,
    size_t instanceCount, int64_t baseVertex,
    size_t baseInstance, const char *label);
MGLPrimitiveRestartEncodeResult
mglEncodePrimitiveRestartedElementDrawForRenderEncoderOwner(
    void *renderEncoderOwner,
    MGLDrawMetalHandle device, GLMContext ctx, Buffer *glElementBuffer,
    MGLDrawMetalHandle metalElementBuffer, GLenum mode,
    uint32_t primitiveType, GLenum glIndexType,
    uint32_t metalIndexType, size_t indexOffset, GLsizei count,
    size_t instanceCount, int64_t baseVertex,
    size_t baseInstance, const char *label);

/* === Indirect-draw skip checks === */

bool mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(GLMContext ctx,
                                                            GLenum glIndexType,
                                                            const char *label);

bool mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(GLMContext ctx,
                                                         GLenum mode,
                                                         const char *label);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_ENCODE_H */
