/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Private pointer/MTL ports implemented in mgl_draw_metal_port.m
 * (O1.6); used by DrawSupport / DrawStageHost / draw runners.
 */
#ifndef MGL_RENDERER_DRAW_SUPPORT_UTIL_H
#define MGL_RENDERER_DRAW_SUPPORT_UTIL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "glcorearb.h"
#include "glm_context.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_air_gs_abi.h"
#include "mgl_renderer_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

void *mglDrawSupportBufferContents(void * buffer);

uint64_t mglDrawSupportBufferLength(void * buffer);

MGLRenderTextureInfo mglDrawSupportTextureInfo(void * texture);

bool mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext);

void * mglDrawSupportCreateBuffer(
    void * device,
    size_t length,
    uint64_t options);

void * mglDrawSupportCreateBufferWithBytes(
    void * device,
    const void *bytes,
    size_t length,
    uint64_t options);

void * mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner);

void mglDrawSupportBlitCopyBuffer(void * encoder,
                                         void * source,
                                         size_t sourceOffset,
                                         void * destination,
                                         size_t destinationOffset,
                                         size_t size);

void mglDrawSupportEndBlitEncoder(void * encoder);

void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    void * buffer,
    size_t offset,
    size_t index);

void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    size_t length,
    size_t index);

void mglDrawSupportDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    size_t indexCount,
    void * indexBuffer,
    size_t indexBufferOffset,
    size_t instanceCount,
    int64_t baseVertex,
    size_t baseInstance);

void mglDrawSupportDrawPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    size_t vertexStart,
    size_t vertexCount,
    size_t instanceCount,
    size_t baseInstance);

void mglDrawSupportDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    void * indirectBuffer,
    size_t indirectBufferOffset);

void mglRendererBindCullDistanceEmu(void *renderer, const void *encode_context,
                                    GLenum mode, GLuint first_vertex,
                                    const uint32_t *explicit_vertices,
                                    uint32_t explicit_vertex_count);

void * mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner);

void mglDrawSupportSetComputePipeline(
    void * encoder,
    void * pipeline);

void mglDrawSupportSetComputeBuffer(
    void * encoder,
    void * buffer,
    size_t offset,
    size_t index);

void mglDrawSupportSetComputeBytes(
    void * encoder,
    const void *bytes,
    size_t length,
    size_t index);

void mglDrawSupportDispatchCompute(
    void * encoder,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsX,
    uint32_t threadsY,
    uint32_t threadsZ);

void mglDrawSupportEndComputeEncoder(
    void * encoder);

void mglRecordGeometryPrimitiveQueries(
    GLMContext ctx,
    GLuint64 generatedStream0,
    GLuint64 writtenStream0,
    bool xfbActive,
    const MGLAIRGSXFBMeta *meta,
    uint32_t streamCount,
    const size_t *bufferWritten,
    const size_t *bufferStride,
    GLuint64 geometryInvocations);

void * mglDefaultTessFactorBuffer(void * device,
                                                GLMState *state,
                                                GLuint patchCount);

void * mglCachedDefaultTessFactorBuffer(
    void * device, MGLRendererBackendHandle *backend, GLMState *state,
    GLuint patchCount);

void * mglNativeTessFactorBuffer(void * device,
                                                void * canonical,
                                                GLenum mode,
                                                GLuint patchCount);

GLuint64 mglNativeTessPrimitiveCount(void * canonical,
                                             Program *tesProgram,
                                             GLuint patchCount,
                                             GLuint instanceCount);

void mglDrawSupportCaptureMarkDirtyAll(void *ctx_ptr);

int mglDrawSupportCaptureProcessGL(void *renderer);

int mglDrawSupportCaptureEncoderReady(void *renderer);

void mglDrawSupportCaptureBindSlots(void *renderer, void *capture,
                                           const uint32_t *params);

void mglDrawSupportCaptureSetActive(void *renderer, int active);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_DRAW_SUPPORT_UTIL_H */
