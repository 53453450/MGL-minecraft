/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Private id/MTL ports implemented in mgl_draw_metal_port.m
 * (O1.6); used by DrawSupport / DrawStageHost / draw runners.
 */
#ifndef MGL_RENDERER_DRAW_SUPPORT_UTIL_H
#define MGL_RENDERER_DRAW_SUPPORT_UTIL_H

#import <Foundation/Foundation.h>
#include "glcorearb.h"
#include "glm_context.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_air_gs_abi.h"
#include "mgl_renderer_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

void *mglDrawSupportBufferContents(id buffer);

uint64_t mglDrawSupportBufferLength(id buffer);

MGLRenderTextureInfo mglDrawSupportTextureInfo(id texture);

BOOL mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext);

id mglDrawSupportCreateBuffer(
    id device,
    NSUInteger length,
    uint64_t options);

id mglDrawSupportCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options);

id mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner);

void mglDrawSupportBlitCopyBuffer(id encoder,
                                         id source,
                                         NSUInteger sourceOffset,
                                         id destination,
                                         NSUInteger destinationOffset,
                                         NSUInteger size);

void mglDrawSupportEndBlitEncoder(id encoder);

void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger index);

void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index);

void mglDrawSupportDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger indexCount,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance);

void mglDrawSupportDrawPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger vertexStart,
    NSUInteger vertexCount,
    NSUInteger instanceCount,
    NSUInteger baseInstance);

void mglDrawSupportDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    id indirectBuffer,
    NSUInteger indirectBufferOffset);

void mglRendererBindCullDistanceEmu(void *renderer, const void *encode_context,
                                    GLenum mode, GLuint first_vertex,
                                    const uint32_t *explicit_vertices,
                                    uint32_t explicit_vertex_count);

id mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner);

void mglDrawSupportSetComputePipeline(
    id encoder,
    id pipeline);

void mglDrawSupportSetComputeBuffer(
    id encoder,
    id buffer,
    NSUInteger offset,
    NSUInteger index);

void mglDrawSupportSetComputeBytes(
    id encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index);

void mglDrawSupportDispatchCompute(
    id encoder,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsX,
    uint32_t threadsY,
    uint32_t threadsZ);

void mglDrawSupportEndComputeEncoder(
    id encoder);

void mglRecordGeometryPrimitiveQueries(
    GLMContext ctx,
    GLuint64 generatedStream0,
    GLuint64 writtenStream0,
    BOOL xfbActive,
    const MGLAIRGSXFBMeta *meta,
    uint32_t streamCount,
    const NSUInteger *bufferWritten,
    const NSUInteger *bufferStride,
    GLuint64 geometryInvocations);

id mglDefaultTessFactorBuffer(id device,
                                                GLMState *state,
                                                GLuint patchCount);

id mglCachedDefaultTessFactorBuffer(
    id device, MGLRendererBackendHandle *backend, GLMState *state,
    GLuint patchCount);

id mglNativeTessFactorBuffer(id device,
                                                id canonical,
                                                GLenum mode,
                                                GLuint patchCount);

GLuint64 mglNativeTessPrimitiveCount(id canonical,
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
