/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/* O1.6: true id/MTL materialization ports shared by draw host runners. */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+DrawSupportUtil.h"
#include "mgl_draw_tess.h"
#include "mgl_draw_gs.h"
#include "mgl_draw_encode.h"
#include "mgl_air_gs_abi.h"
#include "mgl_shader_abi.h"
#include "mgl_renderer_backend.h"

void *mglDrawSupportBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0;
    if (!buffer || mglRenderGetBufferContents(
            (__bridge void *)buffer, &contents, &length) != 0) {
        return NULL;
    }
    return contents;
}

/* AIR stage-out / GS scatter slots carry integers as SIToFP/UIToFP floats.
 * GL transform-feedback stores native int/uint bits — decode before the
 * compact XFB image is published to the GL buffer / CPU shadow.  Packing
 * lives in mglTessPackXFBFieldFromCarrier / mglXfbDecodeIntCarriersInBytes. */

uint64_t mglDrawSupportBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(
        (__bridge void *)buffer, &info) == 0 ? info.length : 0u;
}

MGLRenderTextureInfo mglDrawSupportTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

BOOL mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext)
{
    if (!encodeContext) return NO;
    return mglRenderEncoderOwnerHasCurrent(
        encodeContext->render_encoder_owner) == 1;
}


/* mglGeometryGatherIndices → mgl_draw_tess.cpp (O1.4) */

id mglDrawSupportCreateBuffer(
    id device,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

id mglDrawSupportCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

id mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

void mglDrawSupportBlitCopyBuffer(id encoder,
                                         id source,
                                         NSUInteger sourceOffset,
                                         id destination,
                                         NSUInteger destinationOffset,
                                         NSUInteger size)
{
    (void)mglRenderBlitCopyBuffer(
        (__bridge void *)encoder, (__bridge void *)source, sourceOffset,
        (__bridge void *)destination, destinationOffset, size);
}

void mglDrawSupportEndBlitEncoder(id encoder)
{
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

void mglDrawSupportDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger indexCount,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    (void)mglRenderEncodeDrawForRenderEncoderOwner(renderEncoderOwner,
        &(MGLRenderDrawPlan){
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)MGL_DRAW_INDEX_UINT32,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        }, NULL, 0);
}

void mglDrawSupportDrawPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger vertexStart,
    NSUInteger vertexCount,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

void mglDrawSupportDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    id indirectBuffer,
    NSUInteger indirectBufferOffset)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY_INDIRECT,
            .primitive_type = (uint32_t)primitiveType,
            .indirect_buffer = (__bridge void *)indirectBuffer,
            .indirect_buffer_offset = indirectBufferOffset,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

void mglRendererBindCullDistanceEmu(void *renderer, const void *encode_context,
                                    GLenum mode, GLuint first_vertex,
                                    const uint32_t *explicit_vertices,
                                    uint32_t explicit_vertex_count)
{
    if (!renderer || !encode_context) {
        return;
    }
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    [host bindCullDistanceEmulationBuffers:mode
                                firstVertex:first_vertex
                           explicitVertices:explicit_vertices
                         explicitVertexCount:explicit_vertex_count
                              encodeContext:(const MGLEncodeContext *)encode_context];
}

id mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateComputeEncoderBorrowed(
        commandBufferOwner);
}

void mglDrawSupportSetComputePipeline(
    id encoder,
    id pipeline)
{
    (void)mglRenderSetComputePipelineState((__bridge void *)encoder,
                                              (__bridge void *)pipeline);
}

void mglDrawSupportSetComputeBuffer(
    id encoder,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetComputeBuffer((__bridge void *)encoder,
                                       (__bridge void *)buffer, offset,
                                       (uint32_t)index);
}

void mglDrawSupportSetComputeBytes(
    id encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetComputeBytes((__bridge void *)encoder, bytes,
                                      length, (uint32_t)index);
}

void mglDrawSupportDispatchCompute(
    id encoder,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsX,
    uint32_t threadsY,
    uint32_t threadsZ)
{
    (void)mglRenderDispatchCompute(
        (__bridge void *)encoder, groupsX, groupsY, groupsZ,
        threadsX, threadsY, threadsZ);
}

void mglDrawSupportEndComputeEncoder(
    id encoder)
{
    (void)mglRenderEndComputeEncoder((__bridge void *)encoder);
}

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx,
                                               GLuint64 generated,
                                               GLuint64 written);
extern void mglRecordActivePrimitiveQueryDrawIndexed(GLMContext ctx,
                                                      GLuint index,
                                                      GLuint64 generated,
                                                      GLuint64 written);
extern void mglRecordActiveGeometryShaderQueryDraw(GLMContext ctx,
                                                    GLuint64 invocations,
                                                    GLuint64 primitives);
extern GLboolean mglHasActiveIndexedPrimitiveQuery(GLMContext ctx);
extern GLboolean mglHasActivePrimitiveQuery(GLMContext ctx);
extern GLboolean mglHasActiveGeometryShaderQuery(GLMContext ctx);

void mglRecordGeometryPrimitiveQueries(
    GLMContext ctx,
    GLuint64 generatedStream0,
    GLuint64 writtenStream0,
    BOOL xfbActive,
    const MGLAIRGSXFBMeta *meta,
    uint32_t streamCount,
    const NSUInteger *bufferWritten,
    const NSUInteger *bufferStride,
    GLuint64 geometryInvocations)
{
    mglRecordActiveGeometryShaderQueryDraw(
        ctx, geometryInvocations, generatedStream0);
    mglRecordActivePrimitiveQueryDraw(
        ctx, generatedStream0,
        mglDrawGsStream0QueryWritten(xfbActive ? 1 : 0, writtenStream0));
    if (!meta || !bufferWritten || !bufferStride) return;
    streamCount = mglDrawGsClampStreamCount(streamCount);
    for (uint32_t s = 1u; s < streamCount; s++) {
        /* Indexed stream s query: generated stays in the meta; written is
         * the ordered scatter's whole-primitive bytes for buffer s divided
         * by its per-record stride (streams > 0 are points, vpp = 1). */
        GLuint64 written = mglDrawGsIndexedStreamWritten(
            xfbActive ? 1 : 0, (uint64_t)bufferWritten[s],
            (uint64_t)bufferStride[s]);
        mglRecordActivePrimitiveQueryDrawIndexed(
            ctx, s, (GLuint64)meta->stream[s].generated, written);
    }
}

id mglDefaultTessFactorBuffer(id device,
                                                GLMState *state,
                                                GLuint patchCount)
{
    if (!device || !state || patchCount == 0u) return nil;
    uint64_t factorBytes = 0u;
    if (!mglTessPlanDefaultFactorBytes(patchCount, &factorBytes)) return nil;
    id buffer = mglDrawSupportCreateBuffer(
        device, (NSUInteger)factorBytes, 0u);
    if (!buffer || !mglDrawSupportBufferContents(buffer)) return nil;

    if (mglRenderFillDefaultTessFactorBuffer(
            (void *)mglDrawSupportBufferContents(buffer),
            factorBytes,
            state->var.patch_default_outer_level,
            state->var.patch_default_inner_level,
            patchCount) != 0) {
        return nil;
    }
    return buffer;
}

/* Cached variant of the default factor buffer for the TES-only path:
 * consecutive tess draws reuse one stable allocation unless the default
 * patch levels or patch count actually changed. */
id mglCachedDefaultTessFactorBuffer(
    id device, MGLRendererBackendHandle *backend, GLMState *state,
    GLuint patchCount)
{
    if (!device || !backend || !state || patchCount == 0u) return nil;
    float levels[6];
    mglTessFillDefaultFactorLevels(state->var.patch_default_outer_level,
                                   state->var.patch_default_inner_level,
                                   levels);
    void *cached = NULL;
    if (mglRendererBackendGetTessFactorBuffer(
            backend, patchCount, levels, &cached) == 1 && cached) {
        return (__bridge id)cached;
    }
    id fresh = mglDefaultTessFactorBuffer(device, state, patchCount);
    if (!fresh) return nil;
    if (mglRendererBackendPutTessFactorBuffer(
            backend, patchCount, levels, (__bridge void *)fresh) != 0) {
        return fresh;
    }
    return fresh;
}

id mglNativeTessFactorBuffer(id device,
                                                id canonical,
                                                GLenum mode,
                                                GLuint patchCount)
{
    if (!device || !canonical || !mglDrawSupportBufferContents(canonical) ||
        patchCount == 0u) {
        return nil;
    }
    uint32_t repackBytes = 0u;
    const int factorKind = mglTessPlanNativeFactor(
        (uint32_t)mode, (uint64_t)mglDrawSupportBufferLength(canonical),
        patchCount, &repackBytes);
    if (factorKind == MGL_TESS_NATIVE_FACTOR_REUSE) {
        return canonical;
    }
    if (factorKind != MGL_TESS_NATIVE_FACTOR_REPACK_TRI) {
        return nil;
    }

    id result = mglDrawSupportCreateBuffer(device, (NSUInteger)repackBytes, 0u);
    if (!result || !mglDrawSupportBufferContents(result)) {
        return nil;
    }

    if (mglRenderRepackTessFactorTriangles(
            (const void *)mglDrawSupportBufferContents(canonical), (uint64_t)mglDrawSupportBufferLength(canonical),
            (void *)mglDrawSupportBufferContents(result),
            (uint64_t)repackBytes,
            patchCount) != 0) {
        return nil;
    }
    return result;
}

GLuint64 mglNativeTessPrimitiveCount(id canonical,
                                             Program *tesProgram,
                                             GLuint patchCount,
                                             GLuint instanceCount)
{
    if (!canonical || !mglDrawSupportBufferContents(canonical) || !tesProgram || patchCount == 0u) {
        return 0u;
    }

    return mglTessGeneratedPrimitiveCount(
        tesProgram, (const void *)mglDrawSupportBufferContents(canonical),
        patchCount, instanceCount);
}


/* O1.2/O1.1: C ports for mglTessRunCaptureSession host ops. */
void mglDrawSupportCaptureMarkDirtyAll(void *ctx_ptr)
{
    GLMContext drawCtx = (GLMContext)ctx_ptr;
    if (drawCtx && drawCtx->active_state) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
    }
}

int mglDrawSupportCaptureProcessGL(void *renderer)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    return host && [host processGLState:true] ? 1 : 0;
}

int mglDrawSupportCaptureEncoderReady(void *renderer)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (!host) return 0;
    return mglRenderEncoderOwnerHasCurrent(
               host->_renderPassManager.state->currentRenderEncoderOwner) == 1
               ? 1
               : 0;
}

void mglDrawSupportCaptureBindSlots(void *renderer, void *capture,
                                           const uint32_t *params)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (!host || !capture || !params) return;
    mglTessBindCaptureSlots(
        host->_renderPassManager.state->currentRenderEncoderOwner, capture,
        params);
}

void mglDrawSupportCaptureSetActive(void *renderer, int active)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (host) {
        host->_tessellation.tessVertexCaptureActive = active ? YES : NO;
    }
}

