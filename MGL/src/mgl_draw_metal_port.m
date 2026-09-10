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
#import "MGLRenderer+Tessellation_Private.h"
#include "mgl_draw_tess.h"

#include "mgl_draw_cull.h"
#include "mgl_draw_issue.h"
#include "mgl_index_buffer.h"
#include "mgl_buffer_query.h"
#include "glm_limits.h"
#include "mgl_frame_activity.h"
#include "mgl_draw_gs.h"
#include "mgl_draw_encode.h"
#include "mgl_air_gs_abi.h"
#include "mgl_shader_abi.h"
#include "mgl_renderer_backend.h"
#include <string.h>

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
               mglRendererRenderPassManager(host).state->currentRenderEncoderOwner) == 1
               ? 1
               : 0;
}

void mglDrawSupportCaptureBindSlots(void *renderer, void *capture,
                                           const uint32_t *params)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (!host || !capture || !params) return;
    mglTessBindCaptureSlots(
        mglRendererRenderPassManager(host).state->currentRenderEncoderOwner, capture,
        params);
}

void mglDrawSupportCaptureSetActive(void *renderer, int active)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (host) {
        host->_tessellation.tessVertexCaptureActive = active ? YES : NO;
    }
}

/* ---- O1.4 HostOps ports (thin MTL / renderer ivar materialization) ---- */

static MGLRenderer *mglStageHostSelf(void *renderer)
{
    return renderer ? (__bridge MGLRenderer *)renderer : nil;
}

/* C-ABI accessors for the Metal-facing sub-objects (see declarations in
 * MGLRenderer_Private.h). Defined here so they compile with full knowledge of
 * the MGLRenderer class extension; consumed by file-scope C functions in this
 * and the other encode ports. */
MGLRenderPassManager *mglRendererRenderPassManager(MGLRenderer *r)
{
    return r ? r->_renderPassManager : nil;
}

MGLRendererBackendHandle *mglRendererBackend(MGLRenderer *r)
{
    return r ? r->_backend : NULL;
}

static void mglStageMarkCbHasWork(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) self->_currentCBHasWork = YES;
}

static void mglStageFlushCB(void *renderer, int wait)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) [self flushCommandBuffer:wait ? YES : NO];
}

static void *mglStageBufContents(void *buffer)
{
    return mglDrawSupportBufferContents((__bridge id)buffer);
}

static void *mglDrawPortVertexCaptureArray(void *renderer, GLMContext ctx,
                                             GLint first, GLsizei count,
                                             GLsizei instanceCount,
                                             GLuint baseInstance,
                                             uint64_t *out_offset); /* below */

static void *mglStageCaptureArray(void *renderer, GLMContext ctx, GLint first,
                                  GLsizei count, GLsizei instanceCount,
                                  GLuint baseInstance, uint64_t *out_offset)
{
    return mglDrawPortVertexCaptureArray(renderer, ctx, first, count,
                                         instanceCount, baseInstance,
                                         out_offset);
}

static void *mglDrawPortVertexCaptureIndexed(
    void *renderer, GLMContext ctx, void *index_mtl, GLenum indexType,
    uint64_t index_offset, GLsizei count, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, uint32_t maxIndex,
    uint64_t *out_offset); /* below */

static void *mglStageCaptureIndexed(void *renderer, GLMContext ctx, void *index_mtl,
                                    GLenum indexType, uint64_t index_offset,
                                    GLsizei count, GLint baseVertex,
                                    GLsizei instanceCount, GLuint baseInstance,
                                    uint32_t maxIndex, uint64_t *out_offset)
{
    return mglDrawPortVertexCaptureIndexed(renderer, ctx, index_mtl, indexType,
                                           index_offset, count, baseVertex,
                                           instanceCount, baseInstance, maxIndex,
                                           out_offset);
}

static int mglStageBindProgram(void *renderer, Program *program)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self bindMTLProgram:program] ? 1 : 0;
}

static int mglStageProcessBuffer(void *renderer, Buffer *buf)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self processBuffer:buf] ? 1 : 0;
}

static void *mglStageCreateBuffer(void *renderer, uint64_t length)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    id buf = mglDrawSupportCreateBuffer(((__bridge id)mglRendererBackendGetDevice(self->_backend)), (NSUInteger)length, 0u);
    return (__bridge_retained void *)buf;
}

static void *mglStageCreateBufferBytes(void *renderer, const void *bytes,
                                       uint64_t length)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    id buf = mglDrawSupportCreateBufferWithBytes(((__bridge id)mglRendererBackendGetDevice(self->_backend)), bytes,
                                                 (NSUInteger)length, 0u);
    return (__bridge_retained void *)buf;
}

static void *mglStageCachedFactors(void *renderer, GLMContext ctx,
                                   uint32_t patch_count)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !ctx) return NULL;
    id buf = mglCachedDefaultTessFactorBuffer(((__bridge id)mglRendererBackendGetDevice(self->_backend)), self->_backend,
                                              ctx->active_state, patch_count);
    /* Cached on backend — borrow only. */
    return (__bridge void *)buf;
}

static void *mglStageNativeFactors(void *renderer, void *canonical, GLenum mode,
                                   uint32_t patch_count)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    id buf = mglNativeTessFactorBuffer(((__bridge id)mglRendererBackendGetDevice(self->_backend)), (__bridge id)canonical,
                                       mode, patch_count);
    return (__bridge_retained void *)buf;
}

static int mglStageDispatchTCS(void *renderer, GLMContext ctx, Program *tcs,
                               MGLAIRTessDrawContract *contract)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self dispatchTessControlShader:ctx program:tcs
                                          contract:contract]
               ? 1
               : 0;
}

static int mglStageDispatchAirTES(void *renderer, GLMContext ctx, Program *tes,
                                  MGLAIRTessDrawContract *contract,
                                  uint32_t patch_count, GLsizei instanceCount,
                                  GLuint baseInstance)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self dispatchAIRTessEvalCompute:ctx
                                           program:tes
                                          contract:contract
                                        patchCount:patch_count
                                     instanceCount:instanceCount
                                      baseInstance:baseInstance]
               ? 1
               : 0;
}

static int mglStageDispatchAirTESVertex(void *renderer, GLMContext ctx,
                                        Program *tes,
                                        MGLAIRTessDrawContract *contract,
                                        uint32_t patch_count, GLsizei instanceCount,
                                        GLuint baseInstance)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self dispatchAIRTessEvalVertexRender:ctx
                                                program:tes
                                               contract:contract
                                             patchCount:patch_count
                                          instanceCount:instanceCount
                                           baseInstance:baseInstance]
               ? 1
               : 0;
}

static int mglStageProcessGL(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self processGLState:true] ? 1 : 0;
}

static int mglStageEncoderHasCurrent(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return 0;
    return mglRenderEncoderOwnerHasCurrent(
               self->_renderPassManager.state->currentRenderEncoderOwner)
               ? 1
               : 0;
}

static int mglStageRasterEmpty(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self currentDrawRasterizationIsEmpty] ? 1 : 0;
}

static int mglStageFullyCulled(void *renderer, GLenum mode)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self currentDrawModeIsFullyCulled:mode] ? 1 : 0;
}

static void mglStageApplyPolygonOffset(void *renderer, GLenum mode)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) [self applyPolygonOffsetForDrawMode:mode];
}

static void mglStageEndRender(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) [self endRenderEncoding];
}

static void mglStageClearNativeCB(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) [self clearStageBindingCopyBacks:&self->_tessellation.nativeTESCopyBacks];
}

static int mglStageFlushNativeCB(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self flushStageBindingCopyBacks:&self->_tessellation.nativeTESCopyBacks
                                   requireCPUVisibility:NO]
               ? 1
               : 0;
}

static void mglStageBeginNativeTES(void *renderer, Program *tes)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    self->_tessellation.nativeTESProgram = tes;
    self->_tessellation.nativeTESActive = YES;
}

static void mglStageEndNativeTES(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    self->_tessellation.nativeTESActive = NO;
    self->_tessellation.nativeTESProgram = NULL;
}

static void mglStageResetTessDrawState(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    (void)mglRendererBackendSetTessVertexCaptureBuffer(self->_backend, NULL);
    self->_tessellation.tessVertexCaptureOffset = 0u;
    (void)mglRendererBackendSetTessControlPointIndexBuffer(self->_backend, NULL);
    self->_tessellation.tessIndexedDraw = NO;
    self->_tessellation.tessInstanceRecords = 0u;
    (void)mglRendererBackendSetTcsOutputBuffer(self->_backend, NULL);
    self->_tessellation.tcsOutputOffset = 0u;
    self->_tessellation.tcsOutputStride = 0u;
    self->_tessellation.tcsOutVertices = 0u;
    (void)mglRendererBackendSetCurrentTessFactorBuffer(self->_backend, NULL);
}

static void mglStageSetTessCapture(void *renderer, void *buf, uint64_t offset,
                                   uint64_t instance_records, int indexed)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    (void)mglRendererBackendSetTessVertexCaptureBuffer(self->_backend, buf);
    self->_tessellation.tessVertexCaptureOffset = (NSUInteger)offset;
    self->_tessellation.tessIndexedDraw = indexed ? YES : NO;
    self->_tessellation.tessInstanceRecords = (NSUInteger)instance_records;
    if (!buf) {
        self->_tessellation.tessIndexedDraw = NO;
        self->_tessellation.tessInstanceRecords = 0u;
        self->_tessellation.tessVertexCaptureOffset = 0u;
        (void)mglRendererBackendSetTessControlPointIndexBuffer(self->_backend,
                                                               NULL);
    }
}

static void mglStageSetControlPointIndex(void *renderer, void *gather)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    (void)mglRendererBackendSetTessControlPointIndexBuffer(self->_backend,
                                                           gather);
}

static void mglStageAdoptCaptureAsTCS(void *renderer, uint32_t stride,
                                      uint32_t out_vertices)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    void *cap = mglRendererBackendGetTessVertexCaptureBuffer(self->_backend);
    (void)mglRendererBackendSetTcsOutputBuffer(self->_backend, cap);
    self->_tessellation.tcsOutputOffset =
        self->_tessellation.tessVertexCaptureOffset;
    self->_tessellation.tcsOutputStride = stride;
    self->_tessellation.tcsOutVertices = out_vertices;
}

static void mglStageSetCurrentFactors(void *renderer, void *factors)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    (void)mglRendererBackendSetCurrentTessFactorBuffer(self->_backend, factors);
}

static void *mglStageGetTessCapture(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? mglRendererBackendGetTessVertexCaptureBuffer(self->_backend)
                : NULL;
}

static void *mglStageGetTcsOutput(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? mglRendererBackendGetTcsOutputBuffer(self->_backend) : NULL;
}

static void *mglStageGetFactors(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? mglRendererBackendGetCurrentTessFactorBuffer(self->_backend)
                : NULL;
}

static void *mglStageGetPatchOut(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? mglRendererBackendGetTcsPatchOutBuffer(self->_backend) : NULL;
}

static void *mglStageGetControlPointIndex(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self
               ? mglRendererBackendGetTessControlPointIndexBuffer(self->_backend)
               : NULL;
}

static void *mglStageEncoderOwner(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? self->_renderPassManager.state->currentRenderEncoderOwner
                : NULL;
}

static uint32_t mglStageGetTcsOutVerts(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? (uint32_t)self->_tessellation.tcsOutVertices : 0u;
}

static uint64_t mglStageGetTcsOutStride(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? (uint64_t)self->_tessellation.tcsOutputStride : 0u;
}

static uint64_t mglStageGetTessCaptureOff(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? (uint64_t)self->_tessellation.tessVertexCaptureOffset : 0u;
}

static uint64_t mglStageGetTessInstRecords(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? (uint64_t)self->_tessellation.tessInstanceRecords : 0u;
}

static int mglStageGetTessIndexed(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && self->_tessellation.tessIndexedDraw ? 1 : 0;
}

static uint64_t mglStageNativePrimCount(void *canonical, Program *tes,
                                        uint32_t patch_count,
                                        uint32_t instance_count)
{
    return mglNativeTessPrimitiveCount((__bridge id)canonical, tes, patch_count,
                                       instance_count);
}

static void mglStageRecordQuery(GLMContext ctx, uint64_t generated,
                                uint64_t written)
{
    mglRecordActivePrimitiveQueryDraw(ctx, generated, written);
}

static void mglStageLogError(const char *msg)
{
    if (msg) NSLog(@"%s", msg);
}

static int mglStageEnsurePassthrough(void *renderer, Program *program,
                                     uint32_t output_primitive)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self ensureAIRGeometryPassthroughFunctionForProgram:program
                                                       outputPrimitive:output_primitive]
               ? 1
               : 0;
}

static int mglStagePendingGsActive(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && self->_tessellation.pendingGSInputActive ? 1 : 0;
}

static void *mglStagePendingGsInput(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? self->_tessellation.pendingGSInput : NULL;
}

static uint32_t mglStagePendingGsOff(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? (uint32_t)self->_tessellation.pendingGSInputOffset : 0u;
}

static uint32_t mglStagePendingGsStride(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? (uint32_t)self->_tessellation.pendingGSInputStride : 0u;
}


/* ---- A1 / O1.4: GS Metal expansion HostOps (thin MTL materialization only) ---- */

static void mglGsMetalRelease(void *obj)
{
    if (obj) CFRelease(obj);
}

static uint64_t mglGsMetalBufferLength(void *buffer)
{
    return buffer ? (uint64_t)mglDrawSupportBufferLength((__bridge id)buffer) : 0u;
}

static int mglGsMetalEnsureCB(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self newCommandBuffer] ? 1 : 0;
}

static int mglGsMetalBindDrawTextures(void *renderer, GLMContext ctx)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !ctx) return 0;
    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *image = ctx->active_state->image_units[unit].tex;
        Texture *sampled = ctx->active_state->active_textures[unit];
        if (image && ![self bindMTLTexture:image]) return 0;
        if (sampled && ![self bindMTLTexture:sampled]) return 0;
    }
    return 1;
}

static void *mglGsMetalMtlForBuffer(void *renderer, Buffer *buf)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !buf) return NULL;
    if (!buf->data.mtl_data) {
        [self bindMTLBuffer:buf];
    }
    return buf->data.mtl_data;
}

static int mglGsMetalFillComputeBindings(void *renderer, GLMContext ctx,
                                         MGLRenderComputeExecutionPlan *plan,
                                         MGLRenderCopyBackEntry *copybacks,
                                         uint32_t copybacks_cap,
                                         uint32_t *copybacks_count,
                                         void **temporaries_out)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !plan || !copybacks || !copybacks_count) return 0;
    (void)ctx;
    if (temporaries_out) *temporaries_out = NULL;
    MGLStageBindingCopyBackList stageCopyBacks = {0};
    NSMutableArray *temps = [NSMutableArray array];
    id compute = nil;
    bool buffersOK = [self bindBuffersToComputeEncoder:compute
                                                   stage:_GEOMETRY_SHADER
                                               copyBacks:&stageCopyBacks
                                           executionPlan:plan
                                            temporaries:temps];
    bool texturesOK = buffersOK && [self bindTexturesToComputeEncoder:compute
                                                                stage:_GEOMETRY_SHADER
                                                        executionPlan:plan
                                                         temporaries:temps];
    if (!buffersOK || !texturesOK) {
        if (compute) mglDrawSupportEndComputeEncoder(compute);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return 0;
    }
    uint32_t n = mglRenderCollectCopyBackEntries(
        (const MGLRenderCopyBackEntry *)stageCopyBacks.slots,
        kMGLMaxBufferSlots, copybacks, copybacks_cap);
    *copybacks_count = n;
    [self clearStageBindingCopyBacks:&stageCopyBacks];
    /* The plan only stores borrowed MTL pointers, and this function returns
     * before the C++ side encodes/dispatches it.  Hand the keep-alive set back
     * as a +1 CF reference so the caller can hold it across the encode; ARC
     * would otherwise drop `temps` (and every temporary it retains) here, and
     * setBuffer: would retain a deallocated buffer (EXC_BAD_ACCESS /
     * "message sent to deallocated instance"). */
    if (temporaries_out && temps.count) {
        *temporaries_out = (__bridge_retained void *)temps;
    }
    return 1;
}

static void *mglGsMetalCmdOwner(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? self->_renderPassManager.state->currentCommandBufferOwner : NULL;
}

static void *mglGsMetalRecoveryOwner(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? self->_gpuRecovery.commandRecoveryOwner : NULL;
}

static void mglGsMetalNoteDeviceReset(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) {
        atomic_store_explicit(&self->_core.deviceResetRequested, true,
                              memory_order_release);
    }
}

static void mglGsMetalSetExpansion(void *renderer, Program *program, int active,
                                   GLenum last_draw_mode)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    self->_geometry.expansionActive = active ? YES : NO;
    self->_geometry.program = active ? program : NULL;
    if (active && last_draw_mode) {
        self->_lastDrawPrimitiveMode = last_draw_mode;
    }
}

static void *mglGsMetalBeginBlit(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    id blit = mglDrawSupportCreateBlitEncoder(
        self->_renderPassManager.state->currentCommandBufferOwner);
    return (__bridge_retained void *)blit;
}

static void mglGsMetalBlitCopy(void *blit, void *src, uint64_t src_off, void *dst,
                               uint64_t dst_off, uint64_t bytes)
{
    mglDrawSupportBlitCopyBuffer((__bridge id)blit, (__bridge id)src,
                                 (NSUInteger)src_off, (__bridge id)dst,
                                 (NSUInteger)dst_off, (NSUInteger)bytes);
}

static void mglGsMetalEndBlit(void *blit)
{
    if (!blit) return;
    mglDrawSupportEndBlitEncoder((__bridge id)blit);
    CFRelease(blit);
}

static void *mglGsMetalBindingOwner(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? self->_bindingStateOwner : NULL;
}

/* A1: clears are in mgl_draw_gs_metal.cpp; ObjC only rebinds MTL resources. */
static int mglGsMetalRebindFragment(void *renderer, GLMContext ctx)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !ctx) return 0;
    MGLEncodeContext gsEncCtx = {
        .render_encoder_owner =
            self->_renderPassManager.state->currentRenderEncoderOwner,
    };
    [self bindFragmentBuffersToCurrentRenderEncoder:&gsEncCtx];
    [self bindBufferSizeConstantsForRenderEncoder];
    Program *gsVertexProgram =
        mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *gsFragmentProgram =
        mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    return [self bindStorageImagesForVertexProgram:gsVertexProgram
                                   fragmentProgram:gsFragmentProgram]
               ? 1
               : 0;
}

static void mglGsMetalRecordQueries(GLMContext ctx, uint64_t generated,
                                    uint64_t written, int xfb_active,
                                    const MGLAIRGSXFBMeta *meta,
                                    uint32_t stream_count,
                                    const uint64_t *buffer_written,
                                    const uint64_t *buffer_stride,
                                    uint64_t geometry_invocations)
{
    NSUInteger bw[MGL_AIR_GS_MAX_STREAMS] = {0};
    NSUInteger bs[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint32_t n = stream_count < MGL_AIR_GS_MAX_STREAMS ? stream_count
                                                       : MGL_AIR_GS_MAX_STREAMS;
    for (uint32_t i = 0; i < n; i++) {
        if (buffer_written) bw[i] = (NSUInteger)buffer_written[i];
        if (buffer_stride) bs[i] = (NSUInteger)buffer_stride[i];
    }
    mglRecordGeometryPrimitiveQueries(ctx, generated, written,
                                      xfb_active ? YES : NO, meta, stream_count,
                                      bw, bs, geometry_invocations);
}

static void mglGsMetalGpuCaptureStart(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !getenv("MGL_GPU_CAPTURE")) return;
    id desc = [self mglCaptureDescriptorForDevice:((__bridge id)mglRendererBackendGetDevice(self->_backend))
                                       outputPath:[NSString stringWithUTF8String:getenv("MGL_GPU_CAPTURE")]];
    NSError *capErr = nil;
    if (desc && [self mglStartCaptureWithDescriptor:desc error:&capErr]) {
        NSLog(@"MGL GPU capture started -> %s", getenv("MGL_GPU_CAPTURE"));
    } else {
        NSLog(@"MGL GPU capture start failed: %@", capErr.localizedDescription);
    }
}

static void mglGsMetalGpuCaptureStop(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) [self mglStopCapture];
}

static void mglGsMetalSetVertexBuffer(void *encoder_owner, void *buffer,
                                      uint64_t offset, uint32_t index)
{
    mglDrawSupportSetVertexBuffer(encoder_owner, (__bridge id)buffer,
                                  (NSUInteger)offset, index);
}

static void mglGsMetalDrawPrims(void *encoder_owner, uint32_t output_primitive,
                                uint32_t vertex_start, uint32_t vertex_count,
                                uint32_t instance_count, uint32_t base_instance)
{
    mglDrawSupportDrawPrimitives(encoder_owner, output_primitive, vertex_start,
                                 vertex_count, instance_count, base_instance);
}

static void mglGsMetalDrawPrimsIndirect(void *encoder_owner,
                                        uint32_t output_primitive, void *counts,
                                        uint64_t offset)
{
    mglDrawSupportDrawPrimitivesIndirect(encoder_owner, output_primitive,
                                         (__bridge id)counts,
                                         (NSUInteger)offset);
}

static void mglGsMetalLogDiag(const char *msg)
{
    if (msg) NSLog(@"%s", msg);
}

/* A1: fill nested Metal expansion HostOps (no ObjC expansion middle-man). */
static MGLGsMetalExpansionHostOps mglGsMetalMakeExpansionOps(void *renderer)
{
    return (MGLGsMetalExpansionHostOps){
        .renderer = renderer,
        .create_buffer = mglStageCreateBuffer,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .buffer_contents = mglStageBufContents,
        .buffer_length = mglGsMetalBufferLength,
        .release = mglGsMetalRelease,
        .ensure_command_buffer = mglGsMetalEnsureCB,
        .bind_draw_textures = mglGsMetalBindDrawTextures,
        .mtl_for_buffer = mglGsMetalMtlForBuffer,
        .fill_compute_bindings = mglGsMetalFillComputeBindings,
        .command_buffer_owner = mglGsMetalCmdOwner,
        .recovery_owner = mglGsMetalRecoveryOwner,
        .note_device_reset = mglGsMetalNoteDeviceReset,
        .set_expansion = mglGsMetalSetExpansion,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .begin_blit = mglGsMetalBeginBlit,
        .blit_copy = mglGsMetalBlitCopy,
        .end_blit = mglGsMetalEndBlit,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .raster_empty = mglStageRasterEmpty,
        .fully_culled = mglStageFullyCulled,
        .apply_polygon_offset = mglStageApplyPolygonOffset,
        .binding_state_owner = mglGsMetalBindingOwner,
        .rebind_fragment_after_gs = mglGsMetalRebindFragment,
        .encoder_owner = mglStageEncoderOwner,
        .flush_command_buffer = mglStageFlushCB,
        .record_queries = mglGsMetalRecordQueries,
        .gpu_capture_start = mglGsMetalGpuCaptureStart,
        .gpu_capture_stop = mglGsMetalGpuCaptureStop,
        .set_vertex_buffer = mglGsMetalSetVertexBuffer,
        .draw_primitives = mglGsMetalDrawPrims,
        .draw_primitives_indirect = mglGsMetalDrawPrimsIndirect,
        .log_diag = mglGsMetalLogDiag,
    };
}





/* ---- Port runners: fill HostOps and call C++ domain orchestration ---- */

static int mglStagePrimitiveRestart(GLMContext ctx, GLenum index_type,
                                    uint32_t *out_restart_index)
{
    return mglPrimitiveRestartIndexForType(ctx, index_type, out_restart_index)
               ? 1
               : 0;
}

static void *mglStagePrepareElementIndex(void *renderer, void *index_buffer,
                                         GLenum gl_index_type,
                                         uint64_t *inout_offset,
                                         uint64_t *inout_mtl_type)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !index_buffer || !inout_offset || !inout_mtl_type) return NULL;
    NSUInteger off = (NSUInteger)*inout_offset;
    uint64_t typ = *inout_mtl_type;
    id prepared = mglPreparedElementIndexBuffer(
        ((__bridge id)mglRendererBackendGetDevice(self->_backend)), NULL, (__bridge id)index_buffer, gl_index_type, &off,
        &typ);
    if (!prepared) return NULL;
    *inout_offset = (uint64_t)off;
    *inout_mtl_type = typ;
    return (__bridge void *)prepared;
}

static void mglStageSetCtx(void *renderer, GLMContext ctx)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) self->ctx = ctx;
}

static void mglStageClearCullCapture(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(self->_backend, NULL);
    self->_tessellation.cullDistanceCaptureFirstInstance = 0u;
    self->_tessellation.cullDistanceCaptureInstanceStride = 0u;
}

static void mglStageSetCullCaptureActive(void *renderer, int active)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (self) self->_tessellation.cullDistanceCaptureActive = active ? YES : NO;
}

static void mglStageStoreCullCapture(void *renderer, void *buf,
                                     uint32_t first_instance,
                                     uint32_t instance_stride)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return;
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(self->_backend, buf);
    self->_tessellation.cullDistanceCaptureFirstInstance = first_instance;
    self->_tessellation.cullDistanceCaptureInstanceStride = instance_stride;
}

static void *mglStageLoadCullCapture(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? mglRendererBackendGetCullDistanceCaptureBuffer(self->_backend)
                : NULL;
}

static void *mglStageDevicePtr(void *renderer)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self ? (__bridge void *)((__bridge id)mglRendererBackendGetDevice(self->_backend)) : NULL;
}

static void mglStageBindCullEmu(void *renderer, GLenum mode, GLuint first_vertex,
                                const uint32_t *explicit_vertices,
                                uint32_t explicit_vertex_count,
                                const void *enc_ctx)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !enc_ctx) return;
    [self bindCullDistanceEmulationBuffers:mode
                                firstVertex:first_vertex
                           explicitVertices:explicit_vertices
                         explicitVertexCount:explicit_vertex_count
                              encodeContext:(const MGLEncodeContext *)enc_ctx];
}

static int mglStageTryArraySplitEncode(void *renderer, void *device,
                                       void *encoder_owner, GLenum mode,
                                       GLint first, GLsizei count,
                                       uint64_t instance_count,
                                       uint64_t base_instance,
                                       const void *enc_ctx)
{
    return mglEncodeCullDistanceArraySplitForRenderEncoderOwner(
               encoder_owner, (__bridge MGLDrawMetalHandle)device, mode, first,
               count, (size_t)instance_count,
               (size_t)base_instance, renderer, enc_ctx,
               mglRendererBindCullDistanceEmu)
               ? 1
               : 0;
}

static void mglStageDrawIndexedPrimsPort(void *encoder_owner,
                                         uint32_t primitive_type,
                                         uint64_t index_count,
                                         void *index_buffer,
                                         uint64_t index_offset,
                                         uint64_t instance_count,
                                         int32_t base_vertex,
                                         uint64_t base_instance)
{
    mglDrawSupportDrawIndexedPrimitives(
        encoder_owner, primitive_type, (NSUInteger)index_count,
        (__bridge id)index_buffer, (NSUInteger)index_offset,
        (NSUInteger)instance_count, (NSInteger)base_vertex,
        (NSUInteger)base_instance);
}

static int mglStageEncodeCtxActive(const void *enc_ctx)
{
    return mglDrawSupportEncodeContextIsActive((const MGLEncodeContext *)enc_ctx)
               ? 1
               : 0;
}

static MGLTessVertexCaptureHostOps mglStageMakeVertexCaptureOps(void *renderer)
{
    return (MGLTessVertexCaptureHostOps){
        .renderer = renderer,
        .bind_mtl_program = mglStageBindProgram,
        .create_buffer = mglStageCreateBuffer,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .buffer_contents = mglStageBufContents,
        .buffer_length = mglGsMetalBufferLength,
        .mark_dirty_all = mglDrawSupportCaptureMarkDirtyAll,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .bind_capture_slots = mglDrawSupportCaptureBindSlots,
        .set_capture_active = mglDrawSupportCaptureSetActive,
        .encoder_owner = mglStageEncoderOwner,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .end_render_encoding = mglStageEndRender,
        .primitive_restart = mglStagePrimitiveRestart,
        .prepare_element_index = mglStagePrepareElementIndex,
        .log_gs_diag = mglStageLogError,
    };
}

static MGLCullDistanceHostOps mglStageMakeCullOps(void *renderer)
{
    return (MGLCullDistanceHostOps){
        .renderer = renderer,
        .bind_mtl_program = mglStageBindProgram,
        .create_buffer = mglStageCreateBuffer,
        .set_ctx = mglStageSetCtx,
        .clear_cull_capture = mglStageClearCullCapture,
        .set_cull_capture_active = mglStageSetCullCaptureActive,
        .store_cull_capture = mglStageStoreCullCapture,
        .load_cull_capture = mglStageLoadCullCapture,
        .mark_dirty_all = mglDrawSupportCaptureMarkDirtyAll,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .encoder_owner = mglStageEncoderOwner,
        .device = mglStageDevicePtr,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .end_render_encoding = mglStageEndRender,
        .bind_cull_emu = mglStageBindCullEmu,
        .try_array_split_encode = mglStageTryArraySplitEncode,
        .draw_indexed_primitives = mglStageDrawIndexedPrimsPort,
        .encode_context_active = mglStageEncodeCtxActive,
        .primitive_restart = mglStagePrimitiveRestart,
    };
}

void *mglDrawHostRunVertexCaptureArray(void *renderer, GLMContext ctx,
                                       GLint first, GLsizei count,
                                       GLsizei instanceCount,
                                       GLuint baseInstance,
                                       uint64_t *out_offset)
{
    if (!mglStageHostSelf(renderer)) return NULL;
    MGLTessVertexCaptureHostOps ops = mglStageMakeVertexCaptureOps(renderer);
    return mglTessRunVertexCaptureArray(ctx, first, count, instanceCount,
                                        baseInstance, out_offset, &ops);
}
static void *mglDrawPortVertexCaptureArray(void *renderer, GLMContext ctx,
                                           GLint first, GLsizei count,
                                           GLsizei instanceCount,
                                           GLuint baseInstance,
                                           uint64_t *out_offset)
{
    return mglDrawHostRunVertexCaptureArray(renderer, ctx, first, count,
                                            instanceCount, baseInstance,
                                            out_offset);
}

void *mglDrawHostRunVertexCaptureIndexed(
    void *renderer, GLMContext ctx, void *index_mtl, uint64_t indexType,
    uint64_t index_offset, GLsizei count, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, uint32_t maxIndex,
    uint64_t *out_offset)
{
    if (!mglStageHostSelf(renderer)) return NULL;
    MGLTessVertexCaptureHostOps ops = mglStageMakeVertexCaptureOps(renderer);
    return mglTessRunVertexCaptureIndexed(ctx, index_mtl, indexType, index_offset,
                                          count, baseVertex, instanceCount,
                                          baseInstance, maxIndex, out_offset,
                                          &ops);
}
static void *mglDrawPortVertexCaptureIndexed(
    void *renderer, GLMContext ctx, void *index_mtl, GLenum indexType,
    uint64_t index_offset, GLsizei count, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, uint32_t maxIndex,
    uint64_t *out_offset)
{
    return mglDrawHostRunVertexCaptureIndexed(
        renderer, ctx, index_mtl, indexType, index_offset, count, baseVertex,
        instanceCount, baseInstance, maxIndex, out_offset);
}

static void *mglStageGetValidatedVAOPort(GLMContext ctx, const char *where)
{
    return (void *)mglRendererGetValidatedVAO(ctx, where);
}

static int mglStageAttribEnabledPort(void *vao, uint32_t attrib)
{
    VertexArray *v = (VertexArray *)vao;
    if (!v || attrib >= (uint32_t)MAX_ATTRIBS) return 0;
    return (v->enabled_attribs & (0x1u << attrib)) != 0u ? 1 : 0;
}

static int mglStageResolveAttribPort(GLMContext ctx, void *vao, uint32_t attrib,
                                     const char *where,
                                     MGLValidateArraysAttribInfo *out)
{
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    MGLResolvedVertexAttribBinding resolved = {0};
    if (!mglRendererResolveVertexAttribBinding(ctx, (VertexArray *)vao, attrib,
                                               where, &resolved)) {
        return 0;
    }
    const VertexAttrib *a = resolved.attrib;
    Buffer *vbo = resolved.buffer;
    if (!a || !vbo) return 0;
    out->attrib_index = attrib;
    out->buffer_name = vbo->name;
    out->binding_offset = resolved.binding_offset;
    out->relativeoffset = resolved.relativeoffset;
    out->stride = resolved.stride;
    out->divisor = resolved.divisor;
    out->attrib_type = (uint32_t)a->type;
    out->attrib_size = (uint32_t)a->size;
    out->vbo_size = vbo->size;
    out->has_drawable = mglRendererBufferHasDrawableContents(vbo) ? 1 : 0;
    out->written_min = vbo->written_min;
    out->written_max = vbo->written_max;
    out->last_init_source = vbo->last_init_source;
    out->mapped = vbo->mapped;
    out->access = vbo->access;
    out->access_flags = vbo->access_flags;
    out->has_initialized_data = vbo->has_initialized_data;
    out->last_write_offset = vbo->last_write_offset;
    out->last_write_size = vbo->last_write_size;
    out->last_write_src_ptr = vbo->last_write_src_ptr;
    out->last_write_src_hash = vbo->last_write_src_hash;
    out->buffer_obj = vbo;
    out->mtl_data = vbo->data.mtl_data;
    return 1;
}

static int mglStageEnsureMtlBufferPort(void *renderer,
                                       MGLValidateArraysAttribInfo *info)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !info || !info->buffer_obj) return 0;
    Buffer *vbo = (Buffer *)info->buffer_obj;
    if (!vbo->data.mtl_data) {
        [self bindMTLBuffer:vbo];
    }
    info->mtl_data = vbo->data.mtl_data;
    return info->mtl_data ? 1 : 0;
}

static uint64_t mglStageMtlLenPort(void *mtl_data)
{
    return mtl_data ? mglDrawSupportBufferLength((__bridge id)mtl_data) : 0u;
}

static uint32_t mglStageMaxAttribsPort(void) { return (uint32_t)MAX_ATTRIBS; }

static uint32_t mglStageProgramKeyPort(GLMContext ctx)
{
    return (uint32_t)mglCurrentRenderProgramKey(ctx);
}

static int mglStageShouldInspectPort(uint64_t draw_call, uint32_t program_key)
{
    return mglShouldInspectDrawCall(draw_call, (GLuint)program_key) ? 1 : 0;
}

static void mglStageValidateLogPort(const char *msg)
{
    if (msg) NSLog(@"%s", msg);
}

static MGLValidateArraysHostOps mglStageMakeValidateOps(void *renderer)
{
    return (MGLValidateArraysHostOps){
        .renderer = renderer,
        .get_validated_vao = mglStageGetValidatedVAOPort,
        .attrib_enabled = mglStageAttribEnabledPort,
        .resolve_attrib = mglStageResolveAttribPort,
        .ensure_mtl_buffer = mglStageEnsureMtlBufferPort,
        .mtl_buffer_length = mglStageMtlLenPort,
        .max_attribs = mglStageMaxAttribsPort,
        .current_program_key = mglStageProgramKeyPort,
        .should_inspect = mglStageShouldInspectPort,
        .log_line = mglStageValidateLogPort,
    };
}

/* O1.4: host ABI entry points fill HostOps and call C++ runners. */

bool mglDrawHostHandleXFB(void *renderer, GLMContext ctx, GLenum mode,
                          GLint first, GLsizei count, GLsizei instanceCount,
                          GLuint baseInstance)
{
    if (!mglStageHostSelf(renderer)) return false;
    MGLXfbVsDrawHostOps ops = {
        .renderer = renderer,
        .capture_vs_positions = mglStageCaptureArray,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .flush_command_buffer = mglStageFlushCB,
        .buffer_contents = mglStageBufContents,
        .dispatch_error = NULL,
    };
    return mglXfbRunVsOnlyDraw(ctx, mode, first, count, instanceCount,
                               baseInstance, &ops) != 0;
}

bool mglDrawHostHandleTessellation(void *renderer, GLMContext ctx,
                                   GLenum *mode, GLint first, GLsizei count,
                                   GLenum indexType, const void *indices,
                                   GLint baseVertex, GLsizei instanceCount,
                                   GLuint baseInstance, const char *label)
{
    MGLRenderer *host = mglStageHostSelf(renderer);
    if (!host || !mode) return false;
    host->ctx = ctx;
    MGLTessPatchDrawHostOps ops = {
        .renderer = renderer,
        .device = (__bridge void *)((__bridge id)mglRendererBackendGetDevice(mglRendererBackend(host))),
        .bind_mtl_program = mglStageBindProgram,
        .capture_array = mglStageCaptureArray,
        .capture_indexed = mglStageCaptureIndexed,
        .process_buffer = mglStageProcessBuffer,
        .flush_command_buffer = mglStageFlushCB,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .create_buffer = mglStageCreateBuffer,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .buffer_contents = mglStageBufContents,
        .cached_default_factors = mglStageCachedFactors,
        .native_factor_buffer = mglStageNativeFactors,
        .dispatch_tcs = mglStageDispatchTCS,
        .dispatch_air_tes = mglStageDispatchAirTES,
        .dispatch_air_tes_vertex = mglStageDispatchAirTESVertex,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .raster_empty = mglStageRasterEmpty,
        .fully_culled = mglStageFullyCulled,
        .apply_polygon_offset = mglStageApplyPolygonOffset,
        .end_render_encoding = mglStageEndRender,
        .clear_native_copybacks = mglStageClearNativeCB,
        .flush_native_copybacks = mglStageFlushNativeCB,
        .begin_native_tes = mglStageBeginNativeTES,
        .end_native_tes = mglStageEndNativeTES,
        .reset_tess_draw_state = mglStageResetTessDrawState,
        .set_tess_vertex_capture = mglStageSetTessCapture,
        .set_control_point_index_buffer = mglStageSetControlPointIndex,
        .adopt_capture_as_tcs_output = mglStageAdoptCaptureAsTCS,
        .set_current_factors = mglStageSetCurrentFactors,
        .get_tess_vertex_capture = mglStageGetTessCapture,
        .get_tcs_output = mglStageGetTcsOutput,
        .get_current_factors = mglStageGetFactors,
        .get_tcs_patch_out = mglStageGetPatchOut,
        .get_control_point_index = mglStageGetControlPointIndex,
        .encoder_owner = mglStageEncoderOwner,
        .get_tcs_out_vertices = mglStageGetTcsOutVerts,
        .get_tcs_output_stride = mglStageGetTcsOutStride,
        .get_tess_capture_offset = mglStageGetTessCaptureOff,
        .get_tess_instance_records = mglStageGetTessInstRecords,
        .get_tess_indexed_draw = mglStageGetTessIndexed,
        .native_primitive_count = mglStageNativePrimCount,
        .record_primitive_query = mglStageRecordQuery,
        .dispatch_error = NULL,
        .log_error = mglStageLogError,
    };
    return mglTessRunPatchDraw(ctx, mode, first, count, indexType, indices,
                               baseVertex, instanceCount, baseInstance, label,
                               &ops) != 0;
}

bool mglDrawHostHandleGeometry(void *renderer, GLMContext ctx, GLenum mode,
                               GLint first, GLsizei count, GLenum indexType,
                               const void *indices, GLint baseVertex,
                               GLsizei instanceCount, GLuint baseInstance,
                               const char *label)
{
    MGLRenderer *host = mglStageHostSelf(renderer);
    if (!host) return false;
    host->ctx = ctx;
    MGLGsMetalExpansionHostOps metal_ops = mglGsMetalMakeExpansionOps(renderer);
    MGLGsDrawHostOps ops = {
        .renderer = renderer,
        .bind_mtl_program = mglStageBindProgram,
        .ensure_passthrough = mglStageEnsurePassthrough,
        .process_buffer = mglStageProcessBuffer,
        .capture_array = mglStageCaptureArray,
        .capture_indexed = mglStageCaptureIndexed,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .pending_gs_input_active = mglStagePendingGsActive,
        .pending_gs_input = mglStagePendingGsInput,
        .pending_gs_input_offset = mglStagePendingGsOff,
        .pending_gs_input_stride = mglStagePendingGsStride,
        .metal_ops = &metal_ops,
        .dispatch_error = NULL,
        .log_diag = NULL,
    };
    return mglDrawGsRunDraw(ctx, mode, first, count, indexType, indices,
                            baseVertex, instanceCount, baseInstance, label,
                            &ops) != 0;
}


static MGLRenderer *mglDrawHostSelf(void *renderer)
{
    return renderer ? (__bridge MGLRenderer *)renderer : nil;
}


void mglDrawHostGuardIssueArrays(void *renderer, GLMContext ctx, GLenum mode,
                                 GLint first, GLsizei count,
                                 GLsizei instanceCount, GLuint baseInstance,
                                 const char *label, int with_ms)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || !ctx) {
        return;
    }
    METAL_LOCK();
    host->_lastDrawPrimitiveMode = mode;
    if (with_ms &&
        [host runEmulatedMSSampleDrawLoopIfNeeded:ctx
                                         drawOnce:^{
                                             mglIssueDrawArrays(
                                                 ctx, renderer, mode, first,
                                                 count, instanceCount,
                                                 baseInstance, label);
                                         }]) {
        METAL_UNLOCK();
        return;
    }
    mglIssueDrawArrays(ctx, renderer, mode, first, count, instanceCount,
                       baseInstance, label);
    if (with_ms) {
        [host broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:ctx];
    }
    METAL_UNLOCK();
}

void mglDrawHostGuardIssueElements(void *renderer, GLMContext ctx, GLenum mode,
                                   GLsizei count, GLenum type,
                                   const void *indices, GLsizei instanceCount,
                                   GLint baseVertex, GLuint baseInstance,
                                   const char *label, int with_ms)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || !ctx) {
        return;
    }
    METAL_LOCK();
    host->_lastDrawPrimitiveMode = mode;
    if (with_ms &&
        [host runEmulatedMSSampleDrawLoopIfNeeded:ctx
                                         drawOnce:^{
                                             mglIssueDrawElements(
                                                 ctx, renderer, mode, count,
                                                 type, indices, instanceCount,
                                                 baseVertex, baseInstance,
                                                 label);
                                         }]) {
        METAL_UNLOCK();
        return;
    }
    mglIssueDrawElements(ctx, renderer, mode, count, type, indices,
                         instanceCount, baseVertex, baseInstance, label);
    if (with_ms) {
        [host broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:ctx];
    }
    METAL_UNLOCK();
}

bool mglDrawHostBindContext(void *renderer, GLMContext ctx)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    host->ctx = ctx;
    return true;
}

void mglDrawHostSetLastPrimitiveMode(void *renderer, GLenum mode)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        host->_lastDrawPrimitiveMode = mode;
    }
}

/* mglDrawHostHandle* tess → StageHost O1.4 C++ runners */

/* mglDrawHostHandle* gs → StageHost O1.4 C++ runners */

/* mglDrawHostHandle* xfb → StageHost O1.4 C++ runners */

bool mglDrawHostCaptureCullDistanceArray(void *renderer, GLMContext ctx,
                                         GLint first, GLsizei count,
                                         GLsizei instanceCount,
                                         GLuint baseInstance)
{
    if (!mglStageHostSelf(renderer)) return false;
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawRunCullDistanceArrayCapture(ctx, first, count, instanceCount,
                                              baseInstance, &ops) != 0;
}

bool mglDrawHostCaptureCullDistanceElement(void *renderer, GLMContext ctx,
                                           const uint8_t *indexBytes,
                                           GLenum indexType, GLsizei count,
                                           GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance)
{
    if (!mglStageHostSelf(renderer)) return false;
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawRunCullDistanceElementCapture(ctx, indexBytes, indexType, count,
                                                baseVertex, instanceCount,
                                                baseInstance, &ops) != 0;
}

bool mglDrawHostProcessGLStateLocked(void *renderer, bool draw_command)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host processGLStateLocked:draw_command] ? true : false;
}

bool mglDrawHostRasterizationIsEmpty(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host currentDrawRasterizationIsEmpty];
}

bool mglDrawHostModeFullyCulled(void *renderer, GLenum mode)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host currentDrawModeIsFullyCulled:mode];
}

void mglDrawHostApplyPolygonOffset(void *renderer, GLenum mode)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        [host applyPolygonOffsetForDrawMode:mode];
    }
}

bool mglDrawHostEnsureRasterEncoder(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host ensureRasterEncoderForDraw];
}

bool mglDrawHostValidateArrayVertexInputs(void *renderer, GLMContext ctx,
                                          GLenum mode, GLint first,
                                          GLsizei count)
{
    if (!mglStageHostSelf(renderer)) return false;
    MGLValidateArraysHostOps ops = mglStageMakeValidateOps(renderer);
    const int enabled = mglVboRangeValidationEnabled() ? 1 : 0;
    return mglDrawValidateArraysVertexInputs(ctx, mode, first, count, 0ull,
                                             enabled, &ops) != 0;
}

bool mglDrawHostEncodeCullDistanceArray(void *renderer, GLenum mode,
                                        GLint first, GLsizei count,
                                        GLsizei instanceCount,
                                        GLuint baseInstance)
{
    MGLRenderer *host = mglStageHostSelf(renderer);
    if (!host || mglPolygonModePointForDrawMode(host->ctx, mode)) {
        return false;
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner =
            mglRendererRenderPassManager(host).state->currentRenderEncoderOwner,
    };
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawEncodeCullDistanceArray(host->ctx, mode, first, count,
                                          instanceCount, baseInstance, &encCtx,
                                          &ops) != 0;
}

void *mglDrawHostEncoderOwner(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host ? mglRendererRenderPassManager(host).state->currentRenderEncoderOwner
                : NULL;
}

void *mglDrawHostDevice(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host ? mglRendererBackendGetDevice(mglRendererBackend(host)) : NULL;
}

void mglDrawHostRecordArraySubmitted(void *renderer, GLenum mode,
                                     uint64_t vertexCount)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        [host recordArrayDrawSubmittedMode:mode vertexCount:vertexCount];
    }
}

void mglDrawHostWatchdogArrays(void *renderer, GLMContext ctx)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return;
    }
    mglLogDrawWithoutSwapWatchdog(
        "arrays", 0, ctx,
        mglRendererRenderPassManager(host).state->currentCommandBufferOwner,
        mglRendererRenderPassManager(host).state->currentRenderEncoderOwner,
        mglRendererRenderPassManager(host).state->renderPassStateOwner);
}


bool mglDrawHostEncodeCullDistanceElementBytes(
    void *renderer, GLenum mode, const uint8_t *indexBytes, GLenum type,
    GLsizei count, GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    int polygon_line_mode, const void *enc_ctx)
{
    MGLRenderer *host = mglStageHostSelf(renderer);
    if (!host) return false;
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    GLMContext ctx = host->ctx;
    return mglDrawEncodeCullDistanceElement(
               ctx, mode, indexBytes, type, count, baseVertex, instanceCount,
               baseInstance, polygon_line_mode, enc_ctx, &ops) != 0;
}

bool mglDrawHostPrepareEncodeCullDistanceElement(
    void *renderer, GLenum mode, const uint8_t *indexBytes, GLenum type,
    GLsizei count, GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    int polygon_line_mode)
{
    MGLRenderer *host = mglStageHostSelf(renderer);
    if (!host || !host->ctx) return false;
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawPrepareAndEncodeCullDistanceElement(
               host->ctx, mode, indexBytes, type, count, baseVertex,
               instanceCount, baseInstance, polygon_line_mode, &ops) != 0;
}

bool mglDrawHostEncodeCullDistanceElements(void *renderer, GLenum mode,
                                           GLenum type, const void *indices,
                                           GLsizei count, GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance)
{
    MGLRenderer *host = mglStageHostSelf(renderer);
    if (!host || mglPolygonModePointForDrawMode(host->ctx, mode)) {
        return false;
    }
    Buffer *glBuffer = NULL;
    id metalBuffer = nil;
    if (![host resolveElementBufferForDraw:"drawElements"
                                   context:host->ctx
                                  glBuffer:&glBuffer
                                 mtlBuffer:&metalBuffer]) {
        return false;
    }
    const NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    const uint8_t *cullIndexBytes = mglElementIndexSourceForDraw(
        glBuffer, metalBuffer, type, offset, count);
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawPrepareAndEncodeCullDistanceElement(
               host->ctx, mode, cullIndexBytes, type, count, baseVertex,
               instanceCount, baseInstance,
               mglPolygonModeLineForDrawMode(host->ctx, mode) ? 1 : 0, &ops) != 0;
}

bool mglDrawHostResolveElementBuffer(void *renderer, GLMContext ctx,
                                     const char *label, Buffer **glBufferOut,
                                     void **metalBufferOut)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    id metalBuffer = nil;
    if (![host resolveElementBufferForDraw:label ? label : "drawElements"
                                   context:ctx
                                  glBuffer:glBufferOut
                                 mtlBuffer:&metalBuffer]) {
        return false;
    }
    if (metalBufferOut) {
        *metalBufferOut = (__bridge void *)metalBuffer;
    }
    return true;
}

void mglDrawHostRecordElementSubmitted(void *renderer, GLenum mode,
                                       uint64_t indexCount)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        [host recordElementDrawSubmittedMode:mode indexCount:indexCount];
    }
}

void mglDrawHostWatchdogElements(void *renderer, GLMContext ctx)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return;
    }
    mglLogDrawWithoutSwapWatchdog(
        "elements", 0, ctx,
        mglRendererRenderPassManager(host).state->currentCommandBufferOwner,
        mglRendererRenderPassManager(host).state->currentRenderEncoderOwner,
        mglRendererRenderPassManager(host).state->renderPassStateOwner);
}

bool mglDrawHostResolveIndirectBuffer(void *renderer, GLMContext ctx,
                                      const char *label, Buffer **glBufferOut,
                                      void **metalBufferOut)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    id metalBuffer = nil;
    if (![host resolveIndirectBufferForDraw:label ? label : "indirectDraw"
                                   context:ctx
                                  glBuffer:glBufferOut
                                 mtlBuffer:&metalBuffer]) {
        return false;
    }
    if (metalBufferOut) {
        *metalBufferOut = (__bridge void *)metalBuffer;
    }
    return true;
}

bool mglDrawHostPrepareIndirectCPURead(void *renderer, GLMContext ctx,
                                       const char *label)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host prepareEmulatedIndirectCPURead:ctx
                                                  label:label ? label : "indirectDraw"];
}

bool mglDrawHostHasGeometry(GLMContext ctx)
{
    Program *gsProgram = mglResolveProgramForStageFromState(ctx, _GEOMETRY_SHADER);
    return gsProgram && gsProgram->shader_slots[_GEOMETRY_SHADER];
}

bool mglDrawHostUsesCullDistance(GLMContext ctx)
{
    Program *vertexProgram =
        mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    return vertexProgram && vertexProgram->uses_cull_distance;
}


