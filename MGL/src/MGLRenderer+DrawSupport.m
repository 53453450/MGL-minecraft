/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+DrawSupport.m
// Draw validation, element-buffer resolution and rasterization helper
// methods extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_shader_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_program_reflection.h"
#include "mgl_draw_issue.h"
#include "mgl_draw_gs.h"
#include "mgl_draw_tess.h"
#include "mgl_index_buffer.h"
#include "mgl_draw_mode.h"
#include "mgl_draw_encode.h"

static void *mglDrawSupportBufferContents(id buffer)
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

static uint64_t mglDrawSupportBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(
        (__bridge void *)buffer, &info) == 0 ? info.length : 0u;
}

static MGLRenderTextureInfo mglDrawSupportTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

static BOOL mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext)
{
    if (!encodeContext) return NO;
    return mglRenderEncoderOwnerHasCurrent(
        encodeContext->render_encoder_owner) == 1;
}


static bool mglGeometryGatherIndices(const uint8_t *indexBytes,
                                     GLenum indexType,
                                     GLsizei count,
                                     int32_t baseVertex,
                                     bool restartEnabled,
                                     uint32_t restartIndex,
                                     uint32_t inputVertices,
                                     uint32_t **outGather,
                                     uint32_t *outGatherCount,
                                     uint32_t *outPrimitiveCount,
                                     uint32_t *outMaxIndex)
{

    (void)baseVertex; /* gather stores raw index values (vertex_id) */
    if (!outGather || !outGatherCount || !outPrimitiveCount || !outMaxIndex) {
        return false;
    }
    const uint32_t elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    MGLRenderGeometryGatherResult result = {0};
    if (mglRenderGeometryGatherIndices(
            indexBytes, elemBytes, (uint32_t)count,
            restartEnabled ? 1 : 0, restartIndex, inputVertices,
            &result) != 0) {
        return false;
    }
    *outGather = result.gather;
    *outGatherCount = result.gather_count;
    *outPrimitiveCount = result.primitive_count;
    *outMaxIndex = result.max_index;
    return true;
}

static id mglDrawSupportCreateBuffer(
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

static id mglDrawSupportCreateBufferWithBytes(
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

static id mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

static void mglDrawSupportBlitCopyBuffer(id encoder,
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

static void mglDrawSupportEndBlitEncoder(id encoder)
{
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

static void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglDrawSupportDrawIndexedPrimitives(
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

static void mglDrawSupportDrawPrimitives(
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

static void mglDrawSupportDrawPrimitivesIndirect(
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

static id mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateComputeEncoderBorrowed(
        commandBufferOwner);
}

static void mglDrawSupportSetComputePipeline(
    id encoder,
    id pipeline)
{
    (void)mglRenderSetComputePipelineState((__bridge void *)encoder,
                                              (__bridge void *)pipeline);
}

static void mglDrawSupportSetComputeBuffer(
    id encoder,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetComputeBuffer((__bridge void *)encoder,
                                       (__bridge void *)buffer, offset,
                                       (uint32_t)index);
}

static void mglDrawSupportSetComputeBytes(
    id encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetComputeBytes((__bridge void *)encoder, bytes,
                                      length, (uint32_t)index);
}

static void mglDrawSupportDispatchCompute(
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

static void mglDrawSupportEndComputeEncoder(
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

static void mglRecordGeometryPrimitiveQueries(
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
        ctx, generatedStream0, xfbActive ? writtenStream0 : 0u);
    if (!meta || !bufferWritten || !bufferStride) return;
    if (streamCount > MGL_AIR_GS_MAX_STREAMS) {
        streamCount = MGL_AIR_GS_MAX_STREAMS;
    }
    for (uint32_t s = 1u; s < streamCount; s++) {
        /* Indexed stream s query: generated stays in the meta; written is
         * the ordered scatter's whole-primitive bytes for buffer s divided
         * by its per-record stride (streams > 0 are points, vpp = 1). */
        GLuint64 written = 0u;
        if (xfbActive && bufferStride[s] > 0u) {
            written = (GLuint64)bufferWritten[s] /
                      (GLuint64)bufferStride[s];
        }
        mglRecordActivePrimitiveQueryDrawIndexed(
            ctx, s, (GLuint64)meta->stream[s].generated, written);
    }
}

static id mglDefaultTessFactorBuffer(id device,
                                                GLMState *state,
                                                GLuint patchCount)
{
    if (!device || !state || patchCount == 0u) return nil;
    const NSUInteger stride = MGL_AIR_TESS_FACTOR_RECORD_BYTES;
    if ((NSUInteger)patchCount > NSUIntegerMax / stride) return nil;
    id buffer = mglDrawSupportCreateBuffer(
        device, (NSUInteger)patchCount * stride,
        0u);
    if (!buffer || !mglDrawSupportBufferContents(buffer)) return nil;

    if (mglRenderFillDefaultTessFactorBuffer(
            (void *)mglDrawSupportBufferContents(buffer),
            (uint64_t)((NSUInteger)patchCount * stride),
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
static id mglCachedDefaultTessFactorBuffer(
    id device, MGLRendererBackendHandle *backend, GLMState *state,
    GLuint patchCount)
{
    if (!device || !backend || !state || patchCount == 0u) return nil;
    float levels[6] = {
        state->var.patch_default_outer_level[0],
        state->var.patch_default_outer_level[1],
        state->var.patch_default_outer_level[2],
        state->var.patch_default_outer_level[3],
        state->var.patch_default_inner_level[0],
        state->var.patch_default_inner_level[1],
    };
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

static id mglNativeTessFactorBuffer(id device,
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

static GLuint64 mglNativeTessPrimitiveCount(id canonical,
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

@implementation MGLRenderer (Draw)

- (BOOL)captureAIRCullDistancesForArrayDraw:(GLMContext)drawCtx
                                      first:(GLint)first
                                      count:(GLsizei)count
                              instanceCount:(GLsizei)instanceCount
                               baseInstance:(GLuint)baseInstance
{
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(_backend, NULL);
    _tessellation.cullDistanceCaptureFirstInstance = 0u;
    _tessellation.cullDistanceCaptureInstanceStride = 0u;
    if (!drawCtx || first < 0 || count <= 0 || instanceCount <= 0) return NO;

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (!vertexProgram || !vertexProgram->uses_cull_distance ||
        ![self bindMTLProgram:vertexProgram] ||
        !vertexProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function) {
        return NO;
    }
    uint64_t captureBytes = 0u;
    if (mglRenderCullDistanceCaptureBytes((uint32_t)first, (uint32_t)count,
                                          (uint32_t)instanceCount,
                                          &captureBytes) != 0) {
        return NO;
    }
    id capture = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)captureBytes, 0u);
    if (!capture) return NO;

    self->ctx = drawCtx;
    _tessellation.cullDistanceCaptureActive = YES;
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.cullDistanceCaptureActive = NO;
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return NO;
    }
    MGLCullDistanceEmuParams params;
    mglRenderFillCullDistanceEmuParams(
        1u, (uint32_t)first, NULL, 0u, 0u, 32u,
        MIN(vertexProgram->cull_distance_count, 8u), baseInstance,
        (uint32_t)count, &params);
    mglRenderBindCullDistanceEmuSlots(
        _renderPassManager.state->currentRenderEncoderOwner,
        (__bridge void *)capture, &params);
    mglTessEncodeCaptureArray(
        _renderPassManager.state->currentRenderEncoderOwner, (uint32_t)first,
        (uint32_t)count, (uint32_t)instanceCount, baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.cullDistanceCaptureActive = NO;
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(
        _backend, (__bridge void *)capture);
    _tessellation.cullDistanceCaptureFirstInstance = baseInstance;
    _tessellation.cullDistanceCaptureInstanceStride = (uint32_t)count;
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    return YES;
}

- (BOOL)captureAIRCullDistancesForElementDraw:(GLMContext)drawCtx
                                    indexBytes:(const uint8_t *)indexBytes
                                     indexType:(GLenum)indexType
                                         count:(GLsizei)count
                                    baseVertex:(GLint)baseVertex
                                 instanceCount:(GLsizei)instanceCount
                                  baseInstance:(GLuint)baseInstance
{
    if (!drawCtx || !indexBytes || count <= 0 || instanceCount <= 0) return NO;
    uint32_t restartIndex = 0u;
    const bool restartEnabled =
        mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    const uint32_t elemWidth = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    int32_t first = 0;
    uint32_t vertexCount = 0u;
    if (mglRenderPlanCullDistanceElementRange(
            indexBytes, elemWidth, (uint32_t)count,
            restartEnabled ? 1 : 0, restartIndex, baseVertex,
            &first, &vertexCount) != 0) {
        return NO;
    }
    return [self captureAIRCullDistancesForArrayDraw:drawCtx
                                               first:first
                                               count:(GLsizei)vertexCount
                                       instanceCount:instanceCount
                                        baseInstance:baseInstance];
}

- (BOOL)prepareAndEncodeDirectCullDistanceElementDraw:(GLenum)mode
                                           indexBytes:(const uint8_t *)indexBytes
                                            indexType:(GLenum)indexType
                                                count:(GLsizei)count
                                           baseVertex:(GLint)baseVertex
                                        instanceCount:(GLsizei)instanceCount
                                         baseInstance:(GLuint)baseInstance
                                      polygonLineMode:(BOOL)polygonLineMode
{
    Program *activeProgram =
        ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : NULL;
    if (!activeProgram || !activeProgram->uses_cull_distance) return NO;
    if (!indexBytes || count <= 0 || instanceCount <= 0) return YES;

    if (activeProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function) {
        if (![self captureAIRCullDistancesForElementDraw:ctx
                                             indexBytes:indexBytes
                                              indexType:indexType
                                                  count:count
                                             baseVertex:baseVertex
                                          instanceCount:instanceCount
                                           baseInstance:baseInstance] ||
            ![self processGLState:true] ||
            mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
            return YES;
        }
    }

    MGLEncodeContext encCtx = {
        .render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner,
    };
    return [self encodeCullDistanceElementDraw:mode
                                    indexBytes:indexBytes
                                     indexType:indexType
                                         count:count
                                    baseVertex:baseVertex
                                 instanceCount:instanceCount
                                  baseInstance:baseInstance
                               polygonLineMode:polygonLineMode
                                 encodeContext:&encCtx];
}

- (BOOL)encodeCullDistanceArrayDraw:(GLenum)mode
                               first:(GLint)first
                               count:(GLsizei)count
                       instanceCount:(GLsizei)instanceCount
                        baseInstance:(GLuint)baseInstance
                       encodeContext:(const MGLEncodeContext *)encCtx
{
    Program *activeProgram =
        ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : NULL;
    if (!activeProgram || !activeProgram->uses_cull_distance ||
        !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return NO;
    }
    if (mglEncodeCullDistanceArraySplitForRenderEncoderOwner(
            encCtx->render_encoder_owner, _device, mode, first, count,
            (size_t)instanceCount, (size_t)baseInstance, (__bridge void *)self,
            encCtx, mglRendererBindCullDistanceEmu)) {
        return YES;
    }
    [self bindCullDistanceEmulationBuffers:mode
                                firstVertex:(GLuint)first
                           explicitVertices:NULL
                         explicitVertexCount:0u
                              encodeContext:encCtx];
    return NO;
}

- (BOOL)encodeCullDistanceElementDraw:(GLenum)mode
                            indexBytes:(const uint8_t *)indexBytes
                             indexType:(GLenum)indexType
                                 count:(GLsizei)count
                            baseVertex:(GLint)baseVertex
                         instanceCount:(GLsizei)instanceCount
                          baseInstance:(GLuint)baseInstance
                       polygonLineMode:(BOOL)polygonLineMode
                         encodeContext:(const MGLEncodeContext *)encCtx
{
    Program *activeProgram =
        ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : NULL;
    if (!activeProgram || !activeProgram->uses_cull_distance) return NO;
    id captureBuffer = (__bridge id)
        mglRendererBackendGetCullDistanceCaptureBuffer(_backend);
    if (activeProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function &&
        !captureBuffer) {
        return YES;
    }
    if (!indexBytes || count <= 0 || instanceCount <= 0 ||
        !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return YES;
    }

    uint32_t restartIndex = 0u;
    const bool restartEnabled =
        mglPrimitiveRestartIndexForType(ctx, indexType, &restartIndex);
    void *planOwner = NULL;
    void *indexBufferHandle = NULL;
    uint64_t primitiveCount = 0u;
    if (mglRenderCreateCullDistanceIndexPlan(
            (__bridge void *)_device, indexBytes, indexType,
            (uint64_t)count, mode,
            restartEnabled ? 1 : 0, restartIndex, baseVertex,
            polygonLineMode ? 1 : 0, &planOwner, &indexBufferHandle,
            &primitiveCount) != 0 || !planOwner) {
        return YES;
    }

    id indexBuffer =
        (__bridge id)indexBufferHandle;
    @try {
        for (uint64_t primitiveIndex = 0u;
             primitiveIndex < primitiveCount; ++primitiveIndex) {
            MGLRenderCullDistancePrimitive primitive = {0};
            if (mglRenderGetCullDistanceIndexPrimitive(
                    planOwner, primitiveIndex, &primitive) != 0) {
                break;
            }
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:0u
                                   explicitVertices:primitive.vertices
                                 explicitVertexCount:primitive.vertex_count
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->render_encoder_owner,
                (uint32_t)primitive.primitive_type,
                (NSUInteger)primitive.index_count,
                indexBuffer,
                (NSUInteger)primitive.index_buffer_offset,
                (NSUInteger)instanceCount,
                0,
                (NSUInteger)baseInstance);
        }
    } @finally {
        mglRenderDestroyCullDistanceIndexPlan(&planOwner);
    }
    return YES;
}

- (BOOL)runVertexCaptureSession:(GLMContext)drawCtx
                        capture:(id)capture
                         params:(const uint32_t *)params
{
    if (!drawCtx || !capture || !params) {
        return NO;
    }
    self->ctx = drawCtx;
    _tessellation.tessVertexCaptureActive = YES;
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.tessVertexCaptureActive = NO;
        return NO;
    }
    mglTessBindCaptureSlots(
        _renderPassManager.state->currentRenderEncoderOwner,
        (__bridge void *)capture, params);
    /* Re-apply GL bindings after installing the capture buffers at 28/29.
     * The first capture draw in a context otherwise left VS SSBO/UBO slots
     * unbound (probe: first GS+SSBO write is 0, second is correct). */
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.tessVertexCaptureActive = NO;
        return NO;
    }
    mglTessBindCaptureSlots(
        _renderPassManager.state->currentRenderEncoderOwner,
        (__bridge void *)capture, params);
    return YES;
}

- (id)captureAIRVertexPositionsForTessellation:(GLMContext)drawCtx
                                                    first:(GLint)first
                                                    count:(GLsizei)count
                                            instanceCount:(GLsizei)instanceCount
                                             baseInstance:(GLuint)baseInstance
                                               outOffset:(NSUInteger *)outOffset
{
    if (outOffset) *outOffset = 0u;
    if (!drawCtx || first < 0 || count <= 0 || instanceCount <= 0) return nil;

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (!vertexProgram || ![self bindMTLProgram:vertexProgram] ||
        !vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function) {
        return nil;
    }

    MGLTessVertexCapturePlan plan = {0};
    if (!mglTessPlanVertexCapture(vertexProgram, (uint32_t)count,
                                  (uint32_t)instanceCount, (uint32_t)first,
                                  baseInstance, &plan)) {
        return nil;
    }
    id capture = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)plan.capture_size, 0u);
    if (!capture) return nil;

    if (![self runVertexCaptureSession:drawCtx
                               capture:capture
                                params:plan.params]) {
        return nil;
    }
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG capture-draw POINT first=%d count=%d instances=%d baseInst=%u stride=%u size=%llu",
              (int)first, (int)count, (int)instanceCount, baseInstance,
              (unsigned)plan.capture_stride,
              (unsigned long long)plan.capture_size);
    }
    mglTessEncodeCaptureArray(
        _renderPassManager.state->currentRenderEncoderOwner, (uint32_t)first,
        (uint32_t)count, (uint32_t)instanceCount, baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.tessVertexCaptureActive = NO;
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    if (outOffset) *outOffset = (NSUInteger)plan.capture_offset;
    return capture;
}


- (id)captureAIRVertexPositionsForGeometryIndexed:(GLMContext)drawCtx
                                                  indexBuffer:(id)indexBuffer
                                                    indexType:(uint64_t)indexType
                                                  indexOffset:(NSUInteger)indexOffset
                                                        count:(GLsizei)count
                                                    baseVertex:(GLint)baseVertex
                                                 instanceCount:(GLsizei)instanceCount
                                                  baseInstance:(GLuint)baseInstance
                                                     maxIndex:(uint32_t)maxIndex
                                                    outOffset:(NSUInteger *)outOffset
{
    if (outOffset) *outOffset = 0u;
    if (!drawCtx || count <= 0 || instanceCount <= 0 || !indexBuffer) {
        return nil;
    }

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (!vertexProgram || ![self bindMTLProgram:vertexProgram] ||
        !vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function) {
        return nil;
    }

    MGLTessVertexCapturePlan plan = {0};
    if (!mglTessPlanVertexCapture(vertexProgram, maxIndex + 1u,
                                  (uint32_t)instanceCount, 0u, baseInstance,
                                  &plan)) {
        return nil;
    }
    id capture = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)plan.capture_size, 0u);
    if (!capture) return nil;
    if (![self runVertexCaptureSession:drawCtx
                               capture:capture
                                params:plan.params]) {
        return nil;
    }
    /* The capture kernel indexes records by raw vertex_id with no bounds
     * check; a primitive-restart marker (0xFFFFFFFF for UInt32) in the
     * stream would write past the sparse record span and corrupt the
     * next instance's data.  Sanitize the marker away (to vertex 0, whose
     * record no gathered patch ever references) before drawing. */
    id sanitizedIndexBuffer = indexBuffer;
    NSUInteger sanitizedIndexOffset = indexOffset;
    uint32_t restartIndex = 0u;
    if (mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex)) {
        const NSUInteger elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
            : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
        const NSUInteger streamBytes = (NSUInteger)count * elemBytes;
        if (mglDrawSupportBufferContents(indexBuffer) &&
            (NSUInteger)indexOffset + streamBytes <= mglDrawSupportBufferLength(indexBuffer)) {
            uint8_t *copy = malloc(streamBytes);
            if (copy) {
                if (mglTessSanitizeRestartIndices(
                        copy,
                        (const uint8_t *)mglDrawSupportBufferContents(
                            indexBuffer) +
                            indexOffset,
                        (uint32_t)count, (GLenum)indexType, restartIndex)) {
                    id clean = mglDrawSupportCreateBufferWithBytes(
                        _device, copy, streamBytes, 0u);
                    if (clean) {
                        sanitizedIndexBuffer = clean;
                        sanitizedIndexOffset = 0u;
                    }
                }
                free(copy);
            }
        }
    }
    /* Metal has no UInt8 index type: GL_UNSIGNED_BYTE streams must be
     * expanded to UInt16 before the indexed capture draw, or Metal reads
     * byte pairs as garbage indices and every record past index 0 is lost. */
    id drawIndexBuffer = sanitizedIndexBuffer;
    NSUInteger drawIndexOffset = sanitizedIndexOffset;
    uint64_t mtlIndexType = mglIndexTypeForGLType((GLenum)indexType);
    if ((GLuint)mtlIndexType != 0xFFFFFFFFu) {
        NSUInteger preparedOffset = sanitizedIndexOffset;
        uint64_t preparedType = mtlIndexType;
        id prepared = mglPreparedElementIndexBuffer(
            _device, NULL, sanitizedIndexBuffer, (GLenum)indexType,
            &preparedOffset, &preparedType);
        if (prepared) {
            drawIndexBuffer = prepared;
            drawIndexOffset = preparedOffset;
            mtlIndexType = preparedType;
        }
    }
    mglTessEncodeCaptureIndexed(
        _renderPassManager.state->currentRenderEncoderOwner,
        (__bridge void *)drawIndexBuffer, (uint32_t)mtlIndexType,
        (uint64_t)drawIndexOffset, (uint32_t)count, (int32_t)baseVertex,
        (uint32_t)instanceCount, baseInstance);
    _currentCBHasWork = YES;    [self endRenderEncoding];
    _tessellation.tessVertexCaptureActive = NO;
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    if (outOffset) *outOffset = (NSUInteger)plan.capture_offset;
    return capture;
}

- (BOOL)handleVertexTransformFeedbackDrawIfNeeded:(GLMContext)drawCtx
                                               mode:(GLenum)mode
                                              first:(GLint)first
                                              count:(GLsizei)count
                                      instanceCount:(GLsizei)instanceCount
                                       baseInstance:(GLuint)baseInstance
{
    /* VS-only XFB: capture one record per vertex for POINTS / LINES /
     * TRIANGLES families (GL 4.6 §12.1).  Draw mode must be compatible with
     * BeginTransformFeedback's primitiveMode — not POINTS-only. */
    if (!drawCtx || first < 0 || count <= 0 || instanceCount <= 0) {
        return NO;
    }
    TransformFeedback *xfb = MGL_STATE(drawCtx)->transform_feedback;
    if (!xfb || !xfb->active || xfb->paused) {
        return NO;
    }
    if (!mglXfbPrimitiveModeAccepts(xfb->primitive_mode, mode)) {
        return NO;
    }
    Program *program = mglResolveProgramForStageFromState(
        drawCtx, _VERTEX_SHADER);
    MGLXfbVsPlan plan = {0};
    if (!mglXfbPlanVsCapture(program, &plan)) {
        return NO;
    }

    NSUInteger captureOffset = 0u;
    id capture = [self captureAIRVertexPositionsForTessellation:drawCtx
                                                          first:first
                                                          count:count
                                                  instanceCount:instanceCount
                                                   baseInstance:baseInstance
                                                     outOffset:&captureOffset];
    if (!capture) return NO;
    _currentCBHasWork = YES;
    [self flushCommandBuffer:YES];

    const uint8_t *captureBytes =
        (const uint8_t *)mglDrawSupportBufferContents(capture);
    uint64_t recordCount64 = (uint64_t)(uint32_t)count *
                             (uint64_t)(uint32_t)instanceCount;
    if (!captureBytes || recordCount64 > NSUIntegerMax) {
        return YES;
    }
    NSUInteger recordCount = (NSUInteger)recordCount64;
    /* GL semantics: once any active buffer runs out of room, further
     * primitives are neither written nor counted by
     * TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN (PRIMITIVES_GENERATED keeps
     * counting).  Track the capped count across all buffers. */
    NSUInteger writtenTotal = recordCount;

    for (GLuint buffer = 0u; buffer < plan.buffer_count; buffer++) {
        if (plan.buffer_stride[buffer] == 0u) continue;
        BufferBaseTarget *slot =
            &MGL_STATE(drawCtx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER]
                                        .buffers[buffer];
        BufferMap map = {0};
        map.buf = slot->buf;
        map.offset = slot->offset;
        map.size = slot->size;
        NSUInteger visible = slot->buf
            ? mglBufferMapVisibleBackingBytes(
                  &map, slot->buf->size > 0 ? (size_t)slot->buf->size : 0u)
            : 0u;
        NSUInteger sessionOffset = xfb->buffer_write_offsets[buffer] <=
                (GLuint64)NSUIntegerMax
            ? (NSUInteger)xfb->buffer_write_offsets[buffer] : visible;
        MGLXfbVsBufferDest dest = {0};
        if (!mglXfbPlanVsBufferDest((uint32_t)recordCount,
                                    plan.buffer_stride[buffer],
                                    slot->buf != NULL, slot->offset,
                                    (uint64_t)sessionOffset, (uint64_t)visible,
                                    &dest) ||
            dest.skip) {
            writtenTotal = 0;
            continue;
        }
        if (dest.written_records < writtenTotal) {
            writtenTotal = dest.written_records;
        }
        uint8_t *packed = (uint8_t *)calloc(1u, dest.written_bytes);
        if (!packed) {
            mglDispatchError(drawCtx, "vertexTransformFeedback",
                             GL_OUT_OF_MEMORY);
            return YES;
        }
        mglXfbPackVsRecords(&plan, buffer, captureBytes, captureOffset,
                            plan.capture_stride, dest.written_records, packed,
                            plan.buffer_stride[buffer]);
        NSUInteger destinationOffset = dest.destination_offset;
        NSUInteger writtenBytes = dest.written_bytes;
        mglRendererBufferSubData(drawCtx, slot->buf, destinationOffset,
                                 writtenBytes, packed);
        /* The renderer's subdata routes through the shadow/snapshot pair,
         * so the live Metal allocation may lag until the next snapshot
         * flush while glMapBufferRange serves from it directly.  Mirror
         * the bytes into the live allocation now; XFB capture is a
         * synchronous CPU-side operation by definition here. */
        if (slot->buf->data.mtl_data) {
            MGLRenderBufferInfo liveInfo = {0};
            if (mglRenderGetBufferInfo(slot->buf->data.mtl_data,
                                       &liveInfo) == 0 &&
                destinationOffset + writtenBytes <= liveInfo.length) {
                uint8_t *liveBase = (uint8_t *)mglDrawSupportBufferContents(
                    (__bridge id)(slot->buf->data.mtl_data));
                if (liveBase) {
                    memcpy(liveBase + destinationOffset, packed,
                           writtenBytes);
                }
            }
        }
        /* The renderer's subdata writes straight to the Metal allocation
         * and leaves the CPU shadow untouched; a later glMapBufferRange
         * served from the shadow would otherwise observe pre-capture
         * bytes. */
        if (slot->buf->data.buffer_data &&
            (size_t)slot->buf->size >= destinationOffset + writtenBytes) {
            memcpy((uint8_t *)slot->buf->data.buffer_data + destinationOffset,
                   packed, writtenBytes);
        }
        /* CPU pack is authoritative: do not mark gpu_write_target (flush
         * readback would clobber the shadow with the untouched Metal image)
         * and keep cpu_shadow_pending so MapBuffer skips Metal→CPU sync. */
        mglRenderMarkBufferCPUWrite(slot->buf, (int64_t)destinationOffset,
                                    (int64_t)writtenBytes);
        free(packed);
        xfb->buffer_write_offsets[buffer] = mglXfbAdvanceWriteOffset(
            xfb->buffer_write_offsets[buffer], (uint64_t)writtenBytes);
    }

    xfb->primitives_generated += (GLuint64)recordCount;
    xfb->primitives_written += (GLuint64)writtenTotal;
    mglRecordActivePrimitiveQueryDraw(drawCtx, (GLuint64)recordCount,
                                      (GLuint64)writtenTotal);
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    return YES;
}

- (BOOL)handleGeometryDrawIfNeeded:(GLMContext)drawCtx
                              mode:(GLenum)mode
                             first:(GLint)first
                             count:(GLsizei)count
                         indexType:(GLenum)indexType
                           indices:(const void *)indices
                        baseVertex:(GLint)baseVertex
                     instanceCount:(GLsizei)instanceCount
                      baseInstance:(GLuint)baseInstance
                             label:(const char *)label
{
    if (!drawCtx) {
        return NO;
    }

    Program *program = mglResolveProgramForStageFromState(
        drawCtx, _GEOMETRY_SHADER);
    Shader *geometryShader = program
        ? program->shader_slots[_GEOMETRY_SHADER] : NULL;
    if (!program || !geometryShader) {
        return NO;
    }
    GLenum gsInputMode = GL_TRIANGLES;
    GLenum gsOutputMode = GL_TRIANGLE_STRIP;
    uint32_t outputPrimitive = MGL_DRAW_PRIMITIVE_TRIANGLE;
    mglDrawGsNormalizeTopology(program, &gsInputMode, &gsOutputMode,
                               &outputPrimitive);
    const BOOL indexedDraw = (indexType != 0u);
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG topology mode=0x%x gsIn=0x%x gsOut=0x%x indexed=%d count=%d first=%d vertsOut=%u route=%d",
              (unsigned)mode, (unsigned)gsInputMode, (unsigned)gsOutputMode,
              indexedDraw ? 1 : 0, (int)count, (int)first,
              (unsigned)program->geometry_vertices_out,
              (int)program->gs_route);
    }
    if (!mglDrawGsInputModeAccepts(gsInputMode, mode) || count <= 0 ||
        instanceCount <= 0 || (!indexedDraw && first < 0)) {
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG topology rejected mode=0x%x gsIn=0x%x",
                  (unsigned)mode, (unsigned)gsInputMode);
        }
        static uint64_t unsupportedDrawCount = 0;
        uint64_t hit = ++unsupportedDrawCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL GS ERROR: blocking unsupported %s draw %@ "
                   "mode=0x%x gsIn=0x%x count=%d instances=%d baseInstance=%u",
                  indexedDraw ? "indexed" : "array",
                  label ? [NSString stringWithUTF8String:label] : @"draw",
                  (unsigned)mode, (unsigned)gsInputMode, (int)count, (int)instanceCount,
                  (unsigned)baseInstance);
        }
        /*  contract: never drop a GS draw silently.  A draw whose mode
         * does not match the GS input topology is an invalid operation. */
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        return YES;
    }
    if (![self bindMTLProgram:program] ||
        !program->modules[_GEOMETRY_SHADER].mtl_function) {
        NSLog(@"MGL GS ERROR: failed to load AIR kernel program=%u",
              (unsigned)program->name);
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        return YES;
    }

    self->ctx = drawCtx;
    if (![self ensureAIRGeometryPassthroughFunctionForProgram:program
                                              outputPrimitive:outputPrimitive]) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }


    uint32_t *gatherArray = NULL;
    uint32_t gatherCount = 0u;
    uint32_t gatherPrimitives = 0u;
    uint32_t gatherMaxIndex = 0u;
    const uint8_t *indexBytes = NULL;
    id eboMetal = nil;
    NSUInteger indexOffsetBytes = 0u;
    id gatherBuf = nil;
    MGLAIRGSGatherParams gparams;
    memset(&gparams, 0, sizeof(gparams));
    {
        Buffer *ebo = indexedDraw ? getElementBuffer(drawCtx) : NULL;
        if (indexedDraw &&
            (!ebo || ![self processBuffer:ebo] || !ebo->data.mtl_data)) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_INVALID_OPERATION);
            return YES;
        }
        if (indexedDraw) {
            eboMetal = (__bridge id)ebo->data.mtl_data;
            indexOffsetBytes = (NSUInteger)(uintptr_t)indices;
            indexBytes = mglElementIndexSourceForDraw(
                ebo, eboMetal, indexType, indexOffsetBytes, count);
            if (!indexBytes) {
                mglDispatchError(drawCtx, label ? label : "geometryDraw",
                                 GL_INVALID_OPERATION);
                return YES;
            }
        }
        uint32_t restartIndex = 0u;
        const bool restartEnabled = indexedDraw &&
            mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
        if (!mglDrawGsGatherTopology(
                indexBytes, indexType, count, first, indexedDraw,
                restartEnabled, restartIndex, mode, &gatherArray,
                &gatherCount, &gatherPrimitives, &gatherMaxIndex)) {
            /* Incomplete primitive groups are valid GL draws with no
             * invocations. */
            return YES;
        }
        gatherBuf = mglDrawSupportCreateBufferWithBytes(
            _device, gatherArray, (NSUInteger)gatherCount * 4u, 0u);
        free(gatherArray);
        gatherArray = NULL;
        if (!gatherBuf) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_OUT_OF_MEMORY);
            return YES;
        }
        mglDrawGsFillGatherParams(indexedDraw ? 1 : 0, (uint32_t)count,
                                  (uint32_t)first, gatherMaxIndex,
                                  gatherPrimitives, &gparams);
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG gather mode=0x%x indexed=%d first=%d count=%d gathered=%u prims=%u max=%u params={%u,%u,%u,%u}",
                  (unsigned)mode, indexedDraw ? 1 : 0, (int)first, (int)count,
                  (unsigned)gatherCount, (unsigned)gatherPrimitives,
                  (unsigned)gatherMaxIndex,
                  (unsigned)gparams.vertices_per_instance,
                  (unsigned)gparams.primitives_per_instance,
                  (unsigned)gparams.first_vertex,
                  (unsigned)gparams.gather_enabled);
        }
    }

    const GLuint primitiveCount = (GLuint)gatherPrimitives;
    MGLGsComputeLayout gsLayout;
    if (!mglDrawGsComputeLayout(program, primitiveCount, (uint32_t)instanceCount,
                                gsOutputMode, &gsLayout)) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const GLuint workItemCount = gsLayout.work_item_count;
    const NSUInteger outputStride = gsLayout.output_stride;
    const NSUInteger expandedVertices = gsLayout.expanded_vertices;
    const NSUInteger recordsPerPrimitive = gsLayout.records_per_primitive;
    const uint32_t maxVertices =
        mglDrawGsMaxVerticesOut(program->geometry_vertices_out);

    /* Run the real VS once into the shared per-vertex records used by the AIR GS
     * kernel.  This helper closes the render encoder before compute begins.
     * Isolines/point-mode TES may already have expanded records; consume those
     * instead of re-capturing the (empty) VS attributes. */
    NSUInteger inputOffset = 0u;
    id input = nil;
    Program *captureVS = mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    Program *captureTES = NULL;
    MGLGsInputSourcePlan inputSource = {0};
    mglDrawGsPlanInputSource(
        _tessellation.pendingGSInputActive ? 1 : 0,
        _tessellation.pendingGSInput ? 1 : 0,
        (uint32_t)_tessellation.pendingGSInputOffset,
        (uint32_t)_tessellation.pendingGSInputStride, indexedDraw ? 1 : 0,
        &inputSource);
    if (inputSource.kind == MGL_GS_INPUT_PENDING_TES) {
        input = (__bridge id)_tessellation.pendingGSInput;
        inputOffset = (NSUInteger)inputSource.input_offset;
        captureTES = mglResolveProgramForStageFromState(
            drawCtx, _TESS_EVALUATION_SHADER);
    } else if (inputSource.kind == MGL_GS_INPUT_CAPTURE_INDEXED) {
        input = [self captureAIRVertexPositionsForGeometryIndexed:drawCtx
                                                      indexBuffer:eboMetal
                                                        indexType:indexType
                                                      indexOffset:indexOffsetBytes
                                                            count:count
                                                        baseVertex:baseVertex
                                                     instanceCount:instanceCount
                                                      baseInstance:baseInstance
                                                         maxIndex:gatherMaxIndex
                                                        outOffset:&inputOffset];
    } else {
        input = [self captureAIRVertexPositionsForTessellation:
                         drawCtx
                                 first:first
                                 count:count
                         instanceCount:instanceCount
                          baseInstance:baseInstance
                           outOffset:&inputOffset];
    }
    if (!input) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return YES;
    }
    /* Publish the capture record stride the kernels must use when walking
     * gl_in records.  The capture writes one record per *vertex-stage*
     * output varying slot, so its layout comes from the VS output list and
     * can be wider than what this GS declares as inputs (e.g. a flat
     * instance_id the GS never reads).  A stride mismatch made every
     * gl_in[N>0] read land inside the wrong record. */
    gparams.stage_in_stride = mglDrawGsResolveStageInStride(
        captureVS, captureTES, inputSource.pending_stride);
    /* Publish the GS-input -> capture-offset location map.  The capture
     * lays records out by the *vertex* stage's output locations; a VS
     * output the GS never declares (a flat helper like instance_id) shifts
     * every later slot, so reading by the GS's own locations lands in the
     * wrong fields.  loc_map[gs_loc] = vs_loc + 1; 0 falls back to
     * identity inside the kernel. */
    mglDrawGsFillLocationMap(program, captureVS, captureTES, gparams.loc_map);
    if (getenv("MGL_GS_DIAG")) {
        const MGLShaderResourceList *gsIn2 =
            &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_INPUT_RES];
        const MGLShaderResourceList *vsOut2 =
            captureVS ? &captureVS->shader_resources_list[_VERTEX_SHADER]
                                                        [_STAGE_OUTPUT_RES]
                      : NULL;
        for (GLuint gi2 = 0u; gsIn2 && gsIn2->list && gi2 < gsIn2->count; gi2++)
            NSLog(@"MGL GS DIAG gsIn[%u] name=%s loc=%u active=%d",
                  gi2, gsIn2->list[gi2].name ?: "?",
                  gsIn2->list[gi2].location,
                  (int)gsIn2->list[gi2].resource_active);
        for (GLuint vi2 = 0u; vsOut2 && vsOut2->list && vi2 < vsOut2->count; vi2++)
            NSLog(@"MGL GS DIAG vsOut[%u] name=%s loc=%u active=%d",
                  vi2, vsOut2->list[vi2].name ?: "?",
                  vsOut2->list[vi2].location,
                  (int)vsOut2->list[vi2].resource_active);
    }
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG gparams={%u,%u,%u,%u,%u} loc_map[0..3]={%u,%u,%u,%u}",
              gparams.vertices_per_instance, gparams.primitives_per_instance,
              gparams.first_vertex, gparams.gather_enabled,
              gparams.stage_in_stride,
              gparams.loc_map[0], gparams.loc_map[1],
              gparams.loc_map[2], gparams.loc_map[3]);
    void *pipelineHandle = NULL;
    char pipelineError[2048] = {0};
    int pipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _GEOMETRY_SHADER, &pipelineHandle,
        pipelineError, sizeof(pipelineError));
    id pipeline =
        pipelineResult == 0 && pipelineHandle
            ? (__bridge_transfer id)pipelineHandle
            : nil;
    if (!pipeline) {
        NSLog(@"MGL GS ERROR: compute PSO failed program=%u: %s",
              (unsigned)program->name,
              pipelineError[0] ? pipelineError : "unknown error");
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return YES;
    }
    MGLRenderCommandBufferState commandState = {0};
    const int hasCommandState = mglRenderCommandBufferOwnerHasState(
        _renderPassManager.state->currentCommandBufferOwner, &commandState);
    if (mglTessCommandBufferNeedsNew(hasCommandState, commandState.status)) {
        if (![self newCommandBuffer]) {
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
    }
    const NSUInteger outputSize = (NSUInteger)gsLayout.output_bytes;
    id output = mglDrawSupportCreateBuffer(
        _device, outputSize, 0u);
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG outputSize=%lu stride=%lu recordsPerPrim=%lu workItems=%u mtlLen=%@",
              (unsigned long)outputSize, (unsigned long)outputStride,
              (unsigned long)recordsPerPrimitive, (unsigned)workItemCount,
              [output valueForKey:@"length"]);

    const NSUInteger countsRecordBytes = MGL_AIR_GS_COUNTS_RECORD_BYTES;
    id counts = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)gsLayout.counts_bytes,
        0u);
    if (!output || !counts || !mglDrawSupportBufferContents(output) || !mglDrawSupportBufferContents(counts)) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    memset(mglDrawSupportBufferContents(counts), 0,
           (size_t)workItemCount * countsRecordBytes);
    memset(mglDrawSupportBufferContents(output), 0, outputSize);
    /* Preset the draw parameters the kernel never touches: instance_count=1,
     * base_vertex=0, base_instance=0 (memset already zeroed the rest). */
    mglDrawGsPresetCounts(mglDrawSupportBufferContents(counts), workItemCount);

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *image = MGL_STATE(drawCtx)->image_units[unit].tex;
        Texture *sampled = MGL_STATE(drawCtx)->active_textures[unit];
        if (image && ![self bindMTLTexture:image]) {
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
        if (sampled && ![self bindMTLTexture:sampled]) {
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};

    TransformFeedback *xfbState = MGL_STATE(drawCtx)->transform_feedback;
    const bool xfbActive = mglDrawGsXFBActive(
        xfbState != NULL, xfbState && xfbState->active,
        xfbState && xfbState->paused) != 0;
    const BOOL xfbDiag = getenv("MGL_GS_XFB_DIAG") != NULL;

    /* ---- GL4 ordered multi-buffer XFB (mgl_air_gs_abi.h §5b) ----
     * Replace the prototype per-stream atomic-cursor capture with a
     * per-*buffer* layout driven by the link-time scatter plan
     * (Program.transform_feedback_layout[]).  Records are scattered by the
     * pass-2 aux kernel in emission order with whole-primitive cross-buffer
     * truncation. */
    const bool gsSeparate =
        program->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS;

    /* Per-buffer scatter plan (indexed by transform-feedback buffer 0..3). */
    MGLAIRGSXFBScatterParams scatterParams;
    uint32_t xfbBufferCount = mglDrawGsFillXFBScatterParams(
        xfbActive ? program : NULL, &scatterParams);
    /* Per-buffer GL binding state for copy-back (indexed by buffer index). */
    NSUInteger bufferCapBytes[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferPhysBase[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferDstOffset[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferRemaining[MGL_AIR_GS_MAX_STREAMS] = {0u};
    id bufferDstMTL[MGL_AIR_GS_MAX_STREAMS] = {nil};

    id xfbTemporary = nil;
    id xfbCaptureBuffer = nil;   /* slot-31 capture (always the temporary) */
    id xfbVisBuffer = nil;       /* pass-1 per-(work-item, buffer) bytes    */
    id xfbOffsetBuffer = nil;    /* CPU prefix offsets for pass 2           */
    id xfbWrittenBuffer = nil;   /* pass-2 per-(work-item, buffer) written  */
    id scatterPipeline = nil;

    if (xfbActive) {
        /* Field descriptors, buffer→stream map, and packed record strides
         * come from the link-time scatter plan. */
        const uint32_t fieldCount = scatterParams.field_count;

        if (xfbDiag) {
            fprintf(stderr,
                    "MGL GS XFB DIAG fields=%u buffers=%u varyings=%d mode=0x%x\n",
                    fieldCount, xfbBufferCount,
                    program->transform_feedback_varying_count,
                  program->transform_feedback_buffer_mode);
            for (uint32_t f = 0u; f < fieldCount; f++) {
                NSLog(@"  field[%u] buf=%u src=%u dst=%u bytes=%u", f,
                      scatterParams.fields[f].buffer_index,
                      scatterParams.fields[f].src_offset,
                      scatterParams.fields[f].dst_offset,
                      scatterParams.fields[f].byte_count);
            }
        }

        /* Resolve each active buffer's GL binding, visible capacity and
         * session write offset.  Always capture into a fresh temporary so the
         * pass-2 scatter writes ordered records independent of the GL store
         * address; copy-back moves them afterwards. */
        MGLGsXFBBufferBinding xfbBindings[MGL_AIR_GS_MAX_STREAMS];
        memset(xfbBindings, 0, sizeof(xfbBindings));
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            if (scatterParams.buffers[b].stride == 0u) continue;
            BufferBaseTarget *slot = &MGL_STATE(drawCtx)
                ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[b];
            if (!slot->buf) {
                if (xfbDiag)
                    NSLog(@"MGL GS XFB DIAG buffer[%u] no bound GL buffer", b);
                continue;
            }
            if (!slot->buf->data.mtl_data) {
                [self bindMTLBuffer:slot->buf];
            }
            id mtl = (__bridge id)(slot->buf->data.mtl_data);
            if (!mtl) {
                if (xfbDiag)
                    NSLog(@"MGL GS XFB DIAG buffer[%u] no MTL backing", b);
                continue;
            }
            BufferMap map = {0};
            map.buf = slot->buf;
            map.offset = slot->offset;
            map.size = slot->size;
            NSUInteger visible = mglBufferMapVisibleBackingBytes(
                &map, (size_t)mglDrawSupportBufferLength(mtl));
            NSUInteger sessionOffset = 0u;
            if (xfbState->buffer_write_offsets[b] <= (GLuint64)NSUIntegerMax) {
                sessionOffset = (NSUInteger)xfbState->buffer_write_offsets[b];
            }
            xfbBindings[b].bound = 1u;
            xfbBindings[b].slot_offset = slot->offset;
            xfbBindings[b].session_offset = (uint64_t)sessionOffset;
            xfbBindings[b].visible_bytes = (uint64_t)visible;
            bufferDstMTL[b] = mtl;
        }
        MGLGsXFBDestPlan destPlan = {0};
        mglDrawGsPlanXFBDestinations(&scatterParams, xfbBufferCount,
                                     (uint32_t)workItemCount,
                                     (uint32_t)expandedVertices, xfbBindings,
                                     &destPlan);
        NSUInteger physTotal = destPlan.phys_total;
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            if (!destPlan.buffers[b].valid) continue;
            bufferRemaining[b] = destPlan.buffers[b].remaining;
            bufferDstOffset[b] = destPlan.buffers[b].dst_offset;
            bufferCapBytes[b] = destPlan.buffers[b].cap_bytes;
            bufferPhysBase[b] = destPlan.buffers[b].phys_base;
        }
        if (xfbDiag) {
            for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                NSLog(@"  buffer[%u] stride=%u cap=%u base=%u physTotal=%lu dstMTL=%@",
                      b, scatterParams.buffers[b].stride,
                      scatterParams.buffers[b].capacity_bytes,
                      scatterParams.buffers[b].capture_base,
                      (unsigned long)physTotal, bufferDstMTL[b]);
            }
        }
        mglDrawGsFillXFBScatterRuntime(
            &scatterParams, xfbBufferCount, (uint32_t)workItemCount,
            (uint32_t)outputStride, (uint32_t)recordsPerPrimitive,
            outputPrimitive);

        if (physTotal > 0u && xfbBufferCount > 0u) {
            xfbTemporary = mglDrawSupportCreateBuffer(_device, physTotal, 0u);
            if (xfbTemporary) {
                memset(mglDrawSupportBufferContents(xfbTemporary), 0,
                       physTotal);
                xfbCaptureBuffer = xfbTemporary;
            }
            const NSUInteger visBytes =
                (NSUInteger)mglDrawGsXFBVisBytes((uint32_t)workItemCount);
            xfbVisBuffer = mglDrawSupportCreateBuffer(_device, visBytes, 0u);
            xfbOffsetBuffer = mglDrawSupportCreateBuffer(_device, visBytes, 0u);
            xfbWrittenBuffer =
                mglDrawSupportCreateBuffer(_device, visBytes, 0u);
            if (xfbVisBuffer && mglDrawSupportBufferContents(xfbVisBuffer)) {
                memset(mglDrawSupportBufferContents(xfbVisBuffer), 0,
                       visBytes);
            }
            if (xfbWrittenBuffer &&
                mglDrawSupportBufferContents(xfbWrittenBuffer)) {
                memset(mglDrawSupportBufferContents(xfbWrittenBuffer), 0,
                       visBytes);
            }
            const MGLAuxShaderAsset *scatterAsset =
                mglAuxShaderAssetFind("gs_xfb_scatter");
            if (scatterAsset && scatterAsset->data) {
                void *scatterHandle = NULL;
                char scatterError[256] = {0};
                if (mglRenderGetOrCreateAuxComputePipelineFromMetallib(
                        scatterAsset->data, scatterAsset->size,
                        scatterAsset->hash, "mgl_gs_xfb_scatter",
                        MGL_RENDER_AUX_COMPUTE_GS_XFB_SCATTER, 0u,
                        &scatterHandle, scatterError,
                        sizeof(scatterError)) == 0 &&
                    scatterHandle) {
                    scatterPipeline = (__bridge_transfer id)scatterHandle;
                } else {
                    NSLog(@"MGL GS XFB ERROR: scatter pipeline failed: %s",
                          scatterError[0] ? scatterError : "unknown");
                }
            }
            if (!xfbTemporary || !xfbVisBuffer || !xfbOffsetBuffer ||
                !xfbWrittenBuffer || !scatterPipeline) {
                drawCtx->active_state->dirty_bits = DIRTY_ALL;
                mglDispatchError(drawCtx, label ? label : "geometryDraw",
                                 GL_OUT_OF_MEMORY);
                return YES;
            }
        }
    }
    /* Back-compat locals referenced by the query/copy-back tail below. */
    NSUInteger streamStride[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferStride[MGL_AIR_GS_MAX_STREAMS] = {0u};
    for (uint32_t b = 0u; b < MGL_AIR_GS_MAX_STREAMS; b++) {
        streamStride[b] = scatterParams.buffers[b].stride;
        bufferStride[b] = scatterParams.buffers[b].stride;
    }
    const uint32_t gsStreamCount =
        mglDrawGsStreamCount(program->geometry_stream_count);
    const bool multiStream = gsStreamCount > 1u;
    (void)gsSeparate;
    MGLGsXFBDestPlan destForMeta = {0};
    uint32_t capBytesU32[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint32_t physBaseU32[MGL_AIR_GS_MAX_STREAMS] = {0};
    for (uint32_t b = 0u; b < MGL_AIR_GS_MAX_STREAMS; b++) {
        capBytesU32[b] = (uint32_t)bufferCapBytes[b];
        physBaseU32[b] = (uint32_t)bufferPhysBase[b];
    }
    mglDrawGsFillXFBDestForMeta(capBytesU32, physBaseU32,
                                MGL_AIR_GS_MAX_STREAMS, &destForMeta);
    MGLAIRGSXFBMeta xfbMeta;
    mglDrawGsFillXFBMetaFromDest(&scatterParams, &destForMeta, &xfbMeta);
    mglDrawGsClearXFBMetaIfNoCapture(xfbCaptureBuffer ? 1 : 0, &xfbMeta);
    id xfbMetaBuf = mglDrawSupportCreateBufferWithBytes(
        _device, &xfbMeta, sizeof(xfbMeta), 0u);
    if (!xfbMetaBuf) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const BOOL cppDispatch = YES;
    id compute = nil;
    MGLRenderComputeExecutionResult executionResult = {0};
    BOOL gsQueryCountersReady = NO;
    MGLRenderComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = [NSMutableArray array];
    executionPlan.pipeline = (__bridge void *)pipeline;
    if (!mglDrawGsAppendCoreBindings(
            &executionPlan, (__bridge void *)input, (uint64_t)inputOffset,
            (__bridge void *)output, (__bridge void *)counts,
            (__bridge void *)(gatherBuf ? gatherBuf : counts),
            xfbCaptureBuffer ? (__bridge void *)xfbCaptureBuffer : NULL,
            (__bridge void *)xfbMetaBuf,
            (__bridge void *)(xfbVisBuffer ? xfbVisBuffer : counts),
            &gparams, (uint32_t)sizeof(gparams))) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    if (getenv("MGL_GS_DIAG")) {
        Program *gp = mglResolveProgramForStageFromState(drawCtx, _GEOMETRY_SHADER);
        NSLog(@"MGL GS DIAG GS uniform-constant resources: %u",
              gp ? gp->shader_resources_list[_GEOMETRY_SHADER][_UNIFORM_CONSTANT_RES].count : 0u);
    }
    if (getenv("MGL_GPU_CAPTURE")) {
        id desc = [self mglCaptureDescriptorForDevice:_device
                                          outputPath:[NSString stringWithUTF8String:getenv("MGL_GPU_CAPTURE")]];
        NSError *capErr = nil;
        if (desc && [self mglStartCaptureWithDescriptor:desc error:&capErr]) {
            NSLog(@"MGL GPU capture started -> %s", getenv("MGL_GPU_CAPTURE"));
        } else {
            NSLog(@"MGL GPU capture start failed: %@", capErr.localizedDescription);
        }
    }
    bool buffersOK = [self bindBuffersToComputeEncoder:compute
                                                   stage:_GEOMETRY_SHADER
                                               copyBacks:&stageCopyBacks
                                           executionPlan:&executionPlan
                                            temporaries:executionTemporaries];
    bool texturesOK = buffersOK && [self bindTexturesToComputeEncoder:compute
                                                                stage:_GEOMETRY_SHADER
                                                        executionPlan:&executionPlan
                                                         temporaries:executionTemporaries];
    if (getenv("MGL_GS_DIAG")) {
        for (uint32_t bi = 0; bi < executionPlan.binding_op_count; bi++) {
            const MGLRenderComputeBindingOp *op = &executionPlan.binding_ops[bi];
            NSLog(@"MGL GS DIAG binding[%u] kind=%u slot=%u offset=%llu buffer=%p",
                  (unsigned)bi, (unsigned)op->kind, (unsigned)op->index,
                  (unsigned long long)op->offset, op->buffer);
            if (op->kind == 0u && op->index == 0u && op->buffer) {
                const float *f = (const float *)mglDrawSupportBufferContents(
                    (__bridge id)op->buffer);
                const int32_t *iw = (const int32_t *)f;
                NSLog(@"MGL GS DIAG uniform slot0 words: %d %d %d %d %d %d %d %d",
                      iw[0], iw[1], iw[2], iw[3], iw[4], iw[5], iw[6], iw[7]);
            }
        }
    }
    if (!buffersOK || !texturesOK) {
        if (compute) mglDrawSupportEndComputeEncoder(compute);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return YES;
    }
    if (cppDispatch) {
        MGLRenderCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)stageCopyBacks.slots,
            kMGLMaxBufferSlots, copyBackEntries, kMGLMaxBufferSlots);
        executionPlan.dispatch = (MGLRenderComputePlan){
            .dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT,
            .groups_x = (uint32_t)workItemCount,
            .groups_y = 1u,
            .groups_z = 1u,
            .local_x = 1u,
            .local_y = 1u,
            .local_z = 1u,
        };
        executionPlan.barrier_scope = copyBackEntryCount
            ? MGL_RENDER_COMPUTE_BARRIER_BUFFERS
            : MGL_RENDER_COMPUTE_BARRIER_NONE;
        const BOOL requireCPUVisibility = mglDrawGsNeedCPUVisibility(
            xfbActive ? 1 : 0,
            (mglHasActiveIndexedPrimitiveQuery(drawCtx) ||
             mglHasActivePrimitiveQuery(drawCtx) ||
             mglHasActiveGeometryShaderQuery(drawCtx))
                ? 1
                : 0) != 0;
        const BOOL gsDiagnostic = getenv("MGL_GS_DIAG") != NULL;
        char executionError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount,
                (requireCPUVisibility || gsDiagnostic) ? 1u : 0u, &executionResult,
                executionError, sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL GS ERROR: C++ execution transaction failed: %s",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
        gsQueryCountersReady = executionResult.transaction.waited != 0;
        [self clearStageBindingCopyBacks:&stageCopyBacks];
    }
    _geometry.expansionActive = YES;
    _geometry.program = program;
    /* The passthrough pipeline rasterizes the GS output primitive class, so
     * drive inputPrimitiveTopology from the output mode, not the GL input
     * mode (e.g. points in -> triangle_strip out). */
    _lastDrawPrimitiveMode = mglDrawGsLastDrawMode(outputPrimitive);
    drawCtx->active_state->dirty_bits = DIRTY_ALL;

    /* ---- GL4 ordered XFB: CPU prefix-sum + pass-2 scatter ----
     * pass 1 (above) filled the visibility buffer; compute per-buffer
     * exclusive prefix offsets and run the ordered scatter kernel.  The
     * pass-1 transaction already waited for CPU visibility (requireCPUVisibility
     * includes xfbActive), so the visibility contents are stable here. */
    NSUInteger bufferWritten[MGL_AIR_GS_MAX_STREAMS] = {0u};
    if (xfbActive && xfbVisBuffer && xfbOffsetBuffer && xfbWrittenBuffer &&
        scatterPipeline && xfbCaptureBuffer &&
        mglDrawSupportBufferContents(xfbVisBuffer) &&
        mglDrawSupportBufferContents(xfbOffsetBuffer) &&
        mglDrawSupportBufferContents(xfbWrittenBuffer)) {
        uint32_t *vis =
            (uint32_t *)mglDrawSupportBufferContents(xfbVisBuffer);
        uint32_t *offsets =
            (uint32_t *)mglDrawSupportBufferContents(xfbOffsetBuffer);
        if (xfbDiag && counts &&
            mglDrawSupportBufferContents(counts) && output &&
            mglDrawSupportBufferContents(output)) {
            const uint32_t *cw =
                (const uint32_t *)mglDrawSupportBufferContents(counts);
            const float *outPos = (const float *)
                mglDrawSupportBufferContents(output);
            outPos += (MGL_AIR_GS_HEADER_RECORDS * outputStride) /
                      sizeof(float);
            NSLog(@"MGL GS XFB DIAG pass1 vertex_count=%u emit=%u vis[0]=%u "
                  "out.pos={%g,%g,%g,%g}",
                  cw[0], cw[MGL_AIR_GS_COUNTS_ARGS_WORDS + 2u], vis[0],
                  outPos[0], outPos[1], outPos[2], outPos[3]);
        }
        /* Exclusive prefix-sum per buffer across work items. */
        mglDrawGsExclusivePrefixSum(vis, offsets, (uint32_t)workItemCount,
                                    xfbBufferCount);
        /* Run pass 2 as its own compute transaction on the scatter PSO. */
        MGLRenderComputeExecutionPlan scatterPlan = {0};
        if (!mglDrawGsFillXFBScatterPlan(
                &scatterPlan, (__bridge void *)scatterPipeline, &scatterParams,
                (uint32_t)sizeof(scatterParams),
                (__bridge void *)xfbVisBuffer,
                (__bridge void *)xfbOffsetBuffer, (__bridge void *)output,
                (__bridge void *)xfbCaptureBuffer,
                (__bridge void *)xfbWrittenBuffer, (uint32_t)workItemCount)) {
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_OUT_OF_MEMORY);
            return YES;
        }
        MGLRenderComputeExecutionResult scatterResult = {0};
        char scatterError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &scatterPlan, NULL, 0u, 1u, &scatterResult,
                scatterError, sizeof(scatterError)) != 0) {
            if (scatterResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL GS XFB ERROR: scatter transaction failed: %s",
                  scatterError[0] ? scatterError : "unknown error");
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
        /* Reduce the per-(work-item, buffer) written counters. */
        const uint32_t *written =
            (const uint32_t *)mglDrawSupportBufferContents(xfbWrittenBuffer);
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            bufferWritten[b] = (NSUInteger)mglDrawGsReduceBufferWritten(
                written, (uint32_t)workItemCount, b);
        }
    }

    GLuint64 queryGenerated = 0u;
    GLuint64 queryWritten = 0u;
    const MGLAIRGSXFBMeta *queryMeta = NULL;
    if (xfbActive && xfbMetaBuf && mglDrawSupportBufferContents(xfbMetaBuf)) {

        const MGLAIRGSXFBMeta *meta =
            (const MGLAIRGSXFBMeta *)mglDrawSupportBufferContents(xfbMetaBuf);
        queryMeta = meta;
        /* Ordered multi-buffer copy-back: blit each buffer's written segment
         * (already whole-primitive truncated by the scatter kernel) back to
         * its GL XFB target and advance the session write offset. */
        if (xfbTemporary) {
            id xfbBlit = nil;
            uint8_t *xfbTempBytes =
                (uint8_t *)mglDrawSupportBufferContents(xfbTemporary);
            /* Decode integer float-carriers in the compact temporary before
             * the GPU blit / CPU shadow mirror (same contract as VS XFB). */
            if (xfbTempBytes) {
                for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                    if (scatterParams.buffers[b].stride == 0u) continue;
                    NSUInteger region = bufferWritten[b];
                    if (region == 0u) continue;
                    mglXfbDecodeIntCarriersInBytes(
                        xfbTempBytes + bufferPhysBase[b], (uint64_t)region,
                        scatterParams.buffers[b].stride, program, b,
                        _GEOMETRY_SHADER);
                }
            }
            for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                if (!bufferDstMTL[b] || scatterParams.buffers[b].stride == 0u)
                    continue;
                NSUInteger copyBytes = bufferWritten[b];
                if (copyBytes == 0u) continue;
                if (copyBytes > bufferRemaining[b])
                    copyBytes = bufferRemaining[b];
                if (copyBytes == 0u) continue;
                if (!xfbBlit) {
                    xfbBlit = mglDrawSupportCreateBlitEncoder(
                        _renderPassManager.state->currentCommandBufferOwner);
                    if (!xfbBlit) {
                        _geometry.expansionActive = NO;
                        _geometry.program = NULL;
                        drawCtx->active_state->dirty_bits = DIRTY_ALL;
                        return YES;
                    }
                }
                mglDrawSupportBlitCopyBuffer(xfbBlit, xfbTemporary,
                                             bufferPhysBase[b],
                                             bufferDstMTL[b],
                                             bufferDstOffset[b], copyBytes);
                BufferBaseTarget *slot = &MGL_STATE(drawCtx)
                    ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[b];
                if (slot->buf) {
                    slot->buf->ever_written = GL_TRUE;
                    if (xfbTempBytes && slot->buf->data.buffer_data &&
                        (size_t)slot->buf->size >=
                            bufferDstOffset[b] + copyBytes) {
                        memcpy((uint8_t *)slot->buf->data.buffer_data +
                                   bufferDstOffset[b],
                               xfbTempBytes + bufferPhysBase[b], copyBytes);
                        mglRenderMarkBufferCPUWrite(
                            slot->buf, (int64_t)bufferDstOffset[b],
                            (int64_t)copyBytes);
                    }
                    uint8_t *liveBase = (uint8_t *)mglDrawSupportBufferContents(
                        bufferDstMTL[b]);
                    if (xfbTempBytes && liveBase) {
                        MGLRenderBufferInfo liveInfo = {0};
                        if (mglRenderGetBufferInfo(
                                (__bridge void *)bufferDstMTL[b],
                                &liveInfo) == 0 &&
                            bufferDstOffset[b] + copyBytes <= liveInfo.length) {
                            memcpy(liveBase + bufferDstOffset[b],
                                   xfbTempBytes + bufferPhysBase[b],
                                   copyBytes);
                        }
                    }
                }
                xfbState->buffer_write_offsets[b] = mglXfbAdvanceWriteOffset(
                    xfbState->buffer_write_offsets[b], (uint64_t)copyBytes);
            }
            if (xfbBlit) mglDrawSupportEndBlitEncoder(xfbBlit);
        }
        /* stream 0 (non-indexed) written primitives come from buffer 0's
         * written bytes; the buffer-0 record stride is the per-primitive
         * packed size for the captured stream-0 varyings. */
        queryWritten = mglDrawGsQueryWritten(
            outputPrimitive, scatterParams.buffers[0].stride,
            (uint64_t)bufferWritten[0]);
        /* Indexed stream>0 generated counters stay in the meta. */
    }
    if (!queryMeta && xfbMetaBuf && mglDrawSupportBufferContents(xfbMetaBuf) &&
        (mglHasActiveIndexedPrimitiveQuery(drawCtx) ||
         mglHasActivePrimitiveQuery(drawCtx) ||
         mglHasActiveGeometryShaderQuery(drawCtx))) {
        queryMeta = (const MGLAIRGSXFBMeta *)mglDrawSupportBufferContents(xfbMetaBuf);
    }
    if (gsQueryCountersReady && counts &&
        mglDrawSupportBufferContents(counts)) {
        queryGenerated = mglDrawGsReduceGeneratedPrimitives(
            gsOutputMode, (uint32_t)workItemCount, maxVertices,
            (const uint32_t *)mglDrawSupportBufferContents(counts), queryMeta);
    } else if (queryMeta) {
        queryGenerated = mglDrawGsReduceGeneratedPrimitives(
            gsOutputMode, (uint32_t)workItemCount, maxVertices, NULL,
            queryMeta);
    }
    if (mglDrawGsSkipRaster(xfbActive ? 1 : 0,
                            MGL_STATE(drawCtx)->caps.rasterizer_discard
                                ? 1
                                : 0)) {
        /* GL_RASTERIZER_DISCARD: no pixels by definition; the compute
         * expansion already ran and the primitive query must still count
         * the generated/written primitives (persistent query semantics). */
        _currentCBHasWork = YES;
        mglRecordGeometryPrimitiveQueries(
            drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
            gsStreamCount, bufferWritten, bufferStride, workItemCount);
        _geometry.expansionActive = NO;
        _geometry.program = NULL;
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return YES;
    }
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG rasterize-check empty=%d culled=%d enc=%d",
              (int)[self currentDrawRasterizationIsEmpty],
              (int)[self currentDrawModeIsFullyCulled:gsOutputMode],
              (int)mglRenderEncoderOwnerHasCurrent(
                  _renderPassManager.state->currentRenderEncoderOwner));
    if (!mglDrawGsPassthroughRasterReady(
            [self processGLState:true] ? 1 : 0,
            mglRenderEncoderOwnerHasCurrent(
                _renderPassManager.state->currentRenderEncoderOwner),
            [self currentDrawRasterizationIsEmpty] ? 1 : 0,
            [self currentDrawModeIsFullyCulled:gsOutputMode] ? 1 : 0)) {
        if (xfbActive || mglHasActiveIndexedPrimitiveQuery(drawCtx) ||
            mglHasActivePrimitiveQuery(drawCtx) ||
            mglHasActiveGeometryShaderQuery(drawCtx)) {
            _currentCBHasWork = YES;
            mglRecordGeometryPrimitiveQueries(
                drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
                gsStreamCount, bufferWritten, bufferStride, workItemCount);
        }
        _geometry.expansionActive = NO;
        _geometry.program = NULL;
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return YES;
    }

    /* The GS compute dispatch ended the render encoder and processGLState
     * rebuilt it, but the dirty-domain resource sync may have been marked
     * done for the *previous* encoder. Rebind fragment-stage buffers
     * (plain uniforms etc.) and storage images on the fresh encoder before
     * the indirect draws, or the fragment shader reads unbound slots. */
    if (!getenv("MGL_ABLATE_GS_REBIND")) {
        /* The binding-state dedup still reflects the pre-compute encoder;
         * clear the fragment tables so the rebind below is not skipped. */
        for (uint32_t slot = 0u; slot < 31u; slot++)
            mglRenderBindingClearFragmentBuffer(_bindingStateOwner, slot);
        const uint32_t texSlots = (uint32_t)TEXTURE_UNITS;
        for (uint32_t slot = 0u; slot < texSlots; slot++)
            mglRenderBindingClearFragmentTexture(_bindingStateOwner, slot);
        MGLEncodeContext gsEncCtx = {
            .render_encoder_owner =
                _renderPassManager.state->currentRenderEncoderOwner,
        };
        [self bindFragmentBuffersToCurrentRenderEncoder:&gsEncCtx];
        [self bindBufferSizeConstantsForRenderEncoder];
        Program *gsVertexProgram = mglResolveProgramForStageFromState(
            drawCtx, _VERTEX_SHADER);
        Program *gsFragmentProgram = mglResolveProgramForStageFromState(
            drawCtx, _FRAGMENT_SHADER);
        if (![self bindStorageImagesForVertexProgram:gsVertexProgram
                                     fragmentProgram:gsFragmentProgram]) {
            _geometry.expansionActive = NO;
            _geometry.program = NULL;
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
    }
    [self applyPolygonOffsetForDrawMode:gsOutputMode];
    if (getenv("MGL_SYNC_AFTER_GS")) {
        [self flushCommandBuffer:YES];
        NSLog(@"MGL GS sync: flushed after compute");
    }
    if (getenv("MGL_GS_DIAG")) {
        const uint32_t *cw = (const uint32_t *)mglDrawSupportBufferContents(counts);
        NSLog(@"MGL GS DIAG draw counts w0..6: %u %u %u %u %u %u %u outputBuf=%p",
              cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], output);
        for (NSUInteger w = 0u; w < workItemCount; w++)
            NSLog(@"MGL GS DIAG counts[%lu] full: %u %u %u %u %u %u %u",
                  (unsigned long)w,
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 1],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 2],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 3],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 4],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 5],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 6]);
        {
            const float *of = (const float *)mglDrawSupportBufferContents(output);
            NSLog(@"MGL GS DIAG output floats [1040B]=%g,%g,%g,%g [1920B]=%g,%g [2800B]=%g,%g",
                  of[260], of[261], of[262], of[263],
                  of[480], of[481], of[700], of[701]);
        }
    }
    if (getenv("MGL_GS_SINGLE_DRAW")) {
        /* bisect: one draw over all records; header records carry pos=0
         * and are clipped away by the rasterizer. */
        const uint32_t totalVerts =
            (uint32_t)(workItemCount * recordsPerPrimitive) -
            MGL_AIR_GS_HEADER_RECORDS;
        uint32_t *cw1 = (uint32_t *)mglDrawSupportBufferContents(counts);
        cw1[0] = totalVerts;
        cw1[1] = 1u;
        mglDrawSupportSetVertexBuffer(
            _renderPassManager.state->currentRenderEncoderOwner, output,
            MGL_AIR_GS_HEADER_RECORDS * outputStride, 0u);
        mglDrawSupportDrawPrimitives(
            _renderPassManager.state->currentRenderEncoderOwner,
            outputPrimitive, 0u, totalVerts, 1u, 0u);
        goto after_gs_draws;
    }
    if (getenv("MGL_GS_COPY_DRAW")) {
        /* bisect: copy each work item's records to a fresh offset-0 buffer
         * (CPU readback -> newBufferWithBytes) and draw from that. */
        const uint8_t *src = (const uint8_t *)mglDrawSupportBufferContents(output);
        for (GLuint w = 0u; w < workItemCount; w++) {
            NSUInteger srcOff =
                ((NSUInteger)w * recordsPerPrimitive +
                 MGL_AIR_GS_HEADER_RECORDS) * outputStride;
            NSUInteger bytes = (NSUInteger)9u * outputStride;
            id sub = mglDrawSupportCreateBufferWithBytes(
                _device, src + srcOff, bytes, 0u);
            const uint32_t *cwv = (const uint32_t *)mglDrawSupportBufferContents(counts);
            mglDrawSupportSetVertexBuffer(
                _renderPassManager.state->currentRenderEncoderOwner, sub, 0u, 0u);
            mglDrawSupportDrawPrimitives(
                _renderPassManager.state->currentRenderEncoderOwner,
                outputPrimitive, 0u,
                cwv ? cwv[w * MGL_AIR_GS_COUNTS_RECORD_WORDS] : 0u, 1u, 0u);
        }
        goto after_gs_draws;
    }
    const bool gsDiagEncode =
        getenv("MGL_GS_ONLY_PRIM") || getenv("MGL_GS_DRAW_OFFSET") ||
        getenv("MGL_GS_VSTART_DRAW") || getenv("MGL_GS_BIND_INPUT") ||
        getenv("MGL_GS_DIRECT_DRAW") || getenv("MGL_GS_DRAW_VCOUNT") ||
        getenv("MGL_GS_REVERSE_DRAW") || getenv("MGL_GS_DIAG");
    if (!gsDiagEncode) {
        const MGLGsPassthroughEncodeState gsEnc = {
            .encoder_owner =
                _renderPassManager.state->currentRenderEncoderOwner,
            .output_buffer = (__bridge void *)output,
            .counts_buffer = (__bridge void *)counts,
            .output_primitive = outputPrimitive,
            .work_item_count = (uint32_t)workItemCount,
            .records_per_primitive = (uint32_t)recordsPerPrimitive,
            .output_stride = (uint32_t)outputStride,
            .counts_record_bytes = (uint32_t)countsRecordBytes,
        };
        mglDrawGsEncodePassthrough(&gsEnc);
        goto after_gs_draws;
    }
    const char *onlyPrim = getenv("MGL_GS_ONLY_PRIM");
    for (GLuint iter = 0u; iter < workItemCount; iter++) {
        GLuint primitive = getenv("MGL_GS_REVERSE_DRAW")
            ? (workItemCount - 1u - iter) : iter;
        if (onlyPrim && (GLint)primitive != atoi(onlyPrim)) continue;
        const char *offOverride = getenv("MGL_GS_DRAW_OFFSET");
        NSUInteger offset =
            ((NSUInteger)primitive * recordsPerPrimitive +
             MGL_AIR_GS_HEADER_RECORDS) * outputStride;
        if (offOverride) offset = (NSUInteger)atol(offOverride);
        if (getenv("MGL_GS_VSTART_DRAW")) {
            /* bisect: bind at 0 and use indirect vertexStart to select the
             * work item's records (gl_VertexID starts at vertexStart). */
            static uint32_t *cwStart = NULL;
            cwStart = (uint32_t *)mglDrawSupportBufferContents(counts);
            cwStart[primitive * MGL_AIR_GS_COUNTS_RECORD_WORDS + 2] =
                primitive * (uint32_t)recordsPerPrimitive + 2u;
            offset = 0u;
        }
        id ptvsSource = output;
        NSUInteger ptvsOffset = offset;
        if (getenv("MGL_GS_BIND_INPUT")) {
            /* Diagnostic: point the passthrough VS at the capture buffer
             * instead of the kernel output so the rendered image shows
             * what the GPU actually wrote per input vertex.  An explicit
             * MGL_GS_DRAW_OFFSET still wins, for byte-range scans. */
            ptvsSource = input;
            if (!getenv("MGL_GS_DRAW_OFFSET"))
                ptvsOffset = inputOffset;
            offset = 0u;
        }
        mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, ptvsSource, ptvsOffset, 0u);
        if (getenv("MGL_GS_DIRECT_DRAW")) {
            const uint32_t *cw2 = (const uint32_t *)mglDrawSupportBufferContents(counts);
            mglDrawSupportDrawPrimitives(
                _renderPassManager.state->currentRenderEncoderOwner, outputPrimitive,
                0u, cw2 ? cw2[primitive * MGL_AIR_GS_COUNTS_RECORD_WORDS] : 0u,
                1u, 0u);
        } else if (getenv("MGL_GS_DRAW_VCOUNT")) {
            mglDrawSupportDrawPrimitives(
                _renderPassManager.state->currentRenderEncoderOwner, outputPrimitive,
                0u, (NSUInteger)atol(getenv("MGL_GS_DRAW_VCOUNT")), 1u, 0u);
        } else
        mglDrawSupportDrawPrimitivesIndirect(
            _renderPassManager.state->currentRenderEncoderOwner, outputPrimitive, counts,
            (offOverride ? 0u : (NSUInteger)primitive * countsRecordBytes));
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG pre-draw prim=%u enc=%d",
                  primitive,
                  (int)mglRenderEncoderOwnerHasCurrent(
                      _renderPassManager.state->currentRenderEncoderOwner));
            const float *op = (const float *)((const uint8_t *)mglDrawSupportBufferContents(output) + offset);
            NSLog(@"MGL GS DIAG draw prim=%u offset=%lu firstRec={%g,%g,%g,%g}",
                  primitive, (unsigned long)offset,
                  op[0], op[1], op[2], op[3], 0.0);
        }
    }
after_gs_draws:
    if (getenv("MGL_GS_POST_DIAG")) {
        /* Dump the output records after the frame's GPU work completes
         * (the caller's glFinish/ReadPixels drains the encoders), so the
         * CPU view reflects what the rasterizing draws actually read. */
        [self flushCommandBuffer:YES];
        const uint8_t *outBytes =
            (const uint8_t *)mglDrawSupportBufferContents(output);
        NSLog(@"MGL GS POST-DIAG outputStride=%lu recordsPerPrim=%lu",
              (unsigned long)outputStride, (unsigned long)recordsPerPrimitive);
        if (input) {
            const uint8_t *inBytes =
                (const uint8_t *)mglDrawSupportBufferContents(input);
            NSUInteger inStride = mglAIRPerVertexStrideForResources(
                &program->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
            for (GLuint vtx = 0u; vtx < MIN((GLuint)count, 6u); vtx++) {
                const float *p = (const float *)(inBytes +
                    (NSUInteger)(vtx + (GLuint)first) * inStride);
                NSLog(@"MGL GS POST-DIAG in.cap[%u] pos={%g,%g,%g,%g} vary@64={%g,%g,%g,%g} @80={%g,%g,%g,%g}",
                      (unsigned)(vtx + (GLuint)first),
                      p[0], p[1], p[2], p[3], p[16], p[17], p[18], p[19],
                      p[20], p[21], p[22], p[23]);
            }
        }
        for (GLuint wi = 0u; wi < MIN(workItemCount, 4u); wi++) {
            for (GLuint rec = 0u; rec < MIN(recordsPerPrimitive, 9u); rec++) {
                const float *p = (const float *)(outBytes +
                    ((NSUInteger)wi * recordsPerPrimitive + rec) * outputStride);
                NSLog(@"MGL GS POST-DIAG out[%u].rec[%u] pos={%g,%g,%g,%g} ps=%g cull=%g,%g vary={%g,%g,%g,%g}",
                      (unsigned)wi, (unsigned)rec,
                      p[0], p[1], p[2], p[3],
                      p[4], p[5], p[6],
                      p[16], p[17], p[18], p[19]);
            }
        }
    }
    _currentCBHasWork = YES;
    if (getenv("MGL_GPU_CAPTURE")) {
        [self flushCommandBuffer:YES];
        [self mglStopCapture];
        NSLog(@"MGL GPU capture stopped");
    }
    mglRecordGeometryPrimitiveQueries(
        drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
        gsStreamCount, bufferWritten, bufferStride, workItemCount);
    _geometry.expansionActive = NO;
    _geometry.program = NULL;
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    return YES;
}

- (bool) validateDrawArraysVertexInputs:(GLMContext)drawCtx
                                    mode:(GLenum)mode
                                   first:(GLint)first
                                   count:(GLsizei)count
                                drawCall:(uint64_t)drawCall
{
    if (!mglVboRangeValidationEnabled()) {
        return true;
    }

    if (!drawCtx) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=null_ctx mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    if (count == 0) {
        return false;
    }

    if (count < 0 || first < 0) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_range mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t firstVertex = (uint64_t)(uint32_t)first;
    uint64_t vertexCount = (uint64_t)(uint32_t)count;
    if (vertexCount == 0u || firstVertex > UINT64_MAX - (vertexCount - 1u)) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=vertex_range_overflow mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t lastVertex = firstVertex + vertexCount - 1u;
    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "drawArrays.vboRange");
    if (!vao) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_vao mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    GLuint maxAttribs = MAX_ATTRIBS;

    for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
        if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
            continue;
        }

        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(drawCtx,
                                                   vao,
                                                   attrib,
                                                   "drawArrays.vboRange",
                                                   &resolved)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u reason=invalid_vbo mode=0x%x first=%d count=%d",
                  (unsigned long long)drawCall, (unsigned)attrib, (unsigned)mode, (int)first, (int)count);
            return false;
        }
        const VertexAttrib *a = resolved.attrib;
        Buffer *vbo = resolved.buffer;

        if (!mglRendererBufferHasDrawableContents(vbo)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=never_written "
                  "init(source=%u mapped=%u access=0x%x accessFlags=0x%x full=%u range=[%lld,%lld) lastOff=%lld lastSize=%lld src=%p hash=0x%016llx)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)vbo->last_init_source,
                  (unsigned)vbo->mapped,
                  (unsigned)vbo->access,
                  (unsigned)vbo->access_flags,
                  (unsigned)vbo->has_initialized_data,
                  (long long)vbo->written_min,
                  (long long)vbo->written_max,
                  (long long)vbo->last_write_offset,
                  (long long)vbo->last_write_size,
                  vbo->last_write_src_ptr,
                  (unsigned long long)vbo->last_write_src_hash);
            return false;
        }

        if (resolved.binding_offset < 0 || resolved.relativeoffset < 0) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=negative_attrib_offset bindingOffset=%lld relativeOffset=%lld",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (long long)resolved.binding_offset,
                  (long long)resolved.relativeoffset);
            return false;
        }

        size_t compSize = mglVertexAttribComponentSize(a->type);
        size_t compCount = (size_t)a->size;
        if (compSize == 0u || compCount == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=invalid_attrib_format type=0x%x size=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)a->type,
                  (unsigned)a->size);
            return false;
        }

        if (compCount > (SIZE_MAX / compSize)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=elem_size_overflow compSize=%zu compCount=%zu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  compSize,
                  compCount);
            return false;
        }

        uint64_t elemBytes = (uint64_t)(compSize * compCount);
        uint64_t stride = (resolved.stride > 0u) ? (uint64_t)resolved.stride : elemBytes;
        uint64_t bindingOffset = (uint64_t)resolved.binding_offset;
        uint64_t attrRelativeOffset = (uint64_t)resolved.relativeoffset;
        if (bindingOffset > UINT64_MAX - attrRelativeOffset) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=offset_overflow bindingOffset=%llu relativeOffset=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)attrRelativeOffset);
            return false;
        }
        uint64_t relOffset = bindingOffset + attrRelativeOffset;
        if (stride == 0u || elemBytes == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=zero_stride_or_elem stride=%llu elem=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)stride,
                  (unsigned long long)elemBytes);
            return false;
        }

        // Per-instance attributes are still consumed by a non-instanced draw for
        // instance zero, so validate element zero instead of ignoring them.
        uint64_t rangeFirst = (resolved.divisor != 0u) ? 0u : firstVertex;
        uint64_t rangeLast = (resolved.divisor != 0u) ? 0u : lastVertex;

        if (relOffset > UINT64_MAX - elemBytes) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)resolved.divisor);
            return false;
        }

        if (rangeLast > (UINT64_MAX - relOffset - elemBytes) / stride ||
            rangeFirst > (UINT64_MAX - relOffset) / stride) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow "
                  "range=[%llu,%llu] stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)resolved.divisor);
            return false;
        }

        uint64_t byteStart = relOffset + (rangeFirst * stride);
        uint64_t byteEnd = relOffset + (rangeLast * stride) + elemBytes;
        uint64_t vboSize = (vbo->size > 0) ? (uint64_t)vbo->size : 0u;
        if (byteEnd > vboSize) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=vbo_oob "
                  "vertexRange=[%llu,%llu] byteRange=[%llu,%llu) vboSize=%llu "
                  "mode=0x%x first=%d count=%d stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu type=0x%x size=%u divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned)mode,
                  (int)first,
                  (int)count,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)a->type,
                  (unsigned)a->size,
                  (unsigned)resolved.divisor);
            return false;
        }

        if (!vbo->data.mtl_data) {
            [self bindMTLBuffer:vbo];
        }
        if (!vbo->data.mtl_data) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=no_mtl_buffer byteRange=[%llu,%llu)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd);
            return false;
        }

        id mtlBuffer = (__bridge id)(vbo->data.mtl_data);
        if (!mtlBuffer) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=mtl_bridge_nil",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name);
            return false;
        }

        uint64_t metalLen = (uint64_t)mglDrawSupportBufferLength(mtlBuffer);
        if (byteEnd > metalLen) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=metal_oob "
                  "byteRange=[%llu,%llu) metalLen=%llu vboSize=%llu first=%d count=%d",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)metalLen,
                  (unsigned long long)vboSize,
                  (int)first,
                  (int)count);
            return false;
        }

        if (vbo->written_min >= 0 && vbo->written_max >= 0) {
            uint64_t writtenMin = (uint64_t)vbo->written_min;
            uint64_t writtenMax = (uint64_t)vbo->written_max;
            if (byteStart < writtenMin || byteEnd > writtenMax) {
                NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=unwritten_range "
                      "byteRange=[%llu,%llu) written=[%llu,%llu) first=%d count=%d source=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned long long)byteStart,
                      (unsigned long long)byteEnd,
                      (unsigned long long)writtenMin,
                      (unsigned long long)writtenMax,
                      (int)first,
                      (int)count,
                      (unsigned)vbo->last_init_source);
                return false;
            }
        }

        GLuint drawProgramKey = mglCurrentRenderProgramKey(drawCtx);
        if (mglShouldInspectDrawCall(drawCall, drawProgramKey) && attrib == 0u) {
            mglTraceLogNSString(@"MGL TRACE drawArrays.attrib0 call=%llu program=%u buffer=%u first=%d count=%d "
                  "byteRange=[%llu,%llu) vboSize=%llu metalLen=%llu stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)drawProgramKey,
                  (unsigned)vbo->name,
                  (int)first,
                  (int)count,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned long long)metalLen,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes);
        }
    }

    return true;
}

- (BOOL)resolveElementBufferForDraw:(const char *)label
                            context:(GLMContext)drawCtx
                           glBuffer:(Buffer **)glBufferOut
                          mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = getElementBuffer(drawCtx);
    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBufferForCommand:(const MGLDrawCommand *)cmd
                                  label:(const char *)label
                                context:(GLMContext)drawCtx
                               glBuffer:(Buffer **)glBufferOut
                              mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = NULL;
    if (cmd && cmd->element_buffer_name) {
        gl_element_buffer = mglRendererGetValidatedBuffer(drawCtx,
                                                          mglDrawCommandElementBuffer(drawCtx, cmd),
                                                          label ? label : "deferred indexed draw",
                                                          0);
        if (!gl_element_buffer) {
            return NO;
        }
    } else {
        gl_element_buffer = getElementBuffer(drawCtx);
    }

    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBuffer:(Buffer *)gl_element_buffer
                       label:(const char *)label
                     context:(GLMContext)drawCtx
                    glBuffer:(Buffer **)glBufferOut
                   mtlBuffer:(id *)mtlBufferOut
{
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no element array buffer is bound", label ? label : "indexed draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_element_buffer] == false) {
        return NO;
    }

    id indexBuffer = (__bridge id)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: %s skipped because element buffer %u has no Metal buffer",
              label ? label : "indexed draw",
              gl_element_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_element_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indexBuffer;
    }
    return YES;
}

- (BOOL)resolveIndirectBufferForDraw:(const char *)label
                             context:(GLMContext)drawCtx
                            glBuffer:(Buffer **)glBufferOut
                           mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(drawCtx);
    if (!gl_indirect_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no draw indirect buffer is bound", label ? label : "indirect draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_indirect_buffer] == false) {
        return NO;
    }

    id indirectBuffer = (__bridge id)(gl_indirect_buffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer",
              label ? label : "indirect draw",
              gl_indirect_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_indirect_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indirectBuffer;
    }
    return YES;
}

- (BOOL)prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label
{
    if (!drawCtx) {
        NSLog(@"MGL WARNING: %s skipped because context is NULL",
              label ? label : "indirect emulation");
        return NO;
    }

    /* The C draw-indirect frontends already flush pending command buffers before
     * dispatching into these Metal entry points. If processGLState has just
     * rebuilt a render encoder, keep it; a second flush can discard the fresh
     * pass and make state restoration fail for CPU-emulated indirect modes. */
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        NSLog(@"MGL WARNING: %s skipped because CPU-read synchronization left no render encoder",
              label ? label : "indirect emulation");
        return NO;
    }
    return YES;
}

- (BOOL)currentDrawRasterizationIsEmpty
{
    if (!ctx) {
        return NO;
    }

    GLint vx = MGL_STATE(ctx)->viewport[0];
    GLint vy = MGL_STATE(ctx)->viewport[1];
    GLint vw = MGL_STATE(ctx)->viewport[2];
    GLint vh = MGL_STATE(ctx)->viewport[3];

    NSUInteger passWidth = 0;
    NSUInteger passHeight = 0;
    mglRenderGetRenderTargetSizeOwner(
        _renderPassManager.state->renderPassStateOwner,
        (uint64_t *)&passWidth, (uint64_t *)&passHeight);
    if (passWidth == 0 || passHeight == 0) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id color = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i);
            if (color) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(color);
                passWidth = info.width;
                passHeight = info.height;
                break;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id depth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            if (depth) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(depth);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id stencil = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            if (stencil) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(stencil);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
    }


    return mglRenderRasterizationIsEmpty(
               vx, vy, vw, vh,
               (uint32_t)passWidth, (uint32_t)passHeight,
               MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
               MGL_STATE(ctx)->var.scissor_box[0],
               MGL_STATE(ctx)->var.scissor_box[1],
               MGL_STATE(ctx)->var.scissor_box[2],
               MGL_STATE(ctx)->var.scissor_box[3]) != 0;
}

- (void)applyPolygonOffsetForDrawMode:(GLenum)mode
{
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return;
    }


    MGLRenderPolygonOffsetDecision decision = {0};
    mglRenderPolygonOffsetDecision(
        (uint32_t)mode,
        ctx ? 1 : 0,
        mglDrawModeProducesPolygons(mode) ? 1 : 0,
        (uint32_t)(ctx ? MGL_STATE(ctx)->var.polygon_mode : 0u),
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_point) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_line) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_fill) ? 1 : 0,
        &decision);
    uint32_t triangleFillMode = decision.triangle_fill_mode ? 1u : 0u;
    if (decision.needs_polygon_mode_repair) {
        mglLogRenderStateRepair("polygon_mode", MGL_STATE(ctx)->var.polygon_mode, GL_FILL);
        MGL_STATE(ctx)->var.polygon_mode = GL_FILL;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];

    BOOL enableDepthBias = decision.enable_depth_bias != 0;

    if (enableDepthBias) {
        float _bias = MGL_STATE(ctx)->var.polygon_offset_units;
        float _slope = MGL_STATE(ctx)->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    } else {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            0.0f, 0.0f, 0.0f);
    }
}

- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode
{
    return ctx &&
           MGL_STATE(ctx)->caps.cull_face &&
           MGL_STATE(ctx)->var.cull_face_mode == GL_FRONT_AND_BACK &&
           mglDrawModeProducesPolygons(mode);
}

- (BOOL)ensureRasterEncoderForDraw
{
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }
    [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_DRAW];
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return NO;
    }
    if (!_pipelineCache.state->pipelineState) {
        return NO;
    }

    uint32_t rpColor0Format = 0u;
    uint32_t rpDepthFormat = 0u;
    uint32_t rpStencilFormat = 0u;
    MGLRenderPassAttachmentState colorAttachment = {0};
    MGLRenderPassAttachmentState depthAttachment = {0};
    MGLRenderPassAttachmentState stencilAttachment = {0};
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, &colorAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, &depthAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, &stencilAttachment);
    id rpColor0 = (__bridge id)colorAttachment.texture;
    id rpDepth = (__bridge id)depthAttachment.texture;
    id rpStencil = (__bridge id)stencilAttachment.texture;
    MGLRenderTextureInfo textureInfo = {0};
    if (rpColor0 && mglRenderGetTextureInfo(
            (__bridge void *)rpColor0, &textureInfo) == 0) {
        rpColor0Format = textureInfo.pixel_format;
    }
    if (rpDepth && mglRenderGetTextureInfo(
            (__bridge void *)rpDepth, &textureInfo) == 0) {
        rpDepthFormat = textureInfo.pixel_format;
    }
    if (rpStencil && mglRenderGetTextureInfo(
            (__bridge void *)rpStencil, &textureInfo) == 0) {
        rpStencilFormat = textureInfo.pixel_format;
    }

    const BOOL colorMismatch =
        (_pipelineCache.state->pipelineColor0Format != 0u &&
         rpColor0Format != 0u &&
         _pipelineCache.state->pipelineColor0Format != rpColor0Format);
    const BOOL depthMismatch =
        (_pipelineCache.state->pipelineDepthFormat != rpDepthFormat);
    const BOOL stencilMismatch =
        (_pipelineCache.state->pipelineStencilFormat != rpStencilFormat);
    if (colorMismatch || depthMismatch || stencilMismatch) {
        return NO;
    }
    if (mglRenderSetRenderPipelineStateForOwner(
            _renderPassManager.state->currentRenderEncoderOwner,
            _pipelineCache.state->pipelineState) != 0) {
        return NO;
    }
    mglRenderBindingSetPipelineState(_bindingStateOwner,
                                     _pipelineCache.state->pipelineState);
    MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
    return YES;
}

- (void)bindCullDistanceEmulationBuffers:(GLenum)mode
                             firstVertex:(GLuint)firstVertex
                        explicitVertices:(const GLuint *)explicitVertices
                      explicitVertexCount:(GLuint)explicitVertexCount
                           encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!ctx || !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, "bindCullDistanceEmu");
    if (!vao) {
        return;
    }
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (!activeProgram) {
        return;
    }
    explicitVertexCount = MIN(explicitVertexCount, 4u);

    id captureBuffer = (__bridge id)
        mglRendererBackendGetCullDistanceCaptureBuffer(_backend);
    if (captureBuffer) {
        MGLCullDistanceEmuParams params;
        mglRenderFillCullDistanceEmuParams(
            mglRenderPrimitiveVertexCountForMode((uint32_t)mode), firstVertex,
            explicitVertices, explicitVertexCount, 0u, 32u,
            MIN(activeProgram->cull_distance_count, 8u),
            _tessellation.cullDistanceCaptureFirstInstance,
            _tessellation.cullDistanceCaptureInstanceStride, &params);
        mglRenderBindCullDistanceEmuSlots(encCtx->render_encoder_owner,
                                          (__bridge void *)captureBuffer,
                                          &params);
        [self recordLastBoundVertexBuffer:
                  captureBuffer
                                   offset:0
                                  atIndex:kMGLCullDistanceVertexBufferIndex];
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        [self invalidateLastBoundVertexBufferAtIndex:
                  kMGLCullDistanceParamsBufferIndex];
        return;
    }

    uint32_t attribs[MAX_ATTRIBS];
    const uint32_t attribCount = mglRenderCollectCullDistanceAttribs(
        activeProgram, attribs, MAX_ATTRIBS);
    MGLRenderCullDistanceLayout layout;
    memset(&layout, 0, sizeof(layout));
    for (uint32_t i = 0u; i < attribCount; i++) {
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(
                ctx, vao, attribs[i], "bindCullDistanceEmu", &resolved)) {
            continue;
        }
        if (!resolved.buffer || !resolved.buffer->data.mtl_data) {
            continue;
        }
        mglRenderAccumulateCullDistanceAttrib(
            &layout, resolved.buffer->data.mtl_data, resolved.binding_offset,
            resolved.stride, resolved.relativeoffset);
    }

    void *cullMtlBuffer = layout.mtl_buffer;
    uint32_t cullStride = layout.stride;
    uint32_t cullDistSize = layout.culldist_size;
    if (!cullMtlBuffer || cullDistSize == 0u) {
        cullMtlBuffer = mglRendererBackendGetCullDistanceDummyBuffer(_backend);
        layout.mtl_buffer = cullMtlBuffer;
        layout.binding_offset = 0;
        layout.stride = 4u;
        layout.first_relative_offset = 0;
        layout.culldist_size = 0u;
        cullStride = 4u;
        cullDistSize = 0u;
    }

    MGLCullDistanceEmuParams params;
    mglRenderFillCullDistanceEmuParams(
        mglRenderPrimitiveVertexCountForMode((uint32_t)mode), firstVertex,
        explicitVertices, explicitVertexCount,
        mglRenderCullDistanceLayoutOffset(&layout), cullStride, cullDistSize,
        0u, 0u, &params);
    mglRenderBindCullDistanceEmuSlots(encCtx->render_encoder_owner,
                                      cullMtlBuffer, &params);
    [self recordLastBoundVertexBuffer:(__bridge id)cullMtlBuffer
                               offset:0
                              atIndex:kMGLCullDistanceVertexBufferIndex];
    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    [self invalidateLastBoundVertexBufferAtIndex:kMGLCullDistanceParamsBufferIndex];
}

- (BOOL)handleTessellationPatchDrawIfNeeded:(GLMContext)drawCtx
                                        mode:(GLenum *)mode
                                       first:(GLint)first
                                       count:(GLsizei)count
                                   indexType:(GLenum)indexType
                                     indices:(const void *)indices
                                  baseVertex:(GLint)baseVertex
                               instanceCount:(GLsizei)instanceCount
                                baseInstance:(GLuint)baseInstance
                                       label:(const char *)label
{
    if (!mode) {
        return NO;
    }
    self->ctx = drawCtx;

    Program *tcsProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_CONTROL_SHADER);
    Program *tesProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_EVALUATION_SHADER);
    Program *gsProgram = mglResolveProgramForStageFromState(drawCtx, _GEOMETRY_SHADER);
    MGLTessDrawPathPlan path = {0};
    if (!mglTessPlanDrawPath(drawCtx, *mode, count, instanceCount, tcsProgram,
                             tesProgram, gsProgram, indexType, label, &path)) {
        return NO;
    }
    if (path.classify == MGL_TESS_DRAW_NOT_APPLICABLE) {
        return NO;
    }
    if (path.classify != MGL_TESS_DRAW_ACTIVE) {
        return YES;
    }
    if (!path.has_tcs) {
        tcsProgram = NULL;
    }
    if (!path.has_tes) {
        tesProgram = NULL;
    }

    if (tcsProgram) {
        if (tcsProgram->dirty_bits) {
            [self bindMTLProgram:tcsProgram];
        }
    }

    if (tesProgram) {
        if (tesProgram->dirty_bits) {
            [self bindMTLProgram:tesProgram];
        }
    }

    const BOOL airTES = path.air_tes != 0u;
    BOOL nativeTES = path.native_ok != 0u;

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    MGLAIRTessDrawContract contract;
    mglTessFillDrawContract(&contract, drawCtx, tcsProgram, tesProgram,
                            vertexProgram, first, count, indexType, indices,
                            baseVertex, instanceCount, baseInstance);
    GLuint patchVertices = contract.patch_vertices;
    GLuint patchCount = contract.patch_count;
    const bool restartEnabled = contract.primitive_restart != 0u;
    const uint32_t restartIndex = contract.restart_index;

    (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
    _tessellation.tessVertexCaptureOffset = 0u;
    (void)mglRendererBackendSetTessControlPointIndexBuffer(_backend, NULL);
    _tessellation.tessIndexedDraw = NO;
    _tessellation.tessInstanceRecords = 0u;
    /* A TCS from a previous draw must not leak into a TES-only dispatch
     * (dispatchAIRTessEvalCompute reads tcsOutputBuffer as the gl_in
     * source when non-nil).  The TCS dispatcher re-populates it. */
    (void)mglRendererBackendSetTcsOutputBuffer(_backend, NULL);
    _tessellation.tcsOutputOffset = 0u;
    _tessellation.tcsOutputStride = 0u;
    _tessellation.tcsOutVertices = 0u;
    (void)mglRendererBackendSetCurrentTessFactorBuffer(_backend, NULL);
    /* The VS position capture and the default factor buffer are consumed by
     * both the native patch pipeline and the AIR TES compute expansion
     * (isolines / point_mode with no TCS), so they must exist even when
     * nativeTES is unavailable. */
    if (path.capture == MGL_TESS_CAPTURE_INDEXED_COMPACT) {
            /* TCS reads continuous [patch][control_point] records.  Capture
             * the VS into a sparse [vertex_id] buffer (runs VS atomics /
             * transforms), then compact into patch order for the TCS kernel.
             * Falling back to newTCSStageInBufferForContext would pack raw
             * attributes and skip the VS entirely. */
            nativeTES = NO;
            BOOL sparseCompactOk = NO;
            Buffer *ebo = getElementBuffer(drawCtx);
            if (ebo && [self processBuffer:ebo] && ebo->data.mtl_data) {
                id eboMetal = (__bridge id)ebo->data.mtl_data;
                const NSUInteger indexOffsetBytes =
                    (NSUInteger)(uintptr_t)indices;
                const uint8_t *indexBytes = mglElementIndexSourceForDraw(
                    ebo, eboMetal, indexType, indexOffsetBytes, count);
                uint32_t *gatherArray = NULL;
                uint32_t gatherCount = 0u;
                uint32_t gatherPrimitives = 0u;
                uint32_t gatherMaxIndex = 0u;
                if (indexBytes &&
                    mglGeometryGatherIndices(indexBytes, indexType, count,
                                             baseVertex, restartEnabled,
                                             restartIndex, patchVertices,
                                             &gatherArray, &gatherCount,
                                             &gatherPrimitives,
                                             &gatherMaxIndex) &&
                    gatherCount > 0u && gatherPrimitives > 0u) {
                    NSUInteger captureOffset = 0u;
                    id sparseCapture = [self
                        captureAIRVertexPositionsForGeometryIndexed:drawCtx
                                                        indexBuffer:eboMetal
                                                          indexType:indexType
                                                        indexOffset:indexOffsetBytes
                                                              count:count
                                                          baseVertex:baseVertex
                                                       instanceCount:instanceCount
                                                        baseInstance:baseInstance
                                                           maxIndex:gatherMaxIndex
                                                          outOffset:&captureOffset];
                    Program *captureVS = mglResolveProgramForStageFromState(
                        drawCtx, _VERTEX_SHADER);
                    MGLTessVertexCapturePlan compactPlan = {0};
                    const GLsizei instCount =
                        instanceCount > 0 ? instanceCount : 1;
                    const NSUInteger sparseRecords =
                        (NSUInteger)gatherMaxIndex + 1u;
                    if (sparseCapture &&
                        mglTessPlanVertexCapture(captureVS, gatherCount,
                                                 (uint32_t)instCount, 0u, 0u,
                                                 &compactPlan)) {
                        NSUInteger captureStride = compactPlan.capture_stride;
                        NSUInteger continuousSize =
                            (NSUInteger)compactPlan.capture_size;
                        _currentCBHasWork = YES;
                        [self flushCommandBuffer:YES];
                        const uint8_t *sparseBytes =
                            (const uint8_t *)mglDrawSupportBufferContents(
                                sparseCapture);
                        id continuous = mglDrawSupportCreateBuffer(
                            _device, continuousSize, 0u);
                        uint8_t *continuousBytes = continuous
                            ? (uint8_t *)mglDrawSupportBufferContents(
                                  continuous)
                            : NULL;
                        if (sparseBytes && continuousBytes &&
                            mglTessCompactSparseCapture(
                                sparseBytes, (uint64_t)captureOffset,
                                (uint32_t)sparseRecords,
                                (uint32_t)captureStride, gatherArray,
                                gatherCount, (uint32_t)instCount,
                                continuousBytes, (uint64_t)continuousSize)) {
                            (void)mglRendererBackendSetTessVertexCaptureBuffer(
                                _backend, (__bridge void *)continuous);
                            _tessellation.tessVertexCaptureOffset = 0u;
                            _tessellation.tessIndexedDraw = NO;
                            _tessellation.tessInstanceRecords =
                                (NSUInteger)gatherCount;
                            patchCount = gatherPrimitives;
                            mglTessApplyGatherToContract(&contract, gatherCount,
                                                         gatherPrimitives);
                            sparseCompactOk = YES;
                        }
                    }
                    free(gatherArray);
                } else {
                    free(gatherArray);
                }
            }
            if (!sparseCompactOk) {
                NSLog(@"MGL TESS ERROR: indexed TCS sparse capture failed");
                mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                                 GL_INVALID_OPERATION);
                (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend,
                                                                   NULL);
                _tessellation.tessVertexCaptureOffset = 0u;
                _tessellation.tessIndexedDraw = NO;
                _tessellation.tessInstanceRecords = 0u;
                drawCtx->active_state->dirty_bits = DIRTY_ALL;
                return YES;
            }
    } else if (path.capture == MGL_TESS_CAPTURE_INDEXED_GATHER) {
            /* Indexed native TES (no TCS): capture the VS once into sparse
             * per-vertex records [instance][vertex_id] and let the CPU
             * gather buffer (raw index stream) drive Metal's
             * controlPointIndexBuffer.  baseVertex is already applied by the
             * indexed capture draw.  Instances are drawn one at a time from
             * their contiguous capture spans because Metal patch draws have no
             * per-instance patch-data offset. */
            Buffer *ebo = getElementBuffer(drawCtx);
            if (!ebo || ![self processBuffer:ebo] || !ebo->data.mtl_data) {
                nativeTES = NO;
            } else {
                id eboMetal =
                    (__bridge id)ebo->data.mtl_data;
                const NSUInteger indexOffsetBytes =
                    (NSUInteger)(uintptr_t)indices;
                const uint8_t *indexBytes = mglElementIndexSourceForDraw(
                    ebo, eboMetal, indexType, indexOffsetBytes, count);
                uint32_t *gatherArray = NULL;
                uint32_t gatherCount = 0u;
                uint32_t gatherPrimitives = 0u;
                uint32_t gatherMaxIndex = 0u;
                if (!indexBytes ||
                    !mglGeometryGatherIndices(indexBytes, indexType, count,
                                              baseVertex, restartEnabled,
                                              restartIndex, patchVertices,
                                              &gatherArray, &gatherCount,
                                              &gatherPrimitives,
                                              &gatherMaxIndex)) {
                    nativeTES = NO;
                } else {
                    id gatherBuf =
                        mglDrawSupportCreateBufferWithBytes(
                            _device, gatherArray,
                            (NSUInteger)gatherCount * 4u,
                            0u);
                    free(gatherArray);
                    if (!gatherBuf) {
                        nativeTES = NO;
                    } else {
                        NSUInteger captureOffset = 0u;
                        id capture = [self
                            captureAIRVertexPositionsForGeometryIndexed:drawCtx
                                                            indexBuffer:eboMetal
                                                              indexType:indexType
                                                            indexOffset:indexOffsetBytes
                                                                  count:count
                                                              baseVertex:baseVertex
                                                           instanceCount:instanceCount
                                                            baseInstance:baseInstance
                                                               maxIndex:gatherMaxIndex
                                                               outOffset:&captureOffset];
                        if (!capture) {
                            nativeTES = NO;
                        } else {
                            (void)mglRendererBackendSetTessVertexCaptureBuffer(
                                _backend, (__bridge void *)capture);
                            _tessellation.tessVertexCaptureOffset = captureOffset;
                            (void)mglRendererBackendSetTessControlPointIndexBuffer(
                                _backend, (__bridge void *)gatherBuf);
                            _tessellation.tessIndexedDraw = YES;
                            _tessellation.tessInstanceRecords =
                                (NSUInteger)gatherMaxIndex + 1u;
                            /* The gather stream is already re-grouped into
                             * complete patches, so it is the real count. */
                            patchCount = gatherPrimitives;
                            contract.patch_count = patchCount;
                        }
                    }
                }
            }
    } else if (path.capture == MGL_TESS_CAPTURE_ARRAY) {
            NSUInteger captureOffset = 0u;
            id capture =
                [self captureAIRVertexPositionsForTessellation:drawCtx
                                                         first:first
                                                         count:count
                                                 instanceCount:instanceCount
                                                  baseInstance:baseInstance
                                                    outOffset:&captureOffset];
            if (!capture) {
                nativeTES = NO;
            } else {
                (void)mglRendererBackendSetTessVertexCaptureBuffer(
                    _backend, (__bridge void *)capture);
                _tessellation.tessVertexCaptureOffset = captureOffset;
                _tessellation.tessInstanceRecords = (NSUInteger)count;
            }
    }

    if (nativeTES && !tcsProgram) {
        id tessVertexCaptureBuffer =
            (__bridge id)
                mglRendererBackendGetTessVertexCaptureBuffer(_backend);
        (void)mglRendererBackendSetTcsOutputBuffer(
            _backend, (__bridge void *)tessVertexCaptureBuffer);
        _tessellation.tcsOutputOffset =
            _tessellation.tessVertexCaptureOffset;
        _tessellation.tcsOutputStride = contract.per_vertex_out_stride;
        _tessellation.tcsOutVertices = patchVertices;
        id tessFactorBuffer = mglCachedDefaultTessFactorBuffer(
            _device, _backend, MGL_STATE(drawCtx), patchCount);
        (void)mglRendererBackendSetCurrentTessFactorBuffer(
            _backend, (__bridge void *)tessFactorBuffer);
        if (!tessVertexCaptureBuffer ||
            !tessFactorBuffer) {
            nativeTES = NO;
        }
    }

    if (path.need_default_factors) {
        /* TES-only compute expansion also needs the default levels; the
         * cached buffer is rebuilt only when glPatchParameterfv levels
         * (or the patch count) change between draws. */
        id tessFactorBuffer = mglCachedDefaultTessFactorBuffer(
            _device, _backend, MGL_STATE(drawCtx), patchCount);
        (void)mglRendererBackendSetCurrentTessFactorBuffer(
            _backend, (__bridge void *)tessFactorBuffer);
    }

    if (path.need_tcs) {
        if (![self dispatchTessControlShader:drawCtx
                                     program:tcsProgram
                                    contract:&contract]) {
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }
    }

    id tcsOutputBuffer = (__bridge id)
        mglRendererBackendGetTcsOutputBuffer(_backend);
    id tessFactorBuffer = (__bridge id)
        mglRendererBackendGetCurrentTessFactorBuffer(_backend);

    if (nativeTES) {
        id nativeFactors = mglNativeTessFactorBuffer(
            _device, tessFactorBuffer,
            tesProgram->tess_gen_mode, patchCount);
        if (!mglTessNativeBuffersReady(
                nativeFactors != nil, tcsOutputBuffer != nil,
                (uint32_t)_tessellation.tcsOutputStride)) {
            NSLog(@"MGL TESS ERROR: invalid native TES buffers program=%u",
                  (unsigned)tesProgram->name);
            mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                             GL_OUT_OF_MEMORY);
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }

        _tessellation.nativeTESProgram = tesProgram;
        _tessellation.nativeTESActive = YES;
        [self clearStageBindingCopyBacks:&_tessellation.nativeTESCopyBacks];
        drawCtx->active_state->dirty_bits = DIRTY_ALL;

        BOOL stateReady = [self processGLState:true];
        if (!stateReady || mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
            _tessellation.nativeTESActive = NO;
            _tessellation.nativeTESProgram = NULL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }

        if (![self currentDrawRasterizationIsEmpty] &&
            ![self currentDrawModeIsFullyCulled:GL_TRIANGLES]) {
            [self applyPolygonOffsetForDrawMode:GL_TRIANGLES];
            /* Metal does not advance the post-tessellation control-point
             * pointer correctly for patchStart. Draw each patch separately:
             * slot 0 is rebased to the patch, while slot 30 stays at the
             * instance base so TES varyings can apply patchId exactly once. */
            id tcsPatchOutBuffer = (__bridge id)
                mglRendererBackendGetTcsPatchOutBuffer(_backend);
            uint32_t patchOutStride = mglTessNativePatchOutStride(
                tcsProgram != NULL,
                tcsPatchOutBuffer && tcsProgram
                    ? mglAIRPatchVaryingStride(
                          &tcsProgram->shader_resources_list
                               [_TESS_CONTROL_SHADER][_STAGE_OUTPUT_RES])
                    : 0u);
            MGLTessNativeEncodeState nativeEncode;
            memset(&nativeEncode, 0, sizeof(nativeEncode));
            nativeEncode.encoder_owner =
                _renderPassManager.state->currentRenderEncoderOwner;
            nativeEncode.tcs_output_buffer = (__bridge void *)tcsOutputBuffer;
            nativeEncode.native_factors = (__bridge void *)nativeFactors;
            nativeEncode.control_point_index_buffer =
                mglRendererBackendGetTessControlPointIndexBuffer(_backend);
            nativeEncode.tcs_patch_out_buffer =
                (__bridge void *)tcsPatchOutBuffer;
            nativeEncode.patch_vertices = patchVertices;
            nativeEncode.patch_count = patchCount;
            nativeEncode.instance_count = (uint32_t)instanceCount;
            nativeEncode.base_instance = baseInstance;
            nativeEncode.tess_gen_mode = (uint32_t)tesProgram->tess_gen_mode;
            nativeEncode.tcs_out_vertices = _tessellation.tcsOutVertices;
            nativeEncode.tcs_output_stride = _tessellation.tcsOutputStride;
            nativeEncode.tess_vertex_capture_offset =
                _tessellation.tessVertexCaptureOffset;
            nativeEncode.tess_instance_records =
                _tessellation.tessInstanceRecords;
            nativeEncode.tess_indexed_draw =
                _tessellation.tessIndexedDraw ? 1u : 0u;
            nativeEncode.patch_out_stride = patchOutStride;
            mglTessEncodeNativePatches(&nativeEncode);
            _currentCBHasWork = YES;

            GLuint64 primitives = mglNativeTessPrimitiveCount(
                tessFactorBuffer, tesProgram, patchCount,
                (GLuint)instanceCount);
            mglRecordActivePrimitiveQueryDraw(drawCtx, primitives, primitives);
        }

        [self endRenderEncoding];
        if (![self flushStageBindingCopyBacks:
                      &_tessellation.nativeTESCopyBacks
                               requireCPUVisibility:NO]) {
            NSLog(@"MGL TESS ERROR: failed to copy isolated native TES "
                  "writable buffer prefixes");
            mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                             GL_OUT_OF_MEMORY);
        }

        _tessellation.nativeTESActive = NO;
        _tessellation.nativeTESProgram = NULL;
        (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
        _tessellation.tessVertexCaptureOffset = 0u;
        (void)mglRendererBackendSetTessControlPointIndexBuffer(_backend, NULL);
        _tessellation.tessIndexedDraw = NO;
        _tessellation.tessInstanceRecords = 0u;
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return YES;
    }

    if (airTES) {
        /* Isolines / point_mode have no Metal-native equivalent; XFB also
         * forces compute (native post-tess cannot feed transform feedback).
         * tess_eval_compute is set at link to match the compiled ABI. */
        if (tesProgram && tesProgram->tess_eval_compute) {
            const BOOL dispatched =
                [self dispatchAIRTessEvalCompute:drawCtx
                                        program:tesProgram
                                       contract:&contract
                                     patchCount:patchCount
                                  instanceCount:instanceCount
                                   baseInstance:baseInstance];
            if (!dispatched) {
                mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                                 GL_INVALID_OPERATION);
            }
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }
        NSLog(@"MGL TESS ERROR: native AIR TES interface unsupported for program %u",
              (unsigned)tesProgram->name);
        /*  contract: an unsupported tessellation draw must surface a GL
         * error, not silently drop the patch stream. */
        mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                         GL_INVALID_OPERATION);
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
        _tessellation.tessVertexCaptureOffset = 0u;
        return YES;
    }

    if (tesProgram) {
        if (![self dispatchTessEvaluationShader:drawCtx
                                           program:tesProgram
                                          contract:&contract]) {
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
    _tessellation.tessVertexCaptureOffset = 0u;
    (void)label;
    return YES;
}

- (Texture *)emulatedMSColor0TextureForContext:(GLMContext)glm_ctx
{
    if (!glm_ctx) return NULL;
    Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
    if (!fbo || (fbo->color_attachment_bitfield & 1u) == 0u) return NULL;
    FBOAttachment *att = &fbo->color_attachments[0];
    Texture *tex = [self framebufferAttachmentTexture:att];
    if (!tex) return NULL;
    if (!mglRenderIsEmulatedMSColorTexture((uint32_t)tex->target,
                                           (int32_t)tex->samples)) {
        return NULL;
    }
    /* Metal backing is created during processGLState; before the first draw
     * mtl_data may still be nil. All MS textures are emulated as array
     * sample planes, so the GL target/samples check is sufficient. */
    if (tex->mtl_data) {
        MGLRenderTextureInfo info = {0};
        (void)mglRenderGetTextureInfo(tex->mtl_data, &info);
        if (info.texture_type != MGLTextureType2DArray) return NULL;
    }
    return tex;
}

- (BOOL)fragmentNeedsPerSampleMSValuesForContext:(GLMContext)glm_ctx
{
    Program *fp = mglResolveProgramForStageFromState(glm_ctx, _FRAGMENT_SHADER);
    if (!fp) return NO;
    Shader *fs = fp->shader_slots[_FRAGMENT_SHADER];
    if (!fs || !fs->src) return NO;
    return mglRenderFragmentNeedsPerSampleMSValues(fs->src) != 0;
}

- (BOOL)runEmulatedMSSampleDrawLoopIfNeeded:(GLMContext)glm_ctx
                                   drawOnce:(void (^)(void))drawOnce
{
    if (_mglInMSSampleDrawLoop || !drawOnce) return NO;
    Texture *tex = [self emulatedMSColor0TextureForContext:glm_ctx];
    if (!tex) return NO;
    if (![self fragmentNeedsPerSampleMSValuesForContext:glm_ctx]) return NO;

    const GLint samples = (GLint)MAX(tex->samples, 1);
    _mglInMSSampleDrawLoop = YES;
    for (GLint s = 0; s < samples; s++) {
        _mglForcedMSSampleId = s;
        _mglMSSamplePlaneOffset = s;
        [self endRenderEncodingLocked];
        mglMarkStateDirtyBits(MGL_STATE(glm_ctx), DIRTY_FBO);
        Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
        if (fbo) {
            fbo->dirty_bits |= DIRTY_FBO_BINDING;
        }
        drawOnce();
    }
    _mglForcedMSSampleId = 0;
    _mglMSSamplePlaneOffset = 0;
    _mglInMSSampleDrawLoop = NO;
    return YES;
}

- (void)broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:(GLMContext)glm_ctx
{
    if (_mglInMSSampleDrawLoop) return;
    Texture *tex = [self emulatedMSColor0TextureForContext:glm_ctx];
    if (!tex || !tex->mtl_data) return;
    if ([self fragmentNeedsPerSampleMSValuesForContext:glm_ctx]) {
        /* Per-sample draws already filled each plane. */
        return;
    }

    const NSUInteger samples = MAX((NSUInteger)tex->samples, 1u);
    if (samples <= 1u) return;

    MGLMetalAttachmentSubresource sub =
        mglMetalAttachmentSubresourceForAttachment(
            &MGL_STATE(glm_ctx)->framebuffer->color_attachments[0]);
    const NSUInteger baseSlice = sub.slice;
    const NSUInteger level = sub.level;
    MGLRenderTextureInfo info = {0};
    (void)mglRenderGetTextureInfo(tex->mtl_data, &info);
    if (info.width == 0u || info.height == 0u) return;
    if (baseSlice + samples > info.array_length) return;

    [self endRenderEncodingLocked];
    if (!_renderPassManager.state->currentCommandBufferOwner &&
        ![self newCommandBufferLocked]) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(
        _renderPassManager.state->currentCommandBufferOwner);
    if (!blit) return;
    for (NSUInteger s = 1u; s < samples; s++) {
        (void)mglRenderBlitCopyTexture(
            blit, tex->mtl_data, baseSlice, level, 0u, 0u, 0u,
            info.width, info.height, 1u,
            tex->mtl_data, baseSlice + s, level, 0u, 0u, 0u);
    }
    (void)mglRenderEndBlitEncoder(blit);
}

static MGLRenderer *mglDrawHostSelf(void *renderer)
{
    return renderer ? (__bridge MGLRenderer *)renderer : nil;
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

bool mglDrawHostHandleTessellation(void *renderer, GLMContext ctx,
                                   GLenum *mode, GLint first, GLsizei count,
                                   GLenum indexType, const void *indices,
                                   GLint baseVertex, GLsizei instanceCount,
                                   GLuint baseInstance, const char *label)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || !mode) {
        return false;
    }
    return [host handleTessellationPatchDrawIfNeeded:ctx
                                                mode:mode
                                               first:first
                                               count:count
                                           indexType:indexType
                                             indices:indices
                                          baseVertex:baseVertex
                                       instanceCount:instanceCount
                                        baseInstance:baseInstance
                                               label:label] ? true : false;
}

bool mglDrawHostHandleGeometry(void *renderer, GLMContext ctx, GLenum mode,
                               GLint first, GLsizei count, GLenum indexType,
                               const void *indices, GLint baseVertex,
                               GLsizei instanceCount, GLuint baseInstance,
                               const char *label)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host handleGeometryDrawIfNeeded:ctx
                                       mode:mode
                                      first:first
                                      count:count
                                  indexType:indexType
                                    indices:indices
                                 baseVertex:baseVertex
                              instanceCount:instanceCount
                               baseInstance:baseInstance
                                      label:label] ? true : false;
}

bool mglDrawHostHandleXFB(void *renderer, GLMContext ctx, GLenum mode,
                          GLint first, GLsizei count, GLsizei instanceCount,
                          GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host handleVertexTransformFeedbackDrawIfNeeded:ctx
                                                      mode:mode
                                                     first:first
                                                     count:count
                                             instanceCount:instanceCount
                                              baseInstance:baseInstance]
               ? true
               : false;
}

bool mglDrawHostCaptureCullDistanceArray(void *renderer, GLMContext ctx,
                                         GLint first, GLsizei count,
                                         GLsizei instanceCount,
                                         GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host captureAIRCullDistancesForArrayDraw:ctx
                                               first:first
                                               count:count
                                       instanceCount:instanceCount
                                        baseInstance:baseInstance]
               ? true
               : false;
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
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host validateDrawArraysVertexInputs:ctx
                                           mode:mode
                                          first:first
                                          count:count
                                       drawCall:0];
}

bool mglDrawHostEncodeCullDistanceArray(void *renderer, GLenum mode,
                                        GLint first, GLsizei count,
                                        GLsizei instanceCount,
                                        GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || mglPolygonModePointForDrawMode(host->ctx, mode)) {
        return false;
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner =
            host->_renderPassManager.state->currentRenderEncoderOwner,
    };
    return [host encodeCullDistanceArrayDraw:mode
                                       first:first
                                       count:count
                               instanceCount:instanceCount
                                baseInstance:baseInstance
                               encodeContext:&encCtx]
               ? true
               : false;
}

void *mglDrawHostEncoderOwner(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host ? host->_renderPassManager.state->currentRenderEncoderOwner
                : NULL;
}

void *mglDrawHostDevice(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host ? mglRendererBackendGetDevice(host->_backend) : NULL;
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
        host->_renderPassManager.state->currentCommandBufferOwner,
        host->_renderPassManager.state->currentRenderEncoderOwner,
        host->_renderPassManager.state->renderPassStateOwner);
}

bool mglDrawHostEncodeCullDistanceElements(void *renderer, GLenum mode,
                                           GLenum type, const void *indices,
                                           GLsizei count, GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
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
    return [host prepareAndEncodeDirectCullDistanceElementDraw:mode
                                                   indexBytes:cullIndexBytes
                                                    indexType:type
                                                        count:count
                                                   baseVertex:baseVertex
                                                instanceCount:instanceCount
                                                 baseInstance:baseInstance
                                              polygonLineMode:
                                                  mglPolygonModeLineForDrawMode(
                                                      host->ctx, mode)]
               ? true
               : false;
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
        host->_renderPassManager.state->currentCommandBufferOwner,
        host->_renderPassManager.state->currentRenderEncoderOwner,
        host->_renderPassManager.state->renderPassStateOwner);
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

@end
