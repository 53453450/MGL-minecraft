// MGLRenderer+DrawSupport.m
// Draw validation, element-buffer resolution and rasterization helper
// methods extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp_objc.h"
#include "mgl_shader_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"

static BOOL mglDrawSupportUsesMetalCpp(void)
{
    return mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
           mglRenderCppGetDevice() != NULL;
}

/* CPU index gather for direct indexed GS draws (mgl_air_gs_abi.h §7).
 * Reads `count` indices of `indexType` from indexBytes, splits the stream
 * at primitive-restart markers (dropping the incomplete primitive before a
 * marker), re-groups vertices into input primitives (dropping the trailing
 * incomplete group), and emits a per-instance uint32 gather array of raw
 * index values (baseVertex NOT applied — Metal's vertex_id for indexed
 * draws is the index value, and stage_in fetch applies baseVertex).
 * Fills *outGather (malloc'd; caller frees) and the count/max outputs.
 * Returns false (with *outGather NULL) if nothing can be gathered. */
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
    if (!indexBytes || count <= 0 || inputVertices == 0u || !outGather ||
        !outGatherCount || !outPrimitiveCount || !outMaxIndex) {
        return false;
    }
    (void)baseVertex; /* gather stores raw index values (vertex_id) */
    const uint32_t elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t *gather = malloc((size_t)count * sizeof(uint32_t));
    if (!gather) return false;
    uint32_t gathered = 0u;
    uint32_t primitives = 0u;
    uint32_t maxIndex = 0u;
    uint32_t inPrimitive = 0u;
    for (GLsizei i = 0; i < count; i++) {
        uint32_t index = 0u;
        if (elemBytes == 1u) {
            index = ((const uint8_t *)indexBytes)[i];
        } else if (elemBytes == 2u) {
            index = ((const uint16_t *)indexBytes)[i];
        } else {
            index = ((const uint32_t *)indexBytes)[i];
        }
        if (restartEnabled && index == restartIndex) {
            /* Primitive restart: drop the partial primitive and start a
             * fresh group. */
            inPrimitive = 0u;
            continue;
        }
        gather[gathered++] = index;
        if (index > maxIndex) maxIndex = index;
        if (++inPrimitive == inputVertices) {
            primitives++;
            inPrimitive = 0u;
        }
    }
    if (gathered == 0u || primitives == 0u) {
        free(gather);
        return false;
    }
    /* Drop the trailing incomplete group. */
    if (inPrimitive != 0u) {
        gathered -= inPrimitive;
    }
    *outGather = gather;
    *outGatherCount = gathered;
    *outPrimitiveCount = primitives;
    *outMaxIndex = maxIndex;
    return true;
}

static id<MTLBuffer> mglDrawSupportCreateBuffer(
    id<MTLDevice> device,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglDrawSupportUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
            buffer) {
            return (__bridge_transfer id<MTLBuffer>)buffer;
        }
    }
    return [device newBufferWithLength:length options:options];
}

static id<MTLBuffer> mglDrawSupportCreateBufferWithBytes(
    id<MTLDevice> device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglDrawSupportUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                              &buffer) == 0 && buffer) {
            return (__bridge_transfer id<MTLBuffer>)buffer;
        }
    }
    return [device newBufferWithBytes:bytes length:length options:options];
}

static void mglDrawSupportSetVertexBuffer(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLBuffer> buffer,
    NSUInteger offset,
    NSUInteger index)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppSetRenderBuffer(
            (__bridge void *)encoder, (__bridge void *)buffer, offset,
            MGL_RENDER_CPP_BINDING_STAGE_VERTEX, (uint32_t)index) == 0) {
        return;
    }
    [encoder setVertexBuffer:buffer offset:offset atIndex:index];
}

static void mglDrawSupportSetVertexBytes(
    id<MTLRenderCommandEncoder> encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppSetRenderBytes(
            (__bridge void *)encoder, bytes, length,
            MGL_RENDER_CPP_BINDING_STAGE_VERTEX, (uint32_t)index) == 0) {
        return;
    }
    [encoder setVertexBytes:bytes length:length atIndex:index];
}

static void mglDrawSupportDrawIndexedPrimitives(
    id<MTLRenderCommandEncoder> encoder,
    MTLPrimitiveType primitiveType,
    NSUInteger indexCount,
    id<MTLBuffer> indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppDrawIndexedPrimitives(
            (__bridge void *)encoder, (uint32_t)primitiveType, indexCount,
            (uint32_t)MTLIndexTypeUInt32, (__bridge void *)indexBuffer,
            indexBufferOffset, instanceCount, baseVertex, baseInstance) == 0) {
        return;
    }
    [encoder drawIndexedPrimitives:primitiveType
                        indexCount:indexCount
                       indexType:MTLIndexTypeUInt32
                       indexBuffer:indexBuffer
                 indexBufferOffset:indexBufferOffset
                     instanceCount:instanceCount
                        baseVertex:baseVertex
                      baseInstance:baseInstance];
}

/* Variant that honors the GL index type (UInt8/UInt16/UInt32).  Used by the
 * GS indexed capture so the original EBO drives vertex fetch directly. */
static void mglDrawSupportDrawIndexedPrimitivesType(
    id<MTLRenderCommandEncoder> encoder,
    MTLPrimitiveType primitiveType,
    NSUInteger indexCount,
    MTLIndexType indexType,
    id<MTLBuffer> indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppDrawIndexedPrimitives(
            (__bridge void *)encoder, (uint32_t)primitiveType, indexCount,
            (uint32_t)indexType, (__bridge void *)indexBuffer,
            indexBufferOffset, instanceCount, baseVertex, baseInstance) == 0) {
        return;
    }
    [encoder drawIndexedPrimitives:primitiveType
                        indexCount:indexCount
                       indexType:indexType
                       indexBuffer:indexBuffer
                 indexBufferOffset:indexBufferOffset
                     instanceCount:instanceCount
                        baseVertex:baseVertex
                      baseInstance:baseInstance];
}

static void mglDrawSupportDrawPrimitives(
    id<MTLRenderCommandEncoder> encoder,
    MTLPrimitiveType primitiveType,
    NSUInteger vertexStart,
    NSUInteger vertexCount,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppDrawPrimitives(
            (__bridge void *)encoder, (uint32_t)primitiveType, vertexStart,
            vertexCount, instanceCount, baseInstance) == 0) {
        return;
    }
    [encoder drawPrimitives:primitiveType
                vertexStart:vertexStart
                vertexCount:vertexCount
              instanceCount:instanceCount
               baseInstance:baseInstance];
}

static void mglDrawSupportDrawPrimitivesIndirect(
    id<MTLRenderCommandEncoder> encoder,
    MTLPrimitiveType primitiveType,
    id<MTLBuffer> indirectBuffer,
    NSUInteger indirectBufferOffset)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppDrawPrimitivesIndirect(
            (__bridge void *)encoder, (uint32_t)primitiveType,
            (__bridge void *)indirectBuffer, indirectBufferOffset) == 0) {
        return;
    }
    [encoder drawPrimitives:primitiveType
             indirectBuffer:indirectBuffer
       indirectBufferOffset:indirectBufferOffset];
}

static id<MTLComputeCommandEncoder> mglDrawSupportCreateComputeEncoder(
    id<MTLCommandBuffer> commandBuffer)
{
    if (mglDrawSupportUsesMetalCpp()) {
        void *encoderCPP = NULL;
        if (mglRenderCppCreateComputeEncoder((__bridge void *)commandBuffer,
                                              &encoderCPP) == 0 &&
            encoderCPP) {
            return (__bridge id<MTLComputeCommandEncoder>)encoderCPP;
        }
    }
    return [commandBuffer computeCommandEncoder];
}

static void mglDrawSupportSetComputePipeline(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppSetComputePipelineState((__bridge void *)encoder,
                                            (__bridge void *)pipeline) == 0) {
        return;
    }
    [encoder setComputePipelineState:pipeline];
}

static void mglDrawSupportSetComputeBuffer(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> buffer,
    NSUInteger offset,
    NSUInteger index)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppSetComputeBuffer((__bridge void *)encoder,
                                     (__bridge void *)buffer, offset,
                                     (uint32_t)index) == 0) {
        return;
    }
    [encoder setBuffer:buffer offset:offset atIndex:index];
}

static void mglDrawSupportSetComputeBytes(
    id<MTLComputeCommandEncoder> encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppSetComputeBytes((__bridge void *)encoder, bytes,
                                    length, (uint32_t)index) == 0) {
        return;
    }
    [encoder setBytes:bytes length:length atIndex:index];
}

static void mglDrawSupportDispatchCompute(
    id<MTLComputeCommandEncoder> encoder,
    MTLSize groups,
    MTLSize threads)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppDispatchCompute(
            (__bridge void *)encoder, (uint32_t)groups.width,
            (uint32_t)groups.height, (uint32_t)groups.depth,
            (uint32_t)threads.width, (uint32_t)threads.height,
            (uint32_t)threads.depth) == 0) {
        return;
    }
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static void mglDrawSupportEndComputeEncoder(
    id<MTLComputeCommandEncoder> encoder)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppEndComputeEncoder((__bridge void *)encoder) == 0) {
        return;
    }
    [encoder endEncoding];
}

static void mglDrawSupportSetTessellationFactors(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLBuffer> buffer,
    NSUInteger offset,
    NSUInteger instanceStride)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppSetTessellationFactorBuffer(
            (__bridge void *)encoder, (__bridge void *)buffer, offset,
            instanceStride) == 0) {
        return;
    }
    [encoder setTessellationFactorBuffer:buffer
                                  offset:offset
                          instanceStride:instanceStride];
}

static void mglDrawSupportDrawPatches(
    id<MTLRenderCommandEncoder> encoder,
    NSUInteger controlPointCount,
    NSUInteger patchStart,
    NSUInteger patchCount,
    id<MTLBuffer> patchIndexBuffer,
    NSUInteger patchIndexBufferOffset,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    if (mglDrawSupportUsesMetalCpp() &&
        mglRenderCppDrawPatches(
            (__bridge void *)encoder, controlPointCount, patchStart,
            patchCount, (__bridge void *)patchIndexBuffer,
            patchIndexBufferOffset, instanceCount, baseInstance) == 0) {
        return;
    }
    [encoder drawPatches:controlPointCount
              patchStart:patchStart
              patchCount:patchCount
        patchIndexBuffer:patchIndexBuffer
  patchIndexBufferOffset:patchIndexBufferOffset
           instanceCount:instanceCount
            baseInstance:baseInstance];
}

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx,
                                               GLuint64 generated,
                                               GLuint64 written);

static BOOL mglCheckedTessCaptureSize(GLsizei count, GLsizei instanceCount,
                                      NSUInteger stride,
                                      NSUInteger *sizeOut,
                                      NSUInteger *offsetOut)
{
    if (count <= 0 || instanceCount <= 0 ||
        stride < MGL_AIR_PER_VERTEX_STRIDE ||
        !sizeOut || !offsetOut) return NO;
    NSUInteger records = (NSUInteger)count * (NSUInteger)instanceCount;
    if (records / (NSUInteger)count != (NSUInteger)instanceCount ||
        records > NSUIntegerMax / stride) return NO;
    *sizeOut = records * stride;
    *offsetOut = 0u;
    return YES;
}

static BOOL mglNativeTESInterfaceSupported(Program *tcsProgram,
                                           Program *tesProgram)
{
    if (!tesProgram ||
        !tesProgram->spirv[_TESS_EVALUATION_SHADER].metallib_bytes ||
        !tesProgram->spirv[_TESS_EVALUATION_SHADER].mtl_function ||
        tesProgram->tess_gen_point_mode ||
        tesProgram->transform_feedback_varying_count > 0) {
        return NO;
    }

    if (tcsProgram &&
        (!tcsProgram->spirv[_TESS_CONTROL_SHADER].metallib_bytes ||
         !tcsProgram->spirv[_TESS_CONTROL_SHADER].mtl_function ||
         tcsProgram->tess_control_output_vertices == 0u ||
         tcsProgram->tess_control_output_vertices > 32u)) {
        return NO;
    }

    if (tesProgram->tess_gen_mode != GL_TRIANGLES &&
        tesProgram->tess_gen_mode != GL_QUADS) {
        return NO;
    }

    id<MTLFunction> function = (__bridge id<MTLFunction>)
        tesProgram->spirv[_TESS_EVALUATION_SHADER].mtl_function;
    MTLPatchType expected = tesProgram->tess_gen_mode == GL_QUADS
        ? MTLPatchTypeQuad : MTLPatchTypeTriangle;
    return function.patchType == expected && function.patchControlPointCount == 0u;
}

static id<MTLBuffer> mglDefaultTessFactorBuffer(id<MTLDevice> device,
                                                GLMState *state,
                                                GLuint patchCount)
{
    if (!device || !state || patchCount == 0u) return nil;
    const NSUInteger stride = 12u;
    if ((NSUInteger)patchCount > NSUIntegerMax / stride) return nil;
    id<MTLBuffer> buffer = mglDrawSupportCreateBuffer(
        device, (NSUInteger)patchCount * stride,
        MTLResourceStorageModeShared);
    if (!buffer || !buffer.contents) return nil;
    __fp16 *dst = (__fp16 *)buffer.contents;
    for (GLuint patch = 0u; patch < patchCount; patch++) {
        for (GLuint i = 0u; i < 4u; i++) {
            dst[patch * 6u + i] =
                (__fp16)state->var.patch_default_outer_level[i];
        }
        for (GLuint i = 0u; i < 2u; i++) {
            dst[patch * 6u + 4u + i] =
                (__fp16)state->var.patch_default_inner_level[i];
        }
    }
    return buffer;
}

static id<MTLBuffer> mglNativeTessFactorBuffer(id<MTLDevice> device,
                                                id<MTLBuffer> canonical,
                                                GLenum mode,
                                                GLuint patchCount)
{
    const NSUInteger canonicalStride = 12u;
    if (!device || !canonical || !canonical.contents || patchCount == 0u ||
        canonical.length < (NSUInteger)patchCount * canonicalStride) {
        return nil;
    }
    if (mode == GL_QUADS) {
        return canonical;
    }
    if (mode != GL_TRIANGLES) {
        return nil;
    }

    const NSUInteger triangleStride = 8u;
    id<MTLBuffer> result = mglDrawSupportCreateBuffer(
        device, (NSUInteger)patchCount * triangleStride,
        MTLResourceStorageModeShared);
    if (!result || !result.contents) {
        return nil;
    }
    const uint16_t *src = (const uint16_t *)canonical.contents;
    uint16_t *dst = (uint16_t *)result.contents;
    for (GLuint patch = 0u; patch < patchCount; patch++) {
        const uint16_t *in = src + patch * 6u;
        uint16_t *out = dst + patch * 4u;
        out[0] = in[0];
        out[1] = in[1];
        out[2] = in[2];
        out[3] = in[4];
    }
    return result;
}

static GLuint64 mglNativeTessPrimitiveCount(id<MTLBuffer> canonical,
                                             Program *tesProgram,
                                             GLuint patchCount,
                                             GLuint instanceCount)
{
    if (!canonical || !canonical.contents || !tesProgram || patchCount == 0u) {
        return 0u;
    }
    const uint16_t *factors = (const uint16_t *)canonical.contents;
    GLuint64 total = 0u;
    for (GLuint patch = 0u; patch < patchCount; patch++) {
        const uint16_t *record = factors + patch * 6u;
        float inside0 = *(const __fp16 *)&record[4];
        float inside1 = *(const __fp16 *)&record[5];
        inside0 = MAX(inside0, 1.0f);
        inside1 = MAX(inside1, 1.0f);
        GLuint64 perPatch = tesProgram->tess_gen_mode == GL_QUADS
            ? 2ull * (GLuint64)ceilf(inside0) * (GLuint64)ceilf(inside1)
            : (GLuint64)ceilf(inside0) * (GLuint64)ceilf(inside0);
        total += MAX(perPatch, 1ull);
    }
    return total * (GLuint64)instanceCount;
}

@implementation MGLRenderer (Draw)

- (BOOL)captureAIRCullDistancesForArrayDraw:(GLMContext)drawCtx
                                      first:(GLint)first
                                      count:(GLsizei)count
                              instanceCount:(GLsizei)instanceCount
                               baseInstance:(GLuint)baseInstance
{
    _tessellation.cullDistanceCaptureBuffer = nil;
    _tessellation.cullDistanceCaptureFirstInstance = 0u;
    _tessellation.cullDistanceCaptureInstanceStride = 0u;
    if (!drawCtx || first < 0 || count <= 0 || instanceCount <= 0) return NO;

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (!vertexProgram || !vertexProgram->uses_cull_distance ||
        ![self bindMTLProgram:vertexProgram] ||
        !vertexProgram->spirv[_VERTEX_SHADER].mtl_cull_capture_function) {
        return NO;
    }
    const uint64_t endVertex = (uint64_t)(uint32_t)first +
                               (uint64_t)(uint32_t)count;
    const uint64_t lastCaptureIndex =
        (uint64_t)((uint32_t)instanceCount - 1u) * (uint64_t)(uint32_t)count +
        endVertex;
    if (endVertex == 0u || lastCaptureIndex == 0u ||
        lastCaptureIndex > NSUIntegerMax / 32u) return NO;
    id<MTLBuffer> capture = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)(lastCaptureIndex * 32u),
        MTLResourceStorageModeShared);
    if (!capture) return NO;

    self->ctx = drawCtx;
    _tessellation.cullDistanceCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        !_renderPassManager.state->currentRenderEncoder) {
        _tessellation.cullDistanceCaptureActive = NO;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return NO;
    }
    id<MTLRenderCommandEncoder> encoder =
        _renderPassManager.state->currentRenderEncoder;
    MGLCullDistanceEmuParams params = {
        .prim_vertex_count = 1u,
        .culldist_offset = 0u,
        .vertex_stride = 32u,
        .culldist_size = MIN(vertexProgram->cull_distance_count, 8u),
        .first_vertex = (uint32_t)first,
        .first_instance = baseInstance,
        .instance_stride = (uint32_t)count,
    };
    mglDrawSupportSetVertexBuffer(encoder, capture, 0u, 29u);
    mglDrawSupportSetVertexBytes(
        encoder, &params, sizeof(params), kMGLCullDistanceParamsBufferIndex);
    mglDrawSupportDrawPrimitives(encoder, MTLPrimitiveTypePoint,
                                 (NSUInteger)first, (NSUInteger)count,
                                 (NSUInteger)instanceCount,
                                 (NSUInteger)baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.cullDistanceCaptureActive = NO;
    _tessellation.cullDistanceCaptureBuffer = capture;
    _tessellation.cullDistanceCaptureFirstInstance = baseInstance;
    _tessellation.cullDistanceCaptureInstanceStride = (uint32_t)count;
    drawCtx->state.dirty_bits = DIRTY_ALL;
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
    uint32_t minIndex = 0u;
    uint32_t maxIndex = 0u;
    if (!mglScanIndexRangeIgnoringRestart(
            indexBytes, indexType, count, restartEnabled, restartIndex,
            &minIndex, &maxIndex)) {
        return NO;
    }
    const int64_t first = (int64_t)minIndex + (int64_t)baseVertex;
    const int64_t last = (int64_t)maxIndex + (int64_t)baseVertex;
    if (first < 0 || last < first || last > INT32_MAX) return NO;
    const uint64_t vertexCount = (uint64_t)(last - first) + 1u;
    if (vertexCount > INT32_MAX) return NO;
    return [self captureAIRCullDistancesForArrayDraw:drawCtx
                                               first:(GLint)first
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

    if (activeProgram->spirv[_VERTEX_SHADER].mtl_cull_capture_function) {
        if (![self captureAIRCullDistancesForElementDraw:ctx
                                             indexBytes:indexBytes
                                              indexType:indexType
                                                  count:count
                                             baseVertex:baseVertex
                                          instanceCount:instanceCount
                                           baseInstance:baseInstance] ||
            ![self processGLState:true] ||
            !_renderPassManager.state->currentRenderEncoder) {
            return YES;
        }
    }

    MGLEncodeContext encCtx = {
        .encoder = _renderPassManager.state->currentRenderEncoder,
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
        !encCtx || !encCtx->encoder) {
        return NO;
    }

    if (mode == GL_TRIANGLE_STRIP && count >= 3) {
        NSUInteger indexCount = 0u;
        id<MTLBuffer> indexBuffer = mglNewTriangleStripArrayIndexBuffer(
            _device, (NSUInteger)count, &indexCount);
        if (!indexBuffer || indexCount == 0u) return YES;
        for (NSUInteger primitive = 0u; primitive * 3u < indexCount;
             primitive++) {
            const GLuint vertices[3] = {
                (GLuint)first + (GLuint)primitive,
                (GLuint)first + (GLuint)primitive + 1u,
                (GLuint)first + (GLuint)primitive + 2u,
            };
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)first
                                   explicitVertices:vertices
                                 explicitVertexCount:3u
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->encoder, MTLPrimitiveTypeTriangle, 3u, indexBuffer,
                primitive * 3u * sizeof(uint32_t),
                (NSUInteger)instanceCount, (NSInteger)first,
                (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_TRIANGLE_FAN && count >= 3) {
        NSUInteger indexCount = 0u;
        id<MTLBuffer> indexBuffer = mglNewTriangleFanArrayIndexBuffer(
            _device, (NSUInteger)count, &indexCount);
        if (!indexBuffer || indexCount == 0u) return YES;
        for (NSUInteger primitive = 0u; primitive * 3u < indexCount;
             primitive++) {
            const GLuint vertices[3] = {
                (GLuint)first,
                (GLuint)first + (GLuint)primitive + 1u,
                (GLuint)first + (GLuint)primitive + 2u,
            };
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)first
                                   explicitVertices:vertices
                                 explicitVertexCount:3u
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->encoder, MTLPrimitiveTypeTriangle, 3u, indexBuffer,
                primitive * 3u * sizeof(uint32_t),
                (NSUInteger)instanceCount, (NSInteger)first,
                (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_LINE_STRIP && count >= 2) {
        for (GLsizei primitive = 0; primitive + 1 < count; primitive++) {
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)(first + primitive)
                                   explicitVertices:NULL
                                 explicitVertexCount:0u
                                      encodeContext:encCtx];
            mglDrawSupportDrawPrimitives(
                encCtx->encoder, MTLPrimitiveTypeLine,
                (NSUInteger)(first + primitive), 2u,
                (NSUInteger)instanceCount, (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_LINE_LOOP && count >= 2) {
        NSUInteger indexCount = 0u;
        id<MTLBuffer> indexBuffer = mglNewLineLoopArrayIndexBuffer(
            _device, (NSUInteger)first, (NSUInteger)count, &indexCount);
        if (!indexBuffer || indexCount == 0u) return YES;
        for (NSUInteger primitive = 0u; primitive + 1u < indexCount;
             primitive++) {
            const GLuint vertices[2] = {
                (GLuint)first + (GLuint)primitive,
                (GLuint)first +
                    (GLuint)((primitive + 1u) % (NSUInteger)count),
            };
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)first
                                   explicitVertices:vertices
                                 explicitVertexCount:2u
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->encoder, MTLPrimitiveTypeLine, 2u, indexBuffer,
                primitive * sizeof(uint32_t), (NSUInteger)instanceCount, 0,
                (NSUInteger)baseInstance);
        }
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
    if (activeProgram->spirv[_VERTEX_SHADER].mtl_cull_capture_function &&
        !_tessellation.cullDistanceCaptureBuffer) {
        return YES;
    }
    if (!indexBytes || count <= 0 || instanceCount <= 0 ||
        !encCtx || !encCtx->encoder) {
        return YES;
    }

    uint32_t restartIndex = 0u;
    const bool restartEnabled =
        mglPrimitiveRestartIndexForType(ctx, indexType, &restartIndex);
    void *planOwner = NULL;
    void *indexBufferHandle = NULL;
    uint64_t primitiveCount = 0u;
    if (mglRenderCppCreateCullDistanceIndexPlan(
            (__bridge void *)_device, indexBytes, indexType,
            (uint64_t)count, mode,
            restartEnabled ? 1 : 0, restartIndex, baseVertex,
            polygonLineMode ? 1 : 0, &planOwner, &indexBufferHandle,
            &primitiveCount) != 0 || !planOwner) {
        return YES;
    }

    id<MTLBuffer> indexBuffer =
        (__bridge id<MTLBuffer>)indexBufferHandle;
    @try {
        for (uint64_t primitiveIndex = 0u;
             primitiveIndex < primitiveCount; ++primitiveIndex) {
            MGLRenderCppCullDistancePrimitive primitive = {0};
            if (mglRenderCppGetCullDistanceIndexPrimitive(
                    planOwner, primitiveIndex, &primitive) != 0) {
                break;
            }
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:0u
                                   explicitVertices:primitive.vertices
                                 explicitVertexCount:primitive.vertex_count
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->encoder,
                (MTLPrimitiveType)primitive.primitive_type,
                (NSUInteger)primitive.index_count,
                indexBuffer,
                (NSUInteger)primitive.index_buffer_offset,
                (NSUInteger)instanceCount,
                0,
                (NSUInteger)baseInstance);
        }
    } @finally {
        mglRenderCppDestroyCullDistanceIndexPlan(&planOwner);
    }
    return YES;
}

- (id<MTLBuffer>)captureAIRVertexPositionsForTessellation:(GLMContext)drawCtx
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
        !vertexProgram->spirv[_VERTEX_SHADER].mtl_tess_capture_function) {
        return nil;
    }

    NSUInteger captureSize = 0u;
    NSUInteger captureOffset = 0u;
    NSUInteger captureStride = mglAIRPerVertexStrideForResources(
        &vertexProgram->spirv_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    if (!mglCheckedTessCaptureSize(count, instanceCount, captureStride, &captureSize,
                                   &captureOffset)) {
        return nil;
    }
    id<MTLBuffer> capture = mglDrawSupportCreateBuffer(
        _device, captureSize, MTLResourceStorageModeShared);
    if (!capture) return nil;

    self->ctx = drawCtx;
    _tessellation.tessVertexCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        !_renderPassManager.state->currentRenderEncoder) {
        _tessellation.tessVertexCaptureActive = NO;
        return nil;
    }

    id<MTLRenderCommandEncoder> encoder =
        _renderPassManager.state->currentRenderEncoder;
    mglDrawSupportSetVertexBuffer(encoder, capture, 0u, 29u);
    const uint32_t captureParams[3] = {
        (uint32_t)first, (uint32_t)count, baseInstance,
    };
    mglDrawSupportSetVertexBytes(
        encoder, captureParams, sizeof(captureParams), 28u);
    mglDrawSupportDrawPrimitives(encoder, MTLPrimitiveTypePoint,
                                 (NSUInteger)first, (NSUInteger)count,
                                 (NSUInteger)instanceCount,
                                 (NSUInteger)baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.tessVertexCaptureActive = NO;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (outOffset) *outOffset = captureOffset;
    return capture;
}

/* Indexed variant of the VS capture for direct indexed GS draws
 * (mgl_air_gs_abi.h §7).  Runs drawIndexedPrimitives against the original
 * EBO so Metal's baseVertex is applied to stage_in fetch; the capture
 * kernel's vertex_id is the raw index value, so records are sparse
 * ([instance][vertex_id], span = maxIndex+1 per instance). */
- (id<MTLBuffer>)captureAIRVertexPositionsForGeometryIndexed:(GLMContext)drawCtx
                                                  indexBuffer:(id<MTLBuffer>)indexBuffer
                                                    indexType:(MTLIndexType)indexType
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
        !vertexProgram->spirv[_VERTEX_SHADER].mtl_tess_capture_function) {
        return nil;
    }

    NSUInteger captureSize = 0u;
    NSUInteger captureOffset = 0u;
    NSUInteger captureStride = mglAIRPerVertexStrideForResources(
        &vertexProgram->spirv_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    const NSUInteger recordsPerInstance = (NSUInteger)maxIndex + 1u;
    if (!mglCheckedTessCaptureSize(recordsPerInstance, instanceCount,
                                   captureStride, &captureSize,
                                   &captureOffset)) {
        return nil;
    }
    id<MTLBuffer> capture = mglDrawSupportCreateBuffer(
        _device, captureSize, MTLResourceStorageModeShared);
    if (!capture) return nil;
    self->ctx = drawCtx;
    _tessellation.tessVertexCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        !_renderPassManager.state->currentRenderEncoder) {
        _tessellation.tessVertexCaptureActive = NO;
        return nil;
    }

    id<MTLRenderCommandEncoder> encoder =
        _renderPassManager.state->currentRenderEncoder;
    mglDrawSupportSetVertexBuffer(encoder, capture, 0u, 29u);
    const uint32_t captureParams[3] = {
        0u, (uint32_t)recordsPerInstance, baseInstance,
    };
    mglDrawSupportSetVertexBytes(
        encoder, captureParams, sizeof(captureParams), 28u);
    mglDrawSupportDrawIndexedPrimitivesType(
        encoder, MTLPrimitiveTypePoint, (NSUInteger)count, indexType,
        indexBuffer, indexOffset, (NSUInteger)instanceCount,
        (NSInteger)baseVertex, (NSUInteger)baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.tessVertexCaptureActive = NO;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (outOffset) *outOffset = captureOffset;
    return capture;
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
    /* Keep the old narrow passthrough optimization.  It does not need a
     * compute expansion and remains a normal VS->FS draw. */
    const char *geometrySource = geometryShader->src;
    if (geometrySource && strstr(geometrySource, "EmitVertex()") &&
        strstr(geometrySource, "EndPrimitive()") &&
        strstr(geometrySource, "gl_Position = gl_in[n_vertex_index].gl_Position") &&
        !strstr(geometrySource, "gl_PrimitiveID") &&
        !strstr(geometrySource, "gl_Layer") &&
        !strstr(geometrySource, "gl_ViewportIndex")) {
        return NO;
    }
    GLenum gsInputMode = program->geometry_input_type;
    if (gsInputMode != GL_POINTS && gsInputMode != GL_LINES &&
        gsInputMode != GL_LINES_ADJACENCY &&
        gsInputMode != GL_TRIANGLES &&
        gsInputMode != GL_TRIANGLES_ADJACENCY) {
        gsInputMode = GL_TRIANGLES;
    }
    GLenum gsOutputMode = program->geometry_output_type;
    if (gsOutputMode != GL_POINTS && gsOutputMode != GL_LINE_STRIP &&
        gsOutputMode != GL_TRIANGLE_STRIP) {
        gsOutputMode = GL_TRIANGLE_STRIP;
    }
    GLuint inputVertices = gsInputMode == GL_POINTS ? 1u
        : gsInputMode == GL_LINES ? 2u
        : gsInputMode == GL_LINES_ADJACENCY ? 4u
        : (gsInputMode == GL_TRIANGLES_ADJACENCY) ? 6u : 3u;
    MTLPrimitiveType outputPrimitive = gsOutputMode == GL_POINTS
        ? MTLPrimitiveTypePoint
        : gsOutputMode == GL_LINE_STRIP ? MTLPrimitiveTypeLine
        : MTLPrimitiveTypeTriangle;
    const BOOL indexedDraw = (indexType != 0u);
    if (mode != gsInputMode || count <= 0 || instanceCount <= 0 ||
        (!indexedDraw && (first < 0 ||
                          (count % (GLsizei)inputVertices) != 0))) {
        static uint64_t unsupportedDrawCount = 0;
        uint64_t hit = ++unsupportedDrawCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL GS ERROR: blocking unsupported %s draw %@ "
                   "mode=0x%x count=%d instances=%d baseInstance=%u",
                  indexedDraw ? "indexed" : "array",
                  label ? [NSString stringWithUTF8String:label] : @"draw",
                  (unsigned)mode, (int)count, (int)instanceCount,
                  (unsigned)baseInstance);
        }
        /* P0 contract: never drop a GS draw silently.  A draw whose mode
         * does not match the GS input topology is an invalid operation. */
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        return YES;
    }
    if (program->gs_route != MGL_GS_ROUTE_COMPUTE ||
        !program->spirv[_GEOMETRY_SHADER].metallib_bytes ||
        program->geometry_vertices_out == 0u ||
        program->geometry_vertices_out > 1024u) {
        static uint64_t unsupportedCount = 0;
        uint64_t hit = ++unsupportedCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL GS ERROR: blocking %@; AIR compute route unavailable program=%u",
                  label ? [NSString stringWithUTF8String:label] : @"draw",
                  (unsigned)program->name);
        }
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        return YES;
    }
    if (![self bindMTLProgram:program] ||
        !program->spirv[_GEOMETRY_SHADER].mtl_function) {
        NSLog(@"MGL GS ERROR: failed to load AIR kernel program=%u",
              (unsigned)program->name);
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        return YES;
    }

    self->ctx = drawCtx;
    if (![self ensureAIRGeometryPassthroughFunctionForProgram:program]) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }

    /* Indexed draws first gather the element stream (mgl_air_gs_abi.h §7):
     * restart markers split the stream, vertices re-group into input
     * primitives, and the resulting per-instance uint32 gather array carries
     * raw index values so the GS kernel can locate sparse capture records. */
    uint32_t *gatherArray = NULL;
    uint32_t gatherCount = 0u;
    uint32_t gatherPrimitives = 0u;
    uint32_t gatherMaxIndex = 0u;
    const uint8_t *indexBytes = NULL;
    id<MTLBuffer> eboMetal = nil;
    NSUInteger indexOffsetBytes = 0u;
    MTLIndexType captureIndexType = MTLIndexTypeUInt32;
    id<MTLBuffer> gatherBuf = nil;
    MGLAIRGSGatherParams gparams;
    memset(&gparams, 0, sizeof(gparams));
    if (indexedDraw) {
        Buffer *ebo = getElementBuffer(drawCtx);
        if (!ebo || ![self processBuffer:ebo] || !ebo->data.mtl_data) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_INVALID_OPERATION);
            return YES;
        }
        eboMetal = (__bridge id<MTLBuffer>)ebo->data.mtl_data;
        indexOffsetBytes = (NSUInteger)(uintptr_t)indices;
        indexBytes = mglElementIndexSourceForDraw(ebo, eboMetal, indexType,
                                                  indexOffsetBytes, count);
        if (!indexBytes) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_INVALID_OPERATION);
            return YES;
        }
        uint32_t restartIndex = 0u;
        bool restartEnabled = false;
        restartEnabled = mglPrimitiveRestartIndexForType(
            drawCtx, indexType, &restartIndex);
        if (!mglGeometryGatherIndices(indexBytes, indexType, count,
                                      baseVertex, restartEnabled, restartIndex,
                                      inputVertices, &gatherArray,
                                      &gatherCount, &gatherPrimitives,
                                      &gatherMaxIndex)) {
            /* Nothing drawable after gather/restart handling — a valid
             * empty draw, not an error. */
            return YES;
        }
        captureIndexType = getMTLIndexType(indexType);
        gatherBuf = mglDrawSupportCreateBufferWithBytes(
            _device, gatherArray, (NSUInteger)gatherCount * 4u,
            MTLResourceStorageModeShared);
        free(gatherArray);
        gatherArray = NULL;
        if (!gatherBuf) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_OUT_OF_MEMORY);
            return YES;
        }
        gparams.vertices_per_instance = gatherMaxIndex + 1u;
        gparams.primitives_per_instance = gatherPrimitives;
        gparams.first_vertex = 0u;
        gparams.gather_enabled = 1u;
    }

    const GLuint primitiveCount = indexedDraw
        ? gatherPrimitives : (GLuint)count / inputVertices;
    if ((GLuint)instanceCount > UINT32_MAX / primitiveCount) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const GLuint drawPrimitiveCount =
        primitiveCount * (GLuint)instanceCount;
    const GLuint invocationCount = MAX(1u, program->geometry_invocations);
    if (drawPrimitiveCount > UINT32_MAX / invocationCount) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const GLuint workItemCount = drawPrimitiveCount * invocationCount;
    const NSUInteger outputStride = mglAIRPerVertexStrideForResources(
        &program->spirv_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES]);
    const uint32_t maxVertices = program->geometry_vertices_out;
    /* Fixed ABI layout (mgl_air_gs_abi.h §2): 2 header records + the
     * expanded primitive vertices per work item. */
    const MGLAIRGSOutputPrimitive gsAirOutput = gsOutputMode == GL_POINTS
        ? MGL_AIR_GS_OUT_POINTS
        : gsOutputMode == GL_LINE_STRIP ? MGL_AIR_GS_OUT_LINE_STRIP
        : MGL_AIR_GS_OUT_TRIANGLE_STRIP;
    const NSUInteger expandedVertices =
        mglAIRGSExpandedVertices(gsAirOutput, maxVertices);
    const NSUInteger recordsPerPrimitive =
        mglAIRGSRecordsPerPrimitive(gsAirOutput, maxVertices);
    if (primitiveCount == 0u ||
        recordsPerPrimitive >
            (NSUIntegerMax / outputStride) / workItemCount) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }

    /* Run the real VS once into the shared per-vertex records used by the AIR GS
     * kernel.  This helper closes the render encoder before compute begins. */
    NSUInteger inputOffset = 0u;
    id<MTLBuffer> input = nil;
    if (indexedDraw) {
        input = [self captureAIRVertexPositionsForGeometryIndexed:drawCtx
                                                      indexBuffer:eboMetal
                                                        indexType:captureIndexType
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
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    void *pipelineHandle = NULL;
    char pipelineError[512] = {0};
    int pipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _GEOMETRY_SHADER, &pipelineHandle,
        pipelineError, sizeof(pipelineError));
    id<MTLComputePipelineState> pipeline =
        pipelineResult == 0 && pipelineHandle
            ? (__bridge_transfer id<MTLComputePipelineState>)pipelineHandle
            : nil;
    if (!pipeline) {
        NSLog(@"MGL GS ERROR: compute PSO failed program=%u: %s",
              (unsigned)program->name,
              pipelineError[0] ? pipelineError : "unknown error");
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    if (!_renderPassManager.state->currentCommandBuffer ||
        mglRenderCommandBufferStatus(
            _renderPassManager.state->currentCommandBuffer) >=
            MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }
    const NSUInteger outputSize =
        (NSUInteger)workItemCount * recordsPerPrimitive * outputStride;
    id<MTLBuffer> output = mglDrawSupportCreateBuffer(
        _device, outputSize, MTLResourceStorageModeShared);
    /* ABI (mgl_air_gs_abi.h §3): one 28-byte counts record per work item —
     * 16-byte indirect args + 12 bytes kernel scratch. */
    const NSUInteger countsRecordBytes = MGL_AIR_GS_COUNTS_RECORD_BYTES;
    id<MTLBuffer> counts = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)workItemCount * countsRecordBytes,
        MTLResourceStorageModeShared);
    if (!output || !counts || !output.contents || !counts.contents) {
        drawCtx->state.dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    memset(output.contents, 0, outputSize);
    memset(counts.contents, 0, (NSUInteger)workItemCount * countsRecordBytes);
    /* Preset the draw parameters the kernel never touches: instance_count=1,
     * base_vertex=0, base_instance=0 (memset already zeroed the rest). */
    {
        uint32_t *countsWords = (uint32_t *)counts.contents;
        for (NSUInteger w = 0; w < workItemCount; w++) {
            countsWords[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 1] = 1u;
        }
    }

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *image = MGL_STATE(drawCtx)->image_units[unit].tex;
        Texture *sampled = MGL_STATE(drawCtx)->active_textures[unit];
        if (image && ![self bindMTLTexture:image]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
        if (sampled && ![self bindMTLTexture:sampled]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    id<MTLComputeCommandEncoder> compute =
        mglDrawSupportCreateComputeEncoder(
            _renderPassManager.state->currentCommandBuffer);
    if (!compute) {
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    mglDrawSupportSetComputePipeline(compute, pipeline);
    mglDrawSupportSetComputeBuffer(compute, input, inputOffset,
                                   MGL_AIR_GS_SLOT_INPUT);
    mglDrawSupportSetComputeBuffer(compute, output, 0u,
                                   MGL_AIR_GS_SLOT_OUTPUT);
    mglDrawSupportSetComputeBuffer(compute, counts, 0u,
                                   MGL_AIR_GS_SLOT_COUNTS);
    /* Gather ABI (mgl_air_gs_abi.h §7): the kernel always receives a
     * gather slot + params; array draws bind gather_enabled=0 and a dummy
     * gather buffer (counts works — the kernel never reads it). */
    mglDrawSupportSetComputeBuffer(compute,
                                   indexedDraw ? gatherBuf : counts, 0u,
                                   MGL_AIR_GS_SLOT_GATHER);
    mglDrawSupportSetComputeBytes(compute, &gparams, sizeof(gparams),
                                  MGL_AIR_GS_SLOT_GATHER_PARAMS);
    bool buffersOK = [self bindBuffersToComputeEncoder:compute
                                                   stage:_GEOMETRY_SHADER
                                               copyBacks:&stageCopyBacks];
    bool texturesOK = buffersOK && [self bindTexturesToComputeEncoder:compute
                                                                stage:_GEOMETRY_SHADER];
    if (!buffersOK || !texturesOK) {
        mglDrawSupportEndComputeEncoder(compute);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    mglDrawSupportDispatchCompute(
        compute, MTLSizeMake(workItemCount, 1u, 1u),
        MTLSizeMake(1u, 1u, 1u));
    mglDrawSupportEndComputeEncoder(compute);
    if (![self flushStageBindingCopyBacks:&stageCopyBacks
                     requireCPUVisibility:NO]) {
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    _geometry.expansionActive = YES;
    _geometry.program = program;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        !_renderPassManager.state->currentRenderEncoder ||
        [self currentDrawRasterizationIsEmpty] ||
        [self currentDrawModeIsFullyCulled:gsOutputMode]) {
        _geometry.expansionActive = NO;
        _geometry.program = NULL;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    [self applyPolygonOffsetForDrawMode:gsOutputMode];
    id<MTLRenderCommandEncoder> encoder =
        _renderPassManager.state->currentRenderEncoder;
    for (GLuint primitive = 0u; primitive < workItemCount; primitive++) {
        NSUInteger offset =
            ((NSUInteger)primitive * recordsPerPrimitive +
             MGL_AIR_GS_HEADER_RECORDS) * outputStride;
        mglDrawSupportSetVertexBuffer(encoder, output, offset, 0u);
        mglDrawSupportDrawPrimitivesIndirect(
            encoder, outputPrimitive, counts,
            (NSUInteger)primitive * countsRecordBytes);
    }
    _currentCBHasWork = YES;
    mglRecordActivePrimitiveQueryDraw(
        drawCtx, outputPrimitive == MTLPrimitiveTypePoint
            ? (GLuint64)workItemCount * expandedVertices
            : (GLuint64)workItemCount * expandedVertices /
                  (outputPrimitive == MTLPrimitiveTypeLine ? 2u : 3u),
        outputPrimitive == MTLPrimitiveTypePoint
            ? (GLuint64)workItemCount * expandedVertices
            : (GLuint64)workItemCount * expandedVertices /
                  (outputPrimitive == MTLPrimitiveTypeLine ? 2u : 3u));
    _geometry.expansionActive = NO;
    _geometry.program = NULL;
    drawCtx->state.dirty_bits = DIRTY_ALL;
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

        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
        if (!mtlBuffer) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=mtl_bridge_nil",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name);
            return false;
        }

        uint64_t metalLen = (uint64_t)mtlBuffer.length;
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
                          mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
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
                              mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
{
    Buffer *gl_element_buffer = NULL;
    if (cmd && cmd->elementBuffer) {
        gl_element_buffer = mglRendererGetValidatedBuffer(drawCtx,
                                                          (Buffer *)cmd->elementBuffer,
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
                   mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
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

    id<MTLBuffer> indexBuffer = (__bridge id<MTLBuffer>)(gl_element_buffer->data.mtl_data);
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
                           mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
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

    id<MTLBuffer> indirectBuffer = (__bridge id<MTLBuffer>)(gl_indirect_buffer->data.mtl_data);
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
    if (_renderPassManager.state->currentRenderEncoder) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (!_renderPassManager.state->currentRenderEncoder) {
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
    if (vw <= 0 || vh <= 0) {
        return YES;
    }

    NSUInteger passWidth = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.renderTargetWidth : 0;
    NSUInteger passHeight = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.renderTargetHeight : 0;
    if ((passWidth == 0 || passHeight == 0) && _renderPassManager.state->renderPassDescriptor) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id<MTLTexture> color = _renderPassManager.state->renderPassDescriptor.colorAttachments[i].texture;
            if (color) {
                passWidth = color.width;
                passHeight = color.height;
                break;
            }
        }
        if ((passWidth == 0 || passHeight == 0) && _renderPassManager.state->renderPassDescriptor.depthAttachment.texture) {
            passWidth = _renderPassManager.state->renderPassDescriptor.depthAttachment.texture.width;
            passHeight = _renderPassManager.state->renderPassDescriptor.depthAttachment.texture.height;
        }
        if ((passWidth == 0 || passHeight == 0) && _renderPassManager.state->renderPassDescriptor.stencilAttachment.texture) {
            passWidth = _renderPassManager.state->renderPassDescriptor.stencilAttachment.texture.width;
            passHeight = _renderPassManager.state->renderPassDescriptor.stencilAttachment.texture.height;
        }
    }

    if (passWidth == 0 || passHeight == 0) {
        return NO;
    }

    int64_t fbW = (int64_t)passWidth;
    int64_t fbH = (int64_t)passHeight;
    int64_t vx0 = (int64_t)vx;
    int64_t vy0 = (int64_t)vy;
    int64_t vx1 = vx0 + (int64_t)vw;
    int64_t vy1 = vy0 + (int64_t)vh;
    if (vx1 <= 0 || vy1 <= 0 || vx0 >= fbW || vy0 >= fbH) {
        return YES;
    }

    if (MGL_STATE(ctx)->caps.scissor_test) {
        GLint sx = MGL_STATE(ctx)->var.scissor_box[0];
        GLint sy = MGL_STATE(ctx)->var.scissor_box[1];
        GLint sw = MGL_STATE(ctx)->var.scissor_box[2];
        GLint sh = MGL_STATE(ctx)->var.scissor_box[3];
        if (sw <= 0 || sh <= 0) {
            return YES;
        }

        int64_t sx0 = (int64_t)sx;
        int64_t sy0 = (int64_t)sy;
        int64_t sx1 = sx0 + (int64_t)sw;
        int64_t sy1 = sy0 + (int64_t)sh;
        if (sx1 <= 0 || sy1 <= 0 || sx0 >= fbW || sy0 >= fbH) {
            return YES;
        }
    }

    return NO;
}

- (void)applyPolygonOffsetForDrawMode:(GLenum)mode
{
    if (!_renderPassManager.state->currentRenderEncoder) {
        return;
    }

    MTLTriangleFillMode triangleFillMode = MTLTriangleFillModeFill;
    if (ctx && mglDrawModeProducesPolygons(mode)) {
        if (MGL_STATE(ctx)->var.polygon_mode == GL_LINE) {
            triangleFillMode = MTLTriangleFillModeLines;
        } else if (MGL_STATE(ctx)->var.polygon_mode != GL_FILL &&
                   MGL_STATE(ctx)->var.polygon_mode != GL_POINT) {
            mglLogRenderStateRepair("polygon_mode", MGL_STATE(ctx)->var.polygon_mode, GL_FILL);
            MGL_STATE(ctx)->var.polygon_mode = GL_FILL;
            mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
        }
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];

    BOOL enableDepthBias = NO;

    if (ctx && mglDrawModeProducesPolygons(mode)) {
        switch (MGL_STATE(ctx)->var.polygon_mode) {
            case GL_POINT:
                enableDepthBias = MGL_STATE(ctx)->caps.polygon_offset_point;
                break;
            case GL_LINE:
                enableDepthBias = MGL_STATE(ctx)->caps.polygon_offset_line;
                break;
            case GL_FILL:
            default:
                enableDepthBias = MGL_STATE(ctx)->caps.polygon_offset_fill;
                break;
        }
    }

    if (enableDepthBias) {
        float _bias = MGL_STATE(ctx)->var.polygon_offset_units;
        float _slope = MGL_STATE(ctx)->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderCppBindingSetDepthBiasIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            _bias, _clamp, _slope);
    } else {
        mglRenderCppBindingSetDepthBiasIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
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

- (void)bindCullDistanceEmulationBuffers:(GLenum)mode
                             firstVertex:(GLuint)firstVertex
                        explicitVertices:(const GLuint *)explicitVertices
                      explicitVertexCount:(GLuint)explicitVertexCount
                           encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!ctx || !encCtx->encoder) {
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

    /* Determine primitive vertex count from the draw mode. */
    uint32_t prim_vertex_count = 0;
    switch (mode) {
        case GL_TRIANGLES: prim_vertex_count = 3; break;
        case GL_TRIANGLE_STRIP: prim_vertex_count = 3; break;
        case GL_TRIANGLE_FAN: prim_vertex_count = 3; break;
        case GL_LINES: prim_vertex_count = 2; break;
        case GL_LINE_STRIP: prim_vertex_count = 2; break;
        case GL_LINE_LOOP: prim_vertex_count = 2; break;
        case GL_POINTS: prim_vertex_count = 1; break;
        case GL_QUADS: prim_vertex_count = 4; break;
        default: prim_vertex_count = 1; break;
    }

    if (_tessellation.cullDistanceCaptureBuffer) {
        MGLCullDistanceEmuParams params = {
            .prim_vertex_count = prim_vertex_count,
            .culldist_offset = 0u,
            .vertex_stride = 32u,
            .culldist_size = MIN(activeProgram->cull_distance_count, 8u),
            .first_vertex = firstVertex,
            .explicit_vertex_count = explicitVertexCount,
            .first_instance =
                _tessellation.cullDistanceCaptureFirstInstance,
            .instance_stride =
                _tessellation.cullDistanceCaptureInstanceStride,
        };
        if (explicitVertices) {
            memcpy(params.explicit_vertices, explicitVertices,
                   explicitVertexCount * sizeof(params.explicit_vertices[0]));
        }
        mglDrawSupportSetVertexBuffer(
            encCtx->encoder, _tessellation.cullDistanceCaptureBuffer, 0u,
            kMGLCullDistanceVertexBufferIndex);
        [self recordLastBoundVertexBuffer:
                  _tessellation.cullDistanceCaptureBuffer
                                   offset:0
                                  atIndex:kMGLCullDistanceVertexBufferIndex];
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        mglDrawSupportSetVertexBytes(
            encCtx->encoder, &params, sizeof(params),
            kMGLCullDistanceParamsBufferIndex);
        [self invalidateLastBoundVertexBufferAtIndex:
                  kMGLCullDistanceParamsBufferIndex];
        return;
    }

    /* Scan enabled attributes for cull distance entries. The GLSL source
     * uses "culldistance_data" as the attribute name. We identify them
     * via the SPIRV-Cross resource list (which preserves the name) or
     * by checking the MSL source for [[attribute(N)]] with that name. */
    id<MTLBuffer> cullMtlBuffer = nil;
    GLintptr cullBindingOffset = 0;
    GLuint cullStride = 0;
    GLuint cullDistSize = 0;
    GLintptr cullFirstRelativeOffset = -1;

    SpirvResourceList *vsInputs =
        &activeProgram->spirv_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];

    for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }
        /* Find the resource name for this attribute. */
        const char *attrName = NULL;
        if (vsInputs && vsInputs->list) {
            for (GLuint r = 0; r < vsInputs->count; r++) {
                SpirvResource *res = &vsInputs->list[r];
                if (res->location == attrib) {
                    attrName = res->name;
                    break;
                }
                if (res->gl_array_size > 1 &&
                    attrib >= res->location &&
                    attrib < res->location + (GLuint)res->gl_array_size) {
                    attrName = res->name;
                    break;
                }
            }
        }
        /* Fall back to attrib_location_names if the resource name is missing. */
        if (!attrName && attrib < MAX_ATTRIBS) {
            attrName = activeProgram->attrib_location_names[attrib];
        }
        if (!attrName) {
            continue;
        }
        /* Match "culldistance_data" or "culldistance_data[N]" */
        if (strncmp(attrName, "culldistance_data", 17) != 0) {
            continue;
        }
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(ctx, vao, attrib, "bindCullDistanceEmu", &resolved)) {
            continue;
        }
        if (!resolved.buffer || !resolved.buffer->data.mtl_data) {
            continue;
        }
        if (cullDistSize == 0) {
            /* First cull distance attribute: record buffer/stride/offset. */
            cullMtlBuffer = (__bridge id<MTLBuffer>)resolved.buffer->data.mtl_data;
            cullBindingOffset = resolved.binding_offset;
            cullStride = resolved.stride;
            cullFirstRelativeOffset = resolved.relativeoffset;
        } else {
            /* Subsequent cull distance attributes: verify they share the same
             * buffer and stride. If not, fall back to the first attribute's
             * layout (the CTS test uses a single interleaved buffer). */
            if (resolved.buffer->data.mtl_data != (void *)(__bridge void *)cullMtlBuffer ||
                resolved.stride != cullStride) {
                /* Layout mismatch; keep the first attribute's layout. */
            }
        }
        cullDistSize++;
    }

    if (!cullMtlBuffer || cullDistSize == 0) {
        /* No cull distance attributes found; bind a dummy buffer to satisfy
         * Metal validation (the shader still references the slots). */
        static id<MTLBuffer> sDummyCullBuffer = nil;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            float dummy[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            sDummyCullBuffer = mglDrawSupportCreateBufferWithBytes(
                _device, dummy, sizeof(dummy), MTLResourceStorageModeShared);
        });
        cullMtlBuffer = sDummyCullBuffer;
        cullBindingOffset = 0;
        cullStride = 4;
        cullFirstRelativeOffset = 0;
        cullDistSize = 0; /* zero size means the shader loop is skipped */
    }

    /* The cull distance offset within each vertex is the binding offset plus
     * the relative offset of the first cull distance attribute. */
    uint32_t culldist_offset = (uint32_t)(cullBindingOffset + (cullFirstRelativeOffset >= 0 ? cullFirstRelativeOffset : 0));

    MGLCullDistanceEmuParams params = {0};
    params.prim_vertex_count = prim_vertex_count;
    params.culldist_offset = culldist_offset;
    params.vertex_stride = (uint32_t)cullStride;
    params.culldist_size = cullDistSize;
    params.first_vertex = firstVertex;
    params.explicit_vertex_count = explicitVertexCount;
    memset(params.explicit_vertices, 0, sizeof(params.explicit_vertices));
    if (explicitVertices) {
        memcpy(params.explicit_vertices, explicitVertices,
               explicitVertexCount * sizeof(params.explicit_vertices[0]));
    }

    mglDrawSupportSetVertexBuffer(
        encCtx->encoder, cullMtlBuffer, 0,
        kMGLCullDistanceVertexBufferIndex);
    [self recordLastBoundVertexBuffer:cullMtlBuffer
                               offset:0
                              atIndex:kMGLCullDistanceVertexBufferIndex];
    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    mglDrawSupportSetVertexBytes(
        encCtx->encoder, &params, sizeof(params),
        kMGLCullDistanceParamsBufferIndex);
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
    if (!mode || *mode != GL_PATCHES) {
        return NO;
    }
    if (!drawCtx || count <= 0) {
        return YES;
    }

    self->ctx = drawCtx;

    Program *tcsProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_CONTROL_SHADER);
    Program *tesProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_EVALUATION_SHADER);
    if (tcsProgram && !tcsProgram->shader_slots[_TESS_CONTROL_SHADER]) {
        tcsProgram = NULL;
    }
    if (tesProgram && !tesProgram->shader_slots[_TESS_EVALUATION_SHADER]) {
        tesProgram = NULL;
    }
    if (!tcsProgram && !tesProgram) {
        return NO;
    }

    if (instanceCount <= 0) {
        return YES;
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

    const BOOL airTES = tesProgram &&
        tesProgram->spirv[_TESS_EVALUATION_SHADER].metallib_bytes != NULL;
    const BOOL forceComputeTES = mglEnvFlagEnabled("MGL_TESS_COMPUTE_FALLBACK");
    BOOL nativeTES = !forceComputeTES &&
        mglNativeTESInterfaceSupported(tcsProgram, tesProgram);

    GLuint patchVertices = MAX(1u, (GLuint)MGL_STATE(drawCtx)->var.patch_vertices);
    GLuint patchCount = (GLuint)count / patchVertices;
    if (patchCount == 0u) {
        return YES;
    }

    /* P0 contract (mgl_air_tess_abi.h): the whole GL_PATCHES draw is
     * described by one value-state struct consumed by the TCS/TES
     * dispatchers — no more threading six scalars and re-deriving
     * per-patch layout numbers at each call site. */
    uint32_t restartIndex = 0u;
    bool restartEnabled = false;
    if (indexType != 0u) {
        restartEnabled =
            mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    }
    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    MGLAIRTessDrawContract contract;
    memset(&contract, 0, sizeof(contract));
    contract.patch_vertices = patchVertices;
    contract.vertex_count = (uint32_t)count;
    contract.patch_count = patchCount;
    contract.instance_count = instanceCount > 0 ? (uint32_t)instanceCount : 1u;
    contract.base_instance = baseInstance;
    contract.first = first;
    contract.index_type = indexType;
    contract.index_source = (uint64_t)(uintptr_t)indices;
    contract.index_count = indexType != 0u ? (uint64_t)count : 0u;
    contract.base_vertex = baseVertex;
    contract.primitive_restart = restartEnabled ? 1u : 0u;
    contract.restart_index = restartIndex;
    contract.tess_factor_bytes_per_patch = MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES;
    contract.tess_gen_mode = tesProgram
        ? (uint32_t)tesProgram->tess_gen_mode : (uint32_t)GL_TRIANGLES;
    contract.point_mode = tesProgram
        ? (uint32_t)tesProgram->tess_gen_point_mode : 0u;
    contract.tcs_out_vertices =
        tcsProgram && tcsProgram->tess_control_output_vertices > 0u
            ? tcsProgram->tess_control_output_vertices : patchVertices;
    contract.per_vertex_out_stride = vertexProgram
        ? mglAIRPerVertexStrideForResources(
              &vertexProgram->spirv_resources_list[_VERTEX_SHADER]
                                                   [_STAGE_OUTPUT_RES])
        : MGL_AIR_PER_VERTEX_STRIDE;
    contract.patch_out_stride = 16u; /* refined by the TCS dispatcher */

    _tessellation.tessVertexCaptureBuffer = nil;
    _tessellation.tessVertexCaptureOffset = 0u;
    if (nativeTES) {
        if (indexType != 0u || instanceCount != 1) {
            nativeTES = NO;
        } else {
            NSUInteger captureOffset = 0u;
            id<MTLBuffer> capture =
                [self captureAIRVertexPositionsForTessellation:drawCtx
                                                         first:first
                                                         count:count
                                                 instanceCount:instanceCount
                                                  baseInstance:baseInstance
                                                    outOffset:&captureOffset];
            if (!capture) {
                nativeTES = NO;
            } else {
                _tessellation.tessVertexCaptureBuffer = capture;
                _tessellation.tessVertexCaptureOffset = captureOffset;
            }
        }
    }

    if (nativeTES && !tcsProgram) {
        _tessellation.tcsOutputBuffer =
            _tessellation.tessVertexCaptureBuffer;
        _tessellation.tcsOutputOffset =
            _tessellation.tessVertexCaptureOffset;
        _tessellation.tcsOutputStride = contract.per_vertex_out_stride;
        _tessellation.tcsOutVertices = patchVertices;
        _tessellation.tessFactorBuffer = mglDefaultTessFactorBuffer(
            _device, MGL_STATE(drawCtx), patchCount);
        if (!_tessellation.tcsOutputBuffer ||
            !_tessellation.tessFactorBuffer) {
            nativeTES = NO;
        }
    }

    if (tcsProgram) {
        if (![self dispatchTessControlShader:drawCtx
                                     program:tcsProgram
                                    contract:&contract]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            _tessellation.tessVertexCaptureBuffer = nil;
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }
    }

    if (nativeTES) {
        id<MTLBuffer> nativeFactors = mglNativeTessFactorBuffer(
            _device, _tessellation.tessFactorBuffer,
            tesProgram->tess_gen_mode, patchCount);
        if (!nativeFactors || !_tessellation.tcsOutputBuffer ||
            _tessellation.tcsOutputStride < MGL_AIR_PER_VERTEX_STRIDE) {
            NSLog(@"MGL TESS ERROR: invalid native TES buffers program=%u",
                  (unsigned)tesProgram->name);
            mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                             GL_OUT_OF_MEMORY);
            drawCtx->state.dirty_bits = DIRTY_ALL;
            _tessellation.tessVertexCaptureBuffer = nil;
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }

        _tessellation.nativeTessFactorBuffer = nativeFactors;
        _tessellation.nativeTESProgram = tesProgram;
        _tessellation.nativeTESActive = YES;
        drawCtx->state.dirty_bits = DIRTY_ALL;

        BOOL stateReady = [self processGLState:true];
        if (!stateReady || !_renderPassManager.state->currentRenderEncoder) {
            _tessellation.nativeTESActive = NO;
            _tessellation.nativeTESProgram = NULL;
            _tessellation.tessVertexCaptureBuffer = nil;
            _tessellation.tessVertexCaptureOffset = 0u;
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }

        if (![self currentDrawRasterizationIsEmpty] &&
            ![self currentDrawModeIsFullyCulled:GL_TRIANGLES]) {
            [self applyPolygonOffsetForDrawMode:GL_TRIANGLES];
            id<MTLRenderCommandEncoder> encoder =
                _renderPassManager.state->currentRenderEncoder;
            mglDrawSupportSetVertexBuffer(
                encoder, _tessellation.tcsOutputBuffer,
                _tessellation.tcsOutputOffset, 0u);
            [self recordLastBoundVertexBuffer:_tessellation.tcsOutputBuffer
                                       offset:_tessellation.tcsOutputOffset
                                      atIndex:0u];
            mglDrawSupportSetVertexBuffer(
                encoder, _tessellation.tcsOutputBuffer,
                _tessellation.tcsOutputOffset, 30u);
            [self recordLastBoundVertexBuffer:_tessellation.tcsOutputBuffer
                                       offset:_tessellation.tcsOutputOffset
                                      atIndex:30u];
            GLuint patchInfo[2] = {patchVertices, _tessellation.tcsOutVertices};
            if (patchInfo[1] == 0u) patchInfo[1] = patchVertices;
            mglDrawSupportSetVertexBytes(
                encoder, patchInfo, sizeof(patchInfo), 28u);
            if (_tessellation.tcsPatchOutBuffer) {
                mglDrawSupportSetVertexBuffer(
                    encoder, _tessellation.tcsPatchOutBuffer, 0u, 27u);
                [self recordLastBoundVertexBuffer:
                          _tessellation.tcsPatchOutBuffer
                                           offset:0u
                                          atIndex:27u];
            }
            mglDrawSupportSetTessellationFactors(
                encoder, nativeFactors, 0u, 0u);
            mglDrawSupportDrawPatches(
                encoder, _tessellation.tcsOutVertices, 0u, patchCount,
                nil, 0u, (NSUInteger)instanceCount,
                (NSUInteger)baseInstance);
            _currentCBHasWork = YES;

            GLuint64 primitives = mglNativeTessPrimitiveCount(
                _tessellation.tessFactorBuffer, tesProgram, patchCount,
                (GLuint)instanceCount);
            mglRecordActivePrimitiveQueryDraw(drawCtx, primitives, primitives);
        }

        _tessellation.nativeTESActive = NO;
        _tessellation.nativeTESProgram = NULL;
        _tessellation.tessVertexCaptureBuffer = nil;
        _tessellation.tessVertexCaptureOffset = 0u;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    if (airTES) {
        NSLog(@"MGL TESS ERROR: native AIR TES interface unsupported for program %u%s",
              (unsigned)tesProgram->name,
              forceComputeTES ? " (AIR has no compute fallback variant)" : "");
        /* P0 contract: an unsupported tessellation draw must surface a GL
         * error, not silently drop the patch stream. */
        mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                         GL_INVALID_OPERATION);
        drawCtx->state.dirty_bits = DIRTY_ALL;
        _tessellation.tessVertexCaptureBuffer = nil;
        _tessellation.tessVertexCaptureOffset = 0u;
        return YES;
    }

    if (tesProgram) {
        if (![self dispatchTessEvaluationShader:drawCtx
                                           program:tesProgram
                                          contract:&contract]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    drawCtx->state.dirty_bits = DIRTY_ALL;
    _tessellation.tessVertexCaptureBuffer = nil;
    _tessellation.tessVertexCaptureOffset = 0u;
    (void)label;
    return YES;
}


@end
