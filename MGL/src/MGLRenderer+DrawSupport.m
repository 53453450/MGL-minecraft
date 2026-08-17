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

static BOOL mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext)
{
    if (!encodeContext) return NO;
    return mglRenderCppRenderEncoderOwnerHasCurrent(
        encodeContext->render_encoder_owner) == 1;
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
    /* P4.5 (item 1141/887): 索引流 gather（BYTE/SHORT/INT 元素宽度、原始
     * 重启、完整图元计数、尾不完整组丢弃）迁入 C++
     * （mglRenderCppGeometryGatherIndices，两门共用；调用方释放 gather）。 */
    (void)baseVertex; /* gather stores raw index values (vertex_id) */
    if (!outGather || !outGatherCount || !outPrimitiveCount || !outMaxIndex) {
        return false;
    }
    const uint32_t elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    MGLRenderCppGeometryGatherResult result = {0};
    if (mglRenderCppGeometryGatherIndices(
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

static MGLMetalBufferRef mglDrawSupportCreateBuffer(
    MGLMetalDeviceRef device,
    NSUInteger length,
    MTLResourceOptions options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer MGLMetalBufferRef)buffer;
    }
    return nil;
}

static MGLMetalBufferRef mglDrawSupportCreateBufferWithBytes(
    MGLMetalDeviceRef device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer MGLMetalBufferRef)buffer;
    }
    return nil;
}

static MGLMetalBlitCommandEncoderRef mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner)
{
    return mglRenderCreateBlitEncoderForCommandBufferOwner(
        commandBufferOwner);
}

static void mglDrawSupportBlitCopyBuffer(MGLMetalBlitCommandEncoderRef encoder,
                                         MGLMetalBufferRef source,
                                         NSUInteger sourceOffset,
                                         MGLMetalBufferRef destination,
                                         NSUInteger destinationOffset,
                                         NSUInteger size)
{
    (void)mglRenderCppBlitCopyBuffer(
        (__bridge void *)encoder, (__bridge void *)source, sourceOffset,
        (__bridge void *)destination, destinationOffset, size);
}

static void mglDrawSupportEndBlitEncoder(MGLMetalBlitCommandEncoderRef encoder)
{
    (void)mglRenderCppEndBlitEncoder((__bridge void *)encoder);
}

static void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    MGLMetalBufferRef buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderCppSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_CPP_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderCppSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_CPP_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglDrawSupportDrawIndexedPrimitives(
    void *renderEncoderOwner,
    MTLPrimitiveType primitiveType,
    NSUInteger indexCount,
    MGLMetalBufferRef indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(renderEncoderOwner,
        &(MGLRenderCppDrawPlan){
            .kind = MGL_RENDER_CPP_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)MTLIndexTypeUInt32,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        }, NULL, 0);
}

/* Variant that honors the GL index type (UInt8/UInt16/UInt32).  Used by the
 * GS indexed capture so the original EBO drives vertex fetch directly. */
static void mglDrawSupportDrawIndexedPrimitivesType(
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
    MGLRenderCppDrawPlan plan = {
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
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static void mglDrawSupportDrawPrimitives(
    void *renderEncoderOwner,
    MTLPrimitiveType primitiveType,
    NSUInteger vertexStart,
    NSUInteger vertexCount,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static void mglDrawSupportDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    MTLPrimitiveType primitiveType,
    MGLMetalBufferRef indirectBuffer,
    NSUInteger indirectBufferOffset)
{
    MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_ARRAY_INDIRECT,
            .primitive_type = (uint32_t)primitiveType,
            .indirect_buffer = (__bridge void *)indirectBuffer,
            .indirect_buffer_offset = indirectBufferOffset,
        };
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static MGLMetalComputeCommandEncoderRef mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner)
{
    return mglRenderCreateComputeEncoderForCommandBufferOwner(
        commandBufferOwner);
}

static void mglDrawSupportSetComputePipeline(
    MGLMetalComputeCommandEncoderRef encoder,
    MGLMetalComputePipelineStateRef pipeline)
{
    (void)mglRenderCppSetComputePipelineState((__bridge void *)encoder,
                                              (__bridge void *)pipeline);
}

static void mglDrawSupportSetComputeBuffer(
    MGLMetalComputeCommandEncoderRef encoder,
    MGLMetalBufferRef buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderCppSetComputeBuffer((__bridge void *)encoder,
                                       (__bridge void *)buffer, offset,
                                       (uint32_t)index);
}

static void mglDrawSupportSetComputeBytes(
    MGLMetalComputeCommandEncoderRef encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderCppSetComputeBytes((__bridge void *)encoder, bytes,
                                      length, (uint32_t)index);
}

static void mglDrawSupportDispatchCompute(
    MGLMetalComputeCommandEncoderRef encoder,
    MTLSize groups,
    MTLSize threads)
{
    (void)mglRenderCppDispatchCompute(
        (__bridge void *)encoder, (uint32_t)groups.width,
        (uint32_t)groups.height, (uint32_t)groups.depth,
        (uint32_t)threads.width, (uint32_t)threads.height,
        (uint32_t)threads.depth);
}

static void mglDrawSupportEndComputeEncoder(
    MGLMetalComputeCommandEncoderRef encoder)
{
    (void)mglRenderCppEndComputeEncoder((__bridge void *)encoder);
}

static void mglDrawSupportSetTessellationFactors(
    void *renderEncoderOwner,
    MGLMetalBufferRef buffer,
    NSUInteger offset,
    NSUInteger instanceStride)
{
    (void)mglRenderCppSetTessellationFactorBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset, instanceStride);
}

static void mglDrawSupportDrawPatches(
    void *renderEncoderOwner,
    NSUInteger controlPointCount,
    NSUInteger patchStart,
    NSUInteger patchCount,
    MGLMetalBufferRef patchIndexBuffer,
    NSUInteger patchIndexBufferOffset,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_PATCHES,
            .primitive_type = (uint32_t)MTLPrimitiveTypeTriangle,
            .control_point_count = controlPointCount,
            .patch_start = patchStart,
            .patch_count = patchCount,
            .patch_index_buffer = (__bridge void *)patchIndexBuffer,
            .patch_index_buffer_offset = patchIndexBufferOffset,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static void mglDrawSupportDrawIndexedPatches(
    void *renderEncoderOwner,
    NSUInteger controlPointCount,
    NSUInteger patchStart,
    NSUInteger patchCount,
    MGLMetalBufferRef patchIndexBuffer,
    NSUInteger patchIndexBufferOffset,
    MGLMetalBufferRef controlPointIndexBuffer,
    NSUInteger controlPointIndexBufferOffset,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_INDEXED_PATCHES,
            .primitive_type = (uint32_t)MTLPrimitiveTypeTriangle,
            .control_point_count = controlPointCount,
            .patch_start = patchStart,
            .patch_count = patchCount,
            .patch_index_buffer = (__bridge void *)patchIndexBuffer,
            .patch_index_buffer_offset = patchIndexBufferOffset,
            .control_point_index_buffer =
                (__bridge void *)controlPointIndexBuffer,
            .control_point_index_buffer_offset =
                controlPointIndexBufferOffset,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx,
                                               GLuint64 generated,
                                               GLuint64 written);
extern void mglRecordActivePrimitiveQueryDrawIndexed(GLMContext ctx,
                                                      GLuint index,
                                                      GLuint64 generated,
                                                      GLuint64 written);
extern GLboolean mglHasActiveIndexedPrimitiveQuery(void);

static void mglRecordGeometryPrimitiveQueries(
    GLMContext ctx,
    GLuint64 generatedStream0,
    GLuint64 writtenStream0,
    BOOL xfbActive,
    const MGLAIRGSXFBMeta *meta,
    uint32_t streamCount,
    const NSUInteger *streamStride)
{
    mglRecordActivePrimitiveQueryDraw(
        ctx, generatedStream0, xfbActive ? writtenStream0 : 0u);
    if (!meta || !streamStride) return;
    if (streamCount > MGL_AIR_GS_MAX_STREAMS) {
        streamCount = MGL_AIR_GS_MAX_STREAMS;
    }
    for (uint32_t s = 1u; s < streamCount; s++) {
        GLuint64 written = 0u;
        if (xfbActive && streamStride[s] > 0u) {
            written = meta->stream[s].written /
                      (GLuint64)streamStride[s];
        }
        mglRecordActivePrimitiveQueryDrawIndexed(
            ctx, s, (GLuint64)meta->stream[s].generated, written);
    }
}

static BOOL mglCheckedTessCaptureSize(GLsizei count, GLsizei instanceCount,
                                      NSUInteger stride,
                                      NSUInteger *sizeOut,
                                      NSUInteger *offsetOut)
{
    /* P4.5 (item 1141/887): 溢出检查的 capture size 数学在 C++
     * （mglRenderCppCheckedTessCaptureSize，纯数据变换，两门共用）。 */
    uint64_t size = 0u;
    uint64_t offset = 0u;
    if (mglRenderCppCheckedTessCaptureSize(
            (int64_t)count, (int64_t)instanceCount, (uint64_t)stride,
            (uint64_t)MGL_AIR_PER_VERTEX_STRIDE, &size, &offset) != 0) {
        return NO;
    }
    *sizeOut = (NSUInteger)size;
    *offsetOut = (NSUInteger)offset;
    return YES;
}

static BOOL mglNativeTESInterfaceSupported(Program *tcsProgram,
                                           Program *tesProgram)
{
    if (!tesProgram) {
        return NO;
    }
    /* P4.5 (item 1141/887): 模块/函数存在性 + point-mode/XFB 排除 +
     * TRI/QUADS 门 + MTL::Function patchType/patchControlPointCount 一致
     * 性判定在 C++（mglRenderCppNativeTESInterfaceSupported，经 bridge
     * 读取 MTL::Function，两门共用）。 */
    return mglRenderCppNativeTESInterfaceSupported(
        tesProgram->modules[_TESS_EVALUATION_SHADER].mtl_function,
        (uint64_t)tesProgram->modules[_TESS_EVALUATION_SHADER].metallib_bytes,
        (uint32_t)tesProgram->tess_gen_point_mode,
        (uint32_t)tesProgram->transform_feedback_varying_count,
        (uint32_t)tesProgram->tess_gen_mode,
        tcsProgram ? tcsProgram->modules[_TESS_CONTROL_SHADER].mtl_function : NULL,
        tcsProgram ? (uint64_t)tcsProgram->modules[_TESS_CONTROL_SHADER].metallib_bytes : 0u,
        tcsProgram ? (uint32_t)tcsProgram->tess_control_output_vertices : 0u) != 0;
}

static MGLMetalBufferRef mglDefaultTessFactorBuffer(MGLMetalDeviceRef device,
                                                GLMState *state,
                                                GLuint patchCount)
{
    if (!device || !state || patchCount == 0u) return nil;
    const NSUInteger stride = 12u;
    if ((NSUInteger)patchCount > NSUIntegerMax / stride) return nil;
    MGLMetalBufferRef buffer = mglDrawSupportCreateBuffer(
        device, (NSUInteger)patchCount * stride,
        MTLResourceStorageModeShared);
    if (!buffer || !buffer.contents) return nil;
    /* P4.5 (item 1141/887): 默认 factor 填充在 C++（__fp16 打包，纯数据
     * 变换，两门共用）。 */
    if (mglRenderCppFillDefaultTessFactorBuffer(
            (void *)buffer.contents,
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
static MGLMetalBufferRef mglCachedDefaultTessFactorBuffer(
    MGLMetalDeviceRef device, MGLRendererBackendHandle *backend, GLMState *state,
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
        return (__bridge MGLMetalBufferRef)cached;
    }
    MGLMetalBufferRef fresh = mglDefaultTessFactorBuffer(device, state, patchCount);
    if (!fresh) return nil;
    if (mglRendererBackendPutTessFactorBuffer(
            backend, patchCount, levels, (__bridge void *)fresh) != 0) {
        return fresh;
    }
    return fresh;
}

static MGLMetalBufferRef mglNativeTessFactorBuffer(MGLMetalDeviceRef device,
                                                MGLMetalBufferRef canonical,
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
    MGLMetalBufferRef result = mglDrawSupportCreateBuffer(
        device, (NSUInteger)patchCount * triangleStride,
        MTLResourceStorageModeShared);
    if (!result || !result.contents) {
        return nil;
    }
    /* P4.5 (item 1141/887): canonical->triangle 重打包在 C++
     * （12B/patch -> 8B/patch，纯数据变换，两门共用）。 */
    if (mglRenderCppRepackTessFactorTriangles(
            (const void *)canonical.contents, (uint64_t)canonical.length,
            (void *)result.contents,
            (uint64_t)((NSUInteger)patchCount * triangleStride),
            patchCount) != 0) {
        return nil;
    }
    return result;
}

static GLuint64 mglNativeTessPrimitiveCount(MGLMetalBufferRef canonical,
                                             Program *tesProgram,
                                             GLuint patchCount,
                                             GLuint instanceCount)
{
    if (!canonical || !canonical.contents || !tesProgram || patchCount == 0u) {
        return 0u;
    }
    /* P4.5 (item 1141/887): 原生 primitive count（GL 4.6 §11.2.2.2 ceil
     * 规则 + discard 判定）在 C++（mglRenderCppTessPrimitiveCount，纯数据
     * 变换，两门共用）。 */
    return (GLuint64)mglRenderCppTessPrimitiveCount(
        (const void *)canonical.contents, (uint64_t)canonical.length,
        patchCount, (uint32_t)tesProgram->tess_gen_mode,
        instanceCount);
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
    const uint64_t endVertex = (uint64_t)(uint32_t)first +
                               (uint64_t)(uint32_t)count;
    const uint64_t lastCaptureIndex =
        (uint64_t)((uint32_t)instanceCount - 1u) * (uint64_t)(uint32_t)count +
        endVertex;
    if (endVertex == 0u || lastCaptureIndex == 0u ||
        lastCaptureIndex > NSUIntegerMax / 32u) return NO;
    MGLMetalBufferRef capture = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)(lastCaptureIndex * 32u),
        MTLResourceStorageModeShared);
    if (!capture) return NO;

    self->ctx = drawCtx;
    _tessellation.cullDistanceCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.cullDistanceCaptureActive = NO;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return NO;
    }
    MGLCullDistanceEmuParams params = {
        .prim_vertex_count = 1u,
        .culldist_offset = 0u,
        .vertex_stride = 32u,
        .culldist_size = MIN(vertexProgram->cull_distance_count, 8u),
        .first_vertex = (uint32_t)first,
        .first_instance = baseInstance,
        .instance_stride = (uint32_t)count,
    };
    mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, capture, 0u, 29u);
    mglDrawSupportSetVertexBytes(
        _renderPassManager.state->currentRenderEncoderOwner, &params, sizeof(params), kMGLCullDistanceParamsBufferIndex);
    mglDrawSupportDrawPrimitives(_renderPassManager.state->currentRenderEncoderOwner, MTLPrimitiveTypePoint,
                                 (NSUInteger)first, (NSUInteger)count,
                                 (NSUInteger)instanceCount,
                                 (NSUInteger)baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.cullDistanceCaptureActive = NO;
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(
        _backend, (__bridge void *)capture);
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
    const uint32_t elemWidth = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t scanMin = 0u, scanMax = 0u, scanValid = 0;
    if (mglRenderCppScanIndexRangeIgnoringRestart(
            indexBytes, elemWidth, (uint32_t)count,
            restartEnabled ? 1 : 0, restartIndex,
            &scanMin, &scanMax, &scanValid) != 0 || !scanValid) {
        return NO;
    }
    uint32_t minIndex = scanMin;
    uint32_t maxIndex = scanMax;
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

    if (activeProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function) {
        if (![self captureAIRCullDistancesForElementDraw:ctx
                                             indexBytes:indexBytes
                                              indexType:indexType
                                                  count:count
                                             baseVertex:baseVertex
                                          instanceCount:instanceCount
                                           baseInstance:baseInstance] ||
            ![self processGLState:true] ||
            mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
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

    if (mode == GL_TRIANGLE_STRIP && count >= 3) {
        NSUInteger indexCount = 0u;
        MGLMetalBufferRef indexBuffer = mglNewTriangleStripArrayIndexBuffer(
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
                encCtx->render_encoder_owner, MTLPrimitiveTypeTriangle, 3u, indexBuffer,
                primitive * 3u * sizeof(uint32_t),
                (NSUInteger)instanceCount, (NSInteger)first,
                (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_TRIANGLE_FAN && count >= 3) {
        NSUInteger indexCount = 0u;
        MGLMetalBufferRef indexBuffer = mglNewTriangleFanArrayIndexBuffer(
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
                encCtx->render_encoder_owner, MTLPrimitiveTypeTriangle, 3u, indexBuffer,
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
                encCtx->render_encoder_owner, MTLPrimitiveTypeLine,
                (NSUInteger)(first + primitive), 2u,
                (NSUInteger)instanceCount, (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_LINE_LOOP && count >= 2) {
        NSUInteger indexCount = 0u;
        MGLMetalBufferRef indexBuffer = mglNewLineLoopArrayIndexBuffer(
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
                encCtx->render_encoder_owner, MTLPrimitiveTypeLine, 2u, indexBuffer,
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
    MGLMetalBufferRef captureBuffer = (__bridge MGLMetalBufferRef)
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
    if (mglRenderCppCreateCullDistanceIndexPlan(
            (__bridge void *)_device, indexBytes, indexType,
            (uint64_t)count, mode,
            restartEnabled ? 1 : 0, restartIndex, baseVertex,
            polygonLineMode ? 1 : 0, &planOwner, &indexBufferHandle,
            &primitiveCount) != 0 || !planOwner) {
        return YES;
    }

    MGLMetalBufferRef indexBuffer =
        (__bridge MGLMetalBufferRef)indexBufferHandle;
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
                encCtx->render_encoder_owner,
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

- (MGLMetalBufferRef)captureAIRVertexPositionsForTessellation:(GLMContext)drawCtx
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

    NSUInteger captureSize = 0u;
    NSUInteger captureOffset = 0u;
    NSUInteger captureStride = mglAIRPerVertexStrideForResources(
        &vertexProgram->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    const NSUInteger recordsPerInstance = (NSUInteger)count;
    if (!mglCheckedTessCaptureSize(recordsPerInstance, instanceCount,
                                   captureStride, &captureSize,
                                   &captureOffset)) {
        return nil;
    }
    MGLMetalBufferRef capture = mglDrawSupportCreateBuffer(
        _device, captureSize, MTLResourceStorageModeShared);
    if (!capture) return nil;

    self->ctx = drawCtx;
    _tessellation.tessVertexCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.tessVertexCaptureActive = NO;
        return nil;
    }
    mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, capture, 0u, 29u);
    const uint32_t captureParams[3] = {
        (uint32_t)first, (uint32_t)recordsPerInstance, baseInstance,
    };
    mglDrawSupportSetVertexBytes(
        _renderPassManager.state->currentRenderEncoderOwner, captureParams, sizeof(captureParams), 28u);
    mglDrawSupportDrawPrimitives(_renderPassManager.state->currentRenderEncoderOwner, MTLPrimitiveTypePoint,
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
 * (mgl_air_gs_abi.h §7).  Runs indexed draws against the original
 * EBO so Metal's baseVertex is applied to stage_in fetch; the capture
 * kernel's vertex_id is the raw index value, so records are sparse
 * ([instance][vertex_id], span = maxIndex+1 per instance). */
- (MGLMetalBufferRef)captureAIRVertexPositionsForGeometryIndexed:(GLMContext)drawCtx
                                                  indexBuffer:(MGLMetalBufferRef)indexBuffer
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
        !vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function) {
        return nil;
    }

    NSUInteger captureSize = 0u;
    NSUInteger captureOffset = 0u;
    NSUInteger captureStride = mglAIRPerVertexStrideForResources(
        &vertexProgram->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    const NSUInteger recordsPerInstance = (NSUInteger)maxIndex + 1u;
    if (!mglCheckedTessCaptureSize(recordsPerInstance, instanceCount,
                                   captureStride, &captureSize,
                                   &captureOffset)) {
        return nil;
    }
    MGLMetalBufferRef capture = mglDrawSupportCreateBuffer(
        _device, captureSize, MTLResourceStorageModeShared);
    if (!capture) return nil;
    self->ctx = drawCtx;
    _tessellation.tessVertexCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.tessVertexCaptureActive = NO;
        return nil;
    }
    mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, capture, 0u, 29u);
    const uint32_t captureParams[3] = {
        0u, (uint32_t)recordsPerInstance, baseInstance,
    };
    mglDrawSupportSetVertexBytes(
        _renderPassManager.state->currentRenderEncoderOwner, captureParams, sizeof(captureParams), 28u);
    /* The capture kernel indexes records by raw vertex_id with no bounds
     * check; a primitive-restart marker (0xFFFFFFFF for UInt32) in the
     * stream would write past the sparse record span and corrupt the
     * next instance's data.  Sanitize the marker away (to vertex 0, whose
     * record no gathered patch ever references) before drawing. */
    MGLMetalBufferRef sanitizedIndexBuffer = indexBuffer;
    NSUInteger sanitizedIndexOffset = indexOffset;
    uint32_t restartIndex = 0u;
    if (mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex)) {
        const NSUInteger elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
            : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
        const NSUInteger streamBytes = (NSUInteger)count * elemBytes;
        if (indexBuffer.contents &&
            (NSUInteger)indexOffset + streamBytes <= indexBuffer.length) {
            uint8_t *copy = malloc(streamBytes);
            if (copy) {
                memcpy(copy,
                       (const uint8_t *)indexBuffer.contents + indexOffset,
                       streamBytes);
                if (elemBytes == 1u) {
                    for (GLsizei i = 0; i < count; i++)
                        if (((const uint8_t *)copy)[i] ==
                                (uint8_t)restartIndex)
                            ((uint8_t *)copy)[i] = 0u;
                } else if (elemBytes == 2u) {
                    for (GLsizei i = 0; i < count; i++)
                        if (((const uint16_t *)copy)[i] ==
                                (uint16_t)restartIndex)
                            ((uint16_t *)copy)[i] = 0u;
                } else {
                    for (GLsizei i = 0; i < count; i++)
                        if (((const uint32_t *)copy)[i] == restartIndex)
                            ((uint32_t *)copy)[i] = 0u;
                }
                MGLMetalBufferRef clean =
                    mglDrawSupportCreateBufferWithBytes(
                        _device, copy, streamBytes,
                        MTLResourceStorageModeShared);
                free(copy);
                if (clean) {
                    sanitizedIndexBuffer = clean;
                    sanitizedIndexOffset = 0u;
                }
            }
        }
    }
    mglDrawSupportDrawIndexedPrimitivesType(
        _renderPassManager.state->currentRenderEncoderOwner, MTLPrimitiveTypePoint, (NSUInteger)count, indexType,
        sanitizedIndexBuffer, sanitizedIndexOffset, (NSUInteger)instanceCount,
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
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG enter program=%u mode=0x%x count=%d first=%d",
              (unsigned)program->name, (unsigned)mode, (int)count, (int)first);
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
        !program->modules[_GEOMETRY_SHADER].metallib_bytes ||
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

    /* Indexed draws first gather the element stream (mgl_air_gs_abi.h §7):
     * restart markers split the stream, vertices re-group into input
     * primitives, and the resulting per-instance uint32 gather array carries
     * raw index values so the GS kernel can locate sparse capture records. */
    uint32_t *gatherArray = NULL;
    uint32_t gatherCount = 0u;
    uint32_t gatherPrimitives = 0u;
    uint32_t gatherMaxIndex = 0u;
    const uint8_t *indexBytes = NULL;
    MGLMetalBufferRef eboMetal = nil;
    NSUInteger indexOffsetBytes = 0u;
    MTLIndexType captureIndexType = MTLIndexTypeUInt32;
    MGLMetalBufferRef gatherBuf = nil;
    MGLAIRGSGatherParams gparams;
    memset(&gparams, 0, sizeof(gparams));
    if (indexedDraw) {
        Buffer *ebo = getElementBuffer(drawCtx);
        if (!ebo || ![self processBuffer:ebo] || !ebo->data.mtl_data) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_INVALID_OPERATION);
            return YES;
        }
        eboMetal = (__bridge MGLMetalBufferRef)ebo->data.mtl_data;
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
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES]);
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
    MGLMetalBufferRef input = nil;
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
    MGLMetalComputePipelineStateRef pipeline =
        pipelineResult == 0 && pipelineHandle
            ? (__bridge_transfer MGLMetalComputePipelineStateRef)pipelineHandle
            : nil;
    if (!pipeline) {
        NSLog(@"MGL GS ERROR: compute PSO failed program=%u: %s",
              (unsigned)program->name,
              pipelineError[0] ? pipelineError : "unknown error");
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    MGLRenderCppCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerState(
            _renderPassManager.state->currentCommandBufferOwner,
            &commandState) ||
        commandState.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }
    const NSUInteger outputSize =
        (NSUInteger)workItemCount * recordsPerPrimitive * outputStride;
    MGLMetalBufferRef output = mglDrawSupportCreateBuffer(
        _device, outputSize, MTLResourceStorageModeShared);
    /* ABI (mgl_air_gs_abi.h §3): one 28-byte counts record per work item —
     * 16-byte indirect args + 12 bytes kernel scratch. */
    const NSUInteger countsRecordBytes = MGL_AIR_GS_COUNTS_RECORD_BYTES;
    MGLMetalBufferRef counts = mglDrawSupportCreateBuffer(
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
    /* GS transform feedback (P1, mgl_air_gs_abi.h §5): when GL feedback
     * is active the kernel appends output vertices to the slot-31 stream
     * through the per-stream atomic meta cursors (slot 27).  Stream 0
     * keeps the single-stream path: the GL store is bound directly only
     * when the maximum possible capture fits, otherwise the kernel writes
     * into a full-size temporary and only the prefix containing complete
     * primitives is copied back.  Multi-stream programs share one physical
     * slot-31 buffer split into per-stream segments (capture_base), one
     * segment per used stream, copied back per stream afterward; the GL
     * transform-feedback buffer i receives stream i (documented MGL
     * mapping, stream s has a compact position+varyings record of
     * geometry_stream_xfb_stride[s] bytes). */
    TransformFeedback *xfbState = MGL_STATE(drawCtx)->transform_feedback;
    const bool xfbActive = xfbState && xfbState->active && !xfbState->paused;
    MGLMetalBufferRef xfbTemporary = nil;
    MGLMetalBufferRef xfbCaptureBuffer = nil;
    MGLMetalBufferRef xfbDestinationMTL = nil;
    NSUInteger xfbDestinationOffset = 0u;
    NSUInteger xfbRemainingVisibleBytes = 0u;
    NSUInteger xfbMaxCaptureBytes = 0u;
    const uint32_t gsStreamCount =
        program->geometry_stream_count > 0u ? program->geometry_stream_count
                                            : 1u;
    const bool multiStream = gsStreamCount > 1u;
    NSUInteger streamPhysBase[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger streamCapBytes[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger streamDstOffset[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger streamRemaining[MGL_AIR_GS_MAX_STREAMS] = {0u};
    MGLMetalBufferRef streamDstMTL[MGL_AIR_GS_MAX_STREAMS] = {nil};
    NSUInteger streamStride[MGL_AIR_GS_MAX_STREAMS] = {0u};
    if (xfbActive) {
        streamStride[0] = outputStride;
        for (uint32_t s = 1u; s < gsStreamCount; s++) {
            streamStride[s] = program->geometry_stream_xfb_stride[s];
        }
        if (multiStream) {
            NSUInteger physTotal = 0u;
            for (uint32_t s = 0u; s < gsStreamCount; s++) {
                BufferBaseTarget *slot = &MGL_STATE(drawCtx)
                    ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[s];
                if (!slot->buf) {
                    streamStride[s] = 0u;
                    continue;
                }
                if (!slot->buf->data.mtl_data) {
                    [self bindMTLBuffer:slot->buf];
                }
                MGLMetalBufferRef mtl = (__bridge MGLMetalBufferRef)(
                    slot->buf->data.mtl_data);
                if (!mtl) continue;
                BufferMap map = {0};
                map.buf = slot->buf;
                map.offset = slot->offset;
                map.size = slot->size;
                NSUInteger visible = mglBufferMapVisibleBackingBytes(
                    &map, (size_t)mtl.length);
                NSUInteger sessionOffset = 0u;
                if (xfbState->buffer_write_offsets[s] <=
                    (GLuint64)NSUIntegerMax) {
                    sessionOffset =
                        (NSUInteger)xfbState->buffer_write_offsets[s];
                }
                if (sessionOffset > visible || slot->offset < 0 ||
                    (NSUInteger)slot->offset >
                        NSUIntegerMax - sessionOffset) {
                    continue;
                }
                streamRemaining[s] = visible - sessionOffset;
                streamDstOffset[s] = (NSUInteger)slot->offset + sessionOffset;
                streamDstMTL[s] = mtl;
                NSUInteger maxCap = (s == 0u)
                    ? (NSUInteger)workItemCount * expandedVertices *
                          streamStride[0]
                    : (NSUInteger)workItemCount *
                          (program->geometry_vertices_out > 0u
                               ? program->geometry_vertices_out : 1u) *
                          streamStride[s];
                streamCapBytes[s] = MIN(maxCap, streamRemaining[s]);
                if (streamCapBytes[s] > (NSUInteger)UINT32_MAX) {
                    streamCapBytes[s] = (NSUInteger)UINT32_MAX;
                }
                streamPhysBase[s] = physTotal;
                physTotal += streamCapBytes[s];
            }
            if (physTotal > 0u) {
                xfbTemporary = mglDrawSupportCreateBuffer(
                    _device, physTotal, MTLResourceStorageModeShared);
                if (xfbTemporary) {
                    memset(xfbTemporary.contents, 0, physTotal);
                    xfbCaptureBuffer = xfbTemporary;
                }
            }
        } else {
            xfbMaxCaptureBytes =
                (NSUInteger)workItemCount * expandedVertices * outputStride;
            BufferBaseTarget *xfbSlot = &MGL_STATE(drawCtx)
                ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
            if (xfbSlot->buf) {
                if (!xfbSlot->buf->data.mtl_data) {
                    [self bindMTLBuffer:xfbSlot->buf];
                }
                MGLMetalBufferRef xfbMTL =
                    (__bridge MGLMetalBufferRef)(xfbSlot->buf->data.mtl_data);
                if (xfbMTL) {
                    BufferMap xfbMap = {0};
                    xfbMap.buf = xfbSlot->buf;
                    xfbMap.offset = xfbSlot->offset;
                    xfbMap.size = xfbSlot->size;
                    NSUInteger visibleBytes = mglBufferMapVisibleBackingBytes(
                        &xfbMap, (size_t)xfbMTL.length);
                    NSUInteger sessionOffset = 0u;
                    if (xfbState->buffer_write_offsets[0] <=
                        (GLuint64)NSUIntegerMax) {
                        sessionOffset =
                            (NSUInteger)xfbState->buffer_write_offsets[0];
                    }
                    if (sessionOffset <= visibleBytes && xfbSlot->offset >= 0 &&
                        (NSUInteger)xfbSlot->offset <=
                            NSUIntegerMax - sessionOffset) {
                        xfbRemainingVisibleBytes = visibleBytes - sessionOffset;
                        xfbDestinationOffset =
                            (NSUInteger)xfbSlot->offset + sessionOffset;
                        if (xfbMaxCaptureBytes <= xfbRemainingVisibleBytes) {
                            xfbCaptureBuffer = xfbMTL;
                            xfbDestinationMTL = xfbMTL;
                            xfbSlot->buf->ever_written = GL_TRUE;
                        } else {
                            xfbTemporary = mglDrawSupportCreateBuffer(
                                _device, xfbMaxCaptureBytes,
                                MTLResourceStorageModeShared);
                            if (xfbTemporary) {
                                memset(xfbTemporary.contents, 0,
                                       xfbMaxCaptureBytes);
                                xfbCaptureBuffer = xfbTemporary;
                                xfbDestinationMTL = xfbMTL;
                            }
                        }
                        streamDstMTL[0] = xfbDestinationMTL;
                        streamDstOffset[0] = xfbDestinationOffset;
                        streamRemaining[0] = xfbRemainingVisibleBytes;
                        streamCapBytes[0] = MIN(xfbMaxCaptureBytes,
                                                xfbRemainingVisibleBytes);
                    }
                }
            }
        }
    }
    MGLAIRGSXFBMeta xfbMeta;
    memset(&xfbMeta, 0, sizeof(xfbMeta));
    for (uint32_t s = 0u; s < MGL_AIR_GS_MAX_STREAMS; s++) {
        xfbMeta.stream[s].stride = (xfbCaptureBuffer && streamStride[s] > 0u &&
                                    streamCapBytes[s] > 0u)
            ? (uint32_t)streamStride[s] : 0u;
        xfbMeta.stream[s].capacity_bytes =
            (uint32_t)MIN(streamCapBytes[s], (NSUInteger)UINT32_MAX);
        xfbMeta.stream[s].capture_base =
            (uint32_t)MIN(streamPhysBase[s], (NSUInteger)UINT32_MAX);
    }
    MGLMetalBufferRef xfbMetaBuf = mglDrawSupportCreateBufferWithBytes(
        _device, &xfbMeta, sizeof(xfbMeta), MTLResourceStorageModeShared);
    if (!xfbMetaBuf) {
        drawCtx->state.dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const BOOL cppDispatch = YES;
    MGLMetalComputeCommandEncoderRef compute = nil;
    MGLRenderCppComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = cppDispatch
        ? [NSMutableArray array] : nil;
    if (cppDispatch) {
        executionPlan.pipeline = (__bridge void *)pipeline;
#define MGL_GS_PLAN_BUFFER(resource, bindingOffset, bindingIndex)                \
        do {                                                                     \
            executionPlan.binding_ops[executionPlan.binding_op_count++] =        \
                (MGLRenderCppComputeBindingOp){                                  \
                    0u, (uint32_t)(bindingIndex),                                \
                    (uint64_t)(bindingOffset), (__bridge void *)(resource),      \
                    NULL, 0u};                                                   \
        } while (0)
#define MGL_GS_PLAN_BYTES(data, dataLength, bindingIndex)                        \
        do {                                                                     \
            executionPlan.binding_ops[executionPlan.binding_op_count++] =        \
                (MGLRenderCppComputeBindingOp){                                  \
                    1u, (uint32_t)(bindingIndex), 0u, NULL,                      \
                    (data), (uint32_t)(dataLength)};                             \
        } while (0)
        MGL_GS_PLAN_BUFFER(input, inputOffset, MGL_AIR_GS_SLOT_INPUT);
        MGL_GS_PLAN_BUFFER(output, 0u, MGL_AIR_GS_SLOT_OUTPUT);
        MGL_GS_PLAN_BUFFER(counts, 0u, MGL_AIR_GS_SLOT_COUNTS);
        MGL_GS_PLAN_BUFFER(indexedDraw ? gatherBuf : counts, 0u,
                           MGL_AIR_GS_SLOT_GATHER);
        if (xfbCaptureBuffer) {
            MGL_GS_PLAN_BUFFER(xfbCaptureBuffer, xfbDestinationOffset,
                               MGL_AIR_GS_SLOT_XFB);
        }
        MGL_GS_PLAN_BUFFER(xfbMetaBuf, 0u, MGL_AIR_GS_SLOT_XFB_META);
        MGL_GS_PLAN_BYTES(&gparams, sizeof(gparams),
                          MGL_AIR_GS_SLOT_GATHER_PARAMS);
#undef MGL_GS_PLAN_BYTES
#undef MGL_GS_PLAN_BUFFER
    } else {
        compute = mglDrawSupportCreateComputeEncoder(
            _renderPassManager.state->currentCommandBufferOwner);
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
        /* XFB slots (mgl_air_gs_abi.h §5): the meta record is always bound
         * (stride 0 disables capture in the kernel); the stream is bound only
         * when capture is active. */
        if (xfbCaptureBuffer) {
            mglDrawSupportSetComputeBuffer(compute, xfbCaptureBuffer,
                                           xfbDestinationOffset,
                                           MGL_AIR_GS_SLOT_XFB);
        }
        mglDrawSupportSetComputeBuffer(compute, xfbMetaBuf, 0u,
                                       MGL_AIR_GS_SLOT_XFB_META);
    }
    bool buffersOK = [self bindBuffersToComputeEncoder:compute
                                                   stage:_GEOMETRY_SHADER
                                               copyBacks:&stageCopyBacks
                                           executionPlan:cppDispatch ? &executionPlan : NULL
                                            temporaries:executionTemporaries];
    bool texturesOK = buffersOK && [self bindTexturesToComputeEncoder:compute
                                                                stage:_GEOMETRY_SHADER
                                                        executionPlan:cppDispatch ? &executionPlan : NULL
                                                         temporaries:executionTemporaries];
    if (!buffersOK || !texturesOK) {
        if (compute) mglDrawSupportEndComputeEncoder(compute);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    if (cppDispatch) {
        MGLRenderCppCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = 0u;
        for (NSUInteger slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            MGLStageBindingCopyBack *entry = &stageCopyBacks.slots[slot];
            if (entry->length == 0) continue;
            copyBackEntries[copyBackEntryCount++] =
                (MGLRenderCppCopyBackEntry){
                    .temporary = (__bridge void *)entry->temporary,
                    .destination = (__bridge void *)entry->destination,
                    .destination_buffer = entry->destination_buffer,
                    .destination_offset = entry->destination_offset,
                    .length = entry->length,
                };
        }
        executionPlan.dispatch = (MGLRenderCppComputePlan){
            .dispatch_kind = MGL_RENDER_CPP_COMPUTE_DISPATCH_DIRECT,
            .groups_x = (uint32_t)workItemCount,
            .groups_y = 1u,
            .groups_z = 1u,
            .local_x = 1u,
            .local_y = 1u,
            .local_z = 1u,
        };
        executionPlan.barrier_scope = copyBackEntryCount
            ? MGL_RENDER_CPP_COMPUTE_BARRIER_BUFFERS
            : MGL_RENDER_CPP_COMPUTE_BARRIER_NONE;
        const BOOL requireCPUVisibility =
            xfbActive || mglHasActiveIndexedPrimitiveQuery();
        MGLRenderCppComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderCppExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount,
                requireCPUVisibility ? 1u : 0u, &executionResult,
                executionError, sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL GS ERROR: C++ execution transaction failed: %s",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
        [self clearStageBindingCopyBacks:&stageCopyBacks];
    } else {
        mglDrawSupportDispatchCompute(
            compute, MTLSizeMake(workItemCount, 1u, 1u),
            MTLSizeMake(1u, 1u, 1u));
        mglDrawSupportEndComputeEncoder(compute);
        if (![self flushStageBindingCopyBacks:&stageCopyBacks
                         requireCPUVisibility:(xfbActive ||
                                               mglHasActiveIndexedPrimitiveQuery())]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }
    _geometry.expansionActive = YES;
    _geometry.program = program;
    /* The passthrough pipeline rasterizes the GS output primitive class, so
     * drive inputPrimitiveTopology from the output mode, not the GL input
     * mode (e.g. points in -> triangle_strip out). */
    switch (outputPrimitive) {
        case MTLPrimitiveTypePoint:
            _lastDrawPrimitiveMode = GL_POINTS;
            break;
        case MTLPrimitiveTypeLine:
            _lastDrawPrimitiveMode = GL_LINES;
            break;
        default:
            _lastDrawPrimitiveMode = GL_TRIANGLES;
            break;
    }
    drawCtx->state.dirty_bits = DIRTY_ALL;

    GLuint64 queryGenerated =
        outputPrimitive == MTLPrimitiveTypePoint
            ? (GLuint64)workItemCount * expandedVertices
            : (GLuint64)workItemCount * expandedVertices /
                  (outputPrimitive == MTLPrimitiveTypeLine ? 2u : 3u);
    const GLuint64 vpp = outputPrimitive == MTLPrimitiveTypePoint
        ? 1u
        : (outputPrimitive == MTLPrimitiveTypeLine ? 2u : 3u);
    GLuint64 queryWritten = 0u;
    const MGLAIRGSXFBMeta *queryMeta = NULL;
    if (xfbActive && xfbMetaBuf && xfbMetaBuf.contents) {
        /* The stage synchronization above made the atomic counters CPU
         * visible; the written counter counts exactly the bytes the
         * kernel stored, so culled primitives are excluded.
         *
         * Multi-stream (GL 4.6 §11.1.3.4): each stream's segment lives at
         * streamPhysBase[s] in the temporary; copy each back to its GL XFB
         * destination, rounded down to a whole-primitive prefix so partial
         * records don't corrupt the GL-visible store.  Streams 1..3 are
         * XFB-only and don't contribute to PRIMITIVES_GENERATED /
         * TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN (stream 0 primitives only,
         * GL 4.6 §13.2.4). */
        const MGLAIRGSXFBMeta *meta =
            (const MGLAIRGSXFBMeta *)xfbMetaBuf.contents;
        queryMeta = meta;
        NSUInteger writtenBytesStream0 = (NSUInteger)meta->stream[0].written;
        if (xfbTemporary) {
            /* Slow path (single- or multi-stream): the kernel captured into
             * a temporary; copy each stream's segment back to its GL XFB
             * buffer.  The fast path (direct bind, no temporary) skips this
             * and only advances the write offset below. */
            MGLMetalBlitCommandEncoderRef xfbBlit = nil;
            for (uint32_t s = 0u; s < gsStreamCount; s++) {
                if (!streamDstMTL[s] || streamStride[s] == 0u) continue;
                NSUInteger w = (NSUInteger)meta->stream[s].written;
                if (w == 0u) continue;
                NSUInteger pbytes = (s == 0u)
                    ? (NSUInteger)vpp * outputStride
                    : streamStride[s];  /* streams > 0 are points-only (vpp=1) */
                NSUInteger copyBytes = pbytes > 0u
                    ? (w / pbytes) * pbytes : 0u;
                if (streamRemaining[s] < copyBytes) {
                    copyBytes = streamRemaining[s] > pbytes
                        ? (streamRemaining[s] / pbytes) * pbytes
                        : 0u;
                }
                if (copyBytes == 0u) continue;
                if (!xfbBlit) {
                    xfbBlit = mglDrawSupportCreateBlitEncoder(
                        _renderPassManager.state->currentCommandBufferOwner);
                    if (!xfbBlit) {
                        _geometry.expansionActive = NO;
                        _geometry.program = NULL;
                        drawCtx->state.dirty_bits = DIRTY_ALL;
                        return YES;
                    }
                }
                mglDrawSupportBlitCopyBuffer(xfbBlit, xfbTemporary,
                                             streamPhysBase[s],
                                             streamDstMTL[s],
                                             streamDstOffset[s], copyBytes);
                BufferBaseTarget *slot = &MGL_STATE(drawCtx)
                    ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[s];
                if (slot->buf) slot->buf->ever_written = GL_TRUE;
                const GLuint64 currentOffset =
                    xfbState->buffer_write_offsets[s];
                xfbState->buffer_write_offsets[s] =
                    (GLuint64)copyBytes > UINT64_MAX - currentOffset
                        ? UINT64_MAX
                        : currentOffset + (GLuint64)copyBytes;
                if (s == 0u) writtenBytesStream0 = copyBytes;
            }
            if (xfbBlit) mglDrawSupportEndBlitEncoder(xfbBlit);
        } else if (writtenBytesStream0 > 0u) {
            /* Fast path (single-stream direct bind): the kernel wrote
             * straight into the GL XFB buffer; just advance the offset. */
            const GLuint64 currentOffset =
                xfbState->buffer_write_offsets[0];
            xfbState->buffer_write_offsets[0] =
                (GLuint64)writtenBytesStream0 > UINT64_MAX - currentOffset
                    ? UINT64_MAX
                    : currentOffset + (GLuint64)writtenBytesStream0;
        }
        queryWritten = (outputStride > 0u && vpp > 0u)
            ? (GLuint64)writtenBytesStream0 /
                  ((GLuint64)outputStride * vpp) : 0u;
        /* A non-indexed primitive query addresses stream 0.  Indexed query
         * results for streams 1..3 are recorded from each stream's counters
         * below, after the common stream-0 result is finalized. */
    }
    if (!queryMeta && xfbMetaBuf && xfbMetaBuf.contents &&
        mglHasActiveIndexedPrimitiveQuery()) {
        queryMeta = (const MGLAIRGSXFBMeta *)xfbMetaBuf.contents;
    }
    /* Multi-stream (GL 4.6 §13.2.4.1): PRIMITIVES_GENERATED counts only
     * stream 0 primitives (emitted to the rasterizer).  The static
     * estimate above includes all streams' expanded vertices; replace
     * it with the actual stream 0 visible count from the counts buffer. */
    if (multiStream && counts && counts.contents) {
        const uint32_t *cw = (const uint32_t *)counts.contents;
        GLuint64 stream0Visible = 0u;
        for (GLuint w = 0u; w < workItemCount; w++) {
            stream0Visible += (GLuint64)cw[
                w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 0];
        }
        queryGenerated = (vpp > 0u) ? stream0Visible / vpp : 0u;
    }
    if (xfbActive && MGL_STATE(drawCtx)->caps.rasterizer_discard) {
        /* GL_RASTERIZER_DISCARD: no pixels by definition; the compute
         * expansion already ran and the primitive query must still count
         * the generated/written primitives (persistent query semantics). */
        _currentCBHasWork = YES;
        mglRecordGeometryPrimitiveQueries(
            drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
            gsStreamCount, streamStride);
        _geometry.expansionActive = NO;
        _geometry.program = NULL;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    if (![self processGLState:true] ||
        mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1 ||
        [self currentDrawRasterizationIsEmpty] ||
        [self currentDrawModeIsFullyCulled:gsOutputMode]) {
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG raster-skip: pgl=%d enc=%d empty=%d cull=%d",
                  [self processGLState:true] ? 1 : 0,
                  mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 1 ? 1 : 0,
                  [self currentDrawRasterizationIsEmpty] ? 1 : 0,
                  [self currentDrawModeIsFullyCulled:gsOutputMode] ? 1 : 0);
        }
        if (xfbActive || mglHasActiveIndexedPrimitiveQuery()) {
            _currentCBHasWork = YES;
            mglRecordGeometryPrimitiveQueries(
                drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
                gsStreamCount, streamStride);
        }
        _geometry.expansionActive = NO;
        _geometry.program = NULL;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    [self applyPolygonOffsetForDrawMode:gsOutputMode];
    if (getenv("MGL_GS_DIAG")) {
        const uint32_t *cw = (const uint32_t *)counts.contents;
        const float *ow = (const float *)output.contents;
        NSLog(@"MGL GS DIAG draw loop: workItemCount=%u outputStride=%lu "
              "recordsPerPrimitive=%lu outputPrimitive=%d",
              (unsigned)workItemCount, (unsigned long)outputStride,
              (unsigned long)recordsPerPrimitive, (int)outputPrimitive);
        for (GLuint p = 0u; p < workItemCount && p < 4u; p++) {
            NSUInteger off = ((NSUInteger)p * recordsPerPrimitive +
                              MGL_AIR_GS_HEADER_RECORDS) * outputStride;
            const float *pos = (const float *)(
                (const uint8_t *)output.contents + off);
            NSLog(@"MGL GS DIAG prim=%u counts(vertex=%u inst=%u baseV=%u "
                  "baseI=%u) pos[0]=(%g,%g,%g,%g)",
                  (unsigned)p, cw[p * MGL_AIR_GS_COUNTS_RECORD_WORDS + 0],
                  cw[p * MGL_AIR_GS_COUNTS_RECORD_WORDS + 1],
                  cw[p * MGL_AIR_GS_COUNTS_RECORD_WORDS + 2],
                  cw[p * MGL_AIR_GS_COUNTS_RECORD_WORDS + 3],
                  pos[0], pos[1], pos[2], pos[3]);
        }
        (void)ow;
    }
    for (GLuint primitive = 0u; primitive < workItemCount; primitive++) {
        NSUInteger offset =
            ((NSUInteger)primitive * recordsPerPrimitive +
             MGL_AIR_GS_HEADER_RECORDS) * outputStride;
        mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, output, offset, 0u);
        mglDrawSupportDrawPrimitivesIndirect(
            _renderPassManager.state->currentRenderEncoderOwner, outputPrimitive, counts,
            (NSUInteger)primitive * countsRecordBytes);
    }
    _currentCBHasWork = YES;
    mglRecordGeometryPrimitiveQueries(
        drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
        gsStreamCount, streamStride);
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

        MGLMetalBufferRef mtlBuffer = (__bridge MGLMetalBufferRef)(vbo->data.mtl_data);
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
                          mtlBuffer:(MGLMetalBufferRef *)mtlBufferOut
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
                              mtlBuffer:(MGLMetalBufferRef *)mtlBufferOut
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
                   mtlBuffer:(MGLMetalBufferRef *)mtlBufferOut
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

    MGLMetalBufferRef indexBuffer = (__bridge MGLMetalBufferRef)(gl_element_buffer->data.mtl_data);
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
                           mtlBuffer:(MGLMetalBufferRef *)mtlBufferOut
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

    MGLMetalBufferRef indirectBuffer = (__bridge MGLMetalBufferRef)(gl_indirect_buffer->data.mtl_data);
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
    if (mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
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
    mglRenderPassRenderTargetSizeForState(
        _renderPassManager.state->renderPassStateOwner,
        &passWidth, &passHeight);
    if (passWidth == 0 || passHeight == 0) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            MGLMetalTextureRef color = mglRenderPassAttachmentTextureForState(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i);
            if (color) {
                passWidth = color.width;
                passHeight = color.height;
                break;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            MGLMetalTextureRef depth = mglRenderPassAttachmentTextureForState(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            if (depth) {
                passWidth = depth.width;
                passHeight = depth.height;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            MGLMetalTextureRef stencil = mglRenderPassAttachmentTextureForState(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            if (stencil) {
                passWidth = stencil.width;
                passHeight = stencil.height;
            }
        }
    }

    /* P4.5 (item 1141/887): viewport/scissor/framebuffer 交集判定（纯 CPU
     * 数学，逐点等价）在 C++（mglRenderCppRasterizationIsEmpty，两门共用）。 */
    return mglRenderCppRasterizationIsEmpty(
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
    if (mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return;
    }

    /* P4.5 (item 1141/887): 三角填充模式（GL_LINE -> lines）+ 非法
     * polygon_mode 修复条件 + 按 polygon 模式的 depth-bias 使能判定在 C++
     * （mglRenderCppPolygonOffsetDecision，纯决策，两门共用）。 */
    MGLRenderCppPolygonOffsetDecision decision = {0};
    mglRenderCppPolygonOffsetDecision(
        (uint32_t)mode,
        ctx ? 1 : 0,
        mglDrawModeProducesPolygons(mode) ? 1 : 0,
        (uint32_t)(ctx ? MGL_STATE(ctx)->var.polygon_mode : 0u),
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_point) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_line) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_fill) ? 1 : 0,
        &decision);
    MTLTriangleFillMode triangleFillMode = decision.triangle_fill_mode
        ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
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
        mglRenderCppBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    } else {
        mglRenderCppBindingSetDepthBiasIfNeededForOwner(
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

    /* P4.5 (item 1141/887): 绘制模式 -> 图元顶点数表在 C++
     * （mglRenderCppPrimitiveVertexCountForMode，两门共用）。 */
    uint32_t prim_vertex_count =
        mglRenderCppPrimitiveVertexCountForMode((uint32_t)mode);

    MGLMetalBufferRef captureBuffer = (__bridge MGLMetalBufferRef)
        mglRendererBackendGetCullDistanceCaptureBuffer(_backend);
    if (captureBuffer) {
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
            encCtx->render_encoder_owner, captureBuffer, 0u,
            kMGLCullDistanceVertexBufferIndex);
        [self recordLastBoundVertexBuffer:
                  captureBuffer
                                   offset:0
                                  atIndex:kMGLCullDistanceVertexBufferIndex];
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        mglDrawSupportSetVertexBytes(
            encCtx->render_encoder_owner, &params, sizeof(params),
            kMGLCullDistanceParamsBufferIndex);
        [self invalidateLastBoundVertexBufferAtIndex:
                  kMGLCullDistanceParamsBufferIndex];
        return;
    }

    /* Scan enabled attributes for cull distance entries. The GLSL source
     * uses "culldistance_data" as the attribute name. We identify them
     * via the shader resource list (which preserves the name) or
     * by checking the MSL source for [[attribute(N)]] with that name. */
    MGLMetalBufferRef cullMtlBuffer = nil;
    GLintptr cullBindingOffset = 0;
    GLuint cullStride = 0;
    GLuint cullDistSize = 0;
    GLintptr cullFirstRelativeOffset = -1;

    MGLShaderResourceList *vsInputs =
        &activeProgram->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];

    for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }
        /* Find the resource name for this attribute. */
        const char *attrName = NULL;
        if (vsInputs && vsInputs->list) {
            for (GLuint r = 0; r < vsInputs->count; r++) {
                MGLShaderResource *res = &vsInputs->list[r];
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
            cullMtlBuffer = (__bridge MGLMetalBufferRef)resolved.buffer->data.mtl_data;
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
        static MGLMetalBufferRef sDummyCullBuffer = nil;
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
        encCtx->render_encoder_owner, cullMtlBuffer, 0,
        kMGLCullDistanceVertexBufferIndex);
    [self recordLastBoundVertexBuffer:cullMtlBuffer
                               offset:0
                              atIndex:kMGLCullDistanceVertexBufferIndex];
    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    mglDrawSupportSetVertexBytes(
        encCtx->render_encoder_owner, &params, sizeof(params),
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
        tesProgram->modules[_TESS_EVALUATION_SHADER].metallib_bytes != NULL;
    BOOL nativeTES = mglNativeTESInterfaceSupported(tcsProgram, tesProgram);

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
              &vertexProgram->shader_resources_list[_VERTEX_SHADER]
                                                   [_STAGE_OUTPUT_RES])
        : MGL_AIR_PER_VERTEX_STRIDE;
    contract.patch_out_stride = 16u; /* refined by the TCS dispatcher */

    _tessellation.tessVertexCaptureBuffer = nil;
    _tessellation.tessVertexCaptureOffset = 0u;
    _tessellation.tessControlPointIndexBuffer = nil;
    _tessellation.tessIndexedDraw = NO;
    _tessellation.tessInstanceRecords = 0u;
    /* A TCS from a previous draw must not leak into a TES-only dispatch
     * (dispatchAIRTessEvalCompute reads tcsOutputBuffer as the gl_in
     * source when non-nil).  The TCS dispatcher re-populates it. */
    _tessellation.tcsOutputBuffer = nil;
    _tessellation.tcsOutputOffset = 0u;
    _tessellation.tcsOutputStride = 0u;
    _tessellation.tcsOutVertices = 0u;
    _tessellation.tessFactorBuffer = nil;
    /* The VS position capture and the default factor buffer are consumed by
     * both the native patch pipeline and the AIR TES compute expansion
     * (isolines / point_mode with no TCS), so they must exist even when
     * nativeTES is unavailable. */
    if (nativeTES || airTES) {
        const BOOL indexedDraw = (indexType != 0u);
        if (indexedDraw && tcsProgram) {
            /* TCS consumes the VS capture with continuous per-patch
             * addressing; indexed (sparse) input needs a gather path in the
             * TCS kernel, which is a separate follow-up. */
            nativeTES = NO;
        } else if (indexedDraw) {
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
                MGLMetalBufferRef eboMetal =
                    (__bridge MGLMetalBufferRef)ebo->data.mtl_data;
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
                    MGLMetalBufferRef gatherBuf =
                        mglDrawSupportCreateBufferWithBytes(
                            _device, gatherArray,
                            (NSUInteger)gatherCount * 4u,
                            MTLResourceStorageModeShared);
                    free(gatherArray);
                    if (!gatherBuf) {
                        nativeTES = NO;
                    } else {
                        NSUInteger captureOffset = 0u;
                        MGLMetalBufferRef capture = [self
                            captureAIRVertexPositionsForGeometryIndexed:drawCtx
                                                            indexBuffer:eboMetal
                                                              indexType:getMTLIndexType(indexType)
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
                            _tessellation.tessVertexCaptureBuffer = capture;
                            _tessellation.tessVertexCaptureOffset = captureOffset;
                            _tessellation.tessControlPointIndexBuffer = gatherBuf;
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
        } else {
            NSUInteger captureOffset = 0u;
            MGLMetalBufferRef capture =
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
                _tessellation.tessInstanceRecords = (NSUInteger)count;
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
        _tessellation.tessFactorBuffer = mglCachedDefaultTessFactorBuffer(
            _device, _backend, MGL_STATE(drawCtx), patchCount);
        if (!_tessellation.tcsOutputBuffer ||
            !_tessellation.tessFactorBuffer) {
            nativeTES = NO;
        }
    }

    if (airTES && !tcsProgram) {
        /* TES-only compute expansion also needs the default levels; the
         * cached buffer is rebuilt only when glPatchParameterfv levels
         * (or the patch count) change between draws. */
        _tessellation.tessFactorBuffer = mglCachedDefaultTessFactorBuffer(
            _device, _backend, MGL_STATE(drawCtx), patchCount);
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
        MGLMetalBufferRef nativeFactors = mglNativeTessFactorBuffer(
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

        _tessellation.nativeTESProgram = tesProgram;
        _tessellation.nativeTESActive = YES;
        [self clearStageBindingCopyBacks:&_tessellation.nativeTESCopyBacks];
        drawCtx->state.dirty_bits = DIRTY_ALL;

        BOOL stateReady = [self processGLState:true];
        if (!stateReady || mglRenderCppRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
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
            /* Metal does not advance the post-tessellation control-point
             * pointer correctly for patchStart. Draw each patch separately:
             * slot 0 is rebased to the patch, while slot 30 stays at the
             * instance base so TES varyings can apply patchId exactly once. */
            const NSUInteger instanceRecords =
                _tessellation.tessInstanceRecords;
            const NSUInteger instanceStrideBytes =
                instanceRecords * _tessellation.tcsOutputStride;
            mglDrawSupportSetTessellationFactors(
                _renderPassManager.state->currentRenderEncoderOwner, nativeFactors, 0u, 0u);
            for (GLsizei i = 0; i < instanceCount; i++) {
                const NSUInteger instanceOffset =
                    _tessellation.tessVertexCaptureOffset +
                    (NSUInteger)i * instanceStrideBytes;
                mglDrawSupportSetVertexBuffer(
                    _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutputBuffer,
                    instanceOffset, 0u);
                [self recordLastBoundVertexBuffer:_tessellation.tcsOutputBuffer
                                           offset:instanceOffset
                                          atIndex:0u];
                mglDrawSupportSetVertexBuffer(
                    _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutputBuffer,
                    instanceOffset, 30u);
                [self recordLastBoundVertexBuffer:_tessellation.tcsOutputBuffer
                                           offset:instanceOffset
                                          atIndex:30u];
                GLuint patchInfo[2] = {patchVertices, _tessellation.tcsOutVertices};
                if (patchInfo[1] == 0u) patchInfo[1] = patchVertices;
                mglDrawSupportSetVertexBytes(
                    _renderPassManager.state->currentRenderEncoderOwner, patchInfo, sizeof(patchInfo), 28u);
                if (_tessellation.tcsPatchOutBuffer) {
                    mglDrawSupportSetVertexBuffer(
                        _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsPatchOutBuffer, 0u, 27u);
                    [self recordLastBoundVertexBuffer:
                              _tessellation.tcsPatchOutBuffer
                                               offset:0u
                                              atIndex:27u];
                }
                if (_tessellation.tessIndexedDraw) {
                    mglDrawSupportDrawIndexedPatches(
                        _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutVertices, 0u, patchCount,
                        nil, 0u,
                        _tessellation.tessControlPointIndexBuffer, 0u,
                        1u, (NSUInteger)baseInstance + (NSUInteger)i);
                } else {
                    const NSUInteger cpcStride =
                        (NSUInteger)_tessellation.tcsOutVertices *
                        _tessellation.tcsOutputStride;
                    for (GLuint p = 0u; p < patchCount; p++) {
                        const NSUInteger patchOffset =
                            instanceOffset + (NSUInteger)p * cpcStride;
                        mglDrawSupportSetVertexBuffer(
                            _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutputBuffer,
                            patchOffset, 0u);
                        [self recordLastBoundVertexBuffer:
                                  _tessellation.tcsOutputBuffer
                                                   offset:patchOffset
                                                  atIndex:0u];
                        mglDrawSupportDrawPatches(
                            _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutVertices, p, 1u,
                            nil, 0u, 1u,
                            (NSUInteger)baseInstance + (NSUInteger)i);
                    }
                }
            }
            _currentCBHasWork = YES;

            GLuint64 primitives = mglNativeTessPrimitiveCount(
                _tessellation.tessFactorBuffer, tesProgram, patchCount,
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
        _tessellation.tessVertexCaptureBuffer = nil;
        _tessellation.tessVertexCaptureOffset = 0u;
        _tessellation.tessControlPointIndexBuffer = nil;
        _tessellation.tessIndexedDraw = NO;
        _tessellation.tessInstanceRecords = 0u;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    if (airTES) {
        /* Isolines and layout(point_mode) have no Metal-native equivalent
         * (MTLPatchType is triangle/quad only and patch draws have no output
         * primitive type), so those programs run as an AIR compute kernel
         * expansion + passthrough vertex (line/point rasterization). */
        if (tesProgram && (tesProgram->tess_gen_point_mode ||
                           tesProgram->tess_gen_mode == GL_ISOLINES)) {
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
            drawCtx->state.dirty_bits = DIRTY_ALL;
            _tessellation.tessVertexCaptureBuffer = nil;
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }
        NSLog(@"MGL TESS ERROR: native AIR TES interface unsupported for program %u",
              (unsigned)tesProgram->name);
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
