/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+DrawStageHost.m
// O1.4/O1.6: capture/cull/validate ports + GS Metal expansion residual +
// HostOps fillers. Tess/XFB/GS early orchestration lives in mgl_draw_*.cpp.

#import "MGLRenderer_Private.h"
#import <CoreFoundation/CoreFoundation.h>
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+Tessellation_Private.h"
#import "MGLRenderer+DrawSupportUtil.h"
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

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

@implementation MGLRenderer (DrawStageHost)

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
    const uint32_t elemWidth = mglRenderGLIndexElementSize((uint64_t)indexType);
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
    /* O1.2: double processGLState+bind session in mglTessRunCaptureSession;
     * ObjC only supplies host ports (processGLState / encoder / MTL bind). */
    if (!drawCtx || !capture || !params) {
        return NO;
    }
    self->ctx = drawCtx;
    MGLTessCaptureSessionHostOps ops = {
        .ctx = drawCtx,
        .renderer = (__bridge void *)self,
        .mark_dirty_all = mglDrawSupportCaptureMarkDirtyAll,
        .process_gl_state = mglDrawSupportCaptureProcessGL,
        .encoder_has_current = mglDrawSupportCaptureEncoderReady,
        .bind_capture_slots = mglDrawSupportCaptureBindSlots,
        .set_capture_active = mglDrawSupportCaptureSetActive,
    };
    return mglTessRunCaptureSession((__bridge void *)capture, params, &ops)
               ? YES
               : NO;
}

- (id)captureAIRVertexPositionsForTessellation:(GLMContext)drawCtx
                                                    first:(GLint)first
                                                    count:(GLsizei)count
                                            instanceCount:(GLsizei)instanceCount
                                             baseInstance:(GLuint)baseInstance
                                               outOffset:(NSUInteger *)outOffset
{
    if (outOffset) *outOffset = 0u;
    if (!drawCtx ||
        !mglTessArrayCaptureInputsOk(first, count, instanceCount)) {
        return nil;
    }

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
    /* O1.2: restart sanitize + Metal index prep planned in
     * mglTessPlanIndexedCaptureIndexPrep; ObjC only allocates/copies. */
    id sanitizedIndexBuffer = indexBuffer;
    NSUInteger sanitizedIndexOffset = indexOffset;
    uint32_t restartIndex = 0u;
    const bool restartEnabled =
        mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    MGLTessIndexedCaptureIndexPrep prep = {0};
    const int contentsReadable =
        mglDrawSupportBufferContents(indexBuffer) ? 1 : 0;
    if (!mglTessPlanIndexedCaptureIndexPrep(
            (uint32_t)indexType, (uint64_t)indexOffset, (uint32_t)count,
            (uint64_t)mglDrawSupportBufferLength(indexBuffer), contentsReadable,
            restartEnabled ? 1 : 0, restartIndex, &prep)) {
        _tessellation.tessVertexCaptureActive = NO;
        return nil;
    }
    if (prep.need_sanitize) {
        uint8_t *copy = malloc((size_t)prep.stream_bytes);
        if (copy) {
            if (mglTessSanitizeRestartIndices(
                    copy,
                    (const uint8_t *)mglDrawSupportBufferContents(indexBuffer) +
                        indexOffset,
                    (uint32_t)count, (GLenum)indexType, prep.restart_index)) {
                id clean = mglDrawSupportCreateBufferWithBytes(
                    _device, copy, (NSUInteger)prep.stream_bytes, 0u);
                if (clean) {
                    sanitizedIndexBuffer = clean;
                    sanitizedIndexOffset = 0u;
                }
            }
            free(copy);
        }
    }
    id drawIndexBuffer = sanitizedIndexBuffer;
    NSUInteger drawIndexOffset = sanitizedIndexOffset;
    uint64_t mtlIndexType = (uint64_t)prep.mtl_index_type;
    if (prep.need_metal_index_prep) {
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


/* O1.4 residual: GS Metal expansion after C++ topology/gather/capture. */
int mglDrawHostGsExecuteMetalExpansion(
    void *renderer, GLMContext drawCtx, GLenum mode, GLint first, GLsizei count,
    GLenum indexType, const void *indices, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, const char *label,
    Program *program, GLenum gsInputMode, GLenum gsOutputMode,
    uint32_t outputPrimitive, int indexedDraw, void *gatherBufPtr,
    const void *gparamsPtr, uint32_t gparamsBytes,
    const MGLGsComputeLayout *gsLayoutPtr, void *inputPtr,
    uint64_t inputOffsetIn, Program *captureVS, Program *captureTES,
    uint32_t pendingStride)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    if (!self || !drawCtx || !program || !gsLayoutPtr || !gparamsPtr) {
        return 1;
    }
    (void)mode; (void)first; (void)count; (void)indexType; (void)indices;
    (void)baseVertex; (void)baseInstance; (void)gsInputMode;
    (void)indexedDraw; (void)pendingStride; (void)gparamsBytes;
    self->ctx = drawCtx;
    id gatherBuf = (__bridge id)gatherBufPtr;
    MGLAIRGSGatherParams gparams;
    memcpy(&gparams, gparamsPtr, sizeof(gparams));
    MGLGsComputeLayout gsLayout = *gsLayoutPtr;
    id input = (__bridge id)inputPtr;
    NSUInteger inputOffset = (NSUInteger)inputOffsetIn;
    const GLuint workItemCount = gsLayout.work_item_count;
    const NSUInteger outputStride = gsLayout.output_stride;
    const NSUInteger expandedVertices = gsLayout.expanded_vertices;
    const NSUInteger recordsPerPrimitive = gsLayout.records_per_primitive;
    const uint32_t maxVertices =
        mglDrawGsMaxVerticesOut(program->geometry_vertices_out);
    const GLuint primitiveCount = gsLayout.work_item_count / (uint32_t)(instanceCount > 0 ? instanceCount : 1);
    (void)primitiveCount;
    (void)captureVS;
    (void)captureTES;

    void *pipelineHandle = NULL;
    char pipelineError[2048] = {0};
    int pipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _GEOMETRY_SHADER, &pipelineHandle,
        pipelineError, sizeof(pipelineError));
    id pipeline =
        mglTessComputePipelineReady(pipelineResult, pipelineHandle ? 1 : 0)
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
        self->_renderPassManager.state->currentCommandBufferOwner, &commandState);
    if (mglTessCommandBufferNeedsNew(hasCommandState, commandState.status)) {
        if (![self newCommandBuffer]) {
            drawCtx->active_state->dirty_bits = DIRTY_ALL;
            return YES;
        }
    }
    const NSUInteger outputSize = (NSUInteger)gsLayout.output_bytes;
    id output = mglDrawSupportCreateBuffer(
        self->_device, outputSize, 0u);
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG outputSize=%lu stride=%lu recordsPerPrim=%lu workItems=%u mtlLen=%@",
              (unsigned long)outputSize, (unsigned long)outputStride,
              (unsigned long)recordsPerPrimitive, (unsigned)workItemCount,
              [output valueForKey:@"length"]);

    const NSUInteger countsRecordBytes = MGL_AIR_GS_COUNTS_RECORD_BYTES;
    id counts = mglDrawSupportCreateBuffer(
        self->_device, (NSUInteger)gsLayout.counts_bytes,
        0u);
    if (!output || !counts || !mglDrawSupportBufferContents(output) || !mglDrawSupportBufferContents(counts)) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         (GLenum)mglRenderErrorOutOfMemory());
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
    const bool gsSeparate = mglXfbSeparateAttribs(
        program->transform_feedback_buffer_mode) != 0;

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
            NSUInteger sessionOffset = (NSUInteger)mglXfbSessionOffsetOr(
                (uint64_t)xfbState->buffer_write_offsets[b], 0u);
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
            xfbTemporary = mglDrawSupportCreateBuffer(self->_device, physTotal, 0u);
            if (xfbTemporary) {
                memset(mglDrawSupportBufferContents(xfbTemporary), 0,
                       physTotal);
                xfbCaptureBuffer = xfbTemporary;
            }
            const NSUInteger visBytes =
                (NSUInteger)mglDrawGsXFBVisBytes((uint32_t)workItemCount);
            xfbVisBuffer = mglDrawSupportCreateBuffer(self->_device, visBytes, 0u);
            xfbOffsetBuffer = mglDrawSupportCreateBuffer(self->_device, visBytes, 0u);
            xfbWrittenBuffer =
                mglDrawSupportCreateBuffer(self->_device, visBytes, 0u);
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
                                 (GLenum)mglRenderErrorOutOfMemory());
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
        self->_device, &xfbMeta, sizeof(xfbMeta), 0u);
    if (!xfbMetaBuf) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         (GLenum)mglRenderErrorOutOfMemory());
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
                         (GLenum)mglRenderErrorOutOfMemory());
        return YES;
    }
    if (getenv("MGL_GS_DIAG")) {
        Program *gp = mglResolveProgramForStageFromState(drawCtx, _GEOMETRY_SHADER);
        NSLog(@"MGL GS DIAG GS uniform-constant resources: %u",
              gp ? gp->shader_resources_list[_GEOMETRY_SHADER][_UNIFORM_CONSTANT_RES].count : 0u);
    }
    if (getenv("MGL_GPU_CAPTURE")) {
        id desc = [self mglCaptureDescriptorForDevice:self->_device
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
                self->_renderPassManager.state->currentCommandBufferOwner,
                self->_gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount,
                (requireCPUVisibility || gsDiagnostic) ? 1u : 0u, &executionResult,
                executionError, sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&self->_deviceResetRequested, true,
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
    self->_geometry.expansionActive = YES;
    self->_geometry.program = program;
    /* The passthrough pipeline rasterizes the GS output primitive class, so
     * drive inputPrimitiveTopology from the output mode, not the GL input
     * mode (e.g. points in -> triangle_strip out). */
    self->_lastDrawPrimitiveMode = mglDrawGsLastDrawMode(outputPrimitive);
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
                             (GLenum)mglRenderErrorOutOfMemory());
            return YES;
        }
        MGLRenderComputeExecutionResult scatterResult = {0};
        char scatterError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                self->_renderPassManager.state->currentCommandBufferOwner,
                self->_gpuRecovery.commandRecoveryOwner,
                &scatterPlan, NULL, 0u, 1u, &scatterResult,
                scatterError, sizeof(scatterError)) != 0) {
            if (scatterResult.transaction.device_reset_requested) {
                atomic_store_explicit(&self->_deviceResetRequested, true,
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
                if (!mglDrawGsXFBCopyReady(
                        bufferDstMTL[b] ? 1 : 0,
                        scatterParams.buffers[b].stride,
                        (uint64_t)bufferWritten[b])) {
                    continue;
                }
                NSUInteger copyBytes = (NSUInteger)mglDrawGsClampXFBCopy(
                    (uint64_t)bufferWritten[b],
                    (uint64_t)bufferRemaining[b]);
                if (copyBytes == 0u) continue;
                if (!xfbBlit) {
                    xfbBlit = mglDrawSupportCreateBlitEncoder(
                        self->_renderPassManager.state->currentCommandBufferOwner);
                    if (!xfbBlit) {
                        self->_geometry.expansionActive = NO;
                        self->_geometry.program = NULL;
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
                    slot->buf->ever_written = (GLboolean)mglRenderGLBoolean(1);
                    if (xfbTempBytes &&
                        mglXfbCPUShadowFits(
                            slot->buf->data.buffer_data ? 1 : 0,
                            slot->buf->size, (uint64_t)bufferDstOffset[b],
                            (uint64_t)copyBytes)) {
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
        self->_currentCBHasWork = YES;
        mglRecordGeometryPrimitiveQueries(
            drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
            gsStreamCount, bufferWritten, bufferStride, workItemCount);
        self->_geometry.expansionActive = NO;
        self->_geometry.program = NULL;
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
        return YES;
    }
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG rasterize-check empty=%d culled=%d enc=%d",
              (int)[self currentDrawRasterizationIsEmpty],
              (int)[self currentDrawModeIsFullyCulled:gsOutputMode],
              (int)mglRenderEncoderOwnerHasCurrent(
                  self->_renderPassManager.state->currentRenderEncoderOwner));
    if (!mglDrawGsPassthroughRasterReady(
            [self processGLState:true] ? 1 : 0,
            mglRenderEncoderOwnerHasCurrent(
                self->_renderPassManager.state->currentRenderEncoderOwner),
            [self currentDrawRasterizationIsEmpty] ? 1 : 0,
            [self currentDrawModeIsFullyCulled:gsOutputMode] ? 1 : 0)) {
        if (xfbActive || mglHasActiveIndexedPrimitiveQuery(drawCtx) ||
            mglHasActivePrimitiveQuery(drawCtx) ||
            mglHasActiveGeometryShaderQuery(drawCtx)) {
            self->_currentCBHasWork = YES;
            mglRecordGeometryPrimitiveQueries(
                drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
                gsStreamCount, bufferWritten, bufferStride, workItemCount);
        }
        self->_geometry.expansionActive = NO;
        self->_geometry.program = NULL;
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
            mglRenderBindingClearFragmentBuffer(self->_bindingStateOwner, slot);
        const uint32_t texSlots = (uint32_t)TEXTURE_UNITS;
        for (uint32_t slot = 0u; slot < texSlots; slot++)
            mglRenderBindingClearFragmentTexture(self->_bindingStateOwner, slot);
        MGLEncodeContext gsEncCtx = {
            .render_encoder_owner =
                self->_renderPassManager.state->currentRenderEncoderOwner,
        };
        [self bindFragmentBuffersToCurrentRenderEncoder:&gsEncCtx];
        [self bindBufferSizeConstantsForRenderEncoder];
        Program *gsVertexProgram = mglResolveProgramForStageFromState(
            drawCtx, _VERTEX_SHADER);
        Program *gsFragmentProgram = mglResolveProgramForStageFromState(
            drawCtx, _FRAGMENT_SHADER);
        if (![self bindStorageImagesForVertexProgram:gsVertexProgram
                                     fragmentProgram:gsFragmentProgram]) {
            self->_geometry.expansionActive = NO;
            self->_geometry.program = NULL;
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
            self->_renderPassManager.state->currentRenderEncoderOwner, output,
            MGL_AIR_GS_HEADER_RECORDS * outputStride, 0u);
        mglDrawSupportDrawPrimitives(
            self->_renderPassManager.state->currentRenderEncoderOwner,
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
                self->_device, src + srcOff, bytes, 0u);
            const uint32_t *cwv = (const uint32_t *)mglDrawSupportBufferContents(counts);
            mglDrawSupportSetVertexBuffer(
                self->_renderPassManager.state->currentRenderEncoderOwner, sub, 0u, 0u);
            mglDrawSupportDrawPrimitives(
                self->_renderPassManager.state->currentRenderEncoderOwner,
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
                self->_renderPassManager.state->currentRenderEncoderOwner,
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
        mglDrawSupportSetVertexBuffer(self->_renderPassManager.state->currentRenderEncoderOwner, ptvsSource, ptvsOffset, 0u);
        if (getenv("MGL_GS_DIRECT_DRAW")) {
            const uint32_t *cw2 = (const uint32_t *)mglDrawSupportBufferContents(counts);
            mglDrawSupportDrawPrimitives(
                self->_renderPassManager.state->currentRenderEncoderOwner, outputPrimitive,
                0u, cw2 ? cw2[primitive * MGL_AIR_GS_COUNTS_RECORD_WORDS] : 0u,
                1u, 0u);
        } else if (getenv("MGL_GS_DRAW_VCOUNT")) {
            mglDrawSupportDrawPrimitives(
                self->_renderPassManager.state->currentRenderEncoderOwner, outputPrimitive,
                0u, (NSUInteger)atol(getenv("MGL_GS_DRAW_VCOUNT")), 1u, 0u);
        } else
        mglDrawSupportDrawPrimitivesIndirect(
            self->_renderPassManager.state->currentRenderEncoderOwner, outputPrimitive, counts,
            (offOverride ? 0u : (NSUInteger)primitive * countsRecordBytes));
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG pre-draw prim=%u enc=%d",
                  primitive,
                  (int)mglRenderEncoderOwnerHasCurrent(
                      self->_renderPassManager.state->currentRenderEncoderOwner));
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
    self->_currentCBHasWork = YES;
    if (getenv("MGL_GPU_CAPTURE")) {
        [self flushCommandBuffer:YES];
        [self mglStopCapture];
        NSLog(@"MGL GPU capture stopped");
    }
    mglRecordGeometryPrimitiveQueries(
        drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
        gsStreamCount, bufferWritten, bufferStride, workItemCount);
    self->_geometry.expansionActive = NO;
    self->_geometry.program = NULL;
    drawCtx->active_state->dirty_bits = DIRTY_ALL;
    return YES;
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

        if (!mglRenderAttribOffsetsValid(resolved.binding_offset,
                                         resolved.relativeoffset)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=negative_attrib_offset bindingOffset=%lld relativeOffset=%lld",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (long long)resolved.binding_offset,
                  (long long)resolved.relativeoffset);
            return false;
        }

        MGLRenderAttribFetchPlan fetch = {0};
        if (!mglRenderPlanAttribFetch(
                (uint32_t)a->type, (uint32_t)a->size, resolved.stride,
                resolved.binding_offset, resolved.relativeoffset,
                resolved.divisor, firstVertex, lastVertex, vbo->size,
                &fetch) ||
            fetch.status != MGL_ATTRIB_FETCH_OK) {
            const char *reason = "invalid_attrib_format";
            if (fetch.status == MGL_ATTRIB_FETCH_OVERFLOW) {
                reason = "byte_range_overflow";
            } else if (fetch.status == MGL_ATTRIB_FETCH_OOB) {
                reason = "vbo_oob";
            }
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=%s "
                  "byteRange=[%llu,%llu) stride=%llu elem=%llu type=0x%x size=%u divisor=%u",
                  (unsigned long long)drawCall, (unsigned)attrib,
                  (unsigned)vbo->name, reason,
                  (unsigned long long)fetch.byte_start,
                  (unsigned long long)fetch.byte_end,
                  (unsigned long long)fetch.stride,
                  (unsigned long long)fetch.elem_bytes,
                  (unsigned)a->type, (unsigned)a->size,
                  (unsigned)resolved.divisor);
            return false;
        }
        uint64_t byteStart = fetch.byte_start;
        uint64_t byteEnd = fetch.byte_end;

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
                  (unsigned long long)vbo->size,
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
                  (unsigned long long)vbo->size,
                  (unsigned long long)metalLen,
                  (unsigned long long)fetch.stride,
                  (unsigned long long)resolved.binding_offset,
                  (unsigned long long)resolved.relativeoffset,
                  (unsigned long long)fetch.elem_bytes);
        }
    }

    return true;
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

    /* O1.3: ObjC fills VAO pointer ports; layout (+ dummy) in C++. */
    uint32_t attribs[MAX_ATTRIBS];
    const uint32_t attribCount = mglRenderCollectCullDistanceAttribs(
        activeProgram, attribs, MAX_ATTRIBS);
    MGLRenderCullDistanceAttribPort ports[MAX_ATTRIBS];
    memset(ports, 0, sizeof(ports));
    uint32_t portCount = 0u;
    for (uint32_t i = 0u; i < attribCount && portCount < MAX_ATTRIBS; i++) {
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(
                ctx, vao, attribs[i], "bindCullDistanceEmu", &resolved)) {
            continue;
        }
        if (!resolved.buffer || !resolved.buffer->data.mtl_data) {
            continue;
        }
        ports[portCount].mtl_buffer = resolved.buffer->data.mtl_data;
        ports[portCount].binding_offset = resolved.binding_offset;
        ports[portCount].stride = resolved.stride;
        ports[portCount].relativeoffset = resolved.relativeoffset;
        ports[portCount].valid = 1u;
        portCount++;
    }
    MGLRenderCullDistanceLayout layout;
    mglRenderBuildCullDistanceLayoutFromPorts(
        &layout, ports, portCount,
        mglRendererBackendGetCullDistanceDummyBuffer(_backend));
    void *cullMtlBuffer = layout.mtl_buffer;
    uint32_t cullStride = layout.stride;
    uint32_t cullDistSize = layout.culldist_size;

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


/* ---- O1.4 HostOps ports (thin MTL / renderer ivar materialization) ---- */

static MGLRenderer *mglStageHostSelf(void *renderer)
{
    return renderer ? (__bridge MGLRenderer *)renderer : nil;
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

static void *mglStageCaptureArray(void *renderer, GLMContext ctx, GLint first,
                                  GLsizei count, GLsizei instanceCount,
                                  GLuint baseInstance, uint64_t *out_offset)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    NSUInteger off = 0u;
    id cap = [self captureAIRVertexPositionsForTessellation:ctx
                                                      first:first
                                                      count:count
                                              instanceCount:instanceCount
                                               baseInstance:baseInstance
                                                 outOffset:&off];
    if (out_offset) *out_offset = (uint64_t)off;
    return (__bridge_retained void *)cap;
}

static void *mglStageCaptureIndexed(void *renderer, GLMContext ctx, void *index_mtl,
                                    GLenum indexType, uint64_t index_offset,
                                    GLsizei count, GLint baseVertex,
                                    GLsizei instanceCount, GLuint baseInstance,
                                    uint32_t maxIndex, uint64_t *out_offset)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    NSUInteger off = 0u;
    id cap = [self captureAIRVertexPositionsForGeometryIndexed:ctx
                                                   indexBuffer:(__bridge id)index_mtl
                                                     indexType:indexType
                                                   indexOffset:(NSUInteger)index_offset
                                                         count:count
                                                     baseVertex:baseVertex
                                                  instanceCount:instanceCount
                                                   baseInstance:baseInstance
                                                      maxIndex:maxIndex
                                                     outOffset:&off];
    if (out_offset) *out_offset = (uint64_t)off;
    return (__bridge_retained void *)cap;
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
    id buf = mglDrawSupportCreateBuffer(self->_device, (NSUInteger)length, 0u);
    return (__bridge_retained void *)buf;
}

static void *mglStageCreateBufferBytes(void *renderer, const void *bytes,
                                       uint64_t length)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    id buf = mglDrawSupportCreateBufferWithBytes(self->_device, bytes,
                                                 (NSUInteger)length, 0u);
    return (__bridge_retained void *)buf;
}

static void *mglStageCachedFactors(void *renderer, GLMContext ctx,
                                   uint32_t patch_count)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !ctx) return NULL;
    id buf = mglCachedDefaultTessFactorBuffer(self->_device, self->_backend,
                                              MGL_STATE(ctx), patch_count);
    /* Cached on backend — borrow only. */
    return (__bridge void *)buf;
}

static void *mglStageNativeFactors(void *renderer, void *canonical, GLenum mode,
                                   uint32_t patch_count)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self) return NULL;
    id buf = mglNativeTessFactorBuffer(self->_device, (__bridge id)canonical,
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

static int mglStageDispatchTES(void *renderer, GLMContext ctx, Program *tes,
                               MGLAIRTessDrawContract *contract)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    return self && [self dispatchTessEvaluationShader:ctx program:tes
                                             contract:contract]
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
        .device = (__bridge void *)host->_device,
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
        .dispatch_tes = mglStageDispatchTES,
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
        .execute_metal_expansion = mglDrawHostGsExecuteMetalExpansion,
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
