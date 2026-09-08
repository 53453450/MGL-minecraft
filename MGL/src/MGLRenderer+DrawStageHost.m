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
// O1.4 residual: capture/cull/validate + thin GS Metal HostOps.
// GS Metal expansion orchestration → mgl_draw_gs_metal.cpp.

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


/* GS Metal expansion orchestration → mgl_draw_gs_metal.cpp;
 * ObjC HostOps ports + mglDrawHostGsExecuteMetalExpansion live below
 * (after StageHost thin ports). */

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


/* ---- O1.4 residual: GS Metal expansion HostOps (thin MTL materialization) ---- */

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
        Texture *image = MGL_STATE(ctx)->image_units[unit].tex;
        Texture *sampled = MGL_STATE(ctx)->active_textures[unit];
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
                                         uint32_t *copybacks_count)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !plan || !copybacks || !copybacks_count) return 0;
    (void)ctx;
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
        atomic_store_explicit(&self->_deviceResetRequested, true,
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

static int mglGsMetalRebindFragment(void *renderer, GLMContext ctx)
{
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !ctx) return 0;
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
    id desc = [self mglCaptureDescriptorForDevice:self->_device
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
    MGLRenderer *self = mglStageHostSelf(renderer);
    if (!self || !drawCtx || !program || !gsLayoutPtr || !gparamsPtr) {
        return 1;
    }
    self->ctx = drawCtx;
    MGLGsMetalExpansionHostOps hops = {
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
    return mglDrawGsExecuteMetalExpansion(
        drawCtx, mode, first, count, indexType, indices, baseVertex,
        instanceCount, baseInstance, label, program, gsInputMode, gsOutputMode,
        outputPrimitive, indexedDraw, gatherBufPtr, gparamsPtr, gparamsBytes,
        gsLayoutPtr, inputPtr, inputOffsetIn, captureVS, captureTES,
        pendingStride, &hops);
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
