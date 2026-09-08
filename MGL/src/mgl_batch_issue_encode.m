/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3: MDI/direct/cull/element issue encode split from BatchReplay.m (cluster metric).
 * Same (Draw) category; plans in mgl_batch_issue / mgl_batch_replay.
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"

static BOOL mglBatchReplayHasActiveEncoder(const MGLEncodeContext *encCtx)
{
    if (!encCtx) return NO;
    return mglRenderEncoderOwnerHasCurrent(
        encCtx->render_encoder_owner) != 0;
}

@implementation MGLRenderer (Draw)

- (void)issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                encodeContext:(const MGLEncodeContext *)encCtx
{
    /* O2.5: MDI gate in mgl_batch_replay_mdi_gate (no Metal). */
    size_t argSize = 0;
    size_t neededBytesRaw = 0;
    const int mdiGate = mgl_batch_replay_mdi_gate(
        batch, mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1 : 0, &argSize,
        &neededBytesRaw);
    if (mdiGate != MGL_BATCH_MDI_OK) {
        if (mdiGate == MGL_BATCH_MDI_FALLBACK_EMPTY) {
            return;
        }
        if (mdiGate != MGL_BATCH_MDI_FALLBACK_DISABLED && batch &&
            batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:mgl_batch_replay_mdi_gate_reason(mdiGate)];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    bool indexed = batch->uses_elements;
    NSUInteger neededBytes = (NSUInteger)neededBytesRaw;

    NSUInteger indirectArgsOffset = 0;
    id indirectArgsBuffer =
        [self mdiArgumentScratchBufferWithLength:neededBytes
                                          offset:&indirectArgsOffset];
    if (!indirectArgsBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"mdi_args_alloc"];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }
    void *indirectArgsContents = NULL;
    uint64_t indirectArgsLength = 0;
    if (mglRenderGetBufferContents(
            (__bridge void *)indirectArgsBuffer, &indirectArgsContents,
            &indirectArgsLength) != 0 ||
        !mgl_batch_issue_scratch_range_ok(indirectArgsOffset, neededBytes,
                                          indirectArgsLength)) {
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    uint32_t primType = (uint32_t)batch->key.primitive_type;

    if (indexed) {
        GLenum glIdxType = batch->commands[0].indexType;

        MGLDrawIndexedPrimitivesIndirectArguments *args =
            (MGLDrawIndexedPrimitivesIndirectArguments *)
                ((uint8_t *)indirectArgsContents + indirectArgsOffset);
        if (!mgl_batch_replay_fill_mdi_indexed_args(batch, args)) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"mdi_mixed_index_type"];
            [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
            return;
        }

        for (uint32_t i = 0; i < batch->command_count; i++) {
            MGLDrawCommand *cmd = &batch->commands[i];
            Buffer *glBuf = NULL;
            id idxBuf = nil;
            if (![self resolveElementBufferForCommand:cmd
                                                label:"mdiBatch"
                                              context:glm_ctx
                                             glBuffer:&glBuf
                                            mtlBuffer:&idxBuf]) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"mdi_resolve_element"];
                continue;
            }
            NSUInteger drawIndexOffset = cmd->indexBufferOffset;
            uint64_t drawIndexType = mglIndexTypeForGLType(glIdxType);
            id drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                          glBuf,
                                                                          idxBuf,
                                                                          glIdxType,
                                                                          &drawIndexOffset,
                                                                          &drawIndexType);
            if (!drawIndexBuffer || (GLuint)drawIndexType == 0xFFFFFFFF) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"mdi_prepared_index"];
                continue;
            }
            (void)mgl_batch_mtl_draw_indexed_indirect(
                encCtx->render_encoder_owner, primType,
                (uint32_t)drawIndexType, (__bridge void *)drawIndexBuffer,
                drawIndexOffset, (__bridge void *)indirectArgsBuffer,
                indirectArgsOffset + (i * argSize));
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SUBMIT"
                              reason:"mdi_indexed"];
        }
    } else {
        MGLDrawPrimitivesIndirectArguments *args =
            (MGLDrawPrimitivesIndirectArguments *)
                ((uint8_t *)indirectArgsContents + indirectArgsOffset);
        mgl_batch_replay_fill_mdi_array_args(batch, args);

        for (uint32_t i = 0; i < batch->command_count; i++) {
            (void)mgl_batch_mtl_draw_array_indirect(
                encCtx->render_encoder_owner, primType,
                (__bridge void *)indirectArgsBuffer,
                indirectArgsOffset + (i * argSize));
            [self traceReplayCommand:batch
                             command:&batch->commands[i]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SUBMIT"
                              reason:"mdi_arrays"];
        }
    }
}


- (void)issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
             encodeContext:(const MGLEncodeContext *)encCtx
{
    /* Mutable working copy: texture materialization may rotate the active
     * encoder, so start from the render-pass manager's live owner. */
    MGLEncodeContext liveEncCtx = *encCtx;
    liveEncCtx.render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;

    if ([self tryReplaySimpleBatch:batch
                                  context:glm_ctx
                            encodeContext:&liveEncCtx]) {
        return;
    }
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        /* A previous command may have rotated the render encoder.  Refresh at
         * each command boundary before any replay helper checks the owner. */
        liveEncCtx.render_encoder_owner =
            _renderPassManager.state->currentRenderEncoderOwner;
        Program *batchProgram =
            mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
        BOOL capturedCullDistances = NO;
        const int cullPath = mgl_batch_issue_cull_capture_path(
            (batchProgram && batchProgram->uses_cull_distance) ? 1 : 0,
            (uint32_t)cmd->type);
        if (cullPath == MGL_BATCH_CULL_CAPTURE_ARRAYS) {
            capturedCullDistances =
                [self captureAIRCullDistancesForArrayDraw:glm_ctx
                                                    first:cmd->first
                                                    count:cmd->count
                                            instanceCount:cmd->instanceCount
                                             baseInstance:cmd->baseInstance];
        } else if (cullPath == MGL_BATCH_CULL_CAPTURE_ELEMENTS) {
            Buffer *elementBuffer = NULL;
            id metalElementBuffer = nil;
            if ([self resolveElementBufferForCommand:cmd
                                                label:"cullDistanceCapture"
                                              context:glm_ctx
                                             glBuffer:&elementBuffer
                                            mtlBuffer:&metalElementBuffer]) {
                const uint8_t *source = mglElementIndexSourceForDraw(
                    elementBuffer, metalElementBuffer, cmd->indexType,
                    cmd->indexBufferOffset, cmd->count);
                capturedCullDistances =
                    [self captureAIRCullDistancesForElementDraw:glm_ctx
                                                     indexBytes:source
                                                      indexType:cmd->indexType
                                                          count:cmd->count
                                                     baseVertex:cmd->baseVertex
                                                  instanceCount:cmd->instanceCount
                                                   baseInstance:cmd->baseInstance];
            }
        }
        if (capturedCullDistances) {
            if (![self processGLState:true] ||
                mglRenderEncoderOwnerHasCurrent(
                    _renderPassManager.state->currentRenderEncoderOwner) == 0) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                 phase:"SKIP"
                                  reason:"cull_distance_capture_restore"];
                continue;
            }
            /* Cull-distance capture ends the active render pass and destroys
             * its C++ owner before processGLState creates the replacement
             * encoder.  Refresh the per-batch context so subsequent draw
             * helpers never query the released owner handle. */
            liveEncCtx.render_encoder_owner =
                _renderPassManager.state->currentRenderEncoderOwner;
        }
        if (![self applyDynamicBindingsForCommand:cmd context:glm_ctx encodeContext:&liveEncCtx]) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"dynamic_binding"];
            continue;
        }
        if (mgl_batch_issue_should_apply_cmd_sampler(
                batch->sampler_snapshots_mixed ? 1 : 0,
                batch->has_dynamic_texture_bindings ? 1 : 0) &&
            ![self applySamplerSnapshotForCommand:cmd context:glm_ctx encodeContext:&liveEncCtx]) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"sampler_snapshot"];
            continue;
        }
        GLenum mode = cmd->mode;
        GLsizei count = cmd->count;
        GLsizei instanceCount = cmd->instanceCount;

        MGLBatchReplayDirectPrimPlan primPlan;
        mgl_batch_replay_direct_prim_plan(
            (uint32_t)mode,
            mglPolygonModePointForDrawMode(glm_ctx, mode) ? 1 : 0,
            (uint32_t)batch->key.primitive_type, &primPlan);
        if (primPlan.skip_unsupported_prim) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_unsupported_primitive"];
            continue;
        }
        const BOOL polygonModePoint = primPlan.polygon_mode_point ? YES : NO;
        const BOOL emulateTriangleFan =
            primPlan.emulate_triangle_fan ? YES : NO;
        const BOOL emulateLineLoop = primPlan.emulate_line_loop ? YES : NO;
        const BOOL emulateQuads = primPlan.emulate_quads ? YES : NO;
        const uint32_t primType = primPlan.prim_type;

        if (mgl_batch_replay_cmd_is_array_draw((uint32_t)cmd->type)) {
            GLsizei ic = instanceCount;
            GLuint bi = 0u;
            const char *reason = NULL;
            const char *cullReason = NULL;
            mgl_batch_issue_direct_arrays_params(
                (uint32_t)cmd->type, instanceCount, cmd->baseInstance, &ic, &bi,
                &reason, &cullReason);
            (void)emulateTriangleFan;
            (void)emulateLineLoop;
            (void)emulateQuads;
            (void)primType;
            if (!polygonModePoint &&
                [self issueDirectBatchCullDistanceArrayDraw:mode
                                                      first:cmd->first
                                                      count:count
                                              instanceCount:ic
                                               baseInstance:bi
                                              encodeContext:&liveEncCtx]) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:cullReason];
            } else {
                [self submitDirectBatchArrayEncode:batch
                                           command:cmd
                                           context:glm_ctx
                                        batchIndex:i
                                              mode:mode
                                             count:count
                                     instanceCount:ic
                                      baseInstance:bi
                                 polygonModePoint:polygonModePoint
                                    encodeContext:&liveEncCtx
                                           reason:reason];
            }
        } else {
            [self issueDirectBatchElementDraw:batch
                                      command:cmd
                                       context:glm_ctx
                                    batchIndex:i
                                          mode:mode
                                        count:count
                                instanceCount:instanceCount
                           polygonModePoint:polygonModePoint
                               encodeContext:&liveEncCtx];
        }
    }
}


- (BOOL)issueDirectBatchCullDistanceArrayDraw:(GLenum)mode
                                         first:(GLint)first
                                         count:(GLsizei)count
                                 instanceCount:(GLsizei)instanceCount
                                  baseInstance:(GLuint)baseInstance
                                 encodeContext:(const MGLEncodeContext *)encCtx
{
    Program *batchProgram =
        mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (!batchProgram || !batchProgram->uses_cull_distance ||
        !mglBatchReplayHasActiveEncoder(encCtx)) {
        return NO;
    }
    return mglEncodeCullDistanceArraySplitForRenderEncoderOwner(
        encCtx->render_encoder_owner, _device, mode, first, count,
        (size_t)instanceCount, (size_t)baseInstance, (__bridge void *)self,
        encCtx, mglRendererBindCullDistanceEmu);
}


- (void)submitDirectBatchArrayEncode:(MGLDrawBatch *)batch
                             command:(MGLDrawCommand *)cmd
                             context:(GLMContext)glm_ctx
                          batchIndex:(uint32_t)i
                                mode:(GLenum)mode
                               count:(GLsizei)count
                       instanceCount:(GLsizei)instanceCount
                        baseInstance:(GLuint)baseInstance
                   polygonModePoint:(BOOL)polygonModePoint
                      encodeContext:(const MGLEncodeContext *)encCtx
                             reason:(const char *)reason
{
    if (!polygonModePoint) {
        Program *batchProgram =
            mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
        if (batchProgram && batchProgram->uses_cull_distance) {
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)cmd->first
                                   explicitVertices:NULL
                                 explicitVertexCount:0u
                                      encodeContext:encCtx];
        }
    }
    const bool ok = mglEncodeDrawArraysForRenderEncoderOwner(
        encCtx->render_encoder_owner, glm_ctx, _device, mode, cmd->first, count,
        (size_t)instanceCount, (size_t)baseInstance, "batch");
    [self traceReplayCommand:batch
                     command:cmd
                     context:glm_ctx
                     flushId:_renderPassManager.state->traceReplayFlushId
                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                commandIndex:i
                       phase:(ok ? "SUBMIT" : "SKIP")
                      reason:reason];
}


- (void)issueDirectBatchElementDraw:(MGLDrawBatch *)batch
                           command:(MGLDrawCommand *)cmd
                            context:(GLMContext)glm_ctx
                         batchIndex:(uint32_t)i
                               mode:(GLenum)mode
                              count:(GLsizei)count
                      instanceCount:(GLsizei)instanceCount
                 polygonModePoint:(BOOL)polygonModePoint
                     encodeContext:(const MGLEncodeContext *)encCtx
{
    /* Element-based draws */
    Buffer *glBuf = NULL;
    id idxBuf = nil;
    if (![self resolveElementBufferForCommand:cmd
                                        label:"directBatch"
                                      context:glm_ctx
                                     glBuffer:&glBuf
                                    mtlBuffer:&idxBuf]) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SKIP"
                          reason:"direct_resolve_element"];
        return;
    }
    NSUInteger idxOffset = cmd->indexBufferOffset;
    uint64_t mtlIdxType = mglIndexTypeForGLType(cmd->indexType);
    if ((GLuint)mtlIdxType == 0xFFFFFFFF) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SKIP"
                          reason:"direct_index_type"];
        return;
    }

    const uint8_t *cullDistanceIndexSource =
        mglElementIndexSourceForDraw(glBuf, idxBuf, cmd->indexType,
                                     idxOffset, count);
    if (!polygonModePoint &&
        [self encodeCullDistanceElementDraw:mode
                                  indexBytes:cullDistanceIndexSource
                                   indexType:cmd->indexType
                                       count:count
                                  baseVertex:cmd->baseVertex
                               instanceCount:instanceCount
                                baseInstance:cmd->baseInstance
                             polygonLineMode:mglPolygonModeLineForDrawMode(
                                                 glm_ctx, mode)
                               encodeContext:encCtx]) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_elements_cull_distance_split"];
        return;
    }

    const bool encoded = mglEncodeDrawElementsForRenderEncoderOwner(
        encCtx->render_encoder_owner,
        glm_ctx,
        _device,
        glBuf,
        idxBuf,
        mode,
        cmd->indexType,
        idxOffset,
        count,
        instanceCount,
        cmd->baseVertex,
        cmd->baseInstance,
        "directBatch");
    [self traceReplayCommand:batch
                     command:cmd
                     context:glm_ctx
                     flushId:_renderPassManager.state->traceReplayFlushId
                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                commandIndex:i
                       phase:(encoded ? "SUBMIT" : "SKIP")
                      reason:(encoded ? "direct_elements" : "direct_elements_encode_failed")];
}



@end
