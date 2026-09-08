/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3: ICB + stream-MDI encode split from MGLRenderer+Batch.m (cluster metric).
 * Same (Batch) category; gates/plans in mgl_batch_issue / mgl_batch_replay.
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_batch_path.h"
#include "mgl_batch_replay.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"


static BOOL mglBatchHasActiveEncoder(void *owner)
{
    return mglRenderEncoderOwnerHasCurrent(owner) != 0;
}


@implementation MGLRenderer (Batch)

- (BOOL)issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                    encodeContext:(const MGLEncodeContext *)encCtx
{
    /* A3: stream-MDI gate in mgl_batch_issue_stream_mdi_gate. */
    size_t neededBytesRaw = 0;
    MGLBatchStreamMdiGateIn gateIn = {
        .stream_merged = (batch && batch->stream_merged) ? 1u : 0u,
        .has_encoder =
            (encCtx && mglBatchHasActiveEncoder(encCtx->render_encoder_owner))
                ? 1u
                : 0u,
        .disable_mdi = mglEnvFlagEnabled("MGL_DISABLE_MDI") ? 1u : 0u,
        .primitive_type = batch ? batch->key.primitive_type : 0xFFu,
        .command_count = batch ? batch->command_count : 0u,
        .stream_index_count = batch ? batch->stream_index_count : 0u,
        .arg_size = sizeof(MGLDrawIndexedPrimitivesIndirectArguments),
    };
    const int gate = mgl_batch_issue_stream_mdi_gate(&gateIn, &neededBytesRaw);
    if (gate != MGL_BATCH_STREAM_MDI_OK) {
        if (gate == MGL_BATCH_STREAM_MDI_FAIL_OVERFLOW && batch &&
            batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:mgl_batch_issue_stream_mdi_gate_reason(gate)];
        }
        return NO;
    }

    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;
    if (!indexBuffer || ![self processBuffer:indexBuffer]) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_mdi_index_buffer"];
        }
        return NO;
    }

    id mtlIndexBuffer = (__bridge id)(indexBuffer->data.mtl_data);
    if (!mtlIndexBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_mdi_no_mtl_index"];
        }
        return NO;
    }

    size_t argSize = sizeof(MGLDrawIndexedPrimitivesIndirectArguments);
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
                              reason:"stream_mdi_args_alloc"];
        }
        return NO;
    }
    void *indirectArgsContents = NULL;
    uint64_t indirectArgsLength = 0;
    if (mglRenderGetBufferContents(
            (__bridge void *)indirectArgsBuffer, &indirectArgsContents,
            &indirectArgsLength) != 0 ||
        !mgl_batch_issue_scratch_range_ok(indirectArgsOffset, neededBytes,
                                          indirectArgsLength)) {
        return NO;
    }

    MGLDrawIndexedPrimitivesIndirectArguments *args =
        (MGLDrawIndexedPrimitivesIndirectArguments *)
            ((uint8_t *)indirectArgsContents + indirectArgsOffset);
    mgl_batch_replay_fill_stream_mdi_indexed_args(batch, args);

    uint32_t primType = (uint32_t)batch->key.primitive_type;
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        (void)mgl_batch_mtl_draw_indexed_indirect(
            encCtx->render_encoder_owner, primType, MGL_DRAW_INDEX_UINT32,
            (__bridge void *)mtlIndexBuffer, (uint64_t)cmd->indexBufferOffset,
            (__bridge void *)indirectArgsBuffer,
            indirectArgsOffset + (i * argSize));
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"stream_mdi_indexed"];
    }

    return YES;
}


- (BOOL)issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx
{
    /* O2.4/O2.5: ICB eligibility in mgl_batch_replay_icb_gate + unified env. */
    MGLBatchIcbConfig icb = mgl_batch_icb_config();
    const int icbGate = mgl_batch_replay_icb_gate(
        batch, _device ? 1 : 0,
        mglBatchHasActiveEncoder(encCtx ? encCtx->render_encoder_owner : NULL)
            ? 1
            : 0,
        (int)icb.enable, (int)icb.disable);
    if (icbGate != MGL_BATCH_ICB_OK) {
        if (icbGate != MGL_BATCH_ICB_DISABLED && batch &&
            batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:mgl_batch_replay_icb_gate_reason(icbGate)];
        }
        return NO;
    }

    if (@available(macOS 10.14, *)) {
        BOOL indexed = batch->uses_elements ? YES : NO;
        id icb = nil;
        @try {
            icb = (__bridge_transfer id)mgl_batch_mtl_create_icb(
                indexed ? 1 : 0, (uint64_t)batch->command_count);
        } @catch (NSException *exception) {
            static uint64_t s_icbCreateExceptionCount = 0;
            uint64_t hit = ++s_icbCreateExceptionCount;
            if (hit <= 8ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: ICB creation failed, falling back to indirect draw loop: %@", exception);
            }
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"icb_create_exception"];
            return NO;
        }
        if (!icb) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"icb_create_nil"];
            return NO;
        }

        (void)mgl_batch_mtl_reset_icb(
            (__bridge void *)icb, 0, (uint64_t)batch->command_count);

        uint32_t primType = (uint32_t)batch->key.primitive_type;
        if (indexed) {
            for (uint32_t i = 0; i < batch->command_count; i++) {
                MGLDrawCommand *cmd = &batch->commands[i];
                if (mglRenderIndexTypeIsU8((uint32_t)cmd->indexType)) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_u8_index"];
                    return NO;
                }

                Buffer *glBuf = NULL;
                id idxBuf = nil;
                if (![self resolveElementBufferForCommand:cmd
                                                    label:"icbBatch"
                                                  context:glm_ctx
                                                 glBuffer:&glBuf
                                                mtlBuffer:&idxBuf]) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_resolve_element"];
                    return NO;
                }

                NSUInteger drawIndexOffset = cmd->indexBufferOffset;
                uint64_t drawIndexType = mglIndexTypeForGLType(cmd->indexType);
                id drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                              glBuf,
                                                                              idxBuf,
                                                                              cmd->indexType,
                                                                              &drawIndexOffset,
                                                                              &drawIndexType);
                if (!drawIndexBuffer || (GLuint)drawIndexType == 0xFFFFFFFF) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_prepared_index"];
                    return NO;
                }

                id indirectCommand =
                    (__bridge id)mgl_batch_mtl_icb_command(
                        (__bridge void *)icb, (uint64_t)i);
                if (!indirectCommand) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_command_nil"];
                    return NO;
                }

                (void)mgl_batch_mtl_set_icb_draw_indexed(
                    (__bridge void *)indirectCommand, primType,
                    (uint64_t)cmd->count, (uint32_t)drawIndexType,
                    (__bridge void *)drawIndexBuffer, drawIndexOffset,
                    (uint64_t)cmd->instanceCount, (int64_t)cmd->baseVertex,
                    (uint64_t)cmd->baseInstance);
                (void)mgl_batch_mtl_use_render_resource(
                    encCtx->render_encoder_owner,
                    (__bridge void *)drawIndexBuffer, 1u, 1u);
            }
        } else {
            for (uint32_t i = 0; i < batch->command_count; i++) {
                MGLDrawCommand *cmd = &batch->commands[i];
                id indirectCommand =
                    (__bridge id)mgl_batch_mtl_icb_command(
                        (__bridge void *)icb, (uint64_t)i);
                if (!indirectCommand) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_command_nil"];
                    return NO;
                }
                MGLBatchIcbArrayDrawParams ap;
                mgl_batch_issue_icb_array_draw_params(
                    (uint32_t)cmd->first, (uint32_t)cmd->count,
                    (uint32_t)cmd->instanceCount, (uint32_t)cmd->baseInstance,
                    &ap);
                (void)mgl_batch_mtl_set_icb_draw(
                    (__bridge void *)indirectCommand, primType, ap.vertex_start,
                    ap.vertex_count, ap.instance_count, ap.base_instance);
            }
        }

        (void)mgl_batch_mtl_use_render_resource(
            encCtx->render_encoder_owner, (__bridge void *)icb, 1u, 1u);
        (void)mgl_batch_mtl_execute_icb(
            encCtx->render_encoder_owner, (__bridge void *)icb, 0,
            (uint64_t)batch->command_count);
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch
                             command:&batch->commands[i]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SUBMIT"
                              reason:"icb"];
        }
        return YES;
    }

    return NO;
}


- (id)mdiArgumentScratchBufferWithLength:(NSUInteger)length
                                             offset:(NSUInteger *)offsetOut
{
    return (__bridge id)[_renderPassManager
        mdiArgumentScratchBufferWithDevice:(__bridge void *)_device
                                     length:length
                                     offset:offsetOut];
}

@end
