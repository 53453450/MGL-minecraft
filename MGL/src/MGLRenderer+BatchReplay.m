/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+BatchReplay.m
// Batch replay, dynamic binding and sampler snapshot methods
// extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_byte_hash.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_replay.h"

static const NSUInteger kMaxFragmentSamplerSlots = 16;

static BOOL mglBatchReplayHasActiveEncoder(const MGLEncodeContext *encCtx)
{
    if (!encCtx) return NO;
    return mglRenderEncoderOwnerHasCurrent(
        encCtx->render_encoder_owner) != 0;
}

static void *mglBatchReplayEncoderTraceToken(
    const MGLEncodeContext *encCtx)
{
    if (!encCtx) return NULL;
    return encCtx->render_encoder_owner;
}

static void mglBatchReplayDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    id indirectBuffer,
    NSUInteger indirectBufferOffset)
{
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY_INDIRECT,
            .primitive_type = (uint32_t)primitiveType,
            .indirect_buffer = (__bridge void *)indirectBuffer,
            .indirect_buffer_offset = indirectBufferOffset,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static void mglBatchReplayDrawIndexedPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    uint64_t indexType,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    id indirectBuffer,
    NSUInteger indirectBufferOffset)
{
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_INDEXED_INDIRECT,
            .primitive_type = (uint32_t)primitiveType,
            .index_type = (uint32_t)indexType,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .indirect_buffer = (__bridge void *)indirectBuffer,
            .indirect_buffer_offset = indirectBufferOffset,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static uint64_t mglRendererSamplerSnapshotHash(const MGLSamplerSnapshotKey *key)
{
    return mglHashBytesFNV1a(key, sizeof(*key));
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
        indirectArgsOffset > indirectArgsLength ||
        neededBytes > indirectArgsLength - indirectArgsOffset) {
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
            mglBatchReplayDrawIndexedPrimitivesIndirect(
                encCtx->render_encoder_owner, primType,
                drawIndexType, drawIndexBuffer,
                drawIndexOffset, indirectArgsBuffer,
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
            mglBatchReplayDrawPrimitivesIndirect(
                encCtx->render_encoder_owner, primType,
                indirectArgsBuffer,
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

- (bool)bindDynamicVertexArrayBuffersDirectly:(VertexArray *)vao
                                      command:(const MGLDrawCommand *)cmd
                                       context:(GLMContext)glm_ctx
                                 encodeContext:(const MGLEncodeContext *)encCtx
{
    /* A3: stream plan in mgl_batch_replay_plan_dyn_vertex_streams; ObjC
     * resolves Metal slots + materializes dirty buffers. */
    Program *active_program =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    for (uint8_t binding_index = 0;
         binding_index < cmd->dynamic_vertex_binding_count;
         binding_index++) {
        const MGLDynamicVertexBinding *override =
            &cmd->dynamic_vertex_bindings[binding_index];
        MGLBatchDynVertexStreamPlan plan;
        const int planRc = mgl_batch_replay_plan_dyn_vertex_streams(
            glm_ctx, vao, active_program, override, &plan);
        if (planRc == MGL_BATCH_DYN_VERTEX_FAIL) {
            return false;
        }
        if (planRc == MGL_BATCH_DYN_VERTEX_UNUSED) {
            continue;
        }

        int resolved_slots[MGL_BATCH_DYN_VERTEX_MAX_STREAMS];
        GLuint resolved_slot_count = 0u;
        for (uint32_t stream = 0; stream < plan.stream_count; stream++) {
            int resolved_slot = mglRendererResolveVertexAttributeBufferIndex(
                glm_ctx, vao, plan.representative_attribs[stream], __FUNCTION__);
            if (resolved_slot < 0) {
                continue;
            }
            if (resolved_slot >= (int)kMGLMaxMetalVertexBufferCount) {
                return false;
            }
            if (!mgl_batch_replay_dyn_vertex_stream_can_bind_directly(
                    active_program, vao, &plan, stream)) {
                return false;
            }
            resolved_slots[resolved_slot_count++] = resolved_slot;
        }
        if (resolved_slot_count == 0u) {
            continue;
        }

        Buffer *draw_buffer = plan.buffer;
        if (!draw_buffer) {
            return false;
        }
        NSUInteger dynamic_offset = (NSUInteger)plan.dynamic_offset;
        if (draw_buffer->data.dirty_bits) {
            BufferMapList upload = {0};
            upload.count = 1;
            upload.buffers[0].buf = draw_buffer;
            if (![self updateDirtyBaseBufferList:&upload]) {
                return false;
            }
        }
        if (!draw_buffer->data.mtl_data) {
            [self bindMTLBuffer:draw_buffer];
        }
        if (!draw_buffer->data.mtl_data ||
            (uintptr_t)draw_buffer->data.mtl_data < 0x10000u) {
            return false;
        }

        id metal_buffer =
            (__bridge id)(draw_buffer->data.mtl_data);
        MGLRenderBufferInfo metalBufferInfo = {0};
        if (mglRenderGetBufferInfo((__bridge void *)metal_buffer,
                                      &metalBufferInfo) != 0) {
            return false;
        }
        const BufferBinding *binding = &vao->bindings[plan.binding_index];
        if (binding->offset < 0 ||
            (uint64_t)binding->offset != (uint64_t)dynamic_offset ||
            (uint64_t)binding->offset >= metalBufferInfo.length ||
            dynamic_offset >= metalBufferInfo.length) {
            return false;
        }

        /* Collect the ordered binding updates for one C++ owner replay. */
        MGLRenderBindingSnapshot snapshot = {0};
        for (GLuint stream = 0; stream < resolved_slot_count; stream++) {
            NSUInteger metal_slot = (NSUInteger)resolved_slots[stream];
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_BINDING_STAGE_VERTEX,
                    (__bridge void *)metal_buffer, dynamic_offset,
                    (uint32_t)metal_slot)) {
                mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)metal_buffer,
                    dynamic_offset, (uint32_t)metal_slot);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                mglNoteBufferEncoded(draw_buffer);
                if (snapshot.vertex_op_count <
                    MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
                    snapshot.vertex_ops[snapshot.vertex_op_count++] =
                        (MGLRenderBindingOp){
                            /* kind */ 0u,
                            /* index */ (uint32_t)metal_slot,
                            /* offset */ dynamic_offset,
                            /* buffer */ (__bridge void *)metal_buffer,
                            /* bytes */ NULL,
                            /* length */ 0u};
                }
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
        }
        if (snapshot.vertex_op_count > 0) {
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
                encCtx->render_encoder_owner, &snapshot, NULL, 0);
        }
    }
    return true;
}

- (bool)bindDynamicUniformRangesDirectly:(const MGLDrawCommand *)cmd
                                  context:(GLMContext)glm_ctx
                            encodeContext:(const MGLEncodeContext *)encCtx
{
    BufferMapList *stage_maps[2] = {
        &MGL_STATE(glm_ctx)->vertex_buffer_map_list,
        &MGL_STATE(glm_ctx)->fragment_buffer_map_list,
    };
    const int stages[2] = { _VERTEX_SHADER, _FRAGMENT_SHADER };

    /* Preserve the original setter order in a single C++ owner replay. */
    MGLRenderBindingSnapshot snapshot = {0};

    for (uint8_t dynamic_index = 0;
         dynamic_index < cmd->dynamic_uniform_binding_count;
         dynamic_index++) {
        const MGLDynamicUniformBinding *override =
            &cmd->dynamic_uniform_bindings[dynamic_index];
        BufferBaseTarget *slot =
            &MGL_STATE(glm_ctx)->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override->binding_index];
        if (!slot->buf || !slot->buf->data.mtl_data ||
            (uintptr_t)slot->buf->data.mtl_data < 0x10000u ||
            override->offset < 0 || override->size <= 0) {
            return false;
        }

        id metal_buffer =
            (__bridge id)(slot->buf->data.mtl_data);
        MGLRenderBufferInfo metalBufferInfo = {0};
        if (mglRenderGetBufferInfo((__bridge void *)metal_buffer,
                                      &metalBufferInfo) != 0) {
            return false;
        }
        uint64_t start = (uint64_t)override->offset;
        uint64_t length = (uint64_t)override->size;
        if (!mgl_batch_replay_uniform_range_fits(start, length,
                                                 metalBufferInfo.length)) {
            return false;
        }

        for (int stage_index = 0; stage_index < 2; stage_index++) {
            BufferMapList *maps = stage_maps[stage_index];
            GLuint map_count = maps->count < MAX_MAPPED_BUFFERS
                ? maps->count : MAX_MAPPED_BUFFERS;
            for (GLuint map_index = 0; map_index < map_count; map_index++) {
                BufferMap *map = &maps->buffers[map_index];
                if (map->attribute_mask != 0u ||
                    map->buffer_base_index != override->binding_index ||
                    map->buf != slot->buf) {
                    continue;
                }
                NSUInteger reflected_required_bytes = map->has_metal_binding
                    ? mglRendererGetProgramBindingRequiredSize(
                          ctx, stages[stage_index], (int)map->resource_type,
                          (int)map->resource_index)
                    : mglRendererGetProgramBindingRequiredSizeForStage(
                          ctx, stages[stage_index], override->binding_index);
                NSUInteger required_binding_bytes = kMGLMinimumStageBindingSize;
                if (reflected_required_bytes > required_binding_bytes) {
                    required_binding_bytes = reflected_required_bytes;
                }
                if (length < required_binding_bytes) {
                    return false;
                }

                NSInteger resolved_slot = map->has_metal_binding
                    ? (NSInteger)map->metal_binding_index
                    : mglRendererGetProgramMetalBufferIndexForStage(
                          ctx, stages[stage_index], override->binding_index);
                if (resolved_slot < 0 || resolved_slot >= kMGLMaxBufferSlots) {
                    return false;
                }
                NSUInteger metal_slot = (NSUInteger)resolved_slot;
                if (mglRenderStageMapsVertexAttribs(stages[stage_index])) {
                    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                        !mglBindingStateBufferMatches(
                            _bindingStateOwner,
                            MGL_RENDER_BINDING_STAGE_VERTEX,
                            (__bridge void *)metal_buffer, start,
                            (uint32_t)metal_slot)) {
                        mglRenderBindingUpdateVertexBuffer(
                            _bindingStateOwner, (__bridge void *)metal_buffer,
                            start, (uint32_t)metal_slot);
                        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                        mglNoteBufferEncoded(slot->buf);
                        if (snapshot.vertex_op_count <
                            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
                            snapshot.vertex_ops[
                                snapshot.vertex_op_count++] =
                                (MGLRenderBindingOp){
                                    /* kind */ 0u,
                                    /* index */ (uint32_t)metal_slot,
                                    /* offset */ start,
                                    /* buffer */ (__bridge void *)metal_buffer,
                                    /* bytes */ NULL,
                                    /* length */ 0u};
                        }
                    } else {
                        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                    }
                } else {
                    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                        !mglBindingStateBufferMatches(
                            _bindingStateOwner,
                            MGL_RENDER_BINDING_STAGE_FRAGMENT,
                            (__bridge void *)metal_buffer, start,
                            (uint32_t)metal_slot)) {
                        mglRenderBindingUpdateFragmentBuffer(
                            _bindingStateOwner, (__bridge void *)metal_buffer,
                            start, (uint32_t)metal_slot);
                        MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                        mglNoteBufferEncoded(slot->buf);
                        if (snapshot.fragment_op_count <
                            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
                            snapshot.fragment_ops[
                                snapshot.fragment_op_count++] =
                                (MGLRenderBindingOp){
                                    /* kind */ 0u,
                                    /* index */ (uint32_t)metal_slot,
                                    /* offset */ start,
                                    /* buffer */ (__bridge void *)metal_buffer,
                                    /* bytes */ NULL,
                                    /* length */ 0u};
                        }
                    } else {
                        MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                    }
                }
            }
        }
    }
    if (snapshot.vertex_op_count > 0 || snapshot.fragment_op_count > 0) {
        mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
            encCtx->render_encoder_owner, &snapshot, NULL, 0);
    }
    return true;
}

- (bool)bindDynamicSampledTexturesDirectlyForTouchedUnits:(const bool *)touched_units
                                                   context:(GLMContext)glm_ctx
                                             encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!touched_units || !glm_ctx ||
        !mglBatchReplayHasActiveEncoder(encCtx)) return false;
    MGLRenderResourceBindingSnapshot snapshot = {0};

    for (int stage_index = 0; stage_index < 2; stage_index++) {
        int stage = stage_index == 0 ? _VERTEX_SHADER : _FRAGMENT_SHADER;
        Program *program = mglResolveProgramForStageFromState(glm_ctx, stage);
        GLuint sampled_count = mglRendererGetProgramBindingCount(ctx, stage, _SAMPLED_IMAGE_RES);
        for (GLuint resource_index = 0;
             resource_index < sampled_count;
             resource_index++) {
            GLuint metal_slot = mglRendererGetProgramBinding(ctx, stage, _SAMPLED_IMAGE_RES, (int)resource_index);
            if (metal_slot >= TEXTURE_UNITS) continue;

            MGLShaderResource *resource = NULL;
            if (program &&
                resource_index < program->shader_resources_list[stage]
                                           [_SAMPLED_IMAGE_RES].count) {
                resource = &program->shader_resources_list[stage]
                                   [_SAMPLED_IMAGE_RES]
                                   .list[resource_index];
            }
            if (mglShouldSkipStageTextureResource(
                    program, stage, _SAMPLED_IMAGE_RES, resource)) {
                continue;
            }
            if (resource && resource->is_array) return false;

            GLuint texture_unit = [self textureUnitForSampledResource:resource
                                                          metalBinding:metal_slot
                                                                 stage:stage];
            if (texture_unit >= TEXTURE_UNITS ||
                !touched_units[texture_unit]) {
                continue;
            }

            uint32_t expected_type =
                mglRendererGetProgramExpectedTextureType(ctx, stage, _SAMPLED_IMAGE_RES, (int)resource_index);
            uint32_t lookup_type =
                mglRendererGetProgramDeclaredTextureType(ctx, stage, _SAMPLED_IMAGE_RES, (int)resource_index);
            MGLTextureDataKind expected_kind =
                (MGLTextureDataKind)mglRendererGetProgramExpectedTextureDataKind(ctx, stage, _SAMPLED_IMAGE_RES, (int)resource_index);
            Texture *texture_object =
                [self textureForSampledResource:resource
                                   metalBinding:metal_slot
                                           stage:stage
                                    expectedType:(lookup_type ? lookup_type
                                                              : expected_type)];
            if (!texture_object || !texture_object->mtl_data ||
                texture_object->dirty_bits || texture_object->is_render_target) {
                return false;
            }

            id texture =
                (__bridge id)texture_object->mtl_data;
            texture = (__bridge id)mglSampledTextureViewForBaseLevel(texture_object, (__bridge void *)texture);
            MGLRenderTextureInfo textureInfo = {0};
            if (!texture ||
                mglRenderGetTextureInfo((__bridge void *)texture,
                                           &textureInfo) != 0 ||
                (expected_type != 0 &&
                 textureInfo.texture_type != expected_type) ||
                !mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    textureInfo.pixel_format, expected_kind)) {
                return false;
            }

            uint32_t binding_stage =
                mglRenderTextureBindingStageForShader(stage);
            if (!mgl_batch_replay_collect_resource_binding(
                    &snapshot, binding_stage,
                    MGL_RENDER_RESOURCE_BINDING_TEXTURE,
                    (__bridge void *)texture, metal_slot)) {
                return false;
            }

            if (!resource || resource->has_combined_sampler) {
                id sampler = nil;
                Sampler *bound_sampler =
                    MGL_STATE(glm_ctx)->texture_samplers[texture_unit];
                if (bound_sampler) {
                    if (bound_sampler->dirty_bits || !bound_sampler->mtl_data) {
                        return false;
                    }
                    sampler = (__bridge id)bound_sampler->mtl_data;
                } else if (texture_object->params.mtl_data) {
                    sampler = (__bridge id)texture_object->params.mtl_data;
                } else {
                    return false;
                }

                GLuint sampler_slot = resource
                    ? mglMetalCombinedSamplerSlot(resource) : metal_slot;
                if (sampler_slot >= kMaxFragmentSamplerSlots) return false;
                if (!mgl_batch_replay_collect_resource_binding(
                        &snapshot, binding_stage,
                        MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                        (__bridge void *)sampler, sampler_slot)) {
                    return false;
                }
            }
        }
    }
    return mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
        _bindingStateOwner, encCtx->render_encoder_owner,
        &snapshot, NULL, 0) == 0;
}

- (id)samplerStateForSnapshotKey:(const MGLSamplerSnapshotKey *)key
{
    if (!key) return nil;
    void *cachedState = NULL;
    int cacheResult = mglRendererBackendGetSamplerSnapshotState(
        _backend, key, &cachedState);
    if (cacheResult == 1) {
        return (__bridge id)cachedState;
    }
    if (cacheResult < 0) {
        return nil;
    }

    TextureParameter params;
    mgl_batch_replay_fill_sampler_params(key, &params);
    id state =
        [self createMTLSamplerForTexParam:&params target:key->target];
    if (!state) return nil;
    return mglRendererBackendPutSamplerSnapshotState(
        _backend, key, (__bridge void *)state) == 0 ? state : nil;
}

- (bool)applySamplerSnapshotForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!cmd || !glm_ctx) return false;
    if (cmd->sampler_snapshot_id == MGL_INVALID_SAMPLER_SNAPSHOT_ID) return true;
    if (!mglBatchReplayHasActiveEncoder(encCtx)) return false;

    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cmd->sampler_snapshot_id >= cb->sampler_snapshot_set_count) return false;
    const MGLSamplerSnapshotSet *set =
        &cb->sampler_snapshot_sets[cmd->sampler_snapshot_id];
    if (set->count > MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES) return false;
    MGLRenderResourceBindingSnapshot snapshot = {0};

    for (uint8_t i = 0; i < set->count; i++) {
        const MGLSamplerSnapshotEntry *entry = &set->entries[i];
        if (entry->metal_slot >= 16u) {
            return false;
        }
        id sampler;
        if (entry->key_index == MGL_FALLBACK_SAMPLER_KEY_INDEX) {
            sampler = [self fallbackSamplerState];
        } else {
            if (entry->key_index >= cb->sampler_snapshot_key_count) return false;
            sampler = [self samplerStateForSnapshotKey:
                &cb->sampler_snapshot_keys[entry->key_index]];
        }
        if (!sampler) return false;

        /* The snapshot overrides whatever the resolve path bound, so this is the
         * only place the per-draw sampler is observable under deferred batching. */
        if (mglMipDiagEnabled() && entry->stage == _FRAGMENT_SHADER &&
            entry->key_index != MGL_FALLBACK_SAMPLER_KEY_INDEX) {
            const MGLSamplerSnapshotKey *key = &cb->sampler_snapshot_keys[entry->key_index];
            static uint64_t s_snapshotState[16];
            if (mglMipDiagStateChanged(&s_snapshotState[entry->metal_slot],
                                       mglRendererSamplerSnapshotHash(key))) {
                NSLog(@"MGL MIP_DIAG snapshot slot=%u unit=%u target=0x%x "
                      @"minFilter=0x%x magFilter=0x%x minLod=%.1f maxLod=%.1f aniso=%.1f",
                      (unsigned)entry->metal_slot,
                      (unsigned)entry->texture_unit,
                      (unsigned)key->target,
                      (unsigned)key->min_filter,
                      (unsigned)key->mag_filter,
                      (double)key->min_lod,
                      (double)key->max_lod,
                      (double)key->max_anisotropy);
            }
        }

        uint32_t bindingStage;
        if (!mglRenderSamplerBindingStageForShader((int)entry->stage,
                                                   &bindingStage)) {
            return false;
        }
        if (!mgl_batch_replay_collect_resource_binding(
                &snapshot, bindingStage,
                MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                (__bridge void *)sampler, entry->metal_slot)) {
            return false;
        }
    }
    return mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
        _bindingStateOwner, encCtx->render_encoder_owner,
        &snapshot, NULL, 0) == 0;
}

- (bool)applyDynamicBindingsForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(MGLEncodeContext *)encCtx
{
    if (!cmd || (cmd->dynamic_vertex_binding_count == 0 &&
                 cmd->dynamic_uniform_binding_count == 0 &&
                 cmd->dynamic_texture_binding_count == 0)) {
        return true;
    }
    if (!glm_ctx || !encCtx) {
        return false;
    }
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    if (!mglBatchReplayHasActiveEncoder(encCtx)) {
        return false;
    }

    VertexArray dynamic_vao;
    VertexArray *base_vao = MGL_STATE(glm_ctx)->vao;
    VertexArray *draw_vao = base_vao;
    if (cmd->dynamic_vertex_binding_count > 0) {
        if (!base_vao || base_vao->magic != MGL_VAO_MAGIC ||
            !mgl_batch_replay_build_dynamic_vertex_array(glm_ctx, base_vao, cmd, &dynamic_vao)) {
            return false;
        }
        draw_vao = &dynamic_vao;
    }

    /* O2.3: UBO/texture unit expansion in mgl_batch_replay_*. */
    if (cmd->dynamic_uniform_binding_count > 0 &&
        !mgl_batch_replay_apply_uniform_range_overrides(glm_ctx, cmd)) {
        return false;
    }

    bool touched_texture_units[TEXTURE_UNITS] = { false };
    if (cmd->dynamic_texture_binding_count > 0 &&
        !mgl_batch_replay_apply_texture_overrides(
            glm_ctx, cmd, touched_texture_units, TEXTURE_UNITS)) {
        return false;
    }

    bool direct_texture_ok = true;
    if (cmd->dynamic_texture_binding_count > 0) {
        direct_texture_ok =
            [self bindDynamicSampledTexturesDirectlyForTouchedUnits:
                touched_texture_units
                                                           context:glm_ctx
                                                     encodeContext:encCtx];
        /* Texture binding may close and recreate the render encoder while
         * materializing an RT-sampled texture.  The owner can be destroyed
         * and reallocated, so refresh the handle before inspecting or using
         * it again. */
        encCtx->render_encoder_owner =
            _renderPassManager.state->currentRenderEncoderOwner;
        if (!direct_texture_ok) {
            direct_texture_ok = [self bindTexturesToCurrentRenderEncoder:encCtx];
            encCtx->render_encoder_owner =
                _renderPassManager.state->currentRenderEncoderOwner;
            if (!direct_texture_ok) {
                direct_texture_ok =
                    [self restoreRenderEncoderAfterTextureUploadForDraw:
                        "dynamic-sampled-texture-bind"];
                encCtx->render_encoder_owner =
                    _renderPassManager.state->currentRenderEncoderOwner;
                if (direct_texture_ok) {
                    direct_texture_ok =
                        [self bindTexturesToCurrentRenderEncoder:encCtx];
                    encCtx->render_encoder_owner =
                        _renderPassManager.state->currentRenderEncoderOwner;
                }
            }
        }
    }
    if (!direct_texture_ok) {
        return false;
    }

    /* Keep all per-draw buffer overrides on the live encoder after texture
     * materialization or storage-image binding has rotated its owner. */
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;

    bool direct_vertex_ok = cmd->dynamic_vertex_binding_count == 0 ||
        [self bindDynamicVertexArrayBuffersDirectly:draw_vao
                                            command:cmd
                                            context:glm_ctx
                                      encodeContext:encCtx];
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    bool direct_uniform_ok = cmd->dynamic_uniform_binding_count == 0 ||
        [self bindDynamicUniformRangesDirectly:cmd context:glm_ctx encodeContext:encCtx];
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    if (direct_vertex_ok && direct_uniform_ok) {
        return true;
    }

    /* Uncommon conversion, undersized-range and allocation cases reuse the
     * full validated mapper.  Pipeline, textures and render state remain
     * constant for the containing batch. */
    VertexArray *saved_vao = MGL_STATE(glm_ctx)->vao;
    if (cmd->dynamic_vertex_binding_count > 0) {
        MGL_STATE(glm_ctx)->vao = draw_vao;
    }
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    bool fallback_ok = [self mapBuffersToMTL] &&
        [self bindVertexBuffersToCurrentRenderEncoder:encCtx];
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    if (fallback_ok && cmd->dynamic_uniform_binding_count > 0) {
        fallback_ok = [self bindFragmentBuffersToCurrentRenderEncoder:encCtx];
        encCtx->render_encoder_owner =
            _renderPassManager.state->currentRenderEncoderOwner;
    }
    MGL_STATE(glm_ctx)->vao = saved_vao;
    return fallback_ok;
}


- (BOOL)tryReplaySimpleBatch:(MGLDrawBatch *)batch
                            context:(GLMContext)glm_ctx
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    /* O2.5: eligibility in mgl_batch_replay_simple_eligible. */
    if (!batch || batch->command_count == 0u) {
        return NO;
    }
    Program *batchProgram =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    const GLenum batchMode = batch->commands[0].mode;
    if (!mgl_batch_replay_simple_eligible(
            batch, MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS,
            mglBatchReplayHasActiveEncoder(encCtx) ? 1 : 0,
            (batchProgram && batchProgram->uses_cull_distance) ? 1 : 0,
            MGL_STATE(glm_ctx)->caps.primitive_restart ? 1 : 0,
            mglPolygonModePointForDrawMode(glm_ctx, batchMode) ? 1 : 0,
            mglRenderDrawModeNeedsEmulate((uint32_t)batchMode) ? 1 : 0)) {
        return NO;
    }

    MGLRenderReplayBatchCommand cmds[MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS];
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        MGLRenderReplayBatchCommand *out = &cmds[i];
        mgl_batch_replay_fill_simple_cmd_common(cmd, out);
        if (mgl_batch_replay_cmd_is_array_draw((uint32_t)cmd->type)) {
            continue;
        }
        switch (cmd->type) {
            case MGL_CMD_DRAW_ELEMENTS:
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED:
            case MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX:
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX:
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE:
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE: {
                Buffer *glBuf = NULL;
                id idxBuf = nil;
                if (![self resolveElementBufferForCommand:cmd
                                                    label:"cppBatchReplay"
                                                  context:glm_ctx
                                                 glBuffer:&glBuf
                                                mtlBuffer:&idxBuf]) {
                    return NO;
                }
                NSUInteger idxOffset = cmd->indexBufferOffset;
                uint64_t mtlIdxType = mglIndexTypeForGLType(cmd->indexType);
                if ((GLuint)mtlIdxType == 0xFFFFFFFFu) {
                    return NO;
                }
                id prepared = mglPreparedElementIndexBuffer(
                    _device, glBuf, idxBuf, cmd->indexType,
                    &idxOffset, &mtlIdxType);
                if (!prepared || (GLuint)mtlIdxType == 0xFFFFFFFFu) {
                    return NO;
                }
                out->index_type = (uint32_t)mtlIdxType;
                out->index_buffer_offset = (uint32_t)idxOffset;
                out->index_buffer = (__bridge void *)prepared;
                break;
            }
            default:
                return NO;
        }
    }

    MGLRenderReplayBatch replayBatch = {
        .primitive_type = (uint32_t)batch->key.primitive_type,
        .command_count = batch->command_count,
        .commands = cmds,
    };
    return mglRenderReplayBatchDrawsForRenderEncoderOwner(
        encCtx->render_encoder_owner, &replayBatch, NULL, 0) ==
        MGL_RENDER_REPLAY_BATCH_OK;
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
        if (batchProgram && batchProgram->uses_cull_distance &&
            mgl_batch_replay_cmd_is_array_draw((uint32_t)cmd->type)) {
            capturedCullDistances =
                [self captureAIRCullDistancesForArrayDraw:glm_ctx
                                                    first:cmd->first
                                                    count:cmd->count
                                            instanceCount:cmd->instanceCount
                                             baseInstance:cmd->baseInstance];
        } else if (batchProgram && batchProgram->uses_cull_distance) {
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
        if ((batch->sampler_snapshots_mixed ||
             batch->has_dynamic_texture_bindings) &&
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
            const GLsizei ic =
                (cmd->type == MGL_CMD_DRAW_ARRAYS) ? 1 : instanceCount;
            const GLuint bi = (cmd->type == MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE)
                                  ? cmd->baseInstance
                                  : 0u;
            const char *reason =
                (cmd->type == MGL_CMD_DRAW_ARRAYS)
                    ? "direct_arrays"
                    : ((cmd->type == MGL_CMD_DRAW_ARRAYS_INSTANCED)
                           ? "direct_arrays_instanced"
                           : "direct_arrays_base_instance");
            const char *cullReason =
                (cmd->type == MGL_CMD_DRAW_ARRAYS)
                    ? "direct_arrays_cull_distance_split"
                    : ((cmd->type == MGL_CMD_DRAW_ARRAYS_INSTANCED)
                           ? "direct_arrays_instanced_cull_distance_split"
                           : "direct_arrays_base_instance_cull_distance_split");
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
