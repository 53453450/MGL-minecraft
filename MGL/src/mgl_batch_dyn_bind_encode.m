/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3: dyn-bind / sampler / simple-replay encode split from BatchReplay.m
 * (cluster metric). Same (Draw) category; plans in mgl_batch_replay /
 * mgl_batch_issue. Do not expand mgl_batch_replay_trace.m or metal_port.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_byte_hash.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_issue.h"

static const NSUInteger kMaxFragmentSamplerSlots = 16;

static BOOL mglBatchReplayHasActiveEncoder(const MGLEncodeContext *encCtx)
{
    if (!encCtx) return NO;
    return mglRenderEncoderOwnerHasCurrent(
        encCtx->render_encoder_owner) != 0;
}

static uint64_t mglRendererSamplerSnapshotHash(const MGLSamplerSnapshotKey *key)
{
    return mglHashBytesFNV1a(key, sizeof(*key));
}


@implementation MGLRenderer (Draw)

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
    /* A3: map expansion in mgl_batch_replay_plan_uniform_binds; ObjC validates
     * Metal lengths and encodes the binding snapshot. */
    if (!cmd || !glm_ctx || !encCtx) {
        return false;
    }
    uint64_t mtl_lengths[MGL_MAX_DYNAMIC_UNIFORM_BINDINGS];
    memset(mtl_lengths, 0, sizeof(mtl_lengths));
    if (cmd->dynamic_uniform_binding_count > MGL_MAX_DYNAMIC_UNIFORM_BINDINGS) {
        return false;
    }
    for (uint8_t i = 0; i < cmd->dynamic_uniform_binding_count; i++) {
        const MGLDynamicUniformBinding *override =
            &cmd->dynamic_uniform_bindings[i];
        BufferBaseTarget *slot =
            &MGL_STATE(glm_ctx)->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override->binding_index];
        if (!slot->buf || !slot->buf->data.mtl_data ||
            (uintptr_t)slot->buf->data.mtl_data < 0x10000u) {
            return false;
        }
        id metal_buffer = (__bridge id)(slot->buf->data.mtl_data);
        MGLRenderBufferInfo metalBufferInfo = {0};
        if (mglRenderGetBufferInfo((__bridge void *)metal_buffer,
                                   &metalBufferInfo) != 0) {
            return false;
        }
        mtl_lengths[i] = metalBufferInfo.length;
    }

    MGLBatchUniformBindPlan plan;
    if (!mgl_batch_replay_plan_uniform_binds(
            glm_ctx, cmd, mtl_lengths, cmd->dynamic_uniform_binding_count,
            (uint64_t)kMGLMinimumStageBindingSize, (uint32_t)kMGLMaxBufferSlots,
            &plan)) {
        return false;
    }

    MGLRenderBindingSnapshot snapshot = {0};
    for (uint32_t oi = 0; oi < plan.count; oi++) {
        const MGLBatchUniformBindOp *op = &plan.ops[oi];
        BufferBaseTarget *slot =
            &MGL_STATE(glm_ctx)->buffer_base[_UNIFORM_BUFFER]
                 .buffers[op->binding_index];
        if (!slot->buf || !slot->buf->data.mtl_data) {
            return false;
        }
        id metal_buffer = (__bridge id)(slot->buf->data.mtl_data);
        const uint32_t stage =
            op->is_vertex_stage ? MGL_RENDER_BINDING_STAGE_VERTEX
                                : MGL_RENDER_BINDING_STAGE_FRAGMENT;
        if (!mglBindingStateIsValid(_bindingStateOwner) ||
            !mglBindingStateBufferMatches(
                _bindingStateOwner, stage, (__bridge void *)metal_buffer,
                op->offset, op->metal_slot)) {
            if (op->is_vertex_stage) {
                mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)metal_buffer,
                    op->offset, op->metal_slot);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                if (snapshot.vertex_op_count <
                    MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
                    snapshot.vertex_ops[snapshot.vertex_op_count++] =
                        (MGLRenderBindingOp){
                            0u, op->metal_slot, op->offset,
                            (__bridge void *)metal_buffer, NULL, 0u};
                }
            } else {
                mglRenderBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)metal_buffer,
                    op->offset, op->metal_slot);
                MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                if (snapshot.fragment_op_count <
                    MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
                    snapshot.fragment_ops[snapshot.fragment_op_count++] =
                        (MGLRenderBindingOp){
                            0u, op->metal_slot, op->offset,
                            (__bridge void *)metal_buffer, NULL, 0u};
                }
            }
            mglNoteBufferEncoded(slot->buf);
        } else if (op->is_vertex_stage) {
            MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
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
    /* A3: candidate enum in mgl_batch_replay_plan_sampled_texture_candidates;
     * ObjC resolves units/textures and encodes resource bindings. */
    if (!touched_units || !glm_ctx ||
        !mglBatchReplayHasActiveEncoder(encCtx)) {
        return false;
    }
    MGLBatchSampledTexPlan plan;
    if (!mgl_batch_replay_plan_sampled_texture_candidates(glm_ctx, &plan)) {
        return false;
    }
    MGLRenderResourceBindingSnapshot snapshot = {0};
    for (uint32_t i = 0; i < plan.count; i++) {
        const MGLBatchSampledTexCandidate *e = &plan.entries[i];
        MGLShaderResource *resource = e->resource;
        GLuint texture_unit = [self textureUnitForSampledResource:resource
                                                      metalBinding:e->metal_slot
                                                             stage:(int)e->stage];
        if (texture_unit >= TEXTURE_UNITS || !touched_units[texture_unit]) {
            continue;
        }
        Texture *texture_object =
            [self textureForSampledResource:resource
                               metalBinding:e->metal_slot
                                       stage:(int)e->stage
                                expectedType:(e->lookup_type ? e->lookup_type
                                                             : e->expected_type)];
        if (!texture_object || !texture_object->mtl_data ||
            texture_object->dirty_bits || texture_object->is_render_target) {
            return false;
        }
        id texture = (__bridge id)texture_object->mtl_data;
        texture = (__bridge id)mglSampledTextureViewForBaseLevel(
            texture_object, (__bridge void *)texture);
        MGLRenderTextureInfo textureInfo = {0};
        if (!texture ||
            mglRenderGetTextureInfo((__bridge void *)texture, &textureInfo) !=
                0 ||
            (e->expected_type != 0 &&
             textureInfo.texture_type != e->expected_type) ||
            !mglTexturePixelFormatCompatibleWithExpectedDataKind(
                textureInfo.pixel_format,
                (MGLTextureDataKind)e->expected_kind)) {
            return false;
        }
        uint32_t binding_stage =
            mglRenderTextureBindingStageForShader((int)e->stage);
        if (!mgl_batch_replay_collect_resource_binding(
                &snapshot, binding_stage, MGL_RENDER_RESOURCE_BINDING_TEXTURE,
                (__bridge void *)texture, e->metal_slot)) {
            return false;
        }
        if (e->needs_combined_sampler) {
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
            GLuint sampler_slot =
                resource ? mglMetalCombinedSamplerSlot(resource) : e->metal_slot;
            if (sampler_slot >= kMaxFragmentSamplerSlots) {
                return false;
            }
            if (!mgl_batch_replay_collect_resource_binding(
                    &snapshot, binding_stage,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)sampler, sampler_slot)) {
                return false;
            }
        }
    }
    return mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
               _bindingStateOwner, encCtx->render_encoder_owner, &snapshot,
               NULL, 0) == 0;
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
    if (!cmd ||
        !mgl_batch_issue_dyn_cmd_has_bindings(cmd->dynamic_vertex_binding_count,
                                              cmd->dynamic_uniform_binding_count,
                                              cmd->dynamic_texture_binding_count)) {
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
            !mgl_batch_replay_build_dynamic_vertex_array(glm_ctx, base_vao, cmd,
                                                         &dynamic_vao)) {
            return false;
        }
        draw_vao = &dynamic_vao;
    }

    if (cmd->dynamic_uniform_binding_count > 0 &&
        !mgl_batch_replay_apply_uniform_range_overrides(glm_ctx, cmd)) {
        return false;
    }

    bool touched_texture_units[TEXTURE_UNITS] = {false};
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
        /* Texture binding may rotate the encoder owner — refresh. */
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

    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;

    bool direct_vertex_ok =
        cmd->dynamic_vertex_binding_count == 0 ||
        [self bindDynamicVertexArrayBuffersDirectly:draw_vao
                                            command:cmd
                                            context:glm_ctx
                                      encodeContext:encCtx];
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    bool direct_uniform_ok =
        cmd->dynamic_uniform_binding_count == 0 ||
        [self bindDynamicUniformRangesDirectly:cmd
                                       context:glm_ctx
                                 encodeContext:encCtx];
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    if (!mgl_batch_issue_dyn_needs_mapper_fallback(direct_vertex_ok ? 1 : 0,
                                                   direct_uniform_ok ? 1 : 0)) {
        return true;
    }

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


@end
