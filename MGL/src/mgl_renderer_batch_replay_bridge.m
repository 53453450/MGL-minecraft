/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Extracted from MGLRenderer+BatchReplay.m.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_byte_hash.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"


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

static bool mglBatchReplayCollectResourceBinding(
    MGLRenderResourceBindingSnapshot *snapshot,
    uint32_t stage,
    uint32_t kind,
    void *resource,
    uint32_t index)
{
    if (!snapshot || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        kind > MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return false;
    }
    uint32_t *count = stage == MGL_RENDER_BINDING_STAGE_VERTEX
        ? &snapshot->vertex_op_count : &snapshot->fragment_op_count;
    MGLRenderResourceBindingOp *ops =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? snapshot->vertex_ops : snapshot->fragment_ops;
    if (*count >= MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        return false;
    }
    ops[(*count)++] = (MGLRenderResourceBindingOp){
        .kind = kind,
        .index = index,
        .resource = resource,
    };
    return true;
}

static void mglBatchReplayDrawPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger vertexStart,
    NSUInteger vertexCount,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    const MGLRenderDrawPlan plan = {
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

static void mglBatchReplayDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger indexCount,
    uint64_t indexType,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)indexType,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
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

static bool mglBuildDynamicVertexArray(const VertexArray *base,
                                       const MGLDrawCommand *cmd,
                                       VertexArray *out)
{
    if (!base || !cmd || !out ||
        cmd->dynamic_vertex_binding_count > MGL_MAX_DYNAMIC_VERTEX_BINDINGS) {
        return false;
    }

    *out = *base;
    for (uint8_t i = 0; i < cmd->dynamic_vertex_binding_count; i++) {
        const MGLDynamicVertexBinding *override =
            &cmd->dynamic_vertex_bindings[i];
        if (!override->buffer ||
            override->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
            return false;
        }
        BufferBinding *binding = &out->bindings[override->binding_index];
        binding->buffer = (Buffer *)override->buffer;
        binding->offset = (GLintptr)override->offset;
        /* Keep classic VertexAttribPointer mirror fields in sync so resolve
         * stays correct if a path still reads attrib.binding_offset. */
        for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
            if ((out->enabled_attribs & (1u << attrib)) == 0u ||
                out->attrib[attrib].buffer_bindingindex !=
                    override->binding_index) {
                continue;
            }
            out->attrib[attrib].buffer = (Buffer *)override->buffer;
            out->attrib[attrib].binding_offset = (GLintptr)override->offset;
        }
    }
    return true;
}

static bool mglDynamicVertexAttribCanBindDirectly(Program *active_program,
                                                   GLuint attrib_index,
                                                   const VertexAttrib *attrib)
{
    if (!attrib || attrib->long_attribute || attrib->type == GL_DOUBLE ||
        (!attrib->integer &&
         (attrib->type == GL_INT || attrib->type == GL_UNSIGNED_INT))) {
        return false;
    }
    if (!attrib->integer) {
        return true;
    }

    MGLShaderResource *resource =
        mglRendererProgramVertexAttribResource(active_program, attrib_index);
    GLuint shader_type = resource ? resource->gl_type : 0u;
    return !mglIntegerAttribNeedsConversion(attrib->type,
                                            shader_type,
                                            attrib->size,
                                            NULL);
}

static uint64_t mglRendererSamplerSnapshotHash(const MGLSamplerSnapshotKey *key)
{
    return mglHashBytesFNV1a(key, sizeof(*key));
}


@implementation MGLRenderer (Draw)

- (void)issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!batch || batch->command_count == 0) {
        return;
    }
    if (mglEnvFlagEnabled("MGL_DISABLE_MDI")) {
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }
    if (batch->key.primitive_type == 0xFFu) {
        [self traceReplayCommand:batch
                         command:&batch->commands[0]
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:0
                           phase:"FALLBACK"
                          reason:"mdi_unsupported_primitive"];
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    bool indexed = batch->uses_elements;
    size_t argSize = indexed ? sizeof(MGLDrawIndexedPrimitivesIndirectArguments)
                             : sizeof(MGLDrawPrimitivesIndirectArguments);
    if (batch->command_count > (UINT32_MAX / argSize)) {
        [self traceReplayCommand:batch
                         command:&batch->commands[0]
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:0
                           phase:"FALLBACK"
                          reason:"mdi_args_overflow"];
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }
    NSUInteger neededBytes = (NSUInteger)argSize * (NSUInteger)batch->command_count;

    NSUInteger indirectArgsOffset = 0;
    id indirectArgsBuffer =
        [self mdiArgumentScratchBufferWithLength:neededBytes
                                          offset:&indirectArgsOffset];
    if (!indirectArgsBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
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
        for (uint32_t i = 0; i < batch->command_count; i++) {
            MGLDrawCommand *cmd = &batch->commands[i];
            if (cmd->indexType != glIdxType) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"FALLBACK"
                                  reason:"mdi_mixed_index_type"];
                [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
                return;
            }
            args[i].indexCount = (uint32_t)cmd->count;
            args[i].instanceCount = (uint32_t)cmd->instanceCount;
            args[i].indexStart = 0u;
            args[i].baseVertex = cmd->baseVertex;
            args[i].baseInstance = cmd->baseInstance;
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
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
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
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
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
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SUBMIT"
                              reason:"mdi_indexed"];
        }
    } else {
        MGLDrawPrimitivesIndirectArguments *args =
            (MGLDrawPrimitivesIndirectArguments *)
                ((uint8_t *)indirectArgsContents + indirectArgsOffset);
        for (uint32_t i = 0; i < batch->command_count; i++) {
            MGLDrawCommand *cmd = &batch->commands[i];
            args[i].vertexCount = (uint32_t)cmd->count;
            args[i].instanceCount = (uint32_t)cmd->instanceCount;
            args[i].vertexStart = (uint32_t)cmd->first;
            args[i].baseInstance = cmd->baseInstance;
        }

        for (uint32_t i = 0; i < batch->command_count; i++) {
            mglBatchReplayDrawPrimitivesIndirect(
                encCtx->render_encoder_owner, primType,
                indirectArgsBuffer,
                indirectArgsOffset + (i * argSize));
            [self traceReplayCommand:batch
                             command:&batch->commands[i]
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
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
    Program *active_program =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    for (uint8_t binding_index = 0;
         binding_index < cmd->dynamic_vertex_binding_count;
         binding_index++) {
        const MGLDynamicVertexBinding *override =
            &cmd->dynamic_vertex_bindings[binding_index];
        if (!override->buffer ||
            override->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
            return false;
        }

        const BufferBinding *binding = &vao->bindings[override->binding_index];
        if (binding->buffer != (Buffer *)override->buffer) {
            return false;
        }

        /* Slot assignment depends on the effective stream, not on attribute
         * format or relative offset. Attributes sharing a DSA binding almost
         * always share one stream, so resolve that stream once per draw. */
        GLuint representative_attribs[MAX_ATTRIBS];
        GLuint representative_strides[MAX_ATTRIBS];
        GLuint representative_count = 0u;
        for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
            if ((vao->enabled_attribs & (1u << attrib)) == 0u ||
                vao->attrib[attrib].buffer_bindingindex !=
                    override->binding_index ||
                !mglRendererProgramUsesVertexAttrib(active_program, attrib)) {
                continue;
            }

            GLuint effective_stride = binding->stride > 0
                ? (GLuint)binding->stride : vao->attrib[attrib].stride;
            bool known_stream = false;
            for (GLuint stream = 0; stream < representative_count; stream++) {
                if (representative_strides[stream] == effective_stride) {
                    known_stream = true;
                    break;
                }
            }
            if (!known_stream) {
                representative_attribs[representative_count] = attrib;
                representative_strides[representative_count] = effective_stride;
                representative_count++;
            }
        }

        int resolved_slots[MAX_ATTRIBS];
        GLuint resolved_slot_count = 0u;
        for (GLuint stream = 0; stream < representative_count; stream++) {
            GLuint representative = representative_attribs[stream];
            int resolved_slot = mglRendererResolveVertexAttributeBufferIndex(
                glm_ctx, vao, representative, __FUNCTION__);
            if (resolved_slot < 0) {
                continue;
            }
            if (resolved_slot >= (int)kMGLMaxMetalVertexBufferCount) {
                return false;
            }

            for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
                if ((vao->enabled_attribs & (1u << attrib)) == 0u ||
                    vao->attrib[attrib].buffer_bindingindex !=
                        override->binding_index ||
                    !mglRendererProgramUsesVertexAttrib(active_program, attrib)) {
                    continue;
                }
                GLuint effective_stride = binding->stride > 0
                    ? (GLuint)binding->stride : vao->attrib[attrib].stride;
                if (effective_stride == representative_strides[stream] &&
                    !mglDynamicVertexAttribCanBindDirectly(active_program,
                                                          attrib,
                                                          &vao->attrib[attrib])) {
                    return false;
                }
            }
            resolved_slots[resolved_slot_count++] = resolved_slot;
        }

        /* A captured binding may be unused by this shader. */
        if (resolved_slot_count == 0u) {
            continue;
        }

        Buffer *draw_buffer = (Buffer *)override->buffer;
        NSUInteger dynamic_offset = (NSUInteger)override->offset;
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
        if (start > metalBufferInfo.length ||
            length > metalBufferInfo.length - start) {
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
                if (stages[stage_index] == _VERTEX_SHADER) {
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

            uint32_t binding_stage = stage == _VERTEX_SHADER
                ? MGL_RENDER_BINDING_STAGE_VERTEX
                : MGL_RENDER_BINDING_STAGE_FRAGMENT;
            if (!mglBatchReplayCollectResourceBinding(
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
                if (!mglBatchReplayCollectResourceBinding(
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
    memset(&params, 0, sizeof(params));
    params.min_filter = key->min_filter;
    params.mag_filter = key->mag_filter;
    params.wrap_s = key->wrap_s;
    params.wrap_t = key->wrap_t;
    params.wrap_r = key->wrap_r;
    params.compare_mode = key->compare_mode;
    params.compare_func = key->compare_func;
    params.max_anisotropy = key->max_anisotropy;
    params.min_lod = key->min_lod;
    params.max_lod = key->max_lod;
    memcpy(params.border_color, key->border_color, sizeof(params.border_color));

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
        if (entry->stage == _VERTEX_SHADER) {
            bindingStage = MGL_RENDER_BINDING_STAGE_VERTEX;
        } else if (entry->stage == _FRAGMENT_SHADER) {
            bindingStage = MGL_RENDER_BINDING_STAGE_FRAGMENT;
        } else {
            return false;
        }
        if (!mglBatchReplayCollectResourceBinding(
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
        _commandState.currentRenderEncoderOwner;
    if (!mglBatchReplayHasActiveEncoder(encCtx)) {
        return false;
    }

    VertexArray dynamic_vao;
    VertexArray *base_vao = MGL_STATE(glm_ctx)->vao;
    VertexArray *draw_vao = base_vao;
    if (cmd->dynamic_vertex_binding_count > 0) {
        if (!base_vao || base_vao->magic != MGL_VAO_MAGIC ||
            !mglBuildDynamicVertexArray(base_vao, cmd, &dynamic_vao)) {
            return false;
        }
        draw_vao = &dynamic_vao;
    }

    for (uint8_t i = 0; i < cmd->dynamic_uniform_binding_count; i++) {
        const MGLDynamicUniformBinding *override =
            &cmd->dynamic_uniform_bindings[i];
        if (override->binding_index >= MAX_BINDABLE_BUFFERS) {
            return false;
        }
        BufferBaseTarget *slot =
            &MGL_STATE(glm_ctx)->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override->binding_index];
        if (!slot->buf) {
            return false;
        }
        slot->offset = override->offset;
        slot->size = override->size;
    }

    bool touched_texture_units[TEXTURE_UNITS] = { false };
    for (uint8_t i = 0; i < cmd->dynamic_texture_binding_count; i++) {
        const MGLDynamicTextureBinding *override =
            &cmd->dynamic_texture_bindings[i];
        if (override->unit >= TEXTURE_UNITS ||
            override->target_index >= _MAX_TEXTURE_TYPES ||
            !override->texture) {
            return false;
        }
        Texture *texture = (Texture *)override->texture;
        if (texture->index != override->target_index) {
            return false;
        }
        if (!touched_texture_units[override->unit]) {
            MGL_STATE(glm_ctx)->active_textures[override->unit] = NULL;
            touched_texture_units[override->unit] = true;
        }
        MGL_STATE(glm_ctx)->texture_units[override->unit]
            .textures[override->target_index] = texture;
        if (override->is_active) {
            MGL_STATE(glm_ctx)->active_textures[override->unit] = texture;
        }
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
            _commandState.currentRenderEncoderOwner;
        if (!direct_texture_ok) {
            direct_texture_ok = [self bindTexturesToCurrentRenderEncoder:encCtx];
            encCtx->render_encoder_owner =
                _commandState.currentRenderEncoderOwner;
            if (!direct_texture_ok) {
                direct_texture_ok =
                    [self restoreRenderEncoderAfterTextureUploadForDraw:
                        "dynamic-sampled-texture-bind"];
                encCtx->render_encoder_owner =
                    _commandState.currentRenderEncoderOwner;
                if (direct_texture_ok) {
                    direct_texture_ok =
                        [self bindTexturesToCurrentRenderEncoder:encCtx];
                    encCtx->render_encoder_owner =
                        _commandState.currentRenderEncoderOwner;
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
        _commandState.currentRenderEncoderOwner;

    bool direct_vertex_ok = cmd->dynamic_vertex_binding_count == 0 ||
        [self bindDynamicVertexArrayBuffersDirectly:draw_vao
                                            command:cmd
                                            context:glm_ctx
                                      encodeContext:encCtx];
    encCtx->render_encoder_owner =
        _commandState.currentRenderEncoderOwner;
    bool direct_uniform_ok = cmd->dynamic_uniform_binding_count == 0 ||
        [self bindDynamicUniformRangesDirectly:cmd context:glm_ctx encodeContext:encCtx];
    encCtx->render_encoder_owner =
        _commandState.currentRenderEncoderOwner;
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
        _commandState.currentRenderEncoderOwner;
    bool fallback_ok = [self mapBuffersToMTL] &&
        [self bindVertexBuffersToCurrentRenderEncoder:encCtx];
    encCtx->render_encoder_owner =
        _commandState.currentRenderEncoderOwner;
    if (fallback_ok && cmd->dynamic_uniform_binding_count > 0) {
        fallback_ok = [self bindFragmentBuffersToCurrentRenderEncoder:encCtx];
        encCtx->render_encoder_owner =
            _commandState.currentRenderEncoderOwner;
    }
    MGL_STATE(glm_ctx)->vao = saved_vao;
    return fallback_ok;
}


- (BOOL)tryReplaySimpleBatch:(MGLDrawBatch *)batch
                            context:(GLMContext)glm_ctx
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!mglBatchReplayHasActiveEncoder(encCtx) || !batch ||
        batch->command_count == 0 ||
        batch->command_count > MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS) {
        return NO;
    }
    Program *batchProgram =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    if (batchProgram && batchProgram->uses_cull_distance) {
        return NO;
    }
    if (MGL_STATE(glm_ctx)->caps.primitive_restart) {
        return NO;
    }
    if (batch->has_dynamic_vertex_bindings ||
        batch->has_dynamic_uniform_bindings ||
        batch->has_dynamic_texture_bindings ||
        batch->has_sampler_snapshots || batch->sampler_snapshots_mixed) {
        return NO;
    }
    if (batch->key.primitive_type == 0xFFu) {
        return NO;
    }

    GLenum batchMode = batch->commands[0].mode;
    if (mglPolygonModePointForDrawMode(glm_ctx, batchMode) ||
        batchMode == GL_TRIANGLE_FAN || batchMode == GL_LINE_LOOP ||
        batchMode == GL_QUADS) {
        return NO;
    }

    MGLRenderReplayBatchCommand cmds[MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS];
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        MGLRenderReplayBatchCommand *out = &cmds[i];
        *out = (MGLRenderReplayBatchCommand){
            .cmd_type = (uint32_t)cmd->type,
            .first = cmd->first,
            .count = (uint32_t)cmd->count,
            .instance_count = (uint32_t)cmd->instanceCount,
            .base_vertex = cmd->baseVertex,
            .base_instance = cmd->baseInstance,
        };
        switch (cmd->type) {
            case MGL_CMD_DRAW_ARRAYS:
            case MGL_CMD_DRAW_ARRAYS_INSTANCED:
            case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
                break;
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
        _commandState.currentRenderEncoderOwner;

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
            _commandState.currentRenderEncoderOwner;
        Program *batchProgram =
            mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
        BOOL capturedCullDistances = NO;
        if (batchProgram && batchProgram->uses_cull_distance &&
            (cmd->type == MGL_CMD_DRAW_ARRAYS ||
             cmd->type == MGL_CMD_DRAW_ARRAYS_INSTANCED ||
             cmd->type == MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE)) {
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
                    _commandState.currentRenderEncoderOwner) == 0) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
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
                _commandState.currentRenderEncoderOwner;
        }
        if (![self applyDynamicBindingsForCommand:cmd context:glm_ctx encodeContext:&liveEncCtx]) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
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
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"sampler_snapshot"];
            continue;
        }
        GLenum mode = cmd->mode;
        GLsizei count = cmd->count;
        GLsizei instanceCount = cmd->instanceCount;

        BOOL polygonModePoint = mglPolygonModePointForDrawMode(glm_ctx, mode);
        BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
        BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
        BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
        uint32_t primType = polygonModePoint
            ? MGL_DRAW_PRIMITIVE_POINT
            : (emulateTriangleFan ? MGL_DRAW_PRIMITIVE_TRIANGLE
                                  : (emulateLineLoop ? MGL_DRAW_PRIMITIVE_LINE_STRIP
                                                     : (emulateQuads ? MGL_DRAW_PRIMITIVE_TRIANGLE
                                                                    : (uint32_t)batch->key.primitive_type)));
        if (!polygonModePoint && !emulateTriangleFan && !emulateLineLoop && !emulateQuads &&
            batch->key.primitive_type == 0xFFu) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_unsupported_primitive"];
            continue;
        }

        switch (cmd->type) {
            case MGL_CMD_DRAW_ARRAYS:
                [self issueDirectBatchDrawArrays:batch
                                         command:cmd
                                          context:glm_ctx
                                       batchIndex:i
                                             mode:mode
                                            count:count
                               polygonModePoint:polygonModePoint
                              emulateTriangleFan:emulateTriangleFan
                                 emulateLineLoop:emulateLineLoop
                                   emulateQuads:emulateQuads
                                        primType:primType
                                   encodeContext:&liveEncCtx];
                break;

            case MGL_CMD_DRAW_ARRAYS_INSTANCED:
                [self issueDirectBatchDrawArraysInstanced:batch
                                                  command:cmd
                                                   context:glm_ctx
                                                batchIndex:i
                                                      mode:mode
                                                    count:count
                                            instanceCount:instanceCount
                                       polygonModePoint:polygonModePoint
                              emulateTriangleFan:emulateTriangleFan
                                 emulateLineLoop:emulateLineLoop
                                   emulateQuads:emulateQuads
                                        primType:primType
                                   encodeContext:&liveEncCtx];
                break;

            case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
                [self issueDirectBatchDrawArraysInstancedBaseInstance:batch
                                                              command:cmd
                                                               context:glm_ctx
                                                            batchIndex:i
                                                                  mode:mode
                                                                count:count
                                                        instanceCount:instanceCount
                                                   polygonModePoint:polygonModePoint
                                                  emulateTriangleFan:emulateTriangleFan
                                                     emulateLineLoop:emulateLineLoop
                                                       emulateQuads:emulateQuads
                                                            primType:primType
                                                       encodeContext:&liveEncCtx];
                break;

            default:
                [self issueDirectBatchElementDraw:batch
                                          command:cmd
                                           context:glm_ctx
                                        batchIndex:i
                                              mode:mode
                                            count:count
                                    instanceCount:instanceCount
                               polygonModePoint:polygonModePoint
                              emulateTriangleFan:emulateTriangleFan
                                 emulateLineLoop:emulateLineLoop
                                   emulateQuads:emulateQuads
                                        primType:primType
                                   encodeContext:&liveEncCtx];
                break;
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

    if (mode == GL_TRIANGLE_STRIP && count >= 3) {
        NSUInteger stripIndexCount = 0u;
        id stripIndexBuffer =
            mglNewTriangleStripArrayIndexBuffer(
                _device, (NSUInteger)count, &stripIndexCount);
        if (!stripIndexBuffer || stripIndexCount == 0u) {
            return YES;
        }
        for (NSUInteger primitive = 0u;
             primitive * 3u < stripIndexCount; primitive++) {
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
            mglBatchReplayDrawIndexedPrimitives(
                encCtx->render_encoder_owner,
                MGL_DRAW_PRIMITIVE_TRIANGLE, 3u,
                MGL_DRAW_INDEX_UINT32, stripIndexBuffer,
                primitive * 3u * sizeof(uint32_t), instanceCount, first,
                baseInstance);
        }
        return YES;
    }

    if (mode == GL_TRIANGLE_FAN && count >= 3) {
        NSUInteger fanIndexCount = 0u;
        id fanIndexBuffer =
            mglNewTriangleFanArrayIndexBuffer(
                _device, (NSUInteger)count, &fanIndexCount);
        if (!fanIndexBuffer || fanIndexCount == 0u) {
            return YES;
        }
        const NSUInteger primitiveCount = fanIndexCount / 3u;
        for (NSUInteger primitive = 0u;
             primitive < primitiveCount; primitive++) {
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
            mglBatchReplayDrawIndexedPrimitives(
                encCtx->render_encoder_owner,
                MGL_DRAW_PRIMITIVE_TRIANGLE, 3u,
                MGL_DRAW_INDEX_UINT32, fanIndexBuffer,
                primitive * 3u * sizeof(uint32_t), instanceCount, first,
                baseInstance);
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
            mglBatchReplayDrawPrimitives(
                encCtx->render_encoder_owner,
                MGL_DRAW_PRIMITIVE_LINE, first + primitive, 2u,
                instanceCount, baseInstance);
        }
        return YES;
    }

    if (mode == GL_LINE_LOOP && count >= 2) {
        NSUInteger loopIndexCount = 0u;
        id loopIndexBuffer =
            mglNewLineLoopArrayIndexBuffer(
                _device, (NSUInteger)first, (NSUInteger)count,
                &loopIndexCount);
        if (!loopIndexBuffer || loopIndexCount == 0u) {
            return YES;
        }
        for (NSUInteger primitive = 0u;
             primitive + 1u < loopIndexCount; primitive++) {
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
            mglBatchReplayDrawIndexedPrimitives(
                encCtx->render_encoder_owner,
                MGL_DRAW_PRIMITIVE_LINE, 2u,
                MGL_DRAW_INDEX_UINT32, loopIndexBuffer,
                primitive * sizeof(uint32_t), instanceCount, 0,
                baseInstance);
        }
        return YES;
    }

    return NO;
}

- (void)issueDirectBatchDrawArrays:(MGLDrawBatch *)batch
                           command:(MGLDrawCommand *)cmd
                            context:(GLMContext)glm_ctx
                         batchIndex:(uint32_t)i
                               mode:(GLenum)mode
                              count:(GLsizei)count
                  polygonModePoint:(BOOL)polygonModePoint
                 emulateTriangleFan:(BOOL)emulateTriangleFan
                    emulateLineLoop:(BOOL)emulateLineLoop
                      emulateQuads:(BOOL)emulateQuads
                           primType:(uint32_t)primType
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!polygonModePoint &&
        [self issueDirectBatchCullDistanceArrayDraw:mode
                                              first:cmd->first
                                              count:count
                                      instanceCount:1
                                       baseInstance:0u
                                      encodeContext:encCtx]) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_cull_distance_split"];
        return;
    }
    if (polygonModePoint) {
        mglEncodeArrayPolygonPointForRenderEncoderOwner(encCtx->render_encoder_owner, _device,
            mode, cmd->first, count, 1u, 0u, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_polygon_point"];
    } else if (emulateTriangleFan) {
        if (count >= 3) {
            NSUInteger fanCount = 0;
            id fanBuf = mglNewTriangleFanArrayIndexBuffer(
                _device, (NSUInteger)count, &fanCount);
            if (fanBuf && fanCount > 0) {
                mglBatchReplayDrawIndexedPrimitives(
                    encCtx->render_encoder_owner,
                    MGL_DRAW_PRIMITIVE_TRIANGLE, fanCount,
                    MGL_DRAW_INDEX_UINT32, fanBuf, 0, 1, cmd->first, 0);
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_triangle_fan"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_triangle_fan_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_triangle_fan_small"];
        }
    } else if (emulateLineLoop) {
        if (count >= 2) {
            NSUInteger loopCount = 0;
            id loopBuf = mglNewLineLoopArrayIndexBuffer(
                _device, (NSUInteger)cmd->first, (NSUInteger)count, &loopCount);
            if (loopBuf && loopCount > 0) {
                mglBatchReplayDrawIndexedPrimitives(
                    encCtx->render_encoder_owner,
                    MGL_DRAW_PRIMITIVE_LINE_STRIP, loopCount,
                    MGL_DRAW_INDEX_UINT32, loopBuf, 0, 1, 0, 0);
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_line_loop"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_line_loop_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_line_loop_small"];
        }
    } else if (emulateQuads) {
        BOOL ok = mglEncodeArrayQuadsForRenderEncoderOwner(encCtx->render_encoder_owner, _device, count,
            cmd->first, 1u, 0u,
            mglPolygonModeLineForDrawMode(glm_ctx, mode), "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_arrays_quads" : "direct_arrays_quads_buffer")];
    } else {
        mglTraceLog("DIRECT_BATCH_DRAW_ARRAYS_SUBMIT flush=%llu batch=%u cmd=%u program=%u mode=0x%x first=%d count=%d encoder=%p pipeline=%p",
                    (unsigned long long)_commandState.traceReplayFlushId,
                    (unsigned)_commandState.traceReplayBatchIndex,
                    (unsigned)i,
                    (unsigned)mglCurrentRenderProgramKey(glm_ctx),
                    (unsigned)mode,
                    (int)cmd->first,
                    (int)count,
                    mglBatchReplayEncoderTraceToken(encCtx),
                    _pipelineCacheState.pipelineState);
        /* Cull distance emulation: bind vertex/params buffers before
         * array draw in the deferred batch path. */
        {
            Program *batchProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
            if (batchProgram && batchProgram->uses_cull_distance) {
                [self bindCullDistanceEmulationBuffers:mode
                                            firstVertex:(GLuint)cmd->first
                                       explicitVertices:NULL
                                     explicitVertexCount:0u
                                          encodeContext:encCtx];
            }
        }
        mglBatchReplayDrawPrimitives(
            encCtx->render_encoder_owner, primType,
            cmd->first, count, 1, 0);
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays"];
    }
}

- (void)issueDirectBatchDrawArraysInstanced:(MGLDrawBatch *)batch
                                      command:(MGLDrawCommand *)cmd
                                       context:(GLMContext)glm_ctx
                                    batchIndex:(uint32_t)i
                                          mode:(GLenum)mode
                                        count:(GLsizei)count
                                instanceCount:(GLsizei)instanceCount
                           polygonModePoint:(BOOL)polygonModePoint
                          emulateTriangleFan:(BOOL)emulateTriangleFan
                             emulateLineLoop:(BOOL)emulateLineLoop
                               emulateQuads:(BOOL)emulateQuads
                                    primType:(uint32_t)primType
                               encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!polygonModePoint &&
        [self issueDirectBatchCullDistanceArrayDraw:mode
                                              first:cmd->first
                                              count:count
                                      instanceCount:instanceCount
                                       baseInstance:0u
                                      encodeContext:encCtx]) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_instanced_cull_distance_split"];
        return;
    }
    if (polygonModePoint) {
        mglEncodeArrayPolygonPointForRenderEncoderOwner(encCtx->render_encoder_owner, _device,
            mode, cmd->first, count, instanceCount, 0u, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_instanced_polygon_point"];
    } else if (emulateTriangleFan) {
        if (count >= 3) {
            NSUInteger fanCount = 0;
            id fanBuf = mglNewTriangleFanArrayIndexBuffer(
                _device, (NSUInteger)count, &fanCount);
            if (fanBuf && fanCount > 0) {
                mglBatchReplayDrawIndexedPrimitives(
                    encCtx->render_encoder_owner,
                    MGL_DRAW_PRIMITIVE_TRIANGLE, fanCount,
                    MGL_DRAW_INDEX_UINT32, fanBuf, 0, instanceCount,
                    cmd->first, 0);
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_instanced_triangle_fan"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_instanced_triangle_fan_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_instanced_triangle_fan_small"];
        }
    } else if (emulateLineLoop) {
        if (count >= 2) {
            NSUInteger loopCount = 0;
            id loopBuf = mglNewLineLoopArrayIndexBuffer(
                _device, (NSUInteger)cmd->first, (NSUInteger)count, &loopCount);
            if (loopBuf && loopCount > 0) {
                mglBatchReplayDrawIndexedPrimitives(
                    encCtx->render_encoder_owner,
                    MGL_DRAW_PRIMITIVE_LINE_STRIP, loopCount,
                    MGL_DRAW_INDEX_UINT32, loopBuf, 0, instanceCount, 0, 0);
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_instanced_line_loop"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_instanced_line_loop_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_instanced_line_loop_small"];
        }
    } else if (emulateQuads) {
        BOOL ok = mglEncodeArrayQuadsForRenderEncoderOwner(encCtx->render_encoder_owner, _device, count,
            cmd->first, instanceCount, 0u,
            mglPolygonModeLineForDrawMode(glm_ctx, mode), "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_arrays_instanced_quads" : "direct_arrays_instanced_quads_buffer")];
    } else {
        /* Cull distance emulation: bind vertex/params buffers before
         * array draw in the deferred batch path. */
        {
            Program *batchProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
            if (batchProgram && batchProgram->uses_cull_distance) {
                [self bindCullDistanceEmulationBuffers:mode
                                            firstVertex:(GLuint)cmd->first
                                       explicitVertices:NULL
                                     explicitVertexCount:0u
                                          encodeContext:encCtx];
            }
        }
        mglBatchReplayDrawPrimitives(
            encCtx->render_encoder_owner, primType,
            cmd->first, count, instanceCount, 0);
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_instanced"];
    }
}

- (void)issueDirectBatchDrawArraysInstancedBaseInstance:(MGLDrawBatch *)batch
                                                   command:(MGLDrawCommand *)cmd
                                                    context:(GLMContext)glm_ctx
                                                 batchIndex:(uint32_t)i
                                                       mode:(GLenum)mode
                                                     count:(GLsizei)count
                                             instanceCount:(GLsizei)instanceCount
                                        polygonModePoint:(BOOL)polygonModePoint
                                       emulateTriangleFan:(BOOL)emulateTriangleFan
                                          emulateLineLoop:(BOOL)emulateLineLoop
                                            emulateQuads:(BOOL)emulateQuads
                                                 primType:(uint32_t)primType
                                            encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!polygonModePoint &&
        [self issueDirectBatchCullDistanceArrayDraw:mode
                                              first:cmd->first
                                              count:count
                                      instanceCount:instanceCount
                                       baseInstance:cmd->baseInstance
                                      encodeContext:encCtx]) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_base_instance_cull_distance_split"];
        return;
    }
    if (polygonModePoint) {
        mglEncodeArrayPolygonPointForRenderEncoderOwner(encCtx->render_encoder_owner, _device,
            mode, cmd->first, count, instanceCount, cmd->baseInstance,
            "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_base_instance_polygon_point"];
    } else if (emulateTriangleFan) {
        if (count >= 3) {
            NSUInteger fanCount = 0;
            id fanBuf = mglNewTriangleFanArrayIndexBuffer(
                _device, (NSUInteger)count, &fanCount);
            if (fanBuf && fanCount > 0) {
                mglBatchReplayDrawIndexedPrimitives(
                    encCtx->render_encoder_owner,
                    MGL_DRAW_PRIMITIVE_TRIANGLE, fanCount,
                    MGL_DRAW_INDEX_UINT32, fanBuf, 0, instanceCount,
                    cmd->first, cmd->baseInstance);
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_base_instance_triangle_fan"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_base_instance_triangle_fan_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_base_instance_triangle_fan_small"];
        }
    } else if (emulateLineLoop) {
        if (count >= 2) {
            NSUInteger loopCount = 0;
            id loopBuf = mglNewLineLoopArrayIndexBuffer(
                _device, (NSUInteger)cmd->first, (NSUInteger)count, &loopCount);
            if (loopBuf && loopCount > 0) {
                mglBatchReplayDrawIndexedPrimitives(
                    encCtx->render_encoder_owner,
                    MGL_DRAW_PRIMITIVE_LINE_STRIP, loopCount,
                    MGL_DRAW_INDEX_UINT32, loopBuf, 0, instanceCount, 0,
                    cmd->baseInstance);
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_base_instance_line_loop"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_commandState.traceReplayFlushId
                              batchIndex:_commandState.traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_base_instance_line_loop_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_base_instance_line_loop_small"];
        }
    } else if (emulateQuads) {
        BOOL ok = mglEncodeArrayQuadsForRenderEncoderOwner(encCtx->render_encoder_owner, _device, count,
            cmd->first, instanceCount, cmd->baseInstance,
            mglPolygonModeLineForDrawMode(glm_ctx, mode), "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_arrays_base_instance_quads" : "direct_arrays_base_instance_quads_buffer")];
    } else {
        /* Cull distance emulation: bind vertex/params buffers before
         * array draw in the deferred batch path. */
        {
            Program *batchProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
            if (batchProgram && batchProgram->uses_cull_distance) {
                [self bindCullDistanceEmulationBuffers:mode
                                            firstVertex:(GLuint)cmd->first
                                       explicitVertices:NULL
                                     explicitVertexCount:0u
                                          encodeContext:encCtx];
            }
        }
        mglBatchReplayDrawPrimitives(
            encCtx->render_encoder_owner, primType,
            cmd->first, count, instanceCount,
            cmd->baseInstance);
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_base_instance"];
    }
}

- (void)issueDirectBatchElementDraw:(MGLDrawBatch *)batch
                           command:(MGLDrawCommand *)cmd
                            context:(GLMContext)glm_ctx
                         batchIndex:(uint32_t)i
                               mode:(GLenum)mode
                              count:(GLsizei)count
                      instanceCount:(GLsizei)instanceCount
                 polygonModePoint:(BOOL)polygonModePoint
                emulateTriangleFan:(BOOL)emulateTriangleFan
                   emulateLineLoop:(BOOL)emulateLineLoop
                     emulateQuads:(BOOL)emulateQuads
                          primType:(uint32_t)primType
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
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
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
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
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
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_elements_cull_distance_split"];
        return;
    }

    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDrawForRenderEncoderOwner(encCtx->render_encoder_owner,
                                               _device,
                                               glm_ctx,
                                               glBuf,
                                               idxBuf,
                                               mode,
                                               primType,
                                               cmd->indexType,
                                               mtlIdxType,
                                               idxOffset,
                                               count,
                                               instanceCount,
                                               cmd->baseVertex,
                                               cmd->baseInstance,
                                               "directBatch");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:(restartResult == MGLPrimitiveRestartEncodeHandled ? "SUBMIT" : "SKIP")
                          reason:"direct_primitive_restart"];
        return;
    }

    if (polygonModePoint) {
        mglEncodeElementPolygonPointForRenderEncoderOwner(encCtx->render_encoder_owner, _device,
            glBuf, idxBuf, mode, cmd->indexType, mtlIdxType, idxOffset,
            count, instanceCount, cmd->baseVertex, cmd->baseInstance,
            "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_elements_polygon_point"];
    } else if (emulateTriangleFan) {
        mglEncodeElementTriangleFanForRenderEncoderOwner(encCtx->render_encoder_owner, _device,
            glBuf, idxBuf, cmd->indexType, idxOffset, count, instanceCount,
            cmd->baseVertex, cmd->baseInstance, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:(count >= 3 ? "SUBMIT" : "SKIP")
                          reason:"direct_elements_triangle_fan"];
    } else if (emulateLineLoop) {
        mglEncodeElementLineLoopForRenderEncoderOwner(encCtx->render_encoder_owner, _device,
            glBuf, idxBuf, cmd->indexType, idxOffset, count, instanceCount,
            cmd->baseVertex, cmd->baseInstance, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                         phase:(count >= 2 ? "SUBMIT" : "SKIP")
                          reason:"direct_elements_line_loop"];
    } else if (emulateQuads) {
        BOOL ok = mglEncodeElementQuadsForRenderEncoderOwner(encCtx->render_encoder_owner, _device,
            glBuf, idxBuf, cmd->indexType, idxOffset, count, instanceCount,
            cmd->baseVertex, cmd->baseInstance,
            mglPolygonModeLineForDrawMode(glm_ctx, mode), "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_elements_quads" : "direct_elements_quads_buffer")];
    } else {
        uint64_t drawIndexType = mtlIdxType;
        id drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                      glBuf,
                                                                      idxBuf,
                                                                      cmd->indexType,
                                                                      &idxOffset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer || (GLuint)drawIndexType == 0xFFFFFFFF) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_commandState.traceReplayFlushId
                          batchIndex:_commandState.traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_prepared_index"];
            return;
        }
        mglBatchReplayDrawIndexedPrimitives(
            encCtx->render_encoder_owner, primType,
            count, drawIndexType, drawIndexBuffer,
            idxOffset, instanceCount, cmd->baseVertex, cmd->baseInstance);
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_commandState.traceReplayFlushId
                      batchIndex:_commandState.traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_elements"];
    }
}




@end
