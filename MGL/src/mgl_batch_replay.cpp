/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * O2.3/O2.5: BatchReplay stage/bind + MDI/direct plans — domain ownership.
 */

#include "mgl_batch_replay.h"
#include "mgl_renderer_ports.h"  /* mglRendererBindMTLTexturePort */
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"

#include "mgl_render.h"
#include "mgl_types_texture.h"
#include "mgl_vertex_attrib_query.h"
#include "mgl_vertex_format.h"

#include <string.h>
#include <limits.h>

extern "C" bool mgl_batch_replay_collect_resource_binding(
    MGLRenderResourceBindingSnapshot *snapshot, uint32_t stage, uint32_t kind,
    void *resource, uint32_t index)
{
    if (!snapshot || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        kind > MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return false;
    }
    uint32_t *count = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                          ? &snapshot->vertex_op_count
                          : &snapshot->fragment_op_count;
    MGLRenderResourceBindingOp *ops = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                                          ? snapshot->vertex_ops
                                          : snapshot->fragment_ops;
    if (*count >= MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        return false;
    }
    ops[(*count)++] = MGLRenderResourceBindingOp{
        .kind = kind,
        .index = index,
        .resource = resource,
    };
    return true;
}

extern "C" bool mgl_batch_replay_build_dynamic_vertex_array(
    GLMContext ctx, const VertexArray *base, const MGLDrawCommand *cmd,
    VertexArray *out)
{
    if (!ctx || !base || !cmd || !out ||
        cmd->dynamic_vertex_binding_count > MGL_MAX_DYNAMIC_VERTEX_BINDINGS) {
        return false;
    }

    *out = *base;
    for (uint8_t i = 0; i < cmd->dynamic_vertex_binding_count; i++) {
        const MGLDynamicVertexBinding *override_binding =
            &cmd->dynamic_vertex_bindings[i];
        if (!override_binding->buffer_name ||
            override_binding->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
            return false;
        }
        Buffer *resolved = mglNamedBuffer(ctx, override_binding->buffer_name);
        if (!resolved) {
            return false;
        }
        BufferBinding *binding = &out->bindings[override_binding->binding_index];
        binding->buffer = resolved;
        binding->offset = (GLintptr)override_binding->offset;
        /* Keep classic VertexAttribPointer mirror fields in sync so resolve
         * stays correct if a path still reads attrib.binding_offset. */
        for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
            if ((out->enabled_attribs & (1u << attrib)) == 0u ||
                out->attrib[attrib].buffer_bindingindex !=
                    override_binding->binding_index) {
                continue;
            }
            out->attrib[attrib].buffer = resolved;
            out->attrib[attrib].binding_offset =
                (GLintptr)override_binding->offset;
        }
    }
    return true;
}

extern "C" bool mgl_batch_replay_attrib_can_bind_directly(
    Program *active_program, GLuint attrib_index, const VertexAttrib *attrib)
{
    if (!attrib ||
        mglRenderAttribNeedsConversion(attrib->long_attribute ? 1 : 0,
                                       (uint32_t)attrib->type,
                                       attrib->integer ? 1 : 0)) {
        return false;
    }
    if (!attrib->integer) {
        return true;
    }

    MGLShaderResource *resource =
        mglRendererProgramVertexAttribResource(active_program, attrib_index);
    GLuint shader_type = resource ? resource->gl_type : 0u;
    return !mglIntegerAttribNeedsConversion(attrib->type, shader_type,
                                            attrib->size, NULL);
}

extern "C" bool mgl_batch_replay_apply_uniform_range_overrides(
    GLMContext ctx, const MGLDrawCommand *cmd)
{
    if (!ctx || !cmd || !ctx->active_state) {
        return false;
    }
    for (uint8_t i = 0; i < cmd->dynamic_uniform_binding_count; i++) {
        const MGLDynamicUniformBinding *override_binding =
            &cmd->dynamic_uniform_bindings[i];
        if (override_binding->binding_index >= MAX_BINDABLE_BUFFERS) {
            return false;
        }
        BufferBaseTarget *slot =
            &ctx->active_state->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override_binding->binding_index];
        if (!slot->buf) {
            return false;
        }
        slot->offset = override_binding->offset;
        slot->size = override_binding->size;
    }
    return true;
}

extern "C" bool mgl_batch_replay_apply_texture_overrides(
    GLMContext ctx, const MGLDrawCommand *cmd, bool *touched_units,
    uint32_t touched_units_count)
{
    if (!ctx || !cmd || !ctx->active_state || !touched_units ||
        touched_units_count < (uint32_t)TEXTURE_UNITS) {
        return false;
    }

    for (uint8_t i = 0; i < cmd->dynamic_texture_binding_count; i++) {
        const MGLDynamicTextureBinding *override_binding =
            &cmd->dynamic_texture_bindings[i];
        if (override_binding->unit >= TEXTURE_UNITS ||
            override_binding->target_index >= _MAX_TEXTURE_TYPES ||
            !override_binding->texture_name) {
            return false;
        }
        Texture *texture = mglNamedTexture(ctx, override_binding->texture_name);
        if (!texture) {
            return false;
        }
        if (texture->index != override_binding->target_index) {
            return false;
        }
        if (!touched_units[override_binding->unit]) {
            ctx->active_state->active_textures[override_binding->unit] = NULL;
            touched_units[override_binding->unit] = true;
        }
        ctx->active_state->texture_units[override_binding->unit]
            .textures[override_binding->target_index] = texture;
        if (override_binding->is_active) {
            ctx->active_state->active_textures[override_binding->unit] =
                texture;
        }
    }
    return true;
}

extern "C" int mgl_batch_replay_mdi_gate(const MGLDrawBatch *batch,
                                         int disable_mdi, size_t *arg_size,
                                         size_t *needed_bytes)
{
    if (!batch || batch->command_count == 0u) {
        return MGL_BATCH_MDI_FALLBACK_EMPTY;
    }
    if (disable_mdi) {
        return MGL_BATCH_MDI_FALLBACK_DISABLED;
    }
    if (batch->key.primitive_type == 0xFFu) {
        return MGL_BATCH_MDI_FALLBACK_BAD_PRIM;
    }
    const size_t asz =
        batch->uses_elements ? sizeof(MGLDrawIndexedPrimitivesIndirectArguments)
                             : sizeof(MGLDrawPrimitivesIndirectArguments);
    if (batch->command_count > (UINT32_MAX / asz)) {
        return MGL_BATCH_MDI_FALLBACK_OVERFLOW;
    }
    if (arg_size) {
        *arg_size = asz;
    }
    if (needed_bytes) {
        *needed_bytes = asz * (size_t)batch->command_count;
    }
    return MGL_BATCH_MDI_OK;
}

extern "C" const char *mgl_batch_replay_mdi_gate_reason(int gate)
{
    switch (gate) {
    case MGL_BATCH_MDI_FALLBACK_DISABLED:
        return "mdi_disabled";
    case MGL_BATCH_MDI_FALLBACK_BAD_PRIM:
        return "mdi_unsupported_primitive";
    case MGL_BATCH_MDI_FALLBACK_OVERFLOW:
        return "mdi_args_overflow";
    case MGL_BATCH_MDI_FALLBACK_EMPTY:
        return "mdi_empty";
    default:
        return "mdi_ok";
    }
}

extern "C" int mgl_batch_replay_fill_mdi_indexed_args(
    const MGLDrawBatch *batch, MGLDrawIndexedPrimitivesIndirectArguments *args)
{
    if (!batch || !args || !batch->uses_elements || batch->command_count == 0u) {
        return 0;
    }
    const GLenum glIdxType = batch->commands[0].indexType;
    for (uint32_t i = 0; i < batch->command_count; i++) {
        const MGLDrawCommand *cmd = &batch->commands[i];
        if (cmd->indexType != glIdxType) {
            return 0;
        }
        args[i].indexCount = (uint32_t)cmd->count;
        args[i].instanceCount = (uint32_t)cmd->instanceCount;
        args[i].indexStart = 0u;
        args[i].baseVertex = cmd->baseVertex;
        args[i].baseInstance = cmd->baseInstance;
    }
    return 1;
}

extern "C" void mgl_batch_replay_fill_mdi_array_args(
    const MGLDrawBatch *batch, MGLDrawPrimitivesIndirectArguments *args)
{
    if (!batch || !args) {
        return;
    }
    for (uint32_t i = 0; i < batch->command_count; i++) {
        const MGLDrawCommand *cmd = &batch->commands[i];
        args[i].vertexCount = (uint32_t)cmd->count;
        args[i].instanceCount = (uint32_t)cmd->instanceCount;
        args[i].vertexStart = (uint32_t)cmd->first;
        args[i].baseInstance = cmd->baseInstance;
    }
}

extern "C" void mgl_batch_replay_fill_stream_mdi_indexed_args(
    const MGLDrawBatch *batch, MGLDrawIndexedPrimitivesIndirectArguments *args)
{
    if (!batch || !args) {
        return;
    }
    for (uint32_t i = 0; i < batch->command_count; i++) {
        const MGLDrawCommand *cmd = &batch->commands[i];
        args[i].indexCount = (uint32_t)cmd->count;
        args[i].instanceCount =
            (uint32_t)(cmd->instanceCount > 0 ? cmd->instanceCount : 1);
        args[i].indexStart = 0u;
        args[i].baseVertex = 0;
        args[i].baseInstance = cmd->baseInstance;
    }
}

extern "C" int mgl_batch_replay_simple_eligible(
    const MGLDrawBatch *batch, uint32_t max_commands, int has_active_encoder,
    int uses_cull_distance, int primitive_restart, int polygon_mode_point,
    int mode_needs_emulate)
{
    if (!has_active_encoder || !batch || batch->command_count == 0u ||
        batch->command_count > max_commands) {
        return 0;
    }
    if (uses_cull_distance || primitive_restart) {
        return 0;
    }
    if (batch->has_dynamic_vertex_bindings ||
        batch->has_dynamic_uniform_bindings ||
        batch->has_dynamic_texture_bindings || batch->has_sampler_snapshots ||
        batch->sampler_snapshots_mixed) {
        return 0;
    }
    if (batch->key.primitive_type == 0xFFu) {
        return 0;
    }
    if (polygon_mode_point || mode_needs_emulate) {
        return 0;
    }
    return 1;
}

extern "C" void mgl_batch_replay_direct_prim_plan(
    uint32_t mode, int polygon_mode_point, uint32_t batch_primitive_type,
    MGLBatchReplayDirectPrimPlan *out)
{
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    out->polygon_mode_point = polygon_mode_point ? 1u : 0u;
    out->emulate_triangle_fan =
        mglRenderEmulateTriangleFan(mode, polygon_mode_point) ? 1u : 0u;
    out->emulate_line_loop = mglRenderEmulateLineLoop(mode) ? 1u : 0u;
    out->emulate_quads =
        mglRenderEmulateQuads(mode, polygon_mode_point) ? 1u : 0u;
    if (out->polygon_mode_point) {
        out->prim_type = MGL_DRAW_PRIMITIVE_POINT;
    } else if (out->emulate_triangle_fan) {
        out->prim_type = MGL_DRAW_PRIMITIVE_TRIANGLE;
    } else if (out->emulate_line_loop) {
        out->prim_type = MGL_DRAW_PRIMITIVE_LINE_STRIP;
    } else if (out->emulate_quads) {
        out->prim_type = MGL_DRAW_PRIMITIVE_TRIANGLE;
    } else {
        out->prim_type = batch_primitive_type;
        if (batch_primitive_type == 0xFFu) {
            out->skip_unsupported_prim = 1u;
        }
    }
}

extern "C" int mgl_batch_replay_icb_gate(const MGLDrawBatch *batch,
                                        int has_device, int has_encoder,
                                        int icb_enable, int icb_disable)
{
    if (!batch || batch->command_count == 0u || !has_device || !has_encoder) {
        return MGL_BATCH_ICB_UNAVAILABLE;
    }
    if (batch->key.primitive_type == 0xFFu) {
        return MGL_BATCH_ICB_BAD_PRIM;
    }
    if (!icb_enable || icb_disable) {
        return MGL_BATCH_ICB_DISABLED;
    }
    return MGL_BATCH_ICB_OK;
}

extern "C" const char *mgl_batch_replay_icb_gate_reason(int gate)
{
    switch (gate) {
    case MGL_BATCH_ICB_UNAVAILABLE:
        return "icb_unavailable";
    case MGL_BATCH_ICB_BAD_PRIM:
        return "icb_unsupported_primitive";
    case MGL_BATCH_ICB_DISABLED:
        return "icb_disabled";
    default:
        return "icb_ok";
    }
}

extern "C" int mgl_batch_replay_stream_path(const MGLDrawBatch *batch,
                                            int disable_mdi)
{
    if (!batch || !batch->stream_merged || batch->stream_index_count == 0) {
        return MGL_BATCH_STREAM_EMPTY;
    }
    if (batch->key.primitive_type == 0xFFu) {
        return MGL_BATCH_STREAM_BAD_PRIM;
    }
    if (!disable_mdi) {
        return MGL_BATCH_STREAM_TRY_MDI;
    }
    return MGL_BATCH_STREAM_DIRECT;
}

extern "C" const char *mgl_batch_replay_stream_path_reason(int path)
{
    switch (path) {
    case MGL_BATCH_STREAM_EMPTY:
        return "stream_empty";
    case MGL_BATCH_STREAM_BAD_PRIM:
        return "stream_unsupported_primitive";
    case MGL_BATCH_STREAM_TRY_MDI:
        return "stream_merge_to_mdi";
    case MGL_BATCH_STREAM_DIRECT:
        return "stream_direct_merged";
    default:
        return "stream_unknown";
    }
}

extern "C" void mgl_batch_replay_fill_sampler_params(
    const MGLSamplerSnapshotKey *key, TextureParameter *out)
{
    if (!key || !out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    out->min_filter = key->min_filter;
    out->mag_filter = key->mag_filter;
    out->wrap_s = key->wrap_s;
    out->wrap_t = key->wrap_t;
    out->wrap_r = key->wrap_r;
    out->compare_mode = key->compare_mode;
    out->compare_func = key->compare_func;
    out->max_anisotropy = key->max_anisotropy;
    out->min_lod = key->min_lod;
    out->max_lod = key->max_lod;
    memcpy(out->border_color, key->border_color, sizeof(out->border_color));
}

extern "C" int mgl_batch_replay_plan_dyn_vertex_streams(
    GLMContext ctx, const VertexArray *vao, Program *active_program,
    const MGLDynamicVertexBinding *override_binding,
    MGLBatchDynVertexStreamPlan *out)
{
    if (!ctx || !vao || !override_binding || !out) {
        return MGL_BATCH_DYN_VERTEX_FAIL;
    }
    memset(out, 0, sizeof(*out));
    if (!override_binding->buffer_name ||
        override_binding->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        return MGL_BATCH_DYN_VERTEX_FAIL;
    }
    Buffer *resolved = mglNamedBuffer(ctx, override_binding->buffer_name);
    if (!resolved) {
        return MGL_BATCH_DYN_VERTEX_FAIL;
    }
    const BufferBinding *binding = &vao->bindings[override_binding->binding_index];
    if (binding->buffer != resolved) {
        return MGL_BATCH_DYN_VERTEX_FAIL;
    }
    out->binding_index = override_binding->binding_index;
    out->buffer = resolved;
    out->dynamic_offset = (uint64_t)override_binding->offset;

    for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
        if ((vao->enabled_attribs & (1u << attrib)) == 0u ||
            vao->attrib[attrib].buffer_bindingindex !=
                override_binding->binding_index ||
            !mglRendererProgramUsesVertexAttrib(active_program, attrib)) {
            continue;
        }
        GLuint effective_stride = binding->stride > 0
                                      ? (GLuint)binding->stride
                                      : vao->attrib[attrib].stride;
        bool known_stream = false;
        for (uint32_t stream = 0; stream < out->stream_count; stream++) {
            if (out->representative_strides[stream] == effective_stride) {
                known_stream = true;
                break;
            }
        }
        if (!known_stream) {
            if (out->stream_count >= MGL_BATCH_DYN_VERTEX_MAX_STREAMS) {
                return MGL_BATCH_DYN_VERTEX_FAIL;
            }
            out->representative_attribs[out->stream_count] = attrib;
            out->representative_strides[out->stream_count] = effective_stride;
            out->stream_count++;
        }
    }
    if (out->stream_count == 0u) {
        return MGL_BATCH_DYN_VERTEX_UNUSED;
    }
    return MGL_BATCH_DYN_VERTEX_OK;
}

extern "C" int mgl_batch_replay_dyn_vertex_stream_can_bind_directly(
    Program *active_program, const VertexArray *vao,
    const MGLBatchDynVertexStreamPlan *plan, uint32_t stream_index)
{
    if (!vao || !plan || stream_index >= plan->stream_count ||
        plan->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        return 0;
    }
    const BufferBinding *binding = &vao->bindings[plan->binding_index];
    const uint32_t want_stride = plan->representative_strides[stream_index];
    for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
        if ((vao->enabled_attribs & (1u << attrib)) == 0u ||
            vao->attrib[attrib].buffer_bindingindex != plan->binding_index ||
            !mglRendererProgramUsesVertexAttrib(active_program, attrib)) {
            continue;
        }
        GLuint effective_stride = binding->stride > 0
                                      ? (GLuint)binding->stride
                                      : vao->attrib[attrib].stride;
        if (effective_stride == want_stride &&
            !mgl_batch_replay_attrib_can_bind_directly(active_program, attrib,
                                                       &vao->attrib[attrib])) {
            return 0;
        }
    }
    return 1;
}

extern "C" int mgl_batch_replay_uniform_range_fits(uint64_t offset,
                                                   uint64_t size,
                                                   uint64_t buffer_length)
{
    if (offset > buffer_length || size > buffer_length - offset) {
        return 0;
    }
    return 1;
}

extern "C" int mgl_batch_replay_cmd_is_array_draw(uint32_t cmd_type)
{
    switch (cmd_type) {
    case MGL_CMD_DRAW_ARRAYS:
    case MGL_CMD_DRAW_ARRAYS_INSTANCED:
    case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
        return 1;
    default:
        return 0;
    }
}

extern "C" void mgl_batch_replay_fill_simple_cmd_common(
    const MGLDrawCommand *cmd, MGLRenderReplayBatchCommand *out) /* struct tag OK */
{
    if (!cmd || !out) {
        return;
    }
    *out = MGLRenderReplayBatchCommand{
        .cmd_type = (uint32_t)cmd->type,
        .first = cmd->first,
        .count = (uint32_t)cmd->count,
        .instance_count = (uint32_t)cmd->instanceCount,
        .base_vertex = cmd->baseVertex,
        .base_instance = cmd->baseInstance,
        .index_type = 0u,
        .index_buffer_offset = 0u,
        .index_buffer = nullptr,
    };
}

extern "C" void mgl_batch_replay_copy_object_hash_tables(GLMState *dst,
                                                         const GLMState *src)
{
    if (!dst || !src) {
        return;
    }
    dst->vao_table = src->vao_table;
    dst->buffer_table = src->buffer_table;
    dst->texture_table = src->texture_table;
    dst->shader_table = src->shader_table;
    dst->program_table = src->program_table;
    dst->program_pipeline_table = src->program_pipeline_table;
    dst->transform_feedback_table = src->transform_feedback_table;
    dst->renderbuffer_table = src->renderbuffer_table;
    dst->framebuffer_table = src->framebuffer_table;
    dst->sampler_table = src->sampler_table;
}

extern "C" void mgl_batch_replay_sync_hash_tables_from_replay(
    GLMState *live, const GLMState *replay)
{
    if (!live || !replay) {
        return;
    }
    live->vao_table = replay->vao_table;
    live->buffer_table = replay->buffer_table;
    live->texture_table = replay->texture_table;
    live->shader_table = replay->shader_table;
    live->program_table = replay->program_table;
    live->program_pipeline_table = replay->program_pipeline_table;
    live->transform_feedback_table = replay->transform_feedback_table;
    live->renderbuffer_table = replay->renderbuffer_table;
    live->framebuffer_table = replay->framebuffer_table;
    live->sampler_table = replay->sampler_table;
    live->sync_table = replay->sync_table;
}

#include "mgl_renderer_backend.h"

extern "C" Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);
#include "mgl_program_resource.h"
#include "mgl_types_buffer.h"

extern "C" int mgl_batch_replay_plan_uniform_binds(
    GLMContext ctx, const MGLDrawCommand *cmd, const uint64_t *mtl_lengths,
    uint32_t mtl_lengths_count, uint64_t min_binding_bytes,
    uint32_t max_buffer_slots, MGLBatchUniformBindPlan *out)
{
    if (!ctx || !cmd || !mtl_lengths || !out) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    if (cmd->dynamic_uniform_binding_count > mtl_lengths_count) {
        return 0;
    }

    BufferMapList *stage_maps[2] = {
        &ctx->active_state->vertex_buffer_map_list,
        &ctx->active_state->fragment_buffer_map_list,
    };
    const int stages[2] = {_VERTEX_SHADER, _FRAGMENT_SHADER};

    for (uint8_t dynamic_index = 0;
         dynamic_index < cmd->dynamic_uniform_binding_count; dynamic_index++) {
        const MGLDynamicUniformBinding *override_binding =
            &cmd->dynamic_uniform_bindings[dynamic_index];
        BufferBaseTarget *slot =
            &ctx->active_state
                 ->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override_binding->binding_index];
        if (!slot->buf || override_binding->offset < 0 ||
            override_binding->size <= 0) {
            return 0;
        }
        const uint64_t mtl_len = mtl_lengths[dynamic_index];
        if (mtl_len == 0) {
            return 0;
        }
        uint64_t start = (uint64_t)override_binding->offset;
        uint64_t length = (uint64_t)override_binding->size;
        if (!mgl_batch_replay_uniform_range_fits(start, length, mtl_len)) {
            return 0;
        }

        for (int stage_index = 0; stage_index < 2; stage_index++) {
            BufferMapList *maps = stage_maps[stage_index];
            GLuint map_count = maps->count < MAX_MAPPED_BUFFERS
                                   ? maps->count
                                   : MAX_MAPPED_BUFFERS;
            for (GLuint map_index = 0; map_index < map_count; map_index++) {
                BufferMap *map = &maps->buffers[map_index];
                if (map->attribute_mask != 0u ||
                    map->buffer_base_index != override_binding->binding_index ||
                    map->buf != slot->buf) {
                    continue;
                }
                size_t reflected = map->has_metal_binding
                    ? mglRendererGetProgramBindingRequiredSize(
                          ctx, stages[stage_index], (int)map->resource_type,
                          (int)map->resource_index)
                    : mglRendererGetProgramBindingRequiredSizeForStage(
                          ctx, stages[stage_index],
                          override_binding->binding_index);
                uint64_t required = min_binding_bytes;
                if (reflected > required) {
                    required = reflected;
                }
                if (length < required) {
                    return 0;
                }
                intptr_t resolved_slot = map->has_metal_binding
                    ? (intptr_t)map->metal_binding_index
                    : mglRendererGetProgramMetalBufferIndexForStage(
                          ctx, stages[stage_index],
                          override_binding->binding_index);
                if (resolved_slot < 0 ||
                    (uint32_t)resolved_slot >= max_buffer_slots) {
                    return 0;
                }
                if (out->count >= MGL_BATCH_UNIFORM_BIND_MAX_OPS) {
                    return 0;
                }
                MGLBatchUniformBindOp *op = &out->ops[out->count++];
                op->is_vertex_stage =
                    mglRenderStageMapsVertexAttribs(stages[stage_index]) ? 1u
                                                                         : 0u;
                op->metal_slot = (uint32_t)resolved_slot;
                op->offset = start;
                op->binding_index = override_binding->binding_index;
            }
        }
    }
    return 1;
}

extern "C" int mgl_batch_replay_plan_sampled_texture_candidates(
    GLMContext ctx, MGLBatchSampledTexPlan *out)
{
    if (!ctx || !out) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    const int stages[2] = {_VERTEX_SHADER, _FRAGMENT_SHADER};
    for (int stage_index = 0; stage_index < 2; stage_index++) {
        int stage = stages[stage_index];
        Program *program = mglResolveProgramForStageFromState(ctx, stage);
        int32_t sampled_count =
            mglRendererGetProgramBindingCount(ctx, stage, _SAMPLED_IMAGE_RES);
        if (sampled_count < 0) {
            continue;
        }
        for (int32_t resource_index = 0; resource_index < sampled_count;
             resource_index++) {
            int32_t metal_slot = mglRendererGetProgramBinding(
                ctx, stage, _SAMPLED_IMAGE_RES, resource_index);
            if (metal_slot < 0 || (uint32_t)metal_slot >= TEXTURE_UNITS) {
                continue;
            }
            MGLShaderResource *resource = nullptr;
            if (program &&
                (GLuint)resource_index <
                    program->shader_resources_list[stage][_SAMPLED_IMAGE_RES]
                        .count) {
                resource = &program->shader_resources_list[stage]
                                [_SAMPLED_IMAGE_RES]
                                .list[resource_index];
            }
            if (resource && resource->is_array) {
                return 0;
            }
            if (out->count >= MGL_BATCH_SAMPLED_TEX_MAX) {
                return 0;
            }
            MGLBatchSampledTexCandidate *e = &out->entries[out->count++];
            e->stage = stage;
            e->resource_index = (uint32_t)resource_index;
            e->metal_slot = (uint32_t)metal_slot;
            e->expected_type = mglRendererGetProgramExpectedTextureType(
                ctx, stage, _SAMPLED_IMAGE_RES, resource_index);
            e->lookup_type = mglRendererGetProgramDeclaredTextureType(
                ctx, stage, _SAMPLED_IMAGE_RES, resource_index);
            e->expected_kind = mglRendererGetProgramExpectedTextureDataKind(
                ctx, stage, _SAMPLED_IMAGE_RES, resource_index);
            e->needs_combined_sampler =
                (!resource || resource->has_combined_sampler) ? 1u : 0u;
            e->resource = resource;
        }
    }
    return 1;
}

extern "C" int mgl_batch_replay_dyn_vertex_offset_ok(int64_t binding_offset,
                                                     uint64_t dynamic_offset,
                                                     uint64_t metal_length)
{
    if (binding_offset < 0) {
        return 0;
    }
    if ((uint64_t)binding_offset != dynamic_offset) {
        return 0;
    }
    if ((uint64_t)binding_offset >= metal_length || dynamic_offset >= metal_length) {
        return 0;
    }
    return 1;
}

extern "C" int mgl_batch_replay_sampler_slot_ok(uint32_t metal_slot,
                                                uint32_t max_slots)
{
    return metal_slot < max_slots;
}

extern "C" int mgl_batch_replay_cmd_is_elements_draw(uint32_t cmd_type)
{
    switch (cmd_type) {
    case MGL_CMD_DRAW_ELEMENTS:
    case MGL_CMD_DRAW_ELEMENTS_INSTANCED:
    case MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX:
    case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX:
    case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE:
    case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE:
        return 1;
    default:
        return 0;
    }
}

extern "C" int mgl_batch_replay_mtl_ptr_ok(const void *mtl_data)
{
    return mtl_data != NULL && (uintptr_t)mtl_data >= 0x10000u;
}

extern "C" int mgl_batch_replay_dyn_vertex_slot_ok(int resolved_slot,
                                                  int max_slots)
{
    return resolved_slot >= 0 && resolved_slot < max_slots;
}

extern "C" int mgl_batch_replay_sampled_tex_object_ok(int has_texture,
                                                     int has_mtl, int dirty,
                                                     int is_render_target)
{
    return has_texture && has_mtl && !dirty && !is_render_target;
}

extern "C" int mgl_batch_replay_sampled_tex_info_ok(int has_texture_info_ok,
                                                    uint32_t texture_type,
                                                    uint32_t expected_type,
                                                    int pixel_format_compatible)
{
    if (!has_texture_info_ok) {
        return 0;
    }
    if (expected_type != 0u && texture_type != expected_type) {
        return 0;
    }
    return pixel_format_compatible ? 1 : 0;
}

extern "C" int mgl_batch_replay_sampled_resolve_gate(
    const MGLBatchSampledResolveGateIn *in)
{
    if (!in || !in->unit_ok) {
        return 0;
    }
    if (!mgl_batch_replay_sampled_tex_object_ok(in->has_tex, in->has_mtl,
                                                in->dirty, in->is_rt)) {
        return -1;
    }
    if (!mgl_batch_replay_sampled_tex_info_ok(in->info_ok, in->texture_type,
                                              in->expected_type,
                                              in->format_compat)) {
        return -1;
    }
    if (!in->needs_combined_sampler) {
        return 1;
    }
    if (!in->has_sampler_mtl) {
        return -1;
    }
    return 2;
}


extern "C" void mgl_batch_flush_accum_cmd_frame_stats(
    const MGLDrawBatch *batch, MGLBatchCmdFrameStats *out)
{
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (!batch) {
        return;
    }
    for (uint32_t i = 0; i < batch->command_count; i++) {
        const MGLDrawCommand *cmd = &batch->commands[i];
        MGLBatchCmdStatDelta d;
        const int uses_el = mgl_batch_replay_cmd_is_elements_draw((uint32_t)cmd->type);
        /* cmd_stat_delta lives in mgl_batch_issue — declare via header. */
        mgl_batch_issue_cmd_stat_delta((uint32_t)cmd->type, cmd->count, uses_el,
                                       &d);
        out->array_draws += d.array_draws;
        out->array_vertices += d.array_vertices;
        out->element_draws += d.element_draws;
        out->element_indices += d.element_indices;
    }
}

extern "C" void mgl_batch_issue_stream_merged(const MGLDrawBatch *batch,
                                              int disable_mdi,
                                              const MGLBatchStreamMergedOps *ops)
{
    if (!ops) {
        return;
    }
    const int streamPath = mgl_batch_replay_stream_path(batch, disable_mdi);
    if (streamPath == MGL_BATCH_STREAM_EMPTY) {
        if (ops->trace_cmd0) {
            ops->trace_cmd0(ops->ctx, "SKIP",
                            mgl_batch_replay_stream_path_reason(streamPath));
        }
        return;
    }
    if (streamPath == MGL_BATCH_STREAM_BAD_PRIM) {
        if (ops->trace_cmd0) {
            ops->trace_cmd0(ops->ctx, "FALLBACK",
                            mgl_batch_replay_stream_path_reason(streamPath));
        }
        if (ops->issue_direct) {
            ops->issue_direct(ops->ctx);
        }
        return;
    }
    if (streamPath == MGL_BATCH_STREAM_TRY_MDI) {
        if (ops->trace_cmd0) {
            ops->trace_cmd0(ops->ctx, "ISSUE",
                            mgl_batch_replay_stream_path_reason(streamPath));
        }
        if (ops->try_stream_mdi && ops->try_stream_mdi(ops->ctx)) {
            return;
        }
    }
    void *mtl_index = NULL;
    if (!ops->resolve_stream_index ||
        !ops->resolve_stream_index(ops->ctx, &mtl_index) || !mtl_index) {
        if (ops->issue_direct) {
            ops->issue_direct(ops->ctx);
        }
        return;
    }
    if (ops->draw_stream_indexed) {
        ops->draw_stream_indexed(ops->ctx, mtl_index);
    }
}


extern "C" void mgl_batch_issue_direct_batch(const MGLBatchDirectIssueOps *ops)
{
    if (!ops || !ops->command_count || !ops->fill_cmd) {
        return;
    }
    void *ctx = ops->ctx;
    if (ops->refresh_encoder) {
        ops->refresh_encoder(ctx);
    }
    if (ops->try_simple_replay && ops->try_simple_replay(ctx)) {
        return;
    }
    const uint32_t n = ops->command_count(ctx);
    const int uses_cull =
        ops->uses_cull_distance ? ops->uses_cull_distance(ctx) : 0;
    const uint32_t batch_prim =
        ops->batch_primitive_type ? ops->batch_primitive_type(ctx) : 0xFFu;
    const int snap_mixed = ops->snapshots_mixed ? ops->snapshots_mixed(ctx) : 0;
    const int dyn_tex = ops->has_dyn_texture ? ops->has_dyn_texture(ctx) : 0;

    for (uint32_t i = 0; i < n; i++) {
        if (ops->refresh_encoder) {
            ops->refresh_encoder(ctx);
        }
        MGLBatchDirectCmdView cmd;
        memset(&cmd, 0, sizeof(cmd));
        ops->fill_cmd(ctx, i, &cmd);
        const int cullPath =
            mgl_batch_issue_cull_capture_path(uses_cull, cmd.type);
        if (cullPath != MGL_BATCH_CULL_CAPTURE_NONE && ops->cull_capture) {
            if (ops->cull_capture(ctx, i, cullPath)) {
                if (!ops->after_cull_ok || !ops->after_cull_ok(ctx)) {
                    if (ops->trace_skip) {
                        ops->trace_skip(ctx, i, "cull_distance_capture_restore");
                    }
                    continue;
                }
                if (ops->refresh_encoder) {
                    ops->refresh_encoder(ctx);
                }
            }
        }
        if (ops->apply_dyn_bindings && !ops->apply_dyn_bindings(ctx, i)) {
            if (ops->trace_skip) {
                ops->trace_skip(ctx, i, "dynamic_binding");
            }
            continue;
        }
        if (mgl_batch_issue_should_apply_cmd_sampler(snap_mixed, dyn_tex) &&
            ops->apply_cmd_sampler && !ops->apply_cmd_sampler(ctx, i)) {
            if (ops->trace_skip) {
                ops->trace_skip(ctx, i, "sampler_snapshot");
            }
            continue;
        }

        const int poly_pt =
            ops->polygon_mode_point ? ops->polygon_mode_point(ctx, cmd.mode) : 0;
        MGLBatchReplayDirectPrimPlan primPlan;
        mgl_batch_replay_direct_prim_plan(cmd.mode, poly_pt, batch_prim,
                                          &primPlan);
        if (primPlan.skip_unsupported_prim) {
            if (ops->trace_skip) {
                ops->trace_skip(ctx, i, "direct_unsupported_primitive");
            }
            continue;
        }
        if (mgl_batch_replay_cmd_is_array_draw(cmd.type)) {
            int32_t ic = 0;
            uint32_t bi = 0u;
            const char *reason = NULL;
            const char *cullReason = NULL;
            mgl_batch_issue_direct_arrays_params(cmd.type, cmd.instance_count,
                                                 cmd.base_instance, &ic, &bi,
                                                 &reason, &cullReason);
            MGLBatchDirectArraySubmitOps asub = ops->array_submit;
            if (!asub.ctx) {
                asub.ctx = ctx;
            }
            mgl_batch_issue_submit_direct_arrays(
                i, cmd.mode, cmd.first, cmd.count, ic, bi, poly_pt, uses_cull,
                reason, cullReason, &asub);
        } else {
            MGLBatchDirectElementSubmitOps esub = ops->element_submit;
            if (!esub.ctx) {
                esub.ctx = ctx;
            }
            mgl_batch_issue_submit_direct_elements(
                i, cmd.mode, cmd.count, cmd.instance_count, poly_pt, &esub);
        }
    }
}

extern "C" int mgl_batch_mtl_bind_dyn_vertex(const MGLBatchDynVertexBindOps *ops)
{
    if (!ops || !ops->plan_binding || !ops->resolve_slot || !ops->ensure_mtl ||
        !ops->vao_binding_offset) {
        return 0;
    }
    for (uint8_t bi = 0; bi < ops->binding_count; bi++) {
        MGLBatchDynVertexStreamPlan plan;
        memset(&plan, 0, sizeof(plan));
        const int planRc = ops->plan_binding(ops->ctx, bi, &plan);
        if (planRc == MGL_BATCH_DYN_VERTEX_FAIL) {
            return 0;
        }
        if (planRc == MGL_BATCH_DYN_VERTEX_UNUSED) {
            continue;
        }
        int resolved_slots[MGL_BATCH_DYN_VERTEX_MAX_STREAMS];
        uint32_t resolved_slot_count = 0u;
        for (uint32_t stream = 0; stream < plan.stream_count; stream++) {
            int resolved_slot = -1;
            if (!ops->resolve_slot(ops->ctx, plan.representative_attribs[stream],
                                   &resolved_slot) ||
                resolved_slot < 0) {
                continue;
            }
            if (!mgl_batch_replay_dyn_vertex_slot_ok(resolved_slot,
                                                     ops->max_metal_slots)) {
                return 0;
            }
            if (ops->stream_can_bind &&
                !ops->stream_can_bind(ops->ctx, &plan, stream)) {
                return 0;
            }
            if (resolved_slot_count < MGL_BATCH_DYN_VERTEX_MAX_STREAMS) {
                resolved_slots[resolved_slot_count++] = resolved_slot;
            }
        }
        if (resolved_slot_count == 0u) {
            continue;
        }
        void *mtl = NULL;
        void *gl_buf = NULL;
        uint64_t dyn_off = 0;
        uint64_t mtl_len = 0;
        if (!ops->ensure_mtl(ops->ctx, &plan, &mtl, &gl_buf, &dyn_off,
                             &mtl_len) ||
            !mtl) {
            return 0;
        }
        const uint64_t bind_off = ops->vao_binding_offset(ops->ctx, &plan);
        if (!mgl_batch_replay_dyn_vertex_offset_ok(bind_off, dyn_off, mtl_len)) {
            return 0;
        }
        MGLBatchBufferBindReq reqs[MGL_BATCH_MTL_BUFFER_BIND_MAX];
        uint32_t req_count = 0u;
        for (uint32_t s = 0; s < resolved_slot_count; s++) {
            if (req_count >= MGL_BATCH_MTL_BUFFER_BIND_MAX) {
                break;
            }
            reqs[req_count].mtl_buffer = mtl;
            reqs[req_count].gl_buffer = gl_buf;
            reqs[req_count].offset = dyn_off;
            reqs[req_count].metal_slot = (uint32_t)resolved_slots[s];
            reqs[req_count].is_vertex_stage = 1u;
            req_count++;
        }
        (void)mgl_batch_mtl_encode_buffer_binds(ops->binding_state_owner,
                                                ops->render_encoder_owner, reqs,
                                                req_count);
    }
    return 1;
}

extern "C" int mgl_batch_mtl_bind_dyn_uniforms(const MGLBatchDynUniformBindOps *ops)
{
    if (!ops || !ops->cmd || !ops->glm_ctx || !ops->gather_lengths ||
        !ops->resolve_op) {
        return 0;
    }
    const uint32_t count = ops->cmd->dynamic_uniform_binding_count;
    if (count == 0u) {
        return 1;
    }
    if (count > MGL_MAX_DYNAMIC_UNIFORM_BINDINGS) {
        return 0;
    }
    uint64_t mtl_lengths[MGL_MAX_DYNAMIC_UNIFORM_BINDINGS];
    memset(mtl_lengths, 0, sizeof(mtl_lengths));
    if (!ops->gather_lengths(ops->ctx, mtl_lengths, count)) {
        return 0;
    }
    MGLBatchUniformBindPlan plan;
    if (!mgl_batch_replay_plan_uniform_binds(
            ops->glm_ctx, ops->cmd, mtl_lengths, count,
            ops->min_stage_binding_size, ops->max_buffer_slots, &plan)) {
        return 0;
    }
    MGLBatchBufferBindReq reqs[MGL_BATCH_MTL_BUFFER_BIND_MAX];
    uint32_t req_count = 0u;
    for (uint32_t oi = 0; oi < plan.count; oi++) {
        const MGLBatchUniformBindOp *op = &plan.ops[oi];
        void *mtl = NULL;
        void *gl_buf = NULL;
        if (!ops->resolve_op(ops->ctx, op, &mtl, &gl_buf) || !mtl) {
            return 0;
        }
        if (req_count >= MGL_BATCH_MTL_BUFFER_BIND_MAX) {
            return 0;
        }
        reqs[req_count].mtl_buffer = mtl;
        reqs[req_count].gl_buffer = gl_buf;
        reqs[req_count].offset = op->offset;
        reqs[req_count].metal_slot = op->metal_slot;
        reqs[req_count].is_vertex_stage = op->is_vertex_stage;
        req_count++;
    }
    (void)mgl_batch_mtl_encode_buffer_binds(ops->binding_state_owner,
                                            ops->render_encoder_owner, reqs,
                                            req_count);
    return 1;
}

extern "C" int mgl_batch_mtl_bind_dyn_sampled(const MGLBatchDynSampledBindOps *ops)
{
    if (!ops || !ops->glm_ctx || !ops->resolve_candidate || !ops->touched_units) {
        return 0;
    }
    MGLBatchSampledTexPlan plan;
    if (!mgl_batch_replay_plan_sampled_texture_candidates(ops->glm_ctx, &plan)) {
        return 0;
    }
    MGLBatchResourceBindReq reqs[MGL_BATCH_MTL_RESOURCE_BIND_MAX];
    uint32_t req_count = 0u;
    for (uint32_t i = 0; i < plan.count; i++) {
        void *texture = NULL;
        void *sampler = NULL;
        uint32_t binding_stage = 0u;
        uint32_t sampler_slot = 0u;
        int needs_sampler = 0;
        const int rc = ops->resolve_candidate(
            ops->ctx, &plan.entries[i], ops->touched_units, &texture,
            &binding_stage, &needs_sampler, &sampler, &sampler_slot);
        if (rc < 0) {
            return 0;
        }
        if (rc == 0) {
            continue;
        }
        if (!texture || req_count >= MGL_BATCH_MTL_RESOURCE_BIND_MAX) {
            return 0;
        }
        reqs[req_count].resource = texture;
        reqs[req_count].metal_slot = plan.entries[i].metal_slot;
        reqs[req_count].binding_stage = binding_stage;
        reqs[req_count].kind = MGL_RENDER_RESOURCE_BINDING_TEXTURE;
        req_count++;
        if (!needs_sampler) {
            continue;
        }
        if (!sampler ||
            !mgl_batch_replay_sampler_slot_ok(sampler_slot,
                                              ops->max_sampler_slots) ||
            req_count >= MGL_BATCH_MTL_RESOURCE_BIND_MAX) {
            return 0;
        }
        reqs[req_count].resource = sampler;
        reqs[req_count].metal_slot = sampler_slot;
        reqs[req_count].binding_stage = binding_stage;
        reqs[req_count].kind = MGL_RENDER_RESOURCE_BINDING_SAMPLER;
        req_count++;
    }
    return mgl_batch_mtl_encode_resource_binds(ops->binding_state_owner,
                                               ops->render_encoder_owner, reqs,
                                               req_count);
}

/* === Active-texture binding (former -[MGLRenderer bindActiveTexturesToMTL]) ===
 * The per-unit lookup and stale-mask bookkeeping are plain C; only the Metal
 * texture bind goes through a port. */
namespace {

struct MGLActTexCtx {
    void *r;
    GLMContext glm;
};

int mglActTexBindUnit(void *v, uint32_t unit, int *stale_out)
{
    MGLActTexCtx *c = static_cast<MGLActTexCtx *>(v);
    Texture *tex = c->glm->active_state->active_textures[unit];
    if (!tex) {
        if (stale_out) *stale_out = 1;
        return 0;
    }
    if (stale_out) *stale_out = 0;
    return mglRendererBindMTLTexturePort(c->r, tex) ? 1 : 0;
}

void mglActTexClearStale(void *v, uint32_t word, uint32_t bit)
{
    MGLActTexCtx *c = static_cast<MGLActTexCtx *>(v);
    c->glm->active_state->active_texture_mask[word] &= ~(1u << bit);
    mglInvalidateStateHashCachesForDirtyBits(c->glm->active_state, DIRTY_TEX_BINDING);
}

} // namespace

extern "C" int mglBatchBindActiveTexturesToMTL(void *renderer, GLMContext glm_ctx)
{
    MGLActTexCtx c = {renderer, glm_ctx};
    MGLBatchActiveTexBindOps ops = {
        .ctx = &c,
        .mask4 = glm_ctx->active_state->active_texture_mask,
        .bind_unit = mglActTexBindUnit,
        .clear_stale = mglActTexClearStale,
    };
    return mgl_batch_bind_active_textures(&ops) ? 1 : 0;
}
