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
