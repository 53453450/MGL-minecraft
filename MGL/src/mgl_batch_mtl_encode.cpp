/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3 encode-fold: MTL draw/ICB ports for batch encode ObjC thinning.
 */
#include "mgl_batch_mtl_encode.h"

#include "mgl_batch_restore.h"
#include "draw_command.h"
#include "mgl_render.h"

#include <stddef.h>
#include <string.h>

extern "C" int mgl_batch_mtl_draw_indexed(void *render_encoder_owner,
                                          uint32_t primitive_type,
                                          uint64_t index_count,
                                          uint32_t index_type,
                                          void *index_buffer,
                                          uint64_t index_buffer_offset,
                                          uint64_t instance_count,
                                          int64_t base_vertex,
                                          uint64_t base_instance)
{
    const MGLRenderDrawPlan plan = {
        .kind = MGL_RENDER_DRAW_INDEXED,
        .primitive_type = primitive_type,
        .index_count = index_count,
        .index_type = index_type,
        .index_buffer = index_buffer,
        .index_buffer_offset = index_buffer_offset,
        .instance_count = instance_count,
        .base_vertex = base_vertex,
        .base_instance = base_instance,
    };
    return mglRenderEncodeDrawForRenderEncoderOwner(render_encoder_owner, &plan,
                                                    NULL, 0);
}

extern "C" int mgl_batch_mtl_draw_array_indirect(void *render_encoder_owner,
                                                 uint32_t primitive_type,
                                                 void *indirect_buffer,
                                                 uint64_t indirect_buffer_offset)
{
    const MGLRenderDrawPlan plan = {
        .kind = MGL_RENDER_DRAW_ARRAY_INDIRECT,
        .primitive_type = primitive_type,
        .indirect_buffer = indirect_buffer,
        .indirect_buffer_offset = indirect_buffer_offset,
    };
    return mglRenderEncodeDrawForRenderEncoderOwner(render_encoder_owner, &plan,
                                                    NULL, 0);
}

extern "C" int mgl_batch_mtl_draw_indexed_indirect(
    void *render_encoder_owner, uint32_t primitive_type, uint32_t index_type,
    void *index_buffer, uint64_t index_buffer_offset, void *indirect_buffer,
    uint64_t indirect_buffer_offset)
{
    const MGLRenderDrawPlan plan = {
        .kind = MGL_RENDER_DRAW_INDEXED_INDIRECT,
        .primitive_type = primitive_type,
        .index_type = index_type,
        .index_buffer = index_buffer,
        .index_buffer_offset = index_buffer_offset,
        .indirect_buffer = indirect_buffer,
        .indirect_buffer_offset = indirect_buffer_offset,
    };
    return mglRenderEncodeDrawForRenderEncoderOwner(render_encoder_owner, &plan,
                                                    NULL, 0);
}

extern "C" void *mgl_batch_mtl_create_icb(int indexed,
                                          uint64_t max_command_count)
{
    void *buffer = NULL;
    const uint32_t command_types = indexed ? 2u : 1u;
    if (mglRenderCreateIndirectCommandBuffer(command_types, 1, 1, 0, 0,
                                             max_command_count, 32u,
                                             &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

extern "C" int mgl_batch_mtl_reset_icb(void *icb, uint64_t location,
                                       uint64_t length)
{
    return mglRenderResetIndirectCommandBuffer(icb, location, length);
}

extern "C" void *mgl_batch_mtl_icb_command(void *icb, uint64_t index)
{
    void *command = NULL;
    if (mglRenderGetIndirectRenderCommand(icb, index, &command) == 0) {
        return command;
    }
    return NULL;
}

extern "C" int mgl_batch_mtl_set_icb_draw_indexed(
    void *command, uint32_t primitive_type, uint64_t index_count,
    uint32_t index_type, void *index_buffer, uint64_t index_buffer_offset,
    uint64_t instance_count, int64_t base_vertex, uint64_t base_instance)
{
    return mglRenderSetIndirectDrawIndexed(
        command, primitive_type, index_count, index_type, index_buffer,
        index_buffer_offset, instance_count, base_vertex, base_instance);
}

extern "C" int mgl_batch_mtl_set_icb_draw(void *command, uint32_t primitive_type,
                                          uint64_t vertex_start,
                                          uint64_t vertex_count,
                                          uint64_t instance_count,
                                          uint64_t base_instance)
{
    return mglRenderSetIndirectDraw(command, primitive_type, vertex_start,
                                    vertex_count, instance_count, base_instance);
}

extern "C" int mgl_batch_mtl_use_render_resource(void *render_encoder_owner,
                                                 void *resource, uint32_t usage,
                                                 uint32_t stages)
{
    return mglRenderUseRenderResourceForOwner(render_encoder_owner, resource,
                                              usage, stages);
}

extern "C" int mgl_batch_mtl_execute_icb(void *render_encoder_owner, void *icb,
                                         uint64_t location, uint64_t length)
{
    return mglRenderExecuteIndirectCommandsForOwner(render_encoder_owner, icb,
                                                    location, length);
}

extern "C" void mgl_batch_mtl_issue_mdi_array_draws(
    void *render_encoder_owner, uint32_t primitive_type, void *indirect_buffer,
    uint64_t args_base_offset, uint64_t arg_size, uint32_t command_count,
    const MGLBatchMtlCmdTraceOps *trace)
{
    for (uint32_t i = 0; i < command_count; i++) {
        (void)mgl_batch_mtl_draw_array_indirect(
            render_encoder_owner, primitive_type, indirect_buffer,
            args_base_offset + (uint64_t)i * arg_size);
        if (trace && trace->on_submit) {
            trace->on_submit(trace->ctx, i, "mdi_arrays");
        }
    }
}

extern "C" void mgl_batch_mtl_issue_stream_mdi_draws(
    void *render_encoder_owner, uint32_t primitive_type, uint32_t index_type,
    void *index_buffer, const uint32_t *index_buffer_offsets,
    void *indirect_buffer, uint64_t args_base_offset, uint64_t arg_size,
    uint32_t command_count, const MGLBatchMtlCmdTraceOps *trace)
{
    if (!index_buffer_offsets) {
        return;
    }
    for (uint32_t i = 0; i < command_count; i++) {
        (void)mgl_batch_mtl_draw_indexed_indirect(
            render_encoder_owner, primitive_type, index_type, index_buffer,
            index_buffer_offsets[i], indirect_buffer,
            args_base_offset + (uint64_t)i * arg_size);
        if (trace && trace->on_submit) {
            trace->on_submit(trace->ctx, i, "stream_mdi_indexed");
        }
    }
}

extern "C" void mgl_batch_state_key_view_from_key(const void *state_key,
                                                  MGLBatchStateKeyView *out)
{
    const MGLStateKey *key = (const MGLStateKey *)state_key;
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (!key) {
        return;
    }
    out->program_name = key->program_name;
    out->program_pipeline_name = key->program_pipeline_name;
    out->vertex_program_name = key->vertex_program_name;
    out->fragment_program_name = key->fragment_program_name;
    out->vao_name = key->vao_name;
    out->vertex_layout_hash = key->vertex_layout_hash;
    out->texture_hash = key->texture_hash;
    out->render_state_hash = key->render_state_hash;
    out->uniform_buffer_hash = key->uniform_buffer_hash;
    out->caps_flags = key->caps_flags;
    out->scissor_enabled = key->scissor_enabled;
    out->primitive_type = key->primitive_type;
    memcpy(out->viewport, key->viewport, sizeof(out->viewport));
    memcpy(out->scissor, key->scissor, sizeof(out->scissor));
}
