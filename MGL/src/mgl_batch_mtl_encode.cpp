/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3 encode-fold: MTL draw/ICB ports for batch encode ObjC thinning.
 */
#include "mgl_batch_mtl_encode.h"
#include "mgl_batch_replay.h"
#include "mgl_frame_activity.h"
#include "mgl_index_buffer.h"

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



extern "C" int mgl_batch_mtl_encode_buffer_binds(void *binding_state_owner,
                                                 void *render_encoder_owner,
                                                 const MGLBatchBufferBindReq *reqs,
                                                 uint32_t count)
{
    if (!reqs || count == 0u) {
        return 0;
    }
    MGLRenderBindingSnapshot snapshot = {0};
    for (uint32_t i = 0; i < count; i++) {
        const MGLBatchBufferBindReq *r = &reqs[i];
        if (!r->mtl_buffer) {
            continue;
        }
        const uint32_t stage = r->is_vertex_stage
                                   ? MGL_RENDER_BINDING_STAGE_VERTEX
                                   : MGL_RENDER_BINDING_STAGE_FRAGMENT;
        void *cur = NULL;
        uint64_t cur_off = 0;
        uint32_t valid_flag = 0;
        const int owner_valid =
            binding_state_owner &&
            mglRenderBindingGetValid(binding_state_owner, &valid_flag) == 0 &&
            valid_flag != 0u;
        const int matches =
            owner_valid &&
            mglRenderBindingGetBuffer(binding_state_owner, stage, r->metal_slot,
                                      &cur, &cur_off) == 0 &&
            cur == r->mtl_buffer && cur_off == r->offset;
        if (!matches) {
            if (r->is_vertex_stage) {
                mglRenderBindingUpdateVertexBuffer(binding_state_owner,
                                                   r->mtl_buffer, r->offset,
                                                   r->metal_slot);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                if (snapshot.vertex_op_count <
                    MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
                    snapshot.vertex_ops[snapshot.vertex_op_count++] =
                        (MGLRenderBindingOp){0u, r->metal_slot, r->offset,
                                             r->mtl_buffer, NULL, 0u};
                }
            } else {
                mglRenderBindingUpdateFragmentBuffer(binding_state_owner,
                                                     r->mtl_buffer, r->offset,
                                                     r->metal_slot);
                MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                if (snapshot.fragment_op_count <
                    MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
                    snapshot.fragment_ops[snapshot.fragment_op_count++] =
                        (MGLRenderBindingOp){0u, r->metal_slot, r->offset,
                                             r->mtl_buffer, NULL, 0u};
                }
            }
            if (r->gl_buffer) {
                mglNoteBufferEncoded((Buffer *)r->gl_buffer);
            }
        } else if (r->is_vertex_stage) {
            MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
        }
    }
    if (snapshot.vertex_op_count > 0 || snapshot.fragment_op_count > 0) {
        (void)mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
            render_encoder_owner, &snapshot, NULL, 0);
    }
    return 0;
}

extern "C" int mgl_batch_mtl_encode_resource_binds(
    void *binding_state_owner, void *render_encoder_owner,
    const MGLBatchResourceBindReq *reqs, uint32_t count)
{
    if (!reqs) {
        return 0;
    }
    MGLRenderResourceBindingSnapshot snapshot = {0};
    for (uint32_t i = 0; i < count; i++) {
        const MGLBatchResourceBindReq *r = &reqs[i];
        if (!r->resource) {
            return 0;
        }
        if (!mgl_batch_replay_collect_resource_binding(
                &snapshot, r->binding_stage, r->kind, r->resource,
                r->metal_slot)) {
            return 0;
        }
    }
    return mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
               binding_state_owner, render_encoder_owner, &snapshot, NULL,
               0) == 0;
}

extern "C" uint32_t mgl_batch_mtl_restore_plan_delta_dirty(
    int can_delta, const void *prev_key, const void *cur_key,
    uint32_t full_bits, MGLBatchDirtyDeltaFlags *flags_out)
{
    MGLBatchDirtyDomainMasks masks;
    mgl_batch_restore_default_domain_masks(&masks);
    MGLBatchStateKeyView prevView;
    MGLBatchStateKeyView curView;
    mgl_batch_state_key_view_from_key(prev_key, &prevView);
    mgl_batch_state_key_view_from_key(cur_key, &curView);
    return mgl_batch_compute_key_delta_dirty_bits(
        can_delta, can_delta ? &prevView : NULL, &curView, full_bits, &masks,
        flags_out);
}

extern "C" void mgl_batch_mtl_restore_note_delta_perf(
    const MGLBatchDirtyDeltaFlags *flags)
{
    if (!flags) {
        return;
    }
    if (flags->domain_program) {
        MGL_PERF_INC(g_mglDeltaDomainProgramSinceSwap);
    }
    if (flags->domain_vao) {
        MGL_PERF_INC(g_mglDeltaDomainVAOSinceSwap);
    }
    if (flags->domain_texture) {
        MGL_PERF_INC(g_mglDeltaDomainTextureSinceSwap);
    }
    if (flags->domain_render_state_ubo_only) {
        MGL_PERF_INC(g_mglDeltaDomainRenderStateUboOnlySinceSwap);
    } else if (flags->domain_render_state) {
        MGL_PERF_INC(g_mglDeltaDomainRenderStateSinceSwap);
    }
    if (flags->narrowed) {
        MGL_PERF_INC(g_mglDirtyKeyDeltaNarrowSinceSwap);
    }
}

extern "C" void mgl_batch_mtl_restore_note_skip_fail_perf(int skip_dec)
{
    switch (skip_dec) {
    case MGL_BATCH_SAME_KEY_FAIL_NO_ENCODER:
        MGL_PERF_INC(g_mglSkipFailNoEncoderSinceSwap);
        break;
    case MGL_BATCH_SAME_KEY_FAIL_BIND:
        MGL_PERF_INC(g_mglSkipFailBindInvalidSinceSwap);
        break;
    case MGL_BATCH_SAME_KEY_FAIL_KEY:
        MGL_PERF_INC(g_mglSkipFailKeyDifferSinceSwap);
        break;
    case MGL_BATCH_SAME_KEY_FAIL_PASS:
        MGL_PERF_INC(g_mglSkipFailPassMismatchSinceSwap);
        break;
    default:
        break;
    }
}
