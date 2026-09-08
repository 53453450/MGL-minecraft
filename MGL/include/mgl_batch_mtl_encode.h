/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3 encode-fold: C++ MTL draw / ICB ports. ObjC calls these as one-liners.
 * Not mgl_render.cpp / mgl_draw_metal_port.m / mgl_batch_replay_trace.m.
 */
#ifndef MGL_BATCH_MTL_ENCODE_H
#define MGL_BATCH_MTL_ENCODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int mgl_batch_mtl_draw_indexed(void *render_encoder_owner,
                               uint32_t primitive_type, uint64_t index_count,
                               uint32_t index_type, void *index_buffer,
                               uint64_t index_buffer_offset,
                               uint64_t instance_count, int64_t base_vertex,
                               uint64_t base_instance);

int mgl_batch_mtl_draw_array_indirect(void *render_encoder_owner,
                                      uint32_t primitive_type,
                                      void *indirect_buffer,
                                      uint64_t indirect_buffer_offset);

int mgl_batch_mtl_draw_indexed_indirect(void *render_encoder_owner,
                                        uint32_t primitive_type,
                                        uint32_t index_type,
                                        void *index_buffer,
                                        uint64_t index_buffer_offset,
                                        void *indirect_buffer,
                                        uint64_t indirect_buffer_offset);

void *mgl_batch_mtl_create_icb(int indexed, uint64_t max_command_count);
int mgl_batch_mtl_reset_icb(void *icb, uint64_t location, uint64_t length);
void *mgl_batch_mtl_icb_command(void *icb, uint64_t index);
int mgl_batch_mtl_set_icb_draw_indexed(void *command, uint32_t primitive_type,
                                       uint64_t index_count, uint32_t index_type,
                                       void *index_buffer,
                                       uint64_t index_buffer_offset,
                                       uint64_t instance_count,
                                       int64_t base_vertex,
                                       uint64_t base_instance);
int mgl_batch_mtl_set_icb_draw(void *command, uint32_t primitive_type,
                               uint64_t vertex_start, uint64_t vertex_count,
                               uint64_t instance_count, uint64_t base_instance);
int mgl_batch_mtl_use_render_resource(void *render_encoder_owner,
                                      void *resource, uint32_t usage,
                                      uint32_t stages);
int mgl_batch_mtl_execute_icb(void *render_encoder_owner, void *icb,
                               uint64_t location, uint64_t length);

/* Issue MDI array draws: plan+encode loop in C++; ObjC supplies trace hook. */
typedef struct MGLBatchMtlCmdTraceOps {
    void *ctx;
    void (*on_submit)(void *ctx, uint32_t cmd_index, const char *reason);
} MGLBatchMtlCmdTraceOps;

void mgl_batch_mtl_issue_mdi_array_draws(void *render_encoder_owner,
                                         uint32_t primitive_type,
                                         void *indirect_buffer,
                                         uint64_t args_base_offset,
                                         uint64_t arg_size,
                                         uint32_t command_count,
                                         const MGLBatchMtlCmdTraceOps *trace);

void mgl_batch_mtl_issue_stream_mdi_draws(
    void *render_encoder_owner, uint32_t primitive_type, uint32_t index_type,
    void *index_buffer, const uint32_t *index_buffer_offsets,
    void *indirect_buffer, uint64_t args_base_offset, uint64_t arg_size,
    uint32_t command_count, const MGLBatchMtlCmdTraceOps *trace);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_MTL_ENCODE_H */
