/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3 encode-fold: C++ MTL draw / ICB / dyn-bind ports. ObjC one-liners.
 * Not mgl_render.cpp / mgl_draw_metal_port.m / mgl_batch_replay_trace.m.
 */
#ifndef MGL_BATCH_MTL_ENCODE_H
#define MGL_BATCH_MTL_ENCODE_H

#include "mgl_batch_restore.h"

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

/* ---- A3 encode-fold: whole MDI / stream-MDI / ICB issue loops ---- */

typedef struct MGLBatchMdiIssueOps {
    void *ctx;
    void (*on_trace)(void *ctx, uint32_t cmd_index, const char *phase,
                     const char *reason);
    void (*issue_direct)(void *ctx);
    /* Alloc indirect-args scratch; return MTLBuffer* or NULL. */
    void *(*alloc_scratch)(void *ctx, uint64_t length, uint64_t *offset_out);
    /* Map scratch; *contents_out = base+offset. Return 1 if range ok. */
    int (*map_scratch)(void *ctx, void *buffer, uint64_t offset, uint64_t needed,
                       void **contents_out);
    /* Indexed: prepare index for cmd i. Return 1 on success. */
    int (*resolve_index)(void *ctx, uint32_t cmd_index, uint32_t gl_index_type,
                         void **mtl_index_out, uint64_t *index_offset_inout,
                         uint32_t *mtl_index_type_out);
} MGLBatchMdiIssueOps;

/* Full MDI path: gate + scratch + fill + draw loop. batch is MGLDrawBatch*. */
void mgl_batch_mtl_issue_mdi_batch(const void *batch, int disable_mdi,
                                   void *render_encoder_owner,
                                   const MGLBatchMdiIssueOps *ops);

typedef struct MGLBatchStreamMdiIssueOps {
    void *ctx;
    void (*on_trace)(void *ctx, uint32_t cmd_index, const char *phase,
                     const char *reason);
    /* Process stream index buffer → MTL index (or NULL). */
    void *(*resolve_stream_index)(void *ctx);
    void *(*alloc_scratch)(void *ctx, uint64_t length, uint64_t *offset_out);
    int (*map_scratch)(void *ctx, void *buffer, uint64_t offset, uint64_t needed,
                       void **contents_out);
} MGLBatchStreamMdiIssueOps;

/* Stream-merged MDI. Returns 1 on success, 0 → caller falls back. */
int mgl_batch_mtl_issue_stream_mdi_batch(const void *batch, int disable_mdi,
                                         void *render_encoder_owner,
                                         const MGLBatchStreamMdiIssueOps *ops);

typedef struct MGLBatchIcbIssueOps {
    void *ctx;
    void (*on_trace)(void *ctx, uint32_t cmd_index, const char *phase,
                     const char *reason);
    /* ObjC @try create; return ICB or NULL. */
    void *(*create_icb)(void *ctx, int indexed, uint64_t command_count);
    int (*resolve_index)(void *ctx, uint32_t cmd_index, uint32_t gl_index_type,
                         void **mtl_index_out, uint64_t *index_offset_inout,
                         uint32_t *mtl_index_type_out);
} MGLBatchIcbIssueOps;

/* Full ICB path. Returns 1 on success, 0 → caller falls back to direct. */
int mgl_batch_mtl_issue_icb_batch(const void *batch, int has_device,
                                  int has_encoder, int icb_enable,
                                  int icb_disable, int os_supported,
                                  void *render_encoder_owner,
                                  const MGLBatchIcbIssueOps *ops);


/* Simple replay: eligibility checked by caller; fill+encode loop in C++. */
typedef struct MGLBatchSimpleReplayOps {
    void *ctx;
    /* Elements cmds only: prepare index. Return 1 ok. */
    int (*resolve_index)(void *ctx, uint32_t cmd_index, uint32_t gl_index_type,
                         void **mtl_index_out, uint64_t *index_offset_inout,
                         uint32_t *mtl_index_type_out);
} MGLBatchSimpleReplayOps;

/* Returns 1 on success. Caller must ensure simple_eligible. */
int mgl_batch_mtl_issue_simple_replay(const void *batch,
                                      void *render_encoder_owner,
                                      const MGLBatchSimpleReplayOps *ops);

/* ---- A3 residual: dyn-bind set*Buffer / resource ports ---- */

enum { MGL_BATCH_MTL_BUFFER_BIND_MAX = 64 };

typedef struct MGLBatchBufferBindReq {
    void *mtl_buffer;
    void *gl_buffer; /* optional; mglNoteBufferEncoded when encoded */
    uint64_t offset;
    uint32_t metal_slot;
    uint8_t is_vertex_stage; /* 1 = VS, 0 = FS */
} MGLBatchBufferBindReq;

/* Dedup via binding_state, update owner, encode snapshot, bump set*Buffer
 * perf counters, note encoded GL buffers. Returns 0 on success. */
int mgl_batch_mtl_encode_buffer_binds(void *binding_state_owner,
                                      void *render_encoder_owner,
                                      const MGLBatchBufferBindReq *reqs,
                                      uint32_t count);

enum { MGL_BATCH_MTL_RESOURCE_BIND_MAX = 64 };

typedef struct MGLBatchResourceBindReq {
    void *resource; /* MTLTexture* or MTLSamplerState* */
    uint32_t metal_slot;
    uint32_t binding_stage; /* MGL_RENDER_BINDING_STAGE_* */
    uint32_t kind;          /* MGL_RENDER_RESOURCE_BINDING_* */
} MGLBatchResourceBindReq;

/* Collect + encode resource binding snapshot. Returns 1 on success. */
int mgl_batch_mtl_encode_resource_binds(void *binding_state_owner,
                                        void *render_encoder_owner,
                                        const MGLBatchResourceBindReq *reqs,
                                        uint32_t count);

/* Dirty-key delta from MGLStateKey* (renderer). */
uint32_t mgl_batch_mtl_restore_plan_delta_dirty(int can_delta,
                                                const void *prev_key,
                                                const void *cur_key,
                                                uint32_t full_bits,
                                                MGLBatchDirtyDeltaFlags *flags_out);

void mgl_batch_mtl_restore_note_delta_perf(const MGLBatchDirtyDeltaFlags *flags);
void mgl_batch_mtl_restore_note_skip_fail_perf(int skip_dec);

/* Resolved sampler snapshot entries → resource binds (loop in C++). */
typedef struct MGLBatchResolvedSamplerBind {
    void *sampler;
    uint32_t metal_slot;
    int shader_stage; /* _VERTEX_SHADER / _FRAGMENT_SHADER / … */
} MGLBatchResolvedSamplerBind;

int mgl_batch_mtl_encode_resolved_samplers(void *binding_state_owner,
                                           void *render_encoder_owner,
                                           const MGLBatchResolvedSamplerBind *items,
                                           uint32_t count);



#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_MTL_ENCODE_H */
