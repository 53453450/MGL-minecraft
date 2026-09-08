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
 * mgl_batch_issue.h — A3 / O2.5: stream-MDI / direct-arrays / dyn-bind plans.
 *
 * Pure C, no Metal. ObjC issue and applyDynamic ports fill POD inputs.
 */

#ifndef MGL_BATCH_ISSUE_H
#define MGL_BATCH_ISSUE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MGL_BATCH_STREAM_MDI_OK = 0,
    MGL_BATCH_STREAM_MDI_FAIL_EMPTY = 1,
    MGL_BATCH_STREAM_MDI_FAIL_DISABLED = 2,
    MGL_BATCH_STREAM_MDI_FAIL_BAD_PRIM = 3,
    MGL_BATCH_STREAM_MDI_FAIL_OVERFLOW = 4
};

typedef struct MGLBatchStreamMdiGateIn {
    uint8_t stream_merged;
    uint8_t has_encoder;
    uint8_t disable_mdi;
    uint8_t primitive_type; /* 0xFF = unsupported */
    uint32_t command_count;
    uint32_t stream_index_count;
    size_t arg_size; /* sizeof indexed indirect args */
} MGLBatchStreamMdiGateIn;

/* Gate stream-merged MDI before index materialize / scratch alloc. */
int mgl_batch_issue_stream_mdi_gate(const MGLBatchStreamMdiGateIn *in,
                                    size_t *needed_bytes);
const char *mgl_batch_issue_stream_mdi_gate_reason(int gate);

/* Derive instanceCount / baseInstance / trace reasons for array cmds.
 * cmd_type values match MGL_CMD_DRAW_ARRAYS* enums. */
enum {
    MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS = 0,
    MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED = 2,
    MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE = 6
};

void mgl_batch_issue_direct_arrays_params(uint32_t cmd_type,
                                          int32_t instance_count,
                                          uint32_t base_instance,
                                          int32_t *out_instance_count,
                                          uint32_t *out_base_instance,
                                          const char **out_reason,
                                          const char **out_cull_reason);

int mgl_batch_issue_dyn_cmd_has_bindings(uint8_t vertex_count,
                                         uint8_t uniform_count,
                                         uint8_t texture_count);

int mgl_batch_issue_dyn_needs_mapper_fallback(int vertex_ok, int uniform_ok);

enum {
    MGL_BATCH_CULL_CAPTURE_NONE = 0,
    MGL_BATCH_CULL_CAPTURE_ARRAYS = 1,
    MGL_BATCH_CULL_CAPTURE_ELEMENTS = 2
};
int mgl_batch_issue_cull_capture_path(int uses_cull_distance,
                                      uint32_t cmd_type);

/* Stream non-MDI index materialize readiness (before MTL draw). */
enum {
    MGL_BATCH_STREAM_INDEX_OK = 0,
    MGL_BATCH_STREAM_INDEX_NO_BUFFER = 1,
    MGL_BATCH_STREAM_INDEX_NO_MTL = 2
};
int mgl_batch_issue_stream_index_ready(int has_index_buffer, int process_ok,
                                       int has_mtl_index);
const char *mgl_batch_issue_stream_index_reason(int ready);

/* Stable sampler snapshot apply gate for checkBatchShouldExecute. */
int mgl_batch_issue_should_apply_stable_sampler(int snapshots_mixed,
                                                uint32_t snapshot_id,
                                                uint32_t invalid_id);


/* ---- A3 encode-fold: flush / scratch / ICB / cmd-stats plans ---- */

typedef struct MGLBatchFlushPathStats {
    uint32_t mdi_batches;
    uint32_t mdi_commands;
    uint32_t icb_batches;
    uint32_t icb_commands;
    uint32_t direct_batches;
    uint32_t direct_commands;
    uint32_t stream_batches;
    uint32_t stream_commands;
} MGLBatchFlushPathStats;

/* path: MGL_BATCH_SELECT_* / MGLBatchPath values. */
void mgl_batch_flush_accum_path(MGLBatchFlushPathStats *stats, int path,
                                uint32_t command_count);
const char *mgl_batch_flush_path_phase(int path);
int mgl_batch_flush_should_trace_log(uint64_t hit, uint32_t total_commands,
                                     int diag_enabled, uint32_t skipped_commands,
                                     int replay_error_nonzero);

int mgl_batch_issue_scratch_range_ok(uint64_t offset, uint64_t needed,
                                     uint64_t length);

int mgl_batch_issue_should_apply_cmd_sampler(int snapshots_mixed,
                                             int has_dynamic_texture_bindings);

typedef struct MGLBatchIcbArrayDrawParams {
    uint32_t vertex_start;
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t base_instance;
} MGLBatchIcbArrayDrawParams;

void mgl_batch_issue_icb_array_draw_params(uint32_t first, uint32_t count,
                                           uint32_t instance_count,
                                           uint32_t base_instance,
                                           MGLBatchIcbArrayDrawParams *out);

uint32_t mgl_batch_issue_icb_command_types(int indexed);

typedef struct MGLBatchCmdStatDelta {
    uint32_t array_draws;
    uint64_t array_vertices;
    uint32_t element_draws;
    uint64_t element_indices;
} MGLBatchCmdStatDelta;

void mgl_batch_issue_cmd_stat_delta(uint32_t cmd_type, int32_t count,
                                    int uses_elements,
                                    MGLBatchCmdStatDelta *out);


/* ---- A3 residual: flush path perf + cmd frame stats + stream driver ---- */

enum {
    MGL_BATCH_FLUSH_PERF_NONE = 0,
    MGL_BATCH_FLUSH_PERF_STREAM = 1,
    MGL_BATCH_FLUSH_PERF_DIRECT = 2
};

/* Which batch/draw PERF pair to bump after schedule (MDI/ICB: none). */
int mgl_batch_flush_scheduled_path_perf_kind(int path);



/* ---- A3 encode-fold: applyDynamicBindings orchestration ---- */

typedef struct MGLBatchDynApplyOps {
    void *ctx;
    void (*refresh_owner)(void *ctx);
    int (*has_encoder)(void *ctx);
    int (*build_dyn_vao)(void *ctx); /* 1 ok; only if vertex_count>0 */
    int (*apply_ubo)(void *ctx);
    int (*apply_tex)(void *ctx); /* fills touched; 1 ok */
    int (*bind_tex_direct)(void *ctx);
    int (*bind_tex_mapper)(void *ctx);
    int (*restore_after_tex_upload)(void *ctx);
    int (*bind_vertex_direct)(void *ctx);
    int (*bind_uniform_direct)(void *ctx);
    int (*mapper_fallback)(void *ctx);
} MGLBatchDynApplyOps;

/* Returns 1 on success. vertex/uniform/texture counts from cmd. */
int mgl_batch_issue_apply_dyn_bindings(uint8_t vertex_count, uint8_t uniform_count,
                                       uint8_t texture_count,
                                       const MGLBatchDynApplyOps *ops);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_ISSUE_H */

