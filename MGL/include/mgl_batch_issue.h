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

/* glm_context.h must precede draw_command.h: the latter expects GL base types
 * (GLenum/GLuint/GLintptr/...) to already be declared. */
#include "mgl_batch_restore.h"
#include "glm_context.h"
#include "draw_command.h"
#include "mgl_encode_context.h"
#include "mgl_types_vertex.h"   /* VertexArray */
#include <stdbool.h>

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

/* ---- A3 encode-fold: flush batch loop + check-execute drivers ---- */

typedef struct MGLBatchFlushLoopState {
    uint8_t last_key_valid;
    uint8_t last_execute_ok;
    uint8_t last_was_stream;
    MGLBatchFlushPathStats path_stats;
} MGLBatchFlushLoopState;

typedef struct MGLBatchFlushLoopOps {
    void *ctx;
    uint32_t (*batch_count)(void *ctx);
    uint32_t (*command_count)(void *ctx, uint32_t batch_index);
    /* Fill skip POD except last_* (runner stamps those). *want_abs out. */
    void (*fill_skip_in)(void *ctx, uint32_t batch_index,
                         MGLBatchSameKeySkipIn *in, int *want_abs);
    void (*note_skip_perf)(void *ctx, int skip_dec);
    int (*oracle_keys_equal)(void *ctx, uint32_t batch_index);
    void (*on_oracle_would_skip)(void *ctx);
    void (*apply_same_key_skip)(void *ctx, uint32_t batch_index);
    void (*set_absolute_offsets)(void *ctx, int want_abs);
    void (*restore)(void *ctx, uint32_t batch_index, uint32_t forced_dirty);
    int (*check_execute)(void *ctx, uint32_t batch_index); /* 0 → continue */
    void (*mark_execute_ok)(void *ctx, uint32_t batch_index);
    int (*schedule)(void *ctx, uint32_t batch_index); /* MGL_BATCH_SELECT_* */
    void (*trace_phase)(void *ctx, uint32_t batch_index, const char *phase);
    void (*perf_stream)(void *ctx, uint32_t command_count);
    void (*perf_direct)(void *ctx, uint32_t command_count);
    void (*issue_stream)(void *ctx, uint32_t batch_index);
    void (*issue_mdi)(void *ctx, uint32_t batch_index);
    void (*issue_icb)(void *ctx, uint32_t batch_index);
    void (*issue_direct)(void *ctx, uint32_t batch_index);
    void (*record_stats)(void *ctx, uint32_t batch_index);
    uint32_t vao_buffer_dirty_mask; /* DIRTY_VAO|DIRTY_BUFFER */
    uint8_t skip_enabled;
    uint8_t oracle_env_enabled;
} MGLBatchFlushLoopOps;

void mgl_batch_flush_run_batches(MGLBatchFlushLoopState *st,
                                 const MGLBatchFlushLoopOps *ops);

typedef struct MGLBatchCheckExecOps {
    void *ctx;
    void (*begin_trace)(void *ctx); /* set flush/batch + RESTORE phase */
    int (*prepare_fbo)(void *ctx);  /* 0 → skip fbo_rotation */
    int (*process_gl_state)(void *ctx); /* 0 → skip; may set error */
    void (*capture_error_if_any)(void *ctx);
    int (*should_apply_sampler)(void *ctx);
    int (*apply_sampler)(void *ctx); /* 0 → skip sampler_snapshot */
    void (*trace_ready)(void *ctx);
    int (*empty_raster)(void *ctx);
    int (*fully_culled)(void *ctx);
    void (*apply_polygon_offset)(void *ctx);
    /* Returns 0 always; records skip. phase/reason are literals. */
    int (*trace_skip)(void *ctx, const char *phase, const char *reason);
} MGLBatchCheckExecOps;

/* Returns 1 if batch should execute, 0 if skipped. */
int mgl_batch_check_should_execute(const MGLBatchCheckExecOps *ops);

/* Trace each cmd as SKIP and accumulate skipped_commands. */
void mgl_batch_flush_trace_skip_commands(
    uint32_t command_count, void (*trace_cmd)(void *ctx, uint32_t i), void *ctx,
    uint32_t *skipped_commands_inout);

/* Walk 128-bit active_texture_mask; bind_unit returns 0 fail, sets *stale. */
typedef struct MGLBatchActiveTexBindOps {
    void *ctx;
    const unsigned *mask4; /* 4×32 bits */
    int (*bind_unit)(void *ctx, uint32_t unit, int *stale_out);
    void (*clear_stale)(void *ctx, uint32_t word, uint32_t bit);
} MGLBatchActiveTexBindOps;

int mgl_batch_bind_active_textures(const MGLBatchActiveTexBindOps *ops);

/* ---- A3: direct submit decision trees (MTL hooks via ops) ---- */

typedef struct MGLBatchDirectElementPrep {
    void *gl_buffer;
    void *mtl_buffer;
    uint64_t index_offset;
    uint32_t gl_index_type;
    uint32_t mtl_index_type;
    int32_t base_vertex;
    uint32_t base_instance;
    const uint8_t *cull_index_bytes;
} MGLBatchDirectElementPrep;

typedef struct MGLBatchDirectArraySubmitOps {
    void *ctx;
    int (*try_cull_array_split)(void *ctx, uint32_t cmd_index, uint32_t mode,
                                int32_t first, int32_t count, int32_t ic,
                                uint32_t base_instance);
    void (*bind_cull_emu_arrays)(void *ctx, uint32_t mode, int32_t first);
    int (*encode_arrays)(void *ctx, uint32_t mode, int32_t first, int32_t count,
                         int32_t ic, uint32_t base_instance);
    void (*on_trace)(void *ctx, uint32_t cmd_index, const char *phase,
                     const char *reason);
} MGLBatchDirectArraySubmitOps;

typedef struct MGLBatchDirectElementSubmitOps {
    void *ctx;
    int (*prepare_element)(void *ctx, uint32_t cmd_index,
                           MGLBatchDirectElementPrep *out);
    int (*polygon_mode_line)(void *ctx, uint32_t mode);
    int (*try_cull_element_split)(void *ctx, uint32_t cmd_index, uint32_t mode,
                                  const MGLBatchDirectElementPrep *prep,
                                  int32_t count, int32_t ic, int poly_line);
    int (*encode_elements)(void *ctx, uint32_t mode,
                           const MGLBatchDirectElementPrep *prep, int32_t count,
                           int32_t ic);
    void (*on_trace)(void *ctx, uint32_t cmd_index, const char *phase,
                     const char *reason);
} MGLBatchDirectElementSubmitOps;

void mgl_batch_issue_submit_direct_arrays(
    uint32_t cmd_index, uint32_t mode, int32_t first, int32_t count, int32_t ic,
    uint32_t base_instance, int poly_pt, int uses_cull, const char *reason,
    const char *cull_reason, const MGLBatchDirectArraySubmitOps *ops);

void mgl_batch_issue_submit_direct_elements(uint32_t cmd_index, uint32_t mode,
                                            int32_t count, int32_t ic,
                                            int poly_pt,
                                            const MGLBatchDirectElementSubmitOps *ops);

/* syncResourceBindingsForContext sequence. */
typedef struct MGLBatchSyncResourceOps {
    void *ctx;
    int mapped_buffers_done;
    int updated_base_lists_done;
    int bound_active_textures_done;
    int (*map_buffers)(void *ctx);
    int (*update_vertex_base)(void *ctx);
    int (*update_fragment_base)(void *ctx);
    int (*bind_vertex_buffers)(void *ctx);
    int (*bind_fragment_buffers)(void *ctx);
    int (*bind_buffer_size_constants)(void *ctx);
    int (*bind_active_textures)(void *ctx);
    int (*restore_after_active_tex)(void *ctx);
    int (*bind_textures)(void *ctx);
    int (*restore_after_sampled)(void *ctx);
} MGLBatchSyncResourceOps;

int mgl_batch_sync_resource_bindings(const MGLBatchSyncResourceOps *ops);

/* Renderer-side drivers of the MDI / direct plans above.  They used to be
 * -[MGLRenderer issueMDIBatch:context:encodeContext:] and
 * -[MGLRenderer issueDirectBatch:context:encodeContext:]; the loops now live
 * in C and reach the renderer through the mgl_draw_issue.h /
 * mgl_renderer_ports.h ports. */
void mglBatchIssueMDIBatch(void *renderer, MGLDrawBatch *batch,
                           GLMContext glm_ctx,
                           const MGLEncodeContext *encode_context);
void mglBatchIssueDirectBatch(void *renderer, MGLDrawBatch *batch,
                              GLMContext glm_ctx,
                              const MGLEncodeContext *encode_context);

/* Stream-MDI and indirect-command-buffer issue (mgl_batch_icb_mdi_encode.c).
 * 1 = issued, 0 = the plan declined / fell back. */
int mglBatchIssueStreamMergedMDIBatch(void *renderer, MGLDrawBatch *batch,
                                      GLMContext glm_ctx,
                                      const MGLEncodeContext *encode_context);
int mglBatchIssueIndirectCommandBufferBatch(void *renderer, MGLDrawBatch *batch,
                                            GLMContext glm_ctx,
                                            const MGLEncodeContext *encode_context);

/* Dynamic-binding / sampler-snapshot / simple-replay drivers
 * (mgl_batch_dyn_bind_encode.c).  1 = applied, 0 = declined (the caller then
 * takes its fallback path). */
int mglBatchDynBindVertexDirect(void *renderer, VertexArray *vao,
                                const MGLDrawCommand *cmd, GLMContext glm_ctx,
                                const MGLEncodeContext *encode_context);
int mglBatchDynBindUniformDirect(void *renderer, const MGLDrawCommand *cmd,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encode_context);
int mglBatchDynBindSampledDirect(void *renderer, const bool *touched_units,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encode_context);
int mglBatchApplySamplerSnapshot(void *renderer, const MGLDrawCommand *cmd,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encode_context);
int mglBatchApplyDynamicBindings(void *renderer, const MGLDrawCommand *cmd,
                                 GLMContext glm_ctx,
                                 MGLEncodeContext *encode_context);
int mglBatchTryReplaySimpleBatch(void *renderer, MGLDrawBatch *batch,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encode_context);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_ISSUE_H */

