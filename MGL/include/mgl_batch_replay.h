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
 * mgl_batch_replay.h — BatchReplay stage/bind + draw/MDI plans (O2.3/O2.5).
 *
 * Dynamic VAO / UBO range / texture-unit override expansion, resource
 * binding snapshot collection, MDI arg packing, simple-replay eligibility,
 * direct-path primitive plans, stream path, sampler params, dyn-vertex
 * stream plans, and GLMState HashTable sync (A3). ObjC only materializes
 * MTL* and calls set*Bytes / draw* encode ports.
 */

#ifndef MGL_BATCH_REPLAY_H
#define MGL_BATCH_REPLAY_H

/* glm_context.h must precede draw_command.h: the latter expects GL base
 * types (GLenum/GLuint/GLintptr/...) to already be declared. */
#include "glm_context.h"
#include "draw_command.h"
#include "mgl_batch_issue.h"
#include "mgl_draw_encode.h"
#include "mgl_types_vertex.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MGLRenderResourceBindingSnapshot_t;
struct Program_t;
struct VertexAttrib_t;
struct Buffer_t;
struct TextureParameter_t;
struct MGLRenderReplayBatchCommand_t;

/* Append one texture/sampler op to a stage's snapshot list. Returns false if
 * stage/kind invalid or the stage op table is full. */
bool mgl_batch_replay_collect_resource_binding(
    struct MGLRenderResourceBindingSnapshot_t *snapshot, uint32_t stage,
    uint32_t kind, void *resource, uint32_t index);

/* Expand cmd->dynamic_vertex_bindings onto a VAO copy of base. */
bool mgl_batch_replay_build_dynamic_vertex_array(GLMContext ctx,
                                                 const VertexArray *base,
                                                 const MGLDrawCommand *cmd,
                                                 VertexArray *out);

/* True when an attrib can be rebound via direct Metal setVertexBuffer. */
bool mgl_batch_replay_attrib_can_bind_directly(struct Program_t *active_program,
                                               GLuint attrib_index,
                                               const struct VertexAttrib_t *attrib);

/* Apply captured UBO range overrides onto buffer_base[_UNIFORM_BUFFER]. */
bool mgl_batch_replay_apply_uniform_range_overrides(GLMContext ctx,
                                                    const MGLDrawCommand *cmd);

/* Apply captured texture-unit overrides. touched_units must be at least
 * TEXTURE_UNITS bools (cleared by caller or zeroed here for touched flags). */
bool mgl_batch_replay_apply_texture_overrides(GLMContext ctx,
                                              const MGLDrawCommand *cmd,
                                              bool *touched_units,
                                              uint32_t touched_units_count);

/* ---- O2.5: MDI / simple / direct orchestration (no Metal) ---- */

enum {
    MGL_BATCH_MDI_OK = 0,
    MGL_BATCH_MDI_FALLBACK_DISABLED = 1,
    MGL_BATCH_MDI_FALLBACK_BAD_PRIM = 2,
    MGL_BATCH_MDI_FALLBACK_OVERFLOW = 3,
    MGL_BATCH_MDI_FALLBACK_EMPTY = 4
};

/* Gate MDI before allocating scratch. Sets *arg_size / *needed_bytes on OK. */
int mgl_batch_replay_mdi_gate(const MGLDrawBatch *batch, int disable_mdi,
                              size_t *arg_size, size_t *needed_bytes);

const char *mgl_batch_replay_mdi_gate_reason(int gate);

/* Pack Metal indirect-draw arg structs from batch commands.
 * Indexed fill returns 0 if mixed index types (caller falls back). */
int mgl_batch_replay_fill_mdi_indexed_args(
    const MGLDrawBatch *batch,
    MGLDrawIndexedPrimitivesIndirectArguments *args);
void mgl_batch_replay_fill_mdi_array_args(
    const MGLDrawBatch *batch,
    MGLDrawPrimitivesIndirectArguments *args);
void mgl_batch_replay_fill_stream_mdi_indexed_args(
    const MGLDrawBatch *batch,
    MGLDrawIndexedPrimitivesIndirectArguments *args);

/* True when tryReplaySimpleBatch may proceed (encoder/mtl resolve still ObjC). */
int mgl_batch_replay_simple_eligible(const MGLDrawBatch *batch,
                                     uint32_t max_commands,
                                     int has_active_encoder,
                                     int uses_cull_distance,
                                     int primitive_restart,
                                     int polygon_mode_point,
                                     int mode_needs_emulate);

typedef struct MGLBatchReplayDirectPrimPlan {
    uint8_t polygon_mode_point;
    uint8_t emulate_triangle_fan;
    uint8_t emulate_line_loop;
    uint8_t emulate_quads;
    uint8_t skip_unsupported_prim;
    uint32_t prim_type;
} MGLBatchReplayDirectPrimPlan;

/* Compute emulate flags + Metal prim type for one direct-batch command. */
void mgl_batch_replay_direct_prim_plan(uint32_t mode, int polygon_mode_point,
                                       uint32_t batch_primitive_type,
                                       MGLBatchReplayDirectPrimPlan *out);

enum {
    MGL_BATCH_ICB_OK = 0,
    MGL_BATCH_ICB_UNAVAILABLE = 1,
    MGL_BATCH_ICB_BAD_PRIM = 2,
    MGL_BATCH_ICB_DISABLED = 3
};

int mgl_batch_replay_icb_gate(const MGLDrawBatch *batch, int has_device,
                              int has_encoder, int icb_enable, int icb_disable);
const char *mgl_batch_replay_icb_gate_reason(int gate);

/* ---- A3 / O2.5: stream path, sampler params, dyn-vertex streams, simple cmd ---- */

enum {
    MGL_BATCH_STREAM_EMPTY = 1,
    MGL_BATCH_STREAM_BAD_PRIM = 2,
    MGL_BATCH_STREAM_TRY_MDI = 3,
    MGL_BATCH_STREAM_DIRECT = 4
};

/* Decide stream-merge issue path before Metal materialize. */
int mgl_batch_replay_stream_path(const MGLDrawBatch *batch, int disable_mdi);
const char *mgl_batch_replay_stream_path_reason(int path);

/* Fill TextureParameter fields from an immutable sampler snapshot key. */
void mgl_batch_replay_fill_sampler_params(const MGLSamplerSnapshotKey *key,
                                          struct TextureParameter_t *out);

enum {
    MGL_BATCH_DYN_VERTEX_UNUSED = 0, /* no streams used by shader — skip bind */
    MGL_BATCH_DYN_VERTEX_OK = 1,
    MGL_BATCH_DYN_VERTEX_FAIL = -1
};

#define MGL_BATCH_DYN_VERTEX_MAX_STREAMS 16u

typedef struct MGLBatchDynVertexStreamPlan {
    uint32_t binding_index;
    uint32_t stream_count;
    uint32_t representative_attribs[MGL_BATCH_DYN_VERTEX_MAX_STREAMS];
    uint32_t representative_strides[MGL_BATCH_DYN_VERTEX_MAX_STREAMS];
    struct Buffer_t *buffer;
    uint64_t dynamic_offset;
} MGLBatchDynVertexStreamPlan;

/* Plan stream representatives for one dynamic vertex binding (no Metal). */
int mgl_batch_replay_plan_dyn_vertex_streams(
    GLMContext ctx, const VertexArray *vao, struct Program_t *active_program,
    const MGLDynamicVertexBinding *override_binding,
    MGLBatchDynVertexStreamPlan *out);

/* True when every attrib on stream_index can bind via setVertexBuffer. */
int mgl_batch_replay_dyn_vertex_stream_can_bind_directly(
    struct Program_t *active_program, const VertexArray *vao,
    const MGLBatchDynVertexStreamPlan *plan, uint32_t stream_index);

/* Validate UBO override against Metal buffer length (no Metal types). */
int mgl_batch_replay_uniform_range_fits(uint64_t offset, uint64_t size,
                                        uint64_t buffer_length);

int mgl_batch_replay_cmd_is_array_draw(uint32_t cmd_type);

/* Fill common fields of a simple-replay command (index buffer still ObjC). */
void mgl_batch_replay_fill_simple_cmd_common(
    const MGLDrawCommand *cmd, struct MGLRenderReplayBatchCommand_t *out);

/* Copy the 10 replay HashTables (not sync_table) from src → dst. */
void mgl_batch_replay_copy_object_hash_tables(GLMState *dst,
                                              const GLMState *src);

/* Teardown: sync HashTable structs that may have grown via shared storage. */
void mgl_batch_replay_sync_hash_tables_from_replay(GLMState *live,
                                                   const GLMState *replay);


/* ---- A3 residual: uniform / sampled-texture materialize plans ---- */

enum { MGL_BATCH_UNIFORM_BIND_MAX_OPS = 64 };

typedef struct MGLBatchUniformBindOp {
    uint8_t is_vertex_stage;
    uint32_t metal_slot;
    uint64_t offset;
    uint32_t binding_index; /* UBO binding index for buf re-resolve */
} MGLBatchUniformBindOp;

typedef struct MGLBatchUniformBindPlan {
    uint32_t count;
    MGLBatchUniformBindOp ops[MGL_BATCH_UNIFORM_BIND_MAX_OPS];
} MGLBatchUniformBindPlan;

/* Plan VS/FS set*Buffer ops for dynamic UBO ranges.
 * mtl_lengths[i] is Metal buffer length for dynamic_uniform_bindings[i]
 * (0 = missing/invalid → fail). min_binding_bytes is kMGLMinimumStageBindingSize.
 * Returns 1 on success, 0 on fail (out may be partially filled). */
int mgl_batch_replay_plan_uniform_binds(GLMContext ctx,
                                        const MGLDrawCommand *cmd,
                                        const uint64_t *mtl_lengths,
                                        uint32_t mtl_lengths_count,
                                        uint64_t min_binding_bytes,
                                        uint32_t max_buffer_slots,
                                        MGLBatchUniformBindPlan *out);

enum { MGL_BATCH_SAMPLED_TEX_MAX = 64 };

typedef struct MGLBatchSampledTexCandidate {
    int32_t stage;
    uint32_t resource_index;
    uint32_t metal_slot;
    uint32_t expected_type;
    uint32_t lookup_type;
    uint32_t expected_kind;
    uint8_t needs_combined_sampler;
    struct MGLShaderResource_t *resource;
} MGLBatchSampledTexCandidate;

typedef struct MGLBatchSampledTexPlan {
    uint32_t count;
    MGLBatchSampledTexCandidate entries[MGL_BATCH_SAMPLED_TEX_MAX];
} MGLBatchSampledTexPlan;

/* Enumerate non-skipped sampled-image resources for VS/FS.
 * Returns 0 if any non-skipped resource is an array (ObjC must fail),
 * 1 on success (possibly empty). */
int mgl_batch_replay_plan_sampled_texture_candidates(
    GLMContext ctx, MGLBatchSampledTexPlan *out);


/* A3 encode-fold: dyn-vertex offset / sampler slot gates. */
int mgl_batch_replay_dyn_vertex_offset_ok(int64_t binding_offset,
                                          uint64_t dynamic_offset,
                                          uint64_t metal_length);
int mgl_batch_replay_sampler_slot_ok(uint32_t metal_slot, uint32_t max_slots);
int mgl_batch_replay_cmd_is_elements_draw(uint32_t cmd_type);


/* ---- A3 residual: dyn-bind materialize gates ---- */

int mgl_batch_replay_mtl_ptr_ok(const void *mtl_data);
int mgl_batch_replay_dyn_vertex_slot_ok(int resolved_slot, int max_slots);

/* Texture object ready for direct sampled bind (no Metal types). */
int mgl_batch_replay_sampled_tex_object_ok(int has_texture, int has_mtl,
                                           int dirty, int is_render_target);

/* After GetTextureInfo: type/kind gates. expected_type==0 skips type check. */
int mgl_batch_replay_sampled_tex_info_ok(int has_texture_info_ok,
                                         uint32_t texture_type,
                                         uint32_t expected_type,
                                         int pixel_format_compatible);


/* Dyn sampled resolve gate: 0 skip, 1 ok no-samp, 2 ok needs-samp, -1 fail. */
typedef struct MGLBatchSampledResolveGateIn {
    int unit_ok; /* unit in range + touched */
    int has_tex;
    int has_mtl;
    int dirty;
    int is_rt;
    int info_ok;
    uint64_t texture_type;
    uint32_t expected_type;
    int format_compat;
    int needs_combined_sampler;
    int has_sampler_mtl;
} MGLBatchSampledResolveGateIn;

int mgl_batch_replay_sampled_resolve_gate(const MGLBatchSampledResolveGateIn *in);


/* ---- A3 residual: flush cmd stats + stream-merged driver ---- */

typedef struct MGLBatchCmdFrameStats {
    uint32_t array_draws;
    uint64_t array_vertices;
    uint32_t element_draws;
    uint64_t element_indices;
} MGLBatchCmdFrameStats;

void mgl_batch_flush_accum_cmd_frame_stats(const MGLDrawBatch *batch,
                                           MGLBatchCmdFrameStats *out);

typedef struct MGLBatchStreamMergedOps {
    void *ctx;
    void (*trace_cmd0)(void *ctx, const char *phase, const char *reason);
    int (*try_stream_mdi)(void *ctx);
    void (*issue_direct)(void *ctx);
    int (*resolve_stream_index)(void *ctx, void **mtl_index_out);
    void (*draw_stream_indexed)(void *ctx, void *mtl_index);
} MGLBatchStreamMergedOps;

void mgl_batch_issue_stream_merged(const MGLDrawBatch *batch, int disable_mdi,
                                   const MGLBatchStreamMergedOps *ops);

/* ---- A3 encode-fold: direct-batch command loop (plan in C) ---- */

typedef struct MGLBatchDirectCmdView {
    uint32_t type;
    uint32_t mode;
    int32_t count;
    int32_t first;
    int32_t instance_count;
    uint32_t base_instance;
} MGLBatchDirectCmdView;

typedef struct MGLBatchDirectIssueOps {
    void *ctx;
    void (*refresh_encoder)(void *ctx);
    int (*try_simple_replay)(void *ctx);
    uint32_t (*command_count)(void *ctx);
    void (*fill_cmd)(void *ctx, uint32_t i, MGLBatchDirectCmdView *out);
    int (*uses_cull_distance)(void *ctx);
    uint32_t (*batch_primitive_type)(void *ctx);
    int (*snapshots_mixed)(void *ctx);
    int (*has_dyn_texture)(void *ctx);
    int (*cull_capture)(void *ctx, uint32_t cmd_index, int cull_path);
    int (*after_cull_ok)(void *ctx);
    int (*apply_dyn_bindings)(void *ctx, uint32_t cmd_index);
    int (*apply_cmd_sampler)(void *ctx, uint32_t cmd_index);
    int (*polygon_mode_point)(void *ctx, uint32_t mode);
    void (*trace_skip)(void *ctx, uint32_t cmd_index, const char *reason);
    /* Finer MTL hooks; submit decision trees in mgl_batch_issue_*. */
    MGLBatchDirectArraySubmitOps array_submit;
    MGLBatchDirectElementSubmitOps element_submit;
} MGLBatchDirectIssueOps;

void mgl_batch_issue_direct_batch(const MGLBatchDirectIssueOps *ops);

/* ---- A3 encode-fold: dyn vertex / uniform bind loops ---- */

typedef struct MGLBatchDynVertexBindOps {
    void *ctx;
    uint8_t binding_count;
    int max_metal_slots;
    void *binding_state_owner;
    void *render_encoder_owner;
    /* Returns plan rc: FAIL / UNUSED / OK (see mgl_batch_replay_plan_dyn_vertex*). */
    int (*plan_binding)(void *ctx, uint8_t binding_index,
                        MGLBatchDynVertexStreamPlan *plan_out);
    int (*resolve_slot)(void *ctx, uint32_t attrib_index, int *slot_out);
    int (*stream_can_bind)(void *ctx, const MGLBatchDynVertexStreamPlan *plan,
                           uint32_t stream);
    /* Upload/bind MTL; set mtl/gl/dyn_offset/mtl_length. Return 0 fail. */
    int (*ensure_mtl)(void *ctx, const MGLBatchDynVertexStreamPlan *plan,
                      void **mtl_out, void **gl_out, uint64_t *dyn_offset_out,
                      uint64_t *mtl_length_out);
    uint64_t (*vao_binding_offset)(void *ctx,
                                   const MGLBatchDynVertexStreamPlan *plan);
} MGLBatchDynVertexBindOps;

int mgl_batch_mtl_bind_dyn_vertex(const MGLBatchDynVertexBindOps *ops);

typedef struct MGLBatchDynUniformBindOps {
    void *ctx;
    void *binding_state_owner;
    void *render_encoder_owner;
    uint64_t min_stage_binding_size;
    uint32_t max_buffer_slots;
    /* Fill mtl_lengths[0..count). Return 0 fail. */
    int (*gather_lengths)(void *ctx, uint64_t *mtl_lengths, uint32_t count);
    const MGLDrawCommand *cmd; /* non-null; GLM ctx via gather/plan */
    GLMContext glm_ctx;
    /* After plan: resolve op → mtl/gl. Return 0 fail. */
    int (*resolve_op)(void *ctx, const MGLBatchUniformBindOp *op, void **mtl_out,
                      void **gl_out);
} MGLBatchDynUniformBindOps;

int mgl_batch_mtl_bind_dyn_uniforms(const MGLBatchDynUniformBindOps *ops);

typedef struct MGLBatchDynSampledBindOps {
    void *ctx;
    void *binding_state_owner;
    void *render_encoder_owner;
    uint32_t max_sampler_slots;
    GLMContext glm_ctx;
    /* Return 0 to skip candidate; 1 append texture (+optional sampler). */
    int (*resolve_candidate)(void *ctx, const MGLBatchSampledTexCandidate *e,
                             const bool *touched_units, void **texture_out,
                             uint32_t *binding_stage_out, int *needs_sampler_out,
                             void **sampler_out, uint32_t *sampler_slot_out);
    const bool *touched_units;
} MGLBatchDynSampledBindOps;

int mgl_batch_mtl_bind_dyn_sampled(const MGLBatchDynSampledBindOps *ops);


#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_REPLAY_H */

