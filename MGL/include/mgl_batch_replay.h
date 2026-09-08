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

#include "draw_command.h"
#include "mgl_draw_encode.h"
#include "glm_context.h"
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

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_REPLAY_H */

