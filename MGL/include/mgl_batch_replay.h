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
 * and direct-path primitive plans. ObjC only materializes MTL* and calls
 * set*Bytes / draw* encode ports.
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

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_REPLAY_H */
