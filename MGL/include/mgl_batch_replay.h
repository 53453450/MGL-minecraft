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
 * mgl_batch_replay.h — BatchReplay stage/bind expansion (O2.3).
 *
 * Dynamic VAO / UBO range / texture-unit override expansion and resource
 * binding snapshot collection. ObjC only materializes MTL* and calls
 * set*Bytes / draw* encode ports.
 */

#ifndef MGL_BATCH_REPLAY_H
#define MGL_BATCH_REPLAY_H

#include "draw_command.h"
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

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_REPLAY_H */
