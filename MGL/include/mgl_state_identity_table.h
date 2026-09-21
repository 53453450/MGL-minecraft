/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_state_identity_table.h — T9-1 draw-identity coverage table.
 *
 * Single source that names which GLMState members participate in:
 *   - MGLStateKey fields (deferred batch merge identity)
 *   - hash domains (texture / vertex / render / uniform)
 *   - hot snapshot regions (mglCopyHotStateFields)
 *
 * Hot *copy* stays region-based for speed (contiguous memcpy); this table is
 * the human/oracle map that key writers, hash builders, and G4 coverage
 * must stay aligned with.  Expanding a row here without updating the
 * corresponding writer is a STATE_DATAFLOW regression.
 */

#ifndef MGL_STATE_IDENTITY_TABLE_H
#define MGL_STATE_IDENTITY_TABLE_H

/* Hash-domain bits (combinable). */
#define MGL_ID_HASH_TEX  (1u << 0)
#define MGL_ID_HASH_VAO  (1u << 1)
#define MGL_ID_HASH_RS   (1u << 2)
#define MGL_ID_HASH_UBO  (1u << 3)

/* _X(state_member, key_member_or_, hash_bits, in_hot_snapshot)
 * key_member_or_ is the MGLStateKey field name, or `_` when the member is
 * only folded into a hash (no dedicated key field). */
#define MGL_STATE_IDENTITY_ROWS(_X) \
    /* ---- explicit key fields (mglComputeStateKey) -------------------- */ \
    _X(program_name,              program_name,              MGL_ID_HASH_RS,  1) \
    _X(program_pipeline,          program_pipeline_name,      MGL_ID_HASH_RS,  1) \
    _X(program,                   vertex_program_name,        MGL_ID_HASH_RS,  1) \
    _X(program,                   fragment_program_name,      MGL_ID_HASH_RS,  1) \
    _X(vao,                       vao_name,                   MGL_ID_HASH_VAO, 1) \
    _X(framebuffer,               fbo_name,                   0u,              1) \
    _X(viewport,                  viewport,                   MGL_ID_HASH_RS,  1) \
    _X(var,                       scissor,                    0u,              1) \
    _X(caps,                      scissor_enabled,            0u,              1) \
    _X(caps,                      caps_flags,                 MGL_ID_HASH_RS,  1) \
    _X(_,                         primitive_type,             0u,              0) \
    _X(_,                         _padding,                   0u,              0) \
    _X(active_textures,           texture_hash,               MGL_ID_HASH_TEX, 1) \
    _X(texture_units,             texture_hash,               MGL_ID_HASH_TEX, 1) \
    _X(texture_samplers,          texture_hash,               MGL_ID_HASH_TEX, 1) \
    _X(image_units,               texture_hash,               MGL_ID_HASH_TEX, 1) \
    _X(caps,                      render_state_hash,          MGL_ID_HASH_RS,  1) \
    _X(var,                       render_state_hash,          MGL_ID_HASH_RS,  1) \
    _X(draw_buffer,               render_state_hash,          MGL_ID_HASH_RS,  1) \
    _X(draw_buffers,              render_state_hash,          MGL_ID_HASH_RS,  1) \
    _X(buffer_base,               uniform_buffer_hash,        MGL_ID_HASH_UBO | MGL_ID_HASH_RS, 1) \
    _X(vao,                       vertex_layout_hash,         MGL_ID_HASH_VAO, 1) \
    _X(current_vertex_attrib,     vertex_layout_hash,         MGL_ID_HASH_VAO, 1) \
    /* ---- hot-only (snapshot, not a dedicated key field) -------------- */ \
    _X(shaders,                   _,                          0u,              1) \
    _X(transform_feedback,        _,                          0u,              1) \
    _X(pack,                      _,                          0u,              1) \
    _X(unpack,                    _,                          0u,              1) \
    _X(vertex_buffer_map_list,    _,                          0u,              1) \
    _X(fragment_buffer_map_list,  _,                          0u,              1) \
    _X(compute_buffer_map_list,   _,                          0u,              1) \
    _X(hints,                     _,                          0u,              1)

#endif /* MGL_STATE_IDENTITY_TABLE_H */
