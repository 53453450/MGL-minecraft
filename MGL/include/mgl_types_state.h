/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * mgl_types_state.h
 * MGL
 *
 * GL state (GLMState) and dirty-bit definitions split from glm_context.h.
 */

#ifndef mgl_types_state_h
#define mgl_types_state_h

#include <string.h>
#include <stddef.h>

#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_vertex.h"
#include "mgl_types_program.h"
#include "mgl_types_framebuffer.h"
#include "mgl_types_sync.h"
#include "hash_table.h"
#include "draw_command.h"

typedef struct GLSLState_t {
    glslang_resource_t  resrc;
    glslang_limits_t    limits;
} GLSLState;

enum {
    dirtyVAO = 0,
    dirtyState,
    dirtyBuffer,
    dirtyTexture,
    dirtyTexParam,
    dirtyTexBinding,
    dirtySampler,
    dirtyShader,
    dirtyProgram,
    dirtyFBO,
    dirtyDrawable,
    dirtyRenderState,
    dirtyAlphaState,
    dirtyImageUnit,
    dirtyBufferBase,
    maxDirtyState,
    dirtyAllBit = 31
};

#define DIRTY_VAO       (0x1 << dirtyVAO)
#define DIRTY_STATE     (0x1 << dirtyState)
#define DIRTY_BUFFER    (0x1 << dirtyBuffer)
#define DIRTY_TEX       (0x1 << dirtyTexture)
#define DIRTY_TEX_PARAM   (0x1 << dirtyTexParam)
#define DIRTY_TEX_BINDING (0x1 << dirtyTexBinding)
#define DIRTY_SAMPLER (0x1 << dirtySampler)
#define DIRTY_SHADER    (0x1 << dirtyShader)
#define DIRTY_PROGRAM   (0x1 << dirtyProgram)
#define DIRTY_FBO       (0x1 << dirtyFBO)
#define DIRTY_DRAWABLE      (0x1 << dirtyDrawable)
#define DIRTY_RENDER_STATE  (0x1 << dirtyRenderState)
#define DIRTY_ALPHA_STATE   (0x1 << dirtyAlphaState)
#define DIRTY_IMAGE_UNIT_STATE   (0x1 << dirtyImageUnit)
#define DIRTY_BUFFER_BASE_STATE   (0x1 << dirtyBufferBase)
#define DIRTY_ALL_BIT   ((unsigned)0x1 << dirtyAllBit)    // so we know the dirty all was set.
#define DIRTY_ALL       (0xFFFFFFFF)

/* State-key hash domains. Keep invalidation and recomputation on the same
 * masks so renderer-side dirty-bit consumption cannot silently stale a cache. */
#define MGL_TEXTURE_HASH_DIRTY_BITS \
    (DIRTY_TEX_BINDING | DIRTY_SAMPLER)
#define MGL_VERTEX_LAYOUT_HASH_DIRTY_BITS (DIRTY_VAO)
#define MGL_RENDER_STATE_HASH_DIRTY_BITS \
    (DIRTY_RENDER_STATE | DIRTY_BUFFER_BASE_STATE | DIRTY_PROGRAM)

typedef struct {
    GLuint dirty_bits;

    // clear request clear_bitmask from glClear to Metal
    // NOTE: clear_bitmask is deprecated - clears are recorded per-FBO/attachment
    GLbitfield  clear_bitmask;

    // Default framebuffer clear state (used when framebuffer == NULL)
    GLbitfield  default_fbo_clear_bitmask;
    GLfloat     default_clear_color[4];

    // opengl state

    // keep these out of the var struct for debugging and access

    /* Error queue — per GL 4.6 spec, the error queue must hold at least 16
     * distinct errors.  When the queue is full, new errors are dropped (the
     * spec guarantees at least 16 are retained).  error_head is the index of
     * the next error to return from glGetError; error_count is the number of
     * queued errors.  The legacy `error` field mirrors the head for backwards
     * compatibility with code that reads/writes it directly. */
    #define MGL_ERROR_QUEUE_SIZE 16
    GLenum error;   // glGetError (mirrors error_queue[error_head] for legacy code)
    GLenum error_queue[MGL_ERROR_QUEUE_SIZE];
    GLuint error_head;
    GLuint error_count;

    GLuint draw_buffer; // GL_DRAW_BUFFER / GL_DRAW_BUFFER0
    GLsizei draw_buffer_count;
    GLenum draw_buffers[MAX_COLOR_ATTACHMENTS];
    GLuint read_buffer; // GL_READ_BUFFER
    GLuint default_draw_buffer;
    GLsizei default_draw_buffer_count;
    GLenum default_draw_buffers[MAX_COLOR_ATTACHMENTS];
    GLuint default_read_buffer;
    GLuint max_color_attachments; // GL_MAX_COLOR_ATTACHMENTS
    GLuint max_vertex_attribs; // GL_MAX_VERTEX_ATTRIBS
    GLint viewport[4]; // GL_VIEWPORT
    GLfloat viewport_array[MGL_MAX_VIEWPORTS][4];
    GLint scissor_box_array[MGL_MAX_VIEWPORTS][4];
    GLdouble depth_range_array[MGL_MAX_VIEWPORTS][2];
    GLfloat color_clear_value[4]; // GL_COLOR_CLEAR_VALUE

    Buffer *buffers[MAX_BINDABLE_BUFFERS];
    // Compatibility slot for VAO 0 element-array binding.
    Buffer *default_vao_element_array_buffer;
    // Proxy texture probe state per texture target/index (capability query, no allocation).
    ProxyTextureQueryState proxy_texture_query[_MAX_TEXTURE_TYPES];

    VertexArray *vao;
    Texture     *tex;
    Renderbuffer *renderbuffer;
    Framebuffer *framebuffer;
    Framebuffer *readbuffer;

    GLuint      active_texture; // GL_ACTIVE_TEXTURE
    unsigned    active_texture_mask[4];
    Texture     *active_textures[TEXTURE_UNITS];
    TextureUnit texture_units[TEXTURE_UNITS];
    Texture     *last_sampled_2d_textures[TEXTURE_UNITS];
    Texture     *recent_sampled_2d_textures[TEXTURE_UNITS][MGL_RECENT_SAMPLED_2D_HISTORY];
    Sampler     *texture_samplers[TEXTURE_UNITS];
    ImageUnit   image_units[TEXTURE_UNITS];

    GLsizei sync_name;

    /* tracks live Sync objects so destroyGLMContext can release
     * their Metal resources. Placed in the HashTable block (skipped by
     * mglCopyHotStateFields) because internal keys/states arrays may be
     * reallocated, making shallow snapshot copies unsafe. */
    HashTable sync_table;

    HashTable vao_table;
    HashTable buffer_table;
    HashTable texture_table;
    HashTable shader_table;
    HashTable program_table;
    HashTable program_pipeline_table;
    HashTable transform_feedback_table;
    HashTable renderbuffer_table;
    HashTable framebuffer_table;
    HashTable sampler_table;

    Shader      *shaders[_MAX_SHADER_TYPES];
    Program     *program;
    GLuint      program_name;
    ProgramPipeline *program_pipeline;
    TransformFeedback *transform_feedback;

    BufferBase  buffer_base[_MAX_BUFFER_TYPES];

    // glsl info
    GLSLState   glsl;

    // pixel pack unpack
    PixelStore  pack;
    PixelStore  unpack;
    
    // metal buffer mappings
    BufferMapList vertex_buffer_map_list;

    CurrentVertexAttrib current_vertex_attrib[MAX_ATTRIBS];
    BufferMapList fragment_buffer_map_list;
    BufferMapList compute_buffer_map_list;

    // enable / disable caps
    GLMCaps     caps;

    GLboolean conditional_render_active;
    GLboolean conditional_render_skip;
    GLuint    conditional_render_query;
    GLenum    conditional_render_mode;
    GLboolean query_depth_known;
    GLfloat   query_depth_value;

    // hints
    GLMHints    hints;

    /* Dirty-Flag Hash Optimization
     * Cache hash values to avoid recomputing on every draw when state unchanged.
     * dirty flags indicate which hashes need recomputation. */
    uint64_t cached_texture_hash;
    uint64_t cached_vertex_layout_hash;
    uint64_t cached_render_state_hash;
    uint64_t cached_uniform_buffer_hash;
    uint8_t  texture_dirty;
    uint8_t  vertex_layout_dirty;
    uint8_t  render_state_dirty;
    uint8_t  uniform_buffer_dirty;
    uint8_t  _hash_cache_padding[3];

    // put at end, big chunk of yuck
    GLMParams   var;
} GLMState;

/* State-key invalidation is latched when a live mutation is made. Renderer
 * consumption of the legacy bits must not alter those independent flags. */
static inline void mglInvalidateStateHashCachesForDirtyBits(GLMState *state,
                                                            GLuint dirty_bits)
{
    if (!state || dirty_bits == 0u) return;

    if ((dirty_bits & MGL_TEXTURE_HASH_DIRTY_BITS) != 0u)
        state->texture_dirty = 1;
    if ((dirty_bits & MGL_VERTEX_LAYOUT_HASH_DIRTY_BITS) != 0u)
        state->vertex_layout_dirty = 1;
    if ((dirty_bits & MGL_RENDER_STATE_HASH_DIRTY_BITS) != 0u) {
        state->render_state_dirty = 1;
        /* uniform_buffer_hash inputs (buffer_base[_UNIFORM_BUFFER]) are a
         * strict subset of render_state_hash inputs, so the same dirty bits
         * invalidate both.  Kept as a separate cache field because
         * mglStateKeysEqualIgnoringUniformRanges XORs it back out of
         * render_state_hash during batch merge comparisons. */
        state->uniform_buffer_dirty = 1;
    }
}

static inline void mglMarkRendererDirtyBits(GLMState *state, GLuint dirty_bits)
{
    if (!state || dirty_bits == 0u) return;
    state->dirty_bits |= dirty_bits;
}

/* State-key input mutations preserve the renderer's legacy dirty bits and
 * also invalidate every cache domain fed by those bits. */
static inline void mglMarkStateDirtyBits(GLMState *state, GLuint dirty_bits)
{
    mglMarkRendererDirtyBits(state, dirty_bits);
    mglInvalidateStateHashCachesForDirtyBits(state, dirty_bits);
}

static inline void mglClearStateDirtyBitsPreservingHashInvalidation(GLMState *state)
{
    if (!state) return;
    state->dirty_bits = 0;
}

/* === Selective state snapshot helpers ===
 *
 * GLMState is 82KB, but only ~51KB of fields are read by the Metal encoder
 * during batch replay.  The cold regions are:
 *   - 11 embedded HashTables (sync_table + 10 others, ~2KB): restored from savedState at replay time
 *     (snapshot copies are stale — internal keys/states arrays may have been
 *     reallocated)
 *   - 11 of 16 buffer_base types (29.6KB): never read by the encoder during
 *     graphics replay.  Only _UNIFORM_BUFFER, _UNIFORM_CONSTANT,
 *     _SHADER_STORAGE_BUFFER, _TRANSFORM_FEEDBACK_BUFFER, and
 *     _ATOMIC_COUNTER_BUFFER are accessed by mapShaderBufferResourcesToBufferMap.
 *
 * Using mglCopyHotStateFields instead of full sizeof(GLMState) memcpy saves
 * ~37.5% (31.6KB) per snapshot creation and per restore.
 *
 * Cold buffer_base types need no restore: the hot copy skips them and
 * nothing in replay writes them, so active_state keeps the live values.
 * Only the HashTables are fixed up at the restore call site. */

/* mglCopyHotStateFields copies GLMState in memcpy regions that identify the
 * skipped ranges (the embedded HashTable block and the cold buffer_base
 * slots) purely by offsetof arithmetic.  A field reorder or insertion in
 * those ranges would silently drop data from the snapshot, so lock the
 * layout assumptions into compile errors. */

/* The gap [sync_table, shaders) skipped by region 2 must be exactly the 11
 * embedded HashTables; a hot field inserted there would not be copied. */
_Static_assert(offsetof(GLMState, shaders) - offsetof(GLMState, sync_table)
               == 11 * sizeof(HashTable),
               "GLMState HashTable block changed; revisit mglCopyHotStateFields region boundaries");

/* The gap [buffer_base, glsl) skipped between regions 4 and 5 must be exactly
 * buffer_base[_MAX_BUFFER_TYPES]; region 5 assumes glsl follows the array. */
_Static_assert(offsetof(GLMState, glsl) - offsetof(GLMState, buffer_base)
               == _MAX_BUFFER_TYPES * sizeof(BufferBase),
               "field inserted between buffer_base and glsl; revisit mglCopyHotStateFields region 5");

/* The hot list (mglCopyHotStateFields) and cold list are both expanded from
 * the X-macros in mgl_types_buffer.h; this checks that together they cover
 * every buffer_base type exactly once, so adding a type without classifying
 * it hot or cold breaks the build. */
_Static_assert(kMGLSnapshotHotBufferBaseCount + kMGLSnapshotColdBufferBaseCount
               == _MAX_BUFFER_TYPES,
               "buffer type added without classifying it hot or cold; update "
               "MGL_SNAPSHOT_HOT/COLD_BUFFER_BASE_TYPES in mgl_types_buffer.h");

static inline void mglCopyHotStateFields(GLMState *dst, const GLMState *src)
{
    if (!dst || !src || dst == src) return;

    /* Region 1: [0, sync_table) — everything before the HashTable block. */
    memcpy(dst, src, offsetof(GLMState, sync_table));

    /* Region 2: skip 11 HashTables (sync_table .. sampler_table inclusive).
     * Region 3: [shaders, buffer_base) — small gap: shaders, program,
     * program_pipeline, transform_feedback. */
    {
        size_t gap_start = offsetof(GLMState, shaders);
        size_t gap_end   = offsetof(GLMState, buffer_base);
        memcpy((char *)dst + gap_start,
               (char *)src + gap_start,
               gap_end - gap_start);
    }

    /* Region 4: copy only the hot buffer_base types read by the encoder.
     * The hot set is defined once in mgl_types_buffer.h and expanded here. */
#define MGL_SNAPSHOT_COPY_HOT(_t_) dst->buffer_base[_t_] = src->buffer_base[_t_];
    MGL_SNAPSHOT_HOT_BUFFER_BASE_TYPES(MGL_SNAPSHOT_COPY_HOT)
#undef MGL_SNAPSHOT_COPY_HOT

    /* Region 5: [glsl, end) — everything after buffer_base. */
    {
        size_t post_start = offsetof(GLMState, glsl);
        size_t post_size  = sizeof(GLMState) - post_start;
        memcpy((char *)dst + post_start,
               (char *)src + post_start,
               post_size);
    }
}

#endif /* mgl_types_state_h */
