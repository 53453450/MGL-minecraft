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

    /* P0-4B: tracks live Sync objects so destroyGLMContext can release
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
    
    // put at end, big chunk of yuck
    GLMParams   var;
} GLMState;

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
 * Cold fields must be restored via mglRestoreColdBufferBase after restore,
 * using the pre-replay live state (savedState).  HashTable restoration is
 * already done at the restore call site. */

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

    /* Region 4: copy only the 5 hot buffer_base types read by the encoder. */
    dst->buffer_base[_UNIFORM_BUFFER]            = src->buffer_base[_UNIFORM_BUFFER];
    dst->buffer_base[_UNIFORM_CONSTANT]          = src->buffer_base[_UNIFORM_CONSTANT];
    dst->buffer_base[_SHADER_STORAGE_BUFFER]     = src->buffer_base[_SHADER_STORAGE_BUFFER];
    dst->buffer_base[_TRANSFORM_FEEDBACK_BUFFER] = src->buffer_base[_TRANSFORM_FEEDBACK_BUFFER];
    dst->buffer_base[_ATOMIC_COUNTER_BUFFER]     = src->buffer_base[_ATOMIC_COUNTER_BUFFER];

    /* Region 5: [glsl, end) — everything after buffer_base. */
    {
        size_t post_start = offsetof(GLMState, glsl);
        size_t post_size  = sizeof(GLMState) - post_start;
        memcpy((char *)dst + post_start,
               (char *)src + post_start,
               post_size);
    }
}

/* Restore the 11 cold buffer_base types from savedState.  Called after
 * mglCopyHotStateFields during batch replay restore, alongside the existing
 * HashTable fixup.  Without this, the cold buffer_base slots in active_state
 * retain whatever values were present before restore (live state from prior
 * replay iteration), which may differ from the draw-time state. */
static inline void mglRestoreColdBufferBase(GLMState *dst, const GLMState *savedState)
{
    if (!dst || !savedState) return;

    dst->buffer_base[_TEXTURE_BUFFER]            = savedState->buffer_base[_TEXTURE_BUFFER];
    dst->buffer_base[_ARRAY_BUFFER]              = savedState->buffer_base[_ARRAY_BUFFER];
    dst->buffer_base[_ELEMENT_ARRAY_BUFFER]      = savedState->buffer_base[_ELEMENT_ARRAY_BUFFER];
    dst->buffer_base[_QUERY_BUFFER]              = savedState->buffer_base[_QUERY_BUFFER];
    dst->buffer_base[_PIXEL_PACK_BUFFER]         = savedState->buffer_base[_PIXEL_PACK_BUFFER];
    dst->buffer_base[_PIXEL_UNPACK_BUFFER]       = savedState->buffer_base[_PIXEL_UNPACK_BUFFER];
    dst->buffer_base[_COPY_READ_BUFFER]          = savedState->buffer_base[_COPY_READ_BUFFER];
    dst->buffer_base[_COPY_WRITE_BUFFER]         = savedState->buffer_base[_COPY_WRITE_BUFFER];
    dst->buffer_base[_DISPATCH_INDIRECT_BUFFER]  = savedState->buffer_base[_DISPATCH_INDIRECT_BUFFER];
    dst->buffer_base[_DRAW_INDIRECT_BUFFER]      = savedState->buffer_base[_DRAW_INDIRECT_BUFFER];
    dst->buffer_base[_PARAMETER_BUFFER]          = savedState->buffer_base[_PARAMETER_BUFFER];
}

#endif /* mgl_types_state_h */
