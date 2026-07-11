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

#endif /* mgl_types_state_h */
