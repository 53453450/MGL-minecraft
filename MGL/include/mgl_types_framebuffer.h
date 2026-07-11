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
 * mgl_types_framebuffer.h
 * MGL
 *
 * Framebuffer / renderbuffer domain type definitions split from glm_context.h.
 */

#ifndef mgl_types_framebuffer_h
#define mgl_types_framebuffer_h

#include "mgl_types_texture.h"

#define DIRTY_FBO_BINDING   0x1
#define DIRTY_FBO_TEX      (DIRTY_FBO_BINDING << 1)

#define DIRTY_RENDBUF       0x1
#define DIRTY_RENDBUF_TEX   (DIRTY_RENDBUF << 1)

typedef struct Renderbuffer_t {
    GLuint dirty_bits;
    GLuint  name;
    GLboolean is_draw_buffer;
    Texture *tex;
} Renderbuffer;

typedef struct FBOAttachment_t {
    GLuint dirty_bits;
    GLuint textarget;   // GL_RENDERBUFFER for renderbuffers
    GLuint texture;
    GLuint level;
    GLuint layer;
    GLboolean layered;
    GLbitfield clear_bitmask;
    GLfloat clear_color[4];
    union {
        Texture *tex;
        Renderbuffer *rbo;
    } buf;
} FBOAttachment;

typedef struct Framebuffer_t {
    GLuint dirty_bits;
    GLuint  name;
    GLbitfield color_attachment_bitfield;
    GLuint draw_buffer;
    GLsizei draw_buffer_count;
    GLenum draw_buffers[MAX_COLOR_ATTACHMENTS];
    GLuint read_buffer;
    FBOAttachment color_attachments[MAX_COLOR_ATTACHMENTS];
    FBOAttachment depth;
    FBOAttachment stencil;
    // Default framebuffer parameters (for FBOs with no attachments)
    GLint default_width;
    GLint default_height;
    GLint default_layers;
    GLint default_samples;
    GLboolean default_fixed_sample_locations;
} Framebuffer;

#endif /* mgl_types_framebuffer_h */
