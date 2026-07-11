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
 * mgl_types_vertex.h
 * MGL
 *
 * Vertex array domain type definitions split from glm_context.h.
 */

#ifndef mgl_types_vertex_h
#define mgl_types_vertex_h

#include "mgl_types_buffer.h"

#define DIRTY_VAO_BUFFER_BASE  0x1
#define DIRTY_VAO_ATTRIB       (DIRTY_VAO_BUFFER_BASE << 1)

typedef struct BufferBinding_t {
    Buffer  *buffer;
    GLintptr offset;
    GLsizei stride;
    GLuint divisor;
} BufferBinding;

typedef struct VertexAttrib_t {
    Buffer  *buffer;
    GLuint  size;
    GLenum  type;
    GLuint  normalized;
    GLuint  integer;
    GLuint  long_attribute;
    GLuint  stride;
    GLuint  divisor;
    GLintptr  relativeoffset;
    GLintptr  binding_offset;
    GLuint  buffer_bindingindex;
} VertexAttrib;

typedef struct CurrentVertexAttrib_t {
    GLfloat f[4];
    GLint i[4];
    GLuint u[4];
    GLdouble d[4];
    GLenum type;
    GLuint integer;
    GLuint long_attribute;
} CurrentVertexAttrib;

typedef struct VertexElementArray_t {
    Buffer  *buffer;
    GLenum  type;
    GLuint  size;
    const void *ptr;
} VertexElementArray;

#define MGL_VAO_MAGIC 0x56414F31u

typedef struct VertexArray_t {
    uint32_t magic;
    GLuint dirty_bits;
    unsigned name;
    unsigned enabled_attribs;
    BufferBinding bindings[MGL_MAX_VERTEX_ATTRIB_BINDINGS];
    VertexAttrib attrib[MAX_ATTRIBS];
    VertexElementArray element_array;
    void *mtl_data;
    GLboolean transient_batch_vao;
} VertexArray;

#endif /* mgl_types_vertex_h */
