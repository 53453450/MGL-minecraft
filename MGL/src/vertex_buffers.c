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
 * vertex_buffers.c
 * MGL
 *
 */

#include <assert.h>
#include <strings.h>

#include "glm_context.h"

extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern Buffer *getBuffer(GLMContext ctx, GLenum target, GLuint buffer);
extern int isVAO(GLMContext ctx, GLuint vao);
extern VertexArray *getVAO(GLMContext ctx, GLuint vao);
extern void mglGenVertexArrays(GLMContext ctx, GLsizei n, GLuint *arrays);

bool bindVertexBuffer(GLMContext ctx, GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    VertexArray *vao;

    if (vaobj)
    {
        vao = getVAO(ctx, vaobj);
        // no such vao
        ERROR_CHECK_RETURN_VALUE(vao, GL_INVALID_VALUE, false);
    }
    else
    {
        vao = ctx->state.vao;
        // no vao bound
        ERROR_CHECK_RETURN_VALUE(vao, GL_INVALID_VALUE, false);
    }

    ERROR_CHECK_RETURN_VALUE(offset >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(stride >= 0, GL_INVALID_VALUE, false);

    Buffer *buf = NULL;
    if (buffer != 0)
    {
        buf = findBuffer(ctx, buffer);
        if (!buf)
        {
            buf = getBuffer(ctx, GL_ARRAY_BUFFER, buffer);
        }
        ERROR_CHECK_RETURN_VALUE(buf, GL_INVALID_VALUE, false);
    }

    GLboolean changed =
        vao->bindings[bindingindex].buffer != buf ||
        vao->bindings[bindingindex].offset != offset ||
        vao->bindings[bindingindex].stride != stride;
    if (!changed)
        return true;

    mglFlushPendingDrawsForVertexArray(ctx, vao);

    vao->bindings[bindingindex].buffer = buf;
    vao->bindings[bindingindex].offset = offset;
    vao->bindings[bindingindex].stride = stride;

    vao->dirty_bits |= DIRTY_VAO_BUFFER_BASE;
    if (ctx->state.vao == vao) {
        ctx->state.dirty_bits |= DIRTY_VAO;
    }

    return true;
}

void mglBindVertexBuffer(GLMContext ctx, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    ERROR_CHECK_RETURN(ctx->state.vao, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS, GL_INVALID_VALUE);

    bindVertexBuffer(ctx, 0, bindingindex, buffer, offset, stride);
}

void mglBindVertexBuffers(GLMContext ctx, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizei *strides)
{
    ERROR_CHECK_RETURN(ctx->state.vao, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(first + count <= MGL_MAX_VERTEX_ATTRIB_BINDINGS, GL_INVALID_VALUE);
    if (buffers)
    {
        ERROR_CHECK_RETURN(offsets, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(strides, GL_INVALID_VALUE);
    }

    for(int i=0; i<count; i++)
    {
        GLuint bindingindex;
        GLuint buffer;
        GLintptr offset = 0;
        GLsizei stride = 0;

        bindingindex = first + i;
        buffer = buffers ? buffers[i] : 0;
        if (buffers)
        {
            offset = offsets[i];
            stride = strides[i];
        }

        bindVertexBuffer(ctx, 0, bindingindex, buffer, offset, stride);
    }
}


/*
glBindVertexBuffer and glVertexArrayVertexBuffer bind the buffer named buffer
to the vertex buffer binding point whose index is given by bindingindex.
glBindVertexBuffer modifies the binding of the currently bound vertex array
object, whereas glVertexArrayVertexBuffer allows the caller to specify ID of
the vertex array object with an argument named vaobj, for which the binding
should be modified. offset and stride specify the offset of the first element
within the buffer and the distance between elements within the buffer,
respectively, and are both measured in basic machine units. bindingindex
must be less than the value of GL_MAX_VERTEX_ATTRIB_BINDINGS. offset and
stride must be greater than or equal to zero. If buffer is zero, then any
buffer currently bound to the specified binding point is unbound.

If buffer is not the name of an existing buffer object, the GL first creates
a new state vector, initialized with a zero-sized memory buffer and comprising
all the state and with the same initial values as in case of glBindBuffer.
buffer is then attached to the specified bindingindex of the vertex array object.
 */

void mglVertexArrayVertexBuffer(GLMContext ctx, GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    if (vaobj)
    {
        ERROR_CHECK_RETURN(isVAO(ctx, vaobj), GL_INVALID_OPERATION);
    }
    else
    {
        ERROR_CHECK_RETURN(ctx->state.vao, GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS, GL_INVALID_VALUE);

    bindVertexBuffer(ctx, vaobj, bindingindex, buffer, offset, stride);
}

void mglVertexArrayVertexBuffers(GLMContext ctx, GLuint vaobj, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizei *strides)
{
    if (vaobj)
    {
        ERROR_CHECK_RETURN(isVAO(ctx, vaobj), GL_INVALID_OPERATION);
    }
    else
    {
        ERROR_CHECK_RETURN(ctx->state.vao, GL_INVALID_OPERATION);
    }
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(first + count <= MGL_MAX_VERTEX_ATTRIB_BINDINGS, GL_INVALID_VALUE);
    if (buffers)
    {
        ERROR_CHECK_RETURN(offsets, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(strides, GL_INVALID_VALUE);
    }

    for(int i=0; i<count; i++)
    {
        GLuint bindingindex;
        GLuint buffer;
        GLintptr offset = 0;
        GLsizei stride = 0;

        bindingindex = first + i;
        buffer = buffers ? buffers[i] : 0;
        if (buffers)
        {
            offset = offsets[i];
            stride = strides[i];
        }

        bindVertexBuffer(ctx, vaobj, bindingindex, buffer, offset, stride);
    }
}
