/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_buffer_transient.c -- backend-made stand-in Buffer objects.  The Metal
 * side supplies the store; the shape of a GL Buffer lives here.
 */

#include "glm_context.h"

#include <stdlib.h>

Buffer *mglBufferTransientCreate(GLuint name)
{
    Buffer *buffer = (Buffer *)calloc(1, sizeof(Buffer));
    if (!buffer) {
        return NULL;
    }
    buffer->name = name;
    buffer->target = GL_UNIFORM_BUFFER;
    buffer->usage = GL_STATIC_DRAW;
    buffer->written_min = -1;
    buffer->written_max = -1;
    buffer->transient_batch_buffer = GL_TRUE;
    return buffer;
}

void *mglBufferTransientAdopt(Buffer *buffer, void *mtl_buffer,
                              size_t padded_size)
{
    void *previous = buffer->data.mtl_data;
    buffer->data.mtl_data = mtl_buffer;
    buffer->size = (GLsizeiptr)padded_size;
    buffer->data.buffer_data = 0;
    buffer->data.buffer_size = padded_size;
    buffer->data.dirty_bits = 0;
    buffer->data.mtl_owns_buffer_data = GL_FALSE;
    buffer->has_initialized_data = GL_TRUE;
    buffer->ever_written = GL_TRUE;
    return previous;
}
