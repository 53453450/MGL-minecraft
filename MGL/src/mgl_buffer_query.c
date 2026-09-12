/*
 * mgl_buffer_query.m
 * MGL
 *
 * Implementation of the Buffer Query Subsystem.
 * See mgl_buffer_query.h for the API contract.
 *
 * Function bodies are preserved verbatim from MGLRenderer.m; only the
 * "static" storage-class qualifier was removed to make the symbols
 * externally visible.
 */

#include "mgl_buffer_query.h"
#include <stdbool.h>
#include "mgl_render.h"

bool mglRendererSameVertexStream(Buffer *lhsBuffer,
                                        GLintptr lhsOffset,
                                        GLuint lhsStride,
                                        GLuint lhsDivisor,
                                        Buffer *rhsBuffer,
                                        GLintptr rhsOffset,
                                        GLuint rhsStride,
                                        GLuint rhsDivisor)
{
    /* Offset-strict: used for CPU-converted attribs (GL_DOUBLE, GL_INT→float,
     * FIXED/packed, integer signedness mismatch). Each converted Metal buffer
     * starts at that attrib's binding_offset; sharing a slot would overwrite
     * the prior bind. Plain shared-VBO streams coalesce in
     * mglRendererResolveVertexAttributeBufferIndex instead. */
    if (!lhsBuffer || !rhsBuffer ||
        lhsOffset != rhsOffset ||
        lhsStride != rhsStride ||
        lhsDivisor != rhsDivisor) {
        return false;
    }

    return lhsBuffer == rhsBuffer ||
           (lhsBuffer->name == rhsBuffer->name && lhsBuffer->target == rhsBuffer->target);
}

bool mglRendererBufferMayHaveMappedWrites(Buffer *buffer)
{
    if (!buffer || !buffer->mapped) {
        return false;
    }

    if (mglRenderBufferHasMapWriteBit((uint32_t)buffer->access_flags)) {
        return true;
    }

    return mglRenderImageAccessWritable((uint32_t)buffer->access) != 0;
}

bool mglRendererBufferHasDrawableContents(Buffer *buffer)
{
    if (!buffer) {
        return false;
    }

    if (buffer->ever_written || buffer->has_initialized_data) {
        return true;
    }

    /*
     * Persistent/coherent mapped buffers are commonly written directly by the
     * client through the returned pointer. Those writes do not pass through
     * glBufferSubData, so ever_written can remain false even though the Metal
     * backing contains valid stream vertices.
     */
    return mglRendererBufferMayHaveMappedWrites(buffer) &&
           buffer->data.buffer_data != 0 &&
           buffer->size > 0;
}
