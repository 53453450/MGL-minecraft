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

#import "mgl_buffer_query.h"

BOOL mglRendererSameVertexStream(Buffer *lhsBuffer,
                                        GLintptr lhsOffset,
                                        GLuint lhsStride,
                                        GLuint lhsDivisor,
                                        Buffer *rhsBuffer,
                                        GLintptr rhsOffset,
                                        GLuint rhsStride,
                                        GLuint rhsDivisor)
{
    /* Compare offsets as well: attributes that share the same buffer/stride/
     * divisor but have different binding offsets (e.g. interleaved vertex
     * arrays submitted via a single VBO with per-attribute glVertexAttribPointer
     * offsets) must get separate Metal buffer slots. This is required for
     * CPU-converted attribs (GL_DOUBLE, GL_INT→float, integer signedness
     * mismatch), where each attrib produces its own converted buffer starting
     * at its binding_offset. If they shared a slot, the second binding would
     * overwrite the first. */
    if (!lhsBuffer || !rhsBuffer ||
        lhsOffset != rhsOffset ||
        lhsStride != rhsStride ||
        lhsDivisor != rhsDivisor) {
        return NO;
    }

    return lhsBuffer == rhsBuffer ||
           (lhsBuffer->name == rhsBuffer->name && lhsBuffer->target == rhsBuffer->target);
}

BOOL mglRendererBufferMayHaveMappedWrites(Buffer *buffer)
{
    if (!buffer || !buffer->mapped) {
        return NO;
    }

    if ((buffer->access_flags & GL_MAP_WRITE_BIT) != 0) {
        return YES;
    }

    return buffer->access == GL_WRITE_ONLY || buffer->access == GL_READ_WRITE;
}

BOOL mglRendererBufferHasDrawableContents(Buffer *buffer)
{
    if (!buffer) {
        return NO;
    }

    if (buffer->ever_written || buffer->has_initialized_data) {
        return YES;
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
