/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_buffer_backend.h -- access to the backend state of a Buffer.
 *
 * Buffer::backend (MGLBufferBackend, see mgl_backend_objects.h) holds the
 * buffer's Metal buffer, its copy-on-write snapshot pool and the expanded
 * index copy.  Readers use mglBufferMtl, which is NULL for a buffer the
 * backend has not created anything for; writers use the ensure accessor,
 * which allocates on first use.  If that allocation fails the writer gets a
 * scratch sink instead of NULL, so callers stay straight-line and the buffer
 * simply keeps nothing.  The struct is calloc'd and freed by
 * mglRenderReleaseBufferBackend, from C and C++ alike.
 */

#ifndef MGL_BUFFER_BACKEND_H
#define MGL_BUFFER_BACKEND_H

#include <stdlib.h>
#include <string.h>

#include "glm_context.h"
#include "mgl_backend_objects.h"

/* The Metal buffer kept for `buffer`, or NULL. */
static inline void *mglBufferMtl(const Buffer *buffer)
{
    return buffer && buffer->backend ? buffer->backend->mtl_data : NULL;
}

/* Same as mglBufferMtl: the buffer's Metal object (borrowed), or NULL before
 * it is bound. */
static inline void *mglBufferMetalStore(const Buffer *buffer)
{
    return mglBufferMtl(buffer);
}

static inline struct MGLBufferBackend *
mglBufferBackendEnsure(Buffer *buffer)
{
    static struct MGLBufferBackend sink;
    if (!buffer) {
        memset(&sink, 0, sizeof(sink));
        return &sink;
    }
    if (!buffer->backend) {
        buffer->backend = (struct MGLBufferBackend *)calloc(
            1, sizeof(struct MGLBufferBackend));
        if (!buffer->backend) {
            memset(&sink, 0, sizeof(sink));
            return &sink;
        }
    }
    return buffer->backend;
}

/* Whether the backend's no-copy MTLBuffer deallocator owns the CPU range in
 * BufferData (so the GL layer must not vm_deallocate it). */
static inline int mglBufferBackendOwnsCPUStorage(const Buffer *buffer)
{
    return (buffer && buffer->backend && buffer->backend->owns_cpu_storage)
               ? 1 : 0;
}

static inline void mglBufferBackendSetOwnsCPUStorage(Buffer *buffer, int owns)
{
    mglBufferBackendEnsure(buffer)->owns_cpu_storage = owns ? 1u : 0u;
}

#endif /* MGL_BUFFER_BACKEND_H */
