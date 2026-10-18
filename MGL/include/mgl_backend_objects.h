/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_backend_objects.h -- state the Metal backend keeps for a GL object.
 *
 * A GL object (Sync, Buffer, ...) carries one `struct MGL<Object>Backend *`
 * that the GL layer only stores and never dereferences: the type is
 * incomplete in the GL type headers (mgl_types_*.h).  The backend allocates it
 * the first time it has something to keep, and releases it through the
 * matching mglRenderRelease*() entry, which also frees the struct.
 *
 * Include this header from backend code (mgl_render*.cpp and tests that look
 * at backend state), not from the GL layer.
 */

#ifndef MGL_BACKEND_OBJECTS_H
#define MGL_BACKEND_OBJECTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sync: the command buffer holding every GL command issued before the fence.
 * The wait blocks on its completion. */
struct MGLSyncBackend {
    void *command_buffer; /* retained MTL::CommandBuffer, or NULL */
};

/* Buffer: cold state that is not the buffer's own MTLBuffer. */
struct MGLBufferBackend {
    /* Copy-on-write snapshot pool: an opaque C++ allocation (mgl::BufferCowPool),
     * not a CF object. */
    void *cow_pool;
    /* UInt16 copy of an unsigned-byte element buffer, for APIs without
     * 8-bit indices.  Valid while uint16_expanded_src_hash equals the buffer's
     * last_write_src_hash and the byte count matches; the same invalidation
     * contract as the drawElements index-range scan cache.  Never kept for
     * persistent-mapped buffers or buffers without CPU-side data: their bytes
     * can change without a GL call.  Retained MTL::Buffer. */
    void *uint16_expanded_data;
    uint64_t uint16_expanded_src_hash;
    size_t uint16_expanded_byte_count;
};

#ifdef __cplusplus
}
#endif

#endif /* MGL_BACKEND_OBJECTS_H */
