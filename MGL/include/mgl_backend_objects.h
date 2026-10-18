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

/* Program: what the AIR backend built for one linked stage.  The metallib
 * bytes are the serialized output of the compile; the Metal functions and
 * libraries are created from them on first use. */
struct MGLShaderModuleBackend {
    unsigned char *metallib_bytes;
    size_t metallib_size;
    unsigned char *metallib_tess_capture_bytes;
    size_t metallib_tess_capture_size;
    unsigned char *metallib_cull_capture_bytes;
    size_t metallib_cull_capture_size;
    /* TES render-vertex / compute dual-emit: when the TES is an isolines or
     * point_mode program eligible for the render-vertex path, the compute
     * expansion kernel is also stored here so indexed draws (glDrawElements /
     * glMultiDrawElements) can fall back to the verified compute path, which
     * carries the gather ABI the render-vertex ABI lacks. */
    unsigned char *metallib_bytes_tes_compute;
    size_t metallib_size_tes_compute;
    void *mtl_function;
    void *mtl_library;
    void *mtl_tess_capture_function;
    void *mtl_tess_capture_library;
    void *mtl_cull_capture_function;
    void *mtl_cull_capture_library;
    /* Cached Metal function for metallib_bytes_tes_compute (TES kernel). */
    void *mtl_function_compute;
    uint8_t needs_runtime_array_size_buffer;
};

/* One slot per shader stage (_MAX_SHADER_TYPES; checked in
 * mgl_program_backend.h). */
#define MGL_PROGRAM_BACKEND_STAGES 6

struct MGLProgramBackend {
    struct MGLShaderModuleBackend modules[MGL_PROGRAM_BACKEND_STAGES];
    /* Pipeline cache validity for this program: the cache instance that
     * last built for it and the generation of that build. */
    uint64_t pipeline_cache_instance_id;
    uint64_t pipeline_cache_generation;
};

/* Sampler: the MTLSamplerState built from the sampler's parameters. */
struct MGLSamplerBackend {
    void *state; /* retained MTLSamplerState, or NULL until it is first needed */
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
