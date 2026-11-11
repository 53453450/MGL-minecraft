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
    /* 1 when the fence is known complete with no CB to wait on (empty
     * command stream at FenceSync). Distinguishes "vacuous SIGNALED" from
     * "GetSync failed / no backend" which must stay UNSIGNALED (AUDIT-044). */
    int completed;
};

/* Program: what the AIR backend built for one linked stage.  The metallib
 * bytes are the serialized output of the compile; the Metal functions and
 * libraries are created from them on first use. */
struct MGLShaderModuleBackend {
    unsigned char *metallib_bytes;
    size_t metallib_size;
    /* VS: tessellation capture.  FS: GS/TES gl_PrimitiveID carrier alongside
     * the direct primitive-id variant. */
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

/* Texture: the base-level view of the texture's Metal texture.
 * newTextureViewWithPixelFormat: is expensive and would run per draw whenever
 * base_level != 0, so the view is kept, keyed on (source texture, base level,
 * max level, swizzle); a mismatch on any of them rebuilds it. */
struct MGLBaseLevelViewCache {
    void *view;   /* retained MTL::Texture, or NULL */
    void *source; /* the Metal texture the view was made from (identity only) */
    uint32_t base;
    uint32_t max;
    uint32_t swizzle_r;
    uint32_t swizzle_g;
    uint32_t swizzle_b;
    uint32_t swizzle_a;
};

struct MGLTextureBackend {
    /* The texture's Metal texture (retained), or NULL until the backend first
     * creates it. */
    void *mtl_data;
    /* The texture's own MTLSamplerState, built from its parameters and
     * rebuilt when they change; retained. */
    void *sampler_state;
    struct MGLBaseLevelViewCache base_level_view;
    /* A texture view's Metal texture is a view of its owner's; this is the
     * retained owner texture the view was last built from, compared to see
     * whether the view must be rebuilt. */
    void *view_parent;
    /* The current Metal texture is the fallback texture, not real storage. */
    uint8_t storage_is_fallback;
    /* Generation of the last render-pass write (0 = never written). */
    uint32_t render_target_write_version;
    /* Renderer frame in which a clear last stamped the texture (0 = never). */
    uint32_t rt_frame_generation;
};

/* Image unit: the Metal texture view (or 3D slice staging texture) bound for
 * non-layered or mip image access; released on rebind or reset so the GPU can
 * finish with the previous one. */
struct MGLImageUnitBackend {
    void *view; /* retained MTL::Texture, or NULL */
};

/* Buffer: cold state that is not the buffer's own MTLBuffer. */
struct MGLBufferBackend {
    /* The buffer's Metal buffer (retained), or NULL until the backend first
     * creates it. */
    void *mtl_data;
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
    /* Primitive-restart / draw-mode element emulation (line loop, fan, …).
     * Keyed on content_version + byte offset + count + type + kind + mode +
     * restart index; same eligibility as uint16_expanded_data. */
    void *element_emulation_data;
    uint64_t element_emulation_content_version;
    size_t element_emulation_byte_offset;
    size_t element_emulation_index_count;
    uint32_t element_emulation_type;
    uint32_t element_emulation_kind;
    uint32_t element_emulation_mode;
    uint32_t element_emulation_restart_index;
    size_t element_emulation_out_index_count;
    /* The no-copy MTLBuffer's deallocator owns the CPU range in BufferData. */
    uint8_t owns_cpu_storage;
    /* Generation of the last encode that referenced mtl_data (0 = never).
     * Lets unpooled Shared stores take the same idle in-place upload path as
     * CoW pool slots once gBufferCompletedGeneration catches up. */
    uint64_t mtl_last_use_generation;
};

#ifdef __cplusplus
}
#endif

#endif /* MGL_BACKEND_OBJECTS_H */
