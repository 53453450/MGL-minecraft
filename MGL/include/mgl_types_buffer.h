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
 * mgl_types_buffer.h
 * MGL
 *
 * Buffer domain type definitions split from glm_context.h.
 */

#ifndef mgl_types_buffer_h
#define mgl_types_buffer_h

#include <mach/vm_types.h>
#include "glm_params.h"

enum {
    _TEXTURE_BUFFER = 0, // duplicate of _TEXTURE_BUFFER_TARGET
    _ARRAY_BUFFER,
    _ELEMENT_ARRAY_BUFFER,
    _UNIFORM_BUFFER,
    _UNIFORM_CONSTANT,
    _SHADER_STORAGE_BUFFER,
    _TRANSFORM_FEEDBACK_BUFFER,
    _QUERY_BUFFER,
    _PIXEL_PACK_BUFFER,
    _PIXEL_UNPACK_BUFFER,
    _ATOMIC_COUNTER_BUFFER,
    _COPY_READ_BUFFER,
    _COPY_WRITE_BUFFER,
    _DISPATCH_INDIRECT_BUFFER,
    _DRAW_INDIRECT_BUFFER,
    _PARAMETER_BUFFER,
    _MAX_BUFFER_TYPES
};

/* Single source of truth for the batch-snapshot hot/cold buffer_base split.
 * Two sites must agree on this partition: mglCopyHotStateFields copies the
 * hot set into the snapshot (the cold set stays live in active_state through
 * replay, since nothing in replay writes it), and kMGLSnapshotBufferBaseTypes
 * retains/releases the hot set.  Expanding these X-macro lists at each site
 * keeps them in lockstep.  See mgl_types_state.h. */
#define MGL_SNAPSHOT_HOT_BUFFER_BASE_TYPES(_X_) \
    _X_(_UNIFORM_BUFFER) \
    _X_(_UNIFORM_CONSTANT) \
    _X_(_SHADER_STORAGE_BUFFER) \
    _X_(_TRANSFORM_FEEDBACK_BUFFER) \
    _X_(_ATOMIC_COUNTER_BUFFER)

#define MGL_SNAPSHOT_COLD_BUFFER_BASE_TYPES(_X_) \
    _X_(_TEXTURE_BUFFER) \
    _X_(_ARRAY_BUFFER) \
    _X_(_ELEMENT_ARRAY_BUFFER) \
    _X_(_QUERY_BUFFER) \
    _X_(_PIXEL_PACK_BUFFER) \
    _X_(_PIXEL_UNPACK_BUFFER) \
    _X_(_COPY_READ_BUFFER) \
    _X_(_COPY_WRITE_BUFFER) \
    _X_(_DISPATCH_INDIRECT_BUFFER) \
    _X_(_DRAW_INDIRECT_BUFFER) \
    _X_(_PARAMETER_BUFFER)

enum {
    kMGLSnapshotHotBufferBaseCount = 0
#define MGL_SNAPSHOT_COUNT_ONE(_t_) + 1
        MGL_SNAPSHOT_HOT_BUFFER_BASE_TYPES(MGL_SNAPSHOT_COUNT_ONE),
    kMGLSnapshotColdBufferBaseCount = 0
        MGL_SNAPSHOT_COLD_BUFFER_BASE_TYPES(MGL_SNAPSHOT_COUNT_ONE),
#undef MGL_SNAPSHOT_COUNT_ONE
};

enum {
    _UNIFORM_BASE = 0,
    _TRANSFORM_FEEDBACK_BASE,
    _SHADER_STORAGE_BASE,
    _ATOMIC_COUNTER_BASE,
    _MAX_BASE_TARGET
};

#define DIRTY_BUFFER_DATA   0x1
#define DIRTY_BUFFER_ADDR   (DIRTY_BUFFER_DATA << 1)

typedef struct {
    unsigned int  count;
    unsigned int  instanceCount;
    unsigned int  first;
    unsigned int  baseInstance;
} DrawArraysIndirectCommand;

typedef struct {
    unsigned int  count;
    unsigned int  instanceCount;
    unsigned int  first;
    int  baseVertex;
    unsigned int  baseInstance;
} DrawElementsIndirectCommand;

typedef struct BufferData_t {
    GLuint          dirty_bits;
    size_t          buffer_size;
    vm_address_t    buffer_data;
    void            *mtl_data;
    /* True when a no-copy MTLBuffer deallocator owns buffer_data. */
    GLboolean       mtl_owns_buffer_data;
} BufferData;

#define BUFFER_IMMUTABLE_STORAGE_FLAG   0x1
#define BUFFER_MAP_PERSISTENT_BIT       (BUFFER_IMMUTABLE_STORAGE_FLAG << 1)

typedef enum MGLBufferInitSource_t {
    kInitNone = 0,
    kInitBufferDataNull,
    kInitBufferDataCopy,
    kInitBufferSubData,
    kInitCopyBufferSubData,
    kInitReadPixels,
    kInitMapWrite
} MGLBufferInitSource;

typedef struct Buffer_t {
    GLuint name;
    GLenum target;
    GLuint index;
    GLsizeiptr size;
    GLenum usage;
    GLenum access;
    GLbitfield access_flags;
    GLboolean immutable_storage; // GL_BUFFER_IMMUTABLE_STORAGE
    GLboolean mapped;
    GLuint storage_flags; // GL_BUFFER_STORAGE_FLAGS
    GLsizeiptr mapped_offset;
    GLsizeiptr mapped_length;
    BufferData data;
    GLboolean has_initialized_data;
    GLboolean ever_written;
    /* CPU shadow holds map-write bytes not yet uploaded to the Metal buffer.
     * Set on unmap after a writable map; cleared when encoding uploads the
     * shadow or a GPU write is copied back into it.  Read-map refresh from
     * Metal is skipped while set (GL 4.6 §6.3). */
    GLboolean cpu_shadow_pending;
    /* Sticky: the buffer has been bound to a target a shader may write
     * (SSBO/atomic counter/transform feedback), so the Metal store — not the
     * CPU shadow — is authoritative outside the CPU-written range.  Uploads
     * preserve the rest instead of overwriting it from the shadow. */
    GLboolean gpu_write_target;
    GLintptr written_min;
    GLintptr written_max; // exclusive byte offset, -1 when unknown/unwritten
    MGLBufferInitSource last_init_source;
    GLintptr last_write_offset;
    GLsizeiptr last_write_size;
    const void *last_write_src_ptr;
    uint64_t last_write_src_hash;
    /* Cached UInt8→UInt16 expanded index buffer.
     *
     * Metal does not support GL_UNSIGNED_BYTE indices, so the element buffer
     * must be expanded to UInt16 per draw.  This cache stores the expanded
     * MTLBuffer keyed on (last_write_src_hash, sourceByteCount) so that
     * unchanged EBOs skip the calloc + newBufferWithBytes + free on subsequent
     * draws.
     *
     * Invalidation: every GL write path (glBufferData, glBufferSubData,
     * glCopyBufferSubData, glMapBuffer unmap, glFlushMappedBufferRange) calls
     * mglBufferMarkWrite which updates last_write_src_hash.  The cache-hit
     * check compares the stored hash against the current last_write_src_hash.
     *
     * Safety exclusions (skip cache, always rebuild):
     * - Persistent-mapped buffers (storage_flags & GL_MAP_PERSISTENT_BIT):
     *   the app can modify contents through the persistent pointer without
     *   any GL call, so last_write_src_hash would not update.
     * - Buffers without CPU-side buffer_data: the source bytes come from a
     *   Metal MTLBuffer whose contents may be modified by the GPU without
     *   updating the hash.
     *
     * The stored pointer is retained via CFRetain (not CFBridgingRetain) so
     * that ARC retains its own reference in the .m file; released via
     * mglSafeReleaseMetalObj in the .c file (CFRelease under non-ARC). */
    void       *mtl_uint16_expanded_data;       /* id<MTLBuffer> retained via CFRetain */
    uint64_t    mtl_uint16_expanded_src_hash;
    size_t      mtl_uint16_expanded_byte_count;
    /* Cached drawElements index-range scan (min/max) used by the drawElements
     * VBO-range guard.  Keyed on last_write_src_hash + (offset,count,type,
     * restart) — the same invalidation contract as mtl_uint16_expanded_* :
     * only valid while the hash matches AND the bytes came from CPU-side
     * buffer_data (GPU-own buffers and persistent maps can change without a
     * GL call).  Skips the O(count) scan on every draw of unchanged EBOs. */
    uint32_t    scan_cache_min_index;
    uint32_t    scan_cache_max_index;
    uint64_t    scan_cache_src_hash;
    uint64_t    scan_cache_offset;
    uint32_t    scan_cache_count;
    GLenum      scan_cache_type;
    uint32_t    scan_cache_restart_index;
    uint8_t     scan_cache_restart_enabled;
    uint8_t     scan_cache_valid;
    /* ObjC A/B baseline CoW pool. */
    void *mtl_cow_pool;
    /* Metal-cpp CoW pool. This is an opaque C++ allocation, not a CF object;
     * release it only through mglRenderCppReleaseBufferCowPool. */
    void *mtl_cpp_cow_pool;
    void *mapped_ptr;
    GLboolean transient_batch_buffer;
    /* Program-owned storage for a default-block (glUniform*) location.  Small
     * slots are bound with set*Bytes straight from the CPU shadow, so no
     * MTLBuffer is materialized up front; consumers that need one (compute,
     * isolated stage bindings) create it lazily from the current shadow.
     * See updateDirtyBuffer and bindVertexBuffersToCurrentRenderEncoder. */
    GLboolean plain_uniform_slot;
    /* reference count for deferred buffer lifetime management.
     * Follows the Program refcount pattern (program.c:441-475).
     * - newBuffer sets refcount=1 (caller holds initial reference)
     * - mglRetainBufferReference increments
     * - mglReleaseBufferReference decrements; when refcount==0 && delete_status,
     *   releases mtl_data + buffer_data and frees the shell
     * - mglDeleteBuffers sets delete_status=GL_TRUE and calls release; if
     *   refcount>0 (in-flight batch holds reference), shell becomes tombstone
     *   until the last release frees it */
    int refcount;
    GLboolean delete_status;
} Buffer;

/* Buffer reference counting (mirrors Program refcount pattern).
 * Declared here alongside the Buffer type, matching how Program refcount
 * helpers are declared in mgl_types_program.h. */
void mglRetainBufferReference(Buffer *buf);
void mglReleaseBufferReference(GLMContext ctx, Buffer *buf);
void mglReleaseBufferStorage(Buffer *buf);

typedef struct BufferBaseTarget_t {
    GLuint      buffer;
    GLsizeiptr  offset;
    GLsizeiptr  size;
    Buffer      *buf;
} BufferBaseTarget;

#define MAX_BINDABLE_BUFFERS    84
#define MGL_MAX_VERTEX_ATTRIB_BINDINGS MAX_VERTEX_BUFFER_BINDINGS
#define MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX 25u  /* Metal buffer slot for spvBufferSizeConstants */

/* Active-binding bitmap: 84 bits fit in 2 × uint64_t.  Bit i is set iff
 * buffers[i].buf != NULL.  Maintained at bind/unbind/delete time so that
 * mglTrackPendingBaseBufferReads can skip the ~84-slot scan per target and
 * only visit active slots (typically single digits in Minecraft). */
#define MGL_BUFFER_BASE_ACTIVE_WORDS 2u

typedef struct BufferBase_t {
    BufferBaseTarget    buffers[MAX_BINDABLE_BUFFERS];
    uint64_t            active_mask[MGL_BUFFER_BASE_ACTIVE_WORDS];
} BufferBase;

/* Active-binding bitmap helpers.  Called at bind/unbind/delete time to keep
 * active_mask in sync with which slots have buf != NULL.  mglTrackPending-
 * BaseBufferReads uses the bitmap to skip empty slots without scanning. */
static inline void mglBufferBaseSetActive(BufferBase *base, GLuint index)
{
    if (!base || index >= MAX_BINDABLE_BUFFERS) return;
    base->active_mask[index >> 6] |= (uint64_t)1u << (index & 63u);
}

static inline void mglBufferBaseClearActive(BufferBase *base, GLuint index)
{
    if (!base || index >= MAX_BINDABLE_BUFFERS) return;
    base->active_mask[index >> 6] &= ~((uint64_t)1u << (index & 63u));
}

/* Rebuild the bitmap from scratch by scanning buffers[].  Used after bulk
 * operations (e.g. delete cleanup) that may clear multiple slots. */
static inline void mglBufferBaseRebuildActiveMask(BufferBase *base)
{
    if (!base) return;
    base->active_mask[0] = 0u;
    base->active_mask[1] = 0u;
    for (GLuint i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
        if (base->buffers[i].buf) {
            mglBufferBaseSetActive(base, i);
        }
    }
}

typedef struct BufferMap_t {
    GLuint      buffer_base_index;
    GLuint      attribute_mask;
    GLuint      resource_type;
    GLuint      resource_index;
    GLuint      metal_binding_index;
    GLboolean   has_metal_binding;
    Buffer      *buf;
    GLintptr    offset;
    GLsizeiptr  size;
} BufferMap;

static inline GLsizeiptr mglBufferMapStorageRemaining(const BufferMap *map)
{
    if (!map || !map->buf || map->offset < 0 ||
        map->buf->size <= map->offset) {
        return 0;
    }

    return map->buf->size - map->offset;
}

static inline size_t mglBufferMapAvailableBackingBytes(const BufferMap *map,
                                                       size_t backing_size)
{
    if (!map || map->offset < 0 || (size_t)map->offset >= backing_size) {
        return 0;
    }

    GLsizeiptr storage_remaining = mglBufferMapStorageRemaining(map);
    size_t backing_remaining = backing_size - (size_t)map->offset;
    if (storage_remaining <= 0 || (size_t)storage_remaining > backing_remaining) {
        return storage_remaining > 0 ? backing_remaining : 0;
    }
    return (size_t)storage_remaining;
}

/* OpenGL 4.2+ permits an indexed range to extend beyond the buffer's current
 * data store.  The usable range is resolved when the buffer is consumed;
 * size == 0 is the BindBufferBase sentinel for the whole current store. */
static inline GLsizeiptr mglBufferMapVisibleSize(const BufferMap *map)
{
    GLsizeiptr visible = mglBufferMapStorageRemaining(map);

    if (visible > 0 && map->size > 0 && map->size < visible) {
        visible = map->size;
    }
    return visible;
}

/* Minecraft 1.21.11 binds ChunkSection with glBindBufferRange size=64 while
 * writing the full 96-byte std140 block (TextureSize lives at offset 72).
 * Desktop GL commonly still serves the trailing store; Metal only exposes the
 * bound length.  Extend readonly UBO ranges up to the reflected block size
 * when the underlying buffer still holds those bytes. */
static inline GLsizeiptr mglBufferMapExtendUniformRange(GLsizeiptr bound_size,
                                                        GLsizeiptr buffer_size,
                                                        GLsizeiptr offset,
                                                        size_t reflected_required)
{
    if (bound_size <= 0 || reflected_required == 0 ||
        (size_t)bound_size >= reflected_required ||
        buffer_size <= offset) {
        return bound_size;
    }

    GLsizeiptr remaining = buffer_size - offset;
    GLsizeiptr want = (GLsizeiptr)reflected_required;
    if (remaining < want) {
        want = remaining;
    }
    return want > bound_size ? want : bound_size;
}

static inline size_t mglBufferMapVisibleBackingBytes(const BufferMap *map,
                                                      size_t backing_size)
{
    size_t backed = mglBufferMapAvailableBackingBytes(map, backing_size);
    GLsizeiptr visible = mglBufferMapVisibleSize(map);

    if (visible <= 0) {
        return 0;
    }
    return (size_t)visible < backed ? (size_t)visible : backed;
}

typedef struct BufferMapList_t {
    GLuint      count;
    BufferMap   buffers[MAX_MAPPED_BUFFERS];
} BufferMapList;

#endif /* mgl_types_buffer_h */
