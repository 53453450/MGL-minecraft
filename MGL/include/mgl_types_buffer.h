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
    GLintptr written_min;
    GLintptr written_max; // exclusive byte offset, -1 when unknown/unwritten
    MGLBufferInitSource last_init_source;
    GLintptr last_write_offset;
    GLsizeiptr last_write_size;
    const void *last_write_src_ptr;
    uint64_t last_write_src_hash;
    void *mapped_ptr;
    GLboolean transient_batch_buffer;
    /* P0-4A: reference count for deferred buffer lifetime management.
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

/* P0-4A: Buffer reference counting (mirrors Program refcount pattern).
 * Declared here alongside the Buffer type, matching how Program refcount
 * helpers are declared in mgl_types_program.h. */
void mglRetainBufferReference(Buffer *buf);
void mglReleaseBufferReference(GLMContext ctx, Buffer *buf);

typedef struct BufferBaseTarget_t {
    GLuint      buffer;
    GLsizeiptr  offset;
    GLsizeiptr  size;
    Buffer      *buf;
} BufferBaseTarget;

#define MAX_BINDABLE_BUFFERS    84
#define MGL_MAX_VERTEX_ATTRIB_BINDINGS MAX_VERTEX_BUFFER_BINDINGS
#define MGL_BUFFER_SIZE_BUFFER_INDEX 25u  /* Metal buffer slot for spvBufferSizeConstants */
typedef struct BufferBase_t {
    BufferBaseTarget    buffers[MAX_BINDABLE_BUFFERS];
} BufferBase;

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
