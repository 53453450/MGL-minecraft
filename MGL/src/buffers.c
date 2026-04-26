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
 * buffers.c
 * MGL
 *
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>

#include "glm_context.h"
#include "buffers.h"
#include "pixel_utils.h"
#include "mgl_safety.h"

// Used to recover from a corrupted context pointer (e.g. small non-NULL values like 0x2f)
extern void mgl_lazy_init(void);
extern GLMContext _ctx;

void *mglMapNamedBufferRange(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access);

#ifndef MGL_VERBOSE_BIND_BUFFER_LOGS
#define MGL_VERBOSE_BIND_BUFFER_LOGS 0
#endif

#ifndef MGL_VERBOSE_BUFFER_MAP_LOGS
#define MGL_VERBOSE_BUFFER_MAP_LOGS 0
#endif

static inline bool mglIsTraceBufferTarget(GLenum target)
{
    return (target == GL_ARRAY_BUFFER ||
            target == GL_ELEMENT_ARRAY_BUFFER ||
            target == GL_UNIFORM_BUFFER);
}

static inline bool mglShouldTraceBufferMutation(uint64_t call, GLenum target, GLsizeiptr size)
{
    if (!mglIsTraceBufferTarget(target)) {
        return false;
    }

    if (call <= 128ull) {
        return true;
    }

    if (size > 0 && size <= 256 && (call % 32ull) == 0ull) {
        return true;
    }

    return ((call % 128ull) == 0ull);
}

static uint64_t mglTraceHashBytes(const void *data, size_t len)
{
    if (!data || len == 0) {
        return 0ull;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    size_t head = len < 1024 ? len : 1024;
    uint64_t hash = 1469598103934665603ull;

    for (size_t i = 0; i < head; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ull;
    }

    if (len > head) {
        const uint8_t *tail = bytes + (len - head);
        for (size_t i = 0; i < head; i++) {
            hash ^= (uint64_t)tail[i];
            hash *= 1099511628211ull;
        }
    }

    hash ^= (uint64_t)len;
    hash *= 1099511628211ull;
    return hash;
}

static void mglTraceFormatBytes(const void *data, size_t len, char *out, size_t out_size)
{
    if (!out || out_size == 0) {
        return;
    }

    if (!data || len == 0) {
        snprintf(out, out_size, "-");
        return;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    size_t sample = len < 8 ? len : 8;
    size_t used = 0;

    for (size_t i = 0; i < sample && used + 3 < out_size; i++) {
        int wrote = snprintf(out + used, out_size - used, "%02x", bytes[i]);
        if (wrote <= 0) {
            break;
        }
        used += (size_t)wrote;
        if (i + 1 < sample && used + 2 < out_size) {
            out[used++] = ':';
            out[used] = '\0';
        }
    }

    if (len > sample && used + 4 < out_size) {
        snprintf(out + used, out_size - used, "...");
    }
}

static bool mglTraceSampleAllZero(const void *data, size_t len)
{
    if (!data || len == 0) {
        return true;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        if (bytes[i] != 0u) {
            return false;
        }
    }

    return true;
}

static void mglBufferMarkWrite(Buffer *ptr,
                               MGLBufferInitSource source,
                               GLintptr offset,
                               GLsizeiptr size,
                               const void *src_ptr,
                               uint64_t src_hash)
{
    if (!ptr) {
        return;
    }

    ptr->ever_written = GL_TRUE;
    if (size > 0 &&
        offset >= 0 &&
        offset <= ptr->size &&
        size <= (ptr->size - offset)) {
        GLintptr write_end = offset + size;
        if (ptr->written_min < 0 || offset < ptr->written_min) {
            ptr->written_min = offset;
        }
        if (ptr->written_max < 0 || write_end > ptr->written_max) {
            ptr->written_max = write_end;
        }
        ptr->has_initialized_data = GL_TRUE;
    }

    ptr->last_init_source = source;
    ptr->last_write_offset = offset;
    ptr->last_write_size = size;
    ptr->last_write_src_ptr = src_ptr;
    ptr->last_write_src_hash = src_hash;
}

static void mglBufferMarkAllocatedUninitialized(Buffer *ptr, MGLBufferInitSource source)
{
    if (!ptr) {
        return;
    }

    ptr->has_initialized_data = GL_FALSE;
    ptr->ever_written = GL_FALSE;
    ptr->written_min = -1;
    ptr->written_max = -1;
    ptr->last_init_source = source;
    ptr->last_write_offset = 0;
    ptr->last_write_size = ptr->size;
    ptr->last_write_src_ptr = NULL;
    ptr->last_write_src_hash = 0ull;
}

static inline bool mglBufferMapAllowsWrite(const Buffer *ptr)
{
    if (!ptr) {
        return false;
    }

    if (ptr->access == GL_WRITE_ONLY || ptr->access == GL_READ_WRITE) {
        return true;
    }

    return (ptr->access_flags & GL_MAP_WRITE_BIT) != 0;
}

static void mglBufferMarkMapWrite(Buffer *ptr)
{
    if (!ptr || !mglBufferMapAllowsWrite(ptr)) {
        return;
    }

    GLintptr write_offset = ptr->mapped_offset;
    GLsizeiptr write_size = ptr->mapped_length > 0 ? ptr->mapped_length : ptr->size;
    const uint8_t *src = NULL;
    uint64_t hash = 0ull;

    if (ptr->data.buffer_data && write_size > 0) {
        size_t safe_offset = (write_offset > 0) ? (size_t)write_offset : 0u;
        src = ((const uint8_t *)(uintptr_t)ptr->data.buffer_data) + safe_offset;
        hash = mglTraceHashBytes(src, (size_t)write_size);
    }

    char head[64];
    mglTraceFormatBytes(src, (write_size > 0) ? (size_t)write_size : 0u, head, sizeof(head));
    size_t probe_len = 0u;
    if (src && write_size > 0) {
        size_t ws = (size_t)write_size;
        probe_len = ws < 256u ? ws : 256u;
    }
    bool probe_all_zero = mglTraceSampleAllZero(src, probe_len);
    static uint64_t s_buffer_map_write_trace_count = 0u;
    uint64_t trace_call = ++s_buffer_map_write_trace_count;
    if (MGL_VERBOSE_BUFFER_MAP_LOGS ||
        mglShouldTraceBufferMutation(trace_call, ptr->target, write_size)) {
        fprintf(stderr,
                "MGL TRACE BufferMap.write buffer=%u target=0x%x off=%lld size=%lld src=%p hash=0x%016" PRIx64 " head=%s probe256AllZero=%d access=0x%x accessFlags=0x%x mapped=%u\n",
                ptr->name,
                ptr->target,
                (long long)write_offset,
                (long long)write_size,
                src,
                hash,
                head,
                probe_all_zero ? 1 : 0,
                ptr->access,
                ptr->access_flags,
                (unsigned)ptr->mapped);
    }

    mglBufferMarkWrite(ptr,
                       kInitMapWrite,
                       write_offset,
                       write_size,
                       src,
                       hash);
}

#pragma mark Utility Functions

GLuint bufferIndexFromTarget(GLMContext ctx, GLenum target)
{
    (void)ctx;

    switch(target)
    {
        case GL_ARRAY_BUFFER: return _ARRAY_BUFFER;
        case GL_ELEMENT_ARRAY_BUFFER: return _ELEMENT_ARRAY_BUFFER;
        case GL_UNIFORM_BUFFER: return _UNIFORM_BUFFER;
        case GL_TEXTURE_BUFFER: return _TEXTURE_BUFFER;
        case GL_TRANSFORM_FEEDBACK_BUFFER: return _TRANSFORM_FEEDBACK_BUFFER;
        case GL_QUERY_BUFFER: return _QUERY_BUFFER;
        case GL_PIXEL_PACK_BUFFER: return _PIXEL_PACK_BUFFER;
        case GL_PIXEL_UNPACK_BUFFER: return _PIXEL_UNPACK_BUFFER;
        case GL_ATOMIC_COUNTER_BUFFER: return _ATOMIC_COUNTER_BUFFER;
        case GL_COPY_READ_BUFFER: return _COPY_READ_BUFFER;
        case GL_COPY_WRITE_BUFFER: return _COPY_WRITE_BUFFER;
        case GL_DISPATCH_INDIRECT_BUFFER: return _DISPATCH_INDIRECT_BUFFER;
        case GL_DRAW_INDIRECT_BUFFER: return _DRAW_INDIRECT_BUFFER;
        case GL_SHADER_STORAGE_BUFFER: return _SHADER_STORAGE_BUFFER;

        default:
            fprintf(stderr, "MGL ERROR: bufferIndexFromTarget invalid target=0x%x\n", target);
            return _MAX_BUFFER_TYPES;
    }
}

Buffer *newBuffer(GLMContext ctx, GLenum target, GLuint name)
{
    Buffer *ptr;

    ptr = (Buffer *)malloc(sizeof(Buffer));
    assert(ptr);

    bzero(ptr, sizeof(Buffer));

    ptr->name = name;
    ptr->target = target;
    ptr->written_min = -1;
    ptr->written_max = -1;

    // create buffers doesn't provide a target
    if (target)
    {
        ptr->index = bufferIndexFromTarget(ctx, target);
    }

    return ptr;
}

Buffer *getBuffer(GLMContext ctx, GLenum target, GLuint buffer)
{
    Buffer *ptr;

    if (!ctx || buffer == 0u)
    {
        return NULL;
    }

    ptr = (Buffer *)searchHashTable(&STATE(buffer_table), buffer);

    if (!ptr)
    {
        ptr = newBuffer(ctx, target, buffer);
        if (!ptr)
        {
            return NULL;
        }

        insertHashElement(&STATE(buffer_table), buffer, ptr);
    }

    return ptr;
}

bool isBuffer(GLMContext ctx, GLuint buffer)
{
    if (!ctx || buffer == 0u)
    {
        return false;
    }

    return ((Buffer *)searchHashTable(&STATE(buffer_table), buffer)) != NULL;
}

Buffer *findBuffer(GLMContext ctx, GLuint buffer)
{
    if (!ctx || buffer == 0u)
    {
        return NULL;
    }

    return (Buffer *)searchHashTable(&STATE(buffer_table), buffer);
}

bool checkTarget(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_ARRAY_BUFFER:
        case GL_ELEMENT_ARRAY_BUFFER:
        case GL_UNIFORM_BUFFER:
        case GL_TEXTURE_BUFFER:
        case GL_TRANSFORM_FEEDBACK_BUFFER:
        case GL_QUERY_BUFFER:
        case GL_PIXEL_PACK_BUFFER:
        case GL_PIXEL_UNPACK_BUFFER:
        case GL_ATOMIC_COUNTER_BUFFER:
        case GL_COPY_READ_BUFFER:
        case GL_COPY_WRITE_BUFFER:
        case GL_DISPATCH_INDIRECT_BUFFER:
        case GL_DRAW_INDIRECT_BUFFER:
        case GL_SHADER_STORAGE_BUFFER:
            return true;
    }

    return false;
}

bool checkUsage(GLMContext ctx, GLenum usage)
{
    switch(usage)
    {
        case GL_STREAM_DRAW:
        case GL_STREAM_READ:
        case GL_STREAM_COPY:
        case GL_STATIC_DRAW:
        case GL_STATIC_READ:
        case GL_STATIC_COPY:
        case GL_DYNAMIC_DRAW:
        case GL_DYNAMIC_READ:
        case GL_DYNAMIC_COPY:
            return true;
    }

    return false;
}

static void mglBindNullBufferForTarget(GLMContext ctx, GLenum target, GLint index)
{
    if (target == GL_ELEMENT_ARRAY_BUFFER)
    {
        VertexArray *bound_vao = ctx->state.vao;
        int vao_is_valid = 0;

        if (bound_vao && STATE(vao_table).keys && STATE(vao_table).size > 0)
        {
            for (size_t i = 0; i < STATE(vao_table).size; i++)
            {
                if (STATE(vao_table).states && STATE(vao_table).states[i] != 1u)
                    continue;
                if (STATE(vao_table).keys[i].data == bound_vao)
                {
                    vao_is_valid = 1;
                    break;
                }
            }
        }

        if (bound_vao && !vao_is_valid)
        {
            fprintf(stderr, "MGL WARNING: dropping invalid current VAO pointer %p during EBO unbind\n",
                    (void *)bound_vao);
            ctx->state.vao = NULL;
            bound_vao = NULL;
        }

        if (bound_vao)
        {
            bound_vao->element_array.buffer = NULL;
            bound_vao->dirty_bits |= DIRTY_VAO_BUFFER_BASE;
            STATE(dirty_bits) |= DIRTY_VAO;
        }
        else
        {
            // Compatibility path: emulate VAO 0 element binding behavior.
            STATE(default_vao_element_array_buffer) = NULL;
        }

        STATE(buffers[index]) = NULL;
        STATE_VAR(element_array_buffer_binding) = 0;
        return;
    }

    if (STATE(buffers[index]) != NULL)
    {
        STATE(buffers[index]) = NULL;
        STATE(dirty_bits) |= DIRTY_BUFFER;
    }
}

static inline bool mgl_range_ok_glsize(GLintptr offset, GLsizeiptr size, GLsizeiptr total)
{
    if (offset < 0 || size < 0 || total < 0)
        return false;

    uint64_t uoffset = (uint64_t)offset;
    uint64_t usize = (uint64_t)size;
    uint64_t utotal = (uint64_t)total;

    if (uoffset > utotal)
        return false;
    if (usize > utotal - uoffset)
        return false;

    return true;
}

static inline bool mgl_range_ok_size_t(GLintptr offset, GLsizeiptr size, size_t total)
{
    if (offset < 0 || size < 0)
        return false;

    uint64_t uoffset = (uint64_t)offset;
    uint64_t usize = (uint64_t)size;
    uint64_t utotal = (uint64_t)total;

    if (uoffset > utotal)
        return false;
    if (usize > utotal - uoffset)
        return false;

    return true;
}

static inline GLMContext mgl_sanitize_ctx(GLMContext ctx, const char *func)
{
    // A valid heap pointer will never be a tiny value like 0x2f.
    if (ctx != NULL && (uintptr_t)ctx >= 0x10000u)
        return ctx;

    fprintf(stderr, "MGL ERROR: %s received invalid ctx=%p; attempting to recover\n", func, (void *)ctx);

    mgl_lazy_init();

    if (_ctx != NULL && (uintptr_t)_ctx >= 0x10000u)
        return _ctx;

    fprintf(stderr, "MGL ERROR: %s recovery failed; dropping call\n", func);
    return NULL;
}

static VertexArray *mglGetSafeCurrentVAO(GLMContext ctx)
{
    VertexArray *vao;

    if (!ctx)
        return NULL;

    vao = ctx->state.vao;
    if (!vao)
        return NULL;

    if (!STATE(vao_table).keys || STATE(vao_table).size == 0)
    {
        fprintf(stderr, "MGL WARNING: VAO table unavailable, dropping current VAO pointer %p\n", (void *)vao);
        ctx->state.vao = NULL;
        return NULL;
    }

    for (size_t i = 0; i < STATE(vao_table).size; i++)
    {
        if (STATE(vao_table).states && STATE(vao_table).states[i] != 1u)
            continue;
        if (STATE(vao_table).keys[i].data == vao)
        {
            return vao;
        }
    }

    fprintf(stderr, "MGL WARNING: current VAO pointer %p is not in VAO table; resetting to VAO 0\n", (void *)vao);
    ctx->state.vao = NULL;
    return NULL;
}

static Buffer *mglGetBoundBufferForTarget(GLMContext ctx, GLenum target)
{
    GLuint index;

    if (!ctx || !checkTarget(ctx, target))
        return NULL;

    index = bufferIndexFromTarget(ctx, target);

    if (target == GL_ELEMENT_ARRAY_BUFFER)
    {
        VertexArray *vao = mglGetSafeCurrentVAO(ctx);
        if (vao)
            return vao->element_array.buffer;
        return STATE(default_vao_element_array_buffer);
    }

    if (index >= _MAX_BUFFER_TYPES)
        return NULL;

    return STATE(buffers[index]);
}

size_t page_size_align(size_t size)
{
    if (size & (4096-1))
    {
        size_t pad_size = 0;

        pad_size = 4096 - (size & (4096-1));

        size += pad_size;
    }

    return size;
}

void *getBufferData(GLMContext ctx, Buffer *ptr)
{
    void *buffer_data;

    if (!ptr) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    ERROR_CHECK_RETURN(ptr->mapped == false, GL_INVALID_OPERATION);

    buffer_data = (void *)(uintptr_t)ptr->data.buffer_data;

    ERROR_CHECK_RETURN(buffer_data, GL_INVALID_OPERATION);

    return buffer_data;
}

void bufferStorage(GLMContext ctx, Buffer *ptr, GLenum target, GLuint index, GLsizeiptr size, const void *data, GLbitfield storage_flags, GLenum usage)
{
    kern_return_t err;
    vm_address_t buffer_data;
    size_t buffer_size;

    buffer_size = page_size_align(size);

    // Allocate directly from VM
    err = vm_allocate((vm_map_t) mach_task_self(),
                      (vm_address_t*) &buffer_data,
                      buffer_size,
                      VM_FLAGS_ANYWHERE);
    if (err)
    {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
    }

    ptr->data.buffer_data = buffer_data;

    // copy to new buffer
    if (data)
    {
        memcpy((void *)ptr->data.buffer_data, data, size);

        ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
    }

    // init
    ptr->data.buffer_data = buffer_data;
    ptr->data.buffer_size = buffer_size;
    ptr->index = index;
    ptr->target = target;
    ptr->size = size;
    if (usage == 0)
    {
        ptr->immutable_storage = BUFFER_IMMUTABLE_STORAGE_FLAG;
        ptr->usage = usage;
    }
    else
    {
        ptr->immutable_storage = 0;
        ptr->usage = usage;
    }

    ptr->mapped = GL_FALSE;
    ptr->access = 0;
    ptr->access_flags = 0;
    ptr->storage_flags = storage_flags;

    ptr->data.dirty_bits |= DIRTY_BUFFER_ADDR;

    if (data)
    {
        if (storage_flags & (GL_CLIENT_STORAGE_BIT | GL_DYNAMIC_STORAGE_BIT))
        {
            memcpy((void *)ptr->data.buffer_data, data, ptr->size);
        }
        else
        {
            ctx->mtl_funcs.mtlBufferSubData(ctx, ptr, 0, ptr->size, data);
        }

        mglBufferMarkWrite(ptr,
                           kInitBufferDataCopy,
                           0,
                           ptr->size,
                           data,
                           mglTraceHashBytes(data, (size_t)ptr->size));
    }
    else
    {
        mglBufferMarkAllocatedUninitialized(ptr, kInitBufferDataNull);
    }
}

bool clearBufferData(GLMContext ctx, Buffer *ptr, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    switch(format)
    {
        case GL_R8:
        case GL_R16:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R16I:
        case GL_R32I:
        case GL_R8UI:
        case GL_R16UI:
        case GL_R32UI:
        case GL_RG8:
        case GL_RG16:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RG8I:
        case GL_RG16I:
        case GL_RG32I:
        case GL_RG8UI:
        case GL_RG16UI:
        case GL_RG32UI:
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGBA8:
        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_RGBA8I:
        case GL_RGBA16I:
        case GL_RGBA32I:
        case GL_RGBA8UI:
        case GL_RGBA16UI:
        case GL_RGBA32UI:
            break;

        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }

    if (!ptr) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (!mgl_range_ok_glsize(offset, size, ptr->size))
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (size == 0) {
        return true;
    }

    if (!data) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    size_t pixel_size = sizeForInternalFormat(internalformat, format, type);
    if (!pixel_size) {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }

    if (((size_t)size % pixel_size) != 0u) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    GLubyte *base = (GLubyte *)getBufferData(ctx, ptr);
    if (!base) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    GLubyte *dst = base + (size_t)offset;
    size_t pixel_count = (size_t)size / pixel_size;

    for(size_t i=0; i<pixel_count; i++)
    {
        memcpy(dst + (i * pixel_size), data, pixel_size);
    }

    ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
    ctx->state.dirty_bits |= DIRTY_BUFFER;
    mglBufferMarkWrite(ptr,
                       kInitBufferSubData,
                       offset,
                       size,
                       data,
                       mglTraceHashBytes(data, pixel_size));

    return true;
}


#pragma mark GL Buffer Functions
void mglGenBuffers(GLMContext ctx, GLsizei n, GLuint *buffers)
{
    static uint64_t s_gen_buffers_calls = 0u;
    uint64_t call_id = ++s_gen_buffers_calls;

    if (!ctx || n <= 0 || !buffers) {
        fprintf(stderr,
                "MGL TRACE GenBuffers.skip call=%llu ctx=%p n=%d buffers=%p\n",
                (unsigned long long)call_id,
                (void *)ctx,
                (int)n,
                (void *)buffers);
        return;
    }

    while(n--)
    {
        GLuint name = getNewName(&STATE(buffer_table));
        *buffers++ = name;
        fprintf(stderr,
                "MGL TRACE GenBuffers call=%llu generated=%u currentName=%u tableCount=%zu tableCap=%zu\n",
                (unsigned long long)call_id,
                name,
                STATE(buffer_table).current_name,
                STATE(buffer_table).count,
                STATE(buffer_table).size);
    }
}

void mglCreateBuffers(GLMContext ctx, GLsizei n, GLuint *buffers)
{
    GLuint name;

    while(n--)
    {
        name = getNewName(&STATE(buffer_table));

        // create an unbound buffer
        getBuffer(ctx, 0, name);

        *buffers++ = name;
    }
}

void mglDeleteBuffers(GLMContext ctx, GLsizei n, const GLuint *buffers)
{
    GLuint buffer;

    ctx = mgl_sanitize_ctx(ctx, __FUNCTION__);
    if (!ctx)
        return;

    if (n < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n > 0 && buffers == NULL)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    while(n--)
    {
        buffer = *buffers++;

        // OpenGL: deleting name 0 is silently ignored.
        if (buffer == 0)
            continue;

        if (isBuffer(ctx, buffer))
        {
            Buffer *ptr;

            ptr = findBuffer(ctx, buffer);
            if (!ptr)
                continue;

            if (ptr->data.buffer_data)
            {
                if (ptr->storage_flags & GL_CLIENT_STORAGE_BIT)
                {
                    if (ptr->data.mtl_data)
                    {
                        // the mtl buffer has a deallocator for the vm allocate
                        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->data.mtl_data);
                        ptr->data.mtl_data = NULL;
                    }
                    else
                    {
                        vm_deallocate(mach_host_self(), ptr->data.buffer_data, ptr->data.buffer_size);
                    }
                }
                else
                {
                    if (ptr->data.mtl_data)
                    {
                        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->data.mtl_data);
                        ptr->data.mtl_data = NULL;
                    }
                }

                ptr->data.buffer_data = 0;
            }

            // remove any dangling references from generic buffer bindings
            for (GLuint i = 0; i < MAX_BINDABLE_BUFFERS; i++)
            {
                if (STATE(buffers[i]) == ptr ||
                    (STATE(buffers[i]) && STATE(buffers[i])->name == buffer))
                {
                    STATE(buffers[i]) = NULL;
                }
            }

            // remove dangling VAO references (attribute buffers + element buffer)
            if (VAO())
            {
                if (VAO_STATE(element_array.buffer) == ptr ||
                    (VAO_STATE(element_array.buffer) && VAO_STATE(element_array.buffer)->name == buffer))
                {
                    VAO_STATE(element_array.buffer) = NULL;
                }

                for (GLuint i = 0; i < MAX_ATTRIBS; i++)
                {
                    if (VAO_ATTRIB_STATE(i).buffer == ptr ||
                        (VAO_ATTRIB_STATE(i).buffer && VAO_ATTRIB_STATE(i).buffer->name == buffer))
                    {
                        VAO_ATTRIB_STATE(i).buffer = NULL;
                    }
                }
            }
            if (STATE(default_vao_element_array_buffer) == ptr ||
                (STATE(default_vao_element_array_buffer) && STATE(default_vao_element_array_buffer)->name == buffer))
            {
                STATE(default_vao_element_array_buffer) = NULL;
            }

            // remove any dangling references in indexed buffer-base bindings
            for (GLuint idx = 0; idx < _MAX_BUFFER_TYPES; idx++)
            {
                for (GLuint i = 0; i < MAX_BINDABLE_BUFFERS; i++)
                {
                    if (ctx->state.buffer_base[idx].buffers[i].buf == ptr ||
                        ctx->state.buffer_base[idx].buffers[i].buffer == buffer)
                    {
                        bzero(&ctx->state.buffer_base[idx].buffers[i], sizeof(BufferBaseTarget));
                    }
                }
            }

            deleteHashElement(&STATE(buffer_table), buffer);
            free(ptr);
        } // if (isBuffer(ctx, buffer))
    } // while(--n)
}

GLboolean mglIsBuffer(GLMContext ctx, GLuint buffer)
{
    if (isBuffer(ctx, buffer))
    {
        return GL_TRUE;
    }

    return GL_FALSE;
}

void mglBindBuffer(GLMContext ctx, GLenum target, GLuint buffer)
{
    GLint index;
    Buffer *ptr;

    if (!ctx)
        return;

    if (MGL_VERBOSE_BIND_BUFFER_LOGS) {
        fprintf(stderr, "MGL TRACE BindBuffer target=0x%x buffer=%u ctx=%p vao=%p\n",
                target, buffer, (void *)ctx, ctx ? (void *)ctx->state.vao : NULL);
    }

    // GL_INVALID_ENUM is generated if target is not supported.
    if (!checkTarget(ctx, target))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (buffer == 0)
    {
        index = bufferIndexFromTarget(ctx, target);
        if (index < 0 || index >= _MAX_BUFFER_TYPES)
        {
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
        }
        mglBindNullBufferForTarget(ctx, target, index);
        return;
    }

    if (buffer)
    {
        ptr = getBuffer(ctx, target, buffer);
        if (!ptr)
        {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
    }
    else
    {
        ptr = NULL;
    }

    index = bufferIndexFromTarget(ctx, target);
    if (index < 0 || index >= _MAX_BUFFER_TYPES)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    // GL_ELEMENT_ARRAY_BUFFER binding is part of VAO state, not global state.
    // Keep VAO + compatibility state in sync so indexed draws can find EBO reliably.
    if (target == GL_ELEMENT_ARRAY_BUFFER)
    {
        VertexArray *vao = mglGetSafeCurrentVAO(ctx);
        if (vao)
        {
            if (vao->element_array.buffer != ptr)
            {
                vao->element_array.buffer = ptr;
                vao->dirty_bits |= DIRTY_VAO_BUFFER_BASE;
                STATE(dirty_bits) |= DIRTY_VAO;
            }
        }
        else
        {
            // Compatibility path: treat VAO 0 as a default VAO bucket.
            STATE(default_vao_element_array_buffer) = ptr;
        }

        STATE(buffers[index]) = ptr;
        STATE_VAR(element_array_buffer_binding) = ptr ? ptr->name : 0;
        return;
    }

    if (STATE(buffers[index]) != ptr)
    {
        STATE(buffers[index]) = ptr;
        STATE(dirty_bits) |= DIRTY_BUFFER;
    }
}

void mglBindBufferBase(GLMContext ctx, GLenum target, GLuint index, GLuint buffer)
{
    Buffer  *ptr;
    GLuint buffer_index;

    switch(target)
    {
        case GL_UNIFORM_BUFFER:
        case GL_TRANSFORM_FEEDBACK_BUFFER:
        case GL_SHADER_STORAGE_BUFFER:
        case GL_ATOMIC_COUNTER_BUFFER:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(index >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(index < MAX_BINDABLE_BUFFERS, GL_INVALID_VALUE);

    buffer_index = bufferIndexFromTarget(ctx, target);

    if (buffer)
    {
        ERROR_CHECK_RETURN(isBuffer(ctx, buffer), GL_INVALID_VALUE);
        ptr = getBuffer(ctx, target, buffer);
        assert(ptr);

        // Bind-by-name semantics: bind succeeds even when storage is not yet initialized.
        // Storage/range validity is validated later at draw/upload time.
        ERROR_CHECK_RETURN(ptr->size >= 0, GL_INVALID_VALUE);

        ctx->state.buffer_base[buffer_index].buffers[index].buffer = buffer;
        ctx->state.buffer_base[buffer_index].buffers[index].offset = 0;
        ctx->state.buffer_base[buffer_index].buffers[index].size = ptr->size;
        ctx->state.buffer_base[buffer_index].buffers[index].buf = ptr;

        ptr->target = target;
    }
    else
    {
        bzero(&ctx->state.buffer_base[buffer_index].buffers[index], sizeof(BufferBaseTarget));
    }

    ctx->state.dirty_bits |= (DIRTY_BUFFER | DIRTY_BUFFER_BASE_STATE);
}


void mglBindBuffersBase(GLMContext ctx, GLenum target, GLuint first, GLsizei count, const GLuint *buffers)
{
    while(count--)
    {
        GLuint name = buffers ? *buffers++ : 0u;
        mglBindBufferBase(ctx, target, first++, name);
    }
}

void mglBindBufferRange(GLMContext ctx, GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    Buffer  *ptr;
    GLuint  buffer_index;

    switch(target)
    {
        case GL_UNIFORM_BUFFER:
        case GL_TRANSFORM_FEEDBACK_BUFFER:
        case GL_SHADER_STORAGE_BUFFER:
        case GL_ATOMIC_COUNTER_BUFFER:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(index >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(index < MAX_BINDABLE_BUFFERS, GL_INVALID_VALUE);

    // ERROR_CHECK_RETURN(offset >= 0, GL_INVALID_VALUE);
    if (offset < 0) {
        fprintf(stderr, "MGL Error: mglBindBufferRange: offset < 0 (%ld)\n", offset);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(size > 0, GL_INVALID_VALUE);
    if (size <= 0) {
        fprintf(stderr, "MGL Error: mglBindBufferRange: size <= 0 (%ld)\n", size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    buffer_index = bufferIndexFromTarget(ctx, target);

    if (buffer)
    {
        ERROR_CHECK_RETURN(isBuffer(ctx, buffer), GL_INVALID_VALUE);
        ptr = getBuffer(ctx, target, buffer);
        assert(ptr);

        // GL allows binding ranges before upload; validate against logical size.
        if (!mgl_range_ok_size_t(offset, size, ptr->size)) {
            fprintf(stderr, "MGL Error: mglBindBufferRange: range overflow (offset=%ld size=%ld buffer_size=%ld)\n",
                    offset, size, (long)ptr->size);
            ERROR_RETURN(GL_INVALID_VALUE);
        }

        ctx->state.buffer_base[buffer_index].buffers[index].buffer = buffer;
        ctx->state.buffer_base[buffer_index].buffers[index].offset = offset;
        ctx->state.buffer_base[buffer_index].buffers[index].size = size;
        ctx->state.buffer_base[buffer_index].buffers[index].buf = ptr;

        ptr->target = target;
    }
    else
    {
        bzero(&ctx->state.buffer_base[buffer_index].buffers[index], sizeof(BufferBaseTarget));
    }
}

#pragma mark GL Buffer Data Functions
kern_return_t initBufferData(GLMContext ctx, Buffer *ptr, GLsizeiptr size, const void *data, bool isUniformConstant)
{
    kern_return_t err;
    vm_address_t buffer_data;
    size_t buffer_size;

    if (ptr->data.buffer_data)
    {
        if (isUniformConstant)
        {
            // if its a uniform constant then lets not delete and create a buffer
            // check the old size.. then if it can fit new size then reuse.
            if (size <= ptr->data.buffer_size)
            {
                ptr->size = size;
                if (data)
                {
                    memcpy((void *)ptr->data.buffer_data, data, size);
                    
                    ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
                    mglBufferMarkWrite(ptr,
                                       kInitBufferDataCopy,
                                       0,
                                       size,
                                       data,
                                       mglTraceHashBytes(data, (size_t)size));
                }
                else
                {
                    mglBufferMarkAllocatedUninitialized(ptr, kInitBufferDataNull);
                }
                
                return 0;
            }
        }

        if (ptr->storage_flags & GL_CLIENT_STORAGE_BIT)
        {
            if (ptr->data.mtl_data)
            {
                // the mtl buffer has a deallocator for the vm allocate
                ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->data.mtl_data);
                
                if(isUniformConstant)
                {
                    ptr->data.mtl_data = NULL;
                }
            }
            else
            {
                vm_deallocate(mach_host_self(), ptr->data.buffer_data, ptr->data.buffer_size);
            }
            
            ptr->data.buffer_data = 0;
            ptr->data.buffer_size = 0;
        }
        else
        {
            if (ptr->data.mtl_data)
            {
                ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->data.mtl_data);
            }
            
            if(isUniformConstant)
            {
                ptr->data.mtl_data = NULL;
            }
            
            ptr->data.buffer_data = 0;
            ptr->data.buffer_size = 0;
        }
    }

    buffer_size = page_size_align(size);

    // Allocate directly from VM
    err = vm_allocate((vm_map_t) mach_task_self(),
                      (vm_address_t*) &buffer_data,
                      buffer_size,
                      VM_FLAGS_ANYWHERE);
    if (err)
    {
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, err);
    }

    ptr->size = size;
    ptr->data.buffer_data = buffer_data;
    ptr->data.buffer_size = buffer_size;

    ptr->data.dirty_bits |= DIRTY_BUFFER_ADDR;

    // copy to new buffer
    if (data)
    {
        memcpy((void *)ptr->data.buffer_data, data, size);

        ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
        mglBufferMarkWrite(ptr,
                           kInitBufferDataCopy,
                           0,
                           size,
                           data,
                           mglTraceHashBytes(data, (size_t)size));
    }
    else
    {
        mglBufferMarkAllocatedUninitialized(ptr, kInitBufferDataNull);
    }

    return err;
}

void mglBufferData(GLMContext ctx, GLenum target, GLsizeiptr size, const void *data, GLenum usage)
{
    static uint64_t s_bufferDataCalls = 0;
    uint64_t call = ++s_bufferDataCalls;
    GLuint index;
    Buffer *ptr;

    // GL_INVALID_ENUM is generated if target is not supported.
    if (!checkTarget(ctx, target))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    // GL_INVALID_ENUM is generated if target is not one of the allowable values.
    if (!checkUsage(ctx, usage))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (size < 0)
    {
        fprintf(stderr, "MGL Error: mglBufferData: size < 0 (%ld)\n", size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    index = bufferIndexFromTarget(ctx, target);
    if (index >= _MAX_BUFFER_TYPES)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    ptr = STATE(buffers[index]);
    if (ptr == NULL)
    {
        fprintf(stderr,
                "MGL ERROR: BufferData with no bound buffer call=%" PRIu64 " target=0x%x size=%lld usage=0x%x\n",
                call,
                target,
                (long long)size,
                usage);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    bool trace = mglShouldTraceBufferMutation(call, target, size);
    char src_preview[64];
    src_preview[0] = '\0';
    uint64_t src_hash = 0ull;
    if (trace && data && size > 0) {
        mglTraceFormatBytes(data, (size_t)size, src_preview, sizeof(src_preview));
        src_hash = mglTraceHashBytes(data, (size_t)size);
    }
    if (trace) {
        fprintf(stderr,
                "MGL TRACE BufferData.begin call=%" PRIu64 " target=0x%x buffer=%u size=%lld usage=0x%x data=%p srcHash=0x%016" PRIx64 " srcHead=%s oldSize=%lld dirty=0x%x\n",
                call,
                target,
                ptr->name,
                (long long)size,
                usage,
                data,
                src_hash,
                (data && size > 0) ? src_preview : "-",
                (long long)ptr->size,
                ptr->data.dirty_bits);
    }

    // buffer was created via buffer storage call for immutable storage
    if (ptr->immutable_storage)
    {
        // ERROR_RETURN(GL_INVALID_OPERATION);
        // return;
        // Workaround: Allow re-allocation even if immutable, to support guests that violate spec
        // fprintf(stderr, "MGL WARNING: glBufferData called on immutable buffer %d\n", ptr->name);
    }

    initBufferData(ctx, ptr, size, data, false);

    // init fields local to buffer data
    ptr->index = index;
    ptr->target = target;
    ptr->usage = usage;
    ptr->storage_flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT;

    if (trace) {
        const void *dst_data = (const void *)(uintptr_t)ptr->data.buffer_data;
        char dst_preview[64];
        dst_preview[0] = '\0';
        uint64_t dst_hash = 0ull;
        if (dst_data && size > 0) {
            mglTraceFormatBytes(dst_data, (size_t)size, dst_preview, sizeof(dst_preview));
            dst_hash = mglTraceHashBytes(dst_data, (size_t)size);
        }

        fprintf(stderr,
                "MGL TRACE BufferData.end call=%" PRIu64 " target=0x%x buffer=%u size=%lld vmSize=%llu dataPtr=%p dirty=0x%x dstHash=0x%016" PRIx64 " dstHead=%s\n",
                call,
                target,
                ptr->name,
                (long long)ptr->size,
                (unsigned long long)ptr->data.buffer_size,
                (void *)(uintptr_t)ptr->data.buffer_data,
                ptr->data.dirty_bits,
                dst_hash,
                (dst_data && size > 0) ? dst_preview : "-");
    }
 }

void mglNamedBufferData(GLMContext ctx, GLuint buffer, GLsizeiptr size, const void *data, GLenum usage)
{
    Buffer *ptr;

    ptr = findBuffer(ctx, buffer);

    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (size < 0)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    // GL_INVALID_ENUM is generated if target is not one of the allowable values.
    if (!checkUsage(ctx, usage))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }


    // buffer was created via buffer storage call for immutable storage
    if (ptr->storage_flags)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    initBufferData(ctx, ptr, size, data,false);

    // init fields local to buffer data
    ptr->usage = usage;
    ptr->storage_flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT;
}

void mglBufferSubData(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr size, const void *data)
{
    static uint64_t s_bufferSubDataCalls = 0;
    uint64_t call = ++s_bufferSubDataCalls;

    // Absolute first check - validate ctx before ANY access
    ctx = mgl_sanitize_ctx(ctx, __FUNCTION__);
    if (!ctx)
        return;
    
    GLuint index;
    Buffer * volatile ptr;  // volatile to prevent compiler optimizations
    uint64_t src_hash_for_meta = 0ull;

    // GL_INVALID_ENUM is generated if target is not supported.
    if (!checkTarget(ctx, target)) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_ENUM);
        return;
    }

    // Early exit for zero-size operations
    if (size == 0)
    {
        return;
    }

    // Validate data pointer for non-zero size
    if (data == NULL)
    {
        fprintf(stderr, "MGL Error: mglBufferSubData: data is NULL (target=0x%x off=%ld size=%ld)\n",
                target, (long)offset, (long)size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    src_hash_for_meta = mglTraceHashBytes(data, (size_t)size);

    if (offset < 0 || size < 0)
    {
        fprintf(stderr, "MGL Error: mglBufferSubData: offset (%ld) or size (%ld) < 0\n", offset, size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    index = bufferIndexFromTarget(ctx, target);
    if (index >= _MAX_BUFFER_TYPES) {
        fprintf(stderr, "MGL Error: mglBufferSubData: invalid target index %d\n", index);
        ERROR_RETURN(GL_INVALID_ENUM);
    }
    
    ptr = ctx->state.buffers[index];

    // COMPREHENSIVE BUFFER SAFETY: Validate buffer pointer and get size safely (void function)
    GLsizeiptr buffer_size;
    MGL_GET_BUFFER_SIZE_SAFE_VOID(ptr, buffer_size, "mglBufferSubData");

    if (!mgl_range_ok_glsize(offset, size, buffer_size)) {
        fprintf(stderr, "MGL Error: mglBufferSubData out of bounds: offset %ld size %ld buffer size %ld\n", offset, size, buffer_size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (ptr->mapped && !(ptr->access & GL_MAP_PERSISTENT_BIT))
    {
        fprintf(stderr, "MGL Error: mglBufferSubData: buffer is mapped\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    // GL_INVALID_OPERATION for immutable storage without dynamic bit
    if ((ptr->immutable_storage & BUFFER_IMMUTABLE_STORAGE_FLAG) &&
        !(ptr->storage_flags & GL_DYNAMIC_STORAGE_BIT))
    {
        fprintf(stderr, "MGL Error: mglBufferSubData: immutable storage without dynamic bit\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (ptr->storage_flags & (GL_CLIENT_STORAGE_BIT | GL_DYNAMIC_STORAGE_BIT))
    {
        bool trace = mglShouldTraceBufferMutation(call, target, size);
        uint64_t src_hash = 0ull;
        uint64_t dst_before_hash = 0ull;
        char src_preview[64];
        char dst_before_preview[64];
        src_preview[0] = '\0';
        dst_before_preview[0] = '\0';

        if (trace) {
            src_hash = src_hash_for_meta;
            mglTraceFormatBytes(data, (size_t)size, src_preview, sizeof(src_preview));
        }

        // CRITICAL SECURITY FIX: Proper NULL pointer check with correct type handling
        // buffer_data is vm_address_t (unsigned long), not void*
        if (ptr->data.buffer_data == 0)  // Compare to 0 for vm_address_t
        {
            fprintf(stderr, "MGL Error: mglBufferSubData: buffer_data is NULL (buffer %u, target 0x%x)\n", ptr->name, target);
            ERROR_RETURN(GL_INVALID_OPERATION);
        }

        if (!mgl_range_ok_size_t(offset, size, ptr->data.buffer_size))
        {
            fprintf(stderr, "MGL Error: mglBufferSubData out of backing store bounds: offset %ld size %ld backing %ld\n",
                    offset, size, (long)ptr->data.buffer_size);
            ERROR_RETURN(GL_INVALID_VALUE);
        }

        if (trace) {
            const void *dst_before = (const void *)((uintptr_t)ptr->data.buffer_data + (uintptr_t)offset);
            dst_before_hash = mglTraceHashBytes(dst_before, (size_t)size);
            mglTraceFormatBytes(dst_before, (size_t)size, dst_before_preview, sizeof(dst_before_preview));
            fprintf(stderr,
                    "MGL TRACE BufferSubData.begin call=%" PRIu64 " target=0x%x buffer=%u off=%lld size=%lld srcHash=0x%016" PRIx64 " srcHead=%s dstBeforeHash=0x%016" PRIx64 " dstBeforeHead=%s dirty=0x%x\n",
                    call,
                    target,
                    ptr->name,
                    (long long)offset,
                    (long long)size,
                    src_hash,
                    src_preview,
                    dst_before_hash,
                    dst_before_preview,
                    ptr->data.dirty_bits);
        }
        
        memcpy((char*)ptr->data.buffer_data + offset, data, size);
        ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
        ctx->state.dirty_bits |= DIRTY_BUFFER;
        mglBufferMarkWrite(ptr,
                           kInitBufferSubData,
                           offset,
                           size,
                           data,
                           src_hash_for_meta);

        if (trace) {
            const void *dst_after = (const void *)((uintptr_t)ptr->data.buffer_data + (uintptr_t)offset);
            uint64_t dst_after_hash = mglTraceHashBytes(dst_after, (size_t)size);
            char dst_after_preview[64];
            dst_after_preview[0] = '\0';
            mglTraceFormatBytes(dst_after, (size_t)size, dst_after_preview, sizeof(dst_after_preview));

            fprintf(stderr,
                    "MGL TRACE BufferSubData.end call=%" PRIu64 " target=0x%x buffer=%u off=%lld size=%lld dstAfterHash=0x%016" PRIx64 " dstAfterHead=%s dirty=0x%x stateDirty=0x%x\n",
                    call,
                    target,
                    ptr->name,
                    (long long)offset,
                    (long long)size,
                    dst_after_hash,
                    dst_after_preview,
                    ptr->data.dirty_bits,
                    ctx->state.dirty_bits);
        }
    }
    else
    {
        if (mglShouldTraceBufferMutation(call, target, size)) {
            char src_preview[64];
            src_preview[0] = '\0';
            uint64_t src_hash = src_hash_for_meta;
            mglTraceFormatBytes(data, (size_t)size, src_preview, sizeof(src_preview));
            fprintf(stderr,
                    "MGL TRACE BufferSubData.mtl call=%" PRIu64 " target=0x%x buffer=%u off=%lld size=%lld srcHash=0x%016" PRIx64 " srcHead=%s mtl=%p\n",
                    call,
                    target,
                    ptr->name,
                    (long long)offset,
                    (long long)size,
                    src_hash,
                    src_preview,
                    ptr->data.mtl_data);
        }
        if (ctx->mtl_funcs.mtlBufferSubData)
        {
            ctx->mtl_funcs.mtlBufferSubData(ctx, ptr, offset, size, data);
        }

        mglBufferMarkWrite(ptr,
                           kInitBufferSubData,
                           offset,
                           size,
                           data,
                           src_hash_for_meta);
    }
}

void mglNamedBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data)
{
    Buffer *ptr;
    uint64_t src_hash_for_meta = 0ull;

    ptr = findBuffer(ctx, buffer);

    ERROR_CHECK_RETURN((ptr != NULL), GL_INVALID_OPERATION);

    if (offset < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (size < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (size > 0 && data == NULL)
    {
        fprintf(stderr, "MGL Error: mglNamedBufferSubData: data is NULL (buffer=%u off=%ld size=%ld)\n",
                buffer, (long)offset, (long)size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (!mgl_range_ok_glsize(offset, size, ptr->size))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (ptr->mapped && !(ptr->access & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    // GL_INVALID_OPERATION is generated if the value of the GL_BUFFER_IMMUTABLE_STORAGE flag of the buffer object is GL_TRUE and the value of GL_BUFFER_STORAGE_FLAGS for the buffer object does not have the GL_DYNAMIC_STORAGE_BIT bit set.
    if ((ptr->immutable_storage & BUFFER_IMMUTABLE_STORAGE_FLAG) &&
        !(ptr->storage_flags & GL_DYNAMIC_STORAGE_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    src_hash_for_meta = (size > 0 && data) ? mglTraceHashBytes(data, (size_t)size) : 0ull;

    if (ptr->storage_flags & (GL_CLIENT_STORAGE_BIT | GL_DYNAMIC_STORAGE_BIT))
    {
        // CRITICAL SECURITY FIX: Proper NULL pointer validation for vm_address_t
        // buffer_data is vm_address_t (unsigned long), not void*
        if (ptr->data.buffer_data == 0)
        {
            fprintf(stderr, "MGL WARNING: mglNamedBufferSubData - buffer %u has NULL buffer_data\n", ptr->name);
            ERROR_RETURN(GL_INVALID_OPERATION);
        }

        if (!mgl_range_ok_size_t(offset, size, ptr->data.buffer_size))
        {
            fprintf(stderr, "MGL Error: mglNamedBufferSubData out of backing store bounds: offset %ld size %ld backing %ld\n",
                    offset, size, (long)ptr->data.buffer_size);
            ERROR_RETURN(GL_INVALID_VALUE);
        }
        
        // copy it to the backing and use processGLState to upload new data
        memcpy((char*)ptr->data.buffer_data + offset, data, size);
        mglBufferMarkWrite(ptr,
                           kInitBufferSubData,
                           offset,
                           size,
                           data,
                           src_hash_for_meta);

        if (ptr->data.mtl_data)
        {
            // use use metal to do the subdata call
            ctx->mtl_funcs.mtlBufferSubData(ctx, ptr, offset, size, data);
        }
        else
        {
            ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;

            // probably shouldn't have to do this... if its not bound its an excess
            ctx->state.dirty_bits |= DIRTY_BUFFER;
        }
    }
    else
    {
        // use use metal to do the subdata call
        ctx->mtl_funcs.mtlBufferSubData(ctx, ptr, offset, size, data);
        mglBufferMarkWrite(ptr,
                           kInitBufferSubData,
                           offset,
                           size,
                           data,
                           src_hash_for_meta);
    }
}

void copyBufferSubData(GLMContext ctx, Buffer *src_buf, Buffer *dst_buf, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    uint8_t *src_data = NULL;
    uint8_t *dst_data = NULL;
    bool used_map_callback = false;

    if (size < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (!mgl_range_ok_glsize(readOffset, size, src_buf->size))
    {
        fprintf(stderr, "MGL Error: copyBufferSubData: read overflow (readOffset=%ld size=%ld src_size=%ld)\n", readOffset, size, src_buf->size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (!mgl_range_ok_glsize(writeOffset, size, dst_buf->size))
    {
        fprintf(stderr, "MGL Error: copyBufferSubData: write overflow (writeOffset=%ld size=%ld dst_size=%ld)\n", writeOffset, size, dst_buf->size);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (src_buf == dst_buf)
    {
        if (readOffset == writeOffset)
        {
            ERROR_RETURN(GL_INVALID_VALUE);
        }

        uint64_t r0 = (uint64_t)readOffset;
        uint64_t w0 = (uint64_t)writeOffset;
        uint64_t usize = (uint64_t)size;

        if (writeOffset > readOffset)
        {
            if (r0 + usize > w0)
            {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
        }
        else
        {
            if (w0 + usize > r0)
            {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
        }
    }

    if (src_buf->mapped && !(src_buf->access_flags & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (dst_buf->mapped && !(dst_buf->access_flags & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (ctx->mtl_funcs.mtlMapUnmapBuffer)
    {
        src_data = (uint8_t *)ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, src_buf, readOffset, size, GL_READ_ONLY, true);
        dst_data = (uint8_t *)ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, dst_buf, writeOffset, size, GL_WRITE_ONLY, true);

        if (src_data && dst_data) {
            used_map_callback = true;
        } else {
            if (src_data) {
                ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, src_buf, readOffset, size, GL_READ_ONLY, false);
            }
            if (dst_data) {
                ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, dst_buf, writeOffset, size, GL_WRITE_ONLY, false);
            }
            src_data = NULL;
            dst_data = NULL;
        }
    }

    if (!src_data || !dst_data)
    {
        if (src_buf->data.buffer_data && dst_buf->data.buffer_data)
        {
            src_data = (uint8_t *)(uintptr_t)src_buf->data.buffer_data + (size_t)readOffset;
            dst_data = (uint8_t *)(uintptr_t)dst_buf->data.buffer_data + (size_t)writeOffset;
        }
    }

    if (!src_data || !dst_data)
    {
        fprintf(stderr,
                "MGL Error: copyBufferSubData: unable to resolve src/dst pointers "
                "(src=%p dst=%p mapCb=%p srcBase=%p dstBase=%p)\n",
                src_data,
                dst_data,
                (void *)ctx->mtl_funcs.mtlMapUnmapBuffer,
                (void *)(uintptr_t)src_buf->data.buffer_data,
                (void *)(uintptr_t)dst_buf->data.buffer_data);
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    memcpy(dst_data, src_data, size);
    dst_buf->data.dirty_bits |= DIRTY_BUFFER_DATA;
    ctx->state.dirty_bits |= DIRTY_BUFFER;
    mglBufferMarkWrite(dst_buf,
                       kInitCopyBufferSubData,
                       writeOffset,
                       size,
                       src_data,
                       mglTraceHashBytes(src_data, (size_t)size));

    if (used_map_callback)
    {
        ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, dst_buf, writeOffset, size, GL_WRITE_ONLY, false);
    }
}

void mglCopyBufferSubData(GLMContext ctx, GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    GLuint index;
    Buffer *src_buf, *dst_buf;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, readTarget), GL_INVALID_ENUM);

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, writeTarget), GL_INVALID_ENUM);

    index = bufferIndexFromTarget(ctx, readTarget);
    src_buf = STATE(buffers[index]);

    if (src_buf == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    index = bufferIndexFromTarget(ctx, writeTarget);
    dst_buf = STATE(buffers[index]);

    if (dst_buf == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    copyBufferSubData(ctx, src_buf, dst_buf, readOffset, writeOffset, size);
}

void mglCopyNamedBufferSubData(GLMContext ctx, GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    Buffer *src_buf, *dst_buf;

    if (readBuffer == 0)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (writeBuffer == 0)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    src_buf = findBuffer(ctx, readBuffer);

    if (src_buf == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    dst_buf = findBuffer(ctx, writeBuffer);

    if (dst_buf == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, src_buf->target), GL_INVALID_ENUM);

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, dst_buf->target), GL_INVALID_ENUM);

    copyBufferSubData(ctx, src_buf, dst_buf, readOffset, writeOffset, size);
}

void mglClearBufferData(GLMContext ctx, GLenum target, GLenum internalformat, GLenum format, GLenum type, const void *data)
{
    GLuint index;
    Buffer *ptr;
    GLboolean err;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    err = clearBufferData(ctx, ptr, internalformat, 0, ptr->size, format, type, data);
    ERROR_CHECK_RETURN(err == true, GL_INVALID_ENUM);
}

void mglClearBufferSubData(GLMContext ctx, GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    GLuint index;
    Buffer *ptr;
    GLboolean err;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    err = clearBufferData(ctx, ptr, internalformat, offset, size, format, type, data);
    ERROR_CHECK_RETURN(err == true, GL_INVALID_ENUM);
}

void mglClearNamedBufferData(GLMContext ctx, GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data)
{
    Buffer *ptr;
    GLboolean err;

    ptr = findBuffer(ctx, buffer);

    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    err = clearBufferData(ctx, ptr, internalformat, 0, ptr->size, format, type, data);
    ERROR_CHECK_RETURN(err == true, GL_INVALID_ENUM);
}

void mglClearNamedBufferSubData(GLMContext ctx, GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    Buffer *ptr;
    GLboolean err;

    ptr = findBuffer(ctx, buffer);

    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    err = clearBufferData(ctx, ptr, internalformat, offset, size, format, type, data);
    ERROR_CHECK_RETURN(err == true, GL_INVALID_ENUM);
}

#pragma mark GL Buffer Map Functions
void *mglMapBuffer(GLMContext ctx, GLenum target, GLenum access)
{
    GLuint index;
    Buffer *ptr;
    void *mapped_ptr = NULL;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN_VALUE(checkTarget(ctx, target), GL_INVALID_ENUM, NULL);

    switch(access)
    {
        case GL_READ_ONLY:
        case GL_WRITE_ONLY:
        case GL_READ_WRITE:
            break;

        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, NULL);
    }

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    ERROR_CHECK_RETURN_VALUE((ptr != NULL), GL_INVALID_OPERATION, NULL);

    if (ptr->mapped)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    ptr->mapped = GL_TRUE;
    ptr->access = access;
    ptr->access_flags = 0;
    ptr->mapped_offset = 0;
    ptr->mapped_length = ptr->size;
    if (ctx->mtl_funcs.mtlMapUnmapBuffer) {
        mapped_ptr = ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, ptr, 0, ptr->size, access, true);
    } else if (ptr->data.buffer_data) {
        mapped_ptr = (void *)(uintptr_t)ptr->data.buffer_data;
    }

    char head[64];
    mglTraceFormatBytes(mapped_ptr, (ptr->size > 0) ? (size_t)ptr->size : 0u, head, sizeof(head));
    fprintf(stderr,
            "MGL TRACE MapBuffer.full target=0x%x buffer=%u size=%lld access=0x%x mappedPtr=%p baseData=%p head=%s\n",
            target,
            ptr->name,
            (long long)ptr->size,
            access,
            mapped_ptr,
            (void *)(uintptr_t)ptr->data.buffer_data,
            head);

    return mapped_ptr;
}

void *mglMapNamedBuffer(GLMContext ctx, GLuint buffer, GLenum access)
{
    Buffer *ptr;
    GLbitfield access_flags = 0;

    if (!ctx || buffer == 0u)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    switch(access)
    {
        case GL_READ_ONLY:
            access_flags = GL_MAP_READ_BIT;
            break;
        case GL_WRITE_ONLY:
            access_flags = GL_MAP_WRITE_BIT;
            break;
        case GL_READ_WRITE:
            access_flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
            break;
        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, NULL);
    }

    ptr = findBuffer(ctx, buffer);
    if (!ptr)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    return mglMapNamedBufferRange(ctx, buffer, 0, ptr->size, access_flags);
}

GLboolean mglUnmapBuffer(GLMContext ctx, GLenum target)
{
    GLuint index;
    Buffer *ptr;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN_VALUE(checkTarget(ctx, target), GL_INVALID_ENUM, GL_FALSE);

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    ERROR_CHECK_RETURN_VALUE((ptr != NULL), GL_INVALID_OPERATION, GL_FALSE);

    fprintf(stderr,
            "MGL TRACE UnmapBuffer.begin target=0x%x buffer=%u mapped=%u access=0x%x accessFlags=0x%x mapRange=(%lld,%lld)\n",
            target,
            ptr->name,
            (unsigned)ptr->mapped,
            ptr->access,
            ptr->access_flags,
            (long long)ptr->mapped_offset,
            (long long)ptr->mapped_length);

    if ((ptr->storage_flags & GL_MAP_PERSISTENT_BIT) &&
        (ptr->access_flags & GL_MAP_PERSISTENT_BIT))
    {
        mglBufferMarkMapWrite(ptr);

        // this will cause the buffer to be flushed on next draw command
        ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;

        ptr->mapped = GL_FALSE;
        ptr->access = 0;
        ptr->access_flags = 0;
        ptr->mapped_offset = 0;
        ptr->mapped_length = 0;

        fprintf(stderr,
                "MGL TRACE UnmapBuffer.end persistent target=0x%x buffer=%u dirty=0x%x\n",
                target,
                ptr->name,
                ptr->data.dirty_bits);

        return GL_TRUE;
    }

    mglBufferMarkMapWrite(ptr);

    ptr->mapped = GL_FALSE;
    ptr->access = 0;
    ptr->access_flags = 0;
    ptr->mapped_offset = 0;
    ptr->mapped_length = 0;

    if (ctx->mtl_funcs.mtlMapUnmapBuffer)
    {
        ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, ptr, 0, ptr->size, 0, false);
    }

    fprintf(stderr,
            "MGL TRACE UnmapBuffer.end target=0x%x buffer=%u dirty=0x%x mapped=%u\n",
            target,
            ptr->name,
            ptr->data.dirty_bits,
            (unsigned)ptr->mapped);

    return GL_TRUE;
}

GLboolean mglUnmapNamedBuffer(GLMContext ctx, GLuint buffer)
{
    Buffer *ptr;

    if (!ctx || buffer == 0u)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, GL_FALSE);
    }

    ptr = findBuffer(ctx, buffer);
    if (!ptr)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, GL_FALSE);
    }

    if (!ptr->mapped)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, GL_FALSE);
    }

    mglBufferMarkMapWrite(ptr);

    if (!(ptr->storage_flags & GL_MAP_PERSISTENT_BIT) && ctx->mtl_funcs.mtlMapUnmapBuffer)
    {
        ctx->mtl_funcs.mtlMapUnmapBuffer(ctx,
                                         ptr,
                                         ptr->mapped_offset,
                                         ptr->mapped_length > 0 ? ptr->mapped_length : ptr->size,
                                         ptr->access_flags ? ptr->access_flags : ptr->access,
                                         false);
    }

    ptr->mapped = GL_FALSE;
    ptr->access = 0;
    ptr->access_flags = 0;
    ptr->mapped_offset = 0;
    ptr->mapped_length = 0;

    return GL_TRUE;
}

void *mglMapBufferRange(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access_flags)
{
    Buffer *ptr;
    void *mapped_ptr = NULL;

    if (!ctx || length <= 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN_VALUE(checkTarget(ctx, target), GL_INVALID_ENUM, NULL);

    if (offset < 0)
    {
        fprintf(stderr, "MGL Error: mglMapBufferRange: offset < 0 (%ld)\n", offset);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    if (length < 0)
    {
        fprintf(stderr, "MGL Error: mglMapBufferRange: length < 0 (%ld)\n", length);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    ptr = mglGetBoundBufferForTarget(ctx, target);

    if (!ptr)
    {
        fprintf(stderr, "MGL MapBufferRange NULL bound buffer target=0x%x\n", target);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (!ptr->data.buffer_data || !mgl_range_ok_glsize(offset, length, ptr->size))
    {
        fprintf(stderr, "MGL MapBufferRange invalid range target=0x%x off=%lld len=%lld size=%lld data=%p\n",
                target,
                (long long)offset,
                (long long)length,
                (long long)ptr->size,
                (void *)(uintptr_t)ptr->data.buffer_data);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    static uint64_t s_map_buffer_range_trace_count = 0u;
    uint64_t trace_call = ++s_map_buffer_range_trace_count;
    bool trace_map = MGL_VERBOSE_BUFFER_MAP_LOGS ||
                     mglShouldTraceBufferMutation(trace_call, target, length);

    const uint8_t *base = (const uint8_t *)(uintptr_t)ptr->data.buffer_data;
    const uint8_t *range_ptr = base + (size_t)offset;
    uint64_t pre_hash = mglTraceHashBytes(range_ptr, (size_t)length);
    char pre_head[64];
    mglTraceFormatBytes(range_ptr, (size_t)length, pre_head, sizeof(pre_head));
    size_t probe_len = ((size_t)length < 256u) ? (size_t)length : 256u;
    bool pre_all_zero = mglTraceSampleAllZero(range_ptr, probe_len);
    if (trace_map) {
        fprintf(stderr,
                "MGL TRACE MapBufferRange.begin target=0x%x buffer=%u off=%lld len=%lld accessFlags=0x%x data=%p rangePtr=%p preHash=0x%016" PRIx64 " preHead=%s probe256AllZero=%d\n",
                target,
                ptr->name,
                (long long)offset,
                (long long)length,
                access_flags,
                (void *)(uintptr_t)ptr->data.buffer_data,
                range_ptr,
                pre_hash,
                pre_head,
                pre_all_zero ? 1 : 0);
    }

    if (access_flags & ~(GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT |
                   GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_FLUSH_EXPLICIT_BIT |
                   GL_MAP_UNSYNCHRONIZED_BIT))
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    if (ptr->mapped)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    ptr->access = 0;
    ptr->mapped_offset = offset;
    ptr->mapped_length = length;

    if (access_flags & GL_MAP_PERSISTENT_BIT)
    {
        if (ptr->storage_flags & GL_MAP_PERSISTENT_BIT)
        {
            ptr->access_flags = access_flags;
            ptr->mapped = GL_TRUE;

            ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;

            // return a pointer to the backing data and keep mapped state for flush/unmap semantics
            mapped_ptr = (void *)((uint8_t *)(uintptr_t)ptr->data.buffer_data + (size_t)offset);
            if (trace_map) {
                fprintf(stderr,
                        "MGL TRACE MapBufferRange.return persistent target=0x%x buffer=%u mappedPtr=%p\n",
                        target,
                        ptr->name,
                        mapped_ptr);
            }
            return mapped_ptr;
        }

        // if buffer was not marked with GL_MAP_PERSISTENT_BIT in storage flags
        // it is an error to map with persistant
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }
    else if (access_flags & GL_MAP_COHERENT_BIT)
    {
        // GL_MAP_PERSISTENT_BIT and GL_MAP_COHERENT_BIT need to be together
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    ptr->access_flags = access_flags;
    ptr->mapped = GL_TRUE;

    if (!ctx->mtl_funcs.mtlMapUnmapBuffer)
    {
        // Safety fallback: keep the process alive even if Metal map callback is unavailable.
        mapped_ptr = (void *)((uint8_t *)(uintptr_t)ptr->data.buffer_data + offset);
        if (trace_map) {
            fprintf(stderr,
                    "MGL TRACE MapBufferRange.return fallback target=0x%x buffer=%u mappedPtr=%p\n",
                    target,
                    ptr->name,
                    mapped_ptr);
        }
        return mapped_ptr;
    }

    mapped_ptr = ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, ptr, offset, length, access_flags, true);
    if (trace_map) {
        fprintf(stderr,
                "MGL TRACE MapBufferRange.return mtl target=0x%x buffer=%u mappedPtr=%p\n",
                target,
                ptr->name,
                mapped_ptr);
    }
    return mapped_ptr;
}

void *mglMapNamedBufferRange(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    Buffer *ptr;
    void *mapped_ptr = NULL;

    if (!ctx)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    if (buffer == 0u)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (length <= 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    if (offset < 0)
    {
        fprintf(stderr, "MGL Error: mglMapNamedBufferRange: offset < 0 (%ld)\n", offset);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    if (length < 0)
    {
        fprintf(stderr, "MGL Error: mglMapNamedBufferRange: length < 0 (%ld)\n", length);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    ptr = findBuffer(ctx, buffer);
    if (!ptr)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (!ptr->data.buffer_data || !mgl_range_ok_glsize(offset, length, ptr->size))
    {
        fprintf(stderr, "MGL MapNamedBufferRange invalid range buffer=%u off=%lld len=%lld size=%lld data=%p\n",
                ptr->name,
                (long long)offset,
                (long long)length,
                (long long)ptr->size,
                (void *)(uintptr_t)ptr->data.buffer_data);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    if (access & ~(GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT |
                   GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_FLUSH_EXPLICIT_BIT |
                   GL_MAP_UNSYNCHRONIZED_BIT))
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    if (ptr->mapped)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    ptr->access = 0;
    ptr->mapped_offset = offset;
    ptr->mapped_length = length;

    if (access & GL_MAP_PERSISTENT_BIT)
    {
        if (ptr->storage_flags & GL_MAP_PERSISTENT_BIT)
        {
            ptr->access_flags = access;
            ptr->mapped = GL_TRUE;
            ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
            mapped_ptr = (void *)((uint8_t *)(uintptr_t)ptr->data.buffer_data + (size_t)offset);
            fprintf(stderr,
                    "MGL TRACE MapNamedBufferRange.return persistent buffer=%u mappedPtr=%p\n",
                    ptr->name,
                    mapped_ptr);
            return mapped_ptr;
        }
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }
    else if (access & GL_MAP_COHERENT_BIT)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    ptr->access_flags = access;
    ptr->mapped = GL_TRUE;

    if (!ctx->mtl_funcs.mtlMapUnmapBuffer)
    {
        mapped_ptr = (void *)((uint8_t *)(uintptr_t)ptr->data.buffer_data + (size_t)offset);
        fprintf(stderr,
                "MGL TRACE MapNamedBufferRange.return fallback buffer=%u mappedPtr=%p\n",
                ptr->name,
                mapped_ptr);
        return mapped_ptr;
    }

    mapped_ptr = ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, ptr, offset, length, access, true);
    fprintf(stderr,
            "MGL TRACE MapNamedBufferRange.return mtl buffer=%u mappedPtr=%p\n",
            ptr->name,
            mapped_ptr);
    return mapped_ptr;
}


void mglFlushMappedBufferRange(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr length)
{
    GLuint index;
    Buffer *ptr;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    if (offset < 0)
    {
        fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: offset < 0 (%ld)\n", offset);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (length < 0)
    {
        fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: length < 0 (%ld)\n", length);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    ERROR_CHECK_RETURN((ptr != NULL), GL_INVALID_OPERATION);

    if (ptr->mapped == false)
    {
        fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: buffer not mapped\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (offset < ptr->mapped_offset)
    {
        fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: offset (%ld) < mapped_offset (%ld)\n", offset, ptr->mapped_offset);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    GLsizeiptr rel_offset = offset - ptr->mapped_offset;
    if (!mgl_range_ok_glsize(rel_offset, length, ptr->mapped_length))
    {
        fprintf(stderr,
                "MGL Error: mglFlushMappedBufferRange: range overflow (offset=%ld length=%ld mapped_offset=%ld mapped_length=%ld)\n",
                offset,
                length,
                ptr->mapped_offset,
                ptr->mapped_length);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (ptr->access_flags & GL_MAP_FLUSH_EXPLICIT_BIT)
    {
        if (!ctx->mtl_funcs.mtlFlushBufferRange)
        {
            fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: mtlFlushBufferRange callback unavailable\n");
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
        ctx->mtl_funcs.mtlFlushBufferRange(ctx, ptr, offset, length);
    }
    else
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
}

void mglFlushMappedNamedBufferRange(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    Buffer *ptr;

    if (!ctx || buffer == 0u)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (offset < 0 || length < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    ptr = findBuffer(ctx, buffer);
    if (!ptr)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (!ptr->mapped)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (!(ptr->access_flags & GL_MAP_FLUSH_EXPLICIT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (offset < ptr->mapped_offset)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    {
        GLsizeiptr rel_offset = offset - ptr->mapped_offset;
        if (!mgl_range_ok_glsize(rel_offset, length, ptr->mapped_length))
        {
            ERROR_RETURN(GL_INVALID_VALUE);
        }
    }

    if (!ctx->mtl_funcs.mtlFlushBufferRange)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    ctx->mtl_funcs.mtlFlushBufferRange(ctx, ptr, offset, length);
}

void mglBindBuffersRange(GLMContext ctx, GLenum target, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizeiptr *sizes)
{
    if (!ctx) {
        return;
    }

    if (count < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (count == 0) {
        return;
    }

    switch(target)
    {
        case GL_UNIFORM_BUFFER:
        case GL_TRANSFORM_FEEDBACK_BUFFER:
        case GL_SHADER_STORAGE_BUFFER:
        case GL_ATOMIC_COUNTER_BUFFER:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    if (buffers && (!offsets || !sizes))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    for (GLsizei i = 0; i < count; i++)
    {
        GLuint index = first + (GLuint)i;
        GLuint name = buffers ? buffers[i] : 0u;

        if (name == 0u)
        {
            mglBindBufferBase(ctx, target, index, 0u);
            continue;
        }

        mglBindBufferRange(ctx, target, index, name, offsets[i], sizes[i]);
    }
}

#pragma mark GL Buffer Storage Functions
void mglBufferStorage(GLMContext ctx, GLenum target, GLsizeiptr size, const void *data, GLbitfield storage_flags)
{
    Buffer *ptr;
    GLuint index;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    if (size < 0)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (storage_flags & ~(GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT |
                  GL_CLIENT_STORAGE_BIT))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    index = bufferIndexFromTarget(ctx, target);

    ptr = STATE(buffers[index]);

    ERROR_CHECK_RETURN((ptr != NULL), GL_INVALID_OPERATION);

    bufferStorage(ctx, ptr, target, index, size, data, storage_flags, 0);
}

void mglNamedBufferStorage(GLMContext ctx, GLuint buffer, GLsizeiptr size, const void *data, GLbitfield storage_flags)
{
    Buffer *ptr;

    ptr = findBuffer(ctx, buffer);

    ERROR_CHECK_RETURN((ptr != NULL), GL_INVALID_OPERATION);

    if (size < 0)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (storage_flags & ~(GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT |
                  GL_CLIENT_STORAGE_BIT))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    bufferStorage(ctx, ptr, 0, 0, size, data, storage_flags, 0);
}

void mglInvalidateBufferData(GLMContext ctx, GLuint buffer)
{
    (void)buffer;
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglInvalidateBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    (void)buffer;
    (void)offset;
    (void)length;
    ERROR_RETURN(GL_INVALID_OPERATION);
}

#pragma mark GL Buffer Get Functions
void mglGetBufferParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    GLuint index;
    Buffer *ptr;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN((params != NULL), GL_INVALID_OPERATION);

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    ERROR_CHECK_RETURN((ptr != NULL), GL_INVALID_OPERATION);

    switch(pname)
    {
        case GL_BUFFER_ACCESS:
            *params = ptr->access;
            break;

        case GL_BUFFER_ACCESS_FLAGS:
            *params = ptr->access_flags;
            break;

        case GL_BUFFER_IMMUTABLE_STORAGE:
            break;

        case GL_BUFFER_MAPPED:
            *params = ptr->mapped;
            break;

        case GL_BUFFER_MAP_LENGTH:
            assert(ptr->mapped_length < INT_MAX);
            *params = (GLint)ptr->mapped_length;
            break;

        case GL_BUFFER_MAP_OFFSET:
            assert(ptr->mapped_offset < INT_MAX);
            *params = (GLint)ptr->mapped_offset;
            break;

        case GL_BUFFER_SIZE:
            assert(ptr->size < INT_MAX);
            *params = (GLint)ptr->size;
            break;

        case GL_BUFFER_STORAGE_FLAGS:
            *params = ptr->storage_flags;
            break;

        case GL_BUFFER_USAGE:
            *params = ptr->usage;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglGetBufferPointerv(GLMContext ctx, GLenum target, GLenum pname, void **params)
{
    GLuint index;
    Buffer *ptr;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    switch(pname)
    {
        case GL_BUFFER_MAP_POINTER:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }

    ERROR_CHECK_RETURN((params != NULL), GL_INVALID_OPERATION);

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    ERROR_CHECK_RETURN((ptr != NULL), GL_INVALID_OPERATION);

    if (ptr->mapped) {
        *params = (void *)ptr->data.buffer_data;
    } else {
        *params = NULL;
    }
}

void mglGetBufferSubData(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr size, void *data)
{
    GLuint index;
    Buffer *ptr;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    if (offset < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (size < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    index = bufferIndexFromTarget(ctx, target);
    ptr = STATE(buffers[index]);

    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (!mgl_range_ok_glsize(offset, size, ptr->size))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (ptr->mapped && !(ptr->access & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (data == NULL)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (ptr->data.buffer_data == 0)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (!mgl_range_ok_size_t(offset, size, ptr->data.buffer_size))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    memcpy(data, (const uint8_t *)((uintptr_t)ptr->data.buffer_data) + (uintptr_t)offset, (size_t)size);
}

void mglGetNamedBufferParameteriv(GLMContext ctx, GLuint buffer, GLenum pname, GLint *params)
{
    Buffer *ptr;

    ptr = findBuffer(ctx, buffer);
    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }


    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch(pname)
    {
        case GL_BUFFER_ACCESS:
            *params = ptr->access;
            break;

        case GL_BUFFER_MAPPED:
            *params = ptr->mapped;
            break;

        case GL_BUFFER_SIZE:
            *params = (GLint)ptr->size;
            break;

        case GL_BUFFER_USAGE:
            *params = ptr->usage;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }
}

void mglGetNamedBufferParameteri64v(GLMContext ctx, GLuint buffer, GLenum pname, GLint64 *params)
{
    Buffer *ptr;

    ptr = findBuffer(ctx, buffer);
    if (ptr == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }


    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch(pname)
    {
        case GL_BUFFER_ACCESS:
            *params = ptr->access;
            break;

        case GL_BUFFER_ACCESS_FLAGS:
            *params = ptr->access_flags;
            break;

        case GL_BUFFER_IMMUTABLE_STORAGE:
            *params = ptr->immutable_storage;
            break;

        case GL_BUFFER_MAPPED:
            *params = ptr->mapped;
            break;

        case GL_BUFFER_MAP_LENGTH:
            *params = ptr->mapped_length;
            break;

        case GL_BUFFER_MAP_OFFSET:
            *params = ptr->mapped_offset;
            break;

        case GL_BUFFER_SIZE:
            *params = ptr->size;
            break;

        case GL_BUFFER_USAGE:
            *params = ptr->usage;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }
}

void mglGetNamedBufferPointerv(GLMContext ctx, GLuint buffer, GLenum pname, void **params)
{
    Buffer *ptr;

    if (!ctx)
    {
        return;
    }

    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (pname != GL_BUFFER_MAP_POINTER)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    ptr = findBuffer(ctx, buffer);
    if (!ptr)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (ptr->mapped && ptr->data.buffer_data)
    {
        *params = (void *)((uint8_t *)(uintptr_t)ptr->data.buffer_data + (size_t)ptr->mapped_offset);
    }
    else
    {
        *params = NULL;
    }
}

void mglGetNamedBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr size, void *data)
{
    Buffer *ptr;

    if (!ctx)
    {
        return;
    }

    ptr = findBuffer(ctx, buffer);
    if (!ptr)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (offset < 0 || size < 0 || !data)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!mgl_range_ok_glsize(offset, size, ptr->size))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (ptr->mapped && !(ptr->access_flags & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (ptr->data.buffer_data == 0 || !mgl_range_ok_size_t(offset, size, ptr->data.buffer_size))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    memcpy(data, (const uint8_t *)((uintptr_t)ptr->data.buffer_data) + (uintptr_t)offset, (size_t)size);
}
