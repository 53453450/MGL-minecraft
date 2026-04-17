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
#include <limits.h>
#include <stdint.h>

#include "glm_context.h"
#include "buffers.h"
#include "pixel_utils.h"
#include "mgl_safety.h"

// Used to recover from a corrupted context pointer (e.g. small non-NULL values like 0x2f)
extern void mgl_lazy_init(void);
extern GLMContext _ctx;

#pragma mark Utility Functions

GLuint bufferIndexFromTarget(GLMContext ctx, GLenum target)
{
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
            assert(0); // shouldn't get here without checking for error
            break;
    }

    assert(0);
    return 0xFFFFFFFF;
}

Buffer *newBuffer(GLMContext ctx, GLenum target, GLuint name)
{
    Buffer *ptr;

    ptr = (Buffer *)malloc(sizeof(Buffer));
    assert(ptr);

    bzero(ptr, sizeof(Buffer));

    ptr->name = name;
    ptr->target = target;

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

    ptr = (Buffer *)searchHashTable(&STATE(buffer_table), buffer);

    if (!ptr)
    {
        ptr = newBuffer(ctx, target, buffer);

        insertHashElement(&STATE(buffer_table), buffer, ptr);
    }

    return ptr;
}

bool isBuffer(GLMContext ctx, GLuint buffer)
{
    Buffer *ptr;

    ptr = (Buffer *)searchHashTable(&STATE(buffer_table), buffer);

    if (ptr)
        return true;

    return false;
}

Buffer *findBuffer(GLMContext ctx, GLuint buffer)
{
    Buffer *ptr;

    ptr = (Buffer *)searchHashTable(&STATE(buffer_table), buffer);

    return ptr;
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
            for (size_t i = 1; i < STATE(vao_table).size; i++)
            {
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

    for (size_t i = 1; i < STATE(vao_table).size; i++)
    {
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

    ptr = STATE(buffers[_PIXEL_UNPACK_BUFFER]);

    ERROR_CHECK_RETURN(ptr->mapped == false, GL_INVALID_OPERATION);

    buffer_data = (void *)ptr->data.buffer_data;

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

    // COMPREHENSIVE BUFFER SAFETY: Validate buffer pointer and get size safely
    GLsizeiptr buffer_size;
    MGL_GET_BUFFER_SIZE_SAFE(ptr, buffer_size, "clearBufferData");

    if (!mgl_range_ok_glsize(offset, size, buffer_size))
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    size_t pixel_size = sizeForInternalFormat(internalformat, format, type);
    assert(pixel_size);

    size_t pixel_count;
    pixel_count = size / pixel_size;

    GLubyte *dst;
    dst = (GLubyte *)data + offset;

    for(int i=0; i<pixel_count; i++)
    {
        memcpy(dst, data, pixel_size);
        dst += pixel_size;
    }

    assert(0);
}


#pragma mark GL Buffer Functions
void mglGenBuffers(GLMContext ctx, GLsizei n, GLuint *buffers)
{
    while(n--)
    {
        *buffers++ = getNewName(&STATE(buffer_table));
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

    fprintf(stderr, "MGL BindBuffer target=0x%x buffer=%u ctx=%p vao=%p\n",
            target, buffer, (void *)ctx, ctx ? (void *)ctx->state.vao : NULL);

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
    ERROR_CHECK_RETURN(index < TEXTURE_UNITS, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(isBuffer(ctx, buffer), GL_INVALID_VALUE);

    buffer_index = bufferIndexFromTarget(ctx, target);

    if (buffer)
    {
        ptr = getBuffer(ctx, target, buffer);
        assert(ptr);

        ERROR_CHECK_RETURN(ptr->data.buffer_size, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(ptr->data.buffer_data, GL_INVALID_VALUE);

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
        mglBindBufferBase(ctx, target, first++, *buffers++);
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
    ERROR_CHECK_RETURN(index < TEXTURE_UNITS, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(isBuffer(ctx, buffer), GL_INVALID_VALUE);

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
        ptr = getBuffer(ctx, target, buffer);
        assert(ptr);

        // ERROR_CHECK_RETURN(ptr->data.buffer_data, GL_INVALID_VALUE);
        if (!ptr->data.buffer_data) {
             fprintf(stderr, "MGL Error: mglBindBufferRange: buffer_data is NULL\n");
             ERROR_RETURN(GL_INVALID_VALUE);
        }

        if (!mgl_range_ok_size_t(offset, size, ptr->data.buffer_size)) {
            fprintf(stderr, "MGL Error: mglBindBufferRange: range overflow (offset=%ld size=%ld buffer_size=%ld)\n", offset, size, (long)ptr->data.buffer_size);
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
                if (data)
                {
                    memcpy((void *)ptr->data.buffer_data, data, size);
                    
                    ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
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
    }

    return err;
}

void mglBufferData(GLMContext ctx, GLenum target, GLsizeiptr size, const void *data, GLenum usage)
{
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
        // Some apps call glBufferData without binding a buffer first
        // ERROR_RETURN(GL_INVALID_OPERATION);
        return;
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
    // Absolute first check - validate ctx before ANY access
    ctx = mgl_sanitize_ctx(ctx, __FUNCTION__);
    if (!ctx)
        return;
    
    GLuint index;
    Buffer * volatile ptr;  // volatile to prevent compiler optimizations

    // GL_INVALID_ENUM is generated if target is not supported.
    if (!checkTarget(ctx, target)) {
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_ENUM);
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
        // fprintf(stderr, "MGL WARNING: mglBufferSubData: data is NULL\n");
        return;  // Silent return for NULL data
    }

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
        
        memcpy((char*)ptr->data.buffer_data + offset, data, size);
        ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
        ctx->state.dirty_bits |= DIRTY_BUFFER;
    }
    else
    {
        if (ctx->mtl_funcs.mtlBufferSubData)
        {
            ctx->mtl_funcs.mtlBufferSubData(ctx, ptr, offset, size, data);
        }
    }
}

void mglNamedBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data)
{
    Buffer *ptr;

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
    }
}

void copyBufferSubData(GLMContext ctx, Buffer *src_buf, Buffer *dst_buf, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    GLuint *src_data, *dst_data;

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

    if (src_buf->mapped && !(src_buf->access & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (dst_buf->mapped && !(dst_buf->access & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    src_data = ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, src_buf, readOffset, size, GL_READ_ONLY, true);
    assert(src_data);

    dst_data = ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, dst_buf, writeOffset, size, GL_WRITE_ONLY, true);
    assert(dst_data);

    memcpy(dst_data, src_data, size);

    ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, dst_buf, writeOffset, size, GL_WRITE_ONLY, false);
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

    err = clearBufferData(ctx, ptr, internalformat, offset, ptr->size, format, type, data);
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

    err = clearBufferData(ctx, ptr, internalformat, offset, ptr->size, format, type, data);
    ERROR_CHECK_RETURN(err == true, GL_INVALID_ENUM);
}

#pragma mark GL Buffer Map Functions
void *mglMapBuffer(GLMContext ctx, GLenum target, GLenum access)
{
    GLuint index;
    Buffer *ptr;

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

    return ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, ptr, 0, ptr->size, access, true);
}

void *mglMapNamedBuffer(GLMContext ctx, GLuint buffer, GLenum access)
{
    // Unimplemented function
    assert(0);
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

    if ((ptr->storage_flags & GL_MAP_PERSISTENT_BIT) &&
        (ptr->access & GL_MAP_PERSISTENT_BIT))
    {
        // this will cause the buffer to be flushed on next draw command
        ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;

        assert(ptr->mapped == GL_FALSE);
        ptr->access = 0;

        return GL_TRUE;
    }

    ptr->mapped = GL_FALSE;
    ptr->access = 0;
    ptr->access_flags = 0;
    ptr->mapped_offset = 0;
    ptr->mapped_length = 0;

    if (ctx->mtl_funcs.mtlMapUnmapBuffer)
    {
        ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, ptr, 0, ptr->size, 0, false);
    }

    return GL_TRUE;
}

GLboolean mglUnmapNamedBuffer(GLMContext ctx, GLuint buffer)
{
    GLboolean ret = 0;

    // Unimplemented function
    assert(0);
    return ret;
}

void *mglMapBufferRange(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access_flags)
{
    Buffer *ptr;

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

            ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;

            // return a pointer to the backing data without marking it as mapped
            return (void *)ptr->data.buffer_data;
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
        return (void *)((uint8_t *)(uintptr_t)ptr->data.buffer_data + offset);
    }

    return ctx->mtl_funcs.mtlMapUnmapBuffer(ctx, ptr, offset, length, access_flags, true);
}

void *mglMapNamedBufferRange(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    // Unimplemented function
    assert(0);
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

    bool isCoherent = ptr->access & GL_MAP_COHERENT_BIT;

    if ((ptr->mapped == false) && (isCoherent == false))
    {
        // not mapped by map buffer range
        fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: buffer not mapped\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (offset < ptr->mapped_offset)
    {
        fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: offset (%ld) < mapped_offset (%ld)\n", offset, ptr->mapped_offset);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (offset + length > ptr->mapped_length)
    {
        fprintf(stderr, "MGL Error: mglFlushMappedBufferRange: range overflow (offset=%ld length=%ld mapped_length=%ld)\n", offset, length, ptr->mapped_length);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (ptr->access_flags & GL_MAP_FLUSH_EXPLICIT_BIT)
    {
        ctx->mtl_funcs.mtlFlushBufferRange(ctx, ptr, offset, length);
    }
    else
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
}

void mglFlushMappedNamedBufferRange(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    // Unimplemented function
    assert(0);
}

void mglBindBuffersRange(GLMContext ctx, GLenum target, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizeiptr *sizes)
{
    // Unimplemented function
    assert(0);
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
    // Unimplemented function
    assert(0);
}

void mglInvalidateBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    // Unimplemented function
    assert(0);
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


    ERROR_CHECK_RETURN(params == NULL, GL_INVALID_ENUM);

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
            ERROR_RETURN(GL_INVALID_OPERATION);
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


    ERROR_CHECK_RETURN(params == NULL, GL_INVALID_ENUM);

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
            ERROR_RETURN(GL_INVALID_OPERATION);
    }
}

void mglGetNamedBufferPointerv(GLMContext ctx, GLuint buffer, GLenum pname, void **params)
{
    // Unimplemented function
    assert(0);
}

void mglGetNamedBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr size, void *data)
{
    // Unimplemented function
    assert(0);
}
