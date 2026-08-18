/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

// mgl_index_buffer.cpp - C++ owner for primitive-emulation index buffers.
//
// The GL-facing ABI intentionally carries only opaque handles and scalar
// values. All MTL::Buffer creation, contents access, cache ownership, and
// release operations live in this translation unit.

#include "mgl_metal_cpp.h"
#include "mgl_index_buffer.h"
#include "mgl_render_cpp.h"
#include "mgl_types_buffer.h"
#include "mgl_safety.h"

#include <cstring>
#include <cstdlib>
#include <limits>
#include <os/lock.h>

extern "C" void mglNoteBufferEncoded(Buffer *buffer);

namespace {

constexpr size_t kLineLoopCacheSlots = 4u;
struct ArrayCacheEntry {
    void *buffer = nullptr;
    size_t count = 0;
};
os_unfair_lock g_cache_lock = OS_UNFAIR_LOCK_INIT;
ArrayCacheEntry g_fan;
ArrayCacheEntry g_strip;
ArrayCacheEntry g_quad;
ArrayCacheEntry g_quad_line;
struct LineLoopEntry {
    size_t first = 0;
    size_t count = 0;
    void *buffer = nullptr;
};
LineLoopEntry g_line_loop[kLineLoopCacheSlots];

void releaseObject(void *object) {
    if (object) static_cast<NS::Object *>(object)->release();
}

void *newBuffer(void *device, size_t bytes, void **contents_out) {
    if (contents_out) *contents_out = nullptr;
    if (!device || bytes == 0u) return nullptr;
    void *buffer = nullptr;
    if (mglRenderCppCreateBuffer(
            static_cast<uint64_t>(bytes),
            static_cast<uint64_t>(MTL::ResourceStorageModeShared),
            "MGL.index_emulation", &buffer) != 0 || !buffer) {
        return nullptr;
    }
    if (contents_out) {
        void *contents = nullptr;
        uint64_t length = 0;
        if (mglRenderCppGetBufferContents(buffer, &contents, &length) != 0 ||
            !contents || length < bytes) {
            releaseObject(buffer);
            return nullptr;
        }
        *contents_out = contents;
    }
    return buffer;
}

void *copyIndices(void *device, const uint32_t *indices, size_t count) {
    if (!indices || count == 0u || count > SIZE_MAX / sizeof(uint32_t)) return nullptr;
    void *contents = nullptr;
    void *buffer = newBuffer(device, count * sizeof(uint32_t), &contents);
    if (!buffer) return nullptr;
    std::memcpy(contents, indices, count * sizeof(uint32_t));
    return buffer;
}

void *copyUInt16(void *device, const uint16_t *indices, size_t count) {
    if (!indices || count == 0u || count > SIZE_MAX / sizeof(uint16_t)) return nullptr;
    void *contents = nullptr;
    void *buffer = newBuffer(device, count * sizeof(uint16_t), &contents);
    if (!buffer) return nullptr;
    std::memcpy(contents, indices, count * sizeof(uint16_t));
    return buffer;
}

uint32_t elementWidth(GLenum type) {
    return type == GL_UNSIGNED_BYTE ? 1u :
           type == GL_UNSIGNED_SHORT ? 2u :
           type == GL_UNSIGNED_INT ? 4u : 0u;
}

void *cachedPrefix(void *device, ArrayCacheEntry &entry, size_t requested,
                   size_t index_count, int (*expand)(uint32_t, uint32_t **, uint64_t *),
                   uint32_t argument, size_t *out_count) {
    if (out_count) *out_count = 0u;
    if (!device || requested == 0u || index_count == 0u) return nullptr;
    {
        os_unfair_lock_lock(&g_cache_lock);
        if (entry.buffer && requested <= entry.count) {
            if (out_count) *out_count = index_count;
            void *cached = entry.buffer;
            os_unfair_lock_unlock(&g_cache_lock);
            return cached;
        }
        os_unfair_lock_unlock(&g_cache_lock);
    }
    uint32_t *expanded = nullptr;
    uint64_t expanded_count = 0;
    if (expand(argument, &expanded, &expanded_count) != 0 ||
        expanded_count != index_count) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(expanded_count));
    std::free(expanded);
    if (!buffer) return nullptr;
    os_unfair_lock_lock(&g_cache_lock);
    if (entry.buffer) releaseObject(entry.buffer);
    entry.buffer = buffer;
    entry.count = requested;
    if (out_count) *out_count = index_count;
    void *prepared = entry.buffer;
    os_unfair_lock_unlock(&g_cache_lock);
    return prepared;
}

void *expandElement(void *device, const uint8_t *source, GLenum type,
                    size_t source_count, size_t *out_count,
                    int (*expand)(const uint8_t *, uint32_t, uint32_t,
                                  uint32_t **, uint64_t *)) {
    if (out_count) *out_count = 0u;
    const uint32_t width = elementWidth(type);
    if (!device || !source || width == 0u || source_count == 0u ||
        source_count > UINT32_MAX) return nullptr;
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (expand(source, width, static_cast<uint32_t>(source_count),
               &expanded, &count) != 0 || count == 0u || count > SIZE_MAX) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (buffer && out_count) *out_count = static_cast<size_t>(count);
    return buffer;
}

} // namespace

extern "C" {

MGLIndexMetalHandle mglNewTriangleFanArrayIndexBuffer(
    MGLIndexMetalHandle device, size_t vertex_count, size_t *out_count) {
    if (vertex_count < 3u || vertex_count > UINT32_MAX) {
        if (out_count) *out_count = 0u;
        return nullptr;
    }
    const size_t count = (vertex_count - 2u) * 3u;
    return cachedPrefix(device, g_fan, vertex_count, count,
                        mglRenderCppExpandTriangleFanArrayIndices,
                        static_cast<uint32_t>(vertex_count), out_count);
}

MGLIndexMetalHandle mglNewTriangleStripArrayIndexBuffer(
    MGLIndexMetalHandle device, size_t vertex_count, size_t *out_count) {
    if (vertex_count < 3u || vertex_count > UINT32_MAX) {
        if (out_count) *out_count = 0u;
        return nullptr;
    }
    const size_t count = (vertex_count - 2u) * 3u;
    return cachedPrefix(device, g_strip, vertex_count, count,
                        mglRenderCppExpandTriangleStripArrayIndices,
                        static_cast<uint32_t>(vertex_count), out_count);
}

MGLIndexMetalHandle mglNewQuadArrayIndexBuffer(
    MGLIndexMetalHandle device, size_t vertex_count, size_t *out_count) {
    if (vertex_count < 4u || vertex_count > UINT32_MAX) {
        if (out_count) *out_count = 0u;
        return nullptr;
    }
    const uint32_t quads = static_cast<uint32_t>(vertex_count / 4u);
    const size_t count = static_cast<size_t>(quads) * 6u;
    return cachedPrefix(device, g_quad, vertex_count, count,
                        mglRenderCppExpandQuadArrayIndices, quads, out_count);
}

MGLIndexMetalHandle mglNewQuadArrayLineIndexBuffer(
    MGLIndexMetalHandle device, size_t vertex_count, size_t *out_count) {
    if (vertex_count < 4u || vertex_count > UINT32_MAX) {
        if (out_count) *out_count = 0u;
        return nullptr;
    }
    const uint32_t quads = static_cast<uint32_t>(vertex_count / 4u);
    const size_t count = static_cast<size_t>(quads) * 8u;
    return cachedPrefix(device, g_quad_line, vertex_count, count,
                        mglRenderCppExpandQuadArrayLineIndices, quads, out_count);
}

MGLIndexMetalHandle mglNewLineLoopArrayIndexBuffer(
    MGLIndexMetalHandle device, size_t first, size_t vertex_count,
    size_t *out_count) {
    if (out_count) *out_count = 0u;
    if (!device || vertex_count < 2u || first > UINT32_MAX ||
        vertex_count > UINT32_MAX || first + vertex_count > UINT32_MAX + 1ull) return nullptr;
    {
        os_unfair_lock_lock(&g_cache_lock);
        for (const auto &entry : g_line_loop) {
            if (entry.buffer && entry.first == first && entry.count == vertex_count) {
                if (out_count) *out_count = vertex_count + 1u;
                void *cached = entry.buffer;
                os_unfair_lock_unlock(&g_cache_lock);
                return cached;
            }
        }
        os_unfair_lock_unlock(&g_cache_lock);
    }
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (mglRenderCppExpandLineLoopArrayIndices(
            static_cast<uint32_t>(first), static_cast<uint32_t>(vertex_count),
            &expanded, &count) != 0 || count == 0u) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (!buffer) return nullptr;
    os_unfair_lock_lock(&g_cache_lock);
    for (auto &entry : g_line_loop) {
        if (!entry.buffer) {
            entry.first = first;
            entry.count = vertex_count;
            entry.buffer = buffer;
            if (out_count) *out_count = static_cast<size_t>(count);
            os_unfair_lock_unlock(&g_cache_lock);
            return buffer;
        }
    }
    // Fixed cache full: return the newly created owner to the caller.
    if (out_count) *out_count = static_cast<size_t>(count);
    os_unfair_lock_unlock(&g_cache_lock);
    return buffer;
}

MGLIndexMetalHandle mglNewTriangleFanElementIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t count, size_t *out_count) {
    return expandElement(device, source, type, count, out_count,
                          mglRenderCppExpandTriangleFanIndices);
}

MGLIndexMetalHandle mglNewTriangleStripElementIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t count, size_t *out_count) {
    return expandElement(device, source, type, count, out_count,
                          mglRenderCppExpandTriangleStripIndices);
}

MGLIndexMetalHandle mglNewLineLoopElementIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t count, size_t *out_count) {
    return expandElement(device, source, type, count, out_count,
                          mglRenderCppExpandLineLoopIndices);
}

MGLIndexMetalHandle mglNewQuadElementIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t source_count, size_t *out_count) {
    if (out_count) *out_count = 0u;
    const uint32_t width = elementWidth(type);
    const size_t quads = source_count / 4u;
    if (!device || !source || width == 0u || quads == 0u || quads > UINT32_MAX) return nullptr;
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (mglRenderCppExpandQuadElementIndices(
            source, width, static_cast<uint32_t>(quads), &expanded, &count) != 0) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (buffer && out_count) *out_count = static_cast<size_t>(count);
    return buffer;
}

MGLIndexMetalHandle mglNewQuadElementLineIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t source_count, size_t *out_count) {
    if (out_count) *out_count = 0u;
    const uint32_t width = elementWidth(type);
    const size_t quads = source_count / 4u;
    if (!device || !source || width == 0u || quads == 0u || quads > UINT32_MAX) return nullptr;
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (mglRenderCppExpandQuadElementLineIndices(
            source, width, static_cast<uint32_t>(quads), &expanded, &count) != 0) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (buffer && out_count) *out_count = static_cast<size_t>(count);
    return buffer;
}

MGLIndexMetalHandle mglNewUInt16IndexBufferFromUInt8(
    MGLIndexMetalHandle device, const uint8_t *source, size_t count) {
    if (!device || !source || count == 0u || count > UINT32_MAX) return nullptr;
    uint16_t *expanded = nullptr;
    uint64_t expanded_count = 0;
    if (mglRenderCppExpandUInt8ToUInt16(
            source, static_cast<uint32_t>(count), &expanded, &expanded_count) != 0) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyUInt16(device, expanded, static_cast<size_t>(expanded_count));
    std::free(expanded);
    return buffer;
}

const uint8_t *mglReadableBufferBytes(Buffer *gl_buffer,
                                      MGLIndexMetalHandle metal_buffer,
                                      size_t *out_count) {
    if (out_count) *out_count = 0u;
    if (gl_buffer && gl_buffer->data.buffer_data) {
        size_t count = gl_buffer->data.buffer_size > 0
            ? static_cast<size_t>(gl_buffer->data.buffer_size)
            : static_cast<size_t>(gl_buffer->size > 0 ? gl_buffer->size : 0);
        const uint8_t *bytes = static_cast<const uint8_t *>(
            reinterpret_cast<const void *>(static_cast<uintptr_t>(gl_buffer->data.buffer_data)));
        if (bytes && count && mglPointerRangeIsReadable(bytes, count)) {
            if (out_count) *out_count = count;
            return bytes;
        }
    }
    void *contents = nullptr;
    uint64_t length = 0;
    if (metal_buffer && mglRenderCppGetBufferContents(
            metal_buffer, &contents, &length) == 0 && contents && length) {
        if (out_count) *out_count = static_cast<size_t>(length);
        return static_cast<const uint8_t *>(contents);
    }
    return nullptr;
}

const uint8_t *mglElementIndexSourceBytes(Buffer *gl_buffer,
                                          MGLIndexMetalHandle metal_buffer,
                                          size_t *out_count) {
    return mglReadableBufferBytes(gl_buffer, metal_buffer, out_count);
}

const uint8_t *mglElementIndexSourceForDraw(Buffer *gl_buffer,
                                            MGLIndexMetalHandle metal_buffer,
                                            GLenum type, size_t offset,
                                            GLsizei count) {
    size_t available = 0;
    const uint8_t *bytes = mglElementIndexSourceBytes(gl_buffer, metal_buffer, &available);
    const size_t stride = mglGLIndexElementSize(type);
    if (!bytes || !available || !stride || count <= 0 ||
        static_cast<size_t>(count) > SIZE_MAX / stride) return nullptr;
    const size_t needed = static_cast<size_t>(count) * stride;
    if (offset > available || available - offset < needed) return nullptr;
    return bytes + offset;
}

bool mglReadBufferBytes(Buffer *gl_buffer, MGLIndexMetalHandle metal_buffer,
                        size_t offset, void *dst, size_t count, const char *) {
    if (!dst || count == 0u) return false;
    size_t available = 0;
    const uint8_t *bytes = mglReadableBufferBytes(gl_buffer, metal_buffer, &available);
    if (!bytes || offset > available || available - offset < count) return false;
    std::memcpy(dst, bytes + offset, count);
    return true;
}

MGLIndexMetalHandle mglPreparedElementIndexBuffer(
    MGLIndexMetalHandle device, Buffer *gl_buffer,
    MGLIndexMetalHandle metal_buffer, GLenum type,
    size_t *io_offset, uint64_t *out_type) {
    if (out_type) {
        *out_type =
            type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_SHORT ? 0u :
            type == GL_UNSIGNED_INT ? 1u : UINT32_MAX;
    }
    if (type != GL_UNSIGNED_BYTE) {
        if (gl_buffer) mglNoteBufferEncoded(gl_buffer);
        return metal_buffer;
    }
    const size_t source_offset = io_offset ? *io_offset : 0u;
    size_t source_count = 0;
    const uint8_t *source = mglElementIndexSourceBytes(
        gl_buffer, metal_buffer, &source_count);
    if (!source || source_count == 0u) return nullptr;
    const bool eligible = gl_buffer && gl_buffer->data.buffer_data &&
        !(gl_buffer->storage_flags & GL_MAP_PERSISTENT_BIT);
    if (eligible && gl_buffer->mtl_uint16_expanded_data &&
        gl_buffer->mtl_uint16_expanded_src_hash == gl_buffer->last_write_src_hash &&
        gl_buffer->mtl_uint16_expanded_src_hash != 0u &&
        gl_buffer->mtl_uint16_expanded_byte_count == source_count) {
        if (io_offset) {
            if (source_offset > SIZE_MAX / sizeof(uint16_t)) return nullptr;
            *io_offset = source_offset * sizeof(uint16_t);
        }
        if (out_type) *out_type = 0u;
        return gl_buffer->mtl_uint16_expanded_data;
    }
    void *expanded = mglNewUInt16IndexBufferFromUInt8(device, source, source_count);
    if (!expanded) return nullptr;
    if (eligible) {
        if (gl_buffer->mtl_uint16_expanded_data) {
            releaseObject(gl_buffer->mtl_uint16_expanded_data);
        }
        static_cast<NS::Object *>(expanded)->retain();
        gl_buffer->mtl_uint16_expanded_data = expanded;
        gl_buffer->mtl_uint16_expanded_src_hash = gl_buffer->last_write_src_hash;
        gl_buffer->mtl_uint16_expanded_byte_count = source_count;
    }
    if (io_offset) {
        if (source_offset > SIZE_MAX / sizeof(uint16_t)) return nullptr;
        *io_offset = source_offset * sizeof(uint16_t);
    }
    if (out_type) *out_type = 0u;
    return expanded;
}

} // extern "C"
