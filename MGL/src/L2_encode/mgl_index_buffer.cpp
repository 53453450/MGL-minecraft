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

#include "mgl_backend_objects.h"
#include "mgl_metal.h"
#include "mgl_index_buffer.h"
#include "mgl_buffer_map.h"   /* mglNoteBufferEncoded */
#include "mgl_render_api_buffer.h"
#include "mgl_render_api_pixel.h"
#include "mgl_render_api_draw.h"
#include "mgl_types_buffer.h"
#include "mgl_safety.h"

#include <cstring>
#include <cstdlib>
#include <limits>
#include <os/lock.h>


namespace {

constexpr size_t kLineLoopCacheSlots = 4u;
/* Ring of retired index-cache buffers: delay release so in-flight draws
 * that still reference a replaced cache entry cannot UAF (AUDIT-004). */
constexpr size_t kRetiredIndexSlots = 16u;
struct ArrayCacheEntry {
    void *buffer = nullptr;
    size_t count = 0;
};
os_unfair_lock g_cache_lock = OS_UNFAIR_LOCK_INIT;
ArrayCacheEntry g_fan;
ArrayCacheEntry g_strip;
ArrayCacheEntry g_quad;
ArrayCacheEntry g_quad_line;
struct AdjacencyArrayCacheEntry {
    void *buffer = nullptr;
    GLenum mode = 0;
    size_t vertex_count = 0;
    size_t index_count = 0;
};
AdjacencyArrayCacheEntry g_adjacency_array;
struct AdjacencyElementCacheEntry {
    void *buffer = nullptr;
    GLenum mode = 0;
    GLenum type = 0;
    size_t source_count = 0;
    uint64_t source_hash = 0;
    size_t index_count = 0;
};
AdjacencyElementCacheEntry g_adjacency_element;

static uint64_t hashIndexBytes(const uint8_t *source, size_t byte_count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < byte_count; ++i) {
        hash ^= static_cast<uint64_t>(source[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}
struct LineLoopEntry {
    size_t first = 0;
    size_t count = 0;
    void *buffer = nullptr;
};
LineLoopEntry g_line_loop[kLineLoopCacheSlots];
void *g_retired_index_buffers[kRetiredIndexSlots] = {};
size_t g_retired_index_count = 0;

void releaseObject(void *object) {
    if (object) static_cast<NS::Object *>(object)->release();
}

void deferReleaseObject(void *object) {
    if (!object) return;
    if (g_retired_index_count == kRetiredIndexSlots) {
        releaseObject(g_retired_index_buffers[0]);
        for (size_t i = 1; i < kRetiredIndexSlots; ++i) {
            g_retired_index_buffers[i - 1u] = g_retired_index_buffers[i];
        }
        g_retired_index_count = kRetiredIndexSlots - 1u;
    }
    g_retired_index_buffers[g_retired_index_count++] = object;
}

void *newBuffer(void *device, size_t bytes, void **contents_out) {
    if (contents_out) *contents_out = nullptr;
    if (!device || bytes == 0u) return nullptr;
    void *buffer = nullptr;
    if (mglRenderCreateBuffer(
            static_cast<uint64_t>(bytes),
            static_cast<uint64_t>(MTL::ResourceStorageModeShared),
            "MGL.index_emulation", &buffer) != 0 || !buffer) {
        return nullptr;
    }
    if (contents_out) {
        void *contents = nullptr;
        uint64_t length = 0;
        if (mglRenderGetBufferContents(buffer, &contents, &length) != 0 ||
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
    if (entry.buffer) deferReleaseObject(entry.buffer);
    entry.buffer = buffer;
    entry.count = requested;
    if (out_count) *out_count = index_count;
    void *prepared = entry.buffer;
    os_unfair_lock_unlock(&g_cache_lock);
    return prepared;
}

static int elementExpandCacheEligible(const Buffer *gl_buffer)
{
    return gl_buffer && gl_buffer->data.buffer_data &&
           !(gl_buffer->storage_flags & GL_MAP_PERSISTENT_BIT);
}

static MGLBufferBackend *elementExpandCacheBackend(Buffer *gl_buffer)
{
    if (!elementExpandCacheEligible(gl_buffer)) {
        return nullptr;
    }
    MGLBufferBackend *cache = gl_buffer->backend;
    if (!cache) {
        cache = static_cast<MGLBufferBackend *>(calloc(1, sizeof(MGLBufferBackend)));
        gl_buffer->backend = cache;
    }
    return cache;
}

static void *elementExpandCacheLookup(Buffer *gl_buffer, size_t index_byte_offset,
                                      size_t source_count, GLenum type,
                                      uint32_t kind, uint32_t mode,
                                      uint32_t restart_index, size_t *out_count)
{
    if (out_count) *out_count = 0u;
    MGLBufferBackend *cache = elementExpandCacheBackend(gl_buffer);
    if (!cache || !cache->element_emulation_data) {
        return nullptr;
    }
    if (cache->element_emulation_content_version != gl_buffer->content_version ||
        cache->element_emulation_byte_offset != index_byte_offset ||
        cache->element_emulation_index_count != source_count ||
        cache->element_emulation_type != static_cast<uint32_t>(type) ||
        cache->element_emulation_kind != kind ||
        cache->element_emulation_mode != mode ||
        cache->element_emulation_restart_index != restart_index) {
        return nullptr;
    }
    if (out_count) {
        *out_count = cache->element_emulation_out_index_count;
    }
    return cache->element_emulation_data;
}

static void elementExpandCacheStore(Buffer *gl_buffer, size_t index_byte_offset,
                                    size_t source_count, GLenum type,
                                    uint32_t kind, uint32_t mode,
                                    uint32_t restart_index, size_t out_index_count,
                                    void *buffer)
{
    MGLBufferBackend *cache = elementExpandCacheBackend(gl_buffer);
    if (!cache || !buffer) {
        return;
    }
    if (cache->element_emulation_data) {
        releaseObject(cache->element_emulation_data);
    }
    static_cast<NS::Object *>(buffer)->retain();
    cache->element_emulation_data = buffer;
    cache->element_emulation_content_version = gl_buffer->content_version;
    cache->element_emulation_byte_offset = index_byte_offset;
    cache->element_emulation_index_count = source_count;
    cache->element_emulation_type = static_cast<uint32_t>(type);
    cache->element_emulation_kind = kind;
    cache->element_emulation_mode = mode;
    cache->element_emulation_restart_index = restart_index;
    cache->element_emulation_out_index_count = out_index_count;
}

void *expandElement(void *device, Buffer *gl_buffer, size_t index_byte_offset,
                    uint32_t restart_index, uint32_t kind, uint32_t mode,
                    const uint8_t *source, GLenum type, size_t source_count,
                    size_t *out_count,
                    int (*expand)(const uint8_t *, uint32_t, uint32_t,
                                  uint32_t **, uint64_t *)) {
    if (out_count) *out_count = 0u;
    if (gl_buffer) {
        void *cached = elementExpandCacheLookup(
            gl_buffer, index_byte_offset, source_count, type, kind, mode,
            restart_index, out_count);
        if (cached) {
            return cached;
        }
    }
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
    if (!buffer) {
        return nullptr;
    }
    if (gl_buffer) {
        elementExpandCacheStore(gl_buffer, index_byte_offset, source_count, type,
                                kind, mode, restart_index,
                                static_cast<size_t>(count), buffer);
    }
    if (out_count) *out_count = static_cast<size_t>(count);
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
                        mglRenderExpandTriangleFanArrayIndices,
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
                        mglRenderExpandTriangleStripArrayIndices,
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
                        mglRenderExpandQuadArrayIndices, quads, out_count);
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
                        mglRenderExpandQuadArrayLineIndices, quads, out_count);
}

MGLIndexMetalHandle mglNewAdjacencyArrayIndexBuffer(
    MGLIndexMetalHandle device, GLenum mode, size_t vertex_count,
    size_t *out_count) {
    if (out_count) *out_count = 0u;
    if (!device || vertex_count > UINT32_MAX ||
        !mglRenderEmulateAdjacency((uint32_t)mode)) {
        return nullptr;
    }
    {
        os_unfair_lock_lock(&g_cache_lock);
        if (g_adjacency_array.buffer && g_adjacency_array.mode == mode &&
            g_adjacency_array.vertex_count == vertex_count) {
            if (out_count) {
                *out_count = g_adjacency_array.index_count;
            }
            void *cached = g_adjacency_array.buffer;
            os_unfair_lock_unlock(&g_cache_lock);
            return cached;
        }
        os_unfair_lock_unlock(&g_cache_lock);
    }
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (mglRenderExpandAdjacencyArrayIndices(
            (uint32_t)mode, (uint32_t)vertex_count, &expanded, &count) != 0) {
        std::free(expanded);
        return nullptr;
    }
    if (count == 0u) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (!buffer) {
        return nullptr;
    }
    os_unfair_lock_lock(&g_cache_lock);
    if (g_adjacency_array.buffer) {
        deferReleaseObject(g_adjacency_array.buffer);
    }
    g_adjacency_array.buffer = buffer;
    g_adjacency_array.mode = mode;
    g_adjacency_array.vertex_count = vertex_count;
    g_adjacency_array.index_count = static_cast<size_t>(count);
    if (out_count) *out_count = g_adjacency_array.index_count;
    void *prepared = g_adjacency_array.buffer;
    os_unfair_lock_unlock(&g_cache_lock);
    return prepared;
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
    if (mglRenderExpandLineLoopArrayIndices(
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
    size_t count, size_t *out_count, Buffer *gl_buffer,
    size_t index_byte_offset, uint32_t restart_index) {
    return expandElement(device, gl_buffer, index_byte_offset, restart_index,
                         MGL_ELEMENT_EXPAND_TRIANGLE_FAN, 0u, source, type,
                         count, out_count, mglRenderExpandTriangleFanIndices);
}

MGLIndexMetalHandle mglNewTriangleStripElementIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t count, size_t *out_count, Buffer *gl_buffer,
    size_t index_byte_offset, uint32_t restart_index) {
    return expandElement(device, gl_buffer, index_byte_offset, restart_index,
                         MGL_ELEMENT_EXPAND_TRIANGLE_STRIP, 0u, source, type,
                         count, out_count, mglRenderExpandTriangleStripIndices);
}

MGLIndexMetalHandle mglNewLineLoopElementIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t count, size_t *out_count, Buffer *gl_buffer,
    size_t index_byte_offset, uint32_t restart_index) {
    return expandElement(device, gl_buffer, index_byte_offset, restart_index,
                         MGL_ELEMENT_EXPAND_LINE_LOOP, 0u, source, type, count,
                         out_count, mglRenderExpandLineLoopIndices);
}

MGLIndexMetalHandle mglNewQuadElementIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t source_count, size_t *out_count, Buffer *gl_buffer,
    size_t index_byte_offset, uint32_t restart_index) {
    if (out_count) *out_count = 0u;
    if (gl_buffer) {
        void *cached = elementExpandCacheLookup(
            gl_buffer, index_byte_offset, source_count, type,
            MGL_ELEMENT_EXPAND_QUAD, 0u, restart_index, out_count);
        if (cached) {
            return cached;
        }
    }
    const uint32_t width = elementWidth(type);
    const size_t quads = source_count / 4u;
    if (!device || !source || width == 0u || quads == 0u || quads > UINT32_MAX) return nullptr;
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (mglRenderExpandQuadElementIndices(
            source, width, static_cast<uint32_t>(quads), &expanded, &count) != 0) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (!buffer) {
        return nullptr;
    }
    if (gl_buffer) {
        elementExpandCacheStore(gl_buffer, index_byte_offset, source_count, type,
                                MGL_ELEMENT_EXPAND_QUAD, 0u, restart_index,
                                static_cast<size_t>(count), buffer);
    }
    if (out_count) *out_count = static_cast<size_t>(count);
    return buffer;
}

MGLIndexMetalHandle mglNewQuadElementLineIndexBuffer(
    MGLIndexMetalHandle device, const uint8_t *source, GLenum type,
    size_t source_count, size_t *out_count, Buffer *gl_buffer,
    size_t index_byte_offset, uint32_t restart_index) {
    if (out_count) *out_count = 0u;
    if (gl_buffer) {
        void *cached = elementExpandCacheLookup(
            gl_buffer, index_byte_offset, source_count, type,
            MGL_ELEMENT_EXPAND_QUAD_LINE, 0u, restart_index, out_count);
        if (cached) {
            return cached;
        }
    }
    const uint32_t width = elementWidth(type);
    const size_t quads = source_count / 4u;
    if (!device || !source || width == 0u || quads == 0u || quads > UINT32_MAX) return nullptr;
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (mglRenderExpandQuadElementLineIndices(
            source, width, static_cast<uint32_t>(quads), &expanded, &count) != 0) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (!buffer) {
        return nullptr;
    }
    if (gl_buffer) {
        elementExpandCacheStore(gl_buffer, index_byte_offset, source_count, type,
                                MGL_ELEMENT_EXPAND_QUAD_LINE, 0u, restart_index,
                                static_cast<size_t>(count), buffer);
    }
    if (out_count) *out_count = static_cast<size_t>(count);
    return buffer;
}

MGLIndexMetalHandle mglNewAdjacencyElementIndexBuffer(
    MGLIndexMetalHandle device, GLenum mode, const uint8_t *source, GLenum type,
    size_t source_count, size_t *out_count, Buffer *gl_buffer,
    size_t index_byte_offset, uint32_t restart_index) {
    if (out_count) *out_count = 0u;
    const uint32_t width = elementWidth(type);
    if (!device || !source || width == 0u || source_count > UINT32_MAX ||
        !mglRenderEmulateAdjacency((uint32_t)mode)) {
        return nullptr;
    }
    if (gl_buffer) {
        void *cached = elementExpandCacheLookup(
            gl_buffer, index_byte_offset, source_count, type,
            MGL_ELEMENT_EXPAND_ADJACENCY, static_cast<uint32_t>(mode),
            restart_index, out_count);
        if (cached) {
            return cached;
        }
    }
    const size_t source_bytes = source_count * static_cast<size_t>(width);
    const uint64_t source_hash = hashIndexBytes(source, source_bytes);
    {
        os_unfair_lock_lock(&g_cache_lock);
        if (g_adjacency_element.buffer &&
            g_adjacency_element.mode == mode &&
            g_adjacency_element.type == type &&
            g_adjacency_element.source_count == source_count &&
            g_adjacency_element.source_hash == source_hash) {
            if (out_count) {
                *out_count = g_adjacency_element.index_count;
            }
            void *cached = g_adjacency_element.buffer;
            os_unfair_lock_unlock(&g_cache_lock);
            return cached;
        }
        os_unfair_lock_unlock(&g_cache_lock);
    }
    uint32_t *expanded = nullptr;
    uint64_t count = 0;
    if (mglRenderExpandAdjacencyElementIndices(
            (uint32_t)mode, source, width, (uint32_t)source_count, &expanded,
            &count) != 0) {
        std::free(expanded);
        return nullptr;
    }
    if (count == 0u) {
        std::free(expanded);
        return nullptr;
    }
    void *buffer = copyIndices(device, expanded, static_cast<size_t>(count));
    std::free(expanded);
    if (!buffer) {
        return nullptr;
    }
    if (gl_buffer) {
        elementExpandCacheStore(gl_buffer, index_byte_offset, source_count, type,
                                MGL_ELEMENT_EXPAND_ADJACENCY,
                                static_cast<uint32_t>(mode), restart_index,
                                static_cast<size_t>(count), buffer);
    }
    os_unfair_lock_lock(&g_cache_lock);
    if (g_adjacency_element.buffer) {
        deferReleaseObject(g_adjacency_element.buffer);
    }
    g_adjacency_element.buffer = buffer;
    g_adjacency_element.mode = mode;
    g_adjacency_element.type = type;
    g_adjacency_element.source_count = source_count;
    g_adjacency_element.source_hash = source_hash;
    g_adjacency_element.index_count = static_cast<size_t>(count);
    if (out_count) *out_count = g_adjacency_element.index_count;
    void *prepared = g_adjacency_element.buffer;
    os_unfair_lock_unlock(&g_cache_lock);
    return prepared;
}

MGLIndexMetalHandle mglNewUInt16IndexBufferFromUInt8(
    MGLIndexMetalHandle device, const uint8_t *source, size_t count) {
    if (!device || !source || count == 0u || count > UINT32_MAX) return nullptr;
    uint16_t *expanded = nullptr;
    uint64_t expanded_count = 0;
    if (mglRenderExpandUInt8ToUInt16(
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
        /* The CPU shadow pointer is maintained by gl_buffer->data itself;
         * its size is carried alongside it, so no kernel readability probe
         * is needed (callers bound-check offset/count against it). */
        if (bytes && count) {
            if (out_count) *out_count = count;
            return bytes;
        }
    }
    void *contents = nullptr;
    uint64_t length = 0;
    if (metal_buffer && mglRenderGetBufferContents(
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
    MGLBufferBackend *cache = gl_buffer ? gl_buffer->backend : nullptr;
    const uint64_t expand_cache_key = gl_buffer
        ? (gl_buffer->last_write_src_hash != 0u
               ? gl_buffer->last_write_src_hash
               : gl_buffer->content_version)
        : 0u;
    if (eligible && cache && cache->uint16_expanded_data &&
        cache->uint16_expanded_src_hash == expand_cache_key &&
        cache->uint16_expanded_byte_count == source_count) {
        if (io_offset) {
            if (source_offset > SIZE_MAX / sizeof(uint16_t)) return nullptr;
            *io_offset = source_offset * sizeof(uint16_t);
        }
        if (out_type) *out_type = 0u;
        return cache->uint16_expanded_data;
    }
    void *expanded = mglNewUInt16IndexBufferFromUInt8(device, source, source_count);
    if (!expanded) return nullptr;
    if (eligible && !cache) {
        cache = static_cast<MGLBufferBackend *>(
            calloc(1, sizeof(MGLBufferBackend)));
        gl_buffer->backend = cache;
    }
    if (eligible && cache) {
        if (cache->uint16_expanded_data) {
            releaseObject(cache->uint16_expanded_data);
        }
        static_cast<NS::Object *>(expanded)->retain();
        cache->uint16_expanded_data = expanded;
        cache->uint16_expanded_src_hash = expand_cache_key;
        cache->uint16_expanded_byte_count = source_count;
    }
    if (io_offset) {
        if (source_offset > SIZE_MAX / sizeof(uint16_t)) return nullptr;
        *io_offset = source_offset * sizeof(uint16_t);
    }
    if (out_type) *out_type = 0u;
    return expanded;
}

} // extern "C"
