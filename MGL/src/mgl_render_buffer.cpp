/* SPDX-License-Identifier: LGPL-3.0-only */
#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_render_pixel.h"
extern "C" {
#include "pixel_utils.h"
}
#include "mgl_program_resource.h"
#include "mgl_renderer_backend.h"
#include "mgl_air_loader.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_program_reflection.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"
#include "glm_context.h"
#include "mgl_render_internal.h"
#include "mgl_capability.h"
#include "mgl_sync.h"
#include "glm_limits.h"
#include "mgl_shader_abi.h"
#include "mgl_buffer_slots.h"
#include "mgl_buffer_plan.h"
#include "mgl_tess_domain.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <chrono>
#include <tuple>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <Block.h>
#include <objc/runtime.h>

int mglRenderBufferSubDataStorage(Buffer* buffer,
                                     size_t offset,
                                     size_t size,
                                     const void* bytes,
                                     char* err,
                                     size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    if (size == 0) return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    if (!bytes) {
        if (err && errcap) snprintf(err, errcap, "null source bytes");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    uint8_t* cpuBase = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
    }

    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    uint8_t* metalBase = static_cast<uint8_t*>(metalBuffer->contents());
    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (offset > metalLength || size > metalLength - offset) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "range offset=%zu size=%zu exceeds Metal length=%zu",
                     offset, size, metalLength);
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    if (!metalBase) {
        if (err && errcap) snprintf(err, errcap, "Metal buffer has no contents");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    MTL::Buffer* bufferBeforeSnapshot = metalBuffer;
    if (cpuBase && cpuBase != metalBase) {
        memmove(cpuBase + offset, bytes, size);
        void* snapshotBuffer = nullptr;
        if (mglRenderSnapshotSharedDirtyBuffer(
                buffer, &snapshotBuffer, err, errcap) != 0) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
        metalBase = metalBuffer
            ? static_cast<uint8_t*>(metalBuffer->contents())
            : nullptr;
        if (!metalBuffer || !metalBase) {
            if (err && errcap) snprintf(err, errcap, "snapshot has no contents");
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
    }

    if (metalBuffer == bufferBeforeSnapshot) {
        memcpy(metalBase + offset, bytes, size);
        if (metalBuffer->storageMode() == MTL::StorageModeManaged) {
            metalBuffer->didModifyRange(NS::Range::Make(offset, size));
        }
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

void mglRenderBufferSubData(GLMContext glm_ctx,
                               Buffer* buffer,
                               size_t offset,
                               size_t size,
                               const void* bytes) {
    char error[256] = {};
    int result = mglRenderBufferSubDataStorage(
        buffer, offset, size, bytes, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer subdata failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderMapBufferStorage(Buffer* buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map,
                                 void** mapped_out,
                                 char* err,
                                 size_t errcap) {
    if (mapped_out) *mapped_out = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!buffer || !mapped_out) {
        if (err && errcap) snprintf(err, errcap, "bad arguments");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
    }

    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (offset > metalLength) {
        if (err && errcap) {
            snprintf(err, errcap, "offset=%zu beyond Metal length=%zu",
                     offset, metalLength);
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    const size_t safeLength = std::min(size, metalLength - offset);
    uint8_t* metalBase = static_cast<uint8_t*>(metalBuffer->contents());
    uint8_t* cpuBase = nullptr;
    if (buffer->data.buffer_data >= 0x1000u) {
        cpuBase = reinterpret_cast<uint8_t*>(
            static_cast<uintptr_t>(buffer->data.buffer_data));
    }

    if (map) {
        const bool reads = access == GL_READ_ONLY || access == GL_READ_WRITE ||
                           (access & GL_MAP_READ_BIT) != 0;
        if (cpuBase) {
            uint8_t* cpuPointer = cpuBase + offset;
            if (reads && metalBase && metalBase != cpuBase && safeLength > 0 &&
                !buffer->cpu_shadow_pending) {
                memcpy(cpuPointer, metalBase + offset, safeLength);
            }
            *mapped_out = cpuPointer;
        } else {
            *mapped_out = metalBase ? metalBase + offset : nullptr;
        }
        return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    }

    if (!cpuBase &&
        metalBuffer->storageMode() == MTL::StorageModeManaged) {
        metalBuffer->didModifyRange(NS::Range::Make(offset, safeLength));
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

void* mglRenderMapUnmapBuffer(GLMContext glm_ctx,
                                 Buffer* buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map) {
    void* mapped = nullptr;
    char error[256] = {};
    int result = mglRenderMapBufferStorage(
        buffer, offset, size, access, map, &mapped, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) return mapped;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer map failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
    return nullptr;
}

void mglRenderReadBackBuffer(GLMContext glm_ctx,
                                Buffer* buffer,
                                size_t offset,
                                size_t size) {
    (void)glm_ctx;
    if (!buffer || size == 0 || buffer->cpu_shadow_pending ||
        !buffer->data.mtl_data) {
        return;
    }
    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    if (metalBuffer->storageMode() != MTL::StorageModeShared) return;

    uint8_t* metalBase = static_cast<uint8_t*>(metalBuffer->contents());
    uint8_t* cpuBase = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!metalBase || !cpuBase || metalBase == cpuBase) return;

    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (offset >= metalLength) return;
    size_t safeLength = std::min(size, metalLength - offset);
    const size_t shadowLength = buffer->data.buffer_size;
    if (shadowLength > 0) {
        if (offset >= shadowLength) return;
        safeLength = std::min(safeLength, shadowLength - offset);
    }
    memcpy(cpuBase + offset, metalBase + offset, safeLength);
}

int mglRenderFlushBufferRangeStorage(Buffer* buffer,
                                         intptr_t offset,
                                         intptr_t length,
                                         char* err,
                                         size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer || offset < 0 || length < 0) {
        if (err && errcap) snprintf(err, errcap, "bad buffer or signed range");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    if (length == 0) return MGL_RENDER_BUFFER_OPERATION_HANDLED;

    bool created = false;
    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        created = true;
    }

    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    const size_t safeOffset = static_cast<size_t>(offset);
    const size_t safeLength = static_cast<size_t>(length);
    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (safeOffset > metalLength || safeLength > metalLength - safeOffset) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "range offset=%zu length=%zu exceeds Metal length=%zu",
                     safeOffset, safeLength, metalLength);
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    if (!created) {
        void* snapshotBuffer = nullptr;
        if (mglRenderSnapshotSharedBufferRange(
                buffer, safeOffset, safeLength, &snapshotBuffer,
                err, errcap) != 0) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
    }
    if (metalBuffer &&
        metalBuffer->storageMode() == MTL::StorageModeManaged) {
        metalBuffer->didModifyRange(
            NS::Range::Make(safeOffset, safeLength));
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

void mglRenderFlushBufferRange(GLMContext glm_ctx,
                                  Buffer* buffer,
                                  intptr_t offset,
                                  intptr_t length) {
    char error[256] = {};
    int result = mglRenderFlushBufferRangeStorage(
        buffer, offset, length, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer range flush failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderGetBufferContents(void *buffer,
                                  void **contents_out,
                                  uint64_t *length_out) {
    if (contents_out) *contents_out = nullptr;
    if (length_out) *length_out = 0;
    MTL::Buffer *object = static_cast<MTL::Buffer *>(buffer);
    if (!object || !contents_out || !length_out) return -1;
    *contents_out = object->contents();
    *length_out = static_cast<uint64_t>(object->length());
    return *contents_out ? 0 : -1;
}

int mglRenderGetBufferInfo(const void *buffer,
                              MGLRenderBufferInfo *info_out) {
    if (info_out) *info_out = {};
    const MTL::Buffer *object = static_cast<const MTL::Buffer *>(buffer);
    if (!object || !info_out) return -1;
    info_out->length = static_cast<uint64_t>(object->length());
    return 0;
}

int mglRenderAddBufferDebugMarker(void *buffer,
                                     const char *marker,
                                     uint64_t location,
                                     uint64_t length) {
    MTL::Buffer *object = static_cast<MTL::Buffer *>(buffer);
    if (!object || !marker || location > object->length() ||
        length > object->length() - location) {
        return -1;
    }
    object->addDebugMarker(
        NS::String::string(marker, NS::UTF8StringEncoding),
        NS::Range(location, length));
    return 0;
}

extern "C"
int mglRenderFillDefaultTessFactorBuffer(
    void* dst, uint64_t dst_bytes,
    const float* outer_levels, const float* inner_levels,
    uint32_t patch_count) {
    const uint64_t stride = MGL_AIR_TESS_FACTOR_RECORD_BYTES;
    if (!dst || !outer_levels || !inner_levels || patch_count == 0u ||
        dst_bytes < (uint64_t)patch_count * stride) {
        return -1;
    }
    uint8_t* base = (uint8_t*)dst;
    for (uint32_t patch = 0u; patch < patch_count; patch++) {
        __fp16* halfs = (__fp16*)(base + (uint64_t)patch * stride);
        float* exact = (float*)(base + (uint64_t)patch * stride +
                                MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET);
        for (uint32_t i = 0u; i < 4u; i++) {
            halfs[i] = (__fp16)outer_levels[i];
            exact[i] = outer_levels[i];
        }
        for (uint32_t i = 0u; i < 2u; i++) {
            halfs[4u + i] = (__fp16)inner_levels[i];
            exact[4u + i] = inner_levels[i];
        }
    }
    return 0;
}

int mglRenderHasDirtyBufferBit(uint32_t dirty_bits) {
    return (dirty_bits & DIRTY_BUFFER) ? 1 : 0;
}

int mglRenderDefaultReadBufferIndex(uint32_t read_buffer, uint32_t *out) {
    uint32_t idx = 0u;
    int known = 1;
    switch (read_buffer) {
    case GL_FRONT:
    case GL_BACK:
        idx = 0u; /* _FRONT */
        break;
    case GL_FRONT_LEFT:
    case GL_BACK_LEFT:
    case GL_LEFT:
        idx = 2u; /* _FRONT_LEFT */
        break;
    case GL_FRONT_RIGHT:
    case GL_BACK_RIGHT:
    case GL_RIGHT:
        idx = 3u; /* _FRONT_RIGHT */
        break;
    default:
        known = 0;
        idx = 0u;
        break;
    }
    if (out) {
        *out = idx;
    }
    return known;
}

int mglRenderFBOReadBufferValid(uint32_t read_buffer, uint32_t max_color,
                                uint32_t max_attach) {
    if (read_buffer == GL_NONE) {
        return 0;
    }
    if (read_buffer < GL_COLOR_ATTACHMENT0) {
        return 0;
    }
    if (read_buffer >= GL_COLOR_ATTACHMENT0 + max_color) {
        return 0;
    }
    if (read_buffer >= GL_COLOR_ATTACHMENT0 + max_attach) {
        return 0;
    }
    return 1;
}

int mglRenderDefaultDrawBufferIndex(uint32_t draw_buffer, uint32_t *out) {
    uint32_t idx = 0u;
    int known = 1;
    if (mglRenderDefaultReadBufferIndex(draw_buffer, &idx)) {
        if (out) {
            *out = idx;
        }
        return 1;
    }
    switch (draw_buffer) {
    case GL_FRONT_AND_BACK:
    case GL_COLOR_ATTACHMENT0:
    case GL_NONE:
        idx = 0u; /* _FRONT */
        break;
    default:
        known = 0;
        idx = 0u;
        break;
    }
    if (out) {
        *out = idx;
    }
    return known;
}

uint32_t mglRenderEmptyDrawBuffer(void) {
    return GL_NONE;
}

uint32_t mglRenderDefaultFrontBuffer(void) {
    return GL_FRONT;
}

int mglRenderShouldPresentDrawBuffer(uint32_t draw_buffer) {
    return !mglRenderDrawBufferIsNone(draw_buffer);
}

int mglRenderDrawBufferIsColorAttachment(uint32_t draw_buffer, uint32_t max,
                                         uint32_t *out_index) {
    if (draw_buffer >= GL_COLOR_ATTACHMENT0 &&
        draw_buffer < GL_COLOR_ATTACHMENT0 + max &&
        draw_buffer < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        if (out_index) {
            *out_index = draw_buffer - GL_COLOR_ATTACHMENT0;
        }
        return 1;
    }
    return 0;
}

int mglRenderDrawBufferIsDefaultFBOCompat(uint32_t draw_buffer) {
    uint32_t idx = 0u;
    if (mglRenderDefaultReadBufferIndex(draw_buffer, &idx)) {
        return 1;
    }
    return draw_buffer == GL_FRONT_AND_BACK ? 1 : 0;
}

int mglRenderDefaultDrawBufferIsFront(uint32_t mgl_drawbuffer) {
    return mgl_drawbuffer == 0u /* _FRONT */ ? 1 : 0;
}

int mglRenderDefaultDrawBufferIsOffscreen(uint32_t mgl_drawbuffer,
                                          uint32_t max_draw_buffers) {
    return !mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer) &&
                   mgl_drawbuffer < max_draw_buffers
               ? 1
               : 0;
}

int mglRenderBufferHasMapWriteBit(uint32_t access_flags) {
    return (access_flags & GL_MAP_WRITE_BIT) != 0 ? 1 : 0;
}

int mglRenderBufferHasCPUDirty(uint32_t dirty_bits) {
    return (dirty_bits & (DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR)) ? 1 : 0;
}

int mglRenderBufferNeedsCPUUpload(int64_t size, uint32_t dirty_bits) {
    return size > 0 && mglRenderBufferHasCPUDirty(dirty_bits) ? 1 : 0;
}

int mglRenderBufferSlotInRange(int32_t slot, uint32_t max_slots) {
    return slot >= 0 && (uint32_t)slot < max_slots ? 1 : 0;
}

int mglRenderResolveMappedBufferSlot(int has_metal_binding,
                                     int32_t metal_binding_index,
                                     int32_t buffer_base_index,
                                     uint32_t max_slots, uint32_t *out_slot) {
    const int32_t slot =
        has_metal_binding ? metal_binding_index : buffer_base_index;
    if (!mglRenderBufferSlotInRange(slot, max_slots)) {
        return 0;
    }
    if (out_slot) {
        *out_slot = (uint32_t)slot;
    }
    return 1;
}

int mglRenderBufferMapOffsetValid(int64_t offset) {
    return offset >= 0 ? 1 : 0;
}

int mglRenderBufferSizeValid(int64_t size) {
    return size >= 0 ? 1 : 0;
}

uint64_t mglRenderBufferSizeOrZero(int64_t size) {
    return size >= 0 ? (uint64_t)size : 0u;
}

int mglRenderBufferPlanIsStructPacked(uint32_t flags) {
    return (flags & MGL_BP_FLAG_STRUCT_PACKED) != 0u ? 1 : 0;
}

int mglRenderBufferPlanAllowFallback(int has_fallback, uint32_t flags) {
    return has_fallback && (flags & MGL_BP_FLAG_ALLOW_FALLBACK) != 0u ? 1 : 0;
}

int mglRenderAllowGlobalBufferFallback(int has_fallback, int spvc_type,
                                       uint32_t flags) {
    if (!has_fallback) {
        return 0;
    }
    if (spvc_type != _UNIFORM_CONSTANT_RES) {
        return 1;
    }
    return (flags & MGL_BP_FLAG_ALLOW_FALLBACK) != 0u ? 1 : 0;
}

int mglRenderIsUniformBufferResource(int spvc_type) {
    return spvc_type == _UNIFORM_BUFFER_RES ? 1 : 0;
}

extern "C"
int mglRenderBufferShadowUploadRange(
    int gpu_write_target, int64_t written_min, int64_t written_max,
    uint64_t limit, uint64_t* out_offset, uint64_t* out_length) {
    if (!out_offset || !out_length) return -1;
    uint64_t offset = 0;
    uint64_t length = limit;
    if (gpu_write_target) {
        if (written_min < 0 || written_max <= written_min) {
            return -1;
        }
        const uint64_t min = (uint64_t)written_min;
        const uint64_t max = (uint64_t)written_max;
        offset = min < limit ? min : limit;
        const uint64_t clampedMax = max < limit ? max : limit;
        length = clampedMax - offset;
    }
    if (length == 0) {
        return -1;
    }
    *out_offset = offset;
    *out_length = length;
    return 0;
}

int mglRenderSetComputeBuffer(void* compute_encoder,
                                 void* buffer,
                                 uint64_t offset,
                                 uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setBuffer(static_cast<MTL::Buffer*>(buffer),
                       static_cast<NS::UInteger>(offset), index);
    return 0;
}

void mglRenderMarkBufferCPUWrite(Buffer *buf, int64_t offset, int64_t size) {
    if (!buf) {
        return;
    }
    buf->ever_written = GL_TRUE;
    buf->has_initialized_data = GL_TRUE;
    buf->cpu_shadow_pending = GL_TRUE;
    buf->gpu_write_target = GL_FALSE;
    buf->data.dirty_bits |= DIRTY_BUFFER_DATA;
    buf->last_init_source = kInitMapWrite;
    buf->last_write_offset = (GLintptr)offset;
    buf->last_write_size = (GLsizeiptr)size;
    if (size > 0 && offset >= 0) {
        const GLintptr write_end = (GLintptr)(offset + size);
        if (buf->written_min < 0 || (GLintptr)offset < buf->written_min) {
            buf->written_min = (GLintptr)offset;
        }
        if (buf->written_max < 0 || write_end > buf->written_max) {
            buf->written_max = write_end;
        }
    }
}

int mglRenderSetRenderBuffer(void* render_encoder,
                                void* buffer,
                                uint64_t offset,
                                uint32_t stage,
                                uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) return -1;
    MTL::Buffer* resource = static_cast<MTL::Buffer*>(buffer);
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexBuffer(resource, static_cast<NS::UInteger>(offset),
                                 index);
    } else {
        encoder->setFragmentBuffer(resource,
                                   static_cast<NS::UInteger>(offset), index);
    }
    return 0;
}

int mglRenderSetTessellationFactorBuffer(void* render_encoder,
                                            void* buffer,
                                            uint64_t offset,
                                            uint64_t instance_stride) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* factors = static_cast<MTL::Buffer*>(buffer);
    if (!encoder || !factors) return -1;
    encoder->setTessellationFactorBuffer(
        factors, static_cast<NS::UInteger>(offset),
        static_cast<NS::UInteger>(instance_stride));
    return 0;
}

void mglRenderReleaseBufferMetalData(GLMContext glm_ctx, Buffer* buffer) {
    if (!buffer || !buffer->data.mtl_data) return;
    (void)glm_ctx;
    mgl::releaseBridgedObject(&buffer->data.mtl_data);
}

void mglRenderReleaseBufferCowPool(Buffer* buffer) {
    if (!buffer || !buffer->mtl_cow_pool) return;
    mgl::BufferCowPool* pool =
        static_cast<mgl::BufferCowPool*>(buffer->mtl_cow_pool);
    buffer->mtl_cow_pool = nullptr;
    delete pool;
}

Buffer* mglRenderAcquirePackedStructBuffer(const void* data,
                                               size_t size,
                                               char* err,
                                               size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!data || size == 0) {
        if (err && errcap) snprintf(err, errcap, "invalid packed struct data");
        return nullptr;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "renderer is not initialized");
        return nullptr;
    }

    const size_t paddedSize =
        std::max(size, mgl::kMinimumStageBindingSize);
    MTL::Buffer* metalBuffer = renderer.device->newBuffer(
        static_cast<NS::UInteger>(paddedSize),
        MTL::ResourceStorageModeShared);
    if (!metalBuffer || !metalBuffer->contents()) {
        if (metalBuffer) metalBuffer->release();
        if (err && errcap) {
            snprintf(err, errcap,
                     "packed struct Metal buffer creation failed size=%zu",
                     paddedSize);
        }
        return nullptr;
    }
    std::memcpy(metalBuffer->contents(), data, size);
    if (paddedSize > size) {
        std::memset(static_cast<uint8_t*>(metalBuffer->contents()) + size,
                    0, paddedSize - size);
    }

    const size_t index = renderer.packedStructBufferIndex;
    Buffer*& slot = renderer.packedStructBuffers[index];
    if (!slot) {
        slot = static_cast<Buffer*>(std::calloc(1, sizeof(Buffer)));
        if (!slot) {
            metalBuffer->release();
            if (err && errcap) {
                snprintf(err, errcap,
                         "packed struct Buffer allocation failed");
            }
            return nullptr;
        }
        slot->name = 0xF0000000u | static_cast<GLuint>(index);
        slot->target = GL_UNIFORM_BUFFER;
        slot->usage = GL_STATIC_DRAW;
        slot->written_min = -1;
        slot->written_max = -1;
        slot->transient_batch_buffer = GL_TRUE;
    }

    mgl::releaseBridgedObject(&slot->data.mtl_data);
    mglMetalCountCreate(mgl::kMetalKindBuffer);
    slot->data.mtl_data = metalBuffer;
    slot->size = static_cast<GLsizeiptr>(paddedSize);
    slot->data.buffer_data = 0;
    slot->data.buffer_size = paddedSize;
    slot->data.dirty_bits = 0;
    slot->data.mtl_owns_buffer_data = GL_FALSE;
    slot->has_initialized_data = GL_TRUE;
    slot->ever_written = GL_TRUE;

    renderer.packedStructBufferIndex =
        (index + 1) % mgl::kPackedStructBufferCapacity;
    return slot;
}

uint64_t mglRenderAdvanceBufferGeneration(void) {
    return mgl::gBufferFrameGeneration.fetch_add(
               1, std::memory_order_acq_rel) + 1;
}

void mglRenderRecordBufferGenerationCompleted(uint64_t generation) {
    uint64_t completed =
        mgl::gBufferCompletedGeneration.load(std::memory_order_relaxed);
    while (generation > completed &&
           !mgl::gBufferCompletedGeneration.compare_exchange_weak(
               completed, generation, std::memory_order_release,
               std::memory_order_relaxed)) {
    }
}

uint64_t mglRenderCompletedBufferGeneration(void) {
    return mgl::gBufferCompletedGeneration.load(std::memory_order_acquire);
}

void mglRenderNoteBufferEncoded(Buffer* buffer) {
    if (!buffer || !buffer->data.mtl_data) return;
    mgl::BufferCowPool* pool = mgl::bufferCowPool(buffer, false);
    if (!pool) return;
    MTL::Buffer* current =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    const uint64_t generation =
        mgl::gBufferFrameGeneration.load(std::memory_order_acquire);
    for (mgl::BufferCowSlot& slot : pool->slots) {
        if (slot.buffer == current) {
            slot.lastUseGeneration = generation;
            return;
        }
    }
}

int mglRenderSnapshotSharedDirtyBuffer(Buffer* buffer,
                                          void** metal_buffer_out,
                                          char* err,
                                          size_t errcap) {
    if (metal_buffer_out) *metal_buffer_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Device* device = mgl::renderer().device;
    if (!buffer || !metal_buffer_out || !device) {
        if (err && errcap) snprintf(err, errcap, "bad arguments or renderer");
        return -1;
    }

    MTL::Buffer* current =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    *metal_buffer_out = current;
    uint8_t* cpuData = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!current || buffer->transient_batch_buffer ||
        current->storageMode() != MTL::StorageModeShared || !cpuData ||
        (buffer->storage_flags & GL_CLIENT_STORAGE_BIT) != 0 ||
        cpuData == current->contents()) {
        return 0;
    }

    const size_t metalLength = static_cast<size_t>(current->length());
    size_t snapshotLength = metalLength;
    if (buffer->data.buffer_size > 0) {
        snapshotLength = std::min(snapshotLength, buffer->data.buffer_size);
    }
    if (snapshotLength == 0) return 0;

    MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
    if (current->cpuCacheMode() == MTL::CPUCacheModeWriteCombined) {
        options = static_cast<MTL::ResourceOptions>(
            options | MTL::ResourceCPUCacheModeWriteCombined);
    }
    mgl::BufferCowSnapshot snapshot = mgl::takeBufferCowSnapshot(
        device, current, metalLength, options, buffer);
    if (!snapshot.buffer || !snapshot.buffer->contents() ||
        !current->contents()) {
        if (snapshot.buffer && !snapshot.poolOwnsReference) {
            snapshot.buffer->release();
        }
        if (err && errcap) snprintf(err, errcap, "snapshot allocation failed");
        return -1;
    }

    uint8_t* snapshotData =
        static_cast<uint8_t*>(snapshot.buffer->contents());
    if (buffer->gpu_write_target) {
        memcpy(snapshotData, current->contents(), metalLength);
        size_t uploadOffset = 0;
        size_t uploadLength = 0;
        if (mgl::bufferShadowUploadRange(
                buffer, snapshotLength, &uploadOffset, &uploadLength)) {
            memcpy(snapshotData + uploadOffset, cpuData + uploadOffset,
                   uploadLength);
        }
    } else {
        memcpy(snapshotData, cpuData, snapshotLength);
        if (snapshotLength < metalLength) {
            memset(snapshotData + snapshotLength, 0,
                   metalLength - snapshotLength);
        }
    }

    mgl::installBufferCowSnapshot(buffer, snapshot);
    mglRenderNoteBufferEncoded(buffer);
    mglRecordBufferCowSnapshot(metalLength);
    *metal_buffer_out = buffer->data.mtl_data;
    return 0;
}

int mglRenderSnapshotSharedBufferRange(Buffer* buffer,
                                          size_t offset,
                                          size_t length,
                                          void** metal_buffer_out,
                                          char* err,
                                          size_t errcap) {
    if (metal_buffer_out) *metal_buffer_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Device* device = mgl::renderer().device;
    if (!buffer || !metal_buffer_out || !device) {
        if (err && errcap) snprintf(err, errcap, "bad arguments or renderer");
        return -1;
    }

    MTL::Buffer* current =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    *metal_buffer_out = current;
    uint8_t* cpuData = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!current) return 0;
    const size_t metalLength = static_cast<size_t>(current->length());
    if (offset > metalLength || length > metalLength - offset) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "range offset=%zu length=%zu exceeds Metal length=%zu",
                     offset, length, metalLength);
        }
        return -1;
    }
    if (buffer->transient_batch_buffer ||
        current->storageMode() != MTL::StorageModeShared || !cpuData ||
        (buffer->storage_flags & GL_CLIENT_STORAGE_BIT) != 0 ||
        cpuData == current->contents()) {
        return 0;
    }

    MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
    if (current->cpuCacheMode() == MTL::CPUCacheModeWriteCombined) {
        options = static_cast<MTL::ResourceOptions>(
            options | MTL::ResourceCPUCacheModeWriteCombined);
    }
    mgl::BufferCowSnapshot snapshot = mgl::takeBufferCowSnapshot(
        device, current, metalLength, options, buffer);
    if (!snapshot.buffer || !snapshot.buffer->contents() ||
        !current->contents()) {
        if (snapshot.buffer && !snapshot.poolOwnsReference) {
            snapshot.buffer->release();
        }
        if (err && errcap) snprintf(err, errcap, "snapshot allocation failed");
        return -1;
    }

    memcpy(snapshot.buffer->contents(), current->contents(), metalLength);
    memcpy(static_cast<uint8_t*>(snapshot.buffer->contents()) + offset,
           cpuData + offset, length);
    mgl::installBufferCowSnapshot(buffer, snapshot);
    mglRenderNoteBufferEncoded(buffer);
    mglRecordBufferCowSnapshot(metalLength);
    *metal_buffer_out = buffer->data.mtl_data;
    return 0;
}

int mglRenderUpdateDirtyBuffer(Buffer* buffer,
                                  char* err,
                                  size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    auto ensureMetalBuffer = [&]() -> int {
        if (buffer->data.mtl_data) {
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }
        const int bindResult =
            mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    };

    if (buffer->plain_uniform_slot && !buffer->data.mtl_data &&
        buffer->data.buffer_data && buffer->size > 0 &&
        buffer->size <= 4096) {
        buffer->data.dirty_bits &=
            ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
        return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    }

    if (buffer->size < 4096) {
        if ((buffer->data.dirty_bits & DIRTY_BUFFER_ADDR) &&
            !buffer->data.mtl_data) {
            const int result = ensureMetalBuffer();
            if (result != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
                return result;
            }
        }

        if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
            buffer->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }

        const int bindResult = ensureMetalBuffer();
        if (bindResult != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
            return bindResult;
        }

        MTL::Buffer* metalBuffer =
            static_cast<MTL::Buffer*>(buffer->data.mtl_data);
        MTL::Buffer* bufferBeforeSnapshot = metalBuffer;
        void* snapshotBuffer = nullptr;
        if (mglRenderSnapshotSharedDirtyBuffer(
                buffer, &snapshotBuffer, err, errcap) != 0) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
        if (!metalBuffer) {
            if (err && errcap) snprintf(err, errcap, "missing Metal buffer");
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }

        const size_t metalLength = static_cast<size_t>(metalBuffer->length());
        size_t copyLength = buffer->size > 0
            ? std::min(static_cast<size_t>(buffer->size), metalLength)
            : 0;
        if (buffer->data.buffer_size > 0) {
            copyLength = std::min(copyLength, buffer->data.buffer_size);
        }
        uint8_t* cpuData = buffer->data.buffer_data >= 0x1000u
            ? reinterpret_cast<uint8_t*>(
                  static_cast<uintptr_t>(buffer->data.buffer_data))
            : nullptr;
        uint8_t* metalData =
            static_cast<uint8_t*>(metalBuffer->contents());

        if (metalBuffer == bufferBeforeSnapshot && cpuData && metalData &&
            copyLength > 0) {
            size_t uploadOffset = 0;
            size_t uploadLength = 0;
            if (mgl::bufferShadowUploadRange(
                    buffer, copyLength, &uploadOffset, &uploadLength)) {
                if (cpuData != metalData) {
                    memmove(metalData + uploadOffset,
                            cpuData + uploadOffset, uploadLength);
                }
                if (metalBuffer->storageMode() == MTL::StorageModeManaged) {
                    metalBuffer->didModifyRange(
                        NS::Range::Make(uploadOffset, uploadLength));
                }
            }
        } else if (metalBuffer == bufferBeforeSnapshot && metalData &&
                   copyLength > 0) {
            size_t modifyOffset = 0;
            size_t modifyLength = copyLength;
            if (buffer->mapped_length > 0 && buffer->mapped_offset >= 0 &&
                static_cast<size_t>(buffer->mapped_offset) < metalLength) {
                modifyOffset = static_cast<size_t>(buffer->mapped_offset);
                modifyLength = std::min(
                    static_cast<size_t>(buffer->mapped_length),
                    metalLength - modifyOffset);
            }
            if (modifyLength > 0 &&
                metalBuffer->storageMode() == MTL::StorageModeManaged) {
                metalBuffer->didModifyRange(
                    NS::Range::Make(modifyOffset, modifyLength));
            }
        }

        if ((buffer->access_flags & GL_MAP_COHERENT_BIT) != 0) {
            buffer->data.dirty_bits = DIRTY_BUFFER_DATA;
        } else {
            buffer->data.dirty_bits &=
                ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
            buffer->cpu_shadow_pending = GL_FALSE;
            /* CPU range was absorbed into Metal; do not re-apply it on a
             * later gpu_write_target CoW after shaders overwrite Metal. */
            buffer->written_min = -1;
            buffer->written_max = -1;
        }
        return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    }

    if ((buffer->data.dirty_bits & DIRTY_BUFFER_ADDR) != 0) {
        const int bindResult = ensureMetalBuffer();
        if (bindResult != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
            return bindResult;
        }
        if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
            buffer->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }
    }

    if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "buffer %u has no dirty CPU or Metal backing",
                     static_cast<unsigned>(buffer->name));
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    const int bindResult = ensureMetalBuffer();
    if (bindResult != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
        return bindResult;
    }
    void* snapshotBuffer = nullptr;
    if (mglRenderSnapshotSharedDirtyBuffer(
            buffer, &snapshotBuffer, err, errcap) != 0) {
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    MTL::Buffer* metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
    if (!metalBuffer) {
        if (err && errcap) snprintf(err, errcap, "missing Metal buffer");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    const bool coherentMapped =
        (buffer->access_flags & GL_MAP_COHERENT_BIT) != 0;
    if (coherentMapped) {
        size_t modifyOffset = 0;
        size_t modifyLength = metalLength;
        if (buffer->mapped_length > 0 && buffer->mapped_offset >= 0 &&
            static_cast<size_t>(buffer->mapped_offset) < metalLength) {
            modifyOffset = static_cast<size_t>(buffer->mapped_offset);
            modifyLength = std::min(
                static_cast<size_t>(buffer->mapped_length),
                metalLength - modifyOffset);
        }
        if (modifyLength > 0 &&
            metalBuffer->storageMode() == MTL::StorageModeManaged) {
            metalBuffer->didModifyRange(
                NS::Range::Make(modifyOffset, modifyLength));
        }
        buffer->data.dirty_bits = DIRTY_BUFFER_DATA;
    } else {
        size_t modifyLength = metalLength;
        if (buffer->data.buffer_size > 0) {
            modifyLength = std::min(modifyLength, buffer->data.buffer_size);
        }
        if (modifyLength > 0 &&
            metalBuffer->storageMode() == MTL::StorageModeManaged) {
            metalBuffer->didModifyRange(NS::Range::Make(0, modifyLength));
        }
        buffer->data.dirty_bits = 0;
        buffer->cpu_shadow_pending = GL_FALSE;
        /* Same as the <4096 path: CPU range absorbed; avoid stale overlay. */
        buffer->written_min = -1;
        buffer->written_max = -1;
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

int mglRenderConvertVertexBuffer(
    Buffer* sourceBuffer,
    const MGLRenderVertexConversion* conversion,
    uint64_t* convertedStrideOut,
    void** convertedBufferOut,
    char* err,
    size_t errcap) {
    if (convertedStrideOut) *convertedStrideOut = 0;
    if (convertedBufferOut) *convertedBufferOut = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!sourceBuffer || !conversion || !convertedStrideOut ||
        !convertedBufferOut) {
        if (err && errcap) snprintf(err, errcap, "bad arguments");
        return -1;
    }
    if (conversion->kind > MGL_RENDER_VERTEX_INTEGER_TO_32) {
        if (err && errcap) {
            snprintf(err, errcap, "unknown conversion kind=%u",
                     conversion->kind);
        }
        return -1;
    }
    if (conversion->binding_offset < 0 ||
        conversion->relative_offset < 0) {
        if (err && errcap) snprintf(err, errcap, "negative vertex offset");
        return -1;
    }

    const uint8_t* sourceBytes = nullptr;
    size_t sourceSize = 0;
    if (!mgl::vertexConversionSource(
            sourceBuffer, &sourceBytes, &sourceSize) ||
        !sourceBytes || sourceSize == 0) {
        if (err && errcap) snprintf(err, errcap, "missing source bytes");
        return -1;
    }
    const size_t bindingOffset =
        static_cast<size_t>(conversion->binding_offset);
    const size_t relativeOffset =
        static_cast<size_t>(conversion->relative_offset);
    if (bindingOffset >= sourceSize) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "binding offset=%zu exceeds source size=%zu",
                     bindingOffset, sourceSize);
        }
        return -1;
    }

    const uint32_t kind = conversion->kind;
    uint32_t componentCount = conversion->component_count;
    size_t sourceComponentSize = 0;
    size_t defaultStride = 0;
    size_t minimumConvertedStride = 0;
    switch (kind) {
        case MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize = sizeof(double);
            defaultStride = componentCount * sizeof(double);
            minimumConvertedStride = componentCount * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_INT_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            if (conversion->source_type != GL_INT &&
                conversion->source_type != GL_UNSIGNED_INT) {
                if (err && errcap) snprintf(err, errcap, "invalid int source type");
                return -1;
            }
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = componentCount * sizeof(uint32_t);
            minimumConvertedStride = 0;
            break;
        case MGL_RENDER_VERTEX_FIXED_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize = sizeof(int32_t);
            defaultStride = componentCount * sizeof(int32_t);
            minimumConvertedStride = componentCount * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT:
            componentCount = 4;
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = sizeof(uint32_t);
            minimumConvertedStride = 4u * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT:
            componentCount = 3;
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = sizeof(uint32_t);
            minimumConvertedStride = 3u * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_INTEGER_TO_32:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize =
                mgl::vertexComponentSize(conversion->source_type);
            if (sourceComponentSize == 0 || sourceComponentSize > 4) {
                if (err && errcap) {
                    snprintf(err, errcap, "invalid integer source type");
                }
                return -1;
            }
            defaultStride = componentCount * sourceComponentSize;
            minimumConvertedStride = componentCount * sizeof(uint32_t);
            break;
        default:
            return -1;
    }

    {
        if (conversion->stride >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            if (err && errcap) snprintf(err, errcap, "vertex stride overflow");
            return -1;
        }
        const size_t originalStride = conversion->stride > 0
            ? static_cast<size_t>(conversion->stride)
            : defaultStride;
        if (originalStride == 0) {
            if (err && errcap) snprintf(err, errcap, "zero vertex stride");
            return -1;
        }

        size_t convertedStrideBase = originalStride;
        if (kind == MGL_RENDER_VERTEX_INTEGER_TO_32) {
            convertedStrideBase = minimumConvertedStride;
        } else if (minimumConvertedStride > convertedStrideBase) {
            convertedStrideBase = minimumConvertedStride;
        }
        size_t convertedStride = 0;
        if (!mgl::alignVertexStride(convertedStrideBase, &convertedStride)) {
            if (err && errcap) snprintf(err, errcap, "converted stride overflow");
            return -1;
        }

        const size_t copyLength = sourceSize - bindingOffset;
        const size_t vertexCount =
            copyLength / originalStride +
            ((copyLength % originalStride) != 0 ? 1u : 0u);
        if (vertexCount == 0 ||
            vertexCount > std::numeric_limits<size_t>::max() /
                              convertedStride) {
            if (err && errcap) snprintf(err, errcap, "converted size overflow");
            return -1;
        }
        const size_t convertedLength = vertexCount * convertedStride;
        const uint8_t* sourceBase = sourceBytes + bindingOffset;
        const uint64_t sourceHash =
            mgl::hashVertexBytes(sourceBase, copyLength);
        mgl::ConvertedVertexBufferKey key = {};
        key.sourceHash = sourceHash;
        key.copyLength = copyLength;
        key.originalStride = originalStride;
        key.convertedStride = convertedStride;
        key.bindingOffset = conversion->binding_offset;
        key.relativeOffset = conversion->relative_offset;
        key.sourceName = sourceBuffer->name;
        key.kind = kind;
        key.componentCount = componentCount;
        key.sourceType = conversion->source_type;
        key.normalized = conversion->normalized;
        key.destinationSigned = conversion->destination_signed;

        mgl::Renderer& renderer = mgl::renderer();
        {
            std::lock_guard<std::mutex> lock(renderer.mutex);
            if (!renderer.device) {
                if (err && errcap) snprintf(err, errcap, "renderer is not initialized");
                return -1;
            }
            auto found = renderer.convertedVertexBuffers.find(key);
            if (found != renderer.convertedVertexBuffers.end() &&
                found->second) {
                found->second->retain();
                *convertedStrideOut = convertedStride;
                *convertedBufferOut = found->second;
                return 0;
            }
        }

        std::vector<uint8_t> converted;
        try {
            converted.resize(convertedLength, 0);
        } catch (const std::bad_alloc&) {
            if (err && errcap) snprintf(err, errcap, "converted allocation failed");
            return -1;
        }

        const bool preservesVertexBytes =
            kind == MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT ||
            kind == MGL_RENDER_VERTEX_INT_TO_FLOAT ||
            kind == MGL_RENDER_VERTEX_FIXED_TO_FLOAT;
        for (size_t vertex = 0; vertex < vertexCount; ++vertex) {
            const size_t sourceOffset = vertex * originalStride;
            const size_t destinationOffset = vertex * convertedStride;
            const size_t remaining = sourceOffset < copyLength
                ? copyLength - sourceOffset
                : 0;
            const size_t copyBytes = std::min(originalStride, remaining);
            if (preservesVertexBytes && copyBytes > 0) {
                memcpy(converted.data() + destinationOffset,
                       sourceBase + sourceOffset, copyBytes);
            }

            if (kind == MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT ||
                kind == MGL_RENDER_VERTEX_INT_TO_FLOAT ||
                kind == MGL_RENDER_VERTEX_FIXED_TO_FLOAT) {
                const size_t inputBytes =
                    componentCount * sourceComponentSize;
                const size_t outputBytes = componentCount * sizeof(float);
                if (relativeOffset > copyBytes ||
                    inputBytes > copyBytes - relativeOffset ||
                    relativeOffset > convertedStride ||
                    outputBytes > convertedStride - relativeOffset) {
                    continue;
                }
                float values[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                for (uint32_t component = 0;
                     component < componentCount; ++component) {
                    const uint8_t* componentBytes =
                        sourceBase + sourceOffset + relativeOffset +
                        component * sourceComponentSize;
                    if (kind == MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT) {
                        double value = 0.0;
                        memcpy(&value, componentBytes, sizeof(value));
                        values[component] = static_cast<float>(value);
                    } else if (kind == MGL_RENDER_VERTEX_FIXED_TO_FLOAT) {
                        int32_t value = 0;
                        memcpy(&value, componentBytes, sizeof(value));
                        values[component] = static_cast<float>(
                            static_cast<double>(value) / 65536.0);
                    } else if (conversion->source_type == GL_INT) {
                        int32_t value = 0;
                        memcpy(&value, componentBytes, sizeof(value));
                        if (conversion->normalized) {
                            double normalized =
                                static_cast<double>(value) / 2147483647.0;
                            if (normalized < -1.0) normalized = -1.0;
                            values[component] = static_cast<float>(normalized);
                        } else {
                            values[component] = static_cast<float>(value);
                        }
                    } else {
                        uint32_t value = 0;
                        memcpy(&value, componentBytes, sizeof(value));
                        values[component] = conversion->normalized
                            ? static_cast<float>(
                                  static_cast<double>(value) / 4294967295.0)
                            : static_cast<float>(value);
                    }
                }
                memcpy(converted.data() + destinationOffset + relativeOffset,
                       values, outputBytes);
                continue;
            }

            if (kind == MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT ||
                kind == MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT) {
                if (relativeOffset > remaining ||
                    sizeof(uint32_t) > remaining - relativeOffset) {
                    continue;
                }
                const size_t outputBytes = componentCount * sizeof(float);
                if (relativeOffset > convertedStride ||
                    outputBytes > convertedStride - relativeOffset) {
                    continue;
                }
                uint32_t packed = 0;
                memcpy(&packed,
                       sourceBase + sourceOffset + relativeOffset,
                       sizeof(packed));
                float values[4] = {};
                if (kind == MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT) {
                    values[0] = static_cast<float>((packed >> 22) & 0x3ffu) /
                                1023.0f;
                    values[1] = static_cast<float>((packed >> 12) & 0x3ffu) /
                                1023.0f;
                    values[2] = static_cast<float>((packed >> 2) & 0x3ffu) /
                                1023.0f;
                    values[3] = static_cast<float>(packed & 0x3u) / 3.0f;
                } else {
                    values[0] = mgl::decodeUnsignedFloatComponent(
                        (packed >> 0) & 0x7ffu, 6);
                    values[1] = mgl::decodeUnsignedFloatComponent(
                        (packed >> 11) & 0x7ffu, 6);
                    values[2] = mgl::decodeUnsignedFloatComponent(
                        (packed >> 22) & 0x3ffu, 5);
                }
                memcpy(converted.data() + destinationOffset + relativeOffset,
                       values, outputBytes);
                continue;
            }

            uint8_t* destination = converted.data() + destinationOffset;
            for (uint32_t component = 0;
                 component < componentCount; ++component) {
                const size_t componentOffset =
                    sourceOffset + relativeOffset +
                    component * sourceComponentSize;
                if (componentOffset > copyLength ||
                    sourceComponentSize > copyLength - componentOffset) {
                    break;
                }
                const uint8_t* source = sourceBase + componentOffset;
                uint32_t value = 0;
                switch (conversion->source_type) {
                    case GL_BYTE: {
                        int8_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = static_cast<uint32_t>(static_cast<int32_t>(v));
                        break;
                    }
                    case GL_UNSIGNED_BYTE: {
                        uint8_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = v;
                        break;
                    }
                    case GL_SHORT: {
                        int16_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = static_cast<uint32_t>(static_cast<int32_t>(v));
                        break;
                    }
                    case GL_UNSIGNED_SHORT: {
                        uint16_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = v;
                        break;
                    }
                    case GL_INT: {
                        int32_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = static_cast<uint32_t>(v);
                        break;
                    }
                    case GL_UNSIGNED_INT:
                        memcpy(&value, source, sizeof(value));
                        break;
                    default:
                        break;
                }
                memcpy(destination + component * sizeof(uint32_t),
                       &value, sizeof(value));
            }
        }

        void* createdObject = nullptr;
        if (mglRenderCreateBufferWithBytes(
                converted.data(), convertedLength,
                static_cast<uint64_t>(MTL::ResourceStorageModeShared),
                nullptr, &createdObject) != 0 || !createdObject) {
            if (err && errcap) snprintf(err, errcap, "Metal buffer creation failed");
            return -1;
        }
        MTL::Buffer* created = static_cast<MTL::Buffer*>(createdObject);
        {
            std::lock_guard<std::mutex> lock(renderer.mutex);
            auto found = renderer.convertedVertexBuffers.find(key);
            if (found != renderer.convertedVertexBuffers.end() &&
                found->second) {
                created->release();
                found->second->retain();
                *convertedBufferOut = found->second;
            } else {
                try {
                    renderer.convertedVertexBuffers.emplace(key, created);
                    created->retain();
                    *convertedBufferOut = created;
                    if (renderer.convertedVertexBuffers.size() > 64) {
                        size_t evictCount =
                            renderer.convertedVertexBuffers.size() / 4;
                        while (evictCount-- > 0 &&
                               !renderer.convertedVertexBuffers.empty()) {
                            auto evict = renderer.convertedVertexBuffers.begin();
                            if (evict->second) evict->second->release();
                            renderer.convertedVertexBuffers.erase(evict);
                        }
                    }
                } catch (const std::bad_alloc&) {
                    *convertedBufferOut = created;
                }
            }
        }
        *convertedStrideOut = convertedStride;
        return 0;
    }

bad_components:
    if (err && errcap) {
        snprintf(err, errcap, "invalid component count=%u",
                 conversion->component_count);
    }
    return -1;
}

int mglRenderCreateBuffer(uint64_t length,
                             uint64_t resource_options,
                             const char* label,
                             void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || length == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::Buffer* buffer = renderer.device->newBuffer(
        static_cast<NS::UInteger>(length),
        static_cast<MTL::ResourceOptions>(resource_options));
    if (!buffer) return -1;
    if (label && label[0]) {
        buffer->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *buffer_out = buffer;
    return 0;
}

int mglRenderCreateBufferWithBytes(const void* bytes,
                                      uint64_t length,
                                      uint64_t resource_options,
                                      const char* label,
                                      void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || !bytes || length == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::Buffer* buffer = renderer.device->newBuffer(
        bytes, static_cast<NS::UInteger>(length),
        static_cast<MTL::ResourceOptions>(resource_options));
    if (!buffer) return -1;
    if (label && label[0]) {
        buffer->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *buffer_out = buffer;
    return 0;
}

int mglRenderCreateBufferWithBytesNoCopy(const void* bytes,
                                            uint64_t length,
                                            uint64_t resource_options,
                                            const char* label,
                                            int deallocate_vm,
                                            void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || !bytes || length == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    /* Keep the deallocator in the Metal object rather than releasing the
     * backing VM range when the GL Buffer shell disappears.  Command buffers
     * may retain this MTLBuffer past the GL-side unbind/delete. */
    void (^deallocator)(void*, NS::UInteger) = nil;
    if (deallocate_vm) {
        deallocator = ^(void* pointer, NS::UInteger size) {
            if (!pointer || size == 0) return;
            kern_return_t result = vm_deallocate(
                (vm_map_t)mach_task_self(),
                (vm_address_t)pointer, (vm_size_t)size);
            if (result != KERN_SUCCESS) {
                fprintf(stderr,
                        "MGL WARNING: Metal-cpp no-copy vm_deallocate "
                        "failed err=%d ptr=%p len=%llu\\n",
                        result, pointer,
                        (unsigned long long)size);
            }
        };
    }
    MTL::Buffer* buffer = renderer.device->newBuffer(
        bytes, static_cast<NS::UInteger>(length),
        static_cast<MTL::ResourceOptions>(resource_options), deallocator);
    if (!buffer) return -1;
    if (label && label[0]) {
        buffer->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *buffer_out = buffer;
    return 0;
}

int mglRenderGetQueryVisibilityBuffer(MGLQueryStateOwner * owner_handle, void** visibility_buffer_out) {
    if (visibility_buffer_out) *visibility_buffer_out = nullptr;
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !visibility_buffer_out || !owner->visibilityBuffer) {
        return -1;
    }
    *visibility_buffer_out = owner->visibilityBuffer;
    return 0;
}

int mglRenderSetRenderBufferForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* buffer, uint64_t offset, uint32_t stage, uint32_t index) {
    return mglRenderSetRenderBuffer(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        buffer, offset, stage, index);
}

int mglRenderSetTessellationFactorBufferForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* buffer, uint64_t offset, uint64_t instance_stride) {
    return mglRenderSetTessellationFactorBuffer(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        buffer, offset, instance_stride);
}
