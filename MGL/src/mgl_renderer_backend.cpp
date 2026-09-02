/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#include "mgl_renderer_backend.h"
#include "mgl_renderer_batch.h"
#include "mgl_renderer_blit.h"
#include "mgl_renderer_compute.h"
#include "mgl_renderer_platform.h"
#include "mgl_renderer_texture.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <vector>

#include "glm_context.h"
#include "mgl_metal.h"
#include "mgl_program_resource.h"
#include "mgl_render.h"
#include "mgl_shader_resource.h"

extern "C" Program *mglResolveProgramForStageFromState(
    GLMContext context, int stage);
extern "C" void mglRendererPlatformBackendWillDestroy(
    void *platform_shell, MGLRendererBackendHandle *backend);
extern "C" void mglRendererDrawArrays(GLMContext context,
    uint32_t mode, int32_t first, int32_t count);
extern "C" void mglRendererDrawElements(GLMContext context,
    uint32_t mode, int32_t count, uint32_t type, const void *indices);
extern "C" void mglRendererDrawRangeElements(GLMContext context, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices);
extern "C" void mglRendererDrawArraysInstanced(GLMContext context, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count);
extern "C" void mglRendererDrawElementsInstanced(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count);
extern "C" void mglRendererDrawElementsBaseVertex(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex);
extern "C" void mglRendererDrawRangeElementsBaseVertex(GLMContext context, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex);
extern "C" void mglRendererDrawElementsInstancedBaseVertex(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex);
extern "C" void mglRendererDrawArraysIndirect(GLMContext context,
    uint32_t mode, const void *indirect);
extern "C" void mglRendererDrawElementsIndirect(GLMContext context,
    uint32_t mode, uint32_t type, const void *indirect);
extern "C" void mglRendererDrawArraysInstancedBaseInstance(GLMContext context, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count,
    uint32_t base_instance);
extern "C" void mglRendererDrawElementsInstancedBaseInstance(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, uint32_t base_instance);
extern "C" void mglRendererDrawElementsInstancedBaseVertexBaseInstance(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex, uint32_t base_instance);
extern "C" void mglRendererMultiDrawArrays(GLMContext context, uint32_t mode,
    const int32_t *firsts, const int32_t *counts, int32_t draw_count);
extern "C" void mglRendererMultiDrawElements(GLMContext context, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count);
extern "C" void mglRendererMultiDrawElementsBaseVertex(GLMContext context, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count, const int32_t *base_vertices);
extern "C" void mglRendererMultiDrawArraysIndirect(GLMContext context, uint32_t mode,
    const void *indirect, int32_t draw_count, int32_t stride);
extern "C" void mglRendererMultiDrawElementsIndirect(GLMContext context, uint32_t mode, uint32_t type,
    const void *indirect, int32_t draw_count, int32_t stride);
extern "C" void mglRendererObjCBindTexture(GLMContext context, Texture *texture);
extern "C" void mglRendererObjCFlushDrawBuffer(GLMContext context);
extern "C" void mglRendererObjCGenerateMipmaps(GLMContext context, Texture *texture);

struct MGLRendererBackendPassthroughCache {
    MTL::Library *library = nullptr;
    MTL::Function *function = nullptr;
    uint64_t program_instance_id = 0;
};

static constexpr uint16_t kMGLSamplerSnapshotCacheCapacity = 256u;
static constexpr uint16_t kMGLSamplerSnapshotCacheIndexCapacity = 512u;

struct MGLRendererBackendSamplerSnapshotCache {
    std::array<MGLSamplerSnapshotKey, kMGLSamplerSnapshotCacheCapacity> keys{};
    std::array<MTL::SamplerState *, kMGLSamplerSnapshotCacheCapacity> states{};
    std::array<uint16_t, kMGLSamplerSnapshotCacheIndexCapacity> index{};
    uint16_t count = 0;
    uint16_t next = 0;
};

struct MGLRendererBackendFallbackTextureEntry {
    uint64_t key = 0;
    MTL::Texture *texture = nullptr;
};

struct MGLRendererBackendStageCopyBackSlot {
    MTL::Buffer *temporary = nullptr;
    MTL::Buffer *destination = nullptr;
};

struct MGLRendererBackendStageCopyBackList {
    const void *key = nullptr;
    std::array<MGLRendererBackendStageCopyBackSlot, 31> slots{};
};

struct MGLRendererBackendCurrentAttribCacheEntry {
    MTL::Buffer *buffer = nullptr;
    std::array<uint8_t, 16> bytes{};
    uint64_t stride = 0;
    uint32_t byte_count = 0;
};

struct MGLRendererBackendSizeConstantsCacheEntry {
    MTL::Buffer *buffer = nullptr;
    std::array<uint32_t, 31> constants{};
    bool valid = false;
};

struct MGLRendererBackendHandle {
    std::mutex mutex;
    GLMContext context = nullptr;
    MTL::Device *device = nullptr;
    void *command_queue_owner = nullptr;
    MTL::CommandQueue *command_queue = nullptr;
    void *command_buffer_owner = nullptr;
    void *render_encoder_owner = nullptr;
    void *render_pass_state_owner = nullptr;
    void *query_owner = nullptr;
    void *recovery_owner = nullptr;
    void *binding_owner = nullptr;
    MTL::Texture *fallback_render_target_texture = nullptr;
    MTL::Buffer *fallback_binding_buffer = nullptr;
    uint64_t fallback_binding_buffer_length = 0;
    MTL::Buffer *cull_distance_dummy_buffer = nullptr;
    MTL::Texture *transient_depth_texture = nullptr;
    uint64_t transient_depth_texture_width = 0;
    uint64_t transient_depth_texture_height = 0;
    std::array<MTL::Texture *, 6> default_draw_buffer_colors{};
    std::array<MTL::Texture *, 6> default_draw_buffer_depths{};
    std::array<MTL::Texture *, 6> default_draw_buffer_stencils{};
    std::vector<MGLRendererBackendStageCopyBackList> stage_copy_back_lists;
    std::array<MGLRendererBackendCurrentAttribCacheEntry, MAX_ATTRIBS>
        current_attrib_cache{};
    std::array<MGLRendererBackendSizeConstantsCacheEntry, 2>
        size_constants_cache{};
    MTL::SamplerState *scaled_blit_nearest_sampler = nullptr;
    MTL::SamplerState *scaled_blit_linear_sampler = nullptr;
    MTL::DepthStencilState *clear_rect_depth_state = nullptr;
    MGLRendererBackendPassthroughCache geometry_passthrough;
    MGLRendererBackendPassthroughCache tess_evaluation_passthrough;
    MGLRendererBackendSamplerSnapshotCache sampler_snapshots;
    MTL::Buffer *tess_factor_buffer = nullptr;
    uint32_t tess_factor_patch_count = 0;
    std::array<float, 6> tess_factor_levels{};
    MTL::Buffer *current_tess_factor_buffer = nullptr;
    MTL::Buffer *tess_xfb_dummy_buffer = nullptr;
    MTL::Buffer *cull_distance_capture_buffer = nullptr;
    MTL::Buffer *tess_control_point_index_buffer = nullptr;
    MTL::Buffer *tess_vertex_capture_buffer = nullptr;
    MTL::Buffer *tcs_patch_out_buffer = nullptr;
    MTL::Buffer *tcs_output_buffer = nullptr;
    MTL::Texture *fallback_sampled_texture = nullptr;
    MTL::Texture *fallback_cube_sampled_texture = nullptr;
    MTL::Buffer *fallback_texture_buffer_storage = nullptr;
    MTL::Texture *fallback_sint_texture_buffer = nullptr;
    MTL::SamplerState *fallback_sampler = nullptr;
    std::vector<MGLRendererBackendFallbackTextureEntry>
        fallback_sampled_textures;
    std::vector<MTL::Texture *> proactive_textures;
    bool renderer_initialized = false;
    bool shutdown_started = false;
    bool destroying = false;
};

static void *mglRendererBackendPlatformShell(GLMContext context)
{
    return context && context->renderer_backend
        ? context->platform_renderer_shell
        : nullptr;
}

static void mglRendererBackendReleaseOwnedState(
    MGLRendererBackendHandle *backend)
{
    if (!backend) return;
    if (backend->fallback_render_target_texture) {
        backend->fallback_render_target_texture->release();
        backend->fallback_render_target_texture = nullptr;
    }
    if (backend->fallback_binding_buffer) {
        backend->fallback_binding_buffer->release();
        backend->fallback_binding_buffer = nullptr;
    }
    backend->fallback_binding_buffer_length = 0;
    if (backend->cull_distance_dummy_buffer) {
        backend->cull_distance_dummy_buffer->release();
        backend->cull_distance_dummy_buffer = nullptr;
    }
    if (backend->transient_depth_texture) {
        backend->transient_depth_texture->release();
        backend->transient_depth_texture = nullptr;
    }
    backend->transient_depth_texture_width = 0;
    backend->transient_depth_texture_height = 0;
    for (MTL::Texture *texture : backend->default_draw_buffer_colors) {
        if (texture) texture->release();
    }
    backend->default_draw_buffer_colors = {};
    for (MTL::Texture *texture : backend->default_draw_buffer_depths) {
        if (texture) texture->release();
    }
    backend->default_draw_buffer_depths = {};
    for (MTL::Texture *texture : backend->default_draw_buffer_stencils) {
        if (texture) texture->release();
    }
    backend->default_draw_buffer_stencils = {};
    for (MGLRendererBackendStageCopyBackList &list :
         backend->stage_copy_back_lists) {
        for (MGLRendererBackendStageCopyBackSlot &slot : list.slots) {
            if (slot.temporary) slot.temporary->release();
            if (slot.destination) slot.destination->release();
        }
    }
    backend->stage_copy_back_lists.clear();
    for (MGLRendererBackendCurrentAttribCacheEntry &entry :
         backend->current_attrib_cache) {
        if (entry.buffer) entry.buffer->release();
    }
    backend->current_attrib_cache = {};
    for (MGLRendererBackendSizeConstantsCacheEntry &entry :
         backend->size_constants_cache) {
        if (entry.buffer) entry.buffer->release();
    }
    backend->size_constants_cache = {};
    if (backend->scaled_blit_nearest_sampler) {
        backend->scaled_blit_nearest_sampler->release();
        backend->scaled_blit_nearest_sampler = nullptr;
    }
    if (backend->scaled_blit_linear_sampler) {
        backend->scaled_blit_linear_sampler->release();
        backend->scaled_blit_linear_sampler = nullptr;
    }
    if (backend->clear_rect_depth_state) {
        backend->clear_rect_depth_state->release();
        backend->clear_rect_depth_state = nullptr;
    }
    if (backend->geometry_passthrough.function) {
        backend->geometry_passthrough.function->release();
    }
    if (backend->geometry_passthrough.library) {
        backend->geometry_passthrough.library->release();
    }
    backend->geometry_passthrough = {};
    if (backend->tess_evaluation_passthrough.function) {
        backend->tess_evaluation_passthrough.function->release();
    }
    if (backend->tess_evaluation_passthrough.library) {
        backend->tess_evaluation_passthrough.library->release();
    }
    backend->tess_evaluation_passthrough = {};
    for (uint16_t i = 0; i < backend->sampler_snapshots.count; i++) {
        if (backend->sampler_snapshots.states[i]) {
            backend->sampler_snapshots.states[i]->release();
        }
    }
    backend->sampler_snapshots = {};
    if (backend->tess_factor_buffer) {
        backend->tess_factor_buffer->release();
        backend->tess_factor_buffer = nullptr;
    }
    backend->tess_factor_patch_count = 0;
    backend->tess_factor_levels = {};
    if (backend->current_tess_factor_buffer) {
        backend->current_tess_factor_buffer->release();
        backend->current_tess_factor_buffer = nullptr;
    }
    if (backend->tess_xfb_dummy_buffer) {
        backend->tess_xfb_dummy_buffer->release();
        backend->tess_xfb_dummy_buffer = nullptr;
    }
    if (backend->cull_distance_capture_buffer) {
        backend->cull_distance_capture_buffer->release();
        backend->cull_distance_capture_buffer = nullptr;
    }
    if (backend->tess_control_point_index_buffer) {
        backend->tess_control_point_index_buffer->release();
        backend->tess_control_point_index_buffer = nullptr;
    }
    if (backend->tess_vertex_capture_buffer) {
        backend->tess_vertex_capture_buffer->release();
        backend->tess_vertex_capture_buffer = nullptr;
    }
    if (backend->tcs_patch_out_buffer) {
        backend->tcs_patch_out_buffer->release();
        backend->tcs_patch_out_buffer = nullptr;
    }
    if (backend->tcs_output_buffer) {
        backend->tcs_output_buffer->release();
        backend->tcs_output_buffer = nullptr;
    }
    if (backend->fallback_sampled_texture) {
        backend->fallback_sampled_texture->release();
        backend->fallback_sampled_texture = nullptr;
    }
    if (backend->fallback_cube_sampled_texture) {
        backend->fallback_cube_sampled_texture->release();
        backend->fallback_cube_sampled_texture = nullptr;
    }
    if (backend->fallback_texture_buffer_storage) {
        backend->fallback_texture_buffer_storage->release();
        backend->fallback_texture_buffer_storage = nullptr;
    }
    if (backend->fallback_sint_texture_buffer) {
        backend->fallback_sint_texture_buffer->release();
        backend->fallback_sint_texture_buffer = nullptr;
    }
    if (backend->fallback_sampler) {
        backend->fallback_sampler->release();
        backend->fallback_sampler = nullptr;
    }
    for (MGLRendererBackendFallbackTextureEntry &entry :
         backend->fallback_sampled_textures) {
        if (entry.texture) entry.texture->release();
    }
    backend->fallback_sampled_textures.clear();
    for (MTL::Texture *texture : backend->proactive_textures) {
        if (texture) texture->release();
    }
    backend->proactive_textures.clear();
    mglRenderDestroyCommandQueueOwner(&backend->command_queue_owner);
    backend->command_queue = nullptr;
    mglRenderBindingDestroy(backend->binding_owner);
    backend->binding_owner = nullptr;
    mglRenderDestroyQueryStateOwner(&backend->query_owner);
    mglRenderDestroyCommandRecoveryOwner(&backend->recovery_owner);
    backend->command_buffer_owner = nullptr;
    backend->render_encoder_owner = nullptr;
    backend->render_pass_state_owner = nullptr;
    if (backend->renderer_initialized) {
        mglRenderShutdown();
        backend->renderer_initialized = false;
    }
    if (backend->device) {
        backend->device->release();
        backend->device = nullptr;
    }
}

template <typename T>
static void mglRendererBackendReplaceObject(T *&slot, void *object)
{
    T *replacement = static_cast<T *>(object);
    if (replacement == slot) return;
    if (replacement) replacement->retain();
    if (slot) slot->release();
    slot = replacement;
}

static bool mglRendererBackendStageCopyBackListEmpty(
    const MGLRendererBackendStageCopyBackList &list)
{
    for (const MGLRendererBackendStageCopyBackSlot &slot : list.slots) {
        if (slot.temporary || slot.destination) return false;
    }
    return true;
}

static MGLRendererBackendPassthroughCache *
mglRendererBackendPassthroughCacheForKind(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendPassthroughKind kind)
{
    if (!backend) return nullptr;
    switch (kind) {
        case MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY:
            return &backend->geometry_passthrough;
        case MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION:
            return &backend->tess_evaluation_passthrough;
    }
    return nullptr;
}

static void mglRendererBackendReplacePassthroughCache(
    MGLRendererBackendPassthroughCache *cache,
    void *library, void *function, uint64_t program_instance_id)
{
    if (!cache) return;
    MTL::Library *new_library = static_cast<MTL::Library *>(library);
    MTL::Function *new_function = static_cast<MTL::Function *>(function);
    if (new_library) new_library->retain();
    if (new_function) new_function->retain();
    if (cache->function) cache->function->release();
    if (cache->library) cache->library->release();
    cache->library = new_library;
    cache->function = new_function;
    cache->program_instance_id = new_library && new_function
        ? program_instance_id : 0;
}

static uint64_t mglRendererBackendHashSamplerSnapshotKey(
    const MGLSamplerSnapshotKey *key)
{
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(key);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(*key); i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static int mglRendererBackendFindSamplerSnapshotSlot(
    const MGLRendererBackendSamplerSnapshotCache &cache,
    const MGLSamplerSnapshotKey *key)
{
    const uint32_t mask = kMGLSamplerSnapshotCacheIndexCapacity - 1u;
    uint32_t hash_slot =
        static_cast<uint32_t>(mglRendererBackendHashSamplerSnapshotKey(key)) & mask;
    for (uint32_t probe = 0; probe < kMGLSamplerSnapshotCacheIndexCapacity;
         probe++, hash_slot = (hash_slot + 1u) & mask) {
        uint16_t encoded = cache.index[hash_slot];
        if (encoded == 0u) break;
        if (encoded == UINT16_MAX) continue;
        uint16_t slot = encoded - 1u;
        if (slot < cache.count &&
            std::memcmp(&cache.keys[slot], key, sizeof(*key)) == 0) {
            return static_cast<int>(slot);
        }
    }
    return -1;
}

static void mglRendererBackendRemoveSamplerSnapshotIndex(
    MGLRendererBackendSamplerSnapshotCache &cache, uint16_t slot)
{
    const uint32_t mask = kMGLSamplerSnapshotCacheIndexCapacity - 1u;
    uint32_t hash_slot = static_cast<uint32_t>(
        mglRendererBackendHashSamplerSnapshotKey(&cache.keys[slot])) & mask;
    for (uint32_t probe = 0; probe < kMGLSamplerSnapshotCacheIndexCapacity;
         probe++, hash_slot = (hash_slot + 1u) & mask) {
        uint16_t encoded = cache.index[hash_slot];
        if (encoded == 0u) break;
        if (encoded == slot + 1u) {
            cache.index[hash_slot] = UINT16_MAX;
            break;
        }
    }
}

static int mglRendererBackendInsertSamplerSnapshotIndex(
    MGLRendererBackendSamplerSnapshotCache &cache,
    const MGLSamplerSnapshotKey *key, uint16_t slot)
{
    const uint32_t mask = kMGLSamplerSnapshotCacheIndexCapacity - 1u;
    uint32_t hash_slot =
        static_cast<uint32_t>(mglRendererBackendHashSamplerSnapshotKey(key)) & mask;
    uint32_t first_tombstone = UINT32_MAX;
    for (uint32_t probe = 0; probe < kMGLSamplerSnapshotCacheIndexCapacity;
         probe++, hash_slot = (hash_slot + 1u) & mask) {
        uint16_t encoded = cache.index[hash_slot];
        if (encoded == UINT16_MAX && first_tombstone == UINT32_MAX) {
            first_tombstone = hash_slot;
        } else if (encoded == 0u) {
            if (first_tombstone != UINT32_MAX) hash_slot = first_tombstone;
            cache.index[hash_slot] = slot + 1u;
            return 0;
        }
    }
    if (first_tombstone != UINT32_MAX) {
        cache.index[first_tombstone] = slot + 1u;
        return 0;
    }
    return -1;
}

extern "C" int mglRendererBackendCreate(
    const MGLRendererBackendCreateInfo *info,
    MGLRendererBackendHandle **backend_out)
{
    if (backend_out) *backend_out = nullptr;
    if (!info || !backend_out || !info->objc_device ||
        info->binding_slot_count == 0 || info->query_capacity == 0) {
        return -1;
    }

    MGLRendererBackendHandle *backend = new MGLRendererBackendHandle();
    backend->context = info->context;
    backend->device = static_cast<MTL::Device *>(info->objc_device);
    backend->device->retain();
    if (mglRenderInit(info->objc_device) != 0) {
        backend->device->release();
        backend->device = nullptr;
        delete backend;
        return -1;
    }
    backend->renderer_initialized = true;

    backend->binding_owner =
        mglRenderBindingCreate(info->binding_slot_count);
    if (!backend->binding_owner ||
        mglRenderCreateQueryStateOwner(
            info->query_capacity, &backend->query_owner) != 0 ||
        mglRenderCreateCommandRecoveryOwner(
            &backend->recovery_owner) != 0) {
        mglRendererBackendReleaseOwnedState(backend);
        delete backend;
        return -1;
    }
    *backend_out = backend;
    return 0;
}

extern "C" int mglRendererBackendIsReady(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return 0;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->device && backend->renderer_initialized &&
           !backend->shutdown_started &&
           backend->command_queue_owner && backend->binding_owner &&
           backend->query_owner && backend->recovery_owner;
}

extern "C" void *mglRendererBackendGetDevice(
    const MGLRendererBackendHandle *backend)
{
    return backend ? backend->device : nullptr;
}

extern "C" int mglRendererBackendResetCommandQueue(
    MGLRendererBackendHandle *backend,
    uint32_t max_command_buffers,
    void **command_queue_out)
{
    if (command_queue_out) *command_queue_out = nullptr;
    if (!backend || !command_queue_out) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (!backend->renderer_initialized || backend->shutdown_started) return -1;
    backend->command_queue = nullptr;
    void *queue = nullptr;
    int result = backend->command_queue_owner
        ? mglRenderResetCommandQueueOwner(
              backend->command_queue_owner, max_command_buffers, &queue)
        : mglRenderCreateCommandQueueOwner(
              max_command_buffers, &backend->command_queue_owner, &queue);
    if (result != 0 || !queue) return -1;
    backend->command_queue = static_cast<MTL::CommandQueue *>(queue);
    *command_queue_out = queue;
    return 0;
}

extern "C" void *mglRendererBackendGetCommandQueue(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->command_queue;
}

extern "C" int mglRendererBackendAttachRuntimeOwners(
    MGLRendererBackendHandle *backend,
    void *command_buffer_owner,
    void *render_encoder_owner,
    void *render_pass_state_owner)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    backend->command_buffer_owner = command_buffer_owner;
    backend->render_encoder_owner = render_encoder_owner;
    backend->render_pass_state_owner = render_pass_state_owner;
    return 0;
}

extern "C" int mglRendererBackendSetFallbackRenderTargetTexture(
    MGLRendererBackendHandle *backend, void *texture)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(
        backend->fallback_render_target_texture, texture);
    return 0;
}

extern "C" void *mglRendererBackendGetFallbackRenderTargetTexture(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->fallback_render_target_texture;
}

extern "C" void *mglRendererBackendGetFallbackBindingBuffer(
    MGLRendererBackendHandle *backend, uint64_t minimum_length)
{
    if (!backend || minimum_length == 0u) return nullptr;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying || !backend->device) return nullptr;
    if (!backend->fallback_binding_buffer ||
        backend->fallback_binding_buffer_length < minimum_length) {
        MTL::Buffer *replacement = backend->device->newBuffer(
            static_cast<NS::UInteger>(minimum_length),
            MTL::ResourceStorageModeShared);
        if (!replacement) return nullptr;
        if (backend->fallback_binding_buffer) {
            backend->fallback_binding_buffer->release();
        }
        backend->fallback_binding_buffer = replacement;
        backend->fallback_binding_buffer_length = minimum_length;
    }
    return backend->fallback_binding_buffer;
}

extern "C" void *mglRendererBackendGetCullDistanceDummyBuffer(
    MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying || !backend->device) return nullptr;
    if (!backend->cull_distance_dummy_buffer) {
        const float dummy[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        backend->cull_distance_dummy_buffer = backend->device->newBuffer(
            dummy, sizeof(dummy), MTL::ResourceStorageModeShared);
    }
    return backend->cull_distance_dummy_buffer;
}

extern "C" int mglRendererBackendSetTransientDepthTexture(
    MGLRendererBackendHandle *backend, void *texture,
    uint64_t width, uint64_t height)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(backend->transient_depth_texture, texture);
    backend->transient_depth_texture_width = texture ? width : 0;
    backend->transient_depth_texture_height = texture ? height : 0;
    return 0;
}

extern "C" void *mglRendererBackendGetTransientDepthTexture(
    const MGLRendererBackendHandle *backend,
    uint64_t *width_out, uint64_t *height_out)
{
    if (width_out) *width_out = 0;
    if (height_out) *height_out = 0;
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    if (width_out) *width_out = backend->transient_depth_texture_width;
    if (height_out) *height_out = backend->transient_depth_texture_height;
    return backend->transient_depth_texture;
}

extern "C" int mglRendererBackendSetDefaultDrawBufferAttachment(
    MGLRendererBackendHandle *backend, uint32_t draw_buffer_index,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind, void *texture)
{
    if (!backend || draw_buffer_index >= 6u) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    switch (kind) {
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR:
            mglRendererBackendReplaceObject(
                backend->default_draw_buffer_colors[draw_buffer_index], texture);
            return 0;
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH:
            mglRendererBackendReplaceObject(
                backend->default_draw_buffer_depths[draw_buffer_index], texture);
            return 0;
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL:
            mglRendererBackendReplaceObject(
                backend->default_draw_buffer_stencils[draw_buffer_index], texture);
            return 0;
    }
    return -1;
}

extern "C" void *mglRendererBackendGetDefaultDrawBufferAttachment(
    const MGLRendererBackendHandle *backend, uint32_t draw_buffer_index,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind)
{
    if (!backend || draw_buffer_index >= 6u) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    switch (kind) {
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR:
            return backend->default_draw_buffer_colors[draw_buffer_index];
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH:
            return backend->default_draw_buffer_depths[draw_buffer_index];
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL:
            return backend->default_draw_buffer_stencils[draw_buffer_index];
    }
    return nullptr;
}

extern "C" int mglRendererBackendClearDefaultDrawBuffer(
    MGLRendererBackendHandle *backend, uint32_t draw_buffer_index)
{
    if (!backend || draw_buffer_index >= 6u) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(
        backend->default_draw_buffer_colors[draw_buffer_index], nullptr);
    mglRendererBackendReplaceObject(
        backend->default_draw_buffer_depths[draw_buffer_index], nullptr);
    mglRendererBackendReplaceObject(
        backend->default_draw_buffer_stencils[draw_buffer_index], nullptr);
    return 0;
}

extern "C" int mglRendererBackendSetStageCopyBackResources(
    MGLRendererBackendHandle *backend, const void *copy_back_list_key,
    uint32_t slot, void *temporary, void *destination)
{
    if (!backend || !copy_back_list_key || slot >= 31u ||
        !temporary || !destination) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    auto list_it = std::find_if(
        backend->stage_copy_back_lists.begin(),
        backend->stage_copy_back_lists.end(),
        [copy_back_list_key](const MGLRendererBackendStageCopyBackList &list) {
            return list.key == copy_back_list_key;
        });
    if (list_it == backend->stage_copy_back_lists.end()) {
        backend->stage_copy_back_lists.push_back({});
        list_it = backend->stage_copy_back_lists.end() - 1;
        list_it->key = copy_back_list_key;
    }
    mglRendererBackendReplaceObject(list_it->slots[slot].temporary, temporary);
    mglRendererBackendReplaceObject(list_it->slots[slot].destination, destination);
    return 0;
}

extern "C" int mglRendererBackendGetStageCopyBackResources(
    const MGLRendererBackendHandle *backend, const void *copy_back_list_key,
    uint32_t slot, void **temporary_out, void **destination_out)
{
    if (temporary_out) *temporary_out = nullptr;
    if (destination_out) *destination_out = nullptr;
    if (!backend || !copy_back_list_key || slot >= 31u ||
        !temporary_out || !destination_out) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    auto list_it = std::find_if(
        backend->stage_copy_back_lists.begin(),
        backend->stage_copy_back_lists.end(),
        [copy_back_list_key](const MGLRendererBackendStageCopyBackList &list) {
            return list.key == copy_back_list_key;
        });
    if (list_it == backend->stage_copy_back_lists.end()) return 0;
    *temporary_out = list_it->slots[slot].temporary;
    *destination_out = list_it->slots[slot].destination;
    return (*temporary_out && *destination_out) ? 1 : 0;
}

extern "C" int mglRendererBackendClearStageCopyBackSlot(
    MGLRendererBackendHandle *backend, const void *copy_back_list_key,
    uint32_t slot)
{
    if (!backend || !copy_back_list_key || slot >= 31u) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    auto list_it = std::find_if(
        backend->stage_copy_back_lists.begin(),
        backend->stage_copy_back_lists.end(),
        [copy_back_list_key](const MGLRendererBackendStageCopyBackList &list) {
            return list.key == copy_back_list_key;
        });
    if (list_it == backend->stage_copy_back_lists.end()) return 0;
    mglRendererBackendReplaceObject(list_it->slots[slot].temporary, nullptr);
    mglRendererBackendReplaceObject(list_it->slots[slot].destination, nullptr);
    if (mglRendererBackendStageCopyBackListEmpty(*list_it)) {
        backend->stage_copy_back_lists.erase(list_it);
    }
    return 0;
}

extern "C" int mglRendererBackendClearStageCopyBackList(
    MGLRendererBackendHandle *backend, const void *copy_back_list_key)
{
    if (!backend || !copy_back_list_key) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    auto list_it = std::find_if(
        backend->stage_copy_back_lists.begin(),
        backend->stage_copy_back_lists.end(),
        [copy_back_list_key](const MGLRendererBackendStageCopyBackList &list) {
            return list.key == copy_back_list_key;
        });
    if (list_it == backend->stage_copy_back_lists.end()) return 0;
    for (MGLRendererBackendStageCopyBackSlot &entry : list_it->slots) {
        if (entry.temporary) entry.temporary->release();
        if (entry.destination) entry.destination->release();
    }
    backend->stage_copy_back_lists.erase(list_it);
    return 0;
}

extern "C" void *mglRendererBackendGetCurrentAttribBuffer(
    const MGLRendererBackendHandle *backend, uint32_t attrib,
    const void *bytes, uint32_t byte_count, uint64_t stride)
{
    if (!backend || attrib >= MAX_ATTRIBS || !bytes ||
        byte_count == 0u || byte_count > 16u || stride == 0u) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    if (backend->destroying) return nullptr;
    const MGLRendererBackendCurrentAttribCacheEntry &entry =
        backend->current_attrib_cache[attrib];
    if (!entry.buffer || entry.byte_count != byte_count ||
        entry.stride != stride ||
        std::memcmp(entry.bytes.data(), bytes, byte_count) != 0) {
        return nullptr;
    }
    return entry.buffer;
}

extern "C" int mglRendererBackendSetCurrentAttribBuffer(
    MGLRendererBackendHandle *backend, uint32_t attrib,
    const void *bytes, uint32_t byte_count, uint64_t stride, void *buffer)
{
    if (!backend || attrib >= MAX_ATTRIBS || !bytes ||
        byte_count == 0u || byte_count > 16u || stride == 0u || !buffer) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    MGLRendererBackendCurrentAttribCacheEntry &entry =
        backend->current_attrib_cache[attrib];
    mglRendererBackendReplaceObject(entry.buffer, buffer);
    entry.bytes = {};
    std::memcpy(entry.bytes.data(), bytes, byte_count);
    entry.byte_count = byte_count;
    entry.stride = stride;
    return 0;
}

extern "C" void *mglRendererBackendGetSizeConstantsBuffer(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendSizeConstantsStage stage,
    const uint32_t *constants, uint32_t count)
{
    if (!backend || stage < MGL_RENDERER_BACKEND_SIZE_CONSTANTS_VERTEX ||
        stage > MGL_RENDERER_BACKEND_SIZE_CONSTANTS_FRAGMENT ||
        !constants || count != 31u) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    if (backend->destroying) return nullptr;
    const MGLRendererBackendSizeConstantsCacheEntry &entry =
        backend->size_constants_cache[(size_t)stage];
    if (!entry.valid || !entry.buffer ||
        std::memcmp(entry.constants.data(), constants,
                    sizeof(entry.constants)) != 0) {
        return nullptr;
    }
    return entry.buffer;
}

extern "C" int mglRendererBackendSetSizeConstantsBuffer(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendSizeConstantsStage stage,
    const uint32_t *constants, uint32_t count, void *buffer)
{
    if (!backend || stage < MGL_RENDERER_BACKEND_SIZE_CONSTANTS_VERTEX ||
        stage > MGL_RENDERER_BACKEND_SIZE_CONSTANTS_FRAGMENT ||
        !constants || count != 31u || !buffer) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    MGLRendererBackendSizeConstantsCacheEntry &entry =
        backend->size_constants_cache[(size_t)stage];
    mglRendererBackendReplaceObject(entry.buffer, buffer);
    std::memcpy(entry.constants.data(), constants, sizeof(entry.constants));
    entry.valid = true;
    return 0;
}

extern "C" int mglRendererBackendSetBlitCachedObject(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendBlitCacheKind kind, void *object)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    switch (kind) {
        case MGL_RENDERER_BACKEND_BLIT_CACHE_NEAREST_SAMPLER:
            mglRendererBackendReplaceObject(
                backend->scaled_blit_nearest_sampler, object);
            return 0;
        case MGL_RENDERER_BACKEND_BLIT_CACHE_LINEAR_SAMPLER:
            mglRendererBackendReplaceObject(
                backend->scaled_blit_linear_sampler, object);
            return 0;
        case MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE:
            mglRendererBackendReplaceObject(
                backend->clear_rect_depth_state, object);
            return 0;
    }
    return -1;
}

extern "C" void *mglRendererBackendGetBlitCachedObject(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendBlitCacheKind kind)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    switch (kind) {
        case MGL_RENDERER_BACKEND_BLIT_CACHE_NEAREST_SAMPLER:
            return backend->scaled_blit_nearest_sampler;
        case MGL_RENDERER_BACKEND_BLIT_CACHE_LINEAR_SAMPLER:
            return backend->scaled_blit_linear_sampler;
        case MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE:
            return backend->clear_rect_depth_state;
    }
    return nullptr;
}

extern "C" int mglRendererBackendSetPassthroughFunction(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendPassthroughKind kind,
    void *library, void *function, uint64_t program_instance_id)
{
    if (!backend || ((library == nullptr) != (function == nullptr))) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    MGLRendererBackendPassthroughCache *cache =
        mglRendererBackendPassthroughCacheForKind(backend, kind);
    if (!cache) return -1;
    mglRendererBackendReplacePassthroughCache(
        cache, library, function, program_instance_id);
    return 0;
}

extern "C" int mglRendererBackendGetPassthroughFunction(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendPassthroughKind kind,
    uint64_t program_instance_id, void **function_out)
{
    if (function_out) *function_out = nullptr;
    if (!backend || !function_out) return -1;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    MGLRendererBackendPassthroughCache *cache =
        mglRendererBackendPassthroughCacheForKind(
            const_cast<MGLRendererBackendHandle *>(backend), kind);
    if (!cache) return -1;
    if (!cache->library || !cache->function ||
        cache->program_instance_id != program_instance_id) {
        return 0;
    }
    *function_out = cache->function;
    return 1;
}

extern "C" int mglRendererBackendGetSamplerSnapshotState(
    const MGLRendererBackendHandle *backend,
    const MGLSamplerSnapshotKey *key, void **state_out)
{
    if (state_out) *state_out = nullptr;
    if (!backend || !key || !state_out) return -1;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    int slot = mglRendererBackendFindSamplerSnapshotSlot(
        backend->sampler_snapshots, key);
    if (slot < 0) return 0;
    *state_out = backend->sampler_snapshots.states[slot];
    return *state_out ? 1 : 0;
}

extern "C" int mglRendererBackendPutSamplerSnapshotState(
    MGLRendererBackendHandle *backend,
    const MGLSamplerSnapshotKey *key, void *state)
{
    if (!backend || !key || !state) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    MGLRendererBackendSamplerSnapshotCache &cache = backend->sampler_snapshots;
    int existing_slot = mglRendererBackendFindSamplerSnapshotSlot(cache, key);
    if (existing_slot >= 0) {
        mglRendererBackendReplaceObject(
            cache.states[existing_slot], state);
        return 0;
    }

    uint16_t slot;
    if (cache.count < kMGLSamplerSnapshotCacheCapacity) {
        slot = cache.count++;
    } else {
        slot = cache.next++ % kMGLSamplerSnapshotCacheCapacity;
        mglRendererBackendRemoveSamplerSnapshotIndex(cache, slot);
    }

    MTL::SamplerState *replacement = static_cast<MTL::SamplerState *>(state);
    replacement->retain();
    if (cache.states[slot]) cache.states[slot]->release();
    cache.keys[slot] = *key;
    cache.states[slot] = replacement;
    if (mglRendererBackendInsertSamplerSnapshotIndex(cache, key, slot) != 0) {
        replacement->release();
        cache.states[slot] = nullptr;
        return -1;
    }
    return 0;
}

extern "C" int mglRendererBackendGetTessFactorBuffer(
    const MGLRendererBackendHandle *backend, uint32_t patch_count,
    const float levels[6], void **buffer_out)
{
    if (buffer_out) *buffer_out = nullptr;
    if (!backend || patch_count == 0u || !levels || !buffer_out) return -1;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    if (!backend->tess_factor_buffer ||
        backend->tess_factor_patch_count != patch_count) {
        return 0;
    }
    for (size_t i = 0; i < backend->tess_factor_levels.size(); i++) {
        if (backend->tess_factor_levels[i] != levels[i]) return 0;
    }
    *buffer_out = backend->tess_factor_buffer;
    return 1;
}

extern "C" int mglRendererBackendPutTessFactorBuffer(
    MGLRendererBackendHandle *backend, uint32_t patch_count,
    const float levels[6], void *buffer)
{
    if (!backend || patch_count == 0u || !levels || !buffer) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(backend->tess_factor_buffer, buffer);
    backend->tess_factor_patch_count = patch_count;
    std::copy_n(levels, backend->tess_factor_levels.size(),
                backend->tess_factor_levels.begin());
    return 0;
}

extern "C" int mglRendererBackendSetCurrentTessFactorBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(
        backend->current_tess_factor_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetCurrentTessFactorBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->current_tess_factor_buffer;
}

extern "C" int mglRendererBackendGetTessXfbDummyBuffer(
    const MGLRendererBackendHandle *backend, uint64_t minimum_length,
    void **buffer_out)
{
    if (buffer_out) *buffer_out = nullptr;
    if (!backend || minimum_length == 0u || !buffer_out) return -1;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    if (!backend->tess_xfb_dummy_buffer ||
        backend->tess_xfb_dummy_buffer->length() < minimum_length) {
        return 0;
    }
    *buffer_out = backend->tess_xfb_dummy_buffer;
    return 1;
}

extern "C" int mglRendererBackendPutTessXfbDummyBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend || !buffer) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(backend->tess_xfb_dummy_buffer, buffer);
    return 0;
}

extern "C" int mglRendererBackendSetCullDistanceCaptureBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(
        backend->cull_distance_capture_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetCullDistanceCaptureBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->cull_distance_capture_buffer;
}

extern "C" int mglRendererBackendSetTessControlPointIndexBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(
        backend->tess_control_point_index_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTessControlPointIndexBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->tess_control_point_index_buffer;
}

extern "C" int mglRendererBackendSetTessVertexCaptureBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(
        backend->tess_vertex_capture_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTessVertexCaptureBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->tess_vertex_capture_buffer;
}

extern "C" int mglRendererBackendSetTcsPatchOutBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(backend->tcs_patch_out_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTcsPatchOutBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->tcs_patch_out_buffer;
}

extern "C" int mglRendererBackendSetTcsOutputBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(backend->tcs_output_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTcsOutputBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->tcs_output_buffer;
}

extern "C" int mglRendererBackendSetFallbackResource(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendFallbackResourceKind kind, void *resource)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    switch (kind) {
        case MGL_RENDERER_BACKEND_FALLBACK_SAMPLED_TEXTURE:
            mglRendererBackendReplaceObject(
                backend->fallback_sampled_texture, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE:
            mglRendererBackendReplaceObject(
                backend->fallback_cube_sampled_texture, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE:
            mglRendererBackendReplaceObject(
                backend->fallback_texture_buffer_storage, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER:
            mglRendererBackendReplaceObject(
                backend->fallback_sint_texture_buffer, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_SAMPLER:
            mglRendererBackendReplaceObject(
                backend->fallback_sampler, resource);
            return 0;
    }
    return -1;
}

extern "C" void *mglRendererBackendGetFallbackResource(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendFallbackResourceKind kind)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    switch (kind) {
        case MGL_RENDERER_BACKEND_FALLBACK_SAMPLED_TEXTURE:
            return backend->fallback_sampled_texture;
        case MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE:
            return backend->fallback_cube_sampled_texture;
        case MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE:
            return backend->fallback_texture_buffer_storage;
        case MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER:
            return backend->fallback_sint_texture_buffer;
        case MGL_RENDERER_BACKEND_FALLBACK_SAMPLER:
            return backend->fallback_sampler;
    }
    return nullptr;
}

extern "C" int mglRendererBackendGetFallbackSampledTexture(
    const MGLRendererBackendHandle *backend,
    uint64_t key, void **texture_out)
{
    if (texture_out) *texture_out = nullptr;
    if (!backend || !texture_out) return -1;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    for (const MGLRendererBackendFallbackTextureEntry &entry :
         backend->fallback_sampled_textures) {
        if (entry.key == key) {
            *texture_out = entry.texture;
            return entry.texture ? 1 : 0;
        }
    }
    return 0;
}

extern "C" int mglRendererBackendPutFallbackSampledTexture(
    MGLRendererBackendHandle *backend,
    uint64_t key, void *texture)
{
    if (!backend || !texture) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    for (MGLRendererBackendFallbackTextureEntry &entry :
         backend->fallback_sampled_textures) {
        if (entry.key == key) {
            mglRendererBackendReplaceObject(entry.texture, texture);
            return 0;
        }
    }
    MTL::Texture *retained = static_cast<MTL::Texture *>(texture);
    retained->retain();
    backend->fallback_sampled_textures.push_back({key, retained});
    static constexpr size_t kFallbackSampledTextureCacheLimit = 32u;
    if (backend->fallback_sampled_textures.size() >
        kFallbackSampledTextureCacheLimit) {
        size_t evict_count = backend->fallback_sampled_textures.size() / 4u;
        for (size_t i = 0; i < evict_count; i++) {
            backend->fallback_sampled_textures[i].texture->release();
        }
        backend->fallback_sampled_textures.erase(
            backend->fallback_sampled_textures.begin(),
            backend->fallback_sampled_textures.begin() + evict_count);
    }
    return 0;
}

extern "C" int mglRendererBackendRetainProactiveTexture(
    MGLRendererBackendHandle *backend, void *texture)
{
    if (!backend || !texture) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    MTL::Texture *retained = static_cast<MTL::Texture *>(texture);
    retained->retain();
    backend->proactive_textures.push_back(retained);
    return 0;
}

extern "C" int mglRendererBackendCreateProactiveTexture(
    MGLRendererBackendHandle *backend)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying || !backend->device) return -1;

    MTL::TextureDescriptor *descriptor =
        MTL::TextureDescriptor::alloc()->init();
    if (!descriptor) return -1;
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    descriptor->setWidth(256u);
    descriptor->setHeight(256u);
    descriptor->setDepth(1u);
    descriptor->setMipmapLevelCount(1u);
    descriptor->setSampleCount(1u);
    descriptor->setArrayLength(1u);
    descriptor->setUsage(MTL::TextureUsageShaderRead |
                         MTL::TextureUsageRenderTarget);
    descriptor->setStorageMode(MTL::StorageModeShared);

    MTL::Texture *texture = backend->device->newTexture(descriptor);
    descriptor->release();
    if (!texture) return -1;

    std::vector<uint32_t> gradient(256u * 256u);
    for (uint32_t y = 0; y < 256u; ++y) {
        for (uint32_t x = 0; x < 256u; ++x) {
            const uint8_t r = static_cast<uint8_t>((x * 128u) / 256u + 64u);
            const uint8_t g = static_cast<uint8_t>((y * 128u) / 256u + 64u);
            gradient[y * 256u + x] =
                (UINT32_C(255) << 24) | (UINT32_C(255) << 16) |
                (static_cast<uint32_t>(g) << 8) | r;
        }
    }
    texture->replaceRegion(MTL::Region::Make2D(0u, 0u, 256u, 256u),
                           0u, gradient.data(), 256u * sizeof(uint32_t));
    backend->proactive_textures.push_back(texture);
    return 0;
}

extern "C" int mglRendererBackendIsDestroying(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return 0;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->destroying ? 1 : 0;
}

extern "C" void *mglRendererBackendGetOwner(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendOwnerKind kind)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    switch (kind) {
        case MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE:
            return backend->command_queue_owner;
        case MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER:
            return backend->command_buffer_owner;
        case MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER:
            return backend->render_encoder_owner;
        case MGL_RENDERER_BACKEND_OWNER_RENDER_PASS:
            return backend->render_pass_state_owner;
        case MGL_RENDERER_BACKEND_OWNER_QUERY:
            return backend->query_owner;
        case MGL_RENDERER_BACKEND_OWNER_RECOVERY:
            return backend->recovery_owner;
        case MGL_RENDERER_BACKEND_OWNER_BINDING:
            return backend->binding_owner;
    }
    return nullptr;
}

extern "C" int mglRendererBackendShutdown(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendShutdownResult *result_out)
{
    if (result_out) *result_out = {};
    if (!backend) return -1;

    void *command_owner = nullptr;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->shutdown_started) return 0;
        backend->shutdown_started = true;
        command_owner = backend->command_buffer_owner;
    }

    if (command_owner &&
        mglRenderCommandBufferOwnerHasLastSubmitted(command_owner) == 1) {
        MGLRenderCommandBufferState state = {};
        int wait_result = mglRenderWaitCommandBufferOwnerLastSubmitted(
            command_owner, &state);
        if (result_out) {
            result_out->waited_for_last_submission = 1;
            result_out->last_submission_has_error = state.has_error;
            result_out->last_submission_error_code = state.error_code;
            result_out->status = wait_result;
        }
        return wait_result;
    }
    return 0;
}

extern "C" void mglRendererBackendDestroy(
    MGLRendererBackendHandle **backend_ptr)
{
    if (!backend_ptr || !*backend_ptr) return;
    MGLRendererBackendHandle *backend = *backend_ptr;
    *backend_ptr = nullptr;
    void *platform_shell = nullptr;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->destroying) return;
        backend->destroying = true;
        if (backend->context) {
            platform_shell = backend->context->platform_renderer_shell;
        }
        if (backend->context && backend->context->renderer_backend == backend) {
            backend->context->renderer_backend = nullptr;
        }
    }
    if (platform_shell) {
        mglRendererPlatformBackendWillDestroy(platform_shell, backend);
    }
    (void)mglRendererBackendShutdown(backend, nullptr);
    mglRendererBackendReleaseOwnedState(backend);
    delete backend;
}

extern "C" void mglRendererBindBuffer(GLMContext context, Buffer *buffer)
{
    mglRenderBindBuffer(context, buffer);
}

extern "C" void mglRendererBindTexture(GLMContext context, Texture *texture)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRenderBindTexture(context, texture);
}

extern "C" void mglRendererBindProgram(GLMContext context, Program *program)
{
    mglRenderBindProgram(context, program);
}

extern "C" void mglRendererDeleteMetalObject(GLMContext context, void *object)
{
    mglRenderDeleteMTLObj(context, object);
}

extern "C" void mglRendererReleaseBufferMetalData(
    GLMContext context, Buffer *buffer)
{
    mglRenderReleaseBufferMetalData(context, buffer);
}

extern "C" void mglRendererGetSync(GLMContext context, Sync *sync)
{
    mglRenderGetSync(context, sync);
}

extern "C" void mglRendererWaitForSync(GLMContext context, Sync *sync)
{
    mglRenderWaitForSync(context, sync);
}

extern "C" uint32_t mglRendererGetSyncStatus(
    GLMContext context, Sync *sync)
{
    return mglRenderGetSyncStatus(context, sync);
}

extern "C" void mglRendererReleaseSync(GLMContext context, Sync *sync)
{
    mglRenderReleaseSync(context, sync);
}

extern "C" void mglRendererFlush(GLMContext context, bool finish)
{
    mglRenderFlush(context, finish);
}

extern "C" void mglRendererSwapBuffers(GLMContext context)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRenderSwapBuffers(context);
}

extern "C" void mglRendererFlushDrawBuffer(GLMContext context)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    mglRenderFlushDrawBuffer(context);
}

extern "C" int mglRendererProcessGLState(GLMContext context, int draw_command)
{
    return mglRenderProcessGLState(context, draw_command);
}

extern "C" void mglRendererInvalidateRenderPass(GLMContext context)
{
    mglRenderInvalidateRenderPass(context);
}

extern "C" void mglRendererClearBuffer(
    GLMContext context, uint32_t type, uint32_t mask)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRenderClearBuffer(context, type, mask);
}

extern "C" void mglRendererBlitFramebuffer(
    GLMContext context,
    int32_t src_x0, int32_t src_y0, int32_t src_x1, int32_t src_y1,
    int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1,
    uint32_t mask, uint32_t filter)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderBlitFramebuffer(context, src_x0, src_y0, src_x1, src_y1,
            dst_x0, dst_y0, dst_x1, dst_y1, mask, filter);
    }
}

extern "C" void mglRendererBufferSubData(
    GLMContext context, Buffer *buffer,
    size_t offset, size_t size, const void *bytes)
{
    mglRenderBufferSubData(context, buffer, offset, size, bytes);
}

extern "C" void *mglRendererMapUnmapBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size,
    uint32_t access, bool map)
{
    return mglRenderMapUnmapBuffer(
        context, buffer, offset, size, access, map);
}

extern "C" void mglRendererReadBackBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size)
{
    mglRenderReadBackBuffer(context, buffer, offset, size);
}

extern "C" void mglRendererFlushBufferRange(
    GLMContext context, Buffer *buffer, intptr_t offset, intptr_t length)
{
    mglRenderFlushBufferRange(context, buffer, offset, length);
}

extern "C" void mglRendererReadDrawable(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderReadDrawable(context, pixel_bytes, bytes_per_row, bytes_per_image,
            x, y, width, height);
    }
}

extern "C" void mglRendererReadIntegerPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderReadIntegerPixels(context, pixel_bytes, bytes_per_row, bytes_per_image,
            x, y, width, height, format, type);
    }
}

extern "C" void mglRendererReadDepthPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderReadDepthPixels(context, pixel_bytes, bytes_per_row, bytes_per_image,
            x, y, width, height);
    }
}

extern "C" void mglRendererGetTexImage(
    GLMContext context, Texture *texture, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderGetTexImage(context, texture, pixel_bytes,
            bytes_per_row, bytes_per_image, x, y, width, height,
            format, type, level, slice);
    }
}

extern "C" void mglRendererGenerateMipmaps(
    GLMContext context, Texture *texture)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    mglRenderGenerateMipmaps(context, texture);
}

extern "C" void mglRendererTexSubImage(
    GLMContext context, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch,
    size_t source_image_size, size_t source_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderTexSubImage(context, texture, buffer,
            source_offset, source_pitch, source_image_size, source_size,
            slice, level, width, height, depth,
            x_offset, y_offset, z_offset);
    }
}

extern "C" bool mglRendererTexSubImageBytes(
    GLMContext context, Texture *texture,
    const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    return platform_shell && mglRenderTexSubImageBytes(context, texture, bytes, bytes_size,
        source_offset, source_pitch, source_image_size,
        slice, level, width, height, depth,
        x_offset, y_offset, z_offset);
}

extern "C" void mglRendererCopyTexSubImage(
    GLMContext context, Texture *texture,
    uint32_t slice, int32_t level,
    int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderCopyTexSubImage(context, texture, slice, level, x_offset, y_offset,
            x, y, width, height);
    }
}

extern "C" void mglRendererCopyImageSubData(
    GLMContext context, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture,
    int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderCopyImageSubData(context, source_texture,
            source_level, source_x, source_y, source_z,
            destination_texture, destination_level,
            destination_x, destination_y, destination_z,
            width, height, depth);
    }
}

extern "C" void mglRendererDispatchCompute(
    GLMContext context, uint32_t groups_x,
    uint32_t groups_y, uint32_t groups_z)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderDispatchCompute(context, groups_x, groups_y, groups_z);
    }
}

extern "C" void mglRendererDispatchComputeIndirect(
    GLMContext context, intptr_t indirect)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRenderDispatchComputeIndirect(context, indirect);
    }
}

extern "C" void mglRendererBeginSampleQuery(
    GLMContext context, uint32_t target)
{
    mglRenderBeginSampleQueryCallback(context, target);
}

extern "C" uint64_t mglRendererEndSampleQuery(GLMContext context)
{
    return mglRenderEndSampleQueryCallback(context);
}

extern "C" void mglRendererBeginTimerQuery(GLMContext context)
{
    mglRenderBeginTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererEndTimerQuery(GLMContext context)
{
    return mglRenderEndTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererGetGPUTimestamp(GLMContext context)
{
    return mglRenderGetGPUTimestamp(context);
}

namespace {

bool mglRendererProgramResourceTypeIsSupported(int32_t type,
                                               bool include_separate)
{
    switch (type) {
        case _UNIFORM_BUFFER_RES:
        case _UNIFORM_CONSTANT_RES:
        case _STORAGE_BUFFER_RES:
        case _ATOMIC_COUNTER_RES:
        case _PUSH_CONSTANT_RES:
        case _STAGE_INPUT_RES:
        case _STAGE_OUTPUT_RES:
        case _SAMPLED_IMAGE_RES:
        case _STORAGE_IMAGE_RES:
            return true;
        case _SEPARATE_IMAGE_RES:
        case _SEPARATE_SAMPLERS_RES:
            return include_separate;
        default:
            return false;
    }
}

MGLShaderResource *mglRendererProgramResource(GLMContext context,
                                               int32_t stage,
                                               int32_t type,
                                               int32_t index,
                                               Program **program_out)
{
    if (program_out) *program_out = nullptr;
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return nullptr;
    }
    Program *program = mglResolveProgramForStageFromState(context, stage);
    if (!program) return nullptr;
    MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
    if (index < 0) return nullptr;
    if (type == _SAMPLED_IMAGE_RES) {
        int32_t ordinal = index;
        for (GLuint i = 0; i < list->count; i++) {
            MGLShaderResource *resource = &list->list[i];
            int32_t elements = resource->gl_array_size > 1
                ? resource->gl_array_size : 1;
            if (ordinal < elements) {
                if (program_out) *program_out = program;
                return resource;
            }
            ordinal -= elements;
        }
        return nullptr;
    }
    if (index >= static_cast<int32_t>(list->count)) return nullptr;
    if (program_out) *program_out = program;
    return &list->list[index];
}

}  // namespace

extern "C" uint32_t mglDeclaredTextureTypeFromResource(
    const MGLShaderResource *resource)
{
    return mglRenderTextureTypeForShaderResource(
        resource != nullptr,
        resource ? static_cast<uint32_t>(resource->image_dim) : 0u,
        resource ? static_cast<uint32_t>(resource->image_arrayed) : 0u,
        resource ? static_cast<uint32_t>(resource->image_multisampled) : 0u);
}

extern "C" uint32_t mglExpectedTextureTypeForResource(
    Program *program, int32_t stage, MGLShaderResource *resource)
{
    if (!program || !resource || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    return mglDeclaredTextureTypeFromResource(resource);
}

extern "C" uint32_t mglExpectedTextureDataKindForResource(
    Program *program, int32_t stage, MGLShaderResource *resource)
{
    if (!program || !resource || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return MGL_SHADER_TEXTURE_DATA_UNKNOWN;
    }
    return resource->texture_data_kind != MGL_SHADER_TEXTURE_DATA_UNKNOWN
        ? static_cast<uint32_t>(resource->texture_data_kind)
        : static_cast<uint32_t>(MGL_SHADER_TEXTURE_DATA_FLOAT);
}

extern "C" int32_t mglRendererGetProgramBindingCount(
    GLMContext context, int32_t stage, int32_t type)
{
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        !mglRendererProgramResourceTypeIsSupported(type, true)) {
        return 0;
    }
    Program *program = mglResolveProgramForStageFromState(context, stage);
    if (!program) return 0;
    MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
    if (type != _SAMPLED_IMAGE_RES) return static_cast<int32_t>(list->count);
    int32_t total = 0;
    for (GLuint i = 0; i < list->count; i++)
        total += list->list[i].gl_array_size > 1 ? list->list[i].gl_array_size : 1;
    return total;
}

extern "C" int32_t mglRendererGetProgramBinding(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    if (!mglRendererProgramResourceTypeIsSupported(type, true)) return 0;
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    if (!resource) return 0;
    if (type == _SAMPLED_IMAGE_RES) {
        Program *program = nullptr;
        MGLShaderResource *base = mglRendererProgramResource(context, stage, type, index, &program);
        if (base && program) {
            int32_t ordinal = index;
            MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
            for (GLuint i = 0; i < list->count; i++) {
                int32_t elements = list->list[i].gl_array_size > 1 ? list->list[i].gl_array_size : 1;
                if (&list->list[i] == base) {
                    return static_cast<int32_t>(base->binding + (ordinal < elements ? ordinal : 0));
                }
                ordinal -= elements;
            }
        }
    }
    return static_cast<int32_t>(resource->binding);
}

extern "C" int32_t mglRendererGetProgramGLBinding(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    if (!resource) return 0;
    if (type == _SAMPLED_IMAGE_RES) {
        Program *program = nullptr;
        MGLShaderResource *base = mglRendererProgramResource(context, stage, type, index, &program);
        if (base && program) {
            int32_t ordinal = index;
            MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
            for (GLuint i = 0; i < list->count; i++) {
                int32_t elements = list->list[i].gl_array_size > 1 ? list->list[i].gl_array_size : 1;
                if (&list->list[i] == base)
                    return static_cast<int32_t>(base->gl_binding + (ordinal < elements ? ordinal : 0));
                ordinal -= elements;
            }
        }
    }
    return static_cast<int32_t>(resource->gl_binding);
}

extern "C" int32_t mglRendererGetProgramLocation(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    switch (type) {
        case _UNIFORM_BUFFER_RES:
        case _UNIFORM_CONSTANT_RES:
        case _STORAGE_BUFFER_RES:
        case _ATOMIC_COUNTER_RES:
        case _PUSH_CONSTANT_RES:
        case _STAGE_INPUT_RES:
        case _SAMPLED_IMAGE_RES:
        case _STORAGE_IMAGE_RES:
            break;
        default:
            return 0;
    }
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<int32_t>(resource->location) : 0;
}

extern "C" size_t mglRendererGetProgramBindingRequiredSize(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<size_t>(resource->required_size) : 0u;
}

extern "C" intptr_t mglRendererGetProgramMetalBufferIndexForStage(
    GLMContext context, int32_t stage, uint32_t client_binding)
{
    static constexpr int32_t resource_types[] = {
        _UNIFORM_BUFFER_RES, _UNIFORM_CONSTANT_RES, _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES, _PUSH_CONSTANT_RES,
    };
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return static_cast<intptr_t>(client_binding);
    }
    Program *program = mglResolveProgramForStageFromState(context, stage);
    if (!program) return static_cast<intptr_t>(client_binding);
    for (int32_t type : resource_types) {
        MGLShaderResourceList *list =
            &program->shader_resources_list[stage][type];
        for (uint32_t i = 0; i < list->count; ++i) {
            MGLShaderResource *resource = &list->list[i];
            if (mglShouldSkipStageBufferResource(
                    program, stage, type, resource)) {
                continue;
            }
            if (mglClientBufferBindingForResource(type, resource) ==
                client_binding) {
                return static_cast<intptr_t>(mglMetalResourceSlot(resource));
            }
        }
    }
    return -1;
}

extern "C" size_t mglRendererGetProgramBindingRequiredSizeForStage(
    GLMContext context, int32_t stage, uint32_t client_binding)
{
    static constexpr int32_t resource_types[] = {
        _UNIFORM_BUFFER_RES, _UNIFORM_CONSTANT_RES, _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES, _PUSH_CONSTANT_RES,
    };
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES) return 0u;
    Program *program = mglResolveProgramForStageFromState(context, stage);
    if (!program) return 0u;
    size_t required = 0u;
    for (int32_t type : resource_types) {
        MGLShaderResourceList *list =
            &program->shader_resources_list[stage][type];
        for (uint32_t i = 0; i < list->count; ++i) {
            MGLShaderResource *resource = &list->list[i];
            if (mglShouldSkipStageBufferResource(
                    program, stage, type, resource) ||
                mglClientBufferBindingForResource(type, resource) !=
                    client_binding) {
                continue;
            }
            required = std::max(
                required, static_cast<size_t>(resource->required_size));
        }
    }
    return required;
}

extern "C" uint32_t mglRendererGetProgramDeclaredTextureType(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? mglDeclaredTextureTypeFromResource(resource) : 0u;
}

extern "C" uint32_t mglRendererGetProgramExpectedTextureType(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    Program *program = nullptr;
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, &program);
    return resource
        ? mglExpectedTextureTypeForResource(program, stage, resource)
        : 0u;
}

extern "C" uint32_t mglRendererGetProgramExpectedTextureDataKind(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    Program *program = nullptr;
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, &program);
    return resource
        ? mglExpectedTextureDataKindForResource(program, stage, resource)
        : static_cast<uint32_t>(MGL_SHADER_TEXTURE_DATA_UNKNOWN);
}
