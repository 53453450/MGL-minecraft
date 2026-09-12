/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#include "mgl_renderer_backend.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "glm_context.h"
#include "mgl_metal.h"
#include "mgl_program_resource.h"
#include "mgl_render.h"
#include "mgl_backend_handles.h"
#include "mgl_shader_resource.h"

extern "C" Program *mglResolveProgramForStageFromState(
    GLMContext context, int stage);
extern "C" void mglRendererPlatformBackendWillDestroy(
    void *platform_shell, MGLRendererBackendHandle *backend);
extern "C" void mglRendererDrawArrays(GLMContext context, uint32_t mode,
                                      int32_t first, int32_t count);
extern "C" void mglRendererDrawElements(GLMContext context, uint32_t mode,
                                        int32_t count, uint32_t type,
                                        const void *indices);
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

/* Packed current-value attrib pool (see
 * mglRendererBackendGetPackedCurrentAttribBuffer). */
struct MGLRendererBackendPackedCurrentAttribCacheEntry {
    MTL::Buffer *buffer = nullptr;
    std::vector<uint8_t> values;
    uint32_t repeat_count = 0;
    bool valid = false;
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
    MGLRendererBackendPackedCurrentAttribCacheEntry
        packed_current_attrib_cache{};
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
    uint64_t lease_generation = 1;
    uint64_t active_leases = 0;
    std::condition_variable lease_cv;
    /* Metal objects retired while leases still borrow them. Flushed when
     * active_leases reaches 0 so Set/Put/growth cannot UAF a borrower. */
    std::vector<NS::Object *> deferred_releases;
    /* Fake DrawExecutor is test-only; production never installs a vtable. */
    void *draw_executor = nullptr;
    const MGLDrawExecutorVTable *draw_executor_vt = nullptr;
};

struct MGLRendererBackendTLSLease {
    MGLRendererBackendHandle *backend = nullptr;
    uint64_t generation = 0;
    uint32_t depth = 0;
};

static thread_local MGLRendererBackendTLSLease g_backend_tls_lease;

static bool mglRendererBackendTLSHolds(
    const MGLRendererBackendHandle *backend)
{
    return backend &&
           g_backend_tls_lease.backend == backend &&
           g_backend_tls_lease.depth > 0 &&
           g_backend_tls_lease.generation != 0;
}

static bool mglRendererBackendLeaseMatches(
    const MGLRendererBackendLease *lease)
{
    return lease && lease->backend &&
           mglRendererBackendTLSHolds(lease->backend) &&
           g_backend_tls_lease.generation == lease->generation;
}

static void mglRendererBackendFlushDeferredReleases(
    MGLRendererBackendHandle *backend)
{
    if (!backend) return;
    for (NS::Object *object : backend->deferred_releases) {
        if (object) object->release();
    }
    backend->deferred_releases.clear();
}

static void mglRendererBackendDeferOrRelease(
    MGLRendererBackendHandle *backend, NS::Object *object)
{
    if (!backend || !object) return;
    if (backend->active_leases > 0u) {
        backend->deferred_releases.push_back(object);
        return;
    }
    object->release();
}

template <typename T>
static void mglRendererBackendReplaceObject(
    MGLRendererBackendHandle *backend, T *&slot, void *object)
{
    T *replacement = static_cast<T *>(object);
    if (replacement == slot) return;
    if (replacement) replacement->retain();
    if (slot) mglRendererBackendDeferOrRelease(backend, slot);
    slot = replacement;
}

static void mglRendererBackendReleaseOwnedState(
    MGLRendererBackendHandle *backend)
{
    if (!backend) return;
    mglRendererBackendFlushDeferredReleases(backend);
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
    if (backend->packed_current_attrib_cache.buffer) {
        backend->packed_current_attrib_cache.buffer->release();
    }
    backend->packed_current_attrib_cache = {};
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
    if (backend->draw_executor && backend->draw_executor_vt &&
        backend->draw_executor_vt->destroy) {
        backend->draw_executor_vt->destroy(backend->draw_executor);
    }
    backend->draw_executor = nullptr;
    backend->draw_executor_vt = nullptr;
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
    MGLRendererBackendHandle *backend,
    MGLRendererBackendPassthroughCache *cache,
    void *library, void *function, uint64_t program_instance_id)
{
    if (!backend || !cache) return;
    MTL::Library *new_library = static_cast<MTL::Library *>(library);
    MTL::Function *new_function = static_cast<MTL::Function *>(function);
    if (new_library) new_library->retain();
    if (new_function) new_function->retain();
    if (cache->function) {
        mglRendererBackendDeferOrRelease(backend, cache->function);
    }
    if (cache->library) {
        mglRendererBackendDeferOrRelease(backend, cache->library);
    }
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
    backend->draw_executor = nullptr;
    backend->draw_executor_vt = nullptr;
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

extern "C" int mglRendererBackendBegin(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendLease *lease_out)
{
    if (lease_out) {
        lease_out->backend = nullptr;
        lease_out->generation = 0;
    }
    if (!backend || !lease_out) return -1;

    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->destroying || backend->shutdown_started) return -1;
        generation = backend->lease_generation;
        backend->active_leases++;
    }

    if (g_backend_tls_lease.backend == backend &&
        g_backend_tls_lease.generation == generation &&
        g_backend_tls_lease.depth > 0u) {
        g_backend_tls_lease.depth++;
    } else if (g_backend_tls_lease.depth == 0u) {
        g_backend_tls_lease.backend = backend;
        g_backend_tls_lease.generation = generation;
        g_backend_tls_lease.depth = 1u;
    } else {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->active_leases > 0u) backend->active_leases--;
        if (backend->active_leases == 0u) {
            mglRendererBackendFlushDeferredReleases(backend);
            if (backend->destroying) backend->lease_cv.notify_all();
        }
        return -1;
    }

    lease_out->backend = backend;
    lease_out->generation = generation;
    return 0;
}

extern "C" int mglRendererBackendBeginContext(
    GLMContext context, MGLRendererBackendLease *lease_out)
{
    if (lease_out) {
        lease_out->backend = nullptr;
        lease_out->generation = 0;
    }
    if (!context) return -1;

    /* Attach lock covers load of renderer_backend through active_leases++
     * inside Begin, so Destroy cannot delete the handle in between. */
    if (context->renderer_backend_lock_initialized) {
        if (pthread_mutex_lock(&context->renderer_backend_lock) != 0) {
            return -1;
        }
    }
    MGLRendererBackendHandle *backend =
        static_cast<MGLRendererBackendHandle *>(context->renderer_backend);
    int result = backend ? mglRendererBackendBegin(backend, lease_out) : -1;
    if (context->renderer_backend_lock_initialized) {
        (void)pthread_mutex_unlock(&context->renderer_backend_lock);
    }
    return result;
}

extern "C" void mglRendererBackendEnd(MGLRendererBackendLease *lease)
{
    if (!lease || !lease->backend) return;
    MGLRendererBackendHandle *backend = lease->backend;
    const uint64_t generation = lease->generation;
    /* Clear the caller token first so a second End is a no-op. */
    lease->backend = nullptr;
    lease->generation = 0;

    const bool tls_match =
        g_backend_tls_lease.backend == backend &&
        g_backend_tls_lease.generation == generation &&
        g_backend_tls_lease.depth > 0u;

    if (tls_match) {
        g_backend_tls_lease.depth--;
        if (g_backend_tls_lease.depth == 0u) {
            g_backend_tls_lease.backend = nullptr;
            g_backend_tls_lease.generation = 0;
        }
    } else {
        /* Wrong thread/token: still drop the active_leases accounting so
         * Destroy cannot hang, and invalidate TLS borrows for this backend. */
        if (g_backend_tls_lease.backend == backend) {
            g_backend_tls_lease.backend = nullptr;
            g_backend_tls_lease.generation = 0;
            g_backend_tls_lease.depth = 0;
        }
        fprintf(stderr,
                "MGL WARNING: mglRendererBackendEnd TLS/token mismatch "
                "(backend=%p gen=%llu)\n",
                (void *)backend,
                (unsigned long long)generation);
    }

    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->active_leases > 0u) backend->active_leases--;
    if (backend->active_leases == 0u) {
        mglRendererBackendFlushDeferredReleases(backend);
        if (backend->destroying) backend->lease_cv.notify_all();
    }
}

extern "C" int mglRendererBackendThreadHoldsLease(
    const MGLRendererBackendHandle *backend)
{
    return mglRendererBackendTLSHolds(backend) ? 1 : 0;
}

extern "C" void *mglRendererBackendLeaseGetDevice(
    const MGLRendererBackendLease *lease)
{
    if (!mglRendererBackendLeaseMatches(lease)) return nullptr;
    return mglRendererBackendGetDevice(lease->backend);
}

extern "C" void *mglRendererBackendLeaseGetCommandQueue(
    const MGLRendererBackendLease *lease)
{
    if (!mglRendererBackendLeaseMatches(lease)) return nullptr;
    return mglRendererBackendGetCommandQueue(lease->backend);
}

extern "C" void *mglRendererBackendLeaseGetOwner(
    const MGLRendererBackendLease *lease,
    MGLRendererBackendOwnerKind kind)
{
    if (!mglRendererBackendLeaseMatches(lease)) return nullptr;
    return mglRendererBackendGetOwner(lease->backend, kind);
}

extern "C" void *mglRendererBackendGetDevice(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    /* Lease holders may observe pointers while destroy drains active leases. */
    return backend->device;
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
    if (backend->command_queue && backend->active_leases > 0u) {
        backend->command_queue->retain();
        backend->deferred_releases.push_back(backend->command_queue);
    }
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
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    mglRendererBackendReplaceObject(backend,
        backend->fallback_render_target_texture, texture);
    return 0;
}

extern "C" void *mglRendererBackendGetFallbackRenderTargetTexture(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->fallback_render_target_texture;
}

extern "C" void *mglRendererBackendGetFallbackBindingBuffer(
    MGLRendererBackendHandle *backend, uint64_t minimum_length)
{
    if (!backend || minimum_length == 0u ||
        !mglRendererBackendTLSHolds(backend)) return nullptr;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (!backend->device) return nullptr;
    if (backend->destroying) {
        return (backend->fallback_binding_buffer &&
                backend->fallback_binding_buffer_length >= minimum_length)
            ? backend->fallback_binding_buffer
            : nullptr;
    }
    if (!backend->fallback_binding_buffer ||
        backend->fallback_binding_buffer_length < minimum_length) {
        MTL::Buffer *replacement = backend->device->newBuffer(
            static_cast<NS::UInteger>(minimum_length),
            MTL::ResourceStorageModeShared);
        if (!replacement) return nullptr;
        if (backend->fallback_binding_buffer) {
            mglRendererBackendDeferOrRelease(
                backend, backend->fallback_binding_buffer);
        }
        backend->fallback_binding_buffer = replacement;
        backend->fallback_binding_buffer_length = minimum_length;
    }
    return backend->fallback_binding_buffer;
}

extern "C" void *mglRendererBackendGetCullDistanceDummyBuffer(
    MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (!backend->device) return nullptr;
    if (backend->destroying) return backend->cull_distance_dummy_buffer;
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
    mglRendererBackendReplaceObject(backend, backend->transient_depth_texture, texture);
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
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
            mglRendererBackendReplaceObject(backend,
                backend->default_draw_buffer_colors[draw_buffer_index], texture);
            return 0;
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH:
            mglRendererBackendReplaceObject(backend,
                backend->default_draw_buffer_depths[draw_buffer_index], texture);
            return 0;
        case MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL:
            mglRendererBackendReplaceObject(backend,
                backend->default_draw_buffer_stencils[draw_buffer_index], texture);
            return 0;
    }
    return -1;
}

extern "C" void *mglRendererBackendGetDefaultDrawBufferAttachment(
    const MGLRendererBackendHandle *backend, uint32_t draw_buffer_index,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind)
{
    if (!backend || draw_buffer_index >= 6u ||
        !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    mglRendererBackendReplaceObject(backend,
        backend->default_draw_buffer_colors[draw_buffer_index], nullptr);
    mglRendererBackendReplaceObject(backend,
        backend->default_draw_buffer_depths[draw_buffer_index], nullptr);
    mglRendererBackendReplaceObject(backend,
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
    mglRendererBackendReplaceObject(backend, list_it->slots[slot].temporary, temporary);
    mglRendererBackendReplaceObject(backend, list_it->slots[slot].destination, destination);
    return 0;
}

extern "C" int mglRendererBackendGetStageCopyBackResources(
    const MGLRendererBackendHandle *backend, const void *copy_back_list_key,
    uint32_t slot, void **temporary_out, void **destination_out)
{
    if (temporary_out) *temporary_out = nullptr;
    if (destination_out) *destination_out = nullptr;
    if (!backend || !mglRendererBackendTLSHolds(backend) ||
        !copy_back_list_key || slot >= 31u ||
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
    mglRendererBackendReplaceObject(backend, list_it->slots[slot].temporary, nullptr);
    mglRendererBackendReplaceObject(backend, list_it->slots[slot].destination, nullptr);
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
    if (!backend || !mglRendererBackendTLSHolds(backend) ||
        attrib >= MAX_ATTRIBS || !bytes ||
        byte_count == 0u || byte_count > 16u || stride == 0u) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
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
    mglRendererBackendReplaceObject(backend, entry.buffer, buffer);
    entry.bytes = {};
    std::memcpy(entry.bytes.data(), bytes, byte_count);
    entry.byte_count = byte_count;
    entry.stride = stride;
    return 0;
}

extern "C" void *mglRendererBackendGetPackedCurrentAttribBuffer(
    const MGLRendererBackendHandle *backend, const void *bytes,
    uint32_t byte_count, uint32_t repeat_count)
{
    if (!backend || !mglRendererBackendTLSHolds(backend) || !bytes ||
        byte_count == 0u || repeat_count == 0u) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    const MGLRendererBackendPackedCurrentAttribCacheEntry &entry =
        backend->packed_current_attrib_cache;
    if (!entry.valid || !entry.buffer || entry.repeat_count != repeat_count ||
        entry.values.size() != byte_count ||
        std::memcmp(entry.values.data(), bytes, byte_count) != 0) {
        return nullptr;
    }
    return entry.buffer;
}

extern "C" int mglRendererBackendSetPackedCurrentAttribBuffer(
    MGLRendererBackendHandle *backend, const void *bytes,
    uint32_t byte_count, uint32_t repeat_count, void *buffer)
{
    if (!backend || !bytes || byte_count == 0u || repeat_count == 0u ||
        !buffer) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    MGLRendererBackendPackedCurrentAttribCacheEntry &entry =
        backend->packed_current_attrib_cache;
    mglRendererBackendReplaceObject(backend, entry.buffer, buffer);
    entry.values.assign(static_cast<const uint8_t *>(bytes),
                        static_cast<const uint8_t *>(bytes) + byte_count);
    entry.repeat_count = repeat_count;
    entry.valid = true;
    return 0;
}

extern "C" void *mglRendererBackendGetSizeConstantsBuffer(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendSizeConstantsStage stage,
    const uint32_t *constants, uint32_t count)
{
    if (!backend || !mglRendererBackendTLSHolds(backend) ||
        stage < MGL_RENDERER_BACKEND_SIZE_CONSTANTS_VERTEX ||
        stage > MGL_RENDERER_BACKEND_SIZE_CONSTANTS_FRAGMENT ||
        !constants || count != 31u) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
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
    mglRendererBackendReplaceObject(backend, entry.buffer, buffer);
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
            mglRendererBackendReplaceObject(backend,
                backend->scaled_blit_nearest_sampler, object);
            return 0;
        case MGL_RENDERER_BACKEND_BLIT_CACHE_LINEAR_SAMPLER:
            mglRendererBackendReplaceObject(backend,
                backend->scaled_blit_linear_sampler, object);
            return 0;
        case MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE:
            mglRendererBackendReplaceObject(backend,
                backend->clear_rect_depth_state, object);
            return 0;
    }
    return -1;
}

extern "C" void *mglRendererBackendGetBlitCachedObject(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendBlitCacheKind kind)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
        backend, cache, library, function, program_instance_id);
    return 0;
}

extern "C" int mglRendererBackendGetPassthroughFunction(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendPassthroughKind kind,
    uint64_t program_instance_id, void **function_out)
{
    if (function_out) *function_out = nullptr;
    if (!backend || !mglRendererBackendTLSHolds(backend) || !function_out) return -1;
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
    if (!backend || !mglRendererBackendTLSHolds(backend) || !key || !state_out) return -1;
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
        mglRendererBackendReplaceObject(backend, 
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
    if (cache.states[slot]) {
        mglRendererBackendDeferOrRelease(backend, cache.states[slot]);
    }
    cache.keys[slot] = *key;
    cache.states[slot] = replacement;
    if (mglRendererBackendInsertSamplerSnapshotIndex(cache, key, slot) != 0) {
        mglRendererBackendDeferOrRelease(backend, replacement);
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
    if (!backend || !mglRendererBackendTLSHolds(backend) ||
        patch_count == 0u || !levels || !buffer_out) return -1;
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
    mglRendererBackendReplaceObject(backend, backend->tess_factor_buffer, buffer);
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
    mglRendererBackendReplaceObject(backend,
        backend->current_tess_factor_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetCurrentTessFactorBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->current_tess_factor_buffer;
}

extern "C" int mglRendererBackendGetTessXfbDummyBuffer(
    const MGLRendererBackendHandle *backend, uint64_t minimum_length,
    void **buffer_out)
{
    if (buffer_out) *buffer_out = nullptr;
    if (!backend || !mglRendererBackendTLSHolds(backend) ||
        minimum_length == 0u || !buffer_out) return -1;
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
    mglRendererBackendReplaceObject(backend, backend->tess_xfb_dummy_buffer, buffer);
    return 0;
}

extern "C" int mglRendererBackendSetCullDistanceCaptureBuffer(
    MGLRendererBackendHandle *backend, void *buffer)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (backend->destroying) return -1;
    mglRendererBackendReplaceObject(backend,
        backend->cull_distance_capture_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetCullDistanceCaptureBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    mglRendererBackendReplaceObject(backend,
        backend->tess_control_point_index_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTessControlPointIndexBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    mglRendererBackendReplaceObject(backend,
        backend->tess_vertex_capture_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTessVertexCaptureBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    mglRendererBackendReplaceObject(backend, backend->tcs_patch_out_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTcsPatchOutBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    mglRendererBackendReplaceObject(backend, backend->tcs_output_buffer, buffer);
    return 0;
}

extern "C" void *mglRendererBackendGetTcsOutputBuffer(
    const MGLRendererBackendHandle *backend)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
            mglRendererBackendReplaceObject(backend,
                backend->fallback_sampled_texture, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE:
            mglRendererBackendReplaceObject(backend,
                backend->fallback_cube_sampled_texture, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE:
            mglRendererBackendReplaceObject(backend,
                backend->fallback_texture_buffer_storage, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER:
            mglRendererBackendReplaceObject(backend,
                backend->fallback_sint_texture_buffer, resource);
            return 0;
        case MGL_RENDERER_BACKEND_FALLBACK_SAMPLER:
            mglRendererBackendReplaceObject(backend,
                backend->fallback_sampler, resource);
            return 0;
    }
    return -1;
}

extern "C" void *mglRendererBackendGetFallbackResource(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendFallbackResourceKind kind)
{
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    if (!backend || !mglRendererBackendTLSHolds(backend) || !texture_out) return -1;
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
            mglRendererBackendReplaceObject(backend, entry.texture, texture);
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
    if (!backend || !mglRendererBackendTLSHolds(backend)) return nullptr;
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
    GLMContext context = backend->context;

    /* Detach from context under the attach lock before waiting/deleting so
     * BeginContext cannot observe this handle after teardown starts. */
    if (context && context->renderer_backend_lock_initialized) {
        if (pthread_mutex_lock(&context->renderer_backend_lock) == 0) {
            if (context->renderer_backend == backend) {
                context->renderer_backend = nullptr;
            }
            platform_shell = context->platform_renderer_shell;
            (void)pthread_mutex_unlock(&context->renderer_backend_lock);
        }
    } else if (context) {
        platform_shell = context->platform_renderer_shell;
        if (context->renderer_backend == backend) {
            context->renderer_backend = nullptr;
        }
    }

    {
        std::unique_lock<std::mutex> lock(backend->mutex);
        if (backend->destroying) return;
        backend->destroying = true;
        backend->lease_generation++;
        backend->lease_cv.wait(lock, [backend] {
            return backend->active_leases == 0u;
        });
    }
    if (platform_shell) {
        mglRendererPlatformBackendWillDestroy(platform_shell, backend);
    }
    (void)mglRendererBackendShutdown(backend, nullptr);
    mglRendererBackendReleaseOwnedState(backend);
    delete backend;
}

namespace {

struct MGLBackendLeaseScope {
    MGLRendererBackendLease lease{};
    bool held = false;
    bool allowed_without_backend = false;

    explicit MGLBackendLeaseScope(GLMContext context)
    {
        if (!context) {
            allowed_without_backend = true;
            return;
        }
        held = mglRendererBackendBeginContext(context, &lease) == 0;
        if (held) return;
        if (context->renderer_backend_lock_initialized &&
            pthread_mutex_lock(&context->renderer_backend_lock) == 0) {
            allowed_without_backend = context->renderer_backend == nullptr;
            (void)pthread_mutex_unlock(&context->renderer_backend_lock);
        } else {
            allowed_without_backend = context->renderer_backend == nullptr;
        }
    }

    bool ok() const { return held || allowed_without_backend; }

    ~MGLBackendLeaseScope()
    {
        if (held) mglRendererBackendEnd(&lease);
    }

    MGLBackendLeaseScope(const MGLBackendLeaseScope &) = delete;
    MGLBackendLeaseScope &operator=(const MGLBackendLeaseScope &) = delete;
};

}  // namespace

extern "C" void mglRendererBindBuffer(GLMContext context, Buffer *buffer)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderBindBuffer(context, buffer);
}

extern "C" void mglRendererBindProgram(GLMContext context, Program *program)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderBindProgram(context, program);
}

extern "C" void mglRendererDeleteMetalObject(GLMContext context, void *object)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderDeleteMTLObj(context, object);
}

extern "C" void mglRendererReleaseBufferMetalData(
    GLMContext context, Buffer *buffer)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderReleaseBufferMetalData(context, buffer);
}

extern "C" void mglRendererGetSync(GLMContext context, Sync *sync)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderGetSync(context, sync);
}

extern "C" void mglRendererWaitForSync(GLMContext context, Sync *sync)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderWaitForSync(context, sync);
}

extern "C" uint32_t mglRendererGetSyncStatus(
    GLMContext context, Sync *sync)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return 0;
    return mglRenderGetSyncStatus(context, sync);
}

extern "C" void mglRendererReleaseSync(GLMContext context, Sync *sync)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderReleaseSync(context, sync);
}

extern "C" void mglRendererFlush(GLMContext context, bool finish)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderFlush(context, finish);
}

extern "C" void mglRendererInvalidateRenderPass(GLMContext context)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderInvalidateRenderPass(context);
}

extern "C" void mglRendererBufferSubData(
    GLMContext context, Buffer *buffer,
    size_t offset, size_t size, const void *bytes)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderBufferSubData(context, buffer, offset, size, bytes);
}

extern "C" void *mglRendererMapUnmapBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size,
    uint32_t access, bool map)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return nullptr;
    return mglRenderMapUnmapBuffer(
        context, buffer, offset, size, access, map);
}

extern "C" void mglRendererReadBackBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderReadBackBuffer(context, buffer, offset, size);
}

extern "C" void mglRendererFlushBufferRange(
    GLMContext context, Buffer *buffer, intptr_t offset, intptr_t length)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderFlushBufferRange(context, buffer, offset, length);
}

extern "C" void mglRendererBeginSampleQuery(
    GLMContext context, uint32_t target)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderBeginSampleQueryCallback(context, target);
}

extern "C" uint64_t mglRendererEndSampleQuery(GLMContext context)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return 0;
    return mglRenderEndSampleQueryCallback(context);
}

extern "C" void mglRendererBeginTimerQuery(GLMContext context)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return;
    mglRenderBeginTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererEndTimerQuery(GLMContext context)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return 0;
    return mglRenderEndTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererGetGPUTimestamp(GLMContext context)
{
    MGLBackendLeaseScope lease(context);
    if (!lease.ok()) return 0;
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
    if (type == _SAMPLED_IMAGE_RES || type == _STORAGE_IMAGE_RES) {
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
    if (type != _SAMPLED_IMAGE_RES && type != _STORAGE_IMAGE_RES)
        return static_cast<int32_t>(list->count);
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
    if (type == _SAMPLED_IMAGE_RES || type == _STORAGE_IMAGE_RES) {
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
    if (type == _SAMPLED_IMAGE_RES || type == _STORAGE_IMAGE_RES) {
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
            if (mglClientBufferBindingForResource(type, resource) !=
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
