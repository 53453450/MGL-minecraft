#include "mgl_renderer_backend.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <vector>

#include "glm_context.h"
#include "mgl_metal_cpp.h"
#include "mgl_program_resource.h"
#include "mgl_render_cpp.h"
#include "mgl_shader_resource.h"

extern "C" Program *mglResolveProgramForStageFromState(
    GLMContext context, int stage);
extern "C" void mglRendererPlatformBackendWillDestroy(
    void *platform_shell, MGLRendererBackendHandle *backend);
extern "C" void mglRendererCompatDispatchCompute(GLMContext context,
    unsigned int groups_x, unsigned int groups_y, unsigned int groups_z);
extern "C" void mglRendererCompatDispatchComputeIndirect(GLMContext context, intptr_t indirect);
extern "C" void mglRendererCompatDrawArrays(GLMContext context,
    uint32_t mode, int32_t first, int32_t count);
extern "C" void mglRendererCompatDrawElements(GLMContext context,
    uint32_t mode, int32_t count, uint32_t type, const void *indices);
extern "C" void mglRendererCompatDrawRangeElements(GLMContext context, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices);
extern "C" void mglRendererCompatDrawArraysInstanced(GLMContext context, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count);
extern "C" void mglRendererCompatDrawElementsInstanced(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count);
extern "C" void mglRendererCompatDrawElementsBaseVertex(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex);
extern "C" void mglRendererCompatDrawRangeElementsBaseVertex(GLMContext context, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex);
extern "C" void mglRendererCompatDrawElementsInstancedBaseVertex(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex);
extern "C" void mglRendererCompatDrawArraysIndirect(GLMContext context,
    uint32_t mode, const void *indirect);
extern "C" void mglRendererCompatDrawElementsIndirect(GLMContext context,
    uint32_t mode, uint32_t type, const void *indirect);
extern "C" void mglRendererCompatDrawArraysInstancedBaseInstance(GLMContext context, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count,
    uint32_t base_instance);
extern "C" void mglRendererCompatDrawElementsInstancedBaseInstance(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, uint32_t base_instance);
extern "C" void mglRendererCompatDrawElementsInstancedBaseVertexBaseInstance(GLMContext context, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex, uint32_t base_instance);
extern "C" void mglRendererCompatMultiDrawArrays(GLMContext context, uint32_t mode,
    const int32_t *firsts, const int32_t *counts, int32_t draw_count);
extern "C" void mglRendererCompatMultiDrawElements(GLMContext context, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count);
extern "C" void mglRendererCompatMultiDrawElementsBaseVertex(GLMContext context, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count, const int32_t *base_vertices);
extern "C" void mglRendererCompatMultiDrawArraysIndirect(GLMContext context, uint32_t mode,
    const void *indirect, int32_t draw_count, int32_t stride);
extern "C" void mglRendererCompatMultiDrawElementsIndirect(GLMContext context, uint32_t mode, uint32_t type,
    const void *indirect, int32_t draw_count, int32_t stride);
extern "C" void mglRendererCompatBindTexture(GLMContext context, Texture *texture);
extern "C" void mglRendererCompatFlushDrawBuffer(GLMContext context);
extern "C" void mglRendererCompatSwapBuffers(GLMContext context);
extern "C" void mglRendererCompatClearBuffer(GLMContext context,
    unsigned int type, unsigned int mask);
extern "C" void mglRendererCompatBlitFramebuffer(GLMContext context,
    int src_x0, int src_y0, int src_x1, int src_y1,
    int dst_x0, int dst_y0, int dst_x1, int dst_y1,
    unsigned int mask, unsigned int filter);
extern "C" void mglRendererCompatReadDrawable(GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height);
extern "C" void mglRendererCompatReadIntegerPixels(GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type);
extern "C" void mglRendererCompatReadDepthPixels(GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height);
extern "C" void mglRendererCompatGetTexImage(GLMContext context, Texture *texture,
    void *pixel_bytes, uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice);
extern "C" void mglRendererCompatGenerateMipmaps(GLMContext context, Texture *texture);
extern "C" void mglRendererCompatTexSubImage(GLMContext context, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset);
extern "C" bool mglRendererCompatTexSubImageBytes(GLMContext context, Texture *texture,
    const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset);
extern "C" void mglRendererCompatCopyTexSubImage(GLMContext context, Texture *texture,
    uint32_t slice, int32_t level, int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height);
extern "C" void mglRendererCompatCopyImageSubData(GLMContext context, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth);

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
    MTL::Texture *transient_depth_texture = nullptr;
    uint64_t transient_depth_texture_width = 0;
    uint64_t transient_depth_texture_height = 0;
    MTL::SamplerState *scaled_blit_nearest_sampler = nullptr;
    MTL::SamplerState *scaled_blit_linear_sampler = nullptr;
    MTL::DepthStencilState *clear_rect_depth_state = nullptr;
    MGLRendererBackendPassthroughCache geometry_passthrough;
    MGLRendererBackendPassthroughCache tess_evaluation_passthrough;
    MGLRendererBackendSamplerSnapshotCache sampler_snapshots;
    MTL::Buffer *tess_factor_buffer = nullptr;
    uint32_t tess_factor_patch_count = 0;
    std::array<float, 6> tess_factor_levels{};
    MTL::Buffer *tess_xfb_dummy_buffer = nullptr;
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
    if (backend->transient_depth_texture) {
        backend->transient_depth_texture->release();
        backend->transient_depth_texture = nullptr;
    }
    backend->transient_depth_texture_width = 0;
    backend->transient_depth_texture_height = 0;
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
    if (backend->tess_xfb_dummy_buffer) {
        backend->tess_xfb_dummy_buffer->release();
        backend->tess_xfb_dummy_buffer = nullptr;
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
    mglRenderCppDestroyCommandQueueOwner(&backend->command_queue_owner);
    backend->command_queue = nullptr;
    mglRenderCppBindingDestroy(backend->binding_owner);
    backend->binding_owner = nullptr;
    mglRenderCppDestroyQueryStateOwner(&backend->query_owner);
    mglRenderCppDestroyCommandRecoveryOwner(&backend->recovery_owner);
    backend->command_buffer_owner = nullptr;
    backend->render_encoder_owner = nullptr;
    backend->render_pass_state_owner = nullptr;
    if (backend->renderer_initialized) {
        mglRenderCppShutdown();
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
    if (mglRenderCppInit(info->objc_device) != 0) {
        backend->device->release();
        backend->device = nullptr;
        delete backend;
        return -1;
    }
    backend->renderer_initialized = true;

    backend->binding_owner =
        mglRenderCppBindingCreate(info->binding_slot_count);
    if (!backend->binding_owner ||
        mglRenderCppCreateQueryStateOwner(
            info->query_capacity, &backend->query_owner) != 0 ||
        mglRenderCppCreateCommandRecoveryOwner(
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
        ? mglRenderCppResetCommandQueueOwner(
              backend->command_queue_owner, max_command_buffers, &queue)
        : mglRenderCppCreateCommandQueueOwner(
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
        mglRenderCppCommandBufferOwnerHasLastSubmitted(command_owner) == 1) {
        MGLRenderCppCommandBufferState state = {};
        int wait_result = mglRenderCppWaitCommandBufferOwnerLastSubmitted(
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
    mglRenderCppBindBuffer(context, buffer);
}

extern "C" void mglRendererBindTexture(GLMContext context, Texture *texture)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRendererCompatBindTexture(context, texture);
}

extern "C" void mglRendererBindProgram(GLMContext context, Program *program)
{
    mglRenderCppBindProgram(context, program);
}

extern "C" void mglRendererDeleteMetalObject(GLMContext context, void *object)
{
    mglRenderCppDeleteMTLObj(context, object);
}

extern "C" void mglRendererReleaseBufferMetalData(
    GLMContext context, Buffer *buffer)
{
    mglRenderCppReleaseBufferMetalData(context, buffer);
}

extern "C" void mglRendererGetSync(GLMContext context, Sync *sync)
{
    mglRenderCppGetSync(context, sync);
}

extern "C" void mglRendererWaitForSync(GLMContext context, Sync *sync)
{
    mglRenderCppWaitForSync(context, sync);
}

extern "C" uint32_t mglRendererGetSyncStatus(
    GLMContext context, Sync *sync)
{
    return mglRenderCppGetSyncStatus(context, sync);
}

extern "C" void mglRendererReleaseSync(GLMContext context, Sync *sync)
{
    mglRenderCppReleaseSync(context, sync);
}

extern "C" void mglRendererFlush(GLMContext context, bool finish)
{
    mglRenderCppFlush(context, finish);
}

extern "C" void mglRendererSwapBuffers(GLMContext context)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRendererCompatSwapBuffers(context);
}

extern "C" void mglRendererFlushDrawBuffer(GLMContext context)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRendererCompatFlushDrawBuffer(context);
}

extern "C" void mglRendererInvalidateRenderPass(GLMContext context)
{
    mglRenderCppInvalidateRenderPass(context);
}

extern "C" void mglRendererClearBuffer(
    GLMContext context, uint32_t type, uint32_t mask)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRendererCompatClearBuffer(context, type, mask);
}

extern "C" void mglRendererBlitFramebuffer(
    GLMContext context,
    int32_t src_x0, int32_t src_y0, int32_t src_x1, int32_t src_y1,
    int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1,
    uint32_t mask, uint32_t filter)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatBlitFramebuffer(context, src_x0, src_y0, src_x1, src_y1,
            dst_x0, dst_y0, dst_x1, dst_y1, mask, filter);
    }
}

extern "C" void mglRendererBufferSubData(
    GLMContext context, Buffer *buffer,
    size_t offset, size_t size, const void *bytes)
{
    mglRenderCppBufferSubData(context, buffer, offset, size, bytes);
}

extern "C" void *mglRendererMapUnmapBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size,
    uint32_t access, bool map)
{
    return mglRenderCppMapUnmapBuffer(
        context, buffer, offset, size, access, map);
}

extern "C" void mglRendererReadBackBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size)
{
    mglRenderCppReadBackBuffer(context, buffer, offset, size);
}

extern "C" void mglRendererFlushBufferRange(
    GLMContext context, Buffer *buffer, intptr_t offset, intptr_t length)
{
    mglRenderCppFlushBufferRange(context, buffer, offset, length);
}

extern "C" void mglRendererReadDrawable(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatReadDrawable(context, pixel_bytes, bytes_per_row, bytes_per_image,
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
        mglRendererCompatReadIntegerPixels(context, pixel_bytes, bytes_per_row, bytes_per_image,
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
        mglRendererCompatReadDepthPixels(context, pixel_bytes, bytes_per_row, bytes_per_image,
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
        mglRendererCompatGetTexImage(context, texture, pixel_bytes,
            bytes_per_row, bytes_per_image, x, y, width, height,
            format, type, level, slice);
    }
}

extern "C" void mglRendererGenerateMipmaps(
    GLMContext context, Texture *texture)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRendererCompatGenerateMipmaps(context, texture);
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
        mglRendererCompatTexSubImage(context, texture, buffer,
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
    return platform_shell && mglRendererCompatTexSubImageBytes(context, texture, bytes, bytes_size,
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
        mglRendererCompatCopyTexSubImage(context, texture, slice, level, x_offset, y_offset,
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
        mglRendererCompatCopyImageSubData(context, source_texture,
            source_level, source_x, source_y, source_z,
            destination_texture, destination_level,
            destination_x, destination_y, destination_z,
            width, height, depth);
    }
}

extern "C" void mglRendererDrawArrays(
    GLMContext context, uint32_t mode, int32_t first, int32_t count)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRendererCompatDrawArrays(context, mode, first, count);
}

extern "C" void mglRendererDrawElements(
    GLMContext context, uint32_t mode, int32_t count,
    uint32_t type, const void *indices)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawElements(context, mode, count, type, indices);
    }
}

extern "C" void mglRendererDrawRangeElements(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawRangeElements(context, mode, start, end, count, type, indices);
    }
}

extern "C" void mglRendererDrawArraysInstanced(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawArraysInstanced(context, mode, first, count, instance_count);
    }
}

extern "C" void mglRendererDrawElementsInstanced(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawElementsInstanced(context, mode, count, type, indices, instance_count);
    }
}

extern "C" void mglRendererDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawElementsBaseVertex(context, mode, count, type, indices, base_vertex);
    }
}

extern "C" void mglRendererDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawRangeElementsBaseVertex(context, mode, start, end, count, type,
            indices, base_vertex);
    }
}

extern "C" void mglRendererDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawElementsInstancedBaseVertex(context, mode, count, type, indices,
            instance_count, base_vertex);
    }
}

extern "C" void mglRendererDrawArraysIndirect(
    GLMContext context, uint32_t mode, const void *indirect)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) mglRendererCompatDrawArraysIndirect(context, mode, indirect);
}

extern "C" void mglRendererDrawElementsIndirect(
    GLMContext context, uint32_t mode, uint32_t type, const void *indirect)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawElementsIndirect(context, mode, type, indirect);
    }
}

extern "C" void mglRendererDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawArraysInstancedBaseInstance(context, mode, first, count, instance_count, base_instance);
    }
}

extern "C" void mglRendererDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawElementsInstancedBaseInstance(context, mode, count, type, indices,
            instance_count, base_instance);
    }
}

extern "C" void mglRendererDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDrawElementsInstancedBaseVertexBaseInstance(context, mode, count, type, indices,
            instance_count, base_vertex, base_instance);
    }
}

extern "C" void mglRendererMultiDrawArrays(
    GLMContext context, uint32_t mode,
    const int32_t *firsts, const int32_t *counts, int32_t draw_count)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatMultiDrawArrays(context, mode, firsts, counts, draw_count);
    }
}

extern "C" void mglRendererMultiDrawElements(
    GLMContext context, uint32_t mode, const int32_t *counts,
    uint32_t type, const void *const *indices, int32_t draw_count)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatMultiDrawElements(context, mode, counts, type, indices, draw_count);
    }
}

extern "C" void mglRendererMultiDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, const int32_t *counts,
    uint32_t type, const void *const *indices, int32_t draw_count,
    const int32_t *base_vertices)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatMultiDrawElementsBaseVertex(context, mode, counts, type, indices,
            draw_count, base_vertices);
    }
}

extern "C" void mglRendererMultiDrawArraysIndirect(
    GLMContext context, uint32_t mode, const void *indirect,
    int32_t draw_count, int32_t stride)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatMultiDrawArraysIndirect(context, mode, indirect, draw_count, stride);
    }
}

extern "C" void mglRendererMultiDrawElementsIndirect(
    GLMContext context, uint32_t mode, uint32_t type,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatMultiDrawElementsIndirect(context, mode, type, indirect, draw_count, stride);
    }
}

extern "C" void mglRendererDispatchCompute(
    GLMContext context, uint32_t groups_x,
    uint32_t groups_y, uint32_t groups_z)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDispatchCompute(context, groups_x, groups_y, groups_z);
    }
}

extern "C" void mglRendererDispatchComputeIndirect(
    GLMContext context, intptr_t indirect)
{
    void *platform_shell = mglRendererBackendPlatformShell(context);
    if (platform_shell) {
        mglRendererCompatDispatchComputeIndirect(context, indirect);
    }
}

extern "C" void mglRendererBeginSampleQuery(
    GLMContext context, uint32_t target)
{
    mglRenderCppBeginSampleQueryCallback(context, target);
}

extern "C" uint64_t mglRendererEndSampleQuery(GLMContext context)
{
    return mglRenderCppEndSampleQueryCallback(context);
}

extern "C" void mglRendererBeginTimerQuery(GLMContext context)
{
    mglRenderCppBeginTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererEndTimerQuery(GLMContext context)
{
    return mglRenderCppEndTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererGetGPUTimestamp(GLMContext context)
{
    return mglRenderCppGetGPUTimestamp(context);
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
    if (index < 0 || index >= static_cast<int32_t>(list->count)) return nullptr;
    if (program_out) *program_out = program;
    return &list->list[index];
}

}  // namespace

extern "C" uint32_t mglDeclaredTextureTypeFromResource(
    const MGLShaderResource *resource)
{
    return mglRenderCppTextureTypeForShaderResource(
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
    return program ? static_cast<int32_t>(
        program->shader_resources_list[stage][type].count) : 0;
}

extern "C" int32_t mglRendererGetProgramBinding(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    if (!mglRendererProgramResourceTypeIsSupported(type, true)) return 0;
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<int32_t>(resource->binding) : 0;
}

extern "C" int32_t mglRendererGetProgramGLBinding(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<int32_t>(resource->gl_binding) : 0;
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
