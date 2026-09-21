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

int mglRenderQueryCapability(void* device_ref,
                                MGLRenderCapabilityState* state_out) {
    if (!device_ref || !state_out) return -1;
    MTL::Device* device = static_cast<MTL::Device*>(device_ref);
    if (!device) return -1;

    MGLRenderCapabilityState state = {};
    NS::String* name_string = device->name();
    const char* name = name_string ? name_string->utf8String() : nullptr;
    /* GitHub macos-26 runners expose "Apple Paravirtual device".  The old
     * strstr("AGX") check never matched either Paravirt or real M-series
     * marketing names ("Apple M4"), so is_virtualized stayed 0 on CI. */
    const bool virtualized =
        name && (std::strstr(name, "Paravirtual") != nullptr ||
                 std::strstr(name, "paravirtual") != nullptr);
    const bool apple_family = device->supportsFamily(MTL::GPUFamilyApple1);
    const bool apple_name = name && std::strncmp(name, "Apple ", 6) == 0;

    if (virtualized) {
        state.family = MGL_GPU_FAMILY_VIRTUALIZED;
        state.is_virtualized = 1;
    } else if (apple_family || apple_name) {
        state.family = MGL_GPU_FAMILY_AGX;
    } else {
        state.family = MGL_GPU_FAMILY_OTHER;
    }

    static constexpr uint64_t sample_counts[] = {32u, 16u, 8u, 4u, 2u};
    state.max_sample_count = 1;
    for (uint64_t sample_count : sample_counts) {
        if (device->supportsTextureSampleCount(
                static_cast<NS::UInteger>(sample_count))) {
            state.max_sample_count = sample_count;
            break;
        }
    }
    state.supports8x_msaa = state.max_sample_count >= 8 ? 1u : 0u;

    const bool agx = state.family == MGL_GPU_FAMILY_VIRTUALIZED ||
                     state.family == MGL_GPU_FAMILY_AGX;
    if (agx) {
        /* 3d_getbytes_slice_oob / 3d_replace_region_nonzero_origin used to gate
         * a buffer-mediated 3D copy fallback.  Independent Metal probes showed
         * no driver defect; the fallback only papered over MGL reading empty
         * Metal storage while the CPU shadow held the pixels.  Native blit now
         * pushes the CPU-authoritative region into the source first — do not
         * re-enable these markers without a fresh driver-side reproduction.
         * See docs/AGX_COPY3D_DRIVER_BUG_RECHECK_2026-09-20.md. */
        state.bug_3d_getbytes_slice_oob = 0;
        state.bug_3d_replace_region_nonzero_origin = 0;
        state.bug_msl_pipeline_rejection = 1;
        state.conservative_cpu_cache_mode = 1;
        state.max_concurrent_command_buffers =
            state.is_virtualized ? 16u : 64u;
    } else {
        state.max_concurrent_command_buffers = 64u;
    }
    state.texture_alignment_bytes = 256u;
    state.command_buffer_recovery_limit = 4096u;
    *state_out = state;
    return 0;
}

unsigned int mglRenderGetSyncStatus(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync || !sync->mtl_command_buffer) return GL_SIGNALED;
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(sync->mtl_command_buffer);
    return commandBuffer->status() == MTL::CommandBufferStatusCompleted
        ? GL_SIGNALED
        : GL_UNSIGNALED;
}

uint64_t mglRenderGetGPUTimestamp(GLMContext glm_ctx) {
    if (!glm_ctx) return 0;

    /* The GL semantic layer establishes the ordering boundary before entering
     * this callback. Sampling itself is entirely C++ and does not need the
     * ObjC renderer bridge. */
    uint64_t cpu_timestamp = 0;
    uint64_t gpu_timestamp = 0;
    return mglRenderSampleTimestamps(
               &cpu_timestamp, &gpu_timestamp) == 0
        ? gpu_timestamp : 0;
}





int mglRenderIsSmallRGBA8(uint32_t width, uint32_t height, uint32_t internalformat) {
    return width <= 512u && height <= 512u && internalformat == GL_RGBA8 ? 1 : 0;
}

int mglRenderIsValidGLCompareFunction(uint32_t func) {
    switch (func) {
    case GL_NEVER:
    case GL_LESS:
    case GL_EQUAL:
    case GL_LEQUAL:
    case GL_GREATER:
    case GL_NOTEQUAL:
    case GL_GEQUAL:
    case GL_ALWAYS:
        return 1;
    default:
        return 0;
    }
}

int mglRenderIsValidGLBlendEquation(uint32_t op) {
    uint32_t tmp = 0u;
    return mglRenderBlendOperationFromGL(op, &tmp);
}

int mglRenderIsValidGLBlendFactor(uint32_t factor) {
    uint32_t tmp = 0u;
    return mglRenderBlendFactorFromGL(factor, &tmp);
}

int mglRenderGetDeviceIdentity(const void *device,
                                  uint64_t *registry_id_out,
                                  char *name_out,
                                  size_t name_capacity) {
    if (registry_id_out) *registry_id_out = 0u;
    if (name_out && name_capacity) name_out[0] = '\0';
    const MTL::Device *metal_device =
        static_cast<const MTL::Device *>(device);
    if (!metal_device) return -1;
    if (registry_id_out) *registry_id_out = metal_device->registryID();
    if (name_out && name_capacity) {
        NS::String *name = metal_device->name();
        const char *utf8 = name ? name->utf8String() : nullptr;
        if (utf8) {
            std::snprintf(name_out, name_capacity, "%s", utf8);
        }
    }
    return 0;
}

int mglRenderIsEmulatedMSColorTexture(uint32_t target, int32_t samples) {
    if (target != GL_TEXTURE_2D_MULTISAMPLE &&
        target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        return 0;
    }
    return samples > 1 ? 1 : 0;
}

int mglRenderPlanDirtyDomains(uint32_t dirty_bits, int draw_command,
                              int has_pipeline, int fbo_binding_dirty,
                              MGLDirtyDomainPlan *out) {
    if (!out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (dirty_bits == 0u) {
        out->dirty_buffer_data = 1;
        return 0;
    }
    out->has_dirty = 1;
    uint32_t bits = dirty_bits;
    if (bits & DIRTY_FBO) {
        out->sync_render_pass = 1;
    }
    if (bits & DIRTY_STATE) {
        if ((bits & DIRTY_FBO) && fbo_binding_dirty) {
            out->bind_fbo_attachments = 1;
        }
        bits &= ~DIRTY_STATE;
    }
    if (bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_BUFFER_BASE_STATE)) {
        out->remap_buffers = 1;
        if (draw_command && !has_pipeline && (bits & DIRTY_PROGRAM)) {
            out->defer_buffer_map = 1;
        }
        bits &= ~DIRTY_BUFFER_BASE_STATE;
    }
    if (bits & (DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER)) {
        out->bind_textures = 1;
        bits &= ~(DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER);
    }
    if (bits & DIRTY_VAO) {
        out->vao_path = 1;
        bits &= ~DIRTY_RENDER_STATE;
    } else if (bits & DIRTY_BUFFER) {
        out->buffer_path = 1;
        bits &= ~DIRTY_BUFFER;
    } else if (bits & DIRTY_RENDER_STATE) {
        out->render_state_path = 1;
        bits &= ~DIRTY_RENDER_STATE;
    }
    if (bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_ALPHA_STATE |
                DIRTY_RENDER_STATE)) {
        out->sync_pipeline = 1;
    }
    return 0;
}

int mglRenderIsInitialized(void) {
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    return renderer.device && renderer.users > 0 ? 1 : 0;
}

void mglRenderGetSync(GLMContext glm_ctx, Sync* sync) {
    if (!sync) return;
    BackendLeaseScope lease(glm_ctx);

    mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    mgl::releaseBridgedObject(&sync->mtl_event);
    MGLCommandBufferOwner* command_owner =
        static_cast<MGLCommandBufferOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER));
    if (!command_owner) return;

    MGLRenderEncoderOwner* render_owner =
        static_cast<MGLRenderEncoderOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER));
    if (render_owner &&
        mglRenderEncoderOwnerHasCurrent(render_owner) == 1 &&
        mglRenderEndRenderEncoderOwner(render_owner) != 0) {
        return;
    }

    MGLRenderCommandBufferState state = {};
    if (mglRenderGetCommandBufferOwnerState(command_owner, &state) != 0 ||
        state.status !=
            static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued) ||
        state.has_error) {
        void* next = nullptr;
        (void)mglRenderCommandBufferOwnerCreateNext(command_owner, &next);
        return;
    }

    void* submission = nullptr;
    void* command_buffer = nullptr;
    if (mglRenderTakeCommandBufferSubmission(
            command_owner, &submission, &command_buffer) != 0 ||
        !submission || !command_buffer) {
        mglRenderDestroyCommandBufferSubmission(&submission);
        return;
    }

    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    command->retain();
    sync->mtl_command_buffer = command;

    MGLRenderCommandBufferTransaction transaction = {};
    int result = mglRenderCommitCommandBufferTransaction(
        command_owner, &submission, command_buffer,
        static_cast<MGLCommandBufferRecoveryOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_RECOVERY)),
        0u, &transaction);
    mglRenderDestroyCommandBufferSubmission(&submission);
    if (result != 0 &&
        transaction.result !=
            MGL_RENDER_COMMAND_BUFFER_TRANSACTION_COMMITTED) {
        mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    }
}

int mglRenderCreateQueryStateOwner(uint32_t visibility_slot_count, MGLQueryStateOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out || visibility_slot_count == 0) return -1;
    mgl::QueryStateOwner* owner =
        new (std::nothrow) mgl::QueryStateOwner();
    if (!owner) return -1;
    owner->visibilitySlotCount = visibility_slot_count;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

int mglRenderBeginSampleQuery(MGLQueryStateOwner * owner_handle, uint32_t counting, const char* buffer_label, void** visibility_buffer_out) {
    if (visibility_buffer_out) *visibility_buffer_out = nullptr;
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !visibility_buffer_out ||
        owner->visibilitySlotCount == 0) {
        return -1;
    }
    if (!owner->visibilityBuffer) {
        mgl::Renderer& renderer = mgl::renderer();
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (!renderer.device) return -1;
        const uint64_t byteLength =
            static_cast<uint64_t>(owner->visibilitySlotCount) *
            sizeof(uint64_t);
        owner->visibilityBuffer = renderer.device->newBuffer(
            static_cast<NS::UInteger>(byteLength),
            MTL::ResourceStorageModeShared);
        if (!owner->visibilityBuffer) return -1;
        if (buffer_label && buffer_label[0]) {
            owner->visibilityBuffer->setLabel(
                NS::String::string(buffer_label, NS::UTF8StringEncoding));
        }
    }

    std::memset(owner->visibilityBuffer->contents(), 0,
                owner->visibilityBuffer->length());
    owner->sampleQueryActive = true;
    owner->sampleQueryCounting = counting != 0;
    owner->nextVisibilitySlot = 0;
    *visibility_buffer_out = owner->visibilityBuffer;
    return 0;
}

void mglRenderEndSampleQuery(MGLQueryStateOwner * owner_handle) {
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (owner) owner->sampleQueryActive = false;
}

int mglRenderIsSampleQueryActive(MGLQueryStateOwner * owner_handle, uint32_t* active_out) {
    if (active_out) *active_out = 0;
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !active_out) return -1;
    *active_out = owner->sampleQueryActive ? 1u : 0u;
    return 0;
}

int mglRenderAcquireSampleQuerySlot(MGLQueryStateOwner * owner_handle, uint32_t* mode_out, uint64_t* offset_out) {
    if (mode_out) *mode_out = 0;
    if (offset_out) *offset_out = 0;
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !mode_out || !offset_out ||
        !owner->sampleQueryActive || owner->visibilitySlotCount == 0) {
        return -1;
    }
    uint32_t slot = owner->nextVisibilitySlot;
    if (slot >= owner->visibilitySlotCount) {
        slot = owner->visibilitySlotCount - 1;
    } else {
        owner->nextVisibilitySlot++;
    }
    *mode_out = static_cast<uint32_t>(
        owner->sampleQueryCounting
            ? MTL::VisibilityResultModeCounting
            : MTL::VisibilityResultModeBoolean);
    *offset_out = static_cast<uint64_t>(slot) * sizeof(uint64_t);
    return 0;
}

int mglRenderGetSampleQueryResult(MGLQueryStateOwner * owner_handle, uint64_t* result_out) {
    if (result_out) *result_out = 0;
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !result_out || !owner->visibilityBuffer) return -1;
    const uint64_t* slots =
        static_cast<const uint64_t*>(owner->visibilityBuffer->contents());
    const uint32_t used = std::min(owner->nextVisibilitySlot,
                                   owner->visibilitySlotCount);
    uint64_t result = 0;
    for (uint32_t index = 0; index < used; ++index) {
        result += slots[index];
    }
    *result_out = result;
    return 0;
}

int mglRenderBeginTimerQuery(MGLQueryStateOwner * owner_handle) {
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    uint64_t cpuTimestamp = 0;
    uint64_t gpuTimestamp = 0;
    if (mglRenderSampleTimestamps(
            &cpuTimestamp, &gpuTimestamp) != 0) {
        return -1;
    }
    owner->timerQueryBeginGPU = gpuTimestamp;
    return 0;
}

int mglRenderEndTimerQuery(MGLQueryStateOwner * owner_handle, uint64_t* elapsed_out) {
    if (elapsed_out) *elapsed_out = 0;
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !elapsed_out) return -1;
    uint64_t cpuTimestamp = 0;
    uint64_t gpuTimestamp = 0;
    if (mglRenderSampleTimestamps(
            &cpuTimestamp, &gpuTimestamp) != 0) {
        return -1;
    }
    *elapsed_out = gpuTimestamp >= owner->timerQueryBeginGPU
        ? gpuTimestamp - owner->timerQueryBeginGPU
        : 0;
    return 0;
}

void mglRenderDestroyQueryStateOwner(MGLQueryStateOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::QueryStateOwner* owner =
        reinterpret_cast<mgl::QueryStateOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

extern "C"
int mglRenderExecuteComputeExecutionPlan(MGLCommandBufferOwner * command_buffer_owner, MGLCommandBufferRecoveryOwner * recovery_owner, const MGLRenderComputeExecutionPlan* plan, const MGLRenderCopyBackEntry* copy_backs, uint32_t copy_back_count, uint32_t require_cpu_visibility, MGLRenderComputeExecutionResult* result, char* err, size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (result) {
        memset(result, 0, sizeof(*result));
        result->failed_copy_back_index = copy_back_count;
    }
    if (!command_buffer_owner || !plan || !result ||
        (!copy_backs && copy_back_count)) {
        if (err && errcap) snprintf(err, errcap, "bad compute transaction args");
        return -1;
    }

    /* Reject malformed copy-backs before opening the compute encoder. */
    if (mglRenderEncodeStageBindingCopyBacks(
            copy_backs, copy_back_count, nullptr) != 0) {
        if (err && errcap) snprintf(err, errcap, "invalid compute copy-back");
        for (uint32_t i = 0; i < copy_back_count; i++) {
            const MGLRenderCopyBackEntry& entry = copy_backs[i];
            if (!entry.length) continue;
            MTL::Buffer* temporary = static_cast<MTL::Buffer*>(
                const_cast<void*>(entry.temporary));
            MTL::Buffer* destination = static_cast<MTL::Buffer*>(
                const_cast<void*>(entry.destination));
            if (!temporary || !destination ||
                entry.length > temporary->length() ||
                entry.destination_offset > destination->length() ||
                entry.length > destination->length() - entry.destination_offset) {
                result->failed_copy_back_index = i;
                break;
            }
        }
        return -1;
    }
    if (mglRenderEncodeComputeExecutionPlanForCommandBufferOwner(
            command_buffer_owner, plan, err, errcap) != 0) {
        return -1;
    }

    bool has_copies = false;
    for (uint32_t i = 0; i < copy_back_count; i++) {
        has_copies = has_copies || copy_backs[i].length != 0;
    }
    if (!has_copies && !require_cpu_visibility) return 0;

    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    MTL::CommandBuffer* command_buffer = owner->current;
    if (!command_buffer) {
        if (err && errcap) snprintf(err, errcap, "no compute command buffer");
        return -1;
    }
    if (has_copies) {
        MTL::BlitCommandEncoder* blit = command_buffer->blitCommandEncoder();
        if (!blit) {
            if (err && errcap) snprintf(err, errcap, "compute copy-back encoder failed");
            return -1;
        }
        int encode_result = mglRenderEncodeStageBindingCopyBacks(
            copy_backs, copy_back_count, blit);
        blit->endEncoding();
        if (encode_result != 0) {
            if (err && errcap) snprintf(err, errcap, "compute copy-back encode failed");
            return -1;
        }
    }

    result->submitted = 1u;
    if (mglRenderCommitCommandBufferTransaction(
            command_buffer_owner, nullptr, command_buffer, recovery_owner,
            1u, &result->transaction) != 0 ||
        result->transaction.has_error) {
        if (err && errcap) snprintf(err, errcap, "compute submit/wait failed");
        return -1;
    }
    if (mglRenderCopyBackCPUPrefix(
            copy_backs, copy_back_count,
            &result->failed_copy_back_index) != 0) {
        if (err && errcap) snprintf(err, errcap, "compute CPU prefix sync failed");
        return -1;
    }
    result->cpu_prefix_synchronized = 1u;
    return 0;
}

int mglRenderGetFboMatchCache(MGLRenderPassIdentityOwner * owner_handle, MGLRenderFboMatchCacheState* cache_out) {
    mgl::RenderPassIdentityOwner* owner =
        reinterpret_cast<mgl::RenderPassIdentityOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !cache_out || !owner->cache_valid) return 1;
    *cache_out = owner->cache;
    return 0;
}

int mglRenderGetRenderTargetSizeOwner(MGLRenderPassStateOwner * owner_handle, uint64_t* width_out, uint64_t* height_out) {
    auto* owner = reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    if (width_out) *width_out = owner->state.render_target_width;
    if (height_out) *height_out = owner->state.render_target_height;
    return 0;
}

extern "C"
int mglRenderIsVirtualizedGPU(void) {
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return 0;
    NS::String* name_string = renderer.device->name();
    const char* name = name_string ? name_string->utf8String() : nullptr;
    if (!name) return 0;
    return (std::strstr(name, "Paravirtual") != nullptr ||
            std::strstr(name, "paravirtual") != nullptr)
               ? 1
               : 0;
}

void mglRenderBeginSampleQueryCallback(GLMContext glm_ctx,
                                          unsigned int target) {
    BackendLeaseScope lease(glm_ctx);
    MGLQueryStateOwner* query_owner =
        static_cast<MGLQueryStateOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY));
    if (!query_owner) return;

    void* visibility_buffer = nullptr;
    if (mglRenderBeginSampleQuery(
            query_owner,
            target == GL_SAMPLES_PASSED ? 1u : 0u,
            "MGL Visibility Result", &visibility_buffer) != 0 ||
        !visibility_buffer) {
        return;
    }

    uint32_t mode = 0;
    uint64_t offset = 0;
    if (mglRenderAcquireSampleQuerySlot(
            query_owner, &mode, &offset) != 0) {
        return;
    }

    bool pass_has_visibility = false;
    MGLRenderPassState pass = {};
    MGLRenderPassStateOwner* render_pass_owner =
        static_cast<MGLRenderPassStateOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_PASS));
    if (render_pass_owner &&
        mglRenderGetRenderPassStateOwner(
            render_pass_owner, &pass) == 0) {
        pass_has_visibility = pass.visibility_result_buffer != nullptr;
    }

    MGLRenderEncoderOwner* render_owner =
        static_cast<MGLRenderEncoderOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER));
    if (!render_owner ||
        mglRenderEncoderOwnerHasCurrent(render_owner) != 1) {
        return;
    }
    if (!pass_has_visibility ||
        mglRenderSetVisibilityResultModeForRenderEncoderOwner(
            render_owner, mode, offset) != 0) {
        (void)mglRenderEndRenderEncoderOwner(render_owner);
    }
}

uint64_t mglRenderEndSampleQueryCallback(GLMContext glm_ctx) {
    BackendLeaseScope lease(glm_ctx);
    MGLQueryStateOwner* query_owner =
        static_cast<MGLQueryStateOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY));
    if (!query_owner) return 0;

    MGLRenderEncoderOwner* render_owner =
        static_cast<MGLRenderEncoderOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER));
    if (render_owner &&
        mglRenderEncoderOwnerHasCurrent(render_owner) == 1) {
        (void)mglRenderEndRenderEncoderOwner(render_owner);
    }
    mglRenderEndSampleQuery(query_owner);

    void* visibility_buffer = nullptr;
    if (mglRenderGetQueryVisibilityBuffer(
            query_owner, &visibility_buffer) == 0 &&
        visibility_buffer) {
        Sync boundary = {};
        mglRenderGetSync(glm_ctx, &boundary);
        mglRenderWaitForSync(glm_ctx, &boundary);
    }

    uint64_t result = 0;
    return mglRenderGetSampleQueryResult(
               query_owner, &result) == 0
        ? result : 0;
}

void mglRenderBeginTimerQueryCallback(GLMContext glm_ctx) {
    BackendLeaseScope lease(glm_ctx);
    MGLQueryStateOwner* query_owner =
        static_cast<MGLQueryStateOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY));
    if (!query_owner || mglRenderBeginTimerQuery(query_owner) != 0) {
        fprintf(stderr, "MGL ERROR: failed to begin Metal-cpp timer query\n");
    }
}

uint64_t mglRenderEndTimerQueryCallback(GLMContext glm_ctx) {
    BackendLeaseScope lease(glm_ctx);
    MGLQueryStateOwner* query_owner =
        static_cast<MGLQueryStateOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY));
    uint64_t elapsed = 0;
    return query_owner &&
           mglRenderEndTimerQuery(query_owner, &elapsed) == 0
        ? elapsed : 0;
}

int mglRenderGetOrCreateDepthStencilState(MGLPipelineCacheOwner * owner_handle, const MGLRenderDepthStencilDescriptorState* descriptor, void** depth_stencil_state_out, int* created_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    if (created_out) *created_out = 0;
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !descriptor || !depth_stencil_state_out) return -1;
    std::lock_guard<std::mutex> ownerLock(owner->mutex);
    if (!owner->depthStencilCacheEnabled) return -1;
    const mgl::DepthStencilCacheKey key =
        mgl::PipelineCacheOwner::makeDepthStencilKey(*descriptor);
    auto found = owner->depthStencilCache.find(key);
    if (found != owner->depthStencilCache.end()) {
        *depth_stencil_state_out = found->second->state;
        mgl::PipelineCacheOwner::touch(owner->depthStencilCacheLRU, key);
        return 0;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> rendererLock(renderer.mutex);
    MTL::DepthStencilState* state =
        mglRenderCreateDepthStencilFromStateLocked(renderer, *descriptor);
    if (!state) return -1;
    std::unique_ptr<mgl::PipelineCacheDepthStencilEntry> entry(
        new (std::nothrow) mgl::PipelineCacheDepthStencilEntry());
    if (!entry) {
        state->release();
        return -1;
    }
    entry->state = state;
    try {
        owner->depthStencilCache.emplace(key, std::move(entry));
        mgl::PipelineCacheOwner::touch(owner->depthStencilCacheLRU, key);
        while (owner->depthStencilCache.size() > 64u &&
               !owner->depthStencilCacheLRU.empty()) {
            const mgl::DepthStencilCacheKey oldest =
                owner->depthStencilCacheLRU.front();
            owner->depthStencilCacheLRU.pop_front();
            owner->depthStencilCache.erase(oldest);
        }
    } catch (...) {
        return -1;
    }
    if (created_out) *created_out = 1;
    *depth_stencil_state_out = state;
    return 0;
}
