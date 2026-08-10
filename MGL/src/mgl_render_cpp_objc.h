// Transitional Objective-C adapter for the pure-C Metal-cpp renderer ABI.
#pragma once

#import <Metal/Metal.h>

#include <stdio.h>

#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"

/* Snapshot command-buffer state through Metal-cpp when enabled while keeping
 * the direct Objective-C path as the A/B baseline. */
static inline MGLRenderCppCommandBufferState
mglRenderCommandBufferState(id<MTLCommandBuffer> commandBuffer)
{
    MGLRenderCppCommandBufferState state = {0};
    if (!commandBuffer) return state;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() &&
        mglRenderCppGetCommandBufferState(
            (__bridge void *)commandBuffer, &state) == 0) {
        return state;
    }

    state.status = (uint32_t)commandBuffer.status;
    NSError *error = commandBuffer.error;
    if (!error) return state;
    state.has_error = 1;
    state.error_code = (int64_t)error.code;
    snprintf(state.error_domain, sizeof(state.error_domain), "%s",
             error.domain.UTF8String ?: "");
    snprintf(state.error_description, sizeof(state.error_description), "%s",
             error.localizedDescription.UTF8String ?: "");
    return state;
}

static inline MTLCommandBufferStatus
mglRenderCommandBufferStatus(id<MTLCommandBuffer> commandBuffer)
{
    return (MTLCommandBufferStatus)
        mglRenderCommandBufferState(commandBuffer).status;
}

static inline NSString *
mglRenderCommandBufferErrorString(
    const MGLRenderCppCommandBufferState *state)
{
    if (!state || !state->has_error) return nil;
    return [NSString stringWithFormat:@"%s (domain=%s code=%lld)",
            state->error_description,
            state->error_domain,
            (long long)state->error_code];
}

typedef void (^MGLRenderCommandBufferCompletionBlock)(
    const MGLRenderCppCommandBufferState *state);

static inline void mglRenderInvokeCommandBufferCompletionBlock(
    void *context,
    const MGLRenderCppCommandBufferState *state)
{
    MGLRenderCommandBufferCompletionBlock block =
        (__bridge MGLRenderCommandBufferCompletionBlock)context;
    if (block) block(state);
}

static inline void mglRenderDestroyCommandBufferCompletionBlock(
    void *context)
{
    if (!context) return;
    id releasedBlock = CFBridgingRelease(context);
    (void)releasedBlock;
}

/* Under Metal-cpp, copy the ObjC block into an explicitly retained opaque
 * context. The C++ completion facade invokes it once and releases that retain
 * through destroy_context. The disabled path remains the native ObjC A/B
 * baseline. */
static inline int mglRenderAddCommandBufferCompletion(
    id<MTLCommandBuffer> commandBuffer,
    MGLRenderCommandBufferCompletionBlock block)
{
    if (!commandBuffer || !block) return -1;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice()) {
        MGLRenderCommandBufferCompletionBlock copiedBlock = [block copy];
        void *context = (__bridge_retained void *)copiedBlock;
        int result = mglRenderCppAddCommandBufferCompletion(
            (__bridge void *)commandBuffer,
            mglRenderInvokeCommandBufferCompletionBlock,
            context,
            mglRenderDestroyCommandBufferCompletionBlock);
        if (result == 0) return 0;
        mglRenderDestroyCommandBufferCompletionBlock(context);
    }
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        MGLRenderCppCommandBufferState state =
            mglRenderCommandBufferState(completed);
        block(&state);
    }];
    return 0;
}

/* Returns a borrowed queue retained by the opaque C++ owner. Assign the
 * result to an ObjC strong field before the owner can be reset or destroyed. */
static inline id<MTLCommandQueue>
mglRenderCppCreateOrResetCommandQueueOwner(void **owner,
                                           uint32_t maxCommandBuffers)
{
    if (!owner) return nil;
    void *queue = NULL;
    int result = *owner
        ? mglRenderCppResetCommandQueueOwner(
              *owner, maxCommandBuffers, &queue)
        : mglRenderCppCreateCommandQueueOwner(
              maxCommandBuffers, owner, &queue);
    return result == 0 && queue
        ? (__bridge id<MTLCommandQueue>)queue
        : nil;
}

/* Returns a borrowed command buffer retained by the opaque C++ owner. */
static inline id<MTLCommandBuffer>
mglRenderCppCreateOrResetCommandBufferOwner(
    void **owner,
    id<MTLCommandQueue> commandQueue)
{
    if (!owner || !commandQueue) return nil;
    void *commandBuffer = NULL;
    int result = *owner
        ? mglRenderCppResetCommandBufferOwner(
              *owner, (__bridge void *)commandQueue, &commandBuffer)
        : mglRenderCppCreateCommandBufferOwner(
              (__bridge void *)commandQueue, owner, &commandBuffer);
    return result == 0 && commandBuffer
        ? (__bridge id<MTLCommandBuffer>)commandBuffer
        : nil;
}

static inline MGLRenderCppTextureDescriptorState
mglRenderCppTextureDescriptorStateFromObjC(MTLTextureDescriptor *descriptor)
{
    MGLRenderCppTextureDescriptorState state = {0};
    if (!descriptor) return state;

    state.texture_type = (uint32_t)descriptor.textureType;
    state.pixel_format = (uint32_t)descriptor.pixelFormat;
    state.width = descriptor.width;
    state.height = descriptor.height;
    state.depth = descriptor.depth;
    state.mipmap_level_count = descriptor.mipmapLevelCount;
    state.sample_count = descriptor.sampleCount;
    state.array_length = descriptor.arrayLength;
    state.resource_options = descriptor.resourceOptions;
    state.usage = descriptor.usage;
    state.cpu_cache_mode = (uint32_t)descriptor.cpuCacheMode;
    state.storage_mode = (uint32_t)descriptor.storageMode;
    state.hazard_tracking_mode = (uint32_t)descriptor.hazardTrackingMode;
    state.compression_type = (uint32_t)descriptor.compressionType;
    state.placement_sparse_page_size =
        (uint32_t)descriptor.placementSparsePageSize;
    state.allow_gpu_optimized_contents =
        descriptor.allowGPUOptimizedContents ? 1u : 0u;

    MTLTextureSwizzleChannels swizzle = descriptor.swizzle;
    state.swizzle_red = (uint32_t)swizzle.red;
    state.swizzle_green = (uint32_t)swizzle.green;
    state.swizzle_blue = (uint32_t)swizzle.blue;
    state.swizzle_alpha = (uint32_t)swizzle.alpha;
    return state;
}
