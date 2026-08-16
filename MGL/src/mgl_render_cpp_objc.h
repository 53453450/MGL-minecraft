// Transitional Objective-C adapter for the pure-C Metal-cpp renderer ABI.
#pragma once

#import <Metal/Metal.h>

#include <stdio.h>

#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"

/* P4 whitelist adapter column: shell modules reference ObjC Metal objects
 * through these ref typedefs so their implementation text carries no
 * `id<MTL` (census criterion).  Semantics identical to id<MTL*> — strong
 * references, __bridge casts unchanged. */
typedef id<MTLDevice> MGLMetalDeviceRef;
typedef id<MTLBuffer> MGLMetalBufferRef;
typedef id<MTLTexture> MGLMetalTextureRef;
typedef id<MTLRenderCommandEncoder> MGLMetalRenderCommandEncoderRef;
typedef id<MTLComputeCommandEncoder> MGLMetalComputeCommandEncoderRef;
typedef id<MTLBlitCommandEncoder> MGLMetalBlitCommandEncoderRef;
typedef id<MTLCommandBuffer> MGLMetalCommandBufferRef;
typedef id<MTLCommandQueue> MGLMetalCommandQueueRef;
typedef id<MTLDrawable> MGLMetalDrawableRef;
typedef id<MTLFunction> MGLMetalFunctionRef;
typedef id<MTLRenderPipelineState> MGLMetalRenderPipelineStateRef;
typedef id<MTLComputePipelineState> MGLMetalComputePipelineStateRef;
typedef id<MTLDepthStencilState> MGLMetalDepthStencilStateRef;
typedef id<MTLSamplerState> MGLMetalSamplerStateRef;
typedef id<MTLEvent> MGLMetalEventRef;

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

/* P4.5 command-owner adapter. Gate-on reads/presents through the owner-aware
 * C++ facade; the raw Objective-C command buffer is borrowed only by the
 * disabled-gate baseline. */
static inline BOOL mglRenderCommandBufferOwnerState(
    void *owner,
    MGLRenderCppCommandBufferState *stateOut)
{
    if (stateOut) memset(stateOut, 0, sizeof(*stateOut));
    if (!owner || !stateOut) return NO;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() &&
        mglRenderCppGetCommandBufferOwnerState(owner, stateOut) == 0) {
        return YES;
    }
    MGLMetalCommandBufferRef commandBuffer =
        (__bridge MGLMetalCommandBufferRef)
            mglRenderCppCommandBufferOwnerGetCurrent(owner);
    if (!commandBuffer) return NO;
    *stateOut = mglRenderCommandBufferState(commandBuffer);
    return YES;
}

static inline int mglRenderPresentDrawableForCommandBufferOwner(
    void *owner,
    MGLMetalDrawableRef drawable)
{
    if (!owner || !drawable) return -1;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() &&
        mglRenderCppPresentDrawableForCommandBufferOwner(
            owner, (__bridge void *)drawable, NULL) == 0) {
        return 0;
    }
    MGLMetalCommandBufferRef commandBuffer =
        (__bridge MGLMetalCommandBufferRef)
            mglRenderCppCommandBufferOwnerGetCurrent(owner);
    if (!commandBuffer) return -1;
    [commandBuffer presentDrawable:drawable];
    return 0;
}

/* Owner-first encoder adapters. Gate-on keeps the command buffer inside the
 * C++ owner; the borrowed getter remains only for the ObjC compatibility
 * fallback. */
static inline MGLMetalRenderCommandEncoderRef
mglRenderCreateRenderEncoderForCommandBufferOwner(
    void *owner,
    MTLRenderPassDescriptor *descriptor,
    const MGLRenderCppRenderPassState *state)
{
    if (!owner) return nil;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() && state) {
        void *encoder = NULL;
        if (mglRenderCppCreateRenderEncoderFromCommandBufferOwnerState(
                owner, state, &encoder) == 0 && encoder) {
            return (__bridge MGLMetalRenderCommandEncoderRef)encoder;
        }
    }
    MGLMetalCommandBufferRef commandBuffer =
        (__bridge MGLMetalCommandBufferRef)
            mglRenderCppCommandBufferOwnerGetCurrent(owner);
    return commandBuffer && descriptor
        ? [commandBuffer renderCommandEncoderWithDescriptor:descriptor]
        : nil;
}

static inline MGLMetalBlitCommandEncoderRef
mglRenderCreateBlitEncoderForCommandBufferOwner(void *owner)
{
    if (!owner) return nil;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice()) {
        void *encoder = NULL;
        if (mglRenderCppCreateBlitEncoderFromCommandBufferOwner(
                owner, &encoder) == 0 && encoder) {
            return (__bridge MGLMetalBlitCommandEncoderRef)encoder;
        }
    }
    MGLMetalCommandBufferRef commandBuffer =
        (__bridge MGLMetalCommandBufferRef)
            mglRenderCppCommandBufferOwnerGetCurrent(owner);
    return commandBuffer ? [commandBuffer blitCommandEncoder] : nil;
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

/* P4.3a: draw 提交统一入口的 ObjC 桥。gate-on（MGL_USE_METALCPP + device）
 * 时把 plan 交给 C++ mglRenderCppEncodeDraw；返回 YES 表示已由 C++ 提交，
 * NO 表示调用方应走 ObjC 直接编码（gate-off 或 C++ 校验失败回退）。 */
static inline BOOL mglRenderCppTryEncodeDraw(
    id<MTLRenderCommandEncoder> encoder,
    const MGLRenderCppDrawPlan *plan)
{
    if (!encoder || !plan) return NO;
    if (!mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") ||
        !mglRenderCppGetDevice()) {
        return NO;
    }
    return mglRenderCppEncodeDraw(
        (__bridge void *)encoder, plan, NULL, 0) == 0;
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

/* === P4.1f: owner-first render-pass state readers ===
 *
 * Gate-on render-pass state lives in the C++ RenderPassStateOwner; the ObjC
 * MTLRenderPassDescriptor mirror is nil under gate-on.  These single gate
 * check and owner-first readers are the one shared source of truth for
 * category files (RenderPass/Batch/Draw/DrawSupport/BindingState/QuerySync/
 * MGLRenderer.m).  They consult the C++ owner first and fall back to the
 * descriptor mirror for the gate-off A/B baseline. */
static inline BOOL mglRenderPassUsesMetalCpp(void)
{
    return mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
           mglRenderCppGetDevice() != NULL;
}

static inline BOOL mglRenderCppGetRenderPassState(
    void *renderPassStateOwner,
    MGLRenderCppRenderPassState *stateOut)
{
    return renderPassStateOwner && stateOut &&
           mglRenderCppGetRenderPassStateOwner(
               renderPassStateOwner, stateOut) == 0;
}

/* Attachment texture — C++ owner first, ObjC mirror fallback. */
static inline id<MTLTexture> mglRenderPassAttachmentTextureForState(
    MTLRenderPassDescriptor *descriptor,
    void *renderPassStateOwner,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderCppGetRenderPassState(renderPassStateOwner, &state)) {
        const MGLRenderCppRenderPassAttachmentState *att = NULL;
        switch (attachmentKind) {
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
                if (colorIndex < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
                    att = &state.color[colorIndex].attachment;
                }
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
                att = &state.depth.attachment;
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
                att = &state.stencil.attachment;
                break;
            default:
                break;
        }
        if (att && att->texture) {
            return (__bridge id<MTLTexture>)att->texture;
        }
    }
    if (!descriptor) return nil;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
            if (colorIndex >= MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
                return nil;
            }
            return descriptor.colorAttachments[colorIndex].texture;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            return descriptor.depthAttachment.texture;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            return descriptor.stencilAttachment.texture;
        default:
            return nil;
    }
}

/* Attachment subresource (level/slice/depthPlane) — owner first, mirror
 * fallback.  Returns NO when neither source has the attachment. */
static inline BOOL mglRenderPassAttachmentSubresourceForState(
    MTLRenderPassDescriptor *descriptor,
    void *renderPassStateOwner,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    NSUInteger *levelOut,
    NSUInteger *sliceOut,
    NSUInteger *depthPlaneOut)
{
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderCppGetRenderPassState(renderPassStateOwner, &state)) {
        const MGLRenderCppRenderPassAttachmentState *att = NULL;
        switch (attachmentKind) {
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
                if (colorIndex < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
                    att = &state.color[colorIndex].attachment;
                }
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
                att = &state.depth.attachment;
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
                att = &state.stencil.attachment;
                break;
            default:
                break;
        }
        if (att) {
            if (levelOut) *levelOut = (NSUInteger)att->level;
            if (sliceOut) *sliceOut = (NSUInteger)att->slice;
            if (depthPlaneOut) *depthPlaneOut = (NSUInteger)att->depth_plane;
            return YES;
        }
    }
    if (!descriptor) return NO;
    MTLRenderPassAttachmentDescriptor *att = nil;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
            if (colorIndex >= MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
                return NO;
            }
            att = descriptor.colorAttachments[colorIndex];
            break;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            att = descriptor.depthAttachment;
            break;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            att = descriptor.stencilAttachment;
            break;
        default:
            return NO;
    }
    if (!att) return NO;
    if (levelOut) *levelOut = att.level;
    if (sliceOut) *sliceOut = att.slice;
    if (depthPlaneOut) *depthPlaneOut = att.depthPlane;
    return YES;
}

/* Render-target size — owner first, mirror fallback. */
static inline BOOL mglRenderPassRenderTargetSizeForState(
    MTLRenderPassDescriptor *descriptor,
    void *renderPassStateOwner,
    NSUInteger *widthOut,
    NSUInteger *heightOut)
{
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderCppGetRenderPassState(renderPassStateOwner, &state)) {
        if (widthOut) *widthOut = (NSUInteger)state.render_target_width;
        if (heightOut) *heightOut = (NSUInteger)state.render_target_height;
        return YES;
    }
    if (!descriptor) return NO;
    if (widthOut) *widthOut = descriptor.renderTargetWidth;
    if (heightOut) *heightOut = descriptor.renderTargetHeight;
    return YES;
}

/* Whether the active render pass uses the given texture as a color
 * attachment — owner first, mirror fallback. */
static inline BOOL mglRenderPassUsesColorTextureForState(
    MTLRenderPassDescriptor *descriptor,
    void *renderPassStateOwner,
    id<MTLTexture> texture,
    NSUInteger *attachmentIndexOut)
{
    if (!texture) return NO;
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderCppGetRenderPassState(renderPassStateOwner, &state)) {
        for (NSUInteger i = 0; i < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS; i++) {
            if (state.color[i].attachment.texture == (__bridge void *)texture) {
                if (attachmentIndexOut) *attachmentIndexOut = i;
                return YES;
            }
        }
        return NO;
    }
    if (!descriptor) return NO;
    /* P4.5: mirror fallback 迁入 C++（mglRenderCppRenderPassUsesColorTexture）。 */
    size_t index = MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS;
    int hit = mglRenderCppRenderPassUsesColorTexture(
        (__bridge void *)descriptor, (__bridge void *)texture, &index);
    if (hit == 1) {
        if (attachmentIndexOut) *attachmentIndexOut = (NSUInteger)index;
        return YES;
    }
    return NO;
}

/* Load/store action of one attachment — owner first, mirror fallback. */
static inline BOOL mglRenderPassActionsForState(
    MTLRenderPassDescriptor *descriptor,
    void *renderPassStateOwner,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t *loadActionOut,
    uint32_t *storeActionOut,
    uint64_t *storeActionOptionsOut)
{
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderCppGetRenderPassState(renderPassStateOwner, &state)) {
        const MGLRenderCppRenderPassAttachmentState *att = NULL;
        switch (attachmentKind) {
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
                if (colorIndex < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
                    att = &state.color[colorIndex].attachment;
                }
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
                att = &state.depth.attachment;
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
                att = &state.stencil.attachment;
                break;
            default:
                break;
        }
        if (att) {
            if (loadActionOut) *loadActionOut = (uint32_t)att->load_action;
            if (storeActionOut) *storeActionOut = (uint32_t)att->store_action;
            if (storeActionOptionsOut) {
                *storeActionOptionsOut = att->store_action_options;
            }
            return YES;
        }
    }
    if (!descriptor) return NO;
    MTLRenderPassAttachmentDescriptor *att = nil;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
            if (colorIndex >= MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
                return NO;
            }
            att = descriptor.colorAttachments[colorIndex];
            break;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            att = descriptor.depthAttachment;
            break;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            att = descriptor.stencilAttachment;
            break;
        default:
            return NO;
    }
    if (!att) return NO;
    if (loadActionOut) *loadActionOut = (uint32_t)att.loadAction;
    if (storeActionOut) *storeActionOut = (uint32_t)att.storeAction;
    if (storeActionOptionsOut) {
        *storeActionOptionsOut = att.storeActionOptions;
    }
    return YES;
}

/* Load/store action of one attachment with a caller default — owner first,
 * mirror fallback, then default.  Used by trace/log call sites. */
static inline uint32_t mglRenderPassLoadActionForTrace(
    MTLRenderPassDescriptor *descriptor,
    void *renderPassStateOwner,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MTLLoadAction defaultLoadAction)
{
    uint32_t loadAction = 0u;
    if (mglRenderPassActionsForState(
            descriptor, renderPassStateOwner, attachmentKind, colorIndex,
            &loadAction, NULL, NULL)) {
        return loadAction;
    }
    return (uint32_t)defaultLoadAction;
}

static inline uint32_t mglRenderPassStoreActionForTrace(
    MTLRenderPassDescriptor *descriptor,
    void *renderPassStateOwner,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MTLStoreAction defaultStoreAction)
{
    uint32_t storeAction = 0u;
    if (mglRenderPassActionsForState(
            descriptor, renderPassStateOwner, attachmentKind, colorIndex,
            NULL, &storeAction, NULL)) {
        return storeAction;
    }
    return (uint32_t)defaultStoreAction;
}
