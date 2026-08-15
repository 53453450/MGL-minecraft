// MGLRenderer+RenderPass.m
// Render pass lifecycle methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+RenderPass_Private.h"
#include "mgl_air_loader.h"     /* METALCPP: AIR metallib 加载（Phase 1） */
#include "mgl_aux_assets.h"
#include "mgl_render_cpp_objc.h" /* METALCPP: C ABI + ObjC state adapter */
#include "mgl_shader_abi.h"

#import <objc/message.h>

/* Gate-off path: load a precompiled aux metallib with the plain Metal
 * load-from-data API (safe fallback branch; never runtime compilation). */
static id<MTLLibrary> mglRenderPassLoadAuxLibrary(id<MTLDevice> device,
                                                  const char *assetName,
                                                  NSError **error)
{
    const MGLAuxShaderAsset *asset = mglAuxShaderAssetFind(assetName);
    if (!asset || !asset->data || asset->size == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"MGLRenderPassAuxAsset"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    [NSString stringWithFormat:
                                                        @"aux shader asset '%s' missing",
                                                     assetName]}];
        }
        return nil;
    }
    dispatch_data_t data = dispatch_data_create(
        asset->data, asset->size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    return [device newLibraryWithData:data error:error];
}

/* mglRenderPassUsesMetalCpp: P4.1f 统一 gate 定义移入
 * mgl_render_cpp_objc.h（owner-first reader 需要跨 category 共享）。 */

static MGLRenderCppRenderPassIdentityState mglRenderPassIdentitySnapshot(
    const MGLCommandState *commandState)
{
    MGLRenderCppRenderPassIdentityState identity = {0};
    if (mglRenderPassUsesMetalCpp() && commandState->renderPassIdentityOwner &&
        mglRenderCppGetRenderPassIdentity(
            commandState->renderPassIdentityOwner, &identity) == 0) {
        return identity;
    }
    identity.framebuffer = commandState->renderPassFramebuffer;
    identity.framebuffer_name = commandState->renderPassFramebufferName;
    identity.draw_buffer = commandState->renderPassDrawBuffer;
    identity.draw_buffer_count = commandState->renderPassDrawBufferCount;
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; ++index) {
        identity.draw_buffers[index] = commandState->renderPassDrawBuffers[index];
    }
    return identity;
}

static id<MTLTexture> mglRenderPassCreateTexture(
    id<MTLDevice> device,
    MTLTextureDescriptor *descriptor)
{
    if (mglRenderPassUsesMetalCpp()) {
        void *texture = NULL;
        MGLRenderCppTextureDescriptorState state =
            mglRenderCppTextureDescriptorStateFromObjC(descriptor);
        if (mglRenderCppCreateTextureFromState(&state, NULL, &texture) == 0 &&
            texture) {
            return (__bridge_transfer id<MTLTexture>)texture;
        }
    }
    return [device newTextureWithDescriptor:descriptor];
}

static id<MTLBuffer> mglRenderPassCreateBufferWithBytes(
    id<MTLDevice> device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglRenderPassUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                              &buffer) == 0 && buffer) {
            return (__bridge_transfer id<MTLBuffer>)buffer;
        }
    }
    return [device newBufferWithBytes:bytes length:length options:options];
}

static id<MTLCommandQueue> mglRenderPassCreateCommandQueue(
    id<MTLDevice> device,
    void **owner)
{
    if (mglRenderPassUsesMetalCpp()) {
        id<MTLCommandQueue> queue =
            mglRenderCppCreateOrResetCommandQueueOwner(owner, 0u);
        if (queue) return queue;
    }
    mglRenderCppDestroyCommandQueueOwner(owner);
    return [device newCommandQueue];
}

static id<MTLFunction> mglRenderPassCreateFunction(
    id<MTLLibrary> library,
    NSString *name,
    char *errorText,
    size_t errorCap)
{
    if (mglRenderPassUsesMetalCpp()) {
        void *function = NULL;
        if (mglRenderCppCreateFunction(
                (__bridge void *)library, name.UTF8String, NULL,
                &function, errorText, errorCap) == 0 && function) {
            return (__bridge_transfer id<MTLFunction>)function;
        }
        return nil;
    }
    return [library newFunctionWithName:name];
}

static id<MTLRenderPipelineState> mglRenderPassCreatePipeline(
    id<MTLDevice> device,
    MTLRenderPipelineDescriptor *descriptor,
    NSError **error)
{
    if (mglRenderPassUsesMetalCpp()) {
        void *pipeline = NULL;
        char message[512] = {0};
        if (mglRenderCppCreateRenderPipelineState(
                (__bridge void *)descriptor, &pipeline,
                message, sizeof(message)) == 0 && pipeline) {
            if (error) *error = nil;
            return (__bridge_transfer id<MTLRenderPipelineState>)pipeline;
        }
        if (error) {
            NSString *description = message[0]
                ? [NSString stringWithUTF8String:message]
                : @"Metal-cpp render pipeline creation failed";
            *error = [NSError errorWithDomain:@"MGLRenderPipeline"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    description}];
        }
        return nil;
    }
    return [device newRenderPipelineStateWithDescriptor:descriptor error:error];
}

static void mglRenderPassSetDepthClipMode(
    id<MTLRenderCommandEncoder> encoder,
    MTLDepthClipMode mode)
{
    if (mglRenderPassUsesMetalCpp() &&
        mglRenderCppSetDepthClipMode((__bridge void *)encoder,
                                     (uint32_t)mode) == 0) {
        return;
    }
    [encoder setDepthClipMode:mode];
}

static void mglRenderPassSetStencilReferenceValues(
    id<MTLRenderCommandEncoder> encoder,
    uint32_t frontReference,
    uint32_t backReference)
{
    if (mglRenderPassUsesMetalCpp() &&
        mglRenderCppSetStencilReferenceValues(
            (__bridge void *)encoder, frontReference, backReference) == 0) {
        return;
    }
    [encoder setStencilFrontReferenceValue:frontReference
                        backReferenceValue:backReference];
}

static void mglRenderPassSetFragmentBytes(
    id<MTLRenderCommandEncoder> encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    if (mglRenderPassUsesMetalCpp() &&
        mglRenderCppSetRenderBytes(
            (__bridge void *)encoder, bytes, length,
            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, (uint32_t)index) == 0) {
        return;
    }
    [encoder setFragmentBytes:bytes length:length atIndex:index];
}

static void mglRenderPassSetBuffer(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLBuffer> buffer,
    NSUInteger offset,
    uint32_t stage,
    NSUInteger index)
{
    if (mglRenderPassUsesMetalCpp() &&
        mglRenderCppSetRenderBuffer(
            (__bridge void *)encoder, (__bridge void *)buffer, offset, stage,
            (uint32_t)index) == 0) {
        return;
    }
    if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
        [encoder setVertexBuffer:buffer offset:offset atIndex:index];
    } else {
        [encoder setFragmentBuffer:buffer offset:offset atIndex:index];
    }
}

static void mglRenderPassWaitCommandBuffer(id<MTLCommandBuffer> commandBuffer)
{
    if (mglRenderPassUsesMetalCpp() &&
        mglRenderCppWaitCommandBuffer((__bridge void *)commandBuffer) == 0) {
        return;
    }
    [commandBuffer waitUntilCompleted];
}

static bool mglRenderPassGetPersistentState(
    const MGLCommandState *commandState,
    MGLRenderCppRenderPassState *stateOut)
{
    return commandState && stateOut && commandState->renderPassStateOwner &&
           mglRenderCppGetRenderPassStateOwner(
               commandState->renderPassStateOwner, stateOut) == 0;
}

static bool mglRenderPassGetPersistentAttachmentState(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MGLRenderCppRenderPassAttachmentState *attachmentOut)
{
    if (!attachmentOut) return false;
    MGLRenderCppRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return false;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
            if (colorIndex >= MAX_COLOR_ATTACHMENTS) return false;
            *attachmentOut = state.color[colorIndex].attachment;
            return true;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            *attachmentOut = state.depth.attachment;
            return true;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            *attachmentOut = state.stencil.attachment;
            return true;
        default:
            return false;
    }
}

static const MGLRenderCppRenderPassAttachmentState *
mglRenderPassAttachmentStateFromSnapshot(
    const MGLRenderCppRenderPassState *state,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    if (!state) return NULL;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
            return colorIndex < MAX_COLOR_ATTACHMENTS
                ? &state->color[colorIndex].attachment : NULL;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            return &state->depth.attachment;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            return &state->stencil.attachment;
        default:
            return NULL;
    }
}

static id<MTLTexture> mglRenderPassTextureFromSnapshot(
    const MGLRenderCppRenderPassState *state,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    const MGLRenderCppRenderPassAttachmentState *attachment =
        mglRenderPassAttachmentStateFromSnapshot(
            state, attachmentKind, colorIndex);
    return attachment && attachment->texture
        ? (__bridge id<MTLTexture>)attachment->texture : nil;
}

static bool mglRenderPassSnapshotAttachmentMatchesSubresource(
    const MGLRenderCppRenderPassState *state,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MGLMetalAttachmentSubresource subresource)
{
    const MGLRenderCppRenderPassAttachmentState *attachment =
        mglRenderPassAttachmentStateFromSnapshot(
            state, attachmentKind, colorIndex);
    return attachment &&
           attachment->level == subresource.level &&
           attachment->slice == subresource.slice &&
           attachment->depth_plane == subresource.depthPlane;
}

/* Owner-first render-pass state readers.  Under Metal-cpp the C++
 * RenderPassStateOwner is the writer of record for every attachment field;
 * these helpers consult it first and fall back to the ObjC descriptor mirror
 * for the gate-off A/B path.  They must only be used at read sites outside
 * the mglRenderPassSetPersistent* writer bodies (those write both sides). */
static id<MTLTexture> mglRenderPassAttachmentTextureFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    MGLRenderCppRenderPassAttachmentState attachment = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &attachment)) {
        return attachment.texture
            ? (__bridge id<MTLTexture>)attachment.texture : nil;
    }
    if (!commandState || !commandState->renderPassDescriptor) {
        return nil;
    }
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
            if (colorIndex >= MAX_COLOR_ATTACHMENTS) return nil;
            return commandState->renderPassDescriptor
                .colorAttachments[colorIndex].texture;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            return commandState->renderPassDescriptor.depthAttachment.texture;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            return commandState->renderPassDescriptor.stencilAttachment.texture;
        default:
            return nil;
    }
}

static id<MTLTexture> mglRenderPassColorTextureFor(
    const MGLCommandState *commandState, NSUInteger colorIndex)
{
    return mglRenderPassAttachmentTextureFor(
        commandState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
        colorIndex);
}

static id<MTLTexture> mglRenderPassDepthTextureFor(
    const MGLCommandState *commandState)
{
    return mglRenderPassAttachmentTextureFor(
        commandState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0u);
}

static id<MTLTexture> mglRenderPassStencilTextureFor(
    const MGLCommandState *commandState)
{
    return mglRenderPassAttachmentTextureFor(
        commandState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0u);
}

/* Owner-first load/store actions and clear values for one attachment. */
static BOOL mglRenderPassActionsFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t *loadActionOut,
    uint32_t *storeActionOut,
    uint64_t *storeActionOptionsOut)
{
    MGLRenderCppRenderPassAttachmentState attachment = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &attachment)) {
        if (loadActionOut) *loadActionOut = (uint32_t)attachment.load_action;
        if (storeActionOut) *storeActionOut = (uint32_t)attachment.store_action;
        if (storeActionOptionsOut) {
            *storeActionOptionsOut = attachment.store_action_options;
        }
        return YES;
    }
    if (!commandState || !commandState->renderPassDescriptor) return NO;
    MTLRenderPassAttachmentDescriptor *attachmentDescriptor = nil;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR: {
            if (colorIndex >= MAX_COLOR_ATTACHMENTS) return NO;
            attachmentDescriptor = commandState->renderPassDescriptor
                .colorAttachments[colorIndex];
            break;
        }
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            attachmentDescriptor =
                commandState->renderPassDescriptor.depthAttachment;
            break;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            attachmentDescriptor =
                commandState->renderPassDescriptor.stencilAttachment;
            break;
        default:
            return NO;
    }
    if (!attachmentDescriptor) return NO;
    if (loadActionOut) {
        *loadActionOut = (uint32_t)attachmentDescriptor.loadAction;
    }
    if (storeActionOut) {
        *storeActionOut = (uint32_t)attachmentDescriptor.storeAction;
    }
    if (storeActionOptionsOut) {
        *storeActionOptionsOut = attachmentDescriptor.storeActionOptions;
    }
    return YES;
}

static BOOL mglRenderPassClearValuesFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    double *clearColorOut,   /* RGBA, color attachments */
    double *clearDepthOut,
    uint32_t *clearStencilOut)
{
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderPassGetPersistentState(commandState, &state)) {
        switch (attachmentKind) {
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR: {
                if (colorIndex >= MAX_COLOR_ATTACHMENTS) return NO;
                const MGLRenderCppRenderPassColorState *color =
                    &state.color[colorIndex];
                if (clearColorOut) {
                    clearColorOut[0] = color->clear_red;
                    clearColorOut[1] = color->clear_green;
                    clearColorOut[2] = color->clear_blue;
                    clearColorOut[3] = color->clear_alpha;
                }
                return YES;
            }
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
                if (clearDepthOut) *clearDepthOut = state.depth.clear_depth;
                return YES;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
                if (clearStencilOut) *clearStencilOut = state.stencil.clear_stencil;
                return YES;
            default:
                return NO;
        }
    }
    if (!commandState || !commandState->renderPassDescriptor) return NO;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR: {
            if (colorIndex >= MAX_COLOR_ATTACHMENTS) return NO;
            MTLRenderPassColorAttachmentDescriptor *attachment =
                commandState->renderPassDescriptor
                    .colorAttachments[colorIndex];
            if (!attachment) return NO;
            if (clearColorOut) {
                clearColorOut[0] = attachment.clearColor.red;
                clearColorOut[1] = attachment.clearColor.green;
                clearColorOut[2] = attachment.clearColor.blue;
                clearColorOut[3] = attachment.clearColor.alpha;
            }
            return YES;
        }
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            if (clearDepthOut) {
                *clearDepthOut = commandState->renderPassDescriptor
                    .depthAttachment.clearDepth;
            }
            return YES;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            if (clearStencilOut) {
                *clearStencilOut = commandState->renderPassDescriptor
                    .stencilAttachment.clearStencil;
            }
            return YES;
        default:
            return NO;
    }
}

static BOOL mglRenderPassRenderTargetSizeFor(
    const MGLCommandState *commandState,
    uint64_t *widthOut,
    uint64_t *heightOut)
{
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderPassGetPersistentState(commandState, &state)) {
        if (widthOut) *widthOut = state.render_target_width;
        if (heightOut) *heightOut = state.render_target_height;
        return YES;
    }
    if (!commandState || !commandState->renderPassDescriptor) return NO;
    if (widthOut) *widthOut = commandState->renderPassDescriptor.renderTargetWidth;
    if (heightOut) *heightOut = commandState->renderPassDescriptor.renderTargetHeight;
    return YES;
}

/* Single-value variants of the owner-first readers for scripted call sites.
 * Each falls back to the mirror, then to the provided default. */
static NSUInteger mglRenderPassRenderTargetWidthFor(
    const MGLCommandState *commandState)
{
    uint64_t width = 0;
    if (mglRenderPassRenderTargetSizeFor(commandState, &width, NULL)) {
        return (NSUInteger)width;
    }
    return 0;
}

static NSUInteger mglRenderPassRenderTargetHeightFor(
    const MGLCommandState *commandState)
{
    uint64_t height = 0;
    if (mglRenderPassRenderTargetSizeFor(commandState, NULL, &height)) {
        return (NSUInteger)height;
    }
    return 0;
}

static MTLLoadAction mglRenderPassLoadActionFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MTLLoadAction fallback)
{
    uint32_t action = 0u;
    if (mglRenderPassActionsFor(commandState, attachmentKind, colorIndex,
                                &action, NULL, NULL)) {
        return (MTLLoadAction)action;
    }
    return fallback;
}

static MTLStoreAction mglRenderPassStoreActionFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MTLStoreAction fallback)
{
    uint32_t action = 0u;
    if (mglRenderPassActionsFor(commandState, attachmentKind, colorIndex,
                                NULL, &action, NULL)) {
        return (MTLStoreAction)action;
    }
    return fallback;
}

static MTLClearColor mglRenderPassClearColorFor(
    const MGLCommandState *commandState,
    NSUInteger colorIndex,
    MTLClearColor fallback)
{
    double rgba[4] = {0.0, 0.0, 0.0, 0.0};
    if (mglRenderPassClearValuesFor(
            commandState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
            colorIndex, rgba, NULL, NULL)) {
        return MTLClearColorMake(rgba[0], rgba[1], rgba[2], rgba[3]);
    }
    return fallback;
}

static double mglRenderPassClearDepthFor(
    const MGLCommandState *commandState, double fallback)
{
    double depth = 0.0;
    if (mglRenderPassClearValuesFor(
            commandState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0u,
            NULL, &depth, NULL)) {
        return depth;
    }
    return fallback;
}

static uint32_t mglRenderPassClearStencilFor(
    const MGLCommandState *commandState, uint32_t fallback)
{
    uint32_t stencil = 0u;
    if (mglRenderPassClearValuesFor(
            commandState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0u,
            NULL, NULL, &stencil)) {
        return stencil;
    }
    return fallback;
}

static uint32_t mglRenderPassVisibilityResultTypeFor(
    const MGLCommandState *commandState)
{
    MGLRenderCppRenderPassState state = {0};
    if (mglRenderPassGetPersistentState(commandState, &state)) {
        return state.visibility_result_type;
    }
    if (commandState && commandState->renderPassDescriptor) {
        return (uint32_t)commandState->renderPassDescriptor.visibilityResultType;
    }
    return 0u;
}

static void mglRenderPassSetPersistentAttachment(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    id<MTLTexture> texture,
    NSUInteger level,
    NSUInteger slice,
    NSUInteger depthPlane)
{

    /* Gate-on: the C++ RenderPassStateOwner is the writer of record; the
     * ObjC descriptor mirror does not exist (P4.1f). */
    if (mglRenderPassUsesMetalCpp()) {
        if (commandState->renderPassStateOwner) {
            mglRenderCppSetRenderPassStateAttachmentTexture(
                commandState->renderPassStateOwner, attachmentKind,
                (uint32_t)colorIndex, (__bridge void *)texture,
                level, slice, depthPlane);
        }
        return;
    }

    /* Gate-off A/B baseline: the ObjC descriptor mirror is the writer. */
    if (!commandState->renderPassDescriptor) return;
    MTLRenderPassAttachmentDescriptor *attachment = nil;
    switch (attachmentKind) {
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
            if (colorIndex >= MAX_COLOR_ATTACHMENTS) return;
            attachment = commandState->renderPassDescriptor
                .colorAttachments[colorIndex];
            break;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
            attachment = commandState->renderPassDescriptor.depthAttachment;
            break;
        case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
            attachment = commandState->renderPassDescriptor.stencilAttachment;
            break;
        default:
            return;
    }
    attachment.texture = texture;
    attachment.level = level;
    attachment.slice = slice;
    attachment.depthPlane = depthPlane;

    /* Layered rendering: cap the number of arrays the VS
     * [[render_target_array_index]] output may select.  Keep it at most
     * the largest arrayLength among currently attached color textures, so
     * switching between layered and plain targets stays valid. */
    NSUInteger maxArrayLength = 1u;
    MTLRenderPassColorAttachmentDescriptorArray *colors =
        commandState->renderPassDescriptor.colorAttachments;
    for (NSUInteger i = 0u; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> t = colors[i].texture;
        if (t && t.arrayLength > maxArrayLength) {
            maxArrayLength = t.arrayLength;
        }
    }
    commandState->renderPassDescriptor.renderTargetArrayLength =
        maxArrayLength;
    /* Layered pass: the layer comes from the VS
     * [[render_target_array_index]] output, so a non-zero attachment
     * slice would be ignored (or dropped) by Metal.  GL's
     * glFramebufferTextureLayer slice is honored via the array index
     * instead, keeping the descriptor slice at 0. */
    if (attachment && maxArrayLength > 0u) {
        attachment.slice = 0u;
    }
}

static void mglRenderPassSetPersistentDimensions(
    const MGLCommandState *commandState,
    NSUInteger width,
    NSUInteger height)
{
    if (!commandState) return;
    if (!mglRenderPassUsesMetalCpp() && commandState->renderPassDescriptor) {
        commandState->renderPassDescriptor.renderTargetWidth = width;
        commandState->renderPassDescriptor.renderTargetHeight = height;
    }
    if (mglRenderPassUsesMetalCpp() && commandState->renderPassStateOwner) {
        mglRenderCppSetRenderPassStateDimensions(
            commandState->renderPassStateOwner, width, height);
    }
}

static void mglRenderPassSetPersistentActions(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MTLLoadAction loadAction,
    MTLStoreAction storeAction)
{
    if (!commandState) return;
    uint64_t storeActionOptions = 0u;
    MTLRenderPassAttachmentDescriptor *attachment = nil;
    if (!mglRenderPassUsesMetalCpp() && commandState->renderPassDescriptor) {
        switch (attachmentKind) {
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
                if (colorIndex >= MAX_COLOR_ATTACHMENTS) return;
                attachment = commandState->renderPassDescriptor
                    .colorAttachments[colorIndex];
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
                attachment = commandState->renderPassDescriptor.depthAttachment;
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
                attachment = commandState->renderPassDescriptor.stencilAttachment;
                break;
            default:
                return;
        }
        attachment.loadAction = loadAction;
        attachment.storeAction = storeAction;
        storeActionOptions = attachment.storeActionOptions;
    } else {
        MGLRenderCppRenderPassAttachmentState state = {0};
        if (!mglRenderPassGetPersistentAttachmentState(
                commandState, attachmentKind, colorIndex, &state)) {
            return;
        }
        storeActionOptions = state.store_action_options;
    }
    if (mglRenderPassUsesMetalCpp() && commandState->renderPassStateOwner) {
        mglRenderCppSetRenderPassStateAttachmentActions(
            commandState->renderPassStateOwner, attachmentKind,
            (uint32_t)colorIndex, (uint32_t)loadAction,
            (uint32_t)storeAction, storeActionOptions);
    }
}

static void mglRenderPassSetPersistentLoadAction(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MTLLoadAction loadAction)
{
    MTLStoreAction storeAction = MTLStoreActionDontCare;
    MGLRenderCppRenderPassAttachmentState state = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &state)) {
        storeAction = (MTLStoreAction)state.store_action;
    } else if (commandState && commandState->renderPassDescriptor) {
        MTLRenderPassAttachmentDescriptor *attachment = nil;
        switch (attachmentKind) {
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
                if (colorIndex >= MAX_COLOR_ATTACHMENTS) return;
                attachment = commandState->renderPassDescriptor
                    .colorAttachments[colorIndex];
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
                attachment = commandState->renderPassDescriptor.depthAttachment;
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
                attachment = commandState->renderPassDescriptor.stencilAttachment;
                break;
            default:
                return;
        }
        storeAction = attachment.storeAction;
    } else {
        return;
    }
    mglRenderPassSetPersistentActions(
        commandState, attachmentKind, colorIndex, loadAction, storeAction);
}

static void mglRenderPassSetPersistentStoreAction(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MTLStoreAction storeAction)
{
    MTLLoadAction loadAction = MTLLoadActionDontCare;
    MGLRenderCppRenderPassAttachmentState state = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &state)) {
        loadAction = (MTLLoadAction)state.load_action;
    } else if (commandState && commandState->renderPassDescriptor) {
        MTLRenderPassAttachmentDescriptor *attachment = nil;
        switch (attachmentKind) {
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
                if (colorIndex >= MAX_COLOR_ATTACHMENTS) return;
                attachment = commandState->renderPassDescriptor
                    .colorAttachments[colorIndex];
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
                attachment = commandState->renderPassDescriptor.depthAttachment;
                break;
            case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
                attachment = commandState->renderPassDescriptor.stencilAttachment;
                break;
            default:
                return;
        }
        loadAction = attachment.loadAction;
    } else {
        return;
    }
    mglRenderPassSetPersistentActions(
        commandState, attachmentKind, colorIndex, loadAction, storeAction);
}

static void mglRenderPassSetPersistentColorClear(
    const MGLCommandState *commandState,
    NSUInteger colorIndex,
    MTLClearColor clearColor)
{
    if (!commandState || colorIndex >= MAX_COLOR_ATTACHMENTS) {
        return;
    }
    if (!mglRenderPassUsesMetalCpp() && commandState->renderPassDescriptor) {
        commandState->renderPassDescriptor.colorAttachments[colorIndex].clearColor =
            clearColor;
    }
    if (mglRenderPassUsesMetalCpp() && commandState->renderPassStateOwner) {
        mglRenderCppSetRenderPassStateColorClear(
            commandState->renderPassStateOwner, (uint32_t)colorIndex,
            clearColor.red, clearColor.green, clearColor.blue,
            clearColor.alpha);
    }
}

static void mglRenderPassSetPersistentDepthClear(
    const MGLCommandState *commandState,
    double clearDepth)
{
    if (!commandState) return;
    if (!mglRenderPassUsesMetalCpp() && commandState->renderPassDescriptor) {
        commandState->renderPassDescriptor.depthAttachment.clearDepth = clearDepth;
    }
    if (mglRenderPassUsesMetalCpp() && commandState->renderPassStateOwner) {
        mglRenderCppSetRenderPassStateDepthClear(
            commandState->renderPassStateOwner, clearDepth);
    }
}

static void mglRenderPassSetPersistentStencilClear(
    const MGLCommandState *commandState,
    uint32_t clearStencil)
{
    if (!commandState) return;
    if (commandState->renderPassDescriptor) {
        commandState->renderPassDescriptor.stencilAttachment.clearStencil =
            clearStencil;
    }
    if (mglRenderPassUsesMetalCpp() && commandState->renderPassStateOwner) {
        mglRenderCppSetRenderPassStateStencilClear(
            commandState->renderPassStateOwner, clearStencil);
    }
}

static bool mglGeometryShaderIsPassthrough(const Shader *shader)
{
    const char *src = shader ? shader->src : NULL;
    if (!src) {
        return false;
    }

    /* Metal has no geometry stage.  A few CTS paths insert a geometry shader
     * whose only job is to re-emit each input vertex unchanged while copying
     * clip/cull distance arrays.  Those programs are equivalent to running
     * the VS->FS pipeline directly, and blocking them regresses otherwise
     * valid cull-distance coverage.  Keep this deliberately narrow so real
     * geometry expansion/rewriting remains unsupported instead of being
     * silently misrendered. */
    return strstr(src, "EmitVertex()") &&
           strstr(src, "EndPrimitive()") &&
           strstr(src, "gl_Position = gl_in[n_vertex_index].gl_Position") &&
           !strstr(src, "gl_PrimitiveID") &&
           !strstr(src, "gl_Layer") &&
           !strstr(src, "gl_ViewportIndex");
}

static bool mglLoadAIRMainFunction(id<MTLDevice> device,
                                   const unsigned char *bytes,
                                   size_t size,
                                   id<MTLLibrary> __strong *libraryOut,
                                   id<MTLFunction> __strong *functionOut,
                                   char *errorText,
                                   size_t errorCap)
{
    if (!device || !bytes || size == 0u || !libraryOut || !functionOut) {
        if (errorText && errorCap) snprintf(errorText, errorCap, "bad args");
        return false;
    }
    id<MTLLibrary> library = nil;
    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP")) {
        void *libraryCPP = NULL;
        void *deviceCPP = mglRenderCppGetDevice();
        if (!deviceCPP ||
            mglAirLoadLibrary(deviceCPP, bytes, size, &libraryCPP,
                              errorText, errorCap) != 0 || !libraryCPP) {
            return false;
        }
        library = (__bridge_transfer id<MTLLibrary>)libraryCPP;
    } else {
        dispatch_data_t data = dispatch_data_create(
            bytes, size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        __autoreleasing NSError *error = nil;
        library = [device newLibraryWithData:data error:&error];
        if (!library) {
            if (errorText && errorCap) {
                snprintf(errorText, errorCap, "%s",
                         error.localizedDescription.UTF8String ?: "?");
            }
            return false;
        }
    }
    id<MTLFunction> function =
        mglRenderPassCreateFunction(library, @"main", errorText, errorCap);
    if (!function) {
        if (errorText && errorCap) snprintf(errorText, errorCap, "missing main");
        return false;
    }
    *libraryOut = library;
    *functionOut = function;
    return true;
}

@implementation MGLRenderer (RenderPass)

static const char *mglGeometryPassthroughType(GLenum type)
{
    switch (type) {
        case GL_FLOAT: return "float";
        case GL_FLOAT_VEC2: return "vec2";
        case GL_FLOAT_VEC3: return "vec3";
        case GL_FLOAT_VEC4: return "vec4";
        default: return NULL;
    }
}

static const char *mglGeometryPassthroughSwizzle(GLenum type)
{
    switch (type) {
        case GL_FLOAT: return ".x";
        case GL_FLOAT_VEC2: return ".xy";
        case GL_FLOAT_VEC3: return ".xyz";
        case GL_FLOAT_VEC4: return "";
        default: return NULL;
    }
}

- (BOOL)ensureAIRGeometryPassthroughFunctionForProgram:(Program *)program
                                     outputPrimitive:(MTLPrimitiveType)outputPrimitive
{
    if (!program) return NO;
    if (_geometry.passthroughLibrary && _geometry.passthroughFunction &&
        _geometry.passthroughProgramInstanceId ==
            program->pipeline_cache_instance_id) {
        return YES;
    }
    _geometry.passthroughLibrary = nil;
    _geometry.passthroughFunction = nil;
    _geometry.passthroughProgramInstanceId = 0u;

    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES];
    NSUInteger recordStride = mglAIRPerVertexStrideForResources(outputs);
    NSUInteger vec4Stride = recordStride / 16u;
    NSMutableString *source = [NSMutableString stringWithString:
        @"#version 460 core\n"
         "layout(std430, binding = 0) buffer MGLGSOutput {\n"
         "    vec4 records[];\n"
         "} mgl_gs_output;\n"];
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        /* GL 4.6 §11.1.3.4: only stream 0 is rasterized; outputs on
         * streams > 0 are transform-feedback only and must not appear
         * in the passthrough vertex function (they share location
         * values with stream 0 outputs and would cause conflicts). */
        if (output->stream > 0) continue;
        const char *type = mglGeometryPassthroughType(output->gl_type);
        if (!type || !output->name) {
            NSLog(@"MGL GS ERROR: unsupported passthrough varying type 0x%x",
                  (unsigned)output->gl_type);
            return NO;
        }
        [source appendFormat:@"layout(location = %u) out %s %s;\n",
                             (unsigned)output->location, type, output->name];
    }
    [source appendFormat:
        @"void main() {\n"
         "    int mgl_base = gl_VertexID * %lu;\n"
         "    gl_Position = mgl_gs_output.records[mgl_base];\n",
         (unsigned long)vec4Stride];
    if (outputPrimitive == MTLPrimitiveTypePoint) {
        [source appendString:
            @"    vec4 mgl_point_size = "
             "mgl_gs_output.records[mgl_base + 1];\n"
             "    gl_PointSize = mgl_point_size.x;\n"];
    }
    Shader *mgl_gs = program->shader_slots[_GEOMETRY_SHADER];
    if (mgl_gs && mgl_gs->src &&
        (strstr(mgl_gs->src, "gl_Layer") ||
         strstr(mgl_gs->src, "gl_ViewportIndex"))) {
        /* Per-vertex layer / viewport-index words (offsets 40/44) ride in
         * record vec4 index 2 (z/w); Metal applies them per primitive from
         * its last vertex, matching GL 4.6 §11.1.3.5/§11.1.3.6. */
        [source appendString:
            @"    vec4 mgl_layer_vp = "
             "mgl_gs_output.records[mgl_base + 2];\n"
             "    gl_Layer = floatBitsToInt(mgl_layer_vp.z);\n"
             "    gl_ViewportIndex = floatBitsToInt(mgl_layer_vp.w);\n"];
     }
     for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
         MGLShaderResource *output = &outputs->list[i];
         if (output->is_per_patch) continue;
         if (output->stream > 0) continue;
         const char *swizzle = mglGeometryPassthroughSwizzle(output->gl_type);
         if (!swizzle || !output->name) return NO;
         [source appendFormat:
             @"    vec4 mgl_slot_%u = "
              "mgl_gs_output.records[mgl_base + %u];\n"
              "    %s = mgl_slot_%u%s;\n",
             (unsigned)i,
             (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
output->name, (unsigned)i,
             swizzle];
    }
    [source appendString:@"}\n"];
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG passthrough source:\n%@", source);
    }
    unsigned char *bytes = NULL;
    size_t size = 0u;
    char errorText[512] = {0};
    if (mglShaderCompileGLSL(source.UTF8String, MGL_STAGE_VERTEX, &bytes, &size,
                             errorText, sizeof(errorText)) != 0 ||
        !bytes || size == 0u) {
        NSLog(@"MGL GS ERROR: failed to compile AIR passthrough vertex: %s",
              errorText[0] ? errorText : "?");
        mglShaderFree(bytes);
        return NO;
    }
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    BOOL loaded = mglLoadAIRMainFunction(
        _device, bytes, size, &library, &function,
        errorText, sizeof(errorText));
    mglShaderFree(bytes);
    if (!loaded || !library || !function) {
        NSLog(@"MGL GS ERROR: failed to load AIR passthrough vertex: %s",
              errorText[0] ? errorText : "?");
        return NO;
    }
    _geometry.passthroughLibrary = library;
    _geometry.passthroughFunction = function;
    _geometry.passthroughProgramInstanceId =
        program->pipeline_cache_instance_id;
    return YES;
}

/* TES-compute twin of ensureAIRGeometryPassthroughFunctionForProgram: the
 * isolines/point-mode TES kernel expands one vertex record per work item,
 * so the raster stage is a GLSL passthrough vertex reading the same
 * record layout as the GS expansion (position at 0, point size at 1,
 * varyings at MGL_AIR_PER_VERTEX_STRIDE + location*16).  The records come
 * from the TES stage output resource list. */
- (BOOL)ensureAIRTessEvalPassthroughFunctionForProgram:(Program *)program
{
    if (!program) return NO;
    if (_tessellation.tessPassthroughLibrary &&
        _tessellation.tessPassthroughFunction &&
        _tessellation.tessPassthroughProgramInstanceId ==
            program->pipeline_cache_instance_id) {
        return YES;
    }
    _tessellation.tessPassthroughLibrary = nil;
    _tessellation.tessPassthroughFunction = nil;
    _tessellation.tessPassthroughProgramInstanceId = 0u;

    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_TESS_EVALUATION_SHADER][_STAGE_OUTPUT_RES];
    NSUInteger recordStride = mglAIRPerVertexStrideForResources(outputs);
    NSUInteger vec4Stride = recordStride / 16u;
    NSMutableString *source = [NSMutableString stringWithString:
        @"#version 460 core\n"
         "layout(std430, binding = 0) buffer MGLTESOutput {\n"
         "    vec4 records[];\n"
         "} mgl_tes_output;\n"];
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        const char *type = mglGeometryPassthroughType(output->gl_type);
        if (!type || !output->name) {
            NSLog(@"MGL TESS ERROR: unsupported passthrough varying type 0x%x",
                  (unsigned)output->gl_type);
            return NO;
        }
        [source appendFormat:@"layout(location = %u) out %s %s;\n",
                             (unsigned)output->location, type, output->name];
    }
    [source appendFormat:
        @"void main() {\n"
         "    int mgl_base = gl_VertexID * %lu;\n"
         "    gl_Position = mgl_tes_output.records[mgl_base];\n"
         "    vec4 mgl_point_size = "
         "mgl_tes_output.records[mgl_base + 1];\n"
         "    gl_PointSize = mgl_point_size.x;\n",
         (unsigned long)vec4Stride];
    if (program->tess_cull_distance_count > 0u) {
        /* Post-tess gl_CullDistance culling (GL 4.6 §13.6.1): the record
         * carries 8 distances at bytes 20..51 (vec4 base+1 .yzw, base+2
         * .xyzw, base+3 .x).  A point is culled when any distance < 0; an
         * isoline segment when both endpoints' distance < 0 for the same
         * axis (the partner record index is gl_VertexID^1 — every patch
         * span holds an even item count).  Culled vertices are pushed out
         * of the clip volume so no fragment is produced. */
        if (program->tess_gen_mode == GL_ISOLINES) {
            /* Both endpoints of an isoline segment share the same v, so the
             * cull condition needs the partner record's distances.  The
             * partner record index is (gl_VertexID ^ 1) -- every patch span
             * holds an even item count. */
            [source appendFormat:
                @"    int mgl_partner = (gl_VertexID ^ 1) * %lu;\n"
                 "    vec4 mgl_p0 = mgl_tes_output.records[mgl_partner + 1];\n"
                 "    vec4 mgl_p1 = mgl_tes_output.records[mgl_partner + 2];\n"
                 "    vec4 mgl_p2 = mgl_tes_output.records[mgl_partner + 3];\n",
                 (unsigned long)vec4Stride];
        }
        [source appendFormat:
            @"    vec4 mgl_c0 = mgl_tes_output.records[mgl_base + 1];\n"
             "    vec4 mgl_c1 = mgl_tes_output.records[mgl_base + 2];\n"
             "    vec4 mgl_c2 = mgl_tes_output.records[mgl_base + 3];\n"
             "    bool mgl_culled = false\n"
             "%s"
             "    if (mgl_culled) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);\n",
            program->tess_gen_mode == GL_ISOLINES
                ? "        || (mgl_c0.y < 0.0 && mgl_p0.y < 0.0)\n"
                  "        || (mgl_c0.z < 0.0 && mgl_p0.z < 0.0)\n"
                  "        || (mgl_c0.w < 0.0 && mgl_p0.w < 0.0)\n"
                  "        || (mgl_c1.x < 0.0 && mgl_p1.x < 0.0)\n"
                  "        || (mgl_c1.y < 0.0 && mgl_p1.y < 0.0)\n"
                  "        || (mgl_c1.z < 0.0 && mgl_p1.z < 0.0)\n"
                  "        || (mgl_c1.w < 0.0 && mgl_p1.w < 0.0)\n"
                  "        || (mgl_c2.x < 0.0 && mgl_p2.x < 0.0);\n"
                : "        || mgl_c0.y < 0.0\n"
                  "        || mgl_c0.z < 0.0\n"
                  "        || mgl_c0.w < 0.0\n"
                  "        || mgl_c1.x < 0.0\n"
                  "        || mgl_c1.y < 0.0\n"
                  "        || mgl_c1.z < 0.0\n"
                  "        || mgl_c1.w < 0.0\n"
                  "        || mgl_c2.x < 0.0;\n"];
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        const char *swizzle = mglGeometryPassthroughSwizzle(output->gl_type);
        if (!swizzle || !output->name) return NO;
        [source appendFormat:
            @"    vec4 mgl_slot_%u = "
             "mgl_tes_output.records[mgl_base + %u];\n"
             "    %s = mgl_slot_%u%s;\n",
            (unsigned)i,
            (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
            output->name, (unsigned)i,
            swizzle];
    }
    [source appendString:@"}\n"];
    unsigned char *bytes = NULL;
    size_t size = 0u;
    char errorText[512] = {0};
    if (mglShaderCompileGLSL(source.UTF8String, MGL_STAGE_VERTEX, &bytes, &size,
                             errorText, sizeof(errorText)) != 0 ||
        !bytes || size == 0u) {
        NSLog(@"MGL TESS ERROR: failed to compile AIR TES passthrough vertex: %s\nSOURCE:\n%@",
              errorText[0] ? errorText : "?", source);
        mglShaderFree(bytes);
        return NO;
    }
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    BOOL loaded = mglLoadAIRMainFunction(
        _device, bytes, size, &library, &function,
        errorText, sizeof(errorText));
    mglShaderFree(bytes);
    if (!loaded || !library || !function) {
        NSLog(@"MGL TESS ERROR: failed to load AIR TES passthrough vertex: %s",
              errorText[0] ? errorText : "?");
        return NO;
    }
    _tessellation.tessPassthroughLibrary = library;
    _tessellation.tessPassthroughFunction = function;
    _tessellation.tessPassthroughProgramInstanceId =
        program->pipeline_cache_instance_id;
    return YES;
}

- (void)mtlInvalidateRenderPass:(GLMContext)glm_ctx
{
    if (!glm_ctx || glm_ctx != ctx || !_renderPassManager.state->currentRenderEncoder) {
        return;
    }

    /*
     * When the active framebuffer matches the render-pass framebuffer and the
     * draw-buffer selection hasn't changed, the invalidation is a spurious
     * side-effect of state setup (e.g. glDrawBuffers restoring the same values
     * after a glBindFramebuffer round-trip).  Skip it — no content is lost.
     */
    Framebuffer *curFbo = MGL_STATE(glm_ctx)->framebuffer;
    MGLRenderCppRenderPassIdentityState identity =
        mglRenderPassIdentitySnapshot(_renderPassManager.state);
    if (curFbo == identity.framebuffer &&
        MGL_STATE(glm_ctx)->draw_buffer == identity.draw_buffer) {
        return;
    }

    static uint64_t s_renderPassInvalidateCount = 0;
    uint64_t hit = ++s_renderPassInvalidateCount;
    if (mglTraceLogIsEnabled() && (hit <= 64ull || (hit % 512ull) == 0ull)) {
        Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
        mglTraceLog("RENDERPASS_INVALIDATE hit=%llu fbo=%u(%p) drawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
                    (unsigned long long)hit,
                    (unsigned)(fbo ? fbo->name : 0u),
                    fbo,
                    (unsigned)MGL_STATE(glm_ctx)->draw_buffer,
                    (unsigned)_renderPassManager.state->renderPassFramebufferName,
                    _renderPassManager.state->renderPassFramebuffer,
                    (unsigned)_renderPassManager.state->renderPassDrawBuffer);
        mglLogRenderPassLifecycle("invalidate-before-end",
                                  hit,
                                  glm_ctx,
                                  (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                  _renderPassManager.state->currentRenderEncoder,
                                  _renderPassManager.state->renderPassDescriptor,
                                  _drawable,
                                  _renderPassManager.state->renderPassFramebuffer,
                                  _renderPassManager.state->renderPassFramebufferName,
                                  _renderPassManager.state->renderPassDrawBuffer,
                                  _renderPassManager.state->renderPassDrawBufferCount);
    }

    [self flushDrawBuffer:glm_ctx];
    [self endRenderEncoding];
}

- (Texture *)framebufferAttachmentTexture: (FBOAttachment *)fbo_attachment
{
    Texture *tex = NULL;

    if (!fbo_attachment) {
        NSLog(@"MGL ERROR: framebufferAttachmentTexture called with NULL attachment");
        return NULL;
    }

    if (fbo_attachment->textarget == GL_RENDERBUFFER)
    {
        if (fbo_attachment->buf.rbo) {
            tex = fbo_attachment->buf.rbo->tex;
        }
    }
    else
    {
        tex = fbo_attachment->buf.tex;
        if (!tex && fbo_attachment->texture != 0 && fbo_attachment->textarget != GL_RENDERBUFFER)
        {
            tex = findTexture(ctx, fbo_attachment->texture);
            if (tex)
            {
                fbo_attachment->buf.tex = tex;
            }
        }
    }
    if (!tex) {
        NSLog(@"MGL WARN: framebuffer attachment has no texture (target=0x%x)", fbo_attachment->textarget);
    }

    return tex;
}

- (bool)currentRenderPassMatchesCurrentFramebuffer
{
    if (!ctx ||
        (!_renderPassManager.state->renderPassDescriptor &&
         !_renderPassManager.state->renderPassStateOwner)) {
        return true;
    }

    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLuint fboName = fbo ? fbo->name : 0u;

    /* Fast path — cached result for non-default FBOs when attachment config
     * and render pass are unchanged.  Invalidated by MGLRenderPassManager on
     * encoder/descriptor/identity changes.  Default framebuffer is never
     * cached (its inputs change independently of fbo_attachment_generation). */
    if (fbo != NULL && fboName != 0u) {
        MGLRenderCppFboMatchCacheState cache = {0};
        if (mglRenderPassUsesMetalCpp() &&
            _renderPassManager.state->renderPassIdentityOwner &&
            mglRenderCppGetFboMatchCache(
                _renderPassManager.state->renderPassIdentityOwner,
                &cache) == 0 &&
            cache.fbo_name == fboName &&
            cache.generation == fbo->fbo_attachment_generation) {
            return cache.result != 0;
        }
        if ((!mglRenderPassUsesMetalCpp() ||
             !_renderPassManager.state->renderPassIdentityOwner) &&
            _renderPassManager.state->lastFboMatchFboName == fboName &&
            _renderPassManager.state->lastFboMatchFboGeneration ==
                fbo->fbo_attachment_generation) {
            return _renderPassManager.state->lastFboMatchResult;
        }
    }

    bool result = [self mglRenderPassMatchesFramebufferImpl:fbo name:fboName];

    /* store cache for non-default FBOs only. */
    if (fbo != NULL && fboName != 0u) {
        [_renderPassManager setFboMatchCacheResult:result
                                           fboName:fboName
                                        generation:fbo->fbo_attachment_generation];
    }

    return result;
}

- (bool)mglRenderPassMatchesFramebufferImpl:(Framebuffer *)fbo name:(GLuint)fboName
{
    MGLRenderCppRenderPassState passState = {0};
    bool hasPassState = mglRenderPassUsesMetalCpp() &&
        mglRenderPassGetPersistentState(_renderPassManager.state, &passState);
    if (!ctx ||
        (!hasPassState && !_renderPassManager.state->renderPassDescriptor)) {
        return true;
    }
    MGLRenderCppRenderPassIdentityState identity =
        mglRenderPassIdentitySnapshot(_renderPassManager.state);
    if (identity.framebuffer != fbo ||
        identity.framebuffer_name != fboName ||
        identity.draw_buffer != MGL_STATE(ctx)->draw_buffer ||
        identity.draw_buffer_count != (uint32_t)mglMetalDrawBufferCount(ctx)) {
        return false;
    }
    for (uint32_t i = 0; i < identity.draw_buffer_count; ++i) {
        if (identity.draw_buffers[i] != mglMetalDrawBufferAt(ctx, i)) {
            return false;
        }
    }

    if (!fbo) {
        GLuint mgl_drawbuffer = mglDefaultDrawBufferIndexForGL(MGL_STATE(ctx)->draw_buffer);
        id<MTLTexture> expectedColor0 = nil;
        id<MTLTexture> actualColor0 = hasPassState
            ? mglRenderPassTextureFromSnapshot(
                  &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0)
            : _renderPassManager.state->renderPassDescriptor
                  .colorAttachments[0].texture;

        if (mgl_drawbuffer == _FRONT) {
            expectedColor0 = _drawable ? _drawable.texture : nil;
        } else if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            expectedColor0 = _drawBuffers[mgl_drawbuffer].drawbuffer;
        }

        if (actualColor0 != expectedColor0) {
            return false;
        }

        id<MTLTexture> expectedDepth = nil;
        id<MTLTexture> expectedStencil = nil;
        if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            BOOL defaultPassNeedsDepth = MGL_STATE(ctx)->caps.depth_test ||
                                         _drawBuffers[mgl_drawbuffer].depthbuffer != nil;
            BOOL defaultPassNeedsStencil = MGL_STATE(ctx)->caps.stencil_test ||
                                           ctx->stencil_format.format ||
                                           _drawBuffers[mgl_drawbuffer].stencilbuffer != nil;
            expectedDepth = defaultPassNeedsDepth ? _drawBuffers[mgl_drawbuffer].depthbuffer : nil;
            expectedStencil = defaultPassNeedsStencil ? _drawBuffers[mgl_drawbuffer].stencilbuffer : nil;
            if (MGL_STATE(ctx)->caps.depth_test && !expectedDepth) {
                return false;
            }
            if ((MGL_STATE(ctx)->caps.stencil_test || ctx->stencil_format.format) && !expectedStencil) {
                return false;
            }
        }

        id<MTLTexture> actualDepth = hasPassState
            ? mglRenderPassTextureFromSnapshot(
                  &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0)
            : mglRenderPassDepthTextureFor(_renderPassManager.state);
        id<MTLTexture> actualStencil = hasPassState
            ? mglRenderPassTextureFromSnapshot(
                  &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0)
            : mglRenderPassStencilTextureFor(_renderPassManager.state);
        if (actualDepth != expectedDepth) {
            return false;
        }
        if (actualStencil != expectedStencil) {
            return false;
        }

        return true;
    }

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        BOOL drawSlotPresent =
            mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) != 0u;
        FBOAttachment *attachment = drawSlotPresent ? &fbo->color_attachments[attachmentIndex] : NULL;
        Texture *tex = drawSlotPresent ? [self framebufferAttachmentTexture:attachment] : NULL;
        id<MTLTexture> expected = nil;

        if (tex) {
            tex->is_render_target = true;
            if (!tex->mtl_data) {
                if (![self bindMTLTexture:tex]) {
                    return false;
                }
            }
            expected = (__bridge id<MTLTexture>)(tex->mtl_data);
        }

        id<MTLTexture> actual = hasPassState
            ? mglRenderPassTextureFromSnapshot(
                  &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
                  colorSlot)
            : _renderPassManager.state->renderPassDescriptor
                  .colorAttachments[colorSlot].texture;
        if (actual != expected) {
            return false;
        }

        if (attachment && actual) {
            MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(attachment);
            bool matches = hasPassState
                ? mglRenderPassSnapshotAttachmentMatchesSubresource(
                      &passState,
                      MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
                      colorSlot, subresource)
                : mglMetalRenderPassColorAttachmentMatchesSubresource(
                      _renderPassManager.state->renderPassDescriptor
                          .colorAttachments[colorSlot],
                      subresource);
            if (!matches) {
                return false;
            }
        }

        id<MTLTexture> nextColor = nil;
        if (i + 1u < MAX_COLOR_ATTACHMENTS) {
            nextColor = hasPassState
                ? mglRenderPassTextureFromSnapshot(
                      &passState,
                      MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
                      i + 1u)
                : _renderPassManager.state->renderPassDescriptor
                      .colorAttachments[i + 1u].texture;
        }
        if (i + 1u >= MAX_COLOR_ATTACHMENTS ||
            (mglMetalDrawBufferAt(ctx, i + 1u) == GL_NONE &&
             !nextColor)) {
            break;
        }
    }

    id<MTLTexture> expectedDepth = nil;
    if (fbo->depth.texture) {
        Texture *depthTex = [self framebufferAttachmentTexture:&fbo->depth];
        if (depthTex && !depthTex->mtl_data) {
            depthTex->is_render_target = true;
            if (![self bindMTLTexture:depthTex]) {
                return false;
            }
        }
        expectedDepth = depthTex ? (__bridge id<MTLTexture>)(depthTex->mtl_data) : nil;
    }
    id<MTLTexture> actualDepth = hasPassState
        ? mglRenderPassTextureFromSnapshot(
              &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0)
        : mglRenderPassDepthTextureFor(_renderPassManager.state);
    if (actualDepth != expectedDepth) {
        return false;
    }
    if (fbo->depth.texture && expectedDepth) {
        MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
        bool matches = hasPassState
            ? mglRenderPassSnapshotAttachmentMatchesSubresource(
                  &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH,
                  0, subresource)
            : mglMetalRenderPassDepthAttachmentMatchesSubresource(
                  _renderPassManager.state->renderPassDescriptor.depthAttachment,
                  subresource);
        if (!matches) {
            return false;
        }
    }

    id<MTLTexture> expectedStencil = nil;
    if (fbo->stencil.texture) {
        Texture *stencilTex = [self framebufferAttachmentTexture:&fbo->stencil];
        if (stencilTex && !stencilTex->mtl_data) {
            stencilTex->is_render_target = true;
            if (![self bindMTLTexture:stencilTex]) {
                return false;
            }
        }
        expectedStencil = stencilTex ? (__bridge id<MTLTexture>)(stencilTex->mtl_data) : nil;
    }
    id<MTLTexture> actualStencil = hasPassState
        ? mglRenderPassTextureFromSnapshot(
              &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0)
        : mglRenderPassStencilTextureFor(_renderPassManager.state);
    if (actualStencil != expectedStencil) {
        return false;
    }
    if (fbo->stencil.texture && expectedStencil) {
        MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
        bool matches = hasPassState
            ? mglRenderPassSnapshotAttachmentMatchesSubresource(
                  &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL,
                  0, subresource)
            : mglMetalRenderPassStencilAttachmentMatchesSubresource(
                  _renderPassManager.state->renderPassDescriptor.stencilAttachment,
                  subresource);
        if (!matches) {
            return false;
        }
    }

    return true;
}

- (bool)ensureCurrentRenderPassMatchesFramebufferForDraw
{
    if (!ctx) {
        return true;
    }

    if (!_renderPassManager.state->currentRenderEncoder) {
        return true;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return true;
    }

    static uint64_t s_fboPassMismatchCount = 0;
    uint64_t hit = ++s_fboPassMismatchCount;
    if (hit <= 32ull || (hit % 256ull) == 0ull) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
        id<MTLTexture> color0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
        GLuint mglDefaultDrawbuffer = fbo ? 0u : mglDefaultDrawBufferIndexForGL(MGL_STATE(ctx)->draw_buffer);
        id<MTLTexture> expectedDefaultColor0 = nil;
        if (!fbo) {
            expectedDefaultColor0 = (mglDefaultDrawbuffer == _FRONT)
                ? (_drawable ? _drawable.texture : nil)
                : ((mglDefaultDrawbuffer < _MAX_DRAW_BUFFERS) ? _drawBuffers[mglDefaultDrawbuffer].drawbuffer : nil);
        }
        GLuint fboName = fbo ? fbo->name : 0u;
        GLuint attachment0Name = (fbo && (fbo->color_attachment_bitfield & 1u)) ? fbo->color_attachments[0].texture : 0u;
        NSLog(@"MGL WARNING: render pass/FBO mismatch before draw hit=%llu fbo=%u drawBuffer=0x%x attachment0=%u passColor0=%p expectedDefaultColor0=%p defaultDrawBuffer=%u; rebuilding encoder",
              (unsigned long long)hit,
              (unsigned)fboName,
              (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
              (unsigned)attachment0Name,
              color0,
              expectedDefaultColor0,
              (unsigned)mglDefaultDrawbuffer);
        mglLogRenderPassLifecycle(fbo ? "fbo-mismatch-before-rebuild" : "default-fbo-mismatch-before-rebuild",
                                  hit,
                                  ctx,
                                  (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                  _renderPassManager.state->currentRenderEncoder,
                                  _renderPassManager.state->renderPassDescriptor,
                                  _drawable,
                                  _renderPassManager.state->renderPassFramebuffer,
                                  _renderPassManager.state->renderPassFramebufferName,
                                  _renderPassManager.state->renderPassDrawBuffer,
                                  _renderPassManager.state->renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM |
                             DIRTY_RENDER_STATE | DIRTY_VAO);
    return [self newRenderEncoderWithReason:MGL_ENC_REASON_FBO];
}

- (void)endRenderPassIfFramebufferChangedForNonDraw:(uint64_t)processCall
{
    if (!ctx || !_renderPassManager.state->currentRenderEncoder) {
        return;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return;
    }

    static uint64_t s_nonDrawFboMismatchCount = 0;
    uint64_t hit = ++s_nonDrawFboMismatchCount;
    if (mglTraceLogIsEnabled() && (hit <= 32ull || (hit % 256ull) == 0ull)) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
        GLuint fboName = fbo ? fbo->name : 0u;
        mglTraceLog("RENDERPASS_NON_DRAW_MISMATCH processCall=%llu hit=%llu "
                    "ctxFbo=%u(%p) ctxDrawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
                    (unsigned long long)processCall,
                    (unsigned long long)hit,
                    (unsigned)fboName,
                    fbo,
                    (unsigned)MGL_STATE(ctx)->draw_buffer,
                    (unsigned)_renderPassManager.state->renderPassFramebufferName,
                    _renderPassManager.state->renderPassFramebuffer,
                    (unsigned)_renderPassManager.state->renderPassDrawBuffer);
        mglLogRenderPassLifecycle("non-draw-mismatch-before-end",
                                  hit,
                                  ctx,
                                  (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                  _renderPassManager.state->currentRenderEncoder,
                                  _renderPassManager.state->renderPassDescriptor,
                                  _drawable,
                                  _renderPassManager.state->renderPassFramebuffer,
                                  _renderPassManager.state->renderPassFramebufferName,
                                  _renderPassManager.state->renderPassDrawBuffer,
                                  _renderPassManager.state->renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE);
}

- (bool)restoreRenderEncoderAfterTextureUploadForDraw:(const char *)reason
{
    if (_renderPassManager.state->currentRenderEncoder) {
        return true;
    }
    MGLRenderCppRenderPassState passState = {0};
    bool hasPassState = mglRenderPassUsesMetalCpp() &&
        mglRenderPassGetPersistentState(_renderPassManager.state, &passState);
    if (!ctx ||
        (!hasPassState && !_renderPassManager.state->renderPassDescriptor)) {
        return false;
    }

    static uint64_t s_restoreAfterTextureUploadCount = 0;
    uint64_t hit = ++s_restoreAfterTextureUploadCount;
    if (hit <= 16ull || (hit % 2048ull) == 0ull) {
        NSLog(@"MGL TEXTURE UPLOAD closed render encoder; restoring for draw reason=%s hit=%llu",
              reason ? reason : "(null)",
              (unsigned long long)hit);
    }

    if (![self ensureWritableCommandBuffer:reason ? reason : "restore_render_encoder_after_texture_upload"]) {
        return false;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> texture = hasPassState
            ? mglRenderPassTextureFromSnapshot(
                  &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i)
            : _renderPassManager.state->renderPassDescriptor
                  .colorAttachments[i].texture;
        if (texture) {
            mglRenderPassSetPersistentActions(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i,
                MTLLoadActionLoad, MTLStoreActionStore);
        }
    }
    id<MTLTexture> depthTexture = hasPassState
        ? mglRenderPassTextureFromSnapshot(
              &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0)
        : mglRenderPassDepthTextureFor(_renderPassManager.state);
    if (depthTexture) {
        mglRenderPassSetPersistentActions(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MTLLoadActionLoad, MTLStoreActionStore);
    }
    id<MTLTexture> stencilTexture = hasPassState
        ? mglRenderPassTextureFromSnapshot(
              &passState, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0)
        : mglRenderPassStencilTextureFor(_renderPassManager.state);
    if (stencilTexture) {
        mglRenderPassSetPersistentActions(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MTLLoadActionLoad, MTLStoreActionStore);
    }

    @try {
        id<MTLRenderCommandEncoder> renderEncoder =
            [_renderPassManager createRenderEncoderWithDescriptor:
                _renderPassManager.state->renderPassDescriptor];
        [_renderPassManager installRenderEncoder:renderEncoder];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload failed to create encoder: %@",
              exception.reason);
        [_renderPassManager clearCurrentRenderEncoder];
        [self recordGPUError];
        return false;
    }
    if (!_renderPassManager.state->currentRenderEncoder) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload returned nil encoder reason=%s",
              reason ? reason : "(null)");
        [self recordGPUError];
        return false;
    }
    _renderPassManager.state->currentRenderEncoder.label = @"GL Render Encoder";
    /* When trace is disabled, skip the full-struct memset and trace call
     * and clear only the functional flag fields. */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             reason ? reason : "restore_render_encoder_after_texture_upload",
                                             _resourceFallback.fragmentTextureTraceBindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             _pipelineCache.state->pipelineProgramName);
        memset(_resourceFallback.fragmentTextureTraceBindings, 0,
               sizeof(_resourceFallback.fragmentTextureTraceBindings));
    } else {
        mglClearFragmentTextureTraceFunctionalFlags(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
    }
    [_renderPassManager updateRenderPassIdentityForContext:ctx];
    [self updateCurrentRenderEncoder];

    if (!_pipelineCache.state->pipelineState) {
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    @try {
        if (mglRenderCppBindingSetPipelineIfNeeded(
                _bindingStateOwner,
                (__bridge void *)_renderPassManager.state->currentRenderEncoder,
                (__bridge void *)_pipelineCache.state->pipelineState) > 0) {
            MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload failed to bind pipeline: %@",
              exception.reason);
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    MGLEncodeContext encCtx = { .encoder = _renderPassManager.state->currentRenderEncoder };
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]);
    return true;
}

- (bool)bindFramebufferTexture:(FBOAttachment *)fbo_attachment isDrawBuffer:(bool) isDrawBuffer
{
    Texture *tex;

    tex = [self framebufferAttachmentTexture: fbo_attachment];
    if (!tex) {
        // Incomplete/missing attachment. Do not crash.
        return true;
    }

    if (isDrawBuffer) {
        tex->is_render_target = true;
    }

    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);

    return true;
}

- (void)invalidateCurrentPipelineStateForReason:(NSString *)reason
{
    if (_pipelineCache.state->pipelineState) {
        static uint64_t s_pipelineInvalidateCount = 0;
        uint64_t hit = ++s_pipelineInvalidateCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL WARNING: Invalidating current pipeline state after %@ hit=%llu",
                  reason ?: @"pipeline failure",
                  (unsigned long long)hit);
        }
    }
    [_pipelineCache invalidatePipelineState];
}

-(bool)bindMTLProgram:(Program *)ptr
{
    METAL_LOCK();
    bool result = [self bindMTLProgramLocked:ptr];
    METAL_UNLOCK();
    return result;
}

-(bool)bindMTLProgramLocked:(Program *)ptr
{
    if (ptr->dirty_bits & DIRTY_PROGRAM)
    {
        /* Metal libraries/functions are linked Program products and are
         * invalidated by clearStageCompileState during relink. DIRTY_PROGRAM
         * also covers pre-link state changes, which must not discard the
         * currently linked executable. */
        ptr->dirty_bits &= ~DIRTY_PROGRAM;
    }

    if (mglRenderPassUsesMetalCpp()) {
        int failedStage = -1;
        char bindError[256] = {0};
        int bindResult = mglRenderCppBindAIRProgram(
            ptr, &failedStage, bindError, sizeof(bindError));
        if (bindResult == MGL_RENDER_CPP_AIR_PROGRAM_BOUND) {
            return true;
        }
        if (bindResult == MGL_RENDER_CPP_AIR_PROGRAM_ERROR) {
            NSLog(@"MGL ERROR: Failed to bind AIR program=%u stage=%d: %s",
                  (unsigned)ptr->name, failedStage,
                  bindError[0] ? bindError : "?");
            return false;
        }
    }

	    // Compile linked Program stages on demand.
	    for(int i=_VERTEX_SHADER; i<_MAX_SHADER_TYPES; i++)
	    {
	        Shader *shader;
	        shader = ptr->shader_slots[i];

        if (shader)
        {
            if (i == _GEOMETRY_SHADER) {
                if (mglGeometryShaderIsPassthrough(shader)) {
                    static uint64_t s_passthroughGeometryShaderSkipCount = 0;
                    uint64_t hit = ++s_passthroughGeometryShaderSkipCount;
                    if (hit <= 16ull || (hit % 512ull) == 0ull) {
                        NSLog(@"MGL INFO: Skipping passthrough geometry shader for Metal program=%u hit=%llu",
                              (unsigned)ptr->name,
                              (unsigned long long)hit);
                    }
                    continue;
                }
                if (ptr->gs_route == MGL_GS_ROUTE_COMPUTE &&
                    ptr->modules[i].metallib_bytes &&
                    ptr->modules[i].metallib_size > 0u) {
                    /* The AIR geometry stage is a compute kernel.  Load it
                     * below like TCS/CS; the draw helper owns dispatch and
                     * expanded-output rendering. */
                } else {
                static uint64_t s_geometryShaderMetalSkipCount = 0;
                uint64_t hit = ++s_geometryShaderMetalSkipCount;
                if (hit <= 16ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL WARNING: Blocking draw for unsupported geometry shader program=%u hit=%llu",
                          (unsigned)ptr->name,
                          (unsigned long long)hit);
                }
                return false;
                }
            }
            if (ptr->modules[i].metallib_bytes && ptr->modules[i].metallib_size > 0) {
                /* AIR path: the stage was compiled by the self-hosted
                 * frontend into a metallib blob; load it directly. */
                if (ptr->modules[i].mtl_library == NULL || ptr->modules[i].mtl_function == NULL) {
                    mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_function);
                    mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_library);
                    id<MTLLibrary> library = nil;
                    id<MTLFunction> function = nil;
                    char loadError[256] = {0};
                    if (!mglLoadAIRMainFunction(
                            _device, ptr->modules[i].metallib_bytes,
                            ptr->modules[i].metallib_size, &library, &function,
                            loadError, sizeof loadError)) {
                        NSLog(@"MGL ERROR: Failed to load AIR metallib program=%u stage=%d: %s",
                              (unsigned)ptr->name, i,
                              loadError[0] ? loadError : "?");
                        return false;
                    }
                    ptr->modules[i].mtl_library = (void *)CFBridgingRetain(library);
                    ptr->modules[i].mtl_function = (void *)CFBridgingRetain(function);
                }
                if (i == _VERTEX_SHADER &&
                    ptr->modules[i].metallib_tess_capture_bytes &&
                    (!ptr->modules[i].mtl_tess_capture_library ||
                     !ptr->modules[i].mtl_tess_capture_function)) {
                    id<MTLLibrary> library = nil;
                    id<MTLFunction> function = nil;
                    char loadError[256] = {0};
                    if (!mglLoadAIRMainFunction(
                            _device,
                            ptr->modules[i].metallib_tess_capture_bytes,
                            ptr->modules[i].metallib_tess_capture_size,
                            &library, &function, loadError,
                            sizeof loadError)) {
                        NSLog(@"MGL ERROR: Failed to load AIR tess VS capture program=%u: %s",
                              (unsigned)ptr->name,
                              loadError[0] ? loadError : "?");
                        return false;
                    }
                    ptr->modules[i].mtl_tess_capture_library =
                        (void *)CFBridgingRetain(library);
                    ptr->modules[i].mtl_tess_capture_function =
                        (void *)CFBridgingRetain(function);
                }
                if (i == _VERTEX_SHADER &&
                    ptr->modules[i].metallib_cull_capture_bytes &&
                    (!ptr->modules[i].mtl_cull_capture_library ||
                     !ptr->modules[i].mtl_cull_capture_function)) {
                    id<MTLLibrary> library = nil;
                    id<MTLFunction> function = nil;
                    char loadError[256] = {0};
                    if (!mglLoadAIRMainFunction(
                            _device,
                            ptr->modules[i].metallib_cull_capture_bytes,
                            ptr->modules[i].metallib_cull_capture_size,
                            &library, &function, loadError,
                            sizeof loadError)) {
                        NSLog(@"MGL ERROR: Failed to load AIR cull-distance "
                              "capture program=%u: %s",
                              (unsigned)ptr->name,
                              loadError[0] ? loadError : "?");
                        return false;
                    }
                    ptr->modules[i].mtl_cull_capture_library =
                        (void *)CFBridgingRetain(library);
                    ptr->modules[i].mtl_cull_capture_function =
                        (void *)CFBridgingRetain(function);
                }
            } else {
                NSLog(@"MGL ERROR: Program %u stage %d has no AIR metallib",
                      (unsigned)ptr->name, i);
                return false;
            }
        }
    }

	    return true;
	}

- (void) updateCurrentRenderEncoder
{
    GLMState *state = MGL_STATE(ctx);
    /* P4.1f: gate-on 下 pass 附件在 C++ owner（descriptor 镜像为 nil）。 */
    BOOL hasConfiguredRenderPass = mglRenderPassUsesMetalCpp()
        ? (_renderPassManager.state->renderPassStateOwner != NULL)
        : (_renderPassManager.state->renderPassDescriptor != nil);
    BOOL passHasDepthAttachment =
        (hasConfiguredRenderPass &&
         mglRenderPassDepthTextureFor(_renderPassManager.state) != nil);
    BOOL passHasStencilAttachment =
        (hasConfiguredRenderPass &&
         mglRenderPassStencilTextureFor(_renderPassManager.state) != nil);
    BOOL useDepthState = state->caps.depth_test && passHasDepthAttachment;
    BOOL useStencilState = state->caps.stencil_test && passHasStencilAttachment;

    if (state->caps.depth_test && !passHasDepthAttachment) {
        static uint64_t s_missingDepthAttachmentCount = 0;
        uint64_t hit = ++s_missingDepthAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: depth test/write requested without depth attachment, disabling depth for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (state->caps.stencil_test && !passHasStencilAttachment) {
        static uint64_t s_missingStencilAttachmentCount = 0;
        uint64_t hit = ++s_missingStencilAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: stencil test requested without stencil attachment, disabling stencil for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (useDepthState || useStencilState)
    {
        MTLDepthStencilDescriptor *dsDesc = [[MTLDepthStencilDescriptor alloc] init];

        if (useDepthState)
        {
            if (!mglIsValidGLCompareFunction(state->var.depth_func)) {
                mglLogRenderStateRepair("depth_func", state->var.depth_func, GL_LESS);
                state->var.depth_func = GL_LESS;
                mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
            }

            dsDesc.depthCompareFunction =
                mglMTLCompareFunctionForGL(state->var.depth_func,
                                           MTLCompareFunctionLess,
                                           "depth");
            dsDesc.depthWriteEnabled = state->var.depth_writemask;
        }

        if (useStencilState)
        {
            {
                if (!mglIsValidGLCompareFunction(state->var.stencil_func)) {
                    mglLogRenderStateRepair("stencil_func", state->var.stencil_func, GL_ALWAYS);
                    state->var.stencil_func = GL_ALWAYS;
                    mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
                }

                MTLStencilDescriptor *frontSDesc = [[MTLStencilDescriptor alloc] init];

                frontSDesc.stencilCompareFunction =
                    mglMTLCompareFunctionForGL(state->var.stencil_func,
                                               MTLCompareFunctionAlways,
                                               "front-stencil");
                frontSDesc.stencilFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_fail ];
                frontSDesc.depthFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_pass_depth_fail];
                frontSDesc.depthStencilPassOperation = [self mtlStencilOpForGLOp:state->var.stencil_pass_depth_pass];
                frontSDesc.writeMask = state->var.stencil_writemask;
                frontSDesc.readMask = state->var.stencil_value_mask;    // ????

                dsDesc.frontFaceStencil = frontSDesc;
            }

            {
                if (!mglIsValidGLCompareFunction(state->var.stencil_back_func)) {
                    mglLogRenderStateRepair("stencil_back_func", state->var.stencil_back_func, GL_ALWAYS);
                    state->var.stencil_back_func = GL_ALWAYS;
                    mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
                }

                MTLStencilDescriptor *backSDesc = [[MTLStencilDescriptor alloc] init];

                backSDesc.stencilCompareFunction =
                    mglMTLCompareFunctionForGL(state->var.stencil_back_func,
                                               MTLCompareFunctionAlways,
                                               "back-stencil");
                backSDesc.stencilFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_back_fail ];
                backSDesc.depthFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_back_pass_depth_fail];
                backSDesc.depthStencilPassOperation = [self mtlStencilOpForGLOp:state->var.stencil_back_pass_depth_pass];
                backSDesc.writeMask = state->var.stencil_back_writemask;
                backSDesc.readMask = state->var.stencil_back_value_mask;    // ????

                dsDesc.backFaceStencil = backSDesc;
            }
        }

        id<MTLDepthStencilState> dsState =
            [_pipelineCache depthStencilStateForDescriptor:dsDesc];

        if (mglRenderCppBindingSetDepthStencilIfNeeded(
                _bindingStateOwner,
                (__bridge void *)_renderPassManager.state->currentRenderEncoder,
                (__bridge void *)dsState) > 0) {
        } else {
            MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
        }
        if (useStencilState) {
            mglRenderPassSetStencilReferenceValues(
                _renderPassManager.state->currentRenderEncoder,
                (uint32_t)state->var.stencil_ref,
                (uint32_t)state->var.stencil_back_ref);
        }
    }
    else
    {
        MTLDepthStencilDescriptor *disabledDSDesc = [[MTLDepthStencilDescriptor alloc] init];
        disabledDSDesc.depthCompareFunction = MTLCompareFunctionAlways;
        disabledDSDesc.depthWriteEnabled = NO;

        id<MTLDepthStencilState> disabledDSState =
            [_pipelineCache depthStencilStateForDescriptor:disabledDSDesc];
        if (disabledDSState) {
            if (mglRenderCppBindingSetDepthStencilIfNeeded(
                    _bindingStateOwner,
                    (__bridge void *)_renderPassManager.state->currentRenderEncoder,
                    (__bridge void *)disabledDSState) > 0) {
            } else {
                MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
            }
        }
    }

    {
        float bcRed   = state->var.blend_color[0];
        float bcGreen = state->var.blend_color[1];
        float bcBlue  = state->var.blend_color[2];
        float bcAlpha = state->var.blend_color[3];
        mglRenderCppBindingSetBlendColorIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            bcRed, bcGreen, bcBlue, bcAlpha);
    }

    /* GL_SAMPLE_MASK: Metal does not expose a per-draw sample mask setter on
     * MTLRenderCommandEncoder.  Sample coverage in Metal is controlled via
     * alpha-to-coverage and shader-side [[sample_mask]], neither of which
     * maps cleanly to GL_SAMPLE_MASK.  This remains a known limitation. */

    [self updateViewportAndScissorLocked];

    if (state->var.front_face != GL_CW && state->var.front_face != GL_CCW) {
        mglLogRenderStateRepair("front_face", state->var.front_face, GL_CCW);
        state->var.front_face = GL_CCW;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }

    BOOL defaultFramebufferSampledPass =
        state->framebuffer == NULL &&
        !state->caps.depth_test &&
        [self getProgramBindingCount:_FRAGMENT_SHADER type:_SAMPLED_IMAGE_RES] > 0;
    BOOL rtSampledCopyDraw = _renderPassManager.state->currentDrawUsesRTSampledCopy;

    if (state->caps.cull_face && !defaultFramebufferSampledPass && !rtSampledCopyDraw)
    {
        MTLCullMode cull_mode;

        switch(state->var.cull_face_mode)
        {
            case GL_BACK: cull_mode = MTLCullModeBack; break;
            case GL_FRONT: cull_mode = MTLCullModeFront; break;
            default:
                cull_mode = MTLCullModeNone;
        }

        mglRenderCppBindingSetCullIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            (uint32_t)cull_mode);
        MTLWinding _winding =
            mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                     state->var.clip_origin == GL_UPPER_LEFT);
        mglRenderCppBindingSetWindingIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            (uint32_t)_winding);
    }
    else
    {
        mglRenderCppBindingSetCullIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            (uint32_t)MTLCullModeNone);
        MTLWinding _winding =
            mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                     state->var.clip_origin == GL_UPPER_LEFT);
        mglRenderCppBindingSetWindingIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            (uint32_t)_winding);

        if (state->caps.cull_face && defaultFramebufferSampledPass) {
            static uint64_t s_defaultSampledCullBypassCount = 0;
            uint64_t hit = ++s_defaultSampledCullBypassCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                mglTraceLogNSString(@"MGL TRACE default sampled pass cull bypass hit=%llu program=%u drawBuf=0x%x",
                      (unsigned long long)hit,
                      (unsigned)(ctx ? state->program_name : 0u),
                      (unsigned)(ctx ? state->draw_buffer : 0u));
            }
        }
        if (state->caps.cull_face && rtSampledCopyDraw) {
            static uint64_t s_rtSampledCopyCullBypassCount = 0;
            uint64_t hit = ++s_rtSampledCopyCullBypassCount;
            if (hit <= 64ull || (hit % 256ull) == 0ull) {
                mglTraceLog("RT_SAMPLE_COPY_CULL_BYPASS hit=%llu program=%u pipelineProgram=%u fbo=%u rpFbo=%u depth(test=%d write=%d func=0x%x) blend=%d cullFace=0x%x frontFace=0x%x",
                            (unsigned long long)hit,
                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                            (unsigned)_pipelineCache.state->pipelineProgramName,
                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                            (unsigned)_renderPassManager.state->renderPassFramebufferName,
                            (ctx && state->caps.depth_test) ? 1 : 0,
                            (ctx && state->var.depth_writemask) ? 1 : 0,
                            (unsigned)(ctx ? state->var.depth_func : 0u),
                            (ctx && state->caps.blend) ? 1 : 0,
                            (unsigned)(ctx ? state->var.cull_face_mode : 0u),
                            (unsigned)(ctx ? state->var.front_face : 0u));
            }
        }
    }

    if (state->caps.depth_clamp)
    {
        mglRenderPassSetDepthClipMode(
            _renderPassManager.state->currentRenderEncoder,
            MTLDepthClipModeClamp);
    }

    if (state->caps.polygon_offset_fill ||
        state->caps.polygon_offset_line ||
        state->caps.polygon_offset_point)
    {
        float _bias = state->var.polygon_offset_units;
        float _slope = state->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderCppBindingSetDepthBiasIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            _bias, _clamp, _slope);
    }
    else
    {
        mglRenderCppBindingSetDepthBiasIfNeeded(
            _bindingStateOwner,
            (__bridge void *)_renderPassManager.state->currentRenderEncoder,
            0.0f, 0.0f, 0.0f);
    }

    MTLTriangleFillMode triangleFillMode = MTLTriangleFillModeFill;
    if (state->var.polygon_mode == GL_LINE)
    {
        triangleFillMode = MTLTriangleFillModeLines;
    }
    else if (state->var.polygon_mode != GL_FILL &&
             state->var.polygon_mode != GL_POINT)
    {
        mglLogRenderStateRepair("polygon_mode", state->var.polygon_mode, GL_FILL);
        state->var.polygon_mode = GL_FILL;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];
}
/*
 * Viewport and scissor setup extracted from updateCurrentRenderEncoder.
 * Resolves render-pass dimensions, applies the scissor rect (with GL-to-Metal
 * origin conversion), and sets the viewport. Uses MGL_STATE(ctx) for
 * snapshot-based state access (Principle 2 compliance).
 */
- (void)updateViewportAndScissorLocked
{
    GLMState *state = MGL_STATE(ctx);
    // Metal validates viewport/scissor strictly against the active render pass dimensions.
    // Always derive pass size from the current attachments first (not from window drawable fallback).
    {
        static uint64_t s_encoderStateUpdateCount = 0;
        bool traceEncoderState = kMGLDiagnosticStateLogs || mglShouldTraceCall(++s_encoderStateUpdateCount);

        NSUInteger passWidth = 0;
        NSUInteger passHeight = 0;
        id<MTLTexture> passTexture = nil;

        /* P4.1f: gate-on 下 pass 尺寸在 C++ owner（descriptor 镜像为 nil），
         * 用 owner 存在性作为「pass 已配置」信号，否则会回退到 drawable
         * 尺寸（200x200）做 GL→Metal 视口换算，viewport 落在 128x128
         * render target 之外 → 所有 draw 静默空转。 */
        BOOL hasConfiguredRenderPass = mglRenderPassUsesMetalCpp()
            ? (_renderPassManager.state->renderPassStateOwner != NULL)
            : (_renderPassManager.state->renderPassDescriptor != nil);
        if (hasConfiguredRenderPass) {
            passWidth = mglRenderPassRenderTargetWidthFor(_renderPassManager.state);
            passHeight = mglRenderPassRenderTargetHeightFor(_renderPassManager.state);

            if (passWidth == 0 || passHeight == 0) {
                for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                    id<MTLTexture> candidate = mglRenderPassColorTextureFor(_renderPassManager.state, i);
                    if (candidate) {
                        passTexture = candidate;
                        break;
                    }
                }

                if (!passTexture) {
                    passTexture = mglRenderPassDepthTextureFor(_renderPassManager.state);
                }
                if (!passTexture) {
                    passTexture = mglRenderPassStencilTextureFor(_renderPassManager.state);
                }

                if (passTexture) {
                    passWidth = passTexture.width;
                    passHeight = passTexture.height;
                    mglRenderPassSetPersistentDimensions(
                        _renderPassManager.state, passWidth, passHeight);
                    if (kMGLVerboseFrameLoopLogs) {
                        NSLog(@"MGL INFO: Resolved render pass size from attachment %lux%lu (rtw/rth were unset)",
                              (unsigned long)passWidth, (unsigned long)passHeight);
                    }
                }
            }
        }

        if ((passWidth == 0 || passHeight == 0) && _drawable && _drawable.texture) {
            passWidth = _drawable.texture.width;
            passHeight = _drawable.texture.height;
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to drawable size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if ((passWidth == 0 || passHeight == 0) && _layer) {
            CGSize drawableSize = _layer.drawableSize;
            if (drawableSize.width > 0 && drawableSize.height > 0) {
                passWidth = (NSUInteger)drawableSize.width;
                passHeight = (NSUInteger)drawableSize.height;
            } else {
                NSRect frame = [_layer frame];
                if (frame.size.width > 0 && frame.size.height > 0) {
                    passWidth = (NSUInteger)frame.size.width;
                    passHeight = (NSUInteger)frame.size.height;
                }
            }
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to layer size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if (passWidth > 0 && passHeight > 0) {
            GLint rawSx = 0;
            GLint rawSy = 0;
            GLint rawSw = (GLint)passWidth;
            GLint rawSh = (GLint)passHeight;

            GLint sx = 0;
            GLint sy = 0;
            GLint sw = (GLint)passWidth;
            GLint sh = (GLint)passHeight;

            if (state->caps.scissor_test) {
                rawSx = (GLint)state->var.scissor_box[0];
                rawSy = (GLint)state->var.scissor_box[1];
                rawSw = (GLint)state->var.scissor_box[2];
                rawSh = (GLint)state->var.scissor_box[3];

                sx = rawSx;
                sy = rawSy;
                sw = rawSw;
                sh = rawSh;

                // GL allows negative x/y; clamp origin and shrink extent accordingly.
                if (sx < 0) {
                    sw += sx;
                    sx = 0;
                }
                if (sy < 0) {
                    sh += sy;
                    sy = 0;
                }

                if (sx >= (GLint)passWidth || sy >= (GLint)passHeight) {
                    /* GL 4.6 §14.6.1: a scissor box that starts entirely
                     * outside the viewport clips every fragment.  Keep an
                     * empty (0-size) box instead of resetting to full pass. */
                    sx = 0;
                    sy = 0;
                    sw = 0;
                    sh = 0;
                } else {
                    GLint maxWidth = (GLint)passWidth - sx;
                    GLint maxHeight = (GLint)passHeight - sy;

                    if (sw > maxWidth) {
                        sw = maxWidth;
                    }
                    if (sh > maxHeight) {
                        sh = maxHeight;
                    }

                    if (sw <= 0 || sh <= 0) {
                        /* GL allows 0-size scissor rects; they produce no
                         * fragments.  Do not reset to the full pass. */
                        sx = 0;
                        sy = 0;
                        sw = 0;
                        sh = 0;
                    }
                }
            }

            GLint metalSy = sy;
            if (state->var.clip_origin != GL_UPPER_LEFT) {
                metalSy = (GLint)passHeight - (sy + sh);
                if (metalSy < 0) {
                    metalSy = 0;
                }
            }

	            if (traceEncoderState) {
                NSLog(@"MGL SCISSOR apply pass=%lux%lu scissorEnabled=%d origin=0x%x raw=(%d,%d,%d,%d) glResolved=(%d,%d,%d,%d) metal=(%d,%d,%d,%d)",
                      (unsigned long)passWidth, (unsigned long)passHeight,
                      state->caps.scissor_test ? 1 : 0,
                      state->var.clip_origin,
                      rawSx, rawSy, rawSw, rawSh,
                      sx, sy, sw, sh,
                      sx, metalSy, sw, sh);
            }

            MTLScissorRect rect;
            rect.x = (NSUInteger)sx;
            rect.y = (NSUInteger)metalSy;
            rect.width = (NSUInteger)sw;
            rect.height = (NSUInteger)sh;
            [self setScissorRectIfNeeded:rect];

            GLdouble rawVx = (GLdouble)state->viewport[0];
            GLdouble rawVy = (GLdouble)state->viewport[1];
            GLdouble rawVw = (GLdouble)state->viewport[2];
            GLdouble rawVh = (GLdouble)state->viewport[3];

            GLdouble vx = rawVx;
            GLdouble vy = rawVy;
            GLdouble vw = rawVw;
            GLdouble vh = rawVh;

            if (vw <= 0.0 || vh <= 0.0) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            }

            if (vx < 0.0) {
                vw += vx;
                vx = 0.0;
            }
            if (vy < 0.0) {
                vh += vy;
                vy = 0.0;
            }

            if (vx >= (GLdouble)passWidth || vy >= (GLdouble)passHeight) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            } else {
                GLdouble maxVw = (GLdouble)passWidth - vx;
                GLdouble maxVh = (GLdouble)passHeight - vy;
                if (vw > maxVw) {
                    vw = maxVw;
                }
                if (vh > maxVh) {
                    vh = maxVh;
                }
                if (vw <= 0.0 || vh <= 0.0) {
                    vx = 0.0;
                    vy = 0.0;
                    vw = (GLdouble)passWidth;
                    vh = (GLdouble)passHeight;
                }
            }

            /*
             * glViewport's x/y select the same framebuffer rectangle regardless
             * of glClipControl origin.  The origin only changes how clip-space Y
             * maps within that rectangle; Metal still addresses the texture from
             * the top, so always convert GL's lower-left viewport rectangle to a
             * Metal top-left origin here.
             */
            GLdouble metalVy = (GLdouble)passHeight - (vy + vh);
            if (metalVy < 0.0) {
                metalVy = 0.0;
            }

            Texture *guiRTColor = NULL;
            Texture *guiRTDepth = NULL;
            BOOL guiRTPass =
                mglTraceLogIsEnabled() &&
                mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx,
                                                                 state->framebuffer,
                                                                 &guiRTColor,
                                                                 &guiRTDepth);
            if (guiRTPass) {
                static uint64_t s_guiRTEncoderStateLogCount = 0;
                uint64_t hit = ++s_guiRTEncoderStateLogCount;
                if (hit <= 128ull || (hit % 256ull) == 0ull) {
                    Program *program = mglResolveProgramFromState(ctx);
                    id<MTLTexture> c0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
                    id<MTLTexture> d0 = mglRenderPassDepthTextureFor(_renderPassManager.state);
                    mglTraceLog("RT_SAMPLE_COPY_ENCODER hit=%llu fbo=%u rpFbo=%u program=%u rtTex=%u label=\"%s\" depthTex=%u depthLabel=\"%s\" "
                          "pass=%lux%lu c0=%p fmt=%lu depth=%p fmt=%lu "
                          "loadStore(c=%s/%s d=%s/%s) clipOrigin=0x%x "
                          "scissor(en=%d raw=%d,%d,%d,%d metal=%d,%d,%d,%d) "
                          "viewport(raw=%.1f,%.1f,%.1f,%.1f metal=%.1f,%.1f,%.1f,%.1f) "
                          "depth(test=%d write=%d func=0x%x) blend=%d cull=%d levels=%u mips=%u mipmapped=%u",
                          (unsigned long long)hit,
                          state->framebuffer ? (unsigned)state->framebuffer->name : 0u,
                          (unsigned)_renderPassManager.state->renderPassFramebufferName,
                          program ? (unsigned)program->name : (unsigned)state->program_name,
                          (unsigned)mglTraceTextureName(guiRTColor),
                          mglTraceTextureLabel(guiRTColor),
                          (unsigned)mglTraceTextureName(guiRTDepth),
                          mglTraceTextureLabel(guiRTDepth),
                          (unsigned long)passWidth,
                          (unsigned long)passHeight,
                          c0,
                          (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
                          d0,
                          (unsigned long)(d0 ? d0.pixelFormat : MTLPixelFormatInvalid),
                          mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLLoadActionDontCare)),
                          mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLStoreActionDontCare)),
                          mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLLoadActionDontCare)),
                          mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLStoreActionDontCare)),
                          state->var.clip_origin,
                          state->caps.scissor_test ? 1 : 0,
                          rawSx, rawSy, rawSw, rawSh,
                          sx, metalSy, sw, sh,
                          rawVx, rawVy, rawVw, rawVh,
                          vx, metalVy, vw, vh,
                          state->caps.depth_test ? 1 : 0,
                          state->var.depth_writemask ? 1 : 0,
                          (unsigned)state->var.depth_func,
                          state->caps.blend ? 1 : 0,
                          state->caps.cull_face ? 1 : 0,
                          guiRTColor ? (unsigned)guiRTColor->num_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmap_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmapped : 0u);
                }
            }

            BOOL viewportWasClamped = (vx != rawVx || vy != rawVy || vw != rawVw || vh != rawVh);
            BOOL viewportOriginConverted = (metalVy != vy);
            if (traceEncoderState) {
                mglTraceLogNSString(@"MGL VIEWPORT apply pass=%lux%lu origin=0x%x raw=(%.3f,%.3f,%.3f,%.3f) resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                              (unsigned long)passWidth, (unsigned long)passHeight,
                              state->var.clip_origin,
                              rawVx, rawVy, rawVw, rawVh,
                              vx, vy, vw, vh,
                              vx, metalVy, vw, vh);
            }

            if (kMGLDiagnosticStateLogs && (viewportWasClamped || viewportOriginConverted)) {
                static uint64_t s_viewportClampDetailCount = 0;
                uint64_t clampHit = ++s_viewportClampDetailCount;
                BOOL logClampDetail = (clampHit <= 80ull || (clampHit % 120ull) == 0ull);

                if (logClampDetail) {
                    Framebuffer *debugFbo = state->framebuffer;
                    BOOL debugFboValid = (debugFbo != NULL &&
                                          mglRendererObjectPointerLikelyValid(debugFbo) &&
                                          mglRendererPointerInHashTable(&state->framebuffer_table, debugFbo) &&
                                          mglPointerRangeIsReadable(debugFbo, sizeof(*debugFbo)));
                    id<MTLTexture> rpColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
                    id<MTLTexture> rpDepth = mglRenderPassDepthTextureFor(_renderPassManager.state);
                    id<MTLTexture> drawableTexture = (_drawable ? _drawable.texture : nil);

                    mglTraceLogNSString(@"MGL VIEWPORT CLAMP DETAIL hit=%llu fbo=%p valid=%d fboName=%u drawBuffer=0x%x pass=%lux%lu "
                                  "rpColor0=%p(%lux%lu) rpDepth=%p(%lux%lu) drawable=%p(%lux%lu) raw=(%.3f,%.3f,%.3f,%.3f) "
                                  "resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                                  (unsigned long long)clampHit,
                                  debugFbo,
                                  debugFboValid ? 1 : 0,
                                  (debugFboValid ? debugFbo->name : 0),
                                  state->draw_buffer,
                                  (unsigned long)passWidth,
                                  (unsigned long)passHeight,
                                  rpColor0,
                                  (unsigned long)(rpColor0 ? rpColor0.width : 0),
                                  (unsigned long)(rpColor0 ? rpColor0.height : 0),
                                  rpDepth,
                                  (unsigned long)(rpDepth ? rpDepth.width : 0),
                                  (unsigned long)(rpDepth ? rpDepth.height : 0),
                                  drawableTexture,
                                  (unsigned long)(drawableTexture ? drawableTexture.width : 0),
                                  (unsigned long)(drawableTexture ? drawableTexture.height : 0),
                                  rawVx, rawVy, rawVw, rawVh,
                                  vx, vy, vw, vh,
                                  vx, metalVy, vw, vh);

                    if (debugFboValid) {
                        for (int attIndex = 0; attIndex < MAX_COLOR_ATTACHMENTS; attIndex++) {
                            FBOAttachment *attachment = &debugFbo->color_attachments[attIndex];
                            if (attachment->texture == 0 && attachment->buf.tex == NULL && attachment->buf.rbo == NULL) {
                                continue;
                            }

                            Texture *attachmentTexture = NULL;
                            if (attachment->textarget == GL_RENDERBUFFER) {
                                attachmentTexture = attachment->buf.rbo ? attachment->buf.rbo->tex : NULL;
                            } else {
                                attachmentTexture = attachment->buf.tex;
                                if (!attachmentTexture && attachment->texture != 0) {
                                    attachmentTexture = findTexture(ctx, attachment->texture);
                                }
                            }

                            id<MTLTexture> attachmentMtl = (attachmentTexture && attachmentTexture->mtl_data)
                                ? (__bridge id<MTLTexture>)(attachmentTexture->mtl_data)
                                : nil;
                            id<MTLTexture> rpAttachment = mglRenderPassColorTextureFor(_renderPassManager.state, attIndex);

                            mglTraceLogNSString(@"MGL VIEWPORT CLAMP FBO att=%d name=%u textarget=0x%x level=%d layer=%d tex=%p "
                                          "texName=%u texTarget=0x%x texSize=%ux%ux%u mtl=%p(%lux%lu) rpTex=%p(%lux%lu)",
                                          attIndex,
                                          attachment->texture,
                                          attachment->textarget,
                                          attachment->level,
                                          attachment->layer,
                                          attachmentTexture,
                                          attachmentTexture ? attachmentTexture->name : 0,
                                          attachmentTexture ? attachmentTexture->target : 0,
                                          attachmentTexture ? attachmentTexture->width : 0,
                                          attachmentTexture ? attachmentTexture->height : 0,
                                          attachmentTexture ? attachmentTexture->depth : 0,
                                          attachmentMtl,
                                          (unsigned long)(attachmentMtl ? attachmentMtl.width : 0),
                                          (unsigned long)(attachmentMtl ? attachmentMtl.height : 0),
                                          rpAttachment,
                                          (unsigned long)(rpAttachment ? rpAttachment.width : 0),
                                          (unsigned long)(rpAttachment ? rpAttachment.height : 0));
                        }
                    }
                }
            }

            /* gl_ViewportIndex: when glViewportIndexedf* set any slot
             * beyond 0, bind the whole 16-entry viewport array (Metal
             * selects per vertex via viewport_array_index).  Slot 0 uses
             * the resolved/clamped rectangle computed above. */
            if (state->viewport_array_set) {
                double viewports[MGL_MAX_VIEWPORTS * 6];
                viewports[0] = vx;
                viewports[1] = metalVy;
                viewports[2] = vw;
                viewports[3] = vh;
                viewports[4] = state->var.depth_range[0];
                viewports[5] = state->var.depth_range[1];
                for (int vi = 1; vi < MGL_MAX_VIEWPORTS; vi++) {
                    GLdouble avx = state->viewport_array[vi][0];
                    GLdouble avy = state->viewport_array[vi][1];
                    GLdouble avw = state->viewport_array[vi][2];
                    GLdouble avh = state->viewport_array[vi][3];
                    GLdouble metalAvy = (GLdouble)passHeight - (avy + avh);
                    if (metalAvy < 0.0) metalAvy = 0.0;
                    viewports[vi * 6 + 0] = avx;
                    viewports[vi * 6 + 1] = metalAvy;
                    viewports[vi * 6 + 2] = avw;
                    viewports[vi * 6 + 3] = avh;
                    viewports[vi * 6 + 4] = state->var.depth_range[0];
                    viewports[vi * 6 + 5] = state->var.depth_range[1];
                }
                mglRenderCppBindingSetViewports(
                    _bindingStateOwner,
                    (__bridge void *)_renderPassManager.state->currentRenderEncoder,
                    viewports, (uint64_t)MGL_MAX_VIEWPORTS);
            } else {
                /* A shader may still write gl_ViewportIndex (e.g. gl_Layer
                 * alone binds viewport index to the same value per GL 4.6
                 * §11.1.3.5).  Keep every slot of the array valid even when
                 * glViewportIndexedf was never called: unset slots mirror
                 * slot 0 so such draws rasterize instead of being clipped. */
                double viewports[MGL_MAX_VIEWPORTS * 6];
                for (int vi = 0; vi < MGL_MAX_VIEWPORTS; vi++) {
                    viewports[vi * 6 + 0] = vx;
                    viewports[vi * 6 + 1] = metalVy;
                    viewports[vi * 6 + 2] = vw;
                    viewports[vi * 6 + 3] = vh;
                    viewports[vi * 6 + 4] = state->var.depth_range[0];
                    viewports[vi * 6 + 5] = state->var.depth_range[1];
                }
                mglRenderCppBindingSetViewports(
                    _bindingStateOwner,
                    (__bridge void *)_renderPassManager.state->currentRenderEncoder,
                    viewports, (uint64_t)MGL_MAX_VIEWPORTS);
            }
        } else {
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: updateCurrentRenderEncoder could not resolve pass size; using raw GL viewport");
            }
            [self setViewportIfNeeded:(MTLViewport){state->viewport[0], state->viewport[1],
                                       state->viewport[2], state->viewport[3],
                                       state->var.depth_range[0], state->var.depth_range[1]}];
        }
    }
}

- (bool) newRenderEncoder
{
    return [self newRenderEncoderWithReason:MGL_ENC_REASON_OTHER];
}

- (bool) newRenderEncoderWithReason:(MGLEncoderCreateReason)reason
{
    METAL_LOCK();
    bool result = [self newRenderEncoderLockedWithReason:reason];
    METAL_UNLOCK();
    return result;
}

/*
 * DontCare load-action inference for a color attachment.
 *
 * Returns YES only when it is provably safe to skip loading the attachment's
 * existing tile contents at pass start, i.e. the pass fully defines them:
 *   - env flag MGL_ENABLE_DONTCARE_LOAD is enabled (default OFF);
 *   - a real backing texture exists;
 *   - blending is disabled (a blend reads the destination, so contents live);
 *   - this is the texture's FIRST render-target use this frame (its stamped
 *     generation differs from the renderer's current frame generation) — a
 *     later pass to the same attachment must Load to preserve earlier draws.
 *
 * On a YES it stamps the texture with the current generation so any subsequent
 * pass this frame correctly falls back to Load. Callers invoke this only on the
 * no-pending-clear branch (a clear already fully defines contents via Clear).
 */
- (BOOL)shouldUseDontCareLoadForColorTexture:(Texture *)tex
                             firstUseThisFrame:(BOOL)firstUseThisFrame
{
    /* Pure decision — the caller is responsible for stamping the texture's
     * frame generation on EVERY render-target use (clear/load/dontcare), so
     * this predicate must not mutate state. DontCare is safe only on a frame's
     * first use of the attachment, with no pending clear (caller gates that),
     * blending off, and the flag enabled. */
    if (!mglEnvFlagEnabled("MGL_ENABLE_DONTCARE_LOAD")) {
        return NO;
    }
    if (!tex || !tex->mtl_data) {
        return NO;
    }
    if (ctx && MGL_STATE(ctx)->caps.blend) {
        return NO;
    }
    if (!firstUseThisFrame) {
        return NO;
    }
    return YES;
}

- (bool) configureUserFBOAttachmentsLocked
{
    Framebuffer *fbo;

    fbo = MGL_STATE(ctx)->framebuffer;

    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (int i = 0; i < drawBufferCount; i++)
    {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            (fbo->color_attachment_bitfield & (1u << attachmentIndex)) &&
            fbo->color_attachments[attachmentIndex].texture)
        {
            Texture *tex;

            tex = [self framebufferAttachmentTexture: &fbo->color_attachments[attachmentIndex]];
            if (!tex) {
                continue;
            }

            // Ensure attachment textures are created with RenderTarget usage.
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
            if (!tex->mtl_data) {
                continue;
            }

            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->color_attachments[attachmentIndex]);
            mglRenderPassSetPersistentAttachment(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                mglApplySRGBStateToRenderTarget(
                    (__bridge id<MTLTexture> _Nullable)(tex->mtl_data), ctx),
                subresource.level, subresource.slice,
                subresource.depthPlane);

            if (tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ||
                tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                tex->target == GL_TEXTURE_2D_ARRAY) {
                id<MTLTexture> rpTex = mglRenderPassColorTextureFor(_renderPassManager.state, colorSlot);
                (void)rpTex;
            }

            // Keep render pass dimensions aligned with attached color targets.
            // Some FBO paths use textures (not renderbuffers), and Metal still requires
            // scissor/viewport to be bounded by the attachment dimensions.
            NSUInteger attWidth = mglMetalTextureLevelDimension((NSUInteger)tex->width,
                                                                subresource.level);
            NSUInteger attHeight = mglMetalTextureLevelDimension((NSUInteger)tex->height,
                                                                 subresource.level);
            if (attWidth > 0 && attHeight > 0) {
                if (mglRenderPassRenderTargetWidthFor(_renderPassManager.state) == 0 || mglRenderPassRenderTargetHeightFor(_renderPassManager.state) == 0) {
                    mglRenderPassSetPersistentDimensions(
                        _renderPassManager.state, attWidth, attHeight);
                } else if (mglRenderPassRenderTargetWidthFor(_renderPassManager.state) != attWidth ||
                           mglRenderPassRenderTargetHeightFor(_renderPassManager.state) != attHeight) {
                    NSUInteger oldWidth = mglRenderPassRenderTargetWidthFor(_renderPassManager.state);
                    NSUInteger oldHeight = mglRenderPassRenderTargetHeightFor(_renderPassManager.state);
                    mglRenderPassSetPersistentDimensions(
                        _renderPassManager.state,
                        MIN(mglRenderPassRenderTargetWidthFor(_renderPassManager.state),
                            attWidth),
                        MIN(mglRenderPassRenderTargetHeightFor(_renderPassManager.state),
                            attHeight));
                    NSLog(@"MGL WARNING: FBO color attachment size mismatch slot=%d old=%lux%lu new=%lux%lu resolved=%lux%lu",
                          i,
                          (unsigned long)oldWidth,
                          (unsigned long)oldHeight,
                          (unsigned long)attWidth,
                          (unsigned long)attHeight,
                          (unsigned long)mglRenderPassRenderTargetWidthFor(_renderPassManager.state),
                          (unsigned long)mglRenderPassRenderTargetHeightFor(_renderPassManager.state));
                }
            }
        }
    }

    // depth attachment
    if (fbo->depth.texture)
    {
        Texture *tex;

        tex = [self framebufferAttachmentTexture: &fbo->depth];
        if (tex) {
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
        }
        if (tex && tex->mtl_data) {
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
            mglRenderPassSetPersistentAttachment(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                (__bridge id<MTLTexture> _Nullable)(tex->mtl_data),
                subresource.level, subresource.slice,
                subresource.depthPlane);
        }
    }

    // stencil attachment
    if (fbo->stencil.texture)
    {
        Texture *tex;

        tex = [self framebufferAttachmentTexture: &fbo->stencil];
        if (tex) {
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
        }
        if (tex && tex->mtl_data) {
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
            mglRenderPassSetPersistentAttachment(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                (__bridge id<MTLTexture> _Nullable)(tex->mtl_data),
                subresource.level, subresource.slice,
                subresource.depthPlane);
        }
    }
    return true;
}

- (bool) configureDefaultFramebufferAttachmentsLocked
{
    GLuint mgl_drawbuffer;
    id<MTLTexture> texture = nil;
    id<MTLTexture> depth_texture = nil;
    id<MTLTexture> stencil_texture = nil;

    switch(MGL_STATE(ctx)->draw_buffer)
    {
        case GL_FRONT: mgl_drawbuffer = _FRONT; break;
        case GL_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_FRONT_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_FRONT_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_BACK_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_BACK_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_FRONT_AND_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_COLOR_ATTACHMENT0: mgl_drawbuffer = _FRONT; break;
        case GL_NONE:
            // Handle GL_NONE gracefully - no draw buffer selected
            mgl_drawbuffer = _FRONT; // fallback to front
            DEBUG_PRINT("MGL: draw_buffer is GL_NONE, falling back to FRONT\n");
            break;
        default:
            DEBUG_PRINT("MGL: Unknown draw_buffer value: 0x%x, falling back to FRONT\n", MGL_STATE(ctx)->draw_buffer);
            mgl_drawbuffer = _FRONT; // fallback to front instead of failing render setup
            NSLog(@"MGL WARNING: Unknown draw_buffer value 0x%x, using FRONT fallback", MGL_STATE(ctx)->draw_buffer);
            break;
    }

    if(![self checkDrawBufferSize:mgl_drawbuffer])
    {
        _drawBuffers[mgl_drawbuffer].drawbuffer = NULL;
        _drawBuffers[mgl_drawbuffer].depthbuffer = NULL;
        _drawBuffers[mgl_drawbuffer].stencilbuffer = NULL;
        _drawBuffers[mgl_drawbuffer].width = 0;
        _drawBuffers[mgl_drawbuffer].height = 0;
    }

    // attach color buffer
    if (mgl_drawbuffer == _FRONT)
    {
        // SAFETY: Ensure we have a valid drawable with texture
        if (!_drawable) {
            NSLog(@"MGL ERROR: No drawable available for front buffer");
            return false;
        }

        texture = _drawable.texture;

        // sleep mode will return a null texture - handle gracefully without crashing
        if (!texture) {
            NSLog(@"MGL WARNING: Drawable texture is NULL (sleep mode or window not visible), attempting to get new drawable");

            // Try to get a new drawable
            _drawable = [_layer nextDrawable];
            if (_drawable) {
                texture = _drawable.texture;
                NSLog(@"MGL INFO: Successfully obtained new drawable with texture");
            } else {
                NSLog(@"MGL ERROR: Still no drawable texture available");
                return false;
            }
        }
    }
    else if(_drawBuffers[mgl_drawbuffer].drawbuffer)
    {
        texture = _drawBuffers[mgl_drawbuffer].drawbuffer;
    }
    else
    {
        texture = [self newDrawBuffer: ctx->pixel_format.mtl_pixel_format isDepthStencil:false];
        _drawBuffers[mgl_drawbuffer].drawbuffer = texture;
    }

    // attach depth. The default framebuffer must have a usable depth
    // attachment whenever GL depth testing is active, even if the legacy
    // context format fields were left unset by the window/bootstrap path.
    BOOL defaultPassNeedsDepth = MGL_STATE(ctx)->caps.depth_test ||
                                 _drawBuffers[mgl_drawbuffer].depthbuffer != nil;
    if (defaultPassNeedsDepth)
    {
        MTLPixelFormat depthFormat = ctx->depth_format.mtl_pixel_format;
        if (depthFormat == MTLPixelFormatInvalid) {
            depthFormat = MTLPixelFormatDepth32Float;
        }

        if(_drawBuffers[mgl_drawbuffer].depthbuffer)
        {
            depth_texture = _drawBuffers[mgl_drawbuffer].depthbuffer;
        }
        else
        {
            depth_texture = [self newDrawBufferWithCustomSize:depthFormat isDepthStencil:true customSize: CGSizeMake(texture.width, texture.height) ];
            _drawBuffers[mgl_drawbuffer].depthbuffer = depth_texture;
            if (depth_texture) {
                static uint64_t s_defaultDepthCreateCount = 0;
                uint64_t hit = ++s_defaultDepthCreateCount;
                if (kMGLDiagnosticStateLogs && hit <= 8) {
                    mglTraceLogNSString(@"MGL DEFAULT FBO: created depth attachment fmt=%lu size=%lux%lu drawBuffer=%u",
                                  (unsigned long)depthFormat,
                                  (unsigned long)depth_texture.width,
                                  (unsigned long)depth_texture.height,
                                  mgl_drawbuffer);
                }
            }
        }
    }

    // attach stencil
    BOOL defaultPassNeedsStencil = MGL_STATE(ctx)->caps.stencil_test ||
                                   ctx->stencil_format.format ||
                                   _drawBuffers[mgl_drawbuffer].stencilbuffer != nil;
    if (defaultPassNeedsStencil)
    {
        MTLPixelFormat stencilFormat = ctx->stencil_format.mtl_pixel_format;
        if (stencilFormat == MTLPixelFormatInvalid ||
            stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
            stencilFormat = MTLPixelFormatStencil8;
        }

        if(_drawBuffers[mgl_drawbuffer].stencilbuffer)
        {
            stencil_texture = _drawBuffers[mgl_drawbuffer].stencilbuffer;
        }
        else
        {
            stencil_texture = [self newDrawBufferWithCustomSize:stencilFormat isDepthStencil:true customSize: CGSizeMake(texture.width, texture.height) ];
            _drawBuffers[mgl_drawbuffer].stencilbuffer = stencil_texture;
        }
    }

    mglRenderPassSetPersistentAttachment(
        _renderPassManager.state,
        MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
        mglApplySRGBStateToRenderTarget(texture, ctx), 0, 0, 0);
    mglRenderPassSetPersistentAttachment(
        _renderPassManager.state,
        MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
        depth_texture, 0, 0, 0);
    mglRenderPassSetPersistentAttachment(
        _renderPassManager.state,
        MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
        stencil_texture, 0, 0, 0);

    mglRenderPassSetPersistentDimensions(
        _renderPassManager.state, texture.width, texture.height);
    _drawBuffers[mgl_drawbuffer].width = (GLuint)texture.width;
    _drawBuffers[mgl_drawbuffer].height = (GLuint)texture.height;
    return true;
}

- (void) ensureTransientDepthForDefaultFramebufferLocked
{
    if (!MGL_STATE(ctx)->framebuffer &&
        MGL_STATE(ctx)->caps.depth_test &&
        !mglRenderPassDepthTextureFor(_renderPassManager.state)) {
        NSUInteger depthWidth = mglRenderPassRenderTargetWidthFor(_renderPassManager.state);
        NSUInteger depthHeight = mglRenderPassRenderTargetHeightFor(_renderPassManager.state);

        if (depthWidth == 0 || depthHeight == 0) {
            id<MTLTexture> color0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
            if (color0) {
                depthWidth = color0.width;
                depthHeight = color0.height;
            }
        }

        if (depthWidth > 0 && depthHeight > 0) {
            if (!_renderPassManager.state->transientDepthTexture ||
                _renderPassManager.state->transientDepthTextureWidth != depthWidth ||
                _renderPassManager.state->transientDepthTextureHeight != depthHeight) {
                MTLTextureDescriptor *depthDesc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                       width:depthWidth
                                                                      height:depthHeight
                                                                   mipmapped:NO];
                depthDesc.usage = MTLTextureUsageRenderTarget;
                depthDesc.storageMode = MTLStorageModePrivate;
                id<MTLTexture> transientDepth =
                    mglRenderPassCreateTexture(_device, depthDesc);
                [_renderPassManager setTransientDepthTexture:transientDepth width:depthWidth height:depthHeight];

                if (_renderPassManager.state->transientDepthTexture) {
                    static uint64_t s_transientDepthCreateCount = 0;
                    uint64_t hit = ++s_transientDepthCreateCount;
                    if (hit <= 16 || (hit % 128) == 0) {
                        NSLog(@"MGL TRANSIENT FBO: created depth attachment fmt=%lu size=%lux%lu fbo=%u",
                              (unsigned long)MTLPixelFormatDepth32Float,
                              (unsigned long)depthWidth,
                              (unsigned long)depthHeight,
                              (unsigned)(mglRendererSafeFramebufferName(ctx)));
                    }
                } else {
                    NSLog(@"MGL ERROR: failed to create transient depth attachment size=%lux%lu fbo=%u",
                          (unsigned long)depthWidth,
                          (unsigned long)depthHeight,
                          (unsigned)(mglRendererSafeFramebufferName(ctx)));
                }
            }

            if (_renderPassManager.state->transientDepthTexture) {
                mglRenderPassSetPersistentAttachment(
                    _renderPassManager.state,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    _renderPassManager.state->transientDepthTexture,
                    0, 0, 0);
                mglRenderPassSetPersistentActions(
                    _renderPassManager.state,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    MTLLoadActionClear, MTLStoreActionDontCare);
                mglRenderPassSetPersistentDepthClear(
                    _renderPassManager.state,
                    MGL_STATE(ctx)->var.depth_clear_value);
            }
        }
    }
}

- (void) configureUserFBOLoadStoreActionsLocked:(GLuint *)outFboColorClearCount
                                  fboColorClearMask:(GLbitfield *)outFboColorClearMask
                     fboColorAttachment0ClearMask:(GLbitfield *)outFboColorAttachment0ClearMask
{
    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    /* Read the DontCare flag once per pass so the feature-off
     * (default) path skips both the per-attachment stamp write and the
     * shouldUse call entirely — avoiding per-attachment getenv and a
     * cache-line write on the common no-DontCare path. */
    BOOL dontCareLoadEnabled = mglEnvFlagEnabled("MGL_ENABLE_DONTCARE_LOAD");
    for (int i = 0; i < drawBufferCount; ++i) {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (!mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                   mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                   &attachmentIndex) ||
            attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            mglRenderPassSetPersistentLoadAction(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MTLLoadActionLoad);
            continue;
        }

        FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
        if (attachmentIndex == 0) {
            *outFboColorAttachment0ClearMask = att->clear_bitmask;
        }

        Texture *attachmentTextureForClear = [self framebufferAttachmentTexture:att];
        /* stamp this attachment's frame generation on EVERY
         * render-target use (clear/load/dontcare), capturing whether this
         * is its first use this frame BEFORE stamping. A clear-then-resume
         * within one frame must record the clear as a use so the resume is
         * not mistaken for a first use (which would wrongly DontCare and
         * discard the cleared+drawn content). */
        BOOL colorFirstUseThisFrame = NO;
        if (dontCareLoadEnabled && attachmentTextureForClear) {
            colorFirstUseThisFrame =
                (attachmentTextureForClear->mtl_rt_frame_generation != _renderPassManager.state->dontCareFrameGeneration);
            attachmentTextureForClear->mtl_rt_frame_generation = _renderPassManager.state->dontCareFrameGeneration;
        }
        if (att->clear_bitmask & GL_COLOR_BUFFER_BIT) {
            if (attachmentTextureForClear &&
                attachmentTextureForClear->name == 8u &&
                mglTraceLogIsEnabled()) {
                mglTraceLog("PENDING_COLOR_CLEAR_CONSUME tex=%u fbo=%u attachment=%u slot=%d program=%u clearMask=0x%x rgba=(%.3f,%.3f,%.3f,%.3f) drawBuf=0x%x readBuf=0x%x scissor(test=%d box=%d,%d,%d,%d) colorMask=%d%d%d%d depth(test=%d write=%d)",
                            (unsigned)attachmentTextureForClear->name,
                            (unsigned)fbo->name,
                            (unsigned)attachmentIndex,
                            i,
                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                            (unsigned)att->clear_bitmask,
                            att->clear_color[0],
                            att->clear_color[1],
                            att->clear_color[2],
                            att->clear_color[3],
                            (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
                            (unsigned)(ctx ? MGL_STATE(ctx)->read_buffer : 0u),
                            (ctx && MGL_STATE(ctx)->caps.scissor_test) ? 1 : 0,
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[0] : 0),
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[1] : 0),
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[2] : 0),
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[3] : 0),
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][0]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][1]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][2]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][3]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->caps.depth_test) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.depth_writemask) ? 1 : 0);
            }
            mglRenderPassSetPersistentColorClear(
                _renderPassManager.state, colorSlot,
                MTLClearColorMake(att->clear_color[0],
                                  att->clear_color[1],
                                  att->clear_color[2],
                                  att->clear_color[3]));
            mglRenderPassSetPersistentActions(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MTLLoadActionClear, MTLStoreActionStore);

            att->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
            mglMarkTextureLevelRenderTargetWritten(attachmentTextureForClear, att->level);

            (*outFboColorClearCount)++;
            *outFboColorClearMask |= (GLbitfield)(1u << attachmentIndex);
        } else if (dontCareLoadEnabled &&
                   [self shouldUseDontCareLoadForColorTexture:attachmentTextureForClear
                                                firstUseThisFrame:colorFirstUseThisFrame]) {
            /* first render-target use this frame, no clear, no
             * blend — prior tile contents are dead, skip the load. */
            mglRenderPassSetPersistentLoadAction(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MTLLoadActionDontCare);
        } else {
            mglRenderPassSetPersistentLoadAction(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MTLLoadActionLoad);
        }
    }

    /* Consume any pending color clears on attachments that could not be
     * resolved through a draw buffer — prevents infinite retry when an
     * attachment's clear_bitmask is set but mglMetalResolveFboDrawAttachmentIndex
     * fails for its draw buffer. */
    for (GLuint ai = 0; ai < MAX_COLOR_ATTACHMENTS; ++ai) {
        if ((fbo->color_attachments[ai].clear_bitmask & GL_COLOR_BUFFER_BIT) &&
            ((fbo->color_attachment_bitfield >> ai) & 1u) == 0u) {
            fbo->color_attachments[ai].clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        }
    }

    if (fbo->depth.clear_bitmask & GL_DEPTH_BUFFER_BIT) {
        mglRenderPassSetPersistentDepthClear(
            _renderPassManager.state, fbo->depth.clear_color[0]);
        mglRenderPassSetPersistentActions(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MTLLoadActionClear, MTLStoreActionStore);
        fbo->depth.clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MTLLoadActionLoad);
        if (mglRenderPassDepthTextureFor(_renderPassManager.state)) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MTLStoreActionStore);
        }
    }

    if (fbo->stencil.clear_bitmask & GL_STENCIL_BUFFER_BIT) {
        mglRenderPassSetPersistentStencilClear(
            _renderPassManager.state,
            (uint32_t)fbo->stencil.clear_color[0]);
        mglRenderPassSetPersistentActions(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MTLLoadActionClear, MTLStoreActionStore);
        fbo->stencil.clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MTLLoadActionLoad);
        if (mglRenderPassStencilTextureFor(_renderPassManager.state)) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MTLStoreActionStore);
        }
    }
}

- (void) configureDefaultFramebufferLoadStoreActionsLocked
{
    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLbitfield defaultClearMask = MGL_STATE(ctx)->default_fbo_clear_bitmask;
    if (defaultClearMask & GL_COLOR_BUFFER_BIT) {
        mglRenderPassSetPersistentColorClear(
            _renderPassManager.state, 0,
            MTLClearColorMake(MGL_STATE(ctx)->default_clear_color[0],
                              MGL_STATE(ctx)->default_clear_color[1],
                              MGL_STATE(ctx)->default_clear_color[2],
                              MGL_STATE(ctx)->default_clear_color[3]));
        mglRenderPassSetPersistentActions(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
            MTLLoadActionClear, MTLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
            MTLLoadActionLoad);
        static uint64_t s_defaultFboLoadLogCount = 0;
        uint64_t hit = ++s_defaultFboLoadLogCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            mglTraceLogNSString(@"MGL DEFAULT FBO: using Load (no clear mask) call=%llu drawBuf=0x%x fbo=%u",
                          (unsigned long long)hit,
                          MGL_STATE(ctx)->draw_buffer,
                          fbo ? (unsigned)fbo->name : 0u);
        }
    }

    if (defaultClearMask & GL_DEPTH_BUFFER_BIT) {
        mglRenderPassSetPersistentDepthClear(
            _renderPassManager.state,
            MGL_STATE(ctx)->var.depth_clear_value);
        mglRenderPassSetPersistentActions(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MTLLoadActionClear, MTLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MTLLoadActionLoad);
        if (mglRenderPassDepthTextureFor(_renderPassManager.state)) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MTLStoreActionStore);
        }
    }

    if (defaultClearMask & GL_STENCIL_BUFFER_BIT) {
        mglRenderPassSetPersistentStencilClear(
            _renderPassManager.state,
            MGL_STATE(ctx)->var.stencil_clear_value);
        mglRenderPassSetPersistentActions(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MTLLoadActionClear, MTLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager.state,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MTLLoadActionLoad);
        if (mglRenderPassStencilTextureFor(_renderPassManager.state)) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MTLStoreActionStore);
        }
    }
}

- (void) logRenderPassClearResolveLocked:(uint64_t)renderEncoderCall
                      traceRenderEncoder:(bool)traceRenderEncoder
                        fboColorClearCount:(GLuint)fboColorClearCount
                         fboColorClearMask:(GLbitfield)fboColorClearMask
            fboColorAttachment0ClearMask:(GLbitfield)fboColorAttachment0ClearMask
                 fboDepthClearMaskBefore:(GLbitfield)fboDepthClearMaskBefore
               fboStencilClearMaskBefore:(GLbitfield)fboStencilClearMaskBefore
                             defaultClearMask:(GLbitfield)defaultClearMask
                                         fbo:(Framebuffer *)fbo
{
	    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
	        MTLClearColor c0 = mglRenderPassClearColorFor(_renderPassManager.state, 0, MTLClearColorMake(0, 0, 0, 0));
	        mglTraceLogNSString(@"MGL TRACE clear.resolve call=%llu fbo=%u "
	              "fboColorClears=%u fboColorMask=0x%x fboAtt0ClearMask=0x%x c0LA=%s depthLA=%s stencilLA=%s "
	              "c0Clear=(%.3f,%.3f,%.3f,%.3f) depthClear=%.3f stencilClear=%u",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)fboColorClearCount,
              (unsigned)fboColorClearMask,
              (unsigned)fboColorAttachment0ClearMask,
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLLoadActionDontCare)),
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLLoadActionDontCare)),
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0, MTLLoadActionDontCare)),
              c0.red,
              c0.green,
              c0.blue,
              c0.alpha,
	              mglRenderPassClearDepthFor(_renderPassManager.state, 0.0),
	              (unsigned)(unsigned)mglRenderPassClearStencilFor(_renderPassManager.state, 0));
	    }

            BOOL clearResolveInteresting =
                (fboColorClearCount != 0) ||
                (fboColorAttachment0ClearMask != 0) ||
                (mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLLoadActionDontCare) == MTLLoadActionClear) ||
                (fboDepthClearMaskBefore & GL_DEPTH_BUFFER_BIT) ||
                (!fbo && (defaultClearMask & GL_DEPTH_BUFFER_BIT));
            if (clearResolveInteresting) {
                static uint64_t s_clearResolveDetailLogCount = 0;
                uint64_t hit = ++s_clearResolveDetailLogCount;
                if (mglTraceLogIsEnabled() && (hit <= 256ull || (hit % 512ull) == 0ull)) {
	            MTLClearColor c0 = mglRenderPassClearColorFor(_renderPassManager.state, 0, MTLClearColorMake(0, 0, 0, 0));
	            id<MTLTexture> c0Tex = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
	            id<MTLTexture> dTex = mglRenderPassDepthTextureFor(_renderPassManager.state);
	            id<MTLTexture> sTex = mglRenderPassStencilTextureFor(_renderPassManager.state);
		            mglTraceLog("RENDERPASS_CLEAR call=%llu hit=%llu fbo=%u drawBuf=0x%x readBuf=0x%x "
	                        "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                        "fboColorClears=%u fboColorMask=0x%x fboAtt0Mask=0x%x pending(global=0x%x default=0x%x depth=0x%x stencil=0x%x) "
	                        "c0LA=%s c0SA=%s depthLA=%s depthSA=%s stencilLA=%s stencilSA=%s "
	                        "c0Tex=%p fmt=%lu size=%lux%lu depthTex=%p fmt=%lu size=%lux%lu stencilTex=%p "
	                        "clearRGBA=(%.6f,%.6f,%.6f,%.6f) depthClear=%.6f stencilClear=%u depthState(test=%d write=%d func=0x%x)",
	                        (unsigned long long)renderEncoderCall,
	                        (unsigned long long)hit,
	                        (unsigned)(mglRendererSafeFramebufferName(ctx)),
	                        (unsigned)MGL_STATE(ctx)->draw_buffer,
	                        (unsigned)MGL_STATE(ctx)->read_buffer,
	                        (int)MGL_STATE(ctx)->viewport[0],
	                        (int)MGL_STATE(ctx)->viewport[1],
	                        (int)MGL_STATE(ctx)->viewport[2],
	                        (int)MGL_STATE(ctx)->viewport[3],
	                        MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
	                        (int)MGL_STATE(ctx)->var.scissor_box[0],
	                        (int)MGL_STATE(ctx)->var.scissor_box[1],
	                        (int)MGL_STATE(ctx)->var.scissor_box[2],
	                        (int)MGL_STATE(ctx)->var.scissor_box[3],
	                        (unsigned)fboColorClearCount,
	                        (unsigned)fboColorClearMask,
	                        (unsigned)fboColorAttachment0ClearMask,
	                        (unsigned)MGL_STATE(ctx)->clear_bitmask,
	                        (unsigned)MGL_STATE(ctx)->default_fbo_clear_bitmask,
	                        (unsigned)fboDepthClearMaskBefore,
	                        (unsigned)fboStencilClearMaskBefore,
	                        mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLStoreActionDontCare)),
	                        mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLStoreActionDontCare)),
	                        mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0, MTLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0, MTLStoreActionDontCare)),
	                        c0Tex,
	                        (unsigned long)(c0Tex ? c0Tex.pixelFormat : MTLPixelFormatInvalid),
	                        (unsigned long)(c0Tex ? c0Tex.width : 0),
	                        (unsigned long)(c0Tex ? c0Tex.height : 0),
	                        dTex,
	                        (unsigned long)(dTex ? dTex.pixelFormat : MTLPixelFormatInvalid),
	                        (unsigned long)(dTex ? dTex.width : 0),
	                        (unsigned long)(dTex ? dTex.height : 0),
	                        sTex,
	                        c0.red,
	                        c0.green,
	                        c0.blue,
	                        c0.alpha,
	                        mglRenderPassClearDepthFor(_renderPassManager.state, 0.0),
	                        (unsigned)(unsigned)mglRenderPassClearStencilFor(_renderPassManager.state, 0),
	                        MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
	                        MGL_STATE(ctx)->var.depth_writemask ? 1 : 0,
	                        (unsigned)MGL_STATE(ctx)->var.depth_func);
	        }
	    }
}

- (bool) finalizeRenderPassDescriptorLocked:(uint64_t)renderEncoderCall
                          traceRenderEncoder:(bool)traceRenderEncoder
{
    mglRenderPassSetPersistentStoreAction(
        _renderPassManager.state,
        MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
        MTLStoreActionStore);

    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
        id<MTLTexture> c0Tex = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
        id<MTLTexture> dTex = mglRenderPassDepthTextureFor(_renderPassManager.state);
        id<MTLTexture> sTex = mglRenderPassStencilTextureFor(_renderPassManager.state);
        mglTraceLogNSString(@"MGL TRACE renderpass.attach call=%llu fbo=%u drawBuf=0x%x rt=%lux%lu "
              "c0=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s stencil=%p fmt=%lu size=%lux%lu la/sa=%s/%s",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)MGL_STATE(ctx)->draw_buffer,
              (unsigned long)mglRenderPassRenderTargetWidthFor(_renderPassManager.state),
              (unsigned long)mglRenderPassRenderTargetHeightFor(_renderPassManager.state),
              c0Tex,
              (unsigned long)(c0Tex ? c0Tex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(c0Tex ? c0Tex.usage : 0),
              (unsigned long)(c0Tex ? c0Tex.width : 0),
              (unsigned long)(c0Tex ? c0Tex.height : 0),
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLLoadActionDontCare)),
              mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLStoreActionDontCare)),
              dTex,
              (unsigned long)(dTex ? dTex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(dTex ? dTex.width : 0),
              (unsigned long)(dTex ? dTex.height : 0),
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLLoadActionDontCare)),
              mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLStoreActionDontCare)),
              sTex,
              (unsigned long)(sTex ? sTex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(sTex ? sTex.width : 0),
              (unsigned long)(sTex ? sTex.height : 0),
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0, MTLLoadActionDontCare)),
              mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL, 0, MTLStoreActionDontCare)));
    }

    // create a render encoder from the renderpass descriptor
    // CRITICAL SAFETY: Validate inputs before creating render encoder
    // P4.1f: under gate-on the C++ RenderPassStateOwner is the validity
    // signal (the ObjC descriptor mirror is nil by design).
    BOOL hasRenderPassState = mglRenderPassUsesMetalCpp()
        ? (_renderPassManager.state->renderPassStateOwner != NULL)
        : (_renderPassManager.state->renderPassDescriptor != nil);
    if (!hasRenderPassState) {
        NSLog(@"MGL ERROR: Cannot create render encoder - render pass %s is NULL",
              mglRenderPassUsesMetalCpp() ? "state owner" : "descriptor");
        [self recordGPUError];
        return false;
    }

    // Metal debug layer crashes if render pass has no output attachment.
    // Provide a tiny fallback color attachment for targetless/invalid passes.
    bool hasOutputAttachment = false;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRenderPassColorTextureFor(_renderPassManager.state, i)) {
            hasOutputAttachment = true;
            break;
        }
    }
    if (!hasOutputAttachment &&
        (mglRenderPassDepthTextureFor(_renderPassManager.state) || mglRenderPassStencilTextureFor(_renderPassManager.state))) {
        hasOutputAttachment = true;
    }

    if (!hasOutputAttachment) {
        if (!_renderPassManager.state->fallbackRenderTargetTexture) {
            MTLTextureDescriptor *fbDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            fbDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            fbDesc.storageMode = MTLStorageModeShared;
            [_renderPassManager setFallbackRenderTargetTexture:
                mglRenderPassCreateTexture(_device, fbDesc)];
        }

        if (_renderPassManager.state->fallbackRenderTargetTexture) {
            NSLog(@"MGL WARNING: Render pass had no attachments; binding 1x1 fallback color target");
            mglRenderPassSetPersistentAttachment(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
                _renderPassManager.state->fallbackRenderTargetTexture,
                0, 0, 0);
            mglRenderPassSetPersistentActions(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MTLLoadActionLoad, MTLStoreActionStore);
            mglRenderPassSetPersistentDimensions(
                _renderPassManager.state, 1, 1);
        } else {
            NSLog(@"MGL ERROR: Failed to allocate fallback render target texture");
            [self recordGPUError];
            return false;
        }
    }

    // Final guard: Metal will assert if a color attachment texture is missing RenderTarget usage.
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> attTex = mglRenderPassColorTextureFor(_renderPassManager.state, i);
        if (attTex && ((attTex.usage & MTLTextureUsageRenderTarget) == 0)) {
            NSLog(@"MGL WARNING: colorAttachment[%d] usage=0x%lx lacks RenderTarget; clearing attachment to avoid Metal assert",
                  i, (unsigned long)attTex.usage);
            NSUInteger clearLevel = 0u, clearSlice = 0u, clearDepthPlane = 0u;
            mglRenderPassAttachmentSubresourceForState(
                _renderPassManager.state->renderPassDescriptor,
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i,
                &clearLevel, &clearSlice, &clearDepthPlane);
            mglRenderPassSetPersistentAttachment(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i,
                nil, clearLevel, clearSlice, clearDepthPlane);
        }
    }

    // Default-framebuffer paths expect color attachment 0 specifically.
    // FBO draw-buffer mappings may intentionally leave slot 0 as GL_NONE.
    if (!MGL_STATE(ctx)->framebuffer && !mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
        for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (mglRenderPassColorTextureFor(_renderPassManager.state, i)) {
                NSLog(@"MGL WARNING: colorAttachment[0] missing; remapping colorAttachment[%d] -> [0]", i);
                NSUInteger srcLevel = 0u, srcSlice = 0u, srcDepthPlane = 0u;
                mglRenderPassAttachmentSubresourceForState(
                    _renderPassManager.state->renderPassDescriptor,
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i,
                    &srcLevel, &srcSlice, &srcDepthPlane);
                mglRenderPassSetPersistentAttachment(
                    _renderPassManager.state,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    mglRenderPassColorTextureFor(_renderPassManager.state, i),
                    srcLevel, srcSlice, srcDepthPlane);
                mglRenderPassSetPersistentActions(
                    _renderPassManager.state,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    (MTLLoadAction)mglRenderPassLoadActionFor(
                        _renderPassManager.state,
                        MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i,
                        MTLLoadActionLoad),
                    (MTLStoreAction)mglRenderPassStoreActionFor(
                        _renderPassManager.state,
                        MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, i,
                        MTLStoreActionStore));
                break;
            }
        }
    }

    // Ultimate slot-0 fallback to keep draw path alive and avoid black frame.
    if (!hasOutputAttachment && !mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
        if (!_renderPassManager.state->fallbackRenderTargetTexture) {
            MTLTextureDescriptor *fbDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            fbDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            fbDesc.storageMode = MTLStorageModeShared;
            [_renderPassManager setFallbackRenderTargetTexture:
                mglRenderPassCreateTexture(_device, fbDesc)];
        }
        if (_renderPassManager.state->fallbackRenderTargetTexture) {
            NSLog(@"MGL WARNING: colorAttachment[0] unavailable; binding 1x1 fallback");
            mglRenderPassSetPersistentAttachment(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
                _renderPassManager.state->fallbackRenderTargetTexture,
                0, 0, 0);
            mglRenderPassSetPersistentActions(
                _renderPassManager.state,
                MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MTLLoadActionLoad, MTLStoreActionStore);
            if (mglRenderPassRenderTargetWidthFor(_renderPassManager.state) == 0 || mglRenderPassRenderTargetHeightFor(_renderPassManager.state) == 0) {
                mglRenderPassSetPersistentDimensions(
                    _renderPassManager.state, 1, 1);
            }
        } else {
            NSLog(@"MGL ERROR: Unable to allocate fallback colorAttachment[0] texture");
            [self recordGPUError];
            return false;
        }
    }

    // Ensure renderTargetWidth/Height are always coherent with the active attachments.
    {
        id<MTLTexture> sizeTex = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
        if (!sizeTex) {
            for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
                if (mglRenderPassColorTextureFor(_renderPassManager.state, i)) {
                    sizeTex = mglRenderPassColorTextureFor(_renderPassManager.state, i);
                    break;
                }
            }
        }
        if (!sizeTex) {
            sizeTex = mglRenderPassDepthTextureFor(_renderPassManager.state);
        }
        if (!sizeTex) {
            sizeTex = mglRenderPassStencilTextureFor(_renderPassManager.state);
        }

        if (sizeTex) {
            NSUInteger texWidth = sizeTex.width;
            NSUInteger texHeight = sizeTex.height;
            if (mglRenderPassRenderTargetWidthFor(_renderPassManager.state) == 0 ||
                mglRenderPassRenderTargetHeightFor(_renderPassManager.state) == 0 ||
                mglRenderPassRenderTargetWidthFor(_renderPassManager.state) > texWidth ||
                mglRenderPassRenderTargetHeightFor(_renderPassManager.state) > texHeight) {
                if (kMGLVerboseFrameLoopLogs) {
                    NSLog(@"MGL INFO: Normalizing renderTarget size from %lux%lu to %lux%lu",
                          (unsigned long)mglRenderPassRenderTargetWidthFor(_renderPassManager.state),
                          (unsigned long)mglRenderPassRenderTargetHeightFor(_renderPassManager.state),
                          (unsigned long)texWidth,
                          (unsigned long)texHeight);
                }
                mglRenderPassSetPersistentDimensions(
                    _renderPassManager.state, texWidth, texHeight);
            }
        }
    }
    return true;
}

- (bool) createRenderEncoderLocked:(uint64_t)renderEncoderCall
{
    // CRITICAL FIX: Validate command buffer state before creating render encoder
    if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
        NSLog(@"MGL ERROR: Cannot create render encoder - command buffer is NULL");
        [self recordGPUError];
        return false;
    }

    // Check if command buffer already has an active encoder (Metal API violation)
    if (_renderPassManager.state->currentRenderEncoder) {
        NSLog(@"MGL WARNING: Active render encoder detected - ending it before creating new one");
        [self endRenderEncodingLocked];
    }

    // Validate command buffer status. If already committed/completed, rotate to a new buffer.
    MTLCommandBufferStatus bufferStatus =
        mglRenderCommandBufferStatus(
            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
    if (bufferStatus >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL WARNING: Render encoder requested on finalized command buffer (status: %ld) - creating a fresh command buffer", (long)bufferStatus);
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer before creating render encoder");
            [self recordGPUError];
            return false;
        }

        if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
            NSLog(@"MGL ERROR: newCommandBuffer returned but (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner) is NULL");
            [self recordGPUError];
            return false;
        }

        bufferStatus = mglRenderCommandBufferStatus(
            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
        if (bufferStatus >= MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Fresh command buffer is still finalized (status: %ld)", (long)bufferStatus);
            [self recordGPUError];
            return false;
        }
    }

    if (kMGLVerboseFrameLoopLogs) {
        NSLog(@"MGL DEBUG: About to create render encoder with descriptor and command buffer");
    }
	    {
	        static uint64_t s_renderPassPreCreateLogCount = 0;
	        uint64_t hit = ++s_renderPassPreCreateLogCount;
		        if (mglTraceLogIsEnabled() && (hit <= 128ull || (hit % 512ull) == 0ull)) {
	            mglLogRenderPassLifecycle("pre-create",
	                                      hit,
                                      ctx,
                                      (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                      _renderPassManager.state->currentRenderEncoder,
                                      _renderPassManager.state->renderPassDescriptor,
                                      _drawable,
                                      _renderPassManager.state->renderPassFramebuffer,
	                                      _renderPassManager.state->renderPassFramebufferName,
	                                      _renderPassManager.state->renderPassDrawBuffer,
	                                      _renderPassManager.state->renderPassDrawBufferCount);
	            if (mglTraceLogIsEnabled()) {
	                id<MTLTexture> c0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
	                id<MTLTexture> depth = mglRenderPassDepthTextureFor(_renderPassManager.state);
	                mglTraceLog("RENDERPASS_PRE_CREATE hit=%llu call=%llu program=%u fbo=%u drawBuf=0x%x readBuf=0x%x "
	                            "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                            "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
	                            "depthState(test=%d write=%d func=0x%x) pending(default=0x%x depth=0x%x)",
	                            (unsigned long long)hit,
	                            (unsigned long long)renderEncoderCall,
	                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
	                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
	                            (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
	                            (unsigned)(ctx ? MGL_STATE(ctx)->read_buffer : 0u),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[3] : 0),
	                            (ctx && MGL_STATE(ctx)->caps.scissor_test) ? 1 : 0,
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[3] : 0),
	                            c0,
	                            (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(c0 ? c0.width : 0),
	                            (unsigned long)(c0 ? c0.height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLStoreActionDontCare)),
	                            depth,
	                            (unsigned long)(depth ? depth.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(depth ? depth.width : 0),
	                            (unsigned long)(depth ? depth.height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLStoreActionDontCare)),
	                            mglRenderPassClearDepthFor(_renderPassManager.state, 0.0),
	                            (ctx && MGL_STATE(ctx)->caps.depth_test) ? 1 : 0,
	                            (ctx && MGL_STATE(ctx)->var.depth_writemask) ? 1 : 0,
	                            (unsigned)(ctx ? MGL_STATE(ctx)->var.depth_func : 0u),
	                            (unsigned)(ctx ? MGL_STATE(ctx)->default_fbo_clear_bitmask : 0u),
	                            (unsigned)(ctx && MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->depth.clear_bitmask : 0u));
	            }
	        }
	    }
        /* When a GL sample query (GL_SAMPLES_PASSED / GL_ANY_SAMPLES_PASSED)
         * is active, attach the visibility result buffer to the render pass
         * descriptor and zero it so the GPU accumulates a fresh count. */
        void *queryVisibilityBuffer = NULL;
        if (_queryStateOwner &&
            mglRenderCppGetQueryVisibilityBuffer(
                _queryStateOwner, &queryVisibilityBuffer) == 0 &&
            queryVisibilityBuffer) {
            uint32_t visibilityResultType =
                mglRenderPassVisibilityResultTypeFor(_renderPassManager.state);
            mglRenderCppSetRenderPassStateVisibility(
                _renderPassManager.state->renderPassStateOwner,
                queryVisibilityBuffer, visibilityResultType);
            if (_renderPassManager.state->renderPassDescriptor) {
                _renderPassManager.state->renderPassDescriptor.visibilityResultBuffer =
                    (__bridge id<MTLBuffer>)queryVisibilityBuffer;
            }
        }
        @try {
            id<MTLRenderCommandEncoder> renderEncoder =
                [_renderPassManager createRenderEncoderWithDescriptor:
                    _renderPassManager.state->renderPassDescriptor];
            [_renderPassManager installRenderEncoder:renderEncoder];
            if (!_renderPassManager.state->currentRenderEncoder) {
            NSLog(@"MGL ERROR: Failed to create render encoder - invalid render pass descriptor or command buffer");
            NSLog(@"MGL DEBUG: Command buffer: %@, Render pass descriptor: %@", (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner), _renderPassManager.state->renderPassDescriptor);
            [self recordGPUError];
            return false;
        }
        /* Enable visibility result mode on the encoder for all draws in this
         * pass when a sample query is active. MTLVisibilityResultModeBoolean
         * writes 1 to the buffer if any samples pass per-fragment tests. */
        if (_queryStateOwner &&
            _renderPassManager.state->currentRenderEncoder) {
            uint32_t visibilityMode = 0;
            uint64_t visibilityOffset = 0;
            if (mglRenderCppAcquireSampleQuerySlot(
                    _queryStateOwner, &visibilityMode,
                    &visibilityOffset) == 0) {
                mglRenderCppSetVisibilityResultMode(
                    (__bridge void *)_renderPassManager.state->currentRenderEncoder,
                    visibilityMode, visibilityOffset);
            }
        }
        [_renderPassManager updateRenderPassIdentityForContext:ctx];
        /* When trace is disabled, skip the full-struct memset and trace
         * call and clear only the functional flag fields. */
        if (mglTraceLogIsEnabled()) {
            mglTraceFragmentTextureTraceBindings("CLEAR",
                                                 "new_render_encoder",
                                                 _resourceFallback.fragmentTextureTraceBindings,
                                                 TEXTURE_UNITS,
                                                 ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                 _pipelineCache.state->pipelineProgramName);
            memset(_resourceFallback.fragmentTextureTraceBindings, 0,
                   sizeof(_resourceFallback.fragmentTextureTraceBindings));
        } else {
            mglClearFragmentTextureTraceFunctionalFlags(
                _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
        }
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created Metal render encoder");
        }
        [self recordGPUSuccess];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating render encoder: %@ - continuing with degraded functionality", exception);
        NSLog(@"MGL DEBUG: Exception details - name: %@, reason: %@", exception.name, exception.reason);
        [self recordGPUError];
        [_renderPassManager clearCurrentRenderEncoder];
        return false;
    }
    _renderPassManager.state->currentRenderEncoder.label = @"GL Render Encoder";
	    {
	        static uint64_t s_renderPassCreatedLogCount = 0;
	        uint64_t hit = ++s_renderPassCreatedLogCount;
		        if (mglTraceLogIsEnabled() && (hit <= 128ull || (hit % 512ull) == 0ull)) {
	            mglLogRenderPassLifecycle("created",
	                                      hit,
                                      ctx,
                                      (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                      _renderPassManager.state->currentRenderEncoder,
                                      _renderPassManager.state->renderPassDescriptor,
                                      _drawable,
                                      _renderPassManager.state->renderPassFramebuffer,
	                                      _renderPassManager.state->renderPassFramebufferName,
	                                      _renderPassManager.state->renderPassDrawBuffer,
	                                      _renderPassManager.state->renderPassDrawBufferCount);
	            if (mglTraceLogIsEnabled()) {
	                id<MTLTexture> c0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
	                id<MTLTexture> depth = mglRenderPassDepthTextureFor(_renderPassManager.state);
	                mglTraceLog("RENDERPASS_CREATED hit=%llu call=%llu program=%u fbo=%u rpFbo=%u drawBuf=0x%x readBuf=0x%x "
	                            "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                            "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
	                            "depthState(test=%d write=%d func=0x%x)",
	                            (unsigned long long)hit,
	                            (unsigned long long)renderEncoderCall,
	                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
	                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
	                            (unsigned)_renderPassManager.state->renderPassFramebufferName,
	                            (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
	                            (unsigned)(ctx ? MGL_STATE(ctx)->read_buffer : 0u),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[3] : 0),
	                            (ctx && MGL_STATE(ctx)->caps.scissor_test) ? 1 : 0,
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[3] : 0),
	                            c0,
	                            (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(c0 ? c0.width : 0),
	                            (unsigned long)(c0 ? c0.height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0, MTLStoreActionDontCare)),
	                            depth,
	                            (unsigned long)(depth ? depth.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(depth ? depth.width : 0),
	                            (unsigned long)(depth ? depth.height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager.state, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0, MTLStoreActionDontCare)),
	                            mglRenderPassClearDepthFor(_renderPassManager.state, 0.0),
	                            (ctx && MGL_STATE(ctx)->caps.depth_test) ? 1 : 0,
	                            (ctx && MGL_STATE(ctx)->var.depth_writemask) ? 1 : 0,
	                            (unsigned)(ctx ? MGL_STATE(ctx)->var.depth_func : 0u));
	            }
	        }
	    }
    return true;
}

- (bool) newRenderEncoderLocked
{
    return [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_OTHER];
}

- (bool) newRenderEncoderLockedWithReason:(MGLEncoderCreateReason)reason
{
    /* instrumentation: count every render encoder (re)creation + reason. */
    MGL_PERF_INC(g_mglEncoderCreationsSinceSwap);
    if ((unsigned)reason >= (unsigned)MGL_ENC_REASON_COUNT) {
        reason = MGL_ENC_REASON_OTHER;
    }
    MGL_PERF_INC(g_mglEncoderCreateReasonSinceSwap[reason]);
    // I can't remember why this is here...
    @autoreleasepool {
    /* Invalidate last-bound render encoder state — the new encoder must
     * re-issue all binds rather than skipping them via the dedup fast path.
     * Called here (in addition to endRenderEncodingLocked) so the cache is
     * also cleared on code paths that bypass endRenderEncodingLocked. */
    [self invalidateLastBoundState];

    static uint64_t s_newRenderEncoderCallCount = 0;
    uint64_t renderEncoderCall = ++s_newRenderEncoderCallCount;
    bool traceRenderEncoder = mglShouldTraceCall(renderEncoderCall) ||
                              (kMGLDiagnosticStateLogs && ((renderEncoderCall % 60ull) == 0ull));

    // AGX ERROR THROTTLING: Check if we should skip render encoder creation
    // BUT allow limited render encoder creation for essential functionality
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: Render encoder creation requested during GPU recovery - attempting essential creation");
        // Continue with essential render encoder creation even during recovery
    }

    // CRITICAL SAFETY: Check command buffer before creating render encoder
    if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
        // Attempt recovery: create a new command buffer instead of failing immediately
        if ([self newCommandBufferLocked]) {
            // Successfully created - continue
        } else {
            NSLog(@"MGL ERROR: Cannot create render encoder - no command buffer available");
            [self recordGPUError];
            return false;
        }
    }

    // end encoding on current render encoder
    [self endRenderEncodingLocked];

    // grab the next drawable from CAMetalLayer
    if (_drawable == NULL)
    {
        if (!_layer) {
            NSLog(@"MGL ERROR: Cannot get drawable - no CAMetalLayer available");
            return false;
        }

        CGSize expectedDrawableSize = [self mglApplyPendingDrawableSize];
        _drawable = [_layer nextDrawable];

        // late init of gl scissor box on attachment to window system
        NSUInteger drawableWidth = (NSUInteger)MAX(1.0, expectedDrawableSize.width);
        NSUInteger drawableHeight = (NSUInteger)MAX(1.0, expectedDrawableSize.height);
        if (_drawable && _drawable.texture) {
            drawableWidth = (NSUInteger)_drawable.texture.width;
            drawableHeight = (NSUInteger)_drawable.texture.height;
        }

        if (!MGL_STATE(ctx)->caps.scissor_test) {
            MGL_STATE(ctx)->var.scissor_box[0] = 0;
            MGL_STATE(ctx)->var.scissor_box[1] = 0;
        }
        MGL_STATE(ctx)->var.scissor_box[2] = (GLint)drawableWidth;
        MGL_STATE(ctx)->var.scissor_box[3] = (GLint)drawableHeight;
    }

    [_renderPassManager installNewRenderPassDescriptor];
    if (mglRenderPassUsesMetalCpp()) {
        /* P4.1f: gate-on 不再创建 ObjC descriptor —— 校验 C++ owner。 */
        if (!_renderPassManager.state->renderPassStateOwner) {
            NSLog(@"MGL RENDERPASS ERROR: failed to allocate render pass state owner");
            return false;
        }
    } else if (!_renderPassManager.state->renderPassDescriptor) {
        NSLog(@"MGL RENDERPASS ERROR: failed to allocate render pass descriptor");
        return false;
    }

    // Configure color/depth/stencil attachments based on FBO type
    if (MGL_STATE(ctx)->framebuffer) {
        RETURN_FALSE_ON_FAILURE([self configureUserFBOAttachmentsLocked]);
    } else {
        RETURN_FALSE_ON_FAILURE([self configureDefaultFramebufferAttachmentsLocked]);
    }
    [self ensureTransientDepthForDefaultFramebufferLocked];

    // Capture clear state before load/store resolution for diagnostic logging
    GLuint fboColorClearCount = 0;
    GLbitfield fboColorClearMask = 0;
    GLbitfield fboColorAttachment0ClearMask = 0;

    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLbitfield defaultClearMask = MGL_STATE(ctx)->default_fbo_clear_bitmask;
    GLbitfield fboDepthClearMaskBefore = fbo ? fbo->depth.clear_bitmask : 0u;
    GLbitfield fboStencilClearMaskBefore = fbo ? fbo->stencil.clear_bitmask : 0u;

    if (fbo) {
        [self configureUserFBOLoadStoreActionsLocked:&fboColorClearCount
                                  fboColorClearMask:&fboColorClearMask
                     fboColorAttachment0ClearMask:&fboColorAttachment0ClearMask];
    } else {
        [self configureDefaultFramebufferLoadStoreActionsLocked];
    }

    [self logRenderPassClearResolveLocked:renderEncoderCall
                      traceRenderEncoder:traceRenderEncoder
                        fboColorClearCount:fboColorClearCount
                         fboColorClearMask:fboColorClearMask
            fboColorAttachment0ClearMask:fboColorAttachment0ClearMask
                 fboDepthClearMaskBefore:fboDepthClearMaskBefore
               fboStencilClearMaskBefore:fboStencilClearMaskBefore
                             defaultClearMask:defaultClearMask
                                         fbo:fbo];

    RETURN_FALSE_ON_FAILURE([self finalizeRenderPassDescriptorLocked:renderEncoderCall
                                                  traceRenderEncoder:traceRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self createRenderEncoderLocked:renderEncoderCall]);

    // apply all state that isn't included in a renderPassDescriptor into the render encoder
    [self updateCurrentRenderEncoder];

    // Only bind buffers when creating the encoder. Sampled textures depend on the
    // current GL program/MSL reflection and are rebound after the pipeline state is
    // selected for the draw.
    if (MGL_STATE(ctx)->vao)
    {
        MGLEncodeContext encCtx = { .encoder = _renderPassManager.state->currentRenderEncoder };
        if ([self bindVertexBuffersToCurrentRenderEncoder:&encCtx] == false)
        {
            DEBUG_PRINT("vertex buffer binding failed\n");
            [self recordGPUError];
            return false;
        }

        if ([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx] == false)
        {
            DEBUG_PRINT("fragment buffer binding failed\n");
            [self recordGPUError];
            return false;
        }
    }

    // Record successful render encoder creation (final success)
    [self recordGPUSuccess];
    return true;
        
    } //     @autoreleasepool
}

- (bool) newCommandBuffer
{
    METAL_LOCK();
    bool result = [self newCommandBufferLocked];
    METAL_UNLOCK();
    return result;
}

- (bool) newCommandBufferLocked
{
    // CRITICAL FIX: Proper encoder cleanup BEFORE creating new command buffer
    // Metal API requires ending encoders before creating new command buffers

    // STEP 0: End any existing render encoder to prevent MTLReleaseAssertionFailure
    if (_renderPassManager.state->currentRenderEncoder) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Ending existing render encoder before creating new command buffer");
        }
        [self endRenderEncodingLocked];
    }

    // STEP 1: Clean up sync tracking list safely.
    // IMPORTANT: Do NOT dereference Sync* entries here. Sync objects are owned by GL sync lifecycle
    // and may already be deleted by glDeleteSync on other paths.
    // Both this read/clear path and the append path (mtlGetSync) run on the GL
    // calling thread, so no lock is needed.
    [_renderPassManager clearCurrentCommandBufferSyncListEntries];

    // CRITICAL SAFETY: Validate command queue before creating buffer
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Cannot create command buffer - command queue is NULL");
        [_renderPassManager discardCurrentCommandBuffer];
        return false;
    }

    // STEP 1: Create fresh command buffer FIRST with comprehensive AGX driver validation
    @try {
        // AGX DRIVER COMPATIBILITY: Validate command queue health before creating buffer
        if (!_commandQueue) {
            NSLog(@"MGL AGX ERROR: Command queue is NULL - recreating");
            [self resetMetalState];
            if (!_commandQueue) {
                NSLog(@"MGL AGX CRITICAL: Cannot recreate command queue");
                return false;
            }
        }

        // CRITICAL FIX: Validate _commandQueue before dereferencing to prevent NULL pointer crashes
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: _commandQueue is NULL - cannot create command buffer");
            [self recordGPUError];
            return false;
        }

        // Additional validation: Ensure _commandQueue is a valid Metal object
        @try {
            // Test if _commandQueue is valid by checking its class
            Class queueClass = [_commandQueue class];
            if (!queueClass) {
                NSLog(@"MGL AGX CRITICAL: _commandQueue is invalid (no class) - recreating");
                _commandQueue = mglRenderPassCreateCommandQueue(
                    _device, &_commandQueueOwner);
                if (!_commandQueue) {
                    NSLog(@"MGL AGX CRITICAL: Failed to recreate command queue");
                    [self recordGPUError];
                    return false;
                }
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL AGX CRITICAL: _commandQueue validation exception: %@ - recreating", exception);
            [self recordGPUError];
            _commandQueue = mglRenderPassCreateCommandQueue(
                _device, &_commandQueueOwner);
            if (!_commandQueue) {
                NSLog(@"MGL AGX CRITICAL: Failed to recreate command queue after exception");
                [self recordGPUError];
                return false;
            }
        }

        if (![_renderPassManager installNewCommandBufferFromQueue:_commandQueue]) {
            NSLog(@"MGL AGX ERROR: Failed to create Metal command buffer - command queue may be in error state");
            [self recordGPUError];
            // Force command queue recreation
            [self resetMetalState];
            return false;
        }

        _currentCBHasWork = NO;

        // AGX Driver Validation: Check if the command buffer is immediately invalid
        MGLRenderCppCommandBufferState initialState =
            mglRenderCommandBufferState(
                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
        if (initialState.has_error) {
            NSLog(@"MGL AGX WARNING: New command buffer has immediate error: %@",
                  mglRenderCommandBufferErrorString(&initialState));
            [self recordGPUError];
            // Don't return false immediately - AGX sometimes creates error-state buffers that recover
        }

        // AGX DRIVER COMPATIBILITY: Enhanced validation to prevent rejections
        if (mglRenderCommandBufferStatus(
                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) ==
            MTLCommandBufferStatusError) {
            NSLog(@"MGL AGX CRITICAL: Command buffer immediately in error state");
            [self recordGPUError];
            [_renderPassManager discardCurrentCommandBuffer];
            [self resetMetalState]; // Force full reset
            return false;
        }

        // Additional AGX validation: Check for buffer properties that cause rejections
        initialState = mglRenderCommandBufferState(
            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
        if (initialState.has_error) {
            NSLog(@"MGL AGX WARNING: Command buffer has immediate error: %@",
                  mglRenderCommandBufferErrorString(&initialState));
            [self recordGPUError];
            [_renderPassManager discardCurrentCommandBuffer];
            [self resetMetalState];
            return false;
        }

        // Validate command queue health
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: Command queue became NULL");
            [self resetMetalState];
            return false;
        }

        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created new Metal command buffer (AGX validated)");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: Exception creating command buffer: %@", exception);
        [self recordGPUError];
        [_renderPassManager discardCurrentCommandBuffer];

        // AGX DRIVER COMPATIBILITY: Force reset on exception to clear driver state
        [self resetMetalState];
        return false;
    }

    // STEP 2: Now handle pending event waits on the FRESH command buffer.
    GLuint cachedSyncName = 0;
    id<MTLEvent> cachedEvent =
        [_renderPassManager detachPendingEventWithSyncName:&cachedSyncName];
    if (cachedEvent) {
        if (!cachedSyncName) {
            NSLog(@"MGL WARNING: dropping pending shared-event wait with no sync name");
            return true;
        }

        if (kMGLDisableSharedEventSync) {
            NSLog(@"MGL INFO: Shared event wait disabled (debug no-op), skipping wait encode event=%p syncName=%u",
                  cachedEvent, cachedSyncName);
            return true;
        }

        // SAFELY ENCODE: Event wait functionality on the new command buffer
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Encoding event wait on fresh command buffer");
        }

        // Validate event pointer looks like a valid object address
        uintptr_t eventPtr = (uintptr_t)cachedEvent;
        if (eventPtr == 0x10 || eventPtr == 0x30 || eventPtr == 0x1000) {
            NSLog(@"MGL CRITICAL ERROR: Known corrupted event pointer pattern detected: 0x%lx", eventPtr);
            NSLog(@"MGL CRITICAL ERROR: Skipping event wait to prevent crash");
            return false;
        }

        if (eventPtr < 0x1000 || (eventPtr & 0x7) != 0) {
            NSLog(@"MGL ERROR: Suspicious event pointer value: %p", cachedEvent);
            NSLog(@"MGL INFO: Skipping event wait for safety");
            return false;
        }

        // ADDITIONAL SAFETY: Validate command buffer is still valid before encoding
        if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
            NSLog(@"MGL ERROR: Command buffer became NULL before event wait encoding");
            return false;
        }

        @try {
            NSLog(@"MGL INFO: Encoding safe event wait: event=%p, syncName=%u, cmdbuf=%p", cachedEvent, cachedSyncName, (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));

            // Use conservative approach: only encode if everything looks perfect
            [(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner) encodeWaitForEvent:cachedEvent value:cachedSyncName];

            NSLog(@"MGL SUCCESS: Event wait encoded successfully on fresh command buffer");
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Event wait failed - %@: %@", exception.name, exception.reason);
            NSLog(@"MGL INFO: Continuing without event wait to maintain stability");
            // Continue without event wait - system remains stable
        }

    }

    return true;
}

- (bool)ensureWritableCommandBuffer:(const char *)reason
{
    METAL_LOCK();
    bool result = [self ensureWritableCommandBufferLocked:reason];
    METAL_UNLOCK();
    return result;
}

- (bool)ensureWritableCommandBufferLocked:(const char *)reason
{
    if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
        if (kMGLDiagnosticStateLogs) {
            mglTraceLogNSString(@"MGL INFO: %s requested with NULL command buffer, creating one", reason ? reason : "operation");
        }
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to create command buffer for %s", reason ? reason : "operation");
            return false;
        }
    }

    MTLCommandBufferStatus status =
        mglRenderCommandBufferStatus(
            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
    if (status >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL INFO: %s requested on finalized command buffer (status: %ld), rotating", reason ? reason : "operation", (long)status);
        [self endRenderEncodingLocked];
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer for %s", reason ? reason : "operation");
            return false;
        }

        if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner) ||
            mglRenderCommandBufferStatus(
                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) >=
                MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Unable to obtain writable command buffer for %s", reason ? reason : "operation");
            return false;
        }
    }

    return true;
}

- (bool) newCommandBufferAndRenderEncoder
{
    // AGGRESSIVE MEMORY SAFETY: Validate fundamental Metal objects before use
    if (!_device) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No device available");
        return false;
    }

    if (!_commandQueue) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No command queue available");
        return false;
    }

    // Validate device pointer lower bound only (high canonical addresses are valid on macOS)
    uintptr_t device_addr = (uintptr_t)_device;
    if (device_addr < 0x1000) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Invalid device pointer: 0x%lx", device_addr);
        return false;
    }

    @try {
        if ([self newCommandBuffer] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newCommandBuffer failed");
            return false;
        }

        if ([self newRenderEncoderWithReason:MGL_ENC_REASON_CMD] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newRenderEncoder failed");
            return false;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Metal operation failed: %@", exception);
        return false;
    }

    return true;
}

#pragma mark pipeline descriptor
-(MTLRenderPipelineDescriptor *)generatePipelineDescriptor
{
    if (!ctx) {
        NSLog(@"MGL PIPELINE DESC fail: context is NULL");
        return nil;
    }

    const BOOL nativeTES = _tessellation.nativeTESActive;
    const BOOL tessVertexCapture = _tessellation.tessVertexCaptureActive;
    const BOOL cullDistanceCapture =
        _tessellation.cullDistanceCaptureActive;
    const BOOL geometryExpansion = _geometry.expansionActive;
    const BOOL tessCompute = _tessellation.tessComputeActive;
    const int vertexStage = nativeTES ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    Program *vertexProgram = nativeTES
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *fragmentProgram = (tessVertexCapture || cullDistanceCapture)
        ? NULL : mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    GLuint renderProgramKey = mglCurrentRenderProgramKey(ctx);
    GLuint vertexProgramName = vertexProgram ? vertexProgram->name : 0u;
    GLuint fragmentProgramName = fragmentProgram ? fragmentProgram->name : 0u;
    BOOL rasterizerDiscard = tessVertexCapture || cullDistanceCapture ||
        MGL_STATE(ctx)->caps.rasterizer_discard ? YES : NO;

    if (!vertexProgram || (!fragmentProgram && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing stage program key=%u vs=%p fs=%p current=%u pipeline=%u",
              (unsigned)renderProgramKey,
              vertexProgram,
              fragmentProgram,
              (unsigned)MGL_STATE(ctx)->program_name,
              (unsigned)MGL_STATE(ctx)->var.program_pipeline_binding);
        return nil;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC begin key=%u vsProgram=%u fsProgram=%u",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName);
    }

    if ([self bindMTLProgram:vertexProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for VS program=%u",
              (unsigned)vertexProgramName);
        return nil;
    }
    if (fragmentProgram &&
        fragmentProgram != vertexProgram &&
        [self bindMTLProgram:fragmentProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for FS program=%u",
              (unsigned)fragmentProgramName);
        return nil;
    }

    Shader *vertex_shader = vertexProgram->shader_slots[vertexStage];
    Shader *fragment_shader = fragmentProgram ? fragmentProgram->shader_slots[_FRAGMENT_SHADER] : NULL;
    if (!vertex_shader || (!fragment_shader && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing shaders key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName,
              vertex_shader,
              fragment_shader);
        return nil;
    }

	    void *vertexFunctionPtr = geometryExpansion
            ? (__bridge void *)_geometry.passthroughFunction
            : tessCompute
            ? (__bridge void *)_tessellation.tessPassthroughFunction
            : cullDistanceCapture
            ? vertexProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function
            : tessVertexCapture
            ? vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function
            : vertexProgram->modules[vertexStage].mtl_function;

	    id<MTLFunction> vertexFunction = (__bridge id<MTLFunction>)vertexFunctionPtr;
	    id<MTLFunction> fragmentFunction = fragmentProgram
	        ? (__bridge id<MTLFunction>)fragmentProgram->modules[_FRAGMENT_SHADER].mtl_function
	        : nil;
    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC vs=%@ fs=%@",
              vertexFunction ? vertexFunction.name : @"(null)",
              fragmentFunction ? fragmentFunction.name : @"(null)");
    }
	    if (!vertexFunction || (!fragmentFunction && !rasterizerDiscard)) {
	        NSLog(@"MGL PIPELINE DESC fail: missing MTLFunction key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
	              (unsigned)renderProgramKey,
	              (unsigned)vertexProgramName,
	              (unsigned)fragmentProgramName,
	              vertexFunction,
	              fragmentFunction);
	        return nil;
	    }

	    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    if (!pipelineStateDescriptor) {
        NSLog(@"MGL PIPELINE DESC fail: descriptor allocation failed for key=%u",
              (unsigned)renderProgramKey);
        return nil;
    }
    pipelineStateDescriptor.label = @"GLSL Pipeline";
    pipelineStateDescriptor.vertexFunction = vertexFunction;
    pipelineStateDescriptor.fragmentFunction = fragmentFunction;
    /* Metal requires a concrete inputPrimitiveTopology when the vertex
     * function writes [[render_target_array_index]] / [[viewport_array_index]].
     * Only set it when the active VS or GS actually writes gl_Layer /
     * gl_ViewportIndex; leaving it unspecified lets Metal auto-detect for
     * ordinary pipelines (e.g. VS writing gl_PointSize with triangle
     * draws would otherwise be rejected). */
    {
        BOOL needsLayerTopology = NO;
        Shader *vsSlot = vertexProgram ? vertexProgram->shader_slots[_VERTEX_SHADER] : NULL;
        if (vsSlot && vsSlot->src &&
            (strstr(vsSlot->src, "gl_Layer") ||
             strstr(vsSlot->src, "gl_ViewportIndex"))) {
            needsLayerTopology = YES;
        }
        if (!needsLayerTopology && vertexProgram) {
            Shader *gsSlot = vertexProgram->shader_slots[_GEOMETRY_SHADER];
            if (gsSlot && gsSlot->src &&
                (strstr(gsSlot->src, "gl_Layer") ||
                 strstr(gsSlot->src, "gl_ViewportIndex"))) {
                needsLayerTopology = YES;
            }
        }
        if (needsLayerTopology) {
            switch (_lastDrawPrimitiveMode) {
                case GL_POINTS:
                    pipelineStateDescriptor.inputPrimitiveTopology =
                        MTLPrimitiveTopologyClassPoint;
                    break;
                case GL_LINES:
                case GL_LINE_STRIP:
                case GL_LINE_LOOP:
                case GL_LINES_ADJACENCY:
                case GL_LINE_STRIP_ADJACENCY:
                    pipelineStateDescriptor.inputPrimitiveTopology =
                        MTLPrimitiveTopologyClassLine;
                    break;
                default:
                    pipelineStateDescriptor.inputPrimitiveTopology =
                        MTLPrimitiveTopologyClassTriangle;
                    break;
            }
        }
    }
    if (nativeTES) {
        switch (vertexProgram->tess_gen_spacing) {
            case GL_FRACTIONAL_EVEN:
                pipelineStateDescriptor.tessellationPartitionMode =
                    MTLTessellationPartitionModeFractionalEven;
                break;
            case GL_FRACTIONAL_ODD:
                pipelineStateDescriptor.tessellationPartitionMode =
                    MTLTessellationPartitionModeFractionalOdd;
                break;
            default:
                pipelineStateDescriptor.tessellationPartitionMode =
                    MTLTessellationPartitionModeInteger;
                break;
        }
        pipelineStateDescriptor.maxTessellationFactor = 64u;
        pipelineStateDescriptor.tessellationFactorScaleEnabled = NO;
        pipelineStateDescriptor.tessellationFactorFormat =
            MTLTessellationFactorFormatHalf;
        /* Indexed native TES feeds the CPU gather stream (uint32) as
         * controlPointIndexBuffer; non-indexed uses None. */
        pipelineStateDescriptor.tessellationControlPointIndexType =
            _tessellation.tessIndexedDraw
                ? MTLTessellationControlPointIndexTypeUInt32
                : MTLTessellationControlPointIndexTypeNone;
        pipelineStateDescriptor.tessellationFactorStepFunction =
            MTLTessellationFactorStepFunctionPerPatch;
        pipelineStateDescriptor.tessellationOutputWindingOrder =
            vertexProgram->tess_gen_vertex_order == GL_CW
                ? MTLWindingClockwise : MTLWindingCounterClockwise;
    }
    /* GL_RASTERIZER_DISCARD: When rasterizer discard is active, the
     * fragment function is nil.  Metal requires rasterizationEnabled to
     * match the vertex function's return type:
     *   - vertex returns void  (no stage outputs)  -> rasterizationEnabled = NO
     *   - vertex returns struct (has stage outputs) -> rasterizationEnabled = YES
     * The AIR backend generates a void return type when the vertex shader has
     * no varying outputs (e.g. SSBO-only vertex shaders), and a struct
     * return type when it has varying outputs. */
    if (tessVertexCapture || cullDistanceCapture) {
        pipelineStateDescriptor.rasterizationEnabled = NO;
    } else if (rasterizerDiscard) {
        GLuint vsOutputCount = vertexProgram->shader_resources_list[vertexStage][_STAGE_OUTPUT_RES].count;
        /* Native AIR TES always returns its post-tessellation output record,
         * including the built-in position.  Built-ins are intentionally not
         * present in the reflected user stage-output list, so a zero count
         * does not mean the TES function returns void. */
        pipelineStateDescriptor.rasterizationEnabled =
            (nativeTES || vsOutputCount > 0) ? YES : NO;
    } else {
        pipelineStateDescriptor.rasterizationEnabled = YES;
    }

    if (MGL_STATE(ctx)->framebuffer) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;

        for (int i = 0; i < MGL_STATE(ctx)->max_color_attachments; i++) {
            if (fbo->color_attachments[i].texture) {
                Texture *tex = [self framebufferAttachmentTexture:&fbo->color_attachments[i]];
                if (tex && ![self bindMTLTexture:tex]) {
                    NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for color attachment %d tex=%u",
                          i, tex->name);
                    return nil;
                }
                if (tex && tex->mtl_data) {
                    pipelineStateDescriptor.colorAttachments[i].pixelFormat = mtlPixelFormatForGLTex(tex);
                } else {
                    pipelineStateDescriptor.colorAttachments[i].pixelFormat = MTLPixelFormatInvalid;
                }
            }

            if ((fbo->color_attachment_bitfield >> (i + 1)) == 0) {
                break;
            }
        }

        if (fbo->depth.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->depth];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for depth tex=%u", tex->name);
                return nil;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat depthFormat = mtlPixelFormatForGLTex(tex);
                if (depthFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid depth texture format, falling back to Depth32Float");
                    depthFormat = MTLPixelFormatDepth32Float;
                }
                pipelineStateDescriptor.depthAttachmentPixelFormat = depthFormat;
            } else {
                pipelineStateDescriptor.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
            }
        }

        if (fbo->stencil.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->stencil];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for stencil tex=%u", tex->name);
                return nil;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat stencilFormat = mtlPixelFormatForGLTex(tex);
                if (stencilFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid stencil texture format, falling back to Stencil8");
                    stencilFormat = MTLPixelFormatStencil8;
                }
                pipelineStateDescriptor.stencilAttachmentPixelFormat = stencilFormat;
            } else {
                pipelineStateDescriptor.stencilAttachmentPixelFormat = MTLPixelFormatInvalid;
            }
        }
    } else {
        MTLPixelFormat preferredColor0 = MTLPixelFormatInvalid;
        if (_renderPassManager.state && mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
            preferredColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0).pixelFormat;
        } else if (_drawable && _drawable.texture) {
            preferredColor0 = _drawable.texture.pixelFormat;
        } else {
            preferredColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = preferredColor0;

        if (ctx->depth_format.format) {
            MTLPixelFormat depthFormat = ctx->depth_format.mtl_pixel_format;
            if (depthFormat == MTLPixelFormatInvalid) {
                depthFormat = MTLPixelFormatDepth32Float;
            }
            pipelineStateDescriptor.depthAttachmentPixelFormat = depthFormat;
        }

        if (ctx->stencil_format.format) {
            MTLPixelFormat stencilFormat = ctx->stencil_format.mtl_pixel_format;
            if (stencilFormat == MTLPixelFormatInvalid ||
                stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
                stencilFormat = MTLPixelFormatStencil8;
            }
            pipelineStateDescriptor.stencilAttachmentPixelFormat = stencilFormat;
        }
    }

    if (_renderPassManager.state->renderPassDescriptor) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id<MTLTexture> rpColor = mglRenderPassColorTextureFor(_renderPassManager.state, i);
            if (rpColor) {
                pipelineStateDescriptor.colorAttachments[i].pixelFormat = rpColor.pixelFormat;
            }
        }

        id<MTLTexture> rpDepth = mglRenderPassDepthTextureFor(_renderPassManager.state);
        id<MTLTexture> rpStencil = mglRenderPassStencilTextureFor(_renderPassManager.state);
        pipelineStateDescriptor.depthAttachmentPixelFormat =
            rpDepth ? rpDepth.pixelFormat : MTLPixelFormatInvalid;
        pipelineStateDescriptor.stencilAttachmentPixelFormat =
            rpStencil ? rpStencil.pixelFormat : MTLPixelFormatInvalid;
    }

    BOOL color0IsIntentionallyDisabled =
        MGL_STATE(ctx)->framebuffer &&
        mglMetalDrawBufferAt(ctx, 0u) == GL_NONE;

    if (!color0IsIntentionallyDisabled &&
        (pipelineStateDescriptor.colorAttachments[0].pixelFormat == MTLPixelFormatInvalid ||
         pipelineStateDescriptor.colorAttachments[0].pixelFormat == 0)) {
        MTLPixelFormat fallbackColor0 = MTLPixelFormatInvalid;
        if (_renderPassManager.state && mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
            fallbackColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0).pixelFormat;
        } else if (_drawable && _drawable.texture) {
            fallbackColor0 = _drawable.texture.pixelFormat;
        } else {
            fallbackColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        if (fallbackColor0 == MTLPixelFormatInvalid || fallbackColor0 == 0) {
            fallbackColor0 = MTLPixelFormatBGRA8Unorm;
        }
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE DESC missing color pixel format, fallback pixelFormat=%lu",
                  (unsigned long)fallbackColor0);
        }
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = fallbackColor0;
    }

    /* Owner-first attachment readers (gate-on: C++ owner; gate-off: ObjC
     * descriptor mirror).  The old guard required the ObjC descriptor to
     * exist, which skipped MSAA alignment entirely under gate-on
     * (renderPassDescriptor == nil) and left rasterSampleCount = 1 in a
     * 4x pass -> draws wrote a single sample of four, so resolves came
     * back at 25% coverage. */
    NSUInteger resolvedSampleCount = 1;
    id<MTLTexture> rpColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
    id<MTLTexture> rpDepth = mglRenderPassDepthTextureFor(_renderPassManager.state);
    id<MTLTexture> rpStencil = mglRenderPassStencilTextureFor(_renderPassManager.state);
    if (rpColor0 && rpColor0.sampleCount > 0) {
        resolvedSampleCount = rpColor0.sampleCount;
    } else if (rpDepth && rpDepth.sampleCount > 0) {
        resolvedSampleCount = rpDepth.sampleCount;
    } else if (rpStencil && rpStencil.sampleCount > 0) {
        resolvedSampleCount = rpStencil.sampleCount;
    }
    if (resolvedSampleCount == 0) {
        resolvedSampleCount = 1;
    }
    /* MTLRenderPipelineDescriptor.rasterSampleCount defaults to 1, so the
     * previous "only set when 0" check never overrode it. Always align the
     * pipeline's sample count with the actual render-pass attachment so
     * multisample (2D_MULTISAMPLE / 2D_MULTISAMPLE_ARRAY) targets don't
     * silently mismatch and produce empty draws. */
    pipelineStateDescriptor.rasterSampleCount = resolvedSampleCount;
    if (pipelineStateDescriptor.rasterSampleCount == 0) {
        pipelineStateDescriptor.rasterSampleCount = 1;
    }
    mglNormalizePipelineDepthStencilFormats(pipelineStateDescriptor, "generate");
    mglEnableIndirectCommandBuffersForPipeline(pipelineStateDescriptor);

    /* GL_COLOR_LOGIC_OP: Metal on Apple Silicon does NOT expose
     * MTLLogicOperation or logicOpEnabled on MTLRenderPipelineColorAttachmentDescriptor
     * (verified against MacOSX SDK headers). Logic op would require fragment-shader
     * emulation (reading framebuffer, applying bitwise op), which is not implemented.
     * State tracking (logic_op_mode) is preserved for query correctness. */
    NSUInteger activeColorAttachmentCount = 0;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (pipelineStateDescriptor.colorAttachments[i].pixelFormat != MTLPixelFormatInvalid &&
            pipelineStateDescriptor.colorAttachments[i].pixelFormat != 0) {
            activeColorAttachmentCount++;
        }
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC colorAttachmentCount=%lu depthFormat=%lu stencilFormat=%lu sampleCount=%lu",
              (unsigned long)activeColorAttachmentCount,
              (unsigned long)pipelineStateDescriptor.depthAttachmentPixelFormat,
              (unsigned long)pipelineStateDescriptor.stencilAttachmentPixelFormat,
              (unsigned long)pipelineStateDescriptor.rasterSampleCount);
        NSLog(@"MGL PIPELINE DESC renderTarget[0]=%lu",
              (unsigned long)pipelineStateDescriptor.colorAttachments[0].pixelFormat);
    }

    return pipelineStateDescriptor;
}

/* P4.2 gate-on：与 generatePipelineDescriptor + bindBlendStateToPipelineState-
 * Descriptor 完全等价的 value-state 填充。不创建 MTLRenderPipelineDescriptor；
 * 所有 descriptor 状态（color/depth/stencil 格式、blend、顶点布局、tessellation、
 * rasterization、inputPrimitiveTopology、sample count）逐字段写入
 * MGLRenderCppPipelineDescriptorState。顶点 attribute 状态由
 * generateVertexDescriptorState 填充（GS/TES compute 展开时为空布局，与
 * 原路径的空白 MTLVertexDescriptor 等价）。函数选择与 side effects
 * （bindMTLProgram/bindMTLTexture）与 ObjC 版本一一对应。返回 NO 表示失败
 * （与 generatePipelineDescriptor 返回 nil 的路径一一对应）。 */
- (BOOL)generatePipelineDescriptorState:(MGLRenderCppPipelineDescriptorState *)state
                         vertexFunction:(id<MTLFunction> *)vertexFunctionOut
                       fragmentFunction:(id<MTLFunction> *)fragmentFunctionOut
{
    if (!ctx) {
        NSLog(@"MGL PIPELINE DESC fail: context is NULL");
        return NO;
    }
    if (!state || !vertexFunctionOut || !fragmentFunctionOut) {
        NSLog(@"MGL PIPELINE DESC fail: bad out args");
        return NO;
    }
    *vertexFunctionOut = nil;
    *fragmentFunctionOut = nil;

    const BOOL nativeTES = _tessellation.nativeTESActive;
    const BOOL tessVertexCapture = _tessellation.tessVertexCaptureActive;
    const BOOL cullDistanceCapture =
        _tessellation.cullDistanceCaptureActive;
    const BOOL geometryExpansion = _geometry.expansionActive;
    const BOOL tessCompute = _tessellation.tessComputeActive;
    const int vertexStage = nativeTES ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    Program *vertexProgram = nativeTES
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *fragmentProgram = (tessVertexCapture || cullDistanceCapture)
        ? NULL : mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    GLuint renderProgramKey = mglCurrentRenderProgramKey(ctx);
    GLuint vertexProgramName = vertexProgram ? vertexProgram->name : 0u;
    GLuint fragmentProgramName = fragmentProgram ? fragmentProgram->name : 0u;
    BOOL rasterizerDiscard = tessVertexCapture || cullDistanceCapture ||
        MGL_STATE(ctx)->caps.rasterizer_discard ? YES : NO;

    if (!vertexProgram || (!fragmentProgram && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing stage program key=%u vs=%p fs=%p current=%u pipeline=%u",
              (unsigned)renderProgramKey,
              vertexProgram,
              fragmentProgram,
              (unsigned)MGL_STATE(ctx)->program_name,
              (unsigned)MGL_STATE(ctx)->var.program_pipeline_binding);
        return NO;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC begin key=%u vsProgram=%u fsProgram=%u",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName);
    }

    if ([self bindMTLProgram:vertexProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for VS program=%u",
              (unsigned)vertexProgramName);
        return NO;
    }
    if (fragmentProgram &&
        fragmentProgram != vertexProgram &&
        [self bindMTLProgram:fragmentProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for FS program=%u",
              (unsigned)fragmentProgramName);
        return NO;
    }

    Shader *vertex_shader = vertexProgram->shader_slots[vertexStage];
    Shader *fragment_shader = fragmentProgram ? fragmentProgram->shader_slots[_FRAGMENT_SHADER] : NULL;
    if (!vertex_shader || (!fragment_shader && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing shaders key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName,
              vertex_shader,
              fragment_shader);
        return NO;
    }

    void *vertexFunctionPtr = geometryExpansion
        ? (__bridge void *)_geometry.passthroughFunction
        : tessCompute
        ? (__bridge void *)_tessellation.tessPassthroughFunction
        : cullDistanceCapture
        ? vertexProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function
        : tessVertexCapture
        ? vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function
        : vertexProgram->modules[vertexStage].mtl_function;
    id<MTLFunction> vertexFunction = (__bridge id<MTLFunction>)vertexFunctionPtr;

    id<MTLFunction> fragmentFunction = fragmentProgram
        ? (__bridge id<MTLFunction>)fragmentProgram->modules[_FRAGMENT_SHADER].mtl_function
        : nil;
    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC vs=%@ fs=%@",
              vertexFunction ? vertexFunction.name : @"(null)",
              fragmentFunction ? fragmentFunction.name : @"(null)");
    }
    if (!vertexFunction || (!fragmentFunction && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing MTLFunction key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName,
              vertexFunction,
              fragmentFunction);
        return NO;
    }

    memset(state, 0, sizeof(*state));
    state->vertex_program_instance =
        vertexProgram->pipeline_cache_instance_id;
    state->vertex_program_generation =
        vertexProgram->pipeline_cache_generation;
    state->fragment_program_instance = fragmentProgram
        ? fragmentProgram->pipeline_cache_instance_id : 0u;
    state->fragment_program_generation = fragmentProgram
        ? fragmentProgram->pipeline_cache_generation : 0u;
    state->color_count = MAX_COLOR_ATTACHMENTS;
    state->rasterization_enabled = rasterizerDiscard ? 0 : 1;
    /* maxTessellationFactor 默认 64 —— 与 ObjC descriptor 默认值一致
     * （非 TES pipeline 不设该字段，Metal 默认 64；C++ builder 对 0 跳过
     * 设置）。 */
    state->max_tessellation_factor = 64u;

    /* inputPrimitiveTopology：只有 VS/GS 实际写 gl_Layer / gl_ViewportIndex
     * 时才设置（Metal 要求非 unspecified 的 concrete topology；普通 pipeline
     * 保持 unspecified 让 Metal 自动检测）。 */
    {
        BOOL needsLayerTopology = NO;
        Shader *vsSlot = vertexProgram ? vertexProgram->shader_slots[_VERTEX_SHADER] : NULL;
        if (vsSlot && vsSlot->src &&
            (strstr(vsSlot->src, "gl_Layer") ||
             strstr(vsSlot->src, "gl_ViewportIndex"))) {
            needsLayerTopology = YES;
        }
        if (!needsLayerTopology && vertexProgram) {
            Shader *gsSlot = vertexProgram->shader_slots[_GEOMETRY_SHADER];
            if (gsSlot && gsSlot->src &&
                (strstr(gsSlot->src, "gl_Layer") ||
                 strstr(gsSlot->src, "gl_ViewportIndex"))) {
                needsLayerTopology = YES;
            }
        }
        if (needsLayerTopology) {
            switch (_lastDrawPrimitiveMode) {
                case GL_POINTS:
                    state->input_primitive_topology =
                        (uint32_t)MTLPrimitiveTopologyClassPoint;
                    break;
                case GL_LINES:
                case GL_LINE_STRIP:
                case GL_LINE_LOOP:
                case GL_LINES_ADJACENCY:
                case GL_LINE_STRIP_ADJACENCY:
                    state->input_primitive_topology =
                        (uint32_t)MTLPrimitiveTopologyClassLine;
                    break;
                default:
                    state->input_primitive_topology =
                        (uint32_t)MTLPrimitiveTopologyClassTriangle;
                    break;
            }
        }
    }

    if (nativeTES) {
        switch (vertexProgram->tess_gen_spacing) {
            case GL_FRACTIONAL_EVEN:
                state->tessellation_partition_mode =
                    (uint32_t)MTLTessellationPartitionModeFractionalEven;
                break;
            case GL_FRACTIONAL_ODD:
                state->tessellation_partition_mode =
                    (uint32_t)MTLTessellationPartitionModeFractionalOdd;
                break;
            default:
                state->tessellation_partition_mode =
                    (uint32_t)MTLTessellationPartitionModeInteger;
                break;
        }
        state->max_tessellation_factor = 64u;
        state->tessellation_factor_scale_enabled = 0;
        state->tessellation_factor_format =
            (uint32_t)MTLTessellationFactorFormatHalf;
        /* Indexed native TES 的 CPU gather 流（uint32）作为
         * controlPointIndexBuffer；非 indexed 用 None。 */
        state->tessellation_control_point_index_type =
            _tessellation.tessIndexedDraw
                ? (uint32_t)MTLTessellationControlPointIndexTypeUInt32
                : (uint32_t)MTLTessellationControlPointIndexTypeNone;
        state->tessellation_factor_step_function =
            (uint32_t)MTLTessellationFactorStepFunctionPerPatch;
        state->tessellation_output_winding_order =
            vertexProgram->tess_gen_vertex_order == GL_CW
                ? (uint32_t)MTLWindingClockwise
                : (uint32_t)MTLWindingCounterClockwise;
    }

    /* GL_RASTERIZER_DISCARD / capture 变体：rasterizationEnabled 与 vertex
     * function 的返回类型匹配（void -> NO，有 stage outputs -> YES）。AIR
     * TES 总是返回 post-tessellation record，故 native TES 强制 YES。 */
    if (tessVertexCapture || cullDistanceCapture) {
        state->rasterization_enabled = 0;
    } else if (rasterizerDiscard) {
        GLuint vsOutputCount = vertexProgram->shader_resources_list[vertexStage][_STAGE_OUTPUT_RES].count;
        state->rasterization_enabled =
            (nativeTES || vsOutputCount > 0) ? 1 : 0;
    } else {
        state->rasterization_enabled = 1;
    }

    /* 附件格式：FBO attachment -> rp/drawable/ctx 兜底（与
     * generatePipelineDescriptor 的 4526-4652 一致）。 */
    if (MGL_STATE(ctx)->framebuffer) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;

        for (int i = 0; i < MGL_STATE(ctx)->max_color_attachments; i++) {
            if (fbo->color_attachments[i].texture) {
                Texture *tex = [self framebufferAttachmentTexture:&fbo->color_attachments[i]];
                if (tex && ![self bindMTLTexture:tex]) {
                    NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for color attachment %d tex=%u",
                          i, tex->name);
                    return NO;
                }
                if (tex && tex->mtl_data) {
                    state->color_format[i] = (uint32_t)mtlPixelFormatForGLTex(tex);
                } else {
                    state->color_format[i] = (uint32_t)MTLPixelFormatInvalid;
                }
            }

            if ((fbo->color_attachment_bitfield >> (i + 1)) == 0) {
                break;
            }
        }

        if (fbo->depth.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->depth];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for depth tex=%u", tex->name);
                return NO;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat depthFormat = mtlPixelFormatForGLTex(tex);
                if (depthFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid depth texture format, falling back to Depth32Float");
                    depthFormat = MTLPixelFormatDepth32Float;
                }
                state->depth_format = (uint32_t)depthFormat;
            } else {
                state->depth_format = (uint32_t)MTLPixelFormatInvalid;
            }
        }

        if (fbo->stencil.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->stencil];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for stencil tex=%u", tex->name);
                return NO;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat stencilFormat = mtlPixelFormatForGLTex(tex);
                if (stencilFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid stencil texture format, falling back to Stencil8");
                    stencilFormat = MTLPixelFormatStencil8;
                }
                state->stencil_format = (uint32_t)stencilFormat;
            } else {
                state->stencil_format = (uint32_t)MTLPixelFormatInvalid;
            }
        }
    } else {
        MTLPixelFormat preferredColor0 = MTLPixelFormatInvalid;
        if (_renderPassManager.state && mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
            preferredColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0).pixelFormat;
        } else if (_drawable && _drawable.texture) {
            preferredColor0 = _drawable.texture.pixelFormat;
        } else {
            preferredColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        state->color_format[0] = (uint32_t)preferredColor0;

        if (ctx->depth_format.format) {
            MTLPixelFormat depthFormat = ctx->depth_format.mtl_pixel_format;
            if (depthFormat == MTLPixelFormatInvalid) {
                depthFormat = MTLPixelFormatDepth32Float;
            }
            state->depth_format = (uint32_t)depthFormat;
        }

        if (ctx->stencil_format.format) {
            MTLPixelFormat stencilFormat = ctx->stencil_format.mtl_pixel_format;
            if (stencilFormat == MTLPixelFormatInvalid ||
                stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
                stencilFormat = MTLPixelFormatStencil8;
            }
            state->stencil_format = (uint32_t)stencilFormat;
        }
    }

    /* P4.1f: gate-on 下 pass 状态在 C++ owner，descriptor 镜像为 nil ——
     * 用 owner 存在性作为「pass 已配置」的信号，否则 pipeline 会用
     * FBO/context 推导的格式构建，与真实 pass（如 sRGB 变体、transient
     * depth）不匹配，Metal 会静默丢弃所有 draw。 */
    BOOL hasConfiguredRenderPass = mglRenderPassUsesMetalCpp()
        ? (_renderPassManager.state->renderPassStateOwner != NULL)
        : (_renderPassManager.state->renderPassDescriptor != nil);
    if (hasConfiguredRenderPass) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id<MTLTexture> rpColor = mglRenderPassColorTextureFor(_renderPassManager.state, i);
            if (rpColor) {
                state->color_format[i] = (uint32_t)rpColor.pixelFormat;
            }
        }

        id<MTLTexture> rpDepth = mglRenderPassDepthTextureFor(_renderPassManager.state);
        id<MTLTexture> rpStencil = mglRenderPassStencilTextureFor(_renderPassManager.state);
        state->depth_format =
            rpDepth ? (uint32_t)rpDepth.pixelFormat : (uint32_t)MTLPixelFormatInvalid;
        state->stencil_format =
            rpStencil ? (uint32_t)rpStencil.pixelFormat : (uint32_t)MTLPixelFormatInvalid;
    }

    BOOL color0IsIntentionallyDisabled =
        MGL_STATE(ctx)->framebuffer &&
        mglMetalDrawBufferAt(ctx, 0u) == GL_NONE;

    if (!color0IsIntentionallyDisabled &&
        (state->color_format[0] == (uint32_t)MTLPixelFormatInvalid ||
         state->color_format[0] == 0u)) {
        MTLPixelFormat fallbackColor0 = MTLPixelFormatInvalid;
        if (_renderPassManager.state && mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
            fallbackColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0).pixelFormat;
        } else if (_drawable && _drawable.texture) {
            fallbackColor0 = _drawable.texture.pixelFormat;
        } else {
            fallbackColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        if (fallbackColor0 == MTLPixelFormatInvalid || fallbackColor0 == 0) {
            fallbackColor0 = MTLPixelFormatBGRA8Unorm;
        }
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE DESC missing color pixel format, fallback pixelFormat=%lu",
                  (unsigned long)fallbackColor0);
        }
        state->color_format[0] = (uint32_t)fallbackColor0;
    }

    /* rasterSampleCount：与 render pass attachment 的 sample count 对齐
     * （默认 1）。 */
    /* Owner-first attachment readers (gate-on: C++ owner; gate-off: ObjC
     * descriptor mirror).  The old guard required the ObjC descriptor to
     * exist, which skipped MSAA alignment entirely under gate-on
     * (renderPassDescriptor == nil) and left rasterSampleCount = 1 in a
     * 4x pass -> draws wrote a single sample of four, so resolves came
     * back at 25% coverage. */
    NSUInteger resolvedSampleCount = 1;
    id<MTLTexture> rpColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
    id<MTLTexture> rpDepth = mglRenderPassDepthTextureFor(_renderPassManager.state);
    id<MTLTexture> rpStencil = mglRenderPassStencilTextureFor(_renderPassManager.state);
    if (rpColor0 && rpColor0.sampleCount > 0) {
        resolvedSampleCount = rpColor0.sampleCount;
    } else if (rpDepth && rpDepth.sampleCount > 0) {
        resolvedSampleCount = rpDepth.sampleCount;
    } else if (rpStencil && rpStencil.sampleCount > 0) {
        resolvedSampleCount = rpStencil.sampleCount;
    }
    if (resolvedSampleCount == 0) {
        resolvedSampleCount = 1;
    }
    state->raster_sample_count = (uint32_t)resolvedSampleCount;

    /* 深度/模板 packed normalize（与 mglNormalizePipelineDepthStencilFormats
     * 一致；C++ builder 内还会再兜底一次）。签名必须在 normalize 之后计算，
     * 与 gate-off 的 descriptor 路径保持一致。 */
    {
        uint32_t depthFormat = state->depth_format;
        uint32_t stencilFormat = state->stencil_format;
        if (depthFormat != (uint32_t)MTLPixelFormatInvalid &&
            stencilFormat != (uint32_t)MTLPixelFormatInvalid &&
            depthFormat != stencilFormat) {
            bool depthPacked =
                depthFormat == (uint32_t)MTLPixelFormatDepth24Unorm_Stencil8 ||
                depthFormat == (uint32_t)MTLPixelFormatDepth32Float_Stencil8;
            bool stencilPacked =
                stencilFormat == (uint32_t)MTLPixelFormatDepth24Unorm_Stencil8 ||
                stencilFormat == (uint32_t)MTLPixelFormatDepth32Float_Stencil8;
            if (depthPacked || stencilPacked) {
                uint32_t packedFormat = stencilPacked ? stencilFormat : depthFormat;
                state->depth_format = packedFormat;
                state->stencil_format = packedFormat;
            }
        }
    }

    /* blend / alpha-to-coverage / alpha-to-one（镜像
     * bindBlendStateToPipelineStateDescriptor；blend 值 owner-first 读取）。 */
    state->alpha_to_coverage_enabled = MGL_STATE(ctx)->caps.sample_alpha_to_coverage ? 1 : 0;
    state->alpha_to_one_enabled = MGL_STATE(ctx)->caps.sample_alpha_to_one ? 1 : 0;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (state->color_format[i] == (uint32_t)MTLPixelFormatInvalid) {
            continue;
        }
        if (mglMetalDrawBufferAt(ctx, (GLuint)i) == GL_NONE) {
            state->color_write_mask[i] = 0u;
            continue;
        }
        MGLRenderCppPipelineBlendState blend = {0};
        if (![_pipelineCache blendStateForAttachment:(NSUInteger)i out:&blend]) {
            NSLog(@"MGL PIPELINE DESC fail: blend state unavailable for attachment %d", i);
            return NO;
        }
        state->color_write_mask[i] = blend.color_write_mask;
        if (MGL_STATE(ctx)->caps.blendi[i]) {
            state->blending_enabled_mask |= 1u << i;
        }
        state->source_rgb_blend_factor[i] = blend.source_rgb_factor;
        state->destination_rgb_blend_factor[i] = blend.destination_rgb_factor;
        state->source_alpha_blend_factor[i] = blend.source_alpha_factor;
        state->destination_alpha_blend_factor[i] = blend.destination_alpha_factor;
        state->rgb_blend_operation[i] = blend.rgb_operation;
        state->alpha_blend_operation[i] = blend.alpha_operation;
    }

    /* 顶点布局：GS/TES compute 展开用空布局（等价原路径的空白
     * MTLVertexDescriptor）；其余走 generateVertexDescriptorState。 */
    if (!(geometryExpansion || tessCompute)) {
        if (![self generateVertexDescriptorState:state]) {
            return NO;
        }
    }

    if (kMGLVerbosePipelineLogs) {
        uint32_t activeColorAttachmentCount = 0;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (state->color_format[i] != (uint32_t)MTLPixelFormatInvalid &&
                state->color_format[i] != 0u) {
                activeColorAttachmentCount++;
            }
        }
        NSLog(@"MGL PIPELINE DESC colorAttachmentCount=%u depthFormat=%u stencilFormat=%u sampleCount=%u",
              (unsigned)activeColorAttachmentCount,
              (unsigned)state->depth_format,
              (unsigned)state->stencil_format,
              (unsigned)state->raster_sample_count);
        NSLog(@"MGL PIPELINE DESC renderTarget[0]=%u",
              (unsigned)state->color_format[0]);
    }

    *vertexFunctionOut = vertexFunction;
    *fragmentFunctionOut = fragmentFunction;
    return YES;
}

#pragma mark vertex descriptor

- (void)updateGLSampledCopiesForEndedRenderPassFramebuffer:(Framebuffer *)fbo
                                                  drawCount:(GLsizei)drawCount
                                               drawBuffers:(const GLenum *)drawBuffers
                                                    reason:(const char *)reason
{
    (void)drawCount;
    (void)drawBuffers;

    if (!ctx || !fbo) {
        return;
    }

    /* Early-out: skip the per-attachment copy loop entirely when no texture
     * in this FBO is a sampled render target.  The old code unconditionally
     * iterated all color attachments on every endRenderPass and created a
     * Y-flipped copy for each (~313 copies/frame, most never sampled).  The
     * copy is only needed when the texture will be sampled by a non-yflip
     * shader in a subsequent draw, which we can't know here — but we CAN skip
     * textures that were never written (rtVer==0) or never flagged as RT.
     *
     * Iterate the actual FBO color attachments rather than the draw-buffer
     * snapshot.  MC 1.21.11's render abstraction creates transient FBOs such
     * as the GUI item atlas where the GL draw-buffer state can be incomplete
     * by the time the Metal encoder ends, but the attachment itself is still
     * the texture that was rendered and will be sampled immediately.
     *
     * NOTE: do NOT skip non-zero attachment levels here.  MC 1.21.11's
     * terrain atlas is a mipmapped RT whose mip 1-4 are written by separate
     * FBOs (one per mip level).  Skipping them left the Y-flip copy stale
     * after those passes ended, so terrain sampling mip>0 fell back to the
     * un-flipped Metal RT and rendered stripes.  The per-level blit inside
     * updateGLSampledRenderTargetCopyForTexture handles non-zero levels
     * correctly. */
    bool anySampledRT = false;
    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }
        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *tex = [self framebufferAttachmentTexture:attachment];
        if (tex && tex->mtl_data && tex->is_render_target &&
            tex->mtl_render_target_write_version != 0u) {
            anySampledRT = true;
            break;
        }
    }
    if (!anySampledRT) {
        return;
    }

    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];

        Texture *tex = [self framebufferAttachmentTexture:attachment];
        if (!tex || !tex->mtl_data) {
            continue;
        }

        id<MTLTexture> source = (__bridge id<MTLTexture>)(tex->mtl_data);
        if (![self textureCanUseGLSampledRenderTargetCopy:tex source:source]) {
            continue;
        }

        /* Y-Flip Subsystem: if the RT was rendered by a program whose VS had
         * Y-flip injection, the Metal texture already holds GL-bottom-origin
         * data.  No Y-flipped copy is needed — sampling consumers will use
         * the original via mglDecideYFlipForSampledRT.
         *
         * Defensive hardening: only release the copy when it is STALE
         * (version mismatch).  If a matching copy already exists, keep it —
         * a future sampler bind that defensively distrusts the authority
         * (e.g. after an IR binding remap or program-detection change) can
         * still use the copy instead of falling back to the un-flipped Metal
         * texture.  Releasing a matching copy saves a tiny amount of VRAM
         * but creates a fragile coupling: any change that makes the authority
         * wrong will immediately flip GUI items upside-down. */
        if (mglRTWriteAuthorityIsCurrentAndUsesOriginal(tex)) {
            if (tex->mtl_gl_sampled_data &&
                tex->mtl_gl_sampled_write_version != tex->mtl_render_target_write_version) {
                [self releaseGLSampledRenderTargetCopyForTexture:tex];
                if (mglTraceLogIsEnabled()) {
                    mglTraceLog("RT_SAMPLE_COPY_SKIP_INJECTED_RENDER tex=%u label=\"%s\" reason=render_yflip_injected_stale_released",
                                (unsigned)tex->name,
                                mglTraceTextureLabel(tex));
                }
            }
            continue;
        }

        [self updateGLSampledRenderTargetCopyForTexture:tex
                                                 source:source
                                                 reason:reason ? reason : "end_render_pass"];
    }
}

- (void) endRenderEncoding
{
    METAL_LOCK();
    [self endRenderEncodingLocked];
    METAL_UNLOCK();
}

- (void) endRenderEncodingLocked
{
    /* Invalidate last-bound render encoder state — the next encoder must
     * re-issue all binds rather than skipping them via the dedup fast path. */
    [self invalidateLastBoundState];

    if (_renderPassManager.state->currentRenderEncoder)
    {
        /* An active render encoder means work was encoded into the current
         * CB, so flushCommandBufferLocked: must not skip the commit. */
        _currentCBHasWork = YES;

        Framebuffer *endedFramebuffer = _renderPassManager.state->renderPassFramebuffer;
        GLsizei endedDrawBufferCount = _renderPassManager.state->renderPassDrawBufferCount;
        GLenum endedDrawBuffers[MAX_COLOR_ATTACHMENTS];
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            endedDrawBuffers[i] = _renderPassManager.state->renderPassDrawBuffers[i];
        }

        static uint64_t s_renderPassEndLogCount = 0;
        uint64_t hit = ++s_renderPassEndLogCount;
        if (hit <= 128ull || (hit % 1024ull) == 0ull) {
            mglLogRenderPassLifecycle("end",
                                      hit,
                                      ctx,
                                      (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                      _renderPassManager.state->currentRenderEncoder,
                                      _renderPassManager.state->renderPassDescriptor,
                                      _drawable,
                                      _renderPassManager.state->renderPassFramebuffer,
                                      _renderPassManager.state->renderPassFramebufferName,
                                      _renderPassManager.state->renderPassDrawBuffer,
                                      _renderPassManager.state->renderPassDrawBufferCount);
        }
        @try {
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Ending render encoder");
            }
            [_renderPassManager endCurrentRenderEncoder];
            [_renderPassManager clearCurrentRenderEncoder];
            /* When trace is disabled, skip the full-struct memset and
             * trace call and clear only the functional flag fields. */
            if (mglTraceLogIsEnabled()) {
                mglTraceFragmentTextureTraceBindings("CLEAR",
                                                     "end_render_encoding",
                                                     _resourceFallback.fragmentTextureTraceBindings,
                                                     TEXTURE_UNITS,
                                                     ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                     _pipelineCache.state->pipelineProgramName);
                memset(_resourceFallback.fragmentTextureTraceBindings, 0,
                       sizeof(_resourceFallback.fragmentTextureTraceBindings));
            } else {
                mglClearFragmentTextureTraceFunctionalFlags(
                    _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
            }
            [_renderPassManager clearRenderPassIdentity];
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Render encoder ended successfully");
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception ending render encoder: %@ - ignoring", exception.reason);
            // Force clear the encoder even if ending failed
            [_renderPassManager clearCurrentRenderEncoder];
            /* When trace is disabled, skip the full-struct memset and
             * trace call and clear only the functional flag fields. */
            if (mglTraceLogIsEnabled()) {
                mglTraceFragmentTextureTraceBindings("CLEAR",
                                                     "end_render_encoding_exception",
                                                     _resourceFallback.fragmentTextureTraceBindings,
                                                     TEXTURE_UNITS,
                                                     ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                     _pipelineCache.state->pipelineProgramName);
                memset(_resourceFallback.fragmentTextureTraceBindings, 0,
                       sizeof(_resourceFallback.fragmentTextureTraceBindings));
            } else {
                mglClearFragmentTextureTraceFunctionalFlags(
                    _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
            }
            [_renderPassManager clearRenderPassIdentity];
        }

        /* A later batch may sample this render target before the command
         * buffer is submitted, so refresh its GL-visible copy immediately. */
        if (endedFramebuffer) {
            [self updateGLSampledCopiesForEndedRenderPassFramebuffer:endedFramebuffer
                                                            drawCount:endedDrawBufferCount
                                                         drawBuffers:endedDrawBuffers
                                                              reason:"end_render_pass"];
        }
    }
}

- (BOOL)currentRenderPassUsesTexture:(id<MTLTexture>)texture
{
    if (!texture || !_renderPassManager.state->currentRenderEncoder) {
        return NO;
    }
    /* P4.1f: gate-on validity comes from the C++ owner. */
    if (mglRenderPassUsesMetalCpp()) {
        if (!_renderPassManager.state->renderPassStateOwner) {
            return NO;
        }
    } else if (!_renderPassManager.state->renderPassDescriptor) {
        return NO;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRenderPassColorTextureFor(_renderPassManager.state, i) == texture) {
            return YES;
        }
    }
    if (mglRenderPassDepthTextureFor(_renderPassManager.state) == texture ||
        mglRenderPassStencilTextureFor(_renderPassManager.state) == texture) {
        return YES;
    }

    return NO;
}

/*
 * synchronizeRenderPassForTextureReadback:reason: — heaviest synchronization boundary (GPU write visibility guarantee before CPU readback)
 *
 * Trigger condition: invoked when the texture to be read back is exactly the render target of the current render pass (color/depth/stencil attachment).
 * Guarantee semantics: endRenderEncoding closes the open render encoder → commitCommandBufferWithAGXRecovery:
 *           commits the current CB → waitUntilCompleted blocks until GPU completion → newCommandBuffer creates a new CB.
 *           Ensures that before CPU readback, all GPU rendering writes encoded to that texture have completed and are visible to the CPU.
 * Degradation: if the texture is not the current render target, returns YES directly (no sync needed); if the CB is already finalized, only rotates.
 */
- (BOOL)synchronizeRenderPassForTextureReadback:(id<MTLTexture>)texture
                                         reason:(const char *)reason
{
    BOOL usesTexture = [self currentRenderPassUsesTexture:texture];
    if (!usesTexture) {
        return YES;
    }

    [self endRenderEncoding];

    if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    if (mglRenderCommandBufferStatus(
            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) !=
        MTLCommandBufferStatusNotEnqueued) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    id<MTLCommandBuffer> commandBufferToCommit =
        [_renderPassManager detachCurrentCommandBufferForSubmission];

    @try {
        [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
        _lastCommittedCB = commandBufferToCommit;
        mglRenderPassWaitCommandBuffer(commandBufferToCommit);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: failed to synchronize render pass for texture readback (%s): %@",
              reason ? reason : "texture_readback",
              exception.reason);
        [self recordGPUError];
        [self newCommandBuffer];
        return NO;
    }

    MGLRenderCppCommandBufferState committedState =
        mglRenderCommandBufferState(commandBufferToCommit);
    if (committedState.has_error) {
        NSLog(@"MGL ERROR: render pass texture readback sync failed (%s): %@",
              reason ? reason : "texture_readback",
              mglRenderCommandBufferErrorString(&committedState));
        [self recordGPUError];
        [self newCommandBuffer];
        return NO;
    }

    return [self newCommandBuffer];
}

// ULTIMATE FAILSAFE: Emergency Metal state reset to recover from corruption
- (void) emergencyResetMetalState
{
    NSLog(@"MGL CRITICAL: Performing emergency Metal state reset");

    @try {
        // Force cleanup of all Metal objects
        [self endRenderEncodingLocked];

        [_renderPassManager discardCurrentCommandBuffer];
        [_renderPassManager clearCurrentRenderEncoder];
        _drawable = NULL;

        // Re-initialize basic Metal objects
        if (_device && _commandQueue) {
            NSLog(@"MGL CRITICAL: Re-creating Metal command buffer");
            [_renderPassManager installNewCommandBufferFromQueue:_commandQueue];

            if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
                NSLog(@"MGL CRITICAL: Failed to create new command buffer during recovery");
            }
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: Emergency Metal reset failed: %@", exception);
    }
}

- (bool) processGLState: (bool) draw_command
{
    METAL_LOCK();
    bool result = [self processGLStateLocked:draw_command];
    METAL_UNLOCK();
    return result;
}

- (bool) processGLStateLocked: (bool) draw_command
{
    static uint64_t s_processGLStateCallCount = 0;
    static double s_processGLStateLastCallTime = 0.0;
    static uint64_t s_processGLStateLastCallCount = 0;
    uint64_t processCall = ++s_processGLStateCallCount;
    double processStartSeconds = mglTraceNowSeconds();
    uint64_t processStartNS = mglTraceClockNS();
    bool traceProcess = mglShouldTraceCall(processCall);
    mglLogLoopHeartbeat("processGLState.loop",
                        processCall,
                        processStartSeconds,
                        &s_processGLStateLastCallTime,
                        &s_processGLStateLastCallCount,
                        0.25);
    if (traceProcess) {
        mglTraceLogNSString(@"MGL TRACE processGLState.begin call=%llu draw=%d",
              (unsigned long long)processCall, draw_command ? 1 : 0);
        mglLogStateSnapshot("processGLState.enter",
                            ctx,
                            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                            _renderPassManager.state->currentRenderEncoder,
                            _renderPassManager.state->renderPassDescriptor,
                            _drawable);
    }
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                _renderPassManager.state->currentRenderEncoder,
                                _renderPassManager.state->renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    if (draw_command) {
        /*
         * This flag is derived from the current draw's final fragment sampler
         * binding.  Clear it before any early render-state refresh so the
         * previous draw cannot disable culling while DIRTY_VAO/FBO is handled.
         */
        [_renderPassManager setCurrentDrawUsesRTSampledCopy:NO];
        MGL_FRAME_INC(g_mglProcessDrawCallsSinceSwap);
    }

    uintptr_t earlyCtxAddr = (uintptr_t)ctx;
    if (earlyCtxAddr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", earlyCtxAddr);
        return false;
    }

    // REMOVED: Thread synchronization was causing deadlocks
    // The issue is not thread contention but Metal object corruption

    // ULTIMATE FAILSAFE: Metal state corruption detection and recovery
    static int corruption_recovery_count = 0;
    static int max_recovery_attempts = 3;

    // Check for corrupted Metal objects that might cause crashes.
    // Only reject NULL / obviously invalid low addresses.
    if (!_device || !_commandQueue || ((uintptr_t)_device < 0x1000) || ((uintptr_t)_commandQueue < 0x1000)) {
        NSLog(@"MGL CRITICAL: Metal state corruption detected in processGLState!");
        NSLog(@"MGL CRITICAL: device=0x%lx, queue=0x%lx", (uintptr_t)_device, (uintptr_t)_commandQueue);

        if (corruption_recovery_count < max_recovery_attempts) {
            NSLog(@"MGL CRITICAL: Attempting Metal state recovery (%d/%d)", corruption_recovery_count + 1, max_recovery_attempts);

            // Force a complete Metal state reset
            @try {
                [self emergencyResetMetalState];
                corruption_recovery_count++;

                // Re-check after recovery
                if (!_device || !_commandQueue) {
                    NSLog(@"MGL CRITICAL: Metal recovery failed, aborting operation");
                    return false;
                }
            } @catch (NSException *exception) {
                NSLog(@"MGL CRITICAL: Metal recovery failed: %@", exception);
                return false;
            }
        } else {
            NSLog(@"MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations");
            return false;
        }
    }

    //logDirtyBits(ctx);

    if (!draw_command) {
        [self endRenderPassIfFramebufferChangedForNonDraw:processCall];
    }

    // since a clear is embedded into a render encoder
    if (MGL_STATE(ctx)->vao == NULL)
    {
        if (draw_command)
        {
            NSLog(@"Error: No VAO defined for ctx\n");

            // quietly return if we are not in a draw command with no vao defined
            // like a clear or init call
            return false;
        }

        // for a clear flush sequence...
        if (MGL_STATE(ctx)->dirty_bits & DIRTY_STATE)
        {
            // end encoding on current render encoder
            [self endRenderEncodingLocked];

            // Use GPU throttling to prevent crashes when creating new render encoder
            if (![self validateMetalObjects]) {
                NSLog(@"MGL WARNING: GPU throttling active - deferring render encoder creation");
                MGL_STATE(ctx)->dirty_bits &= ~DIRTY_STATE;
                return true;
            }

            @try {
                [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_CLEAR];
            } @catch (NSException *exception) {
                NSLog(@"MGL ERROR: Render encoder creation failed: %@", exception);
            }

            // Clear the dirty bit to prevent repeated attempts
            MGL_STATE(ctx)->dirty_bits &= ~DIRTY_STATE;
        }

        return true;
    }

    // only draw commands need a functioning render encoder
    // this can mess up a transition between compute and rendering on a flush
    // so just return
    // we may have to create a blank render encoder to safely run compute and
    // rendering correctly
    if (draw_command == false)
    {
        return true;
    }

    // MEMORY SAFETY: Validate context before use
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                _renderPassManager.state->currentRenderEncoder,
                                _renderPassManager.state->renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    // Validate context pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t ctx_addr = (uintptr_t)ctx;
    if (ctx_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", ctx_addr);
        return false;
    }

    // Early circuit-breaker: if a program is currently quarantined due to repeated
    // vertex/fragment interface mismatch, skip draw before creating/rotating buffers.
    GLuint blockedProgramKey = mglCurrentRenderProgramKey(ctx);
    if (blockedProgramKey != 0u &&
        _gpuRecovery.interfaceMismatchBlockedProgram != 0 &&
        blockedProgramKey == _gpuRecovery.interfaceMismatchBlockedProgram)
    {
        CFTimeInterval now = CFAbsoluteTimeGetCurrent();
        if (now < _gpuRecovery.interfaceMismatchBlockedUntil) {
            static uint64_t s_quarantineSkipCount = 0;
            s_quarantineSkipCount++;
            if (s_quarantineSkipCount <= 16 || (s_quarantineSkipCount % 1000) == 0) {
                double remaining = _gpuRecovery.interfaceMismatchBlockedUntil - now;
                if (remaining < 0.0) remaining = 0.0;
                NSLog(@"MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw",
                      (unsigned)_gpuRecovery.interfaceMismatchBlockedProgram, remaining);
            }
            return false;
        }
    }

    // Keep command buffer lifecycle healthy: if the active one is already finalized,
    // rotate to a fresh buffer before any state processing.
    if ((__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner) && _renderPassManager.state->currentRenderEncoder == NULL) {
        MTLCommandBufferStatus preStatus =
            mglRenderCommandBufferStatus(
                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
        if (preStatus >= MTLCommandBufferStatusCommitted) {
            static uint64_t s_rotateFinalizedCount = 0;
            uint64_t rotateHit = ++s_rotateFinalizedCount;
            if (rotateHit <= 16ull || (rotateHit % 500ull) == 0ull) {
                NSLog(@"MGL INFO: processGLState rotating finalized command buffer (status: %ld) hit=%llu",
                      (long)preStatus, (unsigned long long)rotateHit);
            }
            if (![self newCommandBufferLocked]) {
                NSLog(@"MGL ERROR: processGLState failed to create a fresh command buffer");
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.new_cb_rotate",
                                        ctx,
                                        (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                        _renderPassManager.state->currentRenderEncoder,
                                        _renderPassManager.state->renderPassDescriptor,
                                        _drawable);
                }
                return false;
            }
        }
    } else if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: processGLState found NULL command buffer, creating one");
        }
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: processGLState could not create initial command buffer");
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.new_cb_initial",
                                    ctx,
                                    (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                    _renderPassManager.state->currentRenderEncoder,
                                    _renderPassManager.state->renderPassDescriptor,
                                    _drawable);
            }
            return false;
        }
    }

    MGLResourceSyncWork resourceSyncWork = {false, false, false};
    RETURN_FALSE_ON_FAILURE([self processDirtyStateDomainsLocked:draw_command
                                                            work:&resourceSyncWork]);

    // Ensure a render encoder exists for draw commands.
    if (!_renderPassManager.state->currentRenderEncoder) {
        static uint64_t s_nilEncoderRecoveryCount = 0;
        uint64_t nilHit = ++s_nilEncoderRecoveryCount;
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            NSLog(@"MGL WARNING: processGLState - current render encoder is nil, attempting recovery hit=%llu",
                  (unsigned long long)nilHit);
            mglLogRenderPassLifecycle("nil-encoder-before-recovery",
                                      nilHit,
                                      ctx,
                                      (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                      _renderPassManager.state->currentRenderEncoder,
                                      _renderPassManager.state->renderPassDescriptor,
                                      _drawable,
                                      _renderPassManager.state->renderPassFramebuffer,
                                      _renderPassManager.state->renderPassFramebufferName,
                                      _renderPassManager.state->renderPassDrawBuffer,
                                      _renderPassManager.state->renderPassDrawBufferCount);
        }
        RETURN_FALSE_ON_FAILURE(
            [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_NIL]);
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            mglLogRenderPassLifecycle("nil-encoder-after-recovery",
                                      nilHit,
                                      ctx,
                                      (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                      _renderPassManager.state->currentRenderEncoder,
                                      _renderPassManager.state->renderPassDescriptor,
                                      _drawable,
                                      _renderPassManager.state->renderPassFramebuffer,
                                      _renderPassManager.state->renderPassFramebufferName,
                                      _renderPassManager.state->renderPassDrawBuffer,
                                      _renderPassManager.state->renderPassDrawBufferCount);
        }
    }

    if (draw_command) {
        RETURN_FALSE_ON_FAILURE([self ensureCurrentRenderPassMatchesFramebufferForDraw]);
        [self updateCurrentRenderEncoder];
    }

    if (draw_command && kMGLVerbosePipelineLogs) {
        static uint64_t s_drawPipelineLookupCount = 0;
        s_drawPipelineLookupCount++;
        if (s_drawPipelineLookupCount <= 256ull || (s_drawPipelineLookupCount % 1000ull) == 0ull) {
            Program *lookupProgram = mglResolveProgramFromState(ctx);
            Program *lookupVertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *lookupFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            GLuint lookupProgramName = mglCurrentRenderProgramKey(ctx);
            Framebuffer *lookupFBO = MGL_STATE(ctx)->framebuffer;
            GLuint lookupFBOName = lookupFBO ? lookupFBO->name : 0;
            fprintf(stderr, "MGL Draw current program key=%u mono=%p vs=%u fs=%u\n",
                    (unsigned)lookupProgramName,
                    (void *)lookupProgram,
                    lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
                    lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u);
            NSLog(@"MGL DRAW pipeline lookup result=%p key=%u vs=%u fs=%u vao=%p fbo=%u",
                  _pipelineCache.state->pipelineState,
                  (unsigned)lookupProgramName,
                  lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
                  lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u,
                  MGL_STATE(ctx)->vao,
                  (unsigned)lookupFBOName);
        }
    }

    if (!_pipelineCache.state->pipelineState) {
        static uint64_t nil_pipeline_count = 0;
        nil_pipeline_count++;
        if (nil_pipeline_count <= 8 || (nil_pipeline_count % 1000) == 0) {
            mglTraceLogNSString(@"MGL DRAW SKIP: pipelineState is nil, forcing rebuild (occurrence=%llu)",
                          (unsigned long long)nil_pipeline_count);
        }
        // Force rebuild on next state processing pass.
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_pipeline",
                                ctx,
                                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                _renderPassManager.state->currentRenderEncoder,
                                _renderPassManager.state->renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    RETURN_FALSE_ON_FAILURE([self validateRenderPassAttachmentsAndPipelineFormatsLocked:traceProcess]);

    @try {
        if (mglRenderCppBindingSetPipelineIfNeeded(
                _bindingStateOwner,
                (__bridge void *)_renderPassManager.state->currentRenderEncoder,
                (__bridge void *)_pipelineCache.state->pipelineState) > 0) {
            MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: processGLState - setRenderPipelineState failed: %@", exception.reason);
        // Force pipeline/state retranslation on next draw instead of crashing this frame.
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.set_pipeline",
                                ctx,
                                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                _renderPassManager.state->currentRenderEncoder,
                                _renderPassManager.state->renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    // Resource Sync domain (Resource Sync domain): stability rebind before draw. The logic was moved to
    // syncResourceBindingsForContext:, only the dispatch remains here; steps
    // the dirty-domain pass already ran this invocation are skipped.
    RETURN_FALSE_ON_FAILURE([self syncResourceBindingsForContext:ctx
                                                     alreadyDone:&resourceSyncWork]);

    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    BOOL useFragCoordParams =
        fragmentProgram && fragmentProgram->usesFragCoordParams == GL_TRUE;
    if (useFragCoordParams) {
        NSUInteger passHeight = mglRenderPassRenderTargetHeightFor(_renderPassManager.state);
        if (passHeight == 0 && _renderPassManager.state->renderPassDescriptor) {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS && passHeight == 0; i++) {
                id<MTLTexture> color = mglRenderPassColorTextureFor(_renderPassManager.state, i);
                passHeight = color ? color.height : 0;
            }
            if (passHeight == 0 && mglRenderPassDepthTextureFor(_renderPassManager.state)) {
                passHeight = mglRenderPassDepthTextureFor(_renderPassManager.state).height;
            }
            if (passHeight == 0 && mglRenderPassStencilTextureFor(_renderPassManager.state)) {
                passHeight = mglRenderPassStencilTextureFor(_renderPassManager.state).height;
            }
        }

        vector_float4 fragCoordParams = {
            (float)passHeight,
            MGL_STATE(ctx)->var.clip_origin == GL_LOWER_LEFT ? 1.0f : 0.0f,
            0.0f,
            0.0f
        };
        mglRenderPassSetFragmentBytes(
            _renderPassManager.state->currentRenderEncoder, &fragCoordParams,
            sizeof(fragCoordParams), kMGLFragCoordParamsBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLFragCoordParamsBufferIndex];
    }

    /* LOD_BIAS: Metal MTLSamplerDescriptor has no lodBias property, so MGL
     * injects bias() into MSL .sample() calls via mglPatchInjectLodBias.
     * Here we bind the actual bias value as a 4-byte fragment buffer.
     *
     * Global single bias — scan bound textures, use first non-zero
     * lod_bias.  When all are zero, bind 0.0 (no-op, matches Metal default). */
    BOOL useLodBias = fragmentProgram &&
        fragmentProgram->uses_lod_bias == GL_TRUE;
    if (useLodBias) {
        /* P7: Per-texture LOD_BIAS array — one float per sampler slot.
         * GL 4.6 §8.14.1 eq 8.8 defines biastexobj as per-texture state.
         * The injected MSL uses _mglLodBias[sampler_idx] to index this array. */
        const GLfloat biasmax = ctx->state.var.max_texture_lod_bias;
        float lodBiasArr[TEXTURE_UNITS];
        for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
            Texture *tex = MGL_STATE(ctx)->active_textures[unit];
            Sampler *smp = MGL_STATE(ctx)->texture_samplers[unit];
            /* GL 4.6 §8.2: sampler object state overrides texture state
             * when a sampler object is bound to the unit. */
            float bias = smp ? smp->params.lod_bias
                             : (tex ? tex->params.lod_bias : 0.0f);
            /* GL 4.6 §8.14.1 eq 8.8: clamp(biastexobj + biasshader) to
             * [-biasmax, biasmax].  biasshader is added in-shader by P9's
             * bias(clamp((expr) + _mglLodBias[idx], -_mglLodBiasMax,
             * _mglLodBiasMax)) rewrite; the clamp here covers biastexobj
             * only (CPU-side defensive pre-clamp).  Full sum clamp is done
             * in MSL via _mglLodBiasMax bound below. */
            if (biasmax > 0.0f) {
                if (bias > biasmax) bias = biasmax;
                else if (bias < -biasmax) bias = -biasmax;
            }
            lodBiasArr[unit] = bias;
        }
        mglRenderPassSetFragmentBytes(
            _renderPassManager.state->currentRenderEncoder, lodBiasArr,
            sizeof(lodBiasArr), kMGLLodBiasBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLLodBiasBufferIndex];

        /* Bind _mglLodBiasMax scalar (MAX_TEXTURE_LOD_BIAS) for the MSL
         * clamp(biastexobj + biasshader, -biasmax, biasmax) in
         * mglRewriteMSLBiasExpr and in the no-bias injected clamp().
         * GL 4.6 §8.14.1 eq 8.8. */
        mglRenderPassSetFragmentBytes(
            _renderPassManager.state->currentRenderEncoder, &biasmax,
            sizeof(biasmax), kMGLLodBiasMaxBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLLodBiasMaxBufferIndex];
    }

    if (draw_command &&
        mglFragmentTextureTraceBindingsUseRTSampledCopy(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS)) {
        [_renderPassManager setCurrentDrawUsesRTSampledCopy:YES];
        [self updateCurrentRenderEncoder];
    }

    double processElapsedUs = (mglTraceClockNS() - processStartNS) / 1000.0;
    if (traceProcess) {
        mglTraceLogNSString(@"MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.1fus",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedUs);
        mglLogStateSnapshot("processGLState.exit.ok",
                            ctx,
                            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                            _renderPassManager.state->currentRenderEncoder,
                            _renderPassManager.state->renderPassDescriptor,
                            _drawable);
    } else if (processElapsedUs >= 25.0) {
        mglTraceLogNSString(@"MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.1fus",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedUs);
    }
    return true;
}
/*
 * Dirty state domain processing extracted from processGLStateLocked:.
 * Handles all dirty-bits dispatch: DIRTY_FBO, DIRTY_STATE, DIRTY_PROGRAM/
 * VAO/BUFFER_BASE_STATE, DIRTY_TEX, DIRTY_VAO/BUFFER/RENDER_STATE, and the
 * pipeline sync call. Returns false on failure (caller should skip this
 * draw), true on success.
 */
- (bool)processDirtyStateDomainsLocked:(bool)draw_command
                                  work:(MGLResourceSyncWork *)work
{
    bool deferredBufferMapForPipelineBuild = false;
    if (MGL_STATE(ctx)->dirty_bits)
    {
        // FBO binding/attachment changes alter the Metal render pass itself. They must
        // be handled even when no generic DIRTY_STATE bit is present; otherwise the
        // current render encoder can keep drawing into an old attachment while GL state
        // already points at a different FBO. RenderPass Sync domain (RenderPass Sync domain).
        if (MGL_STATE(ctx)->dirty_bits & DIRTY_FBO)
        {
            RETURN_FALSE_ON_FAILURE([self syncRenderPassStateForContext:ctx]);
        }

        // dirty state covers all rendering attachments and general state
        if (MGL_STATE(ctx)->dirty_bits & DIRTY_STATE)
        {
            if (MGL_STATE(ctx)->dirty_bits & DIRTY_FBO)
            {
                // MEMORY SAFETY: Add comprehensive validation to prevent use-after-free crashes
                Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO");
                if (framebuffer)
                {
                    if (framebuffer->dirty_bits & DIRTY_FBO_BINDING)
                    {
                        RETURN_FALSE_ON_FAILURE([self bindFramebufferAttachmentTextures]);

                        // Additional validation after binding
                        framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO.afterBind");
                        if (framebuffer) {
                            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
                        }
                    }
                }

                // dirty FBO state can't be cleared just yet its needed below
            }

            MGL_STATE(ctx)->dirty_bits &= ~DIRTY_STATE;
        }

        // check for dirty program and vao
        // leave program / vao state dirty, buffers need to be mapped before used below
        // dirty program causes buffers to be remapped
        // dirty vao causes attributes to be remapped to new buffers
        // dirty buffer base causes buffers to be remapped to new indexes
        if (MGL_STATE(ctx)->dirty_bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_BUFFER_BASE_STATE))
        {
            // Avoid mapping draw buffers against a nil pipeline during startup/rebuild.
            // We'll map again after a valid pipeline is bound.
            bool deferBufferMapForNilPipeline =
                (draw_command &&
                 _pipelineCache.state->pipelineState == nil &&
                 (MGL_STATE(ctx)->dirty_bits & DIRTY_PROGRAM));

            if (deferBufferMapForNilPipeline) {
                deferredBufferMapForPipelineBuild = true;
                static uint64_t s_deferredMapCount = 0;
                s_deferredMapCount++;
                if (s_deferredMapCount <= 16 || (s_deferredMapCount % 1000ull) == 0ull) {
                    mglTraceLogNSString(@"MGL DRAW SKIP: pipelineState is nil (deferring buffer mapping, occurrence=%llu)",
                                  (unsigned long long)s_deferredMapCount);
                }
            } else {
                // programs are now compiled before execution, we shouldn't get here
                //assert(ctx->state.program->mtl_data); //

                // figure out vertex shader uniforms / buffer mappings
                RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
                if (work) work->mappedBuffers = true;
            }

            MGL_STATE(ctx)->dirty_bits &= ~DIRTY_BUFFER_BASE_STATE;
        }

        // Texture object uploads can be prepared before pipeline selection, but
        // sampled-resource binding must wait until after setRenderPipelineState()
        // so it uses the current program's sampler reflection.
        if (MGL_STATE(ctx)->dirty_bits & (DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER))
        {
            RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);
            if (work) work->boundActiveTextures = true;

            // textures / active textures and samplers are all handled in bindActiveTexturesToMTL
            MGL_STATE(ctx)->dirty_bits &= ~(DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER);
        }

        // A dirty VAO changes vertex buffer bindings and may require a new
        // pipeline descriptor, but it does not change the render-pass
        // attachments. Keep the current encoder alive so GL draw ordering and
        // depth/load-store continuity are preserved across HUD/hand/UI passes.
        if (MGL_STATE(ctx)->dirty_bits & DIRTY_VAO)
        {
            // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &MGL_STATE(ctx)->vertex_buffer_map_list]);
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &MGL_STATE(ctx)->fragment_buffer_map_list]);
            if (work) work->updatedBaseLists = true;

            if (!_renderPassManager.state->currentRenderEncoder) {
                RETURN_FALSE_ON_FAILURE(
                    [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_VAO]);
            }

            [self updateCurrentRenderEncoder];

            // clear dirty render state
            MGL_STATE(ctx)->dirty_bits &= ~DIRTY_RENDER_STATE;
        }
        else if (MGL_STATE(ctx)->dirty_bits & DIRTY_BUFFER)
        {
            // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &MGL_STATE(ctx)->vertex_buffer_map_list]);
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &MGL_STATE(ctx)->fragment_buffer_map_list]);
            if (work) work->updatedBaseLists = true;

            MGL_STATE(ctx)->dirty_bits &= ~DIRTY_BUFFER;
        }
        else if (MGL_STATE(ctx)->dirty_bits & DIRTY_RENDER_STATE)
        {
            if (_renderPassManager.state->currentRenderEncoder == NULL)
            {
                RETURN_FALSE_ON_FAILURE(
                    [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_RS]);
            }

            // a dirty render state may just be something like alpha changes which don't require a new renderbuffer

            // updateCurrentRenderEncoder will update the renderstate outside of creating a new one
            [self updateCurrentRenderEncoder];

            MGL_STATE(ctx)->dirty_bits &= ~DIRTY_RENDER_STATE;
        }

        // new pipeline / vertex / renderbuffer and pipelinestate descriptor, should probably make this a single dirty bit
        // Pipeline Sync domain (Pipeline Sync domain): when program/VAO/FBO/alpha/render-state changes,
        // rebuild or reuse the PSO. The logic was moved entirely to syncPipelineStateWithDeferredBufferMap:,
        // only the dispatch remains here; deferredBufferMap is passed as a value parameter (not read after the block).
        if (MGL_STATE(ctx)->dirty_bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE))
        {
            RETURN_FALSE_ON_FAILURE([self syncPipelineStateWithDeferredBufferMap:deferredBufferMapForPipelineBuild]);
        }

        //if (ctx->state.dirty_bits)
        //    logDirtyBits(ctx);

        // Unconditionally clear all dirty bits after processing.
        // All relevant state has been applied to Metal encoders above; any
        // remaining bits (e.g. DIRTY_DRAWABLE set at init, or bits accumulated
        // via |= in the defer path without DIRTY_ALL_BIT) are stale and would
        // cause false-positive rebinds on the next draw.
        MGL_STATE(ctx)->dirty_bits = 0;
    }
    else // if (ctx->state.dirty_bits)
    {
        // buffer data can be changed but the bindings remain in place.. so we need to update the data if this is the case
        // like a uniform or buffer sub data call
        MGLEncodeContext encCtx = { .encoder = _renderPassManager.state->currentRenderEncoder };

        if( [self checkForDirtyBufferData: &MGL_STATE(ctx)->vertex_buffer_map_list])
        {
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &MGL_STATE(ctx)->vertex_buffer_map_list]);

            RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder:&encCtx]);
        }

        if( [self checkForDirtyBufferData: &MGL_STATE(ctx)->fragment_buffer_map_list])
        {
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &MGL_STATE(ctx)->fragment_buffer_map_list]);

            RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]);
        }
    }
    return true;
}

/*
 * Render pass descriptor and pipeline format validation extracted from
 * processGLStateLocked:. Validates render-pass attachments and checks
 * pipeline/pass color, depth, and stencil format compatibility. Returns
 * false to skip the draw on validation failure, true to continue.
 */
- (bool)validateRenderPassAttachmentsAndPipelineFormatsLocked:(BOOL)traceProcess
{
    // Guard against invalid render pass state before binding pipeline.
    // Metal debug validation can abort the process if the encoder/render pass is incompatible.
    // P4.1f: under gate-on the C++ RenderPassStateOwner is the validity signal.
    BOOL hasRenderPassState = mglRenderPassUsesMetalCpp()
        ? (_renderPassManager.state->renderPassStateOwner != NULL)
        : (_renderPassManager.state->renderPassDescriptor != nil);
    if (!hasRenderPassState) {
        NSLog(@"MGL ERROR: processGLState - render pass %s is nil before pipeline bind",
              mglRenderPassUsesMetalCpp() ? "state owner" : "descriptor");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_rpd",
                                ctx,
                                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                _renderPassManager.state->currentRenderEncoder,
                                _renderPassManager.state->renderPassDescriptor,
                                _drawable);
        }
        return false;
    }
    BOOL passHasAnyAttachment = NO;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> colorAttachment = mglRenderPassColorTextureFor(_renderPassManager.state, i);
        if (colorAttachment) {
            passHasAnyAttachment = YES;
            if ((colorAttachment.usage & MTLTextureUsageRenderTarget) == 0) {
                NSLog(@"MGL WARNING: processGLState - color attachment %d missing RenderTarget usage (usage=0x%lx); skipping draw",
                      i,
                      (unsigned long)colorAttachment.usage);
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.color_usage",
                                        ctx,
                                        (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                        _renderPassManager.state->currentRenderEncoder,
                                        _renderPassManager.state->renderPassDescriptor,
                                        _drawable);
                }
                return false;
            }
        }
    }
    if (mglRenderPassDepthTextureFor(_renderPassManager.state) ||
        mglRenderPassStencilTextureFor(_renderPassManager.state)) {
        passHasAnyAttachment = YES;
    }

    if (!passHasAnyAttachment) {
        NSLog(@"MGL WARNING: processGLState - render pass has no attachments, skipping draw to avoid Metal assert");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.no_attachments",
                                ctx,
                                (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner),
                                _renderPassManager.state->currentRenderEncoder,
                                _renderPassManager.state->renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    MTLPixelFormat currentColor0Format = MTLPixelFormatInvalid;
    MTLPixelFormat currentDepthFormat = MTLPixelFormatInvalid;
    MTLPixelFormat currentStencilFormat = MTLPixelFormatInvalid;

    id<MTLTexture> rpColor0 = mglRenderPassColorTextureFor(_renderPassManager.state, 0);
    id<MTLTexture> rpDepth = mglRenderPassDepthTextureFor(_renderPassManager.state);
    id<MTLTexture> rpStencil = mglRenderPassStencilTextureFor(_renderPassManager.state);
    if (rpColor0) {
        currentColor0Format = rpColor0.pixelFormat;
    }
    if (rpDepth) {
        currentDepthFormat = rpDepth.pixelFormat;
    }
    if (rpStencil) {
        currentStencilFormat = rpStencil.pixelFormat;
    }

    // IMPORTANT:
    // Never mutate depth/stencil attachments here to "fit" an existing pipeline.
    // The active Metal render encoder was already created with a render-pass descriptor,
    // and changing attachments after encoder creation does not make that encoder compatible.
    // We must instead reject mismatched pipeline/pass combinations and rebuild safely.

    if (_pipelineCache.state->pipelineColor0Format != MTLPixelFormatInvalid &&
        currentColor0Format != MTLPixelFormatInvalid &&
        _pipelineCache.state->pipelineColor0Format != currentColor0Format) {
        static uint64_t s_colorFormatMismatchCount = 0;
        s_colorFormatMismatchCount++;
	        if (s_colorFormatMismatchCount <= 16 || (s_colorFormatMismatchCount % 250) == 0) {
	            NSLog(@"MGL WARNING: Pipeline/pass color format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	                  (unsigned long)_pipelineCache.state->pipelineColor0Format, (unsigned long)currentColor0Format);
	        }
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass color format mismatch"];
	        mglMarkRendererDirtyBits(ctx->active_state,
	                                 DIRTY_PROGRAM | DIRTY_VAO |
	                                 DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }

    if (_pipelineCache.state->pipelineDepthFormat != currentDepthFormat) {
        BOOL pipelineHasDepth = (_pipelineCache.state->pipelineDepthFormat != MTLPixelFormatInvalid);
        BOOL passHasDepth = (currentDepthFormat != MTLPixelFormatInvalid);
        if (!pipelineHasDepth && !passHasDepth) {
            goto depth_format_ok;
	        }
	        {
	            static uint64_t s_depthFormatMismatchCount = 0;
	            s_depthFormatMismatchCount++;
	            if (s_depthFormatMismatchCount <= 16 || (s_depthFormatMismatchCount % 250) == 0) {
	                NSLog(@"MGL WARNING: Pipeline/pass depth format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	                      (unsigned long)_pipelineCache.state->pipelineDepthFormat, (unsigned long)currentDepthFormat);
	            }
	        }
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass depth format mismatch"];
	        mglMarkRendererDirtyBits(ctx->active_state,
	                                 DIRTY_PROGRAM | DIRTY_VAO |
	                                 DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }
depth_format_ok:;

    if (_pipelineCache.state->pipelineStencilFormat != currentStencilFormat) {
        BOOL pipelineHasStencil = (_pipelineCache.state->pipelineStencilFormat != MTLPixelFormatInvalid);
        BOOL passHasStencil = (currentStencilFormat != MTLPixelFormatInvalid);
        if (!pipelineHasStencil && !passHasStencil) {
            goto stencil_format_ok;
	        }
	        {
	            static uint64_t s_stencilFormatMismatchCount = 0;
	            s_stencilFormatMismatchCount++;
	            if (s_stencilFormatMismatchCount <= 16 || (s_stencilFormatMismatchCount % 250) == 0) {
	                NSLog(@"MGL WARNING: Pipeline/pass stencil format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	                      (unsigned long)_pipelineCache.state->pipelineStencilFormat, (unsigned long)currentStencilFormat);
	            }
	        }
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass stencil format mismatch"];
	        mglMarkRendererDirtyBits(ctx->active_state,
	                                 DIRTY_PROGRAM | DIRTY_VAO |
	                                 DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }
stencil_format_ok:;
    return true;
}

/*
 * Pipeline Sync domain (Pipeline Sync domain). PSO build/reuse logic moved verbatim from processGLStateLocked:
 * generates pipeline+vertex descriptor, queries/builds PSO cache, interface-mismatch
 * circuit breaker, failure fallback chain. Only operates on Metal pipeline state, state is read via ctx (same as before the move).
 * deferredBufferMap is passed in by the caller (deferred buffer mapping flag for nil pipeline).
 * Returns false to indicate this draw should be skipped (equivalent to the original inline return false semantics).
 */
- (bool)syncPipelineStateWithDeferredBufferMap:(bool)deferredBufferMapForPipelineBuild
{
            GLMState *state = MGL_STATE(ctx);
            /* Force a rebind of the pipeline state on the next setRenderPipelineState
             * call. Dirty program/VAO/FBO/render-state may rebuild or reuse the
             * pipeline, but the encoder still needs the binding re-issued.
             *
             * Task 5 gated fast path: when MGL_PSO_DEDUP is enabled (default ON)
             * and the render
             * encoder is unchanged (the C++ binding cache is valid) and the resolved
             * pipeline state pointer is identical to the previously bound
             * state matches the C++ binding cache, the nil assignment
             * is skipped. This allows the dedup check in
             * processGLStateLocked:'s setRenderPipelineState: path to
             * recognize the encoder already has the correct PSO bound and
             * skip the redundant MTL call. If any condition is false, the
             * original conservative nil assignment executes. */
            if (_pipelineCache.state->psoDedupEnabled &&
                mglBindingStateIsValid(_bindingStateOwner) &&
                mglBindingStatePipelineMatches(
                    _bindingStateOwner,
                    (__bridge void *)_pipelineCache.state->pipelineState)) {
                MGL_PERF_INC(g_mglPSODedupHitsSinceSwap);
            } else {
                mglRenderCppBindingSetPipelineState(_bindingStateOwner, NULL);
                MGL_PERF_INC(g_mglPSODedupMissesSinceSwap);
            }
            CFTimeInterval now = CFAbsoluteTimeGetCurrent();
            bool skipPipelineBuild = false;
            Program *currentVertexProgram = _tessellation.nativeTESActive
                ? _tessellation.nativeTESProgram
                : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *currentFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            GLuint currentProgramName = mglCurrentRenderProgramKey(ctx);
            VertexArray *currentVAO = state->vao;
            Framebuffer *currentFBO = mglRendererGetValidatedFramebuffer(ctx, "processGLState.currentFBO");
            GLuint currentFBOName = currentFBO ? currentFBO->name : 0;

            // Program-level breaker (independent of render-pass signature) to avoid
            // mismatch storms where color/depth/stencil signatures keep changing.
            if (_pipelineCache.state->pipelineState != nil &&
                currentProgramName != 0 &&
                currentProgramName == _gpuRecovery.programMismatchProgramName &&
                now < _gpuRecovery.programMismatchRetryAfter) {
                static uint64_t s_programMismatchSkipCount = 0;
                s_programMismatchSkipCount++;
                if (s_programMismatchSkipCount <= 16 || (s_programMismatchSkipCount % 1000ull) == 0ull) {
                    double remaining = _gpuRecovery.programMismatchRetryAfter - now;
                    if (remaining < 0.0) remaining = 0.0;
                    NSLog(@"MGL WARNING: Program-level mismatch breaker active (program=%u, %.2fs remaining), skipping draw",
                          (unsigned)currentProgramName,
                          remaining);
                }
                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

	            if (now < _gpuRecovery.pipelineRetryAfter) {
	                BOOL retryAppliesToCurrentProgram =
	                    (currentProgramName != 0 &&
	                     (currentProgramName == _gpuRecovery.interfaceMismatchProgramName ||
	                      currentProgramName == _gpuRecovery.programMismatchProgramName ||
	                      currentProgramName == _gpuRecovery.interfaceMismatchBlockedProgram));

	                if (retryAppliesToCurrentProgram) {
	                    if (_pipelineCache.state->pipelineState) {
		                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	                    // Keep existing pipeline, but do not early-return before setRenderPipelineState.
		                    skipPipelineBuild = true;
	                    } else {
	                        _gpuRecovery.pipelineRetryAfter = 0.0;
	                        _gpuRecovery.programMismatchRetryAfter = 0.0;
	                        _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
	                    }
		                } else {
	                    static uint64_t s_retryBypassCount = 0;
	                    s_retryBypassCount++;
	                    if (s_retryBypassCount <= 16 || (s_retryBypassCount % 1000ull) == 0ull) {
	                        NSLog(@"MGL PIPELINE RETRY bypass global retry for unrelated program=%u mismatchProgram=%u blockedProgram=%u",
	                              (unsigned)currentProgramName,
	                              (unsigned)_gpuRecovery.interfaceMismatchProgramName,
	                              (unsigned)_gpuRecovery.interfaceMismatchBlockedProgram);
	                    }
	                }
	            }

            if (!skipPipelineBuild) {
            // create pipeline descriptor
            // P4.2: gate-on 走 value-state（MGLRenderCppPipelineDescriptorState，
            // 不组装 MTLRenderPipelineDescriptor）；gate-off 保留 ObjC descriptor
            // 作为 A/B 基线。
            const BOOL usesMetalCpp = mglRenderPassUsesMetalCpp();
            MTLRenderPipelineDescriptor *pipelineStateDescriptor = nil;
            MTLVertexDescriptor *vertexDescriptor = nil;
            MGLRenderCppPipelineDescriptorState psoState = {0};
            id<MTLFunction> psoVertexFunction = nil;
            id<MTLFunction> psoFragmentFunction = nil;
            uint32_t builtColor0Format = (uint32_t)MTLPixelFormatInvalid;
            uint32_t builtDepthFormat = (uint32_t)MTLPixelFormatInvalid;
            uint32_t builtStencilFormat = (uint32_t)MTLPixelFormatInvalid;

            if (usesMetalCpp) {
                /* gate-on：blend 状态先同步到 C++ owner（value-state 填充从中
                 * owner-first 读取），再整体生成 descriptor state。 */
                [self updateBlendStateCache];
                state->dirty_bits &= ~DIRTY_ALPHA_STATE;
                if (![self generatePipelineDescriptorState:&psoState
                                            vertexFunction:&psoVertexFunction
                                          fragmentFunction:&psoFragmentFunction]) {
                    NSLog(@"MGL PIPELINE CREATE fail error=generatePipelineDescriptorState returned NO");
                    [self invalidateCurrentPipelineStateForReason:@"pipeline descriptor failure"];
                    _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                    mglMarkRendererDirtyBits(state,
                                             DIRTY_PROGRAM | DIRTY_VAO |
                                             DIRTY_FBO | DIRTY_RENDER_STATE);
                    return false;
                }
                builtColor0Format = psoState.color_format[0];
                builtDepthFormat = psoState.depth_format;
                builtStencilFormat = psoState.stencil_format;
            } else {
                pipelineStateDescriptor = [self generatePipelineDescriptor];
                if (!pipelineStateDescriptor) {
                    NSLog(@"MGL PIPELINE CREATE fail error=generatePipelineDescriptor returned nil");
                    [self invalidateCurrentPipelineStateForReason:@"pipeline descriptor failure"];
                    _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                    mglMarkRendererDirtyBits(state,
                                             DIRTY_PROGRAM | DIRTY_VAO |
                                             DIRTY_FBO | DIRTY_RENDER_STATE);
                    return false;
                }
                builtColor0Format = (uint32_t)pipelineStateDescriptor.colorAttachments[0].pixelFormat;
                builtDepthFormat = (uint32_t)pipelineStateDescriptor.depthAttachmentPixelFormat;
                builtStencilFormat = (uint32_t)pipelineStateDescriptor.stencilAttachmentPixelFormat;
            }

            // Circuit breaker for repeated VS/FS interface mismatch.
            if (now < _gpuRecovery.interfaceMismatchRetryAfter &&
                currentProgramName == _gpuRecovery.interfaceMismatchProgramName &&
                builtColor0Format == (uint32_t)_gpuRecovery.interfaceMismatchColor0Format &&
                builtDepthFormat == (uint32_t)_gpuRecovery.interfaceMismatchDepthFormat &&
                builtStencilFormat == (uint32_t)_gpuRecovery.interfaceMismatchStencilFormat) {
                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

            if (!usesMetalCpp) {
            // create vertex descriptor (gate-off；gate-on 在
            // generatePipelineDescriptorState 内完成顶点布局填充)

            vertexDescriptor = (_geometry.expansionActive ||
                                _tessellation.tessComputeActive)
                ? [[MTLVertexDescriptor alloc] init]
                : [self generateVertexDescriptor];
	            if (!vertexDescriptor) {
	                NSLog(@"MGL PIPELINE CREATE fail error=generateVertexDescriptor returned nil");
	                [self invalidateCurrentPipelineStateForReason:@"vertex descriptor failure"];
	                _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                mglMarkRendererDirtyBits(state,
                                         DIRTY_PROGRAM | DIRTY_VAO |
                                         DIRTY_FBO | DIRTY_RENDER_STATE);
                return false;
            }

            [self updateBlendStateCache];
            state->dirty_bits &= ~DIRTY_ALPHA_STATE;
            [self bindBlendStateToPipelineStateDescriptor:pipelineStateDescriptor];

            if (kMGLVerbosePipelineLogs) {
                MTLRenderPipelineColorAttachmentDescriptor *ca0 = pipelineStateDescriptor.colorAttachments[0];
                NSLog(@"MGL PIPELINE DESC c0 state program=%u fmt=%lu writeMask=0x%x blend=%d srcRGB=%lu dstRGB=%lu srcA=%lu dstA=%lu opRGB=%lu opA=%lu",
                      (unsigned)currentProgramName,
                      (unsigned long)ca0.pixelFormat,
                      (unsigned)ca0.writeMask,
                      ca0.blendingEnabled ? 1 : 0,
                      (unsigned long)ca0.sourceRGBBlendFactor,
                      (unsigned long)ca0.destinationRGBBlendFactor,
                      (unsigned long)ca0.sourceAlphaBlendFactor,
                      (unsigned long)ca0.destinationAlphaBlendFactor,
                      (unsigned long)ca0.rgbBlendOperation,
                      (unsigned long)ca0.alphaBlendOperation);
            }

            pipelineStateDescriptor.vertexDescriptor = vertexDescriptor;
            }
            BOOL hasPipelineCacheKey = NO;
            bool pipelineResolvedFromCache = false;
            uint64_t pipelineSig = 0;
            uint64_t vertexSig = 0;
            /* Function-scope key words: filled inside the lookup block below,
             * read by the miss path after it.  Only meaningful when
             * currentProgramName != 0. */
            uint64_t keyWords[MGL_PIPELINE_CACHE_KEY_WORDS] = {0};

            if (!pipelineResolvedFromCache && currentProgramName != 0) {
                /* P4.2: gate-on 对 value-state 计算签名（哈希字段/顺序与
                 * descriptor 版一致，见 mgl_vertex_format.m）。 */
                pipelineSig = usesMetalCpp
                    ? mglPipelineDescriptorSignatureFromState(&psoState)
                    : mglPipelineDescriptorSignature(pipelineStateDescriptor);
                vertexSig = usesMetalCpp
                    ? mglVertexDescriptorSignatureFromState(&psoState)
                    : mglVertexDescriptorSignature(vertexDescriptor);

                /* Keep descriptor signatures and linked Program identities
                 * lossless. GL names can be reused and a Program can relink
                 * without changing its name. */
                uint64_t primaryKey = (((uint64_t)currentProgramName << 32)
                                     | (((uint64_t)state->var.clip_origin & 0xFu) << 28)
                                     | (((uint64_t)state->var.clip_depth_mode & 0xFu) << 24)
                                     | (_tessellation.nativeTESActive ? (1ull << 23) : 0ull)
                                     | (_tessellation.tessVertexCaptureActive ? (1ull << 22) : 0ull)
                                     | (_geometry.expansionActive ? (1ull << 21) : 0ull)
                                     | (_tessellation.cullDistanceCaptureActive ? (1ull << 20) : 0ull)
                                     | (_tessellation.tessComputeActive ? (1ull << 19) : 0ull));
                uint64_t vertexInstance = currentVertexProgram
                    ? currentVertexProgram->pipeline_cache_instance_id : 0u;
                uint64_t vertexGeneration = currentVertexProgram
                    ? currentVertexProgram->pipeline_cache_generation : 0u;
                uint64_t fragmentInstance = currentFragmentProgram
                    ? currentFragmentProgram->pipeline_cache_instance_id : 0u;
                uint64_t fragmentGeneration = currentFragmentProgram
                    ? currentFragmentProgram->pipeline_cache_generation : 0u;
                keyWords[0] = primaryKey;
                keyWords[1] = vertexInstance;
                keyWords[2] = vertexGeneration;
                keyWords[3] = fragmentInstance;
                keyWords[4] = fragmentGeneration;
                keyWords[5] = pipelineSig;
                keyWords[6] = vertexSig;
                /* Hit path uses the reusable zero-alloc query key.  The key
                 * is only valid for lookups; the miss path below allocates a
                 * fresh key for the store/compile path so overwriteWords:
                 * cannot corrupt cache dictionaries. */
                hasPipelineCacheKey = YES;

                /* Two-level cache lookup:
                 * Level 1: PSO cache (fastest - compiled pipeline ready to use)
                 * Level 2: Descriptor cache (fast - skip expensive descriptor regeneration)
                 * On double miss: regenerate descriptor + compile PSO */
                id<MTLRenderPipelineState> cachedPipeline = nil;
                id<MTLFunction> cachedVertexFunction = nil;
                id<MTLFunction> cachedFragmentFunction = nil;
                BOOL cachedFunctionMetadataPresent = [_pipelineCache
                    lookupPipelineForWords:keyWords
                    pipeline:&cachedPipeline
                    vertexFunction:&cachedVertexFunction
                    fragmentFunction:&cachedFragmentFunction];
                if (cachedPipeline) {
                    /* PSO cache hit - fastest path */
                    static uint64_t s_pipelineCacheHitCount = 0;
                    s_pipelineCacheHitCount++;
                    MGL_PERF_INC(g_mglPipelineCacheHitsSinceSwap);
                    if (kMGLVerbosePipelineLogs &&
                            (s_pipelineCacheHitCount <= 128ull || (s_pipelineCacheHitCount % 1000ull) == 0ull)) {
                        NSLog(@"MGL PIPELINE CACHE hit program=%u vao=%p fbo=%u key=%@",
                              (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName,
                              [NSString stringWithFormat:@"%016llx/%016llx/%016llx",
                               (unsigned long long)keyWords[0],
                               (unsigned long long)keyWords[5],
                               (unsigned long long)keyWords[6]]);
                    }

                    [_pipelineCache activatePipelineState:cachedPipeline
                                           color0Format:builtColor0Format
                                            depthFormat:builtDepthFormat
                                          stencilFormat:builtStencilFormat
                                            programName:currentProgramName
                                         vertexFunction:cachedFunctionMetadataPresent
                                             ? cachedVertexFunction
                                             : (usesMetalCpp ? psoVertexFunction
                                                             : pipelineStateDescriptor.vertexFunction)
                                       fragmentFunction:cachedFunctionMetadataPresent
                                             ? cachedFragmentFunction
                                             : (usesMetalCpp ? psoFragmentFunction
                                                             : pipelineStateDescriptor.fragmentFunction)];
                    pipelineResolvedFromCache = true;
                    /* Hit path deliberately skips the LRU touch: touching
                     * would require copying the query-keyed object that must
                     * never enter the LRU (see pipelineQueryKeyForWords:),
                     * reintroducing the per-draw alloc this avoids.  Mirrors
                     * the depth-stencil cache policy. */

	                    // Mirror successful compile-side breaker resets.
	                    _gpuRecovery.interfaceMismatchStreak = 0;
	                    _gpuRecovery.interfaceMismatchProgramName = 0;
	                    _gpuRecovery.interfaceMismatchColor0Format = MTLPixelFormatInvalid;
	                    _gpuRecovery.interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
	                    _gpuRecovery.interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
	                    _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
	                    if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
	                        _gpuRecovery.programMismatchProgramName = 0;
	                        _gpuRecovery.programMismatchRetryAfter = 0.0;
	                        _gpuRecovery.programMismatchStreak = 0u;
	                    }
	                    if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                        _gpuRecovery.interfaceMismatchBlockedProgram = 0;
                        _gpuRecovery.interfaceMismatchBlockedUntil = 0.0;
                        _gpuRecovery.interfaceMismatchBlockedStreak = 0u;
                    }
	                }
	            }

	            // PROPER AGX VIRTUALIZATION COMPATIBILITY: Fix root cause while maintaining Metal functionality
            if (!pipelineResolvedFromCache) {
                /* Compile/store path needs its own key object: the reusable
                 * query key words are overwritten on every lookup and must
                 * never be retained by the cache dictionaries/LRU.  One heap
                 * allocation on a cache miss is negligible against the PSO
                 * compile itself. */
                const uint64_t *storeKeyWords = hasPipelineCacheKey
                    ? keyWords : NULL;
                if (usesMetalCpp) {
                    /* P4.2 gate-on：value-state PSO build（final/simple/safe
                     * 全部在 C++ builder 内完成，无 MTLRenderPipelineDescriptor）。 */
                    return [self buildPipelineStateOnCacheMissWithState:&psoState
                                                         vertexFunction:psoVertexFunction
                                                       fragmentFunction:psoFragmentFunction
                                                           cacheKeyWords:storeKeyWords
                                                            pipelineSig:pipelineSig
                                                             vertexSig:vertexSig
                                                    builtColor0Format:builtColor0Format
                                                     builtDepthFormat:builtDepthFormat
                                                   builtStencilFormat:builtStencilFormat
                                                          programName:currentProgramName
                                                                 now:now];
                }
                return [self buildPipelineStateOnCacheMissWithDescriptor:pipelineStateDescriptor
                                                     vertexDescriptor:vertexDescriptor
                                                        cacheKeyWords:storeKeyWords
                                                         pipelineSig:pipelineSig
                                                          vertexSig:vertexSig
                                                  builtColor0Format:builtColor0Format
                                                   builtDepthFormat:builtDepthFormat
                                                 builtStencilFormat:builtStencilFormat
                                                        programName:currentProgramName
                                                                now:now];
            }

                if (deferredBufferMapForPipelineBuild && _pipelineCache.state->pipelineState != nil) {
                    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
                    deferredBufferMapForPipelineBuild = false;
                }

	            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	            }

    return true;
}

/* P4.2 gate-on：与 buildPipelineStateOnCacheMissUnlockedWithDescriptor 完全
 * 等价的 PSO build，但输入是 MGLRenderCppPipelineDescriptorState value-state
 * （无 MTLRenderPipelineDescriptor）。descriptor state cache 命中时复用缓存
 * state；final/simple/safe 三套组装全部在 C++ builder
 * （mglRenderCppCreateRenderPipelineFromState）内完成；二进制归档在 C++
 * builder 内 apply/add。GPU recovery / interface-mismatch / safe-fallback
 * 语义与 ObjC 版本一致（air_pipeline_safe_fallback 回归依赖 MGL_FORCE_SAFE_
 * FALLBACK_PIPELINE 测试钩子）。 */
- (bool)buildPipelineStateOnCacheMissWithState:(const MGLRenderCppPipelineDescriptorState *)pipelineState
                                vertexFunction:(id<MTLFunction>)vertexFunction
                              fragmentFunction:(id<MTLFunction>)fragmentFunction
                                  cacheKeyWords:(const uint64_t *)pipelineCacheKeyWords
                                    pipelineSig:(uint64_t)pipelineSig
                                     vertexSig:(uint64_t)vertexSig
                            builtColor0Format:(uint32_t)builtColor0Format
                             builtDepthFormat:(uint32_t)builtDepthFormat
                           builtStencilFormat:(uint32_t)builtStencilFormat
                                 programName:(GLuint)currentProgramName
                                        now:(CFTimeInterval)now
{
    GLMState *state = MGL_STATE(ctx);
    Program *currentProgram = mglResolveProgramFromState(ctx);
    Program *currentVertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *currentFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    VertexArray *currentVAO = state->vao;
    Framebuffer *currentFBO = mglRendererGetValidatedFramebuffer(ctx, "buildPipelineCacheOnCacheMiss.currentFBO");
    GLuint currentFBOName = currentFBO ? currentFBO->name : 0;

    /* Two-level descriptor caching (value-state 版) */
    MGLRenderCppPipelineDescriptorState finalState = *pipelineState;
    BOOL stateFromCache = NO;

    /* Check descriptor state cache on PSO miss; cache new states for reuse. */
    if (pipelineCacheKeyWords) {
        MGLRenderCppPipelineDescriptorState cachedState = {0};
        if ([_pipelineCache pipelineDescriptorStateForWords:pipelineCacheKeyWords
                                                      state:&cachedState]) {
            /* Descriptor cache hit - reuse cached state instead of regenerating */
            finalState = cachedState;
            stateFromCache = YES;
            static uint64_t s_descriptorCacheHitCount = 0;
            s_descriptorCacheHitCount++;
            if (kMGLVerbosePipelineLogs && s_descriptorCacheHitCount <= 64ull) {
                NSLog(@"MGL DESCRIPTOR CACHE hit program=%u key=%@ (total %llu)",
                (unsigned)currentProgramName,
                [NSString stringWithFormat:@"%016llx/%016llx/%016llx",
                 (unsigned long long)pipelineCacheKeyWords[0],
                 (unsigned long long)pipelineCacheKeyWords[5],
                 (unsigned long long)pipelineCacheKeyWords[6]],
                (unsigned long long)s_descriptorCacheHitCount);
            }
        }
    }

    MGL_PERF_INC(g_mglPipelineCacheMissesSinceSwap);
    MGLRenderCppPipelineDescriptorState successfulState = {0};
    BOOL haveSuccessfulState = NO;
    id<MTLRenderPipelineState> previousPipelineState = _pipelineCache.state->pipelineState;
    id<MTLRenderPipelineState> compiledPSO = nil;
    bool pipelineReusedPrevious = false;
    char cppError[512] = {0};
    /* 方法级作用域：@try 内赋值、@catch 的 safe fallback 也要用。 */
    void *psoPtr = NULL;
    void *binaryArchive = NULL;

    @try {
        static uint64_t s_pipelineCreateBeginCount = 0;
        s_pipelineCreateBeginCount++;
        if (kMGLVerbosePipelineLogs &&
        (s_pipelineCreateBeginCount <= 128ull || (s_pipelineCreateBeginCount % 500ull) == 0ull)) {
            NSLog(@"MGL PIPELINE CREATE begin program=%u vao=%p fbo=%u",
            (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName);
        }

        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL INFO: Creating Metal pipeline state with AGX virtualization compatibility...");
        }

        /* Test hook (air_pipeline_safe_fallback regression): force the
         * pipeline-creation exception so the safe-fallback branch below is
         * exercised deterministically. */
        if (mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE")) {
            NSLog(@"MGL TEST: forcing safe-fallback pipeline path");
            @throw [NSException exceptionWithName:@"MGLForcedSafeFallback"
                                           reason:@"synthetic pipeline creation failure (test hook)"
                                         userInfo:nil];
        }

        if (_pipelineCache.state->binaryArchiveEnabled) {
            binaryArchive = (__bridge void *)_pipelineCache.state->binaryArchive;
        }
        psoPtr = NULL;
        cppError[0] = '\0';
        if (mglRenderCppCreateRenderPipelineFromState(
                (__bridge void *)vertexFunction,
                fragmentFunction ? (__bridge void *)fragmentFunction : NULL,
                &finalState, binaryArchive, &psoPtr,
                cppError, sizeof(cppError)) != 0 || !psoPtr) {
            if (cppError[0]) {
                NSLog(@"MGL METALCPP PSO fallback: %s", cppError);
            }
        } else {
            compiledPSO = (__bridge_transfer id<MTLRenderPipelineState>)psoPtr;
        }
        if (compiledPSO) {
            mglMetalCountCreate(MGLMetalKindPSO);
            successfulState = finalState;
            haveSuccessfulState = YES;
        }

        if (!compiledPSO) {
            NSString *errDesc = cppError[0]
                ? [NSString stringWithUTF8String:cppError] : @"";
            BOOL isInterfaceMismatch =
                [errDesc containsString:@"mismatching vertex shader output"] ||
                [errDesc containsString:@"not written by vertex shader"];

            if (isInterfaceMismatch) {
                mglWriteProgramMSLDump(currentVertexProgram, errDesc);
                if (currentFragmentProgram && currentFragmentProgram != currentVertexProgram) {
                    mglWriteProgramMSLDump(currentFragmentProgram, errDesc);
                } else if (!currentVertexProgram) {
                    mglWriteProgramMSLDump(currentProgram, errDesc);
                }
                BOOL sameProgram =
                (_pipelineCache.state->pipelineProgramName != 0 &&
                _pipelineCache.state->pipelineProgramName == currentProgramName &&
                _pipelineCache.state->pipelineVertexFunction == vertexFunction &&
                _pipelineCache.state->pipelineFragmentFunction == fragmentFunction);
                BOOL colorCompatible = (_pipelineCache.state->pipelineColor0Format == MTLPixelFormatInvalid ||
                builtColor0Format == (uint32_t)MTLPixelFormatInvalid ||
                (uint32_t)_pipelineCache.state->pipelineColor0Format == builtColor0Format);
                BOOL depthCompatible = (_pipelineCache.state->pipelineDepthFormat == MTLPixelFormatInvalid ||
                builtDepthFormat == (uint32_t)MTLPixelFormatInvalid ||
                (uint32_t)_pipelineCache.state->pipelineDepthFormat == builtDepthFormat);
                BOOL stencilCompatible = (_pipelineCache.state->pipelineStencilFormat == MTLPixelFormatInvalid ||
                builtStencilFormat == (uint32_t)MTLPixelFormatInvalid ||
                (uint32_t)_pipelineCache.state->pipelineStencilFormat == builtStencilFormat);

                if (previousPipelineState && sameProgram && colorCompatible && depthCompatible && stencilCompatible) {
                    NSLog(@"MGL WARNING: Interface mismatch for program %u; reusing previous compatible pipeline once",
                    (unsigned)currentProgramName);
                    compiledPSO = previousPipelineState;
                    pipelineReusedPrevious = true;
                    _gpuRecovery.interfaceMismatchProgramName = currentProgramName;
                    _gpuRecovery.interfaceMismatchColor0Format = (MTLPixelFormat)builtColor0Format;
                    _gpuRecovery.interfaceMismatchDepthFormat = (MTLPixelFormat)builtDepthFormat;
                    _gpuRecovery.interfaceMismatchStencilFormat = (MTLPixelFormat)builtStencilFormat;
                    _gpuRecovery.interfaceMismatchStreak = 1u;
                    _gpuRecovery.interfaceMismatchRetryAfter = now + 0.10;
                    _gpuRecovery.pipelineRetryAfter = _gpuRecovery.interfaceMismatchRetryAfter;
                } else {
                    BOOL sameMismatchSignature =
                    (currentProgramName == _gpuRecovery.interfaceMismatchProgramName &&
                    builtColor0Format == (uint32_t)_gpuRecovery.interfaceMismatchColor0Format &&
                    builtDepthFormat == (uint32_t)_gpuRecovery.interfaceMismatchDepthFormat &&
                    builtStencilFormat == (uint32_t)_gpuRecovery.interfaceMismatchStencilFormat);
                    if (sameMismatchSignature) {
                        if (_gpuRecovery.interfaceMismatchStreak < UINT32_MAX) {
                            _gpuRecovery.interfaceMismatchStreak++;
                        }
                    } else {
                        _gpuRecovery.interfaceMismatchStreak = 1;
                        _gpuRecovery.interfaceMismatchProgramName = currentProgramName;
                        _gpuRecovery.interfaceMismatchColor0Format = (MTLPixelFormat)builtColor0Format;
                        _gpuRecovery.interfaceMismatchDepthFormat = (MTLPixelFormat)builtDepthFormat;
                        _gpuRecovery.interfaceMismatchStencilFormat = (MTLPixelFormat)builtStencilFormat;
                    }

                    // Exponential backoff: 0.10, 0.20, 0.40, 0.80, 1.60, capped at 2.00 sec.
                    uint32_t cappedShift = (_gpuRecovery.interfaceMismatchStreak > 5u) ? 4u : (_gpuRecovery.interfaceMismatchStreak - 1u);
                    double retryDelay = 0.10 * (double)(1u << cappedShift);
                    if (retryDelay > 2.0) {
                        retryDelay = 2.0;
                    }
                    _gpuRecovery.interfaceMismatchRetryAfter = now + retryDelay;

                    if (_gpuRecovery.interfaceMismatchStreak <= 5u || (_gpuRecovery.interfaceMismatchStreak % 200u) == 0u) {
                        NSLog(@"MGL WARNING: Interface mismatch (program=%u, streak=%u), throttling retries for %.2fs",
                        (unsigned)currentProgramName,
                        (unsigned)_gpuRecovery.interfaceMismatchStreak,
                        retryDelay);
                    }

                    // Program-level breaker update (ignores attachment signature).
                    if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
                        if (_gpuRecovery.programMismatchStreak < UINT32_MAX) {
                            _gpuRecovery.programMismatchStreak++;
                        }
                    } else {
                        _gpuRecovery.programMismatchProgramName = currentProgramName;
                        _gpuRecovery.programMismatchStreak = 1u;
                    }
                    double programDelay = 0.25 * (double)(1u << ((_gpuRecovery.programMismatchStreak > 6u) ? 6u : (_gpuRecovery.programMismatchStreak - 1u)));
                    if (programDelay > 20.0) {
                        programDelay = 20.0;
                    }
                    _gpuRecovery.programMismatchRetryAfter = now + programDelay;
                    if (_gpuRecovery.programMismatchStreak <= 8u || (_gpuRecovery.programMismatchStreak % 64u) == 0u) {
                        NSLog(@"MGL WARNING: Program %u mismatch breaker set for %.2fs (streak=%u)",
                        (unsigned)currentProgramName,
                        programDelay,
                        (unsigned)_gpuRecovery.programMismatchStreak);
                    }

                    // Global quarantine for this program to prevent command-buffer storm.
                    if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                        if (_gpuRecovery.interfaceMismatchBlockedStreak < UINT32_MAX) {
                            _gpuRecovery.interfaceMismatchBlockedStreak++;
                        }
                    } else {
                        _gpuRecovery.interfaceMismatchBlockedProgram = currentProgramName;
                        _gpuRecovery.interfaceMismatchBlockedStreak = 1u;
                    }
                    double quarantineDelay = retryDelay * 8.0;
                    if (quarantineDelay < 1.00) quarantineDelay = 1.00;
                    if (quarantineDelay > 15.00) quarantineDelay = 15.00;
                    _gpuRecovery.interfaceMismatchBlockedUntil = now + quarantineDelay;
                    if (_gpuRecovery.interfaceMismatchBlockedStreak <= 6u || (_gpuRecovery.interfaceMismatchBlockedStreak % 64u) == 0u) {
                        NSLog(@"MGL WARNING: Program %u quarantined for %.2fs after interface mismatch (streak=%u)",
                        (unsigned)currentProgramName,
                        quarantineDelay,
                        (unsigned)_gpuRecovery.interfaceMismatchBlockedStreak);
                    }

                    [self invalidateCurrentPipelineStateForReason:@"interface mismatch pipeline failure"];
                    _gpuRecovery.pipelineRetryAfter = (_gpuRecovery.interfaceMismatchBlockedUntil > _gpuRecovery.interfaceMismatchRetryAfter)
                    ? _gpuRecovery.interfaceMismatchBlockedUntil
                    : _gpuRecovery.interfaceMismatchRetryAfter;
                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                    return false;
                }
            }

            if (!compiledPSO &&
            MGLCapabilityHasBug(&_capability,
            MGL_BUG_MSL_PIPELINE_REJECTION)) {
                [self invalidateCurrentPipelineStateForReason:@"pipeline creation failure"];

                // AGX VIRTUALIZATION FALLBACK: Try with minimal state
                @try {
                    NSLog(@"MGL INFO: VIRTUALIZED AGX - Trying simplified compilation fallback...");

                    // Simplify the state to avoid complex shader compilation issues
                    // （与 simpleDescriptor 一致：只保留 color0/depth/stencil 格式、
                    // 顶点布局、functions、rasterization 与 tess 字段）。
                    MGLRenderCppPipelineDescriptorState simpleState = finalState;
                    simpleState.blending_enabled_mask = 0;
                    simpleState.alpha_to_coverage_enabled = 0;
                    simpleState.alpha_to_one_enabled = 0;
                    simpleState.raster_sample_count = 0;   /* C++ builder 默认 1 */
                    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                        simpleState.color_write_mask[i] = 0;
                        simpleState.source_rgb_blend_factor[i] = 0;
                        simpleState.destination_rgb_blend_factor[i] = 0;
                        simpleState.source_alpha_blend_factor[i] = 0;
                        simpleState.destination_alpha_blend_factor[i] = 0;
                        simpleState.rgb_blend_operation[i] = 0;
                        simpleState.alpha_blend_operation[i] = 0;
                        if (i > 0) {
                            simpleState.color_format[i] = (uint32_t)MTLPixelFormatInvalid;
                        }
                    }
                    psoPtr = NULL;
                    cppError[0] = '\0';
                    if (mglRenderCppCreateRenderPipelineFromState(
                            (__bridge void *)vertexFunction,
                            fragmentFunction ? (__bridge void *)fragmentFunction : NULL,
                            &simpleState, binaryArchive, &psoPtr,
                            cppError, sizeof(cppError)) == 0 && psoPtr) {
                        compiledPSO = (__bridge_transfer id<MTLRenderPipelineState>)psoPtr;
                    }
                    if (compiledPSO) {
                        mglMetalCountCreate(MGLMetalKindPSO);
                        successfulState = simpleState;
                        haveSuccessfulState = YES;
                        builtColor0Format = simpleState.color_format[0];
                        builtDepthFormat = simpleState.depth_format;
                        builtStencilFormat = simpleState.stencil_format;
                    }
                } @catch (NSException *innerException) {
                    NSLog(@"MGL ERROR: VIRTUALIZED AGX - Simplified compilation also failed: %@", innerException);
                }
            }
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Metal pipeline creation crashed: %@", exception);
        NSLog(@"MGL CRITICAL: Exception name: %@", [exception name]);
        NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);

        BOOL forceSafeFallback =
            mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE");
        if (!MGLCapabilityHasBug(&_capability,
        MGL_BUG_MSL_PIPELINE_REJECTION) && !forceSafeFallback) {
            [self invalidateCurrentPipelineStateForReason:@"pipeline creation exception"];
            _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return false;
        }

        // VIRTUALIZED AGX ULTIMATE FALLBACK: Create minimal safe pipeline
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating ultimate fallback pipeline for virtualization safety");

        @try {
            MGLRenderCppPipelineDescriptorState safeState = {0};
            safeState.color_count = MAX_COLOR_ATTACHMENTS;
            safeState.rasterization_enabled = 1;   /* safeDescriptor 未设置，默认 YES */
            MTLPixelFormat safeColor0Format = (MTLPixelFormat)finalState.color_format[0];
            if (_renderPassManager.state && mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
                safeColor0Format = mglRenderPassColorTextureFor(_renderPassManager.state, 0).pixelFormat;
            } else if (_drawable && _drawable.texture) {
                safeColor0Format = _drawable.texture.pixelFormat;
            }
            if (safeColor0Format == MTLPixelFormatInvalid) {
                safeColor0Format = MTLPixelFormatBGRA8Unorm;
            }
            safeState.color_format[0] = (uint32_t)safeColor0Format;
            safeState.depth_format = finalState.depth_format;
            safeState.stencil_format = finalState.stencil_format;

            // Precompiled safe shaders (mgl_aux_assets table).  No runtime
            // source compilation; failures log the program/format context and
            // never retry with the source compiler.
            const MGLAuxShaderAsset *safe =
                mglAuxShaderAssetFind("safe_fallback");
            void *safeVS = NULL;
            void *safeFS = NULL;
            char libError[512] = {0};
            if (!safe || !safe->data || safe->size == 0 ||
                mglRenderCppCreateAuxFunctions(
                    safe->data, safe->size, safe->hash,
                    "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
                    &safeVS, &safeFS,
                    libError, sizeof(libError)) != 0 || !safeVS) {
                NSLog(@"MGL CRITICAL: safe fallback asset unavailable "
                      @"program=%u color0=%lu hash=0x%016llx error=%s",
                      (unsigned)currentProgramName,
                      (unsigned long)safeColor0Format,
                      safe ? (unsigned long long)safe->hash : 0ull,
                      libError[0] ? libError : "asset missing");
            } else {
                id<MTLFunction> safeVSFunction =
                    (__bridge_transfer id<MTLFunction>)safeVS;
                id<MTLFunction> safeFSFunction =
                    safeFS ? (__bridge_transfer id<MTLFunction>)safeFS : nil;
                psoPtr = NULL;
                cppError[0] = '\0';
                if (mglRenderCppCreateRenderPipelineFromState(
                        (__bridge void *)safeVSFunction,
                        safeFSFunction ? (__bridge void *)safeFSFunction : NULL,
                        &safeState, binaryArchive, &psoPtr,
                        cppError, sizeof(cppError)) == 0 && psoPtr) {
                    compiledPSO = (__bridge_transfer id<MTLRenderPipelineState>)psoPtr;
                }
            }
            if (compiledPSO) {
                mglMetalCountCreate(MGLMetalKindPSO);
                successfulState = safeState;
                haveSuccessfulState = YES;
                builtColor0Format = safeState.color_format[0];
                builtDepthFormat = safeState.depth_format;
                builtStencilFormat = safeState.stencil_format;
                NSLog(@"MGL INFO: VIRTUALIZED AGX - Safe fallback pipeline created successfully");
            }
        } @catch (NSException *fallbackException) {
            NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Even fallback pipeline failed: %@", fallbackException);
        }

        if (!compiledPSO) {
            NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - All pipeline creation attempts failed, disabling rendering");
            [self invalidateCurrentPipelineStateForReason:@"all pipeline fallbacks failed"];
            _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return false;
        }
    }

    if (!compiledPSO) {
        NSLog(@"MGL ERROR: Failed to create pipeline state: %s", cppError[0] ? cppError : "unknown error");
        NSLog(@"MGL WARNING: Skipping draw for this pipeline build failure; will retry later");
        [self invalidateCurrentPipelineStateForReason:@"pipeline state is nil after creation"];
        _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
        state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
        return false;
    } else {
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE CREATE success pipeline=%p", compiledPSO);
            NSLog(@"MGL INFO: Pipeline state created successfully");
        }
        /* Publish the compile result to the shared state under a short
         * re-acquired lock. */
        METAL_LOCK();
        if (!pipelineReusedPrevious && haveSuccessfulState) {
            // Clear interface-mismatch breaker after a real compile.
            _gpuRecovery.interfaceMismatchStreak = 0;
            _gpuRecovery.interfaceMismatchProgramName = 0;
            _gpuRecovery.interfaceMismatchColor0Format = MTLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
            [_pipelineCache activatePipelineState:compiledPSO
                                   color0Format:(MTLPixelFormat)builtColor0Format
                                    depthFormat:(MTLPixelFormat)builtDepthFormat
                                  stencilFormat:(MTLPixelFormat)builtStencilFormat
                                    programName:currentProgramName
                                 vertexFunction:vertexFunction
                               fragmentFunction:fragmentFunction];
            /* 归档 add 已由 C++ builder 完成（binaryArchive 非空时）。 */
            [self insertPipelineStateIntoCacheWithWords:pipelineCacheKeyWords
                                            pipelineSig:pipelineSig
                                             vertexSig:vertexSig
                                                  state:&successfulState
                                         vertexFunction:vertexFunction
                                       fragmentFunction:fragmentFunction
                                          stateFromCache:stateFromCache];
            if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
                _gpuRecovery.programMismatchProgramName = 0;
                _gpuRecovery.programMismatchRetryAfter = 0.0;
                _gpuRecovery.programMismatchStreak = 0u;
            }
            if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                _gpuRecovery.interfaceMismatchBlockedProgram = 0;
                _gpuRecovery.interfaceMismatchBlockedUntil = 0.0;
                _gpuRecovery.interfaceMismatchBlockedStreak = 0u;
            }
        }
        METAL_UNLOCK();
    }

    return true;
}

- (bool)buildPipelineStateOnCacheMissWithDescriptor:(MTLRenderPipelineDescriptor *)pipelineStateDescriptor
                                     vertexDescriptor:(MTLVertexDescriptor *)vertexDescriptor
                                       cacheKeyWords:(const uint64_t *)pipelineCacheKeyWords
                                         pipelineSig:(uint64_t)pipelineSig
                                          vertexSig:(uint64_t)vertexSig
                                  builtColor0Format:(MTLPixelFormat)builtColor0Format
                                   builtDepthFormat:(MTLPixelFormat)builtDepthFormat
                                 builtStencilFormat:(MTLPixelFormat)builtStencilFormat
                                       programName:(GLuint)currentProgramName
                                              now:(CFTimeInterval)now
{
    /* PSO compilation can take 10-100ms (first frame / new shader).  With the
     * state lock replaced by the GL-thread contract, the entire build runs on
     * the GL calling thread with no lock to release; the deferred reset drain
     * (mtlSwapBuffersLocked:) runs on the same thread, so no lock is needed. */
    return [self buildPipelineStateOnCacheMissUnlockedWithDescriptor:pipelineStateDescriptor
                                                    vertexDescriptor:vertexDescriptor
                                                      cacheKeyWords:pipelineCacheKeyWords
                                                        pipelineSig:pipelineSig
                                                         vertexSig:vertexSig
                                                 builtColor0Format:builtColor0Format
                                                  builtDepthFormat:builtDepthFormat
                                                builtStencilFormat:builtStencilFormat
                                                      programName:currentProgramName
                                                             now:now];
}

- (bool)buildPipelineStateOnCacheMissUnlockedWithDescriptor:(MTLRenderPipelineDescriptor *)pipelineStateDescriptor
                                            vertexDescriptor:(MTLVertexDescriptor *)vertexDescriptor
                                              cacheKeyWords:(const uint64_t *)pipelineCacheKeyWords
                                                pipelineSig:(uint64_t)pipelineSig
                                                 vertexSig:(uint64_t)vertexSig
                                         builtColor0Format:(MTLPixelFormat)builtColor0Format
                                          builtDepthFormat:(MTLPixelFormat)builtDepthFormat
                                        builtStencilFormat:(MTLPixelFormat)builtStencilFormat
                                              programName:(GLuint)currentProgramName
                                                     now:(CFTimeInterval)now
{
    GLMState *state = MGL_STATE(ctx);
    Program *currentProgram = mglResolveProgramFromState(ctx);
    Program *currentVertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *currentFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    VertexArray *currentVAO = state->vao;
    Framebuffer *currentFBO = mglRendererGetValidatedFramebuffer(ctx, "buildPipelineCacheOnCacheMiss.currentFBO");
    GLuint currentFBOName = currentFBO ? currentFBO->name : 0;

    /* Two-level descriptor caching */
    MTLRenderPipelineDescriptor *finalDescriptor = pipelineStateDescriptor;
    BOOL descriptorFromCache = NO;

    /* Check descriptor cache on PSO miss; cache new descriptors for reuse. */
    if (pipelineCacheKeyWords) {
        MTLRenderPipelineDescriptor *cachedDescriptor =
        [_pipelineCache pipelineDescriptorForWords:pipelineCacheKeyWords];
        if (cachedDescriptor) {
            /* Descriptor cache hit - reuse cached descriptor instead of regenerating */
            finalDescriptor = cachedDescriptor;
            descriptorFromCache = YES;

            /* Update vertex descriptor (must be set fresh each time) */
            finalDescriptor.vertexDescriptor = vertexDescriptor;
            static uint64_t s_descriptorCacheHitCount = 0;
            s_descriptorCacheHitCount++;
            if (kMGLVerbosePipelineLogs && s_descriptorCacheHitCount <= 64ull) {
                NSLog(@"MGL DESCRIPTOR CACHE hit program=%u key=%@ (total %llu)",
                (unsigned)currentProgramName,
                [NSString stringWithFormat:@"%016llx/%016llx/%016llx",
                 (unsigned long long)pipelineCacheKeyWords[0],
                 (unsigned long long)pipelineCacheKeyWords[5],
                 (unsigned long long)pipelineCacheKeyWords[6]],
                (unsigned long long)s_descriptorCacheHitCount);
            }
        }
    }

    MGL_PERF_INC(g_mglPipelineCacheMissesSinceSwap);
    NSError *error;
    MTLRenderPipelineDescriptor *successfulDescriptor = nil;
    id<MTLRenderPipelineState> previousPipelineState = _pipelineCache.state->pipelineState;
    /* Compilation result lives in a local while this function runs without
     * holding the state lock; it is published back to _pipelineCache.state
     * at the bottom.  Nothing shared is written before the publish, so
     * resetMetalState cannot race this code (same GL thread). */
    id<MTLRenderPipelineState> compiledPSO = nil;
    bool pipelineReusedPrevious = false;

    @try {
        static uint64_t s_pipelineCreateBeginCount = 0;
        s_pipelineCreateBeginCount++;
        if (kMGLVerbosePipelineLogs &&
        (s_pipelineCreateBeginCount <= 128ull || (s_pipelineCreateBeginCount % 500ull) == 0ull)) {
            NSLog(@"MGL PIPELINE CREATE begin program=%u vao=%p fbo=%u",
            (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName);
        }

        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL INFO: Creating Metal pipeline state with AGX virtualization compatibility...");
        }

        // ROOT CAUSE FIX: The issue is with async shader compilation in virtualized environments
        // Force synchronous pipeline creation to avoid completion queue crashes
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL INFO: Using synchronous pipeline creation to prevent virtualization crashes");
        }

        /* Test hook (air_pipeline_safe_fallback regression): force the
         * pipeline-creation exception so the safe-fallback branch below is
         * exercised deterministically.  Reads the env var uncached (getenv)
         * so an in-process regression can toggle it between draws. */
        if (mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE")) {
            NSLog(@"MGL TEST: forcing safe-fallback pipeline path");
            @throw [NSException exceptionWithName:@"MGLForcedSafeFallback"
                                           reason:@"synthetic pipeline creation failure (test hook)"
                                         userInfo:nil];
        }

        // PROPER FIX: Disable async compilation that causes completion queue crashes
        if (kMGLVerbosePipelineLogs &&
        MGLCapabilityHasBug(&_capability, MGL_BUG_ASYNC_SHADER_COMPILE_IN_VM)) {
            NSLog(@"MGL INFO: AGX virtualization detected - using safe synchronous compilation");
        }

        [_pipelineCache applyBinaryArchiveToDescriptor:finalDescriptor];
        /* P4.2: gate-on 走 buildPipelineStateOnCacheMissWithState:（value-state
         * builder）；本方法只服务 gate-off 的 ObjC descriptor 路径。 */
        compiledPSO = mglRenderPassCreatePipeline(
            _device, finalDescriptor, &error);
        if (compiledPSO) {
            mglMetalCountCreate(MGLMetalKindPSO);
            successfulDescriptor = finalDescriptor;
        }

        if (!compiledPSO) {
            NSLog(@"MGL PIPELINE CREATE fail error=%@", error);
            NSLog(@"MGL ERROR: Pipeline creation failed: %@", error);

            NSString *errDesc = error.localizedDescription ?: @"";
            NSString *errDomain = error.domain ?: @"";
            BOOL isInterfaceMismatch = ((error.code == 3 && [errDomain hasPrefix:@"AGXMetal"]) ||
            [errDesc containsString:@"mismatching vertex shader output"] ||
            [errDesc containsString:@"not written by vertex shader"]);

            if (isInterfaceMismatch) {
                mglWriteProgramMSLDump(currentVertexProgram, errDesc);
                if (currentFragmentProgram && currentFragmentProgram != currentVertexProgram) {
                    mglWriteProgramMSLDump(currentFragmentProgram, errDesc);
                } else if (!currentVertexProgram) {
                    mglWriteProgramMSLDump(currentProgram, errDesc);
                }
                BOOL sameProgram =
                (_pipelineCache.state->pipelineProgramName != 0 &&
                _pipelineCache.state->pipelineProgramName == currentProgramName &&
                _pipelineCache.state->pipelineVertexFunction == pipelineStateDescriptor.vertexFunction &&
                _pipelineCache.state->pipelineFragmentFunction == pipelineStateDescriptor.fragmentFunction);
                BOOL colorCompatible = (_pipelineCache.state->pipelineColor0Format == MTLPixelFormatInvalid ||
                builtColor0Format == MTLPixelFormatInvalid ||
                _pipelineCache.state->pipelineColor0Format == builtColor0Format);
                BOOL depthCompatible = (_pipelineCache.state->pipelineDepthFormat == MTLPixelFormatInvalid ||
                builtDepthFormat == MTLPixelFormatInvalid ||
                _pipelineCache.state->pipelineDepthFormat == builtDepthFormat);
                BOOL stencilCompatible = (_pipelineCache.state->pipelineStencilFormat == MTLPixelFormatInvalid ||
                builtStencilFormat == MTLPixelFormatInvalid ||
                _pipelineCache.state->pipelineStencilFormat == builtStencilFormat);

                if (previousPipelineState && sameProgram && colorCompatible && depthCompatible && stencilCompatible) {
                    NSLog(@"MGL WARNING: Interface mismatch for program %u; reusing previous compatible pipeline once",
                    (unsigned)currentProgramName);
                    compiledPSO = previousPipelineState;
                    pipelineReusedPrevious = true;
                    _gpuRecovery.interfaceMismatchProgramName = currentProgramName;
                    _gpuRecovery.interfaceMismatchColor0Format = builtColor0Format;
                    _gpuRecovery.interfaceMismatchDepthFormat = builtDepthFormat;
                    _gpuRecovery.interfaceMismatchStencilFormat = builtStencilFormat;
                    _gpuRecovery.interfaceMismatchStreak = 1u;
                    _gpuRecovery.interfaceMismatchRetryAfter = now + 0.10;
                    _gpuRecovery.pipelineRetryAfter = _gpuRecovery.interfaceMismatchRetryAfter;
                } else {
                    BOOL sameMismatchSignature =
                    (currentProgramName == _gpuRecovery.interfaceMismatchProgramName &&
                    builtColor0Format == _gpuRecovery.interfaceMismatchColor0Format &&
                    builtDepthFormat == _gpuRecovery.interfaceMismatchDepthFormat &&
                    builtStencilFormat == _gpuRecovery.interfaceMismatchStencilFormat);
                    if (sameMismatchSignature) {
                        if (_gpuRecovery.interfaceMismatchStreak < UINT32_MAX) {
                            _gpuRecovery.interfaceMismatchStreak++;
                        }
                    } else {
                        _gpuRecovery.interfaceMismatchStreak = 1;
                        _gpuRecovery.interfaceMismatchProgramName = currentProgramName;
                        _gpuRecovery.interfaceMismatchColor0Format = builtColor0Format;
                        _gpuRecovery.interfaceMismatchDepthFormat = builtDepthFormat;
                        _gpuRecovery.interfaceMismatchStencilFormat = builtStencilFormat;
                    }

                    // Exponential backoff: 0.10, 0.20, 0.40, 0.80, 1.60, capped at 2.00 sec.
                    uint32_t cappedShift = (_gpuRecovery.interfaceMismatchStreak > 5u) ? 4u : (_gpuRecovery.interfaceMismatchStreak - 1u);
                    double retryDelay = 0.10 * (double)(1u << cappedShift);
                    if (retryDelay > 2.0) {
                        retryDelay = 2.0;
                    }
                    _gpuRecovery.interfaceMismatchRetryAfter = now + retryDelay;

                    if (_gpuRecovery.interfaceMismatchStreak <= 5u || (_gpuRecovery.interfaceMismatchStreak % 200u) == 0u) {
                        NSLog(@"MGL WARNING: Interface mismatch (program=%u, streak=%u), throttling retries for %.2fs",
                        (unsigned)currentProgramName,
                        (unsigned)_gpuRecovery.interfaceMismatchStreak,
                        retryDelay);
                    }

                    // Program-level breaker update (ignores attachment signature).
                    if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
                        if (_gpuRecovery.programMismatchStreak < UINT32_MAX) {
                            _gpuRecovery.programMismatchStreak++;
                        }
                    } else {
                        _gpuRecovery.programMismatchProgramName = currentProgramName;
                        _gpuRecovery.programMismatchStreak = 1u;
                    }
                    double programDelay = 0.25 * (double)(1u << ((_gpuRecovery.programMismatchStreak > 6u) ? 6u : (_gpuRecovery.programMismatchStreak - 1u)));
                    if (programDelay > 20.0) {
                        programDelay = 20.0;
                    }
                    _gpuRecovery.programMismatchRetryAfter = now + programDelay;
                    if (_gpuRecovery.programMismatchStreak <= 8u || (_gpuRecovery.programMismatchStreak % 64u) == 0u) {
                        NSLog(@"MGL WARNING: Program %u mismatch breaker set for %.2fs (streak=%u)",
                        (unsigned)currentProgramName,
                        programDelay,
                        (unsigned)_gpuRecovery.programMismatchStreak);
                    }

                    // Global quarantine for this program to prevent command-buffer storm.
                    if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                        if (_gpuRecovery.interfaceMismatchBlockedStreak < UINT32_MAX) {
                            _gpuRecovery.interfaceMismatchBlockedStreak++;
                        }
                    } else {
                        _gpuRecovery.interfaceMismatchBlockedProgram = currentProgramName;
                        _gpuRecovery.interfaceMismatchBlockedStreak = 1u;
                    }
                    // Use a stronger quarantine window than compile retry backoff.
                    // This prevents pathological draw loops from repeatedly re-entering
                    // pipeline compilation and overwhelming AGX command submission.
                    double quarantineDelay = retryDelay * 8.0;
                    if (quarantineDelay < 1.00) quarantineDelay = 1.00;
                    if (quarantineDelay > 15.00) quarantineDelay = 15.00;
                    _gpuRecovery.interfaceMismatchBlockedUntil = now + quarantineDelay;
                    if (_gpuRecovery.interfaceMismatchBlockedStreak <= 6u || (_gpuRecovery.interfaceMismatchBlockedStreak % 64u) == 0u) {
                        NSLog(@"MGL WARNING: Program %u quarantined for %.2fs after interface mismatch (streak=%u)",
                        (unsigned)currentProgramName,
                        quarantineDelay,
                        (unsigned)_gpuRecovery.interfaceMismatchBlockedStreak);
                    }

                    [self invalidateCurrentPipelineStateForReason:@"interface mismatch pipeline failure"];
                    _gpuRecovery.pipelineRetryAfter = (_gpuRecovery.interfaceMismatchBlockedUntil > _gpuRecovery.interfaceMismatchRetryAfter)
                    ? _gpuRecovery.interfaceMismatchBlockedUntil
                    : _gpuRecovery.interfaceMismatchRetryAfter;
                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                    return false;
                }
            }

            if (!compiledPSO &&
            MGLCapabilityHasBug(&_capability,
            MGL_BUG_MSL_PIPELINE_REJECTION)) {
                [self invalidateCurrentPipelineStateForReason:@"pipeline creation failure"];
                // Avoid destructive global recovery during shader/pipeline compile errors.
                // These are usually content/interface issues, not GPU-state corruption.

                // AGX VIRTUALIZATION FALLBACK: Try with minimal descriptor
                @try {
                    NSLog(@"MGL INFO: VIRTUALIZED AGX - Trying simplified compilation fallback...");

                    // Simplify the descriptor to avoid complex shader compilation issues
                    MTLRenderPipelineDescriptor *simpleDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
                    simpleDescriptor.colorAttachments[0].pixelFormat = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
                    simpleDescriptor.depthAttachmentPixelFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
                    simpleDescriptor.stencilAttachmentPixelFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;
                    simpleDescriptor.vertexDescriptor = pipelineStateDescriptor.vertexDescriptor;
                    simpleDescriptor.vertexFunction = pipelineStateDescriptor.vertexFunction;
                    simpleDescriptor.fragmentFunction = pipelineStateDescriptor.fragmentFunction;
                    simpleDescriptor.rasterizationEnabled = pipelineStateDescriptor.rasterizationEnabled;
                    if (_tessellation.nativeTESActive) {
                        simpleDescriptor.tessellationPartitionMode = pipelineStateDescriptor.tessellationPartitionMode;
                        simpleDescriptor.maxTessellationFactor = pipelineStateDescriptor.maxTessellationFactor;
                        simpleDescriptor.tessellationFactorScaleEnabled = pipelineStateDescriptor.tessellationFactorScaleEnabled;
                        simpleDescriptor.tessellationFactorFormat = pipelineStateDescriptor.tessellationFactorFormat;
                        simpleDescriptor.tessellationControlPointIndexType = pipelineStateDescriptor.tessellationControlPointIndexType;
                        simpleDescriptor.tessellationFactorStepFunction = pipelineStateDescriptor.tessellationFactorStepFunction;
                        simpleDescriptor.tessellationOutputWindingOrder = pipelineStateDescriptor.tessellationOutputWindingOrder;
                    }
                    mglNormalizePipelineDepthStencilFormats(simpleDescriptor, "simple-fallback");
                    mglEnableIndirectCommandBuffersForPipeline(simpleDescriptor);

                    [_pipelineCache applyBinaryArchiveToDescriptor:simpleDescriptor];
                    compiledPSO = mglRenderPassCreatePipeline(
                        _device, simpleDescriptor, &error);
                    if (compiledPSO) {
                        mglMetalCountCreate(MGLMetalKindPSO);
                        successfulDescriptor = simpleDescriptor;
                        builtColor0Format = simpleDescriptor.colorAttachments[0].pixelFormat;
                        builtDepthFormat = simpleDescriptor.depthAttachmentPixelFormat;
                        builtStencilFormat = simpleDescriptor.stencilAttachmentPixelFormat;
                    }
                } @catch (NSException *innerException) {
                    NSLog(@"MGL ERROR: VIRTUALIZED AGX - Simplified compilation also failed: %@", innerException);
                }
            }
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Metal pipeline creation crashed: %@", exception);
        NSLog(@"MGL CRITICAL: Exception name: %@", [exception name]);
        NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);

        BOOL forceSafeFallback =
            mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE");
        if (!MGLCapabilityHasBug(&_capability,
        MGL_BUG_MSL_PIPELINE_REJECTION) && !forceSafeFallback) {
            [self invalidateCurrentPipelineStateForReason:@"pipeline creation exception"];
            _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return false;
        }

        // VIRTUALIZED AGX ULTIMATE FALLBACK: Create minimal safe pipeline
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating ultimate fallback pipeline for virtualization safety");

        @try {
            MTLRenderPipelineDescriptor *safeDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
            MTLPixelFormat safeColor0Format = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
            if (_renderPassManager.state && mglRenderPassColorTextureFor(_renderPassManager.state, 0)) {
                safeColor0Format = mglRenderPassColorTextureFor(_renderPassManager.state, 0).pixelFormat;
            } else if (_drawable && _drawable.texture) {
                safeColor0Format = _drawable.texture.pixelFormat;
            }
            if (safeColor0Format == MTLPixelFormatInvalid) {
                safeColor0Format = MTLPixelFormatBGRA8Unorm;
            }
            safeDescriptor.colorAttachments[0].pixelFormat = safeColor0Format;
            safeDescriptor.depthAttachmentPixelFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
            safeDescriptor.stencilAttachmentPixelFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;
            safeDescriptor.colorAttachments[0].blendingEnabled = NO;
            mglNormalizePipelineDepthStencilFormats(safeDescriptor, "safe-fallback");
            mglEnableIndirectCommandBuffersForPipeline(safeDescriptor);

            // Precompiled safe shaders (mgl_aux_assets table).  No runtime
            // source compilation; failures log the program/format context and
            // never retry with the source compiler.
            if (mglRenderPassUsesMetalCpp()) {
                const MGLAuxShaderAsset *safe =
                    mglAuxShaderAssetFind("safe_fallback");
                void *safeVS = NULL;
                void *safeFS = NULL;
                char libError[512] = {0};
                if (!safe || !safe->data || safe->size == 0 ||
                    mglRenderCppCreateAuxFunctions(
                        safe->data, safe->size, safe->hash,
                        "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
                        &safeVS, &safeFS,
                        libError, sizeof(libError)) != 0 || !safeVS) {
                    NSLog(@"MGL CRITICAL: safe fallback asset unavailable "
                          @"program=%u color0=%lu hash=0x%016llx error=%s",
                          (unsigned)currentProgramName,
                          (unsigned long)safeColor0Format,
                          safe ? (unsigned long long)safe->hash : 0ull,
                          libError[0] ? libError : "asset missing");
                } else {
                    safeDescriptor.vertexFunction =
                        (__bridge_transfer id<MTLFunction>)safeVS;
                    safeDescriptor.fragmentFunction =
                        safeFS ? (__bridge_transfer id<MTLFunction>)safeFS : nil;
                    [_pipelineCache applyBinaryArchiveToDescriptor:safeDescriptor];
                    compiledPSO = mglRenderPassCreatePipeline(
                        _device, safeDescriptor, &error);
                }
            } else {
                NSError *libraryError = nil;
                id<MTLLibrary> safeLibrary =
                    mglRenderPassLoadAuxLibrary(_device, "safe_fallback",
                                                &libraryError);
                if (!safeLibrary) {
                    NSLog(@"MGL CRITICAL: safe fallback asset load failed "
                          @"program=%u color0=%lu error=%@",
                          (unsigned)currentProgramName,
                          (unsigned long)safeColor0Format, libraryError);
                } else {
                    safeDescriptor.vertexFunction =
                        mglRenderPassCreateFunction(
                            safeLibrary, @"mgl_safe_fallback_vs", NULL, 0);
                    safeDescriptor.fragmentFunction =
                        mglRenderPassCreateFunction(
                            safeLibrary, @"mgl_safe_fallback_fs", NULL, 0);
                    if (safeDescriptor.vertexFunction && safeDescriptor.fragmentFunction) {
                        [_pipelineCache applyBinaryArchiveToDescriptor:safeDescriptor];
                        compiledPSO = mglRenderPassCreatePipeline(
                            _device, safeDescriptor, &error);
                    }
                }
            }
            if (compiledPSO) {
                mglMetalCountCreate(MGLMetalKindPSO);
                successfulDescriptor = safeDescriptor;
                builtColor0Format = safeDescriptor.colorAttachments[0].pixelFormat;
                builtDepthFormat = safeDescriptor.depthAttachmentPixelFormat;
                builtStencilFormat = safeDescriptor.stencilAttachmentPixelFormat;
                NSLog(@"MGL INFO: VIRTUALIZED AGX - Safe fallback pipeline created successfully");
            }
        } @catch (NSException *fallbackException) {
            NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Even fallback pipeline failed: %@", fallbackException);
        }

        if (!compiledPSO) {
            NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - All pipeline creation attempts failed, disabling rendering");
            [self invalidateCurrentPipelineStateForReason:@"all pipeline fallbacks failed"];
            _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return false;
        }
    }

    // Pipeline State creation could fail if the pipeline descriptor isn't set up properly.
    //  If the Metal API validation is enabled, you can find out more information about what
    //  went wrong.  (Metal API validation is enabled by default when a debug build is run
    //  from Xcode.)
    if (!compiledPSO) {
        NSLog(@"MGL ERROR: Failed to create pipeline state: %@", error);
        NSLog(@"MGL WARNING: Skipping draw for this pipeline build failure; will retry later");
        [self invalidateCurrentPipelineStateForReason:@"pipeline state is nil after creation"];
        _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
        state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
        return false;
    } else {
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE CREATE success pipeline=%p", compiledPSO);
            NSLog(@"MGL INFO: Pipeline state created successfully");
        }
        /* Publish the compile result to the shared state under a short
         * re-acquired lock, so a concurrent resetMetalState can never race
         * the pipelineState pointer or the pipeline cache dictionary. */
        METAL_LOCK();
        if (!pipelineReusedPrevious && successfulDescriptor) {
            // Clear interface-mismatch breaker after a real compile.
            _gpuRecovery.interfaceMismatchStreak = 0;
            _gpuRecovery.interfaceMismatchProgramName = 0;
            _gpuRecovery.interfaceMismatchColor0Format = MTLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
            [_pipelineCache activatePipelineState:compiledPSO
            color0Format:builtColor0Format
            depthFormat:builtDepthFormat
            stencilFormat:builtStencilFormat
            programName:currentProgramName
            vertexFunction:successfulDescriptor.vertexFunction
            fragmentFunction:successfulDescriptor.fragmentFunction];
            [_pipelineCache addPipelineToBinaryArchive:successfulDescriptor];
            if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
                _gpuRecovery.programMismatchProgramName = 0;
                _gpuRecovery.programMismatchRetryAfter = 0.0;
                _gpuRecovery.programMismatchStreak = 0u;
            }
            if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                _gpuRecovery.interfaceMismatchBlockedProgram = 0;
                _gpuRecovery.interfaceMismatchBlockedUntil = 0.0;
                _gpuRecovery.interfaceMismatchBlockedStreak = 0u;
            }

            [self insertPipelineIntoCacheWithWords:pipelineCacheKeyWords
            pipelineSig:pipelineSig
            vertexSig:vertexSig
            descriptor:successfulDescriptor
            descriptorFromCache:(descriptorFromCache &&
            successfulDescriptor == finalDescriptor)];
        }
        METAL_UNLOCK();
    }

    return true;
}

/*
 * Pipeline cache insertion with LRU eviction, extracted from
 * syncPipelineStateWithDeferredBufferMap:.
 */
- (void)insertPipelineIntoCacheWithWords:(const uint64_t *)pipelineCacheKeyWords
                           pipelineSig:(uint64_t)pipelineSig
                             vertexSig:(uint64_t)vertexSig
                            descriptor:(MTLRenderPipelineDescriptor *)descriptor
                    descriptorFromCache:(BOOL)descriptorFromCache
{
    if (pipelineCacheKeyWords && _pipelineCache.state->pipelineState) {
            (void)pipelineSig;
            (void)vertexSig;
            [_pipelineCache storePipeline:_pipelineCache.state->pipelineState
                           vertexFunction:descriptor.vertexFunction
                         fragmentFunction:descriptor.fragmentFunction
                                 forWords:pipelineCacheKeyWords];

            /* Cache the descriptor for future PSO cache misses.
             * Only cache if descriptor was generated (not from cache).
             * This avoids redundant descriptor generation on next miss. */
            if (!descriptorFromCache && descriptor) {
                [_pipelineCache storePipelineDescriptor:descriptor
                                               forWords:pipelineCacheKeyWords];
            }
    }
}

/* P4.2 gate-on：insertPipelineIntoCacheWithWords 的 value-state 版。 */
- (void)insertPipelineStateIntoCacheWithWords:(const uint64_t *)pipelineCacheKeyWords
                                  pipelineSig:(uint64_t)pipelineSig
                                   vertexSig:(uint64_t)vertexSig
                                        state:(const MGLRenderCppPipelineDescriptorState *)state
                               vertexFunction:(id<MTLFunction>)vertexFunction
                             fragmentFunction:(id<MTLFunction>)fragmentFunction
                                stateFromCache:(BOOL)stateFromCache
{
    if (pipelineCacheKeyWords && _pipelineCache.state->pipelineState) {
            (void)pipelineSig;
            (void)vertexSig;
            [_pipelineCache storePipeline:_pipelineCache.state->pipelineState
                           vertexFunction:vertexFunction
                         fragmentFunction:fragmentFunction
                                 forWords:pipelineCacheKeyWords];

            /* Cache the descriptor state for future PSO cache misses.
             * Only cache if state was generated (not from cache). */
            if (!stateFromCache && state) {
                [_pipelineCache storePipelineDescriptorState:state
                                                    forWords:pipelineCacheKeyWords];
            }
    }
}

/* Bind spvBufferSizeConstants for runtime-sized SSBO arrays in vertex/fragment
 * stages.  The AIR backend emits code that reads uint32 byte-sizes from a
 * constant uint* buffer at MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a shader uses
 * .length() on unsized SSBO arrays.  The render encoder has separate buffer
 * tables for vertex and fragment, so we bind a size buffer for each stage
 * that needs it. */
- (bool) bindBufferSizeConstantsForRenderEncoder
{
    if (!_renderPassManager.state->currentRenderEncoder) {
        return true;
    }

    Program *vertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (vertexProgram && vertexProgram->modules[_VERTEX_SHADER].needs_runtime_array_size_buffer)
    {
        uint32_t sizeConstants[31];
        memset(sizeConstants, 0, sizeof(sizeConstants));

        for (int i = 0; i < MGL_STATE(ctx)->vertex_buffer_map_list.count; i++)
        {
            BufferMap *map = &MGL_STATE(ctx)->vertex_buffer_map_list.buffers[i];
            if (!map->buf)
                continue;
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX)
                continue;
            GLsizeiptr visibleSize = mglBufferMapVisibleSize(map);
            sizeConstants[metalSlot] = (uint32_t)visibleSize;
        }

        /* Reuse the cached MTLBuffer when size constants are unchanged.  When
         * they differ we must allocate a new buffer: Metal records only a
         * reference at setVertexBuffer time and the GPU reads contents at
         * command-buffer execution, so overwriting a buffer that earlier
         * draws in the same CB reference would corrupt them. */
        if (!_vertexSizeConstantsValid ||
            memcmp(_vertexSizeConstantsCache, sizeConstants, sizeof(sizeConstants)) != 0) {
            _vertexSizeBuffer = mglRenderPassCreateBufferWithBytes(
                _device, sizeConstants, sizeof(sizeConstants),
                MTLResourceStorageModeShared);
            memcpy(_vertexSizeConstantsCache, sizeConstants, sizeof(sizeConstants));
            _vertexSizeConstantsValid = YES;
        }
        if (_vertexSizeBuffer) {
            mglRenderPassSetBuffer(
                _renderPassManager.state->currentRenderEncoder,
                _vertexSizeBuffer, 0, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            [self recordLastBoundVertexBuffer:_vertexSizeBuffer
                                       offset:0
                                      atIndex:MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX];
            MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        }
    }

    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    if (fragmentProgram && fragmentProgram->modules[_FRAGMENT_SHADER].needs_runtime_array_size_buffer)
    {
        uint32_t sizeConstants[31];
        memset(sizeConstants, 0, sizeof(sizeConstants));

        for (int i = 0; i < MGL_STATE(ctx)->fragment_buffer_map_list.count; i++)
        {
            BufferMap *map = &MGL_STATE(ctx)->fragment_buffer_map_list.buffers[i];
            if (!map->buf)
                continue;
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX)
                continue;
            GLsizeiptr visibleSize = mglBufferMapVisibleSize(map);
            sizeConstants[metalSlot] = (uint32_t)visibleSize;
        }

        /* Content-comparison cache (see vertex note above). */
        if (!_fragmentSizeConstantsValid ||
            memcmp(_fragmentSizeConstantsCache, sizeConstants, sizeof(sizeConstants)) != 0) {
            _fragmentSizeBuffer = mglRenderPassCreateBufferWithBytes(
                _device, sizeConstants, sizeof(sizeConstants),
                MTLResourceStorageModeShared);
            memcpy(_fragmentSizeConstantsCache, sizeConstants, sizeof(sizeConstants));
            _fragmentSizeConstantsValid = YES;
        }
        if (_fragmentSizeBuffer) {
            mglRenderPassSetBuffer(
                _renderPassManager.state->currentRenderEncoder,
                _fragmentSizeBuffer, 0, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT,
                MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            [self recordLastBoundFragmentBuffer:_fragmentSizeBuffer
                                         offset:0
                                        atIndex:MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX];
            MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
        }
    }

    return true;
}

-(void) flushCommandBuffer: (bool) finish
{
    METAL_LOCK();
    [self flushCommandBufferLocked:finish];
    METAL_UNLOCK();

    /* waitUntilCompleted is outside METAL_LOCK to avoid blocking other
     * threads that need the lock while the GPU finishes.  The CB was already
     * committed inside the lock; waiting outside is safe because the CB
     * retains itself until completion. */
    if (finish && _pendingFinishCB != nil) {
        @try {
            if (mglRenderCommandBufferStatus(_pendingFinishCB) !=
                MTLCommandBufferStatusNotEnqueued) {
                mglRenderPassWaitCommandBuffer(_pendingFinishCB);
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: waitUntilCompleted failed outside lock: %@", exception);
        }
        _pendingFinishCB = nil;
    }
}

-(void) flushCommandBufferLocked: (bool) finish
{
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or queue is NULL in flushCommandBuffer");
        return;
    }

    [self flushDrawBufferLocked:ctx];

    if (![self processGLStateLocked: false]) {
        NSLog(@"MGL WARNING: processGLState failed in flushCommandBuffer, continuing with cleanup");
    }

    /* If processGLStateLocked: left a render encoder active, mark the CB as
     * having work so the commit below is not skipped. */
    if (_renderPassManager.state->currentRenderEncoder) {
        _currentCBHasWork = YES;
    }

    [self endRenderEncodingLocked];

    /* Skip empty-CB commit when finish=true: wait on _lastCommittedCB
     * instead (Metal CBs execute serially on the same queue).  Any path
     * that encodes work (draws/render/blit/compute) into the current CB
     * MUST set _currentCBHasWork before calling flushCommandBuffer:YES,
     * else the skip drops uncommitted work. */
    if (finish && !_currentCBHasWork && _lastCommittedCB != nil) {
        _pendingFinishCB = _lastCommittedCB;
        return;
    }
    if (finish && !_currentCBHasWork && _lastCommittedCB == nil) {
        /* No CB was ever committed — nothing to wait for. */
        _pendingFinishCB = nil;
        return;
    }

    if (![self ensureWritableCommandBufferLocked:"flushCommandBuffer"]) {
        NSLog(@"MGL ERROR: Unable to obtain writable command buffer in flushCommandBuffer");
        return;
    }

    if (!(__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner)) {
        NSLog(@"MGL WARNING: No current command buffer in flushCommandBuffer");
        return;
    }

    MTLCommandBufferStatus currentStatus =
        mglRenderCommandBufferStatus(
            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
    if (currentStatus != MTLCommandBufferStatusNotEnqueued) {
        NSLog(@"MGL INFO: flushCommandBuffer found finalized buffer (status=%ld), rotating", (long)currentStatus);
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer in flushCommandBuffer");
        }
        return;
    }

    MGLRenderCppCommandBufferState preCommitState =
        mglRenderCommandBufferState(
            (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(_renderPassManager.state->currentCommandBufferOwner));
    if (preCommitState.has_error) {
        NSLog(@"MGL ERROR: Command buffer has error before commit: %@",
              mglRenderCommandBufferErrorString(&preCommitState));
        [self cleanupCommandBuffer];
        return;
    }

    if (![self validateMetalObjects]) {
        NSLog(@"MGL WARNING: GPU throttling active - skipping command buffer commit");
        [self cleanupCommandBuffer];
        return;
    }

    id<MTLCommandBuffer> commandBufferToCommit =
        [_renderPassManager detachCurrentCommandBufferForSubmission];

    @try {
        [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
        _lastCommittedCB = commandBufferToCommit;
        /* Don't waitUntilCompleted inside the lock — save the CB for the
         * caller (flushCommandBuffer:) to wait on after unlock. */
        if (finish) {
            _pendingFinishCB = commandBufferToCommit;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Command buffer commit failed in flushCommandBuffer: %@", exception);
        [self recordGPUError];
        [self cleanupCommandBuffer];
    }

    if (!finish) {
        [self newCommandBufferLocked];
    }
}

- (bool)syncRenderPassStateForContext:(GLMContext)glm_ctx
{
    GLMState *state = MGL_STATE(glm_ctx);
    Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(glm_ctx, "processGLState.dirtyFBO");
    BOOL framebufferBindingDirty = framebuffer && (framebuffer->dirty_bits & DIRTY_FBO_BINDING);
    if (_renderPassManager.state->currentRenderEncoder &&
        !framebufferBindingDirty &&
        [self currentRenderPassMatchesCurrentFramebuffer]) {
        state->dirty_bits &= ~DIRTY_FBO;
        return true;
    }

    if (framebuffer && framebufferBindingDirty)
    {
        RETURN_FALSE_ON_FAILURE([self bindFramebufferAttachmentTextures]);
        framebuffer = mglRendererGetValidatedFramebuffer(glm_ctx, "processGLState.dirtyFBO.afterBind");
        if (framebuffer) {
            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
        }
    }

    /* instrumentation: an FBO change forced a real encoder rotation
     * (the "already matches" fast path above returned early without counting).
     * newRenderEncoderLocked also bumps g_mglEncoderCreationsSinceSwap, so
     * fboRot <= new always holds; new-minus-fboRot is non-FBO creation. */
    /* RenderPass Manager: encoder open/close is owned by the RenderPass Manager
     * facade (rotateRenderEncoderForCurrentFramebufferLocked), not by this
     * Sync unit directly. The Sync layer only decides that a rotation is
     * needed and delegates the lifecycle transition. */
    RETURN_FALSE_ON_FAILURE([self rotateRenderEncoderForCurrentFramebufferLocked]);
    return true;
}

/*
 * RenderPass Manager — RenderPass Manager facade: single owner of the FBO-driven
 * encoder rotation (the primary open/close transition). Ends the current
 * render encoder and opens a fresh one against the now-bound framebuffer's
 * render-pass descriptor. Called by the RenderPass Sync unit
 * (syncRenderPassStateForContext:) when currentRenderPassMatchesCurrentFramebuffer
 * is false; the "already matches" fast path returns without rotating. The
 * recovery/nil-encoder paths elsewhere (processGLStateLocked, texture upload,
 * blit) still call newRenderEncoderLocked directly and are documented as
 * out-of-scope for this facade — they are safety nets, not the primary
 * lifecycle transition, and lifting them is deferred until a future pass
 * proves the facade is sufficient under Minecraft workloads.
 */
- (bool)rotateRenderEncoderForCurrentFramebufferLocked
{
    MGL_PERF_INC(g_mglEncoderFBORotationsSinceSwap);
    GLMContext glm_ctx = ctx;
    GLuint fbo_name = 0u;
    if (glm_ctx && glm_ctx->active_state && MGL_STATE(glm_ctx)->framebuffer) {
        fbo_name = MGL_STATE(glm_ctx)->framebuffer->name;
    }
    if (fbo_name == 0u) {
        MGL_PERF_INC(g_mglEncoderFboRotDefaultSinceSwap);
    } else {
        MGL_PERF_INC(g_mglEncoderFboRotNamedSinceSwap);
    }
    [self endRenderEncodingLocked];
    RETURN_FALSE_ON_FAILURE(
        [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_FBO]);
    return true;
}

- (BOOL)prepareRenderPassIfFBOChanged:(MGLDrawBatch *)batch
                              context:(GLMContext)glm_ctx
                          replayError:(GLenum *)replayError
{
    if (!(MGL_STATE(glm_ctx)->dirty_bits & DIRTY_FBO))
        return YES;

    /* Orchestrator-driven FBO rotation (Orchestrator-driven FBO rotation) delegates to the shared
     * RenderPass Sync unit (RenderPass Sync domain), surfacing any GL error as replayError
     * so the batch is skipped rather than drawn against a stale pass. */
    if (![self syncRenderPassStateForContext:glm_ctx]) {
        if (MGL_STATE(glm_ctx)->error != GL_NO_ERROR)
            *replayError = MGL_STATE(glm_ctx)->error;
        return NO;
    }
    return YES;
}

@end
