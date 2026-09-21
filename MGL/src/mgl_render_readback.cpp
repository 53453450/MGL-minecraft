/* SPDX-License-Identifier: LGPL-3.0-only */
#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_air_loader.h"
#include "mgl_buffer_slots.h"
#include "mgl_renderer_backend.h"
#include "glm_context.h"
#include "mgl_render_internal.h"

extern "C"
uint32_t mglRenderReadbackBytesPerPixel(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatRGBA32Float:
            return (uint32_t)(sizeof(float) * 4u);
        case MTL::PixelFormatR8Unorm:
            return 1u;
        case MTL::PixelFormatR16Unorm:
        case MTL::PixelFormatR16Snorm:
        case MTL::PixelFormatRG8Unorm:
        case MTL::PixelFormatABGR4Unorm:
        case MTL::PixelFormatBGR5A1Unorm:
        case MTL::PixelFormatR16Float:
            return 2u;
        case MTL::PixelFormatRG32Float:
        case MTL::PixelFormatRGBA16Float:
        case MTL::PixelFormatRGBA16Unorm:
        case MTL::PixelFormatRGBA16Snorm:
            return 8u;
        case MTL::PixelFormatR8Snorm:
        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatR8Sint:
            return 1u;
        case MTL::PixelFormatRG8Snorm:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRG8Sint:
            return 2u;
        case MTL::PixelFormatRG16Unorm:
        case MTL::PixelFormatRG16Snorm:
        case MTL::PixelFormatRG16Float:
        case MTL::PixelFormatRGBA8Snorm:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatRGBA8Sint:
        default:
            return 4u;
    }
}

extern "C"
int mglRenderReadbackFormatIsBGRA8Compatible(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatBGRA8Unorm:
        case MTL::PixelFormatBGRA8Unorm_sRGB:
        case MTL::PixelFormatRGBA8Unorm:
        case MTL::PixelFormatRGBA8Unorm_sRGB:
        case MTL::PixelFormatRGBA32Float:
        case MTL::PixelFormatR8Unorm:
        case MTL::PixelFormatRG8Unorm:
        case MTL::PixelFormatR16Unorm:
        case MTL::PixelFormatR16Snorm:
        case MTL::PixelFormatRG16Unorm:
        case MTL::PixelFormatRG16Snorm:
        case MTL::PixelFormatRGBA16Unorm:
        case MTL::PixelFormatRGBA16Snorm:
        case MTL::PixelFormatABGR4Unorm:
        case MTL::PixelFormatBGR5A1Unorm:
        case MTL::PixelFormatRG11B10Float:
        case MTL::PixelFormatR32Float:
        case MTL::PixelFormatRG32Float:
        case MTL::PixelFormatRG16Float:
        case MTL::PixelFormatR16Float:
        case MTL::PixelFormatRGBA16Float:
        case MTL::PixelFormatBGR10A2Unorm:
        case MTL::PixelFormatRGB10A2Unorm:
        case MTL::PixelFormatR8Snorm:
        case MTL::PixelFormatRG8Snorm:
        case MTL::PixelFormatRGBA8Snorm:
        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatRGB9E5Float:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderReadbackGLTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_FLOAT:
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

int mglRenderReadbackTypeIsCore(uint32_t type) {
    return type == GL_UNSIGNED_BYTE || type == GL_BYTE ||
                   type == GL_UNSIGNED_SHORT || type == GL_SHORT ||
                   type == GL_UNSIGNED_INT || type == GL_INT ||
                   type == GL_FLOAT || type == GL_HALF_FLOAT
               ? 1
               : 0;
}

int mglRenderReadbackTypeAllowsRGB10A2(uint32_t type) {
    return mglRenderReadbackTypeIsCore(type) ||
                   type == GL_UNSIGNED_INT_10_10_10_2 ||
                   type == GL_UNSIGNED_INT_2_10_10_10_REV ||
                   type == GL_UNSIGNED_INT_5_9_9_9_REV ||
                   type == GL_UNSIGNED_INT_8_8_8_8 ||
                   type == GL_UNSIGNED_INT_8_8_8_8_REV
               ? 1
               : 0;
}

int mglRenderReadbackTypeAllowsRG11B10(uint32_t type) {
    return mglRenderReadbackTypeIsCore(type) ||
                   type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
                   type == GL_UNSIGNED_INT_5_9_9_9_REV ||
                   type == GL_UNSIGNED_INT_8_8_8_8 ||
                   type == GL_UNSIGNED_INT_8_8_8_8_REV
               ? 1
               : 0;
}

int mglRenderReadbackTypeAllows16or32(uint32_t type) {
    return mglRenderReadbackTypeIsCore(type) ||
                   type == GL_UNSIGNED_BYTE_3_3_2 ||
                   type == GL_UNSIGNED_BYTE_2_3_3_REV ||
                   type == GL_UNSIGNED_SHORT_5_6_5 ||
                   type == GL_UNSIGNED_SHORT_5_6_5_REV ||
                   type == GL_UNSIGNED_SHORT_4_4_4_4 ||
                   type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
                   type == GL_UNSIGNED_SHORT_5_5_5_1 ||
                   type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
                   type == GL_UNSIGNED_INT_8_8_8_8 ||
                   type == GL_UNSIGNED_INT_8_8_8_8_REV ||
                   type == GL_UNSIGNED_INT_10_10_10_2 ||
                   type == GL_UNSIGNED_INT_2_10_10_10_REV ||
                   type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
                   type == GL_UNSIGNED_INT_5_9_9_9_REV
               ? 1
               : 0;
}

int mglRenderReadbackTypeIsWideScalar(uint32_t type) {
    return type == GL_BYTE || type == GL_SHORT || type == GL_INT ||
                   type == GL_UNSIGNED_INT || type == GL_UNSIGNED_SHORT ||
                   type == GL_HALF_FLOAT || type == GL_FLOAT
               ? 1
               : 0;
}

int mglRenderReadbackTypeIsPacked(uint32_t type) {
    return mglRenderReadbackTypeAllows16or32(type) &&
                   !mglRenderReadbackTypeIsCore(type)
               ? 1
               : 0;
}

int mglRenderReadbackPixelFormatIsSnorm8(uint32_t pixel_format) {
    switch (pixel_format) {
    case 12u: /* R8Snorm */
    case 32u: /* RG8Snorm */
    case 72u: /* RGBA8Snorm */
        return 1;
    default:
        return 0;
    }
}

int mglRenderReadbackPixelFormatIsRGB10A2(uint32_t pixel_format) {
    return pixel_format == 90u /* RGB10A2Unorm */ ? 1 : 0;
}

int mglRenderReadbackPixelFormatIsRG11B10(uint32_t pixel_format) {
    return pixel_format == 92u /* RG11B10Float */ ? 1 : 0;
}

int mglRenderReadbackPixelFormatIs16or32(uint32_t pixel_format) {
    switch (pixel_format) {
    case 20u:  /* R16Unorm */
    case 22u:  /* R16Snorm */
    case 25u:  /* R16Float */
    case 55u:  /* R32Float */
    case 60u:  /* RG16Unorm */
    case 62u:  /* RG16Snorm */
    case 65u:  /* RG16Float */
    case 105u: /* RG32Float */
    case 110u: /* RGBA16Unorm */
    case 112u: /* RGBA16Snorm */
    case 115u: /* RGBA16Float */
    case 125u: /* RGBA32Float */
        return 1;
    default:
        return 0;
    }
}

int mglRenderReadbackPixelFormatIsRGBA8(uint32_t pixel_format) {
    return pixel_format == 70u /* RGBA8Unorm */ ||
                   pixel_format == 71u /* RGBA8Unorm_sRGB */
               ? 1
               : 0;
}

int mglRenderReadbackPixelFormatIsBGRA8(uint32_t pixel_format) {
    return pixel_format == 80u /* BGRA8Unorm */ ||
                   pixel_format == 81u /* BGRA8Unorm_sRGB */
               ? 1
               : 0;
}

uint32_t mglRenderReadbackBGRA8CarrierFormat(void) {
    return 80u; /* BGRA8Unorm */
}

int mglRenderClearMaskHasColor(uint32_t mask) {
    return (mask & GL_COLOR_BUFFER_BIT) != 0 ? 1 : 0;
}

int mglRenderClearMaskHasDepth(uint32_t mask) {
    return (mask & GL_DEPTH_BUFFER_BIT) != 0 ? 1 : 0;
}

int mglRenderClearMaskHasStencil(uint32_t mask) {
    return (mask & GL_STENCIL_BUFFER_BIT) != 0 ? 1 : 0;
}

int mglRenderClearMaskHasDepthStencil(uint32_t mask) {
    return mglRenderClearMaskHasDepth(mask) || mglRenderClearMaskHasStencil(mask)
               ? 1
               : 0;
}

uint32_t mglRenderClearMaskDepthStencilBits(uint32_t mask) {
    return mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

uint32_t mglRenderClearMaskClearColor(uint32_t mask) {
    return mask & ~GL_COLOR_BUFFER_BIT;
}

uint32_t mglRenderClearMaskClearDepth(uint32_t mask) {
    return mask & ~GL_DEPTH_BUFFER_BIT;
}

uint32_t mglRenderClearMaskClearStencil(uint32_t mask) {
    return mask & ~GL_STENCIL_BUFFER_BIT;
}

int mglRenderClearMaskHasAny(uint32_t mask) {
    return mglRenderClearMaskHasColor(mask) ||
                   mglRenderClearMaskHasDepthStencil(mask)
               ? 1
               : 0;
}

void mglRenderClearEmptyBufferDirty(Buffer *buf) {
    if (buf && buf->size == 0) {
        buf->data.dirty_bits &= ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
    }
}

void mglRenderClearCPUWriteRange(Buffer *buf) {
    if (!buf) {
        return;
    }
    buf->written_min = -1;
    buf->written_max = -1;
}

void mglRenderPendingEventClear(MGLPendingEventOwner * owner_handle) {
    mgl::PendingEventOwner* owner =
        reinterpret_cast<mgl::PendingEventOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    if (owner->event) {
        owner->event->release();
        owner->event = nullptr;
    }
    owner->sync_name = 0;
}

int mglRenderBindingClearFragmentTexture(MGLBindingState * binding_state, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || index >= state->fragmentTextures.size()) return -1;
    mgl::BindingState::replaceObject(
        state->fragmentTextures[index], static_cast<MTL::Texture*>(nullptr));
    return 0;
}

int mglRenderCommandRecoveryClearMode(MGLCommandBufferRecoveryOwner * owner_handle) {
    mgl::CommandBufferRecoveryOwner* owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (!owner->recoveryMode) return 0;
    owner->recoveryMode = false;
    return 1;
}

void mglRenderCommandBufferOwnerClearSyncs(MGLCommandBufferOwner * owner_handle) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    owner->syncs.reset();
}

int mglRenderEncodeColorClear(void* command_buffer,
                                 void* texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double red,
                                 double green,
                                 double blue,
                                 double alpha) {
    if (!command_buffer || !texture) return -1;
    MGLRenderPassState state = mgl::defaultRenderPassState();
    state.color[0].attachment.texture = texture;
    state.color[0].attachment.level = level;
    state.color[0].attachment.slice = slice;
    state.color[0].attachment.depth_plane = depth_plane;
    state.color[0].attachment.load_action =
        static_cast<uint32_t>(MTL::LoadActionClear);
    state.color[0].clear_red = red;
    state.color[0].clear_green = green;
    state.color[0].clear_blue = blue;
    state.color[0].clear_alpha = alpha;
    void* encoder_handle = nullptr;
    if (mglRenderCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderEncodeColorClearForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, void* texture, uint64_t level, uint64_t slice, uint64_t depth_plane, double red, double green, double blue, double alpha) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeColorClear(
        owner->current, texture, level, slice, depth_plane,
        red, green, blue, alpha);
}

int mglRenderEncodeDepthClear(void* command_buffer,
                                 void* texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double clear_depth) {
    if (!command_buffer || !texture) return -1;
    MGLRenderPassState state = mgl::defaultRenderPassState();
    state.depth.attachment.texture = texture;
    state.depth.attachment.level = level;
    state.depth.attachment.slice = slice;
    state.depth.attachment.depth_plane = depth_plane;
    state.depth.attachment.load_action =
        static_cast<uint32_t>(MTL::LoadActionClear);
    state.depth.clear_depth = clear_depth;
    void* encoder_handle = nullptr;
    if (mglRenderCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderEncodeDepthClearForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, void* texture, uint64_t level, uint64_t slice, uint64_t depth_plane, double clear_depth) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeDepthClear(
        owner->current, texture, level, slice, depth_plane, clear_depth);
}

void mglRenderClearFboMatchCache(MGLRenderPassIdentityOwner * owner_handle) {
    mgl::RenderPassIdentityOwner* owner =
        reinterpret_cast<mgl::RenderPassIdentityOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    owner->cache = {};
    owner->cache_valid = false;
}

int clearBufferSlot(std::vector<MTL::Buffer*>& buffers,
                    std::vector<uint64_t>& offsets,
                    uint32_t index,
                    uint64_t offset) {
    if (index >= buffers.size()) return -1;
    mgl::BindingState::replaceObject(
        buffers[index], static_cast<MTL::Buffer*>(nullptr));
    offsets[index] = offset;
    return 0;
}

int mglRenderBindingClearVertexBuffer(MGLBindingState * binding_state, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    return state ? clearBufferSlot(state->vertexBuffers,
                                   state->vertexBufferOffsets, index, 0) : -1;
}

int mglRenderBindingClearFragmentBuffer(MGLBindingState * binding_state, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    return state ? clearBufferSlot(state->fragmentBuffers,
                                   state->fragmentBufferOffsets, index, 0) : -1;
}
