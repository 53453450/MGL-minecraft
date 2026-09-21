/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_entries.c - the GL-API entry surface of the former
 * MGLRenderer+Texture.m (P0-1, log 200).  Every one of these functions was
 * already plain C except for the renderer lookup and the backend lease, which
 * the Objective-C file spelled with mglRendererForContext() and
 * mglRendererEnterBackendLease(); C uses glm_ctx->platform_renderer_shell and
 * mglRendererBackendBeginContext() directly, the same way
 * mgl_blit_drivers.c's BlitFramebuffer entry does.
 *
 * With them re-homed the .m had nothing left but an empty @implementation, so
 * the file is deleted (file count 3 -> 2).
 */

#include "glm_context.h"
#include "mgl_blit_color_state.h"
#include "mgl_blit_drivers.h"
#include "mgl_blit_sampled_copy.h"
#include "mgl_env_flag.h"
#include "mgl_gpu_recovery.h"
#include "mgl_pixel_format.h"
#include "mgl_region_value.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_pass_sync_ops.h"
#include "mgl_renderer_backend.h"
#include "mgl_renderer_ports.h"
#include "mgl_texture_binding_resolve.h"
#include "mgl_texture_create_ops.h"
#include "mgl_texture_mip_ops.h"
#include "mgl_texture_readback_clear.h"
#include "mgl_texture_readback_ops.h"
#include "mgl_texture_upload_ops.h"

enum {
    MGL_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
    MGL_TEXTURE_STORAGE_PRIVATE = 2u,
    MGL_TEXTURE_CPU_CACHE_DEFAULT = 0u,
    MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED = 1u,
    MGL_TEXTURE_USAGE_SHADER_READ = 1u,
    MGL_TEXTURE_USAGE_SHADER_WRITE = 2u,
    /* Matches MTLTextureUsageShaderAtomic (macOS 14+ / Metal 3.1). */
    MGL_TEXTURE_USAGE_SHADER_ATOMIC = 0x20u,
    MGL_TEXTURE_USAGE_RENDER_TARGET = 4u,
    MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW = 16u,
};

static MGLRegionValue mglRendererCompatRegion(int32_t x, int32_t y,
                                         int32_t width, int32_t height)
{
    return (MGLRegionValue){
        .origin = {(int64_t)x, (int64_t)y, 0},
        .size = {(uint64_t)width, (uint64_t)height, 1u},
    };
}

void mglRendererReadDrawable(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTextureReadDrawable(renderer, glm_ctx,
                               pixel_bytes, bytes_per_row, bytes_per_image,
                               mglRendererCompatRegion(x, y, width, height));
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererReadIntegerPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTextureReadIntegerPixels(renderer, glm_ctx,
                                    pixel_bytes, bytes_per_row,
                                    bytes_per_image,
                                    mglRendererCompatRegion(x, y, width, height),
                                    format, type);
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererReadDepthPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTextureReadDepthPixels(renderer, glm_ctx,
                                  pixel_bytes, bytes_per_row, bytes_per_image,
                                  mglRendererCompatRegion(x, y, width, height));
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererGetTexImage(GLMContext glm_ctx, Texture *texture,
    void *pixel_bytes, uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTextureGetTexImage(renderer, glm_ctx, texture,
                              pixel_bytes, bytes_per_row, bytes_per_image,
                              mglRendererCompatRegion(x, y, width, height),
                              format, type, level, slice);
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererGenerateMipmaps(GLMContext glm_ctx, Texture *texture)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTextureGenerateMipmaps(renderer, glm_ctx, texture);
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererSyncTextureBufferFromImage(GLMContext glm_ctx, Texture *texture)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx && texture) {
        mglTextureSyncBufferFromImage(renderer, glm_ctx, texture);
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererPrepareImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTexturePrepareImageUnitSlice(renderer, glm_ctx, unit);
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererFlushImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTextureFlushImageUnitSlice(renderer, glm_ctx, unit);
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererTexSubImage(GLMContext glm_ctx, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglTextureSubImage(renderer, glm_ctx, texture, buffer,
                              source_offset, source_pitch, source_image_size,
                              source_size, slice, level, width, height, depth,
                              x_offset, y_offset, z_offset);
    }
    mglRendererBackendEnd(&backend_lease);
}

bool mglRendererTexSubImageBytes(GLMContext glm_ctx, Texture *texture,
    const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return false;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    bool result = false;
    if (renderer && glm_ctx) {
        result = mglTextureSubImageBytes(renderer, glm_ctx, texture, bytes,
                                   bytes_size, source_offset, source_pitch,
                                   source_image_size, slice, level, width,
                                   height, depth, x_offset, y_offset,
                                   z_offset) != 0;
    }
    mglRendererBackendEnd(&backend_lease);
    return result;
}

void mglRendererCopyTexSubImage(GLMContext glm_ctx, Texture *texture,
    uint32_t slice, int32_t level, int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglBlitCopyTexSubImage(renderer, glm_ctx, texture,
                               slice, level, x_offset, y_offset, x, y, width,
                               height);
    }
    mglRendererBackendEnd(&backend_lease);
}

void mglRendererCopyImageSubData(GLMContext glm_ctx, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglBlitCopyImageSubData(
            renderer, glm_ctx, source_texture, source_level,
            source_x, source_y, source_z, destination_texture,
            destination_level, destination_x, destination_y, destination_z,
            width, height, depth);
    }
    mglRendererBackendEnd(&backend_lease);
}










/* Owner-first adapter for work that is encoded on the renderer's current
 * command buffer. Dedicated command buffers continue to use the raw helper
 * above because they are not owned by MGLRenderPassManager. */





/* AGX replaceRegion/copyFromBuffer require 256-byte row alignment for many
 * depth/stencil pixel formats even when the logical row is smaller. */
/* kMGLDepthStencilUploadRowAlignment moved to the C twin (log 198). */


/* CPU shadow storage uses five bytes per texel for GL_DEPTH32F_STENCIL8
 * (float depth plus one stencil byte), while Metal's packed depth/stencil
 * upload layout uses an eight-byte texel with stencil at byte 4. */




