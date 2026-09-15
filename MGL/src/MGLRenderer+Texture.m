/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Texture.m
// Texture upload/download Metal path methods extracted from MGLRenderer.m

#include "mgl_texture_mip_ops.h"
#include "mgl_render_pass_sync_ops.h"
#include "mgl_render_pass_manager_ops.h"
#import "MGLRenderer_Private.h"
#include "mgl_blit_color_state.h" /* readback helper entries (log 141) */
#include "mgl_texture_readback_clear.h"
#include "mgl_gpu_recovery.h"
#include "mgl_pixel_format.h"
#include "mgl_texture_binding_resolve.h"
#import "MGLRenderer+Texture_Private.h"
#import "mgl_texture_readback_ops.h" /* the readback family is C now (log 181) */
#import "mgl_texture_create_ops.h" /* completeness / packed-DS upload / texel buffer (log 182) */
#import "mgl_texture_upload_ops.h" /* slice upload + dedicated CB copy (log 183) */
#import "mgl_texture_readback_ops.h" /* readPixels family is C (log 185) */
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_renderer_ports.h"  /* mglRendererProcessBuffer */
#include "mgl_blit_sampled_copy.h"  /* sampled RT copy refresh */
#include "mgl_blit_drivers.h"   /* mglBlitCopyImageSubData (log 145) */
#include "mgl_region_value.h"   // canonical region/origin/size constructors (O4 dedup sink)

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
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureReadDrawable((__bridge void *)renderer, glm_ctx,
                               pixel_bytes, bytes_per_row, bytes_per_image,
                               mglRendererCompatRegion(x, y, width, height));
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererReadIntegerPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureReadIntegerPixels((__bridge void *)renderer, glm_ctx,
                                    pixel_bytes, bytes_per_row,
                                    bytes_per_image,
                                    mglRendererCompatRegion(x, y, width, height),
                                    format, type);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererReadDepthPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureReadDepthPixels((__bridge void *)renderer, glm_ctx,
                                  pixel_bytes, bytes_per_row, bytes_per_image,
                                  mglRendererCompatRegion(x, y, width, height));
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererGetTexImage(GLMContext glm_ctx, Texture *texture,
    void *pixel_bytes, uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureGetTexImage((__bridge void *)renderer, glm_ctx, texture,
                              pixel_bytes, bytes_per_row, bytes_per_image,
                              mglRendererCompatRegion(x, y, width, height),
                              format, type, level, slice);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererGenerateMipmaps(GLMContext glm_ctx, Texture *texture)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureGenerateMipmaps((__bridge void *)renderer, glm_ctx, texture);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererSyncTextureBufferFromImage(GLMContext glm_ctx, Texture *texture)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx && texture) {
        mglTextureSyncBufferFromImage((__bridge void *)renderer, glm_ctx, texture);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererPrepareImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTexturePrepareImageUnitSlice((__bridge void *)renderer, glm_ctx, unit);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererFlushImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureFlushImageUnitSlice((__bridge void *)renderer, glm_ctx, unit);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererTexSubImage(GLMContext glm_ctx, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureSubImage((__bridge void *)renderer, glm_ctx, texture, buffer,
                              source_offset, source_pitch, source_image_size,
                              source_size, slice, level, width, height, depth,
                              x_offset, y_offset, z_offset);
    }
    mglRendererBackendEnd(&_backend_lease);
}

bool mglRendererTexSubImageBytes(GLMContext glm_ctx, Texture *texture,
    const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return false;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    bool result = false;
    if (renderer && glm_ctx) {
        result = mglTextureSubImageBytes((__bridge void *)renderer, glm_ctx, texture, bytes,
                                   bytes_size, source_offset, source_pitch,
                                   source_image_size, slice, level, width,
                                   height, depth, x_offset, y_offset,
                                   z_offset) != 0;
    }
    mglRendererBackendEnd(&_backend_lease);
    return result;
}

void mglRendererCopyTexSubImage(GLMContext glm_ctx, Texture *texture,
    uint32_t slice, int32_t level, int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglBlitCopyTexSubImage((__bridge void *)renderer, glm_ctx, texture,
                               slice, level, x_offset, y_offset, x, y, width,
                               height);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererCopyImageSubData(GLMContext glm_ctx, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglBlitCopyImageSubData(
            (__bridge void *)renderer, glm_ctx, source_texture, source_level,
            source_x, source_y, source_z, destination_texture,
            destination_level, destination_x, destination_y, destination_z,
            width, height, depth);
    }
    mglRendererBackendEnd(&_backend_lease);
}

static id mglTextureCreateBuffer(id device,
                                            NSUInteger length,
                                            uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}


static id mglTextureCreateTexture(
    id device,
    const MGLRenderTextureDescriptorState *descriptor)
{
    (void)device;
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(
            descriptor, NULL, &texture) == 0 &&
        texture) {
        return (__bridge_transfer id)texture;
    }
    return nil;
}


static void mglTextureReplaceRegion(id texture,
                                    MGLRegionValue region,
                                    NSUInteger level,
                                    NSUInteger slice,
                                    const void *bytes,
                                    NSUInteger bytesPerRow,
                                    NSUInteger bytesPerImage,
                                    BOOL useSlice)
{
    if (mglRenderTextureReplaceRegion(
            (__bridge void *)texture,
            region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth,
            level, slice, bytes, bytesPerRow, bytesPerImage,
            useSlice ? 1 : 0) != 0) {
        [NSException raise:@"MGLTextureReplaceRegionError"
                    format:@"C++ texture replaceRegion failed (level=%lu slice=%lu)",
                           (unsigned long)level, (unsigned long)slice];
    }
}



static id mglTextureCreateCommandBuffer(
    id queue)
{
    if (!queue) return nil;
    void *commandBuffer = NULL;
    if (mglRenderCreateCommandBuffer((__bridge void *)queue,
                                         &commandBuffer) == 0 &&
        commandBuffer) {
        return (__bridge id)commandBuffer;
    }
    return nil;
}

static id mglTextureCreateBlitEncoder(
    id commandBuffer)
{
    if (!commandBuffer) return nil;
    void *encoder = NULL;
    if (mglRenderCreateBlitEncoder((__bridge void *)commandBuffer,
                                       &encoder) == 0 && encoder) {
        return (__bridge id)encoder;
    }
    return nil;
}

/* Owner-first adapter for work that is encoded on the renderer's current
 * command buffer. Dedicated command buffers continue to use the raw helper
 * above because they are not owned by MGLRenderPassManager. */

static void mglTextureEndBlitEncoder(id encoder)
{
    if (!encoder) return;
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

static void mglTextureCommitCommandBuffer(id commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderCommitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer commit failed");
    }
}

static void mglTextureWaitCommandBuffer(id commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderWaitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer wait failed");
    }
}

static MGLRenderTextureInfo mglTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    return info;
}

/* AGX replaceRegion/copyFromBuffer require 256-byte row alignment for many
 * depth/stencil pixel formats even when the logical row is smaller. */
/* kMGLDepthStencilUploadRowAlignment moved to the C twin (log 198). */


/* CPU shadow storage uses five bytes per texel for GL_DEPTH32F_STENCIL8
 * (float depth plus one stencil byte), while Metal's packed depth/stencil
 * upload layout uses an eight-byte texel with stencil at byte 4. */


static void *mglTextureBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents((__bridge void *)buffer,
                                                   &contents, &length) == 0
        ? contents : NULL;
}

static void mglTextureCopyTextureToBuffer(
    id encoder,
    id source,
    NSUInteger sourceSlice,
    NSUInteger sourceLevel,
    MGLOriginValue sourceOrigin,
    MGLSizeValue sourceSize,
    id destination,
    NSUInteger destinationOffset,
    NSUInteger bytesPerRow,
    NSUInteger bytesPerImage)
{
    (void)mglRenderBlitCopyTextureToBuffer(
            (__bridge void *)encoder, (__bridge void *)source, sourceSlice,
            sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
            sourceSize.width, sourceSize.height, sourceSize.depth,
            (__bridge void *)destination, destinationOffset, bytesPerRow,
            bytesPerImage);
}

@implementation MGLRenderer (Texture)









#pragma mark - Extracted from createMTLTextureFromGLTexture:
















// AGX-SAFE Fallback texture creation for GPU error recovery scenarios
- (id) createFallbackMTLTexture:(Texture *) tex
{
    // Validate texture parameters before creating Metal texture to prevent Metal assertion failures
    if (!tex || tex->width <= 0 || tex->height <= 0 || tex->width > 32768 || tex->height > 32768) {
        NSLog(@"MGL AGX: Skipping fallback texture creation - invalid dimensions %dx%d",
              tex ? tex->width : 0, tex ? tex->height : 0);
        return nil;
    }

    NSLog(@"MGL AGX: Creating emergency fallback texture (size: %dx%dx%d)", tex->width, tex->height, tex->depth);

    @try {
        uint32_t fallbackFormat = mglRenderFallbackPixelFormat(
            mtlPixelFormatForGLTex(tex), (uint32_t)tex->internalformat);

        BOOL isDepthOrStencilFormat =
            mglRenderPixelFormatIsDepthOrStencil(fallbackFormat) != 0;

        MGLRenderTextureDescriptorState fallbackDesc = {
            .texture_type = MGLTextureType2D,
            .pixel_format = fallbackFormat,
            .width = MAX(tex->width, 1), .height = MAX(tex->height, 1),
            .depth = 1u, .mipmap_level_count = 1u,
            .sample_count = 1u, .array_length = 1u,
            .usage = MGL_TEXTURE_USAGE_SHADER_READ,
        };
        if (tex->is_render_target || isDepthOrStencilFormat) {
            fallbackDesc.usage |= MGL_TEXTURE_USAGE_RENDER_TARGET;
        }

        id fallbackTexture =
            mglTextureCreateTexture(_device, &fallbackDesc);

        if (fallbackTexture) {
            // Fill with simple gradient pattern using a simple approach
            NSUInteger width = mglTextureInfo(fallbackTexture).width;
            NSUInteger height = mglTextureInfo(fallbackTexture).height;

            if (!isDepthOrStencilFormat && width <= 512 && height <= 512) {
                uint32_t *gradientData = calloc(width * height, sizeof(uint32_t));
                if (gradientData) {
                    // Create simple red-blue gradient
                    for (NSUInteger y = 0; y < height; y++) {
                        for (NSUInteger x = 0; x < width; x++) {
                            NSUInteger index = y * width + x;
                            uint8_t r = (uint8_t)((x * 255) / width);
                            uint8_t g = 128;
                            uint8_t b = (uint8_t)((y * 255) / height);
                            uint8_t a = 255;
                            gradientData[index] = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
                        }
                    }

                    MGLRegionValue region = mglTextureRegion2D(0, 0, width, height);
                    mglTextureReplaceRegion(
                        fallbackTexture, region, 0, 0, gradientData,
                        width * sizeof(uint32_t), 0, NO);

                    free(gradientData);
                    NSLog(@"MGL AGX: Fallback color texture created with gradient pattern");
                }
            }
        }

        return fallbackTexture;

    } @catch (NSException *exception) {
        NSLog(@"MGL AGX: Even fallback texture creation failed: %@", exception.reason);
        return nil;
    }
}

// Helper function to calculate bytes per pixel for different OpenGL formats





- (void)traceSampledTextureReadback:(id)texture
                              glTex:(Texture *)glTex
                              level:(TextureLevel *)level0
                            program:(GLuint)program
                            binding:(GLuint)binding
                              stage:(NSString *)stage
                             reason:(NSString *)reason
                                hit:(uint64_t)hit
{
    if (!texture || !_device || !_commandQueue) {
        return;
    }

    MGLRenderTextureInfo textureInfo = {0};
    if (mglRenderGetTextureInfo((__bridge void *)texture,
                                   &textureInfo) != 0) {
        return;
    }
    uint32_t fmt = textureInfo.pixel_format;
    BOOL fourByteColor =
        mglRenderPixelFormatIsUnorm8Color(fmt) != 0;
    if (!fourByteColor) {
        mglTraceLog("MGL TRACE sampled.readback skip program=%u binding=%u glTex=%u reason=%s fmt=%lu type=%lu size=%lux%lu hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              [reason UTF8String],
              (unsigned long)fmt,
              (unsigned long)textureInfo.texture_type,
              (unsigned long)textureInfo.width,
              (unsigned long)textureInfo.height,
              (unsigned long long)hit);
        return;
    }

    NSUInteger texWidth = (NSUInteger)textureInfo.width;
    NSUInteger texHeight = (NSUInteger)textureInfo.height;
    if (texWidth == 0 || texHeight == 0) {
        return;
    }

    NSUInteger sampleWidth = MIN(texWidth, 8u);
    NSUInteger sampleHeight = MIN(texHeight, 8u);
    NSUInteger bytesPerPixel = 4u;
    NSUInteger bytesPerRow = sampleWidth * bytesPerPixel;
    NSUInteger byteCount = bytesPerRow * sampleHeight;
    if (byteCount == 0) {
        return;
    }

    id readback = mglTextureCreateBuffer(
        _device, byteCount, MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
    id cb = mglTextureCreateCommandBuffer(_commandQueue);
    id blit = mglTextureCreateBlitEncoder(cb);
    if (!readback || !cb || !blit) {
        mglTraceLog("MGL TRACE sampled.readback setup-fail program=%u binding=%u glTex=%u reason=%s readback=%p cb=%p blit=%p hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              [reason UTF8String],
              readback,
              cb,
              blit,
              (unsigned long long)hit);
        return;
    }

    mglTextureCopyTextureToBuffer(
        blit, texture, 0, 0, mglTextureOrigin(0, 0, 0),
        mglTextureSize(sampleWidth, sampleHeight, 1), readback, 0,
        bytesPerRow, byteCount);
    mglTextureEndBlitEncoder(blit);
    mglTextureCommitCommandBuffer(cb);
    mglTextureWaitCommandBuffer(cb);

    const uint8_t *p = (const uint8_t *)mglTextureBufferContents(readback);
    uint64_t byteSum = 0;
    NSUInteger nonZeroBytes = 0;
    uint32_t firstPixel = 0;
    uint32_t pixelXor = 0;
    uint32_t minPixel = UINT32_MAX;
    uint32_t maxPixel = 0;
    NSUInteger pixelCount = byteCount / sizeof(uint32_t);

    if (p) {
        for (NSUInteger i = 0; i < byteCount; i++) {
            byteSum += (uint64_t)p[i];
            if (p[i] != 0) {
                nonZeroBytes++;
            }
        }
        if (byteCount >= sizeof(firstPixel)) {
            memcpy(&firstPixel, p, sizeof(firstPixel));
        }
        for (NSUInteger i = 0; i < pixelCount; i++) {
            uint32_t pixel = 0;
            memcpy(&pixel, p + (i * sizeof(pixel)), sizeof(pixel));
            pixelXor ^= pixel;
            if (pixel < minPixel) {
                minPixel = pixel;
            }
            if (pixel > maxPixel) {
                maxPixel = pixel;
            }
        }
    }

    MGLRenderCommandBufferState sampledState = {0};
    (void)mglRenderGetCommandBufferState(
        (__bridge void *)cb, &sampledState);
    NSString *sampledError = sampledState.has_error
        ? [NSString stringWithFormat:@"%s (domain=%s code=%lld)",
             sampledState.error_description,
             sampledState.error_domain,
             (long long)sampledState.error_code]
        : nil;
    mglTraceLog("MGL TRACE sampled.readback stage=%s program=%u binding=%u glTex=%u reason=%s hit=%llu "
          "mtl=%p fmt=%lu type=%lu size=%lux%lu sample=%lux%lu status=%s error=%@ "
          "nonZero=%lu/%lu sum=%llu first=0x%08x min=0x%08x max=0x%08x xor=0x%08x "
          "level(init ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx)",
          [stage UTF8String],
          (unsigned)program,
          (unsigned)binding,
          glTex ? (unsigned)glTex->name : 0u,
          reason,
          (unsigned long long)hit,
          texture,
          (unsigned long)fmt,
          (unsigned long)textureInfo.texture_type,
          (unsigned long)texWidth,
          (unsigned long)texHeight,
          (unsigned long)sampleWidth,
          (unsigned long)sampleHeight,
          mglCommandBufferStatusName(
              (uint32_t)sampledState.status),
          sampledError,
          (unsigned long)nonZeroBytes,
          (unsigned long)byteCount,
          (unsigned long long)byteSum,
          firstPixel,
          minPixel == UINT32_MAX ? 0u : minPixel,
          maxPixel,
          pixelXor,
          level0 ? (unsigned)level0->ever_written : 0u,
          level0 ? (unsigned)level0->has_initialized_data : 0u,
          level0 ? (unsigned)level0->suspicious_zero_upload : 0u,
          level0 ? (unsigned)level0->last_init_source : 0u,
          (unsigned long)(level0 ? level0->last_upload_size : 0u),
          level0 ? (void *)level0->last_src_ptr : NULL,
          (unsigned long long)(level0 ? level0->last_src_hash : 0ull));
}
@end
