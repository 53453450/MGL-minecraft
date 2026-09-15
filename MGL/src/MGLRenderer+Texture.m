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
        [renderer mtlTexSubImage:glm_ctx tex:texture buf:buffer
                      src_offset:source_offset src_pitch:source_pitch
                  src_image_size:source_image_size src_size:source_size
                           slice:slice level:level width:width height:height
                           depth:depth xoffset:x_offset yoffset:y_offset
                         zoffset:z_offset];
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
        result = [renderer mtlTexSubImageBytes:glm_ctx tex:texture
                                    bytes:bytes bytesSize:bytes_size
                               src_offset:source_offset src_pitch:source_pitch
                           src_image_size:source_image_size
                                    slice:slice level:level
                                    width:width height:height depth:depth
                                  xoffset:x_offset yoffset:y_offset
                                  zoffset:z_offset];
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

static id mglTextureCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
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
static const NSUInteger kMGLDepthStencilUploadRowAlignment = 256u;

static NSUInteger mglDepthStencilAlignedBytesPerRow(NSUInteger logicalBytesPerRow)
{
    if (logicalBytesPerRow == 0) {
        return 0;
    }
    return ((logicalBytesPerRow + kMGLDepthStencilUploadRowAlignment - 1u) /
            kMGLDepthStencilUploadRowAlignment) * kMGLDepthStencilUploadRowAlignment;
}

/* CPU shadow storage uses five bytes per texel for GL_DEPTH32F_STENCIL8
 * (float depth plus one stencil byte), while Metal's packed depth/stencil
 * upload layout uses an eight-byte texel with stencil at byte 4. */
static void *mglCreateDepthStencilMetalUpload(
    Texture *tex, uint32_t pixelFormat, const uint8_t *src,
    NSUInteger width, NSUInteger height, NSUInteger srcBytesPerRow,
    NSUInteger *outBytesPerRow, NSUInteger *outBytesPerImage)
{
    if (outBytesPerRow) *outBytesPerRow = 0;
    if (outBytesPerImage) *outBytesPerImage = 0;
    if (!tex || !src || width == 0 || height == 0 || srcBytesPerRow == 0 ||
        !mglRenderDepth32FStencil8NeedsUnpack(
            (uint32_t)tex->internalformat, (uint32_t)pixelFormat,
            (uint32_t)srcBytesPerRow, (uint32_t)width)) {
        return NULL;
    }
    NSUInteger logicalBytesPerRow = width * 8u;
    NSUInteger dstBytesPerRow = mglDepthStencilAlignedBytesPerRow(logicalBytesPerRow);
    if (dstBytesPerRow == 0) {
        return NULL;
    }
    NSUInteger dstBytesPerImage = dstBytesPerRow * height;
    uint8_t *dst = calloc(1u, dstBytesPerImage);
    if (!dst) return NULL;
    for (NSUInteger y = 0; y < height; ++y) {
        const uint8_t *srcRow = src + y * srcBytesPerRow;
        uint8_t *dstRow = dst + y * dstBytesPerRow;
        for (NSUInteger x = 0; x < width; ++x) {
            memcpy(dstRow + x * 8u, srcRow + x * 5u, 4u);
            dstRow[x * 8u + 4u] = srcRow[x * 5u + 4u];
        }
    }
    if (outBytesPerRow) *outBytesPerRow = dstBytesPerRow;
    if (outBytesPerImage) *outBytesPerImage = dstBytesPerImage;
    return dst;
}


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





-(void)mtlTexSubImage:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset
{
    METAL_LOCK();
    [self mtlTexSubImageLocked:glm_ctx tex:tex buf:buf src_offset:src_offset src_pitch:src_pitch src_image_size:src_image_size src_size:src_size slice:slice level:level width:width height:height depth:depth xoffset:xoffset yoffset:yoffset zoffset:zoffset];
    METAL_UNLOCK();
}

-(void)mtlTexSubImageLocked:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset
{
    if (!tex || !buf) {
        NSLog(@"MGL ERROR: mtlTexSubImage called with null tex/buf (tex=%p buf=%p)", tex, buf);
        return;
    }

    if (src_pitch == 0 || width == 0 || height == 0) {
        NSLog(@"MGL ERROR: mtlTexSubImage invalid dimensions/pitch tex=%u width=%zu height=%zu src_pitch=%zu",
              tex->name, width, height, src_pitch);
        return;
    }

    // we can deal with a null buffer but we need a texture
    if (buf->data.mtl_data == NULL)
    {
        mglRendererBindMTLBuffer((__bridge void *)self, buf);
        RETURN_ON_NULL(buf->data.mtl_data);
    }

    id buffer = (__bridge id)(buf->data.mtl_data);
    if (!buffer) {
        NSLog(@"MGL ERROR: mtlTexSubImage missing Metal buffer object tex=%u", tex->name);
        return;
    }


    if (tex->mtl_data) {
        id dstTexture = (__bridge id)(tex->mtl_data);
        uint32_t dstPixelFormat = mglTextureInfo(dstTexture).pixel_format;
        BOOL needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat, dstPixelFormat);
        BOOL needsRGBA8Expand = NO;
        if (!needsChannelExpand) {
            needsRGBA8Expand = mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, dstPixelFormat);
        }
        if (needsChannelExpand || needsRGBA8Expand) {
            uint32_t rgbDst = 0u;
            NSUInteger dstBytesPerPixel = 4u;
            if (needsChannelExpand &&
                mglRenderRGBExpandParams(dstPixelFormat, NULL, &rgbDst,
                                         NULL)) {
                dstBytesPerPixel = (NSUInteger)rgbDst * 4u;
            } else if (needsChannelExpand) {
                dstBytesPerPixel = 16u;
            }
            NSUInteger cpuBytesPerPixel = (tex->faces[0].levels && level < tex->num_levels &&
                                           tex->faces[0].levels[level].width > 0u &&
                                           tex->faces[0].levels[level].pitch > 0u)
                ? (NSUInteger)(tex->faces[0].levels[level].pitch / tex->faces[0].levels[level].width)
                : mglTextureBytesPerPixelForFormat(tex->internalformat);
            if (cpuBytesPerPixel == 0u) {
                cpuBytesPerPixel = (NSUInteger)sizeForInternalFormat(tex->internalformat, 0, 0);
            }
            if (cpuBytesPerPixel > 0u && cpuBytesPerPixel != dstBytesPerPixel) {
                NSUInteger copyHeight = MAX((NSUInteger)height, 1UL);
                NSUInteger copyDepth = MAX((NSUInteger)depth, 1UL);
                NSUInteger dstRowBytes = (NSUInteger)width * dstBytesPerPixel;
                NSUInteger dstImageBytes = dstRowBytes * copyHeight;
                size_t sourceImagePitch = src_image_size;
                size_t minimumImagePitch = src_pitch * copyHeight;
                if (sourceImagePitch < minimumImagePitch) {
                    sourceImagePitch = minimumImagePitch;
                }
                size_t packedBytes = dstImageBytes * copyDepth;
                if (packedBytes != 0u && packedBytes <= (512u * 1024u * 1024u)) {
                    const uint8_t *sourceBase = (const uint8_t *)mglTextureBufferContents(buffer);
                    NSMutableData *packedUpload = [NSMutableData dataWithLength:packedBytes];
                    if (packedUpload && packedUpload.mutableBytes && sourceBase) {
                        uint8_t *packedBytesPtr = (uint8_t *)packedUpload.mutableBytes;
                        bool expandOK = true;
                        for (NSUInteger z = 0; z < copyDepth && expandOK; z++) {
                            size_t sliceBaseOff = src_offset + (size_t)z * sourceImagePitch;
                            size_t lastRowOff = sliceBaseOff + (size_t)(copyHeight - 1u) * src_pitch;
                            size_t rowBytesCpu = (NSUInteger)width * cpuBytesPerPixel;
                            if (lastRowOff > src_size || rowBytesCpu > src_size - lastRowOff) {
                                expandOK = false;
                                break;
                            }
                            const uint8_t *sliceSrc = sourceBase + sliceBaseOff;
                            NSUInteger expandedBPR = 0, expandedBPI = 0;
                            uint8_t *expanded = NULL;
                            if (needsRGBA8Expand) {
                                expanded = mglCreateRGBA8ExpandedUpload(tex,
                                                                        sliceSrc,
                                                                        width,
                                                                        copyHeight,
                                                                        src_pitch,
                                                                        &expandedBPR,
                                                                        &expandedBPI);
                            } else {
                                expanded = mglCreateChannelExpandedUpload(tex,
                                                                           dstPixelFormat,
                                                                           sliceSrc,
                                                                           width,
                                                                           copyHeight,
                                                                           src_pitch,
                                                                           &expandedBPR,
                                                                           &expandedBPI);
                            }
                            if (!expanded) {
                                expandOK = false;
                                break;
                            }
                            memcpy(packedBytesPtr + (z * dstImageBytes), expanded, expandedBPI);
                            free(expanded);
                        }
                        if (expandOK) {
                            id uploadBuffer =
                                mglTextureCreateBufferWithBytes(
                                    _device, packedUpload.bytes, packedBytes,
                                    MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
                            if (uploadBuffer) {
                                bool uploaded = mglTextureEncodeBytesUpload( (__bridge void *)self, tex, (__bridge void *)uploadBuffer, 0, dstRowBytes, dstImageBytes, width, height, depth, slice, level, xoffset, yoffset, zoffset, "mtlTexSubImage");
                                if (!uploaded) {
                                    NSLog(@"MGL ERROR: mtlTexSubImage expanded PBO upload failed (tex=%u slice=%u level=%u)",
                                          tex->name, slice, level);
                                }
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    bool uploaded = mglTextureEncodeBytesUpload( (__bridge void *)self, tex, (__bridge void *)buffer, src_offset, src_pitch, src_image_size, width, height, depth, slice, level, xoffset, yoffset, zoffset, "mtlTexSubImage");
    if (!uploaded) {
        NSLog(@"MGL ERROR: mtlTexSubImage dedicated upload failed (tex=%u slice=%u level=%u)",
              tex->name, slice, level);
    }
}

-(bool)mtlTexSubImageBytes:(GLMContext)glm_ctx tex:(Texture *)tex bytes:(const void *)bytes bytesSize:(size_t)bytes_size src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset
{
    (void)glm_ctx;
    if (!tex || !bytes || src_pitch == 0 || width == 0 || height == 0) {
        return false;
    }
    if (src_offset > bytes_size || level >= tex->num_levels) {
        return false;
    }

    NSUInteger bytesPerPixel = mglTextureBytesPerPixelForFormat(tex->internalformat);
    if (bytesPerPixel == 0u &&
        tex->faces[0].levels &&
        tex->faces[0].levels[level].width > 0u) {
        TextureLevel *levelInfo = &tex->faces[0].levels[level];
        if (levelInfo->pitch > 0u &&
            (levelInfo->pitch % levelInfo->width) == 0u) {
            bytesPerPixel = (NSUInteger)(levelInfo->pitch / levelInfo->width);
        }
    }
    if (bytesPerPixel == 0u) {
        return false;
    }

    NSUInteger copyHeight = MAX((NSUInteger)height, 1UL);
    NSUInteger copyDepth = MAX((NSUInteger)depth, 1UL);
    NSUInteger rowBytes = (NSUInteger)width * bytesPerPixel;
    if (rowBytes == 0u || rowBytes > src_pitch) {
        return false;
    }

    if (!tex->mtl_data) {
        return false;
    }

    /* Channel expansion: GL_RGB32* (12 bytes/pixel) -> Metal RGBA32* (16 bytes/pixel).
     * The CPU backing stores 3 channels per pixel, but the Metal texture expects
     * 4 channels. We must expand each pixel by inserting a default alpha before
     * uploading, otherwise the data layout mismatches and pixels shift. */
    id dstTexture = (__bridge id)(tex->mtl_data);
    uint32_t dstPixelFormat = mglTextureInfo(dstTexture).pixel_format;
    BOOL needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat,
                                                              dstPixelFormat);
    NSUInteger dstBytesPerPixel = bytesPerPixel;
    if (needsChannelExpand) {
        uint32_t rgbDst = 0u;
        if (mglRenderRGBExpandParams(dstPixelFormat, NULL, &rgbDst, NULL)) {
            dstBytesPerPixel = (NSUInteger)rgbDst * 4u;
        } else {
            needsChannelExpand = NO;
        }
    }

    /* RGBA8 expansion: Metal has no RGB8 pixel format, so GL_RGB8-family
     * internal formats (3 bytes/pixel in the CPU backing store) are backed
     * by Metal RGBA8 variants (4 bytes/pixel).  Without per-pixel channel
     * expansion the 3-byte source is uploaded directly into a 4-byte Metal
     * texture, shifting pixels and producing vertical stripes.  Every other
     * upload path (createMTLTextureFromGLTexture, refreshMetalTextureCPUData,
     * mtlCopyImageSubData) expands via mglCreateRGBA8ExpandedUpload; the
     * direct mtlTexSubImageBytes path must do the same. */
    BOOL needsRGBA8Expand = NO;
    if (!needsChannelExpand) {
        needsRGBA8Expand = mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat,
                                                                        dstPixelFormat);
        if (needsRGBA8Expand) {
            dstBytesPerPixel = 4;
        }
    }

    size_t sourceImagePitch = src_image_size;
    size_t minimumImagePitch = src_pitch * copyHeight;
    if (sourceImagePitch < minimumImagePitch) {
        sourceImagePitch = minimumImagePitch;
    }

    NSUInteger dstRowBytes = (NSUInteger)width * dstBytesPerPixel;
    NSUInteger dstImageBytes = dstRowBytes * copyHeight;
    size_t packedBytes = dstImageBytes * copyDepth;
    if (packedBytes == 0u || packedBytes > (512u * 1024u * 1024u)) {
        return false;
    }

    NSMutableData *packedUpload = [NSMutableData dataWithLength:packedBytes];
    if (!packedUpload || !packedUpload.mutableBytes) {
        return false;
    }

    const uint8_t *sourceBase = (const uint8_t *)bytes;
    uint8_t *packedBytesPtr = (uint8_t *)packedUpload.mutableBytes;

    if (needsChannelExpand) {
        uint32_t srcCompU = 0u, dstCompU = 0u;
        uint64_t alphaDefault = 0;
        if (!mglRenderRGBExpandParams(dstPixelFormat, &srcCompU, &dstCompU,
                                      &alphaDefault)) {
            return false;
        }
        NSUInteger srcCompBytes = srcCompU;
        NSUInteger dstCompBytes = dstCompU;
        NSUInteger srcPixelBytes = srcCompBytes * 3;  /* 3 channels in source */
        NSUInteger dstPixelBytes = dstCompBytes * 4;  /* 4 channels in destination */

        for (NSUInteger z = 0; z < copyDepth; z++) {
            for (NSUInteger y = 0; y < copyHeight; y++) {
                size_t srcRowOffset = src_offset + ((size_t)z * sourceImagePitch) + ((size_t)y * src_pitch);
                if (srcRowOffset > bytes_size || rowBytes > bytes_size - srcRowOffset) {
                    return false;
                }
                const uint8_t *srcRow = sourceBase + srcRowOffset;
                uint8_t *dstRow = packedBytesPtr + (z * dstImageBytes) + (y * dstRowBytes);
                for (NSUInteger x = 0; x < width; x++) {
                    const uint8_t *srcPixel = srcRow + x * srcPixelBytes;
                    uint8_t *dstPixel = dstRow + x * dstPixelBytes;
                    /* Copy 3 channels (R, G, B) */
                    memcpy(dstPixel, srcPixel, srcPixelBytes);
                    /* Set alpha channel to default value */
                    memcpy(dstPixel + srcPixelBytes, &alphaDefault, dstCompBytes);
                }
            }
        }
    } else if (needsRGBA8Expand) {

        for (NSUInteger z = 0; z < copyDepth; z++) {
            size_t sliceBaseOff = src_offset + (size_t)z * sourceImagePitch;
            size_t lastRowOff = sliceBaseOff + (size_t)(copyHeight - 1u) * src_pitch;
            if (lastRowOff > bytes_size || rowBytes > bytes_size - lastRowOff) {
                return false;
            }
            const uint8_t *sliceSrc = sourceBase + sliceBaseOff;
            NSUInteger expandedBPR = 0, expandedBPI = 0;
            uint8_t *expanded = mglCreateRGBA8ExpandedUpload(tex,
                                                              sliceSrc,
                                                              width,
                                                              copyHeight,
                                                              src_pitch,
                                                              &expandedBPR,
                                                              &expandedBPI);
            if (!expanded) {
                return false;
            }
            memcpy(packedBytesPtr + (z * dstImageBytes), expanded, expandedBPI);
            free(expanded);
        }
    } else {
        /* No channel expansion needed - direct copy */
        for (NSUInteger z = 0; z < copyDepth; z++) {
            for (NSUInteger y = 0; y < copyHeight; y++) {
                size_t srcRowOffset = src_offset + ((size_t)z * sourceImagePitch) + ((size_t)y * src_pitch);
                if (srcRowOffset > bytes_size || rowBytes > bytes_size - srcRowOffset) {
                    static uint64_t s_subUploadRangeFailLogs = 0;
                    uint64_t hit = ++s_subUploadRangeFailLogs;
                    if (hit <= 32ull || (hit % 512ull) == 0ull) {
                        NSLog(@"MGL TEXSUBIMAGE BYTES range fail tex=%u level=%u off=%zu rowBytes=%lu pitch=%zu image=%zu size=%zu z=%lu y=%lu hit=%llu",
                              (unsigned)tex->name,
                              (unsigned)level,
                              srcRowOffset,
                              (unsigned long)rowBytes,
                              src_pitch,
                              sourceImagePitch,
                              bytes_size,
                              (unsigned long)z,
                              (unsigned long)y,
                              (unsigned long long)hit);
                    }
                    return false;
                }
                memcpy(packedBytesPtr + (z * dstImageBytes) + (y * dstRowBytes),
                       sourceBase + srcRowOffset,
                       rowBytes);
            }
        }
    }

    void *dsMetalUpload = NULL;
    const void *uploadBytesPtr = packedBytesPtr;
    NSUInteger uploadRowBytes = dstRowBytes;
    NSUInteger uploadImageBytes = dstImageBytes;
    if (mglRenderDepth32FStencil8NeedsUnpack(
            (uint32_t)tex->internalformat, (uint32_t)dstPixelFormat,
            (uint32_t)dstRowBytes, (uint32_t)width)) {
        NSUInteger expandedBPR = 0;
        NSUInteger expandedBPI = 0;
        dsMetalUpload = mglCreateDepthStencilMetalUpload(
            tex, dstPixelFormat, packedBytesPtr, width, copyHeight,
            dstRowBytes, &expandedBPR, &expandedBPI);
        if (dsMetalUpload) {
            uploadBytesPtr = dsMetalUpload;
            uploadRowBytes = expandedBPR;
            uploadImageBytes = expandedBPI;
        }
    }

    NSUInteger metalSlice = slice;
    if (mglRenderTextureTargetIsArray((uint32_t)tex->target)) {
        metalSlice = zoffset;
    }

    if (mglRenderPixelFormatIsPackedDepthStencil((uint32_t)dstPixelFormat) &&
        mglTextureInfo(dstTexture).storage_mode != MGL_TEXTURE_STORAGE_PRIVATE &&
        uploadRowBytes >= width * 5u) {
        bool uploaded = false;
        @try {
            mglTextureReplaceRegion(
                dstTexture,
                mglTextureRegion2D(xoffset, yoffset, width, copyHeight),
                level, metalSlice, uploadBytesPtr, uploadRowBytes,
                uploadImageBytes, YES);
            uploaded = true;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: depth/stencil texSubImage replaceRegion failed tex=%u: %@",
                  (unsigned)tex->name, exception.reason);
        }
        if (uploaded) {
            uploaded = mglTextureUploadPackedDepthStencilStencilPlane(
        (__bridge void *)dstTexture, tex->name, uploadBytesPtr, width, copyHeight, uploadRowBytes, level, metalSlice, xoffset, yoffset);
        }
        free(dsMetalUpload);
        return uploaded;
    }

    size_t uploadBufferBytes = uploadImageBytes * copyDepth;
    id uploadBuffer = mglTextureCreateBufferWithBytes(
        _device, uploadBytesPtr, uploadBufferBytes,
        MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
    if (!uploadBuffer) {
        free(dsMetalUpload);
        return false;
    }

    bool uploaded = mglTextureEncodeBytesUpload( (__bridge void *)self, tex, (__bridge void *)uploadBuffer, 0, uploadRowBytes, uploadImageBytes, width, height, depth, slice, level, xoffset, yoffset, zoffset, "mtlTexSubImageBytes");
    if (uploaded &&
        mglRenderPixelFormatIsPackedDepthStencil((uint32_t)dstPixelFormat) &&
        uploadRowBytes >= width * 5u) {
        (void)mglTextureUploadPackedDepthStencilStencilPlane(
        (__bridge void *)dstTexture, tex->name, uploadBytesPtr, width, copyHeight, uploadRowBytes, level, metalSlice, xoffset, yoffset);
    }
    free(dsMetalUpload);
    if (uploaded && tex->is_render_target) {
        /* Direct CPU→Metal refresh of an FBO-attached texture must invalidate
         * the Y-flip sampled copy.  textures.c also releases the copy, but
         * bumping write_version keeps any concurrent/lazy refresh coherent
         * with the post-upload Metal contents (KHR-GL46.texture_barrier).
         * Rebuild immediately so every MRT attachment has a fresh Y-flip
         * copy before the first feedback draw (color1+ previously rebuilt
         * only as a side-effect of color0's sample-gate repair). */
        mglMarkTextureLevelMetalFilled(tex, level, packedBytes);
        id source = (__bridge id)(tex->mtl_data);
        if (source) {
            (void)mglBlitUpdateGLSampledRenderTargetCopy((__bridge void *)self, tex, (__bridge void *)source, "texSubImage_metal_fill");
        }
    }
    return uploaded;
}


#pragma mark - Extracted from createMTLTextureFromGLTexture:











- (id) createMTLTextureFromGLTexture:(Texture *) tex
{
    MGL_ASSERT_GL_THREAD();
    mglMetalCountCreate(MGLMetalKindTexture);
    // PROPER FIX: Enhanced pre-creation validation to prevent AGX driver issues
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or command queue not available for texture creation");
        return nil;
    }

    // Check if we're in a recovery state that would make texture creation futile
    if (mglRendererShouldSkipGPUOperations((__bridge void *)self)) {
        NSLog(@"MGL AGX: GPU operations temporarily suspended during recovery");
        return nil;
    }

    // Validate texture dimensions to prevent Metal assertion failures.
    // Texture buffers (GL_TEXTURE_BUFFER) can have very large widths (millions of texels)
    // since they map to MGLTextureTypeTextureBuffer which uses GPU address space.
    if (!mglRenderTextureDimsValid(tex->target, tex->width, tex->height,
                                   tex->depth)) {
            NSLog(@"MGL ERROR: Invalid texture dimensions %dx%dx%d - rejecting",
                  tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->depth : 0);
            tex->dirty_bits = 0;
            return nil;
        }

    if (mglRenderIsTextureBufferTarget(tex->target)) {
        return (__bridge id)mglTextureCreateMTLTexelBufferTexture(
            (__bridge void *)self, tex);
    }

    NSUInteger width, height, depth;

    MGLRenderTextureDescriptorState tex_desc = {0};
    uint32_t tex_type;
    uint32_t pixelFormat;
    uint num_faces;
    GLuint effective_mipmap_levels;
    GLuint upload_level_count;
    BOOL storageMipmapped;
    BOOL mipmapped;
    BOOL is_array;
    BOOL texture1DBackedBy2D;
    BOOL texture1DArrayBackedBy2DArray;

    effective_mipmap_levels = 0;
    upload_level_count = 0;
    storageMipmapped = NO;

    MGLRenderTextureTargetPlan targetPlan = {0};
    if (mglRenderTextureTargetPlan(
            (uint32_t)tex->target,
            (uint32_t)tex->samples,
            &targetPlan) != 0) {
        NSLog(@"MGL TEXTURE ERROR: unsupported texture target 0x%x for Metal texture creation tex=%u",
              tex->target,
              tex->name);
        return nil;
    }
    tex_type = (uint32_t)targetPlan.texture_type;
    num_faces = (uint)targetPlan.num_faces;
    is_array = targetPlan.is_array != 0u;
    texture1DBackedBy2D = targetPlan.texture_1d_backed_by_2d != 0u;
    texture1DArrayBackedBy2DArray =
        targetPlan.texture_1d_array_backed_by_2d_array != 0u;

    int effectiveMipmapped = 0;
    if (!mglTextureCheckCompleteness(tex, tex_type, num_faces,
                                     &effective_mipmap_levels,
                                     &effectiveMipmapped)) {
        return nil;
    }
    storageMipmapped = effectiveMipmapped ? YES : NO;

    // PROPER FIX: Get original texture format and validate for AGX compatibility
    pixelFormat = mtlPixelFormatForGLTex(tex);
    BOOL expandsSingleChannelSwizzle = mglTextureUploadNeedsSingleChannelSwizzle(tex);
    BOOL usesUploadSwizzleBake = mglTextureUploadNeedsSwizzleBake(tex);
    pixelFormat = mglRenderResolveUploadSwizzlePixelFormat(
        pixelFormat, expandsSingleChannelSwizzle ? 1 : 0,
        mglRenderSingleChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat),
        mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) ? 1 : 0,
        mglRenderIntegerMultiChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat),
        mglTextureUploadNeedsStencilSwizzleBake(tex) ? 1 : 0,
        mglRenderStencilSwizzleStoragePixelFormat(),
        mglTextureUploadNeedsDepthStencilDepthSwizzleBake(tex) ? 1 : 0,
        mglRenderSingleChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat));

    // Validate format compatibility with AGX, but preserve original intent
    BOOL needsFormatConversion = NO;
    uint32_t originalFormat = pixelFormat;
    int agxConverted = 0;
    pixelFormat = mglRenderAGXCompatiblePixelFormat(pixelFormat, &agxConverted);
    needsFormatConversion = agxConverted != 0;

    /* Metal does not allow depth/stencil pixel formats with MGLTextureType1DArray.
     * Promote to MGLTextureType2DArray with height=1, mirroring how mipmapped
     * 1D array textures are already promoted below.  Without this, creating a
     * GL_TEXTURE_1D_ARRAY depth texture (e.g. sampler_1d_array_shadow) triggers
     * a Metal validation assertion crash. */
    if (mglRenderPromote1DArrayDepthStencil(tex_type, pixelFormat)) {
        tex_type = MGLTextureType2DArray;
        texture1DArrayBackedBy2DArray = true;
    }

    width = tex->width;
    height = tex->height;
    depth = tex->depth;
    if (tex_type == MGLTextureType2DMultisample ||
        tex_type == MGLTextureType2DMultisampleArray) {
        storageMipmapped = NO;
        effective_mipmap_levels = 1u;
        tex->mipmapped = false;
    }

    mipmapped = storageMipmapped;
    /* GL may allocate num_levels>1 for a single-base-level image; only walk
     * mips that were actually populated unless the texture is mipmapped. */
    upload_level_count = mglRenderUploadLevelCount(
        mipmapped ? 1 : 0, tex->mipmapped ? 1 : 0, effective_mipmap_levels);

    tex_desc.texture_type = tex_type;
    tex_desc.pixel_format = pixelFormat;
    tex_desc.width = width;
    tex_desc.height = mglRenderTextureDescHeight(tex_type, (uint32_t)height);
    bool msEmulatedAsArray = false;
    {
        uint32_t outType = tex_type;
        uint32_t sampleCount = 1u;
        uint64_t arrayLen = 1u;
        uint64_t descDepth = 1u;
        NSUInteger samples = MAX((NSUInteger)2u, (NSUInteger)tex->samples);
        samples = MGLCapabilityClampSampleCount(&_capability, samples);
        if (mglRenderEmulateMSAsArray(tex_type, (uint32_t)samples,
                                      (uint64_t)depth, &outType, &sampleCount,
                                      &arrayLen, &descDepth)) {
            msEmulatedAsArray = true;
            tex_type = outType;
            tex_desc.texture_type = tex_type;
            tex_desc.sample_count = sampleCount;
            tex_desc.array_length = arrayLen;
            tex_desc.depth = descDepth;
        }
    }

    // CONSERVATIVE: Use only Metal API patterns that work reliably with AGX driver
    tex_desc.cpu_cache_mode = MGLCapabilityUseConservativeCPUCache(&_capability)
        ? MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED
        : MGL_TEXTURE_CPU_CACHE_DEFAULT;

    // Use shared storage for textures that need CPU upload (blit/replaceRegion).
    // Private storage is only safe for pure GPU render targets on Apple Silicon.
    bool hasUploadableCPUData = mglTextureHasUploadableCPUData(tex, num_faces, upload_level_count);
    bool needsCpuUpload = ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    bool preferSharedDepthStencil =
        mglMetalPixelFormatIsDepthOrStencil(pixelFormat);
    tex_desc.storage_mode =
        mglRenderPreferSharedStorage(needsCpuUpload ? 1 : 0,
                                     preferSharedDepthStencil ? 1 : 0)
            ? 0u
            : MGL_TEXTURE_STORAGE_PRIVATE;
    tex_desc.sample_count = MAX(tex_desc.sample_count, 1u);
    tex_desc.mipmap_level_count = MAX(tex_desc.mipmap_level_count, 1u);
    tex_desc.array_length = MAX(tex_desc.array_length, 1u);
    tex_desc.depth = MAX(tex_desc.depth, 1u);

    // Normalize depth/array semantics per Metal texture type.
    if ((tex_type == MGLTextureTypeCube ||
         tex_type == MGLTextureTypeCubeArray) &&
        !mglRenderCubeFaceSizeValid((uint64_t)width, (uint64_t)height)) {
            NSLog(@"MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
    }
    if (tex_type == MGLTextureTypeCubeArray) {
        uint64_t cubeCount = (uint64_t)depth;
        if (cubeCount > 1u && (cubeCount % 6u) != 0u) {
            NSLog(@"MGL WARNING: cube-array depth=%lu is not a multiple of 6, treating as cube count",
                  (unsigned long)cubeCount);
        }
    }
    uint64_t arrayLen = tex_desc.array_length;
    uint64_t descDepth = tex_desc.depth;
    if (mglRenderTextureArrayDepthForType(
            tex_type, is_array ? 1 : 0, msEmulatedAsArray ? 1 : 0,
            (uint64_t)width, (uint64_t)height, (uint64_t)depth, &arrayLen,
            &descDepth)) {
        tex_desc.array_length = arrayLen;
        tex_desc.depth = descDepth;
    }

    if (mipmapped)
    {
        if (mglRenderPromoteMipmapped1D(tex_type)) {
            tex_type = MGLTextureType2D;
            texture1DBackedBy2D = true;
        }
        /* Metal does not allow mipmapLevelCount > 1 for MGLTextureType1DArray.
         * Promote to MGLTextureType2DArray with height=1 to support mipmapped
         * 1D array textures.  The upload code checks texture1DArrayBackedBy2DArray
         * to treat each slice as 1 pixel tall. */
        if (mglRenderPromoteMipmapped1DArray(tex_type)) {
            tex_type = MGLTextureType2DArray;
            texture1DArrayBackedBy2DArray = true;
        }
        tex_desc.mipmap_level_count = MAX((GLuint)1, effective_mipmap_levels);
    }

    if (texture1DBackedBy2D || texture1DArrayBackedBy2DArray) {
        uint32_t backedType = tex_desc.texture_type;
        uint64_t backedArray = tex_desc.array_length;
        uint32_t backedHeight = (uint32_t)tex_desc.height;
        mglRenderApply1DBackingToDesc(texture1DBackedBy2D ? 1 : 0,
                                      texture1DArrayBackedBy2DArray ? 1 : 0,
                                      (uint64_t)height, &backedType,
                                      &backedArray, &backedHeight);
        tex_desc.texture_type = backedType;
        tex_desc.array_length = backedArray;
        tex_desc.height = backedHeight;
    }

    /* GL image access mode (GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE)
     * only governs the image binding, NOT the texture's overall capabilities.
     * A texture bound as a write-only image may still be sampled from via
     * sampler2D in the same shader.  Metal requires MGL_TEXTURE_USAGE_SHADER_READ
     * for sampling, so always include it alongside the image write flag.
     *
     * AIR always declares storage images as access::read_write (see
     * mgl_air_backend.cpp).  Binding a ShaderRead-only texture to that slot
     * yields zeroed imageLoad results on AGX, so READ_ONLY also needs
     * ShaderWrite even though GLSL/GL mark the binding readonly. */
    uint32_t accessUsage = 0u;
    if (!mglRenderTextureUsageForAccess((uint32_t)tex->access, &accessUsage)) {
            NSLog(@"MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u",
                  tex->access,
                  tex->name);
            return nil;
    }
    tex_desc.usage = accessUsage;

    /* Metal 3.1 imageAtomic* requires ShaderAtomic on R32{U,S}int textures. */
    if (mglRenderPixelFormatNeedsShaderAtomic(pixelFormat)) {
        tex_desc.usage |= MGL_TEXTURE_USAGE_SHADER_ATOMIC;
    }

    if (tex->is_render_target)
    {
        tex_desc.usage |= MGL_TEXTURE_USAGE_RENDER_TARGET | MGL_TEXTURE_USAGE_SHADER_READ;
    }

    // Allow safe same-memory format reinterpretation (e.g. RGBA8 <-> BGRA8)
    // for blit/present paths where OpenGL attachments and drawable formats differ.
    tex_desc.usage |= MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;

    if (tex_desc.texture_type == MGLTextureTypeCube || tex_desc.texture_type == MGLTextureTypeCubeArray) {
        NSLog(@"MGL CUBE DESC tex=%u glTarget=0x%x type=%lu width=%lu height=%lu depth=%lu arrayLength=%lu pixelFormat=%lu usage=%lu storage=%lu mipmapped=%d",
              tex->name,
              tex->target,
              (unsigned long)tex_desc.texture_type,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height,
              (unsigned long)tex_desc.depth,
              (unsigned long)tex_desc.array_length,
              (unsigned long)tex_desc.pixel_format,
              (unsigned long)tex_desc.usage,
              (unsigned long)tex_desc.storage_mode,
              (int)mipmapped);
    }

    if (tex->params.swizzled && !usesUploadSwizzleBake &&
        !tex->is_render_target)
    {
        mglTextureSwizzleDescriptor(&tex_desc, tex);
    }

    id texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    @try {
        texture = mglTextureCreateTexture(_device, &tex_desc);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating texture: %@", exception);
        mglRendererRecordGPUError((__bridge void *)self);
        return NULL;
    }

    // CRITICAL FIX: Validate texture creation result instead of asserting
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create Metal texture with descriptor");
        return NULL;
    }

    BOOL cpuUploadRequired =
        ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    BOOL cpuUploadVerified = !cpuUploadRequired;
    BOOL allLevelsUploaded = YES;

    if (cpuUploadRequired)
    {
        if (!mglTextureUploadDirty(
                (__bridge void *)self, tex, (__bridge void *)texture, pixelFormat,
                num_faces, upload_level_count, is_array, texture1DBackedBy2D,
                texture1DArrayBackedBy2DArray, tex_type, &allLevelsUploaded)) {
            return nil;
        }
    }
    else
    {
        if (hasUploadableCPUData) {
            mglTextureReUploadExisting((__bridge void *)self, tex, (__bridge void *)texture,
                                        pixelFormat, num_faces, upload_level_count,
                                        is_array, texture1DBackedBy2D,
                                        texture1DArrayBackedBy2DArray, tex_type);
        } else if (tex->is_render_target || mglMetalPixelFormatIsDepthOrStencil(pixelFormat)) {
            static uint64_t s_skipRenderTargetFillLogs = 0;
            uint64_t hit = ++s_skipRenderTargetFillLogs;
            if (hit <= 8ull || (hit % 2048ull) == 0ull) {
                NSLog(@"MGL TEXTURE SKIP implicit fill tex=%u renderTarget=%u format=%lu sourceSafe=0 hit=%llu",
                      (unsigned)tex->name,
                      (unsigned)tex->is_render_target,
                      (unsigned long)pixelFormat,
                      (unsigned long long)hit);
            }
        } else {
            mglTextureFillSafeInitialContents((__bridge void *)self, (__bridge void *)texture, tex,
                                          pixelFormat);
        }
    }

    if (cpuUploadRequired && mglRenderTextureTargetIs2D((uint32_t)tex->target) &&
        mglTextureInfo(texture).texture_type == MGLTextureType2D &&
        !mglTextureUploadNeedsSwizzleBake(tex)) {
        BOOL fullCPUUploadVerified = mglTextureUploadFullCPUData(
            (__bridge void *)self, tex, (__bridge void *)texture,
            "createMTLTexture.cpuData");
        cpuUploadVerified = allLevelsUploaded && fullCPUUploadVerified;
    } else if (cpuUploadRequired) {
        /*
         * Non-2D uploads still use the legacy creation path above. The current GUI
         * atlas failure is 2D; avoid changing array/cube semantics in this pass.
         * If any mip level was skipped (invalid layout, NULL data, etc.) keep
         * DIRTY_TEXTURE_DATA set so the level gets retried on next bind.
         */
        cpuUploadVerified = allLevelsUploaded;
    }

    if (cpuUploadRequired && !cpuUploadVerified) {
        static uint64_t s_createTextureCPUUploadIncompleteLogs = 0;
        uint64_t hit = ++s_createTextureCPUUploadIncompleteLogs;
        if (hit <= 8ull || (hit % 2048ull) == 0ull) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(tex);
            NSLog(@"MGL TEXTURE CREATE CPU-UPLOAD INCOMPLETE tex=%u target=0x%x dirtyBefore=0x%x level0=%ux%u source=%u upload=%lu hit=%llu",
                  (unsigned)tex->name,
                  (unsigned)tex->target,
                  (unsigned)tex->dirty_bits,
                  level0 ? (unsigned)level0->width : 0u,
                  level0 ? (unsigned)level0->height : 0u,
                  level0 ? (unsigned)level0->last_init_source : 0u,
                  (unsigned long)(level0 ? level0->last_upload_size : 0u),
                  (unsigned long long)hit);
        }
        tex->dirty_bits &= ~(DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_ACCESS);
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    } else {
        tex->dirty_bits = 0;
    }

    mglTextureLogMipDiagnostics((__bridge void *)self, tex,
                                (__bridge void *)texture,
                                effective_mipmap_levels);

    mglRendererRecordGPUSuccess((__bridge void *)self);

    return texture;
}





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
