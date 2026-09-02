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

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Texture_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"

enum {
    MGL_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
    MGL_TEXTURE_STORAGE_PRIVATE = 2u,
    MGL_TEXTURE_CPU_CACHE_DEFAULT = 0u,
    MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED = 1u,
    MGL_TEXTURE_USAGE_SHADER_READ = 1u,
    MGL_TEXTURE_USAGE_SHADER_WRITE = 2u,
    MGL_TEXTURE_USAGE_RENDER_TARGET = 4u,
    MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW = 16u,
};

typedef void (^MGLTextureCommandCompletionBlock)(
    const MGLRenderCommandBufferState *state);

static void mglTextureCommandCompletionCallback(
    void *context,
    const MGLRenderCommandBufferState *state)
{
    MGLTextureCommandCompletionBlock block =
        (__bridge MGLTextureCommandCompletionBlock)context;
    if (block) block(state);
}

static void mglTextureCommandCompletionDestroy(void *context)
{
    if (!context) return;
    (void)CFBridgingRelease(context);
}

static int mglTextureAddCommandBufferCompletion(
    void *commandBuffer,
    MGLTextureCommandCompletionBlock block)
{
    if (!commandBuffer || !block) return -1;
    MGLTextureCommandCompletionBlock copied = [block copy];
    void *context = (__bridge_retained void *)copied;
    int result = mglRenderAddCommandBufferCompletion(
        commandBuffer,
        mglTextureCommandCompletionCallback,
        context,
        mglTextureCommandCompletionDestroy);
    if (result != 0) mglTextureCommandCompletionDestroy(context);
    return result;
}

static MGLRegionValue mglRendererObjCRegion(int32_t x, int32_t y,
                                         int32_t width, int32_t height)
{
    return (MGLRegionValue){
        .origin = {(int64_t)x, (int64_t)y, 0},
        .size = {(uint64_t)width, (uint64_t)height, 1u},
    };
}

static MGLOriginValue mglTextureOrigin(uint64_t x, uint64_t y, uint64_t z)
{ return (MGLOriginValue){(int64_t)x, (int64_t)y, (int64_t)z}; }
static MGLSizeValue mglTextureSize(uint64_t width, uint64_t height, uint64_t depth)
{ return (MGLSizeValue){width, height, depth}; }
static MGLRegionValue mglTextureRegion1D(uint64_t x, uint64_t width)
{ return (MGLRegionValue){mglTextureOrigin(x, 0, 0), mglTextureSize(width, 1, 1)}; }
static MGLRegionValue mglTextureRegion2D(uint64_t x, uint64_t y,
                                         uint64_t width, uint64_t height)
{ return (MGLRegionValue){mglTextureOrigin(x, y, 0), mglTextureSize(width, height, 1)}; }
static MGLRegionValue mglTextureRegion3D(uint64_t x, uint64_t y, uint64_t z,
                                         uint64_t width, uint64_t height, uint64_t depth)
{ return (MGLRegionValue){mglTextureOrigin(x, y, z), mglTextureSize(width, height, depth)}; }

void mglRendererObjCReadDrawable(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlReadDrawable:glm_ctx pixelBytes:pixel_bytes
                  bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                   fromRegion:mglRendererObjCRegion(x, y, width, height)];
}

void mglRendererObjCReadIntegerPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlReadIntegerPixels:glm_ctx pixelBytes:pixel_bytes
                       bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                        fromRegion:mglRendererObjCRegion(x, y, width, height)
                            format:format type:type];
}

void mglRendererObjCReadDepthPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlReadDepthPixels:glm_ctx pixelBytes:pixel_bytes
                     bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                      fromRegion:mglRendererObjCRegion(x, y, width, height)];
}

void mglRendererObjCGetTexImage(GLMContext glm_ctx, Texture *texture,
    void *pixel_bytes, uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlGetTexImage:glm_ctx tex:texture pixelBytes:pixel_bytes
                 bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                  fromRegion:mglRendererObjCRegion(x, y, width, height)
                      format:format type:type mipmapLevel:level slice:slice];
}

void mglRendererObjCGenerateMipmaps(GLMContext glm_ctx, Texture *texture)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlGenerateMipmaps:glm_ctx forTexture:texture];
}

void mglRendererObjCTexSubImage(GLMContext glm_ctx, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlTexSubImage:glm_ctx tex:texture buf:buffer
                  src_offset:source_offset src_pitch:source_pitch
              src_image_size:source_image_size src_size:source_size
                       slice:slice level:level width:width height:height
                       depth:depth xoffset:x_offset yoffset:y_offset
                     zoffset:z_offset];
}

bool mglRendererObjCTexSubImageBytes(GLMContext glm_ctx, Texture *texture,
    const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return false;
    return [renderer mtlTexSubImageBytes:glm_ctx tex:texture
                                    bytes:bytes bytesSize:bytes_size
                               src_offset:source_offset src_pitch:source_pitch
                           src_image_size:source_image_size
                                    slice:slice level:level
                                    width:width height:height depth:depth
                                  xoffset:x_offset yoffset:y_offset
                                  zoffset:z_offset];
}

void mglRendererObjCCopyTexSubImage(GLMContext glm_ctx, Texture *texture,
    uint32_t slice, int32_t level, int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlCopyTexSubImage:glm_ctx tex:texture slice:slice
                    mipmapLevel:level xoffset:x_offset yoffset:y_offset
                              x:x y:y width:width height:height];
}

void mglRendererObjCCopyImageSubData(GLMContext glm_ctx, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlCopyImageSubData:glm_ctx srcTexture:source_texture
                         srcLevel:source_level srcX:source_x srcY:source_y
                             srcZ:source_z dstTexture:destination_texture
                         dstLevel:destination_level dstX:destination_x
                             dstY:destination_y dstZ:destination_z
                            width:width height:height depth:depth];
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

static id mglTextureCreateBufferTexture(
    id buffer,
    const MGLRenderTextureDescriptorState *descriptor,
    NSUInteger offset,
    NSUInteger bytesPerRow)
{
    void *texture = NULL;
    if (mglRenderCreateBufferTextureFromState(
            (__bridge void *)buffer, descriptor,
            offset, bytesPerRow,
            &texture) == 0 && texture) {
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

static void mglTextureGetBytes(id texture,
                               void *bytes,
                               NSUInteger bytesPerRow,
                               NSUInteger bytesPerImage,
                               MGLRegionValue region,
                               NSUInteger level,
                               NSUInteger slice,
                               BOOL useSlice)
{
    if (mglRenderTextureGetBytes(
            (__bridge void *)texture, bytes, bytesPerRow, bytesPerImage,
            region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth,
            level, slice, useSlice ? 1 : 0) != 0) {
        [NSException raise:@"MGLTextureGetBytesError"
                    format:@"C++ texture getBytes failed (level=%lu slice=%lu)",
                           (unsigned long)level, (unsigned long)slice];
    }
}

static id mglTextureCreateSampler(id device)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return (__bridge_transfer id)sampler;
    }
    return nil;
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
static id mglTextureCreateCurrentBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

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
        tex->internalformat != GL_DEPTH32F_STENCIL8 ||
        pixelFormat != MGLPixelFormatDepth32Float_Stencil8 ||
        srcBytesPerRow < width * 5u || srcBytesPerRow >= width * 8u) {
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

static uint32_t mglDepthStencilPlaneViewType(uint32_t parentType)
{
    switch (parentType) {
        case MGLTextureType2DArray:
        case MGLTextureTypeCube:
        case MGLTextureTypeCubeArray:
        case MGLTextureType1DArray:
        case MGLTextureType2DMultisampleArray:
        case MGLTextureType3D:
            return MGLTextureType2D;
        default:
            return parentType;
    }
}

static uint64_t mglTextureBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo((__bridge void *)buffer, &info) == 0
        ? info.length : 0u;
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

- (bool)uploadPackedDepthStencilStencilPlane:(id)texture
                                     texName:(GLuint)texName
                                       bytes:(const void *)packedBytes
                                       width:(NSUInteger)width
                                      height:(NSUInteger)height
                                 bytesPerRow:(NSUInteger)bytesPerRow
                                       level:(NSUInteger)level
                                       slice:(NSUInteger)slice
                                      xorigin:(NSUInteger)xorigin
                                      yorigin:(NSUInteger)yorigin
{
    if (!texture || !packedBytes || width == 0 || height == 0) {
        return false;
    }
    const uint32_t parentFormat =
        (uint32_t)mglTextureInfo(texture).pixel_format;
    if (parentFormat != MGLPixelFormatDepth32Float_Stencil8 &&
        parentFormat != MGLPixelFormatDepth24Unorm_Stencil8) {
        return false;
    }

    void *metalUpload = NULL;
    const void *srcBytes = packedBytes;
    NSUInteger srcBytesPerRow = bytesPerRow;
    if (bytesPerRow >= width * 8u) {
        /* Already Metal packed layout. */
    } else if (bytesPerRow >= width * 5u &&
               parentFormat == MGLPixelFormatDepth32Float_Stencil8) {
        srcBytesPerRow = width * 8u;
        const NSUInteger repackBytes = srcBytesPerRow * height;
        metalUpload = calloc(1u, repackBytes);
        if (!metalUpload) return false;
        const uint8_t *srcBase = (const uint8_t *)packedBytes;
        uint8_t *dstBase = (uint8_t *)metalUpload;
        for (NSUInteger y = 0; y < height; ++y) {
            const uint8_t *srcRow = srcBase + y * bytesPerRow;
            uint8_t *dstRow = dstBase + y * srcBytesPerRow;
            for (NSUInteger x = 0; x < width; ++x) {
                memcpy(dstRow + x * 8u, srcRow + x * 5u, 4u);
                dstRow[x * 8u + 4u] = srcRow[x * 5u + 4u];
            }
        }
        srcBytes = metalUpload;
    } else {
        return false;
    }

    const NSUInteger logicalStencilBytesPerRow = width;
    const NSUInteger stencilBytesPerRow =
        mglDepthStencilAlignedBytesPerRow(logicalStencilBytesPerRow);
    if (stencilBytesPerRow == 0) {
        free(metalUpload);
        return false;
    }
    const NSUInteger stencilBytesPerImage = stencilBytesPerRow * height;
    uint8_t *stencilBytes = (uint8_t *)calloc(1u, stencilBytesPerImage);
    if (!stencilBytes) {
        free(metalUpload);
        return false;
    }

    const uint8_t *srcBase = (const uint8_t *)srcBytes;
    for (NSUInteger y = 0; y < height; ++y) {
        const uint8_t *srcRow = srcBase + y * srcBytesPerRow;
        uint8_t *dstRow = stencilBytes + y * stencilBytesPerRow;
        for (NSUInteger x = 0; x < width; ++x) {
            dstRow[x] = srcRow[x * 8u + 4u];
        }
    }
    free(metalUpload);

    void *stencilViewRaw = NULL;
    const uint32_t viewType = mglDepthStencilPlaneViewType(
        (uint32_t)mglTextureInfo(texture).texture_type);
    bool uploaded = false;
    const uint32_t stencilViewFormat =
        parentFormat == MGLPixelFormatDepth24Unorm_Stencil8
            ? MGLPixelFormatX24_Stencil8
            : MGLPixelFormatX32_Stencil8;
    if (mglRenderCreateTextureViewRange(
            (__bridge void *)texture,
            stencilViewFormat, viewType,
            level, 1u, slice, 1u, 0, 0, 0, 0, 0,
            &stencilViewRaw) == 0 && stencilViewRaw) {
        id stencilView = (__bridge_transfer id)stencilViewRaw;
        @try {
            mglTextureReplaceRegion(
                stencilView,
                mglTextureRegion2D(xorigin, yorigin, width, height),
                0u, 0u, stencilBytes, stencilBytesPerRow,
                stencilBytesPerImage, NO);
            uploaded = true;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: depth/stencil stencil-plane upload failed tex=%u slice=%lu: %@",
                  (unsigned)texName, (unsigned long)slice,
                  exception.reason);
        }
    }
    free(stencilBytes);
    if (!uploaded) {
        NSLog(@"MGL WARNING: depth/stencil stencil-plane blit upload failed tex=%u slice=%lu",
              (unsigned)texName, (unsigned long)slice);
    }
    return uploaded;
}

- (bool)copyTextureUploadWithDedicatedCommandBuffer:(id)sourceBuffer
                                        sourceOffset:(NSUInteger)sourceOffset
                                   sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
                                 sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                                  sourceLayerStride:(NSUInteger)sourceLayerStride
                                          layerCount:(NSUInteger)layerCount
                                           sourceSize:(MGLSizeValue)sourceSize
                                            toTexture:(id)texture
                                     destinationSlice:(NSUInteger)destinationSlice
                                     destinationLevel:(NSUInteger)destinationLevel
                                    destinationOrigin:(MGLOriginValue)destinationOrigin
                                               reason:(const char *)reason
{
    MGL_ASSERT_GL_THREAD();
    if (!sourceBuffer || !texture || !_commandQueue || layerCount == 0u ||
        sourceBytesPerRow == 0u || sourceBytesPerImage == 0u ||
        sourceSize.width == 0u || sourceSize.height == 0u ||
        sourceSize.depth == 0u ||
        (layerCount > 1u && sourceLayerStride == 0u)) {
        NSLog(@"MGL ERROR: dedicated texture upload prerequisites missing (source=%p texture=%p queue=%p)",
              sourceBuffer, texture, _commandQueue);
        return false;
    }

    if (!kMGLUseDedicatedTextureUploadCommandBuffer) {
        /*
         * Texture uploads are GL commands and must stay ordered with draws in the
         * same context.  Committing a standalone upload command buffer here can
         * leapfrog an open render command buffer, so encode the blit into the
         * current command buffer after closing the active render encoder.
         */
        [self endRenderEncoding];

        if (![self ensureWritableCommandBuffer:reason ? reason : "texture_upload"]) {
            NSLog(@"MGL ERROR: failed to obtain current command buffer for %s",
                  reason ? reason : "texture_upload");
            return false;
        }

        if (mglRenderEncodeTextureUploadLayersForCommandBufferOwner(
                _commandState.currentCommandBufferOwner,
                (__bridge void *)sourceBuffer, sourceOffset,
                sourceBytesPerRow, sourceBytesPerImage, sourceLayerStride,
                sourceSize.width, sourceSize.height, sourceSize.depth,
                (__bridge void *)texture, destinationSlice, layerCount,
                destinationLevel, destinationOrigin.x,
                destinationOrigin.y, destinationOrigin.z) != 0) {
            NSLog(@"MGL ERROR: C++ ordered upload encode failed (%s)",
                  reason ? reason : "texture_upload");
            [self recordGPUError];
            return false;
        }

        return true;
    }

    id uploadCB = mglTextureCreateCommandBuffer(_commandQueue);
    if (!uploadCB) {
        NSLog(@"MGL ERROR: failed to create dedicated upload command buffer for %s",
              reason ? reason : "texture_upload");
        [self recordGPUError];
        return false;
    }

    if (reason) {
        NSString *label = [NSString stringWithFormat:@"MGL.%s", reason];
        (void)mglRenderSetCommandBufferLabel(
            (__bridge void *)uploadCB, label.UTF8String);
    } else {
        (void)mglRenderSetCommandBufferLabel(
            (__bridge void *)uploadCB, "MGL.texture_upload");
    }

    if (mglRenderEncodeTextureUploadLayers(
            (__bridge void *)uploadCB, (__bridge void *)sourceBuffer,
            sourceOffset, sourceBytesPerRow, sourceBytesPerImage,
            sourceLayerStride,
            sourceSize.width, sourceSize.height, sourceSize.depth,
            (__bridge void *)texture, destinationSlice, layerCount,
            destinationLevel, destinationOrigin.x,
            destinationOrigin.y, destinationOrigin.z) != 0) {
        NSLog(@"MGL ERROR: C++ dedicated upload encode failed (%s)",
              reason ? reason : "texture_upload");
        [self recordGPUError];
        return false;
    }

    dispatch_semaphore_t completionSemaphore = kMGLSynchronizeTextureUploads
        ? dispatch_semaphore_create(0)
        : NULL;
    __block BOOL uploadError = NO;
    __weak typeof(self) weakSelf = self;
    mglTextureAddCommandBufferCompletion(
        (__bridge void *)uploadCB,
        ^(const MGLRenderCommandBufferState *uploadState) {
        if (uploadState->has_error) {
            uploadError = YES;
            NSLog(@"MGL ERROR: dedicated upload command buffer failed (%s): %s",
                  reason ? reason : "texture_upload",
                  mglRenderCommandBufferErrorDescription(uploadState));
            [weakSelf recordGPUError];
        }

        if (completionSemaphore) {
            dispatch_semaphore_signal(completionSemaphore);
        }
    });

    mglTextureCommitCommandBuffer(uploadCB);

    if (!kMGLSynchronizeTextureUploads) {
        // Keep uploads ordered on the same queue but avoid stalling the render thread.
        return true;
    }

    dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW,
                                             (int64_t)(kMGLTextureUploadWaitTimeoutSeconds * NSEC_PER_SEC));
    if (dispatch_semaphore_wait(completionSemaphore, deadline) != 0) {
        NSLog(@"MGL WARNING: dedicated upload wait timed out (%s), continuing asynchronously",
              reason ? reason : "texture_upload");
        return true;
    }

    return !uploadError;
}

- (bool)uploadTextureSliceViaBlit:(id)texture
                          texName:(GLuint)texName
                         texTarget:(GLenum)texTarget
                            bytes:(const void *)bytes
                      bytesPerRow:(NSUInteger)bytesPerRow
                    bytesPerImage:(NSUInteger)bytesPerImage
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                            depth:(NSUInteger)depth
                            level:(NSUInteger)level
                            slice:(NSUInteger)slice
{
    if (!texture || !bytes || bytesPerRow == 0 || bytesPerImage == 0 || width == 0) {
        return false;
    }


    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: Skipping texture upload during recovery");
        return false;
    }

    uint32_t textureType = mglTextureInfo(texture).texture_type;
    MGLRenderTextureUploadPlan uploadPlan = {0};
    if (mglRenderBuildTextureUploadPlan(
            (uint32_t)texTarget, (uint32_t)textureType,
            (uint32_t)mglTextureInfo(texture).usage, (uint32_t)mglTextureInfo(texture).pixel_format,
            MGLCapabilityHasBug(&_capability,
                                MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB) ? 1 : 0,
            width, height, depth, bytesPerRow, bytesPerImage, level, slice,
            &uploadPlan) != 0) {
        NSLog(@"MGL WARNING: Rejecting invalid texture upload plan (tex=%u target=0x%x level=%lu slice=%lu)",
              (unsigned)texName, (unsigned)texTarget, (unsigned long)level,
              (unsigned long)slice);
        return false;
    }

    if (mglTraceLogIsEnabled() &&
        (mglTextureInfo(texture).pixel_format == MGLPixelFormatDepth32Float_Stencil8 ||
         mglTextureInfo(texture).pixel_format == MGLPixelFormatDepth24Unorm_Stencil8) &&
        (texTarget == GL_TEXTURE_2D_ARRAY || texTarget == GL_TEXTURE_3D) &&
        bytesPerRow >= 16) {
        const uint8_t *probe = (const uint8_t *)bytes;
        mglTraceLog("TEXTURE_UPLOAD_DS tex=%u target=0x%x fmt=%lu slice=%lu level=%lu size=%lux%lu bpr=%lu bpi=%lu first=%02x %02x %02x %02x %02x %02x %02x %02x next=%02x %02x %02x %02x %02x %02x %02x %02x",
                    (unsigned)texName, (unsigned)texTarget,
                    (unsigned long)mglTextureInfo(texture).pixel_format,
                    (unsigned long)slice, (unsigned long)level,
                    (unsigned long)width, (unsigned long)height,
                    (unsigned long)bytesPerRow, (unsigned long)bytesPerImage,
                    probe[0], probe[1], probe[2], probe[3], probe[4], probe[5], probe[6], probe[7],
                    probe[8], probe[9], probe[10], probe[11], probe[12], probe[13], probe[14], probe[15]);
    }

    if (textureType == MGLTextureTypeCube || textureType == MGLTextureTypeCubeArray) {
        static uint64_t s_cubeUploadLogs = 0;
        uint64_t hit = ++s_cubeUploadLogs;
        if (hit <= 4ull || (hit % 2048ull) == 0ull) {
            NSLog(@"MGL CUBE UPLOAD tex=%u glTarget=0x%x face=%lu slice=%lu level=%lu origin=(0,0,0) size=%lux%lux%lu bpr=%lu bpi=%lu ptr=%p",
                  texName,
                  texTarget,
                  (unsigned long)slice,
                  (unsigned long)slice,
                  (unsigned long)level,
                  (unsigned long)width,
                  (unsigned long)uploadPlan.normalized_height,
                  (unsigned long)uploadPlan.copy_depth,
                  (unsigned long)bytesPerRow,
                  (unsigned long)uploadPlan.normalized_bytes_per_image,
                  bytes);
        }
    }

    const uint32_t uploadRoute = uploadPlan.route;

    /* Shared packed depth/stencil textures can be updated directly for the
     * depth plane.  AGX requires a separate X32_Stencil8 view upload for the
     * stencil plane, using a 2D view over the selected array slice. */
    if ((mglTextureInfo(texture).pixel_format == MGLPixelFormatDepth32Float_Stencil8 ||
         mglTextureInfo(texture).pixel_format == MGLPixelFormatDepth24Unorm_Stencil8) &&
        mglTextureInfo(texture).storage_mode != MGL_TEXTURE_STORAGE_PRIVATE) {
        bool uploaded = false;
        @try {
            mglTextureReplaceRegion(texture,
                                    mglTextureRegion2D(0, 0, width, uploadPlan.normalized_height),
                                    level, slice, bytes, bytesPerRow,
                                    uploadPlan.normalized_bytes_per_image, YES);
            uploaded = true;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: depth/stencil replaceRegion upload failed tex=%u: %@",
                  (unsigned)texName, exception.reason);
        }
        if (uploaded && bytesPerRow >= width * 5u) {
            uploaded = [self uploadPackedDepthStencilStencilPlane:texture
                                                          texName:texName
                                                            bytes:bytes
                                                            width:width
                                                           height:uploadPlan.normalized_height
                                                      bytesPerRow:bytesPerRow
                                                            level:level
                                                            slice:slice
                                                           xorigin:0u
                                                           yorigin:0u];
        }
        return uploaded;
    }

    /* 1D texture upload via replaceRegion branch:
     * - 1D textures are a low-frequency update path; replaceRegion is safe in this scenario;
     * - Before entering this function, the caller has already flushed CPU-side deferred
     *   draws via mglFlushPendingDrawsBeforeTextureWrite, avoiding ordering races between
     *   the upload and uncommitted render command buffers;
     * - Only available for shared storage; Private storage (e.g. MSAA) must fall back to the blit path. */
    if (uploadRoute == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D) {
        @try {
            MGLRegionValue region = uploadPlan.replace_region_dimension == 1u
                ? mglTextureRegion1D(0, width)
                : mglTextureRegion2D(0, 0, width,
                                  uploadPlan.normalized_height);
            mglTextureReplaceRegion(
                texture, region, uploadPlan.destination_level,
                uploadPlan.destination_slice, bytes, bytesPerRow,
                uploadPlan.normalized_bytes_per_image,
                uploadPlan.replace_use_slice != 0u);
            if (mglTraceLogIsEnabled() &&
                mglTextureInfo(texture).pixel_format == MGLPixelFormatR8Unorm &&
                width > 0) {
                const uint8_t *first = (const uint8_t *)bytes;
                mglTraceLog("TEXTURE_UPLOAD_1D_REPLACE tex=%u target=0x%x mtlType=%lu size=%lux%lu bpr=%lu bpi=%lu first=%u",
                            (unsigned)texName,
                            (unsigned)texTarget,
                            (unsigned long)textureType,
                            (unsigned long)width,
                            (unsigned long)uploadPlan.normalized_height,
                            (unsigned long)bytesPerRow,
                            (unsigned long)uploadPlan.normalized_bytes_per_image,
                            first ? first[0] : 0u);
            }
            return true;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: 1D texture replaceRegion upload failed, falling back to blit (tex=%u level=%lu slice=%lu): %@",
                  (unsigned)texName,
                  (unsigned long)level,
                  (unsigned long)slice,
                  exception.reason);
        }
    }

    /* 3D texture upload via replaceRegion branch:
     * - 3D uses replaceRegion to work around the AGX driver's copyFromBuffer:toTexture: slice OOB
     *   assertion (triggered even when destinationSlice=0);
     *   Driver bug tracked via MGLCapabilityHasBug(MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB).
     * - Metal requires bytesPerImage for 3D replaceRegion uploads, so padded
     *   depth planes are repacked and uploaded with the tight image stride.
     * - Only shared storage supports replaceRegion.  Do not fall back to the
     *   known-bad copyFromBuffer path while the AGX bug marker is active.
     * - The route already rejects private storage while the bug marker is up;
     *   reaching this branch means the upload is shared and must use
     *   replaceRegion (never the known-bad blit). */
    if (uploadRoute == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D) {
        const void *replaceBytes = bytes;
        void *tightlyPackedBytes = NULL;
        if (uploadPlan.requires_repack) {

            tightlyPackedBytes = mglRenderTextureRepackDepthPlanes(
                bytes, uploadPlan.normalized_bytes_per_image,
                uploadPlan.expected_bytes_per_image, uploadPlan.copy_depth);
            if (!tightlyPackedBytes) {
                return false;
            }
            replaceBytes = tightlyPackedBytes;
        }

        @try {
            MGLRegionValue region = mglTextureRegion3D(
                0, 0, 0, width, uploadPlan.normalized_height,
                uploadPlan.copy_depth);
            mglTextureReplaceRegion(
                texture, region, uploadPlan.destination_level,
                uploadPlan.destination_slice, replaceBytes, bytesPerRow,
                uploadPlan.expected_bytes_per_image, YES);
            free(tightlyPackedBytes);
            return true;
        } @catch (NSException *exception) {
            free(tightlyPackedBytes);
            NSLog(@"MGL WARNING: 3D texture replaceRegion upload failed (tex=%u level=%lu): %@",
                  (unsigned)texName, (unsigned long)level,
                  exception.reason);
            return false;
        }
    }

    /* 3D + Private while the AGX copyFromBuffer workaround is required:
     * rejected by the C++ route (blit is known-bad and replaceRegion does
     * not support private storage). */
    if (uploadRoute == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT) {
        NSLog(@"MGL WARNING: Rejecting private 3D upload while AGX copyFromBuffer workaround is required (tex=%u level=%lu)",
              (unsigned)texName, (unsigned long)level);
        return false;
    }

    /* 2D / 2DArray / Cube texture upload via blit path (dedicated CB + completion handler):
     * - replaceRegion must not be used: when the texture is being sampled by an in-flight
     *   command buffer, replaceRegion's CPU direct writes are not subject to GPU-side
     *   ordering constraints, causing data races with in-flight sampling draws (this
     *   previously caused Minecraft GUI item rendering corruption);
     * - The blit path is required to guarantee GPU-side ordering: copyTextureUploadWithDedicatedCommandBuffer
     *   calls endRenderEncoding to close the current render encoder, and encodes
     *   copyFromBuffer:toTexture: on a dedicated CB, with the Metal command queue
     *   guaranteeing submission order relative to existing render CBs;
     * - The 1D/3D branches do not hit this precondition (low-frequency or driver bug
     *   workaround) and have already returned earlier. */
    void *stagingOwner = NULL;
    void *borrowedStagingBuffer = NULL;
    __unsafe_unretained id uploadBuffer = nil;
    if (mglRenderCreateTextureStagingOwner(
            bytes, uploadPlan.buffer_size, MGL_TEXTURE_RESOURCE_STORAGE_SHARED,
            &stagingOwner, &borrowedStagingBuffer) == 0 &&
        stagingOwner && borrowedStagingBuffer) {
        uploadBuffer = (__bridge id)borrowedStagingBuffer;
    }
    if (!uploadBuffer) {
        mglRenderDestroyTextureStagingOwner(&stagingOwner);
        NSLog(@"MGL WARNING: Failed to allocate upload buffer for texture blit");
        return false;
    }

    bool uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:uploadBuffer
                                                         sourceOffset:0
                                                    sourceBytesPerRow:bytesPerRow
                                                  sourceBytesPerImage:uploadPlan.normalized_bytes_per_image
                                                   sourceLayerStride:0
                                                           layerCount:1
                                                            sourceSize:mglTextureSize(width, uploadPlan.normalized_height, uploadPlan.copy_depth)
                                                             toTexture:texture
                                                      destinationSlice:uploadPlan.destination_slice
                                                      destinationLevel:uploadPlan.destination_level
                                                     destinationOrigin:mglTextureOrigin(0, 0, 0)
                                                                reason:"texture_upload_blit"];
    /* The encoded blit retains its source resource until command-buffer
     * completion; release the C++ staging owner as soon as encoding ends. */
    mglRenderDestroyTextureStagingOwner(&stagingOwner);
    if (!uploaded) {
        NSLog(@"MGL WARNING: Dedicated texture upload failed (level=%lu slice=%lu)",
              (unsigned long)level, (unsigned long)slice);
        return false;
    }
    if ((mglTextureInfo(texture).pixel_format ==
             MGLPixelFormatDepth32Float_Stencil8 ||
         mglTextureInfo(texture).pixel_format ==
             MGLPixelFormatDepth24Unorm_Stencil8) &&
        bytesPerRow >= width * 5u) {
        (void)[self uploadPackedDepthStencilStencilPlane:texture
                                                 texName:texName
                                                   bytes:bytes
                                                   width:width
                                                  height:uploadPlan.normalized_height
                                             bytesPerRow:bytesPerRow
                                                   level:level
                                                   slice:slice
                                                  xorigin:0u
                                                  yorigin:0u];
    }
    return true;
}

- (bool)uploadFullCPUTextureDataIntoTexture:(Texture *)tex
                                      metal:(id)texture
                                     reason:(const char *)reason
{
    if (!tex || !texture || !tex->faces[0].levels) {
        return false;
    }
    if (tex->target != GL_TEXTURE_2D ||
        mglTextureInfo(texture).texture_type != MGLTextureType2D) {
        return false;
    }

    int numFaces = 1;
    GLuint levelCount = MIN((GLuint)mglTextureInfo(texture).mipmap_level_count,
                            tex->num_levels ? tex->num_levels : 1u);
    if (levelCount == 0u ||
        !mglTextureHasUploadableCPUData(tex, numFaces, levelCount)) {
        return false;
    }


    MGLRenderLevelUploadOp uploadOps[levelCount ? levelCount : 1u];
    uint32_t opCount = 0;
    uint32_t shortCount = 0;
    uint32_t badCount = 0;
    if (mglRenderBuildLevelUploadOps(
            tex->faces[0].levels, levelCount,
            (uint32_t)mglTextureInfo(texture).texture_type,
            (uint32_t)tex->internalformat,
            (uint32_t)mglTextureInfo(texture).pixel_format,
            uploadOps, levelCount,
            &opCount, &shortCount, &badCount) != 0) {
        return false;
    }

    bool uploadedAny = false;
    bool failedAny = (shortCount + badCount) > 0;
    for (uint32_t i = 0; i < opCount; i++) {
        MGLRenderLevelUploadOp *op = &uploadOps[i];
        if (op->kind == 1u) {
            static uint64_t s_shortBackingLogs = 0;
            uint64_t hit = ++s_shortBackingLogs;
            if (kMGLDiagnosticStateLogs &&
                (hit <= 32ull || (hit % 512ull) == 0ull)) {
                mglTraceLogNSString(@"MGL TEXTURE CPU-REFRESH skip short backing tex=%u level=%u face=0 have=%llu need=%llu reason=%s hit=%llu",
                              (unsigned)tex->name,
                              (unsigned)op->level,
                              (unsigned long long)op->available_bytes,
                              (unsigned long long)op->needed_bytes,
                              reason ? reason : "(null)",
                              (unsigned long long)hit);
            }
            continue;
        }

        bool uploaded = [self uploadTextureSliceViaBlit:texture
                                                texName:tex->name
                                             texTarget:tex->target
                                                 bytes:op->data
                                           bytesPerRow:(NSUInteger)op->bytes_per_row
                                         bytesPerImage:(NSUInteger)op->bytes_per_image
                                                 width:(NSUInteger)op->width
                                                height:(NSUInteger)op->height
                                                 depth:(NSUInteger)op->copy_depth
                                                 level:op->level
                                                 slice:0];
        if (op->owns_data) {
            free((void *)op->data);
        }
        if (uploaded) {
            uploadedAny = true;
        } else {
            failedAny = true;
        }
    }

    static uint64_t s_refreshLogs = 0;
    uint64_t hit = ++s_refreshLogs;
    if (kMGLDiagnosticStateLogs &&
        (uploadedAny || hit <= 32ull || (hit % 512ull) == 0ull)) {
        mglTraceLogNSString(@"MGL TEXTURE CPU-REFRESH tex=%u mtl=%p uploaded=%d failed=%d dirty=0x%x levels=%u reason=%s hit=%llu",
                      (unsigned)tex->name,
                      texture,
                      uploadedAny ? 1 : 0,
                      failedAny ? 1 : 0,
                      (unsigned)tex->dirty_bits,
                      (unsigned)levelCount,
                      reason ? reason : "(null)",
                      (unsigned long long)hit);
    }

    if (uploadedAny && !failedAny) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
        [self recordGPUSuccess];
        return true;
    }

    return false;
}

- (void)mglApplyPendingDefaultColorClearToTexture:(id)texture
{
    if (!ctx || !texture || !(ctx->state.default_fbo_clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    if (mglRenderEncodeColorClearForCommandBufferOwner(
            _commandState.currentCommandBufferOwner,
            (__bridge void *)texture, 0, 0, 0,
            ctx->state.default_clear_color[0],
            ctx->state.default_clear_color[1],
            ctx->state.default_clear_color[2],
            ctx->state.default_clear_color[3]) == 0) {
        ctx->state.default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        return;
    }
    NSLog(@"MGL WARNING: C++ default framebuffer color clear failed");
}

- (void)mglApplyPendingFBOColorClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(id)texture
                                  attachmentEnum:(GLenum)attachmentEnum
{
    (void)attachmentEnum;
    if (!fbo || !attachment || !texture || !(attachment->clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    if (mglRenderEncodeColorClearForCommandBufferOwner(
            _commandState.currentCommandBufferOwner,
            (__bridge void *)texture, subresource.level,
            subresource.slice, subresource.depthPlane,
            attachment->clear_color[0], attachment->clear_color[1],
            attachment->clear_color[2], attachment->clear_color[3]) == 0) {
        attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(
            textureObj, attachment->level);
        return;
    }
    NSLog(@"MGL WARNING: C++ readPixels FBO color clear failed fbo=%u",
          (unsigned)fbo->name);
}


- (id)readbackStageAndWaitTexture:(id)sourceTexture
                                 sourceLevel:(NSUInteger)sourceLevel
                                 sourceSlice:(NSUInteger)sourceSlice
                             sourceDepthPlane:(NSUInteger)sourceDepthPlane
                                   copyOrigin:(MGLOriginValue)copyOrigin
                                     copySize:(MGLSizeValue)copySize
                         stagingBytesPerRow:(NSUInteger)stagingBytesPerRow
                                stagingSize:(NSUInteger)stagingSize
                                     reason:(const char *)reason
                                    logKind:(const char *)logKind
                                     success:(BOOL *)outSuccess
{
    if (outSuccess) {
        *outSuccess = YES;
    }

    id readBuffer = mglTextureCreateBuffer(
        _device, stagingSize, MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
    id blitEncoder = readBuffer
        ? (__bridge id)mglRenderCreateBlitEncoderBorrowed(
              _commandState.currentCommandBufferOwner)
        : nil;
    if (!readBuffer || !blitEncoder) {
        NSLog(@"MGL WARNING: readPixels failed to create %s resources for %s",
              logKind ? logKind : "readback",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return nil;
    }

    BOOL blitEncoderEnded = NO;
    @try {
        mglTextureCopyTextureToBuffer(
            blitEncoder, sourceTexture, sourceSlice, sourceLevel,
            copyOrigin, copySize,
            readBuffer, 0u, stagingBytesPerRow, stagingSize);
        mglTextureEndBlitEncoder(blitEncoder);
        blitEncoderEnded = YES;
    } @catch (NSException *exception) {
        if (!blitEncoderEnded) {
            @try {
                mglTextureEndBlitEncoder(blitEncoder);
            } @catch (NSException *endException) {
                NSLog(@"MGL WARNING: readPixels failed to end %s blit encoder after copy exception for %s: %@",
                      logKind ? logKind : "readback",
                      reason ? reason : "unknown",
                      endException);
            }
        }
        NSLog(@"MGL WARNING: readPixels %s texture copy failed for %s: %@",
              logKind ? logKind : "readback",
              reason ? reason : "unknown",
              exception);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }

    id readbackCommandBuffer =
        (__bridge id)mglCmdDetachCurrentCommandBufferForSubmission(&_commandState);
    MGLRenderCommandBufferTransaction readbackTransaction = {0};
    int readbackTransactionResult = mglCmdCommitCommandBufferTransaction(
        &_commandState, (__bridge void *)readbackCommandBuffer,
        _gpuRecovery.commandRecoveryOwner, YES, &readbackTransaction);
    if (readbackTransactionResult != 0 || readbackTransaction.has_error) {
        NSLog(@"MGL WARNING: readPixels %s owner transaction failed for %s",
              logKind ? logKind : "readback", reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        if (outSuccess) {
            *outSuccess = NO;
        }
    }

    if (readbackTransactionResult == 0 &&
        readbackTransaction.completion.has_error) {
        NSLog(@"MGL WARNING: readPixels %s command buffer failed for %s: %s; returning zeroed data",
              logKind ? logKind : "readback",
              reason ? reason : "unknown",
              mglRenderCommandBufferErrorDescription(
                  &readbackTransaction.completion));
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        if (outSuccess) {
            *outSuccess = NO;
        }
    }

    mglCmdReleaseDetachedCommandBufferIfOwned(&_commandState, (__bridge void *)readbackCommandBuffer);
    [self newCommandBuffer];
    return readBuffer;
}

- (BOOL)mglReadColorTextureAsBGRA8:(id)sourceTexture
                       sourceLevel:(NSUInteger)sourceLevel
                       sourceSlice:(NSUInteger)sourceSlice
                   sourceDepthPlane:(NSUInteger)sourceDepthPlane
                         pixelBytes:(void *)pixelBytes
                        bytesPerRow:(NSUInteger)bytesPerRow
                      bytesPerImage:(NSUInteger)bytesPerImage
                         fromRegion:(MGLRegionValue)region
                             reason:(const char *)reason
{
    NSUInteger readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return NO;
    }

    if (!sourceTexture || region.size.width == 0u || region.size.height == 0u) {
        return sourceTexture != nil;
    }

    if (mglRenderTextureIsFramebufferOnly((__bridge void *)sourceTexture)) {
        static uint64_t s_framebufferOnlyReadCount = 0;
        uint64_t hit = ++s_framebufferOnlyReadCount;
        if (hit <= 16ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels cannot read framebufferOnly texture for %s hit=%llu",
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (!mglMetalReadbackFormatIsBGRA8Compatible(mglTextureInfo(sourceTexture).pixel_format)) {
        static uint64_t s_unsupportedReadFormatCount = 0;
        uint64_t hit = ++s_unsupportedReadFormatCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels unsupported Metal color readback format=%lu for %s hit=%llu; returning zero data",
                  (unsigned long)mglTextureInfo(sourceTexture).pixel_format,
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (mglTextureInfo(sourceTexture).sample_count > 1u) {
        sourceTexture = [self resolvedReadbackTextureForMultisampleTexture:sourceTexture
                                                               sourceLevel:sourceLevel
                                                               sourceSlice:sourceSlice
                                                           sourceDepthPlane:sourceDepthPlane
                                                                    reason:reason];
        if (!sourceTexture) {
            return NO;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
    }

    if (bytesPerRow < region.size.width * 4u) {
        NSLog(@"MGL WARNING: readPixels destination row too small row=%lu width=%lu for %s",
              (unsigned long)bytesPerRow,
              (unsigned long)region.size.width,
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    NSUInteger levelWidth = mglTextureInfo(sourceTexture).width;
    NSUInteger levelHeight = mglTextureInfo(sourceTexture).height;
    if (sourceLevel > 0u) {
        if (sourceLevel >= mglTextureInfo(sourceTexture).mipmap_level_count) {
            NSLog(@"MGL WARNING: readPixels invalid mip level=%lu mipLevels=%lu for %s",
                  (unsigned long)sourceLevel,
                  (unsigned long)mglTextureInfo(sourceTexture).mipmap_level_count,
                  reason ? reason : "unknown");
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NO;
        }
        levelWidth = MAX((NSUInteger)1u, mglTextureInfo(sourceTexture).width >> sourceLevel);
        levelHeight = MAX((NSUInteger)1u, mglTextureInfo(sourceTexture).height >> sourceLevel);
    }


    MGLRenderReadTextureRegionClip clip = {0};
    mglRenderReadTextureRegionClip(
        (int64_t)region.origin.x, (int64_t)region.origin.y,
        (int64_t)region.size.width, (int64_t)region.size.height,
        (int64_t)levelWidth, (int64_t)levelHeight, &clip);
    NSInteger copyW = (NSInteger)clip.copy_w;
    NSInteger copyH = (NSInteger)clip.copy_h;
    NSInteger dstX = (NSInteger)clip.dst_x;
    NSInteger dstY = (NSInteger)clip.dst_y;
    NSInteger metalSrcX = (NSInteger)clip.metal_src_x;
    NSInteger metalSrcY = (NSInteger)clip.metal_src_y;
    if (clip.empty) {
        return YES;
    }

    NSUInteger stagingBytesPerPixel = mglMetalReadbackBytesPerPixel(mglTextureInfo(sourceTexture).pixel_format);
    NSUInteger stagingBytesPerRow = (NSUInteger)copyW * stagingBytesPerPixel;
    NSUInteger stagingSize = stagingBytesPerRow * (NSUInteger)copyH;
    NSUInteger outputBytesPerRow = (NSUInteger)copyW * 4u;
    if (stagingSize == 0u) {
        return YES;
    }


    NSUInteger dstOffset = ((NSUInteger)dstY * bytesPerRow) + ((NSUInteger)dstX * 4u);
    if (dstOffset >= readSize ||
        outputBytesPerRow > bytesPerRow ||
        ((NSUInteger)copyH - 1u) * bytesPerRow + outputBytesPerRow > readSize - dstOffset) {
        NSLog(@"MGL WARNING: readPixels clipped copy exceeds destination storage for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (![self ensureWritableCommandBuffer:"mglReadColorTextureAsBGRA8"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    BOOL readbackSuccess = YES;
    id readBuffer = [self readbackStageAndWaitTexture:sourceTexture
                                                     sourceLevel:sourceLevel
                                                     sourceSlice:sourceSlice
                                                 sourceDepthPlane:sourceDepthPlane
                                                       copyOrigin:mglTextureOrigin((NSUInteger)metalSrcX,
                                                                                (NSUInteger)metalSrcY,
                                                                                sourceDepthPlane)
                                                         copySize:mglTextureSize((NSUInteger)copyW,
                                                                              (NSUInteger)copyH,
                                                                              1u)
                                             stagingBytesPerRow:stagingBytesPerRow
                                                    stagingSize:stagingSize
                                                         reason:reason
                                                        logKind:"readback"
                                                         success:&readbackSuccess];
    if (!readBuffer) {
        return NO;
    }

    if (readbackSuccess) {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
        mglMetalCopyTextureBytesToBGRA8((const uint8_t *)mglTextureBufferContents(readBuffer),
                                        stagingBytesPerRow,
                                        dst,
                                        bytesPerRow,
                                        (NSUInteger)copyW,
                                        (NSUInteger)copyH,
                                        mglTextureInfo(sourceTexture).pixel_format,
                                        YES);
    }

    return readbackSuccess;
}

- (BOOL)mglReadDepthTextureAsFloat:(id)sourceTexture
                       sourceLevel:(NSUInteger)sourceLevel
                       sourceSlice:(NSUInteger)sourceSlice
                   sourceDepthPlane:(NSUInteger)sourceDepthPlane
                         pixelBytes:(void *)pixelBytes
                        bytesPerRow:(NSUInteger)bytesPerRow
                      bytesPerImage:(NSUInteger)bytesPerImage
                         fromRegion:(MGLRegionValue)region
                             reason:(const char *)reason
{
    NSUInteger readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return NO;
    }

    if (!sourceTexture || region.size.width == 0u || region.size.height == 0u) {
        return sourceTexture != nil;
    }

    BOOL sourceIsDepthStencil =
        mglTextureInfo(sourceTexture).pixel_format == MGLPixelFormatDepth32Float_Stencil8;
    BOOL sourceIsDepth16 =
        mglTextureInfo(sourceTexture).pixel_format == MGLPixelFormatDepth16Unorm;
    if (mglTextureInfo(sourceTexture).pixel_format != MGLPixelFormatDepth32Float &&
        mglTextureInfo(sourceTexture).pixel_format != MGLPixelFormatDepth16Unorm &&
        !sourceIsDepthStencil) {
        static uint64_t s_unsupportedDepthReadFormatCount = 0;
        uint64_t hit = ++s_unsupportedDepthReadFormatCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels unsupported Metal depth readback format=%lu for %s hit=%llu",
                  (unsigned long)mglTextureInfo(sourceTexture).pixel_format,
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (mglTextureInfo(sourceTexture).sample_count > 1u) {
        sourceTexture = [self resolvedReadbackTextureForMultisampleTexture:sourceTexture
                                                               sourceLevel:sourceLevel
                                                               sourceSlice:sourceSlice
                                                           sourceDepthPlane:sourceDepthPlane
                                                                    reason:reason];
        if (!sourceTexture) {
            return NO;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
    }

    if (sourceIsDepthStencil) {
        sourceTexture = [self depthFloatTextureForDepthStencilReadback:sourceTexture
                                                                reason:reason];
        if (!sourceTexture) {
            return NO;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
        sourceIsDepthStencil = NO;
        sourceIsDepth16 = NO;
    }

    if (bytesPerRow < region.size.width * sizeof(float)) {
        NSLog(@"MGL WARNING: readPixels depth destination row too small row=%lu width=%lu for %s",
              (unsigned long)bytesPerRow,
              (unsigned long)region.size.width,
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    NSUInteger levelWidth = mglTextureInfo(sourceTexture).width;
    NSUInteger levelHeight = mglTextureInfo(sourceTexture).height;
    if (sourceLevel > 0u) {
        if (sourceLevel >= mglTextureInfo(sourceTexture).mipmap_level_count) {
            NSLog(@"MGL WARNING: readPixels invalid depth mip level=%lu mipLevels=%lu for %s",
                  (unsigned long)sourceLevel,
                  (unsigned long)mglTextureInfo(sourceTexture).mipmap_level_count,
                  reason ? reason : "unknown");
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NO;
        }
        levelWidth = MAX((NSUInteger)1u, mglTextureInfo(sourceTexture).width >> sourceLevel);
        levelHeight = MAX((NSUInteger)1u, mglTextureInfo(sourceTexture).height >> sourceLevel);
    }


    MGLRenderReadTextureRegionClip clip = {0};
    mglRenderReadTextureRegionClip(
        (int64_t)region.origin.x, (int64_t)region.origin.y,
        (int64_t)region.size.width, (int64_t)region.size.height,
        (int64_t)levelWidth, (int64_t)levelHeight, &clip);
    NSInteger copyW = (NSInteger)clip.copy_w;
    NSInteger copyH = (NSInteger)clip.copy_h;
    NSInteger dstX = (NSInteger)clip.dst_x;
    NSInteger dstY = (NSInteger)clip.dst_y;
    NSInteger metalSrcX = (NSInteger)clip.metal_src_x;
    NSInteger metalSrcY = (NSInteger)clip.metal_src_y;
    if (clip.empty) {
        return YES;
    }

    NSUInteger sourceDepthBytes = sourceIsDepthStencil ? sizeof(float) : (sourceIsDepth16 ? sizeof(uint16_t) : sizeof(float));
    NSUInteger stagingBytesPerRow = (NSUInteger)copyW * sourceDepthBytes;
    /* Metal requires destinationBytesPerRow to be a multiple of 4 bytes
     * on macOS. Depth16Unorm (2 bytes/pixel) can produce a non-aligned
     * row for narrow reads, causing the blit to return zeros/garbage. */
    stagingBytesPerRow = (stagingBytesPerRow + 3u) & ~3u;
    NSUInteger stagingSize = stagingBytesPerRow * (NSUInteger)copyH;
    if (stagingSize == 0u) {
        return YES;
    }

    NSUInteger dstOffset = ((NSUInteger)dstY * bytesPerRow) + ((NSUInteger)dstX * sizeof(float));
    NSUInteger destinationCopyBytesPerRow = (NSUInteger)copyW * sizeof(float);
    if (dstOffset >= readSize ||
        destinationCopyBytesPerRow > bytesPerRow ||
        ((NSUInteger)copyH - 1u) * bytesPerRow + destinationCopyBytesPerRow > readSize - dstOffset) {
        NSLog(@"MGL WARNING: readPixels clipped depth copy exceeds destination storage for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (![self ensureWritableCommandBuffer:"mglReadDepthTextureAsFloat"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    BOOL readbackSuccess = YES;
    id readBuffer = [self readbackStageAndWaitTexture:sourceTexture
                                                     sourceLevel:sourceLevel
                                                     sourceSlice:sourceSlice
                                                 sourceDepthPlane:sourceDepthPlane
                                                       copyOrigin:mglTextureOrigin((NSUInteger)metalSrcX,
                                                                                (NSUInteger)metalSrcY,
                                                                                sourceDepthPlane)
                                                         copySize:mglTextureSize((NSUInteger)copyW,
                                                                              (NSUInteger)copyH,
                                                                              1u)
                                             stagingBytesPerRow:stagingBytesPerRow
                                                    stagingSize:stagingSize
                                                         reason:reason
                                                        logKind:"depth readback"
                                                         success:&readbackSuccess];
    if (!readBuffer) {
        return NO;
    }

    if (readbackSuccess) {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
        if (sourceIsDepthStencil || sourceIsDepth16) {
            /* Depth16 / unpacked depth-float -> GL float. */
            mglRenderCopyDepthTextureBytesToFloat(
                mglTextureBufferContents(readBuffer), (uint64_t)stagingBytesPerRow,
                dst, (uint64_t)bytesPerRow,
                (uint64_t)copyW, (uint64_t)copyH,
                (uint64_t)sourceDepthBytes,
                sourceIsDepth16 ? 1 : 0,
                1);
        } else {
            mglMetalCopyRows((const uint8_t *)mglTextureBufferContents(readBuffer),
                             stagingBytesPerRow,
                             dst,
                             bytesPerRow,
                             stagingBytesPerRow,
                             (NSUInteger)copyH,
                             YES);
        }
    }

    return readbackSuccess;
}

- (BOOL)mglReadIntegerTextureAsRGBA32:(id)sourceTexture
                           pixelBytes:(void *)pixelBytes
                           bytesPerRow:(NSUInteger)bytesPerRow
                        bytesPerImage:(NSUInteger)bytesPerImage
                           fromRegion:(MGLRegionValue)region
                     outputComponents:(NSUInteger)outputComponents
                  outputComponentBytes:(NSUInteger)outputComponentBytes
                         componentMap:(const int[4])componentMap
                          packedType:(GLenum)packedType
                        mipmapLevel:(NSUInteger)mipmapLevel
                              slice:(NSUInteger)mtlSlice
                     isRenderTarget:(BOOL)isRenderTarget
{

    MGLRenderIntegerReadbackSource src = {0};
    mglRenderIntegerReadbackSourceClassify(
        (uint32_t)mglTextureInfo(sourceTexture).pixel_format, &src);
    if (!src.recognized) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }
    NSUInteger componentCount = (NSUInteger)src.component_count;
    NSUInteger sourceComponentBytes = (NSUInteger)src.component_bytes;
    BOOL sourceSigned = src.source_signed != 0;
    BOOL sourceRGB10A2Uint = src.source_rgb10a2_uint != 0;

    if (mglTextureInfo(sourceTexture).sample_count > 1u) {
        sourceTexture = [self resolvedReadbackTextureForMultisampleTexture:sourceTexture
                                                               sourceLevel:mipmapLevel
                                                               sourceSlice:mtlSlice
                                                           sourceDepthPlane:0u
                                                                    reason:"integer FBO readback"];
        if (!sourceTexture) {
            return NO;
        }
        mipmapLevel = 0u;
        mtlSlice = 0u;
    }


    MGLRenderIntegerPackedType packed = {0};
    mglRenderIntegerReadbackPackedTypeClassify((uint32_t)packedType, &packed);
    BOOL isPackedType = packed.is_packed != 0;
    uint32_t packedBitWidths[4] = {
        packed.bit_widths[0], packed.bit_widths[1],
        packed.bit_widths[2], packed.bit_widths[3]};
    uint32_t packedShifts[4] = {
        packed.shifts[0], packed.shifts[1],
        packed.shifts[2], packed.shifts[3]};
    NSUInteger packedOutputBytes = (NSUInteger)packed.output_bytes;
    if (packed.output_components > 0u) {
        outputComponents = (NSUInteger)packed.output_components;
    }

    NSUInteger dstPixelBytes = isPackedType ? packedOutputBytes : (outputComponentBytes * outputComponents);
    NSUInteger readSize = bytesPerImage ? bytesPerImage : bytesPerRow * region.size.height;
    if (!pixelBytes || bytesPerRow < region.size.width * dstPixelBytes) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    (void)readSize;
    if (region.size.width == 0u || region.size.height == 0u) {
        return YES;
    }

    NSInteger minX = MAX((NSInteger)0, (NSInteger)region.origin.x);
    NSInteger minY = MAX((NSInteger)0, (NSInteger)region.origin.y);
    NSInteger maxX = MIN((NSInteger)mglTextureInfo(sourceTexture).width,
                         (NSInteger)region.origin.x + (NSInteger)region.size.width);
    NSInteger maxY = MIN((NSInteger)mglTextureInfo(sourceTexture).height,
                         (NSInteger)region.origin.y + (NSInteger)region.size.height);
    NSInteger copyW = maxX - minX;
    NSInteger copyH = maxY - minY;
    if (copyW <= 0 || copyH <= 0) {
        return YES;
    }

    NSUInteger srcPixelBytes = sourceRGB10A2Uint ? 4u : componentCount * sourceComponentBytes;
    NSUInteger srcBytesPerRow = (NSUInteger)copyW * srcPixelBytes;
    NSUInteger stagingSize = srcBytesPerRow * (NSUInteger)copyH;
    if (![self ensureWritableCommandBuffer:"mglReadIntegerTextureAsRGBA32"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    id readBuffer = mglTextureCreateBuffer(
        _device, stagingSize, MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
    id blit = readBuffer
        ? (__bridge id)mglRenderCreateBlitEncoderBorrowed(
              _commandState.currentCommandBufferOwner)
        : nil;
    if (!readBuffer || !blit) {
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NO;
    }

    /* Calculate the texture height at the specified mipmap level. */
    NSUInteger levelHeight = mglTextureInfo(sourceTexture).height;
    if (mipmapLevel > 0u) {
        levelHeight = MAX((NSUInteger)1u, mglTextureInfo(sourceTexture).height >> mipmapLevel);
    }
    /* Render-target textures are stored top-to-bottom in Metal (Metal y=0 =
     * GL y=levelHeight-1), so the blit source origin must be Y-flipped.
     * Non-render-target textures (e.g. storage images written via imageStore)
     * store data in GL order (Metal y=0 = GL y=0), so no source Y-flip is
     * needed.  Using the flipped origin for storage images would read the
     * wrong rows and corrupt the readback. */
    NSUInteger blitSrcY = isRenderTarget
        ? (levelHeight - (NSUInteger)maxY)
        : (NSUInteger)minY;
    mglTextureCopyTextureToBuffer(
        blit, sourceTexture, mtlSlice, mipmapLevel,
        mglTextureOrigin((NSUInteger)minX, blitSrcY, 0u),
        mglTextureSize((NSUInteger)copyW, (NSUInteger)copyH, 1u), readBuffer,
        0u, srcBytesPerRow, stagingSize);
    mglTextureEndBlitEncoder(blit);
    id integerReadbackCommandBuffer =
        (__bridge id)mglCmdDetachCurrentCommandBufferForSubmission(&_commandState);
    MGLRenderCommandBufferTransaction integerReadbackTransaction = {0};
    int integerReadbackResult = mglCmdCommitCommandBufferTransaction(
        &_commandState, (__bridge void *)integerReadbackCommandBuffer,
        _gpuRecovery.commandRecoveryOwner, YES, &integerReadbackTransaction);
    if (integerReadbackResult != 0 || integerReadbackTransaction.has_error) {
        NSLog(@"MGL ERROR: integer texture readback owner transaction failed");
        mglCmdReleaseDetachedCommandBufferIfOwned(&_commandState, (__bridge void *)integerReadbackCommandBuffer);
        return NO;
    }
    mglCmdReleaseDetachedCommandBufferIfOwned(&_commandState, (__bridge void *)integerReadbackCommandBuffer);

    NSUInteger dstX = (NSUInteger)(minX - (NSInteger)region.origin.x);
    NSUInteger dstY = (NSUInteger)(minY - (NSInteger)region.origin.y);

    MGLRenderIntegerReadbackConvertParams convert = {
        .src = (const uint8_t *)mglTextureBufferContents(readBuffer),
        .src_bytes_per_row = srcBytesPerRow,
        .source_component_count = (uint32_t)componentCount,
        .source_component_bytes = (uint32_t)sourceComponentBytes,
        .source_signed = sourceSigned ? 1 : 0,
        .source_rgb10a2_uint = sourceRGB10A2Uint ? 1 : 0,
        .copy_w = (uint32_t)copyW,
        .copy_h = (uint32_t)copyH,
        .dst = (uint8_t *)pixelBytes,
        .dst_bytes_per_row = bytesPerRow,
        .dst_pixel_bytes = dstPixelBytes,
        .dst_x = (uint64_t)dstX,
        .dst_y = (uint64_t)dstY,
        .output_components = (uint32_t)outputComponents,
        .component_map = componentMap,
        .output_component_bytes = (uint32_t)outputComponentBytes,
        .packed_type = (uint32_t)packedType,
        .is_packed_type = isPackedType ? 1 : 0,
        .packed_bit_widths = packedBitWidths,
        .packed_shifts = packedShifts,
        .packed_output_bytes = (uint32_t)packedOutputBytes,
    };
    if (mglRenderConvertIntegerReadback(&convert) != 0) {
        [self newCommandBuffer];
        return NO;
    }

    [self newCommandBuffer];
    return YES;
}

- (void)mglApplyPendingFBODepthClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                     textureObj:(Texture *)textureObj
                                     mtlTexture:(id)texture
{
    if (!fbo || !attachment || !texture || !(attachment->clear_bitmask & GL_DEPTH_BUFFER_BIT)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    if (mglRenderEncodeDepthClearForCommandBufferOwner(
            _commandState.currentCommandBufferOwner,
            (__bridge void *)texture, subresource.level,
            subresource.slice, subresource.depthPlane,
            attachment->clear_color[0]) == 0) {
        attachment->clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(textureObj, attachment->level);
    } else {
        NSLog(@"MGL WARNING: C++ readPixels depth clear failed fbo=%u",
              (unsigned)fbo->name);
    }
}

- (void)mglApplyPendingDefaultDepthClearToTexture:(id)texture
{
    if (!ctx || !texture || !(ctx->state.default_fbo_clear_bitmask & GL_DEPTH_BUFFER_BIT)) {
        return;
    }

    if (mglRenderEncodeDepthClearForCommandBufferOwner(
            _commandState.currentCommandBufferOwner,
            (__bridge void *)texture, 0, 0, 0,
            ctx->state.var.depth_clear_value) == 0) {
        ctx->state.default_fbo_clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        NSLog(@"MGL WARNING: C++ default depth clear failed");
    }
}

- (void)mtlReadDepthPixels:(GLMContext)glm_ctx
                pixelBytes:(void *)pixelBytes
               bytesPerRow:(NSUInteger)bytesPerRow
             bytesPerImage:(NSUInteger)bytesPerImage
                fromRegion:(MGLRegionValue)region
{
    MGL_ASSERT_GL_THREAD();
    ctx = glm_ctx;

    NSUInteger readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return;
    }

    if (glm_ctx->state.readbuffer) {
        Framebuffer *fbo = glm_ctx->state.readbuffer;
        FBOAttachment *attachment = fbo ? &fbo->depth : NULL;
        Texture *readTextureObject = [self framebufferAttachmentTexture:attachment];
        if (!readTextureObject) {
            NSLog(@"MGL WARNING: readPixels FBO has no depth attachment fbo=%u",
                  fbo ? (unsigned)fbo->name : 0u);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        readTextureObject->is_render_target = true;
        if (![self bindMTLTexture:readTextureObject] || !readTextureObject->mtl_data) {
            NSLog(@"MGL WARNING: readPixels could not bind FBO depth texture fbo=%u tex=%u",
                  fbo ? (unsigned)fbo->name : 0u,
                  (unsigned)readTextureObject->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        id texture = (__bridge id)(readTextureObject->mtl_data);
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);

        [self endRenderEncoding];
        if (![self ensureWritableCommandBuffer:"mtlReadDepthPixels.fbo"]) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        [self mglApplyPendingFBODepthClearForReadback:fbo
                                           attachment:attachment
                                           textureObj:readTextureObject
                                           mtlTexture:texture];
        [self mglReadDepthTextureAsFloat:texture
                             sourceLevel:subresource.level
                             sourceSlice:subresource.slice
                         sourceDepthPlane:subresource.depthPlane
                               pixelBytes:pixelBytes
                              bytesPerRow:bytesPerRow
                            bytesPerImage:bytesPerImage
                               fromRegion:region
                                   reason:"FBO depth readback"];
        return;
    }

    GLuint drawBufferIndex = mglDefaultDrawBufferIndexForGL(glm_ctx->state.read_buffer);
    id texture = nil;
    if (drawBufferIndex < _MAX_DRAW_BUFFERS) {
        texture = (__bridge id)
            mglRendererBackendGetDefaultDrawBufferAttachment(
                _backend, drawBufferIndex,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH);
    }

    if (!texture) {
        NSLog(@"MGL WARNING: readPixels default framebuffer has no depth texture slot=%u",
              (unsigned)drawBufferIndex);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlReadDepthPixels.default"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    [self mglApplyPendingDefaultDepthClearToTexture:texture];
    [self mglReadDepthTextureAsFloat:texture
                         sourceLevel:0u
                         sourceSlice:0u
                     sourceDepthPlane:0u
                           pixelBytes:pixelBytes
                          bytesPerRow:bytesPerRow
                        bytesPerImage:bytesPerImage
                           fromRegion:region
                               reason:"default framebuffer depth readback"];
}

-(void)mtlReadIntegerPixels:(GLMContext)glm_ctx
                 pixelBytes:(void *)pixelBytes
                bytesPerRow:(NSUInteger)bytesPerRow
              bytesPerImage:(NSUInteger)bytesPerImage
                 fromRegion:(MGLRegionValue)region
                     format:(GLenum)format
                       type:(GLenum)type
{
    ctx = glm_ctx;
    Framebuffer *fbo = glm_ctx ? glm_ctx->state.readbuffer : NULL;
    GLenum readBuffer = glm_ctx ? glm_ctx->state.read_buffer : GL_NONE;
    if (!fbo || readBuffer < GL_COLOR_ATTACHMENT0 ||
        readBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    FBOAttachment *attachment = &fbo->color_attachments[readBuffer - GL_COLOR_ATTACHMENT0];
    Texture *textureObj = [self framebufferAttachmentTexture:attachment];
    if (!textureObj || ![self bindMTLTexture:textureObj] || !textureObj->mtl_data) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    id texture = (__bridge id)(textureObj->mtl_data);
    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlReadIntegerPixels.fbo"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    [self mglApplyPendingFBOColorClearForReadback:fbo
                                       attachment:attachment
                                       textureObj:textureObj
                                       mtlTexture:texture
                                   attachmentEnum:readBuffer];

    /* Determine output component count and component mapping.
     * componentMap[c] = source component index for output component c, or -1. */
    NSUInteger outputComponents = 4u;
    int componentMap[4] = {0, 1, 2, 3};
    switch (format) {
        case GL_RED_INTEGER:    outputComponents = 1u; componentMap[0]=0; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        case GL_RG_INTEGER:     outputComponents = 2u; componentMap[0]=0; componentMap[1]=1; componentMap[2]=-1; componentMap[3]=-1; break;
        case GL_RGB_INTEGER:    outputComponents = 3u; componentMap[0]=0; componentMap[1]=1; componentMap[2]=2;  componentMap[3]=-1; break;
        case GL_BGR_INTEGER:    outputComponents = 3u; componentMap[0]=2; componentMap[1]=1; componentMap[2]=0;  componentMap[3]=-1; break;
        case GL_RGBA_INTEGER:   outputComponents = 4u; componentMap[0]=0; componentMap[1]=1; componentMap[2]=2;  componentMap[3]=3;  break;
        case GL_BGRA_INTEGER:   outputComponents = 4u; componentMap[0]=2; componentMap[1]=1; componentMap[2]=0;  componentMap[3]=3;  break;
        case 0x8d95: /*GL_GREEN_INTEGER*/ outputComponents = 1u; componentMap[0]=1; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        case 0x8d96: /*GL_BLUE_INTEGER*/  outputComponents = 1u; componentMap[0]=2; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        case 0x8d97: /*GL_ALPHA_INTEGER*/ outputComponents = 1u; componentMap[0]=3; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        default: outputComponents = 4u; break;
    }

    NSUInteger outputComponentBytes = (type == GL_BYTE || type == GL_UNSIGNED_BYTE) ? 1u :
                                      (type == GL_SHORT || type == GL_UNSIGNED_SHORT) ? 2u : 4u;

    [self mglReadIntegerTextureAsRGBA32:texture
                            pixelBytes:pixelBytes
                           bytesPerRow:bytesPerRow
                         bytesPerImage:bytesPerImage
                            fromRegion:region
                      outputComponents:outputComponents
                   outputComponentBytes:outputComponentBytes
                          componentMap:componentMap
                           packedType:type
                          mipmapLevel:subresource.level
                                slice:subresource.slice
                      isRenderTarget:YES];
}

-(void) mtlReadDrawable:(GLMContext) glm_ctx pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MGLRegionValue)region
{
    ctx = glm_ctx;

    NSUInteger readSize = bytesPerImage;
    if (readSize == 0 && bytesPerRow > 0) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0) {
        return;
    }

    if (glm_ctx->state.readbuffer)
    {
        Framebuffer *fbo = glm_ctx->state.readbuffer;
        GLenum readBuffer = glm_ctx->state.read_buffer;
        if (!fbo ||
            readBuffer == GL_NONE ||
            readBuffer < GL_COLOR_ATTACHMENT0 ||
            readBuffer >= GL_COLOR_ATTACHMENT0 + glm_ctx->state.max_color_attachments ||
            readBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
            static uint64_t s_invalidReadFBOCount = 0;
            uint64_t hit = ++s_invalidReadFBOCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: readPixels invalid FBO read buffer=0x%x maxColor=%u hit=%llu; returning zero data",
                      (unsigned)readBuffer,
                      (unsigned)glm_ctx->state.max_color_attachments,
                      (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        GLuint attachmentIndex = (GLuint)(readBuffer - GL_COLOR_ATTACHMENT0);
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            static uint64_t s_missingReadAttachmentCount = 0;
            uint64_t hit = ++s_missingReadAttachmentCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: readPixels FBO read attachment 0x%x is not attached fbo=%u hit=%llu; returning zero data",
                      (unsigned)readBuffer,
                      (unsigned)fbo->name,
                      (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *readTextureObject = [self framebufferAttachmentTexture:attachment];
        if (!readTextureObject) {
            NSLog(@"MGL WARNING: readPixels FBO attachment has no texture fbo=%u attachment=0x%x",
                  (unsigned)fbo->name,
                  (unsigned)readBuffer);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        readTextureObject->is_render_target = true;
        if (![self bindMTLTexture:readTextureObject] || !readTextureObject->mtl_data) {
            NSLog(@"MGL WARNING: readPixels could not bind FBO read texture fbo=%u attachment=0x%x tex=%u",
                  (unsigned)fbo->name,
                  (unsigned)readBuffer,
                  (unsigned)readTextureObject->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        id texture = (__bridge id)(readTextureObject->mtl_data);
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);
        [self endRenderEncoding];
        if (![self ensureWritableCommandBuffer:"mtlReadDrawable.fbo"]) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        [self mglApplyPendingFBOColorClearForReadback:fbo
                                           attachment:attachment
                                           textureObj:readTextureObject
                                           mtlTexture:texture
                                       attachmentEnum:readBuffer];
        [self mglReadColorTextureAsBGRA8:texture
                              sourceLevel:subresource.level
                              sourceSlice:subresource.slice
                          sourceDepthPlane:subresource.depthPlane
                                pixelBytes:pixelBytes
                               bytesPerRow:bytesPerRow
                             bytesPerImage:bytesPerImage
                                fromRegion:region
                                    reason:"FBO color readback"];
        return;
    }

    GLuint mgl_drawbuffer;
    id texture = nil;

    switch(glm_ctx->state.read_buffer)
    {
        case GL_FRONT: mgl_drawbuffer = _FRONT; break;
        case GL_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_FRONT_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_FRONT_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_BACK_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_BACK_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        default:
            NSLog(@"MGL WARNING: readPixels unsupported default read buffer=0x%x; returning zero data",
                  (unsigned)glm_ctx->state.read_buffer);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
    }

    if (mgl_drawbuffer == _FRONT)
    {
        if (!_drawable) {
            (void)[self mglApplyPendingDrawableSize];
            _drawable = [self mglNextDrawable];
        }
        texture = _drawable ? [self mglDrawableTexture] : nil;
    }
    else if (mgl_drawbuffer < _MAX_DRAW_BUFFERS)
    {
        texture = (__bridge id)
            mglRendererBackendGetDefaultDrawBufferAttachment(
                _backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR);
    }

    if (!texture)
    {
        NSLog(@"MGL WARNING: readPixels default drawbuffer slot=%u has no texture; returning zero data",
              (unsigned)mgl_drawbuffer);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlReadDrawable.default"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    if (mgl_drawbuffer == _FRONT) {
        [self mglApplyPendingDefaultColorClearToTexture:texture];
    }
    [self mglReadColorTextureAsBGRA8:texture
                          sourceLevel:0u
                          sourceSlice:0u
                      sourceDepthPlane:0u
                            pixelBytes:pixelBytes
                           bytesPerRow:bytesPerRow
                            bytesPerImage:bytesPerImage
                               fromRegion:region
                                   reason:"default framebuffer readback"];
    return;
}

-(void) mtlGetTexImage:(GLMContext) glm_ctx tex: (Texture *)tex pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MGLRegionValue)region format:(GLenum)format type:(GLenum)type mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice
{
    id texture = nil;

    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGetTexImage called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (!pixelBytes) {
        NSLog(@"MGL WARNING: mtlGetTexImage called with NULL destination for texture %u", tex->name);
        return;
    }

    if (!tex->mtl_data && ![self bindMTLTexture:tex]) {
        NSLog(@"MGL ERROR: mtlGetTexImage failed to bind texture %u", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    texture = (__bridge id)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGetTexImage texture %u has no Metal texture", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (mglRenderTextureIsFramebufferOnly((__bridge void *)texture)) {
        NSLog(@"MGL ERROR: Cannot read from framebuffer only texture %u\n", tex->name);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (![self synchronizeRenderPassForTextureReadback:texture reason:"mtlGetTexImage"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    /* Ensure any pending texture upload blit commands are committed before
     * reading back. Without this, getBytes may return stale/zero data because
     * the blit encoding the upload is still in the uncommitted command buffer. */
    [self endRenderEncoding];
    if (mglRenderCommandBufferOwnerHasCurrent(
            _commandState.currentCommandBufferOwner) == 1) {
        id pendingCB =
            (__bridge id)mglCmdDetachCurrentCommandBufferForSubmission(&_commandState);
        @try {
            [self commitCommandBufferWithAGXRecovery:pendingCB];
            mglTextureWaitCommandBuffer(pendingCB);
        } @catch (NSException *e) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback flush failed: %@", e.reason);
        }
        MGLRenderCommandBufferState pendingState = {0};
        (void)mglRenderGetCommandBufferState(
            (__bridge void *)pendingCB, &pendingState);
        if (pendingState.has_error) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback command buffer error: %s",
                  mglRenderCommandBufferErrorDescription(&pendingState));
        }
        [self newCommandBuffer];
    }

    MGLRegionValue readRegion = region;
    /* Render target textures are stored top-to-bottom in Metal, but OpenGL
     * readPixels expects bottom-to-top order. Flip Y for render targets to
     * match OpenGL semantics. This mirrors the Y flip already done in
     * mglReadColorTextureAsBGRA8 (metalSrcY = levelHeight - glMaxY). */
    BOOL flipRenderTargetRows = tex->is_render_target;
    if (flipRenderTargetRows && region.size.height > 0u) {
        NSUInteger levelHeight = MAX((NSUInteger)1u, mglTextureInfo(texture).height >> level);
        if (region.origin.y > levelHeight ||
            region.size.height > levelHeight - region.origin.y) {
            NSLog(@"MGL ERROR: mtlGetTexImage invalid render-target read region tex=%u y=%lu h=%lu levelHeight=%lu",
                  tex->name,
                  (unsigned long)region.origin.y,
                  (unsigned long)region.size.height,
                  (unsigned long)levelHeight);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
            return;
        }
        readRegion.origin.y = levelHeight - (region.origin.y + region.size.height);
    }

    /* Integer texture readback path: when the source texture is an integer
     * format and the output format is GL_*_INTEGER, use the dedicated integer
     * readback function that handles packed types and component mapping. */

    MGLRenderIntegerReadbackClassify classify = {0};
    mglRenderIntegerReadbackClassify(
        (uint32_t)mglTextureInfo(texture).pixel_format, (uint32_t)format, (uint32_t)type,
        &classify);

    if (classify.source_is_integer_texture &&
        classify.output_is_integer_format) {
        /* Pass the original (non-Y-flipped) region. mglReadIntegerTextureAsRGBA32
         * does its own Y-flip on the blit source origin AND Y-flips the output
         * rows, so passing a pre-Y-flipped readRegion here would double-flip. */
        [self mglReadIntegerTextureAsRGBA32:texture
                                pixelBytes:pixelBytes
                               bytesPerRow:bytesPerRow
                             bytesPerImage:bytesPerImage
                                fromRegion:region
                          outputComponents:(NSUInteger)classify.output_components
                       outputComponentBytes:(NSUInteger)classify.output_component_bytes
                              componentMap:classify.component_map
                               packedType:type
                              mipmapLevel:level
                                    slice:slice
                          isRenderTarget:(BOOL)tex->is_render_target];
        return;
    }

    NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
    BOOL directR32FloatRead =
        (mglTextureInfo(texture).pixel_format == MGLPixelFormatR32Float &&
         format == GL_RED &&
         type == GL_FLOAT);
    BOOL useBGRA8Conversion =
        (dstPixelBytes > 0u &&
         readRegion.size.depth == 1u &&
         !directR32FloatRead &&
         mglMetalReadbackFormatIsBGRA8Compatible(mglTextureInfo(texture).pixel_format));

    // MGL_TEXTURE_STORAGE_PRIVATE textures cannot be read directly with getBytes:.
    // Use a blit-to-buffer path to convert GPU-private tiled memory to linear CPU memory.
    if (mglTextureInfo(texture).storage_mode == MGL_TEXTURE_STORAGE_PRIVATE) {

        MGLRenderGetTexImagePlan plan = {0};
        mglRenderGetTexImagePlan(
            (uint32_t)mglTextureInfo(texture).pixel_format,
            (uint32_t)format,
            (uint32_t)type,
            (uint32_t)readRegion.size.width,
            (uint32_t)readRegion.size.height,
            (uint32_t)readRegion.size.depth,
            (uint32_t)dstPixelBytes,
            (uint32_t)mglMetalReadbackBytesPerPixel(mglTextureInfo(texture).pixel_format),
            mglMetalReadbackFormatIsBGRA8Compatible(mglTextureInfo(texture).pixel_format) ? 1 : 0,
            (uint32_t)bytesPerRow,
            (uint32_t)bytesPerImage,
            1,
            &plan);
        useBGRA8Conversion = plan.use_bgra8_conversion;
        NSUInteger rowBytes = (NSUInteger)plan.row_bytes;
        NSUInteger imageBytes = (NSUInteger)plan.image_bytes;
        NSUInteger totalBytes = (NSUInteger)plan.total_bytes;

        id stagingBuffer = mglTextureCreateBuffer(
            _device, totalBytes, MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
        if (!stagingBuffer) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to allocate staging buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }

        id blitCB = mglTextureCreateCommandBuffer(_commandQueue);
        if (!blitCB) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit command buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        id blitEncoder = mglTextureCreateBlitEncoder(blitCB);
        if (!blitEncoder) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit encoder for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        mglTextureCopyTextureToBuffer(
            blitEncoder, texture, slice, level, readRegion.origin,
            readRegion.size, stagingBuffer, 0, rowBytes, imageBytes);

        mglTextureEndBlitEncoder(blitEncoder);
        mglTextureCommitCommandBuffer(blitCB);
        mglTextureWaitCommandBuffer(blitCB);

        MGLRenderCommandBufferState blitState = {0};
        (void)mglRenderGetCommandBufferState(
            (__bridge void *)blitCB, &blitState);
        if (blitState.has_error) {
            NSLog(@"MGL ERROR: mtlGetTexImage blit failed for texture %u: %s",
                  tex->name, mglRenderCommandBufferErrorDescription(&blitState));
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        if (useBGRA8Conversion) {
            if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL((const uint8_t *)mglTextureBufferContents(stagingBuffer),
                                                             rowBytes,
                                                             (uint8_t *)pixelBytes,
                                                             bytesPerRow,
                                                             readRegion.size.width,
                                                             readRegion.size.height,
	                                                             mglTextureInfo(texture).pixel_format,
	                                                             format,
	                                                             type,
	                                                             flipRenderTargetRows)) {
	                NSLog(@"MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x",
	                      tex->name,
	                      (unsigned)format,
	                      (unsigned)type);
	                mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
	            }
        } else if (flipRenderTargetRows && readRegion.size.depth == 1u) {
            mglMetalCopyRows((const uint8_t *)mglTextureBufferContents(stagingBuffer),
                             rowBytes,
                             (uint8_t *)pixelBytes,
	                             bytesPerRow,
	                             rowBytes,
                             readRegion.size.height,
                             YES);
        } else {
            memcpy(pixelBytes, mglTextureBufferContents(stagingBuffer), totalBytes);
        }
        if (mglTraceLogIsEnabled() &&
            tex->internalformat == GL_R8 &&
            format == GL_RED &&
            type == GL_UNSIGNED_BYTE &&
            readRegion.size.width > 0 &&
            readRegion.size.height > 0) {
            const uint8_t *rb = (const uint8_t *)pixelBytes;
            mglTraceLog("GET_TEX_IMAGE_R8 tex=%u target=0x%x isRT=%d fmt=%lu rowBytes=%lu dstBPR=%lu size=%lux%lu first=%u,%u,%u,%u,%u,%u,%u,%u",
                        (unsigned)tex->name,
                        (unsigned)tex->target,
                        tex->is_render_target ? 1 : 0,
                        (unsigned long)mglTextureInfo(texture).pixel_format,
                        (unsigned long)rowBytes,
                        (unsigned long)bytesPerRow,
                        (unsigned long)readRegion.size.width,
                        (unsigned long)readRegion.size.height,
                        rb[0],
                        rb[MIN((NSUInteger)1, totalBytes - 1)],
                        rb[MIN((NSUInteger)2, totalBytes - 1)],
                        rb[MIN((NSUInteger)3, totalBytes - 1)],
                        rb[MIN((NSUInteger)4, totalBytes - 1)],
                        rb[MIN((NSUInteger)5, totalBytes - 1)],
                        rb[MIN((NSUInteger)6, totalBytes - 1)],
                        rb[MIN((NSUInteger)7, totalBytes - 1)]);
        }
        return;
	    }

	    @try {
	        if (useBGRA8Conversion || (flipRenderTargetRows && readRegion.size.depth == 1u)) {

	            MGLRenderGetTexImagePlan plan = {0};
	            mglRenderGetTexImagePlan(
	                (uint32_t)mglTextureInfo(texture).pixel_format,
	                (uint32_t)format,
	                (uint32_t)type,
	                (uint32_t)readRegion.size.width,
	                (uint32_t)readRegion.size.height,
	                (uint32_t)readRegion.size.depth,
	                (uint32_t)dstPixelBytes,
	                (uint32_t)mglMetalReadbackBytesPerPixel(mglTextureInfo(texture).pixel_format),
	                mglMetalReadbackFormatIsBGRA8Compatible(mglTextureInfo(texture).pixel_format) ? 1 : 0,
	                (uint32_t)bytesPerRow,
	                (uint32_t)bytesPerImage,
	                0,
	                &plan);
	            NSUInteger rowBytes = (NSUInteger)plan.row_bytes;
	            NSUInteger totalBytes = (NSUInteger)plan.image_bytes;
            NSMutableData *readback = [NSMutableData dataWithLength:totalBytes];
            if (!readback) {
                mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                return;
            }
            mglTextureGetBytes(
                texture, readback.mutableBytes, rowBytes, bytesPerImage,
                readRegion, level, slice, YES);
            if (useBGRA8Conversion) {
                if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL((const uint8_t *)readback.bytes,
                                                                 rowBytes,
                                                                 (uint8_t *)pixelBytes,
                                                                 bytesPerRow,
                                                                 readRegion.size.width,
                                                                 readRegion.size.height,
	                                                                 mglTextureInfo(texture).pixel_format,
	                                                                 format,
	                                                                 type,
	                                                                 flipRenderTargetRows)) {
	                    NSLog(@"MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x",
	                          tex->name,
	                          (unsigned)format,
	                          (unsigned)type);
                    mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
                }
            } else {
                mglMetalCopyRows((const uint8_t *)readback.bytes,
                                 rowBytes,
                                 (uint8_t *)pixelBytes,
                                 bytesPerRow,
                                 rowBytes,
                                 readRegion.size.height,
                                 YES);
            }
        } else {
            mglTextureGetBytes(
                texture, pixelBytes, bytesPerRow, bytesPerImage,
                readRegion, level, slice, YES);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: mtlGetTexImage texture read failed for texture %u: %@",
              tex->name,
              exception);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
    }
}

-(void)mtlGenerateMipmaps:(GLMContext)glm_ctx forTexture:(Texture *) tex
{
    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    RETURN_ON_FAILURE([self processGLState: false]);

    // end encoding on current render encoder
    [self endRenderEncoding];

    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlGenerateMipmaps"]);

    // no failure path..?
    RETURN_ON_FAILURE([self bindMTLTexture:tex]);

    id texture;

    texture = (__bridge id)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps texture %u has no Metal texture after bind", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (mglTextureInfo(texture).mipmap_level_count <= 1u) {
        return;
    }

    // start blit encoder
    id blitCommandEncoder;
    blitCommandEncoder = mglTextureCreateCurrentBlitEncoder(
        _commandState.currentCommandBufferOwner);
    if (!blitCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create blit encoder for mipmap generation");
        return;
    }

    @try {
        if (mglRenderBlitGenerateMipmaps(
                (__bridge void *)blitCommandEncoder,
                (__bridge void *)texture) != 0) {
            [NSException raise:@"MGLGenerateMipmapsError"
                        format:@"C++ mipmap generation failed for texture %u",
                               tex->name];
        }
        mglTextureEndBlitEncoder(blitCommandEncoder);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: generateMipmapsForTexture failed for texture %u: %@",
              tex->name,
              exception);
        @try {
            mglTextureEndBlitEncoder(blitCommandEncoder);
        } @catch (NSException *endException) {
            NSLog(@"MGL WARNING: failed to end mipmap blit encoder after exception: %@", endException);
        }
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
    }
}

- (bool)encodeTextureBytesUpload:(Texture *)tex
                          source:(id)buffer
                    sourceOffset:(NSUInteger)sourceOffset
                sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
              sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                            depth:(NSUInteger)depth
                            slice:(NSUInteger)slice
                            level:(NSUInteger)level
                          xoffset:(NSUInteger)xoffset
                          yoffset:(NSUInteger)yoffset
                          zoffset:(NSUInteger)zoffset
                           reason:(const char *)reason
{
    MGL_ASSERT_GL_THREAD();
    if (!tex || !buffer || sourceBytesPerRow == 0 || width == 0 || height == 0) {
        return false;
    }

    if (tex->mtl_data == NULL) {
        [self bindMTLTexture:tex];
        if (tex->mtl_data == NULL) {
            return false;
        }
    }

    id texture = (__bridge id)(tex->mtl_data);
    if (!texture) {
        return false;
    }

    uint32_t textureType = mglTextureInfo(texture).texture_type;
    MGLRenderTextureSubUploadPlan uploadPlan = {0};
    if (mglRenderTextureSubUploadPlan(
            (uint32_t)tex->target, (uint32_t)textureType, (uint64_t)slice,
            (uint64_t)xoffset, (uint64_t)yoffset, (uint64_t)zoffset,
            (uint64_t)width, (uint64_t)height, (uint64_t)depth,
            (uint64_t)sourceBytesPerRow, (uint64_t)sourceBytesPerImage,
            &uploadPlan) != 0) {
        return false;
    }
    NSUInteger destinationSlice = (NSUInteger)uploadPlan.destination_base_slice;
    MGLOriginValue destinationOrigin = mglTextureOrigin(
        (NSUInteger)uploadPlan.destination_x,
        (NSUInteger)uploadPlan.destination_y,
        (NSUInteger)uploadPlan.destination_z);
    NSUInteger copyHeight = (NSUInteger)uploadPlan.copy_height;
    NSUInteger copyDepth = (NSUInteger)uploadPlan.copy_depth;
    NSUInteger layerCount = (NSUInteger)uploadPlan.layer_count;
    NSUInteger sourceLayerStride =
        (NSUInteger)uploadPlan.source_layer_stride;
    if (copyHeight > NSUIntegerMax / sourceBytesPerRow) {
        return false;
    }
    NSUInteger expectedBytesPerImage = sourceBytesPerRow * copyHeight;
    NSUInteger copyBytesPerImage = sourceBytesPerImage;
    if (textureType == MGLTextureTypeCube ||
        textureType == MGLTextureTypeCubeArray ||
        textureType == MGLTextureType2DArray ||
        textureType == MGLTextureType1DArray ||
        textureType == MGLTextureType2DMultisampleArray) {
        copyBytesPerImage = expectedBytesPerImage;
    } else if (textureType == MGLTextureType3D) {
        if (copyBytesPerImage < expectedBytesPerImage) {
            copyBytesPerImage = expectedBytesPerImage;
        }
    } else {
        copyBytesPerImage = expectedBytesPerImage;
    }

    NSUInteger maxDestinationSlices = mglTextureInfo(texture).array_length;
    if (textureType == MGLTextureTypeCube) {
        maxDestinationSlices = 6UL;
    } else if (textureType == MGLTextureTypeCubeArray) {
        maxDestinationSlices = mglTextureInfo(texture).array_length * 6UL;
    }

    if (level >= mglTextureInfo(texture).mipmap_level_count ||
        destinationSlice >= maxDestinationSlices ||
        layerCount > maxDestinationSlices - destinationSlice ||
        destinationOrigin.x > mglTextureInfo(texture).width ||
        destinationOrigin.y > mglTextureInfo(texture).height ||
        destinationOrigin.z > mglTextureInfo(texture).depth ||
        width > mglTextureInfo(texture).width - destinationOrigin.x ||
        copyHeight > mglTextureInfo(texture).height - destinationOrigin.y ||
        copyDepth > mglTextureInfo(texture).depth - destinationOrigin.z) {
        NSLog(@"MGL ERROR: texture sub upload out of bounds tex=%u level=%lu slice=%lu origin=(%lu,%lu,%lu) size=%lux%lux%lu texture=%lux%lux%lu",
              tex->name,
              (unsigned long)level,
              (unsigned long)destinationSlice,
              (unsigned long)destinationOrigin.x,
              (unsigned long)destinationOrigin.y,
              (unsigned long)destinationOrigin.z,
              (unsigned long)width,
              (unsigned long)copyHeight,
              (unsigned long)copyDepth,
              (unsigned long)mglTextureInfo(texture).width,
              (unsigned long)mglTextureInfo(texture).height,
              (unsigned long)mglTextureInfo(texture).depth);
        return false;
    }

    return [self copyTextureUploadWithDedicatedCommandBuffer:buffer
                                                sourceOffset:sourceOffset
                                           sourceBytesPerRow:sourceBytesPerRow
                                         sourceBytesPerImage:copyBytesPerImage
                                          sourceLayerStride:sourceLayerStride
                                                  layerCount:layerCount
                                                   sourceSize:mglTextureSize(
                                                       (NSUInteger)uploadPlan.copy_width,
                                                       copyHeight, copyDepth)
                                                    toTexture:texture
                                             destinationSlice:destinationSlice
                                             destinationLevel:level
                                            destinationOrigin:destinationOrigin
                                                       reason:reason ? reason : "texture_sub_upload"];
}

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
        [self bindMTLBufferLocked: buf];
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
            NSUInteger dstBytesPerPixel = needsChannelExpand
                ? ((dstPixelFormat == MGLPixelFormatRGBA16Unorm ||
                    dstPixelFormat == MGLPixelFormatRGBA16Snorm ||
                    dstPixelFormat == MGLPixelFormatRGBA16Float ||
                    dstPixelFormat == MGLPixelFormatRGBA16Sint ||
                    dstPixelFormat == MGLPixelFormatRGBA16Uint) ? 8 : 16)
                : 4;
            NSUInteger cpuBytesPerPixel = (tex->faces[0].levels && level < tex->num_levels &&
                                           tex->faces[0].levels[level].width > 0u &&
                                           tex->faces[0].levels[level].pitch > 0u)
                ? (NSUInteger)(tex->faces[0].levels[level].pitch / tex->faces[0].levels[level].width)
                : [self bytesPerPixelForFormat:tex->internalformat];
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
                                bool uploaded = [self encodeTextureBytesUpload:tex
                                                                        source:uploadBuffer
                                                                  sourceOffset:0
                                                              sourceBytesPerRow:dstRowBytes
                                                            sourceBytesPerImage:dstImageBytes
                                                                       width:width
                                                                      height:height
                                                                       depth:depth
                                                                       slice:slice
                                                                       level:level
                                                                     xoffset:xoffset
                                                                     yoffset:yoffset
                                                                     zoffset:zoffset
                                                                      reason:"mtlTexSubImage"];
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

    bool uploaded = [self encodeTextureBytesUpload:tex
                                            source:buffer
                                      sourceOffset:src_offset
                                  sourceBytesPerRow:src_pitch
                                sourceBytesPerImage:src_image_size
                                             width:width
                                            height:height
                                             depth:depth
                                             slice:slice
                                             level:level
                                           xoffset:xoffset
                                           yoffset:yoffset
                                           zoffset:zoffset
                                            reason:"mtlTexSubImage"];
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

    NSUInteger bytesPerPixel = [self bytesPerPixelForFormat:tex->internalformat];
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
        switch (dstPixelFormat) {
            case MGLPixelFormatRGBA16Unorm:
            case MGLPixelFormatRGBA16Snorm:
            case MGLPixelFormatRGBA16Float:
            case MGLPixelFormatRGBA16Sint:
            case MGLPixelFormatRGBA16Uint:
                dstBytesPerPixel = 8;
                break;
            case MGLPixelFormatRGBA32Float:
            case MGLPixelFormatRGBA32Sint:
            case MGLPixelFormatRGBA32Uint:
                dstBytesPerPixel = 16;
                break;
            default:
                needsChannelExpand = NO;
                break;
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
        /* Determine component bytes and alpha default for the destination format */
        NSUInteger srcCompBytes = 0;
        NSUInteger dstCompBytes = 0;
        uint64_t alphaDefault = 0;
        switch (dstPixelFormat) {
            case MGLPixelFormatRGBA16Unorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 65535; break;
            case MGLPixelFormatRGBA16Snorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 32767; break;
            case MGLPixelFormatRGBA16Float:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 0x3C00; break;
            case MGLPixelFormatRGBA16Sint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MGLPixelFormatRGBA16Uint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MGLPixelFormatRGBA32Float:
                srcCompBytes = 4; dstCompBytes = 4;
                { float f = 1.0f; memcpy(&alphaDefault, &f, sizeof(f)); }
                break;
            case MGLPixelFormatRGBA32Sint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            case MGLPixelFormatRGBA32Uint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            default:
                return false;
        }
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
    if (tex->internalformat == GL_DEPTH32F_STENCIL8 &&
        dstPixelFormat == MGLPixelFormatDepth32Float_Stencil8 &&
        dstRowBytes >= width * 5u && dstRowBytes < width * 8u) {
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
    if (tex->target == GL_TEXTURE_2D_ARRAY ||
        tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        metalSlice = zoffset;
    }

    if ((dstPixelFormat == MGLPixelFormatDepth32Float_Stencil8 ||
         dstPixelFormat == MGLPixelFormatDepth24Unorm_Stencil8) &&
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
            uploaded = [self uploadPackedDepthStencilStencilPlane:dstTexture
                                                            texName:tex->name
                                                              bytes:uploadBytesPtr
                                                              width:width
                                                             height:copyHeight
                                                        bytesPerRow:uploadRowBytes
                                                              level:level
                                                              slice:metalSlice
                                                             xorigin:xoffset
                                                             yorigin:yoffset];
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

    bool uploaded = [self encodeTextureBytesUpload:tex
                                            source:uploadBuffer
                                      sourceOffset:0
                                  sourceBytesPerRow:uploadRowBytes
                                sourceBytesPerImage:uploadImageBytes
                                             width:width
                                            height:height
                                             depth:depth
                                             slice:slice
                                             level:level
                                           xoffset:xoffset
                                           yoffset:yoffset
                                           zoffset:zoffset
                                            reason:"mtlTexSubImageBytes"];
    if (uploaded &&
        (dstPixelFormat == MGLPixelFormatDepth32Float_Stencil8 ||
         dstPixelFormat == MGLPixelFormatDepth24Unorm_Stencil8) &&
        uploadRowBytes >= width * 5u) {
        (void)[self uploadPackedDepthStencilStencilPlane:dstTexture
                                                 texName:tex->name
                                                   bytes:uploadBytesPtr
                                                   width:width
                                                  height:copyHeight
                                             bytesPerRow:uploadRowBytes
                                                   level:level
                                                   slice:metalSlice
                                                  xorigin:xoffset
                                                  yorigin:yoffset];
    }
    free(dsMetalUpload);
    return uploaded;
}


#pragma mark - Extracted from createMTLTextureFromGLTexture:
- (void)reUploadExistingCPUTextureData:(Texture *)tex
                                metal:(id)texture
                          pixelFormat:(uint32_t)pixelFormat
                            numFaces:(uint)num_faces
                    uploadLevelCount:(GLuint)upload_level_count
                              isArray:(BOOL)is_array
                   texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
             texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                             texType:(uint32_t)tex_type
{
    NSLog(@"MGL INFO: Re-uploading existing CPU texture data (tex=%d, dims=%lux%lu)",

          tex->name, (unsigned long)mglTextureInfo(texture).width, (unsigned long)mglTextureInfo(texture).height);


    for (int face = 0; face < num_faces; face++) {

        for (int level = 0; level < (int)upload_level_count; level++) {

            TextureLevel *uploadLevel = &tex->faces[face].levels[level];

            if (!mglTextureLevelHasUploadableCPUData(uploadLevel)) {

                continue;

            }


            NSUInteger lvlWidth  = tex->faces[face].levels[level].width;

            NSUInteger lvlHeight = tex->faces[face].levels[level].height;

            NSUInteger lvlDepth  = tex->faces[face].levels[level].depth;

            NSUInteger lvlPitch  = tex->faces[face].levels[level].pitch;

            if (lvlPitch == 0 || lvlWidth == 0) continue;


            if (is_array)

            {
                [self reUploadExistingCPUTextureDataArrayLevel:tex
                                                         metal:texture
                                                   pixelFormat:pixelFormat
                                                         face:face
                                                        level:level
                                  texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                                       texType:tex_type];
            }

            else

            {

            /* Non-array re-upload (2D, 3D, 1D, cube).

             * For 3D textures, bytesPerImage must be a single 2D slice

             * (bytesPerRow * height), NOT the full volume data_size.

             * uploadTextureSliceViaBlit computes bufferSize =

             * safeBytesPerImage * copyDepth, so passing the full volume

             * as bytesPerImage AND depth would double-count and cause

             * newBufferWithBytes to read past the source buffer. */

            NSUInteger bytesPerRow = lvlPitch;

            NSUInteger fullDataSize = tex->faces[face].levels[level].data_size;

            if (fullDataSize == 0) fullDataSize = bytesPerRow * MAX((NSUInteger)lvlHeight, 1UL);


            BOOL is3DReupload = (tex->target == GL_TEXTURE_3D && lvlDepth > 1);

            NSUInteger singleSliceBPI = bytesPerRow * MAX((NSUInteger)lvlHeight, 1UL);

            NSUInteger bytesPerImage = is3DReupload ? singleSliceBPI : fullDataSize;

            NSUInteger uploadDepth = is3DReupload ? lvlDepth : (lvlDepth > 1 ? lvlDepth : 1);


            const void *srcData = (const void *)tex->faces[face].levels[level].data;

            void *expandedUploadData = NULL;

            /* Channel expansion for 2D/non-3D only.  3D expansion would

             * require per-slice handling (see DIRTY_TEXTURE_DATA 3D path). */

            if (!is3DReupload) {

                if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

                    NSUInteger expandedBytesPerRow = 0;

                    NSUInteger expandedBytesPerImage = 0;

                    expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,

                                                                      (const uint8_t *)srcData,

                                                                      lvlWidth,

                                                                      MAX((NSUInteger)lvlHeight, 1UL),

                                                                      bytesPerRow,

                                                                      &expandedBytesPerRow,

                                                                      &expandedBytesPerImage);

                    if (expandedUploadData) {

                        srcData = expandedUploadData;

                        bytesPerRow = expandedBytesPerRow;

                        bytesPerImage = expandedBytesPerImage;

                    }

                } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

                    NSUInteger expandedBytesPerRow = 0;

                    NSUInteger expandedBytesPerImage = 0;

                    expandedUploadData = mglCreateChannelExpandedUpload(tex,

                                                                         pixelFormat,

                                                                         (const uint8_t *)srcData,

                                                                         lvlWidth,

                                                                         MAX((NSUInteger)lvlHeight, 1UL),

                                                                         bytesPerRow,

                                                                         &expandedBytesPerRow,

                                                                         &expandedBytesPerImage);

                    if (expandedUploadData) {

                        srcData = expandedUploadData;

                        bytesPerRow = expandedBytesPerRow;

                        bytesPerImage = expandedBytesPerImage;

                    }

                }

            }

            /* Combined depth/stencil CPU shadows use a packed layout
             * (DEPTH32F_STENCIL8 = 5 bytes/texel) while the Metal texture
             * expects 8 bytes/texel with stencil at byte 4; repack here so
             * the non-array refresh path matches the dirty/array paths. */
            NSUInteger dsBytesPerRow = 0;
            NSUInteger dsBytesPerImage = 0;
            void *dsUploadData = mglCreateDepthStencilMetalUpload(
                tex, pixelFormat, (const uint8_t *)srcData,
                lvlWidth, MAX((NSUInteger)lvlHeight, 1UL),
                bytesPerRow, &dsBytesPerRow, &dsBytesPerImage);
            if (dsUploadData) {
                free(expandedUploadData);
                expandedUploadData = dsUploadData;
                srcData = dsUploadData;
                bytesPerRow = dsBytesPerRow;
                bytesPerImage = dsBytesPerImage;
            }

            NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];

            NSUInteger alignedBytesPerRow = bytesPerRow;

            if (alignedBytesPerRow % alignment != 0) {

                alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;

            }


            uintptr_t addr = (uintptr_t)srcData;

            if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {

                NSUInteger rowCount = MAX((NSUInteger)lvlHeight, 1UL);

                NSUInteger alignedSliceBPI = alignedBytesPerRow * rowCount;

                NSUInteger alignedSize = alignedSliceBPI * uploadDepth;

                if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {

                    void *alignedData = aligned_alloc(alignment, alignedSize);

                    if (alignedData) {

                        memset(alignedData, 0, alignedSize);

                        for (NSUInteger z = 0; z < uploadDepth; z++) {

                            for (NSUInteger row = 0; row < rowCount; row++) {

                                NSUInteger copySize = MIN(bytesPerRow, alignedBytesPerRow);

                                memcpy((uint8_t *)alignedData + z * alignedSliceBPI + row * alignedBytesPerRow,

                                       (const uint8_t *)srcData + z * singleSliceBPI + row * bytesPerRow, copySize);

                            }

                        }

                        [self uploadTextureSliceViaBlit:texture

                                               texName:tex->name

                                             texTarget:tex->target

                                                 bytes:alignedData

                                           bytesPerRow:alignedBytesPerRow

                                         bytesPerImage:alignedSliceBPI

                                                 width:lvlWidth

                                                height:lvlHeight

                                                 depth:uploadDepth

                                                 level:level

                                                 slice:face];

                        free(alignedData);

                    }

                }

            } else {

                [self uploadTextureSliceViaBlit:texture

                                       texName:tex->name

                                     texTarget:tex->target

                                         bytes:srcData

                                   bytesPerRow:bytesPerRow

                                 bytesPerImage:bytesPerImage

                                         width:lvlWidth

                                        height:lvlHeight

                                         depth:uploadDepth

                                         level:level

                                         slice:face];

            }

            free(expandedUploadData);

            } /* end else (non-array) */

        }

    }

}

- (void)fillTextureWithSafeInitialContents:(id)texture
                                         tex:(Texture *)tex
                                 pixelFormat:(uint32_t)pixelFormat
{


    if (mglTextureInfo(texture).width == 0 || mglTextureInfo(texture).height == 0 || mglTextureInfo(texture).width > 16384 || mglTextureInfo(texture).height > 16384) {

        NSLog(@"MGL WARNING: Skipping texture fill due to invalid dimensions: %lux%lu", (unsigned long)mglTextureInfo(texture).width, (unsigned long)mglTextureInfo(texture).height);

    } else {

        // Determine pixel format size to create appropriate black data

        NSUInteger bytesPerPixel = 4; // Default to RGBA

        switch(mglTextureInfo(texture).pixel_format) {

            case MGLPixelFormatR8Unorm:

            case MGLPixelFormatR8Uint:

            case MGLPixelFormatR8Sint:

                bytesPerPixel = 1;

                break;

            case MGLPixelFormatRG8Unorm:

            case MGLPixelFormatRG8Uint:

            case MGLPixelFormatRG8Sint:

                bytesPerPixel = 2;

                break;

            case MGLPixelFormatRGBA8Unorm:

            case MGLPixelFormatRGBA8Uint:

            case MGLPixelFormatRGBA8Sint:

                bytesPerPixel = 4;

                break;

            default:

                bytesPerPixel = 4; // Default assumption

                break;

        }


        // Calculate dynamic alignment for Metal textures based on pixel format

        NSUInteger bytesPerRow = mglTextureInfo(texture).width * bytesPerPixel;

        NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:mglTextureInfo(texture).pixel_format];

        if (bytesPerRow % alignment != 0) {

            bytesPerRow = ((bytesPerRow + alignment - 1) / alignment) * alignment;

        }


        NSUInteger dataSize = bytesPerRow * mglTextureInfo(texture).height;


        // Validate that dataSize is reasonable (not too large)

        if (dataSize > 64 * 1024 * 1024) { // 64MB limit per texture level

            NSLog(@"MGL WARNING: Skipping texture fill due to excessive size: %lu bytes", (unsigned long)dataSize);

        } else {

            // Allocate initialization data for texture clear.

            // aligned_alloc has been unreliable in this environment; calloc is safer here.

            (void)alignment;

            void *blackData = calloc(dataSize, 1);

            if (blackData) {

                // CRITICAL SECURITY FIX: Comprehensive validation to prevent Metal driver crashes

                // calloc already zero-initializes


                // Multi-layer validation for all parameters

                if (!blackData) {

                    NSLog(@"MGL SECURITY ERROR: blackData is NULL after memset - CORRUPTION DETECTED");

                    return;
                }

                if (bytesPerRow == 0) {

                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) for texture fill");

                    free(blackData);

                    return;
                }

                if (dataSize == 0) {

                    NSLog(@"MGL SECURITY ERROR: Invalid dataSize (0) for texture fill");

                    free(blackData);

                    return;
                }

                if (!texture) {

                    NSLog(@"MGL SECURITY ERROR: Metal texture is NULL");

                    free(blackData);

                    return;
                }

                if (mglTextureInfo(texture).width == 0 || mglTextureInfo(texture).height == 0) {

                    NSLog(@"MGL SECURITY ERROR: Invalid texture dimensions %lux%lu", (unsigned long)mglTextureInfo(texture).width, (unsigned long)mglTextureInfo(texture).height);

                    free(blackData);

                    return;
                }


                // Additional validation: verify blackData contains expected zeros (anti-corruption check)

                uint8_t *bytes = (uint8_t *)blackData;

                bool dataCorrupted = false;

                for (NSUInteger i = 0; i < MIN(dataSize, 1024); i++) { // Check first 1KB only for performance

                    if (bytes[i] != 0) {

                        dataCorrupted = true;

                        break;

                    }

                }

                if (dataCorrupted) {

                    NSLog(@"MGL SECURITY ERROR: blackData corruption detected - memory safety issue");

                    free(blackData);

                    return;
                }


                NSLog(@"MGL INFO: All validations passed for texture fill (size=%lu, bytesPerRow=%lu)", (unsigned long)dataSize, (unsigned long)bytesPerRow);


                // ULTRA-DEFENSIVE: Final validation immediately before Metal API call

                // This prevents race conditions and memory corruption between validation and use

                if (!blackData) {

                    NSLog(@"MGL CRITICAL ERROR: blackData became NULL before Metal call - RACE CONDITION DETECTED");

                    free(blackData);

                    return;
                }

                if (!texture) {

                    NSLog(@"MGL CRITICAL ERROR: Metal texture became NULL before Metal call - RACE CONDITION DETECTED");

                    free(blackData);

                    return;
                }

                if (bytesPerRow == 0 || dataSize == 0) {

                    NSLog(@"MGL CRITICAL ERROR: Parameters became invalid before Metal call - RACE CONDITION DETECTED");

                    free(blackData);

                    return;
                }


                // Additional verification: Check if Metal texture is still valid

                if (mglTextureInfo(texture).width == 0 || mglTextureInfo(texture).height == 0) {

                    NSLog(@"MGL CRITICAL ERROR: Metal texture dimensions became invalid before Metal call");

                    free(blackData);

                    return;
                }


                // Final integrity check: Verify blackData still contains expected zeros

                uint8_t *finalCheck = (uint8_t *)blackData;

                bool finalCorruption = false;

                for (NSUInteger i = 0; i < MIN(dataSize, 256); i++) { // Check first 256 bytes

                    if (finalCheck[i] != 0) {

                        finalCorruption = true;

                        break;

                    }

                }

                if (finalCorruption) {

                    NSLog(@"MGL CRITICAL ERROR: Memory corruption detected immediately before Metal call");

                    free(blackData);

                    return;
                }


                NSLog(@"MGL INFO: FIXING: Implementing proper texture filling for Apple Metal compatibility");


                // PROPER FIX: Use Apple Metal-compatible texture filling approach

                // The issue was using incorrect bytesPerRow and region parameters

                NSLog(@"MGL INFO: Implementing Metal-compliant texture fill operations");


                // Use Metal's standard pattern for texture filling.

                NSUInteger pixelSize = bytesPerPixel;

                NSUInteger properBytesPerRow = mglTextureInfo(texture).width * pixelSize;


                // Ensure proper alignment for Apple Metal driver

                if (properBytesPerRow % 64 != 0) {

                    properBytesPerRow = ((properBytesPerRow + 63) / 64) * 64;

                }


                // Fill the entire level. A previous 1x1 safety fill left large textures

                // mostly uninitialized while their Metal backing existed.

                MGLRegionValue properRegion = mglTextureRegion2D(0, 0, mglTextureInfo(texture).width, mglTextureInfo(texture).height);


                // Create properly aligned texture data buffer

                NSUInteger fillSize = properBytesPerRow * properRegion.size.height;

                uint8_t *properData = (uint8_t *)calloc(fillSize, 1);


                if (properData) {

                    // Initialize with safe texture data (transparent black with alpha = 0)

                    for (NSUInteger y = 0; y < properRegion.size.height; y++) {

                        uint8_t *row = properData + (y * properBytesPerRow);

                        for (NSUInteger x = 0; x < properRegion.size.width; x++) {

                            uint8_t *pixel = row + (x * pixelSize);

                            pixel[0] = 0;  // R

                            if (pixelSize > 1) pixel[1] = 0;  // G

                            if (pixelSize > 2) pixel[2] = 0;  // B

                            if (pixelSize > 3) pixel[3] = 0; // A = transparent for uninitialized color data

                        }

                    }


                    @try {

                        NSLog(@"MGL INFO: Performing Metal-compliant texture fill:");

                        NSLog(@"  - Region: %dx%d", (int)properRegion.size.width, (int)properRegion.size.height);

                        NSLog(@"  - bytesPerRow: %lu", (unsigned long)properBytesPerRow);

                        NSLog(@"  - dataSize: %lu", (unsigned long)fillSize);


                        // ALTERNATIVE APPROACH: Safe texture filling without replaceRegion

                        NSLog(@"MGL INFO: Using alternative texture filling methods (AGX-safe)");


                        @try {

                            // ALTERNATIVE 1: Try MTLBuffer-to-texture copy approach

                            if (properData && dataSize > 0) {

                                NSLog(@"MGL INFO: Attempting buffer-based texture fill");


                                // Create a temporary MTLBuffer with the texture data

                                id tempBuffer =
                                    mglTextureCreateBufferWithBytes(
                                        _device, properData, fillSize,
                                        MGL_TEXTURE_RESOURCE_STORAGE_SHARED);


                                if (tempBuffer) {

                                    NSLog(@"MGL INFO: Created temporary MTLBuffer for texture data");


                                    if ([self shouldSkipGPUOperations]) {

                                        NSLog(@"MGL AGX: Skipping texture fill during recovery - texture will be empty");

                                    } else {

                                        BOOL uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:tempBuffer

                                                                                              sourceOffset:0

                                                                                         sourceBytesPerRow:properBytesPerRow

                                                                                       sourceBytesPerImage:fillSize

                                                                                        sourceLayerStride:0

                                                                                                layerCount:1

                                                                                                 sourceSize:mglTextureSize(properRegion.size.width, properRegion.size.height, 1)

                                                                                                  toTexture:texture

                                                                                           destinationSlice:0

                                                                                           destinationLevel:0

                                                                                          destinationOrigin:mglTextureOrigin(0, 0, 0)

                                                                                                     reason:"texture_fill_initialization"];

                                        if (uploaded) {

                                            NSLog(@"MGL SUCCESS: Texture data copied using dedicated upload command buffer");

                                            mglMarkTextureLevelMetalFilled(tex, 0, fillSize);

                                        } else {

                                            NSLog(@"MGL WARNING: Dedicated texture fill upload failed - texture may remain uninitialized");

                                        }

                                    }


                                    // Clean up the temporary buffer

                                    tempBuffer = nil;

                                }

                            }

                        } @catch (NSException *exception) {

                            NSLog(@"MGL WARNING: Buffer-based texture fill failed - trying alternative");


                            // ALTERNATIVE 2: Simple direct color filling for basic cases

                            [self fillSmallRGBA8TextureWithGradient:texture tex:tex];

                        }

                    } @catch (NSException *exception) {

                        NSLog(@"MGL ERROR: Metal texture fill failed - investigating root cause");

                        NSLog(@"MGL ERROR: Exception: %@ (Reason: %@)", exception.name, exception.reason);

                        NSLog(@"MGL INFO: This indicates our parameters are still incompatible with AGX driver");

                    }


                    free(properData);

                } else {

                    NSLog(@"MGL ERROR: Failed to allocate properly aligned texture data");

                }

                free(blackData);

            } else {

                NSLog(@"MGL ERROR: Failed to allocate aligned memory for texture fill (%lu bytes)", (unsigned long)dataSize);

            }

        }

    }

}

- (BOOL)uploadDirtyCPUTextureData:(Texture *)tex
                            metal:(id)texture
                      pixelFormat:(uint32_t)pixelFormat
                        numFaces:(uint)num_faces
                uploadLevelCount:(GLuint)upload_level_count
                         isArray:(BOOL)is_array
              texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
        texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                         texType:(uint32_t)tex_type
            outAllLevelsUploaded:(BOOL *)outAllLevelsUploaded
{
    MGL_ASSERT_GL_THREAD();

    if (kMGLDiagnosticStateLogs) {
        mglTraceLogNSString(@"MGL DEBUG: DIRTY_TEXTURE_DATA detected - attempting texture filling");
        mglTraceLogNSString(@"MGL DEBUG: Texture details: target=0x%x, internalformat=0x%x, levels=%d effectiveLevels=%u",
                      tex->target, tex->internalformat, tex->num_levels, upload_level_count);
    }

    MGLRegionValue region;
    NSUInteger width, height, depth;
    BOOL anyLevelSkipped = NO;

    for(int face=0; face<num_faces; face++)
    {
        for (int level=0; level<upload_level_count; level++)
        {
            TextureLevel *uploadLevel = &tex->faces[face].levels[level];
            if (!mglTextureLevelHasUploadableCPUData(uploadLevel)) {
                static uint64_t s_skipStaleUploadLogs = 0;
                uint64_t hit = ++s_skipStaleUploadLogs;
                if (hit <= 8ull || (hit % 2048ull) == 0ull) {
                    NSLog(@"MGL TEXTURE SKIP stale CPU upload tex=%u face=%d level=%d source=%u ever=%u init=%u hit=%llu",
                          (unsigned)tex->name,
                          face,
                          level,
                          uploadLevel ? (unsigned)uploadLevel->last_init_source : 0u,
                          uploadLevel ? (unsigned)uploadLevel->ever_written : 0u,
                          uploadLevel ? (unsigned)uploadLevel->has_initialized_data : 0u,
                          (unsigned long long)hit);
                }
                anyLevelSkipped = YES;
                continue;
            }

            width = tex->faces[face].levels[level].width;
            height = tex->faces[face].levels[level].height;
            depth = tex->faces[face].levels[level].depth;

            if (texture1DBackedBy2D)
                region = mglTextureRegion2D(0,0,width,1);
            else if (depth > 1)
                region = mglTextureRegion3D(0,0,0,width,height,depth);
            else if (height > 1)
                region = mglTextureRegion2D(0,0,width,height);
            else
                region = mglTextureRegion1D(0,width);

            NSUInteger bytesPerRow;
            NSUInteger bytesPerImage;
            bool hasExplicitDataSize = false;

            BOOL levelSkipped = NO;

            if (tex_type == MGLTextureType3D)
            {
                if (![self uploadDirtyCPUTextureData3DLevel:tex
                                                       metal:texture
                                                 pixelFormat:pixelFormat
                                                       face:face
                                                      level:level
                                                      width:width
                                                     height:height
                                                      depth:depth
                                                 outSkipped:&levelSkipped]) {
                    return NO;
                }
            }
            else
            {
                if (![self uploadDirtyCPUTextureDataNon3DLevel:tex
                                                          metal:texture
                                                    pixelFormat:pixelFormat
                                                          face:face
                                                         level:level
                                                         width:width
                                                        height:height
                                                         depth:depth
                                                       isArray:is_array
                                  texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                                        texType:tex_type
                                                     outSkipped:&levelSkipped]) {
                    return NO;
                }
            }

            if (levelSkipped)
                anyLevelSkipped = YES;
        }
    }

    if (outAllLevelsUploaded)
        *outAllLevelsUploaded = !anyLevelSkipped;

    return YES;
}

- (void)reUploadExistingCPUTextureDataArrayLevel:(Texture *)tex
                                          metal:(id)texture
                                    pixelFormat:(uint32_t)pixelFormat
                                          face:(int)face
                                         level:(int)level
                  texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                                       texType:(uint32_t)tex_type
{
    NSUInteger lvlWidth  = tex->faces[face].levels[level].width;
    NSUInteger lvlHeight = tex->faces[face].levels[level].height;
    NSUInteger lvlPitch  = tex->faces[face].levels[level].pitch;


                /* Array texture re-upload: loop over array layers and upload

                 * each slice independently.  Mirrors the DIRTY_TEXTURE_DATA

                 * array path (12861-13087).  The old code only uploaded

                 * slice 0 and passed the entire array's data_size as

                 * bytesPerImage with depth=num_layers, causing a crash in

                 * uploadTextureSliceViaBlit's newBufferWithBytes. */

                GLuint num_layers = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray)

                    ? tex->faces[face].levels[level].height

                    : tex->faces[face].levels[level].depth;

                if (num_layers == 0) return;


                BOOL arraySliceIs1D = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray);

                NSUInteger uploadSliceHeight = arraySliceIs1D ? 1UL : MAX((NSUInteger)lvlHeight, 1UL);

                NSUInteger baseBytesPerRow = lvlPitch;

                NSUInteger uploadSliceRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);

                if (uploadSliceRows == 0 || baseBytesPerRow > (NSUIntegerMax / uploadSliceRows)) {

                    NSLog(@"MGL WARNING: Re-upload array invalid row layout tex=%d face=%d level=%d bpr=%lu rows=%lu",

                          tex->name,

                          face,

                          level,

                          (unsigned long)baseBytesPerRow,

                          (unsigned long)uploadSliceRows);

                    return;

                }

                NSUInteger logicalBytesPerImage = baseBytesPerRow * uploadSliceRows;

                NSUInteger backingBytes = tex->faces[face].levels[level].data_size;

                /* data_size is page-rounded; do not treat the slack as layer
                 * stride or reads land in the wrong slice. */

                NSUInteger requiredArrayBytes = 0;

                NSUInteger safeLayerCount = MAX((NSUInteger)num_layers, 1UL);

                if (logicalBytesPerImage == 0 ||

                    logicalBytesPerImage > (NSUIntegerMax / safeLayerCount) ||

                    backingBytes < (requiredArrayBytes = logicalBytesPerImage * safeLayerCount)) {

                    NSLog(@"MGL WARNING: Re-upload array backing too small tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u",

                          tex->name, face, level,

                          (unsigned long)backingBytes,

                          (unsigned long)logicalBytesPerImage,

                          num_layers);

                    return;

                }


                for (GLuint layer = 0; layer < num_layers; layer++)

                {

                    size_t offset = logicalBytesPerImage * layer;

                    const void *layerSrcData = (const uint8_t *)tex->faces[face].levels[level].data + offset;

                    void *expandedUploadData = NULL;

                    NSUInteger effectiveBytesPerRow = baseBytesPerRow;

                    NSUInteger effectiveBytesPerImage = logicalBytesPerImage;


                    if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

                        NSUInteger expandedBPR = 0, expandedBPI = 0;

                        expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,

                                                                          (const uint8_t *)layerSrcData,

                                                                          lvlWidth,

                                                                          uploadSliceHeight,

                                                                          baseBytesPerRow,

                                                                          &expandedBPR,

                                                                          &expandedBPI);

                        if (expandedUploadData) {

                            layerSrcData = expandedUploadData;

                            effectiveBytesPerRow = expandedBPR;

                            effectiveBytesPerImage = expandedBPI;

                        }

                    } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

                        NSUInteger expandedBPR = 0, expandedBPI = 0;

                        expandedUploadData = mglCreateChannelExpandedUpload(tex,

                                                                             pixelFormat,

                                                                             (const uint8_t *)layerSrcData,

                                                                             lvlWidth,

                                                                             uploadSliceHeight,

                                                                             baseBytesPerRow,

                                                                             &expandedBPR,

                                                                             &expandedBPI);

                        if (expandedUploadData) {

                            layerSrcData = expandedUploadData;

                            effectiveBytesPerRow = expandedBPR;

                            effectiveBytesPerImage = expandedBPI;

                        }

                    }

                    NSUInteger dsBytesPerRow = 0;
                    NSUInteger dsBytesPerImage = 0;
                    void *dsUploadData = mglCreateDepthStencilMetalUpload(
                        tex, pixelFormat, (const uint8_t *)layerSrcData,
                        lvlWidth, uploadSliceHeight, effectiveBytesPerRow,
                        &dsBytesPerRow, &dsBytesPerImage);
                    if (dsUploadData) {
                        free(expandedUploadData);
                        expandedUploadData = dsUploadData;
                        layerSrcData = dsUploadData;
                        effectiveBytesPerRow = dsBytesPerRow;
                        effectiveBytesPerImage = dsBytesPerImage;
                    }


                    NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];

                    NSUInteger alignedBytesPerRow = effectiveBytesPerRow;

                    if (alignedBytesPerRow % alignment != 0) {

                        alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;

                    }


                    uintptr_t addr = (uintptr_t)layerSrcData;

                    if (addr % alignment != 0 || alignedBytesPerRow != effectiveBytesPerRow) {

                        NSUInteger alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);

                        if (alignedUploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / alignedUploadRows)) {

                            NSLog(@"MGL WARNING: Re-upload array rejecting aligned row layout bpr=%lu rows=%lu tex=%d face=%d level=%d layer=%u",

                                  (unsigned long)alignedBytesPerRow,

                                  (unsigned long)alignedUploadRows,

                                  tex->name,

                                  face,

                                  level,

                                  layer);

                            free(expandedUploadData);

                            continue;

                        }

                        NSUInteger alignedSize = alignedBytesPerRow * alignedUploadRows;

                        if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {

                            void *alignedData = aligned_alloc(alignment, alignedSize);

                            if (alignedData) {

                                memset(alignedData, 0, alignedSize);

                                for (NSUInteger row = 0; row < alignedUploadRows; row++) {

                                    NSUInteger copySize = MIN(effectiveBytesPerRow, alignedBytesPerRow);

                                    memcpy((uint8_t *)alignedData + row * alignedBytesPerRow,

                                           (const uint8_t *)layerSrcData + row * effectiveBytesPerRow, copySize);

                                }

                                [self uploadTextureSliceViaBlit:texture

                                                       texName:tex->name

                                                     texTarget:tex->target

                                                         bytes:alignedData

                                                   bytesPerRow:alignedBytesPerRow

                                                 bytesPerImage:alignedSize

                                                         width:lvlWidth

                                                        height:lvlHeight

                                                         depth:1

                                                         level:level

                                                         slice:layer];

                                free(alignedData);

                            }

                        }

                    } else {

                        [self uploadTextureSliceViaBlit:texture

                                               texName:tex->name

                                             texTarget:tex->target

                                                 bytes:layerSrcData

                                           bytesPerRow:effectiveBytesPerRow

                                         bytesPerImage:effectiveBytesPerImage

                                                 width:lvlWidth

                                                height:lvlHeight

                                                 depth:1

                                                 level:level

                                                 slice:layer];

                    }

                    free(expandedUploadData);

                }

}

- (void)fillSmallRGBA8TextureWithGradient:(id)texture tex:(Texture *)tex
{
                            if (mglTextureInfo(texture).width <= 512 && mglTextureInfo(texture).height <= 512 && tex->internalformat == GL_RGBA8) {

                                NSLog(@"MGL INFO: Attempting simple direct color fill for small RGBA8 texture");


                                @try {

                                    // Create a simple pattern that's not magenta

                                    NSUInteger pixelCount = mglTextureInfo(texture).width * mglTextureInfo(texture).height;

                                    uint32_t *simpleData = calloc(pixelCount, sizeof(uint32_t));


                                    if (simpleData) {

                                        // Create a simple gradient pattern instead of magenta

                                        for (NSUInteger y = 0; y < mglTextureInfo(texture).height; y++) {

                                            for (NSUInteger x = 0; x < mglTextureInfo(texture).width; x++) {

                                                NSUInteger index = y * mglTextureInfo(texture).width + x;


                                                // Create a simple gradient from blue to green

                                                uint8_t r = (uint8_t)(x * 255 / mglTextureInfo(texture).width);

                                                uint8_t g = (uint8_t)(y * 255 / mglTextureInfo(texture).height);

                                                uint8_t b = 128;

                                                uint8_t a = 255;


                                                simpleData[index] = (a << 24) | (b << 16) | (g << 8) | r;

                                            }

                                        }


                                        // Try direct replaceRegion for simple cases

                                        MGLRegionValue simpleRegion = mglTextureRegion2D(0, 0, mglTextureInfo(texture).width, mglTextureInfo(texture).height);

                                        mglTextureReplaceRegion(
                                            texture, simpleRegion, 0, 0,
                                            simpleData,
                                            mglTextureInfo(texture).width * sizeof(uint32_t),
                                            mglTextureInfo(texture).width * mglTextureInfo(texture).height * sizeof(uint32_t),
                                            YES);


                                        NSLog(@"MGL SUCCESS: Simple direct color fill completed");

                                        mglMarkTextureLevelMetalFilled(tex, 0, pixelCount * sizeof(uint32_t));

                                        free(simpleData);

                                    }

                                } @catch (NSException *exception) {

                                    NSLog(@"MGL WARNING: Simple direct fill also failed: %@", exception.reason);

                                }

                            } else {

                                NSLog(@"MGL INFO: Skipping complex texture - would use deferred initialization");

                            }
}

- (BOOL)uploadDirtyCPUTextureData3DLevel:(Texture *)tex
                                    metal:(id)texture
                              pixelFormat:(uint32_t)pixelFormat
                                       face:(int)face
                                      level:(int)level
                                      width:(NSUInteger)width
                                     height:(NSUInteger)height
                                      depth:(NSUInteger)depth
                                 outSkipped:(BOOL *)outSkipped
{
    NSUInteger bytesPerRow;
    NSUInteger bytesPerImage;

                bytesPerRow = tex->faces[face].levels[level].pitch;
                if (bytesPerRow == 0) {
                    NSLog(@"MGL WARNING: Invalid 3D bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }

                NSUInteger uploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, MAX((NSUInteger)height, 1UL));
                if (uploadRows == 0 || bytesPerRow > (NSUIntegerMax / uploadRows)) {
                    NSLog(@"MGL WARNING: Invalid 3D bytesPerImage overflow (tex=%d face=%d level=%d rows=%lu bpr=%lu)",
                          tex->name,
                          face,
                          level,
                          (unsigned long)uploadRows,
                          (unsigned long)bytesPerRow);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }
                bytesPerImage = bytesPerRow * uploadRows;

                if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                    void *srcData = (void *)tex->faces[face].levels[level].data;
                    uintptr_t addr = (uintptr_t)srcData;

                    uint8_t *expanded3DUploadData = NULL;
                    if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                        NSUInteger expandedBytesPerRow = 0;
                        NSUInteger expandedBytesPerImagePerSlice = 0;
                        NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                        NSUInteger texHeight = MAX((NSUInteger)height, 1UL);

                        uint8_t *firstSlice = mglCreateRGBA8ExpandedUpload(tex,
                                                                           (const uint8_t *)srcData,
                                                                           width,
                                                                           texHeight,
                                                                           bytesPerRow,
                                                                           &expandedBytesPerRow,
                                                                           &expandedBytesPerImagePerSlice);
                        if (firstSlice) {
                            NSUInteger totalExpandedSize = expandedBytesPerImagePerSlice * texDepth;
                            if (totalExpandedSize > 0 && totalExpandedSize <= (512 * 1024 * 1024)) {
                                expanded3DUploadData = (uint8_t *)malloc(totalExpandedSize);
                                if (expanded3DUploadData) {
                                    memcpy(expanded3DUploadData, firstSlice, expandedBytesPerImagePerSlice);
                                    for (NSUInteger z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc = (const uint8_t *)srcData + z * bytesPerImage;
                                        uint8_t *sliceDst = expanded3DUploadData + z * expandedBytesPerImagePerSlice;
                                        NSUInteger dummyRow = 0, dummyImage = 0;
                                        uint8_t *sliceExpanded = mglCreateRGBA8ExpandedUpload(tex,
                                                                                             sliceSrc,
                                                                                             width,
                                                                                             texHeight,
                                                                                             bytesPerRow,
                                                                                             &dummyRow,
                                                                                             &dummyImage);
                                        if (sliceExpanded) {
                                            memcpy(sliceDst, sliceExpanded, expandedBytesPerImagePerSlice);
                                            free(sliceExpanded);
                                        } else {
                                            memset(sliceDst, 0, expandedBytesPerImagePerSlice);
                                        }
                                    }
                                    srcData = expanded3DUploadData;
                                    bytesPerRow = expandedBytesPerRow;
                                    bytesPerImage = expandedBytesPerImagePerSlice;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                        NSUInteger expandedBytesPerRow = 0;
                        NSUInteger expandedBytesPerImagePerSlice = 0;
                        NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                        NSUInteger texHeight = MAX((NSUInteger)height, 1UL);

                        uint8_t *firstSlice = mglCreateChannelExpandedUpload(tex,
                                                                              pixelFormat,
                                                                              (const uint8_t *)srcData,
                                                                              width,
                                                                              texHeight,
                                                                              bytesPerRow,
                                                                              &expandedBytesPerRow,
                                                                              &expandedBytesPerImagePerSlice);
                        if (firstSlice) {
                            NSUInteger totalExpandedSize = expandedBytesPerImagePerSlice * texDepth;
                            if (totalExpandedSize > 0 && totalExpandedSize <= (512 * 1024 * 1024)) {
                                expanded3DUploadData = (uint8_t *)malloc(totalExpandedSize);
                                if (expanded3DUploadData) {
                                    memcpy(expanded3DUploadData, firstSlice, expandedBytesPerImagePerSlice);
                                    for (NSUInteger z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc = (const uint8_t *)srcData + z * bytesPerImage;
                                        uint8_t *sliceDst = expanded3DUploadData + z * expandedBytesPerImagePerSlice;
                                        NSUInteger dummyRow = 0, dummyImage = 0;
                                        uint8_t *sliceExpanded = mglCreateChannelExpandedUpload(tex,
                                                                                                 pixelFormat,
                                                                                                 sliceSrc,
                                                                                                 width,
                                                                                                 texHeight,
                                                                                                 bytesPerRow,
                                                                                                 &dummyRow,
                                                                                                 &dummyImage);
                                        if (sliceExpanded) {
                                            memcpy(sliceDst, sliceExpanded, expandedBytesPerImagePerSlice);
                                            free(sliceExpanded);
                                        } else {
                                            memset(sliceDst, 0, expandedBytesPerImagePerSlice);
                                        }
                                    }
                                    srcData = expanded3DUploadData;
                                    bytesPerRow = expandedBytesPerRow;
                                    bytesPerImage = expandedBytesPerImagePerSlice;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    }

                    NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                    NSUInteger alignedBytesPerRow = bytesPerRow;
                    if (alignedBytesPerRow % alignment != 0) {
                        alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                    }

                    NSUInteger addrAlignment = MGLCapabilityTextureAlignment(&_capability);
                    if (addr % addrAlignment != 0 || alignedBytesPerRow != bytesPerRow) {
                        NSUInteger alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, MAX((NSUInteger)height, 1UL));
                        if (alignedUploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / alignedUploadRows)) {
                            NSLog(@"MGL WARNING: Rejecting aligned 3D upload row overflow (tex=%d level=%d rows=%lu bpr=%lu)",
                                  tex->name,
                                  level,
                                  (unsigned long)alignedUploadRows,
                                  (unsigned long)alignedBytesPerRow);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        NSUInteger alignedBytesPerImage = alignedBytesPerRow * alignedUploadRows;
                        NSUInteger alignedDepth = MAX((NSUInteger)depth, 1UL);
                        if (alignedBytesPerImage > (NSUIntegerMax / alignedDepth)) {
                            NSLog(@"MGL WARNING: Rejecting aligned 3D upload size overflow (tex=%d level=%d bpi=%lu depth=%lu)",
                                  tex->name,
                                  level,
                                  (unsigned long)alignedBytesPerImage,
                                  (unsigned long)alignedDepth);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        NSUInteger alignedSize = alignedBytesPerImage * alignedDepth;
                        if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                            NSLog(@"MGL WARNING: Rejecting aligned 3D upload staging size=%lu (tex=%d level=%d)",
                                  (unsigned long)alignedSize, tex->name, level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        void *alignedData = aligned_alloc(alignment, alignedSize);

                        if (alignedData) {
                            memset(alignedData, 0, alignedSize);
                            NSUInteger srcRowSize = bytesPerRow;
                            NSUInteger dstRowSize = alignedBytesPerRow;
                            NSUInteger texUploadRows = alignedUploadRows;
                            NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                            uint8_t *srcPtr = (uint8_t *)srcData;
                            uint8_t *dstPtr = (uint8_t *)alignedData;

                            for (NSUInteger z = 0; z < texDepth; z++) {
                                for (NSUInteger row = 0; row < texUploadRows; row++) {
                                    NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                    NSUInteger dstOffset = z * alignedBytesPerImage + row * dstRowSize;
                                    NSUInteger srcOffset = z * bytesPerImage + row * srcRowSize;
                                    memcpy(dstPtr + dstOffset, srcPtr + srcOffset, copySize);
                                    if (dstRowSize > copySize) {
                                        memset(dstPtr + dstOffset + copySize, 0, dstRowSize - copySize);
                                    }
                                }
                            }

                            if (!alignedData) {
                                NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (alignedBytesPerRow == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            @try {
                                BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                       texName:tex->name
                                                                     texTarget:tex->target
                                                                         bytes:alignedData
                                                                   bytesPerRow:alignedBytesPerRow
                                                                 bytesPerImage:alignedBytesPerImage
                                                                         width:width
                                                                        height:height
                                                                         depth:depth
                                                                         level:level
                                                                         slice:0];
                                if (!uploaded) {
                                    NSLog(@"MGL WARNING: 3D aligned blit upload failed (level %d, face %d)", level, face);
                                }
                            } @catch (NSException *exception) {
                                NSLog(@"MGL ERROR: Failed to upload aligned 3D texture data (level %d, face %d): %@", level, face, exception);
                            }
                            free(alignedData);
                        } else {
                            NSLog(@"MGL ERROR: Failed to allocate aligned memory for 3D texture upload");
                        }
                    } else {
                        if (!srcData) {
                            NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        if (bytesPerRow == 0) {
                            NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        if (bytesPerImage == 0) {
                            NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        @try {
                            BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                   texName:tex->name
                                                                 texTarget:tex->target
                                                                     bytes:srcData
                                                               bytesPerRow:bytesPerRow
                                                             bytesPerImage:bytesPerImage
                                                                     width:width
                                                                    height:height
                                                                     depth:depth
                                                                     level:level
                                                                     slice:0];
                            if (!uploaded) {
                                NSLog(@"MGL WARNING: 3D direct blit upload failed (level %d, face %d)", level, face);
                            }
                        } @catch (NSException *exception) {
                            NSLog(@"MGL ERROR: Failed to upload 3D texture data (level %d, face %d): %@", level, face, exception);
                        }
                    }
                    free(expanded3DUploadData);
                } else {
                    NSLog(@"MGL WARNING: Skipping 3D texture upload due to invalid data or parameters");
                }

    return YES;
}

- (BOOL)uploadDirtyCPUTextureDataNon3DLevel:(Texture *)tex
                                       metal:(id)texture
                                 pixelFormat:(uint32_t)pixelFormat
                                       face:(int)face
                                      level:(int)level
                                      width:(NSUInteger)width
                                     height:(NSUInteger)height
                                      depth:(NSUInteger)depth
                                   isArray:(BOOL)is_array
                  texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                                    texType:(uint32_t)tex_type
                                 outSkipped:(BOOL *)outSkipped
{
    NSUInteger bytesPerRow;
    NSUInteger bytesPerImage;
    bool hasExplicitDataSize = false;
    MGLRegionValue region;

                bytesPerRow = tex->faces[face].levels[level].pitch;
                if (bytesPerRow == 0) {
                    NSLog(@"MGL WARNING: Invalid bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }

                bytesPerImage = tex->faces[face].levels[level].data_size;
                hasExplicitDataSize = (bytesPerImage > 0);
                if (bytesPerImage == 0) {
                    NSUInteger fallbackHeight = (height > 0) ? (NSUInteger)height : 1;
                    bytesPerImage = bytesPerRow * fallbackHeight;
                    NSLog(@"MGL WARNING: data_size was 0, using fallback bytesPerImage=%lu (tex=%d face=%d level=%d)",
                          (unsigned long)bytesPerImage, tex->name, face, level);
                }
                if (bytesPerImage == 0) {
                    NSLog(@"MGL WARNING: Invalid bytesPerImage (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }

                if (is_array)
                {
                    GLuint num_layers;
                    size_t offset;
                    GLubyte *tex_data;
                    BOOL arraySliceIs1D;
                    NSUInteger uploadSliceHeight;
                    NSUInteger backingBytes;
                    NSUInteger logicalBytesPerImage;

                    num_layers = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray)
                        ? tex->faces[face].levels[level].height
                        : tex->faces[face].levels[level].depth;
                    if (num_layers == 0) {
                        NSLog(@"MGL WARNING: Array texture has 0 layers, skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                        if (outSkipped) *outSkipped = YES;
                        return YES;
                    }

                    arraySliceIs1D = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray);
                    uploadSliceHeight = arraySliceIs1D ? 1UL : MAX((NSUInteger)height, 1UL);
                    backingBytes = bytesPerImage;
                    NSUInteger uploadSliceRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);
                    if (uploadSliceRows == 0 || bytesPerRow > (NSUIntegerMax / uploadSliceRows)) {
                        NSLog(@"MGL WARNING: Array texture invalid row layout tex=%d face=%d level=%d bpr=%lu rows=%lu",
                              tex->name,
                              face,
                              level,
                              (unsigned long)bytesPerRow,
                              (unsigned long)uploadSliceRows);
                        if (outSkipped) *outSkipped = YES;
                        return YES;
                    }
                    logicalBytesPerImage = bytesPerRow * uploadSliceRows;
                    /* data_size is page-rounded; do not treat the slack as
                     * layer stride or reads land in the wrong slice. */
                    NSUInteger requiredArrayBytes = 0;
                    NSUInteger safeLayerCount = MAX((NSUInteger)num_layers, 1UL);
                    if (logicalBytesPerImage == 0 ||
                        logicalBytesPerImage > (NSUIntegerMax / safeLayerCount) ||
                        backingBytes < (requiredArrayBytes = logicalBytesPerImage * safeLayerCount)) {
                        NSLog(@"MGL WARNING: Array texture backing too small for logical slices tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u",
                              tex->name,
                              face,
                              level,
                              (unsigned long)backingBytes,
                              (unsigned long)logicalBytesPerImage,
                              num_layers);
                        if (outSkipped) *outSkipped = YES;
                        return YES;
                    }
                    bytesPerImage = logicalBytesPerImage;

                    if (!arraySliceIs1D)
                        region = mglTextureRegion2D(0,0,width,height);
                    else if (height >= 1)
                        region = mglTextureRegion2D(0,0,width,1);
                    else {
                        NSLog(@"MGL TEXTURE ERROR: invalid array texture height=%lu for tex=%u face=%d level=%d",
                              (unsigned long)height,
                              tex->name,
                              face,
                              level);
                        return NO;
                    }

                    for(int layer=0; layer<num_layers; layer++)
                    {
                        offset = bytesPerImage * layer;

                        tex_data = (GLubyte *)tex->faces[face].levels[level].data;
                        tex_data += offset;

                        if (tex_data && bytesPerRow > 0 && bytesPerImage > 0) {
                            void *srcData = (void *)tex_data;
                            void *expandedUploadData = NULL;
                            uintptr_t addr = (uintptr_t)srcData;

                            NSUInteger effectiveBytesPerRow = bytesPerRow;
                            NSUInteger effectiveBytesPerImage = bytesPerImage;
                            if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                                NSUInteger expandedBytesPerRow = 0;
                                NSUInteger expandedBytesPerImage = 0;
                                expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,
                                                                                   (const uint8_t *)srcData,
                                                                                   width,
                                                                                   uploadSliceHeight,
                                                                                   bytesPerRow,
                                                                                   &expandedBytesPerRow,
                                                                                   &expandedBytesPerImage);
                                if (expandedUploadData) {
                                    srcData = expandedUploadData;
                                    effectiveBytesPerRow = expandedBytesPerRow;
                                    effectiveBytesPerImage = expandedBytesPerImage;
                                    addr = (uintptr_t)srcData;
                                }
                            } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                                NSUInteger expandedBytesPerRow = 0;
                                NSUInteger expandedBytesPerImage = 0;
                                expandedUploadData = mglCreateChannelExpandedUpload(tex,
                                                                                     pixelFormat,
                                                                                     (const uint8_t *)srcData,
                                                                                     width,
                                                                                     uploadSliceHeight,
                                                                                     bytesPerRow,
                                                                                     &expandedBytesPerRow,
                                                                                     &expandedBytesPerImage);
                                if (expandedUploadData) {
                                    srcData = expandedUploadData;
                                    effectiveBytesPerRow = expandedBytesPerRow;
                                    effectiveBytesPerImage = expandedBytesPerImage;
                                    addr = (uintptr_t)srcData;
                                }
                            }

                            NSUInteger dsBytesPerRow = 0;
                            NSUInteger dsBytesPerImage = 0;
                            void *dsUploadData = mglCreateDepthStencilMetalUpload(
                                tex, pixelFormat, (const uint8_t *)srcData,
                                width, uploadSliceHeight, effectiveBytesPerRow,
                                &dsBytesPerRow, &dsBytesPerImage);
                            if (dsUploadData) {
                                free(expandedUploadData);
                                expandedUploadData = dsUploadData;
                                srcData = dsUploadData;
                                effectiveBytesPerRow = dsBytesPerRow;
                                effectiveBytesPerImage = dsBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }

                            NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                            NSUInteger alignedBytesPerRow = effectiveBytesPerRow;
                            if (alignedBytesPerRow % alignment != 0) {
                                alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                            }

                            if (addr % alignment != 0 || alignedBytesPerRow != effectiveBytesPerRow) {
                                NSUInteger alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);
                                if (alignedUploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / alignedUploadRows)) {
                                    NSLog(@"MGL WARNING: Rejecting aligned array upload row layout bpr=%lu rows=%lu (tex=%d level=%d layer=%d)",
                                          (unsigned long)alignedBytesPerRow,
                                          (unsigned long)alignedUploadRows,
                                          tex->name,
                                          level,
                                          layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                NSUInteger alignedBytesPerImage = alignedBytesPerRow * alignedUploadRows;
                                NSUInteger alignedSize = alignedBytesPerImage;
                                if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                    NSLog(@"MGL WARNING: Rejecting aligned array upload staging size=%lu (tex=%d level=%d layer=%d)",
                                          (unsigned long)alignedSize, tex->name, level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                void *alignedData = aligned_alloc(alignment, alignedSize);

                                if (alignedData) {
                                    memset(alignedData, 0, alignedSize);
                                    NSUInteger srcRowSize = effectiveBytesPerRow;
                                    NSUInteger dstRowSize = alignedBytesPerRow;
                                    uint8_t *srcPtr = (uint8_t *)srcData;
                                    uint8_t *dstPtr = (uint8_t *)alignedData;

                                    for (NSUInteger row = 0; row < alignedUploadRows; row++) {
                                        NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                        memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                        if (dstRowSize > copySize) {
                                            memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                        }
                                    }

                                    if (!alignedData) {
                                        NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    if (alignedBytesPerRow == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    if (bytesPerImage == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    @try {
                                        if (hasExplicitDataSize) {
                                            BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                                   texName:tex->name
                                                                                 texTarget:tex->target
                                                                                     bytes:alignedData
                                                                               bytesPerRow:alignedBytesPerRow
                                                                             bytesPerImage:alignedBytesPerImage
                                                                                     width:width
                                                                                    height:uploadSliceHeight
                                                                                     depth:1
                                                                                     level:level
                                                                                     slice:layer];
                                            if (!uploaded) {
                                                NSLog(@"MGL WARNING: Array texture blit upload failed (level %d, layer %d)", level, layer);
                                            }
                                        } else {
                                            NSLog(@"MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)", level, layer);
                                        }
                                    } @catch (NSException *exception) {
                                        NSLog(@"MGL ERROR: Failed to upload aligned array texture data (level %d, layer %d): %@", level, layer, exception);
                                    }
                                    free(alignedData);
                                } else {
                                    NSLog(@"MGL ERROR: Failed to allocate aligned memory for array texture upload (level %d, layer %d)", level, layer);
                                }
                            } else {
                                if (!srcData) {
                                    NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerRow == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerImage == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (hasExplicitDataSize) {
                                    BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                           texName:tex->name
                                                                         texTarget:tex->target
                                                                             bytes:srcData
                                                                       bytesPerRow:effectiveBytesPerRow
                                                                     bytesPerImage:effectiveBytesPerImage
                                                                             width:width
                                                                            height:uploadSliceHeight
                                                                             depth:1
                                                                             level:level
                                                                             slice:layer];
                                    if (!uploaded) {
                                        NSLog(@"MGL WARNING: Array texture direct blit upload failed (level %d, layer %d)", level, layer);
                                    }
                                } else {
                                    NSLog(@"MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)", level, layer);
                                }
                            }
                            free(expandedUploadData);
                        } else {
                            NSLog(@"MGL WARNING: Skipping array texture upload due to invalid data or parameters");
                        }
                    }
                }
                else
                {
                    DEBUG_PRINT("tex id data update %d\n", tex->name);

                    if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                        void *srcData = (void *)tex->faces[face].levels[level].data;
                        void *swizzledUploadData = NULL;
                        void *expandedUploadData = NULL;
                        uintptr_t addr = (uintptr_t)srcData;
                        if (level == 0 && face == 0 && mglTextureUploadNeedsSingleChannelSwizzle(tex)) {
                            NSUInteger swizzledBytesPerRow = 0;
                            NSUInteger swizzledBytesPerImage = 0;
                            swizzledUploadData = mglCreateSingleChannelSwizzledUpload(tex,
                                                                                      (const uint8_t *)srcData,
                                                                                      width,
                                                                                      MAX((NSUInteger)height, 1UL),
                                                                                      bytesPerRow,
                                                                                      &swizzledBytesPerRow,
                                                                                      &swizzledBytesPerImage);
                            if (swizzledUploadData) {
                                srcData = swizzledUploadData;
                                bytesPerRow = swizzledBytesPerRow;
                                bytesPerImage = swizzledBytesPerImage;
                                addr = (uintptr_t)srcData;
                                if (mglTraceLogIsEnabled()) {
                                    const uint8_t *swz = (const uint8_t *)swizzledUploadData;
                                    mglTraceLog("TEXTURE_SWIZZLE_UPLOAD_R8 tex=%u target=0x%x swzR=0x%x size=%lux%lu bpr=%lu first=%u",
                                                (unsigned)tex->name,
                                                (unsigned)tex->target,
                                                (unsigned)tex->params.swizzle_r,
                                                (unsigned long)width,
                                                (unsigned long)MAX((NSUInteger)height, 1UL),
                                                (unsigned long)bytesPerRow,
                                                swz[0]);
                                }
                            }
                        }
                        if (!swizzledUploadData &&
                            mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                            NSUInteger expandedBytesPerRow = 0;
                            NSUInteger expandedBytesPerImage = 0;
                            expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,
                                                                               (const uint8_t *)srcData,
                                                                               width,
                                                                               MAX((NSUInteger)height, 1UL),
                                                                               bytesPerRow,
                                                                               &expandedBytesPerRow,
                                                                               &expandedBytesPerImage);
                            if (expandedUploadData) {
                                srcData = expandedUploadData;
                                bytesPerRow = expandedBytesPerRow;
                                bytesPerImage = expandedBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }
                        } else if (!swizzledUploadData &&
                                   mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                            NSUInteger expandedBytesPerRow = 0;
                            NSUInteger expandedBytesPerImage = 0;
                            expandedUploadData = mglCreateChannelExpandedUpload(tex,
                                                                                 pixelFormat,
                                                                                 (const uint8_t *)srcData,
                                                                                 width,
                                                                                 MAX((NSUInteger)height, 1UL),
                                                                                 bytesPerRow,
                                                                                 &expandedBytesPerRow,
                                                                                 &expandedBytesPerImage);
                            if (expandedUploadData) {
                                srcData = expandedUploadData;
                                bytesPerRow = expandedBytesPerRow;
                                bytesPerImage = expandedBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }
                        }

                        NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                        NSUInteger alignedBytesPerRow = bytesPerRow;
                        if (alignedBytesPerRow % alignment != 0) {
                            alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                        }

                        if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {
                            NSUInteger texHeight = MAX((NSUInteger)height, 1UL);
                            NSUInteger uploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, texHeight);
                            if (uploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / uploadRows)) {
                                NSLog(@"MGL WARNING: Rejecting aligned 2D upload row layout bpr=%lu rows=%lu (tex=%d level=%d face=%d)",
                                      (unsigned long)alignedBytesPerRow,
                                      (unsigned long)uploadRows,
                                      tex->name,
                                      level,
                                      face);
                                free(swizzledUploadData);
                                free(expandedUploadData);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            NSUInteger alignedBytesPerImage = alignedBytesPerRow * uploadRows;
                            NSUInteger alignedSize = alignedBytesPerImage;
                            if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                NSLog(@"MGL WARNING: Rejecting aligned 2D upload staging size=%lu (tex=%d level=%d face=%d)",
                                      (unsigned long)alignedSize, tex->name, level, face);
                                free(swizzledUploadData);
                                free(expandedUploadData);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            void *alignedData = aligned_alloc(alignment, alignedSize);

                            if (alignedData) {
                                memset(alignedData, 0, alignedSize);
                                NSUInteger srcRowSize = bytesPerRow;
                                NSUInteger dstRowSize = alignedBytesPerRow;
                                uint8_t *srcPtr = (uint8_t *)srcData;
                                uint8_t *dstPtr = (uint8_t *)alignedData;

                                for (NSUInteger row = 0; row < uploadRows; row++) {
                                    NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                    memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                    if (dstRowSize > copySize) {
                                        memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                    }
                                }

                                if (!alignedData) {
                                    NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    free(alignedData);
                                    if (outSkipped) *outSkipped = YES;
                                    return YES;
                                }
                                if (alignedBytesPerRow == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    free(alignedData);
                                    if (outSkipped) *outSkipped = YES;
                                    return YES;
                                }
                                if (bytesPerImage == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    free(alignedData);
                                    if (outSkipped) *outSkipped = YES;
                                    return YES;
                                }
                                if (hasExplicitDataSize) {
                                    BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                           texName:tex->name
                                                                         texTarget:tex->target
                                                                             bytes:alignedData
                                                                       bytesPerRow:alignedBytesPerRow
                                                                     bytesPerImage:alignedBytesPerImage
                                                                             width:width
                                                                            height:height
                                                                             depth:1
                                                                             level:level
                                                                             slice:face];
                                    if (!uploaded) {
                                        NSLog(@"MGL WARNING: Aligned 2D blit upload failed (level %d, face %d)", level, face);
                                    }
                                } else {
                                    NSLog(@"MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)", level, face);
                                }
                                free(alignedData);
                            } else {
                                NSLog(@"MGL ERROR: Failed to allocate aligned memory for 2D texture upload (level %d, face %d)", level, face);
                            }
                        } else {
                            if (!srcData) {
                                NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (bytesPerRow == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (bytesPerImage == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (hasExplicitDataSize) {
                                BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                       texName:tex->name
                                                                     texTarget:tex->target
                                                                         bytes:srcData
                                                                   bytesPerRow:bytesPerRow
                                                                 bytesPerImage:bytesPerImage
                                                                         width:width
                                                                        height:height
                                                                         depth:1
                                                                         level:level
                                                                         slice:face];
                                if (!uploaded) {
                                    NSLog(@"MGL WARNING: 2D direct blit upload failed (level %d, face %d)", level, face);
                                }
                            } else {
                                NSLog(@"MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)", level, face);
                            }
                        }
                        free(swizzledUploadData);
                        free(expandedUploadData);
                    } else {
                        NSLog(@"MGL WARNING: Skipping 2D texture upload due to invalid data or parameters");
                    }
                }

    return YES;
}


- (void)swizzleTexDesc:(MGLRenderTextureDescriptorState *)tex_desc forTex:(Texture*)tex
{
    tex_desc->swizzle_red = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_r);
    tex_desc->swizzle_green = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_g);
    tex_desc->swizzle_blue = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_b);
    tex_desc->swizzle_alpha = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_a);
    tex_desc->has_swizzle = 1u;
}




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
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: GPU operations temporarily suspended during recovery");
        return nil;
    }

    // Validate texture dimensions to prevent Metal assertion failures.
    // Texture buffers (GL_TEXTURE_BUFFER) can have very large widths (millions of texels)
    // since they map to MGLTextureTypeTextureBuffer which uses GPU address space.
    if (tex->target != GL_TEXTURE_BUFFER) {
        if (!tex || tex->width <= 0 || tex->height <= 0 ||
            tex->width > 32768 || tex->height > 32768 || tex->depth > 32768) {
            NSLog(@"MGL ERROR: Invalid texture dimensions %dx%dx%d - rejecting",
                  tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->depth : 0);
            tex->dirty_bits = 0;
            return nil;
        }
    }

    if (tex->target == GL_TEXTURE_BUFFER) {
        return [self createMTLTexelBufferTexture:tex];
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

    if (![self checkTextureCompleteness:tex
                               texType:tex_type
                              numFaces:num_faces
                  effectiveMipmapLevels:&effective_mipmap_levels
                      storageMipmapped:&storageMipmapped]) {
        return nil;
    }

    // PROPER FIX: Get original texture format and validate for AGX compatibility
    pixelFormat = mtlPixelFormatForGLTex(tex);
    BOOL expandsSingleChannelSwizzle = mglTextureUploadNeedsSingleChannelSwizzle(tex);
    if (expandsSingleChannelSwizzle) {
        pixelFormat = MGLPixelFormatRGBA8Unorm;
    }

    // Validate format compatibility with AGX, but preserve original intent
    BOOL needsFormatConversion = NO;
    uint32_t originalFormat = pixelFormat;

    // Check for AGX-incompatible formats and only convert when necessary
    switch(pixelFormat) {
        case MGLPixelFormatB5G6R5Unorm:
        case MGLPixelFormatBGR5A1Unorm:
        case MGLPixelFormatA1BGR5Unorm:
            // 16-bit formats can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MGLPixelFormatRGBA8Unorm;
            break;
        case MGLPixelFormatPVRTC_RGBA_2BPP:
        case MGLPixelFormatPVRTC_RGBA_4BPP:
        case MGLPixelFormatPVRTC_RGB_2BPP:
        case MGLPixelFormatPVRTC_RGB_4BPP:
            // PVRTC compression can cause issues in virtualization
            needsFormatConversion = YES;
            pixelFormat = MGLPixelFormatRGBA8Unorm;
            break;
        case MGLPixelFormatEAC_R11Unorm:
        case MGLPixelFormatEAC_RG11Unorm:
        case MGLPixelFormatEAC_RGBA8:
        case MGLPixelFormatETC2_RGB8:
        case MGLPixelFormatETC2_RGB8A1:
            // ETC/ETC2 compression can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MGLPixelFormatRGBA8Unorm;
            break;
        default:
            // Most modern formats should work fine
            break;
    }

    /* Metal does not allow depth/stencil pixel formats with MGLTextureType1DArray.
     * Promote to MGLTextureType2DArray with height=1, mirroring how mipmapped
     * 1D array textures are already promoted below.  Without this, creating a
     * GL_TEXTURE_1D_ARRAY depth texture (e.g. sampler_1d_array_shadow) triggers
     * a Metal validation assertion crash. */
    if (tex_type == MGLTextureType1DArray) {
        switch (pixelFormat) {
            case MGLPixelFormatDepth16Unorm:
            case MGLPixelFormatDepth32Float:
            case MGLPixelFormatStencil8:
            case MGLPixelFormatDepth24Unorm_Stencil8:
            case MGLPixelFormatDepth32Float_Stencil8:
            case MGLPixelFormatX32_Stencil8:
            case MGLPixelFormatX24_Stencil8:
                tex_type = MGLTextureType2DArray;
                texture1DArrayBackedBy2DArray = true;
                break;
            default:
                break;
        }
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
    upload_level_count =
        (mipmapped && tex->mipmapped) ? effective_mipmap_levels : 1u;

    tex_desc.texture_type = tex_type;
    tex_desc.pixel_format = pixelFormat;
    tex_desc.width = width;
    tex_desc.height = (tex_type == MGLTextureType1D ||
                       tex_type == MGLTextureType1DArray) ? 1 : height;
    if (tex_type == MGLTextureType2DMultisample ||
        tex_type == MGLTextureType2DMultisampleArray) {

        NSUInteger samples = MAX((NSUInteger)2u, (NSUInteger)tex->samples);
        samples = MGLCapabilityClampSampleCount(&_capability, samples);
        tex_desc.sample_count = samples;
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
        (needsCpuUpload || preferSharedDepthStencil) ? 0u : MGL_TEXTURE_STORAGE_PRIVATE;
    tex_desc.sample_count = MAX(tex_desc.sample_count, 1u);
    tex_desc.mipmap_level_count = MAX(tex_desc.mipmap_level_count, 1u);
    tex_desc.array_length = MAX(tex_desc.array_length, 1u);
    tex_desc.depth = MAX(tex_desc.depth, 1u);

    // Normalize depth/array semantics per Metal texture type.
    if (tex_type == MGLTextureTypeCube) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }
        tex_desc.depth = 1;
    } else if (tex_type == MGLTextureTypeCubeArray) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube-array texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }

        // GL cube-map-array depth is usually layer count (faces), so convert to cube count.
        // If depth is already cube-count (non-multiple of 6), keep it as-is.
        NSUInteger cubeCount = depth;
        if (cubeCount >= 6 && (cubeCount % 6) == 0) {
            cubeCount = cubeCount / 6;
        } else if (cubeCount > 1 && (cubeCount % 6) != 0) {
            NSLog(@"MGL WARNING: cube-array depth=%lu is not a multiple of 6, treating as cube count",
                  (unsigned long)cubeCount);
        }

        tex_desc.array_length = MAX((NSUInteger)1, cubeCount);
        tex_desc.depth = 1;
    } else if (tex_type == MGLTextureType1DArray) {
        tex_desc.array_length = MAX((NSUInteger)1, height);
        tex_desc.depth = 1;
    } else if (is_array) {
        tex_desc.array_length = MAX((NSUInteger)1, depth);
        tex_desc.depth = 1;
    } else {
        /* For 3D and other non-array textures, arrayLength must be 1.
         * Some Metal drivers report getNumSlices()==0 when arrayLength
         * is left at its default, causing "slice OOB" assertions. */
        tex_desc.array_length = 1;
        tex_desc.depth = MAX((NSUInteger)1, depth);
    }

    if (mipmapped)
    {
        if (tex_type == MGLTextureType1D) {
            tex_type = MGLTextureType2D;
            texture1DBackedBy2D = true;
        }
        /* Metal does not allow mipmapLevelCount > 1 for MGLTextureType1DArray.
         * Promote to MGLTextureType2DArray with height=1 to support mipmapped
         * 1D array textures.  The upload code checks texture1DArrayBackedBy2DArray
         * to treat each slice as 1 pixel tall. */
        if (tex_type == MGLTextureType1DArray) {
            tex_type = MGLTextureType2DArray;
            texture1DArrayBackedBy2DArray = true;
        }
        tex_desc.mipmap_level_count = MAX((GLuint)1, effective_mipmap_levels);
    }

    if (texture1DBackedBy2D) {
        tex_desc.texture_type = MGLTextureType2D;
        tex_desc.height = 1;
    }
    if (texture1DArrayBackedBy2DArray) {
        tex_desc.texture_type = MGLTextureType2DArray;
        /* For GL_TEXTURE_1D_ARRAY, the GL height parameter is the array slice
         * count.  Since tex_type was promoted to MGLTextureType2DArray above,
         * the arrayLength branch at line ~12397 (keyed on MGLTextureType1DArray)
         * was skipped, leaving arrayLength=1 from the is_array/depth fallback.
         * Set arrayLength from the GL height (slice count) here. */
        tex_desc.array_length = MAX((NSUInteger)1, height);
        tex_desc.height = 1;
    }

    /* GL image access mode (GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE)
     * only governs the image binding, NOT the texture's overall capabilities.
     * A texture bound as a write-only image may still be sampled from via
     * sampler2D in the same shader.  Metal requires MGL_TEXTURE_USAGE_SHADER_READ
     * for sampling, so always include it alongside the image write flag. */
    switch(tex->access)
    {
        case GL_READ_ONLY:
            tex_desc.usage = MGL_TEXTURE_USAGE_SHADER_READ; break;
        case GL_WRITE_ONLY:
            tex_desc.usage = MGL_TEXTURE_USAGE_SHADER_READ | MGL_TEXTURE_USAGE_SHADER_WRITE; break;
        case GL_READ_WRITE:
            tex_desc.usage = MGL_TEXTURE_USAGE_SHADER_READ | MGL_TEXTURE_USAGE_SHADER_WRITE; break;
        default:
            NSLog(@"MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u",
                  tex->access,
                  tex->name);
            return nil;
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

    if (tex->params.swizzled && !expandsSingleChannelSwizzle)
    {
        [self swizzleTexDesc:&tex_desc forTex:tex];
    }

    id texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    @try {
        texture = mglTextureCreateTexture(_device, &tex_desc);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating texture: %@", exception);
        [self recordGPUError];
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
        if (![self uploadDirtyCPUTextureData:tex
                                       metal:texture
                                 pixelFormat:pixelFormat
                                   numFaces:num_faces
                           uploadLevelCount:upload_level_count
                                    isArray:is_array
                         texture1DBackedBy2D:texture1DBackedBy2D
                   texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                    texType:tex_type
                        outAllLevelsUploaded:&allLevelsUploaded]) {
            return nil;
        }
    }
    else
    {
        if (hasUploadableCPUData) {
            [self reUploadExistingCPUTextureData:tex
                                            metal:texture
                                      pixelFormat:pixelFormat
                                        numFaces:num_faces
                                uploadLevelCount:upload_level_count
                                          isArray:is_array
                               texture1DBackedBy2D:texture1DBackedBy2D
                         texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                             texType:tex_type];
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
            [self fillTextureWithSafeInitialContents:texture
                                                 tex:tex
                                         pixelFormat:pixelFormat];
        }
    }

    if (cpuUploadRequired && tex->target == GL_TEXTURE_2D && mglTextureInfo(texture).texture_type == MGLTextureType2D) {
        BOOL fullCPUUploadVerified = [self uploadFullCPUTextureDataIntoTexture:tex
                                                                           metal:texture
                                                                          reason:"createMTLTexture.cpuData"];
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

    [self logMTLTextureMipDiagnostics:tex metal:texture effectiveMipLevels:effective_mipmap_levels];

    [self recordGPUSuccess];

    return texture;
}

- (id)createMTLTexelBufferTexture:(Texture *)tex
{
    Buffer *sourceBuffer = tex->texture_buffer;
    if (!sourceBuffer || tex->texture_buffer_size <= 0) {
        NSLog(@"MGL TEXBUFFER ERROR: tex=%u has no attached buffer/size buffer=%p size=%lld",
              tex->name,
              sourceBuffer,
              (long long)tex->texture_buffer_size);
        return nil;
    }

    if (tex->texture_buffer_offset < 0 ||
        tex->texture_buffer_offset > sourceBuffer->size ||
        tex->texture_buffer_size > sourceBuffer->size - tex->texture_buffer_offset) {
        NSLog(@"MGL TEXBUFFER ERROR: invalid range tex=%u buffer=%u off=%lld size=%lld bufferSize=%lld",
              tex->name,
              sourceBuffer->name,
              (long long)tex->texture_buffer_offset,
              (long long)tex->texture_buffer_size,
              (long long)sourceBuffer->size);
        return nil;
    }

    NSUInteger bytesPerTexel = [self bytesPerPixelForFormat:tex->internalformat];
    if (bytesPerTexel == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: unsupported internal format 0x%x tex=%u buffer=%u",
              tex->internalformat,
              tex->name,
              sourceBuffer->name);
        return nil;
    }

    NSUInteger texelCount = (NSUInteger)tex->texture_buffer_size / bytesPerTexel;
    if (texelCount == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: zero texel count tex=%u buffer=%u size=%lld bpt=%lu",
              tex->name,
              sourceBuffer->name,
              (long long)tex->texture_buffer_size,
              (unsigned long)bytesPerTexel);
        return nil;
    }

    uint32_t bufferPixelFormat = (tex->internalformat == GL_RGBA8)
        ? MGLPixelFormatRGBA8Uint
        : mtlPixelFormatForGLTex(tex);
    if (bufferPixelFormat == MGLPixelFormatInvalid || bufferPixelFormat == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: invalid Metal format for tex=%u internal=0x%x",
              tex->name,
              tex->internalformat);
        return nil;
    }

    if (![self processBuffer:sourceBuffer]) {
        NSLog(@"MGL TEXBUFFER ERROR: failed to process source buffer tex=%u buffer=%u",
              tex->name,
              sourceBuffer->name);
        return nil;
    }

    const uint8_t *sourceBytes = NULL;
    if (sourceBuffer->data.buffer_data) {
        sourceBytes = ((const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data) + (size_t)tex->texture_buffer_offset;
    } else if (sourceBuffer->data.mtl_data) {
        id mtlBuffer = (__bridge id)(sourceBuffer->data.mtl_data);
        if (mtlBuffer && mglTextureBufferContents(mtlBuffer)) {
            sourceBytes = ((const uint8_t *)mglTextureBufferContents(mtlBuffer)) + (size_t)tex->texture_buffer_offset;
        }
    }

    if (!sourceBytes) {
        NSLog(@"MGL TEXBUFFER ERROR: no readable backing for tex=%u buffer=%u cpu=%p mtl=%p",
              tex->name,
              sourceBuffer->name,
              (void *)(uintptr_t)sourceBuffer->data.buffer_data,
              sourceBuffer->data.mtl_data);
        return nil;
    }

    // The AIR backend emits Minecraft's CloudFaces texel buffer as a
    // texture2d<int>. Keep GL lookup semantics as GL_TEXTURE_BUFFER, but
    // create a Metal 2D backing so the generated MSL argument type matches.
    // A texel buffer can be much wider than Metal's max 2D texture width,
    // so pack it into rows instead of creating texelCount x 1.
    /*
     * The AIR backend lowers GL texture buffers to 2D Metal textures and emits
     * spvTexelBufferCoord(tc) using its MSL texel_buffer_texture_width
     * option. Keep this packing width in lockstep with program.c.
     */
    static const NSUInteger kMGLTexelBufferTextureWidth = 4096u;
    NSUInteger max2DSize = (NSUInteger)MIN((GLuint)kMGLTexelBufferTextureWidth,
                                           ctx ? MGL_STATE(ctx)->var.max_texture_size : (GLuint)kMGLTexelBufferTextureWidth);
    if (max2DSize == 0 || max2DSize > kMGLTexelBufferTextureWidth) {
        max2DSize = kMGLTexelBufferTextureWidth;
    }

    NSUInteger texWidth = MIN(texelCount, max2DSize);
    NSUInteger texHeight = (texelCount + texWidth - 1) / texWidth;
    if (texHeight == 0 || texHeight > max2DSize) {
        NSLog(@"MGL TEXBUFFER ERROR: texel buffer too large for 2D fallback tex=%u buffer=%u texels=%lu packed=%lux%lu max=%lu",
              tex->name,
              sourceBuffer->name,
              (unsigned long)texelCount,
              (unsigned long)texWidth,
              (unsigned long)texHeight,
              (unsigned long)max2DSize);
        return nil;
    }

    NSUInteger bytesPerRow = texWidth * bytesPerTexel;
    NSUInteger packedBytes = bytesPerRow * texHeight;
    NSMutableData *packedData = nil;
    const uint8_t *uploadBytes = sourceBytes;

    /* Channel expansion for 3-channel RGB -> 4-channel RGBA Metal formats.
     * GL_RGB32* (12 bytes/texel) maps to Metal RGBA32* (16 bytes/texel).
     * Expand each texel by inserting a default alpha before uploading. */
    NSMutableData *expandedData = nil;
    if (mglTextureNeedsChannelExpansion(tex->internalformat, bufferPixelFormat)) {
        NSUInteger srcCompBytes = 0;
        NSUInteger dstCompBytes = 0;
        uint64_t alphaDefault = 0;
        switch (bufferPixelFormat) {
            case MGLPixelFormatRGBA16Unorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 65535; break;
            case MGLPixelFormatRGBA16Snorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 32767; break;
            case MGLPixelFormatRGBA16Float:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 0x3C00; break;
            case MGLPixelFormatRGBA16Sint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MGLPixelFormatRGBA16Uint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MGLPixelFormatRGBA32Float:
                srcCompBytes = 4; dstCompBytes = 4;
                { float f = 1.0f; memcpy(&alphaDefault, &f, sizeof(f)); }
                break;
            case MGLPixelFormatRGBA32Sint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            case MGLPixelFormatRGBA32Uint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            default:
                break;
        }
        if (srcCompBytes > 0) {
            NSUInteger srcPixelBytes = srcCompBytes * 3;
            NSUInteger dstPixelBytes = dstCompBytes * 4;
            NSUInteger expandedBytesPerRow = texWidth * dstPixelBytes;
            NSUInteger expandedPackedBytes = expandedBytesPerRow * texHeight;
            expandedData = [NSMutableData dataWithLength:expandedPackedBytes];
            if (expandedData && expandedData.mutableBytes) {

                if (mglRenderTextureExpandRGBToRGBA(
                        sourceBytes, expandedData.mutableBytes, texelCount,
                        texWidth, texHeight, srcCompBytes, dstCompBytes,
                        alphaDefault) != 0) {
                    NSLog(@"MGL TEXBUFFER ERROR: channel expansion failed tex=%u buffer=%u",
                          tex->name,
                          sourceBuffer->name);
                    return nil;
                }
                uploadBytes = (const uint8_t *)expandedData.bytes;
                bytesPerRow = expandedBytesPerRow;
                packedBytes = expandedPackedBytes;
            }
        }
    }

    if (texHeight > 1 && !expandedData) {
        packedData = [NSMutableData dataWithLength:packedBytes];
        if (!packedData || !packedData.mutableBytes) {
            NSLog(@"MGL TEXBUFFER ERROR: failed allocating packed data tex=%u buffer=%u bytes=%lu",
                  tex->name,
                  sourceBuffer->name,
                  (unsigned long)packedBytes);
            return nil;
        }

        memcpy(packedData.mutableBytes, sourceBytes, (size_t)tex->texture_buffer_size);
        uploadBytes = (const uint8_t *)packedData.bytes;
    }

    uint64_t sourceHash = mglTraceHashBytes(sourceBytes, (size_t)tex->texture_buffer_size);
    uint64_t uploadHash = mglTraceHashBytes(uploadBytes, packedBytes);
    char sourceHead[64];
    char uploadHead[64];
    sourceHead[0] = '\0';
    uploadHead[0] = '\0';
    mglTraceFormatBytes(sourceBytes, (size_t)MIN((NSUInteger)tex->texture_buffer_size, (NSUInteger)64), sourceHead, sizeof(sourceHead));
    mglTraceFormatBytes(uploadBytes, (size_t)MIN(packedBytes, (NSUInteger)64), uploadHead, sizeof(uploadHead));

    MGLRenderTextureDescriptorState bufferDesc = {
        .texture_type = MGLTextureType2D,
        .pixel_format = bufferPixelFormat,
        .width = texWidth, .height = texHeight, .depth = 1u,
        .mipmap_level_count = 1u, .sample_count = 1u, .array_length = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    id bufferTexture = nil;
    @try {
        bufferTexture = mglTextureCreateTexture(_device, &bufferDesc);
        if (bufferTexture) {
            mglTextureReplaceRegion(
                bufferTexture, mglTextureRegion2D(0, 0, texWidth, texHeight),
                0, 0, uploadBytes, bytesPerRow, 0, NO);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL TEXBUFFER ERROR: failed creating/uploading tex=%u buffer=%u exception=%@",
              tex->name,
              sourceBuffer->name,
              exception);
        return nil;
    }

    if (!bufferTexture) {
        NSLog(@"MGL TEXBUFFER ERROR: Metal texture creation returned nil tex=%u buffer=%u format=%lu texels=%lu",
              tex->name,
              sourceBuffer->name,
              (unsigned long)bufferPixelFormat,
              (unsigned long)texelCount);
        return nil;
    }

    tex->dirty_bits = 0;
    sourceBuffer->data.dirty_bits = 0;

    NSMutableData *readbackData = [NSMutableData dataWithLength:packedBytes];
    uint64_t readbackHash = 0ull;
    char readbackHead[64];
    readbackHead[0] = '\0';
    if (readbackData.mutableBytes) {
        mglTextureGetBytes(
            bufferTexture, readbackData.mutableBytes, bytesPerRow, 0,
            mglTextureRegion2D(0, 0, texWidth, texHeight), 0, 0, NO);
        readbackHash = mglTraceHashBytes(readbackData.bytes, packedBytes);
        mglTraceFormatBytes(readbackData.bytes, (size_t)MIN(packedBytes, (NSUInteger)64), readbackHead, sizeof(readbackHead));
    }

    {
        static uint64_t s_texBufferCreateLogs = 0;
        uint64_t hit = ++s_texBufferCreateLogs;
        if (hit <= 2ull || (hit % 4096ull) == 0ull) {
            NSLog(@"MGL TEXBUFFER CREATE tex=%u buffer=%u internal=0x%x mtlFormat=%lu texels=%lu packed=%lux%lu rowBytes=%lu bytes=%lld offset=%lld as=texture2d sourceHash=0x%016llx uploadHash=0x%016llx readbackHash=0x%016llx sourceHead=%s uploadHead=%s readbackHead=%s",
                  tex->name,
                  sourceBuffer->name,
                  tex->internalformat,
                  (unsigned long)bufferPixelFormat,
                  (unsigned long)texelCount,
                  (unsigned long)texWidth,
                  (unsigned long)texHeight,
                  (unsigned long)bytesPerRow,
                  (long long)tex->texture_buffer_size,
                  (long long)tex->texture_buffer_offset,
                  (unsigned long long)sourceHash,
                  (unsigned long long)uploadHash,
                  (unsigned long long)readbackHash,
                  sourceHead,
                  uploadHead,
                  readbackHead);
        }
    }

    [self recordGPUSuccess];
    return bufferTexture;
}

- (BOOL)checkTextureCompleteness:(Texture *)tex
                          texType:(uint32_t)tex_type
                         numFaces:(uint)num_faces
             effectiveMipmapLevels:(GLuint *)outEffectiveMipmapLevels
                 storageMipmapped:(BOOL *)outStorageMipmapped
{
    (void)tex_type;  /* unused: completeness does not depend on Metal texture type */
    GLuint effective_mipmap_levels = tex->mipmap_levels;
    BOOL storageMipmapped = NO;

    uint completeness_check_faces = (tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) ? 1 : num_faces;

    /* Texture storage is independent from GL_TEXTURE_MAX_LEVEL.  Minecraft
     * uses BASE/MAX_LEVEL to express temporary GpuTextureView mip windows; if
     * those sampler parameters shrink the Metal texture allocation, later
     * full-atlas sampling loses the higher mip levels and distant terrain
     * reads empty/incorrect data.  Apply BASE/MAX only to completeness checks
     * and sampled Metal views, not to the underlying storage level count. */

    /* For CUBE_MAP_ARRAY, glTexImage3D stores all layer data in faces[0] with
     * depth = 6 * num_cubes.  Faces 1-5 are never populated by createTextureLevel,
     * so only check face 0 for completeness.  The upload code also reads from
     * face 0 and distributes slices to Metal array layers. */

    storageMipmapped = (tex->mipmap_levels > 1u) &&
        (tex->num_levels > 1u || tex->is_render_target);

    if (tex->num_levels > 1)
    {
        // mipmapped texture
        if (effective_mipmap_levels == 0) {
            effective_mipmap_levels = tex->num_levels;
        }

        if (!tex->is_render_target && tex->num_levels < effective_mipmap_levels)
        {
            static uint64_t s_mipmap_count_mismatch_logs = 0;
            if (++s_mipmap_count_mismatch_logs <= 8 || (s_mipmap_count_mismatch_logs % 2048) == 0) {
                NSLog(@"MGL TEXTURE MIP COMPAT: tex=%u target=0x%x size=%ux%u num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u immutable=%u isRT=%u; capping Metal mip count to uploaded levels hit=%llu",
                      tex->name,
                      tex->target,
                      tex->width,
                      tex->height,
                      tex->num_levels,
                      tex->mipmap_levels,
                      effective_mipmap_levels,
                      tex->params.base_level,
                      tex->params.max_level,
                      tex->immutable_storage,
                      tex->is_render_target,
                      (unsigned long long)s_mipmap_count_mismatch_logs);
            }
            effective_mipmap_levels = tex->num_levels;
        }

        /* GL texture completeness only requires levels in
         * [base_level, min(max_level, mipmap_levels-1)] to be complete.
         * Levels below base_level may be uninitialised and must NOT cause
         * the texture to be rejected.  Minecraft 1.21.11 sets base_level>0
         * on mipmap texture views (GlCommandEncoder.java). */
        GLuint check_start = tex->params.base_level;
        GLuint check_end = (tex->params.max_level == 1000u)
            ? (tex->mipmap_levels > 0u ? tex->mipmap_levels - 1u : 0u)
            : tex->params.max_level;
        if (check_end >= tex->mipmap_levels)
            check_end = (tex->mipmap_levels > 0u) ? tex->mipmap_levels - 1u : 0u;
        if (check_end < check_start)
            check_end = check_start;

        for(int face=0; face<completeness_check_faces; face++)
        {
            for (GLuint i=check_start; i<=check_end; i++)
            {
                // incomplete texture
                if (tex->faces[face].levels[i].complete == false) {
                    static uint64_t s_incomplete_mip_logs = 0;
                    if (++s_incomplete_mip_logs <= 32 || (s_incomplete_mip_logs % 512) == 0) {
                        NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d level=%u incomplete num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u check=[%u,%u] hit=%llu",
                              tex->name,
                              tex->target,
                              face,
                              i,
                              tex->num_levels,
                              tex->mipmap_levels,
                              effective_mipmap_levels,
                              tex->params.base_level,
                              tex->params.max_level,
                              check_start,
                              check_end,
                              (unsigned long long)s_incomplete_mip_logs);
                    }
                    return NO;
                }
            }
        }

        tex->mipmapped = true;
    }
    else if (tex->num_levels == 1)
    {
        if (!storageMipmapped) {
            effective_mipmap_levels = 1;
        }
        // single level texture
        // incomplete texture
        for(int face=0; face<completeness_check_faces; face++)
        {
            if (tex->faces[face].levels[0].complete == false)
            {
                static uint64_t s_incomplete_base_logs = 0;
                if (++s_incomplete_base_logs <= 32 || (s_incomplete_base_logs % 512) == 0) {
                    NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d base incomplete size=%ux%u hit=%llu",
                          tex->name,
                          tex->target,
                          face,
                          tex->width,
                          tex->height,
                          (unsigned long long)s_incomplete_base_logs);
                }
                return NO;
            }
        }
    }
    else
    {
        NSLog(@"MGL TEXTURE ERROR: texture %u has no complete levels for Metal creation target=0x%x",
              tex->name,
              tex->target);
        return NO;
    }

    tex->complete = true;

    if (outEffectiveMipmapLevels) *outEffectiveMipmapLevels = effective_mipmap_levels;
    if (outStorageMipmapped) *outStorageMipmapped = storageMipmapped;
    return YES;
}

- (void)logMTLTextureMipDiagnostics:(Texture *)tex
                              metal:(id)texture
               effectiveMipLevels:(GLuint)effective_mipmap_levels
{
    static uint64_t s_mipDiagLogs = 0;
    uint64_t diagHit = ++s_mipDiagLogs;
    if (kMGLDiagnosticStateLogs &&
        (diagHit <= 128ull || (diagHit % 512ull) == 0ull)) {
        NSUInteger mtlMipCount = mglTextureInfo(texture).mipmap_level_count;
        uint32_t mtlFmt = mglTextureInfo(texture).pixel_format;
        uint32_t mtlStorage = mglTextureInfo(texture).storage_mode;
        NSUInteger uploadedLevels = 0;
        NSUInteger skippedLevels = 0;
        NSUInteger skippedSourceNone = 0;
        NSUInteger skippedNoData = 0;
        NSMutableString *levelSummary = [NSMutableString stringWithCapacity:256];
        NSUInteger levelsToSummarize = MIN((NSUInteger)tex->num_levels, (NSUInteger)16);
        for (NSUInteger lvl = 0; lvl < levelsToSummarize; lvl++) {
            TextureLevel *tl = (tex->faces[0].levels && lvl < tex->num_levels)
                ? &tex->faces[0].levels[lvl] : NULL;
            if (!tl) { [levelSummary appendString:@"-"]; continue; }
            bool uploadable = mglTextureLevelHasUploadableCPUData(tl);
            if (uploadable) uploadedLevels++; else skippedLevels++;
            if (!uploadable) {
                if (tl->last_init_source == kTexImageNull || tl->last_init_source == kTexInitNone)
                    skippedSourceNone++;
                if (!tl->has_initialized_data && !tl->ever_written)
                    skippedNoData++;
            }
            [levelSummary appendFormat:@"[%u:s%u:w%u:e%u:i%u]",
                (unsigned)lvl, (unsigned)tl->last_init_source,
                (unsigned)tl->width, (unsigned)tl->ever_written,
                (unsigned)tl->has_initialized_data];
        }
        mglTraceLogNSString(@"MGL TEX_MIP_DIAG tex=%u target=0x%x dims=%ux%u internal=0x%x "
                      @"numLevels=%u mipmapLevels=%u effectiveMipLevels=%u mtlMipCount=%lu "
                      @"mtlFmt=%lu mtlStorage=%ld mipmapped=%d baseLevel=%u maxLevel=%u "
                      @"uploadedLevels=%lu skippedLevels=%lu skippedSourceNone=%lu skippedNoData=%lu "
                      @"levels=%@ hit=%llu",
                      (unsigned)tex->name, (unsigned)tex->target,
                      (unsigned)tex->width, (unsigned)tex->height,
                      (unsigned)tex->internalformat,
                      (unsigned)tex->num_levels, (unsigned)tex->mipmap_levels,
                      (unsigned)effective_mipmap_levels, (unsigned long)mtlMipCount,
                      (unsigned long)mtlFmt, (long)mtlStorage, (int)(tex->mipmapped ? 1 : 0),
                      (unsigned)tex->params.base_level, (unsigned)tex->params.max_level,
                      (unsigned long)uploadedLevels, (unsigned long)skippedLevels,
                      (unsigned long)skippedSourceNone, (unsigned long)skippedNoData,
                      levelSummary, (unsigned long long)diagHit);
    }
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
        uint32_t fallbackFormat = mtlPixelFormatForGLTex(tex);
        if (fallbackFormat == MGLPixelFormatInvalid) {
            // Conservative defaults by GL intent when translation is unavailable.
            if (tex->internalformat == GL_DEPTH24_STENCIL8 ||
                tex->internalformat == GL_DEPTH32F_STENCIL8) {
                fallbackFormat = MGLPixelFormatDepth32Float_Stencil8;
            } else if (tex->internalformat == GL_DEPTH_COMPONENT ||
                       tex->internalformat == GL_DEPTH_COMPONENT16 ||
                       tex->internalformat == GL_DEPTH_COMPONENT24 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32F) {
                fallbackFormat = MGLPixelFormatDepth32Float;
            } else {
                fallbackFormat = MGLPixelFormatRGBA8Unorm;
            }
        }

        BOOL isDepthOrStencilFormat =
            (fallbackFormat == MGLPixelFormatDepth16Unorm ||
             fallbackFormat == MGLPixelFormatDepth32Float ||
             fallbackFormat == MGLPixelFormatDepth24Unorm_Stencil8 ||
             fallbackFormat == MGLPixelFormatDepth32Float_Stencil8 ||
             fallbackFormat == MGLPixelFormatStencil8);

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
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat
{
    switch(internalformat) {
        case GL_RED:
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
            return 1;

        case GL_RG:
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_R16:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
            return 2;

        case GL_RGB:
        case GL_RGB8:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_SRGB8:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
            return 3;

        case GL_RGBA:
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_SRGB8_ALPHA8:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_R32F:
            return 4;

        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RG32F:
            return 8;

        case GL_RGB16:
        case GL_RGB16F:
            return 6;

        case GL_RGBA16I:
        case GL_RGBA16UI:
            return 8;

        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return 12;

        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return 16;

        default:
            // Default to 4 bytes for unknown formats
            NSLog(@"MGL WARNING: Unknown internal format 0x%x, defaulting to 4 bytes per pixel", internalformat);
            return 4;
    }
}

- (id) createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target
{
    mglMetalCountCreate(MGLMetalKindSampler);
    void *sampler = NULL;
    char error[256] = {0};
    if (mglRenderCreateSamplerForGL(
            tex_param, target, &sampler, error, sizeof(error)) == 0 &&
        sampler) {
        return (__bridge_transfer id)sampler;
    }
    NSLog(@"MGL SAMPLER ERROR: Metal-cpp sampler creation failed: %s",
          error[0] ? error : "unknown");
    return nil;
}

- (id)fallbackSampledTexture
{
    id cached = (__bridge id)
        mglRendererBackendGetFallbackResource(
            _backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLED_TEXTURE);
    if (cached || !kMGLEnableSampledTextureFallback) {
        return cached;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureType2D,
        .pixel_format = MGLPixelFormatRGBA8Unorm,
        .width = 1u, .height = 1u, .depth = 1u,
        .mipmap_level_count = 1u, .sample_count = 1u, .array_length = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    id texture = mglTextureCreateTexture(_device, &desc);
    if (texture) {
        uint32_t pixel = 0xff000000u;
        mglTextureReplaceRegion(
            texture,
            mglTextureRegion2D(0, 0, 1, 1), 0, 0, &pixel,
            sizeof(pixel), 0, NO);
        if (mglRendererBackendSetFallbackResource(
                _backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLED_TEXTURE,
                (__bridge void *)texture) != 0) {
            return nil;
        }
        NSLog(@"MGL INFO: Created 1x1 fallback sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback sampled texture");
    }

    return texture;
}

- (id)fallbackCubeSampledTexture
{
    id cached = (__bridge id)
        mglRendererBackendGetFallbackResource(
            _backend, MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE);
    if (cached || !kMGLEnableSampledTextureFallback) {
        return cached;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureTypeCube,
        .pixel_format = MGLPixelFormatRGBA8Unorm,
        .width = 1u, .height = 1u, .depth = 1u,
        .array_length = 1u, .mipmap_level_count = 1u, .sample_count = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    id texture = mglTextureCreateTexture(_device, &desc);
    if (texture) {
        uint32_t pixel = 0xff000000u;
        for (NSUInteger face = 0; face < 6; face++) {
            mglTextureReplaceRegion(
                texture,
                mglTextureRegion2D(0, 0, 1, 1), 0, face, &pixel,
                sizeof(pixel), sizeof(pixel), YES);
        }
        if (mglRendererBackendSetFallbackResource(
                _backend, MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE,
                (__bridge void *)texture) != 0) {
            return nil;
        }
        NSLog(@"MGL INFO: Created 1x1 fallback cube sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback cube sampled texture");
    }

    return texture;
}

- (id)fallbackTextureBufferSampledTexture
{
    id cachedTexture = (__bridge id)
        mglRendererBackendGetFallbackResource(
            _backend, MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER);
    if (cachedTexture || !kMGLEnableSampledTextureFallback) {
        return cachedTexture;
    }

    static const NSUInteger kFallbackTexelCount = 64;
    static const NSUInteger kFallbackBytesPerTexel = 4;

    id storage = (__bridge id)
        mglRendererBackendGetFallbackResource(
            _backend, MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE);
    if (!storage) {
        storage = mglTextureCreateBuffer(
            _device, kFallbackTexelCount * kFallbackBytesPerTexel,
            MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
        if (storage && mglTextureBufferContents(storage)) {
            memset(mglTextureBufferContents(storage), 0,
                   kFallbackTexelCount * kFallbackBytesPerTexel);
        }
        if (storage && mglRendererBackendSetFallbackResource(
                _backend,
                MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE,
                (__bridge void *)storage) != 0) {
            storage = nil;
        }
    }

    if (!storage) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer backing storage");
        return nil;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureTypeTextureBuffer,
        .pixel_format = MGLPixelFormatRGBA8Sint,
        .width = kFallbackTexelCount, .height = 1u, .depth = 1u,
        .array_length = 1u, .mipmap_level_count = 1u, .sample_count = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    @try {
        cachedTexture = mglTextureCreateBufferTexture(
            storage, &desc, 0,
            kFallbackTexelCount * kFallbackBytesPerTexel);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer texture: %@", exception);
        cachedTexture = nil;
    }

    if (cachedTexture && mglRendererBackendSetFallbackResource(
            _backend, MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER,
            (__bridge void *)cachedTexture) != 0) {
        cachedTexture = nil;
    }
    if (cachedTexture) {
        NSLog(@"MGL INFO: Created fallback signed integer texture buffer for missing/invalid texel-buffer resources");
    }

    return cachedTexture;
}

- (id)fallbackSampledTextureForExpectedType:(uint32_t)expectedType
                                               dataKind:(MGLTextureDataKind)dataKind
{
    if (!kMGLEnableSampledTextureFallback) {
        return nil;
    }

    uint32_t textureType = expectedType ? expectedType : MGLTextureType2D;
    if (textureType == MGLTextureTypeTextureBuffer) {
        return [self fallbackTextureBufferSampledTexture];
    }

    uint32_t pixelFormat = MGLPixelFormatRGBA8Unorm;
    if (dataKind == MGLTextureDataKindUint) {
        pixelFormat = MGLPixelFormatRGBA8Uint;
    } else if (dataKind == MGLTextureDataKindSint) {
        pixelFormat = MGLPixelFormatRGBA8Sint;
    } else if (dataKind == MGLTextureDataKindDepth) {
        pixelFormat = MGLPixelFormatDepth32Float;
    }

    NSUInteger keyValue = (((NSUInteger)textureType) << 8u) | ((NSUInteger)dataKind);
    void *cachedTexture = NULL;
    int cacheResult = mglRendererBackendGetFallbackSampledTexture(
        _backend, keyValue, &cachedTexture);
    if (cacheResult == 1) {
        return (__bridge id)cachedTexture;
    }
    if (cacheResult < 0) {
        return nil;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = textureType,
        .pixel_format = pixelFormat,
        .width = 1u, .height = 1u, .depth = 1u,
        .array_length = 1u, .mipmap_level_count = 1u,
        .sample_count = 1u, .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };
    if (textureType == MGLTextureType2DMultisample ||
        textureType == MGLTextureType2DMultisampleArray) {
        desc.sample_count = 2u;
    }

    id texture = mglTextureCreateTexture(_device, &desc);
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create %@ fallback sampled texture type=%lu format=%lu",
              [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
              (unsigned long)textureType,
              (unsigned long)pixelFormat);
        return nil;
    }

    uint32_t pixel = dataKind == MGLTextureDataKindDepth ? 0u : 0xff000000u;
    MGLRegionValue region = textureType == MGLTextureType1D ||
                       textureType == MGLTextureType1DArray
        ? mglTextureRegion1D(0, 1)
        : mglTextureRegion2D(0, 0, 1, 1);
    if (textureType == MGLTextureTypeCube || textureType == MGLTextureTypeCubeArray) {
        NSUInteger sliceCount = (textureType == MGLTextureTypeCube) ? 6u : 6u;
        for (NSUInteger slice = 0; slice < sliceCount; slice++) {
            mglTextureReplaceRegion(
                texture, mglTextureRegion2D(0, 0, 1, 1), 0, slice,
                &pixel, sizeof(pixel), sizeof(pixel), YES);
        }
    } else if (textureType == MGLTextureType1DArray ||
               textureType == MGLTextureType2DArray) {
        mglTextureReplaceRegion(
            texture, region, 0, 0, &pixel, sizeof(pixel),
            sizeof(pixel), YES);
    } else {
        mglTextureReplaceRegion(
            texture, region, 0, 0, &pixel, sizeof(pixel), 0, NO);
    }

    if (mglRendererBackendPutFallbackSampledTexture(
            _backend, keyValue, (__bridge void *)texture) != 0) {
        return nil;
    }
    NSLog(@"MGL INFO: Created %@ fallback sampled texture type=%lu format=%lu",
          [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
          (unsigned long)textureType,
          (unsigned long)pixelFormat);

    return texture;
}


- (id)fallbackSampledTextureForExpectedType:(uint32_t)expectedType
{
    if (expectedType == MGLTextureTypeCube) {
        return [self fallbackCubeSampledTexture];
    }
    if (expectedType == MGLTextureTypeTextureBuffer) {
        return [self fallbackTextureBufferSampledTexture];
    }

    return [self fallbackSampledTexture];
}

- (int)textureIndexForExpectedMetalType:(uint32_t)expectedType
{
    return (int)mglRenderTextureIndexForMetalType((uint32_t)expectedType);
}

- (GLuint)textureUnitForSampledResource:(MGLShaderResource *)sampledResource
                                program:(Program *)program
                           metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
{
    if (!program) {
        GLuint candidate = sampledResource &&
                           sampledResource->sampler_unit >= 0 &&
                           sampledResource->sampler_unit < TEXTURE_UNITS
            ? (GLuint)sampledResource->sampler_unit
            : metalBinding;
        return candidate;
    }

    const char *sampledName = NULL;
    if (!sampledResource && metalBinding < TEXTURE_UNITS) {
        sampledResource = mglFindSamplerResourceForMetalBinding(program, stage, metalBinding);
    }
    if (sampledResource) {
        sampledName = sampledResource->name;
    }

    /*
     * Minecraft usually assigns sampler texture units from the RenderPipeline
     * sampler list, not from numeric suffixes like Sampler2. For example, chunk
     * rendering declares Sampler0 and Sampler2, so Sampler2 can be uploaded
     * through glUniform1i(..., 1). Keep sampler units on the exact reflected
     * resource instead of only the Metal binding: vertex and fragment resources
     * commonly share binding numbers, and binding-level state can make entity,
     * hand, and text textures bleed into each other.
     */
    if (sampledResource &&
        sampledResource->sampler_unit_explicit &&
        sampledResource->sampler_unit >= 0 &&
        sampledResource->sampler_unit < TEXTURE_UNITS) {
        GLuint element = metalBinding >= sampledResource->binding
            ? metalBinding - sampledResource->binding : 0u;
        return (GLuint)sampledResource->sampler_unit + element;
    }

    if (metalBinding >= TEXTURE_UNITS) {
        return metalBinding;
    }

    bool stageExplicit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? (program->sampler_units_explicit_by_stage[stage][metalBinding] == GL_TRUE)
        : false;
    bool globalExplicit = (program->sampler_units_explicit[metalBinding] == GL_TRUE);

    GLint unit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];

    if (stageExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    unit = program->sampler_units[metalBinding];
    if (globalExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    GLint defaultUnit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];
    if (defaultUnit < 0 || defaultUnit >= TEXTURE_UNITS) {
        defaultUnit = program->sampler_units[metalBinding];
    }

    if (sampledResource &&
        !sampledResource->sampler_unit_explicit &&
        sampledResource->sampler_unit >= 0 &&
        sampledResource->sampler_unit < TEXTURE_UNITS) {
        GLuint element = metalBinding >= sampledResource->binding
            ? metalBinding - sampledResource->binding : 0u;
        return (GLuint)sampledResource->sampler_unit + element;
    }

    if (defaultUnit >= 0 && defaultUnit < TEXTURE_UNITS) {
        return (GLuint)defaultUnit;
    }

    /*
     * OpenGL's valid default is unit 0, and explicit glUniform1i uploads above
     * are authoritative. No name-based fallback is applied.
     */
    return 0u;
}

- (GLuint)textureUnitForSampledResource:(MGLShaderResource *)sampledResource metalBinding:(GLuint)metalBinding stage:(int)stage
{
    Program *program = mglResolveProgramForStageFromState(ctx, stage);
    return [self textureUnitForSampledResource:sampledResource
                                      program:program
                                 metalBinding:metalBinding
                                        stage:stage];
}

- (GLuint)textureUnitForSampledBinding:(GLuint)metalBinding stage:(int)stage
{
    return [self textureUnitForSampledResource:NULL metalBinding:metalBinding stage:stage];
}

- (Texture *)textureForSampledResource:(MGLShaderResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(uint32_t)expectedType
                          textureUnit:(GLuint)textureUnit
{
    if (!ctx || metalBinding >= TEXTURE_UNITS) {
        return NULL;
    }

    if (textureUnit >= TEXTURE_UNITS) {
        return NULL;
    }

    if (expectedType == 0) {
        return MGL_STATE(ctx)->active_textures[textureUnit];
    }

    int textureIndex = [self textureIndexForExpectedMetalType:expectedType];
    if (textureIndex >= 0 && textureIndex < _MAX_TEXTURE_TYPES) {
        Texture *typedTexture = MGL_STATE(ctx)->texture_units[textureUnit].textures[textureIndex];
        /* The AIR backend lowers sampler1D to texture2d, so expectedType is
         * MGLTextureType2D even for GL_TEXTURE_1D bindings. If the _TEXTURE_2D
         * slot only contains an auto-created default texture (name ==
         * TEX_OBJ_RES_NAME) while the unit's active texture is a real
         * GL_TEXTURE_1D, prefer the 1D texture. Otherwise the default 2D
         * texture leaks across test cases and masks the real 1D binding. */
        if (typedTexture && typedTexture->name == TEX_OBJ_RES_NAME) {
            Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
            if (activeTexture && activeTexture->name != TEX_OBJ_RES_NAME) {
                typedTexture = NULL;
            }
        }
        if (typedTexture) {
            return typedTexture;
        }

        if (expectedType == MGLTextureType2D) {
            Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
            if (activeTexture &&
                activeTexture->target == GL_TEXTURE_1D) {
                return activeTexture;
            }
        }

        // Texel-buffer resources must not silently fall back to GL_TEXTURE_2D.
        // Minecraft's CloudFaces is declared with buffer image_dim, and the lowerer
        // lowers it to a 1-row texture2d<int> in MSL. If no GL_TEXTURE_BUFFER
        // is bound, using the active 2D atlas here feeds float/RGBA data into a
        // signed integer vertex resource and corrupts the whole frame.
        if (expectedType == MGLTextureTypeTextureBuffer) {
            static uint64_t s_missingTextureBufferBindingLogs = 0;
            uint64_t hit = ++s_missingTextureBufferBindingLogs;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
                NSLog(@"MGL TEXBUFFER BIND MISSING binding=%u unit=%u activeTex=%u activeTarget=0x%x hit=%llu",
                      (unsigned)metalBinding,
                      (unsigned)textureUnit,
                      activeTexture ? (unsigned)activeTexture->name : 0u,
                      activeTexture ? (unsigned)activeTexture->target : 0u,
                      (unsigned long long)hit);
            }
            return NULL;
        }

        /*
         * OpenGL texture units keep one binding per texture target. A sampler2D
         * samples the GL_TEXTURE_2D slot for its unit, even if a cubemap or texel
         * buffer was bound more recently on that same unit. Falling back to the
         * unit's "active" texture here lets sky cubemaps and buffer textures bleed
         * into item/entity shaders when Minecraft switches pipelines.
         */
        static uint64_t s_missingTypedTextureBindingLogs = 0;
        uint64_t hit = ++s_missingTypedTextureBindingLogs;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
            NSLog(@"MGL TEX TYPED BIND MISSING binding=%u stage=%s unit=%u expectedType=%lu expectedIndex=%d activeTex=%u activeTarget=0x%x hit=%llu",
                  (unsigned)metalBinding,
                  mglShaderStageName(stage),
                  (unsigned)textureUnit,
                  (unsigned long)expectedType,
                  textureIndex,
                  activeTexture ? (unsigned)activeTexture->name : 0u,
                  activeTexture ? (unsigned)activeTexture->target : 0u,
                  (unsigned long long)hit);
        }
        return NULL;
    }

    return MGL_STATE(ctx)->active_textures[textureUnit];
}

- (Texture *)textureForSampledResource:(MGLShaderResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(uint32_t)expectedType
{
    if (!ctx || metalBinding >= TEXTURE_UNITS) {
        return NULL;
    }
    GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                metalBinding:metalBinding
                                                       stage:stage];
    return [self textureForSampledResource:sampledResource
                              metalBinding:metalBinding
                                      stage:stage
                               expectedType:expectedType
                              textureUnit:textureUnit];
}

- (Texture *)textureForSampledBinding:(GLuint)metalBinding stage:(int)stage expectedType:(uint32_t)expectedType
{
    return [self textureForSampledResource:NULL
                              metalBinding:metalBinding
                                      stage:stage
                               expectedType:expectedType];
}

- (id)fallbackSamplerState
{
    id cached = (__bridge id)
        mglRendererBackendGetFallbackResource(
            _backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLER);
    if (cached) {
        return cached;
    }

    id sampler = mglTextureCreateSampler(_device);
    if (!sampler) {
        NSLog(@"MGL ERROR: Failed to create fallback sampler state");
        return nil;
    }
    if (mglRendererBackendSetFallbackResource(
            _backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLER,
            (__bridge void *)sampler) != 0) {
        return nil;
    }
    return sampler;
}

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
        fmt == MGLPixelFormatRGBA8Unorm ||
        fmt == MGLPixelFormatRGBA8Unorm_sRGB ||
        fmt == MGLPixelFormatBGRA8Unorm ||
        fmt == MGLPixelFormatBGRA8Unorm_sRGB;
    if (!fourByteColor) {
        mglTraceLogNSString(@"MGL TRACE sampled.readback skip program=%u binding=%u glTex=%u reason=%@ fmt=%lu type=%lu size=%lux%lu hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
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
        mglTraceLogNSString(@"MGL TRACE sampled.readback setup-fail program=%u binding=%u glTex=%u reason=%@ readback=%p cb=%p blit=%p hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
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
    mglTraceLogNSString(@"MGL TRACE sampled.readback stage=%@ program=%u binding=%u glTex=%u reason=%@ hit=%llu "
          "mtl=%p fmt=%lu type=%lu size=%lux%lu sample=%lux%lu status=%s error=%@ "
          "nonZero=%lu/%lu sum=%llu first=0x%08x min=0x%08x max=0x%08x xor=0x%08x "
          "level(init ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx)",
          stage,
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
