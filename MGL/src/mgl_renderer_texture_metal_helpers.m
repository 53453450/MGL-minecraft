/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#import "mgl_renderer_texture_metal_helpers.h"


void mglTextureCommandCompletionCallback(
    void *context,
    const MGLRenderCommandBufferState *state)
{
    MGLTextureCommandCompletionBlock block =
        (__bridge MGLTextureCommandCompletionBlock)context;
    if (block) block(state);
}

void mglTextureCommandCompletionDestroy(void *context)
{
    if (!context) return;
    (void)CFBridgingRelease(context);
}

int mglTextureAddCommandBufferCompletion(
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

MGLOriginValue mglTextureOrigin(uint64_t x, uint64_t y, uint64_t z)
{ return (MGLOriginValue){(int64_t)x, (int64_t)y, (int64_t)z}; }
MGLSizeValue mglTextureSize(uint64_t width, uint64_t height, uint64_t depth)
{ return (MGLSizeValue){width, height, depth}; }
MGLRegionValue mglTextureRegion1D(uint64_t x, uint64_t width)
{ return (MGLRegionValue){mglTextureOrigin(x, 0, 0), mglTextureSize(width, 1, 1)}; }
MGLRegionValue mglTextureRegion2D(uint64_t x, uint64_t y,
                                         uint64_t width, uint64_t height)
{ return (MGLRegionValue){mglTextureOrigin(x, y, 0), mglTextureSize(width, height, 1)}; }
MGLRegionValue mglTextureRegion3D(uint64_t x, uint64_t y, uint64_t z,
                                         uint64_t width, uint64_t height, uint64_t depth)
{ return (MGLRegionValue){mglTextureOrigin(x, y, z), mglTextureSize(width, height, depth)}; }

id mglTextureCreateBuffer(id device,
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

id mglTextureCreateBufferWithBytes(
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

id mglTextureCreateTexture(
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

id mglTextureCreateBufferTexture(
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

void mglTextureReplaceRegion(id texture,
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

void mglTextureGetBytes(id texture,
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

id mglTextureCreateSampler(id device)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return (__bridge_transfer id)sampler;
    }
    return nil;
}

id mglTextureCreateCommandBuffer(
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

id mglTextureCreateBlitEncoder(
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
id mglTextureCreateCurrentBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

void mglTextureEndBlitEncoder(id encoder)
{
    if (!encoder) return;
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

void mglTextureCommitCommandBuffer(id commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderCommitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer commit failed");
    }
}

void mglTextureWaitCommandBuffer(id commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderWaitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer wait failed");
    }
}

MGLRenderTextureInfo mglTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    return info;
}

/* AGX replaceRegion/copyFromBuffer require 256-byte row alignment for many
 * depth/stencil pixel formats even when the logical row is smaller. */
static const NSUInteger kMGLDepthStencilUploadRowAlignment = 256u;

NSUInteger mglDepthStencilAlignedBytesPerRow(NSUInteger logicalBytesPerRow)
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
void *mglCreateDepthStencilMetalUpload(
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

uint32_t mglDepthStencilPlaneViewType(uint32_t parentType)
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

uint64_t mglTextureBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo((__bridge void *)buffer, &info) == 0
        ? info.length : 0u;
}

void *mglTextureBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents((__bridge void *)buffer,
                                                   &contents, &length) == 0
        ? contents : NULL;
}

void mglTextureCopyTextureToBuffer(
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

void mglMetalCopyRows(const uint8_t *src,
                      NSUInteger srcBytesPerRow,
                      uint8_t *dst,
                      NSUInteger dstBytesPerRow,
                      NSUInteger rowBytes,
                      NSUInteger height,
                      BOOL flipY)
{
    mglRenderCopyRows(
        src, (uint64_t)srcBytesPerRow,
        dst, (uint64_t)dstBytesPerRow,
        (uint64_t)rowBytes, (uint64_t)height,
        flipY ? 1 : 0);
}

