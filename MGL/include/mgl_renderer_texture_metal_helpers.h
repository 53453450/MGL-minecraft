/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Shared Metal texture helpers used by Texture category and create bridge.
 */

#ifndef MGL_RENDERER_TEXTURE_METAL_HELPERS_H
#define MGL_RENDERER_TEXTURE_METAL_HELPERS_H

#import "MGLRenderer_Private.h"
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

int mglTextureAddCommandBufferCompletion(
    void *commandBuffer,
    MGLTextureCommandCompletionBlock block);

MGLOriginValue mglTextureOrigin(uint64_t x, uint64_t y, uint64_t z);
MGLSizeValue mglTextureSize(uint64_t width, uint64_t height, uint64_t depth);
MGLRegionValue mglTextureRegion1D(uint64_t x, uint64_t width);
MGLRegionValue mglTextureRegion2D(uint64_t x, uint64_t y,
                                  uint64_t width, uint64_t height);
MGLRegionValue mglTextureRegion3D(uint64_t x, uint64_t y, uint64_t z,
                                  uint64_t width, uint64_t height, uint64_t depth);

id mglTextureCreateBuffer(id device, NSUInteger length, uint64_t options);
id mglTextureCreateBufferWithBytes(id device, const void *bytes,
                                   NSUInteger length, uint64_t options);
id mglTextureCreateTexture(id device,
                           const MGLRenderTextureDescriptorState *descriptor);
id mglTextureCreateBufferTexture(id buffer,
                                 const MGLRenderTextureDescriptorState *descriptor,
                                 NSUInteger offset, NSUInteger bytesPerRow);
void mglTextureReplaceRegion(id texture, MGLRegionValue region,
                             NSUInteger level, NSUInteger slice,
                             const void *bytes, NSUInteger bytesPerRow,
                             NSUInteger bytesPerImage, BOOL useSlice);
void mglTextureGetBytes(id texture, void *bytes,
                        NSUInteger bytesPerRow, NSUInteger bytesPerImage,
                        MGLRegionValue region, NSUInteger level,
                        NSUInteger slice, BOOL useSlice);
id mglTextureCreateSampler(id device);
id mglTextureCreateCommandBuffer(id queue);
id mglTextureCreateBlitEncoder(id commandBuffer);
id mglTextureCreateCurrentBlitEncoder(void *commandBufferOwner);
void mglTextureEndBlitEncoder(id encoder);
void mglTextureCommitCommandBuffer(id commandBuffer);
void mglTextureWaitCommandBuffer(id commandBuffer);
MGLRenderTextureInfo mglTextureInfo(id texture);

NSUInteger mglDepthStencilAlignedBytesPerRow(NSUInteger logicalBytesPerRow);
void *mglCreateDepthStencilMetalUpload(
    Texture *tex, uint32_t pixelFormat, const uint8_t *src,
    NSUInteger width, NSUInteger height, NSUInteger srcBytesPerRow,
    NSUInteger *outBytesPerRow, NSUInteger *outBytesPerImage);
uint32_t mglDepthStencilPlaneViewType(uint32_t parentType);
uint64_t mglTextureBufferLength(id buffer);
void *mglTextureBufferContents(id buffer);
void mglTextureCopyTextureToBuffer(
    id encoder, id source, NSUInteger sourceSlice, NSUInteger sourceLevel,
    MGLOriginValue sourceOrigin, MGLSizeValue sourceSize,
    id destination, NSUInteger destinationOffset,
    NSUInteger bytesPerRow, NSUInteger bytesPerImage);

#endif /* MGL_RENDERER_TEXTURE_METAL_HELPERS_H */
