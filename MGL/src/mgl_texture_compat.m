/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */



#import "mgl_texture_compat.h"
#include "mgl_render.h"

MGLTextureDataKind mglTextureDataKindForPixelFormat(uint32_t pixelFormat)
{
    return (MGLTextureDataKind)mglRenderTextureDataKindForPixelFormat(
        (uint32_t)pixelFormat);
}

const char *mglTextureDataKindName(MGLTextureDataKind kind)
{
    return mglRenderTextureDataKindName((uint32_t)kind);
}

size_t mglMetalTextureLevelDimension(size_t base, size_t level)
{

    return (size_t)mglRenderMetalTextureLevelDimension(
        (uint64_t)base, (uint64_t)level);
}

void *mglSampledTextureViewForBaseLevel(Texture *ptr, void *texture)
{
    (void)mglRenderSampledTextureViewForBaseLevel(ptr, texture, &texture);
    return texture;
}

size_t mglStoredColorComponentsForTexture(Texture *tex)
{
    if (!tex) {
        return 4;
    }
    return (size_t)mglRenderStoredColorComponents(
        (uint32_t)tex->internalformat);
}

uint32_t mglMTLSwizzleForGLSwizzle(Texture *tex, GLenum swizzle)
{
    size_t components = mglStoredColorComponentsForTexture(tex);
    return mglRenderMTLSwizzleForGLSwizzle(
        (uint32_t)swizzle, (uint32_t)components);
}

bool mglTextureUploadNeedsSingleChannelSwizzle(Texture *tex)
{
    if (!tex) {
        return false;
    }
    return mglRenderTextureUploadNeedsSingleChannelSwizzle(
        (uint32_t)tex->internalformat, tex->params.swizzled ? 1 : 0) != 0;
}

uint8_t mglResolveR8SwizzledComponent(Texture *tex, GLenum swizzle, uint8_t red)
{
    (void)tex;

    return mglRenderResolveR8SwizzledComponent((uint32_t)swizzle, red);
}

uint8_t *mglCreateSingleChannelSwizzledUpload(Texture *tex,
                                              const uint8_t *srcData,
                                              size_t width,
                                              size_t height,
                                              size_t srcBytesPerRow,
                                              size_t *outBytesPerRow,
                                              size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 || !outBytesPerRow || !outBytesPerImage) {
        return NULL;
    }


    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCreateSingleChannelSwizzledUpload(
        (uint32_t)tex->internalformat,
        (uint32_t)tex->params.swizzle_r,
        (uint32_t)tex->params.swizzle_g,
        (uint32_t)tex->params.swizzle_b,
        (uint32_t)tex->params.swizzle_a,
        srcData, (size_t)width, (size_t)height, (size_t)srcBytesPerRow,
        &outBPR, &outBPI);
    if (result) {
        *outBytesPerRow = outBPR;
        *outBytesPerImage = outBPI;
    }
    return result;
}

bool mglTextureInternalFormatNeedsRGBA8Expansion(GLenum internalformat,
                                                 uint32_t pixelFormat)
{
    return mglRenderTextureInternalFormatNeedsRGBA8Expansion(
        (uint32_t)internalformat, pixelFormat) != 0;
}

bool mglTextureNeedsChannelExpansion(GLenum internalformat,
                                     uint32_t pixelFormat)
{
    return mglRenderTextureNeedsChannelExpansion(
        (uint32_t)internalformat, pixelFormat) != 0;
}

uint8_t *mglCreateChannelExpandedUpload(Texture *tex,
                                        uint32_t pixelFormat,
                                        const uint8_t *srcData,
                                        size_t width,
                                        size_t height,
                                        size_t srcBytesPerRow,
                                        size_t *outBytesPerRow,
                                        size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 ||
        srcBytesPerRow == 0 || !outBytesPerRow || !outBytesPerImage ||
        !mglTextureNeedsChannelExpansion(tex->internalformat,
                                         (uint32_t)pixelFormat)) {
        return NULL;
    }


    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCreateChannelExpandedUpload(
        (uint32_t)tex->internalformat, (uint32_t)pixelFormat,
        srcData, (size_t)width, (size_t)height, (size_t)srcBytesPerRow,
        &outBPR, &outBPI);
    if (result) {
        *outBytesPerRow = outBPR;
        *outBytesPerImage = outBPI;
    }
    return result;
}

uint8_t *mglCreateRGBA8ExpandedUpload(Texture *tex,
                                      const uint8_t *srcData,
                                      size_t width,
                                      size_t height,
                                      size_t srcBytesPerRow,
                                      size_t *outBytesPerRow,
                                      size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 ||
        srcBytesPerRow == 0 || !outBytesPerRow || !outBytesPerImage ||
        !mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, 70u)) {
        return NULL;
    }


    return mglRenderCreateRGBA8ExpandedUpload(
        srcData, width, height, srcBytesPerRow,
        (uint32_t)tex->internalformat, outBytesPerRow, outBytesPerImage);
}


/* === Layer pixel format helpers === */

bool mglMetalLayerPixelFormatIsSupported(uint32_t pixelFormat)
{
    return mglRenderMetalLayerPixelFormatIsSupported(pixelFormat) != 0;
}

uint32_t mglSRGBPixelFormat(uint32_t fmt)
{
    return mglRenderSRGBPixelFormat(fmt);
}

uint32_t mglLinearPixelFormat(uint32_t fmt)
{
    return mglRenderLinearPixelFormat(fmt);
}

uint32_t mglEffectiveMTLPixelFormatForTexture(uint32_t fmt, Texture *tex)
{
    uint32_t decode = tex ? (uint32_t)tex->params.srgb_decode_ext : 0u;
    return mglRenderEffectiveMTLPixelFormat(fmt, decode);
}
