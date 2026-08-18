/*
 * mgl_texture_compat.m
 * MGL
 *
 * Implementation of the Texture Compatibility Subsystem.
 *
 * See mgl_texture_compat.h for the architectural rationale.  This module
 * owns the pure spec-compliance helpers for translating OpenGL texture
 * semantics to Metal:
 *   - Pixel format classification (depth/stencil, packed, data-kind).
 *   - Mipmap level dimension computation.
 *   - Sampled texture view creation for base_level > 0.
 *   - GL swizzle → Metal swizzle mapping (single-channel R8 expansion).
 *   - RGB → RGBA channel expansion for formats Metal does not support
 *     natively.
 *
 * The helpers here are pure: they do not touch the renderer ivar, the
 * command buffer, or the render encoder.  This makes them testable in
 * isolation and lets the texture-upload paths in MGLRenderer.m / MGLTextures.m
 * call them without dragging in renderer-internal state.
 *
 * External dependencies:
 *   - numComponentsForFormat / sizeForInternalFormat (pixel_utils.c) for
 *     GL format introspection.
 *   - Metal framework for MGLPixelFormat / MTLTexture / MTLTextureSwizzle.
 */

#import "mgl_texture_compat.h"
#include "mgl_render_cpp.h"

MGLTextureDataKind mglTextureDataKindForPixelFormat(uint32_t pixelFormat)
{
    return (MGLTextureDataKind)mglRenderCppTextureDataKindForPixelFormat(
        (uint32_t)pixelFormat);
}

const char *mglTextureDataKindName(MGLTextureDataKind kind)
{
    return mglRenderCppTextureDataKindName((uint32_t)kind);
}

size_t mglMetalTextureLevelDimension(size_t base, size_t level)
{
    /* P4.5 (item 1141/887): mip 级维度循环在 C++
     * （mglRenderCppMetalTextureLevelDimension，两门共用）。 */
    return (size_t)mglRenderCppMetalTextureLevelDimension(
        (uint64_t)base, (uint64_t)level);
}

void *mglSampledTextureViewForBaseLevel(Texture *ptr, void *texture)
{
    (void)mglRenderCppSampledTextureViewForBaseLevel(ptr, texture, &texture);
    return texture;
}

size_t mglStoredColorComponentsForTexture(Texture *tex)
{
    if (!tex) {
        return 4;
    }
    return (size_t)mglRenderCppStoredColorComponents(
        (uint32_t)tex->internalformat);
}

uint32_t mglMTLSwizzleForGLSwizzle(Texture *tex, GLenum swizzle)
{
    size_t components = mglStoredColorComponentsForTexture(tex);
    return mglRenderCppMTLSwizzleForGLSwizzle(
        (uint32_t)swizzle, (uint32_t)components);
}

bool mglTextureUploadNeedsSingleChannelSwizzle(Texture *tex)
{
    if (!tex) {
        return false;
    }
    return mglRenderCppTextureUploadNeedsSingleChannelSwizzle(
        (uint32_t)tex->internalformat, tex->params.swizzled ? 1 : 0) != 0;
}

uint8_t mglResolveR8SwizzledComponent(Texture *tex, GLenum swizzle, uint8_t red)
{
    (void)tex;
    /* P4.5 (item 1111): thin delegate — single source of truth in C++. */
    return mglRenderCppResolveR8SwizzledComponent((uint32_t)swizzle, red);
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

    /* P4.5 (item 1111): R8 1B/px → RGBA8 expand in C++.  Non-R8 still
     * returns NULL so callers fall back to MTLTextureDescriptor.swizzle. */
    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCppCreateSingleChannelSwizzledUpload(
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
    return mglRenderCppTextureInternalFormatNeedsRGBA8Expansion(
        (uint32_t)internalformat, pixelFormat) != 0;
}

bool mglTextureNeedsChannelExpansion(GLenum internalformat,
                                     uint32_t pixelFormat)
{
    return mglRenderCppTextureNeedsChannelExpansion(
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

    /* P4.5 (item 1111): 表格 + 校验 + 展开体在 C++
     * （mglRenderCppCreateChannelExpandedUpload，纯数据变换，两门共用；
     * 逐格式位布局与内联版逐字节一致）。 */
    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCppCreateChannelExpandedUpload(
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

    /* P4.4: 旧式 packed 格式 → RGBA8 的展开体在 C++
     * （mglRenderCppCreateRGBA8ExpandedUpload，纯数据变换，两门共用；
     * 逐格式位布局与内联版逐字节一致）。 */
    return mglRenderCppCreateRGBA8ExpandedUpload(
        srcData, width, height, srcBytesPerRow,
        (uint32_t)tex->internalformat, outBytesPerRow, outBytesPerImage);
}


/* === Layer pixel format helpers === */

bool mglMetalLayerPixelFormatIsSupported(uint32_t pixelFormat)
{
    return mglRenderCppMetalLayerPixelFormatIsSupported(pixelFormat) != 0;
}

uint32_t mglSRGBPixelFormat(uint32_t fmt)
{
    return mglRenderCppSRGBPixelFormat(fmt);
}

uint32_t mglLinearPixelFormat(uint32_t fmt)
{
    return mglRenderCppLinearPixelFormat(fmt);
}

uint32_t mglEffectiveMTLPixelFormatForTexture(uint32_t fmt, Texture *tex)
{
    uint32_t decode = tex ? (uint32_t)tex->params.srgb_decode_ext : 0u;
    return mglRenderCppEffectiveMTLPixelFormat(fmt, decode);
}
