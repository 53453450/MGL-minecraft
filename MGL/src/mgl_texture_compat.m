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
 *   - Metal framework for MTLPixelFormat / MTLTexture / MTLTextureSwizzle.
 */

#import "mgl_texture_compat.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>

#import "mgl_trace_log.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"
#include "mgl_render_cpp_objc.h" /* P4: ref typedefs */

/* GL format introspection helpers implemented in pixel_utils.c.  Declared
 * here so this module does not need to include the full MGLRenderer private
 * header. */
GLuint numComponentsForFormat(GLenum format);
GLuint sizeForInternalFormat(GLenum internalformat, GLenum format, GLenum type);

static MGLMetalTextureRef mglTextureCompatCreateView(
    MGLMetalTextureRef texture,
    NSRange levels,
    NSRange slices,
    BOOL useSwizzle,
    MTLTextureSwizzleChannels swizzle)
{
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() != NULL) {
        void *view = NULL;
        if (mglRenderCppCreateTextureViewRange(
                (__bridge void *)texture,
                (uint32_t)texture.pixelFormat,
                (uint32_t)texture.textureType,
                levels.location, levels.length,
                slices.location, slices.length,
                useSwizzle ? 1 : 0,
                (uint32_t)swizzle.red,
                (uint32_t)swizzle.green,
                (uint32_t)swizzle.blue,
                (uint32_t)swizzle.alpha,
                &view) == 0 && view) {
            return (__bridge_transfer MGLMetalTextureRef)view;
        }
    }
    if (useSwizzle) {
        return [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                          textureType:texture.textureType
                                               levels:levels
                                               slices:slices
                                              swizzle:swizzle];
    }
    return [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                      textureType:texture.textureType
                                           levels:levels
                                           slices:slices];
}

static bool mglTextureMinFilterUsesMipmaps(GLenum minFilter)
{
    switch (minFilter) {
        case GL_NEAREST_MIPMAP_NEAREST:
        case GL_LINEAR_MIPMAP_NEAREST:
        case GL_NEAREST_MIPMAP_LINEAR:
        case GL_LINEAR_MIPMAP_LINEAR:
            return true;
        default:
            return false;
    }
}

MGLTextureDataKind mglTextureDataKindForPixelFormat(MTLPixelFormat pixelFormat)
{
    return (MGLTextureDataKind)mglRenderCppTextureDataKindForPixelFormat(
        (uint32_t)pixelFormat);
}

const char *mglTextureDataKindName(MGLTextureDataKind kind)
{
    switch (kind) {
        case MGLTextureDataKindFloat: return "float";
        case MGLTextureDataKindSint:  return "sint";
        case MGLTextureDataKindUint:  return "uint";
        case MGLTextureDataKindDepth: return "depth";
        default:                      return "unknown";
    }
}

NSUInteger mglMetalTextureLevelDimension(NSUInteger base, NSUInteger level)
{
    /* P4.5 (item 1141/887): mip 级维度循环在 C++
     * （mglRenderCppMetalTextureLevelDimension，两门共用）。 */
    return (NSUInteger)mglRenderCppMetalTextureLevelDimension(
        (uint64_t)base, (uint64_t)level);
}

MGLMetalTextureRef mglSampledTextureViewForBaseLevel(Texture *ptr,
                                                 MGLMetalTextureRef texture)
{
    if (!ptr || !texture) return texture;
    if (ptr->mipmap_levels == 0u) return texture;

    GLuint baseLevel = ptr->params.base_level;
    if (baseLevel >= ptr->mipmap_levels) return texture;
    if ((NSUInteger)baseLevel >= texture.mipmapLevelCount) {
        return texture;
    }

    GLuint maxLevel = (ptr->params.max_level == 1000u)
        ? (ptr->mipmap_levels - 1u)
        : ptr->params.max_level;
    if (maxLevel < baseLevel) maxLevel = baseLevel;
    if (maxLevel >= ptr->mipmap_levels) maxLevel = ptr->mipmap_levels - 1u;
    if ((NSUInteger)maxLevel >= texture.mipmapLevelCount) maxLevel = (GLuint)texture.mipmapLevelCount - 1u;

    NSUInteger levelCount = maxLevel - baseLevel + 1u;
    if (levelCount == 0u) return texture;

    /* Fast path: no BASE/MAX window — common case for full-atlas sampling.
     * Still build a view when base==0 but MAX_LEVEL restricts (MC GpuTextureView
     * often uses MAX_LEVEL=0 while the Y-flip copy keeps the full mip chain). */
    if (baseLevel == 0u &&
        (NSUInteger)levelCount >= texture.mipmapLevelCount) {
        return texture;
    }

    /* Cache hit — return the cached view when source texture, base_level,
     * and max_level all match.  This avoids per-draw newTextureViewWithPixelFormat:
     * allocation when the texture params haven't changed (common case). */
    if (ptr->mtl_base_level_view != NULL &&
        ptr->mtl_base_level_view_source == (__bridge void *)texture &&
        ptr->mtl_base_level_view_base == baseLevel &&
        ptr->mtl_base_level_view_max == maxLevel) {
        return (__bridge MGLMetalTextureRef)ptr->mtl_base_level_view;
    }

    NSUInteger sliceCount = texture.arrayLength;
    if (texture.textureType == MTLTextureTypeCube ||
        texture.textureType == MTLTextureTypeCubeArray) {
        sliceCount = texture.arrayLength * 6u;
    }

    /* GL swizzle is texture-object state.  The source Metal texture bakes the
     * swizzle into MTLTextureDescriptor.swizzle at creation time, but a view
     * created with newTextureViewWithPixelFormat:levels:slices: defaults to
     * identity swizzle and does NOT inherit the source's channel routing.
     * Re-apply the GL swizzle via the swizzle-aware view API (macOS 10.15+)
     * so sampling the base-level view matches sampling the source texture. */
    MTLTextureSwizzle sw_r = mglMTLSwizzleForGLSwizzle(ptr, ptr->params.swizzle_r);
    MTLTextureSwizzle sw_g = mglMTLSwizzleForGLSwizzle(ptr, ptr->params.swizzle_g);
    MTLTextureSwizzle sw_b = mglMTLSwizzleForGLSwizzle(ptr, ptr->params.swizzle_b);
    MTLTextureSwizzle sw_a = mglMTLSwizzleForGLSwizzle(ptr, ptr->params.swizzle_a);
    BOOL swizzleIsIdentity = (sw_r == MTLTextureSwizzleRed &&
                              sw_g == MTLTextureSwizzleGreen &&
                              sw_b == MTLTextureSwizzleBlue &&
                              sw_a == MTLTextureSwizzleAlpha);

    MGLMetalTextureRef levelView = nil;
    if (swizzleIsIdentity) {
        levelView = mglTextureCompatCreateView(
            texture, NSMakeRange(baseLevel, levelCount),
            NSMakeRange(0, sliceCount), NO,
            MTLTextureSwizzleChannelsMake(
                MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha));
    } else if (@available(macOS 10.15, *)) {
        MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(sw_r, sw_g, sw_b, sw_a);
        levelView = mglTextureCompatCreateView(
            texture, NSMakeRange(baseLevel, levelCount),
            NSMakeRange(0, sliceCount), YES, swizzle);
    } else {
        /* Pre-10.15 fallback: swizzle-aware view API unavailable.  The view
         * will sample with identity swizzle; the source texture's baked-in
         * swizzle is lost on the view.  This matches the prior behavior. */
        levelView = mglTextureCompatCreateView(
            texture, NSMakeRange(baseLevel, levelCount),
            NSMakeRange(0, sliceCount), NO,
            MTLTextureSwizzleChannelsMake(
                MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha));
    }
    if (levelView) {
        /* Store in cache, releasing the old view if any. */
        if (ptr->mtl_base_level_view) {
            CFRelease(ptr->mtl_base_level_view);
        }
        CFRetain((__bridge CFTypeRef)levelView);
        ptr->mtl_base_level_view = (__bridge void *)levelView;
        ptr->mtl_base_level_view_source = (__bridge void *)texture;
        ptr->mtl_base_level_view_base = baseLevel;
        ptr->mtl_base_level_view_max = maxLevel;
        static uint64_t s_sampledBaseViewTraceCount = 0;
        uint64_t hit = ++s_sampledBaseViewTraceCount;
        if (hit <= 256ull || (hit % 1024ull) == 0ull) {
            mglTraceLogExternal("TEX_BASE_VIEW tex=%u target=0x%x minFilter=0x%x mipFilter=%d base=%u max=%u levels=%lu glLevels=%u mips=%u original=%p originalSize=%lux%lu originalLevels=%lu view=%p viewSize=%lux%lu viewLevels=%lu fmt=%lu type=%lu hit=%llu",
                                (unsigned)ptr->name,
                                (unsigned)ptr->target,
                                (unsigned)ptr->params.min_filter,
                                mglTextureMinFilterUsesMipmaps(ptr->params.min_filter) ? 1 : 0,
                                (unsigned)baseLevel,
                                (unsigned)maxLevel,
                                (unsigned long)levelCount,
                                (unsigned)ptr->num_levels,
                                (unsigned)ptr->mipmap_levels,
                                texture,
                                (unsigned long)texture.width,
                                (unsigned long)texture.height,
                                (unsigned long)texture.mipmapLevelCount,
                                levelView,
                                (unsigned long)levelView.width,
                                (unsigned long)levelView.height,
                                (unsigned long)levelView.mipmapLevelCount,
                                (unsigned long)texture.pixelFormat,
                                (unsigned long)texture.textureType,
                                (unsigned long long)hit);
        }
    }
    return levelView ? levelView : texture;
}

NSUInteger mglStoredColorComponentsForTexture(Texture *tex)
{
    if (!tex) {
        return 4;
    }

    GLuint components = numComponentsForFormat(tex->internalformat);
    return components > 0 ? (NSUInteger)components : 4;
}

MTLTextureSwizzle mglMTLSwizzleForGLSwizzle(Texture *tex, GLenum swizzle)
{
    NSUInteger components = mglStoredColorComponentsForTexture(tex);

    switch (swizzle)
    {
        case GL_ZERO: return MTLTextureSwizzleZero;
        case GL_ONE: return MTLTextureSwizzleOne;
        case GL_RED: return components >= 1 ? MTLTextureSwizzleRed : MTLTextureSwizzleZero;
        case GL_GREEN: return components >= 2 ? MTLTextureSwizzleGreen : MTLTextureSwizzleZero;
        case GL_BLUE: return components >= 3 ? MTLTextureSwizzleBlue : MTLTextureSwizzleZero;
        case GL_ALPHA: return components >= 4 ? MTLTextureSwizzleAlpha : MTLTextureSwizzleOne;
        default:
            NSLog(@"MGL ERROR: Unknown swizzle value 0x%x in swizzleTexDesc", swizzle);
            return MTLTextureSwizzleZero;
    }
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
                                              NSUInteger width,
                                              NSUInteger height,
                                              NSUInteger srcBytesPerRow,
                                              NSUInteger *outBytesPerRow,
                                              NSUInteger *outBytesPerImage)
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
        *outBytesPerRow = (NSUInteger)outBPR;
        *outBytesPerImage = (NSUInteger)outBPI;
    }
    return result;
}

bool mglTextureInternalFormatNeedsRGBA8Expansion(GLenum internalformat,
                                                 uint32_t pixelFormat)
{
    /* Metal has no RGB8 pixel format, so GL_RGB8-family internal formats are
     * backed by RGBA8 variants.  The CPU data is 3 bytes/pixel (RGB) but Metal
     * expects 4 bytes/pixel (RGBA), so expansion is required. */
    bool isRGBA8Variant =
        (pixelFormat == MTLPixelFormatRGBA8Unorm ||
         pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB ||
         pixelFormat == MTLPixelFormatRGBA8Snorm ||
         pixelFormat == MTLPixelFormatRGBA8Sint ||
         pixelFormat == MTLPixelFormatRGBA8Uint);
    if (!isRGBA8Variant) {
        return false;
    }

    switch (internalformat) {
        /* Packed legacy formats (already handled) */
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_R3_G3_B2:
        /* 8-bit RGB formats – 3 bytes/pixel in CPU, 4 bytes/pixel in Metal */
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
        /* GL_RGB565: Metal's MTLPixelFormatB5G6R5Unorm reverses the channel
         * order (B in the high bits vs R in the high bits for GL), so back
         * it with RGBA8Unorm and let the CPU expansion rearrange channels. */
        case GL_RGB565:
            return true;
        default:
            return false;
    }
}

bool mglTextureNeedsChannelExpansion(GLenum internalformat,
                                     uint32_t pixelFormat)
{
    /* Only handle non-RGBA8 Metal pixel formats */
    bool isRGBA16Variant =
        (pixelFormat == MTLPixelFormatRGBA16Unorm ||
         pixelFormat == MTLPixelFormatRGBA16Snorm ||
         pixelFormat == MTLPixelFormatRGBA16Float ||
         pixelFormat == MTLPixelFormatRGBA16Sint ||
         pixelFormat == MTLPixelFormatRGBA16Uint);
    bool isRGBA32Variant =
        (pixelFormat == MTLPixelFormatRGBA32Float ||
         pixelFormat == MTLPixelFormatRGBA32Sint ||
         pixelFormat == MTLPixelFormatRGBA32Uint);
    if (!isRGBA16Variant && !isRGBA32Variant) {
        return false;
    }

    switch (internalformat) {
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_RGB16F:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGB12:
            return true;
        default:
            return false;
    }
}

uint8_t *mglCreateChannelExpandedUpload(Texture *tex,
                                        MTLPixelFormat pixelFormat,
                                        const uint8_t *srcData,
                                        NSUInteger width,
                                        NSUInteger height,
                                        NSUInteger srcBytesPerRow,
                                        NSUInteger *outBytesPerRow,
                                        NSUInteger *outBytesPerImage)
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
        *outBytesPerRow = (NSUInteger)outBPR;
        *outBytesPerImage = (NSUInteger)outBPI;
    }
    return result;
}

uint8_t *mglCreateRGBA8ExpandedUpload(Texture *tex,
                                      const uint8_t *srcData,
                                      NSUInteger width,
                                      NSUInteger height,
                                      NSUInteger srcBytesPerRow,
                                      NSUInteger *outBytesPerRow,
                                      NSUInteger *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 ||
        srcBytesPerRow == 0 || !outBytesPerRow || !outBytesPerImage ||
        !mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, MTLPixelFormatRGBA8Unorm)) {
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

BOOL mglMetalLayerPixelFormatIsSupported(MTLPixelFormat pixelFormat)
{
    return mglRenderCppMetalLayerPixelFormatIsSupported((uint32_t)pixelFormat)
        ? YES : NO;
}

MTLPixelFormat mglSRGBPixelFormat(MTLPixelFormat fmt)
{
    return (MTLPixelFormat)mglRenderCppSRGBPixelFormat((uint32_t)fmt);
}

MTLPixelFormat mglLinearPixelFormat(MTLPixelFormat fmt)
{
    return (MTLPixelFormat)mglRenderCppLinearPixelFormat((uint32_t)fmt);
}

MTLPixelFormat mglEffectiveMTLPixelFormatForTexture(MTLPixelFormat fmt, Texture *tex)
{
    uint32_t decode = tex ? (uint32_t)tex->params.srgb_decode_ext : 0u;
    return (MTLPixelFormat)mglRenderCppEffectiveMTLPixelFormat(
        (uint32_t)fmt, decode);
}
