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
    switch (pixelFormat) {
        case MTLPixelFormatR8Sint:
        case MTLPixelFormatRG8Sint:
        case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatR16Sint:
        case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA32Sint:
            return MGLTextureDataKindSint;

        case MTLPixelFormatR8Uint:
        case MTLPixelFormatRG8Uint:
        case MTLPixelFormatRGBA8Uint:
        case MTLPixelFormatR16Uint:
        case MTLPixelFormatRG16Uint:
        case MTLPixelFormatRGBA16Uint:
        case MTLPixelFormatR32Uint:
        case MTLPixelFormatRG32Uint:
        case MTLPixelFormatRGBA32Uint:
        case MTLPixelFormatRGB10A2Uint:
            return MGLTextureDataKindUint;

        case MTLPixelFormatInvalid:
            return MGLTextureDataKindUnknown;

        case MTLPixelFormatDepth16Unorm:
        case MTLPixelFormatDepth32Float:
        case MTLPixelFormatDepth24Unorm_Stencil8:
        case MTLPixelFormatDepth32Float_Stencil8:
            return MGLTextureDataKindDepth;

        default:
            return MGLTextureDataKindFloat;
    }
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
    NSUInteger value = MAX((NSUInteger)1u, base);
    while (level-- > 0u && value > 1u) {
        value >>= 1u;
    }
    return MAX((NSUInteger)1u, value);
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
    if (!tex || !tex->params.swizzled) {
        return false;
    }

    /* Single-channel GL formats (R-only) rely on GL swizzle to route the lone
     * Red value into G/B/A.  Metal applies swizzle via MTLTextureDescriptor
     * at texture-creation time, but to keep single-channel swizzle behavior
     * consistent across all R-only formats (and aligned with the RGBA8
     * expansion path), expand every single-channel format on the CPU into a
     * 4-channel buffer and let mglResolveR8SwizzledComponent route channels.
     * This avoids depending on MTLTextureDescriptor.swizzle, which some Apple
     * GPU paths handle inconsistently for single-channel textures. */
    switch (tex->internalformat)
    {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
            return true;
        default:
            return false;
    }
}

uint8_t mglResolveR8SwizzledComponent(Texture *tex, GLenum swizzle, uint8_t red)
{
    (void)tex;

    switch (swizzle)
    {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0xff;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0x00;
    }
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

    /* mglTextureUploadNeedsSingleChannelSwizzle gates all R-only formats into
     * this path, but the byte-level expansion below is only correct for R8
     * (1 byte/pixel source → RGBA8 4 bytes/pixel destination).  Multi-byte
     * single-channel formats (R16F/R32F/R16/R32I/...) require a matching
     * RGBA variant destination and per-format zero/one constants, plus a
     * corresponding texture-format promotion at creation time (handled
     * elsewhere).  Return NULL here so the caller falls back to the original
     * upload data and relies on MTLTextureDescriptor.swizzle, preserving the
     * prior behavior for those formats. */
    if (tex->internalformat != GL_R8) {
        return NULL;
    }

    NSUInteger dstBytesPerRow = width * 4u;
    NSUInteger dstBytesPerImage = dstBytesPerRow * height;
    if (dstBytesPerImage == 0 || dstBytesPerImage > (512 * 1024 * 1024)) {
        return NULL;
    }

    uint8_t *dst = (uint8_t *)malloc(dstBytesPerImage);
    if (!dst) {
        return NULL;
    }

    for (NSUInteger row = 0; row < height; row++) {
        uint8_t *dstRow = dst + row * dstBytesPerRow;
        const uint8_t *srcRow = srcData + row * srcBytesPerRow;
        for (NSUInteger x = 0; x < width; x++) {
            uint8_t red = srcRow[x];
            uint8_t *out = dstRow + (x * 4u);
            out[0] = mglResolveR8SwizzledComponent(tex, tex->params.swizzle_r, red);
            out[1] = mglResolveR8SwizzledComponent(tex, tex->params.swizzle_g, red);
            out[2] = mglResolveR8SwizzledComponent(tex, tex->params.swizzle_b, red);
            out[3] = mglResolveR8SwizzledComponent(tex, tex->params.swizzle_a, red);
        }
    }

    *outBytesPerRow = dstBytesPerRow;
    *outBytesPerImage = dstBytesPerImage;
    return dst;
}

bool mglTextureInternalFormatNeedsRGBA8Expansion(GLenum internalformat,
                                                 MTLPixelFormat pixelFormat)
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
                                     MTLPixelFormat pixelFormat)
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
        srcBytesPerRow == 0 || !outBytesPerRow || !outBytesPerImage) {
        return NULL;
    }

    /* Determine source and destination parameters */
    NSUInteger srcCompBytes = 0;  /* bytes per component in source */
    NSUInteger dstCompBytes = 0;  /* bytes per component in destination */
    NSUInteger srcPixelBytes = 0; /* bytes per pixel in source (3 channels) */
    NSUInteger dstPixelBytes = 0; /* bytes per pixel in destination (4 channels) */

    /* Alpha default value as uint64_t to handle all sizes */
    uint64_t alphaDefault = 0;

    switch (pixelFormat) {
        case MTLPixelFormatRGBA16Unorm:
            srcCompBytes = 2; dstCompBytes = 2;
            srcPixelBytes = 6; dstPixelBytes = 8;
            alphaDefault = 65535; /* 1.0 in unorm16 */
            break;
        case MTLPixelFormatRGBA16Snorm:
            srcCompBytes = 2; dstCompBytes = 2;
            srcPixelBytes = 6; dstPixelBytes = 8;
            alphaDefault = 32767; /* 1.0 in snorm16 */
            break;
        case MTLPixelFormatRGBA16Float:
            srcCompBytes = 2; dstCompBytes = 2;
            srcPixelBytes = 6; dstPixelBytes = 8;
            alphaDefault = 0x3C00; /* 1.0 in half float */
            break;
        case MTLPixelFormatRGBA16Sint:
            srcCompBytes = 2; dstCompBytes = 2;
            srcPixelBytes = 6; dstPixelBytes = 8;
            alphaDefault = 1;
            break;
        case MTLPixelFormatRGBA16Uint:
            srcCompBytes = 2; dstCompBytes = 2;
            srcPixelBytes = 6; dstPixelBytes = 8;
            alphaDefault = 1;
            break;
        case MTLPixelFormatRGBA32Float:
            srcCompBytes = 4; dstCompBytes = 4;
            srcPixelBytes = 12; dstPixelBytes = 16;
            { float f = 1.0f; memcpy(&alphaDefault, &f, sizeof(f)); }
            break;
        case MTLPixelFormatRGBA32Sint:
            srcCompBytes = 4; dstCompBytes = 4;
            srcPixelBytes = 12; dstPixelBytes = 16;
            alphaDefault = 1;
            break;
        case MTLPixelFormatRGBA32Uint:
            srcCompBytes = 4; dstCompBytes = 4;
            srcPixelBytes = 12; dstPixelBytes = 16;
            alphaDefault = 1;
            break;
        default:
            return NULL;
    }

    /* Verify source pixel bytes match internal format */
    size_t expectedSrcBytes = sizeForInternalFormat(tex->internalformat, 0, 0);
    if (expectedSrcBytes > 0 && expectedSrcBytes != srcPixelBytes) {
        /* For GL_RGB12, sizeForInternalFormat might return a different value.
         * Use the expected value if it's reasonable. */
        if (tex->internalformat == GL_RGB12 && expectedSrcBytes == 6) {
            /* OK - RGB12 is stored as 3x16-bit = 6 bytes */
        } else if (expectedSrcBytes != srcPixelBytes) {
            return NULL;
        }
    }

    if (srcBytesPerRow < width * srcPixelBytes) {
        return NULL;
    }

    NSUInteger dstBytesPerRow = width * dstPixelBytes;
    NSUInteger dstBytesPerImage = dstBytesPerRow * height;
    if (dstBytesPerImage == 0 || dstBytesPerImage > (512 * 1024 * 1024)) {
        return NULL;
    }

    uint8_t *dst = (uint8_t *)malloc(dstBytesPerImage);
    if (!dst) {
        return NULL;
    }

    for (NSUInteger row = 0; row < height; row++) {
        const uint8_t *srcRow = srcData + row * srcBytesPerRow;
        uint8_t *dstRow = dst + row * dstBytesPerRow;
        for (NSUInteger x = 0; x < width; x++) {
            const uint8_t *srcPixel = srcRow + x * srcPixelBytes;
            uint8_t *dstPixel = dstRow + x * dstPixelBytes;
            /* Copy 3 channels (R, G, B) from source to destination */
            memcpy(dstPixel, srcPixel, srcPixelBytes);
            /* Set alpha channel to default value */
            memcpy(dstPixel + srcPixelBytes, &alphaDefault, dstCompBytes);
        }
    }

    *outBytesPerRow = dstBytesPerRow;
    *outBytesPerImage = dstBytesPerImage;
    return dst;
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
    switch (pixelFormat) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return YES;
        default:
            return NO;
    }
}

MTLPixelFormat mglSRGBPixelFormat(MTLPixelFormat fmt)
{
    switch (fmt) {
        case MTLPixelFormatRGBA8Unorm:   return MTLPixelFormatRGBA8Unorm_sRGB;
        case MTLPixelFormatBGRA8Unorm:   return MTLPixelFormatBGRA8Unorm_sRGB;
        default: return fmt;
    }
}

MTLPixelFormat mglLinearPixelFormat(MTLPixelFormat fmt)
{
    switch (fmt) {
        case MTLPixelFormatRGBA8Unorm_sRGB: return MTLPixelFormatRGBA8Unorm;
        case MTLPixelFormatBGRA8Unorm_sRGB: return MTLPixelFormatBGRA8Unorm;
        default: return fmt;
    }
}

MTLPixelFormat mglEffectiveMTLPixelFormatForTexture(MTLPixelFormat fmt, Texture *tex)
{
    /* GL_EXT_texture_sRGB_decode: GL_SKIP_DECODE_EXT requests that sRGB-backed
     * texture data be sampled as linear, bypassing the automatic sRGB decode
     * that the *_sRGB Metal pixel formats apply.  Downgrade the sRGB pixel
     * format to its linear variant in that case.  An unset (0) field — the
     * bzero default at texture creation — or GL_DECODE_EXT leaves the format
     * unchanged so the sRGB pipeline decodes normally.  Intended to be called
     * from the texture-creation path (MGLRenderer+Texture.m) when selecting
     * the Metal pixel format for an sRGB internal format. */
    if (tex && tex->params.srgb_decode_ext == GL_SKIP_DECODE_EXT) {
        return mglLinearPixelFormat(fmt);
    }
    return fmt;
}
