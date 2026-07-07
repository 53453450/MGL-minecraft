/*
 * mgl_texture_compat.h
 * MGL
 *
 * Texture Compatibility Subsystem.
 *
 * Bridges the semantic gap between OpenGL texture semantics and Metal texture
 * semantics.  Covers several spec-compliance areas:
 *
 *   - Pixel format classification (depth/stencil, packed, data-kind for
 *     shader-side int/uint/float/depth matching).
 *   - GL internal-format classification (looks-depth-or-stencil).
 *   - Mipmap level dimension computation (GL clamps to 1, Metal requires
 *     explicit level dimensions).
 *   - Sampled texture view creation for base_level > 0 (Metal has no base
 *     level concept; must use newTextureViewWithPixelFormat:levels:).
 *   - GL swizzle → Metal swizzle mapping (single-channel R8 swizzle expansion).
 *   - RGB → RGBA channel expansion for formats Metal does not support
 *     natively (GL_RGB8 / GL_RGB16F / GL_RGB32F ... backed by RGBA variants).
 *
 * This module is pure specification-compliance machinery: every OpenGL
 * program that uses the corresponding GL texture features needs these
 * translations when running on Metal, regardless of application.
 */

#ifndef MGL_TEXTURE_COMPAT_H
#define MGL_TEXTURE_COMPAT_H

#include "glm_context.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* === Pixel format classification ===
 *
 * These are static inline so the compiler can fold them at call sites that
 * pass constant pixel formats (very common in switch statements). */

/* Returns true if `format` is a depth, stencil, or packed depth-stencil
 * Metal pixel format. */
static inline bool mglMetalPixelFormatIsDepthOrStencil(MTLPixelFormat format)
{
    return format == MTLPixelFormatDepth16Unorm ||
           format == MTLPixelFormatDepth32Float ||
           format == MTLPixelFormatDepth24Unorm_Stencil8 ||
           format == MTLPixelFormatDepth32Float_Stencil8 ||
           format == MTLPixelFormatStencil8;
}

/* Returns true if `format` is a packed depth-stencil Metal pixel format
 * (Depth24Unorm_Stencil8 or Depth32Float_Stencil8).  Used to decide whether
 * depth and stencil attachments must share the same texture. */
static inline bool mglMetalPixelFormatIsPackedDepthStencil(MTLPixelFormat format)
{
    return format == MTLPixelFormatDepth24Unorm_Stencil8 ||
           format == MTLPixelFormatDepth32Float_Stencil8;
}

/* Returns true if `internalformat` is a GL depth, depth-stencil, or
 * stencil-only internal format.  Used to gate depth/stencil render-target
 * paths and to skip color-only fallbacks. */
static inline bool mglRendererGLInternalFormatLooksDepthOrStencil(GLenum internalformat)
{
    switch (internalformat) {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX:
        case GL_STENCIL_INDEX8:
            return true;
        default:
            return false;
    }
}

/* === Texture data-kind matching ===
 *
 * GLSL uniform/sampler declarations carry a data kind (float, int, uint,
 * depth).  Metal pixel formats are typed; mismatched kinds cause sampler
 * binding to fail or sample garbage.  These helpers classify a Metal pixel
 * format into the matching GLSL data kind so sampler binding can verify
 * compatibility. */
typedef NS_ENUM(NSUInteger, MGLTextureDataKind) {
    MGLTextureDataKindUnknown = 0,
    MGLTextureDataKindFloat = 1,
    MGLTextureDataKindSint = 2,
    MGLTextureDataKindUint = 3,
    MGLTextureDataKindDepth = 4,
};

MGLTextureDataKind mglTextureDataKindForPixelFormat(MTLPixelFormat pixelFormat);

/* Returns true if `pixelFormat`'s data kind matches `expectedKind` (or if
 * `expectedKind` is Unknown, which means "no constraint"). */
static inline bool mglTexturePixelFormatCompatibleWithExpectedDataKind(MTLPixelFormat pixelFormat,
                                                                      MGLTextureDataKind expectedKind)
{
    if (expectedKind == MGLTextureDataKindUnknown) {
        return true;
    }
    return mglTextureDataKindForPixelFormat(pixelFormat) == expectedKind;
}

const char *mglTextureDataKindName(MGLTextureDataKind kind);

/* === Mipmap level dimension ===
 *
 * GL spec 8.14.3: mip level N has dimensions max(1, floor(base >> N)).
 * Metal textures store all levels but the dimension must be computed
 * explicitly when blitting/uploading. */
NSUInteger mglMetalTextureLevelDimension(NSUInteger base, NSUInteger level);

/* === Sampled texture view for base_level > 0 ===
 *
 * GL spec 8.14.2: TEXTURE_BASE_LEVEL selects the lowest mipmap level that
 * is sampled.  Metal has no base level concept — the entire texture is
 * always addressable.  To honor GL semantics, create a texture view that
 * starts at `base_level` and spans `[base_level, max_level]`.
 *
 * Returns `texture` unchanged if no view is needed (base_level == 0 or
 * invalid range).  Returns a new autoreleased view otherwise. */
id<MTLTexture> mglSampledTextureViewForBaseLevel(Texture *ptr,
                                                 id<MTLTexture> texture);

/* === Swizzle ===
 *
 * GL texture objects carry swizzle state (TEXTURE_SWIZZLE_R/G/B/A).  Metal
 * applies swizzle at texture-creation time via MTLTextureSwizzleChannels.
 * These helpers translate GL swizzle enums to Metal and handle the special
 * case of single-channel R8 swizzle expansion at upload time. */
NSUInteger mglStoredColorComponentsForTexture(Texture *tex);
MTLTextureSwizzle mglMTLSwizzleForGLSwizzle(Texture *tex, GLenum swizzle);
bool mglTextureUploadNeedsSingleChannelSwizzle(Texture *tex);
uint8_t mglResolveR8SwizzledComponent(Texture *tex, GLenum swizzle, uint8_t red);
uint8_t *mglCreateSingleChannelSwizzledUpload(Texture *tex,
                                              const uint8_t *srcData,
                                              NSUInteger width,
                                              NSUInteger height,
                                              NSUInteger srcBytesPerRow,
                                              NSUInteger *outBytesPerRow,
                                              NSUInteger *outBytesPerImage);

/* === RGB → RGBA channel expansion ===
 *
 * Metal has no RGB8 / RGB16F / RGB32F pixel format — GL RGB-family formats
 * are backed by RGBA variants.  CPU upload data is 3 channels but Metal
 * expects 4, so expansion is required. */
bool mglTextureInternalFormatNeedsRGBA8Expansion(GLenum internalformat,
                                                 MTLPixelFormat pixelFormat);
bool mglTextureNeedsChannelExpansion(GLenum internalformat,
                                     MTLPixelFormat pixelFormat);
uint32_t mglReadPackedUploadLE(const uint8_t *src, NSUInteger bytes);
uint8_t mglExpandUNormBitsTo8(uint32_t value, uint32_t bits);
uint8_t *mglCreateChannelExpandedUpload(Texture *tex,
                                        MTLPixelFormat pixelFormat,
                                        const uint8_t *srcData,
                                        NSUInteger width,
                                        NSUInteger height,
                                        NSUInteger srcBytesPerRow,
                                        NSUInteger *outBytesPerRow,
                                        NSUInteger *outBytesPerImage);

/* Create expanded upload data for legacy packed RGB formats (GL_RGB4/5/10/12,
 * GL_RGBA2/4, GL_RGB5_A1, GL_R3_G3_B2) that are backed by Metal RGBA8.
 * Reads packed pixels from `srcData` and writes RGBA8 to a newly malloc'd
 * buffer.  Returns NULL on failure. */
uint8_t *mglCreateRGBA8ExpandedUpload(Texture *tex,
                                      const uint8_t *srcData,
                                      NSUInteger width,
                                      NSUInteger height,
                                      NSUInteger srcBytesPerRow,
                                      NSUInteger *outBytesPerRow,
                                      NSUInteger *outBytesPerImage);

/* === Layer pixel format / compressed block helpers === */

/* Returns YES if `pixelFormat` is one of the MTLPixelFormat values supported
 * by the MGL render-target layer (currently BGRA8 linear / sRGB). */
BOOL mglMetalLayerPixelFormatIsSupported(MTLPixelFormat pixelFormat);

/* Returns the sRGB variant of a color-renderable pixel format, or the
 * original format if no sRGB variant exists. */
MTLPixelFormat mglSRGBPixelFormat(MTLPixelFormat fmt);

/* Returns the linear variant of a color-renderable pixel format, or the
 * original format if no linear variant exists. */
MTLPixelFormat mglLinearPixelFormat(MTLPixelFormat fmt);

/* Returns the compressed-block height (in pixels) for a Metal compressed
 * pixel format.  BC1-BC7 and ASTC 4x4 return 4; ASTC variants return
 * their Y dimension (5/6/8/10/12); uncompressed formats return 1. */
static inline NSUInteger mglMetalCompressedBlockHeight(MTLPixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case MTLPixelFormatBC1_RGBA:
        case MTLPixelFormatBC1_RGBA_sRGB:
        case MTLPixelFormatBC2_RGBA:
        case MTLPixelFormatBC2_RGBA_sRGB:
        case MTLPixelFormatBC3_RGBA:
        case MTLPixelFormatBC3_RGBA_sRGB:
        case MTLPixelFormatBC4_RUnorm:
        case MTLPixelFormatBC4_RSnorm:
        case MTLPixelFormatBC5_RGUnorm:
        case MTLPixelFormatBC5_RGSnorm:
        case MTLPixelFormatBC6H_RGBFloat:
        case MTLPixelFormatBC6H_RGBUfloat:
        case MTLPixelFormatBC7_RGBAUnorm:
        case MTLPixelFormatBC7_RGBAUnorm_sRGB:
        case MTLPixelFormatASTC_4x4_sRGB:
        case MTLPixelFormatASTC_4x4_LDR:
        case MTLPixelFormatASTC_4x4_HDR:
        case MTLPixelFormatASTC_5x4_sRGB:
        case MTLPixelFormatASTC_5x4_LDR:
        case MTLPixelFormatASTC_5x4_HDR:
            return 4u;
        case MTLPixelFormatASTC_5x5_sRGB:
        case MTLPixelFormatASTC_5x5_LDR:
        case MTLPixelFormatASTC_5x5_HDR:
        case MTLPixelFormatASTC_6x5_sRGB:
        case MTLPixelFormatASTC_6x5_LDR:
        case MTLPixelFormatASTC_6x5_HDR:
        case MTLPixelFormatASTC_8x5_sRGB:
        case MTLPixelFormatASTC_8x5_LDR:
        case MTLPixelFormatASTC_8x5_HDR:
        case MTLPixelFormatASTC_10x5_sRGB:
        case MTLPixelFormatASTC_10x5_LDR:
        case MTLPixelFormatASTC_10x5_HDR:
            return 5u;
        case MTLPixelFormatASTC_6x6_sRGB:
        case MTLPixelFormatASTC_6x6_LDR:
        case MTLPixelFormatASTC_6x6_HDR:
        case MTLPixelFormatASTC_8x6_sRGB:
        case MTLPixelFormatASTC_8x6_LDR:
        case MTLPixelFormatASTC_8x6_HDR:
        case MTLPixelFormatASTC_10x6_sRGB:
        case MTLPixelFormatASTC_10x6_LDR:
        case MTLPixelFormatASTC_10x6_HDR:
            return 6u;
        case MTLPixelFormatASTC_8x8_sRGB:
        case MTLPixelFormatASTC_8x8_LDR:
        case MTLPixelFormatASTC_8x8_HDR:
        case MTLPixelFormatASTC_10x8_sRGB:
        case MTLPixelFormatASTC_10x8_LDR:
        case MTLPixelFormatASTC_10x8_HDR:
            return 8u;
        case MTLPixelFormatASTC_10x10_sRGB:
        case MTLPixelFormatASTC_10x10_LDR:
        case MTLPixelFormatASTC_10x10_HDR:
        case MTLPixelFormatASTC_12x10_sRGB:
        case MTLPixelFormatASTC_12x10_LDR:
        case MTLPixelFormatASTC_12x10_HDR:
            return 10u;
        case MTLPixelFormatASTC_12x12_sRGB:
        case MTLPixelFormatASTC_12x12_LDR:
        case MTLPixelFormatASTC_12x12_HDR:
            return 12u;
        default:
            return 1u;
    }
}

/* Returns the number of upload rows for a compressed texture of the given
 * pixel height.  For uncompressed formats this equals `pixelHeight` (min 1).
 * For compressed formats the height is rounded up to the block height. */
static inline NSUInteger mglMetalUploadRowsForPixelFormat(MTLPixelFormat pixelFormat, NSUInteger pixelHeight)
{
    NSUInteger height = pixelHeight ? pixelHeight : 1u;
    NSUInteger blockHeight = mglMetalCompressedBlockHeight(pixelFormat);
    if (blockHeight <= 1u) {
        return height;
    }
    return (height + blockHeight - 1u) / blockHeight;
}

/* === Texture level read-only helpers ===
 *
 * Pure read-only accessors over Texture / TextureLevel.  Used by trace
 * logging, sampler binding, and CPU-upload gating paths.  Inline so the
 * compiler can fold NULL guards at call sites. */

/* Returns the base (mip 0) level of `tex` face 0, or NULL if tex has no
 * levels allocated. */
static inline TextureLevel *mglTraceTextureBaseLevel(Texture *tex)
{
    if (!tex || tex->num_levels == 0 || !tex->faces[0].levels) {
        return NULL;
    }

    return &tex->faces[0].levels[0];
}

/* Returns level `level` of `tex` face 0, or NULL if out of range / no
 * levels allocated. */
static inline TextureLevel *mglTextureAttachmentLevel(Texture *tex, GLuint level)
{
    if (!tex || tex->num_levels == 0 || !tex->faces[0].levels || level >= tex->num_levels) {
        return NULL;
    }

    return &tex->faces[0].levels[level];
}

/* Summarizes a texture level's ever_written / has_initialized_data /
 * last_init_source into the out-parameters (any may be NULL).  Missing
 * levels report 0 for all fields. */
static inline void mglTraceTextureLevelSummary(Texture *tex,
                                               GLuint level,
                                               GLuint *ever,
                                               GLuint *full,
                                               GLuint *source)
{
    TextureLevel *texLevel = mglTextureAttachmentLevel(tex, level);
    if (ever) {
        *ever = texLevel ? (GLuint)texLevel->ever_written : 0u;
    }
    if (full) {
        *full = texLevel ? (GLuint)texLevel->has_initialized_data : 0u;
    }
    if (source) {
        *source = texLevel ? (GLuint)texLevel->last_init_source : 0u;
    }
}

/* Returns true if `level` has CPU-side data that can be uploaded to a
 * Metal texture.  Considers data presence, completeness, and the
 * last_init_source (only CPU/PBO/MetalFill sources are uploadable). */
static inline bool mglTextureLevelHasUploadableCPUData(const TextureLevel *level)
{
    if (!level ||
        !level->complete ||
        !level->data ||
        level->data_size == 0u ||
        level->pitch == 0u) {
        return false;
    }

    switch (level->last_init_source) {
        case kTexImageCopy:
        case kTexImagePBO:
        case kTexSubImageCPU:
        case kTexSubImagePBO:
        case kTexMetalFill:
            return (level->has_initialized_data || level->ever_written) ? true : false;
        case kTexInitNone:
        case kTexImageNull:
        case kTexRenderTargetWrite:
        default:
            return false;
    }
}

/* Returns true if any level of any face in `tex` (up to `numFaces` faces
 * and `levelCount` levels) has uploadable CPU data. */
static inline bool mglTextureHasUploadableCPUData(Texture *tex, int numFaces, GLuint levelCount)
{
    if (!tex || numFaces <= 0 || levelCount == 0u) {
        return false;
    }

    for (int face = 0; face < numFaces; face++) {
        if (!tex->faces[face].levels) {
            continue;
        }
        for (GLuint level = 0; level < levelCount; level++) {
            if (mglTextureLevelHasUploadableCPUData(&tex->faces[face].levels[level])) {
                return true;
            }
        }
    }

    return false;
}

/* Returns tex->name, or 0 if tex is NULL. */
static inline GLuint mglTraceTextureName(Texture *tex)
{
    return tex ? tex->name : 0u;
}

/* Returns tex->debug_label, or "" if tex is NULL / unlabelled. */
static inline const char *mglTraceTextureLabel(Texture *tex)
{
    return (tex && tex->debug_label[0] != '\0') ? tex->debug_label : "";
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_COMPAT_H */
