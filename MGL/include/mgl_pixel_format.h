/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * mgl_pixel_format.h
 * MGL — Pure pixel format conversion functions (no GLMContext dependency)
 */
#ifndef mgl_pixel_format_h
#define mgl_pixel_format_h

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "glm_params.h"

/* Custom component types for packed float formats (R11F_G11F_B10F).
 * These don't correspond to real GL enums; they're internal markers used
 * only by MGLCPUPixelComponent.type to select the correct float encoding
 * in mglStoreInternalComponent / mglLoadInternalComponent. */
#define MGL_FLOAT11  0x8FF0u
#define MGL_FLOAT10  0x8FF1u

/* GL format/type tables: how many components a pixel-transfer format carries
 * and how many bytes one element of a transfer type occupies.  Pure GL
 * semantics, no Metal knowledge — the backend consumes them.  C linkage:
 * the definitions live in mgl_pixel_format.c and the callers are C++ TUs. */
#ifdef __cplusplus
extern "C" {
#endif
uint32_t mglSizeForType(uint32_t type);
uint32_t mglNumComponentsForFormat(uint32_t format);

/* Decompose an integer internal format into component count / width / signedness.
 * Returns 0 on success, -1 when the format is not an integer colour format. */
int mglRenderIntegerFormatLayout(
    uint32_t internal_format, uint32_t* out_components,
    uint32_t* out_component_bytes, int* out_signed);

/* Readback acceptance: which GL transfer types the fixed-width readback
 * narrowers can decode, and which output slots a GL format's components map
 * to.  Tables only; the byte shuffling stays in the backend. */
int mglReadbackFormatChannelMap(uint32_t format, int* slots, int src_idx[4]);
int mglPixelTypeIsPacked(uint32_t type);
int mglReadbackRGB10A2TypeAccepted(uint32_t type);
int mglReadbackRG11B10TypeAccepted(uint32_t type);
int mglReadback16or32TypeAccepted(uint32_t type);
int mglReadbackUnorm8ScalarTypeAccepted(uint32_t type);
int mglReadbackUnorm8PackedTypeAccepted(uint32_t type);
#ifdef __cplusplus
}
#endif

typedef struct MGLCPUPixelComponent_t {
    GLenum type;
    GLuint bits;
    size_t offset;
    GLuint bit_offset;
    size_t storage_size;
} MGLCPUPixelComponent;

typedef struct MGLCPUPixelLayout_t {
    GLuint component_count;
    size_t pixel_size;
    bool packed;
    MGLCPUPixelComponent components[4];
} MGLCPUPixelLayout;

/* --- Function declarations (moved from textures.c, static removed) --- */
bool mglFindFirstNonZeroByte(const uint8_t *bytes, size_t len, size_t *offset_out, uint8_t *value_out);
GLenum mglTextureComponentSizePname(GLuint component);
GLenum mglTextureComponentTypePname(GLuint component);
bool mglPackedCPUPixelLayoutForInternalFormat(GLenum internalformat, size_t storage_pixel_size, MGLCPUPixelLayout *layout);
bool mglBuildCPUPixelLayout(GLenum internalformat, size_t storage_pixel_size, MGLCPUPixelLayout *layout);
bool mglExternalFormatIsInteger(GLenum format);
bool mglIsValidPixelTransferFormat(GLenum format);
bool mglIsValidPixelTransferType(GLenum type);
bool mglInternalFormatIsInteger(GLint internalformat);
bool mglInternalFormatIsUnsignedInteger(GLint internalformat);
/* Storage class of the RGBA-expanded texture a swizzle bake writes into.  The
 * backend maps these to MTL::PixelFormat; the GL internal formats that select
 * them stay on this side (C2). */
typedef enum MGLPixelStorageClass {
    MGL_PIXEL_STORAGE_INVALID = 0,
    MGL_PIXEL_STORAGE_RGBA8_UNORM,
    MGL_PIXEL_STORAGE_RGBA8_SNORM,
    MGL_PIXEL_STORAGE_RGBA8_UINT,
    MGL_PIXEL_STORAGE_RGBA8_SINT,
    MGL_PIXEL_STORAGE_RGBA16_UNORM,
    MGL_PIXEL_STORAGE_RGBA16_SNORM,
    MGL_PIXEL_STORAGE_RGBA16_UINT,
    MGL_PIXEL_STORAGE_RGBA16_SINT,
    MGL_PIXEL_STORAGE_RGBA16_FLOAT,
    MGL_PIXEL_STORAGE_RGBA32_UINT,
    MGL_PIXEL_STORAGE_RGBA32_SINT,
    MGL_PIXEL_STORAGE_RGBA32_FLOAT
} MGLPixelStorageClass;

/* Single-channel (R) formats whose swizzle is baked into an RGBA target.
 * Inline: the Metal-cpp smoke test links a subset of the objects and still
 * pulls in the backend pixel TU that asks this. */
static inline MGLPixelStorageClass
mglSingleChannelSwizzleStorageClass(uint32_t internal_format)
{
    switch (internal_format) {
        case GL_R8I: return MGL_PIXEL_STORAGE_RGBA8_SINT;
        case GL_R16I: return MGL_PIXEL_STORAGE_RGBA16_SINT;
        case GL_R32I: return MGL_PIXEL_STORAGE_RGBA32_SINT;
        case GL_R8UI: return MGL_PIXEL_STORAGE_RGBA8_UINT;
        case GL_R16UI: return MGL_PIXEL_STORAGE_RGBA16_UINT;
        case GL_R32UI: return MGL_PIXEL_STORAGE_RGBA32_UINT;
        case GL_R8: return MGL_PIXEL_STORAGE_RGBA8_UNORM;
        case GL_R8_SNORM: return MGL_PIXEL_STORAGE_RGBA8_SNORM;
        case GL_R16: return MGL_PIXEL_STORAGE_RGBA16_UNORM;
        case GL_R16_SNORM: return MGL_PIXEL_STORAGE_RGBA16_SNORM;
        case GL_R16F: return MGL_PIXEL_STORAGE_RGBA16_FLOAT;
        case GL_R32F: return MGL_PIXEL_STORAGE_RGBA32_FLOAT;
        case GL_DEPTH_COMPONENT16: return MGL_PIXEL_STORAGE_RGBA16_UNORM;
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8: return MGL_PIXEL_STORAGE_RGBA32_FLOAT;
        default: return MGL_PIXEL_STORAGE_INVALID;
    }
}

/* Integer multi-channel formats (RG/RGB/RGBA of the same size and signedness). */
static inline MGLPixelStorageClass
mglIntegerMultiChannelSwizzleStorageClass(uint32_t internal_format)
{
    switch (internal_format) {
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I: return MGL_PIXEL_STORAGE_RGBA8_SINT;
        case GL_RG16I:
        case GL_RGB16I:
        case GL_RGBA16I: return MGL_PIXEL_STORAGE_RGBA16_SINT;
        case GL_RG32I:
        case GL_RGB32I:
        case GL_RGBA32I: return MGL_PIXEL_STORAGE_RGBA32_SINT;
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI: return MGL_PIXEL_STORAGE_RGBA8_UINT;
        case GL_RG16UI:
        case GL_RGB16UI:
        case GL_RGBA16UI: return MGL_PIXEL_STORAGE_RGBA16_UINT;
        case GL_RG32UI:
        case GL_RGB32UI:
        case GL_RGBA32UI: return MGL_PIXEL_STORAGE_RGBA32_UINT;
        default: return MGL_PIXEL_STORAGE_INVALID;
    }
}
/* The two sized packed depth-stencil formats: the only ones whose storage has
 * both a depth plane and a stencil plane. */
static inline bool mglInternalFormatIsSizedPackedDepthStencil(uint32_t fmt)
{
    return fmt == GL_DEPTH24_STENCIL8 || fmt == GL_DEPTH32F_STENCIL8;
}

/* Which half of a packed depth-stencil texture an upload is addressing. */
static inline bool mglDepthStencilModeIsDepth(uint32_t mode)
{
    return mode == GL_DEPTH_COMPONENT;
}

static inline bool mglDepthStencilModeIsStencil(uint32_t mode)
{
    return mode == GL_STENCIL_INDEX;
}

/* Formats that must be expanded into an RGBA8 storage texture before Metal
 * can sample them (the packed/odd-width RGB families). */
static inline bool mglInternalFormatNeedsRGBA8Expansion(uint32_t fmt)
{
    switch (fmt) {
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_R3_G3_B2:
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB565:
            return true;
        default:
            return false;
    }
}

/* Formats that must be expanded into an RGBA16/32 storage texture. */
static inline bool mglInternalFormatNeedsWideChannelExpansion(uint32_t fmt)
{
    switch (fmt) {
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

/* Packed read types that carry an alpha channel (so a three-slot read still
 * has a meaningful fourth component). */
static inline bool mglReadbackTypeCarriesAlpha(uint32_t type)
{
    switch (type) {
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return true;
        default:
            return false;
    }
}

/* Which resolver a single-channel bake uses for one texel.  One kind per
 * distinct reader, so the upload loop compares kinds instead of GL formats. */
typedef enum MGLUploadResolverKind {
    MGL_UPLOAD_RESOLVER_INTEGER = 0,   /* generic signed/unsigned integer path */
    MGL_UPLOAD_RESOLVER_UNORM8,
    MGL_UPLOAD_RESOLVER_SNORM8,
    MGL_UPLOAD_RESOLVER_UNORM16,
    MGL_UPLOAD_RESOLVER_SNORM16,
    MGL_UPLOAD_RESOLVER_HALF,
    MGL_UPLOAD_RESOLVER_FLOAT32,
    MGL_UPLOAD_RESOLVER_DEPTH16,
    MGL_UPLOAD_RESOLVER_DEPTH24_8,
    MGL_UPLOAD_RESOLVER_DEPTH32F_8
} MGLUploadResolverKind;

static inline MGLUploadResolverKind
mglSingleChannelUploadResolver(uint32_t internal_format)
{
    switch (internal_format) {
        case GL_R8: return MGL_UPLOAD_RESOLVER_UNORM8;
        case GL_R8_SNORM: return MGL_UPLOAD_RESOLVER_SNORM8;
        case GL_R16: return MGL_UPLOAD_RESOLVER_UNORM16;
        case GL_R16_SNORM: return MGL_UPLOAD_RESOLVER_SNORM16;
        case GL_R16F: return MGL_UPLOAD_RESOLVER_HALF;
        case GL_R32F:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return MGL_UPLOAD_RESOLVER_FLOAT32;
        case GL_DEPTH_COMPONENT16: return MGL_UPLOAD_RESOLVER_DEPTH16;
        case GL_DEPTH24_STENCIL8: return MGL_UPLOAD_RESOLVER_DEPTH24_8;
        case GL_DEPTH32F_STENCIL8: return MGL_UPLOAD_RESOLVER_DEPTH32F_8;
        default: return MGL_UPLOAD_RESOLVER_INTEGER;
    }
}

/* Component layout of a single-channel format that is baked into an RGBA
 * storage texture: how wide the source and destination components are and
 * whether they are signed.  `valid` is false for formats without a bake. */
typedef struct MGLSingleChannelUploadLayout {
    uint32_t dst_component_bytes;
    uint32_t src_component_bytes;
    int dst_signed;
    int src_signed;
    bool valid;
} MGLSingleChannelUploadLayout;

static inline MGLSingleChannelUploadLayout
mglSingleChannelSwizzleUploadLayout(uint32_t internal_format)
{
    MGLSingleChannelUploadLayout out = {1u, 1u, 0, 0, true};
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM: break;
        case GL_R8I:
            out.dst_signed = 1;
            out.src_signed = 1;
            break;
        case GL_R8UI: break;
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
            out.dst_component_bytes = 2u;
            out.src_component_bytes = 2u;
            break;
        case GL_R16I:
            out.dst_component_bytes = 2u;
            out.src_component_bytes = 2u;
            out.dst_signed = 1;
            out.src_signed = 1;
            break;
        case GL_R16UI:
            out.dst_component_bytes = 2u;
            out.src_component_bytes = 2u;
            break;
        case GL_R32F:
        case GL_R32I:
            out.dst_component_bytes = 4u;
            out.src_component_bytes = 4u;
            out.dst_signed = (internal_format == GL_R32I) ? 1 : 0;
            out.src_signed = out.dst_signed;
            break;
        case GL_R32UI:
            out.dst_component_bytes = 4u;
            out.src_component_bytes = 4u;
            break;
        case GL_DEPTH_COMPONENT16:
            out.dst_component_bytes = 2u;
            out.src_component_bytes = 2u;
            break;
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
            out.dst_component_bytes = 4u;
            out.src_component_bytes = 4u;
            break;
        case GL_DEPTH32F_STENCIL8:
            out.dst_component_bytes = 4u;
            out.src_component_bytes = 5u;
            break;
        default:
            out.valid = false;
            out.dst_component_bytes = 0u;
            out.src_component_bytes = 0u;
            break;
    }
    return out;
}

/* Packed read-back output layouts, one kind per bit layout. */
typedef enum MGLPackedOutputKind {
    MGL_PACKED_INVALID = 0,
    MGL_PACKED_RGB332,
    MGL_PACKED_BGR233_REV,
    MGL_PACKED_RGB565,
    MGL_PACKED_BGR565_REV,
    MGL_PACKED_RGBA4444,
    MGL_PACKED_ABGR4444_REV,
    MGL_PACKED_RGBA5551,
    MGL_PACKED_ABGR1555_REV,
    MGL_PACKED_RGBA8888,
    MGL_PACKED_ABGR8888_REV,
    MGL_PACKED_RGB10A2,
    MGL_PACKED_A2BGR10_REV,
    MGL_PACKED_R11FG11FB10F_REV,
    MGL_PACKED_RGB9E5_REV
} MGLPackedOutputKind;

static inline MGLPackedOutputKind mglPackedOutputKind(uint32_t type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2: return MGL_PACKED_RGB332;
        case GL_UNSIGNED_BYTE_2_3_3_REV: return MGL_PACKED_BGR233_REV;
        case GL_UNSIGNED_SHORT_5_6_5: return MGL_PACKED_RGB565;
        case GL_UNSIGNED_SHORT_5_6_5_REV: return MGL_PACKED_BGR565_REV;
        case GL_UNSIGNED_SHORT_4_4_4_4: return MGL_PACKED_RGBA4444;
        case GL_UNSIGNED_SHORT_4_4_4_4_REV: return MGL_PACKED_ABGR4444_REV;
        case GL_UNSIGNED_SHORT_5_5_5_1: return MGL_PACKED_RGBA5551;
        case GL_UNSIGNED_SHORT_1_5_5_5_REV: return MGL_PACKED_ABGR1555_REV;
        case GL_UNSIGNED_INT_8_8_8_8: return MGL_PACKED_RGBA8888;
        case GL_UNSIGNED_INT_8_8_8_8_REV: return MGL_PACKED_ABGR8888_REV;
        case GL_UNSIGNED_INT_10_10_10_2: return MGL_PACKED_RGB10A2;
        case GL_UNSIGNED_INT_2_10_10_10_REV: return MGL_PACKED_A2BGR10_REV;
        case GL_UNSIGNED_INT_10F_11F_11F_REV: return MGL_PACKED_R11FG11FB10F_REV;
        case GL_UNSIGNED_INT_5_9_9_9_REV: return MGL_PACKED_RGB9E5_REV;
        default: return MGL_PACKED_INVALID;
    }
}

/* Source bytes per pixel of an RGB-family format that gets expanded into an
 * RGBA8 target (0 when the format does not go through that bake). */
static inline uint32_t mglExpandedSourcePixelBytes(uint32_t fmt)
{
    switch (fmt) {
        case GL_R3_G3_B2: return 1u;
        case GL_RGBA2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB565:
        case GL_RGBA4:
        case GL_RGB5_A1: return 2u;
        case GL_RGB10:
        case GL_RGB12: return 4u;
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI: return 3u;
        default: return 0u;
    }
}

/* Source bytes per pixel of the two sized packed depth-stencil formats. */
static inline uint32_t mglSizedPackedDepthStencilBytes(uint32_t fmt)
{
    if (fmt == GL_DEPTH24_STENCIL8) return 4u;
    if (fmt == GL_DEPTH32F_STENCIL8) return 5u;
    return 0u;
}

static inline bool mglInternalFormatIsRGB12(uint32_t fmt)
{
    return fmt == GL_RGB12;
}

static inline uint32_t mglDepthStencilModeDepthValue(void)
{
    return GL_DEPTH_COMPONENT;
}

bool mglInternalFormatIsDepthStencil(GLint internalformat);

/* GL_BGRA / GL_BGR read-back formats (swizzled channel order). */
static inline bool mglReadFormatIsBGRFamily(uint32_t format)
{
    return format == GL_BGRA || format == GL_BGR;
}

/* GL_RGB as a read-back format (three slots, no alpha). */
static inline bool mglReadFormatIsRGB(uint32_t format)
{
    return format == GL_RGB;
}

/* Has depth samples in storage (depth-only or packed depth-stencil). */
bool mglInternalFormatCarriesDepth(GLint internalformat);
bool mglInternalFormatIsCombinedDepthStencil(GLint internalformat);
int mglExternalSourceIndexForComponent(GLenum format, GLuint component);
/* GL RGB-only formats whose Metal storage pads an alpha channel (e.g.
 * GL_RGB10 → RGB10A2).  External alpha must not be copied; store 1.0. */
int mglInternalFormatPadsMetalAlpha(GLenum internalformat);
double mglClampDouble(double v, double lo, double hi);
double mglUnsignedMaxForBits(GLuint bits);
uint64_t mglReadUnsignedLE(const uint8_t *src, size_t bytes);
void mglWriteUnsignedLE(uint8_t *dst, size_t bytes, uint64_t value);
void mglSwapPixelBytes(uint8_t *pixel, size_t pixel_size, size_t element_size);
double mglSignedMaxForBits(GLuint bits);
double mglReadExternalComponent(const uint8_t *src, GLenum type, int source_index, bool integer_format, GLuint component);
float mglUE11ToFloat(uint32_t v);
float mglUE10ToFloat(uint32_t v);
void mglStoreInternalComponent(uint8_t *dst, const MGLCPUPixelComponent *component, double value);
double mglLoadInternalComponent(const uint8_t *src, const MGLCPUPixelComponent *component);
void mglWriteExternalComponent(uint8_t *dst, GLenum type, int dest_index, bool integer_format, double value);
bool mglIsIdentityPackedFormat(GLenum internalformat, GLenum format, GLenum type);
bool mglIsIdentityUncompressedFormat(GLenum internalformat, GLenum format, GLenum type);
bool mglIsBGRByteSwapFormat(GLenum internalformat, GLenum format, GLenum type);
bool mglClearTexInternalFormatIsColor(GLenum internalformat);
GLenum mglClearTexFormatCompatibilityError(GLenum internalformat, GLenum format);
size_t mglClearComponentSize(GLenum type);
void mglStoreDefaultAlpha(uint8_t *pixel, size_t storage_pixel_size, GLenum type);
bool mglTextureFormatLooksDepthOrStencil(GLenum internalformat);
bool mglTexStorageInternalFormatValid(GLenum internalformat);
GLuint mglCompressedBytesPerRowOf(GLenum internalformat, GLsizei width);
bool mglCompressedBlockInfoOf(GLenum internalformat, GLuint *out_bw, GLuint *out_bh, GLuint *out_bd, GLuint *out_bs);
bool mglIsGenericCompressedFormat(GLenum format);
bool mglCompressedFormatRequiresHeight(GLenum format);
bool mglCopyTex2DFaceForTarget(GLenum target, GLuint *face_out);
/* Bytes per pixel for a GL internal format, defaulting to 4 with a diagnostic
 * for unknown formats (former -[MGLRenderer bytesPerPixelForFormat:]). */
uint32_t mglTextureBytesPerPixelForFormat(GLenum internalformat);

#endif /* mgl_pixel_format_h */
