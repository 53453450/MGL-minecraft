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
bool mglInternalFormatIsDepthStencil(GLint internalformat);

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
