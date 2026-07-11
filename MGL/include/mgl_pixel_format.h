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
bool mglInternalFormatIsDepthStencil(GLint internalformat);
bool mglInternalFormatIsCombinedDepthStencil(GLint internalformat);
int mglExternalSourceIndexForComponent(GLenum format, GLuint component);
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

#endif /* mgl_pixel_format_h */
