/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * pixel_utils.h
 * MGL
 *
 */

#ifndef pixel_utils_h
#define pixel_utils_h

#include <os/availability.h>
#include <stddef.h>
#include <stdint.h>
#include "glcorearb.h"

GLuint numComponentsForFormat(GLenum format);

GLboolean validFormat(GLuint format);
GLboolean validFormatType(GLuint format, GLuint type);
GLboolean validInternalFormat(GLint internalformat);

GLuint sizeForType(GLenum type);
GLuint sizeForFormatType(GLenum format, GLenum type);
GLuint sizeForInternalFormat(GLenum internalformat, GLenum format, GLenum type);

/*
 * Number of bytes occupied in memory by a single datum indicated by `type`:
 * the storage size of one element for plain types, or the packed storage size
 * (1/2/4) for the bitfield packed types.  Returns 0 for unknown types.  Used
 * for the PBO offset alignment rule of glTex(Sub)Image*.
 */
size_t mglPixelTypeDatumBytes(GLenum type);

GLuint bicountForFormatType(GLenum format, GLenum type, GLenum component);
GLuint bitcountForInternalFormat(GLenum internalformat, GLenum component);

GLenum internalFormatForGLFormatType(GLenum format, GLenum type);

uint32_t mtlFormatForGLInternalFormat(GLenum internal_format);
uint32_t mtlPixelFormatForGLFormatType(GLenum gl_format, GLenum gl_type);

/* Returns true if `internalformat` is a color-renderable GL internal format
 * (i.e. usable as a color attachment of a framebuffer).  Matches the GL 4.6
 * required color-renderable format list used by the CTS packed_pixels test. */
GLboolean mglIsColorRenderableInternalFormat(GLint internalformat);

float mglHalfToFloat(uint16_t value);
uint16_t mglFloatToHalf(float value);

/* Pack a float into 11-bit unsigned float (UE11) format (6-bit mantissa,
 * 5-bit exponent, bias 15). Used by R11F_G11F_B10F packing. */
uint32_t mglFloatToFloat11(float v);

/* Pack a float into 10-bit unsigned float (UE10) format (5-bit mantissa,
 * 5-bit exponent, bias 15). Used by R11F_G11F_B10F packing. */
uint32_t mglFloatToFloat10(float v);

/* Pack 3 RGB doubles into GL_UNSIGNED_INT_5_9_9_9_REV (GL_RGB9_E5) format.
 * All 3 mantissas share one 5-bit exponent. Implements the shared exponent
 * algorithm from the GL spec. */
uint32_t mglPackRGBToSharedExp(double red, double green, double blue);

/* Unpack a GL_UNSIGNED_INT_5_9_9_9_REV (GL_RGB9_E5) packed value to 3 doubles.
 * Layout: R[0:8], G[9:17], B[18:26], shared_exp[27:31]. */
void mglUnpackSharedExp(uint32_t packed, double *r, double *g, double *b);

/* Pack a UNorm8 value (0-255) into an unsigned float component with the given
 * mantissa bit count (6 for R11F/G11F, 5 for B10F).  Used by R11F_G11F_B10F
 * packing when the source is BGRA8/RGBA8. */
uint32_t mglPackUnsignedFloatFromUNorm8(uint32_t value, uint32_t mantissa_bits);

/* Unpack an unsigned float component (N-bit mantissa + 5-bit exponent, bias 15)
 * into a float.  This is the inverse of mglFloatToFloat11/mglFloatToFloat10
 * and handles Inf/NaN (exponent == 31) correctly. */
float mglUnpackUnsignedFloatComponent(uint32_t value, uint32_t mantissa_bits);


#ifndef API_AVAILABLE
#define API_AVAILABLE(...) __API_AVAILABLE_GET_MACRO(__VA_ARGS__,__API_AVAILABLE7, __API_AVAILABLE6, __API_AVAILABLE5, __API_AVAILABLE4, __API_AVAILABLE3, __API_AVAILABLE2, __API_AVAILABLE1, 0)(__VA_ARGS__)
#endif

#ifndef API_UNAVAILABLE
#define API_UNAVAILABLE(...) __API_UNAVAILABLE_GET_MACRO(__VA_ARGS__,__API_UNAVAILABLE7,__API_UNAVAILABLE6, __API_UNAVAILABLE5, __API_UNAVAILABLE4,__API_UNAVAILABLE3,__API_UNAVAILABLE2,__API_UNAVAILABLE1, 0)(__VA_ARGS__)
#endif


typedef enum MGLPixelFormat_t {
    MGLPixelFormatInvalid = 0,

    /* Normal 8 bit formats */

    MGLPixelFormatA8Unorm      = 1,

    MGLPixelFormatR8Unorm                            = 10,
    MGLPixelFormatR8Unorm_sRGB API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 11,
    MGLPixelFormatR8Snorm      = 12,
    MGLPixelFormatR8Uint       = 13,
    MGLPixelFormatR8Sint       = 14,

    /* Normal 16 bit formats */

    MGLPixelFormatR16Unorm     = 20,
    MGLPixelFormatR16Snorm     = 22,
    MGLPixelFormatR16Uint      = 23,
    MGLPixelFormatR16Sint      = 24,
    MGLPixelFormatR16Float     = 25,

    MGLPixelFormatRG8Unorm                            = 30,
    MGLPixelFormatRG8Unorm_sRGB API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 31,
    MGLPixelFormatRG8Snorm                            = 32,
    MGLPixelFormatRG8Uint                             = 33,
    MGLPixelFormatRG8Sint                             = 34,

    /* Packed 16 bit formats */

    MGLPixelFormatB5G6R5Unorm API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 40,
    MGLPixelFormatA1BGR5Unorm API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 41,
    MGLPixelFormatABGR4Unorm  API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 42,
    MGLPixelFormatBGR5A1Unorm API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 43,

    /* Normal 32 bit formats */

    MGLPixelFormatR32Uint  = 53,
    MGLPixelFormatR32Sint  = 54,
    MGLPixelFormatR32Float = 55,

    MGLPixelFormatRG16Unorm  = 60,
    MGLPixelFormatRG16Snorm  = 62,
    MGLPixelFormatRG16Uint   = 63,
    MGLPixelFormatRG16Sint   = 64,
    MGLPixelFormatRG16Float  = 65,

    MGLPixelFormatRGBA8Unorm      = 70,
    MGLPixelFormatRGBA8Unorm_sRGB = 71,
    MGLPixelFormatRGBA8Snorm      = 72,
    MGLPixelFormatRGBA8Uint       = 73,
    MGLPixelFormatRGBA8Sint       = 74,

    MGLPixelFormatBGRA8Unorm      = 80,
    MGLPixelFormatBGRA8Unorm_sRGB = 81,

    /* Packed 32 bit formats */

    MGLPixelFormatRGB10A2Unorm = 90,
    MGLPixelFormatRGB10A2Uint  = 91,

    MGLPixelFormatRG11B10Float = 92,
    MGLPixelFormatRGB9E5Float = 93,

    MGLPixelFormatBGR10A2Unorm  API_AVAILABLE(macos(10.13), ios(11.0)) = 94,

    MGLPixelFormatBGR10_XR      API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(10.0)) = 554,
    MGLPixelFormatBGR10_XR_sRGB API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(10.0)) = 555,

    /* Normal 64 bit formats */

    MGLPixelFormatRG32Uint  = 103,
    MGLPixelFormatRG32Sint  = 104,
    MGLPixelFormatRG32Float = 105,

    MGLPixelFormatRGBA16Unorm  = 110,
    MGLPixelFormatRGBA16Snorm  = 112,
    MGLPixelFormatRGBA16Uint   = 113,
    MGLPixelFormatRGBA16Sint   = 114,
    MGLPixelFormatRGBA16Float  = 115,

    MGLPixelFormatBGRA10_XR      API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(10.0)) = 552,
    MGLPixelFormatBGRA10_XR_sRGB API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(10.0)) = 553,

    /* Normal 128 bit formats */

    MGLPixelFormatRGBA32Uint  = 123,
    MGLPixelFormatRGBA32Sint  = 124,
    MGLPixelFormatRGBA32Float = 125,

    /* Compressed formats. */

    /* S3TC/DXT */
    MGLPixelFormatBC1_RGBA              API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 130,
    MGLPixelFormatBC1_RGBA_sRGB         API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 131,
    MGLPixelFormatBC2_RGBA              API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 132,
    MGLPixelFormatBC2_RGBA_sRGB         API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 133,
    MGLPixelFormatBC3_RGBA              API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 134,
    MGLPixelFormatBC3_RGBA_sRGB         API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 135,

    /* RGTC */
    MGLPixelFormatBC4_RUnorm            API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 140,
    MGLPixelFormatBC4_RSnorm            API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 141,
    MGLPixelFormatBC5_RGUnorm           API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 142,
    MGLPixelFormatBC5_RGSnorm           API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 143,

    /* BPTC */
    MGLPixelFormatBC6H_RGBFloat         API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 150,
    MGLPixelFormatBC6H_RGBUfloat        API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 151,
    MGLPixelFormatBC7_RGBAUnorm         API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 152,
    MGLPixelFormatBC7_RGBAUnorm_sRGB    API_AVAILABLE(macos(10.11), macCatalyst(13.0), ios(14.0)) = 153,

    /* PVRTC */
    MGLPixelFormatPVRTC_RGB_2BPP        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 160,
    MGLPixelFormatPVRTC_RGB_2BPP_sRGB   API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 161,
    MGLPixelFormatPVRTC_RGB_4BPP        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 162,
    MGLPixelFormatPVRTC_RGB_4BPP_sRGB   API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 163,
    MGLPixelFormatPVRTC_RGBA_2BPP       API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 164,
    MGLPixelFormatPVRTC_RGBA_2BPP_sRGB  API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 165,
    MGLPixelFormatPVRTC_RGBA_4BPP       API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 166,
    MGLPixelFormatPVRTC_RGBA_4BPP_sRGB  API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 167,

    /* ETC2 */
    MGLPixelFormatEAC_R11Unorm          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 170,
    MGLPixelFormatEAC_R11Snorm          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 172,
    MGLPixelFormatEAC_RG11Unorm         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 174,
    MGLPixelFormatEAC_RG11Snorm         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 176,
    MGLPixelFormatEAC_RGBA8             API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 178,
    MGLPixelFormatEAC_RGBA8_sRGB        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 179,

    MGLPixelFormatETC2_RGB8             API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 180,
    MGLPixelFormatETC2_RGB8_sRGB        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 181,
    MGLPixelFormatETC2_RGB8A1           API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 182,
    MGLPixelFormatETC2_RGB8A1_sRGB      API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 183,

    /* ASTC */
    MGLPixelFormatASTC_4x4_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 186,
    MGLPixelFormatASTC_5x4_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 187,
    MGLPixelFormatASTC_5x5_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 188,
    MGLPixelFormatASTC_6x5_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 189,
    MGLPixelFormatASTC_6x6_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 190,
    MGLPixelFormatASTC_8x5_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 192,
    MGLPixelFormatASTC_8x6_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 193,
    MGLPixelFormatASTC_8x8_sRGB         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 194,
    MGLPixelFormatASTC_10x5_sRGB        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 195,
    MGLPixelFormatASTC_10x6_sRGB        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 196,
    MGLPixelFormatASTC_10x8_sRGB        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 197,
    MGLPixelFormatASTC_10x10_sRGB       API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 198,
    MGLPixelFormatASTC_12x10_sRGB       API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 199,
    MGLPixelFormatASTC_12x12_sRGB       API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 200,

    MGLPixelFormatASTC_4x4_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 204,
    MGLPixelFormatASTC_5x4_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 205,
    MGLPixelFormatASTC_5x5_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 206,
    MGLPixelFormatASTC_6x5_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 207,
    MGLPixelFormatASTC_6x6_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 208,
    MGLPixelFormatASTC_8x5_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 210,
    MGLPixelFormatASTC_8x6_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 211,
    MGLPixelFormatASTC_8x8_LDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 212,
    MGLPixelFormatASTC_10x5_LDR         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 213,
    MGLPixelFormatASTC_10x6_LDR         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 214,
    MGLPixelFormatASTC_10x8_LDR         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 215,
    MGLPixelFormatASTC_10x10_LDR        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 216,
    MGLPixelFormatASTC_12x10_LDR        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 217,
    MGLPixelFormatASTC_12x12_LDR        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(8.0)) = 218,


    // ASTC HDR (High Dynamic Range) Formats
    MGLPixelFormatASTC_4x4_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 222,
    MGLPixelFormatASTC_5x4_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 223,
    MGLPixelFormatASTC_5x5_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 224,
    MGLPixelFormatASTC_6x5_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 225,
    MGLPixelFormatASTC_6x6_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 226,
    MGLPixelFormatASTC_8x5_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 228,
    MGLPixelFormatASTC_8x6_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 229,
    MGLPixelFormatASTC_8x8_HDR          API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 230,
    MGLPixelFormatASTC_10x5_HDR         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 231,
    MGLPixelFormatASTC_10x6_HDR         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 232,
    MGLPixelFormatASTC_10x8_HDR         API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 233,
    MGLPixelFormatASTC_10x10_HDR        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 234,
    MGLPixelFormatASTC_12x10_HDR        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 235,
    MGLPixelFormatASTC_12x12_HDR        API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = 236,

    /*!
     @constant MGLPixelFormatGBGR422
     @abstract A pixel format where the red and green channels are subsampled horizontally.  Two pixels are stored in 32 bits, with shared red and blue values, and unique green values.
     @discussion This format is equivalent to YUY2, YUYV, yuvs, or GL_RGB_422_APPLE/GL_UNSIGNED_SHORT_8_8_REV_APPLE.   The component order, from lowest addressed byte to highest, is Y0, Cb, Y1, Cr.  There is no implicit colorspace conversion from YUV to RGB, the shader will receive (Cr, Y, Cb, 1).  422 textures must have a width that is a multiple of 2, and can only be used for 2D non-mipmap textures.  When sampling, ClampToEdge is the only usable wrap mode.
     */
    MGLPixelFormatGBGR422 = 240,

    /*!
     @constant MGLPixelFormatBGRG422
     @abstract A pixel format where the red and green channels are subsampled horizontally.  Two pixels are stored in 32 bits, with shared red and blue values, and unique green values.
     @discussion This format is equivalent to UYVY, 2vuy, or GL_RGB_422_APPLE/GL_UNSIGNED_SHORT_8_8_APPLE. The component order, from lowest addressed byte to highest, is Cb, Y0, Cr, Y1.  There is no implicit colorspace conversion from YUV to RGB, the shader will receive (Cr, Y, Cb, 1).  422 textures must have a width that is a multiple of 2, and can only be used for 2D non-mipmap textures.  When sampling, ClampToEdge is the only usable wrap mode.
     */
    MGLPixelFormatBGRG422 = 241,

    /* Depth */
    MGLPixelFormatDepth16Unorm          API_AVAILABLE(macos(10.12), ios(13.0)) = 250,
    MGLPixelFormatDepth32Float  = 252,

    /* Stencil */
    MGLPixelFormatStencil8        = 253,

    /* Depth Stencil */
    MGLPixelFormatDepth24Unorm_Stencil8  API_AVAILABLE(macos(10.11), macCatalyst(13.0)) API_UNAVAILABLE(ios) = 255,
    MGLPixelFormatDepth32Float_Stencil8  API_AVAILABLE(macos(10.11), ios(9.0)) = 260,

    MGLPixelFormatX32_Stencil8  API_AVAILABLE(macos(10.12), ios(10.0)) = 261,
    MGLPixelFormatX24_Stencil8  API_AVAILABLE(macos(10.12), macCatalyst(13.0)) API_UNAVAILABLE(ios) = 262,

} MGLPixelFormat;

#endif /* pixel_utils_h */
