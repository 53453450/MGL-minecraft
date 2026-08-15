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
 * mgl_readback.m
 * MGL
 *
 * Pure-C pixel format readback / conversion helpers extracted from
 * MGLRenderer.m. Function bodies are preserved verbatim; only the
 * "static" storage-class qualifier was removed to make the symbols
 * externally visible.
 */

#import <Metal/Metal.h>
#define MGL_NO_MTL_PIXEL_FORMAT
#import "pixel_utils.h"
#import "mgl_readback.h"
#include "mgl_render_cpp.h"
#include <stdint.h>

BOOL mglMetalReadbackFormatIsBGRA8Compatible(MTLPixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatRGBA32Float:
        case MTLPixelFormatR8Unorm:
        case MTLPixelFormatRG8Unorm:
        case MTLPixelFormatR16Unorm:
        case MTLPixelFormatR16Snorm:
        case MTLPixelFormatRG16Unorm:
        case MTLPixelFormatRG16Snorm:
        case MTLPixelFormatRGBA16Unorm:
        case MTLPixelFormatRGBA16Snorm:
        case MTLPixelFormatABGR4Unorm:
        case MTLPixelFormatBGR5A1Unorm:
        case MTLPixelFormatRG11B10Float:
        case MTLPixelFormatR32Float:
        case MTLPixelFormatRG32Float:
        case MTLPixelFormatRG16Float:
        case MTLPixelFormatR16Float:
        case MTLPixelFormatRGBA16Float:
        case MTLPixelFormatBGR10A2Unorm:
        case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatR8Snorm:
        case MTLPixelFormatRG8Snorm:
        case MTLPixelFormatRGBA8Snorm:
        case MTLPixelFormatR8Uint:
        case MTLPixelFormatR8Sint:
        case MTLPixelFormatRG8Uint:
        case MTLPixelFormatRG8Sint:
        case MTLPixelFormatRGBA8Uint:
        case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGB9E5Float:
            return YES;
        default:
            return NO;
    }
}

BOOL mglMetalPixelFormatIsIntegerColor(MTLPixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case MTLPixelFormatR8Uint:
        case MTLPixelFormatR8Sint:
        case MTLPixelFormatR16Uint:
        case MTLPixelFormatR16Sint:
        case MTLPixelFormatR32Uint:
        case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG8Uint:
        case MTLPixelFormatRG8Sint:
        case MTLPixelFormatRG16Uint:
        case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRG32Uint:
        case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA8Uint:
        case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGBA16Uint:
        case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatRGBA32Uint:
        case MTLPixelFormatRGBA32Sint:
        case MTLPixelFormatRGB10A2Uint:
            return YES;
        default:
            return NO;
    }
}

BOOL mglMetalPixelFormatIsSignedIntegerColor(MTLPixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case MTLPixelFormatR8Sint:
        case MTLPixelFormatR16Sint:
        case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG8Sint:
        case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatRGBA32Sint:
            return YES;
        default:
            return NO;
    }
}

NSUInteger mglMetalReadbackBytesPerPixel(MTLPixelFormat pixelFormat)
{
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppReadbackBytesPerPixel, pixel format as its Apple ABI
     * value), shared by both gates. */
    return (NSUInteger)mglRenderCppReadbackBytesPerPixel(
        (uint32_t)pixelFormat);
}

uint8_t mglMetalFloatToUnorm8(float value)
{
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppFloatToUnorm8), shared by both gates. */
    return mglRenderCppFloatToUnorm8(value);
}

float mglMetalSnorm16ToFloat(int16_t value)
{
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppSnorm16ToFloat), shared by both gates. */
    return mglRenderCppSnorm16ToFloat(value);
}

float mglMetalSnorm8ToFloat(int8_t value)
{
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppSnorm8ToFloat), shared by both gates. */
    return mglRenderCppSnorm8ToFloat(value);
}

void mglMetalCopyTextureBytesToBGRA8(const uint8_t *src,
                                            NSUInteger srcBytesPerRow,
                                            uint8_t *dst,
                                            NSUInteger dstBytesPerRow,
                                            NSUInteger width,
                                            NSUInteger height,
                                            MTLPixelFormat pixelFormat,
                                            BOOL flipY)
{
    if (!src || !dst || width == 0u || height == 0u) {
        return;
    }

    BOOL sourceIsRGBA =
        (pixelFormat == MTLPixelFormatRGBA8Unorm ||
         pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
    BOOL sourceIsRGBA32Float = (pixelFormat == MTLPixelFormatRGBA32Float);
    BOOL sourceIsR8 = (pixelFormat == MTLPixelFormatR8Unorm);
    BOOL sourceIsRG8 = (pixelFormat == MTLPixelFormatRG8Unorm);
    BOOL sourceIsR16Unorm = (pixelFormat == MTLPixelFormatR16Unorm);
    BOOL sourceIsRG16Unorm = (pixelFormat == MTLPixelFormatRG16Unorm);
    BOOL sourceIsRGBA16Unorm = (pixelFormat == MTLPixelFormatRGBA16Unorm);
    BOOL sourceIsR16Snorm = (pixelFormat == MTLPixelFormatR16Snorm);
    BOOL sourceIsRG16Snorm = (pixelFormat == MTLPixelFormatRG16Snorm);
    BOOL sourceIsRGBA16Snorm = (pixelFormat == MTLPixelFormatRGBA16Snorm);
    BOOL sourceIsBGR5A1 = (pixelFormat == MTLPixelFormatBGR5A1Unorm);
    BOOL sourceIsABGR4 = (pixelFormat == MTLPixelFormatABGR4Unorm);
    BOOL sourceIsRG11B10Float = (pixelFormat == MTLPixelFormatRG11B10Float);
    BOOL sourceIsR32Float = (pixelFormat == MTLPixelFormatR32Float);
    BOOL sourceIsRG32Float = (pixelFormat == MTLPixelFormatRG32Float);
    BOOL sourceIsRG16Float = (pixelFormat == MTLPixelFormatRG16Float);
    BOOL sourceIsR16Float = (pixelFormat == MTLPixelFormatR16Float);
    BOOL sourceIsRGBA16Float = (pixelFormat == MTLPixelFormatRGBA16Float);
    BOOL sourceIsBGR10A2 = (pixelFormat == MTLPixelFormatBGR10A2Unorm);
    BOOL sourceIsRGB10A2 = (pixelFormat == MTLPixelFormatRGB10A2Unorm);
    BOOL sourceIsR8Snorm = (pixelFormat == MTLPixelFormatR8Snorm);
    BOOL sourceIsRG8Snorm = (pixelFormat == MTLPixelFormatRG8Snorm);
    BOOL sourceIsRGBA8Snorm = (pixelFormat == MTLPixelFormatRGBA8Snorm);
    BOOL sourceIsR8Uint = (pixelFormat == MTLPixelFormatR8Uint);
    BOOL sourceIsR8Sint = (pixelFormat == MTLPixelFormatR8Sint);
    BOOL sourceIsRG8Uint = (pixelFormat == MTLPixelFormatRG8Uint);
    BOOL sourceIsRG8Sint = (pixelFormat == MTLPixelFormatRG8Sint);
    BOOL sourceIsRGBA8Uint = (pixelFormat == MTLPixelFormatRGBA8Uint);
    BOOL sourceIsRGBA8Sint = (pixelFormat == MTLPixelFormatRGBA8Sint);
    BOOL sourceIsRGB9E5 = (pixelFormat == MTLPixelFormatRGB9E5Float);

    for (NSUInteger y = 0; y < height; y++) {
        const uint8_t *srcRow = src + (y * srcBytesPerRow);
        NSUInteger dstY = flipY ? (height - 1u - y) : y;
        uint8_t *dstRow = dst + (dstY * dstBytesPerRow);

        if (!sourceIsRGBA && !sourceIsRGBA32Float && !sourceIsR8 && !sourceIsRG8 &&
            !sourceIsR16Unorm && !sourceIsRG16Unorm && !sourceIsRGBA16Unorm &&
            !sourceIsR16Snorm && !sourceIsRG16Snorm && !sourceIsRGBA16Snorm &&
            !sourceIsBGR5A1 && !sourceIsABGR4 && !sourceIsRG11B10Float &&
            !sourceIsR32Float && !sourceIsRG32Float && !sourceIsRG16Float &&
            !sourceIsR16Float && !sourceIsRGBA16Float && !sourceIsBGR10A2 &&
            !sourceIsRGB10A2 &&
            !sourceIsR8Snorm && !sourceIsRG8Snorm && !sourceIsRGBA8Snorm &&
            !sourceIsR8Uint && !sourceIsR8Sint && !sourceIsRG8Uint && !sourceIsRG8Sint &&
            !sourceIsRGBA8Uint && !sourceIsRGBA8Sint && !sourceIsRGB9E5) {
            memcpy(dstRow, srcRow, width * 4u);
            continue;
        }

        for (NSUInteger x = 0; x < width; x++) {
            uint8_t *d = dstRow + (x * 4u);
            if (sourceIsRGBA32Float) {
                const float *s = (const float *)(const void *)(srcRow + (x * sizeof(float) * 4u));
                d[0] = mglMetalFloatToUnorm8(s[2]);
                d[1] = mglMetalFloatToUnorm8(s[1]);
                d[2] = mglMetalFloatToUnorm8(s[0]);
                d[3] = mglMetalFloatToUnorm8(s[3]);
            } else if (sourceIsRGBA16Float) {
                uint16_t components[4] = {0u, 0u, 0u, 0u};
                memcpy(components, srcRow + x * sizeof(components), sizeof(components));
                d[0] = mglMetalFloatToUnorm8(mglHalfToFloat(components[2]));
                d[1] = mglMetalFloatToUnorm8(mglHalfToFloat(components[1]));
                d[2] = mglMetalFloatToUnorm8(mglHalfToFloat(components[0]));
                d[3] = mglMetalFloatToUnorm8(mglHalfToFloat(components[3]));
            } else if (sourceIsRG11B10Float) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = mglMetalFloatToUnorm8(mglUnpackUnsignedFloatComponent(packed >> 22u, 5u));
                d[1] = mglMetalFloatToUnorm8(mglUnpackUnsignedFloatComponent(packed >> 11u, 6u));
                d[2] = mglMetalFloatToUnorm8(mglUnpackUnsignedFloatComponent(packed, 6u));
                d[3] = 255u;
            } else if (sourceIsRG32Float) {
                const float *s = (const float *)(const void *)(srcRow + (x * sizeof(float) * 2u));
                d[0] = 0u;
                d[1] = mglMetalFloatToUnorm8(s[1]);
                d[2] = mglMetalFloatToUnorm8(s[0]);
                d[3] = 255u;
            } else if (sourceIsR32Float) {
                float component = 0.0f;
                memcpy(&component, srcRow + x * sizeof(component), sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglMetalFloatToUnorm8(component);
                d[3] = 255u;
            } else if (sourceIsRG16Float) {
                uint16_t components[2] = {0u, 0u};
                memcpy(components, srcRow + x * sizeof(components), sizeof(components));
                d[0] = 0u;
                d[1] = mglMetalFloatToUnorm8(mglHalfToFloat(components[1]));
                d[2] = mglMetalFloatToUnorm8(mglHalfToFloat(components[0]));
                d[3] = 255u;
            } else if (sourceIsR16Float) {
                uint16_t component = 0u;
                memcpy(&component, srcRow + x * sizeof(component), sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglMetalFloatToUnorm8(mglHalfToFloat(component));
                d[3] = 255u;
            } else if (sourceIsRGBA16Unorm) {
                uint16_t components[4] = {0u, 0u, 0u, 0u};
                memcpy(components, srcRow + x * sizeof(components), sizeof(components));
                d[0] = (uint8_t)((components[2] * 255u + 32767u) / 65535u);
                d[1] = (uint8_t)((components[1] * 255u + 32767u) / 65535u);
                d[2] = (uint8_t)((components[0] * 255u + 32767u) / 65535u);
                d[3] = (uint8_t)((components[3] * 255u + 32767u) / 65535u);
            } else if (sourceIsRG16Unorm) {
                uint16_t components[2] = {0u, 0u};
                memcpy(components, srcRow + x * sizeof(components), sizeof(components));
                d[0] = 0u;
                d[1] = (uint8_t)((components[1] * 255u + 32767u) / 65535u);
                d[2] = (uint8_t)((components[0] * 255u + 32767u) / 65535u);
                d[3] = 255u;
            } else if (sourceIsR16Unorm) {
                uint16_t component = 0u;
                memcpy(&component, srcRow + x * sizeof(component), sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = (uint8_t)((component * 255u + 32767u) / 65535u);
                d[3] = 255u;
            } else if (sourceIsRGBA16Snorm) {
                int16_t components[4] = {0, 0, 0, 0};
                memcpy(components, srcRow + x * sizeof(components), sizeof(components));
                d[0] = mglMetalFloatToUnorm8(mglMetalSnorm16ToFloat(components[2]));
                d[1] = mglMetalFloatToUnorm8(mglMetalSnorm16ToFloat(components[1]));
                d[2] = mglMetalFloatToUnorm8(mglMetalSnorm16ToFloat(components[0]));
                d[3] = mglMetalFloatToUnorm8(mglMetalSnorm16ToFloat(components[3]));
            } else if (sourceIsRG16Snorm) {
                int16_t components[2] = {0, 0};
                memcpy(components, srcRow + x * sizeof(components), sizeof(components));
                d[0] = 0u;
                d[1] = mglMetalFloatToUnorm8(mglMetalSnorm16ToFloat(components[1]));
                d[2] = mglMetalFloatToUnorm8(mglMetalSnorm16ToFloat(components[0]));
                d[3] = 255u;
            } else if (sourceIsR16Snorm) {
                int16_t component = 0;
                memcpy(&component, srcRow + x * sizeof(component), sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglMetalFloatToUnorm8(mglMetalSnorm16ToFloat(component));
                d[3] = 255u;
            } else if (sourceIsBGR10A2) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)(((packed & 1023u) * 255u) / 1023u);
                d[1] = (uint8_t)((((packed >> 10u) & 1023u) * 255u) / 1023u);
                d[2] = (uint8_t)((((packed >> 20u) & 1023u) * 255u) / 1023u);
                d[3] = (uint8_t)((((packed >> 30u) & 3u) * 255u) / 3u);
            } else if (sourceIsRGB10A2) {
                /* MTLPixelFormatRGB10A2Unorm: R[0:9], G[10:19], B[20:29], A[30:31]
                 * (LSB-first, same as GL_UNSIGNED_INT_2_10_10_10_REV).
                 * Convert to BGRA8: d[0]=B, d[1]=G, d[2]=R, d[3]=A. */
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)((((packed >> 20u) & 1023u) * 255u) / 1023u);
                d[1] = (uint8_t)((((packed >> 10u) & 1023u) * 255u) / 1023u);
                d[2] = (uint8_t)(((packed & 1023u) * 255u) / 1023u);
                d[3] = (uint8_t)((((packed >> 30u) & 3u) * 255u) / 3u);
            } else if (sourceIsR8) {
                d[0] = 0u;
                d[1] = 0u;
                d[2] = srcRow[x];
                d[3] = 255u;
            } else if (sourceIsRG8) {
                const uint8_t *s = srcRow + x * 2u;
                d[0] = 0u;
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255u;
            } else if (sourceIsBGR5A1) {
                /* MTLPixelFormatBGR5A1Unorm: B[0:4], G[5:9], R[10:14], A[15].
                 * Output BGRA8: d[0]=B, d[1]=G, d[2]=R, d[3]=A. */
                uint16_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)(((packed & 31u) * 255u) / 31u);
                d[1] = (uint8_t)((((packed >> 5u) & 31u) * 255u) / 31u);
                d[2] = (uint8_t)((((packed >> 10u) & 31u) * 255u) / 31u);
                d[3] = ((packed >> 15u) & 1u) ? 255u : 0u;
            } else if (sourceIsABGR4) {
                /* MTLPixelFormatABGR4Unorm: A[0:3], B[4:7], G[8:11], R[12:15].
                 * Output BGRA8: d[0]=B, d[1]=G, d[2]=R, d[3]=A. */
                uint16_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)((((packed >> 4u) & 15u) * 255u) / 15u);
                d[1] = (uint8_t)((((packed >> 8u) & 15u) * 255u) / 15u);
                d[2] = (uint8_t)((((packed >> 12u) & 15u) * 255u) / 15u);
                d[3] = (uint8_t)(((packed & 15u) * 255u) / 15u);
            } else if (sourceIsR8Snorm || sourceIsR8Sint) {
                int8_t s = (int8_t)srcRow[x];
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglMetalFloatToUnorm8(mglMetalSnorm8ToFloat(s));
                d[3] = 255u;
            } else if (sourceIsRG8Snorm || sourceIsRG8Sint) {
                const int8_t *s = (const int8_t *)(srcRow + x * 2u);
                d[0] = 0u;
                d[1] = mglMetalFloatToUnorm8(mglMetalSnorm8ToFloat(s[1]));
                d[2] = mglMetalFloatToUnorm8(mglMetalSnorm8ToFloat(s[0]));
                d[3] = 255u;
            } else if (sourceIsRGBA8Snorm || sourceIsRGBA8Sint) {
                const int8_t *s = (const int8_t *)(srcRow + x * 4u);
                d[0] = mglMetalFloatToUnorm8(mglMetalSnorm8ToFloat(s[2]));
                d[1] = mglMetalFloatToUnorm8(mglMetalSnorm8ToFloat(s[1]));
                d[2] = mglMetalFloatToUnorm8(mglMetalSnorm8ToFloat(s[0]));
                d[3] = mglMetalFloatToUnorm8(mglMetalSnorm8ToFloat(s[3]));
            } else if (sourceIsR8Uint) {
                d[0] = 0u;
                d[1] = 0u;
                d[2] = srcRow[x];
                d[3] = 255u;
            } else if (sourceIsRG8Uint) {
                const uint8_t *s = srcRow + x * 2u;
                d[0] = 0u;
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255u;
            } else if (sourceIsRGBA8Uint) {
                const uint8_t *s = srcRow + x * 4u;
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            } else if (sourceIsRGB9E5) {
                /* MTLPixelFormatRGB9E5Float: 4 bytes/pixel, shared exponent.
                 * Unpack to float R,G,B then convert to BGRA8 UNORM. */
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * 4u, sizeof(packed));
                uint32_t exp = (packed >> 27u) & 31u;
                uint32_t mant_r = packed & 511u;
                uint32_t mant_g = (packed >> 9u) & 511u;
                uint32_t mant_b = (packed >> 18u) & 511u;
                float scale = ldexpf(1.0f, (int)exp - 24);
                float rf = (float)mant_r * scale;
                float gf = (float)mant_g * scale;
                float bf = (float)mant_b * scale;
                d[0] = mglMetalFloatToUnorm8(bf);
                d[1] = mglMetalFloatToUnorm8(gf);
                d[2] = mglMetalFloatToUnorm8(rf);
                d[3] = 255u;
            } else {
                const uint8_t *s = srcRow + (x * 4u);
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        }
    }
}

BOOL mglMetalCopyBGRA8CompatibleTextureBytesToGL(const uint8_t *src,
                                                        NSUInteger srcBytesPerRow,
                                                        uint8_t *dst,
                                                        NSUInteger dstBytesPerRow,
                                                        NSUInteger width,
                                                        NSUInteger height,
                                                        MTLPixelFormat pixelFormat,
                                                        GLenum format,
                                                        GLenum type,
                                                        BOOL flipY)
{
    if (!src || !dst || width == 0u || height == 0u) {
        return NO;
    }

    if (type != GL_UNSIGNED_BYTE &&
        type != GL_UNSIGNED_INT_8_8_8_8 &&
        type != GL_UNSIGNED_INT_8_8_8_8_REV &&
        type != GL_FLOAT &&
        type != GL_BYTE &&
        type != GL_SHORT &&
        type != GL_INT &&
        type != GL_UNSIGNED_INT &&
        type != GL_UNSIGNED_SHORT &&
        type != GL_HALF_FLOAT &&
        type != GL_UNSIGNED_BYTE_3_3_2 &&
        type != GL_UNSIGNED_BYTE_2_3_3_REV &&
        type != GL_UNSIGNED_SHORT_5_6_5 &&
        type != GL_UNSIGNED_SHORT_5_6_5_REV &&
        type != GL_UNSIGNED_SHORT_4_4_4_4 &&
        type != GL_UNSIGNED_SHORT_4_4_4_4_REV &&
        type != GL_UNSIGNED_SHORT_5_5_5_1 &&
        type != GL_UNSIGNED_SHORT_1_5_5_5_REV &&
        type != GL_UNSIGNED_INT_10_10_10_2 &&
        type != GL_UNSIGNED_INT_2_10_10_10_REV &&
        type != GL_UNSIGNED_INT_10F_11F_11F_REV &&
        type != GL_UNSIGNED_INT_5_9_9_9_REV) {
        return NO;
    }

    /* Direct SNORM conversion path: bypass the lossy BGRA8 UNORM intermediate.
     * SNORM int8_t -> BGRA8 UNORM loses sign information, so we convert directly
     * from the native SNORM texture data to the requested GL format/type. */
    BOOL sourceIsSnorm8 =
        (pixelFormat == MTLPixelFormatR8Snorm ||
         pixelFormat == MTLPixelFormatRG8Snorm ||
         pixelFormat == MTLPixelFormatRGBA8Snorm);
    if (sourceIsSnorm8) {
        NSUInteger srcBpp = mglMetalReadbackBytesPerPixel(pixelFormat);
        NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
        if (dstPixelBytes == 0u || dstBytesPerRow < width * dstPixelBytes) {
            return NO;
        }
        int srcChannels = (int)(srcBpp); /* 1 for R8, 2 for RG8, 4 for RGBA8 */
        int slots = 0;
        int srcIdx[4] = {0,0,0,0};
        switch (format) {
            case GL_RGBA: slots = 4; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; srcIdx[3]=3; break;
            case GL_BGRA: slots = 4; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; srcIdx[3]=3; break;
            case GL_RGB:  slots = 3; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; break;
            case GL_BGR:  slots = 3; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; break;
            case GL_RG:   slots = 2; srcIdx[0]=0; srcIdx[1]=1; break;
            case GL_RED:  slots = 1; srcIdx[0]=0; break;
            case GL_GREEN: slots = 1; srcIdx[0]=1; break;
            case GL_BLUE:  slots = 1; srcIdx[0]=2; break;
            case GL_ALPHA: slots = 1; srcIdx[0]=3; break;
            default: return NO;
        }
        NSUInteger compBytes = (NSUInteger)sizeForType(type);
        for (NSUInteger y = 0; y < height; y++) {
            const uint8_t *srcRow = src + (y * srcBytesPerRow);
            NSUInteger dstY = flipY ? (height - 1u - y) : y;
            uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
            for (NSUInteger x = 0; x < width; x++) {
                const int8_t *s = (const int8_t *)(srcRow + (x * srcBpp));
                uint8_t *dp = dstRow + (x * dstPixelBytes);
                for (int c = 0; c < slots; ++c) {
                    int idx = srcIdx[c];
                    if (idx >= srcChannels) idx = srcChannels - 1;
                    int8_t sv = s[idx];
                    float fv = mglMetalSnorm8ToFloat(sv);
                    uint8_t *out = dp + (NSUInteger)c * compBytes;
                    if (type == GL_BYTE) {
                        int32_t iv = (int32_t)lroundf(fv * 127.0f);
                        if (iv > 127) iv = 127;
                        if (iv < -128) iv = -128;
                        int8_t biv = (int8_t)iv;
                        memcpy(out, &biv, sizeof(biv));
                    } else if (type == GL_UNSIGNED_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_FLOAT) {
                        memcpy(out, &fv, sizeof(fv));
                    } else if (type == GL_HALF_FLOAT) {
                        uint16_t iv = mglFloatToHalf(fv);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_SHORT) {
                        int32_t iv = (int32_t)lroundf(fv * 32767.0f);
                        if (iv > 32767) iv = 32767;
                        if (iv < -32768) iv = -32768;
                        int16_t siv = (int16_t)iv;
                        memcpy(out, &siv, sizeof(siv));
                    } else if (type == GL_UNSIGNED_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_INT) {
                        int64_t iv = (int64_t)llroundf(fv * 2147483647.0f);
                        if (iv > 2147483647LL) iv = 2147483647LL;
                        if (iv < -2147483648LL) iv = -2147483648LL;
                        int32_t iiv = (int32_t)iv;
                        memcpy(out, &iiv, sizeof(iiv));
                    } else if (type == GL_UNSIGNED_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                        memcpy(out, &iv, sizeof(iv));
                    }
                }
            }
        }
        return YES;
    }

    /* Direct RGB10A2 conversion path: bypass the lossy BGRA8 UNORM intermediate.
     * RGB10A2 has 10-bit color channels; going through BGRA8 (8-bit) loses ~2 bits
     * of precision, causing CTS gradient comparison failures. Convert directly
     * from the native 10-bit packed data to the requested GL format/type. */
    BOOL sourceIsRGB10A2Direct = (pixelFormat == MTLPixelFormatRGB10A2Unorm);
    if (sourceIsRGB10A2Direct &&
        (type == GL_UNSIGNED_BYTE || type == GL_BYTE ||
         type == GL_UNSIGNED_SHORT || type == GL_SHORT ||
         type == GL_UNSIGNED_INT || type == GL_INT ||
         type == GL_FLOAT || type == GL_HALF_FLOAT ||
         type == GL_UNSIGNED_INT_10_10_10_2 ||
         type == GL_UNSIGNED_INT_2_10_10_10_REV ||
         type == GL_UNSIGNED_INT_5_9_9_9_REV ||
         type == GL_UNSIGNED_INT_8_8_8_8 ||
         type == GL_UNSIGNED_INT_8_8_8_8_REV))
    {
        NSUInteger srcBpp = 4u; /* RGB10A2 is 4 bytes per pixel */
        NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
        if (dstPixelBytes == 0u || dstBytesPerRow < width * dstPixelBytes) {
            return NO;
        }

        /* Determine output channel mapping */
        int slots = 0;
        int srcIdx[4] = {0,0,0,0}; /* indices into R,G,B,A (0=R,1=G,2=B,3=A) */
        switch (format) {
            case GL_RGBA: slots = 4; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; srcIdx[3]=3; break;
            case GL_BGRA: slots = 4; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; srcIdx[3]=3; break;
            case GL_RGB:  slots = 3; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; break;
            case GL_BGR:  slots = 3; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; break;
            case GL_RG:   slots = 2; srcIdx[0]=0; srcIdx[1]=1; break;
            case GL_RED:  slots = 1; srcIdx[0]=0; break;
            case GL_GREEN: slots = 1; srcIdx[0]=1; break;
            case GL_BLUE:  slots = 1; srcIdx[0]=2; break;
            case GL_ALPHA: slots = 1; srcIdx[0]=3; break;
            default: return NO;
        }

        NSUInteger compBytes = (NSUInteger)sizeForType(type);

        for (NSUInteger y = 0; y < height; y++) {
            const uint8_t *srcRow = src + (y * srcBytesPerRow);
            NSUInteger dstY = flipY ? (height - 1u - y) : y;
            uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
            for (NSUInteger x = 0; x < width; x++) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + (x * srcBpp), sizeof(packed));
                /* Extract 10-bit/2-bit unorm values from LSB-first layout */
                uint32_t rgb10a2_vals[4] = {
                    packed & 1023u,           /* R: bits 0-9 */
                    (packed >> 10u) & 1023u,  /* G: bits 10-19 */
                    (packed >> 20u) & 1023u,  /* B: bits 20-29 */
                    (packed >> 30u) & 3u      /* A: bits 30-31 */
                };

                if (type == GL_UNSIGNED_INT_10_10_10_2) {
                    /* MSB-first: R[22:31], G[12:21], B[2:11], A[0:1] */
                    uint32_t r10 = rgb10a2_vals[srcIdx[0]];
                    uint32_t g10 = (slots > 1) ? rgb10a2_vals[srcIdx[1]] : 0u;
                    uint32_t b10 = (slots > 2) ? rgb10a2_vals[srcIdx[2]] : 0u;
                    uint32_t a2 = (slots > 3) ? rgb10a2_vals[srcIdx[3]] : 0u;
                    uint32_t out = (r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2;
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                    /* LSB-first: same layout as source, just remap channels */
                    uint32_t r10 = rgb10a2_vals[srcIdx[0]];
                    uint32_t g10 = (slots > 1) ? rgb10a2_vals[srcIdx[1]] : 0u;
                    uint32_t b10 = (slots > 2) ? rgb10a2_vals[srcIdx[2]] : 0u;
                    uint32_t a2 = (slots > 3) ? rgb10a2_vals[srcIdx[3]] : 0u;
                    uint32_t out = r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u);
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                    /* Pack 10-bit unorm channels to shared-exponent RGB9E5
                     * directly from 10-bit values to avoid 8-bit precision loss. */
                    float rf = (float)rgb10a2_vals[srcIdx[0]] / 1023.0f;
                    float gf = (slots > 1) ? (float)rgb10a2_vals[srcIdx[1]] / 1023.0f : 0.0f;
                    float bf = (slots > 2) ? (float)rgb10a2_vals[srcIdx[2]] / 1023.0f : 0.0f;
                    uint32_t out = mglPackRGBToSharedExp(rf, gf, bf);
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                    /* CTS: R=(val>>24), G=(val>>16), B=(val>>8), A=(val>>0).
                     * Convert 10-bit to 8-bit directly for best precision. */
                    uint8_t r8 = (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[0]] * 255u / 1023u);
                    uint8_t g8 = (slots > 1) ? (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[1]] * 255u / 1023u) : 0u;
                    uint8_t b8 = (slots > 2) ? (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[2]] * 255u / 1023u) : 0u;
                    uint8_t a8 = (slots > 3) ? (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[3]] * 255u / 3u) : 0u;
                    uint32_t out = ((uint32_t)r8 << 24u) | ((uint32_t)g8 << 16u) | ((uint32_t)b8 << 8u) | a8;
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                    /* CTS: R=(val>>0), G=(val>>8), B=(val>>16), A=(val>>24). */
                    uint8_t r8 = (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[0]] * 255u / 1023u);
                    uint8_t g8 = (slots > 1) ? (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[1]] * 255u / 1023u) : 0u;
                    uint8_t b8 = (slots > 2) ? (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[2]] * 255u / 1023u) : 0u;
                    uint8_t a8 = (slots > 3) ? (uint8_t)((uint64_t)rgb10a2_vals[srcIdx[3]] * 255u / 3u) : 0u;
                    uint32_t out = r8 | ((uint32_t)g8 << 8u) | ((uint32_t)b8 << 16u) | ((uint32_t)a8 << 24u);
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else {
                    /* Non-packed types: convert via float to preserve precision */
                    for (int c = 0; c < slots; ++c) {
                        uint32_t raw = rgb10a2_vals[srcIdx[c]];
                        float fv = (srcIdx[c] == 3) ? (float)raw / 3.0f : (float)raw / 1023.0f;
                        uint8_t *out = dstRow + (x * dstPixelBytes) + (NSUInteger)c * compBytes;
                        if (type == GL_UNSIGNED_BYTE) {
                            float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                            uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_BYTE) {
                            float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                            int8_t iv = (int8_t)lroundf(cv * 127.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_UNSIGNED_SHORT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                            uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_SHORT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                            int16_t iv = (int16_t)lroundf(cv * 32767.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_UNSIGNED_INT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                            uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_INT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                            int32_t iv = (int32_t)llroundf(cv * 2147483647.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_FLOAT) {
                            memcpy(out, &fv, sizeof(fv));
                        } else { /* GL_HALF_FLOAT */
                            uint16_t iv = mglFloatToHalf(fv);
                            memcpy(out, &iv, sizeof(iv));
                        }
                    }
                }
            }
        }
        return YES;
    }

    /* Direct RG11B10Float conversion path: bypass the lossy BGRA8 UNORM intermediate.
     * RG11B10Float has float channels; going through BGRA8 (8-bit unorm) loses
     * precision and changes bit patterns, causing CTS copy_image failures.
     * Convert directly from the native packed float data to the requested GL format/type. */
    BOOL sourceIsRG11B10FloatDirect = (pixelFormat == MTLPixelFormatRG11B10Float);
    if (sourceIsRG11B10FloatDirect &&
        (type == GL_UNSIGNED_BYTE || type == GL_BYTE ||
         type == GL_UNSIGNED_SHORT || type == GL_SHORT ||
         type == GL_UNSIGNED_INT || type == GL_INT ||
         type == GL_FLOAT || type == GL_HALF_FLOAT ||
         type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
         type == GL_UNSIGNED_INT_5_9_9_9_REV ||
         type == GL_UNSIGNED_INT_8_8_8_8 ||
         type == GL_UNSIGNED_INT_8_8_8_8_REV))
    {
        NSUInteger srcBpp = 4u; /* RG11B10Float is 4 bytes per pixel */
        NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
        if (dstPixelBytes == 0u || dstBytesPerRow < width * dstPixelBytes) {
            return NO;
        }

        /* For GL_UNSIGNED_INT_10F_11F_11F_REV with GL_RGB, Metal's RG11B10Float
         * uses the exact same LSB-first bit layout (R[0:10], G[11:21], B[22:31]).
         * Raw memcpy preserves the exact bits. */
        if (type == GL_UNSIGNED_INT_10F_11F_11F_REV && format == GL_RGB) {
            for (NSUInteger y = 0; y < height; y++) {
                const uint8_t *srcRow = src + (y * srcBytesPerRow);
                NSUInteger dstY = flipY ? (height - 1u - y) : y;
                uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
                memcpy(dstRow, srcRow, width * srcBpp);
            }
            return YES;
        }

        /* Determine output channel mapping */
        int slots = 0;
        int srcIdx[4] = {0,0,0,0}; /* indices into R,G,B,A (0=R,1=G,2=B,3=A) */
        switch (format) {
            case GL_RGBA: slots = 4; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; srcIdx[3]=3; break;
            case GL_BGRA: slots = 4; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; srcIdx[3]=3; break;
            case GL_RGB:  slots = 3; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; break;
            case GL_BGR:  slots = 3; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; break;
            case GL_RG:   slots = 2; srcIdx[0]=0; srcIdx[1]=1; break;
            case GL_RED:  slots = 1; srcIdx[0]=0; break;
            case GL_GREEN: slots = 1; srcIdx[0]=1; break;
            case GL_BLUE:  slots = 1; srcIdx[0]=2; break;
            case GL_ALPHA: slots = 1; srcIdx[0]=3; break;
            default: return NO;
        }

        NSUInteger compBytes = (NSUInteger)sizeForType(type);

        for (NSUInteger y = 0; y < height; y++) {
            const uint8_t *srcRow = src + (y * srcBytesPerRow);
            NSUInteger dstY = flipY ? (height - 1u - y) : y;
            uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
            for (NSUInteger x = 0; x < width; x++) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + (x * srcBpp), sizeof(packed));
                /* Decode float channels from RG11B10Float packed format.
                 * R: bits 0-10 (11-bit float, 6-bit mantissa)
                 * G: bits 11-21 (11-bit float, 6-bit mantissa)
                 * B: bits 22-31 (10-bit float, 5-bit mantissa) */
                float float_vals[4] = {
                    mglUnpackUnsignedFloatComponent(packed, 6u),         /* R */
                    mglUnpackUnsignedFloatComponent(packed >> 11u, 6u),  /* G */
                    mglUnpackUnsignedFloatComponent(packed >> 22u, 5u),  /* B */
                    1.0f                                                /* A (always 1) */
                };

                if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                    /* Pack as R11G11B10 float with remapped channels.
                     * Layout: R[0:10], G[11:21], B[22:31] */
                    float r = float_vals[srcIdx[0]];
                    float g = (slots > 1) ? float_vals[srcIdx[1]] : 0.0f;
                    float b = (slots > 2) ? float_vals[srcIdx[2]] : 0.0f;
                    uint32_t out = (mglFloatToFloat11(r) & 0x7ffu) |
                                   ((mglFloatToFloat11(g) & 0x7ffu) << 11u) |
                                   ((mglFloatToFloat10(b) & 0x3ffu) << 22u);
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                    float r = float_vals[srcIdx[0]];
                    float g = (slots > 1) ? float_vals[srcIdx[1]] : 0.0f;
                    float b = (slots > 2) ? float_vals[srcIdx[2]] : 0.0f;
                    uint32_t out = mglPackRGBToSharedExp(r, g, b);
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                    uint8_t r8 = mglMetalFloatToUnorm8(float_vals[srcIdx[0]]);
                    uint8_t g8 = (slots > 1) ? mglMetalFloatToUnorm8(float_vals[srcIdx[1]]) : 0u;
                    uint8_t b8 = (slots > 2) ? mglMetalFloatToUnorm8(float_vals[srcIdx[2]]) : 0u;
                    uint8_t a8 = (slots > 3) ? mglMetalFloatToUnorm8(float_vals[srcIdx[3]]) : 0u;
                    uint32_t out = ((uint32_t)r8 << 24u) | ((uint32_t)g8 << 16u) | ((uint32_t)b8 << 8u) | a8;
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                    uint8_t r8 = mglMetalFloatToUnorm8(float_vals[srcIdx[0]]);
                    uint8_t g8 = (slots > 1) ? mglMetalFloatToUnorm8(float_vals[srcIdx[1]]) : 0u;
                    uint8_t b8 = (slots > 2) ? mglMetalFloatToUnorm8(float_vals[srcIdx[2]]) : 0u;
                    uint8_t a8 = (slots > 3) ? mglMetalFloatToUnorm8(float_vals[srcIdx[3]]) : 0u;
                    uint32_t out = r8 | ((uint32_t)g8 << 8u) | ((uint32_t)b8 << 16u) | ((uint32_t)a8 << 24u);
                    memcpy(dstRow + (x * dstPixelBytes), &out, sizeof(out));
                } else {
                    /* Non-packed types: convert via float */
                    for (int c = 0; c < slots; ++c) {
                        float fv = float_vals[srcIdx[c]];
                        uint8_t *out = dstRow + (x * dstPixelBytes) + (NSUInteger)c * compBytes;
                        if (type == GL_UNSIGNED_BYTE) {
                            float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                            uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_BYTE) {
                            float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                            int8_t iv = (int8_t)lroundf(cv * 127.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_UNSIGNED_SHORT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                            uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_SHORT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                            int16_t iv = (int16_t)lroundf(cv * 32767.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_UNSIGNED_INT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                            uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_INT) {
                            float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                            int32_t iv = (int32_t)llroundf(cv * 2147483647.0f);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_FLOAT) {
                            memcpy(out, &fv, sizeof(fv));
                        } else { /* GL_HALF_FLOAT */
                            uint16_t iv = mglFloatToHalf(fv);
                            memcpy(out, &iv, sizeof(iv));
                        }
                    }
                }
            }
        }
        return YES;
    }

    /* Direct 16-bit/32-bit conversion path: bypass the lossy BGRA8 UNORM intermediate.
     * R16/RG16/RGBA16 Unorm/Snorm/Float and R32/RG32/RGBA32 Float -> BGRA8 UNORM
     * loses precision, so we convert directly from the native texture data to the
     * requested GL format/type. */
    BOOL sourceIs16BitUnorm =
        (pixelFormat == MTLPixelFormatR16Unorm ||
         pixelFormat == MTLPixelFormatRG16Unorm ||
         pixelFormat == MTLPixelFormatRGBA16Unorm);
    BOOL sourceIs16BitSnorm =
        (pixelFormat == MTLPixelFormatR16Snorm ||
         pixelFormat == MTLPixelFormatRG16Snorm ||
         pixelFormat == MTLPixelFormatRGBA16Snorm);
    BOOL sourceIs16BitFloat =
        (pixelFormat == MTLPixelFormatR16Float ||
         pixelFormat == MTLPixelFormatRG16Float ||
         pixelFormat == MTLPixelFormatRGBA16Float);
    BOOL sourceIs32BitFloat =
        (pixelFormat == MTLPixelFormatR32Float ||
         pixelFormat == MTLPixelFormatRG32Float ||
         pixelFormat == MTLPixelFormatRGBA32Float);

    if ((sourceIs16BitUnorm || sourceIs16BitSnorm || sourceIs16BitFloat || sourceIs32BitFloat) &&
        (type == GL_UNSIGNED_BYTE || type == GL_BYTE ||
         type == GL_UNSIGNED_SHORT || type == GL_SHORT ||
         type == GL_UNSIGNED_INT || type == GL_INT ||
         type == GL_FLOAT || type == GL_HALF_FLOAT ||
         type == GL_UNSIGNED_BYTE_3_3_2 || type == GL_UNSIGNED_BYTE_2_3_3_REV ||
         type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV ||
         type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
         type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
         type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_INT_8_8_8_8_REV ||
         type == GL_UNSIGNED_INT_10_10_10_2 || type == GL_UNSIGNED_INT_2_10_10_10_REV ||
         type == GL_UNSIGNED_INT_10F_11F_11F_REV || type == GL_UNSIGNED_INT_5_9_9_9_REV))
    {
        NSUInteger srcBpp = mglMetalReadbackBytesPerPixel(pixelFormat);
        NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
        if (dstPixelBytes == 0u || dstBytesPerRow < width * dstPixelBytes) {
            return NO;
        }

        /* Determine source channels: 1 for R16, 2 for RG16, 4 for RGBA16 */
        int srcChannels = 0;
        if (sourceIs32BitFloat) {
            switch (pixelFormat) {
                case MTLPixelFormatR32Float: srcChannels = 1; break;
                case MTLPixelFormatRG32Float: srcChannels = 2; break;
                case MTLPixelFormatRGBA32Float: srcChannels = 4; break;
                default: break;
            }
        } else {
            switch (pixelFormat) {
                case MTLPixelFormatR16Unorm:
                case MTLPixelFormatR16Snorm:
                case MTLPixelFormatR16Float:
                    srcChannels = 1; break;
                case MTLPixelFormatRG16Unorm:
                case MTLPixelFormatRG16Snorm:
                case MTLPixelFormatRG16Float:
                    srcChannels = 2; break;
                case MTLPixelFormatRGBA16Unorm:
                case MTLPixelFormatRGBA16Snorm:
                case MTLPixelFormatRGBA16Float:
                    srcChannels = 4; break;
                default: break;
            }
        }
        if (srcChannels == 0) return NO;

        /* Map output format to source channel indices */
        int slots = 0;
        int srcIdx[4] = {0,0,0,0};
        switch (format) {
            case GL_RGBA: slots = 4; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; srcIdx[3]=3; break;
            case GL_BGRA: slots = 4; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; srcIdx[3]=3; break;
            case GL_RGB:  slots = 3; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; break;
            case GL_BGR:  slots = 3; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; break;
            case GL_RG:   slots = 2; srcIdx[0]=0; srcIdx[1]=1; break;
            case GL_RED:  slots = 1; srcIdx[0]=0; break;
            case GL_GREEN: slots = 1; srcIdx[0]=1; break;
            case GL_BLUE:  slots = 1; srcIdx[0]=2; break;
            case GL_ALPHA: slots = 1; srcIdx[0]=3; break;
            default: return NO;
        }

        NSUInteger compBytes = (NSUInteger)sizeForType(type);

        for (NSUInteger y = 0; y < height; y++) {
            const uint8_t *srcRow = src + (y * srcBytesPerRow);
            NSUInteger dstY = flipY ? (height - 1u - y) : y;
            uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
            for (NSUInteger x = 0; x < width; x++) {
                const uint8_t *s = srcRow + (x * srcBpp);
                uint8_t *dp = dstRow + (x * dstPixelBytes);

                /* Packed output types: extract all source components as float,
                 * then pack into the output word. This preserves precision
                 * compared to going through a BGRA8 intermediate. */
                BOOL outputIsPackedType =
                    (type == GL_UNSIGNED_BYTE_3_3_2 || type == GL_UNSIGNED_BYTE_2_3_3_REV ||
                     type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV ||
                     type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
                     type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
                     type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_INT_8_8_8_8_REV ||
                     type == GL_UNSIGNED_INT_10_10_10_2 || type == GL_UNSIGNED_INT_2_10_10_10_REV ||
                     type == GL_UNSIGNED_INT_10F_11F_11F_REV || type == GL_UNSIGNED_INT_5_9_9_9_REV);
                if (outputIsPackedType) {
                    /* Extract up to 4 source components as float values. */
                    float fvals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int c = 0; c < slots; ++c) {
                        int idx = srcIdx[c];
                        if (idx >= srcChannels) idx = srcChannels - 1;
                        if (sourceIs16BitUnorm) {
                            uint16_t uv;
                            memcpy(&uv, s + (NSUInteger)idx * 2u, sizeof(uv));
                            fvals[c] = (float)uv / 65535.0f;
                        } else if (sourceIs16BitSnorm) {
                            int16_t sv;
                            memcpy(&sv, s + (NSUInteger)idx * 2u, sizeof(sv));
                            fvals[c] = (float)sv / 32767.0f;
                        } else if (sourceIs16BitFloat) {
                            uint16_t hv;
                            memcpy(&hv, s + (NSUInteger)idx * 2u, sizeof(hv));
                            fvals[c] = mglHalfToFloat(hv);
                        } else {
                            memcpy(&fvals[c], s + (NSUInteger)idx * 4u, sizeof(float));
                        }
                    }
                    /* For formats with < 4 components, pad with defaults. */
                    /* R, G, B default to 0; A defaults to 1 for RGBA outputs. */
                    if (slots < 4) {
                        BOOL needsAlpha = (type == GL_UNSIGNED_SHORT_4_4_4_4 ||
                            type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
                            type == GL_UNSIGNED_SHORT_5_5_5_1 ||
                            type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
                            type == GL_UNSIGNED_INT_8_8_8_8 ||
                            type == GL_UNSIGNED_INT_8_8_8_8_REV ||
                            type == GL_UNSIGNED_INT_10_10_10_2 ||
                            type == GL_UNSIGNED_INT_2_10_10_10_REV);
                        if (needsAlpha) fvals[3] = 1.0f;
                    }

                    if (type == GL_UNSIGNED_BYTE_3_3_2) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        dp[0] = (uint8_t)(((uint32_t)lroundf(r * 7.0f) << 5) |
                                          ((uint32_t)lroundf(g * 7.0f) << 2) |
                                          (uint32_t)lroundf(b * 3.0f));
                    } else if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        dp[0] = (uint8_t)((uint32_t)lroundf(r * 7.0f) |
                                          ((uint32_t)lroundf(g * 7.0f) << 3) |
                                          ((uint32_t)lroundf(b * 3.0f) << 6));
                    } else if (type == GL_UNSIGNED_SHORT_5_6_5) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 31.0f) << 11) |
                                                     ((uint32_t)lroundf(g * 63.0f) << 5) |
                                                     (uint32_t)lroundf(b * 31.0f));
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_SHORT_5_6_5_REV) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 31.0f) |
                                                     ((uint32_t)lroundf(g * 63.0f) << 5) |
                                                     ((uint32_t)lroundf(b * 31.0f) << 11));
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_SHORT_4_4_4_4) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 15.0f) << 12) |
                                                     ((uint32_t)lroundf(g * 15.0f) << 8) |
                                                     ((uint32_t)lroundf(b * 15.0f) << 4) |
                                                     (uint32_t)lroundf(a * 15.0f));
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 15.0f) |
                                                     ((uint32_t)lroundf(g * 15.0f) << 4) |
                                                     ((uint32_t)lroundf(b * 15.0f) << 8) |
                                                     ((uint32_t)lroundf(a * 15.0f) << 12));
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 31.0f) << 11) |
                                                     ((uint32_t)lroundf(g * 31.0f) << 6) |
                                                     ((uint32_t)lroundf(b * 31.0f) << 1) |
                                                     (a >= 0.5f ? 1u : 0u));
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 31.0f) |
                                                     ((uint32_t)lroundf(g * 31.0f) << 5) |
                                                     ((uint32_t)lroundf(b * 31.0f) << 10) |
                                                     ((a >= 0.5f ? 1u : 0u) << 15));
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint32_t packed = ((uint32_t)lroundf(r * 255.0f) << 24) |
                                          ((uint32_t)lroundf(g * 255.0f) << 16) |
                                          ((uint32_t)lroundf(b * 255.0f) << 8) |
                                          (uint32_t)lroundf(a * 255.0f);
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint32_t packed = (uint32_t)lroundf(r * 255.0f) |
                                          ((uint32_t)lroundf(g * 255.0f) << 8) |
                                          ((uint32_t)lroundf(b * 255.0f) << 16) |
                                          ((uint32_t)lroundf(a * 255.0f) << 24);
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_10_10_10_2) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint32_t packed = ((uint32_t)lroundf(r * 1023.0f) << 22) |
                                          ((uint32_t)lroundf(g * 1023.0f) << 12) |
                                          ((uint32_t)lroundf(b * 1023.0f) << 2) |
                                          (uint32_t)lroundf(a * 3.0f);
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                        float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                        float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                        float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                        uint32_t packed = (uint32_t)lroundf(r * 1023.0f) |
                                          ((uint32_t)lroundf(g * 1023.0f) << 10) |
                                          ((uint32_t)lroundf(b * 1023.0f) << 20) |
                                          ((uint32_t)lroundf(a * 3.0f) << 30);
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                        float r = fvals[0] < 0.0f ? 0.0f : fvals[0];
                        float g = (slots > 1) ? (fvals[1] < 0.0f ? 0.0f : fvals[1]) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] < 0.0f ? 0.0f : fvals[2]) : 0.0f;
                        uint32_t packed = mglFloatToFloat11(r) |
                                          (mglFloatToFloat11(g) << 11) |
                                          (mglFloatToFloat10(b) << 22);
                        memcpy(dp, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                        float r = fvals[0] < 0.0f ? 0.0f : fvals[0];
                        float g = (slots > 1) ? (fvals[1] < 0.0f ? 0.0f : fvals[1]) : 0.0f;
                        float b = (slots > 2) ? (fvals[2] < 0.0f ? 0.0f : fvals[2]) : 0.0f;
                        uint32_t packed = mglPackRGBToSharedExp(r, g, b);
                        memcpy(dp, &packed, sizeof(packed));
                    }
                    continue; /* Packed output handled, skip per-component loop */
                }

                for (int c = 0; c < slots; ++c) {
                    int idx = srcIdx[c];
                    if (idx >= srcChannels) idx = srcChannels - 1;
                    float fv = 0.0f;
                    if (sourceIs16BitUnorm) {
                        uint16_t uv;
                        memcpy(&uv, s + (NSUInteger)idx * 2u, sizeof(uv));
                        fv = (float)uv / 65535.0f;
                    } else if (sourceIs16BitSnorm) {
                        int16_t sv;
                        memcpy(&sv, s + (NSUInteger)idx * 2u, sizeof(sv));
                        fv = (float)sv / 32767.0f;
                    } else if (sourceIs16BitFloat) {
                        uint16_t hv;
                        memcpy(&hv, s + (NSUInteger)idx * 2u, sizeof(hv));
                        fv = mglHalfToFloat(hv);
                    } else { /* sourceIs32BitFloat */
                        memcpy(&fv, s + (NSUInteger)idx * 4u, sizeof(fv));
                    }
                    uint8_t *out = dp + (NSUInteger)c * compBytes;
                    if (type == GL_UNSIGNED_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int8_t iv = (int8_t)lroundf(cv * 127.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int16_t iv = (int16_t)lroundf(cv * 32767.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int32_t iv = (int32_t)llroundf(cv * 2147483647.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_FLOAT) {
                        memcpy(out, &fv, sizeof(fv));
                    } else { /* GL_HALF_FLOAT */
                        uint16_t iv = mglFloatToHalf(fv);
                        memcpy(out, &iv, sizeof(iv));
                    }
                }
            }
        }
        return YES;
    }

    BOOL sourceIsRGBA =
        (pixelFormat == MTLPixelFormatRGBA8Unorm ||
         pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
    BOOL sourceIsBGRA =
        (pixelFormat == MTLPixelFormatBGRA8Unorm ||
         pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB);
    if (!sourceIsRGBA && !sourceIsBGRA) {
        if (!mglMetalReadbackFormatIsBGRA8Compatible(pixelFormat) ||
            width > NSUIntegerMax / 4u ||
            height > NSUIntegerMax / (width * 4u)) {
            return NO;
        }
        NSUInteger bgraBytesPerRow = width * 4u;
        NSMutableData *bgra = [NSMutableData dataWithLength:bgraBytesPerRow * height];
        if (!bgra) {
            return NO;
        }
        mglMetalCopyTextureBytesToBGRA8(src,
                                        srcBytesPerRow,
                                        (uint8_t *)bgra.mutableBytes,
                                        bgraBytesPerRow,
                                        width,
                                        height,
                                        pixelFormat,
                                        NO);
        return mglMetalCopyBGRA8CompatibleTextureBytesToGL((const uint8_t *)bgra.bytes,
                                                           bgraBytesPerRow,
                                                           dst,
                                                           dstBytesPerRow,
                                                           width,
                                                           height,
                                                           MTLPixelFormatBGRA8Unorm,
                                                           format,
                                                           type,
                                                           flipY);
    }

    NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
    if (dstPixelBytes == 0u || dstBytesPerRow < width * dstPixelBytes) {
        return NO;
    }

    /* Scalar integer / half-float readback from BGRA8 UNORM source.
     * Components are scaled to the destination type's unsigned range. */
    if (type == GL_BYTE || type == GL_SHORT ||
        type == GL_INT || type == GL_UNSIGNED_INT ||
        type == GL_UNSIGNED_SHORT || type == GL_HALF_FLOAT ||
        type == GL_FLOAT) {
        NSUInteger compBytes = (NSUInteger)sizeForType(type);
        int slots = 0;
        int srcIdx[4] = {0,0,0,0};
        switch (format) {
            case GL_RGBA: slots = 4; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; srcIdx[3]=3; break;
            case GL_BGRA: slots = 4; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; srcIdx[3]=3; break;
            case GL_RGB:  slots = 3; srcIdx[0]=0; srcIdx[1]=1; srcIdx[2]=2; break;
            case GL_BGR:  slots = 3; srcIdx[0]=2; srcIdx[1]=1; srcIdx[2]=0; break;
            case GL_RG:   slots = 2; srcIdx[0]=0; srcIdx[1]=1; break;
            case GL_RED:  slots = 1; srcIdx[0]=0; break;
            case GL_GREEN: slots = 1; srcIdx[0]=1; break;
            case GL_BLUE:  slots = 1; srcIdx[0]=2; break;
            case GL_ALPHA: slots = 1; srcIdx[0]=3; break;
            default: return NO;
        }
        for (NSUInteger y = 0; y < height; y++) {
            const uint8_t *srcRow = src + (y * srcBytesPerRow);
            NSUInteger dstY = flipY ? (height - 1u - y) : y;
            uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
            for (NSUInteger x = 0; x < width; x++) {
                const uint8_t *s = srcRow + (x * 4u);
                /* cv[0]=R, cv[1]=G, cv[2]=B, cv[3]=A in logical order.
                 * BGRA8 source: s[0]=B, s[1]=G, s[2]=R, s[3]=A.
                 * RGBA8 source: s[0]=R, s[1]=G, s[2]=B, s[3]=A. */
                const unsigned cv[4] = {
                    sourceIsRGBA ? s[0] : s[2],
                    s[1],
                    sourceIsRGBA ? s[2] : s[0],
                    s[3]
                };
                uint8_t *dp = dstRow + (x * dstPixelBytes);
                for (int c = 0; c < slots; ++c) {
                    unsigned v = cv[srcIdx[c]];
                    uint8_t *out = dp + (NSUInteger)c * compBytes;
                    if (type == GL_BYTE) {
                        /* UNORM (0-255) -> SNORM (-128..127) */
                        float fv = (float)v / 255.0f;
                        int32_t iv = (int32_t)lroundf(fv * 127.0f);
                        if (iv > 127) iv = 127;
                        if (iv < -128) iv = -128;
                        int8_t biv = (int8_t)iv;
                        memcpy(out, &biv, sizeof(biv));
                    } else if (type == GL_UNSIGNED_SHORT) {
                        uint16_t iv = (uint16_t)((uint32_t)v * 257u);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_SHORT) {
                        int32_t scaled = (int32_t)((uint32_t)v * 32767u / 255u);
                        if (scaled > 32767) scaled = 32767;
                        int16_t iv = (int16_t)scaled;
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_INT) {
                        uint32_t iv = (uint32_t)v * 16843009u;
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_INT) {
                        /* Use 64-bit arithmetic to avoid uint32 overflow:
                         * v * 2147483647 overflows uint32 for v > 1. */
                        int32_t scaled = (int32_t)((uint64_t)v * 2147483647ULL / 255u);
                        if (scaled > 2147483647) scaled = 2147483647;
                        memcpy(out, &scaled, sizeof(scaled));
                    } else if (type == GL_FLOAT) {
                        float fv = (float)v / 255.0f;
                        memcpy(out, &fv, sizeof(fv));
                    } else { /* GL_HALF_FLOAT */
                        uint16_t iv = mglFloatToHalf((float)v / 255.0f);
                        memcpy(out, &iv, sizeof(iv));
                    }
                }
            }
        }
        return YES;
    }

    /* Packed type readback from BGRA8/RGBA8 UNORM source.
     * Each pixel is extracted from 4-byte BGRA8/RGBA8 source and packed
     * into the destination packed type. */
    if (type == GL_UNSIGNED_BYTE_3_3_2 ||
        type == GL_UNSIGNED_BYTE_2_3_3_REV ||
        type == GL_UNSIGNED_SHORT_5_6_5 ||
        type == GL_UNSIGNED_SHORT_5_6_5_REV ||
        type == GL_UNSIGNED_SHORT_4_4_4_4 ||
        type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
        type == GL_UNSIGNED_SHORT_5_5_5_1 ||
        type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
        type == GL_UNSIGNED_INT_8_8_8_8 ||
        type == GL_UNSIGNED_INT_8_8_8_8_REV ||
        type == GL_UNSIGNED_INT_10_10_10_2 ||
        type == GL_UNSIGNED_INT_2_10_10_10_REV ||
        type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
        type == GL_UNSIGNED_INT_5_9_9_9_REV) {
        for (NSUInteger y = 0; y < height; y++) {
            const uint8_t *srcRow = src + (y * srcBytesPerRow);
            NSUInteger dstY = flipY ? (height - 1u - y) : y;
            uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
            for (NSUInteger x = 0; x < width; x++) {
                const uint8_t *s = srcRow + (x * 4u);
                uint32_t r = sourceIsRGBA ? s[0] : s[2];
                uint32_t g = s[1];
                uint32_t b = sourceIsRGBA ? s[2] : s[0];
                uint32_t a = s[3];
                /* Apply format channel mapping */
                uint32_t rr = r, gg = g, bb = b, aa = a;
                switch (format) {
                    case GL_RGBA: case GL_RGB: case GL_RED: case GL_RG:
                    case GL_GREEN: case GL_BLUE: case GL_ALPHA:
                        break;
                    case GL_BGRA: case GL_BGR: {
                        uint32_t tmp = rr; rr = bb; bb = tmp;
                        break;
                    }
                    default:
                        break;
                }
                uint8_t *d = dstRow + (x * dstPixelBytes);
                if (type == GL_UNSIGNED_BYTE_3_3_2) {
                    d[0] = (uint8_t)(((rr >> 5u) << 5u) | ((gg >> 5u) << 2u) | (bb >> 6u));
                } else if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
                    d[0] = (uint8_t)((rr >> 5u) | ((gg >> 5u) << 3u) | ((bb >> 6u) << 6u));
                } else if (type == GL_UNSIGNED_SHORT_5_6_5) {
                    uint16_t packed = (uint16_t)(((rr >> 3u) << 11u) | ((gg >> 2u) << 5u) | (bb >> 3u));
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_5_6_5_REV) {
                    uint16_t packed = (uint16_t)((rr >> 3u) | ((gg >> 2u) << 5u) | ((bb >> 3u) << 11u));
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_4_4_4_4) {
                    uint16_t packed = (uint16_t)(((rr >> 4u) << 12u) | ((gg >> 4u) << 8u) | ((bb >> 4u) << 4u) | (aa >> 4u));
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
                    uint16_t packed = (uint16_t)((rr >> 4u) | ((gg >> 4u) << 4u) | ((bb >> 4u) << 8u) | ((aa >> 4u) << 12u));
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
                    uint16_t packed = (uint16_t)(((rr >> 3u) << 11u) | ((gg >> 3u) << 6u) | ((bb >> 3u) << 1u) | (aa >= 128u ? 1u : 0u));
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
                    uint16_t packed = (uint16_t)((rr >> 3u) | ((gg >> 3u) << 5u) | ((bb >> 3u) << 10u) | ((aa >= 128u ? 1u : 0u) << 15u));
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                    /* CTS: R=(val>>24), G=(val>>16), B=(val>>8), A=(val>>0).
                     * On little-endian this stores as [A,B,G,R] in memory. */
                    uint32_t packed = ((uint32_t)rr << 24u) | ((uint32_t)gg << 16u) | ((uint32_t)bb << 8u) | aa;
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                    /* CTS: R=(val>>0), G=(val>>8), B=(val>>16), A=(val>>24).
                     * On little-endian this stores as [R,G,B,A] in memory. */
                    uint32_t packed = rr | ((uint32_t)gg << 8u) | ((uint32_t)bb << 16u) | ((uint32_t)aa << 24u);
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_10_10_10_2) {
                    uint32_t r10 = rr * 1023u / 255u;
                    uint32_t g10 = gg * 1023u / 255u;
                    uint32_t b10 = bb * 1023u / 255u;
                    uint32_t a2 = aa * 3u / 255u;
                    uint32_t packed = (r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2;
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                    uint32_t r10 = rr * 1023u / 255u;
                    uint32_t g10 = gg * 1023u / 255u;
                    uint32_t b10 = bb * 1023u / 255u;
                    uint32_t a2 = aa * 3u / 255u;
                    uint32_t packed = r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u);
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                    uint32_t packed = mglPackUnsignedFloatFromUNorm8(rr, 6u) |
                                      (mglPackUnsignedFloatFromUNorm8(gg, 6u) << 11u) |
                                      (mglPackUnsignedFloatFromUNorm8(bb, 5u) << 22u);
                    memcpy(d, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                    uint32_t packed = mglPackRGBToSharedExp((double)rr / 255.0, (double)gg / 255.0, (double)bb / 255.0);
                    memcpy(d, &packed, sizeof(packed));
                }
            }
        }
        return YES;
    }

    for (NSUInteger y = 0; y < height; y++) {
        const uint8_t *srcRow = src + (y * srcBytesPerRow);
        NSUInteger dstY = flipY ? (height - 1u - y) : y;
        uint8_t *dstRow = dst + (dstY * dstBytesPerRow);

        for (NSUInteger x = 0; x < width; x++) {
            const uint8_t *s = srcRow + (x * 4u);
            uint8_t r = sourceIsRGBA ? s[0] : s[2];
            uint8_t g = s[1];
            uint8_t b = sourceIsRGBA ? s[2] : s[0];
            uint8_t a = s[3];
            uint8_t *d = dstRow + (x * dstPixelBytes);

            switch (format) {
                case GL_BGRA:
                    if (dstPixelBytes != 4u) {
                        return NO;
                    }
                    d[0] = b;
                    d[1] = g;
                    d[2] = r;
                    d[3] = a;
                    break;
                case GL_RGBA:
                    if (dstPixelBytes != 4u && (type != GL_FLOAT || dstPixelBytes != 16u)) {
                        return NO;
                    }
                    if (type == GL_FLOAT) {
                        float *fd = (float *)d;
                        fd[0] = (float)r / 255.0f;
                        fd[1] = (float)g / 255.0f;
                        fd[2] = (float)b / 255.0f;
                        fd[3] = (float)a / 255.0f;
                    } else {
                        d[0] = r;
                        d[1] = g;
                        d[2] = b;
                        d[3] = a;
                    }
                    break;
                case GL_BGR:
                    if (type != GL_UNSIGNED_BYTE || dstPixelBytes != 3u) {
                        return NO;
                    }
                    d[0] = b;
                    d[1] = g;
                    d[2] = r;
                    break;
                case GL_RGB:
                    if (type != GL_UNSIGNED_BYTE || dstPixelBytes != 3u) {
                        return NO;
                    }
                    d[0] = r;
                    d[1] = g;
                    d[2] = b;
                    break;
                case GL_RG:
                    if (type != GL_UNSIGNED_BYTE || dstPixelBytes != 2u) {
                        return NO;
                    }
                    d[0] = r;
                    d[1] = g;
                    break;
                case GL_RED:
                    if (type != GL_UNSIGNED_BYTE || dstPixelBytes != 1u) {
                        return NO;
                    }
                    d[0] = r;
                    break;
                case GL_GREEN:
                    if (type != GL_UNSIGNED_BYTE || dstPixelBytes != 1u) {
                        return NO;
                    }
                    d[0] = g;
                    break;
                case GL_BLUE:
                    if (type != GL_UNSIGNED_BYTE || dstPixelBytes != 1u) {
                        return NO;
                    }
                    d[0] = b;
                    break;
                case GL_ALPHA:
                    if (type != GL_UNSIGNED_BYTE || dstPixelBytes != 1u) {
                        return NO;
                    }
                    d[0] = a;
                    break;
                default:
                    return NO;
            }
        }
    }

    return YES;
}

BOOL mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(const uint8_t *src,
                                                                 NSUInteger srcBytesPerRow,
                                                                 uint8_t *dst,
                                                                 NSUInteger dstBytesPerRow,
                                                                 NSUInteger width,
                                                                 NSUInteger height,
                                                                 MTLPixelFormat pixelFormat,
                                                                 BOOL flipY)
{
    if (!src || !dst || width == 0u || height == 0u) {
        return NO;
    }
    if (srcBytesPerRow < width * 4u || dstBytesPerRow < width * 4u) {
        return NO;
    }

    BOOL destinationIsRGBA =
        (pixelFormat == MTLPixelFormatRGBA8Unorm ||
         pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
    BOOL destinationIsBGRA =
        (pixelFormat == MTLPixelFormatBGRA8Unorm ||
         pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB);
    BOOL destinationIsRGB9E5 = (pixelFormat == MTLPixelFormatRGB9E5Float);
    BOOL destinationIsRGB10A2 = (pixelFormat == MTLPixelFormatRGB10A2Unorm ||
                                 pixelFormat == MTLPixelFormatBGR10A2Unorm);
    if (!destinationIsRGBA && !destinationIsBGRA && !destinationIsRGB9E5 && !destinationIsRGB10A2) {
        return NO;
    }

    for (NSUInteger y = 0; y < height; y++) {
        const uint8_t *srcRow = src + (y * srcBytesPerRow);
        NSUInteger dstY = flipY ? (height - 1u - y) : y;
        uint8_t *dstRow = dst + (dstY * dstBytesPerRow);

        for (NSUInteger x = 0; x < width; x++) {
            const uint8_t *s = srcRow + (x * 4u);
            uint8_t *d = dstRow + (x * 4u);
            uint8_t b = s[0];
            uint8_t g = s[1];
            uint8_t r = s[2];
            uint8_t a = s[3];

            if (destinationIsBGRA) {
                d[0] = b;
                d[1] = g;
                d[2] = r;
                d[3] = a;
            } else if (destinationIsRGB10A2) {
                /* RGB10A2Unorm: bits [0:9]=R, [10:19]=G, [20:29]=B, [30:31]=A.
                 * BGR10A2Unorm: bits [0:9]=B, [10:19]=G, [20:29]=R, [30:31]=A. */
                uint32_t r10 = ((uint32_t)r * 1023u + 127u) / 255u;
                uint32_t g10 = ((uint32_t)g * 1023u + 127u) / 255u;
                uint32_t b10 = ((uint32_t)b * 1023u + 127u) / 255u;
                uint32_t a2 = ((uint32_t)a * 3u + 127u) / 255u;
                uint32_t packed;
                if (pixelFormat == MTLPixelFormatBGR10A2Unorm) {
                    packed = b10 | (g10 << 10) | (r10 << 20) | (a2 << 30);
                } else {
                    packed = r10 | (g10 << 10) | (b10 << 20) | (a2 << 30);
                }
                d[0] = (uint8_t)(packed & 0xFF);
                d[1] = (uint8_t)((packed >> 8) & 0xFF);
                d[2] = (uint8_t)((packed >> 16) & 0xFF);
                d[3] = (uint8_t)((packed >> 24) & 0xFF);
            } else if (destinationIsRGB9E5) {
                /* GL_RGB9_E5 packs three 9-bit mantissas and a 5-bit
                 * shared exponent into a 32-bit word. Source is BGRA8. */
                uint32_t packed = mglPackRGBToSharedExp((double)r / 255.0,
                                                        (double)g / 255.0,
                                                        (double)b / 255.0);
                d[0] = (uint8_t)(packed & 0xFF);
                d[1] = (uint8_t)((packed >> 8) & 0xFF);
                d[2] = (uint8_t)((packed >> 16) & 0xFF);
                d[3] = (uint8_t)((packed >> 24) & 0xFF);
            } else {
                d[0] = r;
                d[1] = g;
                d[2] = b;
                d[3] = a;
            }
        }
    }

    return YES;
}
