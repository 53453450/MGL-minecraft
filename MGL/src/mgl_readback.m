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
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppReadbackFormatIsBGRA8Compatible), shared by both gates. */
    return mglRenderCppReadbackFormatIsBGRA8Compatible(
               (uint32_t)pixelFormat) ? YES : NO;
}

BOOL mglMetalPixelFormatIsIntegerColor(MTLPixelFormat pixelFormat)
{
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppPixelFormatIsIntegerColor), shared by both gates. */
    return mglRenderCppPixelFormatIsIntegerColor(
               (uint32_t)pixelFormat) ? YES : NO;
}

BOOL mglMetalPixelFormatIsSignedIntegerColor(MTLPixelFormat pixelFormat)
{
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppPixelFormatIsSignedIntegerColor), shared by both gates. */
    return mglRenderCppPixelFormatIsSignedIntegerColor(
               (uint32_t)pixelFormat) ? YES : NO;
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
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppCopyTextureBytesToBGRA8), shared by both gates. */
    mglRenderCppCopyTextureBytesToBGRA8(
        src, (uint64_t)srcBytesPerRow,
        dst, (uint64_t)dstBytesPerRow,
        (uint64_t)width, (uint64_t)height,
        (uint32_t)pixelFormat, flipY ? 1 : 0);
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

    /* P4.5 (item 1171): type-accept table in C++. */
    if (!mglRenderCppReadbackGLTypeAccepted((uint32_t)type)) {
        return NO;
    }

    /* P4.5 (item 1171): SNORM8 direct path in C++ (bypass lossy BGRA8). */
    BOOL sourceIsSnorm8 =
        (pixelFormat == MTLPixelFormatR8Snorm ||
         pixelFormat == MTLPixelFormatRG8Snorm ||
         pixelFormat == MTLPixelFormatRGBA8Snorm);
    if (sourceIsSnorm8) {
        return mglRenderCppCopySnorm8TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* P4.5 (item 1171): RGB10A2 direct path in C++ (bypass lossy BGRA8). */
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
        return mglRenderCppCopyRGB10A2TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* P4.5 (item 1171): RG11B10Float direct path in C++ (bypass lossy BGRA8). */
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
        return mglRenderCppCopyRG11B10TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* P4.5 (item 1171): 16/32-bit direct path in C++ (bypass lossy BGRA8). */
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
        return mglRenderCppCopy16or32TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
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

    /* P4.5 (item 1171): BGRA8/RGBA8 scalar readback in C++. */
    if (type == GL_BYTE || type == GL_SHORT ||
        type == GL_INT || type == GL_UNSIGNED_INT ||
        type == GL_UNSIGNED_SHORT || type == GL_HALF_FLOAT ||
        type == GL_FLOAT) {
        return mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* P4.5 (item 1171): BGRA8/RGBA8 packed readback in C++. */
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
        return mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
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
    /* P4.5 (item 1171): thin delegate — single source of truth in C++
     * (mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes), shared by
     * both gates.  Returns 0 on bad args / unsupported format. */
    return mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
               src, (uint64_t)srcBytesPerRow,
               dst, (uint64_t)dstBytesPerRow,
               (uint64_t)width, (uint64_t)height,
               (uint32_t)pixelFormat, flipY ? 1 : 0)
        ? YES : NO;
}
