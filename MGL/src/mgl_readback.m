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
 * mgl_readback.m
 * MGL
 *
 * Pure-C pixel format readback / conversion helpers extracted from
 * MGLRenderer.m. Function bodies are preserved verbatim; only the
 * "static" storage-class qualifier was removed to make the symbols
 * externally visible.
 */

#import <Foundation/Foundation.h>
#import "pixel_utils.h"
#import "mgl_readback.h"
#include "mgl_render.h"
#include <stdint.h>

BOOL mglMetalReadbackFormatIsBGRA8Compatible(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderReadbackFormatIsBGRA8Compatible), shared by both gates. */
    return mglRenderReadbackFormatIsBGRA8Compatible(
               (uint32_t)pixelFormat) ? YES : NO;
}

BOOL mglMetalPixelFormatIsIntegerColor(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderPixelFormatIsIntegerColor), shared by both gates. */
    return mglRenderPixelFormatIsIntegerColor(
               (uint32_t)pixelFormat) ? YES : NO;
}

BOOL mglMetalPixelFormatIsSignedIntegerColor(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderPixelFormatIsSignedIntegerColor), shared by both gates. */
    return mglRenderPixelFormatIsSignedIntegerColor(
               (uint32_t)pixelFormat) ? YES : NO;
}

NSUInteger mglMetalReadbackBytesPerPixel(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderReadbackBytesPerPixel, pixel format as its Apple ABI
     * value), shared by both gates. */
    return (NSUInteger)mglRenderReadbackBytesPerPixel(
        (uint32_t)pixelFormat);
}

uint8_t mglMetalFloatToUnorm8(float value)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderFloatToUnorm8), shared by both gates. */
    return mglRenderFloatToUnorm8(value);
}

float mglMetalSnorm16ToFloat(int16_t value)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderSnorm16ToFloat), shared by both gates. */
    return mglRenderSnorm16ToFloat(value);
}

float mglMetalSnorm8ToFloat(int8_t value)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderSnorm8ToFloat), shared by both gates. */
    return mglRenderSnorm8ToFloat(value);
}

void mglMetalCopyTextureBytesToBGRA8(const uint8_t *src,
                                            NSUInteger srcBytesPerRow,
                                            uint8_t *dst,
                                            NSUInteger dstBytesPerRow,
                                            NSUInteger width,
                                            NSUInteger height,
                                            uint32_t pixelFormat,
                                            BOOL flipY)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderCopyTextureBytesToBGRA8), shared by both gates. */
    mglRenderCopyTextureBytesToBGRA8(
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
                                                        uint32_t pixelFormat,
                                                        GLenum format,
                                                        GLenum type,
                                                        BOOL flipY)
{
    if (!src || !dst || width == 0u || height == 0u) {
        return NO;
    }

    /* type-accept table in C++. */
    if (!mglRenderReadbackGLTypeAccepted((uint32_t)type)) {
        return NO;
    }

    /* SNORM8 direct path in C++ (bypass lossy BGRA8). */
    BOOL sourceIsSnorm8 =
        (pixelFormat == MGLPixelFormatR8Snorm ||
         pixelFormat == MGLPixelFormatRG8Snorm ||
         pixelFormat == MGLPixelFormatRGBA8Snorm);
    if (sourceIsSnorm8) {
        return mglRenderCopySnorm8TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* RGB10A2 direct path in C++ (bypass lossy BGRA8). */
    BOOL sourceIsRGB10A2Direct = (pixelFormat == MGLPixelFormatRGB10A2Unorm);
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
        return mglRenderCopyRGB10A2TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* RG11B10Float direct path in C++ (bypass lossy BGRA8). */
    BOOL sourceIsRG11B10FloatDirect = (pixelFormat == MGLPixelFormatRG11B10Float);
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
        return mglRenderCopyRG11B10TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* 16/32-bit direct path in C++ (bypass lossy BGRA8). */
    BOOL sourceIs16BitUnorm =
        (pixelFormat == MGLPixelFormatR16Unorm ||
         pixelFormat == MGLPixelFormatRG16Unorm ||
         pixelFormat == MGLPixelFormatRGBA16Unorm);
    BOOL sourceIs16BitSnorm =
        (pixelFormat == MGLPixelFormatR16Snorm ||
         pixelFormat == MGLPixelFormatRG16Snorm ||
         pixelFormat == MGLPixelFormatRGBA16Snorm);
    BOOL sourceIs16BitFloat =
        (pixelFormat == MGLPixelFormatR16Float ||
         pixelFormat == MGLPixelFormatRG16Float ||
         pixelFormat == MGLPixelFormatRGBA16Float);
    BOOL sourceIs32BitFloat =
        (pixelFormat == MGLPixelFormatR32Float ||
         pixelFormat == MGLPixelFormatRG32Float ||
         pixelFormat == MGLPixelFormatRGBA32Float);

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
        return mglRenderCopy16or32TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    BOOL sourceIsRGBA =
        (pixelFormat == MGLPixelFormatRGBA8Unorm ||
         pixelFormat == MGLPixelFormatRGBA8Unorm_sRGB);
    BOOL sourceIsBGRA =
        (pixelFormat == MGLPixelFormatBGRA8Unorm ||
         pixelFormat == MGLPixelFormatBGRA8Unorm_sRGB);
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
                                                           MGLPixelFormatBGRA8Unorm,
                                                           format,
                                                           type,
                                                           flipY);
    }

    NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
    if (dstPixelBytes == 0u || dstBytesPerRow < width * dstPixelBytes) {
        return NO;
    }

    /* BGRA8/RGBA8 scalar readback in C++. */
    if (type == GL_BYTE || type == GL_SHORT ||
        type == GL_INT || type == GL_UNSIGNED_INT ||
        type == GL_UNSIGNED_SHORT || type == GL_HALF_FLOAT ||
        type == GL_FLOAT) {
        return mglRenderCopyUnorm8ScalarTextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* BGRA8/RGBA8 packed readback in C++. */
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
        return mglRenderCopyUnorm8PackedTextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? YES : NO;
    }

    /* UNSIGNED_BYTE channel-swizzle tail in C++. */
    return mglRenderCopyUnorm8SwizzleTextureBytesToGL(
               src, (uint64_t)srcBytesPerRow,
               dst, (uint64_t)dstBytesPerRow,
               (uint64_t)width, (uint64_t)height,
               (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
               flipY ? 1 : 0)
        ? YES : NO;
}

BOOL mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(const uint8_t *src,
                                                                 NSUInteger srcBytesPerRow,
                                                                 uint8_t *dst,
                                                                 NSUInteger dstBytesPerRow,
                                                                 NSUInteger width,
                                                                 NSUInteger height,
                                                                 uint32_t pixelFormat,
                                                                 BOOL flipY)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes), shared by
     * both gates.  Returns 0 on bad args / unsupported format. */
    return mglRenderCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
               src, (uint64_t)srcBytesPerRow,
               dst, (uint64_t)dstBytesPerRow,
               (uint64_t)width, (uint64_t)height,
               (uint32_t)pixelFormat, flipY ? 1 : 0)
        ? YES : NO;
}
