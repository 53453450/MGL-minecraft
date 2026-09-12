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

#include "pixel_utils.h"
#include "mgl_readback.h"
#include <stdbool.h>
#include <stdlib.h> /* calloc / free: scratch readback buffer */
#include <stddef.h>
#include "mgl_render.h"
#include <stdint.h>

bool mglMetalReadbackFormatIsBGRA8Compatible(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderReadbackFormatIsBGRA8Compatible), shared by both gates. */
    return mglRenderReadbackFormatIsBGRA8Compatible(
               (uint32_t)pixelFormat) ? true : false;
}

bool mglMetalPixelFormatIsIntegerColor(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderPixelFormatIsIntegerColor), shared by both gates. */
    return mglRenderPixelFormatIsIntegerColor(
               (uint32_t)pixelFormat) ? true : false;
}

bool mglMetalPixelFormatIsSignedIntegerColor(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderPixelFormatIsSignedIntegerColor), shared by both gates. */
    return mglRenderPixelFormatIsSignedIntegerColor(
               (uint32_t)pixelFormat) ? true : false;
}

size_t mglMetalReadbackBytesPerPixel(uint32_t pixelFormat)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderReadbackBytesPerPixel, pixel format as its Apple ABI
     * value), shared by both gates. */
    return (size_t)mglRenderReadbackBytesPerPixel(
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
                                            size_t srcBytesPerRow,
                                            uint8_t *dst,
                                            size_t dstBytesPerRow,
                                            size_t width,
                                            size_t height,
                                            uint32_t pixelFormat,
                                            bool flipY)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderCopyTextureBytesToBGRA8), shared by both gates. */
    mglRenderCopyTextureBytesToBGRA8(
        src, (uint64_t)srcBytesPerRow,
        dst, (uint64_t)dstBytesPerRow,
        (uint64_t)width, (uint64_t)height,
        (uint32_t)pixelFormat, flipY ? 1 : 0);
}

bool mglMetalCopyBGRA8CompatibleTextureBytesToGL(const uint8_t *src,
                                                        size_t srcBytesPerRow,
                                                        uint8_t *dst,
                                                        size_t dstBytesPerRow,
                                                        size_t width,
                                                        size_t height,
                                                        uint32_t pixelFormat,
                                                        GLenum format,
                                                        GLenum type,
                                                        bool flipY)
{
    if (!src || !dst || width == 0u || height == 0u) {
        return false;
    }

    /* type-accept table in C++. */
    if (!mglRenderReadbackGLTypeAccepted((uint32_t)type)) {
        return false;
    }

    /* SNORM8 direct path in C++ (bypass lossy BGRA8). */
    if (mglRenderReadbackPixelFormatIsSnorm8((uint32_t)pixelFormat)) {
        return mglRenderCopySnorm8TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? true : false;
    }

    /* RGB10A2 direct path in C++ (bypass lossy BGRA8). */
    if (mglRenderReadbackPixelFormatIsRGB10A2((uint32_t)pixelFormat) &&
        mglRenderReadbackTypeAllowsRGB10A2((uint32_t)type))
    {
        return mglRenderCopyRGB10A2TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? true : false;
    }

    /* RG11B10Float direct path in C++ (bypass lossy BGRA8). */
    if (mglRenderReadbackPixelFormatIsRG11B10((uint32_t)pixelFormat) &&
        mglRenderReadbackTypeAllowsRG11B10((uint32_t)type))
    {
        return mglRenderCopyRG11B10TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? true : false;
    }

    /* 16/32-bit direct path in C++ (bypass lossy BGRA8). */
    if (mglRenderReadbackPixelFormatIs16or32((uint32_t)pixelFormat) &&
        mglRenderReadbackTypeAllows16or32((uint32_t)type))
    {
        return mglRenderCopy16or32TextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? true : false;
    }

    if (!mglRenderReadbackPixelFormatIsRGBA8((uint32_t)pixelFormat) &&
        !mglRenderReadbackPixelFormatIsBGRA8((uint32_t)pixelFormat)) {
        if (!mglMetalReadbackFormatIsBGRA8Compatible(pixelFormat) ||
            width > SIZE_MAX / 4u ||
            height > SIZE_MAX / (width * 4u)) {
            return false;
        }
        const size_t bgraBytesPerRow = width * 4u;
        /* Zero-filled scratch, like the NSMutableData this replaced. */
        uint8_t *bgra = (uint8_t *)calloc(bgraBytesPerRow * height, 1u);
        if (!bgra) {
            return false;
        }
        mglMetalCopyTextureBytesToBGRA8(src,
                                        srcBytesPerRow,
                                        bgra,
                                        bgraBytesPerRow,
                                        width,
                                        height,
                                        pixelFormat,
                                        false);
        const bool copied =
            mglMetalCopyBGRA8CompatibleTextureBytesToGL(bgra,
                                                        bgraBytesPerRow,
                                                        dst,
                                                        dstBytesPerRow,
                                                        width,
                                                        height,
                                                        mglRenderReadbackBGRA8CarrierFormat(),
                                                        format,
                                                        type,
                                                        flipY);
        free(bgra);
        return copied;
    }

    size_t dstPixelBytes = (size_t)sizeForFormatType(format, type);
    if (dstPixelBytes == 0u || dstBytesPerRow < width * dstPixelBytes) {
        return false;
    }

    /* BGRA8/RGBA8 scalar readback in C++. */
    if (mglRenderReadbackTypeIsWideScalar((uint32_t)type)) {
        return mglRenderCopyUnorm8ScalarTextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? true : false;
    }

    /* BGRA8/RGBA8 packed readback in C++. */
    if (mglRenderReadbackTypeIsPacked((uint32_t)type)) {
        return mglRenderCopyUnorm8PackedTextureBytesToGL(
                   src, (uint64_t)srcBytesPerRow,
                   dst, (uint64_t)dstBytesPerRow,
                   (uint64_t)width, (uint64_t)height,
                   (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
                   flipY ? 1 : 0)
            ? true : false;
    }

    /* UNSIGNED_BYTE channel-swizzle tail in C++. */
    return mglRenderCopyUnorm8SwizzleTextureBytesToGL(
               src, (uint64_t)srcBytesPerRow,
               dst, (uint64_t)dstBytesPerRow,
               (uint64_t)width, (uint64_t)height,
               (uint32_t)pixelFormat, (uint32_t)format, (uint32_t)type,
               flipY ? 1 : 0)
        ? true : false;
}

bool mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(const uint8_t *src,
                                                                 size_t srcBytesPerRow,
                                                                 uint8_t *dst,
                                                                 size_t dstBytesPerRow,
                                                                 size_t width,
                                                                 size_t height,
                                                                 uint32_t pixelFormat,
                                                                 bool flipY)
{
    /* thin delegate — single source of truth in C++
     * (mglRenderCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes), shared by
     * both gates.  Returns 0 on bad args / unsupported format. */
    return mglRenderCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
               src, (uint64_t)srcBytesPerRow,
               dst, (uint64_t)dstBytesPerRow,
               (uint64_t)width, (uint64_t)height,
               (uint32_t)pixelFormat, flipY ? 1 : 0)
        ? true : false;
}
