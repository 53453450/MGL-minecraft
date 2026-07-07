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
 * mgl_readback.h
 * MGL
 *
 * Pure-C pixel format readback / conversion helpers extracted from
 * MGLRenderer.m. These functions do not depend on any instance state and
 * are shared between translation units that perform Metal <-> GL pixel
 * readback conversions.
 */

#ifndef MGL_READBACK_H
#define MGL_READBACK_H

#include <Metal/Metal.h>
#include "glcorearb.h"

#ifdef __cplusplus
extern "C" {
#endif

BOOL mglMetalReadbackFormatIsBGRA8Compatible(MTLPixelFormat pixelFormat);
BOOL mglMetalPixelFormatIsIntegerColor(MTLPixelFormat pixelFormat);
BOOL mglMetalPixelFormatIsSignedIntegerColor(MTLPixelFormat pixelFormat);
NSUInteger mglMetalReadbackBytesPerPixel(MTLPixelFormat pixelFormat);
uint8_t mglMetalFloatToUnorm8(float value);
float mglMetalSnorm16ToFloat(int16_t value);
float mglMetalSnorm8ToFloat(int8_t value);
void mglMetalCopyTextureBytesToBGRA8(const uint8_t *src, NSUInteger srcBytesPerRow, uint8_t *dst, NSUInteger dstBytesPerRow, NSUInteger width, NSUInteger height, MTLPixelFormat pixelFormat, BOOL flipY);
BOOL mglMetalCopyBGRA8CompatibleTextureBytesToGL(const uint8_t *src, NSUInteger srcBytesPerRow, uint8_t *dst, NSUInteger dstBytesPerRow, NSUInteger width, NSUInteger height, MTLPixelFormat pixelFormat, GLenum format, GLenum type, BOOL flipY);
BOOL mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(const uint8_t *src, NSUInteger srcBytesPerRow, uint8_t *dst, NSUInteger dstBytesPerRow, NSUInteger width, NSUInteger height, MTLPixelFormat pixelFormat, BOOL flipY);

#ifdef __cplusplus
}
#endif

#endif /* MGL_READBACK_H */
