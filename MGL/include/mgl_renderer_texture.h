/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Non-drawable texture/compute/blit orchestration (C++ home).
 * ObjC bridge: mgl_renderer_texture_bridge.m
 */

#ifndef MGL_RENDERER_TEXTURE_H
#define MGL_RENDERER_TEXTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderBindTexture(GLMContext context, Texture *texture);
void mglRenderGenerateMipmaps(GLMContext context, Texture *texture);

void mglRenderReadDrawable(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height);
void mglRenderReadIntegerPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type);
void mglRenderReadDepthPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height);
void mglRenderGetTexImage(
    GLMContext context, Texture *texture, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice);
void mglRenderTexSubImage(
    GLMContext context, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset);
bool mglRenderTexSubImageBytes(
    GLMContext context, Texture *texture, const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset);
void mglRenderCopyTexSubImage(
    GLMContext context, Texture *texture, uint32_t slice, int32_t level,
    int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height);
void mglRenderCopyImageSubData(
    GLMContext context, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_TEXTURE_H */
