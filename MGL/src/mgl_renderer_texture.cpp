/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_texture.h"

extern "C" void mglRendererObjCBindTexture(GLMContext context,
                                           Texture *texture);
extern "C" void mglRendererObjCGenerateMipmaps(GLMContext context,
                                               Texture *texture);
extern "C" void mglRendererObjCReadDrawable(
    GLMContext context, void *pixel_bytes, uint32_t bytes_per_row,
    uint32_t bytes_per_image, int32_t x, int32_t y, int32_t width, int32_t height);
extern "C" void mglRendererObjCReadIntegerPixels(
    GLMContext context, void *pixel_bytes, uint32_t bytes_per_row,
    uint32_t bytes_per_image, int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type);
extern "C" void mglRendererObjCReadDepthPixels(
    GLMContext context, void *pixel_bytes, uint32_t bytes_per_row,
    uint32_t bytes_per_image, int32_t x, int32_t y, int32_t width, int32_t height);
extern "C" void mglRendererObjCGetTexImage(
    GLMContext context, Texture *texture, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice);
extern "C" void mglRendererObjCTexSubImage(
    GLMContext context, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset);
extern "C" bool mglRendererObjCTexSubImageBytes(
    GLMContext context, Texture *texture, const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset);
extern "C" void mglRendererObjCCopyTexSubImage(
    GLMContext context, Texture *texture, uint32_t slice, int32_t level,
    int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height);
extern "C" void mglRendererObjCCopyImageSubData(
    GLMContext context, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth);

extern "C" void mglRenderBindTexture(GLMContext context, Texture *texture)
{
    if (!context) {
        return;
    }
    mglRendererObjCBindTexture(context, texture);
}

extern "C" void mglRenderGenerateMipmaps(GLMContext context, Texture *texture)
{
    if (!context) {
        return;
    }
    mglRendererObjCGenerateMipmaps(context, texture);
}

extern "C" void mglRenderReadDrawable(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    if (!context) {
        return;
    }
    mglRendererObjCReadDrawable(context, pixel_bytes, bytes_per_row,
                                bytes_per_image, x, y, width, height);
}

extern "C" void mglRenderReadIntegerPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type)
{
    if (!context) {
        return;
    }
    mglRendererObjCReadIntegerPixels(context, pixel_bytes, bytes_per_row,
                                     bytes_per_image, x, y, width, height,
                                     format, type);
}

extern "C" void mglRenderReadDepthPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    if (!context) {
        return;
    }
    mglRendererObjCReadDepthPixels(context, pixel_bytes, bytes_per_row,
                                   bytes_per_image, x, y, width, height);
}

extern "C" void mglRenderGetTexImage(
    GLMContext context, Texture *texture, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice)
{
    if (!context) {
        return;
    }
    mglRendererObjCGetTexImage(context, texture, pixel_bytes, bytes_per_row,
                               bytes_per_image, x, y, width, height, format,
                               type, level, slice);
}

extern "C" void mglRenderTexSubImage(
    GLMContext context, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    if (!context) {
        return;
    }
    mglRendererObjCTexSubImage(context, texture, buffer, source_offset,
                                 source_pitch, source_image_size, source_size,
                                 slice, level, width, height, depth, x_offset,
                                 y_offset, z_offset);
}

extern "C" bool mglRenderTexSubImageBytes(
    GLMContext context, Texture *texture, const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    if (!context) {
        return false;
    }
    return mglRendererObjCTexSubImageBytes(context, texture, bytes, bytes_size,
                                           source_offset, source_pitch,
                                           source_image_size, slice, level,
                                           width, height, depth, x_offset,
                                           y_offset, z_offset);
}

extern "C" void mglRenderCopyTexSubImage(
    GLMContext context, Texture *texture, uint32_t slice, int32_t level,
    int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    if (!context) {
        return;
    }
    mglRendererObjCCopyTexSubImage(context, texture, slice, level, x_offset,
                                   y_offset, x, y, width, height);
}

extern "C" void mglRenderCopyImageSubData(
    GLMContext context, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth)
{
    if (!context) {
        return;
    }
    mglRendererObjCCopyImageSubData(context, source_texture, source_level,
                                    source_x, source_y, source_z,
                                    destination_texture, destination_level,
                                    destination_x, destination_y,
                                    destination_z, width, height, depth);
}
