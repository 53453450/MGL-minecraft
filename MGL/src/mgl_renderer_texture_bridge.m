/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC texture dispatch bridge — entry from C++ facade to MGLRenderer (Texture).
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Texture_Private.h"
#include "mgl_render.h"

static MGLRegionValue mglRendererObjCRegion(int32_t x, int32_t y,
                                            int32_t width, int32_t height)
{
    return (MGLRegionValue){
        .origin = {(int64_t)x, (int64_t)y, 0},
        .size = {(uint64_t)width, (uint64_t)height, 1u},
    };
}

void mglRendererObjCReadDrawable(GLMContext glm_ctx, void *pixel_bytes,
                               uint32_t bytes_per_row, uint32_t bytes_per_image,
                               int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlReadDrawable:glm_ctx pixelBytes:pixel_bytes
                bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                 fromRegion:mglRendererObjCRegion(x, y, width, height)];
}

void mglRendererObjCReadIntegerPixels(GLMContext glm_ctx, void *pixel_bytes,
                                      uint32_t bytes_per_row,
                                      uint32_t bytes_per_image,
                                      int32_t x, int32_t y, int32_t width,
                                      int32_t height, uint32_t format,
                                      uint32_t type)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlReadIntegerPixels:glm_ctx pixelBytes:pixel_bytes
                       bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                        fromRegion:mglRendererObjCRegion(x, y, width, height)
                            format:format type:type];
}

void mglRendererObjCReadDepthPixels(GLMContext glm_ctx, void *pixel_bytes,
                                    uint32_t bytes_per_row,
                                    uint32_t bytes_per_image,
                                    int32_t x, int32_t y, int32_t width,
                                    int32_t height)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlReadDepthPixels:glm_ctx pixelBytes:pixel_bytes
                     bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                      fromRegion:mglRendererObjCRegion(x, y, width, height)];
}

void mglRendererObjCGetTexImage(GLMContext glm_ctx, Texture *texture,
                                void *pixel_bytes, uint32_t bytes_per_row,
                                uint32_t bytes_per_image, int32_t x, int32_t y,
                                int32_t width, int32_t height, uint32_t format,
                                uint32_t type, uint32_t level, uint32_t slice)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlGetTexImage:glm_ctx tex:texture pixelBytes:pixel_bytes
                 bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                  fromRegion:mglRendererObjCRegion(x, y, width, height)
                      format:format type:type mipmapLevel:level slice:slice];
}

void mglRendererObjCGenerateMipmaps(GLMContext glm_ctx, Texture *texture)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlGenerateMipmaps:glm_ctx forTexture:texture];
}

void mglRendererSyncTextureBufferFromImage(GLMContext glm_ctx, Texture *texture)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx || !texture) return;
    [renderer syncTextureBufferFromImage:glm_ctx tex:texture];
}

void mglRendererPrepareImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer prepareImageUnitSlice:glm_ctx unit:unit];
}

void mglRendererFlushImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer flushImageUnitSlice:glm_ctx unit:unit];
}

void mglRendererObjCTexSubImage(GLMContext glm_ctx, Texture *texture,
                                Buffer *buffer, size_t source_offset,
                                size_t source_pitch, size_t source_image_size,
                                size_t source_size, uint32_t slice,
                                uint32_t level, size_t width, size_t height,
                                size_t depth, size_t x_offset, size_t y_offset,
                                size_t z_offset)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlTexSubImage:glm_ctx tex:texture buf:buffer
                  src_offset:source_offset src_pitch:source_pitch
              src_image_size:source_image_size src_size:source_size
                       slice:slice level:level width:width height:height
                       depth:depth xoffset:x_offset yoffset:y_offset
                     zoffset:z_offset];
}

bool mglRendererObjCTexSubImageBytes(GLMContext glm_ctx, Texture *texture,
                                     const void *bytes, size_t bytes_size,
                                     size_t source_offset, size_t source_pitch,
                                     size_t source_image_size, uint32_t slice,
                                     uint32_t level, size_t width, size_t height,
                                     size_t depth, size_t x_offset,
                                     size_t y_offset, size_t z_offset)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return false;
    return [renderer mtlTexSubImageBytes:glm_ctx tex:texture
                                    bytes:bytes bytesSize:bytes_size
                               src_offset:source_offset src_pitch:source_pitch
                           src_image_size:source_image_size
                                    slice:slice level:level
                                    width:width height:height depth:depth
                                  xoffset:x_offset yoffset:y_offset
                                  zoffset:z_offset];
}

void mglRendererObjCCopyTexSubImage(GLMContext glm_ctx, Texture *texture,
                                    uint32_t slice, int32_t level,
                                    int32_t x_offset, int32_t y_offset,
                                    int32_t x, int32_t y, int32_t width,
                                    int32_t height)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlCopyTexSubImage:glm_ctx tex:texture slice:slice
                    mipmapLevel:level xoffset:x_offset yoffset:y_offset
                              x:x y:y width:width height:height];
}

void mglRendererObjCCopyImageSubData(GLMContext glm_ctx,
                                      Texture *source_texture,
                                      int32_t source_level, int32_t source_x,
                                      int32_t source_y, int32_t source_z,
                                      Texture *destination_texture,
                                      int32_t destination_level,
                                      int32_t destination_x,
                                      int32_t destination_y,
                                      int32_t destination_z, int32_t width,
                                      int32_t height, int32_t depth)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlCopyImageSubData:glm_ctx srcTexture:source_texture
                         srcLevel:source_level srcX:source_x srcY:source_y
                             srcZ:source_z dstTexture:destination_texture
                         dstLevel:destination_level dstX:destination_x
                             dstY:destination_y dstZ:destination_z
                            width:width height:height depth:depth];
}
