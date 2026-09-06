/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * GL/ObjC compatibility bridge declarations (ARCHITECTURE_AUDIT R4).
 * Resource-owner backend must not grow new direct ObjC selector calls;
 * draw/transfer/compute entry points live behind this bridge until the
 * MetalExecutor fully owns encoding.
 */

#ifndef MGL_RENDERER_COMPAT_BRIDGE_H
#define MGL_RENDERER_COMPAT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "glm_context.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRendererCompatDispatchCompute(GLMContext context,
                                      unsigned int groups_x,
                                      unsigned int groups_y,
                                      unsigned int groups_z);
void mglRendererCompatDispatchComputeIndirect(GLMContext context,
                                              intptr_t indirect);
void mglRendererCompatBindTexture(GLMContext context, Texture *texture);
void mglRendererCompatFlushDrawBuffer(GLMContext context);
void mglRendererCompatSwapBuffers(GLMContext context);
void mglRendererCompatClearBuffer(GLMContext context, unsigned int type,
                                  unsigned int mask);
void mglRendererCompatDrawArrays(GLMContext context, uint32_t mode,
                                 int32_t first, int32_t count);
void mglRendererCompatDrawElements(GLMContext context, uint32_t mode,
                                   int32_t count, uint32_t type,
                                   const void *indices);
void mglRendererCompatBlitFramebuffer(GLMContext context, int src_x0,
                                      int src_y0, int src_x1, int src_y1,
                                      int dst_x0, int dst_y0, int dst_x1,
                                      int dst_y1, unsigned int mask,
                                      unsigned int filter);
void mglRendererCompatReadDrawable(GLMContext context, void *pixel_bytes,
                                   uint32_t bytes_per_row,
                                   uint32_t bytes_per_image, int32_t x,
                                   int32_t y, int32_t width, int32_t height);
void mglRendererCompatReadIntegerPixels(GLMContext context, void *pixel_bytes,
                                        uint32_t bytes_per_row,
                                        uint32_t bytes_per_image, int32_t x,
                                        int32_t y, int32_t width,
                                        int32_t height, uint32_t format,
                                        uint32_t type);
void mglRendererCompatReadDepthPixels(GLMContext context, void *pixel_bytes,
                                      uint32_t bytes_per_row,
                                      uint32_t bytes_per_image, int32_t x,
                                      int32_t y, int32_t width, int32_t height);
void mglRendererCompatGetTexImage(GLMContext context, Texture *texture,
                                  void *pixel_bytes, uint32_t bytes_per_row,
                                  uint32_t bytes_per_image, int32_t x,
                                  int32_t y, int32_t width, int32_t height,
                                  uint32_t format, uint32_t type,
                                  uint32_t level, uint32_t slice);
void mglRendererCompatGenerateMipmaps(GLMContext context, Texture *texture);
void mglRendererCompatTexSubImage(GLMContext context, Texture *texture,
                                  Buffer *buffer, size_t source_offset,
                                  size_t source_pitch,
                                  size_t source_image_size, size_t source_size,
                                  uint32_t slice, uint32_t level, size_t width,
                                  size_t height, size_t depth, size_t x_offset,
                                  size_t y_offset, size_t z_offset);
bool mglRendererCompatTexSubImageBytes(GLMContext context, Texture *texture,
                                       const void *bytes, size_t bytes_size,
                                       size_t source_offset, size_t source_pitch,
                                       size_t source_image_size, uint32_t slice,
                                       uint32_t level, size_t width,
                                       size_t height, size_t depth,
                                       size_t x_offset, size_t y_offset,
                                       size_t z_offset);
void mglRendererCompatCopyTexSubImage(GLMContext context, Texture *texture,
                                      uint32_t slice, int32_t level,
                                      int32_t x_offset, int32_t y_offset,
                                      int32_t x, int32_t y, int32_t width,
                                      int32_t height);
void mglRendererCompatCopyImageSubData(
    GLMContext context, Texture *source_texture, int32_t source_level,
    int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_COMPAT_BRIDGE_H */
