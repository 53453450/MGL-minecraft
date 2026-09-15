/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_upload_ops.h - C homes of the texture upload leaves (P0-1,
 * log 183): the slice upload (replaceRegion / blit routes) and the dedicated
 * command-buffer copy.  The Objective-C callers keep calling the C entries.
 */

#ifndef MGL_TEXTURE_UPLOAD_OPS_H
#define MGL_TEXTURE_UPLOAD_OPS_H

#include "glm_context.h"
#include "mgl_region_value.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -copyTextureUploadWithDedicatedCommandBuffer:… */
int mglTextureCopyUploadWithDedicatedCommandBuffer(
    void *renderer, void *sourceBuffer, uint64_t sourceOffset,
    uint64_t sourceBytesPerRow, uint64_t sourceBytesPerImage,
    uint64_t sourceLayerStride, uint64_t layerCount, MGLSizeValue sourceSize,
    void *texture, uint64_t destinationSlice, uint64_t destinationLevel,
    MGLOriginValue destinationOrigin, const char *reason);

/* -uploadTextureSliceViaBlit:… */
int mglTextureUploadSliceViaBlit(void *renderer, void *texture,
                                 unsigned int texName, GLenum texTarget,
                                 const void *bytes, uint64_t bytesPerRow,
                                 uint64_t bytesPerImage, uint64_t width,
                                 uint64_t height, uint64_t depth, uint64_t level,
                                 uint64_t slice);


/* The upload tree moved in log 195. */
int mglTextureUploadFullCPUData(void *renderer, Texture *tex, void *texture,
                                const char *reason);
int mglTextureEncodeBytesUpload(void *renderer, Texture *tex, void *buffer,
                                uint64_t source_offset,
                                uint64_t source_bytes_per_row,
                                uint64_t source_bytes_per_image, uint64_t width,
                                uint64_t height, uint64_t depth, uint64_t slice,
                                uint64_t level, uint64_t xoffset,
                                uint64_t yoffset, uint64_t zoffset,
                                const char *reason);
void mglTextureReUploadExisting(void *renderer, Texture *tex, void *texture,
                                uint32_t pixel_format, uint32_t num_faces,
                                uint32_t upload_level_count, int is_array,
                                int texture1DBackedBy2D,
                                int texture1DArrayBackedBy2DArray,
                                uint32_t tex_type);
void mglTextureReUploadArrayLevel(void *renderer, Texture *tex, void *texture,
                                  uint32_t pixel_format, int face, int level,
                                  int texture1DArrayBackedBy2DArray,
                                  uint32_t tex_type);
void mglTextureFillSmallGradient(void *renderer, void *texture, Texture *tex);
void mglTextureFillSafeInitialContents(void *renderer, void *texture,
                                       Texture *tex, uint32_t pixel_format);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_UPLOAD_OPS_H */
