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


/* The dirty-CPU-data upload tree moved in log 196. */
int mglTextureUploadDirty(void *renderer, Texture *tex, void *texture,
                          uint32_t pixel_format, uint32_t num_faces,
                          uint32_t upload_level_count, int is_array,
                          int texture1DBackedBy2D,
                          int texture1DArrayBackedBy2DArray, uint32_t tex_type,
                          int *out_all_levels_uploaded);
int mglTextureUploadDirty3DLevel(void *renderer, Texture *tex, void *texture,
                                 uint32_t pixel_format, int face, int level,
                                 uint64_t width, uint64_t height, uint64_t depth,
                                 int *out_skipped);
int mglTextureUploadDirtyNon3DLevel(void *renderer, Texture *tex, void *texture,
                                    uint32_t pixel_format, int face, int level,
                                    uint64_t width, uint64_t height,
                                    uint64_t depth, int is_array,
                                    int texture1DArrayBackedBy2DArray,
                                    uint32_t tex_type, int *out_skipped);


/* -createMTLTextureFromGLTexture: moved in log 197. */
void *mglTextureCreateFromGLTexture(void *renderer, Texture *tex);


/* The texSubImage trio moved in log 198. */
struct Buffer_t;
int mglTextureSubImage(void *renderer, GLMContext glm_ctx, Texture *tex,
                       struct Buffer_t *buf, uint64_t src_offset,
                       uint64_t src_pitch, uint64_t src_image_size,
                       uint64_t src_size, uint32_t slice, uint32_t level,
                       uint64_t width, uint64_t height, uint64_t depth,
                       uint64_t xoffset, uint64_t yoffset, uint64_t zoffset);
int mglTextureSubImageBytes(void *renderer, GLMContext glm_ctx, Texture *tex,
                            const void *bytes, uint64_t bytes_size,
                            uint64_t src_offset, uint64_t src_pitch,
                            uint64_t src_image_size, uint32_t slice,
                            uint32_t level, uint64_t width, uint64_t height,
                            uint64_t depth, uint64_t xoffset, uint64_t yoffset,
                            uint64_t zoffset);


/* The last two methods of MGLRenderer+Texture.m moved in log 199. */
struct TextureLevel_t;
void mglTextureTraceSampledReadback(void *renderer, void *texture, Texture *gl_tex,
                                    struct TextureLevel_t *level0,
                                    uint32_t program, uint32_t binding,
                                    const char *stage, const char *reason,
                                    uint64_t hit);
void *mglTextureCreateFallback(void *renderer, Texture *tex);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_UPLOAD_OPS_H */
