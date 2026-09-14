/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_create_ops.h - C homes of the MGLRenderer(Texture) creation
 * helpers that had no self sends (P0-1, log 182).
 */

#ifndef MGL_TEXTURE_CREATE_OPS_H
#define MGL_TEXTURE_CREATE_OPS_H

#include "glm_context.h"
#include "mgl_region_value.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Twin of the .m's mglTextureReplaceRegion (log 183); returns 1 on success
 * instead of raising.  It is also the C home of the region write path. */
int mglTextureReplaceRegionValue(void *texture, MGLRegionValue region,
                                 uint64_t level, uint64_t slice,
                                 const void *bytes, uint64_t bytesPerRow,
                                 uint64_t bytesPerImage, int useSlice);

/* -checkTextureCompleteness:texType:numFaces:effectiveMipmapLevels:
 *  storageMipmapped: */
int mglTextureCheckCompleteness(void *tex, uint32_t tex_type, unsigned num_faces,
                                unsigned int *outEffectiveMipmapLevels,
                                int *outStorageMipmapped);

/* -uploadPackedDepthStencilStencilPlane:texName:bytes:width:height:
 *  bytesPerRow:level:slice:xorigin:yorigin: */
int mglTextureUploadPackedDepthStencilStencilPlane(
    void *texture, unsigned int texName, const void *packedBytes,
    uint64_t width, uint64_t height, uint64_t bytesPerRow, uint64_t level,
    uint64_t slice, uint64_t xorigin, uint64_t yorigin);

/* -createMTLTexelBufferTexture:.  Returns the new Metal texture (+1, the
 * caller releases it) or NULL. */
void *mglTextureCreateMTLTexelBufferTexture(void *renderer, void *tex);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_CREATE_OPS_H */
