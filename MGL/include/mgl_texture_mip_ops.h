/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_mip_ops.h - C home of the mipmap / image-unit / texbuffer leaves
 * of MGLRenderer+Texture.m (P0-1, log 194):
 *
 *   -mtlGenerateMipmaps:forTexture:
 *   -flushImageUnitSlice:unit:
 *   -prepareImageUnitSlice:unit:
 *   -syncTextureBufferFromImage:tex:
 *   -logMTLTextureMipDiagnostics:metal:effectiveMipLevels:
 */

#ifndef MGL_TEXTURE_MIP_OPS_H
#define MGL_TEXTURE_MIP_OPS_H

#include "glm_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Texture_t;

void mglTextureGenerateMipmaps(void *renderer, GLMContext glm_ctx,
                               struct Texture_t *tex);
void mglTextureSyncBufferFromImage(void *renderer, GLMContext glm_ctx,
                                   struct Texture_t *tex);
void mglTexturePrepareImageUnitSlice(void *renderer, GLMContext glm_ctx,
                                     uint32_t unit);
void mglTextureFlushImageUnitSlice(void *renderer, GLMContext glm_ctx,
                                   uint32_t unit);
void mglTextureLogMipDiagnostics(void *renderer, struct Texture_t *tex,
                                 void *texture, uint32_t effective_mip_levels);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_MIP_OPS_H */
