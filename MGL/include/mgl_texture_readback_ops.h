/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_readback_ops.h - the C homes of MGLRenderer(Texture)'s readback
 * family (P0-1, log 181).  The four methods used to read the renderer's device
 * and command-buffer owner through their Objective-C ivars; the C entries take
 * the renderer and read the state areas instead.
 */

#ifndef MGL_TEXTURE_READBACK_OPS_H
#define MGL_TEXTURE_READBACK_OPS_H

#include "glm_context.h"
#include "mgl_region_value.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -readbackStageAndWaitTexture:sourceLevel:sourceSlice:sourceDepthPlane:
 *  copyOrigin:copySize:stagingBytesPerRow:stagingSize:reason:logKind:success:
 *  Returns the staging buffer on success (ownership passes to the caller, which
 *  releases it with mglReleaseMetalObjNoNull once it has read the contents). */
void *mglTextureReadbackStageAndWait(void *renderer, void *sourceTexture,
                                     uint64_t sourceLevel, uint64_t sourceSlice,
                                     uint64_t sourceDepthPlane,
                                     MGLOriginValue copyOrigin,
                                     MGLSizeValue copySize,
                                     uint64_t stagingBytesPerRow,
                                     uint64_t stagingSize, const char *reason,
                                     const char *logKind, int *outSuccess);

int mglTextureReadColorAsBGRA8(void *renderer, void *sourceTexture,
                               uint64_t sourceLevel, uint64_t sourceSlice,
                               uint64_t sourceDepthPlane, void *pixelBytes,
                               uint64_t bytesPerRow, uint64_t bytesPerImage,
                               MGLRegionValue region, const char *reason);

int mglTextureReadDepthAsFloat(void *renderer, void *sourceTexture,
                               uint64_t sourceLevel, uint64_t sourceSlice,
                               uint64_t sourceDepthPlane, void *pixelBytes,
                               uint64_t bytesPerRow, uint64_t bytesPerImage,
                               MGLRegionValue region, const char *reason);

/* -mtlReadDepthPixels:pixelBytes:bytesPerRow:bytesPerImage:fromRegion: */
void mglTextureReadDepthPixels(void *renderer, GLMContext glm_ctx,
                               void *pixelBytes, uint64_t bytesPerRow,
                               uint64_t bytesPerImage, MGLRegionValue region);

/* -mtlReadIntegerPixels:pixelBytes:bytesPerRow:bytesPerImage:fromRegion:
 *  format:type: */
void mglTextureReadIntegerPixels(void *renderer, GLMContext glm_ctx,
                                 void *pixelBytes, uint64_t bytesPerRow,
                                 uint64_t bytesPerImage, MGLRegionValue region,
                                 GLenum format, GLenum type);

/* -mtlGetTexImage:tex:pixelBytes:bytesPerRow:bytesPerImage:fromRegion:format:
 *  type:mipmapLevel:slice: */
void mglTextureGetTexImage(void *renderer, GLMContext glm_ctx, Texture *tex,
                           void *pixelBytes, uint64_t bytesPerRow,
                           uint64_t bytesPerImage, MGLRegionValue region,
                           GLenum format, GLenum type, uint64_t level,
                           uint64_t slice);

/* -mtlReadDrawable:pixelBytes:bytesPerRow:bytesPerImage:fromRegion: */
void mglTextureReadDrawable(void *renderer, GLMContext glm_ctx,
                            void *pixelBytes, uint64_t bytesPerRow,
                            uint64_t bytesPerImage, MGLRegionValue region);

int mglTextureReadIntegerAsRGBA32(void *renderer, void *sourceTexture,
                                  void *pixelBytes, uint64_t bytesPerRow,
                                  uint64_t bytesPerImage, MGLRegionValue region,
                                  uint64_t outputComponents,
                                  uint64_t outputComponentBytes,
                                  const int *componentMap, GLenum packedType,
                                  uint64_t mipmapLevel, uint64_t mtlSlice,
                                  int isRenderTarget);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_READBACK_OPS_H */
