/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_sampled_fallback.h — the sampled-texture fallback chain moved out of
 * MGLRenderer+Texture.m (P0-1, log 132).
 *
 *   -fallbackSampledTextureForExpectedType:dataKind: -> mglSampledFallbackTextureForExpectedType
 *   -fallbackSampledTextureForExpectedType:          -> mglSampledFallbackTextureForType
 *   -fallbackSampledTexture                          -> mglSampledFallbackTexture
 *   -fallbackCubeSampledTexture                      -> mglSampledFallbackCubeTexture
 *   -fallbackTextureBufferSampledTexture             -> mglSampledFallbackTextureBuffer
 *
 * The chain is the two remaining callees of MGLRenderer+BindingState.m's sampled
 * path that had no port, so it becomes C here (no port added) before that file
 * is converted.
 *
 * OWNERSHIP: every entry returns a BORROWED texture, exactly like the methods
 * did (ARC returned +0); the created +1 is handed to the backend cache, which
 * retains it, and a creation failure returns NULL.
 */

#ifndef MGL_SAMPLED_FALLBACK_H
#define MGL_SAMPLED_FALLBACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed 1x1 (or 1-texel) fallback texture for the expected GL texture type
 * and data kind, or NULL. */
void *mglSampledFallbackTextureForExpectedType(void *renderer,
                                               uint32_t expected_type,
                                               uint32_t data_kind);

/* Borrowed fallback texture chosen by expected type alone. */
void *mglSampledFallbackTextureForType(void *renderer, uint32_t expected_type);

/* Borrowed 2D / cube / texture-buffer fallbacks. */
void *mglSampledFallbackTexture(void *renderer);
void *mglSampledFallbackCubeTexture(void *renderer);
void *mglSampledFallbackTextureBuffer(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SAMPLED_FALLBACK_H */
