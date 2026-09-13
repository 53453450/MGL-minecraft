/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_sampler.h — sampler materialization moved out of
 * MGLRenderer+Texture.m (P0-1).  The bodies were already C; the ObjC parts
 * were the method shells, and the fallback sampler reached C callers through a
 * shim port that this move retires.
 *
 * OWNERSHIP: mglTextureCreateSamplerForTexParam returns +1 (the caller owns the
 * reference); mglTextureFallbackSamplerState returns a borrowed reference owned
 * by the backend's fallback-resource cache.
 */

#ifndef MGL_TEXTURE_SAMPLER_H
#define MGL_TEXTURE_SAMPLER_H

#include "glm_context.h"
#include "mgl_types_texture.h"   /* TextureParameter */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sampler state for a GL texture parameter set (+1 reference, or NULL). */
void *mglTextureCreateSamplerForTexParam(const TextureParameter *tex_param,
                                         uint32_t target);

/* Default sampler of the renderer, cached in the backend (borrowed). */
void *mglTextureFallbackSamplerState(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_SAMPLER_H */
