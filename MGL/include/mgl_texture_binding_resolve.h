/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_binding_resolve.h — which texture a sampled resource reads.
 *
 * Formerly -[MGLRenderer textureForSampledResource:metalBinding:stage:
 * expectedType:textureUnit:] and its :expectedType: wrapper (174 + 15 lines in
 * MGLRenderer+Texture.m).  The bodies were pure C over the GL state; keeping
 * them in their own translation unit keeps the state-only binding modules
 * (mgl_binding_texture.c and friends) independently linkable for their
 * harnesses.
 */

#ifndef MGL_TEXTURE_BINDING_RESOLVE_H
#define MGL_TEXTURE_BINDING_RESOLVE_H

#include "glm_context.h"
#include "mgl_types_texture.h"
#include "mgl_shader_resource.h"   /* MGLShaderResource */

#ifdef __cplusplus
extern "C" {
#endif

/* Texture the sampled resource reads for an explicit texture unit. */
Texture *mglTextureForSampledResource(GLMContext ctx,
                                      MGLShaderResource *sampled_resource,
                                      GLuint metal_binding, int stage,
                                      uint32_t expected_type,
                                      GLuint texture_unit);

/* Same, resolving the texture unit from the program bound to `stage`. */
Texture *mglTextureForSampledResourceForStage(GLMContext ctx,
                                              MGLShaderResource *sampled_resource,
                                              GLuint metal_binding, int stage,
                                              uint32_t expected_type);

/* What a sampled-resource resolve needs to know about the texture and the
 * sampler bound to its unit, by value, so the Metal-side encoders do not read
 * the GL objects' state flags themselves. */
typedef struct MGLTextureSampledFacts {
    int has_tex;
    int has_mtl;                /* the texture has a Metal object */
    int dirty;                  /* pending level / data / access / param work */
    int is_rt;                  /* attached to a framebuffer */
    void *mtl_texture;          /* the Metal object (borrowed) */
    void *param_sampler_mtl;    /* the texture's own sampler state (borrowed) */
} MGLTextureSampledFacts;

void mglTextureSampledFacts(const Texture *tex, MGLTextureSampledFacts *out);

/* The sampler object's Metal state when it is up to date, else NULL. */
void *mglSamplerCleanMtl(const Sampler *sampler);

/* The Metal sampler kept for `sampler`, or NULL. */
void *mglSamplerBackendState(const Sampler *sampler);
/* Keep `state` (a reference the caller hands over) as the sampler's Metal
 * sampler; the previous one is released. */
void mglSamplerBackendSetState(Sampler *sampler, void *state);
/* Release the Metal sampler but keep the backend struct. */
void mglSamplerBackendResetState(Sampler *sampler);
/* Release everything the backend keeps for `sampler`. */
void mglSamplerBackendRelease(Sampler *sampler);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_BINDING_RESOLVE_H */
