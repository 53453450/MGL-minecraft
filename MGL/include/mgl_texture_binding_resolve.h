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

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_BINDING_RESOLVE_H */
