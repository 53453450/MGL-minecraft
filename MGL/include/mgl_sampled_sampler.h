/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_sampled_sampler.h — the sampled-sampler materialize leaf moved out of
 * MGLRenderer+BindingState.m (P0-1, log 151).
 *
 * Was -materializeSampledSamplerForTexture:textureUnit:defaultSampler:
 * forceDefault:samplerTarget:programName:spirvBinding:stage:texture:.
 * Returns a BORROWED handle (the renderer's GL sampler, the texture's own
 * params sampler, or `default_sampler`) — the method's +0 `id` return.
 */

#ifndef MGL_SAMPLED_SAMPLER_H
#define MGL_SAMPLED_SAMPLER_H

#include <stdint.h>

#include "glm_context.h"        /* GLMContext, GLuint */
#include "mgl_types_texture.h"  /* Texture */

#ifdef __cplusplus
extern "C" {
#endif

/* Compat fallback plan for a sampled texture whose type/kind does not match the
 * shader.  Was -applySampledCompatFallbackPlan:texture:expectedType:
 * expectedKind:stage:programName:spirvBinding:sampleProgram:usedFallbackOut:.
 * Returns a BORROWED handle (the passed texture, or the cached fallback
 * texture) — the method's +0 `id` return. */
void *mglSampledCompatFallbackPlan(void *renderer, Texture *ptr, void *texture,
                                   uint32_t expected_type, uint32_t expected_kind,
                                   const char *stage, GLuint program_name,
                                   GLuint spirv_binding, void *sample_program,
                                   int *used_fallback_out);

void *mglSampledSamplerMaterialize(void *renderer, Texture *ptr,
                                   GLuint texture_unit, void *default_sampler,
                                   int force_default, GLuint sampler_target,
                                   GLuint program_name, GLuint spirv_binding,
                                   const char *stage, void *texture);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SAMPLED_SAMPLER_H */
