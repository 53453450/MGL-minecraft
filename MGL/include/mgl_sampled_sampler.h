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

#include <stdbool.h>
#include <stdint.h>

#include "glm_context.h"        /* GLMContext, GLuint */
#include "mgl_types_texture.h"  /* Texture */

#ifdef __cplusplus
extern "C" {
#endif

/* Depth-texture recovery for fragment sampling.  Was
 * -recoverFragmentSampledDepthTexture:texture:sampledName:spirvBinding:
 * textureUnit:expectedType:expectedKind:fragmentProgramName:
 * suppressMissingTextureFallback:usedFallbackTexture:.  `ptr_ptr` is an in/out
 * Texture handle, `texture_ptr` an in/out BORROWED texture handle, and the two
 * BOOL out-params are ints in C.  Returns false when the caller must abandon
 * the stage. */
bool mglSampledRecoverFragmentDepthTexture(
    void *renderer, Texture **ptr_ptr, void **texture_ptr,
    const char *sampled_name, GLuint spirv_binding, GLuint texture_unit,
    uint32_t expected_type, uint32_t expected_kind,
    GLuint fragment_program_name, int *suppress_missing_ptr,
    int *used_fallback_ptr);

/* Separate samplers (fragment) plus array-element textures/samplers for both
 * stages.  Was -bindSeparateSamplersAndArrayTextures:fragmentProgram:
 * fragmentProgramName:vertexProgramName:defaultSampler:bindCall:traceBind:
 * separateSamplerCount:boundSeparateSamplers:.  Returns false when the caller
 * must abandon the stage. */
bool mglSampledBindSeparateSamplersAndArrayTextures(
    void *renderer, Program *vertex_program, Program *fragment_program,
    GLuint fragment_program_name, GLuint vertex_program_name,
    void *default_sampler, uint64_t bind_call, int trace_bind,
    GLuint *separate_sampler_count, GLuint *bound_separate_samplers);

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

/* The sampled render-target copy plan.  Was -applySampledRenderTargetCopyPlan:
 * texture:sampleProgram:expectedType:expectedKind:usedTypeFallback:stage:
 * programName:spirvBinding:textureUnit:sampledName:usedSampledCopyOut:
 * directTextureForTrace:sampledCopyForTrace:.  `texture_ptr` is an in/out
 * BORROWED handle (the .m's `id *` out-param), the two trace handles are
 * out-only borrowed handles, and `used_sampled_copy_out` is a BOOL out-param
 * (int in C).  Returns false when the caller must abandon the stage. */
bool mglSampledRenderTargetCopyPlan(
    void *renderer, Texture *ptr, void **texture_ptr, Program *sample_program,
    uint32_t expected_type, uint32_t expected_kind, int used_type_fallback,
    const char *stage, GLuint program_name, GLuint spirv_binding,
    GLuint texture_unit, const char *sampled_name, int *used_sampled_copy_out,
    void **direct_texture_for_trace, void **sampled_copy_for_trace);

void *mglSampledSamplerMaterialize(void *renderer, Texture *ptr,
                                   GLuint texture_unit, void *default_sampler,
                                   int force_default, GLuint sampler_target,
                                   GLuint program_name, GLuint spirv_binding,
                                   const char *stage, void *texture);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SAMPLED_SAMPLER_H */
