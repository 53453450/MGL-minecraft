/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_sampler_compat.h
 * MGL
 *
 * Sampler Compatibility Subsystem.
 *
 * Bridges the semantic gap between OpenGL sampler/resource semantics and
 * Metal sampler binding.  Covers several spec-compliance areas:
 *
 *   - Program shader resource queries (by name, by image dim, by Metal
 *     binding).  These are pure queries over Program->shader_resources_list
 *     and have no dependency on the renderer instance.
 *   - Sampler-like resource classification: GL sampler/uniform-constant
 *     resources that must be bound to Metal texture+sampler pairs, including
 *     heuristics for resources lowered to non-obvious MSL by the AIR
 *     backend (e.g. Minecraft CloudFaces texel buffer → texture2d<int>).
 *   - Binding-trace gating: identify programs whose bind-time state changes
 *     are worth tracing for debugging (ChunkSection/Sampler1/Sampler2).
 *
 * This module is pure specification-compliance machinery: every OpenGL
 * program that uses the corresponding GL sampler features needs these
 * translations when running on Metal, regardless of application.
 */

#ifndef MGL_SAMPLER_COMPAT_H
#define MGL_SAMPLER_COMPAT_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* === Program shader resource queries === */

/* Returns true if `program` has any sampled/separate/storage image resource
 * whose image_dim equals `imageDim` (cube = 3). */
bool mglProgramHasImageDim(Program *program, GLuint imageDim);

/* Returns true if `program` has a resource of the given stage/type whose
 * name matches.  Safe-guards against NULL program/name and out-of-range
 * stage/type. */
bool mglProgramHasResourceName(Program *program,
                               int stage,
                               int type,
                               const char *name);

/* Returns true if `program` has a resource of any stage/type whose name
 * matches. */
bool mglProgramHasAnyResourceName(Program *program, const char *name);

/* Returns true if `program` has a resource of the given stage/type whose
 * name matches.  Similar to mglProgramHasResourceName but without the
 * resources->list NULL guard — kept for callers that rely on the original
 * behavior. */
bool mglProgramHasResourceNamed(Program *program,
                                int stage,
                                int type,
                                const char *name);

/* === Binding-trace gating ===
 *
 * Identifies programs whose bind-time state changes are worth tracing for
 * debugging Minecraft rendering issues (ChunkSection terrain, Sampler1/2
 * entity textures). */
bool mglProgramNeedsBindingTrace(Program *program);

/* === Sampler-like resource classification === */

/* Heuristic: does the resource name look like a GL sampler uniform that
 * the AIR backend lowers to a Metal texture+sampler pair?  Covers names
 * containing "Sampler" and the Minecraft "CloudFaces" texel-buffer
 * workaround. */
bool mglRendererSamplerNameLooksSamplerLike(const char *name);

/* Heuristic: does the shader resource look like a sampler that must be
 * bound to a Metal texture+sampler pair?  Considers resource type and,
 * for _UNIFORM_CONSTANT_RES, image_dim / uniform_location
 * / name heuristics. */
bool mglRendererResourceLooksSamplerLike(const MGLShaderResource *res, int resType);

/* Finds the MGLShaderResource for a given Metal binding in a stage, considering
 * only sampler-like resources.  Returns NULL if not found.  Used by the
 * texture-unit resolution path. */
MGLShaderResource *mglFindSamplerResourceForMetalBinding(Program *program,
                                                     int stage,
                                                     GLuint metalBinding);

/* Resolves a sampler-like reflected resource to its GL texture unit. */
GLint mglResolveSamplerResourceUnit(Program *program,
                                    MGLShaderResource *res,
                                    int stage,
                                    int resType);

/* Returns true if `program` has any sampler-like resource (across all stages
 * and the 5 sampler-like resource types) whose resolved GL texture unit
 * equals `unit`.  Resolution mirrors MGLRenderer
 * -textureUnitForSampledResource:metalBinding:stage: (per-resource
 * sampler_unit -> stage array -> global array -> default 0).
 *
 * Used by the WAR hazard tracker to avoid false-positive flushes when a
 * texture is bound to a unit the program never samples (e.g. an FBO color
 * attachment texture left bound after glTexImage2D).
 *
 * Returns false if `program` is NULL or has no sampler-like resources. */
bool mglProgramSamplesTextureUnit(Program *program, GLuint unit);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SAMPLER_COMPAT_H */
