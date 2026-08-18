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
 * mgl_sampler_compat.m
 * MGL
 *
 * Implementation of the Sampler Compatibility Subsystem.
 *
 * See mgl_sampler_compat.h for the architectural rationale.  This module
 * owns the pure spec-compliance helpers for translating OpenGL sampler /
 * resource semantics to Metal binding:
 *   - Program shader resource queries (by name, image dim, Metal binding).
 *   - Sampler-like resource classification heuristics.
 *   - Binding-trace gating for debugging.
 *
 * The helpers here are pure: they do not touch the renderer ivar, the
 * command buffer, or the render encoder.  They operate only on the
 * Program / MGLShaderResource structures passed in as arguments.
 *
 * External dependencies:
 *   - Program / MGLShaderResource / MGLShaderResourceList types (glm_context.h).
 *   - MGL resource type constants (incl. MGL_MAX_SHADER_RESOURCES and
 *     the image-dim encoding) from the program reflection headers.
 *   - _MAX_SHADER_TYPES / MGL_MAX_SHADER_RESOURCES / _VERTEX_SHADER (glm_context.h).
 *   - TEXTURE_UNITS (glm_limits.h).
 */

#import "mgl_sampler_compat.h"
#import <Foundation/Foundation.h>
#import "mgl_trace_log.h"
#include <string.h>

/* === Program shader resource queries === */

bool mglProgramHasImageDim(Program *program, GLuint imageDim)
{
    if (!program) {
        return false;
    }

    const int resourceTypes[] = {
        _SAMPLED_IMAGE_RES,
        _SEPARATE_IMAGE_RES,
        _STORAGE_IMAGE_RES
    };

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t t = 0; t < sizeof(resourceTypes) / sizeof(resourceTypes[0]); t++) {
            int type = resourceTypes[t];
            if (type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
                continue;
            }
            MGLShaderResourceList *resources = &program->shader_resources_list[stage][type];
            for (GLuint i = 0; i < resources->count; i++) {
                if (resources->list[i].image_dim == imageDim) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool mglProgramHasResourceName(Program *program,
                               int stage,
                               int type,
                               const char *name)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        type < 0 || type >= MGL_MAX_SHADER_RESOURCES || !name) {
        return false;
    }

    MGLShaderResourceList *resources = &program->shader_resources_list[stage][type];
    for (GLuint i = 0; resources->list && i < resources->count; i++) {
        if (resources->list[i].name && strcmp(resources->list[i].name, name) == 0) {
            return true;
        }
    }

    return false;
}

bool mglProgramHasAnyResourceName(Program *program, const char *name)
{
    if (!program || !name) {
        return false;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        for (int type = 0; type < MGL_MAX_SHADER_RESOURCES; type++) {
            if (mglProgramHasResourceName(program, stage, type, name)) {
                return true;
            }
        }
    }

    return false;
}

bool mglProgramHasResourceNamed(Program *program,
                                int stage,
                                int type,
                                const char *name)
{
    if (!program || !name || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return false;
    }

    MGLShaderResourceList *resources = &program->shader_resources_list[stage][type];
    for (GLuint i = 0; i < resources->count; i++) {
        MGLShaderResource *res = &resources->list[i];
        if (res->name && strcmp(res->name, name) == 0) {
            return true;
        }
    }

    return false;
}

/* === Binding-trace gating === */

bool mglProgramNeedsBindingTrace(Program *program)
{
    if (!program || !mglTraceLogIsEnabled()) {
        return false;
    }

    return mglProgramHasAnyResourceName(program, "ChunkSection") ||
           mglProgramHasAnyResourceName(program, "Sampler1") ||
           mglProgramHasAnyResourceName(program, "Sampler2");
}

/* === Sampler-like resource classification === */

bool mglRendererSamplerNameLooksSamplerLike(const char *name)
{
    return name &&
           (strstr(name, "Sampler") ||
            !strcmp(name, "CloudFaces"));
}

bool mglRendererResourceLooksSamplerLike(const MGLShaderResource *res, int resType)
{
    if (!res) {
        return false;
    }

    switch (resType) {
        case _SAMPLED_IMAGE_RES:
        case _SEPARATE_IMAGE_RES:
        case _SEPARATE_SAMPLERS_RES:
        case _STORAGE_IMAGE_RES:
            return true;
        case _UNIFORM_CONSTANT_RES:
            return res->image_dim != 0u ||
                   res->uniform_location >= 0x4000 ||
                   mglRendererSamplerNameLooksSamplerLike(res->name);
        default:
            return false;
    }
}

MGLShaderResource *mglFindSamplerResourceForMetalBinding(Program *program,
                                                     int stage,
                                                     GLuint metalBinding)
{
    static const int samplerResourceTypes[] = {
        _UNIFORM_CONSTANT_RES,
        _SAMPLED_IMAGE_RES,
        _SEPARATE_IMAGE_RES,
        _SEPARATE_SAMPLERS_RES,
        _STORAGE_IMAGE_RES
    };

    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES || metalBinding >= TEXTURE_UNITS) {
        return NULL;
    }

    for (size_t rt = 0; rt < sizeof(samplerResourceTypes) / sizeof(samplerResourceTypes[0]); rt++) {
        int resType = samplerResourceTypes[rt];
        MGLShaderResourceList *resources = &program->shader_resources_list[stage][resType];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            MGLShaderResource *res = &resources->list[i];
            if (res->binding == metalBinding &&
                mglRendererResourceLooksSamplerLike(res, resType)) {
                return res;
            }
        }
    }

    return NULL;
}

/* Resolves the GL texture unit that `res` (in `stage` of `program`) samples,
 * mirroring MGLRenderer -textureUnitForSampledResource:metalBinding:stage:
 * but operating purely on the Program struct.  Returns the resolved unit
 * (0-based), or -1 if the resource is not sampler-like. */
GLint mglResolveSamplerResourceUnit(Program *program,
                                    MGLShaderResource *res,
                                    int stage,
                                    int resType)
{
    if (!program || !res) return -1;
    if (!mglRendererResourceLooksSamplerLike(res, resType)) return -1;

    /* 1. Per-resource explicit assignment (glUniform1i). */
    if (res->sampler_unit_explicit &&
        res->sampler_unit >= 0 &&
        res->sampler_unit < (GLint)TEXTURE_UNITS) {
        return res->sampler_unit;
    }

    GLuint metalBinding = res->binding;
    if (metalBinding >= TEXTURE_UNITS) {
        return (GLint)metalBinding;
    }

    bool stageValid = (stage >= 0 && stage < _MAX_SHADER_TYPES);
    bool stageExplicit = stageValid
        ? (program->sampler_units_explicit_by_stage[stage][metalBinding] == GL_TRUE)
        : false;
    bool globalExplicit = (program->sampler_units_explicit[metalBinding] == GL_TRUE);

    /* 2. Stage array explicit. */
    GLint unit = stageValid
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];
    if (stageExplicit && unit >= 0 && unit < (GLint)TEXTURE_UNITS) {
        return unit;
    }

    /* 3. Global array explicit. */
    unit = program->sampler_units[metalBinding];
    if (globalExplicit && unit >= 0 && unit < (GLint)TEXTURE_UNITS) {
        return unit;
    }

    /* 4. Default unit (stage then global fallback). */
    GLint defaultUnit = stageValid
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];
    if (defaultUnit < 0 || defaultUnit >= (GLint)TEXTURE_UNITS) {
        defaultUnit = program->sampler_units[metalBinding];
    }

    /* 5. Per-resource non-explicit (set by reflection, not glUniform1i). */
    if (!res->sampler_unit_explicit &&
        res->sampler_unit >= 0 &&
        res->sampler_unit < (GLint)TEXTURE_UNITS) {
        return res->sampler_unit;
    }

    if (defaultUnit >= 0 && defaultUnit < (GLint)TEXTURE_UNITS) {
        return defaultUnit;
    }

    /* 6. OpenGL default is unit 0. */
    return 0;
}

bool mglProgramSamplesTextureUnit(Program *program, GLuint unit)
{
    if (!program) return false;
    if (unit >= TEXTURE_UNITS) return false;

    /* Cached bitmap: built once per program and invalidated when sampler
     * bindings change (glUniform1i, relink).  Turns the per-unit
     * O(stages * resources) scan into an O(1) bit test.  The 3 hazard-scan
     * call sites iterate up to 128 units each, so this avoids ~19200
     * resource resolutions per draw in the worst case. */
    if (!program->sampled_texture_unit_mask_valid) {
        memset(program->sampled_texture_unit_mask, 0,
               sizeof(program->sampled_texture_unit_mask));

        static const int samplerResourceTypes[] = {
            _UNIFORM_CONSTANT_RES,
            _SAMPLED_IMAGE_RES,
            _SEPARATE_IMAGE_RES,
            _SEPARATE_SAMPLERS_RES,
            _STORAGE_IMAGE_RES
        };

        for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
            for (size_t rt = 0; rt < sizeof(samplerResourceTypes) / sizeof(samplerResourceTypes[0]); rt++) {
                int resType = samplerResourceTypes[rt];
                if (resType < 0 || resType >= MGL_MAX_SHADER_RESOURCES) continue;
                MGLShaderResourceList *resources = &program->shader_resources_list[stage][resType];
                for (GLuint i = 0; resources->list && i < resources->count; i++) {
                    GLint resolved = mglResolveSamplerResourceUnit(program,
                                                                   &resources->list[i],
                                                                   stage,
                                                                   resType);
                    if (resolved >= 0 && (GLuint)resolved < TEXTURE_UNITS) {
                        program->sampled_texture_unit_mask[resolved >> 5] |=
                            (1u << (resolved & 31u));
                    }
                }
            }
        }
        program->sampled_texture_unit_mask_valid = 1u;
    }

    return (program->sampled_texture_unit_mask[unit >> 5] & (1u << (unit & 31u))) != 0u;
}
