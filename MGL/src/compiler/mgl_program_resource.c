/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#include "mgl_program_resource.h"

#include "mgl_sampler_compat.h"


const char *mglShaderStageName(int stage)
{
    switch (stage) {
        case _VERTEX_SHADER: return "vertex";
        case _TESS_CONTROL_SHADER: return "tess_control";
        case _TESS_EVALUATION_SHADER: return "tess_eval";
        case _GEOMETRY_SHADER: return "geometry";
        case _FRAGMENT_SHADER: return "fragment";
        case _COMPUTE_SHADER: return "compute";
        default: return "unknown";
    }
}

/* The three SPIRV-era "should this stage resource be skipped" predicates are
 * gone.  They existed because the old reflection could not always classify a
 * resource (a sampler parked in the plain-uniform list, image dims guessed from
 * names), so callers had to be told to ignore the odd entry.  On the IR chain
 * every entry is classified exactly, and the last remaining rule
 * (mglShouldSkipStageBufferResource) never fired: probe, 2026-09-12, zero hits
 * across the local suite and the 1328-case GL46 hotspot list. */
/* mglShouldSkipStageTextureResource / mglShouldSkipStageSamplerResource were
 * constant-false stubs (their SPIRV-era skip heuristics are gone); callers no
 * longer consult them. */

uint32_t mglProgramStageBuiltinMask(const Program *program, int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0u;
    }
    return program->air_builtin_mask[stage];
}
