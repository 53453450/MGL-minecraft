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

bool mglShouldSkipStageBufferResource(Program *program,
                                      int stage,
                                      int resource_type,
                                      const MGLShaderResource *resource)
{
    (void)program;
    (void)stage;
    return resource && resource_type == _UNIFORM_CONSTANT_RES &&
           mglRendererResourceLooksSamplerLike(resource, resource_type);
}

bool mglShouldSkipStageTextureResource(Program *program,
                                       int stage,
                                       int resource_type,
                                       const MGLShaderResource *resource)
{
    (void)program;
    (void)stage;
    (void)resource_type;
    (void)resource;
    return false;
}

bool mglShouldSkipStageSamplerResource(Program *program,
                                       int stage,
                                       int resource_type,
                                       const MGLShaderResource *resource)
{
    (void)program;
    (void)stage;
    (void)resource_type;
    (void)resource;
    return false;
}

uint32_t mglProgramStageBuiltinMask(const Program *program, int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0u;
    }
    return program->air_builtin_mask[stage];
}
