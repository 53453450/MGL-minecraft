/* SPDX-License-Identifier: LGPL-3.0-only */

#include "mgl_compile_plan.h"

#include <string.h>

void mglCompilePlanInit(MGLCompilePlan *plan, const char *source, int stage,
                        const char *const *attrib_names)
{
    if (!plan) {
        return;
    }
    memset(plan, 0, sizeof(*plan));
    plan->source = source;
    plan->stage = stage;
    plan->attrib_names = attrib_names;
}

void mglCompilePlanApplyRasterVariant(
    MGLCompilePlan *plan, int gl_stage, bool has_geometry,
    bool program_separable, bool tess_eval_compute,
    bool tess_eval_render_vertex, bool clip_depth_zero_to_one,
    uint32_t clip_distance_count)
{
    if (!plan) {
        return;
    }
    if (has_geometry) {
        plan->air_flags |= MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER;
    }
    if (clip_depth_zero_to_one &&
        (gl_stage == _VERTEX_SHADER || gl_stage == _TESS_EVALUATION_SHADER ||
         gl_stage == _GEOMETRY_SHADER)) {
        plan->air_flags |= MGL_AIR_COMPILE_CLIP_DEPTH_ZERO_TO_ONE;
    }
    if (gl_stage == _FRAGMENT_SHADER && !has_geometry &&
        (tess_eval_compute || tess_eval_render_vertex || program_separable)) {
        plan->air_flags |= MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER;
    }
    if (gl_stage == _VERTEX_SHADER && program_separable && !has_geometry) {
        plan->air_flags |= MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER;
    }
    if (gl_stage == _FRAGMENT_SHADER && clip_distance_count > 0u) {
        plan->air_flags |= MGL_AIR_COMPILE_FS_CLIP_INTERP;
    }
}
