/* SPDX-License-Identifier: LGPL-3.0-only */

#ifndef MGL_COMPILE_PLAN_H
#define MGL_COMPILE_PLAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mgl_shader_abi.h"
#include "mgl_types_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLCompilePlan {
    const char *source;                 /* borrowed */
    int stage;                          /* MGL_STAGE_* */
    const char *const *attrib_names;    /* borrowed MAX_ATTRIBS array */
    const void *iface_peers;            /* borrowed MGLShaderResourceList */
    uint32_t air_flags;
    uint32_t tess_patch_vertices;
} MGLCompilePlan;

void mglCompilePlanInit(MGLCompilePlan *plan, const char *source, int stage,
                        const char *const *attrib_names);

/* Apply decisions shared by raster stages.  GS/TES execution-mode selection
 * remains an explicit input; this function only maps it to the AIR ABI flags. */
void mglCompilePlanApplyRasterVariant(
    MGLCompilePlan *plan, int gl_stage, bool has_geometry,
    bool program_separable, bool tess_eval_compute,
    bool tess_eval_render_vertex, bool clip_depth_zero_to_one,
    uint32_t clip_distance_count);

#ifdef __cplusplus
}
#endif

#endif
