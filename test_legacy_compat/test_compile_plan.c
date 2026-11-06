/* SPDX-License-Identifier: LGPL-3.0-only */

#include "mgl_compile_plan.h"

#include <stdio.h>

static int failures;

#define CHECK(condition, message)                  \
    do {                                           \
        if (!(condition)) {                        \
            fprintf(stderr, "FAIL: %s\n", message); \
            failures++;                            \
        }                                          \
    } while (0)

int main(void)
{
    const char *attribs[MAX_ATTRIBS] = {0};
    MGLCompilePlan plan;
    mglCompilePlanInit(&plan, "void main() {}", MGL_STAGE_VERTEX, attribs);
    CHECK(plan.source != NULL && plan.stage == MGL_STAGE_VERTEX,
          "plan preserves source and AIR stage");
    CHECK(plan.attrib_names == attribs, "plan borrows attribute snapshot");
    CHECK(plan.air_flags == 0u && plan.iface_peers == NULL,
          "plan starts without variants");

    mglCompilePlanApplyRasterVariant(&plan, _VERTEX_SHADER, true, false,
                                     false, false, true, 0u);
    CHECK(plan.air_flags == (MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER |
                             MGL_AIR_COMPILE_CLIP_DEPTH_ZERO_TO_ONE),
          "VS carries GS and ClipControl variants");

    mglCompilePlanInit(&plan, "fs", MGL_STAGE_FRAGMENT, NULL);
    mglCompilePlanApplyRasterVariant(&plan, _FRAGMENT_SHADER, false, true,
                                     false, false, false, 2u);
    CHECK(plan.air_flags == (MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER |
                             MGL_AIR_COMPILE_FS_CLIP_INTERP),
          "separable FS uses location tags and interpolated clip");

    mglCompilePlanInit(&plan, "fs", MGL_STAGE_FRAGMENT, NULL);
    mglCompilePlanApplyRasterVariant(&plan, _FRAGMENT_SHADER, false, false,
                                     true, false, false, 0u);
    CHECK(plan.air_flags == MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER,
          "TES-compute FS uses location tags");

    mglCompilePlanInit(&plan, "fs", MGL_STAGE_FRAGMENT, NULL);
    mglCompilePlanApplyRasterVariant(&plan, _FRAGMENT_SHADER, false, false,
                                     false, true, false, 0u);
    CHECK(plan.air_flags == MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER,
          "TES-vertex FS uses location tags");

    mglCompilePlanInit(&plan, "gs", MGL_STAGE_GEOMETRY, NULL);
    mglCompilePlanApplyRasterVariant(&plan, _GEOMETRY_SHADER, true, false,
                                     false, false, true, 0u);
    CHECK(plan.air_flags & MGL_AIR_COMPILE_CLIP_DEPTH_ZERO_TO_ONE,
          "GS carries ClipControl variant");

    mglCompilePlanInit(&plan, "cs", MGL_STAGE_COMPUTE, NULL);
    mglCompilePlanApplyRasterVariant(&plan, _COMPUTE_SHADER, false, true,
                                     true, true, true, 4u);
    CHECK(plan.air_flags == 0u, "compute ignores raster-only variants");

    if (failures) {
        return 1;
    }
    printf("test_compile_plan: ok\n");
    return 0;
}
