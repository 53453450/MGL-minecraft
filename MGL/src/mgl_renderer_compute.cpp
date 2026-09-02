/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_compute.h"

extern "C" void mglRendererObjCDispatchCompute(
    GLMContext context, unsigned int groups_x, unsigned int groups_y,
    unsigned int groups_z);
extern "C" void mglRendererObjCDispatchComputeIndirect(
    GLMContext context, intptr_t indirect);

extern "C" void mglRenderComputeDispatch(
    GLMContext context, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
    if (!context) {
        return;
    }
    mglRendererObjCDispatchCompute(context, groups_x, groups_y, groups_z);
}

extern "C" void mglRenderComputeDispatchIndirect(
    GLMContext context, intptr_t indirect)
{
    if (!context) {
        return;
    }
    mglRendererObjCDispatchComputeIndirect(context, indirect);
}
