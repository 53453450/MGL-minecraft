/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_binding_state_ops.c — bodies of the binding-state forwarders that used to
 * sit in MGLRenderer+Binding.m.  Each one is a single mglRenderBinding* call
 * with the binding-state owner from the state areas and, for the owner-aware
 * forms, the current render encoder owner from the command state.
 */

#include "mgl_binding_state_ops.h"
#include "mgl_renderer_ports.h"

void mglBindingInvalidateLastBoundState(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingInvalidate(areas.binding_state_owner
                                   ? *areas.binding_state_owner
                                   : NULL);
}

void mglBindingRecordLastBoundVertexBuffer(void *renderer, void *buffer,
                                           uint64_t offset, uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingRecordVertexBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, buffer,
        offset, (uint32_t)index);
}

void mglBindingRecordLastBoundFragmentBuffer(void *renderer, void *buffer,
                                             uint64_t offset, uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingRecordFragmentBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, buffer,
        offset, (uint32_t)index);
}

void mglBindingInvalidateLastBoundVertexBufferAtIndex(void *renderer,
                                                      uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingInvalidateVertexBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL,
        (uint32_t)index);
}

void mglBindingInvalidateLastBoundFragmentBufferAtIndex(void *renderer,
                                                        uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingInvalidateFragmentBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL,
        (uint32_t)index);
}

void mglBindingSetViewportIfNeeded(void *renderer, double origin_x,
                                   double origin_y, double width,
                                   double height, double znear, double zfar)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    mglRenderBindingSetViewportForOwner(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, owner,
        origin_x, origin_y, width, height, znear, zfar);
}

void mglBindingSetScissorRectIfNeeded(void *renderer, int64_t x, int64_t y,
                                      uint64_t width, uint64_t height)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    mglRenderBindingSetScissorForOwner(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, owner,
        x, y, width, height);
}

void mglBindingSetTriangleFillModeIfNeeded(void *renderer, uint32_t mode)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    mglRenderBindingSetTriangleFillForOwner(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, owner,
        mode);
}
