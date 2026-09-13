/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_binding_state_ops.h — the binding-state forwarders of
 * MGLRenderer+Binding.m as C entry points (P0-1). */

#ifndef MGL_BINDING_STATE_OPS_H
#define MGL_BINDING_STATE_OPS_H

#include "glm_context.h"
#include "mgl_render_values.h"
#include "mgl_render.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mglBindingInvalidateLastBoundState(void *renderer);
void mglBindingRecordLastBoundVertexBuffer(void *renderer, void *buffer,
                                           uint64_t offset, uint64_t index);
void mglBindingRecordLastBoundFragmentBuffer(void *renderer, void *buffer,
                                             uint64_t offset, uint64_t index);
void mglBindingInvalidateLastBoundVertexBufferAtIndex(void *renderer,
                                                      uint64_t index);
void mglBindingInvalidateLastBoundFragmentBufferAtIndex(void *renderer,
                                                        uint64_t index);
void mglBindingSetViewportIfNeeded(void *renderer, double origin_x,
                                   double origin_y, double width,
                                   double height, double znear, double zfar);
void mglBindingSetScissorRectIfNeeded(void *renderer, int64_t x, int64_t y,
                                      uint64_t width, uint64_t height);
void mglBindingSetTriangleFillModeIfNeeded(void *renderer, uint32_t mode);

/* === Resource binding sync ==============================================
 * Work already performed by processDirtyStateDomainsLocked within the same
 * processGLState invocation; the sync entry skips these steps instead of
 * repeating the full rebind (which used to run twice per draw).  The record
 * moved here from MGLRenderer+Draw_Private.h when the method became C. */
typedef struct {
    bool mappedBuffers;
    bool updatedBaseLists;
    bool boundActiveTextures;
} MGLResourceSyncWork;

/* Binding work the render pass still owes after its own state processing
 * (formerly -[MGLRenderer syncResourceBindingsForContext:alreadyDone:]). */
bool mglRendererSyncResourceBindingsForContext(
    void *renderer, GLMContext glm_ctx, const MGLResourceSyncWork *done);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BINDING_STATE_OPS_H */
