/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_renderer_binding_ops.h — renderer-level forwarders to the
 * mglRenderBinding* last-bound state of the current renderer. */

#ifndef MGL_RENDERER_BINDING_OPS_H
#define MGL_RENDERER_BINDING_OPS_H

#include "glm_context.h"
#include "mgl_render_values.h"
#include "mgl_render.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mglRendererBindingInvalidateLastBoundState(void *renderer);
void mglRendererBindingRecordLastBoundVertexBuffer(void *renderer, void *buffer,
                                                   uint64_t offset,
                                                   uint64_t index);
void mglRendererBindingRecordLastBoundFragmentBuffer(void *renderer,
                                                     void *buffer,
                                                     uint64_t offset,
                                                     uint64_t index);
void mglRendererBindingInvalidateLastBoundVertexBufferAtIndex(void *renderer,
                                                              uint64_t index);
void mglRendererBindingInvalidateLastBoundFragmentBufferAtIndex(
    void *renderer, uint64_t index);

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

#endif /* MGL_RENDERER_BINDING_OPS_H */
