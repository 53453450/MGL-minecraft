/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_render_pass_manager_ops.h — the render-pass-manager operations that only
 * needed the command state, as C entry points (P0-1). */

#ifndef MGL_RENDER_PASS_MANAGER_OPS_H
#define MGL_RENDER_PASS_MANAGER_OPS_H

#include "glm_context.h"
#include "mgl_binding_state_ops.h" /* MGLResourceSyncWork (log 167) */
#include "mgl_render.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Same bodies as the manager methods of the same name, driven from the
 * renderer's command state (areas.command). */
void mglRenderPassManagerDiscardCurrentCommandBuffer(void *renderer);
void mglRenderPassManagerClearRenderPassIdentity(void *renderer);

/* End the current render encoding: encoder teardown, trace-bindings cleanup and
 * render-pass identity reset, guarded like the Objective-C method was. */
void mglRendererEndRenderEncodingLocked(void *renderer);
/* -processDirtyStateDomainsLocked:work: is C now (log 167). */
bool mglRenderPassProcessDirtyStateDomains(void *renderer, int draw_command,
                                           MGLResourceSyncWork *work);
void mglRenderPassManagerClearCurrentRenderEncoder(void *renderer);
void mglRenderPassManagerEndCurrentRenderEncoder(void *renderer);
int mglRenderPassManagerCommitCommandBufferTransaction(
    void *renderer, void *commandBuffer, void *recoveryOwner,
    int waitForCompletion, MGLRenderCommandBufferTransaction *result);
void mglRenderPassManagerReleaseDetachedCommandBufferIfOwned(
    void *renderer, void *commandBuffer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_PASS_MANAGER_OPS_H */
