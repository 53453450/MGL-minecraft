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
/* -ensureRasterEncoderForDraw is C now (log 168). */
int mglRenderPassEnsureRasterEncoderForDraw(void *renderer);
/* -mglRenderPassMatchesFramebufferImpl:name: is C now (log 169): the render
 * pass the context currently has must still describe `framebuffer`. */
int mglRenderPassMatchesFramebufferImpl(void *renderer, void *framebuffer,
                                        unsigned int framebuffer_name);
/* -configureUserFBOAttachmentsLocked is C now (log 170): the renderer state
 * areas carry the command state and the context, and every helper it needs
 * is already C. */
bool mglRenderPassConfigureUserFBOAttachments(void *renderer);
/* -finalizeRenderPassDescriptorLocked:traceRenderEncoder: is C now
 * (log 171): it had no self sends at all, only render-pass-state twins. */
/* -generatePipelineDescriptorState:vertexFunction:fragmentFunction: is C
 * now (log 172); the out-parameters are the two Metal functions. */
/* The two Metal functions the descriptor plan resolves.  They are borrowed:
 * the program/cache still owns them, so the Objective-C caller takes them
 * with a __bridge cast instead of ARC writing into an id slot from C. */
typedef struct MGLRenderPassPipelineFunctions_t {
    void *vertex_function;
    void *fragment_function;
} MGLRenderPassPipelineFunctions;

/* -newCommandBufferLocked is C now (log 174); its port wrapper is gone and
 * every caller links straight to this entry. */
int mglRenderPassNewCommandBufferLocked(void *renderer);
/* -createRenderEncoderLocked: is C now (log 175); areas gained the
 * drawable and query-state-owner fields it reads. */
/* -ensureWritableCommandBufferLocked: and -flushCommandBufferLocked: are C
 * now (log 176). */
int mglRenderPassEnsureWritableCommandBufferLocked(void *renderer,
                                                  const char *reason);
void mglRenderPassFlushCommandBufferLocked(void *renderer, int finish);
int mglRenderPassCreateRenderEncoderLocked(void *renderer,
                                           uint64_t renderEncoderCall);
int mglRenderPassGeneratePipelineDescriptorState(
    void *renderer, void *state, MGLRenderPassPipelineFunctions *functions_out);
bool mglRenderPassFinalizeRenderPassDescriptor(void *renderer,
                                               uint64_t renderEncoderCall,
                                               int traceRenderEncoder);
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
