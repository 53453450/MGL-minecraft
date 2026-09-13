/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_compute_dispatch.h — the compute dispatch orchestration, formerly the
 * rest of MGLRenderer+Compute.m (P0-1, log 111).
 *
 * What is left of that file after cuts 109/110: the program/pipeline setup and
 * binding driver, the locked orchestration that runs either the direct encode
 * path or the C++ execution-plan transaction, and the two locked entry points
 * (with their texture/indirect-buffer validation).  The two GL-facing entries
 * mglRendererDispatchCompute{,Indirect} stay in the Objective-C shell because
 * they go through mglRendererForContext and the METAL_LOCK()/UNLOCK() frame.
 */

#ifndef MGL_COMPUTE_DISPATCH_H
#define MGL_COMPUTE_DISPATCH_H

#include "glm_context.h"
#include "mgl_render.h"          /* execution plan, compute plan/result */
#include "mgl_types_buffer.h"    /* MGLStageBindingCopyBackList */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bind the compute program, its pipeline and every buffer/texture binding, for
 * one compute encoder or into an execution plan (exactly one of the two). */
bool mglComputeProcess(void *renderer, void *encoder,
                       MGLStageBindingCopyBackList *copy_backs,
                       MGLRenderComputeExecutionPlan *plan, void *temporaries);

/* End the render encoder, bind the writable image/sampled textures, run the
 * dispatch through the execution plan (or encode it directly), then flush or
 * rotate the command buffer.  False reports the failure to the GL caller. */
bool mglComputeRunDispatchOrchestrationLocked(
    void *renderer, GLMContext glm_ctx, uint32_t dispatch_kind, uint32_t groups_x,
    uint32_t groups_y, uint32_t groups_z, void *indirect_buffer,
    size_t indirect_offset, const char *reason);

/* The locked entry points behind glDispatchCompute{,Indirect}. */
void mglComputeMtlDispatchLocked(void *renderer, GLMContext glm_ctx,
                                 uint32_t groups_x, uint32_t groups_y,
                                 uint32_t groups_z);
void mglComputeMtlDispatchIndirectLocked(void *renderer, GLMContext glm_ctx,
                                         intptr_t indirect);

#ifdef __cplusplus
}
#endif

#endif /* MGL_COMPUTE_DISPATCH_H */
