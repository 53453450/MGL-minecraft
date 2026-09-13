/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_compute_bind.h — the compute buffer-binding family, formerly
 * -[MGLRenderer bindBuffersToComputeEncoder:stage:copyBacks:executionPlan:
 *  temporaries:] (P0-1, log 109).
 *
 * The two Objective-C overloads collapse into one C entry: the three-argument
 * form was the same call with a NULL plan and no temporaries.
 *
 * OWNERSHIP: any buffer this creates (an isolated stage buffer, a runtime-array
 * size-constant buffer) is handed to the keep-alive set when the caller builds a
 * plan, and otherwise left to the encoder, which retains what it binds - the
 * reference this function takes is released again before it returns.
 */

#ifndef MGL_COMPUTE_BIND_H
#define MGL_COMPUTE_BIND_H

#include "glm_context.h"
#include "mgl_render.h"          /* MGLRenderComputeExecutionPlan */
#include "mgl_types_buffer.h"    /* BufferMapList, MGLStageBindingCopyBackList */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bind the stage's buffer maps either straight to `encoder` or into `plan`
 * (exactly one of the two is used; `encoder` may be NULL when a plan is given).
 * `temporaries` is the keep-alive set from mglRendererTemporariesCreate. */
bool mglComputeBindBuffersToEncoder(void *renderer, int stage, void *encoder,
                                    MGLStageBindingCopyBackList *copy_backs,
                                    MGLRenderComputeExecutionPlan *plan,
                                    void *temporaries);

/* Bind the stage's sampled/storage textures and their samplers, with the same
 * encoder/plan rule and the same keep-alive set.  This also clears
 * DIRTY_TEX_BINDING | DIRTY_SAMPLER | DIRTY_IMAGE_UNIT_STATE. */
bool mglComputeBindTexturesToEncoder(void *renderer, int stage, void *encoder,
                                     MGLRenderComputeExecutionPlan *plan,
                                     void *temporaries);

#ifdef __cplusplus
}
#endif

#endif /* MGL_COMPUTE_BIND_H */
