/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_compute_ops.h — the tessellation compute-plan helpers, moved out of
 * MGLRenderer+Tessellation.m (P0-1, log 125).
 *
 * mglTessAppendComputeBytesOp was the last Objective-C piece of the tessellation
 * compute path: it kept the bytes alive with an NSData in the caller's
 * NSMutableArray.  The C version uses CFData, whose +1 from CFDataCreate is what
 * mglRendererTemporariesAdd expects, and the plan keeps the byte pointer.
 */

#ifndef MGL_TESS_COMPUTE_OPS_H
#define MGL_TESS_COMPUTE_OPS_H

#include "mgl_render.h"   /* MGLRenderComputeExecutionPlan */

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Append a bytes binding (kind 1) to the plan, keeping `bytes` alive in
 * `temporaries` (the keep-alive set; required). */
bool mglTessAppendComputeBytesOp(MGLRenderComputeExecutionPlan *plan,
                                 void *temporaries, const void *bytes,
                                 size_t length, size_t index);

/* The point-size uniforms of the tessellation path, appended as a bytes binding
 * when the program uses them (formerly
 * -[MGLRenderer bindPointSizeParamsToComputeEncoder:program:stage:
 *  executionPlan:temporaries:]). */
void mglTessBindPointSizeParamsToComputeEncoder(
    void *renderer, Program *program, int stage,
    MGLRenderComputeExecutionPlan *plan, void *temporaries);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TESS_COMPUTE_OPS_H */
