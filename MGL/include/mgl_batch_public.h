/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_batch_public.h — the batch cluster's entry points, and nothing else.
 *
 * WHY THIS HEADER EXISTS
 *
 * The deferred-draw batch cluster (draw_command.c + mgl_batch_*.{c,cpp}) is
 * ~12.7k lines and carries the correctness of everything else: batch merging,
 * the per-batch state snapshot, dirty-domain planning and replay.  Its headers
 * declared 157 functions, and until this header existed there was no statement
 * anywhere of which of those the rest of the tree is allowed to call - the
 * internal machinery and the public entry points were one flat namespace, so a
 * new violation was invisible.
 *
 * This header is that statement.  It declares exactly the symbols that
 * translation units OUTSIDE the cluster actually use, derived by census rather
 * than by intent.  scripts/state_machine_invariants.py check I8 fails when a
 * cross-boundary reference appears that is not listed here.  The point is not
 * to shrink the current number - every entry below is a real call site - but to
 * make the NEXT one a decision instead of an accident.
 *
 * THE CLUSTER BOUNDARY
 *
 *   inside  (may include the mgl_batch_*.h headers freely):
 *     draw_command.c, mgl_batch_*.{c,cpp}, mgl_renderer_core_state.c
 *
 *   outside (includes THIS header, not the mgl_batch_*.h ones):
 *     everything else - in particular the render-pass, platform-shell and draw
 *     submission paths that drive a flush.
 *
 * If you are outside and need something that is not declared here, widen this
 * header deliberately and say who needs it and why - rather than including
 * mgl_batch_issue.h and taking whichever internal symbol looks close enough.
 */

#ifndef MGL_BATCH_PUBLIC_H
#define MGL_BATCH_PUBLIC_H

#include "glm_context.h"

/* MGLSamplerSnapshotKey (draw_command.h) and TextureParameter
 * (mgl_types_texture.h) are the argument/return types of the sampler entry
 * below, so the public surface depends on them.  Both are self-contained C
 * headers with no batch internals in them. */
#include "draw_command.h"
#include "mgl_types_texture.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * 1. Flush lifecycle
 *
 * The platform shell drives a deferred flush in three steps because it owns the
 * exception frame: Begin switches to the replay workspace, RunBatches is the
 * body, TeardownReplay must run even when a draw raises.  The pass record is
 * therefore part of the public contract, not a cluster internal - the caller
 * stack-allocates it (mgl_platform_shell.cpp).
 * ------------------------------------------------------------------------ */
typedef struct MGLBatchFlushPass {
    uint64_t hit;
    uint32_t skipped;
    GLMState saved;
    GLenum saved_error;
    GLenum replay_error;
} MGLBatchFlushPass;

/* Bind the context, snapshot the live state, switch to the replay workspace.
 * Returns 0 when the context has no batches to replay. */
int mglBatchFlushBegin(void *renderer, GLMContext glm_ctx,
                       MGLBatchFlushPass *pass);

/* The body: run the flush loop and log the per-flush summary. */
void mglBatchFlushRunBatches(void *renderer, GLMContext glm_ctx,
                             MGLBatchFlushPass *pass);

/* Replay-workspace teardown, driven by the pass state. */
void mglBatchTeardownReplay(void *renderer, GLMContext glm_ctx,
                            MGLBatchFlushPass *pass);

/* ---------------------------------------------------------------------------
 * 2. Draw submission
 *
 * The draw entry path reports a submitted draw so the batch bookkeeping
 * (submitted-draw counters, the recorded state the next batch starts from)
 * stays accurate.
 * ------------------------------------------------------------------------ */
void mglBatchRecordArrayDrawSubmitted(void *renderer, GLMContext glm_ctx,
                                      GLenum mode, uint64_t vertex_count);
void mglBatchRecordElementDrawSubmitted(void *renderer, GLMContext glm_ctx,
                                        GLenum mode, uint64_t index_count);

/* ---------------------------------------------------------------------------
 * 3. Binding / sampler helpers needed by the flush path from outside
 * ------------------------------------------------------------------------ */

/* Re-materialise the sampled textures a pending batch reads, after the render
 * pass changed under it.  Returns 0 when the binding state is not valid.
 *
 * NOTE: mgl_batch_issue.h used to declare this as `bool` while the definition
 * (mgl_batch_replay.cpp) returns `int` - a genuine mismatch that nothing
 * caught, because every caller wrote it in a boolean context.  This header
 * carries the definition's type. */
int mglBatchBindActiveTexturesToMTL(void *renderer, GLMContext glm_ctx);

/* Expand a sampler-snapshot key back into TextureParameter values. */
void mgl_batch_replay_fill_sampler_params(const MGLSamplerSnapshotKey *key,
                                          TextureParameter *params_out);

/* ---------------------------------------------------------------------------
 * 4. Indirect command buffers
 * ------------------------------------------------------------------------ */

/* Capability probe (OS / compile time), not batch state. */
int mgl_batch_icb_support_indirect_command_buffers(void);

/* Build the encoder-side indirect command buffer. */
void *mgl_batch_mtl_create_icb(int indexed, uint64_t max_command_count);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_PUBLIC_H */
