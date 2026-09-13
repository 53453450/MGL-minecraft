/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_swap_diagnostics.h — swap-time diagnostics, formerly
 * MGLRenderer+SwapDiagnostics.m (P0-1, log 101).
 *
 * Both entry points run at mtlSwapBuffers time:
 *   1. copy the offscreen render-pass color into the drawable when the default
 *      framebuffer's blit path was bypassed;
 *   2. sample the render-pass color source and the drawable target at low
 *      frequency to tell "rendered black" from "copy/present black".
 *
 * The bodies only ever talked to Metal through the mglRender* C facade; what
 * kept them Objective-C was `id`/`NSUInteger`/`NSLog`, two blocks and five ivar
 * reads, and the state areas plus the blit-pipeline C helpers cover all of it.
 * Nothing here allocates or retains on the caller's behalf.
 */

#ifndef MGL_SWAP_DIAGNOSTICS_H
#define MGL_SWAP_DIAGNOSTICS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Copy rp_color0 into drawable_texture when the default framebuffer's render
 * pass still targets an offscreen texture at swap time.  `trace_swap` mirrors
 * the caller's own trace decision for this swap call. */
void mglSwapCopyRenderPassColorToDrawableIfNeeded(void *renderer, void *rp_color0,
                                                  void *drawable_texture,
                                                  uint64_t swap_call,
                                                  bool trace_swap);

/* Low-frequency dual texture sampling for black-screen diagnostics. */
void mglSwapScheduleTextureSampleDiagnostics(void *renderer, void *rp_color0,
                                             void *drawable_texture,
                                             uint64_t swap_call);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SWAP_DIAGNOSTICS_H */
