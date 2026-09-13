/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_dispatch.h — the TCS dispatch entry moved out of
 * MGLRenderer+Tessellation.m (P0-1, log 126).
 *
 * It replaces the Objective-C method
 * -[MGLRenderer(Tessellation) dispatchTessControlShader:program:contract:] and
 * the shell port mglRendererDispatchTessControlShaderPort that forwarded to it
 * (retired in the same cut), so the C draw host calls this function directly.
 */

#ifndef MGL_TESS_DISPATCH_H
#define MGL_TESS_DISPATCH_H

#include <stdbool.h>

#include "glm_context.h"        /* GLMContext */
#include "mgl_air_tess_abi.h"   /* MGLAIRTessDrawContract */
#include "mgl_types_program.h"  /* Program */

#ifdef __cplusplus
extern "C" {
#endif

/* Runs the TCS kernel as a compute dispatch.  Returns true when the dispatch
 * was encoded; every failure path reports on stderr exactly like the method. */
bool mglTessDispatchControlShader(void *renderer, GLMContext glm_ctx,
                                  Program *tcs_program,
                                  const MGLAIRTessDrawContract *contract);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TESS_DISPATCH_H */
