/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Sample transform-feedback stub extension (Phase 6 / G6).
 * Demonstrates hooks-only surface; not linked into libmgl by default.
 */

#ifndef MGL_XFB_STUB_H
#define MGL_XFB_STUB_H

#include "glm_context_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers a no-op GeometryPipelineStage named "xfb_stub" and a segment
 * StateVTable that returns zero dirty masks. Safe to call once per process. */
void mglXfbStubRegister(GLMContext ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_XFB_STUB_H */
