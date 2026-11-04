/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Opaque renderer-backend forward declarations (ARCH-027 / Phase 5).
 * Context and FE TUs include this header; call sites that invoke backend
 * entry points include mgl_renderer_backend.h (or the src/ copy) instead.
 */

#ifndef MGL_RENDERER_BACKEND_FWD_H
#define MGL_RENDERER_BACKEND_FWD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLRendererBackendHandle MGLRendererBackendHandle;

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_BACKEND_FWD_H */
