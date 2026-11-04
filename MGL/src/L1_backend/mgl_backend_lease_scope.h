/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Scoped acquisition of the renderer backend lease.
 *
 * This lives in its own header because there is exactly one caller-side
 * contract that both sides of the backend need, and neither obvious home
 * works: mgl_renderer_backend.h only forward-declares GLMContextRec_t (the
 * ok() probe reads two of its fields), and mgl_render_internal.h pulls in a
 * long chain of render-side headers that the backend translation unit does
 * not otherwise need.  glm_context.h has the full record, so this header is
 * self-contained: include it and nothing else.
 *
 * C++ only - RAII scopes cannot be expressed in C.
 */
#ifndef MGL_BACKEND_LEASE_SCOPE_H
#define MGL_BACKEND_LEASE_SCOPE_H

#include <pthread.h>

#include "glm_context.h"
#include "mgl_renderer_backend.h"

/* RAII lease over the renderer backend.  It must express two things:
 *
 *  - holding the lease pins the backend handle, so mglRendererBackendDestroy
 *    cannot delete it while an owner pointer fetched through rendererOwner()
 *    is in use;
 *  - having no backend at all is not an error for the callers here - they
 *    early-out on a null owner - so ok() reports that case as allowed rather
 *    than failed.
 *
 * Callers must test ok() before touching any backend state. */
struct MGLBackendLeaseScope {
    MGLRendererBackendLease lease{};
    bool held = false;
    bool allowed_without_backend = false;

    explicit MGLBackendLeaseScope(GLMContext context)
    {
        if (!context) {
            allowed_without_backend = true;
            return;
        }
        held = mglRendererBackendBeginContext(context, &lease) == 0;
        if (held) return;
        /* BeginContext only reports failure.  Re-read under the attach lock
         * to tell "there is no backend" (allowed) from "could not take the
         * lease" (not allowed). */
        if (context->renderer_backend_lock_initialized &&
            pthread_mutex_lock(&context->renderer_backend_lock) == 0) {
            allowed_without_backend = context->renderer_backend == nullptr;
            (void)pthread_mutex_unlock(&context->renderer_backend_lock);
        } else {
            allowed_without_backend = context->renderer_backend == nullptr;
        }
    }

    bool ok() const { return held || allowed_without_backend; }

    ~MGLBackendLeaseScope()
    {
        if (held) mglRendererBackendEnd(&lease);
    }

    MGLBackendLeaseScope(const MGLBackendLeaseScope &) = delete;
    MGLBackendLeaseScope &operator=(const MGLBackendLeaseScope &) = delete;
};

#endif /* MGL_BACKEND_LEASE_SCOPE_H */
