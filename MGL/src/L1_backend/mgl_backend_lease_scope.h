/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Scoped acquisition of the renderer backend lease.
 *
 * This lives in its own header because both sides of the backend need the same
 * caller-side contract, while mgl_render_internal.h pulls in a long chain of
 * render-side headers that the backend translation unit does not otherwise
 * need. The context remains opaque; L4 owns attachment and locking.
 *
 * C++ only - RAII scopes cannot be expressed in C.
 */
#ifndef MGL_BACKEND_LEASE_SCOPE_H
#define MGL_BACKEND_LEASE_SCOPE_H

#include "mgl_renderer_backend.h"

/* RAII lease over the renderer backend.  It must express two things:
 *
 *  - holding the lease pins the backend handle, so mglRendererBackendDestroy
 *    cannot delete it while an owner pointer fetched through owner() is in use;
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
        int result = mglRendererBackendBeginContext(context, &lease);
        held = result == 0;
        allowed_without_backend = result == 1;
    }

    bool ok() const { return held || allowed_without_backend; }

    void *owner(MGLRendererBackendOwnerKind kind) const
    {
        return held ? mglRendererBackendLeaseGetOwner(&lease, kind) : nullptr;
    }

    ~MGLBackendLeaseScope()
    {
        if (held) mglRendererBackendEnd(&lease);
    }

    MGLBackendLeaseScope(const MGLBackendLeaseScope &) = delete;
    MGLBackendLeaseScope &operator=(const MGLBackendLeaseScope &) = delete;
};

#endif /* MGL_BACKEND_LEASE_SCOPE_H */
