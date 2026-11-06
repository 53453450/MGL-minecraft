/* SPDX-License-Identifier: LGPL-3.0-only */

#include "glm_context.h"
#include "mgl_renderer_backend.h"

int mglRendererBackendBeginContext(GLMContext context,
                                   MGLRendererBackendLease *lease_out)
{
    if (lease_out) {
        lease_out->backend = NULL;
        lease_out->generation = 0;
    }
    if (!context || !lease_out) return -1;

    /* Hold the attach lock while Begin pins the backend lease. */
    if (context->renderer_backend_lock_initialized &&
        pthread_mutex_lock(&context->renderer_backend_lock) != 0) {
        return -1;
    }
    MGLRendererBackendHandle *backend =
        (MGLRendererBackendHandle *)context->renderer_backend;
    int result = backend ? mglRendererBackendBegin(backend, lease_out) : 1;
    if (context->renderer_backend_lock_initialized) {
        (void)pthread_mutex_unlock(&context->renderer_backend_lock);
    }
    return result;
}

void mglContextDestroyRendererBackend(GLMContext context)
{
    if (!context) return;

    MGLRendererBackendHandle *backend = NULL;
    if (context->renderer_backend_lock_initialized) {
        if (pthread_mutex_lock(&context->renderer_backend_lock) != 0) return;
        backend = (MGLRendererBackendHandle *)context->renderer_backend;
        context->renderer_backend = NULL;
        (void)pthread_mutex_unlock(&context->renderer_backend_lock);
    } else {
        backend = (MGLRendererBackendHandle *)context->renderer_backend;
        context->renderer_backend = NULL;
    }
    mglRendererBackendDestroy(&backend);
}
