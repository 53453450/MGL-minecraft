/* SPDX-License-Identifier: LGPL-3.0-only */

#include "mgl_sampler_backend.h"

#include "mgl_backend_objects.h"
#include "mgl_metal_ref.h"

#include <stdlib.h>

void *mglSamplerBackendState(const Sampler *sampler)
{
    return sampler && sampler->backend ? sampler->backend->state : NULL;
}

void mglSamplerBackendResetState(Sampler *sampler)
{
    if (sampler && sampler->backend) {
        mglSafeReleaseMetalObj(&sampler->backend->state);
    }
}

void mglSamplerBackendSetState(Sampler *sampler, void *state)
{
    if (!sampler) {
        return;
    }
    if (!sampler->backend) {
        sampler->backend = (struct MGLSamplerBackend *)calloc(
            1, sizeof(*sampler->backend));
        if (!sampler->backend) {
            mglSafeReleaseMetalObj(&state);
            return;
        }
    }
    mglSafeReleaseMetalObj(&sampler->backend->state);
    sampler->backend->state = state;
}

void mglSamplerBackendRelease(Sampler *sampler)
{
    if (!sampler || !sampler->backend) {
        return;
    }
    mglSafeReleaseMetalObj(&sampler->backend->state);
    free(sampler->backend);
    sampler->backend = NULL;
}
