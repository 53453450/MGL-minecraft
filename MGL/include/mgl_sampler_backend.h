/* SPDX-License-Identifier: LGPL-3.0-only */

#ifndef MGL_SAMPLER_BACKEND_H
#define MGL_SAMPLER_BACKEND_H

#include <stddef.h>

#include "mgl_types_texture.h"

#ifdef __cplusplus
extern "C" {
#endif

void *mglSamplerBackendState(const Sampler *sampler);
/* SetState takes ownership of `state`; replacing or releasing the backend
 * releases the previously owned Metal object. */
void mglSamplerBackendSetState(Sampler *sampler, void *state);
void mglSamplerBackendResetState(Sampler *sampler);
void mglSamplerBackendRelease(Sampler *sampler);

#ifdef __cplusplus
}
#endif

#endif
