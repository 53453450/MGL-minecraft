/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Pure-C dirty sync domain classification (gtest-friendly, no Metal).
 */

#include "mgl_renderer_sync.h"
#include "mgl_types_state.h"

uint32_t mglRenderClassifyDirtySyncDomains(uint32_t dirty_bits)
{
    uint32_t domains = 0;
    if (dirty_bits & DIRTY_FBO) {
        domains |= MGL_SYNC_DOMAIN_FBO;
    }
    if (dirty_bits & DIRTY_STATE) {
        domains |= MGL_SYNC_DOMAIN_STATE;
    }
    if (dirty_bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_BUFFER_BASE_STATE)) {
        domains |= MGL_SYNC_DOMAIN_PROGRAM_VAO;
    }
    if (dirty_bits &
        (DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER)) {
        domains |= MGL_SYNC_DOMAIN_TEX;
    }
    if (dirty_bits & DIRTY_VAO) {
        domains |= MGL_SYNC_DOMAIN_VAO;
    }
    if (dirty_bits & DIRTY_BUFFER) {
        domains |= MGL_SYNC_DOMAIN_BUFFER;
    }
    if (dirty_bits & DIRTY_RENDER_STATE) {
        domains |= MGL_SYNC_DOMAIN_RENDER_STATE;
    }
    if (dirty_bits &
        (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_ALPHA_STATE |
         DIRTY_RENDER_STATE)) {
        domains |= MGL_SYNC_DOMAIN_PIPELINE;
    }
    return domains;
}
