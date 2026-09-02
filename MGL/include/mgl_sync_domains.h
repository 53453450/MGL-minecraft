/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Pure-C dirty sync domain classification (gtest-friendly, no Metal/mach).
 */

#ifndef MGL_SYNC_DOMAINS_H
#define MGL_SYNC_DOMAINS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MGL_SYNC_DOMAIN_FBO          (1u << 0)
#define MGL_SYNC_DOMAIN_STATE        (1u << 1)
#define MGL_SYNC_DOMAIN_PROGRAM_VAO  (1u << 2)
#define MGL_SYNC_DOMAIN_TEX          (1u << 3)
#define MGL_SYNC_DOMAIN_VAO          (1u << 4)
#define MGL_SYNC_DOMAIN_BUFFER       (1u << 5)
#define MGL_SYNC_DOMAIN_RENDER_STATE (1u << 6)
#define MGL_SYNC_DOMAIN_PIPELINE     (1u << 7)
#define MGL_SYNC_DOMAIN_ALL          0xFFFFFFFFu

uint32_t mglRenderClassifyDirtySyncDomains(uint32_t dirty_bits);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SYNC_DOMAINS_H */
