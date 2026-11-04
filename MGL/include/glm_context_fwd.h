/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * glm_context_fwd.h — opaque GLMContext for cross-layer headers (ARCH-001 / G5).
 *
 * Prefer this over glm_context.h in plan / public / extension headers.
 * Full Rec stays in glm_context.h for L4 / selected L2 TUs.
 */

#ifndef GLM_CONTEXT_FWD_H
#define GLM_CONTEXT_FWD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GLMContextRec_t GLMContextRec;
typedef GLMContextRec *GLMContext;

#ifdef __cplusplus
}
#endif

#endif /* GLM_CONTEXT_FWD_H */
