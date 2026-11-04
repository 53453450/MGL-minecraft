/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_state_apply.h — shared dirty-domain apply (ARCH-016 / Phase 1).
 *
 * Batch and Pass both depend downward on this apply surface; they must not
 * call each other for texture/buffer dirty work.
 */

#ifndef MGL_STATE_APPLY_H
#define MGL_STATE_APPLY_H

#include "mgl_renderer_binding_ops.h" /* MGLResourceSyncWork */

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Apply dirty GL domains to Metal (formerly processDirtyStateDomainsLocked). */
bool mglStateApplyDirtyDomains(void *renderer, int draw_command,
                               MGLResourceSyncWork *work);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STATE_APPLY_H */
