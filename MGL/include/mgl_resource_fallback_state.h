/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_resource_fallback_state.h - the renderer's per-slot trace-binding
 * records, C-visible.
 *
 * Last record to leave the Objective-C MGLRenderer_State.h, for the same reason
 * as the others: the platform shell is C++ now (P0-1, log 206) and mirrors the
 * renderer's ivar layout, which it cannot do through a header that imports
 * Foundation.  MGLRenderer_State.h includes this header, so the Objective-C
 * spelling of the name keeps working.
 */

#ifndef MGL_RESOURCE_FALLBACK_STATE_H
#define MGL_RESOURCE_FALLBACK_STATE_H

#include "glm_context.h"          /* TEXTURE_UNITS */
#include "mgl_trace_strategy.h"   /* MGLFragmentTextureTraceBinding */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLResourceFallbackState_t {
    MGLFragmentTextureTraceBinding fragmentTextureTraceBindings[TEXTURE_UNITS];
} MGLResourceFallbackState;

#ifdef __cplusplus
}
#endif

#endif /* MGL_RESOURCE_FALLBACK_STATE_H */
