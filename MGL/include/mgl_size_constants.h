/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_size_constants.h — the runtime-array size-constant buffers, formerly
 * -[MGLRenderer bindBufferSizeConstantsForRenderEncoder] (P0-1, log 104).
 *
 * The method's two halves only read C records (the buffer map lists), ask the
 * backend for its cached size-constant buffer and, on a miss, materialize one
 * through the mglRender* facade.  Everything it needed is in the state areas.
 */

#ifndef MGL_SIZE_CONSTANTS_H
#define MGL_SIZE_CONSTANTS_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Push the active vertex/fragment programs' runtime-array size constants to the
 * current render encoder.  False refuses the draw; true also covers "nothing to
 * do" (no encoder, or neither program needs the array-size buffer). */
bool mglRendererBindBufferSizeConstantsForRenderEncoder(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SIZE_CONSTANTS_H */
