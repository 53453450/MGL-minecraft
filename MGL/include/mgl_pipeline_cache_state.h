/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_pipeline_cache_state.h — the pipeline cache's state record, C-visible.
 *
 * Moved out of the Objective-C MGLPipelineCache.h for the same reason as the
 * batching and command states: C callers used to need one shim port per field
 * they read (the active pipeline handle and the pipeline's program name).  The
 * cache keeps the record as its ivar and one port hands out the address.
 */

#ifndef MGL_PIPELINE_CACHE_STATE_H
#define MGL_PIPELINE_CACHE_STATE_H

#include "glcorearb.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLPipelineCacheState_t {
    /* Metal objects are owned by the C++ cache owner.  These borrowed opaque
     * identities are retained only so the GL-semantic ObjC layer can test and
     * pass the active handles without importing Metal object types. */
    void * _Nullable pipelineState;
    /* Keep the native NSUInteger-sized value width while remaining a plain
     * C value; Metal enum names stay out of this interface. */
    uint64_t pipelineColor0Format;
    uint64_t pipelineDepthFormat;
    uint64_t pipelineStencilFormat;
    GLuint pipelineProgramName;
    void * _Nullable pipelineVertexFunction;
    void * _Nullable pipelineFragmentFunction;
    uint8_t dsCacheEnabled;
    uint8_t psoDedupEnabled;
} MGLPipelineCacheState;

#ifdef __cplusplus
}
#endif

#endif /* MGL_PIPELINE_CACHE_STATE_H */
