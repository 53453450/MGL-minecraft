/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_pso_build_ops.h - C home of -buildPipelineStateOnCacheMissWithState: and
 * -insertPipelineStateIntoCacheWithWords: (P0-1, log 187).
 */

#ifndef MGL_PSO_BUILD_OPS_H
#define MGL_PSO_BUILD_OPS_H

#include "glm_context.h"

#if defined(__APPLE__)
#include <CoreFoundation/CFBase.h>
#endif
#ifndef CFTimeInterval
typedef double CFTimeInterval;
#endif
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MGLRenderPipelineDescriptorState;

/* -buildPipelineStateOnCacheMissWithState:vertexFunction:fragmentFunction:
 *  cacheKeyWords:pipelineSig:vertexSig:builtColor0Format:builtDepthFormat:
 *  builtStencilFormat:programName:now: */
int mglRenderPassBuildPipelineStateOnCacheMiss(
    void *renderer, const struct MGLRenderPipelineDescriptorState *pipelineState,
    void *vertexFunction, void *fragmentFunction,
    const uint64_t *pipelineCacheKeyWords, uint64_t pipelineSig,
    uint64_t vertexSig, uint32_t builtColor0Format, uint32_t builtDepthFormat,
    uint32_t builtStencilFormat, unsigned int currentProgramName,
    CFTimeInterval now);

/* -insertPipelineStateIntoCacheWithWords:... */
void mglRenderPassInsertPipelineStateIntoCache(
    void *renderer, const uint64_t *pipelineCacheKeyWords, uint64_t pipelineSig,
    uint64_t vertexSig, const struct MGLRenderPipelineDescriptorState *state,
    void *vertexFunction, void *fragmentFunction, int stateFromCache);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PSO_BUILD_OPS_H */
