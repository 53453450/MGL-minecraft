/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * C pipeline-cache facade — replaces MGLPipelineCache ObjC shell.
 * Binary archive path resolution remains in the .m implementation.
 */

#ifndef MGL_PIPELINE_CACHE_FACADE_H
#define MGL_PIPELINE_CACHE_FACADE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"
#include "mgl_air_loader.h"

typedef struct MGLPipelineCacheState_t {
    void *pipelineState;
    uint64_t pipelineColor0Format;
    uint64_t pipelineDepthFormat;
    uint64_t pipelineStencilFormat;
    GLuint pipelineProgramName;
    void *pipelineVertexFunction;
    void *pipelineFragmentFunction;
    bool dsCacheEnabled;
    bool psoDedupEnabled;
} MGLPipelineCacheState;

#define MGL_PIPELINE_CACHE_KEY_WORDS 7u

#ifdef __cplusplus
extern "C" {
#endif

void mglPipelineCacheInit(MGLPipelineCacheState *state, bool psoDedupEnabled,
                          bool depthStencilCacheEnabled,
                          bool binaryArchiveRequested);
void mglPipelineCacheSetDevice(MGLPipelineCacheState *state, void **owner,
                               void *device, bool binaryArchiveRequested);
bool mglPipelineCacheEnsureOwner(MGLPipelineCacheState *state, void **owner,
                                 void *device, bool binaryArchiveRequested);
bool mglPipelineCacheIsBinaryArchiveEnabled(MGLPipelineCacheState *state,
                                            void *owner,
                                            bool binaryArchiveRequested);

void *mglPipelineCacheDepthStencilStateForValueState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested,
    const MGLRenderDepthStencilDescriptorState *descriptorState);

bool mglPipelineCacheLookupPipeline(MGLPipelineCacheState *state, void **owner,
                                    void *device, bool binaryArchiveRequested,
                                    const uint64_t *words, void **pipelineOut,
                                    void **vertexFunctionOut,
                                    void **fragmentFunctionOut);

uint32_t mglPipelineCacheStorePipeline(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, void *pipeline, void *vertexFunction,
    void *fragmentFunction, const uint64_t *words);

bool mglPipelineCachePipelineDescriptorStateForWords(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, const uint64_t *words,
    MGLRenderPipelineDescriptorState *stateOut);

void mglPipelineCacheStorePipelineDescriptorState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested,
    const MGLRenderPipelineDescriptorState *descriptorState,
    const uint64_t *words);

bool mglPipelineCacheBlendStateForAttachment(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, uint32_t index,
    MGLRenderPipelineBlendState *outState);

void mglPipelineCacheLoadBinaryArchive(MGLPipelineCacheState *state,
                                       void **owner, void *device,
                                       bool *binaryArchiveRequested);
void mglPipelineCacheSaveBinaryArchive(MGLPipelineCacheState *state,
                                       void *owner, void *device);
void mglPipelineCacheDisableBinaryArchive(MGLPipelineCacheState *state,
                                          void **owner, void *device,
                                          bool *binaryArchiveRequested);

int mglPipelineCacheCreateRenderPipelineFromState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested,
    const MGLRenderPipelineDescriptorState *descriptorState,
    void *vertexFunction, void *fragmentFunction, void **pipelineOut,
    char *errorMessage, size_t errorCapacity);

void mglPipelineCacheInvalidatePipelineState(MGLPipelineCacheState *state,
                                             void **owner, void *device,
                                             bool binaryArchiveRequested);
void mglPipelineCacheSetPipelineState(MGLPipelineCacheState *state,
                                      void **owner, void *device,
                                      bool binaryArchiveRequested,
                                      void *pipelineState);
void mglPipelineCacheActivatePipelineState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, void *pipelineState, uint32_t color0Format,
    uint32_t depthFormat, uint32_t stencilFormat, GLuint programName,
    void *vertexFunction, void *fragmentFunction);

void mglPipelineCacheSetBlendFactorsForAttachment(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, uint32_t index, uint32_t srcRgbFactor,
    uint32_t srcAlphaFactor, uint32_t dstRgbFactor, uint32_t dstAlphaFactor,
    uint32_t rgbOperation, uint32_t alphaOperation, uint32_t colorMask);

void mglPipelineCacheResetCaches(MGLPipelineCacheState *state, void **owner);
void mglPipelineCacheShutdown(MGLPipelineCacheState *state, void **owner);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PIPELINE_CACHE_FACADE_H */
