/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Pipeline cache lookup key construction (pure C, gtest-friendly).
 */

#ifndef MGL_PIPELINE_CACHE_KEY_H
#define MGL_PIPELINE_CACHE_KEY_H

#include <stdint.h>

#define MGL_PIPELINE_CACHE_KEY_WORDS 7u

#define MGL_PIPELINE_TESS_FLAG_NATIVE_TES (1u << 0)
#define MGL_PIPELINE_TESS_FLAG_VERTEX_CAPTURE (1u << 1)
#define MGL_PIPELINE_TESS_FLAG_GEOMETRY_EXPANSION (1u << 2)
#define MGL_PIPELINE_TESS_FLAG_CULL_DISTANCE (1u << 3)
#define MGL_PIPELINE_TESS_FLAG_TESS_COMPUTE (1u << 4)

typedef struct MGLPipelineCacheKeyInputs_t {
    uint32_t program_name;
    uint32_t clip_origin;
    uint32_t clip_depth_mode;
    uint32_t tess_flags;
    uint64_t vertex_instance_id;
    uint64_t vertex_generation;
    uint64_t fragment_instance_id;
    uint64_t fragment_generation;
    uint64_t pipeline_sig;
    uint64_t vertex_sig;
} MGLPipelineCacheKeyInputs;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t mglPipelineCachePackTessFlags(bool native_tes_active,
                                       bool tess_vertex_capture_active,
                                       bool geometry_expansion_active,
                                       bool cull_distance_capture_active,
                                       bool tess_compute_active);

uint64_t mglPipelineCacheBuildPrimaryKey(
    const MGLPipelineCacheKeyInputs *inputs);

void mglPipelineCacheBuildLookupKeyWords(
    const MGLPipelineCacheKeyInputs *inputs,
    uint64_t words[MGL_PIPELINE_CACHE_KEY_WORDS]);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PIPELINE_CACHE_KEY_H */
