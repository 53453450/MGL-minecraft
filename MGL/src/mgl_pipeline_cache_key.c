/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_pipeline_cache_key.h"

#include <string.h>

uint32_t mglPipelineCachePackTessFlags(bool native_tes_active,
                                       bool tess_vertex_capture_active,
                                       bool geometry_expansion_active,
                                       bool cull_distance_capture_active,
                                       bool tess_compute_active)
{
    uint32_t flags = 0u;
    if (native_tes_active) {
        flags |= MGL_PIPELINE_TESS_FLAG_NATIVE_TES;
    }
    if (tess_vertex_capture_active) {
        flags |= MGL_PIPELINE_TESS_FLAG_VERTEX_CAPTURE;
    }
    if (geometry_expansion_active) {
        flags |= MGL_PIPELINE_TESS_FLAG_GEOMETRY_EXPANSION;
    }
    if (cull_distance_capture_active) {
        flags |= MGL_PIPELINE_TESS_FLAG_CULL_DISTANCE;
    }
    if (tess_compute_active) {
        flags |= MGL_PIPELINE_TESS_FLAG_TESS_COMPUTE;
    }
    return flags;
}

uint64_t mglPipelineCacheBuildPrimaryKey(
    const MGLPipelineCacheKeyInputs *inputs)
{
    if (!inputs) {
        return 0ull;
    }

    uint64_t primary_key = ((uint64_t)inputs->program_name << 32);
    primary_key |= ((uint64_t)(inputs->clip_origin & 0xFu) << 28);
    primary_key |= ((uint64_t)(inputs->clip_depth_mode & 0xFu) << 24);
    if (inputs->tess_flags & MGL_PIPELINE_TESS_FLAG_NATIVE_TES) {
        primary_key |= (1ull << 23);
    }
    if (inputs->tess_flags & MGL_PIPELINE_TESS_FLAG_VERTEX_CAPTURE) {
        primary_key |= (1ull << 22);
    }
    if (inputs->tess_flags & MGL_PIPELINE_TESS_FLAG_GEOMETRY_EXPANSION) {
        primary_key |= (1ull << 21);
    }
    if (inputs->tess_flags & MGL_PIPELINE_TESS_FLAG_CULL_DISTANCE) {
        primary_key |= (1ull << 20);
    }
    if (inputs->tess_flags & MGL_PIPELINE_TESS_FLAG_TESS_COMPUTE) {
        primary_key |= (1ull << 19);
    }
    return primary_key;
}

void mglPipelineCacheBuildLookupKeyWords(
    const MGLPipelineCacheKeyInputs *inputs,
    uint64_t words[MGL_PIPELINE_CACHE_KEY_WORDS])
{
    if (!words) {
        return;
    }
    memset(words, 0, sizeof(uint64_t) * MGL_PIPELINE_CACHE_KEY_WORDS);
    if (!inputs) {
        return;
    }

    words[0] = mglPipelineCacheBuildPrimaryKey(inputs);
    words[1] = inputs->vertex_instance_id;
    words[2] = inputs->vertex_generation;
    words[3] = inputs->fragment_instance_id;
    words[4] = inputs->fragment_generation;
    words[5] = inputs->pipeline_sig;
    words[6] = inputs->vertex_sig;
}
