/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Compile / frontend / layout artifact contracts (ARCHITECTURE_AUDIT R2).
 * Temporary owners hold code + reflection + layout; publish only on full
 * success so callers never bind a half-built program executable.
 */

#ifndef MGL_COMPILE_ARTIFACT_H
#define MGL_COMPILE_ARTIFACT_H

#include <stddef.h>
#include <stdint.h>

#include "mgl_shader_abi.h"
#include "mgl_types_program.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Immutable frontend snapshot for one shader source + stage. */
typedef struct MGLFrontendArtifact {
    const char *source;          /* borrowed; not owned */
    int stage;                   /* MGL_STAGE_* */
    uint32_t layout_version;     /* MGL_AIR_PER_VERTEX_LAYOUT_VERSION */
    char *diagnostics;           /* owned NUL-terminated message or NULL */
    uint32_t builtin_use_mask;   /* reserved for counted frontend reuse */
} MGLFrontendArtifact;

/* Shared resource / stage-record layout plan consumed by reflection,
 * AIR metadata, and buffer binding plans. */
typedef struct MGLResourceLayoutPlan {
    uint32_t layout_version;
    uint32_t per_vertex_stride;
    uint32_t layer_offset;
    uint32_t viewport_index_offset;
    uint32_t stream_offset;
    uint32_t primitive_id_offset;
    uint32_t clip_distance_offset;
    uint32_t cull_distance_offset;
    uint32_t cull_distance_count;
} MGLResourceLayoutPlan;

/* Single-stage compile product: metallib bytes + reflection lists.
 * Owned until mglCompileArtifactDestroy or successful publish into Program. */
typedef struct MGLCompileArtifact {
    MGLFrontendArtifact frontend;
    MGLResourceLayoutPlan layout;
    unsigned char *metallib_bytes; /* owned */
    size_t metallib_size;
    MGLShaderResourceList resources[MGL_MAX_SHADER_RESOURCES]; /* owned lists */
    MGLAIRStageInfo stage_info;
    int complete; /* non-zero only when code and reflection both succeeded */
} MGLCompileArtifact;

static inline void mglResourceLayoutPlanInitDefault(MGLResourceLayoutPlan *plan)
{
    if (!plan) {
        return;
    }
    plan->layout_version = (uint32_t)MGL_AIR_PER_VERTEX_LAYOUT_VERSION;
    plan->per_vertex_stride = (uint32_t)MGL_AIR_PER_VERTEX_STRIDE;
    plan->layer_offset = (uint32_t)MGL_AIR_PER_VERTEX_LAYER_OFFSET;
    plan->viewport_index_offset =
        (uint32_t)MGL_AIR_PER_VERTEX_VIEWPORT_INDEX_OFFSET;
    plan->stream_offset = (uint32_t)MGL_AIR_PER_VERTEX_STREAM_OFFSET;
    plan->primitive_id_offset =
        (uint32_t)MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET;
    plan->clip_distance_offset =
        (uint32_t)MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET;
    plan->cull_distance_offset =
        (uint32_t)MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET;
    plan->cull_distance_count =
        (uint32_t)MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT;
}

void mglCompileArtifactInit(MGLCompileArtifact *art);
void mglCompileArtifactDestroy(MGLCompileArtifact *art);
MGLCompileArtifact *mglCompileArtifactCreate(void);
void mglCompileArtifactFree(MGLCompileArtifact *art);

/* Compile one GLSL stage into a temporary artifact. Returns 0 on full
 * success (complete!=0); on failure *art is cleaned and complete==0.
 * Variant compiles that need air_flags / iface_peers stay on the direct
 * mglAirCompileGLSLWithReflectInfoEx path in program.c. */
int mglCompileArtifactFromGLSL(const char *src, int stage,
                               const char *const *attrib_names,
                               MGLCompileArtifact *art_out,
                               char *err_buf, size_t err_cap);

/* True when a CompileShader-cached artifact can be published at link
 * without recompiling (no variant flags, peers, or VS attrib remaps). */
int mglCompileArtifactCanReuseAtLink(const MGLCompileArtifact *art,
                                     int air_stage,
                                     uint32_t air_flags,
                                     const void *iface_peers,
                                     const char *const *attrib_names);

#ifdef __cplusplus
}
#endif

#endif /* MGL_COMPILE_ARTIFACT_H */
