/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_pipelines.h — blit/clear pipeline, sampler and depth-state cache.
 *
 * Formerly methods of MGLRenderer+Blit.m (-scaledBlitPipelineForPixelFormat:,
 * -scaledBlitSamplerForFilter:, -clearRectDepthState and friends) together with
 * the file-local asset helpers that built them.  The bodies were already plain
 * C; only the pipeline creation went through the metal-cpp facade, so the whole
 * cluster lives in C now and the Objective-C side calls these directly.
 *
 * OWNERSHIP: every function returns a BORROWED reference -- the renderer
 * lifetime caches (the C++ aux pipeline cache and the backend blit cache) own
 * the objects.  NULL means "not available".
 */

#ifndef MGL_BLIT_PIPELINES_H
#define MGL_BLIT_PIPELINES_H

/* glm_context.h must precede the mgl_types_* headers: they expect the GL base
 * types and size_t to be declared already. */
#include "glm_context.h"
#include "mgl_types_texture.h"   /* Texture */

#include <stdint.h>
#include <simd/simd.h>  /* vector_float4 / vector_float3 (scaled-blit params) */

#ifdef __cplusplus
extern "C" {
#endif

/* Scaled-blit uniform block.  Lived in the Objective-C
 * MGLRenderer+Blit_Private.h until the C swap-diagnostics copy path needed the
 * same layout; the Metal shader mirrors it (MGL/aux_shaders/scaled_blit.metal),
 * so keep the field order in sync. */
typedef struct MGLScaledBlitParams_t {
    vector_float4 uvRect; /* xy=min, zw=max in normalized Metal texture coordinates. */
    float forceOpaqueAlpha;
    vector_float3 _padding;
} MGLScaledBlitParams;

/* Diagnostic switch for the swap/present trace-and-copy paths (formerly
 * kMGLSwapPresentDiagnostics in MGLRenderer+Blit_Private.h, which the C swap
 * diagnostics cannot include).  An enum, so includers that never read it do not
 * warn; 0 keeps the diagnostics off, which is how MGL has always been built.
 * The ObjC branches that test it fold away exactly as they did before. */
enum { kMglSwapPresentDiagnostics = 0 };

/* Scaled-blit pipeline for a colour pixel format (BGRA-substituted). */
void *mglBlitScaledPipelineForPixelFormat(void *renderer, uint32_t pixel_format);

/* Scaled depth(-stencil) blit pipeline for a depth pixel format. */
void *mglBlitScaledDepthPipelineForPixelFormat(void *renderer, uint32_t pixel_format);

/* Scaled-blit compute pipeline for a colour pixel format (uint/sint/float
 * variants share the cache with distinct variants). */
void *mglBlitScaledComputePipelineForPixelFormat(void *renderer, uint32_t pixel_format);

/* MSAA integer resolve compute pipeline. */
void *mglBlitMsaaIntegerResolvePipeline(void *renderer, int signed_integer);

/* Scissored clear pipeline for the given colour/depth formats and write masks. */
void *mglBlitClearRectPipeline(void *renderer, uint32_t color_format,
                               uint32_t depth_format, int writes_color,
                               int writes_depth);

/* Nearest/linear scaled-blit sampler (cached in the backend blit cache). */
void *mglBlitScaledSamplerForFilter(void *renderer, uint32_t filter);

/* Depth-stencil state used by the scissored clear (always-compare, write on). */
void *mglBlitClearRectDepthState(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BLIT_PIPELINES_H */
