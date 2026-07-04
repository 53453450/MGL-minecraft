/*
 * mgl_coordinate.h
 * MGL
 *
 * Coordinate Compatibility Subsystem.
 *
 * Bridges the coordinate-system gap between OpenGL (bottom-left origin,
 * NDC z in [-1,1]) and Metal (top-left origin, NDC z in [0,1]).
 *
 * The Y-Flip Authority model records per render-target whether the RT was
 * written by a program whose vertex shader had Y-flip injection.  Sampling
 * consumers query `mglDecideYFlipForSampledRT` to choose between the original
 * texture and a pre-flipped copy, preventing double-flip when both the
 * rendering program and the sampling program inject Y-flip.
 *
 * This module is pure specification-compliance machinery: every OpenGL
 * program needs framebuffer/texture origin translation when running on
 * Metal, regardless of application.
 */

#ifndef MGL_COORDINATE_H
#define MGL_COORDINATE_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Y-Flip decision returned by `mglDecideYFlipForSampledRT`.
 *
 *   MGL_YFLIP_USE_ORIGINAL              RT already holds GL-origin data
 *                                       (rendered with VS injection); sampler
 *                                       has no injection — use original.
 *
 *   MGL_YFLIP_USE_SAMPLED_COPY          RT holds Metal-top-origin data; sampler
 *                                       has no injection — use the Y-flipped
 *                                       copy maintained by RT Sync.
 *
 *   MGL_YFLIP_USE_ORIGINAL_AND_INJECT   Sampler program has VS injection that
 *                                       will flip on read — use original and
 *                                       let the injection handle the flip.
 *                                       (Also used when both render and sample
 *                                       have injection; injection wins.)
 */
typedef enum {
    MGL_YFLIP_USE_ORIGINAL = 0,
    MGL_YFLIP_USE_SAMPLED_COPY,
    MGL_YFLIP_USE_ORIGINAL_AND_INJECT,
} MGLYFlipDecision;

/* Returns true if `program`'s vertex shader had Y-flip injection applied
 * during MSL post-processing.  The flag is set in program.c when MGL injects
 * the texCoord Y-flip for fullscreen sampled-framebuffer shaders; this avoids
 * false negatives from fragile string matching of MSL source. */
bool mglProgramHasExistingFramebufferSampleYFlip(Program *program);

/* Unified Y-Flip decision for sampling a render-target texture.
 *
 * Authority is stored per-RT in `tex->mtl_render_yflip_authority`, packed as
 * (mtl_render_target_write_version << 1) | render_yflip_injected.  It records
 * whether the RT was written by a program whose VS had Y-flip injection.
 *
 * Decision matrix:
 *   render_yflip | sample_yflip | decision
 *   --------------+--------------+----------------------------------
 *   false         | false        | USE_SAMPLED_COPY  (copy flips once)
 *   false         | true         | USE_ORIGINAL_AND_INJECT
 *   true          | false        | USE_ORIGINAL  (render already flipped)
 *   true          | true         | USE_ORIGINAL_AND_INJECT
 *
 * The key fix: when render_yflip=true and sample_yflip=false (the MC 1.21.8
 * lightmap case — blit_screen.vsh injected, terrain.vsh not), we use the
 * original texture instead of the Y-flipped copy, avoiding double-flip.
 *
 * Defensive downgrade: if the authority version does not match the current
 * `mtl_render_target_write_version`, treat as "not injected" so the safe
 * pre-fix behavior (Y-flipped copy) is used.
 */
MGLYFlipDecision mglDecideYFlipForSampledRT(Texture *tex, Program *samplingProgram);

/* Returns true if the RT write recorded in `tex->mtl_render_yflip_authority`
 * was performed by a program with VS Y-flip injection AND the authority is
 * still current (version matches `mtl_render_target_write_version`).
 *
 * Used by RT Sync to skip generating a Y-flipped copy for injected-rendered
 * RTs — sampling consumers will use the original via the decision above. */
bool mglRTWriteAuthorityIsCurrentAndInjected(Texture *tex);

#ifdef __cplusplus
}
#endif

#endif /* MGL_COORDINATE_H */
