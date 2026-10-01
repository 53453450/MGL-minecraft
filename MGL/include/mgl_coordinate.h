/*
 * mgl_coordinate.h
 * MGL
 *
 * Coordinate Compatibility Subsystem.
 *
 * Bridges the coordinate-system gap between OpenGL (bottom-left origin,
 * NDC z in [-1,1]) and Metal (top-left origin, NDC z in [0,1]).
 *
 * Rendered render-target storage is Y-flipped relative to GL; sampling
 * consumers query `mglDecideYFlipForSampledRT` to choose between the original
 * texture and the pre-flipped copy maintained by RT Sync.
 */

#ifndef MGL_COORDINATE_H
#define MGL_COORDINATE_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Y-Flip decision returned by `mglDecideYFlipForSampledRT`.
 *
 *   MGL_YFLIP_USE_ORIGINAL       Storage holds GL-origin data — use it as is.
 *
 *   MGL_YFLIP_USE_SAMPLED_COPY   Storage holds Metal-top-origin data — use the
 *                                Y-flipped copy maintained by RT Sync.
 */
typedef enum {
    MGL_YFLIP_USE_ORIGINAL = 0,
    MGL_YFLIP_USE_SAMPLED_COPY,
} MGLYFlipDecision;

MGLYFlipDecision mglDecideYFlipForSampledRT(Texture *tex);

#ifdef __cplusplus
}
#endif

#endif /* MGL_COORDINATE_H */
