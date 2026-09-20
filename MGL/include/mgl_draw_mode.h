/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_draw_mode.h
 * MGL
 *
 * GL Draw-Mode Classification Subsystem.
 *
 * Pure inline predicates over GL primitive-mode enums and polygon-mode state.
 * Used by 50+ draw-call sites in MGLRenderer.m to decide whether a draw mode
 * produces polygons (triangles/quads), whether polygon-mode point/line
 * emulation is needed, and whether a primitive has enough vertices to draw.
 *
 * All functions are `static inline` because they're called from per-draw hot
 * paths and the compiler can fold the result into the caller's branch tree.
 *
 * Dependencies: glcorearb.h (GL enums) + glm_context.h (GLMContext).
 * Plain C: the inlines below return bool, not BOOL, so this header is usable
 * from C translation units (it is included by mgl_draw_support.c).
 */

#ifndef MGL_DRAW_MODE_H
#define MGL_DRAW_MODE_H

#include "glcorearb.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decls: the draw-mode classification logic lives in
 * mgl_render.cpp (both gates); the inlines below delegate to it. */
int mglRenderDrawModeProducesPolygons(uint64_t gl_mode);
int mglRenderPrimitiveModeHasDrawableSegment(uint64_t gl_mode,
                                                uint64_t index_count);

/* Returns true if `mode` with `indexCount` vertices produces at least one
 * drawable segment (point/line/triangle/quad).  Used to skip degenerate
 * draws early. */
/* `uint64_t`, not `NSUInteger`: this header declares an extern "C" interface
 * and is included from plain C translation units, where the NS types are
 * unavailable.  The value was already cast to uint64_t in the body. */
static inline bool mglPrimitiveModeHasDrawableSegment(
    GLenum mode, uint64_t indexCount)
{
    return mglRenderPrimitiveModeHasDrawableSegment((uint64_t)mode,
                                                       (uint64_t)indexCount) != 0;
}

/* Returns true if `mode` produces polygonal primitives (triangles/quads)
 * that are subject to glPolygonMode point/line emulation. */
static inline bool mglDrawModeProducesPolygons(GLenum mode)
{
    return mglRenderDrawModeProducesPolygons((uint64_t)mode) != 0;
}

/* Returns true if the context's polygon_mode is GL_POINT and `mode` produces
 * polygons — the draw path must expand the draw into indexed points. */
static inline bool mglPolygonModePointForDrawMode(GLMContext ctx, GLenum mode)
{
    if (!ctx || ctx->active_state->var.polygon_mode != GL_POINT) {
        return false;
    }

    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_QUADS:
            return true;
        default:
            return false;
    }
}

/* Returns true if the context's polygon_mode is GL_LINE and `mode` produces
 * polygons — the draw path must expand the draw into indexed lines. */
static inline bool mglPolygonModeLineForDrawMode(GLMContext ctx, GLenum mode)
{
    return ctx &&
           ctx->active_state->var.polygon_mode == GL_LINE &&
           mglDrawModeProducesPolygons(mode);
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_MODE_H */
