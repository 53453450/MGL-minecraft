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
 * Dependencies: glcorearb.h (GL enums) + glm_context.h (GLMContext) +
 * objc/objc.h (BOOL).
 */

#ifndef MGL_DRAW_MODE_H
#define MGL_DRAW_MODE_H

#include "glcorearb.h"

#include <stdbool.h>
#include <stdint.h>

#include <objc/objc.h>   /* BOOL */

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returns true if `mode` with `indexCount` vertices produces at least one
 * drawable segment (point/line/triangle/quad).  Used to skip degenerate
 * draws early. */
static inline bool mglPrimitiveModeHasDrawableSegment(GLenum mode, NSUInteger indexCount)
{
    switch (mode) {
        case GL_POINTS:
            return indexCount >= 1u;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
            return indexCount >= 2u;
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return indexCount >= 3u;
        case GL_QUADS:
            return indexCount >= 4u;
        default:
            return indexCount > 0u;
    }
}

/* Returns true if `mode` produces polygonal primitives (triangles/quads)
 * that are subject to glPolygonMode point/line emulation. */
static inline BOOL mglDrawModeProducesPolygons(GLenum mode)
{
    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_QUADS:
            return YES;
        default:
            return NO;
    }
}

/* Returns YES if the context's polygon_mode is GL_POINT and `mode` produces
 * polygons — the draw path must expand the draw into indexed points. */
static inline BOOL mglPolygonModePointForDrawMode(GLMContext ctx, GLenum mode)
{
    if (!ctx || ctx->active_state->var.polygon_mode != GL_POINT) {
        return NO;
    }

    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_QUADS:
            return YES;
        default:
            return NO;
    }
}

/* Returns YES if the context's polygon_mode is GL_LINE and `mode` produces
 * polygons — the draw path must expand the draw into indexed lines. */
static inline BOOL mglPolygonModeLineForDrawMode(GLMContext ctx, GLenum mode)
{
    return ctx &&
           ctx->active_state->var.polygon_mode == GL_LINE &&
           mglDrawModeProducesPolygons(mode);
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_MODE_H */
