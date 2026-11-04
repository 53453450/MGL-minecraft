/* SPDX-License-Identifier: LGPL-3.0-only */
/*
 * mgl_draw_mode.c
 * MGL
 *
 * GL draw-mode classification: which modes produce polygons, whether a given
 * index count yields a drawable segment, which modes the backend must emulate,
 * and how many vertices one primitive of a mode consumes.  Pure GL semantics;
 * the declarations live in mgl_draw_mode.h / mgl_render_api_draw.h.
 */

#include "mgl_draw_mode.h"

#include <stdint.h>

int mglRenderDrawModeProducesPolygons(uint64_t gl_mode) {
    switch (gl_mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_QUADS:
            return 1;
        default:
            return 0;
    }
}

int mglRenderPrimitiveModeHasDrawableSegment(uint64_t gl_mode,
                                                uint64_t index_count) {
    switch (gl_mode) {
        case GL_POINTS:
            return index_count >= 1u ? 1 : 0;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
            return index_count >= 2u ? 1 : 0;
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return index_count >= 3u ? 1 : 0;
        case GL_QUADS:
            return index_count >= 4u ? 1 : 0;
        case GL_LINES_ADJACENCY:
        case GL_LINE_STRIP_ADJACENCY:
            return index_count >= 4u ? 1 : 0;
        case GL_TRIANGLES_ADJACENCY:
        case GL_TRIANGLE_STRIP_ADJACENCY:
            return index_count >= 6u ? 1 : 0;
        default:
            return index_count > 0u ? 1 : 0;
    }
}

int mglRenderDrawModeNeedsEmulate(uint32_t mode) {
    return mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP || mode == GL_QUADS ||
                   mode == GL_LINES_ADJACENCY ||
                   mode == GL_LINE_STRIP_ADJACENCY ||
                   mode == GL_TRIANGLES_ADJACENCY ||
                   mode == GL_TRIANGLE_STRIP_ADJACENCY
               ? 1
               : 0;
}

uint32_t mglRenderPrimitiveVertexCountForMode(uint32_t mode) {
    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        /* AUDIT-034: adjacency assembly drops adjacent verts; the rasterized
         * primitive still has 3 (tri) / 2 (line) vertices for cull emu. */
        case GL_TRIANGLES_ADJACENCY:
        case GL_TRIANGLE_STRIP_ADJACENCY:
            return 3u;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
        case GL_LINES_ADJACENCY:
        case GL_LINE_STRIP_ADJACENCY:
            return 2u;
        case GL_QUADS:
            return 4u;
        case GL_POINTS:
        default:
            return 1u;
    }
}
