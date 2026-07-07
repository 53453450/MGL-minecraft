/*
 * mgl_buffer_query.h
 * MGL
 *
 * Buffer Query Subsystem.
 *
 * Read-only predicates over Buffer* state: mapped-write detection,
 * drawable-contents check, vertex-stream equivalence.  Used by the per-draw
 * attribute-binding path to decide whether to bind a buffer's CPU or Metal
 * backing and whether to invalidate cached vertex stream state.
 *
 * All functions take Buffer* parameters (no self/ivar/ctx).
 *
 * Dependencies: glm_context.h (Buffer struct, GL_MAP_WRITE_BIT,
 * GL_WRITE_ONLY, GL_READ_WRITE) + glcorearb.h.
 */

#ifndef MGL_BUFFER_QUERY_H
#define MGL_BUFFER_QUERY_H

#include "glcorearb.h"

#include <stdbool.h>
#include <objc/objc.h>   /* BOOL */

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

BOOL mglRendererSameVertexStream(Buffer *lhsBuffer,
                                 GLintptr lhsOffset,
                                 GLuint lhsStride,
                                 GLuint lhsDivisor,
                                 Buffer *rhsBuffer,
                                 GLintptr rhsOffset,
                                 GLuint rhsStride,
                                 GLuint rhsDivisor);
BOOL mglRendererBufferMayHaveMappedWrites(Buffer *buffer);
BOOL mglRendererBufferHasDrawableContents(Buffer *buffer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BUFFER_QUERY_H */
