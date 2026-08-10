/*
 * mgl_vertex_attrib_query.h
 * MGL
 *
 * Vertex Attrib Query Subsystem.
 *
 * Read-only queries over a Program's SPIR-V stage-input resources and
 * VertexArray attribute state.  Used by the per-draw attribute-binding path
 * to decide which vertex attributes need binding, which are color inputs,
 * and which use current-value fallbacks.
 *
 * All functions are pure (take Program* and VertexArray* params, no self/ivar).
 *
 * Dependencies: glm_context.h (Program, SpirvResource, SpirvResourceList,
 * VertexArray, MAX_ATTRIBS, _VERTEX_SHADER) + spirv_cross_c.h
 * (_STAGE_INPUT_RES) + <strings.h> (strcasecmp).
 */

#ifndef MGL_VERTEX_ATTRIB_QUERY_H
#define MGL_VERTEX_ATTRIB_QUERY_H

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

/* Returns true if `program`'s vertex stage has a stage-input resource
 * matching `attribIndex`. */
BOOL mglRendererProgramUsesVertexAttrib(Program *program, GLuint attribute);

/* Returns the SpirvResource for `attribIndex` in `program`'s vertex stage,
 * or NULL if not found. */
SpirvResource *mglRendererProgramVertexAttribResource(Program *program,
                                                       GLuint attribute);

/* Returns true if the vertex attrib at `attribIndex` is a color input
 * (name starts with "gl_Color" or matches a color-input heuristic). */
bool mglRendererVertexAttribIsColorInput(Program *program, GLuint attribute);

/* Returns true if the vertex attrib at `attribIndex` uses the current-value
 * fallback (no bound vertex buffer, relies on ctx current attrib state). */
BOOL mglRendererVertexAttribUsesCurrentValue(VertexArray *vao, GLuint attribute);

#ifdef __cplusplus
}
#endif

#endif /* MGL_VERTEX_ATTRIB_QUERY_H */
