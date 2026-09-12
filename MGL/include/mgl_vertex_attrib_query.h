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
 * mgl_vertex_attrib_query.h
 * MGL
 *
 * Vertex Attrib Query Subsystem.
 *
 * Read-only queries over a Program's shader stage-input resources and
 * VertexArray attribute state.  Used by the per-draw attribute-binding path
 * to decide which vertex attributes need binding, which are color inputs,
 * and which use current-value fallbacks.
 *
 * All functions are pure (take Program* and VertexArray* params, no self/ivar).
 *
 * Dependencies: glm_context.h (Program, MGLShaderResource, MGLShaderResourceList,
 * VertexArray, MAX_ATTRIBS, _VERTEX_SHADER)
 * (_STAGE_INPUT_RES) + <strings.h> (strcasecmp).
 */

#ifndef MGL_VERTEX_ATTRIB_QUERY_H
#define MGL_VERTEX_ATTRIB_QUERY_H

#include "glcorearb.h"

#include <stdbool.h>
#include <stddef.h>
#ifdef __OBJC__
#endif

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returns true if `program`'s vertex stage has a stage-input resource
 * matching `attribIndex`. */
bool mglRendererProgramUsesVertexAttrib(Program *program, GLuint attribute);

/* Returns the MGLShaderResource for `attribIndex` in `program`'s vertex stage,
 * or NULL if not found. */
MGLShaderResource *mglRendererProgramVertexAttribResource(Program *program,
                                                       GLuint attribute);

/* Returns true if the vertex attrib at `attribIndex` is a color input
 * (name starts with "gl_Color" or matches a color-input heuristic). */
bool mglRendererVertexAttribIsColorInput(Program *program, GLuint attribute);

/* Returns true if the vertex attrib at `attribIndex` uses the current-value
 * fallback (no bound vertex buffer, relies on ctx current attrib state). */
bool mglRendererVertexAttribUsesCurrentValue(VertexArray *vao, GLuint attribute);

/* The VAO bound to the context, after the pointer-plausibility / hashtable
 * checks; drops the binding and returns NULL when it looks invalid.  Defined in
 * MGLRenderer.m (it owns the drop path) and declared here so C translation
 * units can use it; `where` is a label for the drop diagnostic. */
VertexArray *mglRendererGetValidatedVAO(GLMContext ctx, const char *where);

#ifdef __cplusplus
}
#endif

#endif /* MGL_VERTEX_ATTRIB_QUERY_H */
