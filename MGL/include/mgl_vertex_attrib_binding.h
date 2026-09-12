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
 * mgl_vertex_attrib_binding.h
 * MGL
 *
 * Resolved vertex-attribute binding value type + resolver.
 *
 * This is the seam between the ObjC renderer (which owns current GL state and
 * the validated buffer table) and the pure-C binding plan layer
 * (`mgl_buffer_plan.c`): the plan decides *how* resolved attribute bindings
 * are grouped into Metal vertex buffer slots, while resolving an attribute
 * against the live GL state stays on the caller's side of the seam.
 *
 * Moved out of MGLRenderer+Draw_Private.h (which imports ObjC headers) so the
 * plan layer and its unit-test harness can use the type without dragging
 * ObjC / Metal / LLVM.  Only `glm_context.h` types are referenced.
 */

#ifndef MGL_VERTEX_ATTRIB_BINDING_H
#define MGL_VERTEX_ATTRIB_BINDING_H

#include "glcorearb.h"
#include "glm_context.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Resolved vertex-attrib binding === */
typedef struct MGLResolvedVertexAttribBinding_t {
    const VertexAttrib *attrib;
    Buffer *buffer;
    GLintptr binding_offset;
    GLuint stride;
    GLuint divisor;
    GLintptr relativeoffset;
    GLuint binding_index;
    bool uses_binding_table;
} MGLResolvedVertexAttribBinding;

/* Resolve attribute `attribute` of `vao` against the current GL state
 * (buffer bindings table, validated buffer object) and fill `out`.
 * Returns false when the attribute has no usable buffer (the caller skips it).
 * Defined in MGLRenderer.m (C linkage); `ctx` may be required for validation. */
bool mglRendererResolveVertexAttribBinding(GLMContext ctx,
                                           VertexArray *vao,
                                           GLuint attribute,
                                           const char *where,
                                           MGLResolvedVertexAttribBinding *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_VERTEX_ATTRIB_BINDING_H */
