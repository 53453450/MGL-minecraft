/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_vertex_layout.h — vertex-descriptor state and the blend-state cache
 * update, moved out of MGLRenderer+VertexLayout.m (P0-1). */

#ifndef MGL_VERTEX_LAYOUT_H
#define MGL_VERTEX_LAYOUT_H

#include "mgl_render.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build the vertex descriptor for the current program/VAO/tessellation state.
 * Returns 1 on success. */
int mglRendererGenerateVertexDescriptorState(
    void *renderer, MGLRenderPipelineDescriptorState *state);

/* Repair out-of-range blend factors/equations, then upload every attachment's
 * blend state to the pipeline cache. */
void mglRendererUpdateBlendStateCache(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_VERTEX_LAYOUT_H */
