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
 * mgl_vertex_attrib_plan.h
 * MGL
 *
 * Vertex-attribute buffer map plan.
 *
 * Mapping the VAO's attribute buffers onto Metal vertex buffer slots used to
 * live in MGLRenderer+Buffer.m (mapVertexAttributeBuffersToBufferMap:).  Its
 * decisions are pure data -- which attributes are candidates, which resolved
 * bindings share one stream, which slot each stream takes, and the capacity /
 * index-overflow guards -- so they live here.
 *
 * Resolving a single attribute against live GL state stays behind a callback
 * supplied by the caller (MGLRenderer+Buffer.m passes
 * mglResolveVertexAttribForPlan).  That seam is what keeps this layer free of
 * Metal / ObjC / GL state, and therefore unit-testable:
 * test_legacy_compat/test_buffer_plan.c drives it with fixtures and a stub
 * resolver.
 *
 * The implementation deliberately lives in its own TU rather than in
 * mgl_buffer_plan.c: the latter's plan-build path pulls in the shader-resource
 * and program-resource layers (Foundation-flavoured), which would drag ObjC /
 * Metal into this harness.
 */

#ifndef MGL_VERTEX_ATTRIB_PLAN_H
#define MGL_VERTEX_ATTRIB_PLAN_H

#include "glm_context.h"
#include "mgl_vertex_attrib_binding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve attribute `attribute` into `out`; return 0 on success and non-zero
 * when the attribute has no usable buffer (the plan then skips it). */
typedef int (*MGLVertexAttribResolveFn)(void *user, GLuint attribute,
                                        MGLResolvedVertexAttribBinding *out);

typedef struct MGLVertexAttribBufferPlanInput {
    /* Attributes to consider, one bit per attribute index: enabled in the VAO
     * (or every attribute when the VAO carries no explicit enable mask) and
     * consumed by the program's vertex stage. */
    uint32_t candidate_mask;
    /* Program stage-input resource count; drives the mismatch warning when
     * fewer buffers were mapped than the program declares. */
    int stage_input_count;
    int stage;                     /* GL stage id (diagnostic only)          */
    uint32_t map_capacity;         /* MAX_MAPPED_BUFFERS                     */
    const void *pipeline_state;    /* opaque diagnostic payload               */
    const void *index_buffer_metal;/* opaque diagnostic payload               */
    const void *vao;               /* opaque diagnostic payload (VAO pointer) */
    MGLVertexAttribResolveFn resolve;
    void *resolve_user;
} MGLVertexAttribBufferPlanInput;

/* Append one BufferMap entry per distinct attribute stream to `buffer_map`
 * (starting at buffer_map->count) and stamp each entry's attribute_mask with
 * the attributes it serves.  Attributes that share a buffer object (same name
 * and target) and the same stride/divisor are grouped into one entry whose
 * buffer_base_index is the Metal vertex buffer slot; per-attribute offsets stay
 * in the vertex descriptor.  Returns 0 on success, -1 when the map has no room
 * for the first entry.  `buffer_map` must be non-NULL. */
int mglRenderPlanVertexAttribBuffers(BufferMapList *buffer_map,
                                     const MGLVertexAttribBufferPlanInput *in);

/* Value predicate: does `count` still fit in a map of `max` slots?  Owned by
 * this TU (pure value predicate, no dependencies) so the plan layer and its
 * harness stay free of heavy includes; mgl_render.h keeps the declaration for
 * the remaining callers. */
int mglRenderMappedBufferCountOK(uint32_t count, uint32_t max);

#ifdef __cplusplus
}
#endif

#endif /* MGL_VERTEX_ATTRIB_PLAN_H */
