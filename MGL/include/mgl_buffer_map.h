/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_buffer_map.h — GL-buffer-to-Metal mapping, formerly
 * MGLRenderer+Buffer.m (P0-1, log 102).
 *
 * The file held nine renderer methods whose bodies were struct/plan
 * arithmetic over C records plus a handful of renderer lookups; every renderer
 * fact they needed is either in the state areas (`ctx`, `core.activeState`,
 * `core.defaultDrawableWrittenSinceLastSwap`, `pipeline_cache`,
 * `tess_native_tes_active`) or in an existing C entry point, so the
 * translation needed no new port.
 *
 * The `where` labels the C entries pass to the shared validation helpers keep
 * the exact strings the Objective-C methods' __FUNCTION__ produced
 * ("-[MGLRenderer(Buffer) …]"), so diagnostics stay byte-identical.
 */

#ifndef MGL_BUFFER_MAP_H
#define MGL_BUFFER_MAP_H

#include "glm_context.h"
#include "mgl_types_buffer.h"          /* Buffer, BufferMapList */
#include "mgl_types_program.h"         /* Program */
#include "mgl_buffer_plan.h"           /* MGLStageBufferPlan */
#include "mgl_vertex_attrib_binding.h" /* MGLResolvedVertexAttribBinding */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Group the candidate vertex attributes and map shader buffer resources for one
 * stage into `buffer_map`.  False refuses the draw. */
bool mglRendererMapGLBuffersToMTLBufferMap(void *renderer,
                                           BufferMapList *buffer_map, int stage);

/* Replay the cached per-stage buffer binding plan into `buffer_map`. */
bool mglRendererMapShaderBufferResourcesViaPlan(
    void *renderer, BufferMapList *buffer_map, int stage, Program *program,
    const MGLStageBufferPlan *stage_plan);

/* Map the stage's shader buffer resources, rebuilding the plan when needed. */
bool mglRendererMapShaderBufferResourcesToBufferMap(void *renderer,
                                                    BufferMapList *buffer_map,
                                                    int stage);

/* Map the vertex (or native-TES) and fragment buffer lists. */
bool mglRendererMapBuffersToMTL(void *renderer);

/* Push a dirty buffer's CPU shadow into its Metal store. */
bool mglRendererUpdateDirtyBuffer(void *renderer, Buffer *ptr);

/* Does any buffer of the list still carry dirty bits? */
bool mglRendererCheckForDirtyBufferData(void *renderer,
                                        BufferMapList *buffer_map_list);

/* Upload every dirty buffer of the list into its Metal store. */
bool mglRendererUpdateDirtyBaseBufferList(void *renderer,
                                          BufferMapList *buffer_map_list);

/* Metal vertex buffer slot a GL attribute set maps to. */
int mglRendererGetVertexBufferIndexWithAttributeSet(void *renderer,
                                                    int attribute);

/* Converted vertex buffer for one attribute (+1 reference the caller owns, or
 * NULL when the conversion is unavailable).  `out_stride` may be NULL. */
void *mglBufferCreateConvertedVertexBufferForAttribKind(
    void *renderer, int attrib_kind, Buffer *source_buffer,
    const MGLResolvedVertexAttribBinding *resolved, uint32_t component_count,
    uint32_t type, int normalized, int dst_is_int, size_t *out_stride);

/* Release a reference returned by the function above. */
void mglBufferReleaseConvertedVertexBuffer(void *buffer);

/* === Copy-on-write snapshot pool: frame-generation gates ================
 * MGL note: the two snapshot wrappers that used to live next to these
 * (mglSnapshotSharedDirtyBuffer / mglSnapshotSharedBufferRange) had no caller
 * anywhere in the tree and were dropped with the file; the C++ backend
 * entry points they wrapped are unreferenced now as well. */
uint64_t mglAdvanceFrameGeneration(void);
void mglRecordFrameCompleted(uint64_t generation);
/* Also declared in mgl_batch_mtl_encode.h / mgl_index_buffer.h. */
void mglNoteBufferEncoded(Buffer *buf);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BUFFER_MAP_H */
