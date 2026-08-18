/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_gs_abi.h
 * MGL - fixed C ABI contract between the ObjC/C renderer runtime and the
 * AIR backend for the M3 geometry-shader compute-expansion path.
 *
 * The AIR GS kernel runs one compute work item per (input primitive x
 * invocation).  It consumes the vertex pre-pass records produced by the
 * rasterization-disabled VS capture, expands each input primitive through
 * the GS body, and writes a fixed per-work-item record run plus an indirect
 * draw-args record.  The renderer then issues one `drawPrimitivesIndirect`
 * per work item against that record run.
 *
 * Everything the renderer and the AIR backend must agree on lives here so
 * the two sides cannot silently drift when built separately.  All layout
 * invariants below are enforced by MGL_AIR_STATIC_ASSERT at file scope.
 *
 * History: the layout constants previously lived as hard-coded literals on
 * both sides (`2 + expandedVertices` in MGLRenderer+DrawSupport.m and
 * `cg.geometryRecordCount = 2u + cg.geometryOutputVertices` in
 * mgl_air_backend.cpp).  This header replaces those implicit contracts.
 */

#ifndef MGL_AIR_GS_ABI_H
#define MGL_AIR_GS_ABI_H

#include <stddef.h>
#include <stdint.h>

#include "mgl_buffer_slots.h" /* compute physical buffer-index boundary */
#include "mgl_shader_abi.h" /* MGLAIRPerVertexRecord, MGL_AIR_PER_VERTEX_* */

#if defined(__cplusplus)
#define MGL_AIR_STATIC_ASSERT(c, m) static_assert((c), m)
extern "C" {
#else
#define MGL_AIR_STATIC_ASSERT(c, m) _Static_assert((c), m)
#endif

/* =====================================================================
 * 1. GS compute-kernel reserved Metal buffer slots
 *
 * These slots are reserved on the *compute* encoder used for GS expansion.
 * They are safe to reuse on disjoint encoders (render pass, TCS/TES compute)
 * because the paths never run in the same encoder; register any new reuse
 * in mgl_buffer_slots.h so the reserved-slot conflict gate stays honest.
 * ===================================================================== */
enum {
    /* Vertex pre-pass records (MGLAIRPerVertexRecord stream, one record per
     * input vertex, packed [instance][primitive][vertex]).  Written by the
     * VS tessellation-capture variant at slot 29 on the *render* encoder,
     * read by the GS kernel here at slot 24 on the *compute* encoder. */
    MGL_AIR_GS_SLOT_INPUT  = 24,

    /* Expanded per-work-item output records.  Layout: see section 2. */
    MGL_AIR_GS_SLOT_OUTPUT = 28,

    /* MGLAIRGSIndirectArgs, one 16-byte record per work item.  See 3. */
    MGL_AIR_GS_SLOT_COUNTS = 29,

    /* GS transform-feedback record output (section 5).  Bound only when
     * the GS program is linked with transform feedback.  Numerically the
     * same slot as the TES XFB stream (MGL_AIR_TESS_SLOT_XFB_OUT) but the
     * encoders are disjoint. */
    MGL_AIR_GS_SLOT_XFB    = 31,

    /* GS XFB meta record (section 5): capacity/stride words written by the
     * renderer plus the GPU-atomic write cursor / written-byte counters.
     * Kept below 32: the AGX compiler encodes buffer slots in a 5-bit
     * mask, so slot indices >= 32 crash the shader-compiler service. */
    MGL_AIR_GS_SLOT_XFB_META = 27,
};

MGL_AIR_STATIC_ASSERT((int)MGL_AIR_GS_SLOT_XFB ==
                          (int)kMGLMaxMetalComputeBufferIndex,
                      "GS XFB must occupy the last physical compute slot");
MGL_AIR_STATIC_ASSERT((int)MGL_AIR_GS_SLOT_XFB <
                          (int)kMGLMaxMetalComputeBufferCount,
                      "GS XFB exceeds the physical compute slot domain");
MGL_AIR_STATIC_ASSERT((int)MGL_AIR_GS_SLOT_XFB >
                          (int)kMGLMaxMetalUserBufferIndex,
                      "GS XFB slot must remain internal-only");
MGL_AIR_STATIC_ASSERT(kMGLMaxMetalUserBufferCount ==
                          kMGLMaxMetalUserBufferIndex + 1,
                      "user buffer count must remain the 0..30 domain");
MGL_AIR_STATIC_ASSERT(kMGLMaxMetalComputeBufferCount ==
                          kMGLMaxMetalComputeBufferIndex + 1,
                      "compute physical count must include slot 31");

/* =====================================================================
 * 2. Output record layout
 *
 * Each work item owns a contiguous run of
 * `MGL_AIR_GS_RECORDS_PER_PRIMITIVE` per-vertex records, each of
 * `mglAIRPerVertexStrideForResources(...)` bytes:
 *
 *   [0]  rolling strip cache: previous emitted vertex
 *   [1]  rolling strip cache: previous-previous emitted vertex
 *   [2, 2+expanded_vertices)  expanded primitive vertices, written in
 *                             primitive order (points: 1 vertex per
 *                             primitive, lines: 2, triangles: 3).
 *
 * The two header records exist so the strip emulation (line/triangle strip)
 * can roll its previous-vertex cache without re-reading the emitted stream.
 * For points they are scratch.  The renderer binds the run at record 2 and
 * draws `vertex_count` vertices read from the indirect args.
 * ===================================================================== */
#define MGL_AIR_GS_HEADER_RECORDS 2u

/* Backend-neutral GS output primitive type (GLSL 4.60 4.3.8.2). */
typedef enum MGLAIRGSOutputPrimitive {
    MGL_AIR_GS_OUT_POINTS = 0,
    MGL_AIR_GS_OUT_LINE_STRIP,
    MGL_AIR_GS_OUT_TRIANGLE_STRIP,
} MGLAIRGSOutputPrimitive;

/* Maximum number of expanded vertices one input primitive can emit.
 * GLSL converts a `layout(points)` output to 1 vertex per point, a
 * `layout(line_strip, max_vertices=N)` to 2*(N-1) line vertices, and a
 * `layout(triangle_strip, max_vertices=N)` to 3*(N-2) triangle vertices. */
static inline uint32_t mglAIRGSExpandedVertices(
    MGLAIRGSOutputPrimitive output_type, uint32_t max_vertices)
{
    switch (output_type) {
    case MGL_AIR_GS_OUT_POINTS:
        return max_vertices;
    case MGL_AIR_GS_OUT_LINE_STRIP:
        return max_vertices > 1u ? 2u * (max_vertices - 1u) : 0u;
    case MGL_AIR_GS_OUT_TRIANGLE_STRIP:
    default:
        return max_vertices > 2u ? 3u * (max_vertices - 2u) : 0u;
    }
}

/* Total per-vertex records each work item owns (2 headers + expanded). */
static inline uint32_t mglAIRGSRecordsPerPrimitive(
    MGLAIRGSOutputPrimitive output_type, uint32_t max_vertices)
{
    return MGL_AIR_GS_HEADER_RECORDS +
           mglAIRGSExpandedVertices(output_type, max_vertices);
}

/* Byte offset of a work item's expanded-vertex region inside its record
 * run, given the per-record stride. */
static inline uint64_t mglAIRGSExpandedOffset(uint64_t record_stride)
{
    return (uint64_t)MGL_AIR_GS_HEADER_RECORDS * record_stride;
}

/* =====================================================================
 * 3. Counts / indirect draw-args record
 *
 * Each work item owns one 28-byte counts record (see
 * MGL_AIR_GS_COUNTS_RECORD_BYTES): a 16-byte MGLAIRGSIndirectArgs followed
 * by 12 bytes of kernel scratch.  The layout is byte-identical to
 * MTLDrawPrimitivesIndirectArguments for the first 16 bytes.
 *
 * The GS kernel writes exactly one draw parameter — the visible expanded
 * vertex count at counter 0 (MGLAIRGSIndirectArgs::vertex_count) — and
 * rolls its strip/emit state in the scratch words at offsets 16/20
 * (counters MGL_AIR_GS_COUNT_STRIP / MGL_AIR_GS_COUNT_EMITTED).  The
 * instance_count / base_vertex / base_instance words are preset by the
 * renderer (1 / 0 / 0) and are never written by the kernel, so the
 * rasterizing indirect draw is well-defined before the kernel runs:
 * exactly [record 2, record 2 + vertex_count) of the work item's run,
 * one instance.  (A vertex_count of 0 is a valid empty draw.)
 *
 * History: counters 1..2 previously aliased instanceCount/baseVertex,
 * which made every rasterizing draw use stripCount instances starting at
 * emitCount — a latent bug the pixel tests could not observe.  The ABI
 * now separates the two concerns at the cost of one 28-byte record.
 * ===================================================================== */
typedef struct MGLAIRGSIndirectArgs {
    uint32_t vertex_count;   /* 0: visible expanded vertices (GPU written)   */
    uint32_t instance_count; /* 1: draw instances; renderer presets 1        */
    uint32_t base_vertex;    /* 2: draw start; renderer presets 0            */
    uint32_t base_instance;  /* 3: draw base instance; renderer presets 0    */
} MGLAIRGSIndirectArgs;

/* Per-work-item counts record: indirect args + kernel scratch words. */
#define MGL_AIR_GS_COUNTS_ARGS_WORDS 4u   /* MGLAIRGSIndirectArgs           */
#define MGL_AIR_GS_COUNTS_SCRATCH_WORDS 3u /* strip / emit / reserved       */
#define MGL_AIR_GS_COUNTS_RECORD_WORDS \
    (MGL_AIR_GS_COUNTS_ARGS_WORDS + MGL_AIR_GS_COUNTS_SCRATCH_WORDS)
#define MGL_AIR_GS_COUNTS_RECORD_BYTES (MGL_AIR_GS_COUNTS_RECORD_WORDS * 4u)

/* Kernel scratch counter indices inside the record (offsets 16/20/24). */
enum {
    MGL_AIR_GS_COUNT_VERTEX_COUNT = 0, /* visible expanded vertex count  */
    MGL_AIR_GS_COUNT_STRIP         = 1, /* rolling strip emit count       */
    MGL_AIR_GS_COUNT_EMITTED       = 2, /* total EmitVertex calls         */
};

MGL_AIR_STATIC_ASSERT(sizeof(MGLAIRGSIndirectArgs) == 16u,
                      "GS indirect args must match MTLDrawPrimitivesIndirectArguments");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSIndirectArgs, vertex_count) == 0u,
                      "vertex_count must be offset 0");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSIndirectArgs, instance_count) == 4u,
                      "instance_count must be offset 4");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSIndirectArgs, base_vertex) == 8u,
                      "base_vertex must be offset 8");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSIndirectArgs, base_instance) == 12u,
                      "base_instance must be offset 12");
MGL_AIR_STATIC_ASSERT(MGL_AIR_GS_COUNTS_RECORD_BYTES == 28u,
                      "counts record is 16-byte args + 12-byte scratch");

/* =====================================================================
 * 4. Index-gather parameters (direct indexed GS draws, )
 *
 * The array-draw path feeds the VS capture `first/count` directly.  The
 * indexed path must first gather `count` indices from the element buffer
 * (honoring base-vertex and primitive restart) into a contiguous input
 * vertex stream so the VS capture and the GS kernel can share the same
 * per-vertex record layout.  This block is the fixed, backend-neutral
 * description of that gather; it is consumed by the renderer's gather
 * helper (CPU or GPU) — never interpreted by the AIR kernel itself.
 * ===================================================================== */
typedef struct MGLAIRGSIndexGatherParams {
    uint32_t index_type;              /* GL_UNSIGNED_BYTE / SHORT / INT     */
    uint32_t index_count;             /* number of indices in the draw      */
    uint64_t index_offset_bytes;      /* byte offset into the element buffer*/
    int32_t  base_vertex;             /* GL baseVertex added after fetch    */
    uint32_t primitive_restart;       /* bool: primitive restart enabled    */
    uint32_t restart_index;           /* restart marker for index_type      */
    uint32_t input_vertices;          /* vertices per GS input primitive    */
    uint32_t first_primitive;         /* first input primitive to gather    */
    uint32_t instance_count;          /* draw instance count                */
    uint32_t base_instance;           /* draw base instance                 */
} MGLAIRGSIndexGatherParams;

/* =====================================================================
 * 5. GS transform-feedback records (, multi-stream 2026-08-12)
 *
 * GS XFB output reuses the per-vertex record layout (position + varyings)
 * written by the GS kernel into a dedicated record buffer (slot 31), then
 * the renderer copies whole primitives back into the GL transform-feedback
 * store, honoring session offset / overflow the same way the TES XFB path
 * does (see MGLRenderer+Tessellation.m).
 *
 * Streams 0..3 (GL 4.6 §11.1.3.4, GLSL 4.60 §4.3.8.2/§8.13): only stream 0
 * is rasterized; streams 1..3 exist solely for transform feedback and are
 * only legal when the output primitive type is points.  The single
 * physical slot-31 buffer is split into per-stream segments
 * (capture_base = byte offset of the segment); each stream owns one
 * MGLAIRGSXFBStreamMeta with its own emitted-point, atomic cursor, and
 * written counters.  The emitted-point counter is independent of capture so
 * indexed PRIMITIVES_GENERATED remains valid when no XFB buffer is bound.
 *
 * The GS expanded output is variable-length (culled primitives contribute
 * nothing, GL 4.6 §13.2.4), so the kernel appends the visible expanded
 * vertices of each work item through a GPU-atomic cursor instead of a
 * compile-time fixed offset.  Slot 27 carries the 4-stream meta block:
 * the renderer pre-writes `stride` (0 disables capture) and `capacity`
 * (GL-visible store bytes available from the bound offset) per stream;
 * the kernel atomically reserves `visible * stride` bytes at the stream's
 * `cursor`, stores the visible records only when the reservation fits,
 * and counts the actually written bytes in `written`.
 * ===================================================================== */
#define MGL_AIR_GS_MAX_STREAMS 4u

typedef struct MGLAIRGSXFBStreamMeta {
    uint32_t stride;          /* bytes per XFB vertex; 0 = capture off    */
    uint32_t capacity_bytes;  /* store capacity from the bound offset     */
    uint32_t capture_base;    /* byte offset of this stream's segment in
                               * the slot-31 buffer (renderer preset)     */
    uint32_t generated;       /* emitted visible points (stream > 0 query) */
    uint64_t cursor;          /* atomic reservation cursor (GPU written)  */
    uint64_t written;         /* atomic written-byte counter (GPU written)*/
} MGLAIRGSXFBStreamMeta;

typedef struct MGLAIRGSXFBMeta {
    MGLAIRGSXFBStreamMeta stream[MGL_AIR_GS_MAX_STREAMS];
} MGLAIRGSXFBMeta;

MGL_AIR_STATIC_ASSERT(sizeof(MGLAIRGSXFBStreamMeta) == 32u,
                      "GS XFB stream meta is 12 + 4 generated + 8 + 8 bytes");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSXFBStreamMeta, stride) == 0u,
                      "stride must lead the stream meta");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSXFBStreamMeta, generated) == 12u,
                      "generated counter must remain in the ABI padding word");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSXFBStreamMeta, cursor) == 16u,
                      "cursor must be 64-bit aligned at offset 16");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSXFBStreamMeta, written) == 24u,
                      "written must be 64-bit aligned at offset 24");
MGL_AIR_STATIC_ASSERT(sizeof(MGLAIRGSXFBMeta) == 128u,
                      "GS XFB meta is 4 x 32-byte stream blocks");

/* =====================================================================
 * 6. Static layout invariants shared with the AIR backend
 * ===================================================================== */
MGL_AIR_STATIC_ASSERT(MGL_AIR_PER_VERTEX_STRIDE == 64u,
                      "per-vertex record stride changed; update both sides");
MGL_AIR_STATIC_ASSERT(MGL_AIR_PER_VERTEX_POSITION_OFFSET == 0u,
                      "position must stay at record offset 0");
MGL_AIR_STATIC_ASSERT(MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET == 16u,
                      "point size must stay at record offset 16");
MGL_AIR_STATIC_ASSERT(MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET == 20u,
                      "cull distances must stay at record offset 20");
MGL_AIR_STATIC_ASSERT(MGL_AIR_GS_HEADER_RECORDS == 2u,
                      "GS record run header count is part of the ABI");

/* =====================================================================
 * 7. Indexed-draw gather ABI (direct indexed GS, )
 *
 * For glDrawElements* the renderer gathers the element stream on the CPU
 * into a compact uint32 `gather` array (one entry per input vertex of each
 * primitive, restart markers removed and primitives re-grouped), then runs
 * the VS capture as a drawIndexedPrimitives against the original EBO with
 * the GL baseVertex — so stage_in still reads VBO[baseVertex + index].
 *
 * The GS kernel then locates each gl_in[] entry indirectly: the gather
 * entry holds the raw index value (Metal vertex_id, baseVertex NOT added),
 * and the input record lives at `gather[...] + instance * vertices_per_-
 * instance` in the (sparse) capture record stream.  This keeps the capture
 * record layout identical to the array path ([instance][vertex_id]).
 *
 * The gather buffer is only bound for indexed draws; the params constant
 * carries a gather_enabled flag so one kernel serves both paths.
 * ===================================================================== */
enum {
    /* GS indexed gather buffer (uint32 stream), compute encoder only.
     * Reused across disjoint encoders; see mgl_buffer_slots.h. */
    MGL_AIR_GS_SLOT_GATHER = 30,

    /* GS indexed gather params (setBytes constant), compute encoder only.
     * Ordinary stages use runtime-array size slot 25, but GS kernels move that
     * hidden table to MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX (23), so
     * `.length()` and indexed/array gather parameters can coexist. */
    MGL_AIR_GS_SLOT_GATHER_PARAMS = 25,
};

MGL_AIR_STATIC_ASSERT(MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX <
                          MGL_AIR_GS_SLOT_INPUT,
                      "GS runtime-size table must stay below the fixed ABI");
MGL_AIR_STATIC_ASSERT(MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX !=
                          MGL_AIR_GS_SLOT_GATHER_PARAMS,
                      "GS runtime-size table must not alias gather params");

typedef struct MGLAIRGSGatherParams {
    uint32_t vertices_per_instance;   /* record span per instance (capture
                                       * stride = max index + 1 for indexed,
                                       * = vertex count for array)          */
    uint32_t primitives_per_instance; /* primitives per instance            */
    uint32_t first_vertex;            /* capture firstVertex (0 for indexed)*/
    uint32_t gather_enabled;          /* 0 = array path, 1 = indexed path   */
} MGLAIRGSGatherParams;

MGL_AIR_STATIC_ASSERT(sizeof(MGLAIRGSGatherParams) == 16u,
                      "GS gather params is four 32-bit words");
MGL_AIR_STATIC_ASSERT(offsetof(MGLAIRGSGatherParams, vertices_per_instance) == 0u,
                      "vertices_per_instance must lead the params");
MGL_AIR_STATIC_ASSERT(MGL_AIR_GS_SLOT_GATHER == 30u,
                      "GS gather slot is 30");

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MGL_AIR_GS_ABI_H */
