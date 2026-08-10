/*
 * mgl_air_tess_abi.h
 * MGL - fixed C ABI contract for the M3 tessellation draw path
 * (TCS/TES compute expansion today; Metal 4 native tessellation target).
 *
 * Design doc: docs/AIR_M3_CPP_TODO.md §3 P0 ("把 TCS/TES native draw
 * contract 写成 value state").
 *
 * A GL_PATCHES draw is described by a single value-state struct
 * (MGLAIRTessDrawContract) that travels from the GL entry point into the
 * TCS/TES dispatchers.  Previously the same six scalars (first/count/
 * indexType/indices/baseVertex/instanceCount/baseInstance) were threaded
 * through every call and re-derived independently inside each dispatcher;
 * the per-patch/tess-factor/patch-varying layout numbers were bare
 * literals.  This header fixes both.
 *
 * All layout invariants are enforced by MGL_AIR_STATIC_ASSERT (C11
 * _Static_assert / C++11 static_assert) at file scope, mirroring
 * mgl_air_gs_abi.h.
 */

#ifndef MGL_AIR_TESS_ABI_H
#define MGL_AIR_TESS_ABI_H

#include <stddef.h>
#include <stdint.h>

#include "mgl_shader_abi.h" /* MGL_AIR_PER_VERTEX_*, SpirvResourceList */

#if defined(__cplusplus)
#define MGL_AIR_TESS_STATIC_ASSERT(c, m) static_assert((c), m)
extern "C" {
#else
#define MGL_AIR_TESS_STATIC_ASSERT(c, m) _Static_assert((c), m)
#endif

/* =====================================================================
 * 1. Tessellation draw contract (value state)
 *
 * All fields are plain values; no pointers into renderer state.  The GL
 * entry points build one of these per draw and pass it down; the TCS and
 * TES dispatchers read only from the contract.
 * ===================================================================== */
typedef struct MGLAIRTessDrawContract {
    /* ---- draw shape ---- */
    uint32_t patch_vertices;   /* GL_PATCH_VERTICES (>= 1)              */
    uint32_t vertex_count;     /* raw vertex/index count passed to draw */
    uint32_t patch_count;      /* ceil(vertex_count / patch_vertices)   */
    uint32_t instance_count;   /* draw instance count (>= 1)            */
    uint32_t base_instance;    /* draw base instance                    */
    int32_t  first;            /* GL first (array draw)                 */

    /* ---- index source (0 index_type = non-indexed draw) ----
     * index_source is the GL draw's `indices` argument reinterpreted as a
     * byte offset into the bound element buffer (GL semantics: it is the
     * pointer value passed to glDrawElements).  The TCS stage-in gather
     * resolves it against the CPU-visible EBO mapping. */
    uint32_t index_type;       /* GL_UNSIGNED_BYTE / SHORT / INT, or 0  */
    uint64_t index_source;     /* byte offset of first index (indices)  */
    uint64_t index_count;      /* indices consumed by this draw         */
    int32_t  base_vertex;      /* GL baseVertex (indexed draws)         */
    uint32_t primitive_restart;/* bool: restart enabled                 */
    uint32_t restart_index;    /* restart marker for index_type         */

    /* ---- tess factor layout ---- */
    uint32_t tess_factor_bytes_per_patch; /* 12 = quad half factors     */
    uint32_t tess_gen_mode;   /* GL_TRIANGLES / GL_QUADS / GL_ISOLINES  */
    uint32_t point_mode;      /* bool: layout(point_mode)               */

    /* ---- patch / per-vertex interface ---- */
    uint32_t tcs_out_vertices;    /* TCS layout(vertices=N) out; falls
                                   * back to patch_vertices             */
    uint32_t per_vertex_out_stride; /* TCS/TES per-vertex record stride */
    uint32_t patch_out_stride;    /* per-patch varying record stride    */
} MGLAIRTessDrawContract;

/* ---- Tess factor buffer layouts (Metal) ----
 * The compute path (buffer 26) always allocates the quad layout because
 * MTLQuadTessellationFactorsHalf covers all three modes at the cost of two
 * unused half floats for triangles/isolines. */
#define MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES 12u /* 4 edge + 2 inner halves */
#define MGL_AIR_TESS_FACTOR_TRI_HALF_BYTES   8u /* 3 edge + 1 inner half   */

/* =====================================================================
 * 2. Per-patch varying record layout
 *
 * Patch-qualified varyings (and per-vertex stage outputs, where the
 * resource list carries them) use one 16-byte slot per reflected location.
 * This mirrors the stable-location ABI used by mglAIRPerVertexStrideFor-
 * Resources: stride = max(16, (max_location + 1) * 16) over the
 * `is_per_patch` resources of the given list (caller passes the stage
 * output list of the writer stage, e.g. TCS _STAGE_OUTPUT_RES).
 * ===================================================================== */
static inline uint32_t mglAIRPatchVaryingStride(const SpirvResourceList *resources)
{
    uint32_t stride = 16u;
    if (!resources || !resources->list) return stride;
    for (uint32_t i = 0; i < resources->count; i++) {
        const SpirvResource *resource = &resources->list[i];
        if (!resource->is_per_patch) continue;
        if (resource->location >= 0x0fffffffu) continue;
        uint32_t end = (resource->location + 1u) * 16u;
        if (end > stride) stride = end;
    }
    return stride;
}

/* =====================================================================
 * 3. Reserved Metal slots for the TCS/TES compute dispatch
 *
 * Canonical names (see mgl_buffer_slots.h for the cross-stage reuse
 * registry).  These are the values the backend and the renderer both
 * hard-code today; keeping them here prevents drift.
 * ===================================================================== */
enum {
    MGL_AIR_TESS_SLOT_TESS_FACTOR   = 26, /* TCS write / TES read factors */
    MGL_AIR_TESS_SLOT_PATCH_OUT     = 27, /* TCS patchOut / TES patchIn   */
    MGL_AIR_TESS_SLOT_PATCH_INFO    = 28, /* TES {patch_vertices,tcs_out_vert}
                                             * constant (setBytes)         */
    MGL_AIR_TESS_SLOT_TCS_OUTPUT    = 28, /* TCS spvOut stage output.
                                             * Same numeric slot as
                                             * PATCH_INFO, reused across
                                             * disjoint TCS/TES encoders.   */
    MGL_AIR_TESS_SLOT_INDIRECT      = 29, /* {vertex_count,instance_count}*/
    MGL_AIR_TESS_SLOT_GL_IN         = 30, /* TES gl_in (TCS output verts) */
    MGL_AIR_TESS_SLOT_TCS_STAGE_IN  = 24, /* TCS packed stage_in repl     */
};

MGL_AIR_TESS_STATIC_ASSERT(MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES == 12u,
                           "quad half tess factors are 12 bytes");
MGL_AIR_TESS_STATIC_ASSERT(offsetof(MGLAIRTessDrawContract, patch_vertices) == 0u,
                           "patch_vertices must lead the contract");
MGL_AIR_TESS_STATIC_ASSERT(MGL_AIR_TESS_SLOT_TCS_STAGE_IN == MGL_AIR_TESS_SLOT_GL_IN - 6u,
                           "TCS stage-in slot is 24");

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MGL_AIR_TESS_ABI_H */
