/*
 * mgl_buffer_slots.h
 * MGL
 *
 * Reserved Metal buffer slot registry.
 *
 * MGL user/vertex-buffer tables expose indices 0..30 (count 31).  Fixed AIR
 * compute ABIs may additionally use physical index 31; current GS/TES kernels
 * exercise that index on Apple M4, while indices >= 32 cross the AGX 5-bit
 * compiler boundary.  Slot 31 is therefore internal-only and MUST NOT expand
 * the ordinary user-resource or vertex-layout tables to 32 entries.
 *
 * MGL reserves the high end of these domains for internal use (tessellation,
 * transform feedback, gl_FragCoord fixup, cull-distance emulation, runtime
 * sizing).  A low vertex-stage slot is also reserved for fixed-function point
 * size emulation.  GL user buffer bindings (UBO/SSBO/atomic-counter) MUST NOT
 * land in the reserved range; the program-aware link gate is
 * `mglBufferSlotConflictsForProgram`.
 *
 * IMPORTANT — cross-stage slot reuse:
 * Several slots are reused across disjoint pipeline stages (e.g. slot 28 is
 * `_mgl_patch_info` in the TCS/TES compute path AND `kMGLCullDistanceParams` in
 * the VS draw path).  This is safe because the paths never execute in the same
 * encoder.  The enum below documents the *primary* owner of each slot; see the
 * per-slot comments for reuse notes.
 *
 * Adding a new reserved slot:
 *   1. Pick the lowest free slot in [25, 30] for a user-visible stage or a
 *      documented compute-only physical slot in [25, 31].
 *   2. Add an entry here with a doc comment + reuse notes.
 *   3. Update `mglBufferSlotIsReservedForStage` if the slot is stage-specific.
 *   4. Extend `mglBufferSlotConflictsForProgram` and its link-time regression
 *      coverage if a reflected user buffer could collide.
 */

#ifndef MGL_BUFFER_SLOTS_H
#define MGL_BUFFER_SLOTS_H

#include "glcorearb.h"

#ifdef __cplusplus
extern "C" {
#endif

/* spvBufferSizeConstants slot for runtime-sized SSBO arrays.
 * Set via SPVC_COMPILER_OPTION_MSL_BUFFER_SIZE_BUFFER_INDEX.  Bound in all
 * ordinary stages that have `needs_runtime_array_size_buffer`. */
#define MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX 25u

/* GS and compute-TES already use slot 25 for gather parameters.  Their AIR
 * kernels place the hidden runtime-array size table at slot 23 instead. */
#define MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX 23u

typedef enum {
    /* Fixed-function point size parameter for vertex shaders.  Vertex attribute
     * buffers start at slot 16, so this does not collide with VAO streams. */
    kMGLPointSizeBufferIndex      = 15,

    /* ---- Slots shared across the TCS/TES compute path ---- */

    /* Tessellation factor output buffer.  TCS writes, tessellator reads.
     * TCS/TES compute path only. */
    kMGLBufferSlot_TessFactor       = 26,

    /* Per-patch output buffer (TCS patchOut / TES patchIn).
     * TCS/TES compute path only. */
    kMGLBufferSlot_PatchOutput      = 27,

    /* Patch info constant ({patch_vertices_in, tcs_out_vertices}).
     * TCS/TES compute path only.  Reused as kMGLCullDistanceParams in VS —
     * see note below. */
    kMGLBufferSlot_PatchInfo        = 28,

    /* Indirect draw parameter buffer (vertexCount, instanceCount, ...).
     * TCS/TES compute dispatch path.  Reused as kMGLCullDistanceVertex in VS.
     * TES compute XFB uses the separate internal-only physical slot 31. */
    kMGLBufferSlot_IndirectParams   = 29,

    /* TES gl_in buffer (TCS output vertices).  Reused as
     * kMGLFragCoordParamsBufferIndex in FS — disjoint stages, no conflict. */
    kMGLBufferSlot_TESGlIn          = 30,

    /* TCS [[stage_in]] replacement buffer.  TCS compute kernel only.
     * Keep this below 25 so it cannot collide with
     * spvBufferSizeConstants at slot 25 or tessellation helper slots 26-30. */
    kMGLBufferSlot_TCSStageInRepl   = 24,

    /* ---- Slots reused by VS/FS draw path (NOT TCS/TES compute) ---- */

    /* Cull-distance emulation params constant (VS path).  Same numeric slot
     * as kMGLBufferSlot_PatchInfo but only used in VS draw calls, never in
     * TCS/TES compute dispatch. */
    kMGLCullDistanceParamsBufferIndex = 28,

    /* Cull-distance emulation sibling-vertex data (VS path).  Same slot as
     * kMGLBufferSlot_IndirectParams, VS-only. */
    kMGLCullDistanceVertexBufferIndex = 29,

    /* gl_FragCoord fixup params constant (FS path).  Same slot as
     * kMGLBufferSlot_TESGlIn, FS-only. */
    kMGLFragCoordParamsBufferIndex = 30,

    /* LOD_BIAS_MAX uniform buffer (FS path only).  Holds a single float
     * (MAX_TEXTURE_LOD_BIAS) for GL 4.6 §8.14.1 eq 8.8:
     *   clamp(biastexobj + biasshader, -biasmax, biasmax)
     * The value is injected as `constant float& _mglLodBiasMax [[buffer(14)]]`
     * and referenced by clamp() in mglRewriteMSLBiasExpr.
     * Slot 14 is in the user UBO range; mglInjectMSLLodBiasParam checks
     * availability via strstr before injecting. */
    kMGLLodBiasMaxBufferIndex = 14,

    /* LOD_BIAS uniform buffer (FS path only).  Same numeric slot as
     * kMGLPointSizeBufferIndex (VS-only), disjoint stages, no conflict.
     * Holds an array of TEXTURE_UNITS floats for GL_TEXTURE_LOD_BIAS
     * emulation via MSL bias() injection.  Occupied only when the FS has
     * .sample() calls; mglPatchInjectLodBias checks slot availability
     * before injecting. */
    kMGLLodBiasBufferIndex = 15,

    /* ---- Vertex attribute slots ---- */

    /* Vertex attribute buffers start at slot 16 and grow upward.
     * Slots 0..14 are available for plain uniform buffers and low-index
     * resource bindings; slot 15 is reserved for point-size emulation. */
    kMGLVertexAttribBufferBase      = 16,

    /* Reflected user resources and vertex-buffer layout tables are capped at
     * 0..30.  The historical vertex names remain aliases because renderer
     * table sizes and pipeline signatures already use them. */
    kMGLMaxMetalUserBufferIndex    = 30,
    kMGLMaxMetalUserBufferCount    = 31,
    kMGLMaxMetalVertexBufferIndex  = kMGLMaxMetalUserBufferIndex,
    kMGLMaxMetalVertexBufferCount  = kMGLMaxMetalUserBufferCount,

    /* Physical compute-only ABI domain.  Slot 31 is reserved for fixed GS/TES
     * transform-feedback streams and is never assigned to a user resource. */
    kMGLMaxMetalComputeBufferIndex = 31,
    kMGLMaxMetalComputeBufferCount = 32,
} MGLReservedBufferSlot;

/* Returns GL_TRUE if `slot` is reserved by MGL for the given shader `stage`
 * and therefore MUST NOT be assigned to a GL user buffer (UBO/SSBO/atomic).
 *
 * `stage` is a _MAX_SHADER_TYPES index (see glm_context.h).  Pass -1 to check
 * against all stages conservatively.
 *
 * NOTE: this conservative helper does not know whether a stage uses the
 * runtime-array size table at slot 25.  Link-time callers must use
 * `mglBufferSlotConflictsForProgram`, which reserves slot 25 exactly when
 * `modules[stage].needs_runtime_array_size_buffer` is true. */
GLboolean mglBufferSlotIsReservedForStage(GLuint slot, int stage);

/* Returns GL_TRUE if `slot` is reserved in ANY stage (conservative check
 * for callers that do not know the target stage). */
GLboolean mglBufferSlotIsReserved(GLuint slot);

/* Returns GL_TRUE for the legacy cross-route tessellation helper range 26..30
 * (factors, patch output/info, indirect params, TES gl_in).  It intentionally
 * does not model route-specific slots 24, 25 or 31; link-time callers must use
 * `mglBufferSlotConflictsForProgram` for the exact TCS/native-TES/compute-TES
 * ownership set. */
GLboolean mglBufferSlotIsReservedForTessellation(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program whose vertex shader
 * uses cull-distance emulation.  Slots 28 (params) and 29 (vertex data) are
 * reserved by the VS cull-distance emulation path. */
GLboolean mglBufferSlotIsReservedForCullDistance(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program with a geometry
 * shader running on the M3 compute-expansion path (mgl_air_gs_abi.h).
 * The current GS compute ABI owns slots 24..31, including gather params,
 * output/count buffers and transform-feedback stream/meta buffers. */
GLboolean mglBufferSlotIsReservedForGeometry(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program whose fragment shader
 * uses the gl_FragCoord fixup.  Slot 30 is reserved by the FS FragCoord
 * params path. */
GLboolean mglBufferSlotIsReservedForFragCoordFixup(GLuint slot);

struct Program_t;

/* Returns the hidden runtime-array size-table slot emitted for this program's
 * stage.  GS and isolines/point-mode TES use the compute-ABI slot 23; all
 * other stages use the ordinary slot 25. */
GLuint mglRuntimeArraySizeBufferIndexForProgram(
    const struct Program_t *program, int stage);

/* Returns GL_TRUE when a reflected user buffer at `slot` would collide with
 * an internal buffer used by the active execution path for `program` and
 * `stage`.  Unlike the conservative registry helpers above, this query is
 * program-aware: ordinary programs are not rejected merely because they use
 * a numeric slot that is reserved by an inactive GS/TES/emulation path. */
GLboolean mglBufferSlotConflictsForProgram(const struct Program_t *program,
                                           GLuint slot,
                                           int stage);

/* Human-readable name for a reserved slot, or NULL if not reserved.
 * Useful for conflict-reporting logs. */
const char *mglBufferSlotReservedName(GLuint slot);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BUFFER_SLOTS_H */
