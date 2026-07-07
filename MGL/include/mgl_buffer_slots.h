/*
 * mgl_buffer_slots.h
 * MGL
 *
 * Reserved Metal buffer slot registry.
 *
 * Metal vertex/compute pipelines expose 31 buffer slots (0..30).  MGL reserves
 * the high end of this range for internal use (tessellation, transform
 * feedback, gl_FragCoord fixup, cull-distance emulation, SPIRV-Cross runtime
 * sizing).  A low vertex-stage slot is also reserved for fixed-function point
 * size emulation.  GL user buffer bindings (UBO/SSBO/atomic-counter) MUST NOT
 * land in the reserved range — `mglBufferSlotIsReservedForStage` is the
 * conflict-detection gate used by `applyMSLResourceBindings`.
 *
 * IMPORTANT — cross-stage slot reuse:
 * Several slots are reused across disjoint pipeline stages (e.g. slot 28 is
 * `_mgl_patch_info` in the TCS/TES compute path AND `kMGLCullDistanceParams` in
 * the VS draw path).  This is safe because the paths never execute in the same
 * encoder.  The enum below documents the *primary* owner of each slot; see the
 * per-slot comments for reuse notes.
 *
 * Adding a new reserved slot:
 *   1. Pick the lowest free slot in [25, 30] for high internal slots, or a
 *      documented low slot if the feature must not collide with vertex attrs.
 *   2. Add an entry here with a doc comment + reuse notes.
 *   3. Update `mglBufferSlotIsReservedForStage` if the slot is stage-specific.
 *   4. Add a conflict check in `applyMSLResourceBindings` if user buffers could
 *      collide.
 */

#ifndef MGL_BUFFER_SLOTS_H
#define MGL_BUFFER_SLOTS_H

#include "glcorearb.h"

#ifdef __cplusplus
extern "C" {
#endif

/* SPIRV-Cross spvBufferSizeConstants slot for runtime-sized SSBO arrays.
 * Set via SPVC_COMPILER_OPTION_MSL_BUFFER_SIZE_BUFFER_INDEX.  Bound in all
 * stages that have `needs_buffer_size_buffer`. */
#define MGL_BUFFER_SIZE_BUFFER_INDEX 25u

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
     * TCS/TES compute dispatch path.  Reused as kMGLCullDistanceVertex in VS. */
    kMGLBufferSlot_IndirectParams   = 29,

    /* TES gl_in buffer (TCS output vertices).  Reused as
     * kMGLFragCoordParamsBufferIndex in FS — disjoint stages, no conflict. */
    kMGLBufferSlot_TESGlIn          = 30,

    /* TCS [[stage_in]] replacement buffer.  TCS compute kernel only.
     * Keep this below 25 so it cannot collide with SPIRV-Cross's
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

    /* ---- Vertex attribute slots ---- */

    /* Vertex attribute buffers start at slot 16 and grow upward.
     * Slots 0..14 are available for plain uniform buffers and low-index
     * resource bindings; slot 15 is reserved for point-size emulation. */
    kMGLVertexAttribBufferBase      = 16,

    /* Metal vertex buffer layout indices are 0..30 (count = 31). */
    kMGLMaxMetalVertexBufferIndex  = 30,
    kMGLMaxMetalVertexBufferCount  = 31,
} MGLReservedBufferSlot;

/* Returns GL_TRUE if `slot` is reserved by MGL for the given shader `stage`
 * and therefore MUST NOT be assigned to a GL user buffer (UBO/SSBO/atomic).
 *
 * `stage` is a _MAX_SHADER_TYPES index (see glm_context.h).  Pass -1 to check
 * against all stages conservatively.
 *
 * NOTE: slot 25 (MGL_BUFFER_SIZE_BUFFER_INDEX) is NOT considered reserved
 * here — SPIRV-Cross manages its own binding and it is intentionally
 * assignable to user SSBOs that need runtime-sized array sizing.  The
 * renderer binds the size buffer at slot 25 only when
 * `spirv[stage].needs_buffer_size_buffer` is true, and SPIRV-Cross's own
 * decoration logic avoids collisions with user bindings. */
GLboolean mglBufferSlotIsReservedForStage(GLuint slot, int stage);

/* Returns GL_TRUE if `slot` is reserved in ANY stage (conservative check
 * for callers that do not know the target stage). */
GLboolean mglBufferSlotIsReserved(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program that uses
 * tessellation (TCS and/or TES shader attached).  Slots 26-30 are reserved
 * by the TCS/TES compute dispatch path (TessFactor=26, PatchOutput=27,
 * PatchInfo=28, IndirectParams=29, TESGlIn=30).  Call this from
 * `applyMSLResourceBindings` when `pptr` has TCS/TES stages attached to
 * detect UBO/SSBO bindings that would silently collide with tessellation
 * reserved buffers. */
GLboolean mglBufferSlotIsReservedForTessellation(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program whose vertex shader
 * uses cull-distance emulation.  Slots 28 (params) and 29 (vertex data) are
 * reserved by the VS cull-distance emulation path. */
GLboolean mglBufferSlotIsReservedForCullDistance(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program whose fragment shader
 * uses the gl_FragCoord fixup.  Slot 30 is reserved by the FS FragCoord
 * params path. */
GLboolean mglBufferSlotIsReservedForFragCoordFixup(GLuint slot);

/* Human-readable name for a reserved slot, or NULL if not reserved.
 * Useful for conflict-reporting logs. */
const char *mglBufferSlotReservedName(GLuint slot);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BUFFER_SLOTS_H */
