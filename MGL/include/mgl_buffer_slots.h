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
 * mgl_buffer_slots.h
 * MGL
 *
 * Reserved Metal buffer slot registry.
 *
 * Metal allows 31 buffer arguments per function (`air.max_device_buffers`),
 * so every stage, including the fixed AIR compute ABIs, uses indices 0..30.
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
 *   1. Pick the lowest free slot in [25, 30]; compute ABIs that already fill
 *      [23, 30] take slots below 23 (GS/TES XFB stream at 22).
 *   2. Add an entry here with a doc comment + reuse notes.
 *   3. Update `mglBufferSlotIsReservedForStage` if the slot is stage-specific.
 *   4. Extend `mglBufferSlotConflictsForProgram` and its link-time regression
 *      coverage if a reflected user buffer could collide.
 */

#ifndef MGL_BUFFER_SLOTS_H
#define MGL_BUFFER_SLOTS_H

#include "glcorearb.h"
#include <stdint.h>

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
     * TES compute XFB uses the separate internal slot 22. */
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

    /* Integer textureGather wrap/border table (VS and FS).  Indexed by
     * Metal texture slot; see MGLSamplerWrapSlot.  Slot 23 is the compute
     * ABI runtime-array table, but that kernel does not share this encoder. */
    kMGLSamplerWrapBufferIndex = 23,

    /* Emulated-MS plane offset for the GS passthrough VS: adds the current
     * sample-loop plane onto gl_Layer*stride so layered MS_ARRAY FBOs write
     * the forced sample slice (flat = layer * 8 + sample).  AIR places the
     * PTVS UBO at ubo_base after the GS-output SSBO at slot 0, so the Metal
     * location is 1 (GLSL layout(binding=N) is not the Metal slot). */
    kMGLEmulatedMSPlaneBufferIndex = 1,

    /* Default sparse texture commit-rectangle table slot.  Link relocates the
     * table when this or any other candidate is occupied by the stage. */
    kMGLSparseResidencyDefaultBufferIndex = 12,

    /* ARB_shader_draw_parameters host stamp for VS: {draw_id, base_vertex}
     * as two int32 words.  Bound only when the linked VS uses gl_DrawID /
     * gl_BaseVertex; Metal has no portable [[draw_id]], and [[base_vertex]]
     * equals vertexStart on non-indexed draws (≠ GL).  Slot 13 is below the
     * VAO attribute base (16) and unused by other VS reserved slots. */
    kMGLShaderDrawParamsBufferIndex = 13,

    /* Packed LOD bias table for VS and FS (AIR).  Layout:
     *   float biasmax;               // MAX_TEXTURE_LOD_BIAS
     *   float bias[TEXTURE_UNITS];   // TEXTURE_LOD_BIAS per Metal tex slot
     * GL 4.6 §8.14.1 eq 8.8: clamp(biastexobj + biasshader, ±biasmax).
     * Slot 14 is below the VAO attribute base and unused by VS point-size
     * (slot 15), so both stages can share this pack. */
    kMGLLodBiasMaxBufferIndex = 14,

    /* Legacy alias: older MSL injection used a separate bias array at 15.
     * AIR uses the packed table at kMGLLodBiasMaxBufferIndex instead.
     * Slot 15 remains VS point-size (kMGLPointSizeBufferIndex). */
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
} MGLReservedBufferSlot;

/* ---- Binding-state slot limits (C-safe home) ----
 * kMGLMaxBufferSlots: client-visible buffer binding slots the binding state
 * tracks (was a #define in the ObjC MGLRenderer_State.h).
 * kMGLMinimumStageBindingSize: smallest buffer range a stage binding may cover,
 * in bytes (was a static const in the ObjC MGLRenderer+Draw_Private.h). */
#define kMGLMaxBufferSlots 31
enum { kMGLMinimumStageBindingSize = 256 };

enum {
    /* Private post-depth coverage attachments, outside the sampled image's
     * ordinary low slots and below the upper end of the 128-slot range. */
    kMGLPostDepthTextureIndex = 120,
    kMGLPostStencilTextureIndex = 121,
    kMGLPostDepthSamplerIndex = 15,
    kMGLPostDepthParamsBufferIndex = 24,
};

#define MGL_SAMPLER_WRAP_SLOT_COUNT 16
typedef struct MGLSamplerWrapSlot {
    int32_t wrap_s;
    int32_t wrap_t;
    int32_t wrap_r;
    uint32_t reduction_mode; /* 0=average, 1=min, 2=max */
    int32_t border[4];
    uint32_t filters;        /* min_filter | (mag_filter << 16) */
    uint32_t border_float[4]; /* IEEE754 float border, independent of int view */
} MGLSamplerWrapSlot;

/* What the reserved-slot policy needs to know about one stage of a linked
 * program, by value.  mglBufferSlotFactsFromProgram is the only place that
 * reads the Program; mglBufferSlotConflicts and mglBufferSlotCheckResource are
 * pure and are what the link gate and the unit test use. */
typedef struct MGLBufferSlotFacts {
    int stage;                         /* GL stage index (_VERTEX_SHADER ...), -1 = none */
    uint32_t runtime_array_size_slot;  /* where this stage's hidden size table lives */
    int needs_runtime_array_size_buffer;
    int stage_has_samplers;
    int uses_point_size_params;        /* VS */
    int uses_shader_draw_params;       /* VS */
    int uses_cull_distance;            /* VS */
    int vertex_has_stage_inputs;       /* the program's vertex stage has inputs */
    int tess_eval_render_vertex;       /* TES */
    int tess_eval_compute;             /* TES */
    int gs_compute_route;              /* GS */
    int uses_frag_coord_params;        /* FS */
    uint32_t sparse_residency_slot;    /* selected commit-table buffer slot */
    int uses_sparse_residency;
} MGLBufferSlotFacts;

enum {
    MGL_BUFFER_SLOT_OK = 0,
    MGL_BUFFER_SLOT_EXCEEDS_LIMIT = 1, /* past the user slot limit */
    MGL_BUFFER_SLOT_RESERVED = 2       /* collides with an internal buffer */
};

struct Program_t;
void mglBufferSlotFactsFromProgram(const struct Program_t *program, int stage,
                                   MGLBufferSlotFacts *out);

/* GL_TRUE when `slot` is taken by an internal buffer of the stage described by
 * `facts`. */
GLboolean mglBufferSlotConflicts(const MGLBufferSlotFacts *facts, GLuint slot);

/* Check a reflected resource's slots [binding, binding + element_count): the
 * first one that passes the limit or hits a reserved buffer decides
 * the result, and *bad_slot (when non-NULL) is that slot. */
int mglBufferSlotCheckResource(const MGLBufferSlotFacts *facts,
                               uint32_t binding, uint32_t element_count,
                               uint32_t *bad_slot);

/* Pick a slot for the stage's sparse commit table without displacing user
 * resources.  The preferred slot is retained whenever it is available. */
GLboolean mglBufferSlotFindSparseResidency(const struct Program_t *program,
                                           int stage, GLuint preferred,
                                           GLuint *out_slot);

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
 * does not model route-specific slots 22, 24 or 25; link-time callers must use
 * `mglBufferSlotConflictsForProgram` for the exact TCS/native-TES/compute-TES
 * ownership set. */
GLboolean mglBufferSlotIsReservedForTessellation(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program whose vertex shader
 * uses cull-distance emulation.  Slots 28 (params) and 29 (vertex data) are
 * reserved by the VS cull-distance emulation path. */
GLboolean mglBufferSlotIsReservedForCullDistance(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program with a geometry
 * shader running on the M3 compute-expansion path (mgl_air_gs_abi.h).
 * The current GS compute ABI owns slots 22 and 24..30, including gather params,
 * output/count buffers and transform-feedback stream/meta buffers. */
GLboolean mglBufferSlotIsReservedForGeometry(GLuint slot);

/* Returns GL_TRUE if `slot` is reserved for a program whose fragment shader
 * uses the gl_FragCoord fixup.  Slot 30 is reserved by the FS FragCoord
 * params path. */
GLboolean mglBufferSlotIsReservedForFragCoordFixup(GLuint slot);

struct Program_t;

/* Returns the hidden runtime-array size-table slot emitted for this program's
 * stage.  GS and isolines/point-mode TES (compute or render-vertex) use the
 * compute-ABI slot 23; all other stages use the ordinary slot 25. */
GLuint mglRuntimeArraySizeBufferIndexForProgram(
    const struct Program_t *program, int stage);

/* GL_TRUE when `prog` declares any sampler/image resource for `stage`.
 * Single source of truth for the draw-time LOD-bias pack bind condition
 * (slot 14) and for the link-time gate below; the bind site
 * (mgl_render_pass_manager_ops.c) and this gate must never drift apart. */
int mglProgramStageHasSamplers(const struct Program_t *prog, int stage);

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
