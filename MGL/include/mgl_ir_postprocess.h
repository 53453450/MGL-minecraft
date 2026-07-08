/*
 * mgl_ir_postprocess.h
 * MGL
 *
 * IR-level shader postprocessing subsystem.
 *
 * This module owns the explicit IR postprocess phase that runs after
 * SPIRV-Cross reflection and before spvc_compiler_compile.  It replaces
 * ad-hoc MSL-string rewrites with SPIR-V decoration edits where possible,
 * keeping string-level patches as diagnostics or last-resort shims.
 *
 * Pipeline ordering (registered in mglRunIRPostprocessPipeline):
 *   1. ir_reflect_active_builtins   — cache path-detection flags from
 *      reflected builtin/resource data (no IR mutation).
 *   2. ir_reserve_internal_slots    — build the reserved-slot set for the
 *      current program/stage and stash it on the context (no IR mutation).
 *   3. ir_pre_map_buffer_bindings   — remap conflicting user buffers to free
 *      Metal slots via spvc_compiler_set_decoration (destructive IR edit).
 *   4. ir_validate_binding_uniqueness — assert no two user buffers share a
 *      Metal slot after remapping (diagnostic, no IR mutation).
 *   5. ir_fix_std140_array_strides  — repair std140 ArrayStride for SSBO
 *      members affected by a glslang bug (destructive IR edit).
 *
 * Env vars:
 *   MGL_DEBUG_IR_REMAP=1              log every IR-level binding decision.
 *   MGL_DISABLE_IR_REMAP=1            bypass ONLY the binding pre-mapping
 *                                     pass (ir_pre_map_buffer_bindings);
 *                                     std140 fix and other passes still run.
 *                                     Forces legacy string-level fallback for
 *                                     binding conflicts.
 *   MGL_ASSERT_NO_MSL_BINDING_REWRITE=1
 *                                     fail loudly if applyMSLResourceBindings
 *                                     still needs string-level replacement
 *                                     after IR pre-mapping.
 *
 * See docs/IR_LEVEL_POSTPROCESS_TODO.md for the design rationale and
 * remaining work items.
 */

#ifndef MGL_IR_POSTPROCESS_H
#define MGL_IR_POSTPROCESS_H

#include "glcorearb.h"
#include "glm_context.h"
#include "spirv_cross_c.h"
#include "spirv.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLIRPatchContext {
    GLMContext ctx;
    Program *program;
    int stage;
    spvc_compiler compiler;

    /* Cached path-detection flags, populated by ir_reflect_active_builtins.
     * Cached once so later passes (and the unified conflict predicate) do
     * not re-scan reflection/source on every call. */
    GLboolean has_tessellation;      /* TCS and/or TES attached. */
    GLboolean vs_uses_cull_distance; /* VS declares gl_CullDistance / mgl_CullDistance. */
    GLboolean fs_uses_frag_coord;    /* FS uses gl_FragCoord (triggers _mglFragCoordParams injection). */

    /* Reserved-slot bitmap for [0,30], populated by ir_reserve_internal_slots.
     * Bit N set means slot N is reserved by MGL for this program/stage and
     * must not be assigned to a user buffer. */
    GLuint reserved_slot_mask;

    /* Counters for diagnostics. */
    int remapped_count;
} MGLIRPatchContext;

typedef GLboolean (*MGLIRPatchFn)(MGLIRPatchContext *ctx);

/* Unified buffer-slot conflict detection.
 *
 * Returns GL_TRUE if `slot` is reserved by MGL for the given program/stage
 * or is outside Metal's valid buffer-slot range, and therefore MUST NOT be
 * assigned to a GL user buffer (UBO/SSBO/atomic).
 *
 * This is the single source of truth used by BOTH the IR pre-mapping path
 * (before spvc_compiler_compile) and the MSL string fallback
 * (applyMSLResourceBindings, after compile).  Keeping one predicate prevents
 * the two paths from drifting.
 *
 * Path detection:
 *   - Stage-specific slots (15 VS point-size, 24 TCS stage-in): from `stage`.
 *   - Tessellation slots 26-30: pptr->shader_slots[TCS/TES].
 *   - CullDistance slots 28-29: VS reflection (mgl_CullDistance in stage
 *     outputs) with GLSL-source fallback.
 *   - FragCoord slot 30: FS reflection (SpvBuiltInFragCoord in builtin
 *     inputs) with GLSL-source fallback. */
GLboolean mglBufferSlotConflictsForProgram(Program *pptr, int stage, GLuint slot);

/* Run the IR postprocess pipeline.
 *
 * Called from parseSPIRVShaderToMetal after reflection and before
 * spvc_compiler_compile.  Returns GL_TRUE on success.  When
 * MGL_DISABLE_IR_REMAP=1 is set, only the binding pre-mapping pass is
 * bypassed; diagnostic passes and std140 ArrayStride repair still run. */
GLboolean mglRunIRPostprocessPipeline(GLMContext ctx, Program *pptr, int stage,
                                       spvc_compiler compiler);

/* Returns GL_TRUE if the MGL_ASSERT_NO_MSL_BINDING_REWRITE gate is active.
 * Used by applyMSLResourceBindings to fail loudly when string-level
 * [[buffer(N)]] replacement is still required after IR pre-mapping. */
GLboolean mglAssertNoMSLBindingRewriteEnabled(void);

#ifdef __cplusplus
}
#endif

#endif /* MGL_IR_POSTPROCESS_H */
