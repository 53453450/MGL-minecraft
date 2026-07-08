/*
 * mgl_ir_postprocess.c
 * MGL
 *
 * Implementation of the IR-level postprocessing subsystem.
 * See mgl_ir_postprocess.h for the architectural rationale and
 * docs/IR_LEVEL_POSTPROCESS_TODO.md for the design roadmap.
 *
 * This module is the single source of truth for:
 *   - Buffer-slot conflict detection (mglBufferSlotConflictsForProgram),
 *     shared by the IR pre-mapping path and the MSL string fallback so the
 *     two cannot drift.
 *   - The IR pass pipeline that runs between SPIRV-Cross reflection and
 *     spvc_compiler_compile.
 *
 * External dependencies:
 *   - Program / SpirvResource types (glm_context.h).
 *   - Reserved-slot predicates (mgl_buffer_slots.h).
 *   - SPIRV-Cross C API (spirv_cross_c.h) for decoration edits.
 */

#include "mgl_ir_postprocess.h"
#include "mgl_buffer_slots.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* === Env var gates (evaluated once and cached) ===
 *
 * Thread-safety: these caches use non-atomic int reads/writes, consistent
 * with MGL's single-threaded-per-context assumption (the GL context itself
 * is not thread-safe, and shader compilation happens on the context thread).
 * The first read of each flag performs an unsynchronized getenv()+store; a
 * concurrent reader could observe a stale -1 or a torn write in theory, but
 * this cannot occur under the current single-threaded-per-context model.
 * If MGL ever supports multi-threaded shader compilation, these need atomic
 * guards (e.g. stdatomic / OSAtomic) or dispatch_once for one-shot init. */

static int g_ir_remap_debug = -1;
static int g_ir_remap_disabled = -1;
static int g_assert_no_msl_rewrite = -1;

static GLboolean mglIRRemapDebugEnabled(void)
{
    if (g_ir_remap_debug < 0) {
        g_ir_remap_debug = getenv("MGL_DEBUG_IR_REMAP") != NULL ? 1 : 0;
    }
    return g_ir_remap_debug ? GL_TRUE : GL_FALSE;
}

static GLboolean mglIRRemapDisabled(void)
{
    if (g_ir_remap_disabled < 0) {
        g_ir_remap_disabled = getenv("MGL_DISABLE_IR_REMAP") != NULL ? 1 : 0;
    }
    return g_ir_remap_disabled ? GL_TRUE : GL_FALSE;
}

GLboolean mglAssertNoMSLBindingRewriteEnabled(void)
{
    if (g_assert_no_msl_rewrite < 0) {
        g_assert_no_msl_rewrite =
            getenv("MGL_ASSERT_NO_MSL_BINDING_REWRITE") != NULL ? 1 : 0;
    }
    return g_assert_no_msl_rewrite ? GL_TRUE : GL_FALSE;
}

/* === Path-detection helpers ===
 *
 * These compute the cached flags stored on MGLIRPatchContext.  They are also
 * used implicitly by mglBufferSlotConflictsForProgram so the unified
 * predicate stays consistent with the cached flags. */

static GLboolean mglProgramHasTessellation(Program *pptr)
{
    return (pptr->shader_slots[_TESS_CONTROL_SHADER] ||
            pptr->shader_slots[_TESS_EVALUATION_SHADER]) ? GL_TRUE : GL_FALSE;
}

/* Detects whether the vertex shader declares gl_CullDistance (renamed to
 * mgl_CullDistance by GLSL preprocessing in shaders.c).  Uses VS reflection
 * first (available for all stages since VS is parsed first), with a GLSL
 * source fallback for the rare case where VS reflection is incomplete. */
static GLboolean mglVSUsesCullDistance(Program *pptr)
{
    /* Reflection: scan VS stage outputs for mgl_CullDistance.  The GLSL
     * preprocessor renames gl_CullDistance -> mgl_CullDistance (shaders.c),
     * so the SPIR-V has a user variable (not a builtin) with that name.
     * This only happens when gl_CullDistance is declared OUTSIDE a
     * gl_PerVertex block — the block case keeps it as a builtin and the
     * cull-distance emulation path does NOT activate. */
    SpirvResourceList *vs_outs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    for (GLuint i = 0; i < vs_outs->count; i++) {
        if (vs_outs->list[i].name &&
            strstr(vs_outs->list[i].name, "mgl_CullDistance")) {
            return GL_TRUE;
        }
    }

    /* Fallback: GLSL source.  shader->src holds the ORIGINAL GLSL source
     * (not preprocessed), so it contains gl_CullDistance (never
     * mgl_CullDistance).  We must mirror the rename gate in shaders.c:
     * gl_CullDistance is renamed to mgl_CullDistance ONLY when the source
     * does NOT contain "gl_PerVertex".  When a gl_PerVertex block is
     * present, the rename is skipped and the cull-distance emulation path
     * (which injects [[buffer(28)]]/[[buffer(29)]]) does NOT activate, so
     * slots 28/29 must NOT be reserved.
     *
     * Checking gl_CullDistance without the gl_PerVertex guard would cause
     * a false positive for gl_PerVertex block users, leading to their
     * UBOs/SSBOs being wrongly remapped off slots 28/29. */
    if (pptr->shader_slots[_VERTEX_SHADER] &&
        pptr->shader_slots[_VERTEX_SHADER]->src) {
        const char *vs_src = pptr->shader_slots[_VERTEX_SHADER]->src;
        if (strstr(vs_src, "gl_CullDistance") &&
            !strstr(vs_src, "gl_PerVertex")) {
            return GL_TRUE;
        }
    }
    return GL_FALSE;
}

/* Detects whether the fragment shader uses gl_FragCoord, which triggers
 * _mglFragCoordParams [[buffer(30)]] injection by applyMSLFragCoordOriginFix.
 * Uses FS reflected builtins when available (FS stage being processed), with
 * a GLSL source fallback for cross-stage checks (e.g. when processing VS
 * before FS has been reflected). */
static GLboolean mglFSUsesFragCoord(Program *pptr)
{
    /* Reflection: check FS builtin inputs for gl_FragCoord.  The builtin
     * reflection loop in parseSPIRVShaderToMetal maps SpvBuiltInFragCoord
     * to "gl_FragCoord" and stores it in builtin_program_inputs[FS].  This
     * is populated when FS itself is being processed. */
    for (GLuint i = 0; i < pptr->builtin_program_input_count[_FRAGMENT_SHADER]; i++) {
        if (pptr->builtin_program_inputs[_FRAGMENT_SHADER][i].name &&
            strcmp(pptr->builtin_program_inputs[_FRAGMENT_SHADER][i].name,
                   "gl_FragCoord") == 0) {
            return GL_TRUE;
        }
    }

    /* Fallback: GLSL source.  Needed when processing non-FS stages (FS
     * reflection not yet populated) but slot 30 must still be reserved
     * across all stages because MGL binds the FragCoord params buffer in
     * the FS and the conservative cross-stage reservation avoids pipeline
     * state confusion. */
    if (pptr->shader_slots[_FRAGMENT_SHADER] &&
        pptr->shader_slots[_FRAGMENT_SHADER]->src) {
        if (strstr(pptr->shader_slots[_FRAGMENT_SHADER]->src, "gl_FragCoord")) {
            return GL_TRUE;
        }
    }
    return GL_FALSE;
}

/* === Unified conflict predicate === */

GLboolean mglBufferSlotConflictsForProgram(Program *pptr, int stage, GLuint slot)
{
    if (!pptr) {
        return GL_FALSE;
    }
    /* Fix #8: Metal vertex/compute pipelines expose 31 buffer slots (0..30),
     * so slot 31 (kMGLMaxMetalVertexBufferCount) and above are out of range
     * and always conflict.  The threshold is >= 31 (not >= 32) because slot
     * 31 is already invalid. */
    if (slot >= kMGLMaxMetalVertexBufferCount) {
        return GL_TRUE;
    }

    /* Conservative "any stage" check + stage-specific check. */
    GLboolean slot_conflicts = mglBufferSlotIsReserved(slot);
    if (!slot_conflicts && mglBufferSlotIsReservedForStage(slot, stage)) {
        slot_conflicts = GL_TRUE;
    }

    /* Tessellation slots 26-30: reserved when TCS/TES attached. */
    if (!slot_conflicts &&
        mglProgramHasTessellation(pptr) &&
        mglBufferSlotIsReservedForTessellation(slot)) {
        slot_conflicts = GL_TRUE;
    }

    /* CullDistance slots 28-29: reserved when VS uses cull distance. */
    if (!slot_conflicts &&
        mglBufferSlotIsReservedForCullDistance(slot) &&
        mglVSUsesCullDistance(pptr)) {
        slot_conflicts = GL_TRUE;
    }

    /* FragCoord slot 30: reserved (cross-stage) when FS uses gl_FragCoord. */
    if (!slot_conflicts &&
        mglBufferSlotIsReservedForFragCoordFixup(slot) &&
        mglFSUsesFragCoord(pptr)) {
        slot_conflicts = GL_TRUE;
    }

    return slot_conflicts;
}

/* === IR passes === */

/* --- std140 ArrayStride helpers (moved from program.c) ---
 *
 * These compute the correct std140/std430 ArrayStride for a single array
 * element and detect types affected by a glslang bug that emits the natural
 * (std430) stride instead of the std140 rounded stride.  Used by
 * ir_fix_std140_array_strides to repair SSBO member strides at IR level
 * before spvc_compiler_compile. */

static unsigned compute_std140_stride(spvc_basetype bt, unsigned vecsize,
                                      unsigned cols, spvc_bool row_major)
{
    if (bt != SPVC_BASETYPE_FP32 && bt != SPVC_BASETYPE_INT32 &&
        bt != SPVC_BASETYPE_UINT32)
        return 0;
    if (cols <= 1)
        return 16;
    unsigned num_vectors = row_major ? vecsize : cols;
    return num_vectors * 16;
}

static unsigned compute_std430_stride(spvc_basetype bt, unsigned vecsize,
                                      unsigned cols, spvc_bool row_major)
{
    if (bt != SPVC_BASETYPE_FP32 && bt != SPVC_BASETYPE_INT32 &&
        bt != SPVC_BASETYPE_UINT32)
        return 0;
    if (cols <= 1)
    {
        unsigned natural = vecsize * 4;
        unsigned align = (vecsize >= 3) ? 16 : natural;
        return (natural + align - 1) / align * align;
    }
    unsigned vector_components = row_major ? cols : vecsize;
    unsigned num_vectors = row_major ? vecsize : cols;
    unsigned natural = vector_components * 4;
    unsigned align = (vector_components >= 3) ? 16 : natural;
    unsigned matrix_stride = (natural + align - 1) / align * align;
    return num_vectors * matrix_stride;
}

static GLboolean is_glslang_bug_affected(spvc_basetype bt, unsigned vecsize,
                                         unsigned cols, spvc_bool row_major)
{
    if (bt != SPVC_BASETYPE_FP32 && bt != SPVC_BASETYPE_INT32 &&
        bt != SPVC_BASETYPE_UINT32)
        return GL_FALSE;
    if (cols <= 1)
        return (bt != SPVC_BASETYPE_FP32 && vecsize == 2) ? GL_TRUE : GL_FALSE;
    unsigned vector_components = row_major ? cols : vecsize;
    return (vector_components == 2) ? GL_TRUE : GL_FALSE;
}

/* Pass 1: Reflect active builtins and cache path-detection flags.
 * No IR mutation — pure reflection/state read.
 *
 * Fix #9: these cached flags are NOT dead — they are consumed by
 * irBufferSlotConflictsForContext (which reads ctx->has_tessellation,
 * ctx->vs_uses_cull_distance, ctx->fs_uses_frag_coord directly instead of
 * re-computing them).  irBufferSlotConflictsForContext is in turn called by
 * the downstream passes ir_reserve_internal_slots (pass 2) and
 * ir_pre_map_buffer_bindings (pass 3), so the cache is read on every conflict
 * check rather than re-scanning reflection/GLSL source each time.  The public
 * mglBufferSlotConflictsForProgram (used by the MSL string fallback, which has
 * no MGLIRPatchContext) still re-computes via mglVSUsesCullDistance etc.
 * because it lacks a cache — that is expected, not dead code. */
static GLboolean ir_reflect_active_builtins(MGLIRPatchContext *ctx)
{
    ctx->has_tessellation = mglProgramHasTessellation(ctx->program);
    ctx->vs_uses_cull_distance = mglVSUsesCullDistance(ctx->program);
    ctx->fs_uses_frag_coord = mglFSUsesFragCoord(ctx->program);
    return GL_TRUE;
}

static GLboolean irBufferSlotConflictsForContext(const MGLIRPatchContext *ctx,
                                                 GLuint slot)
{
    if (!ctx) {
        return GL_FALSE;
    }
    if (slot >= kMGLMaxMetalVertexBufferCount) {
        return GL_TRUE;
    }

    GLboolean slot_conflicts = mglBufferSlotIsReserved(slot);
    if (!slot_conflicts &&
        mglBufferSlotIsReservedForStage(slot, ctx->stage)) {
        slot_conflicts = GL_TRUE;
    }
    if (!slot_conflicts &&
        ctx->has_tessellation &&
        mglBufferSlotIsReservedForTessellation(slot)) {
        slot_conflicts = GL_TRUE;
    }
    if (!slot_conflicts &&
        ctx->vs_uses_cull_distance &&
        mglBufferSlotIsReservedForCullDistance(slot)) {
        slot_conflicts = GL_TRUE;
    }
    if (!slot_conflicts &&
        ctx->fs_uses_frag_coord &&
        mglBufferSlotIsReservedForFragCoordFixup(slot)) {
        slot_conflicts = GL_TRUE;
    }

    return slot_conflicts;
}

/* Pass 2: Build the reserved-slot bitmap for this program/stage.
 * No IR mutation — populates ctx->reserved_slot_mask. */
static GLboolean ir_reserve_internal_slots(MGLIRPatchContext *ctx)
{
    ctx->reserved_slot_mask = 0;
    for (GLuint s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
        if (irBufferSlotConflictsForContext(ctx, s)) {
            ctx->reserved_slot_mask |= (1u << s);
        }
    }

    if (mglIRRemapDebugEnabled()) {
        fprintf(stderr,
                "MGL IR DEBUG: program=%u stage=%d reserved slots:",
                ctx->program->name, ctx->stage);
        for (GLuint s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (ctx->reserved_slot_mask & (1u << s)) {
                fprintf(stderr, " %u", s);
            }
        }
        fprintf(stderr, " (tess=%d cull=%d fragcoord=%d)\n",
                ctx->has_tessellation, ctx->vs_uses_cull_distance,
                ctx->fs_uses_frag_coord);
    }
    return GL_TRUE;
}

/* Keep in sync with program.c's synthetic sampler location namespace. */
#define MGL_IR_SYNTHETIC_SAMPLER_LOCATION_BASE 0x4000

static GLboolean irUniformNameLooksSamplerLike(const char *name)
{
    if (!name || !*name) {
        return GL_FALSE;
    }
    return (strstr(name, "Sampler") != NULL ||
            strcmp(name, "CloudFaces") == 0) ? GL_TRUE : GL_FALSE;
}

static GLboolean irGLTypeLooksSamplerLike(GLuint gl_type)
{
    switch (gl_type) {
        case GL_SAMPLER_1D:
        case GL_SAMPLER_2D:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_1D_SHADOW:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_2D_RECT:
        case GL_SAMPLER_2D_RECT_SHADOW:
        case GL_SAMPLER_1D_ARRAY:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_1D_ARRAY_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_BUFFER:
        case GL_INT_SAMPLER_1D:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_2D_RECT:
        case GL_INT_SAMPLER_1D_ARRAY:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_BUFFER:
        case GL_UNSIGNED_INT_SAMPLER_1D:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
        case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_BUFFER:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_CUBE_MAP_ARRAY:
        case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW:
        case GL_INT_SAMPLER_CUBE_MAP_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean irUniformConstantLooksSamplerLike(const SpirvResource *res)
{
    if (!res) {
        return GL_FALSE;
    }
    return (res->image_dim != 0u ||
            res->uniform_location >= MGL_IR_SYNTHETIC_SAMPLER_LOCATION_BASE ||
            irUniformNameLooksSamplerLike(res->name) ||
            irGLTypeLooksSamplerLike(res->gl_type)) ? GL_TRUE : GL_FALSE;
}

/* Helper: classify a resource as buffer-backed (eligible for IR remap).
 *
 * Fix #5: UNIFORM_CONSTANT is only buffer-backed when it is not sampler-like.
 * Image/sampler UNIFORM_CONSTANT entries occupy texture/sampler slots, not
 * [[buffer(N)]] slots, so remapping them as buffers would corrupt binding
 * assignment.  Mirror program.c's sampler detection where possible:
 * image_dim, synthetic sampler locations, Minecraft sampler naming, and the
 * reflected GL sampler type (the IR resource cache no longer carries the
 * original SPIRV-Cross basetype). */
static GLboolean irResourceIsBufferBacked(SpirvResource *res, int res_type)
{
    if (!res) return GL_FALSE;
    switch (res_type) {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
        case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
        case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
            return GL_TRUE;
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
            return irUniformConstantLooksSamplerLike(res) ? GL_FALSE : GL_TRUE;
        default:
            return GL_FALSE;
    }
}

/* Pass 3: Pre-map conflicting buffer bindings to free Metal slots.
 * Destructive IR edit — calls spvc_compiler_set_decoration. */
static GLboolean ir_pre_map_buffer_bindings(MGLIRPatchContext *ctx)
{
    Program *pptr = ctx->program;
    int stage = ctx->stage;
    spvc_compiler compiler = ctx->compiler;

    /* Fix #5: UNIFORM_CONSTANT is included so non-sampler-like uniforms get
     * remapped off reserved slots; sampler/image entries are filtered out by
     * irResourceIsBufferBacked and keep their texture/sampler namespace. */
    const int buffer_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT
    };

    /* Build used-slot map.  This includes:
     *   - Slots already occupied by user buffers (their current Metal slot).
     *   - Slots reserved by MGL (from the reserved_slot_mask).
     * This prevents the free-slot search from picking a reserved slot or
     * a slot already taken by another user buffer. */
    GLboolean used_slots[kMGLMaxMetalVertexBufferCount] = {0};

    /* Mark reserved slots as used so the free-slot search skips them
     * without needing a separate conflict check on every candidate. */
    for (GLuint s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
        if (ctx->reserved_slot_mask & (1u << s)) {
            used_slots[s] = GL_TRUE;
        }
    }

    /* Mark slots occupied by existing user buffers.  Use res->binding (the
     * current Metal slot) rather than gl_binding, since a prior pass could
     * have already moved it.  At entry, res->binding == gl_binding, but
     * using res->binding is robust against future pass insertion. */
    for (size_t t = 0; t < sizeof(buffer_resource_types) / sizeof(buffer_resource_types[0]); t++) {
        int res_type = buffer_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (!irResourceIsBufferBacked(res, res_type)) {
                continue;
            }
            GLuint current = res->binding;
            if (current < kMGLMaxMetalVertexBufferCount) {
                used_slots[current] = GL_TRUE;
            }
        }
    }

    /* Remap pass. */
    for (size_t t = 0; t < sizeof(buffer_resource_types) / sizeof(buffer_resource_types[0]); t++) {
        int res_type = buffer_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (!irResourceIsBufferBacked(res, res_type)) {
                continue;
            }

            /* Current Metal slot: read the actual SpvDecorationBinding from
             * the IR (source of truth for what MSL will emit).  Falls back
             * to res->binding if the decoration read fails. */
            GLuint current_slot = res->binding;
            if (spvc_compiler_has_decoration(compiler, res->_id, SpvDecorationBinding)) {
                current_slot = spvc_compiler_get_decoration(compiler, res->_id,
                                                            SpvDecorationBinding);
            }

            /* Check conflict against the unified predicate. */
            if (!irBufferSlotConflictsForContext(ctx, current_slot)) {
                continue; /* no conflict — nothing to do */
            }

            /* Find a free slot in the low user-buffer range.
             *
             * Fix #7: vertex shaders must stop before slot 16
             * (kMGLVertexAttribBufferBase) so vertex-stage UBOs are never
             * assigned to vertex attribute slots 16-24, which Metal uses for
             * vertex buffer bindings.  Slot 15 (kMGLPointSizeBufferIndex,
             * point-size emulation) is still excluded for VS — it is reserved
             * via mglBufferSlotIsReservedForStage, so used_slots[15] is set
             * by the reserved_slot_mask and skipped by the search below.
             * Non-vertex stages can use 0..24; 25 is SPIRV-Cross's
             * runtime-sized SSBO size buffer. */
            const GLuint free_limit =
                (stage == _VERTEX_SHADER) ? (GLuint)kMGLVertexAttribBufferBase
                                          : (GLuint)MGL_BUFFER_SIZE_BUFFER_INDEX;
            GLuint free_slot = kMGLMaxMetalVertexBufferCount;
            for (GLuint s = 0; s < free_limit; s++) {
                if (!used_slots[s]) {
                    free_slot = s;
                    break;
                }
            }

            if (free_slot >= kMGLMaxMetalVertexBufferCount) {
                fprintf(stderr,
                        "MGL IR PRE-REMAP: no free slot for program=%u stage=%d "
                        "%s buffer '%s' (id=%u) at slot %u "
                        "(will try string-level fallback)\n",
                        pptr->name, stage,
                        res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER ? "UBO" :
                        res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ? "SSBO" :
                        res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER ? "atomic" :
                        res_type == SPVC_RESOURCE_TYPE_PUSH_CONSTANT ? "push" : "buffer",
                        res->name ? res->name : "(null)",
                        (unsigned)res->_id,
                        (unsigned)current_slot);
                continue;
            }

            /* Remap at IR level.  spvc_compiler_compile will emit
             * [[buffer(free_slot)]] instead of [[buffer(current_slot)]]. */
            spvc_compiler_set_decoration(compiler, res->_id,
                                         SpvDecorationBinding, free_slot);

            /* Update reflection: binding (Metal slot) changes to free_slot.
             * gl_binding (GL client binding point) is PRESERVED so GL-side
             * buffer binding lookups (glUniformBlockBinding etc.) keep
             * working.
             *
             * Arrayed UBO/SSBO note: ubo_array_bindings[] holds the GL
             * client bindings for each array element (gl_binding + ai).
             * These are GL-side and MUST NOT be touched — Metal sees the
             * whole array as a single [[buffer(free_slot)]].  The runtime
             * buffer lookup uses gl_binding/ubo_array_bindings to find the
             * GL buffer, then maps it to the Metal slot via res->binding. */
            res->binding = free_slot;
            used_slots[free_slot] = GL_TRUE;
            /* Note: used_slots[current_slot] is intentionally left set.
             * current_slot is a reserved MGL slot (outside [0,24] search
             * range), so leaving it marked prevents no false free-slot hit.
             * If the free-slot search range is ever extended to include
             * reserved slots, clear it here: used_slots[current_slot] = GL_FALSE; */

            ctx->remapped_count++;

            if (mglIRRemapDebugEnabled()) {
                fprintf(stderr,
                        "MGL IR PRE-REMAP: program=%u stage=%d %s buffer '%s' "
                        "(id=%u) gl_binding=%u remapped slot %u -> %u\n",
                        pptr->name, stage,
                        res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER ? "UBO" :
                        res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ? "SSBO" :
                        res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER ? "atomic" :
                        res_type == SPVC_RESOURCE_TYPE_PUSH_CONSTANT ? "push" : "buffer",
                        res->name ? res->name : "(null)",
                        (unsigned)res->_id,
                        (unsigned)res->gl_binding,
                        (unsigned)current_slot,
                        (unsigned)free_slot);
            }
        }
    }

    return GL_TRUE;
}

/* Pass 4: Validate that no two user buffers share a Metal slot after
 * remapping.  Diagnostic only — no IR mutation. */
static GLboolean ir_validate_binding_uniqueness(MGLIRPatchContext *ctx)
{
    Program *pptr = ctx->program;
    int stage = ctx->stage;

    /* Fix #5: keep this list in sync with ir_pre_map_buffer_bindings —
     * UNIFORM_CONSTANT is included so non-image/sampler entries are validated;
     * image/sampler-backed entries are filtered by irResourceIsBufferBacked. */
    const int buffer_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT
    };

    /* Track which resource name occupies each slot. */
    const char *slot_owner[kMGLMaxMetalVertexBufferCount] = {0};
    GLboolean ok = GL_TRUE;

    for (size_t t = 0; t < sizeof(buffer_resource_types) / sizeof(buffer_resource_types[0]); t++) {
        int res_type = buffer_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (!irResourceIsBufferBacked(res, res_type)) {
                continue;
            }
            GLuint slot = res->binding;
            if (slot >= kMGLMaxMetalVertexBufferCount) {
                continue;
            }
            if (slot_owner[slot]) {
                fprintf(stderr,
                        "MGL IR VALIDATE: program=%u stage=%d DUPLICATE Metal slot %u "
                        "shared by '%s' and '%s'\n",
                        pptr->name, stage, slot,
                        slot_owner[slot], res->name ? res->name : "(null)");
                ok = GL_FALSE;
            } else {
                slot_owner[slot] = res->name ? res->name : "(null)";
            }
        }
    }

    /* Also verify no user buffer landed on a reserved slot. */
    for (GLuint s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
        if (slot_owner[s] && (ctx->reserved_slot_mask & (1u << s))) {
            fprintf(stderr,
                    "MGL IR VALIDATE: program=%u stage=%d user buffer '%s' "
                    "still on reserved slot %u after remap\n",
                    pptr->name, stage, slot_owner[s], s);
            ok = GL_FALSE;
        }
    }

    return ok;
}

/* Pass 5: Fix std140 ArrayStride for SSBO members affected by a glslang bug.
 *
 * In std140 SSBO blocks, glslang emits ArrayStride using the vector's natural
 * size (std430 rule) instead of the std140 rounded size (16) for:
 *   - int/uint vec2 direct arrays (float vec2 is correct)
 *   - matrices whose stored vector has 2 components
 *
 * This causes SPIRV-Cross to generate wrong .length() calculations and
 * incorrect data access patterns.  We detect std140 blocks by looking for
 * "definite std140" indicator members, then fix bug-affected members.
 *
 * ArrayStride is a type-level decoration (OpDecorate on the array type), so
 * we use spvc_compiler_set_decoration on the member's type ID.
 *
 * Heuristic detection limitations (this is NOT definitive):
 *   The std140/std430 classification is inferred from stride values rather
 *   than read from a layout qualifier, because SPIR-V does not record which
 *   std140 vs std430 rule a storage block was declared with.  The detector
 *   works by scanning SSBO members for a "definite std140" indicator: a
 *   member whose actual stride equals the std140 value AND differs from the
 *   std430 value.  Bug-affected types are excluded as indicators because
 *   their buggy std140 stride coincidentally equals std430, making them
 *   ambiguous.
 *
 *   Ambiguous cases that MAY be misclassified:
 *     - Blocks where every array member is a bug-affected type provide no
 *       definite indicator at all, so a std140 block can be missed entirely.
 *     - Blocks whose only non-bug-affected members happen to use strides
 *       that match both layouts (e.g. vec4 arrays, where std140 == std430)
 *       cannot disambiguate the layout.
 *     - A std430 block that contains no definite std430 indicator member
 *       may be treated as std140 and have its bug-affected members "fixed"
 *       unnecessarily.
 *
 *   Misclassification of non-bug-affected types is harmless because Pass 2
 *   only rewrites strides on glslang-bug-affected members (int/uint vec2
 *   arrays and 2-component matrices); correctly-laid-out members are left
 *   untouched.  The only risk is a bug-affected member in a misclassified
 *   std430 block receiving an unwanted std140 stride, which is the same
 *   trade-off chosen to repair the far more common genuine std140 case.
 *
 * Destructive IR edit — calls spvc_compiler_set_decoration. */
static GLboolean ir_fix_std140_array_strides(MGLIRPatchContext *ctx)
{
    spvc_compiler compiler = ctx->compiler;
    Program *pptr = ctx->program;

    /* Use the resources snapshot cached on the context by
     * mglRunIRPostprocessPipeline.  NULL means reflection failed at
     * pipeline start — treat as "no resources" and return GL_TRUE so a
     * reflection failure does not block compilation. */
    spvc_resources resources = ctx->resources;
    if (!resources) {
        return GL_TRUE; /* no resources — nothing to fix */
    }

    const spvc_reflected_resource *ssbo_list = NULL;
    size_t ssbo_count = 0;
    spvc_resources_get_resource_list_for_type(resources,
                                               SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
                                               &ssbo_list, &ssbo_count);
    if (ssbo_count == 0) {
        return GL_TRUE;
    }

    /* Pass 1: Check if the shader has any "definite std140" indicator. */
    GLboolean shader_has_std140 = GL_FALSE;
    for (size_t si = 0; si < ssbo_count && !shader_has_std140; si++)
    {
        spvc_type struct_type = spvc_compiler_get_type_handle(compiler,
            ssbo_list[si].type_id);
        if (!struct_type || spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT)
            struct_type = spvc_compiler_get_type_handle(compiler,
                ssbo_list[si].base_type_id);
        if (!struct_type || spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT)
            continue;

        unsigned num_members = spvc_type_get_num_member_types(struct_type);
        for (unsigned mi = 0; mi < num_members; mi++)
        {
            spvc_type member_type = spvc_compiler_get_type_handle(compiler,
                spvc_type_get_member_type(struct_type, mi));
            if (!member_type)
                continue;
            if (spvc_type_get_num_array_dimensions(member_type) == 0)
                continue;

            spvc_basetype bt = spvc_type_get_basetype(member_type);
            unsigned vecsize = spvc_type_get_vector_size(member_type);
            unsigned cols = spvc_type_get_columns(member_type);

            unsigned stride = 0;
            if (spvc_compiler_type_struct_member_array_stride(
                    compiler, struct_type, mi, &stride) != SPVC_SUCCESS)
                continue;

            spvc_bool row_major = spvc_compiler_has_member_decoration(
                compiler, ssbo_list[si].base_type_id, mi,
                SpvDecorationRowMajor);

            unsigned std140_stride = compute_std140_stride(bt, vecsize, cols, row_major);
            unsigned std430_stride = compute_std430_stride(bt, vecsize, cols, row_major);

            /* "Definite std140" = actual stride matches the correct std140
             * value AND differs from std430.  Bug-affected types are
             * never used as indicators here because their buggy std140
             * stride equals std430, so a matching stride is ambiguous. */
            if (std140_stride > 0 && std430_stride > 0 &&
                std140_stride != std430_stride &&
                stride == std140_stride)
            {
                shader_has_std140 = GL_TRUE;
                break;
            }
        }
    }

    if (!shader_has_std140) {
        return GL_TRUE;
    }

    /* Pass 2: Fix strides in std140 blocks. */
    int fixed_count = 0;
    for (size_t si = 0; si < ssbo_count; si++)
    {
        spvc_type_id struct_id = ssbo_list[si].type_id;
        spvc_type struct_type = spvc_compiler_get_type_handle(compiler, struct_id);
        if (!struct_type || spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT)
        {
            struct_id = ssbo_list[si].base_type_id;
            struct_type = spvc_compiler_get_type_handle(compiler, struct_id);
        }
        if (!struct_type || spvc_type_get_basetype(struct_type) != SPVC_BASETYPE_STRUCT)
            continue;

        /* Skip blocks that are definitely std430 (have a member whose
         * stride matches std430 and differs from std140).  This catches
         * the Output buffer and any explicitly std430 blocks. */
        GLboolean block_is_std430 = GL_FALSE;
        unsigned num_members = spvc_type_get_num_member_types(struct_type);
        for (unsigned mi = 0; mi < num_members; mi++)
        {
            spvc_type member_type = spvc_compiler_get_type_handle(compiler,
                spvc_type_get_member_type(struct_type, mi));
            if (!member_type)
                continue;
            if (spvc_type_get_num_array_dimensions(member_type) == 0)
                continue;

            spvc_basetype bt = spvc_type_get_basetype(member_type);
            unsigned vecsize = spvc_type_get_vector_size(member_type);
            unsigned cols = spvc_type_get_columns(member_type);

            unsigned stride = 0;
            if (spvc_compiler_type_struct_member_array_stride(
                    compiler, struct_type, mi, &stride) != SPVC_SUCCESS)
                continue;

            spvc_bool row_major = spvc_compiler_has_member_decoration(
                compiler, ssbo_list[si].base_type_id, mi,
                SpvDecorationRowMajor);

            /* Bug-affected types have buggy std140 stride == std430
             * stride, so they cannot distinguish the two layouts.
             * Skip them when looking for std430 indicators. */
            if (is_glslang_bug_affected(bt, vecsize, cols, row_major))
                continue;

            unsigned std140_stride = compute_std140_stride(bt, vecsize, cols, row_major);
            unsigned std430_stride = compute_std430_stride(bt, vecsize, cols, row_major);

            /* "Definite std430" = actual matches std430 AND differs from std140. */
            if (std140_stride > 0 && std430_stride > 0 &&
                std140_stride != std430_stride &&
                stride == std430_stride)
            {
                block_is_std430 = GL_TRUE;
                break;
            }
        }

        if (block_is_std430)
            continue;

        /* Fix any member whose stride doesn't match the correct std140 value. */
        for (unsigned mi = 0; mi < num_members; mi++)
        {
            spvc_type_id member_type_id = spvc_type_get_member_type(struct_type, mi);
            if (!member_type_id)
                continue;

            spvc_type member_type = spvc_compiler_get_type_handle(compiler, member_type_id);
            if (!member_type)
                continue;
            if (spvc_type_get_num_array_dimensions(member_type) == 0)
                continue;

            spvc_basetype bt = spvc_type_get_basetype(member_type);
            unsigned vecsize = spvc_type_get_vector_size(member_type);
            unsigned cols = spvc_type_get_columns(member_type);

            unsigned stride = 0;
            if (spvc_compiler_type_struct_member_array_stride(
                    compiler, struct_type, mi, &stride) != SPVC_SUCCESS)
                continue;

            spvc_bool row_major = spvc_compiler_has_member_decoration(
                compiler, ssbo_list[si].base_type_id, mi,
                SpvDecorationRowMajor);

            /* Only fix members affected by the glslang bug.  Other
             * members in std140 blocks already have correct strides,
             * and fixing them could harm undetected std430 blocks. */
            if (!is_glslang_bug_affected(bt, vecsize, cols, row_major))
                continue;

            /* Matrices with 2-component stored vectors are also
             * bug-affected, but correcting their stride causes
             * SPIRV-Cross to fail with "cannot represent in MSL"
             * because MSL cannot express std140's vec4-padding of
             * 2-component matrix vectors.  Skip them — these tests
             * remain failing, same as the baseline. */
            if (cols > 1)
                continue;

            unsigned std140_stride = compute_std140_stride(bt, vecsize, cols, row_major);
            if (std140_stride == 0)
                continue;

            if (stride != std140_stride)
            {
                spvc_compiler_set_decoration(compiler, member_type_id,
                                             SpvDecorationArrayStride,
                                             std140_stride);
                fixed_count++;

                if (mglIRRemapDebugEnabled()) {
                    fprintf(stderr,
                            "MGL IR STD140: program=%u stage=%d SSBO '%s' "
                            "member %u stride %u -> %u\n",
                            pptr->name, ctx->stage,
                            ssbo_list[si].name ? ssbo_list[si].name : "(null)",
                            mi, stride, std140_stride);
                }
            }
        }
    }

    if (fixed_count > 0 && mglIRRemapDebugEnabled()) {
        fprintf(stderr,
                "MGL IR STD140: program=%u stage=%d fixed %d member strides\n",
                pptr->name, ctx->stage, fixed_count);
    }

    return GL_TRUE;
}

/* === Pipeline runner === */

GLboolean mglRunIRPostprocessPipeline(GLMContext ctx, Program *pptr, int stage,
                                       spvc_compiler compiler)
{
    if (!pptr || stage < 0 || stage >= _MAX_SHADER_TYPES || !compiler) {
        return GL_TRUE; /* nothing to do — let compile proceed */
    }

    MGLIRPatchContext ictx;
    memset(&ictx, 0, sizeof(ictx));
    ictx.ctx = ctx;
    ictx.program = pptr;
    ictx.stage = stage;
    ictx.compiler = compiler;

    /* Create the shader-resources snapshot once and cache it on the
     * context for all passes.  ir_fix_std140_array_strides (pass 5) is the
     * only pass that needs it today; caching avoids a redundant
     * spvc_compiler_create_shader_resources call per pipeline run.  On
     * failure we leave ictx.resources NULL — passes treat NULL as "no
     * resources" and return GL_TRUE, so a reflection failure here does not
     * block compilation.  (Note: this snapshot is taken before any
     * destructive decoration edits; passes must query decoration values
     * live on the compiler rather than trusting the snapshot.) */
    spvc_resources cached_resources = NULL;
    if (spvc_compiler_create_shader_resources(compiler, &cached_resources) != SPVC_SUCCESS) {
        cached_resources = NULL;
    }
    ictx.resources = cached_resources;

    /* MGL_DISABLE_IR_REMAP gates ONLY the binding pre-mapping pass (pass 3).
     * The std140 ArrayStride fix (pass 5) runs unconditionally because it
     * repairs a glslang codegen bug that has nothing to do with MGL's
     * reserved-slot remapping — disabling it would silently break std140
     * SSBO access.  The reflection/reserve/validation passes (1, 2, 4) are
     * non-destructive diagnostics and always run. */
    const GLboolean remap_disabled = mglIRRemapDisabled();
    if (remap_disabled && mglIRRemapDebugEnabled()) {
        fprintf(stderr,
                "MGL IR DEBUG: program=%u stage=%d binding remap DISABLED via env "
                "(std140 fix still runs)\n",
                pptr->name, stage);
    }

    /* Pass ordering is documented in mgl_ir_postprocess.h.  Each pass is
     * independent and idempotent within a single pipeline run.  SPIRV-Cross
     * decoration edits (passes 3 and 5) are not trivially reversible, so
     * validation passes (1, 2, 4) run before/after destructive edits to
     * catch issues early. */
    MGLIRPatchFn passes[] = {
        ir_reflect_active_builtins,
        ir_reserve_internal_slots,
        ir_pre_map_buffer_bindings,
        ir_validate_binding_uniqueness,
        ir_fix_std140_array_strides
    };
    const char *pass_names[] = {
        "ir_reflect_active_builtins",
        "ir_reserve_internal_slots",
        "ir_pre_map_buffer_bindings",
        "ir_validate_binding_uniqueness",
        "ir_fix_std140_array_strides"
    };
    /* Pass 2 (index 2) is ir_pre_map_buffer_bindings — the only pass gated
     * by MGL_DISABLE_IR_REMAP. */
    const size_t remap_pass_index = 2;
    const size_t pass_count = sizeof(passes) / sizeof(passes[0]);

    for (size_t p = 0; p < pass_count; p++) {
        if (p == remap_pass_index && remap_disabled) {
            continue;
        }
        GLboolean ok = passes[p](&ictx);
        if (!ok) {
            fprintf(stderr,
                    "MGL IR PIPELINE: program=%u stage=%d pass '%s' failed\n",
                    pptr->name, stage, pass_names[p]);
            /* A validation failure is diagnostic — do not block compilation.
             * The string-level fallback can still repair issues. */
            break;
        }
    }

    if (mglIRRemapDebugEnabled()) {
        fprintf(stderr,
                "MGL IR PIPELINE: program=%u stage=%d complete, %d bindings remapped\n",
                pptr->name, stage, ictx.remapped_count);
    }

    return GL_TRUE;
}
