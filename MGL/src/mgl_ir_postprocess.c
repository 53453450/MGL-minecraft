/*
 * mgl_ir_postprocess.c
 * MGL
 *
 * Implementation of the IR-level postprocessing subsystem.
 * See mgl_ir_postprocess.h for the architectural rationale.
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
 * Thread-safety: these caches use non-atomic int reads/writes.  This is
 * consistent with the rest of the MGL codebase, which assumes a single-
 * threaded GL context (the GL context itself is not thread-safe, and
 * shader compilation happens on the context thread).  If MGL ever supports
 * multi-threaded shader compilation, these need atomic guards (e.g.
 * stdatomic / OSAtomic compare-and-swap on the cached value) or a
 * dispatch_once for the one-shot env-var init, so the lazy `if (x < 0)`
 * read-modify-write cannot race between threads. */

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
    /* Metal exposes buffer slots 0..30 (kMGLMaxMetalVertexBufferCount == 31
     * is the COUNT of valid slots, not the max index).  Slot 31 and above
     * are therefore out of range, so the threshold is `>= 31` (not `>= 32`).
     * Treat any such slot as conflicting so it is never assigned to a user
     * buffer. */
    if (slot >= kMGLMaxMetalVertexBufferCount) {
        return GL_TRUE;
    }

    /* Cache the reflection-based flags: mglVSUsesCullDistance /
     * mglFSUsesFragCoord re-scan SPIR-V reflection + GLSL source, and this
     * runs per slot check (31 slots × 6 stages per link).  The cache is
     * invalidated at link start. */
    if (!pptr->ir_cache_valid) {
        pptr->ir_uses_cull_distance = mglVSUsesCullDistance(pptr);
        pptr->ir_uses_frag_coord = mglFSUsesFragCoord(pptr);
        pptr->ir_cache_valid = GL_TRUE;
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
        pptr->ir_uses_cull_distance) {
        slot_conflicts = GL_TRUE;
    }

    /* FragCoord slot 30: reserved (cross-stage) when FS uses gl_FragCoord. */
    if (!slot_conflicts &&
        mglBufferSlotIsReservedForFragCoordFixup(slot) &&
        pptr->ir_uses_frag_coord) {
        slot_conflicts = GL_TRUE;
    }

    return slot_conflicts;
}

/* === IR passes === */

/* Pass 1: Reflect active builtins and cache path-detection flags.
 * No IR mutation — pure reflection/state read.
 *
 * The cached flags (has_tessellation, vs_uses_cull_distance,
 * fs_uses_frag_coord) are consumed by irBufferSlotConflictsForContext,
 * which reads the ctx-> fields directly instead of re-running the
 * path-detection helpers.  irBufferSlotConflictsForContext is in turn
 * called by two downstream passes:
 *   - ir_reserve_internal_slots (pass 1 of conflict checking): builds the
 *     reserved-slot bitmap by probing every slot.
 *   - ir_pre_map_buffer_bindings (pass 2 of conflict checking): probes
 *     individual user-buffer slots to decide which need remapping.
 * Caching the flags here means those passes avoid re-scanning SPIRV-Cross
 * reflection and GLSL source on every conflict check. */
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

/* Sampler-like detection helpers for UNIFORM_CONSTANT classification.
 *
 * These mirror the same-pattern helpers in program.c and are kept here for
 * any future pass that needs to distinguish sampler-backed from plain-data
 * UNIFORM_CONSTANT entries.  Currently UNIFORM_CONSTANT resources are NOT
 * included in the buffer-backed set (irResourceIsBufferBacked below) because
 * with MSL_ARGUMENT_BUFFERS=FALSE they never occupy [[buffer(N)]] slots, but
 * the helpers are available if that policy ever changes. */
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

__attribute__((unused)) static GLboolean irUniformConstantLooksSamplerLike(const SpirvResource *res)
{
    if (!res) {
        return GL_FALSE;
    }
    return (res->image_dim != 0u ||
            res->uniform_location >= (GLint)MGL_IR_SYNTHETIC_SAMPLER_LOCATION_BASE ||
            irUniformNameLooksSamplerLike(res->name) ||
            irGLTypeLooksSamplerLike(res->gl_type)) ? GL_TRUE : GL_FALSE;
}

/* Helper: classify a resource as buffer-backed (eligible for IR remap).
 * Plain UNIFORM_CONSTANT resources are intentionally excluded here.  With
 * MSL argument buffers disabled, SPIRV-Cross emits many non-sampler uniforms
 * as ordinary MSL parameters that do not occupy [[buffer(N)]] slots.  Treating
 * those as buffer-backed makes slot 0 look falsely occupied. */
static GLboolean irResourceIsBufferBacked(SpirvResource *res, int res_type)
{
    if (!res) return GL_FALSE;
    switch (res_type) {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
        case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
        case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
            return GL_TRUE;
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

    const int buffer_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
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
             * Vertex shaders must stop before kMGLVertexAttribBufferBase (16)
             * so vertex-stage UBOs are never assigned to vertex attribute
             * slots 16-24, which Metal uses for stage_in vertex buffer
             * bindings.  Slot 15 (kMGLPointSizeBufferIndex, point-size
             * emulation) is still excluded for VS — it is reserved via
             * mglBufferSlotIsReservedForStage, so used_slots[15] is set by
             * the reserved_slot_mask and skipped by the search below.
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

    const int buffer_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
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

    /* MGL_DISABLE_IR_REMAP gates ONLY the binding pre-mapping pass (pass 3).
     * The reflection/reserve/validation passes (1, 2, 4) are non-destructive
     * diagnostics and always run. */
    const GLboolean remap_disabled = mglIRRemapDisabled();
    if (remap_disabled && mglIRRemapDebugEnabled()) {
        fprintf(stderr,
                "MGL IR DEBUG: program=%u stage=%d binding remap DISABLED via env\n",
                pptr->name, stage);
    }

    /* Pass ordering is documented in mgl_ir_postprocess.h.  Each pass is
     * independent and idempotent within a single pipeline run.  SPIRV-Cross
     * decoration edits (pass 3) are not trivially reversible, so validation
     * passes (1, 2, 4) run before/after destructive edits to catch issues
     * early. */
    MGLIRPatchFn passes[] = {
        ir_reflect_active_builtins,
        ir_reserve_internal_slots,
        ir_pre_map_buffer_bindings,
        ir_validate_binding_uniqueness
    };
    const char *pass_names[] = {
        "ir_reflect_active_builtins",
        "ir_reserve_internal_slots",
        "ir_pre_map_buffer_bindings",
        "ir_validate_binding_uniqueness"
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
