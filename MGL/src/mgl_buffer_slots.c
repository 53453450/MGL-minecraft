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
 * mgl_buffer_slots.c
 * MGL
 *
 * Implementation of the reserved Metal buffer slot registry.
 * See mgl_buffer_slots.h for the full slot-ownership documentation and
 * cross-stage reuse notes.
 */

#include "mgl_buffer_slots.h"
#include "glm_context.h"  /* brings in Program after GLMContext typedef */
#include <stddef.h>  /* NULL */

/* Shader stage indices — must match the enum order in glm_context.h
 * (_MAX_SHADER_TYPES: VERTEX=0, TESS_CONTROL=1, TESS_EVALUATION=2,
 * GEOMETRY=3, FRAGMENT=4, COMPUTE=5).  Reproduced here to avoid pulling
 * the full GL state header into this leaf utility. */
#define MGL_STAGE_VERTEX              0
#define MGL_STAGE_TESS_CONTROL        1
#define MGL_STAGE_TESS_EVALUATION     2
#define MGL_STAGE_GEOMETRY            3
#define MGL_STAGE_FRAGMENT            4
#define MGL_STAGE_COMPUTE             5

GLboolean mglBufferSlotIsReservedForStage(GLuint slot, int stage)
{
    /* Fixed-function point size emulation parameter, vertex path only. */
    if (slot == kMGLPointSizeBufferIndex) {
        return (stage == MGL_STAGE_VERTEX || stage < 0) ? GL_TRUE : GL_FALSE;
    }

    /* TCS stage_in replacement, TCS compute kernel only.  Note: slot 24 is
     * also the GS compute-expansion input slot (MGL_AIR_GS_SLOT_INPUT), so
     * the TCS-only early return must NOT shadow the GEOMETRY case below —
     * only TESS_CONTROL (or the generic stage < 0) claims it here. */
    if (slot == kMGLBufferSlot_TCSStageInRepl &&
        (stage == MGL_STAGE_TESS_CONTROL || stage < 0)) {
        return GL_TRUE;
    }

    /* GS compute-expansion path (mgl_air_gs_abi.h §1): the GS kernel owns
     * slot 24 (VS capture input), 25 (index-gather params), 28 (expanded
     * output), 29 (counts/indirect), 30 (index gather), 31 (XFB stream) and
     * 27 (XFB meta/atomic cursor).  Slots 27-30 also overlap the
     * tessellation / VS/FS emulation paths but the encoders are disjoint;
     * a UBO/SSBO bound for the geometry stage at any of these would corrupt
     * the expansion or the transform-feedback output. */
    if (stage == MGL_STAGE_GEOMETRY || stage < 0) {
        if (slot == 24u || slot == 25u || slot == 26u || slot == 27u ||
            slot == 28u || slot == 29u || slot == 30u || slot == 31u) {
            return GL_TRUE;
        }
    }

    /* No other slot is *always* reserved regardless of stage.  Slots 26-30
     * are reserved only within specific paths (TCS/TES compute vs VS/FS
     * draw), and whether a user buffer binding collides depends on whether
     * that path is active for the program being linked.  The conservative
     * `mglBufferSlotIsReserved` below covers the "any stage" case. */
    (void)stage;
    return GL_FALSE;
}

GLboolean mglBufferSlotIsReserved(GLuint slot)
{
    /* Slots 26-30 are reserved *when* the program uses tessellation or the
     * VS/FS emulation paths, but a program that uses neither could legally
     * bind user buffers there.  We do NOT mark them universally reserved
     * to avoid false positives on simple programs.
     *
     * Callers that know whether tessellation/cull-distance/FragCoord-fixup
     * is active should use the specific mglBufferSlotIsReservedFor* helpers
     * below for accurate per-path detection.  The stage-specific point-size
     * and TCS stage-in slots are handled by mglBufferSlotIsReservedForStage. */
    return GL_FALSE;
}

GLboolean mglBufferSlotIsReservedForTessellation(GLuint slot)
{
    /* TCS/TES compute dispatch path reserves slots 26-30 for tessellation
     * factors, per-patch output, patch info, indirect params, and TES gl_in.
     * A UBO/SSBO bound at any of these slots in a tessellation program would
     * silently corrupt tessellation data.  The TCS stage-in replacement slot
     * is stage-specific and covered by mglBufferSlotIsReservedForStage. */
    switch (slot) {
        case 26u:  /* TessFactor */
        case 27u:  /* PatchOutput */
        case 28u:  /* PatchInfo */
        case 29u:  /* IndirectParams */
        case 30u:  /* TESGlIn */
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

GLboolean mglBufferSlotIsReservedForGeometry(GLuint slot)
{
    /* GS compute-expansion path (mgl_air_gs_abi.h §1): the GS kernel
     * reserves slot 24 (VS capture input), 25 (index-gather params),
     * 26 (reserved with tessellation factors), 28 (expanded output),
     * 29 (counts / indirect args), 30 (index gather), 31 (GS XFB stream) and
     * 27 (GS XFB meta / atomic cursor).  A UBO/SSBO bound at any of these
     * slots in a geometry program would silently corrupt the expansion, the
     * gather, or the transform-feedback output.  The stage-specific check is
     * also handled by mglBufferSlotIsReservedForStage. */
    switch (slot) {
        case 24u:  /* GS input records */
        case 25u:  /* GS gather params */
        case 26u:  /* TCS/TES factors (shared domain) */
        case 27u:  /* GS XFB meta / atomic cursor */
        case 28u:  /* GS output records */
        case 29u:  /* GS counts / indirect args */
        case 30u:  /* GS index gather */
        case 31u:  /* GS XFB stream output */
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

GLboolean mglBufferSlotIsReservedForCullDistance(GLuint slot)
{
    /* VS cull-distance emulation path reserves slot 28 (params constant) and
     * 29 (sibling-vertex data).  These overlap with tessellation slots but
     * are in a disjoint path (VS draw vs TCS/TES compute). */
    switch (slot) {
        case 28u:  /* CullDistanceParams */
        case 29u:  /* CullDistanceVertex */
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

GLboolean mglBufferSlotIsReservedForFragCoordFixup(GLuint slot)
{
    /* FS gl_FragCoord fixup path reserves slot 30 for the FragCoord params
     * constant.  Overlaps with TES gl_in (slot 30) but disjoint stages. */
    return (slot == 30u) ? GL_TRUE : GL_FALSE;
}

GLuint mglRuntimeArraySizeBufferIndexForProgram(const Program *program,
                                                int stage)
{
    if (program &&
        (stage == _GEOMETRY_SHADER ||
         (stage == _TESS_EVALUATION_SHADER &&
          (program->tess_gen_mode == GL_ISOLINES ||
           program->tess_gen_point_mode)))) {
        return MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX;
    }
    return MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX;
}

GLboolean mglBufferSlotConflictsForProgram(const Program *program,
                                           GLuint slot,
                                           int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return GL_FALSE;
    }

    /* The AIR backend exposes the runtime-array byte-size table at a fixed
     * Metal slot.  A reflected user buffer at that slot would be overwritten
     * when the stage binds the hidden table, so reject the collision at link
     * time rather than allowing a draw/dispatch-time data corruption. */
    if (slot == mglRuntimeArraySizeBufferIndexForProgram(program, stage) &&
        program->modules[stage].needs_runtime_array_size_buffer) {
        return GL_TRUE;
    }

    switch (stage) {
        case _VERTEX_SHADER:
            if (program->uses_point_size_params &&
                slot == kMGLPointSizeBufferIndex) {
                return GL_TRUE;
            }
            if (program->uses_cull_distance &&
                mglBufferSlotIsReservedForCullDistance(slot)) {
                return GL_TRUE;
            }
            break;

        case _TESS_CONTROL_SHADER:
            /* The TCS AIR compute kernel always owns stage_in(24), factors
             * (26), patch output(27), stage output(28), and indirect(29).
             * Slot 30 belongs to TES, so do not reject a TCS-only user
             * resource there. */
            if (slot == 24u || (slot >= 26u && slot <= 29u)) {
                return GL_TRUE;
            }
            break;

        case _TESS_EVALUATION_SHADER:
            if (program->tess_gen_mode == GL_ISOLINES ||
                program->tess_gen_point_mode) {
                /* Isolines and point-mode TES execute as a compute kernel
                 * whose fixed ABI occupies every slot in [24, 31]. */
                if (slot >= 24u && slot <= 31u) {
                    return GL_TRUE;
                }
            } else if (slot == 27u || slot == 28u || slot == 30u) {
                /* Native triangle/quad TES: patch input, patch info, gl_in. */
                return GL_TRUE;
            }
            break;

        case _GEOMETRY_SHADER:
            if (program->gs_route == MGL_GS_ROUTE_COMPUTE &&
                mglBufferSlotIsReservedForGeometry(slot)) {
                return GL_TRUE;
            }
            break;

        case _FRAGMENT_SHADER:
            if (program->usesFragCoordParams &&
                mglBufferSlotIsReservedForFragCoordFixup(slot)) {
                return GL_TRUE;
            }
            break;

        default:
            break;
    }

    return GL_FALSE;
}

const char *mglBufferSlotReservedName(GLuint slot)
{
    switch (slot) {
        case 23:
            return "MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX (GS/compute-TES runtime-sized SSBO sizing)";
        case 14:
            return "kMGLLodBiasMaxBufferIndex (FS LOD_BIAS clamp max)";
        case 15:
            return "kMGLPointSizeBufferIndex (VS point size) / kMGLLodBiasBufferIndex (FS LOD_BIAS)";
        case 24:
            return "kMGLBufferSlot_TCSStageInRepl (TCS [[stage_in]] replacement) / MGL_AIR_GS_SLOT_INPUT (GS compute expansion)";
        case 25:
            return "MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX (ordinary runtime-sized SSBO sizing) / MGL_AIR_GS_SLOT_GATHER_PARAMS / MGL_AIR_TESS_SLOT_GATHER_PARAMS";
        case 26:
            return "kMGLBufferSlot_TessFactor (TCS/TES compute path)";
        case 27:
            return "kMGLBufferSlot_PatchOutput (TCS/TES compute path) / MGL_AIR_GS_SLOT_XFB_META (GS XFB atomic cursor / written counter)";
        case 28:
            return "kMGLBufferSlot_PatchInfo / kMGLCullDistanceParamsBufferIndex / MGL_AIR_GS_SLOT_OUTPUT (TCS/TES compute OR VS cull-distance OR GS expansion)";
        case 29:
            return "kMGLBufferSlot_IndirectParams / kMGLCullDistanceVertexBufferIndex / MGL_AIR_GS_SLOT_COUNTS (TCS/TES compute OR VS cull-distance OR GS expansion)";
        case 30:
            return "kMGLBufferSlot_TESGlIn / kMGLFragCoordParamsBufferIndex / MGL_AIR_GS_SLOT_GATHER (TES gl_in OR FS gl_FragCoord fixup OR GS indexed gather)";
        case 31:
            return "MGL_AIR_TESS_SLOT_XFB_OUT / MGL_AIR_GS_SLOT_XFB (TES/GS transform-feedback stream, disjoint encoders)";
        default:
            return NULL;
    }
}
