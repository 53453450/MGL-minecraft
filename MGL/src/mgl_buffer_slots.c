/*
 * mgl_buffer_slots.c
 * MGL
 *
 * Implementation of the reserved Metal buffer slot registry.
 * See mgl_buffer_slots.h for the full slot-ownership documentation and
 * cross-stage reuse notes.
 */

#include "mgl_buffer_slots.h"
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

    /* TCS stage_in replacement, TCS compute kernel only. */
    if (slot == kMGLBufferSlot_TCSStageInRepl) {
        return (stage == MGL_STAGE_TESS_CONTROL || stage < 0) ? GL_TRUE : GL_FALSE;
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

const char *mglBufferSlotReservedName(GLuint slot)
{
    switch (slot) {
        case 14:
            return "kMGLLodBiasMaxBufferIndex (FS LOD_BIAS clamp max)";
        case 15:
            return "kMGLPointSizeBufferIndex (VS point size) / kMGLLodBiasBufferIndex (FS LOD_BIAS)";
        case 24:
            return "kMGLBufferSlot_TCSStageInRepl (TCS [[stage_in]] replacement)";
        case 25:
            return "MGL_BUFFER_SIZE_BUFFER_INDEX (SPIRV-Cross runtime-sized SSBO sizing)";
        case 26:
            return "kMGLBufferSlot_TessFactor (TCS/TES compute path)";
        case 27:
            return "kMGLBufferSlot_PatchOutput (TCS/TES compute path)";
        case 28:
            return "kMGLBufferSlot_PatchInfo / kMGLCullDistanceParamsBufferIndex (TCS/TES compute OR VS cull-distance)";
        case 29:
            return "kMGLBufferSlot_IndirectParams / kMGLCullDistanceVertexBufferIndex (TCS/TES compute OR VS cull-distance)";
        case 30:
            return "kMGLBufferSlot_TESGlIn / kMGLFragCoordParamsBufferIndex (TES gl_in OR FS gl_FragCoord fixup)";
        default:
            return NULL;
    }
}
