/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tessellation_state.h — the renderer's tessellation and geometry records,
 * moved out of the Objective-C MGLRenderer_State.h (P0-1, log 113) so C code can
 * read and write their fields directly.  The state areas hand out the address of
 * each record (`areas.tessellation` / `areas.geometry`), which is the "one state
 * struct, one port" pattern the core, batching, command and pipeline-cache
 * records already follow.  BOOL became bool and NSUInteger became size_t; both
 * are the same types on the 64-bit targets MGL builds for.
 */

#ifndef MGL_TESSELLATION_STATE_H
#define MGL_TESSELLATION_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"
#include "mgl_binding_stage.h"   /* MGLStageBindingCopyBackList */
#include "mgl_types_program.h"   /* Program */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLTessellationState_t {
    size_t tessVertexCaptureOffset;
    bool tessVertexCaptureActive;
    bool cullDistanceCaptureActive;
    uint32_t cullDistanceCaptureFirstInstance;
    uint32_t cullDistanceCaptureInstanceStride;
    size_t tcsOutputOffset;
    size_t tcsOutputStride;
    GLuint tcsOutVertices;
    bool nativeTESActive;
    Program *nativeTESProgram;
    MGLStageBindingCopyBackList nativeTESCopyBacks;
    bool tessIndexedDraw;
    /* 256-aligned per-instance record span of the VS capture, used as the
     * per-instance draw offset when instanced native TES loops instances. */
    size_t tessInstanceRecords;
    /* Isolines / point-mode TES: vertices expanded by the AIR TES compute
     * kernel (per-patch dispatch, contract at slot 29) and consumed by a
     * passthrough vertex stage drawing lines / points. */
    bool tessComputeActive;
    Program *tessComputeProgram;
    /* True only for the draw currently taking the TES render-vertex path.
     * A program can carry both the render-vertex function and the compute
     * kernel (indexed draws fall back to compute), so the TES-vertex binding
     * plan must follow the per-draw choice, not tess_eval_render_vertex. */
    bool tessVertexRenderActive;
    /* When a GS follows isolines/point-mode TES compute, the expanded
     * records are handed to mglDrawHostHandleGeometry / mglDrawGsRunDraw instead of the
     * TES passthrough vertex.  pendingGSInput is a retained MTLBuffer. */
    bool pendingGSInputActive;
    void *pendingGSInput;
    size_t pendingGSInputOffset;
    size_t pendingGSInputStride;
    GLsizei pendingGSVertexCount;
} MGLTessellationState;

typedef struct MGLGeometryState_t {
    bool expansionActive;
    Program *program;
} MGLGeometryState;

#ifdef __cplusplus
}
#endif

#endif /* MGL_TESSELLATION_STATE_H */
