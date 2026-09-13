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
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MGLRenderer_State.h
 *
 * Private state containers for MGLRenderer.  These structs are intentionally
 * behavior-free: categories own behavior, while this header names which
 * subsystem owns each mutable state group.
 */

#ifndef MGLRenderer_State_h
#define MGLRenderer_State_h

#import <Foundation/Foundation.h>

#include "glm_context.h"
#import "mgl_capability.h"
#import "mgl_trace_strategy.h"

#include "mgl_buffer_slots.h"   /* kMGLMaxBufferSlots */
#include "mgl_binding_stage.h"  /* MGLStageBindingCopyBackList */
#include "mgl_batching_state.h" /* MGLBatchingState */
#include "mgl_renderer_core_state.h" /* MGLRendererCoreState */

/* The drawable index enum moved to the C-safe mgl_renderer_core_state.h
 * (the core state sizes its drawBuffers array with _MAX_DRAW_BUFFERS). */

/* MGLStageBindingCopyBack / MGLStageBindingCopyBackList moved to the C-safe
 * mgl_binding_stage.h (the C compute binder fills the list). */

/* MGLDrawable and MGLRendererCoreState moved to the C-safe
 * mgl_renderer_core_state.h (with the dual-proxy helpers). */

typedef struct MGLGPURecoveryState_t {
    void *commandRecoveryOwner;
    GLuint interfaceMismatchBlockedProgram;
    CFTimeInterval interfaceMismatchBlockedUntil;
    uint32_t interfaceMismatchBlockedStreak;
    CFTimeInterval pipelineRetryAfter;
    CFTimeInterval interfaceMismatchRetryAfter;
    GLuint interfaceMismatchProgramName;
    uint32_t interfaceMismatchColor0Format;
    uint32_t interfaceMismatchDepthFormat;
    uint32_t interfaceMismatchStencilFormat;
    uint32_t interfaceMismatchStreak;
    GLuint programMismatchProgramName;
    CFTimeInterval programMismatchRetryAfter;
    uint32_t programMismatchStreak;
} MGLGPURecoveryState;

typedef struct MGLResourceFallbackState_t {
    MGLFragmentTextureTraceBinding fragmentTextureTraceBindings[TEXTURE_UNITS];
} MGLResourceFallbackState;

typedef struct MGLTessellationState_t {
    NSUInteger tessVertexCaptureOffset;
    BOOL tessVertexCaptureActive;
    BOOL cullDistanceCaptureActive;
    uint32_t cullDistanceCaptureFirstInstance;
    uint32_t cullDistanceCaptureInstanceStride;
    NSUInteger tcsOutputOffset;
    NSUInteger tcsOutputStride;
    GLuint tcsOutVertices;
    BOOL nativeTESActive;
    Program *nativeTESProgram;
    MGLStageBindingCopyBackList nativeTESCopyBacks;
    BOOL tessIndexedDraw;
    /* 256-aligned per-instance record span of the VS capture, used as the
     * per-instance draw offset when instanced native TES loops instances. */
    NSUInteger tessInstanceRecords;
    /* Isolines / point-mode TES: vertices expanded by the AIR TES compute
     * kernel (per-patch dispatch, contract at slot 29) and consumed by a
     * passthrough vertex stage drawing lines / points. */
    BOOL tessComputeActive;
    Program *tessComputeProgram;
    /* True only for the draw currently taking the TES render-vertex path.
     * A program can carry both the render-vertex function and the compute
     * kernel (indexed draws fall back to compute), so the TES-vertex binding
     * plan must follow the per-draw choice, not tess_eval_render_vertex. */
    BOOL tessVertexRenderActive;
    /* When a GS follows isolines/point-mode TES compute, the expanded
     * records are handed to mglDrawHostHandleGeometry / mglDrawGsRunDraw instead of the
     * TES passthrough vertex.  pendingGSInput is a retained MTLBuffer. */
    BOOL pendingGSInputActive;
    void *pendingGSInput;
    NSUInteger pendingGSInputOffset;
    NSUInteger pendingGSInputStride;
    GLsizei pendingGSVertexCount;
} MGLTessellationState;

typedef struct MGLGeometryState_t {
    BOOL expansionActive;
    Program *program;
} MGLGeometryState;

/* MGLBatchingState moved to the C-safe mgl_batching_state.h so the C batch
 * drivers can read and write its fields directly (one port hands out the
 * address) instead of going through a port per flag. */

#endif /* MGLRenderer_State_h */
