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
#include "mgl_gpu_recovery_state.h" /* MGLGPURecoveryState */

#include "glm_context.h"
#import "mgl_capability.h"
#import "mgl_trace_strategy.h"

#include "mgl_buffer_slots.h"   /* kMGLMaxBufferSlots */
#include "mgl_binding_stage.h"  /* MGLStageBindingCopyBackList */
#include "mgl_batching_state.h" /* MGLBatchingState */
#include "mgl_renderer_core_state.h" /* MGLRendererCoreState */
#include "mgl_tessellation_state.h" /* MGLTessellationState/MGLGeometryState */

/* The drawable index enum moved to the C-safe mgl_renderer_core_state.h
 * (the core state sizes its drawBuffers array with _MAX_DRAW_BUFFERS). */

/* MGLStageBindingCopyBack / MGLStageBindingCopyBackList moved to the C-safe
 * mgl_binding_stage.h (the C compute binder fills the list). */

/* MGLDrawable and MGLRendererCoreState moved to the C-safe
 * mgl_renderer_core_state.h (with the dual-proxy helpers). */

/* MGLGPURecoveryState moved to the C-safe mgl_gpu_recovery_state.h (the PSO
 * build path writes it from C). */

typedef struct MGLResourceFallbackState_t {
    MGLFragmentTextureTraceBinding fragmentTextureTraceBindings[TEXTURE_UNITS];
} MGLResourceFallbackState;

/* MGLTessellationState / MGLGeometryState moved to the C-safe
 * mgl_tessellation_state.h (the C draw host port mutates them). */


/* MGLBatchingState moved to the C-safe mgl_batching_state.h so the C batch
 * drivers can read and write its fields directly (one port hands out the
 * address) instead of going through a port per flag. */

#endif /* MGLRenderer_State_h */
