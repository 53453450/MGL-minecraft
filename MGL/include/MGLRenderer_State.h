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

#ifndef kMGLMaxBufferSlots
#define kMGLMaxBufferSlots 31
#endif

typedef struct MGLDrawable_t {
    GLuint width;
    GLuint height;
} MGLDrawable;

enum {
    _FRONT,
    _BACK,
    _FRONT_LEFT,
    _FRONT_RIGHT,
    _BACK_LEFT,
    _BACK_RIGHT,
    _MAX_DRAW_BUFFERS
};

typedef struct {
    void *temporary;
    void *destination;
    Buffer *destination_buffer;
    NSUInteger destination_offset;
    NSUInteger length;
} MGLStageBindingCopyBack;

typedef struct {
    MGLStageBindingCopyBack slots[kMGLMaxBufferSlots];
} MGLStageBindingCopyBackList;

typedef struct MGLRendererCoreState_t {
    MGLCapability capability;
    MGLDrawable drawBuffers[_MAX_DRAW_BUFFERS];
    BOOL defaultDrawableWrittenSinceLastSwap;
    /* Lock-free hand-off channels.  Written by the completion-handler thread
     * / main queue, drained (and resynchronized) on the GL thread. */
    _Atomic bool deviceResetRequested;
    _Atomic uint32_t pendingDrawableW;
    _Atomic uint32_t pendingDrawableH;
    _Atomic bool drawableSizeDirty;
} MGLRendererCoreState;

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
} MGLTessellationState;

typedef struct MGLGeometryState_t {
    BOOL expansionActive;
    Program *program;
} MGLGeometryState;

typedef struct MGLBatchingState_t {
    MGLBatchArena batchArena;
    BOOL arenaSnapshotEnabled;
    BOOL skipSameKeyRestoreEnabled;
    BOOL dirtyKeyDeltaEnabled;
    /* Replay of a BindNoFlush batch that captured per-draw BindVertexBuffer
     * overrides.  Descriptor bakes only relativeoffset; setVertexBuffer uses
     * the absolute VERTEX_BINDING_OFFSET so overrides are not double-counted. */
    BOOL absoluteVertexBindingOffsets;
} MGLBatchingState;

#endif /* MGLRenderer_State_h */
