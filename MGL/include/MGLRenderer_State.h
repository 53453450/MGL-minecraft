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
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#import <simd/simd.h>
#include <os/lock.h>

#include "glm_context.h"
#import "mgl_capability.h"
#import "mgl_trace_strategy.h"
#import "mgl_texture_compat.h"

#ifndef kMGLMaxBufferSlots
#define kMGLMaxBufferSlots 31
#endif

#ifndef kMGLSamplerSnapshotCacheCapacity
#define kMGLSamplerSnapshotCacheCapacity 256
#endif

#ifndef kMGLSamplerSnapshotCacheIndexCapacity
#define kMGLSamplerSnapshotCacheIndexCapacity 512
#endif

typedef struct MGLDrawable_t {
    GLuint width;
    GLuint height;
    id<MTLTexture> __strong drawbuffer;
    id<MTLTexture> __strong depthbuffer;
    id<MTLTexture> __strong stencilbuffer;
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
    id<MTLBuffer> __strong temporary;
    id<MTLBuffer> __strong destination;
    Buffer *destination_buffer;
    NSUInteger destination_offset;
    NSUInteger length;
} MGLStageBindingCopyBack;

typedef struct {
    MGLStageBindingCopyBack slots[kMGLMaxBufferSlots];
} MGLStageBindingCopyBackList;

typedef struct MGLRendererCoreState_t {
    NSView *__strong view;
    CAMetalLayer *__strong layer;
    id<CAMetalDrawable> __strong drawable;
    GLMState *activeState;
    id<MTLDevice> __strong device;
    MGLCapability capability;
    NSMutableArray *__strong proactiveTextures;
    MGLDrawable drawBuffers[_MAX_DRAW_BUFFERS];
    BOOL defaultDrawableWrittenSinceLastSwap;
    void *commandQueueOwner;
    id<MTLCommandQueue> __strong commandQueue;
    /* Lock-free hand-off channels.  Written by the completion-handler thread
     * / main queue, drained (and resynchronized) on the GL thread. */
    _Atomic bool deviceResetRequested;
    _Atomic uint32_t pendingDrawableW;
    _Atomic uint32_t pendingDrawableH;
    _Atomic bool drawableSizeDirty;
} MGLRendererCoreState;

typedef struct MGLGPURecoveryState_t {
    os_unfair_lock gpuErrorLock;
    NSUInteger consecutiveGPUErrors;
    NSUInteger consecutiveGPUSuccesses;
    NSTimeInterval lastGPUErrorTime;
    BOOL gpuErrorRecoveryMode;
    GLuint interfaceMismatchBlockedProgram;
    CFTimeInterval interfaceMismatchBlockedUntil;
    uint32_t interfaceMismatchBlockedStreak;
    CFTimeInterval pipelineRetryAfter;
    CFTimeInterval interfaceMismatchRetryAfter;
    GLuint interfaceMismatchProgramName;
    MTLPixelFormat interfaceMismatchColor0Format;
    MTLPixelFormat interfaceMismatchDepthFormat;
    MTLPixelFormat interfaceMismatchStencilFormat;
    uint32_t interfaceMismatchStreak;
    GLuint programMismatchProgramName;
    CFTimeInterval programMismatchRetryAfter;
    uint32_t programMismatchStreak;
} MGLGPURecoveryState;

typedef struct MGLResourceFallbackState_t {
    id<MTLTexture> __strong fallbackSampledTexture;
    id<MTLTexture> __strong fallbackCubeSampledTexture;
    id<MTLBuffer> __strong fallbackTextureBufferStorage;
    id<MTLTexture> __strong fallbackSintTextureBuffer;
    NSMutableDictionary<NSNumber *, id<MTLTexture>> *__strong fallbackSampledTextureCache;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *__strong doubleVertexAttribBufferCache;
    id<MTLSamplerState> __strong fallbackSamplerState;
    MGLSamplerSnapshotKey samplerSnapshotCacheKeys[kMGLSamplerSnapshotCacheCapacity];
    id<MTLSamplerState> __strong samplerSnapshotCacheStates[kMGLSamplerSnapshotCacheCapacity];
    uint16_t samplerSnapshotCacheIndex[kMGLSamplerSnapshotCacheIndexCapacity];
    uint16_t samplerSnapshotCacheCount;
    uint16_t samplerSnapshotCacheNext;
    MGLFragmentTextureTraceBinding fragmentTextureTraceBindings[TEXTURE_UNITS];
} MGLResourceFallbackState;

typedef struct MGLBlitState_t {
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *__strong scaledBlitPipelineCache;
    id<MTLSamplerState> __strong scaledBlitNearestSampler;
    id<MTLSamplerState> __strong scaledBlitLinearSampler;
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *__strong scaledDepthBlitPipelineCache;
    NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *__strong msaaIntegerResolvePipelineCache;
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *__strong clearRectPipelineCache;
    id<MTLDepthStencilState> __strong clearRectDepthState;
    NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *__strong scaledBlitComputePipelineCache;
} MGLBlitState;

typedef struct MGLTessellationState_t {
    id<MTLBuffer> __strong tessFactorBuffer;
    id<MTLBuffer> __strong nativeTessFactorBuffer;
    id<MTLBuffer> __strong tcsOutputBuffer;
    id<MTLBuffer> __strong tcsPatchOutBuffer;
    id<MTLBuffer> __strong tessVertexCaptureBuffer;
    NSUInteger tessVertexCaptureOffset;
    BOOL tessVertexCaptureActive;
    id<MTLBuffer> __strong cullDistanceCaptureBuffer;
    BOOL cullDistanceCaptureActive;
    uint32_t cullDistanceCaptureFirstInstance;
    uint32_t cullDistanceCaptureInstanceStride;
    NSUInteger tcsOutputOffset;
    NSUInteger tcsOutputStride;
    GLuint tcsOutVertices;
    BOOL nativeTESActive;
    Program *nativeTESProgram;
    /* Indexed native TES: sparse VS capture records [vertex_id] +
     * CPU gather buffer fed as Metal controlPointIndexBuffer. */
    id<MTLBuffer> __strong tessControlPointIndexBuffer;
    BOOL tessIndexedDraw;
    /* 256-aligned per-instance record span of the VS capture, used as the
     * per-instance draw offset when instanced native TES loops instances. */
    NSUInteger tessInstanceRecords;
    /* Isolines / point-mode TES: vertices expanded by the AIR TES compute
     * kernel (per-patch dispatch, contract at slot 29) and consumed by a
     * passthrough vertex stage drawing lines / points. */
    BOOL tessComputeActive;
    id<MTLBuffer> __strong tessComputeOutputBuffer;
    NSUInteger tessComputeOutputStride;
    GLuint tessComputeItems;
    MTLPrimitiveType tessComputePrimitiveType;
    Program *tessComputeProgram;
    id<MTLLibrary> __strong tessPassthroughLibrary;
    id<MTLFunction> __strong tessPassthroughFunction;
    uint64_t tessPassthroughProgramInstanceId;
} MGLTessellationState;

typedef struct MGLGeometryState_t {
    id<MTLLibrary> __strong passthroughLibrary;
    id<MTLFunction> __strong passthroughFunction;
    uint64_t passthroughProgramInstanceId;
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
