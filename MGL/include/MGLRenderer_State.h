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

#ifndef MGL_LOCK_TIMING_STACK_CAPACITY
#define MGL_LOCK_TIMING_STACK_CAPACITY 64
#endif

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
    NSRecursiveLock *__strong metalStateLock;
    double metalLockHoldStartStack[MGL_LOCK_TIMING_STACK_CAPACITY];
    NSUInteger metalLockHoldDepth;
    os_unfair_lock syncListLock;
    NSMutableArray *__strong proactiveTextures;
    MGLDrawable drawBuffers[_MAX_DRAW_BUFFERS];
    BOOL defaultDrawableWrittenSinceLastSwap;
    id<MTLCommandQueue> __strong commandQueue;
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
    BOOL mslCacheEnabled;
    NSCache<NSString *, NSNumber *> *__strong mslTextureTypeCache;
} MGLResourceFallbackState;

typedef struct MGLBlitState_t {
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *__strong scaledBlitPipelineCache;
    id<MTLSamplerState> __strong scaledBlitNearestSampler;
    id<MTLSamplerState> __strong scaledBlitLinearSampler;
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *__strong scaledDepthBlitPipelineCache;
    NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *__strong msaaIntegerResolvePipelineCache;
    NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *__strong clearRectPipelineCache;
    id<MTLDepthStencilState> __strong clearRectDepthState;
} MGLBlitState;

typedef struct MGLTessellationState_t {
    id<MTLBuffer> __strong tessFactorBuffer;
    id<MTLBuffer> __strong tcsOutputBuffer;
    id<MTLBuffer> __strong tcsPatchOutBuffer;
    NSUInteger tcsOutputStride;
    GLuint tcsOutVertices;
} MGLTessellationState;

typedef struct MGLBatchingState_t {
    MGLBatchArena batchArena;
    BOOL arenaSnapshotEnabled;
    BOOL skipSameKeyRestoreEnabled;
    BOOL dirtyKeyDeltaEnabled;
} MGLBatchingState;

#endif /* MGLRenderer_State_h */
