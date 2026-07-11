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
 * MGLRenderer_Private.h — shared class extension for MGLRenderer.  Holds ivar
 * declarations (ObjC categories can't declare ivars), shared types, shared
 * macros, and the aggregate import of per-category private headers.
 */

#ifndef MGLRenderer_Private_h
#define MGLRenderer_Private_h

#import "MGLRenderer.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>   // CAMetalLayer
#import <simd/simd.h>               // vector_float4, vector_uint2, etc.
#include <os/lock.h>

/* glm_context.h pulls in GLMContext, Texture, Buffer, Program, Framebuffer,
 * Sync, GLMState, MGLBatchPath, MGLDrawBatch, MAX_COLOR_ATTACHMENTS,
 * TEXTURE_UNITS, and GL types (GLenum, GLuint, GLsizei, ...). */
#include "glm_context.h"
#import "mgl_capability.h"          // ivar type: MGLCapability
#import "mgl_texture_compat.h"      // MGLTextureDataKind
#import "mgl_trace_strategy.h"      // ivar type: MGLFragmentTextureTraceBinding
#import "mgl_readback.h"
#import "mgl_metal_ref.h"
#import "mgl_sync.h"
#import "mgl_rt_sync.h"
#import "mgl_blit_clip.h"
#import "mgl_state_compat.h"
#import "mgl_msl_compat.h"
#import "mgl_safety.h"
#import "mgl_vertex_format.h"
#include "spirv_cross_c.h"
#define MGL_NO_MTL_PIXEL_FORMAT
#import "pixel_utils.h"
#undef MGL_NO_MTL_PIXEL_FORMAT
#import "mgl_frame_activity.h"      // mglPerfLockTimingEnabled, MGL_FRAME_ADD
#import "mgl_draw_buffer.h"
#import "mgl_buffer_slots.h"
#import "mgl_vertex_attrib_query.h"
#import "mgl_coordinate.h"
#import "mgl_spirv_resource.h"
#import "mgl_buffer_query.h"
#import "mgl_focus_program.h"
#import "mgl_draw_mode.h"
#import "mgl_index_buffer.h"
#import "mgl_draw_encode.h"

/* FBO attachment / texture lookup helpers — defined in framebuffers.c /
 * textures.c, extern-declared here because they're not in any public header. */
extern bool isColorAttachment(GLMContext ctx, GLuint attachment);
extern FBOAttachment *getFBOAttachment(GLMContext ctx, Framebuffer *fbo, GLenum attachment);
extern Texture *findTexture(GLMContext ctx, GLuint texture);

/* MTL4 compiler support (conditional) — needed by the _mtl4Compiler ivar. */
#if __has_include(<Metal/MTL4Compiler.h>) && __has_include(<Metal/MTL4LibraryDescriptor.h>)
#import <Metal/MTL4Compiler.h>
#import <Metal/MTL4LibraryDescriptor.h>
#define MGL_HAS_MTL4_COMPILER 1
#else
#define MGL_HAS_MTL4_COMPILER 0
#endif

/* === Types needed by ivar declarations below === */

typedef struct SyncList_t {
    GLuint count;
    GLuint  size;
    Sync **list;
} SyncList;

typedef struct MGLDrawable_t {
    GLuint width;
    GLuint height;
    id<MTLTexture> drawbuffer;
    id<MTLTexture> depthbuffer;
    id<MTLTexture> stencilbuffer;
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

/* Last-bound state cache for render encoder dedup. */
typedef struct {
    id<MTLBuffer> __strong buffer;
    NSUInteger offset;
} MGLLastBoundBuffer;

#define kMGLMaxBufferSlots 31

/* Referenced by the _metalLockHoldStartStack ivar and the METAL_LOCK macro. */
#define MGL_LOCK_TIMING_STACK_CAPACITY 64

/* Shared helper — declared here because inline functions in per-category
 * private headers (e.g. mglTraceRTYFlipDiagnosticsEnabled) call it. */
BOOL mglEnvFlagEnabled(const char *name);

/* === Lock infrastructure ===
 * These macros reference MGLRenderer ivars directly and therefore can only
 * be expanded inside @implementation MGLRenderer methods. */
static inline double mglNowSeconds(void)
{
    return CFAbsoluteTimeGetCurrent();
}

static inline void mglMetalLock(os_unfair_lock *lock) {
    os_unfair_lock_lock(lock);
}
static inline void mglMetalUnlock(os_unfair_lock *lock) {
    os_unfair_lock_unlock(lock);
}

#define METAL_LOCK()   do { \
    if (mglPerfLockTimingEnabled()) { \
        double _mlw = mglNowSeconds(); \
        [_metalStateLock lock]; \
        double _mln = mglNowSeconds(); \
        MGL_FRAME_ADD(g_mglLockWaitTimeSinceSwap, _mln - _mlw); \
        if (_metalLockHoldDepth < MGL_LOCK_TIMING_STACK_CAPACITY) { \
            _metalLockHoldStartStack[_metalLockHoldDepth] = _mln; \
        } \
        _metalLockHoldDepth++; \
    } else { \
        [_metalStateLock lock]; \
    } \
} while (0)
#define METAL_UNLOCK() do { \
    if (mglPerfLockTimingEnabled()) { \
        double _mln = mglNowSeconds(); \
        if (_metalLockHoldDepth > 0) { \
            _metalLockHoldDepth--; \
            if (_metalLockHoldDepth < MGL_LOCK_TIMING_STACK_CAPACITY) { \
                MGL_FRAME_ADD(g_mglLockHoldTimeSinceSwap, _mln - _metalLockHoldStartStack[_metalLockHoldDepth]); \
            } \
        } \
    } \
    [_metalStateLock unlock]; \
} while (0)
#define SYNC_LOCK()    do { mglMetalLock(&_syncListLock); } while (0)
#define SYNC_UNLOCK()  do { mglMetalUnlock(&_syncListLock); } while (0)

/* Returns the active GLMState pointer for sync functions.
 * During batch replay, _activeState points to the snapshot; otherwise
 * it falls back to the live &ctx->state. */
#define MGL_STATE(context)  (_activeState ? _activeState : &(context)->state)

@interface MGLRenderer () {
    NSView *_view;
    CAMetalLayer *_layer;
    id<CAMetalDrawable> _drawable;
    GLMContext  ctx;    // context macros need this exact name
    GLMState *_activeState;  // NULL = use live ctx->state (normal path)
    id<MTLDevice> _device;
    MGLCapability _capability;
    NSRecursiveLock *_metalStateLock;  // reentrant — dense call graph requires it
    double _metalLockHoldStartStack[MGL_LOCK_TIMING_STACK_CAPACITY];
    NSUInteger _metalLockHoldDepth;
    os_unfair_lock _syncListLock;
    NSUInteger _consecutiveGPUErrors;
    NSUInteger _consecutiveGPUSuccesses;
    NSTimeInterval _lastGPUErrorTime;
    BOOL _gpuErrorRecoveryMode;
    GLuint _interfaceMismatchBlockedProgram;
    CFTimeInterval _interfaceMismatchBlockedUntil;
    uint32_t _interfaceMismatchBlockedStreak;
    NSMutableArray *_proactiveTextures;
    MGLDrawable _drawBuffers[_MAX_DRAW_BUFFERS];
    BOOL _defaultDrawableWrittenSinceLastSwap;
    MTLBlendFactor _src_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _dst_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _src_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _dst_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation _rgb_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation _alpha_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLColorWriteMask _color_mask[MAX_COLOR_ATTACHMENTS];
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _pipelineState;
    MTLPixelFormat _pipelineColor0Format;
    MTLPixelFormat _pipelineDepthFormat;
    MTLPixelFormat _pipelineStencilFormat;
    GLuint _pipelineProgramName;
    NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *_pipelineStateCache;
    MTLRenderPassDescriptor *_renderPassDescriptor;
    Framebuffer *_renderPassFramebuffer;
    GLuint _renderPassFramebufferName;
    GLenum _renderPassDrawBuffer;
    GLsizei _renderPassDrawBufferCount;
    GLenum _renderPassDrawBuffers[MAX_COLOR_ATTACHMENTS];
    uint64_t _traceReplayFlushId;
    uint32_t _traceReplayBatchIndex;
    GLuint _dontCareFrameGeneration;
    id<MTLCommandBuffer> _currentCommandBuffer;
    SyncList  *_currentCommandBufferSyncList;
    id<MTLBuffer> _mdiArgsScratchBuffer;
    NSUInteger _mdiArgsScratchCapacity;
    NSUInteger _mdiArgsScratchOffset;
    id<MTLRenderCommandEncoder> _currentRenderEncoder;
#if MGL_HAS_MTL4_COMPILER
    id<MTL4Compiler> _mtl4Compiler;
#endif
    id<MTLTexture> _fallbackRenderTargetTexture;
    id<MTLBuffer> _visibilityResultBuffer;
    BOOL _sampleQueryActive;
    uint64_t _timerQueryBeginGPU;
    id<MTLTexture> _transientDepthTexture;
    NSUInteger _transientDepthTextureWidth;
    NSUInteger _transientDepthTextureHeight;
    id<MTLTexture> _fallbackSampledTexture;
    id<MTLTexture> _fallbackCubeSampledTexture;
    id<MTLBuffer> _fallbackTextureBufferStorage;
    id<MTLBuffer> _tessFactorBuffer;
    id<MTLBuffer> _tcsOutputBuffer;     /* TCS per-vertex output (spvOut, buffer 28) */
    id<MTLBuffer> _tcsPatchOutBuffer;   /* TCS per-patch output (spvPatchOut, buffer 27) */
    NSUInteger _tcsOutputStride;        /* bytes per TCS output vertex */
    GLuint _tcsOutVertices;             /* TCS output vertices per patch */
    id<MTLTexture> _fallbackSintTextureBuffer;
    NSMutableDictionary<NSNumber *, id<MTLTexture>> *_fallbackSampledTextureCache;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *_doubleVertexAttribBufferCache;
    id<MTLSamplerState> _fallbackSamplerState;
    MGLFragmentTextureTraceBinding _fragmentTextureTraceBindings[TEXTURE_UNITS];
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *_scaledBlitPipelineCache;
    id<MTLSamplerState> _scaledBlitNearestSampler;
    id<MTLSamplerState> _scaledBlitLinearSampler;
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *_scaledDepthBlitPipelineCache;
    NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *_msaaIntegerResolvePipelineCache;
    NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *_clearRectPipelineCache;
    id<MTLDepthStencilState> _clearRectDepthState;
    BOOL _currentDrawUsesRTSampledCopy;
    GLuint _blitOperationComplete;
    id<MTLEvent> _currentEvent;
    GLsizei _currentSyncName;
    BOOL _isCommittingCommandBuffer;
    MGLLastBoundBuffer _lastBoundVertexBuffers[kMGLMaxBufferSlots];
    MGLLastBoundBuffer _lastBoundFragmentBuffers[kMGLMaxBufferSlots];
    id<MTLTexture> _lastBoundVertexTextures[TEXTURE_UNITS];
    id<MTLTexture> _lastBoundFragmentTextures[TEXTURE_UNITS];
    id<MTLSamplerState> _lastBoundVertexSamplers[TEXTURE_UNITS];
    id<MTLSamplerState> _lastBoundFragmentSamplers[TEXTURE_UNITS];
    id<MTLRenderPipelineState> _lastPipelineState;
    id<MTLDepthStencilState> _lastDepthStencilState;
    MTLViewport _lastViewport;
    MTLScissorRect _lastScissorRect;
    MTLCullMode _lastCullMode;
    MTLWinding _lastFrontFacingWinding;
    MTLTriangleFillMode _lastTriangleFillMode;
    float _lastDepthBias;
    float _lastDepthBiasClamp;
    float _lastDepthSlopeScale;
    BOOL _lastBoundValid;
}

@end

/* === Aggregate imports of per-category private headers ===
 * These headers declare ObjC methods and C helpers implemented in each
 * category file.  They import MGLRenderer_Private.h for ivar/types access;
 * the include guards prevent infinite recursion.  Importing them here means
 * existing code that imports just MGLRenderer_Private.h continues to see ALL
 * declarations. */
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+RenderPass_Private.h"
#import "MGLRenderer+Blit_Private.h"
#import "MGLRenderer+Texture_Private.h"
#import "MGLRenderer+Query_Private.h"

#endif /* MGLRenderer_Private_h */
