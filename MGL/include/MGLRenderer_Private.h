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
#import "MGLPlatformRendererShell.h"
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>   // CAMetalLayer
#import <simd/simd.h>               // vector_float4, vector_uint2, etc.
#include <os/lock.h>

/* glm_context.h pulls in GLMContext, Texture, Buffer, Program, Framebuffer,
 * Sync, GLMState, MGLBatchPath, MGLDrawBatch, MAX_COLOR_ATTACHMENTS,
 * TEXTURE_UNITS, and GL types (GLenum, GLuint, GLsizei, ...). */
#include "glm_context.h"
#include "mgl_render_cpp.h"
#include "mgl_renderer_backend.h"
#import "mgl_capability.h"          // ivar type: MGLCapability
#import "mgl_texture_compat.h"      // MGLTextureDataKind
#import "mgl_trace_strategy.h"      // ivar type: MGLFragmentTextureTraceBinding
#import "mgl_readback.h"
#import "mgl_metal_ref.h"
#import "mgl_sync.h"
#import "mgl_rt_sync.h"
#import "mgl_blit_clip.h"
#import "mgl_state_compat.h"
#import "mgl_program_resource.h"
#import "mgl_safety.h"
#import "mgl_vertex_format.h"
#import "mgl_thread_affinity.h"      // MGL_ASSERT_GL_THREAD / mglClaimGLThread
#import "pixel_utils.h"
#import "mgl_frame_activity.h"      // mglPerfLockTimingEnabled, MGL_FRAME_ADD
#import "mgl_draw_buffer.h"
#import "mgl_buffer_slots.h"
#import "mgl_vertex_attrib_query.h"
#import "mgl_coordinate.h"
#import "mgl_shader_resource.h"
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

/* State container types and independent renderer subsystems. */
#import "MGLRenderer_State.h"
#import "MGLPipelineCache.h"
#import "MGLRenderPassManager.h"

#ifndef MGL_VALUE_GEOMETRY_TYPES
#define MGL_VALUE_GEOMETRY_TYPES 1
typedef struct MGLSizeValue_t { uint64_t width, height, depth; } MGLSizeValue;
typedef struct MGLOriginValue_t { int64_t x, y, z; } MGLOriginValue;
typedef struct MGLRegionValue_t { MGLOriginValue origin; MGLSizeValue size; } MGLRegionValue;
#endif

/* Shared helpers — declared here because inline functions in per-category
 * private headers (e.g. mglTraceRTYFlipDiagnosticsEnabled) call them.
 * mglEnvFlagEnabled: unset → OFF.
 * mglEnvFlagEnabledDefaultOn: unset → ON; =0/false/no/off → OFF. */
BOOL mglEnvFlagEnabled(const char *name);
BOOL mglEnvFlagEnabledDefaultOn(const char *name);

static inline MGLRenderer *mglRendererForContext(GLMContext context)
{
    return context
        ? (__bridge MGLRenderer *)context->platform_renderer_shell
        : nil;
}

static inline BOOL mglBindingStateIsValid(void *owner)
{
    uint32_t valid = 0;
    return owner && mglRenderCppBindingGetValid(owner, &valid) == 0 && valid;
}

static inline BOOL mglBindingStateBufferMatches(void *owner,
                                                uint32_t stage,
                                                void *buffer,
                                                uint64_t offset,
                                                uint32_t index)
{
    void *current = NULL;
    uint64_t currentOffset = 0;
    return owner && mglRenderCppBindingGetBuffer(
                        owner, stage, index, &current, &currentOffset) == 0 &&
           current == buffer && currentOffset == offset;
}

static inline BOOL mglBindingStatePipelineMatches(void *owner, void *pipeline)
{
    void *current = NULL;
    return owner &&
           mglRenderCppBindingGetPipelineState(owner, &current) == 0 &&
           current == pipeline;
}

static inline int mglBindingStateTextureSlotCount(void *owner)
{
    uint64_t mask[2] = {0, 0};
    if (!owner || mglRenderCppBindingGetTextureSlotMask(owner, mask) != 0) {
        return 0;
    }
    return __builtin_popcountll(mask[0]) + __builtin_popcountll(mask[1]);
}

/* MGL_MIP_DIAG=1 reports the effective sampler and mip chain of sampled
 * textures.  Independent of MGL_TRACE_LOG because the per-binding trace lines
 * are too dense to keep a frame rate high enough to observe view-dependent
 * artifacts.  Output goes to NSLog under the "MGL MIP_DIAG" prefix. */
static inline BOOL mglMipDiagEnabled(void)
{
    static BOOL enabled;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ enabled = mglEnvFlagEnabled("MGL_MIP_DIAG"); });
    return enabled;
}

/* Emits only on transitions, so a scene whose state is stable logs nothing and
 * a burst of lines pinpoints the state that flipped. */
static inline BOOL mglMipDiagStateChanged(uint64_t *cache, uint64_t signature)
{
    if (!cache || *cache == signature) {
        return NO;
    }
    *cache = signature;
    return YES;
}

static inline uint64_t mglMipDiagMixState(uint64_t signature, uint64_t value)
{
    return (signature ^ value) * 1099511628211ULL;
}

/* === GL-thread contract ===
 * The Metal layer is owned by a single thread (see mgl_thread_affinity.h).
 * These macros used to acquire _metalStateLock; the lock has been replaced by
 * the thread-affinity assertion (compiled out in Release), so every former
 * METAL_LOCK() site now validates the single-threaded contract instead.
 * Keep the LOCK/UNLOCK pairing so existing call sites remain structurally
 * unchanged. */
static inline double mglNowSeconds(void)
{
    return CFAbsoluteTimeGetCurrent();
}

/* Monotonic clock for duration/gap/age measurements (heartbeats, watchdog,
 * perf intervals).  Wall-clock mglNowSeconds is unsuitable: NTP steps can
 * make elapsed values negative or spuriously large. */
static inline double mglTraceNowSeconds(void)
{
    return (double)mglTraceClockNS() / 1000000000.0;
}

#define METAL_LOCK()   do { MGL_ASSERT_GL_THREAD(); } while (0)
#define METAL_UNLOCK() do { } while (0)

/* Returns the active GLMState pointer for Metal-layer sync functions.
 *
 * DUAL-PROXY INVARIANT: _activeState (ivar) and ctx->active_state (context
 * pointer) must always refer to the same logical GLMState.  They are two
 * proxies for the same concept:
 *   - ctx->active_state: used by STATE()/STATE_VAR()/VAO() macros in the
 *     C GL layer (glm_context.h)
 *   - _activeState: used by MGL_STATE() in the Metal layer
 *
 * Two valid configurations:
 *   (A) _activeState == NULL  -> MGL_STATE() falls through to ctx->active_state
 *                                (the default / post-teardown mode; both
 *                                conceptually refer to &ctx->state)
 *   (B) _activeState != NULL  -> _activeState MUST equal ctx->active_state
 *                                (the redirected mode used during batch replay)
 *
 * Enforcement: all writes to either proxy MUST go through the centralized
 * helpers declared below this macro:
 *   - mglRestoreLiveActiveStateForContext:     -> config (A)
 *   - mglAssertDualProxyInSyncForContext:      -> debug-mode checkpoint
 * These prevent proxy desync — a caller writing one proxy without the other.
 *
 * Checkpoints (NSCAssert, compiled out in release): flushDrawBuffer entry,
 * restoreStateForBatch entry, teardownBatchReplayForContext entry/exit.
 *
 * A desync causes STATE() and MGL_STATE() to read different GLMState objects,
 * producing wrong binds/dirty bits — intermittent render errors that are
 * extremely hard to debug. */
#define MGL_STATE(context)  (_activeState ? _activeState : (context)->active_state)

@interface MGLRenderer () {
    /* Keep this ivar named `ctx`: C GLM macros and older helper code expect
     * that identifier to exist inside MGLRenderer methods. */
    GLMContext  ctx;    // context macros need this exact name
    MGLRendererBackendHandle *_backend;
    /* Window whose resize/backing notifications are observed for geometry
     * publishing (see MGLRenderer+Lifecycle.m).  Weak: never retain a window
     * the host owns. */
    __weak NSWindow *_observedWindow;    MGLRendererCoreState _core;
    MGLGPURecoveryState _gpuRecovery;
    MGLPipelineCache *_pipelineCache;
    void *_queryStateOwner;
    MGLRenderPassManager *_renderPassManager;
    MGLResourceFallbackState _resourceFallback;
    void *_bindingStateOwner;
    MGLTessellationState _tessellation;
    MGLGeometryState _geometry;
    /* Most recent GL primitive mode handed to a draw entry point, used to
     * derive MTLRenderPipelineDescriptor.inputPrimitiveTopology (required by
     * Metal when the VS writes [[render_target_array_index]]). */
    GLenum _lastDrawPrimitiveMode;
    MGLBatchingState _batching;
    /* Track whether the current command buffer has encoded work. The C++
     * CommandBufferOwner retains the most recent submission for glFinish. */
    BOOL                 _currentCBHasWork;
}

/* Methods called from MGLRenderer+Compute.m.
 * mapGLBuffersToMTLBufferMap:stage: now declared in MGLRenderer+Buffer_Private.h. */
- (id)isolatedStageBindingBufferForMap:(const BufferMap *)map
                                           source:(id)source
                                   requiredLength:(NSUInteger)requiredLength;
- (void)clearStageBindingCopyBacks:(MGLStageBindingCopyBackList *)copyBacks;
- (void)clearStageBindingCopyBack:(MGLStageBindingCopyBackList *)copyBacks
                           atIndex:(NSUInteger)index;
- (bool)recordStageBindingCopyBack:(MGLStageBindingCopyBackList *)copyBacks
                           atIndex:(NSUInteger)index
                         temporary:(id)temporary
                        destination:(id)destination
                  destinationBuffer:(Buffer *)destinationBuffer
                 destinationOffset:(NSUInteger)destinationOffset
                             length:(NSUInteger)length;
- (bool)flushStageBindingCopyBacks:(MGLStageBindingCopyBackList *)copyBacks
              requireCPUVisibility:(BOOL)requireCPUVisibility;

/* DUAL-PROXY INVARIANT HELPERS: centralize writes to _core.activeState and
 * ctx->active_state to prevent desync.  See the DUAL-PROXY INVARIANT comment
 * above MGL_STATE() for the invariant definition.
 *
 * Use these instead of writing either proxy directly:
 *   - mglRestoreLiveActiveStateForContext:  batch replay teardown (revert to
 *                                          live ctx->state, ivar = NULL)
 *   - mglAssertDualProxyInSyncForContext:   debug-mode checkpoint
 *                                          (NSCAssert compiled out in release)
 */
- (void)mglRestoreLiveActiveStateForContext:(GLMContext)glm_ctx;
- (void)mglAssertDualProxyInSyncForContext:(GLMContext)glm_ctx;

/* Locked variant of flushDrawBuffer: — caller must hold METAL_LOCK.
 * Defined in MGLRenderer+Batch.m, called from already-locked callers
 * (mtlSwapBuffersLocked:, flushCommandBufferLocked:). */
- (void)flushDrawBuffer:(GLMContext)glm_ctx;
- (void)flushDrawBufferLocked:(GLMContext)glm_ctx;
- (void)mtlDeleteMTLObj:(GLMContext)glm_ctx buffer:(void *)obj;
- (void)mtlFlush:(GLMContext)glm_ctx finish:(bool)finish;
- (void)mtlInvalidateRenderPass:(GLMContext)glm_ctx;
- (void)mtlBufferSubData:(GLMContext)glm_ctx buf:(Buffer *)buf
                  offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr;
- (void *)mtlMapUnmapBuffer:(GLMContext)glm_ctx buf:(Buffer *)buf
                      offset:(size_t)offset size:(size_t)size
                      access:(GLenum)access map:(bool)map;
- (void)mtlReadBackBuffer:(GLMContext)glm_ctx buf:(Buffer *)buf
                   offset:(size_t)offset size:(size_t)size;
- (void)mtlFlushMappedBufferRange:(GLMContext)glm_ctx buf:(Buffer *)buf
                           offset:(GLintptr)offset length:(GLsizeiptr)length;

/* Drawable-geometry hand-off (component 3 of the lock replacement).
 * mglMainThreadSyncViewGeometry: main-thread only; reads NSView/NSWindow/
 * NSScreen geometry, writes the layer frame/contentsScale and the atomic
 * drawable-size snapshot.  mglApplyPendingDrawableSize: GL thread only;
 * consumes the snapshot and sets CAMetalLayer.drawableSize (safe off main). */
- (void)mglMainThreadSyncViewGeometry;
- (CGSize)mglApplyPendingDrawableSize;

@end

/* Temporary compatibility aliases for the state-container migration.
 * New or touched code should prefer the explicit container fields directly.
 * These aliases keep the existing category implementations behavior-identical
 * while shrinking MGLRenderer's ivar surface. */
#define _view self.view
#define _layer self.layer
#define _drawable self.drawable
#define _activeState _core.activeState
#define _device ((__bridge id) \
    mglRendererBackendGetDevice(_backend))
#define _capability _core.capability
#define _drawBuffers _core.drawBuffers
#define _defaultDrawableWrittenSinceLastSwap _core.defaultDrawableWrittenSinceLastSwap
#define _commandQueueOwner mglRendererBackendGetOwner( \
    _backend, MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE)
#define _commandQueue ((__bridge id) \
    mglRendererBackendGetCommandQueue(_backend))
#define _deviceResetRequested _core.deviceResetRequested
#define _pendingDrawableW _core.pendingDrawableW
#define _pendingDrawableH _core.pendingDrawableH
#define _drawableSizeDirty _core.drawableSizeDirty

/* === Aggregate imports of per-category private headers ===
 * These headers declare ObjC methods and C helpers implemented in each
 * category file.  They import MGLRenderer.h (interface + glm_context types)
 * only — no include cycle with MGLRenderer_Private.h.  Importing them here
 * means existing code that imports just MGLRenderer_Private.h continues to
 * see ALL declarations. */
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+RenderPass_Private.h"
#import "MGLRenderer+Binding_Private.h"
#import "MGLRenderer+VertexLayout_Private.h"
#import "MGLRenderer+GPURecovery_Private.h"
#import "MGLRenderer+Blit_Private.h"
#import "MGLRenderer+Texture_Private.h"
#import "MGLRenderer+Tessellation_Private.h"
#import "MGLRenderer+Lifecycle_Private.h"
#import "MGLRenderer+Buffer_Private.h"
#import "MGLRenderer+SwapDiagnostics_Private.h"

#endif /* MGLRenderer_Private_h */
