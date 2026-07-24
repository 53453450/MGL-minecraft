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
/* Kept: many .m files transitively rely on this header for SPVC_* constants
 * (SPVC_RESOURCE_TYPE_*, spvc_compiler_*).  Removing it breaks 7+ categories. */
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

/* State container types and independent renderer subsystems. */
#import "MGLRenderer_State.h"
#import "MGLPipelineCache.h"
#import "MGLBindingSync.h"
#import "MGLQueryManager.h"
#import "MGLRenderPassManager.h"

/* Shared helpers — declared here because inline functions in per-category
 * private headers (e.g. mglTraceRTYFlipDiagnosticsEnabled) call them.
 * mglEnvFlagEnabled: unset → OFF.
 * mglEnvFlagEnabledDefaultOn: unset → ON; =0/false/no/off → OFF. */
BOOL mglEnvFlagEnabled(const char *name);
BOOL mglEnvFlagEnabledDefaultOn(const char *name);

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
        MGL_FRAME_ADD(g_mglLockWaitTimeSinceSwap, (uint64_t)((_mln - _mlw) * 1e9)); \
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
                MGL_FRAME_ADD(g_mglLockHoldTimeSinceSwap, (uint64_t)((_mln - _metalLockHoldStartStack[_metalLockHoldDepth]) * 1e9)); \
            } \
        } \
    } \
    [_metalStateLock unlock]; \
} while (0)
#define SYNC_LOCK()    do { mglMetalLock(&_syncListLock); } while (0)
#define SYNC_UNLOCK()  do { mglMetalUnlock(&_syncListLock); } while (0)

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
    MGLRendererCoreState _core;
    MGLGPURecoveryState _gpuRecovery;
    MGLPipelineCache *_pipelineCache;
    MGLQueryManager *_queryManager;
    MGLRenderPassManager *_renderPassManager;
    MGLResourceFallbackState _resourceFallback;
    MGLBlitState _blit;
    MGLBindingSync *_bindingSync;
    MGLTessellationState _tessellation;
    MGLBatchingState _batching;
    /* Command buffer that needs waitUntilCompleted after METAL_UNLOCK.
     * Set by flushCommandBufferLocked: when finish=true, consumed by
     * flushCommandBuffer: after unlock.  GL is single-threaded so no race. */
    id<MTLCommandBuffer> _pendingFinishCB;
    /* Cached current-vertex-attrib MTLBuffers.
     * Each slot caches the repeated (4096×) MTLBuffer built from
     * current_vertex_attrib[attrib], keyed on (attribBytes, stride).
     * Avoids per-draw NSMutableData + newBufferWithBytes when the
     * current value hasn't changed (common in Minecraft GUI/item pass). */
    id<MTLBuffer> _currentAttribBuffers[MAX_ATTRIBS];
    uint8_t       _currentAttribCacheBytes[MAX_ATTRIBS][16];
    NSUInteger    _currentAttribCacheStride[MAX_ATTRIBS];
    BOOL          _currentAttribCacheValid[MAX_ATTRIBS];
    /* Cached spvBufferSizeConstants MTLBuffers.
     * Each stage's size-constants buffer is a fixed 124-byte (31×uint32)
     * buffer bound at MGL_BUFFER_SIZE_BUFFER_INDEX when a shader uses
     * .length() on unsized SSBO arrays.  Cache the last buffer + its
     * contents and reuse it when the size constants are unchanged (the
     * common case — buffer sizes rarely change between draws in the same
     * frame).  When contents differ we allocate a new buffer so each draw
     * gets its own snapshot: the GPU reads buffer contents at command
     * buffer execution time, not at setVertexBuffer time, so we cannot
     * overwrite a buffer that earlier draws in the same CB still reference. */
    id<MTLBuffer> _vertexSizeBuffer;
    uint32_t      _vertexSizeConstantsCache[31];
    BOOL          _vertexSizeConstantsValid;
    id<MTLBuffer> _fragmentSizeBuffer;
    uint32_t      _fragmentSizeConstantsCache[31];
    BOOL          _fragmentSizeConstantsValid;
    /* Track whether the current command buffer has any encoded work
     * and the most recently committed CB.  When mtlFlush(ctx, true) is
     * called and the current CB has no work, we can wait on _lastCommittedCB
     * instead of committing an empty CB — Metal CBs on the same queue
     * execute serially, so waiting on the last committed CB guarantees all
     * prior GPU work is done.  This avoids a kernel-level commit + wait
     * for redundant sync calls (e.g., repeated glFinish, buffer read maps). */
    id<MTLCommandBuffer> _lastCommittedCB;
    BOOL                 _currentCBHasWork;
}

/* P1-5: cap an auxiliary cache at `limit` entries with FIFO eviction of
 * the oldest 1/4 on overflow.  Mirrors the pipeline state cache eviction
 * strategy.  Keeps unbounded auxiliary caches (blit/clear/resolve pipelines,
 * fallback textures, double-vertex buffers) from growing without bound. */
- (void)mglCapAuxCache:(NSMutableDictionary *)cache
                 limit:(NSUInteger)limit;

/* P2-1: Methods called from MGLRenderer+Compute.m.
 * mapGLBuffersToMTLBufferMap:stage: now declared in MGLRenderer+Buffer_Private.h. */
- (id<MTLBuffer>)isolatedStageBindingBufferForMap:(const BufferMap *)map
                                           source:(id<MTLBuffer>)source
                                   requiredLength:(NSUInteger)requiredLength;
- (void)clearStageBindingCopyBacks:(MGLStageBindingCopyBackList *)copyBacks;
- (void)clearStageBindingCopyBack:(MGLStageBindingCopyBackList *)copyBacks
                           atIndex:(NSUInteger)index;
- (bool)recordStageBindingCopyBack:(MGLStageBindingCopyBackList *)copyBacks
                           atIndex:(NSUInteger)index
                         temporary:(id<MTLBuffer>)temporary
                        destination:(id<MTLBuffer>)destination
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

/* P1-11: Locked variant of flushDrawBuffer: — caller must hold METAL_LOCK.
 * Defined in MGLRenderer+Batch.m, called from already-locked callers
 * (mtlSwapBuffersLocked:, flushCommandBufferLocked:). */
- (void)flushDrawBufferLocked:(GLMContext)glm_ctx;

@end

/* Temporary compatibility aliases for the state-container migration.
 * New or touched code should prefer the explicit container fields directly.
 * These aliases keep the existing category implementations behavior-identical
 * while shrinking MGLRenderer's ivar surface. */
#define _view _core.view
#define _layer _core.layer
#define _drawable _core.drawable
#define _activeState _core.activeState
#define _device _core.device
#define _capability _core.capability
#define _metalStateLock _core.metalStateLock
#define _metalLockHoldStartStack _core.metalLockHoldStartStack
#define _metalLockHoldDepth _core.metalLockHoldDepth
#define _syncListLock _core.syncListLock
#define _proactiveTextures _core.proactiveTextures
#define _drawBuffers _core.drawBuffers
#define _defaultDrawableWrittenSinceLastSwap _core.defaultDrawableWrittenSinceLastSwap
#define _commandQueue _core.commandQueue

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
#import "MGLRenderer+QuerySync_Private.h"
#import "MGLRenderer+Tessellation_Private.h"
#import "MGLRenderer+Lifecycle_Private.h"
#import "MGLRenderer+Buffer_Private.h"
#import "MGLRenderer+ProgramBinding_Private.h"
#import "MGLRenderer+SwapDiagnostics_Private.h"

#endif /* MGLRenderer_Private_h */
