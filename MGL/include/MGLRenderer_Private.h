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
 * MGLRenderer_Private.h
 * MGL
 *
 * Private class extension for MGLRenderer — exposes ivars and private methods
 * needed by ObjC category files (MGLRenderer+Draw.m, +RenderPass.m, etc.).
 *
 * ObjC categories in separate files CANNOT access ivars declared in
 * @implementation blocks.  This header moves all ivars into a class extension
 * so that category files importing this header can reference them directly.
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

/* Import glm_context.h for GLMContext, Texture, Buffer, Program, Framebuffer,
 * Sync, GLMState, MGLBatchPath, MGLDrawBatch, MAX_COLOR_ATTACHMENTS,
 * TEXTURE_UNITS, and GL types (GLenum, GLuint, GLsizei, ...).
 * glm_context.h transitively pulls in glm_params.h → glcorearb.h + glm_limits.h,
 * draw_command.h, and mgl_types_*.h. */
#include "glm_context.h"

/* ivar type: MGLCapability */
#import "mgl_capability.h"

/* method-decl type: MGLTextureDataKind */
#import "mgl_texture_compat.h"

/* ivar type: MGLFragmentTextureTraceBinding */
#import "mgl_trace_strategy.h"

/* C helpers used by category files — declared in these headers but not
 * transitively pulled in by the imports above. */
#import "mgl_readback.h"       // mglMetalPixelFormatIsIntegerColor, etc.
#import "mgl_metal_ref.h"      // mglSafeReleaseMetalObj
#import "mgl_sync.h"           // mglMetalAttachmentSubresourceForAttachment
#import "mgl_rt_sync.h"        // mglTextureCanUseGLSampledRenderTargetCopy
#import "mgl_blit_clip.h"      // MGLBlitAxis, mglClipBlitAxis
#import "mgl_state_compat.h"   // mglNearlyEqual
#import "mgl_msl_compat.h"     // mglZeroToOneVertexMSLSource, etc.
#import "mgl_safety.h"         // mglPointerRangeIsReadable, mglObjectPointerLooksPlausible
#import "mgl_vertex_format.h"  // mglMaybeInvertMTLWinding, mglVertexFormatName
#import "spirv_cross_c.h"      // SPVC_RESOURCE_TYPE_* enums
/* pixel_utils.h defines its own MTLPixelFormat enum unless
 * MGL_NO_MTL_PIXEL_FORMAT is defined; Metal.framework already provides it. */
#define MGL_NO_MTL_PIXEL_FORMAT
#import "pixel_utils.h"        // sizeForInternalFormat
#undef MGL_NO_MTL_PIXEL_FORMAT
#import "mgl_frame_activity.h" // mglPerfLockTimingEnabled, MGL_FRAME_ADD, globals
#import "mgl_draw_buffer.h"    // mglDefaultDrawBufferIndexForGL
#import "mgl_buffer_slots.h"   // kMGLCullDistanceVertexBufferIndex, etc.
#import "mgl_vertex_attrib_query.h"  // mglRendererProgramUsesVertexAttrib, mglRendererVertexAttribUsesCurrentValue
#import "mgl_coordinate.h"           // mglRTWriteAuthorityIsCurrentAndUsesOriginal
#import "mgl_spirv_resource.h"       // mglSpirvResourceTypeName, mglStageBufferResourceElementCount, etc.
#import "mgl_buffer_query.h"         // mglRendererBufferHasDrawableContents
#import "mgl_focus_program.h"        // mglFocusLoadingProgram, mglIsFocusedLoadingProgram
#import "mgl_draw_mode.h"            // mglPolygonModePointForDrawMode, mglPrimitiveRestartIndexForType
#import "mgl_index_buffer.h"         // mglPreparedElementIndexBuffer
#import "mgl_draw_encode.h"          // mglEncodeArrayQuads, MGLPrimitiveRestartEncodeResult, etc.

/* FBO attachment helpers — defined in framebuffers.c, extern-declared here
 * because they're not in any public header.  MGLRenderer.m has the same
 * extern declarations at the point of use. */
extern bool isColorAttachment(GLMContext ctx, GLuint attachment);
extern FBOAttachment *getFBOAttachment(GLMContext ctx, Framebuffer *fbo, GLenum attachment);

/* Texture lookup — defined in textures.c, extern-declared here because it's
 * not in any public header.  MGLRenderer.m has the same extern declaration. */
extern Texture *findTexture(GLMContext ctx, GLuint texture);

/* === MTL4 compiler support (conditional) ===
 * Needed by the _mtl4Compiler ivar below.  Mirrors the same conditional in
 * MGLRenderer.m; #import prevents double-inclusion. */
#if __has_include(<Metal/MTL4Compiler.h>) && __has_include(<Metal/MTL4LibraryDescriptor.h>)
#import <Metal/MTL4Compiler.h>
#import <Metal/MTL4LibraryDescriptor.h>
#define MGL_HAS_MTL4_COMPILER 1
#else
#define MGL_HAS_MTL4_COMPILER 0
#endif

/* === Types previously defined in MGLRenderer.m, needed by ivars below === */

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

/* Last-bound state cache for render encoder dedup.
 * Avoids redundant setVertexBuffer/setFragmentBuffer/setRenderPipelineState
 * calls when the resource and offset haven't changed. */
typedef struct {
    id<MTLBuffer> __strong buffer;
    NSUInteger offset;
} MGLLastBoundBuffer;

#define kMGLMaxBufferSlots 31

/* Per-worker context for parallel command recording (Stage 5.3).
 *
 * Each worker thread encodes draws onto its own MTLRenderCommandEncoder
 * obtained from a shared MTLParallelRenderCommandEncoder.  The dedup
 * state (last-bound buffers/textures/pipeline/etc.) must be per-worker
 * to prevent one worker's bindings from causing another to skip a
 * needed bind.
 *
 * During sequential replay (the current default), the renderer's ivars
 * are used directly.  When parallel recording is enabled, a small array
 * of MGLWorkerContext is created and the issue/state-apply methods read
 * from the active worker instead of from shared ivars. */
typedef struct {
    id<MTLRenderCommandEncoder> encoder;

    /* Dedup state — mirrors the corresponding MGLRenderer ivars. */
    MGLLastBoundBuffer lastBoundVertexBuffers[kMGLMaxBufferSlots];
    MGLLastBoundBuffer lastBoundFragmentBuffers[kMGLMaxBufferSlots];
    id<MTLTexture> lastBoundVertexTextures[TEXTURE_UNITS];
    id<MTLTexture> lastBoundFragmentTextures[TEXTURE_UNITS];
    id<MTLSamplerState> lastBoundVertexSamplers[TEXTURE_UNITS];
    id<MTLSamplerState> lastBoundFragmentSamplers[TEXTURE_UNITS];
    id<MTLRenderPipelineState> lastPipelineState;
    id<MTLDepthStencilState> lastDepthStencilState;
    MTLViewport lastViewport;
    MTLScissorRect lastScissorRect;
    MTLCullMode lastCullMode;
    MTLWinding lastFrontFacingWinding;
    MTLTriangleFillMode lastTriangleFillMode;
    float lastDepthBias;
    float lastDepthBiasClamp;
    float lastDepthSlopeScale;
    BOOL lastBoundValid;

    /* Per-worker pipeline state (points into the shared _pipelineStateCache). */
    id<MTLRenderPipelineState> pipelineState;
    MTLPixelFormat pipelineColor0Format;
    MTLPixelFormat pipelineDepthFormat;
    MTLPixelFormat pipelineStencilFormat;
    GLuint pipelineProgramName;

    /* Per-worker MDI scratch offset (shares the renderer's buffer). */
    NSUInteger mdiArgsScratchOffset;

    /* Per-worker trace. */
    uint64_t traceReplayFlushId;
    uint32_t traceReplayBatchIndex;
} MGLWorkerContext;

/* === Shader parameter structs (moved from MGLRenderer.m) ===
 * Used by blit/copy/resolve pipelines in MGLRenderer+Blit.m and MGLRenderer.m.
 * The Metal shader string mirrors these layouts — keep field order in sync. */
typedef struct MGLScaledBlitParams_t {
    vector_float4 uvRect; // xy=min, zw=max in normalized Metal texture coordinates.
    float forceOpaqueAlpha;
    vector_float3 _padding;
} MGLScaledBlitParams;

typedef struct MGLMSAAIntegerResolveParams_t {
    vector_uint2 srcOrigin;
    vector_uint2 dstOrigin;
    vector_uint2 size;
    vector_uint2 _padding;
} MGLMSAAIntegerResolveParams;

typedef struct MGLClearRectParams_t {
    vector_float4 color;
    float depth;
    vector_float3 _padding;
} MGLClearRectParams;

/* === Lock timing stack capacity ===
 * Referenced by the _metalLockHoldStartStack ivar below and by the
 * METAL_LOCK/METAL_UNLOCK macros in MGLRenderer.m. */
#define MGL_LOCK_TIMING_STACK_CAPACITY 64

/* === C functions defined in MGLRenderer.m, used by category files ===
 * These were `static` in MGLRenderer.m; removing `static` and declaring
 * them here allows MGLRenderer+Blit.m / +Texture.m / +Query.m to call them. */
void mglEnableIndirectCommandBuffersForPipeline(MTLRenderPipelineDescriptor *pipelineStateDescriptor);
void mglMetalCopyRows(const uint8_t *src,
                      NSUInteger srcBytesPerRow,
                      uint8_t *dst,
                      NSUInteger dstBytesPerRow,
                      NSUInteger rowBytes,
                      NSUInteger height,
                      BOOL flipY);
BOOL mglEnvFlagEnabled(const char *name);

/* GL internal format → CPU (format, type) mapping — defined in MGLRenderer.m.
 * Used by mtlCopyImageSubData for format-converting readback. */
GLboolean mglGetCPUFormatTypeForInternalFormat(GLenum internalformat,
                                               GLenum *outFormat,
                                               GLenum *outType);

/* Render-pass lifecycle / pipeline helpers — defined in MGLRenderer.m,
 * used by MGLRenderer+RenderPass.m.  Formerly `static` in MGLRenderer.m. */
void mglLogRenderPassLifecycle(const char *tag,
                               uint64_t call,
                               GLMContext ctx,
                               id<MTLCommandBuffer> commandBuffer,
                               id<MTLRenderCommandEncoder> renderEncoder,
                               MTLRenderPassDescriptor *renderPassDescriptor,
                               id<CAMetalDrawable> drawable,
                               Framebuffer *renderPassFramebuffer,
                               GLuint renderPassFramebufferName,
                               GLenum renderPassDrawBuffer,
                               GLsizei renderPassDrawBufferCount);
NSRange mglRendererFindMSLEntryParameterClose(NSString *msl, const char *entryPoint);
GLuint mglCurrentRenderProgramKey(GLMContext ctx);
void mglWriteProgramMSLDump(Program *program, NSString *reason);
GLuint mglRendererSafeFramebufferName(GLMContext ctx);
id<MTLTexture> mglApplySRGBStateToRenderTarget(id<MTLTexture> texture, GLMContext ctx);
Program *mglResolveProgramFromState(GLMContext ctx);
BOOL mglRendererPointerInHashTable(HashTable *table, const void *ptr);

/* Render-pass pipeline helpers — defined in MGLRenderer.m, used by
 * MGLRenderer+RenderPass.m.  Formerly `static` in MGLRenderer.m. */
Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);
void mglNormalizePipelineDepthStencilFormats(MTLRenderPipelineDescriptor *desc,
                                             const char *label);
VertexArray *mglRendererGetValidatedVAO(GLMContext ctx, const char *where);

/* Resolved vertex-attrib binding — formerly a static typedef in MGLRenderer.m. */
typedef struct MGLResolvedVertexAttribBinding_t {
    const VertexAttrib *attrib;
    Buffer *buffer;
    GLintptr binding_offset;
    GLuint stride;
    GLuint divisor;
    GLintptr relativeoffset;
    GLuint binding_index;
    bool uses_binding_table;
} MGLResolvedVertexAttribBinding;

bool mglRendererResolveVertexAttribBinding(GLMContext ctx,
                                           VertexArray *vao,
                                           GLuint attribute,
                                           const char *where,
                                           MGLResolvedVertexAttribBinding *out);
int mglRendererResolveVertexAttributeBufferIndex(GLMContext ctx,
                                                 VertexArray *vao,
                                                 GLuint attribute,
                                                 const char *where);

/* GL type/size → Metal vertex format — defined in MGLRenderer.m (non-static). */
MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized);

/* GL texture → Metal pixel format — defined in pixel_utils.c. */
MTLPixelFormat mtlPixelFormatForGLTex(Texture *gl_tex);

/* Render-pass logging / validation helpers — defined in MGLRenderer.m,
 * used by MGLRenderer+RenderPass.m.  Formerly `static` in MGLRenderer.m. */
void mglLogLoopHeartbeat(const char *tag,
                         uint64_t callCount,
                         double nowSeconds,
                         double *lastCallSeconds,
                         uint64_t *lastCallCount,
                         double warnGapSeconds);
void mglLogStateSnapshot(const char *tag,
                         GLMContext ctx,
                         id<MTLCommandBuffer> commandBuffer,
                         id<MTLRenderCommandEncoder> renderEncoder,
                         MTLRenderPassDescriptor *renderPassDescriptor,
                         id<CAMetalDrawable> drawable);
Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx, const char *where);

/* MSL identifier constant — formerly `static` in MGLRenderer.m. */
static const char *kMGLFragCoordParamsMSLName = "_mglFragCoordParams";

/* === Diagnostic constants — moved from MGLRenderer.m ===
 * These are used by MGLRenderer.m and category files alike.  Keeping them
 * `static const` in the header gives each TU its own copy (fine for consts).
 * Declared here (before the inline helpers below) so that helpers such as
 * mglShouldTraceCall can reference kMGLDiagnosticStateLogs. */
static const BOOL kMGLDisableSharedEventSync = YES;
static const BOOL kMGLVerboseFrameLoopLogs = NO;
static const BOOL kMGLVerbosePipelineLogs = NO;
static const BOOL kMGLDiagnosticStateLogs = NO;
static const BOOL kMGLDrawSubmitDiagnostics = NO;
static const BOOL kMGLSynchronizeTextureUploads = NO;
static const NSTimeInterval kMGLTextureUploadWaitTimeoutSeconds = 0.25;
static const BOOL kMGLUseDedicatedTextureUploadCommandBuffer = NO;

/* Swap-present diagnostics flag — intentionally low-frequency so Prism's
 * 100k-line cap does not hide the final compositing evidence. */
static const BOOL kMGLSwapPresentDiagnostics = NO;

/* === Draw binding/validation constants — moved from MGLRenderer.m ===
 * Used by MGLRenderer+Draw.m (vertex buffer binding, texture binding,
 * draw validation).  Keeping them `static const` in the header gives each
 * TU its own copy (fine for consts). */

/* Verbose bind tracing — off by default; per-draw logging stalls the render thread. */
static const BOOL kMGLVerboseBindLogs = NO;

/* Metal validation requires bound stage buffers to satisfy argument byte length.
 * Conservative minimum for low-index base/resource slots. */
static const NSUInteger kMGLMinimumStageBindingSize = 256;
static const NSUInteger kMGLDefaultStageFallbackBufferSize = 4096;
static const NSUInteger kMGLStageBindingStackScratchSize = 1024;

/* Keep low-index vertex resource slots bound during diagnostics. */
static const BOOL kMGLEnableVertexAllSlotFallback = YES;
static const BOOL kMGLEnableSampledTextureFallback = YES;

/* Mirror Metal's drawArrays vertex-buffer range validation before calling into
 * the debug layer. Metal aborts the process for these errors; we want a log and
 * a skipped draw instead. */
static const BOOL kMGLValidateDrawArraysVboRange = YES;
static const BOOL kMGLValidateDrawElementsVboRange = YES;

/* Slot indices for point-size params and TCS stage-in have different names
 * in the renderer than in the header; #define bridges them.  Identically-named
 * constants (FragCoordParams, CullDistance*) resolve to the header's enum
 * values automatically. */
#define kMGLPointSizeParamBufferIndex     kMGLPointSizeBufferIndex
#define kMGLTCSStageInReplBufferIndex     kMGLBufferSlot_TCSStageInRepl

/* Inline helpers — moved from MGLRenderer.m so category files can use them.
 * Each TU gets its own copy (fine for small functions). */
static inline BOOL mglRendererObjectPointerLikelyValid(const void *ptr)
{
    return mglObjectPointerLooksPlausible(ptr);
}

static inline bool mglShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

/* RT Metal-fill marker — inline because it's small and called from
 * both MGLRenderer.m and MGLRenderer+Blit.m. */
static inline void mglMarkTextureLevelMetalFilled(Texture *tex, GLuint level, size_t uploadSize)
{
    TextureLevel *texLevel = mglTextureAttachmentLevel(tex, level);
    if (!texLevel) {
        return;
    }

    texLevel->ever_written = GL_TRUE;
    texLevel->has_initialized_data = GL_TRUE;
    texLevel->suspicious_zero_upload = GL_FALSE;
    texLevel->last_init_source = kTexMetalFill;
    texLevel->last_upload_size = uploadSize;
    texLevel->last_src_ptr = NULL;
    texLevel->last_src_hash = 0ull;

    if (tex->is_render_target) {
        tex->mtl_render_target_write_version++;
    }
}

/* === RT-write marker — used by Blit.m and Texture.m ===
 * The impl lives in MGLRenderer.m (non-static); the macro is here so category
 * files can call mglMarkTextureLevelRenderTargetWritten(tex, level). */
void mglMarkTextureLevelRenderTargetWrittenImpl(Texture *tex,
                                                GLuint level,
                                                const char *caller,
                                                int line);

#define mglMarkTextureLevelRenderTargetWritten(tex, level) \
    mglMarkTextureLevelRenderTargetWrittenImpl((tex), (level), __func__, __LINE__)

/* === Timing helper — moved from MGLRenderer.m === */
static inline double mglNowSeconds(void)
{
    return CFAbsoluteTimeGetCurrent();
}

/* === RT Y-flip diagnostic helpers — moved from MGLRenderer.m === */
static inline BOOL mglTraceRTYFlipDiagnosticsEnabled(void)
{
    return mglTraceLogIsEnabled() && mglEnvFlagEnabled("MGL_TRACE_RT_YFLIP");
}

static inline const char *mglYFlipDecisionName(MGLYFlipDecision decision)
{
    switch (decision) {
        case MGL_YFLIP_USE_ORIGINAL: return "original";
        case MGL_YFLIP_USE_SAMPLED_COPY: return "sampled-copy";
        case MGL_YFLIP_USE_ORIGINAL_AND_INJECT: return "original-inject";
        default: return "unknown";
    }
}

/* === Lock infrastructure — moved from MGLRenderer.m ===
 * These macros reference MGLRenderer ivars directly and therefore can only
 * be expanded inside @implementation MGLRenderer methods. */
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

@interface MGLRenderer () {
    NSView *_view;

    CAMetalLayer *_layer;
    id<CAMetalDrawable> _drawable;

    GLMContext  ctx;    // context macros need this exact name

    id<MTLDevice> _device;

    /* AGX Capability Layer: centralized device detection + driver bug markers.
     * Initialized once in initMetalLayer.  Queries are cached, no Metal API
     * calls after init.  See mgl_capability.h. */
    MGLCapability _capability;

    // CRITICAL FIX: Thread synchronization to prevent race conditions.
    // NSRecursiveLock is reentrant — required because the MGLRenderer call
    // graph is densely interconnected (draw → processGLState → newRenderEncoder
    // → endRenderEncoding → updateGLSampledRenderTargetCopyForTexture →
    // ensureWritableCommandBuffer, etc.).  A non-reentrant lock (os_unfair_lock)
    // deadlocked on indirect re-entry through non-target helper methods.
    // The Locked pattern (public wrapper + *Locked impl) is retained for
    // structural clarity but no longer relies on non-reentrancy.
    NSRecursiveLock *_metalStateLock;
    double _metalLockHoldStartStack[MGL_LOCK_TIMING_STACK_CAPACITY];
    NSUInteger _metalLockHoldDepth;
    os_unfair_lock _syncListLock;   // independent lock for _currentCommandBufferSyncList

    // AGX GPU Error Tracking - Prevent command queue from entering error state
    NSUInteger _consecutiveGPUErrors;
    NSUInteger _consecutiveGPUSuccesses;
    NSTimeInterval _lastGPUErrorTime;
    BOOL _gpuErrorRecoveryMode;

    // Quarantine programs that repeatedly fail VS/FS interface validation.
    GLuint _interfaceMismatchBlockedProgram;
    CFTimeInterval _interfaceMismatchBlockedUntil;
    uint32_t _interfaceMismatchBlockedStreak;

    // PROACTIVE TEXTURE STORAGE - Essential textures created during initialization
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

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _commandQueue;

    // The render pipeline generated from the vertex and fragment shaders in the .metal shader file.
    id<MTLRenderPipelineState> _pipelineState;
    MTLPixelFormat _pipelineColor0Format;
    MTLPixelFormat _pipelineDepthFormat;
    MTLPixelFormat _pipelineStencilFormat;
    GLuint _pipelineProgramName;
    NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *_pipelineStateCache;

    // render pass descriptor containts the binding information for VAO's and such
    MTLRenderPassDescriptor *_renderPassDescriptor;
    Framebuffer *_renderPassFramebuffer;
    GLuint _renderPassFramebufferName;
    GLenum _renderPassDrawBuffer;
    GLsizei _renderPassDrawBufferCount;
    GLenum _renderPassDrawBuffers[MAX_COLOR_ATTACHMENTS];
    uint64_t _traceReplayFlushId;
    uint32_t _traceReplayBatchIndex;

    /* Stage 4.2 DontCare inference: bumped once per swap. A color attachment
     * whose texture's mtl_rt_frame_generation != this value has not yet been
     * written this frame, so (with no pending clear and blending off) its
     * first pass can use loadAction=DontCare instead of Load. Starts at 1 so
     * a texture's zero-initialized stamp never accidentally matches. */
    GLuint _dontCareFrameGeneration;

    // each pass a new command buffer is created
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

    /* Metal visibility result buffer for GL occlusion queries
     * (GL_SAMPLES_PASSED / GL_ANY_SAMPLES_PASSED). When a sample query is
     * active, the render pass descriptor's visibilityResultBuffer is set and
     * the encoder enables MTLVisibilityResultModeBoolean so the GPU writes 1
     * if any samples pass per-fragment tests. */
    id<MTLBuffer> _visibilityResultBuffer;
    BOOL _sampleQueryActive;

    /* GPU timer query state — stores the GPU timestamp sampled at
     * mtlBeginTimerQuery so mtlEndTimerQuery can compute the delta. */
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
    GLuint _tcsOutVertices;             /* TCS output vertices per patch (layout(vertices=N) out) */
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

    /* Last-bound state for render encoder dedup */
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
    BOOL _lastBoundValid;  /* NO after encoder recreation, YES after first bind */
}

// === Private method declarations ===
- (void)initializeMTL4CompilerIfAvailable;
- (id<MTLLibrary>)newMetalLibraryWithSource:(NSString *)source
                                    options:(MTLCompileOptions *)options
                                      label:(NSString *)label
                                      error:(NSError **)error;
- (MGLBatchPath)scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (bool)syncRenderPassStateForContext:(GLMContext)glm_ctx;
- (bool)rotateRenderEncoderForCurrentFramebufferLocked;
- (bool)syncPipelineStateWithDeferredBufferMap:(bool)deferredBufferMapForPipelineBuild;
- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx;
- (BOOL)shouldUseDontCareLoadForColorTexture:(Texture *)tex
                             firstUseThisFrame:(BOOL)firstUseThisFrame;
- (BOOL)prepareRenderPassIfFBOChanged:(MGLDrawBatch *)batch
                              context:(GLMContext)glm_ctx
                          replayError:(GLenum *)replayError;
- (BOOL)checkBatchShouldExecute:(MGLDrawBatch *)batch
                        context:(GLMContext)glm_ctx
                        flushId:(uint64_t)flushId
                     batchIndex:(uint32_t)batchIndex
                    replayError:(GLenum *)replayError
                skippedCommands:(uint32_t *)skippedCommands;
- (void)recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (BOOL)issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (BOOL)issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (id<MTLBuffer>)mdiArgumentScratchBufferWithLength:(NSUInteger)length
                                             offset:(NSUInteger *)offsetOut;
- (BOOL)shouldSkipGPUOperations;
- (void)recordGPUError;
- (void)recordGPUSuccess;
- (NSUInteger)getOptimalAlignmentForPixelFormat:(MTLPixelFormat)format;
- (void)commitCommandBufferWithAGXRecovery:(id<MTLCommandBuffer>)commandBuffer;
- (void)resetMetalState;
- (BOOL)validateMetalObjects;
- (void)cleanupCommandBuffer;

// Stage 5.3: Parallel command recording infrastructure
- (void)saveDedupStateToWorker:(MGLWorkerContext *)worker;
- (void)loadDedupStateFromWorker:(const MGLWorkerContext *)worker;
- (BOOL)parallelEncodeEnabled;
- (MGLBatchPath)encodeBatchForParallelWorker:(MGLWorkerContext *)worker
                                       batch:(MGLDrawBatch *)batch
                                     context:(GLMContext)glm_ctx
                                     flushId:(uint64_t)flushId
                                  batchIndex:(uint32_t)batchIndex
                                  savedState:(const GLMState *)savedState
                                    executed:(BOOL *)executedOut;

// Phase 3 Thread Safety: *Locked variants (private, no lock — called from public wrappers)
- (void)endRenderEncodingLocked;
- (void)mtlDeleteMTLObjLocked:(GLMContext)glm_ctx buffer:(void *)obj;
- (void)bindMTLBufferLocked:(Buffer *)ptr;
- (bool)bindMTLProgramLocked:(Program *)ptr;
- (bool)newCommandBufferLocked;
- (bool)ensureWritableCommandBufferLocked:(const char *)reason;
- (bool)bindMTLTextureLocked:(Texture *)tex;
- (bool)newRenderEncoderLocked;
- (void)flushCommandBufferLocked:(bool)finish;
- (bool)processGLStateLocked:(bool)draw_command;
- (void)invalidateLastBoundState;
- (void)recordLastBoundVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)recordLastBoundFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index;
- (void)invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index;
- (void)setVertexTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setFragmentTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setVertexSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index;
- (void)setFragmentSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index;
- (void)setViewportIfNeeded:(MTLViewport)viewport;
- (void)setScissorRectIfNeeded:(MTLScissorRect)rect;
- (void)setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode;
- (id<MTLTexture>)freshGLSampledRenderTargetCopyForSampling:(Texture *)tex
                                                     source:(id<MTLTexture>)source
                                                      stage:(const char *)stage
                                                    program:(GLuint)programName
                                                    binding:(GLuint)binding
                                                       unit:(GLuint)unit
                                               expectedType:(MTLTextureType)expectedType
                                               expectedKind:(MGLTextureDataKind)expectedKind;
- (void)mtlDrawArraysLocked:(GLMContext)ctx mode:(GLenum)mode first:(GLint)first count:(GLsizei)count;
- (void)mtlDrawElementsLocked:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices;
- (void)mtlSwapBuffersLocked:(GLMContext)glm_ctx;
- (void)mtlBufferSubDataLocked:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr;
- (void)mtlTexSubImageLocked:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset;

// === Public wrapper methods (non-locking; call the *Locked variants) ===
// Declared here so category files can call them on `self`.
- (void)endRenderEncoding;
- (bool)ensureWritableCommandBuffer:(const char *)reason;
- (bool)newCommandBuffer;
- (bool)bindMTLTexture:(Texture *)tex;
- (bool)processGLState:(bool)draw_command;
- (void)flushCommandBuffer:(bool)finish;

// === Other methods defined in MGLRenderer.m, called from category files ===
- (Texture *)framebufferAttachmentTexture:(FBOAttachment *)fbo_attachment;
- (BOOL)currentRenderPassUsesTexture:(id<MTLTexture>)texture;
- (bool)restoreRenderEncoderAfterTextureUploadForDraw:(const char *)reason;
- (BOOL)mglEnsureLayerDrawableSizeAtLeastWidth:(NSUInteger)requiredWidth
                                        height:(NSUInteger)requiredHeight
                                        reason:(const char *)reason;
- (BOOL)synchronizeRenderPassForTextureReadback:(id<MTLTexture>)texture
                                          reason:(const char *)reason;
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat;
- (CGSize)mglSyncLayerDrawableSizeFromView:(const char *)reason;

// === Cross-category methods (defined in one category, called from another) ===
// Blit.m — called from Texture.m and MGLRenderer.m
- (id<MTLTexture>)resolvedReadbackTextureForMultisampleTexture:(id<MTLTexture>)sourceTexture
                                                   sourceLevel:(NSUInteger)sourceLevel
                                                   sourceSlice:(NSUInteger)sourceSlice
                                               sourceDepthPlane:(NSUInteger)sourceDepthPlane
                                                        reason:(const char *)reason;
- (id<MTLTexture>)depthFloatTextureForDepthStencilReadback:(id<MTLTexture>)sourceTexture
                                                    reason:(const char *)reason;
- (id<MTLRenderPipelineState>)scaledBlitPipelineForPixelFormat:(MTLPixelFormat)pixelFormat;
- (id<MTLSamplerState>)scaledBlitSamplerForFilter:(GLuint)filter;
- (id<MTLRenderPipelineState>)clearRectPipelineForColorFormat:(MTLPixelFormat)colorFormat
                                                  depthFormat:(MTLPixelFormat)depthFormat
                                                  writesColor:(BOOL)writesColor
                                                  writesDepth:(BOOL)writesDepth;
- (id<MTLDepthStencilState>)clearRectDepthState;
- (BOOL)textureCanUseGLSampledRenderTargetCopy:(Texture *)tex
                                        source:(id<MTLTexture>)source;
- (void)releaseGLSampledRenderTargetCopyForTexture:(Texture *)tex;
- (BOOL)updateGLSampledRenderTargetCopyForTexture:(Texture *)tex
                                           source:(id<MTLTexture>)source
                                           reason:(const char *)reason;
// Texture.m — called from Blit.m and MGLRenderer.m
- (bool)copyTextureUploadWithDedicatedCommandBuffer:(id<MTLBuffer>)sourceBuffer
                                        sourceOffset:(NSUInteger)sourceOffset
                                   sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
                                 sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                                           sourceSize:(MTLSize)sourceSize
                                            toTexture:(id<MTLTexture>)texture
                                     destinationSlice:(NSUInteger)destinationSlice
                                     destinationLevel:(NSUInteger)destinationLevel
                                    destinationOrigin:(MTLOrigin)destinationOrigin
                                               reason:(const char *)reason;
- (void)mtlReadDrawable:(GLMContext)glm_ctx
             pixelBytes:(void *)pixelBytes
            bytesPerRow:(NSUInteger)bytesPerRow
          bytesPerImage:(NSUInteger)bytesPerImage
             fromRegion:(MTLRegion)region;
- (void)mglApplyPendingFBODepthClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(id<MTLTexture>)texture;
- (void)mglApplyPendingFBOColorClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(id<MTLTexture>)texture
                                  attachmentEnum:(GLenum)attachmentEnum;
- (bool)uploadTextureSliceViaBlit:(id<MTLTexture>)texture
                          texName:(GLuint)texName
                         texTarget:(GLenum)texTarget
                            bytes:(const void *)bytes
                      bytesPerRow:(NSUInteger)bytesPerRow
                    bytesPerImage:(NSUInteger)bytesPerImage
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                            depth:(NSUInteger)depth
                            level:(NSUInteger)level
                            slice:(NSUInteger)slice;
- (bool)uploadFullCPUTextureDataIntoTexture:(Texture *)tex
                                      metal:(id<MTLTexture>)texture
                                     reason:(const char *)reason;

// === Methods defined in MGLRenderer.m, called from MGLRenderer+RenderPass.m ===
- (bool)mapBuffersToMTL;
- (bool)bindVertexBuffersToCurrentRenderEncoder;
- (bool)bindFragmentBuffersToCurrentRenderEncoder;
- (id<MTLTexture>)createMTLTextureFromGLTexture:(Texture *)tex;
- (id<MTLTexture>)createFallbackMTLTexture:(Texture *)tex;
- (id<MTLSamplerState>)createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target;
- (id<MTLLibrary>)compileShader:(const char *)str;
- (id<MTLFunction>)newFunctionFromLibrary:(id<MTLLibrary>)library
                                entryName:(NSString *)entryName
                                   source:(const char *)source
                                    label:(NSString *)label;
- (MTLStencilOperation)mtlStencilOpForGLOp:(GLenum)op;
- (bool)checkDrawBufferSize:(GLuint)index;
- (id)newDrawBuffer:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil;
- (id)newDrawBufferWithCustomSize:(MTLPixelFormat)pixelFormat
                     isDepthStencil:(bool)depthStencil
                        customSize:(CGSize)size;
- (int)getProgramBindingCount:(int)stage type:(int)type;
- (MTLBlendFactor)blendFactorFromGL:(GLenum)gl_blend;
- (MTLBlendOperation)blendOperationFromGL:(GLenum)gl_blend_op;
- (bool)bindActiveTexturesToMTL;
- (bool)updateDirtyBaseBufferList:(BufferMapList *)buffer_map_list;
- (bool)checkForDirtyBufferData:(BufferMapList *)buffer_map_list;
- (bool)newRenderEncoder;
- (bool)bindBufferSizeConstantsForRenderEncoder;
- (bool)currentRenderPassMatchesCurrentFramebuffer;
- (bool)bindFramebufferAttachmentTextures;

// === Methods defined in MGLRenderer.m, called from MGLRenderer+Draw.m ===
- (int)getVertexBufferIndexWithAttributeSet:(int)attribute;
- (NSUInteger)getProgramBindingRequiredSize:(int)stage type:(int)type index:(int)index;
- (NSUInteger)getProgramBindingRequiredSizeForStage:(int)stage clientBinding:(GLuint)clientBinding;
- (NSInteger)getProgramMetalBufferIndexForStage:(int)stage clientBinding:(GLuint)clientBinding;
- (id<MTLBuffer>)floatVertexBufferForDoubleAttrib:(Buffer *)sourceBuffer
                                         resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                             size:(GLuint)componentCount
                                         outStride:(NSUInteger *)outStride;
- (id<MTLBuffer>)floatVertexBufferForIntAttrib:(Buffer *)sourceBuffer
                                      resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                          size:(GLuint)componentCount
                                    normalized:(GLboolean)normalized
                                          type:(GLenum)type
                                     outStride:(NSUInteger *)outStride;
- (id<MTLBuffer>)integerVertexBufferForAttrib:(Buffer *)sourceBuffer
                                     resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                         size:(GLuint)componentCount
                                       srcType:(GLenum)srcType
                                     dstIsInt:(BOOL)dstIsInt
                                    outStride:(NSUInteger *)outStride;
- (id<MTLSamplerState>)fallbackSamplerState;
- (int)getProgramBinding:(int)stage type:(int)type index:(int)index;
- (int)getProgramGLBinding:(int)stage type:(int)type index:(int)index;
- (GLuint)textureUnitForSampledResource:(SpirvResource *)sampledResource
                            metalBinding:(GLuint)metalBinding
                                  stage:(int)stage;
- (MTLTextureType)getProgramExpectedTextureType:(int)stage type:(int)type index:(int)index;
- (MTLTextureType)getProgramDeclaredTextureType:(int)stage type:(int)type index:(int)index;
- (MGLTextureDataKind)getProgramExpectedTextureDataKind:(int)stage type:(int)type index:(int)index;
- (Texture *)textureForSampledResource:(SpirvResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(MTLTextureType)expectedType;
- (id<MTLTexture>)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
                                               dataKind:(MGLTextureDataKind)dataKind;
- (int)textureIndexForExpectedMetalType:(MTLTextureType)expectedType;
- (void)traceSampledTextureReadback:(id<MTLTexture>)texture
                              glTex:(Texture *)glTex
                              level:(TextureLevel *)level0
                            program:(GLuint)program
                            binding:(GLuint)binding
                              stage:(NSString *)stage
                             reason:(NSString *)reason
                                hit:(uint64_t)hit;
- (bool)processBuffer:(Buffer *)ptr;
- (bool)dispatchTessControlShader:(GLMContext)glm_ctx
                          program:(Program *)tcsProgram
                            first:(GLint)first
                            count:(GLsizei)count
                        indexType:(GLenum)indexType
                          indices:(const void *)indices
                       baseVertex:(GLint)baseVertex
                     instanceCount:(GLsizei)drawInstanceCount
                     baseInstance:(GLuint)baseInstance;
- (bool)dispatchTessEvaluationShader:(GLMContext)glm_ctx
                            program:(Program *)tesProgram
                              first:(GLint)first
                              count:(GLsizei)count;

/* === Additional C functions made non-static for Draw category ===
 * Formerly `static` in MGLRenderer.m. */
Program *mglTraceResolveDrawProgram(GLMContext traceCtx);
bool mglTraceShouldLogReplay(GLMContext traceCtx, Program *program);
BOOL mglRendererTextureLooksRecoverableSampled2D(GLMContext glctx,
                                                  Texture *tex,
                                                  MTLTextureType expectedType,
                                                  MGLTextureDataKind expectedKind);
BOOL mglRendererTextureLooksLikeSampledColor2D(GLMContext glctx, Texture *tex);
MTLIndexType getMTLIndexType(GLenum type);
Buffer *getElementBuffer(GLMContext ctx);
Buffer *getIndirectBuffer(GLMContext ctx);
MTLPrimitiveType getMTLPrimitiveType(GLenum mode);

/* Cull distance emulation params — formerly a static typedef in MGLRenderer.m. */
typedef struct {
    uint32_t prim_vertex_count;
    uint32_t culldist_offset;
    uint32_t vertex_stride;
    uint32_t culldist_size;
} MGLCullDistanceEmuParams;


/* === C functions made non-static for Draw category ===
 * Formerly `static` in MGLRenderer.m; declared here so
 * MGLRenderer+Draw.m can call them. */
void mglRestoreProgramPipelinePair(GLMContext ctx, GLuint programName, GLuint pipelineName);
void mglRendererSyncFramebufferBindingNames(GLMContext ctx);
Texture *mglTraceFramebufferAttachmentTexture(GLMContext glctx, FBOAttachment *attachment);
BOOL mglRendererGLSampledCopyLooksUsable(Texture *tex,
                                                MTLTextureType expectedType,
                                                MGLTextureDataKind expectedKind,
                                                BOOL allowPreviousWriteVersion,
                                                id<MTLTexture> *copyOut,
                                                BOOL *usedPreviousWriteVersionOut);
void mglLogDrawWithoutSwapWatchdog(const char *kind,
                                          uint64_t drawCall,
                                          GLMContext ctx,
                                          id<MTLCommandBuffer> commandBuffer,
                                          id<MTLRenderCommandEncoder> renderEncoder,
                                          MTLRenderPassDescriptor *renderPassDescriptor);
Texture *mglFindFramebufferColorTexturePairedWithDepth(GLMContext glctx,
                                                              Texture *depthTexture,
                                                              GLuint *fboNameOut);
BOOL mglCurrentDrawFramebufferUsesColorTexture(GLMContext glctx,
                                                      Texture *texture,
                                                      GLuint expectedFboName,
                                                      NSUInteger *attachmentIndexOut);
Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate, const char *where, NSUInteger slot);
NSUInteger mglRendererBuildCurrentVertexAttribBytes(GLMContext ctx,
                                                           GLuint attribute,
                                                           const VertexAttrib *attrib,
                                                           uint8_t bytes[16]);
void mglLogSkippedGLSampledRenderTargetCopy(GLMContext glctx,
                                                   Program *program,
                                                   Texture *tex,
                                                   const char *stage,
                                                   const char *sampledName,
                                                   GLuint binding,
                                                   GLuint textureUnit,
                                                   const char *reason);
bool mglShouldInspectDrawCall(uint64_t drawCall, GLuint programName);
void mglTraceDrawElementsAttrib(GLMContext ctx,
                                       VertexArray *vao,
                                       uint64_t drawCall,
                                       GLuint programName,
                                       const uint8_t *indexBytes,
                                       GLenum indexType,
                                       NSUInteger indexElement,
                                       GLint baseVertex,
                                       GLuint attrib,
                                       bool traceFile);
void mglTraceReplayCommandVertexAttribSamples(GLMContext traceCtx,
                                                     Program *program,
                                                     const MGLDrawCommand *cmd,
                                                     Buffer *ebo,
                                                     uint64_t flushId,
                                                     uint32_t batchIndex,
                                                     uint32_t commandIndex,
                                                     bool forceTrace);
bool mglResolvePassthroughPatchModeForContext(GLMContext drawCtx,
                                                     GLenum *mode,
                                                     const char *label);

@end

#endif /* MGLRenderer_Private_h */
