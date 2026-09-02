/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * processGLState preamble/tail/dirty-domain ObjC hooks for C++ sync facade.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+RenderPass_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+Buffer_Private.h"
#import "MGLRenderer+GPURecovery_Private.h"
#include "mgl_renderer_sync.h"
#include "mgl_renderer_pipeline.h"
#include "mgl_renderer_binding.h"
#include "mgl_render.h"
#include "mgl_buffer_slots.h"
#include "mgl_frame_activity.h"
#include "mgl_trace_strategy.h"
#include "mgl_render_pass_coordinator.h"


@interface MGLRenderer (SyncOpsBridge)
- (bool)ensureCurrentRenderPassMatchesFramebufferForDraw;
- (bool)validateRenderPassAttachmentsAndPipelineFormatsLocked:(BOOL)traceProcess;
- (void)updateCurrentRenderEncoder;
@end

@interface MGLRenderer (SyncOpsAccessors)
- (MGLCommandState *)mglSyncOpsCommandState;
- (MGLGPURecoveryState *)mglSyncOpsGPURecovery;
- (MGLPipelineCacheState *)mglSyncOpsPipelineCacheState;
- (void *)mglSyncOpsBindingStateOwner;
- (MGLResourceFallbackState *)mglSyncOpsResourceFallback;
- (GLMContext)mglSyncOpsCtx;
- (id)mglSyncOpsDevice;
- (id)mglSyncOpsCommandQueue;
- (id)mglSyncOpsDrawable;
@end

static MGLRenderTextureInfo mglSyncOpsBridgeTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

static bool mglSyncOpsBridgeGetPersistentState(
    const MGLCommandState *commandState,
    MGLRenderPassState *stateOut)
{
    return commandState && stateOut &&
           mglCmdGetRenderPassPersistentState(commandState, stateOut) == 0;
}

static bool mglSyncOpsBridgeGetPersistentAttachmentState(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MGLRenderPassAttachmentState *attachmentOut)
{
    if (!attachmentOut) return false;
    MGLRenderPassState state = {0};
    if (!mglSyncOpsBridgeGetPersistentState(commandState, &state)) return false;
    switch (attachmentKind) {
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
            if (colorIndex >= MAX_COLOR_ATTACHMENTS) return false;
            *attachmentOut = state.color[colorIndex].attachment;
            return true;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
            *attachmentOut = state.depth.attachment;
            return true;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
            *attachmentOut = state.stencil.attachment;
            return true;
        default:
            return false;
    }
}

static id mglSyncOpsBridgeAttachmentTextureFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglSyncOpsBridgeGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &attachment)) {
        return attachment.texture ? (__bridge id)attachment.texture : nil;
    }
    return nil;
}

static id mglSyncOpsBridgeColorTextureFor(
    const MGLCommandState *commandState, NSUInteger colorIndex)
{
    return mglSyncOpsBridgeAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorIndex);
}

static id mglSyncOpsBridgeDepthTextureFor(const MGLCommandState *commandState)
{
    return mglSyncOpsBridgeAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u);
}

static id mglSyncOpsBridgeStencilTextureFor(const MGLCommandState *commandState)
{
    return mglSyncOpsBridgeAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u);
}

static NSUInteger mglSyncOpsBridgeRenderTargetHeightFor(
    const MGLCommandState *commandState)
{
    MGLRenderPassState state = {0};
    if (!mglSyncOpsBridgeGetPersistentState(commandState, &state)) {
        return 0;
    }
    return (NSUInteger)state.render_target_height;
}

static bool mglProcessGLStatePreambleBridgeEnsureMetal(void *renderer)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    static int corruption_recovery_count = 0;
    static const int max_recovery_attempts = 3;

    if ([self mglSyncOpsDevice] && [self mglSyncOpsCommandQueue] &&
        (uintptr_t)[self mglSyncOpsDevice] >= 0x1000 &&
        (uintptr_t)[self mglSyncOpsCommandQueue] >= 0x1000) {
        return true;
    }

    NSLog(@"MGL CRITICAL: Metal state corruption detected in processGLState!");
    NSLog(@"MGL CRITICAL: device=0x%lx, queue=0x%lx",
          (uintptr_t)[self mglSyncOpsDevice], (uintptr_t)[self mglSyncOpsCommandQueue]);

    if (corruption_recovery_count >= max_recovery_attempts) {
        NSLog(@"MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations");
        return false;
    }

    NSLog(@"MGL CRITICAL: Attempting Metal state recovery (%d/%d)",
          corruption_recovery_count + 1, max_recovery_attempts);
    @try {
        [self emergencyResetMetalState];
        corruption_recovery_count++;
        if (![self mglSyncOpsDevice] || ![self mglSyncOpsCommandQueue]) {
            NSLog(@"MGL CRITICAL: Metal recovery failed, aborting operation");
            return false;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: Metal recovery failed: %@", exception);
        return false;
    }
    return true;
}

static void mglProcessGLStatePreambleBridgeRejectDrawNoVao(void *renderer,
                                                           GLMContext context)
{
    (void)renderer;
    (void)context;
    NSLog(@"Error: No VAO defined for ctx\n");
}

static void mglProcessGLStatePreambleBridgeDrawBegin(
    void *renderer, GLMContext context, MGLCommandState *command_state)
{
    (void)renderer;
    (void)context;
    mglCmdSetCurrentDrawUsesRTSampledCopy(command_state, NO);
    MGL_FRAME_INC(g_mglProcessDrawCallsSinceSwap);
}

static void mglProcessGLStatePreambleBridgeEndRenderPassNonDraw(
    void *renderer, uint64_t process_call)
{
    [(__bridge MGLRenderer *)renderer
        endRenderPassIfFramebufferChangedForNonDraw:process_call];
}

int mglProcessGLStatePreambleBridgeHandleNullVao(void *renderer,
                                                        GLMContext context,
                                                        int draw_command)
{
    (void)draw_command;
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    GLMState *state = MGL_STATE(context);
    if (!(state->dirty_bits & DIRTY_STATE)) {
        return MGL_PREAMBLE_DONE_OK;
    }
    [self endRenderEncodingLocked];
    if (![self validateMetalObjects]) {
        NSLog(@"MGL WARNING: GPU throttling active - deferring render encoder creation");
        state->dirty_bits &= ~DIRTY_STATE;
        return MGL_PREAMBLE_DONE_OK;
    }
    @try {
        [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_CLEAR];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Render encoder creation failed: %@", exception);
    }
    state->dirty_bits &= ~DIRTY_STATE;
    return MGL_PREAMBLE_DONE_OK;
}

static bool mglProcessGLStatePreambleBridgeCheckQuarantine(void *renderer,
                                                           GLMContext context)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    GLuint blockedProgramKey = mglCurrentRenderProgramKey(context);
    if (blockedProgramKey == 0u ||
        [self mglSyncOpsGPURecovery]->interfaceMismatchBlockedProgram == 0 ||
        blockedProgramKey != [self mglSyncOpsGPURecovery]->interfaceMismatchBlockedProgram) {
        return true;
    }
    CFTimeInterval now = CFAbsoluteTimeGetCurrent();
    if (now >= [self mglSyncOpsGPURecovery]->interfaceMismatchBlockedUntil) {
        return true;
    }
    static uint64_t s_quarantineSkipCount = 0;
    s_quarantineSkipCount++;
    if (s_quarantineSkipCount <= 16 || (s_quarantineSkipCount % 1000) == 0) {
        double remaining =
            [self mglSyncOpsGPURecovery]->interfaceMismatchBlockedUntil - now;
        if (remaining < 0.0) {
            remaining = 0.0;
        }
        NSLog(@"MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw",
              (unsigned)[self mglSyncOpsGPURecovery]->interfaceMismatchBlockedProgram,
              remaining);
    }
    return false;
}

static bool mglProcessGLStatePreambleBridgeRotateCommandBuffer(
    void *renderer, GLMContext context, int trace_process)
{
    (void)context;
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    static uint64_t s_rotateFinalizedCount = 0;
    uint64_t rotateHit = ++s_rotateFinalizedCount;
    if (rotateHit <= 16ull || (rotateHit % 500ull) == 0ull) {
        NSLog(@"MGL INFO: processGLState rotating finalized command buffer hit=%llu",
              (unsigned long long)rotateHit);
    }
    if ([self newCommandBufferLocked]) {
        return true;
    }
    NSLog(@"MGL ERROR: processGLState failed to create a fresh command buffer");
    if (trace_process) {
        mglLogStateSnapshot("processGLState.fail.new_cb_rotate",
                            [self mglSyncOpsCtx],
                            [self mglSyncOpsCommandState]->currentCommandBufferOwner,
                            [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
                            [self mglSyncOpsCommandState]->renderPassStateOwner,
                            [self mglSyncOpsDrawable]);
    }
    return false;
}

static bool mglProcessGLStatePreambleBridgeCreateCommandBuffer(
    void *renderer, GLMContext context, int trace_process)
{
    (void)context;
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    if (kMGLVerboseFrameLoopLogs) {
        NSLog(@"MGL INFO: processGLState found NULL command buffer, creating one");
    }
    if ([self newCommandBufferLocked]) {
        return true;
    }
    NSLog(@"MGL ERROR: processGLState could not create initial command buffer");
    if (trace_process) {
        mglLogStateSnapshot("processGLState.fail.new_cb_initial",
                            [self mglSyncOpsCtx],
                            [self mglSyncOpsCommandState]->currentCommandBufferOwner,
                            [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
                            [self mglSyncOpsCommandState]->renderPassStateOwner,
                            [self mglSyncOpsDrawable]);
    }
    return false;
}

int mglProcessGLStatePreambleBridge(MGLRenderer *self, bool draw_command,
                                           uint64_t process_call,
                                           bool trace_process)
{
    static const MGLProcessGLStatePreambleOps kPreambleOpsTemplate = {
        .ensure_metal_objects_ready =
            mglProcessGLStatePreambleBridgeEnsureMetal,
        .reject_draw_without_vao =
            mglProcessGLStatePreambleBridgeRejectDrawNoVao,
        .on_draw_command_begin = mglProcessGLStatePreambleBridgeDrawBegin,
        .end_render_pass_non_draw =
            mglProcessGLStatePreambleBridgeEndRenderPassNonDraw,
        .handle_null_vao_path = mglProcessGLStatePreambleBridgeHandleNullVao,
        .check_program_quarantine =
            mglProcessGLStatePreambleBridgeCheckQuarantine,
        .rotate_finalized_command_buffer =
            mglProcessGLStatePreambleBridgeRotateCommandBuffer,
        .create_initial_command_buffer =
            mglProcessGLStatePreambleBridgeCreateCommandBuffer,
    };
    MGLProcessGLStatePreambleOps preambleOps = kPreambleOpsTemplate;
    preambleOps.renderer = (__bridge void *)self;
    return mglRenderProcessGLStatePreamble(
        [self mglSyncOpsCtx], [self mglSyncOpsCommandState], draw_command ? 1 : 0, process_call,
        trace_process ? 1 : 0, &preambleOps);
}

bool mglProcessGLStateTailBridgeRecoverNilEncoder(void *renderer,
                                                         GLMContext context)
{
    (void)context;
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    static uint64_t s_nilEncoderRecoveryCount = 0;
    uint64_t nilHit = ++s_nilEncoderRecoveryCount;
    if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
        NSLog(@"MGL WARNING: processGLState - current render encoder is nil, attempting recovery hit=%llu",
              (unsigned long long)nilHit);
        mglLogRenderPassLifecycle("nil-encoder-before-recovery",
                                  nilHit,
                                  [self mglSyncOpsCtx],
                                  [self mglSyncOpsCommandState]->currentCommandBufferOwner,
                                  [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
                                  [self mglSyncOpsCommandState]->renderPassStateOwner,
                                  [self mglSyncOpsDrawable],
                                  [self mglSyncOpsCommandState]->renderPassFramebuffer,
                                  [self mglSyncOpsCommandState]->renderPassFramebufferName,
                                  [self mglSyncOpsCommandState]->renderPassDrawBuffer,
                                  [self mglSyncOpsCommandState]->renderPassDrawBufferCount);
    }
    if (![self newRenderEncoderLockedWithReason:MGL_ENC_REASON_NIL]) {
        return false;
    }
    if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
        mglLogRenderPassLifecycle("nil-encoder-after-recovery",
                                  nilHit,
                                  [self mglSyncOpsCtx],
                                  [self mglSyncOpsCommandState]->currentCommandBufferOwner,
                                  [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
                                  [self mglSyncOpsCommandState]->renderPassStateOwner,
                                  [self mglSyncOpsDrawable],
                                  [self mglSyncOpsCommandState]->renderPassFramebuffer,
                                  [self mglSyncOpsCommandState]->renderPassFramebufferName,
                                  [self mglSyncOpsCommandState]->renderPassDrawBuffer,
                                  [self mglSyncOpsCommandState]->renderPassDrawBufferCount);
    }
    return true;
}

bool mglProcessGLStateTailBridgePrepareDrawPass(void *renderer,
                                                       GLMContext context)
{
    (void)context;
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    if (![self ensureCurrentRenderPassMatchesFramebufferForDraw]) {
        return false;
    }
    [self updateCurrentRenderEncoder];
    return true;
}

static void mglProcessGLStateTailBridgeLogDrawPipelineLookup(void *renderer,
                                                             GLMContext context)
{
    if (!kMGLVerbosePipelineLogs) {
        return;
    }
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    static uint64_t s_drawPipelineLookupCount = 0;
    s_drawPipelineLookupCount++;
    if (s_drawPipelineLookupCount > 256ull &&
        (s_drawPipelineLookupCount % 1000ull) != 0ull) {
        return;
    }
    Program *lookupProgram = mglResolveProgramFromState(context);
    Program *lookupVertexProgram =
        mglResolveProgramForStageFromState(context, _VERTEX_SHADER);
    Program *lookupFragmentProgram =
        mglResolveProgramForStageFromState(context, _FRAGMENT_SHADER);
    GLuint lookupProgramName = mglCurrentRenderProgramKey(context);
    Framebuffer *lookupFBO = MGL_STATE(context)->framebuffer;
    GLuint lookupFBOName = lookupFBO ? lookupFBO->name : 0;
    fprintf(stderr, "MGL Draw current program key=%u mono=%p vs=%u fs=%u\n",
            (unsigned)lookupProgramName, (void *)lookupProgram,
            lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
            lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u);
    NSLog(@"MGL DRAW pipeline lookup result=%p key=%u vs=%u fs=%u vao=%p fbo=%u",
          [self mglSyncOpsPipelineCacheState]->pipelineState,
          (unsigned)lookupProgramName,
          lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
          lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u,
          MGL_STATE(context)->vao, (unsigned)lookupFBOName);
}

bool mglProcessGLStateTailBridgeEnsurePipelineReady(
    void *renderer, GLMContext context, int trace_process)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    if ([self mglSyncOpsPipelineCacheState]->pipelineState) {
        return true;
    }
    static uint64_t nil_pipeline_count = 0;
    nil_pipeline_count++;
    if (nil_pipeline_count <= 8 || (nil_pipeline_count % 1000) == 0) {
        mglTraceLogNSString(
            @"MGL DRAW SKIP: pipelineState is nil, forcing rebuild (occurrence=%llu)",
            (unsigned long long)nil_pipeline_count);
    }
    mglMarkRendererDirtyBits(context->active_state,
                             DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                 DIRTY_RENDER_STATE);
    if (trace_process) {
        mglLogStateSnapshot("processGLState.fail.nil_pipeline",
                            [self mglSyncOpsCtx],
                            [self mglSyncOpsCommandState]->currentCommandBufferOwner,
                            [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
                            [self mglSyncOpsCommandState]->renderPassStateOwner,
                            [self mglSyncOpsDrawable]);
    }
    return false;
}

bool mglProcessGLStateTailBridgeValidateRenderPass(
    void *renderer, GLMContext context, int trace_process)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer
        validateRenderPassAttachmentsAndPipelineFormatsLocked:trace_process];
}

bool mglProcessGLStateTailBridgeBindPipeline(void *renderer,
                                                    GLMContext context,
                                                    int trace_process)
{
    (void)context;
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    @try {
        if (mglRenderBindingSetPipelineIfNeededForOwner(
                [self mglSyncOpsBindingStateOwner],
                [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
                [self mglSyncOpsPipelineCacheState]->pipelineState) > 0) {
            MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: processGLState - setRenderPipelineState failed: %@",
              exception.reason);
        mglMarkRendererDirtyBits([self mglSyncOpsCtx]->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        if (trace_process) {
            mglLogStateSnapshot("processGLState.fail.set_pipeline",
                                [self mglSyncOpsCtx],
                                [self mglSyncOpsCommandState]->currentCommandBufferOwner,
                                [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
                                [self mglSyncOpsCommandState]->renderPassStateOwner,
                                [self mglSyncOpsDrawable]);
        }
        return false;
    }
    return true;
}

bool mglProcessGLStateTailBridgeApplyPostBindDrawState(void *renderer,
                                                              GLMContext context)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    Program *fragmentProgram =
        mglResolveProgramForStageFromState(context, _FRAGMENT_SHADER);
    if (fragmentProgram && fragmentProgram->usesFragCoordParams == GL_TRUE) {
        NSUInteger passHeight =
            mglSyncOpsBridgeRenderTargetHeightFor([self mglSyncOpsCommandState]);
        if (passHeight == 0) {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS && passHeight == 0; i++) {
                id color = mglSyncOpsBridgeColorTextureFor([self mglSyncOpsCommandState], i);
                passHeight =
                    color ? mglSyncOpsBridgeTextureInfo(color).height : 0;
            }
            if (passHeight == 0 &&
                mglSyncOpsBridgeDepthTextureFor([self mglSyncOpsCommandState])) {
                passHeight = mglSyncOpsBridgeTextureInfo(
                    mglSyncOpsBridgeDepthTextureFor([self mglSyncOpsCommandState])).height;
            }
            if (passHeight == 0 &&
                mglSyncOpsBridgeStencilTextureFor([self mglSyncOpsCommandState])) {
                passHeight = mglSyncOpsBridgeTextureInfo(
                    mglSyncOpsBridgeStencilTextureFor([self mglSyncOpsCommandState])).height;
            }
        }
        vector_float4 fragCoordParams = {
            (float)passHeight,
            MGL_STATE(context)->var.clip_origin == GL_LOWER_LEFT ? 1.0f
                                                                 : 0.0f,
            0.0f,
            0.0f};
        mglRenderSetRenderBytesForOwner(
            [self mglSyncOpsCommandState]->currentRenderEncoderOwner, &fragCoordParams,
            sizeof(fragCoordParams), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLFragCoordParamsBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:
                  kMGLFragCoordParamsBufferIndex];
    }
    if (fragmentProgram && fragmentProgram->uses_lod_bias == GL_TRUE) {
        const GLfloat biasmax = context->state.var.max_texture_lod_bias;
        float lodBiasArr[TEXTURE_UNITS];
        for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
            Texture *tex = MGL_STATE(context)->active_textures[unit];
            Sampler *smp = MGL_STATE(context)->texture_samplers[unit];
            float bias =
                smp ? smp->params.lod_bias
                    : (tex ? tex->params.lod_bias : 0.0f);
            if (biasmax > 0.0f) {
                if (bias > biasmax) {
                    bias = biasmax;
                } else if (bias < -biasmax) {
                    bias = -biasmax;
                }
            }
            lodBiasArr[unit] = bias;
        }
        mglRenderSetRenderBytesForOwner(
            [self mglSyncOpsCommandState]->currentRenderEncoderOwner, lodBiasArr,
            sizeof(lodBiasArr), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLLodBiasBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLLodBiasBufferIndex];
        mglRenderSetRenderBytesForOwner(
            [self mglSyncOpsCommandState]->currentRenderEncoderOwner, &biasmax,
            sizeof(biasmax), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLLodBiasMaxBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLLodBiasMaxBufferIndex];
    }
    if (mglFragmentTextureTraceBindingsUseRTSampledCopy(
            [self mglSyncOpsResourceFallback]->fragmentTextureTraceBindings,
            TEXTURE_UNITS)) {
        mglCmdSetCurrentDrawUsesRTSampledCopy([self mglSyncOpsCommandState], YES);
        [self updateCurrentRenderEncoder];
    }
    return true;
}

bool mglProcessGLStateTailBridge(MGLRenderer *self, bool draw_command,
                                        bool trace_process,
                                        MGLResourceSyncWork *resource_sync_work)
{
    static const MGLProcessGLStateTailOps kTailOpsTemplate = {
        .recover_nil_render_encoder =
            mglProcessGLStateTailBridgeRecoverNilEncoder,
        .prepare_draw_pass = mglProcessGLStateTailBridgePrepareDrawPass,
        .log_draw_pipeline_lookup =
            mglProcessGLStateTailBridgeLogDrawPipelineLookup,
        .ensure_pipeline_ready = mglProcessGLStateTailBridgeEnsurePipelineReady,
        .validate_render_pass = mglProcessGLStateTailBridgeValidateRenderPass,
        .bind_pipeline = mglProcessGLStateTailBridgeBindPipeline,
        .apply_post_bind_draw_state =
            mglProcessGLStateTailBridgeApplyPostBindDrawState,
    };
    MGLProcessGLStateTailOps tailOps = kTailOpsTemplate;
    tailOps.renderer = (__bridge void *)self;
    return mglRenderProcessGLStateTail(
               [self mglSyncOpsCtx], [self mglSyncOpsCommandState], draw_command ? 1 : 0,
               trace_process ? 1 : 0, resource_sync_work, &tailOps) != 0;
}

/*
 * Dirty state domain processing - orchestration lives in mgl_renderer_sync.cpp;
 * ObjC hooks implement platform-specific steps.
 */
static Framebuffer *mglSyncBridgeGetValidatedFramebuffer(void *renderer,
                                                         GLMContext context,
                                                         const char *where);
static bool mglSyncBridgeRenderPassMatchesFramebuffer(void *renderer,
                                                      GLMContext context);
static bool mglSyncBridgeBindFramebufferAttachments(void *renderer,
                                                    GLMContext context);
static bool mglSyncBridgeRotateRenderEncoderForFbo(void *renderer,
                                                   GLMContext context);

bool mglSyncBridgeSyncFbo(void *renderer, GLMContext context)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    static const MGLRenderPassSyncOps kPassOpsTemplate = {
        .get_validated_framebuffer = mglSyncBridgeGetValidatedFramebuffer,
        .render_pass_matches_framebuffer =
            mglSyncBridgeRenderPassMatchesFramebuffer,
        .bind_framebuffer_attachment_textures =
            mglSyncBridgeBindFramebufferAttachments,
        .rotate_render_encoder_for_fbo =
            mglSyncBridgeRotateRenderEncoderForFbo,
    };
    MGLRenderPassSyncOps passOps = kPassOpsTemplate;
    passOps.renderer = renderer;
    return mglRenderSyncRenderPassForFbo(context, [self mglSyncOpsCommandState],
                                         &passOps) != 0;
}

static Framebuffer *mglSyncBridgeGetValidatedFramebuffer(void *renderer,
                                                         GLMContext context,
                                                         const char *where)
{
    (void)renderer;
    return mglRendererGetValidatedFramebuffer(context, where);
}

static bool mglSyncBridgeRenderPassMatchesFramebuffer(void *renderer,
                                                      GLMContext context)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer
        currentRenderPassMatchesCurrentFramebuffer];
}

static bool mglSyncBridgeBindFramebufferAttachments(void *renderer,
                                                    GLMContext context)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer bindFramebufferAttachmentTextures];
}

static bool mglSyncBridgeRotateRenderEncoderForFbo(void *renderer,
                                                   GLMContext context)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer
        rotateRenderEncoderForCurrentFramebufferLocked];
}

static bool mglSyncBridgeBindFramebufferInStateBlock(void *renderer,
                                                       GLMContext context)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    Framebuffer *framebuffer =
        mglRendererGetValidatedFramebuffer(context,
                                           "processGLState.dirtyStateFBO");
    if (!framebuffer) {
        return true;
    }
    if (!(framebuffer->dirty_bits & DIRTY_FBO_BINDING)) {
        return true;
    }
    if (![self bindFramebufferAttachmentTextures]) {
        return false;
    }
    framebuffer = mglRendererGetValidatedFramebuffer(
        context, "processGLState.dirtyStateFBO.afterBind");
    if (framebuffer) {
        framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
    }
    return true;
}

static bool mglSyncBridgeShouldDeferBufferMap(void *renderer,
                                              GLMContext context,
                                              int draw_command)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    if (!draw_command || [self mglSyncOpsPipelineCacheState]->pipelineState != NULL ||
        !(MGL_STATE(context)->dirty_bits & DIRTY_PROGRAM)) {
        return false;
    }
    static uint64_t s_deferredMapCount = 0;
    s_deferredMapCount++;
    if (s_deferredMapCount <= 16 || (s_deferredMapCount % 1000ull) == 0ull) {
        mglTraceLogNSString(
            @"MGL DRAW SKIP: pipelineState is nil (deferring buffer mapping, "
            @"occurrence=%llu)",
            (unsigned long long)s_deferredMapCount);
    }
    return true;
}

static bool mglSyncBridgeMapBuffers(void *renderer, GLMContext context)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer mapBuffersToMTL];
}

static bool mglSyncBridgeBindActiveTextures(void *renderer, GLMContext context)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer bindActiveTexturesToMTL];
}

static bool mglSyncBridgeUpdateBaseBufferLists(void *renderer,
                                               GLMContext context)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    GLMState *state = MGL_STATE(context);
    if (![self updateDirtyBaseBufferList:&state->vertex_buffer_map_list]) {
        return false;
    }
    return [self updateDirtyBaseBufferList:&state->fragment_buffer_map_list];
}

static bool mglSyncBridgeEnsureRenderEncoder(void *renderer,
                                             GLMContext context,
                                             MGLEncoderCreateReason reason)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer
        newRenderEncoderLockedWithReason:reason];
}

static bool mglSyncBridgeUpdateRenderEncoder(void *renderer, GLMContext context)
{
    (void)context;
    [(__bridge MGLRenderer *)renderer updateCurrentRenderEncoder];
    return true;
}

static bool mglSyncBridgeSyncPipeline(void *renderer, GLMContext context,
                                      int deferred_buffer_map)
{
    (void)renderer;
    return mglRenderSyncPipeline(context, deferred_buffer_map) != 0;
}

static bool mglSyncBridgeIncidentalBufferData(void *renderer, GLMContext context)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    GLMState *state = MGL_STATE(context);
    MGLEncodeContext encCtx = {
        .render_encoder_owner = [self mglSyncOpsCommandState]->currentRenderEncoderOwner,
    };

    if ([self checkForDirtyBufferData:&state->vertex_buffer_map_list]) {
        if (![self updateDirtyBaseBufferList:&state->vertex_buffer_map_list]) {
            return false;
        }
        if (![self bindVertexBuffersToCurrentRenderEncoder:&encCtx]) {
            return false;
        }
    }

    if ([self checkForDirtyBufferData:&state->fragment_buffer_map_list]) {
        if (![self updateDirtyBaseBufferList:&state->fragment_buffer_map_list]) {
            return false;
        }
        if (![self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]) {
            return false;
        }
    }
    return true;
}

bool mglProcessDirtyStateDomainsBridge(MGLRenderer *self, bool draw_command,
                                      MGLResourceSyncWork *work)
{
    static const MGLRendererSyncOps kSyncOpsTemplate = {
        .sync_render_pass_for_fbo = mglSyncBridgeSyncFbo,
        .bind_framebuffer_attachments_in_state_block =
            mglSyncBridgeBindFramebufferInStateBlock,
        .should_defer_buffer_map = mglSyncBridgeShouldDeferBufferMap,
        .map_buffers = mglSyncBridgeMapBuffers,
        .bind_active_textures = mglSyncBridgeBindActiveTextures,
        .update_base_buffer_lists = mglSyncBridgeUpdateBaseBufferLists,
        .ensure_render_encoder = mglSyncBridgeEnsureRenderEncoder,
        .update_render_encoder = mglSyncBridgeUpdateRenderEncoder,
        .sync_pipeline = mglSyncBridgeSyncPipeline,
        .sync_incidental_buffer_data = mglSyncBridgeIncidentalBufferData,
    };
    MGLRendererSyncOps ops = kSyncOpsTemplate;
    ops.renderer = (__bridge void *)self;
    return mglRenderProcessDirtyStateDomains(
               [self mglSyncOpsCtx], MGL_SYNC_DOMAIN_ALL, draw_command ? 1 : 0,
               [self mglSyncOpsCommandState], work, &ops) != 0;
}

@implementation MGLRenderer (SyncOpsBridge)

- (MGLCommandState *)mglSyncOpsCommandState
{
    return &_commandState;
}

- (MGLGPURecoveryState *)mglSyncOpsGPURecovery
{
    return &_gpuRecovery;
}

- (MGLPipelineCacheState *)mglSyncOpsPipelineCacheState
{
    return &_pipelineCacheState;
}

- (void *)mglSyncOpsBindingStateOwner
{
    return _bindingStateOwner;
}

- (MGLResourceFallbackState *)mglSyncOpsResourceFallback
{
    return &_resourceFallback;
}

- (GLMContext)mglSyncOpsCtx
{
    return ctx;
}

- (id)mglSyncOpsDevice
{
    return _device;
}

- (id)mglSyncOpsCommandQueue
{
    return _commandQueue;
}

- (id)mglSyncOpsDrawable
{
    return _drawable;
}

@end
