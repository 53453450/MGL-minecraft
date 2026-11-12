/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_manager_ops.c — C twins of the MGLRenderPassManager methods
 * that only touch the command state.  The manager keeps calling its own
 * methods; C callers use these.
 */

#include "mgl_render_pass_sync_ops.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_pass_ops_shared.h"
#include "mgl_state_apply.h"
#include "mgl_rt_sync.h"   /* mglRenderTargetMarkGPUAuthoritative */
#include "mgl_render_encoder_ops.h"
#include "mgl_attachment_binding.h"   /* mglRendererBindFramebufferAttachmentTextures */
#include "mgl_buffer_map.h"           /* map / dirty base buffers */
#include "mgl_texture_bind.h"          /* mglRendererBindMTLTexture */
#include "mgl_draw_buffer.h"          /* Metal draw-buffer mapping */
#include "mgl_sync.h"                 /* MGLMetalAttachmentSubresource */
#include "mgl_renderer_backend.h"     /* default draw buffer attachments */
#include "mgl_renderer_binding_ops.h"  /* mglRendererBindActiveTexturesToMTL */
#include "mgl_stage_encode_drivers.h" /* stage encode bind drivers */
#include "mgl_frame_activity.h"     /* MGL_ENC_REASON_* */
#include "mgl_render_api_query.h"  /* MGLRenderPipelineDescriptorState */
#include "mgl_render_api_lifecycle.h" /* mglRenderLoadAIRMainFunction */
#include "mgl_byte_hash.h"         /* mglHashStepU64 */
#include "mgl_draw_tess.h"         /* mglTessRasterGLMode */
#include "mgl_vertex_layout.h"     /* mglRendererGenerateVertexDescriptorState */
#include "mgl_render_pass_manager.h" /* pass-manager transaction entries */
#include "mgl_render_pass_clear.h"   /* mglRenderPassPlanClearValues */
#include "mgl_program_resource.h"   /* mglProgramStageBuiltinMask */
#include "mgl_shader_abi.h"         /* mglAIRPerVertexStrideForResources, mglShaderCompileGLSL */
#include "mgl_buffer_slots.h"       /* mglProgramStageHasSamplers (shared with the link gate) */
#include "mgl_metal_ref.h"         /* mglReleaseMetalObjNoNull */
#include "mgl_draw_gs.h"           /* mglDrawGsPassthroughDeclType */
#include "mgl_render_pass_plan.h"    /* mglRenderProcessGLState*, plans */
#include "mgl_buffer_slots.h"        /* kMGL*BufferIndex */
#include "mgl_renderer_binding_ops.h"  /* mglRendererSyncResourceBindingsForContext */
#include "mgl_swap_diagnostics.h"     /* swap colour copy + diagnostics */
#include <dispatch/dispatch.h>       /* dispatch_async_f */
#include "mgl_trace_log.h"         /* mglTraceLog, kMGLDiagnosticStateLogs */
#include "mgl_gpu_recovery.h"      /* mglRendererRecordGPUError */
#include "mgl_pso_format_class.h"  /* mglRenderDefaultColorPixelFormat */
#include "mgl_render.h"           /* attachment kinds, MS plane adjust */
#include "mgl_texture_compat.h"   /* mglMetalTextureLevelDimension */

#include "mgl_pso_build_ops.h"    /* mglRenderPassSyncPipelineState */
#include "mgl_renderer_ports.h"
#include "mgl_renderer_binding_ops.h"      /* mglRendererBindingInvalidateLastBoundState */
#include "mgl_trace_strategy.h"         /* mglClearFragmentTraceBindingsForRenderer */

#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <stdarg.h>   /* va_list for the GLSL source builder */
#include <stdlib.h>   /* malloc / realloc / free */
#include <string.h>   /* memcpy / strlen */
#include <math.h>     /* lrintf */
#include "mgl_readback_policy.h"
#include "mgl_program_backend.h"
#include "mgl_texture_backend.h"
#include "mgl_raster_orientation_state.h"
#include "mgl_sparse_residency.h"
#include "mgl_texture_binding_resolve.h"
#include "mgl_render_api_buffer.h"
#include "mgl_render_api_command.h"
#include "mgl_ports_present.h"
#include "mgl_ms_sample_loop.h"            /* post-depth coverage inputs */

extern void mglLogRenderPassLifecycle(const char *tag, uint64_t call,
                                      GLMContext ctx, void *commandBufferOwner,
                                      void *renderEncoderOwner,
                                      void *renderPassStateOwner, void *drawable,
                                      Framebuffer *renderPassFramebuffer,
                                      GLuint renderPassFramebufferName,
                                      GLenum renderPassDrawBuffer,
                                      GLsizei renderPassDrawBufferCount);
extern GLuint mglRendererSafeFramebufferName(GLMContext ctx);
extern void mglLogStateSnapshot(const char *tag, GLMContext ctx,
                                void *commandBufferOwner,
                                void *renderEncoderOwner,
                                void *renderPassStateOwner, void *drawable);
extern void *mglPlatformShellDrawable(void *renderer);
extern void mglPlatformShellSetDrawable(void *renderer, void *drawable);
extern MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);
extern void mglRendererNextDrawable(void *renderer);
extern void mglRendererPrefetchNextDrawable(void *renderer);
extern int mglPlatformShellGuardedCall(void *renderer, const char *what,
                                       int (*body)(void *));
extern int mglPlatformShellMSSampleInLoop(void *renderer);
static const int kMglPdVerbosePipelineLogs = 0;
enum { kMGLLodBiasPackHeaderFloats = 4 };

void mglRenderPassInvalidateCurrentPipelineState(void *renderer,
                                                 const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    const MGLPipelineCachePort cache = mglRendererPipelineCachePort(renderer);
    if (cache.state && cache.state->pipelineState) {
        static uint64_t s_pipelineInvalidateCount = 0;
        const uint64_t hit = ++s_pipelineInvalidateCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: Invalidating current pipeline state after %s hit=%llu\n",
                    reason ? reason : "pipeline failure",
                    (unsigned long long)hit);
        }
    }
    if (cache.invalidate) {
        cache.invalidate(cache.object);
    }
}

/* -ensureCurrentRenderPassMatchesFramebufferForDraw. */
int mglRenderPassEnsureCurrentRenderPassMatchesFramebufferForDraw(void *renderer)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;

    if (!ctx) {
        return 1;
    }

    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) != 1) {
        return 1;
    }

    if (mglRenderPassMatchesCurrentFramebuffer(renderer)) {
        return 1;
    }

    static uint64_t s_fboPassMismatchCount = 0;
    const uint64_t hit = ++s_fboPassMismatchCount;
    if (hit <= 32ull || (hit % 256ull) == 0ull) {
        Framebuffer *fbo = mglPdStateView(&areas)->framebuffer;
        void *color0 = mglPdColorTextureFor(commandState, 0);
        const GLuint mglDefaultDrawbuffer =
            fbo ? 0u
                : mglDefaultDrawBufferIndexForGL(mglPdStateView(&areas)->draw_buffer);
        void *expectedDefaultColor0 = NULL;
        if (!fbo) {
            expectedDefaultColor0 =
                mglRenderDefaultDrawBufferIsFront(mglDefaultDrawbuffer)
                    ? mglRendererDrawableTexture(renderer)
                    : (mglRenderDefaultDrawBufferIsOffscreen(
                           mglDefaultDrawbuffer, _MAX_DRAW_BUFFERS)
                           ? mglRenderPassDefaultDrawBufferAttachment(
                                 areas.backend, mglDefaultDrawbuffer,
                                 MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR)
                           : NULL);
        }
        const GLuint fboName = fbo ? fbo->name : 0u;
        const GLuint attachment0Name =
            (fbo && (fbo->color_attachment_bitfield & 1u))
                ? fbo->color_attachments[0].texture
                : 0u;
        fprintf(stderr,
                "MGL WARNING: render pass/FBO mismatch before draw hit=%llu fbo=%u drawBuffer=0x%x attachment0=%u passColor0=%p expectedDefaultColor0=%p defaultDrawBuffer=%u; rebuilding encoder\n",
                (unsigned long long)hit, (unsigned)fboName,
                (unsigned)(ctx ? mglPdStateView(&areas)->draw_buffer : 0u),
                (unsigned)attachment0Name, color0, expectedDefaultColor0,
                (unsigned)mglDefaultDrawbuffer);
        mglLogRenderPassLifecycle(
            fbo ? "fbo-mismatch-before-rebuild"
                : "default-fbo-mismatch-before-rebuild",
            hit, ctx, commandState->currentCommandBufferOwner,
            commandState->currentRenderEncoderOwner,
            commandState->renderPassStateOwner, areas.drawable,
            commandState->renderPassFramebuffer,
            commandState->renderPassFramebufferName,
            commandState->renderPassDrawBuffer,
            commandState->renderPassDrawBufferCount);
    }

    mglRendererEndRenderEncodingLocked(renderer);
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE |
                                 DIRTY_VAO);
    return mglRenderPassNewRenderEncoderLockedWithReason(
        renderer, MGL_ENC_REASON_FBO);
}

/* -emergencyResetMetalState. */
typedef struct MglPdEmergencyResetCtx_t {
    int result;
} MglPdEmergencyResetCtx;

static int mglPdEmergencyResetTryBody(void *renderer, void *rawCtx)
{
    MglPdEmergencyResetCtx *ctx = (MglPdEmergencyResetCtx *)rawCtx;
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLRenderPassManager *manager = areas.render_pass_manager;

    /* Force cleanup of all Metal objects */
    mglRendererEndRenderEncodingLocked(renderer);

    mglPassManagerDiscardCurrentCommandBuffer(manager);
    mglPassManagerClearCurrentRenderEncoder(manager);
    mglPlatformShellSetDrawable(renderer, NULL);

    /* Re-initialize basic Metal objects */
    if (mglRendererBackendGetDevice(areas.backend) &&
        mglRendererBackendGetCommandQueue(areas.backend)) {
        fprintf(stderr, "MGL CRITICAL: Re-creating Metal command buffer\n");
        (void)mglPassManagerInstallNewCommandBufferFromQueue(
            manager, mglRendererBackendGetCommandQueue(areas.backend));

        if (mglRenderCommandBufferOwnerHasCurrent(
                areas.command->currentCommandBufferOwner) != 1) {
            fprintf(stderr,
                    "MGL CRITICAL: Failed to create new command buffer during recovery\n");
        }
    }
    ctx->result = 1;
    return 1;
}

void mglRenderPassEmergencyResetMetalState(void *renderer)
{
    fprintf(stderr, "MGL CRITICAL: Performing emergency Metal state reset\n");

    MglPdEmergencyResetCtx ctx = {0};
    /* @catch (NSException *exception) only logs and returns. */
    (void)mglPlatformShellGuardedCallCtx(renderer, "emergency metal reset",
                                         mglPdEmergencyResetTryBody, &ctx,
                                         NULL);
}

/* -validateRenderPassAttachmentsAndPipelineFormatsLocked:. */
int mglRenderPassValidateAttachmentsAndPipelineFormats(void *renderer,
                                                       int traceProcess)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;

    /* Guard against invalid render pass state before binding pipeline.  Metal
     * debug validation can abort the process if the encoder/render pass is
     * incompatible. */
    const int hasRenderPassState =
        commandState->renderPassStateOwner != NULL;
    if (!hasRenderPassState) {
        fprintf(stderr,
                "MGL ERROR: processGLState - render pass state owner is nil before pipeline bind\n");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_rpd", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }
    int passHasAnyAttachment = 0;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        void *colorAttachment = mglPdColorTextureFor(commandState, (size_t)i);
        if (colorAttachment) {
            passHasAnyAttachment = 1;
            if ((mglPdTextureInfo(colorAttachment).usage &
                 MGLTextureUsageRenderTarget) == 0) {
                fprintf(stderr,
                        "MGL WARNING: processGLState - color attachment %d missing RenderTarget usage (usage=0x%lx); skipping draw\n",
                        i,
                        (unsigned long)mglPdTextureInfo(colorAttachment).usage);
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.color_usage", ctx,
                                        commandState->currentCommandBufferOwner,
                                        commandState->currentRenderEncoderOwner,
                                        commandState->renderPassStateOwner,
                                        areas.drawable);
                }
                return 0;
            }
        }
    }
    if (mglPdDepthTextureFor(commandState) ||
        mglPdStencilTextureFor(commandState)) {
        passHasAnyAttachment = 1;
    }

    if (!passHasAnyAttachment) {
        fprintf(stderr,
                "MGL WARNING: processGLState - render pass has no attachments, skipping draw to avoid Metal assert\n");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.no_attachments", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }

    uint32_t currentColor0Format = mglRenderInvalidPixelFormat();
    uint32_t currentDepthFormat = mglRenderInvalidPixelFormat();
    uint32_t currentStencilFormat = mglRenderInvalidPixelFormat();

    void *rpColor0 = mglPdColorTextureFor(commandState, 0);
    void *rpDepth = mglPdDepthTextureFor(commandState);
    void *rpStencil = mglPdStencilTextureFor(commandState);
    if (rpColor0) {
        currentColor0Format = mglPdTextureInfo(rpColor0).pixel_format;
    }
    if (rpDepth) {
        currentDepthFormat = mglPdTextureInfo(rpDepth).pixel_format;
    }
    if (rpStencil) {
        currentStencilFormat = mglPdTextureInfo(rpStencil).pixel_format;
    }

    /* IMPORTANT: never mutate depth/stencil attachments here to "fit" an
     * existing pipeline.  The active Metal render encoder was already created
     * with a render-pass descriptor, and changing attachments after encoder
     * creation does not make that encoder compatible.  We must instead reject
     * mismatched pipeline/pass combinations and rebuild safely. */
    const MGLRenderPassFormatCompatInput formatInput = {
        (uint32_t)areas.pipeline_cache->pipelineColor0Format,
        (uint32_t)areas.pipeline_cache->pipelineDepthFormat,
        (uint32_t)areas.pipeline_cache->pipelineStencilFormat,
        currentColor0Format, currentDepthFormat, currentStencilFormat};
    MGLRenderPassFormatCompat formatCompat;
    if (mglRenderPassPlanFormatCompat(&formatInput, &formatCompat) != 0) {
        return 0;
    }

    if (formatCompat.color_mismatch) {
        static uint64_t s_colorFormatMismatchCount = 0;
        s_colorFormatMismatchCount++;
        if (s_colorFormatMismatchCount <= 16 ||
            (s_colorFormatMismatchCount % 250) == 0) {
            fprintf(stderr,
                    "MGL WARNING: Pipeline/pass color format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild\n",
                    (unsigned long)areas.pipeline_cache->pipelineColor0Format,
                    (unsigned long)currentColor0Format);
        }
        mglRenderPassInvalidateCurrentPipelineState(
            renderer, "pipeline/pass color format mismatch");
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        return 0;
    }

    if (formatCompat.depth_mismatch) {
        static uint64_t s_depthFormatMismatchCount = 0;
        s_depthFormatMismatchCount++;
        if (s_depthFormatMismatchCount <= 16 ||
            (s_depthFormatMismatchCount % 250) == 0) {
            fprintf(stderr,
                    "MGL WARNING: Pipeline/pass depth format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild\n",
                    (unsigned long)areas.pipeline_cache->pipelineDepthFormat,
                    (unsigned long)currentDepthFormat);
        }
        mglRenderPassInvalidateCurrentPipelineState(
            renderer, "pipeline/pass depth format mismatch");
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        return 0;
    }

    if (formatCompat.stencil_mismatch) {
        static uint64_t s_stencilFormatMismatchCount = 0;
        s_stencilFormatMismatchCount++;
        if (s_stencilFormatMismatchCount <= 16 ||
            (s_stencilFormatMismatchCount % 250) == 0) {
            fprintf(stderr,
                    "MGL WARNING: Pipeline/pass stencil format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild\n",
                    (unsigned long)areas.pipeline_cache->pipelineStencilFormat,
                    (unsigned long)currentStencilFormat);
        }
        mglRenderPassInvalidateCurrentPipelineState(
            renderer, "pipeline/pass stencil format mismatch");
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        return 0;
    }
    return 1;
}

/* === processGLStateLocked (log 179) ==================================== */

/* Declared in the Objective-C MGLRenderer+Draw_Private.h. */
/* MGLRenderer+Draw_Private.h has this as a `static inline`; C needs its own
 * twin (kMGLDiagnosticStateLogs is 0 in mgl_trace_log.h). */
static bool mglPdShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}
extern void mglLogLoopHeartbeat(const char *tag, uint64_t call,
                                double now_seconds, double *last_time,
                                uint64_t *last_count, double interval);
extern Program *mglResolveProgramFromState(GLMContext ctx);

/* File-scope twins of the .m's function-local statics. */
static uint64_t s_pdProcessGLStateCallCount = 0;
static double s_pdProcessGLStateLastCallTime = 0.0;
static uint64_t s_pdProcessGLStateLastCallCount = 0;
static int s_pdCorruptionRecoveryCount = 0;
static const int kMglPdMaxRecoveryAttempts = 3;
static uint64_t s_pdQuarantineSkipCount = 0;
static uint64_t s_pdRotateFinalizedCount = 0;
static uint64_t s_pdNilEncoderRecoveryCount = 0;
static uint64_t s_pdDrawPipelineLookupCount = 0;
static uint64_t s_pdNilPipelineCount = 0;


/* @try of the corruption-recovery block: 1 = device and queue are usable. */
static int mglPdRecoveryTryBody(void *renderer)
{
    mglRenderPassEmergencyResetMetalState(renderer);
    s_pdCorruptionRecoveryCount++;
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    const int deviceOk =
        mglRendererBackendGetDevice(areas.backend) &&
        ((uintptr_t)mglRendererBackendGetDevice(areas.backend) >= 0x1000);
    const int queueOk =
        mglRendererBackendGetCommandQueue(areas.backend) &&
        ((uintptr_t)mglRendererBackendGetCommandQueue(areas.backend) >= 0x1000);
    if (!deviceOk || !queueOk) {
        fprintf(stderr,
                "MGL CRITICAL: Metal recovery failed, aborting operation\n");
        return 0;
    }
    return 1;
}

/* @try of the no-VAO clear path: the catch only logs. */
static int mglPdNoVaoEncoderTryBody(void *renderer)
{
    (void)mglRenderPassNewRenderEncoderLockedWithReason(renderer,
                                                          MGL_ENC_REASON_CLEAR);
    return 1;
}

typedef struct MglPdSetPipelineCtx_t {
    MGLRendererStateAreas *areas;
} MglPdSetPipelineCtx;

/* @try of the set-pipeline block: the body has no failure path, so 0 from the
 * guarded call means "an exception was thrown". */
static int mglPdSetPipelineTryBody(void *renderer, void *rawCtx)
{
    MglPdSetPipelineCtx *ctx = (MglPdSetPipelineCtx *)rawCtx;
    MGLCommandState *commandState = ctx->areas->command;
    /* The areas field is the ADDRESS of the owner slot (mgl_renderer_ports.h);
     * the .m this block came from passed the ivar. */
    void *bindingOwner = ctx->areas->binding_state_owner
                             ? *ctx->areas->binding_state_owner
                             : NULL;
    if (mglRenderBindingSetPipelineIfNeededForOwner(
            bindingOwner,
            commandState->currentRenderEncoderOwner,
            ctx->areas->pipeline_cache
                ? ctx->areas->pipeline_cache->pipelineState
                : NULL) > 0) {
        MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
    } else {
        MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
    }
    return 1;
}

static int mglRenderBindFragmentSparseResidency(void *renderer,
                                                 GLMContext ctx,
                                                 Program *program)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLRenderEncoderOwner *encoderOwner =
        areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    if (!program ||
        !mglProgramModuleView(program, _FRAGMENT_SHADER)->needs_sparse_residency) {
        return 1;
    }
    if (!encoderOwner) return 0;

    MGLSparseResidencySlot slots[TEXTURE_UNITS];
    MGLSparseResidencyRect *rects = NULL;
    size_t rectCount = 0u, rectCap = 0u;
    mglSparseResidencyResetSlots(slots, TEXTURE_UNITS);
    MGLShaderResourceList *resources =
        &program->shader_resources_list[_FRAGMENT_SHADER][_SAMPLED_IMAGE_RES];
    for (GLuint ri = 0u; resources->list && ri < resources->count; ri++) {
        MGLShaderResource *resource = &resources->list[ri];
        const uint32_t declaredType = mglDeclaredTextureTypeFromResource(resource);
        const GLint elements = resource->gl_array_size > 0
            ? resource->gl_array_size : 1;
        for (GLint element = 0; element < elements; element++) {
            const GLuint metalSlot = resource->binding + (GLuint)element;
            if (metalSlot >= TEXTURE_UNITS) break;
            const uint32_t base = resource->sampler_unit >= 0
                ? (uint32_t)resource->sampler_unit : resource->gl_binding;
            Texture *texture = mglTextureForSampledResource(
                ctx, resource, metalSlot, _FRAGMENT_SHADER, declaredType,
                base + (uint32_t)element);
            if (!texture) continue;
            if (mglSparseResidencyBuildSlot(texture, NULL, &slots[metalSlot],
                                            &rects, &rectCount, &rectCap) != 0) {
                free(rects);
                return 0;
            }
        }
    }

    const size_t headerBytes = sizeof(slots);
    if (rectCount > (SIZE_MAX - headerBytes) / sizeof(*rects)) {
        free(rects);
        return 0;
    }
    const size_t packBytes = headerBytes + rectCount * sizeof(*rects);
    uint8_t *pack = malloc(packBytes);
    if (!pack) {
        free(rects);
        return 0;
    }
    memcpy(pack, slots, headerBytes);
    if (rectCount) {
        memcpy(pack + headerBytes, rects, rectCount * sizeof(*rects));
    }
    free(rects);

    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(pack, packBytes, 0u, NULL, &buffer) != 0 ||
        !buffer) {
        free(pack);
        return 0;
    }
    free(pack);
    const uint32_t slot = mglProgramModuleView(program, _FRAGMENT_SHADER)
        ->sparse_residency_buffer_index;
    const int rc = mglRenderSetRenderBufferForOwner(
        encoderOwner, buffer, 0u, MGL_RENDER_BINDING_STAGE_FRAGMENT, slot);
    if (rc != 0) {
        CFRelease((CFTypeRef)buffer);
        return 0;
    }

    void *temporaries = mglRendererTemporariesCreate();
    if (!temporaries) {
        CFRelease((CFTypeRef)buffer);
        return 0;
    }
    mglRendererTemporariesAdd(temporaries, buffer);
    CFRelease((CFTypeRef)buffer);
    void *commandBuffer = mglRenderCommandBufferOwnerGetCurrent(
        areas.command ? areas.command->currentCommandBufferOwner : NULL);
    mglRendererTemporariesReleaseOnCommandBuffer(temporaries, commandBuffer);
    return 1;
}

/* -processGLStateLocked:. */
int mglRenderPassProcessGLStateLocked(void *renderer, int draw_command)
{
    /* PERF-016 已修: clean GL state + active encoder skips full sync on draws. */
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;
    GLMState *glState = mglPdState(&areas);

    const uint64_t processCall = ++s_pdProcessGLStateCallCount;
    const bool traceProcess = mglPdShouldTraceCall(processCall);
    /* F4: clock/heartbeat only when tracing or diagnostic logs — every draw
     * previously paid TraceClockNS×2 (~33 ns) even with kMGLDiagnosticStateLogs=0. */
    const int timeProcess = (traceProcess || kMGLDiagnosticStateLogs) ? 1 : 0;
    double processStartSeconds = 0.0;
    uint64_t processStartNS = 0u;
    if (timeProcess) {
        processStartSeconds = mglTraceNowSeconds();
        processStartNS = mglTraceClockNS();
        mglLogLoopHeartbeat("processGLState.loop", processCall, processStartSeconds,
                            &s_pdProcessGLStateLastCallTime,
                            &s_pdProcessGLStateLastCallCount, 0.25);
    }
    if (traceProcess) {
        mglTraceLog("MGL TRACE processGLState.begin call=%llu draw=%d",
                    (unsigned long long)processCall, draw_command ? 1 : 0);
        mglLogStateSnapshot("processGLState.enter", ctx,
                            commandState->currentCommandBufferOwner,
                            commandState->currentRenderEncoderOwner,
                            commandState->renderPassStateOwner, areas.drawable);
    }
    if (!ctx) {
        fprintf(stderr, "MGL ERROR: NULL context detected in processGLState\n");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }
    /* T0-1: processGLState reads via mglPdState (ctx->active_state);
     * dual-proxy must already agree with ctx->active_state. */
    mglCoreAssertDualProxy(areas.core, ctx);

    const uintptr_t earlyCtxAddr = (uintptr_t)ctx;
    const int ctxPtrSane = earlyCtxAddr >= 0x1000 ? 1 : 0;
    if (!ctxPtrSane) {
        fprintf(stderr, "MGL ERROR: Invalid context pointer detected: 0x%lx\n",
                (unsigned long)earlyCtxAddr);
        return 0;
    }

    /* Metal corruption recovery is platform materialization - try before plan. */
    int deviceOk = mglRendererBackendGetDevice(areas.backend) &&
                   ((uintptr_t)mglRendererBackendGetDevice(areas.backend) >=
                    0x1000);
    int queueOk = mglRendererBackendGetCommandQueue(areas.backend) &&
                  ((uintptr_t)mglRendererBackendGetCommandQueue(areas.backend) >=
                   0x1000);
    if (!deviceOk || !queueOk) {
        fprintf(stderr,
                "MGL CRITICAL: Metal state corruption detected in processGLState!\n");
        fprintf(stderr, "MGL CRITICAL: device=0x%lx, queue=0x%lx\n",
                (unsigned long)(uintptr_t)mglRendererBackendGetDevice(
                    areas.backend),
                (unsigned long)(uintptr_t)mglRendererBackendGetCommandQueue(
                    areas.backend));
        if (s_pdCorruptionRecoveryCount < kMglPdMaxRecoveryAttempts) {
            fprintf(stderr,
                    "MGL CRITICAL: Attempting Metal state recovery (%d/%d)\n",
                    s_pdCorruptionRecoveryCount + 1, kMglPdMaxRecoveryAttempts);
            if (!mglPlatformShellGuardedCall(renderer, "metal state recovery",
                                             mglPdRecoveryTryBody)) {
                return 0;
            }
            deviceOk = mglRendererBackendGetDevice(areas.backend) &&
                       ((uintptr_t)mglRendererBackendGetDevice(areas.backend) >=
                        0x1000);
            queueOk =
                mglRendererBackendGetCommandQueue(areas.backend) &&
                ((uintptr_t)mglRendererBackendGetCommandQueue(areas.backend) >=
                 0x1000);
            if (!deviceOk || !queueOk) {
                fprintf(stderr,
                        "MGL CRITICAL: Metal recovery failed, aborting operation\n");
                return 0;
            }
        } else {
            fprintf(stderr,
                    "MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations\n");
            return 0;
        }
    }

    int quarantineBlocks = 0;
    if (draw_command) {
        const GLuint blockedProgramKey = mglCurrentRenderProgramKey(ctx);
        if (blockedProgramKey != 0u &&
            areas.gpu_interface_mismatch_blocked_program != 0 &&
            blockedProgramKey == areas.gpu_interface_mismatch_blocked_program) {
            const double now = CFAbsoluteTimeGetCurrent();
            if (now < areas.gpu_interface_mismatch_blocked_until) {
                quarantineBlocks = 1;
                s_pdQuarantineSkipCount++;
                if (s_pdQuarantineSkipCount <= 16 ||
                    (s_pdQuarantineSkipCount % 1000) == 0) {
                    double remaining =
                        areas.gpu_interface_mismatch_blocked_until - now;
                    if (remaining < 0.0) remaining = 0.0;
                    fprintf(stderr,
                            "MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw\n",
                            (unsigned)areas
                                .gpu_interface_mismatch_blocked_program,
                            remaining);
                }
            }
        }
    }

    if (draw_command && !quarantineBlocks && glState->vao != NULL &&
        glState->dirty_bits == 0 &&
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1) {
        MGL_FRAME_INC(g_mglProcessDrawCallsSinceSwap);
        return 1;
    }

    MGLRenderCommandBufferState processCommandState = {0};
    const int processHasCommand =
        mglRenderGetCommandBufferOwnerState(
            commandState->currentCommandBufferOwner,
            &processCommandState) == 0;
    const int encoderCurrent =
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1
            ? 1
            : 0;

    MGLProcessGLStateInputs planIn = {0};
    planIn.has_ctx = 1u;
    planIn.draw_command = draw_command ? 1u : 0u;
    planIn.has_vao = glState->vao != NULL ? 1u : 0u;
    planIn.dirty_state = (glState->dirty_bits & DIRTY_STATE) ? 1u : 0u;
    planIn.ctx_ptr_sane = 1u;
    planIn.device_ok = deviceOk ? 1u : 0u;
    planIn.queue_ok = queueOk ? 1u : 0u;
    planIn.quarantine_blocks_draw = quarantineBlocks ? 1u : 0u;
    planIn.has_command_buffer = processHasCommand ? 1u : 0u;
    planIn.encoder_has_current = encoderCurrent ? 1u : 0u;
    planIn.command_buffer_status =
        processHasCommand ? (uint32_t)processCommandState.status : 0u;

    MGLProcessGLStatePlan plan = {0};
    if (mglRenderProcessGLState(&planIn, &plan) != 0) {
        return 0;
    }

    if (draw_command) {
        MGL_FRAME_INC(g_mglProcessDrawCallsSinceSwap);
    }

    if (plan.result == MGL_PGL_RESULT_ABORT) {
        if (draw_command && !planIn.has_vao &&
            plan.process_class == MGL_PROCESS_GL_ABORT) {
            fprintf(stderr, "Error: No VAO defined for ctx\n\n");
        }
        return 0;
    }

    if (plan.non_draw_end_pass_if_fbo_changed) {
        mglRenderPassEndIfFramebufferChangedForNonDraw(renderer,
                                                                   processCall);
    }
    if (plan.no_vao_clear_path) {
        mglRendererEndRenderEncodingLocked(renderer);
        if (!mglRendererValidateMetalObjects(renderer)) {
            fprintf(stderr,
                    "MGL WARNING: GPU throttling active - deferring render encoder creation\n");
            glState->dirty_bits &= ~DIRTY_STATE;
            return 1;
        }
        (void)mglPlatformShellGuardedCall(renderer, "clear-path render encoder",
                                          mglPdNoVaoEncoderTryBody);
        glState->dirty_bits &= ~DIRTY_STATE;
        return 1;
    }
    if (plan.result == MGL_PGL_RESULT_EARLY_OK) {
        return 1;
    }

    if (plan.rotate_finalized_command_buffer) {
        const uint64_t rotateHit = ++s_pdRotateFinalizedCount;
        if (rotateHit <= 16ull || (rotateHit % 500ull) == 0ull) {
            fprintf(stderr,
                    "MGL INFO: processGLState rotating finalized command buffer (status: %ld) hit=%llu\n",
                    (long)planIn.command_buffer_status,
                    (unsigned long long)rotateHit);
        }
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: processGLState failed to create a fresh command buffer\n");
            if (traceProcess) {
                mglLogStateSnapshot(
                    "processGLState.fail.new_cb_rotate", ctx,
                    commandState->currentCommandBufferOwner,
                    commandState->currentRenderEncoderOwner,
                    commandState->renderPassStateOwner, areas.drawable);
            }
            return 0;
        }
    } else if (plan.create_initial_command_buffer) {
        if (kMglPdVerboseFrameLoopLogs) {
            fprintf(stderr,
                    "MGL INFO: processGLState found NULL command buffer, creating one\n");
        }
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: processGLState could not create initial command buffer\n");
            if (traceProcess) {
                mglLogStateSnapshot(
                    "processGLState.fail.new_cb_initial", ctx,
                    commandState->currentCommandBufferOwner,
                    commandState->currentRenderEncoderOwner,
                    commandState->renderPassStateOwner, areas.drawable);
            }
            return 0;
        }
    }

    /* PERF-038: after unlocked swap (swapInterval==0) the drawable is cleared at
     * post_commit; acquire once here before default-FBO encoding instead of at swap. */
    if (areas.layer != NULL && areas.drawable == NULL &&
        glState->framebuffer == NULL) {
        const GLuint mgl_drawbuffer =
            mglDefaultDrawBufferIndexForGL(glState->draw_buffer);
        if (mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer) &&
            mglRenderShouldPresentDrawBuffer((uint32_t)glState->draw_buffer) !=
                0) {
            (void)mglPlatformShellApplyPendingDrawableSize(renderer);
            (void)mglRendererNextDrawable(renderer);
        }
    }

    MGLResourceSyncWork resourceSyncWork = {false, false, false};
    if (plan.process_dirty_domains) {
        if (!mglRenderPassProcessDirtyStateDomains(renderer,
                                                   draw_command ? 1 : 0,
                                                   &resourceSyncWork)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    }

    /* Phase 2: re-sample encoder/pipeline after dirty-domain materialization. */
    Program *fragmentProgram =
        mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    MGLProcessGLStateAfterInputs afterIn = {0};
    afterIn.draw_command = draw_command ? 1u : 0u;
    afterIn.encoder_has_current =
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1
            ? 1u
            : 0u;
    afterIn.has_pipeline_state =
        (areas.pipeline_cache && areas.pipeline_cache->pipelineState) ? 1u : 0u;
    afterIn.frag_needs_fragcoord =
        fragmentProgram && mglRenderSamplerUnitExplicit(
                               (uint32_t)fragmentProgram->usesFragCoordParams)
            ? 1u
            : 0u;
    afterIn.frag_needs_sample =
        ((fragmentProgram && mglRenderSamplerUnitExplicit(
                                 (uint32_t)fragmentProgram->uses_sample_params)) ||
         mglPlatformShellMSSampleInLoop(renderer))
            ? 1u
            : 0u;
    afterIn.frag_needs_lod_bias =
        fragmentProgram && mglProgramStageHasSamplers(fragmentProgram,
                                                     _FRAGMENT_SHADER)
            ? 1u
            : 0u;

    MGLProcessGLStateAfterPlan after = {0};
    if (mglRenderProcessGLStateAfterDirty(&afterIn, &after) != 0) {
        return 0;
    }

    if (after.recover_nil_encoder) {
        const uint64_t nilHit = ++s_pdNilEncoderRecoveryCount;
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: processGLState - current render encoder is nil, attempting recovery hit=%llu\n",
                    (unsigned long long)nilHit);
            mglLogRenderPassLifecycle(
                "nil-encoder-before-recovery", nilHit, ctx,
                commandState->currentCommandBufferOwner,
                commandState->currentRenderEncoderOwner,
                commandState->renderPassStateOwner, areas.drawable,
                commandState->renderPassFramebuffer,
                commandState->renderPassFramebufferName,
                commandState->renderPassDrawBuffer,
                commandState->renderPassDrawBufferCount);
        }
        if (!mglRenderPassNewRenderEncoderLockedWithReason(
                renderer, MGL_ENC_REASON_NIL)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            mglLogRenderPassLifecycle(
                "nil-encoder-after-recovery", nilHit, ctx,
                commandState->currentCommandBufferOwner,
                commandState->currentRenderEncoderOwner,
                commandState->renderPassStateOwner, areas.drawable,
                commandState->renderPassFramebuffer,
                commandState->renderPassFramebufferName,
                commandState->renderPassDrawBuffer,
                commandState->renderPassDrawBufferCount);
        }
    }

    if (after.ensure_pass_matches_fbo) {
        if (!mglRenderPassEnsureCurrentRenderPassMatchesFramebufferForDraw(
                renderer)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
        mglRenderPassUpdateCurrentRenderEncoder(renderer);
    }

    if (draw_command && kMglPdVerbosePipelineLogs) {
        s_pdDrawPipelineLookupCount++;
        if (s_pdDrawPipelineLookupCount <= 256ull ||
            (s_pdDrawPipelineLookupCount % 1000ull) == 0ull) {
            Program *lookupProgram = mglResolveProgramFromState(ctx);
            Program *lookupVertexProgram =
                mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *lookupFragmentProgram =
                mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            const GLuint lookupProgramName = mglCurrentRenderProgramKey(ctx);
            Framebuffer *lookupFBO = glState->framebuffer;
            const GLuint lookupFBOName = lookupFBO ? lookupFBO->name : 0;
            fprintf(stderr, "MGL Draw current program key=%u mono=%p vs=%u fs=%u\n",
                    (unsigned)lookupProgramName, (void *)lookupProgram,
                    lookupVertexProgram ? (unsigned)lookupVertexProgram->name
                                        : 0u,
                    lookupFragmentProgram
                        ? (unsigned)lookupFragmentProgram->name
                        : 0u);
            fprintf(stderr,
                    "MGL DRAW pipeline lookup result=%p key=%u vs=%u fs=%u vao=%p fbo=%u\n",
                    areas.pipeline_cache ? areas.pipeline_cache->pipelineState
                                         : NULL,
                    (unsigned)lookupProgramName,
                    lookupVertexProgram ? (unsigned)lookupVertexProgram->name
                                        : 0u,
                    lookupFragmentProgram
                        ? (unsigned)lookupFragmentProgram->name
                        : 0u,
                    glState->vao, (unsigned)lookupFBOName);
        }
    }

    if (after.fail_nil_pipeline) {
        s_pdNilPipelineCount++;
        if (s_pdNilPipelineCount <= 8 || (s_pdNilPipelineCount % 1000) == 0) {
            mglTraceLog(
                "MGL DRAW SKIP: pipelineState is nil, forcing rebuild (occurrence=%llu)",
                (unsigned long long)s_pdNilPipelineCount);
        }
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_pipeline", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }

    if (after.validate_attachments) {
        if (!mglRenderPassValidateAttachmentsAndPipelineFormats(
                renderer, traceProcess ? 1 : 0)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    }

    {
        Program *vertexProgram =
            mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
        const int need_reduction =
            (fragmentProgram &&
             mglProgramBoundSamplersNeedReduction(ctx, fragmentProgram,
                                                _FRAGMENT_SHADER)) ||
            (vertexProgram &&
             mglProgramBoundSamplersNeedReduction(ctx, vertexProgram,
                                                _VERTEX_SHADER));
        if (fragmentProgram)
            fragmentProgram->active_sampler_reduction =
                need_reduction ? GL_TRUE : GL_FALSE;
        if (vertexProgram)
            vertexProgram->active_sampler_reduction =
                need_reduction ? GL_TRUE : GL_FALSE;
    }

    if (after.set_pipeline) {
        MglPdSetPipelineCtx setCtx = {&areas};
        if (!mglPlatformShellGuardedCallCtx(renderer, "set render pipeline state",
                                            mglPdSetPipelineTryBody, &setCtx,
                                            NULL)) {
            /* @catch (NSException *exception) */
            fprintf(stderr,
                    "MGL ERROR: processGLState - setRenderPipelineState failed\n");
            mglMarkRendererDirtyBits(ctx->active_state,
                                     DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                         DIRTY_RENDER_STATE);
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.set_pipeline", ctx,
                                    commandState->currentCommandBufferOwner,
                                    commandState->currentRenderEncoderOwner,
                                    commandState->renderPassStateOwner,
                                    areas.drawable);
            }
            return 0;
        }
    }

    if (after.sync_resources) {
        if (!mglRendererSyncResourceBindingsForContext(renderer, ctx,
                                                       &resourceSyncWork)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    }

    if (after.bind_frag_coord_slot) {
        const int useFragCoordParams = afterIn.frag_needs_fragcoord ? 1 : 0;
        const int useSampleParams = afterIn.frag_needs_sample ? 1 : 0;
        uint64_t passHeight = mglPdRenderTargetHeightFor(commandState);
        if (passHeight == 0) {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS && passHeight == 0; i++) {
                void *color = mglPdColorTextureFor(commandState, (size_t)i);
                passHeight = color ? mglPdTextureInfo(color).height : 0;
            }
            if (passHeight == 0 && mglPdDepthTextureFor(commandState)) {
                passHeight =
                    mglPdTextureInfo(mglPdDepthTextureFor(commandState)).height;
            }
            if (passHeight == 0 && mglPdStencilTextureFor(commandState)) {
                passHeight = mglPdTextureInfo(mglPdStencilTextureFor(commandState))
                                 .height;
            }
        }

        uint32_t numSamples = 1;
        uint32_t sampleBuffers = 0;
        Framebuffer *fbo = glState->framebuffer;
        if (fbo && (fbo->color_attachment_bitfield & 1u)) {
            FBOAttachment *att = &fbo->color_attachments[0];
            Texture *tex = NULL;
            if (mglRenderTargetIsRenderbuffer((uint32_t)att->textarget) &&
                att->buf.rbo) {
                tex = att->buf.rbo->tex;
            } else {
                tex = att->buf.tex;
            }
            if (tex) {
                (void)mglRenderTextureSampleParams(
                    (uint32_t)tex->target, tex->samples, &numSamples,
                    &sampleBuffers);
            }
        } else {
            void *rpColor0 = mglPdColorTextureFor(commandState, 0);
            if (rpColor0) {
                const uint64_t sc = mglPdTextureInfo(rpColor0).sample_count;
                if (sc > 1) {
                    sampleBuffers = 1;
                    numSamples = (uint32_t)sc;
                }
            }
        }
        float fragCoordParams[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
        const MGLRasterOrientationPlan orientation =
            mglRasterOrientationForState(glState);
        /* Metal's point_coord counts rows; a GL-order target counts them from
         * the bottom, so the sprite origin flips with it. */
        const int pointLowerLeft =
            (glState->var.point_sprite_coord_origin == GL_LOWER_LEFT) !=
            (mglRasterTargetStoresGLOrder(glState) != 0);
        mglRenderFillFragCoordSlot(
            useFragCoordParams, useSampleParams, (uint32_t)passHeight,
            orientation.fragcoord_from_top != 0,
            numSamples, sampleBuffers, mglPlatformShellMSSampleInLoop(renderer),
            (uint32_t)areas.mssample_forced_id,
            pointLowerLeft,
            fragCoordParams);
        (void)mglRenderSetRenderBytesForOwner(
            commandState->currentRenderEncoderOwner, fragCoordParams,
            sizeof(fragCoordParams), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLFragCoordParamsBufferIndex);
        mglRendererBindingInvalidateLastBoundFragmentBufferAtIndex(
            renderer, kMGLFragCoordParamsBufferIndex);
    }

    {
        Program *vertexProgram =
            mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
        const int bind_fs_lod = after.bind_lod_bias_slot != 0u;
        const int bind_vs_lod =
            vertexProgram &&
            mglProgramStageHasSamplers(vertexProgram, _VERTEX_SHADER);
        if (bind_fs_lod || bind_vs_lod) {
            float pack[kMGLLodBiasPackHeaderFloats + TEXTURE_UNITS];
            if (bind_fs_lod) {
                mglFillLodBiasPack(ctx, fragmentProgram, _FRAGMENT_SHADER,
                                   pack);
                mglRenderSetRenderBytesForOwner(
                    commandState->currentRenderEncoderOwner, pack, sizeof(pack),
                    MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    kMGLLodBiasMaxBufferIndex);
                mglRendererBindingInvalidateLastBoundFragmentBufferAtIndex(
                    renderer, kMGLLodBiasMaxBufferIndex);
            }
            if (bind_vs_lod) {
                mglFillLodBiasPack(ctx, vertexProgram, _VERTEX_SHADER, pack);
                mglRenderSetRenderBytesForOwner(
                    commandState->currentRenderEncoderOwner, pack, sizeof(pack),
                    MGL_RENDER_BINDING_STAGE_VERTEX,
                    kMGLLodBiasMaxBufferIndex);
                mglRendererBindingInvalidateLastBoundVertexBufferAtIndex(
                    renderer, kMGLLodBiasMaxBufferIndex);
            }
        }
    }

    if (!mglRenderBindFragmentSparseResidency(renderer, ctx, fragmentProgram)) {
        return 0;
    }
    if (draw_command &&
        !mglRendererBindPostDepthCoverageInputs(renderer, ctx,
                                                fragmentProgram)) {
        return 0;
    }

    if (after.bind_sampler_wrap_slot) {
        MGLSamplerWrapSlot vs_slots[MGL_SAMPLER_WRAP_SLOT_COUNT];
        MGLSamplerWrapSlot fs_slots[MGL_SAMPLER_WRAP_SLOT_COUNT];
        Program *vertexProgram =
            mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
        mglFillSamplerWrapTable(ctx, vertexProgram, _VERTEX_SHADER, vs_slots);
        mglFillSamplerWrapTable(ctx, fragmentProgram, _FRAGMENT_SHADER,
                                fs_slots);
        mglRenderSetRenderBytesForOwner(
            commandState->currentRenderEncoderOwner, vs_slots, sizeof(vs_slots),
            MGL_RENDER_BINDING_STAGE_VERTEX, kMGLSamplerWrapBufferIndex);
        mglRendererBindingInvalidateLastBoundVertexBufferAtIndex(
            renderer, kMGLSamplerWrapBufferIndex);
        mglRenderSetRenderBytesForOwner(
            commandState->currentRenderEncoderOwner, fs_slots, sizeof(fs_slots),
            MGL_RENDER_BINDING_STAGE_FRAGMENT, kMGLSamplerWrapBufferIndex);
        mglRendererBindingInvalidateLastBoundFragmentBufferAtIndex(
            renderer, kMGLSamplerWrapBufferIndex);
    }

    if (timeProcess) {
        const double processElapsedUs =
            (mglTraceClockNS() - processStartNS) / 1000.0;
        if (traceProcess) {
            mglTraceLog(
                "MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.1fus",
                (unsigned long long)processCall, draw_command ? 1 : 0,
                processElapsedUs);
            mglLogStateSnapshot(
                "processGLState.exit.ok", ctx,
                commandState->currentCommandBufferOwner,
                commandState->currentRenderEncoderOwner,
                commandState->renderPassStateOwner, areas.drawable);
        } else if (processElapsedUs >= 25.0) {
            mglTraceLog(
                "MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.1fus",
                (unsigned long long)processCall, draw_command ? 1 : 0,
                processElapsedUs);
        }
    }
    return 1;
}

/* === mtlSwapBuffersLocked (log 180) ==================================== */

/* Shell forwarders for swap-path state that must be computed on the shell
 * object (the interval and the layer travel in the areas instead). */
extern int mglPlatformShellShouldSkipPresentForUnlockedSwap(void *renderer);
extern MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);
extern void *mglPlatformShellDrawablePointer(void *renderer);

/* The .m's file-local constants this TU needs (values copied verbatim from
 * MGLRenderer.m's enum, as mgl_renderer_host.c already does). */
enum {
    MGL_PD_CB_NOT_ENQUEUED = 0u,
    MGL_PD_CB_ERROR = 5u,
};

/* MGLRenderer.m had this as a static inline. */static bool mglPdContextLikelyValid(GLMContext ctx)
{
    return (ctx != NULL) && ((uintptr_t)ctx >= 0x10000u);
}

/* File-scope twins of the .m's function-local statics. */
static uint64_t s_pdSwapCallCount = 0;
static double s_pdSwapLastCallTime = 0.0;
static uint64_t s_pdSwapLastCallCount = 0;
static volatile double s_pdMainThreadHeartbeatSeconds = 0.0;
static volatile uint64_t s_pdMainThreadPingCount = 0;
static uint64_t s_pdSwapProcessStateFailCount = 0;
static uint64_t s_pdSwapFinalizedBufferCount = 0;

/* dispatch_async(dispatch_get_main_queue(), ^{ ... }) became dispatch_async_f:
 * the block only touched these two statics, so no context is needed. */
static void mglPdMainThreadPing(void *unused)
{
    (void)unused;
    s_pdMainThreadHeartbeatSeconds = mglTraceNowSeconds();
    s_pdMainThreadPingCount++;
}

typedef struct MglPdPresentCtx_t {
    void *command_state_owner;
    void *drawable;
    int result;
} MglPdPresentCtx;

/* @try of the drawable presentation.  -1 = in-body failure (already logged and
 * the caller just returns), 0 = exception (the caller runs the @catch block). */
static int mglPdPresentTryBody(void *renderer, void *rawCtx)
{
    MglPdPresentCtx *ctx = (MglPdPresentCtx *)rawCtx;
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLCommandState *commandState = areas.command;
    GLMContext glmCtx = areas.ctx;

    if (!mglRendererDrawableTexture(renderer)) {
        fprintf(stderr,
                "MGL ERROR: Drawable texture is NULL, cannot present\n");
        ctx->result = -1;
        return 0;
    }

    void *currentDrawableTexture = mglRendererDrawableTexture(renderer);
    MGLRenderTextureInfo currentDrawableInfo =
        mglPdTextureInfo(currentDrawableTexture);
    if (currentDrawableInfo.width == 0 || currentDrawableInfo.height == 0) {
        fprintf(stderr, "MGL ERROR: Drawable has invalid dimensions: %dx%d\n",
                (int)currentDrawableInfo.width,
                (int)currentDrawableInfo.height);
        ctx->result = -1;
        return 0;
    }

    if (kMglPdVerboseFrameLoopLogs) {
        fprintf(stderr,
                "MGL INFO: Presenting drawable with texture: %dx%d, format: %lu\n",
                (int)currentDrawableInfo.width,
                (int)currentDrawableInfo.height,
                (unsigned long)currentDrawableInfo.pixel_format);
    }

    if (mglRenderPresentDrawableForCommandBufferOwner(
            commandState->currentCommandBufferOwner, ctx->drawable, NULL) != 0) {
        fprintf(stderr,
                "MGL ERROR: No command buffer available for drawable presentation\n");
        ctx->result = -1;
        return 0;
    }
    (void)glmCtx;
    ctx->result = 1;
    return 1;
}

typedef struct MglPdCommitSwapCtx_t {
    void *command_buffer;
    uint64_t swap_call;
    int trace_swap;
} MglPdCommitSwapCtx;

/* @try of the frame commit; the catch records a GPU error. */
static int mglPdCommitSwapTryBody(void *renderer, void *rawCtx)
{
    MglPdCommitSwapCtx *ctx = (MglPdCommitSwapCtx *)rawCtx;

    if (ctx->trace_swap) {
        char commandBufferLabel[256] = {0};
        if (ctx->command_buffer) {
            (void)mglRenderGetCommandBufferLabel(
                ctx->command_buffer, commandBufferLabel,
                sizeof(commandBufferLabel));
        }
        mglTraceLog(
            "MGL TRACE swap.commit.begin call=%llu cb=%p status=%s label=%s",
            (unsigned long long)ctx->swap_call, ctx->command_buffer,
            mglCommandBufferStatusName(
                ctx->command_buffer
                    ? (uint32_t)mglRenderCommandBufferStatus(ctx->command_buffer)
                    : MGL_PD_CB_ERROR),
            commandBufferLabel[0] ? commandBufferLabel : "(nil)");
    }
    /* The copy-on-write buffer generation is paired with this command buffer's
     * completion by the commit transaction itself, which every submit path
     * funnels through (see mglRegisterBufferGenerationCompletion).  Registering
     * it here as well would double-record the same generation. */
    mglRendererCommitCommandBufferWithAGXRecovery(renderer,
                                                  ctx->command_buffer);
    if (ctx->trace_swap) {
        mglTraceLog("MGL TRACE swap.commit.end call=%llu",
                    (unsigned long long)ctx->swap_call);
    }
    return 1;
}

/* -mtlSwapBuffersLocked:. */
void mglRenderPassMTLSwapBuffersLocked(void *renderer, GLMContext glm_ctx)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;

    const uint64_t swapCall = ++s_pdSwapCallCount;
    const bool traceSwap = mglPdShouldTraceCall(swapCall);
    const int timeSwap =
        (traceSwap || kMGLDiagnosticStateLogs || kMglPdVerboseFrameLoopLogs) ? 1
                                                                            : 0;
    double swapStartSeconds = 0.0;
    uint64_t swapStartNS = 0u;
    if (timeSwap) {
        swapStartSeconds = mglTraceNowSeconds();
        swapStartNS = mglTraceClockNS();
    } else {
        swapStartSeconds = mglTraceNowSeconds(); /* still stamp LastSwapSeconds */
    }
    mglTraceNoteFrameBoundary();
    MGL_FRAME_STORE(g_mglSwapCallCount, swapCall);
    /* advance the DontCare frame generation. Any color attachment written
     * before this point belongs to the previous frame, so its next write this
     * frame is a "first use" that may skip loading prior contents.  Skips 0 so
     * a zero-initialized texture stamp never matches. */
    mglPassManagerIncrementDontCareFrameGenerationWithWrap(manager);
    MGL_FRAME_STORE(g_mglLastSwapSeconds, swapStartSeconds);
    if (swapCall <= 20ull || (swapCall % 60ull) == 0ull) {
        mglTraceLog(
            "SWAP_RENDERER_ENTRY call=%llu drawArraysSinceSwap=%llu drawElementsSinceSwap=%llu processDrawCallsSinceSwap=%llu",
            (unsigned long long)swapCall,
            (unsigned long long)MGL_FRAME_LOAD(g_mglDrawArraysSinceSwap),
            (unsigned long long)MGL_FRAME_LOAD(g_mglDrawElementsSinceSwap),
            (unsigned long long)MGL_FRAME_LOAD(g_mglProcessDrawCallsSinceSwap));
    }
    if (timeSwap) {
        mglLogLoopHeartbeat("swap.loop", swapCall, swapStartSeconds,
                            &s_pdSwapLastCallTime, &s_pdSwapLastCallCount, 0.25);
    }

    if (!mglPdContextLikelyValid(glm_ctx)) {
        fprintf(stderr, "MGL CRITICAL: swap.begin invalid glm_ctx=%p\n",
                (void *)glm_ctx);
        return;
    }

    if (areas.ctx != glm_ctx) {
        mglTraceLog("MGL TRACE swap.contextSync old=%p new=%p", (void *)areas.ctx,
                    (void *)glm_ctx);
        mglPlatformShellSetContext(renderer, glm_ctx);
    }
    /* The .m rebinds its `ctx` ivar; C keeps the fresh value in the snapshot and
     * in `activeCtx`. */
    areas.ctx = glm_ctx;
    GLMContext activeCtx = glm_ctx;
    const GLenum drawBuffer = activeCtx->state.draw_buffer;
    const bool shouldPresent =
        mglRenderShouldPresentDrawBuffer((uint32_t)drawBuffer) != 0;
    if (traceSwap) {
        mglTraceLog("MGL TRACE swap.begin call=%llu shouldPresent=%d draw_buffer=0x%x",
                    (unsigned long long)swapCall, shouldPresent ? 1 : 0,
                    (unsigned)drawBuffer);
        mglLogStateSnapshot("swap.enter", activeCtx,
                            commandState->currentCommandBufferOwner,
                            commandState->currentRenderEncoderOwner,
                            commandState->renderPassStateOwner, areas.drawable);
    }

    /* Main-thread responsiveness probe for beachball diagnostics.  Render thread
     * periodically posts a ping to main queue; stale heartbeat means main thread
     * is blocked. */
    if (kMGLDiagnosticStateLogs &&
        (swapCall <= 20ull || (swapCall % 30ull) == 0ull)) {
        dispatch_async_f(dispatch_get_main_queue(), NULL, mglPdMainThreadPing);

        const double hb = s_pdMainThreadHeartbeatSeconds;
        if (hb > 0.0) {
            const double lagMs = (swapStartSeconds - hb) * 1000.0;
            if (lagMs > 500.0) {
                mglTraceLog(
                    "MGL TRACE mainthread.stall suspected lag=%.2fms swapCall=%llu pingCount=%llu",
                    lagMs, (unsigned long long)swapCall,
                    (unsigned long long)s_pdMainThreadPingCount);
                if (traceSwap || (swapCall % 120ull) == 0ull) {
                    mglLogStateSnapshot("mainthread.stall.snapshot", activeCtx,
                                        commandState->currentCommandBufferOwner,
                                        commandState->currentRenderEncoderOwner,
                                        commandState->renderPassStateOwner,
                                        areas.drawable);
                }
            } else if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE mainthread.heartbeat lag=%.2fms swapCall=%llu pingCount=%llu",
                    lagMs, (unsigned long long)swapCall,
                    (unsigned long long)s_pdMainThreadPingCount);
            }
        } else if (traceSwap) {
            mglTraceLog(
                "MGL TRACE mainthread.heartbeat uninitialized swapCall=%llu",
                (unsigned long long)swapCall);
        }
    }

    if (kMGLDiagnosticStateLogs) {
        MGLSwapDrawCounters frameCounters = mglSnapshotSwapDrawCounters();
        mglResetSwapDrawCounters();

        const uint64_t lastDrawArraysCall =
            MGL_FRAME_LOAD(g_mglLastDrawArraysCall);
        const uint64_t lastDrawElementsCall =
            MGL_FRAME_LOAD(g_mglLastDrawElementsCall);
        const double lastDrawArraysSeconds =
            MGL_FRAME_LOAD(g_mglLastDrawArraysSeconds);
        const double lastDrawElementsSeconds =
            MGL_FRAME_LOAD(g_mglLastDrawElementsSeconds);
        const GLuint lastDrawArraysProgram =
            MGL_FRAME_LOAD(g_mglLastDrawArraysProgram);
        const GLuint lastDrawArraysMode =
            MGL_FRAME_LOAD(g_mglLastDrawArraysMode);
        const GLsizei lastDrawArraysCount =
            MGL_FRAME_LOAD(g_mglLastDrawArraysCount);
        const GLuint lastDrawElementsProgram =
            MGL_FRAME_LOAD(g_mglLastDrawElementsProgram);
        const GLuint lastDrawElementsMode =
            MGL_FRAME_LOAD(g_mglLastDrawElementsMode);
        const GLsizei lastDrawElementsCount =
            MGL_FRAME_LOAD(g_mglLastDrawElementsCount);
        const double drawArraysAgeMs =
            (lastDrawArraysSeconds > 0.0)
                ? ((swapStartSeconds - lastDrawArraysSeconds) * 1000.0)
                : -1.0;
        const double drawElementsAgeMs =
            (lastDrawElementsSeconds > 0.0)
                ? ((swapStartSeconds - lastDrawElementsSeconds) * 1000.0)
                : -1.0;
        const int hasFrameWork =
            (frameCounters.draw_arrays > 0 ||
             frameCounters.draw_elements > 0 ||
             frameCounters.draw_arrays_skipped > 0 ||
             frameCounters.draw_elements_skipped > 0 ||
             frameCounters.process_draw_calls > 0);
        if (traceSwap || hasFrameWork || swapCall <= 20ull ||
            (swapCall % 20ull) == 0ull) {
            mglTraceLog(
                "MGL TRACE swap.drawActivity call=%llu processDrawCalls=%llu drawArrays=%llu verts=%llu "
                "drawElements=%llu indices=%llu skipArrays=%llu skipElements=%llu "
                "lastDrawArrays=%llu prog=%u mode=0x%x count=%d age=%.2fms "
                "lastDrawElements=%llu prog=%u mode=0x%x count=%d age=%.2fms",
                (unsigned long long)swapCall,
                (unsigned long long)frameCounters.process_draw_calls,
                (unsigned long long)frameCounters.draw_arrays,
                (unsigned long long)frameCounters.array_vertices,
                (unsigned long long)frameCounters.draw_elements,
                (unsigned long long)frameCounters.element_indices,
                (unsigned long long)frameCounters.draw_arrays_skipped,
                (unsigned long long)frameCounters.draw_elements_skipped,
                (unsigned long long)lastDrawArraysCall,
                (unsigned)lastDrawArraysProgram, (unsigned)lastDrawArraysMode,
                (int)lastDrawArraysCount, drawArraysAgeMs,
                (unsigned long long)lastDrawElementsCall,
                (unsigned)lastDrawElementsProgram, (unsigned)lastDrawElementsMode,
                (int)lastDrawElementsCount, drawElementsAgeMs);
        }
    }

    if (shouldPresent) {
        mglRendererFlushDrawBufferLocked(renderer, activeCtx);

        if (!mglRenderPassProcessGLStateLocked(renderer, 0)) {
            s_pdSwapProcessStateFailCount++;
            if (s_pdSwapProcessStateFailCount <= 16 ||
                (s_pdSwapProcessStateFailCount % 500) == 0) {
                fprintf(stderr,
                        "MGL WARNING: mtlSwapBuffers continuing despite processGLState failure (occurrence=%llu)\n",
                        (unsigned long long)s_pdSwapProcessStateFailCount);
            }
        }

        mglRendererEndRenderEncodingLocked(renderer);

        /* Deferred device reset drain.  This is the only safe reset point: the
         * render encoder is closed and the command buffer has not been rebuilt
         * yet, so resetMetalState can swap the command queue / clear caches
         * without racing an active encoder.  The request flag is set by the
         * Metal completion handler (GPURecovery.m) via release-store. */
        /* Completion workers latch recovery requests in the C++ owner on both
         * gates; consume them only at this GL-thread frame boundary. */
        if (areas.gpu_recovery_command_owner &&
            mglRenderCommandRecoveryTakeResetRequest(
                *areas.gpu_recovery_command_owner) == 1) {
            mglRendererRequestDeviceReset(renderer);
        }
        if (mglRendererConsumeDeviceReset(renderer)) {
            mglRendererResetMetalState(renderer);
        }

        if (!mglRenderPassEnsureWritableCommandBufferLocked(
                renderer, "mtlSwapBuffers")) {
            fprintf(stderr,
                    "MGL ERROR: Failed to obtain writable command buffer in mtlSwapBuffers\n");
            return;
        }

        const int swapInterval = areas.swap_interval;
        const int skipPresent =
            (swapInterval == 0) &&
            mglPlatformShellShouldSkipPresentForUnlockedSwap(renderer);

        if (mglPlatformShellDrawablePointer(renderer) == NULL) {
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present",
                    (unsigned long long)swapCall);
            }
            (void)mglPlatformShellApplyPendingDrawableSize(renderer);
            (void)mglRendererNextDrawable(renderer);
            if (traceSwap) {
                void *tex = mglRendererDrawableTexture(renderer);
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present drawable=%p tex=%p size=%lux%lu",
                    (unsigned long long)swapCall,
                    mglPlatformShellDrawablePointer(renderer), tex,
                    (unsigned long)(tex ? mglPdTextureInfo(tex).width : 0),
                    (unsigned long)(tex ? mglPdTextureInfo(tex).height : 0));
            }
        }

        if (mglPlatformShellDrawablePointer(renderer) == NULL) {
            fprintf(stderr,
                    "MGL WARNING: Drawable is NULL in mtlSwapBuffers, getting new drawable\n");
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present_retry",
                    (unsigned long long)swapCall);
            }
            (void)mglPlatformShellApplyPendingDrawableSize(renderer);
            (void)mglRendererNextDrawable(renderer);
            if (traceSwap) {
                void *tex = mglRendererDrawableTexture(renderer);
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present_retry drawable=%p tex=%p size=%lux%lu",
                    (unsigned long long)swapCall,
                    mglPlatformShellDrawablePointer(renderer), tex,
                    (unsigned long)(tex ? mglPdTextureInfo(tex).width : 0),
                    (unsigned long)(tex ? mglPdTextureInfo(tex).height : 0));
            }
            if (mglPlatformShellDrawablePointer(renderer) == NULL) {
                fprintf(stderr,
                        "MGL ERROR: Failed to obtain any drawable from Metal layer\n");
                return;
            }
        }

        void *rpColor0 = mglRenderGetRenderPassAttachmentTextureOwner(
            commandState->renderPassStateOwner,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
        void *drawableTexture = mglRendererDrawableTexture(renderer);
        if (!skipPresent) {
            mglSwapCopyRenderPassColorToDrawableIfNeeded(
                renderer, rpColor0, drawableTexture, swapCall, traceSwap ? 1 : 0);

            mglSwapScheduleTextureSampleDiagnostics(renderer, rpColor0,
                                                    drawableTexture, swapCall);
        }

        if (areas.layer == NULL) {
            fprintf(stderr,
                    "MGL ERROR: Metal layer is NULL, cannot present drawable\n");
            return;
        }

        MGLRenderCommandBufferState presentCommandState = {0};
        if (!mglRenderCommandBufferOwnerHasState(
                commandState->currentCommandBufferOwner,
                &presentCommandState)) {
            fprintf(stderr,
                    "MGL ERROR: No command buffer available for presentation\n");
            return;
        }

        const uint32_t bufferStatus = (uint32_t)presentCommandState.status;
        if (bufferStatus != MGL_PD_CB_NOT_ENQUEUED) {
            const uint64_t swapFinHit = ++s_pdSwapFinalizedBufferCount;
            if (swapFinHit <= 16ull || (swapFinHit % 500ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: mtlSwapBuffers found finalized command buffer (status: %ld), rotating (hit=%llu)\n",
                        (long)bufferStatus, (unsigned long long)swapFinHit);
            }
            mglRendererEndRenderEncodingLocked(renderer);
            (void)mglRenderPassNewCommandBufferLocked(renderer);
            if (!mglRenderCommandBufferOwnerHasState(
                    commandState->currentCommandBufferOwner,
                    &presentCommandState)) {
                fprintf(stderr,
                        "MGL ERROR: Failed to create new command buffer for presentation\n");
                return;
            }
        }

        MglPdPresentCtx presentCtx = {commandState->currentCommandBufferOwner,
                                      mglPlatformShellDrawablePointer(renderer), 0};
        if (!skipPresent) {
            if (!mglPlatformShellGuardedCallCtx(renderer, "drawable presentation",
                                                mglPdPresentTryBody, &presentCtx,
                                                NULL)) {
                if (presentCtx.result != -1) {
                    /* @catch (NSException *exception) */
                    fprintf(stderr,
                            "MGL ERROR: Critical drawable presentation failure\n");
                    (void)mglPlatformShellGuardedCall(
                        renderer, "command buffer cleanup",
                        mglRendererCleanupCommandBufferBody);
                }
                return;
            }
            if (traceSwap) {
                mglTraceLog("MGL TRACE swap.present call=%llu cbOwner=%p drawable=%p",
                            (unsigned long long)swapCall,
                            commandState->currentCommandBufferOwner,
                            presentCtx.drawable);
            }
        } else if (traceSwap) {
            mglTraceLog(
                "MGL TRACE swap.present.skipped call=%llu reason=unlocked_hidden",
                (unsigned long long)swapCall);
        }

        void *commandBufferToCommit =
            mglPassManagerDetachCurrentCommandBufferForSubmission(manager);
        /* Sweep the bound buffer maps so base/attrib/uniform/SSBO buffers that
         * were encoded this frame keep their pool slots pinned for the committed
         * command buffer (copy-on-write snapshot reuse). */
        BufferMapList *boundLists[3] = {
            &mglPdStateView(&areas)->vertex_buffer_map_list,
            &mglPdStateView(&areas)->fragment_buffer_map_list,
            &mglPdStateView(&areas)->compute_buffer_map_list,
        };
        for (int li = 0; li < 3; ++li) {
            for (GLuint mi = 0; mi < boundLists[li]->count; ++mi) {
                mglNoteBufferEncoded(boundLists[li]->buffers[mi].buf);
            }
        }
        MglPdCommitSwapCtx commitCtx = {commandBufferToCommit, swapCall,
                                        traceSwap ? 1 : 0};
        if (!mglPlatformShellGuardedCallCtx(renderer, "command buffer commit",
                                            mglPdCommitSwapTryBody, &commitCtx,
                                            NULL)) {
            /* @catch (NSException *exception) */
            fprintf(stderr, "MGL ERROR: Failed to commit command buffer\n");
            mglRendererRecordGPUError(renderer);
        }

        if (skipPresent) {
            /* Keep the current drawable: nothing was presented, so the surface
             * remains a valid render target for the next frame. */
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.reuse call=%llu stage=post_commit",
                    (unsigned long long)swapCall);
            }
        } else if (swapInterval == 0) {
            /* PERF-038: visible unlocked — drop consumed drawable; next frame
             * acquires in processGLState (not here). Never present NULL. */
            mglPlatformShellSetDrawable(renderer, NULL);
            /* PERF-038: prefetch off GL hot path for next frame acquire. */
            mglRendererPrefetchNextDrawable(renderer);
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.deferred call=%llu stage=post_commit",
                    (unsigned long long)swapCall);
            }
        } else {
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.begin call=%llu stage=post_commit",
                    (unsigned long long)swapCall);
            }
            (void)mglRendererNextDrawable(renderer);
            if (traceSwap) {
                void *tex = mglRendererDrawableTexture(renderer);
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.end call=%llu stage=post_commit drawable=%p tex=%p size=%lux%lu",
                    (unsigned long long)swapCall,
                    mglPlatformShellDrawablePointer(renderer), tex,
                    (unsigned long)(tex ? mglPdTextureInfo(tex).width : 0),
                    (unsigned long)(tex ? mglPdTextureInfo(tex).height : 0));
            }
            if (mglPlatformShellDrawablePointer(renderer) == NULL) {
                fprintf(stderr,
                        "MGL WARNING: Failed to get next drawable in mtlSwapBuffers\n");
                return;
            }
        }
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: Failed to create post-swap command buffer\n");
            return;
        }
        mglRendererSetDefaultDrawableWritten(renderer, 0);
        mglMarkRendererDirtyBits(areas.ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        if (timeSwap) {
            const double swapElapsedUs =
                (mglTraceClockNS() - swapStartNS) / 1000.0;
            if (traceSwap) {
                mglTraceLog("MGL TRACE swap.end call=%llu elapsed=%.1fus",
                            (unsigned long long)swapCall, swapElapsedUs);
                mglLogStateSnapshot("swap.exit.ok", areas.ctx,
                                    commandState->currentCommandBufferOwner,
                                    commandState->currentRenderEncoderOwner,
                                    commandState->renderPassStateOwner,
                                    areas.drawable);
            } else if (swapElapsedUs >= 25000.0) {
                mglTraceLog("MGL TRACE swap.slow call=%llu elapsed=%.1fus",
                            (unsigned long long)swapCall, swapElapsedUs);
            }
        }
    } else if (kMglPdVerboseFrameLoopLogs || traceSwap) {
        fprintf(stderr,
                "MGL INFO: mtlSwapBuffers skipped present because draw_buffer is GL_NONE\n");
    }

    /* Perf summary: snapshot + reset per-frame counters at the swap boundary.
     * Runs on every normal exit path (present + GL_NONE skip).  Early-return
     * error paths intentionally skip this so their counters roll into the next
     * successful frame. */
    if (mglPerfSummaryEnabled()) {
        const double now = mglTraceNowSeconds();
        static _Atomic double s_last_swap_time = 0.0;
        double interval = 0.0;
        const double prev =
            atomic_load_explicit(&s_last_swap_time, memory_order_relaxed);
        if (prev > 0.0) interval = (now - prev) * 1000.0;
        atomic_store_explicit(&s_last_swap_time, now, memory_order_relaxed);
        mglPrintPerfSummary(interval);
        mglResetPerfCounters();
    }
}
