/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+RenderPass.m
// Render pass lifecycle methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "mgl_clear_buffer_ops.h" /* draw-buffer creators (log 184) */
#import "mgl_render_encoder_ops.h" /* newRenderEncoder + layers (log 186) */
#import "mgl_pso_build_ops.h" /* PSO cache-miss build (log 187) */
#include "mgl_stage_encode_drivers.h" /* stage binding drivers (log 131) */
#include "mgl_draw_encode.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_pass_sync_ops.h" /* the sync/close leaves are C (log 191) */
#include "mgl_trace_strategy.h"
#include "mgl_gpu_recovery.h"
#include "mgl_binding_state_ops.h"
#include "mgl_vertex_layout.h"  /* vertex descriptor / blend cache */
#include "mgl_attachment_binding.h"  /* FBO attachment bind */
#include "mgl_draw_mode.h"
#import "MGLRenderer+DrawSupportUtil.h"
#include "mgl_blit_sampled_copy.h"
#include "mgl_batch_issue.h"
#include "mgl_texture_bind.h"
#include "mgl_buffer_map.h"  /* mglRendererBindMTLTexture (was -bindMTLTextureLocked:) */
#import "MGLRenderer+RenderPass_Private.h"
#include "mgl_air_loader.h"     /* AIR metallib loader. */
#include "mgl_aux_assets.h"
#include "mgl_renderer_backend.h"
#include "mgl_env_flag.h"
#include "mgl_byte_hash.h"
#include "mgl_renderer_ports.h"  /* C ports of renderer accessors (T4) */
#include "mgl_shader_abi.h"
#include "mgl_program_reflection.h"
#include "mgl_draw_tess.h"
#include "mgl_draw_gs.h"
#include "mgl_render.h"
#include "mgl_render_pass_plan.h"
#include "mgl_render_pass_clear.h"   /* O3.1 clear-value plan (mglRenderPassPlanClearValues) */
#include "mgl_program_resource.h"    /* per-stage builtin usage mask (gl_* scans retired) */

#import <objc/message.h>

typedef struct MGLRenderPassClearColorValue {
    double red;
    double green;
    double blue;
    double alpha;
} MGLRenderPassClearColorValue;


/* VS-only + GL_RASTERIZER_DISCARD cannot leave Metal rasterization disabled:
 * AGX drops vertex texture/SSBO stores. A no-op fragment keeps rasterization
 * on while color write masks stay cleared (see below). */

typedef NS_ENUM(uint32_t, MGLStubFSValueClass) {
    MGLStubFSFloat = 0,
    MGLStubFSInt,
    MGLStubFSUint,
};

static id mglRasterizerDiscardStubFragmentFunctionForClass(
    MGLStubFSValueClass valueClass)
{
    static id s_fs[MGLStubFSUint + 1] = { nil, nil, nil };
    static dispatch_once_t once[MGLStubFSUint + 1];

    dispatch_once(&once[valueClass], ^{
        void *fs = NULL;
        char err[256] = {0};
        if (valueClass == MGLStubFSFloat) {
            /* Precompiled aux asset (no runtime source compile). */
            const MGLAuxShaderAsset *safe =
                mglAuxShaderAssetFind("safe_fallback");
            void *vs = NULL;
            if (!safe || !safe->data || safe->size == 0 ||
                mglRenderCreateAuxFunctions(
                    safe->data, safe->size, safe->hash,
                    "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
                    &vs, &fs, err, sizeof(err)) != 0 || !fs) {
                NSLog(@"MGL ERROR: discard stub FS unavailable: %s",
                      err[0] ? err : "asset missing");
                if (vs) {
                    (void)(__bridge_transfer id)vs;
                }
                return;
            }
            (void)(__bridge_transfer id)vs;
        } else {
            /* Integer-format targets reject a float4 output, and no
             * precompiled integer stub asset ships in the aux table.
             * Compile the integer zero stub at runtime through the
             * self-hosted GLSL->AIR backend (the same path real programs
             * take; its fragment output carries the correct
             * air.render_target int/uint type). */
            static const char *s_stubSource[MGLStubFSUint + 1] = {
                NULL,
                "#version 330\n"
                "out ivec4 mgl_stub_color_int;\n"
                "void main() { mgl_stub_color_int = ivec4(0); }\n",
                "#version 330\n"
                "out uvec4 mgl_stub_color_uint;\n"
                "void main() { mgl_stub_color_uint = uvec4(0u); }\n",
            };
            unsigned char *bytes = NULL;
            size_t size = 0;
            if (mglShaderCompileGLSL(
                    s_stubSource[valueClass], MGL_STAGE_FRAGMENT,
                    &bytes, &size, err, sizeof(err)) != 0 || !bytes) {
                NSLog(@"MGL ERROR: stub FS compile failed: %s",
                      err[0] ? err : "unknown");
                return;
            }
            /* mglRenderCreateAuxFunctions supports fragment-only blobs by
             * accepting a NULL vertex entry, but the vertex output argument
             * itself is still required so the API can publish both results.
             * Passing NULL here made every integer render target fail with
             * "bad args" before the stub function was even looked up. */
            void *unusedVertex = NULL;
            if (mglRenderCreateAuxFunctions(
                    bytes, size, 0u, NULL, "main",
                    &unusedVertex, &fs, err, sizeof(err)) != 0 || !fs) {
                NSLog(@"MGL ERROR: stub FS function load failed: %s",
                      err[0] ? err : "unknown");
                if (unusedVertex) {
                    (void)(__bridge_transfer id)unusedVertex;
                }
                free(bytes);
                return;
            }
            if (unusedVertex) {
                (void)(__bridge_transfer id)unusedVertex;
            }
            free(bytes);
        }
        s_fs[valueClass] = (__bridge_transfer id)fs;
    });
    return s_fs[valueClass];
}

/* C-callable bridge for the C pipeline-descriptor host (log 172): the stub
 * factory itself stays in Objective-C (dispatch_once + blocks). */
void *mglRenderPassDiscardStubFragmentFunction(uint32_t valueClass)
{
    return (__bridge void *)mglRasterizerDiscardStubFragmentFunctionForClass(
        (MGLStubFSValueClass)valueClass);
}


static bool mglRenderPassGetPersistentState(
    const MGLCommandState *commandState,
    MGLRenderPassState *stateOut)
{
    return commandState && stateOut && commandState->renderPassStateOwner &&
           mglRenderGetRenderPassStateOwner(
               commandState->renderPassStateOwner, stateOut) == 0;
}

static bool mglRenderPassGetPersistentAttachmentState(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MGLRenderPassAttachmentState *attachmentOut)
{
    if (!attachmentOut) return false;
    MGLRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return false;
    switch (mglRenderPassAttachmentClass(attachmentKind)) {
        case 1:
            if (!mglRenderPassColorAttachmentIndexValid(
                    (uint32_t)colorIndex, MAX_COLOR_ATTACHMENTS))
                return false;
            *attachmentOut = state.color[colorIndex].attachment;
            return true;
        case 2:
            *attachmentOut = state.depth.attachment;
            return true;
        case 3:
            *attachmentOut = state.stencil.attachment;
            return true;
        default:
            return false;
    }
}

static const MGLRenderPassAttachmentState *
mglRenderPassAttachmentStateFromSnapshot(
    const MGLRenderPassState *state,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    if (!state) return NULL;
    switch (mglRenderPassAttachmentClass(attachmentKind)) {
        case 1:
            return mglRenderPassColorAttachmentIndexValid(
                       (uint32_t)colorIndex, MAX_COLOR_ATTACHMENTS)
                ? &state->color[colorIndex].attachment : NULL;
        case 2:
            return &state->depth.attachment;
        case 3:
            return &state->stencil.attachment;
        default:
            return NULL;
    }
}

static id mglRenderPassTextureFromSnapshot(
    const MGLRenderPassState *state,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    const MGLRenderPassAttachmentState *attachment =
        mglRenderPassAttachmentStateFromSnapshot(
            state, attachmentKind, colorIndex);
    return attachment && attachment->texture
        ? (__bridge id)attachment->texture : nil;
}

/* RenderPassStateOwner is the writer of record for every attachment field. */




/* Owner-first load/store actions and clear values for one attachment. */


/* Single-value variants use zero or the caller-provided explicit default. */





static void mglRenderPassSetPersistentActions(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t loadAction,
    uint32_t storeAction)
{
    if (!commandState) return;
    MGLRenderPassAttachmentState state = {0};
    if (!mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &state)) {
        return;
    }
    if (commandState->renderPassStateOwner) {
        mglRenderSetRenderPassStateAttachmentActions(
            commandState->renderPassStateOwner, attachmentKind,
            (uint32_t)colorIndex, (uint32_t)loadAction,
            (uint32_t)storeAction, state.store_action_options);
    }
}

/* Geometry shaders always execute through the AIR compute expansion.
 * A source-string "passthrough" skip used to drop the GS stage (plain
 * VS->FS), which left invocation / primitives-emitted queries at zero. */

static bool mglLoadAIRMainFunction(const unsigned char *bytes,
                                   size_t size,
                                   id __strong *libraryOut,
                                   id __strong *functionOut,
                                   char *errorText,
                                   size_t errorCap)
{
    if (libraryOut) *libraryOut = nil;
    if (functionOut) *functionOut = nil;
    if (!bytes || size == 0u || !libraryOut || !functionOut) {
        if (errorText && errorCap) snprintf(errorText, errorCap, "bad args");
        return false;
    }
    void *libraryHandle = NULL;
    void *functionHandle = NULL;
    if (mglRenderLoadAIRMainFunction(
            bytes, size, &libraryHandle, &functionHandle,
            errorText, errorCap) != 0 || !libraryHandle || !functionHandle) {
        return false;
    }
    id library =
        (__bridge_transfer id)libraryHandle;
    id function =
        (__bridge_transfer id)functionHandle;
    *libraryOut = library;
    *functionOut = function;
    return true;
}

@implementation MGLRenderer (RenderPass)

/* Matrix column count / row count for stage-out record layout (GL 4.6
 * §4.4.1: one location per column).  Returns 0 for non-matrix types. */
/* Integer varyings are stored as SIToFP/UIToFP float carriers in the
 * stage-out record (see air backend).  The passthrough VS therefore
 * declares float attributes and forwards the float swizzle as-is; the
 * fragment stage converts with fptosi/fptoui.  GLSL still requires the
 * `flat` qualifier on integer varyings. */


/* The stage-out record stores every varying as a full vec4 slot, so a GS
 * output's reflected gl_type is promoted to the record width.  When the
 * fragment shader consumes the varying with a narrower declared type
 * (legal GL: GS out vec3 + FS in vec3), the passthrough VS must declare
 * the interface with the fragment type -- Metal rejects a pipeline whose
 * vertex output type differs from the fragment input. */
/* TES-compute twin of ensureAIRGeometryPassthroughFunctionForProgram: the
 * isolines/point-mode TES kernel expands one vertex record per work item,
 * so the raster stage is a GLSL passthrough vertex reading the same
 * record layout as the GS expansion (position at 0, point size at 1,
 * varyings at MGL_AIR_PER_VERTEX_STRIDE + location*16).  The records come
 * from the TES stage output resource list. */





- (bool)restoreRenderEncoderAfterTextureUploadForDraw:(const char *)reason
{
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager->state->currentRenderEncoderOwner) == 1) {
        return true;
    }
    MGLRenderPassState passState = {0};
    bool hasPassState =
        mglRenderPassGetPersistentState(_renderPassManager->state, &passState);
    if (!ctx || !hasPassState) {
        return false;
    }

    static uint64_t s_restoreAfterTextureUploadCount = 0;
    uint64_t hit = ++s_restoreAfterTextureUploadCount;
    if (hit <= 16ull || (hit % 2048ull) == 0ull) {
        NSLog(@"MGL TEXTURE UPLOAD closed render encoder; restoring for draw reason=%s hit=%llu",
              reason ? reason : "(null)",
              (unsigned long long)hit);
    }

    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            (__bridge void *)self,
            reason ? reason : "restore_render_encoder_after_texture_upload")) {
        return false;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id texture = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i);
        if (texture) {
            mglRenderPassSetPersistentActions(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i,
                MGLLoadActionLoad, MGLStoreActionStore);
        }
    }
    id depthTexture = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    if (depthTexture) {
        mglRenderPassSetPersistentActions(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionLoad, MGLStoreActionStore);
    }
    id stencilTexture = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
    if (stencilTexture) {
        mglRenderPassSetPersistentActions(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionLoad, MGLStoreActionStore);
    }

    @try {
        id renderEncoder =
            (__bridge id)mglPassManagerCreateRenderEncoder(_renderPassManager);
        mglPassManagerInstallRenderEncoder(_renderPassManager, (__bridge void *)renderEncoder);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload failed to create encoder: %@",
              exception.reason);
        mglPassManagerClearCurrentRenderEncoder(_renderPassManager);
        mglRendererRecordGPUError((__bridge void *)self);
        return false;
    }
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager->state->currentRenderEncoderOwner) != 1) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload returned nil encoder reason=%s",
              reason ? reason : "(null)");
        mglRendererRecordGPUError((__bridge void *)self);
        return false;
    }
    mglRenderSetRenderEncoderOwnerLabel(
        _renderPassManager->state->currentRenderEncoderOwner,
        "GL Render Encoder");
    /* When trace is disabled, skip the full-struct memset and trace call
     * and clear only the functional flag fields. */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             reason ? reason : "restore_render_encoder_after_texture_upload",
                                             _resourceFallback.fragmentTextureTraceBindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             _pipelineCache.state->pipelineProgramName);
        memset(_resourceFallback.fragmentTextureTraceBindings, 0,
               sizeof(_resourceFallback.fragmentTextureTraceBindings));
    } else {
        mglClearFragmentTextureTraceFunctionalFlags(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
    }
    mglPassManagerUpdateRenderPassIdentityForContext(_renderPassManager, ctx);
    mglRenderPassUpdateCurrentRenderEncoder((__bridge void *)self);

    if (!_pipelineCache.state->pipelineState) {
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    @try {
        if (mglRenderBindingSetPipelineIfNeededForOwner(
                _bindingStateOwner,
                _renderPassManager->state->currentRenderEncoderOwner,
                _pipelineCache.state->pipelineState) > 0) {
            MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload failed to bind pipeline: %@",
              exception.reason);
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    RETURN_FALSE_ON_FAILURE(mglRendererMapBuffersToMTL((__bridge void *)self));
    MGLEncodeContext encCtx = {
        .render_encoder_owner = _renderPassManager->state->currentRenderEncoderOwner,
    };
    RETURN_FALSE_ON_FAILURE(mglStageEncodeBindVertexBuffers((__bridge void *)self, &encCtx));
    RETURN_FALSE_ON_FAILURE(mglStageEncodeBindFragmentBuffers((__bridge void *)self, &encCtx));
    return true;
}


-(bool)bindMTLProgram:(Program *)ptr
{
    METAL_LOCK();
    bool result = [self bindMTLProgramLocked:ptr];
    METAL_UNLOCK();
    return result;
}

-(bool)bindMTLProgramLocked:(Program *)ptr
{
    if (ptr->dirty_bits & DIRTY_PROGRAM)
    {
        /* Metal libraries/functions are linked Program products and are
         * invalidated by clearStageCompileState during relink. DIRTY_PROGRAM
         * also covers pre-link state changes, which must not discard the
         * currently linked executable. */
        ptr->dirty_bits &= ~DIRTY_PROGRAM;
    }

    int failedStage = -1;
    char bindError[256] = {0};
    int bindResult = mglRenderBindAIRProgram(
        ptr, &failedStage, bindError, sizeof(bindError));
    if (bindResult == MGL_RENDER_AIR_PROGRAM_BOUND) {
        return true;
    }
    if (bindResult == MGL_RENDER_AIR_PROGRAM_ERROR) {
        NSLog(@"MGL ERROR: Failed to bind AIR program=%u stage=%d: %s",
              (unsigned)ptr->name, failedStage,
              bindError[0] ? bindError : "?");
        return false;
    }

	    // Compile linked Program stages on demand.
	    for(int i=_VERTEX_SHADER; i<_MAX_SHADER_TYPES; i++)
	    {
	        Shader *shader;
	        shader = ptr->shader_slots[i];

        if (shader)
        {
            if (mglDrawGsStageShouldBlockDraw(
                    i, (uint32_t)ptr->gs_route, ptr->modules[i].metallib_bytes,
                    (uint32_t)ptr->modules[i].metallib_size)) {
                static uint64_t s_geometryShaderMetalSkipCount = 0;
                uint64_t hit = ++s_geometryShaderMetalSkipCount;
                if (hit <= 16ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL WARNING: Blocking draw for unsupported geometry shader program=%u hit=%llu",
                          (unsigned)ptr->name,
                          (unsigned long long)hit);
                }
                return false;
            }
            if (ptr->modules[i].metallib_bytes && ptr->modules[i].metallib_size > 0) {
                /* AIR path: the stage was compiled by the self-hosted
                 * frontend into a metallib blob; load it directly. */
                if (ptr->modules[i].mtl_library == NULL || ptr->modules[i].mtl_function == NULL) {
                    mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_function);
                    mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_library);
                    id library = nil;
                    id function = nil;
                    char loadError[256] = {0};
                    if (!mglLoadAIRMainFunction(
                            ptr->modules[i].metallib_bytes,
                            ptr->modules[i].metallib_size, &library, &function,
                            loadError, sizeof loadError)) {
                        NSLog(@"MGL ERROR: Failed to load AIR metallib program=%u stage=%d: %s",
                              (unsigned)ptr->name, i,
                              loadError[0] ? loadError : "?");
                        return false;
                    }
                    ptr->modules[i].mtl_library = (void *)CFBridgingRetain(library);
                    ptr->modules[i].mtl_function = (void *)CFBridgingRetain(function);
                }
                if (mglRenderVertexCaptureNeedsLoad(
                        i, ptr->modules[i].metallib_tess_capture_bytes,
                        ptr->modules[i].mtl_tess_capture_library,
                        ptr->modules[i].mtl_tess_capture_function)) {
                    id library = nil;
                    id function = nil;
                    char loadError[256] = {0};
                    if (!mglLoadAIRMainFunction(
                            ptr->modules[i].metallib_tess_capture_bytes,
                            ptr->modules[i].metallib_tess_capture_size,
                            &library, &function, loadError,
                            sizeof loadError)) {
                        NSLog(@"MGL ERROR: Failed to load AIR tess VS capture program=%u: %s",
                              (unsigned)ptr->name,
                              loadError[0] ? loadError : "?");
                        return false;
                    }
                    ptr->modules[i].mtl_tess_capture_library =
                        (void *)CFBridgingRetain(library);
                    ptr->modules[i].mtl_tess_capture_function =
                        (void *)CFBridgingRetain(function);
                }
                if (mglRenderVertexCaptureNeedsLoad(
                        i, ptr->modules[i].metallib_cull_capture_bytes,
                        ptr->modules[i].mtl_cull_capture_library,
                        ptr->modules[i].mtl_cull_capture_function)) {
                    id library = nil;
                    id function = nil;
                    char loadError[256] = {0};
                    if (!mglLoadAIRMainFunction(
                            ptr->modules[i].metallib_cull_capture_bytes,
                            ptr->modules[i].metallib_cull_capture_size,
                            &library, &function, loadError,
                            sizeof loadError)) {
                        NSLog(@"MGL ERROR: Failed to load AIR cull-distance "
                              "capture program=%u: %s",
                              (unsigned)ptr->name,
                              loadError[0] ? loadError : "?");
                        return false;
                    }
                    ptr->modules[i].mtl_cull_capture_library =
                        (void *)CFBridgingRetain(library);
                    ptr->modules[i].mtl_cull_capture_function =
                        (void *)CFBridgingRetain(function);
                }
            } else {
                NSLog(@"MGL ERROR: Program %u stage %d has no AIR metallib",
                      (unsigned)ptr->name, i);
                return false;
            }
        }
    }

	    return true;
	}

/* The viewport/scissor block moved with the encoder-state block: see
 * mglRenderPassUpdateViewportAndScissorLocked (log 192). */








#pragma mark pipeline descriptor
/* Build the renderer pipeline as C ABI value-state. Color/depth/stencil,
 * blend, vertex layout, tessellation, rasterization, topology, and sample
 * count are consumed by the C++ pipeline builder. */
#pragma mark vertex descriptor







// ULTIMATE FAILSAFE: Emergency Metal state reset to recover from corruption
- (bool) processGLState: (bool) draw_command
{
    METAL_LOCK();
    bool result = mglRenderPassProcessGLStateLocked(
        (__bridge void *)self, draw_command ? 1 : 0) != 0;
    METAL_UNLOCK();
    return result;
}

/*
 * Dirty state domain processing extracted from processGLStateLocked:.
 * Handles all dirty-bits dispatch: DIRTY_FBO, DIRTY_STATE, DIRTY_PROGRAM/
 * VAO/BUFFER_BASE_STATE, DIRTY_TEX, DIRTY_VAO/BUFFER/RENDER_STATE, and the
 * pipeline sync call. Returns false on failure (caller should skip this
 * draw), true on success.
 */

/*
 * Render pass descriptor and pipeline format validation extracted from
 * processGLStateLocked:. Validates render-pass attachments and checks
 * pipeline/pass color, depth, and stencil format compatibility. Returns
 * false to skip the draw on validation failure, true to continue.
 */
/* The Pipeline Sync domain (this comment's block) is C now (log 189):
 * mglRenderPassSyncPipelineState in mgl_pso_build_ops.c owns the PSO
 * dedup fast path, the two breakers, the two-level cache lookup and the
 * cache-miss build.  The port wrapper is retired with it. */

/* Build final/simple/safe PSO variants from value-state in the C++ builder.
 * The same owner also handles descriptor caching and binary archives. */
/* Store the compiled pipeline and its value-state descriptor. */
/* Bind spvBufferSizeConstants for runtime-sized SSBO arrays in vertex/fragment
 * stages.  The AIR backend emits code that reads uint32 byte-sizes from a
 * constant uint* buffer at MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a shader uses
 * .length() on unsized SSBO arrays.  The render encoder has separate buffer
 * tables for vertex and fragment, so we bind a size buffer for each stage
 * that needs it. */

-(void) flushCommandBuffer: (bool) finish
{
    METAL_LOCK();
    mglRenderPassFlushCommandBufferLocked((__bridge void *)self,
                                        finish ? 1 : 0);
    METAL_UNLOCK();

    /* The C++ command owner retains the last accepted submission. Waiting
     * through its value-state API keeps completion lifetime out of the
     * renderer's ObjC ivar mirror and preserves the old outside-lock wait. */
    if (finish) {
        MGLRenderCommandBufferState finishState = {0};
        int waitResult = mglPassManagerWaitForLastSubmittedCommandBuffer(_renderPassManager, &finishState);
        if (waitResult < 0 || finishState.has_error) {
            NSLog(@"MGL ERROR: owner waitUntilCompleted failed status=%u domain=%s code=%lld",
                  finishState.status, finishState.error_domain,
                  (long long)finishState.error_code);
        }
    }
}




- (BOOL)prepareRenderPassIfFBOChanged:(MGLDrawBatch *)batch
                              context:(GLMContext)glm_ctx
                          replayError:(GLenum *)replayError
{
    if (!(MGL_STATE(glm_ctx)->dirty_bits & DIRTY_FBO))
        return YES;

    /* Orchestrator-driven FBO rotation (Orchestrator-driven FBO rotation) delegates to the shared
     * RenderPass Sync unit (RenderPass Sync domain), surfacing any GL error as replayError
     * so the batch is skipped rather than drawn against a stale pass. */
    if (!mglRenderPassSyncRenderPassStateForContext((__bridge void *)self, glm_ctx)) {
        if (!mglRenderErrorIsNone((uint32_t)MGL_STATE(glm_ctx)->error))
            *replayError = MGL_STATE(glm_ctx)->error;
        return NO;
    }
    return YES;
}


/* === moved from MGLRenderer+DrawSupport.m (category merge) =================
 * Both need the render-pass machinery that lives here: the first calls
 * -flushCommandBuffer: / -processGLState:, the second
 * -newRenderEncoderLockedWithReason:. */

- (BOOL)prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label
{
    if (!drawCtx) {
        NSLog(@"MGL WARNING: %s skipped because context is NULL",
              label ? label : "indirect emulation");
        return NO;
    }

    /* The C draw-indirect frontends already flush pending command buffers before
     * dispatching into these Metal entry points. If processGLState has just
     * rebuilt a render encoder, keep it; a second flush can discard the fresh
     * pass and make state restoration fail for CPU-emulated indirect modes. */
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager->state->currentRenderEncoderOwner) == 1) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager->state->currentRenderEncoderOwner) != 1) {
        NSLog(@"MGL WARNING: %s skipped because CPU-read synchronization left no render encoder",
              label ? label : "indirect emulation");
        return NO;
    }
    return YES;
}


@end
