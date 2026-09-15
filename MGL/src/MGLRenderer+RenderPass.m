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

static MGLRenderTextureInfo mglRenderPassTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

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
static id mglRenderPassAttachmentTextureFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &attachment)) {
        return attachment.texture
            ? (__bridge id)attachment.texture : nil;
    }
    return nil;
}

static id mglRenderPassColorTextureFor(
    const MGLCommandState *commandState, NSUInteger colorIndex)
{
    return mglRenderPassAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
        colorIndex);
}

static id mglRenderPassDepthTextureFor(
    const MGLCommandState *commandState)
{
    return mglRenderPassAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u);
}

static id mglRenderPassStencilTextureFor(
    const MGLCommandState *commandState)
{
    return mglRenderPassAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u);
}

/* Owner-first load/store actions and clear values for one attachment. */
static BOOL mglRenderPassActionsFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t *loadActionOut,
    uint32_t *storeActionOut,
    uint64_t *storeActionOptionsOut)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &attachment)) {
        if (loadActionOut) *loadActionOut = (uint32_t)attachment.load_action;
        if (storeActionOut) *storeActionOut = (uint32_t)attachment.store_action;
        if (storeActionOptionsOut) {
            *storeActionOptionsOut = attachment.store_action_options;
        }
        return YES;
    }
    return NO;
}

static BOOL mglRenderPassRenderTargetSizeFor(
    const MGLCommandState *commandState,
    uint64_t *widthOut,
    uint64_t *heightOut)
{
    MGLRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return NO;
    if (widthOut) *widthOut = state.render_target_width;
    if (heightOut) *heightOut = state.render_target_height;
    return YES;
}

/* Single-value variants use zero or the caller-provided explicit default. */
static NSUInteger mglRenderPassRenderTargetWidthFor(
    const MGLCommandState *commandState)
{
    uint64_t width = 0;
    if (mglRenderPassRenderTargetSizeFor(commandState, &width, NULL)) {
        return (NSUInteger)width;
    }
    return 0;
}

static NSUInteger mglRenderPassRenderTargetHeightFor(
    const MGLCommandState *commandState)
{
    uint64_t height = 0;
    if (mglRenderPassRenderTargetSizeFor(commandState, NULL, &height)) {
        return (NSUInteger)height;
    }
    return 0;
}

static uint32_t mglRenderPassLoadActionFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t fallback)
{
    uint32_t action = 0u;
    if (mglRenderPassActionsFor(commandState, attachmentKind, colorIndex,
                                &action, NULL, NULL)) {
        return (uint32_t)action;
    }
    return fallback;
}

static uint32_t mglRenderPassStoreActionFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t fallback)
{
    uint32_t action = 0u;
    if (mglRenderPassActionsFor(commandState, attachmentKind, colorIndex,
                                NULL, &action, NULL)) {
        return (uint32_t)action;
    }
    return fallback;
}

static void mglRenderPassSetPersistentDimensions(
    const MGLCommandState *commandState,
    NSUInteger width,
    NSUInteger height)
{
    if (commandState && commandState->renderPassStateOwner) {
        mglRenderSetRenderPassStateDimensions(
            commandState->renderPassStateOwner, width, height);
    }
}

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
    [self updateCurrentRenderEncoder];

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

- (void) updateCurrentRenderEncoder
{
    GLMState *state = MGL_STATE(ctx);
    BOOL hasConfiguredRenderPass =
        _renderPassManager->state->renderPassStateOwner != NULL;
    BOOL passHasDepthAttachment =
        (hasConfiguredRenderPass &&
         mglRenderPassDepthTextureFor(_renderPassManager->state) != nil);
    BOOL passHasStencilAttachment =
        (hasConfiguredRenderPass &&
         mglRenderPassStencilTextureFor(_renderPassManager->state) != nil);
    BOOL useDepthState = mglRenderUseDepthState(
                             state->caps.depth_test ? 1 : 0,
                             passHasDepthAttachment ? 1 : 0) != 0;
    BOOL useStencilState = mglRenderUseStencilState(
                               state->caps.stencil_test ? 1 : 0,
                               passHasStencilAttachment ? 1 : 0) != 0;

    if (state->caps.depth_test && !passHasDepthAttachment) {
        static uint64_t s_missingDepthAttachmentCount = 0;
        uint64_t hit = ++s_missingDepthAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: depth test/write requested without depth attachment, disabling depth for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (state->caps.stencil_test && !passHasStencilAttachment) {
        static uint64_t s_missingStencilAttachmentCount = 0;
        uint64_t hit = ++s_missingStencilAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: stencil test requested without stencil attachment, disabling stencil for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (useDepthState || useStencilState)
    {
        MGLRenderDepthStencilDescriptorState dsDesc = {0};
        /* MTLDepthStencilDescriptor initializes depth comparison to Always.
         * Preserve that default for stencil-only passes; leaving the value
         * zero would map to Never and reject every fragment before stencil. */
        dsDesc.depth_compare_function = MGLCompareFunctionAlways;

        if (useDepthState)
        {
            uint32_t depthFunc = mglRenderRepairDepthFunc(
                (uint32_t)state->var.depth_func);
            if (depthFunc != (uint32_t)state->var.depth_func) {
                mglLogRenderStateRepair("depth_func", state->var.depth_func,
                                        (GLenum)depthFunc);
                state->var.depth_func = (GLenum)depthFunc;
                mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
            }

            dsDesc.depth_compare_function = (uint32_t)
                mglMTLCompareFunctionForGL(state->var.depth_func,
                                           MGLCompareFunctionLess,
                                           "depth");
            dsDesc.depth_write_enabled = mglRenderDepthWriteEnabled(
                state->var.depth_writemask ? 1 : 0, 0);
        }

        /* GL_RASTERIZER_DISCARD / VS capture: no fragment is produced, so
         * depth/stencil writes must not mutate attachments (color masks are
         * cleared separately in the pipeline descriptor path). */
        const BOOL suppressDepthStencilWrites =
            mglRenderSuppressDepthStencilWrites(
                state->caps.rasterizer_discard ? 1 : 0,
                _tessellation.tessVertexCaptureActive ? 1 : 0,
                _tessellation.cullDistanceCaptureActive ? 1 : 0) != 0;
        if (suppressDepthStencilWrites) {
            dsDesc.depth_write_enabled = mglRenderDepthWriteEnabled(
                state->var.depth_writemask ? 1 : 0, 1);
        }

        if (useStencilState)
        {
            if (mglTraceLogIsEnabled()) {
                mglTraceLog("STENCIL_STATE fbo=%u func=0x%x back=0x%x ref=%u backRef=%u readMask=0x%x backReadMask=0x%x writeMask=0x%x attachment=%p layered=%d",
                            (unsigned)mglRendererSafeFramebufferName(ctx),
                            (unsigned)state->var.stencil_func,
                            (unsigned)state->var.stencil_back_func,
                            (unsigned)state->var.stencil_ref,
                            (unsigned)state->var.stencil_back_ref,
                            (unsigned)state->var.stencil_value_mask,
                            (unsigned)state->var.stencil_back_value_mask,
                            (unsigned)state->var.stencil_writemask,
                            mglRenderPassStencilTextureFor(_renderPassManager->state),
                            (int)(MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->stencil.layered : 0));
            }
            {
                uint32_t stencilFunc = mglRenderRepairStencilFunc(
                    (uint32_t)state->var.stencil_func);
                if (stencilFunc != (uint32_t)state->var.stencil_func) {
                    mglLogRenderStateRepair("stencil_func", state->var.stencil_func,
                                            (GLenum)stencilFunc);
                    state->var.stencil_func = (GLenum)stencilFunc;
                    mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
                }

                dsDesc.front.present = 1u;
                dsDesc.front.compare_function = (uint32_t)
                    mglMTLCompareFunctionForGL(state->var.stencil_func,
                                               MGLCompareFunctionAlways,
                                               "front-stencil");
                if (mglEnvFlagEnabled("MGL_FORCE_STENCIL_ALWAYS")) {
                    dsDesc.front.compare_function = MGLCompareFunctionAlways;
                }
                uint32_t failOp = 0u, depthFailOp = 0u, passOp = 0u;
                (void)mglRenderStencilOpFromGL((uint32_t)state->var.stencil_fail,
                                               &failOp);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_pass_depth_fail, &depthFailOp);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_pass_depth_pass, &passOp);
                dsDesc.front.stencil_failure_operation = failOp;
                dsDesc.front.depth_failure_operation = depthFailOp;
                dsDesc.front.depth_stencil_pass_operation = passOp;
                dsDesc.front.write_mask = mglRenderStencilWriteMask(
                    suppressDepthStencilWrites ? 1 : 0,
                    (uint32_t)state->var.stencil_writemask);
                dsDesc.front.read_mask = state->var.stencil_value_mask;
            }

            {
                uint32_t stencilBack = mglRenderRepairStencilFunc(
                    (uint32_t)state->var.stencil_back_func);
                if (stencilBack != (uint32_t)state->var.stencil_back_func) {
                    mglLogRenderStateRepair("stencil_back_func",
                                            state->var.stencil_back_func,
                                            (GLenum)stencilBack);
                    state->var.stencil_back_func = (GLenum)stencilBack;
                    mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
                }

                dsDesc.back.present = 1u;
                dsDesc.back.compare_function = (uint32_t)
                    mglMTLCompareFunctionForGL(state->var.stencil_back_func,
                                               MGLCompareFunctionAlways,
                                               "back-stencil");
                if (mglEnvFlagEnabled("MGL_FORCE_STENCIL_ALWAYS")) {
                    dsDesc.back.compare_function = MGLCompareFunctionAlways;
                }
                uint32_t backFail = 0u, backDepthFail = 0u, backPass = 0u;
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_back_fail, &backFail);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_back_pass_depth_fail,
                    &backDepthFail);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_back_pass_depth_pass,
                    &backPass);
                dsDesc.back.stencil_failure_operation = backFail;
                dsDesc.back.depth_failure_operation = backDepthFail;
                dsDesc.back.depth_stencil_pass_operation = backPass;
                dsDesc.back.write_mask = mglRenderStencilWriteMask(
                    suppressDepthStencilWrites ? 1 : 0,
                    (uint32_t)state->var.stencil_back_writemask);
                dsDesc.back.read_mask = state->var.stencil_back_value_mask;
            }
        }

        id dsState = [_pipelineCache depthStencilStateForValueState:&dsDesc];

        if (mglRenderBindingSetDepthStencilIfNeededForOwner(
                _bindingStateOwner,
                _renderPassManager->state->currentRenderEncoderOwner,
                (__bridge void *)dsState) > 0) {
        } else {
            MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
        }
        if (useStencilState) {
            mglRenderSetStencilReferenceValuesForOwner(
                _renderPassManager->state->currentRenderEncoderOwner,
                (uint32_t)state->var.stencil_ref,
                (uint32_t)state->var.stencil_back_ref);
        }
    }
    else
    {
        MGLRenderDepthStencilDescriptorState disabledDSDesc = {0};
        disabledDSDesc.depth_compare_function = MGLCompareFunctionAlways;
        disabledDSDesc.depth_write_enabled = 0u;

        id disabledDSState =
            [_pipelineCache depthStencilStateForValueState:&disabledDSDesc];
        if (disabledDSState) {
            if (mglRenderBindingSetDepthStencilIfNeededForOwner(
                    _bindingStateOwner,
                    _renderPassManager->state->currentRenderEncoderOwner,
                    (__bridge void *)disabledDSState) > 0) {
            } else {
                MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
            }
        }
    }

    {
        float bcRed   = state->var.blend_color[0];
        float bcGreen = state->var.blend_color[1];
        float bcBlue  = state->var.blend_color[2];
        float bcAlpha = state->var.blend_color[3];
        mglRenderBindingSetBlendColorIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager->state->currentRenderEncoderOwner,
            bcRed, bcGreen, bcBlue, bcAlpha);
    }

    /* GL_SAMPLE_MASK: Metal does not expose a per-draw sample mask setter on
     * MTLRenderCommandEncoder.  Sample coverage in Metal is controlled via
     * alpha-to-coverage and shader-side [[sample_mask]], neither of which
     * maps cleanly to GL_SAMPLE_MASK.  This remains a known limitation. */

    [self updateViewportAndScissorLocked];

    if (!mglRenderFrontFaceValid((uint32_t)state->var.front_face)) {
        uint32_t repaired = mglRenderFrontFaceOrCCW(
            (uint32_t)state->var.front_face);
        mglLogRenderStateRepair("front_face", state->var.front_face,
                                (GLenum)repaired);
        state->var.front_face = (GLenum)repaired;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }

    BOOL rtSampledCopyDraw = _renderPassManager->state->currentDrawUsesRTSampledCopy;
    BOOL defaultFramebufferSampledPass =
        mglRenderSkipCullForSampledPass(
            state->framebuffer ? 1 : 0, state->caps.depth_test ? 1 : 0,
            mglRendererGetProgramBindingCount(ctx, _FRAGMENT_SHADER,
                                              _SAMPLED_IMAGE_RES) > 0
                ? 1
                : 0,
            rtSampledCopyDraw ? 1 : 0) != 0 &&
        !rtSampledCopyDraw;

    uint32_t cull_mode = mglRenderCullModeFromGL(
        (state->caps.cull_face && !defaultFramebufferSampledPass &&
         !rtSampledCopyDraw)
            ? 1
            : 0,
        (uint32_t)state->var.cull_face_mode);
    mglRenderBindingSetCullIfNeededForOwner(
        _bindingStateOwner,
        _renderPassManager->state->currentRenderEncoderOwner, cull_mode);
    uint32_t _winding =
        mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                 !mglRenderClipOriginIsLowerLeft(
                                     (uint32_t)state->var.clip_origin));
    mglRenderBindingSetWindingIfNeededForOwner(
        _bindingStateOwner,
        _renderPassManager->state->currentRenderEncoderOwner,
        (uint32_t)_winding);

    if (state->caps.cull_face && defaultFramebufferSampledPass) {
        static uint64_t s_defaultSampledCullBypassCount = 0;
        uint64_t hit = ++s_defaultSampledCullBypassCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            mglTraceLog("MGL TRACE default sampled pass cull bypass hit=%llu program=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  (unsigned)(ctx ? state->program_name : 0u),
                  (unsigned)(ctx ? state->draw_buffer : 0u));
        }
    }
    if (state->caps.cull_face && rtSampledCopyDraw) {
        static uint64_t s_rtSampledCopyCullBypassCount = 0;
        uint64_t hit = ++s_rtSampledCopyCullBypassCount;
        if (hit <= 64ull || (hit % 256ull) == 0ull) {
            mglTraceLog("RT_SAMPLE_COPY_CULL_BYPASS hit=%llu program=%u pipelineProgram=%u fbo=%u rpFbo=%u depth(test=%d write=%d func=0x%x) blend=%d cullFace=0x%x frontFace=0x%x",
                        (unsigned long long)hit,
                        (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                        (unsigned)_pipelineCache.state->pipelineProgramName,
                        (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                        (unsigned)_renderPassManager->state->renderPassFramebufferName,
                        (ctx && state->caps.depth_test) ? 1 : 0,
                        (ctx && state->var.depth_writemask) ? 1 : 0,
                        (unsigned)(ctx ? state->var.depth_func : 0u),
                        (ctx && state->caps.blend) ? 1 : 0,
                        (unsigned)(ctx ? state->var.cull_face_mode : 0u),
                        (unsigned)(ctx ? state->var.front_face : 0u));
        }
    }

    if (state->caps.depth_clamp)
    {
        mglRenderSetDepthClipModeForOwner(
            _renderPassManager->state->currentRenderEncoderOwner,
            mglRenderDepthClipMode(state->caps.depth_clamp ? 1 : 0));
    }

    if (mglRenderPolygonOffsetEnabled(
            state->caps.polygon_offset_fill ? 1 : 0,
            state->caps.polygon_offset_line ? 1 : 0,
            state->caps.polygon_offset_point ? 1 : 0))
    {
        float _bias = state->var.polygon_offset_units;
        float _slope = state->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager->state->currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    }
    else
    {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager->state->currentRenderEncoderOwner,
            0.0f, 0.0f, 0.0f);
    }

    uint32_t triangleFillMode = mglRenderTriangleFillMode(
        (uint32_t)state->var.polygon_mode);
    if (!mglRenderPolygonModeValid((uint32_t)state->var.polygon_mode)) {
        uint32_t repaired = mglRenderPolygonModeOrFill(
            (uint32_t)state->var.polygon_mode);
        mglLogRenderStateRepair("polygon_mode", state->var.polygon_mode,
                                (GLenum)repaired);
        state->var.polygon_mode = (GLenum)repaired;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }
    mglBindingSetTriangleFillModeIfNeeded((__bridge void *)self, triangleFillMode);
}
/*
 * Viewport and scissor setup extracted from updateCurrentRenderEncoder.
 * Resolves render-pass dimensions, applies the scissor rect (with GL-to-Metal
 * origin conversion), and sets the viewport. Uses MGL_STATE(ctx) for
 * snapshot-based state access (Principle 2 compliance).
 */
- (void)updateViewportAndScissorLocked
{
    GLMState *state = MGL_STATE(ctx);
    // Metal validates viewport/scissor strictly against the active render pass dimensions.
    // Always derive pass size from the current attachments first (not from window drawable fallback).
    {
        static uint64_t s_encoderStateUpdateCount = 0;
        bool traceEncoderState = kMGLDiagnosticStateLogs || mglShouldTraceCall(++s_encoderStateUpdateCount);

        NSUInteger passWidth = 0;
        NSUInteger passHeight = 0;
        id passTexture = nil;

        /* The C++ owner is the authoritative configured-pass signal. */
        BOOL hasConfiguredRenderPass =
            _renderPassManager->state->renderPassStateOwner != NULL;
        if (hasConfiguredRenderPass) {
            passWidth = mglRenderPassRenderTargetWidthFor(_renderPassManager->state);
            passHeight = mglRenderPassRenderTargetHeightFor(_renderPassManager->state);

            if (passWidth == 0 || passHeight == 0) {
                for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                    id candidate = mglRenderPassColorTextureFor(_renderPassManager->state, i);
                    if (candidate) {
                        passTexture = candidate;
                        break;
                    }
                }

                if (!passTexture) {
                    passTexture = mglRenderPassDepthTextureFor(_renderPassManager->state);
                }
                if (!passTexture) {
                    passTexture = mglRenderPassStencilTextureFor(_renderPassManager->state);
                }

                if (passTexture) {
                    passWidth = mglRenderPassTextureInfo(passTexture).width;
                    passHeight = mglRenderPassTextureInfo(passTexture).height;
                    mglRenderPassSetPersistentDimensions(
                        _renderPassManager->state, passWidth, passHeight);
                    if (kMGLVerboseFrameLoopLogs) {
                        NSLog(@"MGL INFO: Resolved render pass size from attachment %lux%lu (rtw/rth were unset)",
                              (unsigned long)passWidth, (unsigned long)passHeight);
                    }
                }
            }
        }

        if ((passWidth == 0 || passHeight == 0) && _drawable && [self mglDrawableTexture]) {
            passWidth = mglRenderPassTextureInfo([self mglDrawableTexture]).width;
            passHeight = mglRenderPassTextureInfo([self mglDrawableTexture]).height;
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to drawable size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if ((passWidth == 0 || passHeight == 0) && [self mglHasMetalLayer]) {
            CGSize drawableSize = [self mglMetalLayerDrawableSize];
            if (drawableSize.width > 0 && drawableSize.height > 0) {
                passWidth = (NSUInteger)drawableSize.width;
                passHeight = (NSUInteger)drawableSize.height;
            } else {
                NSRect frame = [self mglMetalLayerFrame];
                if (frame.size.width > 0 && frame.size.height > 0) {
                    passWidth = (NSUInteger)frame.size.width;
                    passHeight = (NSUInteger)frame.size.height;
                }
            }
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to layer size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if (passWidth > 0 && passHeight > 0) {
            GLint rawSx = 0;
            GLint rawSy = 0;
            GLint rawSw = (GLint)passWidth;
            GLint rawSh = (GLint)passHeight;

            GLint sx = 0;
            GLint sy = 0;
            GLint sw = (GLint)passWidth;
            GLint sh = (GLint)passHeight;

            if (state->caps.scissor_test) {
                rawSx = (GLint)state->var.scissor_box[0];
                rawSy = (GLint)state->var.scissor_box[1];
                rawSw = (GLint)state->var.scissor_box[2];
                rawSh = (GLint)state->var.scissor_box[3];

                sx = rawSx;
                sy = rawSy;
                sw = rawSw;
                sh = rawSh;
                mglRenderClampScissorRect(&sx, &sy, &sw, &sh,
                                          (uint32_t)passWidth,
                                          (uint32_t)passHeight);
            }

            GLint metalSy = mglRenderMetalScissorY(
                sx, sh, (uint32_t)passHeight,
                (uint32_t)state->var.clip_origin);

	            if (traceEncoderState) {
                NSLog(@"MGL SCISSOR apply pass=%lux%lu scissorEnabled=%d origin=0x%x raw=(%d,%d,%d,%d) glResolved=(%d,%d,%d,%d) metal=(%d,%d,%d,%d)",
                      (unsigned long)passWidth, (unsigned long)passHeight,
                      state->caps.scissor_test ? 1 : 0,
                      state->var.clip_origin,
                      rawSx, rawSy, rawSw, rawSh,
                      sx, sy, sw, sh,
                      sx, metalSy, sw, sh);
            }

            MGLScissorRectValue rect;
            rect.x = (NSUInteger)sx;
            rect.y = (NSUInteger)metalSy;
            rect.width = (NSUInteger)sw;
            rect.height = (NSUInteger)sh;
            mglBindingSetScissorRectIfNeeded((__bridge void *)self, rect.x, rect.y, rect.width, rect.height);

            GLdouble rawVx = (GLdouble)state->viewport[0];
            GLdouble rawVy = (GLdouble)state->viewport[1];
            GLdouble rawVw = (GLdouble)state->viewport[2];
            GLdouble rawVh = (GLdouble)state->viewport[3];

            GLdouble vx = rawVx;
            GLdouble vy = rawVy;
            GLdouble vw = rawVw;
            GLdouble vh = rawVh;
            mglRenderClampViewport(&vx, &vy, &vw, &vh, (uint32_t)passWidth,
                                   (uint32_t)passHeight);
            GLdouble metalVy = mglRenderMetalViewportY(vy, vh,
                                                       (uint32_t)passHeight);

            Texture *guiRTColor = NULL;
            Texture *guiRTDepth = NULL;
            BOOL guiRTPass =
                mglTraceLogIsEnabled() &&
                mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx,
                                                                 state->framebuffer,
                                                                 &guiRTColor,
                                                                 &guiRTDepth);
            if (guiRTPass) {
                static uint64_t s_guiRTEncoderStateLogCount = 0;
                uint64_t hit = ++s_guiRTEncoderStateLogCount;
                if (hit <= 128ull || (hit % 256ull) == 0ull) {
                    Program *program = mglResolveProgramFromState(ctx);
                    id c0 = mglRenderPassColorTextureFor(_renderPassManager->state, 0);
                    id d0 = mglRenderPassDepthTextureFor(_renderPassManager->state);
                    mglTraceLog("RT_SAMPLE_COPY_ENCODER hit=%llu fbo=%u rpFbo=%u program=%u rtTex=%u label=\"%s\" depthTex=%u depthLabel=\"%s\" "
                          "pass=%lux%lu c0=%p fmt=%lu depth=%p fmt=%lu "
                          "loadStore(c=%s/%s d=%s/%s) clipOrigin=0x%x "
                          "scissor(en=%d raw=%d,%d,%d,%d metal=%d,%d,%d,%d) "
                          "viewport(raw=%.1f,%.1f,%.1f,%.1f metal=%.1f,%.1f,%.1f,%.1f) "
                          "depth(test=%d write=%d func=0x%x) blend=%d cull=%d levels=%u mips=%u mipmapped=%u",
                          (unsigned long long)hit,
                          state->framebuffer ? (unsigned)state->framebuffer->name : 0u,
                          (unsigned)_renderPassManager->state->renderPassFramebufferName,
                          program ? (unsigned)program->name : (unsigned)state->program_name,
                          (unsigned)mglTraceTextureName(guiRTColor),
                          mglTraceTextureLabel(guiRTColor),
                          (unsigned)mglTraceTextureName(guiRTDepth),
                          mglTraceTextureLabel(guiRTDepth),
                          (unsigned long)passWidth,
                          (unsigned long)passHeight,
                          c0,
                          (unsigned long)(c0 ? mglRenderPassTextureInfo(c0).pixel_format : MGLPixelFormatInvalid),
                          d0,
                          (unsigned long)(d0 ? mglRenderPassTextureInfo(d0).pixel_format : MGLPixelFormatInvalid),
                          mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
                          mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
                          mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
                          mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
                          state->var.clip_origin,
                          state->caps.scissor_test ? 1 : 0,
                          rawSx, rawSy, rawSw, rawSh,
                          sx, metalSy, sw, sh,
                          rawVx, rawVy, rawVw, rawVh,
                          vx, metalVy, vw, vh,
                          state->caps.depth_test ? 1 : 0,
                          state->var.depth_writemask ? 1 : 0,
                          (unsigned)state->var.depth_func,
                          state->caps.blend ? 1 : 0,
                          state->caps.cull_face ? 1 : 0,
                          guiRTColor ? (unsigned)guiRTColor->num_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmap_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmapped : 0u);
                }
            }

            BOOL viewportWasClamped = (vx != rawVx || vy != rawVy || vw != rawVw || vh != rawVh);
            BOOL viewportOriginConverted = (metalVy != vy);
            if (traceEncoderState) {
                mglTraceLog("MGL VIEWPORT apply pass=%lux%lu origin=0x%x raw=(%.3f,%.3f,%.3f,%.3f) resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                              (unsigned long)passWidth, (unsigned long)passHeight,
                              state->var.clip_origin,
                              rawVx, rawVy, rawVw, rawVh,
                              vx, vy, vw, vh,
                              vx, metalVy, vw, vh);
            }

            if (kMGLDiagnosticStateLogs && (viewportWasClamped || viewportOriginConverted)) {
                static uint64_t s_viewportClampDetailCount = 0;
                uint64_t clampHit = ++s_viewportClampDetailCount;
                BOOL logClampDetail = (clampHit <= 80ull || (clampHit % 120ull) == 0ull);

                if (logClampDetail) {
                    Framebuffer *debugFbo = state->framebuffer;
                    BOOL debugFboValid = (debugFbo != NULL &&
                                          mglRendererObjectPointerLikelyValid(debugFbo) &&
                                          mglRendererPointerInHashTable(&state->framebuffer_table, debugFbo) &&
                                          mglPointerRangeIsReadable(debugFbo, sizeof(*debugFbo)));
                    id rpColor0 = mglRenderPassColorTextureFor(_renderPassManager->state, 0);
                    id rpDepth = mglRenderPassDepthTextureFor(_renderPassManager->state);
                    id drawableTexture = (_drawable ? [self mglDrawableTexture] : nil);

                    mglTraceLog("MGL VIEWPORT CLAMP DETAIL hit=%llu fbo=%p valid=%d fboName=%u drawBuffer=0x%x pass=%lux%lu "
                                  "rpColor0=%p(%lux%lu) rpDepth=%p(%lux%lu) drawable=%p(%lux%lu) raw=(%.3f,%.3f,%.3f,%.3f) "
                                  "resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                                  (unsigned long long)clampHit,
                                  debugFbo,
                                  debugFboValid ? 1 : 0,
                                  (debugFboValid ? debugFbo->name : 0),
                                  state->draw_buffer,
                                  (unsigned long)passWidth,
                                  (unsigned long)passHeight,
                                  rpColor0,
                                  (unsigned long)(rpColor0 ? mglRenderPassTextureInfo(rpColor0).width : 0),
                                  (unsigned long)(rpColor0 ? mglRenderPassTextureInfo(rpColor0).height : 0),
                                  rpDepth,
                                  (unsigned long)(rpDepth ? mglRenderPassTextureInfo(rpDepth).width : 0),
                                  (unsigned long)(rpDepth ? mglRenderPassTextureInfo(rpDepth).height : 0),
                                  drawableTexture,
                                  (unsigned long)(drawableTexture ? mglRenderPassTextureInfo(drawableTexture).width : 0),
                                  (unsigned long)(drawableTexture ? mglRenderPassTextureInfo(drawableTexture).height : 0),
                                  rawVx, rawVy, rawVw, rawVh,
                                  vx, vy, vw, vh,
                                  vx, metalVy, vw, vh);

                    if (debugFboValid) {
                        for (int attIndex = 0; attIndex < MAX_COLOR_ATTACHMENTS; attIndex++) {
                            FBOAttachment *attachment = &debugFbo->color_attachments[attIndex];
                            if (attachment->texture == 0 && attachment->buf.tex == NULL && attachment->buf.rbo == NULL) {
                                continue;
                            }

                            Texture *attachmentTexture = NULL;
                            if (mglRenderTargetIsRenderbuffer((uint32_t)attachment->textarget)) {
                                attachmentTexture = attachment->buf.rbo ? attachment->buf.rbo->tex : NULL;
                            } else {
                                attachmentTexture = attachment->buf.tex;
                                if (!attachmentTexture && attachment->texture != 0) {
                                    attachmentTexture = findTexture(ctx, attachment->texture);
                                }
                            }

                            id attachmentMtl = (attachmentTexture && attachmentTexture->mtl_data)
                                ? (__bridge id)(attachmentTexture->mtl_data)
                                : nil;
                            id rpAttachment = mglRenderPassColorTextureFor(_renderPassManager->state, attIndex);

                            mglTraceLog("MGL VIEWPORT CLAMP FBO att=%d name=%u textarget=0x%x level=%d layer=%d tex=%p "
                                          "texName=%u texTarget=0x%x texSize=%ux%ux%u mtl=%p(%lux%lu) rpTex=%p(%lux%lu)",
                                          attIndex,
                                          attachment->texture,
                                          attachment->textarget,
                                          attachment->level,
                                          attachment->layer,
                                          attachmentTexture,
                                          attachmentTexture ? attachmentTexture->name : 0,
                                          attachmentTexture ? attachmentTexture->target : 0,
                                          attachmentTexture ? attachmentTexture->width : 0,
                                          attachmentTexture ? attachmentTexture->height : 0,
                                          attachmentTexture ? attachmentTexture->depth : 0,
                                          attachmentMtl,
                                          (unsigned long)(attachmentMtl ? mglRenderPassTextureInfo(attachmentMtl).width : 0),
                                          (unsigned long)(attachmentMtl ? mglRenderPassTextureInfo(attachmentMtl).height : 0),
                                          rpAttachment,
                                          (unsigned long)(rpAttachment ? mglRenderPassTextureInfo(rpAttachment).width : 0),
                                          (unsigned long)(rpAttachment ? mglRenderPassTextureInfo(rpAttachment).height : 0));
                        }
                    }
                }
            }

            /* gl_ViewportIndex: when glViewportIndexedf* set any slot
             * beyond 0, bind the whole 16-entry viewport array (Metal
             * selects per vertex via viewport_array_index).  Slot 0 uses
             * the resolved/clamped rectangle computed above. */
            if (state->viewport_array_set) {
                double viewports[MGL_MAX_VIEWPORTS * 6];
                viewports[0] = vx;
                viewports[1] = metalVy;
                viewports[2] = vw;
                viewports[3] = vh;
                viewports[4] = state->var.depth_range[0];
                viewports[5] = state->var.depth_range[1];
                for (int vi = 1; vi < MGL_MAX_VIEWPORTS; vi++) {
                    GLdouble avx = state->viewport_array[vi][0];
                    GLdouble avy = state->viewport_array[vi][1];
                    GLdouble avw = state->viewport_array[vi][2];
                    GLdouble avh = state->viewport_array[vi][3];
                    GLdouble metalAvy = (GLdouble)passHeight - (avy + avh);
                    if (metalAvy < 0.0) metalAvy = 0.0;
                    viewports[vi * 6 + 0] = avx;
                    viewports[vi * 6 + 1] = metalAvy;
                    viewports[vi * 6 + 2] = avw;
                    viewports[vi * 6 + 3] = avh;
                    viewports[vi * 6 + 4] = state->var.depth_range[0];
                    viewports[vi * 6 + 5] = state->var.depth_range[1];
                }
                mglRenderBindingSetViewportsForOwner(
                    _bindingStateOwner,
                    _renderPassManager->state->currentRenderEncoderOwner,
                    viewports, (uint64_t)MGL_MAX_VIEWPORTS);
            } else {

                double viewports[MGL_MAX_VIEWPORTS * 6];
                for (int vi = 0; vi < MGL_MAX_VIEWPORTS; vi++) {
                    viewports[vi * 6 + 0] = vx;
                    viewports[vi * 6 + 1] = metalVy;
                    viewports[vi * 6 + 2] = vw;
                    viewports[vi * 6 + 3] = vh;
                    viewports[vi * 6 + 4] = state->var.depth_range[0];
                    viewports[vi * 6 + 5] = state->var.depth_range[1];
                }
                mglRenderBindingSetViewportsForOwner(
                    _bindingStateOwner,
                    _renderPassManager->state->currentRenderEncoderOwner,
                    viewports, (uint64_t)MGL_MAX_VIEWPORTS);
            }
        } else {
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: updateCurrentRenderEncoder could not resolve pass size; using raw GL viewport");
            }
            mglBindingSetViewportIfNeeded((__bridge void *)self, state->viewport[0], state->viewport[1], state->viewport[2], state->viewport[3], state->var.depth_range[0], state->var.depth_range[1]);
        }
    }
}








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
