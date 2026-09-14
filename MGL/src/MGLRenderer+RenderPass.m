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
#include "mgl_stage_encode_drivers.h" /* stage binding drivers (log 131) */
#include "mgl_draw_encode.h"
#include "mgl_render_pass_manager_ops.h"
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

static id mglRenderPassCreateTexture(
    const MGLRenderTextureDescriptorState *descriptor)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(
            descriptor, NULL, &texture) == 0 &&
        texture) {
        return (__bridge_transfer id)texture;
    }
    return nil;
}

static MGLRendererBackendHandle *mglRenderPassBackend(GLMContext context)
{
    return context
        ? (MGLRendererBackendHandle *)context->renderer_backend
        : NULL;
}

/* VS-only + GL_RASTERIZER_DISCARD cannot leave Metal rasterization disabled:
 * AGX drops vertex texture/SSBO stores. A no-op fragment keeps rasterization
 * on while color write masks stay cleared (see below). */

typedef NS_ENUM(uint32_t, MGLStubFSValueClass) {
    MGLStubFSFloat = 0,
    MGLStubFSInt,
    MGLStubFSUint,
};

static MGLStubFSValueClass mglPixelFormatValueClass(uint32_t fmt)
{
    return (MGLStubFSValueClass)mglRenderMetalPixelFormatValueClass(fmt);
}

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

static id mglRenderPassDefaultDrawBufferAttachment(
    MGLRendererBackendHandle *backend, GLuint drawBufferIndex,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind)
{
    return (__bridge id)
        mglRendererBackendGetDefaultDrawBufferAttachment(
            backend, drawBufferIndex, kind);
}

static id mglRenderPassTransientDepthTexture(
    GLMContext context, NSUInteger *widthOut, NSUInteger *heightOut)
{
    uint64_t width = 0;
    uint64_t height = 0;
    void *texture = mglRendererBackendGetTransientDepthTexture(
        mglRenderPassBackend(context), &width, &height);
    if (widthOut) *widthOut = (NSUInteger)width;
    if (heightOut) *heightOut = (NSUInteger)height;
    return (__bridge id)texture;
}

static id mglRenderPassCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static void mglRenderPassWaitCommandBuffer(id commandBuffer)
{
    if (mglRenderWaitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp render-pass wait failed");
    }
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

static BOOL mglRenderPassClearValuesFor(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    double *clearColorOut,   /* RGBA, color attachments */
    double *clearDepthOut,
    uint32_t *clearStencilOut)
{
    MGLRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return NO;
    /* O3.1: the class/index/clear-resolution decision now lives in the pure-C
     * plan layer (mgl_render_pass_plan.c); ObjC only fetches persistent state
     * and forwards.  Covered by the clear-value harness
     * (test-render-pass-clear-plan). */
    return mglRenderPassPlanClearValues(&state, attachmentKind,
                                        (uint32_t)colorIndex,
                                        clearColorOut, clearDepthOut,
                                        clearStencilOut) ? YES : NO;
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

static MGLRenderPassClearColorValue mglRenderPassClearColorFor(
    const MGLCommandState *commandState,
    NSUInteger colorIndex,
    MGLRenderPassClearColorValue fallback)
{
    double rgba[4] = {0.0, 0.0, 0.0, 0.0};
    if (mglRenderPassClearValuesFor(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
            colorIndex, rgba, NULL, NULL)) {
        return (MGLRenderPassClearColorValue){rgba[0], rgba[1], rgba[2], rgba[3]};
    }
    return fallback;
}

static double mglRenderPassClearDepthFor(
    const MGLCommandState *commandState, double fallback)
{
    double depth = 0.0;
    if (mglRenderPassClearValuesFor(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u,
            NULL, &depth, NULL)) {
        return depth;
    }
    return fallback;
}

static uint32_t mglRenderPassClearStencilFor(
    const MGLCommandState *commandState, uint32_t fallback)
{
    uint32_t stencil = 0u;
    if (mglRenderPassClearValuesFor(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u,
            NULL, NULL, &stencil)) {
        return stencil;
    }
    return fallback;
}

static void mglRenderPassSetPersistentAttachment(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    id texture,
    NSUInteger level,
    NSUInteger slice,
    NSUInteger depthPlane,
    BOOL layered)
{

    if (commandState && commandState->renderPassStateOwner) {
        mglRenderSetRenderPassStateAttachmentTexture(
            commandState->renderPassStateOwner, attachmentKind,
            (uint32_t)colorIndex, (__bridge void *)texture,
            level, slice, depthPlane, layered ? 1u : 0u);
    }
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

static void mglRenderPassSetPersistentLoadAction(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t loadAction)
{
    uint32_t storeAction = MGLStoreActionDontCare;
    MGLRenderPassAttachmentState state = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &state)) {
        storeAction = (uint32_t)state.store_action;
    } else {
        return;
    }
    mglRenderPassSetPersistentActions(
        commandState, attachmentKind, colorIndex, loadAction, storeAction);
}

static void mglRenderPassSetPersistentStoreAction(
    const MGLCommandState *commandState,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    uint32_t storeAction)
{
    uint32_t loadAction = MGLLoadActionDontCare;
    MGLRenderPassAttachmentState state = {0};
    if (mglRenderPassGetPersistentAttachmentState(
            commandState, attachmentKind, colorIndex, &state)) {
        loadAction = (uint32_t)state.load_action;
    } else {
        return;
    }
    mglRenderPassSetPersistentActions(
        commandState, attachmentKind, colorIndex, loadAction, storeAction);
}

static void mglRenderPassSetPersistentColorClear(
    const MGLCommandState *commandState,
    NSUInteger colorIndex,
    MGLRenderPassClearColorValue clearColor)
{
    if (!commandState || colorIndex >= MAX_COLOR_ATTACHMENTS) {
        return;
    }
    if (commandState->renderPassStateOwner) {
        mglRenderSetRenderPassStateColorClear(
            commandState->renderPassStateOwner, (uint32_t)colorIndex,
            clearColor.red, clearColor.green, clearColor.blue,
            clearColor.alpha);
    }
}

static void mglRenderPassSetPersistentDepthClear(
    const MGLCommandState *commandState,
    double clearDepth)
{
    if (commandState && commandState->renderPassStateOwner) {
        mglRenderSetRenderPassStateDepthClear(
            commandState->renderPassStateOwner, clearDepth);
    }
}

static void mglRenderPassSetPersistentStencilClear(
    const MGLCommandState *commandState,
    uint32_t clearStencil)
{
    if (commandState && commandState->renderPassStateOwner) {
        mglRenderSetRenderPassStateStencilClear(
            commandState->renderPassStateOwner, clearStencil);
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
/* MSAA array textures are represented by a 2D array whose physical slices
 * are laid out as [gl_layer][sample] with a fixed eight-slice stride.  A
 * layered render pass therefore needs to translate the logical GL layer
 * before Metal consumes [[render_target_array_index]].  Keep this decision
 * in the render-pass domain: ordinary 2D arrays remain a one-to-one map and
 * non-layered framebufferTextureLayer attachments keep their fixed slice. */
static uint32_t mglGeometryPassthroughLayerStride(GLMContext context)
{
    if (!context || !context->active_state ||
        !context->active_state->framebuffer) {
        return 1u;
    }
    Framebuffer *fbo = context->active_state->framebuffer;
    for (GLuint i = 0u; i < MAX_COLOR_ATTACHMENTS; i++) {
        const FBOAttachment *attachment = &fbo->color_attachments[i];
        uint32_t stride = mglRenderMSAAArrayLayerStride(
            attachment->layered ? 1 : 0, (uint32_t)attachment->textarget);
        if (stride > 1u) {
            return stride;
        }
    }
    uint32_t depthStride = mglRenderMSAAArrayLayerStride(
        fbo->depth.layered ? 1 : 0, (uint32_t)fbo->depth.textarget);
    if (depthStride > 1u) {
        return depthStride;
    }
    return mglRenderMSAAArrayLayerStride(
        fbo->stencil.layered ? 1 : 0, (uint32_t)fbo->stencil.textarget);
}

/* The backend keeps one passthrough function per kind.  Include the render
 * target layer convention in the key so switching between ordinary and
 * emulated-MSAA layered FBOs cannot reuse a function compiled for the other
 * convention. */
static uint64_t mglGeometryPipelineFunctionKey(
    const Program *vertexProgram, const Program *geometryProgram,
    uint32_t layerStride)
{
    uint64_t hash = 1469598103934665603ull;
    hash = mglHashStepU64(hash,
                          vertexProgram ? vertexProgram->pipeline_cache_instance_id : 0u);
    hash = mglHashStepU64(hash,
                          vertexProgram ? vertexProgram->pipeline_cache_generation : 0u);
    hash = mglHashStepU64(hash,
                          geometryProgram ? geometryProgram->pipeline_cache_instance_id : 0u);
    hash = mglHashStepU64(hash,
                          geometryProgram ? geometryProgram->pipeline_cache_generation : 0u);
    return mglHashStepU64(hash, layerStride);
}

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

- (Texture *)framebufferAttachmentTexture: (FBOAttachment *)fbo_attachment
{
    /* C port (ObjC-zeroing T4): the resolution lives in
     * mglRendererAttachmentTextureFor(). */
    return mglRendererAttachmentTextureFor(ctx, fbo_attachment);
}

- (bool)currentRenderPassMatchesCurrentFramebuffer
{
    if (!ctx || !_renderPassManager->state->renderPassStateOwner) {
        return true;
    }

    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLuint fboName = fbo ? fbo->name : 0u;


    if (fbo != NULL && fboName != 0u) {
        MGLRenderFboMatchCacheState cache = {0};
        if (_renderPassManager->state->renderPassIdentityOwner &&
            mglRenderGetFboMatchCache(
                _renderPassManager->state->renderPassIdentityOwner,
                &cache) == 0 &&
            cache.fbo_name == fboName &&
            cache.generation == fbo->fbo_attachment_generation) {
            return cache.result != 0;
        }
    }

    bool result = mglRenderPassMatchesFramebufferImpl(
                      (__bridge void *)self, fbo, fboName) != 0;

    /* store cache for non-default FBOs only. */
    if (fbo != NULL && fboName != 0u) {
        mglPassManagerSetFboMatchCacheResult(_renderPassManager, result, fboName, fbo->fbo_attachment_generation);
    }

    return result;
}


- (void)endRenderPassIfFramebufferChangedForNonDraw:(uint64_t)processCall
{
    if (!ctx || mglRenderEncoderOwnerHasCurrent(
                    _renderPassManager->state->currentRenderEncoderOwner) != 1) {
        return;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return;
    }

    static uint64_t s_nonDrawFboMismatchCount = 0;
    uint64_t hit = ++s_nonDrawFboMismatchCount;
    if (mglTraceLogIsEnabled() && (hit <= 32ull || (hit % 256ull) == 0ull)) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
        GLuint fboName = fbo ? fbo->name : 0u;
        mglTraceLog("RENDERPASS_NON_DRAW_MISMATCH processCall=%llu hit=%llu "
                    "ctxFbo=%u(%p) ctxDrawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
                    (unsigned long long)processCall,
                    (unsigned long long)hit,
                    (unsigned)fboName,
                    fbo,
                    (unsigned)MGL_STATE(ctx)->draw_buffer,
                    (unsigned)_renderPassManager->state->renderPassFramebufferName,
                    _renderPassManager->state->renderPassFramebuffer,
                    (unsigned)_renderPassManager->state->renderPassDrawBuffer);
        mglLogRenderPassLifecycle("non-draw-mismatch-before-end",
                                  hit,
                                  ctx,
                                  _renderPassManager->state->currentCommandBufferOwner,
                                  _renderPassManager->state->currentRenderEncoderOwner,
                                  _renderPassManager->state->renderPassStateOwner,
                                  (__bridge void *)_drawable,
                                  _renderPassManager->state->renderPassFramebuffer,
                                  _renderPassManager->state->renderPassFramebufferName,
                                  _renderPassManager->state->renderPassDrawBuffer,
                                  _renderPassManager->state->renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE);
}

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

    if (![self ensureWritableCommandBuffer:reason ? reason : "restore_render_encoder_after_texture_upload"]) {
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

- (bool) newRenderEncoder
{
    return [self newRenderEncoderWithReason:MGL_ENC_REASON_OTHER];
}

- (bool) newRenderEncoderWithReason:(MGLEncoderCreateReason)reason
{
    METAL_LOCK();
    bool result = [self newRenderEncoderLockedWithReason:reason];
    METAL_UNLOCK();
    return result;
}


- (bool) configureDefaultFramebufferAttachmentsLocked
{
    GLuint mgl_drawbuffer;
    id texture = nil;
    id depth_texture = nil;
    id stencil_texture = nil;

    uint32_t mappedDraw = 0u;
    if (!mglRenderDefaultDrawBufferIndex(
            (uint32_t)MGL_STATE(ctx)->draw_buffer, &mappedDraw)) {
        DEBUG_PRINT("MGL: Unknown draw_buffer value: 0x%x, falling back to FRONT\n", MGL_STATE(ctx)->draw_buffer);
        NSLog(@"MGL WARNING: Unknown draw_buffer value 0x%x, using FRONT fallback", MGL_STATE(ctx)->draw_buffer);
    } else if (mglRenderDrawBufferIsNone(
                   (uint32_t)MGL_STATE(ctx)->draw_buffer)) {
        DEBUG_PRINT("MGL: draw_buffer is GL_NONE, falling back to FRONT\n");
    }
    mgl_drawbuffer = (GLuint)mappedDraw;

    if(![self checkDrawBufferSize:mgl_drawbuffer])
    {
        (void)mglRendererBackendClearDefaultDrawBuffer(
            _backend, mgl_drawbuffer);
        _drawBuffers[mgl_drawbuffer].width = 0;
        _drawBuffers[mgl_drawbuffer].height = 0;
    }

    // attach color buffer
    if (mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer))
    {
        // SAFETY: Ensure we have a valid drawable with texture
        if (!_drawable) {
            NSLog(@"MGL ERROR: No drawable available for front buffer");
            return false;
        }

        texture = [self mglDrawableTexture];

        // sleep mode will return a null texture - handle gracefully without crashing
        if (!texture) {
            NSLog(@"MGL WARNING: Drawable texture is NULL (sleep mode or window not visible), attempting to get new drawable");

            // Try to get a new drawable
            _drawable = [self mglNextDrawable];
            if (_drawable) {
                texture = [self mglDrawableTexture];
                NSLog(@"MGL INFO: Successfully obtained new drawable with texture");
            } else {
                NSLog(@"MGL ERROR: Still no drawable texture available");
                return false;
            }
        }
    }
    else
    {
        texture = mglRenderPassDefaultDrawBufferAttachment(
            _backend, mgl_drawbuffer,
            MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR);
        if (!texture) {
            texture = [self newDrawBuffer:ctx->pixel_format.mtl_pixel_format
                           isDepthStencil:false];
            (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                _backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR,
                (__bridge void *)texture);
        }
    }

    // attach depth. The default framebuffer must have a usable depth
    // attachment whenever GL depth testing is active, even if the legacy
    // context format fields were left unset by the window/bootstrap path.
    id cachedDepth =
        mglRenderPassDefaultDrawBufferAttachment(
            _backend, mgl_drawbuffer,
            MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH);
    BOOL defaultPassNeedsDepth = MGL_STATE(ctx)->caps.depth_test ||
                                 cachedDepth != nil;
    if (defaultPassNeedsDepth)
    {
        uint32_t depthFormat = mglRenderDepthFormatOrFallback(
            ctx->depth_format.mtl_pixel_format);

        if(cachedDepth)
        {
            depth_texture = cachedDepth;
        }
        else
        {
            MGLRenderTextureInfo textureInfo = mglRenderPassTextureInfo(texture);
            depth_texture = [self newDrawBufferWithCustomSize:depthFormat
                                                   isDepthStencil:true
                                                     customSize:CGSizeMake(textureInfo.width,
                                                                           textureInfo.height)];
            (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                _backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH,
                (__bridge void *)depth_texture);
            if (depth_texture) {
                static uint64_t s_defaultDepthCreateCount = 0;
                uint64_t hit = ++s_defaultDepthCreateCount;
                if (kMGLDiagnosticStateLogs && hit <= 8) {
                    mglTraceLog("MGL DEFAULT FBO: created depth attachment fmt=%lu size=%lux%lu drawBuffer=%u",
                                  (unsigned long)depthFormat,
                                  (unsigned long)mglRenderPassTextureInfo(depth_texture).width,
                                  (unsigned long)mglRenderPassTextureInfo(depth_texture).height,
                                  mgl_drawbuffer);
                }
            }
        }
    }

    // attach stencil
    id cachedStencil =
        mglRenderPassDefaultDrawBufferAttachment(
            _backend, mgl_drawbuffer,
            MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL);
    BOOL defaultPassNeedsStencil = MGL_STATE(ctx)->caps.stencil_test ||
                                   ctx->stencil_format.format ||
                                   cachedStencil != nil;
    if (defaultPassNeedsStencil)
    {
        uint32_t stencilFormat = mglRenderRepairedDefaultStencilFormat(
            ctx->stencil_format.mtl_pixel_format);

        if(cachedStencil)
        {
            stencil_texture = cachedStencil;
        }
        else
        {
            MGLRenderTextureInfo textureInfo = mglRenderPassTextureInfo(texture);
            stencil_texture = [self newDrawBufferWithCustomSize:stencilFormat
                                                     isDepthStencil:true
                                                       customSize:CGSizeMake(textureInfo.width,
                                                                             textureInfo.height)];
            (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                _backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL,
                (__bridge void *)stencil_texture);
        }
    }

    mglRenderPassSetPersistentAttachment(
        _renderPassManager->state,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
        mglApplySRGBStateToRenderTarget(texture, ctx), 0, 0, 0, NO);
    mglRenderPassSetPersistentAttachment(
        _renderPassManager->state,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
        depth_texture, 0, 0, 0, NO);
    mglRenderPassSetPersistentAttachment(
        _renderPassManager->state,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
        stencil_texture, 0, 0, 0, NO);

    mglRenderPassSetPersistentDimensions(
        _renderPassManager->state, mglRenderPassTextureInfo(texture).width, mglRenderPassTextureInfo(texture).height);
    _drawBuffers[mgl_drawbuffer].width = (GLuint)mglRenderPassTextureInfo(texture).width;
    _drawBuffers[mgl_drawbuffer].height = (GLuint)mglRenderPassTextureInfo(texture).height;
    return true;
}

- (void) ensureTransientDepthForDefaultFramebufferLocked
{
    if (!MGL_STATE(ctx)->framebuffer &&
        MGL_STATE(ctx)->caps.depth_test &&
        !mglRenderPassDepthTextureFor(_renderPassManager->state)) {
        NSUInteger depthWidth = mglRenderPassRenderTargetWidthFor(_renderPassManager->state);
        NSUInteger depthHeight = mglRenderPassRenderTargetHeightFor(_renderPassManager->state);

        if (depthWidth == 0 || depthHeight == 0) {
            id color0 = mglRenderPassColorTextureFor(_renderPassManager->state, 0);
            if (color0) {
                depthWidth = mglRenderPassTextureInfo(color0).width;
                depthHeight = mglRenderPassTextureInfo(color0).height;
            }
        }

        if (depthWidth > 0 && depthHeight > 0) {
            NSUInteger cachedDepthWidth = 0;
            NSUInteger cachedDepthHeight = 0;
            id transientDepth =
                mglRenderPassTransientDepthTexture(
                    ctx, &cachedDepthWidth, &cachedDepthHeight);
            if (!transientDepth || cachedDepthWidth != depthWidth ||
                cachedDepthHeight != depthHeight) {
                MGLRenderTextureDescriptorState depthDesc = {0};
                depthDesc.texture_type = MGLTextureType2D;
                depthDesc.pixel_format = mglRenderDefaultDepthPixelFormat();
                depthDesc.width = depthWidth;
                depthDesc.height = depthHeight;
                depthDesc.depth = 1;
                depthDesc.mipmap_level_count = 1;
                depthDesc.sample_count = 1;
                depthDesc.array_length = 1;
                depthDesc.usage = MGLTextureUsageRenderTarget;
                depthDesc.storage_mode = MGLStorageModePrivate;
                transientDepth =
                    mglRenderPassCreateTexture(&depthDesc);
                if (mglRendererBackendSetTransientDepthTexture(
                        mglRenderPassBackend(ctx),
                        (__bridge void *)transientDepth,
                        depthWidth, depthHeight) != 0) {
                    transientDepth = nil;
                } else {
                    transientDepth = mglRenderPassTransientDepthTexture(
                        ctx, NULL, NULL);
                }

                if (transientDepth) {
                    static uint64_t s_transientDepthCreateCount = 0;
                    uint64_t hit = ++s_transientDepthCreateCount;
                    if (hit <= 16 || (hit % 128) == 0) {
                        NSLog(@"MGL TRANSIENT FBO: created depth attachment fmt=%lu size=%lux%lu fbo=%u",
                              (unsigned long)mglRenderDefaultDepthPixelFormat(),
                              (unsigned long)depthWidth,
                              (unsigned long)depthHeight,
                              (unsigned)(mglRendererSafeFramebufferName(ctx)));
                    }
                } else {
                    NSLog(@"MGL ERROR: failed to create transient depth attachment size=%lux%lu fbo=%u",
                          (unsigned long)depthWidth,
                          (unsigned long)depthHeight,
                          (unsigned)(mglRendererSafeFramebufferName(ctx)));
                }
            }

            if (transientDepth) {
                mglRenderPassSetPersistentAttachment(
                    _renderPassManager->state,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    transientDepth,
                    0, 0, 0, NO);
                mglRenderPassSetPersistentActions(
                    _renderPassManager->state,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    MGLLoadActionClear, MGLStoreActionDontCare);
                mglRenderPassSetPersistentDepthClear(
                    _renderPassManager->state,
                    MGL_STATE(ctx)->var.depth_clear_value);
            }
        }
    }
}

- (void) configureUserFBOLoadStoreActionsLocked:(GLuint *)outFboColorClearCount
                                  fboColorClearMask:(GLbitfield *)outFboColorClearMask
                     fboColorAttachment0ClearMask:(GLbitfield *)outFboColorAttachment0ClearMask
{
    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);

    BOOL dontCareLoadEnabled = mglEnvFlagEnabled("MGL_ENABLE_DONTCARE_LOAD");
    for (int i = 0; i < drawBufferCount; ++i) {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (!mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                   mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                   &attachmentIndex) ||
            attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            mglRenderPassSetPersistentLoadAction(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MGLLoadActionLoad);
            continue;
        }

        FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
        if (attachmentIndex == 0) {
            *outFboColorAttachment0ClearMask = att->clear_bitmask;
        }

        Texture *attachmentTextureForClear = [self framebufferAttachmentTexture:att];
        /* stamp this attachment's frame generation on EVERY
         * render-target use (clear/load/dontcare), capturing whether this
         * is its first use this frame BEFORE stamping. A clear-then-resume
         * within one frame must record the clear as a use so the resume is
         * not mistaken for a first use (which would wrongly DontCare and
         * discard the cleared+drawn content). */
        BOOL colorFirstUseThisFrame = NO;
        if (dontCareLoadEnabled && attachmentTextureForClear) {
            colorFirstUseThisFrame =
                (attachmentTextureForClear->mtl_rt_frame_generation != _renderPassManager->state->dontCareFrameGeneration);
            attachmentTextureForClear->mtl_rt_frame_generation = _renderPassManager->state->dontCareFrameGeneration;
        }
        MGLRenderPassLoadStoreInput loadStore = {0};
        loadStore.attachment_kind = MGL_RP_ATTACHMENT_COLOR;
        loadStore.attachment_present = 1;
        loadStore.has_clear_pending =
            mglRenderClearMaskHasColor((uint32_t)att->clear_bitmask) ? 1 : 0;
        loadStore.texture_present =
            (attachmentTextureForClear && attachmentTextureForClear->mtl_data) ? 1 : 0;
        loadStore.dontcare_enabled = dontCareLoadEnabled ? 1 : 0;
        loadStore.first_use_this_frame = colorFirstUseThisFrame ? 1 : 0;
        loadStore.blend_enabled =
            (ctx && MGL_STATE(ctx)->caps.blend) ? 1 : 0;
        MGLRenderPassLoadStorePlan loadStorePlan = {0};
        if (mglRenderPassPlanLoadStore(&loadStore, &loadStorePlan) != 0) {
            mglRenderPassSetPersistentLoadAction(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MGLLoadActionLoad);
            continue;
        }
        if (loadStorePlan.load_action == MGLLoadActionClear) {
            if (attachmentTextureForClear &&
                attachmentTextureForClear->name == 8u &&
                mglTraceLogIsEnabled()) {
                mglTraceLog("PENDING_COLOR_CLEAR_CONSUME tex=%u fbo=%u attachment=%u slot=%d program=%u clearMask=0x%x rgba=(%.3f,%.3f,%.3f,%.3f) drawBuf=0x%x readBuf=0x%x scissor(test=%d box=%d,%d,%d,%d) colorMask=%d%d%d%d depth(test=%d write=%d)",
                            (unsigned)attachmentTextureForClear->name,
                            (unsigned)fbo->name,
                            (unsigned)attachmentIndex,
                            i,
                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                            (unsigned)att->clear_bitmask,
                            att->clear_color[0],
                            att->clear_color[1],
                            att->clear_color[2],
                            att->clear_color[3],
                            (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
                            (unsigned)(ctx ? MGL_STATE(ctx)->read_buffer : 0u),
                            (ctx && MGL_STATE(ctx)->caps.scissor_test) ? 1 : 0,
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[0] : 0),
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[1] : 0),
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[2] : 0),
                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[3] : 0),
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][0]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][1]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][2]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.color_writemask[0][3]) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->caps.depth_test) ? 1 : 0,
                            (ctx && MGL_STATE(ctx)->var.depth_writemask) ? 1 : 0);
            }
            mglRenderPassSetPersistentColorClear(
                _renderPassManager->state, colorSlot,
                (MGLRenderPassClearColorValue){att->clear_color[0],
                                               att->clear_color[1],
                                               att->clear_color[2],
                                               att->clear_color[3]});
            mglRenderPassSetPersistentActions(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                loadStorePlan.load_action,
                loadStorePlan.set_store_action ? loadStorePlan.store_action
                                               : MGLStoreActionStore);

            /* MS textures are texture2d_array sample planes; LoadActionClear
             * only hits the attached base slice. Clear the other sample
             * planes so imageLoad(sample) sees the same clear value. */
            if (attachmentTextureForClear &&
                attachmentTextureForClear->mtl_data &&
                mglRenderIsMultisampleTextureTarget(
                    (uint32_t)attachmentTextureForClear->target)) {
                MGLMetalAttachmentSubresource sub =
                    mglMetalAttachmentSubresourceForAttachment(att);
                NSUInteger samples =
                    MAX((NSUInteger)attachmentTextureForClear->samples, 1u);
                for (NSUInteger s = 1u; s < samples; s++) {
                    (void)mglRenderEncodeColorClearForCommandBufferOwner(
                        _renderPassManager->state->currentCommandBufferOwner,
                        attachmentTextureForClear->mtl_data,
                        sub.level, sub.slice + s, sub.depthPlane,
                        att->clear_color[0], att->clear_color[1],
                        att->clear_color[2], att->clear_color[3]);
                }
            }

            att->clear_bitmask =
                (GLbitfield)mglRenderClearMaskClearColor((uint32_t)att->clear_bitmask);
            mglMarkTextureLevelRenderTargetWritten(attachmentTextureForClear, att->level);

            (*outFboColorClearCount)++;
            *outFboColorClearMask |= (GLbitfield)(1u << attachmentIndex);
        } else {
            /* DontCare when the plan allows discarding, Load otherwise: the
             * predicate (flag, texture, first use this frame, blending) lives
             * in the plan. */
            mglRenderPassSetPersistentLoadAction(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                loadStorePlan.load_action);
        }
    }


    for (GLuint ai = 0; ai < MAX_COLOR_ATTACHMENTS; ++ai) {
        /* A pending clear for an attachment this pass does not have cannot be
         * consumed; the rule lives in the plan. */
        if (mglRenderPassDropsStaleColorClear(
                (uint32_t)fbo->color_attachments[ai].clear_bitmask,
                (uint32_t)fbo->color_attachment_bitfield, (uint32_t)ai)) {
            fbo->color_attachments[ai].clear_bitmask =
                (GLbitfield)mglRenderClearMaskClearColor(
                    (uint32_t)fbo->color_attachments[ai].clear_bitmask);
        }
    }

    MGLRenderPassLoadStoreInput depthLoadStore = {0};
    depthLoadStore.attachment_kind = MGL_RP_ATTACHMENT_DEPTH;
    depthLoadStore.has_clear_pending =
        mglRenderClearMaskHasDepth((uint32_t)fbo->depth.clear_bitmask) ? 1 : 0;
    depthLoadStore.texture_present =
        mglRenderPassDepthTextureFor(_renderPassManager->state) ? 1 : 0;
    MGLRenderPassLoadStorePlan depthPlan = {0};
    if (mglRenderPassPlanLoadStore(&depthLoadStore, &depthPlan) == 0 &&
        depthPlan.load_action == MGLLoadActionClear) {
        mglRenderPassSetPersistentDepthClear(
            _renderPassManager->state, fbo->depth.clear_color[0]);
        mglRenderPassSetPersistentActions(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            depthPlan.load_action, depthPlan.store_action);
        fbo->depth.clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearDepth(
                (uint32_t)fbo->depth.clear_bitmask);
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            depthPlan.load_action);
        if (depthPlan.set_store_action) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                depthPlan.store_action);
        }
    }

    MGLRenderPassLoadStoreInput stencilLoadStore = {0};
    stencilLoadStore.attachment_kind = MGL_RP_ATTACHMENT_STENCIL;
    stencilLoadStore.has_clear_pending =
        mglRenderClearMaskHasStencil((uint32_t)fbo->stencil.clear_bitmask) ? 1 : 0;
    stencilLoadStore.texture_present =
        mglRenderPassStencilTextureFor(_renderPassManager->state) ? 1 : 0;
    MGLRenderPassLoadStorePlan stencilPlan = {0};
    if (mglRenderPassPlanLoadStore(&stencilLoadStore, &stencilPlan) == 0 &&
        stencilPlan.load_action == MGLLoadActionClear) {
        mglRenderPassSetPersistentStencilClear(
            _renderPassManager->state,
            (uint32_t)fbo->stencil.clear_color[0]);
        mglRenderPassSetPersistentActions(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            stencilPlan.load_action, stencilPlan.store_action);
        fbo->stencil.clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearStencil(
                (uint32_t)fbo->stencil.clear_bitmask);
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            stencilPlan.load_action);
        if (stencilPlan.set_store_action) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                stencilPlan.store_action);
        }
    }
}

- (void) configureDefaultFramebufferLoadStoreActionsLocked
{
    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLbitfield defaultClearMask = MGL_STATE(ctx)->default_fbo_clear_bitmask;
    if (mglRenderClearMaskHasColor((uint32_t)defaultClearMask)) {
        mglRenderPassSetPersistentColorClear(
            _renderPassManager->state, 0,
            (MGLRenderPassClearColorValue){MGL_STATE(ctx)->default_clear_color[0],
                                           MGL_STATE(ctx)->default_clear_color[1],
                                           MGL_STATE(ctx)->default_clear_color[2],
                                           MGL_STATE(ctx)->default_clear_color[3]});
        mglRenderPassSetPersistentActions(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearColor(
                (uint32_t)MGL_STATE(ctx)->default_fbo_clear_bitmask);
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
            MGLLoadActionLoad);
        static uint64_t s_defaultFboLoadLogCount = 0;
        uint64_t hit = ++s_defaultFboLoadLogCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            mglTraceLog("MGL DEFAULT FBO: using Load (no clear mask) call=%llu drawBuf=0x%x fbo=%u",
                          (unsigned long long)hit,
                          MGL_STATE(ctx)->draw_buffer,
                          fbo ? (unsigned)fbo->name : 0u);
        }
    }

    if (mglRenderClearMaskHasDepth((uint32_t)defaultClearMask)) {
        mglRenderPassSetPersistentDepthClear(
            _renderPassManager->state,
            MGL_STATE(ctx)->var.depth_clear_value);
        mglRenderPassSetPersistentActions(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearDepth(
                (uint32_t)MGL_STATE(ctx)->default_fbo_clear_bitmask);
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionLoad);
        if (mglRenderPassDepthTextureFor(_renderPassManager->state)) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MGLStoreActionStore);
        }
    }

    if (mglRenderClearMaskHasStencil((uint32_t)defaultClearMask)) {
        mglRenderPassSetPersistentStencilClear(
            _renderPassManager->state,
            MGL_STATE(ctx)->var.stencil_clear_value);
        mglRenderPassSetPersistentActions(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearStencil(
                (uint32_t)MGL_STATE(ctx)->default_fbo_clear_bitmask);
    } else {
        mglRenderPassSetPersistentLoadAction(
            _renderPassManager->state,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionLoad);
        if (mglRenderPassStencilTextureFor(_renderPassManager->state)) {
            mglRenderPassSetPersistentStoreAction(
                _renderPassManager->state,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MGLStoreActionStore);
        }
    }
}

- (void) logRenderPassClearResolveLocked:(uint64_t)renderEncoderCall
                      traceRenderEncoder:(bool)traceRenderEncoder
                        fboColorClearCount:(GLuint)fboColorClearCount
                         fboColorClearMask:(GLbitfield)fboColorClearMask
            fboColorAttachment0ClearMask:(GLbitfield)fboColorAttachment0ClearMask
                 fboDepthClearMaskBefore:(GLbitfield)fboDepthClearMaskBefore
               fboStencilClearMaskBefore:(GLbitfield)fboStencilClearMaskBefore
                             defaultClearMask:(GLbitfield)defaultClearMask
                                         fbo:(Framebuffer *)fbo
{
	    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
	        MGLRenderPassClearColorValue c0 = mglRenderPassClearColorFor(_renderPassManager->state, 0, (MGLRenderPassClearColorValue){0, 0, 0, 0});
	        mglTraceLog("MGL TRACE clear.resolve call=%llu fbo=%u "
	              "fboColorClears=%u fboColorMask=0x%x fboAtt0ClearMask=0x%x c0LA=%s depthLA=%s stencilLA=%s "
	              "c0Clear=(%.3f,%.3f,%.3f,%.3f) depthClear=%.3f stencilClear=%u",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)fboColorClearCount,
              (unsigned)fboColorClearMask,
              (unsigned)fboColorAttachment0ClearMask,
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
              mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLLoadActionDontCare)),
              c0.red,
              c0.green,
              c0.blue,
              c0.alpha,
	              mglRenderPassClearDepthFor(_renderPassManager->state, 0.0),
	              (unsigned)(unsigned)mglRenderPassClearStencilFor(_renderPassManager->state, 0));
	    }

            BOOL clearResolveInteresting =
                (fboColorClearCount != 0) ||
                (fboColorAttachment0ClearMask != 0) ||
                (mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare) == MGLLoadActionClear) ||
                mglRenderClearMaskHasDepth((uint32_t)fboDepthClearMaskBefore) ||
                (!fbo && mglRenderClearMaskHasDepth((uint32_t)defaultClearMask));
            if (clearResolveInteresting) {
                static uint64_t s_clearResolveDetailLogCount = 0;
                uint64_t hit = ++s_clearResolveDetailLogCount;
                if (mglTraceLogIsEnabled() && (hit <= 256ull || (hit % 512ull) == 0ull)) {
	            MGLRenderPassClearColorValue c0 = mglRenderPassClearColorFor(_renderPassManager->state, 0, (MGLRenderPassClearColorValue){0, 0, 0, 0});
	            id c0Tex = mglRenderPassColorTextureFor(_renderPassManager->state, 0);
	            id dTex = mglRenderPassDepthTextureFor(_renderPassManager->state);
	            id sTex = mglRenderPassStencilTextureFor(_renderPassManager->state);
		            mglTraceLog("RENDERPASS_CLEAR call=%llu hit=%llu fbo=%u drawBuf=0x%x readBuf=0x%x "
	                        "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                        "fboColorClears=%u fboColorMask=0x%x fboAtt0Mask=0x%x pending(global=0x%x default=0x%x depth=0x%x stencil=0x%x) "
	                        "c0LA=%s c0SA=%s depthLA=%s depthSA=%s stencilLA=%s stencilSA=%s "
	                        "c0Tex=%p fmt=%lu size=%lux%lu depthTex=%p fmt=%lu size=%lux%lu stencilTex=%p "
	                        "clearRGBA=(%.6f,%.6f,%.6f,%.6f) depthClear=%.6f stencilClear=%u depthState(test=%d write=%d func=0x%x)",
	                        (unsigned long long)renderEncoderCall,
	                        (unsigned long long)hit,
	                        (unsigned)(mglRendererSafeFramebufferName(ctx)),
	                        (unsigned)MGL_STATE(ctx)->draw_buffer,
	                        (unsigned)MGL_STATE(ctx)->read_buffer,
	                        (int)MGL_STATE(ctx)->viewport[0],
	                        (int)MGL_STATE(ctx)->viewport[1],
	                        (int)MGL_STATE(ctx)->viewport[2],
	                        (int)MGL_STATE(ctx)->viewport[3],
	                        MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
	                        (int)MGL_STATE(ctx)->var.scissor_box[0],
	                        (int)MGL_STATE(ctx)->var.scissor_box[1],
	                        (int)MGL_STATE(ctx)->var.scissor_box[2],
	                        (int)MGL_STATE(ctx)->var.scissor_box[3],
	                        (unsigned)fboColorClearCount,
	                        (unsigned)fboColorClearMask,
	                        (unsigned)fboColorAttachment0ClearMask,
	                        (unsigned)MGL_STATE(ctx)->clear_bitmask,
	                        (unsigned)MGL_STATE(ctx)->default_fbo_clear_bitmask,
	                        (unsigned)fboDepthClearMaskBefore,
	                        (unsigned)fboStencilClearMaskBefore,
	                        mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
	                        mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
	                        mglLoadActionName(mglRenderPassLoadActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(_renderPassManager->state, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLStoreActionDontCare)),
	                        c0Tex,
	                        (unsigned long)(c0Tex ? mglRenderPassTextureInfo(c0Tex).pixel_format : MGLPixelFormatInvalid),
	                        (unsigned long)(c0Tex ? mglRenderPassTextureInfo(c0Tex).width : 0),
	                        (unsigned long)(c0Tex ? mglRenderPassTextureInfo(c0Tex).height : 0),
	                        dTex,
	                        (unsigned long)(dTex ? mglRenderPassTextureInfo(dTex).pixel_format : MGLPixelFormatInvalid),
	                        (unsigned long)(dTex ? mglRenderPassTextureInfo(dTex).width : 0),
	                        (unsigned long)(dTex ? mglRenderPassTextureInfo(dTex).height : 0),
	                        sTex,
	                        c0.red,
	                        c0.green,
	                        c0.blue,
	                        c0.alpha,
	                        mglRenderPassClearDepthFor(_renderPassManager->state, 0.0),
	                        (unsigned)(unsigned)mglRenderPassClearStencilFor(_renderPassManager->state, 0),
	                        MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
	                        MGL_STATE(ctx)->var.depth_writemask ? 1 : 0,
	                        (unsigned)MGL_STATE(ctx)->var.depth_func);
	        }
	    }
}

- (bool) newRenderEncoderLocked
{
    return [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_OTHER];
}

- (bool) newRenderEncoderLockedWithReason:(MGLEncoderCreateReason)reason
{
    /* instrumentation: count every render encoder (re)creation + reason. */
    MGL_PERF_INC(g_mglEncoderCreationsSinceSwap);
    if ((unsigned)reason >= (unsigned)MGL_ENC_REASON_COUNT) {
        reason = MGL_ENC_REASON_OTHER;
    }
    MGL_PERF_INC(g_mglEncoderCreateReasonSinceSwap[reason]);
    // I can't remember why this is here...
    @autoreleasepool {

    mglBindingInvalidateLastBoundState((__bridge void *)self);

    static uint64_t s_newRenderEncoderCallCount = 0;
    uint64_t renderEncoderCall = ++s_newRenderEncoderCallCount;
    bool traceRenderEncoder = mglShouldTraceCall(renderEncoderCall) ||
                              (kMGLDiagnosticStateLogs && ((renderEncoderCall % 60ull) == 0ull));

    // AGX ERROR THROTTLING: Check if we should skip render encoder creation
    // BUT allow limited render encoder creation for essential functionality
    if (mglRendererShouldSkipGPUOperations((__bridge void *)self)) {
        NSLog(@"MGL AGX: Render encoder creation requested during GPU recovery - attempting essential creation");
        // Continue with essential render encoder creation even during recovery
    }

    // CRITICAL SAFETY: Check command buffer before creating render encoder
    if (mglRenderCommandBufferOwnerHasCurrent(
            _renderPassManager->state->currentCommandBufferOwner) != 1) {
        // Attempt recovery: create a new command buffer instead of failing immediately
        if (mglRenderPassNewCommandBufferLocked((__bridge void *)self)) {
            // Successfully created - continue
        } else {
            NSLog(@"MGL ERROR: Cannot create render encoder - no command buffer available");
            mglRendererRecordGPUError((__bridge void *)self);
            return false;
        }
    }

    // end encoding on current render encoder
    mglRendererEndRenderEncodingLocked((__bridge void *)self);

    // grab the next drawable from CAMetalLayer
    if (_drawable == NULL)
    {
        if (!_layer) {
            NSLog(@"MGL ERROR: Cannot get drawable - no CAMetalLayer available");
            return false;
        }

        CGSize expectedDrawableSize = [self mglApplyPendingDrawableSize];
        _drawable = [self mglNextDrawable];

        // late init of gl scissor box on attachment to window system
        NSUInteger drawableWidth = (NSUInteger)MAX(1.0, expectedDrawableSize.width);
        NSUInteger drawableHeight = (NSUInteger)MAX(1.0, expectedDrawableSize.height);
        if (_drawable && [self mglDrawableTexture]) {
            drawableWidth = (NSUInteger)mglRenderPassTextureInfo([self mglDrawableTexture]).width;
            drawableHeight = (NSUInteger)mglRenderPassTextureInfo([self mglDrawableTexture]).height;
        }

        if (!MGL_STATE(ctx)->caps.scissor_test) {
            MGL_STATE(ctx)->var.scissor_box[0] = 0;
            MGL_STATE(ctx)->var.scissor_box[1] = 0;
        }
        MGL_STATE(ctx)->var.scissor_box[2] = (GLint)drawableWidth;
        MGL_STATE(ctx)->var.scissor_box[3] = (GLint)drawableHeight;
    }

    mglPassManagerInstallNewRenderPassDescriptor(_renderPassManager);
    if (!_renderPassManager->state->renderPassStateOwner) {
        NSLog(@"MGL RENDERPASS ERROR: failed to allocate render pass state owner");
        return false;
    }

    // Configure color/depth/stencil attachments based on FBO type
    if (MGL_STATE(ctx)->framebuffer) {
        RETURN_FALSE_ON_FAILURE(
            mglRenderPassConfigureUserFBOAttachments((__bridge void *)self));
    } else {
        RETURN_FALSE_ON_FAILURE([self configureDefaultFramebufferAttachmentsLocked]);
    }
    [self ensureTransientDepthForDefaultFramebufferLocked];

    // Capture clear state before load/store resolution for diagnostic logging
    GLuint fboColorClearCount = 0;
    GLbitfield fboColorClearMask = 0;
    GLbitfield fboColorAttachment0ClearMask = 0;

    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLbitfield defaultClearMask = MGL_STATE(ctx)->default_fbo_clear_bitmask;
    GLbitfield fboDepthClearMaskBefore = fbo ? fbo->depth.clear_bitmask : 0u;
    GLbitfield fboStencilClearMaskBefore = fbo ? fbo->stencil.clear_bitmask : 0u;

    if (fbo) {
        [self configureUserFBOLoadStoreActionsLocked:&fboColorClearCount
                                  fboColorClearMask:&fboColorClearMask
                     fboColorAttachment0ClearMask:&fboColorAttachment0ClearMask];
    } else {
        [self configureDefaultFramebufferLoadStoreActionsLocked];
    }

    [self logRenderPassClearResolveLocked:renderEncoderCall
                      traceRenderEncoder:traceRenderEncoder
                        fboColorClearCount:fboColorClearCount
                         fboColorClearMask:fboColorClearMask
            fboColorAttachment0ClearMask:fboColorAttachment0ClearMask
                 fboDepthClearMaskBefore:fboDepthClearMaskBefore
               fboStencilClearMaskBefore:fboStencilClearMaskBefore
                             defaultClearMask:defaultClearMask
                                         fbo:fbo];

    RETURN_FALSE_ON_FAILURE(mglRenderPassFinalizeRenderPassDescriptor(
        (__bridge void *)self, renderEncoderCall, (int)traceRenderEncoder));
    RETURN_FALSE_ON_FAILURE(mglRenderPassCreateRenderEncoderLocked(
        (__bridge void *)self, renderEncoderCall));

    // Apply dynamic state that is not part of the render-pass owner state.
    [self updateCurrentRenderEncoder];

    // Only bind buffers when creating the encoder. Sampled textures depend on the
    // current GL program/MSL reflection and are rebound after the pipeline state is
    // selected for the draw.
    if (MGL_STATE(ctx)->vao)
    {
        MGLEncodeContext encCtx = {
            .render_encoder_owner = _renderPassManager->state->currentRenderEncoderOwner,
        };
        if (mglStageEncodeBindVertexBuffers((__bridge void *)self, &encCtx) == false)
        {
            DEBUG_PRINT("vertex buffer binding failed\n");
            mglRendererRecordGPUError((__bridge void *)self);
            return false;
        }

        if (mglStageEncodeBindFragmentBuffers((__bridge void *)self, &encCtx) == false)
        {
            DEBUG_PRINT("fragment buffer binding failed\n");
            mglRendererRecordGPUError((__bridge void *)self);
            return false;
        }
    }

    // Record successful render encoder creation (final success)
    mglRendererRecordGPUSuccess((__bridge void *)self);
    return true;

    } //     @autoreleasepool
}

- (bool) newCommandBuffer
{
    METAL_LOCK();
    bool result = mglRenderPassNewCommandBufferLocked((__bridge void *)self);
    METAL_UNLOCK();
    return result;
}

- (bool)ensureWritableCommandBuffer:(const char *)reason
{
    METAL_LOCK();
    bool result = mglRenderPassEnsureWritableCommandBufferLocked(
        (__bridge void *)self, reason) != 0;
    METAL_UNLOCK();
    return result;
}


#pragma mark pipeline descriptor
/* Build the renderer pipeline as C ABI value-state. Color/depth/stencil,
 * blend, vertex layout, tessellation, rasterization, topology, and sample
 * count are consumed by the C++ pipeline builder. */
#pragma mark vertex descriptor


- (void) endRenderEncoding
{
    METAL_LOCK();
    mglRendererEndRenderEncodingLocked((__bridge void *)self);
    METAL_UNLOCK();
}


- (BOOL)currentRenderPassUsesTexture:(id)texture
{
    if (!texture || mglRenderEncoderOwnerHasCurrent(
                        _renderPassManager->state->currentRenderEncoderOwner) != 1) {
        return NO;
    }
    if (!_renderPassManager->state->renderPassStateOwner) {
        return NO;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRenderPassColorTextureFor(_renderPassManager->state, i) == texture) {
            return YES;
        }
    }
    if (mglRenderPassDepthTextureFor(_renderPassManager->state) == texture ||
        mglRenderPassStencilTextureFor(_renderPassManager->state) == texture) {
        return YES;
    }

    return NO;
}


- (BOOL)synchronizeRenderPassForTextureReadback:(id)texture
                                         reason:(const char *)reason
{
    BOOL usesTexture = [self currentRenderPassUsesTexture:texture];
    if (!usesTexture) {
        return YES;
    }

    [self endRenderEncoding];

    MGLRenderCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            _renderPassManager->state->currentCommandBufferOwner,
            &commandState)) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    if (commandState.status != MGLCommandBufferStatusNotEnqueued) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    id commandBufferToCommit =
        (__bridge id)mglPassManagerDetachCurrentCommandBufferForSubmission(_renderPassManager);

    @try {
        mglRendererCommitCommandBufferWithAGXRecovery((__bridge void *)self, (__bridge void *)commandBufferToCommit);
        mglRenderPassWaitCommandBuffer(commandBufferToCommit);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: failed to synchronize render pass for texture readback (%s): %@",
              reason ? reason : "texture_readback",
              exception.reason);
        mglRendererRecordGPUError((__bridge void *)self);
        [self newCommandBuffer];
        return NO;
    }

    MGLRenderCommandBufferState committedState = {0};
    (void)mglRenderGetCommandBufferState(
        (__bridge void *)commandBufferToCommit, &committedState);
    if (committedState.has_error) {
        NSLog(@"MGL ERROR: render pass texture readback sync failed (%s): %s",
              reason ? reason : "texture_readback",
              mglRenderCommandBufferErrorDescription(&committedState));
        mglRendererRecordGPUError((__bridge void *)self);
        [self newCommandBuffer];
        return NO;
    }

    return [self newCommandBuffer];
}

// ULTIMATE FAILSAFE: Emergency Metal state reset to recover from corruption
- (bool) processGLState: (bool) draw_command
{
    METAL_LOCK();
    bool result = [self processGLStateLocked:draw_command];
    METAL_UNLOCK();
    return result;
}

- (bool) processGLStateLocked: (bool) draw_command
{
    static uint64_t s_processGLStateCallCount = 0;
    static double s_processGLStateLastCallTime = 0.0;
    static uint64_t s_processGLStateLastCallCount = 0;
    uint64_t processCall = ++s_processGLStateCallCount;
    double processStartSeconds = mglTraceNowSeconds();
    uint64_t processStartNS = mglTraceClockNS();
    bool traceProcess = mglShouldTraceCall(processCall);
    mglLogLoopHeartbeat("processGLState.loop",
                        processCall,
                        processStartSeconds,
                        &s_processGLStateLastCallTime,
                        &s_processGLStateLastCallCount,
                        0.25);
    if (traceProcess) {
        mglTraceLog("MGL TRACE processGLState.begin call=%llu draw=%d",
              (unsigned long long)processCall, draw_command ? 1 : 0);
        mglLogStateSnapshot("processGLState.enter",
                            ctx,
                            _renderPassManager->state->currentCommandBufferOwner,
                            _renderPassManager->state->currentRenderEncoderOwner,
                            _renderPassManager->state->renderPassStateOwner,
                            _drawable);
    }
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                _renderPassManager->state->currentCommandBufferOwner,
                                _renderPassManager->state->currentRenderEncoderOwner,
                                _renderPassManager->state->renderPassStateOwner,
                                _drawable);
        }
        return false;
    }

    uintptr_t earlyCtxAddr = (uintptr_t)ctx;
    const int ctxPtrSane = earlyCtxAddr >= 0x1000 ? 1 : 0;
    if (!ctxPtrSane) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", earlyCtxAddr);
        return false;
    }

    /* Metal corruption recovery is platform materialization — try before plan. */
    static int corruption_recovery_count = 0;
    static int max_recovery_attempts = 3;
    int deviceOk = (_device && ((uintptr_t)_device >= 0x1000)) ? 1 : 0;
    int queueOk = (_commandQueue && ((uintptr_t)_commandQueue >= 0x1000)) ? 1 : 0;
    if (!deviceOk || !queueOk) {
        NSLog(@"MGL CRITICAL: Metal state corruption detected in processGLState!");
        NSLog(@"MGL CRITICAL: device=0x%lx, queue=0x%lx", (uintptr_t)_device,
              (uintptr_t)_commandQueue);
        if (corruption_recovery_count < max_recovery_attempts) {
            NSLog(@"MGL CRITICAL: Attempting Metal state recovery (%d/%d)",
                  corruption_recovery_count + 1, max_recovery_attempts);
            @try {
                mglRenderPassEmergencyResetMetalState((__bridge void *)self);
                corruption_recovery_count++;
                deviceOk = (_device && ((uintptr_t)_device >= 0x1000)) ? 1 : 0;
                queueOk = (_commandQueue && ((uintptr_t)_commandQueue >= 0x1000)) ? 1 : 0;
                if (!deviceOk || !queueOk) {
                    NSLog(@"MGL CRITICAL: Metal recovery failed, aborting operation");
                    return false;
                }
            } @catch (NSException *exception) {
                NSLog(@"MGL CRITICAL: Metal recovery failed: %@", exception);
                return false;
            }
        } else {
            NSLog(@"MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations");
            return false;
        }
    }

    int quarantineBlocks = 0;
    if (draw_command) {
        GLuint blockedProgramKey = mglCurrentRenderProgramKey(ctx);
        if (blockedProgramKey != 0u &&
            _gpuRecovery.interfaceMismatchBlockedProgram != 0 &&
            blockedProgramKey == _gpuRecovery.interfaceMismatchBlockedProgram) {
            CFTimeInterval now = CFAbsoluteTimeGetCurrent();
            if (now < _gpuRecovery.interfaceMismatchBlockedUntil) {
                quarantineBlocks = 1;
                static uint64_t s_quarantineSkipCount = 0;
                s_quarantineSkipCount++;
                if (s_quarantineSkipCount <= 16 || (s_quarantineSkipCount % 1000) == 0) {
                    double remaining = _gpuRecovery.interfaceMismatchBlockedUntil - now;
                    if (remaining < 0.0) remaining = 0.0;
                    NSLog(@"MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw",
                          (unsigned)_gpuRecovery.interfaceMismatchBlockedProgram, remaining);
                }
            }
        }
    }

    MGLRenderCommandBufferState processCommandState = {0};
    int processHasCommand = mglRenderGetCommandBufferOwnerState(
        _renderPassManager->state->currentCommandBufferOwner,
        &processCommandState) == 0;
    const int encoderCurrent = mglRenderEncoderOwnerHasCurrent(
            _renderPassManager->state->currentRenderEncoderOwner) == 1 ? 1 : 0;

    MGLProcessGLStateInputs planIn = {0};
    planIn.has_ctx = 1u;
    planIn.draw_command = draw_command ? 1u : 0u;
    planIn.has_vao = MGL_STATE(ctx)->vao != NULL ? 1u : 0u;
    planIn.dirty_state =
        (MGL_STATE(ctx)->dirty_bits & DIRTY_STATE) ? 1u : 0u;
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
        return false;
    }

    if (plan.clear_rt_sampled_copy) {
        /*
         * This flag is derived from the current draw's final fragment sampler
         * binding.  Clear it before any early render-state refresh so the
         * previous draw cannot disable culling while DIRTY_VAO/FBO is handled.
         */
        mglPassManagerSetCurrentDrawUsesRTSampledCopy(_renderPassManager, 0);
        MGL_FRAME_INC(g_mglProcessDrawCallsSinceSwap);
    }

    if (plan.result == MGL_PGL_RESULT_ABORT) {
        if (draw_command && !planIn.has_vao &&
            plan.process_class == MGL_PROCESS_GL_ABORT) {
            NSLog(@"Error: No VAO defined for ctx\n");
        }
        return false;
    }

    if (plan.non_draw_end_pass_if_fbo_changed) {
        [self endRenderPassIfFramebufferChangedForNonDraw:processCall];
    }
    if (plan.no_vao_clear_path) {
        mglRendererEndRenderEncodingLocked((__bridge void *)self);
        if (!mglRendererValidateMetalObjects((__bridge void *)self)) {
            NSLog(@"MGL WARNING: GPU throttling active - deferring render encoder creation");
            MGL_STATE(ctx)->dirty_bits &= ~DIRTY_STATE;
            return true;
        }
        @try {
            [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_CLEAR];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Render encoder creation failed: %@", exception);
        }
        MGL_STATE(ctx)->dirty_bits &= ~DIRTY_STATE;
        return true;
    }
    if (plan.result == MGL_PGL_RESULT_EARLY_OK) {
        return true;
    }

    if (plan.rotate_finalized_command_buffer) {
        static uint64_t s_rotateFinalizedCount = 0;
        uint64_t rotateHit = ++s_rotateFinalizedCount;
        if (rotateHit <= 16ull || (rotateHit % 500ull) == 0ull) {
            NSLog(@"MGL INFO: processGLState rotating finalized command buffer (status: %ld) hit=%llu",
                  (long)planIn.command_buffer_status,
                  (unsigned long long)rotateHit);
        }
        if (!mglRenderPassNewCommandBufferLocked((__bridge void *)self)) {
            NSLog(@"MGL ERROR: processGLState failed to create a fresh command buffer");
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.new_cb_rotate",
                                    ctx,
                                    _renderPassManager->state->currentCommandBufferOwner,
                                    _renderPassManager->state->currentRenderEncoderOwner,
                                    _renderPassManager->state->renderPassStateOwner,
                                    _drawable);
            }
            return false;
        }
    } else if (plan.create_initial_command_buffer) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: processGLState found NULL command buffer, creating one");
        }
        if (!mglRenderPassNewCommandBufferLocked((__bridge void *)self)) {
            NSLog(@"MGL ERROR: processGLState could not create initial command buffer");
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.new_cb_initial",
                                    ctx,
                                    _renderPassManager->state->currentCommandBufferOwner,
                                    _renderPassManager->state->currentRenderEncoderOwner,
                                    _renderPassManager->state->renderPassStateOwner,
                                    _drawable);
            }
            return false;
        }
    }

    MGLResourceSyncWork resourceSyncWork = {false, false, false};
    if (plan.process_dirty_domains) {
        RETURN_FALSE_ON_FAILURE(mglRenderPassProcessDirtyStateDomains(
            (__bridge void *)self, draw_command ? 1 : 0, &resourceSyncWork));
    }

    /* Phase 2: re-sample encoder/pipeline after dirty-domain materialization. */
    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    MGLProcessGLStateAfterInputs afterIn = {0};
    afterIn.draw_command = draw_command ? 1u : 0u;
    afterIn.encoder_has_current = mglRenderEncoderOwnerHasCurrent(
            _renderPassManager->state->currentRenderEncoderOwner) == 1 ? 1u : 0u;
    afterIn.has_pipeline_state = _pipelineCache.state->pipelineState ? 1u : 0u;
    afterIn.frag_needs_fragcoord =
        fragmentProgram && mglRenderSamplerUnitExplicit(
                               (uint32_t)fragmentProgram->usesFragCoordParams)
            ? 1u
            : 0u;
    afterIn.frag_needs_sample =
        ((fragmentProgram && mglRenderSamplerUnitExplicit(
                                 (uint32_t)fragmentProgram->uses_sample_params)) ||
         _mglInMSSampleDrawLoop)
            ? 1u
            : 0u;
    afterIn.frag_needs_lod_bias =
        fragmentProgram && mglRenderSamplerUnitExplicit(
                               (uint32_t)fragmentProgram->uses_lod_bias)
            ? 1u
            : 0u;
    afterIn.fragment_trace_uses_rt_sampled_copy =
        mglFragmentTextureTraceBindingsUseRTSampledCopy(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS)
            ? 1u
            : 0u;

    MGLProcessGLStateAfterPlan after = {0};
    if (mglRenderProcessGLStateAfterDirty(&afterIn, &after) != 0) {
        return false;
    }

    if (after.recover_nil_encoder) {
        static uint64_t s_nilEncoderRecoveryCount = 0;
        uint64_t nilHit = ++s_nilEncoderRecoveryCount;
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            NSLog(@"MGL WARNING: processGLState - current render encoder is nil, attempting recovery hit=%llu",
                  (unsigned long long)nilHit);
            mglLogRenderPassLifecycle("nil-encoder-before-recovery",
                                      nilHit,
                                      ctx,
                                      _renderPassManager->state->currentCommandBufferOwner,
                                      _renderPassManager->state->currentRenderEncoderOwner,
                                      _renderPassManager->state->renderPassStateOwner,
                                      (__bridge void *)_drawable,
                                      _renderPassManager->state->renderPassFramebuffer,
                                      _renderPassManager->state->renderPassFramebufferName,
                                      _renderPassManager->state->renderPassDrawBuffer,
                                      _renderPassManager->state->renderPassDrawBufferCount);
        }
        RETURN_FALSE_ON_FAILURE(
            [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_NIL]);
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            mglLogRenderPassLifecycle("nil-encoder-after-recovery",
                                      nilHit,
                                      ctx,
                                      _renderPassManager->state->currentCommandBufferOwner,
                                      _renderPassManager->state->currentRenderEncoderOwner,
                                      _renderPassManager->state->renderPassStateOwner,
                                      (__bridge void *)_drawable,
                                      _renderPassManager->state->renderPassFramebuffer,
                                      _renderPassManager->state->renderPassFramebufferName,
                                      _renderPassManager->state->renderPassDrawBuffer,
                                      _renderPassManager->state->renderPassDrawBufferCount);
        }
    }

    if (after.ensure_pass_matches_fbo) {
        RETURN_FALSE_ON_FAILURE(mglRenderPassEnsureCurrentRenderPassMatchesFramebufferForDraw((__bridge void *)self));
        [self updateCurrentRenderEncoder];
    }

    if (draw_command && kMGLVerbosePipelineLogs) {
        static uint64_t s_drawPipelineLookupCount = 0;
        s_drawPipelineLookupCount++;
        if (s_drawPipelineLookupCount <= 256ull || (s_drawPipelineLookupCount % 1000ull) == 0ull) {
            Program *lookupProgram = mglResolveProgramFromState(ctx);
            Program *lookupVertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *lookupFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            GLuint lookupProgramName = mglCurrentRenderProgramKey(ctx);
            Framebuffer *lookupFBO = MGL_STATE(ctx)->framebuffer;
            GLuint lookupFBOName = lookupFBO ? lookupFBO->name : 0;
            fprintf(stderr, "MGL Draw current program key=%u mono=%p vs=%u fs=%u\n",
                    (unsigned)lookupProgramName,
                    (void *)lookupProgram,
                    lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
                    lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u);
            NSLog(@"MGL DRAW pipeline lookup result=%p key=%u vs=%u fs=%u vao=%p fbo=%u",
                  _pipelineCache.state->pipelineState,
                  (unsigned)lookupProgramName,
                  lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
                  lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u,
                  MGL_STATE(ctx)->vao,
                  (unsigned)lookupFBOName);
        }
    }

    if (after.fail_nil_pipeline) {
        static uint64_t nil_pipeline_count = 0;
        nil_pipeline_count++;
        if (nil_pipeline_count <= 8 || (nil_pipeline_count % 1000) == 0) {
            mglTraceLog("MGL DRAW SKIP: pipelineState is nil, forcing rebuild (occurrence=%llu)",
                          (unsigned long long)nil_pipeline_count);
        }
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_pipeline",
                                ctx,
                                _renderPassManager->state->currentCommandBufferOwner,
                                _renderPassManager->state->currentRenderEncoderOwner,
                                _renderPassManager->state->renderPassStateOwner,
                                _drawable);
        }
        return false;
    }

    if (after.validate_attachments) {
        RETURN_FALSE_ON_FAILURE(mglRenderPassValidateAttachmentsAndPipelineFormats(
            (__bridge void *)self, traceProcess ? 1 : 0));
    }

    if (after.set_pipeline) {
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
            NSLog(@"MGL ERROR: processGLState - setRenderPipelineState failed: %@", exception.reason);
            mglMarkRendererDirtyBits(ctx->active_state,
                                     DIRTY_PROGRAM | DIRTY_VAO |
                                     DIRTY_FBO | DIRTY_RENDER_STATE);
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.set_pipeline",
                                    ctx,
                                    _renderPassManager->state->currentCommandBufferOwner,
                                    _renderPassManager->state->currentRenderEncoderOwner,
                                    _renderPassManager->state->renderPassStateOwner,
                                    _drawable);
            }
            return false;
        }
    }

    if (after.sync_resources) {
        RETURN_FALSE_ON_FAILURE(mglRendererSyncResourceBindingsForContext(
            (__bridge void *)self, ctx, &resourceSyncWork));
    }

    if (after.bind_frag_coord_slot) {
        BOOL useFragCoordParams = afterIn.frag_needs_fragcoord ? YES : NO;
        BOOL useSampleParams = afterIn.frag_needs_sample ? YES : NO;
        NSUInteger passHeight = mglRenderPassRenderTargetHeightFor(_renderPassManager->state);
        if (passHeight == 0) {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS && passHeight == 0; i++) {
                id color = mglRenderPassColorTextureFor(_renderPassManager->state, i);
                passHeight = color ? mglRenderPassTextureInfo(color).height : 0;
            }
            if (passHeight == 0 && mglRenderPassDepthTextureFor(_renderPassManager->state)) {
                passHeight = mglRenderPassTextureInfo(mglRenderPassDepthTextureFor(_renderPassManager->state)).height;
            }
            if (passHeight == 0 && mglRenderPassStencilTextureFor(_renderPassManager->state)) {
                passHeight = mglRenderPassTextureInfo(mglRenderPassStencilTextureFor(_renderPassManager->state)).height;
            }
        }

        uint32_t numSamples = 1;
        uint32_t sampleBuffers = 0;
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
        if (fbo && (fbo->color_attachment_bitfield & 1u)) {
            FBOAttachment *att = &fbo->color_attachments[0];
            Texture *tex = NULL;
            if (mglRenderTargetIsRenderbuffer((uint32_t)att->textarget) && att->buf.rbo)
                tex = att->buf.rbo->tex;
            else
                tex = att->buf.tex;
            if (tex) {
                (void)mglRenderTextureSampleParams(
                    (uint32_t)tex->target, tex->samples, &numSamples,
                    &sampleBuffers);
            }
        } else {
            id rpColor0 = mglRenderPassColorTextureFor(_renderPassManager->state, 0);
            if (rpColor0) {
                NSUInteger sc = mglRenderPassTextureInfo(rpColor0).sample_count;
                if (sc > 1) {
                    sampleBuffers = 1;
                    numSamples = (uint32_t)sc;
                }
            }
        }
        float fragCoordParams[4] = {0.f, 0.f, 0.f, 0.f};
        mglRenderFillFragCoordSlot(
            useFragCoordParams ? 1 : 0, useSampleParams ? 1 : 0,
            (uint32_t)passHeight,
            mglRenderClipOriginIsLowerLeft(
                (uint32_t)MGL_STATE(ctx)->var.clip_origin),
            numSamples, sampleBuffers,
            _mglInMSSampleDrawLoop ? 1 : 0,
            (uint32_t)_mglForcedMSSampleId, fragCoordParams);
        mglRenderSetRenderBytesForOwner(
            _renderPassManager->state->currentRenderEncoderOwner,
            fragCoordParams, sizeof(fragCoordParams),
            MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLFragCoordParamsBufferIndex);
        mglBindingInvalidateLastBoundFragmentBufferAtIndex((__bridge void *)self, kMGLFragCoordParamsBufferIndex);
    }

    if (after.bind_lod_bias_slot) {
        const GLfloat biasmax = STATE(var).max_texture_lod_bias;
        float lodBiasArr[TEXTURE_UNITS];
        for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
            Texture *tex = MGL_STATE(ctx)->active_textures[unit];
            Sampler *smp = MGL_STATE(ctx)->texture_samplers[unit];

            lodBiasArr[unit] = smp ? smp->params.lod_bias
                                   : (tex ? tex->params.lod_bias : 0.0f);
        }
        mglRenderClampLodBiasArray(lodBiasArr, TEXTURE_UNITS, biasmax);
        mglRenderSetRenderBytesForOwner(
            _renderPassManager->state->currentRenderEncoderOwner,
            lodBiasArr, sizeof(lodBiasArr),
            MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLLodBiasBufferIndex);
        mglBindingInvalidateLastBoundFragmentBufferAtIndex((__bridge void *)self, kMGLLodBiasBufferIndex);

        mglRenderSetRenderBytesForOwner(
            _renderPassManager->state->currentRenderEncoderOwner,
            &biasmax, sizeof(biasmax),
            MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLLodBiasMaxBufferIndex);
        mglBindingInvalidateLastBoundFragmentBufferAtIndex((__bridge void *)self, kMGLLodBiasMaxBufferIndex);
    }

    if (after.maybe_mark_rt_sampled_copy) {
        mglPassManagerSetCurrentDrawUsesRTSampledCopy(_renderPassManager, 1);
        [self updateCurrentRenderEncoder];
    }

    double processElapsedUs = (mglTraceClockNS() - processStartNS) / 1000.0;
    if (traceProcess) {
        mglTraceLog("MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.1fus",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedUs);
        mglLogStateSnapshot("processGLState.exit.ok",
                            ctx,
                            _renderPassManager->state->currentCommandBufferOwner,
                            _renderPassManager->state->currentRenderEncoderOwner,
                            _renderPassManager->state->renderPassStateOwner,
                            _drawable);
    } else if (processElapsedUs >= 25.0) {
        mglTraceLog("MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.1fus",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedUs);
    }
    return true;
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
/*
 * Pipeline Sync domain (Pipeline Sync domain). PSO build/reuse logic moved verbatim from processGLStateLocked:
 * generates pipeline+vertex descriptor, queries/builds PSO cache, interface-mismatch
 * circuit breaker, failure fallback chain. Only operates on Metal pipeline state, state is read via ctx (same as before the move).
 * deferredBufferMap is passed in by the caller (deferred buffer mapping flag for nil pipeline).
 * Returns false to indicate this draw should be skipped (equivalent to the original inline return false semantics).
 */
- (bool)syncPipelineStateWithDeferredBufferMap:(bool)deferredBufferMapForPipelineBuild
{
            GLMState *state = MGL_STATE(ctx);
            /* Force a rebind of the pipeline state on the next setRenderPipelineState
             * call. Dirty program/VAO/FBO/render-state may rebuild or reuse the
             * pipeline, but the encoder still needs the binding re-issued.
             *
             * Task 5 gated fast path: when MGL_PSO_DEDUP is enabled (default ON)
             * and the render
             * encoder is unchanged (the C++ binding cache is valid) and the resolved
             * pipeline state pointer is identical to the previously bound
             * state matches the C++ binding cache, the nil assignment
             * is skipped. This allows the dedup check in
             * processGLStateLocked:'s setRenderPipelineState: path to
             * recognize the encoder already has the correct PSO bound and
             * skip the redundant MTL call. If any condition is false, the
             * original conservative nil assignment executes. */
            if (_pipelineCache.state->psoDedupEnabled &&
                mglBindingStateIsValid(_bindingStateOwner) &&
                mglBindingStatePipelineMatches(
                    _bindingStateOwner,
                _pipelineCache.state->pipelineState)) {
                MGL_PERF_INC(g_mglPSODedupHitsSinceSwap);
            } else {
                mglRenderBindingSetPipelineState(_bindingStateOwner, NULL);
                MGL_PERF_INC(g_mglPSODedupMissesSinceSwap);
            }
            CFTimeInterval now = CFAbsoluteTimeGetCurrent();
            bool skipPipelineBuild = false;
            Program *currentVertexProgram = _tessellation.nativeTESActive
                ? _tessellation.nativeTESProgram
                : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *currentFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            GLuint currentProgramName = mglCurrentRenderProgramKey(ctx);
            VertexArray *currentVAO = state->vao;
            Framebuffer *currentFBO = mglRendererGetValidatedFramebuffer(ctx, "processGLState.currentFBO");
            GLuint currentFBOName = currentFBO ? currentFBO->name : 0;

            // Program-level breaker (independent of render-pass signature) to avoid
            // mismatch storms where color/depth/stencil signatures keep changing.
            if (_pipelineCache.state->pipelineState != nil &&
                currentProgramName != 0 &&
                currentProgramName == _gpuRecovery.programMismatchProgramName &&
                now < _gpuRecovery.programMismatchRetryAfter) {
                static uint64_t s_programMismatchSkipCount = 0;
                s_programMismatchSkipCount++;
                if (s_programMismatchSkipCount <= 16 || (s_programMismatchSkipCount % 1000ull) == 0ull) {
                    double remaining = _gpuRecovery.programMismatchRetryAfter - now;
                    if (remaining < 0.0) remaining = 0.0;
                    NSLog(@"MGL WARNING: Program-level mismatch breaker active (program=%u, %.2fs remaining), skipping draw",
                          (unsigned)currentProgramName,
                          remaining);
                }
                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

	            if (now < _gpuRecovery.pipelineRetryAfter) {
	                BOOL retryAppliesToCurrentProgram =
	                    (currentProgramName != 0 &&
	                     (currentProgramName == _gpuRecovery.interfaceMismatchProgramName ||
	                      currentProgramName == _gpuRecovery.programMismatchProgramName ||
	                      currentProgramName == _gpuRecovery.interfaceMismatchBlockedProgram));

	                if (retryAppliesToCurrentProgram) {
	                    if (_pipelineCache.state->pipelineState) {
		                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	                    // Keep existing pipeline, but do not early-return before setRenderPipelineState.
		                    skipPipelineBuild = true;
	                    } else {
	                        _gpuRecovery.pipelineRetryAfter = 0.0;
	                        _gpuRecovery.programMismatchRetryAfter = 0.0;
	                        _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
	                    }
		                } else {
	                    static uint64_t s_retryBypassCount = 0;
	                    s_retryBypassCount++;
	                    if (s_retryBypassCount <= 16 || (s_retryBypassCount % 1000ull) == 0ull) {
	                        NSLog(@"MGL PIPELINE RETRY bypass global retry for unrelated program=%u mismatchProgram=%u blockedProgram=%u",
	                              (unsigned)currentProgramName,
	                              (unsigned)_gpuRecovery.interfaceMismatchProgramName,
	                              (unsigned)_gpuRecovery.interfaceMismatchBlockedProgram);
	                    }
	                }
	            }

            if (!skipPipelineBuild) {
            // Build the only renderer pipeline representation: C ABI value-state.
            MGLRenderPipelineDescriptorState psoState = {0};
            id psoVertexFunction = nil;
            id psoFragmentFunction = nil;
            uint32_t builtColor0Format = mglRenderInvalidPixelFormat();
            uint32_t builtDepthFormat = mglRenderInvalidPixelFormat();
            uint32_t builtStencilFormat = mglRenderInvalidPixelFormat();

            mglRendererUpdateBlendStateCache((__bridge void *)self);
            state->dirty_bits &= ~DIRTY_ALPHA_STATE;
            if (getenv("MGL_TOPO_TRACE") != NULL) {
                fprintf(stderr, "MGLTOPO tessCompute=%d active=%d prog=%p topology=%u\n",
                        (int)(_tessellation.tessComputeActive ? 1 : 0),
                        (int)(_tessellation.tessVertexRenderActive ? 1 : 0),
                        (void *)_tessellation.tessComputeProgram,
                        (unsigned)psoState.input_primitive_topology);
                fflush(stderr);
            }
            MGLRenderPassPipelineFunctions psoFunctions = { NULL, NULL };
            if (!mglRenderPassGeneratePipelineDescriptorState(
                    (__bridge void *)self, &psoState, &psoFunctions)) {
                NSLog(@"MGL PIPELINE CREATE fail error=generatePipelineDescriptorState returned NO");
                mglRenderPassInvalidateCurrentPipelineState(
                (__bridge void *)self, "pipeline descriptor failure");
                _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                mglMarkRendererDirtyBits(state,
                                         DIRTY_PROGRAM | DIRTY_VAO |
                                         DIRTY_FBO | DIRTY_RENDER_STATE);
                return false;
            }
            /* Borrowed Metal functions: take them with __bridge so ARC owns
             * the +1 it releases (the program/cache keeps them alive). */
            psoVertexFunction = (__bridge id)psoFunctions.vertex_function;
            psoFragmentFunction = (__bridge id)psoFunctions.fragment_function;
            builtColor0Format = psoState.color_format[0];
            builtDepthFormat = psoState.depth_format;
            builtStencilFormat = psoState.stencil_format;

            // Circuit breaker for repeated VS/FS interface mismatch.
            if (now < _gpuRecovery.interfaceMismatchRetryAfter &&
                currentProgramName == _gpuRecovery.interfaceMismatchProgramName &&
                builtColor0Format == _gpuRecovery.interfaceMismatchColor0Format &&
                builtDepthFormat == _gpuRecovery.interfaceMismatchDepthFormat &&
                builtStencilFormat == _gpuRecovery.interfaceMismatchStencilFormat) {
                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

            BOOL hasPipelineCacheKey = NO;
            bool pipelineResolvedFromCache = false;
            uint64_t pipelineSig = 0;
            uint64_t vertexSig = 0;
            /* Function-scope key words: filled inside the lookup block below,
             * read by the miss path after it.  Only meaningful when
             * currentProgramName != 0. */
            uint64_t keyWords[MGL_PIPELINE_CACHE_KEY_WORDS] = {0};

            if (!pipelineResolvedFromCache && currentProgramName != 0) {
                pipelineSig = mglPipelineDescriptorSignatureFromState(&psoState);
                vertexSig = mglVertexDescriptorSignatureFromState(&psoState);

                /* Keep descriptor signatures and linked Program identities
                 * lossless. GL names can be reused and a Program can relink
                 * without changing its name.
                 *
                 * tessVertexRenderActive must be part of the key: it decides
                 * whether the raster vertex function is the TES render-vertex
                 * function itself or the generated slot-28 record passthrough
                 * (see the tessPassthroughFunction selection in the pipeline
                 * descriptor), so an isolines/point-mode program that is drawn
                 * through both paths -- non-indexed draws take the vertex path,
                 * indexed ones fall back to the compute expansion -- would
                 * otherwise reuse the first pipeline for the second draw and
                 * rasterize the record stream with the wrong ABI. */
                uint64_t primaryKey = (((uint64_t)currentProgramName << 32)
                                     | (((uint64_t)state->var.clip_origin & 0xFu) << 28)
                                     | (((uint64_t)state->var.clip_depth_mode & 0xFu) << 24)
                                     | (_tessellation.nativeTESActive ? (1ull << 23) : 0ull)
                                     | (_tessellation.tessVertexCaptureActive ? (1ull << 22) : 0ull)
                                     | (_geometry.expansionActive ? (1ull << 21) : 0ull)
                                     | (_tessellation.cullDistanceCaptureActive ? (1ull << 20) : 0ull)
                                     | (_tessellation.tessComputeActive ? (1ull << 19) : 0ull)
                                     | (_tessellation.tessVertexRenderActive ? (1ull << 18) : 0ull));
                uint64_t vertexInstance = currentVertexProgram
                    ? currentVertexProgram->pipeline_cache_instance_id : 0u;
                if (_geometry.expansionActive && _geometry.program) {
                    /* The raster vertex function is generated from both the
                     * real VS/FS interface and the GS output record.  Fold
                     * both program identities plus the emulated-MS layer
                     * convention into the key so an old PTVS/PSO cannot be
                     * reused after a GS or framebuffer change. */
                    vertexInstance = mglGeometryPipelineFunctionKey(
                        currentVertexProgram, _geometry.program,
                        mglGeometryPassthroughLayerStride(ctx));
                }
                uint64_t vertexGeneration = currentVertexProgram
                    ? currentVertexProgram->pipeline_cache_generation : 0u;
                uint64_t fragmentInstance = currentFragmentProgram
                    ? currentFragmentProgram->pipeline_cache_instance_id : 0u;
                uint64_t fragmentGeneration = currentFragmentProgram
                    ? currentFragmentProgram->pipeline_cache_generation : 0u;
                keyWords[0] = primaryKey;
                keyWords[1] = vertexInstance;
                keyWords[2] = vertexGeneration;
                keyWords[3] = fragmentInstance;
                keyWords[4] = fragmentGeneration;
                keyWords[5] = pipelineSig;
                keyWords[6] = vertexSig;
                /* Hit path uses the reusable zero-alloc query key.  The key
                 * is only valid for lookups; the miss path below allocates a
                 * fresh key for the store/compile path so overwriteWords:
                 * cannot corrupt cache dictionaries. */
                hasPipelineCacheKey = YES;

                /* Two-level cache lookup:
                 * Level 1: PSO cache (fastest - compiled pipeline ready to use)
                 * Level 2: Descriptor cache (fast - skip expensive descriptor regeneration)
                 * On double miss: regenerate descriptor + compile PSO */
                id cachedPipeline = nil;
                id cachedVertexFunction = nil;
                id cachedFragmentFunction = nil;
                BOOL cachedFunctionMetadataPresent = [_pipelineCache
                    lookupPipelineForWords:keyWords
                    pipeline:&cachedPipeline
                    vertexFunction:&cachedVertexFunction
                    fragmentFunction:&cachedFragmentFunction];
                if (cachedPipeline) {
                    /* PSO cache hit - fastest path */
                    static uint64_t s_pipelineCacheHitCount = 0;
                    s_pipelineCacheHitCount++;
                    MGL_PERF_INC(g_mglPipelineCacheHitsSinceSwap);
                    if (kMGLVerbosePipelineLogs &&
                            (s_pipelineCacheHitCount <= 128ull || (s_pipelineCacheHitCount % 1000ull) == 0ull)) {
                        NSLog(@"MGL PIPELINE CACHE hit program=%u vao=%p fbo=%u key=%@",
                              (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName,
                              [NSString stringWithFormat:@"%016llx/%016llx/%016llx",
                               (unsigned long long)keyWords[0],
                               (unsigned long long)keyWords[5],
                               (unsigned long long)keyWords[6]]);
                    }

                    [_pipelineCache activatePipelineState:cachedPipeline
                                           color0Format:builtColor0Format
                                            depthFormat:builtDepthFormat
                                          stencilFormat:builtStencilFormat
                                            programName:currentProgramName
                                         vertexFunction:cachedFunctionMetadataPresent
                                             ? cachedVertexFunction
                                             : psoVertexFunction
                                       fragmentFunction:cachedFunctionMetadataPresent
                                             ? cachedFragmentFunction
                                             : psoFragmentFunction];
                    pipelineResolvedFromCache = true;
                    /* Hit path deliberately skips the LRU touch: touching
                     * would require copying the query-keyed object that must
                     * never enter the LRU (see pipelineQueryKeyForWords:),
                     * reintroducing the per-draw alloc this avoids.  Mirrors
                     * the depth-stencil cache policy. */

	                    // Mirror successful compile-side breaker resets.
	                    _gpuRecovery.interfaceMismatchStreak = 0;
	                    _gpuRecovery.interfaceMismatchProgramName = 0;
	                    _gpuRecovery.interfaceMismatchColor0Format = mglRenderInvalidPixelFormat();
	                    _gpuRecovery.interfaceMismatchDepthFormat = mglRenderInvalidPixelFormat();
	                    _gpuRecovery.interfaceMismatchStencilFormat = mglRenderInvalidPixelFormat();
	                    _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
	                    if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
	                        _gpuRecovery.programMismatchProgramName = 0;
	                        _gpuRecovery.programMismatchRetryAfter = 0.0;
	                        _gpuRecovery.programMismatchStreak = 0u;
	                    }
	                    if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                        _gpuRecovery.interfaceMismatchBlockedProgram = 0;
                        _gpuRecovery.interfaceMismatchBlockedUntil = 0.0;
                        _gpuRecovery.interfaceMismatchBlockedStreak = 0u;
                    }
	                }
	            }

	            // PROPER AGX VIRTUALIZATION COMPATIBILITY: Fix root cause while maintaining Metal functionality
            if (!pipelineResolvedFromCache) {
                /* Compile/store path needs its own key object: the reusable
                 * query key words are overwritten on every lookup and must
                 * never be retained by the cache dictionaries/LRU.  One heap
                 * allocation on a cache miss is negligible against the PSO
                 * compile itself. */
                const uint64_t *storeKeyWords = hasPipelineCacheKey
                    ? keyWords : NULL;
                return [self buildPipelineStateOnCacheMissWithState:&psoState
                                                     vertexFunction:psoVertexFunction
                                                   fragmentFunction:psoFragmentFunction
                                                       cacheKeyWords:storeKeyWords
                                                        pipelineSig:pipelineSig
                                                         vertexSig:vertexSig
                                                builtColor0Format:builtColor0Format
                                                 builtDepthFormat:builtDepthFormat
                                               builtStencilFormat:builtStencilFormat
                                                      programName:currentProgramName
                                                             now:now];
            }

                if (deferredBufferMapForPipelineBuild && _pipelineCache.state->pipelineState != nil) {
                    RETURN_FALSE_ON_FAILURE(mglRendererMapBuffersToMTL((__bridge void *)self));
                    deferredBufferMapForPipelineBuild = false;
                }

	            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	            }

    return true;
}

/* Build final/simple/safe PSO variants from value-state in the C++ builder.
 * The same owner also handles descriptor caching and binary archives. */
- (bool)buildPipelineStateOnCacheMissWithState:(const MGLRenderPipelineDescriptorState *)pipelineState
                                vertexFunction:(id)vertexFunction
                              fragmentFunction:(id)fragmentFunction
                                  cacheKeyWords:(const uint64_t *)pipelineCacheKeyWords
                                    pipelineSig:(uint64_t)pipelineSig
                                     vertexSig:(uint64_t)vertexSig
                            builtColor0Format:(uint32_t)builtColor0Format
                             builtDepthFormat:(uint32_t)builtDepthFormat
                           builtStencilFormat:(uint32_t)builtStencilFormat
                                 programName:(GLuint)currentProgramName
                                        now:(CFTimeInterval)now
{
    GLMState *state = MGL_STATE(ctx);
    Program *currentProgram = mglResolveProgramFromState(ctx);
    Program *currentVertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *currentFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    VertexArray *currentVAO = state->vao;
    Framebuffer *currentFBO = mglRendererGetValidatedFramebuffer(ctx, "buildPipelineCacheOnCacheMiss.currentFBO");
    GLuint currentFBOName = currentFBO ? currentFBO->name : 0;


    MGLRenderPipelineDescriptorState finalState = *pipelineState;
    BOOL stateFromCache = NO;

    /* Check descriptor state cache on PSO miss; cache new states for reuse. */
    if (pipelineCacheKeyWords) {
        MGLRenderPipelineDescriptorState cachedState = {0};
        if ([_pipelineCache pipelineDescriptorStateForWords:pipelineCacheKeyWords
                                                      state:&cachedState]) {
            /* Descriptor cache hit - reuse cached state instead of regenerating */
            finalState = cachedState;
            stateFromCache = YES;
            static uint64_t s_descriptorCacheHitCount = 0;
            s_descriptorCacheHitCount++;
            if (kMGLVerbosePipelineLogs && s_descriptorCacheHitCount <= 64ull) {
                NSLog(@"MGL DESCRIPTOR CACHE hit program=%u key=%@ (total %llu)",
                (unsigned)currentProgramName,
                [NSString stringWithFormat:@"%016llx/%016llx/%016llx",
                 (unsigned long long)pipelineCacheKeyWords[0],
                 (unsigned long long)pipelineCacheKeyWords[5],
                 (unsigned long long)pipelineCacheKeyWords[6]],
                (unsigned long long)s_descriptorCacheHitCount);
            }
        }
    }

    MGL_PERF_INC(g_mglPipelineCacheMissesSinceSwap);
    MGLRenderPipelineDescriptorState successfulState = {0};
    BOOL haveSuccessfulState = NO;
    id previousPipelineState = (__bridge id)_pipelineCache.state->pipelineState;
    id compiledPSO = nil;
    bool pipelineReusedPrevious = false;
    char cppError[512] = {0};

    void *psoPtr = NULL;

    @try {
        static uint64_t s_pipelineCreateBeginCount = 0;
        s_pipelineCreateBeginCount++;
        if (kMGLVerbosePipelineLogs &&
        (s_pipelineCreateBeginCount <= 128ull || (s_pipelineCreateBeginCount % 500ull) == 0ull)) {
            NSLog(@"MGL PIPELINE CREATE begin program=%u vao=%p fbo=%u",
            (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName);
        }

        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL INFO: Creating Metal pipeline state with AGX virtualization compatibility...");
        }

        /* Test hook (air_pipeline_safe_fallback regression): force the
         * pipeline-creation exception so the safe-fallback branch below is
         * exercised deterministically. */
        if (mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE")) {
            NSLog(@"MGL TEST: forcing safe-fallback pipeline path");
            @throw [NSException exceptionWithName:@"MGLForcedSafeFallback"
                                           reason:@"synthetic pipeline creation failure (test hook)"
                                         userInfo:nil];
        }

        psoPtr = NULL;
        cppError[0] = '\0';
        if ([_pipelineCache
                createRenderPipelineFromState:&finalState
                vertexFunction:(__bridge void *)vertexFunction
                fragmentFunction:fragmentFunction
                    ? (__bridge void *)fragmentFunction : NULL
                pipelineOut:&psoPtr
                errorMessage:cppError
                errorCapacity:sizeof(cppError)] != 0 || !psoPtr) {
            if (cppError[0]) {
                NSLog(@"MGL METALCPP PSO fallback: %s", cppError);
            }
        } else {
            compiledPSO = (__bridge_transfer id)psoPtr;
        }
        if (compiledPSO) {
            mglMetalCountCreate(MGLMetalKindPSO);
            successfulState = finalState;
            haveSuccessfulState = YES;
        }

        if (!compiledPSO) {
            NSString *errDesc = cppError[0]
                ? [NSString stringWithUTF8String:cppError] : @"";
            BOOL isInterfaceMismatch =
                [errDesc containsString:@"mismatching vertex shader output"] ||
                [errDesc containsString:@"not written by vertex shader"];

            if (isInterfaceMismatch) {
                const char *errText = cppError[0] ? cppError : "";
                mglWriteProgramMSLDump(currentVertexProgram, errText);
                if (currentFragmentProgram && currentFragmentProgram != currentVertexProgram) {
                    mglWriteProgramMSLDump(currentFragmentProgram, errText);
                } else if (!currentVertexProgram) {
                    mglWriteProgramMSLDump(currentProgram, errText);
                }
                BOOL sameProgram =
                (_pipelineCache.state->pipelineProgramName != 0 &&
                _pipelineCache.state->pipelineProgramName == currentProgramName &&
                _pipelineCache.state->pipelineVertexFunction == (__bridge void *)vertexFunction &&
                _pipelineCache.state->pipelineFragmentFunction == (__bridge void *)fragmentFunction);
                BOOL colorCompatible = mglRenderPipelineFormatCompatible(
                    (uint32_t)_pipelineCache.state->pipelineColor0Format,
                    builtColor0Format) != 0;
                BOOL depthCompatible = mglRenderPipelineFormatCompatible(
                    (uint32_t)_pipelineCache.state->pipelineDepthFormat,
                    builtDepthFormat) != 0;
                BOOL stencilCompatible = mglRenderPipelineFormatCompatible(
                    (uint32_t)_pipelineCache.state->pipelineStencilFormat,
                    builtStencilFormat) != 0;

                if (previousPipelineState && sameProgram && colorCompatible && depthCompatible && stencilCompatible) {
                    NSLog(@"MGL WARNING: Interface mismatch for program %u; not reusing previous PSO",
                    (unsigned)currentProgramName);
                    compiledPSO = nil;
                    pipelineReusedPrevious = false;
                    _gpuRecovery.interfaceMismatchProgramName = currentProgramName;
                    _gpuRecovery.interfaceMismatchColor0Format = builtColor0Format;
                    _gpuRecovery.interfaceMismatchDepthFormat = builtDepthFormat;
                    _gpuRecovery.interfaceMismatchStencilFormat = builtStencilFormat;
                    _gpuRecovery.interfaceMismatchStreak = 1u;
                    _gpuRecovery.interfaceMismatchRetryAfter = now + 0.10;
                    _gpuRecovery.pipelineRetryAfter = _gpuRecovery.interfaceMismatchRetryAfter;
                } else {
                    BOOL sameMismatchSignature =
                    (currentProgramName == _gpuRecovery.interfaceMismatchProgramName &&
                    builtColor0Format == _gpuRecovery.interfaceMismatchColor0Format &&
                    builtDepthFormat == _gpuRecovery.interfaceMismatchDepthFormat &&
                    builtStencilFormat == _gpuRecovery.interfaceMismatchStencilFormat);
                    if (sameMismatchSignature) {
                        if (_gpuRecovery.interfaceMismatchStreak < UINT32_MAX) {
                            _gpuRecovery.interfaceMismatchStreak++;
                        }
                    } else {
                        _gpuRecovery.interfaceMismatchStreak = 1;
                        _gpuRecovery.interfaceMismatchProgramName = currentProgramName;
                        _gpuRecovery.interfaceMismatchColor0Format = builtColor0Format;
                        _gpuRecovery.interfaceMismatchDepthFormat = builtDepthFormat;
                        _gpuRecovery.interfaceMismatchStencilFormat = builtStencilFormat;
                    }

                    // Exponential backoff: 0.10, 0.20, 0.40, 0.80, 1.60, capped at 2.00 sec.
                    uint32_t cappedShift = (_gpuRecovery.interfaceMismatchStreak > 5u) ? 4u : (_gpuRecovery.interfaceMismatchStreak - 1u);
                    double retryDelay = 0.10 * (double)(1u << cappedShift);
                    if (retryDelay > 2.0) {
                        retryDelay = 2.0;
                    }
                    _gpuRecovery.interfaceMismatchRetryAfter = now + retryDelay;

                    if (_gpuRecovery.interfaceMismatchStreak <= 5u || (_gpuRecovery.interfaceMismatchStreak % 200u) == 0u) {
                        NSLog(@"MGL WARNING: Interface mismatch (program=%u, streak=%u), throttling retries for %.2fs",
                        (unsigned)currentProgramName,
                        (unsigned)_gpuRecovery.interfaceMismatchStreak,
                        retryDelay);
                    }

                    // Program-level breaker update (ignores attachment signature).
                    if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
                        if (_gpuRecovery.programMismatchStreak < UINT32_MAX) {
                            _gpuRecovery.programMismatchStreak++;
                        }
                    } else {
                        _gpuRecovery.programMismatchProgramName = currentProgramName;
                        _gpuRecovery.programMismatchStreak = 1u;
                    }
                    double programDelay = 0.25 * (double)(1u << ((_gpuRecovery.programMismatchStreak > 6u) ? 6u : (_gpuRecovery.programMismatchStreak - 1u)));
                    if (programDelay > 20.0) {
                        programDelay = 20.0;
                    }
                    _gpuRecovery.programMismatchRetryAfter = now + programDelay;
                    if (_gpuRecovery.programMismatchStreak <= 8u || (_gpuRecovery.programMismatchStreak % 64u) == 0u) {
                        NSLog(@"MGL WARNING: Program %u mismatch breaker set for %.2fs (streak=%u)",
                        (unsigned)currentProgramName,
                        programDelay,
                        (unsigned)_gpuRecovery.programMismatchStreak);
                    }

                    // Global quarantine for this program to prevent command-buffer storm.
                    if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                        if (_gpuRecovery.interfaceMismatchBlockedStreak < UINT32_MAX) {
                            _gpuRecovery.interfaceMismatchBlockedStreak++;
                        }
                    } else {
                        _gpuRecovery.interfaceMismatchBlockedProgram = currentProgramName;
                        _gpuRecovery.interfaceMismatchBlockedStreak = 1u;
                    }
                    double quarantineDelay = retryDelay * 8.0;
                    if (quarantineDelay < 1.00) quarantineDelay = 1.00;
                    if (quarantineDelay > 15.00) quarantineDelay = 15.00;
                    _gpuRecovery.interfaceMismatchBlockedUntil = now + quarantineDelay;
                    if (_gpuRecovery.interfaceMismatchBlockedStreak <= 6u || (_gpuRecovery.interfaceMismatchBlockedStreak % 64u) == 0u) {
                        NSLog(@"MGL WARNING: Program %u quarantined for %.2fs after interface mismatch (streak=%u)",
                        (unsigned)currentProgramName,
                        quarantineDelay,
                        (unsigned)_gpuRecovery.interfaceMismatchBlockedStreak);
                    }

                    mglRenderPassInvalidateCurrentPipelineState(
                (__bridge void *)self, "interface mismatch pipeline failure");
                    _gpuRecovery.pipelineRetryAfter = (_gpuRecovery.interfaceMismatchBlockedUntil > _gpuRecovery.interfaceMismatchRetryAfter)
                    ? _gpuRecovery.interfaceMismatchBlockedUntil
                    : _gpuRecovery.interfaceMismatchRetryAfter;
                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                    return false;
                }
            }

            if (!compiledPSO &&
            MGLCapabilityHasBug(&_capability,
            MGL_BUG_MSL_PIPELINE_REJECTION)) {
                mglRenderPassInvalidateCurrentPipelineState(
                (__bridge void *)self, "pipeline creation failure");

                // AGX VIRTUALIZATION FALLBACK: Try with minimal state
                @try {
                    NSLog(@"MGL INFO: VIRTUALIZED AGX - Trying simplified compilation fallback...");

                    // Simplify the state to avoid complex shader compilation issues


                    MGLRenderPipelineDescriptorState simpleState = finalState;
                    simpleState.blending_enabled_mask = 0;
                    simpleState.alpha_to_coverage_enabled = 0;
                    simpleState.alpha_to_one_enabled = 0;
                    simpleState.raster_sample_count = 0;
                    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                        simpleState.source_rgb_blend_factor[i] = 0;
                        simpleState.destination_rgb_blend_factor[i] = 0;
                        simpleState.source_alpha_blend_factor[i] = 0;
                        simpleState.destination_alpha_blend_factor[i] = 0;
                        simpleState.rgb_blend_operation[i] = 0;
                        simpleState.alpha_blend_operation[i] = 0;
                        if (i > 0) {
                            simpleState.color_write_mask[i] = 0;
                            simpleState.color_format[i] = mglRenderInvalidPixelFormat();
                        }
                    }
                    psoPtr = NULL;
                    cppError[0] = '\0';
                    if ([_pipelineCache
                            createRenderPipelineFromState:&simpleState
                            vertexFunction:(__bridge void *)vertexFunction
                            fragmentFunction:fragmentFunction
                                ? (__bridge void *)fragmentFunction : NULL
                            pipelineOut:&psoPtr
                            errorMessage:cppError
                            errorCapacity:sizeof(cppError)] == 0 && psoPtr) {
                        compiledPSO = (__bridge_transfer id)psoPtr;
                    }
                    if (compiledPSO) {
                        mglMetalCountCreate(MGLMetalKindPSO);
                        successfulState = simpleState;
                        haveSuccessfulState = YES;
                        builtColor0Format = simpleState.color_format[0];
                        builtDepthFormat = simpleState.depth_format;
                        builtStencilFormat = simpleState.stencil_format;
                    }
                } @catch (NSException *innerException) {
                    NSLog(@"MGL ERROR: VIRTUALIZED AGX - Simplified compilation also failed: %@", innerException);
                }
            }
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Metal pipeline creation crashed: %@", exception);
        NSLog(@"MGL CRITICAL: Exception name: %@", [exception name]);
        NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);

        BOOL forceSafeFallback =
            mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE");
        if (!MGLCapabilityHasBug(&_capability,
        MGL_BUG_MSL_PIPELINE_REJECTION) && !forceSafeFallback) {
            mglRenderPassInvalidateCurrentPipelineState(
                (__bridge void *)self, "pipeline creation exception");
            _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return false;
        }

        // VIRTUALIZED AGX ULTIMATE FALLBACK: Create minimal safe pipeline
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating ultimate fallback pipeline for virtualization safety");

        @try {
            MGLRenderPipelineDescriptorState safeState = {0};
            safeState.color_count = MAX_COLOR_ATTACHMENTS;
            safeState.rasterization_enabled = 1;
            uint32_t safeColor0Format = (uint32_t)finalState.color_format[0];
            if (_renderPassManager->state && mglRenderPassColorTextureFor(_renderPassManager->state, 0)) {
                safeColor0Format = mglRenderPassTextureInfo(
                    mglRenderPassColorTextureFor(_renderPassManager->state, 0)).pixel_format;
            } else if (_drawable && [self mglDrawableTexture]) {
                safeColor0Format = mglRenderPassTextureInfo([self mglDrawableTexture]).pixel_format;
            }
            safeColor0Format = mglRenderColorFormatOrBGRA(safeColor0Format);
            safeState.color_format[0] = (uint32_t)safeColor0Format;
            safeState.depth_format = finalState.depth_format;
            safeState.stencil_format = finalState.stencil_format;

            /* VS from the precompiled safe_fallback aux asset.  FS reuses the
             * discard stub helper so int/uint color0 gets a matching zero
             * output (aux table only ships float4 mgl_safe_fallback_fs). */
            const MGLAuxShaderAsset *safe =
                mglAuxShaderAssetFind("safe_fallback");
            void *safeVS = NULL;
            void *unusedFS = NULL;
            char libError[512] = {0};
            MGLStubFSValueClass safeClass =
                mglPixelFormatValueClass(safeColor0Format);
            id safeFSFunction =
                mglRasterizerDiscardStubFragmentFunctionForClass(safeClass);
            if (!safe || !safe->data || safe->size == 0 ||
                mglRenderCreateAuxFunctions(
                    safe->data, safe->size, safe->hash,
                    "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
                    &safeVS, &unusedFS,
                    libError, sizeof(libError)) != 0 || !safeVS ||
                !safeFSFunction) {
                NSLog(@"MGL CRITICAL: safe fallback asset unavailable "
                      @"program=%u color0=%lu class=%u hash=0x%016llx error=%s",
                      (unsigned)currentProgramName,
                      (unsigned long)safeColor0Format,
                      (unsigned)safeClass,
                      safe ? (unsigned long long)safe->hash : 0ull,
                      libError[0] ? libError : "asset or stub FS missing");
                if (safeVS) {
                    (void)(__bridge_transfer id)safeVS;
                }
                if (unusedFS) {
                    (void)(__bridge_transfer id)unusedFS;
                }
            } else {
                if (unusedFS) {
                    (void)(__bridge_transfer id)unusedFS;
                }
                id safeVSFunction =
                    (__bridge_transfer id)safeVS;
                psoPtr = NULL;
                cppError[0] = '\0';
                if ([_pipelineCache
                        createRenderPipelineFromState:&safeState
                        vertexFunction:(__bridge void *)safeVSFunction
                        fragmentFunction:(__bridge void *)safeFSFunction
                        pipelineOut:&psoPtr
                        errorMessage:cppError
                        errorCapacity:sizeof(cppError)] == 0 && psoPtr) {
                    compiledPSO = (__bridge_transfer id)psoPtr;
                }
            }
            if (compiledPSO) {
                mglMetalCountCreate(MGLMetalKindPSO);
                successfulState = safeState;
                haveSuccessfulState = YES;
                builtColor0Format = safeState.color_format[0];
                builtDepthFormat = safeState.depth_format;
                builtStencilFormat = safeState.stencil_format;
                NSLog(@"MGL INFO: VIRTUALIZED AGX - Safe fallback pipeline created successfully");
            }
        } @catch (NSException *fallbackException) {
            NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Even fallback pipeline failed: %@", fallbackException);
        }

        if (!compiledPSO) {
            NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - All pipeline creation attempts failed, disabling rendering");
            mglRenderPassInvalidateCurrentPipelineState(
                (__bridge void *)self, "all pipeline fallbacks failed");
            _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return false;
        }
    }

    if (!compiledPSO) {
        NSLog(@"MGL ERROR: Failed to create pipeline state: %s", cppError[0] ? cppError : "unknown error");
        NSLog(@"MGL WARNING: Skipping draw for this pipeline build failure; will retry later");
        mglRenderPassInvalidateCurrentPipelineState(
                (__bridge void *)self, "pipeline state is nil after creation");
        _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
        state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
        return false;
    } else {
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE CREATE success pipeline=%p", compiledPSO);
            NSLog(@"MGL INFO: Pipeline state created successfully");
        }
        /* Publish the compile result to the shared state under a short
         * re-acquired lock. */
        METAL_LOCK();
        if (!pipelineReusedPrevious && haveSuccessfulState) {
            // Clear interface-mismatch breaker after a real compile.
            _gpuRecovery.interfaceMismatchStreak = 0;
            _gpuRecovery.interfaceMismatchProgramName = 0;
            _gpuRecovery.interfaceMismatchColor0Format = mglRenderInvalidPixelFormat();
            _gpuRecovery.interfaceMismatchDepthFormat = mglRenderInvalidPixelFormat();
            _gpuRecovery.interfaceMismatchStencilFormat = mglRenderInvalidPixelFormat();
            _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
            [_pipelineCache activatePipelineState:compiledPSO
                                   color0Format:(uint32_t)builtColor0Format
                                    depthFormat:(uint32_t)builtDepthFormat
                                  stencilFormat:(uint32_t)builtStencilFormat
                                    programName:currentProgramName
                                 vertexFunction:vertexFunction
                               fragmentFunction:fragmentFunction];
            /* Archive lookup/add is owned by PipelineCacheOwner's C++ builder. */
            [self insertPipelineStateIntoCacheWithWords:pipelineCacheKeyWords
                                            pipelineSig:pipelineSig
                                             vertexSig:vertexSig
                                                  state:&successfulState
                                         vertexFunction:vertexFunction
                                       fragmentFunction:fragmentFunction
                                          stateFromCache:stateFromCache];
            if (_gpuRecovery.programMismatchProgramName == currentProgramName) {
                _gpuRecovery.programMismatchProgramName = 0;
                _gpuRecovery.programMismatchRetryAfter = 0.0;
                _gpuRecovery.programMismatchStreak = 0u;
            }
            if (_gpuRecovery.interfaceMismatchBlockedProgram == currentProgramName) {
                _gpuRecovery.interfaceMismatchBlockedProgram = 0;
                _gpuRecovery.interfaceMismatchBlockedUntil = 0.0;
                _gpuRecovery.interfaceMismatchBlockedStreak = 0u;
            }
        }
        METAL_UNLOCK();
    }

    return true;
}

/* Store the compiled pipeline and its value-state descriptor. */
- (void)insertPipelineStateIntoCacheWithWords:(const uint64_t *)pipelineCacheKeyWords
                                  pipelineSig:(uint64_t)pipelineSig
                                   vertexSig:(uint64_t)vertexSig
                                        state:(const MGLRenderPipelineDescriptorState *)state
                               vertexFunction:(id)vertexFunction
                             fragmentFunction:(id)fragmentFunction
                                stateFromCache:(BOOL)stateFromCache
{
    if (pipelineCacheKeyWords && _pipelineCache.state->pipelineState) {
            (void)pipelineSig;
            (void)vertexSig;
                [_pipelineCache storePipeline:(__bridge id)_pipelineCache.state->pipelineState
                           vertexFunction:vertexFunction
                         fragmentFunction:fragmentFunction
                                 forWords:pipelineCacheKeyWords];

            /* Cache the descriptor state for future PSO cache misses.
             * Only cache if state was generated (not from cache). */
            if (!stateFromCache && state) {
                [_pipelineCache storePipelineDescriptorState:state
                                                    forWords:pipelineCacheKeyWords];
            }
    }
}

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

- (bool)syncRenderPassStateForContext:(GLMContext)glm_ctx
{
    GLMState *state = MGL_STATE(glm_ctx);
    Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(glm_ctx, "processGLState.dirtyFBO");
    BOOL framebufferBindingDirty = framebuffer && (framebuffer->dirty_bits & DIRTY_FBO_BINDING);
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager->state->currentRenderEncoderOwner) == 1 &&
        !framebufferBindingDirty &&
        [self currentRenderPassMatchesCurrentFramebuffer]) {
        state->dirty_bits &= ~DIRTY_FBO;
        return true;
    }

    if (framebuffer && framebufferBindingDirty)
    {
        RETURN_FALSE_ON_FAILURE(mglRendererBindFramebufferAttachmentTextures((__bridge void *)self));
        framebuffer = mglRendererGetValidatedFramebuffer(glm_ctx, "processGLState.dirtyFBO.afterBind");
        if (framebuffer) {
            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
        }
    }

    /* instrumentation: an FBO change forced a real encoder rotation
     * (the "already matches" fast path above returned early without counting).
     * newRenderEncoderLocked also bumps g_mglEncoderCreationsSinceSwap, so
     * fboRot <= new always holds; new-minus-fboRot is non-FBO creation. */
    /* RenderPass Manager: encoder open/close is owned by the RenderPass Manager
     * facade (rotateRenderEncoderForCurrentFramebufferLocked), not by this
     * Sync unit directly. The Sync layer only decides that a rotation is
     * needed and delegates the lifecycle transition. */
    RETURN_FALSE_ON_FAILURE([self rotateRenderEncoderForCurrentFramebufferLocked]);
    return true;
}


- (bool)rotateRenderEncoderForCurrentFramebufferLocked
{
    MGL_PERF_INC(g_mglEncoderFBORotationsSinceSwap);
    GLMContext glm_ctx = ctx;
    GLuint fbo_name = 0u;
    if (glm_ctx && glm_ctx->active_state && MGL_STATE(glm_ctx)->framebuffer) {
        fbo_name = MGL_STATE(glm_ctx)->framebuffer->name;
    }
    if (fbo_name == 0u) {
        MGL_PERF_INC(g_mglEncoderFboRotDefaultSinceSwap);
    } else {
        MGL_PERF_INC(g_mglEncoderFboRotNamedSinceSwap);
    }
    mglRendererEndRenderEncodingLocked((__bridge void *)self);
    RETURN_FALSE_ON_FAILURE(
        [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_FBO]);
    return true;
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
    if (![self syncRenderPassStateForContext:glm_ctx]) {
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
