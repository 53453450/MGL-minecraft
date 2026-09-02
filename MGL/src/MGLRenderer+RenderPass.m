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
#import "MGLRenderer+RenderPass_Private.h"
#include "mgl_air_loader.h"     /* AIR metallib loader. */
#include "mgl_aux_assets.h"
#include "mgl_renderer_backend.h"
#include "mgl_renderer_pipeline.h"
#include "mgl_renderer_sync.h"
#include "mgl_pipeline_cache_key.h"
#include "mgl_pipeline_recovery.h"
#include "mgl_env_flag.h"
#include "mgl_shader_abi.h"
#include "mgl_program_reflection.h"

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

static MGLRenderPassIdentityState mglRenderPassIdentitySnapshot(
    const MGLCommandState *commandState)
{
    MGLRenderPassIdentityState identity = {0};
    (void)mglCmdGetRenderPassIdentity(commandState, &identity);
    return identity;
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

static id mglRenderPassDefaultDrawBufferAttachment(
    MGLRendererBackendHandle *backend, GLuint drawBufferIndex,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind)
{
    return (__bridge id)
        mglRendererBackendGetDefaultDrawBufferAttachment(
            backend, drawBufferIndex, kind);
}

static id mglRenderPassFallbackRenderTarget(
    GLMContext context)
{
    return (__bridge id)
        mglRendererBackendGetFallbackRenderTargetTexture(
            mglRenderPassBackend(context));
}

static id mglRenderPassFallbackRenderTargetForSize(
    GLMContext context, NSUInteger width, NSUInteger height,
    NSUInteger layerCount, NSUInteger sampleCount)
{
    width = MAX(width, 1u);
    height = MAX(height, 1u);
    sampleCount = MAX(sampleCount, 1u);
    const BOOL layered = layerCount > 0u;
    const NSUInteger arrayLength = layered ? MAX(layerCount, 1u) : 1u;
    const uint32_t textureType = layered
        ? (uint32_t)MGLTextureType2DArray
        : (uint32_t)MGLTextureType2D;
    id texture = mglRenderPassFallbackRenderTarget(context);
    MGLRenderTextureInfo info = mglRenderPassTextureInfo(texture);
    if (texture && info.width == width && info.height == height &&
        info.array_length == arrayLength &&
        info.texture_type == textureType &&
        info.sample_count == sampleCount) {
        return texture;
    }

    MGLRenderTextureDescriptorState desc = {0};
    desc.texture_type = textureType;
    desc.pixel_format = MGLPixelFormatBGRA8Unorm;
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.mipmap_level_count = 1;
    desc.sample_count = sampleCount;
    desc.array_length = arrayLength;
    desc.usage = MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
    desc.storage_mode = MGLStorageModeShared;
    id replacement = mglRenderPassCreateTexture(&desc);
    if (!replacement ||
        mglRendererBackendSetFallbackRenderTargetTexture(
            mglRenderPassBackend(context),
            (__bridge void *)replacement) != 0) {
        return nil;
    }
    return mglRenderPassFallbackRenderTarget(context);
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
    return commandState && stateOut &&
           mglCmdGetRenderPassPersistentState(commandState, stateOut) == 0;
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

static const MGLRenderPassAttachmentState *
mglRenderPassAttachmentStateFromSnapshot(
    const MGLRenderPassState *state,
    uint32_t attachmentKind,
    NSUInteger colorIndex)
{
    if (!state) return NULL;
    switch (attachmentKind) {
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
            return colorIndex < MAX_COLOR_ATTACHMENTS
                ? &state->color[colorIndex].attachment : NULL;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
            return &state->depth.attachment;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
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

static bool mglRenderPassSnapshotAttachmentMatchesSubresource(
    const MGLRenderPassState *state,
    uint32_t attachmentKind,
    NSUInteger colorIndex,
    MGLMetalAttachmentSubresource subresource)
{
    const MGLRenderPassAttachmentState *attachment =
        mglRenderPassAttachmentStateFromSnapshot(
            state, attachmentKind, colorIndex);
    return attachment &&
           attachment->level == subresource.level &&
           attachment->slice == subresource.slice &&
           attachment->depth_plane == subresource.depthPlane;
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
    switch (attachmentKind) {
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR: {
            if (colorIndex >= MAX_COLOR_ATTACHMENTS) return NO;
            const MGLRenderPassColorState *color =
                &state.color[colorIndex];
            if (clearColorOut) {
                clearColorOut[0] = color->clear_red;
                clearColorOut[1] = color->clear_green;
                clearColorOut[2] = color->clear_blue;
                clearColorOut[3] = color->clear_alpha;
            }
            return YES;
        }
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
            if (clearDepthOut) *clearDepthOut = state.depth.clear_depth;
            return YES;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
            if (clearStencilOut) *clearStencilOut = state.stencil.clear_stencil;
            return YES;
        default:
            return NO;
    }
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

static uint32_t mglRenderPassVisibilityResultTypeFor(
    const MGLCommandState *commandState)
{
    MGLRenderPassState state = {0};
    if (mglRenderPassGetPersistentState(commandState, &state)) {
        return state.visibility_result_type;
    }
    return 0u;
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

static const char *mglGeometryPassthroughType(GLenum type)
{
    switch (type) {
        case GL_FLOAT: return "float";
        case GL_FLOAT_VEC2: return "vec2";
        case GL_FLOAT_VEC3: return "vec3";
        case GL_FLOAT_VEC4: return "vec4";
        case GL_INT: return "int";
        case GL_INT_VEC2: return "ivec2";
        case GL_INT_VEC3: return "ivec3";
        case GL_INT_VEC4: return "ivec4";
        case GL_UNSIGNED_INT: return "uint";
        case GL_UNSIGNED_INT_VEC2: return "uvec2";
        case GL_UNSIGNED_INT_VEC3: return "uvec3";
        case GL_UNSIGNED_INT_VEC4: return "uvec4";
        default: return NULL;
    }
}

static const char *mglGeometryPassthroughSwizzle(GLenum type)
{
    switch (type) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT: return ".x";
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2: return ".xy";
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3: return ".xyz";
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4: return "";
        default: return NULL;
    }
}

static const char *mglGeometryPassthroughFloatType(GLenum type)
{
    switch (type) {
        case GL_INT:
        case GL_UNSIGNED_INT: return "float";
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2: return "vec2";
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3: return "vec3";
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4: return "vec4";
        default: return NULL;
    }
}

/* Read-back conversion for integer varyings: the stage-out record stores
 * raw bits, so integer components need floatBitsToInt/Uint; float varyings
 * need none.  GLSL also requires the `flat` qualifier on integer
 * varyings. */
static const char *mglGeometryPassthroughConversion(GLenum type)
{
    switch (type) {
        case GL_INT:
        case GL_INT_VEC2:
        case GL_INT_VEC3:
        case GL_INT_VEC4: return "floatBitsToInt";
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_INT_VEC2:
        case GL_UNSIGNED_INT_VEC3:
        case GL_UNSIGNED_INT_VEC4: return "floatBitsToUint";
        default: return NULL;
    }
}

static bool mglGeometryPassthroughNeedsFlat(GLenum type)
{
    return mglGeometryPassthroughConversion(type) != NULL;
}

/* The stage-out record stores every varying as a full vec4 slot, so a GS
 * output's reflected gl_type is promoted to the record width.  When the
 * fragment shader consumes the varying with a narrower declared type
 * (legal GL: GS out vec3 + FS in vec3), the passthrough VS must declare
 * the interface with the fragment type -- Metal rejects a pipeline whose
 * vertex output type differs from the fragment input. */
static GLenum mglPassthroughDeclType(
    const MGLShaderResourceList *fsInputs,
    const MGLShaderResource *output)
{
    for (GLuint fi = 0; fsInputs && fsInputs->list && fi < fsInputs->count;
         fi++) {
        const MGLShaderResource *in = &fsInputs->list[fi];
        if (in->name && output->name &&
            strcmp(in->name, output->name) == 0 &&
            in->gl_type != output->gl_type) {
            return in->gl_type;
        }
    }
    return output->gl_type;
}

- (BOOL)ensureAIRGeometryPassthroughFunctionForProgram:(Program *)program
                                       outputPrimitive:(uint32_t)outputPrimitive
{
    if (!program) return NO;
    void *cachedFunction = NULL;
    if (mglRendererBackendGetPassthroughFunction(
            _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
            program->pipeline_cache_instance_id, &cachedFunction) == 1) {
        return YES;
    }
    (void)mglRendererBackendSetPassthroughFunction(
        _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
        NULL, NULL, 0u);

    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES];
    NSUInteger recordStride = mglAIRPerVertexStrideForResources(outputs);
    NSUInteger vec4Stride = recordStride / 16u;
    NSMutableString *source = [NSMutableString stringWithString:
        @"#version 460 core\n"
         "layout(std430, binding = 0) buffer MGLGSOutput {\n"
         "    vec4 records[];\n"
         "} mgl_gs_output;\n"];
    /* The stage-out record stores every varying as a full vec4 slot, so the
     * reflected gl_type of a GS output is promoted to the record width.
     * The passthrough VS must declare the interface with the *fragment
     * shader's* input type instead: Metal rejects a pipeline whose vertex
     * output type differs from the fragment input (e.g. record-promoted
     * vec4 vs a declared vec3). */
    const MGLShaderResourceList *fsInputs =
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES];

    /* gl_PointSize never appears in the reflected output list (all gl_
     * builtins are filtered during reflection), so detect it straight from
     * the GS source -- the same gate the AIR backend uses for its
     * point-size store.  Forwarding matters because the pipeline builder
     * rejects a vertex stage writing point size on a Line/Triangle
     * topology, while Points-topology programs expect the real size. */
    Shader *mgl_gs_for_ps = program->shader_slots[_GEOMETRY_SHADER];
    BOOL hasPointSize = mgl_gs_for_ps && mgl_gs_for_ps->src &&
                        strstr(mgl_gs_for_ps->src, "gl_PointSize") != NULL;
    /* GS-written gl_PrimitiveID is parked at record offset 52 (vec4 slot 3,
     * component y) and ferried to the fragment stage as a flat int varying
     * at the reserved location below. */
    BOOL hasPrimitiveId = mgl_gs_for_ps && mgl_gs_for_ps->src &&
                          strstr(mgl_gs_for_ps->src, "gl_PrimitiveID") != NULL;
    if (hasPrimitiveId) {
        /* Float carrier: the GS kernel stores sitofp(id) and a flat int
         * stage_input that is actually read crashes Apple's AGX compiler
         * (see storeGeometryPrimitiveId). */
        [source appendFormat:
            @"layout(location = %u) flat out float mgl_primitive_id;\n",
             (unsigned)MGL_AIR_PRIMITIVE_ID_LOCATION];
    }    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;

        if (output->stream > 0) continue;
        /* gl_PointSize is a built-in: it cannot carry a layout(location)
         * redeclaration.  The kernel parks it in slot 1.x; main() only
         * forwards it when the GS actually declared it, because the
         * pipeline builder rejects a vertex stage that writes point size
         * on a Line/Triangle topology. */
        if (strcmp(output->name, "gl_PointSize") == 0) continue;
        if (getenv("MGL_DUMP_AIR"))
            fprintf(stderr,
                    "MGL PTVS varying: name=%s gl_type=0x%x loc=%u\n",
                    output->name ? output->name : "?",
                    (unsigned)output->gl_type,
                    (unsigned)output->location);
        GLenum declType = mglPassthroughDeclType(fsInputs, output);
        /* Integer varyings ride as float carriers (the AIR backend pairs
         * this with an fptosi at the fragment entry; raw int attributes do
         * not survive the GS-expansion pipeline plumbing). */
        const char *type =
            mglGeometryPassthroughConversion(declType)
                ? mglGeometryPassthroughFloatType(declType)
                : mglGeometryPassthroughType(declType);
        if (!type || !output->name) {
            NSLog(@"MGL GS ERROR: unsupported passthrough varying type 0x%x",
                  (unsigned)output->gl_type);
            return NO;
        }
        [source appendFormat:@"layout(location = %u) %sout %s %s;\n",
                             (unsigned)output->location,
                             mglGeometryPassthroughNeedsFlat(output->gl_type)
                                 ? "flat " : "",
                             type, output->name];
    }
    [source appendFormat:
        @"void main() {\n"
         "    int mgl_base = gl_VertexID * %lu;\n"
         "    gl_Position = mgl_gs_output.records[mgl_base];\n",
         (unsigned long)vec4Stride];
    if (hasPointSize) {
        /* Forward the kernel's point size (slot 1.x).  Only emitted when
         * the GS declared gl_PointSize -- the pipeline builder rejects a
         * vertex stage writing point size on a Line/Triangle topology.
         * Two-step load: the frontend rejects a member access directly on
         * an SSBO array element. */
        [source appendString:
            @"    vec4 mgl_point_size = mgl_gs_output.records[mgl_base + 1];\n"
             "    gl_PointSize = mgl_point_size.x;\n"];
    }
    if (hasPrimitiveId) {
        /* Two-step load: the frontend rejects a member access directly on
         * an SSBO array element.  The record already holds the float
         * carrier, so forward it unchanged. */
        [source appendString:
            @"    vec4 mgl_prim_vec = mgl_gs_output.records[mgl_base + 3];\n"
             "    mgl_primitive_id = mgl_prim_vec.y;\n"];
    }
    if (getenv("MGL_GS_PROBE")) {
        /* Pixel probe: R = vertex id, G = GPU-read position.y remapped,
         * B = GPU-read varying.r.  Renders the real positions so the
         * geometry stays identifiable. */
        [source appendString:
            @"    vec4 mgl_probe_pos = mgl_gs_output.records[mgl_base];\n"
             "    vec4 mgl_probe_col = mgl_gs_output.records[mgl_base + 4];\n"
             "    gl_Position = mgl_probe_pos;\n"
             "    gs_fs_color = vec4(float(gl_VertexID) / 6.0,\n"
             "                        mgl_probe_pos.y * 0.5 + 0.5,\n"
             "                        abs(mgl_probe_col.r), 1.0);\n"
             "    return;\n"];
    }
    if (getenv("MGL_GS_PROBE_VID")) {
        /* Probe 2: ignore the SSBO entirely; geometry is derived from
         * gl_VertexID alone (six points spread horizontally at mid
         * height).  Correct render => vertex ids / draw are sane and the
         * defect is in the SSBO read path. */
        [source appendString:
            @"    float mgl_vid = float(gl_VertexID);\n"
             "    gl_Position = vec4(mgl_vid / 3.0 - 1.0, 0.25, 0.0, 1.0);\n"
             "    gs_fs_color = vec4(mgl_vid / 6.0, 1.0, 0.0, 1.0);\n"
             "    return;\n"];
    }
    if (getenv("MGL_GS_PROBE_WAVE")) {
        /* Probe 3: oscilloscope.  Vertex x is fixed by vid; the polyline
         * y traces records[mgl_base].x as read on the GPU, and color
         * carries .y/.z/.w.  One render reconstructs every slot the
         * passthrough VS actually sees. */
        [source appendString:
            @"    float mgl_vid = float(gl_VertexID);\n"
             "    vec4 mgl_p0 = mgl_gs_output.records[mgl_base];\n"
             "    gl_Position = vec4(mgl_vid / 3.0 - 1.0, mgl_p0.x, 0.0, 1.0);\n"
             "    gs_fs_color = vec4(mgl_p0.y * 0.5 + 0.5,\n"
             "                       mgl_p0.z * 0.5 + 0.5,\n"
             "                       mgl_p0.w * 0.5 + 0.5, 1.0);\n"
             "    return;\n"];
    }
    Shader *mgl_gs = program->shader_slots[_GEOMETRY_SHADER];
    Shader *mgl_fs = program->shader_slots[_FRAGMENT_SHADER];
    BOOL fsNeedsLayer = NO;
    BOOL fsNeedsViewport = NO;
    for (GLuint fi = 0; fsInputs && fsInputs->list && fi < fsInputs->count;
         fi++) {
        const MGLShaderResource *in = &fsInputs->list[fi];
        if (!in->name) continue;
        if (strcmp(in->name, "gl_Layer") == 0) fsNeedsLayer = YES;
        if (strcmp(in->name, "gl_ViewportIndex") == 0) fsNeedsViewport = YES;
    }
    if (mgl_fs && mgl_fs->src) {
        if (!fsNeedsLayer && strstr(mgl_fs->src, "gl_Layer"))
            fsNeedsLayer = YES;
        if (!fsNeedsViewport && strstr(mgl_fs->src, "gl_ViewportIndex"))
            fsNeedsViewport = YES;
    }
    if (getenv("MGL_PTVS_NO_SPECIALS")) {
        /* Diagnostic: omit the layer/viewport special outputs entirely so
         * the vertex return carries only position + user varyings. */
    } else if (mgl_gs && mgl_gs->src &&
        (strstr(mgl_gs->src, "gl_Layer") ||
         strstr(mgl_gs->src, "gl_ViewportIndex") ||
         fsNeedsLayer || fsNeedsViewport)) {

        [source appendString:
            @"    vec4 mgl_layer_vp = "
             "mgl_gs_output.records[mgl_base + 2];\n"
             "    gl_Layer = floatBitsToInt(mgl_layer_vp.z);\n"
             "    gl_ViewportIndex = floatBitsToInt(mgl_layer_vp.w);\n"];
     }
     for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
         MGLShaderResource *output = &outputs->list[i];
         if (output->is_per_patch) continue;
         if (output->stream > 0) continue;
         GLenum declType = mglPassthroughDeclType(fsInputs, output);
         const char *swizzle = mglGeometryPassthroughSwizzle(declType);
         if (!swizzle || !output->name) return NO;
         const char *convert =
             mglGeometryPassthroughConversion(output->gl_type);
         if (convert) {
             const char *carrierType =
                 mglGeometryPassthroughFloatType(declType);
             [source appendFormat:
                 @"    vec4 mgl_slot_%u = "
                  "mgl_gs_output.records[mgl_base + %u];\n"
                  "    %s = %s(%s(mgl_slot_%u%s));\n",
                 (unsigned)i,
                 (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
                 output->name, carrierType, convert, (unsigned)i, swizzle];
         } else {
             [source appendFormat:
                 @"    vec4 mgl_slot_%u = "
                  "mgl_gs_output.records[mgl_base + %u];\n"
                  "    %s = mgl_slot_%u%s;\n",
                 (unsigned)i,
                 (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
                 output->name, (unsigned)i,
                 swizzle];
         }
    }
    [source appendString:@"}\n"];
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG passthrough VS source:\n%@", source);
    }
    unsigned char *bytes = NULL;
    size_t size = 0u;
    char errorText[512] = {0};
    if (mglShaderCompileGLSL(source.UTF8String, MGL_STAGE_VERTEX, &bytes, &size,
                             errorText, sizeof(errorText)) != 0 ||
        !bytes || size == 0u) {
        NSLog(@"MGL GS ERROR: failed to compile AIR passthrough vertex: %s",
              errorText[0] ? errorText : "?");
        mglShaderFree(bytes);
        return NO;
    }
    if (getenv("MGL_DUMP_AIR")) {
        FILE *f = fopen("/tmp/poison_ptvs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: ptvs.air %zu bytes\n", size);
        }
    }
    id library = nil;
    id function = nil;
    BOOL loaded = mglLoadAIRMainFunction(
        bytes, size, &library, &function,
        errorText, sizeof(errorText));
    mglShaderFree(bytes);
    if (!loaded || !library || !function) {
        NSLog(@"MGL GS ERROR: failed to load AIR passthrough vertex: %s",
              errorText[0] ? errorText : "?");
        return NO;
    }
    return mglRendererBackendSetPassthroughFunction(
        _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
        (__bridge void *)library, (__bridge void *)function,
        program->pipeline_cache_instance_id) == 0;
}

/* TES-compute twin of ensureAIRGeometryPassthroughFunctionForProgram: the
 * isolines/point-mode TES kernel expands one vertex record per work item,
 * so the raster stage is a GLSL passthrough vertex reading the same
 * record layout as the GS expansion (position at 0, point size at 1,
 * varyings at MGL_AIR_PER_VERTEX_STRIDE + location*16).  The records come
 * from the TES stage output resource list. */
- (BOOL)ensureAIRTessEvalPassthroughFunctionForProgram:(Program *)program
{
    if (!program) return NO;
    void *cachedFunction = NULL;
    if (mglRendererBackendGetPassthroughFunction(
            _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
            program->pipeline_cache_instance_id, &cachedFunction) == 1) {
        return YES;
    }
    (void)mglRendererBackendSetPassthroughFunction(
        _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
        NULL, NULL, 0u);

    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_TESS_EVALUATION_SHADER][_STAGE_OUTPUT_RES];
    NSUInteger recordStride = mglAIRPerVertexStrideForResources(outputs);
    NSUInteger vec4Stride = recordStride / 16u;
    NSMutableString *source = [NSMutableString stringWithString:
        @"#version 460 core\n"
         "layout(std430, binding = 0) buffer MGLTESOutput {\n"
         "    vec4 records[];\n"
         "} mgl_tes_output;\n"];
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        const char *type = mglGeometryPassthroughType(output->gl_type);
        if (!type || !output->name) {
            NSLog(@"MGL TESS ERROR: unsupported passthrough varying type 0x%x",
                  (unsigned)output->gl_type);
            return NO;
        }
        [source appendFormat:@"layout(location = %u) out %s %s;\n",
                             (unsigned)output->location, type, output->name];
    }
    [source appendFormat:
        @"void main() {\n"
         "    int mgl_base = gl_VertexID * %lu;\n"
         "    gl_Position = mgl_tes_output.records[mgl_base];\n"
         "    vec4 mgl_point_size = "
         "mgl_tes_output.records[mgl_base + 1];\n"
         "    gl_PointSize = mgl_point_size.x;\n",
         (unsigned long)vec4Stride];
    if (program->tess_cull_distance_count > 0u) {

        if (program->tess_gen_mode == GL_ISOLINES) {
            /* Both endpoints of an isoline segment share the same v, so the
             * cull condition needs the partner record's distances.  The
             * partner record index is (gl_VertexID ^ 1) -- every patch span
             * holds an even item count. */
            [source appendFormat:
                @"    int mgl_partner = (gl_VertexID ^ 1) * %lu;\n"
                 "    vec4 mgl_p0 = mgl_tes_output.records[mgl_partner + 1];\n"
                 "    vec4 mgl_p1 = mgl_tes_output.records[mgl_partner + 2];\n"
                 "    vec4 mgl_p2 = mgl_tes_output.records[mgl_partner + 3];\n",
                 (unsigned long)vec4Stride];
        }
        [source appendFormat:
            @"    vec4 mgl_c0 = mgl_tes_output.records[mgl_base + 1];\n"
             "    vec4 mgl_c1 = mgl_tes_output.records[mgl_base + 2];\n"
             "    vec4 mgl_c2 = mgl_tes_output.records[mgl_base + 3];\n"
             "    bool mgl_culled = false\n"
             "%s"
             "    if (mgl_culled) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);\n",
            program->tess_gen_mode == GL_ISOLINES
                ? "        || (mgl_c0.y < 0.0 && mgl_p0.y < 0.0)\n"
                  "        || (mgl_c0.z < 0.0 && mgl_p0.z < 0.0)\n"
                  "        || (mgl_c0.w < 0.0 && mgl_p0.w < 0.0)\n"
                  "        || (mgl_c1.x < 0.0 && mgl_p1.x < 0.0)\n"
                  "        || (mgl_c1.y < 0.0 && mgl_p1.y < 0.0)\n"
                  "        || (mgl_c1.z < 0.0 && mgl_p1.z < 0.0)\n"
                  "        || (mgl_c1.w < 0.0 && mgl_p1.w < 0.0)\n"
                  "        || (mgl_c2.x < 0.0 && mgl_p2.x < 0.0);\n"
                : "        || mgl_c0.y < 0.0\n"
                  "        || mgl_c0.z < 0.0\n"
                  "        || mgl_c0.w < 0.0\n"
                  "        || mgl_c1.x < 0.0\n"
                  "        || mgl_c1.y < 0.0\n"
                  "        || mgl_c1.z < 0.0\n"
                  "        || mgl_c1.w < 0.0\n"
                  "        || mgl_c2.x < 0.0;\n"];
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        const char *swizzle = mglGeometryPassthroughSwizzle(output->gl_type);
        if (!swizzle || !output->name) return NO;
        [source appendFormat:
            @"    vec4 mgl_slot_%u = "
             "mgl_tes_output.records[mgl_base + %u];\n"
             "    %s = mgl_slot_%u%s;\n",
            (unsigned)i,
            (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
            output->name, (unsigned)i,
            swizzle];
    }
    [source appendString:@"}\n"];
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG passthrough VS source:\n%@", source);
    }
    unsigned char *bytes = NULL;
    size_t size = 0u;
    char errorText[512] = {0};
    if (mglShaderCompileGLSL(source.UTF8String, MGL_STAGE_VERTEX, &bytes, &size,
                             errorText, sizeof(errorText)) != 0 ||
        !bytes || size == 0u) {
        NSLog(@"MGL TESS ERROR: failed to compile AIR TES passthrough vertex: %s\nSOURCE:\n%@",
              errorText[0] ? errorText : "?", source);
        mglShaderFree(bytes);
        return NO;
    }
    id library = nil;
    id function = nil;
    BOOL loaded = mglLoadAIRMainFunction(
        bytes, size, &library, &function,
        errorText, sizeof(errorText));
    mglShaderFree(bytes);
    if (!loaded || !library || !function) {
        NSLog(@"MGL TESS ERROR: failed to load AIR TES passthrough vertex: %s",
              errorText[0] ? errorText : "?");
        return NO;
    }
    return mglRendererBackendSetPassthroughFunction(
        _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
        (__bridge void *)library, (__bridge void *)function,
        program->pipeline_cache_instance_id) == 0;
}

- (void)mtlInvalidateRenderPass:(GLMContext)glm_ctx
{
    if (!glm_ctx || glm_ctx != ctx ||
        mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) != 1) {
        return;
    }


    Framebuffer *curFbo = MGL_STATE(glm_ctx)->framebuffer;
    MGLRenderPassIdentityState identity =
        mglRenderPassIdentitySnapshot(&_commandState);
    if (curFbo == identity.framebuffer &&
        MGL_STATE(glm_ctx)->draw_buffer == identity.draw_buffer) {
        return;
    }

    static uint64_t s_renderPassInvalidateCount = 0;
    uint64_t hit = ++s_renderPassInvalidateCount;
    if (mglTraceLogIsEnabled() && (hit <= 64ull || (hit % 512ull) == 0ull)) {
        Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
        mglTraceLog("RENDERPASS_INVALIDATE hit=%llu fbo=%u(%p) drawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
                    (unsigned long long)hit,
                    (unsigned)(fbo ? fbo->name : 0u),
                    fbo,
                    (unsigned)MGL_STATE(glm_ctx)->draw_buffer,
                    (unsigned)_commandState.renderPassFramebufferName,
                    _commandState.renderPassFramebuffer,
                    (unsigned)_commandState.renderPassDrawBuffer);
        mglLogRenderPassLifecycle("invalidate-before-end",
                                  hit,
                                  glm_ctx,
                                  _commandState.currentCommandBufferOwner,
                                  _commandState.currentRenderEncoderOwner,
                                  _commandState.renderPassStateOwner,
                                  _drawable,
                                  _commandState.renderPassFramebuffer,
                                  _commandState.renderPassFramebufferName,
                                  _commandState.renderPassDrawBuffer,
                                  _commandState.renderPassDrawBufferCount);
    }

    [self flushDrawBuffer:glm_ctx];
    [self endRenderEncoding];
}

- (Texture *)framebufferAttachmentTexture: (FBOAttachment *)fbo_attachment
{
    Texture *tex = NULL;

    if (!fbo_attachment) {
        NSLog(@"MGL ERROR: framebufferAttachmentTexture called with NULL attachment");
        return NULL;
    }

    if (fbo_attachment->textarget == GL_RENDERBUFFER)
    {
        if (fbo_attachment->buf.rbo) {
            tex = fbo_attachment->buf.rbo->tex;
        }
    }
    else
    {
        tex = fbo_attachment->buf.tex;
        if (!tex && fbo_attachment->texture != 0 && fbo_attachment->textarget != GL_RENDERBUFFER)
        {
            tex = findTexture(ctx, fbo_attachment->texture);
            if (tex)
            {
                fbo_attachment->buf.tex = tex;
            }
        }
    }
    if (!tex) {
        NSLog(@"MGL WARN: framebuffer attachment has no texture (target=0x%x)", fbo_attachment->textarget);
    }

    return tex;
}

- (bool)currentRenderPassMatchesCurrentFramebuffer
{
    if (!ctx || !_commandState.renderPassStateOwner) {
        return true;
    }

    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLuint fboName = fbo ? fbo->name : 0u;


    if (fbo != NULL && fboName != 0u) {
        bool cachedResult = false;
        if (mglCmdProbeFboMatchCache(&_commandState, fboName,
                                     fbo->fbo_attachment_generation,
                                     &cachedResult)) {
            return cachedResult;
        }
    }

    bool result = [self mglRenderPassMatchesFramebufferImpl:fbo name:fboName];

    /* store cache for non-default FBOs only. */
    if (fbo != NULL && fboName != 0u) {
        mglCmdSetFboMatchCacheResult(&_commandState, result, fboName,
                                     fbo->fbo_attachment_generation);
    }

    return result;
}

- (bool)mglRenderPassMatchesFramebufferImpl:(Framebuffer *)fbo name:(GLuint)fboName
{
    MGLRenderPassState passState = {0};
    bool hasPassState =
        mglRenderPassGetPersistentState(&_commandState, &passState);
    if (!ctx || !hasPassState) {
        return true;
    }
    MGLRenderPassIdentityState identity =
        mglRenderPassIdentitySnapshot(&_commandState);
    if (identity.framebuffer != fbo ||
        identity.framebuffer_name != fboName ||
        identity.draw_buffer != MGL_STATE(ctx)->draw_buffer ||
        identity.draw_buffer_count != (uint32_t)mglMetalDrawBufferCount(ctx)) {
        return false;
    }
    for (uint32_t i = 0; i < identity.draw_buffer_count; ++i) {
        if (identity.draw_buffers[i] != mglMetalDrawBufferAt(ctx, i)) {
            return false;
        }
    }

    if (!fbo) {
        GLuint mgl_drawbuffer = mglDefaultDrawBufferIndexForGL(MGL_STATE(ctx)->draw_buffer);
        id expectedColor0 = nil;
        id actualColor0 = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);

        if (mgl_drawbuffer == _FRONT) {
            expectedColor0 = _drawable ? [self mglDrawableTexture] : nil;
        } else if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            expectedColor0 = mglRenderPassDefaultDrawBufferAttachment(
                _backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR);
        }

        if (actualColor0 != expectedColor0) {
            return false;
        }

        id expectedDepth = nil;
        id expectedStencil = nil;
        if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            id cachedDepth =
                mglRenderPassDefaultDrawBufferAttachment(
                    _backend, mgl_drawbuffer,
                    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH);
            id cachedStencil =
                mglRenderPassDefaultDrawBufferAttachment(
                    _backend, mgl_drawbuffer,
                    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL);
            BOOL defaultPassNeedsDepth = MGL_STATE(ctx)->caps.depth_test ||
                                         cachedDepth != nil;
            BOOL defaultPassNeedsStencil = MGL_STATE(ctx)->caps.stencil_test ||
                                           ctx->stencil_format.format ||
                                           cachedStencil != nil;
            expectedDepth = defaultPassNeedsDepth ? cachedDepth : nil;
            expectedStencil = defaultPassNeedsStencil ? cachedStencil : nil;
            if (MGL_STATE(ctx)->caps.depth_test && !expectedDepth) {
                return false;
            }
            if ((MGL_STATE(ctx)->caps.stencil_test || ctx->stencil_format.format) && !expectedStencil) {
                return false;
            }
        }

        id actualDepth = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
        id actualStencil = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
        if (actualDepth != expectedDepth) {
            return false;
        }
        if (actualStencil != expectedStencil) {
            return false;
        }

        return true;
    }

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        BOOL drawSlotPresent =
            mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) != 0u;
        FBOAttachment *attachment = drawSlotPresent ? &fbo->color_attachments[attachmentIndex] : NULL;
        Texture *tex = drawSlotPresent ? [self framebufferAttachmentTexture:attachment] : NULL;
        id expected = nil;

        if (tex) {
            tex->is_render_target = true;
            if (!tex->mtl_data) {
                if (![self bindMTLTexture:tex]) {
                    return false;
                }
            }
            expected = (__bridge id)(tex->mtl_data);
        }

        id actual = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
            colorSlot);
        if (actual != expected) {
            return false;
        }

        if (attachment && actual) {
            MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(attachment);
            bool matches = mglRenderPassSnapshotAttachmentMatchesSubresource(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                colorSlot, subresource);
            if (!matches) {
                return false;
            }
        }

        id nextColor = nil;
        if (i + 1u < MAX_COLOR_ATTACHMENTS) {
            nextColor = mglRenderPassTextureFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                i + 1u);
        }
        if (i + 1u >= MAX_COLOR_ATTACHMENTS ||
            (mglMetalDrawBufferAt(ctx, i + 1u) == GL_NONE &&
             !nextColor)) {
            break;
        }
    }

    id expectedDepth = nil;
    if (fbo->depth.texture) {
        Texture *depthTex = [self framebufferAttachmentTexture:&fbo->depth];
        if (depthTex && !depthTex->mtl_data) {
            depthTex->is_render_target = true;
            if (![self bindMTLTexture:depthTex]) {
                return false;
            }
        }
        expectedDepth = depthTex ? (__bridge id)(depthTex->mtl_data) : nil;
    }
    id actualDepth = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    if (actualDepth != expectedDepth) {
        return false;
    }
    if (fbo->depth.texture && expectedDepth) {
        MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
        bool matches = mglRenderPassSnapshotAttachmentMatchesSubresource(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH,
            0, subresource);
        if (!matches) {
            return false;
        }
    }

    id expectedStencil = nil;
    if (fbo->stencil.texture) {
        Texture *stencilTex = [self framebufferAttachmentTexture:&fbo->stencil];
        if (stencilTex && !stencilTex->mtl_data) {
            stencilTex->is_render_target = true;
            if (![self bindMTLTexture:stencilTex]) {
                return false;
            }
        }
        expectedStencil = stencilTex ? (__bridge id)(stencilTex->mtl_data) : nil;
    }
    id actualStencil = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
    if (actualStencil != expectedStencil) {
        return false;
    }
    if (fbo->stencil.texture && expectedStencil) {
        MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
        bool matches = mglRenderPassSnapshotAttachmentMatchesSubresource(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL,
            0, subresource);
        if (!matches) {
            return false;
        }
    }

    return true;
}

- (bool)ensureCurrentRenderPassMatchesFramebufferForDraw
{
    if (!ctx) {
        return true;
    }

    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) != 1) {
        return true;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return true;
    }

    static uint64_t s_fboPassMismatchCount = 0;
    uint64_t hit = ++s_fboPassMismatchCount;
    if (hit <= 32ull || (hit % 256ull) == 0ull) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
        id color0 = mglRenderPassColorTextureFor(&_commandState, 0);
        GLuint mglDefaultDrawbuffer = fbo ? 0u : mglDefaultDrawBufferIndexForGL(MGL_STATE(ctx)->draw_buffer);
        id expectedDefaultColor0 = nil;
        if (!fbo) {
            expectedDefaultColor0 = (mglDefaultDrawbuffer == _FRONT)
                ? (_drawable ? [self mglDrawableTexture] : nil)
                : ((mglDefaultDrawbuffer < _MAX_DRAW_BUFFERS)
                    ? mglRenderPassDefaultDrawBufferAttachment(
                          _backend, mglDefaultDrawbuffer,
                          MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR)
                    : nil);
        }
        GLuint fboName = fbo ? fbo->name : 0u;
        GLuint attachment0Name = (fbo && (fbo->color_attachment_bitfield & 1u)) ? fbo->color_attachments[0].texture : 0u;
        NSLog(@"MGL WARNING: render pass/FBO mismatch before draw hit=%llu fbo=%u drawBuffer=0x%x attachment0=%u passColor0=%p expectedDefaultColor0=%p defaultDrawBuffer=%u; rebuilding encoder",
              (unsigned long long)hit,
              (unsigned)fboName,
              (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
              (unsigned)attachment0Name,
              color0,
              expectedDefaultColor0,
              (unsigned)mglDefaultDrawbuffer);
        mglLogRenderPassLifecycle(fbo ? "fbo-mismatch-before-rebuild" : "default-fbo-mismatch-before-rebuild",
                                  hit,
                                  ctx,
                                  _commandState.currentCommandBufferOwner,
                                  _commandState.currentRenderEncoderOwner,
                                  _commandState.renderPassStateOwner,
                                  _drawable,
                                  _commandState.renderPassFramebuffer,
                                  _commandState.renderPassFramebufferName,
                                  _commandState.renderPassDrawBuffer,
                                  _commandState.renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM |
                             DIRTY_RENDER_STATE | DIRTY_VAO);
    return [self newRenderEncoderWithReason:MGL_ENC_REASON_FBO];
}

- (void)endRenderPassIfFramebufferChangedForNonDraw:(uint64_t)processCall
{
    if (!ctx || mglRenderEncoderOwnerHasCurrent(
                    _commandState.currentRenderEncoderOwner) != 1) {
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
                    (unsigned)_commandState.renderPassFramebufferName,
                    _commandState.renderPassFramebuffer,
                    (unsigned)_commandState.renderPassDrawBuffer);
        mglLogRenderPassLifecycle("non-draw-mismatch-before-end",
                                  hit,
                                  ctx,
                                  _commandState.currentCommandBufferOwner,
                                  _commandState.currentRenderEncoderOwner,
                                  _commandState.renderPassStateOwner,
                                  _drawable,
                                  _commandState.renderPassFramebuffer,
                                  _commandState.renderPassFramebufferName,
                                  _commandState.renderPassDrawBuffer,
                                  _commandState.renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE);
}

- (bool)restoreRenderEncoderAfterTextureUploadForDraw:(const char *)reason
{
    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) == 1) {
        return true;
    }
    MGLRenderPassState passState = {0};
    bool hasPassState =
        mglRenderPassGetPersistentState(&_commandState, &passState);
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
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i,
                MGLLoadActionLoad, MGLStoreActionStore);
        }
    }
    id depthTexture = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    if (depthTexture) {
        mglRenderPassSetPersistentActions(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionLoad, MGLStoreActionStore);
    }
    id stencilTexture = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
    if (stencilTexture) {
        mglRenderPassSetPersistentActions(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionLoad, MGLStoreActionStore);
    }

    @try {
        id renderEncoder =
            (__bridge id)mglCmdCreateRenderEncoder(&_commandState);
        mglCmdInstallRenderEncoder(&_commandState, (__bridge void *)renderEncoder);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload failed to create encoder: %@",
              exception.reason);
        mglCmdClearCurrentRenderEncoder(&_commandState);
        [self recordGPUError];
        return false;
    }
    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) != 1) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload returned nil encoder reason=%s",
              reason ? reason : "(null)");
        [self recordGPUError];
        return false;
    }
    mglRenderSetRenderEncoderOwnerLabel(
        _commandState.currentRenderEncoderOwner,
        "GL Render Encoder");
    /* When trace is disabled, skip the full-struct memset and trace call
     * and clear only the functional flag fields. */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             reason ? reason : "restore_render_encoder_after_texture_upload",
                                             _resourceFallback.fragmentTextureTraceBindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             _pipelineCacheState.pipelineProgramName);
        memset(_resourceFallback.fragmentTextureTraceBindings, 0,
               sizeof(_resourceFallback.fragmentTextureTraceBindings));
    } else {
        mglClearFragmentTextureTraceFunctionalFlags(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
    }
    mglCmdUpdateRenderPassIdentityForContext(&_commandState, ctx);
    [self updateCurrentRenderEncoder];

    if (!_pipelineCacheState.pipelineState) {
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO |
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    @try {
        if (mglRenderBindingSetPipelineIfNeededForOwner(
                _bindingStateOwner,
                _commandState.currentRenderEncoderOwner,
                _pipelineCacheState.pipelineState) > 0) {
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

    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    MGLEncodeContext encCtx = {
        .render_encoder_owner = _commandState.currentRenderEncoderOwner,
    };
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]);
    return true;
}

- (bool)bindFramebufferTexture:(FBOAttachment *)fbo_attachment isDrawBuffer:(bool) isDrawBuffer
{
    Texture *tex;

    tex = [self framebufferAttachmentTexture: fbo_attachment];
    if (!tex) {
        // Incomplete/missing attachment. Do not crash.
        return true;
    }

    if (isDrawBuffer) {
        tex->is_render_target = true;
    }

    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);

    return true;
}

- (void)invalidateCurrentPipelineStateForReason:(NSString *)reason
{
    if (_pipelineCacheState.pipelineState) {
        static uint64_t s_pipelineInvalidateCount = 0;
        uint64_t hit = ++s_pipelineInvalidateCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL WARNING: Invalidating current pipeline state after %@ hit=%llu",
                  reason ?: @"pipeline failure",
                  (unsigned long long)hit);
        }
    }
    mglPipelineCacheInvalidatePipelineState(
        &_pipelineCacheState, &_pipelineCacheOwner, (__bridge void *)_device,
        _pipelineCacheBinaryArchiveRequested);
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
            if (i == _GEOMETRY_SHADER) {
                if (ptr->gs_route == MGL_GS_ROUTE_COMPUTE &&
                    ptr->modules[i].metallib_bytes &&
                    ptr->modules[i].metallib_size > 0u) {
                    /* The AIR geometry stage is a compute kernel.  Load it
                     * below like TCS/CS; the draw helper owns dispatch and
                     * expanded-output rendering. */
                } else {
                static uint64_t s_geometryShaderMetalSkipCount = 0;
                uint64_t hit = ++s_geometryShaderMetalSkipCount;
                if (hit <= 16ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL WARNING: Blocking draw for unsupported geometry shader program=%u hit=%llu",
                          (unsigned)ptr->name,
                          (unsigned long long)hit);
                }
                return false;
                }
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
                if (i == _VERTEX_SHADER &&
                    ptr->modules[i].metallib_tess_capture_bytes &&
                    (!ptr->modules[i].mtl_tess_capture_library ||
                     !ptr->modules[i].mtl_tess_capture_function)) {
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
                if (i == _VERTEX_SHADER &&
                    ptr->modules[i].metallib_cull_capture_bytes &&
                    (!ptr->modules[i].mtl_cull_capture_library ||
                     !ptr->modules[i].mtl_cull_capture_function)) {
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
        _commandState.renderPassStateOwner != NULL;
    BOOL passHasDepthAttachment =
        (hasConfiguredRenderPass &&
         mglRenderPassDepthTextureFor(&_commandState) != nil);
    BOOL passHasStencilAttachment =
        (hasConfiguredRenderPass &&
         mglRenderPassStencilTextureFor(&_commandState) != nil);
    BOOL useDepthState = state->caps.depth_test && passHasDepthAttachment;
    BOOL useStencilState = state->caps.stencil_test && passHasStencilAttachment;

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
            if (!mglIsValidGLCompareFunction(state->var.depth_func)) {
                mglLogRenderStateRepair("depth_func", state->var.depth_func, GL_LESS);
                state->var.depth_func = GL_LESS;
                mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
            }

            dsDesc.depth_compare_function = (uint32_t)
                mglMTLCompareFunctionForGL(state->var.depth_func,
                                           MGLCompareFunctionLess,
                                           "depth");
            dsDesc.depth_write_enabled = state->var.depth_writemask ? 1u : 0u;
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
                            mglRenderPassStencilTextureFor(&_commandState),
                            (int)(MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->stencil.layered : 0));
            }
            {
                if (!mglIsValidGLCompareFunction(state->var.stencil_func)) {
                    mglLogRenderStateRepair("stencil_func", state->var.stencil_func, GL_ALWAYS);
                    state->var.stencil_func = GL_ALWAYS;
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
                dsDesc.front.stencil_failure_operation =
                    [self mtlStencilOpForGLOp:state->var.stencil_fail];
                dsDesc.front.depth_failure_operation =
                    [self mtlStencilOpForGLOp:state->var.stencil_pass_depth_fail];
                dsDesc.front.depth_stencil_pass_operation =
                    [self mtlStencilOpForGLOp:state->var.stencil_pass_depth_pass];
                dsDesc.front.write_mask = state->var.stencil_writemask;
                dsDesc.front.read_mask = state->var.stencil_value_mask;
            }

            {
                if (!mglIsValidGLCompareFunction(state->var.stencil_back_func)) {
                    mglLogRenderStateRepair("stencil_back_func", state->var.stencil_back_func, GL_ALWAYS);
                    state->var.stencil_back_func = GL_ALWAYS;
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
                dsDesc.back.stencil_failure_operation =
                    [self mtlStencilOpForGLOp:state->var.stencil_back_fail];
                dsDesc.back.depth_failure_operation =
                    [self mtlStencilOpForGLOp:state->var.stencil_back_pass_depth_fail];
                dsDesc.back.depth_stencil_pass_operation =
                    [self mtlStencilOpForGLOp:state->var.stencil_back_pass_depth_pass];
                dsDesc.back.write_mask = state->var.stencil_back_writemask;
                dsDesc.back.read_mask = state->var.stencil_back_value_mask;
            }
        }

        id dsState = (__bridge id)mglPipelineCacheDepthStencilStateForValueState(
            &_pipelineCacheState, &_pipelineCacheOwner, (__bridge void *)_device,
            _pipelineCacheBinaryArchiveRequested, &dsDesc);

        if (mglRenderBindingSetDepthStencilIfNeededForOwner(
                _bindingStateOwner,
                _commandState.currentRenderEncoderOwner,
                (__bridge void *)dsState) > 0) {
        } else {
            MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
        }
        if (useStencilState) {
            mglRenderSetStencilReferenceValuesForOwner(
                _commandState.currentRenderEncoderOwner,
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
            (__bridge id)mglPipelineCacheDepthStencilStateForValueState(
                &_pipelineCacheState, &_pipelineCacheOwner,
                (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                &disabledDSDesc);
        if (disabledDSState) {
            if (mglRenderBindingSetDepthStencilIfNeededForOwner(
                    _bindingStateOwner,
                    _commandState.currentRenderEncoderOwner,
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
            _commandState.currentRenderEncoderOwner,
            bcRed, bcGreen, bcBlue, bcAlpha);
    }

    /* GL_SAMPLE_MASK: Metal does not expose a per-draw sample mask setter on
     * MTLRenderCommandEncoder.  Sample coverage in Metal is controlled via
     * alpha-to-coverage and shader-side [[sample_mask]], neither of which
     * maps cleanly to GL_SAMPLE_MASK.  This remains a known limitation. */

    [self updateViewportAndScissorLocked];

    if (state->var.front_face != GL_CW && state->var.front_face != GL_CCW) {
        mglLogRenderStateRepair("front_face", state->var.front_face, GL_CCW);
        state->var.front_face = GL_CCW;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }

    BOOL defaultFramebufferSampledPass =
        state->framebuffer == NULL &&
        !state->caps.depth_test &&
        mglRendererGetProgramBindingCount(ctx, _FRAGMENT_SHADER, _SAMPLED_IMAGE_RES) > 0;
    BOOL rtSampledCopyDraw = _commandState.currentDrawUsesRTSampledCopy;

    if (state->caps.cull_face && !defaultFramebufferSampledPass && !rtSampledCopyDraw)
    {
        uint32_t cull_mode;

        switch(state->var.cull_face_mode)
        {
            case GL_BACK: cull_mode = MGLCullModeBack; break;
            case GL_FRONT: cull_mode = MGLCullModeFront; break;
            default:
                cull_mode = MGLCullModeNone;
        }

        mglRenderBindingSetCullIfNeededForOwner(
            _bindingStateOwner,
            _commandState.currentRenderEncoderOwner,
            (uint32_t)cull_mode);
        uint32_t _winding =
            mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                     state->var.clip_origin == GL_UPPER_LEFT);
        mglRenderBindingSetWindingIfNeededForOwner(
            _bindingStateOwner,
            _commandState.currentRenderEncoderOwner,
            (uint32_t)_winding);
    }
    else
    {
        mglRenderBindingSetCullIfNeededForOwner(
            _bindingStateOwner,
            _commandState.currentRenderEncoderOwner,
            (uint32_t)MGLCullModeNone);
        uint32_t _winding =
            mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                     state->var.clip_origin == GL_UPPER_LEFT);
        mglRenderBindingSetWindingIfNeededForOwner(
            _bindingStateOwner,
            _commandState.currentRenderEncoderOwner,
            (uint32_t)_winding);

        if (state->caps.cull_face && defaultFramebufferSampledPass) {
            static uint64_t s_defaultSampledCullBypassCount = 0;
            uint64_t hit = ++s_defaultSampledCullBypassCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                mglTraceLogNSString(@"MGL TRACE default sampled pass cull bypass hit=%llu program=%u drawBuf=0x%x",
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
                            (unsigned)_pipelineCacheState.pipelineProgramName,
                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                            (unsigned)_commandState.renderPassFramebufferName,
                            (ctx && state->caps.depth_test) ? 1 : 0,
                            (ctx && state->var.depth_writemask) ? 1 : 0,
                            (unsigned)(ctx ? state->var.depth_func : 0u),
                            (ctx && state->caps.blend) ? 1 : 0,
                            (unsigned)(ctx ? state->var.cull_face_mode : 0u),
                            (unsigned)(ctx ? state->var.front_face : 0u));
            }
        }
    }

    if (state->caps.depth_clamp)
    {
        mglRenderSetDepthClipModeForOwner(
            _commandState.currentRenderEncoderOwner,
            (uint32_t)MGLDepthClipModeClamp);
    }

    if (state->caps.polygon_offset_fill ||
        state->caps.polygon_offset_line ||
        state->caps.polygon_offset_point)
    {
        float _bias = state->var.polygon_offset_units;
        float _slope = state->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _commandState.currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    }
    else
    {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _commandState.currentRenderEncoderOwner,
            0.0f, 0.0f, 0.0f);
    }

    uint32_t triangleFillMode = 0u;
    if (state->var.polygon_mode == GL_LINE)
    {
        triangleFillMode = 1u;
    }
    else if (state->var.polygon_mode != GL_FILL &&
             state->var.polygon_mode != GL_POINT)
    {
        mglLogRenderStateRepair("polygon_mode", state->var.polygon_mode, GL_FILL);
        state->var.polygon_mode = GL_FILL;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];
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
            _commandState.renderPassStateOwner != NULL;
        if (hasConfiguredRenderPass) {
            passWidth = mglRenderPassRenderTargetWidthFor(&_commandState);
            passHeight = mglRenderPassRenderTargetHeightFor(&_commandState);

            if (passWidth == 0 || passHeight == 0) {
                for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                    id candidate = mglRenderPassColorTextureFor(&_commandState, i);
                    if (candidate) {
                        passTexture = candidate;
                        break;
                    }
                }

                if (!passTexture) {
                    passTexture = mglRenderPassDepthTextureFor(&_commandState);
                }
                if (!passTexture) {
                    passTexture = mglRenderPassStencilTextureFor(&_commandState);
                }

                if (passTexture) {
                    passWidth = mglRenderPassTextureInfo(passTexture).width;
                    passHeight = mglRenderPassTextureInfo(passTexture).height;
                    mglRenderPassSetPersistentDimensions(
                        &_commandState, passWidth, passHeight);
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

                // GL allows negative x/y; clamp origin and shrink extent accordingly.
                if (sx < 0) {
                    sw += sx;
                    sx = 0;
                }
                if (sy < 0) {
                    sh += sy;
                    sy = 0;
                }

                if (sx >= (GLint)passWidth || sy >= (GLint)passHeight) {

                    sx = 0;
                    sy = 0;
                    sw = 0;
                    sh = 0;
                } else {
                    GLint maxWidth = (GLint)passWidth - sx;
                    GLint maxHeight = (GLint)passHeight - sy;

                    if (sw > maxWidth) {
                        sw = maxWidth;
                    }
                    if (sh > maxHeight) {
                        sh = maxHeight;
                    }

                    if (sw <= 0 || sh <= 0) {
                        /* GL allows 0-size scissor rects; they produce no
                         * fragments.  Do not reset to the full pass. */
                        sx = 0;
                        sy = 0;
                        sw = 0;
                        sh = 0;
                    }
                }
            }

            GLint metalSy = sy;
            if (state->var.clip_origin != GL_UPPER_LEFT) {
                metalSy = (GLint)passHeight - (sy + sh);
                if (metalSy < 0) {
                    metalSy = 0;
                }
            }

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
            [self setScissorRectIfNeeded:rect];

            GLdouble rawVx = (GLdouble)state->viewport[0];
            GLdouble rawVy = (GLdouble)state->viewport[1];
            GLdouble rawVw = (GLdouble)state->viewport[2];
            GLdouble rawVh = (GLdouble)state->viewport[3];

            GLdouble vx = rawVx;
            GLdouble vy = rawVy;
            GLdouble vw = rawVw;
            GLdouble vh = rawVh;

            if (vw <= 0.0 || vh <= 0.0) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            }

            if (vx < 0.0) {
                vw += vx;
                vx = 0.0;
            }
            if (vy < 0.0) {
                vh += vy;
                vy = 0.0;
            }

            if (vx >= (GLdouble)passWidth || vy >= (GLdouble)passHeight) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            } else {
                GLdouble maxVw = (GLdouble)passWidth - vx;
                GLdouble maxVh = (GLdouble)passHeight - vy;
                if (vw > maxVw) {
                    vw = maxVw;
                }
                if (vh > maxVh) {
                    vh = maxVh;
                }
                if (vw <= 0.0 || vh <= 0.0) {
                    vx = 0.0;
                    vy = 0.0;
                    vw = (GLdouble)passWidth;
                    vh = (GLdouble)passHeight;
                }
            }

            /*
             * glViewport's x/y select the same framebuffer rectangle regardless
             * of glClipControl origin.  The origin only changes how clip-space Y
             * maps within that rectangle; Metal still addresses the texture from
             * the top, so always convert GL's lower-left viewport rectangle to a
             * Metal top-left origin here.
             */
            GLdouble metalVy = (GLdouble)passHeight - (vy + vh);
            if (metalVy < 0.0) {
                metalVy = 0.0;
            }

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
                    id c0 = mglRenderPassColorTextureFor(&_commandState, 0);
                    id d0 = mglRenderPassDepthTextureFor(&_commandState);
                    mglTraceLog("RT_SAMPLE_COPY_ENCODER hit=%llu fbo=%u rpFbo=%u program=%u rtTex=%u label=\"%s\" depthTex=%u depthLabel=\"%s\" "
                          "pass=%lux%lu c0=%p fmt=%lu depth=%p fmt=%lu "
                          "loadStore(c=%s/%s d=%s/%s) clipOrigin=0x%x "
                          "scissor(en=%d raw=%d,%d,%d,%d metal=%d,%d,%d,%d) "
                          "viewport(raw=%.1f,%.1f,%.1f,%.1f metal=%.1f,%.1f,%.1f,%.1f) "
                          "depth(test=%d write=%d func=0x%x) blend=%d cull=%d levels=%u mips=%u mipmapped=%u",
                          (unsigned long long)hit,
                          state->framebuffer ? (unsigned)state->framebuffer->name : 0u,
                          (unsigned)_commandState.renderPassFramebufferName,
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
                          mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
                          mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
                          mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
                          mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
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
                mglTraceLogNSString(@"MGL VIEWPORT apply pass=%lux%lu origin=0x%x raw=(%.3f,%.3f,%.3f,%.3f) resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
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
                    id rpColor0 = mglRenderPassColorTextureFor(&_commandState, 0);
                    id rpDepth = mglRenderPassDepthTextureFor(&_commandState);
                    id drawableTexture = (_drawable ? [self mglDrawableTexture] : nil);

                    mglTraceLogNSString(@"MGL VIEWPORT CLAMP DETAIL hit=%llu fbo=%p valid=%d fboName=%u drawBuffer=0x%x pass=%lux%lu "
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
                            if (attachment->textarget == GL_RENDERBUFFER) {
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
                            id rpAttachment = mglRenderPassColorTextureFor(&_commandState, attIndex);

                            mglTraceLogNSString(@"MGL VIEWPORT CLAMP FBO att=%d name=%u textarget=0x%x level=%d layer=%d tex=%p "
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
                    _commandState.currentRenderEncoderOwner,
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
                    _commandState.currentRenderEncoderOwner,
                    viewports, (uint64_t)MGL_MAX_VIEWPORTS);
            }
        } else {
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: updateCurrentRenderEncoder could not resolve pass size; using raw GL viewport");
            }
            [self setViewportIfNeeded:(MGLViewportValue){state->viewport[0], state->viewport[1],
                                       state->viewport[2], state->viewport[3],
                                       state->var.depth_range[0], state->var.depth_range[1]}];
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


- (BOOL)shouldUseDontCareLoadForColorTexture:(Texture *)tex
                             firstUseThisFrame:(BOOL)firstUseThisFrame
{

    if (!mglEnvFlagEnabled("MGL_ENABLE_DONTCARE_LOAD")) {
        return NO;
    }
    if (!tex || !tex->mtl_data) {
        return NO;
    }
    if (ctx && MGL_STATE(ctx)->caps.blend) {
        return NO;
    }
    if (!firstUseThisFrame) {
        return NO;
    }
    return YES;
}

- (bool) configureUserFBOAttachmentsLocked
{
    Framebuffer *fbo;

    fbo = MGL_STATE(ctx)->framebuffer;

    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (int i = 0; i < drawBufferCount; i++)
    {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            (fbo->color_attachment_bitfield & (1u << attachmentIndex)) &&
            fbo->color_attachments[attachmentIndex].texture)
        {
            Texture *tex;

            tex = [self framebufferAttachmentTexture: &fbo->color_attachments[attachmentIndex]];
            if (!tex) {
                continue;
            }

            // Ensure attachment textures are created with RenderTarget usage.
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
            if (!tex->mtl_data) {
                continue;
            }

            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->color_attachments[attachmentIndex]);
            mglRenderPassSetPersistentAttachment(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                mglApplySRGBStateToRenderTarget(
                    (__bridge id _Nullable)(tex->mtl_data), ctx),
                subresource.level, subresource.slice,
                subresource.depthPlane,
                fbo->color_attachments[attachmentIndex].layered);

            if (tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ||
                tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                tex->target == GL_TEXTURE_2D_ARRAY) {
                id rpTex = mglRenderPassColorTextureFor(&_commandState, colorSlot);
                (void)rpTex;
            }

            // Keep render pass dimensions aligned with attached color targets.
            // Some FBO paths use textures (not renderbuffers), and Metal still requires
            // scissor/viewport to be bounded by the attachment dimensions.
            NSUInteger attWidth = mglMetalTextureLevelDimension((NSUInteger)tex->width,
                                                                subresource.level);
            NSUInteger attHeight = mglMetalTextureLevelDimension((NSUInteger)tex->height,
                                                                 subresource.level);
            if (attWidth > 0 && attHeight > 0) {
                if (mglRenderPassRenderTargetWidthFor(&_commandState) == 0 || mglRenderPassRenderTargetHeightFor(&_commandState) == 0) {
                    mglRenderPassSetPersistentDimensions(
                        &_commandState, attWidth, attHeight);
                } else if (mglRenderPassRenderTargetWidthFor(&_commandState) != attWidth ||
                           mglRenderPassRenderTargetHeightFor(&_commandState) != attHeight) {
                    NSUInteger oldWidth = mglRenderPassRenderTargetWidthFor(&_commandState);
                    NSUInteger oldHeight = mglRenderPassRenderTargetHeightFor(&_commandState);
                    mglRenderPassSetPersistentDimensions(
                        &_commandState,
                        MIN(mglRenderPassRenderTargetWidthFor(&_commandState),
                            attWidth),
                        MIN(mglRenderPassRenderTargetHeightFor(&_commandState),
                            attHeight));
                    NSLog(@"MGL WARNING: FBO color attachment size mismatch slot=%d old=%lux%lu new=%lux%lu resolved=%lux%lu",
                          i,
                          (unsigned long)oldWidth,
                          (unsigned long)oldHeight,
                          (unsigned long)attWidth,
                          (unsigned long)attHeight,
                          (unsigned long)mglRenderPassRenderTargetWidthFor(&_commandState),
                          (unsigned long)mglRenderPassRenderTargetHeightFor(&_commandState));
                }
            }
        }
    }

    // depth attachment
    if (fbo->depth.texture)
    {
        Texture *tex;

        tex = [self framebufferAttachmentTexture: &fbo->depth];
        if (tex) {
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
        }
        if (tex && tex->mtl_data) {
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
            mglRenderPassSetPersistentAttachment(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                (__bridge id _Nullable)(tex->mtl_data),
                subresource.level, subresource.slice,
                subresource.depthPlane, fbo->depth.layered);
        }
    }

    // stencil attachment
    if (fbo->stencil.texture)
    {
        Texture *tex;

        tex = [self framebufferAttachmentTexture: &fbo->stencil];
        if (tex) {
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
        }
        if (tex && tex->mtl_data) {
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
            mglRenderPassSetPersistentAttachment(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                (__bridge id _Nullable)(tex->mtl_data),
                subresource.level, subresource.slice,
                subresource.depthPlane, fbo->stencil.layered);
        }
    }
    return true;
}

- (bool) configureDefaultFramebufferAttachmentsLocked
{
    GLuint mgl_drawbuffer;
    id texture = nil;
    id depth_texture = nil;
    id stencil_texture = nil;

    switch(MGL_STATE(ctx)->draw_buffer)
    {
        case GL_FRONT: mgl_drawbuffer = _FRONT; break;
        case GL_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_FRONT_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_FRONT_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_BACK_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_BACK_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_FRONT_AND_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_COLOR_ATTACHMENT0: mgl_drawbuffer = _FRONT; break;
        case GL_NONE:
            // Handle GL_NONE gracefully - no draw buffer selected
            mgl_drawbuffer = _FRONT; // fallback to front
            DEBUG_PRINT("MGL: draw_buffer is GL_NONE, falling back to FRONT\n");
            break;
        default:
            DEBUG_PRINT("MGL: Unknown draw_buffer value: 0x%x, falling back to FRONT\n", MGL_STATE(ctx)->draw_buffer);
            mgl_drawbuffer = _FRONT; // fallback to front instead of failing render setup
            NSLog(@"MGL WARNING: Unknown draw_buffer value 0x%x, using FRONT fallback", MGL_STATE(ctx)->draw_buffer);
            break;
    }

    if(![self checkDrawBufferSize:mgl_drawbuffer])
    {
        (void)mglRendererBackendClearDefaultDrawBuffer(
            _backend, mgl_drawbuffer);
        _drawBuffers[mgl_drawbuffer].width = 0;
        _drawBuffers[mgl_drawbuffer].height = 0;
    }

    // attach color buffer
    if (mgl_drawbuffer == _FRONT)
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
        uint32_t depthFormat = ctx->depth_format.mtl_pixel_format;
        if (depthFormat == MGLPixelFormatInvalid) {
            depthFormat = MGLPixelFormatDepth32Float;
        }

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
                    mglTraceLogNSString(@"MGL DEFAULT FBO: created depth attachment fmt=%lu size=%lux%lu drawBuffer=%u",
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
        uint32_t stencilFormat = ctx->stencil_format.mtl_pixel_format;
        if (stencilFormat == MGLPixelFormatInvalid ||
            stencilFormat == MGLPixelFormatDepth32Float_Stencil8) {
            stencilFormat = MGLPixelFormatStencil8;
        }

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
        &_commandState,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
        mglApplySRGBStateToRenderTarget(texture, ctx), 0, 0, 0, NO);
    mglRenderPassSetPersistentAttachment(
        &_commandState,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
        depth_texture, 0, 0, 0, NO);
    mglRenderPassSetPersistentAttachment(
        &_commandState,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
        stencil_texture, 0, 0, 0, NO);

    mglRenderPassSetPersistentDimensions(
        &_commandState, mglRenderPassTextureInfo(texture).width, mglRenderPassTextureInfo(texture).height);
    _drawBuffers[mgl_drawbuffer].width = (GLuint)mglRenderPassTextureInfo(texture).width;
    _drawBuffers[mgl_drawbuffer].height = (GLuint)mglRenderPassTextureInfo(texture).height;
    return true;
}

- (void) ensureTransientDepthForDefaultFramebufferLocked
{
    if (!MGL_STATE(ctx)->framebuffer &&
        MGL_STATE(ctx)->caps.depth_test &&
        !mglRenderPassDepthTextureFor(&_commandState)) {
        NSUInteger depthWidth = mglRenderPassRenderTargetWidthFor(&_commandState);
        NSUInteger depthHeight = mglRenderPassRenderTargetHeightFor(&_commandState);

        if (depthWidth == 0 || depthHeight == 0) {
            id color0 = mglRenderPassColorTextureFor(&_commandState, 0);
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
                depthDesc.pixel_format = MGLPixelFormatDepth32Float;
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
                              (unsigned long)MGLPixelFormatDepth32Float,
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
                    &_commandState,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    transientDepth,
                    0, 0, 0, NO);
                mglRenderPassSetPersistentActions(
                    &_commandState,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    MGLLoadActionClear, MGLStoreActionDontCare);
                mglRenderPassSetPersistentDepthClear(
                    &_commandState,
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
                &_commandState,
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
                (attachmentTextureForClear->mtl_rt_frame_generation != _commandState.dontCareFrameGeneration);
            attachmentTextureForClear->mtl_rt_frame_generation = _commandState.dontCareFrameGeneration;
        }
        if (att->clear_bitmask & GL_COLOR_BUFFER_BIT) {
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
                &_commandState, colorSlot,
                (MGLRenderPassClearColorValue){att->clear_color[0],
                                               att->clear_color[1],
                                               att->clear_color[2],
                                               att->clear_color[3]});
            mglRenderPassSetPersistentActions(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MGLLoadActionClear, MGLStoreActionStore);

            att->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
            mglMarkTextureLevelRenderTargetWritten(attachmentTextureForClear, att->level);

            (*outFboColorClearCount)++;
            *outFboColorClearMask |= (GLbitfield)(1u << attachmentIndex);
        } else if (dontCareLoadEnabled &&
                   [self shouldUseDontCareLoadForColorTexture:attachmentTextureForClear
                                                firstUseThisFrame:colorFirstUseThisFrame]) {

            mglRenderPassSetPersistentLoadAction(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MGLLoadActionDontCare);
        } else {
            mglRenderPassSetPersistentLoadAction(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                MGLLoadActionLoad);
        }
    }


    for (GLuint ai = 0; ai < MAX_COLOR_ATTACHMENTS; ++ai) {
        if ((fbo->color_attachments[ai].clear_bitmask & GL_COLOR_BUFFER_BIT) &&
            ((fbo->color_attachment_bitfield >> ai) & 1u) == 0u) {
            fbo->color_attachments[ai].clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        }
    }

    if (fbo->depth.clear_bitmask & GL_DEPTH_BUFFER_BIT) {
        mglRenderPassSetPersistentDepthClear(
            &_commandState, fbo->depth.clear_color[0]);
        mglRenderPassSetPersistentActions(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        fbo->depth.clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionLoad);
        if (mglRenderPassDepthTextureFor(&_commandState)) {
            mglRenderPassSetPersistentStoreAction(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MGLStoreActionStore);
        }
    }

    if (fbo->stencil.clear_bitmask & GL_STENCIL_BUFFER_BIT) {
        mglRenderPassSetPersistentStencilClear(
            &_commandState,
            (uint32_t)fbo->stencil.clear_color[0]);
        mglRenderPassSetPersistentActions(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        fbo->stencil.clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionLoad);
        if (mglRenderPassStencilTextureFor(&_commandState)) {
            mglRenderPassSetPersistentStoreAction(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MGLStoreActionStore);
        }
    }
}

- (void) configureDefaultFramebufferLoadStoreActionsLocked
{
    Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
    GLbitfield defaultClearMask = MGL_STATE(ctx)->default_fbo_clear_bitmask;
    if (defaultClearMask & GL_COLOR_BUFFER_BIT) {
        mglRenderPassSetPersistentColorClear(
            &_commandState, 0,
            (MGLRenderPassClearColorValue){MGL_STATE(ctx)->default_clear_color[0],
                                           MGL_STATE(ctx)->default_clear_color[1],
                                           MGL_STATE(ctx)->default_clear_color[2],
                                           MGL_STATE(ctx)->default_clear_color[3]});
        mglRenderPassSetPersistentActions(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
            MGLLoadActionLoad);
        static uint64_t s_defaultFboLoadLogCount = 0;
        uint64_t hit = ++s_defaultFboLoadLogCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            mglTraceLogNSString(@"MGL DEFAULT FBO: using Load (no clear mask) call=%llu drawBuf=0x%x fbo=%u",
                          (unsigned long long)hit,
                          MGL_STATE(ctx)->draw_buffer,
                          fbo ? (unsigned)fbo->name : 0u);
        }
    }

    if (defaultClearMask & GL_DEPTH_BUFFER_BIT) {
        mglRenderPassSetPersistentDepthClear(
            &_commandState,
            MGL_STATE(ctx)->var.depth_clear_value);
        mglRenderPassSetPersistentActions(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
            MGLLoadActionLoad);
        if (mglRenderPassDepthTextureFor(&_commandState)) {
            mglRenderPassSetPersistentStoreAction(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MGLStoreActionStore);
        }
    }

    if (defaultClearMask & GL_STENCIL_BUFFER_BIT) {
        mglRenderPassSetPersistentStencilClear(
            &_commandState,
            MGL_STATE(ctx)->var.stencil_clear_value);
        mglRenderPassSetPersistentActions(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionClear, MGLStoreActionStore);
        MGL_STATE(ctx)->default_fbo_clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
    } else {
        mglRenderPassSetPersistentLoadAction(
            &_commandState,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionLoad);
        if (mglRenderPassStencilTextureFor(&_commandState)) {
            mglRenderPassSetPersistentStoreAction(
                &_commandState,
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
	        MGLRenderPassClearColorValue c0 = mglRenderPassClearColorFor(&_commandState, 0, (MGLRenderPassClearColorValue){0, 0, 0, 0});
	        mglTraceLogNSString(@"MGL TRACE clear.resolve call=%llu fbo=%u "
	              "fboColorClears=%u fboColorMask=0x%x fboAtt0ClearMask=0x%x c0LA=%s depthLA=%s stencilLA=%s "
	              "c0Clear=(%.3f,%.3f,%.3f,%.3f) depthClear=%.3f stencilClear=%u",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)fboColorClearCount,
              (unsigned)fboColorClearMask,
              (unsigned)fboColorAttachment0ClearMask,
              mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
              mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
              mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLLoadActionDontCare)),
              c0.red,
              c0.green,
              c0.blue,
              c0.alpha,
	              mglRenderPassClearDepthFor(&_commandState, 0.0),
	              (unsigned)(unsigned)mglRenderPassClearStencilFor(&_commandState, 0));
	    }

            BOOL clearResolveInteresting =
                (fboColorClearCount != 0) ||
                (fboColorAttachment0ClearMask != 0) ||
                (mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare) == MGLLoadActionClear) ||
                (fboDepthClearMaskBefore & GL_DEPTH_BUFFER_BIT) ||
                (!fbo && (defaultClearMask & GL_DEPTH_BUFFER_BIT));
            if (clearResolveInteresting) {
                static uint64_t s_clearResolveDetailLogCount = 0;
                uint64_t hit = ++s_clearResolveDetailLogCount;
                if (mglTraceLogIsEnabled() && (hit <= 256ull || (hit % 512ull) == 0ull)) {
	            MGLRenderPassClearColorValue c0 = mglRenderPassClearColorFor(&_commandState, 0, (MGLRenderPassClearColorValue){0, 0, 0, 0});
	            id c0Tex = mglRenderPassColorTextureFor(&_commandState, 0);
	            id dTex = mglRenderPassDepthTextureFor(&_commandState);
	            id sTex = mglRenderPassStencilTextureFor(&_commandState);
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
	                        mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
	                        mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
	                        mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLLoadActionDontCare)),
	                        mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLStoreActionDontCare)),
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
	                        mglRenderPassClearDepthFor(&_commandState, 0.0),
	                        (unsigned)(unsigned)mglRenderPassClearStencilFor(&_commandState, 0),
	                        MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
	                        MGL_STATE(ctx)->var.depth_writemask ? 1 : 0,
	                        (unsigned)MGL_STATE(ctx)->var.depth_func);
	        }
	    }
}

- (bool) finalizeRenderPassDescriptorLocked:(uint64_t)renderEncoderCall
                          traceRenderEncoder:(bool)traceRenderEncoder
{
    mglRenderPassSetPersistentStoreAction(
        &_commandState,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
        MGLStoreActionStore);

    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
        id c0Tex = mglRenderPassColorTextureFor(&_commandState, 0);
        id dTex = mglRenderPassDepthTextureFor(&_commandState);
        id sTex = mglRenderPassStencilTextureFor(&_commandState);
        mglTraceLogNSString(@"MGL TRACE renderpass.attach call=%llu fbo=%u drawBuf=0x%x rt=%lux%lu "
              "c0=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s stencil=%p fmt=%lu size=%lux%lu la/sa=%s/%s",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)MGL_STATE(ctx)->draw_buffer,
              (unsigned long)mglRenderPassRenderTargetWidthFor(&_commandState),
              (unsigned long)mglRenderPassRenderTargetHeightFor(&_commandState),
              c0Tex,
              (unsigned long)(c0Tex ? mglRenderPassTextureInfo(c0Tex).pixel_format : MGLPixelFormatInvalid),
              (unsigned long)(c0Tex ? mglRenderPassTextureInfo(c0Tex).usage : 0),
              (unsigned long)(c0Tex ? mglRenderPassTextureInfo(c0Tex).width : 0),
              (unsigned long)(c0Tex ? mglRenderPassTextureInfo(c0Tex).height : 0),
              mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
              mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
              dTex,
              (unsigned long)(dTex ? mglRenderPassTextureInfo(dTex).pixel_format : MGLPixelFormatInvalid),
              (unsigned long)(dTex ? mglRenderPassTextureInfo(dTex).width : 0),
              (unsigned long)(dTex ? mglRenderPassTextureInfo(dTex).height : 0),
              mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
              mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
              sTex,
              (unsigned long)(sTex ? mglRenderPassTextureInfo(sTex).pixel_format : MGLPixelFormatInvalid),
              (unsigned long)(sTex ? mglRenderPassTextureInfo(sTex).width : 0),
              (unsigned long)(sTex ? mglRenderPassTextureInfo(sTex).height : 0),
              mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLLoadActionDontCare)),
              mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, MGLStoreActionDontCare)));
    }

    // create a render encoder from the renderpass descriptor
    // CRITICAL SAFETY: Validate inputs before creating render encoder
    BOOL hasRenderPassState =
        _commandState.renderPassStateOwner != NULL;
    if (!hasRenderPassState) {
        NSLog(@"MGL ERROR: Cannot create render encoder - state owner is NULL");
        [self recordGPUError];
        return false;
    }

    // Metal debug layer crashes if render pass has no output attachment.
    // Provide a tiny fallback color attachment for targetless/invalid passes.
    bool hasOutputAttachment = false;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRenderPassColorTextureFor(&_commandState, i)) {
            hasOutputAttachment = true;
            break;
        }
    }
    if (!hasOutputAttachment &&
        (mglRenderPassDepthTextureFor(&_commandState) || mglRenderPassStencilTextureFor(&_commandState))) {
        hasOutputAttachment = true;
    }

    if (!hasOutputAttachment) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
        NSUInteger fallbackWidth = fbo && fbo->default_width > 0
            ? (NSUInteger)fbo->default_width : 1u;
        NSUInteger fallbackHeight = fbo && fbo->default_height > 0
            ? (NSUInteger)fbo->default_height : 1u;
        NSUInteger fallbackLayers = fbo && fbo->default_layers > 0
            ? (NSUInteger)fbo->default_layers : 0u;
        NSUInteger fallbackSamples = fbo && fbo->default_samples > 0
            ? (NSUInteger)fbo->default_samples : 1u;
        id fallbackRenderTarget =
            mglRenderPassFallbackRenderTargetForSize(
                ctx, fallbackWidth, fallbackHeight, fallbackLayers,
                fallbackSamples);

        if (fallbackRenderTarget) {
            NSLog(@"MGL WARNING: Render pass had no attachments; binding %lux%lu fallback color target",
                  (unsigned long)fallbackWidth,
                  (unsigned long)fallbackHeight);
            mglRenderPassSetPersistentAttachment(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                fallbackRenderTarget,
                0, 0, 0, fallbackLayers > 0u ? YES : NO);
            mglRenderPassSetPersistentActions(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MGLLoadActionLoad, MGLStoreActionStore);
            mglRenderPassSetPersistentDimensions(
                &_commandState, fallbackWidth, fallbackHeight);
        } else {
            NSLog(@"MGL ERROR: Failed to allocate fallback render target texture");
            [self recordGPUError];
            return false;
        }
    }

    // Final guard: Metal will assert if a color attachment texture is missing RenderTarget usage.
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id attTex = mglRenderPassColorTextureFor(&_commandState, i);
        if (attTex && ((mglRenderPassTextureInfo(attTex).usage & MGLTextureUsageRenderTarget) == 0)) {
            NSLog(@"MGL WARNING: colorAttachment[%d] usage=0x%lx lacks RenderTarget; clearing attachment to avoid Metal assert",
                  i, (unsigned long)mglRenderPassTextureInfo(attTex).usage);
            NSUInteger clearLevel = 0u, clearSlice = 0u, clearDepthPlane = 0u;
            mglRenderGetRenderPassAttachmentSubresourceOwner(
                _commandState.renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i,
                (uint64_t *)&clearLevel, (uint64_t *)&clearSlice,
                (uint64_t *)&clearDepthPlane);
            mglRenderPassSetPersistentAttachment(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i,
                nil, clearLevel, clearSlice, clearDepthPlane, NO);
        }
    }

    // Default-framebuffer paths expect color attachment 0 specifically.
    // FBO draw-buffer mappings may intentionally leave slot 0 as GL_NONE.
    if (!MGL_STATE(ctx)->framebuffer && !mglRenderPassColorTextureFor(&_commandState, 0)) {
        for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (mglRenderPassColorTextureFor(&_commandState, i)) {
                NSLog(@"MGL WARNING: colorAttachment[0] missing; remapping colorAttachment[%d] -> [0]", i);
                NSUInteger srcLevel = 0u, srcSlice = 0u, srcDepthPlane = 0u;
                mglRenderGetRenderPassAttachmentSubresourceOwner(
                    _commandState.renderPassStateOwner,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i,
                    (uint64_t *)&srcLevel, (uint64_t *)&srcSlice,
                    (uint64_t *)&srcDepthPlane);
                mglRenderPassSetPersistentAttachment(
                    &_commandState,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    mglRenderPassColorTextureFor(&_commandState, i),
                    srcLevel, srcSlice, srcDepthPlane, NO);
                mglRenderPassSetPersistentActions(
                    &_commandState,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    (uint32_t)mglRenderPassLoadActionFor(
                        &_commandState,
                        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i,
                        MGLLoadActionLoad),
                    (uint32_t)mglRenderPassStoreActionFor(
                        &_commandState,
                        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i,
                        MGLStoreActionStore));
                break;
            }
        }
    }

    // Ultimate slot-0 fallback to keep draw path alive and avoid black frame.
    if (!hasOutputAttachment && !mglRenderPassColorTextureFor(&_commandState, 0)) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;
        NSUInteger fallbackWidth = fbo && fbo->default_width > 0
            ? (NSUInteger)fbo->default_width : 1u;
        NSUInteger fallbackHeight = fbo && fbo->default_height > 0
            ? (NSUInteger)fbo->default_height : 1u;
        NSUInteger fallbackLayers = fbo && fbo->default_layers > 0
            ? (NSUInteger)fbo->default_layers : 0u;
        NSUInteger fallbackSamples = fbo && fbo->default_samples > 0
            ? (NSUInteger)fbo->default_samples : 1u;
        id fallbackRenderTarget =
            mglRenderPassFallbackRenderTargetForSize(
                ctx, fallbackWidth, fallbackHeight, fallbackLayers,
                fallbackSamples);
        if (fallbackRenderTarget) {
            NSLog(@"MGL WARNING: colorAttachment[0] unavailable; binding %lux%lu fallback",
                  (unsigned long)fallbackWidth,
                  (unsigned long)fallbackHeight);
            mglRenderPassSetPersistentAttachment(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                fallbackRenderTarget,
                0, 0, 0, fallbackLayers > 0u ? YES : NO);
            mglRenderPassSetPersistentActions(
                &_commandState,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MGLLoadActionLoad, MGLStoreActionStore);
            mglRenderPassSetPersistentDimensions(
                &_commandState, fallbackWidth, fallbackHeight);
        } else {
            NSLog(@"MGL ERROR: Unable to allocate fallback colorAttachment[0] texture");
            [self recordGPUError];
            return false;
        }
    }

    // Ensure renderTargetWidth/Height are always coherent with the active attachments.
    {
        id sizeTex = mglRenderPassColorTextureFor(&_commandState, 0);
        if (!sizeTex) {
            for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
                if (mglRenderPassColorTextureFor(&_commandState, i)) {
                    sizeTex = mglRenderPassColorTextureFor(&_commandState, i);
                    break;
                }
            }
        }
        if (!sizeTex) {
            sizeTex = mglRenderPassDepthTextureFor(&_commandState);
        }
        if (!sizeTex) {
            sizeTex = mglRenderPassStencilTextureFor(&_commandState);
        }

        if (sizeTex) {
            NSUInteger texWidth = mglRenderPassTextureInfo(sizeTex).width;
            NSUInteger texHeight = mglRenderPassTextureInfo(sizeTex).height;
            if (mglRenderPassRenderTargetWidthFor(&_commandState) == 0 ||
                mglRenderPassRenderTargetHeightFor(&_commandState) == 0 ||
                mglRenderPassRenderTargetWidthFor(&_commandState) > texWidth ||
                mglRenderPassRenderTargetHeightFor(&_commandState) > texHeight) {
                if (kMGLVerboseFrameLoopLogs) {
                    NSLog(@"MGL INFO: Normalizing renderTarget size from %lux%lu to %lux%lu",
                          (unsigned long)mglRenderPassRenderTargetWidthFor(&_commandState),
                          (unsigned long)mglRenderPassRenderTargetHeightFor(&_commandState),
                          (unsigned long)texWidth,
                          (unsigned long)texHeight);
                }
                mglRenderPassSetPersistentDimensions(
                    &_commandState, texWidth, texHeight);
            }
        }
    }
    return true;
}

- (bool) createRenderEncoderLocked:(uint64_t)renderEncoderCall
{
    // CRITICAL FIX: Validate command buffer state before creating render encoder
    MGLRenderCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            _commandState.currentCommandBufferOwner,
            &commandState)) {
        NSLog(@"MGL ERROR: Cannot create render encoder - command buffer is NULL");
        [self recordGPUError];
        return false;
    }

    // Check if command buffer already has an active encoder (Metal API violation)
    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) == 1) {
        NSLog(@"MGL WARNING: Active render encoder detected - ending it before creating new one");
        [self endRenderEncodingLocked];
    }

    // Validate command buffer status. If already committed/completed, rotate to a new buffer.
    uint32_t bufferStatus =
        (uint32_t)commandState.status;
    if (bufferStatus >= MGLCommandBufferStatusCommitted) {
        NSLog(@"MGL WARNING: Render encoder requested on finalized command buffer (status: %ld) - creating a fresh command buffer", (long)bufferStatus);
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer before creating render encoder");
            [self recordGPUError];
            return false;
        }

        if (!mglRenderCommandBufferOwnerHasState(
                _commandState.currentCommandBufferOwner,
                &commandState)) {
            NSLog(@"MGL ERROR: newCommandBuffer returned without a current command buffer");
            [self recordGPUError];
            return false;
        }

        bufferStatus = (uint32_t)commandState.status;
        if (bufferStatus >= MGLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Fresh command buffer is still finalized (status: %ld)", (long)bufferStatus);
            [self recordGPUError];
            return false;
        }
    }

    if (kMGLVerboseFrameLoopLogs) {
        NSLog(@"MGL DEBUG: About to create render encoder with descriptor and command buffer");
    }
	    {
	        static uint64_t s_renderPassPreCreateLogCount = 0;
	        uint64_t hit = ++s_renderPassPreCreateLogCount;
		        if (mglTraceLogIsEnabled() && (hit <= 128ull || (hit % 512ull) == 0ull)) {
	            mglLogRenderPassLifecycle("pre-create",
	                                      hit,
                                      ctx,
                                      _commandState.currentCommandBufferOwner,
                                      _commandState.currentRenderEncoderOwner,
                                      _commandState.renderPassStateOwner,
                                      _drawable,
                                      _commandState.renderPassFramebuffer,
	                                      _commandState.renderPassFramebufferName,
	                                      _commandState.renderPassDrawBuffer,
	                                      _commandState.renderPassDrawBufferCount);
	            if (mglTraceLogIsEnabled()) {
	                id c0 = mglRenderPassColorTextureFor(&_commandState, 0);
	                id depth = mglRenderPassDepthTextureFor(&_commandState);
                MGLRenderPassState rpSnapshot = {0};
                (void)mglRenderPassGetPersistentState(&_commandState, &rpSnapshot);
                mglTraceLog("RENDERPASS_PRE_CREATE hit=%llu call=%llu program=%u fbo=%u drawBuf=0x%x readBuf=0x%x arrayLen=%lu colorLayered=%d depthLayered=%d stencilLayered=%d "
	                            "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                            "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
	                            "depthState(test=%d write=%d func=0x%x) pending(default=0x%x depth=0x%x)",
	                            (unsigned long long)hit,
	                            (unsigned long long)renderEncoderCall,
	                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
	                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                            (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
                            (unsigned)(ctx ? MGL_STATE(ctx)->read_buffer : 0u),
                            (unsigned long)rpSnapshot.render_target_array_length,
                            (int)rpSnapshot.color[0].attachment.layered,
                            (int)rpSnapshot.depth.attachment.layered,
                            (int)rpSnapshot.stencil.attachment.layered,
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[3] : 0),
	                            (ctx && MGL_STATE(ctx)->caps.scissor_test) ? 1 : 0,
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[3] : 0),
	                            c0,
	                            (unsigned long)(c0 ? mglRenderPassTextureInfo(c0).pixel_format : MGLPixelFormatInvalid),
	                            (unsigned long)(c0 ? mglRenderPassTextureInfo(c0).width : 0),
	                            (unsigned long)(c0 ? mglRenderPassTextureInfo(c0).height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
	                            depth,
	                            (unsigned long)(depth ? mglRenderPassTextureInfo(depth).pixel_format : MGLPixelFormatInvalid),
	                            (unsigned long)(depth ? mglRenderPassTextureInfo(depth).width : 0),
	                            (unsigned long)(depth ? mglRenderPassTextureInfo(depth).height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
	                            mglRenderPassClearDepthFor(&_commandState, 0.0),
	                            (ctx && MGL_STATE(ctx)->caps.depth_test) ? 1 : 0,
	                            (ctx && MGL_STATE(ctx)->var.depth_writemask) ? 1 : 0,
	                            (unsigned)(ctx ? MGL_STATE(ctx)->var.depth_func : 0u),
	                            (unsigned)(ctx ? MGL_STATE(ctx)->default_fbo_clear_bitmask : 0u),
	                            (unsigned)(ctx && MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->depth.clear_bitmask : 0u));
	            }
	        }
	    }
        /* When a GL sample query (GL_SAMPLES_PASSED / GL_ANY_SAMPLES_PASSED)
         * is active, attach the visibility result buffer to the render-pass
         * owner state so the GPU accumulates a fresh count. */
        void *queryVisibilityBuffer = NULL;
        if (_queryStateOwner &&
            mglRenderGetQueryVisibilityBuffer(
                _queryStateOwner, &queryVisibilityBuffer) == 0 &&
            queryVisibilityBuffer) {
            uint32_t visibilityResultType =
                mglRenderPassVisibilityResultTypeFor(&_commandState);
            mglRenderSetRenderPassStateVisibility(
                _commandState.renderPassStateOwner,
                queryVisibilityBuffer, visibilityResultType);
        }
        @try {
            id renderEncoder =
                (__bridge id)mglCmdCreateRenderEncoder(&_commandState);
            mglCmdInstallRenderEncoder(&_commandState, (__bridge void *)renderEncoder);
            if (mglRenderEncoderOwnerHasCurrent(
                    _commandState.currentRenderEncoderOwner) != 1) {
            NSLog(@"MGL ERROR: Failed to create render encoder - invalid render pass state or command buffer");
            NSLog(@"MGL DEBUG: Command buffer owner: %p, Render pass state owner: %p",
                  _commandState.currentCommandBufferOwner,
                  _commandState.renderPassStateOwner);
            [self recordGPUError];
            return false;
        }
        /* Enable visibility result mode on the encoder for all draws in this
         * pass when a sample query is active. MTLVisibilityResultModeBoolean
         * writes 1 to the buffer if any samples pass per-fragment tests. */
        if (_queryStateOwner &&
            mglRenderEncoderOwnerHasCurrent(
                _commandState.currentRenderEncoderOwner) == 1) {
            uint32_t visibilityMode = 0;
            uint64_t visibilityOffset = 0;
            if (mglRenderAcquireSampleQuerySlot(
                    _queryStateOwner, &visibilityMode,
                    &visibilityOffset) == 0) {
                mglRenderSetVisibilityResultModeForRenderEncoderOwner(
                    _commandState.currentRenderEncoderOwner,
                    visibilityMode, visibilityOffset);
            }
        }
        mglCmdUpdateRenderPassIdentityForContext(&_commandState, ctx);
        /* When trace is disabled, skip the full-struct memset and trace
         * call and clear only the functional flag fields. */
        if (mglTraceLogIsEnabled()) {
            mglTraceFragmentTextureTraceBindings("CLEAR",
                                                 "new_render_encoder",
                                                 _resourceFallback.fragmentTextureTraceBindings,
                                                 TEXTURE_UNITS,
                                                 ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                 _pipelineCacheState.pipelineProgramName);
            memset(_resourceFallback.fragmentTextureTraceBindings, 0,
                   sizeof(_resourceFallback.fragmentTextureTraceBindings));
        } else {
            mglClearFragmentTextureTraceFunctionalFlags(
                _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
        }
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created Metal render encoder");
        }
        [self recordGPUSuccess];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating render encoder: %@ - continuing with degraded functionality", exception);
        NSLog(@"MGL DEBUG: Exception details - name: %@, reason: %@", exception.name, exception.reason);
        [self recordGPUError];
        mglCmdClearCurrentRenderEncoder(&_commandState);
        return false;
    }
    mglRenderSetRenderEncoderOwnerLabel(
        _commandState.currentRenderEncoderOwner,
        "GL Render Encoder");
	    {
	        static uint64_t s_renderPassCreatedLogCount = 0;
	        uint64_t hit = ++s_renderPassCreatedLogCount;
		        if (mglTraceLogIsEnabled() && (hit <= 128ull || (hit % 512ull) == 0ull)) {
	            mglLogRenderPassLifecycle("created",
	                                      hit,
                                      ctx,
                                      _commandState.currentCommandBufferOwner,
                                      _commandState.currentRenderEncoderOwner,
                                      _commandState.renderPassStateOwner,
                                      _drawable,
                                      _commandState.renderPassFramebuffer,
	                                      _commandState.renderPassFramebufferName,
	                                      _commandState.renderPassDrawBuffer,
	                                      _commandState.renderPassDrawBufferCount);
	            if (mglTraceLogIsEnabled()) {
	                id c0 = mglRenderPassColorTextureFor(&_commandState, 0);
	                id depth = mglRenderPassDepthTextureFor(&_commandState);
	                mglTraceLog("RENDERPASS_CREATED hit=%llu call=%llu program=%u fbo=%u rpFbo=%u drawBuf=0x%x readBuf=0x%x "
	                            "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                            "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
	                            "depthState(test=%d write=%d func=0x%x)",
	                            (unsigned long long)hit,
	                            (unsigned long long)renderEncoderCall,
	                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
	                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
	                            (unsigned)_commandState.renderPassFramebufferName,
	                            (unsigned)(ctx ? MGL_STATE(ctx)->draw_buffer : 0u),
	                            (unsigned)(ctx ? MGL_STATE(ctx)->read_buffer : 0u),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->viewport[3] : 0),
	                            (ctx && MGL_STATE(ctx)->caps.scissor_test) ? 1 : 0,
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[0] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[1] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[2] : 0),
	                            (int)(ctx ? MGL_STATE(ctx)->var.scissor_box[3] : 0),
	                            c0,
	                            (unsigned long)(c0 ? mglRenderPassTextureInfo(c0).pixel_format : MGLPixelFormatInvalid),
	                            (unsigned long)(c0 ? mglRenderPassTextureInfo(c0).width : 0),
	                            (unsigned long)(c0 ? mglRenderPassTextureInfo(c0).height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
	                            depth,
	                            (unsigned long)(depth ? mglRenderPassTextureInfo(depth).pixel_format : MGLPixelFormatInvalid),
	                            (unsigned long)(depth ? mglRenderPassTextureInfo(depth).width : 0),
	                            (unsigned long)(depth ? mglRenderPassTextureInfo(depth).height : 0),
	                            mglLoadActionName(mglRenderPassLoadActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
	                            mglStoreActionName(mglRenderPassStoreActionFor(&_commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
	                            mglRenderPassClearDepthFor(&_commandState, 0.0),
	                            (ctx && MGL_STATE(ctx)->caps.depth_test) ? 1 : 0,
	                            (ctx && MGL_STATE(ctx)->var.depth_writemask) ? 1 : 0,
	                            (unsigned)(ctx ? MGL_STATE(ctx)->var.depth_func : 0u));
	            }
	        }
	    }
    return true;
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

    [self invalidateLastBoundState];

    static uint64_t s_newRenderEncoderCallCount = 0;
    uint64_t renderEncoderCall = ++s_newRenderEncoderCallCount;
    bool traceRenderEncoder = mglShouldTraceCall(renderEncoderCall) ||
                              (kMGLDiagnosticStateLogs && ((renderEncoderCall % 60ull) == 0ull));

    // AGX ERROR THROTTLING: Check if we should skip render encoder creation
    // BUT allow limited render encoder creation for essential functionality
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: Render encoder creation requested during GPU recovery - attempting essential creation");
        // Continue with essential render encoder creation even during recovery
    }

    // CRITICAL SAFETY: Check command buffer before creating render encoder
    if (mglRenderCommandBufferOwnerHasCurrent(
            _commandState.currentCommandBufferOwner) != 1) {
        // Attempt recovery: create a new command buffer instead of failing immediately
        if ([self newCommandBufferLocked]) {
            // Successfully created - continue
        } else {
            NSLog(@"MGL ERROR: Cannot create render encoder - no command buffer available");
            [self recordGPUError];
            return false;
        }
    }

    // end encoding on current render encoder
    [self endRenderEncodingLocked];

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

    mglCmdInstallNewRenderPassDescriptor(&_commandState);
    if (!_commandState.renderPassStateOwner) {
        NSLog(@"MGL RENDERPASS ERROR: failed to allocate render pass state owner");
        return false;
    }

    // Configure color/depth/stencil attachments based on FBO type
    if (MGL_STATE(ctx)->framebuffer) {
        RETURN_FALSE_ON_FAILURE([self configureUserFBOAttachmentsLocked]);
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

    RETURN_FALSE_ON_FAILURE([self finalizeRenderPassDescriptorLocked:renderEncoderCall
                                                  traceRenderEncoder:traceRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self createRenderEncoderLocked:renderEncoderCall]);

    // Apply dynamic state that is not part of the render-pass owner state.
    [self updateCurrentRenderEncoder];

    // Only bind buffers when creating the encoder. Sampled textures depend on the
    // current GL program/MSL reflection and are rebound after the pipeline state is
    // selected for the draw.
    if (MGL_STATE(ctx)->vao)
    {
        MGLEncodeContext encCtx = {
            .render_encoder_owner = _commandState.currentRenderEncoderOwner,
        };
        if ([self bindVertexBuffersToCurrentRenderEncoder:&encCtx] == false)
        {
            DEBUG_PRINT("vertex buffer binding failed\n");
            [self recordGPUError];
            return false;
        }

        if ([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx] == false)
        {
            DEBUG_PRINT("fragment buffer binding failed\n");
            [self recordGPUError];
            return false;
        }
    }

    // Record successful render encoder creation (final success)
    [self recordGPUSuccess];
    return true;

    } //     @autoreleasepool
}

- (bool) newCommandBuffer
{
    METAL_LOCK();
    bool result = [self newCommandBufferLocked];
    METAL_UNLOCK();
    return result;
}

- (bool) newCommandBufferLocked
{
    // CRITICAL FIX: Proper encoder cleanup BEFORE creating new command buffer
    // Metal API requires ending encoders before creating new command buffers

    // STEP 0: End any existing render encoder to prevent MTLReleaseAssertionFailure
    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) == 1) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Ending existing render encoder before creating new command buffer");
        }
        [self endRenderEncodingLocked];
    }

    // STEP 1: Clean up sync tracking list safely.
    // IMPORTANT: Do NOT dereference Sync* entries here. Sync objects are owned by GL sync lifecycle
    // and may already be deleted by glDeleteSync on other paths.
    // Both this read/clear path and the backend sync append path run on the GL
    // calling thread, so no lock is needed.
    mglCmdClearCurrentCommandBufferSyncListEntries(&_commandState);

    /* A successful C++ submit transaction rotates the owner to the next
     * current command buffer before returning. Consume that exact buffer so
     * the ObjC adapter does not immediately allocate and release another one.
     * Unmarked current buffers still follow the ordinary fresh-rotate path. */
    if (mglCmdConsumeTransactionCreatedCurrentCommandBuffer(&_commandState)) {
        _currentCBHasWork = NO;
        return true;
    }

    // CRITICAL SAFETY: Validate command queue before creating buffer
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Cannot create command buffer - command queue is NULL");
        mglCmdDiscardCurrentCommandBuffer(&_commandState);
        return false;
    }

    // STEP 1: Create fresh command buffer FIRST with comprehensive AGX driver validation
    @try {
        // AGX DRIVER COMPATIBILITY: Validate command queue health before creating buffer
        if (!_commandQueue) {
            NSLog(@"MGL AGX ERROR: Command queue is NULL - recreating");
            [self resetMetalState];
            if (!_commandQueue) {
                NSLog(@"MGL AGX CRITICAL: Cannot recreate command queue");
                return false;
            }
        }

        // CRITICAL FIX: Validate _commandQueue before dereferencing to prevent NULL pointer crashes
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: _commandQueue is NULL - cannot create command buffer");
            [self recordGPUError];
            return false;
        }

        if (!mglCmdInstallNewCommandBufferFromQueue(&_commandState, (__bridge void *)_commandQueue)) {
            NSLog(@"MGL AGX ERROR: Failed to create Metal command buffer - command queue may be in error state");
            [self recordGPUError];
            // Force command queue recreation
            [self resetMetalState];
            return false;
        }

        _currentCBHasWork = NO;

        // AGX Driver Validation: Check if the command buffer is immediately invalid
        MGLRenderCommandBufferState initialState = {0};
        if (!mglRenderCommandBufferOwnerHasState(
                _commandState.currentCommandBufferOwner,
                &initialState)) {
            NSLog(@"MGL AGX CRITICAL: New command buffer owner has no current buffer");
            [self recordGPUError];
            return false;
        }
        if (initialState.has_error) {
            NSLog(@"MGL AGX WARNING: New command buffer has immediate error: %s",
                  mglRenderCommandBufferErrorDescription(&initialState));
            [self recordGPUError];
            // Don't return false immediately - AGX sometimes creates error-state buffers that recover
        }

        // AGX DRIVER COMPATIBILITY: Enhanced validation to prevent rejections
        if (initialState.status == MGLCommandBufferStatusError) {
            NSLog(@"MGL AGX CRITICAL: Command buffer immediately in error state");
            [self recordGPUError];
            mglCmdDiscardCurrentCommandBuffer(&_commandState);
            [self resetMetalState]; // Force full reset
            return false;
        }

        // Additional AGX validation: Check for buffer properties that cause rejections
        memset(&initialState, 0, sizeof(initialState));
        (void)mglRenderCommandBufferOwnerHasState(
            _commandState.currentCommandBufferOwner,
            &initialState);
        if (initialState.has_error) {
            NSLog(@"MGL AGX WARNING: Command buffer has immediate error: %s",
                  mglRenderCommandBufferErrorDescription(&initialState));
            [self recordGPUError];
            mglCmdDiscardCurrentCommandBuffer(&_commandState);
            [self resetMetalState];
            return false;
        }

        // Validate command queue health
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: Command queue became NULL");
            [self resetMetalState];
            return false;
        }

        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created new Metal command buffer (AGX validated)");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: Exception creating command buffer: %@", exception);
        [self recordGPUError];
        mglCmdDiscardCurrentCommandBuffer(&_commandState);

        // AGX DRIVER COMPATIBILITY: Force reset on exception to clear driver state
        [self resetMetalState];
        return false;
    }

    // STEP 2: Now handle pending event waits on the FRESH command buffer.
    GLuint cachedSyncName = 0;
    id cachedEvent =
        (__bridge_transfer id)mglCmdDetachPendingEventWithSyncName(&_commandState, &cachedSyncName);
    if (cachedEvent) {
        if (!cachedSyncName) {
            NSLog(@"MGL WARNING: dropping pending shared-event wait with no sync name");
            return true;
        }

        if (kMGLDisableSharedEventSync) {
            NSLog(@"MGL INFO: Shared event wait disabled (debug no-op), skipping wait encode event=%p syncName=%u",
                  cachedEvent, cachedSyncName);
            return true;
        }

        // SAFELY ENCODE: Event wait functionality on the new command buffer
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Encoding event wait on fresh command buffer");
        }

        // Validate event pointer looks like a valid object address
        uintptr_t eventPtr = (uintptr_t)cachedEvent;
        if (eventPtr == 0x10 || eventPtr == 0x30 || eventPtr == 0x1000) {
            NSLog(@"MGL CRITICAL ERROR: Known corrupted event pointer pattern detected: 0x%lx", eventPtr);
            NSLog(@"MGL CRITICAL ERROR: Skipping event wait to prevent crash");
            return false;
        }

        if (eventPtr < 0x1000 || (eventPtr & 0x7) != 0) {
            NSLog(@"MGL ERROR: Suspicious event pointer value: %p", cachedEvent);
            NSLog(@"MGL INFO: Skipping event wait for safety");
            return false;
        }

        // ADDITIONAL SAFETY: Validate command buffer is still valid before encoding
        if (mglRenderCommandBufferOwnerHasCurrent(
                _commandState.currentCommandBufferOwner) != 1) {
            NSLog(@"MGL ERROR: Command buffer became NULL before event wait encoding");
            return false;
        }

        @try {
            NSLog(@"MGL INFO: Encoding safe event wait: event=%p, syncName=%u",
                  cachedEvent, cachedSyncName);
            if (mglRenderEncodeWaitForEventForCommandBufferOwner(
                    _commandState.currentCommandBufferOwner,
                    (__bridge void *)cachedEvent, cachedSyncName) != 0) {
                NSLog(@"MGL ERROR: Event wait owner facade rejected the request");
                return false;
            }
            NSLog(@"MGL SUCCESS: Event wait encoded successfully on fresh command buffer");
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Event wait failed - %@: %@", exception.name, exception.reason);
            NSLog(@"MGL INFO: Continuing without event wait to maintain stability");
            // Continue without event wait - system remains stable
        }

    }

    return true;
}

- (bool)ensureWritableCommandBuffer:(const char *)reason
{
    METAL_LOCK();
    bool result = [self ensureWritableCommandBufferLocked:reason];
    METAL_UNLOCK();
    return result;
}

- (bool)ensureWritableCommandBufferLocked:(const char *)reason
{
    MGLRenderCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            _commandState.currentCommandBufferOwner,
            &commandState)) {
        if (kMGLDiagnosticStateLogs) {
            mglTraceLogNSString(@"MGL INFO: %s requested with NULL command buffer, creating one", reason ? reason : "operation");
        }
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to create command buffer for %s", reason ? reason : "operation");
            return false;
        }
        if (!mglRenderCommandBufferOwnerHasState(
                _commandState.currentCommandBufferOwner,
                &commandState)) {
            NSLog(@"MGL ERROR: Created command buffer owner has no current buffer for %s",
                  reason ? reason : "operation");
            return false;
        }
    }

    uint32_t status =
        (uint32_t)commandState.status;
    if (status >= MGLCommandBufferStatusCommitted) {
        NSLog(@"MGL INFO: %s requested on finalized command buffer (status: %ld), rotating", reason ? reason : "operation", (long)status);
        [self endRenderEncodingLocked];
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer for %s", reason ? reason : "operation");
            return false;
        }

        memset(&commandState, 0, sizeof(commandState));
        if (!mglRenderCommandBufferOwnerHasState(
                _commandState.currentCommandBufferOwner,
                &commandState) ||
            commandState.status >= MGLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Unable to obtain writable command buffer for %s", reason ? reason : "operation");
            return false;
        }
    }

    return true;
}

- (bool) newCommandBufferAndRenderEncoder
{
    // AGGRESSIVE MEMORY SAFETY: Validate fundamental Metal objects before use
    if (!_device) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No device available");
        return false;
    }

    if (!_commandQueue) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No command queue available");
        return false;
    }

    // Validate device pointer lower bound only (high canonical addresses are valid on macOS)
    uintptr_t device_addr = (uintptr_t)_device;
    if (device_addr < 0x1000) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Invalid device pointer: 0x%lx", device_addr);
        return false;
    }

    @try {
        if ([self newCommandBuffer] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newCommandBuffer failed");
            return false;
        }

        if ([self newRenderEncoderWithReason:MGL_ENC_REASON_CMD] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newRenderEncoder failed");
            return false;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Metal operation failed: %@", exception);
        return false;
    }

    return true;
}

#pragma mark pipeline descriptor
/* Build the renderer pipeline as C ABI value-state. Color/depth/stencil,
 * blend, vertex layout, tessellation, rasterization, topology, and sample
 * count are consumed by the C++ pipeline builder. */
- (BOOL)generatePipelineDescriptorState:(MGLRenderPipelineDescriptorState *)state
                         vertexFunction:(id *)vertexFunctionOut
                       fragmentFunction:(id *)fragmentFunctionOut
{
    if (!ctx) {
        NSLog(@"MGL PIPELINE DESC fail: context is NULL");
        return NO;
    }
    if (!state || !vertexFunctionOut || !fragmentFunctionOut) {
        NSLog(@"MGL PIPELINE DESC fail: bad out args");
        return NO;
    }
    *vertexFunctionOut = nil;
    *fragmentFunctionOut = nil;

    const BOOL nativeTES = _tessellation.nativeTESActive;
    const BOOL tessVertexCapture = _tessellation.tessVertexCaptureActive;
    const BOOL cullDistanceCapture =
        _tessellation.cullDistanceCaptureActive;
    const BOOL geometryExpansion = _geometry.expansionActive;
    const BOOL tessCompute = _tessellation.tessComputeActive;
    const int vertexStage = nativeTES ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    Program *vertexProgram = nativeTES
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *fragmentProgram = (tessVertexCapture || cullDistanceCapture)
        ? NULL : mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    GLuint renderProgramKey = mglCurrentRenderProgramKey(ctx);
    GLuint vertexProgramName = vertexProgram ? vertexProgram->name : 0u;
    GLuint fragmentProgramName = fragmentProgram ? fragmentProgram->name : 0u;
    BOOL rasterizerDiscard = tessVertexCapture || cullDistanceCapture ||
        MGL_STATE(ctx)->caps.rasterizer_discard ? YES : NO;

    if (!vertexProgram || (!fragmentProgram && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing stage program key=%u vs=%p fs=%p current=%u pipeline=%u",
              (unsigned)renderProgramKey,
              vertexProgram,
              fragmentProgram,
              (unsigned)MGL_STATE(ctx)->program_name,
              (unsigned)MGL_STATE(ctx)->var.program_pipeline_binding);
        return NO;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC begin key=%u vsProgram=%u fsProgram=%u",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName);
    }

    if ([self bindMTLProgram:vertexProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for VS program=%u",
              (unsigned)vertexProgramName);
        return NO;
    }
    if (fragmentProgram &&
        fragmentProgram != vertexProgram &&
        [self bindMTLProgram:fragmentProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for FS program=%u",
              (unsigned)fragmentProgramName);
        return NO;
    }

    Shader *vertex_shader = vertexProgram->shader_slots[vertexStage];
    Shader *fragment_shader = fragmentProgram ? fragmentProgram->shader_slots[_FRAGMENT_SHADER] : NULL;
    if (!vertex_shader || (!fragment_shader && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing shaders key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName,
              vertex_shader,
              fragment_shader);
        return NO;
    }

    void *geometryPassthroughFunction = NULL;
    if (geometryExpansion && _geometry.program) {
        (void)mglRendererBackendGetPassthroughFunction(
            _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
            _geometry.program->pipeline_cache_instance_id,
            &geometryPassthroughFunction);
    }
    void *tessPassthroughFunction = NULL;
    if (tessCompute && _tessellation.tessComputeProgram) {
        (void)mglRendererBackendGetPassthroughFunction(
            _backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
            _tessellation.tessComputeProgram->pipeline_cache_instance_id,
            &tessPassthroughFunction);
    }
    void *vertexFunctionPtr = geometryExpansion
        ? geometryPassthroughFunction
        : tessCompute
        ? tessPassthroughFunction
        : cullDistanceCapture
        ? vertexProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function
        : tessVertexCapture
        ? vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function
        : vertexProgram->modules[vertexStage].mtl_function;
    id vertexFunction = (__bridge id)vertexFunctionPtr;


    id fragmentFunction = fragmentProgram
        ? (__bridge id)fragmentProgram->modules[_FRAGMENT_SHADER].mtl_function
        : nil;
    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC vs=%p fs=%p",
              vertexFunction, fragmentFunction);
    }
    if (!vertexFunction || (!fragmentFunction && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing MTLFunction key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName,
              vertexFunction,
              fragmentFunction);
        return NO;
    }

    memset(state, 0, sizeof(*state));
    state->vertex_program_instance =
        vertexProgram->pipeline_cache_instance_id;
    state->vertex_program_generation =
        vertexProgram->pipeline_cache_generation;
    state->fragment_program_instance = fragmentProgram
        ? fragmentProgram->pipeline_cache_instance_id : 0u;
    state->fragment_program_generation = fragmentProgram
        ? fragmentProgram->pipeline_cache_generation : 0u;
    state->color_count = MAX_COLOR_ATTACHMENTS;
    state->rasterization_enabled = rasterizerDiscard ? 0 : 1;

    state->max_tessellation_factor = 64u;


    {
        /* Metal requires the pipeline's primitive topology class to match
         * the drawn primitive type: an unspecified-class pipeline silently
         * drops point draws.  A compute-routed geometry expansion always
         * gets its output class explicitly.  For ordinary draws only the
         * point case is forced: leaving triangles on the historical
         * unspecified value keeps programs that write gl_PointSize while
         * drawing triangles linkable (Metal rejects a triangle-class
         * pipeline whose vertex function writes point size). */
        if (geometryExpansion || _lastDrawPrimitiveMode == GL_POINTS) {
            switch (_lastDrawPrimitiveMode) {
                case GL_POINTS:
                    state->input_primitive_topology =
                        (uint32_t)MGLPrimitiveTopologyClassPoint;
                    break;
                case GL_LINES:
                case GL_LINE_STRIP:
                case GL_LINE_LOOP:
                case GL_LINES_ADJACENCY:
                case GL_LINE_STRIP_ADJACENCY:
                    state->input_primitive_topology =
                        (uint32_t)MGLPrimitiveTopologyClassLine;
                    break;
                default:
                    state->input_primitive_topology =
                        (uint32_t)MGLPrimitiveTopologyClassTriangle;
                    break;
            }
        }
    }

    if (nativeTES) {
        switch (vertexProgram->tess_gen_spacing) {
            case GL_FRACTIONAL_EVEN:
                state->tessellation_partition_mode =
                    (uint32_t)MGLTessellationPartitionModeFractionalEven;
                break;
            case GL_FRACTIONAL_ODD:
                state->tessellation_partition_mode =
                    (uint32_t)MGLTessellationPartitionModeFractionalOdd;
                break;
            default:
                state->tessellation_partition_mode =
                    (uint32_t)MGLTessellationPartitionModeInteger;
                break;
        }
        state->max_tessellation_factor = 64u;
        state->tessellation_factor_scale_enabled = 0;
        state->tessellation_factor_format =
            (uint32_t)MGLTessellationFactorFormatHalf;

        state->tessellation_control_point_index_type =
            _tessellation.tessIndexedDraw
                ? (uint32_t)MGLTessellationControlPointIndexTypeUInt32
                : (uint32_t)MGLTessellationControlPointIndexTypeNone;
        state->tessellation_factor_step_function =
            (uint32_t)MGLTessellationFactorStepFunctionPerPatch;
        state->tessellation_output_winding_order =
            vertexProgram->tess_gen_vertex_order == GL_CW
                ? (uint32_t)MGLWindingClockwise
                : (uint32_t)MGLWindingCounterClockwise;
    }


    if (tessVertexCapture || cullDistanceCapture) {
        state->rasterization_enabled = 0;
    } else if (rasterizerDiscard) {
        GLuint vsOutputCount = vertexProgram->shader_resources_list[vertexStage][_STAGE_OUTPUT_RES].count;
        state->rasterization_enabled =
            (nativeTES || vsOutputCount > 0) ? 1 : 0;
    } else {
        state->rasterization_enabled = 1;
    }

    /* Attachment formats: FBO attachment -> pass/drawable/context fallback. */
    if (MGL_STATE(ctx)->framebuffer) {
        Framebuffer *fbo = MGL_STATE(ctx)->framebuffer;

        for (int i = 0; i < MGL_STATE(ctx)->max_color_attachments; i++) {
            if (fbo->color_attachments[i].texture) {
                Texture *tex = [self framebufferAttachmentTexture:&fbo->color_attachments[i]];
                if (tex && ![self bindMTLTexture:tex]) {
                    NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for color attachment %d tex=%u",
                          i, tex->name);
                    return NO;
                }
                if (tex && tex->mtl_data) {
                    state->color_format[i] = (uint32_t)mtlPixelFormatForGLTex(tex);
                } else {
                    state->color_format[i] = (uint32_t)MGLPixelFormatInvalid;
                }
            }

            if ((fbo->color_attachment_bitfield >> (i + 1)) == 0) {
                break;
            }
        }

        if (fbo->depth.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->depth];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for depth tex=%u", tex->name);
                return NO;
            }
            if (tex && tex->mtl_data) {
                uint32_t depthFormat = mtlPixelFormatForGLTex(tex);
                if (depthFormat == MGLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid depth texture format, falling back to Depth32Float");
                    depthFormat = MGLPixelFormatDepth32Float;
                }
                state->depth_format = (uint32_t)depthFormat;
            } else {
                state->depth_format = (uint32_t)MGLPixelFormatInvalid;
            }
        }

        if (fbo->stencil.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->stencil];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for stencil tex=%u", tex->name);
                return NO;
            }
            if (tex && tex->mtl_data) {
                uint32_t stencilFormat = mtlPixelFormatForGLTex(tex);
                if (stencilFormat == MGLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid stencil texture format, falling back to Stencil8");
                    stencilFormat = MGLPixelFormatStencil8;
                }
                state->stencil_format = (uint32_t)stencilFormat;
            } else {
                state->stencil_format = (uint32_t)MGLPixelFormatInvalid;
            }
        }
    } else {
        uint32_t preferredColor0 = MGLPixelFormatInvalid;
        if (&_commandState && mglRenderPassColorTextureFor(&_commandState, 0)) {
            preferredColor0 = mglRenderPassTextureInfo(
                mglRenderPassColorTextureFor(&_commandState, 0)).pixel_format;
        } else if (_drawable && [self mglDrawableTexture]) {
            preferredColor0 = mglRenderPassTextureInfo([self mglDrawableTexture]).pixel_format;
        } else {
            preferredColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        state->color_format[0] = (uint32_t)preferredColor0;

        if (ctx->depth_format.format) {
            uint32_t depthFormat = ctx->depth_format.mtl_pixel_format;
            if (depthFormat == MGLPixelFormatInvalid) {
                depthFormat = MGLPixelFormatDepth32Float;
            }
            state->depth_format = (uint32_t)depthFormat;
        }

        if (ctx->stencil_format.format) {
            uint32_t stencilFormat = ctx->stencil_format.mtl_pixel_format;
            if (stencilFormat == MGLPixelFormatInvalid ||
                stencilFormat == MGLPixelFormatDepth32Float_Stencil8) {
                stencilFormat = MGLPixelFormatStencil8;
            }
            state->stencil_format = (uint32_t)stencilFormat;
        }
    }

    /* Derive pipeline attachment formats from the configured C++ pass. */
    BOOL hasConfiguredRenderPass =
        _commandState.renderPassStateOwner != NULL;
    if (hasConfiguredRenderPass) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id rpColor = mglRenderPassColorTextureFor(&_commandState, i);
            if (rpColor) {
                state->color_format[i] =
                    mglRenderPassTextureInfo(rpColor).pixel_format;
            }
        }

        id rpDepth = mglRenderPassDepthTextureFor(&_commandState);
        id rpStencil = mglRenderPassStencilTextureFor(&_commandState);
        state->depth_format =
            rpDepth ? (uint32_t)mglRenderPassTextureInfo(rpDepth).pixel_format : (uint32_t)MGLPixelFormatInvalid;
        state->stencil_format =
            rpStencil ? (uint32_t)mglRenderPassTextureInfo(rpStencil).pixel_format : (uint32_t)MGLPixelFormatInvalid;
    }

    BOOL color0IsIntentionallyDisabled =
        MGL_STATE(ctx)->framebuffer &&
        mglMetalDrawBufferAt(ctx, 0u) == GL_NONE;

    if (!color0IsIntentionallyDisabled &&
        (state->color_format[0] == (uint32_t)MGLPixelFormatInvalid ||
         state->color_format[0] == 0u)) {
        uint32_t fallbackColor0 = MGLPixelFormatInvalid;
        if (&_commandState && mglRenderPassColorTextureFor(&_commandState, 0)) {
            fallbackColor0 = mglRenderPassTextureInfo(
                mglRenderPassColorTextureFor(&_commandState, 0)).pixel_format;
        } else if (_drawable && [self mglDrawableTexture]) {
            fallbackColor0 = mglRenderPassTextureInfo([self mglDrawableTexture]).pixel_format;
        } else {
            fallbackColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        if (fallbackColor0 == MGLPixelFormatInvalid || fallbackColor0 == 0) {
            fallbackColor0 = MGLPixelFormatBGRA8Unorm;
        }
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE DESC missing color pixel format, fallback pixelFormat=%lu",
                  (unsigned long)fallbackColor0);
        }
        state->color_format[0] = (uint32_t)fallbackColor0;
    }

    /* Resolve the pipeline sample count from the C++ render-pass state. */
    NSUInteger resolvedSampleCount = 1;
    id rpColor0 = mglRenderPassColorTextureFor(&_commandState, 0);
    id rpDepth = mglRenderPassDepthTextureFor(&_commandState);
    id rpStencil = mglRenderPassStencilTextureFor(&_commandState);
    if (rpColor0 && mglRenderPassTextureInfo(rpColor0).sample_count > 0) {
        resolvedSampleCount = mglRenderPassTextureInfo(rpColor0).sample_count;
    } else if (rpDepth && mglRenderPassTextureInfo(rpDepth).sample_count > 0) {
        resolvedSampleCount = mglRenderPassTextureInfo(rpDepth).sample_count;
    } else if (rpStencil && mglRenderPassTextureInfo(rpStencil).sample_count > 0) {
        resolvedSampleCount = mglRenderPassTextureInfo(rpStencil).sample_count;
    }
    if (resolvedSampleCount == 0) {
        resolvedSampleCount = 1;
    }
    state->raster_sample_count = (uint32_t)resolvedSampleCount;


    {
        uint32_t depthFormat = state->depth_format;
        uint32_t stencilFormat = state->stencil_format;
        if (depthFormat != (uint32_t)MGLPixelFormatInvalid &&
            stencilFormat != (uint32_t)MGLPixelFormatInvalid &&
            depthFormat != stencilFormat) {
            bool depthPacked =
                depthFormat == (uint32_t)MGLPixelFormatDepth24Unorm_Stencil8 ||
                depthFormat == (uint32_t)MGLPixelFormatDepth32Float_Stencil8;
            bool stencilPacked =
                stencilFormat == (uint32_t)MGLPixelFormatDepth24Unorm_Stencil8 ||
                stencilFormat == (uint32_t)MGLPixelFormatDepth32Float_Stencil8;
            if (depthPacked || stencilPacked) {
                uint32_t packedFormat = stencilPacked ? stencilFormat : depthFormat;
                state->depth_format = packedFormat;
                state->stencil_format = packedFormat;
            }
        }
    }


    state->alpha_to_coverage_enabled = MGL_STATE(ctx)->caps.sample_alpha_to_coverage ? 1 : 0;
    state->alpha_to_one_enabled = MGL_STATE(ctx)->caps.sample_alpha_to_one ? 1 : 0;


    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (state->color_format[i] == (uint32_t)MGLPixelFormatInvalid) {
            continue;
        }
        if (mglMetalDrawBufferAt(ctx, (GLuint)i) == GL_NONE) {
            state->color_write_mask[i] = 0u;
            continue;
        }
        MGLRenderPipelineBlendState blend = {0};
        if (!mglPipelineCacheBlendStateForAttachment(
                &_pipelineCacheState, &_pipelineCacheOwner,
                (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                (uint32_t)i, &blend)) {
            NSLog(@"MGL PIPELINE DESC fail: blend state unavailable for attachment %d", i);
            return NO;
        }
        state->color_write_mask[i] = blend.color_write_mask;
        if (MGL_STATE(ctx)->caps.blendi[i]) {
            state->blending_enabled_mask |= 1u << i;
        }
        state->source_rgb_blend_factor[i] = blend.source_rgb_factor;
        state->destination_rgb_blend_factor[i] = blend.destination_rgb_factor;
        state->source_alpha_blend_factor[i] = blend.source_alpha_factor;
        state->destination_alpha_blend_factor[i] = blend.destination_alpha_factor;
        state->rgb_blend_operation[i] = blend.rgb_operation;
        state->alpha_blend_operation[i] = blend.alpha_operation;
    }


    if (!(geometryExpansion || tessCompute)) {
        if (![self generateVertexDescriptorState:state]) {
            return NO;
        }
    }

    if (kMGLVerbosePipelineLogs) {
        uint32_t activeColorAttachmentCount = 0;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (state->color_format[i] != (uint32_t)MGLPixelFormatInvalid &&
                state->color_format[i] != 0u) {
                activeColorAttachmentCount++;
            }
        }
        NSLog(@"MGL PIPELINE DESC colorAttachmentCount=%u depthFormat=%u stencilFormat=%u sampleCount=%u",
              (unsigned)activeColorAttachmentCount,
              (unsigned)state->depth_format,
              (unsigned)state->stencil_format,
              (unsigned)state->raster_sample_count);
        NSLog(@"MGL PIPELINE DESC renderTarget[0]=%u",
              (unsigned)state->color_format[0]);
    }

    *vertexFunctionOut = vertexFunction;
    *fragmentFunctionOut = fragmentFunction;
    return YES;
}

#pragma mark vertex descriptor

- (void)updateGLSampledCopiesForEndedRenderPassFramebuffer:(Framebuffer *)fbo
                                                  drawCount:(GLsizei)drawCount
                                               drawBuffers:(const GLenum *)drawBuffers
                                                    reason:(const char *)reason
{
    (void)drawCount;
    (void)drawBuffers;

    if (!ctx || !fbo) {
        return;
    }


    bool anySampledRT = false;
    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }
        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *tex = [self framebufferAttachmentTexture:attachment];
        if (tex && tex->mtl_data && tex->is_render_target &&
            tex->mtl_render_target_write_version != 0u) {
            anySampledRT = true;
            break;
        }
    }
    if (!anySampledRT) {
        return;
    }

    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];

        Texture *tex = [self framebufferAttachmentTexture:attachment];
        if (!tex || !tex->mtl_data) {
            continue;
        }

        id source = (__bridge id)(tex->mtl_data);
        if (![self textureCanUseGLSampledRenderTargetCopy:tex source:source]) {
            continue;
        }


        if (mglRTWriteAuthorityIsCurrentAndUsesOriginal(tex)) {
            if (tex->mtl_gl_sampled_data &&
                tex->mtl_gl_sampled_write_version != tex->mtl_render_target_write_version) {
                [self releaseGLSampledRenderTargetCopyForTexture:tex];
                if (mglTraceLogIsEnabled()) {
                    mglTraceLog("RT_SAMPLE_COPY_SKIP_INJECTED_RENDER tex=%u label=\"%s\" reason=render_yflip_injected_stale_released",
                                (unsigned)tex->name,
                                mglTraceTextureLabel(tex));
                }
            }
            continue;
        }

        [self updateGLSampledRenderTargetCopyForTexture:tex
                                                 source:source
                                                 reason:reason ? reason : "end_render_pass"];
    }
}

- (void) endRenderEncoding
{
    METAL_LOCK();
    [self endRenderEncodingLocked];
    METAL_UNLOCK();
}

- (void) endRenderEncodingLocked
{

    [self invalidateLastBoundState];

    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) == 1)
    {
        /* An active render encoder means work was encoded into the current
         * CB, so flushCommandBufferLocked: must not skip the commit. */
        _currentCBHasWork = YES;

        Framebuffer *endedFramebuffer = _commandState.renderPassFramebuffer;
        GLsizei endedDrawBufferCount = _commandState.renderPassDrawBufferCount;
        GLenum endedDrawBuffers[MAX_COLOR_ATTACHMENTS];
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            endedDrawBuffers[i] = _commandState.renderPassDrawBuffers[i];
        }

        static uint64_t s_renderPassEndLogCount = 0;
        uint64_t hit = ++s_renderPassEndLogCount;
        if (hit <= 128ull || (hit % 1024ull) == 0ull) {
            mglLogRenderPassLifecycle("end",
                                      hit,
                                      ctx,
                                      _commandState.currentCommandBufferOwner,
                                      _commandState.currentRenderEncoderOwner,
                                      _commandState.renderPassStateOwner,
                                      _drawable,
                                      _commandState.renderPassFramebuffer,
                                      _commandState.renderPassFramebufferName,
                                      _commandState.renderPassDrawBuffer,
                                      _commandState.renderPassDrawBufferCount);
        }
        @try {
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Ending render encoder");
            }
            mglCmdEndCurrentRenderEncoder(&_commandState);
            mglCmdClearCurrentRenderEncoder(&_commandState);
            /* When trace is disabled, skip the full-struct memset and
             * trace call and clear only the functional flag fields. */
            if (mglTraceLogIsEnabled()) {
                mglTraceFragmentTextureTraceBindings("CLEAR",
                                                     "end_render_encoding",
                                                     _resourceFallback.fragmentTextureTraceBindings,
                                                     TEXTURE_UNITS,
                                                     ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                     _pipelineCacheState.pipelineProgramName);
                memset(_resourceFallback.fragmentTextureTraceBindings, 0,
                       sizeof(_resourceFallback.fragmentTextureTraceBindings));
            } else {
                mglClearFragmentTextureTraceFunctionalFlags(
                    _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
            }
            mglCmdClearRenderPassIdentity(&_commandState);
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Render encoder ended successfully");
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception ending render encoder: %@ - ignoring", exception.reason);
            // Force clear the encoder even if ending failed
            mglCmdClearCurrentRenderEncoder(&_commandState);
            /* When trace is disabled, skip the full-struct memset and
             * trace call and clear only the functional flag fields. */
            if (mglTraceLogIsEnabled()) {
                mglTraceFragmentTextureTraceBindings("CLEAR",
                                                     "end_render_encoding_exception",
                                                     _resourceFallback.fragmentTextureTraceBindings,
                                                     TEXTURE_UNITS,
                                                     ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                     _pipelineCacheState.pipelineProgramName);
                memset(_resourceFallback.fragmentTextureTraceBindings, 0,
                       sizeof(_resourceFallback.fragmentTextureTraceBindings));
            } else {
                mglClearFragmentTextureTraceFunctionalFlags(
                    _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
            }
            mglCmdClearRenderPassIdentity(&_commandState);
        }

        /* A later batch may sample this render target before the command
         * buffer is submitted, so refresh its GL-visible copy immediately. */
        if (endedFramebuffer) {
            [self updateGLSampledCopiesForEndedRenderPassFramebuffer:endedFramebuffer
                                                            drawCount:endedDrawBufferCount
                                                         drawBuffers:endedDrawBuffers
                                                              reason:"end_render_pass"];
        }
    }
}

- (BOOL)currentRenderPassUsesTexture:(id)texture
{
    if (!texture || mglRenderEncoderOwnerHasCurrent(
                        _commandState.currentRenderEncoderOwner) != 1) {
        return NO;
    }
    if (!_commandState.renderPassStateOwner) {
        return NO;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRenderPassColorTextureFor(&_commandState, i) == texture) {
            return YES;
        }
    }
    if (mglRenderPassDepthTextureFor(&_commandState) == texture ||
        mglRenderPassStencilTextureFor(&_commandState) == texture) {
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
            _commandState.currentCommandBufferOwner,
            &commandState)) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    if (commandState.status != MGLCommandBufferStatusNotEnqueued) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    id commandBufferToCommit =
        (__bridge id)mglCmdDetachCurrentCommandBufferForSubmission(&_commandState);

    @try {
        [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
        mglRenderPassWaitCommandBuffer(commandBufferToCommit);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: failed to synchronize render pass for texture readback (%s): %@",
              reason ? reason : "texture_readback",
              exception.reason);
        [self recordGPUError];
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
        [self recordGPUError];
        [self newCommandBuffer];
        return NO;
    }

    return [self newCommandBuffer];
}

// ULTIMATE FAILSAFE: Emergency Metal state reset to recover from corruption
- (void) emergencyResetMetalState
{
    NSLog(@"MGL CRITICAL: Performing emergency Metal state reset");

    @try {
        // Force cleanup of all Metal objects
        [self endRenderEncodingLocked];

        mglCmdDiscardCurrentCommandBuffer(&_commandState);
        mglCmdClearCurrentRenderEncoder(&_commandState);
        _drawable = NULL;

        // Re-initialize basic Metal objects
        if (_device && _commandQueue) {
            NSLog(@"MGL CRITICAL: Re-creating Metal command buffer");
            mglCmdInstallNewCommandBufferFromQueue(&_commandState, (__bridge void *)_commandQueue);

            if (mglRenderCommandBufferOwnerHasCurrent(
                    _commandState.currentCommandBufferOwner) != 1) {
                NSLog(@"MGL CRITICAL: Failed to create new command buffer during recovery");
            }
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: Emergency Metal reset failed: %@", exception);
    }
}

static bool mglProcessGLStatePreambleBridgeEnsureMetal(void *renderer)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    static int corruption_recovery_count = 0;
    static const int max_recovery_attempts = 3;

    if (self->_device && self->_commandQueue &&
        (uintptr_t)self->_device >= 0x1000 &&
        (uintptr_t)self->_commandQueue >= 0x1000) {
        return true;
    }

    NSLog(@"MGL CRITICAL: Metal state corruption detected in processGLState!");
    NSLog(@"MGL CRITICAL: device=0x%lx, queue=0x%lx",
          (uintptr_t)self->_device, (uintptr_t)self->_commandQueue);

    if (corruption_recovery_count >= max_recovery_attempts) {
        NSLog(@"MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations");
        return false;
    }

    NSLog(@"MGL CRITICAL: Attempting Metal state recovery (%d/%d)",
          corruption_recovery_count + 1, max_recovery_attempts);
    @try {
        [self emergencyResetMetalState];
        corruption_recovery_count++;
        if (!self->_device || !self->_commandQueue) {
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

static int mglProcessGLStatePreambleBridgeHandleNullVao(void *renderer,
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
        self->_gpuRecovery.interfaceMismatchBlockedProgram == 0 ||
        blockedProgramKey != self->_gpuRecovery.interfaceMismatchBlockedProgram) {
        return true;
    }
    CFTimeInterval now = CFAbsoluteTimeGetCurrent();
    if (now >= self->_gpuRecovery.interfaceMismatchBlockedUntil) {
        return true;
    }
    static uint64_t s_quarantineSkipCount = 0;
    s_quarantineSkipCount++;
    if (s_quarantineSkipCount <= 16 || (s_quarantineSkipCount % 1000) == 0) {
        double remaining =
            self->_gpuRecovery.interfaceMismatchBlockedUntil - now;
        if (remaining < 0.0) {
            remaining = 0.0;
        }
        NSLog(@"MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw",
              (unsigned)self->_gpuRecovery.interfaceMismatchBlockedProgram,
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
                            self->ctx,
                            self->_commandState.currentCommandBufferOwner,
                            self->_commandState.currentRenderEncoderOwner,
                            self->_commandState.renderPassStateOwner,
                            self->_drawable);
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
                            self->ctx,
                            self->_commandState.currentCommandBufferOwner,
                            self->_commandState.currentRenderEncoderOwner,
                            self->_commandState.renderPassStateOwner,
                            self->_drawable);
    }
    return false;
}

static int mglProcessGLStatePreambleBridge(MGLRenderer *self, bool draw_command,
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
        self->ctx, &self->_commandState, draw_command ? 1 : 0, process_call,
        trace_process ? 1 : 0, &preambleOps);
}

static bool mglProcessGLStateTailBridgeRecoverNilEncoder(void *renderer,
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
                                  self->ctx,
                                  self->_commandState.currentCommandBufferOwner,
                                  self->_commandState.currentRenderEncoderOwner,
                                  self->_commandState.renderPassStateOwner,
                                  self->_drawable,
                                  self->_commandState.renderPassFramebuffer,
                                  self->_commandState.renderPassFramebufferName,
                                  self->_commandState.renderPassDrawBuffer,
                                  self->_commandState.renderPassDrawBufferCount);
    }
    if (![self newRenderEncoderLockedWithReason:MGL_ENC_REASON_NIL]) {
        return false;
    }
    if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
        mglLogRenderPassLifecycle("nil-encoder-after-recovery",
                                  nilHit,
                                  self->ctx,
                                  self->_commandState.currentCommandBufferOwner,
                                  self->_commandState.currentRenderEncoderOwner,
                                  self->_commandState.renderPassStateOwner,
                                  self->_drawable,
                                  self->_commandState.renderPassFramebuffer,
                                  self->_commandState.renderPassFramebufferName,
                                  self->_commandState.renderPassDrawBuffer,
                                  self->_commandState.renderPassDrawBufferCount);
    }
    return true;
}

static bool mglProcessGLStateTailBridgePrepareDrawPass(void *renderer,
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
          self->_pipelineCacheState.pipelineState,
          (unsigned)lookupProgramName,
          lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
          lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u,
          MGL_STATE(context)->vao, (unsigned)lookupFBOName);
}

static bool mglProcessGLStateTailBridgeEnsurePipelineReady(
    void *renderer, GLMContext context, int trace_process)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    if (self->_pipelineCacheState.pipelineState) {
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
                            self->ctx,
                            self->_commandState.currentCommandBufferOwner,
                            self->_commandState.currentRenderEncoderOwner,
                            self->_commandState.renderPassStateOwner,
                            self->_drawable);
    }
    return false;
}

static bool mglProcessGLStateTailBridgeValidateRenderPass(
    void *renderer, GLMContext context, int trace_process)
{
    (void)context;
    return [(__bridge MGLRenderer *)renderer
        validateRenderPassAttachmentsAndPipelineFormatsLocked:trace_process];
}

static bool mglProcessGLStateTailBridgeBindPipeline(void *renderer,
                                                    GLMContext context,
                                                    int trace_process)
{
    (void)context;
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    @try {
        if (mglRenderBindingSetPipelineIfNeededForOwner(
                self->_bindingStateOwner,
                self->_commandState.currentRenderEncoderOwner,
                self->_pipelineCacheState.pipelineState) > 0) {
            MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: processGLState - setRenderPipelineState failed: %@",
              exception.reason);
        mglMarkRendererDirtyBits(self->ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        if (trace_process) {
            mglLogStateSnapshot("processGLState.fail.set_pipeline",
                                self->ctx,
                                self->_commandState.currentCommandBufferOwner,
                                self->_commandState.currentRenderEncoderOwner,
                                self->_commandState.renderPassStateOwner,
                                self->_drawable);
        }
        return false;
    }
    return true;
}

static bool mglProcessGLStateTailBridgeApplyPostBindDrawState(void *renderer,
                                                              GLMContext context)
{
    MGLRenderer *self = (__bridge MGLRenderer *)renderer;
    Program *fragmentProgram =
        mglResolveProgramForStageFromState(context, _FRAGMENT_SHADER);
    if (fragmentProgram && fragmentProgram->usesFragCoordParams == GL_TRUE) {
        NSUInteger passHeight =
            mglRenderPassRenderTargetHeightFor(&self->_commandState);
        if (passHeight == 0) {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS && passHeight == 0; i++) {
                id color = mglRenderPassColorTextureFor(&self->_commandState, i);
                passHeight =
                    color ? mglRenderPassTextureInfo(color).height : 0;
            }
            if (passHeight == 0 &&
                mglRenderPassDepthTextureFor(&self->_commandState)) {
                passHeight = mglRenderPassTextureInfo(
                    mglRenderPassDepthTextureFor(&self->_commandState)).height;
            }
            if (passHeight == 0 &&
                mglRenderPassStencilTextureFor(&self->_commandState)) {
                passHeight = mglRenderPassTextureInfo(
                    mglRenderPassStencilTextureFor(&self->_commandState)).height;
            }
        }
        vector_float4 fragCoordParams = {
            (float)passHeight,
            MGL_STATE(context)->var.clip_origin == GL_LOWER_LEFT ? 1.0f
                                                                 : 0.0f,
            0.0f,
            0.0f};
        mglRenderSetRenderBytesForOwner(
            self->_commandState.currentRenderEncoderOwner, &fragCoordParams,
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
            self->_commandState.currentRenderEncoderOwner, lodBiasArr,
            sizeof(lodBiasArr), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLLodBiasBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLLodBiasBufferIndex];
        mglRenderSetRenderBytesForOwner(
            self->_commandState.currentRenderEncoderOwner, &biasmax,
            sizeof(biasmax), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLLodBiasMaxBufferIndex);
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLLodBiasMaxBufferIndex];
    }
    if (mglFragmentTextureTraceBindingsUseRTSampledCopy(
            self->_resourceFallback.fragmentTextureTraceBindings,
            TEXTURE_UNITS)) {
        mglCmdSetCurrentDrawUsesRTSampledCopy(&self->_commandState, YES);
        [self updateCurrentRenderEncoder];
    }
    return true;
}

static bool mglProcessGLStateTailBridge(MGLRenderer *self, bool draw_command,
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
               self->ctx, &self->_commandState, draw_command ? 1 : 0,
               trace_process ? 1 : 0, resource_sync_work, &tailOps) != 0;
}

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
        mglTraceLogNSString(@"MGL TRACE processGLState.begin call=%llu draw=%d",
              (unsigned long long)processCall, draw_command ? 1 : 0);
        mglLogStateSnapshot("processGLState.enter",
                            ctx,
                            _commandState.currentCommandBufferOwner,
                            _commandState.currentRenderEncoderOwner,
                            _commandState.renderPassStateOwner,
                            _drawable);
    }
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                _commandState.currentCommandBufferOwner,
                                _commandState.currentRenderEncoderOwner,
                                _commandState.renderPassStateOwner,
                                _drawable);
        }
        return false;
    }

    uintptr_t earlyCtxAddr = (uintptr_t)ctx;
    if (earlyCtxAddr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", earlyCtxAddr);
        return false;
    }

    int preambleResult = mglProcessGLStatePreambleBridge(
        self, draw_command, processCall, traceProcess);
    if (preambleResult == MGL_PREAMBLE_FAIL) {
        return false;
    }
    if (preambleResult == MGL_PREAMBLE_DONE_OK) {
        double processElapsedUs = (mglTraceClockNS() - processStartNS) / 1000.0;
        if (traceProcess) {
            mglTraceLogNSString(
                @"MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.1fus",
                (unsigned long long)processCall, draw_command ? 1 : 0,
                processElapsedUs);
            mglLogStateSnapshot("processGLState.exit.ok",
                                ctx,
                                _commandState.currentCommandBufferOwner,
                                _commandState.currentRenderEncoderOwner,
                                _commandState.renderPassStateOwner,
                                _drawable);
        } else if (processElapsedUs >= 25.0) {
            mglTraceLogNSString(
                @"MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.1fus",
                (unsigned long long)processCall, draw_command ? 1 : 0,
                processElapsedUs);
        }
        return true;
    }

    MGLResourceSyncWork resourceSyncWork = {false, false, false};
    RETURN_FALSE_ON_FAILURE([self processDirtyStateDomainsLocked:draw_command
                                                            work:&resourceSyncWork]);

    RETURN_FALSE_ON_FAILURE(
        mglProcessGLStateTailBridge(self, draw_command, traceProcess,
                                    &resourceSyncWork));

    double processElapsedUs = (mglTraceClockNS() - processStartNS) / 1000.0;
    if (traceProcess) {
        mglTraceLogNSString(@"MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.1fus",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedUs);
        mglLogStateSnapshot("processGLState.exit.ok",
                            ctx,
                            _commandState.currentCommandBufferOwner,
                            _commandState.currentRenderEncoderOwner,
                            _commandState.renderPassStateOwner,
                            _drawable);
    } else if (processElapsedUs >= 25.0) {
        mglTraceLogNSString(@"MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.1fus",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedUs);
    }
    return true;
}
/*
 * Dirty state domain processing — orchestration lives in mgl_renderer_sync.cpp;
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

static bool mglSyncBridgeSyncFbo(void *renderer, GLMContext context)
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
    return mglRenderSyncRenderPassForFbo(context, &self->_commandState,
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
    if (!draw_command || self->_pipelineCacheState.pipelineState != nil ||
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
        .render_encoder_owner = self->_commandState.currentRenderEncoderOwner,
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

- (bool)processDirtyStateDomainsLocked:(bool)draw_command
                                  work:(MGLResourceSyncWork *)work
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
               ctx, MGL_SYNC_DOMAIN_ALL, draw_command ? 1 : 0, &_commandState,
               work, &ops) != 0;
}

/*
 * Render pass descriptor and pipeline format validation extracted from
 * processGLStateLocked:. Validates render-pass attachments and checks
 * pipeline/pass color, depth, and stencil format compatibility. Returns
 * false to skip the draw on validation failure, true to continue.
 */
- (bool)validateRenderPassAttachmentsAndPipelineFormatsLocked:(BOOL)traceProcess
{
    // Guard against invalid render pass state before binding pipeline.
    // Metal debug validation can abort the process if the encoder/render pass is incompatible.
    BOOL hasRenderPassState =
        _commandState.renderPassStateOwner != NULL;
    if (!hasRenderPassState) {
        NSLog(@"MGL ERROR: processGLState - render pass state owner is nil before pipeline bind");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_rpd",
                                ctx,
                                _commandState.currentCommandBufferOwner,
                                _commandState.currentRenderEncoderOwner,
                                _commandState.renderPassStateOwner,
                                _drawable);
        }
        return false;
    }
    BOOL passHasAnyAttachment = NO;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id colorAttachment = mglRenderPassColorTextureFor(&_commandState, i);
        if (colorAttachment) {
            passHasAnyAttachment = YES;
            if ((mglRenderPassTextureInfo(colorAttachment).usage & MGLTextureUsageRenderTarget) == 0) {
                NSLog(@"MGL WARNING: processGLState - color attachment %d missing RenderTarget usage (usage=0x%lx); skipping draw",
                      i,
                      (unsigned long)mglRenderPassTextureInfo(colorAttachment).usage);
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.color_usage",
                                        ctx,
                                        _commandState.currentCommandBufferOwner,
                                        _commandState.currentRenderEncoderOwner,
                                        _commandState.renderPassStateOwner,
                                        _drawable);
                }
                return false;
            }
        }
    }
    if (mglRenderPassDepthTextureFor(&_commandState) ||
        mglRenderPassStencilTextureFor(&_commandState)) {
        passHasAnyAttachment = YES;
    }

    if (!passHasAnyAttachment) {
        NSLog(@"MGL WARNING: processGLState - render pass has no attachments, skipping draw to avoid Metal assert");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.no_attachments",
                                ctx,
                                _commandState.currentCommandBufferOwner,
                                _commandState.currentRenderEncoderOwner,
                                _commandState.renderPassStateOwner,
                                _drawable);
        }
        return false;
    }

    uint32_t currentColor0Format = MGLPixelFormatInvalid;
    uint32_t currentDepthFormat = MGLPixelFormatInvalid;
    uint32_t currentStencilFormat = MGLPixelFormatInvalid;

    id rpColor0 = mglRenderPassColorTextureFor(&_commandState, 0);
    id rpDepth = mglRenderPassDepthTextureFor(&_commandState);
    id rpStencil = mglRenderPassStencilTextureFor(&_commandState);
    if (rpColor0) {
        currentColor0Format = mglRenderPassTextureInfo(rpColor0).pixel_format;
    }
    if (rpDepth) {
        currentDepthFormat = mglRenderPassTextureInfo(rpDepth).pixel_format;
    }
    if (rpStencil) {
        currentStencilFormat = mglRenderPassTextureInfo(rpStencil).pixel_format;
    }

    // IMPORTANT:
    // Never mutate depth/stencil attachments here to "fit" an existing pipeline.
    // The active Metal render encoder was already created with a render-pass descriptor,
    // and changing attachments after encoder creation does not make that encoder compatible.
    // We must instead reject mismatched pipeline/pass combinations and rebuild safely.

    if (_pipelineCacheState.pipelineColor0Format != MGLPixelFormatInvalid &&
        currentColor0Format != MGLPixelFormatInvalid &&
        _pipelineCacheState.pipelineColor0Format != currentColor0Format) {
        static uint64_t s_colorFormatMismatchCount = 0;
        s_colorFormatMismatchCount++;
	        if (s_colorFormatMismatchCount <= 16 || (s_colorFormatMismatchCount % 250) == 0) {
	            NSLog(@"MGL WARNING: Pipeline/pass color format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	                  (unsigned long)_pipelineCacheState.pipelineColor0Format, (unsigned long)currentColor0Format);
	        }
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass color format mismatch"];
	        mglMarkRendererDirtyBits(ctx->active_state,
	                                 DIRTY_PROGRAM | DIRTY_VAO |
	                                 DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }

    if (_pipelineCacheState.pipelineDepthFormat != currentDepthFormat) {
        BOOL pipelineHasDepth = (_pipelineCacheState.pipelineDepthFormat != MGLPixelFormatInvalid);
        BOOL passHasDepth = (currentDepthFormat != MGLPixelFormatInvalid);
        if (!pipelineHasDepth && !passHasDepth) {
            goto depth_format_ok;
	        }
	        {
	            static uint64_t s_depthFormatMismatchCount = 0;
	            s_depthFormatMismatchCount++;
	            if (s_depthFormatMismatchCount <= 16 || (s_depthFormatMismatchCount % 250) == 0) {
	                NSLog(@"MGL WARNING: Pipeline/pass depth format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	                      (unsigned long)_pipelineCacheState.pipelineDepthFormat, (unsigned long)currentDepthFormat);
	            }
	        }
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass depth format mismatch"];
	        mglMarkRendererDirtyBits(ctx->active_state,
	                                 DIRTY_PROGRAM | DIRTY_VAO |
	                                 DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }
depth_format_ok:;

    if (_pipelineCacheState.pipelineStencilFormat != currentStencilFormat) {
        BOOL pipelineHasStencil = (_pipelineCacheState.pipelineStencilFormat != MGLPixelFormatInvalid);
        BOOL passHasStencil = (currentStencilFormat != MGLPixelFormatInvalid);
        if (!pipelineHasStencil && !passHasStencil) {
            goto stencil_format_ok;
	        }
	        {
	            static uint64_t s_stencilFormatMismatchCount = 0;
	            s_stencilFormatMismatchCount++;
	            if (s_stencilFormatMismatchCount <= 16 || (s_stencilFormatMismatchCount % 250) == 0) {
	                NSLog(@"MGL WARNING: Pipeline/pass stencil format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	                      (unsigned long)_pipelineCacheState.pipelineStencilFormat, (unsigned long)currentStencilFormat);
	            }
	        }
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass stencil format mismatch"];
	        mglMarkRendererDirtyBits(ctx->active_state,
	                                 DIRTY_PROGRAM | DIRTY_VAO |
	                                 DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }
stencil_format_ok:;
    return true;
}

static MGLPipelineRecoveryState mglPipelineRecoveryViewFromGPU(
    const MGLGPURecoveryState *gpu)
{
    MGLPipelineRecoveryState view = {0};
    if (!gpu) {
        return view;
    }
    view.pipeline_retry_after = gpu->pipelineRetryAfter;
    view.interface_mismatch_retry_after = gpu->interfaceMismatchRetryAfter;
    view.program_mismatch_retry_after = gpu->programMismatchRetryAfter;
    view.interface_mismatch_program_name = gpu->interfaceMismatchProgramName;
    view.interface_mismatch_color0_format = gpu->interfaceMismatchColor0Format;
    view.interface_mismatch_depth_format = gpu->interfaceMismatchDepthFormat;
    view.interface_mismatch_stencil_format = gpu->interfaceMismatchStencilFormat;
    view.interface_mismatch_streak = gpu->interfaceMismatchStreak;
    view.program_mismatch_program_name = gpu->programMismatchProgramName;
    view.program_mismatch_streak = gpu->programMismatchStreak;
    view.interface_mismatch_blocked_program = gpu->interfaceMismatchBlockedProgram;
    view.interface_mismatch_blocked_until = gpu->interfaceMismatchBlockedUntil;
    view.interface_mismatch_blocked_streak = gpu->interfaceMismatchBlockedStreak;
    return view;
}

static void mglPipelineRecoveryApplyToGPU(MGLGPURecoveryState *gpu,
                                          const MGLPipelineRecoveryState *view)
{
    if (!gpu || !view) {
        return;
    }
    gpu->pipelineRetryAfter = view->pipeline_retry_after;
    gpu->interfaceMismatchRetryAfter = view->interface_mismatch_retry_after;
    gpu->programMismatchRetryAfter = view->program_mismatch_retry_after;
    gpu->interfaceMismatchProgramName = view->interface_mismatch_program_name;
    gpu->interfaceMismatchColor0Format = view->interface_mismatch_color0_format;
    gpu->interfaceMismatchDepthFormat = view->interface_mismatch_depth_format;
    gpu->interfaceMismatchStencilFormat = view->interface_mismatch_stencil_format;
    gpu->interfaceMismatchStreak = view->interface_mismatch_streak;
    gpu->programMismatchProgramName = view->program_mismatch_program_name;
    gpu->programMismatchStreak = view->program_mismatch_streak;
    gpu->interfaceMismatchBlockedProgram = view->interface_mismatch_blocked_program;
    gpu->interfaceMismatchBlockedUntil = view->interface_mismatch_blocked_until;
    gpu->interfaceMismatchBlockedStreak = view->interface_mismatch_blocked_streak;
}

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
            if (_pipelineCacheState.psoDedupEnabled &&
                mglBindingStateIsValid(_bindingStateOwner) &&
                mglBindingStatePipelineMatches(
                    _bindingStateOwner,
                _pipelineCacheState.pipelineState)) {
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

            MGLPipelineRecoveryState recoveryView =
                mglPipelineRecoveryViewFromGPU(&_gpuRecovery);

            // Program-level breaker (independent of render-pass signature) to avoid
            // mismatch storms where color/depth/stencil signatures keep changing.
            if (mglPipelineRecoveryShouldAbortForProgramMismatch(
                    &recoveryView, now, (uint32_t)currentProgramName,
                    _pipelineCacheState.pipelineState)) {
                static uint64_t s_programMismatchSkipCount = 0;
                s_programMismatchSkipCount++;
                if (s_programMismatchSkipCount <= 16 || (s_programMismatchSkipCount % 1000ull) == 0ull) {
                    double remaining = recoveryView.program_mismatch_retry_after - now;
                    if (remaining < 0.0) remaining = 0.0;
                    NSLog(@"MGL WARNING: Program-level mismatch breaker active (program=%u, %.2fs remaining), skipping draw",
                          (unsigned)currentProgramName,
                          remaining);
                }
                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

            if (mglPipelineRecoveryEvaluatePipelineRetry(
                    &recoveryView, now, (uint32_t)currentProgramName,
                    _pipelineCacheState.pipelineState, &skipPipelineBuild)) {
                if (skipPipelineBuild) {
                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                }
            } else if (now < recoveryView.pipeline_retry_after) {
                static uint64_t s_retryBypassCount = 0;
                s_retryBypassCount++;
                if (s_retryBypassCount <= 16 || (s_retryBypassCount % 1000ull) == 0ull) {
                    NSLog(@"MGL PIPELINE RETRY bypass global retry for unrelated program=%u mismatchProgram=%u blockedProgram=%u",
                          (unsigned)currentProgramName,
                          (unsigned)recoveryView.interface_mismatch_program_name,
                          (unsigned)recoveryView.interface_mismatch_blocked_program);
                }
            }
            mglPipelineRecoveryApplyToGPU(&_gpuRecovery, &recoveryView);

            if (!skipPipelineBuild) {
            // Build the only renderer pipeline representation: C ABI value-state.
            MGLRenderPipelineDescriptorState psoState = {0};
            id psoVertexFunction = nil;
            id psoFragmentFunction = nil;
            uint32_t builtColor0Format = (uint32_t)MGLPixelFormatInvalid;
            uint32_t builtDepthFormat = (uint32_t)MGLPixelFormatInvalid;
            uint32_t builtStencilFormat = (uint32_t)MGLPixelFormatInvalid;

            [self updateBlendStateCache];
            state->dirty_bits &= ~DIRTY_ALPHA_STATE;
            if (![self generatePipelineDescriptorState:&psoState
                                        vertexFunction:&psoVertexFunction
                                      fragmentFunction:&psoFragmentFunction]) {
                NSLog(@"MGL PIPELINE CREATE fail error=generatePipelineDescriptorState returned NO");
                [self invalidateCurrentPipelineStateForReason:@"pipeline descriptor failure"];
                _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                mglMarkRendererDirtyBits(state,
                                         DIRTY_PROGRAM | DIRTY_VAO |
                                         DIRTY_FBO | DIRTY_RENDER_STATE);
                return false;
            }
            builtColor0Format = psoState.color_format[0];
            builtDepthFormat = psoState.depth_format;
            builtStencilFormat = psoState.stencil_format;

            // Circuit breaker for repeated VS/FS interface mismatch.
            if (mglPipelineRecoveryShouldAbortForInterfaceMismatch(
                    &recoveryView, now, (uint32_t)currentProgramName,
                    builtColor0Format, builtDepthFormat, builtStencilFormat)) {
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

                MGLPipelineCacheKeyInputs keyInputs = {
                    .program_name = currentProgramName,
                    .clip_origin = state->var.clip_origin,
                    .clip_depth_mode = state->var.clip_depth_mode,
                    .tess_flags = mglPipelineCachePackTessFlags(
                        _tessellation.nativeTESActive,
                        _tessellation.tessVertexCaptureActive,
                        _geometry.expansionActive,
                        _tessellation.cullDistanceCaptureActive,
                        _tessellation.tessComputeActive),
                    .vertex_instance_id = currentVertexProgram
                        ? currentVertexProgram->pipeline_cache_instance_id : 0u,
                    .vertex_generation = currentVertexProgram
                        ? currentVertexProgram->pipeline_cache_generation : 0u,
                    .fragment_instance_id = currentFragmentProgram
                        ? currentFragmentProgram->pipeline_cache_instance_id : 0u,
                    .fragment_generation = currentFragmentProgram
                        ? currentFragmentProgram->pipeline_cache_generation : 0u,
                    .pipeline_sig = pipelineSig,
                    .vertex_sig = vertexSig,
                };
                mglPipelineCacheBuildLookupKeyWords(&keyInputs, keyWords);
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
                BOOL cachedFunctionMetadataPresent = mglPipelineCacheLookupPipeline(
                    &_pipelineCacheState, &_pipelineCacheOwner,
                    (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                    keyWords, (void **)&cachedPipeline,
                    (void **)&cachedVertexFunction,
                    (void **)&cachedFragmentFunction);
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

                    mglPipelineCacheActivatePipelineState(
                        &_pipelineCacheState, &_pipelineCacheOwner,
                        (__bridge void *)_device,
                        _pipelineCacheBinaryArchiveRequested,
                        (__bridge void *)cachedPipeline, builtColor0Format,
                        builtDepthFormat, builtStencilFormat, currentProgramName,
                        (__bridge void *)(cachedFunctionMetadataPresent
                                              ? cachedVertexFunction
                                              : psoVertexFunction),
                        (__bridge void *)(cachedFunctionMetadataPresent
                                              ? cachedFragmentFunction
                                              : psoFragmentFunction));
                    pipelineResolvedFromCache = true;
                    /* Hit path deliberately skips the LRU touch: touching
                     * would require copying the query-keyed object that must
                     * never enter the LRU (see pipelineQueryKeyForWords:),
                     * reintroducing the per-draw alloc this avoids.  Mirrors
                     * the depth-stencil cache policy. */

                    mglPipelineRecoveryOnCacheHit(
                        &recoveryView, (uint32_t)currentProgramName,
                        (uint32_t)MGLPixelFormatInvalid);
                    mglPipelineRecoveryApplyToGPU(&_gpuRecovery, &recoveryView);
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

                if (deferredBufferMapForPipelineBuild && _pipelineCacheState.pipelineState != nil) {
                    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
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
        if (mglPipelineCachePipelineDescriptorStateForWords(
                &_pipelineCacheState, &_pipelineCacheOwner,
                (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                pipelineCacheKeyWords, &cachedState)) {
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
    id previousPipelineState = (__bridge id)_pipelineCacheState.pipelineState;
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
        if (mglPipelineCacheCreateRenderPipelineFromState(
                &_pipelineCacheState, &_pipelineCacheOwner,
                (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                &finalState, (__bridge void *)vertexFunction,
                fragmentFunction ? (__bridge void *)fragmentFunction : NULL,
                &psoPtr, cppError, sizeof(cppError)) != 0 || !psoPtr) {
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
                mglWriteProgramMSLDump(currentVertexProgram, errDesc);
                if (currentFragmentProgram && currentFragmentProgram != currentVertexProgram) {
                    mglWriteProgramMSLDump(currentFragmentProgram, errDesc);
                } else if (!currentVertexProgram) {
                    mglWriteProgramMSLDump(currentProgram, errDesc);
                }
                BOOL sameProgram =
                (_pipelineCacheState.pipelineProgramName != 0 &&
                _pipelineCacheState.pipelineProgramName == currentProgramName &&
                _pipelineCacheState.pipelineVertexFunction == (__bridge void *)vertexFunction &&
                _pipelineCacheState.pipelineFragmentFunction == (__bridge void *)fragmentFunction);
                BOOL colorCompatible = (_pipelineCacheState.pipelineColor0Format == MGLPixelFormatInvalid ||
                builtColor0Format == (uint32_t)MGLPixelFormatInvalid ||
                (uint32_t)_pipelineCacheState.pipelineColor0Format == builtColor0Format);
                BOOL depthCompatible = (_pipelineCacheState.pipelineDepthFormat == MGLPixelFormatInvalid ||
                builtDepthFormat == (uint32_t)MGLPixelFormatInvalid ||
                (uint32_t)_pipelineCacheState.pipelineDepthFormat == builtDepthFormat);
                BOOL stencilCompatible = (_pipelineCacheState.pipelineStencilFormat == MGLPixelFormatInvalid ||
                builtStencilFormat == (uint32_t)MGLPixelFormatInvalid ||
                (uint32_t)_pipelineCacheState.pipelineStencilFormat == builtStencilFormat);

                if (previousPipelineState && sameProgram && colorCompatible && depthCompatible && stencilCompatible) {
                    NSLog(@"MGL WARNING: Interface mismatch for program %u; reusing previous compatible pipeline once",
                    (unsigned)currentProgramName);
                    compiledPSO = previousPipelineState;
                    pipelineReusedPrevious = true;
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

                    [self invalidateCurrentPipelineStateForReason:@"interface mismatch pipeline failure"];
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
                [self invalidateCurrentPipelineStateForReason:@"pipeline creation failure"];

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
                        simpleState.color_write_mask[i] = 0;
                        simpleState.source_rgb_blend_factor[i] = 0;
                        simpleState.destination_rgb_blend_factor[i] = 0;
                        simpleState.source_alpha_blend_factor[i] = 0;
                        simpleState.destination_alpha_blend_factor[i] = 0;
                        simpleState.rgb_blend_operation[i] = 0;
                        simpleState.alpha_blend_operation[i] = 0;
                        if (i > 0) {
                            simpleState.color_format[i] = (uint32_t)MGLPixelFormatInvalid;
                        }
                    }
                    psoPtr = NULL;
                    cppError[0] = '\0';
                    if (mglPipelineCacheCreateRenderPipelineFromState(
                            &_pipelineCacheState, &_pipelineCacheOwner,
                            (__bridge void *)_device,
                            _pipelineCacheBinaryArchiveRequested, &simpleState,
                            (__bridge void *)vertexFunction,
                            fragmentFunction ? (__bridge void *)fragmentFunction
                                             : NULL,
                            &psoPtr, cppError, sizeof(cppError)) == 0 &&
                        psoPtr) {
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
            [self invalidateCurrentPipelineStateForReason:@"pipeline creation exception"];
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
            if (&_commandState && mglRenderPassColorTextureFor(&_commandState, 0)) {
                safeColor0Format = mglRenderPassTextureInfo(
                    mglRenderPassColorTextureFor(&_commandState, 0)).pixel_format;
            } else if (_drawable && [self mglDrawableTexture]) {
                safeColor0Format = mglRenderPassTextureInfo([self mglDrawableTexture]).pixel_format;
            }
            if (safeColor0Format == MGLPixelFormatInvalid) {
                safeColor0Format = MGLPixelFormatBGRA8Unorm;
            }
            safeState.color_format[0] = (uint32_t)safeColor0Format;
            safeState.depth_format = finalState.depth_format;
            safeState.stencil_format = finalState.stencil_format;

            // Precompiled safe shaders (mgl_aux_assets table).  No runtime
            // source compilation; failures log the program/format context and
            // never retry with the source compiler.
            const MGLAuxShaderAsset *safe =
                mglAuxShaderAssetFind("safe_fallback");
            void *safeVS = NULL;
            void *safeFS = NULL;
            char libError[512] = {0};
            if (!safe || !safe->data || safe->size == 0 ||
                mglRenderCreateAuxFunctions(
                    safe->data, safe->size, safe->hash,
                    "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
                    &safeVS, &safeFS,
                    libError, sizeof(libError)) != 0 || !safeVS) {
                NSLog(@"MGL CRITICAL: safe fallback asset unavailable "
                      @"program=%u color0=%lu hash=0x%016llx error=%s",
                      (unsigned)currentProgramName,
                      (unsigned long)safeColor0Format,
                      safe ? (unsigned long long)safe->hash : 0ull,
                      libError[0] ? libError : "asset missing");
            } else {
                id safeVSFunction =
                    (__bridge_transfer id)safeVS;
                id safeFSFunction =
                    safeFS ? (__bridge_transfer id)safeFS : nil;
                psoPtr = NULL;
                cppError[0] = '\0';
                if (mglPipelineCacheCreateRenderPipelineFromState(
                        &_pipelineCacheState, &_pipelineCacheOwner,
                        (__bridge void *)_device,
                        _pipelineCacheBinaryArchiveRequested, &safeState,
                        (__bridge void *)safeVSFunction,
                        safeFSFunction ? (__bridge void *)safeFSFunction : NULL,
                        &psoPtr, cppError, sizeof(cppError)) == 0 && psoPtr) {
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
            [self invalidateCurrentPipelineStateForReason:@"all pipeline fallbacks failed"];
            _gpuRecovery.pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return false;
        }
    }

    if (!compiledPSO) {
        NSLog(@"MGL ERROR: Failed to create pipeline state: %s", cppError[0] ? cppError : "unknown error");
        NSLog(@"MGL WARNING: Skipping draw for this pipeline build failure; will retry later");
        [self invalidateCurrentPipelineStateForReason:@"pipeline state is nil after creation"];
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
            _gpuRecovery.interfaceMismatchColor0Format = (uint32_t)MGLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchDepthFormat = (uint32_t)MGLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchStencilFormat = (uint32_t)MGLPixelFormatInvalid;
            _gpuRecovery.interfaceMismatchRetryAfter = 0.0;
            mglPipelineCacheActivatePipelineState(
                &_pipelineCacheState, &_pipelineCacheOwner,
                (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                (__bridge void *)compiledPSO, (uint32_t)builtColor0Format,
                (uint32_t)builtDepthFormat, (uint32_t)builtStencilFormat,
                currentProgramName, (__bridge void *)vertexFunction,
                (__bridge void *)fragmentFunction);
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
    if (pipelineCacheKeyWords && _pipelineCacheState.pipelineState) {
            (void)pipelineSig;
            (void)vertexSig;
            mglPipelineCacheStorePipeline(
                &_pipelineCacheState, &_pipelineCacheOwner,
                (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                _pipelineCacheState.pipelineState, (__bridge void *)vertexFunction,
                (__bridge void *)fragmentFunction, pipelineCacheKeyWords);

            if (!stateFromCache && state) {
                mglPipelineCacheStorePipelineDescriptorState(
                    &_pipelineCacheState, &_pipelineCacheOwner,
                    (__bridge void *)_device, _pipelineCacheBinaryArchiveRequested,
                    state, pipelineCacheKeyWords);
            }
    }
}

/* Bind spvBufferSizeConstants for runtime-sized SSBO arrays in vertex/fragment
 * stages.  The AIR backend emits code that reads uint32 byte-sizes from a
 * constant uint* buffer at MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a shader uses
 * .length() on unsized SSBO arrays.  The render encoder has separate buffer
 * tables for vertex and fragment, so we bind a size buffer for each stage
 * that needs it. */
- (bool) bindBufferSizeConstantsForRenderEncoder
{
    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) != 1) {
        return true;
    }

    Program *vertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (vertexProgram && vertexProgram->modules[_VERTEX_SHADER].needs_runtime_array_size_buffer)
    {
        uint32_t sizeConstants[31];
        memset(sizeConstants, 0, sizeof(sizeConstants));

        for (int i = 0; i < MGL_STATE(ctx)->vertex_buffer_map_list.count; i++)
        {
            BufferMap *map = &MGL_STATE(ctx)->vertex_buffer_map_list.buffers[i];
            if (!map->buf)
                continue;
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX)
                continue;
            GLsizeiptr visibleSize = mglBufferMapVisibleSize(map);
            sizeConstants[metalSlot] = (uint32_t)visibleSize;
        }

        id vertexSizeBuffer = (__bridge id)
            mglRendererBackendGetSizeConstantsBuffer(
                _backend, MGL_RENDERER_BACKEND_SIZE_CONSTANTS_VERTEX,
                sizeConstants, 31u);
        if (!vertexSizeBuffer) {
            vertexSizeBuffer = mglRenderPassCreateBufferWithBytes(
                _device, sizeConstants, sizeof(sizeConstants),
                MGLResourceStorageModeShared);
            if (vertexSizeBuffer &&
                mglRendererBackendSetSizeConstantsBuffer(
                    _backend, MGL_RENDERER_BACKEND_SIZE_CONSTANTS_VERTEX,
                    sizeConstants, 31u, (__bridge void *)vertexSizeBuffer) != 0) {
                vertexSizeBuffer = nil;
            }
        }
        if (vertexSizeBuffer) {
            mglRenderSetRenderBufferForOwner(
                _commandState.currentRenderEncoderOwner,
                (__bridge void *)vertexSizeBuffer, 0,
                MGL_RENDER_BINDING_STAGE_VERTEX,
                MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            [self recordLastBoundVertexBuffer:vertexSizeBuffer
                                       offset:0
                                      atIndex:MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX];
            MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        }
    }

    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    if (fragmentProgram && fragmentProgram->modules[_FRAGMENT_SHADER].needs_runtime_array_size_buffer)
    {
        uint32_t sizeConstants[31];
        memset(sizeConstants, 0, sizeof(sizeConstants));

        for (int i = 0; i < MGL_STATE(ctx)->fragment_buffer_map_list.count; i++)
        {
            BufferMap *map = &MGL_STATE(ctx)->fragment_buffer_map_list.buffers[i];
            if (!map->buf)
                continue;
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX)
                continue;
            GLsizeiptr visibleSize = mglBufferMapVisibleSize(map);
            sizeConstants[metalSlot] = (uint32_t)visibleSize;
        }

        id fragmentSizeBuffer = (__bridge id)
            mglRendererBackendGetSizeConstantsBuffer(
                _backend, MGL_RENDERER_BACKEND_SIZE_CONSTANTS_FRAGMENT,
                sizeConstants, 31u);
        if (!fragmentSizeBuffer) {
            fragmentSizeBuffer = mglRenderPassCreateBufferWithBytes(
                _device, sizeConstants, sizeof(sizeConstants),
                MGLResourceStorageModeShared);
            if (fragmentSizeBuffer &&
                mglRendererBackendSetSizeConstantsBuffer(
                    _backend, MGL_RENDERER_BACKEND_SIZE_CONSTANTS_FRAGMENT,
                    sizeConstants, 31u, (__bridge void *)fragmentSizeBuffer) != 0) {
                fragmentSizeBuffer = nil;
            }
        }
        if (fragmentSizeBuffer) {
            mglRenderSetRenderBufferForOwner(
                _commandState.currentRenderEncoderOwner,
                (__bridge void *)fragmentSizeBuffer, 0,
                MGL_RENDER_BINDING_STAGE_FRAGMENT,
                MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            [self recordLastBoundFragmentBuffer:fragmentSizeBuffer
                                         offset:0
                                        atIndex:MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX];
            MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
        }
    }

    return true;
}

-(void) flushCommandBuffer: (bool) finish
{
    METAL_LOCK();
    [self flushCommandBufferLocked:finish];
    METAL_UNLOCK();

    /* The C++ command owner retains the last accepted submission. Waiting
     * through its value-state API keeps completion lifetime out of the
     * renderer's ObjC ivar mirror and preserves the old outside-lock wait. */
    if (finish) {
        MGLRenderCommandBufferState finishState = {0};
        int waitResult = mglCmdWaitForLastSubmittedCommandBuffer(&_commandState, &finishState);
        if (waitResult < 0 || finishState.has_error) {
            NSLog(@"MGL ERROR: owner waitUntilCompleted failed status=%u domain=%s code=%lld",
                  finishState.status, finishState.error_domain,
                  (long long)finishState.error_code);
        }
    }
}

-(void) flushCommandBufferLocked: (bool) finish
{
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or queue is NULL in flushCommandBuffer");
        return;
    }

    [self flushDrawBufferLocked:ctx];

    if (![self processGLStateLocked: false]) {
        NSLog(@"MGL WARNING: processGLState failed in flushCommandBuffer, continuing with cleanup");
    }

    /* If processGLStateLocked: left a render encoder active, mark the CB as
     * having work so the commit below is not skipped. */
    if (mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) == 1) {
        _currentCBHasWork = YES;
    }

    [self endRenderEncodingLocked];

    /* Skip empty-CB commit when finish=true: wait on the owner's last submit
     * instead (Metal CBs execute serially on the same queue).  Any path
     * that encodes work (draws/render/blit/compute) into the current CB
     * MUST set _currentCBHasWork before calling flushCommandBuffer:YES,
     * else the skip drops uncommitted work. */
    if (finish && !_currentCBHasWork &&
        mglCmdHasLastSubmittedCommandBuffer(&_commandState)) {
        return;
    }
    if (finish && !_currentCBHasWork &&
        !mglCmdHasLastSubmittedCommandBuffer(&_commandState)) {

        return;
    }

    if (![self ensureWritableCommandBufferLocked:"flushCommandBuffer"]) {
        NSLog(@"MGL ERROR: Unable to obtain writable command buffer in flushCommandBuffer");
        return;
    }

    MGLRenderCommandBufferState currentState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            _commandState.currentCommandBufferOwner,
            &currentState)) {
        NSLog(@"MGL WARNING: No current command buffer in flushCommandBuffer");
        return;
    }

    uint32_t currentStatus =
        (uint32_t)currentState.status;
    if (currentStatus != MGLCommandBufferStatusNotEnqueued) {
        NSLog(@"MGL INFO: flushCommandBuffer found finalized buffer (status=%ld), rotating", (long)currentStatus);
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer in flushCommandBuffer");
        }
        return;
    }

    MGLRenderCommandBufferState preCommitState = currentState;
    if (preCommitState.has_error) {
        NSLog(@"MGL ERROR: Command buffer has error before commit: %s",
              mglRenderCommandBufferErrorDescription(&preCommitState));
        [self cleanupCommandBuffer];
        return;
    }

    if (![self validateMetalObjects]) {
        NSLog(@"MGL WARNING: GPU throttling active - skipping command buffer commit");
        [self cleanupCommandBuffer];
        return;
    }

    id commandBufferToCommit =
        (__bridge id)mglCmdDetachCurrentCommandBufferForSubmission(&_commandState);

    @try {
        [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
        /* The owner now retains the last submit; flushCommandBuffer waits on
         * that state after releasing METAL_LOCK. */
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Command buffer commit failed in flushCommandBuffer: %@", exception);
        [self recordGPUError];
        [self cleanupCommandBuffer];
    }

    if (!finish) {
        [self newCommandBufferLocked];
    }
}

- (bool)syncRenderPassStateForContext:(GLMContext)glm_ctx
{
    return mglSyncBridgeSyncFbo((__bridge void *)self, glm_ctx);
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
    [self endRenderEncodingLocked];
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
        if (MGL_STATE(glm_ctx)->error != GL_NO_ERROR)
            *replayError = MGL_STATE(glm_ctx)->error;
        return NO;
    }
    return YES;
}

bool mglRendererObjCSyncPipeline(GLMContext context, int deferred_buffer_map)
{
    MGLRenderer *renderer = mglRendererForContext(context);
    if (!renderer) {
        return false;
    }
    return [renderer syncPipelineStateWithDeferredBufferMap:deferred_buffer_map != 0];
}

bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command)
{
    MGLRenderer *renderer = mglRendererForContext(context);
    if (!renderer) {
        return false;
    }
    return [renderer processGLState:draw_command];
}

@end
