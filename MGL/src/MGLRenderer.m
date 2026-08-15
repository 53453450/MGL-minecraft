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
 * MGLRenderer.m
 * MGL
 *
 */

/* MGLRenderer_Private.h transitively imports Foundation, Metal, simd, os/lock.h,
 * glm_context.h, pixel_utils.h, and all mgl_* compatibility
 * headers listed below.  Only imports unique to this TU are listed here. */
#import <objc/runtime.h>
#import <MetalKit/MetalKit.h>

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <string.h>
#include "mgl_render_cpp_objc.h"
#include <strings.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <libgen.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include <dispatch/dispatch.h>

#import "MGLRenderer_Private.h"
#import "mgl.h"
#import "mgl_sampler_compat.h"
#import "mgl_state_log.h"
#import "mgl_trace_log.h"
#import "mgl_byte_hash.h"
#import "mgl_compute_pipeline_cache.h"
#import "mgl_metal_bridge.h"

#define TRACE_FUNCTION()    DEBUG_PRINT("%s\n", __FUNCTION__);

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

static void mglRecordFrameCommandBufferCompleted(
    void *context,
    const MGLRenderCppCommandBufferState *state)
{
    (void)state;
    mglRecordFrameCompleted((uint64_t)(uintptr_t)context);
}

/* MGLFragmentTextureTraceBinding typedef moved to mgl_trace_strategy.h. */

/* Draw buffer mapping helpers moved to mgl_draw_buffer.h/.m. */

/* Pixel readback helpers (7 functions) moved to mgl_readback.m */
/* Layer pixel format / sRGB / linear helpers moved to mgl_texture_compat */

static id<MTLTexture> mglRendererCreateTextureView(
    id<MTLTexture> texture,
    MTLPixelFormat pixelFormat)
{
    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() != NULL) {
        void *view = NULL;
        if (mglRenderCppCreateTextureView(
                (__bridge void *)texture, (uint32_t)pixelFormat,
                &view) == 0 && view) {
            return (__bridge_transfer id<MTLTexture>)view;
        }
    }
    return [texture newTextureViewWithPixelFormat:pixelFormat];
}

// Applies GL_FRAMEBUFFER_SRGB state to a render-target texture by creating
// a Metal texture view with the appropriate pixel format. The view shares
// the same underlying storage so no memory copy occurs.
// Returns the (possibly wrapped) texture that should be used as the render target.
id<MTLTexture> mglApplySRGBStateToRenderTarget(id<MTLTexture> texture, GLMContext ctx)
{
    if (!texture || !ctx) return texture;

    MTLPixelFormat currentFmt = texture.pixelFormat;
    MTLPixelFormat desiredFmt;

    if (ctx->active_state->caps.framebuffer_srgb) {
        // GL_FRAMEBUFFER_SRGB enabled: shader writes linear, GPU should encode to sRGB
        desiredFmt = mglSRGBPixelFormat(currentFmt);
    } else {
        // GL_FRAMEBUFFER_SRGB disabled: shader writes final values, no encoding
        desiredFmt = mglLinearPixelFormat(currentFmt);
    }

    if (desiredFmt == currentFmt) {
        return texture;  // Already the correct format
    }

    id<MTLTexture> view =
        mglRendererCreateTextureView(texture, desiredFmt);
    if (view) {
        return view;
    }

    // Texture view creation can fail if formats are incompatible;
    // fall back to the original texture.
    static uint64_t s_srgbViewFailCount = 0;
    if (++s_srgbViewFailCount <= 8) {
        NSLog(@"MGL WARNING: newTextureViewWithPixelFormat failed current=%lu desired=%lu srgb=%d",
              (unsigned long)currentFmt, (unsigned long)desiredFmt,
              ctx->active_state->caps.framebuffer_srgb ? 1 : 0);
    }
    return texture;
}

/* mglMetalCopyTextureBytesToBGRA8 moved to mgl_readback.m */
void mglMetalCopyRows(const uint8_t *src,
                      NSUInteger srcBytesPerRow,
                      uint8_t *dst,
                      NSUInteger dstBytesPerRow,
                      NSUInteger rowBytes,
                      NSUInteger height,
                      BOOL flipY)
{
    if (!src || !dst || rowBytes == 0u || height == 0u) {
        return;
    }

    for (NSUInteger y = 0; y < height; y++) {
        const uint8_t *srcRow = src + (y * srcBytesPerRow);
        NSUInteger dstY = flipY ? (height - 1u - y) : y;
        uint8_t *dstRow = dst + (dstY * dstBytesPerRow);
        memcpy(dstRow, srcRow, rowBytes);
    }
}

/* MGLScaledBlitParams / MGLMSAAIntegerResolveParams / MGLClearRectParams
 * typedefs moved to MGLRenderer_Private.h. */
/* MGLBlitAxis struct + blit axis clipping helpers (mglClipBlitAxisToDestination,
 * mglClipBlitAxisToSource, mglClipBlitAxis) moved to mgl_blit_clip.h/.m. */

/* mglMetalTextureLevelDimension now lives in mgl_texture_compat.m — see
 * mgl_texture_compat.h. */

/* mglInitTraceLogIfNeeded / mglTraceLog / mglTraceLogExternal are now
 * declared in mgl_trace_log.h. */

__attribute__((constructor))
static void mglRendererDiagnosticBuildMarker(void)
{
    mglInitTraceLogIfNeeded();
    mglTraceLog("MGL DIAG BUILD marker=gui-rt-cull-v8-20260608 built=%s %s renderer-loaded",
                __DATE__,
                __TIME__);
}


NSRange mglRendererFindMSLEntryParameterClose(NSString *msl, const char *entryPoint)
{
    if (!msl || !entryPoint || entryPoint[0] == '\0') {
        return NSMakeRange(NSNotFound, 0);
    }

    NSString *entryName = [NSString stringWithUTF8String:entryPoint];
    if (!entryName) {
        return NSMakeRange(NSNotFound, 0);
    }

    NSString *needle = [entryName stringByAppendingString:@"("];
    NSRange entryRange = [msl rangeOfString:needle];
    if (entryRange.location == NSNotFound) {
        return NSMakeRange(NSNotFound, 0);
    }

    NSUInteger openParen = entryRange.location + entryRange.length - 1u;
    NSUInteger length = [msl length];
    NSInteger depth = 0;
    for (NSUInteger idx = openParen; idx < length; idx++) {
        unichar ch = [msl characterAtIndex:idx];
        if (ch == '(') {
            depth++;
        } else if (ch == ')') {
            depth--;
            if (depth == 0) {
                return NSMakeRange(idx, 1);
            }
        }
    }

    return NSMakeRange(NSNotFound, 0);
}

// Debug switch: temporarily disable shared-event synchronization path to isolate GPU timeout sources.
// kMGLDisableSharedEventSync moved to MGLRenderer_Private.h
// Leave verbose bind tracing off by default; per-draw logging can stall the render thread.
/* kMGLVerboseBindLogs moved to MGLRenderer_Private.h */
// Pipeline/descriptor tracing is similarly noisy; keep it opt-in.
/* kMGLVerbosePipelineLogs moved to MGLRenderer_Private.h */
// Frame-loop/state tracing is extremely hot; keep broad tracing off so the log
// reaches the actual crash site instead of Prism's 100k-line cap.
// kMGLVerboseFrameLoopLogs moved to MGLRenderer_Private.h
// kMGLDisableSharedEventSync moved to MGLRenderer_Private.h
// kMGLDiagnosticStateLogs moved to MGLRenderer_Private.h
// kMGLSwapPresentDiagnostics moved to MGLRenderer_Private.h
// kMGLDrawSubmitDiagnostics moved to MGLRenderer_Private.h
// kMGLSynchronizeTextureUploads moved to MGLRenderer_Private.h
// kMGLTextureUploadWaitTimeoutSeconds moved to MGLRenderer_Private.h
// kMGLUseDedicatedTextureUploadCommandBuffer moved to MGLRenderer_Private.h
// Keep vertex attribute buffers in a dedicated high slot range so they do not collide
// with UBO/SSBO bindings that are expected at low indices.
// NOTE: This is the Metal buffer index where vertex attrib buffers start, NOT the
// GL binding count.  Metal only has 31 vertex buffer slots (0..30), so this must
// stay below 31 regardless of MAX_BINDABLE_BUFFERS (which tracks GL state only).
// kMGLVertexAttribBufferBase = 16, kMGLMaxMetalVertexBufferCount = 31,
// kMGLMaxMetalVertexBufferIndex = 30 come from mgl_buffer_slots.h.
//
// Slot indices for point-size params and TCS stage-in have different names
// in the renderer than in the header; #define bridges them.  Identically-named
// constants (FragCoordParams, CullDistance*) resolve to the header's enum
// values automatically.
/* kMGLPointSizeParamBufferIndex, kMGLTCSStageInReplBufferIndex moved to MGLRenderer_Private.h */
/* kMGLFragCoordParamsMSLName moved to MGLRenderer_Private.h */
// Metal validation requires bound stage buffers to satisfy argument byte length.
// Keep a conservative minimum for low-index base/resource slots.
/* kMGLMinimumStageBindingSize, kMGLDefaultStageFallbackBufferSize, kMGLStageBindingStackScratchSize moved to MGLRenderer_Private.h */
// Keep low-index vertex resource slots bound during diagnostics. Attribute VBOs
// live at kMGLVertexAttribBufferBase+, so this does not overwrite vertex input slots.
/* kMGLEnableVertexAllSlotFallback, kMGLEnableSampledTextureFallback moved to MGLRenderer_Private.h */
// Mirror Metal's drawArrays vertex-buffer range validation before calling into
// the debug layer. Metal aborts the process for these errors; we want a log and
// a skipped draw instead.
/* kMGLValidateDrawArraysVboRange, kMGLValidateDrawElementsVboRange moved to MGLRenderer_Private.h */

/* Env var names are always string literals (stable addresses), so we cache by
 * pointer.  s_mglEnvCache is only ever read/written on the GL calling thread,
 * never from the Metal completion-handler thread or the main queue, so no
 * locking is needed here.  The worst race case would be a few extra getenv
 * calls during startup before the cache fills. */
#define MGL_ENV_CACHE_CAPACITY 32
static struct {
    const char *name;   /* string literal address — key */
    BOOL value;
    BOOL default_on;    /* distinguishes mglEnvFlagEnabled vs DefaultOn */
    BOOL valid;
} s_mglEnvCache[MGL_ENV_CACHE_CAPACITY];

#include "mgl_env_flag.h"

static BOOL mglEnvFlagEnabledCached(const char *name, BOOL default_on)
{
    if (!name) {
        return default_on;
    }

    /* Cache lookup by pointer (string literals have stable addresses). */
    for (int i = 0; i < MGL_ENV_CACHE_CAPACITY; i++) {
        if (s_mglEnvCache[i].valid &&
            s_mglEnvCache[i].name == name &&
            s_mglEnvCache[i].default_on == default_on) {
            return s_mglEnvCache[i].value;
        }
    }

    /* Cache miss: compute.  Truthiness (0/false/no/off) is delegated to the
     * single-source parser in mgl_env_flag.h; only the "unset => default_on"
     * semantics are applied here. */
    const char *value = getenv(name);
    BOOL result;
    if (!value || value[0] == '\0') {
        result = default_on;
    } else {
        result = mgl_env_flag_enabled(name) ? YES : NO;
    }

    /* Store in cache (find first empty slot). */
    for (int i = 0; i < MGL_ENV_CACHE_CAPACITY; i++) {
        if (!s_mglEnvCache[i].valid) {
            s_mglEnvCache[i].name = name;
            s_mglEnvCache[i].value = result;
            s_mglEnvCache[i].default_on = default_on;
            s_mglEnvCache[i].valid = YES;
            break;
        }
    }

    return result;
}

BOOL mglEnvFlagEnabled(const char *name)
{
    return mglEnvFlagEnabledCached(name, NO);
}

/* Unset/empty → YES (default ON); "0"/"false"/"no"/"off" → NO; other non-empty → YES.
 * Use for kill-switchable optimizations that should ship enabled. */
BOOL mglEnvFlagEnabledDefaultOn(const char *name)
{
    return mglEnvFlagEnabledCached(name, YES);
}

static BOOL mglRendererUsesMetalCpp(void)
{
    return mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP") &&
           mglRenderCppGetDevice() != NULL;
}

static id<MTLBuffer> mglRendererCreateBuffer(id<MTLDevice> device,
                                             NSUInteger length,
                                             MTLResourceOptions options)
{
    if (mglRendererUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
            buffer) {
            return (__bridge_transfer id<MTLBuffer>)buffer;
        }
    }
    return [device newBufferWithLength:length options:options];
}

static id<MTLTexture> mglRendererCreateTexture(
    id<MTLDevice> device,
    MTLTextureDescriptor *descriptor)
{
    if (mglRendererUsesMetalCpp()) {
        void *texture = NULL;
        MGLRenderCppTextureDescriptorState state =
            mglRenderCppTextureDescriptorStateFromObjC(descriptor);
        if (mglRenderCppCreateTextureFromState(&state, NULL, &texture) == 0 &&
            texture) {
            return (__bridge_transfer id<MTLTexture>)texture;
        }
    }
    return [device newTextureWithDescriptor:descriptor];
}

static id<MTLRenderCommandEncoder> mglRendererCreateRenderEncoder(
    id<MTLCommandBuffer> commandBuffer,
    MTLRenderPassDescriptor *descriptor,
    const MGLRenderCppRenderPassState *state)
{
    if (mglRendererUsesMetalCpp() && state) {
        void *encoder = NULL;
        if (mglRenderCppCreateRenderEncoderFromState(
                (__bridge void *)commandBuffer, state, &encoder) == 0 &&
            encoder) {
            return (__bridge id<MTLRenderCommandEncoder>)encoder;
        }
    }
    return [commandBuffer renderCommandEncoderWithDescriptor:descriptor];
}

static void mglRendererEndRenderEncoder(id<MTLRenderCommandEncoder> encoder)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppEndRenderEncoder((__bridge void *)encoder) == 0) {
        return;
    }
    [encoder endEncoding];
}

static void mglRendererSetRenderPipeline(id<MTLRenderCommandEncoder> encoder,
                                         id<MTLRenderPipelineState> pipeline)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppSetRenderPipelineState((__bridge void *)encoder,
                                           (__bridge void *)pipeline) == 0) {
        return;
    }
    [encoder setRenderPipelineState:pipeline];
}

static void mglRendererSetDepthStencil(id<MTLRenderCommandEncoder> encoder,
                                       id<MTLDepthStencilState> state)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppSetRenderDepthStencilState((__bridge void *)encoder,
                                               (__bridge void *)state) == 0) {
        return;
    }
    [encoder setDepthStencilState:state];
}

static void mglRendererSetRenderBytes(id<MTLRenderCommandEncoder> encoder,
                                      const void *bytes,
                                      NSUInteger length,
                                      uint32_t stage,
                                      NSUInteger index)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppSetRenderBytes((__bridge void *)encoder, bytes, length,
                                   stage, (uint32_t)index) == 0) {
        return;
    }
    if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
        [encoder setVertexBytes:bytes length:length atIndex:index];
    } else {
        [encoder setFragmentBytes:bytes length:length atIndex:index];
    }
}

static void mglRendererSetViewport(id<MTLRenderCommandEncoder> encoder,
                                   MTLViewport viewport)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppSetRenderViewport(
            (__bridge void *)encoder, viewport.originX, viewport.originY,
            viewport.width, viewport.height, viewport.znear,
            viewport.zfar) == 0) {
        return;
    }
    [encoder setViewport:viewport];
}

static void mglRendererSetScissor(id<MTLRenderCommandEncoder> encoder,
                                  MTLScissorRect scissor)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppSetRenderScissor((__bridge void *)encoder, scissor.x,
                                     scissor.y, scissor.width,
                                     scissor.height) == 0) {
        return;
    }
    [encoder setScissorRect:scissor];
}

static void mglRendererDrawPrimitives(id<MTLRenderCommandEncoder> encoder,
                                      MTLPrimitiveType primitiveType,
                                      NSUInteger vertexStart,
                                      NSUInteger vertexCount)
{
    /* P4.3a: 统一 draw plan 提交（gate-on 走 C++ EncodeDraw）。 */
    if (mglRenderCppTryEncodeDraw(encoder, &(MGLRenderCppDrawPlan){
            .kind = MGL_RENDER_CPP_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = 1u,
            .base_instance = 0u,
        })) {
        return;
    }
    [encoder drawPrimitives:primitiveType vertexStart:vertexStart
                vertexCount:vertexCount];
}

static id<MTLBlitCommandEncoder> mglRendererCreateBlitEncoder(
    id<MTLCommandBuffer> commandBuffer)
{
    if (mglRendererUsesMetalCpp()) {
        void *encoder = NULL;
        if (mglRenderCppCreateBlitEncoder((__bridge void *)commandBuffer,
                                          &encoder) == 0 && encoder) {
            return (__bridge id<MTLBlitCommandEncoder>)encoder;
        }
    }
    return [commandBuffer blitCommandEncoder];
}

static void mglRendererBlitCopyBuffer(id<MTLBlitCommandEncoder> encoder,
                                      id<MTLBuffer> source,
                                      NSUInteger sourceOffset,
                                      id<MTLBuffer> destination,
                                      NSUInteger destinationOffset,
                                      NSUInteger size)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppBlitCopyBuffer(
            (__bridge void *)encoder, (__bridge void *)source, sourceOffset,
            (__bridge void *)destination, destinationOffset, size) == 0) {
        return;
    }
    [encoder copyFromBuffer:source sourceOffset:sourceOffset
                   toBuffer:destination destinationOffset:destinationOffset
                       size:size];
}

static void mglRendererEndBlitEncoder(id<MTLBlitCommandEncoder> encoder)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppEndBlitEncoder((__bridge void *)encoder) == 0) {
        return;
    }
    [encoder endEncoding];
}

static void mglRendererWaitCommandBuffer(id<MTLCommandBuffer> commandBuffer)
{
    if (mglRendererUsesMetalCpp() &&
        mglRenderCppWaitCommandBuffer((__bridge void *)commandBuffer) == 0) {
        return;
    }
    [commandBuffer waitUntilCompleted];
}

/* Trace log core infrastructure (3 static globals, mglInitTraceLogIfNeeded,
 * mglTraceLogIsEnabled, mglTraceLogV, mglTraceLog, mglTraceLogExternal,
 * mglTraceLogNSString) moved to mgl_trace_log.h/.m. */

/* mglTraceRTYFlipDiagnosticsEnabled moved to MGLRenderer_Private.h */
/* mglYFlipDecisionName moved to MGLRenderer_Private.h */

/* Fragment texture trace binding helpers moved to mgl_trace_strategy.h/.m. */

/* Frame activity breadcrumbs (19 volatile globals + MGLSwapDrawCounters
 * struct + mglSnapshotSwapDrawCounters/mglResetSwapDrawCounters inline
 * helpers) moved to mgl_frame_activity.h/.m. */

/* mglRendererPointerInHashTable, mglRendererSafeFramebufferName, and
 * mglRendererGetValidatedFramebuffer declared in MGLRenderer_Private.h */
static inline BOOL mglRendererContextLikelyValid(GLMContext ctx)
{
    return (ctx != NULL) && ((uintptr_t)ctx >= 0x10000u);
}

Program *mglResolveProgramFromState(GLMContext ctx)
{
    if (!mglRendererContextLikelyValid(ctx)) {
        return NULL;
    }

    /*
     * glUseProgram(0) means there is no monolithic current program. In that
     * state separable pipelines, if any, are resolved per stage below; never
     * resurrect a stale cached program pointer as GL_CURRENT_PROGRAM.
     */
    if (ctx->active_state->program_name == 0) {
        ctx->active_state->program = NULL;
        return NULL;
    }

    Program *program = ctx->active_state->program;
    if (program) {
        GLuint expectedName = ctx->active_state->program_name ? ctx->active_state->program_name : program->name;
        if (!mglProgramPointerUsableForName(ctx, program, expectedName)) {
            NSLog(@"MGL PROGRAM RESOLVE invalid cached pointer=%p name=%u",
                  program,
                  (unsigned)ctx->active_state->program_name);
            ctx->active_state->program = NULL;
            program = NULL;
        }
    }

    if (program) {
        if (ctx->active_state->program_name == 0 || ctx->active_state->program_name != program->name) {
            ctx->active_state->program_name = program->name;
        }
        return program;
    }

    if (ctx->active_state->program_name == 0) {
        return NULL;
    }

    Program *resolved = (Program *)searchHashTable(&ctx->active_state->program_table, ctx->active_state->program_name);
    if (!resolved) {
        NSLog(@"MGL PROGRAM RESOLVE fail: name=%u missing in table", (unsigned)ctx->active_state->program_name);
        ctx->active_state->program_name = 0;
        return NULL;
    }

    if (!resolved->link_success) {
        NSLog(@"MGL PROGRAM RESOLVE pending: name=%u ptr=%p not linked",
              (unsigned)ctx->active_state->program_name, resolved);
        return NULL;
    }

    ctx->active_state->program = resolved;
    resolved->refcount++;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);

    NSLog(@"MGL PROGRAM RESOLVE recovered name=%u ptr=%p",
          (unsigned)ctx->active_state->program_name, resolved);
    return resolved;
}

static ProgramPipeline *mglResolveProgramPipelineFromState(GLMContext ctx)
{
    if (!mglRendererContextLikelyValid(ctx)) {
        return NULL;
    }

    ProgramPipeline *pipeline = ctx->active_state->program_pipeline;
    if (pipeline) {
        if (!mglRendererObjectPointerLikelyValid(pipeline) ||
            !mglRendererPointerInHashTable(&ctx->active_state->program_pipeline_table, pipeline) ||
            !mglPointerRangeIsReadable(pipeline, sizeof(*pipeline))) {
            NSLog(@"MGL PROGRAM PIPELINE RESOLVE invalid cached pointer=%p binding=%u",
                  pipeline,
                  (unsigned)ctx->active_state->var.program_pipeline_binding);
            ctx->active_state->program_pipeline = NULL;
            pipeline = NULL;
        } else {
            if (ctx->active_state->var.program_pipeline_binding == 0 ||
                ctx->active_state->var.program_pipeline_binding != pipeline->name) {
                ctx->active_state->var.program_pipeline_binding = pipeline->name;
            }
            return pipeline;
        }
    }

    GLuint pipelineName = ctx->active_state->var.program_pipeline_binding;
    if (pipelineName == 0) {
        return NULL;
    }

    ProgramPipeline *resolved =
        (ProgramPipeline *)searchHashTable(&ctx->active_state->program_pipeline_table, pipelineName);
    if (!resolved ||
        !mglRendererObjectPointerLikelyValid(resolved) ||
        !mglPointerRangeIsReadable(resolved, sizeof(*resolved))) {
        NSLog(@"MGL PROGRAM PIPELINE RESOLVE fail: name=%u missing/invalid",
              (unsigned)pipelineName);
        ctx->active_state->program_pipeline = NULL;
        ctx->active_state->var.program_pipeline_binding = 0;
        return NULL;
    }

    ctx->active_state->program_pipeline = resolved;
    return resolved;
}

static Program *mglRestoreMonolithicProgramBinding(GLMContext ctx, GLuint programName)
{
    if (!mglRendererContextLikelyValid(ctx)) {
        return NULL;
    }

    if (programName == 0u) {
        ctx->active_state->program = NULL;
        ctx->active_state->program_name = 0u;
        return NULL;
    }

    Program *program = ctx->active_state->program;
    if (!mglProgramPointerUsableForName(ctx, program, programName)) {
        program = (Program *)searchHashTable(&ctx->active_state->program_table, programName);
    }
    if (!program ||
        !mglProgramPointerUsableForName(ctx, program, programName)) {
        NSLog(@"MGL PROGRAM RESTORE missing/invalid program=%u", (unsigned)programName);
        program = NULL;
    }

    ctx->active_state->program = program;
    ctx->active_state->program_name = programName;
    return program;
}

static ProgramPipeline *mglRestoreProgramPipelineBinding(GLMContext ctx, GLuint pipelineName)
{
    if (!mglRendererContextLikelyValid(ctx)) {
        return NULL;
    }

    if (pipelineName == 0u) {
        ctx->active_state->program_pipeline = NULL;
        ctx->active_state->var.program_pipeline_binding = 0u;
        return NULL;
    }

    ProgramPipeline *pipeline =
        (ProgramPipeline *)searchHashTable(&ctx->active_state->program_pipeline_table, pipelineName);
    if (!pipeline ||
        !mglRendererObjectPointerLikelyValid(pipeline) ||
        !mglPointerRangeIsReadable(pipeline, sizeof(*pipeline))) {
        NSLog(@"MGL PROGRAM PIPELINE RESTORE missing/invalid pipeline=%u",
              (unsigned)pipelineName);
        pipeline = NULL;
    }

    ctx->active_state->program_pipeline = pipeline;
    ctx->active_state->var.program_pipeline_binding = pipelineName;
    return pipeline;
}

void mglRestoreProgramPipelinePair(GLMContext ctx, GLuint programName, GLuint pipelineName)
{
    if (!mglRendererContextLikelyValid(ctx)) {
        return;
    }

    (void)mglRestoreMonolithicProgramBinding(ctx, programName);
    (void)mglRestoreProgramPipelineBinding(ctx, pipelineName);
}

Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage)
{
    if (!mglRendererContextLikelyValid(ctx) || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return NULL;
    }

    Program *program = mglResolveProgramFromState(ctx);
    if (program) {
        return program;
    }

    /*
     * Separable program pipelines are only active when GL_CURRENT_PROGRAM is 0.
     * Keep glUseProgram semantics authoritative and only fall back to the
     * per-stage pipeline table for true pipeline draws.
     */
    if (ctx->active_state->program_name != 0) {
        return NULL;
    }

    ProgramPipeline *pipeline = mglResolveProgramPipelineFromState(ctx);
    if (!pipeline) {
        return NULL;
    }

    Program *stageProgram = pipeline->stage_programs[stage];
    if (!stageProgram) {
        return NULL;
    }

    if (!mglRendererObjectPointerLikelyValid(stageProgram) ||
        !mglPointerRangeIsReadable(stageProgram, sizeof(*stageProgram)) ||
        !mglProgramPointerUsableForName(ctx, stageProgram, stageProgram->name)) {
        NSLog(@"MGL PROGRAM PIPELINE RESOLVE invalid stage program pipeline=%u stage=%s ptr=%p",
              (unsigned)pipeline->name,
              mglShaderStageName(stage),
              stageProgram);
        /* Drop the dangling slot reference (retain was taken in
         * mglUseProgramStages) to avoid leaking the program object. */
        pipeline->stage_programs[stage] = NULL;
        mglReleaseProgramReference(ctx, stageProgram);
        return NULL;
    }

    if (!stageProgram->link_success) {
        NSLog(@"MGL PROGRAM PIPELINE RESOLVE pending stage program pipeline=%u stage=%s program=%u",
              (unsigned)pipeline->name,
              mglShaderStageName(stage),
              (unsigned)stageProgram->name);
        return NULL;
    }

    return stageProgram;
}

void mglRendererSyncFramebufferBindingNames(GLMContext ctx)
{
    if (!ctx) {
        return;
    }

    ctx->active_state->var.draw_framebuffer_binding =
        ctx->active_state->framebuffer ? ctx->active_state->framebuffer->name : 0u;
    ctx->active_state->var.read_framebuffer_binding =
        ctx->active_state->readbuffer ? ctx->active_state->readbuffer->name : 0u;
}

GLuint mglCurrentRenderProgramKey(GLMContext ctx)
{
    Program *program = mglResolveProgramFromState(ctx);
    if (program) {
        return program->name;
    }

    if (!mglRendererContextLikelyValid(ctx) ||
        ctx->active_state->program_name != 0) {
        return ctx ? ctx->active_state->program_name : 0u;
    }

    ProgramPipeline *pipeline = mglResolveProgramPipelineFromState(ctx);
    if (!pipeline) {
        return 0u;
    }

    GLuint vsName = pipeline->stage_programs[_VERTEX_SHADER]
        ? pipeline->stage_programs[_VERTEX_SHADER]->name
        : 0u;
    GLuint fsName = pipeline->stage_programs[_FRAGMENT_SHADER]
        ? pipeline->stage_programs[_FRAGMENT_SHADER]->name
        : 0u;
    uint32_t hash = 2166136261u;
    hash = (hash ^ pipeline->name) * 16777619u;
    hash = (hash ^ vsName) * 16777619u;
    hash = (hash ^ fsName) * 16777619u;
    hash |= 0x80000000u;
    return hash ? hash : 0x80000000u;
}

static void mglLogProgramResourceInterface(Program *program, int stage, int type)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES || type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return;
    }

    MGLShaderResourceList *resources = &program->shader_resources_list[stage][type];
    mglTraceLogNSString(@"MGL IFACE program=%u stage=%s type=%s count=%u",
                  (unsigned)program->name,
                  mglShaderStageName(stage),
                  mglMGLShaderResourceTypeName(type),
                  (unsigned)resources->count);

    for (GLuint i = 0; i < resources->count; i++) {
        MGLShaderResource *res = &resources->list[i];
        mglTraceLogNSString(@"MGL IFACE   #%u name=%s loc=%u glBinding=%u metalBinding=%u set=%u typeId=%u baseTypeId=%u required=%zu imageDim=%u arrayed=%u",
                      (unsigned)i,
                      res->name ? res->name : "(null)",
                      (unsigned)res->location,
                      (unsigned)res->gl_binding,
                      (unsigned)res->binding,
                      (unsigned)res->set,
                      (unsigned)res->type_id,
                      (unsigned)res->base_type_id,
                      res->required_size,
                      (unsigned)res->image_dim,
                      (unsigned)res->image_arrayed);
    }
}

void mglWriteProgramMSLDump(Program *program, NSString *reason)
{
    /* Early return when trace logging is disabled — avoids expensive MSL
     * file I/O and resource interface logging that would be discarded. */
    if (!mglTraceLogIsEnabled()) {
        return;
    }

    if (!program) {
        return;
    }

    BOOL forceDump = false;
    if (reason) {
        NSString *lowerReason = [reason lowercaseString];
        forceDump = [lowerReason containsString:@"tex"];
    }

    static GLuint s_dumpedPrograms[64] = {0};
    static GLuint s_forcedDumpedPrograms[64] = {0};
    static uint32_t s_dumpedProgramCount = 0;
    static uint32_t s_forcedDumpedProgramCount = 0;
    static uint32_t s_dumpGeneration = 0;
    if (forceDump) {
        for (uint32_t i = 0; i < s_forcedDumpedProgramCount; i++) {
            if (s_forcedDumpedPrograms[i] == program->name) {
                return;
            }
        }
    } else {
        for (uint32_t i = 0; i < s_dumpedProgramCount; i++) {
            if (s_dumpedPrograms[i] == program->name) {
                return;
            }
        }
    }

    if (forceDump && s_forcedDumpedProgramCount < (uint32_t)(sizeof(s_forcedDumpedPrograms) / sizeof(s_forcedDumpedPrograms[0]))) {
        s_forcedDumpedPrograms[s_forcedDumpedProgramCount++] = program->name;
    } else if (!forceDump && s_dumpedProgramCount < (uint32_t)(sizeof(s_dumpedPrograms) / sizeof(s_dumpedPrograms[0]))) {
        s_dumpedPrograms[s_dumpedProgramCount++] = program->name;
    } else {
        return;
    }
    s_dumpGeneration++;

    mglTraceLogNSString(@"MGL IFACE DUMP begin program=%u reason=%@ generation=%u",
                  (unsigned)program->name,
                  reason ?: @"(none)",
                  (unsigned)s_dumpGeneration);

    mglLogProgramResourceInterface(program, _VERTEX_SHADER, _STAGE_OUTPUT_RES);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, _STAGE_INPUT_RES);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, _STAGE_INPUT_RES);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, _STAGE_OUTPUT_RES);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, _UNIFORM_BUFFER_RES);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, _UNIFORM_BUFFER_RES);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, _UNIFORM_CONSTANT_RES);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, _UNIFORM_CONSTANT_RES);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, _SAMPLED_IMAGE_RES);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, _SAMPLED_IMAGE_RES);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, _SEPARATE_IMAGE_RES);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, _SEPARATE_IMAGE_RES);
}

/* Focus program observation state machine (g_mglFocusedLoadingPrograms,
 * g_mglFocusedLoadingProgramCount, mglFocusLoadingProgram,
 * mglObserveProgramDrawForFocus, mglIsFocusedLoadingProgram) moved to
 * mgl_focus_program.h/.m. */

/* Program trace gating helpers moved to mgl_trace_strategy.h/.m. */

/* Draw command classification helpers (mglDrawCommandTypeName,
 * mglDrawCommandUsesElements) moved to draw_command.h/.c. */

Program *mglTraceResolveDrawProgram(GLMContext traceCtx)
{
    if (!mglRendererContextLikelyValid(traceCtx)) {
        return NULL;
    }

    Program *program = mglResolveProgramFromState(traceCtx);
    if (program) {
        return program;
    }

    Program *fragmentProgram = mglResolveProgramForStageFromState(traceCtx, _FRAGMENT_SHADER);
    if (fragmentProgram) {
        return fragmentProgram;
    }

    return mglResolveProgramForStageFromState(traceCtx, _VERTEX_SHADER);
}

bool mglTraceShouldLogReplay(GLMContext traceCtx, Program *program)
{
    if (!mglTraceLogIsEnabled()) {
        return false;
    }
    if (mglTraceLogDrawAll()) {
        return true;
    }
    if (mglProgramNeedsTraceLog(program)) {
        return true;
    }
    GLuint programKey = mglCurrentRenderProgramKey(traceCtx);
    return mglIsFocusedLoadingProgram(programKey);
}

/* Y-Flip Subsystem decision logic (mglDecideYFlipForSampledRT,
 * mglProgramHasExistingFramebufferSampleYFlip, and the MGLYFlipDecision enum)
 * now lives in mgl_coordinate.m — see mgl_coordinate.h.  Keeping the decision
 * matrix in a dedicated module lets the VS/FS sampler-binding paths below
 * call a single unified query and makes the coordinate-compatibility
 * subsystem testable in isolation. */

/* findTexture, isColorAttachment, getFBOAttachment declared in MGLRenderer_Private.h */

Texture *mglTraceFramebufferAttachmentTexture(GLMContext glctx, FBOAttachment *attachment)
{
    if (!glctx || !attachment) {
        return NULL;
    }
    if (attachment->textarget == GL_RENDERBUFFER) {
        return attachment->buf.rbo ? attachment->buf.rbo->tex : NULL;
    }
    if (attachment->buf.tex) {
        return attachment->buf.tex;
    }
    if (attachment->texture != 0u) {
        return findTexture(glctx, attachment->texture);
    }
    return NULL;
}

void mglMarkTextureLevelRenderTargetWrittenImpl(Texture *tex,
                                                 GLuint level,
                                                 const char *caller,
                                                 int line)
{
    TextureLevel *texLevel = mglTextureAttachmentLevel(tex, level);
    if (!texLevel) {
        return;
    }

    GLuint oldRenderTargetWriteVersion = tex->mtl_render_target_write_version;

    texLevel->ever_written = GL_TRUE;
    texLevel->has_initialized_data = GL_TRUE;
    texLevel->suspicious_zero_upload = GL_FALSE;
    texLevel->last_init_source = kTexRenderTargetWrite;
    texLevel->last_upload_size = 0u;
    texLevel->last_src_ptr = NULL;
    texLevel->last_src_hash = 0ull;

    tex->mtl_render_target_write_version++;
    if (level < 32u) {
        tex->mtl_gl_sampled_dirty_mip_mask |= (uint32_t)1u << level;
    } else {
        tex->mtl_gl_sampled_dirty_mip_mask = UINT32_MAX;
    }

    /* Y-Flip Authority: default to "not injected".  The draw-call path
     * (markCurrentFramebufferColorAttachmentWrittenAtIndex) overwrites the
     * low bit when the rendering program had VS Y-flip injection.  Clear/blit
     * paths leave it 0, which is correct — they don't involve program
     * injection and the RT holds Metal-top-origin data. */
    tex->mtl_render_yflip_authority = (tex->mtl_render_target_write_version << 1);

    if (tex->name == 8u && mglEnvFlagEnabled("MGL_TRACE_RT_WRITE_MARKS")) {
        id<MTLTexture> mtlTexture = tex->mtl_data ? (__bridge id<MTLTexture>)(tex->mtl_data) : nil;
        mglTraceLog("RT_WRITE_MARK tex=%u level=%u oldRtVer=%u newRtVer=%u caller=%s:%d mtl=%p fmt=%lu size=%lux%lu dirty=0x%x sampledVer=%u copy=%p",
                    (unsigned)tex->name,
                    (unsigned)level,
                    (unsigned)oldRenderTargetWriteVersion,
                    (unsigned)tex->mtl_render_target_write_version,
                    caller ? caller : "(unknown)",
                    line,
                    mtlTexture,
                    (unsigned long)(mtlTexture ? mtlTexture.pixelFormat : MTLPixelFormatInvalid),
                    (unsigned long)(mtlTexture ? mtlTexture.width : 0),
                    (unsigned long)(mtlTexture ? mtlTexture.height : 0),
                    (unsigned)tex->dirty_bits,
                    (unsigned)tex->mtl_gl_sampled_write_version,
                    tex->mtl_gl_sampled_data);
    }

    /*
     * Once Metal has rendered into a texture, the CPU-side backing copy is stale.
     * Keeping DIRTY_TEXTURE_DATA set lets a later sampler bind recreate the Metal
     * texture and upload old all-zero or placeholder bytes over the rendered
     * contents. Minecraft 1.21.8's item atlas and post-chain render targets hit
     * this path frequently.
     */
    tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
}

/* mglMarkTextureLevelRenderTargetWritten macro moved to MGLRenderer_Private.h */

/* mglMarkTextureLevelMetalFilled moved to MGLRenderer_Private.h as static inline */

/* Compressed block height / upload row helpers (mglMetalCompressedBlockHeight,
 * mglMetalUploadRowsForPixelFormat) moved to mgl_texture_compat.h as
 * static inline helpers. */

/* Pixel format classification (mglMetalPixelFormatIsDepthOrStencil,
 * mglMetalPixelFormatIsPackedDepthStencil) and GL internal-format
 * classification (mglRendererGLInternalFormatLooksDepthOrStencil) now live
 * as static inline helpers in mgl_texture_compat.h — included above. */

void mglNormalizePipelineDepthStencilFormats(MTLRenderPipelineDescriptor *desc, const char *label)
{
    if (!desc) {
        return;
    }

    MTLPixelFormat depthFormat = desc.depthAttachmentPixelFormat;
    MTLPixelFormat stencilFormat = desc.stencilAttachmentPixelFormat;
    if (depthFormat == MTLPixelFormatInvalid ||
        stencilFormat == MTLPixelFormatInvalid ||
        depthFormat == stencilFormat) {
        return;
    }

    bool depthPacked = mglMetalPixelFormatIsPackedDepthStencil(depthFormat);
    bool stencilPacked = mglMetalPixelFormatIsPackedDepthStencil(stencilFormat);
    if (!depthPacked && !stencilPacked) {
        return;
    }

    MTLPixelFormat packedFormat = stencilPacked ? stencilFormat : depthFormat;
    static uint64_t s_normalizeCount = 0;
    s_normalizeCount++;
    if (s_normalizeCount <= 16ull || (s_normalizeCount % 250ull) == 0ull) {
        NSLog(@"MGL WARNING: normalizing incompatible pipeline depth/stencil formats for Metal (%s depth=%lu stencil=%lu -> %lu/%lu)",
              label ? label : "pipeline",
              (unsigned long)depthFormat,
              (unsigned long)stencilFormat,
              (unsigned long)packedFormat,
              (unsigned long)packedFormat);
    }
    desc.depthAttachmentPixelFormat = packedFormat;
    desc.stencilAttachmentPixelFormat = packedFormat;
}

/* RT Sync gate helpers (mglTextureCanUseGLSampledRenderTargetCopy,
 * mglTextureIsAttachmentOfFramebuffer, mglFramebufferLooksLikeGLSampledCopyRenderTarget)
 * now live in mgl_rt_sync.m — see mgl_rt_sync.h.  The gate logic is pure
 * spec-compliance: any GL_TEXTURE_2D render target qualifies for a
 * Y-flipped sampled copy, regardless of size or game-specific heuristics. */

/* MGLTextureDataKind enum and the data-kind helpers
 * (mglTextureDataKindForPixelFormat,
 *  mglTexturePixelFormatCompatibleWithExpectedDataKind,
 *  mglTextureDataKindName,
 *  mglRendererGLInternalFormatLooksDepthOrStencil) now live in
 * mgl_texture_compat.h — included above. */

/* mglTextureDataKindForPixelFormat, mglTexturePixelFormatCompatibleWithExpectedDataKind,
 * mglTextureDataKindName, and mglRendererGLInternalFormatLooksDepthOrStencil
 * now live in mgl_texture_compat.m — see mgl_texture_compat.h. */

BOOL mglRendererTextureLooksRecoverableSampled2D(GLMContext glctx,
                                                        Texture *tex,
                                                        MTLTextureType expectedType,
                                                        MGLTextureDataKind expectedKind)
{
    if (!glctx || !tex) {
        return NO;
    }
    if (expectedType != 0 && expectedType != MTLTextureType2D) {
        return NO;
    }
    if (!mglRendererObjectPointerLikelyValid(tex) ||
        !mglRendererPointerInHashTable(&glctx->active_state->texture_table, tex) ||
        !mglPointerRangeIsReadable(tex, sizeof(*tex))) {
        return NO;
    }
    if (tex->target != GL_TEXTURE_2D ||
        tex->index != _TEXTURE_2D ||
        tex->is_render_target ||
        mglRendererGLInternalFormatLooksDepthOrStencil(tex->internalformat)) {
        return NO;
    }

    TextureLevel *level0 = mglTraceTextureBaseLevel(tex);
    if (!level0 ||
        !level0->complete ||
        (!level0->ever_written && !level0->has_initialized_data)) {
        return NO;
    }

    id<MTLTexture> mtlTexture = tex->mtl_data ? (__bridge id<MTLTexture>)(tex->mtl_data) : nil;
    if (mtlTexture) {
        if (mglMetalPixelFormatIsDepthOrStencil(mtlTexture.pixelFormat) ||
            !mglTexturePixelFormatCompatibleWithExpectedDataKind(mtlTexture.pixelFormat, expectedKind)) {
            return NO;
        }
        if (expectedType != 0 && mtlTexture.textureType != expectedType) {
            return NO;
        }
    }

    return YES;
}

BOOL mglRendererTextureLooksLikeSampledColor2D(GLMContext glctx,
                                                      Texture *tex)
{
    if (!glctx || !tex) {
        return NO;
    }
    if (!mglRendererObjectPointerLikelyValid(tex) ||
        !mglRendererPointerInHashTable(&glctx->active_state->texture_table, tex) ||
        !mglPointerRangeIsReadable(tex, sizeof(*tex))) {
        return NO;
    }
    if (tex->target != GL_TEXTURE_2D ||
        tex->index != _TEXTURE_2D ||
        mglRendererGLInternalFormatLooksDepthOrStencil(tex->internalformat)) {
        return NO;
    }

    return YES;
}

BOOL mglRendererGLSampledCopyLooksUsable(Texture *tex,
                                                MTLTextureType expectedType,
                                                MGLTextureDataKind expectedKind,
                                                BOOL allowPreviousWriteVersion,
                                                id<MTLTexture> *copyOut,
                                                BOOL *usedPreviousWriteVersionOut)
{
    if (copyOut) {
        *copyOut = nil;
    }
    if (usedPreviousWriteVersionOut) {
        *usedPreviousWriteVersionOut = NO;
    }
    if (!tex || !tex->mtl_gl_sampled_data) {
        return NO;
    }

    id<MTLTexture> sampledCopy = (__bridge id<MTLTexture>)(tex->mtl_gl_sampled_data);
    if (!sampledCopy ||
        mglMetalPixelFormatIsDepthOrStencil(sampledCopy.pixelFormat) ||
        !mglTexturePixelFormatCompatibleWithExpectedDataKind(sampledCopy.pixelFormat, expectedKind) ||
        (expectedType != 0 && sampledCopy.textureType != expectedType)) {
        return NO;
    }
    if (tex->mtl_gl_sampled_width != (GLuint)sampledCopy.width ||
        tex->mtl_gl_sampled_height != (GLuint)sampledCopy.height ||
        tex->mtl_gl_sampled_format != (GLuint)sampledCopy.pixelFormat) {
        return NO;
    }

    BOOL exactVersion =
        tex->mtl_gl_sampled_write_version != 0u &&
        tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version;
    BOOL previousVersion =
        allowPreviousWriteVersion &&
        tex->mtl_gl_sampled_write_version != 0u &&
        tex->mtl_render_target_write_version != 0u &&
        tex->mtl_gl_sampled_write_version + 1u == tex->mtl_render_target_write_version;
    if (!exactVersion && !previousVersion) {
        return NO;
    }

    if (copyOut) {
        *copyOut = sampledCopy;
    }
    if (usedPreviousWriteVersionOut) {
        *usedPreviousWriteVersionOut = previousVersion;
    }
    return YES;
}

/* mglNowSeconds moved to MGLRenderer_Private.h as static inline */

void mglLogLoopHeartbeat(const char *tag,
                                       uint64_t callCount,
                                       double nowSeconds,
                                       double *lastCallSeconds,
                                       uint64_t *lastCallCount,
                                       double warnGapSeconds)
{
    if (!kMGLDiagnosticStateLogs || !lastCallSeconds || !lastCallCount) {
        return;
    }

    uint64_t deltaCalls = (*lastCallCount > 0) ? (callCount - *lastCallCount) : 0;
    double deltaMs = (*lastCallSeconds > 0.0) ? ((nowSeconds - *lastCallSeconds) * 1000.0) : 0.0;

    if (*lastCallSeconds > 0.0 &&
        warnGapSeconds > 0.0 &&
        (nowSeconds - *lastCallSeconds) >= warnGapSeconds) {
        mglTraceLogNSString(@"MGL TRACE %s gap=%.2fms deltaCalls=%llu call=%llu",
              tag ? tag : "loop",
              deltaMs,
              (unsigned long long)deltaCalls,
              (unsigned long long)callCount);
    } else if (mglShouldTraceCall(callCount) &&
               (callCount <= 20ull || (callCount % 60ull) == 0ull)) {
        mglTraceLogNSString(@"MGL TRACE %s heartbeat delta=%.2fms deltaCalls=%llu call=%llu",
              tag ? tag : "loop",
              deltaMs,
              (unsigned long long)deltaCalls,
              (unsigned long long)callCount);
    }

    *lastCallSeconds = nowSeconds;
    *lastCallCount = callCount;
}

/* Dirty-bits formatting helpers (mglAppendFlagName, mglFormatDirtyBits)
 * moved to mgl_state_log.h/.m. */

void mglLogStateSnapshot(const char *tag,
                                GLMContext ctx,
                                id<MTLCommandBuffer> commandBuffer,
                                id<MTLRenderCommandEncoder> renderEncoder,
                                MTLRenderPassDescriptor *renderPassDescriptor,
                                id<CAMetalDrawable> drawable)
{
    if (!kMGLDiagnosticStateLogs) {
        return;
    }

    if (!mglRendererContextLikelyValid(ctx)) {
        mglTraceLogNSString(@"MGL TRACE %s ctx=%p(invalid) cb=%p enc=%p rpd=%p drawable=%p",
              tag ? tag : "snapshot", ctx, commandBuffer, renderEncoder, renderPassDescriptor, drawable);
        return;
    }

    Program *program = mglResolveProgramFromState(ctx);
    GLuint programName = ctx->active_state->program_name ? ctx->active_state->program_name : (program ? program->name : 0);
    Framebuffer *drawFBO = ctx->active_state->framebuffer;
    GLuint drawFBOName = 0;
    if (drawFBO) {
        if (mglRendererObjectPointerLikelyValid(drawFBO) &&
            mglRendererPointerInHashTable(&ctx->active_state->framebuffer_table, drawFBO) &&
            mglPointerRangeIsReadable(drawFBO, sizeof(*drawFBO))) {
            drawFBOName = drawFBO->name;
        } else {
            mglTraceLogNSString(@"MGL TRACE %s invalid drawFBO=%p", tag ? tag : "snapshot", drawFBO);
            drawFBO = NULL;
        }
    }

    MTLCommandBufferStatus cbStatus = commandBuffer
        ? mglRenderCommandBufferStatus(commandBuffer)
        : MTLCommandBufferStatusNotEnqueued;
    NSString *cbLabel = commandBuffer ? (commandBuffer.label ?: @"(no-label)") : @"(nil)";
    char dirtyNames[256];
    mglFormatDirtyBits((uint32_t)ctx->active_state->dirty_bits, dirtyNames, sizeof(dirtyNames));

    id<MTLTexture> rpColor0 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> rpDepth = renderPassDescriptor ? renderPassDescriptor.depthAttachment.texture : nil;
    id<MTLTexture> rpStencil = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.texture : nil;
    MTLLoadAction colorLoadAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare;
    MTLStoreAction colorStoreAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare;
    MTLLoadAction depthLoadAction = renderPassDescriptor ? renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare;
    MTLStoreAction depthStoreAction = renderPassDescriptor ? renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare;
    MTLLoadAction stencilLoadAction = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.loadAction : MTLLoadActionDontCare;
    MTLStoreAction stencilStoreAction = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.storeAction : MTLStoreActionDontCare;
    MTLClearColor rpClearColor = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].clearColor : MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    id<MTLTexture> drawableTexture = drawable ? drawable.texture : nil;

    mglTraceLogNSString(@"MGL TRACE %s prog=%u dirty=0x%x[%s] clear=0x%x drawBuf=0x%x readBuf=0x%x vao=%p drawFBO=%p(%u) "
          "vp=(%u,%u,%u,%u) scissor(en=%d box=%d,%d,%d,%d) caps(depth=%d blend=%d cull=%d) "
          "stateClear=(%.3f,%.3f,%.3f,%.3f) cb=%p[%s] label=%@ enc=%p rpd=%p rt=%lux%lu "
          "c0=%p fmt=%lu usage=0x%lx la/sa=%s/%s clear=(%.3f,%.3f,%.3f,%.3f) "
          "depth=%p(%lu %s/%s) stencil=%p(%lu %s/%s) drawable=%p tex=%p d=%lux%lu",
          tag ? tag : "snapshot",
          (unsigned)programName,
          (unsigned)ctx->active_state->dirty_bits,
          dirtyNames,
          (unsigned)ctx->active_state->clear_bitmask,
          (unsigned)ctx->active_state->draw_buffer,
          (unsigned)ctx->active_state->read_buffer,
          ctx->active_state->vao,
          drawFBO,
          (unsigned)drawFBOName,
          (unsigned)ctx->active_state->viewport[0],
          (unsigned)ctx->active_state->viewport[1],
          (unsigned)ctx->active_state->viewport[2],
          (unsigned)ctx->active_state->viewport[3],
          ctx->active_state->caps.scissor_test ? 1 : 0,
          (int)ctx->active_state->var.scissor_box[0],
          (int)ctx->active_state->var.scissor_box[1],
          (int)ctx->active_state->var.scissor_box[2],
          (int)ctx->active_state->var.scissor_box[3],
          ctx->active_state->caps.depth_test ? 1 : 0,
          ctx->active_state->caps.blend ? 1 : 0,
          ctx->active_state->caps.cull_face ? 1 : 0,
          ctx->active_state->color_clear_value[0],
          ctx->active_state->color_clear_value[1],
          ctx->active_state->color_clear_value[2],
          ctx->active_state->color_clear_value[3],
          commandBuffer,
          mglCommandBufferStatusName(cbStatus),
          cbLabel,
          renderEncoder,
          renderPassDescriptor,
          (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetWidth : 0),
          (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetHeight : 0),
          rpColor0,
          (unsigned long)(rpColor0 ? rpColor0.pixelFormat : MTLPixelFormatInvalid),
          (unsigned long)(rpColor0 ? rpColor0.usage : 0),
          mglLoadActionName(colorLoadAction),
          mglStoreActionName(colorStoreAction),
          rpClearColor.red,
          rpClearColor.green,
          rpClearColor.blue,
          rpClearColor.alpha,
          rpDepth,
          (unsigned long)(rpDepth ? rpDepth.pixelFormat : MTLPixelFormatInvalid),
          mglLoadActionName(depthLoadAction),
          mglStoreActionName(depthStoreAction),
          rpStencil,
          (unsigned long)(rpStencil ? rpStencil.pixelFormat : MTLPixelFormatInvalid),
          mglLoadActionName(stencilLoadAction),
          mglStoreActionName(stencilStoreAction),
          drawable,
          drawableTexture,
          (unsigned long)(drawableTexture ? drawableTexture.width : 0),
          (unsigned long)(drawableTexture ? drawableTexture.height : 0));

    mglTraceLogNSString(@"MGL TRACE %s masks color0(use=%d rgba=%d%d%d%d) depthWrite=%d stencilWrite=0x%x",
          tag ? tag : "snapshot",
          ctx->active_state->caps.use_color_mask[0] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][3] ? 1 : 0,
          ctx->active_state->var.depth_writemask ? 1 : 0,
          (unsigned)ctx->active_state->var.stencil_writemask);
}

void mglLogDrawWithoutSwapWatchdog(const char *kind,
                                          uint64_t drawCall,
                                          GLMContext ctx,
                                          id<MTLCommandBuffer> commandBuffer,
                                          id<MTLRenderCommandEncoder> renderEncoder,
                                          MTLRenderPassDescriptor *renderPassDescriptor)
{
    uint64_t drawArrays = MGL_FRAME_LOAD(g_mglDrawArraysSinceSwap);
    uint64_t drawElements = MGL_FRAME_LOAD(g_mglDrawElementsSinceSwap);
    uint64_t totalDraws = drawArrays + drawElements;
    if (totalDraws < 16384ull || (totalDraws % 16384ull) != 0ull) {
        return;
    }

    double now = mglTraceNowSeconds();
    double lastSwap = MGL_FRAME_LOAD(g_mglLastSwapSeconds);
    double lastSwapAgeMs = (lastSwap > 0.0) ? ((now - lastSwap) * 1000.0) : -1.0;
    if (lastSwapAgeMs >= 0.0 && lastSwapAgeMs < 250.0) {
        return;
    }
    MTLCommandBufferStatus cbStatus = commandBuffer
        ? mglRenderCommandBufferStatus(commandBuffer)
        : MTLCommandBufferStatusNotEnqueued;
    id<MTLTexture> rpColor0 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].texture : nil;
    MTLLoadAction colorLoadAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare;
    MTLStoreAction colorStoreAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare;
    MTLClearColor clear = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].clearColor : MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    NSLog(@"MGL WATCHDOG: draws-without-swap kind=%s drawCall=%llu total=%llu arrays=%llu elements=%llu "
          "swapCalls=%llu lastSwapAgeMs=%.2f program=%u drawBuf=0x%x fbo=%p vao=%p cb=%p[%s] enc=%p "
          "rpd=%p c0=%p fmt=%lu la/sa=%s/%s clear=(%.3f,%.3f,%.3f,%.3f)",
          kind ? kind : "draw",
          (unsigned long long)drawCall,
          (unsigned long long)totalDraws,
          (unsigned long long)drawArrays,
          (unsigned long long)drawElements,
          (unsigned long long)MGL_FRAME_LOAD(g_mglSwapCallCount),
          lastSwapAgeMs,
          (unsigned)(ctx ? ctx->active_state->program_name : 0u),
          (unsigned)(ctx ? ctx->active_state->draw_buffer : 0u),
          ctx ? ctx->active_state->framebuffer : NULL,
          ctx ? ctx->active_state->vao : NULL,
          commandBuffer,
          mglCommandBufferStatusName(cbStatus),
          renderEncoder,
          renderPassDescriptor,
          rpColor0,
          (unsigned long)(rpColor0 ? rpColor0.pixelFormat : MTLPixelFormatInvalid),
          mglLoadActionName(colorLoadAction),
          mglStoreActionName(colorStoreAction),
          clear.red,
          clear.green,
          clear.blue,
          clear.alpha);
}

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
                                      GLsizei renderPassDrawBufferCount)
{
    if (!mglTraceLogIsEnabled()) {
        return;
    }

    MTLCommandBufferStatus cbStatus = commandBuffer
        ? mglRenderCommandBufferStatus(commandBuffer)
        : MTLCommandBufferStatusNotEnqueued;
    id<MTLTexture> c0 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> c1 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[1].texture : nil;
    id<MTLTexture> depth = renderPassDescriptor ? renderPassDescriptor.depthAttachment.texture : nil;
    id<MTLTexture> stencil = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.texture : nil;
    id<MTLTexture> drawableTexture = drawable ? drawable.texture : nil;
    MTLClearColor clear = renderPassDescriptor
        ? renderPassDescriptor.colorAttachments[0].clearColor
        : MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    Framebuffer *fbo = ctx ? ctx->active_state->framebuffer : NULL;
    if (fbo &&
        (!mglRendererObjectPointerLikelyValid(fbo) ||
         !mglRendererPointerInHashTable(&ctx->active_state->framebuffer_table, fbo) ||
         !mglPointerRangeIsReadable(fbo, sizeof(*fbo)))) {
        mglTraceLog("RENDERPASS_%s invalid lifecycle fbo=%p", tag ? tag : "unknown", fbo);
        fbo = NULL;
    }
    GLuint fboName = fbo ? fbo->name : 0u;
    GLuint color0Name = 0u;
    GLuint color1Name = 0u;
    GLuint depthName = 0u;
    if (fbo) {
        color0Name = fbo->color_attachments[0].texture;
        color1Name = fbo->color_attachments[1].texture;
        depthName = fbo->depth.texture;
    }

    mglTraceLog("RENDERPASS_%s call=%llu program=%u dirty=0x%x drawBuf=0x%x readBuf=0x%x "
                "fbo=%u(%p) rpFbo=%u(%p) rpDrawBuf=0x%x rpDrawCount=%d vao=%p cb=%p[%s] enc=%p rpd=%p rt=%lux%lu "
                "c0Name=%u c0=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s clear=(%.3f,%.3f,%.3f,%.3f) "
                "c1Name=%u c1=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s "
                "depthName=%u depth=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s "
                "stencil=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s "
                "drawable=%p tex=%p size=%lux%lu",
                tag ? tag : "unknown",
                (unsigned long long)call,
                (unsigned)(ctx ? ctx->active_state->program_name : 0u),
                (unsigned)(ctx ? ctx->active_state->dirty_bits : 0u),
                (unsigned)(ctx ? ctx->active_state->draw_buffer : 0u),
                (unsigned)(ctx ? ctx->active_state->read_buffer : 0u),
                (unsigned)fboName,
                fbo,
                (unsigned)renderPassFramebufferName,
                renderPassFramebuffer,
                (unsigned)renderPassDrawBuffer,
                (int)renderPassDrawBufferCount,
                ctx ? ctx->active_state->vao : NULL,
                commandBuffer,
                mglCommandBufferStatusName(cbStatus),
                renderEncoder,
                renderPassDescriptor,
                (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetWidth : 0),
                (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetHeight : 0),
                (unsigned)color0Name,
                c0,
                (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
                (unsigned long)(c0 ? c0.usage : 0),
                (unsigned long)(c0 ? c0.width : 0),
                (unsigned long)(c0 ? c0.height : 0),
                mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare),
                mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare),
                clear.red,
                clear.green,
                clear.blue,
                clear.alpha,
                (unsigned)color1Name,
                c1,
                (unsigned long)(c1 ? c1.pixelFormat : MTLPixelFormatInvalid),
                (unsigned long)(c1 ? c1.usage : 0),
                (unsigned long)(c1 ? c1.width : 0),
                (unsigned long)(c1 ? c1.height : 0),
                mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[1].loadAction : MTLLoadActionDontCare),
                mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[1].storeAction : MTLStoreActionDontCare),
                (unsigned)depthName,
                depth,
                (unsigned long)(depth ? depth.pixelFormat : MTLPixelFormatInvalid),
                (unsigned long)(depth ? depth.usage : 0),
                (unsigned long)(depth ? depth.width : 0),
                (unsigned long)(depth ? depth.height : 0),
                mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare),
                mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare),
                stencil,
                (unsigned long)(stencil ? stencil.pixelFormat : MTLPixelFormatInvalid),
                (unsigned long)(stencil ? stencil.usage : 0),
                (unsigned long)(stencil ? stencil.width : 0),
                (unsigned long)(stencil ? stencil.height : 0),
                mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.stencilAttachment.loadAction : MTLLoadActionDontCare),
                mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.stencilAttachment.storeAction : MTLStoreActionDontCare),
                drawable,
                drawableTexture,
                (unsigned long)(drawableTexture ? drawableTexture.width : 0),
                (unsigned long)(drawableTexture ? drawableTexture.height : 0));
}

BOOL mglRendererPointerInHashTable(HashTable *table, const void *ptr)
{
    return mglRendererObjectPointerLikelyValid(ptr) &&
           mglHashTableContainsData(table, ptr);
}

Texture *mglFindFramebufferColorTexturePairedWithDepth(GLMContext glctx,
                                                              Texture *depthTexture,
                                                              GLuint *fboNameOut)
{
    if (fboNameOut) {
        *fboNameOut = 0u;
    }
    if (!glctx || !depthTexture) {
        return NULL;
    }

    Framebuffer *currentFbo = glctx->active_state->framebuffer;
    if (currentFbo &&
        mglRendererObjectPointerLikelyValid(currentFbo) &&
        mglPointerRangeIsReadable(currentFbo, sizeof(*currentFbo))) {
        BOOL depthMatches =
            currentFbo->depth.buf.tex == depthTexture ||
            currentFbo->stencil.buf.tex == depthTexture ||
            currentFbo->depth.texture == depthTexture->name ||
            currentFbo->stencil.texture == depthTexture->name;
        if (depthMatches && (currentFbo->color_attachment_bitfield & 1u) != 0u) {
            FBOAttachment *colorAttachment = &currentFbo->color_attachments[0];
            Texture *colorTexture = colorAttachment->buf.tex;
            if (!colorTexture && colorAttachment->texture != 0u) {
                colorTexture = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                          colorAttachment->texture);
            }
            /* Validate raw pointer is still registered (see table-scan path). */
            if (colorTexture) {
                Texture *verified = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                                colorTexture->name);
                if (verified != colorTexture) {
                    colorAttachment->buf.tex = NULL;
                    colorAttachment->texture = 0u;
                    colorTexture = NULL;
                }
            }
            if (colorTexture &&
                colorTexture != depthTexture &&
                mglRendererObjectPointerLikelyValid(colorTexture) &&
                mglPointerRangeIsReadable(colorTexture, sizeof(*colorTexture)) &&
                (!colorTexture->mtl_data ||
                 !mglMetalPixelFormatIsDepthOrStencil([(__bridge id<MTLTexture>)colorTexture->mtl_data pixelFormat]))) {
                if (fboNameOut) {
                    *fboNameOut = currentFbo->name;
                }
                return colorTexture;
            }
        }
    }

    HashTable *table = &glctx->active_state->framebuffer_table;
    if (!mglHashTableValidateStorage(table, "findPairedFramebufferColor") ||
        !table->keys || !table->states || table->size == 0u) {
        return NULL;
    }

    for (size_t slot = 0; slot < table->size; slot++) {
        if (table->states[slot] != 1u || !table->keys[slot].data) {
            continue;
        }

        Framebuffer *fbo = (Framebuffer *)table->keys[slot].data;
        if (!mglRendererObjectPointerLikelyValid(fbo) ||
            !mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
            continue;
        }

        BOOL depthMatches =
            fbo->depth.buf.tex == depthTexture ||
            fbo->stencil.buf.tex == depthTexture ||
            fbo->depth.texture == depthTexture->name ||
            fbo->stencil.texture == depthTexture->name;
        if (!depthMatches) {
            continue;
        }

        FBOAttachment *colorAttachment = &fbo->color_attachments[0];
        Texture *colorTexture = colorAttachment->buf.tex;
        if (!colorTexture && colorAttachment->texture != 0u) {
            colorTexture = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                      colorAttachment->texture);
        }

        /* Validate that the raw pointer is still registered in the texture
         * table.  glDeleteTextures frees the Texture struct but stale raw
         * pointers can survive in FBO attachments (and mglPointerRangeIsReadable
         * cannot reliably detect freed-but-mapped malloc memory). */
        if (colorTexture) {
            Texture *verified = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                            colorTexture->name);
            if (verified != colorTexture) {
                /* Stale pointer — clear it and skip. */
                colorAttachment->buf.tex = NULL;
                colorAttachment->texture = 0u;
                continue;
            }
        }

        if (!colorTexture ||
            colorTexture == depthTexture ||
            !mglRendererObjectPointerLikelyValid(colorTexture) ||
            !mglPointerRangeIsReadable(colorTexture, sizeof(*colorTexture))) {
            continue;
        }

        if (colorTexture->mtl_data &&
            mglMetalPixelFormatIsDepthOrStencil([(__bridge id<MTLTexture>)colorTexture->mtl_data pixelFormat])) {
            continue;
        }

        if (fboNameOut) {
            *fboNameOut = fbo->name;
        }
        return colorTexture;
    }

    return NULL;
}

BOOL mglCurrentDrawFramebufferUsesColorTexture(GLMContext glctx,
                                                      Texture *texture,
                                                      GLuint expectedFboName,
                                                      NSUInteger *attachmentIndexOut)
{
    if (attachmentIndexOut) {
        *attachmentIndexOut = MAX_COLOR_ATTACHMENTS;
    }
    if (!glctx || !texture) {
        return NO;
    }

    Framebuffer *fbo = glctx->active_state->framebuffer;
    if (!fbo ||
        !mglRendererObjectPointerLikelyValid(fbo) ||
        !mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        return NO;
    }
    if (expectedFboName != 0u && fbo->name != expectedFboName) {
        return NO;
    }

    GLsizei drawBufferCount = mglMetalDrawBufferCount(glctx);
    for (GLsizei i = 0; i < drawBufferCount; i++) {
        GLuint attachmentIndex = MAX_COLOR_ATTACHMENTS;
        if (!mglMetalResolveFboDrawAttachmentIndex(glctx,
                                                   mglMetalDrawBufferAt(glctx, (GLuint)i),
                                                   &attachmentIndex) ||
            attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        if (attachment->buf.tex == texture || attachment->texture == texture->name) {
            if (attachmentIndexOut) {
                *attachmentIndexOut = attachmentIndex;
            }
            return YES;
        }
    }

    return NO;
}

static void mglRendererDropCurrentVAO(GLMContext ctx)
{
    if (!ctx) {
        return;
    }

    ctx->active_state->vao = NULL;
    ctx->active_state->buffers[_ELEMENT_ARRAY_BUFFER] = ctx->active_state->default_vao_element_array_buffer;
    ctx->active_state->var.element_array_buffer_binding =
        ctx->active_state->default_vao_element_array_buffer ? ctx->active_state->default_vao_element_array_buffer->name : 0;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_VAO);
}

VertexArray *mglRendererGetValidatedVAO(GLMContext ctx, const char *where)
{
    if (!ctx) {
        return NULL;
    }

    VertexArray *vao = ctx->active_state->vao;
    if (!vao) {
        return NULL;
    }

    if (!mglRendererObjectPointerLikelyValid(vao)) {
        NSLog(@"MGL VAO INVALID in %s: vao=%p (suspicious pseudo-pointer)",
              where ? where : "unknown", vao);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    /* Fast path: hashtable membership implies the table holds a live
     * reference, so the memory is valid and we can safely read fields
     * without the expensive vm_region_64 syscall.  The generation cache
     * in mglHashTableContainsData makes this O(1) in the common case. */
    if (mglRendererPointerInHashTable(&ctx->active_state->vao_table, vao)) {
        if (vao->magic != MGL_VAO_MAGIC) {
            NSLog(@"MGL VAO INVALID in %s: vao=%p magic=0x%x",
                  where ? where : "unknown", vao, vao->magic);
            mglRendererDropCurrentVAO(ctx);
            return NULL;
        }
        return vao;
    }

    /* Slow path: not in table — could be a transient_batch_vao or a
     * dangling pointer.  Use the syscall to determine which. */
    if (!mglPointerRangeIsReadable(vao, sizeof(*vao))) {
        NSLog(@"MGL VAO INVALID in %s: vao=%p (unreadable object memory)",
              where ? where : "unknown", vao);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    if (vao->magic != MGL_VAO_MAGIC) {
        NSLog(@"MGL VAO INVALID in %s: vao=%p magic=0x%x",
              where ? where : "unknown", vao, vao->magic);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    if (vao->transient_batch_vao) {
        return vao;
    }

    NSLog(@"MGL VAO INVALID in %s: vao=%p (not found in sane vao_table)",
          where ? where : "unknown", vao);
    mglRendererDropCurrentVAO(ctx);
    return NULL;
}

Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate, const char *where, NSUInteger slot)
{
    if (!candidate) {
        return NULL;
    }

    if (!mglRendererObjectPointerLikelyValid(candidate)) {
        NSLog(@"MGL BUFFER INVALID in %s: slot=%lu candidate=%p (suspicious pseudo-pointer)",
              where ? where : "unknown", (unsigned long)slot, candidate);
        return NULL;
    }

    /* Fast path: hashtable membership implies memory is valid (table holds
     * a live reference), so we can skip the vm_region_64 syscall. */
    if (ctx && mglRendererPointerInHashTable(&ctx->active_state->buffer_table, candidate)) {
        return candidate;
    }

    /* Slow path: not in table — could be transient_batch_buffer or dangling. */
    if (!mglPointerRangeIsReadable(candidate, sizeof(*candidate))) {
        NSLog(@"MGL BUFFER INVALID in %s: slot=%lu candidate=%p (unreadable object memory)",
              where ? where : "unknown", (unsigned long)slot, candidate);
        return NULL;
    }

    if (candidate->transient_batch_buffer) {
        return candidate;
    }

    NSLog(@"MGL BUFFER INVALID in %s: slot=%lu candidate=%p (not found in sane buffer_table)",
          where ? where : "unknown", (unsigned long)slot, candidate);
    return NULL;
}

/* MGLResolvedVertexAttribBinding typedef moved to MGLRenderer_Private.h */

bool mglRendererResolveVertexAttribBinding(GLMContext ctx,
                                                  VertexArray *vao,
                                                  GLuint attribute,
                                                  const char *where,
                                                  MGLResolvedVertexAttribBinding *out)
{
    if (!ctx || !vao || attribute >= MAX_ATTRIBS || !out) {
        return false;
    }

    const VertexAttrib *attrib = &vao->attrib[attribute];
    Buffer *buffer = attrib->buffer;
    GLintptr bindingOffset = attrib->binding_offset;
    GLuint stride = attrib->stride;
    GLuint divisor = attrib->divisor;
    GLuint bindingIndex = attrib->buffer_bindingindex;
    bool usesBindingTable = false;

    if (bindingIndex < MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        const BufferBinding *binding = &vao->bindings[bindingIndex];
        if (binding->buffer) {
            buffer = binding->buffer;
            bindingOffset = binding->offset;
            stride = (binding->stride > 0) ? (GLuint)binding->stride : attrib->stride;
            divisor = binding->divisor;
            usesBindingTable = true;
        }
    }

    Buffer *validated = mglRendererGetValidatedBuffer(ctx, buffer, where, attribute);
    if (!validated) {
        return false;
    }

    out->attrib = attrib;
    out->buffer = validated;
    out->binding_offset = bindingOffset;
    out->stride = stride;
    out->divisor = divisor;
    out->relativeoffset = attrib->relativeoffset;
    out->binding_index = bindingIndex;
    out->uses_binding_table = usesBindingTable;
    return true;
}

Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx, const char *where)
{
    if (!ctx) {
        return NULL;
    }

    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!fbo) {
        return NULL;
    }

    if (!mglRendererObjectPointerLikelyValid(fbo)) {
        NSLog(@"MGL FBO INVALID in %s: framebuffer=%p (suspicious pseudo-pointer)",
              where ? where : "unknown", fbo);
        if (ctx->active_state->readbuffer == fbo) {
            ctx->active_state->readbuffer = NULL;
        }
        ctx->active_state->framebuffer = NULL;
        mglRendererSyncFramebufferBindingNames(ctx);
        mglMarkStateDirtyBits(ctx->active_state, (DIRTY_FBO | DIRTY_STATE));
        return NULL;
    }

    /* Fast path: hashtable membership implies memory is valid, so we can
     * skip the vm_region_64 syscall that was previously unconditionally
     * performed on every per-draw/per-batch call to this helper. */
    if (mglRendererPointerInHashTable(&ctx->active_state->framebuffer_table, fbo)) {
        return fbo;
    }

    /* Slow path: not in table — do the syscall for diagnostics. */
    if (!mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        NSLog(@"MGL FBO INVALID in %s: framebuffer=%p (not found in sane framebuffer_table or unreadable)",
              where ? where : "unknown", fbo);
        if (ctx->active_state->readbuffer == fbo) {
            ctx->active_state->readbuffer = NULL;
        }
        ctx->active_state->framebuffer = NULL;
        mglRendererSyncFramebufferBindingNames(ctx);
        mglMarkStateDirtyBits(ctx->active_state, (DIRTY_FBO | DIRTY_STATE));
        return NULL;
    }

    NSLog(@"MGL FBO INVALID in %s: framebuffer=%p (not found in sane framebuffer_table)",
          where ? where : "unknown", fbo);
    if (ctx->active_state->readbuffer == fbo) {
        ctx->active_state->readbuffer = NULL;
    }
    ctx->active_state->framebuffer = NULL;
    mglRendererSyncFramebufferBindingNames(ctx);
    mglMarkStateDirtyBits(ctx->active_state, (DIRTY_FBO | DIRTY_STATE));
    return NULL;
}

GLuint mglRendererSafeFramebufferName(GLMContext ctx)
{
    Framebuffer *fbo = mglRendererGetValidatedFramebuffer(ctx, "safeFramebufferName");
    return fbo ? fbo->name : 0u;
}

/* Buffer query helpers moved to mgl_buffer_query.h/.m. */

/* Vertex attrib query helpers moved to mgl_vertex_attrib_query.h/.m. */

NSUInteger mglRendererBuildCurrentVertexAttribBytes(GLMContext ctx,
                                                           GLuint attribute,
                                                           const VertexAttrib *attrib,
                                                           uint8_t bytes[16])
{
    if (!ctx || !attrib || !bytes || attribute >= MAX_ATTRIBS) {
        return 0u;
    }

    bzero(bytes, 16);
    const CurrentVertexAttrib *current = &ctx->active_state->current_vertex_attrib[attribute];
    GLuint size = attrib->size;
    if (size == 0u || size > 4u) {
        size = 4u;
    }

    switch (attrib->type) {
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT:
        {
            size_t componentBytes = (attrib->type == GL_BYTE) ? sizeof(int8_t) :
                                    (attrib->type == GL_SHORT) ? sizeof(int16_t) :
                                    sizeof(int32_t);
            if (componentBytes == 0u || componentBytes * size > 16u) {
                return 0u;
            }
            for (GLuint i = 0; i < size; i++) {
                GLint value = current->i[i];
                if (attrib->type == GL_BYTE) {
                    int8_t packed = (int8_t)value;
                    memcpy(bytes + i * componentBytes, &packed, componentBytes);
                } else if (attrib->type == GL_SHORT) {
                    int16_t packed = (int16_t)value;
                    memcpy(bytes + i * componentBytes, &packed, componentBytes);
                } else {
                    int32_t packed = (int32_t)value;
                    memcpy(bytes + i * componentBytes, &packed, componentBytes);
                }
            }
            return 16u;
        }
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_UNSIGNED_INT:
        {
            size_t componentBytes = (attrib->type == GL_UNSIGNED_BYTE) ? sizeof(uint8_t) :
                                    (attrib->type == GL_UNSIGNED_SHORT) ? sizeof(uint16_t) :
                                    sizeof(uint32_t);
            if (componentBytes == 0u || componentBytes * size > 16u) {
                return 0u;
            }
            for (GLuint i = 0; i < size; i++) {
                GLuint value = current->u[i];
                if (attrib->type == GL_UNSIGNED_BYTE) {
                    uint8_t packed = (uint8_t)value;
                    memcpy(bytes + i * componentBytes, &packed, componentBytes);
                } else if (attrib->type == GL_UNSIGNED_SHORT) {
                    uint16_t packed = (uint16_t)value;
                    memcpy(bytes + i * componentBytes, &packed, componentBytes);
                } else {
                    uint32_t packed = (uint32_t)value;
                    memcpy(bytes + i * componentBytes, &packed, componentBytes);
                }
            }
            return 16u;
        }
        case GL_DOUBLE:
        case GL_FLOAT:
        default:
        {
            GLfloat packed[4] = {
                current->f[0],
                current->f[1],
                current->f[2],
                current->f[3],
            };
            memcpy(bytes, packed, sizeof(packed));
            return sizeof(packed);
        }
    }
}

void mglLogSkippedGLSampledRenderTargetCopy(GLMContext glctx,
                                                   Program *program,
                                                   Texture *tex,
                                                   const char *stage,
                                                   const char *sampledName,
                                                   GLuint binding,
                                                   GLuint textureUnit,
                                                   const char *reason)
{
    if (!mglTextureCanUseGLSampledRenderTargetCopy(tex)) {
        return;
    }

    if (mglTraceLogIsEnabled()) {
        mglTraceLog("RT_SAMPLE_COPY_SKIP stage=%s program=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" size=%ux%u reason=%s yflip=%d",
                    stage ? stage : "",
                    glctx ? (unsigned)glctx->active_state->program_name : 0u,
                    sampledName ? sampledName : "",
                    (unsigned)binding,
                    (unsigned)textureUnit,
                    (unsigned)tex->name,
                    mglTraceTextureLabel(tex),
                    tex ? (unsigned)tex->width : 0u,
                    tex ? (unsigned)tex->height : 0u,
                    reason ? reason : "",
                    mglProgramHasExistingFramebufferSampleYFlip(program) ? 1 : 0);
    }
}

int mglRendererResolveVertexAttributeBufferIndex(GLMContext ctx,
                                                 VertexArray *vao,
                                                 GLuint attribute,
                                                 const char *where)
{
    if (!ctx || !vao || attribute >= MAX_ATTRIBS) {
        return -1;
    }

    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (!mglRendererProgramUsesVertexAttrib(activeProgram, attribute)) {
        return -1;
    }

    Buffer *seenBuffers[MAX_ATTRIBS] = {0};
    GLintptr seenOffsets[MAX_ATTRIBS] = {0};
    GLuint seenStrides[MAX_ATTRIBS] = {0};
    GLuint seenDivisors[MAX_ATTRIBS] = {0};
    BOOL seenCurrentAttribs[MAX_ATTRIBS] = {NO};
    GLuint seenCount = 0;
    GLuint maxAttribs = MAX_ATTRIBS;

    bool vaoHasExplicitAttribs = (vao->enabled_attribs != 0u);
    for (GLuint i = 0; i < maxAttribs; i++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, i)) {
            continue;
        }

        BOOL usesCurrentValue = mglRendererVertexAttribUsesCurrentValue(vao, i);
        int slot = -1;
        if (usesCurrentValue) {
            for (GLuint s = 0; s < seenCount; s++) {
                if (seenCurrentAttribs[s] && seenOffsets[s] == (GLintptr)i) {
                    slot = (int)s;
                    break;
                }
            }
            if (slot < 0) {
                if (kMGLVertexAttribBufferBase + seenCount > kMGLMaxMetalVertexBufferIndex) {
                    NSLog(@"MGL ERROR: Vertex attrib current-value mapping overflow (seen=%u base=%lu maxIndex=%lu)",
                          seenCount, (unsigned long)kMGLVertexAttribBufferBase, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                    return -1;
                }
                seenCurrentAttribs[seenCount] = YES;
                seenOffsets[seenCount] = (GLintptr)i;
                seenStrides[seenCount] = 16u;
                seenDivisors[seenCount] = 0u;
                slot = (int)seenCount;
                seenCount++;
            }
        } else {
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(ctx, vao, i, where, &resolved)) {
            continue;
        }
        if (resolved.binding_offset < 0) {
            NSLog(@"MGL ERROR: attribute %u has negative vertex binding offset=%lld in %s",
                  i, (long long)resolved.binding_offset, where);
            return -1;
        }
        Buffer *attribBuffer = resolved.buffer;

        for (GLuint s = 0; s < seenCount; s++) {
            if (seenCurrentAttribs[s]) {
                continue;
            }
            Buffer *known = seenBuffers[s];
            if (mglRendererSameVertexStream(known,
                                            seenOffsets[s],
                                            seenStrides[s],
                                            seenDivisors[s],
                                            attribBuffer,
                                            resolved.binding_offset,
                                            resolved.stride,
                                            resolved.divisor)) {
                slot = (int)s;
                break;
            }
        }

        if (slot < 0) {
            if (kMGLVertexAttribBufferBase + seenCount > kMGLMaxMetalVertexBufferIndex) {
                NSLog(@"MGL ERROR: Vertex attrib mapping overflow (seen=%u base=%lu maxIndex=%lu)",
                      seenCount, (unsigned long)kMGLVertexAttribBufferBase, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }

            seenBuffers[seenCount] = attribBuffer;
            seenOffsets[seenCount] = resolved.binding_offset;
            seenStrides[seenCount] = resolved.stride;
            seenDivisors[seenCount] = resolved.divisor;
            slot = (int)seenCount;
            seenCount++;
        }
        }

        if (i == attribute) {
            NSUInteger resolvedIndex = kMGLVertexAttribBufferBase + (NSUInteger)slot;
            if (resolvedIndex > kMGLMaxMetalVertexBufferIndex) {
                NSLog(@"MGL ERROR: Vertex attrib index out of Metal range (attrib=%u resolved=%lu max=%lu)",
                      attribute, (unsigned long)resolvedIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }
            return (int)resolvedIndex;
        }

        (void)vaoHasExplicitAttribs;
    }

    return -1;
}

// === GL-thread contract (lock replacement) ===
//
// The Metal layer is owned by a single thread.  METAL_LOCK/METAL_UNLOCK
// (defined in MGLRenderer_Private.h) no longer acquire a lock — they expand
// to MGL_ASSERT_GL_THREAD(), validating the single-thread contract in
// Debug builds and compiling to nothing in Release.
//
// Former lock roles are now explicit thread-affinity roles:
//
// 1. GL calling thread — executes gl* entry points and all MGLRenderer
//    state operations (draw/encode paths) including waitUntilCompleted
//    (RenderPass.m commitFinish/wait paths).  May call
//    recordGPUError/recordGPUSuccess (gpuErrorLock).
//
// 2. Metal worker thread — addCompletedHandler: completion callbacks
//    (commitCommandBufferWithAGXRecovery).  Only touches the
//    _gpuRecovery.* error-tracking ivars under _gpuRecovery.gpuErrorLock;
//    never runs MGLRenderer state operations.  May request resetMetalState
//    via the _deviceResetRequested atomic flag (drained on the GL thread
//    at the swap frame boundary).
//
// 3. Main queue — AppKit view geometry only: KVO/NSWindow notifications
//    call mglMainThreadSyncViewGeometry, which publishes the geometry into
//    the pending-drawable-size atomics.  The GL thread consumes the
//    snapshot in mglApplyPendingDrawableSize.  Main queue never runs
//    MGLRenderer state operations.
//
// The Locked pattern (public wrapper + *Locked impl) is retained for
// structural clarity but no longer relies on any lock.
//
// Static helper-state variables each have a single owning thread role;
// see the C annotations at their definitions.

// Forward declarations for private helpers extracted from
// createMTLTextureFromGLTexture:, mapGLBuffersToMTLBufferMap:stage:, and
// mtlSwapBuffersLocked:.  These are only called within this file.
@interface MGLRenderer ()
// createMTLTextureFromGLTexture: helpers
- (id<MTLTexture>)createMTLTexelBufferTexture:(Texture *)tex;
- (BOOL)checkTextureCompleteness:(Texture *)tex
                          texType:(MTLTextureType)tex_type
                         numFaces:(uint)num_faces
             effectiveMipmapLevels:(GLuint *)outEffectiveMipmapLevels
                 storageMipmapped:(BOOL *)outStorageMipmapped;
- (void)logMTLTextureMipDiagnostics:(Texture *)tex
                              metal:(id<MTLTexture>)texture
               effectiveMipLevels:(GLuint)effective_mipmap_levels;
// mtlSwapBuffersLocked: helpers (copyRenderPassColorToDrawableIfNeeded: and
// scheduleSwapTextureSampleDiagnostics:) moved to
// MGLRenderer+SwapDiagnostics.m
@end

// Main class performing the rendering
@implementation MGLRenderer

MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUCharNormalized;
                    case 2: return MTLVertexFormatUChar2Normalized;
                    case 3: return MTLVertexFormatUChar3Normalized;
                    case 4: return MTLVertexFormatUChar4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUChar;
                    case 2: return MTLVertexFormatUChar2;
                    case 3: return MTLVertexFormatUChar3;
                    case 4: return MTLVertexFormatUChar4;
                }
            }
            break;

        case GL_BYTE:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatCharNormalized;
                    case 2: return MTLVertexFormatChar2Normalized;
                    case 3: return MTLVertexFormatChar3Normalized;
                    case 4: return MTLVertexFormatChar4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatChar;
                    case 2: return MTLVertexFormatChar2;
                    case 3: return MTLVertexFormatChar3;
                    case 4: return MTLVertexFormatChar4;
                }
            }
            break;

        case GL_UNSIGNED_SHORT:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUShortNormalized;
                    case 2: return MTLVertexFormatUShort2Normalized;
                    case 3: return MTLVertexFormatUShort3Normalized;
                    case 4: return MTLVertexFormatUShort4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUShort;
                    case 2: return MTLVertexFormatUShort2;
                    case 3: return MTLVertexFormatUShort3;
                    case 4: return MTLVertexFormatUShort4;
                }
            }
            break;

        case GL_SHORT:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatShortNormalized;
                    case 2: return MTLVertexFormatShort2Normalized;
                    case 3: return MTLVertexFormatShort3Normalized;
                    case 4: return MTLVertexFormatShort4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatShort;
                    case 2: return MTLVertexFormatShort2;
                    case 3: return MTLVertexFormatShort3;
                    case 4: return MTLVertexFormatShort4;
                }
            }
            break;

            case GL_HALF_FLOAT:
                switch(size)
                {
                    case 1: return MTLVertexFormatHalf;
                    case 2: return MTLVertexFormatHalf2;
                    case 3: return MTLVertexFormatHalf3;
                    case 4: return MTLVertexFormatHalf4;
                }
                break;

            case GL_FLOAT:
                switch(size)
                {
                    case 1: return MTLVertexFormatFloat;
                    case 2: return MTLVertexFormatFloat2;
                    case 3: return MTLVertexFormatFloat3;
                    case 4: return MTLVertexFormatFloat4;
                }
                break;

            case GL_INT:
                switch(size)
                {
                    case 1: return MTLVertexFormatInt;
                    case 2: return MTLVertexFormatInt2;
                    case 3: return MTLVertexFormatInt3;
                    case 4: return MTLVertexFormatInt4;
                }
                break;

            case GL_UNSIGNED_INT:
                switch(size)
                {
                    case 1: return MTLVertexFormatUInt;
                    case 2: return MTLVertexFormatUInt2;
                    case 3: return MTLVertexFormatUInt3;
                    case 4: return MTLVertexFormatUInt4;
                }
                break;

            case GL_RGB10:
                if (normalized)
                    return MTLVertexFormatInt1010102Normalized;
                break;

            case GL_INT_2_10_10_10_REV:
                if (normalized)
                    return MTLVertexFormatInt1010102Normalized;
                break;

            case GL_UNSIGNED_INT_2_10_10_10_REV:
                if (normalized)
                    return MTLVertexFormatUInt1010102Normalized;
                break;

            case GL_UNSIGNED_INT_10_10_10_2:
                /* Non-REV layout: R in MSB (bits 22-31), A in LSB (bits 0-1).
                 * Metal's UInt1010102Normalized is the REV layout (R in LSB,
                 * A in MSB); the two bit orders are incompatible and cannot be
                 * mapped directly.  Return Invalid so the caller falls back to
                 * the CPU conversion path (the mglDoubleVertexAttribFloatFormat
                 * path in generateVertexDescriptor). */
                break;

            /* GL_UNSIGNED_INT_10F_11F_11F_REV: 11/11/10 float packed format,
             * with no corresponding Metal vertex format.  Return Invalid; the
             * CPU must unpack it to float (like the GL_DOUBLE
             * mglDoubleVertexAttribFloatFormat path).  The CPU conversion
             * entry point lives in generateVertexDescriptor
             * (MGLRenderer+RenderPass.m) and needs extending to recognize
             * this type. */
            case GL_UNSIGNED_INT_10F_11F_11F_REV:
                break;

            /* GL_FIXED: 16.16 fixed-point format, with no corresponding Metal
             * vertex format.  Return Invalid; the CPU must unpack it to float.
             * The CPU conversion entry point lives in generateVertexDescriptor
             * (MGLRenderer+RenderPass.m) and needs extending to recognize
             * this type. */
            case GL_FIXED:
                break;
        }

    return MTLVertexFormatInvalid;
}

/* mglVertexAttribComponentSize / mglVertexFormatName moved to mgl_vertex_format.h/.m. */

bool mglShouldInspectDrawCall(uint64_t drawCall, GLuint programName)
{
    if (!kMGLDrawSubmitDiagnostics) {
        return false;
    }

    if (drawCall <= 120ull) {
        return true;
    }

    if (mglIsFocusedLoadingProgram(programName)) {
        return (drawCall <= 512ull) || ((drawCall % 64ull) == 0ull);
    }

    // Keep a denser trail for active Minecraft pipeline churn without flooding.
    if ((programName == 3u || programName == 74u) && ((drawCall % 40ull) == 0ull)) {
        return true;
    }

    return ((drawCall % 128ull) == 0ull);
}

/* mglGLIndexElementSize / mglReadGLIndexValue moved to mgl_vertex_format.h/.m. */

/* Index buffer builder helpers moved to mgl_index_buffer.h/.m. */

/* GL draw-mode classification helpers (mglPrimitiveModeHasDrawableSegment,
 * mglDrawModeProducesPolygons, mglPolygonModePointForDrawMode,
 * mglPolygonModeLineForDrawMode) moved to mgl_draw_mode.h. */

/* Index buffer builder helpers moved to mgl_index_buffer.h/.m. */

/* Draw encode helpers (mglEncodeArrayLineLoop, mglEncodeArrayTriangleFan,
 * mglEncodeElementLineLoop, mglEncodeElementTriangleFan, mglEncodeArrayQuads,
 * mglEncodeElementQuads, mglEncodeArrayPolygonPoint, mglEncodeElementPolygonPoint,
 * mglEncodeRestartSegment, mglEncodePrimitiveRestartedElementDraw,
 * mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled,
 * mglSkipIndirectDrawWhenPolygonPointEmulationNeeded) moved to
 * mgl_draw_encode.h/.m. */

/* mglHashStepU64 moved to mgl_byte_hash.h as static inline. */

/* mglVertexDescriptorSignature / mglPipelineDescriptorSignature / mglMaybeInvertMTLWinding moved to mgl_vertex_format.h/.m. */

void mglEnableIndirectCommandBuffersForPipeline(MTLRenderPipelineDescriptor *pipelineStateDescriptor)
{
    if (!pipelineStateDescriptor) {
        return;
    }

    /*
     * Some Minecraft shaders and helper blit/clear shaders are rejected by AGX
     * when supportIndirectCommandBuffers is enabled on the pipeline descriptor
     * ("Fragment shader cannot be used with indirect command buffers"). Keep
     * ICB-capable pipelines behind an explicit opt-in so normal rendering and
     * swap-to-drawable copy pipelines stay compatible.
     */
    if (!mglEnvFlagEnabled("MGL_ENABLE_ICB_PIPELINES")) {
        return;
    }

    if (@available(macOS 10.14, *)) {
        pipelineStateDescriptor.supportIndirectCommandBuffers = YES;
    }
}


/* mglTraceHashBytes / mglTraceFormatBytes / mglDumpBytesToLog moved to
 * mgl_byte_hash.h/.m. */

/* mglVertexAttribElementBytes / mglDoubleVertexAttribFloatFormat moved to mgl_vertex_format.h/.m. */

/* mglIntegerAttribNeedsConversion (incl. preceding doc comment) moved to mgl_vertex_format.h/.m. */

/* mglHashVertexBytesFNV1a moved to mgl_byte_hash.h/.m. */

/* mglAlignVertexStrideForMetal / mglDecodeVertexAttribComponent moved to mgl_vertex_format.h/.m. */

void mglTraceDrawElementsAttrib(GLMContext ctx,
                                       VertexArray *vao,
                                       uint64_t drawCall,
                                       GLuint programName,
                                       const uint8_t *indexBytes,
                                       GLenum indexType,
                                       NSUInteger indexElement,
                                       GLint baseVertex,
                                       GLuint attrib,
                                       bool traceFile)
{
    if (!ctx || !vao || attrib >= MAX_ATTRIBS ||
        (vao->enabled_attribs & (0x1u << attrib)) == 0u) {
        return;
    }

    MGLResolvedVertexAttribBinding resolved = {0};
    if (!mglRendererResolveVertexAttribBinding(ctx,
                                               vao,
                                               attrib,
                                               "drawElements.attrib",
                                               &resolved)) {
        mglTraceLogNSString(@"MGL TRACE drawElements.attrib%u call=%llu program=%u invalid buffer",
              (unsigned)attrib,
              (unsigned long long)drawCall,
              (unsigned)programName);
        if (traceFile && mglTraceLogIsEnabled()) {
            mglTraceLog("VATTR_SAMPLE call=%llu program=%u attrib=%u reason=invalid_buffer",
                        (unsigned long long)drawCall,
                        (unsigned)programName,
                        (unsigned)attrib);
        }
        return;
    }
    const VertexAttrib *a = resolved.attrib;
    Buffer *vbo = resolved.buffer;

    const uint8_t *vboBytes = NULL;
    if (vbo->data.buffer_data && ((uintptr_t)vbo->data.buffer_data >= 0x1000ull)) {
        vboBytes = (const uint8_t *)vbo->data.buffer_data;
    } else if (vbo->data.mtl_data) {
        id<MTLBuffer> vb = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
        vboBytes = (const uint8_t *)vb.contents;
    }

    if (!vboBytes) {
        mglTraceLogNSString(@"MGL TRACE drawElements.attrib%u call=%llu program=%u vbo=%u no readable bytes",
              (unsigned)attrib,
              (unsigned long long)drawCall,
              (unsigned)programName,
              (unsigned)vbo->name);
        if (traceFile && mglTraceLogIsEnabled()) {
            mglTraceLog("VATTR_SAMPLE call=%llu program=%u attrib=%u vbo=%u reason=no_readable_bytes",
                        (unsigned long long)drawCall,
                        (unsigned)programName,
                        (unsigned)attrib,
                        (unsigned)vbo->name);
        }
        return;
    }

    uint32_t firstIndex = mglReadGLIndexValue(indexBytes, indexType, indexElement);
    int64_t vertexIndex64 = (int64_t)firstIndex + (int64_t)baseVertex;
    if (vertexIndex64 < 0) {
        mglTraceLogNSString(@"MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu vbo=%u negative vertexIndex rawIndex=%u baseVertex=%d",
              (unsigned)attrib,
              (unsigned long long)drawCall,
              (unsigned)programName,
              (unsigned long)indexElement,
              (unsigned)vbo->name,
              (unsigned)firstIndex,
              (int)baseVertex);
        if (traceFile && mglTraceLogIsEnabled()) {
            mglTraceLog("VATTR_SAMPLE call=%llu program=%u attrib=%u indexElement=%lu vbo=%u rawIndex=%u baseVertex=%d reason=negative_vertex_index",
                        (unsigned long long)drawCall,
                        (unsigned)programName,
                        (unsigned)attrib,
                        (unsigned long)indexElement,
                        (unsigned)vbo->name,
                        (unsigned)firstIndex,
                        (int)baseVertex);
        }
        return;
    }
    NSUInteger vertexIndex = (NSUInteger)vertexIndex64;
    NSUInteger bindingOffset = (resolved.binding_offset > 0) ? (NSUInteger)resolved.binding_offset : 0u;
    NSUInteger relativeOffset = (resolved.relativeoffset > 0) ? (NSUInteger)resolved.relativeoffset : 0u;
    NSUInteger stride = (resolved.stride > 0u) ? (NSUInteger)resolved.stride : mglVertexAttribElementBytes(a->type, a->size);
    NSUInteger vertexOffset = bindingOffset + relativeOffset + (vertexIndex * stride);
    size_t elemBytes = mglVertexAttribElementBytes(a->type, a->size);
    GLboolean effectiveNormalized = a->normalized;
    Program *program = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (!effectiveNormalized &&
        a->type == GL_UNSIGNED_BYTE &&
        a->size == 4 &&
        mglRendererVertexAttribIsColorInput(program, attrib)) {
        effectiveNormalized = GL_TRUE;
    }

    if (elemBytes == 0u ||
        vertexOffset > (NSUInteger)vbo->size ||
        ((NSUInteger)vbo->size - vertexOffset) < elemBytes) {
        mglTraceLogNSString(@"MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu vbo=%u OOB rawIndex=%u baseVertex=%d vertexIndex=%llu bindingOffset=%lu relOffset=%lu stride=%lu size=%u type=0x%x normalized=%u elemBytes=%zu vboSize=%lld",
              (unsigned)attrib,
              (unsigned long long)drawCall,
              (unsigned)programName,
              (unsigned long)indexElement,
              (unsigned)vbo->name,
              (unsigned)firstIndex,
              (int)baseVertex,
              (unsigned long long)vertexIndex64,
              (unsigned long)bindingOffset,
              (unsigned long)relativeOffset,
              (unsigned long)stride,
              (unsigned)a->size,
              (unsigned)a->type,
              (unsigned)a->normalized,
              elemBytes,
              (long long)vbo->size);
        if (traceFile && mglTraceLogIsEnabled()) {
            mglTraceLog("VATTR_SAMPLE call=%llu program=%u attrib=%u indexElement=%lu vbo=%u rawIndex=%u baseVertex=%d vertexIndex=%llu bindingOffset=%lu relOffset=%lu stride=%lu size=%u type=0x%x normalized=%u elemBytes=%zu vboSize=%lld reason=oob",
                        (unsigned long long)drawCall,
                        (unsigned)programName,
                        (unsigned)attrib,
                        (unsigned long)indexElement,
                        (unsigned)vbo->name,
                        (unsigned)firstIndex,
                        (int)baseVertex,
                        (unsigned long long)vertexIndex64,
                        (unsigned long)bindingOffset,
                        (unsigned long)relativeOffset,
                        (unsigned long)stride,
                        (unsigned)a->size,
                        (unsigned)a->type,
                        (unsigned)a->normalized,
                        elemBytes,
                        (long long)vbo->size);
        }
        return;
    }

    const uint8_t *attribBytes = vboBytes + vertexOffset;
    double comps[4] = {0.0, 0.0, 0.0, 0.0};
    for (NSUInteger c = 0; c < MIN((NSUInteger)a->size, (NSUInteger)4); c++) {
        comps[c] = mglDecodeVertexAttribComponent(attribBytes, a->type, effectiveNormalized, c);
    }

    char raw[3 * 16 + 1] = {0};
    size_t rawLen = MIN((size_t)16u, elemBytes);
    size_t rawPos = 0u;
    for (size_t i = 0; i < rawLen && rawPos + 3u < sizeof(raw); i++) {
        int wrote = snprintf(raw + rawPos,
                             sizeof(raw) - rawPos,
                             "%02x%s",
                             attribBytes[i],
                             (i + 1u < rawLen) ? ":" : "");
        if (wrote <= 0) {
            break;
        }
        rawPos += (size_t)wrote;
    }
    MTLVertexFormat format = glTypeSizeToMtlType(a->type, a->size, effectiveNormalized);
    int mappedIndex = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, attrib, "drawElements.attrib.trace");
    MGLShaderResource *resource = mglRendererProgramVertexAttribResource(program, attrib);
    mglTraceLogNSString(@"MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu resource=%s metalSlot=%d vbo=%u rawIndex=%u baseVertex=%d vertexIndex=%llu bindingIndex=%u bindingOffset=%lu relOffset=%lu vertexOffset=%lu stride=%lu size=%u type=0x%x normalized=%u/%u format=%lu(%s) decoded=(%.6f,%.6f,%.6f,%.6f) raw=%s",
          (unsigned)attrib,
          (unsigned long long)drawCall,
          (unsigned)programName,
          (unsigned long)indexElement,
          resource && resource->name ? resource->name : "(unknown)",
          mappedIndex,
          (unsigned)vbo->name,
          (unsigned)firstIndex,
          (int)baseVertex,
          (unsigned long long)vertexIndex64,
          (unsigned)resolved.binding_index,
          (unsigned long)bindingOffset,
          (unsigned long)relativeOffset,
          (unsigned long)vertexOffset,
          (unsigned long)stride,
          (unsigned)a->size,
          (unsigned)a->type,
          (unsigned)a->normalized,
          (unsigned)effectiveNormalized,
          (unsigned long)format,
          mglVertexFormatName(format),
          comps[0], comps[1], comps[2], comps[3],
          raw);
    if (traceFile && mglTraceLogIsEnabled()) {
        mglTraceLog("VATTR_SAMPLE call=%llu program=%u attrib=%u indexElement=%lu resource=%s metalSlot=%d vbo=%u rawIndex=%u baseVertex=%d vertexIndex=%llu bindingIndex=%u bindingOffset=%lu relOffset=%lu vertexOffset=%lu stride=%lu size=%u type=0x%x normalized=%u/%u format=%lu(%s) decoded=(%.6f,%.6f,%.6f,%.6f) raw=%s",
                    (unsigned long long)drawCall,
                    (unsigned)programName,
                    (unsigned)attrib,
                    (unsigned long)indexElement,
                    resource && resource->name ? resource->name : "(unknown)",
                    mappedIndex,
                    (unsigned)vbo->name,
                    (unsigned)firstIndex,
                    (int)baseVertex,
                    (unsigned long long)vertexIndex64,
                    (unsigned)resolved.binding_index,
                    (unsigned long)bindingOffset,
                    (unsigned long)relativeOffset,
                    (unsigned long)vertexOffset,
                    (unsigned long)stride,
                    (unsigned)a->size,
                    (unsigned)a->type,
                    (unsigned)a->normalized,
                    (unsigned)effectiveNormalized,
                    (unsigned long)format,
                    mglVertexFormatName(format),
                    comps[0], comps[1], comps[2], comps[3],
                    raw);
    }
}

void mglTraceReplayCommandVertexAttribSamples(GLMContext traceCtx,
                                                     Program *program,
                                                     const MGLDrawCommand *cmd,
                                                     Buffer *ebo,
                                                     uint64_t flushId,
                                                     uint32_t batchIndex,
                                                     uint32_t commandIndex,
                                                     bool forceTrace)
{
    if (!mglTraceLogIsEnabled() ||
        !traceCtx ||
        !program ||
        !cmd ||
        !ebo ||
        !mglDrawCommandUsesElements(cmd) ||
        cmd->count <= 0) {
        return;
    }

    if (!forceTrace && !mglProgramNeedsTraceLog(program)) {
        return;
    }

    static uint64_t s_replayAttribSampleLogs = 0;
    if (!forceTrace && !mglShouldLogFocusedBinding(&s_replayAttribSampleLogs)) {
        return;
    }

    const uint8_t *indexBytes = NULL;
    NSUInteger indexBytesAvailable = 0u;
    if (ebo->data.buffer_data && ((uintptr_t)ebo->data.buffer_data >= 0x1000ull)) {
        indexBytes = (const uint8_t *)ebo->data.buffer_data;
        indexBytesAvailable = (ebo->size > 0) ? (NSUInteger)ebo->size : 0u;
    } else if (ebo->data.mtl_data) {
        id<MTLBuffer> indexBuffer = (__bridge id<MTLBuffer>)(ebo->data.mtl_data);
        if (indexBuffer && indexBuffer.contents) {
            indexBytes = (const uint8_t *)indexBuffer.contents;
            indexBytesAvailable = indexBuffer.length;
        }
    }

    if (!indexBytes) {
        mglTraceLog("VATTR_REPLAY_BEGIN flush=%llu batch=%u cmd=%u program=%u type=%s count=%d indexType=0x%x indexOffset=%u baseVertex=%d ebo=%u reason=no_index_bytes",
                    (unsigned long long)flushId,
                    (unsigned)batchIndex,
                    (unsigned)commandIndex,
                    (unsigned)program->name,
                    mglDrawCommandTypeName(cmd->type),
                    (int)cmd->count,
                    (unsigned)cmd->indexType,
                    (unsigned)cmd->indexBufferOffset,
                    (int)cmd->baseVertex,
                    (unsigned)ebo->name);
        return;
    }

    NSUInteger indexOffset = (NSUInteger)cmd->indexBufferOffset;
    NSUInteger indexStride = mglGLIndexElementSize(cmd->indexType);
    if (indexStride == 0u ||
        indexOffset > indexBytesAvailable ||
        indexBytesAvailable - indexOffset < indexStride) {
        mglTraceLog("VATTR_REPLAY_BEGIN flush=%llu batch=%u cmd=%u program=%u type=%s count=%d indexType=0x%x indexOffset=%u baseVertex=%d ebo=%u available=%lu reason=index_oob",
                    (unsigned long long)flushId,
                    (unsigned)batchIndex,
                    (unsigned)commandIndex,
                    (unsigned)program->name,
                    mglDrawCommandTypeName(cmd->type),
                    (int)cmd->count,
                    (unsigned)cmd->indexType,
                    (unsigned)cmd->indexBufferOffset,
                    (int)cmd->baseVertex,
                    (unsigned)ebo->name,
                    (unsigned long)indexBytesAvailable);
        return;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(traceCtx, "replay.attrib.trace");
    if (!vao) {
        mglTraceLog("VATTR_REPLAY_BEGIN flush=%llu batch=%u cmd=%u program=%u type=%s count=%d indexType=0x%x indexOffset=%u baseVertex=%d ebo=%u reason=no_vao",
                    (unsigned long long)flushId,
                    (unsigned)batchIndex,
                    (unsigned)commandIndex,
                    (unsigned)program->name,
                    mglDrawCommandTypeName(cmd->type),
                    (int)cmd->count,
                    (unsigned)cmd->indexType,
                    (unsigned)cmd->indexBufferOffset,
                    (int)cmd->baseVertex,
                    (unsigned)ebo->name);
        return;
    }

    const uint8_t *start = indexBytes + indexOffset;
    uint32_t firstIndex = mglReadGLIndexValue(start, cmd->indexType, 0u);
    mglTraceLog("VATTR_REPLAY_BEGIN flush=%llu batch=%u cmd=%u program=%u type=%s count=%d indexType=0x%x indexOffset=%u baseVertex=%d firstIndex=%u ebo=%u vao=%p enabled=0x%x forceRTCopy=%d",
                (unsigned long long)flushId,
                (unsigned)batchIndex,
                (unsigned)commandIndex,
                (unsigned)program->name,
                mglDrawCommandTypeName(cmd->type),
                (int)cmd->count,
                (unsigned)cmd->indexType,
                (unsigned)cmd->indexBufferOffset,
                (int)cmd->baseVertex,
                (unsigned)firstIndex,
                (unsigned)ebo->name,
                vao,
                (unsigned)vao->enabled_attribs,
                forceTrace ? 1 : 0);

    NSUInteger sampleCount = forceTrace ? MIN((NSUInteger)cmd->count, (NSUInteger)6u) : (NSUInteger)1u;
    GLuint traceAttribLimit = MIN((GLuint)6u, traceCtx->state.max_vertex_attribs);
    for (NSUInteger sample = 0; sample < sampleCount; sample++) {
        if (indexBytesAvailable - indexOffset < ((sample + 1u) * indexStride)) {
            break;
        }
        for (GLuint attrib = 0; attrib < traceAttribLimit; attrib++) {
            if (!mglRendererProgramUsesVertexAttrib(program, attrib)) {
                continue;
            }
            mglTraceDrawElementsAttrib(traceCtx,
                                       vao,
                                       flushId,
                                       program->name,
                                       start,
                                       cmd->indexType,
                                       sample,
                                       cmd->baseVertex,
                                       attrib,
                                       true);
        }
    }
}

#pragma mark debug code
void printDirtyBit(unsigned dirty_bits, unsigned dirty_flag, const char *name)
{
    if (dirty_bits & dirty_flag)
        DEBUG_PRINT("%s", name);
}

void logDirtyBits(GLMContext ctx)
{
    if(ctx->active_state->dirty_bits)
    {
        if (ctx->active_state->dirty_bits & DIRTY_ALL_BIT)
        {
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_ALL_BIT, "DIRTY_ALL_BIT set");
        }
        else
        {
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_VAO, "DIRTY_VAO ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_STATE, "DIRTY_STATE ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_BUFFER, "DIRTY_BUFFER ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_TEX, "DIRTY_TEX ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_TEX_PARAM, "DIRTY_TEX_PARAM ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_TEX_BINDING, "DIRTY_TEX_BINDING ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_SAMPLER, "DIRTY_SAMPLER ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_SHADER, "DIRTY_SHADER ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_PROGRAM, "DIRTY_PROGRAM ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_FBO, "DIRTY_FBO ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_DRAWABLE, "DIRTY_DRAWABLE ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_RENDER_STATE, "DIRTY_RENDER_STATE ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_ALPHA_STATE, "DIRTY_ALPHA_STATE ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_IMAGE_UNIT_STATE, "DIRTY_IMAGE_UNIT_STATE ");
            printDirtyBit(ctx->active_state->dirty_bits, DIRTY_BUFFER_BASE_STATE, "DIRTY_BUFFER_BASE_STATE ");
        }
        DEBUG_PRINT("\n");
    }
}


#pragma mark textures

/* mglStoredColorComponentsForTexture and mglMTLSwizzleForGLSwizzle now live
 * in mgl_texture_compat.m — see mgl_texture_compat.h. */


/*
 * GL_TEXTURE_BASE_LEVEL / MAX_LEVEL select the mip window used for sampling.
 * Metal textures always start at level 0, so when that window is narrower
 * than the full mip chain a texture view is created spanning
 * [base_level, max_level] (including base_level==0 with a restricted
 * MAX_LEVEL).  This lets Metal sampler lod clamps operate in the same
 * coordinate space as GL (relative to the view's level 0).  When the window
 * covers the whole texture the original is returned (no overhead).
 */
/* mglSampledTextureViewForBaseLevel now lives in mgl_texture_compat.m — see
 * mgl_texture_compat.h. */

/* bindTexturesToCurrentRenderEncoder moved to MGLRenderer+Draw.m */

#pragma mark framebuffers

/* isColorAttachment, getFBOAttachment, findTexture declared in MGLRenderer_Private.h */


/* mtlInvalidateRenderPass: moved to MGLRenderer+RenderPass.m */

/* framebufferAttachmentTexture: moved to MGLRenderer+RenderPass.m */

/* static bool mglRendererProgramHasSampledResourceNamed moved to MGLRenderer+Draw.m */

/* markCurrentFramebufferColorAttachmentWrittenAtIndex:(GLuint)attachmentIndex moved to MGLRenderer+Draw.m */

/* markCurrentFramebufferDrawAttachmentsWritten moved to MGLRenderer+Draw.m */

/* recordArrayDrawSubmittedMode:(GLenum)mode vertexCount:(uint64_t)vertexCount moved to MGLRenderer+Draw.m */

/* recordElementDrawSubmittedMode:(GLenum)mode indexCount:(uint64_t)indexCount moved to MGLRenderer+Draw.m */

/* currentRenderPassMatchesCurrentFramebuffer moved to MGLRenderer+RenderPass.m */

/* ensureCurrentRenderPassMatchesFramebufferForDraw moved to MGLRenderer+RenderPass.m */

/* endRenderPassIfFramebufferChangedForNonDraw: moved to MGLRenderer+RenderPass.m */

/* bindMTLTexture: moved to MGLRenderer+RenderPass.m */

/* bindMTLTextureLocked: moved to MGLRenderer+RenderPass.m */

/* bindActiveTexturesToMTL moved to MGLRenderer+Draw.m */

/* restoreRenderEncoderAfterTextureUploadForDraw: moved to MGLRenderer+RenderPass.m */

/* bindFramebufferTexture:isDrawBuffer: moved to MGLRenderer+RenderPass.m */

/* getProgramBinding* / getProgramExpectedTexture* / getProgramLocation moved
 * to MGLRenderer+ProgramBinding.m */

/* invalidateCurrentPipelineStateForReason: moved to MGLRenderer+RenderPass.m */

/* bindMTLProgram: moved to MGLRenderer+RenderPass.m */

/* mglGeometryShaderIsPassthrough moved to MGLRenderer+RenderPass.m (static helper) */

/* bindMTLProgramLocked: moved to MGLRenderer+RenderPass.m */

#pragma mark draw buffers
/* AppKit-backed drawable-size hand-off.  The GL thread never touches
 * NSView/NSWindow/NSScreen; it only consumes the atomic snapshot published by
 * mglMainThreadSyncViewGeometry (see MGLRenderer+Lifecycle.m) and sets
 * CAMetalLayer.drawableSize, which Metal explicitly allows off the main
 * thread. */
- (CGSize)mglApplyPendingDrawableSize
{
    MGL_ASSERT_GL_THREAD();
    if (atomic_exchange_explicit(&_drawableSizeDirty, false, memory_order_acquire)) {
        uint32_t w = atomic_load_explicit(&_pendingDrawableW, memory_order_relaxed);
        uint32_t h = atomic_load_explicit(&_pendingDrawableH, memory_order_relaxed);
        CGSize s = CGSizeMake((CGFloat)MAX(1u, w), (CGFloat)MAX(1u, h));
        _layer.drawableSize = s;
        return s;
    }
    return _layer.drawableSize;
}

- (BOOL)mglEnsureLayerDrawableSizeAtLeastWidth:(NSUInteger)requiredWidth
                                        height:(NSUInteger)requiredHeight
                                        reason:(const char *)reason
{
    if (!_layer || requiredWidth == 0 || requiredHeight == 0) {
        return NO;
    }

    CGSize viewDrawableSize = [self mglApplyPendingDrawableSize];
    NSUInteger targetWidth = MAX(requiredWidth, (NSUInteger)MAX(1.0, viewDrawableSize.width));
    NSUInteger targetHeight = MAX(requiredHeight, (NSUInteger)MAX(1.0, viewDrawableSize.height));
    CGSize oldDrawableSize = _layer.drawableSize;

    if ((NSUInteger)oldDrawableSize.width == targetWidth &&
        (NSUInteger)oldDrawableSize.height == targetHeight) {
        return NO;
    }

    _layer.drawableSize = CGSizeMake((CGFloat)targetWidth, (CGFloat)targetHeight);
    if (_drawable) {
        _drawable = nil;
    }

    static uint64_t s_forcedDrawableResizeCount = 0;
    uint64_t hit = ++s_forcedDrawableResizeCount;
    if (hit <= 32ull || (hit % 120ull) == 0ull) {
        NSLog(@"MGL SIZE force drawable reason=%s hit=%llu required=%lux%lu viewSync=%.0fx%.0f old=%.0fx%.0f new=%lux%lu",
              reason ? reason : "unknown",
              (unsigned long long)hit,
              (unsigned long)requiredWidth,
              (unsigned long)requiredHeight,
              viewDrawableSize.width,
              viewDrawableSize.height,
              oldDrawableSize.width,
              oldDrawableSize.height,
              (unsigned long)targetWidth,
              (unsigned long)targetHeight);
    }

    return YES;
}

- (id)newDrawBuffer:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil
{
    id<MTLTexture> texture;
    MTLTextureDescriptor *tex_desc;
    CGSize drawableSize;

    if (!_layer) {
        NSLog(@"MGL DRAWBUFFER ERROR: cannot create draw buffer without CAMetalLayer");
        return nil;
    }
    drawableSize = [self mglApplyPendingDrawableSize];

    tex_desc = [[MTLTextureDescriptor alloc] init];
    if (!tex_desc) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to allocate draw buffer descriptor");
        return nil;
    }
    tex_desc.width = (NSUInteger)MAX(1.0, drawableSize.width);
    tex_desc.height = (NSUInteger)MAX(1.0, drawableSize.height);
    tex_desc.pixelFormat = pixelFormat;
    tex_desc.usage = MTLTextureUsageRenderTarget;

    if (depthStencil)
    {
        tex_desc.storageMode = MTLStorageModePrivate;
    }

    texture = mglRendererCreateTexture(_device, tex_desc);
    if (!texture) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to create draw buffer texture format=%lu size=%lux%lu",
              (unsigned long)pixelFormat,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height);
        return nil;
    }

    return texture;
}

- (id)newDrawBufferWithCustomSize:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil customSize:(CGSize)size
{
    id<MTLTexture> texture;
    MTLTextureDescriptor *tex_desc;

    tex_desc = [[MTLTextureDescriptor alloc] init];
    if (!tex_desc) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to allocate custom draw buffer descriptor");
        return nil;
    }
    tex_desc.width = (NSUInteger)MAX(1.0, size.width);
    tex_desc.height = (NSUInteger)MAX(1.0, size.height);
    tex_desc.pixelFormat = pixelFormat;
    tex_desc.usage = MTLTextureUsageRenderTarget;

    if (depthStencil)
    {
        tex_desc.storageMode = MTLStorageModePrivate;
    }

    texture = mglRendererCreateTexture(_device, tex_desc);
    if (!texture) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to create custom draw buffer texture format=%lu size=%lux%lu",
              (unsigned long)pixelFormat,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height);
        return nil;
    }

    return texture;
}

- (bool) checkDrawBufferSize:(GLuint) index;
{
    CGSize drawableSize;

    drawableSize = [self mglApplyPendingDrawableSize];

    if ((GLuint)drawableSize.width != _drawBuffers[index].width)
        return false;

    if ((GLuint)drawableSize.height != _drawBuffers[index].height)
        return false;

    return true;
}

#pragma mark render encoder and command buffer init code
- (MTLStencilOperation) mtlStencilOpForGLOp:(GLenum) op
{
    switch(op)
    {
        case GL_KEEP: return MTLStencilOperationKeep;
        case GL_ZERO: return MTLStencilOperationZero;
        case GL_REPLACE: return MTLStencilOperationReplace;
        case GL_INCR: return MTLStencilOperationIncrementClamp;
        case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
        case GL_DECR: return MTLStencilOperationDecrementClamp;
        case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
        case GL_INVERT: return MTLStencilOperationInvert;
        default:
            NSLog(@"MGL WARNING: Unknown stencil operation 0x%x, falling back to KEEP", op);
            return MTLStencilOperationKeep;
    }
}

/* updateCurrentRenderEncoder moved to MGLRenderer+RenderPass.m */

/* newRenderEncoder moved to MGLRenderer+RenderPass.m */

/* shouldUseDontCareLoadForColorTexture:firstUseThisFrame: moved to MGLRenderer+RenderPass.m */

/* newRenderEncoderLocked moved to MGLRenderer+RenderPass.m */

/* newCommandBuffer moved to MGLRenderer+RenderPass.m */

/* newCommandBufferLocked moved to MGLRenderer+RenderPass.m */

/* ensureWritableCommandBuffer: moved to MGLRenderer+RenderPass.m */

/* ensureWritableCommandBufferLocked: moved to MGLRenderer+RenderPass.m */

/*
 * copyTextureUploadWithDedicatedCommandBuffer:... — texture upload blit path
 *
 * Texture upload is a GL command and must preserve call ordering with draws on the
 * same context; an independent CB must not leapfrog an uncommitted render CB
 * (otherwise the upload may complete before an already-encoded draw, breaking GL implicit ordering).
 *
 * Default mode (!kMGLUseDedicatedTextureUploadCommandBuffer): endRenderEncoding closes the open
 *   render encoder, then encodes the blit (copyFromBuffer:toTexture:) on the current CB, ensuring
 *   GPU-side ordering between the upload and draws within the same CB.
 * Dedicated mode (kMGLUseDedicatedTextureUploadCommandBuffer): encodes the blit on an independent CB,
 *   optionally using addCompletedHandler + semaphore for synchronous wait; this mode is only enabled
 *   when asynchronous upload is genuinely required.
 */

/*
 * uploadTextureSliceViaBlit:... — single-slice texture upload dispatch
 *
 * Selects the upload path based on Metal texture type:
 *   - 1D / 1DArray: low frequency, uses replaceRegion (see the 1D branch comment below).
 *   - 3D: uses replaceRegion to avoid the AGX driver's copyFromBuffer slice OOB assertion (see the 3D branch comment below).
 *   - 2D / Array / Cube: must not use replaceRegion (unsafe when sampled by an in-flight CB); must take the blit
 *     path (allocates uploadBuffer below and calls copyTextureUploadWithDedicatedCommandBuffer),
 *     relying on GPU-side CB ordering to guarantee visibility ordering between upload and sampling.
 * A replaceRegion failure for 1D/3D falls back to the blit path.
 */


/* newCommandBufferAndRenderEncoder moved to MGLRenderer+RenderPass.m */

/* generatePipelineDescriptor moved to MGLRenderer+RenderPass.m */

/* generateVertexDescriptor moved to MGLRenderer+RenderPass.m */

#pragma mark utility funcs for processGLState
- (MTLBlendFactor) blendFactorFromGL:(GLenum)gl_blend
{
    MTLBlendFactor factor;

    switch(gl_blend)
    {
        case GL_ZERO: factor = MTLBlendFactorZero; break;
        case GL_ONE: factor = MTLBlendFactorOne; break;
        case GL_SRC_COLOR: factor = MTLBlendFactorSourceColor; break;
        case GL_ONE_MINUS_SRC_COLOR: factor = MTLBlendFactorOneMinusSourceColor; break;
        case GL_DST_COLOR: factor = MTLBlendFactorDestinationColor; break;
        case GL_ONE_MINUS_DST_COLOR: factor = MTLBlendFactorOneMinusDestinationColor; break;
        case GL_SRC_ALPHA: factor = MTLBlendFactorSourceAlpha; break;
        case GL_ONE_MINUS_SRC_ALPHA: factor = MTLBlendFactorOneMinusSourceAlpha; break;
        case GL_DST_ALPHA: factor = MTLBlendFactorDestinationAlpha; break;
        case GL_ONE_MINUS_DST_ALPHA: factor = MTLBlendFactorOneMinusDestinationAlpha; break;
        case GL_CONSTANT_COLOR: factor = MTLBlendFactorBlendColor; break;
        case GL_ONE_MINUS_CONSTANT_COLOR: factor = MTLBlendFactorOneMinusBlendColor; break;
        case GL_CONSTANT_ALPHA: factor = MTLBlendFactorBlendAlpha; break;
        case GL_ONE_MINUS_CONSTANT_ALPHA: factor = MTLBlendFactorOneMinusBlendAlpha; break;
        case GL_SRC_ALPHA_SATURATE: factor = MTLBlendFactorSourceAlphaSaturated; break;
        /* Dual-source blend factors (GL 4.0+, requires dualSourceBlendingEnabled) */
        case GL_SRC1_COLOR: factor = MTLBlendFactorSource1Color; break;
        case GL_ONE_MINUS_SRC1_COLOR: factor = MTLBlendFactorOneMinusSource1Color; break;
        case GL_SRC1_ALPHA: factor = MTLBlendFactorSource1Alpha; break;
        case GL_ONE_MINUS_SRC1_ALPHA: factor = MTLBlendFactorOneMinusSource1Alpha; break;

        default:
            // CRITICAL FIX: Handle assertion gracefully instead of crashing
            static uint64_t s_unknownBlendFactorCount = 0;
            uint64_t hit = ++s_unknownBlendFactorCount;
            if (hit <= 32 || (hit % 512) == 0) {
                NSLog(@"MGL ERROR: Unknown blend factor 0x%x hit=%llu",
                      gl_blend, (unsigned long long)hit);
            }
            return MTLBlendFactorZero;
    }

    return factor;
}

- (MTLBlendOperation) blendOperationFromGL:(GLenum)gl_blend_op
{
    MTLBlendOperation op;

    switch(gl_blend_op)
    {
        case GL_FUNC_ADD: op = MTLBlendOperationAdd; break;
        case GL_FUNC_SUBTRACT: op = MTLBlendOperationSubtract; break;
        case GL_FUNC_REVERSE_SUBTRACT: op = MTLBlendOperationReverseSubtract; break;
        case GL_MIN: op = MTLBlendOperationMin; break;
        case GL_MAX: op = MTLBlendOperationMax; break;

        default:
            // CRITICAL FIX: Handle assertion gracefully instead of crashing
            static uint64_t s_unknownBlendOperationCount = 0;
            uint64_t hit = ++s_unknownBlendOperationCount;
            if (hit <= 32 || (hit % 512) == 0) {
                NSLog(@"MGL ERROR: Unknown blend operation 0x%x hit=%llu",
                      gl_blend_op, (unsigned long long)hit);
            }
            return MTLBlendOperationAdd;
    }

    return op;
}

/* updateBlendStateCache moved to MGLRenderer+RenderPass.m */

/* bindBlendStateToPipelineStateDescriptor: moved to MGLRenderer+RenderPass.m */

/* bindFramebufferAttachmentTextures moved to MGLRenderer+RenderPass.m */

/* updateGLSampledCopiesForEndedRenderPassFramebuffer:drawCount:drawBuffers:reason: moved to MGLRenderer+RenderPass.m */

/* endRenderEncoding moved to MGLRenderer+RenderPass.m */

/* endRenderEncodingLocked moved to MGLRenderer+RenderPass.m */

/* currentRenderPassUsesTexture: moved to MGLRenderer+RenderPass.m */

/* synchronizeRenderPassForTextureReadback:reason: moved to MGLRenderer+RenderPass.m */

/* emergencyResetMetalState moved to MGLRenderer+RenderPass.m */

#pragma mark ------------------------------------------------------------------------------------------
#pragma mark processGLState for resolving opengl state into metal state
#pragma mark ------------------------------------------------------------------------------------------

/* Invalidate all last-bound render encoder state. Called whenever the
 * encoder is recreated or ended so the next bind is not incorrectly skipped
 * by the dedup fast path. */
/* invalidateLastBoundState moved to MGLRenderer+Draw.m */

/* recordLastBoundVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* recordLastBoundFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* setVertexTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* setFragmentTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* setVertexSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* setFragmentSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index moved to MGLRenderer+Draw.m */

/* setViewportIfNeeded:(MTLViewport)viewport moved to MGLRenderer+Draw.m */

/* setScissorRectIfNeeded:(MTLScissorRect)rect moved to MGLRenderer+Draw.m */

/* setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode moved to MGLRenderer+Draw.m */

/* processGLState: moved to MGLRenderer+RenderPass.m */

/* processGLStateLocked: moved to MGLRenderer+RenderPass.m */

/*
 * Resource Sync domain (Resource Sync domain). "Stability rebind" before draw: command buffer rotation /
 * encoder reconstruction discards latched bindings, so before each draw vertex/fragment
 * buffers, buffer-size constants, active textures and sampled textures are remapped and rebound.
 * Only Metal encoder bindings are touched; state is read via glm_ctx (unchanged from before extraction).
 * Returns false to indicate that this draw should be skipped (semantically equivalent to the
 * original inline return false).
 */
/* syncResourceBindingsForContext:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* syncPipelineStateWithDeferredBufferMap: moved to MGLRenderer+RenderPass.m */

/* bindBufferSizeConstantsForRenderEncoder moved to MGLRenderer+RenderPass.m */

-(bool) processBuffer:(Buffer*)ptr
{
    if (ptr == NULL)
    {
        NSLog(@"Error: processBuffer failed\n");

        return false;
    }

    if (ptr->data.mtl_data == NULL)
    {
        [self bindMTLBuffer: ptr];
        RETURN_FALSE_ON_NULL(ptr->data.mtl_data);
    }

    if (ptr->data.dirty_bits)
    {
        [self updateDirtyBuffer: ptr];
    }

    return true;
}
/* flushCommandBuffer: moved to MGLRenderer+RenderPass.m */

/* flushCommandBufferLocked: moved to MGLRenderer+RenderPass.m */
#pragma mark C interface to mtlDeleteMTLObj
-(void) mtlDeleteMTLObj:(GLMContext) glm_ctx buffer: (void *)obj
{
    METAL_LOCK();
    [self mtlDeleteMTLObjLocked:glm_ctx buffer:obj];
    METAL_UNLOCK();
}

-(void) mtlDeleteMTLObjLocked:(GLMContext) glm_ctx buffer: (void *)obj
{
    if (!obj)
        return;

    // Do not force-flush per-object destruction.
    // Metal command buffers retain referenced resources, so immediate release is safe and
    // avoids shutdown-time command-buffer storms (one commit per deleted object).
    CFBridgingRelease(obj);
}


#pragma mark Draw command buffer flush

/*
 * Restore GL state from a batch state key and set appropriate dirty bits
 * so that the next processGLState / draw call picks up the right state.
 */
/* restoreStateFromKey:(const MGLStateKey *)key context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* traceReplayBatch:(MGLDrawBatch *)batch moved to MGLRenderer+Draw.m */

/* traceReplayCommand:(MGLDrawBatch *)batch moved to MGLRenderer+Draw.m */

/* flushDrawBuffer:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* restoreStateForBatch:(MGLDrawBatch *)batch moved to MGLRenderer+Draw.m */

/* teardownBatchReplayForContext:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/*
 * RenderPass Sync domain (RenderPass Sync domain).
 *
 * Maps a DIRTY_FBO transition onto the Metal render pass: if the current
 * encoder already targets the bound framebuffer nothing changes (dirty bit
 * cleared); otherwise attachment textures are (re)bound and the encoder is
 * rotated. Callers gate on DIRTY_FBO before invoking. This is the single
 * owner of FBO-driven encoder rotation — processGLState no longer inlines it.
 */
/* syncRenderPassStateForContext: moved to MGLRenderer+RenderPass.m */

/* rotateRenderEncoderForCurrentFramebufferLocked moved to MGLRenderer+RenderPass.m */

/* prepareRenderPassIfFBOChanged:context:replayError: moved to MGLRenderer+RenderPass.m */

/* checkBatchShouldExecute:(MGLDrawBatch *)batch moved to MGLRenderer+Draw.m */

/* recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueStreamMergedBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* mdiArgumentScratchBufferWithLength:(NSUInteger)length moved to MGLRenderer+Draw.m */

/* issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

#pragma mark C interface to mtlFlush
-(void) mtlFlush:(GLMContext) glm_ctx finish:(bool)finish
{
    [self flushCommandBuffer: finish];
}

#pragma mark C interface to mtlSwapBuffers
-(void) mtlSwapBuffers:(GLMContext) glm_ctx
{
    mglClaimGLThread();            /* idempotent; the rendering loop is GL-thread */
    @autoreleasepool {
        METAL_LOCK();
        [self mtlSwapBuffersLocked:glm_ctx];
        METAL_UNLOCK();
    }
}

-(void) mtlSwapBuffersLocked:(GLMContext) glm_ctx
{
    static uint64_t s_swapCallCount = 0;
    static double s_swapLastCallTime = 0.0;
    static uint64_t s_swapLastCallCount = 0;
    /* Heartbeat diagnostics: written only on the main queue
     * (dispatch_async ping below), read only on the GL calling thread.
     * Diagnostic-only; a torn read is tolerated. */
    static volatile double s_mainThreadHeartbeatSeconds = 0.0;
    static volatile uint64_t s_mainThreadPingCount = 0;
    uint64_t swapCall = ++s_swapCallCount;
    double swapStartSeconds = mglTraceNowSeconds();
    uint64_t swapStartNS = mglTraceClockNS();
    bool traceSwap = mglShouldTraceCall(swapCall);
    mglTraceNoteFrameBoundary();
    MGL_FRAME_STORE(g_mglSwapCallCount, swapCall);
    /* advance the DontCare frame generation. Any color attachment
     * written before this point belongs to the previous frame, so its next
     * write this frame is a "first use" that may skip loading prior contents.
     * Skips 0 so a zero-initialized texture stamp never matches. */
    [_renderPassManager incrementDontCareFrameGenerationWithWrap];
    MGL_FRAME_STORE(g_mglLastSwapSeconds, swapStartSeconds);
    if (swapCall <= 20ull || (swapCall % 60ull) == 0ull) {
        mglTraceLog("SWAP_RENDERER_ENTRY call=%llu drawArraysSinceSwap=%llu drawElementsSinceSwap=%llu processDrawCallsSinceSwap=%llu",
                    (unsigned long long)swapCall,
                    (unsigned long long)MGL_FRAME_LOAD(g_mglDrawArraysSinceSwap),
                    (unsigned long long)MGL_FRAME_LOAD(g_mglDrawElementsSinceSwap),
                    (unsigned long long)MGL_FRAME_LOAD(g_mglProcessDrawCallsSinceSwap));
    }
    mglLogLoopHeartbeat("swap.loop",
                        swapCall,
                        swapStartSeconds,
                        &s_swapLastCallTime,
                        &s_swapLastCallCount,
                        0.25);

    if (!mglRendererContextLikelyValid(glm_ctx)) {
        NSLog(@"MGL CRITICAL: swap.begin invalid glm_ctx=%p", glm_ctx);
        return;
    }

    if (ctx != glm_ctx) {
        mglTraceLogNSString(@"MGL TRACE swap.contextSync old=%p new=%p", ctx, glm_ctx);
        ctx = glm_ctx;
    }

    GLMContext activeCtx = glm_ctx;
    GLenum drawBuffer = activeCtx->state.draw_buffer;
    bool shouldPresent = (drawBuffer != GL_NONE);
    if (traceSwap) {
        mglTraceLogNSString(@"MGL TRACE swap.begin call=%llu shouldPresent=%d draw_buffer=0x%x",
              (unsigned long long)swapCall, shouldPresent ? 1 : 0, (unsigned)drawBuffer);
        mglLogStateSnapshot("swap.enter",
                            activeCtx,
                            _renderPassManager.state->currentCommandBuffer,
                            _renderPassManager.state->currentRenderEncoder,
                            _renderPassManager.state->renderPassDescriptor,
                            _drawable);
    }

    // Main-thread responsiveness probe for beachball diagnostics.
    // Render thread periodically posts a ping to main queue; stale heartbeat means main thread is blocked.
    if (kMGLDiagnosticStateLogs &&
        (swapCall <= 20ull || (swapCall % 30ull) == 0ull)) {
        dispatch_async(dispatch_get_main_queue(), ^{
            s_mainThreadHeartbeatSeconds = mglTraceNowSeconds();
            s_mainThreadPingCount++;
        });

        double hb = s_mainThreadHeartbeatSeconds;
        if (hb > 0.0) {
            double lagMs = (swapStartSeconds - hb) * 1000.0;
            if (lagMs > 500.0) {
                mglTraceLogNSString(@"MGL TRACE mainthread.stall suspected lag=%.2fms swapCall=%llu pingCount=%llu",
                      lagMs,
                      (unsigned long long)swapCall,
                      (unsigned long long)s_mainThreadPingCount);
                if (traceSwap || (swapCall % 120ull) == 0ull) {
                    mglLogStateSnapshot("mainthread.stall.snapshot",
                                        activeCtx,
                                        _renderPassManager.state->currentCommandBuffer,
                                        _renderPassManager.state->currentRenderEncoder,
                                        _renderPassManager.state->renderPassDescriptor,
                                        _drawable);
                }
            } else if (traceSwap) {
                mglTraceLogNSString(@"MGL TRACE mainthread.heartbeat lag=%.2fms swapCall=%llu pingCount=%llu",
                      lagMs,
                      (unsigned long long)swapCall,
                      (unsigned long long)s_mainThreadPingCount);
            }
        } else if (traceSwap) {
            mglTraceLogNSString(@"MGL TRACE mainthread.heartbeat uninitialized swapCall=%llu", (unsigned long long)swapCall);
        }
    }

    if (kMGLDiagnosticStateLogs) {
        MGLSwapDrawCounters frameCounters = mglSnapshotSwapDrawCounters();
        mglResetSwapDrawCounters();

        uint64_t lastDrawArraysCall = MGL_FRAME_LOAD(g_mglLastDrawArraysCall);
        uint64_t lastDrawElementsCall = MGL_FRAME_LOAD(g_mglLastDrawElementsCall);
        double lastDrawArraysSeconds = MGL_FRAME_LOAD(g_mglLastDrawArraysSeconds);
        double lastDrawElementsSeconds = MGL_FRAME_LOAD(g_mglLastDrawElementsSeconds);
        GLuint lastDrawArraysProgram = MGL_FRAME_LOAD(g_mglLastDrawArraysProgram);
        GLuint lastDrawArraysMode = MGL_FRAME_LOAD(g_mglLastDrawArraysMode);
        GLsizei lastDrawArraysCount = MGL_FRAME_LOAD(g_mglLastDrawArraysCount);
        GLuint lastDrawElementsProgram = MGL_FRAME_LOAD(g_mglLastDrawElementsProgram);
        GLuint lastDrawElementsMode = MGL_FRAME_LOAD(g_mglLastDrawElementsMode);
        GLsizei lastDrawElementsCount = MGL_FRAME_LOAD(g_mglLastDrawElementsCount);
        double drawArraysAgeMs = (lastDrawArraysSeconds > 0.0)
            ? ((swapStartSeconds - lastDrawArraysSeconds) * 1000.0)
            : -1.0;
        double drawElementsAgeMs = (lastDrawElementsSeconds > 0.0)
            ? ((swapStartSeconds - lastDrawElementsSeconds) * 1000.0)
            : -1.0;
        BOOL hasFrameWork = (frameCounters.draw_arrays > 0 ||
                             frameCounters.draw_elements > 0 ||
                             frameCounters.draw_arrays_skipped > 0 ||
                             frameCounters.draw_elements_skipped > 0 ||
                             frameCounters.process_draw_calls > 0);
        if (traceSwap || hasFrameWork || swapCall <= 20ull || (swapCall % 20ull) == 0ull) {
            mglTraceLogNSString(@"MGL TRACE swap.drawActivity call=%llu processDrawCalls=%llu drawArrays=%llu verts=%llu "
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
                  (unsigned)lastDrawArraysProgram,
                  (unsigned)lastDrawArraysMode,
                  (int)lastDrawArraysCount,
                  drawArraysAgeMs,
                  (unsigned long long)lastDrawElementsCall,
                  (unsigned)lastDrawElementsProgram,
                  (unsigned)lastDrawElementsMode,
                  (int)lastDrawElementsCount,
                  drawElementsAgeMs);
        }
    }

    if (shouldPresent)
    {
        [self flushDrawBufferLocked:activeCtx];

        if (![self processGLStateLocked: false]) {
            static uint64_t s_swapProcessStateFailCount = 0;
            s_swapProcessStateFailCount++;
            if (s_swapProcessStateFailCount <= 16 || (s_swapProcessStateFailCount % 500) == 0) {
                NSLog(@"MGL WARNING: mtlSwapBuffers continuing despite processGLState failure (occurrence=%llu)",
                      (unsigned long long)s_swapProcessStateFailCount);
            }
        }

        [self endRenderEncodingLocked];

        /* Deferred device reset drain.  This is the only safe reset point: the
         * render encoder is closed and the command buffer has not been rebuilt
         * yet, so resetMetalState can swap the command queue / clear caches
         * without racing an active encoder.  The request flag is set by the
         * Metal completion handler (GPURecovery.m) via release-store. */
        if (atomic_exchange_explicit(&_deviceResetRequested, false, memory_order_acquire)) {
            [self resetMetalState];
        }

        if (![self ensureWritableCommandBufferLocked:"mtlSwapBuffers"]) {
            NSLog(@"MGL ERROR: Failed to obtain writable command buffer in mtlSwapBuffers");
            return;
        }

        if (_drawable == NULL)
        {
            if (traceSwap) {
                mglTraceLogNSString(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present", (unsigned long long)swapCall);
            }
            [self mglApplyPendingDrawableSize];
            _drawable = [_layer nextDrawable];
            if (traceSwap) {
                id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
                mglTraceLogNSString(@"MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present drawable=%p tex=%p size=%lux%lu",
                      (unsigned long long)swapCall,
                      _drawable,
                      tex,
                      (unsigned long)(tex ? tex.width : 0),
                      (unsigned long)(tex ? tex.height : 0));
            }
        }

        if (_drawable == NULL) {
            NSLog(@"MGL WARNING: Drawable is NULL in mtlSwapBuffers, getting new drawable");
            if (traceSwap) {
                mglTraceLogNSString(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present_retry", (unsigned long long)swapCall);
            }
            [self mglApplyPendingDrawableSize];
            _drawable = [_layer nextDrawable];
            if (traceSwap) {
                id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
                mglTraceLogNSString(@"MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present_retry drawable=%p tex=%p size=%lux%lu",
                      (unsigned long long)swapCall,
                      _drawable,
                      tex,
                      (unsigned long)(tex ? tex.width : 0),
                      (unsigned long)(tex ? tex.height : 0));
            }
            if (_drawable == NULL) {
                NSLog(@"MGL ERROR: Failed to obtain any drawable from Metal layer");
                return;
            }
        }

        id<MTLTexture> rpColor0 = mglRenderPassAttachmentTextureForState(
            _renderPassManager.state->renderPassDescriptor,
            _renderPassManager.state->renderPassStateOwner,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0);
        id<MTLTexture> drawableTexture = _drawable ? _drawable.texture : nil;
        [self copyRenderPassColorToDrawableIfNeeded:rpColor0 drawableTexture:drawableTexture swapCall:swapCall traceSwap:traceSwap];

        [self scheduleSwapTextureSampleDiagnostics:rpColor0 drawableTexture:drawableTexture swapCall:swapCall];

        if (_layer == NULL) {
            NSLog(@"MGL ERROR: Metal layer is NULL, cannot present drawable");
            return;
        }

        if (!_renderPassManager.state->currentCommandBuffer) {
            NSLog(@"MGL ERROR: No command buffer available for presentation");
            return;
        }

        MTLCommandBufferStatus bufferStatus =
            mglRenderCommandBufferStatus(
                _renderPassManager.state->currentCommandBuffer);
        if (bufferStatus != MTLCommandBufferStatusNotEnqueued) {
            static uint64_t s_swapFinalizedBufferCount = 0;
            uint64_t swapFinHit = ++s_swapFinalizedBufferCount;
            if (swapFinHit <= 16ull || (swapFinHit % 500ull) == 0ull) {
                NSLog(@"MGL WARNING: mtlSwapBuffers found finalized command buffer (status: %ld), rotating (hit=%llu)",
                      (long)bufferStatus, (unsigned long long)swapFinHit);
            }
            [self endRenderEncodingLocked];
            [self newCommandBufferLocked];
            if (!_renderPassManager.state->currentCommandBuffer) {
                NSLog(@"MGL ERROR: Failed to create new command buffer for presentation");
                return;
            }
        }

        @try {
            if (_drawable.texture == NULL) {
                NSLog(@"MGL ERROR: Drawable texture is NULL, cannot present");
                return;
            }

            if (_drawable.texture.width == 0 || _drawable.texture.height == 0) {
                NSLog(@"MGL ERROR: Drawable has invalid dimensions: %dx%d",
                      (int)_drawable.texture.width, (int)_drawable.texture.height);
                return;
            }

            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL INFO: Presenting drawable with texture: %dx%d, format: %lu",
                      (int)_drawable.texture.width, (int)_drawable.texture.height,
                      (unsigned long)_drawable.texture.pixelFormat);
            }

            if (!(mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
                  mglRenderCppGetDevice() &&
                  mglRenderCppPresentDrawable(
                      (__bridge void *)_renderPassManager.state->currentCommandBuffer,
                      (__bridge void *)_drawable) == 0)) {
                [_renderPassManager.state->currentCommandBuffer presentDrawable: _drawable];
            }
            if (traceSwap) {
                mglTraceLogNSString(@"MGL TRACE swap.present call=%llu cb=%p drawable=%p",
                      (unsigned long long)swapCall, _renderPassManager.state->currentCommandBuffer, _drawable);
            }

        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Critical drawable presentation failure: %@", exception);
            NSLog(@"MGL ERROR: Exception name: %@, reason: %@", [exception name], [exception reason]);
            [self cleanupCommandBuffer];
            return;
        }

        id<MTLCommandBuffer> commandBufferToCommit =
            [_renderPassManager detachCurrentCommandBufferForSubmission];
        uint64_t committedGeneration = mglAdvanceFrameGeneration();
        /* Sweep the bound buffer maps so base/attrib/uniform/SSBO buffers that
         * were encoded this frame keep their pool slots pinned for the
         * committed command buffer (P3 CoW snapshot reuse). */
        BufferMapList *boundLists[3] = {
            &MGL_STATE(glm_ctx)->vertex_buffer_map_list,
            &MGL_STATE(glm_ctx)->fragment_buffer_map_list,
            &MGL_STATE(glm_ctx)->compute_buffer_map_list,
        };
        for (int li = 0; li < 3; ++li) {
            for (GLuint mi = 0; mi < boundLists[li]->count; ++mi) {
                mglNoteBufferEncoded(boundLists[li]->buffers[mi].buf);
            }
        }
        @try {
            if (traceSwap) {
                mglTraceLogNSString(@"MGL TRACE swap.commit.begin call=%llu cb=%p status=%s label=%@",
                      (unsigned long long)swapCall,
                      commandBufferToCommit,
                      mglCommandBufferStatusName(commandBufferToCommit
                          ? mglRenderCommandBufferStatus(commandBufferToCommit)
                          : MTLCommandBufferStatusError),
                      commandBufferToCommit ? (commandBufferToCommit.label ?: @"(no-label)") : @"(nil)");
            }
            /* Register the frame-completion handler BEFORE commit:
             * commitCommandBufferWithAGXRecovery: commits the CB, and Metal
             * asserts if addCompletedHandler: is called after commit. */
            if (commandBufferToCommit) {
                BOOL completionRegistered = NO;
                if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
                    mglRenderCppGetDevice()) {
                    int completionResult =
                        mglRenderCppAddCommandBufferCompletion(
                        (__bridge void *)commandBufferToCommit,
                        mglRecordFrameCommandBufferCompleted,
                        (void *)(uintptr_t)committedGeneration,
                        NULL);
                    completionRegistered = completionResult == 0;
                }
                if (!completionRegistered) {
                    [commandBufferToCommit addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                        mglRecordFrameCompleted(committedGeneration);
                    }];
                }
            }
            [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
            _lastCommittedCB = commandBufferToCommit;
            if (traceSwap) {
                mglTraceLogNSString(@"MGL TRACE swap.commit.end call=%llu", (unsigned long long)swapCall);
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Failed to commit command buffer: %@", exception);
            [self recordGPUError];
        }

        if (traceSwap) {
            mglTraceLogNSString(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=post_commit", (unsigned long long)swapCall);
        }
        _drawable = [_layer nextDrawable];
        if (traceSwap) {
            id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
            mglTraceLogNSString(@"MGL TRACE swap.nextDrawable.end call=%llu stage=post_commit drawable=%p tex=%p size=%lux%lu",
                  (unsigned long long)swapCall,
                  _drawable,
                  tex,
                  (unsigned long)(tex ? tex.width : 0),
                  (unsigned long)(tex ? tex.height : 0));
        }
        if (_drawable == NULL) {
            NSLog(@"MGL WARNING: Failed to get next drawable in mtlSwapBuffers");
            return;
        }

        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to create post-swap command buffer");
            return;
        }
        _defaultDrawableWrittenSinceLastSwap = NO;
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        double swapElapsedUs = (mglTraceClockNS() - swapStartNS) / 1000.0;
        if (traceSwap) {
            mglTraceLogNSString(@"MGL TRACE swap.end call=%llu elapsed=%.1fus",
                  (unsigned long long)swapCall,
                  swapElapsedUs);
            mglLogStateSnapshot("swap.exit.ok",
                                ctx,
                                _renderPassManager.state->currentCommandBuffer,
                                _renderPassManager.state->currentRenderEncoder,
                                _renderPassManager.state->renderPassDescriptor,
                                _drawable);
        } else if (swapElapsedUs >= 25000.0) {
            mglTraceLogNSString(@"MGL TRACE swap.slow call=%llu elapsed=%.1fus",
                  (unsigned long long)swapCall,
                  swapElapsedUs);
        }
    }
    else if (kMGLVerboseFrameLoopLogs || traceSwap)
    {
        NSLog(@"MGL INFO: mtlSwapBuffers skipped present because draw_buffer is GL_NONE");
    }

    /* Perf summary: snapshot + reset per-frame counters at the swap boundary.
     * Runs on every normal exit path (present + GL_NONE skip).  Early-return
     * error paths intentionally skip this so their counters roll into the
     * next successful frame.  Uses mglNowSeconds() (CFAbsoluteTimeGetCurrent)
     * for consistency with the swap-interval measurement above. */
    if (mglPerfSummaryEnabled()) {
        double now = mglTraceNowSeconds();
        static _Atomic double s_last_swap_time = 0.0;
        double interval = 0.0;
        double prev = atomic_load_explicit(&s_last_swap_time, memory_order_relaxed);
        if (prev > 0.0) interval = (now - prev) * 1000.0;
        atomic_store_explicit(&s_last_swap_time, now, memory_order_relaxed);
        mglPrintPerfSummary(interval);
        mglResetPerfCounters();
    }
}

/* copyRenderPassColorToDrawableIfNeeded: and
 * scheduleSwapTextureSampleDiagnostics: moved to
 * MGLRenderer+SwapDiagnostics.m */

#pragma mark C interface to mtlClearBuffer
-(void) mtlClearBuffer:(GLMContext) glm_ctx type:(GLuint) type mask:(GLbitfield) mask
{
    (void)type;
    if (!glm_ctx || (mask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) == 0) {
        return;
    }

    ctx = glm_ctx;

    if (!MGL_STATE(glm_ctx)->caps.scissor_test) {
        [self endRenderEncoding];

        if (!_renderPassManager.state->currentCommandBuffer && ![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: immediate clear failed to create command buffer");
            return;
        }

        Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
        if (fbo && (fbo->dirty_bits & DIRTY_FBO_BINDING)) {
            RETURN_ON_FAILURE([self bindFramebufferAttachmentTextures]);
            fbo->dirty_bits &= ~DIRTY_FBO_BINDING;
        }

        RETURN_ON_FAILURE([self newRenderEncoderWithReason:MGL_ENC_REASON_CLEAR]);
        [self endRenderEncoding];
        mglMarkRendererDirtyBits(glm_ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return;
    }

    GLint rawX = MGL_STATE(glm_ctx)->var.scissor_box[0];
    GLint rawY = MGL_STATE(glm_ctx)->var.scissor_box[1];
    GLint rawW = MGL_STATE(glm_ctx)->var.scissor_box[2];
    GLint rawH = MGL_STATE(glm_ctx)->var.scissor_box[3];
    if (rawW <= 0 || rawH <= 0) {
        return;
    }

    Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
    Texture *colorTexObj = NULL;
    Texture *depthTexObj = NULL;
    FBOAttachment *colorAttachment = NULL;
    FBOAttachment *depthAttachment = NULL;
    id<MTLTexture> colorTexture = nil;
    id<MTLTexture> depthTexture = nil;
    MGLMetalAttachmentSubresource colorSubresource = {0u, 0u, 0u};
    MGLMetalAttachmentSubresource depthSubresource = {0u, 0u, 0u};

    BOOL wantsColor = ((mask & GL_COLOR_BUFFER_BIT) != 0);
    BOOL wantsDepth = ((mask & GL_DEPTH_BUFFER_BIT) != 0) && MGL_STATE(glm_ctx)->var.depth_writemask;

    if (wantsColor) {
        BOOL colorMaskAllowsWrite =
            !MGL_STATE(glm_ctx)->caps.use_color_mask[0] ||
            MGL_STATE(glm_ctx)->var.color_writemask[0][0] ||
            MGL_STATE(glm_ctx)->var.color_writemask[0][1] ||
            MGL_STATE(glm_ctx)->var.color_writemask[0][2] ||
            MGL_STATE(glm_ctx)->var.color_writemask[0][3];
        if (!colorMaskAllowsWrite) {
            wantsColor = NO;
        }
    }

    if (fbo) {
        if (wantsColor) {
            GLsizei drawBufferCount = mglMetalDrawBufferCount(glm_ctx);
            for (GLsizei slot = 0; slot < drawBufferCount; ++slot) {
                GLuint attachmentIndex = 0u;
                if (!mglMetalResolveFboDrawAttachmentIndex(glm_ctx,
                                                           mglMetalDrawBufferAt(glm_ctx, (GLuint)slot),
                                                           &attachmentIndex) ||
                    attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
                    ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
                    continue;
                }

                colorAttachment = &fbo->color_attachments[attachmentIndex];
                colorTexObj = [self framebufferAttachmentTexture:colorAttachment];
                if (!colorTexObj) {
                    continue;
                }

                colorTexObj->is_render_target = true;
                if (![self bindMTLTexture:colorTexObj] || !colorTexObj->mtl_data) {
                    colorTexObj = NULL;
                    colorAttachment = NULL;
                    continue;
                }

                colorTexture = (__bridge id<MTLTexture>)(colorTexObj->mtl_data);
                colorSubresource = mglMetalAttachmentSubresourceForAttachment(colorAttachment);
                break;
            }

            if (!colorTexture) {
                wantsColor = NO;
            }
        }

        if (wantsDepth && fbo->depth.texture) {
            depthAttachment = &fbo->depth;
            depthTexObj = [self framebufferAttachmentTexture:depthAttachment];
            if (depthTexObj) {
                depthTexObj->is_render_target = true;
                if ([self bindMTLTexture:depthTexObj] && depthTexObj->mtl_data) {
                    depthTexture = (__bridge id<MTLTexture>)(depthTexObj->mtl_data);
                    depthSubresource = mglMetalAttachmentSubresourceForAttachment(depthAttachment);
                }
            }
        }
        if (wantsDepth && !depthTexture) {
            wantsDepth = NO;
        }
    } else {
        GLuint drawBufferIndex = mglDefaultDrawBufferIndexForGL(MGL_STATE(glm_ctx)->draw_buffer);
        if (wantsColor) {
            if (drawBufferIndex == _FRONT) {
                if (!_drawable && _layer) {
                    [self mglApplyPendingDrawableSize];
                    _drawable = [_layer nextDrawable];
                }
                colorTexture = _drawable ? _drawable.texture : nil;
            } else if (drawBufferIndex < _MAX_DRAW_BUFFERS) {
                colorTexture = _drawBuffers[drawBufferIndex].drawbuffer;
                if (!colorTexture) {
                    colorTexture = [self newDrawBuffer:glm_ctx->pixel_format.mtl_pixel_format isDepthStencil:false];
                    _drawBuffers[drawBufferIndex].drawbuffer = colorTexture;
                }
            }
            if (!colorTexture) {
                wantsColor = NO;
            }
        }

        if (wantsDepth && drawBufferIndex < _MAX_DRAW_BUFFERS) {
            depthTexture = _drawBuffers[drawBufferIndex].depthbuffer;
            if (!depthTexture) {
                MTLPixelFormat depthFormat = glm_ctx->depth_format.mtl_pixel_format;
                if (depthFormat == MTLPixelFormatInvalid) {
                    depthFormat = MTLPixelFormatDepth32Float;
                }
                NSUInteger depthWidth = colorTexture ? colorTexture.width : (NSUInteger)MAX(MGL_STATE(glm_ctx)->viewport[2], 1);
                NSUInteger depthHeight = colorTexture ? colorTexture.height : (NSUInteger)MAX(MGL_STATE(glm_ctx)->viewport[3], 1);
                depthTexture = [self newDrawBufferWithCustomSize:depthFormat
                                                  isDepthStencil:true
                                                      customSize:CGSizeMake(depthWidth, depthHeight)];
                _drawBuffers[drawBufferIndex].depthbuffer = depthTexture;
            }
            if (!depthTexture) {
                wantsDepth = NO;
            }
        }
    }

    if (!wantsColor && !wantsDepth) {
        return;
    }

    NSUInteger passWidth = 0u;
    NSUInteger passHeight = 0u;
    id<MTLTexture> sizeTexture = colorTexture ? colorTexture : depthTexture;
    if (sizeTexture) {
        passWidth = sizeTexture.width;
        passHeight = sizeTexture.height;
    }
    if (passWidth == 0u || passHeight == 0u) {
        return;
    }

    GLint x0 = rawX;
    GLint y0 = rawY;
    GLint x1 = rawX + rawW;
    GLint y1 = rawY + rawH;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (GLint)passWidth) x1 = (GLint)passWidth;
    if (y1 > (GLint)passHeight) y1 = (GLint)passHeight;
    if (x1 <= x0 || y1 <= y0) {
        return;
    }

    GLint clearW = x1 - x0;
    GLint clearH = y1 - y0;
    GLint metalY = y0;
    if (MGL_STATE(glm_ctx)->var.clip_origin != GL_UPPER_LEFT) {
        metalY = (GLint)passHeight - y1;
        if (metalY < 0) {
            metalY = 0;
        }
    }

    MTLPixelFormat colorFormat = colorTexture ? colorTexture.pixelFormat : MTLPixelFormatInvalid;
    MTLPixelFormat depthFormat = depthTexture ? depthTexture.pixelFormat : MTLPixelFormatInvalid;
    id<MTLRenderPipelineState> pipeline = [self clearRectPipelineForColorFormat:colorFormat
                                                                    depthFormat:depthFormat
                                                                    writesColor:wantsColor
                                                                    writesDepth:wantsDepth];
    if (!pipeline) {
        NSLog(@"MGL ERROR: scissored clear missing pipeline color=%lu depth=%lu wantsColor=%d wantsDepth=%d",
              (unsigned long)colorFormat,
              (unsigned long)depthFormat,
              wantsColor ? 1 : 0,
              wantsDepth ? 1 : 0);
        return;
    }

    MGLClearRectParams params;
    params.color = (vector_float4){
        MGL_STATE(glm_ctx)->color_clear_value[0],
        MGL_STATE(glm_ctx)->color_clear_value[1],
        MGL_STATE(glm_ctx)->color_clear_value[2],
        MGL_STATE(glm_ctx)->color_clear_value[3]
    };
    params.depth = (float)MGL_STATE(glm_ctx)->var.depth_clear_value;
    params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

    MTLViewport viewport = {
        0.0, 0.0,
        (double)passWidth, (double)passHeight,
        0.0, 1.0
    };
    MTLScissorRect scissor = {
        (NSUInteger)x0,
        (NSUInteger)metalY,
        (NSUInteger)clearW,
        (NSUInteger)clearH
    };

    /* Optimization: reuse the current render encoder when it targets the same
     * framebuffer attachments we're about to clear.  This avoids ending the
     * current encoder + creating a new MTLRenderPassDescriptor + new encoder
     * for every scissored clear (3-8 times per frame in MC).
     *
     * Conditions: an encoder is active, the render pass matches the current
     * FBO, no visibility query is active (which would require an encoder
     * rebuild to attach the visibility buffer), and the render pass's color
     * attachment 0 / depth attachment textures match the ones we resolved
     * from the FBO.  When any condition fails, fall back to the original
     * endRenderEncoding + new-encoder path. */
    BOOL canReuseCurrentEncoder = NO;
    uint32_t sampleQueryActive = 0;
    if (_queryStateOwner) {
        mglRenderCppIsSampleQueryActive(
            _queryStateOwner, &sampleQueryActive);
    }
    if (_renderPassManager.state->currentRenderEncoder &&
        [self currentRenderPassMatchesCurrentFramebuffer] &&
        !sampleQueryActive) {
        if (_renderPassManager.state->renderPassStateOwner ||
            _renderPassManager.state->renderPassDescriptor) {
            BOOL colorMatches = !wantsColor;
            if (wantsColor) {
                id<MTLTexture> rpColor0 = mglRenderPassAttachmentTextureForState(
                    _renderPassManager.state->renderPassDescriptor,
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0);
                NSUInteger rpLevel = 0u, rpSlice = 0u, rpDepthPlane = 0u;
                mglRenderPassAttachmentSubresourceForState(
                    _renderPassManager.state->renderPassDescriptor,
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    &rpLevel, &rpSlice, &rpDepthPlane);
                colorMatches = (rpColor0 == colorTexture &&
                                rpLevel == colorSubresource.level &&
                                rpSlice == colorSubresource.slice);
            }
            BOOL depthMatches = !wantsDepth;
            if (wantsDepth) {
                id<MTLTexture> rpDepth = mglRenderPassAttachmentTextureForState(
                    _renderPassManager.state->renderPassDescriptor,
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0);
                NSUInteger rpLevel = 0u, rpSlice = 0u, rpDepthPlane = 0u;
                mglRenderPassAttachmentSubresourceForState(
                    _renderPassManager.state->renderPassDescriptor,
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    &rpLevel, &rpSlice, &rpDepthPlane);
                depthMatches = (rpDepth == depthTexture &&
                                rpLevel == depthSubresource.level &&
                                rpSlice == depthSubresource.slice);
            }
            canReuseCurrentEncoder = colorMatches && depthMatches;
        }
    }

    if (canReuseCurrentEncoder) {
        /* Reuse the current encoder: set scissor + pipeline + params, draw
         * the clear quad.  No encoder creation/destruction overhead.
         *
         * The clear draw touches exactly viewport, scissor, pipeline,
         * depth-stencil (when clearing depth) and VS/FS buffer slot 0 —
         * record those real values / invalidate those slots in the bind
         * cache instead of invalidating everything, so textures, samplers
         * and the other buffer slots keep their dedup state. */
        id<MTLRenderCommandEncoder> encoder = _renderPassManager.state->currentRenderEncoder;

        mglRenderCppBindingSetViewport(
            _bindingStateOwner, (__bridge void *)encoder,
            viewport.originX, viewport.originY, viewport.width, viewport.height,
            viewport.znear, viewport.zfar);
        mglRenderCppBindingSetScissor(
            _bindingStateOwner, (__bridge void *)encoder,
            scissor.x, scissor.y, scissor.width, scissor.height);
        mglRendererSetRenderPipeline(encoder, pipeline);
        mglRenderCppBindingSetPipelineState(
            _bindingStateOwner, (__bridge void *)pipeline);
        if (wantsDepth) {
            id<MTLDepthStencilState> depthState = [self clearRectDepthState];
            if (depthState) {
                mglRendererSetDepthStencil(encoder, depthState);
                mglRenderCppBindingSetDepthStencilState(
                    _bindingStateOwner, (__bridge void *)depthState);
            }
        }
        mglRendererSetRenderBytes(encoder, &params, sizeof(params),
                                  MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0);
        mglRenderCppBindingInvalidateVertexBuffer(_bindingStateOwner, 0);
        if (wantsColor) {
            mglRendererSetRenderBytes(encoder, &params, sizeof(params),
                                      MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0);
            mglRenderCppBindingInvalidateFragmentBuffer(_bindingStateOwner, 0);
        }
        mglRendererDrawPrimitives(encoder, MTLPrimitiveTypeTriangleStrip, 0, 4);

        if (wantsColor && colorTexObj && colorAttachment) {
            colorAttachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
            mglMarkTextureLevelRenderTargetWritten(colorTexObj, colorAttachment->level);
        }
        if (wantsDepth && depthTexObj && depthAttachment) {
            depthAttachment->clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
            mglMarkTextureLevelRenderTargetWritten(depthTexObj, depthAttachment->level);
        }

        mglMarkRendererDirtyBits(glm_ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return;
    }

    /* Fallback: end the current encoder and create a dedicated clear encoder.
     * Used when no encoder is active, the FBO doesn't match, a visibility
     * query is active, or the attachment textures don't match. */
    [self endRenderEncoding];
    if (!_renderPassManager.state->currentCommandBuffer && ![self newCommandBuffer]) {
        NSLog(@"MGL ERROR: scissored clear failed to create command buffer");
        return;
    }

    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    MGLRenderCppRenderPassState clearState = {0};
    if (colorTexture) {
        clearPass.colorAttachments[0].texture = colorTexture;
        clearPass.colorAttachments[0].level = colorSubresource.level;
        clearPass.colorAttachments[0].slice = colorSubresource.slice;
        clearPass.colorAttachments[0].depthPlane = colorSubresource.depthPlane;
        clearPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
        clearState.color[0].attachment.texture =
            (__bridge void *)colorTexture;
        clearState.color[0].attachment.level = colorSubresource.level;
        clearState.color[0].attachment.slice = colorSubresource.slice;
        clearState.color[0].attachment.depth_plane =
            colorSubresource.depthPlane;
        clearState.color[0].attachment.load_action = MTLLoadActionLoad;
        clearState.color[0].attachment.store_action = MTLStoreActionStore;
    }
    if (depthTexture) {
        clearPass.depthAttachment.texture = depthTexture;
        clearPass.depthAttachment.level = depthSubresource.level;
        clearPass.depthAttachment.slice = depthSubresource.slice;
        clearPass.depthAttachment.depthPlane = depthSubresource.depthPlane;
        clearPass.depthAttachment.loadAction = MTLLoadActionLoad;
        clearPass.depthAttachment.storeAction = MTLStoreActionStore;
        clearState.depth.attachment.texture =
            (__bridge void *)depthTexture;
        clearState.depth.attachment.level = depthSubresource.level;
        clearState.depth.attachment.slice = depthSubresource.slice;
        clearState.depth.attachment.depth_plane =
            depthSubresource.depthPlane;
        clearState.depth.attachment.load_action = MTLLoadActionLoad;
        clearState.depth.attachment.store_action = MTLStoreActionStore;
    }
    clearPass.renderTargetWidth = passWidth;
    clearPass.renderTargetHeight = passHeight;
    clearState.render_target_width = passWidth;
    clearState.render_target_height = passHeight;

    id<MTLRenderCommandEncoder> clearEncoder = mglRendererCreateRenderEncoder(
        _renderPassManager.state->currentCommandBuffer, clearPass,
        &clearState);
    if (!clearEncoder) {
        NSLog(@"MGL ERROR: scissored clear failed to create render encoder");
        return;
    }

    mglRendererSetViewport(clearEncoder, viewport);
    mglRendererSetScissor(clearEncoder, scissor);
    mglRendererSetRenderPipeline(clearEncoder, pipeline);
    if (wantsDepth) {
        id<MTLDepthStencilState> depthState = [self clearRectDepthState];
        if (depthState) {
            mglRendererSetDepthStencil(clearEncoder, depthState);
        }
    }
    mglRendererSetRenderBytes(clearEncoder, &params, sizeof(params),
                              MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0);
    if (wantsColor) {
        mglRendererSetRenderBytes(clearEncoder, &params, sizeof(params),
                                  MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0);
    }
    mglRendererDrawPrimitives(clearEncoder, MTLPrimitiveTypeTriangleStrip, 0, 4);
    mglRendererEndRenderEncoder(clearEncoder);

    if (wantsColor && colorTexObj && colorAttachment) {
        colorAttachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(colorTexObj, colorAttachment->level);
    }
    if (wantsDepth && depthTexObj && depthAttachment) {
        depthAttachment->clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(depthTexObj, depthAttachment->level);
    }

    mglMarkRendererDirtyBits(glm_ctx->active_state,
                             DIRTY_FBO | DIRTY_RENDER_STATE);
}

#pragma mark C interface to mtlBufferSubData

-(void) mtlBufferSubData:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr
{
    METAL_LOCK();
    [self mtlBufferSubDataLocked:glm_ctx buf:buf offset:offset size:size ptr:ptr];
    METAL_UNLOCK();
}

-(void) mtlBufferSubDataLocked:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr
{
    if (mglRendererUsesMetalCpp()) {
        char subDataError[256] = {0};
        int subDataResult = mglRenderCppBufferSubDataStorage(
            buf, offset, size, ptr, subDataError, sizeof(subDataError));
        if (subDataResult == MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) {
            return;
        }
        if (subDataResult == MGL_RENDER_CPP_BUFFER_OPERATION_ERROR) {
            NSLog(@"MGL ERROR: Metal-cpp buffer subdata failed buffer=%u: %s",
                  buf ? (unsigned)buf->name : 0u,
                  subDataError[0] ? subDataError : "?");
            return;
        }
    }

    static uint64_t s_mtlBufferSubDataCalls = 0;
    uint64_t call = ++s_mtlBufferSubDataCalls;
    bool trace = kMGLDiagnosticStateLogs && mglShouldTraceBufferTransferCall(call);
    id<MTLBuffer> mtl_buffer;
    void *data;

    if (!buf) {
        NSLog(@"MGL ERROR: mtlBufferSubData null buffer offset=%zu size=%zu", offset, size);
        return;
    }

    if (size == 0) {
        return;
    }

    if (!ptr) {
        NSLog(@"MGL WARNING: mtlBufferSubData null source ptr buffer=%u offset=%zu size=%zu", buf->name, offset, size);
        return;
    }

    if (trace) {
        char srcHead[64];
        srcHead[0] = '\0';
        mglTraceFormatBytes(ptr, size, srcHead, sizeof(srcHead));
        uint64_t srcHash = mglTraceHashBytes(ptr, size);
        mglTraceLogNSString(@"MGL TRACE mtlBufferSubData.begin call=%llu buffer=%u size=%lld off=%zu len=%zu mtl=%p cpu=%p dirty=0x%x srcHash=0x%016llx srcHead=%s",
              (unsigned long long)call,
              buf->name,
              (long long)buf->size,
              offset,
              size,
              buf->data.mtl_data,
              (void *)(uintptr_t)buf->data.buffer_data,
              buf->data.dirty_bits,
              (unsigned long long)srcHash,
              srcHead);
    }

    if (buf->data.mtl_data == NULL)
    {
        [self bindMTLBufferLocked:buf];
    }

    // AGX Driver Compatibility: For small buffers, bindMTLBuffer may still have NULL mtl_data
    // In this case, we should update the buffer_data directly
    if (buf->data.mtl_data == NULL)
    {
        // Small buffer case - update buffer_data directly
        if (buf->data.buffer_data)
        {
            memcpy((void *)(buf->data.buffer_data + offset), ptr, size);
            if (trace) {
                const void *dst = (const void *)((uintptr_t)buf->data.buffer_data + offset);
                char dstHead[64];
                dstHead[0] = '\0';
                mglTraceFormatBytes(dst, size, dstHead, sizeof(dstHead));
                uint64_t dstHash = mglTraceHashBytes(dst, size);
                mglTraceLogNSString(@"MGL TRACE mtlBufferSubData.cpuFallback call=%llu buffer=%u off=%zu len=%zu dstHash=0x%016llx dstHead=%s",
                      (unsigned long long)call,
                      buf->name,
                      offset,
                      size,
                      (unsigned long long)dstHash,
                      dstHead);
            }
        }
        return;
    }

    mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtl_buffer) {
        NSLog(@"MGL ERROR: mtlBufferSubData buffer=%u has invalid Metal buffer", buf->name);
        return;
    }

    uint8_t *cpuData = (uint8_t *)(uintptr_t)buf->data.buffer_data;
    id<MTLBuffer> bufferBeforeSnapshot = mtl_buffer;
    if (cpuData && cpuData != mtl_buffer.contents) {
        memmove(cpuData + offset, ptr, size);
        if (!mglSnapshotSharedDirtyBuffer(_device, buf, &mtl_buffer)) {
            return;
        }
    }

    if (offset > mtl_buffer.length || size > (mtl_buffer.length - offset)) {
        NSLog(@"MGL ERROR: mtlBufferSubData range exceeds Metal buffer buffer=%u off=%zu size=%zu len=%lu",
              buf->name,
              offset,
              size,
              (unsigned long)mtl_buffer.length);
        return;
    }

    data = mtl_buffer.contents;
    if (!data) {
        NSLog(@"MGL ERROR: mtlBufferSubData buffer=%u has NULL contents", buf->name);
        return;
    }
    /* A COW snapshot already copied the CPU shadow (including this range when
     * written_min/max covers it).  Only write the Metal store in place when
     * no new buffer was allocated. */
    if (mtl_buffer == bufferBeforeSnapshot) {
        memcpy((uint8_t *)data + offset, ptr, size);
        if (mtl_buffer.storageMode == MTLStorageModeManaged) {
            [mtl_buffer didModifyRange:NSMakeRange(offset, size)];
        }
    }

    if (trace) {
        const void *dst = (const void *)((const uint8_t *)mtl_buffer.contents + offset);
        char dstHead[64];
        dstHead[0] = '\0';
        mglTraceFormatBytes(dst, size, dstHead, sizeof(dstHead));
        uint64_t dstHash = mglTraceHashBytes(dst, size);
        mglTraceLogNSString(@"MGL TRACE mtlBufferSubData.end call=%llu buffer=%u off=%zu len=%zu mtlLen=%lu dstHash=0x%016llx dstHead=%s",
              (unsigned long long)call,
              buf->name,
              offset,
              size,
              (unsigned long)mtl_buffer.length,
              (unsigned long long)dstHash,
              dstHead);
    }
}

#pragma mark C interface to mtlMapUnmapBuffer
-(void *) mtlMapUnmapBuffer:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t) offset size:(size_t) size access:(GLenum) access map:(bool)map
{
    METAL_LOCK();
    void *result = [self mtlMapUnmapBufferLocked:glm_ctx buf:buf offset:offset size:size access:access map:map];
    METAL_UNLOCK();
    return result;
}

/* Copy GPU-written bytes back from the Metal buffer into the CPU shadow so
 * later glGetBufferSubData / glGetNamedBufferSubData reads return the shader
 * results (GL 4.6 §6.2).  The caller must have waited for the GPU (commit +
 * waitUntilCompleted) before this runs.  Mirrors the read side of
 * mtlMapUnmapBufferLocked: skips when the shadow holds un-uploaded CPU writes
 * (cpu_shadow_pending) or when the Metal buffer shares the shadow memory. */
- (void)mtlReadBackBuffer:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size
{
    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() != NULL) {
        mglRenderCppReadBackBuffer(glm_ctx, buf, offset, size);
        return;
    }
    if (!buf || size == 0 || buf->cpu_shadow_pending) {
        return;
    }

    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtlBuffer || mtlBuffer.storageMode != MTLStorageModeShared) {
        return;
    }

    uint8_t *mtlBase = (uint8_t *)mtlBuffer.contents;
    uint8_t *cpuBase = (uint8_t *)(uintptr_t)buf->data.buffer_data;
    if (!mtlBase || !cpuBase || mtlBase == cpuBase) {
        return;
    }

    NSUInteger mtlLen = mtlBuffer.length;
    NSUInteger shadowLen = (NSUInteger)buf->data.buffer_size;
    if (offset >= mtlLen) {
        return;
    }
    size_t safeLen = MIN((NSUInteger)size, mtlLen - (NSUInteger)offset);
    if (shadowLen > 0 && (NSUInteger)offset + safeLen > shadowLen) {
        if ((NSUInteger)offset >= shadowLen) {
            return;
        }
        safeLen = shadowLen - (NSUInteger)offset;
    }

    memcpy(cpuBase + offset, mtlBase + offset, safeLen);
}

-(void *) mtlMapUnmapBufferLocked:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t) offset size:(size_t) size access:(GLenum) access map:(bool)map
{
    id<MTLBuffer> mtl_buffer = nil;

    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() != NULL) {
        void *mapped = NULL;
        char mapError[256] = {0};
        int mapResult = mglRenderCppMapBufferStorage(
            buf, offset, size, (unsigned int)access, map,
            &mapped, mapError, sizeof(mapError));
        if (mapResult == MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) {
            return mapped;
        }
        if (mapResult == MGL_RENDER_CPP_BUFFER_OPERATION_ERROR) {
            NSLog(@"MGL ERROR: Metal-cpp buffer map failed buffer=%u: %s",
                  buf ? (unsigned)buf->name : 0u,
                  mapError[0] ? mapError : "?");
            return NULL;
        }
    }

    if (!buf) {
        NSLog(@"MGL ERROR: mtlMapUnmapBuffer called with NULL buffer");
        return NULL;
    }

    if (buf->data.mtl_data == NULL)
    {
        [self bindMTLBufferLocked:buf];
    }

    mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtl_buffer) {
        NSLog(@"MGL ERROR: mtlMapUnmapBuffer buffer=%u has NULL Metal buffer after bind", buf->name);
        return NULL;
    }

    uint8_t *mtlBase = (uint8_t *)mtl_buffer.contents;
    NSUInteger mtlLen = mtl_buffer.length;
    if (offset > mtlLen) {
        NSLog(@"MGL ERROR: mtlMapUnmapBuffer buffer=%u offset=%zu beyond mtlLen=%lu",
              buf->name, offset, (unsigned long)mtlLen);
        return NULL;
    }
    NSUInteger safeLen = MIN((NSUInteger)size, (mtlLen - (NSUInteger)offset));

    uint8_t *cpuBase = NULL;
    if (buf->data.buffer_data && ((uintptr_t)buf->data.buffer_data >= 0x1000ull)) {
        cpuBase = (uint8_t *)(uintptr_t)buf->data.buffer_data;
    }

    if (map)
    {
        bool reads = access == GL_READ_ONLY || access == GL_READ_WRITE ||
                     (access & GL_MAP_READ_BIT) != 0;
        if (cpuBase) {
            uint8_t *cpuPtr = cpuBase + offset;
            /* cpu_shadow_pending: the shadow holds map-write bytes not yet
             * uploaded to the Metal buffer; refreshing from Metal here would
             * clobber them with stale data (GL 4.6 §6.3: unmapped writes
             * must stay visible).  The flag is cleared once encoding uploads
             * the shadow or a GPU write is copied back into it, so GPU
             * readback (SSBO/XFB writes) still refreshes from Metal. */
            if (reads && mtlBase && mtlBase != cpuBase && safeLen > 0 &&
                !buf->cpu_shadow_pending) {
                memcpy(cpuPtr, mtlBase + offset, (size_t)safeLen);
            }

            if (kMGLDiagnosticStateLogs) {
                uint64_t mtlHash = mglTraceHashBytes(mtlBase ? mtlBase + offset : NULL, (size_t)safeLen);
                uint64_t cpuHash = mglTraceHashBytes(cpuPtr, (size_t)safeLen);
                char mtlHead[64];
                char cpuHead[64];
                mtlHead[0] = '\0';
                cpuHead[0] = '\0';
                mglTraceFormatBytes(mtlBase ? mtlBase + offset : NULL, (size_t)safeLen, mtlHead, sizeof(mtlHead));
                mglTraceFormatBytes(cpuPtr, (size_t)safeLen, cpuHead, sizeof(cpuHead));
                mglTraceLogNSString(@"MGL TRACE mtlMap.map buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
                      buf->name,
                      offset,
                      size,
                      (unsigned long)safeLen,
                      (unsigned)access,
                      mtlBase ? mtlBase + offset : NULL,
                      cpuPtr,
                      (mtlBase && mtlBase + offset == cpuPtr) ? 1 : 0,
                      (unsigned long long)mtlHash,
                      (unsigned long long)cpuHash,
                      mtlHead,
                      cpuHead);
            }
            return cpuPtr;
        }

        uint8_t *mappedPtr = mtlBase ? (mtlBase + offset) : NULL;
        if (kMGLDiagnosticStateLogs) {
            uint64_t mtlHash = mglTraceHashBytes(mappedPtr, (size_t)safeLen);
            char mtlHead[64];
            mtlHead[0] = '\0';
            mglTraceFormatBytes(mappedPtr, (size_t)safeLen, mtlHead, sizeof(mtlHead));

            uint8_t *cpuPtr = cpuBase ? (cpuBase + offset) : NULL;
            uint64_t cpuHash = mglTraceHashBytes(cpuPtr, (size_t)safeLen);
            char cpuHead[64];
            cpuHead[0] = '\0';
            mglTraceFormatBytes(cpuPtr, (size_t)safeLen, cpuHead, sizeof(cpuHead));

            mglTraceLogNSString(@"MGL TRACE mtlMap.map buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
                  buf->name,
                  offset,
                  size,
                  (unsigned long)safeLen,
                  (unsigned)access,
                  mappedPtr,
                  cpuPtr,
                  (mappedPtr && cpuPtr && mappedPtr == cpuPtr) ? 1 : 0,
                  (unsigned long long)mtlHash,
                  (unsigned long long)cpuHash,
                  mtlHead,
                  cpuHead);
        }

        return mappedPtr;
    }

    if (!cpuBase && mtl_buffer.storageMode == MTLStorageModeManaged) {
        [mtl_buffer didModifyRange:NSMakeRange(offset, safeLen)];
    }

    if (kMGLDiagnosticStateLogs) {
        uint8_t *mtlPtr = mtlBase ? (mtlBase + offset) : NULL;
        uint8_t *cpuPtr = cpuBase ? (cpuBase + offset) : NULL;
        uint64_t mtlHash = mglTraceHashBytes(mtlPtr, (size_t)safeLen);
        uint64_t cpuHash = mglTraceHashBytes(cpuPtr, (size_t)safeLen);
        char mtlHead[64];
        char cpuHead[64];
        mtlHead[0] = '\0';
        cpuHead[0] = '\0';
        mglTraceFormatBytes(mtlPtr, (size_t)safeLen, mtlHead, sizeof(mtlHead));
        mglTraceFormatBytes(cpuPtr, (size_t)safeLen, cpuHead, sizeof(cpuHead));
        mglTraceLogNSString(@"MGL TRACE mtlMap.unmap buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
              buf->name,
              offset,
              size,
              (unsigned long)safeLen,
              (unsigned)access,
              mtlPtr,
              cpuPtr,
              (mtlPtr && cpuPtr && mtlPtr == cpuPtr) ? 1 : 0,
              (unsigned long long)mtlHash,
              (unsigned long long)cpuHash,
              mtlHead,
              cpuHead);
    }

    return NULL;
}

#pragma mark C interface to mtlFlushMappedBufferRange
-(void) mtlFlushMappedBufferRange:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(GLintptr) offset length:(GLsizeiptr) length
{
    METAL_LOCK();
    [self mtlFlushMappedBufferRangeLocked:glm_ctx buf:buf offset:offset length:length];
    METAL_UNLOCK();
}

-(void) mtlFlushMappedBufferRangeLocked:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(GLintptr) offset length:(GLsizeiptr) length
{
    if (mglRendererUsesMetalCpp()) {
        char flushError[256] = {0};
        int flushResult = mglRenderCppFlushBufferRangeStorage(
            buf, offset, length, flushError, sizeof(flushError));
        if (flushResult == MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) {
            return;
        }
        if (flushResult == MGL_RENDER_CPP_BUFFER_OPERATION_ERROR) {
            NSLog(@"MGL ERROR: Metal-cpp buffer range flush failed buffer=%u: %s",
                  buf ? (unsigned)buf->name : 0u,
                  flushError[0] ? flushError : "?");
            return;
        }
    }

    id<MTLBuffer> mtl_buffer;

    if (!buf) {
        NSLog(@"MGL ERROR: mtlFlushMappedBufferRange called with NULL buffer");
        return;
    }

    mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtl_buffer) {
        [self bindMTLBufferLocked:buf];
        mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
        if (!mtl_buffer) {
            return;
        }
    }

    if (offset > mtl_buffer.length || length > (mtl_buffer.length - offset)) {
        NSLog(@"MGL ERROR: mtlFlushMappedBufferRange out of range buffer=%u off=%ld len=%ld mtlLen=%lu",
              buf->name,
              offset,
              length,
              (unsigned long)mtl_buffer.length);
        return;
    }

    if (!mglSnapshotSharedBufferRange(_device,
                                      buf,
                                      &mtl_buffer,
                                      (NSUInteger)offset,
                                      (NSUInteger)length)) {
        return;
    }

    if (mtl_buffer.storageMode == MTLStorageModeManaged) {
        [mtl_buffer didModifyRange:NSMakeRange(offset, length)];
    }
}



/*
 * mglReadColorTextureAsBGRA8:... — readPixels color readback staging buffer path
 *
 * Trigger: glReadPixels color readback (BGRA8-compatible format) goes through this staging buffer path.
 * Guarantees: ensureWritableCommandBuffer acquires a writable CB → newBufferWithLength creates a staging
 *             buffer → blitCommandEncoder copyFromTexture copies GPU texture data into the staging buffer →
 *             addCompletedHandler + dispatch_semaphore_wait (250ms timeout) blocks until the CB completes →
 *             copies from stagingBuffer.contents into the user buffer → newCommandBuffer creates a new CB.
 *             Ensures that all GPU writes to this texture have completed via the CB and are visible to the CPU before readback.
 * Degradation: a 250ms timeout returns zero data and reports GL_INVALID_OPERATION; a command buffer error reports the same.
 */



/*
 * mglApplyPendingFBODepthClearForReadback:attachment:textureObj:mtlTexture: — deferred depth clear materialization
 *
 * Trigger: before depth readback, if the FBO depth attachment has an unmaterialized deferred lazy clear
 *          (attachment->clear_bitmask & GL_DEPTH_BUFFER_BIT).
 * Guarantees: constructs a render pass with loadAction=Clear (depthAttachment.loadAction=Clear),
 *             immediately calls endEncoding to materialize the clear, so subsequent readback observes
 *             cleared values rather than undefined data; clears the corresponding bit in clear_bitmask
 *             to avoid a duplicate clear. Must complete before the readback blit, otherwise the GPU write
 *             (clear) is not visible to the CPU before readback.
 */


/*
 * mtlReadDepthPixels: — depth readback path
 *
 * Trigger: glReadPixels depth component readback.
 * Guarantees: endRenderEncoding closes the open render encoder → ensureWritableCommandBuffer acquires
 *             a writable CB → mglApplyPendingFBODepthClearForReadback materializes the deferred lazy
 *             depth clear (loadAction=Clear) so readback observes cleared values → delegates to the
 *             staging buffer readback path (copyFromTexture + completed-handler semaphore).
 *             Ensures that all GPU depth writes and deferred clears have completed and are visible to the CPU before readback.
 */

#pragma mark C interface to mtlReadDrawable



#pragma mark C interface to mtlGetTexImage
/*
 * mtlGetTexImage: — texture image readback path
 *
 * Trigger: glGetTexImage reads back an entire texture level.
 * Guarantees: calls synchronizeRenderPassForTextureReadback on the target texture (if it is a render target,
 *             then endRenderEncoding + commit + waitUntilCompleted + newCommandBuffer);
 *             then endRenderEncoding + commit + waitUntilCompleted commits and waits on the dedicated blit CB
 *             (encoding copyFromTexture to the staging buffer), ensuring that all GPU writes to this texture
 *             (rendering / upload blit) have completed and are visible to the CPU before readback.
 */

#pragma mark C interface to mtlGenerateMipmaps


/* Map GL internal format to the (format, type) pair that matches the CPU
 * storage layout used by mglCreateRGBA8ExpandedUpload / channel expansion.
 * Used for format-converting readback in mtlCopyImageSubData when CPU bpp
 * differs from Metal bpp.  Returns GL_FALSE if no mapping is known. */
GLboolean mglGetCPUFormatTypeForInternalFormat(GLenum internalformat,
                                               GLenum *outFormat,
                                               GLenum *outType)
{
    if (!outFormat || !outType) return GL_FALSE;
    switch (internalformat) {
        case GL_R3_G3_B2:
            *outFormat = GL_RGB; *outType = GL_UNSIGNED_BYTE_3_3_2; return GL_TRUE;
        case GL_RGB4:
        case GL_RGB5:
            *outFormat = GL_RGB; *outType = GL_UNSIGNED_SHORT_5_6_5; return GL_TRUE;
        case GL_RGB5_A1:
            *outFormat = GL_RGBA; *outType = GL_UNSIGNED_SHORT_5_5_5_1; return GL_TRUE;
        case GL_RGBA2:
        case GL_RGBA4:
            *outFormat = GL_RGBA; *outType = GL_UNSIGNED_SHORT_4_4_4_4; return GL_TRUE;
        case GL_RGB12:
            *outFormat = GL_RGB; *outType = GL_UNSIGNED_SHORT; return GL_TRUE;
        case GL_RGB32F:
            *outFormat = GL_RGB; *outType = GL_FLOAT; return GL_TRUE;
        default:
            return GL_FALSE;
    }
}



#pragma mark C interface to mtlTexSubImage





#pragma mark utility functions for draw commands
MTLPrimitiveType getMTLPrimitiveType(GLenum mode)
{
    const GLuint err = 0xFFFFFFFF;

    switch(mode)
    {
        case GL_POINTS:
            return MTLPrimitiveTypePoint;

        case GL_LINES:
            return MTLPrimitiveTypeLine;

        case GL_LINE_STRIP:
            return MTLPrimitiveTypeLineStrip;

        case GL_TRIANGLES:
            return MTLPrimitiveTypeTriangle;

        case GL_TRIANGLE_STRIP:
            return MTLPrimitiveTypeTriangleStrip;

        case GL_LINE_LOOP:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_LINES_ADJACENCY:
        case GL_TRIANGLE_FAN:
        case GL_QUADS:
        case GL_TRIANGLE_STRIP_ADJACENCY:
        case GL_PATCHES:
            return (MTLPrimitiveType)0xFFFFFFFF;
            break;
    }

    return err;
}

MTLIndexType getMTLIndexType(GLenum type)
{
    const GLuint err = 0xFFFFFFFF;

    switch(type)
    {
        case GL_UNSIGNED_BYTE:
            return MTLIndexTypeUInt16;

        case GL_UNSIGNED_SHORT:
            return MTLIndexTypeUInt16;

        case GL_UNSIGNED_INT:
            return MTLIndexTypeUInt32;
    }

    return err;
}

Buffer *getElementBuffer(GLMContext ctx)
{
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    Buffer *gl_element_buffer = vao ? vao->element_array.buffer : NULL;

    return gl_element_buffer;
}

/* validateDrawArraysVertexInputs:(GLMContext)drawCtx moved to MGLRenderer+Draw.m */

Buffer *getIndirectBuffer(GLMContext ctx)
{
    Buffer *gl_indirect_buffer = ctx->active_state->buffers[_DRAW_INDIRECT_BUFFER];

    return gl_indirect_buffer;
}

/* resolveElementBufferForDraw:(const char *)label moved to MGLRenderer+Draw.m */

/* resolveElementBufferForCommand:(const MGLDrawCommand *)cmd moved to MGLRenderer+Draw.m */

/* resolveElementBuffer:(Buffer *)gl_element_buffer moved to MGLRenderer+Draw.m */

/* resolveIndirectBufferForDraw:(const char *)label moved to MGLRenderer+Draw.m */

/* prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label moved to MGLRenderer+Draw.m */

/* currentDrawRasterizationIsEmpty moved to MGLRenderer+Draw.m */

/* applyPolygonOffsetForDrawMode:(GLenum)mode moved to MGLRenderer+Draw.m */

/* currentDrawModeIsFullyCulled:(GLenum)mode moved to MGLRenderer+Draw.m */

/* Cull distance emulation: bind the vertex buffer to slot 29 and a params
 * buffer to slot 28 so the injected vertex-shader code can read sibling-vertex
 * cull distance values. The params encode the primitive vertex count, the
 * byte offset of the first cull distance entry within each vertex, the byte
 * stride between vertices, and the number of cull distance entries.
 *
 * The cull distance offset and stride are discovered by scanning the VAO for
 * the first enabled attribute whose name maps to mgl_CullDistance. All cull
 * distance entries are assumed to share the same buffer and stride (which is
 * the case for the CTS test and typical GL apps). */
/* MGLCullDistanceEmuParams typedef moved to MGLRenderer_Private.h */

/* bindCullDistanceEmulationBuffers:(GLenum)mode moved to MGLRenderer+Draw.m */

#pragma mark Tessellation dispatch

- (id<MTLBuffer>)isolatedStageBindingBufferForMap:(const BufferMap *)map
                                           source:(id<MTLBuffer>)source
                                   requiredLength:(NSUInteger)requiredLength
{
    if (!map || !map->buf || requiredLength == 0) {
        return nil;
    }

    id<MTLBuffer> isolated = mglRendererCreateBuffer(
        _device, requiredLength, MTLResourceStorageModeShared);
    if (!isolated || !isolated.contents) {
        return nil;
    }

    memset(isolated.contents, 0, requiredLength);
    if (!source || map->offset < 0 || !source.contents) {
        return isolated;
    }

    /* For UBOs, prefer the underlying store over the (possibly short) indexed
     * range so trailing std140 members remain visible after padding. */
    size_t copyLength = (map->resource_type == _UNIFORM_BUFFER_RES)
        ? mglBufferMapAvailableBackingBytes(map, source.length)
        : mglBufferMapVisibleBackingBytes(map, source.length);
    if (copyLength > requiredLength) {
        copyLength = requiredLength;
    }
    if (copyLength > 0) {
        memcpy(isolated.contents,
               ((const uint8_t *)source.contents) + (size_t)map->offset,
               copyLength);
    }
    return isolated;
}

- (void)clearStageBindingCopyBacks:(MGLStageBindingCopyBackList *)copyBacks
{
    if (!copyBacks) {
        return;
    }
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        [self clearStageBindingCopyBack:copyBacks atIndex:i];
    }
}

- (void)clearStageBindingCopyBack:(MGLStageBindingCopyBackList *)copyBacks
                           atIndex:(NSUInteger)index
{
    if (!copyBacks || index >= kMGLMaxBufferSlots) {
        return;
    }
    MGLStageBindingCopyBack *entry = &copyBacks->slots[index];
    entry->temporary = nil;
    entry->destination = nil;
    entry->destination_buffer = NULL;
    entry->destination_offset = 0;
    entry->length = 0;
}

- (bool)recordStageBindingCopyBack:(MGLStageBindingCopyBackList *)copyBacks
                           atIndex:(NSUInteger)index
                         temporary:(id<MTLBuffer>)temporary
                       destination:(id<MTLBuffer>)destination
                 destinationBuffer:(Buffer *)destinationBuffer
                destinationOffset:(NSUInteger)destinationOffset
                            length:(NSUInteger)length
{
    if (!copyBacks || index >= kMGLMaxBufferSlots) {
        return false;
    }
    [self clearStageBindingCopyBack:copyBacks atIndex:index];
    if (length == 0) {
        return true;
    }
    if (!temporary || !destination ||
        length > temporary.length ||
        destinationOffset > destination.length ||
        length > destination.length - destinationOffset) {
        return false;
    }

    MGLStageBindingCopyBack *entry = &copyBacks->slots[index];
    entry->temporary = temporary;
    entry->destination = destination;
    entry->destination_buffer = destinationBuffer;
    entry->destination_offset = destinationOffset;
    entry->length = length;
    return true;
}

- (bool)flushStageBindingCopyBacks:(MGLStageBindingCopyBackList *)copyBacks
              requireCPUVisibility:(BOOL)requireCPUVisibility
{
    if (!copyBacks) {
        return true;
    }

    /* P4.5 (item 1138): 把 copy-back 条目桥接成 C-ABI 数组；校验 + blit
     * 编码 + CPU 前缀同步在 C++（mglRenderCppEncodeStageBindingCopyBacks /
     * mglRenderCppCopyBackCPUPrefix——纯数据/编码，两门共用；CB 排序
     * （detach/commit/wait/AGX 恢复）仍在本方法）。 */
    MGLRenderCppCopyBackEntry entries[kMGLMaxBufferSlots];
    memset(entries, 0, sizeof(entries));
    uint32_t entryCount = 0;
    BOOL hasCopies = NO;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLStageBindingCopyBack *entry = &copyBacks->slots[i];
        if (entry->length == 0) {
            continue;
        }
        entries[entryCount].temporary = (__bridge void *)entry->temporary;
        entries[entryCount].destination = (__bridge void *)entry->destination;
        entries[entryCount].destination_buffer = entry->destination_buffer;
        entries[entryCount].destination_offset = entry->destination_offset;
        entries[entryCount].length = entry->length;
        entryCount++;
        hasCopies = YES;
    }

    if (mglRenderCppEncodeStageBindingCopyBacks(
            entries, entryCount, NULL) != 0) {
        [self clearStageBindingCopyBacks:copyBacks];
        return false;
    }

    if (!hasCopies && !requireCPUVisibility) {
        [self clearStageBindingCopyBacks:copyBacks];
        return true;
    }
    if (!_renderPassManager.state->currentCommandBuffer ||
        mglRenderCommandBufferStatus(
            _renderPassManager.state->currentCommandBuffer) !=
            MTLCommandBufferStatusNotEnqueued) {
        [self clearStageBindingCopyBacks:copyBacks];
        return false;
    }

    if (hasCopies) {
        id<MTLBlitCommandEncoder> blit =
            mglRendererCreateBlitEncoder(
                _renderPassManager.state->currentCommandBuffer);
        if (!blit) {
            [self clearStageBindingCopyBacks:copyBacks];
            return false;
        }
        if (mglRenderCppEncodeStageBindingCopyBacks(
                entries, entryCount, (__bridge void *)blit) != 0) {
            mglRendererEndBlitEncoder(blit);
            [self clearStageBindingCopyBacks:copyBacks];
            return false;
        }
        mglRendererEndBlitEncoder(blit);
    }

    /* Isolated copy-backs must become CPU-visible before another short binding
     * snapshots their destination. TCS also forces this boundary because TES
     * sizing and query accounting currently read its factor buffer on the CPU. */
    id<MTLCommandBuffer> stageCommandBuffer =
        [_renderPassManager detachCurrentCommandBufferForSubmission];
    @try {
        [self commitCommandBufferWithAGXRecovery:stageCommandBuffer];
        _lastCommittedCB = stageCommandBuffer;
        mglRendererWaitCommandBuffer(stageCommandBuffer);
    } @catch (NSException *exception) {
        NSLog(@"MGL BUFFER RANGE: stage synchronization failed: %@",
              exception.reason);
        [self clearStageBindingCopyBacks:copyBacks];
        [self newCommandBufferLocked];
        return false;
    }
    MGLRenderCppCommandBufferState stageState =
        mglRenderCommandBufferState(stageCommandBuffer);
    if (stageState.has_error) {
        NSLog(@"MGL BUFFER RANGE: stage command failed: %@",
              mglRenderCommandBufferErrorString(&stageState));
        [self clearStageBindingCopyBacks:copyBacks];
        [self newCommandBufferLocked];
        return false;
    }

    /* CPU 前缀同步（BufferSubData 的 CPU 快照保真 + 边界守卫 + memmove）
     * 在 C++。失败时 entries[failedIndex] 携带诊断所需字段。 */
    uint32_t failedIndex = entryCount;
    if (mglRenderCppCopyBackCPUPrefix(entries, entryCount, &failedIndex) != 0) {
        const MGLRenderCppCopyBackEntry *failed =
            failedIndex < entryCount ? &entries[failedIndex] : NULL;
        Buffer *failedBuffer = failed
            ? (Buffer *)(uintptr_t)failed->destination_buffer
            : NULL;
        NSLog(@"MGL BUFFER RANGE: cannot synchronize copied-back prefix to CPU buffer=%u offset=%llu length=%llu cpuSize=%llu",
              failedBuffer ? (unsigned)failedBuffer->name : 0u,
              (unsigned long long)(failed ? failed->destination_offset : 0ull),
              (unsigned long long)(failed ? failed->length : 0ull),
              (unsigned long long)(failedBuffer ? failedBuffer->data.buffer_size : 0ull));
        [self clearStageBindingCopyBacks:copyBacks];
        [self newCommandBufferLocked];
        return false;
    }
    [self clearStageBindingCopyBacks:copyBacks];
    return [self newCommandBufferLocked];
}


/* handleTessellationPatchDrawIfNeeded:(GLMContext)drawCtx moved to MGLRenderer+Draw.m */

#pragma mark C interface to mtlDrawArrays
/* mtlDrawArrays: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count moved to MGLRenderer+Draw.m */

/* mtlDrawArraysLocked: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count moved to MGLRenderer+Draw.m */

#pragma mark C interface to mtlDrawElements
/* mtlDrawElements: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices moved to MGLRenderer+Draw.m */

/* mtlDrawElementsLocked: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawRangeElements
/* mtlDrawRangeElements: (GLMContext) glm_ctx mode:(GLenum) mode start:(GLuint) start end:(GLuint) end count: (GLsizei) count type: (GLenum) type indices:(const void *)indices moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawArraysInstanced
/* mtlDrawArraysInstanced: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawElementsInstanced
/* mtlDrawElementsInstanced: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawElementsBaseVertex
/* mtlDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawRangeElementsBaseVertex
/* mtlDrawRangeElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode start: (GLuint) start end: (GLuint) end count:(GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawElementsInstancedBaseVertex
/* mtlDrawElementsInstancedBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count:(GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex moved to MGLRenderer+Draw.m */

#pragma mark C interface to mtlDrawArraysIndirect
/* mtlDrawArraysIndirect: (GLMContext) glm_ctx mode:(GLenum) mode indirect: (const void *) indirect moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawElementsIndirect
/* mtlDrawElementsIndirect: (GLMContext) glm_ctx mode:(GLenum) mode type:(GLenum) type indirect: (const void *) indirect moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawArraysInstancedBaseInstance
/* mtlDrawArraysInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawElementsInstancedBaseInstance
/* mtlDrawElementsInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode  count: (GLsizei) count type:(GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlDrawElementsInstancedBaseVertexBaseInstance
/* mtlDrawElementsInstancedBaseVertexBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type:(GLenum) type indices:(const void *)indices moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlMultiDrawArrays
/* mtlMultiDrawArrays: (GLMContext)glm_ctx mode:(GLenum) mode first:(const GLint *)first count:(const GLsizei *)count drawcount:(GLsizei) drawcount moved to MGLRenderer+Draw.m */


#pragma mark C interface to mtlMultiDrawElements
/* mtlMultiDrawElements: (GLMContext)glm_ctx mode:(GLenum) mode count:(const GLsizei *)count type:(GLenum)type indices:(const void *const*)indices drawcount:(GLsizei) drawcount moved to MGLRenderer+Draw.m */




#pragma mark C interface to mtlMultiDrawElementsBaseVertex
/* mtlMultiDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (const GLsizei *) count type: (GLenum) type indices:(const void *const *)indices drawcount:(GLsizei) drawcount basevertex:(const GLint *) basevertex moved to MGLRenderer+Draw.m */


/* mtlMultiDrawArraysIndirect: (GLMContext)glm_ctx mode:(GLenum) mode indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride moved to MGLRenderer+Draw.m */


/* mtlMultiDrawElementsIndirect: (GLMContext)glm_ctx mode:(GLenum) mode type:(GLenum)type indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride moved to MGLRenderer+Draw.m */

/* FIFO eviction for auxiliary caches.  NSDictionary enumerates in
 * insertion order on recent macOS runtimes, so removing the first 1/4 of
 * allKeys evicts the oldest entries — matching the _pipelineCache.state->pipelineStateCache
 * strategy.  Called at each insertion site after the new entry is added. */
- (void)mglCapAuxCache:(NSMutableDictionary *)cache
                 limit:(NSUInteger)limit
{
    if (!cache || limit == 0) return;
    if (cache.count <= limit) return;
    NSArray *keys = cache.allKeys;
    NSUInteger evictCount = keys.count / 4;
    if (evictCount < 1) evictCount = 1;
    for (NSUInteger i = 0; i < evictCount; i++) {
        [cache removeObjectForKey:keys[i]];
    }
}

@end
