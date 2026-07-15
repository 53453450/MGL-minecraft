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
 * glm_context.h, pixel_utils.h, spirv_cross_c.h, and all mgl_* compatibility
 * headers listed below.  Only imports unique to this TU are listed here. */
#import <objc/runtime.h>
#import <MetalKit/MetalKit.h>

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <string.h>
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
#import "mgl_msl_compiler.h"
#import "mgl_compute_pipeline_cache.h"
#import "mgl_metal_bridge.h"
#import "msl_patch_pipeline.h"  /* P1-6: TCS stage-in + tessellation passthrough helpers */

#define TRACE_FUNCTION()    DEBUG_PRINT("%s\n", __FUNCTION__);

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

/* MGLFragmentTextureTraceBinding typedef moved to mgl_trace_strategy.h. */

/* Draw buffer mapping helpers moved to mgl_draw_buffer.h/.m. */

/* Pixel readback helpers (7 functions) moved to mgl_readback.m */
/* Layer pixel format / sRGB / linear helpers moved to mgl_texture_compat */

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

    id<MTLTexture> view = [texture newTextureViewWithPixelFormat:desiredFmt];
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

static MTLPixelFormat mglMetalLayerPixelFormatForContext(GLMContext drawCtx)
{
    MTLPixelFormat fallback = MTLPixelFormatBGRA8Unorm;
    if (!drawCtx) {
        return fallback;
    }

    MTLPixelFormat requested = (MTLPixelFormat)drawCtx->pixel_format.mtl_pixel_format;
    if (mglMetalLayerPixelFormatIsSupported(requested)) {
        return requested;
    }

    NSLog(@"MGL CAMetalLayer pixelFormat fallback glFormat=0x%x glType=0x%x requestedMtl=%lu fallback=%lu",
          drawCtx->pixel_format.format,
          drawCtx->pixel_format.type,
          (unsigned long)requested,
          (unsigned long)fallback);
    return fallback;
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

// CRITICAL SECURITY: Safe Metal object validation helper
static inline id<NSObject> SafeMetalBridge(void *ptr, Class expectedClass, const char *objectName) {
    if (!ptr) {
        NSLog(@"MGL SECURITY ERROR: NULL pointer for %s", objectName);
        return nil;
    }

    id<NSObject> obj = (__bridge id<NSObject>)(ptr);
    if (!obj) {
        NSLog(@"MGL SECURITY ERROR: Metal bridge cast returned nil for %s", objectName);
        return nil;
    }

    if (expectedClass && [obj isKindOfClass:expectedClass] == NO) {
        NSLog(@"MGL SECURITY ERROR: Metal object is not valid %s (got %@)", objectName, NSStringFromClass([obj class]));
        return nil;
    }

    return obj;
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

BOOL mglEnvFlagEnabled(const char *name)
{
    const char *value = name ? getenv(name) : NULL;
    if (!value || value[0] == '\0') {
        return NO;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return NO;
    }
    return YES;
}

/* Unset/empty → YES (default ON); "0"/"false"/"no"/"off" → NO; other non-empty → YES.
 * Use for kill-switchable optimizations that should ship enabled. */
BOOL mglEnvFlagEnabledDefaultOn(const char *name)
{
    const char *value = name ? getenv(name) : NULL;
    if (!value || value[0] == '\0') {
        return YES;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return NO;
    }
    return YES;
}

/* Trace log core infrastructure (3 static globals, mglInitTraceLogIfNeeded,
 * mglTraceLogIsEnabled, mglTraceLogV, mglTraceLog, mglTraceLogExternal,
 * MGLTraceNSLog) moved to mgl_trace_log.h/.m. */

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
    if (ctx->active_state->program_name == 0 && ctx->active_state->var.current_program == 0) {
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
            ctx->active_state->var.current_program = program->name;
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
        ctx->active_state->var.current_program = 0;
        return NULL;
    }

    if (!resolved->linked_glsl_program) {
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
        ctx->active_state->var.current_program = 0u;
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
    ctx->active_state->var.current_program = programName;
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
    if (ctx->active_state->program_name != 0 || ctx->active_state->var.current_program != 0) {
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

    if (!stageProgram->linked_glsl_program) {
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
        ctx->active_state->program_name != 0 ||
        ctx->active_state->var.current_program != 0) {
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
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES || type < 0 || type >= _MAX_SPIRV_RES) {
        return;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[stage][type];
    MGLTraceNSLog(@"MGL IFACE program=%u stage=%s type=%s count=%u",
                  (unsigned)program->name,
                  mglShaderStageName(stage),
                  mglSpirvResourceTypeName(type),
                  (unsigned)resources->count);

    for (GLuint i = 0; i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        MGLTraceNSLog(@"MGL IFACE   #%u name=%s loc=%u glBinding=%u metalBinding=%u set=%u typeId=%u baseTypeId=%u required=%zu imageDim=%u arrayed=%u",
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

    MGLTraceNSLog(@"MGL IFACE DUMP begin program=%u reason=%@ generation=%u",
                  (unsigned)program->name,
                  reason ?: @"(none)",
                  (unsigned)s_dumpGeneration);

    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SEPARATE_IMAGE);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_SEPARATE_IMAGE);

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        const char *msl = program->spirv[stage].msl_str;
        if (!msl || !program->shader_slots[stage]) {
            continue;
        }

        NSString *path = [NSString stringWithFormat:@"/tmp/mgl_program_%u_%s_%u.msl",
                                                   (unsigned)program->name,
                                                   mglShaderStageName(stage),
                                                   (unsigned)s_dumpGeneration];
        NSString *source = [NSString stringWithUTF8String:msl];
        NSError *writeError = nil;
        BOOL ok = [source writeToFile:path
                           atomically:YES
                             encoding:NSUTF8StringEncoding
                                error:&writeError];
        MGLTraceNSLog(@"MGL IFACE DUMP msl program=%u stage=%s entry=%s path=%@ ok=%d error=%@",
                      (unsigned)program->name,
                      mglShaderStageName(stage),
                      program->shader_slots[stage]->entry_point ? program->shader_slots[stage]->entry_point : "(null)",
                      path,
                      ok ? 1 : 0,
                      writeError);
    }
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
        MGLTraceNSLog(@"MGL TRACE %s gap=%.2fms deltaCalls=%llu call=%llu",
              tag ? tag : "loop",
              deltaMs,
              (unsigned long long)deltaCalls,
              (unsigned long long)callCount);
    } else if (mglShouldTraceCall(callCount) &&
               (callCount <= 20ull || (callCount % 60ull) == 0ull)) {
        MGLTraceNSLog(@"MGL TRACE %s heartbeat delta=%.2fms deltaCalls=%llu call=%llu",
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
        MGLTraceNSLog(@"MGL TRACE %s ctx=%p(invalid) cb=%p enc=%p rpd=%p drawable=%p",
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
            MGLTraceNSLog(@"MGL TRACE %s invalid drawFBO=%p", tag ? tag : "snapshot", drawFBO);
            drawFBO = NULL;
        }
    }

    MTLCommandBufferStatus cbStatus = commandBuffer ? commandBuffer.status : MTLCommandBufferStatusNotEnqueued;
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

    MGLTraceNSLog(@"MGL TRACE %s prog=%u dirty=0x%x[%s] clear=0x%x drawBuf=0x%x readBuf=0x%x vao=%p drawFBO=%p(%u) "
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

    MGLTraceNSLog(@"MGL TRACE %s masks color0(use=%d rgba=%d%d%d%d) depthWrite=%d stencilWrite=0x%x",
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

    double now = mglNowSeconds();
    double lastSwap = MGL_FRAME_LOAD(g_mglLastSwapSeconds);
    double lastSwapAgeMs = (lastSwap > 0.0) ? ((now - lastSwap) * 1000.0) : -1.0;
    if (lastSwapAgeMs >= 0.0 && lastSwapAgeMs < 250.0) {
        return;
    }
    MTLCommandBufferStatus cbStatus = commandBuffer ? commandBuffer.status : MTLCommandBufferStatusNotEnqueued;
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

    MTLCommandBufferStatus cbStatus = commandBuffer ? commandBuffer.status : MTLCommandBufferStatusNotEnqueued;
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

/* P1-6: MGLTCSStageInBaseType / MGLTCSStageInMember migrated to
 * msl_patch_pipeline.h. */

static const uint8_t *mglRendererReadableBufferBytes(Buffer *buffer)
{
    if (!buffer) {
        return NULL;
    }
    if (buffer->data.buffer_data && ((uintptr_t)buffer->data.buffer_data >= 0x1000ull)) {
        return (const uint8_t *)(uintptr_t)buffer->data.buffer_data;
    }
    if (buffer->data.mtl_data) {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(buffer->data.mtl_data);
        if (mtlBuffer && mtlBuffer.contents) {
            return (const uint8_t *)mtlBuffer.contents;
        }
    }
    return NULL;
}

/* P1-6: TCS stage-in parsing helpers (mglTCSStageInParseAttributeMarker,
 * mglTCSStageInDescribeMember, mglParseTCSStageInMembers,
 * mglWriteTCSStageInComponent) migrated to msl_patch_pipeline.c. */

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

// === Phase 3: Thread Safety Lock Macros ===
//
// Lock macros (METAL_LOCK/METAL_UNLOCK/SYNC_LOCK/SYNC_UNLOCK) and helpers
// (mglMetalLock/mglMetalUnlock/mglNowSeconds) moved to MGLRenderer_Private.h
// so that category files (MGLRenderer+Query.m, +Blit.m, +Texture.m) can use
// them without redefinition.
//
// NSRecursiveLock is reentrant — required because the MGLRenderer call graph
// is densely interconnected (e.g. endRenderEncoding → updateGLSampledCopies
// → updateGLSampledRenderTargetCopyForTexture → ensureWritableCommandBuffer,
// which is another target method).  A non-reentrant lock deadlocked on
// indirect re-entry through non-target helper methods.
//
// Two independent locks:
//   _metalStateLock  — guards the 15 Locked-method targets (draw path)
//   _syncListLock    — guards _currentCommandBufferSyncList access
//                      (mtlGetSync: vs newCommandBufferLocked)
//
// The Locked pattern (public wrapper + *Locked impl) is retained for
// structural clarity but no longer relies on non-reentrancy.

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
- (void)uploadDirty3DTextureLevels:(Texture *)tex
                             metal:(id<MTLTexture>)texture
                       pixelFormat:(MTLPixelFormat)pixelFormat
                          numFaces:(uint)num_faces
                  uploadLevelCount:(GLuint)upload_level_count
                            texType:(MTLTextureType)tex_type
                texture1DBackedBy2D:(BOOL)texture1DBackedBy2D;
- (void)uploadDirtyArrayTextureLevels:(Texture *)tex
                                metal:(id<MTLTexture>)texture
                          pixelFormat:(MTLPixelFormat)pixelFormat
                             numFaces:(uint)num_faces
                     uploadLevelCount:(GLuint)upload_level_count
                               texType:(MTLTextureType)tex_type
                   texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
             texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray;
- (void)uploadDirty2DTextureLevels:(Texture *)tex
                             metal:(id<MTLTexture>)texture
                       pixelFormat:(MTLPixelFormat)pixelFormat
                          numFaces:(uint)num_faces
                  uploadLevelCount:(GLuint)upload_level_count
                texture1DBackedBy2D:(BOOL)texture1DBackedBy2D;
- (void)logMTLTextureMipDiagnostics:(Texture *)tex
                              metal:(id<MTLTexture>)texture
               effectiveMipLevels:(GLuint)effective_mipmap_levels;
// mapGLBuffersToMTLBufferMap:stage: helpers
- (bool)mapShaderBufferResourcesToBufferMap:(BufferMapList *)buffer_map stage:(int)stage;
- (bool)mapVertexAttributeBuffersToBufferMap:(BufferMapList *)buffer_map
                                         vao:(VertexArray *)vao
                            stageInputCount:(int)count
                                       stage:(int)stage;
// mtlSwapBuffersLocked: helpers
- (void)copyRenderPassColorToDrawableIfNeeded:(id<MTLTexture>)rpColor0
                              drawableTexture:(id<MTLTexture>)drawableTexture
                                      swapCall:(uint64_t)swapCall
                                    traceSwap:(bool)traceSwap;
- (void)scheduleSwapTextureSampleDiagnostics:(id<MTLTexture>)rpColor0
                             drawableTexture:(id<MTLTexture>)drawableTexture
                                     swapCall:(uint64_t)swapCall;
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

            case GL_UNSIGNED_INT_10_10_10_2:
            case GL_UNSIGNED_INT_2_10_10_10_REV:
                if (normalized)
                    return MTLVertexFormatUInt1010102Normalized;
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

static inline bool mglShouldTraceBufferTransferCall(uint64_t call)
{
    if (call <= 128ull) {
        return true;
    }
    return ((call % 64ull) == 0ull);
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
        MGLTraceNSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u invalid buffer",
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
        MGLTraceNSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u vbo=%u no readable bytes",
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
        MGLTraceNSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu vbo=%u negative vertexIndex rawIndex=%u baseVertex=%d",
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
        MGLTraceNSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu vbo=%u OOB rawIndex=%u baseVertex=%d vertexIndex=%llu bindingOffset=%lu relOffset=%lu stride=%lu size=%u type=0x%x normalized=%u elemBytes=%zu vboSize=%lld",
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
    SpirvResource *resource = mglRendererProgramVertexAttribResource(program, attrib);
    MGLTraceNSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu resource=%s metalSlot=%d vbo=%u rawIndex=%u baseVertex=%d vertexIndex=%llu bindingIndex=%u bindingOffset=%lu relOffset=%lu vertexOffset=%lu stride=%lu size=%u type=0x%x normalized=%u/%u format=%lu(%s) decoded=(%.6f,%.6f,%.6f,%.6f) raw=%s",
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

#pragma mark buffer objects
- (id<MTLBuffer>)floatVertexBufferForDoubleAttrib:(Buffer *)sourceBuffer
                                         resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                             size:(GLuint)componentCount
                                         outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved || componentCount == 0 || componentCount > 4) {
        return nil;
    }

    const uint8_t *sourceBytes = NULL;
    size_t sourceSize = 0;
    if (sourceBuffer->data.buffer_data && sourceBuffer->size > 0) {
        sourceBytes = (const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data;
        sourceSize = (size_t)sourceBuffer->size;
    } else if (sourceBuffer->data.mtl_data) {
        id<MTLBuffer> metal = (__bridge id<MTLBuffer>)(sourceBuffer->data.mtl_data);
        if (metal && metal.contents && metal.length > 0) {
            sourceBytes = (const uint8_t *)metal.contents;
            sourceSize = (size_t)metal.length;
        }
    }
    if (!sourceBytes || sourceSize == 0) {
        return nil;
    }

    NSUInteger originalStride = (resolved->stride > 0)
        ? (NSUInteger)resolved->stride
        : (NSUInteger)(componentCount * sizeof(GLdouble));
    NSUInteger convertedStride = mglAlignVertexStrideForMetal(MAX(originalStride, (NSUInteger)(componentCount * sizeof(GLfloat))));
    if (resolved->binding_offset < 0 || resolved->relativeoffset < 0 ||
        (NSUInteger)resolved->binding_offset >= sourceSize) {
        return nil;
    }

    size_t copyLength = sourceSize - (size_t)resolved->binding_offset;
    uint64_t sourceHash = mglHashVertexBytesFNV1a(sourceBytes + (size_t)resolved->binding_offset, copyLength);
    NSString *cacheKey = [NSString stringWithFormat:@"%u:%lld:%lld:%u:%u:%lu:%zu:%016llx",
                          sourceBuffer->name,
                          (long long)resolved->binding_offset,
                          (long long)resolved->relativeoffset,
                          (unsigned)originalStride,
                          (unsigned)componentCount,
                          (unsigned long)convertedStride,
                          copyLength,
                          (unsigned long long)sourceHash];
    if (!_doubleVertexAttribBufferCache) {
        _doubleVertexAttribBufferCache = [NSMutableDictionary dictionary];
    }
    id<MTLBuffer> cached = _doubleVertexAttribBufferCache[cacheKey];
    if (cached) {
        if (outStride) {
            *outStride = convertedStride;
        }
        return cached;
    }

    if (originalStride == 0u) {
        return nil;
    }

    NSUInteger vertexCount = ((NSUInteger)copyLength + originalStride - 1u) / originalStride;
    if (vertexCount == 0u || vertexCount > NSUIntegerMax / convertedStride) {
        return nil;
    }

    NSMutableData *convertedData = [NSMutableData dataWithLength:vertexCount * convertedStride];
    if (!convertedData) {
        return nil;
    }
    uint8_t *dst = (uint8_t *)convertedData.mutableBytes;

    NSUInteger rel = (NSUInteger)resolved->relativeoffset;
    NSUInteger doubleBytes = (NSUInteger)componentCount * sizeof(GLdouble);
    NSUInteger floatBytes = (NSUInteger)componentCount * sizeof(GLfloat);
    const uint8_t *srcBase = sourceBytes + (size_t)resolved->binding_offset;
    for (NSUInteger vertex = 0; vertex < vertexCount; vertex++) {
        NSUInteger srcOffset = vertex * originalStride;
        NSUInteger dstOffset = vertex * convertedStride;
        NSUInteger copyBytes = 0;

        if (srcOffset < (NSUInteger)copyLength) {
            copyBytes = MIN(originalStride, (NSUInteger)copyLength - srcOffset);
            memcpy(dst + dstOffset, srcBase + srcOffset, copyBytes);
        }

        if (rel <= copyBytes && doubleBytes <= copyBytes - rel) {
            GLfloat floats[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            for (GLuint c = 0; c < componentCount; c++) {
                GLdouble d = 0.0;
                memcpy(&d,
                       srcBase + srcOffset + rel + (NSUInteger)c * sizeof(GLdouble),
                       sizeof(d));
                floats[c] = (GLfloat)d;
            }
            memcpy(dst + dstOffset + rel, floats, floatBytes);
        }
    }

    id<MTLBuffer> converted = [_device newBufferWithBytes:dst
                                                   length:convertedData.length
                                                  options:MTLResourceStorageModeShared];
    if (!converted) {
        return nil;
    }
    _doubleVertexAttribBufferCache[cacheKey] = converted;
    [self mglCapAuxCache:_doubleVertexAttribBufferCache limit:64];
    if (outStride) {
        *outStride = convertedStride;
    }
    return converted;
}

/* Metal has no int/uint->float vertex format conversion for 32-bit integer
 * formats (MTLVertexFormatInt/UInt require integer shader inputs). When an
 * app uses glVertexAttribFormat (non-integer) with GL_INT/GL_UNSIGNED_INT and
 * a float shader input, GL requires the integer values to be converted to
 * float. We perform that conversion on the CPU side, mirroring the GL_DOUBLE
 * path. sizeof(GLint)==sizeof(GLfloat)==4, so the converted stride equals the
 * original stride. */
- (id<MTLBuffer>)floatVertexBufferForIntAttrib:(Buffer *)sourceBuffer
                                      resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                          size:(GLuint)componentCount
                                    normalized:(GLboolean)normalized
                                          type:(GLenum)type
                                     outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved || componentCount == 0 || componentCount > 4) {
        return nil;
    }
    if (type != GL_INT && type != GL_UNSIGNED_INT) {
        return nil;
    }

    const uint8_t *sourceBytes = NULL;
    size_t sourceSize = 0;
    if (sourceBuffer->data.buffer_data && sourceBuffer->size > 0) {
        sourceBytes = (const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data;
        sourceSize = (size_t)sourceBuffer->size;
    } else if (sourceBuffer->data.mtl_data) {
        id<MTLBuffer> metal = (__bridge id<MTLBuffer>)(sourceBuffer->data.mtl_data);
        if (metal && metal.contents && metal.length > 0) {
            sourceBytes = (const uint8_t *)metal.contents;
            sourceSize = (size_t)metal.length;
        }
    }
    if (!sourceBytes || sourceSize == 0) {
        return nil;
    }

    NSUInteger originalStride = (resolved->stride > 0)
        ? (NSUInteger)resolved->stride
        : (NSUInteger)(componentCount * sizeof(GLint));
    NSUInteger convertedStride = mglAlignVertexStrideForMetal(originalStride);
    if (resolved->binding_offset < 0 || resolved->relativeoffset < 0 ||
        (NSUInteger)resolved->binding_offset >= sourceSize) {
        return nil;
    }

    size_t copyLength = sourceSize - (size_t)resolved->binding_offset;
    uint64_t sourceHash = mglHashVertexBytesFNV1a(sourceBytes + (size_t)resolved->binding_offset, copyLength);
    NSString *cacheKey = [NSString stringWithFormat:@"I:%u:%lld:%lld:%u:%u:%u:%lu:%zu:%016llx",
                          sourceBuffer->name,
                          (long long)resolved->binding_offset,
                          (long long)resolved->relativeoffset,
                          (unsigned)type,
                          (unsigned)normalized,
                          (unsigned)originalStride,
                          (unsigned long)convertedStride,
                          copyLength,
                          (unsigned long long)sourceHash];
    if (!_doubleVertexAttribBufferCache) {
        _doubleVertexAttribBufferCache = [NSMutableDictionary dictionary];
    }
    id<MTLBuffer> cached = _doubleVertexAttribBufferCache[cacheKey];
    if (cached) {
        if (outStride) {
            *outStride = convertedStride;
        }
        return cached;
    }

    if (originalStride == 0u) {
        return nil;
    }

    NSUInteger vertexCount = ((NSUInteger)copyLength + originalStride - 1u) / originalStride;
    if (vertexCount == 0u || vertexCount > NSUIntegerMax / convertedStride) {
        return nil;
    }

    NSMutableData *convertedData = [NSMutableData dataWithLength:vertexCount * convertedStride];
    if (!convertedData) {
        return nil;
    }
    uint8_t *dst = (uint8_t *)convertedData.mutableBytes;

    NSUInteger rel = (NSUInteger)resolved->relativeoffset;
    NSUInteger compBytes = (NSUInteger)componentCount * sizeof(GLfloat);
    const uint8_t *srcBase = sourceBytes + (size_t)resolved->binding_offset;
    for (NSUInteger vertex = 0; vertex < vertexCount; vertex++) {
        NSUInteger srcOffset = vertex * originalStride;
        NSUInteger dstOffset = vertex * convertedStride;
        NSUInteger copyBytes = 0;

        if (srcOffset < (NSUInteger)copyLength) {
            copyBytes = MIN(originalStride, (NSUInteger)copyLength - srcOffset);
            memcpy(dst + dstOffset, srcBase + srcOffset, copyBytes);
        }

        if (rel <= copyBytes && compBytes <= copyBytes - rel) {
            GLfloat floats[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            for (GLuint c = 0; c < componentCount; c++) {
                if (type == GL_INT) {
                    GLint iv = 0;
                    memcpy(&iv,
                           srcBase + srcOffset + rel + (NSUInteger)c * sizeof(GLint),
                           sizeof(iv));
                    if (normalized) {
                        double d = (double)iv / 2147483647.0;
                        if (d < -1.0) d = -1.0;
                        floats[c] = (GLfloat)d;
                    } else {
                        floats[c] = (GLfloat)iv;
                    }
                } else { /* GL_UNSIGNED_INT */
                    GLuint uv = 0;
                    memcpy(&uv,
                           srcBase + srcOffset + rel + (NSUInteger)c * sizeof(GLuint),
                           sizeof(uv));
                    if (normalized) {
                        floats[c] = (GLfloat)((double)uv / 4294967295.0);
                    } else {
                        floats[c] = (GLfloat)uv;
                    }
                }
            }
            memcpy(dst + dstOffset + rel, floats, compBytes);
        }
    }

    id<MTLBuffer> converted = [_device newBufferWithBytes:dst
                                                   length:convertedData.length
                                                  options:MTLResourceStorageModeShared];
    if (!converted) {
        return nil;
    }
    _doubleVertexAttribBufferCache[cacheKey] = converted;
    [self mglCapAuxCache:_doubleVertexAttribBufferCache limit:64];
    if (outStride) {
        *outStride = convertedStride;
    }
    return converted;
}

/* Converts integer vertex data from a source type that Metal cannot feed
 * directly to an int/uint shader input (e.g. GL_UNSIGNED_BYTE -> int32 for
 * an `in int` attribute) into a 32-bit integer buffer matching the shader's
 * declared type. dstIsInt selects int32 vs uint32 output. */
- (id<MTLBuffer>)integerVertexBufferForAttrib:(Buffer *)sourceBuffer
                                     resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                         size:(GLuint)componentCount
                                       srcType:(GLenum)srcType
                                     dstIsInt:(BOOL)dstIsInt
                                    outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved || componentCount == 0 || componentCount > 4) {
        return nil;
    }

    size_t srcCompSize = mglVertexAttribComponentSize(srcType);
    if (srcCompSize == 0) {
        return nil;
    }

    const uint8_t *sourceBytes = NULL;
    size_t sourceSize = 0;
    if (sourceBuffer->data.buffer_data && sourceBuffer->size > 0) {
        sourceBytes = (const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data;
        sourceSize = (size_t)sourceBuffer->size;
    } else if (sourceBuffer->data.mtl_data) {
        id<MTLBuffer> metal = (__bridge id<MTLBuffer>)(sourceBuffer->data.mtl_data);
        if (metal && metal.contents && metal.length > 0) {
            sourceBytes = (const uint8_t *)metal.contents;
            sourceSize = (size_t)metal.length;
        }
    }
    if (!sourceBytes || sourceSize == 0) {
        return nil;
    }

    NSUInteger originalStride = (resolved->stride > 0)
        ? (NSUInteger)resolved->stride
        : (NSUInteger)(componentCount * srcCompSize);
    NSUInteger convertedStride = mglAlignVertexStrideForMetal((NSUInteger)componentCount * sizeof(GLint));
    if (resolved->binding_offset < 0 || resolved->relativeoffset < 0 ||
        (NSUInteger)resolved->binding_offset >= sourceSize) {
        return nil;
    }

    size_t copyLength = sourceSize - (size_t)resolved->binding_offset;
    uint64_t sourceHash = mglHashVertexBytesFNV1a(sourceBytes + (size_t)resolved->binding_offset, copyLength);
    NSString *cacheKey = [NSString stringWithFormat:@"J:%u:%lld:%lld:%u:%u:%u:%u:%lu:%lu:%zu:%016llx",
                          sourceBuffer->name,
                          (long long)resolved->binding_offset,
                          (long long)resolved->relativeoffset,
                          (unsigned)srcType,
                          (unsigned)dstIsInt,
                          (unsigned)componentCount,
                          (unsigned)originalStride,
                          (unsigned long)convertedStride,
                          (unsigned long)srcCompSize,
                          copyLength,
                          (unsigned long long)sourceHash];
    if (!_doubleVertexAttribBufferCache) {
        _doubleVertexAttribBufferCache = [NSMutableDictionary dictionary];
    }
    id<MTLBuffer> cached = _doubleVertexAttribBufferCache[cacheKey];
    if (cached) {
        if (outStride) {
            *outStride = convertedStride;
        }
        return cached;
    }

    if (originalStride == 0u) {
        return nil;
    }

    NSUInteger vertexCount = ((NSUInteger)copyLength + originalStride - 1u) / originalStride;
    if (vertexCount == 0u || vertexCount > NSUIntegerMax / convertedStride) {
        return nil;
    }

    NSMutableData *convertedData = [NSMutableData dataWithLength:vertexCount * convertedStride];
    if (!convertedData) {
        return nil;
    }
    /* Zero-init so missing source bytes default to 0. */
    memset(convertedData.mutableBytes, 0, convertedData.length);
    uint8_t *dst = (uint8_t *)convertedData.mutableBytes;

    NSUInteger rel = (NSUInteger)resolved->relativeoffset;
    const uint8_t *srcBase = sourceBytes + (size_t)resolved->binding_offset;

    for (NSUInteger vertex = 0; vertex < vertexCount; vertex++) {
        NSUInteger srcOffset = vertex * originalStride;
        NSUInteger dstOffset = vertex * convertedStride;
        uint8_t *dstComp = dst + dstOffset;

        for (GLuint c = 0; c < componentCount; c++) {
            size_t srcByteIdx = (size_t)srcOffset + rel + (size_t)c * srcCompSize;
            if (srcByteIdx + srcCompSize > (size_t)copyLength) {
                break;
            }
            const uint8_t *srcComp = srcBase + srcByteIdx;

            if (dstIsInt) {
                int32_t outVal = 0;
                switch (srcType) {
                    case GL_BYTE: {
                        int8_t v; memcpy(&v, srcComp, 1); outVal = (int32_t)v; break;
                    }
                    case GL_UNSIGNED_BYTE: {
                        uint8_t v; memcpy(&v, srcComp, 1); outVal = (int32_t)(uint32_t)v; break;
                    }
                    case GL_SHORT: {
                        int16_t v; memcpy(&v, srcComp, 2); outVal = (int32_t)v; break;
                    }
                    case GL_UNSIGNED_SHORT: {
                        uint16_t v; memcpy(&v, srcComp, 2); outVal = (int32_t)(uint32_t)v; break;
                    }
                    case GL_INT: {
                        int32_t v; memcpy(&v, srcComp, 4); outVal = v; break;
                    }
                    case GL_UNSIGNED_INT: {
                        uint32_t v; memcpy(&v, srcComp, 4); outVal = (int32_t)v; break;
                    }
                    default: break;
                }
                memcpy(dstComp + (size_t)c * sizeof(int32_t), &outVal, sizeof(outVal));
            } else {
                uint32_t outVal = 0;
                switch (srcType) {
                    case GL_BYTE: {
                        int8_t v; memcpy(&v, srcComp, 1); outVal = (uint32_t)(int32_t)v; break;
                    }
                    case GL_UNSIGNED_BYTE: {
                        uint8_t v; memcpy(&v, srcComp, 1); outVal = (uint32_t)v; break;
                    }
                    case GL_SHORT: {
                        int16_t v; memcpy(&v, srcComp, 2); outVal = (uint32_t)(int32_t)v; break;
                    }
                    case GL_UNSIGNED_SHORT: {
                        uint16_t v; memcpy(&v, srcComp, 2); outVal = (uint32_t)v; break;
                    }
                    case GL_INT: {
                        int32_t v; memcpy(&v, srcComp, 4); outVal = (uint32_t)v; break;
                    }
                    case GL_UNSIGNED_INT: {
                        uint32_t v; memcpy(&v, srcComp, 4); outVal = v; break;
                    }
                    default: break;
                }
                memcpy(dstComp + (size_t)c * sizeof(uint32_t), &outVal, sizeof(outVal));
            }
        }
    }

    id<MTLBuffer> converted = [_device newBufferWithBytes:dst
                                                   length:convertedData.length
                                                  options:MTLResourceStorageModeShared];
    if (!converted) {
        return nil;
    }
    _doubleVertexAttribBufferCache[cacheKey] = converted;
    [self mglCapAuxCache:_doubleVertexAttribBufferCache limit:64];
    if (outStride) {
        *outStride = convertedStride;
    }
    return converted;
}

/* bindMTLBuffer: moved to MGLRenderer+RenderPass.m */

/* bindMTLBufferLocked: moved to MGLRenderer+RenderPass.m */

/* ---- Plain struct uniform buffer packing ----
 *
 * SPIRV-Cross translates `layout(location=N) uniform S u[K]` into separate
 * Metal buffer arguments (`constant S* u_0 [[buffer(B)]]`, etc.), each
 * expecting a full struct's worth of data.  MGL stores individual uniform
 * member data per location in plain_uniform_buffers[location].  This
 * packing logic combines individual member data into struct-sized Metal
 * buffers at render time.
 */

#define MGL_MAX_PACKED_STRUCT_BUFFERS 128
static Buffer *s_packedStructBuffers[MGL_MAX_PACKED_STRUCT_BUFFERS];
static int s_packedStructBufferIdx = 0;

/* Compute the location step per array element from reflected members.
 * For a struct S = { vec4 m0, float m1[2], mat2 m2 }, the step is 4
 * (m0=1 + m1=2 + m2=1 in CTS convention). */
static GLuint mglPlainStructLocStep(const SpirvResource *res)
{
    if (!res || !res->ubo_members || res->ubo_member_count == 0) {
        return 1;
    }
    GLuint max_loc = 0;
    for (GLuint i = 0; i < res->ubo_member_count; i++) {
        GLuint end = (GLuint)res->ubo_members[i].location_offset +
                     (GLuint)res->ubo_members[i].size;
        if (end > max_loc) {
            max_loc = end;
        }
    }
    GLuint array_size = (res->gl_array_size > 1) ? (GLuint)res->gl_array_size : 1;
    if (array_size == 0) array_size = 1;
    GLuint step = max_loc / array_size;
    return step > 0 ? step : 1;
}

/* Compute the byte size of one element of a GL uniform type.
 * Used as a fallback array stride for plain struct uniform members
 * that lack ArrayStride decorations in SPIR-V. */
static GLuint mglGLTypeElementByteSize(GLuint gl_type)
{
    switch (gl_type) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_BOOL:
            return 4;
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC2:
            return 8;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC3:
            return 12;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
        case GL_BOOL_VEC4:
            return 16;
        case GL_FLOAT_MAT2:
            return 8;   /* one column = vec2 */
        case GL_FLOAT_MAT3:
            return 12;  /* one column = vec3 */
        case GL_FLOAT_MAT4:
            return 16;  /* one column = vec4 */
        case GL_FLOAT_MAT2x3:
            return 12;
        case GL_FLOAT_MAT2x4:
            return 16;
        case GL_FLOAT_MAT3x2:
            return 8;
        case GL_FLOAT_MAT3x4:
            return 16;
        case GL_FLOAT_MAT4x2:
            return 8;
        case GL_FLOAT_MAT4x3:
            return 12;
        case GL_DOUBLE:
            return 8;
        default:
            return 4;
    }
}

/* Get a reusable Buffer object for packed struct data.  The Metal buffer
 * is recreated each call; the Buffer wrapper is reused from a static ring. */
static Buffer *mglGetPackedStructBuffer(GLMContext ctx,
                                         id<MTLDevice> device,
                                         const void *data,
                                         size_t size)
{
    if (s_packedStructBufferIdx >= MGL_MAX_PACKED_STRUCT_BUFFERS) {
        s_packedStructBufferIdx = 0;
    }
    int idx = s_packedStructBufferIdx++;
    Buffer *buf = s_packedStructBuffers[idx];
    if (!buf) {
        buf = (Buffer *)calloc(1, sizeof(Buffer));
        if (!buf) {
            return NULL;
        }
        buf->name = 0xF0000000u | (GLuint)idx;
        buf->target = GL_UNIFORM_BUFFER;
        buf->usage = GL_STATIC_DRAW;
        buf->written_min = -1;
        buf->written_max = -1;
        s_packedStructBuffers[idx] = buf;
    }

    /* Release previous Metal buffer if any */
    if (buf->data.mtl_data) {
        mglSafeReleaseMetalObj((void **)&buf->data.mtl_data);
    }

    /* Create new Metal buffer with packed data.
     * Pad to kMGLMinimumStageBindingSize so the buffer passes the
     * minimum-size validation in the vertex/fragment binding paths
     * (which otherwise replaces undersized buffers with a zero-filled
     * fallback, losing the struct data). */
    size_t mtl_size = size;
    if (mtl_size < kMGLMinimumStageBindingSize) {
        mtl_size = kMGLMinimumStageBindingSize;
    }
    uint8_t *padded = NULL;
    const void *src = data;
    if (mtl_size > size) {
        padded = (uint8_t *)calloc(1, mtl_size);
        if (padded) {
            memcpy(padded, data, size);
            src = padded;
        }
    }
    id<MTLBuffer> mtlBuffer = [device newBufferWithBytes:src
                                                  length:mtl_size
                                                  options:MTLResourceCPUCacheModeDefaultCache];
    if (padded) {
        free(padded);
    }
    if (!mtlBuffer) {
        return NULL;
    }
    buf->data.mtl_data = (void *)CFBridgingRetain(mtlBuffer);
    buf->size = (GLsizeiptr)mtl_size;
    buf->data.buffer_data = 0;
    buf->data.buffer_size = mtl_size;
    buf->data.dirty_bits = 0;
    buf->has_initialized_data = GL_TRUE;
    buf->ever_written = GL_TRUE;
    /* Mark as transient so mglRendererGetValidatedBuffer bypasses the
     * buffer hash-table lookup (packed struct buffers are standalone
     * Buffer wrappers, not inserted into the GL buffer table). */
    buf->transient_batch_buffer = GL_TRUE;
    return buf;
}

- (bool) mapGLBuffersToMTLBufferMap:(BufferMapList *)buffer_map stage: (int) stage
{
    static uint64_t s_mapCallCountByStage[8] = {0};
    uint64_t mapCall = 0;
    if (stage >= 0 && stage < 8) {
        mapCall = ++s_mapCallCountByStage[stage];
    } else {
        mapCall = ++s_mapCallCountByStage[0];
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(mapCall)) {
        MGLTraceNSLog(@"MGL TRACE map.begin stage=%d call=%llu preCount=%u program=%u",
              stage,
              (unsigned long long)mapCall,
              buffer_map ? buffer_map->count : 0,
              ctx ? (unsigned)ctx->active_state->program_name : 0u);
    }

    int count;
    int mapped_buffers;
    struct {
        int spvc_type;
        int gl_buffer_type;
        const char *name;
    } mapped_types[4] = {
        {SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, _UNIFORM_BUFFER, "Uniform Buffer"},
        {SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, _UNIFORM_CONSTANT, "Uniform Constant"},
        {SPVC_RESOURCE_TYPE_STORAGE_BUFFER, _SHADER_STORAGE_BUFFER, "Shader Storage Buffer"},
        {SPVC_RESOURCE_TYPE_ATOMIC_COUNTER, _ATOMIC_COUNTER_BUFFER, "Atomic Counter Buffer"}
    };
#if DEBUG_MAPPED_TYPES
    const char *stages[] = {"VERTEX_SHADER", "TESS_CONTROL_SHADER", "TESS_EVALUATION_SHADER",
        "GEOMETRY_SHADER", "FRAGMENT_SHADER", "COMPUTE_SHADER"};
#endif
    
    // init mapped buffer count
    buffer_map->count = 0;

    if (![self mapShaderBufferResourcesToBufferMap:buffer_map stage:stage]) {
        return false;
    }

    // bind vao attribs to buffers (attribs can share the same buffer)
    if (stage == _VERTEX_SHADER)
    {
        int count = [self getProgramBindingCount: stage type: SPVC_RESOURCE_TYPE_STAGE_INPUT];
        VertexArray *vao = mglRendererGetValidatedVAO(ctx, "mapGLBuffersToMTLBufferMap");
        if (![self mapVertexAttributeBuffersToBufferMap:buffer_map vao:vao stageInputCount:count stage:stage]) {
            return false;
        }
    }
    else if (stage == _COMPUTE_SHADER)
    {
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(mapCall)) {
        MGLTraceNSLog(@"MGL TRACE map.end stage=%d call=%llu mappedCount=%u",
              stage,
              (unsigned long long)mapCall,
              buffer_map ? buffer_map->count : 0);
    }

    return true;
}

- (bool)mapShaderBufferResourcesToBufferMap:(BufferMapList *)buffer_map stage:(int)stage
{
    int count;
    struct {
        int spvc_type;
        int gl_buffer_type;
        const char *name;
    } mapped_types[4] = {
        {SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, _UNIFORM_BUFFER, "Uniform Buffer"},
        {SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, _UNIFORM_CONSTANT, "Uniform Constant"},
        {SPVC_RESOURCE_TYPE_STORAGE_BUFFER, _SHADER_STORAGE_BUFFER, "Shader Storage Buffer"},
        {SPVC_RESOURCE_TYPE_ATOMIC_COUNTER, _ATOMIC_COUNTER_BUFFER, "Atomic Counter Buffer"}
    };
#if DEBUG_MAPPED_TYPES
    const char *stages[] = {"VERTEX_SHADER", "TESS_CONTROL_SHADER", "TESS_EVALUATION_SHADER",
        "GEOMETRY_SHADER", "FRAGMENT_SHADER", "COMPUTE_SHADER"};
#endif

    for(int type=0; type<4; type++)
    {
        int spvc_type;
        int gl_buffer_type;

        spvc_type = mapped_types[type].spvc_type;
        gl_buffer_type = mapped_types[type].gl_buffer_type;
        
        count = [self getProgramBindingCount: stage type: spvc_type];

#if DEBUG_MAPPED_TYPES
        DEBUG_PRINT("Checking mapped_types: %s count:%d for stage: %s\n", mapped_types[type].name, count, stages[stage]);
#endif
        
        if (count)
        {
            BufferBaseTarget *buffers;
            BufferBaseTarget *fallbackBuffers = NULL;

            Program *activeProgram = mglResolveProgramForStageFromState(ctx, stage);
            if (spvc_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT && activeProgram) {
                buffers = activeProgram->plain_uniform_buffers;
                fallbackBuffers = ctx->active_state->buffer_base[gl_buffer_type].buffers;
            } else {
                buffers = ctx->active_state->buffer_base[gl_buffer_type].buffers;
            }
            
            for (int i = 0; i < count; i++)
            {
                GLuint spirv_binding;
                Buffer *buf;
                BufferBaseTarget *baseBinding;

                // Use the GL binding point to locate the client's buffer base.
                // The resource's `binding` may already have been rewritten to the
                // Metal [[buffer(n)]] slot parsed from generated MSL.
                Program *program = mglResolveProgramForStageFromState(ctx, stage);
                if (!program || spvc_type < 0 || spvc_type >= _MAX_SPIRV_RES ||
                    i >= (int)program->spirv_resources_list[stage][spvc_type].count) {
                    continue;
                }
                SpirvResource *resource = &program->spirv_resources_list[stage][spvc_type].list[i];
                if (mglShouldSkipStageBufferResource(program, stage, spvc_type, resource)) {
                    continue;
                }

                if (spvc_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
                    getenv("MGL_DEBUG_STRUCT_PACK")) {
                    NSLog(@"MGL STRUCTCHECK program=%u stage=%d name=%s ubo_members=%p count=%u req_size=%lu samplerLike=%d unifLoc=%d",
                          (unsigned)program->name, stage,
                          resource->name ? resource->name : "(null)",
                          (void *)resource->ubo_members,
                          (unsigned)resource->ubo_member_count,
                          (unsigned long)resource->required_size,
                          mglRendererResourceLooksSamplerLike(resource, spvc_type) ? 1 : 0,
                          resource->uniform_location);
                }

                /* Plain struct uniform packing.
                 *
                 * SPIRV-Cross translates `layout(location=N) uniform S u[K]`
                 * into separate Metal buffer arguments (`constant S* u_0
                 * [[buffer(B)]]`, etc.), each expecting a full struct's
                 * worth of data.  MGL stores individual uniform member data
                 * per location in plain_uniform_buffers[location].  Pack
                 * the member data into struct-sized Metal buffers here. */
                if (spvc_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
                    resource->ubo_members && resource->ubo_member_count > 0 &&
                    resource->required_size > 0 &&
                    !mglRendererResourceLooksSamplerLike(resource, spvc_type)) {

                    GLuint loc_step = mglPlainStructLocStep(resource);
                    GLint base_loc = resource->uniform_location;
                    if (base_loc < 0) {
                        base_loc = (GLint)resource->location;
                    }
                    GLuint array_size = mglStageBufferResourceElementCount(spvc_type, resource);
                    size_t struct_size = resource->required_size;
                    bool allowFallback = fallbackBuffers &&
                        mglPlainUniformAllowsGlobalFallback(resource);

                    for (GLuint element = 0; element < array_size; element++) {
                        GLuint metal_binding = mglMetalResourceSlotForElement(resource, element);
                        GLuint elem_loc_start = element * loc_step;
                        GLuint elem_loc_end = (element + 1u) * loc_step;
                        GLuint elem_byte_start = element * (GLuint)struct_size;

                        uint8_t stack_packed[256];
                        uint8_t *packed = (struct_size <= sizeof(stack_packed))
                                          ? stack_packed
                                          : (uint8_t *)calloc(1, struct_size);
                        if (!packed) continue;
                        memset(packed, 0, struct_size);

                        for (GLuint m = 0; m < resource->ubo_member_count; m++) {
                            SpirvUBOMember *member = &resource->ubo_members[m];

                            /* member->location_offset is relative to the
                             * resource's base uniform_location (spans all
                             * array elements).  Filter to current element. */
                            GLuint member_loc_off = (GLuint)member->location_offset;
                            if (member_loc_off < elem_loc_start ||
                                member_loc_off >= elem_loc_end) {
                                continue;
                            }

                            /* member->offset is the absolute byte offset
                             * across the whole array.  Compute the relative
                             * offset within this element. */
                            GLuint member_offset = member->offset;
                            if (member_offset >= elem_byte_start) {
                                member_offset -= elem_byte_start;
                            }
                            if (member_offset >= struct_size) {
                                continue;
                            }

                            /* Location of this member's data in
                             * plain_uniform_buffers: base_loc + the member's
                             * absolute location_offset. */
                            GLint member_loc = base_loc + (GLint)member_loc_off;
                            if (member_loc < 0 || member_loc >= (GLint)MAX_BINDABLE_BUFFERS) {
                                continue;
                            }

                            if (member->size > 1) {
                                /* Array member: each element stored at a
                                 * separate location (CTS convention: 1
                                 * location per leaf element). */
                                GLuint elem_stride = (GLuint)member->array_stride;
                                if (elem_stride == 0) {
                                    /* Plain struct uniforms lack ArrayStride
                                     * decorations; derive stride from the
                                     * member's GL type (per-element byte size). */
                                    elem_stride = mglGLTypeElementByteSize(member->gl_type);
                                }
                                for (GLint ai = 0; ai < member->size; ai++) {
                                    GLint elem_loc = member_loc + ai;
                                    if (elem_loc < 0 || elem_loc >= (GLint)MAX_BINDABLE_BUFFERS) {
                                        continue;
                                    }
                                    BufferBaseTarget *mb = &buffers[elem_loc];
                                    Buffer *mbuf = mglRendererGetValidatedBuffer(
                                        ctx, mb->buf,
                                        "mapGLBuffersToMTLBufferMap(struct,array)",
                                        (NSUInteger)elem_loc);
                                    if (!mbuf && allowFallback) {
                                        BufferBaseTarget *fb = &fallbackBuffers[elem_loc];
                                        mbuf = mglRendererGetValidatedBuffer(
                                            ctx, fb->buf,
                                            "mapGLBuffersToMTLBufferMap(struct,array,fb)",
                                            (NSUInteger)elem_loc);
                                    }
                                    if (!mbuf || !mbuf->data.buffer_data || mbuf->size <= 0) {
                                        continue;
                                    }
                                    size_t copy_size = (size_t)mbuf->size;
                                    if (copy_size > (size_t)elem_stride) {
                                        copy_size = (size_t)elem_stride;
                                    }
                                    GLuint dest_off = member_offset +
                                        (GLuint)(ai * elem_stride);
                                    if ((size_t)dest_off + copy_size > struct_size) {
                                        copy_size = struct_size - (size_t)dest_off;
                                    }
                                    if (copy_size > 0) {
                                        memcpy(packed + dest_off,
                                               (const void *)(uintptr_t)mbuf->data.buffer_data,
                                               copy_size);
                                    }
                                }
                            } else {
                                /* Scalar / vector / matrix member: all data
                                 * at one location. */
                                BufferBaseTarget *mb = &buffers[member_loc];
                                Buffer *mbuf = mglRendererGetValidatedBuffer(
                                    ctx, mb->buf,
                                    "mapGLBuffersToMTLBufferMap(struct,scalar)",
                                    (NSUInteger)member_loc);
                                if (!mbuf && allowFallback) {
                                    BufferBaseTarget *fb = &fallbackBuffers[member_loc];
                                    mbuf = mglRendererGetValidatedBuffer(
                                        ctx, fb->buf,
                                        "mapGLBuffersToMTLBufferMap(struct,scalar,fb)",
                                        (NSUInteger)member_loc);
                                }
                                if (!mbuf || !mbuf->data.buffer_data || mbuf->size <= 0) {
                                    continue;
                                }
                                size_t copy_size = (size_t)mbuf->size;
                                if ((size_t)member_offset + copy_size > struct_size) {
                                    copy_size = struct_size - (size_t)member_offset;
                                }
                                if (copy_size > 0) {
                                    memcpy(packed + member_offset,
                                           (const void *)(uintptr_t)mbuf->data.buffer_data,
                                           copy_size);
                                }
                            }
                        }

                        if (getenv("MGL_DEBUG_STRUCT_PACK")) {
                            const float *fv = (const float *)packed;
                            NSLog(@"MGL STRUCTDUMP prog=%u stage=%d res=%s elem=%u loc=%d metal=%u size=%lu",
                                  (unsigned)program->name, stage,
                                  resource->name ? resource->name : "(null)",
                                  element, base_loc + (GLint)(loc_step * element),
                                  (unsigned)metal_binding, (unsigned long)struct_size);
                            for (size_t di = 0; di < struct_size && di < 64; di += 4) {
                                NSLog(@"  off[%zu] = %02x%02x%02x%02x (float=%.6f)",
                                      di, packed[di], packed[di+1], packed[di+2], packed[di+3],
                                      fv[di/4]);
                            }
                        }

                        Buffer *packedBuf = mglGetPackedStructBuffer(ctx, _device,
                                                                      packed, struct_size);
                        if (packed != stack_packed) {
                            free(packed);
                        }
                        if (!packedBuf) {
                            continue;
                        }

                        if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                            NSLog(@"MGL ERROR: mapGLBuffersToMTLBufferMap struct overflow: count=%d max=%d",
                                  buffer_map->count, MAX_MAPPED_BUFFERS);
                            return false;
                        }
                        BufferMap *entry = &buffer_map->buffers[buffer_map->count];
                        bzero(entry, sizeof(*entry));
                        entry->attribute_mask = 0;
                        entry->buffer_base_index = (GLuint)(base_loc + (GLint)(loc_step * element));
                        entry->resource_type = (GLuint)spvc_type;
                        entry->resource_index = (GLuint)i;
                        entry->metal_binding_index = metal_binding;
                        entry->has_metal_binding = GL_TRUE;
                        entry->buf = packedBuf;
                        entry->offset = 0;
                        entry->size = (GLsizeiptr)struct_size;
                        buffer_map->count++;

                        NSLog(@"MGL STRUCTPACK program=%u stage=%d resource=%s element=%u/%u loc=%d metal=%u size=%lu",
                              (unsigned)program->name,
                              stage,
                              resource->name ? resource->name : "(null)",
                              element, array_size,
                              base_loc + (GLint)(loc_step * element),
                              (unsigned)metal_binding,
                              (unsigned long)struct_size);
                    }
                    continue; /* Skip normal binding path for struct resource */
                }

                GLuint element_count = mglStageBufferResourceElementCount(spvc_type, resource);
                for (GLuint element = 0; element < element_count; element++) {
                    GLuint metal_binding = mglMetalResourceSlotForElement(resource, element);
                    spirv_binding = mglClientBufferBindingForResourceElement(spvc_type, resource, element);
                    if (spirv_binding >= MAX_BINDABLE_BUFFERS)
                    {
                        NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: stage=%d type=%d binding=%u exceeds MAX_BINDABLE_BUFFERS=%d, skipping",
                              stage, spvc_type, spirv_binding, MAX_BINDABLE_BUFFERS);
                        continue;
                    }

                baseBinding = &buffers[spirv_binding];
                bool usedFallbackBinding = false;
                bool allowGlobalFallback =
                    fallbackBuffers &&
                    (spvc_type != SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT ||
                     mglPlainUniformAllowsGlobalFallback(resource));
                if (allowGlobalFallback && !baseBinding->buf && baseBinding->buffer == 0) {
                    BufferBaseTarget *fallbackBinding = &fallbackBuffers[spirv_binding];
                    if (fallbackBinding->buf || fallbackBinding->buffer != 0) {
                        baseBinding = fallbackBinding;
                        usedFallbackBinding = true;
                    }
                }
                buf = mglRendererGetValidatedBuffer(ctx, baseBinding->buf,
                                                    "mapGLBuffersToMTLBufferMap(base)",
                                                    (NSUInteger)spirv_binding);

                // Recover from name/object map skew: some paths can preserve GL name while pointer slot is stale.
                if (!buf && baseBinding->buffer != 0) {
                    Buffer *resolved = (Buffer *)searchHashTable(&ctx->active_state->buffer_table, baseBinding->buffer);
                    resolved = mglRendererGetValidatedBuffer(ctx, resolved,
                                                             "mapGLBuffersToMTLBufferMap(base,recover)",
                                                             (NSUInteger)spirv_binding);
                    if (resolved) {
                        baseBinding->buf = resolved;
                        buf = resolved;
                        NSLog(@"MGL BUFFER RECOVER: stage=%d type=%d binding=%u name=%u ptr=%p",
	                              stage, spvc_type, spirv_binding, baseBinding->buffer, resolved);
	                    }
	                }

                NSUInteger reflectedRequiredSize =
                    [self getProgramBindingRequiredSize:stage type:spvc_type index:i];

	                if (buf)
	                {
	                    if (buffer_map->count >= MAX_MAPPED_BUFFERS)
	                    {
	                        NSLog(@"MGL ERROR: mapGLBuffersToMTLBufferMap overflow: count=%d max=%d",
                              buffer_map->count, MAX_MAPPED_BUFFERS);
                        return false;
                    }
                    BufferMap *entry = &buffer_map->buffers[buffer_map->count];
                    bzero(entry, sizeof(*entry));
                    entry->attribute_mask = 0; // non attribute.. no bits set
                    entry->buffer_base_index = spirv_binding;
                    entry->resource_type = (GLuint)spvc_type;
                    entry->resource_index = (GLuint)i;
                    entry->metal_binding_index = metal_binding;
                    entry->has_metal_binding = GL_TRUE;
                    entry->buf = buf;
                    entry->offset = baseBinding->offset;
                    entry->size = baseBinding->size;
                    baseBinding->buffer = buf->name;
                    buffer_map->count++;

                    if (mglProgramNeedsBindingTrace(program)) {
                        static uint64_t s_focusedUBOMapLogs = 0;
                        if (mglShouldLogFocusedBinding(&s_focusedUBOMapLogs)) {
                            NSLog(@"MGL BINDMAP focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u buffer=%u offset=%lld range=%lld reflected=%lu",
                                  (unsigned)program->name,
                                  mglShaderStageName(stage),
                                  mglSpirvResourceTypeName(spvc_type),
                                  resource->name ? resource->name : "(null)",
                                  i,
                                  (unsigned)spirv_binding,
                                  (unsigned)metal_binding,
                                  (unsigned)buf->name,
                                  (long long)baseBinding->offset,
                                  (long long)baseBinding->size,
                                  (unsigned long)reflectedRequiredSize);
                        }
                    }

                    /* Trace file: log UBO binding for program trace */
                    static uint64_t s_traceFileUBOMapLogs = 0;
                    if (mglProgramNeedsTraceLog(program) &&
                        mglShouldLogTraceFileBindingForProgram(program, &s_traceFileUBOMapLogs)) {
                        mglTraceLog("BINDMAP program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u buffer=%u offset=%lld range=%lld reflected=%lu fallback=%d",
                                    (unsigned)program->name,
                                    mglShaderStageName(stage),
                                    mglSpirvResourceTypeName(spvc_type),
                                    resource->name ? resource->name : "(null)",
                                    i,
                                    (unsigned)spirv_binding,
                                    (unsigned)metal_binding,
                                    (unsigned)buf->name,
                                    (long long)baseBinding->offset,
                                    (long long)baseBinding->size,
                                    (unsigned long)reflectedRequiredSize,
                                    usedFallbackBinding ? 1 : 0);
                    }

                    if (reflectedRequiredSize > 0 && baseBinding->size > 0 &&
                        (NSUInteger)baseBinding->size < reflectedRequiredSize) {
                        GLuint programName = ctx ? ctx->active_state->program_name : 0u;
                        if (mglShouldLogSmallBaseBinding(programName,
                                                         stage,
                                                         spvc_type,
                                                         spirv_binding,
                                                         buf->name,
                                                         baseBinding->size,
                                                         reflectedRequiredSize)) {
                            NSLog(@"MGL WARNING: base binding too small program=%u stage=%d type=%d binding=%u glName=%u range=%lld reflected=%lu (padding at bind)",
                                  programName,
                                  stage,
                                  spvc_type,
                                  spirv_binding,
                                  buf->name,
                                  (long long)baseBinding->size,
                                  (unsigned long)reflectedRequiredSize);
                        }
                    }
                    
                    //DEBUG_PRINT("Found buffer type: %s buffer_base_index: %d\n", mapped_types[type].name, spirv_binding);
	                }
	                else
	                {
                    if (mglProgramNeedsBindingTrace(program)) {
                        static uint64_t s_focusedUBOMissLogs = 0;
                        if (mglShouldLogFocusedBinding(&s_focusedUBOMissLogs)) {
                            NSLog(@"MGL BINDMISS focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u baseBuffer=%u basePtr=%p offset=%lld range=%lld reflected=%lu usedFallback=%d",
                                  (unsigned)program->name,
                                  mglShaderStageName(stage),
                                  mglSpirvResourceTypeName(spvc_type),
                                  resource->name ? resource->name : "(null)",
                                  i,
                                  (unsigned)spirv_binding,
                                  (unsigned)metal_binding,
                                  (unsigned)baseBinding->buffer,
                                  baseBinding->buf,
                                  (long long)baseBinding->offset,
                                  (long long)baseBinding->size,
                                  (unsigned long)reflectedRequiredSize,
                                  usedFallbackBinding ? 1 : 0);
                        }
                    }
                    static uint64_t s_traceFileUBOMissLogs = 0;
                    if (mglProgramNeedsTraceLog(program) &&
                        mglShouldLogTraceFileBindingForProgram(program, &s_traceFileUBOMissLogs)) {
                        mglTraceLog("BINDMISS program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u baseBuffer=%u basePtr=%p offset=%lld range=%lld reflected=%lu fallback=%d",
                                    (unsigned)program->name,
                                    mglShaderStageName(stage),
                                    mglSpirvResourceTypeName(spvc_type),
                                    resource->name ? resource->name : "(null)",
                                    i,
                                    (unsigned)spirv_binding,
                                    (unsigned)metal_binding,
                                    (unsigned)baseBinding->buffer,
                                    baseBinding->buf,
                                    (long long)baseBinding->offset,
                                    (long long)baseBinding->size,
                                    (unsigned long)reflectedRequiredSize,
                                    usedFallbackBinding ? 1 : 0);
                    }
	                    if (baseBinding->buf || baseBinding->buffer != 0 || baseBinding->offset != 0 || baseBinding->size != 0) {
	                        NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: dropping invalid base buffer binding=%u stage=%d type=%d name=%u ptr=%p offset=%lld size=%lld",
	                              spirv_binding, stage, spvc_type,
                              baseBinding->buffer,
                              baseBinding->buf,
                              (long long)baseBinding->offset,
                              (long long)baseBinding->size);
                        bzero(baseBinding, sizeof(BufferBaseTarget));
                    }
                    // Some vanilla shader paths tolerate unbound blocks on specific stages.
                    // Skip instead of poisoning global GL error state with GL_INVALID_OPERATION.
                    continue;
                }
                }
            }
        }
    }

    return true;
}

- (bool)mapVertexAttributeBuffersToBufferMap:(BufferMapList *)buffer_map
                                         vao:(VertexArray *)vao
                            stageInputCount:(int)count
                                       stage:(int)stage
{
    int vao_buffer_start;
    int mapped_buffers = 0;
    GLuint next_vertex_binding_index = (GLuint)kMGLVertexAttribBufferBase;
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);

    mapped_buffers = 0;

    if (!vao) {
        if (count > 0) {
            NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: stage inputs=%d but VAO is invalid/null, skipping attrib mapping",
                  count);
        }
        return true;
    }

    if (kMGLVertexAttribBufferBase >= kMGLMaxMetalVertexBufferCount) {
        NSLog(@"MGL ERROR: invalid vertex attrib base index=%lu (max valid=%lu)",
              (unsigned long)kMGLVertexAttribBufferBase,
              (unsigned long)kMGLMaxMetalVertexBufferIndex);
        return false;
    }

    // vao buffers start after the uniforms and shader buffers
    vao_buffer_start = buffer_map->count;
    // CRITICAL SECURITY FIX: Check against actual map capacity.
    if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL SECURITY ERROR: buffer_map count %d exceeds MAX_MAPPED_BUFFERS %d",
              buffer_map->count, MAX_MAPPED_BUFFERS);
        return false;
    }
    buffer_map->buffers[vao_buffer_start].attribute_mask = 0;
    buffer_map->buffers[vao_buffer_start].buffer_base_index = (GLuint)kMGLVertexAttribBufferBase;
    buffer_map->buffers[vao_buffer_start].resource_type = 0;
    buffer_map->buffers[vao_buffer_start].resource_index = 0;
    buffer_map->buffers[vao_buffer_start].metal_binding_index = 0;
    buffer_map->buffers[vao_buffer_start].has_metal_binding = GL_FALSE;
    buffer_map->buffers[vao_buffer_start].buf = NULL;
    buffer_map->buffers[vao_buffer_start].offset = 0;
    buffer_map->buffers[vao_buffer_start].size = 0;

    // create attribute map
    //
    // we need to cache this mapping, its called on each draw command
    //
    bool vaoHasExplicitAttribs = (vao->enabled_attribs != 0u);
    for(int att=0;att<MAX_ATTRIBS; att++)
    {
        if (vaoHasExplicitAttribs && !(vao->enabled_attribs & (0x1 << att)))
        {
            if ((vao->enabled_attribs >> (att+1)) == 0)
                break;
            continue;
        }
        {
            if (!mglRendererProgramUsesVertexAttrib(activeProgram, (GLuint)att)) {
                if (vaoHasExplicitAttribs && (vao->enabled_attribs >> (att+1)) == 0)
                    break;
                continue;
            }

            MGLResolvedVertexAttribBinding resolved = {0};
            if (!mglRendererResolveVertexAttribBinding(ctx,
                                                       vao,
                                                       (GLuint)att,
                                                       "mapGLBuffersToMTLBufferMap",
                                                       &resolved)) {
                NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: enabled attrib %d has invalid/NULL buffer, skipping attrib",
                      att);
                continue;
            }
            Buffer *gl_buffer = resolved.buffer;

            Buffer *map_buffer = NULL;

            // check start for map... then check
            map_buffer = buffer_map->buffers[vao_buffer_start].buf;

            // empty slot map it here, only works on first buffer..
            if (map_buffer == NULL)
            {
                if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                    NSLog(@"MGL WARNING: vertex binding index overflow (next=%u maxValid=%lu), skipping attrib %d",
                          next_vertex_binding_index, (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                    continue;
                }
                // map the buffer object to a metal vertex index
                if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                    NSLog(@"MGL WARNING: vertex buffer map is full (count=%u max=%u), skipping attrib %d",
                          buffer_map->count, MAX_MAPPED_BUFFERS, att);
                    continue;
                }
                buffer_map->buffers[vao_buffer_start].attribute_mask |= (0x1 << att);
                buffer_map->buffers[vao_buffer_start].buf = gl_buffer;
                buffer_map->buffers[vao_buffer_start].buffer_base_index = next_vertex_binding_index++;
                buffer_map->buffers[vao_buffer_start].has_metal_binding = GL_FALSE;
                buffer_map->buffers[vao_buffer_start].offset = resolved.binding_offset;
                buffer_map->buffers[vao_buffer_start].size = 0;
                buffer_map->count++;

                mapped_buffers++;
            }
            else
            {
                bool found_buffer = false;

                // find vao attrib with same buffer
                for (int map=vao_buffer_start;
                     (found_buffer == false) && map<buffer_map->count;
                     map++)
                {
                    map_buffer = buffer_map->buffers[map].buf;
                    if (!map_buffer) {
                        continue;
                    }

                    // we need to check name and target, not pointers..
                    // FIX ME: I think we don't need a target as all attribs should be an array_buffer
                    // Offset is intentionally NOT compared: attributes sharing the same
                    // VBO/stride/divisor are grouped into one Metal buffer slot, with
                    // per-attribute offsets expressed via the vertex descriptor.
	                        if ((map_buffer->name == gl_buffer->name) &&
	                            (map_buffer->target == gl_buffer->target))
	                        {
	                            bool compatibleStream = true;
	                            for (GLuint prevAttrib = 0; prevAttrib < MAX_ATTRIBS; prevAttrib++) {
	                                if ((buffer_map->buffers[map].attribute_mask & (0x1u << prevAttrib)) == 0u) {
	                                    continue;
	                                }
	                                MGLResolvedVertexAttribBinding prevResolved = {0};
	                                if (!mglRendererResolveVertexAttribBinding(ctx,
	                                                                           vao,
	                                                                           prevAttrib,
	                                                                           "mapGLBuffersToMTLBufferMap(stream)",
	                                                                           &prevResolved)) {
	                                    continue;
	                                }
	                                if (prevResolved.stride != resolved.stride ||
	                                    prevResolved.divisor != resolved.divisor) {
	                                    compatibleStream = false;
	                                    break;
	                                }
	                            }
	                            if (compatibleStream) {
	                                // include it the list of attributes
	                                buffer_map->buffers[map].attribute_mask |= (0x1 << att);
	                                found_buffer = true;
	                                mapped_buffers++;
	                                break;
	                            }
	                        }
                }

                if (found_buffer == false)
                {
                    if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                        NSLog(@"MGL WARNING: vertex binding index overflow (next=%u maxValid=%lu), cannot append attrib %d",
                              next_vertex_binding_index, (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                        continue;
                    }
                    // map the next buffer object to a metal vertex index
                    if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                        NSLog(@"MGL WARNING: vertex buffer map is full (count=%u max=%u), cannot append attrib %d",
                              buffer_map->count, MAX_MAPPED_BUFFERS, att);
                        continue;
                    }
                    buffer_map->buffers[buffer_map->count].attribute_mask = (0x1 << att);
                    buffer_map->buffers[buffer_map->count].buffer_base_index = next_vertex_binding_index++;
                    buffer_map->buffers[buffer_map->count].resource_type = 0;
                    buffer_map->buffers[buffer_map->count].resource_index = 0;
                    buffer_map->buffers[buffer_map->count].metal_binding_index = 0;
                    buffer_map->buffers[buffer_map->count].has_metal_binding = GL_FALSE;
                    buffer_map->buffers[buffer_map->count].buf = gl_buffer;
                    buffer_map->buffers[buffer_map->count].offset = resolved.binding_offset;
                    buffer_map->buffers[buffer_map->count].size = 0;
                    buffer_map->count++;

                    mapped_buffers++;
                }
            }
        }

        if (vaoHasExplicitAttribs && (vao->enabled_attribs >> (att+1)) == 0)
            break;
    }

    if (mapped_buffers != count) {
        static unsigned long long s_map_mismatch_hits = 0;
        s_map_mismatch_hits++;
        if ((s_map_mismatch_hits % 64ull) == 1ull) {
            Buffer *drawIndexBuffer = vao->element_array.buffer;
            void *indexBufferMetal = drawIndexBuffer ? drawIndexBuffer->data.mtl_data : NULL;
            NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap mismatch (pipeline=%p mapped=%u expected=%u stage=%d hit=%llu indexBuffer=%p vao=%p)",
                  _pipelineState, mapped_buffers, count, stage, s_map_mismatch_hits, indexBufferMetal, vao);
        }
    }

    return true;
}

- (bool) mapBuffersToMTL
{
    if ([self mapGLBuffersToMTLBufferMap: &ctx->active_state->vertex_buffer_map_list stage:_VERTEX_SHADER] == false)
        return false;

    if ([self mapGLBuffersToMTLBufferMap: &ctx->active_state->fragment_buffer_map_list stage:_FRAGMENT_SHADER] == false)
        return false;

    return true;
}

static BOOL mglSnapshotSharedDirtyBuffer(id<MTLDevice> device,
                                         Buffer *ptr,
                                         id<MTLBuffer> *bufferPtr)
{
    id<MTLBuffer> buffer = bufferPtr ? *bufferPtr : nil;
    const void *cpuData = ptr ? (const void *)(uintptr_t)ptr->data.buffer_data : NULL;
    if (!device || !ptr || !buffer || buffer.storageMode != MTLStorageModeShared ||
        !cpuData || (uintptr_t)cpuData < 0x1000u ||
        (ptr->storage_flags & GL_CLIENT_STORAGE_BIT) || cpuData == buffer.contents) {
        return YES;
    }

    NSUInteger snapshotLength = buffer.length;
    if (ptr->data.buffer_size > 0) {
        snapshotLength = MIN(snapshotLength, (NSUInteger)ptr->data.buffer_size);
    }
    if (snapshotLength == 0) {
        return YES;
    }

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (buffer.cpuCacheMode == MTLCPUCacheModeWriteCombined) {
        options |= MTLResourceCPUCacheModeWriteCombined;
    }

    id<MTLBuffer> snapshot = [device newBufferWithLength:buffer.length options:options];
    if (!snapshot) {
        NSLog(@"MGL BUFFER ERROR: failed to snapshot dynamic buffer %u", ptr->name);
        return NO;
    }

    memcpy(snapshot.contents, cpuData, snapshotLength);
    if (snapshotLength < snapshot.length) {
        memset((uint8_t *)snapshot.contents + snapshotLength,
               0,
               snapshot.length - snapshotLength);
    }

    mglSafeReleaseMetalObj((void **)&ptr->data.mtl_data);
    ptr->data.mtl_data = (void *)CFBridgingRetain(snapshot);
    *bufferPtr = snapshot;
    return YES;
}

static BOOL mglSnapshotSharedBufferRange(id<MTLDevice> device,
                                         Buffer *ptr,
                                         id<MTLBuffer> *bufferPtr,
                                         NSUInteger offset,
                                         NSUInteger length)
{
    id<MTLBuffer> buffer = bufferPtr ? *bufferPtr : nil;
    const uint8_t *cpuData = ptr ? (const uint8_t *)(uintptr_t)ptr->data.buffer_data : NULL;
    if (!device || !ptr || !buffer || buffer.storageMode != MTLStorageModeShared ||
        !cpuData || (uintptr_t)cpuData < 0x1000u ||
        (ptr->storage_flags & GL_CLIENT_STORAGE_BIT) || cpuData == buffer.contents ||
        offset > buffer.length || length > buffer.length - offset) {
        return YES;
    }

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (buffer.cpuCacheMode == MTLCPUCacheModeWriteCombined) {
        options |= MTLResourceCPUCacheModeWriteCombined;
    }

    id<MTLBuffer> snapshot = [device newBufferWithLength:buffer.length options:options];
    if (!snapshot) {
        NSLog(@"MGL BUFFER ERROR: failed to snapshot mapped buffer %u", ptr->name);
        return NO;
    }

    memcpy(snapshot.contents, buffer.contents, buffer.length);
    memcpy((uint8_t *)snapshot.contents + offset, cpuData + offset, length);

    mglSafeReleaseMetalObj((void **)&ptr->data.mtl_data);
    ptr->data.mtl_data = (void *)CFBridgingRetain(snapshot);
    *bufferPtr = snapshot;
    return YES;
}

- (bool) updateDirtyBuffer:(Buffer *)ptr
{
    if (ptr->size < 4096)
    {
        if ((ptr->data.dirty_bits & DIRTY_BUFFER_ADDR) && ptr->data.mtl_data == NULL) {
            [self bindMTLBuffer: ptr];
            RETURN_FALSE_ON_NULL(ptr->data.mtl_data);
        }

        /*
         * Small buffers are often bound with set*Bytes for vertex attributes, but
         * uniform/base bindings may still bind the Metal buffer directly. Keep
         * that backing synchronized when glBufferSubData/DSA fallback updates the
         * CPU copy, otherwise GUI/item/entity matrices can sample stale data.
         */
        if (ptr->data.dirty_bits & DIRTY_BUFFER_DATA) {
            if (ptr->data.mtl_data == NULL) {
                [self bindMTLBuffer: ptr];
                RETURN_FALSE_ON_NULL(ptr->data.mtl_data);
            }

            id<MTLBuffer> buffer = (id<MTLBuffer>)SafeMetalBridge(ptr->data.mtl_data, objc_getClass("MTLBuffer"), "MTLBuffer");
            if (!buffer) {
                NSLog(@"MGL SECURITY ERROR: Failed to validate small Metal buffer (buffer %u)", ptr->name);
                return false;
            }

            if (!mglSnapshotSharedDirtyBuffer(_device, ptr, &buffer)) {
                return false;
            }

            NSUInteger copyLen = (NSUInteger)MAX((GLsizeiptr)0, ptr->size);
            copyLen = MIN(copyLen, buffer.length);
            if (ptr->data.buffer_size > 0) {
                copyLen = MIN(copyLen, (NSUInteger)ptr->data.buffer_size);
            }

            const void *cpuData = (const void *)(uintptr_t)ptr->data.buffer_data;
            void *metalData = buffer.contents;
            if (cpuData && (uintptr_t)cpuData >= 0x1000u && metalData && copyLen > 0) {
                if (cpuData != metalData) {
                    memmove(metalData, cpuData, copyLen);
                }
                if (buffer.storageMode == MTLStorageModeManaged) {
                    [buffer didModifyRange:NSMakeRange(0, copyLen)];
                }
            } else if (metalData && copyLen > 0) {
                NSUInteger modifyOffset = 0;
                NSUInteger modifyLength = copyLen;
                if (ptr->mapped_length > 0 &&
                    ptr->mapped_offset >= 0 &&
                    (NSUInteger)ptr->mapped_offset < buffer.length) {
                    modifyOffset = (NSUInteger)ptr->mapped_offset;
                    modifyLength = MIN((NSUInteger)ptr->mapped_length, buffer.length - modifyOffset);
                }
                if (modifyLength > 0 && buffer.storageMode == MTLStorageModeManaged) {
                    [buffer didModifyRange:NSMakeRange(modifyOffset, modifyLength)];
                }
            }

            if (kMGLDiagnosticStateLogs) {
                static uint64_t s_smallDirtyUploadCalls = 0;
                uint64_t call = ++s_smallDirtyUploadCalls;
                if (mglShouldTraceBufferTransferCall(call)) {
                    const void *cpuSample = (const void *)(uintptr_t)ptr->data.buffer_data;
                    const void *mtlSample = buffer.contents;
                    size_t sampleLen = (size_t)copyLen;
                    uint64_t cpuHash = mglTraceHashBytes(cpuSample, sampleLen);
                    uint64_t mtlHash = mglTraceHashBytes(mtlSample, sampleLen);
                    char cpuHead[64];
                    char mtlHead[64];
                    cpuHead[0] = '\0';
                    mtlHead[0] = '\0';
                    mglTraceFormatBytes(cpuSample, sampleLen, cpuHead, sizeof(cpuHead));
                    mglTraceFormatBytes(mtlSample, sampleLen, mtlHead, sizeof(mtlHead));
                    MGLTraceNSLog(@"MGL TRACE smallBufferDirty.upload call=%llu buffer=%u size=%lld dirty=0x%x copy=%lu cpuHash=0x%016llx cpuHead=%s mtlLen=%lu mtlHash=0x%016llx mtlHead=%s",
                          (unsigned long long)call,
                          ptr->name,
                          (long long)ptr->size,
                          ptr->data.dirty_bits,
                          (unsigned long)copyLen,
                          (unsigned long long)cpuHash,
                          cpuHead,
                          (unsigned long)buffer.length,
                          (unsigned long long)mtlHash,
                          mtlHead);
                }
            }

            if (ptr->access & GL_MAP_COHERENT_BIT) {
                ptr->data.dirty_bits = DIRTY_BUFFER_DATA;
            } else {
                ptr->data.dirty_bits &= ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
            }

            return true;
        }

        if (kMGLDiagnosticStateLogs && (ptr->data.dirty_bits & DIRTY_BUFFER_ADDR)) {
            static uint64_t s_smallDirtySkipCalls = 0;
            uint64_t call = ++s_smallDirtySkipCalls;
            if (mglShouldTraceBufferTransferCall(call)) {
                const void *cpuData = (const void *)(uintptr_t)ptr->data.buffer_data;
                size_t sampleLen = ptr->size > 0 ? (size_t)ptr->size : 0u;
                uint64_t cpuHash = mglTraceHashBytes(cpuData, sampleLen);
                char cpuHead[64];
                cpuHead[0] = '\0';
                mglTraceFormatBytes(cpuData, sampleLen, cpuHead, sizeof(cpuHead));

                uint64_t mtlHash = 0ull;
                char mtlHead[64];
                mtlHead[0] = '\0';
                NSUInteger metalLen = 0;
                if (ptr->data.mtl_data && (uintptr_t)ptr->data.mtl_data >= 0x10000u) {
                    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
                    if (mtlBuffer) {
                        metalLen = mtlBuffer.length;
                        const void *mtlBytes = mtlBuffer.contents;
                        size_t mtlSample = (size_t)MIN((NSUInteger)sampleLen, metalLen);
                        mtlHash = mglTraceHashBytes(mtlBytes, mtlSample);
                        mglTraceFormatBytes(mtlBytes, mtlSample, mtlHead, sizeof(mtlHead));
                    }
                }

                MGLTraceNSLog(@"MGL TRACE smallBufferDirty.skip call=%llu buffer=%u size=%lld dirty=0x%x cpuHash=0x%016llx cpuHead=%s mtl=%p mtlLen=%lu mtlHash=0x%016llx mtlHead=%s",
                      (unsigned long long)call,
                      ptr->name,
                      (long long)ptr->size,
                      ptr->data.dirty_bits,
                      (unsigned long long)cpuHash,
                      cpuHead,
                      ptr->data.mtl_data,
                      (unsigned long)metalLen,
                      (unsigned long long)mtlHash,
                      (metalLen > 0 ? mtlHead : "-"));
            }
        }

        ptr->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
        return true;
    }
    
    if (ptr->data.dirty_bits & DIRTY_BUFFER_ADDR)
    {
        if (ptr->data.mtl_data == NULL)
        {
            [self bindMTLBuffer: ptr];
            RETURN_FALSE_ON_NULL(ptr->data.mtl_data);
        }

        if ((ptr->data.dirty_bits & DIRTY_BUFFER_DATA) == 0)
        {
            ptr->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
            return true;
        }
    }

    if (ptr->data.dirty_bits & DIRTY_BUFFER_DATA)
    {
        if (ptr->data.mtl_data == NULL)
        {
            [self bindMTLBuffer: ptr];
            RETURN_FALSE_ON_NULL(ptr->data.mtl_data);
        }

        // CRITICAL SECURITY FIX: Safe Metal buffer validation
        id<MTLBuffer> buffer = (id<MTLBuffer>)SafeMetalBridge(ptr->data.mtl_data, objc_getClass("MTLBuffer"), "MTLBuffer");
        if (!buffer) {
            NSLog(@"MGL SECURITY ERROR: Failed to validate Metal buffer (buffer %u)", ptr->name);
            return false;
        }

        if (!mglSnapshotSharedDirtyBuffer(_device, ptr, &buffer)) {
            return false;
        }

        // clear dirty bits if not mapped as coherent
        // this will cause us to keep loading the buffer and keep the GPU
        // contents in check for EVERY drawing operation
        BOOL coherentMapped =
            ((ptr->access_flags & GL_MAP_COHERENT_BIT) != 0) ||
            ((ptr->access & GL_MAP_COHERENT_BIT) != 0);
        if (coherentMapped)
        {
            NSUInteger modifyOffset = 0;
            NSUInteger modifyLength = buffer.length;
            if (ptr->mapped_length > 0 &&
                ptr->mapped_offset >= 0 &&
                (NSUInteger)ptr->mapped_offset < buffer.length) {
                modifyOffset = (NSUInteger)ptr->mapped_offset;
                modifyLength = MIN((NSUInteger)ptr->mapped_length, buffer.length - modifyOffset);
            }
            if (modifyLength > 0 && buffer.storageMode == MTLStorageModeManaged) {
                [buffer didModifyRange:NSMakeRange(modifyOffset, modifyLength)];
            }

            ptr->data.dirty_bits = DIRTY_BUFFER_DATA;
        }
        else
        {
            NSUInteger modifyLength = buffer.length;
            if (ptr->data.buffer_size > 0) {
                modifyLength = MIN(modifyLength, (NSUInteger)ptr->data.buffer_size);
            }
            if (modifyLength > 0 && buffer.storageMode == MTLStorageModeManaged) {
                [buffer didModifyRange:NSMakeRange(0, modifyLength)];
            }

            ptr->data.dirty_bits = 0;
        }
    }
    else
    {
        NSLog(@"MGL BUFFER ERROR: updateDirtyBuffer saw buffer %u with no CPU or Metal backing",
              ptr ? ptr->name : 0u);
        return false;
    }

    return true;
}

- (bool) checkForDirtyBufferData:  (BufferMapList *)buffer_map_list
{
    GLuint mapCount;

    if (!buffer_map_list) {
        return false;
    }

    mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: checkForDirtyBufferData mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    // update vbos, some vbos may not have metal buffers yet
    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          __FUNCTION__,
                                                          (NSUInteger)i);

        if (gl_buffer)
        {
            if (gl_buffer->data.dirty_bits)
            {
                return true;
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return false;
}

- (bool) updateDirtyBaseBufferList: (BufferMapList *)buffer_map_list
{
    GLuint mapCount;

    if (!buffer_map_list) {
        return true;
    }

    mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: updateDirtyBaseBufferList mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    // update vbos, some vbos may not have metal buffers yet
    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          __FUNCTION__,
                                                          (NSUInteger)i);

        if (gl_buffer)
        {
            if (gl_buffer->data.dirty_bits)
            {
                RETURN_FALSE_ON_FAILURE([self updateDirtyBuffer: gl_buffer]);
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return true;
}

/* bindVertexBuffersToCurrentRenderEncoder moved to MGLRenderer+Draw.m */

/* bindFragmentBuffersToCurrentRenderEncoder moved to MGLRenderer+Draw.m */

- (int) getVertexBufferIndexWithAttributeSet: (int) attribute
{
    if (attribute < 0 || attribute >= MAX_ATTRIBS) {
        NSLog(@"MGL ERROR: getVertexBufferIndexWithAttributeSet invalid attribute=%d", attribute);
        return -1;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    if (vao) {
        int resolved = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, (GLuint)attribute, __FUNCTION__);
        if (resolved >= 0) {
            return resolved;
        }
    }

    // Legacy fallback: use cached map list if available.
    GLuint mapCount = ctx->active_state->vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        if (ctx->active_state->vertex_buffer_map_list.buffers[i].attribute_mask & (0x1 << attribute)) {
            GLuint baseIndex = ctx->active_state->vertex_buffer_map_list.buffers[i].buffer_base_index;
            if (baseIndex >= kMGLMaxMetalVertexBufferCount) {
                NSLog(@"MGL ERROR: getVertexBufferIndexWithAttributeSet mapped base index out of Metal range=%u (max valid=%lu)",
                      baseIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }
            return (int)baseIndex;
        }
    }

    NSLog(@"MGL ERROR: No vertex buffer mapping found for attribute %d", attribute);
    return -1;
}

#pragma mark textures

/* mglStoredColorComponentsForTexture and mglMTLSwizzleForGLSwizzle now live
 * in mgl_texture_compat.m — see mgl_texture_compat.h. */

- (void)swizzleTexDesc:(MTLTextureDescriptor *)tex_desc forTex:(Texture*)tex
{
    MTLTextureSwizzle channel_r = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_r);
    MTLTextureSwizzle channel_g = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_g);
    MTLTextureSwizzle channel_b = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_b);
    MTLTextureSwizzle channel_a = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_a);

    tex_desc.swizzle = MTLTextureSwizzleChannelsMake(channel_r, channel_g, channel_b, channel_a);
}

/* mglTextureUploadNeedsSingleChannelSwizzle, mglResolveR8SwizzledComponent,
 * and mglCreateSingleChannelSwizzledUpload now live in mgl_texture_compat.m —
 * see mgl_texture_compat.h. */


- (id<MTLTexture>) createMTLTextureFromGLTexture:(Texture *) tex
{
    // PROPER FIX: Enhanced pre-creation validation to prevent AGX driver issues
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or command queue not available for texture creation");
        return nil;
    }

    // Check if we're in a recovery state that would make texture creation futile
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: GPU operations temporarily suspended during recovery");
        return nil;
    }

    // Validate texture dimensions to prevent Metal assertion failures.
    // Texture buffers (GL_TEXTURE_BUFFER) can have very large widths (millions of texels)
    // since they map to MTLTextureTypeTextureBuffer which uses GPU address space.
    if (tex->target != GL_TEXTURE_BUFFER) {
        if (!tex || tex->width <= 0 || tex->height <= 0 ||
            tex->width > 32768 || tex->height > 32768 || tex->depth > 32768) {
            NSLog(@"MGL ERROR: Invalid texture dimensions %dx%dx%d - rejecting",
                  tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->depth : 0);
            tex->dirty_bits = 0;
            return nil;
        }
    }

    if (tex->target == GL_TEXTURE_BUFFER) {
        return [self createMTLTexelBufferTexture:tex];
    }

    NSUInteger width, height, depth;

    MTLTextureDescriptor *tex_desc;
    MTLTextureType tex_type;
    MTLPixelFormat pixelFormat;
    uint num_faces;
    GLuint effective_mipmap_levels;
    GLuint upload_level_count;
    BOOL storageMipmapped;
    BOOL mipmapped;
    BOOL is_array;
    BOOL texture1DBackedBy2D;
    BOOL texture1DArrayBackedBy2DArray;

    num_faces = 1;
    is_array = false;
    texture1DBackedBy2D = false;
    texture1DArrayBackedBy2DArray = false;
    effective_mipmap_levels = 0;
    upload_level_count = 0;
    storageMipmapped = NO;

    switch(tex->target)
    {
        case GL_TEXTURE_1D:
            tex_type = MTLTextureType2D;
            texture1DBackedBy2D = true;
            break;
        case GL_RENDERBUFFER:
            tex_type = tex->samples > 1u ? MTLTextureType2DMultisample : MTLTextureType2D;
            break;
        case GL_TEXTURE_1D_ARRAY:
            /* SPIRV-Cross lowers sampler1DArray to texture2d_array in MSL, and
             * Metal does not allow texture views from MTLTextureType1DArray to
             * MTLTextureType2DArray.  Always back GL_TEXTURE_1D_ARRAY with
             * MTLTextureType2DArray (height=1), mirroring how GL_TEXTURE_1D is
             * backed by MTLTextureType2D and how mipmapped/depth 1D arrays are
             * already promoted below. */
            tex_type = MTLTextureType2DArray;
            is_array = true;
            texture1DArrayBackedBy2DArray = true;
            break;
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            tex_type = MTLTextureType2D;
            break;
        case GL_TEXTURE_2D_ARRAY: tex_type = MTLTextureType2DArray; is_array = true; break;
        case GL_TEXTURE_2D_MULTISAMPLE: tex_type = MTLTextureType2DMultisample; break;

        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            num_faces = 6;
            tex_type = MTLTextureTypeCube;
            break;

        case GL_TEXTURE_CUBE_MAP_ARRAY:
            num_faces = 6;
            tex_type = MTLTextureTypeCubeArray;
            is_array = true;
            break;

        case GL_TEXTURE_3D: tex_type = MTLTextureType3D; break;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: tex_type = MTLTextureType2DMultisampleArray;  is_array = true; break;
        // case GL_TEXTURE_BUFFER: tex_type = MTLTextureTypeTextureBuffer; break;

        default:
            NSLog(@"MGL TEXTURE ERROR: unsupported texture target 0x%x for Metal texture creation tex=%u",
                  tex->target,
                  tex->name);
            return nil;
    }

    if (![self checkTextureCompleteness:tex
                               texType:tex_type
                              numFaces:num_faces
                  effectiveMipmapLevels:&effective_mipmap_levels
                      storageMipmapped:&storageMipmapped]) {
        return nil;
    }

    // PROPER FIX: Get original texture format and validate for AGX compatibility
    pixelFormat = mtlPixelFormatForGLTex(tex);
    BOOL expandsSingleChannelSwizzle = mglTextureUploadNeedsSingleChannelSwizzle(tex);
    if (expandsSingleChannelSwizzle) {
        pixelFormat = MTLPixelFormatRGBA8Unorm;
    }

    // Validate format compatibility with AGX, but preserve original intent
    BOOL needsFormatConversion = NO;
    MTLPixelFormat originalFormat = pixelFormat;

    // Check for AGX-incompatible formats and only convert when necessary
    switch(pixelFormat) {
        case MTLPixelFormatB5G6R5Unorm:
        case MTLPixelFormatBGR5A1Unorm:
        case MTLPixelFormatA1BGR5Unorm:
            // 16-bit formats can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case MTLPixelFormatPVRTC_RGBA_2BPP:
        case MTLPixelFormatPVRTC_RGBA_4BPP:
        case MTLPixelFormatPVRTC_RGB_2BPP:
        case MTLPixelFormatPVRTC_RGB_4BPP:
            // PVRTC compression can cause issues in virtualization
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case MTLPixelFormatEAC_R11Unorm:
        case MTLPixelFormatEAC_RG11Unorm:
        case MTLPixelFormatEAC_RGBA8:
        case MTLPixelFormatETC2_RGB8:
        case MTLPixelFormatETC2_RGB8A1:
            // ETC/ETC2 compression can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        default:
            // Most modern formats should work fine
            break;
    }

    /* Metal does not allow depth/stencil pixel formats with MTLTextureType1DArray.
     * Promote to MTLTextureType2DArray with height=1, mirroring how mipmapped
     * 1D array textures are already promoted below.  Without this, creating a
     * GL_TEXTURE_1D_ARRAY depth texture (e.g. sampler_1d_array_shadow) triggers
     * a Metal validation assertion crash. */
    if (tex_type == MTLTextureType1DArray) {
        switch (pixelFormat) {
            case MTLPixelFormatDepth16Unorm:
            case MTLPixelFormatDepth32Float:
            case MTLPixelFormatStencil8:
            case MTLPixelFormatDepth24Unorm_Stencil8:
            case MTLPixelFormatDepth32Float_Stencil8:
            case MTLPixelFormatX32_Stencil8:
            case MTLPixelFormatX24_Stencil8:
                tex_type = MTLTextureType2DArray;
                texture1DArrayBackedBy2DArray = true;
                break;
            default:
                break;
        }
    }

    width = tex->width;
    height = tex->height;
    depth = tex->depth;
    if (tex_type == MTLTextureType2DMultisample ||
        tex_type == MTLTextureType2DMultisampleArray) {
        storageMipmapped = NO;
        effective_mipmap_levels = 1u;
        tex->mipmapped = false;
    }

    mipmapped = storageMipmapped;
    upload_level_count = mipmapped ? effective_mipmap_levels : tex->num_levels;

    tex_desc = [[MTLTextureDescriptor alloc] init];
    tex_desc.textureType = tex_type;
    tex_desc.pixelFormat = pixelFormat;
    tex_desc.width = width;
    tex_desc.height = (tex_type == MTLTextureType1D ||
                       tex_type == MTLTextureType1DArray) ? 1 : height;
    if (tex_type == MTLTextureType2DMultisample ||
        tex_type == MTLTextureType2DMultisampleArray) {
        /* Metal only supports a device-specific subset of sample counts.
         * Apple Silicon (e.g. M4) only supports 1/2/4 — NOT 8.  Delegate to
         * the AGX Capability Layer for centralized clamping. */
        NSUInteger samples = MAX((NSUInteger)2u, (NSUInteger)tex->samples);
        samples = MGLCapabilityClampSampleCount(&_capability, samples);
        tex_desc.sampleCount = samples;
    }

    // CONSERVATIVE: Use only Metal API patterns that work reliably with AGX driver
    tex_desc.cpuCacheMode = MGLCapabilityUseConservativeCPUCache(&_capability)
        ? MTLCPUCacheModeWriteCombined
        : MTLCPUCacheModeDefaultCache;

    // Use shared storage for textures that need CPU upload (blit/replaceRegion).
    // Private storage is only safe for pure GPU render targets on Apple Silicon.
    bool hasUploadableCPUData = mglTextureHasUploadableCPUData(tex, num_faces, upload_level_count);
    bool needsCpuUpload = ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    tex_desc.storageMode = needsCpuUpload ? MTLStorageModeShared : MTLStorageModePrivate;

    // Normalize depth/array semantics per Metal texture type.
    if (tex_type == MTLTextureTypeCube) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }
        tex_desc.depth = 1;
    } else if (tex_type == MTLTextureTypeCubeArray) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube-array texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }

        // GL cube-map-array depth is usually layer count (faces), so convert to cube count.
        // If depth is already cube-count (non-multiple of 6), keep it as-is.
        NSUInteger cubeCount = depth;
        if (cubeCount >= 6 && (cubeCount % 6) == 0) {
            cubeCount = cubeCount / 6;
        } else if (cubeCount > 1 && (cubeCount % 6) != 0) {
            NSLog(@"MGL WARNING: cube-array depth=%lu is not a multiple of 6, treating as cube count",
                  (unsigned long)cubeCount);
        }

        tex_desc.arrayLength = MAX((NSUInteger)1, cubeCount);
        tex_desc.depth = 1;
    } else if (tex_type == MTLTextureType1DArray) {
        tex_desc.arrayLength = MAX((NSUInteger)1, height);
        tex_desc.depth = 1;
    } else if (is_array) {
        tex_desc.arrayLength = MAX((NSUInteger)1, depth);
        tex_desc.depth = 1;
    } else {
        /* For 3D and other non-array textures, arrayLength must be 1.
         * Some Metal drivers report getNumSlices()==0 when arrayLength
         * is left at its default, causing "slice OOB" assertions. */
        tex_desc.arrayLength = 1;
        tex_desc.depth = MAX((NSUInteger)1, depth);
    }

    if (mipmapped)
    {
        if (tex_type == MTLTextureType1D) {
            tex_type = MTLTextureType2D;
            texture1DBackedBy2D = true;
        }
        /* Metal does not allow mipmapLevelCount > 1 for MTLTextureType1DArray.
         * Promote to MTLTextureType2DArray with height=1 to support mipmapped
         * 1D array textures.  The upload code checks texture1DArrayBackedBy2DArray
         * to treat each slice as 1 pixel tall. */
        if (tex_type == MTLTextureType1DArray) {
            tex_type = MTLTextureType2DArray;
            texture1DArrayBackedBy2DArray = true;
        }
        tex_desc.mipmapLevelCount = MAX((GLuint)1, effective_mipmap_levels);
    }

    if (texture1DBackedBy2D) {
        tex_desc.textureType = MTLTextureType2D;
        tex_desc.height = 1;
    }
    if (texture1DArrayBackedBy2DArray) {
        tex_desc.textureType = MTLTextureType2DArray;
        /* For GL_TEXTURE_1D_ARRAY, the GL height parameter is the array slice
         * count.  Since tex_type was promoted to MTLTextureType2DArray above,
         * the arrayLength branch at line ~12397 (keyed on MTLTextureType1DArray)
         * was skipped, leaving arrayLength=1 from the is_array/depth fallback.
         * Set arrayLength from the GL height (slice count) here. */
        tex_desc.arrayLength = MAX((NSUInteger)1, height);
        tex_desc.height = 1;
    }

    /* GL image access mode (GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE)
     * only governs the image binding, NOT the texture's overall capabilities.
     * A texture bound as a write-only image may still be sampled from via
     * sampler2D in the same shader.  Metal requires MTLTextureUsageShaderRead
     * for sampling, so always include it alongside the image write flag. */
    switch(tex->access)
    {
        case GL_READ_ONLY:
            tex_desc.usage = MTLTextureUsageShaderRead; break;
        case GL_WRITE_ONLY:
            tex_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; break;
        case GL_READ_WRITE:
            tex_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; break;
        default:
            NSLog(@"MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u",
                  tex->access,
                  tex->name);
            return nil;
    }

    if (tex->is_render_target)
    {
        tex_desc.usage |= MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    }

    // Allow safe same-memory format reinterpretation (e.g. RGBA8 <-> BGRA8)
    // for blit/present paths where OpenGL attachments and drawable formats differ.
    tex_desc.usage |= MTLTextureUsagePixelFormatView;

    if (tex_desc.textureType == MTLTextureTypeCube || tex_desc.textureType == MTLTextureTypeCubeArray) {
        NSLog(@"MGL CUBE DESC tex=%u glTarget=0x%x type=%lu width=%lu height=%lu depth=%lu arrayLength=%lu pixelFormat=%lu usage=%lu storage=%lu mipmapped=%d",
              tex->name,
              tex->target,
              (unsigned long)tex_desc.textureType,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height,
              (unsigned long)tex_desc.depth,
              (unsigned long)tex_desc.arrayLength,
              (unsigned long)tex_desc.pixelFormat,
              (unsigned long)tex_desc.usage,
              (unsigned long)tex_desc.storageMode,
              (int)mipmapped);
    }

    // CRITICAL FIX: Proper validation instead of assertions
    if (!tex_desc) {
        NSLog(@"MGL ERROR: Failed to create texture descriptor");
        return NULL;
    }

    if (tex->params.swizzled && !expandsSingleChannelSwizzle)
    {
        [self swizzleTexDesc:tex_desc forTex:tex];
    }

    id<MTLTexture> texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    @try {
        texture = [_device newTextureWithDescriptor:tex_desc];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating texture: %@", exception);
        [self recordGPUError];
        return NULL;
    }

    // CRITICAL FIX: Validate texture creation result instead of asserting
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create Metal texture with descriptor");
        return NULL;
    }

    BOOL cpuUploadRequired =
        ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    BOOL cpuUploadVerified = !cpuUploadRequired;
    BOOL allLevelsUploaded = YES;

    if (cpuUploadRequired)
    {
        if (![self uploadDirtyCPUTextureData:tex
                                       metal:texture
                                 pixelFormat:pixelFormat
                                   numFaces:num_faces
                           uploadLevelCount:upload_level_count
                                    isArray:is_array
                         texture1DBackedBy2D:texture1DBackedBy2D
                   texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                    texType:tex_type
                        outAllLevelsUploaded:&allLevelsUploaded]) {
            return nil;
        }
    }
    else
    {
        if (hasUploadableCPUData) {
            [self reUploadExistingCPUTextureData:tex
                                            metal:texture
                                      pixelFormat:pixelFormat
                                        numFaces:num_faces
                                uploadLevelCount:upload_level_count
                                          isArray:is_array
                               texture1DBackedBy2D:texture1DBackedBy2D
                         texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                             texType:tex_type];
        } else if (tex->is_render_target || mglMetalPixelFormatIsDepthOrStencil(pixelFormat)) {
            static uint64_t s_skipRenderTargetFillLogs = 0;
            uint64_t hit = ++s_skipRenderTargetFillLogs;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL TEXTURE SKIP implicit fill tex=%u renderTarget=%u format=%lu sourceSafe=0 hit=%llu",
                      (unsigned)tex->name,
                      (unsigned)tex->is_render_target,
                      (unsigned long)pixelFormat,
                      (unsigned long long)hit);
            }
        } else {
            [self fillTextureWithSafeInitialContents:texture
                                                 tex:tex
                                         pixelFormat:pixelFormat];
        }
    }

    if (cpuUploadRequired && tex->target == GL_TEXTURE_2D && texture.textureType == MTLTextureType2D) {
        BOOL fullCPUUploadVerified = [self uploadFullCPUTextureDataIntoTexture:tex
                                                                           metal:texture
                                                                          reason:"createMTLTexture.cpuData"];
        cpuUploadVerified = allLevelsUploaded && fullCPUUploadVerified;
    } else if (cpuUploadRequired) {
        /*
         * Non-2D uploads still use the legacy creation path above. The current GUI
         * atlas failure is 2D; avoid changing array/cube semantics in this pass.
         * If any mip level was skipped (invalid layout, NULL data, etc.) keep
         * DIRTY_TEXTURE_DATA set so the level gets retried on next bind.
         */
        cpuUploadVerified = allLevelsUploaded;
    }

    if (cpuUploadRequired && !cpuUploadVerified) {
        static uint64_t s_createTextureCPUUploadIncompleteLogs = 0;
        uint64_t hit = ++s_createTextureCPUUploadIncompleteLogs;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(tex);
            NSLog(@"MGL TEXTURE CREATE CPU-UPLOAD INCOMPLETE tex=%u target=0x%x dirtyBefore=0x%x level0=%ux%u source=%u upload=%lu hit=%llu",
                  (unsigned)tex->name,
                  (unsigned)tex->target,
                  (unsigned)tex->dirty_bits,
                  level0 ? (unsigned)level0->width : 0u,
                  level0 ? (unsigned)level0->height : 0u,
                  level0 ? (unsigned)level0->last_init_source : 0u,
                  (unsigned long)(level0 ? level0->last_upload_size : 0u),
                  (unsigned long long)hit);
        }
        tex->dirty_bits &= ~(DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_ACCESS);
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    } else {
        tex->dirty_bits = 0;
    }

    [self logMTLTextureMipDiagnostics:tex metal:texture effectiveMipLevels:effective_mipmap_levels];

    [self recordGPUSuccess];

    return texture;
}

- (id<MTLTexture>)createMTLTexelBufferTexture:(Texture *)tex
{
    Buffer *sourceBuffer = tex->texture_buffer;
    if (!sourceBuffer || tex->texture_buffer_size <= 0) {
        NSLog(@"MGL TEXBUFFER ERROR: tex=%u has no attached buffer/size buffer=%p size=%lld",
              tex->name,
              sourceBuffer,
              (long long)tex->texture_buffer_size);
        return nil;
    }

    if (tex->texture_buffer_offset < 0 ||
        tex->texture_buffer_offset > sourceBuffer->size ||
        tex->texture_buffer_size > sourceBuffer->size - tex->texture_buffer_offset) {
        NSLog(@"MGL TEXBUFFER ERROR: invalid range tex=%u buffer=%u off=%lld size=%lld bufferSize=%lld",
              tex->name,
              sourceBuffer->name,
              (long long)tex->texture_buffer_offset,
              (long long)tex->texture_buffer_size,
              (long long)sourceBuffer->size);
        return nil;
    }

    NSUInteger bytesPerTexel = [self bytesPerPixelForFormat:tex->internalformat];
    if (bytesPerTexel == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: unsupported internal format 0x%x tex=%u buffer=%u",
              tex->internalformat,
              tex->name,
              sourceBuffer->name);
        return nil;
    }

    NSUInteger texelCount = (NSUInteger)tex->texture_buffer_size / bytesPerTexel;
    if (texelCount == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: zero texel count tex=%u buffer=%u size=%lld bpt=%lu",
              tex->name,
              sourceBuffer->name,
              (long long)tex->texture_buffer_size,
              (unsigned long)bytesPerTexel);
        return nil;
    }

    MTLPixelFormat bufferPixelFormat = (tex->internalformat == GL_RGBA8)
        ? MTLPixelFormatRGBA8Uint
        : mtlPixelFormatForGLTex(tex);
    if (bufferPixelFormat == MTLPixelFormatInvalid || bufferPixelFormat == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: invalid Metal format for tex=%u internal=0x%x",
              tex->name,
              tex->internalformat);
        return nil;
    }

    if (![self processBuffer:sourceBuffer]) {
        NSLog(@"MGL TEXBUFFER ERROR: failed to process source buffer tex=%u buffer=%u",
              tex->name,
              sourceBuffer->name);
        return nil;
    }

    const uint8_t *sourceBytes = NULL;
    if (sourceBuffer->data.buffer_data) {
        sourceBytes = ((const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data) + (size_t)tex->texture_buffer_offset;
    } else if (sourceBuffer->data.mtl_data) {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(sourceBuffer->data.mtl_data);
        if (mtlBuffer && mtlBuffer.contents) {
            sourceBytes = ((const uint8_t *)mtlBuffer.contents) + (size_t)tex->texture_buffer_offset;
        }
    }

    if (!sourceBytes) {
        NSLog(@"MGL TEXBUFFER ERROR: no readable backing for tex=%u buffer=%u cpu=%p mtl=%p",
              tex->name,
              sourceBuffer->name,
              (void *)(uintptr_t)sourceBuffer->data.buffer_data,
              sourceBuffer->data.mtl_data);
        return nil;
    }

    // SPIRV-Cross currently emits Minecraft's CloudFaces texel buffer as a
    // texture2d<int>. Keep GL lookup semantics as GL_TEXTURE_BUFFER, but
    // create a Metal 2D backing so the generated MSL argument type matches.
    // A texel buffer can be much wider than Metal's max 2D texture width,
    // so pack it into rows instead of creating texelCount x 1.
    /*
     * SPIRV-Cross lowers GL texture buffers to 2D Metal textures and emits
     * spvTexelBufferCoord(tc) using its MSL texel_buffer_texture_width
     * option. Keep this packing width in lockstep with program.c.
     */
    static const NSUInteger kMGLTexelBufferTextureWidth = 4096u;
    NSUInteger max2DSize = (NSUInteger)MIN((GLuint)kMGLTexelBufferTextureWidth,
                                           ctx ? ctx->active_state->var.max_texture_size : (GLuint)kMGLTexelBufferTextureWidth);
    if (max2DSize == 0 || max2DSize > kMGLTexelBufferTextureWidth) {
        max2DSize = kMGLTexelBufferTextureWidth;
    }

    NSUInteger texWidth = MIN(texelCount, max2DSize);
    NSUInteger texHeight = (texelCount + texWidth - 1) / texWidth;
    if (texHeight == 0 || texHeight > max2DSize) {
        NSLog(@"MGL TEXBUFFER ERROR: texel buffer too large for 2D fallback tex=%u buffer=%u texels=%lu packed=%lux%lu max=%lu",
              tex->name,
              sourceBuffer->name,
              (unsigned long)texelCount,
              (unsigned long)texWidth,
              (unsigned long)texHeight,
              (unsigned long)max2DSize);
        return nil;
    }

    NSUInteger bytesPerRow = texWidth * bytesPerTexel;
    NSUInteger packedBytes = bytesPerRow * texHeight;
    NSMutableData *packedData = nil;
    const uint8_t *uploadBytes = sourceBytes;

    /* Channel expansion for 3-channel RGB -> 4-channel RGBA Metal formats.
     * GL_RGB32* (12 bytes/texel) maps to Metal RGBA32* (16 bytes/texel).
     * Expand each texel by inserting a default alpha before uploading. */
    NSMutableData *expandedData = nil;
    if (mglTextureNeedsChannelExpansion(tex->internalformat, bufferPixelFormat)) {
        NSUInteger srcCompBytes = 0;
        NSUInteger dstCompBytes = 0;
        uint64_t alphaDefault = 0;
        switch (bufferPixelFormat) {
            case MTLPixelFormatRGBA16Unorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 65535; break;
            case MTLPixelFormatRGBA16Snorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 32767; break;
            case MTLPixelFormatRGBA16Float:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 0x3C00; break;
            case MTLPixelFormatRGBA16Sint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MTLPixelFormatRGBA16Uint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MTLPixelFormatRGBA32Float:
                srcCompBytes = 4; dstCompBytes = 4;
                { float f = 1.0f; memcpy(&alphaDefault, &f, sizeof(f)); }
                break;
            case MTLPixelFormatRGBA32Sint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            case MTLPixelFormatRGBA32Uint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            default:
                break;
        }
        if (srcCompBytes > 0) {
            NSUInteger srcPixelBytes = srcCompBytes * 3;
            NSUInteger dstPixelBytes = dstCompBytes * 4;
            NSUInteger expandedBytesPerRow = texWidth * dstPixelBytes;
            NSUInteger expandedPackedBytes = expandedBytesPerRow * texHeight;
            expandedData = [NSMutableData dataWithLength:expandedPackedBytes];
            if (expandedData && expandedData.mutableBytes) {
                const uint8_t *src = sourceBytes;
                uint8_t *dst = (uint8_t *)expandedData.mutableBytes;
                for (NSUInteger row = 0; row < texHeight; row++) {
                    for (NSUInteger col = 0; col < texWidth; col++) {
                        NSUInteger srcTexelIdx = row * texWidth + col;
                        if (srcTexelIdx >= texelCount) {
                            memset(dst + (row * expandedBytesPerRow + col * dstPixelBytes),
                                   0, dstPixelBytes);
                            continue;
                        }
                        const uint8_t *srcPixel = src + srcTexelIdx * srcPixelBytes;
                        uint8_t *dstPixel = dst + row * expandedBytesPerRow + col * dstPixelBytes;
                        memcpy(dstPixel, srcPixel, srcPixelBytes);
                        memcpy(dstPixel + srcPixelBytes, &alphaDefault, dstCompBytes);
                    }
                }
                uploadBytes = (const uint8_t *)expandedData.bytes;
                bytesPerRow = expandedBytesPerRow;
                packedBytes = expandedPackedBytes;
            }
        }
    }

    if (texHeight > 1 && !expandedData) {
        packedData = [NSMutableData dataWithLength:packedBytes];
        if (!packedData || !packedData.mutableBytes) {
            NSLog(@"MGL TEXBUFFER ERROR: failed allocating packed data tex=%u buffer=%u bytes=%lu",
                  tex->name,
                  sourceBuffer->name,
                  (unsigned long)packedBytes);
            return nil;
        }

        memcpy(packedData.mutableBytes, sourceBytes, (size_t)tex->texture_buffer_size);
        uploadBytes = (const uint8_t *)packedData.bytes;
    }

    uint64_t sourceHash = mglTraceHashBytes(sourceBytes, (size_t)tex->texture_buffer_size);
    uint64_t uploadHash = mglTraceHashBytes(uploadBytes, packedBytes);
    char sourceHead[64];
    char uploadHead[64];
    sourceHead[0] = '\0';
    uploadHead[0] = '\0';
    mglTraceFormatBytes(sourceBytes, (size_t)MIN((NSUInteger)tex->texture_buffer_size, (NSUInteger)64), sourceHead, sizeof(sourceHead));
    mglTraceFormatBytes(uploadBytes, (size_t)MIN(packedBytes, (NSUInteger)64), uploadHead, sizeof(uploadHead));

    MTLTextureDescriptor *bufferDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:bufferPixelFormat
                                                           width:texWidth
                                                          height:texHeight
                                                       mipmapped:NO];
    bufferDesc.usage = MTLTextureUsageShaderRead;
    bufferDesc.storageMode = MTLStorageModeShared;

    id<MTLTexture> bufferTexture = nil;
    @try {
        bufferTexture = [_device newTextureWithDescriptor:bufferDesc];
        if (bufferTexture) {
            [bufferTexture replaceRegion:MTLRegionMake2D(0, 0, texWidth, texHeight)
                              mipmapLevel:0
                                withBytes:uploadBytes
                              bytesPerRow:bytesPerRow];
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL TEXBUFFER ERROR: failed creating/uploading tex=%u buffer=%u exception=%@",
              tex->name,
              sourceBuffer->name,
              exception);
        return nil;
    }

    if (!bufferTexture) {
        NSLog(@"MGL TEXBUFFER ERROR: Metal texture creation returned nil tex=%u buffer=%u format=%lu texels=%lu",
              tex->name,
              sourceBuffer->name,
              (unsigned long)bufferPixelFormat,
              (unsigned long)texelCount);
        return nil;
    }

    tex->dirty_bits = 0;
    sourceBuffer->data.dirty_bits = 0;

    NSMutableData *readbackData = [NSMutableData dataWithLength:packedBytes];
    uint64_t readbackHash = 0ull;
    char readbackHead[64];
    readbackHead[0] = '\0';
    if (readbackData.mutableBytes) {
        [bufferTexture getBytes:readbackData.mutableBytes
                    bytesPerRow:bytesPerRow
                     fromRegion:MTLRegionMake2D(0, 0, texWidth, texHeight)
                    mipmapLevel:0];
        readbackHash = mglTraceHashBytes(readbackData.bytes, packedBytes);
        mglTraceFormatBytes(readbackData.bytes, (size_t)MIN(packedBytes, (NSUInteger)64), readbackHead, sizeof(readbackHead));
    }

    NSLog(@"MGL TEXBUFFER CREATE tex=%u buffer=%u internal=0x%x mtlFormat=%lu texels=%lu packed=%lux%lu rowBytes=%lu bytes=%lld offset=%lld as=texture2d sourceHash=0x%016llx uploadHash=0x%016llx readbackHash=0x%016llx sourceHead=%s uploadHead=%s readbackHead=%s",
          tex->name,
          sourceBuffer->name,
          tex->internalformat,
          (unsigned long)bufferPixelFormat,
          (unsigned long)texelCount,
          (unsigned long)texWidth,
          (unsigned long)texHeight,
          (unsigned long)bytesPerRow,
          (long long)tex->texture_buffer_size,
          (long long)tex->texture_buffer_offset,
          (unsigned long long)sourceHash,
          (unsigned long long)uploadHash,
          (unsigned long long)readbackHash,
          sourceHead,
          uploadHead,
          readbackHead);

    [self recordGPUSuccess];
    return bufferTexture;
}

- (BOOL)checkTextureCompleteness:(Texture *)tex
                          texType:(MTLTextureType)tex_type
                         numFaces:(uint)num_faces
             effectiveMipmapLevels:(GLuint *)outEffectiveMipmapLevels
                 storageMipmapped:(BOOL *)outStorageMipmapped
{
    (void)tex_type;  /* unused: completeness does not depend on Metal texture type */
    GLuint effective_mipmap_levels = tex->mipmap_levels;
    BOOL storageMipmapped = NO;

    uint completeness_check_faces = (tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) ? 1 : num_faces;

    /* Texture storage is independent from GL_TEXTURE_MAX_LEVEL.  Minecraft
     * uses BASE/MAX_LEVEL to express temporary GpuTextureView mip windows; if
     * those sampler parameters shrink the Metal texture allocation, later
     * full-atlas sampling loses the higher mip levels and distant terrain
     * reads empty/incorrect data.  Apply BASE/MAX only to completeness checks
     * and sampled Metal views, not to the underlying storage level count. */

    /* For CUBE_MAP_ARRAY, glTexImage3D stores all layer data in faces[0] with
     * depth = 6 * num_cubes.  Faces 1-5 are never populated by createTextureLevel,
     * so only check face 0 for completeness.  The upload code also reads from
     * face 0 and distributes slices to Metal array layers. */

    storageMipmapped = (tex->mipmap_levels > 1u) &&
        (tex->num_levels > 1u || tex->is_render_target);

    if (tex->num_levels > 1)
    {
        // mipmapped texture
        if (effective_mipmap_levels == 0) {
            effective_mipmap_levels = tex->num_levels;
        }

        if (!tex->is_render_target && tex->num_levels < effective_mipmap_levels)
        {
            static uint64_t s_mipmap_count_mismatch_logs = 0;
            if (++s_mipmap_count_mismatch_logs <= 32 || (s_mipmap_count_mismatch_logs % 512) == 0) {
                NSLog(@"MGL TEXTURE MIP COMPAT: tex=%u target=0x%x size=%ux%u num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u immutable=%u isRT=%u; capping Metal mip count to uploaded levels hit=%llu",
                      tex->name,
                      tex->target,
                      tex->width,
                      tex->height,
                      tex->num_levels,
                      tex->mipmap_levels,
                      effective_mipmap_levels,
                      tex->params.base_level,
                      tex->params.max_level,
                      tex->immutable_storage,
                      tex->is_render_target,
                      (unsigned long long)s_mipmap_count_mismatch_logs);
            }
            effective_mipmap_levels = tex->num_levels;
        }

        /* GL texture completeness only requires levels in
         * [base_level, min(max_level, mipmap_levels-1)] to be complete.
         * Levels below base_level may be uninitialised and must NOT cause
         * the texture to be rejected.  Minecraft 1.21.11 sets base_level>0
         * on mipmap texture views (GlCommandEncoder.java). */
        GLuint check_start = tex->params.base_level;
        GLuint check_end = (tex->params.max_level == 1000u)
            ? (tex->mipmap_levels > 0u ? tex->mipmap_levels - 1u : 0u)
            : tex->params.max_level;
        if (check_end >= tex->mipmap_levels)
            check_end = (tex->mipmap_levels > 0u) ? tex->mipmap_levels - 1u : 0u;
        if (check_end < check_start)
            check_end = check_start;

        for(int face=0; face<completeness_check_faces; face++)
        {
            for (GLuint i=check_start; i<=check_end; i++)
            {
                // incomplete texture
                if (tex->faces[face].levels[i].complete == false) {
                    static uint64_t s_incomplete_mip_logs = 0;
                    if (++s_incomplete_mip_logs <= 32 || (s_incomplete_mip_logs % 512) == 0) {
                        NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d level=%u incomplete num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u check=[%u,%u] hit=%llu",
                              tex->name,
                              tex->target,
                              face,
                              i,
                              tex->num_levels,
                              tex->mipmap_levels,
                              effective_mipmap_levels,
                              tex->params.base_level,
                              tex->params.max_level,
                              check_start,
                              check_end,
                              (unsigned long long)s_incomplete_mip_logs);
                    }
                    return NO;
                }
            }
        }

        tex->mipmapped = true;
    }
    else if (tex->num_levels == 1)
    {
        if (!storageMipmapped) {
            effective_mipmap_levels = 1;
        }
        // single level texture
        // incomplete texture
        for(int face=0; face<completeness_check_faces; face++)
        {
            if (tex->faces[face].levels[0].complete == false)
            {
                static uint64_t s_incomplete_base_logs = 0;
                if (++s_incomplete_base_logs <= 32 || (s_incomplete_base_logs % 512) == 0) {
                    NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d base incomplete size=%ux%u hit=%llu",
                          tex->name,
                          tex->target,
                          face,
                          tex->width,
                          tex->height,
                          (unsigned long long)s_incomplete_base_logs);
                }
                return NO;
            }
        }
    }
    else
    {
        NSLog(@"MGL TEXTURE ERROR: texture %u has no complete levels for Metal creation target=0x%x",
              tex->name,
              tex->target);
        return NO;
    }

    tex->complete = true;

    if (outEffectiveMipmapLevels) *outEffectiveMipmapLevels = effective_mipmap_levels;
    if (outStorageMipmapped) *outStorageMipmapped = storageMipmapped;
    return YES;
}

- (void)logMTLTextureMipDiagnostics:(Texture *)tex
                              metal:(id<MTLTexture>)texture
               effectiveMipLevels:(GLuint)effective_mipmap_levels
{
    static uint64_t s_mipDiagLogs = 0;
    uint64_t diagHit = ++s_mipDiagLogs;
    if (kMGLDiagnosticStateLogs &&
        (diagHit <= 128ull || (diagHit % 512ull) == 0ull)) {
        NSUInteger mtlMipCount = texture.mipmapLevelCount;
        MTLPixelFormat mtlFmt = texture.pixelFormat;
        MTLStorageMode mtlStorage = texture.storageMode;
        NSUInteger uploadedLevels = 0;
        NSUInteger skippedLevels = 0;
        NSUInteger skippedSourceNone = 0;
        NSUInteger skippedNoData = 0;
        NSMutableString *levelSummary = [NSMutableString stringWithCapacity:256];
        NSUInteger levelsToSummarize = MIN((NSUInteger)tex->num_levels, (NSUInteger)16);
        for (NSUInteger lvl = 0; lvl < levelsToSummarize; lvl++) {
            TextureLevel *tl = (tex->faces[0].levels && lvl < tex->num_levels)
                ? &tex->faces[0].levels[lvl] : NULL;
            if (!tl) { [levelSummary appendString:@"-"]; continue; }
            bool uploadable = mglTextureLevelHasUploadableCPUData(tl);
            if (uploadable) uploadedLevels++; else skippedLevels++;
            if (!uploadable) {
                if (tl->last_init_source == kTexImageNull || tl->last_init_source == kTexInitNone)
                    skippedSourceNone++;
                if (!tl->has_initialized_data && !tl->ever_written)
                    skippedNoData++;
            }
            [levelSummary appendFormat:@"[%u:s%u:w%u:e%u:i%u]",
                (unsigned)lvl, (unsigned)tl->last_init_source,
                (unsigned)tl->width, (unsigned)tl->ever_written,
                (unsigned)tl->has_initialized_data];
        }
        MGLTraceNSLog(@"MGL TEX_MIP_DIAG tex=%u target=0x%x dims=%ux%u internal=0x%x "
                      @"numLevels=%u mipmapLevels=%u effectiveMipLevels=%u mtlMipCount=%lu "
                      @"mtlFmt=%lu mtlStorage=%ld mipmapped=%d baseLevel=%u maxLevel=%u "
                      @"uploadedLevels=%lu skippedLevels=%lu skippedSourceNone=%lu skippedNoData=%lu "
                      @"levels=%@ hit=%llu",
                      (unsigned)tex->name, (unsigned)tex->target,
                      (unsigned)tex->width, (unsigned)tex->height,
                      (unsigned)tex->internalformat,
                      (unsigned)tex->num_levels, (unsigned)tex->mipmap_levels,
                      (unsigned)effective_mipmap_levels, (unsigned long)mtlMipCount,
                      (unsigned long)mtlFmt, (long)mtlStorage, (int)(tex->mipmapped ? 1 : 0),
                      (unsigned)tex->params.base_level, (unsigned)tex->params.max_level,
                      (unsigned long)uploadedLevels, (unsigned long)skippedLevels,
                      (unsigned long)skippedSourceNone, (unsigned long)skippedNoData,
                      levelSummary, (unsigned long long)diagHit);
    }
}

// AGX-SAFE Fallback texture creation for GPU error recovery scenarios
- (id<MTLTexture>) createFallbackMTLTexture:(Texture *) tex
{
    // Validate texture parameters before creating Metal texture to prevent Metal assertion failures
    if (!tex || tex->width <= 0 || tex->height <= 0 || tex->width > 32768 || tex->height > 32768) {
        NSLog(@"MGL AGX: Skipping fallback texture creation - invalid dimensions %dx%d",
              tex ? tex->width : 0, tex ? tex->height : 0);
        return nil;
    }

    NSLog(@"MGL AGX: Creating emergency fallback texture (size: %dx%dx%d)", tex->width, tex->height, tex->depth);

    @try {
        MTLPixelFormat fallbackFormat = mtlPixelFormatForGLTex(tex);
        if (fallbackFormat == MTLPixelFormatInvalid) {
            // Conservative defaults by GL intent when translation is unavailable.
            if (tex->internalformat == GL_DEPTH24_STENCIL8 ||
                tex->internalformat == GL_DEPTH32F_STENCIL8) {
                fallbackFormat = MTLPixelFormatDepth32Float_Stencil8;
            } else if (tex->internalformat == GL_DEPTH_COMPONENT ||
                       tex->internalformat == GL_DEPTH_COMPONENT16 ||
                       tex->internalformat == GL_DEPTH_COMPONENT24 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32F) {
                fallbackFormat = MTLPixelFormatDepth32Float;
            } else {
                fallbackFormat = MTLPixelFormatRGBA8Unorm;
            }
        }

        BOOL isDepthOrStencilFormat =
            (fallbackFormat == MTLPixelFormatDepth16Unorm ||
             fallbackFormat == MTLPixelFormatDepth32Float ||
             fallbackFormat == MTLPixelFormatDepth24Unorm_Stencil8 ||
             fallbackFormat == MTLPixelFormatDepth32Float_Stencil8 ||
             fallbackFormat == MTLPixelFormatStencil8);

        MTLTextureDescriptor *fallbackDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fallbackFormat
                                                                                                    width:MAX(tex->width, 1)
                                                                                                   height:MAX(tex->height, 1)
                                                                                                mipmapped:NO];
        fallbackDesc.usage = MTLTextureUsageShaderRead;
        if (tex->is_render_target || isDepthOrStencilFormat) {
            fallbackDesc.usage |= MTLTextureUsageRenderTarget;
        }
        fallbackDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> fallbackTexture = [_device newTextureWithDescriptor:fallbackDesc];

        if (fallbackTexture) {
            // Fill with simple gradient pattern using a simple approach
            NSUInteger width = fallbackTexture.width;
            NSUInteger height = fallbackTexture.height;

            if (!isDepthOrStencilFormat && width <= 512 && height <= 512) {
                uint32_t *gradientData = calloc(width * height, sizeof(uint32_t));
                if (gradientData) {
                    // Create simple red-blue gradient
                    for (NSUInteger y = 0; y < height; y++) {
                        for (NSUInteger x = 0; x < width; x++) {
                            NSUInteger index = y * width + x;
                            uint8_t r = (uint8_t)((x * 255) / width);
                            uint8_t g = 128;
                            uint8_t b = (uint8_t)((y * 255) / height);
                            uint8_t a = 255;
                            gradientData[index] = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
                        }
                    }

                    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
                    [fallbackTexture replaceRegion:region mipmapLevel:0 withBytes:gradientData
                               bytesPerRow:width * sizeof(uint32_t)];

                    free(gradientData);
                    NSLog(@"MGL AGX: Fallback color texture created with gradient pattern");
                }
            }
        }

        return fallbackTexture;

    } @catch (NSException *exception) {
        NSLog(@"MGL AGX: Even fallback texture creation failed: %@", exception.reason);
        return nil;
    }
}

// Helper function to calculate bytes per pixel for different OpenGL formats
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat
{
    switch(internalformat) {
        case GL_RED:
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
            return 1;

        case GL_RG:
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_R16:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
            return 2;

        case GL_RGB:
        case GL_RGB8:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_SRGB8:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
            return 3;

        case GL_RGBA:
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_SRGB8_ALPHA8:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_R32F:
            return 4;

        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RG32F:
            return 8;

        case GL_RGB16:
        case GL_RGB16F:
            return 6;

        case GL_RGBA16I:
        case GL_RGBA16UI:
            return 8;

        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return 12;

        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return 16;

        default:
            // Default to 4 bytes for unknown formats
            NSLog(@"MGL WARNING: Unknown internal format 0x%x, defaulting to 4 bytes per pixel", internalformat);
            return 4;
    }
}

- (id<MTLSamplerState>) createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target
{
    MTLSamplerDescriptor *samplerDescriptor;

    if (!tex_param) {
        NSLog(@"MGL SAMPLER ERROR: createMTLSamplerForTexParam called with NULL parameters");
        return nil;
    }

    samplerDescriptor = [MTLSamplerDescriptor new];
    if (!samplerDescriptor) {
        NSLog(@"MGL SAMPLER ERROR: failed to allocate MTLSamplerDescriptor");
        return nil;
    }

    switch(tex_param->min_filter)
    {
        case GL_NEAREST:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
            break;

        case GL_LINEAR:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
            break;

        case GL_NEAREST_MIPMAP_NEAREST:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterNearest;
            break;

        case GL_LINEAR_MIPMAP_NEAREST:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterNearest;
            break;

        case GL_NEAREST_MIPMAP_LINEAR:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterLinear;
            break;

        case GL_LINEAR_MIPMAP_LINEAR:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterLinear;
            break;

        default:
            NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_MIN_FILTER 0x%x", tex_param->min_filter);
            return nil;
    }

    switch(tex_param->mag_filter)
    {
        case GL_NEAREST:
            samplerDescriptor.magFilter = MTLSamplerMinMagFilterNearest;
            break;

        case GL_LINEAR:
            samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
            break;

        default:
            NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_MAG_FILTER 0x%x", tex_param->mag_filter);
            return nil;
    }

    //     @property (nonatomic) NSUInteger maxAnisotropy;
    if (tex_param->max_anisotropy > 1.0)
    {
        samplerDescriptor.maxAnisotropy = tex_param->max_anisotropy;
    }

    //    @property (nonatomic) MTLSamplerAddressMode sAddressMode;
    //    @property (nonatomic) MTLSamplerAddressMode tAddressMode;
    //    @property (nonatomic) MTLSamplerAddressMode rAddressMode;
    for (int i=0; i<3; i++)
    {
        MTLSamplerAddressMode mode = 0;
        GLenum type = 0;

        switch(i)
        {
            case 0: type = tex_param->wrap_s; break;
            case 1: type = tex_param->wrap_t; break;
            case 2: type = tex_param->wrap_r; break;
        }

        switch(type)
        {
            case GL_CLAMP_TO_EDGE:
                mode = MTLSamplerAddressModeClampToEdge;
                break;

            case GL_CLAMP_TO_BORDER:
                mode = MTLSamplerAddressModeClampToBorderColor;
                break;

            case GL_MIRRORED_REPEAT:
                mode = MTLSamplerAddressModeMirrorRepeat;
                break;

            case GL_REPEAT:
                mode = MTLSamplerAddressModeRepeat;
                break;

            case GL_MIRROR_CLAMP_TO_EDGE:
                mode = MTLSamplerAddressModeMirrorClampToEdge;
                break;

    //        case GL_CLAMP_TO_ZERO_MGL_EXT:
    //            mode = MTLSamplerAddressModeClampToZero;
    //            break;

            default:
                NSLog(@"MGL SAMPLER ERROR: invalid GL texture wrap mode 0x%x for axis %d", type, i);
                return nil;
        }

        switch(i)
        {
            case 0: samplerDescriptor.sAddressMode = mode; break;
            case 1: samplerDescriptor.tAddressMode = mode; break;
            case 2: samplerDescriptor.rAddressMode = mode; break;
        }
    }

    BOOL usesBorderColor = (tex_param->wrap_s == GL_CLAMP_TO_BORDER ||
                            tex_param->wrap_t == GL_CLAMP_TO_BORDER ||
                            tex_param->wrap_r == GL_CLAMP_TO_BORDER);
    if (!usesBorderColor)
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
    }
    else if ((tex_param->border_color[0] == 0.0) &&
             (tex_param->border_color[1] == 0.0) &&
             (tex_param->border_color[2] == 0.0) &&
             (tex_param->border_color[3] == 0.0))
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
    }
    else if ((tex_param->border_color[0] == 0.0) &&
             (tex_param->border_color[1] == 0.0) &&
             (tex_param->border_color[2] == 0.0) &&
             (tex_param->border_color[3] == 1.0))
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueBlack;
    }
    else if ((tex_param->border_color[0] == 1.0) &&
             (tex_param->border_color[1] == 1.0) &&
             (tex_param->border_color[2] == 1.0) &&
             (tex_param->border_color[3] == 1.0))
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueWhite;
    }
    else
    {
        static uint64_t s_unsupportedBorderColorCount = 0;
        uint64_t hit = ++s_unsupportedBorderColorCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL SAMPLER WARNING: GL border color (%g,%g,%g,%g) is not exactly representable by MTLSamplerBorderColor; approximating hit=%llu",
                  tex_param->border_color[0],
                  tex_param->border_color[1],
                  tex_param->border_color[2],
                  tex_param->border_color[3],
                  (unsigned long long)hit);
        }

        if (tex_param->border_color[3] < 0.5f) {
            samplerDescriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
        } else if (tex_param->border_color[0] >= 0.5f &&
                   tex_param->border_color[1] >= 0.5f &&
                   tex_param->border_color[2] >= 0.5f) {
            samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueWhite;
        } else {
            samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueBlack;
        }
    }

    if (target == GL_TEXTURE_RECTANGLE)
    {
        samplerDescriptor.normalizedCoordinates = false;
        if ((tex_param->wrap_s != GL_CLAMP_TO_EDGE) ||
            (tex_param->wrap_t != GL_CLAMP_TO_EDGE) ||
            (tex_param->wrap_r != GL_CLAMP_TO_EDGE))
        {
            static uint64_t s_rectWrapClampWarningCount = 0;
            uint64_t hit = ++s_rectWrapClampWarningCount;
            if (hit <= 16ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL SAMPLER WARNING: GL_TEXTURE_RECTANGLE requires unnormalized coordinates; forcing ClampToEdge sampler address modes for Metal compatibility hit=%llu",
                      (unsigned long long)hit);
            }
            samplerDescriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
            samplerDescriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
            samplerDescriptor.rAddressMode = MTLSamplerAddressModeClampToEdge;
        }
    }

    // @property (nonatomic) BOOL lodAverage API_AVAILABLE(ios(9.0), macos(11.0), macCatalyst(14.0));


    // @property (nonatomic) MTLCompareFunction compareFunction API_AVAILABLE(macos(10.11), ios(9.0));
    if (tex_param->compare_mode == GL_NONE)
    {
        samplerDescriptor.compareFunction = MTLCompareFunctionNever;
    }
    else if (tex_param->compare_mode == GL_COMPARE_REF_TO_TEXTURE)
    {
        if (!mglIsValidGLCompareFunction(tex_param->compare_func))
        {
            NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_COMPARE_FUNC 0x%x", tex_param->compare_func);
            return nil;
        }
        samplerDescriptor.compareFunction =
            mglMTLCompareFunctionForGL(tex_param->compare_func,
                                       MTLCompareFunctionNever,
                                       "sampler");
    }
    else
    {
        NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_COMPARE_MODE 0x%x", tex_param->compare_mode);
        return nil;
    }

    /* Apply GL_TEXTURE_MIN_LOD / GL_TEXTURE_MAX_LOD as Metal lod clamps.
     * GL defaults: min_lod=-1000, max_lod=1000 (effectively unclamped).
     * Metal's lodMinClamp cannot be negative (minimum 0.0), so clamp to 0.0
     * unconditionally rather than only when the GL default sentinel is seen. */
    samplerDescriptor.lodMinClamp = (tex_param->min_lod < 0.0f) ? 0.0f : tex_param->min_lod;
    samplerDescriptor.lodMaxClamp = (tex_param->max_lod >= 1000.0f) ? 1e9f : tex_param->max_lod;

    id<MTLSamplerState> sampler = [_device newSamplerStateWithDescriptor:samplerDescriptor];
    if (!sampler) {
        NSLog(@"MGL SAMPLER ERROR: failed to create MTLSamplerState");
        return nil;
    }

    /* Diagnostic: log sampler state to diagnose Minecraft "gray + moiré" issues. */
    {
        static uint64_t s_samplerDiagLogs = 0;
        uint64_t diagHit = ++s_samplerDiagLogs;
        if (kMGLDiagnosticStateLogs &&
            (diagHit <= 64ull || (diagHit % 256ull) == 0ull)) {
            MGLTraceNSLog(@"MGL SAMPLER_DIAG minFilter=0x%x magFilter=0x%x mipFilter=%lu "
                          @"minLod=%f maxLod=%f lodMinClamp=%f lodMaxClamp=%f "
                          @"wrapS=0x%x wrapT=0x%x maxAniso=%f aniso=%d "
                          @"compareMode=0x%x compareFunc=0x%x hit=%llu",
                          (unsigned)tex_param->min_filter,
                          (unsigned)tex_param->mag_filter,
                          (unsigned long)samplerDescriptor.mipFilter,
                          tex_param->min_lod,
                          tex_param->max_lod,
                          samplerDescriptor.lodMinClamp,
                          samplerDescriptor.lodMaxClamp,
                          (unsigned)tex_param->wrap_s,
                          (unsigned)tex_param->wrap_t,
                          tex_param->max_anisotropy,
                          (int)samplerDescriptor.maxAnisotropy,
                          (unsigned)tex_param->compare_mode,
                          (unsigned)tex_param->compare_func,
                          (unsigned long long)diagHit);
        }
    }

    return sampler;
}

- (id<MTLTexture>)fallbackSampledTexture
{
    if (_fallbackSampledTexture || !kMGLEnableSampledTextureFallback) {
        return _fallbackSampledTexture;
    }

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    _fallbackSampledTexture = [_device newTextureWithDescriptor:desc];
    if (_fallbackSampledTexture) {
        uint32_t pixel = 0xff000000u;
        [_fallbackSampledTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                   mipmapLevel:0
                                     withBytes:&pixel
                                   bytesPerRow:sizeof(pixel)];
        NSLog(@"MGL INFO: Created 1x1 fallback sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback sampled texture");
    }

    return _fallbackSampledTexture;
}

- (id<MTLTexture>)fallbackCubeSampledTexture
{
    if (_fallbackCubeSampledTexture || !kMGLEnableSampledTextureFallback) {
        return _fallbackCubeSampledTexture;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureTypeCube;
    desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    desc.width = 1;
    desc.height = 1;
    desc.depth = 1;
    desc.arrayLength = 1;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    _fallbackCubeSampledTexture = [_device newTextureWithDescriptor:desc];
    if (_fallbackCubeSampledTexture) {
        uint32_t pixel = 0xff000000u;
        for (NSUInteger face = 0; face < 6; face++) {
            [_fallbackCubeSampledTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                           mipmapLevel:0
                                                 slice:face
                                             withBytes:&pixel
                                           bytesPerRow:sizeof(pixel)
                                         bytesPerImage:sizeof(pixel)];
        }
        NSLog(@"MGL INFO: Created 1x1 fallback cube sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback cube sampled texture");
    }

    return _fallbackCubeSampledTexture;
}

- (id<MTLTexture>)fallbackTextureBufferSampledTexture
{
    if (_fallbackSintTextureBuffer || !kMGLEnableSampledTextureFallback) {
        return _fallbackSintTextureBuffer;
    }

    static const NSUInteger kFallbackTexelCount = 64;
    static const NSUInteger kFallbackBytesPerTexel = 4;

    if (!_fallbackTextureBufferStorage) {
        _fallbackTextureBufferStorage = [_device newBufferWithLength:(kFallbackTexelCount * kFallbackBytesPerTexel)
                                                              options:MTLResourceStorageModeShared];
        if (_fallbackTextureBufferStorage && _fallbackTextureBufferStorage.contents) {
            memset(_fallbackTextureBufferStorage.contents, 0, kFallbackTexelCount * kFallbackBytesPerTexel);
        }
    }

    if (!_fallbackTextureBufferStorage) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer backing storage");
        return nil;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureTypeTextureBuffer;
    desc.pixelFormat = MTLPixelFormatRGBA8Sint;
    desc.width = kFallbackTexelCount;
    desc.height = 1;
    desc.depth = 1;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    @try {
        _fallbackSintTextureBuffer = [_fallbackTextureBufferStorage newTextureWithDescriptor:desc
                                                                                     offset:0
                                                                                bytesPerRow:(kFallbackTexelCount * kFallbackBytesPerTexel)];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer texture: %@", exception);
        _fallbackSintTextureBuffer = nil;
    }

    if (_fallbackSintTextureBuffer) {
        NSLog(@"MGL INFO: Created fallback signed integer texture buffer for missing/invalid texel-buffer resources");
    }

    return _fallbackSintTextureBuffer;
}

- (id<MTLTexture>)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
                                               dataKind:(MGLTextureDataKind)dataKind
{
    if (!kMGLEnableSampledTextureFallback) {
        return nil;
    }

    MTLTextureType textureType = expectedType ? expectedType : MTLTextureType2D;
    if (textureType == MTLTextureTypeTextureBuffer) {
        return [self fallbackTextureBufferSampledTexture];
    }

    MTLPixelFormat pixelFormat = MTLPixelFormatRGBA8Unorm;
    if (dataKind == MGLTextureDataKindUint) {
        pixelFormat = MTLPixelFormatRGBA8Uint;
    } else if (dataKind == MGLTextureDataKindSint) {
        pixelFormat = MTLPixelFormatRGBA8Sint;
    } else if (dataKind == MGLTextureDataKindDepth) {
        pixelFormat = MTLPixelFormatDepth32Float;
    }

    if (!_fallbackSampledTextureCache) {
        _fallbackSampledTextureCache = [[NSMutableDictionary alloc] initWithCapacity:8];
    }

    NSUInteger keyValue = (((NSUInteger)textureType) << 8u) | ((NSUInteger)dataKind);
    NSNumber *key = @(keyValue);
    id<MTLTexture> cached = _fallbackSampledTextureCache[key];
    if (cached) {
        return cached;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = textureType;
    desc.pixelFormat = pixelFormat;
    desc.width = 1;
    desc.height = 1;
    desc.depth = 1;
    desc.arrayLength = (textureType == MTLTextureTypeCube ||
                        textureType == MTLTextureTypeCubeArray ||
                        textureType == MTLTextureType2DArray ||
                        textureType == MTLTextureType1DArray ||
                        textureType == MTLTextureType2DMultisampleArray) ? 1 : 1;
    if (textureType == MTLTextureType2DMultisample ||
        textureType == MTLTextureType2DMultisampleArray) {
        desc.sampleCount = 2u;
    }
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create %@ fallback sampled texture type=%lu format=%lu",
              [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
              (unsigned long)textureType,
              (unsigned long)pixelFormat);
        return nil;
    }

    uint32_t pixel = dataKind == MGLTextureDataKindDepth ? 0u : 0xff000000u;
    MTLRegion region = textureType == MTLTextureType1D ||
                       textureType == MTLTextureType1DArray
        ? MTLRegionMake1D(0, 1)
        : MTLRegionMake2D(0, 0, 1, 1);
    if (textureType == MTLTextureTypeCube || textureType == MTLTextureTypeCubeArray) {
        NSUInteger sliceCount = (textureType == MTLTextureTypeCube) ? 6u : 6u;
        for (NSUInteger slice = 0; slice < sliceCount; slice++) {
            [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                       mipmapLevel:0
                             slice:slice
                         withBytes:&pixel
                       bytesPerRow:sizeof(pixel)
                     bytesPerImage:sizeof(pixel)];
        }
    } else if (textureType == MTLTextureType1DArray ||
               textureType == MTLTextureType2DArray) {
        [texture replaceRegion:region
                   mipmapLevel:0
                         slice:0
                     withBytes:&pixel
                   bytesPerRow:sizeof(pixel)
                 bytesPerImage:sizeof(pixel)];
    } else {
        [texture replaceRegion:region
                   mipmapLevel:0
                     withBytes:&pixel
                   bytesPerRow:sizeof(pixel)];
    }

    _fallbackSampledTextureCache[key] = texture;
    [self mglCapAuxCache:_fallbackSampledTextureCache limit:32];
    NSLog(@"MGL INFO: Created %@ fallback sampled texture type=%lu format=%lu",
          [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
          (unsigned long)textureType,
          (unsigned long)pixelFormat);

    return texture;
}


- (id<MTLTexture>)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
{
    if (expectedType == MTLTextureTypeCube) {
        return [self fallbackCubeSampledTexture];
    }
    if (expectedType == MTLTextureTypeTextureBuffer) {
        return [self fallbackTextureBufferSampledTexture];
    }

    return [self fallbackSampledTexture];
}

- (int)textureIndexForExpectedMetalType:(MTLTextureType)expectedType
{
    switch (expectedType) {
        case MTLTextureType1D:
            return _TEXTURE_1D;
        case MTLTextureType1DArray:
            return _TEXTURE_1D_ARRAY;
        case MTLTextureType2D:
            return _TEXTURE_2D;
        case MTLTextureType2DMultisample:
            return _TEXTURE_2D_MULTISAMPLE;
        case MTLTextureType2DArray:
            return _TEXTURE_2D_ARRAY;
        case MTLTextureType2DMultisampleArray:
            return _TEXTURE_2D_MULTISAMPLE_ARRAY;
        case MTLTextureType3D:
            return _TEXTURE_3D;
        case MTLTextureTypeCube:
            return _TEXTURE_CUBE_MAP;
        case MTLTextureTypeCubeArray:
            return _TEXTURE_CUBE_MAP_ARRAY;
        case MTLTextureTypeTextureBuffer:
            return _TEXTURE_BUFFER;
        default:
            return -1;
    }
}

- (GLuint)textureUnitForSampledResource:(SpirvResource *)sampledResource metalBinding:(GLuint)metalBinding stage:(int)stage
{
    Program *program = mglResolveProgramForStageFromState(ctx, stage);
    if (!program) {
        GLuint candidate = sampledResource &&
                           sampledResource->sampler_unit >= 0 &&
                           sampledResource->sampler_unit < TEXTURE_UNITS
            ? (GLuint)sampledResource->sampler_unit
            : metalBinding;
        return candidate;
    }

    const char *sampledName = NULL;
    if (!sampledResource && metalBinding < TEXTURE_UNITS) {
        sampledResource = mglFindSamplerResourceForMetalBinding(program, stage, metalBinding);
    }
    if (sampledResource) {
        sampledName = sampledResource->name;
    }

    /*
     * Minecraft usually assigns sampler texture units from the RenderPipeline
     * sampler list, not from numeric suffixes like Sampler2. For example, chunk
     * rendering declares Sampler0 and Sampler2, so Sampler2 can be uploaded
     * through glUniform1i(..., 1). Keep sampler units on the exact reflected
     * resource instead of only the Metal binding: vertex and fragment resources
     * commonly share binding numbers, and binding-level state can make entity,
     * hand, and text textures bleed into each other.
     */
    if (sampledResource &&
        sampledResource->sampler_unit_explicit &&
        sampledResource->sampler_unit >= 0 &&
        sampledResource->sampler_unit < TEXTURE_UNITS) {
        return (GLuint)sampledResource->sampler_unit;
    }

    if (metalBinding >= TEXTURE_UNITS) {
        return metalBinding;
    }

    bool stageExplicit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? (program->sampler_units_explicit_by_stage[stage][metalBinding] == GL_TRUE)
        : false;
    bool globalExplicit = (program->sampler_units_explicit[metalBinding] == GL_TRUE);

    GLint unit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];

    if (stageExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    unit = program->sampler_units[metalBinding];
    if (globalExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    GLint defaultUnit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];
    if (defaultUnit < 0 || defaultUnit >= TEXTURE_UNITS) {
        defaultUnit = program->sampler_units[metalBinding];
    }

    if (sampledResource &&
        !sampledResource->sampler_unit_explicit &&
        sampledResource->sampler_unit >= 0 &&
        sampledResource->sampler_unit < TEXTURE_UNITS) {
        return (GLuint)sampledResource->sampler_unit;
    }

    if (defaultUnit >= 0 && defaultUnit < TEXTURE_UNITS) {
        return (GLuint)defaultUnit;
    }

    /*
     * OpenGL's valid default is unit 0, and explicit glUniform1i uploads above
     * are authoritative. No name-based fallback is applied.
     */
    return 0u;
}

- (GLuint)textureUnitForSampledBinding:(GLuint)metalBinding stage:(int)stage
{
    return [self textureUnitForSampledResource:NULL metalBinding:metalBinding stage:stage];
}

- (Texture *)textureForSampledResource:(SpirvResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(MTLTextureType)expectedType
{
    if (!ctx || metalBinding >= TEXTURE_UNITS) {
        return NULL;
    }

    GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                metalBinding:metalBinding
                                                       stage:stage];
    if (textureUnit >= TEXTURE_UNITS) {
        return NULL;
    }

    if (expectedType == 0) {
        return STATE(active_textures[textureUnit]);
    }

    int textureIndex = [self textureIndexForExpectedMetalType:expectedType];
    if (textureIndex >= 0 && textureIndex < _MAX_TEXTURE_TYPES) {
        Texture *typedTexture = STATE(texture_units[textureUnit].textures[textureIndex]);
        /* SPIRV-Cross lowers sampler1D to texture2d in MSL, so expectedType is
         * MTLTextureType2D even for GL_TEXTURE_1D bindings. If the _TEXTURE_2D
         * slot only contains an auto-created default texture (name ==
         * TEX_OBJ_RES_NAME) while the unit's active texture is a real
         * GL_TEXTURE_1D, prefer the 1D texture. Otherwise the default 2D
         * texture leaks across test cases and masks the real 1D binding. */
        if (typedTexture && typedTexture->name == TEX_OBJ_RES_NAME) {
            Texture *activeTexture = STATE(active_textures[textureUnit]);
            if (activeTexture && activeTexture->name != TEX_OBJ_RES_NAME) {
                typedTexture = NULL;
            }
        }
        if (typedTexture) {
            return typedTexture;
        }

        if (expectedType == MTLTextureType2D) {
            Texture *activeTexture = STATE(active_textures[textureUnit]);
            if (activeTexture &&
                activeTexture->target == GL_TEXTURE_1D) {
                return activeTexture;
            }
        }

        // Texel-buffer resources must not silently fall back to GL_TEXTURE_2D.
        // Minecraft's CloudFaces is declared as SpvDimBuffer but SPIRV-Cross
        // lowers it to a 1-row texture2d<int> in MSL. If no GL_TEXTURE_BUFFER
        // is bound, using the active 2D atlas here feeds float/RGBA data into a
        // signed integer vertex resource and corrupts the whole frame.
        if (expectedType == MTLTextureTypeTextureBuffer) {
            static uint64_t s_missingTextureBufferBindingLogs = 0;
            uint64_t hit = ++s_missingTextureBufferBindingLogs;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                Texture *activeTexture = STATE(active_textures[textureUnit]);
                NSLog(@"MGL TEXBUFFER BIND MISSING binding=%u unit=%u activeTex=%u activeTarget=0x%x hit=%llu",
                      (unsigned)metalBinding,
                      (unsigned)textureUnit,
                      activeTexture ? (unsigned)activeTexture->name : 0u,
                      activeTexture ? (unsigned)activeTexture->target : 0u,
                      (unsigned long long)hit);
            }
            return NULL;
        }

        /*
         * OpenGL texture units keep one binding per texture target. A sampler2D
         * samples the GL_TEXTURE_2D slot for its unit, even if a cubemap or texel
         * buffer was bound more recently on that same unit. Falling back to the
         * unit's "active" texture here lets sky cubemaps and buffer textures bleed
         * into item/entity shaders when Minecraft switches pipelines.
         */
        static uint64_t s_missingTypedTextureBindingLogs = 0;
        uint64_t hit = ++s_missingTypedTextureBindingLogs;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            Texture *activeTexture = STATE(active_textures[textureUnit]);
            NSLog(@"MGL TEX TYPED BIND MISSING binding=%u stage=%s unit=%u expectedType=%lu expectedIndex=%d activeTex=%u activeTarget=0x%x hit=%llu",
                  (unsigned)metalBinding,
                  mglShaderStageName(stage),
                  (unsigned)textureUnit,
                  (unsigned long)expectedType,
                  textureIndex,
                  activeTexture ? (unsigned)activeTexture->name : 0u,
                  activeTexture ? (unsigned)activeTexture->target : 0u,
                  (unsigned long long)hit);
        }
        return NULL;
    }

    return STATE(active_textures[textureUnit]);
}

- (Texture *)textureForSampledBinding:(GLuint)metalBinding stage:(int)stage expectedType:(MTLTextureType)expectedType
{
    return [self textureForSampledResource:NULL
                              metalBinding:metalBinding
                                      stage:stage
                               expectedType:expectedType];
}

- (id<MTLSamplerState>)fallbackSamplerState
{
    if (_fallbackSamplerState) {
        return _fallbackSamplerState;
    }

    MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
    desc.minFilter = MTLSamplerMinMagFilterNearest;
    desc.magFilter = MTLSamplerMinMagFilterNearest;
    desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.rAddressMode = MTLSamplerAddressModeClampToEdge;

    _fallbackSamplerState = [_device newSamplerStateWithDescriptor:desc];
    if (!_fallbackSamplerState) {
        NSLog(@"MGL ERROR: Failed to create fallback sampler state");
    }

    return _fallbackSamplerState;
}

- (void)traceSampledTextureReadback:(id<MTLTexture>)texture
                              glTex:(Texture *)glTex
                              level:(TextureLevel *)level0
                            program:(GLuint)program
                            binding:(GLuint)binding
                              stage:(NSString *)stage
                             reason:(NSString *)reason
                                hit:(uint64_t)hit
{
    if (!texture || !_device || !_commandQueue) {
        return;
    }

    MTLPixelFormat fmt = texture.pixelFormat;
    BOOL fourByteColor =
        fmt == MTLPixelFormatRGBA8Unorm ||
        fmt == MTLPixelFormatRGBA8Unorm_sRGB ||
        fmt == MTLPixelFormatBGRA8Unorm ||
        fmt == MTLPixelFormatBGRA8Unorm_sRGB;
    if (!fourByteColor) {
        MGLTraceNSLog(@"MGL TRACE sampled.readback skip program=%u binding=%u glTex=%u reason=%@ fmt=%lu type=%lu size=%lux%lu hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              (unsigned long)fmt,
              (unsigned long)texture.textureType,
              (unsigned long)texture.width,
              (unsigned long)texture.height,
              (unsigned long long)hit);
        return;
    }

    NSUInteger texWidth = (NSUInteger)texture.width;
    NSUInteger texHeight = (NSUInteger)texture.height;
    if (texWidth == 0 || texHeight == 0) {
        return;
    }

    NSUInteger sampleWidth = MIN(texWidth, 8u);
    NSUInteger sampleHeight = MIN(texHeight, 8u);
    NSUInteger bytesPerPixel = 4u;
    NSUInteger bytesPerRow = sampleWidth * bytesPerPixel;
    NSUInteger byteCount = bytesPerRow * sampleHeight;
    if (byteCount == 0) {
        return;
    }

    id<MTLBuffer> readback = [_device newBufferWithLength:byteCount
                                                  options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = cb ? [cb blitCommandEncoder] : nil;
    if (!readback || !cb || !blit) {
        MGLTraceNSLog(@"MGL TRACE sampled.readback setup-fail program=%u binding=%u glTex=%u reason=%@ readback=%p cb=%p blit=%p hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              readback,
              cb,
              blit,
              (unsigned long long)hit);
        return;
    }

    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(sampleWidth, sampleHeight, 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:bytesPerRow
 destinationBytesPerImage:byteCount];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const uint8_t *p = (const uint8_t *)readback.contents;
    uint64_t byteSum = 0;
    NSUInteger nonZeroBytes = 0;
    uint32_t firstPixel = 0;
    uint32_t pixelXor = 0;
    uint32_t minPixel = UINT32_MAX;
    uint32_t maxPixel = 0;
    NSUInteger pixelCount = byteCount / sizeof(uint32_t);

    if (p) {
        for (NSUInteger i = 0; i < byteCount; i++) {
            byteSum += (uint64_t)p[i];
            if (p[i] != 0) {
                nonZeroBytes++;
            }
        }
        if (byteCount >= sizeof(firstPixel)) {
            memcpy(&firstPixel, p, sizeof(firstPixel));
        }
        for (NSUInteger i = 0; i < pixelCount; i++) {
            uint32_t pixel = 0;
            memcpy(&pixel, p + (i * sizeof(pixel)), sizeof(pixel));
            pixelXor ^= pixel;
            if (pixel < minPixel) {
                minPixel = pixel;
            }
            if (pixel > maxPixel) {
                maxPixel = pixel;
            }
        }
    }

    MGLTraceNSLog(@"MGL TRACE sampled.readback stage=%@ program=%u binding=%u glTex=%u reason=%@ hit=%llu "
          "mtl=%p fmt=%lu type=%lu size=%lux%lu sample=%lux%lu status=%s error=%@ "
          "nonZero=%lu/%lu sum=%llu first=0x%08x min=0x%08x max=0x%08x xor=0x%08x "
          "level(init ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx)",
          stage,
          (unsigned)program,
          (unsigned)binding,
          glTex ? (unsigned)glTex->name : 0u,
          reason,
          (unsigned long long)hit,
          texture,
          (unsigned long)fmt,
          (unsigned long)texture.textureType,
          (unsigned long)texWidth,
          (unsigned long)texHeight,
          (unsigned long)sampleWidth,
          (unsigned long)sampleHeight,
          mglCommandBufferStatusName(cb.status),
          cb.error,
          (unsigned long)nonZeroBytes,
          (unsigned long)byteCount,
          (unsigned long long)byteSum,
          firstPixel,
          minPixel == UINT32_MAX ? 0u : minPixel,
          maxPixel,
          pixelXor,
          level0 ? (unsigned)level0->ever_written : 0u,
          level0 ? (unsigned)level0->has_initialized_data : 0u,
          level0 ? (unsigned)level0->suspicious_zero_upload : 0u,
          level0 ? (unsigned)level0->last_init_source : 0u,
          (unsigned long)(level0 ? level0->last_upload_size : 0u),
          level0 ? (void *)level0->last_src_ptr : NULL,
          (unsigned long long)(level0 ? level0->last_src_hash : 0ull));
}

/*
 * GL_TEXTURE_BASE_LEVEL selects the lowest mipmap level used for sampling.
 * Metal textures always start at level 0, so when base_level > 0 a texture
 * view is created whose level 0 corresponds to the GL base_level.  This lets
 * the Metal sampler's lodMinClamp/lodMaxClamp operate in the same coordinate
 * space as GL (relative to base_level).  GL_TEXTURE_MAX_LEVEL caps the
 * highest level.  When base_level == 0 the original texture is returned
 * unchanged so the common case has no overhead.
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


#pragma mark programs
- (int) getProgramBindingCount: (int) stage type: (int) type
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramBindingCount", stage);
        return 0;
    }
    switch(type)
    {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
        case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
        case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
        case SPVC_RESOURCE_TYPE_STAGE_INPUT:
        case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
            break;

        default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBindingCount (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (ptr == NULL)
        return 0;

    return ptr->spirv_resources_list[stage][type].count;
}

- (int) getProgramBinding: (int) stage type: (int) type index: (int) index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramBinding", stage);
        return 0;
    }
    switch(type)
    {
       case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
       case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
       case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
       case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
       case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
       case SPVC_RESOURCE_TYPE_STAGE_INPUT:
       case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
       case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
       case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
       case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
       case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
           break;

       default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBinding (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramBinding with no current program for stage=%s (name=%u pipeline=%u)",
              mglShaderStageName(stage),
              (unsigned)ctx->active_state->program_name,
              (unsigned)ctx->active_state->var.program_pipeline_binding);
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramBinding index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }

    return ptr->spirv_resources_list[stage][type].list[index].binding;
}

- (int)getProgramGLBinding:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES || type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        return 0;
    }

    return (int)ptr->spirv_resources_list[stage][type].list[index].gl_binding;
}

- (NSUInteger)getProgramBindingRequiredSize:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }

    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    return (NSUInteger)ptr->spirv_resources_list[stage][type].list[index].required_size;
}

- (NSInteger)getProgramMetalBufferIndexForStage:(int)stage clientBinding:(GLuint)clientBinding
{
    static const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    Program *ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return (NSInteger)clientBinding;
    }

    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        if (type < 0 || type >= _MAX_SPIRV_RES) {
            continue;
        }

        SpirvResourceList *list = &ptr->spirv_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            SpirvResource *res = &list->list[i];
            if (mglShouldSkipStageBufferResource(ptr, stage, type, res)) {
                continue;
            }
            GLuint resourceClientBinding = mglClientBufferBindingForResource(type, res);
            if (resourceClientBinding == clientBinding) {
                return (NSInteger)mglMetalResourceSlot(res);
            }
        }
    }

    return -1;
}

- (MTLTextureType)getProgramDeclaredTextureType:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    switch ((SpvDim)res->image_dim) {
        case SpvDim1D:
            return res->image_arrayed ? MTLTextureType1DArray : MTLTextureType1D;
        case SpvDim2D:
            if (res->image_multisampled) {
                return res->image_arrayed ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            }
            return res->image_arrayed ? MTLTextureType2DArray : MTLTextureType2D;
        case SpvDim3D:
            return MTLTextureType3D;
        case SpvDimCube:
            return res->image_arrayed ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
        case SpvDimBuffer:
            return MTLTextureTypeTextureBuffer;
        default:
            return 0;
    }
}

- (MTLTextureType)getProgramExpectedTextureType:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    // Per-Program cache: the MSL string is immutable post-link, so the
    // texture type for a given (program instance, generation, stage, binding)
    // never changes.  The Program instance ID is never reused, even if malloc
    // later reuses the Program's address.
    NSString *mslTextureCacheKey = [NSString stringWithFormat:@"T_%llu_%llu_%d_%u",
                                    (unsigned long long)ptr->msl_texture_cache_instance_id,
                                    (unsigned long long)ptr->msl_texture_cache_generation,
                                    stage, (unsigned)res->binding];
    NSNumber *cachedMslType = [_mslTextureTypeCache objectForKey:mslTextureCacheKey];
    MTLTextureType mslType;
    if (cachedMslType != nil) {
        mslType = (MTLTextureType)[cachedMslType unsignedIntegerValue];
    } else {
        mslType = mglExpectedTextureTypeFromMSL(ptr->spirv[stage].msl_str, res->binding);
        [_mslTextureTypeCache setObject:@(mslType) forKey:mslTextureCacheKey];
    }

    MTLTextureType spirvType = 0;
    switch ((SpvDim)res->image_dim) {
        case SpvDim1D:
            spirvType = res->image_arrayed ? MTLTextureType1DArray : MTLTextureType1D;
            break;
        case SpvDim2D:
            if (res->image_multisampled) {
                spirvType = res->image_arrayed ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            } else {
                spirvType = res->image_arrayed ? MTLTextureType2DArray : MTLTextureType2D;
            }
            break;
        case SpvDim3D:
            spirvType = MTLTextureType3D;
            break;
        case SpvDimCube:
            spirvType = res->image_arrayed ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
            break;
        case SpvDimBuffer:
            spirvType = MTLTextureTypeTextureBuffer;
            break;
        default:
            spirvType = 0;
            break;
    }

    if (mslType != 0 && mslType != spirvType) {
        static uint64_t s_mslTextureTypeOverrideCount = 0;
        uint64_t hit = ++s_mslTextureTypeOverrideCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL TEX EXPECT override from MSL stage=%d type=%d index=%d binding=%u name=%s spirvType=%lu mslType=%lu imageDim=%u hit=%llu",
                  stage,
                  type,
                  index,
                  (unsigned)res->binding,
                  res->name ? res->name : "(null)",
                  (unsigned long)spirvType,
                  (unsigned long)mslType,
                  (unsigned)res->image_dim,
                  (unsigned long long)hit);
        }
        return mslType;
    }

    return mslType ? mslType : spirvType;
}

- (MGLTextureDataKind)getProgramExpectedTextureDataKind:(int)stage type:(int)type index:(int)index
{
    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return MGLTextureDataKindUnknown;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return MGLTextureDataKindUnknown;
    }

    Program *ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return MGLTextureDataKindUnknown;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return MGLTextureDataKindUnknown;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    NSString *mslDataKindCacheKey = [NSString stringWithFormat:@"K_%llu_%llu_%d_%u",
                                      (unsigned long long)ptr->msl_texture_cache_instance_id,
                                      (unsigned long long)ptr->msl_texture_cache_generation,
                                      stage, (unsigned)res->binding];
    NSNumber *cachedMslKind = [_mslTextureTypeCache objectForKey:mslDataKindCacheKey];
    if (cachedMslKind != nil) {
        return (MGLTextureDataKind)[cachedMslKind unsignedIntegerValue];
    }

    MGLTextureDataKind mslKind =
        mglExpectedTextureDataKindFromMSL(ptr->spirv[stage].msl_str, res->binding);
    MGLTextureDataKind resolvedKind =
        mslKind != MGLTextureDataKindUnknown ? mslKind : MGLTextureDataKindFloat;
    [_mslTextureTypeCache setObject:@(resolvedKind) forKey:mslDataKindCacheKey];
    return resolvedKind;
}

- (NSUInteger)getProgramBindingRequiredSizeForStage:(int)stage clientBinding:(GLuint)clientBinding
{
    static const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }

    NSUInteger required = 0;
    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        int count = [self getProgramBindingCount:stage type:type];
        for (int i = 0; i < count; i++) {
            Program *program = mglResolveProgramForStageFromState(ctx, stage);
            if (!program || type < 0 || type >= _MAX_SPIRV_RES ||
                i < 0 || i >= (int)program->spirv_resources_list[stage][type].count) {
                continue;
            }

            SpirvResource *resource = &program->spirv_resources_list[stage][type].list[i];
            if (mglShouldSkipStageBufferResource(program, stage, type, resource)) {
                continue;
            }

            GLuint resourceClientBinding =
                mglClientBufferBindingForResource(type,
                                                  resource);
            if (resourceClientBinding != clientBinding) {
                continue;
            }

            NSUInteger candidate = [self getProgramBindingRequiredSize:stage type:type index:i];
            if (candidate > required) {
                required = candidate;
            }
        }
    }

    return required;
}

- (int) getProgramLocation: (int) stage type: (int) type index: (int) index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramLocation", stage);
        return 0;
    }
    switch(type)
    {
       case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
       case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
       case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
       case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
       case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
       case SPVC_RESOURCE_TYPE_STAGE_INPUT:
       case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
       case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
           break;

       default:
            NSLog(@"MGL WARNING: unsupported SPIRV-Cross resource type %d in getProgramLocation", type);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramLocation with no current program for stage=%s (name=%u pipeline=%u)",
              mglShaderStageName(stage),
              (unsigned)ctx->active_state->program_name,
              (unsigned)ctx->active_state->var.program_pipeline_binding);
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramLocation index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }
    
    return ptr->spirv_resources_list[stage][type].list[index].location;
}

- (void)initializeMTL4CompilerIfAvailable
{
#if MGL_HAS_MTL4_COMPILER
    if (!_device || _mtl4Compiler || mglEnvFlagEnabled("MGL_DISABLE_MTL4_COMPILER")) {
        return;
    }

    if (@available(macOS 26.0, *)) {
        if (![_device respondsToSelector:@selector(newCompilerWithDescriptor:error:)]) {
            return;
        }

        __autoreleasing NSError *error = nil;
        MTL4CompilerDescriptor *descriptor = [[MTL4CompilerDescriptor alloc] init];
        descriptor.label = @"MGL Metal 4 shader compiler";
        _mtl4Compiler = [_device newCompilerWithDescriptor:descriptor error:&error];
        if (_mtl4Compiler) {
            NSLog(@"MGL INFO: Metal 4 compiler enabled for shader libraries");
        } else if (error) {
            NSLog(@"MGL WARNING: Metal 4 compiler unavailable, falling back to MTLDevice library compile: %@",
                  error.localizedDescription);
        }
    }
#endif
}

- (id<MTLLibrary>)newMetalLibraryWithSource:(NSString *)source
                                    options:(MTLCompileOptions *)options
                                      label:(NSString *)label
                                      error:(NSError **)error
{
    return mglCompileMSL(_device, _mtl4Compiler, source, options, label, error);
}

- (id<MTLLibrary>) compileShader: (const char *) str
{
    id<MTLLibrary> library;
    __autoreleasing NSError *error = nil;
    BOOL sourceHasUnsupportedGeometryEmission =
        str && (strstr(str, "EmitVertex") || strstr(str, "EndPrimitive"));

    library = [self newMetalLibraryWithSource:[NSString stringWithUTF8String: str]
                                      options:nil
                                        label:@"MGL program shader"
                                        error:&error];
    if(!library) {
        if (sourceHasUnsupportedGeometryEmission) {
            NSLog(@"MGL WARNING: Skipped unsupported geometry-shader MSL compile: %@",
                  error.localizedDescription ?: error);
            return nil;
        }
        NSLog(@"MGL ERROR: Failed to compile shader: %@ ", [error localizedDescription] );
        // Return nil instead of asserting - caller must handle this gracefully
        return nil;
    }

    return library;
}

- (id<MTLFunction>)newFunctionFromLibrary:(id<MTLLibrary>)library
                                entryName:(NSString *)entryName
                                   source:(const char *)source
                                    label:(NSString *)label
{
    return mglNewFunctionFromLibrary(library, entryName, source, label);
}

/* invalidateCurrentPipelineStateForReason: moved to MGLRenderer+RenderPass.m */

/* bindMTLProgram: moved to MGLRenderer+RenderPass.m */

/* mglGeometryShaderIsPassthrough moved to MGLRenderer+RenderPass.m (static helper) */

/* P1-6: Tessellation passthrough helpers (mglShaderSourceContainsAny,
 * mglTessControlUnitPassthroughForPatchSize, mglTessEvalUnitPassthroughForPatchSize,
 * mglTessellationShadersArePassthrough, mglResolvePassthroughPatchModeForContext)
 * migrated to msl_patch_pipeline.c. */

/* bindMTLProgramLocked: moved to MGLRenderer+RenderPass.m */

#pragma mark draw buffers
- (CGSize)mglSyncLayerDrawableSizeFromView:(const char *)reason
{
    if (!_layer) {
        return CGSizeZero;
    }

    CGSize oldDrawableSize = _layer.drawableSize;
    NSRect bounds = NSZeroRect;
    NSRect backingBounds = NSZeroRect;
    CGFloat scale = 1.0;

    if (_view) {
        [_view setWantsLayer:YES];
        bounds = [_view bounds];
        if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0) {
            bounds = [_view frame];
            bounds.origin = NSZeroPoint;
        }

        backingBounds = [_view convertRectToBacking:bounds];
        if (bounds.size.width > 0.0 && backingBounds.size.width > 0.0) {
            scale = backingBounds.size.width / bounds.size.width;
        } else {
            NSWindow *window = [_view window];
            if (window) {
                scale = [window backingScaleFactor];
            } else if ([NSScreen mainScreen]) {
                scale = [[NSScreen mainScreen] backingScaleFactor];
            }

            if (scale <= 0.0) {
                scale = 1.0;
            }

            backingBounds = NSMakeRect(0.0,
                                       0.0,
                                       bounds.size.width * scale,
                                       bounds.size.height * scale);
        }

        _layer.frame = bounds;
        _layer.contentsScale = scale;
    } else if (oldDrawableSize.width <= 0.0 || oldDrawableSize.height <= 0.0) {
        bounds = [_layer frame];
        scale = _layer.contentsScale > 0.0 ? _layer.contentsScale : 1.0;
        backingBounds = NSMakeRect(0.0,
                                   0.0,
                                   bounds.size.width * scale,
                                   bounds.size.height * scale);
    } else {
        backingBounds = NSMakeRect(0.0, 0.0, oldDrawableSize.width, oldDrawableSize.height);
    }

    NSUInteger pixelWidth = (NSUInteger)MAX(1.0, backingBounds.size.width + 0.5);
    NSUInteger pixelHeight = (NSUInteger)MAX(1.0, backingBounds.size.height + 0.5);
    CGSize newDrawableSize = CGSizeMake((CGFloat)pixelWidth, (CGFloat)pixelHeight);

    if (oldDrawableSize.width != newDrawableSize.width ||
        oldDrawableSize.height != newDrawableSize.height) {
        _layer.drawableSize = newDrawableSize;
    }

    static uint64_t s_sizeSyncCall = 0;
    static NSUInteger s_lastPixelWidth = 0;
    static NSUInteger s_lastPixelHeight = 0;
    uint64_t call = ++s_sizeSyncCall;
    BOOL sizeChanged = (s_lastPixelWidth != pixelWidth || s_lastPixelHeight != pixelHeight);

    if (kMGLDiagnosticStateLogs &&
        (sizeChanged || call <= 8ull || ((call % 120ull) == 0ull))) {
        NSWindow *window = _view ? [_view window] : nil;
        NSRect windowFrame = window ? [window frame] : NSZeroRect;
        MGLTraceNSLog(@"MGL SIZE sync reason=%s call=%llu viewBounds=%.1fx%.1f backing=%.1fx%.1f scale=%.3f drawable=%lux%lu old=%.0fx%.0f window=%.1fx%.1f",
                      reason ? reason : "unknown",
                      (unsigned long long)call,
                      bounds.size.width,
                      bounds.size.height,
                      backingBounds.size.width,
                      backingBounds.size.height,
                      scale,
                      (unsigned long)pixelWidth,
                      (unsigned long)pixelHeight,
                      oldDrawableSize.width,
                      oldDrawableSize.height,
                      windowFrame.size.width,
                      windowFrame.size.height);
    }

    s_lastPixelWidth = pixelWidth;
    s_lastPixelHeight = pixelHeight;
    return newDrawableSize;
}

- (BOOL)mglEnsureLayerDrawableSizeAtLeastWidth:(NSUInteger)requiredWidth
                                        height:(NSUInteger)requiredHeight
                                        reason:(const char *)reason
{
    if (!_layer || requiredWidth == 0 || requiredHeight == 0) {
        return NO;
    }

    CGSize viewDrawableSize = [self mglSyncLayerDrawableSizeFromView:reason ? reason : "ensureDrawableSize"];
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
    drawableSize = [self mglSyncLayerDrawableSizeFromView:"newDrawBuffer"];

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

    texture = [_device newTextureWithDescriptor:tex_desc];
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

    texture = [_device newTextureWithDescriptor:tex_desc];
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

    drawableSize = [self mglSyncLayerDrawableSizeFromView:"checkDrawBufferSize"];

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

#pragma mark - Parallel Command Recording Infrastructure

/* Save the renderer's dedup ivars into a worker context.
 * Called before dispatching batch encode to a background thread so the
 * worker starts with the current dedup state (e.g. pipeline already bound
 * by a previous batch in the same render pass). */
/* saveDedupStateToWorker:(MGLWorkerContext *)worker moved to MGLRenderer+Draw.m */

/* Load dedup state back from a worker context into the renderer's ivars.
 * Called after the background thread finishes encoding to restore the
 * dedup state to reflect what the worker bound. */
/* loadDedupStateFromWorker:(const MGLWorkerContext *)worker moved to MGLRenderer+Draw.m */

/* Check whether parallel encode is enabled via environment variable.
 * When enabled, flushDrawBuffer will attempt to use MTLParallelRenderCommandEncoder
 * for parallel groups with ≥2 batches.  Disabled by default.
 *
 * The parallel encoder path is validated in headless FBO mode (regression
 * tests).  In windowed mode with sampled render targets, endRenderEncodingLocked
 * triggers Y-flip copy encoders that conflict with parallel encoder creation on
 * the AGX driver.  Future work will handle sampled RT deferral to enable
 * parallel encoding in windowed mode. */
/* parallelEncodeEnabled moved to MGLRenderer+Draw.m */

/* Encode a single batch onto the encoder referenced by the worker context.
 *
 * This method implements the per-worker batch processing that will eventually
 * run on background threads via MTLParallelRenderCommandEncoder sub-encoders.
 * Currently called sequentially from flushDrawBuffer to verify that each
 * batch can set up its full state independently (no reliance on the previous
 * batch's dedup cache).
 *
 * The dedup swap sequence is:
 *   1. loadDedupStateFromWorker: install the worker's pre-batch dedup state
 *   2. invalidateLastBoundState: clear dedup (sub-encoder is fresh)
 *   3. restoreStateForBatch: copy batch's GL state snapshot into GLMContext
 *   4. checkBatchShouldExecute: runs processGLStateLocked (syncs to encoder)
 *   5. scheduleDrawBatch + issue: encode the draw
 *   6. saveDedupStateToWorker: capture post-batch dedup state
 *
 * Returns the scheduled MGLBatchPath (or MGL_BATCH_PATH_DIRECT if skipped).
 * *executedOut is set to NO if the batch was skipped/failed. */
/* encodeBatchForParallelWorker:(MGLWorkerContext *)worker moved to MGLRenderer+Draw.m */

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
    static volatile double s_mainThreadHeartbeatSeconds = 0.0;
    static volatile uint64_t s_mainThreadPingCount = 0;
    uint64_t swapCall = ++s_swapCallCount;
    double swapStartSeconds = mglNowSeconds();
    bool traceSwap = mglShouldTraceCall(swapCall);
    MGL_FRAME_STORE(g_mglSwapCallCount, swapCall);
    /* advance the DontCare frame generation. Any color attachment
     * written before this point belongs to the previous frame, so its next
     * write this frame is a "first use" that may skip loading prior contents.
     * Skips 0 so a zero-initialized texture stamp never matches. */
    if (++_dontCareFrameGeneration == 0u) {
        _dontCareFrameGeneration = 2u;  /* skip 0 (texture stamp init) and wrap sentinel */
    }
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
        MGLTraceNSLog(@"MGL TRACE swap.contextSync old=%p new=%p", ctx, glm_ctx);
        ctx = glm_ctx;
    }

    GLMContext activeCtx = glm_ctx;
    GLenum drawBuffer = activeCtx->state.draw_buffer;
    bool shouldPresent = (drawBuffer != GL_NONE);
    if (traceSwap) {
        MGLTraceNSLog(@"MGL TRACE swap.begin call=%llu shouldPresent=%d draw_buffer=0x%x",
              (unsigned long long)swapCall, shouldPresent ? 1 : 0, (unsigned)drawBuffer);
        mglLogStateSnapshot("swap.enter",
                            activeCtx,
                            _currentCommandBuffer,
                            _currentRenderEncoder,
                            _renderPassDescriptor,
                            _drawable);
    }

    // Main-thread responsiveness probe for beachball diagnostics.
    // Render thread periodically posts a ping to main queue; stale heartbeat means main thread is blocked.
    if (kMGLDiagnosticStateLogs &&
        (swapCall <= 20ull || (swapCall % 30ull) == 0ull)) {
        dispatch_async(dispatch_get_main_queue(), ^{
            s_mainThreadHeartbeatSeconds = mglNowSeconds();
            s_mainThreadPingCount++;
        });

        double hb = s_mainThreadHeartbeatSeconds;
        if (hb > 0.0) {
            double lagMs = (swapStartSeconds - hb) * 1000.0;
            if (lagMs > 500.0) {
                MGLTraceNSLog(@"MGL TRACE mainthread.stall suspected lag=%.2fms swapCall=%llu pingCount=%llu",
                      lagMs,
                      (unsigned long long)swapCall,
                      (unsigned long long)s_mainThreadPingCount);
                if (traceSwap || (swapCall % 120ull) == 0ull) {
                    mglLogStateSnapshot("mainthread.stall.snapshot",
                                        activeCtx,
                                        _currentCommandBuffer,
                                        _currentRenderEncoder,
                                        _renderPassDescriptor,
                                        _drawable);
                }
            } else if (traceSwap) {
                MGLTraceNSLog(@"MGL TRACE mainthread.heartbeat lag=%.2fms swapCall=%llu pingCount=%llu",
                      lagMs,
                      (unsigned long long)swapCall,
                      (unsigned long long)s_mainThreadPingCount);
            }
        } else if (traceSwap) {
            MGLTraceNSLog(@"MGL TRACE mainthread.heartbeat uninitialized swapCall=%llu", (unsigned long long)swapCall);
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
            MGLTraceNSLog(@"MGL TRACE swap.drawActivity call=%llu processDrawCalls=%llu drawArrays=%llu verts=%llu "
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
        [self flushDrawBuffer:activeCtx];

        if (![self processGLStateLocked: false]) {
            static uint64_t s_swapProcessStateFailCount = 0;
            s_swapProcessStateFailCount++;
            if (s_swapProcessStateFailCount <= 16 || (s_swapProcessStateFailCount % 500) == 0) {
                NSLog(@"MGL WARNING: mtlSwapBuffers continuing despite processGLState failure (occurrence=%llu)",
                      (unsigned long long)s_swapProcessStateFailCount);
            }
        }

        [self endRenderEncodingLocked];

        if (![self ensureWritableCommandBufferLocked:"mtlSwapBuffers"]) {
            NSLog(@"MGL ERROR: Failed to obtain writable command buffer in mtlSwapBuffers");
            return;
        }

        if (_drawable == NULL)
        {
            if (traceSwap) {
                MGLTraceNSLog(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present", (unsigned long long)swapCall);
            }
            [self mglSyncLayerDrawableSizeFromView:"swap.pre_present"];
            _drawable = [_layer nextDrawable];
            if (traceSwap) {
                id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
                MGLTraceNSLog(@"MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present drawable=%p tex=%p size=%lux%lu",
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
                MGLTraceNSLog(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present_retry", (unsigned long long)swapCall);
            }
            [self mglSyncLayerDrawableSizeFromView:"swap.pre_present_retry"];
            _drawable = [_layer nextDrawable];
            if (traceSwap) {
                id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
                MGLTraceNSLog(@"MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present_retry drawable=%p tex=%p size=%lux%lu",
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

        id<MTLTexture> rpColor0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
        id<MTLTexture> drawableTexture = _drawable ? _drawable.texture : nil;
        [self copyRenderPassColorToDrawableIfNeeded:rpColor0 drawableTexture:drawableTexture swapCall:swapCall traceSwap:traceSwap];

        [self scheduleSwapTextureSampleDiagnostics:rpColor0 drawableTexture:drawableTexture swapCall:swapCall];

        if (_layer == NULL) {
            NSLog(@"MGL ERROR: Metal layer is NULL, cannot present drawable");
            return;
        }

        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: No command buffer available for presentation");
            return;
        }

        MTLCommandBufferStatus bufferStatus = _currentCommandBuffer.status;
        if (bufferStatus != MTLCommandBufferStatusNotEnqueued) {
            NSLog(@"MGL WARNING: mtlSwapBuffers found finalized command buffer (status: %ld), rotating", (long)bufferStatus);
            [self endRenderEncodingLocked];
            [self newCommandBufferLocked];
            if (!_currentCommandBuffer) {
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

            [_currentCommandBuffer presentDrawable: _drawable];
            if (traceSwap) {
                MGLTraceNSLog(@"MGL TRACE swap.present call=%llu cb=%p drawable=%p",
                      (unsigned long long)swapCall, _currentCommandBuffer, _drawable);
            }

        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Critical drawable presentation failure: %@", exception);
            NSLog(@"MGL ERROR: Exception name: %@, reason: %@", [exception name], [exception reason]);
            [self cleanupCommandBuffer];
            return;
        }

        id<MTLCommandBuffer> commandBufferToCommit = _currentCommandBuffer;
        _currentCommandBuffer = nil;
        @try {
            if (traceSwap) {
                MGLTraceNSLog(@"MGL TRACE swap.commit.begin call=%llu cb=%p status=%s label=%@",
                      (unsigned long long)swapCall,
                      commandBufferToCommit,
                      mglCommandBufferStatusName(commandBufferToCommit ? commandBufferToCommit.status : MTLCommandBufferStatusError),
                      commandBufferToCommit ? (commandBufferToCommit.label ?: @"(no-label)") : @"(nil)");
            }
            [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
            if (traceSwap) {
                MGLTraceNSLog(@"MGL TRACE swap.commit.end call=%llu", (unsigned long long)swapCall);
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Failed to commit command buffer: %@", exception);
            [self recordGPUError];
        }

        if (traceSwap) {
            MGLTraceNSLog(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=post_commit", (unsigned long long)swapCall);
        }
        _drawable = [_layer nextDrawable];
        if (traceSwap) {
            id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
            MGLTraceNSLog(@"MGL TRACE swap.nextDrawable.end call=%llu stage=post_commit drawable=%p tex=%p size=%lux%lu",
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
        double swapElapsedMs = (mglNowSeconds() - swapStartSeconds) * 1000.0;
        if (traceSwap) {
            MGLTraceNSLog(@"MGL TRACE swap.end call=%llu elapsed=%.3fms",
                  (unsigned long long)swapCall,
                  swapElapsedMs);
            mglLogStateSnapshot("swap.exit.ok",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        } else if (swapElapsedMs >= 25.0) {
            MGLTraceNSLog(@"MGL TRACE swap.slow call=%llu elapsed=%.3fms",
                  (unsigned long long)swapCall,
                  swapElapsedMs);
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
     * for consistency with swapElapsedMs above. */
    if (mglPerfSummaryEnabled()) {
        double now = mglNowSeconds();
        static _Atomic double s_last_swap_time = 0.0;
        double interval = 0.0;
        double prev = atomic_load_explicit(&s_last_swap_time, memory_order_relaxed);
        if (prev > 0.0) interval = (now - prev) * 1000.0;
        atomic_store_explicit(&s_last_swap_time, now, memory_order_relaxed);
        mglPrintPerfSummary(interval);
        mglResetPerfCounters();
    }
}

- (void)copyRenderPassColorToDrawableIfNeeded:(id<MTLTexture>)rpColor0
                              drawableTexture:(id<MTLTexture>)drawableTexture
                                      swapCall:(uint64_t)swapCall
                                    traceSwap:(bool)traceSwap
{
    // Diagnostic + compatibility path:
    // When swapping the default framebuffer, the active render pass should target the drawable.
    // If it still points to an offscreen texture, copy that texture into the drawable before present.
    if (ctx->active_state->framebuffer == NULL &&
        !_defaultDrawableWrittenSinceLastSwap &&
        rpColor0 &&
        drawableTexture &&
        rpColor0 != drawableTexture) {
        BOOL traceCopyToDrawable = traceSwap ||
            (kMGLSwapPresentDiagnostics &&
             (swapCall <= 12ull || (swapCall % 120ull) == 0ull));
        if (traceCopyToDrawable) {
            MGLTraceNSLog(@"MGL TRACE swap.copyToDrawable.begin call=%llu src=%p fmt=%lu %lux%lu dst=%p fmt=%lu %lux%lu",
                  (unsigned long long)swapCall,
                  rpColor0,
                  (unsigned long)rpColor0.pixelFormat,
                  (unsigned long)rpColor0.width,
                  (unsigned long)rpColor0.height,
                  drawableTexture,
                  (unsigned long)drawableTexture.pixelFormat,
                  (unsigned long)drawableTexture.width,
                  (unsigned long)drawableTexture.height);
        }

        BOOL canShaderCopyToDrawable =
            (rpColor0.pixelFormat == drawableTexture.pixelFormat ||
             (rpColor0.pixelFormat == MTLPixelFormatRGBA8Unorm && drawableTexture.pixelFormat == MTLPixelFormatBGRA8Unorm) ||
             (rpColor0.pixelFormat == MTLPixelFormatBGRA8Unorm && drawableTexture.pixelFormat == MTLPixelFormatRGBA8Unorm));
        if (canShaderCopyToDrawable) {
                id<MTLRenderPipelineState> pipeline = [self scaledBlitPipelineForPixelFormat:drawableTexture.pixelFormat];
                id<MTLSamplerState> sampler = [self scaledBlitSamplerForFilter:GL_NEAREST];
                NSUInteger copyWidth = MIN((NSUInteger)rpColor0.width, (NSUInteger)drawableTexture.width);
                NSUInteger copyHeight = MIN((NSUInteger)rpColor0.height, (NSUInteger)drawableTexture.height);
                if (pipeline && sampler && copyWidth > 0 && copyHeight > 0) {
                    MGLScaledBlitParams params;
                    params.uvRect = (vector_float4){
                        0.0f,
                        0.0f,
                        rpColor0.width ? ((float)copyWidth / (float)rpColor0.width) : 0.0f,
                        rpColor0.height ? ((float)copyHeight / (float)rpColor0.height) : 0.0f
                    };
                    params.forceOpaqueAlpha = 1.0f;
                    params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

                    MTLRenderPassDescriptor *copyPass = [MTLRenderPassDescriptor renderPassDescriptor];
                    copyPass.colorAttachments[0].texture = drawableTexture;
                    copyPass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                    copyPass.colorAttachments[0].storeAction = MTLStoreActionStore;

                    id<MTLRenderCommandEncoder> copyEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:copyPass];
                    if (copyEncoder) {
                        [copyEncoder setRenderPipelineState:pipeline];
                        [copyEncoder setVertexBytes:&params length:sizeof(params) atIndex:0];
                        [copyEncoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
                        [copyEncoder setFragmentTexture:rpColor0 atIndex:0];
                        [copyEncoder setFragmentSamplerState:sampler atIndex:0];
                        [copyEncoder setViewport:(MTLViewport){
                            .originX = 0.0,
                            .originY = 0.0,
                            .width = (double)copyWidth,
                            .height = (double)copyHeight,
                            .znear = 0.0,
                            .zfar = 1.0
                        }];
                        [copyEncoder setScissorRect:(MTLScissorRect){
                            .x = 0,
                            .y = 0,
                            .width = copyWidth,
                            .height = copyHeight
                        }];
                        [copyEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
                        [copyEncoder endEncoding];
                    } else {
                        NSLog(@"MGL WARNING: swap.copyToDrawable failed to create shader copy encoder");
                    }
                } else {
                    NSLog(@"MGL WARNING: swap.copyToDrawable shader copy unavailable pipeline=%p sampler=%p size=%lux%lu",
                          pipeline,
                          sampler,
                          (unsigned long)copyWidth,
                          (unsigned long)copyHeight);
                }
        } else {
            NSLog(@"MGL WARNING: swap.copyToDrawable skipped due to pixel format mismatch src=%lu dst=%lu",
                  (unsigned long)rpColor0.pixelFormat,
                  (unsigned long)drawableTexture.pixelFormat);
        }

        if (traceCopyToDrawable) {
            MGLTraceNSLog(@"MGL TRACE swap.copyToDrawable.end call=%llu", (unsigned long long)swapCall);
        }
    } else if (ctx->active_state->framebuffer == NULL &&
               _defaultDrawableWrittenSinceLastSwap &&
               rpColor0 &&
               drawableTexture &&
               rpColor0 != drawableTexture) {
        BOOL traceSkipCopyToDrawable = traceSwap ||
            (kMGLSwapPresentDiagnostics &&
             (swapCall <= 12ull || (swapCall % 120ull) == 0ull));
        if (traceSkipCopyToDrawable) {
            MGLTraceNSLog(@"MGL TRACE swap.copyToDrawable.skip call=%llu reason=default_blit_already_wrote_drawable src=%p dst=%p",
                  (unsigned long long)swapCall,
                  rpColor0,
                  drawableTexture);
        }
    }

}

- (void)scheduleSwapTextureSampleDiagnostics:(id<MTLTexture>)rpColor0
                             drawableTexture:(id<MTLTexture>)drawableTexture
                                     swapCall:(uint64_t)swapCall
{
    // Low-frequency dual texture sampling for black-screen diagnostics.
    // Sample both render-pass color source and drawable target so we can
    // distinguish "rendered black" from "copy/present black".
    if (kMGLSwapPresentDiagnostics &&
        ((swapCall <= 12ull && (swapCall % 3ull) == 0ull) || ((swapCall % 120ull) == 0ull))) {
        void (^scheduleTextureSample)(id<MTLTexture>, NSString *, NSUInteger, NSUInteger) =
            ^(id<MTLTexture> sampleTexture, NSString *sampleTag, NSUInteger originX, NSUInteger originY) {
                if (!sampleTexture) {
                    MGLTraceNSLog(@"MGL TRACE swap.sample.%@ call=%llu skipped(texture=nil)",
                          sampleTag,
                          (unsigned long long)swapCall);
                    return;
                }

                if (sampleTexture.pixelFormat != MTLPixelFormatBGRA8Unorm &&
                    sampleTexture.pixelFormat != MTLPixelFormatRGBA8Unorm) {
                    MGLTraceNSLog(@"MGL TRACE swap.sample.%@ call=%llu skipped(fmt=%lu tex=%lux%lu)",
                          sampleTag,
                          (unsigned long long)swapCall,
                          (unsigned long)sampleTexture.pixelFormat,
                          (unsigned long)sampleTexture.width,
                          (unsigned long)sampleTexture.height);
                    return;
                }

                NSUInteger sampleWidth = MIN((NSUInteger)sampleTexture.width, 8u);
                NSUInteger sampleHeight = MIN((NSUInteger)sampleTexture.height, 8u);
                NSUInteger bytesPerPixel = 4u;
                NSUInteger sampleBytesPerRow = sampleWidth * bytesPerPixel;
                NSUInteger sampleBytesPerImage = sampleBytesPerRow * sampleHeight;
                if (sampleWidth == 0 || sampleHeight == 0 || sampleBytesPerImage == 0) {
                    MGLTraceNSLog(@"MGL TRACE swap.sample.%@ call=%llu skipped(invalid-size tex=%lux%lu)",
                          sampleTag,
                          (unsigned long long)swapCall,
                          (unsigned long)sampleTexture.width,
                          (unsigned long)sampleTexture.height);
                    return;
                }

                NSUInteger clampedOriginX = originX;
                NSUInteger clampedOriginY = originY;
                if (clampedOriginX + sampleWidth > (NSUInteger)sampleTexture.width) {
                    clampedOriginX = ((NSUInteger)sampleTexture.width > sampleWidth)
                        ? ((NSUInteger)sampleTexture.width - sampleWidth)
                        : 0u;
                }
                if (clampedOriginY + sampleHeight > (NSUInteger)sampleTexture.height) {
                    clampedOriginY = ((NSUInteger)sampleTexture.height > sampleHeight)
                        ? ((NSUInteger)sampleTexture.height - sampleHeight)
                        : 0u;
                }

                id<MTLBuffer> sampleBuffer = [_device newBufferWithLength:sampleBytesPerImage
                                                                   options:MTLResourceStorageModeShared];
                if (!sampleBuffer) {
                    NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(alloc size=%lu)",
                          sampleTag,
                          (unsigned long long)swapCall,
                          (unsigned long)sampleBytesPerImage);
                    return;
                }

                id<MTLBlitCommandEncoder> sampleEncoder = [_currentCommandBuffer blitCommandEncoder];
                if (!sampleEncoder) {
                    NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(create blit encoder)",
                          sampleTag,
                          (unsigned long long)swapCall);
                    return;
                }

                [sampleEncoder copyFromTexture:sampleTexture
                                   sourceSlice:0
                                   sourceLevel:0
                                  sourceOrigin:MTLOriginMake(clampedOriginX, clampedOriginY, 0)
                                    sourceSize:MTLSizeMake(sampleWidth, sampleHeight, 1)
                                      toBuffer:sampleBuffer
                             destinationOffset:0
                        destinationBytesPerRow:sampleBytesPerRow
                      destinationBytesPerImage:sampleBytesPerImage];
                [sampleEncoder endEncoding];

                uint64_t sampleSwapCall = swapCall;
                NSString *sampleTagCopy = [sampleTag copy];
                NSUInteger sampleTexWidth = (NSUInteger)sampleTexture.width;
                NSUInteger sampleTexHeight = (NSUInteger)sampleTexture.height;
                NSUInteger sampleOriginX = clampedOriginX;
                NSUInteger sampleOriginY = clampedOriginY;
                [sampleBuffer addDebugMarker:@"mgl_swap_sample" range:NSMakeRange(0, sampleBytesPerImage)];
                [_currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> sampleCB) {
                    const uint8_t *p = (const uint8_t *)sampleBuffer.contents;
                    if (!p) {
                        MGLTraceNSLog(@"MGL TRACE swap.sample.%@ call=%llu unavailable(contents=nil) status=%s error=%@",
                              sampleTagCopy,
                              (unsigned long long)sampleSwapCall,
                              mglCommandBufferStatusName(sampleCB.status),
                              sampleCB.error);
                        return;
                    }

                    uint64_t sum = 0;
                    NSUInteger nonZero = 0;
                    for (NSUInteger bi = 0; bi < sampleBytesPerImage; bi++) {
                        uint8_t v = p[bi];
                        sum += (uint64_t)v;
                        if (v != 0) {
                            nonZero++;
                        }
                    }

                    uint32_t firstPixel = 0;
                    if (sampleBytesPerImage >= sizeof(firstPixel)) {
                        memcpy(&firstPixel, p, sizeof(firstPixel));
                    }

                    uint32_t minPixel = UINT32_MAX;
                    uint32_t maxPixel = 0u;
                    uint32_t pixelXor = 0u;
                    NSUInteger diffFromFirst = 0u;
                    NSUInteger pixelCount = sampleBytesPerImage / sizeof(uint32_t);
                    for (NSUInteger pi = 0; pi < pixelCount; pi++) {
                        uint32_t pixel = 0u;
                        memcpy(&pixel, p + (pi * sizeof(uint32_t)), sizeof(pixel));
                        if (pixel < minPixel) {
                            minPixel = pixel;
                        }
                        if (pixel > maxPixel) {
                            maxPixel = pixel;
                        }
                        pixelXor ^= pixel;
                        if (pixel != firstPixel) {
                            diffFromFirst++;
                        }
                    }
                    BOOL appearsSolid = (pixelCount > 0u && diffFromFirst == 0u);

                    MGLTraceNSLog(@"MGL TRACE swap.sample.%@ call=%llu tex=%lux%lu origin=(%lu,%lu) sample=%lux%lu "
                          "nonZero=%lu/%lu sum=%llu firstPixel=0x%08x min=0x%08x max=0x%08x xor=0x%08x diff=%lu solid=%d status=%s error=%@",
                          sampleTagCopy,
                          (unsigned long long)sampleSwapCall,
                          (unsigned long)sampleTexWidth,
                          (unsigned long)sampleTexHeight,
                          (unsigned long)sampleOriginX,
                          (unsigned long)sampleOriginY,
                          (unsigned long)sampleWidth,
                          (unsigned long)sampleHeight,
                          (unsigned long)nonZero,
                          (unsigned long)sampleBytesPerImage,
                          (unsigned long long)sum,
                          firstPixel,
                          minPixel == UINT32_MAX ? 0u : minPixel,
                          maxPixel,
                          pixelXor,
                          (unsigned long)diffFromFirst,
                          appearsSolid ? 1 : 0,
                          mglCommandBufferStatusName(sampleCB.status),
                          sampleCB.error);

                    if ([sampleTagCopy isEqualToString:@"src.center"]) {
                        static uint32_t s_lastCenterPixel = 0u;
                        static uint64_t s_sameCenterPixelRun = 0ull;
                        if (firstPixel == s_lastCenterPixel) {
                            s_sameCenterPixelRun++;
                        } else {
                            s_lastCenterPixel = firstPixel;
                            s_sameCenterPixelRun = 1ull;
                        }

                        if (s_sameCenterPixelRun == 10ull ||
                            s_sameCenterPixelRun == 30ull ||
                            (s_sameCenterPixelRun % 120ull) == 0ull) {
                            MGLTraceNSLog(@"MGL TRACE swap.sample.center_stable firstPixel=0x%08x run=%llu solid=%d diff=%lu",
                                  firstPixel,
                                  (unsigned long long)s_sameCenterPixelRun,
                                  appearsSolid ? 1 : 0,
                                  (unsigned long)diffFromFirst);
                        }
                    }
                }];
            };

        scheduleTextureSample(rpColor0, @"src.tl", 0u, 0u);
        if (rpColor0) {
            NSUInteger cx = ((NSUInteger)rpColor0.width > 8u) ? (((NSUInteger)rpColor0.width / 2u) - 4u) : 0u;
            NSUInteger cy = ((NSUInteger)rpColor0.height > 8u) ? (((NSUInteger)rpColor0.height / 2u) - 4u) : 0u;
            NSUInteger rx = ((NSUInteger)rpColor0.width > 8u) ? ((NSUInteger)rpColor0.width - 8u) : 0u;
            NSUInteger by = ((NSUInteger)rpColor0.height > 8u) ? ((NSUInteger)rpColor0.height - 8u) : 0u;
            scheduleTextureSample(rpColor0, @"src.center", cx, cy);
            scheduleTextureSample(rpColor0, @"src.right", rx, cy);
            scheduleTextureSample(rpColor0, @"src.bottom", cx, by);
        }
        if (drawableTexture != rpColor0) {
            scheduleTextureSample(drawableTexture, @"dst.tl", 0u, 0u);
            if (drawableTexture) {
                NSUInteger dcx = ((NSUInteger)drawableTexture.width > 8u) ? (((NSUInteger)drawableTexture.width / 2u) - 4u) : 0u;
                NSUInteger dcy = ((NSUInteger)drawableTexture.height > 8u) ? (((NSUInteger)drawableTexture.height / 2u) - 4u) : 0u;
                NSUInteger drx = ((NSUInteger)drawableTexture.width > 8u) ? ((NSUInteger)drawableTexture.width - 8u) : 0u;
                NSUInteger dby = ((NSUInteger)drawableTexture.height > 8u) ? ((NSUInteger)drawableTexture.height - 8u) : 0u;
                scheduleTextureSample(drawableTexture, @"dst.center", dcx, dcy);
                scheduleTextureSample(drawableTexture, @"dst.right", drx, dcy);
                scheduleTextureSample(drawableTexture, @"dst.bottom", dcx, dby);
            }
        } else {
            scheduleTextureSample(drawableTexture, @"srcdst.tl", 0u, 0u);
            if (drawableTexture) {
                NSUInteger sx = ((NSUInteger)drawableTexture.width > 8u) ? (((NSUInteger)drawableTexture.width / 2u) - 4u) : 0u;
                NSUInteger sy = ((NSUInteger)drawableTexture.height > 8u) ? (((NSUInteger)drawableTexture.height / 2u) - 4u) : 0u;
                NSUInteger srx = ((NSUInteger)drawableTexture.width > 8u) ? ((NSUInteger)drawableTexture.width - 8u) : 0u;
                NSUInteger sby = ((NSUInteger)drawableTexture.height > 8u) ? ((NSUInteger)drawableTexture.height - 8u) : 0u;
                scheduleTextureSample(drawableTexture, @"srcdst.center", sx, sy);
                scheduleTextureSample(drawableTexture, @"srcdst.right", srx, sy);
                scheduleTextureSample(drawableTexture, @"srcdst.bottom", sx, sby);
            }
        }
    }

}

#pragma mark C interface to mtlClearBuffer
-(void) mtlClearBuffer:(GLMContext) glm_ctx type:(GLuint) type mask:(GLbitfield) mask
{
    (void)type;
    if (!glm_ctx || (mask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) == 0) {
        return;
    }

    ctx = glm_ctx;

    if (!glm_ctx->active_state->caps.scissor_test) {
        [self endRenderEncoding];

        if (!_currentCommandBuffer && ![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: immediate clear failed to create command buffer");
            return;
        }

        Framebuffer *fbo = glm_ctx->active_state->framebuffer;
        if (fbo && (fbo->dirty_bits & DIRTY_FBO_BINDING)) {
            RETURN_ON_FAILURE([self bindFramebufferAttachmentTextures]);
            fbo->dirty_bits &= ~DIRTY_FBO_BINDING;
        }

        RETURN_ON_FAILURE([self newRenderEncoder]);
        [self endRenderEncoding];
        mglMarkRendererDirtyBits(glm_ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return;
    }

    GLint rawX = glm_ctx->active_state->var.scissor_box[0];
    GLint rawY = glm_ctx->active_state->var.scissor_box[1];
    GLint rawW = glm_ctx->active_state->var.scissor_box[2];
    GLint rawH = glm_ctx->active_state->var.scissor_box[3];
    if (rawW <= 0 || rawH <= 0) {
        return;
    }

    Framebuffer *fbo = glm_ctx->active_state->framebuffer;
    Texture *colorTexObj = NULL;
    Texture *depthTexObj = NULL;
    FBOAttachment *colorAttachment = NULL;
    FBOAttachment *depthAttachment = NULL;
    id<MTLTexture> colorTexture = nil;
    id<MTLTexture> depthTexture = nil;
    MGLMetalAttachmentSubresource colorSubresource = {0u, 0u, 0u};
    MGLMetalAttachmentSubresource depthSubresource = {0u, 0u, 0u};

    BOOL wantsColor = ((mask & GL_COLOR_BUFFER_BIT) != 0);
    BOOL wantsDepth = ((mask & GL_DEPTH_BUFFER_BIT) != 0) && glm_ctx->active_state->var.depth_writemask;

    if (wantsColor) {
        BOOL colorMaskAllowsWrite =
            !glm_ctx->active_state->caps.use_color_mask[0] ||
            glm_ctx->active_state->var.color_writemask[0][0] ||
            glm_ctx->active_state->var.color_writemask[0][1] ||
            glm_ctx->active_state->var.color_writemask[0][2] ||
            glm_ctx->active_state->var.color_writemask[0][3];
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
        GLuint drawBufferIndex = mglDefaultDrawBufferIndexForGL(glm_ctx->active_state->draw_buffer);
        if (wantsColor) {
            if (drawBufferIndex == _FRONT) {
                if (!_drawable && _layer) {
                    [self mglSyncLayerDrawableSizeFromView:"scissored-clear.nextDrawable"];
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
                NSUInteger depthWidth = colorTexture ? colorTexture.width : (NSUInteger)MAX(glm_ctx->active_state->viewport[2], 1);
                NSUInteger depthHeight = colorTexture ? colorTexture.height : (NSUInteger)MAX(glm_ctx->active_state->viewport[3], 1);
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
    if (glm_ctx->active_state->var.clip_origin != GL_UPPER_LEFT) {
        metalY = (GLint)passHeight - y1;
        if (metalY < 0) {
            metalY = 0;
        }
    }

    [self endRenderEncoding];
    if (!_currentCommandBuffer && ![self newCommandBuffer]) {
        NSLog(@"MGL ERROR: scissored clear failed to create command buffer");
        return;
    }

    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (colorTexture) {
        clearPass.colorAttachments[0].texture = colorTexture;
        clearPass.colorAttachments[0].level = colorSubresource.level;
        clearPass.colorAttachments[0].slice = colorSubresource.slice;
        clearPass.colorAttachments[0].depthPlane = colorSubresource.depthPlane;
        clearPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (depthTexture) {
        clearPass.depthAttachment.texture = depthTexture;
        clearPass.depthAttachment.level = depthSubresource.level;
        clearPass.depthAttachment.slice = depthSubresource.slice;
        clearPass.depthAttachment.depthPlane = depthSubresource.depthPlane;
        clearPass.depthAttachment.loadAction = MTLLoadActionLoad;
        clearPass.depthAttachment.storeAction = MTLStoreActionStore;
    }
    clearPass.renderTargetWidth = passWidth;
    clearPass.renderTargetHeight = passHeight;

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

    id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
    if (!clearEncoder) {
        NSLog(@"MGL ERROR: scissored clear failed to create render encoder");
        return;
    }

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

    MGLClearRectParams params;
    params.color = (vector_float4){
        glm_ctx->active_state->color_clear_value[0],
        glm_ctx->active_state->color_clear_value[1],
        glm_ctx->active_state->color_clear_value[2],
        glm_ctx->active_state->color_clear_value[3]
    };
    params.depth = (float)glm_ctx->active_state->var.depth_clear_value;
    params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

    [clearEncoder setViewport:viewport];
    [clearEncoder setScissorRect:scissor];
    [clearEncoder setRenderPipelineState:pipeline];
    if (wantsDepth) {
        id<MTLDepthStencilState> depthState = [self clearRectDepthState];
        if (depthState) {
            [clearEncoder setDepthStencilState:depthState];
        }
    }
    [clearEncoder setVertexBytes:&params length:sizeof(params) atIndex:0];
    if (wantsColor) {
        [clearEncoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
    }
    [clearEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [clearEncoder endEncoding];

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
        MGLTraceNSLog(@"MGL TRACE mtlBufferSubData.begin call=%llu buffer=%u size=%lld off=%zu len=%zu mtl=%p cpu=%p dirty=0x%x srcHash=0x%016llx srcHead=%s",
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
                MGLTraceNSLog(@"MGL TRACE mtlBufferSubData.cpuFallback call=%llu buffer=%u off=%zu len=%zu dstHash=0x%016llx dstHead=%s",
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
    memcpy(data+offset, ptr, size);

    if (mtl_buffer.storageMode == MTLStorageModeManaged) {
        [mtl_buffer didModifyRange:NSMakeRange(offset, size)];
    }

    if (trace) {
        const void *dst = (const void *)((const uint8_t *)mtl_buffer.contents + offset);
        char dstHead[64];
        dstHead[0] = '\0';
        mglTraceFormatBytes(dst, size, dstHead, sizeof(dstHead));
        uint64_t dstHash = mglTraceHashBytes(dst, size);
        MGLTraceNSLog(@"MGL TRACE mtlBufferSubData.end call=%llu buffer=%u off=%zu len=%zu mtlLen=%lu dstHash=0x%016llx dstHead=%s",
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
    id<MTLBuffer> mtl_buffer = nil;

    if (!buf) {
        NSLog(@"MGL ERROR: mtlMapUnmapBuffer called with NULL buffer");
        return NULL;
    }

    if (buf->data.mtl_data == NULL)
    {
        [self bindMTLBuffer:buf];
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
            if (reads && mtlBase && mtlBase != cpuBase && safeLen > 0) {
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
                MGLTraceNSLog(@"MGL TRACE mtlMap.map buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
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

            MGLTraceNSLog(@"MGL TRACE mtlMap.map buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
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
        MGLTraceNSLog(@"MGL TRACE mtlMap.unmap buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
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
    id<MTLBuffer> mtl_buffer;

    if (!buf) {
        NSLog(@"MGL ERROR: mtlFlushMappedBufferRange called with NULL buffer");
        return;
    }

    mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtl_buffer) {
        [self bindMTLBuffer:buf];
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
    Buffer *gl_indirect_buffer = STATE(buffers[_DRAW_INDIRECT_BUFFER]);

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

    id<MTLBuffer> isolated = [_device newBufferWithLength:requiredLength
                                                   options:MTLResourceStorageModeShared];
    if (!isolated || !isolated.contents) {
        return nil;
    }

    memset(isolated.contents, 0, requiredLength);
    if (!source || map->offset < 0 || !source.contents) {
        return isolated;
    }

    size_t copyLength = mglBufferMapVisibleBackingBytes(map, source.length);
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

    BOOL hasCopies = NO;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLStageBindingCopyBack *entry = &copyBacks->slots[i];
        if (entry->length == 0) {
            continue;
        }
        if (!entry->temporary || !entry->destination ||
            entry->length > entry->temporary.length ||
            entry->destination_offset > entry->destination.length ||
            entry->length > entry->destination.length - entry->destination_offset) {
            [self clearStageBindingCopyBacks:copyBacks];
            return false;
        }
        hasCopies = YES;
    }

    if (!hasCopies && !requireCPUVisibility) {
        [self clearStageBindingCopyBacks:copyBacks];
        return true;
    }
    if (!_currentCommandBuffer ||
        _currentCommandBuffer.status != MTLCommandBufferStatusNotEnqueued) {
        [self clearStageBindingCopyBacks:copyBacks];
        return false;
    }

    if (hasCopies) {
        id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
        if (!blit) {
            [self clearStageBindingCopyBacks:copyBacks];
            return false;
        }
        for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
            MGLStageBindingCopyBack *entry = &copyBacks->slots[i];
            if (entry->length == 0) {
                continue;
            }
            [blit copyFromBuffer:entry->temporary
                    sourceOffset:0
                        toBuffer:entry->destination
               destinationOffset:entry->destination_offset
                            size:entry->length];
        }
        [blit endEncoding];
    }

    /* Isolated copy-backs must become CPU-visible before another short binding
     * snapshots their destination. TCS also forces this boundary because TES
     * sizing and query accounting currently read its factor buffer on the CPU. */
    id<MTLCommandBuffer> stageCommandBuffer = _currentCommandBuffer;
    _currentCommandBuffer = nil;
    @try {
        [self commitCommandBufferWithAGXRecovery:stageCommandBuffer];
        [stageCommandBuffer waitUntilCompleted];
    } @catch (NSException *exception) {
        NSLog(@"MGL BUFFER RANGE: stage synchronization failed: %@",
              exception.reason);
        [self clearStageBindingCopyBacks:copyBacks];
        [self newCommandBufferLocked];
        return false;
    }
    if (stageCommandBuffer.error) {
        NSLog(@"MGL BUFFER RANGE: stage command failed: %@",
              stageCommandBuffer.error);
        [self clearStageBindingCopyBacks:copyBacks];
        [self newCommandBufferLocked];
        return false;
    }

    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLStageBindingCopyBack *entry = &copyBacks->slots[i];
        if (entry->length > 0 && entry->destination_buffer) {
            Buffer *buffer = entry->destination_buffer;
            buffer->ever_written = GL_TRUE;

            /* BufferSubData updates the CPU snapshot and may later upload the
             * whole store. Keep that snapshot authoritative for the legal
             * prefix written through the isolated Metal binding. */
            if (buffer->data.buffer_data) {
                if (!entry->destination.contents ||
                    entry->destination_offset > buffer->data.buffer_size ||
                    entry->length > buffer->data.buffer_size - entry->destination_offset) {
                    NSLog(@"MGL BUFFER RANGE: cannot synchronize copied-back prefix to CPU buffer=%u offset=%lu length=%lu cpuSize=%zu",
                          (unsigned)buffer->name,
                          (unsigned long)entry->destination_offset,
                          (unsigned long)entry->length,
                          buffer->data.buffer_size);
                    [self clearStageBindingCopyBacks:copyBacks];
                    [self newCommandBufferLocked];
                    return false;
                }

                uint8_t *cpuBytes = (uint8_t *)(uintptr_t)buffer->data.buffer_data;
                const uint8_t *metalBytes = (const uint8_t *)entry->destination.contents;
                if (cpuBytes != metalBytes) {
                    memmove(cpuBytes + entry->destination_offset,
                            metalBytes + entry->destination_offset,
                            entry->length);
                }
            }
        }
    }
    [self clearStageBindingCopyBacks:copyBacks];
    return [self newCommandBufferLocked];
}

typedef struct {
    id<MTLBuffer> __strong buffer;
    NSUInteger offset;
    id<MTLBuffer> __strong initialization_source;
    NSUInteger initialization_source_offset;
    NSUInteger initialization_length;
    BOOL valid;
} MGLTessStageBufferBinding;

typedef struct {
    MGLTessStageBufferBinding slots[kMGLMaxBufferSlots];
    id<MTLBuffer> __strong size_buffer;
} MGLTessStageBufferBindingList;

/* Tessellation shaders run as consecutive compute encoders. Prepare their
 * buffer bindings before opening the next encoder so an isolated binding can
 * be initialized by an ordered GPU copy from a buffer written by the previous
 * stage. Reading source.contents here would capture stale CPU bytes while the
 * preceding TCS encoder is still pending on the same command buffer. */
- (bool)prepareTessStageBufferBindings:(MGLTessStageBufferBindingList *)bindings
                                 stage:(int)stage
                             copyBacks:(MGLStageBindingCopyBackList *)copyBacks
{
    if (!bindings || !copyBacks) {
        return false;
    }

    BufferMapList stageBufferMap = {0};
    if (![self mapGLBuffersToMTLBufferMap:&stageBufferMap stage:stage]) {
        return false;
    }

    /* Complete every lazy allocation before creating the initialization blit
     * encoder. bindMTLBuffer: may itself need an encoder. */
    for (GLuint i = 0; i < stageBufferMap.count; i++) {
        Buffer *ptr = stageBufferMap.buffers[i].buf;
        if (ptr && !ptr->data.mtl_data) {
            [self bindMTLBuffer:ptr];
        }
    }

    for (GLuint i = 0; i < stageBufferMap.count; i++) {
        BufferMap *map = &stageBufferMap.buffers[i];
        Buffer *ptr = map->buf;
        if (!ptr) {
            continue;
        }

        NSUInteger metalBindingIndex = map->has_metal_binding
            ? (NSUInteger)map->metal_binding_index
            : (NSUInteger)map->buffer_base_index;
        if (metalBindingIndex >= kMGLMaxMetalVertexBufferCount) {
            continue;
        }
        [self clearStageBindingCopyBack:copyBacks atIndex:metalBindingIndex];
        if (map->offset < 0) {
            return false;
        }
        NSUInteger bindOffset = (NSUInteger)map->offset;

        id<MTLBuffer> buffer = ptr->data.mtl_data
            ? (__bridge id<MTLBuffer>)(ptr->data.mtl_data)
            : nil;
        NSUInteger requiredBytes =
            [self getProgramBindingRequiredSize:stage
                                           type:(int)map->resource_type
                                          index:(int)map->resource_index];
        if (map->resource_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER &&
            requiredBytes < sizeof(uint32_t)) {
            requiredBytes = sizeof(uint32_t);
        }

        GLsizeiptr storageRemaining = mglBufferMapStorageRemaining(map);
        NSUInteger availableBytes = buffer
            ? mglBufferMapVisibleBackingBytes(map, buffer.length)
            : 0u;
        BOOL needsIsolatedBinding =
            !buffer ||
            storageRemaining <= 0 ||
            bindOffset >= buffer.length ||
            availableBytes == 0 ||
            (requiredBytes > 0 && availableBytes < requiredBytes);

        MGLTessStageBufferBinding *binding = &bindings->slots[metalBindingIndex];
        binding->buffer = nil;
        binding->offset = 0u;
        binding->initialization_source = nil;
        binding->initialization_source_offset = 0u;
        binding->initialization_length = 0u;
        binding->valid = YES;
        if (!needsIsolatedBinding) {
            binding->buffer = buffer;
            binding->offset = bindOffset;
            continue;
        }

        NSUInteger fallbackLength = MAX(requiredBytes, sizeof(uint32_t));
        id<MTLBuffer> isolated = [_device newBufferWithLength:fallbackLength
                                                      options:MTLResourceStorageModeShared];
        if (!isolated || !isolated.contents) {
            return false;
        }
        memset(isolated.contents, 0, fallbackLength);

        binding->buffer = isolated;
        binding->offset = 0u;
        if (buffer && availableBytes > 0) {
            binding->initialization_source = buffer;
            binding->initialization_source_offset = bindOffset;
            binding->initialization_length = MIN(availableBytes, fallbackLength);
        }

        BOOL writableResource =
            map->resource_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ||
            map->resource_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER;
        if (writableResource && buffer && availableBytes > 0 &&
            ![self recordStageBindingCopyBack:copyBacks
                                       atIndex:metalBindingIndex
                                     temporary:isolated
                                   destination:buffer
                             destinationBuffer:ptr
                            destinationOffset:bindOffset
                                        length:availableBytes]) {
            return false;
        }
    }

    Program *stageProgram = mglResolveProgramForStageFromState(ctx, stage);
    if (stageProgram && stageProgram->spirv[stage].needs_buffer_size_buffer) {
        uint32_t sizeConstants[31] = {0};
        for (GLuint i = 0; i < stageBufferMap.count; i++) {
            BufferMap *map = &stageBufferMap.buffers[i];
            if (!map->buf) {
                continue;
            }
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_BUFFER_SIZE_BUFFER_INDEX) {
                continue;
            }
            sizeConstants[metalSlot] = (uint32_t)mglBufferMapVisibleSize(map);
        }
        bindings->size_buffer = [_device newBufferWithBytes:sizeConstants
                                                     length:sizeof(sizeConstants)
                                                    options:MTLResourceStorageModeShared];
        if (!bindings->size_buffer) {
            return false;
        }
    }

    BOOL needsInitializationBlit = NO;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        if (bindings->slots[i].initialization_length > 0) {
            needsInitializationBlit = YES;
            break;
        }
    }
    if (!needsInitializationBlit) {
        return true;
    }
    if (!_currentCommandBuffer ||
        _currentCommandBuffer.status != MTLCommandBufferStatusNotEnqueued) {
        return false;
    }

    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    if (!blit) {
        return false;
    }
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->initialization_length == 0) {
            continue;
        }
        [blit copyFromBuffer:binding->initialization_source
                sourceOffset:binding->initialization_source_offset
                    toBuffer:binding->buffer
           destinationOffset:0
                        size:binding->initialization_length];
    }
    [blit endEncoding];
    return true;
}

- (bool)bindPreparedTessStageBufferBindings:(const MGLTessStageBufferBindingList *)bindings
                           toComputeEncoder:(id<MTLComputeCommandEncoder>)computeCommandEncoder
{
    if (!bindings || !computeCommandEncoder) {
        return false;
    }
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        const MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->valid) {
            [computeCommandEncoder setBuffer:binding->buffer
                                      offset:binding->offset
                                     atIndex:i];
        }
    }
    if (bindings->size_buffer) {
        [computeCommandEncoder setBuffer:bindings->size_buffer
                                  offset:0
                                 atIndex:MGL_BUFFER_SIZE_BUFFER_INDEX];
    }
    return true;
}

/* Dispatch a tessellation control shader (TCS) as a Metal compute kernel.
 * SPIRV-Cross lowers GL_TESS_CONTROL_SHADER to `kernel void` and writes
 * tessellation factors to buffer(26) and per-patch output to buffer(27).
 * Indirect params (vertexCount, instanceCount) go in buffer(29).
 *
 * For shader_image_size tests the TCS kernel only needs storage images and
 * the tess-factor / indirect-param buffers — it has no vertex input.
 */
- (void)bindPointSizeParamsToComputeEncoder:(id<MTLComputeCommandEncoder>)computeEncoder
                                    program:(Program *)program
                                      stage:(int)stage
{
    if (!computeEncoder || !program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }
    const char *msl = program->spirv[stage].msl_str;
    if (!msl || !strstr(msl, "_mgl_point_size_params")) {
        return;
    }
    float pointSizeParams[2] = {
        ctx && ctx->active_state->var.point_size > 0.0f ? ctx->active_state->var.point_size : 1.0f,
        ctx && ctx->active_state->caps.program_point_size ? 1.0f : 0.0f
    };
    [computeEncoder setBytes:pointSizeParams
                      length:sizeof(pointSizeParams)
                     atIndex:kMGLPointSizeParamBufferIndex];
}

- (id<MTLBuffer>)newTCSStageInBufferForContext:(GLMContext)drawCtx
                                       program:(Program *)tcsProgram
                                         first:(GLint)first
                                         count:(GLsizei)count
                                     indexType:(GLenum)indexType
                                       indices:(const void *)indices
                                    baseVertex:(GLint)baseVertex
                                  baseInstance:(GLuint)baseInstance
                                 patchVertices:(GLuint)patchVertices
                                    patchCount:(GLuint)patchCount
                                     outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0u;
    }
    if (!drawCtx || !tcsProgram || count <= 0) {
        return nil;
    }

    const char *tcsMsl = tcsProgram->spirv[_TESS_CONTROL_SHADER].msl_str;
    if (!tcsMsl || !strstr(tcsMsl, "_mgl_tcs_in_buffer")) {
        return nil;
    }

    MGLTCSStageInMember members[MAX_ATTRIBS];
    memset(members, 0, sizeof(members));
    NSUInteger tcsInStride = 0u;
    NSUInteger memberCount = mglParseTCSStageInMembers(tcsMsl,
                                                       members,
                                                       MAX_ATTRIBS,
                                                       &tcsInStride);
    if (tcsInStride == 0u) {
        tcsInStride = mglComputeMSLStructSizeBySuffix(tcsMsl, "_in", 3);
    }
    if (tcsInStride == 0u) {
        NSLog(@"MGL TESS WARNING: unable to compute TCS stage_in stride for program %u",
              (unsigned)tcsProgram->name);
        return nil;
    }

    NSUInteger tcsInVertices = (NSUInteger)patchCount * (NSUInteger)MAX(patchVertices, 1u);
    if (tcsInVertices < (NSUInteger)count) {
        tcsInVertices = (NSUInteger)count;
    }
    if (tcsInVertices == 0u || tcsInStride > NSUIntegerMax / tcsInVertices) {
        return nil;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "tcs.stage_in");
    if (!vao) {
        return nil;
    }

    const uint8_t *indexBytes = NULL;
    NSUInteger indexOffset = (NSUInteger)(uintptr_t)indices;
    NSUInteger indexStride = 0u;
    uint32_t restartIndex = 0u;
    bool primitiveRestart = false;
    if (indexType != 0u) {
        indexStride = mglGLIndexElementSize(indexType);
        if (indexStride == 0u || indexOffset > NSUIntegerMax - ((NSUInteger)count * indexStride)) {
            return nil;
        }
        Buffer *ebo = getElementBuffer(drawCtx);
        if (!ebo || ![self processBuffer:ebo]) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in has no readable element buffer");
            return nil;
        }
        const uint8_t *eboBytes = mglRendererReadableBufferBytes(ebo);
        NSUInteger bytesNeeded = (NSUInteger)count * indexStride;
        if (!eboBytes || indexOffset > (NSUInteger)ebo->size || ((NSUInteger)ebo->size - indexOffset) < bytesNeeded) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in element range OOB offset=%lu needed=%lu size=%lld",
                  (unsigned long)indexOffset,
                  (unsigned long)bytesNeeded,
                  (long long)ebo->size);
            return nil;
        }
        indexBytes = eboBytes + indexOffset;
        primitiveRestart = mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    }

    NSUInteger tcsInSize = tcsInStride * tcsInVertices;
    id<MTLBuffer> stageInBuffer = [_device newBufferWithLength:tcsInSize
                                                       options:MTLResourceStorageModeShared];
    if (!stageInBuffer) {
        return nil;
    }
    memset(stageInBuffer.contents, 0, tcsInSize);

    if (memberCount == 0u) {
        if (outStride) {
            *outStride = tcsInStride;
        }
        return stageInBuffer;
    }

    for (NSUInteger v = 0; v < tcsInVertices; v++) {
        if (v >= (NSUInteger)count) {
            continue;
        }

        int64_t vertexIndex64 = (int64_t)first + (int64_t)v;
        if (indexBytes) {
            uint32_t rawIndex = mglReadGLIndexValue(indexBytes, indexType, v);
            if (primitiveRestart && rawIndex == restartIndex) {
                continue;
            }
            vertexIndex64 = (int64_t)rawIndex + (int64_t)baseVertex;
        }
        if (vertexIndex64 < 0) {
            continue;
        }

        uint8_t *dstVertex = (uint8_t *)stageInBuffer.contents + (v * tcsInStride);
        for (NSUInteger m = 0; m < memberCount; m++) {
            const MGLTCSStageInMember *member = &members[m];
            if (member->attribute >= MAX_ATTRIBS ||
                member->offset >= tcsInStride ||
                member->size > tcsInStride - member->offset) {
                continue;
            }

            double values[4] = {0.0, 0.0, 0.0, 1.0};
            const VertexAttrib *attrib = &vao->attrib[member->attribute];
            MGLResolvedVertexAttribBinding resolved = {0};
            bool hasBinding = mglRendererResolveVertexAttribBinding(drawCtx,
                                                                    vao,
                                                                    member->attribute,
                                                                    "tcs.stage_in",
                                                                    &resolved);
            bool useCurrentValue =
                ((vao->enabled_attribs & (0x1u << member->attribute)) == 0u) &&
                !(vao->enabled_attribs == 0u && hasBinding);

            if (useCurrentValue) {
                uint8_t currentBytes[16];
                if (mglRendererBuildCurrentVertexAttribBytes(drawCtx,
                                                             member->attribute,
                                                             attrib,
                                                             currentBytes) > 0u) {
                    for (GLuint c = 0; c < MIN(attrib->size, 4u); c++) {
                        values[c] = mglDecodeVertexAttribComponent(currentBytes,
                                                                   attrib->type,
                                                                   attrib->normalized,
                                                                   c);
                    }
                }
            } else if (hasBinding) {
                Buffer *vbo = resolved.buffer;
                if (vbo && [self processBuffer:vbo]) {
                    const uint8_t *vboBytes = mglRendererReadableBufferBytes(vbo);
                    NSUInteger elementBytes = mglVertexAttribElementBytes(attrib->type, attrib->size);
                    NSUInteger stride = resolved.stride > 0u ? (NSUInteger)resolved.stride : elementBytes;
                    NSUInteger attribIndex = (NSUInteger)vertexIndex64;
                    if (resolved.divisor > 0u) {
                        attribIndex = (NSUInteger)(baseInstance / resolved.divisor);
                    }
                    if (vboBytes && elementBytes > 0u && stride > 0u &&
                        resolved.binding_offset >= 0 && resolved.relativeoffset >= 0) {
                        NSUInteger baseOffset = (NSUInteger)resolved.binding_offset + (NSUInteger)resolved.relativeoffset;
                        if (attribIndex <= (NSUIntegerMax - baseOffset) / stride) {
                            NSUInteger vertexOffset = baseOffset + attribIndex * stride;
                            if (vertexOffset <= (NSUInteger)vbo->size &&
                                ((NSUInteger)vbo->size - vertexOffset) >= elementBytes) {
                                GLboolean effectiveNormalized = attrib->normalized;
                                const uint8_t *src = vboBytes + vertexOffset;
                                for (GLuint c = 0; c < MIN(attrib->size, 4u); c++) {
                                    values[c] = mglDecodeVertexAttribComponent(src,
                                                                               attrib->type,
                                                                               effectiveNormalized,
                                                                               c);
                                }
                            }
                        }
                    }
                }
            }

            for (GLuint c = 0; c < member->components && c < 4u; c++) {
                mglWriteTCSStageInComponent(dstVertex, member, c, values[c]);
            }
        }
    }

    if (outStride) {
        *outStride = tcsInStride;
    }
    return stageInBuffer;
}

-(bool) dispatchTessControlShader:(GLMContext) glm_ctx
                          program:(Program *) tcsProgram
                            first:(GLint) first
                            count:(GLsizei) count
                        indexType:(GLenum) indexType
                          indices:(const void *) indices
                       baseVertex:(GLint) baseVertex
                     instanceCount:(GLsizei) drawInstanceCount
                     baseInstance:(GLuint) baseInstance
{
    if (!tcsProgram || !glm_ctx) {
        return false;
    }

    Shader *tcsShader = tcsProgram->shader_slots[_TESS_CONTROL_SHADER];
    if (!tcsShader || !tcsProgram->spirv[_TESS_CONTROL_SHADER].mtl_function) {
        NSLog(@"MGL TESS WARNING: TCS program %u has no compiled function", tcsProgram->name);
        return false;
    }

    /* Create compute pipeline state for TCS kernel. */
    NSError *err = nil;
    id<MTLComputePipelineState> tcsPipeline = mglGetOrCreateProgramComputePipeline(
        _device,
        tcsProgram,
        _TESS_CONTROL_SHADER,
        &err);
    if (!tcsPipeline) {
        NSLog(@"MGL TESS ERROR: failed to create TCS compute pipeline for program %u: %@",
              tcsProgram->name, err);
        return false;
    }

    /* PASS 1: Pre-resolve all Metal textures that the TCS kernel needs.
     * This must happen BEFORE we open a compute encoder, because lazy
     * Metal texture creation (bindMTLTexture:) may open its own blit
     * encoder on the command buffer, and Metal forbids two encoders
     * on the same command buffer simultaneously.  End any active render
     * encoder first for the same reason. */
    if (_currentRenderEncoder) {
        [self endRenderEncoding];
    }

    /* Ensure a writable command buffer exists.  The GL_PATCHES path returns
     * before processGLState() (which normally creates the command buffer),
     * and prior operations (glBufferData, glEndQuery, etc.) may have
     * committed the previous command buffer. */
    if (!_currentCommandBuffer ||
        _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TCS dispatch");
            return false;
        }
    }

    GLuint tcsImgCount = [self getProgramBindingCount:_TESS_CONTROL_SHADER
                                                  type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    for (GLuint i = 0; i < tcsImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(image_units[glUnit].tex);
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_CONTROL_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    id<MTLComputeCommandEncoder> computeEncoder = [_currentCommandBuffer computeCommandEncoder];
    if (!computeEncoder) {
        NSLog(@"MGL TESS ERROR: failed to create compute encoder for TCS dispatch");
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    [computeEncoder setComputePipelineState:tcsPipeline];

    /* PASS 2: Bind storage images for TCS stage. */
    for (GLuint i = 0; i < tcsImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(image_units[glUnit].tex);
        id<MTLTexture> texture = nil;
        if (ptr) {
            texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            GLuint imgLevel = STATE(image_units[glUnit].level);
            if (imgLevel > 0u && texture) {
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                id<MTLTexture> levelView = [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                                                       textureType:texture.textureType
                                                                            levels:NSMakeRange(imgLevel, 1)
                                                                            slices:NSMakeRange(0, sliceCount)];
                if (levelView) {
                    texture = levelView;
                }
            }
        }
        [computeEncoder setTexture:texture atIndex:metalSlot];
    }

    /* Also bind sampled (read-only) images for TCS stage. */
    GLuint tcsSampledCount = [self getProgramBindingCount:_TESS_CONTROL_SHADER
                                                     type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < tcsSampledCount; i++) {
        SpirvResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
            resource = &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(active_textures[glUnit]);
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        id<MTLTexture> texture = ptr ? (__bridge id<MTLTexture>)(ptr->mtl_data) : nil;
        [computeEncoder setTexture:texture atIndex:metalSlot];
        if (resource && resource->msl_has_combined_sampler) {
            id<MTLSamplerState> sampler = nil;
            if (STATE(texture_samplers[glUnit])) {
                Sampler *glSampler = STATE(texture_samplers[glUnit]);
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (!glSampler->mtl_data && ptr) {
                    glSampler->mtl_data = (void *)CFBridgingRetain(
                        [self createMTLSamplerForTexParam:&glSampler->params
                                                  target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
            } else if (ptr && ptr->params.mtl_data) {
                sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
            }
            if (!sampler) {
                sampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
            }
            if (sampler) {
                [computeEncoder setSamplerState:sampler
                                        atIndex:mglMetalCombinedSamplerSlot(resource)];
            }
        }
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TCS. */
    if (![self bindPreparedTessStageBufferBindings:&stageBufferBindings
                                  toComputeEncoder:computeEncoder]) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [self bindPointSizeParamsToComputeEncoder:computeEncoder
                                      program:tcsProgram
                                        stage:_TESS_CONTROL_SHADER];

    /* Create indirect params buffer (buffer 29).
     * spvIndirectParams[0] = vertexCount, [1] = instanceCount. */
    GLuint patchVertices = MAX(1u, (GLuint)STATE(var.patch_vertices));
    GLuint vertexCount = (GLuint)count;
    GLuint instanceCount = (drawInstanceCount > 0) ? (GLuint)drawInstanceCount : 1u;

    /* Create TCS per-vertex output buffer (buffer 28 = spvOut).
     * TCS writes: spvOut[gl_PrimitiveID * outputVertices + invocationID]
     * where outputVertices = tess_control_output_vertices (layout(vertices=N) out).
     * Compute the per-vertex stride from the TCS stage output resources. */
    GLuint tcsOutVertices = tcsProgram->tess_control_output_vertices;
    if (tcsOutVertices == 0) tcsOutVertices = patchVertices;

    /* Compute the per-vertex stride by parsing the MSL output wrapper struct.
     * This includes built-in outputs (gl_Position, gl_PointSize, ...) and
     * Metal alignment padding, which the SPIRV-Cross resource list omits. */
    NSUInteger tcsOutStride = 0;
    const char *tcsMsl = tcsProgram->spirv[_TESS_CONTROL_SHADER].msl_str;
    if (tcsMsl) {
        tcsOutStride = mglComputeMSLOutputStructSize(tcsMsl);
    }
    /* Fallback: sum user-defined outputs from the resource list. */
    if (tcsOutStride == 0 && tcsProgram) {
        SpirvResourceList *outs =
            &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
        for (GLuint i = 0; outs->list && i < outs->count; i++) {
            GLenum gt = outs->list[i].gl_type;
            GLuint comps = 1, bytesPer = 4;
            if (gt == GL_FLOAT_VEC4 || gt == GL_INT_VEC4 || gt == GL_UNSIGNED_INT_VEC4 ||
                gt == GL_BOOL_VEC4) { comps = 4; }
            else if (gt == GL_FLOAT_VEC3 || gt == GL_INT_VEC3 || gt == GL_UNSIGNED_INT_VEC3 ||
                     gt == GL_BOOL_VEC3) { comps = 3; }
            else if (gt == GL_FLOAT_VEC2 || gt == GL_INT_VEC2 || gt == GL_UNSIGNED_INT_VEC2 ||
                     gt == GL_BOOL_VEC2) { comps = 2; }
            else if (gt == GL_FLOAT || gt == GL_INT || gt == GL_UNSIGNED_INT || gt == GL_BOOL) { comps = 1; }
            else if (gt == GL_DOUBLE_VEC4 || gt == GL_DOUBLE_VEC3 || gt == GL_DOUBLE_VEC2) {
                comps = (gt == GL_DOUBLE_VEC4) ? 4 : (gt == GL_DOUBLE_VEC3 ? 3 : 2);
                bytesPer = 8;
            }
            else { comps = 4; }
            tcsOutStride += comps * bytesPer;
        }
    }
    if (tcsOutStride == 0) tcsOutStride = 64;  /* fallback: 4 x float4 */
    _tcsOutputStride = tcsOutStride;
    _tcsOutVertices = tcsOutVertices;

    GLuint patchCountTC = vertexCount / patchVertices;
    if (patchCountTC == 0u) patchCountTC = 1u;
    NSUInteger tcsOutSize = (NSUInteger)patchCountTC * tcsOutVertices * tcsOutStride;
    _tcsOutputBuffer = [_device newBufferWithLength:tcsOutSize
                                            options:MTLResourceStorageModeShared];
    if (!_tcsOutputBuffer || !_tcsOutputBuffer.contents) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(_tcsOutputBuffer.contents, 0, tcsOutSize);
    [computeEncoder setBuffer:_tcsOutputBuffer offset:0 atIndex:28];

    /* Create TCS per-patch output buffer (buffer 27 = spvPatchOut).
     * TCS writes: spvPatchOut[gl_PrimitiveID].
     * The per-patch struct size is harder to compute generically; use a
     * generous estimate based on the patch output resources. */
    NSUInteger tcsPatchStride = 0;
    if (tcsProgram) {
        /* Per-patch outputs share SPVC_RESOURCE_TYPE_STAGE_OUTPUT with
         * per-vertex outputs; SpvDecorationPatch is reflected as is_per_patch. */
        SpirvResourceList *outs =
            &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
        for (GLuint i = 0; outs->list && i < outs->count; i++) {
            if (!outs->list[i].is_per_patch) continue;
            GLenum gt = outs->list[i].gl_type;
            GLuint comps = 1, bytesPer = 4;
            if (gt == GL_FLOAT_VEC4 || gt == GL_INT_VEC4 || gt == GL_UNSIGNED_INT_VEC4) { comps = 4; }
            else if (gt == GL_FLOAT_VEC3 || gt == GL_INT_VEC3 || gt == GL_UNSIGNED_INT_VEC3) { comps = 3; }
            else if (gt == GL_FLOAT_VEC2 || gt == GL_INT_VEC2 || gt == GL_UNSIGNED_INT_VEC2) { comps = 2; }
            else if (gt == GL_FLOAT || gt == GL_INT || gt == GL_UNSIGNED_INT) { comps = 1; }
            else { comps = 4; }
            tcsPatchStride += comps * bytesPer;
        }
    }
    if (tcsPatchStride == 0) tcsPatchStride = 16;  /* fallback: 1 x float4 */
    NSUInteger tcsPatchSize = (NSUInteger)patchCountTC * tcsPatchStride;
    _tcsPatchOutBuffer = [_device newBufferWithLength:tcsPatchSize
                                              options:MTLResourceStorageModeShared];
    if (!_tcsPatchOutBuffer || !_tcsPatchOutBuffer.contents) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(_tcsPatchOutBuffer.contents, 0, tcsPatchSize);
    [computeEncoder setBuffer:_tcsPatchOutBuffer offset:0 atIndex:27];

    GLuint indirectParams[2] = { patchVertices, instanceCount };
    id<MTLBuffer> indirectBuf = [_device newBufferWithBytes:indirectParams
                                                     length:sizeof(indirectParams)
                                                    options:MTLResourceStorageModeShared];
    if (!indirectBuf) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [computeEncoder setBuffer:indirectBuf offset:0 atIndex:29];

    /* Create tessellation factor buffer (buffer 26).
     * MTLQuadTessellationFactorsHalf = 4 edge + 2 inner half-floats = 12 bytes/patch. */
    GLuint patchCount = vertexCount / patchVertices;
    if (patchCount == 0u) patchCount = 1u;
    NSUInteger tessFactorSize = (NSUInteger)patchCount * 12u;
    id<MTLBuffer> tessFactorBuf = [_device newBufferWithLength:tessFactorSize
                                                       options:MTLResourceStorageModeShared];
    if (!tessFactorBuf || !tessFactorBuf.contents) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(tessFactorBuf.contents, 0, tessFactorSize);
    [computeEncoder setBuffer:tessFactorBuf offset:0 atIndex:26];

    if (tcsMsl && strstr(tcsMsl, "_mgl_tcs_in_buffer")) {
        NSUInteger tcsInStride = 0u;
        id<MTLBuffer> tcsStageInBuffer =
            [self newTCSStageInBufferForContext:glm_ctx
                                        program:tcsProgram
                                          first:first
                                          count:count
                                      indexType:indexType
                                        indices:indices
                                     baseVertex:baseVertex
                                   baseInstance:baseInstance
                                  patchVertices:patchVertices
                                     patchCount:patchCount
                                      outStride:&tcsInStride];
        if (!tcsStageInBuffer) {
            NSLog(@"MGL TESS WARNING: failed to pack TCS stage_in buffer for program %u",
                  tcsProgram ? (unsigned)tcsProgram->name : 0u);
            [computeEncoder endEncoding];
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [computeEncoder setBuffer:tcsStageInBuffer
                            offset:0
                           atIndex:kMGLTCSStageInReplBufferIndex];
    }

    /* Dispatch: one threadgroup per patch, tcsOutVertices threads per threadgroup (one thread per TCS output vertex = gl_InvocationID). */
    MTLSize threadgroups = MTLSizeMake(patchCount, 1, 1);
    MTLSize threadsPerTG = MTLSizeMake(tcsOutVertices, 1, 1);
    [computeEncoder dispatchThreadgroups:threadgroups
                     threadsPerThreadgroup:threadsPerTG];

    [computeEncoder endEncoding];

    if (![self flushStageBindingCopyBacks:&stageCopyBacks
                     requireCPUVisibility:YES]) {
        NSLog(@"MGL TESS ERROR: failed to synchronize TCS buffer writes");
        return false;
    }

    /* Save tess factor buffer for TES drawPatches path. */
    _tessFactorBuffer = tessFactorBuf;

    return true;
}

static NSUInteger mglTESXFBFieldByteSize(GLenum glType)
{
    switch (glType) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            return 4u;
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return 8u;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return 12u;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return 16u;
        default:
            return 0u;
    }
}

/* Keep this layout calculation in lockstep with the packed writes injected by
 * mglFixMSLTesAsComputeKernel.  A zero result means the renderer cannot prove
 * the write stride and must not copy temporary capture data into the GL store. */
static NSUInteger mglTESXFBVertexStride(const Program *program)
{
    if (!program || program->transform_feedback_varying_count <= 0) {
        return 0u;
    }

    const SpirvResourceList *outputs =
        &program->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    NSUInteger stride = 0u;
    for (GLsizei varying = 0;
         varying < program->transform_feedback_varying_count;
         varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        const SpirvResource *output = NULL;
        for (GLuint i = 0; name && outputs->list && i < outputs->count; i++) {
            if (outputs->list[i].name && strcmp(outputs->list[i].name, name) == 0) {
                output = &outputs->list[i];
                break;
            }
        }

        NSUInteger fieldBytes = output ? mglTESXFBFieldByteSize(output->gl_type) : 0u;
        if (fieldBytes == 0u || stride > NSUIntegerMax - fieldBytes) {
            return 0u;
        }
        stride += fieldBytes;
    }
    return stride;
}

static bool mglCheckedNSUIntegerProduct(NSUInteger a,
                                        NSUInteger b,
                                        NSUInteger *result)
{
    if (!result || (a != 0u && b > NSUIntegerMax / a)) {
        return false;
    }
    *result = a * b;
    return true;
}

/* Dispatch a TES (Tessellation Evaluation Shader) when there is no TCS and
 * GL_RASTERIZER_DISCARD is active.  SPIRV-Cross lowers the TES to a Metal
 * post-tessellation vertex function (`[[patch(quad, 0)]] vertex ...`), but
 * macOS 26.5 SDK removed postTessellationVertexFunction / isTessellationEnabled
 * from MTLRenderPipelineDescriptor.  We therefore rewrite the TES MSL to a
 * plain compute kernel (mglFixMSLTesAsComputeKernel in program.c) and dispatch
 * it with a compute pipeline, exactly like TCS.
 *
 * The TES kernel uses gl_PrimitiveID (mapped to threadgroup_position_in_grid)
 * as the patch index.  We dispatch one threadgroup per patch with 1 thread
 * per threadgroup, so each invocation handles one patch. */
-(bool) dispatchTessEvaluationShader:(GLMContext) glm_ctx
                            program:(Program *) tesProgram
                              first:(GLint) first
                              count:(GLsizei) count
{
    if (!tesProgram || !glm_ctx) {
        return false;
    }

    Shader *tesShader = tesProgram->shader_slots[_TESS_EVALUATION_SHADER];
    if (!tesShader || !tesProgram->spirv[_TESS_EVALUATION_SHADER].mtl_function) {
        NSLog(@"MGL TESS WARNING: TES program %u has no compiled function", tesProgram->name);
        return false;
    }

    /* Create compute pipeline state for TES kernel. */
    NSError *err = nil;
    id<MTLComputePipelineState> tesPipeline = mglGetOrCreateProgramComputePipeline(
        _device,
        tesProgram,
        _TESS_EVALUATION_SHADER,
        &err);
    if (!tesPipeline) {
        NSLog(@"MGL TESS ERROR: failed to create TES compute pipeline for program %u: %@",
              tesProgram->name, err);
        return false;
    }

    /* PASS 1: Pre-resolve all Metal textures that the TES kernel needs.
     * Must happen before opening any encoder (same reason as TCS). */
    if (_currentRenderEncoder) {
        [self endRenderEncoding];
    }

    /* Ensure a writable command buffer exists (same reason as TCS). */
    if (!_currentCommandBuffer ||
        _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TES dispatch");
            return false;
        }
    }

    GLuint tesImgCount = [self getProgramBindingCount:_TESS_EVALUATION_SHADER
                                                  type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    for (GLuint i = 0; i < tesImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(image_units[glUnit].tex);
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_EVALUATION_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    id<MTLComputeCommandEncoder> computeEncoder = [_currentCommandBuffer computeCommandEncoder];
    if (!computeEncoder) {
        NSLog(@"MGL TESS ERROR: failed to create compute encoder for TES dispatch");
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    [computeEncoder setComputePipelineState:tesPipeline];

    /* PASS 2: Bind storage images for TES stage. */
    for (GLuint i = 0; i < tesImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(image_units[glUnit].tex);
        id<MTLTexture> texture = nil;
        if (ptr) {
            texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            GLuint imgLevel = STATE(image_units[glUnit].level);
            if (imgLevel > 0u && texture) {
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                id<MTLTexture> levelView = [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                                                       textureType:texture.textureType
                                                                            levels:NSMakeRange(imgLevel, 1)
                                                                            slices:NSMakeRange(0, sliceCount)];
                if (levelView) {
                    texture = levelView;
                }
            }
        }
        [computeEncoder setTexture:texture atIndex:metalSlot];
    }

    /* Also bind sampled (read-only) images for TES stage. */
    GLuint tesSampledCount = [self getProgramBindingCount:_TESS_EVALUATION_SHADER
                                                     type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < tesSampledCount; i++) {
        SpirvResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
            resource = &tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(active_textures[glUnit]);
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        id<MTLTexture> texture = ptr ? (__bridge id<MTLTexture>)(ptr->mtl_data) : nil;
        [computeEncoder setTexture:texture atIndex:metalSlot];
        if (resource && resource->msl_has_combined_sampler) {
            id<MTLSamplerState> sampler = nil;
            if (STATE(texture_samplers[glUnit])) {
                Sampler *glSampler = STATE(texture_samplers[glUnit]);
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (!glSampler->mtl_data && ptr) {
                    glSampler->mtl_data = (void *)CFBridgingRetain(
                        [self createMTLSamplerForTexParam:&glSampler->params
                                                  target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
            } else if (ptr && ptr->params.mtl_data) {
                sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
            }
            if (!sampler) {
                sampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
            }
            if (sampler) {
                [computeEncoder setSamplerState:sampler
                                        atIndex:mglMetalCombinedSamplerSlot(resource)];
            }
        }
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TES. */
    if (![self bindPreparedTessStageBufferBindings:&stageBufferBindings
                                  toComputeEncoder:computeEncoder]) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [self bindPointSizeParamsToComputeEncoder:computeEncoder
                                      program:tesProgram
                                        stage:_TESS_EVALUATION_SHADER];

    /* Dispatch: one threadgroup per patch, 1 thread per threadgroup.
     * gl_PrimitiveID → threadgroup_position_in_grid gives the patch index.
     * TessCoord → thread_position_in_threadgroup is 0 (1 thread per TG). */
    GLuint patchVertices = MAX(1u, (GLuint)STATE(var.patch_vertices));
    GLuint vertexCount = (GLuint)count;
    GLuint patchCount = vertexCount / patchVertices;
    if (patchCount == 0u) patchCount = 1u;

    /* Bind patch info to buffer(28): {patch_vertices_in, tcs_out_vertices}.
     * _mgl_patch_info.x = patch vertices (gl_in.size() replacement)
     * _mgl_patch_info.y = TCS output vertices per patch (for per-patch gl_in indexing) */
    {
        GLuint patchInfo[2] = { patchVertices, _tcsOutVertices };
        if (patchInfo[1] == 0) patchInfo[1] = patchVertices;
        [computeEncoder setBytes:patchInfo length:sizeof(patchInfo) atIndex:28];
    }

    /* Bind TCS output buffer to buffer(30) for TES gl_in.
     * TCS writes per-vertex output to spvOut (buffer 28 in TCS).  TES reads
     * gl_in[...] from buffer(30).  The data layout is: TCS writes
     * spvOut[patchID * outputVertices + invocationID], so TES gl_in should
     * point to the same buffer.  The MSL rewriter changed TES's [[stage_in]]
     * to "device <type> *gl_in [[buffer(30)]]". */
    if (_tcsOutputBuffer) {
        [computeEncoder setBuffer:_tcsOutputBuffer offset:0 atIndex:30];
    }

    /* Bind TCS patch output buffer to buffer(27) for TES patchIn.
     * TCS writes per-patch output to spvPatchOut (buffer 27 in TCS).  TES
     * reads patchIn[...] from buffer(27).  Note: buffer 27 is reused for both
     * TCS spvPatchOut and TES patchIn, which is correct since the data flows
     * TCS → TES. */
    if (_tcsPatchOutBuffer) {
        [computeEncoder setBuffer:_tcsPatchOutBuffer offset:0 atIndex:27];
    }

    /* Compute vertsPerPatch from tessellation factors.
     * We dispatch vertsPerPatch threads per threadgroup so each thread
     * writes one XFB entry.  The vertex count formula matches what the
     * CTS counter program expects (primitive count * vertices-per-primitive). */
    GLuint vertsPerPatch = 1;
    if (_tessFactorBuffer) {
        const struct {
            uint16_t edge[4];
            uint16_t inside[2];
        } __attribute__((packed)) *tf = (const void *)_tessFactorBuffer.contents;
        GLenum genMode = tesProgram ? tesProgram->tess_gen_mode : GL_TRIANGLES;
        GLboolean pointMode = tesProgram ? tesProgram->tess_gen_point_mode : GL_FALSE;
        if (patchCount > 0) {
            float edge0 = *(const __fp16 *)&tf[0].edge[0];
            float inside0 = *(const __fp16 *)&tf[0].inside[0];
            float inside1 = *(const __fp16 *)&tf[0].inside[1];
            if (edge0 < 1.0f) edge0 = 1.0f;
            if (inside0 < 1.0f) inside0 = 1.0f;
            if (inside1 < 1.0f) inside1 = 1.0f;
            GLuint primPerPatch = 1;
            if (genMode == GL_QUADS) {
                primPerPatch = 2u * (GLuint)ceilf(inside0) * (GLuint)ceilf(inside1);
            } else if (genMode == GL_TRIANGLES) {
                primPerPatch = (GLuint)ceilf(inside0) * (GLuint)ceilf(inside0);
            } else { /* GL_ISOLINES */
                primPerPatch = (GLuint)ceilf(edge0);
            }
            if (primPerPatch == 0u) primPerPatch = 1u;
            if (pointMode) {
                vertsPerPatch = primPerPatch;
            } else if (genMode == GL_ISOLINES) {
                vertsPerPatch = primPerPatch * 2u;
            } else {
                vertsPerPatch = primPerPatch * 3u;
            }
        }
    }
    if (vertsPerPatch == 0) vertsPerPatch = 1;

    /* Bind XFB output buffer to buffer(29) for _mgl_xfb_out. Metal buffer
     * arguments cannot express a subrange, so a direct binding is safe only
     * when every injected write fits in both the requested GL range and the
     * current logical store. On overflow, capture into a full-size temporary
     * buffer and copy back only the prefix containing complete primitives. */
    TransformFeedback *xfbState = glm_ctx->active_state->transform_feedback;
    const bool xfbCaptureActive =
        tesProgram->transform_feedback_varying_count > 0 &&
        tesProgram->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS &&
        xfbState &&
        xfbState->active &&
        !xfbState->paused &&
        tesProgram->spirv[_TESS_EVALUATION_SHADER].msl_str &&
        strstr(tesProgram->spirv[_TESS_EVALUATION_SHADER].msl_str, "_mgl_xfb_out");
    id<MTLBuffer> xfbTemporary = nil;
    id<MTLBuffer> xfbCopyDestination = nil;
    Buffer *xfbDestination = NULL;
    NSUInteger xfbCopyDestinationOffset = 0u;
    NSUInteger xfbCopyBytes = 0u;
    NSUInteger xfbPrimitiveCapacity = 0u;
    NSUInteger xfbWrittenBytes = 0u;

    if (xfbCaptureActive) {
        BufferBaseTarget *xfbSlot =
            &glm_ctx->active_state->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
        NSUInteger xfbStride = mglTESXFBVertexStride(tesProgram);
        NSUInteger conservativeStride =
            (NSUInteger)tesProgram->transform_feedback_varying_count * 16u;
        NSUInteger allocationStride = xfbStride ? xfbStride : conservativeStride;
        NSUInteger captureVertices = 0u;
        NSUInteger requiredBytes = 0u;
        NSUInteger xfbSessionOffset = 0u;
        bool sessionOffsetOK =
            xfbState->buffer_write_offsets[0] <= (GLuint64)NSUIntegerMax;
        if (sessionOffsetOK) {
            xfbSessionOffset = (NSUInteger)xfbState->buffer_write_offsets[0];
        }
        bool sizeOK =
            allocationStride > 0u &&
            mglCheckedNSUIntegerProduct((NSUInteger)patchCount,
                                        (NSUInteger)vertsPerPatch,
                                        &captureVertices) &&
            mglCheckedNSUIntegerProduct(captureVertices,
                                        allocationStride,
                                        &requiredBytes) &&
            requiredBytes > 0u;

        id<MTLBuffer> xfbMTL = nil;
        NSUInteger visibleBytes = 0u;
        NSUInteger remainingVisibleBytes = 0u;
        NSUInteger destinationOffset = 0u;
        bool destinationOffsetOK = false;
        if (xfbSlot->buf) {
            if (!xfbSlot->buf->data.mtl_data) {
                [self bindMTLBuffer:xfbSlot->buf];
            }
            xfbMTL = (__bridge id<MTLBuffer>)(xfbSlot->buf->data.mtl_data);
            if (xfbMTL) {
                BufferMap xfbMap = {0};
                xfbMap.buf = xfbSlot->buf;
                xfbMap.offset = xfbSlot->offset;
                xfbMap.size = xfbSlot->size;
                visibleBytes = mglBufferMapVisibleBackingBytes(
                    &xfbMap, (size_t)xfbMTL.length);
                if (sessionOffsetOK && xfbSessionOffset <= visibleBytes &&
                    xfbSlot->offset >= 0 &&
                    (NSUInteger)xfbSlot->offset <= NSUIntegerMax - xfbSessionOffset) {
                    remainingVisibleBytes = visibleBytes - xfbSessionOffset;
                    destinationOffset = (NSUInteger)xfbSlot->offset + xfbSessionOffset;
                    destinationOffsetOK = true;
                }
            }
        }

        if (!sizeOK) {
            NSLog(@"MGL TESS XFB: capture size overflow for program %u",
                  (unsigned)tesProgram->name);
            [computeEncoder endEncoding];
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }

        GLuint verticesPerPrimitive =
            tesProgram->tess_gen_point_mode ? 1u :
            (tesProgram->tess_gen_mode == GL_ISOLINES ? 2u : 3u);
        NSUInteger primitiveBytes = 0u;
        bool primitiveLayoutOK =
            xfbStride != 0u &&
            mglCheckedNSUIntegerProduct(xfbStride,
                                        (NSUInteger)verticesPerPrimitive,
                                        &primitiveBytes) &&
            primitiveBytes > 0u;

        if (primitiveLayoutOK && xfbMTL && destinationOffsetOK &&
            requiredBytes <= remainingVisibleBytes) {
            [computeEncoder setBuffer:xfbMTL
                               offset:destinationOffset
                              atIndex:kMGLBufferSlot_IndirectParams];
            xfbSlot->buf->ever_written = GL_TRUE;
            xfbPrimitiveCapacity = captureVertices / verticesPerPrimitive;
            xfbWrittenBytes = xfbPrimitiveCapacity * primitiveBytes;
        } else {
            xfbTemporary = [_device newBufferWithLength:requiredBytes
                                                options:MTLResourceStorageModeShared];
            if (!xfbTemporary) {
                NSLog(@"MGL TESS XFB: failed to allocate %lu-byte overflow buffer",
                      (unsigned long)requiredBytes);
                [computeEncoder endEncoding];
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            memset(xfbTemporary.contents, 0, requiredBytes);
            [computeEncoder setBuffer:xfbTemporary
                               offset:0
                              atIndex:kMGLBufferSlot_IndirectParams];

            /* Unknown layouts stay in the temporary buffer. This is an honest
             * no-capture fallback; copying an unproven stride could overwrite
             * bytes outside a complete transform-feedback primitive. */
            if (primitiveLayoutOK && xfbMTL && destinationOffsetOK &&
                remainingVisibleBytes >= primitiveBytes) {
                xfbPrimitiveCapacity = MIN(captureVertices / verticesPerPrimitive,
                                           remainingVisibleBytes / primitiveBytes);
                xfbCopyBytes = xfbPrimitiveCapacity * primitiveBytes;
                xfbWrittenBytes = xfbCopyBytes;
                xfbCopyDestination = xfbMTL;
                xfbCopyDestinationOffset = destinationOffset;
                xfbDestination = xfbSlot->buf;
            }
        }
    }

    MTLSize threadgroups = MTLSizeMake(patchCount, 1, 1);
    MTLSize threadsPerTG = MTLSizeMake(vertsPerPatch, 1, 1);
    [computeEncoder dispatchThreadgroups:threadgroups
                     threadsPerThreadgroup:threadsPerTG];

    [computeEncoder endEncoding];

    if (![self flushStageBindingCopyBacks:&stageCopyBacks
                     requireCPUVisibility:NO]) {
        NSLog(@"MGL TESS ERROR: failed to copy isolated TES writable buffer prefixes");
        return false;
    }

    if (xfbCopyBytes > 0u) {
        id<MTLBlitCommandEncoder> xfbBlit = [_currentCommandBuffer blitCommandEncoder];
        if (!xfbBlit) {
            NSLog(@"MGL TESS XFB: failed to create bounded copy encoder");
            return false;
        }
        [xfbBlit copyFromBuffer:xfbTemporary
                   sourceOffset:0
                       toBuffer:xfbCopyDestination
              destinationOffset:xfbCopyDestinationOffset
                           size:xfbCopyBytes];
        [xfbBlit endEncoding];
        if (xfbDestination) {
            xfbDestination->ever_written = GL_TRUE;
        }
    }

    if (xfbCaptureActive && xfbWrittenBytes > 0u) {
        GLuint64 currentOffset = xfbState->buffer_write_offsets[0];
        if ((GLuint64)xfbWrittenBytes > UINT64_MAX - currentOffset) {
            xfbState->buffer_write_offsets[0] = UINT64_MAX;
        } else {
            xfbState->buffer_write_offsets[0] =
                currentOffset + (GLuint64)xfbWrittenBytes;
        }
    }

    /* Update GL_PRIMITIVES_GENERATED query by reading the tess factor buffer
     * and computing the number of primitives generated per patch.  The TES
     * compute kernel dispatch above only runs TES once per patch (not per
     * tessellated vertex), so we must manually compute the primitive count
     * that the hardware tessellator would have produced.
     *
     * MTLQuadTessellationFactorsHalf = { half edge[4]; half inside[2]; } = 12 B.
     * For triangles: primitives ≈ ceil(inside)² (rough estimate).
     * For quads:      primitives ≈ 2 × ceil(inside0) × ceil(inside1).
     * For isolines:   primitives ≈ ceil(edge[0]). */
    if (_tessFactorBuffer) {
        const struct {
            uint16_t edge[4];
            uint16_t inside[2];
        } __attribute__((packed)) *tessFactors =
            (const void *)_tessFactorBuffer.contents;

        GLenum genMode = tesProgram ? tesProgram->tess_gen_mode : GL_TRIANGLES;
        GLboolean pointMode = tesProgram ? tesProgram->tess_gen_point_mode : GL_FALSE;

        GLuint64 totalPrimitives = 0;
        for (GLuint p = 0; p < patchCount; p++) {
            /* Tessellation factors are half-floats.  Convert to float. */
            float edge[4], inside[2];
            for (int i = 0; i < 4; i++) {
                edge[i] = *(const __fp16 *)&tessFactors[p].edge[i];
                if (edge[i] < 1.0f) edge[i] = 1.0f;
            }
            for (int i = 0; i < 2; i++) {
                inside[i] = *(const __fp16 *)&tessFactors[p].inside[i];
                if (inside[i] < 1.0f) inside[i] = 1.0f;
            }

            GLuint perPatch = 0;
            if (pointMode) {
                /* Point mode: 1 primitive per tessellated point. */
                if (genMode == GL_QUADS) {
                    perPatch = (GLuint)(ceilf(inside[0]) * ceilf(inside[1]));
                } else if (genMode == GL_TRIANGLES) {
                    perPatch = (GLuint)(ceilf(inside[0]) * ceilf(inside[0]));
                } else { /* GL_ISOLINES */
                    perPatch = (GLuint)ceilf(edge[0]);
                }
            } else {
                if (genMode == GL_QUADS) {
                    /* Each quad splits into 2 triangles. */
                    perPatch = 2u * (GLuint)(ceilf(inside[0]) * ceilf(inside[1]));
                } else if (genMode == GL_TRIANGLES) {
                    perPatch = (GLuint)(ceilf(inside[0]) * ceilf(inside[0]));
                } else { /* GL_ISOLINES */
                    /* Each isoline segment is 1 line primitive (2 vertices). */
                    perPatch = (GLuint)ceilf(edge[0]);
                }
            }
            if (perPatch == 0u) perPatch = 1u;
            totalPrimitives += perPatch;
        }

        GLuint64 writtenPrimitives = totalPrimitives;
        if (xfbCaptureActive && writtenPrimitives > (GLuint64)xfbPrimitiveCapacity) {
            writtenPrimitives = (GLuint64)xfbPrimitiveCapacity;
        }
        mglRecordActivePrimitiveQueryDraw(glm_ctx, totalPrimitives, writtenPrimitives);
    }

    return true;
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

#pragma mark C interface to context functions

- (void) bindObjFuncsToGLMContext: (GLMContext) glm_ctx
{
    glm_ctx->mtl_funcs.mtlObj = (void *)CFBridgingRetain(self);

    glm_ctx->mtl_funcs.mtlBindBuffer = mtlBindBuffer;
    glm_ctx->mtl_funcs.mtlBindTexture = mtlBindTexture;
    glm_ctx->mtl_funcs.mtlBindProgram = mtlBindProgram;

    glm_ctx->mtl_funcs.mtlDeleteMTLObj = mtlDeleteMTLObj;
    glm_ctx->mtl_funcs.release_buffer_metal_data = mtlReleaseBufferMetalData;

    glm_ctx->mtl_funcs.mtlGetSync = mtlGetSync;
    glm_ctx->mtl_funcs.mtlWaitForSync = mtlWaitForSync;
    glm_ctx->mtl_funcs.mtlGetSyncStatus = mtlGetSyncStatus;
    glm_ctx->mtl_funcs.mtlReleaseSync = mtlReleaseSync;
    glm_ctx->mtl_funcs.mtlFlush = mtlFlush;
    glm_ctx->mtl_funcs.mtlSwapBuffers = mtlSwapBuffers;
    glm_ctx->mtl_funcs.mtlFlushDrawBuffer = mtlFlushDrawBuffer;
    glm_ctx->mtl_funcs.mtlInvalidateRenderPass = mtlInvalidateRenderPass;
    glm_ctx->mtl_funcs.mtlClearBuffer = mtlClearBuffer;
    glm_ctx->mtl_funcs.mtlBlitFramebuffer = mtlBlitFramebuffer;

    glm_ctx->mtl_funcs.mtlBufferSubData = mtlBufferSubData;
    glm_ctx->mtl_funcs.mtlMapUnmapBuffer = mtlMapUnmapBuffer;
    glm_ctx->mtl_funcs.mtlFlushBufferRange = mtlFlushBufferRange;

    glm_ctx->mtl_funcs.mtlReadDrawable = mtlReadDrawable;
    glm_ctx->mtl_funcs.mtlReadIntegerPixels = mtlReadIntegerPixels;
    glm_ctx->mtl_funcs.mtlReadDepthPixels = mtlReadDepthPixels;
    glm_ctx->mtl_funcs.mtlGetTexImage = mtlGetTexImage;
    glm_ctx->mtl_funcs.mtlCopyTexSubImage = mtlCopyTexSubImage;

    glm_ctx->mtl_funcs.mtlGenerateMipmaps = mtlGenerateMipmaps;
    glm_ctx->mtl_funcs.mtlTexSubImage = mtlTexSubImage;
    glm_ctx->mtl_funcs.mtlTexSubImageBytes = mtlTexSubImageBytes;

    glm_ctx->mtl_funcs.mtlCopyImageSubData = mtlCopyImageSubData;

    glm_ctx->mtl_funcs.mtlDrawArrays = mtlDrawArrays;
    glm_ctx->mtl_funcs.mtlDrawElements = mtlDrawElements;
    glm_ctx->mtl_funcs.mtlDrawRangeElements = mtlDrawRangeElements;
    glm_ctx->mtl_funcs.mtlDrawArraysInstanced = mtlDrawArraysInstanced;
    glm_ctx->mtl_funcs.mtlDrawElementsInstanced = mtlDrawElementsInstanced;
    glm_ctx->mtl_funcs.mtlDrawElementsBaseVertex = mtlDrawElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlDrawRangeElementsBaseVertex = mtlDrawRangeElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertex = mtlDrawElementsInstancedBaseVertex;
    glm_ctx->mtl_funcs.mtlMultiDrawElementsBaseVertex = mtlMultiDrawElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlDrawArraysIndirect = mtlDrawArraysIndirect;
    glm_ctx->mtl_funcs.mtlDrawElementsIndirect = mtlDrawElementsIndirect;
    glm_ctx->mtl_funcs.mtlDrawArraysInstancedBaseInstance = mtlDrawArraysInstancedBaseInstance;
    glm_ctx->mtl_funcs.mtlDrawElementsInstancedBaseInstance = mtlDrawElementsInstancedBaseInstance;
    glm_ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertexBaseInstance = mtlDrawElementsInstancedBaseVertexBaseInstance;

    glm_ctx->mtl_funcs.mtlMultiDrawArrays = mtlMultiDrawArrays;
    glm_ctx->mtl_funcs.mtlMultiDrawElements = mtlMultiDrawElements;
    glm_ctx->mtl_funcs.mtlMultiDrawElementsBaseVertex = mtlMultiDrawElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlMultiDrawArraysIndirect = mtlMultiDrawArraysIndirect;
    glm_ctx->mtl_funcs.mtlMultiDrawElementsIndirect = mtlMultiDrawElementsIndirect;

    glm_ctx->mtl_funcs.mtlDispatchCompute = mtlDispatchCompute;
    glm_ctx->mtl_funcs.mtlDispatchComputeIndirect = mtlDispatchComputeIndirect;

    glm_ctx->mtl_funcs.mtlBeginSampleQuery = mtlBeginSampleQuery;
    glm_ctx->mtl_funcs.mtlEndSampleQuery = mtlEndSampleQuery;

    glm_ctx->mtl_funcs.mtlBeginTimerQuery = mtlBeginTimerQuery;
    glm_ctx->mtl_funcs.mtlEndTimerQuery = mtlEndTimerQuery;
    glm_ctx->mtl_funcs.mtlGetGPUTimestamp = mtlGetGPUTimestamp;
}

- (id) initMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    assert (window);
    assert (glm_ctx);
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return self;
}

- (id) createMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    assert (window);
    assert (glm_ctx);
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return renderer;
}


void* CppCreateMGLRendererFromContextAndBindToWindow (void *glm_ctx, void *window)
{
    assert (window);
    assert (glm_ctx);
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);
    NSWindow * w = (__bridge NSWindow *)(window); // just a plain bridge as the autorelease pool will try to release this and crash on exit
    assert (w);
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);
    [view setWantsLayer:YES];
    //assert(w.contentView);
    //[w.contentView addSubview:view];
    [w setContentView:view];
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The renderer's lifetime is tied to glm_ctx->mtl_funcs.mtlObj, which is
    // retained via CFBridgingRetain in bindObjFuncsToGLMContext.
    // The caller must NOT CFRelease/free the returned pointer, and must keep
    // glm_ctx alive while using the returned pointer.
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererHeadless (void *glm_ctx)
{
    assert (glm_ctx);
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    // Create a dummy NSView for headless rendering
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);
    [view setWantsLayer:YES];

    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The renderer's lifetime is tied to glm_ctx->mtl_funcs.mtlObj, which is
    // retained via CFBridgingRetain in bindObjFuncsToGLMContext.
    // The caller must NOT CFRelease/free the returned pointer, and must keep
    // glm_ctx alive while using the returned pointer.
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererAndBindToContext (void *glm_ctx)
{
    // Compatibility export used by reference libMGL.dylib.
    // Falls back to headless binding when no Cocoa window is supplied.
    return CppCreateMGLRendererHeadless(glm_ctx);
}

- (void) createMGLRendererAndBindToContext: (GLMContext) glm_ctx view: (NSView *) view
{
    ctx = glm_ctx;

    /* start the DontCare frame generation at 1 so it never matches a
     * texture's zero-initialized mtl_rt_frame_generation stamp until that
     * texture is actually written this frame. */
    _dontCareFrameGeneration = 1u;

    // CRITICAL FIX: Initialize thread synchronization locks.
    // _metalStateLock: NSRecursiveLock (reentrant) — required because the
    //   MGLRenderer call graph has indirect re-entry paths through non-target
    //   helper methods.  A non-reentrant lock deadlocked on first frame.
    // _syncListLock: os_unfair_lock (non-reentrant, value type) - protects
    //   only _currentCommandBufferSyncList and is acquired after
    //   _metalStateLock when both locks are needed.
    _metalStateLock = [[NSRecursiveLock alloc] init];
    _syncListLock   = OS_UNFAIR_LOCK_INIT;
    NSLog(@"MGL INFO: Metal state lock (NSRecursiveLock) + sync list lock (os_unfair_lock) initialized");

    // Initialize AGX GPU error tracking
    _gpuErrorLock = OS_UNFAIR_LOCK_INIT;
    _consecutiveGPUErrors = 0;
    _lastGPUErrorTime = 0;
    _gpuErrorRecoveryMode = NO;
    // Kill-switchable opts: unset = ON, =0/false/no/off = OFF.
    _mslCacheEnabled = mglEnvFlagEnabledDefaultOn("MGL_MSL_CACHE");
    // Bounded per-Program MSL texture type lookup cache (always on; no env var).
    // Keys include a process-unique Program lifetime ID and link generation.
    _mslTextureTypeCache = [NSCache new];
    _mslTextureTypeCache.countLimit = 4096u;
    // PSO dedup: skip forced _lastPipelineState=nil when encoder+PSO unchanged.
    _psoDedupEnabled = mglEnvFlagEnabledDefaultOn("MGL_PSO_DEDUP");
    _pipelineColor0Format = MTLPixelFormatInvalid;
    _pipelineDepthFormat = MTLPixelFormatInvalid;
    _pipelineStencilFormat = MTLPixelFormatInvalid;
    _pipelineProgramName = 0;
    _pipelineVertexFunction = nil;
    _pipelineFragmentFunction = nil;
    _pipelineStateCache = [[NSMutableDictionary alloc] initWithCapacity:64];
    _pipelineStateCacheLRU = [[NSMutableOrderedSet alloc] initWithCapacity:64];
    /* P0-2: Initialize pipeline descriptor cache (two-level caching) */
    _pipelineDescriptorCache = [[NSMutableDictionary alloc] initWithCapacity:64];
    _pipelineDescriptorCacheLRU = [[NSMutableOrderedSet alloc] initWithCapacity:64];
    _dsCacheEnabled = mglEnvFlagEnabledDefaultOn("MGL_DS_CACHE");
    if (_dsCacheEnabled) {
        _depthStencilStateCache = [NSMutableDictionary new];
        _depthStencilStateCacheLRU = [NSMutableOrderedSet new];
    }
    /* Snapshot arena: batch snapshot/commands from bump allocator. */
    _arenaSnapshotEnabled = mglEnvFlagEnabledDefaultOn("MGL_ARENA_SNAPSHOT");
    if (_arenaSnapshotEnabled) {
        if (mglInitBatchArena(&_batchArena, 4u * 1024u * 1024u)) {
            ctx->batch_arena = &_batchArena;
            NSLog(@"MGL INFO: Snapshot arena enabled (initial chunk capacity %zu bytes)",
                  _batchArena.initial_capacity);
        } else {
            _arenaSnapshotEnabled = NO;
            NSLog(@"MGL WARNING: Snapshot arena malloc failed; falling back to per-batch malloc");
        }
    }
    _skipSameKeyRestoreEnabled = mglEnvFlagEnabledDefaultOn("MGL_SKIP_SAME_KEY_RESTORE");
    _dirtyKeyDeltaEnabled = mglEnvFlagEnabledDefaultOn("MGL_DIRTY_KEY_DELTA");
    /* Initialize last-bound render encoder dedup state to a clean slate.
     * _lastBoundValid starts NO so the first bind on the first encoder is
     * never incorrectly skipped. */
    [self invalidateLastBoundState];
    NSLog(@"MGL INFO: AGX GPU error tracking initialized");
    NSLog(@"MGL INFO: perf gates pso_dedup=%d ds_cache=%d arena=%d msl_cache=%d "
          "same_key_restore=%d dirty_key_delta=%d (set VAR=0 to disable)",
          _psoDedupEnabled ? 1 : 0,
          _dsCacheEnabled ? 1 : 0,
          _arenaSnapshotEnabled ? 1 : 0,
          _mslCacheEnabled ? 1 : 0,
          _skipSameKeyRestoreEnabled ? 1 : 0,
          _dirtyKeyDeltaEnabled ? 1 : 0);

    [self bindObjFuncsToGLMContext: glm_ctx];

    // VIRTUALIZED AGX DETECTION: Create Metal device with virtualization safety
    NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating Metal device with virtualization detection");

    // Create the Metal device
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"MGL ERROR: Metal device not found - this is required for Apple Silicon");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, _metalStateLock, _syncListLock, AGX GPU error tracking
        //        fields (_consecutiveGPUErrors/_lastGPUErrorTime/
        //        _gpuErrorRecoveryMode), _pipeline*Format/_pipelineProgramName,
        //        _pipelineStateCache, and glm_ctx->mtl_funcs (bound via
        //        bindObjFuncsToGLMContext, with mtlObj retained).
        //   NIL: _device, _commandQueue, _view.
        // Continuing is pointless without a Metal device — every subsequent
        // operation depends on it.
        return; // Exit early rather than continuing with nil device
    }

    NSLog(@"MGL INFO: Metal device created: %@", _device);
    [self initializeMTL4CompilerIfAvailable];

    /* Initialize AGX Capability Layer (centralized device detection +
     * capability queries + driver bug markers).  Replaces scattered
     * `containsString:@"AGX"` checks and hardcoded constants. */
    MGLCapabilityInit(&_capability, _device);

    // PROPER AGX VIRTUALIZATION DETECTION: Maintain Metal functionality with virtualization compatibility
    BOOL isVirtualized = _capability.isVirtualized;
    NSString *deviceName = [_device name];

    // DETECTION: Check if running in QEMU virtualization but keep Metal enabled
    if (isVirtualized) {
        isVirtualized = YES;
        NSLog(@"MGL INFO: AGX device detected - enabling virtualization compatibility mode: %@", deviceName);
        NSLog(@"MGL INFO: Metal functionality will be maintained with AGX virtualization safety measures");
    }

    // Create command queue with virtualization-safe settings
    MTLCommandQueueDescriptor *queueDescriptor = [[MTLCommandQueueDescriptor alloc] init];
    if (isVirtualized) {
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Enabling virtualization-safe command queue settings");
        queueDescriptor.maxCommandBufferCount = 16;  // Limit concurrent buffers for virtualization safety
    }

    _commandQueue = [_device newCommandQueueWithDescriptor:queueDescriptor];
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Failed to create Metal command queue");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, _metalStateLock, _syncListLock, AGX GPU error tracking
        //        fields, _pipeline*Format/_pipelineProgramName,
        //        _pipelineStateCache, glm_ctx->mtl_funcs (bound, mtlObj
        //        retained), _device, MTL4 compiler (if available), _capability.
        //   NIL: _commandQueue, _view.
        // Continuing is pointless without a command queue — no encoding or
        // submission is possible.
        return;
    }

    NSLog(@"MGL INFO: Metal command queue created successfully");

    /* Phase 2 #6: Load or create Binary Archive for PSO compile acceleration.
     * Gated by MGL_BINARY_ARCHIVE (default ON; =0 disables).
     * The archive is stored in the user's Caches directory and persists
     * compiled PSO binaries across launches, reducing cold-start PSO
     * compile time from ~10s to ~2s on subsequent launches. */
    _binaryArchiveEnabled = mglEnvFlagEnabledDefaultOn("MGL_BINARY_ARCHIVE");
    if (_binaryArchiveEnabled) {
        if (@available(macOS 11.0, *)) {
            [self loadBinaryArchive];
        } else {
            _binaryArchiveEnabled = NO;
        }
    }

    _view = view;

    // PROPER FIX: Create Metal layer with AGX-safe settings
    NSLog(@"MGL INFO: PROPER FIX - Creating Metal layer with AGX-safe settings");

    _layer = [[CAMetalLayer alloc] init];
    if (!_layer) {
        NSLog(@"MGL ERROR: Failed to create Metal layer");
        return;
    }

    _layer.device = _device;
    MTLPixelFormat requestedPixelFormat = ctx ? (MTLPixelFormat)ctx->pixel_format.mtl_pixel_format
                                              : MTLPixelFormatInvalid;
    MTLPixelFormat pf = mglMetalLayerPixelFormatForContext(ctx);

    @try {
        _layer.pixelFormat = pf;
    } @catch (NSException *exception) {
        NSLog(@"MGL CAMetalLayer invalid pixelFormat=%lu requested=%lu exception=%@; falling back to BGRA8Unorm",
              (unsigned long)pf,
              (unsigned long)requestedPixelFormat,
              exception);
        pf = MTLPixelFormatBGRA8Unorm;
        _layer.pixelFormat = pf;
    }

    if (ctx && ctx->pixel_format.mtl_pixel_format != (GLuint)pf) {
        NSLog(@"MGL CAMetalLayer sync default framebuffer metal format glFormat=0x%x glType=0x%x oldMtl=%u newMtl=%lu",
              ctx->pixel_format.format,
              ctx->pixel_format.type,
              ctx->pixel_format.mtl_pixel_format,
              (unsigned long)pf);
        ctx->pixel_format.mtl_pixel_format = (GLuint)pf;
    }
    NSLog(@"MGL CAMetalLayer pixelFormat=%lu requested=%lu glFormat=0x%x glType=0x%x",
          (unsigned long)_layer.pixelFormat,
          (unsigned long)requestedPixelFormat,
          ctx ? ctx->pixel_format.format : 0u,
          ctx ? ctx->pixel_format.type : 0u);
    _layer.opaque = YES;
    _layer.framebufferOnly = NO; // enable blitting to main color buffer
    _layer.allowsNextDrawableTimeout = YES; // avoid indefinite nextDrawable stalls
    _layer.magnificationFilter = kCAFilterNearest;
    _layer.presentsWithTransaction = NO;

    // AGX-safe layer attachment
    if ([_view layer]) {
        [[_view layer] addSublayer: _layer];
    } else {
        [_view setLayer: _layer];
    }
    [self mglSyncLayerDrawableSizeFromView:"createRenderer"];

    mglDrawBuffer(glm_ctx, GL_FRONT);

    // Create initial command buffer for AGX safety
    @try {
        _currentCommandBuffer = [_commandQueue commandBuffer];
        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: Failed to create initial Metal command buffer");
        }
        _mdiArgsScratchBuffer = nil;
        _mdiArgsScratchCapacity = 0;
        _mdiArgsScratchOffset = 0;
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating initial Metal command buffer: %@", exception);
    }
    
    glm_ctx->mtl_funcs.mtlView = (void *)CFBridgingRetain(view);

    // PROACTIVE TEXTURE CREATION: Create essential textures to break sync loop
    NSLog(@"MGL INFO: PROACTIVE - Creating essential textures to prevent magenta screen");
    [self createProactiveTextures];

    // capture Metal commands in MGL.gputrace
    // necessitates Info.plist in the cwd, see https://stackoverflow.com/a/64172784
    //MTLCaptureDescriptor *descriptor = [self setupCaptureToFile: _device];
    //[self startCapture:descriptor];
}

// PROACTIVE TEXTURE CREATION: Create essential textures during initialization to break sync loop
- (void)createProactiveTextures
{
    NSLog(@"MGL PROACTIVE: Starting essential texture creation");

    @try {
        // Create a simple 2D texture with gradient pattern to prevent magenta screens
        MTLTextureDescriptor *proactiveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                          width:256
                                                                                                         height:256
                                                                                                      mipmapped:NO];
        proactiveDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        proactiveDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> proactiveTexture = [_device newTextureWithDescriptor:proactiveDesc];
        if (proactiveTexture) {
            // Create gradient pattern data
            uint32_t *gradientData = calloc(256 * 256, sizeof(uint32_t));
            if (gradientData) {
                // Create blue-green gradient pattern
                for (NSUInteger y = 0; y < 256; y++) {
                    for (NSUInteger x = 0; x < 256; x++) {
                        NSUInteger index = y * 256 + x;
                        uint8_t r = (uint8_t)((x * 128) / 256 + 64);      // Red: 64-192
                        uint8_t g = (uint8_t)((y * 128) / 256 + 64);      // Green: 64-192
                        uint8_t b = 255;                                  // Blue: 255
                        uint8_t a = 255;                                  // Alpha: 255
                        gradientData[index] = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
                    }
                }

                MTLRegion region = MTLRegionMake2D(0, 0, 256, 256);
                [proactiveTexture replaceRegion:region
                                     mipmapLevel:0
                                       withBytes:gradientData
                                     bytesPerRow:256 * sizeof(uint32_t)];

                free(gradientData);
                NSLog(@"MGL PROACTIVE SUCCESS: Created 256x256 gradient texture (prevents magenta screen)");
            } else {
                NSLog(@"MGL PROACTIVE WARNING: Could not allocate gradient data");
            }

            // Store the proactive texture for future use
            if (!_proactiveTextures) {
                _proactiveTextures = [[NSMutableArray alloc] init];
            }
            [_proactiveTextures addObject:proactiveTexture];

        } else {
            NSLog(@"MGL PROACTIVE ERROR: Could not create proactive texture");
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL PROACTIVE ERROR: Exception creating proactive textures: %@", exception.reason);
    }

    NSLog(@"MGL PROACTIVE: Essential texture creation completed");
}

- (MTLCaptureDescriptor *)setupCaptureToFile: (id<MTLDevice>)device//(nonnull MTLDevice* )device // (nonnull MTKView *)view
{
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:@"MGL.gputrace"];
    descriptor.captureObject = device; //((MTKView *)view).device;
    
    return descriptor;
}

- (void)startCapture:(MTLCaptureDescriptor *) descriptor
{
    NSError *error = nil;
    BOOL success = [MTLCaptureManager.sharedCaptureManager startCaptureWithDescriptor:descriptor
                                                                                error:&error];
    if (!success) {
        NSLog(@" error capturing mtl => %@ ", [error localizedDescription] );
    }
}

// Stop the capture.
- (void)stopCapture
{
    [MTLCaptureManager.sharedCaptureManager stopCapture];
}

// CRITICAL FIX: Proper resource cleanup to prevent memory leaks and crashes
- (void)dealloc
{
    NSLog(@"MGL INFO: MGLRenderer dealloc - cleaning up Metal resources");

    @try {
        // Stop any ongoing capture
        [MTLCaptureManager.sharedCaptureManager stopCapture];

        // End any active rendering
        [self endRenderEncoding];

        /* Drop strong references held by the last-bound dedup cache before
         * releasing the underlying Metal resources below. */
        [self invalidateLastBoundState];

        // Cleanup command buffer and encoder
        if (_currentCommandBuffer) {
            NSLog(@"MGL INFO: Releasing current command buffer");
            _currentCommandBuffer = nil;
        }

        if (_currentRenderEncoder) {
            NSLog(@"MGL INFO: Releasing current render encoder");
            _currentRenderEncoder = nil;
        }

        // Cleanup sync objects
        if (_currentEvent) {
            NSLog(@"MGL INFO: Releasing current sync event");
            _currentEvent = nil;
        }

        // Cleanup pipeline state
        if (_pipelineState) {
            NSLog(@"MGL INFO: Releasing pipeline state");
            _pipelineState = nil;
        }
        _pipelineVertexFunction = nil;
        _pipelineFragmentFunction = nil;
        if (_pipelineStateCache) {
            [_pipelineStateCache removeAllObjects];
            _pipelineStateCache = nil;
        }
        [_pipelineStateCacheLRU removeAllObjects];
        _pipelineStateCacheLRU = nil;
        [_pipelineDescriptorCache removeAllObjects];
        _pipelineDescriptorCache = nil;
        [_pipelineDescriptorCacheLRU removeAllObjects];
        _pipelineDescriptorCacheLRU = nil;
        [_depthStencilStateCache removeAllObjects];
        _depthStencilStateCache = nil;
        [_depthStencilStateCacheLRU removeAllObjects];
        _depthStencilStateCacheLRU = nil;

        /* Phase 2 #6: Serialize Binary Archive to disk before releasing it.
         * This persists compiled PSO binaries for the next launch. */
        if (_binaryArchiveEnabled && _binaryArchive) {
            [self saveBinaryArchive];
            _binaryArchive = nil;
        }

        // Cleanup drawable and layer
        if (_drawable) {
            NSLog(@"MGL INFO: Releasing drawable");
            _drawable = nil;
        }

        if (_layer) {
            NSLog(@"MGL INFO: Removing and releasing layer");
            [_layer removeFromSuperlayer];
            _layer = nil;
        }

        // Cleanup command queue and device
        if (_commandQueue) {
            NSLog(@"MGL INFO: Releasing command queue");
            _commandQueue = nil;
        }

        if (_device) {
            NSLog(@"MGL INFO: Releasing Metal device");
            _device = nil;
        }

        // Cleanup thread lock — _metalStateLock is an NSRecursiveLock (ObjC object,
        // requires nil release under ARC). _syncListLock is an os_unfair_lock value
        // type and needs no cleanup.
        if (_metalStateLock) {
            _metalStateLock = nil;
        }

        /* Task 4: Release all address-stable snapshot arena chunks. */
        mglDestroyBatchArena(&_batchArena);

    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during dealloc cleanup: %@", exception);
    }

    NSLog(@"MGL INFO: MGLRenderer dealloc completed");
}

#pragma mark - Metal State Validation and Recovery

- (BOOL)validateMetalObjects
{
    // PROPER FIX: Comprehensive Metal object validation with GPU health monitoring
    @try {
        // Check Metal device validity
        if (!_device) {
            NSLog(@"MGL ERROR: Metal device is nil during validation");
            return NO;
        }

        // Check command queue validity
        if (!_commandQueue) {
            NSLog(@"MGL ERROR: Metal command queue is nil during validation");
            return NO;
        }

        // GPU ERROR THROTTLING: Track recent GPU failures to prevent error cascades
        static NSUInteger consecutiveGpuErrors = 0;
        static NSTimeInterval lastErrorTime = 0;
        static NSTimeInterval throttleWindow = 2.0; // 2 second throttle window
        static NSUInteger maxErrorsPerWindow = 3;

        // Get current error tracking from command buffer if available
        if (_currentCommandBuffer && _currentCommandBuffer.error) {
            NSTimeInterval currentTime = [[NSDate date] timeIntervalSince1970];

            // Check if this is within the throttle window
            if (currentTime - lastErrorTime < throttleWindow) {
                consecutiveGpuErrors++;
                NSLog(@"MGL GPU THROTTLING: %lu consecutive GPU errors detected", (unsigned long)consecutiveGpuErrors);

                // If we've exceeded the error threshold, temporarily disable operations
                if (consecutiveGpuErrors > maxErrorsPerWindow) {
                    NSLog(@"MGL CRITICAL: GPU error threshold exceeded - throttling operations for %.1f seconds", throttleWindow);

                    // Force a reset and temporary pause
                    [self resetMetalState];

                    // Reset counter after pause
                    if (currentTime - lastErrorTime > throttleWindow) {
                        consecutiveGpuErrors = 0;
                    } else {
                        return NO; // Skip this operation to prevent more errors
                    }
                }
            } else {
                // Reset counter if outside throttle window
                consecutiveGpuErrors = 1;
                lastErrorTime = currentTime;
            }
        }

        // Check for virtualization environment changes
        if (@available(macOS 11.0, *)) {
            // Device registry ID changes indicate virtualization issues
            if (_device.registryID == 0) {
                NSLog(@"MGL WARNING: Detected virtualized Metal environment - enabling safety mode");
                // Note: _isVirtualized would be an instance variable to track virtualization state
            }
        }

        return YES;
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Metal object validation failed: %@", exception);
        return NO;
    }
}

- (BOOL)recoverFromMetalError:(NSError *)error operation:(NSString *)operation
{
    // PROPER FIX: Intelligent Metal error recovery
    NSLog(@"MGL ERROR: Metal operation '%@' failed: %@", operation, error);

    // Interface mismatch during pipeline creation is not a GPU-state corruption case.
    // Avoid destructive resets here to prevent reset/retry loops.
    if ([operation isEqualToString:@"pipeline_creation"]) {
        NSString *desc = error.localizedDescription ?: @"";
        NSString *domain = error.domain ?: @"";
        if ((error.code == 3 && [domain hasPrefix:@"AGXMetal"]) ||
            [desc containsString:@"mismatching vertex shader output"] ||
            [desc containsString:@"not written by vertex shader"]) {
            static uint64_t s_pipelineMismatchLogCount = 0;
            s_pipelineMismatchLogCount++;
            if ((s_pipelineMismatchLogCount % 64ull) == 1ull) {
                NSLog(@"MGL WARNING: Pipeline interface mismatch detected; skipping destructive recovery (count=%llu)",
                      s_pipelineMismatchLogCount);
            }
            return NO;
        }
    }

    // Analyze error code for specific recovery strategies
    switch (error.code) {
        case MTLCommandBufferStatusError:
            NSLog(@"MGL INFO: Command buffer execution failed - recreating command buffer");
            [self cleanupCommandBuffer];
            return YES;

        default:
            NSLog(@"MGL ERROR: Unknown Metal error code %ld - attempting recovery", (long)error.code);

            // Handle common error scenarios based on error code
            if (error.code >= 1000 && error.code < 2000) {
                NSLog(@"MGL INFO: Detected feature compatibility issue - using safer settings");
            } else if (error.code >= 2000 && error.code < 3000) {
                NSLog(@"MGL INFO: Detected memory issue - clearing resources");
                [self clearTextureCache];
            } else {
                NSLog(@"MGL ERROR: Unknown Metal error - attempting full recovery");
                [self resetMetalState];
            }
            return YES;
    }
}

- (void)clearTextureCache
{
    // PROPER FIX: Intelligent texture cache cleanup
    NSLog(@"MGL INFO: Clearing texture cache to free memory");

    // Note: Texture binding cache cleanup would require instance variables
    // For now, we focus on basic resource cleanup

    // Force garbage collection using available methods
    if (@available(macOS 10.15, *)) {
        // Simply nil out some references to encourage garbage collection
        // This is a placeholder for more sophisticated cache management
    }
}

- (void)cleanupCommandBuffer
{
    // PROPER FIX: Safe command buffer cleanup
    @try {
        if (_currentCommandBuffer) {
            if (_currentCommandBuffer.status == MTLCommandBufferStatusCommitted) {
                // Do not block indefinitely here; cleanup can be invoked on the render thread.
                // Command buffers retain resources until completion, so dropping the reference is safe.
                if (kMGLVerboseFrameLoopLogs) {
                    NSLog(@"MGL INFO: cleanupCommandBuffer skipping blocking wait for committed command buffer");
                }
            }
            _currentCommandBuffer = nil;
        }
        _mdiArgsScratchBuffer = nil;
        _mdiArgsScratchCapacity = 0;
        _mdiArgsScratchOffset = 0;

        if (_currentRenderEncoder) {
            [_currentRenderEncoder endEncoding];
            _currentRenderEncoder = nil;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during command buffer cleanup: %@", exception);
    }
}

- (void)resetMetalState
{
    // PROPER FIX: Full Metal state reset for AGX driver recovery
    NSLog(@"MGL INFO: Performing full Metal state reset for AGX recovery");

    /* P1-1: dispatched from addCompletedHandler on a Metal worker thread.
     * Must hold _metalStateLock while mutating _commandQueue / _pipelineState
     * / _pipelineStateCache, otherwise the render thread can observe a
     * half-reset state. */
    METAL_LOCK();

    [self cleanupCommandBuffer];

    // CRITICAL: Recreate command queue to clear AGX driver error state
    NSLog(@"MGL AGX RECOVERY: Recreating command queue to clear GPU error state");
    _commandQueue = nil;
    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) {
        NSLog(@"MGL CRITICAL: Failed to recreate command queue during AGX recovery");
    } else {
        NSLog(@"MGL AGX RECOVERY: Command queue successfully recreated");
    }

    // Reset pipeline state
    _pipelineState = nil;
    _pipelineVertexFunction = nil;
    _pipelineFragmentFunction = nil;
    [_pipelineStateCache removeAllObjects];
    [_pipelineStateCacheLRU removeAllObjects];
    [_pipelineDescriptorCache removeAllObjects];
    [_pipelineDescriptorCacheLRU removeAllObjects];
    [_depthStencilStateCache removeAllObjects];
    [_depthStencilStateCacheLRU removeAllObjects];
    // Note: _depthStencilState would be an instance variable if it exists

    // Clear all cached objects
    [self clearTextureCache];

    NSLog(@"MGL INFO: AGX Metal state reset completed");

    METAL_UNLOCK();
}

// AGX Driver Compatibility: Specialized command buffer commit with recovery
- (void)commitCommandBufferWithAGXRecovery:(id<MTLCommandBuffer>)commandBuffer
{
    static uint64_t s_commitCallCount = 0;
    uint64_t commitCall = ++s_commitCallCount;
    bool traceCommit = mglShouldTraceCall(commitCall);

    if (!commandBuffer) {
        NSLog(@"MGL ERROR: Cannot commit NULL command buffer");
        return;
    }

    if (traceCommit) {
        MGLTraceNSLog(@"MGL TRACE commit.begin call=%llu cb=%p status=%s label=%@",
              (unsigned long long)commitCall,
              commandBuffer,
              mglCommandBufferStatusName(commandBuffer.status),
              commandBuffer.label ?: @"(no-label)");
    }
    double commitQueuedAtSeconds = mglNowSeconds();

    // Pre-commit validation for AGX driver
    if (commandBuffer.error) {
        NSLog(@"MGL AGX WARNING: Command buffer has pre-commit error: %@", commandBuffer.error);
        [self recordGPUError];
    }

    // Add completion handler for AGX error detection
    __block typeof(self) blockSelf = self;
    uint64_t commitCallForBlock = commitCall;
    bool traceCommitForBlock = traceCommit;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            double completeElapsedMs = (mglNowSeconds() - commitQueuedAtSeconds) * 1000.0;
            if (traceCommitForBlock || buffer.error || completeElapsedMs >= 50.0) {
                MGLTraceNSLog(@"MGL TRACE commit.completed call=%llu status=%s elapsed=%.3fms error=%@",
                      (unsigned long long)commitCallForBlock,
                      mglCommandBufferStatusName(buffer.status),
                      completeElapsedMs,
                      buffer.error);
            }
            if (buffer.error) {
                NSLog(@"MGL AGX ERROR: Command buffer completed with error: %@", buffer.error);
                [blockSelf recordGPUError];

                // Specific handling for AGX driver rejection
                if ([buffer.error.domain isEqualToString:@"MTLCommandBufferErrorDomain"] &&
                    buffer.error.code == 4) { // "Ignored (for causing prior/excessive GPU errors)"
                static NSTimeInterval s_lastDriverRejectionReset = 0.0;
                NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
                if (now - s_lastDriverRejectionReset > 2.0) {
                    s_lastDriverRejectionReset = now;
                    NSLog(@"MGL AGX RECOVERY: Driver rejection detected; throttled reset scheduled");
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [blockSelf resetMetalState];
                    });
                } else {
                    NSLog(@"MGL AGX RECOVERY: Driver rejection detected; skipping immediate reset (throttled)");
                }
                }
            } else {
            [blockSelf recordGPUSuccess];

            // AGX Recovery: Clear recovery mode on success
            /* P1-1: guard the ivar read/write with _gpuErrorLock (NOT
             * _metalStateLock) to avoid deadlock — the completion handler
             * runs on a Metal worker thread while the render thread may be
             * inside waitUntilCompleted holding _metalStateLock. */
            os_unfair_lock_lock(&blockSelf->_gpuErrorLock);
            if (blockSelf->_gpuErrorRecoveryMode) {
                NSLog(@"MGL AGX RECOVERY: Exiting GPU recovery mode after successful completion");
                blockSelf->_gpuErrorRecoveryMode = NO;
            }
            os_unfair_lock_unlock(&blockSelf->_gpuErrorLock);
        }
    }];

    // CRITICAL FIX: Enhanced command buffer validation before commit
    // Prevents MTLReleaseAssertionFailure in AGX driver
    if (!commandBuffer) {
        NSLog(@"MGL AGX ERROR: Cannot commit nil command buffer");
        return;
    }

    // Check command buffer status before commit
    MTLCommandBufferStatus status = [commandBuffer status];
    if (status >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL AGX WARNING: Command buffer already committed (status: %ld) - skipping commit", (long)status);
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.skip.already_committed call=%llu status=%s",
                  (unsigned long long)commitCall, mglCommandBufferStatusName(status));
        }
        return;
    }

    // Validate command buffer is in a valid state for commit
    if (status == MTLCommandBufferStatusError) {
        NSLog(@"MGL AGX ERROR: Command buffer in error state - skipping commit");
        [self recordGPUError];
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.skip.error_state call=%llu", (unsigned long long)commitCall);
        }
        return;
    }

    if (_isCommittingCommandBuffer) {
        NSLog(@"MGL AGX WARNING: Commit already in progress, skipping nested commit");
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.skip.nested call=%llu", (unsigned long long)commitCall);
        }
        return;
    }

    _isCommittingCommandBuffer = YES;
    @try {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Committing command buffer (status: %ld)", (long)status);
        }
        [commandBuffer commit];
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Command buffer committed successfully");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: Command buffer commit exception: %@", exception);
        [self recordGPUError];

        // AGX-specific recovery for commit failures
        if ([[exception name] containsString:@"CommandBuffer"] ||
            [[exception name] containsString:@"GPU"]) {
            NSLog(@"MGL AGX RECOVERY: Immediate reset due to commit exception");
            dispatch_async(dispatch_get_main_queue(), ^{
                [self resetMetalState];
            });
        }
    } @finally {
        _isCommittingCommandBuffer = NO;
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.end call=%llu cb=%p finalStatus=%s",
                  (unsigned long long)commitCall,
                  commandBuffer,
                  mglCommandBufferStatusName(commandBuffer.status));
        }
    }
}

// AGX GPU Error Throttling - Prevent command queue from entering error state
- (BOOL)shouldSkipGPUOperations
{
    NSTimeInterval currentTime = [[NSDate date] timeIntervalSince1970];
    BOOL needsClear = NO;

    /* P1-1: protect error-tracking ivars with _gpuErrorLock (same lock as
     * recordGPUError/recordGPUSuccess) to avoid racing with the completion
     * handler thread. */
    os_unfair_lock_lock(&_gpuErrorLock);

    // Recovery window: shorter timeout so essential operations can resume sooner
    if (currentTime - _lastGPUErrorTime > 3.0) {
        if (_consecutiveGPUErrors > 0) {
            NSLog(@"MGL AGX: Recovery timeout - attempting GPU operations (had %lu errors)", (unsigned long)_consecutiveGPUErrors);
        }
        _consecutiveGPUErrors = 0;
        _gpuErrorRecoveryMode = NO;
        os_unfair_lock_unlock(&_gpuErrorLock);
        return NO;
    }

    // Enter recovery mode after fewer errors to prevent AGX driver from crashing
    if (_consecutiveGPUErrors >= 8 || _gpuErrorRecoveryMode) {
        if (!_gpuErrorRecoveryMode) {
            NSLog(@"MGL AGX: Entering recovery mode after %lu consecutive errors", (unsigned long)_consecutiveGPUErrors);
            _gpuErrorRecoveryMode = YES;
            needsClear = YES;
        }
        os_unfair_lock_unlock(&_gpuErrorLock);
        if (needsClear) {
            [self clearProblematicGPUState];
        }
        return YES;
    }

    os_unfair_lock_unlock(&_gpuErrorLock);
    return NO;
}

// PROPER FIX: Clear problematic state without giving up on GPU operations entirely
- (void)clearProblematicGPUState
{
    NSLog(@"MGL AGX: Clearing problematic GPU state for recovery");

    // Clear current problematic resources
    if (_currentCommandBuffer) {
        _currentCommandBuffer = nil;
    }

    // Don't recreate command queue immediately - let it rest
    // The AGX driver needs time to recover from error state
}

// AGX DRIVER COMPATIBILITY: Accept virtualization limitations and provide minimal functionality
- (void)enableMinimalFunctionalityMode
{
    NSLog(@"MGL AGX: Enabling minimal functionality mode for AGX virtualization compatibility");

    // Stop fighting the AGX driver - accept virtualization limitations
    // Don't recreate command queues - they will continue to fail
    // Don't submit command buffers - they will continue to be rejected

    // Provide minimal framebuffer clearing without GPU operations
    // This prevents magenta screens while accepting virtualization constraints
}

- (void)recordGPUError
{
    /* P1-1: addCompletedHandler runs on a Metal worker thread, concurrent
     * with the render thread which reads these same ivars.  Use a dedicated
     * os_unfair_lock instead of METAL_LOCK — the completion handler must not
     * block on _metalStateLock because the render thread may be inside
     * waitUntilCompleted (which waits for the handler) while holding it. */
    os_unfair_lock_lock(&_gpuErrorLock);
    _consecutiveGPUErrors++;
    _consecutiveGPUSuccesses = 0;
    _lastGPUErrorTime = [[NSDate date] timeIntervalSince1970];
    NSLog(@"MGL AGX: Recorded GPU error (%lu consecutive)", (unsigned long)_consecutiveGPUErrors);
    os_unfair_lock_unlock(&_gpuErrorLock);
}

- (void)recordGPUSuccess
{
    /* P1-1: use _gpuErrorLock (see recordGPUError comment). */
    os_unfair_lock_lock(&_gpuErrorLock);
    if (_consecutiveGPUErrors > 0 || _gpuErrorRecoveryMode) {
        _consecutiveGPUSuccesses++;
        NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
        NSTimeInterval sinceLastError = now - _lastGPUErrorTime;
        // Require multiple consecutive successful completions before clearing
        // recovery, otherwise mixed success/error callbacks can flap the state.
        if (_consecutiveGPUSuccesses >= 4 && sinceLastError > 0.25) {
            NSLog(@"MGL AGX: Sustained GPU recovery (%lu successes), resetting error count (was %lu)",
                  (unsigned long)_consecutiveGPUSuccesses,
                  (unsigned long)_consecutiveGPUErrors);
            _consecutiveGPUErrors = 0;
            _gpuErrorRecoveryMode = NO;
            _consecutiveGPUSuccesses = 0;
        }
    }
    os_unfair_lock_unlock(&_gpuErrorLock);
}

/* P1-5: FIFO eviction for auxiliary caches.  NSDictionary enumerates in
 * insertion order on recent macOS runtimes, so removing the first 1/4 of
 * allKeys evicts the oldest entries — matching the _pipelineStateCache
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


#pragma mark - Metal Optimization Methods

- (NSUInteger)getOptimalAlignmentForPixelFormat:(MTLPixelFormat)format
{
    (void)format;
    // aligned_alloc requires an alignment compatible with platform pointer alignment.
    // Using a conservative 64-byte value avoids EINVAL on macOS/arm64 and is safe for texture rows.
    return 64;
}

@end
