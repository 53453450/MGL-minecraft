/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_entries.c - the file-scope functions of the former
 * MGLRenderer.m (P0-1, log 202).  The methods had already moved out one cluster
 * at a time; what remained was ~60 plain C functions (program resolution
 * helpers, validated-object lookups, trace/log sinks, dirty-bit dumps and GL
 * entry points), two __attribute__((constructor)) probes, the class extension's
 * stale declarations and an @implementation block that only wrapped the C
 * functions.
 *
 * The Objective-C spellings that had to change are the usual ones: __bridge and
 * id become plain handles, NSLog becomes fprintf, BOOL/YES/NO/nil become
 * int/1/0, the two-line backend lease becomes mglRendererBackendBeginContext,
 * the renderer lookup becomes glm_ctx->platform_renderer_shell, the swap path's
 * @autoreleasepool and @try go through the shell's pool and guard bridges, and
 * the Objective-C-private header (which declared @interface and the class
 * extension) is replaced by the C-safe headers plus local restatements.
 *
 * The two constructors are deliberate entry points: nothing calls them, the
 * loader does (rule 64).
 */

#include <objc/runtime.h>

#include <mach/mach_vm.h>
#include "mgl_render_pass_manager_ops.h"
#include "mgl_gpu_recovery.h"
#include "mgl_attachment_binding.h"  /* FBO attachment bind */
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

#include "mgl_blit_pipelines.h"
#include "mgl_swap_diagnostics.h"  /* swap-time diagnostics (was the SwapDiagnostics category) */
#include "mgl_buffer_map.h"  /* buffer mapping + frame-generation gates */
#include "mgl_renderer_host.h"
#include "mgl_clear_buffer_ops.h" /* mtlClearBuffer + draw-buffer creators (log 184) */  /* shared buffer helpers (log 161) */
#include "mgl_stage_copy_back.h"  /* copy-back list helpers (log 158) */
#include "mgl_renderer_ports.h"
#include "mgl_draw_tess.h"
#include "mgl_air_loader.h"
#include "mgl.h"
#include "mgl_buffer_slots.h"
#include "mgl_sampler_compat.h"
#include "mgl_state_log.h"
#include "mgl_trace_log.h"
#include "mgl_byte_hash.h"
#include "mgl_compute_pipeline_cache.h"

#include "mgl_focus_program.h"
#include "mgl_program_resource.h"
#include "mgl_safety.h"
#include "mgl_thread_affinity.h"   /* mglClaimGLThread / MGL_ASSERT_GL_THREAD */
#include "mgl_shader_resource.h"
#include "mgl_texture_compat.h"
#include "mgl_trace_strategy.h"
#include "mgl_blit_pipelines.h"
#include "mgl_swap_diagnostics.h"  /* swap-time diagnostics (was the SwapDiagnostics category) */
#include "mgl_buffer_map.h"  /* buffer mapping + frame-generation gates */
#include "mgl_renderer_host.h"
#include "mgl_clear_buffer_ops.h" /* mtlClearBuffer + draw-buffer creators (log 184) */  /* shared buffer helpers (log 161) */
#include "mgl_stage_copy_back.h"  /* copy-back list helpers (log 158) */
#include "mgl_renderer_ports.h"
#include "mgl_draw_tess.h"
#include "mgl_air_loader.h"
#include "mgl.h"
#include "mgl_buffer_slots.h"
#include "mgl_sampler_compat.h"
#include "mgl_state_log.h"
#include "mgl_trace_log.h"
#include "mgl_byte_hash.h"
#include "mgl_compute_pipeline_cache.h"


/* Restated from the Objective-C headers a .c file cannot include, the way the
 * other C hosts in this tree do it. */
extern Texture *findTexture(GLMContext ctx, GLuint texture);
static int mglMexObjectPointerLikelyValid(const void *pointer)
{
    return pointer && (uintptr_t)pointer >= 0x1000u &&
           mglPointerRangeIsReadable(pointer, 1u);
}
static int mglMexShouldTraceCall(uint64_t count)
{
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}


/* The Objective-C private headers' file-local constants, copied verbatim
 * (rule 62 (b)). */
static const int kMGLDrawSubmitDiagnostics = 0;
static const int kMGLVerbosePipelineLogs = 0;
#define kMGLCurrentAttribPoolStride ((uint32_t)4096u * 16u)

/* C-safe headers the moved code needs. */
#include "mgl_buffer_query.h"
#include "mgl_coordinate.h"
#include "mgl_draw_buffer.h"
#include "mgl_focus_program.h"
#include "mgl_program_resource.h"
#include "mgl_rt_sync.h"
#include "mgl_safety.h"
#include "mgl_thread_affinity.h"   /* mglClaimGLThread / MGL_ASSERT_GL_THREAD */
#include "mgl_shader_resource.h"
#include "mgl_sync.h"
#include "mgl_texture_compat.h"
#include "mgl_trace_strategy.h"
#include "mgl_vertex_attrib_query.h"
#include "mgl_vertex_format.h"
#include "mgl_size_constants.h"

/* Restated from the Objective-C headers a .c file cannot include, verbatim
 * except for the BOOL -> int spelling (the other C hosts in this tree do the
 * same). */
extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern int mglRendererPointerInHashTable(HashTable *table, const void *ptr);
extern void *mglPlatformRendererShellTextureForDrawable(void *drawable);
extern int mglPlatformShellAutoreleasePoolCall(void *renderer,
                                               int (*body)(void *));

#define mglMexMin(a, b) ((a) < (b) ? (a) : (b))
#define mglMexMax(a, b) ((a) > (b) ? (a) : (b))
static void *mglMexBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents((void *)buffer,
                                                   &contents, &length) == 0
        ? contents : NULL;
}

static uint64_t mglMexBufferLength(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    if (!buffer || mglRenderGetBufferContents(buffer, &contents, &length) != 0) {
        return 0u;
    }
    return length;
}

#define TRACE_FUNCTION()    DEBUG_PRINT("%s\n", __FUNCTION__);

enum {
    MGL_RENDERER_RESOURCE_STORAGE_SHARED = 0u,
    MGL_RENDERER_STORAGE_PRIVATE = 2u,
    MGL_RENDERER_PIXEL_FORMAT_INVALID = 0u,
    MGL_RENDERER_DEPTH32_FLOAT = 252u,
    MGL_RENDERER_CB_NOT_ENQUEUED = 0u,
    MGL_RENDERER_CB_ERROR = 5u,
    MGL_RENDERER_PRIMITIVE_TRIANGLE_STRIP = 4u,
    MGL_RENDERER_LOAD_DONT_CARE = 0u,
    MGL_RENDERER_LOAD_LOAD = 1u,
    MGL_RENDERER_LOAD_CLEAR = 2u,
    MGL_RENDERER_STORE_DONT_CARE = 0u,
    MGL_RENDERER_STORE_STORE = 1u,
    MGL_RENDERER_TEXTURE_USAGE_RENDER_TARGET = 4u,
};

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

/* MGLFragmentTextureTraceBinding typedef moved to mgl_trace_strategy.h. */

/* Draw buffer mapping helpers moved to mgl_draw_buffer.h/.m. */

/* Pixel readback helpers (7 functions) moved to mgl_readback.m */
/* Layer pixel format / sRGB / linear helpers moved to mgl_texture_compat */

static void *mglRendererCreateTextureView(void *texture, uint32_t pixelFormat)
{
    void *view = NULL;
    if (mglRenderCreateTextureView(
            (void *)texture, (uint32_t)pixelFormat,
            &view) == 0 && view) {
        return (void *)view;
    }
    return NULL;
}

typedef struct MGLRendererClearColorValue {
    double red, green, blue, alpha;
} MGLRendererClearColorValue;

static MGLRendererClearColorValue mglRendererMakeClearColor(double red,
                                                            double green,
                                                            double blue,
                                                            double alpha)
{
    return (MGLRendererClearColorValue){red, green, blue, alpha};
}

static MGLRenderTextureInfo mglRendererTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((void *)texture, &info);
    }
    return info;
}

static uint64_t mglRendererTextureFieldWidth(void *texture)
{ return mglRendererTextureInfo(texture).width; }
static uint64_t mglRendererTextureFieldHeight(void *texture)
{ return mglRendererTextureInfo(texture).height; }
static uint32_t mglRendererTextureFieldFormat(void *texture)
{ return mglRendererTextureInfo(texture).pixel_format; }
static uint64_t mglRendererTextureFieldUsage(void *texture)
{ return mglRendererTextureInfo(texture).usage; }
// Applies GL_FRAMEBUFFER_SRGB state to a render-target texture by creating
// a Metal texture view with the appropriate pixel format. The view shares
// the same underlying storage so no memory copy occurs.
// Returns the (possibly wrapped) texture that should be used as the render target.
void *mglApplySRGBStateToRenderTarget(void *texture, GLMContext ctx)
{
    if (!texture || !ctx) return texture;

    uint32_t currentFmt = mglRendererTextureInfo(texture).pixel_format;
    uint32_t desiredFmt;

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

    void *view =
        mglRendererCreateTextureView(texture, desiredFmt);
    if (view) {
        return view;
    }

    // Texture view creation can fail if formats are incompatible;
    // fall back to the original texture.
    static uint64_t s_srgbViewFailCount = 0;
    if (++s_srgbViewFailCount <= 8) {
        fprintf(stderr, "MGL WARNING: newTextureViewWithPixelFormat failed current=%lu desired=%lu srgb=%d\n",
              (unsigned long)currentFmt, (unsigned long)desiredFmt,
              ctx->active_state->caps.framebuffer_srgb ? 1 : 0);
    }
    return texture;
}

/* mglMetalCopyTextureBytesToBGRA8 moved to mgl_readback.m */
void mglMetalCopyRows(const uint8_t *src,
                      unsigned long srcBytesPerRow,
                      uint8_t *dst,
                      unsigned long dstBytesPerRow,
                      unsigned long rowBytes,
                      unsigned long height,
                      int flipY)
{

    mglRenderCopyRows(
        src, (uint64_t)srcBytesPerRow,
        dst, (uint64_t)dstBytesPerRow,
        (uint64_t)rowBytes, (uint64_t)height,
        flipY ? 1 : 0);
}

/* MGLScaledBlitParams / MGLMSAAIntegerResolveParams / MGLClearRectParams
 * typedefs moved to MGLRenderer_Private.h. */
/* MGLBlitAxis struct + blit axis clipping helpers (mglClipBlitAxisToDestination,
 * mglClipBlitAxisToSource, mglClipBlitAxis) moved to mgl_blit_clip.h/.m. */



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

__attribute__((constructor))

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
// kMglSwapPresentDiagnostics moved to mgl_blit_pipelines.h
// kMGLDrawSubmitDiagnostics moved to MGLRenderer_Private.h
// kMGLSynchronizeTextureUploads moved to MGLRenderer_Private.h
// kMGLTextureUploadWaitTimeoutSeconds moved to MGLRenderer_Private.h
// kMGLUseDedicatedTextureUploadCommandBuffer moved to MGLRenderer_Private.h
// Keep vertex attribute buffers in a dedicated high slot range so they do not collide
// with UBO/SSBO bindings that are expected at low indices.
// NOTE: This is the Metal buffer index where vertex attrib buffers start, NOT the
// GL binding count.  MGL user/vertex tables have 31 slots (0..30), so this
// must stay below 31 regardless of MAX_BINDABLE_BUFFERS (which tracks GL state
// only).  Fixed AIR compute ABIs may use internal physical slot 31; that does
// not expand this vertex/user table.
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
    const char *name;
    int value;
    int default_on;    /* distinguishes mglEnvFlagEnabled vs DefaultOn */
    int valid;
} s_mglEnvCache[MGL_ENV_CACHE_CAPACITY];

#include "mgl_env_flag.h"

static int mglEnvFlagEnabledCached(const char *name, int default_on)
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
    int result;
    if (!value || value[0] == '\0') {
        result = default_on;
    } else {
        result = mgl_env_flag_enabled(name) ? 1 : 0;
    }

    /* Store in cache (find first empty slot). */
    for (int i = 0; i < MGL_ENV_CACHE_CAPACITY; i++) {
        if (!s_mglEnvCache[i].valid) {
            s_mglEnvCache[i].name = name;
            s_mglEnvCache[i].value = result;
            s_mglEnvCache[i].default_on = default_on;
            s_mglEnvCache[i].valid = 1;
            break;
        }
    }

    return result;
}

int mglEnvFlagEnabled(const char *name)
{
    return mglEnvFlagEnabledCached(name, 0);
}


int mglEnvFlagEnabledDefaultOn(const char *name)
{
    return mglEnvFlagEnabledCached(name, 1);
}


/* Trace log core infrastructure (3 static globals, mglInitTraceLogIfNeeded,
 * mglTraceLogIsEnabled, mglTraceLogV, mglTraceLog, mglTraceLogExternal,
 * the trace log) moved to mgl_trace_log.h/.c. */

/* mglTraceRTYFlipDiagnosticsEnabled moved to MGLRenderer_Private.h */
/* mglYFlipDecisionName moved to MGLRenderer_Private.h */

/* Fragment texture trace binding helpers moved to mgl_trace_strategy.h/.m. */

/* Frame activity breadcrumbs (19 volatile globals + MGLSwapDrawCounters
 * struct + mglSnapshotSwapDrawCounters/mglResetSwapDrawCounters inline
 * helpers) moved to mgl_frame_activity.h/.m. */

/* mglRendererPointerInHashTable, mglRendererSafeFramebufferName, and
 * mglRendererGetValidatedFramebuffer declared in MGLRenderer_Private.h */
static inline int mglRendererContextLikelyValid(GLMContext ctx)
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
            fprintf(stderr, "MGL PROGRAM RESOLVE invalid cached pointer=%p name=%u\n",
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
        fprintf(stderr, "MGL PROGRAM RESOLVE fail: name=%u missing in table\n", (unsigned)ctx->active_state->program_name);
        ctx->active_state->program_name = 0;
        return NULL;
    }

    if (!resolved->link_success &&
        !resolved->modules[_VERTEX_SHADER].metallib_bytes &&
        !resolved->modules[_FRAGMENT_SHADER].metallib_bytes &&
        !resolved->modules[_COMPUTE_SHADER].metallib_bytes) {
        fprintf(stderr, "MGL PROGRAM RESOLVE pending: name=%u ptr=%p not linked\n",
              (unsigned)ctx->active_state->program_name, resolved);
        return NULL;
    }

    ctx->active_state->program = resolved;
    resolved->refcount++;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);

    fprintf(stderr, "MGL PROGRAM RESOLVE recovered name=%u ptr=%p\n",
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
        if (!mglMexObjectPointerLikelyValid(pipeline) ||
            !mglRendererPointerInHashTable(&ctx->active_state->program_pipeline_table, pipeline) ||
            !mglPointerRangeIsReadable(pipeline, sizeof(*pipeline))) {
            fprintf(stderr, "MGL PROGRAM PIPELINE RESOLVE invalid cached pointer=%p binding=%u\n",
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
        !mglMexObjectPointerLikelyValid(resolved) ||
        !mglPointerRangeIsReadable(resolved, sizeof(*resolved))) {
        fprintf(stderr, "MGL PROGRAM PIPELINE RESOLVE fail: name=%u missing/invalid\n",
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
        fprintf(stderr, "MGL PROGRAM RESTORE missing/invalid program=%u\n", (unsigned)programName);
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
        !mglMexObjectPointerLikelyValid(pipeline) ||
        !mglPointerRangeIsReadable(pipeline, sizeof(*pipeline))) {
        fprintf(stderr, "MGL PROGRAM PIPELINE RESTORE missing/invalid pipeline=%u\n",
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

    if (!mglMexObjectPointerLikelyValid(stageProgram) ||
        !mglPointerRangeIsReadable(stageProgram, sizeof(*stageProgram)) ||
        !mglProgramPointerUsableForName(ctx, stageProgram, stageProgram->name)) {
        fprintf(stderr, "MGL PROGRAM PIPELINE RESOLVE invalid stage program pipeline=%u stage=%s ptr=%p\n",
              (unsigned)pipeline->name,
              mglShaderStageName(stage),
              stageProgram);
        /* Drop the dangling slot reference (retain was taken in
         * mglUseProgramStages) to avoid leaking the program object. */
        pipeline->stage_programs[stage] = NULL;
        mglReleaseProgramReference(ctx, stageProgram);
        return NULL;
    }

    if (!stageProgram->link_success &&
        !stageProgram->modules[_VERTEX_SHADER].metallib_bytes &&
        !stageProgram->modules[_FRAGMENT_SHADER].metallib_bytes &&
        !stageProgram->modules[_COMPUTE_SHADER].metallib_bytes &&
        !stageProgram->modules[_GEOMETRY_SHADER].metallib_bytes &&
        !stageProgram->modules[_TESS_CONTROL_SHADER].metallib_bytes &&
        !stageProgram->modules[_TESS_EVALUATION_SHADER].metallib_bytes) {
        fprintf(stderr, "MGL PROGRAM PIPELINE RESOLVE pending stage program pipeline=%u stage=%s program=%u\n",
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
    mglTraceLog("MGL IFACE program=%u stage=%s type=%s count=%u",
                  (unsigned)program->name,
                  mglShaderStageName(stage),
                  mglMGLShaderResourceTypeName(type),
                  (unsigned)resources->count);

    for (GLuint i = 0; i < resources->count; i++) {
        MGLShaderResource *res = &resources->list[i];
        mglTraceLog("MGL IFACE   #%u name=%s loc=%u glBinding=%u metalBinding=%u set=%u typeId=%u baseTypeId=%u required=%zu imageDim=%u arrayed=%u",
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

void mglWriteProgramMSLDump(Program *program, const char *reason)
{

    if (!mglTraceLogIsEnabled()) {
        return;
    }

    if (!program) {
        return;
    }

    /* Reasons containing "tex" (any case) force the dump past the "dump once
     * per program" gate: they name a texture-binding mismatch the caller wants
     * to see in full. */
    int forceDump = reason && strcasestr(reason, "tex") != NULL;

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

    mglTraceLog("MGL IFACE DUMP begin program=%u reason=%s generation=%u",
                  (unsigned)program->name,
                  reason ? reason : "(none)",
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



/* findTexture, isColorAttachment, getFBOAttachment declared in MGLRenderer_Private.h */

Texture *mglTraceFramebufferAttachmentTexture(GLMContext glctx, FBOAttachment *attachment)
{
    if (!glctx || !attachment) {
        return NULL;
    }
    if (mglRenderTargetIsRenderbuffer((uint32_t)attachment->textarget)) {
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

void mglMarkGLSampledCopyLevelDirty(Texture *tex, GLuint level)
{
    if (!tex || !tex->mtl_gl_sampled_data) {
        return;
    }
    if (level < 32u) {
        tex->mtl_gl_sampled_dirty_mip_mask |= (uint32_t)1u << level;
    } else {
        tex->mtl_gl_sampled_dirty_mip_mask = UINT32_MAX;
    }
}


/* mglMarkTextureLevelRenderTargetWritten macro moved to MGLRenderer_Private.h */

/* mglMarkTextureLevelMetalFilled moved to MGLRenderer_Private.h as static inline */

/* Compressed block height / upload row helpers (mglMetalCompressedBlockHeight,
 * mglMetalUploadRowsForPixelFormat) moved to mgl_texture_compat.h as
 * static inline helpers. */












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
        mglTraceLog("MGL TRACE %s gap=%.2fms deltaCalls=%llu call=%llu",
              tag ? tag : "loop",
              deltaMs,
              (unsigned long long)deltaCalls,
              (unsigned long long)callCount);
    } else if (mglMexShouldTraceCall(callCount) &&
               (callCount <= 20ull || (callCount % 60ull) == 0ull)) {
        mglTraceLog("MGL TRACE %s heartbeat delta=%.2fms deltaCalls=%llu call=%llu",
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
                                void *commandBufferOwner,
                                void *renderEncoderOwner,
                                void *renderPassStateOwner,
                                void *drawable)
{
    if (!kMGLDiagnosticStateLogs) {
        return;
    }

    if (!mglRendererContextLikelyValid(ctx)) {
        mglTraceLog("MGL TRACE %s ctx=%p(invalid) cbOwner=%p encOwner=%p rpOwner=%p drawable=%p",
              tag ? tag : "snapshot", ctx, commandBufferOwner,
              renderEncoderOwner, renderPassStateOwner, drawable);
        return;
    }

    Program *program = mglResolveProgramFromState(ctx);
    GLuint programName = ctx->active_state->program_name ? ctx->active_state->program_name : (program ? program->name : 0);
    Framebuffer *drawFBO = ctx->active_state->framebuffer;
    GLuint drawFBOName = 0;
    if (drawFBO) {
        if (mglMexObjectPointerLikelyValid(drawFBO) &&
            mglRendererPointerInHashTable(&ctx->active_state->framebuffer_table, drawFBO) &&
            mglPointerRangeIsReadable(drawFBO, sizeof(*drawFBO))) {
            drawFBOName = drawFBO->name;
        } else {
            mglTraceLog("MGL TRACE %s invalid drawFBO=%p", tag ? tag : "snapshot", drawFBO);
            drawFBO = NULL;
        }
    }

    MGLRenderCommandBufferState commandState = {0};
    int hasCommandBuffer = mglRenderCommandBufferOwnerHasState(
        commandBufferOwner, &commandState);
    uint32_t cbStatus = hasCommandBuffer
        ? (uint32_t)commandState.status
        : MGL_RENDERER_CB_NOT_ENQUEUED;
    int hasRenderEncoder =
        mglRenderEncoderOwnerHasCurrent(renderEncoderOwner) == 1;
    char dirtyNames[256];
    mglFormatDirtyBits((uint32_t)ctx->active_state->dirty_bits, dirtyNames, sizeof(dirtyNames));

    MGLRenderPassState renderPassState = {0};
    int hasRenderPassState = renderPassStateOwner &&
        mglRenderGetRenderPassStateOwner(
            renderPassStateOwner, &renderPassState) == 0;
    void *rpColor0 = hasRenderPassState && renderPassState.color[0].attachment.texture
        ? (void *)renderPassState.color[0].attachment.texture : NULL;
    void *rpDepth = hasRenderPassState && renderPassState.depth.attachment.texture
        ? (void *)renderPassState.depth.attachment.texture : NULL;
    void *rpStencil = hasRenderPassState && renderPassState.stencil.attachment.texture
        ? (void *)renderPassState.stencil.attachment.texture : NULL;
    uint32_t colorLoadAction = hasRenderPassState
        ? (uint32_t)renderPassState.color[0].attachment.load_action : MGL_RENDERER_LOAD_DONT_CARE;
    uint32_t colorStoreAction = hasRenderPassState
        ? (uint32_t)renderPassState.color[0].attachment.store_action : MGL_RENDERER_STORE_DONT_CARE;
    uint32_t depthLoadAction = hasRenderPassState
        ? (uint32_t)renderPassState.depth.attachment.load_action : MGL_RENDERER_LOAD_DONT_CARE;
    uint32_t depthStoreAction = hasRenderPassState
        ? (uint32_t)renderPassState.depth.attachment.store_action : MGL_RENDERER_STORE_DONT_CARE;
    uint32_t stencilLoadAction = hasRenderPassState
        ? (uint32_t)renderPassState.stencil.attachment.load_action : MGL_RENDERER_LOAD_DONT_CARE;
    uint32_t stencilStoreAction = hasRenderPassState
        ? (uint32_t)renderPassState.stencil.attachment.store_action : MGL_RENDERER_STORE_DONT_CARE;
    MGLRendererClearColorValue rpClearColor = hasRenderPassState
        ? mglRendererMakeClearColor(renderPassState.color[0].clear_red,
                            renderPassState.color[0].clear_green,
                            renderPassState.color[0].clear_blue,
                            renderPassState.color[0].clear_alpha)
        : mglRendererMakeClearColor(0.0, 0.0, 0.0, 0.0);

    void *drawableTexture = drawable
        ? (void *)mglPlatformRendererShellTextureForDrawable((void *)drawable)
        : NULL;

    mglTraceLog("MGL TRACE %s prog=%u dirty=0x%x[%s] clear=0x%x drawBuf=0x%x readBuf=0x%x vao=%p drawFBO=%p(%u) "
          "vp=(%u,%u,%u,%u) scissor(en=%d box=%d,%d,%d,%d) caps(depth=%d blend=%d cull=%d) "
          "stateClear=(%.3f,%.3f,%.3f,%.3f) cbOwner=%p[%s] encOwner=%p(active=%d) rpOwner=%p rt=%lux%lu "
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
          commandBufferOwner,
          mglCommandBufferStatusName(cbStatus),
          renderEncoderOwner,
          hasRenderEncoder,
          renderPassStateOwner,
          (unsigned long)(hasRenderPassState ? renderPassState.render_target_width : 0),
          (unsigned long)(hasRenderPassState ? renderPassState.render_target_height : 0),
          rpColor0,
          (unsigned long)(rpColor0 ? mglRendererTextureFieldFormat(rpColor0) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
          (unsigned long)(rpColor0 ? mglRendererTextureFieldUsage(rpColor0) : 0),
          mglLoadActionName(colorLoadAction),
          mglStoreActionName(colorStoreAction),
          rpClearColor.red,
          rpClearColor.green,
          rpClearColor.blue,
          rpClearColor.alpha,
          rpDepth,
          (unsigned long)(rpDepth ? mglRendererTextureFieldFormat(rpDepth) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
          mglLoadActionName(depthLoadAction),
          mglStoreActionName(depthStoreAction),
          rpStencil,
          (unsigned long)(rpStencil ? mglRendererTextureFieldFormat(rpStencil) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
          mglLoadActionName(stencilLoadAction),
          mglStoreActionName(stencilStoreAction),
          drawable,
          drawableTexture,
          (unsigned long)(drawableTexture ? mglRendererTextureFieldWidth(drawableTexture) : 0),
          (unsigned long)(drawableTexture ? mglRendererTextureFieldHeight(drawableTexture) : 0));

    mglTraceLog("MGL TRACE %s masks color0(use=%d rgba=%d%d%d%d) depthWrite=%d stencilWrite=0x%x",
          tag ? tag : "snapshot",
          ctx->active_state->caps.use_color_mask[0] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
          ctx->active_state->var.color_writemask[0][3] ? 1 : 0,
          ctx->active_state->var.depth_writemask ? 1 : 0,
          (unsigned)ctx->active_state->var.stencil_writemask);
}


void mglLogRenderPassLifecycle(const char *tag,
                                      uint64_t call,
                                      GLMContext ctx,
                                      void *commandBufferOwner,
                                      void *renderEncoderOwner,
                                      void *renderPassStateOwner,
                                      void *drawable,
                                      Framebuffer *renderPassFramebuffer,
                                      GLuint renderPassFramebufferName,
                                      GLenum renderPassDrawBuffer,
                                      GLsizei renderPassDrawBufferCount)
{
    if (!mglTraceLogIsEnabled()) {
        return;
    }

    MGLRenderCommandBufferState commandState = {0};
    int hasCommandBuffer = mglRenderCommandBufferOwnerHasState(
        commandBufferOwner, &commandState);
    uint32_t cbStatus = hasCommandBuffer
        ? (uint32_t)commandState.status
        : MGL_RENDERER_CB_NOT_ENQUEUED;
    int hasRenderEncoder =
        mglRenderEncoderOwnerHasCurrent(renderEncoderOwner) == 1;
    MGLRenderPassState renderPassState = {0};
    int hasRenderPassState = renderPassStateOwner &&
        mglRenderGetRenderPassStateOwner(
            renderPassStateOwner, &renderPassState) == 0;
    void *c0 = hasRenderPassState && renderPassState.color[0].attachment.texture
        ? (void *)renderPassState.color[0].attachment.texture : NULL;
    void *c1 = hasRenderPassState && renderPassState.color[1].attachment.texture
        ? (void *)renderPassState.color[1].attachment.texture : NULL;
    void *depth = hasRenderPassState && renderPassState.depth.attachment.texture
        ? (void *)renderPassState.depth.attachment.texture : NULL;
    void *stencil = hasRenderPassState && renderPassState.stencil.attachment.texture
        ? (void *)renderPassState.stencil.attachment.texture : NULL;
    void *drawableTexture = drawable
        ? (void *)mglPlatformRendererShellTextureForDrawable(drawable)
        : NULL;
    MGLRendererClearColorValue clear = hasRenderPassState
        ? mglRendererMakeClearColor(renderPassState.color[0].clear_red,
                            renderPassState.color[0].clear_green,
                            renderPassState.color[0].clear_blue,
                            renderPassState.color[0].clear_alpha)
        : mglRendererMakeClearColor(0.0, 0.0, 0.0, 0.0);

    Framebuffer *fbo = ctx ? ctx->active_state->framebuffer : NULL;
    if (fbo &&
        (!mglMexObjectPointerLikelyValid(fbo) ||
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
                "fbo=%u(%p) rpFbo=%u(%p) rpDrawBuf=0x%x rpDrawCount=%d vao=%p cbOwner=%p[%s] encOwner=%p(active=%d) rpOwner=%p rt=%lux%lu "
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
                commandBufferOwner,
                mglCommandBufferStatusName(cbStatus),
                renderEncoderOwner,
                hasRenderEncoder,
                renderPassStateOwner,
                (unsigned long)(hasRenderPassState ? renderPassState.render_target_width : 0),
                (unsigned long)(hasRenderPassState ? renderPassState.render_target_height : 0),
                (unsigned)color0Name,
                c0,
                (unsigned long)(c0 ? mglRendererTextureFieldFormat(c0) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
                (unsigned long)(c0 ? mglRendererTextureFieldUsage(c0) : 0),
                (unsigned long)(c0 ? mglRendererTextureFieldWidth(c0) : 0),
                (unsigned long)(c0 ? mglRendererTextureFieldHeight(c0) : 0),
                mglLoadActionName(hasRenderPassState ? (uint32_t)renderPassState.color[0].attachment.load_action : MGL_RENDERER_LOAD_DONT_CARE),
                mglStoreActionName(hasRenderPassState ? (uint32_t)renderPassState.color[0].attachment.store_action : MGL_RENDERER_STORE_DONT_CARE),
                clear.red,
                clear.green,
                clear.blue,
                clear.alpha,
                (unsigned)color1Name,
                c1,
                (unsigned long)(c1 ? mglRendererTextureFieldFormat(c1) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
                (unsigned long)(c1 ? mglRendererTextureFieldUsage(c1) : 0),
                (unsigned long)(c1 ? mglRendererTextureFieldWidth(c1) : 0),
                (unsigned long)(c1 ? mglRendererTextureFieldHeight(c1) : 0),
                mglLoadActionName(hasRenderPassState ? (uint32_t)renderPassState.color[1].attachment.load_action : MGL_RENDERER_LOAD_DONT_CARE),
                mglStoreActionName(hasRenderPassState ? (uint32_t)renderPassState.color[1].attachment.store_action : MGL_RENDERER_STORE_DONT_CARE),
                (unsigned)depthName,
                depth,
                (unsigned long)(depth ? mglRendererTextureFieldFormat(depth) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
                (unsigned long)(depth ? mglRendererTextureFieldUsage(depth) : 0),
                (unsigned long)(depth ? mglRendererTextureFieldWidth(depth) : 0),
                (unsigned long)(depth ? mglRendererTextureFieldHeight(depth) : 0),
                mglLoadActionName(hasRenderPassState ? (uint32_t)renderPassState.depth.attachment.load_action : MGL_RENDERER_LOAD_DONT_CARE),
                mglStoreActionName(hasRenderPassState ? (uint32_t)renderPassState.depth.attachment.store_action : MGL_RENDERER_STORE_DONT_CARE),
                stencil,
                (unsigned long)(stencil ? mglRendererTextureFieldFormat(stencil) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
                (unsigned long)(stencil ? mglRendererTextureFieldUsage(stencil) : 0),
                (unsigned long)(stencil ? mglRendererTextureFieldWidth(stencil) : 0),
                (unsigned long)(stencil ? mglRendererTextureFieldHeight(stencil) : 0),
                mglLoadActionName(hasRenderPassState ? (uint32_t)renderPassState.stencil.attachment.load_action : MGL_RENDERER_LOAD_DONT_CARE),
                mglStoreActionName(hasRenderPassState ? (uint32_t)renderPassState.stencil.attachment.store_action : MGL_RENDERER_STORE_DONT_CARE),
                drawable,
                drawableTexture,
                (unsigned long)(drawableTexture ? mglRendererTextureFieldWidth(drawableTexture) : 0),
                (unsigned long)(drawableTexture ? mglRendererTextureFieldHeight(drawableTexture) : 0));
}

int mglRendererPointerInHashTable(HashTable *table, const void *ptr)
{
    return mglMexObjectPointerLikelyValid(ptr) &&
           mglHashTableContainsData(table, ptr);
}


int mglCurrentDrawFramebufferUsesColorTexture(GLMContext glctx,
                                                      Texture *texture,
                                                      GLuint expectedFboName,
                                                      unsigned long *attachmentIndexOut)
{
    if (attachmentIndexOut) {
        *attachmentIndexOut = MAX_COLOR_ATTACHMENTS;
    }
    if (!glctx || !texture) {
        return 0;
    }

    Framebuffer *fbo = glctx->active_state->framebuffer;
    if (!fbo ||
        !mglMexObjectPointerLikelyValid(fbo) ||
        !mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        return 0;
    }
    if (expectedFboName != 0u && fbo->name != expectedFboName) {
        return 0;
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
            return 1;
        }
    }

    return 0;
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

    if (!mglMexObjectPointerLikelyValid(vao)) {
        fprintf(stderr, "MGL VAO INVALID in %s: vao=%p (suspicious pseudo-pointer)\n",
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
            fprintf(stderr, "MGL VAO INVALID in %s: vao=%p magic=0x%x\n",
                  where ? where : "unknown", vao, vao->magic);
            mglRendererDropCurrentVAO(ctx);
            return NULL;
        }
        return vao;
    }


    if (!mglPointerRangeIsReadable(vao, sizeof(*vao))) {
        fprintf(stderr, "MGL VAO INVALID in %s: vao=%p (unreadable object memory)\n",
              where ? where : "unknown", vao);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    if (vao->magic != MGL_VAO_MAGIC) {
        fprintf(stderr, "MGL VAO INVALID in %s: vao=%p magic=0x%x\n",
              where ? where : "unknown", vao, vao->magic);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    if (vao->transient_batch_vao) {
        return vao;
    }

    fprintf(stderr, "MGL VAO INVALID in %s: vao=%p (not found in sane vao_table)\n",
          where ? where : "unknown", vao);
    mglRendererDropCurrentVAO(ctx);
    return NULL;
}

Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate, const char *where, unsigned long slot)
{
    if (!candidate) {
        return NULL;
    }

    if (!mglMexObjectPointerLikelyValid(candidate)) {
        fprintf(stderr, "MGL BUFFER INVALID in %s: slot=%lu candidate=%p (suspicious pseudo-pointer)\n",
              where ? where : "unknown", (unsigned long)slot, candidate);
        return NULL;
    }

    /* Fast path: hashtable membership implies memory is valid (table holds
     * a live reference), so we can skip the vm_region_64 syscall. */
    if (ctx && mglRendererPointerInHashTable(&ctx->active_state->buffer_table, candidate)) {
        return candidate;
    }


    if (!mglPointerRangeIsReadable(candidate, sizeof(*candidate))) {
        fprintf(stderr, "MGL BUFFER INVALID in %s: slot=%lu candidate=%p (unreadable object memory)\n",
              where ? where : "unknown", (unsigned long)slot, candidate);
        return NULL;
    }

    if (candidate->transient_batch_buffer) {
        return candidate;
    }

    fprintf(stderr, "MGL BUFFER INVALID in %s: slot=%lu candidate=%p (not found in sane buffer_table)\n",
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
    GLuint bindingIndex = attrib->buffer_bindingindex;

    const BufferBinding *tableBinding =
        (bindingIndex < MGL_MAX_VERTEX_ATTRIB_BINDINGS)
            ? &vao->bindings[bindingIndex] : NULL;
    const int tableActive = (tableBinding != NULL && tableBinding->buffer);
    if (tableActive) {
        buffer = tableBinding->buffer;
    }
    MGLRenderVertexAttribResolve resolve = {0};
    if (mglRenderResolveVertexAttribBinding(
            bindingIndex,
            tableActive ? 1 : 0,
            tableActive ? (int64_t)tableBinding->offset : 0,
            tableActive ? (uint32_t)tableBinding->stride : 0u,
            (int64_t)attrib->binding_offset,
            (uint32_t)attrib->stride,
            tableActive ? (uint32_t)tableBinding->divisor : 0u,
            (uint32_t)attrib->divisor,
            &resolve) != 0) {
        return false;
    }
    GLintptr bindingOffset = (GLintptr)resolve.binding_offset;
    GLuint stride = (GLuint)resolve.stride;
    GLuint divisor = (GLuint)resolve.divisor;
    bool usesBindingTable = resolve.use_binding_table != 0;

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

    if (!mglMexObjectPointerLikelyValid(fbo)) {
        fprintf(stderr, "MGL FBO INVALID in %s: framebuffer=%p (suspicious pseudo-pointer)\n",
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


    if (!mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        fprintf(stderr, "MGL FBO INVALID in %s: framebuffer=%p (not found in sane framebuffer_table or unreadable)\n",
              where ? where : "unknown", fbo);
        if (ctx->active_state->readbuffer == fbo) {
            ctx->active_state->readbuffer = NULL;
        }
        ctx->active_state->framebuffer = NULL;
        mglRendererSyncFramebufferBindingNames(ctx);
        mglMarkStateDirtyBits(ctx->active_state, (DIRTY_FBO | DIRTY_STATE));
        return NULL;
    }

    fprintf(stderr, "MGL FBO INVALID in %s: framebuffer=%p (not found in sane framebuffer_table)\n",
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

unsigned long mglRendererBuildCurrentVertexAttribBytes(GLMContext ctx,
                                                           GLuint attribute,
                                                           const VertexAttrib *attrib,
                                                           uint8_t bytes[16])
{
    if (!ctx || !attrib || !bytes || attribute >= MAX_ATTRIBS) {
        return 0u;
    }
    const CurrentVertexAttrib *current =
        &ctx->active_state->current_vertex_attrib[attribute];
    return (unsigned long)mglRenderBuildCurrentVertexAttribBytes(
        (uint32_t)attrib->type, (uint32_t)attrib->size, current->i, current->u,
        current->f, bytes);
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

/* CPU-converted vertex streams bind a fresh Metal buffer per attribute
 * (DOUBLE→float, INT→float, FIXED/packed unpack, integer signedness fix).
 * Those must keep distinct Metal slots when binding_offset differs; plain
 * shared-VBO attributes can share one slot and encode offsets in the
 * vertex descriptor instead (CTS enable_disable: 15 attrs on one VBO). */
static int mglVertexAttribNeedsConvertedMetalStream(Program *program,
                                                     VertexArray *vao,
                                                     GLuint attrib)
{
    if (!vao || attrib >= MAX_ATTRIBS) {
        return 0;
    }
    VertexAttrib *a = &vao->attrib[attrib];
    if (mglRenderAttribNeedsConvertedMetalStream((uint32_t)a->type,
                                                 a->integer ? 1 : 0)) {
        return 1;
    }
    if (a->integer == 1 && program) {
        MGLShaderResource *attrRes =
            mglRendererProgramVertexAttribResource(program, attrib);
        GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
        if (mglIntegerAttribNeedsConversion(a->type, shaderGlType, a->size, NULL)) {
            return 1;
        }
    }
    return 0;
}

int mglRenderVertexBufferIndexForAttribute(GLMContext ctx, GLMState *state, int attribute, const char *where)
{
    if (attribute < 0 || attribute >= MAX_ATTRIBS) {
        fprintf(stderr, "MGL ERROR: getVertexBufferIndexWithAttributeSet invalid attribute=%d\n", attribute);
        return -1;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(ctx, where);
    if (vao) {
        int resolved = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, (GLuint)attribute, where);
        if (resolved >= 0) {
            return resolved;
        }
    }

    /* Legacy fallback: use cached map list if available. */
    GLuint mapCount = state->vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        if (state->vertex_buffer_map_list.buffers[i].attribute_mask & (0x1u << attribute)) {
            GLuint baseIndex = state->vertex_buffer_map_list.buffers[i].buffer_base_index;
            if (baseIndex >= kMGLMaxMetalVertexBufferCount) {
                fprintf(stderr, "MGL ERROR: getVertexBufferIndexWithAttributeSet mapped base index out of Metal range=%u (max valid=%lu)\n",
                      baseIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }
            return (int)baseIndex;
        }
    }

    fprintf(stderr, "MGL ERROR: No vertex buffer mapping found for attribute %d\n", attribute);
    return -1;
}

bool mglRenderCheckForDirtyBufferData(GLMContext ctx, BufferMapList *buffer_map_list, const char *where)
{
    if (!buffer_map_list) {
        return false;
    }

    GLuint mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        fprintf(stderr, "MGL WARNING: checkForDirtyBufferData mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping\n",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          where,
                                                          (unsigned long)i);
        if (gl_buffer) {
            if (gl_buffer->data.dirty_bits) {
                return true;
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return false;
}

bool mglRenderUpdateDirtyBaseBufferList(GLMContext ctx, BufferMapList *buffer_map_list, const char *where)
{
    if (!buffer_map_list) {
        return true;
    }

    GLuint mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        fprintf(stderr, "MGL WARNING: updateDirtyBaseBufferList mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping\n",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          where,
                                                          (unsigned long)i);
        if (gl_buffer) {
            if (gl_buffer->data.dirty_bits) {
                char error[256] = {0};
                int result = mglRenderUpdateDirtyBuffer(gl_buffer, error, sizeof(error));
                if (result != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
                    fprintf(stderr, "MGL BUFFER ERROR: Metal-cpp dirty update failed buffer=%u: %s\n",
                          gl_buffer ? gl_buffer->name : 0u, error[0] ? error : "?");
                    return false;
                }
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return true;
}

bool mglRenderGenerateVertexDescriptorState(GLMContext ctx,
                                           MGLRenderPipelineDescriptorState *state,
                                           int nativeTESActive,
                                           const Program *nativeTESProgram,
                                           uint32_t tcsOutputStride,
                                           int absoluteVertexBindingOffsets,
                                           const char *where)
{
    if (!state) {
        return false;
    }
    state->attrib_count = 0u;
    if (nativeTESActive) {
        MGLTessNativeVertexPlan nativePlan = {0};
        if (!mglTessPlanNativeVertexDescriptor(
                nativeTESProgram,
                (uint32_t)tcsOutputStride, &nativePlan)) {
            fprintf(stderr, "MGL TESS ERROR: unsupported native TES control-point layout\n");
            return false;
        }
        for (uint32_t a = 0u; a < nativePlan.n_attribs; a++) {
            const uint32_t attribute = nativePlan.attribs[a].index;
            if (!mglRenderNativeAttribIndexValid(attribute)) {
                continue;
            }
            state->attrib_format[attribute] = nativePlan.attribs[a].format;
            state->attrib_offset[attribute] = nativePlan.attribs[a].offset;
            state->attrib_buffer_index[attribute] = 0u;
            state->attrib_stride[attribute] = nativePlan.stride;
            state->attrib_step_function[attribute] =
                mglRenderNativeAttribStepFunction();
            state->attrib_step_rate[attribute] = 1u;
        }
        state->attrib_count = nativePlan.attrib_count;
        return true;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, where);
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    GLuint activeProgramName = activeProgram ? activeProgram->name : (ctx ? mglCurrentRenderProgramKey(ctx) : 0);
    GLuint maxAttribs;

    if (!vao) {
        fprintf(stderr, "MGL PIPELINE DESC fail: cannot build vertex descriptor without a valid VAO\n");
        return false;
    }

    if (kMGLVerbosePipelineLogs) {
        fprintf(stderr, "MGL VERTEX DESC begin program=%u vao=%p enabledMask=0x%x\n",
                (unsigned)activeProgramName, (void *)vao, vao->enabled_attribs);
    }

    maxAttribs = MAX_ATTRIBS;

    unsigned long layoutStride[31] = {0};
    for (GLuint i = 0; i < maxAttribs; i++)
    {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, i)) {
            continue;
        }
        int usesCurrentValue = mglRendererVertexAttribUsesCurrentValue(vao, i);
        MGLResolvedVertexAttribBinding resolved = {0};
        bool hasAttribBinding = mglRendererResolveVertexAttribBinding(ctx,
                                                                      vao,
                                                                      i,
                                                                      where,
                                                                      &resolved);
        if (mglRenderSkipUnboundAttrib(usesCurrentValue ? 1 : 0,
                                       hasAttribBinding ? 1 : 0)) {
            continue;
        }

        {
            Buffer *attribBuffer = hasAttribBinding ? resolved.buffer : NULL;

            if (!usesCurrentValue && !attribBuffer)
            {
                fprintf(stderr, "MGL PIPELINE DESC fail: attrib %u enabled but buffer is invalid\n", (unsigned)i);
                return false;
            }

            MGLShaderResource *attrRes =
                mglRendererProgramVertexAttribResource(activeProgram, i);
            GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
            uint32_t format = 0u;
            int needsConversion = 0;
            int effectiveNormalized = 0;
            int conversionKind = 0;
            mglRenderPlanVertexAttribFormat(
                (uint32_t)vao->attrib[i].type, (uint32_t)vao->attrib[i].size,
                vao->attrib[i].integer ? 1 : 0,
                vao->attrib[i].normalized ? 1 : 0,
                mglRendererVertexAttribIsColorInput(activeProgram, i) ? 1 : 0,
                (uint32_t)shaderGlType, &format, &needsConversion,
                &effectiveNormalized, &conversionKind);
            (void)effectiveNormalized;

            if (!mglRenderAttribFormatMapped(format))
            {
                fprintf(stderr, "MGL PIPELINE DESC fail: unable to map attrib %u type/size/normalize to MTL format\n", (unsigned)i);
                return false;
            }

            int mapped_buffer_index;

            mapped_buffer_index = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, i, where);
            if (!mglRenderVertexBufferIndexValid(
                    mapped_buffer_index,
                    (uint32_t)kMGLMaxMetalVertexBufferCount)) {
                fprintf(stderr, "MGL ERROR: Invalid vertex buffer index %d for attribute %d (max valid=%lu)\n",
                        mapped_buffer_index, (unsigned)i, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return false;
            }

            uint32_t attribOffset = mglRenderPlanVertexAttribOffset(
                usesCurrentValue ? 1 : 0, needsConversion,
                absoluteVertexBindingOffsets ? 1 : 0, i,
                kMGLCurrentAttribPoolStride,
                (uint32_t)resolved.relativeoffset,
                (uint32_t)resolved.binding_offset);

            uint32_t stride = mglRenderPlanVertexAttribStride(
                (uint32_t)vao->attrib[i].type, (uint32_t)vao->attrib[i].size,
                vao->attrib[i].integer ? 1 : 0, usesCurrentValue ? 1 : 0,
                conversionKind == MGL_ATTRIB_CONV_INTEGER_SIGN ? 1 : 0,
                (uint32_t)resolved.stride,
                (uint32_t)layoutStride[mapped_buffer_index]);
            layoutStride[mapped_buffer_index] = stride;

            state->attrib_format[i] = (uint32_t)format;
            state->attrib_offset[i] = attribOffset;
            state->attrib_buffer_index[i] = (uint32_t)mapped_buffer_index;
            state->attrib_stride[i] = (uint32_t)stride;
            mglRenderAttribStepFromDivisor(
                usesCurrentValue ? 1 : 0, (uint32_t)resolved.divisor,
                &state->attrib_step_function[i], &state->attrib_step_rate[i]);
            state->attrib_count =
                mglRenderAttribCountAfter(state->attrib_count, i);
        }
    }

    // clear all dirty bits as they have been translated into a vertex descriptor
    vao->dirty_bits = 0;

    return true;
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
    int seenCurrentAttribs[MAX_ATTRIBS] = {0};
    int seenNeedsConverted[MAX_ATTRIBS] = {0};
    GLuint seenCount = 0;
    GLuint maxAttribs = MAX_ATTRIBS;

    bool vaoHasExplicitAttribs = (vao->enabled_attribs != 0u);
    for (GLuint i = 0; i < maxAttribs; i++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, i)) {
            continue;
        }

        int usesCurrentValue = mglRendererVertexAttribUsesCurrentValue(vao, i);
        int slot = -1;
        if (usesCurrentValue) {
            /* Packed current-value pool: ALL current-value attribs share
             * ONE Metal slot.  Per-attrib data is addressed by the vertex
             * descriptor offset (attrib index × pool stride), so N
             * current-value attribs cost one slot instead of N — a
             * 16-element attrib array driven entirely by glVertexAttrib4f
             * would otherwise need slots 16..31 and overflow the 31-slot
             * Metal vertex-buffer budget at attribute 15. */
            for (GLuint s = 0; s < seenCount; s++) {
                if (seenCurrentAttribs[s]) {
                    slot = (int)s;
                    break;
                }
            }
            if (slot < 0) {
                if (kMGLVertexAttribBufferBase + seenCount > kMGLMaxMetalVertexBufferIndex) {
                    fprintf(stderr, "MGL ERROR: Vertex attrib current-value mapping overflow (seen=%u base=%lu maxIndex=%lu)\n",
                          seenCount, (unsigned long)kMGLVertexAttribBufferBase, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                    return -1;
                }
                seenCurrentAttribs[seenCount] = 1;
                seenOffsets[seenCount] = (GLintptr)-1;
                seenStrides[seenCount] = 0u;
                seenDivisors[seenCount] = 0u;
                seenNeedsConverted[seenCount] = 0;
                slot = (int)seenCount;
                seenCount++;
            }
        } else {
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(ctx, vao, i, where, &resolved)) {
            continue;
        }
        if (resolved.binding_offset < 0) {
            fprintf(stderr, "MGL ERROR: attribute %u has negative vertex binding offset=%lld in %s\n",
                  i, (long long)resolved.binding_offset, where);
            return -1;
        }
        Buffer *attribBuffer = resolved.buffer;
        int curNeedsConverted =
            mglVertexAttribNeedsConvertedMetalStream(activeProgram, vao, i);

        for (GLuint s = 0; s < seenCount; s++) {
            if (seenCurrentAttribs[s]) {
                continue;
            }
            Buffer *known = seenBuffers[s];
            int sameStream = 0;
            if (curNeedsConverted || seenNeedsConverted[s]) {
                /* Converted clones start at each attrib's binding_offset;
                 * sharing a Metal slot would overwrite the prior bind. */
                sameStream = mglRendererSameVertexStream(known,
                                                         seenOffsets[s],
                                                         seenStrides[s],
                                                         seenDivisors[s],
                                                         attribBuffer,
                                                         resolved.binding_offset,
                                                         resolved.stride,
                                                         resolved.divisor);
            } else if (known && attribBuffer &&
                       seenStrides[s] == resolved.stride &&
                       seenDivisors[s] == resolved.divisor &&
                       (known == attribBuffer ||
                        (known->name == attribBuffer->name &&
                         known->target == attribBuffer->target))) {
                /* Plain shared VBO: one Metal slot; descriptor holds
                 * binding_offset + relativeoffset per attribute. */
                sameStream = 1;
            }
            if (sameStream) {
                slot = (int)s;
                break;
            }
        }

        if (slot < 0) {
            if (kMGLVertexAttribBufferBase + seenCount > kMGLMaxMetalVertexBufferIndex) {
                fprintf(stderr, "MGL ERROR: Vertex attrib mapping overflow (seen=%u base=%lu maxIndex=%lu)\n",
                      seenCount, (unsigned long)kMGLVertexAttribBufferBase, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }

            seenBuffers[seenCount] = attribBuffer;
            seenOffsets[seenCount] = resolved.binding_offset;
            seenStrides[seenCount] = resolved.stride;
            seenDivisors[seenCount] = resolved.divisor;
            seenNeedsConverted[seenCount] = curNeedsConverted;
            slot = (int)seenCount;
            seenCount++;
        }
        }

        if (i == attribute) {
            unsigned long resolvedIndex = kMGLVertexAttribBufferBase + (unsigned long)slot;
            if (resolvedIndex > kMGLMaxMetalVertexBufferIndex) {
                fprintf(stderr, "MGL ERROR: Vertex attrib index out of Metal range (attrib=%u resolved=%lu max=%lu)\n",
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

// to MGL_ASSERT_GL_THREAD(), validating the single-thread contract in
// Debug builds and compiling to nothing in Release.
//
// Former lock roles are now explicit thread-affinity roles:
//

//    state operations (draw/encode paths) including waitUntilCompleted
//    (RenderPass.m commitFinish/wait paths).  May call
//    recordGPUError/recordGPUSuccess (C++ command recovery owner).
//

//    (commitCommandBufferWithAGXRecovery).  Only touches the
//    thread-safe C++ command recovery owner;
//    never runs MGLRenderer state operations.  May request resetMetalState
//    via the _deviceResetRequested atomic flag (drained on the GL thread
//    at the swap frame boundary).
//

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

// Main class performing the rendering

uint32_t glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized)
{
    return mglRenderGLTypeSizeToVertexFormat((uint32_t)type, (uint32_t)size,
                                             normalized ? 1 : 0);
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
 * mglEncodeRestartSegment, mglEncodePrimitiveRestartedElementDraw) moved to
 * mgl_draw_encode.h/.m.  The two indirect-skip predicates
 * (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled,
 * mglSkipIndirectDrawWhenPolygonPointEmulationNeeded) are pure C and were
 * relocated to mgl_draw_issue.cpp (O5.4: no encode logic left in ObjC). */

/* mglHashStepU64 moved to mgl_byte_hash.h as static inline. */

/* mglVertexDescriptorSignature / mglPipelineDescriptorSignature / mglMaybeInvertMTLWinding moved to mgl_vertex_format.h/.m. */

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
                                       unsigned long indexElement,
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
        mglTraceLog("MGL TRACE drawElements.attrib%u call=%llu program=%u invalid buffer",
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
        void *vb = (void *)(vbo->data.mtl_data);
        vboBytes = (const uint8_t *)mglMexBufferContents(vb);
    }

    if (!vboBytes) {
        mglTraceLog("MGL TRACE drawElements.attrib%u call=%llu program=%u vbo=%u no readable bytes",
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
        mglTraceLog("MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu vbo=%u negative vertexIndex rawIndex=%u baseVertex=%d",
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
    unsigned long vertexIndex = (unsigned long)vertexIndex64;
    unsigned long bindingOffset = (resolved.binding_offset > 0) ? (unsigned long)resolved.binding_offset : 0u;
    unsigned long relativeOffset = (resolved.relativeoffset > 0) ? (unsigned long)resolved.relativeoffset : 0u;
    unsigned long stride = (resolved.stride > 0u) ? (unsigned long)resolved.stride : mglVertexAttribElementBytes(a->type, a->size);
    unsigned long vertexOffset = bindingOffset + relativeOffset + (vertexIndex * stride);
    size_t elemBytes = mglVertexAttribElementBytes(a->type, a->size);
    GLboolean effectiveNormalized = a->normalized;
    Program *program = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (mglRenderAttribColorUByteNeedsNormalize(
            (uint32_t)a->type, (uint32_t)a->size,
            effectiveNormalized ? 1 : 0) &&
        mglRendererVertexAttribIsColorInput(program, attrib)) {
        effectiveNormalized = (GLboolean)mglRenderAttribEffectiveNormalized(
            (uint32_t)effectiveNormalized, 1);
    }

    if (elemBytes == 0u ||
        vertexOffset > (unsigned long)vbo->size ||
        ((unsigned long)vbo->size - vertexOffset) < elemBytes) {
        mglTraceLog("MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu vbo=%u OOB rawIndex=%u baseVertex=%d vertexIndex=%llu bindingOffset=%lu relOffset=%lu stride=%lu size=%u type=0x%x normalized=%u elemBytes=%zu vboSize=%lld",
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
    for (unsigned long c = 0; c < mglMexMin((unsigned long)a->size, (unsigned long)4); c++) {
        comps[c] = mglDecodeVertexAttribComponent(attribBytes, a->type, effectiveNormalized, c);
    }

    char raw[3 * 16 + 1] = {0};
    size_t rawLen = mglMexMin((size_t)16u, elemBytes);
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
    uint32_t format = glTypeSizeToMtlType(a->type, a->size, effectiveNormalized);
    int mappedIndex = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, attrib, "drawElements.attrib.trace");
    MGLShaderResource *resource = mglRendererProgramVertexAttribResource(program, attrib);
    mglTraceLog("MGL TRACE drawElements.attrib%u call=%llu program=%u indexElement=%lu resource=%s metalSlot=%d vbo=%u rawIndex=%u baseVertex=%d vertexIndex=%llu bindingIndex=%u bindingOffset=%lu relOffset=%lu vertexOffset=%lu stride=%lu size=%u type=0x%x normalized=%u/%u format=%lu(%s) decoded=(%.6f,%.6f,%.6f,%.6f) raw=%s",
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




/*
 * GL_TEXTURE_BASE_LEVEL / MAX_LEVEL select the mip window used for sampling.
 * Metal textures always start at level 0, so when that window is narrower
 * than the full mip chain a texture view is created spanning
 * [base_level, max_level] (including base_level==0 with a restricted
 * MAX_LEVEL).  This lets Metal sampler lod clamps operate in the same
 * coordinate space as GL (relative to the view's level 0).  When the window
 * covers the whole texture the original is returned (no overhead).
 */


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

/* Program reflection queries are owned by the C++ renderer backend. */

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



#pragma mark render encoder and command buffer init code

/* updateCurrentRenderEncoder moved to MGLRenderer+RenderPass.m */

/* newRenderEncoder moved to MGLRenderer+RenderPass.m */

/* shouldUseDontCareLoadForColorTexture:firstUseThisFrame: moved to MGLRenderer+RenderPass.m */

/* newRenderEncoderLocked moved to MGLRenderer+RenderPass.m */

/* newCommandBuffer moved to MGLRenderer+RenderPass.m */

/* newCommandBufferLocked moved to MGLRenderer+RenderPass.m */

/* ensureWritableCommandBuffer: moved to MGLRenderer+RenderPass.m */

/* ensureWritableCommandBufferLocked: moved to MGLRenderer+RenderPass.m */






/* newCommandBufferAndRenderEncoder moved to MGLRenderer+RenderPass.m */

/* generatePipelineDescriptorState moved to MGLRenderer+RenderPass.m */

/* generateVertexDescriptorState moved to MGLRenderer+VertexLayout.m */

#pragma mark utility funcs for processGLState


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

/* recordLastBoundVertexBuffer:(id)buffer offset:(unsigned long)offset atIndex:(unsigned long)index moved to MGLRenderer+Draw.m */

/* recordLastBoundFragmentBuffer:(id)buffer offset:(unsigned long)offset atIndex:(unsigned long)index moved to MGLRenderer+Draw.m */

/* invalidateLastBoundVertexBufferAtIndex:(unsigned long)index moved to MGLRenderer+Draw.m */

/* invalidateLastBoundFragmentBufferAtIndex:(unsigned long)index moved to MGLRenderer+Draw.m */

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
/* syncResourceBindingsForContext:(GLMContext)glm_ctx is the C entry
 * mglRendererSyncResourceBindingsForContext (mgl_binding_state_ops.h). */

/* syncPipelineStateWithDeferredBufferMap: moved to MGLRenderer+RenderPass.m */

/* Assigns the context ivar for the C dispatch path (the Objective-C compute
 * entry points did `ctx = glm_ctx;` inline).  A method because the ivar is not
 * visible outside the class body; the shell exposes it to C as
 * mglPlatformShellSetContext(). */

/* bindBufferSizeConstantsForRenderEncoder is the C entry
 * mglRendererBindBufferSizeConstantsForRenderEncoder (mgl_size_constants.h). */

/* flushCommandBuffer: moved to MGLRenderer+RenderPass.m */

/* flushCommandBufferLocked: moved to MGLRenderer+RenderPass.m */
#pragma mark C interface to mtlDeleteMTLObj



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


/* syncRenderPassStateForContext: moved to MGLRenderer+RenderPass.m */

/* rotateRenderEncoderForCurrentFramebufferLocked moved to MGLRenderer+RenderPass.m */

/* prepareRenderPassIfFBOChanged:context:replayError: moved to MGLRenderer+RenderPass.m */

/* checkBatchShouldExecute:(MGLDrawBatch *)batch moved to MGLRenderer+Draw.m */

/* recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueStreamMergedBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* mdiArgumentScratchBufferWithLength:(unsigned long)length moved to MGLRenderer+Draw.m */

/* issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

/* issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx moved to MGLRenderer+Draw.m */

#pragma mark C interface to mtlFlush

#pragma mark C interface to mtlSwapBuffers
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
    unsigned long indexBytesAvailable = 0u;
    if (ebo->data.buffer_data && ((uintptr_t)ebo->data.buffer_data >= 0x1000ull)) {
        indexBytes = (const uint8_t *)ebo->data.buffer_data;
        indexBytesAvailable = (ebo->size > 0) ? (unsigned long)ebo->size : 0u;
    } else if (ebo->data.mtl_data) {
        void *indexBuffer = (void *)(ebo->data.mtl_data);
        if (indexBuffer && mglMexBufferContents(indexBuffer)) {
            indexBytes = (const uint8_t *)mglMexBufferContents(indexBuffer);
            indexBytesAvailable = mglMexBufferLength(indexBuffer);
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

    unsigned long indexOffset = (unsigned long)cmd->indexBufferOffset;
    unsigned long indexStride = mglGLIndexElementSize(cmd->indexType);
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

    unsigned long sampleCount = forceTrace ? mglMexMin((unsigned long)cmd->count, (unsigned long)6u) : (unsigned long)1u;
    GLuint traceAttribLimit = mglMexMin((GLuint)6u, traceCtx->state.max_vertex_attribs);
    for (unsigned long sample = 0; sample < sampleCount; sample++) {
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


static GLMContext mglMexSwapContext;

static int mglMexSwapInner(void *renderer, void *rawCtx)
{
    (void)rawCtx;
    mglRenderPassMTLSwapBuffersLocked(renderer, mglMexSwapContext);
    return 1;
}

static int mglMexSwapBody(void *renderer)
{
    mglClaimGLThread();
    char swapFailure[256] = {0};
    if (!mglPlatformShellGuardedCallCtxReason(
            renderer, "swap command buffer", mglMexSwapInner, NULL,
            swapFailure, sizeof(swapFailure))) {
        fprintf(stderr, "MGL CRITICAL: callback swap exception: %s\n",
                swapFailure[0] ? swapFailure : "(null)");
    }
    return 1;
}

void mglRendererSwapBuffers(GLMContext glm_ctx)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglMexSwapContext = glm_ctx;
        (void)mglPlatformShellAutoreleasePoolCall(renderer, mglMexSwapBody);
    }
    mglRendererBackendEnd(&backend_lease);
}


/* copyRenderPassColorToDrawableIfNeeded: and
 * scheduleSwapTextureSampleDiagnostics: are the C functions of
 * mgl_swap_diagnostics.h now. */

#pragma mark C interface to mtlClearBuffer
/* AGX recovery: recreate the command queue through the backend and report
 * whether the renderer now holds one.  A method rather than a C function because
 * the queue and backend ivars are not visible outside the class body; the shell
 * TU exposes it to C as mglPlatformShellRecreateCommandQueue(). */
/* Metal device/queue probes for the recovery validation path (see
 * mglRecreateCommandQueue for why these are methods, not C functions). */


/* Drawable pointer for the render-pass lifecycle log (see mglRecreateCommandQueue
 * for why these probes are methods rather than C functions). */
/* Emulated-MS sample-loop state, kept as methods because these ivars are private
 * (see mglRecreateCommandQueue for the same reasoning). */





void mglRendererClearBuffer(GLMContext glm_ctx,
                                  unsigned int type,
                                  unsigned int mask)
{
    MGLRendererBackendLease backend_lease = {};
    if (mglRendererBackendBeginContext(glm_ctx, &backend_lease) != 0) return;
    void *renderer = glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
    if (renderer && glm_ctx) {
        mglRendererMTLClearBuffer((void *)renderer, glm_ctx, type, mask);
    }
    mglRendererBackendEnd(&backend_lease);
}

#pragma mark C interface to mtlBufferSubData


#pragma mark C interface to mtlMapUnmapBuffer


#pragma mark C interface to mtlFlushMappedBufferRange

#pragma mark C interface to mtlReadDrawable



#pragma mark C interface to mtlGetTexImage


#pragma mark C interface to mtlGenerateMipmaps


/* Map GL internal format to the (format, type) pair that matches the CPU
 * storage layout used by mglCreateRGBA8ExpandedUpload / channel expansion.
 * Used for format-converting readback in mtlCopyImageSubData when CPU bpp
 * differs from Metal bpp.  Returns GL_FALSE if no mapping is known. */
GLboolean mglGetCPUFormatTypeForInternalFormat(GLenum internalformat,
                                               GLenum *outFormat,
                                               GLenum *outType)
{
    if (!outFormat || !outType) return (GLboolean)mglRenderGLBoolean(0);
    uint32_t format = 0u;
    uint32_t type = 0u;
    if (!mglRenderCPUFormatTypeForInternalFormat(
            (uint32_t)internalformat, &format, &type)) {
        return (GLboolean)mglRenderGLBoolean(0);
    }
    *outFormat = (GLenum)format;
    *outType = (GLenum)type;
    return (GLboolean)mglRenderGLBoolean(1);
}



#pragma mark C interface to mtlTexSubImage





#pragma mark utility functions for draw commands
uint32_t mglPrimitiveTypeForGLMode(GLenum mode)
{
    return mglRenderMTLPrimitiveTypeForGLMode((uint32_t)mode);
}

uint64_t mglIndexTypeForGLType(GLenum type)
{
    return mglRenderMTLIndexTypeForGLType((uint32_t)type);
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



_Static_assert(sizeof(MGLStageBindingCopyBack) == sizeof(MGLRenderCopyBackEntry),
               "copy-back slot ABI matches C entry");





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
