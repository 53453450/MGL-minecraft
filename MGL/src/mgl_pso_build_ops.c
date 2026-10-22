/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_pso_build_ops.c - C home of the PSO cache-miss build path (P0-1,
 * log 187): -buildPipelineStateOnCacheMissWithState: plus the cache insert it
 * calls.  The pipeline-cache value-state bridges and the GPU-recovery record
 * travel through the renderer state areas.
 */

#include "mgl_pso_build_ops.h"

#include "mgl_air_loader.h"        /* MGLRenderPipelineDescriptorState */
#include "mgl_buffer_map.h"      /* mglRendererMapBuffersToMTL */
#include "mgl_byte_hash.h"      /* mglHashStepU64 (geometry PSO key) */
#include "mgl_aux_assets.h"        /* mglAuxShaderAssetFind */
#include "mgl_capability.h"        /* MGLCapabilityHasBug */
#include "mgl_frame_activity.h"
#include "mgl_gpu_recovery.h"      /* mglPlatformShellGuardedCallCtx */
#include "mgl_env_flag.h"        /* mgl_env_flag_enabled (uncached) */
#include "mgl_metal_ref.h"
#include "mgl_pso_format_class.h"
#include "mgl_readback_policy.h"  /* mglRenderMSAAArrayLayerStride */
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_pass_manager.h"
#include "mgl_renderer_ports.h"
#include "mgl_trace_log.h"
#include "mgl_vertex_format.h"    /* mgl*DescriptorSignatureFromState */
#include "mgl_vertex_layout.h"    /* mglRendererUpdateBlendStateCache */

#include "mgl_render_api_command.h"

#include "mgl_render_api_binding.h"

#include "mgl_render_api_texture.h"

#include "mgl_render_api_query.h"

#include "mgl_render_api_lifecycle.h"

#include <stdio.h>
#include <stdlib.h>   /* getenv (MGL_TOPO_TRACE) */
#include <string.h>
#include "mgl_program_backend.h"

/* Objective-C private header declarations restated for C. */
extern Program *mglResolveProgramFromState(GLMContext ctx);
extern void mglWriteProgramMSLDump(Program *program, const char *reason);
extern GLuint mglRendererSafeFramebufferName(GLMContext ctx);

/* The renderer's alpha-should-be-one guard for the ultimate fallback. */
static const int kMglPdVerbosePipelineLogs = 0;

/* C-callable bridge for the stub factory (declared in the .m; log 172). */
extern void *mglRenderPassDiscardStubFragmentFunction(uint32_t valueClass);

static MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo(texture, &info);
    return info;
}

static void *mglPdAttachmentTextureFor(const MGLCommandState *commandState,
                                       uint32_t attachmentKind,
                                       size_t colorIndex)
{
    if (!commandState || !commandState->renderPassStateOwner) return NULL;
    MGLRenderPassState state = {0};
    if (mglRenderGetRenderPassStateOwner(commandState->renderPassStateOwner,
                                         &state) != 0) {
        return NULL;
    }
    switch (mglRenderPassAttachmentClass(attachmentKind)) {
    case 1:
        return (colorIndex < MAX_COLOR_ATTACHMENTS)
                   ? state.color[colorIndex].attachment.texture
                   : NULL;
    case 2:
        return state.depth.attachment.texture;
    case 3:
        return state.stencil.attachment.texture;
    default:
        return NULL;
    }
}

static void *mglPdColorTextureFor(const MGLCommandState *commandState,
                                  size_t colorIndex)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorIndex);
}

static uint32_t mglPdStubFSValueClass(uint32_t fmt)
{
    return mglRenderMetalPixelFormatValueClass(fmt);
}

/* The .m wrapped every cache create in @try/@catch: a throwing create must
 * enter the fallback path instead of escaping into C. */
typedef struct MglPdPsoCreateCtx_t {
    MGLPipelineCachePort cache;
    const MGLRenderPipelineDescriptorState *state;
    void *vertex_function;
    void *fragment_function;
    void **pso_out;
    char *error_text;
    size_t error_capacity;
    int result;
} MglPdPsoCreateCtx;

static int mglPdPsoCreateBody(void *renderer, void *rawCtx)
{
    MglPdPsoCreateCtx *ctx = (MglPdPsoCreateCtx *)rawCtx;
    (void)renderer;
    ctx->result = ctx->cache.create_pso(
        ctx->cache.object, ctx->state, ctx->vertex_function,
        ctx->fragment_function, ctx->pso_out, ctx->error_text,
        ctx->error_capacity);
    return 1;
}

/* Calls the cache's create entry; returns -1 when the cache has no bridge or
 * when the call threw. */
static int mglPdPsoCreateGuarded(void *renderer, MglPdPsoCreateCtx *ctx)
{
    if (!ctx->cache.create_pso) {
        return -1;
    }
    if (!mglPlatformShellGuardedCallCtx(renderer, "pipeline creation",
                                        mglPdPsoCreateBody, ctx, NULL)) {
        return -1;
    }
    return ctx->result;
}

/* -insertPipelineStateIntoCacheWithWords:... */
void mglRenderPassInsertPipelineStateIntoCache(
    void *renderer, const uint64_t *pipelineCacheKeyWords, uint64_t pipelineSig,
    uint64_t vertexSig, const struct MGLRenderPipelineDescriptorState *state,
    void *vertexFunction, void *fragmentFunction, int stateFromCache)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    const MGLPipelineCachePort cache = mglRendererPipelineCachePort(renderer);
    (void)pipelineSig;
    (void)vertexSig;

    if (pipelineCacheKeyWords && cache.state &&
        cache.state->pipelineState) {
        if (cache.store_pipeline) {
            cache.store_pipeline(
                cache.object,
                cache.state->pipelineState, vertexFunction,
                fragmentFunction, pipelineCacheKeyWords);
        }

        /* Cache the descriptor state for future PSO cache misses.  Only cache if
         * state was generated (not from cache). */
        if (!stateFromCache && state && cache.store_descriptor_state) {
            cache.store_descriptor_state(
                cache.object, state, pipelineCacheKeyWords);
        }
    }
}

/* -buildPipelineStateOnCacheMissWithState:... */
int mglRenderPassBuildPipelineStateOnCacheMiss(
    void *renderer, const struct MGLRenderPipelineDescriptorState *pipelineState,
    void *vertexFunction, void *fragmentFunction,
    const uint64_t *pipelineCacheKeyWords, uint64_t pipelineSig,
    uint64_t vertexSig, uint32_t builtColor0Format, uint32_t builtDepthFormat,
    uint32_t builtStencilFormat, unsigned int currentProgramName,
    CFTimeInterval now)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    const MGLPipelineCachePort cache = mglRendererPipelineCachePort(renderer);
    GLMContext ctx = areas.ctx;
    GLMState *state = areas.core && areas.core->activeState
                          ? areas.core->activeState
                          : (ctx ? ctx->active_state : NULL);
    MGLGPURecoveryState *recovery = areas.gpu_recovery;
    Program *currentProgram = mglResolveProgramFromState(ctx);
    Program *currentVertexProgram =
        mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *currentFragmentProgram =
        mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    const GLuint currentFBOName = mglRendererSafeFramebufferName(ctx);

    MGLRenderPipelineDescriptorState finalState = *pipelineState;
    int stateFromCache = 0;

    /* Check descriptor state cache on PSO miss; cache new states for reuse. */
    if (pipelineCacheKeyWords && cache.descriptor_state_for_words) {
        MGLRenderPipelineDescriptorState cachedState = {0};
        if (cache.descriptor_state_for_words(
                cache.object, pipelineCacheKeyWords,
                &cachedState)) {
            /* Descriptor cache hit - reuse cached state instead of regenerating */
            finalState = cachedState;
            stateFromCache = 1;
            static uint64_t s_descriptorCacheHitCount = 0;
            s_descriptorCacheHitCount++;
            if (kMglPdVerbosePipelineLogs && s_descriptorCacheHitCount <= 64ull) {
                fprintf(stderr,
                        "MGL DESCRIPTOR CACHE hit program=%u key=%016llx/%016llx/%016llx (total %llu)\n",
                        (unsigned)currentProgramName,
                        (unsigned long long)pipelineCacheKeyWords[0],
                        (unsigned long long)pipelineCacheKeyWords[5],
                        (unsigned long long)pipelineCacheKeyWords[6],
                        (unsigned long long)s_descriptorCacheHitCount);
            }
        }
    }

    MGL_PERF_INC(g_mglPipelineCacheMissesSinceSwap);
    MGLRenderPipelineDescriptorState successfulState = {0};
    int haveSuccessfulState = 0;
    void *previousPipelineState =
        cache.state ? cache.state->pipelineState : NULL;
    void *compiledPSO = NULL;
    int pipelineReusedPrevious = 0;
    /* Set when the PSO below came from the virtualized-AGX emergency stub
     * rather than from the program's own shaders.  It decides whether the
     * result may be filed in the pipeline cache (see the publish block). */
    int safeFallbackUsed = 0;
    char cppError[512] = {0};

    void *psoPtr = NULL;
    int failed = 0;
    int hookFailure = 0;

    /* @try of the pipeline build; the catch runs the ultimate fallback. */
    {
        static uint64_t s_pipelineCreateBeginCount = 0;
        s_pipelineCreateBeginCount++;
        if (kMglPdVerbosePipelineLogs &&
            (s_pipelineCreateBeginCount <= 128ull ||
             (s_pipelineCreateBeginCount % 500ull) == 0ull)) {
            fprintf(stderr,
                    "MGL PIPELINE CREATE begin program=%u vao=%p fbo=%u\n",
                    (unsigned)currentProgramName, (void *)state->vao,
                    (unsigned)currentFBOName);
        }

        if (kMglPdVerbosePipelineLogs) {
            fprintf(stderr,
                    "MGL INFO: Creating Metal pipeline state with AGX virtualization compatibility...\n");
        }

        /* Test hook (air_pipeline_safe_fallback regression): force the
         * pipeline-creation exception so the safe-fallback branch below is
         * exercised deterministically.  The exception is synthetic, so the C
         * path records the failure and runs the same catch logic. */
        if (mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE")) {
            fprintf(stderr, "MGL TEST: forcing safe-fallback pipeline path\n");
            failed = 1;
            hookFailure = 1;
        } else {
            psoPtr = NULL;
            cppError[0] = '\0';
            MglPdPsoCreateCtx createCtx = {
                .cache = mglRendererPipelineCachePort(renderer),
                .state = &finalState, .vertex_function = vertexFunction,
                .fragment_function = fragmentFunction, .pso_out = &psoPtr,
                .error_text = cppError, .error_capacity = sizeof(cppError),
                .result = 0};
            const int createResult = mglPdPsoCreateGuarded(renderer, &createCtx);
            if (createResult < 0) {
                failed = 1; /* a throwing create takes the fallback path */
            } else if (createResult != 0 || !psoPtr) {
                if (cppError[0]) {
                    fprintf(stderr, "MGL METALCPP PSO fallback: %s\n", cppError);
                }
            } else {
                compiledPSO = psoPtr;
            }
            if (compiledPSO) {
                mglMetalCountCreate(MGLMetalKindPSO);
                successfulState = finalState;
                haveSuccessfulState = 1;
            }
        }

        if (!failed && !compiledPSO) {
            const int isInterfaceMismatch =
                cppError[0] &&
                (strstr(cppError, "mismatching vertex shader output") != NULL ||
                 strstr(cppError, "not written by vertex shader") != NULL);

            if (isInterfaceMismatch) {
                const char *errText = cppError[0] ? cppError : "";
                mglWriteProgramMSLDump(currentVertexProgram, errText);
                if (currentFragmentProgram &&
                    currentFragmentProgram != currentVertexProgram) {
                    mglWriteProgramMSLDump(currentFragmentProgram, errText);
                } else if (!currentVertexProgram) {
                    mglWriteProgramMSLDump(currentProgram, errText);
                }
                const int sameProgram =
                    (cache.state->pipelineProgramName != 0 &&
                     cache.state->pipelineProgramName ==
                         currentProgramName &&
                     cache.state->pipelineVertexFunction ==
                         vertexFunction &&
                     cache.state->pipelineFragmentFunction ==
                         fragmentFunction);
                const int colorCompatible = mglRenderPipelineFormatCompatible(
                    (uint32_t)cache.state->pipelineColor0Format,
                    builtColor0Format) != 0;
                const int depthCompatible = mglRenderPipelineFormatCompatible(
                    (uint32_t)cache.state->pipelineDepthFormat,
                    builtDepthFormat) != 0;
                const int stencilCompatible = mglRenderPipelineFormatCompatible(
                    (uint32_t)cache.state->pipelineStencilFormat,
                    builtStencilFormat) != 0;

                if (previousPipelineState && sameProgram && colorCompatible &&
                    depthCompatible && stencilCompatible) {
                    fprintf(stderr,
                            "MGL WARNING: Interface mismatch for program %u; not reusing previous PSO\n",
                            (unsigned)currentProgramName);
                    compiledPSO = NULL;
                    pipelineReusedPrevious = 0;
                    recovery->interfaceMismatchProgramName = currentProgramName;
                    recovery->interfaceMismatchColor0Format = builtColor0Format;
                    recovery->interfaceMismatchDepthFormat = builtDepthFormat;
                    recovery->interfaceMismatchStencilFormat =
                        builtStencilFormat;
                    recovery->interfaceMismatchStreak = 1u;
                    recovery->interfaceMismatchRetryAfter = now + 0.10;
                    recovery->pipelineRetryAfter =
                        recovery->interfaceMismatchRetryAfter;
                } else {
                    const int sameMismatchSignature =
                        (currentProgramName ==
                             recovery->interfaceMismatchProgramName &&
                         builtColor0Format ==
                             recovery->interfaceMismatchColor0Format &&
                         builtDepthFormat ==
                             recovery->interfaceMismatchDepthFormat &&
                         builtStencilFormat ==
                             recovery->interfaceMismatchStencilFormat);
                    if (sameMismatchSignature) {
                        if (recovery->interfaceMismatchStreak < UINT32_MAX) {
                            recovery->interfaceMismatchStreak++;
                        }
                    } else {
                        recovery->interfaceMismatchStreak = 1;
                        recovery->interfaceMismatchProgramName =
                            currentProgramName;
                        recovery->interfaceMismatchColor0Format =
                            builtColor0Format;
                        recovery->interfaceMismatchDepthFormat =
                            builtDepthFormat;
                        recovery->interfaceMismatchStencilFormat =
                            builtStencilFormat;
                    }

                    /* Exponential backoff: 0.10, 0.20, 0.40, 0.80, 1.60, capped
                     * at 2.00 sec. */
                    const uint32_t cappedShift =
                        (recovery->interfaceMismatchStreak > 5u)
                            ? 4u
                            : (recovery->interfaceMismatchStreak - 1u);
                    double retryDelay = 0.10 * (double)(1u << cappedShift);
                    if (retryDelay > 2.0) {
                        retryDelay = 2.0;
                    }
                    recovery->interfaceMismatchRetryAfter = now + retryDelay;

                    if (recovery->interfaceMismatchStreak <= 5u ||
                        (recovery->interfaceMismatchStreak % 200u) == 0u) {
                        fprintf(stderr,
                                "MGL WARNING: Interface mismatch (program=%u, streak=%u), throttling retries for %.2fs\n",
                                (unsigned)currentProgramName,
                                (unsigned)recovery->interfaceMismatchStreak,
                                retryDelay);
                    }

                    /* Program-level breaker update (ignores attachment
                     * signature). */
                    if (recovery->programMismatchProgramName ==
                        currentProgramName) {
                        if (recovery->programMismatchStreak < UINT32_MAX) {
                            recovery->programMismatchStreak++;
                        }
                    } else {
                        recovery->programMismatchProgramName =
                            currentProgramName;
                        recovery->programMismatchStreak = 1u;
                    }
                    double programDelay =
                        0.25 *
                        (double)(1u << ((recovery->programMismatchStreak > 6u)
                                            ? 6u
                                            : (recovery->programMismatchStreak -
                                               1u)));
                    if (programDelay > 20.0) {
                        programDelay = 20.0;
                    }
                    recovery->programMismatchRetryAfter = now + programDelay;
                    if (recovery->programMismatchStreak <= 8u ||
                        (recovery->programMismatchStreak % 64u) == 0u) {
                        fprintf(stderr,
                                "MGL WARNING: Program %u mismatch breaker set for %.2fs (streak=%u)\n",
                                (unsigned)currentProgramName, programDelay,
                                (unsigned)recovery->programMismatchStreak);
                    }

                    /* Global quarantine for this program to prevent
                     * command-buffer storm. */
                    if (recovery->interfaceMismatchBlockedProgram ==
                        currentProgramName) {
                        if (recovery->interfaceMismatchBlockedStreak <
                            UINT32_MAX) {
                            recovery->interfaceMismatchBlockedStreak++;
                        }
                    } else {
                        recovery->interfaceMismatchBlockedProgram =
                            currentProgramName;
                        recovery->interfaceMismatchBlockedStreak = 1u;
                    }
                    double quarantineDelay = retryDelay * 8.0;
                    if (quarantineDelay < 1.00) quarantineDelay = 1.00;
                    if (quarantineDelay > 15.00) quarantineDelay = 15.00;
                    recovery->interfaceMismatchBlockedUntil =
                        now + quarantineDelay;
                    if (recovery->interfaceMismatchBlockedStreak <= 6u ||
                        (recovery->interfaceMismatchBlockedStreak % 64u) == 0u) {
                        fprintf(stderr,
                                "MGL WARNING: Program %u quarantined for %.2fs after interface mismatch (streak=%u)\n",
                                (unsigned)currentProgramName, quarantineDelay,
                                (unsigned)recovery->interfaceMismatchBlockedStreak);
                    }

                    mglRenderPassInvalidateCurrentPipelineState(
                        renderer, "interface mismatch pipeline failure");
                    recovery->pipelineRetryAfter =
                        (recovery->interfaceMismatchBlockedUntil >
                         recovery->interfaceMismatchRetryAfter)
                            ? recovery->interfaceMismatchBlockedUntil
                            : recovery->interfaceMismatchRetryAfter;
                    mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                    return 0;
                }
            }

            if (!compiledPSO &&
                MGLCapabilityHasBug(areas.core ? &areas.core->capability : NULL,
                                    MGL_BUG_MSL_PIPELINE_REJECTION)) {
                mglRenderPassInvalidateCurrentPipelineState(
                    renderer, "pipeline creation failure");

                /* AGX VIRTUALIZATION FALLBACK: Try with minimal state.  The .m
                 * wrapped this in @try/@catch; the only throwing call is the
                 * cache create, which reports failure instead (see
                 * mglPdTextureReplaceRegion's convention). */
                fprintf(stderr,
                        "MGL INFO: VIRTUALIZED AGX - Trying simplified compilation fallback...\n");

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
                MglPdPsoCreateCtx simpleCtx = {
                    .cache = mglRendererPipelineCachePort(renderer),
                    .state = &simpleState, .vertex_function = vertexFunction,
                    .fragment_function = fragmentFunction, .pso_out = &psoPtr,
                    .error_text = cppError, .error_capacity = sizeof(cppError),
                    .result = 0};
                if (mglPdPsoCreateGuarded(renderer, &simpleCtx) == 0 && psoPtr) {
                    compiledPSO = psoPtr;
                }
                if (compiledPSO) {
                    mglMetalCountCreate(MGLMetalKindPSO);
                    successfulState = simpleState;
                    haveSuccessfulState = 1;
                    builtColor0Format = simpleState.color_format[0];
                    builtDepthFormat = simpleState.depth_format;
                    builtStencilFormat = simpleState.stencil_format;
                }
            }
        }
    }

    if (failed) {
        /* @catch (NSException *exception) of the outer build block.  The .m
         * logged the exception object, its name and its reason; the synthetic
         * test hook reproduces that text verbatim, while a real exception from
         * the cache create only reaches the first line (its text is the
         * shell guard's). */
        if (hookFailure) {
            fprintf(stderr,
                    "MGL CRITICAL: VIRTUALIZED AGX - Metal pipeline creation crashed: synthetic pipeline creation failure (test hook)\n");
            fprintf(stderr,
                    "MGL CRITICAL: Exception name: MGLForcedSafeFallback\n");
            fprintf(stderr,
                    "MGL CRITICAL: Exception reason: synthetic pipeline creation failure (test hook)\n");
        } else {
            fprintf(stderr,
                    "MGL CRITICAL: VIRTUALIZED AGX - Metal pipeline creation crashed\n");
            /* MGL_DUMP_SHADER_DIR: record the vertex layout and attachment
             * formats the failing pipeline was built with, next to the
             * program's dumped shaders. */
            const char *dumpDir = getenv("MGL_DUMP_SHADER_DIR");
            if (dumpDir && dumpDir[0]) {
                char path[1024];
                snprintf(path, sizeof path, "%s/prog%u_pso.txt", dumpDir,
                         (unsigned)currentProgramName);
                FILE *df = fopen(path, "w");
                if (df) {
                    fprintf(df, "attrib_count %u\n",
                            (unsigned)finalState.attrib_count);
                    for (uint32_t i = 0; i < finalState.attrib_count && i < 32;
                         i++) {
                        fprintf(df,
                                "attr %u format %u offset %u buffer %u "
                                "stride %u step %u rate %u\n",
                                (unsigned)i,
                                (unsigned)finalState.attrib_format[i],
                                (unsigned)finalState.attrib_offset[i],
                                (unsigned)finalState.attrib_buffer_index[i],
                                (unsigned)finalState.attrib_stride[i],
                                (unsigned)finalState.attrib_step_function[i],
                                (unsigned)finalState.attrib_step_rate[i]);
                    }
                    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                        fprintf(df, "color %d format %u mask %u\n", i,
                                (unsigned)finalState.color_format[i],
                                (unsigned)finalState.color_write_mask[i]);
                    }
                    fprintf(df, "depth %u stencil %u samples %u\n",
                            (unsigned)finalState.depth_format,
                            (unsigned)finalState.stencil_format,
                            (unsigned)finalState.raster_sample_count);
                    fclose(df);
                }
            }
            if (cppError[0]) {
                fprintf(stderr, "MGL CRITICAL: %s\n", cppError);
            }
        }
        const int forceSafeFallback =
            mgl_env_flag_enabled("MGL_FORCE_SAFE_FALLBACK_PIPELINE");
        if (!MGLCapabilityHasBug(areas.core ? &areas.core->capability : NULL,
                                 MGL_BUG_MSL_PIPELINE_REJECTION) &&
            !forceSafeFallback) {
            mglRenderPassInvalidateCurrentPipelineState(
                renderer, "pipeline creation exception");
            recovery->pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return 0;
        }

        /* VIRTUALIZED AGX ULTIMATE FALLBACK: Create minimal safe pipeline */
        fprintf(stderr,
                "MGL INFO: VIRTUALIZED AGX - Creating ultimate fallback pipeline for virtualization safety\n");

        MGLRenderPipelineDescriptorState safeState = {0};
        safeState.color_count = MAX_COLOR_ATTACHMENTS;
        safeState.rasterization_enabled = 1;
        uint32_t safeColor0Format = (uint32_t)finalState.color_format[0];
        if (areas.command && mglPdColorTextureFor(areas.command, 0)) {
            safeColor0Format =
                mglPdTextureInfo(mglPdColorTextureFor(areas.command, 0))
                    .pixel_format;
        } else if (mglRendererDrawableTexture(renderer)) {
            safeColor0Format =
                mglPdTextureInfo(mglRendererDrawableTexture(renderer))
                    .pixel_format;
        }
        safeColor0Format = mglRenderColorFormatOrBGRA(safeColor0Format);
        safeState.color_format[0] = safeColor0Format;
        safeState.depth_format = finalState.depth_format;
        safeState.stencil_format = finalState.stencil_format;

        /* VS from the precompiled safe_fallback aux asset.  FS reuses the
         * discard stub helper so int/uint color0 gets a matching zero
         * output. */
        const MGLAuxShaderAsset *safe = mglAuxShaderAssetFind("safe_fallback");
        void *safeVS = NULL;
        void *unusedFS = NULL;
        char libError[512] = {0};
        const uint32_t safeClass = mglPdStubFSValueClass(safeColor0Format);
        void *safeFSFunction = mglRenderPassDiscardStubFragmentFunction(safeClass);
        if (!safe || !safe->data || safe->size == 0 ||
            mglRenderCreateAuxFunctions(safe->data, safe->size, safe->hash,
                                        "mgl_safe_fallback_vs",
                                        "mgl_safe_fallback_fs", &safeVS,
                                        &unusedFS, libError,
                                        sizeof(libError)) != 0 ||
            !safeVS || !safeFSFunction) {
            fprintf(stderr,
                    "MGL CRITICAL: safe fallback asset unavailable program=%u color0=%lu class=%u hash=0x%016llx error=%s\n",
                    (unsigned)currentProgramName, (unsigned long)safeColor0Format,
                    (unsigned)safeClass,
                    safe ? (unsigned long long)safe->hash : 0ull,
                    libError[0] ? libError : "asset or stub FS missing");
            if (safeVS) mglReleaseMetalObjNoNull(safeVS);
            if (unusedFS) mglReleaseMetalObjNoNull(unusedFS);
        } else {
            if (unusedFS) mglReleaseMetalObjNoNull(unusedFS);
            psoPtr = NULL;
            cppError[0] = '\0';
            MglPdPsoCreateCtx safeCtx = {
                .cache = mglRendererPipelineCachePort(renderer),
                .state = &safeState, .vertex_function = safeVS,
                .fragment_function = safeFSFunction, .pso_out = &psoPtr,
                .error_text = cppError, .error_capacity = sizeof(cppError),
                .result = 0};
            if (mglPdPsoCreateGuarded(renderer, &safeCtx) == 0 && psoPtr) {
                compiledPSO = psoPtr;
            }
            mglReleaseMetalObjNoNull(safeVS);
        }
        if (compiledPSO) {
            mglMetalCountCreate(MGLMetalKindPSO);
            successfulState = safeState;
            haveSuccessfulState = 1;
            safeFallbackUsed = 1;
            builtColor0Format = safeState.color_format[0];
            builtDepthFormat = safeState.depth_format;
            builtStencilFormat = safeState.stencil_format;
            fprintf(stderr,
                    "MGL INFO: VIRTUALIZED AGX - Safe fallback pipeline created successfully\n");
        }

        if (!compiledPSO) {
            fprintf(stderr,
                    "MGL CRITICAL: VIRTUALIZED AGX - All pipeline creation attempts failed, disabling rendering\n");
            mglRenderPassInvalidateCurrentPipelineState(
                renderer, "all pipeline fallbacks failed");
            recovery->pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
            mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return 0;
        }
    }

    if (!compiledPSO) {
        fprintf(stderr,
                "MGL ERROR: Failed to create pipeline state: %s\n",
                cppError[0] ? cppError : "unknown error");
        fprintf(stderr,
                "MGL WARNING: Skipping draw for this pipeline build failure; will retry later\n");
        mglRenderPassInvalidateCurrentPipelineState(
            renderer, "pipeline state is nil after creation");
        recovery->pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
        mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
        return 0;
    }

    if (kMglPdVerbosePipelineLogs) {
        fprintf(stderr, "MGL PIPELINE CREATE success pipeline=%p\n", compiledPSO);
        fprintf(stderr,
                "MGL INFO: Pipeline state created successfully\n");
    }
    /* Publish the compile result to the shared state (the .m re-acquired the
     * lock here; METAL_LOCK is only a GL-thread assertion). */
    if (!pipelineReusedPrevious && haveSuccessfulState) {
        /* Clear interface-mismatch breaker after a real compile. */
        recovery->interfaceMismatchStreak = 0;
        recovery->interfaceMismatchProgramName = 0;
        recovery->interfaceMismatchColor0Format = mglRenderInvalidPixelFormat();
        recovery->interfaceMismatchDepthFormat = mglRenderInvalidPixelFormat();
        recovery->interfaceMismatchStencilFormat = mglRenderInvalidPixelFormat();
        recovery->interfaceMismatchRetryAfter = 0.0;
        if (cache.activate) {
            cache.activate(
                cache.object, compiledPSO, builtColor0Format,
                builtDepthFormat, builtStencilFormat, currentProgramName,
                vertexFunction, fragmentFunction);
        }
        if (safeFallbackUsed) {
            /* The emergency stub was accepted as this draw's pipeline, but it
             * must not be filed under the program's real cache key.  A cache
             * entry here outlives the condition that produced the fallback:
             * the next draw takes the PSO-cache hit path, re-activates the
             * stub and never retries the real compile, so a one-off failure
             * would render as a permanently blank program until resetCaches.
             * Leaving the key unpopulated hands the retry back to the next
             * pipeline sync (which runs as soon as program/VAO/FBO/render
             * state changes again) and keeps the failure loud instead of
             * papering over it with a cached stub. */
            fprintf(stderr,
                    "MGL WARNING: safe-fallback pipeline for program=%u not "
                    "cached; the real pipeline will be retried on the next "
                    "pipeline sync\n",
                    (unsigned)currentProgramName);
        } else {
            /* Archive lookup/add is owned by PipelineCacheOwner's C++ builder. */
            mglRenderPassInsertPipelineStateIntoCache(
                renderer, pipelineCacheKeyWords, pipelineSig, vertexSig,
                &successfulState, vertexFunction, fragmentFunction,
                stateFromCache);
        }
        if (recovery->programMismatchProgramName == currentProgramName) {
            recovery->programMismatchProgramName = 0;
            recovery->programMismatchRetryAfter = 0.0;
            recovery->programMismatchStreak = 0u;
        }
        if (recovery->interfaceMismatchBlockedProgram == currentProgramName) {
            recovery->interfaceMismatchBlockedProgram = 0;
            recovery->interfaceMismatchBlockedUntil = 0.0;
            recovery->interfaceMismatchBlockedStreak = 0u;
        }
    }

    return 1;
}

/* === Pipeline Sync domain (log 189) =======================================
 * -syncPipelineStateWithDeferredBufferMap: moved verbatim from
 * MGLRenderer+RenderPass.m (301 lines / 11 syntax).  The only Objective-C left
 * in it was the two Metal-object locals, the cache lookup and the hit log:
 *   * psoVertexFunction / psoFragmentFunction become plain void * (the .m
 *     borrowed them with __bridge);
 *   * the cache lookup travels through a new areas bridge
 *     (pipeline_cache_lookup_pipeline), like the other four cache bridges;
 *   * the cache-hit log drops the NSString formatting and prints the same key
 *     triple with fprintf (it is behind the same always-0 verbosity flag).
 * The two file-local statics the method used - mglGeometryPassthroughLayerStride
 * and mglGeometryPipelineFunctionKey - are plain C and moved with it.
 */

/* mglGeometryPassthroughLayerStride() from MGLRenderer+RenderPass.m: MSAA
 * array textures are represented by a 2D array whose physical slices are laid
 * out as [gl_layer][sample] with a fixed eight-slice stride.  A layered render
 * pass therefore needs to translate the logical GL layer before Metal consumes
 * [[render_target_array_index]].  Keep this decision in the render-pass domain:
 * ordinary 2D arrays remain a one-to-one map and non-layered
 * framebufferTextureLayer attachments keep their fixed slice. */
static uint32_t mglPdGeometryPassthroughLayerStride(GLMContext context)
{
    if (!context || !context->active_state) {
        return 1u;
    }
    return mglFramebufferMultisampleLayerStride(
        context->active_state->framebuffer);
}

/* mglGeometryPipelineFunctionKey() from MGLRenderer+RenderPass.m: the backend
 * keeps one passthrough function per kind.  Include the render target layer
 * convention in the key so switching between ordinary and emulated-MSAA layered
 * FBOs cannot reuse a function compiled for the other convention. */
static uint64_t mglPdGeometryPipelineFunctionKey(
    const Program *vertexProgram, const Program *geometryProgram,
    uint32_t layerStride)
{
    uint64_t hash = 1469598103934665603ull;
    hash = mglHashStepU64(hash,
                          vertexProgram ? mglProgramBackendView(vertexProgram)->pipeline_cache_instance_id : 0u);
    hash = mglHashStepU64(hash,
                          vertexProgram ? mglProgramBackendView(vertexProgram)->pipeline_cache_generation : 0u);
    hash = mglHashStepU64(hash,
                          geometryProgram ? mglProgramBackendView(geometryProgram)->pipeline_cache_instance_id : 0u);
    hash = mglHashStepU64(hash,
                          geometryProgram ? mglProgramBackendView(geometryProgram)->pipeline_cache_generation : 0u);
    return mglHashStepU64(hash, layerStride);
}

/* The one-line predicates of MGLRenderer_Private.h, in C (same shape as the
 * mglSe* / mglBatch* twins). */
static int mglPdBindingStateIsValid(void *owner)
{
    uint32_t valid = 0;
    return owner && mglRenderBindingGetValid(owner, &valid) == 0 && valid;
}

static int mglPdBindingStatePipelineMatches(void *owner, void *pipeline)
{
    void *current = NULL;
    return owner &&
           mglRenderBindingGetPipelineState(owner, &current) == 0 &&
           current == pipeline;
}

/* -syncPipelineStateWithDeferredBufferMap: */
int mglRenderPassSyncPipelineState(void *renderer,
                                   int deferredBufferMapForPipelineBuild)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    const MGLPipelineCachePort cache = mglRendererPipelineCachePort(renderer);
    GLMContext ctx = areas.ctx;
    GLMState *state = areas.core && areas.core->activeState
                          ? areas.core->activeState
                          : (ctx ? ctx->active_state : NULL);
    MGLGPURecoveryState *recovery = areas.gpu_recovery;
    const MGLPipelineCacheState *cacheState = cache.state;
    /* areas.binding_state_owner is the ADDRESS of the owner slot (rule 59). */
    void *bindingOwner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    const MGLTessellationState *tessellation = areas.tessellation;
    const MGLGeometryState *geometry = areas.geometry;
    void *currentPipelineState = cacheState ? cacheState->pipelineState : NULL;

    /* Force a rebind of the pipeline state on the next setRenderPipelineState
     * call.  Dirty program/VAO/FBO/render-state may rebuild or reuse the
     * pipeline, but the encoder still needs the binding re-issued.
     *
     * Task 5 gated fast path: when MGL_PSO_DEDUP is enabled (default ON)
     * and the render encoder is unchanged (the C++ binding cache is valid) and
     * the resolved pipeline state pointer is identical to the previously bound
     * state matches the C++ binding cache, the nil assignment is skipped.  This
     * allows the dedup check in processGLStateLocked:'s setRenderPipelineState:
     * path to recognize the encoder already has the correct PSO bound and skip
     * the redundant MTL call.  If any condition is false, the original
     * conservative nil assignment executes. */
    if (cacheState && cacheState->psoDedupEnabled &&
        mglPdBindingStateIsValid(bindingOwner) &&
        mglPdBindingStatePipelineMatches(bindingOwner, currentPipelineState)) {
        MGL_PERF_INC(g_mglPSODedupHitsSinceSwap);
    } else {
        mglRenderBindingSetPipelineState(bindingOwner, NULL);
        MGL_PERF_INC(g_mglPSODedupMissesSinceSwap);
    }
    CFTimeInterval now = CFAbsoluteTimeGetCurrent();
    int skipPipelineBuild = 0;
    Program *currentVertexProgram =
        (tessellation && tessellation->nativeTESActive)
            ? tessellation->nativeTESProgram
            : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *currentFragmentProgram =
        mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    GLuint currentProgramName = mglCurrentRenderProgramKey(ctx);
    GLuint currentFBOName = mglRendererSafeFramebufferName(ctx);

    /* Program-level breaker (independent of render-pass signature) to avoid
     * mismatch storms where color/depth/stencil signatures keep changing. */
    if (currentPipelineState != NULL && currentProgramName != 0 &&
        recovery &&
        currentProgramName == recovery->programMismatchProgramName &&
        now < recovery->programMismatchRetryAfter) {
        static uint64_t s_programMismatchSkipCount = 0;
        s_programMismatchSkipCount++;
        if (s_programMismatchSkipCount <= 16 ||
            (s_programMismatchSkipCount % 1000ull) == 0ull) {
            double remaining = recovery->programMismatchRetryAfter - now;
            if (remaining < 0.0) remaining = 0.0;
            fprintf(stderr,
                    "MGL WARNING: Program-level mismatch breaker active (program=%u, %.2fs remaining), skipping draw\n",
                    (unsigned)currentProgramName, remaining);
        }
        mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
        return 0;
    }

    if (recovery && now < recovery->pipelineRetryAfter) {
        int retryAppliesToCurrentProgram =
            (currentProgramName != 0 &&
             (currentProgramName == recovery->interfaceMismatchProgramName ||
              currentProgramName == recovery->programMismatchProgramName ||
              currentProgramName == recovery->interfaceMismatchBlockedProgram));

        if (retryAppliesToCurrentProgram) {
            if (currentPipelineState) {
                mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                /* Keep existing pipeline, but do not early-return before
                 * setRenderPipelineState. */
                skipPipelineBuild = 1;
            } else {
                recovery->pipelineRetryAfter = 0.0;
                recovery->programMismatchRetryAfter = 0.0;
                recovery->interfaceMismatchRetryAfter = 0.0;
            }
        } else {
            static uint64_t s_retryBypassCount = 0;
            s_retryBypassCount++;
            if (s_retryBypassCount <= 16 ||
                (s_retryBypassCount % 1000ull) == 0ull) {
                fprintf(stderr,
                        "MGL PIPELINE RETRY bypass global retry for unrelated program=%u mismatchProgram=%u blockedProgram=%u\n",
                        (unsigned)currentProgramName,
                        (unsigned)recovery->interfaceMismatchProgramName,
                        (unsigned)recovery->interfaceMismatchBlockedProgram);
            }
        }
    }

    if (!skipPipelineBuild) {
        /* Build the only renderer pipeline representation: C ABI value-state. */
        MGLRenderPipelineDescriptorState psoState = {0};
        void *psoVertexFunction = NULL;
        void *psoFragmentFunction = NULL;
        uint32_t builtColor0Format = mglRenderInvalidPixelFormat();
        uint32_t builtDepthFormat = mglRenderInvalidPixelFormat();
        uint32_t builtStencilFormat = mglRenderInvalidPixelFormat();

        mglRendererUpdateBlendStateCache(renderer);
        mglClearRendererDirtyBits(state, DIRTY_ALPHA_STATE);
        if (getenv("MGL_TOPO_TRACE") != NULL) {
            fprintf(stderr,
                    "MGLTOPO tessCompute=%d active=%d prog=%p topology=%u\n",
                    (int)((tessellation && tessellation->tessComputeActive) ? 1 : 0),
                    (int)((tessellation && tessellation->tessVertexRenderActive) ? 1 : 0),
                    (void *)(tessellation ? tessellation->tessComputeProgram : NULL),
                    (unsigned)psoState.input_primitive_topology);
            fflush(stderr);
        }
        MGLRenderPassPipelineFunctions psoFunctions = { NULL, NULL };
        if (!mglRenderPassGeneratePipelineDescriptorState(
                renderer, &psoState, &psoFunctions)) {
            fprintf(stderr,
                    "MGL PIPELINE CREATE fail error=generatePipelineDescriptorState returned NO\n");
            mglRenderPassInvalidateCurrentPipelineState(
                renderer, "pipeline descriptor failure");
            if (recovery) {
                recovery->pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
            }
            mglMarkRendererDirtyBits(state,
                                     DIRTY_PROGRAM | DIRTY_VAO |
                                     DIRTY_FBO | DIRTY_RENDER_STATE);
            return 0;
        }
        /* Borrowed Metal functions: the .m took them with __bridge so ARC owns
         * the +1 it releases (the program/cache keeps them alive); in C they
         * stay borrowed raw handles. */
        psoVertexFunction = psoFunctions.vertex_function;
        psoFragmentFunction = psoFunctions.fragment_function;
        builtColor0Format = psoState.color_format[0];
        builtDepthFormat = psoState.depth_format;
        builtStencilFormat = psoState.stencil_format;

        /* Circuit breaker for repeated VS/FS interface mismatch. */
        if (recovery && now < recovery->interfaceMismatchRetryAfter &&
            currentProgramName == recovery->interfaceMismatchProgramName &&
            builtColor0Format == recovery->interfaceMismatchColor0Format &&
            builtDepthFormat == recovery->interfaceMismatchDepthFormat &&
            builtStencilFormat == recovery->interfaceMismatchStencilFormat) {
            mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
            return 0;
        }

        int hasPipelineCacheKey = 0;
        int pipelineResolvedFromCache = 0;
        uint64_t pipelineSig = 0;
        uint64_t vertexSig = 0;
        /* Function-scope key words: filled inside the lookup block below, read
         * by the miss path after it.  Only meaningful when
         * currentProgramName != 0. */
        uint64_t keyWords[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS] = {0};

        if (!pipelineResolvedFromCache && currentProgramName != 0) {
            pipelineSig = mglPipelineDescriptorSignatureFromState(&psoState);
            vertexSig = mglVertexDescriptorSignatureFromState(&psoState);

            /* Keep descriptor signatures and linked Program identities
             * lossless.  GL names can be reused and a Program can relink
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
                                 | ((tessellation && tessellation->nativeTESActive) ? (1ull << 23) : 0ull)
                                 | ((tessellation && tessellation->tessVertexCaptureActive) ? (1ull << 22) : 0ull)
                                 | ((geometry && geometry->expansionActive) ? (1ull << 21) : 0ull)
                                 | ((tessellation && tessellation->cullDistanceCaptureActive) ? (1ull << 20) : 0ull)
                                 | ((tessellation && tessellation->tessComputeActive) ? (1ull << 19) : 0ull)
                                 | ((tessellation && tessellation->tessVertexRenderActive) ? (1ull << 18) : 0ull));
            uint64_t vertexInstance = currentVertexProgram
                ? mglProgramBackendView(currentVertexProgram)->pipeline_cache_instance_id : 0u;
            if (geometry && geometry->expansionActive && geometry->program) {
                /* The raster vertex function is generated from both the real
                 * VS/FS interface and the GS output record.  Fold both program
                 * identities plus the emulated-MS layer convention into the key
                 * so an old PTVS/PSO cannot be reused after a GS or framebuffer
                 * change. */
                vertexInstance = mglPdGeometryPipelineFunctionKey(
                    currentVertexProgram, geometry->program,
                    mglPdGeometryPassthroughLayerStride(ctx));
            }
            uint64_t vertexGeneration = currentVertexProgram
                ? mglProgramBackendView(currentVertexProgram)->pipeline_cache_generation : 0u;
            uint64_t fragmentInstance = currentFragmentProgram
                ? mglProgramBackendView(currentFragmentProgram)->pipeline_cache_instance_id : 0u;
            uint64_t fragmentGeneration = currentFragmentProgram
                ? mglProgramBackendView(currentFragmentProgram)->pipeline_cache_generation : 0u;
            keyWords[0] = primaryKey;
            keyWords[1] = vertexInstance;
            keyWords[2] = vertexGeneration;
            keyWords[3] = fragmentInstance;
            keyWords[4] = fragmentGeneration;
            keyWords[5] = pipelineSig;
            keyWords[6] = vertexSig;
            /* Hit path uses the reusable zero-alloc query key.  The key is only
             * valid for lookups; the miss path below allocates a fresh key for
             * the store/compile path so overwriteWords: cannot corrupt cache
             * dictionaries. */
            hasPipelineCacheKey = 1;

            /* Two-level cache lookup:
             * Level 1: PSO cache (fastest - compiled pipeline ready to use)
             * Level 2: Descriptor cache (fast - skip expensive descriptor
             * regeneration).  On double miss: regenerate descriptor + compile
             * PSO. */
            void *cachedPipeline = NULL;
            void *cachedVertexFunction = NULL;
            void *cachedFragmentFunction = NULL;
            int cachedFunctionMetadataPresent = 0;
            if (cache.lookup_pipeline) {
                cachedFunctionMetadataPresent =
                    cache.lookup_pipeline(
                        cache.object, keyWords, &cachedPipeline,
                        &cachedVertexFunction, &cachedFragmentFunction);
            }
            if (cachedPipeline) {
                /* PSO cache hit - fastest path */
                static uint64_t s_pipelineCacheHitCount = 0;
                s_pipelineCacheHitCount++;
                MGL_PERF_INC(g_mglPipelineCacheHitsSinceSwap);
                if (kMglPdVerbosePipelineLogs &&
                    (s_pipelineCacheHitCount <= 128ull ||
                     (s_pipelineCacheHitCount % 1000ull) == 0ull)) {
                    fprintf(stderr,
                            "MGL PIPELINE CACHE hit program=%u vao=%p fbo=%u key=%016llx/%016llx/%016llx\n",
                            (unsigned)currentProgramName, (void *)state->vao,
                            (unsigned)currentFBOName,
                            (unsigned long long)keyWords[0],
                            (unsigned long long)keyWords[5],
                            (unsigned long long)keyWords[6]);
                }

                if (cache.activate) {
                    cache.activate(
                        cache.object, cachedPipeline,
                        builtColor0Format, builtDepthFormat, builtStencilFormat,
                        currentProgramName,
                        cachedFunctionMetadataPresent ? cachedVertexFunction
                                                      : psoVertexFunction,
                        cachedFunctionMetadataPresent ? cachedFragmentFunction
                                                      : psoFragmentFunction);
                }
                pipelineResolvedFromCache = 1;
                /* Hit path deliberately skips the LRU touch: touching would
                 * require copying the query-keyed object that must never enter
                 * the LRU (see pipelineQueryKeyForWords:), reintroducing the
                 * per-draw alloc this avoids.  Mirrors the depth-stencil cache
                 * policy. */

                /* Mirror successful compile-side breaker resets. */
                if (recovery) {
                    recovery->interfaceMismatchStreak = 0;
                    recovery->interfaceMismatchProgramName = 0;
                    recovery->interfaceMismatchColor0Format = mglRenderInvalidPixelFormat();
                    recovery->interfaceMismatchDepthFormat = mglRenderInvalidPixelFormat();
                    recovery->interfaceMismatchStencilFormat = mglRenderInvalidPixelFormat();
                    recovery->interfaceMismatchRetryAfter = 0.0;
                    if (recovery->programMismatchProgramName == currentProgramName) {
                        recovery->programMismatchProgramName = 0;
                        recovery->programMismatchRetryAfter = 0.0;
                        recovery->programMismatchStreak = 0u;
                    }
                    if (recovery->interfaceMismatchBlockedProgram == currentProgramName) {
                        recovery->interfaceMismatchBlockedProgram = 0;
                        recovery->interfaceMismatchBlockedUntil = 0.0;
                        recovery->interfaceMismatchBlockedStreak = 0u;
                    }
                }
            }
        }

        /* PROPER AGX VIRTUALIZATION COMPATIBILITY: Fix root cause while
         * maintaining Metal functionality. */
        if (!pipelineResolvedFromCache) {
            /* Compile/store path needs its own key object: the reusable query
             * key words are overwritten on every lookup and must never be
             * retained by the cache dictionaries/LRU.  One heap allocation on a
             * cache miss is negligible against the PSO compile itself. */
            const uint64_t *storeKeyWords = hasPipelineCacheKey ? keyWords : NULL;
            return mglRenderPassBuildPipelineStateOnCacheMiss(
                       renderer, &psoState, psoVertexFunction,
                       psoFragmentFunction, storeKeyWords, pipelineSig,
                       vertexSig, builtColor0Format, builtDepthFormat,
                       builtStencilFormat, currentProgramName, now) != 0;
        }

        if (deferredBufferMapForPipelineBuild && currentPipelineState != NULL) {
            RETURN_FALSE_ON_FAILURE(mglRendererMapBuffersToMTL(renderer));
            deferredBufferMapForPipelineBuild = 0;
        }

        mglClearRendererDirtyBits(state, DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
    }

    return 1;
}
