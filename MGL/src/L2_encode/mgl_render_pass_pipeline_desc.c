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
#include "mgl_renderer_state_areas.h"
#include "mgl_renderer_binding_ops.h"      /* mglRendererBindingInvalidateLastBoundState */
#include "mgl_trace_strategy.h"         /* mglClearFragmentTraceBindingsForRenderer */

#include <stdio.h>
#include <stdarg.h>   /* va_list for the GLSL source builder */
#include <stdlib.h>   /* malloc / realloc / free */
#include <string.h>   /* memcpy / strlen */
#include <math.h>     /* lrintf */
#include "mgl_readback_policy.h"
#include "mgl_program_backend.h"
#include "mgl_texture_backend.h"
#include "mgl_raster_orientation_state.h"

uint32_t mglPdGeometryPassthroughLayerStride(GLMContext ctx);
uint64_t mglPdGeometryPassthroughCacheKey(const Program *program, uint32_t layerStride);


/* MGLRenderer+RenderPass_Private.h: `static const BOOL
 * kMGLVerbosePipelineLogs = NO;` - kept so the branches stay verbatim. */
static const int kMglPdVerbosePipelineLogs = 0;

/* -generatePipelineDescriptorState:vertexFunction:fragmentFunction:. */
int mglRenderPassGeneratePipelineDescriptorState(
    void *renderer, void *state, MGLRenderPassPipelineFunctions *functions_out)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    const MGLPipelineCachePort cache = mglRendererPipelineCachePort(renderer);
    GLMContext ctx = areas.ctx;
    MGLTessellationState *tess = areas.tessellation;
    MGLGeometryState *geom = areas.geometry;
    MGLCommandState *commandState = areas.command;
    GLMState *glState = mglPdState(&areas);
    MGLRenderPipelineDescriptorState *desc =
        (MGLRenderPipelineDescriptorState *)state;

    if (!ctx) {
        fprintf(stderr, "MGL PIPELINE DESC fail: context is NULL\n");
        return 0;
    }
    if (!desc || !functions_out) {
        fprintf(stderr, "MGL PIPELINE DESC fail: bad out args\n");
        return 0;
    }
    functions_out->vertex_function = NULL;
    functions_out->fragment_function = NULL;

    const int nativeTES = tess->nativeTESActive;
    const int tessVertexCapture = tess->tessVertexCaptureActive;
    const int cullDistanceCapture = tess->cullDistanceCaptureActive;
    const int geometryExpansion = geom->expansionActive;
    const int tessCompute = tess->tessComputeActive;
    const int tessVertexRender = tessCompute && tess->tessVertexRenderActive;
    const int tessVertex = tessVertexRender;
    const int vertexStage = nativeTES ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    Program *vertexProgram = nativeTES ? tess->nativeTESProgram
        : tessVertex ? tess->tessComputeProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *fragmentProgram = (tessVertexCapture || cullDistanceCapture)
        ? NULL : mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    const GLuint renderProgramKey = mglCurrentRenderProgramKey(ctx);
    const GLuint vertexProgramName = vertexProgram ? vertexProgram->name : 0u;
    const GLuint fragmentProgramName =
        fragmentProgram ? fragmentProgram->name : 0u;
    const int rasterizerDiscard =
        (tessVertexCapture || cullDistanceCapture ||
         glState->caps.rasterizer_discard) ? 1 : 0;

    /* A missing fragment stage is legal; only the vertex stage is mandatory.
     * AGX still needs a stub FS so Metal rasterization stays on — otherwise
     * vertex device-buffer stores (atomics / imageStore) are dropped. */
    if (!vertexProgram) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: missing stage program key=%u vs=%p fs=%p current=%u pipeline=%u\n",
                (unsigned)renderProgramKey, (void *)vertexProgram,
                (void *)fragmentProgram, (unsigned)glState->program_name,
                (unsigned)glState->var.program_pipeline_binding);
        return 0;
    }

    if (kMglPdVerbosePipelineLogs) {
        fprintf(stderr,
                "MGL PIPELINE DESC begin key=%u vsProgram=%u fsProgram=%u\n",
                (unsigned)renderProgramKey, (unsigned)vertexProgramName,
                (unsigned)fragmentProgramName);
    }

    if (!mglRenderPassBindMTLProgram(renderer, vertexProgram)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: bindMTLProgram failed for VS program=%u\n",
                (unsigned)vertexProgramName);
        return 0;
    }
    if (fragmentProgram && fragmentProgram != vertexProgram &&
        !mglRenderPassBindMTLProgram(renderer, fragmentProgram)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: bindMTLProgram failed for FS program=%u\n",
                (unsigned)fragmentProgramName);
        return 0;
    }

    /* tessVertexRender uses the TES program as the Metal vertex function
     * (see tessPassthroughFunction below).  Its shader lives in the
     * TESS_EVALUATION slot — a separable TES-only program has no VS slot,
     * so checking shader_slots[VERTEX] aborted SSO tess draws with
     * INVALID_OPERATION (CTS shader_subroutine.subroutines_with_separate_
     * shader_objects). */
    const int vertexSlotStage =
        (nativeTES || tessVertex) ? _TESS_EVALUATION_SHADER : vertexStage;
    Shader *vertex_shader = vertexProgram->shader_slots[vertexSlotStage];
    Shader *fragment_shader =
        fragmentProgram ? fragmentProgram->shader_slots[_FRAGMENT_SHADER] : NULL;
    if (!vertex_shader) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: missing shaders key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)\n",
                (unsigned)renderProgramKey, (unsigned)vertexProgramName,
                (unsigned)fragmentProgramName, (void *)vertex_shader,
                (void *)fragment_shader);
        return 0;
    }

    void *geometryPassthroughFunction = NULL;
    if (geometryExpansion && geom->program) {
        const uint32_t layerStride =
            mglPdGeometryPassthroughLayerStride(ctx);
        const uint64_t passthroughKey =
            mglPdGeometryPassthroughCacheKey(geom->program, layerStride);
        (void)mglRendererBackendGetPassthroughFunction(
            areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
            passthroughKey, &geometryPassthroughFunction);
    }
    void *tessPassthroughFunction = NULL;
    if (tessCompute && tess->tessComputeProgram) {
        /* A TES-vertex program is its own vertex function (the expanded
         * stream rasterizes directly); only the record-passthrough of the
         * compute expansion uses the generated slot-28 reader. */
        if (tess->tessVertexRenderActive) {
            tessPassthroughFunction =
                mglProgramModuleView(tess->tessComputeProgram,
                                     _TESS_EVALUATION_SHADER)->mtl_function;
        } else {
            (void)mglRendererBackendGetPassthroughFunction(
                areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
                mglProgramBackendView(tess->tessComputeProgram)->pipeline_cache_instance_id,
                &tessPassthroughFunction);
        }
    }
    void *vertexFunctionPtr = geometryExpansion
        ? geometryPassthroughFunction
        : tessCompute
        ? tessPassthroughFunction
        : cullDistanceCapture
        ? mglProgramModuleView(vertexProgram, _VERTEX_SHADER)->mtl_cull_capture_function
        : tessVertexCapture
        ? mglProgramModuleView(vertexProgram, _VERTEX_SHADER)->mtl_tess_capture_function
        : mglProgramModuleView(vertexProgram, vertexStage)->mtl_function;
    void *vertexFunction = vertexFunctionPtr;

    const struct MGLShaderModuleBackend *fragmentModule = fragmentProgram
        ? mglProgramModuleView(fragmentProgram, _FRAGMENT_SHADER) : NULL;
    void *fragmentFunction = fragmentModule ? fragmentModule->mtl_function : NULL;
    if (fragmentModule &&
        (geometryExpansion || tessCompute || tessVertexRender) &&
        fragmentModule->mtl_tess_capture_function) {
        /* FS stores its GS/TES primitive-id carrier in the per-stage
         * alternate-function slot; direct VS/native-TES keeps mtl_function. */
        fragmentFunction = fragmentModule->mtl_tess_capture_function;
    }
    /* AGX drops vertex device-buffer stores when Metal rasterization is off
     * (RASTERIZER_DISCARD, tess/cull VS capture, and VS-only programs such as
     * sparse_buffer case_e). Always attach a stub FS so rasterization stays
     * on; color write masks are cleared below when there is no real FS. */
    int used_fs_stub = 0;
    if (!fragmentFunction) {
        /* Metal validates the stub FS output against color attachment 0's
         * format: float4 stubs are rejected by integer-format targets.
         * Resolve the format early (read-only; the FBO walk below re-binds
         * the same textures) and pick the matching zero-return variant. */
        uint32_t stubColor0 = mglRenderInvalidPixelFormat();
        if (glState->framebuffer) {
            Framebuffer *stubFbo = glState->framebuffer;
            for (int i = 0; i < glState->max_color_attachments; i++) {
                if (!stubFbo->color_attachments[i].texture) {
                    if ((stubFbo->color_attachment_bitfield >> (i + 1)) == 0) {
                        break;
                    }
                    continue;
                }
                Texture *stubTex = mglRendererAttachmentTextureFor(
                    ctx, &stubFbo->color_attachments[i]);
                if (stubTex && mglTextureMtl(stubTex)) {
                    stubColor0 = mtlPixelFormatForGLTex(stubTex);
                    if (!mglRenderPixelFormatIsInvalid(stubColor0)) {
                        break;
                    }
                }
            }
        } else if (commandState && mglPdColorTextureFor(commandState, 0)) {
            stubColor0 = mglPdTextureInfo(
                mglPdColorTextureFor(commandState, 0)).pixel_format;
        } else if (mglRendererDrawableTexture(renderer)) {
            stubColor0 = mglPdTextureInfo(
                mglRendererDrawableTexture(renderer)).pixel_format;
        } else {
            stubColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        fragmentFunction = mglRenderPassDiscardStubFragmentFunction(
            mglPdStubFSValueClass(stubColor0));
        used_fs_stub = fragmentFunction ? 1 : 0;
    }
    if (kMglPdVerbosePipelineLogs) {
        fprintf(stderr, "MGL PIPELINE DESC vs=%p fs=%p\n", vertexFunction,
                fragmentFunction);
    }
    if (!mglRenderPipelineFunctionsReady(vertexFunction ? 1 : 0,
                                         fragmentFunction ? 1 : 0,
                                         rasterizerDiscard)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: missing MTLFunction key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)\n",
                (unsigned)renderProgramKey, (unsigned)vertexProgramName,
                (unsigned)fragmentProgramName, vertexFunction,
                fragmentFunction);
        return 0;
    }

    memset(desc, 0, sizeof(*desc));
    desc->vertex_program_instance = mglProgramBackendView(vertexProgram)->pipeline_cache_instance_id;
    desc->vertex_program_generation = mglProgramBackendView(vertexProgram)->pipeline_cache_generation;
    desc->fragment_program_instance =
        fragmentProgram ? mglProgramBackendView(fragmentProgram)->pipeline_cache_instance_id : 0u;
    desc->fragment_program_generation =
        fragmentProgram ? mglProgramBackendView(fragmentProgram)->pipeline_cache_generation : 0u;
    desc->color_count = MAX_COLOR_ATTACHMENTS;
    desc->rasterization_enabled = 1;

    desc->max_tessellation_factor = mglRenderMaxTessellationFactor();

    {
        /* Metal requires the pipeline's primitive topology class to match
         * the drawn primitive type: an unspecified-class pipeline silently
         * drops point draws.  A compute-routed geometry expansion always
         * gets its output class explicitly.  For ordinary draws only the
         * point case is forced: leaving triangles on the historical
         * unspecified value keeps programs that write gl_PointSize while
         * drawing triangles linkable (Metal rejects a triangle-class
         * pipeline whose vertex function writes point size).
         *
         * Exception: VS writing [[render_target_array_index]] (gl_Layer)
         * requires an explicit topology.  Real AGX often tolerates
         * Unspecified; Apple Paravirtual rejects with CompilerError. */
        const int needsExplicitTopology = mglRenderNeedsExplicitTopology(
            geometryExpansion ? 1 : 0, areas.core->lastDrawPrimitiveMode,
            vertexProgram ? mglRenderVSWritesLayer(vertexProgram) : 0);
        if (needsExplicitTopology) {
            /* A geometry expansion emits the geometry shader's output
             * primitive type, not the GL draw mode: a layout(points) geometry
             * shader drawing GL_PATCHES still rasterizes points.  Classifying
             * by the draw mode gave it MTLPrimitiveTopologyClassTriangle, and
             * Metal refuses to build a pipeline whose vertex function writes
             * [[point_size]] against a triangle class ("Vertex shader writes
             * point size but inputPrimitiveTopology is
             * MTLPrimitiveTopologyClassTriangle").  The geometry pass then
             * produced nothing and the draw rendered all zeroes
             * (tessellation_shader_point_mode.point_rendering), while the
             * sibling points_verification case only passed because its
             * geometry shader happens to emit triangle strips. */
            GLenum topologyMode = (GLenum)areas.core->lastDrawPrimitiveMode;
            if (geometryExpansion && geom->program) {
                switch (geom->program->geometry_output_type) {
                case GL_POINTS:
                    topologyMode = GL_POINTS;
                    break;
                case GL_LINE_STRIP:
                    topologyMode = GL_LINES;
                    break;
                default:
                    topologyMode = GL_TRIANGLES;
                    break;
                }
            }
            desc->input_primitive_topology =
                mglRenderPrimitiveTopologyClass((uint32_t)topologyMode);
        }
        /* isolines / point_mode rasterize the expanded point / line stream,
         * either with the TES acting as its own vertex function (TES-vertex)
         * or through the generated record-passthrough vertex function (TES
         * compute).  Both write gl_PointSize under point_mode, and Metal
         * refuses to link such a vertex function against a triangle topology
         * class.  Force the class from the tessellation raster mode whenever
         * a TES compute draw is being submitted. */
        if (tessCompute && tess->tessComputeProgram) {
            desc->input_primitive_topology =
                mglRenderPrimitiveTopologyClass(
                    (uint32_t)mglTessRasterGLMode(tess->tessComputeProgram));
        }
    }

    if (nativeTES) {
        desc->tessellation_partition_mode =
            mglRenderTessPartitionMode(vertexProgram->tess_gen_spacing);
        desc->max_tessellation_factor = mglRenderMaxTessellationFactor();
        desc->tessellation_factor_scale_enabled = 0;
        desc->tessellation_factor_format =
            (uint32_t)MGLTessellationFactorFormatHalf;

        desc->tessellation_control_point_index_type =
            mglRenderTessControlPointIndexType(tess->tessIndexedDraw ? 1 : 0);
        desc->tessellation_factor_step_function =
            (uint32_t)MGLTessellationFactorStepFunctionPerPatch;
        desc->tessellation_output_winding_order =
            mglRenderTessOutputWinding(vertexProgram->tess_gen_vertex_order);
    }

    /* AGX drops vertex texture/SSBO stores when Metal rasterization is
     * disabled - including tess/cull VS capture draws. Keep rasterization
     * on (real FS or discard stub above); color write masks cleared below. */
    desc->rasterization_enabled = mglRenderRasterizationEnabled(
        rasterizerDiscard, fragmentFunction ? 1 : 0);

    /* Attachment formats: FBO attachment -> pass/drawable/context fallback. */
    if (glState->framebuffer) {
        Framebuffer *fbo = glState->framebuffer;

        for (int i = 0; i < glState->max_color_attachments; i++) {
            if (fbo->color_attachments[i].texture) {
                Texture *tex = mglRendererAttachmentTextureFor(
                    ctx, &fbo->color_attachments[i]);
                if (tex && !mglRendererBindMTLTexture(renderer, tex)) {
                    fprintf(stderr,
                            "MGL PIPELINE DESC fail: bindMTLTexture failed for color attachment %d tex=%u\n",
                            i, tex->name);
                    return 0;
                }
                if (tex && mglTextureMtl(tex)) {
                    desc->color_format[i] =
                        (uint32_t)mtlPixelFormatForGLTex(tex);
                } else {
                    desc->color_format[i] = mglRenderInvalidPixelFormat();
                }
            }

            if (mglRenderColorAttachmentBitfieldDone(
                    (uint32_t)fbo->color_attachment_bitfield, i)) {
                break;
            }
        }

        if (fbo->depth.texture) {
            Texture *tex = mglRendererAttachmentTextureFor(ctx, &fbo->depth);
            if (tex && !mglRendererBindMTLTexture(renderer, tex)) {
                fprintf(stderr,
                        "MGL PIPELINE DESC fail: bindMTLTexture failed for depth tex=%u\n",
                        tex->name);
                return 0;
            }
            if (tex && mglTextureMtl(tex)) {
                const uint32_t rawDepth =
                    (uint32_t)mtlPixelFormatForGLTex(tex);
                const uint32_t depthFormat =
                    mglRenderDepthFormatOrFallback(rawDepth);
                if (mglRenderPixelFormatIsInvalid(rawDepth)) {
                    fprintf(stderr,
                            "MGL ERROR: Invalid depth texture format, falling back to Depth32Float\n");
                }
                desc->depth_format = depthFormat;
            } else {
                desc->depth_format = mglRenderInvalidPixelFormat();
            }
        }

        if (fbo->stencil.texture) {
            Texture *tex = mglRendererAttachmentTextureFor(ctx, &fbo->stencil);
            if (tex && !mglRendererBindMTLTexture(renderer, tex)) {
                fprintf(stderr,
                        "MGL PIPELINE DESC fail: bindMTLTexture failed for stencil tex=%u\n",
                        tex->name);
                return 0;
            }
            if (tex && mglTextureMtl(tex)) {
                const uint32_t rawStencil =
                    (uint32_t)mtlPixelFormatForGLTex(tex);
                const uint32_t stencilFormat =
                    mglRenderStencilFormatOrFallback(rawStencil);
                if (mglRenderPixelFormatIsInvalid(rawStencil)) {
                    fprintf(stderr,
                            "MGL ERROR: Invalid stencil texture format, falling back to Stencil8\n");
                }
                desc->stencil_format = stencilFormat;
            } else {
                desc->stencil_format = mglRenderInvalidPixelFormat();
            }
        }
    } else {
        uint32_t preferredColor0 = mglRenderInvalidPixelFormat();
        if (commandState && mglPdColorTextureFor(commandState, 0)) {
            preferredColor0 =
                mglPdTextureInfo(mglPdColorTextureFor(commandState, 0))
                    .pixel_format;
        } else if (mglRendererDrawableTexture(renderer)) {
            preferredColor0 = mglPdTextureInfo(
                mglRendererDrawableTexture(renderer)).pixel_format;
        } else {
            preferredColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        desc->color_format[0] = preferredColor0;

        if (ctx->depth_format.format) {
            desc->depth_format = mglRenderDepthFormatOrFallback(
                ctx->depth_format.mtl_pixel_format);
        }

        if (ctx->stencil_format.format) {
            desc->stencil_format = mglRenderDefaultFBOStencilFormat(
                ctx->stencil_format.mtl_pixel_format);
        }
    }

    /* Derive pipeline attachment formats from the configured C++ pass. */
    const int hasConfiguredRenderPass =
        commandState->renderPassStateOwner != NULL;
    if (hasConfiguredRenderPass) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            void *rpColor = mglPdColorTextureFor(commandState, (size_t)i);
            if (rpColor) {
                desc->color_format[i] =
                    mglPdTextureInfo(rpColor).pixel_format;
            }
        }

        void *rpDepth = mglPdDepthTextureFor(commandState);
        void *rpStencil = mglPdStencilTextureFor(commandState);
        if (rpDepth) {
            desc->depth_format = mglRenderDepthFormatOrFallback(
                (uint32_t)mglPdTextureInfo(rpDepth).pixel_format);
        } else {
            desc->depth_format = mglRenderInvalidPixelFormat();
        }
        if (rpStencil) {
            desc->stencil_format = mglRenderStencilFormatOrFallback(
                (uint32_t)mglPdTextureInfo(rpStencil).pixel_format);
        } else {
            desc->stencil_format = mglRenderInvalidPixelFormat();
        }
    }

    const int color0IsIntentionallyDisabled =
        mglRenderColor0IntentionallyDisabled(
            glState->framebuffer ? 1 : 0,
            (uint32_t)mglMetalDrawBufferAt(ctx, 0u)) != 0;

    if (!color0IsIntentionallyDisabled &&
        mglRenderColorFormatNeedsFallback(desc->color_format[0])) {
        uint32_t fallbackColor0 = mglRenderInvalidPixelFormat();
        if (commandState && mglPdColorTextureFor(commandState, 0)) {
            fallbackColor0 =
                mglPdTextureInfo(mglPdColorTextureFor(commandState, 0))
                    .pixel_format;
        } else if (mglRendererDrawableTexture(renderer)) {
            fallbackColor0 = mglPdTextureInfo(
                mglRendererDrawableTexture(renderer)).pixel_format;
        } else {
            fallbackColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        fallbackColor0 = mglRenderColorFormatOrBGRA(fallbackColor0);
        if (kMglPdVerbosePipelineLogs) {
            fprintf(stderr,
                    "MGL PIPELINE DESC missing color pixel format, fallback pixelFormat=%lu\n",
                    (unsigned long)fallbackColor0);
        }
        desc->color_format[0] = fallbackColor0;
    }

    /* Resolve the pipeline sample count from the C++ render-pass state. */
    uint64_t resolvedSampleCount = 1;
    void *rpColor0 = mglPdColorTextureFor(commandState, 0);
    void *rpDepth = mglPdDepthTextureFor(commandState);
    void *rpStencil = mglPdStencilTextureFor(commandState);
    if (rpColor0 && mglPdTextureInfo(rpColor0).sample_count > 0) {
        resolvedSampleCount = mglPdTextureInfo(rpColor0).sample_count;
    } else if (rpDepth && mglPdTextureInfo(rpDepth).sample_count > 0) {
        resolvedSampleCount = mglPdTextureInfo(rpDepth).sample_count;
    } else if (rpStencil && mglPdTextureInfo(rpStencil).sample_count > 0) {
        resolvedSampleCount = mglPdTextureInfo(rpStencil).sample_count;
    }
    if (resolvedSampleCount == 0) {
        resolvedSampleCount = 1;
    }
    desc->raster_sample_count = (uint32_t)resolvedSampleCount;

    {
        uint32_t packedFormat = 0u;
        if (mglRenderPassUnifyPackedDS(desc->depth_format,
                                       desc->stencil_format, &packedFormat)) {
            desc->depth_format = packedFormat;
            desc->stencil_format = packedFormat;
        }
    }

    desc->alpha_to_coverage_enabled =
        glState->caps.sample_alpha_to_coverage ? 1 : 0;
    desc->alpha_to_one_enabled =
        glState->caps.sample_alpha_to_one ? 1 : 0;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRenderSkipInvalidColorAttachment(desc->color_format[i])) {
            continue;
        }
        if (mglRenderDrawBufferIsNone(
                (uint32_t)mglMetalDrawBufferAt(ctx, (GLuint)i))) {
            desc->color_write_mask[i] = 0u;
            continue;
        }
        MGLRenderPipelineBlendState blend = {0};
        if (!cache.blend_state || !cache.blend_state(cache.object, (uint32_t)i,
                                                     &blend)) {
            fprintf(stderr,
                    "MGL PIPELINE DESC fail: blend state unavailable for attachment %d\n",
                    i);
            return 0;
        }
        desc->color_write_mask[i] = blend.color_write_mask;
        desc->blending_enabled_mask |= mglRenderBlendingEnabledMaskBit(
            glState->caps.blendi[i] ? 1 : 0, i);
        desc->source_rgb_blend_factor[i] = blend.source_rgb_factor;
        desc->destination_rgb_blend_factor[i] = blend.destination_rgb_factor;
        desc->source_alpha_blend_factor[i] = blend.source_alpha_factor;
        desc->destination_alpha_blend_factor[i] = blend.destination_alpha_factor;
        desc->rgb_blend_operation[i] = blend.rgb_operation;
        desc->alpha_blend_operation[i] = blend.alpha_operation;
    }

    if (mglRenderClearColorWriteMasks(
            (glState->caps.rasterizer_discard || used_fs_stub) ? 1 : 0,
            tessVertexCapture ? 1 : 0, cullDistanceCapture ? 1 : 0)) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            desc->color_write_mask[i] = 0u;
        }
    }

    if (mglRenderNeedsVertexDescriptor(geometryExpansion ? 1 : 0,
                                       tessCompute ? 1 : 0)) {
        if (!mglRendererGenerateVertexDescriptorState(renderer, desc)) {
            return 0;
        }
    }

    if (kMglPdVerbosePipelineLogs) {
        uint32_t activeColorAttachmentCount = 0;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (desc->color_format[i] != mglRenderInvalidPixelFormat() &&
                desc->color_format[i] != 0u) {
                activeColorAttachmentCount++;
            }
        }
        fprintf(stderr,
                "MGL PIPELINE DESC colorAttachmentCount=%u depthFormat=%u stencilFormat=%u sampleCount=%u\n",
                (unsigned)activeColorAttachmentCount,
                (unsigned)desc->depth_format, (unsigned)desc->stencil_format,
                (unsigned)desc->raster_sample_count);
        fprintf(stderr, "MGL PIPELINE DESC renderTarget[0]=%u\n",
                (unsigned)desc->color_format[0]);
    }

    functions_out->vertex_function = vertexFunction;
    functions_out->fragment_function = fragmentFunction;
    return 1;
}
