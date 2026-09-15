/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_sampled_copy.c — the sampled-copy cluster moved out of
 * MGLRenderer+Blit.m: the eligibility predicate and the copy refresh/update
 * itself.  Only the writable-command-buffer step is still a method (it rotates
 * through -newCommandBufferLocked), reached through one port.
 */

#include "mgl_render_pass_manager_ops.h"
#include "mgl_blit_sampled_copy.h"
#include "mgl_render.h"
#include "mgl_renderer_ports.h"  /* state areas, ensure-writable command buffer */
#include "mgl_blit_pipelines.h"  /* scaled copy pipeline / compute pipeline / sampler */
#include "mgl_texture_compat.h"  /* release sampled copy, data kind name, trace label */
#include "mgl_rt_sync.h"
#include "mgl_coordinate.h"      /* mglRTWriteAuthorityIsCurrentAndUsesOriginal */        /* mglTextureCanUseGLSampledRenderTargetCopy */
#include "mgl_region_value.h"  /* MGLSizeValue / mglBlitSize */
#include "mgl_trace_log.h"      /* mglTraceLog / mglTraceLogIsEnabled */
#include "mgl_thread_affinity.h" /* MGL_ASSERT_GL_THREAD */

#include <simd/simd.h>          /* vector_float4 / vector_uint2 (shader layouts) */
#include <stdio.h>              /* fprintf for the RT-SAMPLE-COPY diagnostics */

/* Local twin of the file-static mglBlitTextureInfo in MGLRenderer+Blit.m
 * (same three lines: fetch the Metal texture info, zero when absent). */
static MGLRenderTextureInfo mglBlitSampledCopyTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

int mglBlitTextureCanUseGLSampledRenderTargetCopy(Texture *tex, void *source)
{
    if (!tex || !source || !tex->is_render_target) {
        return 0;
    }

    if (!mglRenderTextureTargetIs2D((uint32_t)tex->target) ||
        tex->width == 0u ||
        tex->height == 0u ||
        mglBlitSampledCopyTextureInfo(source).texture_type != MGLTextureType2D ||
        mglBlitSampledCopyTextureInfo(source).mipmap_level_count == 0u ||
        mglBlitSampledCopyTextureInfo(source).width == 0u ||
        mglBlitSampledCopyTextureInfo(source).height == 0u ||
        mglMetalPixelFormatIsDepthOrStencil(mglBlitSampledCopyTextureInfo(source).pixel_format)) {
        return 0;
    }

    /* Float + integer color RTs need a GL-sampled copy for FBO feedback
     * (same texture as attachment and sampler).  Depth/stencil stay out. */
    MGLTextureDataKind kind =
        mglTextureDataKindForPixelFormat(mglBlitSampledCopyTextureInfo(source).pixel_format);
    if (kind != MGLTextureDataKindFloat &&
        kind != MGLTextureDataKindUint &&
        kind != MGLTextureDataKindSint) {
        return 0;
    }

    /* Apply sampled-copy protection to all 2D float render targets
     * regardless of size.  The previous size-based gating was a
     * Minecraft-specific heuristic that broke on larger render targets. */
    if (!mglTextureCanUseGLSampledRenderTargetCopy(tex)) {
        return 0;
    }

    return 1;
}

/* === the GL-sampled render-target copy update =============================
 *
 * Body of the former -[MGLRenderer updateGLSampledRenderTargetCopyForTexture:
 * source:reason:].  Everything it needed was already C: the eligibility
 * predicate above, the scaled-copy pipelines/sampler (mgl_blit_pipelines.c),
 * the compute/render encoders and their setters (mgl_render.h), the texture
 * create/view/release helpers and the command state through the state areas.
 * The only ObjC steps left are the two that had to stay methods: the writable
 * command buffer (it rotates through -newCommandBufferLocked) and nothing else.
 */

/* Local mirror of MGLScaledBlitParams (MGLRenderer+Blit_Private.h); the Metal
 * shader string mirrors the same layout, so the field order must not change. */
typedef struct {
    vector_float4 uvRect;
    float forceOpaqueAlpha;
    vector_float3 _padding;
} MGLBlitSampledCopyScaledParams;

typedef struct {
    vector_uint2 dstSize;
    uint32_t srcLevel;
    uint32_t dstLevel;
} MGLBlitSampledCopyComputeParams;

static void mglBlitSampledCopyReleaseViews(void *srcLvl, int srcOwned,
                                           void *dstLvl, int dstOwned)
{
    if (srcOwned && srcLvl) {
        mglRenderReleaseMetalObject(srcLvl);
    }
    if (dstOwned && dstLvl) {
        mglRenderReleaseMetalObject(dstLvl);
    }
}

int mglBlitUpdateGLSampledRenderTargetCopy(void *renderer, Texture *tex,
                                           void *source, const char *reason)
{
    MGL_ASSERT_GL_THREAD();
    if (!mglBlitTextureCanUseGLSampledRenderTargetCopy(tex, source)) {
        return 0;
    }
    if (tex->mtl_render_target_write_version == 0u) {
        return 0;
    }

    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    MGLCommandState *cs = areas.command;

    /* Copy the full GL mip chain of the RT (capped to the source's
     * mipmap_level_count), not the transient GpuTextureView BASE/MAX_LEVEL
     * window: shrinking the Y-flip copy to that window left higher mips stale
     * and re-broke the ced1a99 stripe fix.  Sampling windows are applied later
     * via mglSampledTextureViewForBaseLevel. */
    uint64_t copyLevelCount = 1u;
    if (mglBlitSampledCopyTextureInfo(source).mipmap_level_count > 1u) {
        GLuint highestGLLevel = tex->num_levels > 0u ? tex->num_levels - 1u : 0u;
        if (tex->mipmap_levels > 0u && highestGLLevel >= tex->mipmap_levels) {
            highestGLLevel = tex->mipmap_levels - 1u;
        }
        uint64_t highestSourceLevel =
            (uint64_t)mglBlitSampledCopyTextureInfo(source).mipmap_level_count - 1u;
        if ((uint64_t)highestGLLevel > highestSourceLevel) {
            highestGLLevel = (GLuint)highestSourceLevel;
        }
        copyLevelCount = (uint64_t)highestGLLevel + 1u;
    }

    if (tex->mtl_gl_sampled_data &&
        tex->mtl_gl_sampled_width == (GLuint)mglBlitSampledCopyTextureInfo(source).width &&
        tex->mtl_gl_sampled_height == (GLuint)mglBlitSampledCopyTextureInfo(source).height &&
        tex->mtl_gl_sampled_format == (GLuint)mglBlitSampledCopyTextureInfo(source).pixel_format &&
        tex->mtl_gl_sampled_levels == (GLuint)copyLevelCount &&
        tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version &&
        tex->mtl_gl_sampled_dirty_mip_mask == 0u) {
        return 1;
    }

    int needsNewCopy =
        tex->mtl_gl_sampled_data == NULL ||
        tex->mtl_gl_sampled_width != (GLuint)mglBlitSampledCopyTextureInfo(source).width ||
        tex->mtl_gl_sampled_height != (GLuint)mglBlitSampledCopyTextureInfo(source).height ||
        tex->mtl_gl_sampled_format != (GLuint)mglBlitSampledCopyTextureInfo(source).pixel_format ||
        tex->mtl_gl_sampled_levels != (GLuint)copyLevelCount;
    if (needsNewCopy) {
        mglTextureReleaseGLSampledCopy(tex);

        MGLRenderTextureDescriptorState desc = {0};
        desc.texture_type = MGLTextureType2D;
        desc.pixel_format = mglBlitSampledCopyTextureInfo(source).pixel_format;
        desc.width = mglBlitSampledCopyTextureInfo(source).width;
        desc.height = mglBlitSampledCopyTextureInfo(source).height;
        desc.depth = 1;
        desc.mipmap_level_count = copyLevelCount;
        desc.sample_count = 1;
        desc.array_length = 1;
        desc.usage = MGLTextureUsageShaderRead | MGLTextureUsageRenderTarget | MGLTextureUsageShaderWrite;
        desc.storage_mode = MGLStorageModePrivate;

        void *copy = NULL;
        if (mglRenderCreateTextureFromState(&desc, NULL, &copy) != 0 || !copy) {
            static uint64_t s_copyCreateFailCount = 0;
            uint64_t hit = ++s_copyCreateFailCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL RT-SAMPLE-COPY create failed tex=%u size=%lux%lu fmt=%lu reason=%s hit=%llu\n",
                        (unsigned)tex->name,
                        (unsigned long)mglBlitSampledCopyTextureInfo(source).width,
                        (unsigned long)mglBlitSampledCopyTextureInfo(source).height,
                        (unsigned long)mglBlitSampledCopyTextureInfo(source).pixel_format,
                        reason ? reason : "(null)",
                        (unsigned long long)hit);
            }
            return 0;
        }

        /* The C create helper returns +1; the copy keeps that reference. */
        tex->mtl_gl_sampled_data = copy;
        tex->mtl_gl_sampled_width = (GLuint)mglBlitSampledCopyTextureInfo(source).width;
        tex->mtl_gl_sampled_height = (GLuint)mglBlitSampledCopyTextureInfo(source).height;
        tex->mtl_gl_sampled_format = (GLuint)mglBlitSampledCopyTextureInfo(source).pixel_format;
        tex->mtl_gl_sampled_levels = (GLuint)copyLevelCount;
    }

    void *destination = tex->mtl_gl_sampled_data;
    void *sampler = mglBlitScaledSamplerForFilter(renderer, (uint32_t)mglRenderNearestFilter());
    if (!destination || !sampler) {
        static uint64_t s_copySetupFailCount = 0;
        uint64_t hit = ++s_copySetupFailCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            fprintf(stderr,
                    "MGL RT-SAMPLE-COPY setup failed tex=%u dst=%p sampler=%p reason=%s hit=%llu\n",
                    (unsigned)tex->name,
                    destination,
                    sampler,
                    reason ? reason : "(null)",
                    (unsigned long long)hit);
        }
        return 0;
    }

    uint64_t mipLevels = copyLevelCount > 1u ? copyLevelCount : 1u;
    if (mipLevels > mglBlitSampledCopyTextureInfo(destination).mipmap_level_count) {
        mipLevels = mglBlitSampledCopyTextureInfo(destination).mipmap_level_count;
    }
    if (mipLevels > (uint64_t)mglBlitSampledCopyTextureInfo(source).mipmap_level_count) {
        mipLevels = (uint64_t)mglBlitSampledCopyTextureInfo(source).mipmap_level_count;
    }
    uint32_t mipMask = mipLevels >= 32u
        ? UINT32_MAX
        : (((uint32_t)1u << mipLevels) - 1u);
    uint32_t copyMask = needsNewCopy
        ? mipMask
        : (tex->mtl_gl_sampled_dirty_mip_mask & mipMask);
    if (copyMask == 0u &&
        tex->mtl_gl_sampled_write_version != tex->mtl_render_target_write_version) {
        copyMask = mipMask;
    }

    if (!mglRenderPassEnsureWritableCommandBufferLocked(renderer,
                                                    reason ? reason : "rt_sample_copy")) {
        return 0;
    }

    /* Sampled render-target copy: flip rows once so that Metal row 0 (top, which
     * is what Metal's texture::sample sees at v=0) holds GL row 0 (bottom).
     * See the longer comment block in the fallback render path below for the
     * full Metal-vs-GL Y-origin rationale. */
    int yFlipCopy = 1;

    uint32_t dirtyBefore = tex->mtl_gl_sampled_dirty_mip_mask;
    uint32_t copiedMask = 0u;

    /* Prefer compute path: single MTLComputeCommandEncoder dispatches all dirty
     * mip levels, avoiding per-mip render-encoder creation overhead. */
    int useComputePath =
        (mglBlitSampledCopyTextureInfo(destination).usage & MGLTextureUsageShaderWrite) != 0;
    void *computePipeline = NULL;
    if (useComputePath) {
        computePipeline = mglBlitScaledComputePipelineForPixelFormat(
            renderer, mglBlitSampledCopyTextureInfo(destination).pixel_format);
        if (!computePipeline) {
            useComputePath = 0;
        }
    }

    if (useComputePath) {
        void *computeEncoder =
            mglRenderCreateComputeEncoderBorrowed(cs ? cs->currentCommandBufferOwner : NULL);
        if (!computeEncoder) {
            static uint64_t s_computeEncoderFailCount = 0;
            uint64_t hit = ++s_computeEncoderFailCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL RT-SAMPLE-COPY compute encoder failed tex=%u reason=%s hit=%llu\n",
                        (unsigned)tex->name,
                        reason ? reason : "(null)",
                        (unsigned long long)hit);
            }
            useComputePath = 0;
        } else {
            MGLBlitSampledCopyComputeParams params;
            mglRenderSetComputePipelineState(computeEncoder, computePipeline);
            mglRenderSetComputeTexture(computeEncoder, source, 0);
            mglRenderSetComputeTexture(computeEncoder, destination, 1);

            uint64_t tgW = 16u;
            uint32_t totalThreads =
                mglRenderComputePipelineMaxTotalThreads(computePipeline);
            if ((uint64_t)totalThreads < tgW) {
                tgW = (uint64_t)totalThreads;
            }
            uint64_t tgH = 16u;
            if ((uint64_t)totalThreads / (tgW ? tgW : 1u) < tgH) {
                tgH = (uint64_t)totalThreads / (tgW ? tgW : 1u);
            }
            if (tgH == 0u) {
                tgH = 1u;
            }
            MGLSizeValue threadgroup = mglBlitSize(tgW, tgH, 1u);

            for (uint64_t lvl = 0u; lvl < mipLevels; lvl++) {
                if ((copyMask & ((uint32_t)1u << lvl)) == 0u) {
                    continue;
                }

                uint64_t mipW = mglBlitSampledCopyTextureInfo(source).width >> lvl;
                uint64_t mipH = mglBlitSampledCopyTextureInfo(source).height >> lvl;
                if (mipW == 0u) mipW = 1u;
                if (mipH == 0u) mipH = 1u;

                params.dstSize = (vector_uint2){(uint32_t)mipW, (uint32_t)mipH};
                params.srcLevel = (uint32_t)lvl;
                params.dstLevel = (uint32_t)lvl;

                mglRenderSetComputeBytes(computeEncoder, &params, sizeof(params), 0);

                MGLSizeValue threads = mglBlitSize(mipW, mipH, 1u);
                mglRenderDispatchComputeThreads(computeEncoder,
                                                (uint32_t)threads.width,
                                                (uint32_t)threads.height,
                                                (uint32_t)threads.depth,
                                                (uint32_t)threadgroup.width,
                                                (uint32_t)threadgroup.height,
                                                (uint32_t)threadgroup.depth);
                copiedMask |= (uint32_t)1u << lvl;
            }
            mglRenderEndComputeEncoder(computeEncoder);
        }
    }

    if (!useComputePath) {
        /* Render-path scaled blit is float-only; integer RTs must use
         * the uint/int compute kernels. */
        MGLTextureDataKind copyKind =
            mglTextureDataKindForPixelFormat(
                mglBlitSampledCopyTextureInfo(destination).pixel_format);
        if (copyKind == MGLTextureDataKindUint ||
            copyKind == MGLTextureDataKindSint) {
            static uint64_t s_intCopyFailCount = 0;
            uint64_t hit = ++s_intCopyFailCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL RT-SAMPLE-COPY integer compute path unavailable tex=%u kind=%s reason=%s hit=%llu\n",
                        (unsigned)tex->name,
                        mglTextureDataKindName(copyKind),
                        reason ? reason : "(null)",
                        (unsigned long long)hit);
            }
            return 0;
        }

        void *pipeline = mglBlitScaledPipelineForPixelFormat(
            renderer, mglBlitSampledCopyTextureInfo(destination).pixel_format);
        if (!pipeline) {
            static uint64_t s_copySetupFailCount = 0;
            uint64_t hit = ++s_copySetupFailCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL RT-SAMPLE-COPY render pipeline setup failed tex=%u reason=%s hit=%llu\n",
                        (unsigned)tex->name,
                        reason ? reason : "(null)",
                        (unsigned long long)hit);
            }
            return 0;
        }

        MGLBlitSampledCopyScaledParams params;
        params.uvRect = yFlipCopy
            ? (vector_float4){0.0f, 1.0f, 1.0f, 0.0f}
            : (vector_float4){0.0f, 0.0f, 1.0f, 1.0f};
        params.forceOpaqueAlpha = 0.0f;
        params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

        for (uint64_t lvl = 0u; lvl < mipLevels; lvl++) {
            if ((copyMask & ((uint32_t)1u << lvl)) == 0u) {
                continue;
            }
            /* No @autoreleasepool here: the two level views are the only
             * temporaries and they are released explicitly. */
            void *srcLvl = source;
            void *dstLvl = destination;
            int srcOwned = 0;
            int dstOwned = 0;
            if (mipLevels > 1u) {
                if (mglRenderCreateTextureViewRange(
                        source, mglBlitSampledCopyTextureInfo(source).pixel_format,
                        MGLTextureType2D, lvl, 1u, 0u, 1u, 0, 0, 0, 0, 0,
                        &srcLvl) != 0 || !srcLvl) {
                    srcLvl = NULL;
                } else {
                    srcOwned = 1;
                }
                if (mglRenderCreateTextureViewRange(
                        destination,
                        mglBlitSampledCopyTextureInfo(destination).pixel_format,
                        MGLTextureType2D, lvl, 1u, 0u, 1u, 0, 0, 0, 0, 0,
                        &dstLvl) != 0 || !dstLvl) {
                    dstLvl = NULL;
                } else {
                    dstOwned = 1;
                }
                if (!srcLvl || !dstLvl) {
                    static uint64_t s_levelViewFailCount = 0;
                    uint64_t hit = ++s_levelViewFailCount;
                    if (hit <= 32ull || (hit % 512ull) == 0ull) {
                        fprintf(stderr,
                                "MGL RT-SAMPLE-COPY level view failed tex=%u lvl=%lu hit=%llu\n",
                                (unsigned)tex->name,
                                (unsigned long)lvl,
                                (unsigned long long)hit);
                    }
                    mglBlitSampledCopyReleaseViews(srcLvl, srcOwned, dstLvl, dstOwned);
                    continue;
                }
            }

            MGLRenderPassState copyState;
            mglRenderInitDefaultRenderPassState(&copyState);
            copyState.color[0].attachment = (MGLRenderPassAttachmentState){
                .texture = dstLvl,
                .level = 0u,
                .slice = 0u,
                .depth_plane = 0u,
                .load_action = MGLLoadActionDontCare,
                .store_action = MGLStoreActionStore,
            };
            copyState.render_target_width = mglBlitSampledCopyTextureInfo(dstLvl).width;
            copyState.render_target_height = mglBlitSampledCopyTextureInfo(dstLvl).height;

            void *copyEncoder = NULL;
            if (mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
                    cs ? cs->currentCommandBufferOwner : NULL, &copyState,
                    &copyEncoder) != 0 || !copyEncoder) {
                static uint64_t s_copyEncoderFailCount = 0;
                uint64_t hit = ++s_copyEncoderFailCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    fprintf(stderr,
                            "MGL RT-SAMPLE-COPY encoder failed tex=%u lvl=%lu reason=%s hit=%llu\n",
                            (unsigned)tex->name,
                            (unsigned long)lvl,
                            reason ? reason : "(null)",
                            (unsigned long long)hit);
                }
                mglBlitSampledCopyReleaseViews(srcLvl, srcOwned, dstLvl, dstOwned);
                continue;
            }

            mglRenderSetRenderPipelineState(copyEncoder, pipeline);
            mglRenderSetRenderBytes(copyEncoder, &params, sizeof(params),
                                    MGL_RENDER_BINDING_STAGE_VERTEX, 0);
            mglRenderSetRenderBytes(copyEncoder, &params, sizeof(params),
                                    MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
            mglRenderSetRenderTexture(copyEncoder, srcLvl,
                                      MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
            mglRenderSetRenderSampler(copyEncoder, sampler,
                                      MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
            mglRenderSetRenderViewport(copyEncoder,
                                       0.0, 0.0,
                                       (double)mglBlitSampledCopyTextureInfo(dstLvl).width,
                                       (double)mglBlitSampledCopyTextureInfo(dstLvl).height,
                                       0.0, 1.0);
            mglRenderSetRenderScissor(copyEncoder,
                                      0u, 0u,
                                      mglBlitSampledCopyTextureInfo(dstLvl).width,
                                      mglBlitSampledCopyTextureInfo(dstLvl).height);
            (void)mglRenderEncodeDraw(copyEncoder,
                &(MGLRenderDrawPlan){
                    .kind = MGL_RENDER_DRAW_ARRAY,
                    .primitive_type = MGLPrimitiveTypeTriangleStrip,
                    .vertex_start = 0u,
                    .vertex_count = 4u,
                    .instance_count = 1u,
                    .base_instance = 0u,
                }, NULL, 0);
            mglRenderEndRenderEncoder(copyEncoder);
            copiedMask |= (uint32_t)1u << lvl;
            mglBlitSampledCopyReleaseViews(srcLvl, srcOwned, dstLvl, dstOwned);
        }
    }

    tex->mtl_gl_sampled_dirty_mip_mask &= ~copiedMask;
    if ((tex->mtl_gl_sampled_dirty_mip_mask & mipMask) == 0u) {
        tex->mtl_gl_sampled_write_version = tex->mtl_render_target_write_version;
    }

    if (mglTraceLogIsEnabled()) {
        char levelSizes[160];
        size_t levelSizesLen = 0;
        levelSizes[0] = '\0';
        for (uint64_t lvl = 0u; lvl < mipLevels && lvl < 8u; lvl++) {
            uint64_t mipW = mglBlitSampledCopyTextureInfo(source).width >> lvl;
            uint64_t mipH = mglBlitSampledCopyTextureInfo(source).height >> lvl;
            if (mipW == 0u) mipW = 1u;
            if (mipH == 0u) mipH = 1u;
            int n = snprintf(levelSizes + levelSizesLen,
                             sizeof(levelSizes) - levelSizesLen,
                             "%s%lux%lu",
                             levelSizesLen ? "," : "",
                             (unsigned long)mipW,
                             (unsigned long)mipH);
            if (n < 0 || (size_t)n >= sizeof(levelSizes) - levelSizesLen) {
                break;
            }
            levelSizesLen += (size_t)n;
        }
        mglTraceLog("RT_SAMPLE_COPY_UPDATED tex=%u label=\"%s\" lightmap=%d yFlip=%d src=%p dst=%p size=%lux%lu fmt=%lu srcLevels=%lu dstLevels=%lu glLevels=%u mips=%u base=%u max=%u writeVersion=%u dirtyBefore=0x%x copyMask=0x%x copiedMask=0x%x dirtyAfter=0x%x levelSizes=%s reason=%s compute=%d",
                    (unsigned)tex->name,
                    mglTraceTextureLabel(tex),
                    0,
                    yFlipCopy ? 1 : 0,
                    source,
                    destination,
                    (unsigned long)mglBlitSampledCopyTextureInfo(destination).width,
                    (unsigned long)mglBlitSampledCopyTextureInfo(destination).height,
                    (unsigned long)mglBlitSampledCopyTextureInfo(destination).pixel_format,
                    (unsigned long)mglBlitSampledCopyTextureInfo(source).mipmap_level_count,
                    (unsigned long)mglBlitSampledCopyTextureInfo(destination).mipmap_level_count,
                    (unsigned)tex->num_levels,
                    (unsigned)tex->mipmap_levels,
                    (unsigned)tex->params.base_level,
                    (unsigned)tex->params.max_level,
                    (unsigned)tex->mtl_gl_sampled_write_version,
                    (unsigned)dirtyBefore,
                    (unsigned)copyMask,
                    (unsigned)copiedMask,
                    (unsigned)tex->mtl_gl_sampled_dirty_mip_mask,
                    levelSizes,
                    reason ? reason : "(null)",
                    useComputePath ? 1 : 0);
    }

    return 1;
}


/* Body of the former -[MGLRenderer updateGLSampledCopiesForEndedRenderPassFramebuffer:
 * drawCount:drawBuffers:reason:] (P0-1): the context comes from the state areas and
 * the attachment texture from the C port, so nothing here needs Objective-C. */
void mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer(
    void *renderer, Framebuffer *fbo, const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);

    if (!areas.ctx || !fbo) {
        return;
    }


    bool anySampledRT = false;
    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (!mglRenderColorAttachmentBitSet(
                (uint32_t)fbo->color_attachment_bitfield, attachmentIndex)) {
            continue;
        }
        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *tex = mglRendererAttachmentTextureFor(areas.ctx, attachment);
        if (tex && tex->mtl_data &&
            mglRenderSampledRTNeedsCopy(tex->is_render_target ? 1 : 0,
                                        tex->mtl_render_target_write_version)) {
            anySampledRT = true;
            break;
        }
    }
    if (!anySampledRT) {
        return;
    }

    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (!mglRenderColorAttachmentBitSet(
                (uint32_t)fbo->color_attachment_bitfield, attachmentIndex)) {
            continue;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];

        Texture *tex = mglRendererAttachmentTextureFor(areas.ctx, attachment);
        if (!tex || !tex->mtl_data) {
            continue;
        }

        void *source = tex->mtl_data;
        if (!mglBlitTextureCanUseGLSampledRenderTargetCopy(tex, source)) {
            continue;
        }


        if (mglRTWriteAuthorityIsCurrentAndUsesOriginal(tex)) {
            if (tex->mtl_gl_sampled_data &&
                mglRenderSampledRTCopyStale(tex->mtl_gl_sampled_write_version,
                                            tex->mtl_render_target_write_version)) {
                mglTextureReleaseGLSampledCopy(tex);
                if (mglTraceLogIsEnabled()) {
                    mglTraceLog("RT_SAMPLE_COPY_SKIP_INJECTED_RENDER tex=%u label=\"%s\" reason=render_yflip_injected_stale_released",
                                (unsigned)tex->name,
                                mglTraceTextureLabel(tex));
                }
            }
            continue;
        }

        (void)mglBlitUpdateGLSampledRenderTargetCopy(renderer, tex, source, reason ? reason : "end_render_pass");
    }
}
