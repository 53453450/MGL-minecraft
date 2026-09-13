/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_swap_diagnostics.c — MGLRenderer+SwapDiagnostics.m moved here (P0-1,
 * log 101).  The file was already written against the mglRender* C facade, so
 * the translation is `id`/`NSUInteger` -> `void *`/`size_t`, `NSLog` ->
 * fprintf on the same sink, and the two blocks replaced by a static function
 * plus a completion context:
 *
 *   - `scheduleTextureSample` (a local block invoked ten times) is
 *     mglSwapScheduleTextureSample();
 *   - the completion block that reads back the sample buffer is
 *     mglSwapSampleCompletion(), registered through
 *     mglRenderAddCommandBufferOwnerCompletion() with a heap context.  The
 *     block's captured NSString tag becomes a `const char *` (every call site
 *     passes a literal, so the pointer outlives the completion); its captured
 *     sample buffer moves its +1 into the context and is released by
 *     mglSwapSampleCompletionDestroy().
 *
 * The five ivar reads all have C homes now: `ctx` -> areas.ctx, `_activeState`
 * -> areas.core->activeState (the MGL_STATE() dual-proxy rule), and
 * `_defaultDrawableWrittenSinceLastSwap` + `_renderPassManager.state` ->
 * areas.core->defaultDrawableWrittenSinceLastSwap and areas.command.  No new
 * port is needed.
 */

#include "mgl_swap_diagnostics.h"
#include "glm_context.h"
#include "mgl_renderer_ports.h"  /* state areas */
#include "mgl_blit_pipelines.h"  /* scaled-blit pipeline/sampler + params */
#include "mgl_render.h"          /* the mglRender* facade */
#include "mgl_sync.h"            /* mglCommandBufferStatusName */
#include "mgl_trace_log.h"       /* mglTraceLog */
#include "mgl_metal_ref.h"       /* mglSafeReleaseMetalObj */
#include "utils.h"             /* MIN (the Objective-C file got it from Foundation) */

#include <simd/simd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === helpers ============================================================ */

/* MGL_STATE() from MGLRenderer_Private.h, expressed in C: the core state's
 * active pointer wins, NULL means "the context's own state". */
static GLMState *mglSwapStateForRenderer(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

static void *mglSwapCommandBufferOwner(const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentCommandBufferOwner : NULL;
}

/* +1 sample buffer, or NULL; the caller owns the reference. */
static void *mglSwapDiagnosticsCreateBuffer(size_t length)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((uint64_t)length, 0u,
                              "MGL Swap Diagnostic Sample", &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

/* Borrowed render encoder that renders into color_texture. */
static void *mglSwapDiagnosticsCreateRenderEncoder(void *command_buffer_owner,
                                                   void *color_texture)
{
    if (!command_buffer_owner || !color_texture) {
        return NULL;
    }
    MGLRenderPassState state = {0};
    state.color[0].attachment.texture = color_texture;
    state.color[0].attachment.load_action = 0u;
    state.color[0].attachment.store_action = 1u;
    return mglRenderCreateRenderEncoderBorrowed(command_buffer_owner, &state);
}

/* === sample completion ==================================================
 * The block that used to run when the sample command buffer completed: read the
 * buffer back, summarize it, and trace.  Its captured state lives here. */

typedef struct MGLSwapSampleCompletion_t {
    void *sample_buffer;      /* +1, released in Destroy */
    const char *sample_tag;   /* call-site literal, never freed */
    uint64_t swap_call;
    size_t sample_width;
    size_t sample_height;
    size_t texture_width;
    size_t texture_height;
    size_t origin_x;
    size_t origin_y;
    size_t bytes_per_image;
} MGLSwapSampleCompletion;

static void mglSwapSampleCompletionDestroy(void *context)
{
    MGLSwapSampleCompletion *completion = (MGLSwapSampleCompletion *)context;
    if (!completion) {
        return;
    }
    mglSafeReleaseMetalObj(&completion->sample_buffer);
    free(completion);
}

static void mglSwapSampleCompletion(void *context,
                                    const MGLRenderCommandBufferState *state)
{
    MGLSwapSampleCompletion *completion = (MGLSwapSampleCompletion *)context;
    if (!completion || !state) {
        return;
    }

    /* The Objective-C version built an NSString here and printed it with %@;
     * the trace logger is a C varargs sink, so the text is formatted directly
     * (and a missing error prints as "(null)", which is what %@ did). */
    char error_text[MGL_RENDER_ERROR_DESCRIPTION_CAPACITY +
                    MGL_RENDER_ERROR_DOMAIN_CAPACITY + 64];
    const char *sample_error = NULL;
    if (state->has_error) {
        snprintf(error_text, sizeof(error_text),
                 "%s (domain=%s code=%lld)",
                 state->error_description,
                 state->error_domain,
                 (long long)state->error_code);
        sample_error = error_text;
    }

    void *sample_contents = NULL;
    uint64_t sample_buffer_length = 0;
    (void)mglRenderGetBufferContents(completion->sample_buffer,
                                     &sample_contents, &sample_buffer_length);
    const uint8_t *p = sample_buffer_length >= completion->bytes_per_image
        ? (const uint8_t *)sample_contents : NULL;
    if (!p) {
        mglTraceLog("MGL TRACE swap.sample.%s call=%llu unavailable(contents=nil) status=%s error=%s",
                    completion->sample_tag,
                    (unsigned long long)completion->swap_call,
                    mglCommandBufferStatusName(state->status),
                    sample_error ? sample_error : "(null)");
        return;
    }

    uint64_t sum = 0;
    size_t nonZero = 0;
    for (size_t bi = 0; bi < completion->bytes_per_image; bi++) {
        uint8_t v = p[bi];
        sum += (uint64_t)v;
        if (v != 0) {
            nonZero++;
        }
    }

    uint32_t firstPixel = 0;
    if (completion->bytes_per_image >= sizeof(firstPixel)) {
        memcpy(&firstPixel, p, sizeof(firstPixel));
    }

    uint32_t minPixel = UINT32_MAX;
    uint32_t maxPixel = 0u;
    uint32_t pixelXor = 0u;
    size_t diffFromFirst = 0u;
    size_t pixelCount = completion->bytes_per_image / sizeof(uint32_t);
    for (size_t pi = 0; pi < pixelCount; pi++) {
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
    bool appearsSolid = (pixelCount > 0u && diffFromFirst == 0u);

    mglTraceLog("MGL TRACE swap.sample.%s call=%llu tex=%lux%lu origin=(%lu,%lu) sample=%lux%lu "
                "nonZero=%lu/%lu sum=%llu firstPixel=0x%08x min=0x%08x max=0x%08x xor=0x%08x diff=%lu solid=%d status=%s error=%s",
                completion->sample_tag,
                (unsigned long long)completion->swap_call,
                (unsigned long)completion->texture_width,
                (unsigned long)completion->texture_height,
                (unsigned long)completion->origin_x,
                (unsigned long)completion->origin_y,
                (unsigned long)completion->sample_width,
                (unsigned long)completion->sample_height,
                (unsigned long)nonZero,
                (unsigned long)completion->bytes_per_image,
                (unsigned long long)sum,
                firstPixel,
                minPixel == UINT32_MAX ? 0u : minPixel,
                maxPixel,
                pixelXor,
                (unsigned long)diffFromFirst,
                appearsSolid ? 1 : 0,
                mglCommandBufferStatusName(state->status),
                sample_error ? sample_error : "(null)");

    if (strcmp(completion->sample_tag, "src.center") == 0) {
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
            mglTraceLog("MGL TRACE swap.sample.center_stable firstPixel=0x%08x run=%llu solid=%d diff=%lu",
                        firstPixel,
                        (unsigned long long)s_sameCenterPixelRun,
                        appearsSolid ? 1 : 0,
                        (unsigned long)diffFromFirst);
        }
    }
}

/* === one sample: blit the texture region into a fresh buffer and arm the
 * completion (the former `scheduleTextureSample` block) ==================== */

static void mglSwapScheduleTextureSample(void *renderer, void *sample_texture,
                                         const char *sample_tag,
                                         size_t origin_x, size_t origin_y,
                                         uint64_t swap_call)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!sample_texture) {
        mglTraceLog("MGL TRACE swap.sample.%s call=%llu skipped(texture=nil)",
                    sample_tag, (unsigned long long)swap_call);
        return;
    }

    MGLRenderTextureInfo sample_info = {0};
    if (mglRenderGetTextureInfo(sample_texture, &sample_info) != 0) {
        return;
    }
    if (sample_info.pixel_format != 80u &&
        sample_info.pixel_format != 70u) {
        mglTraceLog("MGL TRACE swap.sample.%s call=%llu skipped(fmt=%lu tex=%lux%lu)",
                    sample_tag,
                    (unsigned long long)swap_call,
                    (unsigned long)sample_info.pixel_format,
                    (unsigned long)sample_info.width,
                    (unsigned long)sample_info.height);
        return;
    }

    size_t sample_width = MIN((size_t)sample_info.width, 8u);
    size_t sample_height = MIN((size_t)sample_info.height, 8u);
    size_t bytes_per_pixel = 4u;
    size_t sample_bytes_per_row = sample_width * bytes_per_pixel;
    size_t sample_bytes_per_image = sample_bytes_per_row * sample_height;
    if (sample_width == 0 || sample_height == 0 || sample_bytes_per_image == 0) {
        mglTraceLog("MGL TRACE swap.sample.%s call=%llu skipped(invalid-size tex=%lux%lu)",
                    sample_tag,
                    (unsigned long long)swap_call,
                    (unsigned long)sample_info.width,
                    (unsigned long)sample_info.height);
        return;
    }

    size_t clamped_origin_x = origin_x;
    size_t clamped_origin_y = origin_y;
    if (clamped_origin_x + sample_width > (size_t)sample_info.width) {
        clamped_origin_x = ((size_t)sample_info.width > sample_width)
            ? ((size_t)sample_info.width - sample_width)
            : 0u;
    }
    if (clamped_origin_y + sample_height > (size_t)sample_info.height) {
        clamped_origin_y = ((size_t)sample_info.height > sample_height)
            ? ((size_t)sample_info.height - sample_height)
            : 0u;
    }

    void *sample_buffer = mglSwapDiagnosticsCreateBuffer(sample_bytes_per_image);
    if (!sample_buffer) {
        fprintf(stderr,
                "MGL WARNING: swap.sample.%s call=%llu failed(alloc size=%lu)\n",
                sample_tag,
                (unsigned long long)swap_call,
                (unsigned long)sample_bytes_per_image);
        return;
    }

    void *sample_encoder =
        mglRenderCreateBlitEncoderBorrowed(mglSwapCommandBufferOwner(&areas));
    if (!sample_encoder) {
        fprintf(stderr,
                "MGL WARNING: swap.sample.%s call=%llu failed(create blit encoder)\n",
                sample_tag,
                (unsigned long long)swap_call);
        mglSafeReleaseMetalObj(&sample_buffer);
        return;
    }

    (void)mglRenderBlitCopyTextureToBuffer(
        sample_encoder, sample_texture, 0, 0,
        clamped_origin_x, clamped_origin_y, 0u,
        sample_width, sample_height, 1u,
        sample_buffer, 0, sample_bytes_per_row, sample_bytes_per_image);
    (void)mglRenderEndBlitEncoder(sample_encoder);
    (void)mglRenderAddBufferDebugMarker(sample_buffer, "mgl_swap_sample", 0u,
                                        sample_bytes_per_image);

    MGLSwapSampleCompletion *completion =
        (MGLSwapSampleCompletion *)calloc(1, sizeof(*completion));
    if (!completion) {
        mglSafeReleaseMetalObj(&sample_buffer);
        return;
    }
    /* The buffer's +1 moves into the completion; Destroy releases it. */
    completion->sample_buffer = sample_buffer;
    completion->sample_tag = sample_tag;
    completion->swap_call = swap_call;
    completion->sample_width = sample_width;
    completion->sample_height = sample_height;
    completion->texture_width = (size_t)sample_info.width;
    completion->texture_height = (size_t)sample_info.height;
    completion->origin_x = clamped_origin_x;
    completion->origin_y = clamped_origin_y;
    completion->bytes_per_image = sample_bytes_per_image;
    if (mglRenderAddCommandBufferOwnerCompletion(
            mglSwapCommandBufferOwner(&areas), mglSwapSampleCompletion,
            completion, mglSwapSampleCompletionDestroy) != 0) {
        mglSwapSampleCompletionDestroy(completion);
    }
}

/* === entry points ======================================================= */

void mglSwapCopyRenderPassColorToDrawableIfNeeded(void *renderer, void *rp_color0,
                                                  void *drawable_texture,
                                                  uint64_t swap_call,
                                                  bool trace_swap)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    MGLRenderTextureInfo sourceInfo = {0};
    MGLRenderTextureInfo drawableInfo = {0};
    if (rp_color0) {
        (void)mglRenderGetTextureInfo(rp_color0, &sourceInfo);
    }
    if (drawable_texture) {
        (void)mglRenderGetTextureInfo(drawable_texture, &drawableInfo);
    }
    /* Diagnostic + compatibility path:
     * When swapping the default framebuffer, the active render pass should
     * target the drawable.  If it still points to an offscreen texture, copy
     * that texture into the drawable before present. */
    GLMState *gl_state = mglSwapStateForRenderer(&areas);
    if (gl_state->framebuffer == NULL &&
        !areas.core->defaultDrawableWrittenSinceLastSwap &&
        rp_color0 &&
        drawable_texture &&
        rp_color0 != drawable_texture) {
        bool traceCopyToDrawable = trace_swap ||
            (kMglSwapPresentDiagnostics &&
             (swap_call <= 12ull || (swap_call % 120ull) == 0ull));
        if (traceCopyToDrawable) {
            mglTraceLog("MGL TRACE swap.copyToDrawable.begin call=%llu src=%p fmt=%lu %lux%lu dst=%p fmt=%lu %lux%lu",
                        (unsigned long long)swap_call,
                        rp_color0,
                        (unsigned long)sourceInfo.pixel_format,
                        (unsigned long)sourceInfo.width,
                        (unsigned long)sourceInfo.height,
                        drawable_texture,
                        (unsigned long)drawableInfo.pixel_format,
                        (unsigned long)drawableInfo.width,
                        (unsigned long)drawableInfo.height);
        }

        bool canShaderCopyToDrawable =
            (sourceInfo.pixel_format == drawableInfo.pixel_format ||
             (sourceInfo.pixel_format == 70u && drawableInfo.pixel_format == 80u) ||
             (sourceInfo.pixel_format == 80u && drawableInfo.pixel_format == 70u));
        if (canShaderCopyToDrawable) {
            void *pipeline = mglBlitScaledPipelineForPixelFormat(
                renderer, drawableInfo.pixel_format);
            void *sampler = mglBlitScaledSamplerForFilter(
                renderer, (uint32_t)mglRenderNearestFilter());
            size_t copyWidth = MIN((size_t)sourceInfo.width, (size_t)drawableInfo.width);
            size_t copyHeight = MIN((size_t)sourceInfo.height, (size_t)drawableInfo.height);
            if (pipeline && sampler && copyWidth > 0 && copyHeight > 0) {
                MGLScaledBlitParams params;
                params.uvRect = (vector_float4){
                    0.0f,
                    0.0f,
                    sourceInfo.width ? ((float)copyWidth / (float)sourceInfo.width) : 0.0f,
                    sourceInfo.height ? ((float)copyHeight / (float)sourceInfo.height) : 0.0f
                };
                params.forceOpaqueAlpha = 1.0f;
                params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

                void *copyEncoder = mglSwapDiagnosticsCreateRenderEncoder(
                    mglSwapCommandBufferOwner(&areas), drawable_texture);
                if (copyEncoder) {
                    (void)mglRenderSetRenderPipelineState(copyEncoder, pipeline);
                    (void)mglRenderSetRenderBytes(
                        copyEncoder, &params, sizeof(params),
                        MGL_RENDER_BINDING_STAGE_VERTEX, 0);
                    (void)mglRenderSetRenderBytes(
                        copyEncoder, &params, sizeof(params),
                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                    (void)mglRenderSetRenderTexture(
                        copyEncoder, rp_color0,
                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                    (void)mglRenderSetRenderSampler(
                        copyEncoder, sampler,
                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                    (void)mglRenderSetRenderViewport(
                        copyEncoder, 0.0, 0.0,
                        (double)copyWidth, (double)copyHeight, 0.0, 1.0);
                    (void)mglRenderSetRenderScissor(
                        copyEncoder, 0u, 0u, copyWidth, copyHeight);
                    (void)mglRenderEncodeDraw(
                        copyEncoder,
                        &(MGLRenderDrawPlan){
                            .kind = MGL_RENDER_DRAW_ARRAY,
                            .primitive_type = 4u,
                            .vertex_start = 0,
                            .vertex_count = 4,
                            .instance_count = 1u,
                            .base_instance = 0u,
                        }, NULL, 0);
                    (void)mglRenderEndRenderEncoder(copyEncoder);
                } else {
                    fprintf(stderr,
                            "MGL WARNING: swap.copyToDrawable failed to create shader copy encoder\n");
                }
            } else {
                fprintf(stderr,
                        "MGL WARNING: swap.copyToDrawable shader copy unavailable pipeline=%p sampler=%p size=%lux%lu\n",
                        pipeline,
                        sampler,
                        (unsigned long)copyWidth,
                        (unsigned long)copyHeight);
            }
        } else {
            fprintf(stderr,
                    "MGL WARNING: swap.copyToDrawable skipped due to pixel format mismatch src=%lu dst=%lu\n",
                    (unsigned long)sourceInfo.pixel_format,
                    (unsigned long)drawableInfo.pixel_format);
        }

        if (traceCopyToDrawable) {
            mglTraceLog("MGL TRACE swap.copyToDrawable.end call=%llu",
                        (unsigned long long)swap_call);
        }
    } else if (gl_state->framebuffer == NULL &&
               areas.core->defaultDrawableWrittenSinceLastSwap &&
               rp_color0 &&
               drawable_texture &&
               rp_color0 != drawable_texture) {
        bool traceSkipCopyToDrawable = trace_swap ||
            (kMglSwapPresentDiagnostics &&
             (swap_call <= 12ull || (swap_call % 120ull) == 0ull));
        if (traceSkipCopyToDrawable) {
            mglTraceLog("MGL TRACE swap.copyToDrawable.skip call=%llu reason=default_blit_already_wrote_drawable src=%p dst=%p",
                        (unsigned long long)swap_call,
                        rp_color0,
                        drawable_texture);
        }
    }
}

void mglSwapScheduleTextureSampleDiagnostics(void *renderer, void *rp_color0,
                                             void *drawable_texture,
                                             uint64_t swap_call)
{
    /* Low-frequency dual texture sampling for black-screen diagnostics.
     * Sample both render-pass color source and drawable target so we can
     * distinguish "rendered black" from "copy/present black". */
    if (kMglSwapPresentDiagnostics &&
        ((swap_call <= 12ull && (swap_call % 3ull) == 0ull) || ((swap_call % 120ull) == 0ull))) {
        MGLRenderTextureInfo sourceInfo = {0};
        MGLRenderTextureInfo drawableInfo = {0};
        if (rp_color0) {
            (void)mglRenderGetTextureInfo(rp_color0, &sourceInfo);
        }
        if (drawable_texture) {
            (void)mglRenderGetTextureInfo(drawable_texture, &drawableInfo);
        }
        mglSwapScheduleTextureSample(renderer, rp_color0, "src.tl", 0u, 0u, swap_call);
        if (rp_color0) {
            size_t cx = ((size_t)sourceInfo.width > 8u) ? (((size_t)sourceInfo.width / 2u) - 4u) : 0u;
            size_t cy = ((size_t)sourceInfo.height > 8u) ? (((size_t)sourceInfo.height / 2u) - 4u) : 0u;
            size_t rx = ((size_t)sourceInfo.width > 8u) ? ((size_t)sourceInfo.width - 8u) : 0u;
            size_t by = ((size_t)sourceInfo.height > 8u) ? ((size_t)sourceInfo.height - 8u) : 0u;
            mglSwapScheduleTextureSample(renderer, rp_color0, "src.center", cx, cy, swap_call);
            mglSwapScheduleTextureSample(renderer, rp_color0, "src.right", rx, cy, swap_call);
            mglSwapScheduleTextureSample(renderer, rp_color0, "src.bottom", cx, by, swap_call);
        }
        if (drawable_texture != rp_color0) {
            mglSwapScheduleTextureSample(renderer, drawable_texture, "dst.tl", 0u, 0u, swap_call);
            if (drawable_texture) {
                size_t dcx = ((size_t)drawableInfo.width > 8u) ? (((size_t)drawableInfo.width / 2u) - 4u) : 0u;
                size_t dcy = ((size_t)drawableInfo.height > 8u) ? (((size_t)drawableInfo.height / 2u) - 4u) : 0u;
                size_t drx = ((size_t)drawableInfo.width > 8u) ? ((size_t)drawableInfo.width - 8u) : 0u;
                size_t dby = ((size_t)drawableInfo.height > 8u) ? ((size_t)drawableInfo.height - 8u) : 0u;
                mglSwapScheduleTextureSample(renderer, drawable_texture, "dst.center", dcx, dcy, swap_call);
                mglSwapScheduleTextureSample(renderer, drawable_texture, "dst.right", drx, dcy, swap_call);
                mglSwapScheduleTextureSample(renderer, drawable_texture, "dst.bottom", dcx, dby, swap_call);
            }
        } else {
            mglSwapScheduleTextureSample(renderer, drawable_texture, "srcdst.tl", 0u, 0u, swap_call);
            if (drawable_texture) {
                size_t sx = ((size_t)drawableInfo.width > 8u) ? (((size_t)drawableInfo.width / 2u) - 4u) : 0u;
                size_t sy = ((size_t)drawableInfo.height > 8u) ? (((size_t)drawableInfo.height / 2u) - 4u) : 0u;
                size_t srx = ((size_t)drawableInfo.width > 8u) ? ((size_t)drawableInfo.width - 8u) : 0u;
                size_t sby = ((size_t)drawableInfo.height > 8u) ? ((size_t)drawableInfo.height - 8u) : 0u;
                mglSwapScheduleTextureSample(renderer, drawable_texture, "srcdst.center", sx, sy, swap_call);
                mglSwapScheduleTextureSample(renderer, drawable_texture, "srcdst.right", srx, sy, swap_call);
                mglSwapScheduleTextureSample(renderer, drawable_texture, "srcdst.bottom", sx, sby, swap_call);
            }
        }
    }
}
