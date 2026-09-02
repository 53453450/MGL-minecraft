/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+SwapDiagnostics.m
// Swap-time diagnostic helpers extracted from MGLRenderer.m:
//   - copyRenderPassColorToDrawableIfNeeded:drawableTexture:swapCall:traceSwap:
//   - scheduleSwapTextureSampleDiagnostics:drawableTexture:swapCall:
// These methods run at mtlSwapBuffers time to (1) copy offscreen render-pass
// color into the drawable when the default framebuffer's blit path was
// bypassed, and (2) sample both source and destination textures for
// low-frequency black-screen diagnostics.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+SwapDiagnostics_Private.h"
#import "MGLRenderer+Blit_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"

typedef void (^MGLSwapCommandCompletionBlock)(
    const MGLRenderCommandBufferState *state);

static void mglSwapCommandCompletionCallback(
    void *context,
    const MGLRenderCommandBufferState *state)
{
    MGLSwapCommandCompletionBlock block =
        (__bridge MGLSwapCommandCompletionBlock)context;
    if (block) block(state);
}

static void mglSwapCommandCompletionDestroy(void *context)
{
    if (!context) return;
    (void)CFBridgingRelease(context);
}

static int mglSwapAddCommandBufferOwnerCompletion(
    void *owner,
    MGLSwapCommandCompletionBlock block)
{
    if (!owner || !block) return -1;
    MGLSwapCommandCompletionBlock copied = [block copy];
    void *context = (__bridge_retained void *)copied;
    int result = mglRenderAddCommandBufferOwnerCompletion(
        owner,
        mglSwapCommandCompletionCallback,
        context,
        mglSwapCommandCompletionDestroy);
    if (result != 0) mglSwapCommandCompletionDestroy(context);
    return result;
}

static id mglSwapDiagnosticsCreateBuffer(NSUInteger length)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer(
            length, 0u,
            "MGL Swap Diagnostic Sample", &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglSwapDiagnosticsCreateRenderEncoder(
    void *commandBufferOwner,
    id colorTexture)
{
    if (!commandBufferOwner || !colorTexture) return nil;
    MGLRenderPassState state = {0};
    state.color[0].attachment.texture = (__bridge void *)colorTexture;
    state.color[0].attachment.load_action = 0u;
    state.color[0].attachment.store_action = 1u;
    return (__bridge id)mglRenderCreateRenderEncoderBorrowed(
        commandBufferOwner, &state);
}

static void mglSwapDiagnosticsSetRenderPipeline(
    id encoder,
    id pipeline)
{
    (void)mglRenderSetRenderPipelineState(
        (__bridge void *)encoder, (__bridge void *)pipeline);
}

static void mglSwapDiagnosticsSetRenderBytes(
    id encoder,
    const void *bytes,
    NSUInteger length,
    uint32_t stage)
{
    (void)mglRenderSetRenderBytes(
        (__bridge void *)encoder, bytes, length, stage, 0);
}

static void mglSwapDiagnosticsSetFragmentTexture(
    id encoder,
    id texture)
{
    (void)mglRenderSetRenderTexture(
        (__bridge void *)encoder, (__bridge void *)texture,
        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
}

static void mglSwapDiagnosticsSetFragmentSampler(
    id encoder,
    id sampler)
{
    (void)mglRenderSetRenderSampler(
        (__bridge void *)encoder, (__bridge void *)sampler,
        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
}

static void mglSwapDiagnosticsSetViewport(
    id encoder,
    double width,
    double height)
{
    (void)mglRenderSetRenderViewport(
        (__bridge void *)encoder, 0.0, 0.0, width, height, 0.0, 1.0);
}

static void mglSwapDiagnosticsSetScissor(
    id encoder,
    NSUInteger width,
    NSUInteger height)
{
    (void)mglRenderSetRenderScissor(
        (__bridge void *)encoder, 0u, 0u, width, height);
}

static void mglSwapDiagnosticsDrawTriangleStrip(id encoder)
{
    (void)mglRenderEncodeDraw((__bridge void *)encoder,
        &(MGLRenderDrawPlan){
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = 4u,
            .vertex_start = 0,
            .vertex_count = 4,
            .instance_count = 1u,
            .base_instance = 0u,
        }, NULL, 0);
}

static void mglSwapDiagnosticsEndRenderEncoder(id encoder)
{
    (void)mglRenderEndRenderEncoder((__bridge void *)encoder);
}

static id mglSwapDiagnosticsCreateBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

static void mglSwapDiagnosticsCopyTextureToBuffer(
    id encoder,
    id texture,
    NSUInteger originX,
    NSUInteger originY,
    NSUInteger width,
    NSUInteger height,
    id buffer,
    NSUInteger bytesPerRow,
    NSUInteger bytesPerImage)
{
    (void)mglRenderBlitCopyTextureToBuffer(
        (__bridge void *)encoder, (__bridge void *)texture, 0, 0,
        originX, originY, 0u, width, height, 1u,
        (__bridge void *)buffer, 0, bytesPerRow, bytesPerImage);
}

static void mglSwapDiagnosticsEndBlitEncoder(id encoder)
{
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

@implementation MGLRenderer (SwapDiagnostics)

- (void)copyRenderPassColorToDrawableIfNeeded:(id)rpColor0
                              drawableTexture:(id)drawableTexture
                                      swapCall:(uint64_t)swapCall
                                    traceSwap:(bool)traceSwap
{
    MGLRenderTextureInfo sourceInfo = {0};
    MGLRenderTextureInfo drawableInfo = {0};
    if (rpColor0) {
        (void)mglRenderGetTextureInfo(
            (__bridge const void *)rpColor0, &sourceInfo);
    }
    if (drawableTexture) {
        (void)mglRenderGetTextureInfo(
            (__bridge const void *)drawableTexture, &drawableInfo);
    }
    // Diagnostic + compatibility path:
    // When swapping the default framebuffer, the active render pass should target the drawable.
    // If it still points to an offscreen texture, copy that texture into the drawable before present.
    if (MGL_STATE(ctx)->framebuffer == NULL &&
        !_defaultDrawableWrittenSinceLastSwap &&
        rpColor0 &&
        drawableTexture &&
        rpColor0 != drawableTexture) {
        BOOL traceCopyToDrawable = traceSwap ||
            (kMGLSwapPresentDiagnostics &&
             (swapCall <= 12ull || (swapCall % 120ull) == 0ull));
        if (traceCopyToDrawable) {
            mglTraceLogNSString(@"MGL TRACE swap.copyToDrawable.begin call=%llu src=%p fmt=%lu %lux%lu dst=%p fmt=%lu %lux%lu",
                  (unsigned long long)swapCall,
                  rpColor0,
                  (unsigned long)sourceInfo.pixel_format,
                  (unsigned long)sourceInfo.width,
                  (unsigned long)sourceInfo.height,
                  drawableTexture,
                  (unsigned long)drawableInfo.pixel_format,
                  (unsigned long)drawableInfo.width,
                  (unsigned long)drawableInfo.height);
        }

        BOOL canShaderCopyToDrawable =
            (sourceInfo.pixel_format == drawableInfo.pixel_format ||
             (sourceInfo.pixel_format == 70u && drawableInfo.pixel_format == 80u) ||
             (sourceInfo.pixel_format == 80u && drawableInfo.pixel_format == 70u));
        if (canShaderCopyToDrawable) {
                id pipeline = [self scaledBlitPipelineForPixelFormat:drawableInfo.pixel_format];
                id sampler = [self scaledBlitSamplerForFilter:GL_NEAREST];
                NSUInteger copyWidth = MIN((NSUInteger)sourceInfo.width, (NSUInteger)drawableInfo.width);
                NSUInteger copyHeight = MIN((NSUInteger)sourceInfo.height, (NSUInteger)drawableInfo.height);
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

                    id copyEncoder =
                        mglSwapDiagnosticsCreateRenderEncoder(
                            _commandState.currentCommandBufferOwner,
                            drawableTexture);
                    if (copyEncoder) {
                        mglSwapDiagnosticsSetRenderPipeline(copyEncoder, pipeline);
                        mglSwapDiagnosticsSetRenderBytes(
                            copyEncoder, &params, sizeof(params),
                            MGL_RENDER_BINDING_STAGE_VERTEX);
                        mglSwapDiagnosticsSetRenderBytes(
                            copyEncoder, &params, sizeof(params),
                            MGL_RENDER_BINDING_STAGE_FRAGMENT);
                        mglSwapDiagnosticsSetFragmentTexture(copyEncoder, rpColor0);
                        mglSwapDiagnosticsSetFragmentSampler(copyEncoder, sampler);
                        mglSwapDiagnosticsSetViewport(
                            copyEncoder, (double)copyWidth, (double)copyHeight);
                        mglSwapDiagnosticsSetScissor(
                            copyEncoder, copyWidth, copyHeight);
                        mglSwapDiagnosticsDrawTriangleStrip(copyEncoder);
                        mglSwapDiagnosticsEndRenderEncoder(copyEncoder);
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
                  (unsigned long)sourceInfo.pixel_format,
                  (unsigned long)drawableInfo.pixel_format);
        }

        if (traceCopyToDrawable) {
            mglTraceLogNSString(@"MGL TRACE swap.copyToDrawable.end call=%llu", (unsigned long long)swapCall);
        }
    } else if (MGL_STATE(ctx)->framebuffer == NULL &&
               _defaultDrawableWrittenSinceLastSwap &&
               rpColor0 &&
               drawableTexture &&
               rpColor0 != drawableTexture) {
        BOOL traceSkipCopyToDrawable = traceSwap ||
            (kMGLSwapPresentDiagnostics &&
             (swapCall <= 12ull || (swapCall % 120ull) == 0ull));
        if (traceSkipCopyToDrawable) {
            mglTraceLogNSString(@"MGL TRACE swap.copyToDrawable.skip call=%llu reason=default_blit_already_wrote_drawable src=%p dst=%p",
                  (unsigned long long)swapCall,
                  rpColor0,
                  drawableTexture);
        }
    }

}

- (void)scheduleSwapTextureSampleDiagnostics:(id)rpColor0
                             drawableTexture:(id)drawableTexture
                                     swapCall:(uint64_t)swapCall
{
    // Low-frequency dual texture sampling for black-screen diagnostics.
    // Sample both render-pass color source and drawable target so we can
    // distinguish "rendered black" from "copy/present black".
    if (kMGLSwapPresentDiagnostics &&
        ((swapCall <= 12ull && (swapCall % 3ull) == 0ull) || ((swapCall % 120ull) == 0ull))) {
        void (^scheduleTextureSample)(id, NSString *, NSUInteger, NSUInteger) =
            ^(id sampleTexture, NSString *sampleTag, NSUInteger originX, NSUInteger originY) {
                if (!sampleTexture) {
                    mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu skipped(texture=nil)",
                          sampleTag,
                          (unsigned long long)swapCall);
                    return;
                }

                MGLRenderTextureInfo sampleInfo = {0};
                if (mglRenderGetTextureInfo(
                        (__bridge const void *)sampleTexture, &sampleInfo) != 0) {
                    return;
                }
                if (sampleInfo.pixel_format != 80u &&
                    sampleInfo.pixel_format != 70u) {
                    mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu skipped(fmt=%lu tex=%lux%lu)",
                          sampleTag,
                          (unsigned long long)swapCall,
                          (unsigned long)sampleInfo.pixel_format,
                          (unsigned long)sampleInfo.width,
                          (unsigned long)sampleInfo.height);
                    return;
                }

                NSUInteger sampleWidth = MIN((NSUInteger)sampleInfo.width, 8u);
                NSUInteger sampleHeight = MIN((NSUInteger)sampleInfo.height, 8u);
                NSUInteger bytesPerPixel = 4u;
                NSUInteger sampleBytesPerRow = sampleWidth * bytesPerPixel;
                NSUInteger sampleBytesPerImage = sampleBytesPerRow * sampleHeight;
                if (sampleWidth == 0 || sampleHeight == 0 || sampleBytesPerImage == 0) {
                    mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu skipped(invalid-size tex=%lux%lu)",
                          sampleTag,
                          (unsigned long long)swapCall,
                          (unsigned long)sampleInfo.width,
                          (unsigned long)sampleInfo.height);
                    return;
                }

                NSUInteger clampedOriginX = originX;
                NSUInteger clampedOriginY = originY;
                if (clampedOriginX + sampleWidth > (NSUInteger)sampleInfo.width) {
                    clampedOriginX = ((NSUInteger)sampleInfo.width > sampleWidth)
                        ? ((NSUInteger)sampleInfo.width - sampleWidth)
                        : 0u;
                }
                if (clampedOriginY + sampleHeight > (NSUInteger)sampleInfo.height) {
                    clampedOriginY = ((NSUInteger)sampleInfo.height > sampleHeight)
                        ? ((NSUInteger)sampleInfo.height - sampleHeight)
                        : 0u;
                }

                id sampleBuffer =
                    mglSwapDiagnosticsCreateBuffer(sampleBytesPerImage);
                if (!sampleBuffer) {
                    NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(alloc size=%lu)",
                          sampleTag,
                          (unsigned long long)swapCall,
                          (unsigned long)sampleBytesPerImage);
                    return;
                }

                id sampleEncoder =
                    mglSwapDiagnosticsCreateBlitEncoder(
                        _commandState.currentCommandBufferOwner);
                if (!sampleEncoder) {
                    NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(create blit encoder)",
                          sampleTag,
                          (unsigned long long)swapCall);
                    return;
                }

                mglSwapDiagnosticsCopyTextureToBuffer(
                    sampleEncoder, sampleTexture,
                    clampedOriginX, clampedOriginY,
                    sampleWidth, sampleHeight, sampleBuffer,
                    sampleBytesPerRow, sampleBytesPerImage);
                mglSwapDiagnosticsEndBlitEncoder(sampleEncoder);

                uint64_t sampleSwapCall = swapCall;
                NSString *sampleTagCopy = [sampleTag copy];
                NSUInteger sampleTexWidth = (NSUInteger)sampleInfo.width;
                NSUInteger sampleTexHeight = (NSUInteger)sampleInfo.height;
                NSUInteger sampleOriginX = clampedOriginX;
                NSUInteger sampleOriginY = clampedOriginY;
                (void)mglRenderAddBufferDebugMarker(
                    (__bridge void *)sampleBuffer,
                    "mgl_swap_sample", 0u, sampleBytesPerImage);
                mglSwapAddCommandBufferOwnerCompletion(
                    _commandState.currentCommandBufferOwner,
                    ^(const MGLRenderCommandBufferState *sampleState) {
                    NSString *sampleError = sampleState->has_error
                        ? [NSString stringWithFormat:@"%s (domain=%s code=%lld)",
                             sampleState->error_description,
                             sampleState->error_domain,
                             (long long)sampleState->error_code]
                        : nil;
                    void *sampleContents = NULL;
                    uint64_t sampleBufferLength = 0;
                    (void)mglRenderGetBufferContents(
                        (__bridge void *)sampleBuffer,
                        &sampleContents, &sampleBufferLength);
                    const uint8_t *p = sampleBufferLength >= sampleBytesPerImage
                        ? (const uint8_t *)sampleContents : NULL;
                    if (!p) {
                        mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu unavailable(contents=nil) status=%s error=%@",
                              sampleTagCopy,
                              (unsigned long long)sampleSwapCall,
                              mglCommandBufferStatusName(
                                  sampleState->status),
                              sampleError);
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

                    mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu tex=%lux%lu origin=(%lu,%lu) sample=%lux%lu "
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
                          mglCommandBufferStatusName(
                              sampleState->status),
                          sampleError);

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
                            mglTraceLogNSString(@"MGL TRACE swap.sample.center_stable firstPixel=0x%08x run=%llu solid=%d diff=%lu",
                                  firstPixel,
                                  (unsigned long long)s_sameCenterPixelRun,
                                  appearsSolid ? 1 : 0,
                                  (unsigned long)diffFromFirst);
                        }
                    }
                });
            };

        MGLRenderTextureInfo sourceInfo = {0};
        MGLRenderTextureInfo drawableInfo = {0};
        if (rpColor0) {
            (void)mglRenderGetTextureInfo(
                (__bridge const void *)rpColor0, &sourceInfo);
        }
        if (drawableTexture) {
            (void)mglRenderGetTextureInfo(
                (__bridge const void *)drawableTexture, &drawableInfo);
        }
        scheduleTextureSample(rpColor0, @"src.tl", 0u, 0u);
        if (rpColor0) {
            NSUInteger cx = ((NSUInteger)sourceInfo.width > 8u) ? (((NSUInteger)sourceInfo.width / 2u) - 4u) : 0u;
            NSUInteger cy = ((NSUInteger)sourceInfo.height > 8u) ? (((NSUInteger)sourceInfo.height / 2u) - 4u) : 0u;
            NSUInteger rx = ((NSUInteger)sourceInfo.width > 8u) ? ((NSUInteger)sourceInfo.width - 8u) : 0u;
            NSUInteger by = ((NSUInteger)sourceInfo.height > 8u) ? ((NSUInteger)sourceInfo.height - 8u) : 0u;
            scheduleTextureSample(rpColor0, @"src.center", cx, cy);
            scheduleTextureSample(rpColor0, @"src.right", rx, cy);
            scheduleTextureSample(rpColor0, @"src.bottom", cx, by);
        }
        if (drawableTexture != rpColor0) {
            scheduleTextureSample(drawableTexture, @"dst.tl", 0u, 0u);
            if (drawableTexture) {
                NSUInteger dcx = ((NSUInteger)drawableInfo.width > 8u) ? (((NSUInteger)drawableInfo.width / 2u) - 4u) : 0u;
                NSUInteger dcy = ((NSUInteger)drawableInfo.height > 8u) ? (((NSUInteger)drawableInfo.height / 2u) - 4u) : 0u;
                NSUInteger drx = ((NSUInteger)drawableInfo.width > 8u) ? ((NSUInteger)drawableInfo.width - 8u) : 0u;
                NSUInteger dby = ((NSUInteger)drawableInfo.height > 8u) ? ((NSUInteger)drawableInfo.height - 8u) : 0u;
                scheduleTextureSample(drawableTexture, @"dst.center", dcx, dcy);
                scheduleTextureSample(drawableTexture, @"dst.right", drx, dcy);
                scheduleTextureSample(drawableTexture, @"dst.bottom", dcx, dby);
            }
        } else {
            scheduleTextureSample(drawableTexture, @"srcdst.tl", 0u, 0u);
            if (drawableTexture) {
                NSUInteger sx = ((NSUInteger)drawableInfo.width > 8u) ? (((NSUInteger)drawableInfo.width / 2u) - 4u) : 0u;
                NSUInteger sy = ((NSUInteger)drawableInfo.height > 8u) ? (((NSUInteger)drawableInfo.height / 2u) - 4u) : 0u;
                NSUInteger srx = ((NSUInteger)drawableInfo.width > 8u) ? ((NSUInteger)drawableInfo.width - 8u) : 0u;
                NSUInteger sby = ((NSUInteger)drawableInfo.height > 8u) ? ((NSUInteger)drawableInfo.height - 8u) : 0u;
                scheduleTextureSample(drawableTexture, @"srcdst.center", sx, sy);
                scheduleTextureSample(drawableTexture, @"srcdst.right", srx, sy);
                scheduleTextureSample(drawableTexture, @"srcdst.bottom", sx, sby);
            }
        }
    }

}

@end
