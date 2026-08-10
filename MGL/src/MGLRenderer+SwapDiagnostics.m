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
#include "mgl_render_cpp_objc.h"

static BOOL mglSwapDiagnosticsUsesMetalCpp(void)
{
    return mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
           mglRenderCppGetDevice() != NULL;
}

static id<MTLBuffer> mglSwapDiagnosticsCreateBuffer(
    id<MTLDevice> device,
    NSUInteger length)
{
    if (mglSwapDiagnosticsUsesMetalCpp()) {
        void *bufferCPP = NULL;
        if (mglRenderCppCreateBuffer(
                length, MTLResourceStorageModeShared,
                "MGL Swap Diagnostic Sample", &bufferCPP) == 0 &&
            bufferCPP) {
            return (__bridge_transfer id<MTLBuffer>)bufferCPP;
        }
    }
    return [device newBufferWithLength:length
                               options:MTLResourceStorageModeShared];
}

static id<MTLRenderCommandEncoder> mglSwapDiagnosticsCreateRenderEncoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> colorTexture)
{
    if (mglSwapDiagnosticsUsesMetalCpp() && commandBuffer && colorTexture) {
        void *encoderCPP = NULL;
        MGLRenderCppRenderPassState state = {0};
        state.color[0].attachment.texture =
            (__bridge void *)colorTexture;
        state.color[0].attachment.load_action = MTLLoadActionDontCare;
        state.color[0].attachment.store_action = MTLStoreActionStore;
        if (mglRenderCppCreateRenderEncoderFromState(
                (__bridge void *)commandBuffer, &state, &encoderCPP) == 0 &&
            encoderCPP) {
            return (__bridge id<MTLRenderCommandEncoder>)encoderCPP;
        }
    }
    MTLRenderPassDescriptor *descriptor =
        [MTLRenderPassDescriptor renderPassDescriptor];
    descriptor.colorAttachments[0].texture = colorTexture;
    descriptor.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
    return commandBuffer
        ? [commandBuffer renderCommandEncoderWithDescriptor:descriptor]
        : nil;
}

static void mglSwapDiagnosticsSetRenderPipeline(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLRenderPipelineState> pipeline)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppSetRenderPipelineState((__bridge void *)encoder,
                                            (__bridge void *)pipeline) == 0) {
        return;
    }
    [encoder setRenderPipelineState:pipeline];
}

static void mglSwapDiagnosticsSetRenderBytes(
    id<MTLRenderCommandEncoder> encoder,
    const void *bytes,
    NSUInteger length,
    uint32_t stage)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppSetRenderBytes((__bridge void *)encoder, bytes, length,
                                   stage, 0) == 0) {
        return;
    }
    if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
        [encoder setVertexBytes:bytes length:length atIndex:0];
    } else {
        [encoder setFragmentBytes:bytes length:length atIndex:0];
    }
}

static void mglSwapDiagnosticsSetFragmentTexture(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLTexture> texture)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppSetRenderTexture(
            (__bridge void *)encoder, (__bridge void *)texture,
            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0) == 0) {
        return;
    }
    [encoder setFragmentTexture:texture atIndex:0];
}

static void mglSwapDiagnosticsSetFragmentSampler(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLSamplerState> sampler)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppSetRenderSampler(
            (__bridge void *)encoder, (__bridge void *)sampler,
            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0) == 0) {
        return;
    }
    [encoder setFragmentSamplerState:sampler atIndex:0];
}

static void mglSwapDiagnosticsSetViewport(
    id<MTLRenderCommandEncoder> encoder,
    MTLViewport viewport)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppSetRenderViewport(
            (__bridge void *)encoder, viewport.originX, viewport.originY,
            viewport.width, viewport.height, viewport.znear, viewport.zfar) == 0) {
        return;
    }
    [encoder setViewport:viewport];
}

static void mglSwapDiagnosticsSetScissor(
    id<MTLRenderCommandEncoder> encoder,
    MTLScissorRect scissor)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppSetRenderScissor(
            (__bridge void *)encoder, scissor.x, scissor.y,
            scissor.width, scissor.height) == 0) {
        return;
    }
    [encoder setScissorRect:scissor];
}

static void mglSwapDiagnosticsDrawTriangleStrip(
    id<MTLRenderCommandEncoder> encoder)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppDrawPrimitives(
            (__bridge void *)encoder, (uint32_t)MTLPrimitiveTypeTriangleStrip,
            0, 4, 1, 0) == 0) {
        return;
    }
    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                vertexStart:0
                vertexCount:4];
}

static void mglSwapDiagnosticsEndRenderEncoder(
    id<MTLRenderCommandEncoder> encoder)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppEndRenderEncoder((__bridge void *)encoder) == 0) {
        return;
    }
    [encoder endEncoding];
}

static id<MTLBlitCommandEncoder> mglSwapDiagnosticsCreateBlitEncoder(
    id<MTLCommandBuffer> commandBuffer)
{
    if (mglSwapDiagnosticsUsesMetalCpp()) {
        void *encoderCPP = NULL;
        if (mglRenderCppCreateBlitEncoder((__bridge void *)commandBuffer,
                                          &encoderCPP) == 0 &&
            encoderCPP) {
            return (__bridge id<MTLBlitCommandEncoder>)encoderCPP;
        }
    }
    return [commandBuffer blitCommandEncoder];
}

static void mglSwapDiagnosticsCopyTextureToBuffer(
    id<MTLBlitCommandEncoder> encoder,
    id<MTLTexture> texture,
    MTLOrigin origin,
    MTLSize size,
    id<MTLBuffer> buffer,
    NSUInteger bytesPerRow,
    NSUInteger bytesPerImage)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppBlitCopyTextureToBuffer(
            (__bridge void *)encoder, (__bridge void *)texture, 0, 0,
            origin.x, origin.y, origin.z, size.width, size.height, size.depth,
            (__bridge void *)buffer, 0, bytesPerRow, bytesPerImage) == 0) {
        return;
    }
    [encoder copyFromTexture:texture
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:origin
                  sourceSize:size
                    toBuffer:buffer
           destinationOffset:0
      destinationBytesPerRow:bytesPerRow
    destinationBytesPerImage:bytesPerImage];
}

static void mglSwapDiagnosticsEndBlitEncoder(
    id<MTLBlitCommandEncoder> encoder)
{
    if (mglSwapDiagnosticsUsesMetalCpp() &&
        mglRenderCppEndBlitEncoder((__bridge void *)encoder) == 0) {
        return;
    }
    [encoder endEncoding];
}

@implementation MGLRenderer (SwapDiagnostics)

- (void)copyRenderPassColorToDrawableIfNeeded:(id<MTLTexture>)rpColor0
                              drawableTexture:(id<MTLTexture>)drawableTexture
                                      swapCall:(uint64_t)swapCall
                                    traceSwap:(bool)traceSwap
{
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

                    id<MTLRenderCommandEncoder> copyEncoder =
                        mglSwapDiagnosticsCreateRenderEncoder(
                            _renderPassManager.state->currentCommandBuffer,
                            drawableTexture);
                    if (copyEncoder) {
                        mglSwapDiagnosticsSetRenderPipeline(copyEncoder, pipeline);
                        mglSwapDiagnosticsSetRenderBytes(
                            copyEncoder, &params, sizeof(params),
                            MGL_RENDER_CPP_BINDING_STAGE_VERTEX);
                        mglSwapDiagnosticsSetRenderBytes(
                            copyEncoder, &params, sizeof(params),
                            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT);
                        mglSwapDiagnosticsSetFragmentTexture(copyEncoder, rpColor0);
                        mglSwapDiagnosticsSetFragmentSampler(copyEncoder, sampler);
                        mglSwapDiagnosticsSetViewport(copyEncoder, (MTLViewport){
                            .originX = 0.0,
                            .originY = 0.0,
                            .width = (double)copyWidth,
                            .height = (double)copyHeight,
                            .znear = 0.0,
                            .zfar = 1.0
                        });
                        mglSwapDiagnosticsSetScissor(copyEncoder, (MTLScissorRect){
                            .x = 0,
                            .y = 0,
                            .width = copyWidth,
                            .height = copyHeight
                        });
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
                  (unsigned long)rpColor0.pixelFormat,
                  (unsigned long)drawableTexture.pixelFormat);
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
                    mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu skipped(texture=nil)",
                          sampleTag,
                          (unsigned long long)swapCall);
                    return;
                }

                if (sampleTexture.pixelFormat != MTLPixelFormatBGRA8Unorm &&
                    sampleTexture.pixelFormat != MTLPixelFormatRGBA8Unorm) {
                    mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu skipped(fmt=%lu tex=%lux%lu)",
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
                    mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu skipped(invalid-size tex=%lux%lu)",
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

                id<MTLBuffer> sampleBuffer =
                    mglSwapDiagnosticsCreateBuffer(_device,
                                                   sampleBytesPerImage);
                if (!sampleBuffer) {
                    NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(alloc size=%lu)",
                          sampleTag,
                          (unsigned long long)swapCall,
                          (unsigned long)sampleBytesPerImage);
                    return;
                }

                id<MTLBlitCommandEncoder> sampleEncoder =
                    mglSwapDiagnosticsCreateBlitEncoder(
                        _renderPassManager.state->currentCommandBuffer);
                if (!sampleEncoder) {
                    NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(create blit encoder)",
                          sampleTag,
                          (unsigned long long)swapCall);
                    return;
                }

                mglSwapDiagnosticsCopyTextureToBuffer(
                    sampleEncoder, sampleTexture,
                    MTLOriginMake(clampedOriginX, clampedOriginY, 0),
                    MTLSizeMake(sampleWidth, sampleHeight, 1), sampleBuffer,
                    sampleBytesPerRow, sampleBytesPerImage);
                mglSwapDiagnosticsEndBlitEncoder(sampleEncoder);

                uint64_t sampleSwapCall = swapCall;
                NSString *sampleTagCopy = [sampleTag copy];
                NSUInteger sampleTexWidth = (NSUInteger)sampleTexture.width;
                NSUInteger sampleTexHeight = (NSUInteger)sampleTexture.height;
                NSUInteger sampleOriginX = clampedOriginX;
                NSUInteger sampleOriginY = clampedOriginY;
                [sampleBuffer addDebugMarker:@"mgl_swap_sample" range:NSMakeRange(0, sampleBytesPerImage)];
                mglRenderAddCommandBufferCompletion(
                    _renderPassManager.state->currentCommandBuffer,
                    ^(const MGLRenderCppCommandBufferState *sampleState) {
                    NSString *sampleError = sampleState->has_error
                        ? [NSString stringWithFormat:@"%s (domain=%s code=%lld)",
                             sampleState->error_description,
                             sampleState->error_domain,
                             (long long)sampleState->error_code]
                        : nil;
                    const uint8_t *p = (const uint8_t *)sampleBuffer.contents;
                    if (!p) {
                        mglTraceLogNSString(@"MGL TRACE swap.sample.%@ call=%llu unavailable(contents=nil) status=%s error=%@",
                              sampleTagCopy,
                              (unsigned long long)sampleSwapCall,
                              mglCommandBufferStatusName(
                                  (MTLCommandBufferStatus)sampleState->status),
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
                              (MTLCommandBufferStatus)sampleState->status),
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

@end
