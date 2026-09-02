/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Blit.m
// Blit/copy/resolve operations extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Blit_Private.h"
#include "mgl_env_flag.h"
#include "mgl_aux_assets.h"

/* Shared state for mtlBlitFramebuffer color blit helpers.
 * Filled after attachment resolution and clip computation, then
 * passed to the integer / scaled / direct-copy helpers. */
typedef struct MGLBlitColorState {
    GLMContext glm_ctx;
    Framebuffer *readfbo;
    Framebuffer *drawfbo;
    GLenum filter;
    FBOAttachment *readFBOAttachment;
    FBOAttachment *drawFBOAttachment;
    Texture *readTextureObject;
    Texture *drawTextureObject;
    MGLMetalAttachmentSubresource readSubresource;
    MGLMetalAttachmentSubresource drawSubresource;
    id readtexid;
    id drawtexid;
    NSUInteger srcTexW, srcTexH, dstTexW, dstTexH;
    BOOL needsFormatConversionBlit;
    BOOL needsRenderTargetSyncBlit;
    BOOL didMsaaResolve;
    BOOL blitNeedsFlip;
    BOOL needsScaledBlit;
    BOOL srcXForward, srcYForward, dstXForward, dstYForward;
    double srcMinX, srcMaxX, srcMinY, srcMaxY;
    double dstMinX, dstMaxX, dstMinY, dstMaxY;
    double srcW, srcH, dstW, dstH;
    NSInteger copySrcX, copySrcY, copyDstX, copyDstY, copyW, copyH;
    NSInteger srcMetalY, dstMetalY;
    double scaledDstMetalY;
} MGLBlitColorState;

static MGLRenderTextureInfo mglBlitTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

static MGLSizeValue mglBlitSize(NSUInteger width, NSUInteger height,
                                NSUInteger depth)
{
    return (MGLSizeValue){width, height, depth};
}

static MGLOriginValue mglBlitOrigin(NSUInteger x, NSUInteger y, NSUInteger z)
{
    return (MGLOriginValue){(int64_t)x, (int64_t)y, (int64_t)z};
}

static MGLRegionValue mglBlitRegion2D(NSUInteger x, NSUInteger y,
                                      NSUInteger width, NSUInteger height)
{
    return (MGLRegionValue){mglBlitOrigin(x, y, 0),
                            mglBlitSize(width, height, 1)};
}

static MGLRegionValue mglBlitRegion3D(NSUInteger x, NSUInteger y,
                                      NSUInteger z, NSUInteger width,
                                      NSUInteger height, NSUInteger depth)
{
    return (MGLRegionValue){mglBlitOrigin(x, y, z),
                            mglBlitSize(width, height, depth)};
}

static id mglBlitCreateBuffer(id device,
                                         NSUInteger length,
                                         uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglBlitCreateBufferWithBytes(
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

static id mglBlitCreateTexture(
    id device,
    const MGLRenderTextureDescriptorState *descriptor)
{
    (void)device;
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(
            descriptor, NULL, &texture) == 0 &&
        texture) {
        return (__bridge_transfer id)texture;
    }
    return nil;
}

static id mglBlitCreateTextureView(
    id texture,
    uint32_t pixelFormat,
    uint32_t textureType,
    NSRange levels,
    NSRange slices)
{
    void *view = NULL;
    if (mglRenderCreateTextureViewRange(
            (__bridge void *)texture, (uint32_t)pixelFormat,
            (uint32_t)textureType, levels.location, levels.length,
            slices.location, slices.length, 0, 0, 0, 0, 0,
            &view) == 0 && view) {
        return (__bridge_transfer id)view;
    }
    return nil;
}

static void mglBlitReplaceTextureRegion(id texture,
                                        MGLRegionValue region,
                                        NSUInteger level,
                                        NSUInteger slice,
                                        const void *bytes,
                                        NSUInteger bytesPerRow,
                                        NSUInteger bytesPerImage,
                                        BOOL useSlice)
{
    (void)mglRenderTextureReplaceRegion(
        (__bridge void *)texture,
        region.origin.x, region.origin.y, region.origin.z,
        region.size.width, region.size.height, region.size.depth,
        level, slice, bytes, bytesPerRow, bytesPerImage,
        useSlice ? 1 : 0);
}

static void mglBlitGetTextureBytes(id texture,
                                   void *bytes,
                                   NSUInteger bytesPerRow,
                                   NSUInteger bytesPerImage,
                                   MGLRegionValue region,
                                   NSUInteger level,
                                   NSUInteger slice,
                                   BOOL useSlice)
{
    (void)mglRenderTextureGetBytes(
        (__bridge void *)texture, bytes, bytesPerRow, bytesPerImage,
        region.origin.x, region.origin.y, region.origin.z,
        region.size.width, region.size.height, region.size.depth,
        level, slice, useSlice ? 1 : 0);
}

static id mglBlitCreateSampler(
    id device,
    uint32_t nearest)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCreateFilterSampler(nearest, &sampler) == 0 && sampler) {
        return (__bridge_transfer id)sampler;
    }
    return nil;
}

static id mglBlitCreateDepthStencilState(
    id device,
    const MGLRenderDepthStencilDescriptorState *descriptor)
{
    (void)device;
    void *state = NULL;
    if (mglRenderCreateDepthStencilStateFromState(
            descriptor, &state) == 0 && state) {
        return (__bridge_transfer id)state;
    }
    return nil;
}

static id mglBlitCreateRenderEncoder(
    MGLCommandState *commandState,
    const MGLRenderPassState *state)
{
    if (!state) return nil;
    void *encoder = NULL;
    if (mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
            commandState->currentCommandBufferOwner,
            state, &encoder) == 0 && encoder) {
        return (__bridge id)encoder;
    }
    return nil;
}

static MGLRenderPassState mglBlitDefaultRenderPassState(void)
{
    MGLRenderPassState state;
    mglRenderInitDefaultRenderPassState(&state);
    return state;
}

static MGLRenderPassAttachmentState mglBlitRenderPassAttachment(
    id texture,
    NSUInteger level,
    NSUInteger slice,
    NSUInteger depthPlane,
    uint32_t loadAction,
    uint32_t storeAction)
{
    MGLRenderPassAttachmentState attachment = {0};
    attachment.texture = (__bridge void *)texture;
    attachment.level = level;
    attachment.slice = slice;
    attachment.depth_plane = depthPlane;
    attachment.load_action = (uint32_t)loadAction;
    attachment.store_action = (uint32_t)storeAction;
    return attachment;
}

static void mglBlitEndRenderEncoder(id encoder)
{
    if (!encoder) return;
    (void)mglRenderEndRenderEncoder((__bridge void *)encoder);
}

static void mglBlitSetRenderPipeline(id encoder,
                                     id pipeline)
{
    (void)mglRenderSetRenderPipelineState(
        (__bridge void *)encoder, (__bridge void *)pipeline);
}

static void mglBlitSetDepthStencil(id encoder,
                                   id state)
{
    (void)mglRenderSetRenderDepthStencilState(
        (__bridge void *)encoder, (__bridge void *)state);
}

static void mglBlitSetRenderBytes(id encoder,
                                  const void *bytes,
                                  NSUInteger length,
                                  uint32_t stage,
                                  NSUInteger index)
{
    (void)mglRenderSetRenderBytes(
        (__bridge void *)encoder, bytes, length, stage, (uint32_t)index);
}

static void mglBlitSetRenderTexture(id encoder,
                                    id texture,
                                    uint32_t stage,
                                    NSUInteger index)
{
    (void)mglRenderSetRenderTexture(
        (__bridge void *)encoder, (__bridge void *)texture, stage,
        (uint32_t)index);
}

static void mglBlitSetRenderSampler(id encoder,
                                    id sampler,
                                    uint32_t stage,
                                    NSUInteger index)
{
    (void)mglRenderSetRenderSampler(
        (__bridge void *)encoder, (__bridge void *)sampler, stage,
        (uint32_t)index);
}

static void mglBlitSetRenderViewport(id encoder,
                                     MGLViewportValue viewport)
{
    (void)mglRenderSetRenderViewport(
        (__bridge void *)encoder, viewport.origin_x, viewport.origin_y,
        viewport.width, viewport.height, viewport.znear, viewport.zfar);
}

static void mglBlitSetRenderScissor(id encoder,
                                    MGLScissorRectValue rect)
{
    (void)mglRenderSetRenderScissor(
        (__bridge void *)encoder, rect.x, rect.y, rect.width, rect.height);
}

static void mglBlitDrawPrimitives(id encoder,
                                  uint32_t primitiveType,
                                  NSUInteger vertexStart,
                                  NSUInteger vertexCount)
{
    (void)mglRenderEncodeDraw((__bridge void *)encoder,
        &(MGLRenderDrawPlan){
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = 1u,
            .base_instance = 0u,
        }, NULL, 0);
}

static void mglBlitEndComputeEncoder(id encoder)
{
    if (!encoder) return;
    (void)mglRenderEndComputeEncoder((__bridge void *)encoder);
}

static void mglBlitSetComputePipeline(id encoder,
                                      id pipeline)
{
    (void)mglRenderSetComputePipelineState(
        (__bridge void *)encoder, (__bridge void *)pipeline);
}

static void mglBlitSetComputeTexture(id encoder,
                                     id texture,
                                     NSUInteger index)
{
    (void)mglRenderSetComputeTexture(
        (__bridge void *)encoder, (__bridge void *)texture,
        (uint32_t)index);
}

static void mglBlitSetComputeBytes(id encoder,
                                   const void *bytes,
                                   NSUInteger length,
                                   NSUInteger index)
{
    (void)mglRenderSetComputeBytes(
        (__bridge void *)encoder, bytes, length, (uint32_t)index);
}

static void mglBlitDispatchThreads(id encoder,
                                    MGLSizeValue threads,
                                    MGLSizeValue threadgroup)
{
    (void)mglRenderDispatchComputeThreads(
            (__bridge void *)encoder,
            (uint32_t)threads.width, (uint32_t)threads.height,
            (uint32_t)threads.depth,
            (uint32_t)threadgroup.width, (uint32_t)threadgroup.height,
            (uint32_t)threadgroup.depth);
}

static void mglBlitEndBlitEncoder(id encoder)
{
    if (!encoder) return;
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

static void mglBlitCopyTexture(id encoder,
                               id source,
                               NSUInteger sourceSlice,
                               NSUInteger sourceLevel,
                               MGLOriginValue sourceOrigin,
                               MGLSizeValue sourceSize,
                               id destination,
                               NSUInteger destinationSlice,
                               NSUInteger destinationLevel,
                               MGLOriginValue destinationOrigin)
{
    (void)mglRenderBlitCopyTexture(
            (__bridge void *)encoder, (__bridge void *)source, sourceSlice,
            sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
            sourceSize.width, sourceSize.height, sourceSize.depth,
            (__bridge void *)destination, destinationSlice, destinationLevel,
            destinationOrigin.x, destinationOrigin.y, destinationOrigin.z);
}

static void mglBlitCopyTextureToBuffer(id encoder,
                                       id source,
                                       NSUInteger sourceSlice,
                                       NSUInteger sourceLevel,
                                       MGLOriginValue sourceOrigin,
                                       MGLSizeValue sourceSize,
                                       id destination,
                                       NSUInteger destinationOffset,
                                       NSUInteger bytesPerRow,
                                       NSUInteger bytesPerImage)
{
    (void)mglRenderBlitCopyTextureToBuffer(
            (__bridge void *)encoder, (__bridge void *)source, sourceSlice,
            sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
            sourceSize.width, sourceSize.height, sourceSize.depth,
            (__bridge void *)destination, destinationOffset, bytesPerRow,
            bytesPerImage);
}

static void mglBlitCopyBufferToTexture(id encoder,
                                       id source,
                                       NSUInteger sourceOffset,
                                       NSUInteger bytesPerRow,
                                       NSUInteger bytesPerImage,
                                       MGLSizeValue sourceSize,
                                       id destination,
                                       NSUInteger destinationSlice,
                                       NSUInteger destinationLevel,
                                       MGLOriginValue destinationOrigin)
{
    (void)mglRenderBlitCopyBufferToTexture(
            (__bridge void *)encoder, (__bridge void *)source, sourceOffset,
            bytesPerRow, bytesPerImage, sourceSize.width, sourceSize.height,
            sourceSize.depth, (__bridge void *)destination, destinationSlice,
            destinationLevel, destinationOrigin.x, destinationOrigin.y,
            destinationOrigin.z);
}

static void mglBlitSynchronizeTexture(id encoder,
                                      id texture,
                                      NSUInteger slice,
                                      NSUInteger level)
{
    (void)mglRenderBlitSynchronizeTexture(
        (__bridge void *)encoder, (__bridge void *)texture, slice, level);
}

static id mglLookupAuxComputePipeline(
    uint32_t kind, uint64_t variant)
{
    void *pipeline = NULL;
    if (mglRenderGetOrCreateAuxComputePipeline(
            NULL, kind, variant, &pipeline, NULL, 0) == 0 && pipeline) {
        return (__bridge_transfer id)pipeline;
    }
    return nil;
}

static id mglCreateAuxComputePipelineFromAsset(
    const char *assetName, const char *entryName,
    uint32_t kind, uint64_t variant, NSError **error)
{
    const MGLAuxShaderAsset *asset = mglAuxShaderAssetFind(assetName);
    if (!asset || !asset->data || asset->size == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"MGLBlitAuxAsset"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    [NSString stringWithFormat:
                                                        @"aux shader asset '%s' missing", assetName]}];
        }
        return nil;
    }
    void *pipeline = NULL;
    char message[512] = {0};
    if (mglRenderGetOrCreateAuxComputePipelineFromMetallib(
            asset->data, asset->size, asset->hash, entryName,
            kind, variant, &pipeline, message, sizeof(message)) == 0 &&
        pipeline) {
        return (__bridge_transfer id)pipeline;
    }
    if (error) {
        NSString *description = message[0]
            ? [NSString stringWithUTF8String:message]
            : @"Metal-cpp auxiliary compute pipeline creation failed";
        *error = [NSError errorWithDomain:@"MGLBlitPipeline"
                                     code:1
                                 userInfo:@{NSLocalizedDescriptionKey:
                                                description}];
    }
    return nil;
}

static id mglCreateAuxRenderPipelineFromAsset(
    const char *assetName, const char *vsEntry, const char *fsEntry,
    uint32_t kind, uint64_t variant,
    uint32_t colorFormat, uint32_t depthFormat,
    uint32_t stencilFormat, uint32_t colorWriteMask,
    uint32_t rasterSampleCount, NSError **error)
{
    const MGLAuxShaderAsset *asset = mglAuxShaderAssetFind(assetName);
    if (!asset || !asset->data || asset->size == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"MGLBlitAuxAsset"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    [NSString stringWithFormat:
                                                        @"aux shader asset '%s' missing", assetName]}];
        }
        return nil;
    }
    void *pipeline = NULL;
    char message[512] = {0};
    int icbEnabled = mgl_env_flag_enabled("MGL_ENABLE_ICB_PIPELINES");
    if (mglRenderGetOrCreateAuxRenderPipelineFromMetallib(
            asset->data, asset->size, asset->hash,
            vsEntry, fsEntry, kind, variant,
            (uint32_t)colorFormat, (uint32_t)depthFormat,
            (uint32_t)stencilFormat, (uint32_t)colorWriteMask, icbEnabled,
            rasterSampleCount, &pipeline, message, sizeof(message)) == 0 &&
        pipeline) {
        return (__bridge_transfer id)pipeline;
    }
    if (error) {
        NSString *description = message[0]
            ? [NSString stringWithUTF8String:message]
            : @"Metal-cpp auxiliary render pipeline creation failed";
        *error = [NSError errorWithDomain:@"MGLBlitPipeline"
                                     code:2
                                 userInfo:@{NSLocalizedDescriptionKey:
                                                description}];
    }
    return nil;
}

static id mglLookupAuxRenderPipeline(
    uint32_t kind, uint64_t variant,
    uint32_t colorFormat, uint32_t depthFormat,
    uint32_t stencilFormat, uint32_t colorWriteMask,
    uint32_t rasterSampleCount)
{
    void *pipeline = NULL;
    int icbEnabled = mgl_env_flag_enabled("MGL_ENABLE_ICB_PIPELINES");
    if (mglRenderGetOrCreateAuxRenderPipeline(
            NULL, NULL, kind, variant, (uint32_t)colorFormat,
            (uint32_t)depthFormat, (uint32_t)stencilFormat,
            (uint32_t)colorWriteMask, icbEnabled, rasterSampleCount,
            &pipeline, NULL, 0) == 0 && pipeline) {
        return (__bridge_transfer id)pipeline;
    }
    return nil;
}

@implementation MGLRenderer (Blit)
- (id)scaledBlitSamplerForFilter:(GLuint)filter
{
    BOOL wantsNearest = (filter == GL_NEAREST);
    MGLRendererBackendBlitCacheKind cacheKind = wantsNearest
        ? MGL_RENDERER_BACKEND_BLIT_CACHE_NEAREST_SAMPLER
        : MGL_RENDERER_BACKEND_BLIT_CACHE_LINEAR_SAMPLER;
    id cached = (__bridge id)
        mglRendererBackendGetBlitCachedObject(_backend, cacheKind);
    if (cached) {
        return cached;
    }

    id sampler = mglBlitCreateSampler(_device, wantsNearest ? 1u : 0u);
    if (!sampler) {
        NSLog(@"MGL ERROR: failed to create scaled blit sampler filter=0x%x", filter);
        return nil;
    }

    if (mglRendererBackendSetBlitCachedObject(
            _backend, cacheKind, (__bridge void *)sampler) != 0) {
        return nil;
    }
    return (__bridge id)
        mglRendererBackendGetBlitCachedObject(_backend, cacheKind);
}

- (id)scaledBlitPipelineForPixelFormat:(uint32_t)pixelFormat
{
    if (pixelFormat == MGLPixelFormatInvalid || pixelFormat == 0) {
        pixelFormat = MGLPixelFormatBGRA8Unorm;
    }

    uint64_t variant = (uint64_t)pixelFormat;
    id cached =
        mglLookupAuxRenderPipeline(
            MGL_RENDER_AUX_RENDER_SCALED_BLIT, variant,
            pixelFormat, MGLPixelFormatInvalid, MGLPixelFormatInvalid,
            MGLColorWriteMaskAll, 1u);
    if (cached) return cached;

    NSError *error = nil;
    id pipeline =
        mglCreateAuxRenderPipelineFromAsset(
            "scaled_blit", "mgl_scaled_blit_vs", "mgl_scaled_blit_fs",
            MGL_RENDER_AUX_RENDER_SCALED_BLIT, variant,
            pixelFormat, MGLPixelFormatInvalid, MGLPixelFormatInvalid,
            MGLColorWriteMaskAll, 1u, &error);
    if (!pipeline) {
        NSLog(@"MGL ERROR: scaled blit asset pipeline create failed pixelFormat=%lu error=%@",
              (unsigned long)pixelFormat, error);
        if (ctx) mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }
    NSLog(@"MGL INFO: created scaled blit pipeline pixelFormat=%lu (Metal-cpp asset)",
          (unsigned long)pixelFormat);
    return pipeline;
}

/* Compute-based Y-flip blit pipeline.  Used by
 * updateGLSampledRenderTargetCopyForTexture to batch all dirty mip levels of
 * a sampled render-target copy into a single MTLComputeCommandEncoder, instead
 * of creating one MTLRenderCommandEncoder per mip level.  This eliminates the
 * per-mip render-encoder creation overhead that dominated the CPU-bound frame
 * (42 render encoders/frame, ~60ms CPU).
 *
 * The kernel samples the source texture at an explicit level (so the full
 * mipmap source can be bound once) and writes to the destination at an
 * explicit level (so the full mipmap destination can be bound once).  Y-flip
 * is baked into the UV calculation: destination Metal row 0 (top) receives
 * the source's bottom row, restoring GL lower-left sampling semantics. */
- (id)scaledBlitComputePipelineForPixelFormat:(uint32_t)pixelFormat
{
    if (pixelFormat == MGLPixelFormatInvalid || pixelFormat == 0) {
        pixelFormat = MGLPixelFormatBGRA8Unorm;
    }

    id cached =
        mglLookupAuxComputePipeline(
            MGL_RENDER_AUX_COMPUTE_SCALED_BLIT,
            (uint64_t)pixelFormat);
    if (cached) return cached;

    NSError *error = nil;
    id pipeline =
        mglCreateAuxComputePipelineFromAsset(
            "scaled_blit_cs", "mgl_scaled_blit_cs",
            MGL_RENDER_AUX_COMPUTE_SCALED_BLIT,
            (uint64_t)pixelFormat, &error);
    if (!pipeline) {
        NSLog(@"MGL ERROR: scaled blit asset compute pipeline create failed pixelFormat=%lu error=%@",
              (unsigned long)pixelFormat, error);
        if (ctx) mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }
    NSLog(@"MGL INFO: created scaled blit compute pipeline pixelFormat=%lu (Metal-cpp asset)",
          (unsigned long)pixelFormat);
    return pipeline;
}

- (id)scaledDepthBlitPipelineForPixelFormat:(uint32_t)pixelFormat
{
    if (pixelFormat == MGLPixelFormatInvalid || pixelFormat == 0) {
        return nil;
    }

    uint32_t stencilFormat =
        mglMetalPixelFormatIsPackedDepthStencil(pixelFormat)
            ? pixelFormat : MGLPixelFormatInvalid;
    uint64_t variant = ((uint64_t)pixelFormat << 1) |
                       (stencilFormat != MGLPixelFormatInvalid ? 1u : 0u);
    id cached =
        mglLookupAuxRenderPipeline(
            MGL_RENDER_AUX_RENDER_SCALED_DEPTH_BLIT, variant,
            MGLPixelFormatInvalid, pixelFormat, stencilFormat,
            MGLColorWriteMaskNone, 1u);
    if (cached) return cached;

    NSError *error = nil;
    id pipeline =
        mglCreateAuxRenderPipelineFromAsset(
            "scaled_depth_blit", "mgl_scaled_depth_blit_vs",
            "mgl_scaled_depth_blit_fs",
            MGL_RENDER_AUX_RENDER_SCALED_DEPTH_BLIT, variant,
            MGLPixelFormatInvalid, pixelFormat, stencilFormat,
            MGLColorWriteMaskNone, 1u, &error);
    if (!pipeline) {
        NSLog(@"MGL ERROR: scaled depth asset pipeline create failed depthPixelFormat=%lu error=%@",
              (unsigned long)pixelFormat, error);
        if (ctx) mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }
    NSLog(@"MGL INFO: created scaled depth blit pipeline depthPixelFormat=%lu (Metal-cpp asset)",
          (unsigned long)pixelFormat);
    return pipeline;
}

- (id)msaaIntegerResolvePipelineForSigned:(BOOL)signedInteger
{
    const char *entryName = signedInteger
        ? "mgl_msaa_resolve_int" : "mgl_msaa_resolve_uint";
    id cached =
        mglLookupAuxComputePipeline(
            MGL_RENDER_AUX_COMPUTE_MSAA_INTEGER_RESOLVE,
            signedInteger ? 1u : 0u);
    if (cached) return cached;

    NSError *error = nil;
    id pipeline =
        mglCreateAuxComputePipelineFromAsset(
            "msaa_integer_resolve", entryName,
            MGL_RENDER_AUX_COMPUTE_MSAA_INTEGER_RESOLVE,
            signedInteger ? 1u : 0u, &error);
    if (!pipeline) {
        NSLog(@"MGL ERROR: MSAA integer resolve asset pipeline create failed signed=%d error=%@",
              signedInteger ? 1 : 0, error);
        if (ctx) mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }
    return pipeline;
}

- (BOOL)resolveIntegerMultisampleTexture:(id)sourceTexture
                               toTexture:(id)destTexture
                                srcOrigin:(MGLOriginValue)srcOrigin
                                dstOrigin:(MGLOriginValue)dstOrigin
                                     size:(MGLSizeValue)size
                                   reason:(const char *)reason
{
    if (!sourceTexture || !destTexture ||
        mglBlitTextureInfo(sourceTexture).sample_count <= 1u ||
        mglBlitTextureInfo(destTexture).sample_count > 1u ||
        mglBlitTextureInfo(sourceTexture).pixel_format != mglBlitTextureInfo(destTexture).pixel_format ||
        !mglMetalPixelFormatIsIntegerColor(mglBlitTextureInfo(sourceTexture).pixel_format) ||
        size.width == 0u || size.height == 0u) {
        return NO;
    }

    id pipeline =
        [self msaaIntegerResolvePipelineForSigned:mglMetalPixelFormatIsSignedIntegerColor(mglBlitTextureInfo(sourceTexture).pixel_format)];
    if (!pipeline) {
        return NO;
    }

    if (![self ensureWritableCommandBuffer:"blitFramebuffer.msaaIntegerResolve"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    id encoder =
        (__bridge id)mglRenderCreateComputeEncoderBorrowed(
            _commandState.currentCommandBufferOwner);
    if (!encoder) {
        NSLog(@"MGL WARN: failed to create MSAA integer resolve encoder for %s",
              reason ? reason : "unknown");
        return NO;
    }

    MGLMSAAIntegerResolveParams params;
    params.srcOrigin = (vector_uint2){(uint32_t)srcOrigin.x, (uint32_t)srcOrigin.y};
    params.dstOrigin = (vector_uint2){(uint32_t)dstOrigin.x, (uint32_t)dstOrigin.y};
    params.size = (vector_uint2){(uint32_t)size.width, (uint32_t)size.height};
    params._padding = (vector_uint2){0u, 0u};

    mglBlitSetComputePipeline(encoder, pipeline);
    mglBlitSetComputeTexture(encoder, sourceTexture, 0);
    mglBlitSetComputeTexture(encoder, destTexture, 1);
    mglBlitSetComputeBytes(encoder, &params, sizeof(params), 0);

    MGLSizeValue threads = mglBlitSize(size.width, size.height, 1u);
    NSUInteger w = MIN((NSUInteger)16u, mglRenderComputePipelineMaxTotalThreads((__bridge void *)pipeline));
    NSUInteger h = MAX((NSUInteger)1u, MIN((NSUInteger)16u, mglRenderComputePipelineMaxTotalThreads((__bridge void *)pipeline) / w));
    MGLSizeValue threadgroup = mglBlitSize(w, h, 1u);
    mglBlitDispatchThreads(encoder, threads, threadgroup);
    mglBlitEndComputeEncoder(encoder);

    return YES;
}

- (id)resolvedReadbackTextureForMultisampleTexture:(id)sourceTexture
                                                   sourceLevel:(NSUInteger)sourceLevel
                                                   sourceSlice:(NSUInteger)sourceSlice
                                               sourceDepthPlane:(NSUInteger)sourceDepthPlane
                                                        reason:(const char *)reason
{
    if (!sourceTexture || mglBlitTextureInfo(sourceTexture).sample_count <= 1u) {
        return sourceTexture;
    }

    if (sourceLevel != 0u ||
        sourceDepthPlane != 0u ||
        (mglBlitTextureInfo(sourceTexture).texture_type != MGLTextureType2DMultisample &&
         mglBlitTextureInfo(sourceTexture).texture_type != MGLTextureType2DMultisampleArray)) {
        NSLog(@"MGL WARNING: readPixels cannot resolve MSAA texture for %s level=%lu slice=%lu depth=%lu type=%lu",
              reason ? reason : "unknown",
              (unsigned long)sourceLevel,
              (unsigned long)sourceSlice,
              (unsigned long)sourceDepthPlane,
              (unsigned long)mglBlitTextureInfo(sourceTexture).texture_type);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }

    MGLRenderTextureDescriptorState desc = {0};
    desc.texture_type = MGLTextureType2D;
    desc.pixel_format = mglBlitTextureInfo(sourceTexture).pixel_format;
    desc.width = mglBlitTextureInfo(sourceTexture).width;
    desc.height = mglBlitTextureInfo(sourceTexture).height;
    desc.depth = 1;
    desc.mipmap_level_count = 1;
    desc.sample_count = 1;
    desc.array_length = 1;
    desc.usage = MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
    desc.storage_mode = MGLStorageModePrivate;

    id resolvedTexture = mglBlitCreateTexture(_device, &desc);
    if (!resolvedTexture) {
        NSLog(@"MGL WARNING: readPixels failed to allocate MSAA resolve texture for %s fmt=%lu size=%lux%lu samples=%lu",
              reason ? reason : "unknown",
              (unsigned long)mglBlitTextureInfo(sourceTexture).pixel_format,
              (unsigned long)mglBlitTextureInfo(sourceTexture).width,
              (unsigned long)mglBlitTextureInfo(sourceTexture).height,
              (unsigned long)mglBlitTextureInfo(sourceTexture).sample_count);
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return nil;
    }

    if (![self ensureWritableCommandBuffer:"readPixels.msaaResolve"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }

    BOOL resolvesDepth =
        mglMetalPixelFormatIsDepthOrStencil(mglBlitTextureInfo(sourceTexture).pixel_format);
    if (mglRenderEncodeMultisampleResolveForCommandBufferOwner(
            _commandState.currentCommandBufferOwner,
            resolvesDepth
                ? MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH
                : MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
            (__bridge void *)sourceTexture, sourceLevel, sourceSlice,
            sourceDepthPlane, (__bridge void *)resolvedTexture,
            0, 0, 0,
            resolvesDepth
                ? (uint32_t)MGLMultisampleDepthResolveFilterSample0
                : 0u) == 0) {
        return resolvedTexture;
    }
    NSLog(@"MGL WARNING: readPixels failed to encode MSAA resolve for %s",
          reason ? reason : "unknown");
    mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
    return nil;
}

- (id)depthFloatTextureForDepthStencilReadback:(id)sourceTexture
                                                    reason:(const char *)reason
{
    if (!sourceTexture ||
        mglBlitTextureInfo(sourceTexture).sample_count > 1u ||
        !mglMetalPixelFormatIsPackedDepthStencil(mglBlitTextureInfo(sourceTexture).pixel_format)) {
        return sourceTexture;
    }

    MGLRenderTextureDescriptorState desc = {0};
    desc.texture_type = MGLTextureType2D;
    desc.pixel_format = MGLPixelFormatDepth32Float;
    desc.width = mglBlitTextureInfo(sourceTexture).width;
    desc.height = mglBlitTextureInfo(sourceTexture).height;
    desc.depth = 1;
    desc.mipmap_level_count = 1;
    desc.sample_count = 1;
    desc.array_length = 1;
    desc.usage = MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
    desc.storage_mode = MGLStorageModePrivate;
    id depthTexture = mglBlitCreateTexture(_device, &desc);
    if (!depthTexture) {
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return nil;
    }

    id pipeline =
        [self scaledDepthBlitPipelineForPixelFormat:MGLPixelFormatDepth32Float];
    id sampler = [self scaledBlitSamplerForFilter:GL_NEAREST];
    if (!pipeline || !sampler) {
        NSLog(@"MGL WARNING: readPixels DS depth extract unavailable for %s pipeline=%p sampler=%p",
              reason ? reason : "unknown",
              pipeline,
              sampler);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }

    if (![self ensureWritableCommandBuffer:"readPixels.depthStencilExtract"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }

    MGLScaledBlitParams params;
    params.uvRect = (vector_float4){0.0f, 0.0f, 1.0f, 1.0f};
    params.forceOpaqueAlpha = 0.0f;
    params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

    MGLRenderPassState passState =
        mglBlitDefaultRenderPassState();
    passState.depth.attachment = mglBlitRenderPassAttachment(
        depthTexture, 0u, 0u, 0u, MGLLoadActionDontCare,
        MGLStoreActionStore);

    id encoder =
        mglBlitCreateRenderEncoder(&_commandState, &passState);
    if (!encoder) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }

    mglBlitSetRenderPipeline(encoder, pipeline);
    mglBlitSetDepthStencil(encoder, [self clearRectDepthState]);
    mglBlitSetRenderBytes(encoder, &params, sizeof(params),
                          MGL_RENDER_BINDING_STAGE_VERTEX, 0);
    mglBlitSetRenderBytes(encoder, &params, sizeof(params),
                          MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
    mglBlitSetRenderTexture(encoder, sourceTexture,
                            MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
    mglBlitSetRenderSampler(encoder, sampler,
                            MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
    mglBlitSetRenderViewport(encoder, (MGLViewportValue){
        .origin_x = 0.0,
        .origin_y = 0.0,
        .width = (double)mglBlitTextureInfo(sourceTexture).width,
        .height = (double)mglBlitTextureInfo(sourceTexture).height,
        .znear = 0.0,
        .zfar = 1.0
    });
    mglBlitDrawPrimitives(encoder, MGLPrimitiveTypeTriangleStrip, 0, 4);
    mglBlitEndRenderEncoder(encoder);

    return depthTexture;
}

- (BOOL)textureCanUseGLSampledRenderTargetCopy:(Texture *)tex
                                        source:(id)source
{
    if (!tex || !source || !tex->is_render_target) {
        return NO;
    }

    if (tex->target != GL_TEXTURE_2D ||
        tex->width == 0u ||
        tex->height == 0u ||
        mglBlitTextureInfo(source).texture_type != MGLTextureType2D ||
        mglBlitTextureInfo(source).mipmap_level_count == 0u ||
        mglBlitTextureInfo(source).width == 0u ||
        mglBlitTextureInfo(source).height == 0u ||
        mglMetalPixelFormatIsDepthOrStencil(mglBlitTextureInfo(source).pixel_format) ||
        mglTextureDataKindForPixelFormat(mglBlitTextureInfo(source).pixel_format) != MGLTextureDataKindFloat) {
        return NO;
    }

    /* Apply sampled-copy protection to all 2D float render targets
     * regardless of size.  The previous size-based gating was a
     * Minecraft-specific heuristic that broke on larger render targets. */
    if (!mglTextureCanUseGLSampledRenderTargetCopy(tex)) {
        return NO;
    }

    return YES;
}

- (void)releaseGLSampledRenderTargetCopyForTexture:(Texture *)tex
{
    if (!tex) {
        return;
    }

    if (tex->mtl_gl_sampled_data) {
        mglSafeReleaseMetalObj((void **)&tex->mtl_gl_sampled_data);
    }

    tex->mtl_gl_sampled_width = 0u;
    tex->mtl_gl_sampled_height = 0u;
    tex->mtl_gl_sampled_format = 0u;
    tex->mtl_gl_sampled_levels = 0u;
    tex->mtl_gl_sampled_write_version = 0u;
    tex->mtl_gl_sampled_dirty_mip_mask = 0u;
}


- (BOOL)lazyRefreshGLSampledRenderTargetCopyForTexture:(Texture *)tex
                                                 stage:(const char *)stage
                                               program:(GLuint)programName
                                               binding:(GLuint)binding
                                                  unit:(GLuint)unit
{
    if (!tex || !tex->mtl_data) {
        return NO;
    }
    if (!mglTextureCanUseGLSampledRenderTargetCopy(tex)) {
        return NO;
    }
    if (tex->mtl_render_target_write_version == 0u) {
        return NO;
    }

    if (tex->mtl_gl_sampled_data &&
        tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version) {
        return YES;
    }
    /* Stale (or no copy yet): safe to refresh only if the texture is not an
     * attachment of the active render pass.  Reading a render-pass
     * attachment mid-pass is a Metal read-after-write hazard. */
    if (mglTextureIsAttachmentOfFramebuffer(_commandState.renderPassFramebuffer, tex)) {
        if (mglTraceLogIsEnabled()) {
            mglTraceLog("RT_SAMPLE_COPY_LAZY_SKIP stage=%s program=%u binding=%u unit=%u tex=%u writeVer=%u rtVer=%u reason=current-pass-attachment",
                        stage ? stage : "",
                        (unsigned)programName,
                        (unsigned)binding,
                        (unsigned)unit,
                        (unsigned)tex->name,
                        (unsigned)tex->mtl_gl_sampled_write_version,
                        (unsigned)tex->mtl_render_target_write_version);
        }
        return NO;
    }
    id source = (__bridge id)(tex->mtl_data);
    BOOL ok = [self updateGLSampledRenderTargetCopyForTexture:tex
                                                       source:source
                                                       reason:"lazy_sample_copy_refresh"];
    if (mglTraceLogIsEnabled()) {
        mglTraceLog("RT_SAMPLE_COPY_LAZY_REFRESH stage=%s program=%u binding=%u unit=%u tex=%u ok=%d writeVer=%u rtVer=%u",
                    stage ? stage : "",
                    (unsigned)programName,
                    (unsigned)binding,
                    (unsigned)unit,
                    (unsigned)tex->name,
                    ok ? 1 : 0,
                    (unsigned)tex->mtl_gl_sampled_write_version,
                    (unsigned)tex->mtl_render_target_write_version);
    }
    return ok && tex->mtl_gl_sampled_data &&
                tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version;
}

- (id)freshGLSampledRenderTargetCopyForSampling:(Texture *)tex
                                                     source:(id)source
                                                      stage:(const char *)stage
                                                    program:(GLuint)programName
                                                    binding:(GLuint)binding
                                                       unit:(GLuint)unit
                                               expectedType:(uint32_t)expectedType
                                               expectedKind:(MGLTextureDataKind)expectedKind
{
    if (!tex || !source || !mglTextureCanUseGLSampledRenderTargetCopy(tex)) {
        return nil;
    }
    if (tex->mtl_render_target_write_version == 0u) {
        return nil;
    }

    id sampledCopy = tex->mtl_gl_sampled_data
        ? (__bridge id)(tex->mtl_gl_sampled_data)
        : nil;
    if (sampledCopy &&
        tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version &&
        (expectedType == 0 || mglBlitTextureInfo(sampledCopy).texture_type == expectedType) &&
        mglTexturePixelFormatCompatibleWithExpectedDataKind(mglBlitTextureInfo(sampledCopy).pixel_format, expectedKind)) {
        return sampledCopy;
    }

    BOOL isFbAttachment = mglTextureIsAttachmentOfFramebuffer(_commandState.renderPassFramebuffer, tex);

    if ([self currentRenderPassUsesTexture:source] && !isFbAttachment) {
        /* The texture is used by the current render pass in a non-attachment
         * role (e.g. bound to another sampler).  We cannot safely end and
         * restore the pass in this case because the texture might be written
         * by the pass itself. */
        if (mglTraceLogIsEnabled()) {
            mglTraceLog("RT_SAMPLE_COPY_REPAIR_SKIP stage=%s program=%u binding=%u unit=%u tex=%u label=\"%s\" reason=current-pass-uses-texture writeVer=%u rtVer=%u",
                        stage ? stage : "",
                        (unsigned)programName,
                        (unsigned)binding,
                        (unsigned)unit,
                        (unsigned)tex->name,
                        mglTraceTextureLabel(tex),
                        (unsigned)tex->mtl_gl_sampled_write_version,
                        (unsigned)tex->mtl_render_target_write_version);
        }
        return nil;
    }


    if (isFbAttachment) {
        if (mglTraceLogIsEnabled()) {
            mglTraceLog("RT_SAMPLE_COPY_REPAIR_ATTEMPT stage=%s program=%u binding=%u unit=%u tex=%u label=\"%s\" reason=fb-attachment writeVer=%u rtVer=%u",
                        stage ? stage : "",
                        (unsigned)programName,
                        (unsigned)binding,
                        (unsigned)unit,
                        (unsigned)tex->name,
                        mglTraceTextureLabel(tex),
                        (unsigned)tex->mtl_gl_sampled_write_version,
                        (unsigned)tex->mtl_render_target_write_version);
        }
    }

    BOOL hadRenderEncoder =
        mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) == 1;
    if (hadRenderEncoder) {
        [self endRenderEncodingLocked];
    }

    sampledCopy = tex->mtl_gl_sampled_data
        ? (__bridge id)(tex->mtl_gl_sampled_data)
        : nil;
    if (!(sampledCopy &&
          tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version &&
          (expectedType == 0 || mglBlitTextureInfo(sampledCopy).texture_type == expectedType) &&
          mglTexturePixelFormatCompatibleWithExpectedDataKind(mglBlitTextureInfo(sampledCopy).pixel_format, expectedKind))) {
        source = tex->mtl_data ? (__bridge id)(tex->mtl_data) : nil;
        if (source) {
            [self updateGLSampledRenderTargetCopyForTexture:tex
                                                     source:source
                                                     reason:"sample_gate_miss_repair"];
        }
        sampledCopy = tex->mtl_gl_sampled_data
            ? (__bridge id)(tex->mtl_gl_sampled_data)
            : nil;
    }

    if (hadRenderEncoder &&
        mglRenderEncoderOwnerHasCurrent(
            _commandState.currentRenderEncoderOwner) != 1) {
        if (![self restoreRenderEncoderAfterTextureUploadForDraw:"sample_gate_miss_repair"]) {
            return nil;
        }
    }

    BOOL fresh =
        sampledCopy &&
        tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version &&
        (expectedType == 0 || mglBlitTextureInfo(sampledCopy).texture_type == expectedType) &&
        mglTexturePixelFormatCompatibleWithExpectedDataKind(mglBlitTextureInfo(sampledCopy).pixel_format, expectedKind);
    if (mglTraceLogIsEnabled()) {
        mglTraceLog("RT_SAMPLE_COPY_REPAIR stage=%s program=%u binding=%u unit=%u tex=%u label=\"%s\" ok=%d copy=%p writeVer=%u rtVer=%u expectedType=%lu",
                    stage ? stage : "",
                    (unsigned)programName,
                    (unsigned)binding,
                    (unsigned)unit,
                    (unsigned)tex->name,
                    mglTraceTextureLabel(tex),
                    fresh ? 1 : 0,
                    sampledCopy,
                    (unsigned)tex->mtl_gl_sampled_write_version,
                    (unsigned)tex->mtl_render_target_write_version,
                    (unsigned long)expectedType);
    }
    return fresh ? sampledCopy : nil;
}

- (BOOL)updateGLSampledRenderTargetCopyForTexture:(Texture *)tex
                                           source:(id)source
                                           reason:(const char *)reason
{
    MGL_ASSERT_GL_THREAD();
    if (![self textureCanUseGLSampledRenderTargetCopy:tex source:source]) {
        return NO;
    }

    if (tex->mtl_render_target_write_version == 0u) {
        return NO;
    }

    /* Copy the full GL mip chain of the RT (capped to mglBlitTextureInfo(source).mipmap_level_count),
     * not the transient GpuTextureView BASE/MAX_LEVEL window.  Shrinking the
     * Y-flip copy to that window left higher mips stale and re-broke the
     * ced1a99 stripe fix (mipmapped sampling fell back to the un-flipped RT).
     * Sampling windows are applied later via mglSampledTextureViewForBaseLevel. */
    NSUInteger copyLevelCount = 1u;
    if (mglBlitTextureInfo(source).mipmap_level_count > 1u) {
        GLuint highestGLLevel = tex->num_levels > 0u ? tex->num_levels - 1u : 0u;
        if (tex->mipmap_levels > 0u && highestGLLevel >= tex->mipmap_levels) {
            highestGLLevel = tex->mipmap_levels - 1u;
        }

        NSUInteger highestSourceLevel = mglBlitTextureInfo(source).mipmap_level_count - 1u;
        if ((NSUInteger)highestGLLevel > highestSourceLevel) {
            highestGLLevel = (GLuint)highestSourceLevel;
        }

        copyLevelCount = (NSUInteger)highestGLLevel + 1u;
    }

    if (tex->mtl_gl_sampled_data &&
        tex->mtl_gl_sampled_width == (GLuint)mglBlitTextureInfo(source).width &&
        tex->mtl_gl_sampled_height == (GLuint)mglBlitTextureInfo(source).height &&
        tex->mtl_gl_sampled_format == (GLuint)mglBlitTextureInfo(source).pixel_format &&
        tex->mtl_gl_sampled_levels == (GLuint)copyLevelCount &&
        tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version &&
        tex->mtl_gl_sampled_dirty_mip_mask == 0u) {
        return YES;
    }

    BOOL needsNewCopy =
        tex->mtl_gl_sampled_data == NULL ||
        tex->mtl_gl_sampled_width != (GLuint)mglBlitTextureInfo(source).width ||
        tex->mtl_gl_sampled_height != (GLuint)mglBlitTextureInfo(source).height ||
        tex->mtl_gl_sampled_format != (GLuint)mglBlitTextureInfo(source).pixel_format ||
        tex->mtl_gl_sampled_levels != (GLuint)copyLevelCount;
    if (needsNewCopy) {
        [self releaseGLSampledRenderTargetCopyForTexture:tex];


        MGLRenderTextureDescriptorState desc = {0};
        desc.texture_type = MGLTextureType2D;
        desc.pixel_format = mglBlitTextureInfo(source).pixel_format;
        desc.width = mglBlitTextureInfo(source).width;
        desc.height = mglBlitTextureInfo(source).height;
        desc.depth = 1;
        desc.mipmap_level_count = copyLevelCount;
        desc.sample_count = 1;
        desc.array_length = 1;
        desc.usage = MGLTextureUsageShaderRead | MGLTextureUsageRenderTarget | MGLTextureUsageShaderWrite;
        desc.storage_mode = MGLStorageModePrivate;

        id copy = mglBlitCreateTexture(_device, &desc);
        if (!copy) {
            static uint64_t s_copyCreateFailCount = 0;
            uint64_t hit = ++s_copyCreateFailCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL RT-SAMPLE-COPY create failed tex=%u size=%lux%lu fmt=%lu reason=%s hit=%llu",
                      (unsigned)tex->name,
                      (unsigned long)mglBlitTextureInfo(source).width,
                      (unsigned long)mglBlitTextureInfo(source).height,
                      (unsigned long)mglBlitTextureInfo(source).pixel_format,
                      reason ? reason : "(null)",
                      (unsigned long long)hit);
            }
            return NO;
        }

        tex->mtl_gl_sampled_data = (void *)CFBridgingRetain(copy);
        tex->mtl_gl_sampled_width = (GLuint)mglBlitTextureInfo(source).width;
        tex->mtl_gl_sampled_height = (GLuint)mglBlitTextureInfo(source).height;
        tex->mtl_gl_sampled_format = (GLuint)mglBlitTextureInfo(source).pixel_format;
        tex->mtl_gl_sampled_levels = (GLuint)copyLevelCount;
    }

    id destination = (__bridge id)(tex->mtl_gl_sampled_data);
    id sampler = [self scaledBlitSamplerForFilter:GL_NEAREST];
    if (!destination || !sampler) {
        static uint64_t s_copySetupFailCount = 0;
        uint64_t hit = ++s_copySetupFailCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL RT-SAMPLE-COPY setup failed tex=%u dst=%p sampler=%p reason=%s hit=%llu",
                  (unsigned)tex->name,
                  destination,
                  sampler,
                  reason ? reason : "(null)",
                  (unsigned long long)hit);
        }
        return NO;
    }


    NSUInteger mipLevels = MAX(copyLevelCount, 1u);
    if (mipLevels > mglBlitTextureInfo(destination).mipmap_level_count) {
        mipLevels = mglBlitTextureInfo(destination).mipmap_level_count;
    }
    if (mipLevels > (NSUInteger)mglBlitTextureInfo(source).mipmap_level_count) {
        mipLevels = (NSUInteger)mglBlitTextureInfo(source).mipmap_level_count;
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

    if (![self ensureWritableCommandBuffer:reason ? reason : "rt_sample_copy"]) {
        return NO;
    }

    // Sampled render-target copy: flip rows once so that Metal row 0 (top, which
    // is what Metal's texture::sample sees at v=0) holds GL row 0 (bottom).
    // See the longer comment block in the fallback render path below for the
    // full Metal-vs-GL Y-origin rationale.
    BOOL yFlipCopy = YES;

    uint32_t copiedMask = 0u;

    /* Prefer compute path: single MTLComputeCommandEncoder dispatches all dirty
     * mip levels, avoiding per-mip render-encoder creation overhead. */
    BOOL useComputePath = (mglBlitTextureInfo(destination).usage & MGLTextureUsageShaderWrite) != 0;
    id computePipeline = nil;
    if (useComputePath) {
        computePipeline = [self scaledBlitComputePipelineForPixelFormat:mglBlitTextureInfo(destination).pixel_format];
        if (!computePipeline) {
            useComputePath = NO;
        }
    }

    if (useComputePath) {
        id computeEncoder =
            (__bridge id)mglRenderCreateComputeEncoderBorrowed(
                _commandState.currentCommandBufferOwner);
        if (!computeEncoder) {
            static uint64_t s_computeEncoderFailCount = 0;
            uint64_t hit = ++s_computeEncoderFailCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL RT-SAMPLE-COPY compute encoder failed tex=%u reason=%s hit=%llu",
                      (unsigned)tex->name,
                      reason ? reason : "(null)",
                      (unsigned long long)hit);
            }
            useComputePath = NO;
        } else {
            typedef struct {
                vector_uint2 dstSize;
                uint32_t srcLevel;
                uint32_t dstLevel;
            } MGLScaledBlitComputeParams;

            mglBlitSetComputePipeline(computeEncoder, computePipeline);
            mglBlitSetComputeTexture(computeEncoder, source, 0);
            mglBlitSetComputeTexture(computeEncoder, destination, 1);

            NSUInteger tgW = MIN((NSUInteger)16u, mglRenderComputePipelineMaxTotalThreads((__bridge void *)computePipeline));
            NSUInteger tgH = MAX((NSUInteger)1u,
                                 MIN((NSUInteger)16u,
                                     mglRenderComputePipelineMaxTotalThreads((__bridge void *)computePipeline) / tgW));
            MGLSizeValue threadgroup = mglBlitSize(tgW, tgH, 1u);

            for (NSUInteger lvl = 0u; lvl < mipLevels; lvl++) {
                if ((copyMask & ((uint32_t)1u << lvl)) == 0u) {
                    continue;
                }

                NSUInteger mipW = MAX(1u, mglBlitTextureInfo(source).width >> lvl);
                NSUInteger mipH = MAX(1u, mglBlitTextureInfo(source).height >> lvl);

                MGLScaledBlitComputeParams params;
                params.dstSize = (vector_uint2){(uint32_t)mipW, (uint32_t)mipH};
                params.srcLevel = (uint32_t)lvl;
                params.dstLevel = (uint32_t)lvl;

                mglBlitSetComputeBytes(computeEncoder, &params, sizeof(params), 0);

                MGLSizeValue threads = mglBlitSize(mipW, mipH, 1u);
                mglBlitDispatchThreads(computeEncoder, threads, threadgroup);

                copiedMask |= (uint32_t)1u << lvl;
            }
            mglBlitEndComputeEncoder(computeEncoder);
        }
    }

    if (!useComputePath) {

        id pipeline = [self scaledBlitPipelineForPixelFormat:mglBlitTextureInfo(destination).pixel_format];
        if (!pipeline) {
            static uint64_t s_copySetupFailCount = 0;
            uint64_t hit = ++s_copySetupFailCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL RT-SAMPLE-COPY render pipeline setup failed tex=%u reason=%s hit=%llu",
                      (unsigned)tex->name,
                      reason ? reason : "(null)",
                      (unsigned long long)hit);
            }
            return NO;
        }

        MGLScaledBlitParams params;
        params.uvRect = yFlipCopy
            ? (vector_float4){0.0f, 1.0f, 1.0f, 0.0f}
            : (vector_float4){0.0f, 0.0f, 1.0f, 1.0f};
        params.forceOpaqueAlpha = 0.0f;
        params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

        for (NSUInteger lvl = 0u; lvl < mipLevels; lvl++) {
            if ((copyMask & ((uint32_t)1u << lvl)) == 0u) {
                continue;
            }
            @autoreleasepool {
                id srcLvl = source;
                id dstLvl = destination;
                if (mipLevels > 1u) {
                    srcLvl = mglBlitCreateTextureView(
                        source, mglBlitTextureInfo(source).pixel_format, MGLTextureType2D,
                        NSMakeRange(lvl, 1u), NSMakeRange(0, 1u));
                    dstLvl = mglBlitCreateTextureView(
                        destination, mglBlitTextureInfo(destination).pixel_format,
                        MGLTextureType2D, NSMakeRange(lvl, 1u),
                        NSMakeRange(0, 1u));
                    if (!srcLvl || !dstLvl) {
                        static uint64_t s_levelViewFailCount = 0;
                        uint64_t hit = ++s_levelViewFailCount;
                        if (hit <= 32ull || (hit % 512ull) == 0ull) {
                            NSLog(@"MGL RT-SAMPLE-COPY level view failed tex=%u lvl=%lu hit=%llu",
                                  (unsigned)tex->name,
                                  (unsigned long)lvl,
                                  (unsigned long long)hit);
                        }
                        continue;
                    }
                }

                MGLRenderPassState copyState =
                    mglBlitDefaultRenderPassState();
                copyState.color[0].attachment =
                    mglBlitRenderPassAttachment(
                        dstLvl, 0u, 0u, 0u, MGLLoadActionDontCare,
                        MGLStoreActionStore);
                copyState.render_target_width = mglBlitTextureInfo(dstLvl).width;
                copyState.render_target_height = mglBlitTextureInfo(dstLvl).height;

                id copyEncoder =
                    mglBlitCreateRenderEncoder(&_commandState, &copyState);
                if (!copyEncoder) {
                    static uint64_t s_copyEncoderFailCount = 0;
                    uint64_t hit = ++s_copyEncoderFailCount;
                    if (hit <= 32ull || (hit % 512ull) == 0ull) {
                        NSLog(@"MGL RT-SAMPLE-COPY encoder failed tex=%u lvl=%lu reason=%s hit=%llu",
                              (unsigned)tex->name,
                              (unsigned long)lvl,
                              reason ? reason : "(null)",
                              (unsigned long long)hit);
                    }
                    continue;
                }

                mglBlitSetRenderPipeline(copyEncoder, pipeline);
                mglBlitSetRenderBytes(copyEncoder, &params, sizeof(params),
                                      MGL_RENDER_BINDING_STAGE_VERTEX, 0);
                mglBlitSetRenderBytes(copyEncoder, &params, sizeof(params),
                                      MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                mglBlitSetRenderTexture(copyEncoder, srcLvl,
                                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                mglBlitSetRenderSampler(copyEncoder, sampler,
                                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                mglBlitSetRenderViewport(copyEncoder, (MGLViewportValue){
                    .origin_x = 0.0,
                    .origin_y = 0.0,
                    .width = (double)mglBlitTextureInfo(dstLvl).width,
                    .height = (double)mglBlitTextureInfo(dstLvl).height,
                    .znear = 0.0,
                    .zfar = 1.0
                });
                mglBlitSetRenderScissor(copyEncoder, (MGLScissorRectValue){
                    .x = 0,
                    .y = 0,
                    .width = mglBlitTextureInfo(dstLvl).width,
                    .height = mglBlitTextureInfo(dstLvl).height
                });
                mglBlitDrawPrimitives(copyEncoder, MGLPrimitiveTypeTriangleStrip, 0, 4);
                mglBlitEndRenderEncoder(copyEncoder);
                copiedMask |= (uint32_t)1u << lvl;
            }
        }
    }

    tex->mtl_gl_sampled_dirty_mip_mask &= ~copiedMask;
    if ((tex->mtl_gl_sampled_dirty_mip_mask & mipMask) == 0u) {
        tex->mtl_gl_sampled_write_version = tex->mtl_render_target_write_version;
    }

    if (mglTraceLogIsEnabled()) {
        mglTraceLog("RT_SAMPLE_COPY_UPDATED tex=%u label=\"%s\" lightmap=%d yFlip=%d src=%p dst=%p size=%lux%lu fmt=%lu srcLevels=%lu dstLevels=%lu glLevels=%u mips=%u base=%u max=%u writeVersion=%u reason=%s compute=%d",
                    (unsigned)tex->name,
                    mglTraceTextureLabel(tex),
                    0,
                    yFlipCopy ? 1 : 0,
                    source,
                    destination,
                    (unsigned long)mglBlitTextureInfo(destination).width,
                    (unsigned long)mglBlitTextureInfo(destination).height,
                    (unsigned long)mglBlitTextureInfo(destination).pixel_format,
                    (unsigned long)mglBlitTextureInfo(source).mipmap_level_count,
                    (unsigned long)mglBlitTextureInfo(destination).mipmap_level_count,
                    (unsigned)tex->num_levels,
                    (unsigned)tex->mipmap_levels,
                    (unsigned)tex->params.base_level,
                    (unsigned)tex->params.max_level,
                    (unsigned)tex->mtl_gl_sampled_write_version,
                    reason ? reason : "(null)",
                    useComputePath ? 1 : 0);
    }

    return YES;
}

- (id)clearRectPipelineForColorFormat:(uint32_t)colorFormat
                                                  depthFormat:(uint32_t)depthFormat
                                                  writesColor:(BOOL)writesColor
                                                  writesDepth:(BOOL)writesDepth
{
    if (!writesColor && !writesDepth) {
        return nil;
    }
    if (writesColor && colorFormat == MGLPixelFormatInvalid) {
        return nil;
    }
    if (writesDepth && depthFormat == MGLPixelFormatInvalid) {
        return nil;
    }

    uint64_t variant = (uint64_t)(uint32_t)colorFormat |
                       ((uint64_t)(uint32_t)depthFormat << 16) |
                       ((uint64_t)(writesColor ? 1u : 0u) << 32) |
                       ((uint64_t)(writesDepth ? 1u : 0u) << 33);
    id cached =
        mglLookupAuxRenderPipeline(
            MGL_RENDER_AUX_RENDER_CLEAR_RECT, variant,
            colorFormat, depthFormat, MGLPixelFormatInvalid,
            writesColor ? MGLColorWriteMaskAll : MGLColorWriteMaskNone,
            1u);
    if (cached) return cached;

    NSError *error = nil;
    id pipeline =
        mglCreateAuxRenderPipelineFromAsset(
            "clear_rect", "mgl_clear_rect_vs",
            writesColor ? "mgl_clear_rect_fs" : NULL,
            MGL_RENDER_AUX_RENDER_CLEAR_RECT, variant,
            colorFormat, depthFormat, MGLPixelFormatInvalid,
            writesColor ? MGLColorWriteMaskAll : MGLColorWriteMaskNone,
            1u, &error);
    if (!pipeline) {
        NSLog(@"MGL ERROR: scissored clear asset pipeline create failed color=%lu depth=%lu writesColor=%d writesDepth=%d error=%@",
              (unsigned long)colorFormat, (unsigned long)depthFormat,
              writesColor ? 1 : 0, writesDepth ? 1 : 0, error);
        if (ctx) mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }
    NSLog(@"MGL INFO: created scissored clear pipeline (Metal-cpp asset)");
    return pipeline;
}

- (id)clearRectDepthState
{
    id cached = (__bridge id)
        mglRendererBackendGetBlitCachedObject(
            _backend, MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE);
    if (cached) {
        return cached;
    }

    MGLRenderDepthStencilDescriptorState desc = {0};
    desc.depth_compare_function = MGLCompareFunctionAlways;
    desc.depth_write_enabled = 1u;
    id depthState =
        mglBlitCreateDepthStencilState(_device, &desc);
    if (!depthState ||
        mglRendererBackendSetBlitCachedObject(
            _backend, MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE,
            (__bridge void *)depthState) != 0) {
        return nil;
    }
    return (__bridge id)
        mglRendererBackendGetBlitCachedObject(
            _backend, MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE);
}

/* Depth/stencil blit path for mtlBlitFramebuffer.
 * Handles GL_DEPTH_BUFFER_BIT / GL_STENCIL_BUFFER_BIT via Metal render-pass
 * resolve (MSAA), MTLBlitCommandEncoder (same-size), or scaled depth shader.
 * Returns the updated mask with completed depth/stencil bits cleared. */
- (GLbitfield)blitFramebufferDepthStencil:(GLMContext)glm_ctx
                                    srcX0:(GLint)srcX0 srcY0:(GLint)srcY0 srcX1:(GLint)srcX1 srcY1:(GLint)srcY1
                                    dstX0:(GLint)dstX0 dstY0:(GLint)dstY0 dstX1:(GLint)dstX1 dstY1:(GLint)dstY1
                                     mask:(GLbitfield)mask filter:(GLenum)filter
{
    MGL_ASSERT_GL_THREAD();
    GLbitfield depthStencilMask = mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    if (depthStencilMask != 0u && glm_ctx->state.readbuffer && glm_ctx->state.framebuffer) {
        Framebuffer *depthReadFBO = glm_ctx->state.readbuffer;
        Framebuffer *depthDrawFBO = glm_ctx->state.framebuffer;
        FBOAttachment *depthReadAttachment =
            (depthStencilMask & GL_DEPTH_BUFFER_BIT) ? &depthReadFBO->depth : &depthReadFBO->stencil;
        FBOAttachment *depthDrawAttachment =
            (depthStencilMask & GL_DEPTH_BUFFER_BIT) ? &depthDrawFBO->depth : &depthDrawFBO->stencil;
        Texture *depthReadObject = [self framebufferAttachmentTexture:depthReadAttachment];
        Texture *depthDrawObject = [self framebufferAttachmentTexture:depthDrawAttachment];

        if (depthReadObject && depthDrawObject &&
            [self bindMTLTexture:depthReadObject] &&
            [self bindMTLTexture:depthDrawObject]) {
            id depthReadTexture = (__bridge id)depthReadObject->mtl_data;
            id depthDrawTexture = (__bridge id)depthDrawObject->mtl_data;
            MGLMetalAttachmentSubresource depthReadSubresource =
                mglMetalAttachmentSubresourceForAttachment(depthReadAttachment);
            MGLMetalAttachmentSubresource depthDrawSubresource =
                mglMetalAttachmentSubresourceForAttachment(depthDrawAttachment);
            GLint srcWidth = srcX1 - srcX0;
            GLint srcHeight = srcY1 - srcY0;
            GLint dstWidth = dstX1 - dstX0;
            GLint dstHeight = dstY1 - dstY0;

            if (depthReadTexture && depthDrawTexture &&
                srcWidth > 0 && srcHeight > 0 &&
                srcWidth == dstWidth && srcHeight == dstHeight &&
                mglBlitTextureInfo(depthReadTexture).pixel_format == mglBlitTextureInfo(depthDrawTexture).pixel_format &&
                mglBlitTextureInfo(depthReadTexture).sample_count > 1u && mglBlitTextureInfo(depthDrawTexture).sample_count <= 1u &&
                depthReadSubresource.level == 0u && depthDrawSubresource.level == 0u &&
                depthReadSubresource.depthPlane == 0u && depthDrawSubresource.depthPlane == 0u &&
                srcX0 == 0 && srcY0 == 0 && dstX0 == 0 && dstY0 == 0 &&
                (NSUInteger)srcWidth <= mglBlitTextureInfo(depthReadTexture).width &&
                (NSUInteger)srcHeight <= mglBlitTextureInfo(depthReadTexture).height &&
                (NSUInteger)dstWidth <= mglBlitTextureInfo(depthDrawTexture).width &&
                (NSUInteger)dstHeight <= mglBlitTextureInfo(depthDrawTexture).height) {
                [self endRenderEncoding];
                if ([self ensureWritableCommandBuffer:"mtlBlitFramebuffer.depthMsaaResolve"]) {
                    if (depthStencilMask & GL_DEPTH_BUFFER_BIT) {
                        [self mglApplyPendingFBODepthClearForReadback:depthReadFBO
                                                           attachment:depthReadAttachment
                                                           textureObj:depthReadObject
                                                           mtlTexture:depthReadTexture];
                    }

                    BOOL resolvedAny = NO;
                    if (depthStencilMask & GL_DEPTH_BUFFER_BIT) {
                        resolvedAny = YES;
                    }
                    if ((depthStencilMask & GL_STENCIL_BUFFER_BIT) &&
                        mglMetalPixelFormatIsPackedDepthStencil(mglBlitTextureInfo(depthReadTexture).pixel_format)) {
                        resolvedAny = YES;
                    }

                    if (resolvedAny) {
                        MGLRenderPassState resolveState =
                            mglBlitDefaultRenderPassState();
                        if (depthStencilMask & GL_DEPTH_BUFFER_BIT) {
                            resolveState.depth.attachment =
                                mglBlitRenderPassAttachment(
                                    depthReadTexture, 0u,
                                    depthReadSubresource.slice, 0u,
                                    MGLLoadActionLoad,
                                    MGLStoreActionMultisampleResolve);
                            resolveState.depth.attachment.resolve_texture =
                                (__bridge void *)depthDrawTexture;
                            resolveState.depth.attachment.resolve_slice =
                                depthDrawSubresource.slice;
                            resolveState.depth.resolve_filter =
                                (uint32_t)MGLMultisampleDepthResolveFilterSample0;
                        }
                        if ((depthStencilMask & GL_STENCIL_BUFFER_BIT) &&
                            mglMetalPixelFormatIsPackedDepthStencil(
                                mglBlitTextureInfo(depthReadTexture).pixel_format)) {
                            resolveState.stencil.attachment =
                                mglBlitRenderPassAttachment(
                                    depthReadTexture, 0u,
                                    depthReadSubresource.slice, 0u,
                                    MGLLoadActionLoad,
                                    MGLStoreActionMultisampleResolve);
                            resolveState.stencil.attachment.resolve_texture =
                                (__bridge void *)depthDrawTexture;
                            resolveState.stencil.attachment.resolve_slice =
                                depthDrawSubresource.slice;
                            resolveState.stencil.resolve_filter =
                                (uint32_t)MGLMultisampleStencilResolveFilterSample0;
                        }
                        id resolveEncoder =
                            mglBlitCreateRenderEncoder(&_commandState,
                                                       &resolveState);
                        if (resolveEncoder) {
                            mglBlitEndRenderEncoder(resolveEncoder);
                            mglMarkTextureLevelRenderTargetWritten(depthDrawObject, depthDrawAttachment->level);
                            if (depthStencilMask & GL_DEPTH_BUFFER_BIT) {
                                mask &= ~GL_DEPTH_BUFFER_BIT;
                            }
                            if ((depthStencilMask & GL_STENCIL_BUFFER_BIT) &&
                                mglMetalPixelFormatIsPackedDepthStencil(mglBlitTextureInfo(depthReadTexture).pixel_format)) {
                                mask &= ~GL_STENCIL_BUFFER_BIT;
                            }
                        }
                    }
                }
            }

            if (depthReadTexture && depthDrawTexture &&
                srcWidth > 0 && srcHeight > 0 &&
                mglBlitTextureInfo(depthReadTexture).pixel_format == mglBlitTextureInfo(depthDrawTexture).pixel_format &&
                mglBlitTextureInfo(depthReadTexture).sample_count == 1u && mglBlitTextureInfo(depthDrawTexture).sample_count == 1u) {
                BOOL depthIsScaled = (srcWidth != dstWidth) || (srcHeight != dstHeight);

                if (!depthIsScaled) {
                    /* Same-size depth blit via MTLBlitCommandEncoder */
                    GLint copyDstX0 = dstX0;
                    GLint copyDstY0 = dstY0;
                    GLint copyDstX1 = dstX1;
                    GLint copyDstY1 = dstY1;
                    if (glm_ctx->state.caps.scissor_test) {
                        GLint scissorX0 = glm_ctx->state.var.scissor_box[0];
                        GLint scissorY0 = glm_ctx->state.var.scissor_box[1];
                        GLint scissorX1 = scissorX0 + glm_ctx->state.var.scissor_box[2];
                        GLint scissorY1 = scissorY0 + glm_ctx->state.var.scissor_box[3];
                        copyDstX0 = MAX(copyDstX0, scissorX0);
                        copyDstY0 = MAX(copyDstY0, scissorY0);
                        copyDstX1 = MIN(copyDstX1, scissorX1);
                        copyDstY1 = MIN(copyDstY1, scissorY1);
                    }

                    GLint copyWidth = copyDstX1 - copyDstX0;
                    GLint copyHeight = copyDstY1 - copyDstY0;
                    GLint copySrcX = srcX0 + (copyDstX0 - dstX0);
                    GLint copySrcY = srcY0 + (copyDstY0 - dstY0);
                    if (copyWidth > 0 && copyHeight > 0 &&
                        copySrcX >= 0 && copySrcY >= 0 &&
                        copySrcX + copyWidth <= (GLint)mglBlitTextureInfo(depthReadTexture).width &&
                        copySrcY + copyHeight <= (GLint)mglBlitTextureInfo(depthReadTexture).height &&
                        copyDstX0 >= 0 && copyDstY0 >= 0 &&
                        copyDstX1 <= (GLint)mglBlitTextureInfo(depthDrawTexture).width &&
                        copyDstY1 <= (GLint)mglBlitTextureInfo(depthDrawTexture).height) {
                        [self endRenderEncoding];
                        if ([self ensureWritableCommandBuffer:"mtlBlitFramebuffer.depthStencil"]) {
                            if (depthStencilMask & GL_DEPTH_BUFFER_BIT) {
                                [self mglApplyPendingFBODepthClearForReadback:depthReadFBO
                                                                   attachment:depthReadAttachment
                                                                   textureObj:depthReadObject
                                                                   mtlTexture:depthReadTexture];
                                [self mglApplyPendingFBODepthClearForReadback:depthDrawFBO
                                                                   attachment:depthDrawAttachment
                                                                   textureObj:depthDrawObject
                                                                   mtlTexture:depthDrawTexture];
                            }
                            id depthBlit =
                                (__bridge id)mglRenderCreateBlitEncoderBorrowed(
                                    _commandState.currentCommandBufferOwner);
                            if (depthBlit) {
                                NSUInteger sourceMetalY =
                                    mglBlitTextureInfo(depthReadTexture).height - (NSUInteger)(copySrcY + copyHeight);
                                NSUInteger destinationMetalY =
                                    mglBlitTextureInfo(depthDrawTexture).height - (NSUInteger)(copyDstY0 + copyHeight);
                                mglBlitCopyTexture(
                                    depthBlit, depthReadTexture,
                                    depthReadSubresource.slice,
                                    depthReadSubresource.level,
                                    mglBlitOrigin((NSUInteger)copySrcX,
                                                  sourceMetalY,
                                                  depthReadSubresource.depthPlane),
                                    mglBlitSize((NSUInteger)copyWidth,
                                                (NSUInteger)copyHeight, 1u),
                                    depthDrawTexture,
                                    depthDrawSubresource.slice,
                                    depthDrawSubresource.level,
                                    mglBlitOrigin((NSUInteger)copyDstX0,
                                                  destinationMetalY,
                                                  depthDrawSubresource.depthPlane));
                                mglBlitEndBlitEncoder(depthBlit);
                                mglMarkTextureLevelRenderTargetWritten(depthDrawObject, depthDrawAttachment->level);
                            }
                        }
                    }
                } else {
                    /* Scaled depth blit via render pass with depth-writing shader.
                     * Only GL_NEAREST is supported (GL_LINEAR for depth is not allowed
                     * by the GL spec; filter must be GL_NEAREST when depth/stencil is
                     * in the mask). */
                    if (filter == GL_NEAREST &&
                        depthReadSubresource.level == 0u &&
                        depthReadSubresource.slice == 0u &&
                        depthReadSubresource.depthPlane == 0u &&
                        mglBlitTextureInfo(depthReadTexture).texture_type == MGLTextureType2D &&
                        depthDrawSubresource.level == 0u &&
                        depthDrawSubresource.slice == 0u &&
                        depthDrawSubresource.depthPlane == 0u &&
                        mglBlitTextureInfo(depthDrawTexture).texture_type == MGLTextureType2D) {
                        /* Apply pending depth clears before the scaled blit so the
                         * source texture reflects any lazy glClear operations. */
                        if (depthStencilMask & GL_DEPTH_BUFFER_BIT) {
                            [self endRenderEncoding];
                            if ([self ensureWritableCommandBuffer:"mtlBlitFramebuffer.depthScaledClear"]) {
                                [self mglApplyPendingFBODepthClearForReadback:depthReadFBO
                                                                   attachment:depthReadAttachment
                                                                   textureObj:depthReadObject
                                                                   mtlTexture:depthReadTexture];
                                [self mglApplyPendingFBODepthClearForReadback:depthDrawFBO
                                                                   attachment:depthDrawAttachment
                                                                   textureObj:depthDrawObject
                                                                   mtlTexture:depthDrawTexture];
                            }
                        }

                        id depthPipeline =
                            [self scaledDepthBlitPipelineForPixelFormat:mglBlitTextureInfo(depthDrawTexture).pixel_format];
                        id sampler = [self scaledBlitSamplerForFilter:GL_NEAREST];
                        if (depthPipeline && sampler) {
                            [self endRenderEncoding];
                            if ([self ensureWritableCommandBuffer:"mtlBlitFramebuffer.depthScaled"]) {
                                /* For packed depth+stencil formats, also set the stencil
                                 * attachment to the same texture so Metal preserves the
                                 * stencil component during the render pass. */
                                BOOL isPackedDepthStencil =
                                    mglMetalPixelFormatIsPackedDepthStencil(mglBlitTextureInfo(depthDrawTexture).pixel_format);

                                MGLRenderPassState scaledDepthState =
                                    mglBlitDefaultRenderPassState();
                                scaledDepthState.depth.attachment =
                                    mglBlitRenderPassAttachment(
                                        depthDrawTexture, 0u, 0u, 0u,
                                        MGLLoadActionLoad,
                                        MGLStoreActionStore);
                                if (isPackedDepthStencil) {
                                    scaledDepthState.stencil.attachment =
                                        mglBlitRenderPassAttachment(
                                            depthDrawTexture, 0u, 0u, 0u,
                                            MGLLoadActionLoad,
                                            MGLStoreActionStore);
                                }

                                id depthEncoder =
                                    mglBlitCreateRenderEncoder(&_commandState,
                                                               &scaledDepthState);
                                if (depthEncoder) {
                                    mglBlitSetRenderPipeline(depthEncoder, depthPipeline);
                                    mglBlitSetDepthStencil(depthEncoder,
                                                           [self clearRectDepthState]);

                                    /* Compute UVs for the source region in Metal's
                                     * texture coordinate space (Y-flipped). */
                                    NSUInteger srcTexW = mglBlitTextureInfo(depthReadTexture).width;
                                    NSUInteger srcTexH = mglBlitTextureInfo(depthReadTexture).height;
                                    float invSrcW = srcTexW ? (1.0f / (float)srcTexW) : 0.0f;
                                    float invSrcH = srcTexH ? (1.0f / (float)srcTexH) : 0.0f;
                                    float srcMinXf = (float)srcX0;
                                    float srcMaxXf = (float)srcX1;
                                    float srcMinYf = (float)srcY0;
                                    float srcMaxYf = (float)srcY1;
                                    float uvLeft = MAX(0.0f, MIN(1.0f, srcMinXf * invSrcW));
                                    float uvRight = MAX(0.0f, MIN(1.0f, srcMaxXf * invSrcW));
                                    /* Metal Y is top-down; GL Y is bottom-up.
                                     * uvTop maps to the top of the source region in
                                     * Metal space, which is (srcTexH - srcMaxY). */
                                    float uvTop = MAX(0.0f, MIN(1.0f, (float)((double)srcTexH - srcMaxYf) * invSrcH));
                                    float uvBottom = MAX(0.0f, MIN(1.0f, (float)((double)srcTexH - srcMinYf) * invSrcH));

                                    MGLScaledBlitParams params;
                                    params.uvRect = (vector_float4){
                                        uvLeft,
                                        uvTop,
                                        uvRight,
                                        uvBottom
                                    };
                                    params.forceOpaqueAlpha = 0.0f;
                                    params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

                                    mglBlitSetRenderBytes(depthEncoder, &params,
                                                          sizeof(params),
                                                          MGL_RENDER_BINDING_STAGE_VERTEX,
                                                          0);
                                    mglBlitSetRenderBytes(depthEncoder, &params,
                                                          sizeof(params),
                                                          MGL_RENDER_BINDING_STAGE_FRAGMENT,
                                                          0);
                                    mglBlitSetRenderTexture(depthEncoder,
                                                            depthReadTexture,
                                                            MGL_RENDER_BINDING_STAGE_FRAGMENT,
                                                            0);
                                    mglBlitSetRenderSampler(depthEncoder, sampler,
                                                            MGL_RENDER_BINDING_STAGE_FRAGMENT,
                                                            0);

                                    /* Set viewport to the destination region in
                                     * Metal's coordinate space (Y-flipped). */
                                    float dstMinXf = (float)dstX0;
                                    float dstMaxXf = (float)dstX1;
                                    float dstMinYf = (float)dstY0;
                                    float dstMaxYf = (float)dstY1;
                                    NSUInteger dstTexW = mglBlitTextureInfo(depthDrawTexture).width;
                                    NSUInteger dstTexH = mglBlitTextureInfo(depthDrawTexture).height;
                                    double dstMinXd = fmin(dstMinXf, dstMaxXf);
                                    double dstMaxXd = fmax(dstMinXf, dstMaxXf);
                                    double dstMinYd = fmin(dstMinYf, dstMaxYf);
                                    double dstMaxYd = fmax(dstMinYf, dstMaxYf);
                                    double dstWd = dstMaxXd - dstMinXd;
                                    double dstHd = dstMaxYd - dstMinYd;
                                    double scaledDstMetalY = (double)dstTexH - dstMaxYd;

                                    /* Scissor rect to limit writes to the
                                     * destination region. */
                                    NSInteger scissorX0 = (NSInteger)floor(dstMinXd + 0.00001);
                                    NSInteger scissorX1 = (NSInteger)ceil(dstMaxXd - 0.00001);
                                    NSInteger scissorY0 = (NSInteger)floor(scaledDstMetalY + 0.00001);
                                    NSInteger scissorY1 = (NSInteger)ceil(scaledDstMetalY + dstHd - 0.00001);
                                    scissorX0 = MAX((NSInteger)0, MIN(scissorX0, (NSInteger)dstTexW));
                                    scissorX1 = MAX((NSInteger)0, MIN(scissorX1, (NSInteger)dstTexW));
                                    scissorY0 = MAX((NSInteger)0, MIN(scissorY0, (NSInteger)dstTexH));
                                    scissorY1 = MAX((NSInteger)0, MIN(scissorY1, (NSInteger)dstTexH));
                                    if (glm_ctx && glm_ctx->state.caps.scissor_test) {
                                        NSInteger glScissorX0 = glm_ctx->state.var.scissor_box[0];
                                        NSInteger glScissorY0 = glm_ctx->state.var.scissor_box[1];
                                        NSInteger glScissorX1 = glScissorX0 + glm_ctx->state.var.scissor_box[2];
                                        NSInteger glScissorY1 = glScissorY0 + glm_ctx->state.var.scissor_box[3];
                                        NSInteger metalScissorY0 = (NSInteger)dstTexH - glScissorY1;
                                        NSInteger metalScissorY1 = (NSInteger)dstTexH - glScissorY0;
                                        scissorX0 = MAX(scissorX0, glScissorX0);
                                        scissorX1 = MIN(scissorX1, glScissorX1);
                                        scissorY0 = MAX(scissorY0, metalScissorY0);
                                        scissorY1 = MIN(scissorY1, metalScissorY1);
                                    }
                                    if (scissorX1 > scissorX0 && scissorY1 > scissorY0) {
                                        mglBlitSetRenderViewport(depthEncoder, (MGLViewportValue){
                                            .origin_x = dstMinXd,
                                            .origin_y = scaledDstMetalY,
                                            .width = dstWd,
                                            .height = dstHd,
                                            .znear = 0.0,
                                            .zfar = 1.0
                                        });
                                        mglBlitSetRenderScissor(depthEncoder, (MGLScissorRectValue){
                                            .x = (NSUInteger)scissorX0,
                                            .y = (NSUInteger)scissorY0,
                                            .width = (NSUInteger)(scissorX1 - scissorX0),
                                            .height = (NSUInteger)(scissorY1 - scissorY0)
                                        });
                                        mglBlitDrawPrimitives(depthEncoder,
                                                              MGLPrimitiveTypeTriangleStrip,
                                                              0, 4);
                                    }
                                    mglBlitEndRenderEncoder(depthEncoder);
                                    mglMarkTextureLevelRenderTargetWritten(depthDrawObject, depthDrawAttachment->level);
                                }
                            }
                        } else {
                            static uint64_t s_scaledDepthBlitSkipCount = 0;
                            uint64_t hit = ++s_scaledDepthBlitSkipCount;
                            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                                NSLog(@"MGL WARN: mtlBlitFramebuffer scaled depth blit unavailable pipeline=%p sampler=%p hit=%llu",
                                      depthPipeline, sampler, (unsigned long long)hit);
                            }
                        }
                    }
                }
            }
        }
    }
    return mask;
}

/* Resolve read/draw framebuffer attachments for mtlBlitFramebuffer.
 * Fills the MGLBlitColorState struct with source/destination textures,
 * attachments, and subresources.  Returns NO on early-exit (missing
 * attachment / texture); YES on success. */
- (BOOL)resolveBlitFramebufferAttachments:(GLMContext)glm_ctx
                                    srcX0:(GLint)srcX0 srcY0:(GLint)srcY0 srcX1:(GLint)srcX1 srcY1:(GLint)srcY1
                                    dstX0:(GLint)dstX0 dstY0:(GLint)dstY0 dstX1:(GLint)dstX1 dstY1:(GLint)dstY1
                                outState:(MGLBlitColorState *)st
                       outReadAttachment:(GLenum *)outReadAttachment
{
    MGL_ASSERT_GL_THREAD();
    Framebuffer * readfbo, * drawfbo;
    GLenum readAttachment, drawAttachment;
    FBOAttachment *readFBOAttachment = NULL;
    Texture *readTextureObject = NULL;
    FBOAttachment *drawFBOAttachment = NULL;
    Texture *drawTextureObject = NULL;
    MGLMetalAttachmentSubresource readSubresource = {0u, 0u, 0u};
    MGLMetalAttachmentSubresource drawSubresource = {0u, 0u, 0u};
    //int readtex, drawtex;

    readfbo = glm_ctx->state.readbuffer;
    drawfbo = glm_ctx->state.framebuffer;

    if (drawfbo == NULL) {
        NSUInteger requestedDrawableWidth = (NSUInteger)MAX(0, MAX(dstX0, dstX1));
        NSUInteger requestedDrawableHeight = (NSUInteger)MAX(0, MAX(dstY0, dstY1));
        if ([self mglEnsureLayerDrawableSizeAtLeastWidth:requestedDrawableWidth
                                                  height:requestedDrawableHeight
                                                  reason:"blitFramebuffer.defaultDraw"]) {
            _drawable = [self mglNextDrawable];
        }
    }

    id readtexid;

    if (readfbo==NULL) {
        if (!_drawable || ![self mglDrawableTexture]) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer has no drawable source texture");
            return NO;
        }
        readtexid = [self mglDrawableTexture];
    } else {
        readAttachment = glm_ctx->state.read_buffer;
        if (readAttachment == GL_NONE) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer skipped color blit with GL_READ_BUFFER=GL_NONE");
            return NO;
        }
        if (!isColorAttachment(glm_ctx, readAttachment) &&
            readAttachment != GL_DEPTH_ATTACHMENT &&
            readAttachment != GL_STENCIL_ATTACHMENT &&
            readAttachment != GL_DEPTH_STENCIL_ATTACHMENT)
        {
            // OpenGL compatibility enums (e.g. GL_FRONT/GL_BACK) are not valid
            // FBO attachment enums. For user FBO blits, treat them as COLOR_ATTACHMENT0.
            readAttachment = GL_COLOR_ATTACHMENT0;
        }

        readFBOAttachment = getFBOAttachment(glm_ctx, readfbo, readAttachment);
        if (!readFBOAttachment) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer read attachment missing");
            return NO;
        }
        readSubresource = mglMetalAttachmentSubresourceForAttachment(readFBOAttachment);
        if (readFBOAttachment->textarget == GL_RENDERBUFFER)
        {
            readTextureObject = readFBOAttachment->buf.rbo->tex;
        }
        else
        {
            readTextureObject = readFBOAttachment->buf.tex;
        }
        if (!readTextureObject) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer read texture object missing");
            return NO;
        }
        if (!readTextureObject->mtl_data || readTextureObject->dirty_bits) {
            if (![self bindMTLTexture:readTextureObject]) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer failed to bind read texture to Metal");
                return NO;
            }
        }
        readtexid = (__bridge id)(readTextureObject->mtl_data);
        if (!readtexid) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer read MTL texture missing");
            return NO;
        }
    }


    id drawtexid;
    if (drawfbo==NULL) {
        if (!_drawable || ![self mglDrawableTexture]) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer has no drawable destination texture");
            return NO;
        }
        drawtexid = [self mglDrawableTexture];
    } else {
        drawAttachment = glm_ctx->state.draw_buffer;
        if (drawAttachment == GL_NONE) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer skipped color blit with GL_DRAW_BUFFER=GL_NONE");
            return NO;
        }
        if (!isColorAttachment(glm_ctx, drawAttachment) &&
            drawAttachment != GL_DEPTH_ATTACHMENT &&
            drawAttachment != GL_STENCIL_ATTACHMENT &&
            drawAttachment != GL_DEPTH_STENCIL_ATTACHMENT)
        {
            drawAttachment = GL_COLOR_ATTACHMENT0;
        }

        drawFBOAttachment = getFBOAttachment(glm_ctx, drawfbo, drawAttachment);
        if (!drawFBOAttachment) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer draw attachment missing");
            return NO;
        }
        drawSubresource = mglMetalAttachmentSubresourceForAttachment(drawFBOAttachment);
        if (drawFBOAttachment->textarget == GL_RENDERBUFFER)
        {
            drawTextureObject = drawFBOAttachment->buf.rbo->tex;
        }
        else
        {
            drawTextureObject = drawFBOAttachment->buf.tex;
        }
        if (!drawTextureObject) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer draw texture object missing");
            return NO;
        }
        drawTextureObject->is_render_target = true;
        if (!drawTextureObject->mtl_data || drawTextureObject->dirty_bits) {
            if (![self bindMTLTexture:drawTextureObject]) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer failed to bind draw texture to Metal");
                return NO;
            }
        }
        drawtexid = (__bridge id)(drawTextureObject->mtl_data);
        if (!drawtexid) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer draw MTL texture missing");
            return NO;
        }
    }

    st->readfbo = readfbo;
    st->drawfbo = drawfbo;
    st->readFBOAttachment = readFBOAttachment;
    st->drawFBOAttachment = drawFBOAttachment;
    st->readTextureObject = readTextureObject;
    st->drawTextureObject = drawTextureObject;
    st->readSubresource = readSubresource;
    st->drawSubresource = drawSubresource;
    st->readtexid = readtexid;
    st->drawtexid = drawtexid;
    *outReadAttachment = readAttachment;
    return YES;
}

/* Multisample resolve for mtlBlitFramebuffer color blit.
 * When the source is multisample and the destination is single-sample,
 * resolves the source to a temporary single-sample texture.
 * Updates *readtexidPtr / *readSubresourcePtr to the resolved texture.
 * Returns NO on failure (caller should return); YES on success. */
- (BOOL)blitFramebufferResolveMsaaSource:(id *)readtexidPtr
                                drawtexid:(id)drawtexid
                        readSubresource:(MGLMetalAttachmentSubresource *)readSubresourcePtr
                                  srcTexW:(NSUInteger)srcTexW srcTexH:(NSUInteger)srcTexH
                       readTextureObject:(Texture *)readTextureObject
                       outDidMsaaResolve:(BOOL *)outDidMsaaResolve
{
    id readtexid = *readtexidPtr;
    MGLMetalAttachmentSubresource readSubresource = *readSubresourcePtr;
    BOOL didMsaaResolve = NO;
    if (mglBlitTextureInfo(readtexid).sample_count > 1u &&
        mglBlitTextureInfo(drawtexid).sample_count <= 1u &&
        !mglMetalPixelFormatIsIntegerColor(mglBlitTextureInfo(readtexid).pixel_format)) {
        MGLRenderTextureDescriptorState resolveDesc = {0};
        resolveDesc.texture_type = MGLTextureType2D;
        resolveDesc.pixel_format = mglBlitTextureInfo(readtexid).pixel_format;
        resolveDesc.width = srcTexW;
        resolveDesc.height = srcTexH;
        resolveDesc.depth = 1;
        resolveDesc.mipmap_level_count = 1;
        resolveDesc.sample_count = 1;
        resolveDesc.array_length = 1;
        resolveDesc.usage = MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
        resolveDesc.storage_mode = MGLStorageModePrivate;
        id resolveTex =
            mglBlitCreateTexture(_device, &resolveDesc);
        if (!resolveTex) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create MSAA resolve texture srcSamples=%lu",
                  (unsigned long)mglBlitTextureInfo(readtexid).sample_count);
            return NO;
        }

        BOOL resolveEncoded =
            mglRenderEncodeMultisampleResolveForCommandBufferOwner(
                _commandState.currentCommandBufferOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                (__bridge void *)readtexid, readSubresource.level,
                readSubresource.slice, readSubresource.depthPlane,
                (__bridge void *)resolveTex, 0, 0, 0, 0) == 0;
        if (!resolveEncoded) return NO;

        /* Synchronize the resolved texture so the subsequent blit/shader can
         * read it on a tile-based Apple GPU without stale tile memory. */
        id syncBlit =
            (__bridge id)mglRenderCreateBlitEncoderBorrowed(
                _commandState.currentCommandBufferOwner);
        if (syncBlit) {
            mglBlitSynchronizeTexture(syncBlit, resolveTex, 0, 0);
            mglBlitEndBlitEncoder(syncBlit);
        }

        static uint64_t s_msaaResolveLogCount = 0;
        uint64_t msaaHit = ++s_msaaResolveLogCount;
        if (msaaHit <= 8ull || (msaaHit % 256ull) == 0ull) {
            mglTraceLogNSString(@"MGL TRACE blitFramebuffer.msaaResolve hit=%llu srcSamples=%lu srcTex=%lux%lu srcObj=%u",
                  (unsigned long long)msaaHit,
                  (unsigned long)mglBlitTextureInfo(readtexid).sample_count,
                  (unsigned long)srcTexW, (unsigned long)srcTexH,
                  readTextureObject ? (unsigned)readTextureObject->name : 0u);
        }

        /* Replace the source with the resolved single-sample texture. The
         * resolved texture has the same dimensions, so srcTexW/srcTexH remain
         * valid. Reset the subresource to {0,0,0} (fresh 2D texture). */
        readtexid = resolveTex;
        readSubresource.level = 0u;
        readSubresource.slice = 0u;
        readSubresource.depthPlane = 0u;
        didMsaaResolve = YES;
    }
    *readtexidPtr = readtexid;
    *readSubresourcePtr = readSubresource;
    *outDidMsaaResolve = didMsaaResolve;
    return YES;
}

/* Integer-color blit paths for mtlBlitFramebuffer.
 * Handles MSAA-resolve and direct-blit for integer pixel formats via
 * resolveIntegerMultisampleTexture: or MTLBlitCommandEncoder.
 * Returns YES if a path was taken (caller should return). */
- (BOOL)blitFramebufferIntegerColorWithState:(MGLBlitColorState *)st
{
    id readtexid = st->readtexid;
    id drawtexid = st->drawtexid;
    MGLMetalAttachmentSubresource readSubresource = st->readSubresource;
    MGLMetalAttachmentSubresource drawSubresource = st->drawSubresource;
    NSInteger copyW = st->copyW;
    NSInteger copyH = st->copyH;
    NSInteger copySrcX = st->copySrcX;
    NSInteger srcMetalY = st->srcMetalY;
    NSInteger copyDstX = st->copyDstX;
    NSInteger dstMetalY = st->dstMetalY;
    NSUInteger srcTexW = st->srcTexW;
    NSUInteger srcTexH = st->srcTexH;
    NSUInteger dstTexW = st->dstTexW;
    NSUInteger dstTexH = st->dstTexH;
    Texture *readTextureObject = st->readTextureObject;
    Texture *drawTextureObject = st->drawTextureObject;
    FBOAttachment *drawFBOAttachment = st->drawFBOAttachment;
    BOOL blitNeedsFlip = st->blitNeedsFlip;
    double srcW = st->srcW;
    double srcH = st->srcH;
    double dstW = st->dstW;
    double dstH = st->dstH;
    if (mglBlitTextureInfo(readtexid).sample_count > 1u &&
        mglBlitTextureInfo(drawtexid).sample_count <= 1u &&
        mglMetalPixelFormatIsIntegerColor(mglBlitTextureInfo(readtexid).pixel_format)) {
        if (copyW <= 0 || copyH <= 0 ||
            copySrcX < 0 || srcMetalY < 0 || copyDstX < 0 || dstMetalY < 0 ||
            copySrcX + copyW > (NSInteger)srcTexW ||
            srcMetalY + copyH > (NSInteger)srcTexH ||
            copyDstX + copyW > (NSInteger)dstTexW ||
            dstMetalY + copyH > (NSInteger)dstTexH) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer integer MSAA resolve invalid src=(%ld,%ld %ldx%ld) dst=(%ld,%ld) srcTex=%lux%lu dstTex=%lux%lu",
                  (long)copySrcX, (long)srcMetalY, (long)copyW, (long)copyH,
                  (long)copyDstX, (long)dstMetalY,
                  (unsigned long)srcTexW,
                  (unsigned long)srcTexH,
                  (unsigned long)dstTexW,
                  (unsigned long)dstTexH);
            return YES;
        }

        BOOL resolvedInteger =
            [self resolveIntegerMultisampleTexture:readtexid
                                         toTexture:drawtexid
                                         srcOrigin:mglBlitOrigin((NSUInteger)copySrcX,
                                                                 (NSUInteger)srcMetalY,
                                                                 readSubresource.depthPlane)
                                         dstOrigin:mglBlitOrigin((NSUInteger)copyDstX,
                                                                 (NSUInteger)dstMetalY,
                                                                 drawSubresource.depthPlane)
                                              size:mglBlitSize((NSUInteger)copyW,
                                                               (NSUInteger)copyH,
                                                               1u)
                                            reason:"blitFramebuffer.integerMsaa"];
        if (!resolvedInteger) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer integer MSAA resolve failed fmt=%lu",
                  (unsigned long)mglBlitTextureInfo(readtexid).pixel_format);
            return YES;
        }
        if (drawTextureObject && drawFBOAttachment) {
            mglMarkTextureLevelRenderTargetWritten(drawTextureObject, drawFBOAttachment->level);
            [self updateGLSampledRenderTargetCopyForTexture:drawTextureObject
                                                     source:drawtexid
                                                     reason:"blit_framebuffer_integer_msaa"];
        }
        return YES;
    }

    if (mglBlitTextureInfo(readtexid).sample_count <= 1u &&
        mglBlitTextureInfo(drawtexid).sample_count <= 1u &&
        mglBlitTextureInfo(readtexid).pixel_format == mglBlitTextureInfo(drawtexid).pixel_format &&
        mglMetalPixelFormatIsIntegerColor(mglBlitTextureInfo(readtexid).pixel_format) &&
        !blitNeedsFlip &&
        mglNearlyEqual(srcW, dstW) &&
        mglNearlyEqual(srcH, dstH)) {
        if (copyW <= 0 || copyH <= 0 ||
            copySrcX < 0 || srcMetalY < 0 || copyDstX < 0 || dstMetalY < 0 ||
            copySrcX + copyW > (NSInteger)srcTexW ||
            srcMetalY + copyH > (NSInteger)srcTexH ||
            copyDstX + copyW > (NSInteger)dstTexW ||
            dstMetalY + copyH > (NSInteger)dstTexH) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer integer direct blit invalid src=(%ld,%ld %ldx%ld) dst=(%ld,%ld) srcTex=%lux%lu dstTex=%lux%lu",
                  (long)copySrcX, (long)srcMetalY, (long)copyW, (long)copyH,
                  (long)copyDstX, (long)dstMetalY,
                  (unsigned long)srcTexW,
                  (unsigned long)srcTexH,
                  (unsigned long)dstTexW,
                  (unsigned long)dstTexH);
            return YES;
        }

        id integerBlit =
            (__bridge id)mglRenderCreateBlitEncoderBorrowed(
                _commandState.currentCommandBufferOwner);
        if (!integerBlit) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create integer direct blit encoder");
            return YES;
        }
        if (readTextureObject && readTextureObject->is_render_target) {
            mglBlitSynchronizeTexture(integerBlit, readtexid,
                                      readSubresource.slice,
                                      readSubresource.level);
        }
        mglBlitCopyTexture(
            integerBlit, readtexid, readSubresource.slice,
            readSubresource.level,
            mglBlitOrigin((NSUInteger)copySrcX, (NSUInteger)srcMetalY,
                          readSubresource.depthPlane),
            mglBlitSize((NSUInteger)copyW, (NSUInteger)copyH, 1u),
            drawtexid, drawSubresource.slice, drawSubresource.level,
            mglBlitOrigin((NSUInteger)copyDstX, (NSUInteger)dstMetalY,
                          drawSubresource.depthPlane));
        mglBlitEndBlitEncoder(integerBlit);
        if (drawTextureObject && drawFBOAttachment) {
            mglMarkTextureLevelRenderTargetWritten(drawTextureObject, drawFBOAttachment->level);
            [self updateGLSampledRenderTargetCopyForTexture:drawTextureObject
                                                     source:drawtexid
                                                     reason:"blit_framebuffer_integer_direct"];
        }
        return YES;
    }
    return NO;
}

/* Scaled / format-converted / Y-flipped color blit for mtlBlitFramebuffer.
 * Uses a render pass with a scaled-blit shader pipeline.
 * Returns YES if the scaled blit was performed (caller should return). */
- (BOOL)blitFramebufferScaledColorWithState:(MGLBlitColorState *)st
{
    GLMContext glm_ctx = st->glm_ctx;
    Framebuffer *drawfbo = st->drawfbo;
    GLenum filter = st->filter;
    FBOAttachment *drawFBOAttachment = st->drawFBOAttachment;
    Texture *readTextureObject = st->readTextureObject;
    Texture *drawTextureObject = st->drawTextureObject;
    MGLMetalAttachmentSubresource readSubresource = st->readSubresource;
    MGLMetalAttachmentSubresource drawSubresource = st->drawSubresource;
    id readtexid = st->readtexid;
    id drawtexid = st->drawtexid;
    NSUInteger srcTexW = st->srcTexW;
    NSUInteger srcTexH = st->srcTexH;
    NSUInteger dstTexW = st->dstTexW;
    NSUInteger dstTexH = st->dstTexH;
    BOOL srcXForward = st->srcXForward;
    BOOL srcYForward = st->srcYForward;
    BOOL dstXForward = st->dstXForward;
    BOOL dstYForward = st->dstYForward;
    double srcMinX = st->srcMinX;
    double srcMaxX = st->srcMaxX;
    double srcMinY = st->srcMinY;
    double srcMaxY = st->srcMaxY;
    double dstMinX = st->dstMinX;
    double dstMaxX = st->dstMaxX;
    double dstMinY = st->dstMinY;
    double dstMaxY = st->dstMaxY;
    double srcW = st->srcW;
    double srcH = st->srcH;
    double dstW = st->dstW;
    double dstH = st->dstH;
    double scaledDstMetalY = st->scaledDstMetalY;
    BOOL needsScaledBlit = st->needsScaledBlit;
    if (needsScaledBlit) {
        if (readtexid == drawtexid) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled self-blit unsupported texture=%p, skipping", readtexid);
            return YES;
        }
        if (readSubresource.level != 0u ||
            readSubresource.slice != 0u ||
            readSubresource.depthPlane != 0u ||
            mglBlitTextureInfo(readtexid).texture_type != MGLTextureType2D) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled source subresource/type unsupported level=%lu slice=%lu depth=%lu type=%lu, skipping",
                  (unsigned long)readSubresource.level,
                  (unsigned long)readSubresource.slice,
                  (unsigned long)readSubresource.depthPlane,
                  (unsigned long)mglBlitTextureInfo(readtexid).texture_type);
            return YES;
        }

        id pipeline = [self scaledBlitPipelineForPixelFormat:mglBlitTextureInfo(drawtexid).pixel_format];
        id sampler = [self scaledBlitSamplerForFilter:filter];
        if (!pipeline || !sampler) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled path unavailable pipeline=%p sampler=%p", pipeline, sampler);
            return YES;
        }


        MGLRenderScaledBlitUVs uvs = {0};
        mglRenderScaledBlitUVs(
            (uint32_t)srcTexW, (uint32_t)srcTexH,
            srcMinX, srcMaxX, srcMinY, srcMaxY,
            srcXForward ? 1 : 0, srcYForward ? 1 : 0,
            dstXForward ? 1 : 0, dstYForward ? 1 : 0,
            &uvs);
        MGLScaledBlitParams params;
        params.uvRect = (vector_float4){
            uvs.uv_left,
            uvs.uv_top,
            uvs.uv_right,
            uvs.uv_bottom
        };
        params.forceOpaqueAlpha = (drawfbo == NULL && drawtexid == (_drawable ? [self mglDrawableTexture] : nil)) ? 1.0f : 0.0f;
        params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

        MGLRenderPassState scaledState =
            mglBlitDefaultRenderPassState();
        scaledState.color[0].attachment = mglBlitRenderPassAttachment(
            drawtexid, drawSubresource.level, drawSubresource.slice,
            drawSubresource.depthPlane, MGLLoadActionLoad,
            MGLStoreActionStore);

        id encoder =
            mglBlitCreateRenderEncoder(&_commandState, &scaledState);
        if (!encoder) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create scaled render encoder");
            return YES;
        }

        mglBlitSetRenderPipeline(encoder, pipeline);
        mglBlitSetRenderBytes(encoder, &params, sizeof(params),
                              MGL_RENDER_BINDING_STAGE_VERTEX, 0);
        mglBlitSetRenderBytes(encoder, &params, sizeof(params),
                              MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
        mglBlitSetRenderTexture(encoder, readtexid,
                                MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
        mglBlitSetRenderSampler(encoder, sampler,
                                MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);


        MGLRenderBlitScissorRect scissorBase = {0};
        mglRenderBlitScissorRect(
            dstMinX, dstMaxX, scaledDstMetalY, dstH,
            (uint32_t)dstTexW, (uint32_t)dstTexH, &scissorBase);
        NSInteger scissorX0 = (NSInteger)scissorBase.x0;
        NSInteger scissorX1 = (NSInteger)scissorBase.x1;
        NSInteger scissorY0 = (NSInteger)scissorBase.y0;
        NSInteger scissorY1 = (NSInteger)scissorBase.y1;
        if (glm_ctx && glm_ctx->state.caps.scissor_test) {
            NSInteger glScissorX0 = glm_ctx->state.var.scissor_box[0];
            NSInteger glScissorY0 = glm_ctx->state.var.scissor_box[1];
            NSInteger glScissorX1 = glScissorX0 + glm_ctx->state.var.scissor_box[2];
            NSInteger glScissorY1 = glScissorY0 + glm_ctx->state.var.scissor_box[3];
            NSInteger metalScissorY0 = (NSInteger)dstTexH - glScissorY1;
            NSInteger metalScissorY1 = (NSInteger)dstTexH - glScissorY0;
            scissorX0 = MAX(scissorX0, glScissorX0);
            scissorX1 = MIN(scissorX1, glScissorX1);
            scissorY0 = MAX(scissorY0, metalScissorY0);
            scissorY1 = MIN(scissorY1, metalScissorY1);
        }
        if (scissorX1 <= scissorX0 || scissorY1 <= scissorY0) {
            mglBlitEndRenderEncoder(encoder);
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled scissor is empty after clipping, skipping draw");
            return YES;
        }

        mglBlitSetRenderViewport(encoder, (MGLViewportValue){
            .origin_x = dstMinX,
            .origin_y = scaledDstMetalY,
            .width = dstW,
            .height = dstH,
            .znear = 0.0,
            .zfar = 1.0
        });
        mglBlitSetRenderScissor(encoder, (MGLScissorRectValue){
            .x = (NSUInteger)scissorX0,
            .y = (NSUInteger)scissorY0,
            .width = (NSUInteger)(scissorX1 - scissorX0),
            .height = (NSUInteger)(scissorY1 - scissorY0)
        });
        mglBlitDrawPrimitives(encoder, MGLPrimitiveTypeTriangleStrip, 0, 4);
        mglBlitEndRenderEncoder(encoder);
        if (drawfbo == NULL) {
            _defaultDrawableWrittenSinceLastSwap = YES;
        }
        if (drawTextureObject && drawFBOAttachment) {
            mglMarkTextureLevelRenderTargetWritten(drawTextureObject, drawFBOAttachment->level);
            [self updateGLSampledRenderTargetCopyForTexture:drawTextureObject
                                                     source:drawtexid
                                                     reason:"blit_framebuffer_scaled"];
        }
        // When the source is also a render target, refresh its sampled copy
        // so future fragment-shader samples see useCopy=1 instead of falling
        // back to the direct texture (useCopy=0).
        if (readTextureObject &&
            readTextureObject->is_render_target &&
            readtexid) {
            [self updateGLSampledRenderTargetCopyForTexture:readTextureObject
                                                     source:readtexid
                                                     reason:"blit_framebuffer_scaled_src"];
        }
        return YES;
    }
    return NO;
}

/* Direct MTLBlitCommandEncoder color copy for mtlBlitFramebuffer.
 * Same-size, same-format, no-flip blit via copyFromTexture:toTexture:. */
- (void)blitFramebufferDirectColorCopyWithState:(MGLBlitColorState *)st
{
    Framebuffer *drawfbo = st->drawfbo;
    FBOAttachment *drawFBOAttachment = st->drawFBOAttachment;
    Texture *readTextureObject = st->readTextureObject;
    Texture *drawTextureObject = st->drawTextureObject;
    MGLMetalAttachmentSubresource readSubresource = st->readSubresource;
    MGLMetalAttachmentSubresource drawSubresource = st->drawSubresource;
    id readtexid = st->readtexid;
    id drawtexid = st->drawtexid;
    NSUInteger srcTexW = st->srcTexW;
    NSUInteger srcTexH = st->srcTexH;
    NSUInteger dstTexW = st->dstTexW;
    NSUInteger dstTexH = st->dstTexH;
    NSInteger copyW = st->copyW;
    NSInteger copyH = st->copyH;
    NSInteger copySrcX = st->copySrcX;
    NSInteger copySrcY = st->copySrcY;
    NSInteger copyDstX = st->copyDstX;
    NSInteger copyDstY = st->copyDstY;
    NSInteger srcMetalY = st->srcMetalY;
    NSInteger dstMetalY = st->dstMetalY;
    BOOL didMsaaResolve = st->didMsaaResolve;
    // start blit encoder
    id blitCommandEncoder;
    blitCommandEncoder =
        (__bridge id)mglRenderCreateBlitEncoderBorrowed(
            _commandState.currentCommandBufferOwner);
    if (!blitCommandEncoder) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create blit encoder");
        return;
    }
    if (copyW <= 0 || copyH <= 0 ||
        copySrcX < 0 || copySrcY < 0 || copyDstX < 0 || copyDstY < 0 ||
        srcMetalY < 0 || dstMetalY < 0 ||
        copySrcX + copyW > (NSInteger)srcTexW ||
        copySrcY + copyH > (NSInteger)srcTexH ||
        copyDstX + copyW > (NSInteger)dstTexW ||
        copyDstY + copyH > (NSInteger)dstTexH) {
        mglBlitEndBlitEncoder(blitCommandEncoder);
        NSLog(@"MGL WARN: mtlBlitFramebuffer direct copy invalid after clipping src=(%ld,%ld %ldx%ld) dst=(%ld,%ld) srcTex=%lux%lu dstTex=%lux%lu",
              (long)copySrcX, (long)copySrcY, (long)copyW, (long)copyH,
              (long)copyDstX, (long)copyDstY,
              (unsigned long)srcTexW, (unsigned long)srcTexH,
              (unsigned long)dstTexW, (unsigned long)dstTexH);
        return;
    }

    // If the source is a render target, ensure all GPU writes are visible
    // before the blit encoder reads it.  Without this synchronizeTexture
    // call, a tile-based Apple GPU may read stale tile memory when the
    // texture was recently written by a preceding render pass.
    if (readTextureObject && readTextureObject->is_render_target) {
        mglBlitSynchronizeTexture(blitCommandEncoder, readtexid,
                                  readSubresource.slice,
                                  readSubresource.level);
    }

    mglBlitCopyTexture(
        blitCommandEncoder, readtexid, readSubresource.slice,
        readSubresource.level,
        mglBlitOrigin((NSUInteger)copySrcX, (NSUInteger)srcMetalY,
                      readSubresource.depthPlane),
        mglBlitSize((NSUInteger)copyW, (NSUInteger)copyH, 1u), drawtexid,
        drawSubresource.slice, drawSubresource.level,
        mglBlitOrigin((NSUInteger)copyDstX, (NSUInteger)dstMetalY,
                      drawSubresource.depthPlane));
    mglBlitEndBlitEncoder(blitCommandEncoder);
    if (drawfbo == NULL) {
        _defaultDrawableWrittenSinceLastSwap = YES;
    }
    if (drawTextureObject && drawFBOAttachment) {
        mglMarkTextureLevelRenderTargetWritten(drawTextureObject, drawFBOAttachment->level);
        [self updateGLSampledRenderTargetCopyForTexture:drawTextureObject
                                                 source:drawtexid
                                                 reason:"blit_framebuffer_copy"];
    }
    // When the source is also a render target, refresh its sampled copy
    // so future fragment-shader samples use the synchronized copy instead
    // of falling back to the direct texture (useCopy=0). Skip this when we

    // must not become the sampled copy of the (multisample) source object.
    if (readTextureObject &&
        readTextureObject->is_render_target &&
        readtexid &&
        !didMsaaResolve) {
        [self updateGLSampledRenderTargetCopyForTexture:readTextureObject
                                                 source:readtexid
                                                 reason:"blit_framebuffer_copy_src"];
    }
}

-(void)mtlBlitFramebuffer:(GLMContext)glm_ctx srcX0:(GLint)srcX0 srcY0:(GLint)srcY0 srcX1:(GLint)srcX1 srcY1:(GLint)srcY1 dstX0:(GLint)dstX0 dstY0:(GLint)dstY0 dstX1:(GLint)dstX1 dstY1:(GLint)dstY1 mask:(GLbitfield)mask filter:(GLenum)filter
{
    if (!glm_ctx || ((uintptr_t)glm_ctx < 0x1000)) {
        NSLog(@"MGL ERROR: mtlBlitFramebuffer called with invalid glm_ctx=%p", glm_ctx);
        return;
    }

    if (srcX1 == srcX0 || srcY1 == srcY0 || dstX1 == dstX0 || dstY1 == dstY0) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer ignored empty rect src=(%d,%d)-(%d,%d) dst=(%d,%d)-(%d,%d)",
              srcX0, srcY0, srcX1, srcY1,
              dstX0, dstY0, dstX1, dstY1);
        return;
    }

    ctx = glm_ctx;

    /* Replay pending deferred draw batches BEFORE the blit reads the source
     * attachment: draws are queued into the batch buffer and encoded only at
     * flush points (draw/FBO-switch/swap/finish).  FBO bind switches skip
     * this flush while deferFboRotation is active (batches carry their own
     * FBO snapshot), so glBlitFramebuffer right after a draw would otherwise
     * copy stale pre-draw content.  Mirrors mtlInvalidateRenderPass (flush +
     * end encoding); no-op when the batch buffer is empty. */
    [self flushDrawBuffer:glm_ctx];
    [self endRenderEncoding];

    mask = [self blitFramebufferDepthStencil:glm_ctx
                                       srcX0:srcX0 srcY0:srcY0 srcX1:srcX1 srcY1:srcY1
                                       dstX0:dstX0 dstY0:dstY0 dstX1:dstX1 dstY1:dstY1
                                         mask:mask filter:filter];

    if ((mask & GL_COLOR_BUFFER_BIT) == 0u) {
        if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u) {
            static uint64_t s_depthStencilOnlyBlitWarnCount = 0;
            uint64_t hit = ++s_depthStencilOnlyBlitWarnCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer depth/stencil-only blit is not implemented; skipping mask=0x%x hit=%llu",
                      mask,
                      (unsigned long long)hit);
            }
        }
        return;
    }

    if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u) {
        static uint64_t s_depthStencilBlitWarnCount = 0;
        uint64_t hit = ++s_depthStencilBlitWarnCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer only copies color; depth/stencil bits in mask=0x%x ignored hit=%llu",
                  mask,
                  (unsigned long long)hit);
        }
    }

    // Keep renderer ivar state consistent with the call site context.
    ctx = glm_ctx;

    MGLBlitColorState st;
    memset(&st, 0, sizeof(st));
    st.glm_ctx = glm_ctx;
    st.filter = filter;
    GLenum readAttachment = GL_NONE;
    if (![self resolveBlitFramebufferAttachments:glm_ctx
                                            srcX0:srcX0 srcY0:srcY0 srcX1:srcX1 srcY1:srcY1
                                            dstX0:dstX0 dstY0:dstY0 dstX1:dstX1 dstY1:dstY1
                                        outState:&st
                               outReadAttachment:&readAttachment]) {
        return;
    }
    Framebuffer *readfbo = st.readfbo;
    Framebuffer *drawfbo = st.drawfbo;
    FBOAttachment *readFBOAttachment = st.readFBOAttachment;
    Texture *readTextureObject = st.readTextureObject;
    FBOAttachment *drawFBOAttachment = st.drawFBOAttachment;
    Texture *drawTextureObject = st.drawTextureObject;
    MGLMetalAttachmentSubresource readSubresource = st.readSubresource;
    MGLMetalAttachmentSubresource drawSubresource = st.drawSubresource;
    id readtexid = st.readtexid;
    id drawtexid = st.drawtexid;

    // end encoding on current render encoder
    [self endRenderEncoding];

    if (![self ensureWritableCommandBuffer:"mtlBlitFramebuffer"]) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer could not obtain writable command buffer");
        return;
    }

    if (readfbo &&
        readFBOAttachment &&
        readTextureObject &&
        readtexid &&
        isColorAttachment(glm_ctx, readAttachment) &&
        (readFBOAttachment->clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        BOOL clearEncoded =
            mglRenderEncodeColorClearForCommandBufferOwner(
                _commandState.currentCommandBufferOwner,
                (__bridge void *)readtexid, readSubresource.level,
                readSubresource.slice, readSubresource.depthPlane,
                readFBOAttachment->clear_color[0],
                readFBOAttachment->clear_color[1],
                readFBOAttachment->clear_color[2],
                readFBOAttachment->clear_color[3]) == 0;
        if (clearEncoded) {
            readFBOAttachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
            mglMarkTextureLevelRenderTargetWritten(readTextureObject, readFBOAttachment->level);
            mglTraceLogNSString(@"MGL TRACE blitFramebuffer.appliedPendingReadClear fbo=%u attachment=0x%x tex=%u rgba=(%.3f,%.3f,%.3f,%.3f)",
                  (unsigned)readfbo->name,
                  (unsigned)readAttachment,
                  (unsigned)readTextureObject->name,
                  readFBOAttachment->clear_color[0],
                  readFBOAttachment->clear_color[1],
                  readFBOAttachment->clear_color[2],
                  readFBOAttachment->clear_color[3]);
        } else {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to apply pending read clear fbo=%u attachment=0x%x",
                  (unsigned)readfbo->name,
                  (unsigned)readAttachment);
        }
    }

    // Validate and clamp blit coordinates to avoid Metal validation aborts
    if (!readtexid || !drawtexid) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer missing source/destination Metal textures");
        return;
    }

    BOOL needsFormatConversionBlit = NO;
    if (mglBlitTextureInfo(readtexid).pixel_format != mglBlitTextureInfo(drawtexid).pixel_format) {
        BOOL rgbaBgraPair =
            ((mglBlitTextureInfo(readtexid).pixel_format == MGLPixelFormatRGBA8Unorm && mglBlitTextureInfo(drawtexid).pixel_format == MGLPixelFormatBGRA8Unorm) ||
             (mglBlitTextureInfo(readtexid).pixel_format == MGLPixelFormatBGRA8Unorm && mglBlitTextureInfo(drawtexid).pixel_format == MGLPixelFormatRGBA8Unorm));

        if (rgbaBgraPair) {
            needsFormatConversionBlit = YES;
            static uint64_t s_rgbaBgraBlitLogCount = 0;
            uint64_t hit = ++s_rgbaBgraBlitLogCount;
            if (hit <= 4ull || (hit % 2048ull) == 0ull) {
                NSLog(@"MGL INFO: mtlBlitFramebuffer using shader conversion for RGBA/BGRA pair (src=%lu dst=%lu hit=%llu)",
                      (unsigned long)mglBlitTextureInfo(readtexid).pixel_format,
                      (unsigned long)mglBlitTextureInfo(drawtexid).pixel_format,
                      (unsigned long long)hit);
            }
        } else {
            NSLog(@"MGL WARN: mtlBlitFramebuffer pixel format mismatch (src=%lu dst=%lu), skipping blit",
                  (unsigned long)mglBlitTextureInfo(readtexid).pixel_format, (unsigned long)mglBlitTextureInfo(drawtexid).pixel_format);
            return;
        }
    }

    // When the source texture is a render target and its sampled copy isn't
    // current, force the blit through the render-pass (scaled) path to ensure
    // proper Metal synchronization.  On tile-based Apple GPUs a
    // MTLBlitCommandEncoder may read stale tile memory if the render target
    // was recently written by a preceding render pass, leading to intermittent
    // GUI icon / entity rendering errors.
    BOOL needsRenderTargetSyncBlit = NO;
    if (readTextureObject &&
        readTextureObject->is_render_target &&
        readTextureObject->mtl_render_target_write_version > 0u) {
        if (readTextureObject->mtl_gl_sampled_write_version !=
            readTextureObject->mtl_render_target_write_version) {
            needsRenderTargetSyncBlit = YES;
            static uint64_t s_rtSyncBlitLogCount = 0;
            uint64_t hit = ++s_rtSyncBlitLogCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL RT-SYNC-BLIT read-tex=%u rtVer=%u sampledVer=%u size=%lux%lu hit=%llu",
                      (unsigned)readTextureObject->name,
                      (unsigned)readTextureObject->mtl_render_target_write_version,
                      (unsigned)readTextureObject->mtl_gl_sampled_write_version,
                      (unsigned long)mglBlitTextureInfo(readtexid).width,
                      (unsigned long)mglBlitTextureInfo(readtexid).height,
                      (unsigned long long)hit);
            }
        }
    }

    if (readSubresource.level >= mglBlitTextureInfo(readtexid).mipmap_level_count ||
        drawSubresource.level >= mglBlitTextureInfo(drawtexid).mipmap_level_count) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer invalid mip level read=%lu/%lu draw=%lu/%lu, skipping",
              (unsigned long)readSubresource.level,
              (unsigned long)mglBlitTextureInfo(readtexid).mipmap_level_count,
              (unsigned long)drawSubresource.level,
              (unsigned long)mglBlitTextureInfo(drawtexid).mipmap_level_count);
        return;
    }

    NSUInteger srcTexW = mglMetalTextureLevelDimension(mglBlitTextureInfo(readtexid).width, readSubresource.level);
    NSUInteger srcTexH = mglMetalTextureLevelDimension(mglBlitTextureInfo(readtexid).height, readSubresource.level);
    NSUInteger dstTexW = mglMetalTextureLevelDimension(mglBlitTextureInfo(drawtexid).width, drawSubresource.level);
    NSUInteger dstTexH = mglMetalTextureLevelDimension(mglBlitTextureInfo(drawtexid).height, drawSubresource.level);


    BOOL didMsaaResolve = NO;
    if (![self blitFramebufferResolveMsaaSource:&readtexid
                                        drawtexid:drawtexid
                                readSubresource:&readSubresource
                                          srcTexW:srcTexW srcTexH:srcTexH
                               readTextureObject:readTextureObject
                               outDidMsaaResolve:&didMsaaResolve]) {
        return;
    }

    MGLBlitAxis axisX = { (double)srcX0, (double)srcX1, (double)dstX0, (double)dstX1 };
    MGLBlitAxis axisY = { (double)srcY0, (double)srcY1, (double)dstY0, (double)dstY1 };
    if (!mglClipBlitAxis(&axisX, (double)srcTexW, (double)dstTexW) ||
        !mglClipBlitAxis(&axisY, (double)srcTexH, (double)dstTexH)) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer clipped region is empty srcTex=%lux%lu dstTex=%lux%lu req src=(%d,%d)-(%d,%d) dst=(%d,%d)-(%d,%d)",
              (unsigned long)srcTexW,
              (unsigned long)srcTexH,
              (unsigned long)dstTexW,
              (unsigned long)dstTexH,
              srcX0, srcY0, srcX1, srcY1,
              dstX0, dstY0, dstX1, dstY1);
        return;
    }


    MGLRenderBlitFramebufferPlan plan = {0};
    if (mglRenderBlitFramebufferPlan(
            axisX.src0, axisX.src1, axisY.src0, axisY.src1,
            axisX.dst0, axisX.dst1, axisY.dst0, axisY.dst1,
            (uint32_t)srcTexW, (uint32_t)srcTexH,
            (uint32_t)dstTexW, (uint32_t)dstTexH,
            needsFormatConversionBlit ? 1 : 0,
            needsRenderTargetSyncBlit ? 1 : 0,
            (glm_ctx && glm_ctx->state.caps.scissor_test) ? 1 : 0,
            &plan) != 0) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer empty clipped region src=%.3fx%.3f dst=%.3fx%.3f, skipping",
              fabs(axisX.src1 - axisX.src0),
              fabs(axisY.src1 - axisY.src0),
              fabs(axisX.dst1 - axisX.dst0),
              fabs(axisY.dst1 - axisY.dst0));
        return;
    }
    BOOL srcXForward = plan.src_x_forward;
    BOOL srcYForward = plan.src_y_forward;
    BOOL dstXForward = plan.dst_x_forward;
    BOOL dstYForward = plan.dst_y_forward;
    BOOL blitNeedsFlip = plan.blit_needs_flip;
    double srcMinX = plan.src_min_x;
    double srcMaxX = plan.src_max_x;
    double srcMinY = plan.src_min_y;
    double srcMaxY = plan.src_max_y;
    double dstMinX = plan.dst_min_x;
    double dstMaxX = plan.dst_max_x;
    double dstMinY = plan.dst_min_y;
    double dstMaxY = plan.dst_max_y;
    double srcW = plan.src_w;
    double srcH = plan.src_h;
    double dstW = plan.dst_w;
    double dstH = plan.dst_h;
    BOOL needsScaledBlit = plan.needs_scaled_blit;
    NSInteger copySrcX = (NSInteger)plan.copy_src_x;
    NSInteger copySrcY = (NSInteger)plan.copy_src_y;
    NSInteger copyDstX = (NSInteger)plan.copy_dst_x;
    NSInteger copyDstY = (NSInteger)plan.copy_dst_y;
    NSInteger copyW = (NSInteger)plan.copy_w;
    NSInteger copyH = (NSInteger)plan.copy_h;
    NSInteger srcMetalY = (NSInteger)plan.src_metal_y;
    NSInteger dstMetalY = (NSInteger)plan.dst_metal_y;
    double scaledDstMetalY = plan.scaled_dst_metal_y;

    static uint64_t s_blitDiagCount = 0;
    uint64_t blitDiag = ++s_blitDiagCount;
    BOOL traceBlitToFile = mglTraceLogIsEnabled() && mglEnvFlagEnabled("MGL_TRACE_BLIT");
    BOOL traceBlit = (kMGLSwapPresentDiagnostics || traceBlitToFile) &&
        (blitDiag <= 24ull || (blitDiag % 120ull) == 0ull || needsScaledBlit);
    if (traceBlit) {
        const char *fmt =
            "MGL TRACE blitFramebuffer call=%llu readFBO=%p drawFBO=%p mask=0x%x filter=0x%x "
            "srcReq=(%d,%d)-(%d,%d) dstReq=(%d,%d)-(%d,%d) "
            "copy srcGL=(%.3f,%.3f %.3fx%.3f) dstGL=(%.3f,%.3f %.3fx%.3f) srcMTL=(%ld,%ld) dstMTL=(%ld,%ld) scaled=%d flip=%d "
            "srcObj=%u dstObj=%u srcRT=%d dstRT=%d srcAuth=0x%x dstAuth=0x%x srcRtVer=%u dstRtVer=%u srcCopyVer=%u dstCopyVer=%u "
            "srcTex=%p fmt=%lu %lux%lu dstTex=%p fmt=%lu %lux%lu drawBuf=0x%x readBuf=0x%x";
        if (traceBlitToFile) {
            mglTraceLog(fmt,
                        (unsigned long long)blitDiag,
                        readfbo,
                        drawfbo,
                        mask,
                        (unsigned)filter,
                        srcX0, srcY0, srcX1, srcY1,
                        dstX0, dstY0, dstX1, dstY1,
                        srcMinX, srcMinY, srcW, srcH,
                        dstMinX, dstMinY, dstW, dstH,
                        (long)copySrcX, (long)srcMetalY,
                        (long)copyDstX, (long)dstMetalY,
                        needsScaledBlit ? 1 : 0,
                        blitNeedsFlip ? 1 : 0,
                        readTextureObject ? (unsigned)readTextureObject->name : 0u,
                        drawTextureObject ? (unsigned)drawTextureObject->name : 0u,
                        (readTextureObject && readTextureObject->is_render_target) ? 1 : 0,
                        (drawTextureObject && drawTextureObject->is_render_target) ? 1 : 0,
                        readTextureObject ? (unsigned)readTextureObject->mtl_render_yflip_authority : 0u,
                        drawTextureObject ? (unsigned)drawTextureObject->mtl_render_yflip_authority : 0u,
                        readTextureObject ? (unsigned)readTextureObject->mtl_render_target_write_version : 0u,
                        drawTextureObject ? (unsigned)drawTextureObject->mtl_render_target_write_version : 0u,
                        readTextureObject ? (unsigned)readTextureObject->mtl_gl_sampled_write_version : 0u,
                        drawTextureObject ? (unsigned)drawTextureObject->mtl_gl_sampled_write_version : 0u,
                        readtexid,
                        (unsigned long)mglBlitTextureInfo(readtexid).pixel_format,
                        (unsigned long)srcTexW,
                        (unsigned long)srcTexH,
                        drawtexid,
                        (unsigned long)mglBlitTextureInfo(drawtexid).pixel_format,
                        (unsigned long)dstTexW,
                        (unsigned long)dstTexH,
                        (unsigned)(glm_ctx ? glm_ctx->state.draw_buffer : 0u),
                        (unsigned)(glm_ctx ? glm_ctx->state.read_buffer : 0u));
        } else {
            mglTraceLogNSString(@"MGL TRACE blitFramebuffer call=%llu readFBO=%p drawFBO=%p mask=0x%x filter=0x%x "
                  "srcReq=(%d,%d)-(%d,%d) dstReq=(%d,%d)-(%d,%d) "
                  "copy srcGL=(%.3f,%.3f %.3fx%.3f) dstGL=(%.3f,%.3f %.3fx%.3f) srcMTL=(%ld,%ld) dstMTL=(%ld,%ld) scaled=%d flip=%d "
                  "srcObj=%u dstObj=%u srcRT=%d dstRT=%d srcAuth=0x%x dstAuth=0x%x srcRtVer=%u dstRtVer=%u srcCopyVer=%u dstCopyVer=%u "
                  "srcTex=%p fmt=%lu %lux%lu dstTex=%p fmt=%lu %lux%lu drawBuf=0x%x readBuf=0x%x",
                  (unsigned long long)blitDiag,
                  readfbo,
                  drawfbo,
                  mask,
                  (unsigned)filter,
                  srcX0, srcY0, srcX1, srcY1,
                  dstX0, dstY0, dstX1, dstY1,
                  srcMinX, srcMinY, srcW, srcH,
                  dstMinX, dstMinY, dstW, dstH,
                  (long)copySrcX, (long)srcMetalY,
                  (long)copyDstX, (long)dstMetalY,
                  needsScaledBlit ? 1 : 0,
                  blitNeedsFlip ? 1 : 0,
                  readTextureObject ? (unsigned)readTextureObject->name : 0u,
                  drawTextureObject ? (unsigned)drawTextureObject->name : 0u,
                  (readTextureObject && readTextureObject->is_render_target) ? 1 : 0,
                  (drawTextureObject && drawTextureObject->is_render_target) ? 1 : 0,
                  readTextureObject ? (unsigned)readTextureObject->mtl_render_yflip_authority : 0u,
                  drawTextureObject ? (unsigned)drawTextureObject->mtl_render_yflip_authority : 0u,
                  readTextureObject ? (unsigned)readTextureObject->mtl_render_target_write_version : 0u,
                  drawTextureObject ? (unsigned)drawTextureObject->mtl_render_target_write_version : 0u,
                  readTextureObject ? (unsigned)readTextureObject->mtl_gl_sampled_write_version : 0u,
                  drawTextureObject ? (unsigned)drawTextureObject->mtl_gl_sampled_write_version : 0u,
                  readtexid,
                  (unsigned long)mglBlitTextureInfo(readtexid).pixel_format,
                  (unsigned long)srcTexW,
                  (unsigned long)srcTexH,
                  drawtexid,
                  (unsigned long)mglBlitTextureInfo(drawtexid).pixel_format,
                  (unsigned long)dstTexW,
                  (unsigned long)dstTexH,
                  (unsigned)(glm_ctx ? glm_ctx->state.draw_buffer : 0u),
                  (unsigned)(glm_ctx ? glm_ctx->state.read_buffer : 0u));
        }
    }

    /* Fill shared state for color blit helpers. */
    st.glm_ctx = glm_ctx;
    st.readfbo = readfbo;
    st.drawfbo = drawfbo;
    st.filter = filter;
    st.readFBOAttachment = readFBOAttachment;
    st.drawFBOAttachment = drawFBOAttachment;
    st.readTextureObject = readTextureObject;
    st.drawTextureObject = drawTextureObject;
    st.readSubresource = readSubresource;
    st.drawSubresource = drawSubresource;
    st.readtexid = readtexid;
    st.drawtexid = drawtexid;
    st.srcTexW = srcTexW;
    st.srcTexH = srcTexH;
    st.dstTexW = dstTexW;
    st.dstTexH = dstTexH;
    st.needsFormatConversionBlit = needsFormatConversionBlit;
    st.needsRenderTargetSyncBlit = needsRenderTargetSyncBlit;
    st.didMsaaResolve = didMsaaResolve;
    st.blitNeedsFlip = blitNeedsFlip;
    st.needsScaledBlit = needsScaledBlit;
    st.srcXForward = srcXForward;
    st.srcYForward = srcYForward;
    st.dstXForward = dstXForward;
    st.dstYForward = dstYForward;
    st.srcMinX = srcMinX;
    st.srcMaxX = srcMaxX;
    st.srcMinY = srcMinY;
    st.srcMaxY = srcMaxY;
    st.dstMinX = dstMinX;
    st.dstMaxX = dstMaxX;
    st.dstMinY = dstMinY;
    st.dstMaxY = dstMaxY;
    st.srcW = srcW;
    st.srcH = srcH;
    st.dstW = dstW;
    st.dstH = dstH;
    st.copySrcX = copySrcX;
    st.copySrcY = copySrcY;
    st.copyDstX = copyDstX;
    st.copyDstY = copyDstY;
    st.copyW = copyW;
    st.copyH = copyH;
    st.srcMetalY = srcMetalY;
    st.dstMetalY = dstMetalY;
    st.scaledDstMetalY = scaledDstMetalY;

    if ([self blitFramebufferIntegerColorWithState:&st]) {
        return;
    }

    if ([self blitFramebufferScaledColorWithState:&st]) {
        return;
    }

    [self blitFramebufferDirectColorCopyWithState:&st];
}

void mglRendererCompatBlitFramebuffer(GLMContext glm_ctx,
                                      int src_x0,
                                      int src_y0,
                                      int src_x1,
                                      int src_y1,
                                      int dst_x0,
                                      int dst_y0,
                                      int dst_x1,
                                      int dst_y1,
                                      unsigned int mask,
                                      unsigned int filter)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlBlitFramebuffer:glm_ctx
                           srcX0:src_x0 srcY0:src_y0
                           srcX1:src_x1 srcY1:src_y1
                           dstX0:dst_x0 dstY0:dst_y0
                           dstX1:dst_x1 dstY1:dst_y1
                            mask:mask filter:filter];
}


/* Texture-to-texture blit path for glCopyTexImage2D / glCopyTexSubImage when
 * the destination texture uses a non-BGRA8-compatible Metal pixel format
 * (depth, integer, packed). Resolves the matching framebuffer attachment
 * (depth attachment for depth-format destinations, color attachment for
 * color-format destinations) and performs a direct GPU blit when the source
 * and destination Metal pixel formats match. Returns YES if the blit
 * succeeded and the caller should return; NO to fall through to the CPU
 * BGRA8 conversion path. */
-(BOOL)mtlCopyTexSubImageViaTextureBlit:(GLMContext)glm_ctx
                                    tex:(Texture *)tex
                           destTexture:(id)destTexture
                                  slice:(NSUInteger)slice
                                 level:(NSUInteger)level
                               xoffset:(NSInteger)xoffset
                               yoffset:(NSInteger)yoffset
                                    x:(NSInteger)x
                                    y:(NSInteger)y
                                width:(NSUInteger)width
                               height:(NSUInteger)height
{
    if (!glm_ctx || !tex || !destTexture || width == 0u || height == 0u) {
        return NO;
    }

    uint32_t destFormat = mglBlitTextureInfo(destTexture).pixel_format;
    BOOL destIsDepth = mglMetalPixelFormatIsDepthOrStencil(destFormat);

    /* Resolve the source framebuffer attachment. For depth destinations we
     * read from the depth attachment; for color destinations we read from
     * the current read buffer's color attachment. */
    Framebuffer *fbo = glm_ctx->state.readbuffer;
    if (!fbo) {
        /* Default framebuffer: not supported via this path. */
        return NO;
    }

    FBOAttachment *srcAttachment = NULL;
    if (destIsDepth) {
        srcAttachment = &fbo->depth;
    } else {
        GLenum readBuffer = glm_ctx->state.read_buffer;
        if (readBuffer < GL_COLOR_ATTACHMENT0 ||
            readBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
            return NO;
        }
        GLuint attachmentIndex = (GLuint)(readBuffer - GL_COLOR_ATTACHMENT0);
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            return NO;
        }
        srcAttachment = &fbo->color_attachments[attachmentIndex];
    }

    Texture *srcTexObj = [self framebufferAttachmentTexture:srcAttachment];
    if (!srcTexObj) {
        return NO;
    }
    srcTexObj->is_render_target = true;
    if (![self bindMTLTexture:srcTexObj] || !srcTexObj->mtl_data) {
        return NO;
    }
    id srcTexture = (__bridge id)(srcTexObj->mtl_data);
    if (!srcTexture) {
        return NO;
    }

    /* Only blit when source and destination Metal pixel formats match. */
    if (mglBlitTextureInfo(srcTexture).pixel_format != destFormat) {
        return NO;
    }

    if (mglRenderTextureIsFramebufferOnly(srcTexObj->mtl_data)) {
        return NO;
    }

    if (level >= mglBlitTextureInfo(destTexture).mipmap_level_count) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return YES; /* Consumed the call; report an error. */
    }

    NSUInteger destLevelWidth = mglMetalTextureLevelDimension(mglBlitTextureInfo(destTexture).width, level);
    NSUInteger destLevelHeight = mglMetalTextureLevelDimension(mglBlitTextureInfo(destTexture).height, level);
    if ((NSUInteger)xoffset > destLevelWidth ||
        (NSUInteger)yoffset > destLevelHeight ||
        width > destLevelWidth - (NSUInteger)xoffset ||
        height > destLevelHeight - (NSUInteger)yoffset) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return YES;
    }

    MGLMetalAttachmentSubresource srcSubresource =
        mglMetalAttachmentSubresourceForAttachment(srcAttachment);

    /* Metal's texture coordinate origin is top-left, GL's is bottom-left.
     * Flip the source Y so the copied region matches GL semantics. */
    NSUInteger srcLevelHeight = mglMetalTextureLevelDimension(mglBlitTextureInfo(srcTexture).height, srcSubresource.level);
    NSInteger srcY = (NSInteger)srcLevelHeight - ((NSInteger)y + (NSInteger)height);
    if (srcY < 0) {
        srcY = 0;
    }

    /* End any active render encoder so the blit encoder can run. */
    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlCopyTexSubImageViaTextureBlit"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return YES;
    }

    /* Apply any pending FBO clear so the source texture has authoritative
     * data before the blit reads from it. */
    if (destIsDepth) {
        [self mglApplyPendingFBODepthClearForReadback:fbo
                                            attachment:srcAttachment
                                            textureObj:srcTexObj
                                            mtlTexture:srcTexture];
    } else {
        GLenum readBuffer = glm_ctx->state.read_buffer;
        [self mglApplyPendingFBOColorClearForReadback:fbo
                                            attachment:srcAttachment
                                            textureObj:srcTexObj
                                            mtlTexture:srcTexture
                                       attachmentEnum:readBuffer];
    }

    id blitEncoder =
        (__bridge id)mglRenderCreateBlitEncoderBorrowed(
            _commandState.currentCommandBufferOwner);
    if (!blitEncoder) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return YES;
    }

    BOOL blitEnded = NO;
    @try {
        mglBlitCopyTexture(
            blitEncoder, srcTexture, srcSubresource.slice,
            srcSubresource.level,
            mglBlitOrigin((NSUInteger)x, (NSUInteger)srcY, 0u),
            mglBlitSize(width, height, 1u), destTexture, slice, level,
            mglBlitOrigin((NSUInteger)xoffset, (NSUInteger)yoffset, 0u));
        mglBlitEndBlitEncoder(blitEncoder);
        blitEnded = YES;
    } @catch (NSException *exception) {
        if (!blitEnded) {
            @try { mglBlitEndBlitEncoder(blitEncoder); } @catch (NSException *endException) { }
        }
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return YES;
    }

    mglMarkTextureLevelMetalFilled(tex, (GLuint)level, 0);
    [self updateGLSampledRenderTargetCopyForTexture:tex
                                             source:destTexture
                                             reason:"copy_tex_sub_image_blit"];
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    mglMarkRendererDirtyBits(&glm_ctx->state, DIRTY_TEX | DIRTY_TEX_BINDING);
    return YES;
}

-(void)mtlCopyTexSubImage:(GLMContext)glm_ctx
                      tex:(Texture *)tex
                    slice:(NSUInteger)slice
            mipmapLevel:(NSUInteger)level
                  xoffset:(NSInteger)xoffset
                  yoffset:(NSInteger)yoffset
                        x:(NSInteger)x
                        y:(NSInteger)y
                    width:(NSUInteger)width
                   height:(NSUInteger)height
{
    ctx = glm_ctx;

    if (!tex || width == 0u || height == 0u) {
        return;
    }
    if ((NSInteger)level < 0 || xoffset < 0 || yoffset < 0) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }

    /* Bind the destination texture so we can inspect its Metal pixel format. */
    if (!tex->mtl_data && ![self bindMTLTexture:tex]) {
        NSLog(@"MGL ERROR: mtlCopyTexSubImage failed to bind destination texture %u", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    id destTexture = tex->mtl_data ? (__bridge id)(tex->mtl_data) : nil;
    if (!destTexture) {
        NSLog(@"MGL ERROR: mtlCopyTexSubImage destination texture %u has no Metal texture", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    /* Fast path: try a direct GPU texture-to-texture blit from the matching
     * framebuffer attachment. This handles glCopyTexImage2D/glCopyTexSubImage
     * for depth, integer, and packed internal formats where the source FBO
     * attachment shares the same Metal pixel format as the destination
     * texture. For BGRA8/RGBA8 destinations the CPU path below is sufficient,
     * so we skip the blit attempt to avoid unnecessary encoder churn. */
    BOOL destIsPlainBGRA8 =
        (mglBlitTextureInfo(destTexture).pixel_format == MGLPixelFormatBGRA8Unorm ||
         mglBlitTextureInfo(destTexture).pixel_format == MGLPixelFormatBGRA8Unorm_sRGB ||
         mglBlitTextureInfo(destTexture).pixel_format == MGLPixelFormatRGBA8Unorm ||
         mglBlitTextureInfo(destTexture).pixel_format == MGLPixelFormatRGBA8Unorm_sRGB);
    if (!destIsPlainBGRA8) {
        BOOL blitted = [self mtlCopyTexSubImageViaTextureBlit:glm_ctx
                                                          tex:tex
                                                  destTexture:destTexture
                                                        slice:slice
                                                          level:level
                                                       xoffset:xoffset
                                                       yoffset:yoffset
                                                            x:x
                                                            y:y
                                                        width:width
                                                       height:height];
        if (blitted) {
            return;
        }
        /* Fall through to the BGRA8 path if the blit was not applicable. */
    }

    if (width > (NSUInteger)(SIZE_MAX / 4u)) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }
    size_t bgraRowBytes = (size_t)width * 4u;
    if (height > 0u && bgraRowBytes > SIZE_MAX / (size_t)height) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }
    size_t bgraSize = bgraRowBytes * (size_t)height;

    NSMutableData *bgraReadback = [NSMutableData dataWithLength:bgraSize];
    NSMutableData *uploadData = [NSMutableData dataWithLength:bgraSize];
    if (!bgraReadback || !uploadData) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }

    /*
     * Reuse the readPixels read-buffer resolver so default/FBO, clipping, clears,
     * and GL bottom-left row order all follow the same path as glReadPixels.
     */
    [self mtlReadDrawable:glm_ctx
               pixelBytes:bgraReadback.mutableBytes
              bytesPerRow:bgraRowBytes
            bytesPerImage:bgraSize
               fromRegion:mglBlitRegion2D(x, y, width, height)];

    id texture = destTexture;
    if (!mglMetalReadbackFormatIsBGRA8Compatible(mglBlitTextureInfo(texture).pixel_format)) {
        NSLog(@"MGL ERROR: mtlCopyTexSubImage unsupported destination Metal format=%lu texture=%u",
              (unsigned long)mglBlitTextureInfo(texture).pixel_format,
              tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    if (level >= mglBlitTextureInfo(texture).mipmap_level_count) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }

    NSUInteger levelWidth = mglMetalTextureLevelDimension(mglBlitTextureInfo(texture).width, level);
    NSUInteger levelHeight = mglMetalTextureLevelDimension(mglBlitTextureInfo(texture).height, level);
    NSUInteger levelDepth = mglMetalTextureLevelDimension(mglBlitTextureInfo(texture).depth, level);
    if ((NSUInteger)xoffset > levelWidth ||
        (NSUInteger)yoffset > levelHeight ||
        width > levelWidth - (NSUInteger)xoffset ||
        height > levelHeight - (NSUInteger)yoffset) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }

    uint32_t textureType = mglBlitTextureInfo(texture).texture_type;
    NSUInteger destinationSlice = slice;
    NSUInteger copyDepth = 1u;
    MGLOriginValue destinationOrigin = mglBlitOrigin((NSUInteger)xoffset, 0u, 0u);
    if (textureType == MGLTextureType3D) {
        if (slice >= levelDepth) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
            return;
        }
        destinationSlice = 0u;
        destinationOrigin = mglBlitOrigin((NSUInteger)xoffset, 0u, slice);
    } else {
        NSUInteger maxDestinationSlices = mglBlitTextureInfo(texture).array_length;
        if (textureType == MGLTextureTypeCube) {
            maxDestinationSlices = 6u;
        } else if (textureType == MGLTextureTypeCubeArray) {
            maxDestinationSlices = mglBlitTextureInfo(texture).array_length * 6u;
        }
        if (destinationSlice >= maxDestinationSlices) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
            return;
        }
    }

    BOOL destinationIsRenderTarget = tex->is_render_target ? YES : NO;
    NSUInteger destinationY = (NSUInteger)yoffset;
    if (destinationIsRenderTarget) {
        destinationY = levelHeight - ((NSUInteger)yoffset + height);
    }
    destinationOrigin.y = destinationY;

    if (!mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes((const uint8_t *)bgraReadback.bytes,
                                                              bgraRowBytes,
                                                              (uint8_t *)uploadData.mutableBytes,
                                                              bgraRowBytes,
                                                              width,
                                                              height,
                                                              mglBlitTextureInfo(texture).pixel_format,
                                                              destinationIsRenderTarget)) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    id uploadBuffer = mglBlitCreateBufferWithBytes(
        _device, uploadData.bytes, bgraSize, MGLResourceStorageModeShared);
    if (!uploadBuffer) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }

    bool uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:uploadBuffer
                                                         sourceOffset:0u
                                                    sourceBytesPerRow:bgraRowBytes
                                                  sourceBytesPerImage:bgraSize
                                                   sourceLayerStride:0u
                                                           layerCount:1u
                                                            sourceSize:mglBlitSize(width, height, copyDepth)
                                                             toTexture:texture
                                                      destinationSlice:destinationSlice
                                                      destinationLevel:level
                                                     destinationOrigin:destinationOrigin
                                                                reason:"copy_tex_sub_image"];
    if (!uploaded) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    mglMarkTextureLevelMetalFilled(tex, (GLuint)level, bgraSize);
    [self updateGLSampledRenderTargetCopyForTexture:tex
                                             source:texture
                                             reason:"copy_tex_sub_image"];
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    if (glm_ctx) {
        mglMarkRendererDirtyBits(&glm_ctx->state,
                                 DIRTY_TEX | DIRTY_TEX_BINDING);
    }
}

#pragma mark C interface to mtlCopyImageSubData

- (BOOL)readTextureRegionViaBlit:(id)texture
                          region:(MGLRegionValue)region
                           slice:(NSUInteger)slice
                           level:(NSUInteger)level
                           bytes:(void *)bytes
                     bytesPerRow:(NSUInteger)bytesPerRow
                   bytesPerImage:(NSUInteger)bytesPerImage
                          reason:(const char *)reason
{
    NSUInteger depth = MAX(region.size.depth, 1u);
    if (!texture || !bytes || bytesPerRow == 0 || bytesPerImage == 0 ||
        depth > NSUIntegerMax / bytesPerImage) {
        return NO;
    }

    NSUInteger totalBytes = bytesPerImage * depth;
    id stagingBuffer = mglBlitCreateBuffer(
        _device, totalBytes, MGLResourceStorageModeShared);
    if (!stagingBuffer) {
        return NO;
    }

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:reason ? reason : "texture_readback_blit"]) {
        return NO;
    }

    id readEncoder =
        (__bridge id)mglRenderCreateBlitEncoderBorrowed(
            _commandState.currentCommandBufferOwner);
    if (!readEncoder) {
        return NO;
    }
    /* A blit encoder is now active on the current CB.  Mark it as having
     * work so flushCommandBuffer:YES below does not skip the commit. */
    _currentCBHasWork = YES;

    @try {
        mglBlitCopyTextureToBuffer(readEncoder, texture, slice, level,
                                   region.origin, region.size, stagingBuffer,
                                   0, bytesPerRow, bytesPerImage);
        mglBlitEndBlitEncoder(readEncoder);
    } @catch (NSException *exception) {
        @try {
            mglBlitEndBlitEncoder(readEncoder);
        } @catch (__unused NSException *endException) {
        }
        NSLog(@"MGL WARNING: texture readback blit failed (%s): %@",
              reason ? reason : "texture_readback_blit", exception.reason);
        return NO;
    }

    [self flushCommandBuffer:YES];
    MGLRenderCommandBufferState readState = {0};
    if (mglCmdWaitForLastSubmittedCommandBuffer(&_commandState, &readState) != 0 ||
        readState.has_error) {
        return NO;
    }
    void *stagingContents = NULL;
    uint64_t stagingLength = 0;
    if (mglRenderGetBufferContents((__bridge void *)stagingBuffer,
                                      &stagingContents,
                                      &stagingLength) != 0 ||
        !stagingContents || stagingLength < totalBytes) {
        return NO;
    }
    memcpy(bytes, stagingContents, totalBytes);
    return YES;
}

/* CPU-to-CPU copy path for mtlCopyImageSubData.
 * Raw memcpy between matching-format textures that both have CPU data.
 * Returns YES if the copy succeeded (caller should return). */
- (BOOL)copyImageSubDataCpuToCpu:(GLMContext)glm_ctx
                          srcTex:(Texture *)srcTex
                      srcTexture:(id)srcTexture
                         srcType:(uint32_t)srcType
                        srcLevel:(GLint)srcLevel
                            srcX:(GLint)srcX srcY:(GLint)srcY srcZ:(GLint)srcZ
                          dstTex:(Texture *)dstTex
                      dstTexture:(id)dstTexture
                         dstType:(uint32_t)dstType
                        dstLevel:(GLint)dstLevel
                            dstX:(GLint)dstX dstY:(GLint)dstY dstZ:(GLint)dstZ
                           width:(GLsizei)width height:(GLsizei)height depth:(GLsizei)depth
{

    if (!srcTex->metal_data_authoritative && !srcTex->is_render_target &&
        srcTex->faces && dstTex->faces &&
        (NSUInteger)srcLevel < srcTex->num_levels &&
        (NSUInteger)dstLevel < dstTex->num_levels) {

            GLuint srcPixelSize = sizeForInternalFormat(srcTex->internalformat, 0, 0);
            GLuint dstPixelSize = sizeForInternalFormat(dstTex->internalformat, 0, 0);
            if (srcPixelSize > 0 && srcPixelSize == dstPixelSize &&
                mglBlitTextureInfo(srcTexture).pixel_format == mglBlitTextureInfo(dstTexture).pixel_format) {
                NSUInteger copyWidth = MAX((NSUInteger)width, 1u);
                NSUInteger copyHeight = MAX((NSUInteger)height, 1u);
                NSUInteger rowBytes = copyWidth * srcPixelSize;
                NSUInteger numSlices = MAX((NSUInteger)depth, 1u);

                bool cpuCopyOK = true;
                for (NSUInteger s = 0; s < numSlices && cpuCopyOK; s++) {
                    /* Determine src face/level */
                    GLuint srcFace = 0;
                    if (srcType == MGLTextureTypeCube || srcType == MGLTextureTypeCubeArray) {
                        srcFace = ((GLuint)srcZ + s) % 6;
                    }
                    TextureLevel *srcLvl = (srcFace < 6 && srcTex->faces[srcFace].levels) ?
                        &srcTex->faces[srcFace].levels[srcLevel] : NULL;

                    /* Determine dst face/level */
                    GLuint dstFace = 0;
                    if (dstType == MGLTextureTypeCube || dstType == MGLTextureTypeCubeArray) {
                        dstFace = ((GLuint)dstZ + s) % 6;
                    }
                    TextureLevel *dstLvl = (dstFace < 6 && dstTex->faces[dstFace].levels) ?
                        &dstTex->faces[dstFace].levels[dstLevel] : NULL;

                    if (!srcLvl || !dstLvl || !srcLvl->data || !dstLvl->data ||
                        srcLvl->width <= 0 || dstLvl->width <= 0) {
                        cpuCopyOK = false;
                        break;
                    }

                    /* For 3D and 2D-array textures, slices are depth planes
                     * within one level.  For cube textures, each slice is a
                     * separate face.  For 2D/rectangle, there is only one
                     * slice. */
                    size_t srcSlicePitch = srcLvl->pitch * MAX(srcLvl->height, 1u);
                    size_t dstSlicePitch = dstLvl->pitch * MAX(dstLvl->height, 1u);
                    bool srcSliced = (srcType == MGLTextureType3D ||
                                      srcType == MGLTextureType2DArray);
                    bool dstSliced = (dstType == MGLTextureType3D ||
                                      dstType == MGLTextureType2DArray);
                    size_t srcSliceOff = srcSliced ?
                        ((NSUInteger)srcZ + s) * srcSlicePitch : 0;
                    size_t dstSliceOff = dstSliced ?
                        ((NSUInteger)dstZ + s) * dstSlicePitch : 0;

                    /* Copy region row by row */
                    for (NSUInteger y = 0; y < copyHeight; y++) {
                        size_t srcOff = srcSliceOff +
                                        ((NSUInteger)srcY + y) * srcLvl->pitch +
                                        (NSUInteger)srcX * srcPixelSize;
                        size_t dstOff = dstSliceOff +
                                        ((NSUInteger)dstY + y) * dstLvl->pitch +
                                        (NSUInteger)dstX * dstPixelSize;
                        if (srcOff + rowBytes > srcLvl->data_size ||
                            dstOff + rowBytes > dstLvl->data_size) {
                            cpuCopyOK = false;
                            break;
                        }
                        memcpy((uint8_t *)(uintptr_t)dstLvl->data + dstOff,
                               (const uint8_t *)(uintptr_t)srcLvl->data + srcOff,
                               rowBytes);
                    }


                    if (cpuCopyOK) {
                        NSUInteger mtlSlice = 0;
                        MGLRegionValue region;
                        if (dstType == MGLTextureType3D) {
                            mtlSlice = 0;
                            region = mglBlitRegion3D((NSUInteger)dstX,
                                                      (NSUInteger)dstY,
                                                      (NSUInteger)dstZ + s,
                                                      copyWidth, copyHeight, 1);
                        } else {
                            mtlSlice = (dstType == MGLTextureTypeCube ||
                                        dstType == MGLTextureTypeCubeArray) ?
                                dstFace : ((NSUInteger)dstZ + s);
                            region = mglBlitRegion2D((NSUInteger)dstX,
                                                      (NSUInteger)dstY,
                                                      copyWidth, copyHeight);
                        }
                        if (mglBlitTextureInfo(dstTexture).storage_mode != MGLStorageModePrivate) {
                            /* For CPU-backed RGB8-family / RGB16 / RGB32 family destinations,
                             * CPU bpp (3/6/12) != Metal bpp (4/8/16).  The CPU memcpy
                             * above preserved the CPU layout, so expand the copied
                             * region to Metal texel layout before replaceRegion,
                             * otherwise N-byte rows are uploaded to a 4/8/16-byte
                             * Metal texture (pixel shift / stripes).  Mirrors the
                             * private-storage sibling below. */
                            NSUInteger dstMetalBpp = mglMetalReadbackBytesPerPixel(mglBlitTextureInfo(dstTexture).pixel_format);
                            size_t dstCpuBpp = (dstLvl->width > 0) ?
                                (dstLvl->pitch / dstLvl->width) : 0;

                            const void *upSrcPtr = (const uint8_t *)(uintptr_t)dstLvl->data + dstSliceOff;
                            NSUInteger upBytesPerRow = dstLvl->pitch;
                            NSUInteger upBytesPerImage = dstSlicePitch;
                            void *expandedData = NULL;
                            if (dstMetalBpp > 0 && dstCpuBpp != dstMetalBpp) {
                                if (mglTextureInternalFormatNeedsRGBA8Expansion(
                                        dstTex->internalformat, mglBlitTextureInfo(dstTexture).pixel_format)) {
                                    NSUInteger expandedBPR = 0, expandedBPI = 0;
                                    expandedData = mglCreateRGBA8ExpandedUpload(
                                        dstTex, (const uint8_t *)upSrcPtr,
                                        copyWidth, copyHeight, upBytesPerRow,
                                        &expandedBPR, &expandedBPI);
                                    if (expandedData) {
                                        upSrcPtr = expandedData;
                                        upBytesPerRow = expandedBPR;
                                        upBytesPerImage = expandedBPI;
                                    }
                                } else if (mglTextureNeedsChannelExpansion(
                                        dstTex->internalformat, mglBlitTextureInfo(dstTexture).pixel_format)) {
                                    NSUInteger expandedBPR = 0, expandedBPI = 0;
                                    expandedData = mglCreateChannelExpandedUpload(
                                        dstTex, mglBlitTextureInfo(dstTexture).pixel_format,
                                        (const uint8_t *)upSrcPtr,
                                        copyWidth, copyHeight, upBytesPerRow,
                                        &expandedBPR, &expandedBPI);
                                    if (expandedData) {
                                        upSrcPtr = expandedData;
                                        upBytesPerRow = expandedBPR;
                                        upBytesPerImage = expandedBPI;
                                    }
                                }
                            }
                            @try {
                                mglBlitReplaceTextureRegion(
                                    dstTexture, region, (NSUInteger)dstLevel,
                                    mtlSlice, upSrcPtr, upBytesPerRow,
                                    upBytesPerImage, YES);
                            } @catch (NSException *exception) {
                                NSLog(@"MGL WARNING: CPU-to-CPU Metal update failed: %@",
                                      exception);
                            }
                            free(expandedData);
                        } else {
                            /* Private storage: blit from a staging buffer.
                             * For bpp mismatch formats (CPU bpp != Metal bpp),
                             * expand CPU data to Metal format before blitting,
                             * otherwise sourceBytesPerRow won't match the
                             * Metal texture's expected row stride. */
                            NSUInteger dstMetalBpp = mglMetalReadbackBytesPerPixel(mglBlitTextureInfo(dstTexture).pixel_format);
                            size_t dstCpuBpp = (dstLvl->width > 0) ?
                                (dstLvl->pitch / dstLvl->width) : 0;

                            const void *srcPtr = (const uint8_t *)(uintptr_t)dstLvl->data + dstSliceOff;
                            NSUInteger srcBytesPerRow = dstLvl->pitch;
                            NSUInteger srcImageBytes = srcBytesPerRow * copyHeight;

                            void *expandedData = NULL;
                            if (dstMetalBpp > 0 && dstCpuBpp != dstMetalBpp) {
                                if (mglTextureInternalFormatNeedsRGBA8Expansion(
                                        dstTex->internalformat, mglBlitTextureInfo(dstTexture).pixel_format)) {
                                    NSUInteger expandedBPR = 0, expandedBPI = 0;
                                    expandedData = mglCreateRGBA8ExpandedUpload(
                                        dstTex, (const uint8_t *)srcPtr,
                                        copyWidth, copyHeight, srcBytesPerRow,
                                        &expandedBPR, &expandedBPI);
                                    if (expandedData) {
                                        srcPtr = expandedData;
                                        srcBytesPerRow = expandedBPR;
                                        srcImageBytes = expandedBPI;
                                    }
                                } else if (mglTextureNeedsChannelExpansion(
                                        dstTex->internalformat, mglBlitTextureInfo(dstTexture).pixel_format)) {
                                    NSUInteger expandedBPR = 0, expandedBPI = 0;
                                    expandedData = mglCreateChannelExpandedUpload(
                                        dstTex, mglBlitTextureInfo(dstTexture).pixel_format,
                                        (const uint8_t *)srcPtr,
                                        copyWidth, copyHeight, srcBytesPerRow,
                                        &expandedBPR, &expandedBPI);
                                    if (expandedData) {
                                        srcPtr = expandedData;
                                        srcBytesPerRow = expandedBPR;
                                        srcImageBytes = expandedBPI;
                                    }
                                }
                            }

                            id stagingBuf =
                                mglBlitCreateBufferWithBytes(
                                    _device, srcPtr, srcImageBytes,
                                    MGLResourceStorageModeShared);
                            if (stagingBuf) {
                                id uploadEncoder =
                                    (__bridge id)mglRenderCreateBlitEncoderBorrowed(
                                        _commandState.currentCommandBufferOwner);
                                if (uploadEncoder) {
                                    mglBlitCopyBufferToTexture(
                                        uploadEncoder, stagingBuf, 0,
                                        srcBytesPerRow, srcImageBytes,
                                        mglBlitSize(copyWidth, copyHeight, 1),
                                        dstTexture, mtlSlice,
                                        (NSUInteger)dstLevel, region.origin);
                                    mglBlitEndBlitEncoder(uploadEncoder);
                                }
                            }
                            free(expandedData);
                        }
                    }
                }

                if (cpuCopyOK) {
                    /* CPU data is now authoritative for dst level */
                    if (dstTex->faces[0].levels) {
                        dstTex->faces[0].levels[dstLevel].metal_data_authoritative = GL_FALSE;
                    }
                    return YES;
                }
            }
        }
    return NO;
}

/* Metal-to-Metal format-conversion copy for mtlCopyImageSubData.
 * Reads source via getBytes and writes destination via replaceRegion when
 * source and destination have different Metal pixel formats.
 * Returns YES if formats differ and the path was taken (caller should
 * return); NO if formats match (caller should continue). */
- (BOOL)copyImageSubDataFormatConversion:(GLMContext)glm_ctx
                                  srcTex:(Texture *)srcTex
                              srcTexture:(id)srcTexture
                                 srcType:(uint32_t)srcType
                                srcLevel:(GLint)srcLevel
                                    srcX:(GLint)srcX srcY:(GLint)srcY srcZ:(GLint)srcZ
                                  dstTex:(Texture *)dstTex
                              dstTexture:(id)dstTexture
                                 dstType:(uint32_t)dstType
                                dstLevel:(GLint)dstLevel
                                    dstX:(GLint)dstX dstY:(GLint)dstY dstZ:(GLint)dstZ
                                   width:(GLsizei)width height:(GLsizei)height depth:(GLsizei)depth
{
    /* Metal-to-Metal copy path for format conversion cases (different
     * Metal pixel formats).  Read source pixels from Metal via getBytes,
     * then write to destination Metal via replaceRegion.  GL CopyImageSubData
     * does raw memcpy of pixel data, so format reinterpretation is OK.
     * This path has proper render pass synchronization, which the blit
     * path lacks for renderbuffer sources. */
    if (mglBlitTextureInfo(srcTexture).pixel_format != mglBlitTextureInfo(dstTexture).pixel_format) {
        if (mglBlitTextureInfo(dstTexture).storage_mode != MGLStorageModePrivate) {
            NSUInteger srcMetalBpp = mglMetalReadbackBytesPerPixel(mglBlitTextureInfo(srcTexture).pixel_format);
            NSUInteger dstMetalBpp = mglMetalReadbackBytesPerPixel(mglBlitTextureInfo(dstTexture).pixel_format);
            if (srcMetalBpp > 0 && dstMetalBpp > 0 && srcMetalBpp == dstMetalBpp) {
                /* Ensure any pending render passes are flushed before reading
                 * from the source (especially important for renderbuffers). */
                [self endRenderEncoding];
                [self synchronizeRenderPassForTextureReadback:srcTexture reason:"copyImageSubData.formatConv"];
                [self flushCommandBuffer: YES];
                NSUInteger copyWidth = MAX((NSUInteger)width, 1u);
                NSUInteger copyHeight = MAX((NSUInteger)height, 1u);
                NSUInteger numSlices = MAX((NSUInteger)depth, 1u);
                NSUInteger rowBytes = copyWidth * srcMetalBpp;
                NSUInteger imageBytes = rowBytes * copyHeight;
                void *stagingBuf = malloc(imageBytes);
                bool metalCopyOK = (stagingBuf != NULL);

                for (NSUInteger s = 0; s < numSlices && metalCopyOK; s++) {
                    /* Read source slice.  Prefer CPU data when available
                     * (metal_data_authoritative == false) to avoid AGX
                     * getBytes bugs on 3D and 2D-array textures. */
                    NSUInteger srcMtlSlice = 0;
                    MGLRegionValue srcRegion;
                    if (srcType == MGLTextureType3D) {
                        srcMtlSlice = 0;
                        srcRegion = mglBlitRegion3D((NSUInteger)srcX,
                                                     (NSUInteger)srcY,
                                                     (NSUInteger)srcZ + s,
                                                     copyWidth, copyHeight, 1);
                    } else if (srcType == MGLTextureTypeCube ||
                               srcType == MGLTextureTypeCubeArray) {
                        srcMtlSlice = ((NSUInteger)srcZ + s) % 6;
                        srcRegion = mglBlitRegion2D((NSUInteger)srcX,
                                                     (NSUInteger)srcY,
                                                     copyWidth, copyHeight);
                    } else {
                        srcMtlSlice = (NSUInteger)srcZ + s;
                        srcRegion = mglBlitRegion2D((NSUInteger)srcX,
                                                     (NSUInteger)srcY,
                                                     copyWidth, copyHeight);
                    }

                    bool srcReadFromCPU = false;
                    if (!srcTex->metal_data_authoritative && srcTex->faces &&
                        (NSUInteger)srcLevel < srcTex->num_levels) {
                        GLuint srcFace = 0;
                        if (srcType == MGLTextureTypeCube ||
                            srcType == MGLTextureTypeCubeArray) {
                            srcFace = ((GLuint)srcZ + s) % 6;
                        }
                        TextureLevel *srcLvl = (srcFace < 6 && srcTex->faces[srcFace].levels) ?
                            &srcTex->faces[srcFace].levels[srcLevel] : NULL;
                        if (srcLvl && srcLvl->data && srcLvl->pitch > 0 &&
                            srcLvl->width > 0) {
                            size_t srcCpuBpp = srcLvl->pitch / srcLvl->width;
                            if (srcCpuBpp == srcMetalBpp) {
                                size_t srcCpuPitch = srcLvl->pitch;
                                size_t srcCpuImgSize = srcCpuPitch * MAX(srcLvl->height, 1u);
                                size_t srcCpuOff = 0;
                                if (srcType == MGLTextureType3D) {
                                    srcCpuOff = ((NSUInteger)srcZ + s) * srcCpuImgSize +
                                                (NSUInteger)srcY * srcCpuPitch +
                                                (NSUInteger)srcX * srcCpuBpp;
                                } else if (srcType == MGLTextureType2DArray ||
                                           srcType == MGLTextureTypeCubeArray) {
                                    /* 2D array: all slices in one TextureLevel */
                                    GLuint arraySlice = (srcType == MGLTextureTypeCubeArray) ?
                                        ((GLuint)srcZ + s) / 6 : ((GLuint)srcZ + s);
                                    srcCpuOff = arraySlice * srcCpuImgSize +
                                                (NSUInteger)srcY * srcCpuPitch +
                                                (NSUInteger)srcX * srcCpuBpp;
                                } else {
                                    srcCpuOff = (NSUInteger)srcY * srcCpuPitch +
                                                (NSUInteger)srcX * srcCpuBpp;
                                }
                                size_t lastRowEnd = srcCpuOff + (copyHeight > 0 ? (copyHeight - 1) * srcCpuPitch : 0) + rowBytes;
                                if (lastRowEnd <= srcLvl->data_size) {
                                    for (NSUInteger y = 0; y < copyHeight; y++) {
                                        memcpy((uint8_t *)stagingBuf + y * rowBytes,
                                               (const uint8_t *)(uintptr_t)srcLvl->data + srcCpuOff + y * srcCpuPitch,
                                               rowBytes);
                                    }
                                    srcReadFromCPU = true;
                                }
                            }
                        }
                    }

                    if (!srcReadFromCPU) {
                        if (srcType == MGLTextureType3D &&
                            MGLCapabilityHasBug(&_capability,
                                                MGL_BUG_3D_GETBYTES_SLICE_OOB)) {
                            if (![self readTextureRegionViaBlit:srcTexture
                                                        region:srcRegion
                                                         slice:srcMtlSlice
                                                         level:(NSUInteger)srcLevel
                                                         bytes:stagingBuf
                                                   bytesPerRow:rowBytes
                                                 bytesPerImage:imageBytes
                                                        reason:"copyImageSubData.formatConv3DReadback"]) {
                                metalCopyOK = false;
                                break;
                            }
                        } else {
                            @try {
                                mglBlitGetTextureBytes(
                                    srcTexture, stagingBuf, rowBytes,
                                    imageBytes, srcRegion,
                                    (NSUInteger)srcLevel, srcMtlSlice, YES);
                            } @catch (NSException *exception) {
                                NSLog(@"MGL WARNING: format conv renderbuffer readback failed: %@",
                                      exception);
                                metalCopyOK = false;
                                break;
                            }
                        }
                    }

                    /* Write to destination Metal via replaceRegion */
                    @try {
                        NSUInteger dstMtlSlice = 0;
                        MGLRegionValue dstRegion;
                        if (dstType == MGLTextureType3D) {
                            dstMtlSlice = 0;
                            dstRegion = mglBlitRegion3D((NSUInteger)dstX,
                                                         (NSUInteger)dstY,
                                                         (NSUInteger)dstZ + s,
                                                         copyWidth, copyHeight, 1);
                        } else if (dstType == MGLTextureTypeCube ||
                                   dstType == MGLTextureTypeCubeArray) {
                            dstMtlSlice = ((NSUInteger)dstZ + s) % 6;
                            dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                         (NSUInteger)dstY,
                                                         copyWidth, copyHeight);
                        } else {
                            dstMtlSlice = (NSUInteger)dstZ + s;
                            dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                         (NSUInteger)dstY,
                                                         copyWidth, copyHeight);
                        }
                        mglBlitReplaceTextureRegion(
                            dstTexture, dstRegion, (NSUInteger)dstLevel,
                            dstMtlSlice, stagingBuf, rowBytes, imageBytes,
                            YES);
                    } @catch (NSException *exception) {
                        NSLog(@"MGL WARNING: format conv renderbuffer Metal update failed: %@",
                              exception);
                    }

                    /* Also update dst CPU data if available */
                    if (dstTex->faces && (NSUInteger)dstLevel < dstTex->num_levels) {
                        GLuint dstFace = 0;
                        if (dstType == MGLTextureTypeCube || dstType == MGLTextureTypeCubeArray) {
                            dstFace = ((GLuint)dstZ + s) % 6;
                        }
                        TextureLevel *curDstLvl = (dstFace < 6 && dstTex->faces[dstFace].levels) ?
                            &dstTex->faces[dstFace].levels[dstLevel] : NULL;
                        if (curDstLvl && curDstLvl->data && curDstLvl->pitch > 0 &&
                            curDstLvl->width > 0) {
                            size_t dstCpuBpp = curDstLvl->pitch / curDstLvl->width;
                            if (dstCpuBpp == dstMetalBpp) {
                                size_t dstSlicePitch = curDstLvl->pitch * MAX(curDstLvl->height, 1u);
                                bool dstSliced = (dstType == MGLTextureType3D ||
                                                  dstType == MGLTextureType2DArray);
                                size_t dstSliceOff = dstSliced ?
                                    ((NSUInteger)dstZ + s) * dstSlicePitch : 0;
                                for (NSUInteger y = 0; y < copyHeight; y++) {
                                    size_t dstOff = dstSliceOff +
                                                    ((NSUInteger)dstY + y) * curDstLvl->pitch +
                                                    (NSUInteger)dstX * dstMetalBpp;
                                    if (dstOff + rowBytes <= curDstLvl->data_size) {
                                        memcpy((uint8_t *)(uintptr_t)curDstLvl->data + dstOff,
                                               (const uint8_t *)stagingBuf + y * rowBytes,
                                               rowBytes);
                                    }
                                }
                            }
                        }
                    }
                }
                free(stagingBuf);
                if (metalCopyOK) {
                    /* Do NOT set metal_data_authoritative = GL_TRUE here.
                     *
                     * Previously, this code set the destination level's
                     * metal_data_authoritative flag to force glGetTexImage
                     * to read from Metal.  However, this causes failures
                     * for destination textures whose Metal data may not be
                     * fully initialized (e.g., RGB9_E5 2D-array textures
                     * where replaceRegion only updates the copied region,
                     * leaving non-copied regions with stale Metal data).
                     *
                     * glCopyImageSubData does a raw bit copy.  The CPU data
                     * was updated above with the source's raw bits at the
                     * copy region, and non-copied regions retain their
                     * original values from glTexImage*.  This is correct
                     * for both memcmp and float-epsilon comparisons used
                     * by CTS.  Keeping CPU data authoritative avoids AGX
                     * Metal readback bugs on 3D and certain packed formats. */
                    return YES;
                }
            }
        }


        return YES;
    }
    return NO;
}


- (BOOL)copyImageSubData3DFallback:(GLMContext)glm_ctx
                            srcTex:(Texture *)srcTex
                        srcTexture:(id)srcTexture
                           srcType:(uint32_t)srcType
                          srcLevel:(GLint)srcLevel
                              srcX:(GLint)srcX srcY:(GLint)srcY srcZ:(GLint)srcZ
                            dstTex:(Texture *)dstTex
                        dstTexture:(id)dstTexture
                           dstType:(uint32_t)dstType
                          dstLevel:(GLint)dstLevel
                              dstX:(GLint)dstX dstY:(GLint)dstY dstZ:(GLint)dstZ
                             width:(GLsizei)width height:(GLsizei)height depth:(GLsizei)depth
{
    /* Fallback for 3D texture destinations: AGX drivers have a bug where
     * copyFromTexture:toTexture: triggers "slice OOB" assertions when the
     * destination is a 3D texture.  Use a buffer-mediated copy instead:
     *   1. Read source region into a staging buffer (getBytes for shared
     *      textures, or blit-to-buffer for private textures)
     *   2. Write staging buffer to 3D destination via replaceRegion
     * This bypasses the buggy blit path entirely.  Private 3D destinations
     * cannot use replaceRegion and fall through to the blit path below.
     * Driver bug is tracked via MGLCapabilityHasBug(MGL_BUG_3D_GETBYTES_SLICE_OOB). */
    bool needs3DWorkaround =
        MGLCapabilityHasBug(&_capability, MGL_BUG_3D_GETBYTES_SLICE_OOB) ||
        MGLCapabilityHasBug(&_capability, MGL_BUG_3D_REPLACE_REGION_NONZERO_ORIGIN) ||
        MGLCapabilityHasBug(&_capability, MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB);
    if (needs3DWorkaround &&
        dstType == MGLTextureType3D &&
        mglBlitTextureInfo(dstTexture).storage_mode != MGLStorageModePrivate) {
        NSUInteger bpp = mglMetalReadbackBytesPerPixel(mglBlitTextureInfo(srcTexture).pixel_format);
        if (bpp == 0u) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return YES;
        }


        TextureLevel *earlyDstLevelInfo = NULL;
        if (dstTex->faces && dstTex->faces[0].levels &&
            (NSUInteger)dstLevel < dstTex->num_levels) {
            earlyDstLevelInfo = &dstTex->faces[0].levels[dstLevel];
        }
        if (!earlyDstLevelInfo || !earlyDstLevelInfo->data ||
            dstTex->metal_data_authoritative) {

            return NO;  /* fall through to blit path */
        }
        bool bppMismatch = false;
        size_t cpuBpp = 0;
        {
            size_t earlyPitch = earlyDstLevelInfo->pitch;
            if (earlyPitch == 0) {
                earlyPitch = (size_t)earlyDstLevelInfo->width * bpp;
            }
            cpuBpp = (earlyDstLevelInfo->width > 0) ?
                (earlyPitch / earlyDstLevelInfo->width) : 0;
            if (cpuBpp == 0) {
                return NO;  /* fall through to blit path */
            }
            if (cpuBpp != (size_t)bpp) {
                bppMismatch = true;
            }
        }

        NSUInteger copyWidth = MAX((NSUInteger)width, 1u);
        NSUInteger copyHeight = MAX((NSUInteger)height, 1u);
        NSUInteger copyDepth3D = MAX((NSUInteger)depth, 1u);
        NSUInteger rowBytes = copyWidth * bpp;
        NSUInteger imageBytes = rowBytes * copyHeight;
        NSUInteger totalBytes = imageBytes * copyDepth3D;

        void *stagingBytes = malloc(totalBytes);
        if (!stagingBytes) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return YES;
        }

        /* Read from source into staging buffer.
         * Prefer CPU data when available (metal_data_authoritative == false)
         * to avoid Metal getBytes/blit issues with certain texture types.
         * Fall back to Metal readback for Private textures or when Metal
         * data is authoritative (e.g. renderbuffers). */
        bool srcReadFromCPU = false;
        if (!srcTex->metal_data_authoritative && srcTex->faces &&
            srcTex->faces[0].levels &&
            (NSUInteger)srcLevel < srcTex->num_levels) {
            TextureLevel *srcLevelInfo = &srcTex->faces[0].levels[srcLevel];
            if (srcLevelInfo->data && srcLevelInfo->width > 0 &&
                srcLevelInfo->height > 0 && srcLevelInfo->pitch > 0) {
                size_t srcBpp = srcLevelInfo->pitch / srcLevelInfo->width;
                if (srcBpp == bpp) {
                    /* Read source pixels from CPU data */
                    size_t srcPitch = srcLevelInfo->pitch;
                    size_t srcImageBytes = srcPitch * srcLevelInfo->height;
                    if (srcType == MGLTextureType3D) {
                        /* 3D source: srcZ is depth origin */
                        for (NSUInteger z = 0; z < copyDepth3D; z++) {
                            for (NSUInteger y = 0; y < copyHeight; y++) {
                                NSUInteger srcOff = ((NSUInteger)srcZ + z) * srcImageBytes +
                                                    ((NSUInteger)srcY + y) * srcPitch +
                                                    (NSUInteger)srcX * bpp;
                                NSUInteger dstOff = z * imageBytes + y * rowBytes;
                                if (srcOff + rowBytes <= srcImageBytes * srcLevelInfo->depth &&
                                    dstOff + rowBytes <= totalBytes) {
                                    memcpy((uint8_t *)stagingBytes + dstOff,
                                           (const uint8_t *)srcLevelInfo->data + srcOff,
                                           rowBytes);
                                }
                            }
                        }
                    } else {
                        /* Non-3D source (2D array, cube, etc.): srcZ is slice */
                        for (NSUInteger z = 0; z < copyDepth3D; z++) {
                            NSUInteger face = 0;
                            if (srcType == MGLTextureTypeCube ||
                                srcType == MGLTextureTypeCubeArray) {
                                face = (NSUInteger)srcZ + z;
                            }
                            TextureLevel *sliceLevel = (face < 6 && srcTex->faces[face].levels) ?
                                &srcTex->faces[face].levels[srcLevel] : srcLevelInfo;
                            if (!sliceLevel || !sliceLevel->data) {
                                srcReadFromCPU = false;
                                break;
                            }
                            size_t sPitch = sliceLevel->pitch;
                            size_t sBpp = (sliceLevel->width > 0) ?
                                (sPitch / sliceLevel->width) : 0;
                            if (sBpp != bpp) {
                                srcReadFromCPU = false;
                                break;
                            }
                            /* For 2D array, all slices are in one
                             * TextureLevel; add slice offset. */
                            size_t srcSliceOff = 0;
                            if (srcType == MGLTextureType2DArray) {
                                srcSliceOff = ((NSUInteger)srcZ + z) *
                                    sPitch * MAX(sliceLevel->height, 1u);
                            }
                            for (NSUInteger y = 0; y < copyHeight; y++) {
                                NSUInteger srcOff = srcSliceOff +
                                    ((NSUInteger)srcY + y) * sPitch +
                                    (NSUInteger)srcX * bpp;
                                NSUInteger dstOff = z * imageBytes + y * rowBytes;
                                if (srcOff + rowBytes <= sliceLevel->data_size &&
                                    dstOff + rowBytes <= totalBytes) {
                                    memcpy((uint8_t *)stagingBytes + dstOff,
                                           (const uint8_t *)sliceLevel->data + srcOff,
                                           rowBytes);
                                }
                            }
                        }
                    }
                    srcReadFromCPU = true;
                }
            }
        }

        if (!srcReadFromCPU) {
        /* Read from source Metal texture.
         * For 3D sources, read the entire 3D region in one call.
         * For non-3D sources (2D array, cube, etc.), loop over slices
         * and read each slice separately. */
        @try {
            if (srcType == MGLTextureType3D) {
                MGLRegionValue srcRegion = mglBlitRegion3D((NSUInteger)srcX, (NSUInteger)srcY,
                                                      (NSUInteger)srcZ, copyWidth,
                                                      copyHeight, copyDepth3D);
                if (mglBlitTextureInfo(srcTexture).storage_mode != MGLStorageModePrivate &&
                    !MGLCapabilityHasBug(&_capability,
                                         MGL_BUG_3D_GETBYTES_SLICE_OOB)) {
                    mglBlitGetTextureBytes(
                        srcTexture, stagingBytes, rowBytes, imageBytes,
                        srcRegion, (NSUInteger)srcLevel, 0, YES);
                } else if (![self readTextureRegionViaBlit:srcTexture
                                                        region:srcRegion
                                                         slice:0
                                                         level:(NSUInteger)srcLevel
                                                         bytes:stagingBytes
                                                   bytesPerRow:rowBytes
                                                 bytesPerImage:imageBytes
                                                        reason:"copyImageSubData.3DReadback"]) {
                    free(stagingBytes);
                    mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
                    return YES;
                }
            } else {
                /* Non-3D source (2D array, cube, rectangle, etc.):
                 * Read each slice separately and place at the correct offset. */
                for (NSUInteger z = 0; z < copyDepth3D; z++) {
                    NSUInteger sliceOffset = z * imageBytes;
                    NSUInteger srcSlice = (NSUInteger)srcZ + z;
                    MGLRegionValue sliceRegion = mglBlitRegion3D((NSUInteger)srcX, (NSUInteger)srcY,
                                                            0, copyWidth, copyHeight, 1u);
                    if (mglBlitTextureInfo(srcTexture).storage_mode != MGLStorageModePrivate) {
                        mglBlitGetTextureBytes(
                            srcTexture,
                            (uint8_t *)stagingBytes + sliceOffset,
                            rowBytes, imageBytes, sliceRegion,
                            (NSUInteger)srcLevel, srcSlice, YES);
                    } else {
                        id sliceBuffer = mglBlitCreateBuffer(
                            _device, imageBytes, MGLResourceStorageModeShared);
                        if (!sliceBuffer) {
                            free(stagingBytes);
                            mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                            return YES;
                        }
                        id readEncoder =
                            (__bridge id)mglRenderCreateBlitEncoderBorrowed(
                                _commandState.currentCommandBufferOwner);
                        if (!readEncoder) {
                            free(stagingBytes);
                            mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                            return YES;
                        }
                        _currentCBHasWork = YES;
                        mglBlitCopyTextureToBuffer(
                            readEncoder, srcTexture, srcSlice,
                            (NSUInteger)srcLevel, sliceRegion.origin,
                            sliceRegion.size, sliceBuffer, 0, rowBytes,
                            imageBytes);
                        mglBlitEndBlitEncoder(readEncoder);
                        [self flushCommandBuffer:YES];
                        void *sliceContents = NULL;
                        uint64_t sliceLength = 0;
                        if (mglRenderGetBufferContents(
                                (__bridge void *)sliceBuffer,
                                &sliceContents, &sliceLength) != 0 ||
                            !sliceContents || sliceLength < imageBytes) {
                            free(stagingBytes);
                            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
                            return YES;
                        }
                        memcpy((uint8_t *)stagingBytes + sliceOffset,
                               sliceContents, imageBytes);
                    }
                }
            }
        } @catch (NSException *exception) {
            free(stagingBytes);
            NSLog(@"MGL ERROR: mtlCopyImageSubData 3D fallback read failed: %@",
                  exception);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return YES;
        }
        } /* end if (!srcReadFromCPU) */

        /* For bpp mismatch formats, convert staging from Metal format to
         * CPU storage format so the RMW merge uses matching pixel sizes. */
        if (bppMismatch) {
            GLenum cpuFormat = 0, cpuType = 0;
            if (mglGetCPUFormatTypeForInternalFormat(dstTex->internalformat,
                                                      &cpuFormat, &cpuType)) {
                NSUInteger cpuRowBytes = copyWidth * (NSUInteger)cpuBpp;
                NSUInteger cpuImageBytes = cpuRowBytes * copyHeight;
                NSUInteger cpuTotalBytes = cpuImageBytes * copyDepth3D;
                void *cpuStaging = malloc(cpuTotalBytes);
                if (cpuStaging) {
                    bool convOK = true;
                    for (NSUInteger z = 0; z < copyDepth3D && convOK; z++) {
                        const uint8_t *metalSrc = (const uint8_t *)stagingBytes + z * imageBytes;
                        uint8_t *cpuDst = (uint8_t *)cpuStaging + z * cpuImageBytes;
                        if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL(
                                metalSrc, rowBytes, cpuDst, cpuRowBytes,
                                copyWidth, copyHeight, mglBlitTextureInfo(srcTexture).pixel_format,
                                cpuFormat, cpuType, NO)) {
                            convOK = false;
                        }
                    }
                    if (convOK) {
                        free(stagingBytes);
                        stagingBytes = cpuStaging;
                        rowBytes = cpuRowBytes;
                        imageBytes = cpuImageBytes;
                        totalBytes = cpuTotalBytes;
                        bpp = (NSUInteger)cpuBpp;
                    } else {
                        free(cpuStaging);
                        free(stagingBytes);
                        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
                        return YES;
                    }
                }
            }
        }

        /* Write to 3D destination via CPU-data read-modify-write.
         * AGX drivers have bugs where replaceRegion with non-zero origin,
         * getBytes, copyFromTexture:toTexture:, and copyFromBuffer:toTexture:
         * all trigger "slice OOB" assertions on 3D textures.  The only safe
         * Metal write path for 3D textures is replaceRegion with origin
         * (0,0,0).  So we use the CPU-side level data as the base, merge the
         * source pixels into it, and write the entire level back. */
        {
            /* Early checks already verified: dstLevelInfo exists, has data,
             * metal_data_authoritative == false, and cpuBpp == bpp. */
            TextureLevel *dstLevelInfo = &dstTex->faces[0].levels[dstLevel];
            GLuint levelWidth = dstLevelInfo->width;
            GLuint levelHeight = dstLevelInfo->height;
            GLuint levelDepth = dstLevelInfo->depth;
            size_t levelPitch = dstLevelInfo->pitch;
            if (levelPitch == 0) {
                levelPitch = (size_t)levelWidth * bpp;
            }
            size_t levelImageBytes = levelPitch * levelHeight;
            size_t fullTotalBytes = levelImageBytes * levelDepth;

            void *fullLevelBytes = malloc(fullTotalBytes);
            if (!fullLevelBytes) {
                free(stagingBytes);
                mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                return YES;
            }

            /* Copy CPU data as the base */
            memcpy(fullLevelBytes, (const void *)dstLevelInfo->data, fullTotalBytes);

            /* Merge source pixels into the full level buffer */
            for (NSUInteger z = 0; z < copyDepth3D; z++) {
                for (NSUInteger y = 0; y < copyHeight; y++) {
                    NSUInteger srcOff = z * imageBytes + y * rowBytes;
                    NSUInteger dstOff = ((NSUInteger)dstZ + z) * levelImageBytes +
                                        ((NSUInteger)dstY + y) * levelPitch +
                                        (NSUInteger)dstX * bpp;
                    if (dstOff + rowBytes <= fullTotalBytes &&
                        srcOff + rowBytes <= totalBytes) {
                        memcpy((uint8_t *)fullLevelBytes + dstOff,
                               (uint8_t *)stagingBytes + srcOff,
                               rowBytes);
                    }
                }
            }

            /* Write the full level back with origin (0,0,0).
             * For bpp mismatch, expand CPU data to Metal format first. */
            @try {
                MGLRegionValue fullRegion = mglBlitRegion3D(0, 0, 0, levelWidth, levelHeight, levelDepth);
                if (bppMismatch) {
                    NSUInteger expandedBPR = 0, expandedBPI = 0;
                    void *expandedData = NULL;
                    if (mglTextureInternalFormatNeedsRGBA8Expansion(
                            dstTex->internalformat, mglBlitTextureInfo(dstTexture).pixel_format)) {
                        expandedData = mglCreateRGBA8ExpandedUpload(
                            dstTex, (const uint8_t *)fullLevelBytes,
                            levelWidth, levelHeight * levelDepth,
                            levelPitch, &expandedBPR, &expandedBPI);
                    } else if (mglTextureNeedsChannelExpansion(
                            dstTex->internalformat, mglBlitTextureInfo(dstTexture).pixel_format)) {
                        expandedData = mglCreateChannelExpandedUpload(
                            dstTex, mglBlitTextureInfo(dstTexture).pixel_format,
                            (const uint8_t *)fullLevelBytes,
                            levelWidth, levelHeight * levelDepth,
                            levelPitch, &expandedBPR, &expandedBPI);
                    }
                    if (expandedData) {
                        NSUInteger expandedImageBytes = expandedBPR * levelHeight;
                        mglBlitReplaceTextureRegion(
                            dstTexture, fullRegion, (NSUInteger)dstLevel, 0,
                            expandedData, expandedBPR, expandedImageBytes,
                            YES);
                        free(expandedData);
                    } else {
                        mglBlitReplaceTextureRegion(
                            dstTexture, fullRegion, (NSUInteger)dstLevel, 0,
                            fullLevelBytes, levelPitch, levelImageBytes,
                            YES);
                    }
                } else {
                    mglBlitReplaceTextureRegion(
                        dstTexture, fullRegion, (NSUInteger)dstLevel, 0,
                        fullLevelBytes, levelPitch, levelImageBytes, YES);
                }
            } @catch (NSException *exception) {
                free(stagingBytes);
                free(fullLevelBytes);
                NSLog(@"MGL ERROR: mtlCopyImageSubData 3D replaceRegion failed: %@",
                      exception);
                mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
                return YES;
            }

            /* Update CPU data to reflect the merged result */
            memcpy((void *)dstLevelInfo->data, fullLevelBytes, fullTotalBytes);

            free(fullLevelBytes);
            free(stagingBytes);
            /* Do NOT set metal_data_authoritative = GL_TRUE here.
             * The AGX driver corrupts 3D texture readback (getBytes
             * triggers "slice OOB"), so subsequent glGetTexImage calls
             * must read from CPU data instead.  The Metal texture was
             * updated via replaceRegion for sampling, but CPU data
             * remains the authoritative source. */
            return YES;
        }

        /* If we get here, the 3D fallback didn't work (e.g., no CPU data or
         * Metal data is authoritative).  Fall through to the blit path. */
    }

    if (dstType == MGLTextureType3D) {
        /* 3D destination with Private storage or no CPU data: fall through
         * to the blit path.  This may trigger AGX "slice OOB" assertions,
         * but there is no safe alternative for Private 3D textures. */
    }
    return NO;
}

/* Post-blit CPU readback for mtlCopyImageSubData.
 * Reads the blitted region back from the destination Metal texture to CPU
 * data so that CPU data remains authoritative.  Handles both matching-bpp
 * and format-converting (bpp mismatch) readback paths.
 * Returns YES if readback succeeded (readbackDone). */
- (BOOL)copyImageSubDataPostBlitReadback:(Texture *)dstTex
                              dstTexture:(id)dstTexture
                                 dstType:(uint32_t)dstType
                               dstLevel:(GLint)dstLevel
                                   dstX:(GLint)dstX dstY:(GLint)dstY dstZ:(GLint)dstZ
                                  width:(GLsizei)width height:(GLsizei)height depth:(GLsizei)depth
{
    /* After blit, read back the blitted region from dst Metal to dst CPU
     * so that CPU data is authoritative.  This avoids the need for the
     * metal_data_authoritative flag, which causes "modified contents
     * outside of copied region" / "wrong layer" errors when non-blitted
     * regions of the same level have stale Metal data.
     *
     * Skip readback for 3D destinations (AGX getBytes bug on 3D textures,
     * tracked via MGLCapabilityHasBug(MGL_BUG_3D_GETBYTES_SLICE_OOB))
     * and fall back to per-level authoritative instead. */
    bool readbackDone = false;
    bool skip3DReadback = MGLCapabilityHasBug(&_capability, MGL_BUG_3D_GETBYTES_SLICE_OOB);
    if ((!skip3DReadback || dstType != MGLTextureType3D) &&
        mglBlitTextureInfo(dstTexture).storage_mode != MGLStorageModePrivate &&
        dstTex->faces && (NSUInteger)dstLevel < dstTex->num_levels) {

        /* Check that dst has CPU data for this level */
        TextureLevel *dstLvl0 = (dstTex->faces[0].levels) ?
            &dstTex->faces[0].levels[dstLevel] : NULL;
        if (dstLvl0 && dstLvl0->data && dstLvl0->pitch > 0 && dstLvl0->width > 0) {
            NSUInteger dstMetalBpp = mglMetalReadbackBytesPerPixel(mglBlitTextureInfo(dstTexture).pixel_format);
            size_t dstCpuBpp = dstLvl0->pitch / dstLvl0->width;
            if (dstMetalBpp > 0 && dstCpuBpp == dstMetalBpp) {
                [self synchronizeRenderPassForTextureReadback:dstTexture
                                                       reason:"copyImageSubData.blitReadback"];
                [self flushCommandBuffer: YES];

                NSUInteger copyWidth = MAX((NSUInteger)width, 1u);
                NSUInteger copyHeight = MAX((NSUInteger)height, 1u);
                NSUInteger numSlices = MAX((NSUInteger)depth, 1u);
                NSUInteger rowBytes = copyWidth * dstMetalBpp;
                NSUInteger imageBytes = rowBytes * copyHeight;
                void *stagingBuf = malloc(imageBytes);

                if (stagingBuf) {
                    bool readbackOK = true;
                    for (NSUInteger s = 0; s < numSlices && readbackOK; s++) {
                        NSUInteger dstMtlSlice = 0;
                        GLuint dstFace = 0;
                        MGLRegionValue dstRegion;

                        if (dstType == MGLTextureTypeCube ||
                            dstType == MGLTextureTypeCubeArray) {
                            dstMtlSlice = ((NSUInteger)dstZ + s) % 6;
                            dstFace = (GLuint)dstMtlSlice;
                            dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                        (NSUInteger)dstY,
                                                        copyWidth, copyHeight);
                        } else if (dstType == MGLTextureType2DArray) {
                            dstMtlSlice = (NSUInteger)dstZ + s;
                            dstFace = 0;
                            dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                        (NSUInteger)dstY,
                                                        copyWidth, copyHeight);
                        } else {
                            dstMtlSlice = 0;
                            dstFace = 0;
                            dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                        (NSUInteger)dstY,
                                                        copyWidth, copyHeight);
                        }

                        @try {
                            mglBlitGetTextureBytes(
                                dstTexture, stagingBuf, rowBytes, imageBytes,
                                dstRegion, (NSUInteger)dstLevel,
                                dstMtlSlice, YES);
                        } @catch (NSException *exception) {
                            NSLog(@"MGL WARNING: blit readback getBytes failed: %@",
                                  exception);
                            readbackOK = false;
                            break;
                        }

                        /* Update dst CPU data for this slice */
                        TextureLevel *curDstLvl = (dstFace < 6 &&
                            dstTex->faces[dstFace].levels) ?
                            &dstTex->faces[dstFace].levels[dstLevel] : NULL;
                        if (curDstLvl && curDstLvl->data && curDstLvl->pitch > 0 &&
                            curDstLvl->width > 0) {
                            size_t curCpuBpp = curDstLvl->pitch / curDstLvl->width;
                            if (curCpuBpp == dstMetalBpp) {
                                size_t slicePitch = curDstLvl->pitch *
                                                    MAX(curDstLvl->height, 1u);
                                size_t dstSliceOff = 0;
                                if (dstType == MGLTextureType2DArray) {
                                    dstSliceOff = ((NSUInteger)dstZ + s) * slicePitch;
                                }
                                for (NSUInteger y = 0; y < copyHeight; y++) {
                                    size_t dstOff = dstSliceOff +
                                        ((NSUInteger)dstY + y) * curDstLvl->pitch +
                                        (NSUInteger)dstX * dstMetalBpp;
                                    if (dstOff + rowBytes <= curDstLvl->data_size) {
                                        memcpy((uint8_t *)(uintptr_t)curDstLvl->data + dstOff,
                                               (const uint8_t *)stagingBuf + y * rowBytes,
                                               rowBytes);
                                    }
                                }
                            }
                        }
                    }
                    free(stagingBuf);

                    if (readbackOK) {

                        for (int f = 0; f < 6; f++) {
                            if (dstTex->faces[f].levels) {
                                dstTex->faces[f].levels[dstLevel].metal_data_authoritative = GL_FALSE;
                            }
                        }
                        readbackDone = true;
                    }
                }
            }
        }
    }

    /* Format-converting readback fallback for bpp mismatch cases (e.g.
     * R3_G3_B2, RGB12, RGB32F where CPU bpp != Metal bpp).  Read the
     * blitted region from dst Metal and convert to CPU storage format
     * so that CPU data is authoritative without setting per-texture
     * metal_data_authoritative (which would corrupt non-blitted levels). */
    if (!readbackDone &&
        dstType != MGLTextureType3D &&
        mglBlitTextureInfo(dstTexture).storage_mode != MGLStorageModePrivate &&
        dstTex->faces && (NSUInteger)dstLevel < dstTex->num_levels) {

        GLenum cpuFormat = 0, cpuType = 0;
        if (mglGetCPUFormatTypeForInternalFormat(dstTex->internalformat,
                                                  &cpuFormat, &cpuType)) {
            TextureLevel *dstLvl0 = (dstTex->faces[0].levels) ?
                &dstTex->faces[0].levels[dstLevel] : NULL;
            if (dstLvl0 && dstLvl0->data && dstLvl0->pitch > 0 &&
                dstLvl0->width > 0) {
                NSUInteger dstMetalBpp = mglMetalReadbackBytesPerPixel(mglBlitTextureInfo(dstTexture).pixel_format);
                NSUInteger cpuBpp = (NSUInteger)sizeForFormatType(cpuFormat, cpuType);
                if (dstMetalBpp > 0 && cpuBpp > 0) {
                    [self synchronizeRenderPassForTextureReadback:dstTexture
                                                           reason:"copyImageSubData.fmtConvReadback"];
                    [self flushCommandBuffer: YES];

                    NSUInteger copyWidth = MAX((NSUInteger)width, 1u);
                    NSUInteger copyHeight = MAX((NSUInteger)height, 1u);
                    NSUInteger numSlices = MAX((NSUInteger)depth, 1u);
                    NSUInteger metalRowBytes = copyWidth * dstMetalBpp;
                    NSUInteger metalImageBytes = metalRowBytes * copyHeight;
                    NSUInteger cpuRowBytes = copyWidth * cpuBpp;
                    void *metalStaging = malloc(metalImageBytes);
                    void *cpuStaging = malloc(cpuRowBytes * copyHeight);

                    if (metalStaging && cpuStaging) {
                        bool fmtReadbackOK = true;
                        for (NSUInteger s = 0; s < numSlices && fmtReadbackOK; s++) {
                            NSUInteger dstMtlSlice = 0;
                            MGLRegionValue dstRegion;
                            if (dstType == MGLTextureTypeCube ||
                                dstType == MGLTextureTypeCubeArray) {
                                dstMtlSlice = ((NSUInteger)dstZ + s) % 6;
                                dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                            (NSUInteger)dstY,
                                                            copyWidth, copyHeight);
                            } else if (dstType == MGLTextureType2DArray) {
                                dstMtlSlice = (NSUInteger)dstZ + s;
                                dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                            (NSUInteger)dstY,
                                                            copyWidth, copyHeight);
                            } else {
                                dstMtlSlice = 0;
                                dstRegion = mglBlitRegion2D((NSUInteger)dstX,
                                                            (NSUInteger)dstY,
                                                            copyWidth, copyHeight);
                            }

                            @try {
                                mglBlitGetTextureBytes(
                                    dstTexture, metalStaging, metalRowBytes,
                                    metalImageBytes, dstRegion,
                                    (NSUInteger)dstLevel, dstMtlSlice, YES);
                            } @catch (NSException *exception) {
                                NSLog(@"MGL WARNING: fmt-conv readback getBytes failed: %@",
                                      exception);
                                fmtReadbackOK = false;
                                break;
                            }

                            /* Convert from Metal format to CPU format */
                            if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL(
                                    (const uint8_t *)metalStaging,
                                    metalRowBytes,
                                    (uint8_t *)cpuStaging,
                                    cpuRowBytes,
                                    copyWidth, copyHeight,
                                    mglBlitTextureInfo(dstTexture).pixel_format,
                                    cpuFormat, cpuType, NO)) {
                                NSLog(@"MGL WARNING: fmt-conv readback conversion failed for fmt=0x%x",
                                      (unsigned)dstTex->internalformat);
                                fmtReadbackOK = false;
                                break;
                            }

                            /* Write to dst CPU data for this slice */
                            GLuint dstFace = 0;
                            if (dstType == MGLTextureTypeCube ||
                                dstType == MGLTextureTypeCubeArray) {
                                dstFace = (GLuint)(((NSUInteger)dstZ + s) % 6);
                            }
                            TextureLevel *curDstLvl = (dstFace < 6 &&
                                dstTex->faces[dstFace].levels) ?
                                &dstTex->faces[dstFace].levels[dstLevel] : NULL;
                            if (curDstLvl && curDstLvl->data && curDstLvl->pitch > 0 &&
                                curDstLvl->width > 0) {
                                size_t curCpuBpp = curDstLvl->pitch / curDstLvl->width;
                                if (curCpuBpp == cpuBpp) {
                                    size_t slicePitch = curDstLvl->pitch *
                                                        MAX(curDstLvl->height, 1u);
                                    size_t dstSliceOff = 0;
                                    if (dstType == MGLTextureType2DArray) {
                                        dstSliceOff = ((NSUInteger)dstZ + s) * slicePitch;
                                    }
                                    for (NSUInteger y = 0; y < copyHeight; y++) {
                                        size_t dstOff = dstSliceOff +
                                            ((NSUInteger)dstY + y) * curDstLvl->pitch +
                                            (NSUInteger)dstX * cpuBpp;
                                        if (dstOff + cpuRowBytes <= curDstLvl->data_size) {
                                            memcpy((uint8_t *)(uintptr_t)curDstLvl->data + dstOff,
                                                   (const uint8_t *)cpuStaging + y * cpuRowBytes,
                                                   cpuRowBytes);
                                        }
                                    }
                                }
                            }
                        }

                        if (fmtReadbackOK) {
                            for (int f = 0; f < 6; f++) {
                                if (dstTex->faces[f].levels) {
                                    dstTex->faces[f].levels[dstLevel].metal_data_authoritative = GL_FALSE;
                                }
                            }
                            readbackDone = true;
                        }
                    }
                    free(metalStaging);
                    free(cpuStaging);
                }
            }
        }
    }
    return readbackDone ? YES : NO;
}

-(void)mtlCopyImageSubData:(GLMContext)glm_ctx
                 srcTexture:(Texture *)srcTex
                  srcLevel:(GLint)srcLevel
                      srcX:(GLint)srcX
                      srcY:(GLint)srcY
                      srcZ:(GLint)srcZ
                 dstTexture:(Texture *)dstTex
                  dstLevel:(GLint)dstLevel
                      dstX:(GLint)dstX
                      dstY:(GLint)dstY
                      dstZ:(GLint)dstZ
                     width:(GLsizei)width
                    height:(GLsizei)height
                    depth:(GLsizei)depth
{
    ctx = glm_ctx;

    if (!srcTex || !dstTex || width <= 0 || height <= 0 || depth <= 0) {
        return;
    }


    if (![self bindMTLTexture:srcTex]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    if (![self bindMTLTexture:dstTex]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    id srcTexture = (__bridge id)(srcTex->mtl_data);
    id dstTexture = (__bridge id)(dstTex->mtl_data);
    if (!srcTexture || !dstTexture) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    uint32_t srcType = mglBlitTextureInfo(srcTexture).texture_type;
    uint32_t dstType = mglBlitTextureInfo(dstTexture).texture_type;

    if ((NSUInteger)srcLevel >= mglBlitTextureInfo(srcTexture).mipmap_level_count ||
        (NSUInteger)dstLevel >= mglBlitTextureInfo(dstTexture).mipmap_level_count) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }

    bool needs3DDestinationWorkaround = dstType == MGLTextureType3D &&
        (MGLCapabilityHasBug(&_capability, MGL_BUG_3D_GETBYTES_SLICE_OOB) ||
         MGLCapabilityHasBug(&_capability, MGL_BUG_3D_REPLACE_REGION_NONZERO_ORIGIN) ||
         MGLCapabilityHasBug(&_capability, MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB));
    if (needs3DDestinationWorkaround) {
        [self endRenderPassIfFramebufferChangedForNonDraw:0];
        [self endRenderEncoding];
        RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlCopyImageSubData.3D"]);
        if ([self copyImageSubData3DFallback:glm_ctx
                                     srcTex:srcTex
                                 srcTexture:srcTexture
                                    srcType:srcType
                                   srcLevel:srcLevel
                                       srcX:srcX srcY:srcY srcZ:srcZ
                                     dstTex:dstTex
                                 dstTexture:dstTexture
                                    dstType:dstType
                                   dstLevel:dstLevel
                                       dstX:dstX dstY:dstY dstZ:dstZ
                                      width:width height:height depth:depth]) {
            return;
        }
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if ([self copyImageSubDataCpuToCpu:glm_ctx
                                srcTex:srcTex
                            srcTexture:srcTexture
                               srcType:srcType
                              srcLevel:srcLevel
                                  srcX:srcX srcY:srcY srcZ:srcZ
                                dstTex:dstTex
                            dstTexture:dstTexture
                               dstType:dstType
                              dstLevel:dstLevel
                                  dstX:dstX dstY:dstY dstZ:dstZ
                                 width:width height:height depth:depth]) {
        return;
    }

    if ([self copyImageSubDataFormatConversion:glm_ctx
                                        srcTex:srcTex
                                    srcTexture:srcTexture
                                       srcType:srcType
                                      srcLevel:srcLevel
                                          srcX:srcX srcY:srcY srcZ:srcZ
                                        dstTex:dstTex
                                    dstTexture:dstTexture
                                       dstType:dstType
                                      dstLevel:dstLevel
                                          dstX:dstX dstY:dstY dstZ:dstZ
                                         width:width height:height depth:depth]) {
        return;
    }

    // End a stale render pass (if the render encoder's FBO no longer matches
    // the current context FBO) so the blit encoder is not interleaved with
    // a live render encoder.  This is the only GL state the blit path depends
    // on; the full processGLState:false sync is unnecessary here.
    [self endRenderPassIfFramebufferChangedForNonDraw:0];
    [self endRenderEncoding];
    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlCopyImageSubData"]);

    /* For cube / cube-array / 2D-array / 1D-array targets, srcZ selects
     * the slice.  For 3D textures, srcZ is the depth origin. */
    NSUInteger srcSlice = 0;
    NSUInteger dstSlice = 0;
    NSUInteger srcDepthPlane = 0;
    NSUInteger dstDepthPlane = 0;
    NSUInteger copyDepth = MAX((NSUInteger)depth, 1u);

    if (srcType == MGLTextureType3D) {
        srcDepthPlane = (NSUInteger)srcZ;
        srcSlice = 0;
    } else {
        srcSlice = (NSUInteger)srcZ;
        srcDepthPlane = 0;
    }

    if (dstType == MGLTextureType3D) {
        dstDepthPlane = (NSUInteger)dstZ;
        dstSlice = 0;
    } else {
        dstSlice = (NSUInteger)dstZ;
        dstDepthPlane = 0;
    }


    NSUInteger iterations;
    NSUInteger srcSizeDepth;

    if (srcType == MGLTextureType3D && dstType == MGLTextureType3D) {
        iterations = 1u;
        srcSizeDepth = copyDepth;
    } else {
        iterations = copyDepth;
        srcSizeDepth = 1u;
    }

    /* Debug: read source renderbuffer data before blit to verify it has content */
    if (srcTex->is_render_target || dstTex->is_render_target) {
        [self synchronizeRenderPassForTextureReadback:srcTexture reason:"copyImageSubData.srcCheck"];
        [self endRenderEncoding];
    }

    id blitEncoder =
        (__bridge id)mglRenderCreateBlitEncoderBorrowed(
            _commandState.currentCommandBufferOwner);
    if (!blitEncoder) {
        NSLog(@"MGL ERROR: mtlCopyImageSubData failed to create blit encoder");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }

    @try {
        for (NSUInteger i = 0; i < iterations; i++) {
            NSUInteger curSrcSlice = srcSlice;
            NSUInteger curSrcDepth = srcDepthPlane;
            NSUInteger curDstSlice = dstSlice;
            NSUInteger curDstDepth = dstDepthPlane;

            if (srcType == MGLTextureType3D && dstType != MGLTextureType3D) {
                /* 3D -> 2D/array: read depth plane i from src */
                curSrcDepth = srcDepthPlane + i;
                curSrcSlice = 0;
                curDstSlice = dstSlice + i;
            } else if (srcType != MGLTextureType3D && dstType == MGLTextureType3D) {
                /* 2D/array -> 3D: read slice i from src, write to dst depth */
                curSrcSlice = srcSlice + i;
                curDstDepth = dstDepthPlane + i;
                curDstSlice = 0;
            } else if (srcType != MGLTextureType3D && dstType != MGLTextureType3D) {
                /* 2D/array -> 2D/array: copy slice i to slice i */
                curSrcSlice = srcSlice + i;
                curDstSlice = dstSlice + i;
            }


            mglBlitCopyTexture(
                blitEncoder, srcTexture, curSrcSlice, (NSUInteger)srcLevel,
                mglBlitOrigin((NSUInteger)srcX, (NSUInteger)srcY,
                              curSrcDepth),
                mglBlitSize((NSUInteger)width, (NSUInteger)height,
                            srcSizeDepth),
                dstTexture, curDstSlice, (NSUInteger)dstLevel,
                mglBlitOrigin((NSUInteger)dstX, (NSUInteger)dstY,
                              curDstDepth));
        }
        mglBlitEndBlitEncoder(blitEncoder);
    } @catch (NSException *exception) {
        @try {
            mglBlitEndBlitEncoder(blitEncoder);
        } @catch (NSException *endException) {
            NSLog(@"MGL WARNING: mtlCopyImageSubData failed to end blit encoder: %@",
                  endException);
        }
        NSLog(@"MGL ERROR: mtlCopyImageSubData blit failed: %@", exception);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    /* Flush the command buffer to ensure the blit is executed before any
     * subsequent readback (e.g. glGetTexImage).  Without this, the blit
     * may still be pending in the command buffer when the readback occurs. */
    [self flushCommandBuffer: NO];

    bool readbackDone = [self copyImageSubDataPostBlitReadback:dstTex
                                                        dstTexture:dstTexture
                                                           dstType:dstType
                                                          dstLevel:dstLevel
                                                              dstX:dstX dstY:dstY dstZ:dstZ
                                                             width:width height:height depth:depth];


    if (!readbackDone) {
        if (dstType == MGLTextureType3D &&
            dstTex->faces && (NSUInteger)dstLevel < dstTex->num_levels &&
            dstTex->faces[0].levels) {
            dstTex->faces[0].levels[dstLevel].metal_data_authoritative = GL_TRUE;
        } else {
            dstTex->metal_data_authoritative = GL_TRUE;
        }
    }
}

@end
