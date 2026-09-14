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
#include "mgl_blit_drivers.h" /* blit leaf paths (log 135) */
#include "mgl_blit_color_state.h" /* shared blit color state (log 136) */
#include "mgl_blit_sampled_copy.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_texture_readback_clear.h"
#import "MGLRenderer+Blit_Private.h"
#include "mgl_render.h"
#include "mgl_blit_pipelines.h"
#include "mgl_env_flag.h"
#include "mgl_batch_path.h"
#include "mgl_aux_assets.h"
#include <stdio.h>
#include "mgl_region_value.h"   // canonical region/origin/size constructors (O4 dedup sink)
#include "mgl_blit_plan.h"      // depth/stencil blit gates (O4.4)

/* MGLBlitColorState now lives in mgl_blit_color_state.h (log 136) so the C
 * color paths can take it; its two `id` fields are opaque handles there, so
 * the Objective-C uses below bridge them explicitly. */

static MGLRenderTextureInfo mglBlitTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
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

static id mglBlitCreateRenderEncoder(
    MGLRenderPassManager *renderPassManager,
    const MGLRenderPassState *state)
{
    if (!state) return nil;
    void *encoder = NULL;
    if (mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
            renderPassManager->state->currentCommandBufferOwner,
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

@implementation MGLRenderer (Blit)
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
    BOOL copyTypeOk =
        sampledCopy &&
        (expectedType == 0 || mglBlitTextureInfo(sampledCopy).texture_type == expectedType) &&
        mglTexturePixelFormatCompatibleWithExpectedDataKind(mglBlitTextureInfo(sampledCopy).pixel_format, expectedKind);
    if (sampledCopy && mglGLSampledCopyContentFresh(tex) && copyTypeOk) {
        return sampledCopy;
    }

    BOOL isFbAttachment = mglTextureIsAttachmentOfFramebuffer(_renderPassManager->state->renderPassFramebuffer, tex);

    /* Feedback sampling of a color attachment mid-pass must keep the
     * pre-pass Y-flip copy.  Rebuilding from the live RT between drawArrays
     * (version miss after MarkRenderTargetWritten) feeds already-written
     * texels back into later draws and breaks KHR-GL46.texture_barrier
     * same-texel-rw (cover-once across multiple draws, no barrier).
     * glTextureBarrier / end_render_pass refresh the copy instead. */
    if (isFbAttachment && copyTypeOk) {
        if (mglTraceLogIsEnabled()) {
            mglTraceLog("RT_SAMPLE_COPY_REPAIR_KEEP stage=%s program=%u binding=%u unit=%u tex=%u label=\"%s\" reason=fb-attachment-prepass-copy writeVer=%u rtVer=%u",
                        stage ? stage : "",
                        (unsigned)programName,
                        (unsigned)binding,
                        (unsigned)unit,
                        (unsigned)tex->name,
                        mglTraceTextureLabel(tex),
                        (unsigned)tex->mtl_gl_sampled_write_version,
                        (unsigned)tex->mtl_render_target_write_version);
        }
        return sampledCopy;
    }

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
            _renderPassManager->state->currentRenderEncoderOwner) == 1;
    if (hadRenderEncoder) {
        mglRendererEndRenderEncodingLocked((__bridge void *)self);
    }

    /* texSubImage may leave DIRTY_TEXTURE_DATA after releasing the sampled
     * copy (direct MTL upload skipped/failed).  Rebuild the Y-flip copy from
     * Metal only after flushing CPU backing, otherwise feedback sampling sees
     * the previous clear/RT contents (KHR-GL46.texture_barrier). */
    if ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0 &&
        tex->mtl_data &&
        !tex->metal_data_authoritative) {
        id dirtyMetal = (__bridge id)(tex->mtl_data);
        BOOL flushed = NO;
        if (mglRenderTextureTargetIs2D((uint32_t)tex->target) &&
            mglBlitTextureInfo(dirtyMetal).texture_type == MGLTextureType2D &&
            !mglTextureUploadNeedsSwizzleBake(tex)) {
            flushed = [self uploadFullCPUTextureDataIntoTexture:tex
                                                          metal:dirtyMetal
                                                         reason:"sample_gate_miss_repair.dirty"];
        }
        if (flushed) {
            tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
            if (tex->is_render_target) {
                tex->mtl_render_target_write_version++;
                mglMarkGLSampledCopyLevelDirty(tex, 0u);
            }
        }
    }

    sampledCopy = tex->mtl_gl_sampled_data
        ? (__bridge id)(tex->mtl_gl_sampled_data)
        : nil;
    if (!(sampledCopy &&
          mglGLSampledCopyContentFresh(tex) &&
          (expectedType == 0 || mglBlitTextureInfo(sampledCopy).texture_type == expectedType) &&
          mglTexturePixelFormatCompatibleWithExpectedDataKind(mglBlitTextureInfo(sampledCopy).pixel_format, expectedKind))) {
        source = tex->mtl_data ? (__bridge id)(tex->mtl_data) : nil;
        if (source) {
            (void)mglBlitUpdateGLSampledRenderTargetCopy((__bridge void *)self, tex, (__bridge void *)source, "sample_gate_miss_repair");
        }
        sampledCopy = tex->mtl_gl_sampled_data
            ? (__bridge id)(tex->mtl_gl_sampled_data)
            : nil;
    }

    if (hadRenderEncoder &&
        mglRenderEncoderOwnerHasCurrent(
            _renderPassManager->state->currentRenderEncoderOwner) != 1) {
        if (![self restoreRenderEncoderAfterTextureUploadForDraw:"sample_gate_miss_repair"]) {
            return nil;
        }
    }

    BOOL fresh =
        sampledCopy &&
        mglGLSampledCopyContentFresh(tex) &&
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

/* Depth/stencil blit path for mtlBlitFramebuffer.
 * Handles GL_DEPTH_BUFFER_BIT / GL_STENCIL_BUFFER_BIT via Metal render-pass
 * resolve (MSAA), MTLBlitCommandEncoder (same-size), or scaled depth shader.
 * Returns the updated mask with completed depth/stencil bits cleared. */

/* Multisample resolve for mtlBlitFramebuffer color blit.
 * When the source is multisample and the destination is single-sample,
 * resolves the source to a temporary single-sample texture.
 * Updates *readtexidPtr / *readSubresourcePtr to the resolved texture.
 * Returns NO on failure (caller should return); YES on success. */

/* Integer-color blit paths for mtlBlitFramebuffer.
 * Handles MSAA-resolve and direct-blit for integer pixel formats via
 * resolveIntegerMultisampleTexture: or MTLBlitCommandEncoder.
 * Returns YES if a path was taken (caller should return). */


/* Direct MTLBlitCommandEncoder color copy for mtlBlitFramebuffer.
 * Same-size, same-format, no-flip blit via copyFromTexture:toTexture:. */


void mglRendererBlitFramebuffer(GLMContext glm_ctx,
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
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglBlitFramebufferDispatch((__bridge void *)renderer, glm_ctx, src_x0,
                                   src_y0, src_x1, src_y1, dst_x0, dst_y0,
                                   dst_x1, dst_y1, mask, filter);
    }
    mglRendererBackendEnd(&_backend_lease);
}

/* Texture-to-texture blit path for glCopyTexImage2D / glCopyTexSubImage when
 * the destination texture uses a non-BGRA8-compatible Metal pixel format
 * (depth, integer, packed). Resolves the matching framebuffer attachment
 * (depth attachment for depth-format destinations, color attachment for
 * color-format destinations) and performs a direct GPU blit when the source
 * and destination Metal pixel formats match. Returns YES if the blit
 * succeeded and the caller should return; NO to fall through to the CPU
 * BGRA8 conversion path. */


#pragma mark C interface to mtlCopyImageSubData


/* CPU-to-CPU copy path for mtlCopyImageSubData.
 * Raw memcpy between matching-format textures that both have CPU data.
 * Returns YES if the copy succeeded (caller should return). */






@end
