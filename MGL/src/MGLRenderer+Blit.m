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

    readfbo = glm_ctx->active_state->readbuffer;
    drawfbo = glm_ctx->active_state->framebuffer;

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
        readAttachment = glm_ctx->active_state->read_buffer;
        if (mglRenderDrawBufferIsNone((uint32_t)readAttachment)) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer skipped color blit with GL_READ_BUFFER=GL_NONE");
            return NO;
        }
        readAttachment = (GLenum)mglRenderFBOBlitAttachmentOrColor0(
            (uint32_t)readAttachment,
            isColorAttachment(glm_ctx, readAttachment) ? 1 : 0);

        readFBOAttachment = getFBOAttachment(glm_ctx, readfbo, readAttachment);
        if (!readFBOAttachment) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer read attachment missing");
            return NO;
        }
        readSubresource = mglMetalAttachmentSubresourceForAttachment(readFBOAttachment);
        if (mglRenderTargetIsRenderbuffer((uint32_t)readFBOAttachment->textarget))
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
        drawAttachment = glm_ctx->active_state->draw_buffer;
        if (mglRenderDrawBufferIsNone((uint32_t)drawAttachment)) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer skipped color blit with GL_DRAW_BUFFER=GL_NONE");
            return NO;
        }
        drawAttachment = (GLenum)mglRenderFBOBlitAttachmentOrColor0(
            (uint32_t)drawAttachment,
            isColorAttachment(glm_ctx, drawAttachment) ? 1 : 0);

        drawFBOAttachment = getFBOAttachment(glm_ctx, drawfbo, drawAttachment);
        if (!drawFBOAttachment) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer draw attachment missing");
            return NO;
        }
        drawSubresource = mglMetalAttachmentSubresourceForAttachment(drawFBOAttachment);
        if (mglRenderTargetIsRenderbuffer((uint32_t)drawFBOAttachment->textarget))
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
        /* The texture may already have a sampled-only Metal backing from
         * glTexStorage.  Setting is_render_target above changes the required
         * Metal usage even when mtl_data is otherwise clean, so always run the
         * binding transition before creating the blit encoder. */
        if (![self bindMTLTexture:drawTextureObject]) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to bind draw texture to Metal");
            return NO;
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
    st->readtexid = (__bridge void *)readtexid;
    st->drawtexid = (__bridge void *)drawtexid;
    *outReadAttachment = readAttachment;
    return YES;
}

/* Multisample resolve for mtlBlitFramebuffer color blit.
 * When the source is multisample and the destination is single-sample,
 * resolves the source to a temporary single-sample texture.
 * Updates *readtexidPtr / *readSubresourcePtr to the resolved texture.
 * Returns NO on failure (caller should return); YES on success. */

/* Integer-color blit paths for mtlBlitFramebuffer.
 * Handles MSAA-resolve and direct-blit for integer pixel formats via
 * resolveIntegerMultisampleTexture: or MTLBlitCommandEncoder.
 * Returns YES if a path was taken (caller should return). */

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
    id readtexid = (__bridge id)st->readtexid;
    id drawtexid = (__bridge id)st->drawtexid;
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
        const MGLRenderTextureInfo readInfo = mglBlitTextureInfo(readtexid);
        if (readSubresource.depthPlane != 0u) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled source subresource/type unsupported level=%lu slice=%lu depth=%lu type=%lu, skipping",
                  (unsigned long)readSubresource.level,
                  (unsigned long)readSubresource.slice,
                  (unsigned long)readSubresource.depthPlane,
                  (unsigned long)readInfo.texture_type);
            return YES;
        }

        /* The scaled-blit fragment shader consumes texture2d<float>.  A
         * layered/cube framebuffer attachment is backed by an array or cube
         * Metal texture, so expose the selected GL subresource as a single
         * 2D view.  Keep readtexid unchanged for render-target bookkeeping;
         * the local view is retained by the encoder until the command is
         * complete and released automatically at scope end. */
        id scaledReadTexture = readtexid;
        if (readSubresource.level != 0u ||
            readSubresource.slice != 0u ||
            readInfo.texture_type != MGLTextureType2D) {
            const BOOL viewableArraySource =
                readInfo.texture_type == MGLTextureType2DArray ||
                readInfo.texture_type == MGLTextureTypeCube ||
                readInfo.texture_type == MGLTextureTypeCubeArray;
            NSUInteger sliceCount = (NSUInteger)readInfo.array_length;
            if (readInfo.texture_type == MGLTextureTypeCube ||
                readInfo.texture_type == MGLTextureTypeCubeArray) {
                sliceCount *= 6u;
            }
            if (!viewableArraySource ||
                readSubresource.slice >= sliceCount) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer scaled source subresource/type unsupported level=%lu slice=%lu depth=%lu type=%lu slices=%lu, skipping",
                      (unsigned long)readSubresource.level,
                      (unsigned long)readSubresource.slice,
                      (unsigned long)readSubresource.depthPlane,
                      (unsigned long)readInfo.texture_type,
                      (unsigned long)sliceCount);
                return YES;
            }
            scaledReadTexture = mglBlitCreateTextureView(
                readtexid, readInfo.pixel_format, MGLTextureType2D,
                NSMakeRange((NSUInteger)readSubresource.level, 1u),
                NSMakeRange((NSUInteger)readSubresource.slice, 1u));
            if (!scaledReadTexture) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create scaled source 2D view level=%lu slice=%lu type=%lu",
                      (unsigned long)readSubresource.level,
                      (unsigned long)readSubresource.slice,
                      (unsigned long)readInfo.texture_type);
                return YES;
            }
        }

        id pipeline = (__bridge id)mglBlitScaledPipelineForPixelFormat((__bridge void *)self, mglBlitTextureInfo(drawtexid).pixel_format);
        id sampler = (__bridge id)mglBlitScaledSamplerForFilter((__bridge void *)self, filter);
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
            mglBlitCreateRenderEncoder(_renderPassManager, &scaledState);
        if (!encoder) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create scaled render encoder");
            return YES;
        }

        mglBlitSetRenderPipeline(encoder, pipeline);
        mglBlitSetRenderBytes(encoder, &params, sizeof(params),
                              MGL_RENDER_BINDING_STAGE_VERTEX, 0);
        mglBlitSetRenderBytes(encoder, &params, sizeof(params),
                              MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
        mglBlitSetRenderTexture(encoder, scaledReadTexture,
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
        if (glm_ctx && glm_ctx->active_state->caps.scissor_test) {
            NSInteger glScissorX0 = glm_ctx->active_state->var.scissor_box[0];
            NSInteger glScissorY0 = glm_ctx->active_state->var.scissor_box[1];
            NSInteger glScissorX1 = glScissorX0 + glm_ctx->active_state->var.scissor_box[2];
            NSInteger glScissorY1 = glScissorY0 + glm_ctx->active_state->var.scissor_box[3];
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
            (void)mglBlitUpdateGLSampledRenderTargetCopy((__bridge void *)self, drawTextureObject, (__bridge void *)drawtexid, "blit_framebuffer_scaled");
        }
        // When the source is also a render target, refresh its sampled copy
        // so future fragment-shader samples see useCopy=1 instead of falling
        // back to the direct texture (useCopy=0).
        if (readTextureObject &&
            readTextureObject->is_render_target &&
            readtexid) {
            (void)mglBlitUpdateGLSampledRenderTargetCopy((__bridge void *)self, readTextureObject, (__bridge void *)readtexid, "blit_framebuffer_scaled_src");
        }
        return YES;
    }
    return NO;
}

/* Direct MTLBlitCommandEncoder color copy for mtlBlitFramebuffer.
 * Same-size, same-format, no-flip blit via copyFromTexture:toTexture:. */

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

    /* The depth/stencil blit is C now (log 138). */
    mask = mglBlitDepthStencil((__bridge void *)self, glm_ctx, srcX0, srcY0,
                               srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask,
                               filter);

    if (!mglRenderClearMaskHasColor((uint32_t)mask)) {
        if (mglRenderClearMaskHasDepthStencil((uint32_t)mask)) {
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

    if (mglRenderClearMaskHasDepthStencil((uint32_t)mask)) {
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
    GLenum readAttachment = (GLenum)mglRenderEmptyDrawBuffer();
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
    id readtexid = (__bridge id)st.readtexid;
    id drawtexid = (__bridge id)st.drawtexid;

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
        mglRenderClearMaskHasColor((uint32_t)readFBOAttachment->clear_bitmask)) {
        BOOL clearEncoded =
            mglRenderEncodeColorClearForCommandBufferOwner(
                _renderPassManager->state->currentCommandBufferOwner,
                (__bridge void *)readtexid, readSubresource.level,
                readSubresource.slice, readSubresource.depthPlane,
                readFBOAttachment->clear_color[0],
                readFBOAttachment->clear_color[1],
                readFBOAttachment->clear_color[2],
                readFBOAttachment->clear_color[3]) == 0;
        if (clearEncoded) {
            readFBOAttachment->clear_bitmask =
                (GLbitfield)mglRenderClearMaskClearColor(
                    (uint32_t)readFBOAttachment->clear_bitmask);
            mglMarkTextureLevelRenderTargetWritten(readTextureObject, readFBOAttachment->level);
            mglTraceLog("MGL TRACE blitFramebuffer.appliedPendingReadClear fbo=%u attachment=0x%x tex=%u rgba=(%.3f,%.3f,%.3f,%.3f)",
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
        BOOL rgbaBgraPair = mglRenderBlitIsRGBA8BGRA8Pair(
            (uint32_t)mglBlitTextureInfo(readtexid).pixel_format,
            (uint32_t)mglBlitTextureInfo(drawtexid).pixel_format) != 0;

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
    /* The MSAA-resolve path is C now (log 135).  ARC forbids casting the
     * address of a strong local to void**, so the handle travels through a
     * plain void* temporary (the object stays owned by the caller's local). */
    void *readtexidHandle = (__bridge void *)readtexid;
    int didMsaaResolveRaw = didMsaaResolve ? 1 : 0;
    if (!mglBlitResolveMsaaSource((__bridge void *)self, &readtexidHandle,
                                  (__bridge void *)drawtexid, &readSubresource,
                                  srcTexW, srcTexH, readTextureObject,
                                  &didMsaaResolveRaw)) {
        return;
    }
    /* The C entry returns a +1 handle (created or retained); ARC adopts it. */
    readtexid = (__bridge_transfer id)readtexidHandle;
    didMsaaResolve = didMsaaResolveRaw ? YES : NO;

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
            (glm_ctx && glm_ctx->active_state->caps.scissor_test) ? 1 : 0,
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
    BOOL traceBlit = (kMglSwapPresentDiagnostics || traceBlitToFile) &&
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
                        (unsigned)(glm_ctx ? glm_ctx->active_state->draw_buffer : 0u),
                        (unsigned)(glm_ctx ? glm_ctx->active_state->read_buffer : 0u));
        } else {
            mglTraceLog("MGL TRACE blitFramebuffer call=%llu readFBO=%p drawFBO=%p mask=0x%x filter=0x%x "
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
                  (unsigned)(glm_ctx ? glm_ctx->active_state->draw_buffer : 0u),
                  (unsigned)(glm_ctx ? glm_ctx->active_state->read_buffer : 0u));
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
    st.readtexid = (__bridge void *)readtexid;
    st.drawtexid = (__bridge void *)drawtexid;
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

    if (mglBlitIntegerColorWithState((__bridge void *)self, &st)) {
        return;
    }

    if ([self blitFramebufferScaledColorWithState:&st]) {
        return;
    }

    mglBlitDirectColorWithState((__bridge void *)self, &st);
}

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
        [renderer mtlBlitFramebuffer:glm_ctx
                               srcX0:src_x0 srcY0:src_y0
                               srcX1:src_x1 srcY1:src_y1
                               dstX0:dst_x0 dstY0:dst_y0
                               dstX1:dst_x1 dstY1:dst_y1
                                mask:mask filter:filter];
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
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
        return;
    }

    /* Bind the destination texture so we can inspect its Metal pixel format. */
    if (!tex->mtl_data && ![self bindMTLTexture:tex]) {
        NSLog(@"MGL ERROR: mtlCopyTexSubImage failed to bind destination texture %u", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    id destTexture = tex->mtl_data ? (__bridge id)(tex->mtl_data) : nil;
    if (!destTexture) {
        NSLog(@"MGL ERROR: mtlCopyTexSubImage destination texture %u has no Metal texture", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    /* Fast path: try a direct GPU texture-to-texture blit from the matching
     * framebuffer attachment. This handles glCopyTexImage2D/glCopyTexSubImage
     * for depth, integer, and packed internal formats where the source FBO
     * attachment shares the same Metal pixel format as the destination
     * texture. For BGRA8/RGBA8 destinations the CPU path below is sufficient,
     * so we skip the blit attempt to avoid unnecessary encoder churn. */
    BOOL destIsPlainBGRA8 = mglRenderPixelFormatIsUnorm8Color(
        (uint32_t)mglBlitTextureInfo(destTexture).pixel_format) != 0;
    if (!destIsPlainBGRA8) {
        BOOL blitted = mglBlitCopyTexSubImageViaTextureBlit(
            (__bridge void *)self, glm_ctx, tex, (__bridge void *)destTexture,
            slice, level, xoffset, yoffset, x, y, width, height);
        if (blitted) {
            return;
        }
        /* Fall through to the BGRA8 path if the blit was not applicable. */
    }

    if (width > (NSUInteger)(SIZE_MAX / 4u)) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
        return;
    }
    size_t bgraRowBytes = (size_t)width * 4u;
    if (height > 0u && bgraRowBytes > SIZE_MAX / (size_t)height) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
        return;
    }
    size_t bgraSize = bgraRowBytes * (size_t)height;

    NSMutableData *bgraReadback = [NSMutableData dataWithLength:bgraSize];
    NSMutableData *uploadData = [NSMutableData dataWithLength:bgraSize];
    if (!bgraReadback || !uploadData) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
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
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    if (level >= mglBlitTextureInfo(texture).mipmap_level_count) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
        return;
    }

    NSUInteger levelWidth = mglMetalTextureLevelDimension(mglBlitTextureInfo(texture).width, level);
    NSUInteger levelHeight = mglMetalTextureLevelDimension(mglBlitTextureInfo(texture).height, level);
    NSUInteger levelDepth = mglMetalTextureLevelDimension(mglBlitTextureInfo(texture).depth, level);
    if ((NSUInteger)xoffset > levelWidth ||
        (NSUInteger)yoffset > levelHeight ||
        width > levelWidth - (NSUInteger)xoffset ||
        height > levelHeight - (NSUInteger)yoffset) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
        return;
    }

    uint32_t textureType = mglBlitTextureInfo(texture).texture_type;
    NSUInteger destinationSlice = slice;
    NSUInteger copyDepth = 1u;
    MGLOriginValue destinationOrigin = mglBlitOrigin((NSUInteger)xoffset, 0u, 0u);
    if (textureType == MGLTextureType3D) {
        if (slice >= levelDepth) {
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
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
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
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
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    id uploadBuffer = mglBlitCreateBufferWithBytes(
        _device, uploadData.bytes, bgraSize, MGLResourceStorageModeShared);
    if (!uploadBuffer) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
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
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    mglMarkTextureLevelMetalFilled(tex, (GLuint)level, bgraSize);
    (void)mglBlitUpdateGLSampledRenderTargetCopy((__bridge void *)self, tex, (__bridge void *)texture, "copy_tex_sub_image");
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    if (glm_ctx) {
        mglMarkRendererDirtyBits(&glm_ctx->state,
                                 DIRTY_TEX | DIRTY_TEX_BINDING);
    }
}

#pragma mark C interface to mtlCopyImageSubData


/* CPU-to-CPU copy path for mtlCopyImageSubData.
 * Raw memcpy between matching-format textures that both have CPU data.
 * Returns YES if the copy succeeded (caller should return). */






@end
