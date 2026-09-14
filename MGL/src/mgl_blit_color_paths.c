/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_color_paths.c — two of the mtlBlitFramebuffer color helpers moved
 * out of MGLRenderer+Blit.m (P0-1, log 136).
 *
 *   -resolveIntegerMultisampleTexture:toTexture:srcOrigin:dstOrigin:size:reason:
 *       -> mglBlitResolveIntegerMultisampleTexture
 *   -blitFramebufferDirectColorCopyWithState:
 *       -> mglBlitDirectColorWithState
 *
 * The shared MGLBlitColorState now lives in mgl_blit_color_state.h (its two
 * `id` fields are opaque handles there).  The compute/blit helpers of the .m
 * are repeated as mglBc* twins over the same C render entries; `NSLog` keeps
 * the same text on fprintf, and `_batching` / the command-buffer owner arrive
 * through the state areas (the owner is re-read at the point of use).
 */

#include <stdio.h>
#include <string.h>

#include "mgl_blit_color_state.h"
#include "mgl_renderer_ports.h"    /* state areas, writable command buffer */
#include "mgl_renderer_backend.h"  /* device */
#include "mgl_render.h"            /* compute encoders, blit encodes */
#include "mgl_blit_plan.h"         /* blit plan helpers */
#include "mgl_blit_pipelines.h"    /* mglBlitMsaaIntegerResolvePipeline */
#include "mgl_blit_sampled_copy.h" /* mglBlitUpdateGLSampledRenderTargetCopy */
#include "mgl_readback.h"          /* integer-color format predicates */
#include "mgl_texture_compat.h"   /* mglMarkTextureLevelRenderTargetWritten */
#include "mgl_types_texture.h"
#include "mgl_region_value.h"      /* regions / origins / sizes */
#include "error.h"               /* mglDispatchError */
#include "mgl_metal_ref.h"         /* metal reference helpers */

/* Repeated from MGLRenderer+Blit_Private.h (an Objective-C header a .c file
 * cannot include): the MSAA-integer-resolve compute parameters. */
typedef struct MGLMSAAIntegerResolveParams_t {
    vector_uint2 srcOrigin;
    vector_uint2 dstOrigin;
    vector_uint2 size;
    vector_uint2 _padding;
} MGLMSAAIntegerResolveParams;

/* --- twins of the .m statics --------------------------------------------- */

static MGLRenderTextureInfo mglBcTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

static void mglBcSetComputePipeline(void *encoder, void *pipeline)
{
    (void)mglRenderSetComputePipelineState(encoder, pipeline);
}

static void mglBcSetComputeTexture(void *encoder, void *texture, size_t index)
{
    (void)mglRenderSetComputeTexture(encoder, texture, (uint32_t)index);
}

static void mglBcSetComputeBytes(void *encoder, const void *bytes, size_t length,
                                size_t index)
{
    (void)mglRenderSetComputeBytes(encoder, bytes, length, (uint32_t)index);
}

static void mglBcDispatchThreads(void *encoder, MGLSizeValue threads,
                                MGLSizeValue threadgroup)
{
    (void)mglRenderDispatchComputeThreads(
        encoder, (uint32_t)threads.width, (uint32_t)threads.height,
        (uint32_t)threads.depth, (uint32_t)threadgroup.width,
        (uint32_t)threadgroup.height, (uint32_t)threadgroup.depth);
}

static void mglBcEndComputeEncoder(void *encoder)
{
    if (!encoder) {
        return;
    }
    (void)mglRenderEndComputeEncoder(encoder);
}

static void mglBcSynchronizeTexture(void *encoder, void *texture, size_t slice,
                                   size_t level)
{
    (void)mglRenderBlitSynchronizeTexture(encoder, texture, slice, level);
}

static void mglBcCopyTexture(void *encoder, void *source, size_t source_slice,
                             size_t source_level, MGLOriginValue source_origin,
                             MGLSizeValue source_size, void *destination,
                             size_t destination_slice, size_t destination_level,
                             MGLOriginValue destination_origin)
{
    (void)mglRenderBlitCopyTexture(
        encoder, source, source_slice, source_level, source_origin.x,
        source_origin.y, source_origin.z, source_size.width, source_size.height,
        source_size.depth, destination, destination_slice, destination_level,
        destination_origin.x, destination_origin.y, destination_origin.z);
}

static void mglBcEndBlitEncoder(void *encoder)
{
    if (!encoder) {
        return;
    }
    (void)mglRenderEndBlitEncoder(encoder);
}

static void *mglBcCommandBufferOwner(const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentCommandBufferOwner : NULL;
}

/* --- -resolveIntegerMultisampleTexture:… --------------------------------- */

bool mglBlitResolveIntegerMultisampleTexture(void *renderer, void *source_texture,
                                             void *dest_texture,
                                             MGLOriginValue src_origin,
                                             MGLOriginValue dst_origin,
                                             MGLSizeValue size,
                                             const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!source_texture || !dest_texture ||
        mglBcTextureInfo(source_texture).sample_count <= 1u ||
        mglBcTextureInfo(dest_texture).sample_count > 1u ||
        mglBcTextureInfo(source_texture).pixel_format !=
            mglBcTextureInfo(dest_texture).pixel_format ||
        !mglMetalPixelFormatIsIntegerColor(
            mglBcTextureInfo(source_texture).pixel_format) ||
        size.width == 0u || size.height == 0u) {
        return false;
    }

    void *pipeline = mglBlitMsaaIntegerResolvePipeline(
        renderer,
        mglMetalPixelFormatIsSignedIntegerColor(
            mglBcTextureInfo(source_texture).pixel_format));
    if (!pipeline) {
        return false;
    }

    if (!mglRendererEnsureWritableCommandBufferPort(
            renderer, "blitFramebuffer.msaaIntegerResolve")) {
        mglDispatchError(areas.ctx,
                         "-[MGLRenderer(Blit) resolveIntegerMultisampleTexture:"
                         "toTexture:srcOrigin:dstOrigin:size:reason:]",
                         (GLenum)mglRenderErrorInvalidOperation());
        return false;
    }

    void *encoder =
        mglRenderCreateComputeEncoderBorrowed(mglBcCommandBufferOwner(&areas));
    if (!encoder) {
        fprintf(stderr,
                "MGL WARN: failed to create MSAA integer resolve encoder for %s\n",
                reason ? reason : "unknown");
        return false;
    }

    MGLMSAAIntegerResolveParams params;
    params.srcOrigin = (vector_uint2){(uint32_t)src_origin.x,
                                      (uint32_t)src_origin.y};
    params.dstOrigin = (vector_uint2){(uint32_t)dst_origin.x,
                                      (uint32_t)dst_origin.y};
    params.size = (vector_uint2){(uint32_t)size.width, (uint32_t)size.height};
    params._padding = (vector_uint2){0u, 0u};

    mglBcSetComputePipeline(encoder, pipeline);
    mglBcSetComputeTexture(encoder, source_texture, 0);
    mglBcSetComputeTexture(encoder, dest_texture, 1);
    mglBcSetComputeBytes(encoder, &params, sizeof(params), 0);

    MGLSizeValue threads = mglBlitSize(size.width, size.height, 1u);
    size_t w = 16u < (size_t)mglRenderComputePipelineMaxTotalThreads(pipeline)
                   ? 16u
                   : (size_t)mglRenderComputePipelineMaxTotalThreads(pipeline);
    size_t remaining =
        (size_t)mglRenderComputePipelineMaxTotalThreads(pipeline) / (w ? w : 1u);
    size_t h = remaining > 1u ? (remaining < 16u ? remaining : 16u) : 1u;
    MGLSizeValue threadgroup = mglBlitSize(w, h, 1u);
    mglBcDispatchThreads(encoder, threads, threadgroup);
    mglBcEndComputeEncoder(encoder);

    return true;
}

/* --- -blitFramebufferDirectColorCopyWithState: --------------------------- */

void mglBlitDirectColorWithState(void *renderer, const MGLBlitColorState *st)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    Framebuffer *drawfbo = st->drawfbo;
    FBOAttachment *draw_fbo_attachment = st->drawFBOAttachment;
    Texture *read_texture_object = st->readTextureObject;
    Texture *draw_texture_object = st->drawTextureObject;
    MGLMetalAttachmentSubresource read_subresource = st->readSubresource;
    MGLMetalAttachmentSubresource draw_subresource = st->drawSubresource;
    void *readtexid = st->readtexid;
    void *drawtexid = st->drawtexid;
    size_t src_tex_w = st->srcTexW;
    size_t src_tex_h = st->srcTexH;
    size_t dst_tex_w = st->dstTexW;
    size_t dst_tex_h = st->dstTexH;
    int64_t copy_w = st->copyW;
    int64_t copy_h = st->copyH;
    int64_t copy_src_x = st->copySrcX;
    int64_t copy_src_y = st->copySrcY;
    int64_t copy_dst_x = st->copyDstX;
    int64_t copy_dst_y = st->copyDstY;
    int64_t src_metal_y = st->srcMetalY;
    int64_t dst_metal_y = st->dstMetalY;
    int did_msaa_resolve = st->didMsaaResolve;
    /* start blit encoder */
    void *blit_command_encoder =
        mglRenderCreateBlitEncoderBorrowed(mglBcCommandBufferOwner(&areas));
    if (!blit_command_encoder) {
        fprintf(stderr,
                "MGL WARN: mtlBlitFramebuffer failed to create blit encoder\n");
        return;
    }
    if (copy_w <= 0 || copy_h <= 0 || copy_src_x < 0 || copy_src_y < 0 ||
        copy_dst_x < 0 || copy_dst_y < 0 || src_metal_y < 0 ||
        dst_metal_y < 0 || copy_src_x + copy_w > (int64_t)src_tex_w ||
        copy_src_y + copy_h > (int64_t)src_tex_h ||
        copy_dst_x + copy_w > (int64_t)dst_tex_w ||
        copy_dst_y + copy_h > (int64_t)dst_tex_h) {
        mglBcEndBlitEncoder(blit_command_encoder);
        fprintf(stderr,
                "MGL WARN: mtlBlitFramebuffer direct copy invalid after clipping "
                "src=(%ld,%ld %ldx%ld) dst=(%ld,%ld) srcTex=%lux%lu "
                "dstTex=%lux%lu\n",
                (long)copy_src_x, (long)copy_src_y, (long)copy_w, (long)copy_h,
                (long)copy_dst_x, (long)copy_dst_y, (unsigned long)src_tex_w,
                (unsigned long)src_tex_h, (unsigned long)dst_tex_w,
                (unsigned long)dst_tex_h);
        return;
    }

    /* If the source is a render target, ensure all GPU writes are visible
     * before the blit encoder reads it.  Without this synchronizeTexture
     * call, a tile-based Apple GPU may read stale tile memory when the
     * texture was recently written by a preceding render pass. */
    if (read_texture_object && read_texture_object->is_render_target) {
        mglBcSynchronizeTexture(blit_command_encoder, readtexid,
                                read_subresource.slice, read_subresource.level);
    }

    mglBcCopyTexture(
        blit_command_encoder, readtexid, read_subresource.slice,
        read_subresource.level,
        mglBlitOrigin((size_t)copy_src_x, (size_t)src_metal_y,
                      read_subresource.depthPlane),
        mglBlitSize((size_t)copy_w, (size_t)copy_h, 1u), drawtexid,
        draw_subresource.slice, draw_subresource.level,
        mglBlitOrigin((size_t)copy_dst_x, (size_t)dst_metal_y,
                      draw_subresource.depthPlane));
    mglBcEndBlitEncoder(blit_command_encoder);
    if (drawfbo == NULL) {
        areas.core->defaultDrawableWrittenSinceLastSwap = 1;
    }
    if (draw_texture_object && draw_fbo_attachment) {
        /* The ObjC header wraps this in a macro with __FILE__/__LINE__; the C
         * entry takes the caller tag explicitly. */
        mglMarkTextureLevelRenderTargetWrittenImpl(
            draw_texture_object, draw_fbo_attachment->level,
            "mgl_blit_color_paths.c", __LINE__);
        (void)mglBlitUpdateGLSampledRenderTargetCopy(
            renderer, draw_texture_object, drawtexid, "blit_framebuffer_copy");
    }
    /* When the source is also a render target, refresh its sampled copy
     * so future fragment-shader samples use the synchronized copy instead
     * of falling back to the direct texture (useCopy=0). Skip this when we
     * must not become the sampled copy of the (multisample) source object. */
    if (read_texture_object && read_texture_object->is_render_target && readtexid &&
        !did_msaa_resolve) {
        (void)mglBlitUpdateGLSampledRenderTargetCopy(
            renderer, read_texture_object, readtexid,
            "blit_framebuffer_copy_src");
    }
}
