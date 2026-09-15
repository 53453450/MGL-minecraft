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

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#include "mgl_render_pass_manager_ops.h"
#include "mgl_blit_color_state.h"
#include "mgl_renderer_ports.h"    /* state areas, writable command buffer */
#include "mgl_renderer_backend.h"  /* device */
#include "mgl_render.h"            /* compute encoders, blit encodes */
#include "mgl_blit_plan.h"         /* blit plan helpers */
#include "mgl_blit_pipelines.h"    /* mglBlitMsaaIntegerResolvePipeline */
#include "mgl_blit_sampled_copy.h" /* mglBlitUpdateGLSampledRenderTargetCopy */
#include "mgl_readback.h"          /* integer-color format predicates */
#include "mgl_texture_compat.h"   /* mglMarkTextureLevelRenderTargetWritten */
#include "mgl_state_compat.h"   /* mglNearlyEqual */
#include "mgl_types_texture.h"
#include "mgl_region_value.h"      /* regions / origins / sizes */
#include "error.h"               /* mglDispatchError */
#include "mgl_metal_ref.h"         /* metal reference helpers */
#include "mgl_texture_bind.h"      /* mglRendererBindMTLTexture */
#include "mgl_texture_readback_clear.h" /* pending FBO depth clear */

/* Repeated from MGLRenderer+Blit_Private.h (an Objective-C header a .c file
 * cannot include): the MSAA-integer-resolve compute parameters. */
typedef struct MGLMSAAIntegerResolveParams_t {
    vector_uint2 srcOrigin;
    vector_uint2 dstOrigin;
    vector_uint2 size;
    vector_uint2 _padding;
} MGLMSAAIntegerResolveParams;

/* Repeated from the Objective-C MGLRenderer+Draw_Private.h (a .c file cannot
 * include it): the viewport / scissor value structs. */
typedef struct MGLViewportValue_t {
    double origin_x;
    double origin_y;
    double width;
    double height;
    double znear;
    double zfar;
} MGLViewportValue;

typedef struct MGLScissorRectValue_t {
    uint64_t x;
    uint64_t y;
    uint64_t width;
    uint64_t height;
} MGLScissorRectValue;

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

static void mglBcEndRenderEncodingPort(void *renderer);

static void *mglBcCommandBufferOwner(const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentCommandBufferOwner : NULL;
}

static void mglBcEndRenderEncodingPort(void *renderer)
{
    mglRendererEndRenderEncodingLocked(renderer);
}

/* --- render-pass + render-encoder twins (depth/stencil path) ------------- */

static MGLRenderPassState mglBcDefaultRenderPassState(void)
{
    MGLRenderPassState state;
    mglRenderInitDefaultRenderPassState(&state);
    return state;
}

static MGLRenderPassAttachmentState mglBcRenderPassAttachment(
    void *texture, size_t level, size_t slice, size_t depth_plane,
    uint32_t load_action, uint32_t store_action)
{
    MGLRenderPassAttachmentState attachment = {0};
    attachment.texture = texture;
    attachment.level = level;
    attachment.slice = slice;
    attachment.depth_plane = depth_plane;
    attachment.load_action = load_action;
    attachment.store_action = store_action;
    return attachment;
}

/* The .m statics took the pass manager but only ever used its command-buffer
 * owner, so the twin takes the owner directly (re-read at each use). */
static void *mglBcCreateRenderEncoder(const MGLRendererStateAreas *areas,
                                      const MGLRenderPassState *state)
{
    if (!state) {
        return NULL;
    }
    void *encoder = NULL;
    if (mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
            mglBcCommandBufferOwner(areas), state, &encoder) == 0 &&
        encoder) {
        return encoder;
    }
    return NULL;
}

static void mglBcEndRenderEncoder(void *encoder)
{
    if (!encoder) {
        return;
    }
    (void)mglRenderEndRenderEncoder(encoder);
}

static void mglBcSetRenderPipeline(void *encoder, void *pipeline)
{
    (void)mglRenderSetRenderPipelineState(encoder, pipeline);
}

static void mglBcSetRenderDepthStencilState(void *encoder, void *state)
{
    (void)mglRenderSetRenderDepthStencilState(encoder, state);
}

static void mglBcSetRenderBytes(void *encoder, const void *bytes, size_t length,
                                uint32_t stage, size_t index)
{
    (void)mglRenderSetRenderBytes(encoder, bytes, length, stage,
                                  (uint32_t)index);
}

static void mglBcSetRenderTexture(void *encoder, void *texture, uint32_t stage,
                                  size_t index)
{
    (void)mglRenderSetRenderTexture(encoder, texture, stage, (uint32_t)index);
}

static void mglBcSetRenderSampler(void *encoder, void *sampler, uint32_t stage,
                                  size_t index)
{
    (void)mglRenderSetRenderSampler(encoder, sampler, stage, (uint32_t)index);
}

static void mglBcSetRenderViewport(void *encoder, MGLViewportValue viewport)
{
    (void)mglRenderSetRenderViewport(encoder, viewport.origin_x,
                                     viewport.origin_y, viewport.width,
                                     viewport.height, viewport.znear,
                                     viewport.zfar);
}

static void mglBcSetRenderScissor(void *encoder, MGLScissorRectValue rect)
{
    (void)mglRenderSetRenderScissor(encoder, rect.x, rect.y, rect.width,
                                    rect.height);
}

static void mglBcEncodeDrawPrimitives(void *encoder, uint32_t primitive_type,
                                      size_t vertex_start, size_t vertex_count)
{
    (void)mglRenderEncodeDraw(encoder,
                              &(MGLRenderDrawPlan){
                                  .kind = MGL_RENDER_DRAW_ARRAY,
                                  .primitive_type = (uint32_t)primitive_type,
                                  .vertex_start = vertex_start,
                                  .vertex_count = vertex_count,
                                  .instance_count = 1u,
                                  .base_instance = 0u,
                              },
                              NULL, 0);
}

static void *mglBcCreateTexture(const MGLRenderTextureDescriptorState *desc)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(desc, NULL, &texture) == 0 && texture) {
        return texture;
    }
    return NULL;
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

    if (!mglRenderPassEnsureWritableCommandBufferLocked(
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

/* --- -blitFramebufferIntegerColorWithState: ------------------------------ */

bool mglBlitIntegerColorWithState(void *renderer, const MGLBlitColorState *st)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    void *readtexid = st->readtexid;
    void *drawtexid = st->drawtexid;
    MGLMetalAttachmentSubresource read_subresource = st->readSubresource;
    MGLMetalAttachmentSubresource draw_subresource = st->drawSubresource;
    int64_t copy_w = st->copyW;
    int64_t copy_h = st->copyH;
    int64_t copy_src_x = st->copySrcX;
    int64_t src_metal_y = st->srcMetalY;
    int64_t copy_dst_x = st->copyDstX;
    int64_t dst_metal_y = st->dstMetalY;
    size_t src_tex_w = st->srcTexW;
    size_t src_tex_h = st->srcTexH;
    size_t dst_tex_w = st->dstTexW;
    size_t dst_tex_h = st->dstTexH;
    Texture *read_texture_object = st->readTextureObject;
    Texture *draw_texture_object = st->drawTextureObject;
    FBOAttachment *draw_fbo_attachment = st->drawFBOAttachment;
    int blit_needs_flip = st->blitNeedsFlip;
    double src_w = st->srcW;
    double src_h = st->srcH;
    double dst_w = st->dstW;
    double dst_h = st->dstH;
    if (mglBcTextureInfo(readtexid).sample_count > 1u &&
        mglBcTextureInfo(drawtexid).sample_count <= 1u &&
        mglMetalPixelFormatIsIntegerColor(
            mglBcTextureInfo(readtexid).pixel_format)) {
        if (copy_w <= 0 || copy_h <= 0 || copy_src_x < 0 || src_metal_y < 0 ||
            copy_dst_x < 0 || dst_metal_y < 0 ||
            copy_src_x + copy_w > (int64_t)src_tex_w ||
            src_metal_y + copy_h > (int64_t)src_tex_h ||
            copy_dst_x + copy_w > (int64_t)dst_tex_w ||
            dst_metal_y + copy_h > (int64_t)dst_tex_h) {
            fprintf(stderr,
                    "MGL WARN: mtlBlitFramebuffer integer MSAA resolve invalid "
                    "src=(%ld,%ld %ldx%ld) dst=(%ld,%ld) srcTex=%lux%lu "
                    "dstTex=%lux%lu\n",
                    (long)copy_src_x, (long)src_metal_y, (long)copy_w,
                    (long)copy_h, (long)copy_dst_x, (long)dst_metal_y,
                    (unsigned long)src_tex_w, (unsigned long)src_tex_h,
                    (unsigned long)dst_tex_w, (unsigned long)dst_tex_h);
            return true;
        }

        bool resolved_integer = mglBlitResolveIntegerMultisampleTexture(
            renderer, readtexid, drawtexid,
            mglBlitOrigin((size_t)copy_src_x, (size_t)src_metal_y,
                          read_subresource.depthPlane),
            mglBlitOrigin((size_t)copy_dst_x, (size_t)dst_metal_y,
                          draw_subresource.depthPlane),
            mglBlitSize((size_t)copy_w, (size_t)copy_h, 1u),
            "blitFramebuffer.integerMsaa");
        if (!resolved_integer) {
            fprintf(stderr,
                    "MGL WARN: mtlBlitFramebuffer integer MSAA resolve failed "
                    "fmt=%lu\n",
                    (unsigned long)mglBcTextureInfo(readtexid).pixel_format);
            return true;
        }
        if (draw_texture_object && draw_fbo_attachment) {
            mglMarkTextureLevelRenderTargetWrittenImpl(
                draw_texture_object, draw_fbo_attachment->level,
                "mgl_blit_color_paths.c", __LINE__);
            (void)mglBlitUpdateGLSampledRenderTargetCopy(
                renderer, draw_texture_object, drawtexid,
                "blit_framebuffer_integer_msaa");
        }
        return true;
    }

    if (mglBcTextureInfo(readtexid).sample_count <= 1u &&
        mglBcTextureInfo(drawtexid).sample_count <= 1u &&
        mglBcTextureInfo(readtexid).pixel_format ==
            mglBcTextureInfo(drawtexid).pixel_format &&
        mglMetalPixelFormatIsIntegerColor(
            mglBcTextureInfo(readtexid).pixel_format) &&
        !blit_needs_flip && mglNearlyEqual(src_w, dst_w) &&
        mglNearlyEqual(src_h, dst_h)) {
        if (copy_w <= 0 || copy_h <= 0 || copy_src_x < 0 || src_metal_y < 0 ||
            copy_dst_x < 0 || dst_metal_y < 0 ||
            copy_src_x + copy_w > (int64_t)src_tex_w ||
            src_metal_y + copy_h > (int64_t)src_tex_h ||
            copy_dst_x + copy_w > (int64_t)dst_tex_w ||
            dst_metal_y + copy_h > (int64_t)dst_tex_h) {
            fprintf(stderr,
                    "MGL WARN: mtlBlitFramebuffer integer direct blit invalid "
                    "src=(%ld,%ld %ldx%ld) dst=(%ld,%ld) srcTex=%lux%lu "
                    "dstTex=%lux%lu\n",
                    (long)copy_src_x, (long)src_metal_y, (long)copy_w,
                    (long)copy_h, (long)copy_dst_x, (long)dst_metal_y,
                    (unsigned long)src_tex_w, (unsigned long)src_tex_h,
                    (unsigned long)dst_tex_w, (unsigned long)dst_tex_h);
            return true;
        }

        void *integer_blit =
            mglRenderCreateBlitEncoderBorrowed(mglBcCommandBufferOwner(&areas));
        if (!integer_blit) {
            fprintf(stderr,
                    "MGL WARN: mtlBlitFramebuffer failed to create integer direct "
                    "blit encoder\n");
            return true;
        }
        if (read_texture_object && read_texture_object->is_render_target) {
            mglBcSynchronizeTexture(integer_blit, readtexid,
                                    read_subresource.slice,
                                    read_subresource.level);
        }
        mglBcCopyTexture(
            integer_blit, readtexid, read_subresource.slice,
            read_subresource.level,
            mglBlitOrigin((size_t)copy_src_x, (size_t)src_metal_y,
                          read_subresource.depthPlane),
            mglBlitSize((size_t)copy_w, (size_t)copy_h, 1u), drawtexid,
            draw_subresource.slice, draw_subresource.level,
            mglBlitOrigin((size_t)copy_dst_x, (size_t)dst_metal_y,
                          draw_subresource.depthPlane));
        mglBcEndBlitEncoder(integer_blit);
        if (draw_texture_object && draw_fbo_attachment) {
            mglMarkTextureLevelRenderTargetWrittenImpl(
                draw_texture_object, draw_fbo_attachment->level,
                "mgl_blit_color_paths.c", __LINE__);
            (void)mglBlitUpdateGLSampledRenderTargetCopy(
                renderer, draw_texture_object, drawtexid,
                "blit_framebuffer_integer_direct");
        }
        return true;
    }
    return false;
}

/* --- -blitFramebufferDepthStencil:… -------------------------------------- */

GLbitfield mglBlitDepthStencil(void *renderer, GLMContext glm_ctx, GLint src_x0,
                               GLint src_y0, GLint src_x1, GLint src_y1,
                               GLint dst_x0, GLint dst_y0, GLint dst_x1,
                               GLint dst_y1, GLbitfield mask, GLenum filter)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    GLbitfield depth_stencil_mask =
        (GLbitfield)mglRenderClearMaskDepthStencilBits((uint32_t)mask);
    if (depth_stencil_mask != 0u && glm_ctx->active_state->readbuffer &&
        glm_ctx->active_state->framebuffer) {
        Framebuffer *depth_read_fbo = glm_ctx->active_state->readbuffer;
        Framebuffer *depth_draw_fbo = glm_ctx->active_state->framebuffer;
        FBOAttachment *depth_read_attachment =
            mglRenderClearMaskHasDepth((uint32_t)depth_stencil_mask)
                ? &depth_read_fbo->depth
                : &depth_read_fbo->stencil;
        FBOAttachment *depth_draw_attachment =
            mglRenderClearMaskHasDepth((uint32_t)depth_stencil_mask)
                ? &depth_draw_fbo->depth
                : &depth_draw_fbo->stencil;
        Texture *depth_read_object =
            mglRendererAttachmentTextureFor(glm_ctx, depth_read_attachment);
        Texture *depth_draw_object =
            mglRendererAttachmentTextureFor(glm_ctx, depth_draw_attachment);

        if (depth_read_object && depth_draw_object &&
            mglRendererBindMTLTexture(renderer, depth_read_object) &&
            mglRendererBindMTLTexture(renderer, depth_draw_object)) {
            void *depth_read_texture = depth_read_object->mtl_data;
            void *depth_draw_texture = depth_draw_object->mtl_data;
            MGLMetalAttachmentSubresource depth_read_subresource =
                mglMetalAttachmentSubresourceForAttachment(depth_read_attachment);
            MGLMetalAttachmentSubresource depth_draw_subresource =
                mglMetalAttachmentSubresourceForAttachment(depth_draw_attachment);
            const MGLRenderTextureInfo ds_read_info =
                mglBcTextureInfo(depth_read_texture);
            const MGLRenderTextureInfo ds_draw_info =
                mglBcTextureInfo(depth_draw_texture);

            /* Which of the three depth/stencil paths this rectangle and these
             * textures allow, plus the scissor-clipped copy rectangle: the
             * gates live in the plan (O4.4). */
            MGLBlitDSInput ds_in;
            mglBlitFillDSTextureInput(
                &ds_in, (uint32_t)ds_read_info.pixel_format,
                (uint32_t)ds_draw_info.pixel_format,
                (uint32_t)ds_read_info.sample_count,
                (uint32_t)ds_draw_info.sample_count,
                (uint32_t)ds_read_info.texture_type,
                (uint32_t)ds_draw_info.texture_type, (uint32_t)ds_read_info.width,
                (uint32_t)ds_read_info.height, (uint32_t)ds_draw_info.width,
                (uint32_t)ds_draw_info.height,
                mglRenderPixelFormatIsPackedDepthStencil(
                    (uint32_t)ds_read_info.pixel_format));
            mglBlitFillDSSubresourceInput(
                &ds_in, (uint32_t)depth_read_subresource.level,
                (uint32_t)depth_read_subresource.slice,
                (uint32_t)depth_read_subresource.depthPlane,
                (uint32_t)depth_draw_subresource.level,
                (uint32_t)depth_draw_subresource.slice,
                (uint32_t)depth_draw_subresource.depthPlane);
            mglBlitFillDSRectInput(
                &ds_in, src_x0, src_y0, src_x1, src_y1, dst_x0, dst_y0, dst_x1,
                dst_y1, glm_ctx->active_state->caps.scissor_test ? 1 : 0,
                glm_ctx->active_state->var.scissor_box[0],
                glm_ctx->active_state->var.scissor_box[1],
                glm_ctx->active_state->var.scissor_box[2],
                glm_ctx->active_state->var.scissor_box[3]);
            mglBlitFillDSMaskInput(
                &ds_in, mglRenderClearMaskHasDepth((uint32_t)depth_stencil_mask),
                mglRenderClearMaskHasStencil((uint32_t)depth_stencil_mask),
                mglRenderFilterIsNearest((uint32_t)filter));
            MGLBlitDSPlan ds_plan = {0};
            if (mglBlitPlanDepthStencil(&ds_in, &ds_plan) != 0) {
                return mask;
            }

            if (depth_read_texture && depth_draw_texture &&
                ds_plan.msaa_resolve) {
                mglBcEndRenderEncodingPort(renderer);
                if (mglRenderPassEnsureWritableCommandBufferLocked(
                        renderer, "mtlBlitFramebuffer.depthMsaaResolve")) {
                    if (ds_plan.resolve_depth) {
                        mglTextureApplyPendingFBODepthClearForReadback(
                            renderer, depth_read_fbo, depth_read_attachment,
                            depth_read_object, depth_read_texture);
                    }

                    const int resolved_any =
                        (ds_plan.resolve_depth || ds_plan.resolve_stencil) ? 1 : 0;

                    if (resolved_any) {
                        MGLRenderPassState resolve_state =
                            mglBcDefaultRenderPassState();
                        if (ds_plan.resolve_depth) {
                            resolve_state.depth.attachment =
                                mglBcRenderPassAttachment(
                                    depth_read_texture, 0u,
                                    depth_read_subresource.slice, 0u,
                                    MGLLoadActionLoad,
                                    MGLStoreActionMultisampleResolve);
                            resolve_state.depth.attachment.resolve_texture =
                                depth_draw_texture;
                            resolve_state.depth.attachment.resolve_slice =
                                depth_draw_subresource.slice;
                            resolve_state.depth.resolve_filter =
                                (uint32_t)MGLMultisampleDepthResolveFilterSample0;
                        }
                        if (ds_plan.resolve_stencil) {
                            resolve_state.stencil.attachment =
                                mglBcRenderPassAttachment(
                                    depth_read_texture, 0u,
                                    depth_read_subresource.slice, 0u,
                                    MGLLoadActionLoad,
                                    MGLStoreActionMultisampleResolve);
                            resolve_state.stencil.attachment.resolve_texture =
                                depth_draw_texture;
                            resolve_state.stencil.attachment.resolve_slice =
                                depth_draw_subresource.slice;
                            resolve_state.stencil.resolve_filter =
                                (uint32_t)MGLMultisampleStencilResolveFilterSample0;
                        }
                        void *resolve_encoder =
                            mglBcCreateRenderEncoder(&areas, &resolve_state);
                        if (resolve_encoder) {
                            mglBcEndRenderEncoder(resolve_encoder);
                            mglMarkTextureLevelRenderTargetWrittenImpl(
                                depth_draw_object, depth_draw_attachment->level,
                                "mgl_blit_color_paths.c", __LINE__);
                            if (ds_plan.resolve_depth) {
                                mask = (GLbitfield)mglRenderClearMaskClearDepth(
                                    (uint32_t)mask);
                            }
                            if (ds_plan.resolve_stencil) {
                                mask = (GLbitfield)mglRenderClearMaskClearStencil(
                                    (uint32_t)mask);
                            }
                        }
                    }
                }
            }

            if (depth_read_texture && depth_draw_texture &&
                (ds_plan.same_size_copy || ds_plan.scaled_render)) {
                if (ds_plan.same_size_copy) {
                    /* Same-size depth blit via MTLBlitCommandEncoder; the plan
                     * already clipped the rectangle by the scissor box. */
                    const GLint copy_dst_x0 = ds_plan.copy_dst_x0;
                    const GLint copy_dst_y0 = ds_plan.copy_dst_y0;
                    const GLint copy_dst_x1 = ds_plan.copy_dst_x1;
                    const GLint copy_dst_y1 = ds_plan.copy_dst_y1;
                    const GLint copy_width = copy_dst_x1 - copy_dst_x0;
                    const GLint copy_height = copy_dst_y1 - copy_dst_y0;
                    const GLint copy_src_x = ds_plan.copy_src_x0;
                    const GLint copy_src_y = ds_plan.copy_src_y0;
                    if (ds_plan.copy_valid) {
                        mglBcEndRenderEncodingPort(renderer);
                        if (mglRenderPassEnsureWritableCommandBufferLocked(
                                renderer, "mtlBlitFramebuffer.depthStencil")) {
                            if (mglRenderClearMaskHasDepth(
                                    (uint32_t)depth_stencil_mask)) {
                                mglTextureApplyPendingFBODepthClearForReadback(
                                    renderer, depth_read_fbo,
                                    depth_read_attachment, depth_read_object,
                                    depth_read_texture);
                                mglTextureApplyPendingFBODepthClearForReadback(
                                    renderer, depth_draw_fbo,
                                    depth_draw_attachment, depth_draw_object,
                                    depth_draw_texture);
                            }
                            void *depth_blit = mglRenderCreateBlitEncoderBorrowed(
                                mglBcCommandBufferOwner(&areas));
                            if (depth_blit) {
                                size_t source_metal_y =
                                    ds_read_info.height -
                                    (size_t)(copy_src_y + copy_height);
                                size_t destination_metal_y =
                                    ds_draw_info.height -
                                    (size_t)(copy_dst_y0 + copy_height);
                                mglBcCopyTexture(
                                    depth_blit, depth_read_texture,
                                    depth_read_subresource.slice,
                                    depth_read_subresource.level,
                                    mglBlitOrigin((size_t)copy_src_x,
                                                  source_metal_y,
                                                  depth_read_subresource.depthPlane),
                                    mglBlitSize((size_t)copy_width,
                                                (size_t)copy_height, 1u),
                                    depth_draw_texture,
                                    depth_draw_subresource.slice,
                                    depth_draw_subresource.level,
                                    mglBlitOrigin((size_t)copy_dst_x0,
                                                  destination_metal_y,
                                                  depth_draw_subresource.depthPlane));
                                mglBcEndBlitEncoder(depth_blit);
                                mglMarkTextureLevelRenderTargetWrittenImpl(
                                    depth_draw_object,
                                    depth_draw_attachment->level,
                                    "mgl_blit_color_paths.c", __LINE__);
                            }
                        }
                    }
                } else {
                    /* Scaled depth blit via render pass with depth-writing
                     * shader.  Only GL_NEAREST is supported (GL_LINEAR for depth
                     * is not allowed by the GL spec; filter must be GL_NEAREST
                     * when depth/stencil is in the mask). */
                    if (ds_plan.scaled_render) {
                        /* Apply pending depth clears before the scaled blit so
                         * the source texture reflects any lazy glClear. */
                        if (ds_in.has_depth) {
                            mglBcEndRenderEncodingPort(renderer);
                            if (mglRenderPassEnsureWritableCommandBufferLocked(
                                    renderer,
                                    "mtlBlitFramebuffer.depthScaledClear")) {
                                mglTextureApplyPendingFBODepthClearForReadback(
                                    renderer, depth_read_fbo,
                                    depth_read_attachment, depth_read_object,
                                    depth_read_texture);
                                mglTextureApplyPendingFBODepthClearForReadback(
                                    renderer, depth_draw_fbo,
                                    depth_draw_attachment, depth_draw_object,
                                    depth_draw_texture);
                            }
                        }

                        void *depth_pipeline =
                            mglBlitScaledDepthPipelineForPixelFormat(
                                renderer, mglBcTextureInfo(depth_draw_texture)
                                              .pixel_format);
                        void *sampler = mglBlitScaledSamplerForFilter(
                            renderer, (GLuint)mglRenderNearestFilter());
                        if (depth_pipeline && sampler) {
                            mglBcEndRenderEncodingPort(renderer);
                            if (mglRenderPassEnsureWritableCommandBufferLocked(
                                    renderer, "mtlBlitFramebuffer.depthScaled")) {
                                /* For packed depth+stencil formats, also set the
                                 * stencil attachment to the same texture so
                                 * Metal preserves the stencil component during
                                 * the render pass. */
                                int is_packed_depth_stencil =
                                    mglRenderPixelFormatIsPackedDepthStencil(
                                        (uint32_t)ds_draw_info.pixel_format);

                                MGLRenderPassState scaled_depth_state =
                                    mglBcDefaultRenderPassState();
                                scaled_depth_state.depth.attachment =
                                    mglBcRenderPassAttachment(
                                        depth_draw_texture, 0u, 0u, 0u,
                                        MGLLoadActionLoad, MGLStoreActionStore);
                                if (is_packed_depth_stencil) {
                                    scaled_depth_state.stencil.attachment =
                                        mglBcRenderPassAttachment(
                                            depth_draw_texture, 0u, 0u, 0u,
                                            MGLLoadActionLoad,
                                            MGLStoreActionStore);
                                }

                                void *depth_encoder =
                                    mglBcCreateRenderEncoder(&areas,
                                                             &scaled_depth_state);
                                if (depth_encoder) {
                                    mglBcSetRenderPipeline(depth_encoder,
                                                           depth_pipeline);
                                    mglBcSetRenderDepthStencilState(
                                        depth_encoder,
                                        mglBlitClearRectDepthState(renderer));

                                    /* Compute UVs for the source region in
                                     * Metal's texture coordinate space
                                     * (Y-flipped). */
                                    size_t src_tex_w = ds_read_info.width;
                                    size_t src_tex_h = ds_read_info.height;
                                    float inv_src_w = src_tex_w
                                                          ? (1.0f / (float)src_tex_w)
                                                          : 0.0f;
                                    float inv_src_h = src_tex_h
                                                          ? (1.0f / (float)src_tex_h)
                                                          : 0.0f;
                                    float src_min_xf = (float)src_x0;
                                    float src_max_xf = (float)src_x1;
                                    float src_min_yf = (float)src_y0;
                                    float src_max_yf = (float)src_y1;
                                    float uv_left = MAX(0.0f, MIN(1.0f, src_min_xf * inv_src_w));
                                    float uv_right = MAX(0.0f, MIN(1.0f, src_max_xf * inv_src_w));
                                    /* Metal Y is top-down; GL Y is bottom-up.
                                     * uvTop maps to the top of the source region
                                     * in Metal space: (srcTexH - srcMaxY). */
                                    float uv_top = MAX(0.0f, MIN(1.0f, (float)((double)src_tex_h - src_max_yf) * inv_src_h));
                                    float uv_bottom = MAX(0.0f, MIN(1.0f, (float)((double)src_tex_h - src_min_yf) * inv_src_h));

                                    MGLScaledBlitParams params;
                                    params.uvRect = (vector_float4){
                                        uv_left, uv_top, uv_right, uv_bottom};
                                    params.forceOpaqueAlpha = 0.0f;
                                    params._padding =
                                        (vector_float3){0.0f, 0.0f, 0.0f};

                                    mglBcSetRenderBytes(
                                        depth_encoder, &params, sizeof(params),
                                        MGL_RENDER_BINDING_STAGE_VERTEX, 0);
                                    mglBcSetRenderBytes(
                                        depth_encoder, &params, sizeof(params),
                                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                                    mglBcSetRenderTexture(
                                        depth_encoder, depth_read_texture,
                                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
                                    mglBcSetRenderSampler(
                                        depth_encoder, sampler,
                                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);

                                    /* Set the viewport to the destination region
                                     * in Metal's coordinate space (Y-flipped). */
                                    float dst_min_xf = (float)dst_x0;
                                    float dst_max_xf = (float)dst_x1;
                                    float dst_min_yf = (float)dst_y0;
                                    float dst_max_yf = (float)dst_y1;
                                    size_t dst_tex_w = ds_draw_info.width;
                                    size_t dst_tex_h = ds_draw_info.height;
                                    double dst_min_xd = fmin(dst_min_xf, dst_max_xf);
                                    double dst_max_xd = fmax(dst_min_xf, dst_max_xf);
                                    double dst_min_yd = fmin(dst_min_yf, dst_max_yf);
                                    double dst_max_yd = fmax(dst_min_yf, dst_max_yf);
                                    double dst_wd = dst_max_xd - dst_min_xd;
                                    double dst_hd = dst_max_yd - dst_min_yd;
                                    double scaled_dst_metal_y =
                                        (double)dst_tex_h - dst_max_yd;

                                    /* Scissor rect to limit writes to the
                                     * destination region. */
                                    int64_t scissor_x0 = (int64_t)floor(dst_min_xd + 0.00001);
                                    int64_t scissor_x1 = (int64_t)ceil(dst_max_xd - 0.00001);
                                    int64_t scissor_y0 = (int64_t)floor(scaled_dst_metal_y + 0.00001);
                                    int64_t scissor_y1 = (int64_t)ceil(scaled_dst_metal_y + dst_hd - 0.00001);
                                    scissor_x0 = MAX((int64_t)0, MIN(scissor_x0, (int64_t)dst_tex_w));
                                    scissor_x1 = MAX((int64_t)0, MIN(scissor_x1, (int64_t)dst_tex_w));
                                    scissor_y0 = MAX((int64_t)0, MIN(scissor_y0, (int64_t)dst_tex_h));
                                    scissor_y1 = MAX((int64_t)0, MIN(scissor_y1, (int64_t)dst_tex_h));
                                    if (glm_ctx && glm_ctx->active_state->caps.scissor_test) {
                                        int64_t gl_scissor_x0 = glm_ctx->active_state->var.scissor_box[0];
                                        int64_t gl_scissor_y0 = glm_ctx->active_state->var.scissor_box[1];
                                        int64_t gl_scissor_x1 = gl_scissor_x0 + glm_ctx->active_state->var.scissor_box[2];
                                        int64_t gl_scissor_y1 = gl_scissor_y0 + glm_ctx->active_state->var.scissor_box[3];
                                        int64_t metal_scissor_y0 = (int64_t)dst_tex_h - gl_scissor_y1;
                                        int64_t metal_scissor_y1 = (int64_t)dst_tex_h - gl_scissor_y0;
                                        scissor_x0 = MAX(scissor_x0, gl_scissor_x0);
                                        scissor_x1 = MIN(scissor_x1, gl_scissor_x1);
                                        scissor_y0 = MAX(scissor_y0, metal_scissor_y0);
                                        scissor_y1 = MIN(scissor_y1, metal_scissor_y1);
                                    }
                                    if (scissor_x1 > scissor_x0 &&
                                        scissor_y1 > scissor_y0) {
                                        mglBcSetRenderViewport(
                                            depth_encoder,
                                            (MGLViewportValue){
                                                .origin_x = dst_min_xd,
                                                .origin_y = scaled_dst_metal_y,
                                                .width = dst_wd,
                                                .height = dst_hd,
                                                .znear = 0.0,
                                                .zfar = 1.0});
                                        mglBcSetRenderScissor(
                                            depth_encoder,
                                            (MGLScissorRectValue){
                                                .x = (size_t)scissor_x0,
                                                .y = (size_t)scissor_y0,
                                                .width = (size_t)(scissor_x1 - scissor_x0),
                                                .height = (size_t)(scissor_y1 - scissor_y0)});
                                        mglBcEncodeDrawPrimitives(
                                            depth_encoder,
                                            MGLPrimitiveTypeTriangleStrip, 0, 4);
                                    }
                                    mglBcEndRenderEncoder(depth_encoder);
                                    mglMarkTextureLevelRenderTargetWrittenImpl(
                                        depth_draw_object,
                                        depth_draw_attachment->level,
                                        "mgl_blit_color_paths.c", __LINE__);
                                }
                            }
                        } else {
                            static uint64_t s_scaled_depth_blit_skip_count = 0;
                            uint64_t hit = ++s_scaled_depth_blit_skip_count;
                            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                                fprintf(stderr,
                                        "MGL WARN: mtlBlitFramebuffer scaled "
                                        "depth blit unavailable pipeline=%p "
                                        "sampler=%p hit=%llu\n",
                                        depth_pipeline, sampler,
                                        (unsigned long long)hit);
                            }
                        }
                    }
                }
            }
        }
    }
    return mask;
}

/* --- readPixels helper leaves -------------------------------------------- */

/* OWNERSHIP (log 140 rule 13): both entries return a uniform +1 handle — a
 * newly created texture owns one, and the borrowed passthrough gets a retain —
 * so the Objective-C callers adopt the result with __bridge_transfer. */

void *mglBlitResolvedReadbackTexture(void *renderer, void *source_texture,
                                     size_t source_level, size_t source_slice,
                                     size_t source_depth_plane,
                                     const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!source_texture || mglBcTextureInfo(source_texture).sample_count <= 1u) {
        if (source_texture) {
            CFRetain((CFTypeRef)source_texture);
        }
        return source_texture;
    }

    if (source_level != 0u || source_depth_plane != 0u ||
        (mglBcTextureInfo(source_texture).texture_type !=
             MGLTextureType2DMultisample &&
         mglBcTextureInfo(source_texture).texture_type !=
             MGLTextureType2DMultisampleArray)) {
        fprintf(stderr,
                "MGL WARNING: readPixels cannot resolve MSAA texture for %s "
                "level=%lu slice=%lu depth=%lu type=%lu\n",
                reason ? reason : "unknown", (unsigned long)source_level,
                (unsigned long)source_slice, (unsigned long)source_depth_plane,
                (unsigned long)mglBcTextureInfo(source_texture).texture_type);
        mglDispatchError(
            areas.ctx,
            "-[MGLRenderer(Blit) resolvedReadbackTextureForMultisampleTexture:"
            "sourceLevel:sourceSlice:sourceDepthPlane:reason:]",
            (GLenum)mglRenderErrorInvalidOperation());
        return NULL;
    }

    MGLRenderTextureDescriptorState desc = {0};
    desc.texture_type = MGLTextureType2D;
    desc.pixel_format = mglBcTextureInfo(source_texture).pixel_format;
    desc.width = mglBcTextureInfo(source_texture).width;
    desc.height = mglBcTextureInfo(source_texture).height;
    desc.depth = 1;
    desc.mipmap_level_count = 1;
    desc.sample_count = 1;
    desc.array_length = 1;
    desc.usage = MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
    desc.storage_mode = MGLStorageModePrivate;

    void *resolved_texture = mglBcCreateTexture(&desc);
    if (!resolved_texture) {
        fprintf(stderr,
                "MGL WARNING: readPixels failed to allocate MSAA resolve texture "
                "for %s fmt=%lu size=%lux%lu samples=%lu\n",
                reason ? reason : "unknown",
                (unsigned long)mglBcTextureInfo(source_texture).pixel_format,
                (unsigned long)mglBcTextureInfo(source_texture).width,
                (unsigned long)mglBcTextureInfo(source_texture).height,
                (unsigned long)mglBcTextureInfo(source_texture).sample_count);
        mglDispatchError(
            areas.ctx,
            "-[MGLRenderer(Blit) resolvedReadbackTextureForMultisampleTexture:"
            "sourceLevel:sourceSlice:sourceDepthPlane:reason:]",
            (GLenum)mglRenderErrorOutOfMemory());
        return NULL;
    }

    if (!mglRenderPassEnsureWritableCommandBufferLocked(renderer,
                                                    "readPixels.msaaResolve")) {
        mglDispatchError(
            areas.ctx,
            "-[MGLRenderer(Blit) resolvedReadbackTextureForMultisampleTexture:"
            "sourceLevel:sourceSlice:sourceDepthPlane:reason:]",
            (GLenum)mglRenderErrorInvalidOperation());
        mglSafeReleaseMetalObj(&resolved_texture);
        return NULL;
    }

    int resolves_depth = mglMetalPixelFormatIsDepthOrStencil(
        mglBcTextureInfo(source_texture).pixel_format);
    if (mglRenderEncodeMultisampleResolveForCommandBufferOwner(
            mglBcCommandBufferOwner(&areas),
            resolves_depth ? MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH
                           : MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
            source_texture, source_level, source_slice, source_depth_plane,
            resolved_texture, 0, 0, 0,
            resolves_depth
                ? (uint32_t)MGLMultisampleDepthResolveFilterSample0
                : 0u) == 0) {
        return resolved_texture;
    }
    fprintf(stderr,
            "MGL WARNING: readPixels failed to encode MSAA resolve for %s\n",
            reason ? reason : "unknown");
    mglDispatchError(
        areas.ctx,
        "-[MGLRenderer(Blit) resolvedReadbackTextureForMultisampleTexture:"
        "sourceLevel:sourceSlice:sourceDepthPlane:reason:]",
        (GLenum)mglRenderErrorInvalidOperation());
    mglSafeReleaseMetalObj(&resolved_texture);
    return NULL;
}

void *mglBlitDepthFloatTextureForReadback(void *renderer, void *source_texture,
                                          const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!source_texture || mglBcTextureInfo(source_texture).sample_count > 1u ||
        !mglRenderPixelFormatIsPackedDepthStencil(
            (uint32_t)mglBcTextureInfo(source_texture).pixel_format)) {
        if (source_texture) {
            CFRetain((CFTypeRef)source_texture);
        }
        return source_texture;
    }

    MGLRenderTextureDescriptorState desc = {0};
    desc.texture_type = MGLTextureType2D;
    desc.pixel_format = mglRenderDefaultDepthPixelFormat();
    desc.width = mglBcTextureInfo(source_texture).width;
    desc.height = mglBcTextureInfo(source_texture).height;
    desc.depth = 1;
    desc.mipmap_level_count = 1;
    desc.sample_count = 1;
    desc.array_length = 1;
    desc.usage = MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
    desc.storage_mode = MGLStorageModePrivate;
    void *depth_texture = mglBcCreateTexture(&desc);
    if (!depth_texture) {
        mglDispatchError(areas.ctx,
                         "-[MGLRenderer(Blit) depthFloatTextureForDepthStencil"
                         "Readback:reason:]",
                         (GLenum)mglRenderErrorOutOfMemory());
        return NULL;
    }

    void *pipeline = mglBlitScaledDepthPipelineForPixelFormat(
        renderer, mglRenderDefaultDepthPixelFormat());
    void *sampler = mglBlitScaledSamplerForFilter(
        renderer, (GLuint)mglRenderNearestFilter());
    if (!pipeline || !sampler) {
        fprintf(stderr,
                "MGL WARNING: readPixels DS depth extract unavailable for %s "
                "pipeline=%p sampler=%p\n",
                reason ? reason : "unknown", pipeline, sampler);
        mglDispatchError(areas.ctx,
                         "-[MGLRenderer(Blit) depthFloatTextureForDepthStencil"
                         "Readback:reason:]",
                         (GLenum)mglRenderErrorInvalidOperation());
        mglSafeReleaseMetalObj(&depth_texture);
        return NULL;
    }

    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            renderer, "readPixels.depthStencilExtract")) {
        mglDispatchError(areas.ctx,
                         "-[MGLRenderer(Blit) depthFloatTextureForDepthStencil"
                         "Readback:reason:]",
                         (GLenum)mglRenderErrorInvalidOperation());
        mglSafeReleaseMetalObj(&depth_texture);
        return NULL;
    }

    MGLScaledBlitParams params;
    params.uvRect = (vector_float4){0.0f, 0.0f, 1.0f, 1.0f};
    params.forceOpaqueAlpha = 0.0f;
    params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

    MGLRenderPassState pass_state = mglBcDefaultRenderPassState();
    pass_state.depth.attachment = mglBcRenderPassAttachment(
        depth_texture, 0u, 0u, 0u, MGLLoadActionDontCare, MGLStoreActionStore);

    void *encoder = mglBcCreateRenderEncoder(&areas, &pass_state);
    if (!encoder) {
        mglDispatchError(areas.ctx,
                         "-[MGLRenderer(Blit) depthFloatTextureForDepthStencil"
                         "Readback:reason:]",
                         (GLenum)mglRenderErrorInvalidOperation());
        mglSafeReleaseMetalObj(&depth_texture);
        return NULL;
    }

    mglBcSetRenderPipeline(encoder, pipeline);
    mglBcSetRenderDepthStencilState(encoder,
                                    mglBlitClearRectDepthState(renderer));
    mglBcSetRenderBytes(encoder, &params, sizeof(params),
                        MGL_RENDER_BINDING_STAGE_VERTEX, 0);
    mglBcSetRenderBytes(encoder, &params, sizeof(params),
                        MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
    mglBcSetRenderTexture(encoder, source_texture,
                          MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
    mglBcSetRenderSampler(encoder, sampler, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                          0);
    mglBcSetRenderViewport(encoder, (MGLViewportValue){
                                        .origin_x = 0.0,
                                        .origin_y = 0.0,
                                        .width = (double)mglBcTextureInfo(source_texture).width,
                                        .height = (double)mglBcTextureInfo(source_texture).height,
                                        .znear = 0.0,
                                        .zfar = 1.0});
    mglBcEncodeDrawPrimitives(encoder, MGLPrimitiveTypeTriangleStrip, 0, 4);
    mglBcEndRenderEncoder(encoder);

    return depth_texture;
}
