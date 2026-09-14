/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_drivers.c — two self-contained blit paths moved out of
 * MGLRenderer+Blit.m (P0-1, log 135).
 *
 * Mechanical translation: `self` becomes the renderer handle, `_device` /
 * `_renderPassManager->state.currentCommandBufferOwner` / `_batching` arrive
 * through MGLRendererStateAreas, `NSInteger`/`NSUInteger`/`BOOL`/`YES`/`NO`/`nil`
 * become int64_t/size_t/int/1/0/NULL, `NSLog` becomes fprintf on the same sink,
 * and the two `@try/@catch` blocks around the Metal calls become the existing
 * shell guarded call (mgl_gpu_recovery.h) — the C-side home of that frame.
 *
 * OWNERSHIP (log 128 rule): the created textures/buffers are +1 handles kept in
 * locals for the extent of the call, exactly like the ARC locals they replace;
 * `resolveTex` is handed to the caller as a borrowed handle after the encode,
 * which is what the method's +0 return did.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_blit_drivers.h"
#include "mgl_renderer_ports.h"    /* state areas */
#include "mgl_renderer_backend.h"  /* device */
#include "mgl_render.h"            /* blit helpers, texture info */
#include "mgl_blit_plan.h"         /* blit plan helpers */
#include "mgl_texture_compat.h"   /* expansion helpers */
#include "mgl_readback.h"         /* pixel-format predicates */
#include "pixel_utils.h"          /* sizeForInternalFormat */
#include "mgl_render_values.h"    /* storage/usage enums */
#include "mgl_region_value.h"     /* MGLOriginValue / MGLSizeValue / regions */
#include "mgl_gpu_recovery.h"      /* guarded call (@try/@catch) */
#include "mgl_capability.h"        /* MGLCapabilityHasBug + MGL_BUG_* (log 144) */
#include "mgl_renderer_core_state.h" /* core area (capability snapshot) */
#include "mgl_metal_ref.h"        /* mglSafeReleaseMetalObj */
#include "mgl_texture_bind.h"     /* mglRendererBindMTLTexture */
#include "mgl_texture_readback_clear.h" /* pending FBO clear application */
#include "mgl_blit_sampled_copy.h" /* mglBlitUpdateGLSampledRenderTargetCopy */
#include "mgl_trace_log.h"         /* mglTraceLog */
#include "mgl_types_state.h"       /* GL boolean helper */

/* The .m's blit statics, in C (thin wrappers over the C render entries). */
static MGLRenderTextureInfo mglBdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

static void mglBdSynchronizeTexture(void *encoder, void *texture, size_t slice,
                                    size_t level)
{
    (void)mglRenderBlitSynchronizeTexture(encoder, texture, slice, level);
}

static void mglBdCopyTexture(void *encoder, void *source, size_t source_slice,
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

static void mglBdEndBlitEncoder(void *encoder)
{
    if (!encoder) {
        return;
    }
    (void)mglRenderEndBlitEncoder(encoder);
}

static void mglBdCopyBufferToTexture(void *encoder, void *source,
                                     size_t source_offset, size_t bytes_per_row,
                                     size_t bytes_per_image,
                                     MGLSizeValue source_size, void *destination,
                                     size_t destination_slice,
                                     size_t destination_level,
                                     MGLOriginValue destination_origin)
{
    (void)mglRenderBlitCopyBufferToTexture(
        encoder, source, source_offset, bytes_per_row, bytes_per_image,
        source_size.width, source_size.height, source_size.depth, destination,
        destination_slice, destination_level, destination_origin.x,
        destination_origin.y, destination_origin.z);
}

static void mglBdReplaceTextureRegion(void *texture, MGLRegionValue region,
                                     size_t level, size_t slice,
                                     const void *bytes, size_t bytes_per_row,
                                     size_t bytes_per_image, int use_slice)
{
    (void)mglRenderTextureReplaceRegion(
        texture, region.origin.x, region.origin.y, region.origin.z,
        region.size.width, region.size.height, region.size.depth, level, slice,
        bytes, bytes_per_row, bytes_per_image, use_slice ? 1 : 0);
}

/* The command record's current command buffer (read at the point of use). */
static void *mglBdCommandBufferOwner(const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentCommandBufferOwner : NULL;
}

/* The .m's file statics used by these two paths. */
static void *mglBdCreateBufferWithBytes(const void *bytes, size_t length,
                                        uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL, &buffer) ==
            0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

static void *mglBdCreateTexture(const MGLRenderTextureDescriptorState *desc)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(desc, NULL, &texture) == 0 && texture) {
        return texture;
    }
    return NULL;
}

/* -blitFramebufferResolveMsaaSource:drawtexid:readSubresource:srcTexW:srcTexH:
 *  readTextureObject:outDidMsaaResolve: */
bool mglBlitResolveMsaaSource(void *renderer, void **read_texid_ptr,
                              void *draw_texid,
                              MGLMetalAttachmentSubresource *read_subresource_ptr,
                              size_t src_tex_w, size_t src_tex_h,
                              Texture *read_texture_object,
                              int *out_did_msaa_resolve)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    void *read_texid = *read_texid_ptr;
    MGLMetalAttachmentSubresource read_subresource = *read_subresource_ptr;
    int did_msaa_resolve = 0;
    const MGLRenderTextureInfo read_info = mglBdTextureInfo(read_texid);
    const MGLRenderTextureInfo draw_info = mglBdTextureInfo(draw_texid);
    /* Native Metal MSAA, or the AIR FBO path that stores MS planes as a
     * 2DArray (sample_count==1, array_length=GL samples). */
    const int native_msaa = read_info.sample_count > 1u;
    const int emulated_msaa =
        !native_msaa && read_texture_object && read_texture_object->samples > 1u &&
        read_info.texture_type == MGLTextureType2DArray &&
        draw_info.sample_count <= 1u;
    if ((native_msaa || emulated_msaa) && draw_info.sample_count <= 1u &&
        !mglMetalPixelFormatIsIntegerColor(read_info.pixel_format)) {
        MGLRenderTextureDescriptorState resolve_desc = {0};
        resolve_desc.texture_type = MGLTextureType2D;
        resolve_desc.pixel_format = read_info.pixel_format;
        resolve_desc.width = src_tex_w;
        resolve_desc.height = src_tex_h;
        resolve_desc.depth = 1;
        resolve_desc.mipmap_level_count = 1;
        resolve_desc.sample_count = 1;
        resolve_desc.array_length = 1;
        resolve_desc.usage =
            MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
        resolve_desc.storage_mode = MGLStorageModePrivate;
        void *resolve_tex = mglBdCreateTexture(&resolve_desc);
        if (!resolve_tex) {
            fprintf(stderr,
                    "MGL WARN: mtlBlitFramebuffer failed to create MSAA resolve "
                    "texture srcSamples=%lu emulated=%d\n",
                    (unsigned long)(native_msaa
                                        ? read_info.sample_count
                                        : (size_t)read_texture_object->samples),
                    emulated_msaa ? 1 : 0);
            return false;
        }

        int resolve_encoded = 0;
        if (native_msaa) {
            resolve_encoded =
                mglRenderEncodeMultisampleResolveForCommandBufferOwner(
                    mglBdCommandBufferOwner(&areas),
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, read_texid,
                    read_subresource.level, read_subresource.slice,
                    read_subresource.depthPlane, resolve_tex, 0, 0, 0, 0) == 0;
        } else {
            /* Emulated MS: GL NEAREST resolve picks one sample; use plane 0. */
            void *copy_blit =
                mglRenderCreateBlitEncoderBorrowed(mglBdCommandBufferOwner(&areas));
            if (copy_blit) {
                if (read_texture_object->is_render_target) {
                    mglBdSynchronizeTexture(copy_blit, read_texid,
                                              read_subresource.slice,
                                              read_subresource.level);
                }
                mglBdCopyTexture(copy_blit, read_texid, read_subresource.slice,
                                   read_subresource.level,
                                   mglBlitOrigin(0u, 0u, 0u),
                                   mglBlitSize(src_tex_w, src_tex_h, 1u),
                                   resolve_tex, 0u, 0u,
                                   mglBlitOrigin(0u, 0u, 0u));
                mglBdEndBlitEncoder(copy_blit);
                resolve_encoded = 1;
            }
        }
        if (!resolve_encoded) {
            return false;
        }

        /* Synchronize the resolved texture so the subsequent blit/shader can
         * read it on a tile-based Apple GPU without stale tile memory. */
        void *sync_blit =
            mglRenderCreateBlitEncoderBorrowed(mglBdCommandBufferOwner(&areas));
        if (sync_blit) {
            mglBdSynchronizeTexture(sync_blit, resolve_tex, 0, 0);
            mglBdEndBlitEncoder(sync_blit);
        }

        static uint64_t s_msaa_resolve_log_count = 0;
        uint64_t msaa_hit = ++s_msaa_resolve_log_count;
        if (msaa_hit <= 8ull || (msaa_hit % 256ull) == 0ull) {
            mglTraceLog(
                "MGL TRACE blitFramebuffer.msaaResolve hit=%llu srcSamples=%lu "
                "emulated=%d srcTex=%lux%lu srcObj=%u",
                (unsigned long long)msaa_hit,
                (unsigned long)(native_msaa
                                    ? read_info.sample_count
                                    : (size_t)read_texture_object->samples),
                emulated_msaa ? 1 : 0, (unsigned long)src_tex_w,
                (unsigned long)src_tex_h,
                read_texture_object ? (unsigned)read_texture_object->name : 0u);
        }

        /* Replace the source with the resolved single-sample texture. The
         * resolved texture has the same dimensions, so srcTexW/srcTexH remain
         * valid. Reset the subresource to {0,0,0} (fresh 2D texture). */
        read_texid = resolve_tex;
        read_subresource.level = 0u;
        read_subresource.slice = 0u;
        read_subresource.depthPlane = 0u;
        did_msaa_resolve = 1;
    }
    /* OWNERSHIP CONTRACT (log 140): the returned handle always carries a +1.
     * A newly created resolve texture already owns one; the borrowed original
     * gets a retain so both paths look the same to the Objective-C caller,
     * which adopts it with __bridge_transfer.  Without this the created
     * texture's +1 leaked (the caller's __bridge assignment only retained). */
    if (read_texid) {
        CFRetain((CFTypeRef)read_texid);
    }
    *read_texid_ptr = read_texid;
    *read_subresource_ptr = read_subresource;
    *out_did_msaa_resolve = did_msaa_resolve;
    return true;
}

/* --- -readTextureRegionViaBlit:… ----------------------------------------- */

static void mglBdCopyTextureToBuffer(void *encoder, void *source,
                                    size_t source_slice, size_t source_level,
                                    MGLOriginValue source_origin,
                                    MGLSizeValue source_size, void *destination,
                                    size_t destination_offset,
                                    size_t bytes_per_row, size_t bytes_per_image);

typedef struct {
    void *encoder;
    void *texture;
    size_t slice;
    size_t level;
    MGLOriginValue origin;
    MGLSizeValue size;
    void *staging_buffer;
    size_t bytes_per_row;
    size_t bytes_per_image;
    int ended;
} MglBdReadBlitCtx;

static int mglBdReadBlitGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdReadBlitCtx *ctx = (MglBdReadBlitCtx *)ctx_raw;
    mglBdCopyTextureToBuffer(ctx->encoder, ctx->texture, ctx->slice, ctx->level,
                            ctx->origin, ctx->size, ctx->staging_buffer, 0,
                            ctx->bytes_per_row, ctx->bytes_per_image);
    mglBdEndBlitEncoder(ctx->encoder);
    ctx->ended = 1;
    return 1;
}

/* The method's @catch re-ended the encoder inside its own @try/@catch: only a
 * throw can leave it open, and ending it twice is what AGX asserts on, so this
 * runs only when the body did not reach its own end. */
static int mglBdReadBlitCleanupGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdReadBlitCtx *ctx = (MglBdReadBlitCtx *)ctx_raw;
    if (!ctx->ended) {
        mglBdEndBlitEncoder(ctx->encoder);
        ctx->ended = 1;
    }
    return 1;
}

static void *mglBdCreateBuffer(size_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((uint64_t)length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

static void mglBdCopyTextureToBuffer(void *encoder, void *source, size_t source_slice,
                                    size_t source_level, MGLOriginValue source_origin,
                                    MGLSizeValue source_size, void *destination,
                                    size_t destination_offset, size_t bytes_per_row,
                                    size_t bytes_per_image)
{
    (void)mglRenderBlitCopyTextureToBuffer(
        encoder, source, source_slice, source_level, source_origin.x,
        source_origin.y, source_origin.z, source_size.width, source_size.height,
        source_size.depth, destination, destination_offset, bytes_per_row,
        bytes_per_image);
}

/* -readTextureRegionViaBlit:region:slice:level:bytes:bytesPerRow:
 *  bytesPerImage:reason: */
bool mglBlitReadTextureRegion(void *renderer, void *texture,
                              MGLRegionValue region, size_t slice, size_t level,
                              void *bytes, size_t bytes_per_row,
                              size_t bytes_per_image, const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    size_t depth = region.size.depth > 1u ? (size_t)region.size.depth : 1u;
    if (!texture || !bytes || bytes_per_row == 0 || bytes_per_image == 0 ||
        depth > SIZE_MAX / bytes_per_image) {
        return false;
    }

    size_t total_bytes = bytes_per_image * depth;
    void *staging_buffer = mglBdCreateBuffer(
        total_bytes, MGLResourceStorageModeShared);
    if (!staging_buffer) {
        return false;
    }

    mglRendererEndRenderEncodingPort(renderer);
    if (!mglRendererEnsureWritableCommandBufferPort(
            renderer, reason ? reason : "texture_readback_blit")) {
        mglSafeReleaseMetalObj(&staging_buffer);
        return false;
    }

    void *read_encoder = mglRenderCreateBlitEncoderBorrowed(
        mglBdCommandBufferOwner(&areas));
    if (!read_encoder) {
        mglSafeReleaseMetalObj(&staging_buffer);
        return false;
    }
    /* A blit encoder is now active on the current CB.  Mark it as having
     * work so flushCommandBuffer:YES below does not skip the commit. */
    if (areas.batching) {
        areas.batching->currentCommandBufferHasWork = 1;
    }

    MglBdReadBlitCtx blit_ctx = {
        .encoder = read_encoder,
        .texture = texture,
        .slice = slice,
        .level = level,
        .origin = region.origin,
        .size = region.size,
        .staging_buffer = staging_buffer,
        .bytes_per_row = bytes_per_row,
        .bytes_per_image = bytes_per_image,
    };
    if (!mglPlatformShellGuardedCallCtx(renderer, "texture readback blit",
                                        mglBdReadBlitGuarded, &blit_ctx, NULL)) {
        (void)mglPlatformShellGuardedCallCtx(renderer,
                                             "texture readback blit cleanup",
                                             mglBdReadBlitCleanupGuarded,
                                             &blit_ctx, NULL);
        fprintf(stderr,
                "MGL WARNING: texture readback blit failed (%s): caught "
                "exception\n",
                reason ? reason : "texture_readback_blit");
        mglSafeReleaseMetalObj(&staging_buffer);
        return false;
    }

    mglRendererFlushCommandBufferPort(renderer, 1);
    MGLRenderCommandBufferState read_state = {0};
    if (mglRenderWaitCommandBufferOwnerLastSubmitted(
            mglBdCommandBufferOwner(&areas), &read_state) != 0 ||
        read_state.has_error) {
        mglSafeReleaseMetalObj(&staging_buffer);
        return false;
    }
    void *staging_contents = NULL;
    uint64_t staging_length = 0;
    if (mglRenderGetBufferContents(staging_buffer, &staging_contents,
                                   &staging_length) != 0 ||
        !staging_contents || staging_length < total_bytes) {
        mglSafeReleaseMetalObj(&staging_buffer);
        return false;
    }
    memcpy(bytes, staging_contents, total_bytes);
    mglSafeReleaseMetalObj(&staging_buffer);
    return true;
}

/* --- the CPU-to-CPU path ------------------------------------------------- */

typedef struct {
    void *dst_texture;
    MGLRegionValue region;
    size_t level;
    size_t slice;
    const void *src;
    size_t bytes_per_row;
    size_t bytes_per_image;
    int ok;
} MglBdReplaceCtx;

static int mglBdReplaceTextureRegionGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdReplaceCtx *ctx = (MglBdReplaceCtx *)ctx_raw;
    mglBdReplaceTextureRegion(ctx->dst_texture, ctx->region, ctx->level,
                               ctx->slice, ctx->src, ctx->bytes_per_row,
                               ctx->bytes_per_image, 1);
    ctx->ok = 1;
    return 1;
}

/* -copyImageSubDataCpuToCpu:… */
bool mglBlitCopyImageSubDataCpuToCpu(
    void *renderer, GLMContext glm_ctx, Texture *src_tex, void *src_texture,
    uint32_t src_type, GLint src_level, GLint src_x, GLint src_y, GLint src_z,
    Texture *dst_tex, void *dst_texture, uint32_t dst_type, GLint dst_level,
    GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height,
    GLsizei depth)
{
    (void)glm_ctx;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!src_tex->metal_data_authoritative && !src_tex->is_render_target &&
        src_tex->faces && dst_tex->faces &&
        (size_t)src_level < src_tex->num_levels &&
        (size_t)dst_level < dst_tex->num_levels) {
        GLuint src_pixel_size =
            sizeForInternalFormat(src_tex->internalformat, 0, 0);
        GLuint dst_pixel_size =
            sizeForInternalFormat(dst_tex->internalformat, 0, 0);
        if (src_pixel_size > 0 && src_pixel_size == dst_pixel_size &&
            mglBdTextureInfo(src_texture).pixel_format ==
                mglBdTextureInfo(dst_texture).pixel_format) {
            size_t copy_width = (size_t)(width > 1 ? width : 1);
            size_t copy_height = (size_t)(height > 1 ? height : 1);
            size_t row_bytes = copy_width * src_pixel_size;
            size_t num_slices = (size_t)(depth > 1 ? depth : 1);

            bool cpu_copy_ok = true;
            for (size_t s = 0; s < num_slices && cpu_copy_ok; s++) {
                /* Determine src face/level */
                GLuint src_face = 0;
                if (src_type == MGLTextureTypeCube ||
                    src_type == MGLTextureTypeCubeArray) {
                    src_face = ((GLuint)src_z + (GLuint)s) % 6;
                }
                TextureLevel *src_lvl =
                    (src_face < 6 && src_tex->faces[src_face].levels)
                        ? &src_tex->faces[src_face].levels[src_level]
                        : NULL;

                /* Determine dst face/level */
                GLuint dst_face = 0;
                if (dst_type == MGLTextureTypeCube ||
                    dst_type == MGLTextureTypeCubeArray) {
                    dst_face = ((GLuint)dst_z + (GLuint)s) % 6;
                }
                TextureLevel *dst_lvl =
                    (dst_face < 6 && dst_tex->faces[dst_face].levels)
                        ? &dst_tex->faces[dst_face].levels[dst_level]
                        : NULL;

                if (!src_lvl || !dst_lvl || !src_lvl->data || !dst_lvl->data ||
                    src_lvl->width <= 0 || dst_lvl->width <= 0) {
                    cpu_copy_ok = false;
                    break;
                }

                /* For 3D and 2D-array textures, slices are depth planes
                 * within one level.  For cube textures, each slice is a
                 * separate face.  For 2D/rectangle, there is only one
                 * slice. */
                size_t src_slice_pitch =
                    (size_t)src_lvl->pitch * (size_t)(src_lvl->height > 1 ? src_lvl->height : 1);
                size_t dst_slice_pitch =
                    (size_t)dst_lvl->pitch * (size_t)(dst_lvl->height > 1 ? dst_lvl->height : 1);
                bool src_sliced = (src_type == MGLTextureType3D ||
                                   src_type == MGLTextureType2DArray);
                bool dst_sliced = (dst_type == MGLTextureType3D ||
                                   dst_type == MGLTextureType2DArray);
                size_t src_slice_off =
                    src_sliced ? ((size_t)src_z + s) * src_slice_pitch : 0;
                size_t dst_slice_off =
                    dst_sliced ? ((size_t)dst_z + s) * dst_slice_pitch : 0;

                /* Copy region row by row */
                for (size_t y = 0; y < copy_height; y++) {
                    size_t src_off = src_slice_off +
                                     ((size_t)src_y + y) * (size_t)src_lvl->pitch +
                                     (size_t)src_x * src_pixel_size;
                    size_t dst_off = dst_slice_off +
                                     ((size_t)dst_y + y) * (size_t)dst_lvl->pitch +
                                     (size_t)dst_x * dst_pixel_size;
                    if (src_off + row_bytes > src_lvl->data_size ||
                        dst_off + row_bytes > dst_lvl->data_size) {
                        cpu_copy_ok = false;
                        break;
                    }
                    memcpy((uint8_t *)(uintptr_t)dst_lvl->data + dst_off,
                           (const uint8_t *)(uintptr_t)src_lvl->data + src_off,
                           row_bytes);
                }

                if (cpu_copy_ok) {
                    size_t mtl_slice = 0;
                    MGLRegionValue region;
                    if (dst_type == MGLTextureType3D) {
                        mtl_slice = 0;
                        region = mglBlitRegion3D((size_t)dst_x, (size_t)dst_y,
                                                 (size_t)dst_z + s, copy_width,
                                                 copy_height, 1);
                    } else {
                        mtl_slice =
                            (dst_type == MGLTextureTypeCube ||
                             dst_type == MGLTextureTypeCubeArray)
                                ? dst_face
                                : ((size_t)dst_z + s);
                        region = mglBlitRegion2D((size_t)dst_x, (size_t)dst_y,
                                                 copy_width, copy_height);
                    }
                    if (mglBdTextureInfo(dst_texture).storage_mode !=
                        MGLStorageModePrivate) {
                        /* For CPU-backed RGB8-family / RGB16 / RGB32 family
                         * destinations, CPU bpp (3/6/12) != Metal bpp (4/8/16).
                         * The CPU memcpy above preserved the CPU layout, so
                         * expand the copied region to Metal texel layout before
                         * replaceRegion, otherwise N-byte rows are uploaded to a
                         * 4/8/16-byte Metal texture (pixel shift / stripes).
                         * Mirrors the private-storage sibling below. */
                        size_t dst_metal_bpp = mglMetalReadbackBytesPerPixel(
                            mglBdTextureInfo(dst_texture).pixel_format);
                        size_t dst_cpu_bpp =
                            (dst_lvl->width > 0)
                                ? (size_t)(dst_lvl->pitch / dst_lvl->width)
                                : 0;

                        const void *up_src_ptr =
                            (const uint8_t *)(uintptr_t)dst_lvl->data +
                            dst_slice_off;
                        size_t up_bytes_per_row = (size_t)dst_lvl->pitch;
                        size_t up_bytes_per_image = dst_slice_pitch;
                        void *expanded_data = NULL;
                        if (dst_metal_bpp > 0 && dst_cpu_bpp != dst_metal_bpp) {
                            if (mglTextureInternalFormatNeedsRGBA8Expansion(
                                    dst_tex->internalformat,
                                    mglBdTextureInfo(dst_texture)
                                        .pixel_format)) {
                                size_t expanded_bpr = 0, expanded_bpi = 0;
                                expanded_data = mglCreateRGBA8ExpandedUpload(
                                    dst_tex, (const uint8_t *)up_src_ptr,
                                    copy_width, copy_height, up_bytes_per_row,
                                    &expanded_bpr, &expanded_bpi);
                                if (expanded_data) {
                                    up_src_ptr = expanded_data;
                                    up_bytes_per_row = expanded_bpr;
                                    up_bytes_per_image = expanded_bpi;
                                }
                            } else if (mglTextureNeedsChannelExpansion(
                                           dst_tex->internalformat,
                                           mglBdTextureInfo(dst_texture)
                                               .pixel_format)) {
                                size_t expanded_bpr = 0, expanded_bpi = 0;
                                expanded_data = mglCreateChannelExpandedUpload(
                                    dst_tex,
                                    mglBdTextureInfo(dst_texture).pixel_format,
                                    (const uint8_t *)up_src_ptr, copy_width,
                                    copy_height, up_bytes_per_row,
                                    &expanded_bpr, &expanded_bpi);
                                if (expanded_data) {
                                    up_src_ptr = expanded_data;
                                    up_bytes_per_row = expanded_bpr;
                                    up_bytes_per_image = expanded_bpi;
                                }
                            }
                        }
                        MglBdReplaceCtx replace_ctx = {
                            .dst_texture = dst_texture,
                            .region = region,
                            .level = (size_t)dst_level,
                            .slice = mtl_slice,
                            .src = up_src_ptr,
                            .bytes_per_row = up_bytes_per_row,
                            .bytes_per_image = up_bytes_per_image,
                            .ok = 0,
                        };
                        if (!mglPlatformShellGuardedCallCtx(
                                renderer, "cpu-to-cpu texture region replace",
                                mglBdReplaceTextureRegionGuarded, &replace_ctx,
                                NULL)) {
                            fprintf(stderr,
                                    "MGL WARNING: CPU-to-CPU Metal update failed "
                                    "(caught exception)\n");
                        }
                        free(expanded_data);
                    } else {
                        /* Private storage: blit from a staging buffer.  For bpp
                         * mismatch formats (CPU bpp != Metal bpp), expand CPU
                         * data to Metal format before blitting, otherwise
                         * sourceBytesPerRow won't match the Metal texture's
                         * expected row stride. */
                        size_t dst_metal_bpp = mglMetalReadbackBytesPerPixel(
                            mglBdTextureInfo(dst_texture).pixel_format);
                        size_t dst_cpu_bpp =
                            (dst_lvl->width > 0)
                                ? (size_t)(dst_lvl->pitch / dst_lvl->width)
                                : 0;

                        const void *src_ptr =
                            (const uint8_t *)(uintptr_t)dst_lvl->data +
                            dst_slice_off;
                        size_t src_bytes_per_row = (size_t)dst_lvl->pitch;
                        size_t src_image_bytes = src_bytes_per_row * copy_height;

                        void *expanded_data = NULL;
                        if (dst_metal_bpp > 0 && dst_cpu_bpp != dst_metal_bpp) {
                            if (mglTextureInternalFormatNeedsRGBA8Expansion(
                                    dst_tex->internalformat,
                                    mglBdTextureInfo(dst_texture)
                                        .pixel_format)) {
                                size_t expanded_bpr = 0, expanded_bpi = 0;
                                expanded_data = mglCreateRGBA8ExpandedUpload(
                                    dst_tex, (const uint8_t *)src_ptr,
                                    copy_width, copy_height, src_bytes_per_row,
                                    &expanded_bpr, &expanded_bpi);
                                if (expanded_data) {
                                    src_ptr = expanded_data;
                                    src_bytes_per_row = expanded_bpr;
                                    src_image_bytes = expanded_bpi;
                                }
                            } else if (mglTextureNeedsChannelExpansion(
                                           dst_tex->internalformat,
                                           mglBdTextureInfo(dst_texture)
                                               .pixel_format)) {
                                size_t expanded_bpr = 0, expanded_bpi = 0;
                                expanded_data = mglCreateChannelExpandedUpload(
                                    dst_tex,
                                    mglBdTextureInfo(dst_texture).pixel_format,
                                    (const uint8_t *)src_ptr, copy_width,
                                    copy_height, src_bytes_per_row,
                                    &expanded_bpr, &expanded_bpi);
                                if (expanded_data) {
                                    src_ptr = expanded_data;
                                    src_bytes_per_row = expanded_bpr;
                                    src_image_bytes = expanded_bpi;
                                }
                            }
                        }

                        void *staging_buf = mglBdCreateBufferWithBytes(
                            src_ptr, src_image_bytes,
                            MGLResourceStorageModeShared);
                        if (staging_buf) {
                            void *upload_encoder = mglRenderCreateBlitEncoderBorrowed(
                                mglBdCommandBufferOwner(&areas));
                            if (upload_encoder) {
                                mglBdCopyBufferToTexture(
                                    upload_encoder, staging_buf, 0,
                                    src_bytes_per_row, src_image_bytes,
                                    mglBlitSize(copy_width, copy_height, 1),
                                    dst_texture, mtl_slice, (size_t)dst_level,
                                    region.origin);
                                mglBdEndBlitEncoder(upload_encoder);
                            }
                            mglSafeReleaseMetalObj(&staging_buf);
                        }
                        free(expanded_data);
                    }
                }
            }

            if (cpu_copy_ok) {
                /* CPU data is now authoritative for dst level */
                if (dst_tex->faces[0].levels) {
                    dst_tex->faces[0].levels[dst_level].metal_data_authoritative =
                        (GLboolean)mglRenderGLBoolean(0);
                }
                return true;
            }
        }
    }
    return false;
}

/* Declared next to its definition in the Objective-C
 * MGLRenderer+Blit_Private.h; repeated here for the C twin below. */
extern void mglMarkGLSampledCopyLevelDirty(Texture *tex, GLuint level);

/* The ObjC-private header's inline RT Metal-fill marker, in C (same body). */
static void mglBdMarkTextureLevelMetalFilled(Texture *tex, GLuint level,
                                            size_t upload_size)
{
    TextureLevel *tex_level = mglTextureAttachmentLevel(tex, level);
    if (!tex_level) {
        return;
    }
    mglRenderMarkTextureLevelWritten(&tex_level->ever_written,
                                     &tex_level->has_initialized_data,
                                     &tex_level->suspicious_zero_upload);
    tex_level->last_init_source = kTexMetalFill;
    tex_level->last_upload_size = upload_size;
    tex_level->last_src_ptr = NULL;
    tex_level->last_src_hash = 0ull;
    if (tex->is_render_target) {
        tex->mtl_render_target_write_version++;
        mglMarkGLSampledCopyLevelDirty(tex, level);
    }
}

/* --- -mtlCopyTexSubImageViaTextureBlit:… --------------------------------- */

typedef struct {
    void *blit_encoder;
    void *src_texture;
    MGLMetalAttachmentSubresource src_subresource;
    void *dest_texture;
    size_t slice;
    size_t level;
    size_t x;
    size_t src_y;
    size_t xoffset;
    size_t yoffset;
    size_t width;
    size_t height;
    int ended;
} MglBdCopyTexBlitCtx;

static int mglBdCopyTexBlitGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdCopyTexBlitCtx *ctx = (MglBdCopyTexBlitCtx *)ctx_raw;
    mglBdCopyTexture(ctx->blit_encoder, ctx->src_texture,
                     ctx->src_subresource.slice, ctx->src_subresource.level,
                     mglBlitOrigin(ctx->x, ctx->src_y, 0u),
                     mglBlitSize(ctx->width, ctx->height, 1u),
                     ctx->dest_texture, ctx->slice, ctx->level,
                     mglBlitOrigin(ctx->xoffset, ctx->yoffset, 0u));
    mglBdEndBlitEncoder(ctx->blit_encoder);
    ctx->ended = 1;
    return 1;
}

static int mglBdCopyTexBlitCleanupGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdCopyTexBlitCtx *ctx = (MglBdCopyTexBlitCtx *)ctx_raw;
    if (!ctx->ended) {
        mglBdEndBlitEncoder(ctx->blit_encoder);
        ctx->ended = 1;
    }
    return 1;
}

/* -(BOOL)mtlCopyTexSubImageViaTextureBlit:tex:destTexture:slice:level:xoffset:
 *  yoffset:x:y:width:height: */
bool mglBlitCopyTexSubImageViaTextureBlit(
    void *renderer, GLMContext glm_ctx, Texture *tex, void *dest_texture,
    size_t slice, size_t level, int64_t xoffset, int64_t yoffset, int64_t x,
    int64_t y, size_t width, size_t height)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!glm_ctx || !tex || !dest_texture || width == 0u || height == 0u) {
        return false;
    }

    uint32_t dest_format = mglBdTextureInfo(dest_texture).pixel_format;
    int dest_is_depth = mglMetalPixelFormatIsDepthOrStencil(dest_format);

    /* Resolve the source framebuffer attachment. For depth destinations we
     * read from the depth attachment; for color destinations we read from
     * the current read buffer's color attachment. */
    Framebuffer *fbo = glm_ctx->active_state->readbuffer;
    if (!fbo) {
        /* Default framebuffer: not supported via this path. */
        return false;
    }

    FBOAttachment *src_attachment = NULL;
    if (dest_is_depth) {
        src_attachment = &fbo->depth;
    } else {
        GLenum read_buffer = glm_ctx->active_state->read_buffer;
        uint32_t attachment_index = 0u;
        if (!mglRenderDrawBufferIsColorAttachment(
                (uint32_t)read_buffer, (uint32_t)MAX_COLOR_ATTACHMENTS,
                &attachment_index)) {
            return false;
        }
        if (((fbo->color_attachment_bitfield >> attachment_index) & 1u) == 0u) {
            return false;
        }
        src_attachment = &fbo->color_attachments[attachment_index];
    }

    Texture *src_tex_obj =
        mglRendererAttachmentTextureFor(glm_ctx, src_attachment);
    if (!src_tex_obj) {
        return false;
    }
    src_tex_obj->is_render_target = true;
    if (!mglRendererBindMTLTexture(renderer, src_tex_obj) ||
        !src_tex_obj->mtl_data) {
        return false;
    }
    void *src_texture = src_tex_obj->mtl_data;
    if (!src_texture) {
        return false;
    }

    /* Only blit when source and destination Metal pixel formats match. */
    if (mglBdTextureInfo(src_texture).pixel_format != dest_format) {
        return false;
    }

    if (mglRenderTextureIsFramebufferOnly(src_tex_obj->mtl_data)) {
        return false;
    }

    if (level >= mglBdTextureInfo(dest_texture).mipmap_level_count) {
        mglDispatchError(
            glm_ctx,
            "-[MGLRenderer(Blit) mtlCopyTexSubImageViaTextureBlit:tex:"
            "destTexture:slice:level:xoffset:yoffset:x:y:width:height:]",
            (GLenum)mglRenderErrorInvalidValue());
        return true; /* Consumed the call; report an error. */
    }

    size_t dest_level_width = mglMetalTextureLevelDimension(
        mglBdTextureInfo(dest_texture).width, level);
    size_t dest_level_height = mglMetalTextureLevelDimension(
        mglBdTextureInfo(dest_texture).height, level);
    if ((size_t)xoffset > dest_level_width ||
        (size_t)yoffset > dest_level_height ||
        width > dest_level_width - (size_t)xoffset ||
        height > dest_level_height - (size_t)yoffset) {
        mglDispatchError(
            glm_ctx,
            "-[MGLRenderer(Blit) mtlCopyTexSubImageViaTextureBlit:tex:"
            "destTexture:slice:level:xoffset:yoffset:x:y:width:height:]",
            (GLenum)mglRenderErrorInvalidValue());
        return true;
    }

    MGLMetalAttachmentSubresource src_subresource =
        mglMetalAttachmentSubresourceForAttachment(src_attachment);

    /* Metal's texture coordinate origin is top-left, GL's is bottom-left.
     * Flip the source Y so the copied region matches GL semantics. */
    size_t src_level_height = mglMetalTextureLevelDimension(
        mglBdTextureInfo(src_texture).height, src_subresource.level);
    int64_t src_y = (int64_t)src_level_height - (y + (int64_t)height);
    if (src_y < 0) {
        src_y = 0;
    }

    /* End any active render encoder so the blit encoder can run. */
    mglRendererEndRenderEncodingPort(renderer);
    if (!mglRendererEnsureWritableCommandBufferPort(
            renderer, "mtlCopyTexSubImageViaTextureBlit")) {
        mglDispatchError(
            glm_ctx,
            "-[MGLRenderer(Blit) mtlCopyTexSubImageViaTextureBlit:tex:"
            "destTexture:slice:level:xoffset:yoffset:x:y:width:height:]",
            (GLenum)mglRenderErrorInvalidOperation());
        return true;
    }

    /* Apply any pending FBO clear so the source texture has authoritative
     * data before the blit reads from it. */
    if (dest_is_depth) {
        mglTextureApplyPendingFBODepthClearForReadback(
            renderer, fbo, src_attachment, src_tex_obj, src_texture);
    } else {
        GLenum read_buffer = glm_ctx->active_state->read_buffer;
        mglTextureApplyPendingFBOColorClearForReadback(
            renderer, fbo, src_attachment, src_tex_obj, src_texture,
            read_buffer);
    }

    void *blit_encoder =
        mglRenderCreateBlitEncoderBorrowed(mglBdCommandBufferOwner(&areas));
    if (!blit_encoder) {
        mglDispatchError(
            glm_ctx,
            "-[MGLRenderer(Blit) mtlCopyTexSubImageViaTextureBlit:tex:"
            "destTexture:slice:level:xoffset:yoffset:x:y:width:height:]",
            (GLenum)mglRenderErrorInvalidOperation());
        return true;
    }

    MglBdCopyTexBlitCtx blit_ctx = {
        .blit_encoder = blit_encoder,
        .src_texture = src_texture,
        .src_subresource = src_subresource,
        .dest_texture = dest_texture,
        .slice = slice,
        .level = level,
        .x = (size_t)x,
        .src_y = (size_t)src_y,
        .xoffset = (size_t)xoffset,
        .yoffset = (size_t)yoffset,
        .width = width,
        .height = height,
        .ended = 0,
    };
    if (!mglPlatformShellGuardedCallCtx(renderer, "copyTexSubImage texture blit",
                                        mglBdCopyTexBlitGuarded, &blit_ctx,
                                        NULL)) {
        (void)mglPlatformShellGuardedCallCtx(
            renderer, "copyTexSubImage texture blit cleanup",
            mglBdCopyTexBlitCleanupGuarded, &blit_ctx, NULL);
        mglDispatchError(
            glm_ctx,
            "-[MGLRenderer(Blit) mtlCopyTexSubImageViaTextureBlit:tex:"
            "destTexture:slice:level:xoffset:yoffset:x:y:width:height:]",
            (GLenum)mglRenderErrorInvalidOperation());
        return true;
    }

    mglBdMarkTextureLevelMetalFilled(tex, (GLuint)level, 0);
    (void)mglBlitUpdateGLSampledRenderTargetCopy(
        renderer, tex, dest_texture, "copy_tex_sub_image_blit");
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    mglMarkRendererDirtyBits(&glm_ctx->state, DIRTY_TEX | DIRTY_TEX_BINDING);
    return true;
}

/* === copyImageSubData: format conversion + 3D fallback (P0-1, log 144) ====
 *
 * Mechanical translation of -copyImageSubDataFormatConversion:… and
 * -copyImageSubData3DFallback:… out of MGLRenderer+Blit.m.  `self` becomes the
 * renderer handle; `_capability` is the core area's snapshot (the .m reaches it
 * through the `#define _capability _core.capability` alias), the current command
 * buffer owner comes from areas.command, `_batching` from areas.batching.  Each
 * `@try/@catch` frame becomes a shell guarded call, and the NSException object
 * the old NSLog printed is reported as "caught exception".  `__FUNCTION__` now
 * reports these C entries.
 */

/* The .m's helper from MGLRenderer+Blit_Private.h, restated for this TU. */
extern GLboolean mglGetCPUFormatTypeForInternalFormat(GLenum internalformat,
                                                      GLenum *outFormat,
                                                      GLenum *outType);

/* MAX() on size_t, without pulling in a macro. */
static size_t mglBdMaxSize(size_t a, size_t b) { return a > b ? a : b; }

static void mglBdGetTextureBytes(void *texture, void *bytes,
                                 size_t bytes_per_row, size_t bytes_per_image,
                                 MGLRegionValue region, size_t level,
                                 size_t slice, int use_slice)
{
    (void)mglRenderTextureGetBytes(
        texture, bytes, bytes_per_row, bytes_per_image, region.origin.x,
        region.origin.y, region.origin.z, region.size.width, region.size.height,
        region.size.depth, level, slice, use_slice ? 1 : 0);
}

typedef struct {
    void *texture;
    void *bytes;
    size_t bytes_per_row;
    size_t bytes_per_image;
    MGLRegionValue region;
    size_t level;
    size_t slice;
} MglBdGetBytesCtx;

static int mglBdGetBytesGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdGetBytesCtx *ctx = (MglBdGetBytesCtx *)ctx_raw;
    mglBdGetTextureBytes(ctx->texture, ctx->bytes, ctx->bytes_per_row,
                         ctx->bytes_per_image, ctx->region, ctx->level,
                         ctx->slice, 1);
    return 1;
}

typedef struct {
    void *texture;
    MGLRegionValue region;
    size_t level;
    size_t slice;
    const void *bytes;
    size_t bytes_per_row;
    size_t bytes_per_image;
} MglBdReplaceRegionCtx;

static int mglBdReplaceRegionGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdReplaceRegionCtx *ctx = (MglBdReplaceRegionCtx *)ctx_raw;
    mglBdReplaceTextureRegion(ctx->texture, ctx->region, ctx->level, ctx->slice,
                              ctx->bytes, ctx->bytes_per_row,
                              ctx->bytes_per_image, 1);
    return 1;
}

/* -copyImageSubDataFormatConversion:srcTex:srcTexture:srcType:srcLevel:srcX:
 *  srcY:srcZ:dstTex:dstTexture:dstType:dstLevel:dstX:dstY:dstZ:width:height:
 *  depth:
 *
 * The method only ever answered NO for "pixel formats match"; every other exit
 * was YES (handled), so the nested guards collapse into early returns. */
bool mglBlitCopyImageSubDataFormatConversion(
    void *renderer, GLMContext glm_ctx, Texture *src_tex, void *src_texture,
    uint32_t src_type, GLint src_level, GLint src_x, GLint src_y, GLint src_z,
    Texture *dst_tex, void *dst_texture, uint32_t dst_type, GLint dst_level,
    GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height,
    GLsizei depth)
{
    /* Metal-to-Metal copy path for format conversion cases (different Metal
     * pixel formats).  Read source pixels from Metal via getBytes, then write
     * to destination Metal via replaceRegion.  GL CopyImageSubData does raw
     * memcpy of pixel data, so format reinterpretation is OK.  This path has
     * proper render pass synchronization, which the blit path lacks for
     * renderbuffer sources. */
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRendererCoreState *core = areas.core;
    if (mglBdTextureInfo(src_texture).pixel_format ==
        mglBdTextureInfo(dst_texture).pixel_format) {
        return false;
    }
    if (mglBdTextureInfo(dst_texture).storage_mode == MGLStorageModePrivate) {
        return true;
    }
    size_t src_metal_bpp =
        mglMetalReadbackBytesPerPixel(mglBdTextureInfo(src_texture).pixel_format);
    size_t dst_metal_bpp =
        mglMetalReadbackBytesPerPixel(mglBdTextureInfo(dst_texture).pixel_format);
    if (src_metal_bpp == 0 || dst_metal_bpp == 0 ||
        src_metal_bpp != dst_metal_bpp) {
        return true;
    }

    /* Ensure any pending render passes are flushed before reading from the
     * source (especially important for renderbuffers). */
    mglRendererEndRenderEncodingPort(renderer);
    (void)mglRendererSynchronizeRenderPassForTextureReadbackPort(
        renderer, src_texture, "copyImageSubData.formatConv");
    mglRendererFlushCommandBufferPort(renderer, 1);

    size_t copy_width = mglBdMaxSize((size_t)width, 1u);
    size_t copy_height = mglBdMaxSize((size_t)height, 1u);
    size_t num_slices = mglBdMaxSize((size_t)depth, 1u);
    size_t row_bytes = copy_width * src_metal_bpp;
    size_t image_bytes = row_bytes * copy_height;
    void *staging = malloc(image_bytes);
    int metal_copy_ok = (staging != NULL);

    for (size_t s = 0; s < num_slices && metal_copy_ok; s++) {
        /* Read source slice.  Prefer CPU data when available
         * (metal_data_authoritative == false) to avoid AGX getBytes bugs on 3D
         * and 2D-array textures. */
        size_t src_mtl_slice = 0;
        MGLRegionValue src_region;
        if (src_type == MGLTextureType3D) {
            src_mtl_slice = 0;
            src_region = mglBlitRegion3D((size_t)src_x, (size_t)src_y,
                                         (size_t)src_z + s, copy_width,
                                         copy_height, 1);
        } else if (src_type == MGLTextureTypeCube ||
                   src_type == MGLTextureTypeCubeArray) {
            src_mtl_slice = ((size_t)src_z + s) % 6;
            src_region = mglBlitRegion2D((size_t)src_x, (size_t)src_y, copy_width,
                                         copy_height);
        } else {
            src_mtl_slice = (size_t)src_z + s;
            src_region = mglBlitRegion2D((size_t)src_x, (size_t)src_y, copy_width,
                                         copy_height);
        }

        int src_read_from_cpu = 0;
        if (!src_tex->metal_data_authoritative && src_tex->faces &&
            (size_t)src_level < src_tex->num_levels) {
            GLuint src_face = 0;
            if (src_type == MGLTextureTypeCube ||
                src_type == MGLTextureTypeCubeArray) {
                src_face = ((GLuint)src_z + (GLuint)s) % 6;
            }
            TextureLevel *src_lvl =
                (src_face < 6 && src_tex->faces[src_face].levels)
                    ? &src_tex->faces[src_face].levels[src_level]
                    : NULL;
            if (src_lvl && src_lvl->data && src_lvl->pitch > 0 &&
                src_lvl->width > 0) {
                size_t src_cpu_bpp = src_lvl->pitch / src_lvl->width;
                if (src_cpu_bpp == src_metal_bpp) {
                    size_t src_cpu_pitch = src_lvl->pitch;
                    size_t src_cpu_img_size =
                        src_cpu_pitch * mglBdMaxSize(src_lvl->height, 1u);
                    size_t src_cpu_off = 0;
                    if (src_type == MGLTextureType3D) {
                        src_cpu_off = ((size_t)src_z + s) * src_cpu_img_size +
                                      (size_t)src_y * src_cpu_pitch +
                                      (size_t)src_x * src_cpu_bpp;
                    } else if (src_type == MGLTextureType2DArray ||
                               src_type == MGLTextureTypeCubeArray) {
                        /* 2D array: all slices in one TextureLevel */
                        GLuint array_slice = (src_type == MGLTextureTypeCubeArray)
                                                 ? ((GLuint)src_z + (GLuint)s) / 6
                                                 : ((GLuint)src_z + (GLuint)s);
                        src_cpu_off = array_slice * src_cpu_img_size +
                                      (size_t)src_y * src_cpu_pitch +
                                      (size_t)src_x * src_cpu_bpp;
                    } else {
                        src_cpu_off = (size_t)src_y * src_cpu_pitch +
                                      (size_t)src_x * src_cpu_bpp;
                    }
                    size_t last_row_end =
                        src_cpu_off +
                        (copy_height > 0 ? (copy_height - 1) * src_cpu_pitch : 0) +
                        row_bytes;
                    if (last_row_end <= src_lvl->data_size) {
                        for (size_t y = 0; y < copy_height; y++) {
                            memcpy((uint8_t *)staging + y * row_bytes,
                                   (const uint8_t *)(uintptr_t)src_lvl->data +
                                       src_cpu_off + y * src_cpu_pitch,
                                   row_bytes);
                        }
                        src_read_from_cpu = 1;
                    }
                }
            }
        }

        if (!src_read_from_cpu) {
            if (src_type == MGLTextureType3D &&
                MGLCapabilityHasBug(&core->capability,
                                    MGL_BUG_3D_GETBYTES_SLICE_OOB)) {
                if (!mglBlitReadTextureRegion(
                        renderer, src_texture, src_region, src_mtl_slice,
                        (size_t)src_level, staging, row_bytes, image_bytes,
                        "copyImageSubData.formatConv3DReadback")) {
                    metal_copy_ok = 0;
                    break;
                }
            } else {
                MglBdGetBytesCtx read_ctx = {
                    .texture = src_texture,
                    .bytes = staging,
                    .bytes_per_row = row_bytes,
                    .bytes_per_image = image_bytes,
                    .region = src_region,
                    .level = (size_t)src_level,
                    .slice = src_mtl_slice,
                };
                if (!mglPlatformShellGuardedCallCtx(
                        renderer, "format conv renderbuffer readback",
                        mglBdGetBytesGuarded, &read_ctx, NULL)) {
                    fprintf(stderr,
                            "MGL WARNING: format conv renderbuffer readback "
                            "failed: caught exception\n");
                    metal_copy_ok = 0;
                    break;
                }
            }
        }

        /* Write to destination Metal via replaceRegion */
        {
            size_t dst_mtl_slice = 0;
            MGLRegionValue dst_region;
            if (dst_type == MGLTextureType3D) {
                dst_mtl_slice = 0;
                dst_region = mglBlitRegion3D((size_t)dst_x, (size_t)dst_y,
                                             (size_t)dst_z + s, copy_width,
                                             copy_height, 1);
            } else if (dst_type == MGLTextureTypeCube ||
                       dst_type == MGLTextureTypeCubeArray) {
                dst_mtl_slice = ((size_t)dst_z + s) % 6;
                dst_region = mglBlitRegion2D((size_t)dst_x, (size_t)dst_y,
                                             copy_width, copy_height);
            } else {
                dst_mtl_slice = (size_t)dst_z + s;
                dst_region = mglBlitRegion2D((size_t)dst_x, (size_t)dst_y,
                                             copy_width, copy_height);
            }
            MglBdReplaceRegionCtx write_ctx = {
                .texture = dst_texture,
                .region = dst_region,
                .level = (size_t)dst_level,
                .slice = dst_mtl_slice,
                .bytes = staging,
                .bytes_per_row = row_bytes,
                .bytes_per_image = image_bytes,
            };
            if (!mglPlatformShellGuardedCallCtx(
                    renderer, "format conv renderbuffer Metal update",
                    mglBdReplaceRegionGuarded, &write_ctx, NULL)) {
                fprintf(stderr,
                        "MGL WARNING: format conv renderbuffer Metal update "
                        "failed: caught exception\n");
            }
        }

        /* Also update dst CPU data if available */
        if (dst_tex->faces && (size_t)dst_level < dst_tex->num_levels) {
            GLuint dst_face = 0;
            if (dst_type == MGLTextureTypeCube ||
                dst_type == MGLTextureTypeCubeArray) {
                dst_face = ((GLuint)dst_z + (GLuint)s) % 6;
            }
            TextureLevel *cur_dst_lvl =
                (dst_face < 6 && dst_tex->faces[dst_face].levels)
                    ? &dst_tex->faces[dst_face].levels[dst_level]
                    : NULL;
            if (cur_dst_lvl && cur_dst_lvl->data && cur_dst_lvl->pitch > 0 &&
                cur_dst_lvl->width > 0) {
                size_t dst_cpu_bpp = cur_dst_lvl->pitch / cur_dst_lvl->width;
                if (dst_cpu_bpp == dst_metal_bpp) {
                    size_t dst_slice_pitch =
                        cur_dst_lvl->pitch * mglBdMaxSize(cur_dst_lvl->height, 1u);
                    int dst_sliced = (dst_type == MGLTextureType3D ||
                                      dst_type == MGLTextureType2DArray);
                    size_t dst_slice_off =
                        dst_sliced ? ((size_t)dst_z + s) * dst_slice_pitch : 0;
                    for (size_t y = 0; y < copy_height; y++) {
                        size_t dst_off = dst_slice_off +
                                         ((size_t)dst_y + y) * cur_dst_lvl->pitch +
                                         (size_t)dst_x * dst_metal_bpp;
                        if (dst_off + row_bytes <= cur_dst_lvl->data_size) {
                            memcpy((uint8_t *)(uintptr_t)cur_dst_lvl->data +
                                       dst_off,
                                   (const uint8_t *)staging + y * row_bytes,
                                   row_bytes);
                        }
                    }
                }
            }
        }
    }
    free(staging);
    if (metal_copy_ok) {
        /* Do NOT set metal_data_authoritative = GL_TRUE here.
         *
         * Previously, this code set the destination level's
         * metal_data_authoritative flag to force glGetTexImage to read from
         * Metal.  However, this causes failures for destination textures whose
         * Metal data may not be fully initialized (e.g., RGB9_E5 2D-array
         * textures where replaceRegion only updates the copied region, leaving
         * non-copied regions with stale Metal data).
         *
         * glCopyImageSubData does a raw bit copy.  The CPU data was updated
         * above with the source's raw bits at the copy region, and non-copied
         * regions retain their original values from glTexImage*.  This is
         * correct for both memcmp and float-epsilon comparisons used by CTS.
         * Keeping CPU data authoritative avoids AGX Metal readback bugs on 3D
         * and certain packed formats. */
        return true;
    }
    return true;
}

/* The 3D fallback's two @try frames.  The first one carries the reads (and the
 * method's own `return YES` exits, reported through `early_exit`); the second
 * one only wraps the replaceRegion writes.  Neither body frees the staging
 * buffer: the caller owns it and frees once on every path. */
typedef struct {
    void *renderer;
    GLMContext glm_ctx;
    void *src_texture;
    uint32_t src_type;
    GLint src_level;
    GLint src_x;
    GLint src_y;
    GLint src_z;
    size_t copy_width;
    size_t copy_height;
    size_t copy_depth;
    size_t row_bytes;
    size_t image_bytes;
    void *staging;
    MGLRendererStateAreas areas;
    MGLRendererCoreState *core;
    int early_exit;
} MglBd3DReadCtx;

static int mglBd3DReadGuarded(void *renderer, void *ctx_raw)
{
    MglBd3DReadCtx *c = (MglBd3DReadCtx *)ctx_raw;
    /* Read from source Metal texture.  For 3D sources, read the entire 3D
     * region in one call.  For non-3D sources (2D array, cube, etc.), loop
     * over slices and read each slice separately. */
    if (c->src_type == MGLTextureType3D) {
        MGLRegionValue src_region =
            mglBlitRegion3D((size_t)c->src_x, (size_t)c->src_y,
                            (size_t)c->src_z, c->copy_width, c->copy_height,
                            c->copy_depth);
        if (mglBdTextureInfo(c->src_texture).storage_mode !=
                MGLStorageModePrivate &&
            !MGLCapabilityHasBug(&c->core->capability,
                                 MGL_BUG_3D_GETBYTES_SLICE_OOB)) {
            mglBdGetTextureBytes(c->src_texture, c->staging, c->row_bytes,
                                 c->image_bytes, src_region,
                                 (size_t)c->src_level, 0, 1);
        } else if (!mglBlitReadTextureRegion(c->renderer, c->src_texture,
                                             src_region, 0,
                                             (size_t)c->src_level, c->staging,
                                             c->row_bytes, c->image_bytes,
                                             "copyImageSubData.3DReadback")) {
            c->early_exit = 1;
            return 1;
        }
    } else {
        /* Non-3D source (2D array, cube, rectangle, etc.): read each slice
         * separately and place at the correct offset. */
        for (size_t z = 0; z < c->copy_depth; z++) {
            size_t slice_offset = z * c->image_bytes;
            size_t src_slice = (size_t)c->src_z + z;
            MGLRegionValue slice_region =
                mglBlitRegion3D((size_t)c->src_x, (size_t)c->src_y, 0,
                                c->copy_width, c->copy_height, 1u);
            if (mglBdTextureInfo(c->src_texture).storage_mode !=
                MGLStorageModePrivate) {
                mglBdGetTextureBytes(c->src_texture,
                                     (uint8_t *)c->staging + slice_offset,
                                     c->row_bytes, c->image_bytes, slice_region,
                                     (size_t)c->src_level, src_slice, 1);
            } else {
                void *slice_buffer =
                    mglBdCreateBuffer(c->image_bytes,
                                      MGLResourceStorageModeShared);
                if (!slice_buffer) {
                    c->early_exit = 1;
                    return 1;
                }
                void *read_encoder = mglRenderCreateBlitEncoderBorrowed(
                    mglBdCommandBufferOwner(&c->areas));
                if (!read_encoder) {
                    mglSafeReleaseMetalObj(&slice_buffer);
                    c->early_exit = 1;
                    return 1;
                }
                c->areas.batching->currentCommandBufferHasWork = 1;
                mglBdCopyTextureToBuffer(
                    read_encoder, c->src_texture, src_slice,
                    (size_t)c->src_level, slice_region.origin, slice_region.size,
                    slice_buffer, 0, c->row_bytes, c->image_bytes);
                mglBdEndBlitEncoder(read_encoder);
                mglRendererFlushCommandBufferPort(c->renderer, 1);
                void *slice_contents = NULL;
                uint64_t slice_length = 0;
                if (mglRenderGetBufferContents(slice_buffer, &slice_contents,
                                               &slice_length) != 0 ||
                    !slice_contents || slice_length < c->image_bytes) {
                    mglSafeReleaseMetalObj(&slice_buffer);
                    c->early_exit = 1;
                    return 1;
                }
                memcpy((uint8_t *)c->staging + slice_offset, slice_contents,
                       c->image_bytes);
                mglSafeReleaseMetalObj(&slice_buffer);
            }
        }
    }
    return 1;
}

typedef struct {
    void *dst_texture;
    Texture *dst_tex;
    GLint dst_level;
    int bpp_mismatch;
    GLuint level_width;
    GLuint level_height;
    GLuint level_depth;
    size_t level_pitch;
    size_t level_image_bytes;
    const void *full_level_bytes;
} MglBd3DWriteCtx;

static int mglBd3DWriteGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBd3DWriteCtx *c = (MglBd3DWriteCtx *)ctx_raw;
    /* Write the full level back with origin (0,0,0).  For bpp mismatch, expand
     * CPU data to Metal format first. */
    MGLRegionValue full_region = mglBlitRegion3D(
        0, 0, 0, c->level_width, c->level_height, c->level_depth);
    if (c->bpp_mismatch) {
        size_t expanded_bpr = 0;
        size_t expanded_bpi = 0;
        uint8_t *expanded_data = NULL;
        if (mglTextureInternalFormatNeedsRGBA8Expansion(
                c->dst_tex->internalformat,
                mglBdTextureInfo(c->dst_texture).pixel_format)) {
            expanded_data = mglCreateRGBA8ExpandedUpload(
                c->dst_tex, (const uint8_t *)c->full_level_bytes,
                c->level_width, c->level_height * c->level_depth,
                c->level_pitch, &expanded_bpr, &expanded_bpi);
        } else if (mglTextureNeedsChannelExpansion(
                       c->dst_tex->internalformat,
                       mglBdTextureInfo(c->dst_texture).pixel_format)) {
            expanded_data = mglCreateChannelExpandedUpload(
                c->dst_tex, mglBdTextureInfo(c->dst_texture).pixel_format,
                (const uint8_t *)c->full_level_bytes, c->level_width,
                c->level_height * c->level_depth, c->level_pitch, &expanded_bpr,
                &expanded_bpi);
        }
        if (expanded_data) {
            size_t expanded_image_bytes = expanded_bpr * c->level_height;
            mglBdReplaceTextureRegion(c->dst_texture, full_region,
                                      (size_t)c->dst_level, 0, expanded_data,
                                      expanded_bpr, expanded_image_bytes, 1);
            free(expanded_data);
        } else {
            mglBdReplaceTextureRegion(c->dst_texture, full_region,
                                      (size_t)c->dst_level, 0,
                                      c->full_level_bytes, c->level_pitch,
                                      c->level_image_bytes, 1);
        }
    } else {
        mglBdReplaceTextureRegion(c->dst_texture, full_region,
                                  (size_t)c->dst_level, 0, c->full_level_bytes,
                                  c->level_pitch, c->level_image_bytes, 1);
    }
    return 1;
}

/* -copyImageSubData3DFallback:srcTex:srcTexture:srcType:srcLevel:srcX:srcY:
 *  srcZ:dstTex:dstTexture:dstType:dstLevel:dstX:dstY:dstZ:width:height:depth: */
bool mglBlitCopyImageSubData3DFallback(
    void *renderer, GLMContext glm_ctx, Texture *src_tex, void *src_texture,
    uint32_t src_type, GLint src_level, GLint src_x, GLint src_y, GLint src_z,
    Texture *dst_tex, void *dst_texture, uint32_t dst_type, GLint dst_level,
    GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height,
    GLsizei depth)
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
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRendererCoreState *core = areas.core;
    int needs_3d_workaround =
        MGLCapabilityHasBug(&core->capability, MGL_BUG_3D_GETBYTES_SLICE_OOB) ||
        MGLCapabilityHasBug(&core->capability,
                            MGL_BUG_3D_REPLACE_REGION_NONZERO_ORIGIN) ||
        MGLCapabilityHasBug(&core->capability,
                            MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB);
    if (!needs_3d_workaround || dst_type != MGLTextureType3D ||
        mglBdTextureInfo(dst_texture).storage_mode == MGLStorageModePrivate) {
        return false;
    }

    size_t bpp =
        mglMetalReadbackBytesPerPixel(mglBdTextureInfo(src_texture).pixel_format);
    if (bpp == 0u) {
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return true;
    }

    TextureLevel *early_dst_level_info = NULL;
    if (dst_tex->faces && dst_tex->faces[0].levels &&
        (size_t)dst_level < dst_tex->num_levels) {
        early_dst_level_info = &dst_tex->faces[0].levels[dst_level];
    }
    if (!early_dst_level_info || !early_dst_level_info->data ||
        dst_tex->metal_data_authoritative) {
        return false; /* fall through to blit path */
    }
    int bpp_mismatch = 0;
    size_t cpu_bpp = 0;
    {
        size_t early_pitch = early_dst_level_info->pitch;
        if (early_pitch == 0) {
            early_pitch = (size_t)early_dst_level_info->width * bpp;
        }
        cpu_bpp = (early_dst_level_info->width > 0)
                      ? (early_pitch / early_dst_level_info->width)
                      : 0;
        if (cpu_bpp == 0) {
            return false; /* fall through to blit path */
        }
        if (cpu_bpp != bpp) {
            bpp_mismatch = 1;
        }
    }

    size_t copy_width = mglBdMaxSize((size_t)width, 1u);
    size_t copy_height = mglBdMaxSize((size_t)height, 1u);
    size_t copy_depth = mglBdMaxSize((size_t)depth, 1u);
    size_t row_bytes = copy_width * bpp;
    size_t image_bytes = row_bytes * copy_height;
    size_t total_bytes = image_bytes * copy_depth;

    void *staging = malloc(total_bytes);
    if (!staging) {
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorOutOfMemory());
        return true;
    }

    /* Read from source into staging buffer.  Prefer CPU data when available
     * (metal_data_authoritative == false) to avoid Metal getBytes/blit issues
     * with certain texture types.  Fall back to Metal readback for Private
     * textures or when Metal data is authoritative (e.g. renderbuffers). */
    int src_read_from_cpu = 0;
    if (!src_tex->metal_data_authoritative && src_tex->faces &&
        src_tex->faces[0].levels && (size_t)src_level < src_tex->num_levels) {
        TextureLevel *src_level_info = &src_tex->faces[0].levels[src_level];
        if (src_level_info->data && src_level_info->width > 0 &&
            src_level_info->height > 0 && src_level_info->pitch > 0) {
            size_t src_bpp = src_level_info->pitch / src_level_info->width;
            if (src_bpp == bpp) {
                /* Read source pixels from CPU data */
                size_t src_pitch = src_level_info->pitch;
                size_t src_image_bytes = src_pitch * src_level_info->height;
                if (src_type == MGLTextureType3D) {
                    /* 3D source: srcZ is depth origin */
                    for (size_t z = 0; z < copy_depth; z++) {
                        for (size_t y = 0; y < copy_height; y++) {
                            size_t src_off = ((size_t)src_z + z) * src_image_bytes +
                                             ((size_t)src_y + y) * src_pitch +
                                             (size_t)src_x * bpp;
                            size_t dst_off = z * image_bytes + y * row_bytes;
                            if (src_off + row_bytes <=
                                    src_image_bytes * src_level_info->depth &&
                                dst_off + row_bytes <= total_bytes) {
                                memcpy((uint8_t *)staging + dst_off,
                                       (const uint8_t *)src_level_info->data +
                                           src_off,
                                       row_bytes);
                            }
                        }
                    }
                } else {
                    /* Non-3D source (2D array, cube, etc.): srcZ is slice */
                    for (size_t z = 0; z < copy_depth; z++) {
                        size_t face = 0;
                        if (src_type == MGLTextureTypeCube ||
                            src_type == MGLTextureTypeCubeArray) {
                            face = (size_t)src_z + z;
                        }
                        TextureLevel *slice_level =
                            (face < 6 && src_tex->faces[face].levels)
                                ? &src_tex->faces[face].levels[src_level]
                                : src_level_info;
                        if (!slice_level || !slice_level->data) {
                            src_read_from_cpu = 0;
                            break;
                        }
                        size_t s_pitch = slice_level->pitch;
                        size_t s_bpp = (slice_level->width > 0)
                                           ? (s_pitch / slice_level->width)
                                           : 0;
                        if (s_bpp != bpp) {
                            src_read_from_cpu = 0;
                            break;
                        }
                        /* For 2D array, all slices are in one TextureLevel;
                         * add slice offset. */
                        size_t src_slice_off = 0;
                        if (src_type == MGLTextureType2DArray) {
                            src_slice_off =
                                ((size_t)src_z + z) * s_pitch *
                                mglBdMaxSize(slice_level->height, 1u);
                        }
                        for (size_t y = 0; y < copy_height; y++) {
                            size_t src_off = src_slice_off +
                                             ((size_t)src_y + y) * s_pitch +
                                             (size_t)src_x * bpp;
                            size_t dst_off = z * image_bytes + y * row_bytes;
                            if (src_off + row_bytes <= slice_level->data_size &&
                                dst_off + row_bytes <= total_bytes) {
                                memcpy((uint8_t *)staging + dst_off,
                                       (const uint8_t *)slice_level->data +
                                           src_off,
                                       row_bytes);
                            }
                        }
                    }
                }
                src_read_from_cpu = 1;
            }
        }
    }

    if (!src_read_from_cpu) {
        MglBd3DReadCtx read_ctx = {
            .renderer = renderer,
            .glm_ctx = glm_ctx,
            .src_texture = src_texture,
            .src_type = src_type,
            .src_level = src_level,
            .src_x = src_x,
            .src_y = src_y,
            .src_z = src_z,
            .copy_width = copy_width,
            .copy_height = copy_height,
            .copy_depth = copy_depth,
            .row_bytes = row_bytes,
            .image_bytes = image_bytes,
            .staging = staging,
            .areas = areas,
            .core = core,
            .early_exit = 0,
        };
        if (!mglPlatformShellGuardedCallCtx(renderer,
                                            "copyImageSubData 3D read",
                                            mglBd3DReadGuarded, &read_ctx,
                                            NULL)) {
            free(staging);
            fprintf(stderr,
                    "MGL ERROR: mtlCopyImageSubData 3D fallback read failed: "
                    "caught exception\n");
            mglDispatchError(glm_ctx, __FUNCTION__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return true;
        }
        if (read_ctx.early_exit) {
            free(staging);
            mglDispatchError(glm_ctx, __FUNCTION__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return true;
        }
    }

    /* For bpp mismatch formats, convert staging from Metal format to CPU
     * storage format so the RMW merge uses matching pixel sizes. */
    if (bpp_mismatch) {
        GLenum cpu_format = 0;
        GLenum cpu_type = 0;
        if (mglGetCPUFormatTypeForInternalFormat(dst_tex->internalformat,
                                                 &cpu_format, &cpu_type)) {
            size_t cpu_row_bytes = copy_width * cpu_bpp;
            size_t cpu_image_bytes = cpu_row_bytes * copy_height;
            size_t cpu_total_bytes = cpu_image_bytes * copy_depth;
            void *cpu_staging = malloc(cpu_total_bytes);
            if (cpu_staging) {
                int conv_ok = 1;
                for (size_t z = 0; z < copy_depth && conv_ok; z++) {
                    const uint8_t *metal_src =
                        (const uint8_t *)staging + z * image_bytes;
                    uint8_t *cpu_dst = (uint8_t *)cpu_staging + z * cpu_image_bytes;
                    if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL(
                            metal_src, row_bytes, cpu_dst, cpu_row_bytes,
                            copy_width, copy_height,
                            mglBdTextureInfo(src_texture).pixel_format,
                            cpu_format, cpu_type, 0)) {
                        conv_ok = 0;
                    }
                }
                if (conv_ok) {
                    free(staging);
                    staging = cpu_staging;
                    row_bytes = cpu_row_bytes;
                    image_bytes = cpu_image_bytes;
                    total_bytes = cpu_total_bytes;
                    bpp = cpu_bpp;
                } else {
                    free(cpu_staging);
                    free(staging);
                    mglDispatchError(glm_ctx, __FUNCTION__,
                                     (GLenum)mglRenderErrorInvalidOperation());
                    return true;
                }
            }
        }
    }

    /* Write to 3D destination via CPU-data read-modify-write.  AGX drivers have
     * bugs where replaceRegion with non-zero origin, getBytes,
     * copyFromTexture:toTexture:, and copyFromBuffer:toTexture: all trigger
     * "slice OOB" assertions on 3D textures.  The only safe Metal write path
     * for 3D textures is replaceRegion with origin (0,0,0).  So we use the
     * CPU-side level data as the base, merge the source pixels into it, and
     * write the entire level back. */
    {
        /* Early checks already verified: dstLevelInfo exists, has data,
         * metal_data_authoritative == false, and cpuBpp == bpp. */
        TextureLevel *dst_level_info = &dst_tex->faces[0].levels[dst_level];
        GLuint level_width = dst_level_info->width;
        GLuint level_height = dst_level_info->height;
        GLuint level_depth = dst_level_info->depth;
        size_t level_pitch = dst_level_info->pitch;
        if (level_pitch == 0) {
            level_pitch = (size_t)level_width * bpp;
        }
        size_t level_image_bytes = level_pitch * level_height;
        size_t full_total_bytes = level_image_bytes * level_depth;

        void *full_level_bytes = malloc(full_total_bytes);
        if (!full_level_bytes) {
            free(staging);
            mglDispatchError(glm_ctx, __FUNCTION__,
                             (GLenum)mglRenderErrorOutOfMemory());
            return true;
        }

        /* Copy CPU data as the base */
        memcpy(full_level_bytes, (const void *)(uintptr_t)dst_level_info->data,
               full_total_bytes);

        /* Merge source pixels into the full level buffer */
        for (size_t z = 0; z < copy_depth; z++) {
            for (size_t y = 0; y < copy_height; y++) {
                size_t src_off = z * image_bytes + y * row_bytes;
                size_t dst_off = ((size_t)dst_z + z) * level_image_bytes +
                                 ((size_t)dst_y + y) * level_pitch +
                                 (size_t)dst_x * bpp;
                if (dst_off + row_bytes <= full_total_bytes &&
                    src_off + row_bytes <= total_bytes) {
                    memcpy((uint8_t *)full_level_bytes + dst_off,
                           (uint8_t *)staging + src_off, row_bytes);
                }
            }
        }

        MglBd3DWriteCtx write_ctx = {
            .dst_texture = dst_texture,
            .dst_tex = dst_tex,
            .dst_level = dst_level,
            .bpp_mismatch = bpp_mismatch,
            .level_width = level_width,
            .level_height = level_height,
            .level_depth = level_depth,
            .level_pitch = level_pitch,
            .level_image_bytes = level_image_bytes,
            .full_level_bytes = full_level_bytes,
        };
        if (!mglPlatformShellGuardedCallCtx(renderer,
                                            "copyImageSubData 3D replaceRegion",
                                            mglBd3DWriteGuarded, &write_ctx,
                                            NULL)) {
            free(staging);
            free(full_level_bytes);
            fprintf(stderr,
                    "MGL ERROR: mtlCopyImageSubData 3D replaceRegion failed: "
                    "caught exception\n");
            mglDispatchError(glm_ctx, __FUNCTION__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return true;
        }

        /* Update CPU data to reflect the merged result */
        memcpy((void *)(uintptr_t)dst_level_info->data, full_level_bytes,
               full_total_bytes);

        free(full_level_bytes);
        free(staging);
        /* Do NOT set metal_data_authoritative = GL_TRUE here.  The AGX driver
         * corrupts 3D texture readback (getBytes triggers "slice OOB"), so
         * subsequent glGetTexImage calls must read from CPU data instead.  The
         * Metal texture was updated via replaceRegion for sampling, but CPU
         * data remains the authoritative source. */
        return true;
    }
}

/* === copyImageSubData: post-blit readback + the dispatch itself (log 145) ==
 *
 * Mechanical translation of -copyImageSubDataPostBlitReadback:… and
 * -(void)mtlCopyImageSubData:….  The dispatcher's `ctx = glm_ctx;` becomes
 * mglPlatformShellSetContext (the existing entry for that ivar), `_capability`
 * is core->capability, the current command buffer owner comes from
 * areas.command, and the three @try/@catch frames become shell guarded calls.
 */

/* -copyImageSubDataPostBlitReadback:dstTexture:dstType:dstLevel:dstX:dstY:
 *  dstZ:width:height:depth: */
bool mglBlitCopyImageSubDataPostBlitReadback(
    void *renderer, Texture *dst_tex, void *dst_texture, uint32_t dst_type,
    GLint dst_level, GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width,
    GLsizei height, GLsizei depth)
{
    /* After blit, read back the blitted region from dst Metal to dst CPU so
     * that CPU data is authoritative.  This avoids the need for the
     * metal_data_authoritative flag, which causes "modified contents outside
     * of copied region" / "wrong layer" errors when non-blitted regions of the
     * same level have stale Metal data.
     *
     * Skip readback for 3D destinations (AGX getBytes bug on 3D textures,
     * tracked via MGLCapabilityHasBug(MGL_BUG_3D_GETBYTES_SLICE_OOB)) and fall
     * back to per-level authoritative instead. */
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRendererCoreState *core = areas.core;

    int readback_done = 0;
    int skip_3d_readback =
        MGLCapabilityHasBug(&core->capability, MGL_BUG_3D_GETBYTES_SLICE_OOB);
    if ((!skip_3d_readback || dst_type != MGLTextureType3D) &&
        mglBdTextureInfo(dst_texture).storage_mode != MGLStorageModePrivate &&
        dst_tex->faces && (size_t)dst_level < dst_tex->num_levels) {
        /* Check that dst has CPU data for this level */
        TextureLevel *dst_lvl0 = dst_tex->faces[0].levels
                                     ? &dst_tex->faces[0].levels[dst_level]
                                     : NULL;
        if (dst_lvl0 && dst_lvl0->data && dst_lvl0->pitch > 0 &&
            dst_lvl0->width > 0) {
            size_t dst_metal_bpp = mglMetalReadbackBytesPerPixel(
                mglBdTextureInfo(dst_texture).pixel_format);
            size_t dst_cpu_bpp = dst_lvl0->pitch / dst_lvl0->width;
            if (dst_metal_bpp > 0 && dst_cpu_bpp == dst_metal_bpp) {
                (void)mglRendererSynchronizeRenderPassForTextureReadbackPort(
                    renderer, dst_texture, "copyImageSubData.blitReadback");
                mglRendererFlushCommandBufferPort(renderer, 1);

                size_t copy_width = mglBdMaxSize((size_t)width, 1u);
                size_t copy_height = mglBdMaxSize((size_t)height, 1u);
                size_t num_slices = mglBdMaxSize((size_t)depth, 1u);
                size_t row_bytes = copy_width * dst_metal_bpp;
                size_t image_bytes = row_bytes * copy_height;
                void *staging = malloc(image_bytes);

                if (staging) {
                    int readback_ok = 1;
                    for (size_t s = 0; s < num_slices && readback_ok; s++) {
                        size_t dst_mtl_slice = 0;
                        GLuint dst_face = 0;
                        MGLRegionValue dst_region;

                        if (dst_type == MGLTextureTypeCube ||
                            dst_type == MGLTextureTypeCubeArray) {
                            dst_mtl_slice = ((size_t)dst_z + s) % 6;
                            dst_face = (GLuint)dst_mtl_slice;
                            dst_region = mglBlitRegion2D(
                                (size_t)dst_x, (size_t)dst_y, copy_width,
                                copy_height);
                        } else if (dst_type == MGLTextureType2DArray) {
                            dst_mtl_slice = (size_t)dst_z + s;
                            dst_face = 0;
                            dst_region = mglBlitRegion2D(
                                (size_t)dst_x, (size_t)dst_y, copy_width,
                                copy_height);
                        } else {
                            dst_mtl_slice = 0;
                            dst_face = 0;
                            dst_region = mglBlitRegion2D(
                                (size_t)dst_x, (size_t)dst_y, copy_width,
                                copy_height);
                        }

                        MglBdGetBytesCtx read_ctx = {
                            .texture = dst_texture,
                            .bytes = staging,
                            .bytes_per_row = row_bytes,
                            .bytes_per_image = image_bytes,
                            .region = dst_region,
                            .level = (size_t)dst_level,
                            .slice = dst_mtl_slice,
                        };
                        if (!mglPlatformShellGuardedCallCtx(
                                renderer, "blit readback getBytes",
                                mglBdGetBytesGuarded, &read_ctx, NULL)) {
                            fprintf(stderr,
                                    "MGL WARNING: blit readback getBytes "
                                    "failed: caught exception\n");
                            readback_ok = 0;
                            break;
                        }

                        /* Update dst CPU data for this slice */
                        TextureLevel *cur_dst_lvl =
                            (dst_face < 6 && dst_tex->faces[dst_face].levels)
                                ? &dst_tex->faces[dst_face].levels[dst_level]
                                : NULL;
                        if (cur_dst_lvl && cur_dst_lvl->data &&
                            cur_dst_lvl->pitch > 0 && cur_dst_lvl->width > 0) {
                            size_t cur_cpu_bpp =
                                cur_dst_lvl->pitch / cur_dst_lvl->width;
                            if (cur_cpu_bpp == dst_metal_bpp) {
                                size_t slice_pitch =
                                    cur_dst_lvl->pitch *
                                    mglBdMaxSize(cur_dst_lvl->height, 1u);
                                size_t dst_slice_off = 0;
                                if (dst_type == MGLTextureType2DArray) {
                                    dst_slice_off =
                                        ((size_t)dst_z + s) * slice_pitch;
                                }
                                for (size_t y = 0; y < copy_height; y++) {
                                    size_t dst_off =
                                        dst_slice_off +
                                        ((size_t)dst_y + y) * cur_dst_lvl->pitch +
                                        (size_t)dst_x * dst_metal_bpp;
                                    if (dst_off + row_bytes <=
                                        cur_dst_lvl->data_size) {
                                        memcpy(
                                            (uint8_t *)(uintptr_t)
                                                    cur_dst_lvl->data +
                                                dst_off,
                                            (const uint8_t *)staging +
                                                y * row_bytes,
                                            row_bytes);
                                    }
                                }
                            }
                        }
                    }
                    free(staging);

                    if (readback_ok) {
                        for (int f = 0; f < 6; f++) {
                            if (dst_tex->faces[f].levels) {
                                dst_tex->faces[f]
                                    .levels[dst_level]
                                    .metal_data_authoritative =
                                    (GLboolean)mglRenderGLBoolean(0);
                            }
                        }
                        readback_done = 1;
                    }
                }
            }
        }
    }

    /* Format-converting readback fallback for bpp mismatch cases (e.g.
     * R3_G3_B2, RGB12, RGB32F where CPU bpp != Metal bpp).  Read the blitted
     * region from dst Metal and convert to CPU storage format so that CPU data
     * is authoritative without setting per-texture metal_data_authoritative
     * (which would corrupt non-blitted levels). */
    if (!readback_done && dst_type != MGLTextureType3D &&
        mglBdTextureInfo(dst_texture).storage_mode != MGLStorageModePrivate &&
        dst_tex->faces && (size_t)dst_level < dst_tex->num_levels) {
        GLenum cpu_format = 0;
        GLenum cpu_type = 0;
        if (mglGetCPUFormatTypeForInternalFormat(dst_tex->internalformat,
                                                 &cpu_format, &cpu_type)) {
            TextureLevel *dst_lvl0 = dst_tex->faces[0].levels
                                         ? &dst_tex->faces[0].levels[dst_level]
                                         : NULL;
            if (dst_lvl0 && dst_lvl0->data && dst_lvl0->pitch > 0 &&
                dst_lvl0->width > 0) {
                size_t dst_metal_bpp = mglMetalReadbackBytesPerPixel(
                    mglBdTextureInfo(dst_texture).pixel_format);
                size_t cpu_bpp = (size_t)sizeForFormatType(cpu_format, cpu_type);
                if (dst_metal_bpp > 0 && cpu_bpp > 0) {
                    (void)mglRendererSynchronizeRenderPassForTextureReadbackPort(
                        renderer, dst_texture,
                        "copyImageSubData.fmtConvReadback");
                    mglRendererFlushCommandBufferPort(renderer, 1);

                    size_t copy_width = mglBdMaxSize((size_t)width, 1u);
                    size_t copy_height = mglBdMaxSize((size_t)height, 1u);
                    size_t num_slices = mglBdMaxSize((size_t)depth, 1u);
                    size_t metal_row_bytes = copy_width * dst_metal_bpp;
                    size_t metal_image_bytes = metal_row_bytes * copy_height;
                    size_t cpu_row_bytes = copy_width * cpu_bpp;
                    void *metal_staging = malloc(metal_image_bytes);
                    void *cpu_staging = malloc(cpu_row_bytes * copy_height);

                    if (metal_staging && cpu_staging) {
                        int fmt_readback_ok = 1;
                        for (size_t s = 0;
                             s < num_slices && fmt_readback_ok; s++) {
                            size_t dst_mtl_slice = 0;
                            MGLRegionValue dst_region;
                            if (dst_type == MGLTextureTypeCube ||
                                dst_type == MGLTextureTypeCubeArray) {
                                dst_mtl_slice = ((size_t)dst_z + s) % 6;
                                dst_region = mglBlitRegion2D(
                                    (size_t)dst_x, (size_t)dst_y, copy_width,
                                    copy_height);
                            } else if (dst_type == MGLTextureType2DArray) {
                                dst_mtl_slice = (size_t)dst_z + s;
                                dst_region = mglBlitRegion2D(
                                    (size_t)dst_x, (size_t)dst_y, copy_width,
                                    copy_height);
                            } else {
                                dst_mtl_slice = 0;
                                dst_region = mglBlitRegion2D(
                                    (size_t)dst_x, (size_t)dst_y, copy_width,
                                    copy_height);
                            }

                            MglBdGetBytesCtx read_ctx = {
                                .texture = dst_texture,
                                .bytes = metal_staging,
                                .bytes_per_row = metal_row_bytes,
                                .bytes_per_image = metal_image_bytes,
                                .region = dst_region,
                                .level = (size_t)dst_level,
                                .slice = dst_mtl_slice,
                            };
                            if (!mglPlatformShellGuardedCallCtx(
                                    renderer, "fmt-conv readback getBytes",
                                    mglBdGetBytesGuarded, &read_ctx, NULL)) {
                                fprintf(stderr,
                                        "MGL WARNING: fmt-conv readback "
                                        "getBytes failed: caught exception\n");
                                fmt_readback_ok = 0;
                                break;
                            }

                            /* Convert from Metal format to CPU format */
                            if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL(
                                    (const uint8_t *)metal_staging,
                                    metal_row_bytes, (uint8_t *)cpu_staging,
                                    cpu_row_bytes, copy_width, copy_height,
                                    mglBdTextureInfo(dst_texture).pixel_format,
                                    cpu_format, cpu_type, 0)) {
                                fprintf(stderr,
                                        "MGL WARNING: fmt-conv readback "
                                        "conversion failed for fmt=0x%x\n",
                                        (unsigned)dst_tex->internalformat);
                                fmt_readback_ok = 0;
                                break;
                            }

                            /* Write to dst CPU data for this slice */
                            GLuint dst_face = 0;
                            if (dst_type == MGLTextureTypeCube ||
                                dst_type == MGLTextureTypeCubeArray) {
                                dst_face =
                                    (GLuint)(((size_t)dst_z + s) % 6);
                            }
                            TextureLevel *cur_dst_lvl =
                                (dst_face < 6 &&
                                 dst_tex->faces[dst_face].levels)
                                    ? &dst_tex->faces[dst_face]
                                           .levels[dst_level]
                                    : NULL;
                            if (cur_dst_lvl && cur_dst_lvl->data &&
                                cur_dst_lvl->pitch > 0 &&
                                cur_dst_lvl->width > 0) {
                                size_t cur_cpu_bpp =
                                    cur_dst_lvl->pitch / cur_dst_lvl->width;
                                if (cur_cpu_bpp == cpu_bpp) {
                                    size_t slice_pitch =
                                        cur_dst_lvl->pitch *
                                        mglBdMaxSize(cur_dst_lvl->height, 1u);
                                    size_t dst_slice_off = 0;
                                    if (dst_type == MGLTextureType2DArray) {
                                        dst_slice_off =
                                            ((size_t)dst_z + s) * slice_pitch;
                                    }
                                    for (size_t y = 0; y < copy_height; y++) {
                                        size_t dst_off =
                                            dst_slice_off +
                                            ((size_t)dst_y + y) *
                                                cur_dst_lvl->pitch +
                                            (size_t)dst_x * cpu_bpp;
                                        if (dst_off + cpu_row_bytes <=
                                            cur_dst_lvl->data_size) {
                                            memcpy(
                                                (uint8_t *)(uintptr_t)
                                                        cur_dst_lvl->data +
                                                    dst_off,
                                                (const uint8_t *)cpu_staging +
                                                    y * cpu_row_bytes,
                                                cpu_row_bytes);
                                        }
                                    }
                                }
                            }
                        }

                        if (fmt_readback_ok) {
                            for (int f = 0; f < 6; f++) {
                                if (dst_tex->faces[f].levels) {
                                    dst_tex->faces[f]
                                        .levels[dst_level]
                                        .metal_data_authoritative =
                                        (GLboolean)mglRenderGLBoolean(0);
                                }
                            }
                            readback_done = 1;
                        }
                    }
                    free(metal_staging);
                    free(cpu_staging);
                }
            }
        }
    }
    return readback_done ? true : false;
}

typedef struct {
    void *encoder;
    void *source;
    void *destination;
    uint32_t src_type;
    uint32_t dst_type;
    GLint src_level;
    GLint dst_level;
    GLint src_x;
    GLint src_y;
    GLint dst_x;
    GLint dst_y;
    GLsizei width;
    GLsizei height;
    size_t src_slice;
    size_t dst_slice;
    size_t src_depth_plane;
    size_t dst_depth_plane;
    size_t iterations;
    size_t src_size_depth;
} MglBdCopyImageDispatchCtx;

static int mglBdCopyImageDispatchGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdCopyImageDispatchCtx *c = (MglBdCopyImageDispatchCtx *)ctx_raw;
    for (size_t i = 0; i < c->iterations; i++) {
        size_t cur_src_slice = c->src_slice;
        size_t cur_src_depth = c->src_depth_plane;
        size_t cur_dst_slice = c->dst_slice;
        size_t cur_dst_depth = c->dst_depth_plane;

        if (c->src_type == MGLTextureType3D && c->dst_type != MGLTextureType3D) {
            /* 3D -> 2D/array: read depth plane i from src */
            cur_src_depth = c->src_depth_plane + i;
            cur_src_slice = 0;
            cur_dst_slice = c->dst_slice + i;
        } else if (c->src_type != MGLTextureType3D &&
                   c->dst_type == MGLTextureType3D) {
            /* 2D/array -> 3D: read slice i from src, write to dst depth */
            cur_src_slice = c->src_slice + i;
            cur_dst_depth = c->dst_depth_plane + i;
            cur_dst_slice = 0;
        } else if (c->src_type != MGLTextureType3D &&
                   c->dst_type != MGLTextureType3D) {
            /* 2D/array -> 2D/array: copy slice i to slice i */
            cur_src_slice = c->src_slice + i;
            cur_dst_slice = c->dst_slice + i;
        }

        mglBdCopyTexture(
            c->encoder, c->source, cur_src_slice, (size_t)c->src_level,
            mglBlitOrigin((uint64_t)c->src_x, (uint64_t)c->src_y,
                          (uint64_t)cur_src_depth),
            mglBlitSize((uint64_t)c->width, (uint64_t)c->height,
                        (uint64_t)c->src_size_depth),
            c->destination, cur_dst_slice, (size_t)c->dst_level,
            mglBlitOrigin((uint64_t)c->dst_x, (uint64_t)c->dst_y,
                          (uint64_t)cur_dst_depth));
    }
    mglBdEndBlitEncoder(c->encoder);
    return 1;
}

typedef struct {
    void *encoder;
} MglBdEncoderCtx;

static int mglBdEndBlitEncoderGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglBdEncoderCtx *ctx = (MglBdEncoderCtx *)ctx_raw;
    mglBdEndBlitEncoder(ctx->encoder);
    return 1;
}

/* -(void)mtlCopyImageSubData:srcTexture:srcLevel:srcX:srcY:srcZ:dstTexture:
 *  dstLevel:dstX:dstY:dstZ:width:height:depth: */
void mglBlitCopyImageSubData(void *renderer, GLMContext glm_ctx, Texture *src_tex,
                             GLint src_level, GLint src_x, GLint src_y,
                             GLint src_z, Texture *dst_tex, GLint dst_level,
                             GLint dst_x, GLint dst_y, GLint dst_z,
                             GLsizei width, GLsizei height, GLsizei depth)
{
    mglPlatformShellSetContext(renderer, glm_ctx);

    if (!src_tex || !dst_tex || width <= 0 || height <= 0 || depth <= 0) {
        return;
    }

    if (!mglRendererBindMTLTexture(renderer, src_tex)) {
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    if (!mglRendererBindMTLTexture(renderer, dst_tex)) {
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    void *src_texture = src_tex->mtl_data;
    void *dst_texture = dst_tex->mtl_data;
    if (!src_texture || !dst_texture) {
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    uint32_t src_type = mglBdTextureInfo(src_texture).texture_type;
    uint32_t dst_type = mglBdTextureInfo(dst_texture).texture_type;

    if ((size_t)src_level >= mglBdTextureInfo(src_texture).mipmap_level_count ||
        (size_t)dst_level >= mglBdTextureInfo(dst_texture).mipmap_level_count) {
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidValue());
        return;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRendererCoreState *core = areas.core;

    int needs_3d_destination_workaround =
        dst_type == MGLTextureType3D &&
        (MGLCapabilityHasBug(&core->capability, MGL_BUG_3D_GETBYTES_SLICE_OOB) ||
         MGLCapabilityHasBug(&core->capability,
                             MGL_BUG_3D_REPLACE_REGION_NONZERO_ORIGIN) ||
         MGLCapabilityHasBug(&core->capability,
                             MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB));
    if (needs_3d_destination_workaround) {
        mglRendererEndRenderPassIfFramebufferChangedForNonDrawPort(renderer, 0);
        mglRendererEndRenderEncodingPort(renderer);
        RETURN_ON_FAILURE(mglRendererEnsureWritableCommandBufferPort(
            renderer, "mtlCopyImageSubData.3D"));
        if (mglBlitCopyImageSubData3DFallback(
                renderer, glm_ctx, src_tex, src_texture, src_type, src_level,
                src_x, src_y, src_z, dst_tex, dst_texture, dst_type, dst_level,
                dst_x, dst_y, dst_z, width, height, depth)) {
            return;
        }
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    /* The CPU-to-CPU path is C now (log 135). */
    if (mglBlitCopyImageSubDataCpuToCpu(
            renderer, glm_ctx, src_tex, src_texture, src_type, src_level, src_x,
            src_y, src_z, dst_tex, dst_texture, dst_type, dst_level, dst_x,
            dst_y, dst_z, width, height, depth)) {
        return;
    }

    if (mglBlitCopyImageSubDataFormatConversion(
            renderer, glm_ctx, src_tex, src_texture, src_type, src_level, src_x,
            src_y, src_z, dst_tex, dst_texture, dst_type, dst_level, dst_x,
            dst_y, dst_z, width, height, depth)) {
        return;
    }

    /* End a stale render pass (if the render encoder's FBO no longer matches
     * the current context FBO) so the blit encoder is not interleaved with a
     * live render encoder.  This is the only GL state the blit path depends
     * on; the full processGLState:false sync is unnecessary here. */
    mglRendererEndRenderPassIfFramebufferChangedForNonDrawPort(renderer, 0);
    mglRendererEndRenderEncodingPort(renderer);
    RETURN_ON_FAILURE(
        mglRendererEnsureWritableCommandBufferPort(renderer, "mtlCopyImageSubData"));

    /* For cube / cube-array / 2D-array / 1D-array targets, srcZ selects the
     * slice.  For 3D textures, srcZ is the depth origin. */
    size_t src_slice = 0;
    size_t dst_slice = 0;
    size_t src_depth_plane = 0;
    size_t dst_depth_plane = 0;
    size_t copy_depth = mglBdMaxSize((size_t)depth, 1u);

    if (src_type == MGLTextureType3D) {
        src_depth_plane = (size_t)src_z;
        src_slice = 0;
    } else {
        src_slice = (size_t)src_z;
        src_depth_plane = 0;
    }

    if (dst_type == MGLTextureType3D) {
        dst_depth_plane = (size_t)dst_z;
        dst_slice = 0;
    } else {
        dst_slice = (size_t)dst_z;
        dst_depth_plane = 0;
    }

    size_t iterations;
    size_t src_size_depth;

    if (src_type == MGLTextureType3D && dst_type == MGLTextureType3D) {
        iterations = 1u;
        src_size_depth = copy_depth;
    } else {
        iterations = copy_depth;
        src_size_depth = 1u;
    }

    /* Debug: read source renderbuffer data before blit to verify it has
     * content */
    if (src_tex->is_render_target || dst_tex->is_render_target) {
        (void)mglRendererSynchronizeRenderPassForTextureReadbackPort(
            renderer, src_texture, "copyImageSubData.srcCheck");
        mglRendererEndRenderEncodingPort(renderer);
    }

    void *blit_encoder = mglRenderCreateBlitEncoderBorrowed(
        mglBdCommandBufferOwner(&areas));
    if (!blit_encoder) {
        fprintf(stderr,
                "MGL ERROR: mtlCopyImageSubData failed to create blit "
                "encoder\n");
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorOutOfMemory());
        return;
    }

    MglBdCopyImageDispatchCtx dispatch_ctx = {
        .encoder = blit_encoder,
        .source = src_texture,
        .destination = dst_texture,
        .src_type = src_type,
        .dst_type = dst_type,
        .src_level = src_level,
        .dst_level = dst_level,
        .src_x = src_x,
        .src_y = src_y,
        .dst_x = dst_x,
        .dst_y = dst_y,
        .width = width,
        .height = height,
        .src_slice = src_slice,
        .dst_slice = dst_slice,
        .src_depth_plane = src_depth_plane,
        .dst_depth_plane = dst_depth_plane,
        .iterations = iterations,
        .src_size_depth = src_size_depth,
    };
    if (!mglPlatformShellGuardedCallCtx(renderer, "mtlCopyImageSubData blit",
                                        mglBdCopyImageDispatchGuarded,
                                        &dispatch_ctx, NULL)) {
        /* The method's @catch re-ended the encoder inside its own @try/@catch */
        MglBdEncoderCtx end_ctx = {.encoder = blit_encoder};
        if (!mglPlatformShellGuardedCallCtx(
                renderer, "mtlCopyImageSubData blit encoder end",
                mglBdEndBlitEncoderGuarded, &end_ctx, NULL)) {
            fprintf(stderr,
                    "MGL WARNING: mtlCopyImageSubData failed to end blit "
                    "encoder: caught exception\n");
        }
        fprintf(stderr,
                "MGL ERROR: mtlCopyImageSubData blit failed: caught "
                "exception\n");
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    /* Flush the command buffer to ensure the blit is executed before any
     * subsequent readback (e.g. glGetTexImage).  Without this, the blit may
     * still be pending in the command buffer when the readback occurs. */
    mglRendererFlushCommandBufferPort(renderer, 0);

    bool readback_done = mglBlitCopyImageSubDataPostBlitReadback(
        renderer, dst_tex, dst_texture, dst_type, dst_level, dst_x, dst_y, dst_z,
        width, height, depth);

    if (!readback_done) {
        if (dst_type == MGLTextureType3D && dst_tex->faces &&
            (size_t)dst_level < dst_tex->num_levels && dst_tex->faces[0].levels) {
            dst_tex->faces[0].levels[dst_level].metal_data_authoritative =
                (GLboolean)mglRenderGLBoolean(1);
        } else {
            dst_tex->metal_data_authoritative =
                (GLboolean)mglRenderGLBoolean(1);
        }
    }
}
