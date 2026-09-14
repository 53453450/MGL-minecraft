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
#include "mgl_metal_ref.h"        /* mglSafeReleaseMetalObj */
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
    *read_texid_ptr = read_texid;
    *read_subresource_ptr = read_subresource;
    *out_did_msaa_resolve = did_msaa_resolve;
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
