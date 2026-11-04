/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * textures.c
 * MGL
 *
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <mach/mach_time.h>

#include <errno.h>
#include <execinfo.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <Accelerate/Accelerate.h>

#include "pixel_utils.h"
#include "utils.h"
#include "glm_context.h"
#include "draw_command.h"
#include "mgl_frame_activity.h"
#include "mgl_pixel_format.h"
#include "mgl_render_api_texture.h"
#include "mgl_render_api_lifecycle.h"
#include "mgl_texture_debug.h"
#include "mgl_texture_transfer.h"
#include "mgl_texture_compat.h"
#include "mgl_readback_policy.h"
#include "mgl_rt_sync.h"

extern void *getBufferData(GLMContext ctx, Buffer *ptr);
extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern GLsizei mglSafeMaxTextureSize(GLMContext ctx);
extern GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
extern void mglReadPixels(GLMContext ctx, GLint x, GLint y, GLsizei width,
                          GLsizei height, GLenum format, GLenum type,
                          void *pixels);
extern bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *iparam, GLfloat *fparam);
#include "mgl_trace_log.h"
#include "mgl_textures_internal.h"

bool mglResolvePackBufferDst(GLMContext ctx, const char *api,
                                    void **pixels, Buffer **pack_buffer_out)
{
    Buffer *ptr = STATE(buffers[_PIXEL_PACK_BUFFER]);
    *pack_buffer_out = NULL;
    if (!ptr) {
        if (!*pixels) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
        return true;
    }

    if (ptr->mapped) {
        GLboolean persistent_map =
            ((ptr->storage_flags & GL_MAP_PERSISTENT_BIT) != 0u) &&
            ((ptr->access_flags & GL_MAP_PERSISTENT_BIT) != 0u);
        if (!persistent_map) {
            fprintf(stderr, "MGL Error: %s: pixel pack buffer is mapped non-persistently\n", api);
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
    }

    if (ptr->size < 0) {
        fprintf(stderr, "MGL Error: %s: pixel pack buffer has negative size\n", api);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (!ptr->data.buffer_data) {
        fprintf(stderr, "MGL Error: %s: pixel pack buffer has no CPU storage\n", api);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    uintptr_t offset = (uintptr_t)*pixels;
    if (offset > (uintptr_t)ptr->size) {
        fprintf(stderr, "MGL Error: %s: pixel pack offset overflow off=%" PRIuPTR " size=%lld\n",
                api, offset, (long long)ptr->size);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }
    *pixels = (void *)((uint8_t *)(uintptr_t)ptr->data.buffer_data + offset);
    *pack_buffer_out = ptr;
    return true;
}

/* A pack buffer offset that is not a multiple of the type's datum size, or
 * packing past the end of the buffer, is INVALID_OPERATION (§8.11.4). */
bool mglCheckPackBufferRange(GLMContext ctx, const Buffer *pack_buffer,
                                    uintptr_t offset, size_t bytes, GLenum type)
{
    if (!pack_buffer) {
        return true;
    }
    size_t datum = mglPixelTypeDatumBytes(type);
    if ((datum > 1u && offset % datum != 0u) ||
        bytes > (size_t)pack_buffer->size - offset) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }
    return true;
}

void mglApplyPackSwapBytes(GLMContext ctx, uint8_t *dst, size_t pitch,
                                  size_t row_bytes, size_t rows, GLenum type)
{
    size_t elem_size = mglPixelTypeDatumBytes(type);
    if (STATE(pack).swap_bytes != GL_TRUE || elem_size <= 1u) {
        return;
    }
    for (size_t r = 0; r < rows; r++) {
        mglSwapPixelBytes(dst + r * pitch, row_bytes, elem_size);
    }
}

/* §8.11.4.  `tex` NULL means the texture bound to `target`; a negative
 * bufSize means no bufSize limit (GetTexImage). */
void mglGetTexImageImpl(GLMContext ctx, Texture *tex, GLenum target,
                               GLint level, GLenum format, GLenum type,
                               GLsizei bufSize, void *pixels)
{
    GLuint slice = 0;

    switch (target) {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            break;
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            break;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            slice = (GLuint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
    Buffer *pack_buffer = NULL;
    uintptr_t pack_offset = (uintptr_t)pixels;
    if (!mglResolvePackBufferDst(ctx, "glGetTexImage", &pixels, &pack_buffer)) {
        return;
    }

    if (!tex) {
        tex = getTex(ctx, 0, target);
    }
    if (!tex) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - no texture bound\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    if (level < 0 || level >= (GLint)tex->num_levels) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - invalid level %d (max %d)\n", level, tex->num_levels);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    /* Unlike TexImage, GetTexImage may read STENCIL_INDEX from a depth-stencil
     * texture (GL 4.6 §8.11.4), so only the format/type pairing is checked. */
    GLenum verify_internalformat =
        (format == GL_STENCIL_INDEX &&
         mglInternalFormatIsCombinedDepthStencil(tex->internalformat))
            ? GL_STENCIL_INDEX8
            : tex->internalformat;
    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, verify_internalformat, format, type)) {
        return;
    }
    /* GetTexImage-specific validation (CTS isFormatValid OUTPUT_GETTEXIMAGE):
     * GL_DEPTH_STENCIL requires combined depth-stencil internal format;
     * GL_STENCIL_INDEX requires stencil or combined depth-stencil internal format. */
    if (format == GL_DEPTH_STENCIL && !mglInternalFormatIsCombinedDepthStencil(tex->internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (format == GL_STENCIL_INDEX) {
        bool is_stencil = (tex->internalformat == GL_STENCIL_INDEX ||
                           tex->internalformat == GL_STENCIL_INDEX1 ||
                           tex->internalformat == GL_STENCIL_INDEX4 ||
                           tex->internalformat == GL_STENCIL_INDEX8 ||
                           tex->internalformat == GL_STENCIL_INDEX16);
        if (!is_stencil && !mglInternalFormatIsCombinedDepthStencil(tex->internalformat)) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }
    if (sizeForFormatType(format, type) == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    TextureLevel *lvl = &tex->faces[slice].levels[level];
    if (!lvl || !lvl->complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    /* Views share the root store.  Compressed views have no CPU shadow —
     * decode from the root level with the view's internalformat so
     * signed↔unsigned RGTC reinterprets match GPU sampling. */
    TextureLevel *read_lvl = lvl;
    if (tex->view_root) {
        Texture *root = tex->view_root;
        GLuint root_level = tex->view_min_level + (GLuint)level;
        GLuint root_face = slice;
        if (tex->view_min_layer > 0u &&
            (root->target == GL_TEXTURE_CUBE_MAP ||
             root->target == GL_TEXTURE_CUBE_MAP_ARRAY)) {
            root_face = tex->view_min_layer + slice;
        }
        if (root_face < 6u && root->faces[root_face].levels &&
            root_level < root->num_levels &&
            root->faces[root_face].levels[root_level].data) {
            read_lvl = &root->faces[root_face].levels[root_level];
        }
    }

    GLsizei width = (GLsizei)lvl->width;
    GLsizei height = (GLsizei)lvl->height;
    GLsizei depth = (target == GL_TEXTURE_3D ||
                     target == GL_TEXTURE_2D_ARRAY ||
                     target == GL_TEXTURE_CUBE_MAP_ARRAY)
        ? (GLsizei)lvl->depth
        : 1;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    if (depth < 1) depth = 1;
    
    size_t pixel_size = sizeForFormatType(format, type);
    MGLTexturePackLayout pack_layout;
    if (!mglComputeTexturePackLayout(ctx,
                                     width,
                                     height,
                                     depth,
                                     pixel_size,
                                     "glGetTexImage",
                                     &pack_layout)) {
        return;
    }
    if (pack_layout.dst_pitch > UINT_MAX ||
        pack_layout.dst_image_size > UINT_MAX) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }
    ERROR_CHECK_RETURN(bufSize < 0 || pack_buffer ||
                       pack_layout.required_bytes <= (size_t)bufSize,
                       GL_INVALID_OPERATION);
    if (!mglCheckPackBufferRange(ctx, pack_buffer, pack_offset,
                                 pack_layout.required_bytes, type)) {
        return;
    }

    /* Deferred draws mark their render-target writes when flushed. */
    mglFlushPendingDraws(ctx);
    bool render_target_needs_readback =
        (tex->is_render_target && mglTextureWriteVersion(tex) != 0u) ||
        tex->metal_data_authoritative ||
        lvl->metal_data_authoritative;

    /* FS imageStore updates Metal storage but leaves CPU lvl->data stale until
     * MemoryBarrier. Several CTS cases (e.g. single-byte_data_alignment) omit
     * the barrier and call GetTexImage while the texture is still bound to a
     * writable image unit. Mirror the barrier's authoritative mark + finish so
     * we observe GPU image writes instead of the stale CPU upload. */
    if (!render_target_needs_readback) {
        GLuint max_units = STATE(var).max_image_units;
        for (GLuint i = 0; i < max_units && i < TEXTURE_UNITS; i++) {
            ImageUnit *iu = &STATE(image_units)[i];
            if (iu->tex != tex) {
                continue;
            }
            if (iu->access != GL_WRITE_ONLY && iu->access != GL_READ_WRITE) {
                continue;
            }
            mglFlushCommandBuffer(ctx);
            mglRendererFlush(ctx, true);
            tex->metal_data_authoritative = GL_TRUE;
            if (tex->faces[0].levels &&
                iu->level >= 0 &&
                iu->level < (GLint)tex->num_levels) {
                tex->faces[0].levels[iu->level].metal_data_authoritative = GL_TRUE;
            }
            mglRendererFlushImageUnitSlice(ctx, i);
            render_target_needs_readback = true;
            break;
        }
    }

    /* Packed depth-stencil uploads live on the GPU even before the texture is
     * used as a render target; CPU lvl->data can disagree with Metal layout. */
    if (!render_target_needs_readback &&
        mglInternalFormatIsCombinedDepthStencil(tex->internalformat) &&
        (mglRenderTextureHandle(tex) || (tex->dirty_bits & DIRTY_TEXTURE_DATA))) {
        render_target_needs_readback = true;
    }

    /* Prefer CPU decode for RGTC / BPTC-float even when metal-authoritative.
     * TextureView marks the root authoritative for GPU sampling; without this
     * gate GetTexImage falls into Metal getBytes and returns raw BC4/BC6H
     * blocks (CTS view_classes then builds a bogus FLOAT reference).  Decode
     * with tex->internalformat so signed↔unsigned views reinterpret the same
     * root blocks the way Metal sampling does. */
    {
        const GLenum view_if = (GLenum)tex->internalformat;
        const bool prefer_block_cpu =
            read_lvl && read_lvl->data &&
            (mglInternalFormatIsRGTC(view_if) ||
             view_if == GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT ||
             view_if == GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT);
        if (prefer_block_cpu || !render_target_needs_readback) {
            if (mglCopyTextureLevelToPackBuffer(
                    read_lvl, view_if, width, height, depth, format,
                    type, &pack_layout, pixels,
                    STATE(pack).swap_bytes == GL_TRUE)) {
                return;
            }
        }
    }

    if (target == GL_TEXTURE_3D ||
        target == GL_TEXTURE_1D_ARRAY ||
        target == GL_TEXTURE_2D_ARRAY ||
        target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        /* Levels without CPU data (e.g. from GenerateMipmap) live only in
         * Metal storage. */
        if (render_target_needs_readback || mglRenderTextureHandle(tex)) {
            if (!mglRenderTextureHandle(tex)) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }

            mglFlushCommandBuffer(ctx);
            uint8_t *dst_base = (uint8_t *)pixels + pack_layout.skip_offset_bytes;
            /* GL_TEXTURE_1D_ARRAY stores layer count in height; each Metal
             * slice is 1 texel tall (2D-array backing). */
            GLsizei layer_count = (target == GL_TEXTURE_1D_ARRAY)
                ? height
                : depth;
            GLsizei slice_height = (target == GL_TEXTURE_1D_ARRAY) ? 1 : height;
            size_t slice_image_size = (target == GL_TEXTURE_1D_ARRAY)
                ? ((size_t)width * pixel_size)
                : pack_layout.dst_image_size;
            if (target == GL_TEXTURE_1D_ARRAY) {
                /* Recompute tightly-packed layer stride for 1D array. */
                MGLTexturePackLayout layer_layout;
                if (!mglComputeTexturePackLayout(ctx, width, 1, 1, pixel_size,
                                                 "glGetTexImage",
                                                 &layer_layout)) {
                    return;
                }
                slice_image_size = layer_layout.dst_image_size;
            }
            for (GLsizei z = 0; z < layer_count; z++) {
                mglRendererGetTexImage(ctx,
                                              tex,
                                              dst_base + ((size_t)z * slice_image_size),
                                              (GLuint)pack_layout.dst_pitch,
                                              (GLuint)slice_image_size,
                                              0,
                                              0,
                                              width,
                                              slice_height,
                                              format,
                                              type,
                                              level,
                                              (GLuint)z);
                if (STATE(error) != GL_NO_ERROR) {
                    return;
                }
                mglApplyPackSwapBytes(ctx, dst_base + ((size_t)z * slice_image_size),
                                      pack_layout.dst_pitch, (size_t)width * pixel_size,
                                      (size_t)slice_height, type);
            }
            return;
        }
        memset((uint8_t *)pixels + pack_layout.skip_offset_bytes, 0, pack_layout.write_span_bytes);
        return;
    }

    /* If the texture has no Metal data, try the CPU data fallback first.
     * If that fails, let mtlGetTexImage handle Metal texture creation via
     * bindMTLTexture. */
    if (!mglRenderTextureHandle(tex)) {
        /* CPU data fallback: use stored CPU data directly if available */
        if (lvl->data && lvl->pitch >= (size_t)width * pixel_size &&
            (tex->internalformat == GL_RGBA8 || tex->internalformat == GL_RGB8) &&
            format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
            uint8_t *dst = (uint8_t *)pixels + pack_layout.skip_offset_bytes;
            const uint8_t *src = (const uint8_t *)lvl->data;
            size_t rowBytes = (size_t)width * pixel_size;
            for (GLsizei y = 0; y < height; y++) {
                memcpy(dst + ((size_t)y * pack_layout.dst_pitch),
                       src + ((size_t)y * lvl->pitch),
                       rowBytes);
            }
            return;
        }
        /* No CPU fallback available - let mtlGetTexImage try to create
         * the Metal texture on-demand via bindMTLTexture. */
    }

    // Use the Metal function to read the texture
    mglFlushCommandBuffer(ctx);
    mglRendererGetTexImage(ctx,
                                  tex,
                                  (uint8_t *)pixels + pack_layout.skip_offset_bytes,
                                  (GLuint)pack_layout.dst_pitch,
                                  (GLuint)pack_layout.dst_image_size,
                                  0,
                                  0,
                                  width,
                                  height,
                                  format,
                                  type,
                                  level,
                                  slice);
    mglApplyPackSwapBytes(ctx, (uint8_t *)pixels + pack_layout.skip_offset_bytes,
                          pack_layout.dst_pitch, (size_t)width * pixel_size,
                          (size_t)height, type);
}

void mglGetTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, void *pixels)
{
    mglGetTexImageImpl(ctx, NULL, target, level, format, type, -1, pixels);
}

void mglGetnTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    ERROR_CHECK_RETURN(bufSize >= 0, GL_INVALID_VALUE);
    mglGetTexImageImpl(ctx, NULL, target, level, format, type, bufSize, pixels);
}

void mglGetTextureImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    ERROR_CHECK_RETURN(bufSize >= 0, GL_INVALID_VALUE);
    Texture *tex = findTexture(ctx, texture);
    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);
    switch (tex->target) {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_RECTANGLE:
            mglGetTexImageImpl(ctx, tex, tex->target, level, format, type,
                               bufSize, pixels);
            return;
        case GL_TEXTURE_CUBE_MAP:
            break;
        default:
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
    }

    /* A cube map reads as a depth-6 image, faces in table 9.3 order. */
    ERROR_CHECK_RETURN(level >= 0 && level < (GLint)tex->num_levels,
                       GL_INVALID_VALUE);
    const TextureLevel *face0 = &tex->faces[0].levels[level];
    for (GLuint face = 0; face < 6u; face++) {
        const TextureLevel *lvl = &tex->faces[face].levels[level];
        ERROR_CHECK_RETURN(lvl->complete && lvl->width == face0->width &&
                           lvl->height == face0->height,
                           GL_INVALID_OPERATION);
    }
    size_t pixel_size = sizeForFormatType(format, type);
    ERROR_CHECK_RETURN(pixel_size != 0u, GL_INVALID_ENUM);
    MGLTexturePackLayout layout;
    if (!mglComputeTexturePackLayout(ctx, (GLsizei)face0->width,
                                     (GLsizei)face0->height, 6, pixel_size,
                                     "glGetTextureImage", &layout)) {
        return;
    }
    ERROR_CHECK_RETURN(STATE(buffers[_PIXEL_PACK_BUFFER]) ||
                       layout.required_bytes <= (size_t)bufSize,
                       GL_INVALID_OPERATION);
    GLuint raises_before = LIVE_STATE(error_raises);
    for (GLuint face = 0; face < 6u; face++) {
        mglGetTexImageImpl(ctx, tex, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                           level, format, type, -1,
                           (uint8_t *)pixels + face * layout.dst_image_size);
        if (LIVE_STATE(error_raises) != raises_before) {
            return;
        }
    }
}

void mglGetTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL: glGetTextureSubImage called - texture=%u\n", texture);
    if (bufSize < 0 || level < 0 || xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (width == 0 || height == 0 || depth == 0) {
        return;
    }
    Buffer *pack_buffer = NULL;
    uintptr_t pack_offset = (uintptr_t)pixels;
    if (!mglResolvePackBufferDst(ctx, "glGetTextureSubImage", &pixels, &pack_buffer)) {
        return;
    }
    
    Texture *tex = findTexture(ctx, texture);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    /* GL 4.5 §8.11.4: buffer and multisample textures are rejected. */
    if (tex->target == GL_TEXTURE_BUFFER ||
        tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
        tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (level >= (GLint)tex->num_levels || !tex->faces[0].levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLuint face = 0u;
    GLint level_zoffset = zoffset;
    GLsizei level_depth = depth;
    if (tex->target == GL_TEXTURE_CUBE_MAP) {
        if (zoffset < 0 || depth < 0 || zoffset + depth > 6) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        face = (GLuint)zoffset;
        level_zoffset = 0;
        level_depth = 1;
    }

    TextureLevel *lvl = &tex->faces[face].levels[level];
    if (!lvl->complete ||
        xoffset > (GLint)lvl->width ||
        yoffset > (GLint)lvl->height ||
        level_zoffset > (GLint)lvl->depth ||
        width > (GLsizei)(lvl->width - (GLuint)xoffset) ||
        height > (GLsizei)(lvl->height - (GLuint)yoffset) ||
        level_depth > (GLsizei)(lvl->depth - (GLuint)level_zoffset)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    
    size_t pixel_size = sizeForFormatType(format, type);
    if (pixel_size == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    MGLTexturePackLayout pack_layout;
    if (!mglComputeTexturePackLayout(ctx,
                                     width,
                                     height,
                                     depth,
                                     pixel_size,
                                     "glGetTextureSubImage",
                                     &pack_layout)) {
        return;
    }
    if ((!pack_buffer && pack_layout.required_bytes > (size_t)bufSize) ||
        pack_layout.dst_pitch > UINT_MAX ||
        pack_layout.dst_image_size > UINT_MAX) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!mglCheckPackBufferRange(ctx, pack_buffer, pack_offset,
                                 pack_layout.required_bytes, type)) {
        return;
    }

    mglFlushPendingDraws(ctx);
    bool gpu_authoritative =
        (tex->is_render_target && mglTextureWriteVersion(tex) != 0u) ||
        tex->metal_data_authoritative ||
        lvl->metal_data_authoritative;

    if (!gpu_authoritative && tex->target == GL_TEXTURE_CUBE_MAP) {
        uint8_t *dst_base = (uint8_t *)pixels + pack_layout.skip_offset_bytes;
        for (GLsizei z = 0; z < depth; z++) {
            GLuint cube_face = (GLuint)(zoffset + z);
            TextureLevel *face_level = &tex->faces[cube_face].levels[level];
            MGLTexturePackLayout slice_layout = pack_layout;
            slice_layout.skip_offset_bytes = 0u;
            if (!mglCopyTextureSubRectToPackBuffer(face_level,
                                                   tex->internalformat,
                                                   xoffset,
                                                   yoffset,
                                                   0,
                                                   width,
                                                   height,
                                                   1,
                                                   format,
                                                   type,
                                                   &slice_layout,
                                                   dst_base + ((size_t)z * pack_layout.dst_image_size),
                                                   STATE(pack).swap_bytes == GL_TRUE)) {
                memset(dst_base + ((size_t)z * pack_layout.dst_image_size), 0, pack_layout.write_span_bytes);
            }
        }
        return;
    }

    if (!gpu_authoritative &&
        mglCopyTextureSubRectToPackBuffer(lvl,
                                          tex->internalformat,
                                          xoffset,
                                          yoffset,
                                          level_zoffset,
                                          width,
                                          height,
                                          level_depth,
                                          format,
                                          type,
                                          &pack_layout,
                                          pixels,
                                          STATE(pack).swap_bytes == GL_TRUE)) {
        return;
    }

    if (!mglRenderTextureHandle(tex)) {
        memset((uint8_t *)pixels + pack_layout.skip_offset_bytes, 0, pack_layout.write_span_bytes);
        return;
    }

    mglFlushCommandBuffer(ctx);
    uint8_t *dst_base = (uint8_t *)pixels + pack_layout.skip_offset_bytes;
    for (GLsizei z = 0; z < depth; z++) {
        uint8_t *dst = dst_base + (size_t)z * pack_layout.dst_image_size;
        mglRendererGetTexImage(ctx,
                                      tex,
                                      dst,
                                      (GLuint)pack_layout.dst_pitch,
                                      (GLuint)pack_layout.dst_image_size,
                                      xoffset,
                                      yoffset,
                                      width,
                                      height,
                                      format,
                                      type,
                                      level,
                                      (GLuint)(zoffset + z));
        mglApplyPackSwapBytes(ctx, dst, pack_layout.dst_pitch,
                              (size_t)width * pixel_size, (size_t)height, type);
    }
}

void mglGetCompressedTexImage(GLMContext ctx, GLenum target, GLint level, void *img)
{
    if (!img || level < 0) {
        ERROR_RETURN(level < 0 ? GL_INVALID_VALUE : GL_INVALID_OPERATION);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (tex && level >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!tex || !mglTexLevelInternalFormatCompressed(tex->internalformat) ||
        !tex->faces[0].levels ||
        !tex->faces[0].levels[level].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[level];
    if (tex->view_root) {
        Texture *root = tex->view_root;
        GLuint root_level = tex->view_min_level + (GLuint)level;
        if (root->faces[0].levels && root_level < root->num_levels &&
            root->faces[0].levels[root_level].data) {
            lvl = &root->faces[0].levels[root_level];
        }
    }
    if (lvl->data && lvl->data_size > 0u) {
        memcpy(img, (const void *)(uintptr_t)lvl->data, lvl->data_size);
    }
}

void mglGetnCompressedTexImage(GLMContext ctx, GLenum target, GLint lod, GLsizei bufSize, void *pixels)
{
    if (bufSize < 0 || lod < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!pixels && bufSize > 0) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (tex && lod >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!tex || !mglTexLevelInternalFormatCompressed(tex->internalformat) ||
        !tex->faces[0].levels ||
        !tex->faces[0].levels[lod].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[lod];
    if (tex->view_root) {
        Texture *root = tex->view_root;
        GLuint root_level = tex->view_min_level + (GLuint)lod;
        if (root->faces[0].levels && root_level < root->num_levels &&
            root->faces[0].levels[root_level].data) {
            lvl = &root->faces[0].levels[root_level];
        }
    }
    if (lvl->data_size > (size_t)bufSize) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (lvl->data && lvl->data_size > 0u) {
        memcpy(pixels, (const void *)(uintptr_t)lvl->data, lvl->data_size);
    }
}

void mglGetCompressedTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void *pixels)
{
    if (bufSize < 0 || level < 0 || xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (width == 0 || height == 0 || depth == 0) {
        return;
    }
    if (!pixels) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *tex = findTexture(ctx, texture);
    if (!tex || !mglTexLevelInternalFormatCompressed(tex->internalformat) ||
        level >= (GLint)tex->num_levels ||
        !tex->faces[0].levels) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    GLuint face = 0u;
    GLint level_zoffset = zoffset;
    GLsizei level_depth = depth;
    if (tex->target == GL_TEXTURE_CUBE_MAP) {
        if (zoffset + depth > 6) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        face = (GLuint)zoffset;
        level_zoffset = 0;
        level_depth = 1;
    }

    TextureLevel *lvl = &tex->faces[face].levels[level];
    GLsizei effective_level_depth = (GLsizei)lvl->depth;
    if (effective_level_depth == 0 &&
        (tex->target == GL_TEXTURE_1D ||
         tex->target == GL_TEXTURE_2D ||
         tex->target == GL_TEXTURE_RECTANGLE ||
         tex->target == GL_TEXTURE_CUBE_MAP)) {
        effective_level_depth = 1;
    }
    if (!lvl->complete ||
        xoffset > (GLint)lvl->width ||
        yoffset > (GLint)lvl->height ||
        level_zoffset > effective_level_depth ||
        width > (GLsizei)(lvl->width - (GLuint)xoffset) ||
        height > (GLsizei)(lvl->height - (GLuint)yoffset) ||
        level_depth > (GLsizei)(effective_level_depth - (GLuint)level_zoffset)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    const size_t block_w = 4u;
    const size_t block_h = 4u;
    const size_t block_bytes = 16u;
    size_t src_blocks_x = (lvl->width + block_w - 1u) / block_w;
    size_t src_blocks_y = (lvl->height + block_h - 1u) / block_h;
    size_t dst_blocks_x = ((size_t)width + block_w - 1u) / block_w;
    size_t dst_blocks_y = ((size_t)height + block_h - 1u) / block_h;
    size_t dst_size = dst_blocks_x * dst_blocks_y * (size_t)depth * block_bytes;
    if (dst_size > (size_t)bufSize) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    memset(pixels, 0, dst_size);
    if (!lvl->data || lvl->data_size == 0u) {
        return;
    }

    uint8_t *dst = (uint8_t *)pixels;
    for (GLsizei z = 0; z < depth; z++) {
        TextureLevel *src_level = lvl;
        GLint src_z = level_zoffset + z;
        if (tex->target == GL_TEXTURE_CUBE_MAP) {
            src_level = &tex->faces[(GLuint)(zoffset + z)].levels[level];
            src_z = 0;
        }
        if (!src_level->data || src_level->data_size == 0u) {
            continue;
        }
        const uint8_t *src = (const uint8_t *)(uintptr_t)src_level->data;
        for (size_t by = 0; by < dst_blocks_y; by++) {
            for (size_t bx = 0; bx < dst_blocks_x; bx++) {
                size_t src_bx = ((size_t)xoffset / block_w) + bx;
                size_t src_by = ((size_t)yoffset / block_h) + by;
                size_t src_index = (((size_t)src_z * src_blocks_y + src_by) * src_blocks_x + src_bx) * block_bytes;
                size_t dst_index = (((size_t)z * dst_blocks_y + by) * dst_blocks_x + bx) * block_bytes;
                if (src_index + block_bytes <= src_level->data_size &&
                    dst_index + block_bytes <= dst_size) {
                    memcpy(dst + dst_index, src + src_index, block_bytes);
                }
            }
        }
    }
}

GLenum mglTextureViewClass(GLenum internalformat)
{
    switch (internalformat) {
        case GL_RGBA32F: case GL_RGBA32UI: case GL_RGBA32I:
            return GL_VIEW_CLASS_128_BITS;
        case GL_RGB32F: case GL_RGB32UI: case GL_RGB32I:
            return GL_VIEW_CLASS_96_BITS;
        case GL_RGBA16F: case GL_RG32F: case GL_RGBA16UI: case GL_RG32UI:
        case GL_RGBA16I: case GL_RG32I: case GL_RGBA16: case GL_RGBA16_SNORM:
            return GL_VIEW_CLASS_64_BITS;
        case GL_RGB16: case GL_RGB16_SNORM: case GL_RGB16F: case GL_RGB16UI:
        case GL_RGB16I:
            return GL_VIEW_CLASS_48_BITS;
        case GL_RG16F: case GL_R11F_G11F_B10F: case GL_R32F: case GL_RGB10_A2UI:
        case GL_RGBA8UI: case GL_RG16UI: case GL_R32UI: case GL_RGBA8I:
        case GL_RG16I: case GL_R32I: case GL_RGB10_A2: case GL_RGBA8:
        case GL_RG16: case GL_RGBA8_SNORM: case GL_RG16_SNORM:
        case GL_SRGB8_ALPHA8: case GL_RGB9_E5:
            return GL_VIEW_CLASS_32_BITS;
        case GL_RGB8: case GL_RGB8_SNORM: case GL_SRGB8: case GL_RGB8UI:
        case GL_RGB8I:
            return GL_VIEW_CLASS_24_BITS;
        case GL_R16F: case GL_RG8UI: case GL_R16UI: case GL_RG8I: case GL_R16I:
        case GL_RG8: case GL_R16: case GL_RG8_SNORM: case GL_R16_SNORM:
            return GL_VIEW_CLASS_16_BITS;
        case GL_R8UI: case GL_R8I: case GL_R8: case GL_R8_SNORM:
            return GL_VIEW_CLASS_8_BITS;
        case GL_COMPRESSED_RED_RGTC1: case GL_COMPRESSED_SIGNED_RED_RGTC1:
            return GL_VIEW_CLASS_RGTC1_RED;
        case GL_COMPRESSED_RG_RGTC2: case GL_COMPRESSED_SIGNED_RG_RGTC2:
            return GL_VIEW_CLASS_RGTC2_RG;
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
            return GL_VIEW_CLASS_BPTC_UNORM;
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
            return GL_VIEW_CLASS_BPTC_FLOAT;
        default:
            return GL_NONE;
    }
}

bool mglTextureViewParameter(const Texture *tex, GLenum pname, GLint *out)
{
    GLuint value;
    switch (pname) {
        case GL_TEXTURE_IMMUTABLE_LEVELS: value = tex->immutable_levels; break;
        case GL_TEXTURE_VIEW_MIN_LEVEL:   value = tex->view_min_level;   break;
        case GL_TEXTURE_VIEW_NUM_LEVELS:  value = tex->view_num_levels;  break;
        case GL_TEXTURE_VIEW_MIN_LAYER:   value = tex->view_min_layer;   break;
        case GL_TEXTURE_VIEW_NUM_LAYERS:  value = tex->view_num_layers;  break;
        default: return false;
    }
    *out = tex->immutable_storage ? (GLint)value : 0;
    return true;
}

bool mglTextureFormatSamplesStencil(const Texture *tex)
{
    switch (tex->internalformat) {
        case GL_STENCIL_INDEX:
        case GL_STENCIL_INDEX1:
        case GL_STENCIL_INDEX4:
        case GL_STENCIL_INDEX8:
        case GL_STENCIL_INDEX16:
            return true;
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return tex->params.depth_stencil_mode == GL_STENCIL_INDEX;
        default:
            return false;
    }
}

/* §8.17 completeness of `tex` when sampled with the given filters. */
bool mglTextureSamplingComplete(const Texture *tex, GLenum min_filter, GLenum mag_filter)
{
    switch (tex->target) {
        case GL_TEXTURE_BUFFER:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
        default:
            break;
    }

    bool nearest = mag_filter == GL_NEAREST &&
                   (min_filter == GL_NEAREST || min_filter == GL_NEAREST_MIPMAP_NEAREST);
    if (!nearest && (mglInternalFormatIsInteger(tex->internalformat) ||
                     mglTextureFormatSamplesStencil(tex))) {
        return false;
    }
    /* §8.14.3: immutable levels clamp base/max into the allocated chain. */
    if (tex->immutable_storage) {
        return tex->num_levels > 0u;
    }

    GLuint base = tex->params.base_level;
    GLuint faces = tex->target == GL_TEXTURE_CUBE_MAP ? _CUBE_MAP_MAX_FACE : 1u;
    bool square = tex->target == GL_TEXTURE_CUBE_MAP ||
                  tex->target == GL_TEXTURE_CUBE_MAP_ARRAY;
    if (base >= tex->num_levels) {
        return false;
    }
    for (GLuint f = 0; f < faces; f++) {
        if (!tex->faces[f].levels) {
            return false;
        }
    }
    const TextureLevel *b0 = &tex->faces[0].levels[base];
    for (GLuint f = 0; f < faces; f++) {
        const TextureLevel *lvl = &tex->faces[f].levels[base];
        if (!lvl->complete || lvl->width == 0u || lvl->height == 0u || lvl->depth == 0u ||
            lvl->width != b0->width || lvl->height != b0->height ||
            (square && lvl->width != lvl->height)) {
            return false;
        }
    }
    if (min_filter == GL_NEAREST || min_filter == GL_LINEAR) {
        return true;
    }

    GLuint max_level = tex->params.max_level;
    if (base > max_level) {
        return false;
    }
    GLuint maxsize = b0->width;
    if (tex->target != GL_TEXTURE_1D && tex->target != GL_TEXTURE_1D_ARRAY) {
        maxsize = MAX(maxsize, b0->height);
    }
    if (tex->target == GL_TEXTURE_3D) {
        maxsize = MAX(maxsize, b0->depth);
    }
    GLuint q = base;
    while (maxsize > 1u && q < max_level) {
        maxsize >>= 1;
        q++;
    }
    for (GLuint k = base + 1u; k <= q; k++) {
        if (k >= tex->num_levels) {
            return false;
        }
        GLuint w = 1u, h = 1u, d = 1u;
        mglTextureTargetLevelDimensions(tex->target, b0->width, b0->height, b0->depth,
                                        k - base, &w, &h, &d);
        for (GLuint f = 0; f < faces; f++) {
            const TextureLevel *lvl = &tex->faces[f].levels[k];
            if (!lvl->complete || lvl->width != w || lvl->height != h || lvl->depth != d) {
                return false;
            }
        }
    }
    return true;
}

