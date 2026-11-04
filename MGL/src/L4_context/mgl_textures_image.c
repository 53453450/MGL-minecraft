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

void mglTexImage1D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    bool proxy;

    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_1D:
            break;

        case GL_PROXY_TEXTURE_1D:
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, internalformat, format, type)) {
        fprintf(stderr,
                "MGL Error: mglTexImage2D rejected internalformat=0x%x format=0x%x type=0x%x target=0x%x\n",
                internalformat,
                format,
                type,
                target);
        return;
    }

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(border == 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, level, internalformat, width, 1, 1, border);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    tex->access = GL_READ_ONLY;

    createTextureLevel(ctx, tex, 0, level, false, internalformat, width, 1, 1, format, type, (void *)pixels, proxy);
}

void mglTexImage2D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    GLuint face;
    GLboolean is_array;
    GLboolean proxy;
    bool created_ok;

    face = 0;
    is_array = false;
    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_2D:
            break;

        /*
         * Some compatibility callers allocate the first slice through the
         * TexImage2D entry point. The storage path below already handles it;
         * avoid leaving a stale GL_INVALID_ENUM behind.
         */
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_1D_ARRAY:
            is_array = true;
            break;

        case GL_PROXY_TEXTURE_2D:
        case GL_PROXY_TEXTURE_CUBE_MAP:
            proxy = true;
            break;

        case GL_PROXY_TEXTURE_1D_ARRAY:
            is_array = true;
            proxy = true;
            break;

        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            break;

        case GL_PROXY_TEXTURE_RECTANGLE:
            proxy = true;
            /* GL 4.6 §8.5: INVALID_VALUE if TEXTURE_RECTANGLE and level != 0. */
            ERROR_CHECK_RETURN(level==0, GL_INVALID_VALUE);
            break;

        case GL_TEXTURE_RECTANGLE:
            ERROR_CHECK_RETURN(level==0, GL_INVALID_VALUE);
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);

    if (proxy)
    {
        if (border != 0) {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
            mglHandleProxyTexImageQuery(ctx, target, level, internalformat, 0, 0, 0, border);
            return;
        }
        mglHandleProxyTexImageQuery(ctx, target, level, internalformat, width, height, 1, border);
        return;
    }

    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, internalformat, format, type)) {
        return;
    }

    /* GL_STENCIL_INDEX as a pixel transfer format is only valid when the
     * texture's internalformat is a stencil-only format (GL_STENCIL_INDEX,
     * GL_STENCIL_INDEX1/4/8/16).  See GL_ARB_texture_stencil8 / OpenGL 4.4+.
     * PackedPixelsTests verifies non-stencil internalformats reject it;
     * texture_stencil8/multisample verifies stencil internalformats accept it. */
    if (format == GL_STENCIL_INDEX) {
        bool stencil_if = (internalformat == GL_STENCIL_INDEX ||
                            internalformat == GL_STENCIL_INDEX1 ||
                            internalformat == GL_STENCIL_INDEX4 ||
                            internalformat == GL_STENCIL_INDEX8 ||
                            internalformat == GL_STENCIL_INDEX16);
        ERROR_CHECK_RETURN(stencil_if, GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(border == 0, GL_INVALID_VALUE);

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    /* Per GL 4.6 spec §8.5, calling glTexImage* on a texture with immutable
     * storage (created via glTexStorage*) generates GL_INVALID_OPERATION.
     * The previous behavior silently redirected to glTexSubImage* for
     * compatibility, which violates the spec and masks application bugs. */
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);

    tex->access = GL_READ_ONLY;

    if (pixels == NULL && !STATE(buffers[_PIXEL_UNPACK_BUFFER]))
    {
        created_ok = createTextureLevel(ctx, tex, face, level, is_array, internalformat, width, height, 1, format, type, NULL, proxy);
        if (created_ok)
        {
            tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
            tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
            mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);
        }

        if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS) {
            fprintf(stderr,
                    "MGL TexImage2D allocate-only tex=%u target=0x%x %dx%d pixels=NULL\n",
                    tex->name, target, width, height);
        }
        return;
    }

    createTextureLevel(ctx, tex, face, level, is_array, internalformat, width, height, 1, format, type, (void *)pixels, proxy);
}

void mglTexImage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    Texture *tex;
    GLboolean proxy = (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE);

    if (target != GL_TEXTURE_2D_MULTISAMPLE &&
        target != GL_PROXY_TEXTURE_2D_MULTISAMPLE) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    {
        GLenum sample_err = mglMultisampleCountValidationError(ctx, samples);
        if (sample_err != GL_NO_ERROR) {
            ERROR_RETURN(sample_err);
            return;
        }
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (proxy) {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, height, 1, 0);
        return;
    }

    tex = getTex(ctx, 0, target);
    if (width == 0 || height == 0) {
        if (tex) {
            invalidateTexture(ctx, tex);
            tex->internalformat = internalformat;
            tex->width = 0u;
            tex->height = 0u;
            tex->depth = 1u;
            tex->samples = mglPromoteMultisampleCount(ctx, samples);
            tex->fixed_sample_locations = fixedsamplelocations ? GL_TRUE : GL_FALSE;
            tex->complete = GL_FALSE;
            tex->num_levels = 0u;
            tex->mipmap_levels = 0u;
        }
        return;
    }
    if (!mglTextureStorageMultisampleMetadata(ctx, tex, target, samples, internalformat, width, height, 1, fixedsamplelocations, GL_FALSE)) {
        return;
    }
    tex->immutable_storage = GL_FALSE;
}

/* GL 4.6 Table 8.14: RGTC formats do not support 3D textures. */
bool mglInternalFormatIsRGTC(GLenum internalformat)
{
    return internalformat == GL_COMPRESSED_RED_RGTC1 ||
           internalformat == GL_COMPRESSED_SIGNED_RED_RGTC1 ||
           internalformat == GL_COMPRESSED_RG_RGTC2 ||
           internalformat == GL_COMPRESSED_SIGNED_RG_RGTC2;
}

void mglTexImage3D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    GLboolean is_array;
    GLboolean proxy;

    is_array = false;
    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_3D:
            break;

        case GL_PROXY_TEXTURE_3D:
            proxy = true;
            break;

        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            is_array = true;
            break;

        case GL_PROXY_TEXTURE_2D_ARRAY:
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
            is_array = true;
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, internalformat, format, type)) {
        return;
    }

    /* GL_STENCIL_INDEX as a pixel transfer format is only valid when the
     * texture's internalformat is a stencil-only format.  See glTexImage2D
     * for the full rationale. */
    if (format == GL_STENCIL_INDEX) {
        bool stencil_if = (internalformat == GL_STENCIL_INDEX ||
                            internalformat == GL_STENCIL_INDEX1 ||
                            internalformat == GL_STENCIL_INDEX4 ||
                            internalformat == GL_STENCIL_INDEX8 ||
                            internalformat == GL_STENCIL_INDEX16);
        ERROR_CHECK_RETURN(stencil_if, GL_INVALID_OPERATION);
    }

    /* CTS isFormatValid: 3D textures must not use compressed RGTC formats
     * or depth/stencil internal formats. */
    if (target == GL_TEXTURE_3D) {
        ERROR_CHECK_RETURN(!mglInternalFormatIsRGTC((GLenum)internalformat), GL_INVALID_OPERATION);
        ERROR_CHECK_RETURN(!mglInternalFormatIsDepthStencil(internalformat), GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(depth >= 0, GL_INVALID_VALUE);
    if (target == GL_TEXTURE_CUBE_MAP_ARRAY ||
        target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY) {
        ERROR_CHECK_RETURN(width == height, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN((depth % 6) == 0, GL_INVALID_VALUE);
    }

    ERROR_CHECK_RETURN(border == 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, level, internalformat, width, height, depth, border);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);

    tex->access = GL_READ_ONLY;

    createTextureLevel(ctx, tex, 0, level, is_array, internalformat, width, height, depth, format, type, (void *)pixels, proxy);
}

void mglTexImage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    Texture *tex;
    GLboolean proxy = (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY);

    if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY &&
        target != GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    {
        GLenum sample_err = mglMultisampleCountValidationError(ctx, samples);
        if (sample_err != GL_NO_ERROR) {
            ERROR_RETURN(sample_err);
            return;
        }
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (proxy) {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, height, depth, 0);
        return;
    }

    tex = getTex(ctx, 0, target);
    if (width == 0 || height == 0 || depth == 0) {
        if (tex) {
            invalidateTexture(ctx, tex);
            tex->internalformat = internalformat;
            tex->width = 0u;
            tex->height = 0u;
            tex->depth = 0u;
            tex->samples = mglPromoteMultisampleCount(ctx, samples);
            tex->fixed_sample_locations = fixedsamplelocations ? GL_TRUE : GL_FALSE;
            tex->complete = GL_FALSE;
            tex->num_levels = 0u;
            tex->mipmap_levels = 0u;
        }
        return;
    }
    if (!mglTextureStorageMultisampleMetadata(ctx, tex, target, samples, internalformat, width, height, depth, fixedsamplelocations, GL_FALSE)) {
        return;
    }
    tex->immutable_storage = GL_FALSE;
}

#pragma mark texSubImage
bool texSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels)
{
    static uint64_t s_tex_sub_image_calls = 0u;
    uint64_t call_id = ++s_tex_sub_image_calls;
    double start_ms = mglTextureNowMs();
    const void *pixels_raw = pixels;
    const uint8_t *resolved_src = NULL;
    Buffer *resolved_unpack_buf = NULL;
    uint64_t resolved_src_hash = 0ull;
    Buffer *initial_unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    GLuint initial_unpack_name = initial_unpack_buf ? initial_unpack_buf->name : 0u;
    bool trace_upload = mglShouldTraceTextureUpload(tex,
                                                    initial_unpack_name,
                                                    width,
                                                    height,
                                                    depth,
                                                    0u);
    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    // Debug: Log large texture uploads (VM framebuffer size)
    if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS && width >= 640 && height >= 400) {
        fprintf(stderr, "MGL DEBUG: texSubImage tex_id=%u face=%u level=%d %dx%dx%d at (%d,%d,%d) pixels=%p\n",
                tex ? tex->name : 0, face, level, width, height, depth, xoffset, yoffset, zoffset, pixels);
    }

    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE texSubImage.begin call=%" PRIu64 " tex=%u target=0x%x face=%u level=%d off=(%d,%d,%d) dims=%dx%dx%d fmt=0x%x type=0x%x pixelsRaw=%p\n",
                call_id,
                tex ? tex->name : 0u,
                tex ? tex->target : 0u,
                face,
                level,
                xoffset,
                yoffset,
                zoffset,
                width,
                height,
                depth,
                format,
                type,
                pixels_raw);
    }
    
    // ERROR_CHECK_RETURN_VALUE(tex != NULL, GL_INVALID_OPERATION, false);
    if (tex == NULL) {
        fprintf(stderr, "MGL Error: texSubImage: tex is NULL\n");
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (tex->target == 0) {
        fprintf(stderr,
                "MGL ERROR: texSubImage called with invalid texture object tex=%p target=0x%x\n",
                (void *)tex,
                tex->target);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (face >= _CUBE_MAP_MAX_FACE) {
        fprintf(stderr, "MGL ERROR: texSubImage invalid face=%u\n", face);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (width <= 0 || height <= 0 || depth <= 0) {
        fprintf(stderr,
                "MGL texSubImage skip invalid size tex=%u %dx%dx%d\n",
                tex->name, width, height, depth);
        return true;
    }

    // ERROR_CHECK_RETURN_VALUE(level <= tex->num_levels, GL_INVALID_OPERATION, false);
    if (level >= (GLint)tex->num_levels) {
        fprintf(stderr, "MGL Error: texSubImage: level %d >= num_levels %d\n", level, tex->num_levels);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }
    
    if (!tex->faces[face].levels) {
        fprintf(stderr, "MGL Error: texSubImage: levels is NULL\n");
        ERROR_CHECK_RETURN_VALUE(false, GL_INVALID_OPERATION, false);
    }
    
    // ERROR_CHECK_RETURN_VALUE(tex->faces[face].levels[level].complete, GL_INVALID_OPERATION, false);
    if (!tex->faces[face].levels[level].complete) {
        fprintf(stderr, "MGL Error: texSubImage: level %d not complete\n", level);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (!pixels && !STATE(buffers[_PIXEL_UNPACK_BUFFER]))
    {
        fprintf(stderr,
                "MGL texSubImage skip upload: pixels=NULL tex=%u target=0x%x %dx%d\n",
                tex->name, tex->target, width, height);
        return true;
    }

    size_t pixel_size;
    size_t src_pitch;
    size_t src_image_size;
    size_t skip_offset_bytes;
    size_t required_bytes;
    MGLTextureUnpackLayout unpack_layout;

    pixel_size = sizeForFormatType(format, type);
    ERROR_CHECK_RETURN_VALUE(pixel_size > 0u, GL_INVALID_ENUM, false);

    if (!mglComputeTextureUnpackLayout(ctx,
                                       width,
                                       height,
                                       depth,
                                       pixel_size,
                                       "texSubImage",
                                       &unpack_layout)) {
        return false;
    }

    src_pitch = unpack_layout.src_pitch;
    src_image_size = unpack_layout.src_image_size;
    skip_offset_bytes = unpack_layout.skip_offset_bytes;
    required_bytes = unpack_layout.required_bytes;

    if (!trace_upload) {
        trace_upload = mglShouldTraceTextureUpload(tex,
                                                   initial_unpack_name,
                                                   width,
                                                   height,
                                                   depth,
                                                   required_bytes);
    }

    if (!mglResolveTexSubImageSource(ctx,
                                     tex,
                                     face,
                                     level,
                                     xoffset,
                                     yoffset,
                                     zoffset,
                                     width,
                                     height,
                                     depth,
                                     format,
                                     type,
                                     pixels_raw,
                                     skip_offset_bytes,
                                     required_bytes,
                                     trace_upload,
                                     &resolved_src,
                                     &resolved_unpack_buf)) {
        return false;
    }

    if (!resolved_src) {
        fprintf(stderr,
                "MGL texSubImage skip upload: resolved source is NULL tex=%u target=0x%x %dx%dx%d\n",
                tex->name,
                tex->target,
                width,
                height,
                depth);
        return true;
    }

    if (resolved_unpack_buf) {
        resolved_src_hash = mglHashBytesSampled(resolved_src, required_bytes);
    } else {
        /*
         * CPU uploads can point at a direct/native buffer whose exact mapped
         * extent is unknown to us. Avoid diagnostic scans here; the actual
         * row-wise unpack below is the first safe consumer, and the destination
         * texture backing store can be inspected after that.
         */
        resolved_src_hash = 0u;
    }

    if (resolved_unpack_buf && mglLooksAllZeroSampled(resolved_src, required_bytes)) {
        static uint64_t s_zero_upload_warning_count = 0u;
        uint64_t zero_warning_id = ++s_zero_upload_warning_count;
        size_t first_nonzero = 0u;
        uint8_t first_value = 0u;
        bool has_nonzero = mglFindFirstNonZeroByte(resolved_src,
                                                   required_bytes,
                                                   &first_nonzero,
                                                   &first_value);
        if (trace_upload || zero_warning_id <= 32u || (zero_warning_id % 512u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: texSubImage source sampled head/mid/tail all-zero tex=%u face=%u level=%d required=%zu src=%p fullZero=%d firstNonZero=0x%zx value=0x%02x warn=%" PRIu64 "\n",
                    tex->name,
                    face,
                    level,
                    required_bytes,
                    resolved_src,
                    has_nonzero ? 0 : 1,
                    has_nonzero ? first_nonzero : 0u,
                    has_nonzero ? first_value : 0u,
                    zero_warning_id);
            if (has_nonzero && trace_upload) {
                size_t dump_offset = first_nonzero;
                size_t dump_available = required_bytes - first_nonzero;
                if (dump_available > 64u) {
                    dump_available = 64u;
                }
                mglDumpBytesToStderr("texSubImage.source.firstNonZero", resolved_src + dump_offset, dump_available, dump_offset);
            }
        }
    }

    /* Debug-only tex13 dump removed: hardcoded texture-name sniffing
     * belongs in a targeted debugger, not production stderr.  The
     * zero-CPU-upload diagnostic path above already captures the
     * relevant backtrace and state when a real problem occurs. */

    void *texture_data;
    TextureLevel *lvl = &tex->faces[face].levels[level];
    size_t compact_upload_bytes = 0u;
    size_t compact_upload_row_bytes = 0u;
    size_t storage_pixel_size = lvl->pitch > 0u && lvl->width > 0u ? (lvl->pitch / (size_t)lvl->width) : pixel_size;
    if (storage_pixel_size == 0u) {
        storage_pixel_size = pixel_size;
    }
    if (!mglMulSizeT((size_t)width, storage_pixel_size, &compact_upload_row_bytes) ||
        !mglMulSizeT(compact_upload_row_bytes, (size_t)MAX(height, 1), &compact_upload_bytes) ||
        !mglMulSizeT(compact_upload_bytes, (size_t)MAX(depth, 1), &compact_upload_bytes)) {
        fprintf(stderr,
                "MGL ERROR: texSubImage compact byte computation overflow tex=%u dims=%dx%dx%d pixelSize=%zu\n",
                tex->name,
                width,
                height,
                depth,
                pixel_size);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    texture_data = (void *)tex->faces[face].levels[level].data;
    if (!texture_data) {
        fprintf(stderr,
                "MGL ERROR: texSubImage texture_data is NULL tex=%u face=%u level=%d target=0x%x\n",
                tex->name,
                face,
                level,
                tex->target);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    GLuint block_w = 0u, block_h = 0u, block_d = 1u, block_bytes = 0u;
    const GLenum store_if = tex->compressed_internalformat
                                ? tex->compressed_internalformat
                                : (GLenum)tex->internalformat;
    const int is_rgtc =
        store_if == GL_COMPRESSED_RED_RGTC1 ||
        store_if == GL_COMPRESSED_SIGNED_RED_RGTC1 ||
        store_if == GL_COMPRESSED_RG_RGTC2 ||
        store_if == GL_COMPRESSED_SIGNED_RG_RGTC2;
    const int is_bptc_float =
        store_if == GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT ||
        store_if == GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;
    if (is_rgtc && lvl->pitch != 0u &&
        mglCompressedBlockInfoOf(store_if, &block_w, &block_h, &block_d,
                                 &block_bytes)) {
        GLenum encode_error = mglEncodeRGTCRectToCPU(
            store_if, lvl, xoffset, yoffset, zoffset, width, height, depth,
            format, type, resolved_src, src_pitch, src_image_size);
        if (encode_error != GL_NO_ERROR) {
            ERROR_RETURN_VALUE(encode_error, false);
        }
        lvl->has_initialized_data = GL_TRUE;
        lvl->ever_written = GL_TRUE;
        lvl->suspicious_zero_upload = GL_FALSE;
        lvl->metal_data_authoritative = GL_FALSE;
        lvl->last_init_source = resolved_unpack_buf ? kTexSubImagePBO : kTexSubImageCPU;
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);
        return true;
    }
    if (is_bptc_float && lvl->pitch != 0u &&
        mglCompressedBlockInfoOf(store_if, &block_w, &block_h, &block_d,
                                 &block_bytes)) {
        GLenum encode_error = mglEncodeBPTCFloatRectToCPU(
            store_if, lvl, xoffset, yoffset, zoffset, width, height, depth,
            format, type, resolved_src, src_pitch, src_image_size);
        if (encode_error != GL_NO_ERROR) {
            ERROR_RETURN_VALUE(encode_error, false);
        }
        lvl->has_initialized_data = GL_TRUE;
        lvl->ever_written = GL_TRUE;
        lvl->suspicious_zero_upload = GL_FALSE;
        lvl->metal_data_authoritative = GL_FALSE;
        lvl->last_init_source = resolved_unpack_buf ? kTexSubImagePBO : kTexSubImageCPU;
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);
        return true;
    }
    
    if (!mglConvertTextureRectToCPU(tex->internalformat,
                                    lvl,
                                    xoffset,
                                    yoffset,
                                    zoffset,
                                    width,
                                    height,
                                    depth,
                                    format,
                                    type,
                                    resolved_src,
                                    src_pitch,
                                    src_image_size,
                                    STATE(unpack).swap_bytes == GL_TRUE)) {
        unpackTexture(ctx, tex, face, level, (void *)resolved_src, texture_data, src_pitch, src_image_size, pixel_size, xoffset, yoffset, zoffset, width, height, depth);
    }

    size_t upload_dst_offset = 0u;
    size_t upload_dst_span = 0u;
    const uint8_t *upload_dst = (const uint8_t *)texture_data;
    size_t dst_pitch = lvl->pitch;
    size_t dst_image_pitch = dst_pitch * (size_t)MAX(lvl->height, 1);
    bool upload_rect_valid = mglTextureRectByteRange(lvl,
                                                     storage_pixel_size,
                                                     (size_t)xoffset,
                                                     (size_t)yoffset,
                                                     (size_t)zoffset,
                                                     (size_t)width,
                                                     (size_t)height,
                                                     (size_t)depth,
                                                     &upload_dst_offset,
                                                     &upload_dst_span);
    if (upload_rect_valid) {
        upload_dst = (const uint8_t *)texture_data + upload_dst_offset;
    }

    /* Gate the expensive per-byte scans behind their actual consumers.
     * dst_hash is only read by the trace_upload fprintf below; upload_rect_zero
     * is only read by the suspicious-zero check, which itself requires
     * !resolved_unpack_buf and a trace-or-large-size gate.  Compute them lazily
     * so small/typical uploads skip the multi-KB to multi-MB CPU scans. */
    uint64_t dst_hash = 0ull;
    bool upload_rect_zero = false;
    bool suspicious_zero_cpu_upload = false;
    const bool need_zero_probe =
        !resolved_unpack_buf &&
        (trace_upload || compact_upload_bytes >= (256u * 1024u));
    if (trace_upload || need_zero_probe) {
        if (upload_rect_valid) {
            upload_rect_zero = mglTextureRectLooksAllZero(upload_dst,
                                                          dst_pitch,
                                                          dst_image_pitch,
                                                          compact_upload_row_bytes,
                                                          (size_t)MAX(height, 1),
                                                          (size_t)MAX(depth, 1));
        } else {
            upload_rect_zero = mglLooksAllZeroSampled((const uint8_t *)texture_data,
                                                     compact_upload_bytes);
        }
        suspicious_zero_cpu_upload = upload_rect_zero; /* zero-probe already gated above */
    }
    if (trace_upload) {
        dst_hash = upload_rect_valid
            ? mglHashTextureRect(upload_dst,
                                 dst_pitch,
                                 dst_image_pitch,
                                 compact_upload_row_bytes,
                                 (size_t)MAX(height, 1),
                                 (size_t)MAX(depth, 1))
            : mglHashBytesSampled(texture_data, compact_upload_bytes);
    }
    if (suspicious_zero_cpu_upload) {
        static uint64_t s_cpu_zero_upload_warning_count = 0u;
        uint64_t zero_warning_id = ++s_cpu_zero_upload_warning_count;
        if (trace_upload || zero_warning_id <= 32u || (zero_warning_id % 512u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: texSubImage CPU upload produced all-zero uploaded rect tex=%u label=\"%s\" face=%u level=%d required=%zu src=%p dst=%p rectOffset=%zu rectSpan=%zu warn=%" PRIu64 "\n",
                    tex->name,
                    tex->debug_label[0] != '\0' ? tex->debug_label : "(none)",
                    face,
                    level,
                    compact_upload_bytes,
                    resolved_src,
                    upload_dst,
                    upload_dst_offset,
                    upload_dst_span,
                    zero_warning_id);

            mglDumpTexSubImageZeroCpuResourceTag(ctx,
                                                 tex,
                                                 lvl,
                                                 face,
                                                 level,
                                                 xoffset,
                                                 yoffset,
                                                 zoffset,
                                                 width,
                                                 height,
                                                 depth,
                                                 format,
                                                 type,
                                                 pixels_raw,
                                                 resolved_src,
                                                 resolved_unpack_buf,
                                                 required_bytes,
                                                 compact_upload_bytes,
                                                 src_pitch,
                                                 compact_upload_row_bytes,
                                                 pixel_size,
                                                 zero_warning_id);

            if (trace_upload || zero_warning_id <= 8u) {
                mglDumpNativeBacktraceToStderr("texSubImage.zeroCPU", 32u);
            }

            mglRequestJavaThreadDumpForZeroCpuUpload(tex,
                                                     face,
                                                     level,
                                                     width,
                                                     height,
                                                     depth,
                                                     zero_warning_id);

            /*
             * At this point unpackTexture has already consumed the CPU pointer
             * row-by-row without faulting.  Dump only three small windows from
             * source and destination so the next log can prove whether the
             * incoming CPU image is really zero or our unpack path zeroed it.
             */
            size_t dst_total = upload_rect_valid ? upload_dst_span : lvl->data_size;
            size_t dump_dst_pitch = lvl->pitch;
            if (dst_total == 0u) {
                dst_total = compact_upload_bytes;
            }
            if (dump_dst_pitch == 0u) {
                dump_dst_pitch = compact_upload_row_bytes;
            }
            mglDumpTextureUploadSamples(tex,
                                        face,
                                        level,
                                        (const uint8_t *)resolved_src,
                                        required_bytes,
                                        src_pitch,
                                        upload_rect_valid ? upload_dst : (const uint8_t *)texture_data,
                                        dst_total,
                                        dump_dst_pitch,
                                        pixel_size,
                                        width,
                                        height,
                                        depth);
        }
    }

    /* Debug-only tex13 dump removed (see comment above). */

    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE texSubImage.afterUnpack call=%" PRIu64 " tex=%u face=%u level=%d requiredBytes=%zu srcHash=0x%016" PRIx64 " dstHash=0x%016" PRIx64 " elapsed=%.3fms\n",
                call_id,
                tex->name,
                face,
                level,
                required_bytes,
                resolved_src_hash,
                dst_hash,
                mglTextureNowMs() - start_ms);
    }

    // use a blit command to update data
    do
    {
        Buffer *buf;

        buf = resolved_unpack_buf;

        if (buf == NULL)
            continue;

        if (mglRenderTextureHandle(tex) == NULL)
            continue;

        if (src_pitch < unpack_layout.row_copy_bytes) {
            continue;
        }

        size_t metal_min_image_size = 0u;
        if (!mglMulSizeT(src_pitch, (size_t)height, &metal_min_image_size) ||
            src_image_size < metal_min_image_size) {
            continue;
        }

        size_t src_offset;
        size_t src_size;

        const uint8_t *pbo_data = (const uint8_t *)getBufferData(ctx, buf);
        if (!pbo_data || resolved_src < pbo_data) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage PBO direct upload has invalid source base tex=%u unpack=%u base=%p resolved=%p\n",
                    tex->name,
                    buf ? buf->name : 0u,
                    pbo_data,
                    resolved_src);
            break;
        }

        src_offset = (size_t)(resolved_src - pbo_data);
        if (src_offset > (size_t)buf->size) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage PBO direct upload source offset out of range tex=%u unpack=%u offset=%zu size=%lld\n",
                    tex->name,
                    buf ? buf->name : 0u,
                    src_offset,
                    (long long)buf->size);
            break;
        }

        src_size = required_bytes;

        // Preserve cube-map / array target slice information. zoffset is for 3D origin, not array/cube slice.
        mglRendererTexSubImage(ctx, tex, buf, src_offset, src_pitch, src_image_size, src_size, face, level, width, height, depth, xoffset, yoffset, zoffset);
        lvl->ever_written = GL_TRUE;
        lvl->suspicious_zero_upload = GL_FALSE;
        lvl->metal_data_authoritative = GL_FALSE;
        lvl->has_initialized_data = GL_TRUE;
        lvl->last_init_source = kTexSubImagePBO;
        lvl->last_upload_size = required_bytes;
        lvl->last_src_ptr = resolved_src;
        lvl->last_src_hash = dst_hash;
        mglTextureViewFamilyWritten(tex);

        if (trace_upload) {
            fprintf(stderr,
                    "MGL TRACE texSubImage.end call=%" PRIu64 " tex=%u face=%u level=%d upload=PBO ok=1 elapsed=%.3fms\n",
                    call_id,
                    tex->name,
                    face,
                    level,
                    mglTextureNowMs() - start_ms);
        }

        return true;
    } while(false);

    if (tex->params.sparse)
        mglSparseZeroUncommitted(tex, face, level, xoffset, yoffset, zoffset,
                                 width, height, depth);

    lvl->ever_written = GL_TRUE;
    lvl->suspicious_zero_upload = GL_FALSE;
    lvl->metal_data_authoritative = GL_FALSE;
    lvl->has_initialized_data = GL_TRUE;
    lvl->last_init_source = resolved_unpack_buf ? kTexSubImagePBO : kTexSubImageCPU;
    lvl->last_upload_size = required_bytes;
    lvl->last_src_ptr = resolved_src;
    lvl->last_src_hash = dst_hash;

    /* A view family reads its shared store from Metal, so a deferred CPU
     * upload would never be applied. */
    if ((tex->view_root || tex->view_count) && !mglRenderTextureHandle(tex)) {
        mglRendererBindTexture(ctx, tex);
    }
    bool uploaded_direct = false;
    bool had_pending_texture_data_before_direct = (tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0;
    if (!resolved_unpack_buf &&
        mglRenderTextureHandle(tex) &&
        (tex->dirty_bits & DIRTY_TEXTURE_LEVEL) == 0 &&
        upload_rect_valid) {
        size_t upload_src_image_size = dst_pitch * (size_t)MAX(height, 1);
        if (depth > 1) {
            upload_src_image_size = dst_image_pitch;
        }
        uploaded_direct = mglRendererTexSubImageBytes(ctx,
                                                             tex,
                                                             texture_data,
                                                             lvl->data_size,
                                                             upload_dst_offset,
                                                             dst_pitch,
                                                             upload_src_image_size,
                                                             face,
                                                             (GLuint)level,
                                                             (size_t)width,
                                                             (size_t)height,
                                                             (size_t)depth,
                                                             (size_t)xoffset,
                                                             (size_t)yoffset,
                                                             (size_t)zoffset);
        if (trace_upload) {
            fprintf(stderr,
                    "MGL TRACE texSubImage.directMTL call=%" PRIu64 " tex=%u face=%u level=%d uploaded=%d offset=%zu pitch=%zu image=%zu elapsed=%.3fms\n",
                    call_id,
                    tex->name,
                    face,
                    level,
                    uploaded_direct ? 1 : 0,
                    upload_dst_offset,
                    dst_pitch,
                    dst_image_pitch,
                    mglTextureNowMs() - start_ms);
        }
    } else if (trace_upload && (tex->dirty_bits & DIRTY_TEXTURE_LEVEL) != 0) {
        fprintf(stderr,
                "MGL TRACE texSubImage.directMTL skip call=%" PRIu64 " tex=%u face=%u level=%d reason=dirty-level dirty=0x%x pendingData=%d\n",
                call_id,
                tex->name,
                face,
                level,
                tex->dirty_bits,
                had_pending_texture_data_before_direct ? 1 : 0);
    }

    if (uploaded_direct && !had_pending_texture_data_before_direct) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
    } else {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    }
    mglTextureViewFamilyWritten(tex);

    /* SubImage onto a sparse texture may Map uncommitted tiles.  Zero the
     * CPU shadow above; scrub Metal so later imageLoad sees zeros too
     * (ARB_sparse_texture2 / UncommittedRegionsAccess, esp. r11f). */
    if (tex->params.sparse) {
        if (uploaded_direct) {
            mglFlushCommandBuffer(ctx);
            mglRendererFlush(ctx, true);
        }
        mglSparseScrubTextureLevel(tex, level, 1);
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE texSubImage.end call=%" PRIu64 " tex=%u face=%u level=%d upload=%s ok=1 elapsed=%.3fms\n",
                call_id,
                tex->name,
                face,
                level,
                uploaded_direct ? "DIRECT-MTL" : "DEFER",
                mglTextureNowMs() - start_ms);
    }
    
    return true;
}

#pragma mark texSubImage1D
void texSubImage1D(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    /* A/B: format and type must each be an accepted transfer constant. */
    ERROR_CHECK_RETURN(mglIsValidPixelTransferFormat(format), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(mglIsValidPixelTransferType(type), GL_INVALID_ENUM);

    /* E: level must not exceed log2 of the implementation's max texture size. */
    ERROR_CHECK_RETURN(level < (GLint)tex->num_levels, GL_INVALID_VALUE);

    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, tex->internalformat, format, type)) {
        return;
    }

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);

    /* F: xoffset must not be less than -border (border is 0 here). */
    ERROR_CHECK_RETURN(xoffset >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(width + xoffset <= tex->width, GL_INVALID_VALUE);

    texSubImage(ctx, tex, face, level, xoffset, 0, 0, width, 1, 1, format, type, (void *)pixels);
}

void mglTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    switch(target)
    {
        case GL_TEXTURE_1D:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    texSubImage1D(ctx, tex, 0, level, xoffset, width, format, type, pixels);
}

void mglTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    tex = getTex(ctx, texture, 0);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

   texSubImage1D(ctx, tex, 0, level, xoffset, width, format, type, pixels);
}

#pragma mark texSubImage2D
bool texSubImage2D(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    TextureLevel *lvl = NULL;

    ERROR_CHECK_RETURN_VALUE(face < _CUBE_MAP_MAX_FACE, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(level >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(tex != NULL, GL_INVALID_OPERATION, false);
    ERROR_CHECK_RETURN_VALUE(level < (GLint)tex->num_levels, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(tex->faces[face].levels != NULL, GL_INVALID_OPERATION, false);

    lvl = &tex->faces[face].levels[level];
    ERROR_CHECK_RETURN_VALUE(lvl->complete, GL_INVALID_OPERATION, false);

    /* A/B: format and type must each be an accepted transfer constant. */
    ERROR_CHECK_RETURN_VALUE(mglIsValidPixelTransferFormat(format), GL_INVALID_ENUM, false);
    ERROR_CHECK_RETURN_VALUE(mglIsValidPixelTransferType(type), GL_INVALID_ENUM, false);

    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, tex->internalformat, format, type)) {
        return false;
    }

    ERROR_CHECK_RETURN_VALUE(width >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(xoffset >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(yoffset >= 0, GL_INVALID_VALUE, false);

    ERROR_CHECK_RETURN_VALUE(width + xoffset <= (GLsizei)lvl->width, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height + yoffset <= (GLsizei)lvl->height, GL_INVALID_VALUE, false);

    return texSubImage(ctx, tex, face, level, xoffset, yoffset, 0, width, height, 1, format, type, (void *)pixels);
}

void mglTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    static uint64_t s_tex_sub_image2d_calls = 0u;
    uint64_t call_id = ++s_tex_sub_image2d_calls;
    double start_ms = mglTextureNowMs();
    Texture *tex;
    GLuint face;
    bool updated_ok;
    Buffer *unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    GLuint unpack_name = unpack_buf ? unpack_buf->name : 0u;
    bool trace_call = MGL_VERBOSE_TEXTURE_UPLOAD_LOGS ||
                      (mglTraceLogIsEnabled() &&
                       (unpack_name != 0u ||
                        (width >= 512 && height >= 512)));

    if (trace_call) {
        fprintf(stderr,
                "MGL TRACE mglTexSubImage2D.entry call=%" PRIu64 " target=0x%x level=%d off=(%d,%d) size=%dx%d format=0x%x type=0x%x "
                "unpackBufferName=%u pixelsRaw=%p rowLength=%d alignment=%d skipPixels=%d skipRows=%d skipImages=%d\n",
                call_id,
                target,
                level,
                xoffset,
                yoffset,
                width,
                height,
                format,
                type,
                unpack_name,
                pixels,
                STATE(unpack).row_length,
                STATE(unpack).alignment,
                STATE(unpack).skip_pixels,
                STATE(unpack).skip_rows,
                STATE(unpack).skip_images);
    }

    face = 0;

    switch(target)
    {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_RECTANGLE:
            break;

        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    {
        GLuint active_unit = STATE(active_texture);
        GLuint tex_index = textureIndexFromTarget(ctx, target);
        Texture *bound_tex = NULL;
        if (tex_index < _MAX_TEXTURE_TYPES) {
            bound_tex = STATE(texture_units[active_unit].textures[tex_index]);
        }
        if (mglTraceLogIsEnabled() && bound_tex && bound_tex->name == 13u) {
            trace_call = true;
        }

        if (trace_call) {
            fprintf(stderr,
                    "MGL TRACE mglTexSubImage2D.bound call=%" PRIu64 " activeUnit=%u target=0x%x texIndex=%u boundTex=%p boundName=%u boundTarget=0x%x\n",
                    call_id,
                    active_unit,
                    target,
                    tex_index,
                    (void *)bound_tex,
                    bound_tex ? bound_tex->name : 0u,
                    bound_tex ? bound_tex->target : 0u);
        }
    }

    tex = getTex(ctx, 0, target);

    if (!tex) {
        fprintf(stderr,
                "MGL ERROR: mglTexSubImage2D getTex returned NULL call=%" PRIu64 " target=0x%x level=%d\n",
                call_id,
                target,
                level);
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    updated_ok = texSubImage2D(ctx, tex, face, level, xoffset, yoffset, width, height, format, type, pixels);

    if (trace_call || (tex && tex->name == 13u) || !updated_ok) {
        fprintf(stderr,
                "MGL TRACE mglTexSubImage2D.exit call=%" PRIu64 " tex=%u face=%u level=%d ok=%d elapsed=%.3fms\n",
                call_id,
                tex ? tex->name : 0u,
                face,
                level,
                updated_ok ? 1 : 0,
                mglTextureNowMs() - start_ms);
    }
}

void mglTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    bool updated_ok;

    tex = getTex(ctx, texture, 0);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    updated_ok = texSubImage2D(ctx, tex, 0, level, xoffset, yoffset, width, height, format, type, pixels);
    if (!updated_ok) {
        return;
    }
}

#pragma mark texSubImage3D
bool texSubImage3D(GLMContext ctx, Texture *tex, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    TextureLevel *lvl = NULL;

    ERROR_CHECK_RETURN_VALUE(level >= 0, GL_INVALID_VALUE, false);

    ERROR_CHECK_RETURN_VALUE(tex, GL_INVALID_OPERATION, false);
    ERROR_CHECK_RETURN_VALUE(level < (GLint)tex->num_levels, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(tex->faces[0].levels != NULL, GL_INVALID_OPERATION, false);
    lvl = &tex->faces[0].levels[level];
    ERROR_CHECK_RETURN_VALUE(lvl->complete, GL_INVALID_OPERATION, false);

    /* A/B: format and type must each be an accepted transfer constant. */
    ERROR_CHECK_RETURN_VALUE(mglIsValidPixelTransferFormat(format), GL_INVALID_ENUM, false);
    ERROR_CHECK_RETURN_VALUE(mglIsValidPixelTransferType(type), GL_INVALID_ENUM, false);

    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, tex->internalformat, format, type)) {
        return false;
    }

    ERROR_CHECK_RETURN_VALUE(width >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(depth >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(xoffset >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(yoffset >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(zoffset >= 0, GL_INVALID_VALUE, false);

    ERROR_CHECK_RETURN_VALUE(width + xoffset <= (GLsizei)lvl->width, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height + yoffset <= (GLsizei)lvl->height, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(depth + zoffset <= (GLsizei)lvl->depth, GL_INVALID_VALUE, false);

    return texSubImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, type, (void *)pixels);
}

void mglTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    switch(target)
    {
        case GL_TEXTURE_3D:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    if (!texSubImage3D(ctx, tex, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels)) {
        return;
    }
}

void mglTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    tex = getTex(ctx, texture, 0);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    /* GL 4.6 §8.6: a cube map is addressed as six z-slices, one per face. */
    if (tex->target == GL_TEXTURE_CUBE_MAP) {
        MGLTextureUnpackLayout layout;
        size_t pixel_size = sizeForFormatType(format, type);

        ERROR_CHECK_RETURN(zoffset >= 0 && depth >= 0 && zoffset + depth <= _CUBE_MAP_MAX_FACE, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(pixel_size > 0u, GL_INVALID_ENUM);
        if (!mglComputeTextureUnpackLayout(ctx, width, height, 1, pixel_size, "TextureSubImage3D", &layout)) {
            return;
        }
        for (GLsizei i = 0; i < depth; i++) {
            if (!texSubImage2D(ctx, tex, (GLuint)(zoffset + i), level, xoffset, yoffset, width, height,
                               format, type, (const uint8_t *)pixels + (size_t)i * layout.src_image_size)) {
                return;
            }
        }
        return;
    }

    if (!texSubImage3D(ctx, tex, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels)) {
        return;
    }
}

#pragma mark TexStorage

/* ARB_sparse_texture2 Table 8.X.  Each page is 64KB.  Returns 1 when the
 * format is in the table.  Other formats get one implementation page. */
int mglSparseStandardPageSize(GLenum internalformat, GLint *x, GLint *y, GLint *z)
{
    GLint px = 64;
    GLint py = 64;
    int standard = 1;

    switch (internalformat) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R8I:
        case GL_R8UI:
            px = 256; py = 256; break;
        case GL_R16:
        case GL_R16_SNORM:
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RGB565:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
        case GL_RG8I:
        case GL_RG8UI:
            px = 256; py = 128; break;
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RG16F:
        case GL_R32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_R32I:
        case GL_R32UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RGBA8I:
        case GL_RGBA8UI:
            px = 128; py = 128; break;
        case GL_RGBA16:
        case GL_RGBA16_SNORM:
        case GL_RGBA16F:
        case GL_RG32F:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
            px = 128; py = 64; break;
        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            px = 64; py = 64; break;
        default:
            standard = 0;
            break;
    }
    if (x) *x = px;
    if (y) *y = py;
    if (z) *z = 1;
    return standard;
}

void mglSparseCommitReset(Texture *tex)
{
    free(tex->sparse_commits);
    tex->sparse_commits = NULL;
    tex->sparse_commit_count = 0;
    tex->sparse_commit_cap = 0;
}

int mglSparseCommitAppend(Texture *tex, GLint level,
                                 GLint x, GLint y, GLint z,
                                 GLint w, GLint h, GLint d)
{
    MGLSparseCommit *grown;

    if (w <= 0 || h <= 0 || d <= 0)
        return 1;
    if (tex->sparse_commit_count == tex->sparse_commit_cap) {
        GLuint cap = tex->sparse_commit_cap ? tex->sparse_commit_cap * 2u : 8u;
        grown = realloc(tex->sparse_commits, (size_t)cap * sizeof *grown);
        if (!grown)
            return 0;
        tex->sparse_commits = grown;
        tex->sparse_commit_cap = cap;
    }
    tex->sparse_commits[tex->sparse_commit_count++] = (MGLSparseCommit){
        level, x, y, z, w, h, d
    };
    return 1;
}

/* A texel is resident when any committed box on this level covers it.
 * Nothing committed means the whole level reads as zero. */
int mglSparseTexelCommitted(const Texture *tex, GLint level,
                                   GLint x, GLint y, GLint z)
{
    GLuint i;

    if (!tex->params.sparse)
        return 1;
    for (i = 0; i < tex->sparse_commit_count; i++) {
        const MGLSparseCommit *c = &tex->sparse_commits[i];
        if (c->level == level &&
            x >= c->x && x < c->x + c->w &&
            y >= c->y && y < c->y + c->h &&
            z >= c->z && z < c->z + c->d)
            return 1;
    }
    return 0;
}

/* ARB_sparse_texture2: a write that covers an uncommitted texel is discarded.
 * The CPU shadow is what GetTexImage and the following Metal upload read. */
void mglSparseZeroUncommitted(Texture *tex, GLuint face, GLint level,
                                     GLint xoffset, GLint yoffset, GLint zoffset,
                                     GLsizei width, GLsizei height, GLsizei depth)
{
    TextureLevel *lvl;
    size_t bpp;
    int zbase;
    GLsizei x, y, z;

    if (!tex || !tex->params.sparse || face >= 6u)
        return;
    if (!tex->faces[face].levels || level < 0 || (GLuint)level >= tex->num_levels)
        return;
    lvl = &tex->faces[face].levels[level];
    if (!lvl->data || lvl->width == 0 || lvl->pitch == 0)
        return;
    bpp = lvl->pitch / (size_t)lvl->width;
    if (bpp == 0)
        return;
    zbase = (tex->target == GL_TEXTURE_CUBE_MAP) ? (int)face : zoffset;
    for (z = 0; z < depth; z++) {
        for (y = 0; y < height; y++) {
            uint8_t *row = (uint8_t *)lvl->data +
                           (size_t)(zoffset + z) * lvl->pitch * (size_t)MAX(lvl->height, 1) +
                           (size_t)(yoffset + y) * lvl->pitch +
                           (size_t)xoffset * bpp;
            for (x = 0; x < width; x++) {
                if (!mglSparseTexelCommitted(tex, level,
                                             xoffset + x, yoffset + y, zbase + z))
                    memset(row + (size_t)x * bpp, 0, bpp);
            }
        }
    }
}

/* Drop the decommitted box out of every committed box. */
int mglSparseCommitSubtract(Texture *tex, GLint level,
                                   GLint x, GLint y, GLint z,
                                   GLint w, GLint h, GLint d)
{
    MGLSparseCommit *old = tex->sparse_commits;
    GLuint n = tex->sparse_commit_count;
    GLuint i;

    tex->sparse_commits = NULL;
    tex->sparse_commit_count = 0;
    tex->sparse_commit_cap = 0;
    for (i = 0; i < n; i++) {
        MGLSparseCommit a = old[i];
        GLint ix0, ix1, iy0, iy1, iz0, iz1;

        if (a.level != level) {
            if (!mglSparseCommitAppend(tex, a.level, a.x, a.y, a.z, a.w, a.h, a.d))
                goto oom;
            continue;
        }
        ix0 = a.x > x ? a.x : x;
        iy0 = a.y > y ? a.y : y;
        iz0 = a.z > z ? a.z : z;
        ix1 = (a.x + a.w) < (x + w) ? (a.x + a.w) : (x + w);
        iy1 = (a.y + a.h) < (y + h) ? (a.y + a.h) : (y + h);
        iz1 = (a.z + a.d) < (z + d) ? (a.z + a.d) : (z + d);
        if (ix0 >= ix1 || iy0 >= iy1 || iz0 >= iz1) {
            if (!mglSparseCommitAppend(tex, a.level, a.x, a.y, a.z, a.w, a.h, a.d))
                goto oom;
            continue;
        }
        if (!mglSparseCommitAppend(tex, level, a.x, a.y, a.z, ix0 - a.x, a.h, a.d) ||
            !mglSparseCommitAppend(tex, level, ix1, a.y, a.z, a.x + a.w - ix1, a.h, a.d) ||
            !mglSparseCommitAppend(tex, level, ix0, a.y, a.z, ix1 - ix0, iy0 - a.y, a.d) ||
            !mglSparseCommitAppend(tex, level, ix0, iy1, a.z, ix1 - ix0, a.y + a.h - iy1, a.d) ||
            !mglSparseCommitAppend(tex, level, ix0, iy0, a.z, ix1 - ix0, iy1 - iy0, iz0 - a.z) ||
            !mglSparseCommitAppend(tex, level, ix0, iy0, iz1, ix1 - ix0, iy1 - iy0, a.z + a.d - iz1))
            goto oom;
    }
    free(old);
    return 1;
oom:
    free(old);
    mglSparseCommitReset(tex);
    return 0;
}

/* Reject a sparse TexStorage that violates ARB_sparse_texture2, and record
 * how many levels can be committed a page at a time.  Returns false with the
 * GL error already set.  texture2 drops the "base size must be a whole number
 * of pages" error. */
bool mglSparseTexStorageAllowed(GLMContext ctx, Texture *tex,
                                      GLenum internalformat,
                                      GLsizei levels, GLsizei width,
                                      GLsizei height, GLsizei depth)
{
    GLuint max_side = STATE(var).max_texture_size;
    GLuint max_3d = STATE(var).max_3d_texture_size;
    GLuint max_layers = STATE(var).max_array_texture_layers;
    GLuint w;
    GLuint h;
    GLint page_x, page_y, page_z;
    int level;

    /* Only index 0 is advertised (NUM_VIRTUAL_PAGE_SIZES is 1). */
    if (tex->params.virtual_page_size_index != 0) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    if (tex->target == GL_TEXTURE_3D) {
        if ((GLuint)width > max_3d || (GLuint)height > max_3d ||
            (GLuint)depth > max_3d) {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
            return false;
        }
    } else if ((GLuint)width > max_side || (GLuint)height > max_side) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
        return false;
    }

    if ((tex->target == GL_TEXTURE_2D_ARRAY ||
         tex->target == GL_TEXTURE_CUBE_MAP_ARRAY ||
         tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) &&
        (GLuint)depth > max_layers) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
        return false;
    }

    mglSparseStandardPageSize(internalformat, &page_x, &page_y, &page_z);
    tex->sparse_page_x = page_x;
    tex->sparse_page_y = page_y;
    tex->sparse_page_z = page_z;
    mglSparseCommitReset(tex);

    /* A level can be partially committed while every dimension is at least
     * one page.  The first smaller level starts the mipmap tail. */
    tex->params.num_sparse_levels = 0;
    w = (GLuint)width;
    h = (GLuint)height;
    for (level = 0; level < levels; level++) {
        if (w < (GLuint)page_x || h < (GLuint)page_y)
            break;
        tex->params.num_sparse_levels++;
        if (w > 1u) w /= 2u;
        if (h > 1u) h /= 2u;
    }
    return true;
}

void texStorage(GLMContext ctx, Texture *tex, GLuint faces, GLsizei levels, GLboolean is_array, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean proxy)
{
    if (!proxy && tex->params.sparse &&
        !mglSparseTexStorageAllowed(ctx, tex, internalformat, levels, width, height, depth)) {
        return;
    }

    tex->access = GL_READ_ONLY;

    for(int face=0; face<faces; face++)
    {
        for(int level=0; level<levels; level++)
        {
            GLuint level_width = 1u;
            GLuint level_height = 1u;
            GLuint level_depth = 1u;
            mglTextureTargetLevelDimensions(tex->target,
                                            (GLuint)width,
                                            (GLuint)height,
                                            (GLuint)depth,
                                            (GLuint)level,
                                            &level_width,
                                            &level_height,
                                            &level_depth);
            createTextureLevel(ctx, tex, face, level, is_array, internalformat,
                               level_width, level_height, level_depth,
                               0, 0, NULL, proxy);
        }
    }

    /*
     * TexStorage declares the exact immutable mip count. initBaseTexLevel()
     * derives a full chain from the base size, which is correct for legacy
     * TexImage completeness but too strict for Minecraft atlases such as
     * 2048x2048x4. Keep the allocated level arrays, but make completeness and
     * the Metal descriptor use the GL-declared storage level count.
     */
    if (levels > 0) {
        tex->mipmap_levels = (GLuint)levels;
        tex->num_levels = MAX(tex->num_levels, (GLuint)levels);
    }

    // mark it immutable
    tex->immutable_storage = BUFFER_IMMUTABLE_STORAGE_FLAG;
    if (!tex->view_root) {
        tex->immutable_levels = (GLuint)levels;
        tex->view_num_levels = (GLuint)levels;
        tex->view_num_layers = tex->target == GL_TEXTURE_1D_ARRAY ? (GLuint)height
                             : faces == 6u ? 6u
                             : is_array ? (GLuint)depth : 1u;
    }

    // bind it to metal
    mglRendererBindTexture(ctx, tex);

    /* F24: this check used to pass unconditionally because the silent fallback
     * installs a non-NULL gradient substitute on failure, so the one "loud"
     * failure report in the texture-creation paths was defeated by the very
     * mechanism it was meant to catch.  Reject the substitute too. */
    ERROR_CHECK_RETURN(mglRenderTextureHandle(tex) && !mglRenderTextureStorageIsFallback(tex), GL_OUT_OF_MEMORY);
}

