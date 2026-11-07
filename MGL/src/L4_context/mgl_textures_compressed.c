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

void mglCompressedTexImage3D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data)
{
    if (target != GL_TEXTURE_3D &&
        target != GL_TEXTURE_2D_ARRAY &&
        target != GL_TEXTURE_CUBE_MAP_ARRAY) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    (void)mglStoreCompressedTextureImage(ctx, target, level, internalformat, width, height, depth, border, imageSize, data);
}

void mglCompressedTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data)
{
    switch (target) {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            (void)mglStoreCompressedTextureImage(ctx, target, level, internalformat, width, height, 1, border, imageSize, data);
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglCompressedTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data)
{
    if (target != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    (void)mglStoreCompressedTextureImage(ctx, target, level, internalformat, width, 1, 1, border, imageSize, data);
}

void mglCompressedTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    Texture *tex = NULL;
    switch (target) {
        case GL_TEXTURE_3D:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            tex = getTex(ctx, 0, target);
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
    (void)mglCompressedSubImageUpdate(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
}

void mglCompressedTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    GLuint face = 0u;
    Texture *tex = NULL;
    switch (target) {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            tex = getTex(ctx, 0, target);
            break;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            face = (GLuint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
            tex = getTex(ctx, 0, target);
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
    (void)mglCompressedSubImageUpdate(ctx, tex, face, level, xoffset, yoffset, 0, width, height, 1, format, imageSize, data);
}

void mglCompressedTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    if (target != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    Texture *tex = getTex(ctx, 0, target);
    (void)mglCompressedSubImageUpdate(ctx, tex, 0, level, xoffset, 0, 0, width, 1, 1, format, imageSize, data);
}


void mglCopyTexImageCommon(GLMContext ctx, GLenum target, GLuint face, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    if (level < 0 || width < 0 || height < 0 || border != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_TEXTURE_RECTANGLE && level != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!mglValidateReadFramebuffer(ctx)) {
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    /* Per the GL 4.6 spec, CopyTexImage2D with a depth+stencil internalformat
     * requires the read framebuffer to have both depth AND stencil attached.
     * If only one (or neither) is attached, generate INVALID_OPERATION.
     * Similarly, a depth-only internalformat requires depth attached, and a
     * stencil-only internalformat requires stencil attached. */
    {
        Framebuffer *readFBO = STATE(readbuffer);
        if (readFBO)
        {
            GLboolean depth_attached = (readFBO->depth.texture != 0u);
            GLboolean stencil_attached = (readFBO->stencil.texture != 0u);

            if (internalformat == GL_DEPTH_STENCIL ||
                internalformat == GL_DEPTH24_STENCIL8 ||
                internalformat == GL_DEPTH32F_STENCIL8)
            {
                if (depth_attached != stencil_attached)
                {
                    ERROR_RETURN(GL_INVALID_OPERATION);
                    return;
                }
            }
        }
    }

    // Get or create texture
    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* glCopyTexImage2D has no client pixel data (the source is the framebuffer),
     * but createTextureLevel validates the (format,type) pair against the
     * internalformat.  Pick a compatible pair so depth / integer / packed
     * internalformats pass that validation instead of returning
     * GL_INVALID_OPERATION. */
    GLenum srcFormat = GL_BGRA;
    GLenum srcType   = GL_UNSIGNED_BYTE;
    switch (internalformat) {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            srcFormat = GL_DEPTH_COMPONENT;
            srcType   = GL_FLOAT;
            break;
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_DEPTH_STENCIL:
            srcFormat = GL_DEPTH_STENCIL;
            srcType = (internalformat == GL_DEPTH32F_STENCIL8)
                          ? GL_FLOAT_32_UNSIGNED_INT_24_8_REV
                          : GL_UNSIGNED_INT_24_8;
            break;
        case GL_RGB10_A2UI:
            srcFormat = GL_RGBA_INTEGER;
            srcType   = GL_UNSIGNED_INT_2_10_10_10_REV;
            break;
        case GL_RGB9_E5:
            srcFormat = GL_RGB;
            srcType   = GL_UNSIGNED_INT_5_9_9_9_REV;
            break;
        default:
            break;
    }

    if (!createTextureLevel(ctx,
                            tex,
                            face,
                            level,
                            false,
                            internalformat,
                            width,
                            height,
                            1,
                            srcFormat,
                            srcType,
                            NULL,
                            false)) {
        return;
    }

    // Copy from framebuffer to texture
    mglFlushPendingDraws(ctx);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    /* Packed depth-stencil cannot use Metal texture-to-texture copies (stencil
     * is dropped). Read the FBO as DEPTH_STENCIL and TexSubImage the dest. */
    if (mglInternalFormatIsCombinedDepthStencil(internalformat)) {
        GLenum dsType = (internalformat == GL_DEPTH32F_STENCIL8)
                            ? GL_FLOAT_32_UNSIGNED_INT_24_8_REV
                            : GL_UNSIGNED_INT_24_8;
        size_t px = (dsType == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) ? 8u : 4u;
        void *tmp = malloc((size_t)width * (size_t)height * px);
        if (!tmp) {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
        mglReadPixels(ctx, x, y, width, height, GL_DEPTH_STENCIL, dsType, tmp);
        mglTexSubImage2D(ctx, target, level, 0, 0, width, height,
                         GL_DEPTH_STENCIL, dsType, tmp);
        free(tmp);
        return;
    }

    mglRendererCopyTexSubImage(ctx, tex, face, level, 0, 0, x, y, width, height);
}

/* GL 4.6 §8.6: CopyTexImage1D is CopyTexImage2D with a height of 1. */
void mglCopyTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
{
    if (target != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    mglCopyTexImageCommon(ctx, target, 0u, level, internalformat, x, y, width, 1, border);
}

void mglCopyTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    GLuint face = 0u;

    if (!mglCopyTex2DFaceForTarget(target, &face)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    mglCopyTexImageCommon(ctx, target, face, level, internalformat, x, y, width, height, border);
}

void mglCopyTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    if (target != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (!mglValidateReadFramebuffer(ctx)) {
        return;
    }
    if (width == 0) {
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, 0, 0, width, 1)) {
        return;
    }
    mglFlushPendingDraws(ctx);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    mglRendererCopyTexSubImage(ctx, tex, 0, level, xoffset, 0, x, y, width, 1);
}

void mglCopyTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    GLuint face = 0u;

    if (!mglCopyTex2DFaceForTarget(target, &face)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || xoffset < 0 || yoffset < 0 || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_TEXTURE_RECTANGLE && level != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!mglValidateReadFramebuffer(ctx)) {
        return;
    }
    if (width == 0 || height == 0) {
        return;
    }

    // Get the bound texture
    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        fprintf(stderr, "MGL ERROR: glCopyTexSubImage2D - no texture bound\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (level >= (GLint)tex->num_levels ||
        !tex->faces[face].levels ||
        !tex->faces[face].levels[level].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[face].levels[level];
    if ((GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    // This copies from the current read framebuffer to the texture; pending
    // draws may still be writing the read attachment, not just the texture.
    mglFlushPendingDraws(ctx);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    mglRendererCopyTexSubImage(ctx, tex, face, level, xoffset, yoffset, x, y, width, height);
}

void mglCopyTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (target != GL_TEXTURE_3D &&
        target != GL_TEXTURE_2D_ARRAY &&
        target != GL_TEXTURE_CUBE_MAP_ARRAY) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (!mglValidateReadFramebuffer(ctx)) {
        return;
    }
    if (width == 0 || height == 0) {
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, yoffset, zoffset, width, height)) {
        return;
    }
    if ((GLuint)zoffset >= tex->faces[0].levels[level].depth) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    mglFlushPendingDraws(ctx);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    mglRendererCopyTexSubImage(ctx, tex, (GLuint)zoffset, level, xoffset, yoffset, x, y, width, height);
}

void mglCopyTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    if (texture == 0) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!mglValidateReadFramebuffer(ctx)) {
        return;
    }
    if (width == 0) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex || tex->target != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, 0, 0, width, 1)) {
        return;
    }
    mglFlushPendingDraws(ctx);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    mglRendererCopyTexSubImage(ctx, tex, 0, level, xoffset, 0, x, y, width, 1);
}

void mglCopyTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL: glCopyTextureSubImage2D called - texture=%u %dx%d\n", texture, width, height);
    if (texture == 0 || level < 0 || xoffset < 0 || yoffset < 0 || width < 0 || height < 0) {
        ERROR_RETURN(texture == 0 ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
        return;
    }
    if (!mglValidateReadFramebuffer(ctx)) {
        return;
    }
    if (width == 0 || height == 0) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (tex &&
        tex->target != GL_TEXTURE_2D &&
        tex->target != GL_TEXTURE_1D_ARRAY &&
        tex->target != GL_TEXTURE_RECTANGLE) {
        tex = NULL;
    }
    if (tex) {
        if (level >= (GLint)tex->num_levels ||
            !tex->faces[0].levels ||
            !tex->faces[0].levels[level].complete) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        TextureLevel *lvl = &tex->faces[0].levels[level];
        if ((GLuint)xoffset > lvl->width ||
            (GLuint)yoffset > lvl->height ||
            (GLuint)width > lvl->width - (GLuint)xoffset ||
            (GLuint)height > lvl->height - (GLuint)yoffset) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        mglFlushPendingDraws(ctx);

        /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
        if (ctx->sync_strict) {
            mglFlushCommandBuffer(ctx);
            mglRendererFlush(ctx, true);
        }

        mglRendererCopyTexSubImage(ctx, tex, 0, level, xoffset, yoffset, x, y, width, height);
    } else {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
}

void mglCopyTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (texture == 0) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!mglValidateReadFramebuffer(ctx)) {
        return;
    }
    if (width == 0 || height == 0) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex ||
        (tex->target != GL_TEXTURE_3D &&
         tex->target != GL_TEXTURE_2D_ARRAY &&
         tex->target != GL_TEXTURE_CUBE_MAP_ARRAY &&
         tex->target != GL_TEXTURE_CUBE_MAP)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    GLuint slice = 0u;
    if (tex->target == GL_TEXTURE_CUBE_MAP) {
        /* Table 8.15: DSA 3D copy accepts TEXTURE_CUBE_MAP; zoffset is the
         * face index (GL 4.6 §8.6). */
        if (zoffset < 0 || zoffset >= 6) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        slice = (GLuint)zoffset;
        if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, yoffset,
                                            0, width, height)) {
            return;
        }
        if (!tex->faces[slice].levels ||
            !tex->faces[slice].levels[level].complete) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    } else {
        if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, yoffset,
                                            zoffset, width, height)) {
            return;
        }
        if ((GLuint)zoffset >= tex->faces[0].levels[level].depth) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        slice = (GLuint)zoffset;
    }
    mglFlushPendingDraws(ctx);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    mglRendererCopyTexSubImage(ctx, tex, slice, level, xoffset, yoffset, x, y, width, height);
}

#pragma mark get tex image

/* With a PIXEL_PACK_BUFFER bound, `*pixels` is a byte offset into it;
 * redirect it to the buffer's CPU storage. */
