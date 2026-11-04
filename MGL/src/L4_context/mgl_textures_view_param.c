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

void mglTextureViewFamilyWritten(Texture *writer)
{
    Texture *root = writer ? (writer->view_root ? writer->view_root : writer) : NULL;
    if (!root || root->view_count == 0u ||
        !mglRenderTargetHasGPUContent(writer->is_render_target,
                                        mglTextureWriteVersion(writer))) {
        return;
    }
    for (GLuint i = 0; i <= root->view_count; i++) {
        Texture *member = i < root->view_count ? root->views[i] : root;
        if (member == writer || (member == root && root->view_root_deleted)) {
            continue;
        }
        member->is_render_target = GL_TRUE;
        mglTextureBumpWriteVersion(member);
    }
}

/* Table 8.21. */
bool mglTextureViewTargetCompatible(GLenum orig, GLenum target)
{
    switch (orig) {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_1D_ARRAY:
            return target == GL_TEXTURE_1D || target == GL_TEXTURE_1D_ARRAY;
        case GL_TEXTURE_2D:
            return target == GL_TEXTURE_2D || target == GL_TEXTURE_2D_ARRAY;
        case GL_TEXTURE_3D:
        case GL_TEXTURE_RECTANGLE:
            return target == orig;
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            return target == GL_TEXTURE_2D || target == GL_TEXTURE_2D_ARRAY ||
                   target == GL_TEXTURE_CUBE_MAP ||
                   target == GL_TEXTURE_CUBE_MAP_ARRAY;
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return target == GL_TEXTURE_2D_MULTISAMPLE ||
                   target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
        default:
            return false;
    }
}

/* Returns true when `tex` owns a store that views still reference, in which
 * case the struct must outlive its name. */
bool mglTextureViewUnlink(GLMContext ctx, Texture *tex)
{
    Texture *root = tex->view_root;
    if (root) {
        for (GLuint i = 0; i < root->view_count; i++) {
            if (root->views[i] == tex) {
                root->views[i] = root->views[--root->view_count];
                break;
            }
        }
        if (mglRenderTextureViewParentHandle(tex)) {
            mglRendererDeleteMetalObject(ctx, mglRenderTextureViewParentHandle(tex));
            mglRenderTextureClearViewParentHandle(tex);
        }
        if (root->view_root_deleted && root->view_count == 0u) {
            free(root->views);
            free(root);
        }
        return false;
    }
    if (tex->view_count) {
        tex->view_root_deleted = GL_TRUE;
        return true;
    }
    free(tex->views);
    return false;
}

void mglTextureView(GLMContext ctx, GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
{
    ERROR_CHECK_RETURN(texture != 0u, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(texture <= STATE(texture_table).current_name &&
                       !findTexture(ctx, texture), GL_INVALID_OPERATION);
    Texture *orig = findTexture(ctx, origtexture);
    ERROR_CHECK_RETURN(orig != NULL, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(orig->immutable_storage, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(mglTextureViewTargetCompatible(orig->target, target),
                       GL_INVALID_OPERATION);
    GLenum orig_class = mglTextureViewClass(orig->internalformat);
    ERROR_CHECK_RETURN(orig_class != GL_NONE
                           ? mglTextureViewClass(internalformat) == orig_class
                           : (GLenum)orig->internalformat == internalformat,
                       GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(minlevel < orig->view_num_levels &&
                       minlayer < orig->view_num_layers, GL_INVALID_VALUE);

    GLuint levels = orig->view_num_levels - minlevel;
    if (numlevels < levels) levels = numlevels;
    GLuint layers = orig->view_num_layers - minlayer;
    if (numlayers < layers) layers = numlayers;

    switch (target) {
        case GL_TEXTURE_CUBE_MAP:
            ERROR_CHECK_RETURN(layers == 6u, GL_INVALID_VALUE);
            break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            ERROR_CHECK_RETURN(layers % 6u == 0u, GL_INVALID_VALUE);
            break;
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_2D_MULTISAMPLE:
            ERROR_CHECK_RETURN(numlayers == 1u, GL_INVALID_VALUE);
            break;
        default:
            break;
    }

    const TextureLevel *base = &orig->faces[0].levels[minlevel];
    GLuint width = base->width;
    GLuint height = (orig->target == GL_TEXTURE_1D ||
                     orig->target == GL_TEXTURE_1D_ARRAY) ? 1u : base->height;
    if (target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        ERROR_CHECK_RETURN(width == height, GL_INVALID_OPERATION);
        ERROR_CHECK_RETURN(width <= STATE(var).max_cube_map_texture_size,
                           GL_INVALID_OPERATION);
    }
    ERROR_CHECK_RETURN(levels > 0u && layers > 0u, GL_INVALID_OPERATION);

    mglFlushPendingDraws(ctx);

    Texture *root = orig->view_root ? orig->view_root : orig;
    if (!root->view_root_deleted) {
        /* Pending CPU uploads must reach the shared store before every
         * family member starts reading it back from Metal.  For MS parents,
         * also force RenderTarget usage *before* the view is registered so
         * BindViewStorage does not recreate the store under views>0. */
        const int root_ms =
            root->target == GL_TEXTURE_2D_MULTISAMPLE ||
            root->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
        if (root_ms) {
            root->is_render_target = GL_TRUE;
        }
        mglRendererBindTexture(ctx, root);
        if (root_ms) {
            root->is_render_target = GL_FALSE;
        }
        ERROR_CHECK_RETURN(mglRenderTextureHandle(root) && !mglRenderTextureStorageIsFallback(root),
                           GL_OUT_OF_MEMORY);
        root->metal_data_authoritative = GL_TRUE;
    }
    if (root->view_count == root->view_capacity) {
        GLuint capacity = root->view_capacity ? root->view_capacity * 2u : 4u;
        Texture **views = realloc(root->views, capacity * sizeof(*views));
        ERROR_CHECK_RETURN(views != NULL, GL_OUT_OF_MEMORY);
        root->views = views;
        root->view_capacity = capacity;
    }

    Texture *view = getTexture(ctx, target, texture);
    if (!view) {
        return;
    }
    root->views[root->view_count++] = view;
    view->view_root = root;
    view->view_min_level = orig->view_min_level + minlevel;
    view->view_num_levels = levels;
    view->view_min_layer = orig->view_min_layer + minlayer;
    view->view_num_layers = layers;
    view->immutable_levels = orig->immutable_levels;
    if (mglRenderTargetHasGPUContent(orig->is_render_target,
                                       mglTextureWriteVersion(orig))) {
        view->is_render_target = GL_TRUE;
        mglTextureSetWriteVersion(view, 1u);
    }

    GLuint depth = orig->target == GL_TEXTURE_3D ? base->depth : 1u;
    GLboolean is_array = GL_FALSE;
    switch (target) {
        case GL_TEXTURE_1D_ARRAY:
            height = layers;
            is_array = GL_TRUE;
            break;
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            depth = layers;
            is_array = GL_TRUE;
            break;
        default:
            break;
    }

    if (target == GL_TEXTURE_2D_MULTISAMPLE ||
        target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        if (!mglTextureStorageMultisampleMetadata(ctx, view, target,
                                                  (GLsizei)orig->samples,
                                                  internalformat,
                                                  (GLsizei)width,
                                                  (GLsizei)height,
                                                  (GLsizei)depth,
                                                  orig->fixed_sample_locations,
                                                  GL_FALSE)) {
            return;
        }
        mglRendererBindTexture(ctx, view);
        ERROR_CHECK_RETURN(mglRenderTextureHandle(view) != NULL, GL_OUT_OF_MEMORY);
    } else {
        texStorage(ctx, view, target == GL_TEXTURE_CUBE_MAP ? 6u : 1u,
                   (GLsizei)levels, is_array, internalformat, (GLsizei)width,
                   (GLsizei)height, (GLsizei)depth, GL_FALSE);
    }
    /* Set after storage: level creation resets it.  The view's CPU levels
     * are never filled from the shared store. */
    view->metal_data_authoritative = GL_TRUE;
}

GLboolean mglTexBufferInternalFormatValid(GLenum internalformat)
{
    switch (internalformat) {
        case GL_R8:
        case GL_R16:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R16I:
        case GL_R32I:
        case GL_R8UI:
        case GL_R16UI:
        case GL_R32UI:
        case GL_RG8:
        case GL_RG16:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RG8I:
        case GL_RG16I:
        case GL_RG32I:
        case GL_RG8UI:
        case GL_RG16UI:
        case GL_RG32UI:
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGBA8:
        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_RGBA8I:
        case GL_RGBA16I:
        case GL_RGBA32I:
        case GL_RGBA8UI:
        case GL_RGBA16UI:
        case GL_RGBA32UI:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

void mglTextureBufferRangeImpl(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer,
                                      GLintptr offset, GLsizeiptr size, bool whole_buffer);

void mglTextureBuffer(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer)
{
    mglTextureBufferRangeImpl(ctx, texture, internalformat, buffer, 0, 0, true);
}

void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    mglTextureBufferRangeImpl(ctx, texture, internalformat, buffer, offset, size, false);
}

void mglTextureBufferRangeImpl(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer,
                                      GLintptr offset, GLsizeiptr size, bool whole_buffer)
{
    Texture *tex;
    Buffer *buf = NULL;
    size_t bytes_per_texel;
    GLsizeiptr attach_size;

    ERROR_CHECK_RETURN(texture != 0, GL_INVALID_OPERATION);

    tex = findTexture(ctx, texture);
    /* Texture name 0 is represented by a lazily-created default object in
     * the texture-unit state, not by an entry in the object hash table.  The
     * glTexBuffer(..., buffer = 0) reset path reaches this helper with that
     * object's synthetic name, so recover the currently bound buffer texture
     * before reporting INVALID_OPERATION. */
    if (!tex) {
        Texture *bound = currentTexture(ctx, _TEXTURE_BUFFER_TARGET);
        if (bound && bound->name == texture) {
            tex = bound;
        }
    }
    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(tex->target == GL_TEXTURE_BUFFER, GL_INVALID_OPERATION);

    /* GL 4.6 Table 8.15 / §8.9: TexBuffer internalformat must be a sized
     * color format from that table. DEPTH_COMPONENT32F is sized but not
     * allowed (CTS texture_buffer_errors). */
    ERROR_CHECK_RETURN(mglTexBufferInternalFormatValid(internalformat), GL_INVALID_ENUM);

    /* GL 4.6 §8.9 error list: "An INVALID_VALUE error is generated if offset
     * is not an integer multiple of the value of TEXTURE_BUFFER_OFFSET_ALIGNMENT."
     * glTexBuffer always passes offset 0, so only the Range entry point can
     * violate this; gate on whole_buffer to keep the two paths distinct. */
    if (!whole_buffer && offset > 0) {
        GLuint tb_align = STATE(var).texture_buffer_offset_alignment;
        if (tb_align == 0u || tb_align > 4096u) {
            tb_align = 16u;   /* same fallback as the alignment query */
        }
        ERROR_CHECK_RETURN((offset % (GLintptr)tb_align) == 0, GL_INVALID_VALUE);
    }

    mglFlushPendingDraws(ctx);

    if (buffer == 0)
    {
        if (mglRenderTextureHandle(tex))
        {
            mglRendererDeleteMetalObject(ctx, mglRenderTextureHandle(tex));
        }

        tex->texture_buffer = NULL;
        tex->texture_buffer_offset = 0;
        tex->texture_buffer_size = 0;
        tex->texture_buffer_whole = GL_FALSE;
        tex->internalformat = internalformat;
        tex->width = 0;
        tex->height = 1;
        tex->depth = 1;
        tex->complete = GL_FALSE;
        mglRenderTextureForgetHandle(tex);
        tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);
        return;
    }

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);
    bytes_per_texel = sizeForInternalFormat(internalformat, 0, 0);
    ERROR_CHECK_RETURN(bytes_per_texel != 0, GL_INVALID_ENUM);

    buf = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(buf, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(buf->size >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(offset >= 0, GL_INVALID_VALUE);

    if (whole_buffer)
    {
        attach_size = buf->size - offset;
    }
    else
    {
        ERROR_CHECK_RETURN(size > 0, GL_INVALID_VALUE);
        attach_size = size;
    }

    ERROR_CHECK_RETURN(attach_size >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(offset <= buf->size, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(attach_size <= buf->size - offset, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(((size_t)attach_size / bytes_per_texel) > 0, GL_INVALID_VALUE);

    /*
     * Some render paths re-specify the same texel-buffer attachment before
     * each draw. GL treats that as the same association, so keep the existing
     * Metal texture alive and let bindMTLTexture refresh it only when the
     * source buffer itself is dirty.
     */
    if (tex->texture_buffer == buf &&
        tex->texture_buffer_offset == offset &&
        tex->texture_buffer_size == attach_size &&
        tex->internalformat == internalformat &&
        tex->complete == GL_TRUE)
    {
        tex->dirty_bits &= ~DIRTY_TEXTURE_LEVEL;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);

        static uint64_t s_texBufferUnchangedLogs = 0;
        uint64_t hit = ++s_texBufferUnchangedLogs;
        if (hit <= 8ull || (hit % 2048ull) == 0ull) {
            fprintf(stderr,
                    "MGL TRACE TexBuffer unchanged hit=%llu texture=%u internal=0x%x buffer=%u offset=%lld size=%lld texels=%u bpt=%zu dirty=0x%x bufferDirty=0x%x\n",
                    (unsigned long long)hit,
                    texture,
                    internalformat,
                    buffer,
                    (long long)offset,
                    (long long)attach_size,
                    tex->width,
                    bytes_per_texel,
                    tex->dirty_bits,
                    buf->data.dirty_bits);
        }
        return;
    }

    if (mglRenderTextureHandle(tex))
    {
        mglRendererDeleteMetalObject(ctx, mglRenderTextureHandle(tex));
        mglRenderTextureForgetHandle(tex);
    }

    tex->texture_buffer = buf;
    tex->texture_buffer_offset = offset;
    tex->texture_buffer_size = attach_size;
    tex->texture_buffer_whole = whole_buffer ? GL_TRUE : GL_FALSE;
    tex->internalformat = internalformat;
    /* GL 4.6 §8.9: the texel count is clamped to MAX_TEXTURE_BUFFER_SIZE. */
    size_t texel_count = (size_t)attach_size / bytes_per_texel;
    if (STATE(var).max_texture_buffer_size &&
        texel_count > STATE(var).max_texture_buffer_size)
        texel_count = STATE(var).max_texture_buffer_size;
    tex->width = (GLuint)texel_count;
    tex->height = 1;
    tex->depth = 1;
    tex->is_array = GL_FALSE;
    tex->complete = GL_TRUE;
    tex->num_levels = 1;
    tex->mipmap_levels = 1;
    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);

    {
        static uint64_t s_texBufferCreateLogs = 0;
        uint64_t hit = ++s_texBufferCreateLogs;
        if (hit <= 4ull || (hit % 2048ull) == 0ull) {
            fprintf(stderr,
                    "MGL TRACE TexBuffer texture=%u internal=0x%x buffer=%u offset=%lld size=%lld texels=%u bpt=%zu\n",
                    texture,
                    internalformat,
                    buffer,
                    (long long)offset,
                    (long long)attach_size,
                    tex->width,
                    bytes_per_texel);
        }
    }
}

void mglCompressedTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    Texture *tex = getTex(ctx, texture, 0);
    (void)mglCompressedSubImageUpdate(ctx, tex, 0, level, xoffset, 0, 0, width, 1, 1, format, imageSize, data);
}void mglCompressedTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    Texture *tex = getTex(ctx, texture, 0);
    (void)mglCompressedSubImageUpdate(ctx, tex, 0, level, xoffset, yoffset, 0, width, height, 1, format, imageSize, data);
}

void mglCompressedTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    Texture *tex = getTex(ctx, texture, 0);
    (void)mglCompressedSubImageUpdate(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
}

void mglGetCompressedTextureImage(GLMContext ctx, GLuint texture, GLint level, GLsizei bufSize, void *pixels)
{
    if (bufSize < 0 || level < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    /* PIXEL_PACK_BUFFER: pixels is a byte offset, not a client pointer. */
    Buffer *pack_buffer = NULL;
    uintptr_t pack_offset = (uintptr_t)pixels;
    if (!mglResolvePackBufferDst(ctx, "glGetCompressedTextureImage", &pixels,
                                 &pack_buffer)) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
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
    if (lvl->data_size > (size_t)bufSize) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!mglCheckPackBufferRange(ctx, pack_buffer, pack_offset, lvl->data_size,
                                 GL_UNSIGNED_BYTE)) {
        return;
    }
    if (lvl->data && lvl->data_size > 0u) {
        memcpy(pixels, (const void *)(uintptr_t)lvl->data, lvl->data_size);
    }
}

void mglGetTextureLevelParameteriv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *params)
{
    TextureLevel *tex_level = NULL;
    GLint internalformat = 0;

    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (texture == 0 || level < 0) {
        ERROR_RETURN(texture == 0 ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* GL 4.6 §8.11: level > log2(MAX_TEXTURE_SIZE) is INVALID_VALUE.
     * TEXTURE_COMPRESSED_IMAGE_SIZE on uncompressed storage is
     * INVALID_OPERATION (CTS textures_level_parameter_errors). */
    if (level > (GLint)ilog2(STATE(var).max_texture_size)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (pname == GL_TEXTURE_COMPRESSED_IMAGE_SIZE &&
        !mglTexLevelInternalFormatCompressed((GLint)tex->internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    {
        GLenum tb_err = GL_NO_ERROR;
        int tb = mglTryTexBufferLevelParameter(tex, level, pname, params, &tb_err);
        if (tb < 0) {
            ERROR_RETURN(tb_err);
            return;
        }
        if (tb > 0)
            return;
    }

    if (level >= (GLint)tex->num_levels || !tex->faces[0].levels) {
        switch (pname) {
            case GL_TEXTURE_WIDTH:
            case GL_TEXTURE_HEIGHT:
            case GL_TEXTURE_DEPTH:
            case GL_TEXTURE_INTERNAL_FORMAT:
            case GL_TEXTURE_RED_SIZE:
            case GL_TEXTURE_GREEN_SIZE:
            case GL_TEXTURE_BLUE_SIZE:
            case GL_TEXTURE_ALPHA_SIZE:
            case GL_TEXTURE_DEPTH_SIZE:
            case GL_TEXTURE_STENCIL_SIZE:
            case GL_TEXTURE_COMPRESSED:
            case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
            case GL_TEXTURE_SAMPLES:
            case GL_TEXTURE_SHARED_SIZE:
                *params = 0;
                return;
            case GL_TEXTURE_RED_TYPE:
            case GL_TEXTURE_GREEN_TYPE:
            case GL_TEXTURE_BLUE_TYPE:
            case GL_TEXTURE_ALPHA_TYPE:
            case GL_TEXTURE_DEPTH_TYPE:
                *params = GL_NONE;
                return;
            case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
                *params = GL_TRUE;
                return;
            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }
    }

    tex_level = &tex->faces[0].levels[level];
    internalformat = tex->internalformat;
    switch (pname) {
        case GL_TEXTURE_WIDTH:
            *params = (GLint)tex_level->width;
            break;
        case GL_TEXTURE_HEIGHT:
            *params = (GLint)tex_level->height;
            break;
        case GL_TEXTURE_DEPTH:
            *params = (GLint)tex_level->depth;
            break;
        case GL_TEXTURE_INTERNAL_FORMAT:
            *params = (GLint)mglTextureQueryInternalFormat(tex);
            break;
        case GL_TEXTURE_RED_SIZE:
        case GL_TEXTURE_GREEN_SIZE:
        case GL_TEXTURE_BLUE_SIZE:
        case GL_TEXTURE_ALPHA_SIZE:
        case GL_TEXTURE_DEPTH_SIZE:
        case GL_TEXTURE_STENCIL_SIZE:
            *params = mglTexLevelComponentBits(internalformat, pname);
            break;
        case GL_TEXTURE_COMPRESSED:
            *params = mglTexLevelInternalFormatCompressed((GLint)tex->internalformat) ? GL_TRUE : GL_FALSE;
            break;
        case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
            *params = mglTextureQueryCompressedImageSize(tex, level);
            break;
        case GL_TEXTURE_RED_TYPE:
        case GL_TEXTURE_GREEN_TYPE:
        case GL_TEXTURE_BLUE_TYPE:
        case GL_TEXTURE_ALPHA_TYPE:
        case GL_TEXTURE_DEPTH_TYPE:
            *params = mglTexLevelComponentType(internalformat, pname);
            break;
        case GL_TEXTURE_SAMPLES:
            *params = (tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                       tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) ? (GLint)tex->samples : 0;
            break;
        case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
            *params = (tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                       tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) ? tex->fixed_sample_locations : GL_TRUE;
            break;
        case GL_TEXTURE_SHARED_SIZE:
            *params = (mglTexLevelCanonicalInternalFormat(internalformat) == GL_RGB9_E5) ? 5 : 0;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglGetTextureLevelParameterfv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLfloat *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLint iparams;
    mglGetTextureLevelParameteriv(ctx, texture, level, pname, &iparams);
    *params = (GLfloat)iparams;
}

bool mglTextureParameterGetiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *params)
{
    if (!tex_params || !params)
        return false;

    switch (pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; i++)
                params[i] = (GLint)tex_params->border_color[i];
            return true;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = tex_params->swizzle_r;
            params[1] = tex_params->swizzle_g;
            params[2] = tex_params->swizzle_b;
            params[3] = tex_params->swizzle_a;
            return true;
        case GL_TEXTURE_MIN_LOD:
            *params = (GLint)tex_params->min_lod;
            return true;
        case GL_TEXTURE_MAX_LOD:
            *params = (GLint)tex_params->max_lod;
            return true;
        case GL_TEXTURE_LOD_BIAS:
            *params = (GLint)tex_params->lod_bias;
            return true;
        case GL_TEXTURE_MAX_ANISOTROPY:
            *params = (GLint)tex_params->max_anisotropy;
            return true;
        default:
        {
            GLfloat fparam = 0.0f;
            return getParam(ctx, tex_params, pname, params, &fparam);
        }
    }
}

bool mglTextureParameterGetfv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLfloat *params)
{
    if (!tex_params || !params)
        return false;

    switch (pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; i++)
                params[i] = tex_params->border_color[i];
            return true;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = (GLfloat)tex_params->swizzle_r;
            params[1] = (GLfloat)tex_params->swizzle_g;
            params[2] = (GLfloat)tex_params->swizzle_b;
            params[3] = (GLfloat)tex_params->swizzle_a;
            return true;
        case GL_TEXTURE_MIN_LOD:
            *params = tex_params->min_lod;
            return true;
        case GL_TEXTURE_MAX_LOD:
            *params = tex_params->max_lod;
            return true;
        case GL_TEXTURE_LOD_BIAS:
            *params = tex_params->lod_bias;
            return true;
        case GL_TEXTURE_MAX_ANISOTROPY:
            *params = tex_params->max_anisotropy;
            return true;
        default:
        {
            GLint iparam = 0;
            if (getParam(ctx, tex_params, pname, &iparam, params)) {
                *params = (GLfloat)iparam;
                return true;
            }
            return false;
        }
    }
}

bool mglTextureParameterGetIiv(TextureParameter *tex_params, GLenum pname, GLint *params)
{
    if (!tex_params || !params)
        return false;

    if (pname == GL_TEXTURE_BORDER_COLOR) {
        for (int i = 0; i < 4; i++)
            params[i] = tex_params->border_color_i[i];
        return true;
    }

    return false;
}

bool mglTextureParameterGetIuiv(TextureParameter *tex_params, GLenum pname, GLuint *params)
{
    if (!tex_params || !params)
        return false;

    if (pname == GL_TEXTURE_BORDER_COLOR) {
        for (int i = 0; i < 4; i++)
            params[i] = tex_params->border_color_ui[i];
        return true;
    }

    return false;
}

bool mglTextureParameterGetTarget(GLMContext ctx, Texture *tex, GLenum pname, GLfloat *fparams,
                                         GLint *iparams, GLuint *uiparams)
{
    if (!tex)
        return false;

    /* Texture-level parameters that live on Texture, not TextureParameter. */
    if (pname == GL_TEXTURE_IMMUTABLE_FORMAT) {
        GLint val = tex->immutable_storage ? GL_TRUE : GL_FALSE;
        if (fparams) { *fparams = (GLfloat)val; return true; }
        if (iparams) { *iparams = val; return true; }
        if (uiparams) { *uiparams = (GLuint)val; return true; }
        return true;
    }
    GLint val = 0;
    if (mglTextureViewParameter(tex, pname, &val)) {
        if (fparams) { *fparams = (GLfloat)val; return true; }
        if (iparams) { *iparams = val; return true; }
        if (uiparams) { *uiparams = (GLuint)val; return true; }
        return true;
    }

    if (pname != GL_TEXTURE_TARGET)
        return false;

    if (tex->target == GL_TEXTURE_BUFFER) {
        if (ctx) {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        }
        return true;
    }

    if (fparams) {
        *fparams = (GLfloat)tex->target;
        return true;
    }

    if (iparams) {
        *iparams = (GLint)tex->target;
        return true;
    }

    if (uiparams) {
        *uiparams = (GLuint)tex->target;
        return true;
    }

    (void)ctx;
    return false;
}

void mglGetTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, GLfloat *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (mglTextureParameterGetTarget(ctx, tex, pname, params, NULL, NULL))
        return;

    if (!mglTextureParameterGetfv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (mglTextureParameterGetTarget(ctx, tex, pname, NULL, params, NULL))
        return;

    if (mglTextureParameterGetIiv(&tex->params, pname, params))
        return;

    if (!mglTextureParameterGetiv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, GLuint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (mglTextureParameterGetTarget(ctx, tex, pname, NULL, NULL, params))
        return;

    if (mglTextureParameterGetIuiv(&tex->params, pname, params))
        return;

    GLint iparam = 0;
    if (mglTextureParameterGetiv(ctx, &tex->params, pname, &iparam)) {
        *params = (GLuint)iparam;
        return;
    }

    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (mglTextureParameterGetTarget(ctx, tex, pname, NULL, params, NULL))
        return;

    if (!mglTextureParameterGetiv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

bool mglGetTexParameterITargetValid(GLenum target)
{
    /* Same set as TexParameter targets: object targets only — not cube-map
     * faces and not GL_TEXTURE_BUFFER (GL 4.6 §8.10). */
    switch (target) {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
        default:
            return false;
    }
}

void mglGetTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!mglGetTexParameterITargetValid(target)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex)
        return;

    if (pname == GL_TEXTURE_IMMUTABLE_FORMAT) {
        *params = tex->immutable_storage ? GL_TRUE : GL_FALSE;
        return;
    }
    if (mglTextureViewParameter(tex, pname, params))
        return;

    if (pname == GL_IMAGE_FORMAT_COMPATIBILITY_TYPE) {
        *params = (GLint)GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE;
        return;
    }

    if (mglTextureParameterGetIiv(&tex->params, pname, params))
        return;

    if (!mglTextureParameterGetiv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, GLuint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!mglGetTexParameterITargetValid(target)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex)
        return;

    if (pname == GL_TEXTURE_IMMUTABLE_FORMAT) {
        *params = (GLuint)(tex->immutable_storage ? GL_TRUE : GL_FALSE);
        return;
    }
    {
        GLint view_value = 0;
        if (mglTextureViewParameter(tex, pname, &view_value)) {
            *params = (GLuint)view_value;
            return;
        }
    }

    if (pname == GL_IMAGE_FORMAT_COMPATIBILITY_TYPE) {
        *params = (GLuint)GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE;
        return;
    }

    if (mglTextureParameterGetIuiv(&tex->params, pname, params))
        return;

    GLint iparam = 0;
    if (mglTextureParameterGetiv(ctx, &tex->params, pname, &iparam)) {
        *params = (GLuint)iparam;
        return;
    }

    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglSampleCoverage(GLMContext ctx, GLfloat value, GLboolean invert)
{
    if (!ctx)
        return;

    STATE(var.sample_coverage_value) = clamp(value, 0.0f, 1.0f);
    STATE(var.sample_coverage_invert) = invert ? GL_TRUE : GL_FALSE;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
}
