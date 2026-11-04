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

void mglBindTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    GLuint active_texture;
    GLint index;
    Texture *ptr;

    if (!ctx)
        return;

    if (MGL_VERBOSE_TEXTURE_BIND_LOGS) {
        fprintf(stderr,
                "MGL TRACE BindTexture target=0x%x texture=%u activeUnit=%u ctx=%p\n",
                target,
                texture,
                ctx ? STATE(active_texture) : 0u,
                (void *)ctx);
    }

    if (!mglIsTextureObjectTarget(target))
    {
        fprintf(stderr, "MGL Error: mglBindTexture: invalid target 0x%x\n", (unsigned)target);
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (texture)
    {
        ptr = getTexture(ctx, target, texture);
        if (!ptr) {
            fprintf(stderr,
                    "MGL Error: mglBindTexture failed to resolve/create texture=%u target=0x%x\n",
                    texture,
                    target);
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
    }
    else
    {
        ptr = NULL;
    }

    active_texture = STATE(active_texture);
    if (active_texture >= TEXTURE_UNITS) {
        fprintf(stderr,
                "MGL ERROR: mglBindTexture active unit out of range unit=%u target=0x%x texture=%u\n",
                active_texture,
                target,
                texture);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *old_typed_ptr = STATE(texture_units[active_texture].textures[index]);
    Texture *old_active_ptr = STATE(active_textures[active_texture]);
    bool pending_write = ptr && mglPendingDrawsWriteTexture(ctx, ptr);
    bool binding_changed = (old_typed_ptr != ptr) || (ptr && old_active_ptr != ptr);
    if (!binding_changed && !pending_write) {
        return;
    }
    if (pending_write) {
        /* PERF-014: only flush draws that hazard this texture, not the whole
         * pending CB. */
        MGL_PERF_INC(g_mglFlushReasonOtherSinceSwap);
        mglFlushPendingDrawsForTexture(ctx, ptr);
    } else if (binding_changed && !mglBindNoFlushEnabled()) {
        MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
        mglFlushPendingDraws(ctx);
    }

    STATE(texture_units[active_texture].textures[index]) = ptr;
    if (ptr) {
        STATE(active_textures[active_texture]) = ptr;
    }
    mglUpdateTextureUnitActiveMask(ctx, active_texture);
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);

    mglTraceTextureUnitState(ctx, ptr ? "BindTexture" : "BindTexture.unbindTarget", active_texture, target, texture, STATE(active_textures[active_texture]));
}

GLboolean mglTextureTargetUsesImageLayerParameter(GLenum target)
{
    switch (target) {
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

/* GL 4.6 Table 8.26 — formats accepted by BindImageTexture <format>. */
GLboolean mglIsLegalImageUnitFormat(GLenum format)
{
    switch (format) {
        case GL_RGBA32F:
        case GL_RGBA16F:
        case GL_RG32F:
        case GL_RG16F:
        case GL_R11F_G11F_B10F:
        case GL_R32F:
        case GL_R16F:
        case GL_RGBA32UI:
        case GL_RGBA16UI:
        case GL_RGB10_A2UI:
        case GL_RGBA8UI:
        case GL_RG32UI:
        case GL_RG16UI:
        case GL_RG8UI:
        case GL_R32UI:
        case GL_R16UI:
        case GL_R8UI:
        case GL_RGBA32I:
        case GL_RGBA16I:
        case GL_RGBA8I:
        case GL_RG32I:
        case GL_RG16I:
        case GL_RG8I:
        case GL_R32I:
        case GL_R16I:
        case GL_R8I:
        case GL_RGBA16:
        case GL_RGB10_A2:
        case GL_RGBA8:
        case GL_RG16:
        case GL_RG8:
        case GL_R16:
        case GL_R8:
        case GL_RGBA16_SNORM:
        case GL_RGBA8_SNORM:
        case GL_RG16_SNORM:
        case GL_RG8_SNORM:
        case GL_R16_SNORM:
        case GL_R8_SNORM:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

void mglBindImageTexture(GLMContext ctx, GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum internalformat)
{
    Texture *ptr;

    /* Per the GL 4.6 spec, glBindImageTexture generates GL_INVALID_VALUE if
     * <unit> is greater than or equal to GL_MAX_IMAGE_UNITS.  MGL reports
     * GL_MAX_IMAGE_UNITS == 16 (independent of TEXTURE_UNITS == 128). */
    if (unit >= STATE(var).max_image_units) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: unit >= max_image_units (%d)\n", unit);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    /* Format must be validated even when texture==0 (unbind): CTS
     * negative-bind expects INVALID_VALUE for an illegal <format>. */
    if (!mglIsLegalImageUnitFormat(internalformat)) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: illegal format 0x%x\n", internalformat);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (texture == 0u) {
        ImageUnit *iu = &STATE(image_units)[unit];
        if (iu->texture != 0u || iu->tex != NULL) {
            /* Image-unit rebinds are not covered by dynamic texture capture /
             * MGL_BIND_NO_FLUSH batch merge; flush so pending draws keep the
             * pre-rebind snapshot (non-layered layer switches). */
            mglFlushPendingDraws(ctx);
            mglRendererFlushImageUnitSlice(ctx, unit);
        }
        mglResetImageUnit(iu);
        mglMarkStateDirtyBits(&ctx->state, DIRTY_IMAGE_UNIT_STATE);
        return;
    }

    /* findTexture, not getTex: getTex reports INVALID_OPERATION for a
     * missing name, but BindImageTexture must report INVALID_VALUE
     * (GL 4.6 / ES 3.1; CTS negative-bind texture=123). */
    ptr = findTexture(ctx, texture);

    if (!ptr) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: texture %d not found\n", texture);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (level < 0) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: level < 0 (%d)\n", level);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    /* GL 4.6 §8.26 / ES 3.1: INVALID_VALUE if layer is negative, including
     * non-array 2D (CTS es_31_compatibility.negative-bind). */
    if (layer < 0) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: layer < 0 (%d)\n", layer);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch(access)
    {
        case GL_READ_ONLY:
        case GL_WRITE_ONLY:
        case GL_READ_WRITE:
            break;

        default:
            fprintf(stderr, "MGL Error: mglBindImageTexture: invalid access 0x%x\n", access);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    /* GL 4.6 §8.26 Errors: BindImageTexture does not reject incomplete or
     * mutable textures.  Loads from incomplete images return 0 and stores
     * are ignored (image completeness), which is handled at access time. */

    /* Spec: an incompatible <format> vs texture internalformat makes image
     * loads/stores undefined; it is not a BindImageTexture error. CTS
     * basic-api-bind intentionally rebinds one R32F texture with RGBA8/RG16/
     * R32I image formats. Keep the requested format on the image unit. */

    /* CTS advanced-cast: RGBA8 bound as R32I/UI.  PixelFormatView of RGBA8
     * cannot take ShaderAtomic, so promote the Metal store to R32* while
     * leaving GL internalformat alone for GetTexImage bit-cast reads. */
    if (ptr->target != GL_TEXTURE_BUFFER &&
        (internalformat == GL_R32I || internalformat == GL_R32UI ||
         internalformat == GL_R32F)) {
        int known_tex = 0, known_bind = 0;
        const uint32_t bpp_tex =
            mglRenderBytesPerPixelForInternalFormat(ptr->internalformat,
                                                    &known_tex);
        const uint32_t bpp_bind =
            mglRenderBytesPerPixelForInternalFormat(internalformat,
                                                    &known_bind);
        const uint32_t native_mtl =
            mtlFormatForGLInternalFormat(ptr->internalformat);
        if (known_tex && known_bind && bpp_tex == bpp_bind && bpp_tex == 4u &&
            !mglRenderPixelFormatNeedsShaderAtomic(native_mtl) &&
            ptr->image_metal_format != internalformat) {
            ptr->image_metal_format = internalformat;
            ptr->dirty_bits |= DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA;
            {
                GLuint iu;
                for (iu = 0u; iu < STATE(var).max_image_units; iu++) {
                    if (STATE(image_units)[iu].tex == ptr) {
                        mglImageUnitBackendRelease(&STATE(image_units)[iu]);
                    }
                }
            }
            mglRendererBindTexture(ctx, ptr);
        }
    }

    /* GL_TEXTURE_BUFFER has no mipmap faces/levels array; completeness is
     * tracked on tex->complete itself and only level==0 is valid per spec. */
    if (ptr->target == GL_TEXTURE_BUFFER) {
        if (level != 0) {
            fprintf(stderr, "MGL Error: mglBindImageTexture: level must be 0 for texture buffer (got %d)\n", level);
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        if (!ptr->complete || !ptr->texture_buffer) {
            fprintf(stderr, "MGL Error: mglBindImageTexture: incomplete texture buffer %u\n", texture);
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
    } else {
        /* BindImageTexture records the requested level even if that mip is
         * not allocated.  Missing-level loads return 0 and stores are
         * ignored (GL 4.6 image completeness; CTS incomplete_textures).
         * ES 3.1 / KHR-GL46.es_31_compatibility.basic-api-bind also binds
         * level==immutable_levels (TexStorage levels=4, bind level=4) and
         * expects the unit to take that binding with no error.  Reject only
         * levels past TEXTURE_MAX_LEVEL. */
        {
            GLint max_level = (GLint)ptr->params.max_level;
            if (max_level < 0) {
                max_level = 1000;
            }
            if (level > max_level) {
                fprintf(stderr, "MGL Error: mglBindImageTexture: level %d > TEXTURE_MAX_LEVEL %d\n",
                        level, max_level);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
        }
        if (!layered &&
            mglTextureTargetUsesImageLayerParameter(ptr->target) &&
            ptr->faces[0].levels &&
            level < (GLint)ptr->num_levels &&
            ptr->faces[0].levels[level].complete) {
            /* GL_TEXTURE_1D_ARRAY stores slice count in height. Cube maps
             * store each face as a separate 2D level (depth==1); layer still
             * selects the face. Cube arrays / 2D arrays / 3D use depth. */
            GLuint slice_count;
            if (ptr->target == GL_TEXTURE_1D_ARRAY) {
                slice_count = ptr->faces[0].levels[level].height;
            } else if (ptr->target == GL_TEXTURE_CUBE_MAP) {
                slice_count = 6u;
            } else {
                slice_count = ptr->faces[0].levels[level].depth;
            }
            if (layer >= (GLint)slice_count) {
                fprintf(stderr, "MGL Error: mglBindImageTexture: layer %d out of range (slices=%u)\n",
                        layer, slice_count);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
        }
    }

    
    ImageUnit unit_params;

    /* Image-unit access is per-binding, not a texture-object property.
     * Mutating tex->access here used to set DIRTY_TEXTURE_ACCESS and
     * recreate the Metal texture (wiping prior imageStore contents) when
     * CTS rebound WRITE_ONLY → READ_ONLY between store and load draws. */

    unit_params.texture = texture;
    unit_params.level = level;
    unit_params.layered = layered;
    unit_params.layer = layer;
    unit_params.access = access;
    unit_params.internalformat = internalformat;
    unit_params.tex = ptr;
    unit_params.backend = NULL;

    {
        const ImageUnit *cur = &STATE(image_units)[unit];
        const bool binding_changed =
            cur->texture != unit_params.texture ||
            cur->level != unit_params.level ||
            cur->layered != unit_params.layered ||
            cur->layer != unit_params.layer ||
            cur->access != unit_params.access ||
            cur->internalformat != unit_params.internalformat ||
            cur->tex != unit_params.tex;
        if (binding_changed) {
            /* No per-draw image-unit override in batch merge; always flush.
             * MGL_BIND_NO_FLUSH dynamic texture capture ignores texture_hash. */
            mglFlushPendingDraws(ctx);
            mglRendererFlushImageUnitSlice(ctx, unit);
        }
    }

    mglImageUnitBackendRelease(&STATE(image_units)[unit]);
    STATE(image_units)[unit] = unit_params;

    mglMarkStateDirtyBits(&ctx->state, DIRTY_IMAGE_UNIT_STATE);
    mglRendererPrepareImageUnitSlice(ctx, unit);
}

/* Callback for mglHashTableForEach: detach a deleted texture from every FBO
 * attachment that still holds a raw pointer to it.  Runs before the texture
 * is freed so the pointer is still valid for comparison. */
void mglDetachTextureFromFramebuffers(GLuint name, void *data, void *user)
{
    (void)name;
    Framebuffer *fbo = (Framebuffer *)data;
    Texture *deleted = (Texture *)user;

    if (!fbo || !deleted) {
        return;
    }

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (fbo->color_attachments[i].buf.tex == deleted) {
            fbo->color_attachments[i].buf.tex = NULL;
            fbo->color_attachments[i].texture = 0u;
            fbo->dirty_bits |= DIRTY_FBO_TEX;
        }
    }
    if (fbo->depth.buf.tex == deleted) {
        fbo->depth.buf.tex = NULL;
        fbo->depth.texture = 0u;
        fbo->dirty_bits |= DIRTY_FBO_TEX;
    }
    if (fbo->stencil.buf.tex == deleted) {
        fbo->stencil.buf.tex = NULL;
        fbo->stencil.texture = 0u;
        fbo->dirty_bits |= DIRTY_FBO_TEX;
    }
}

bool mglTextureViewUnlink(GLMContext ctx, Texture *tex);

void mglDeleteTextures(GLMContext ctx, GLsizei n, const GLuint *textures)
{
    if (!ctx || n <= 0 || !textures)
        return;

    mglFlushPendingDraws(ctx);

    while(n--)
    {
        GLuint name;

        name = *textures++;
        if (name == 0)
            continue;

        Texture *tex;

        tex = findTexture(ctx, name);

        if(tex)
        {
            for(int i=0; i<TEXTURE_UNITS; i++)
            {
                GLboolean cleared_unit = GL_FALSE;

                if(STATE(active_textures)[i] == tex) {
                    STATE(active_textures)[i] = NULL;
                    cleared_unit = GL_TRUE;
                }

                for (int target_index = 0; target_index < _MAX_TEXTURE_TYPES; target_index++) {
                    if (STATE(texture_units)[i].textures[target_index] == tex) {
                        STATE(texture_units)[i].textures[target_index] = NULL;
                        cleared_unit = GL_TRUE;
                    }
                }

                if (cleared_unit) {
                    mglUpdateTextureUnitActiveMask(ctx, (GLuint)i);
                    mglMarkStateDirtyBits(&ctx->state, DIRTY_TEX_BINDING);
                }
            }

            for(int i=0; i<TEXTURE_UNITS; i++)
            {
                if(STATE(image_units)[i].texture == name)
                {
                    mglResetImageUnit(&STATE(image_units)[i]);

                    mglMarkStateDirtyBits(&ctx->state, DIRTY_IMAGE_UNIT_STATE);
                }
            }

            GLenum store_format = tex->internalformat;
            invalidateTexture(ctx, tex);

            /* OpenGL spec: when a texture is deleted, it is detached from any
             * framebuffer it is bound to.  MGL stores raw Texture* pointers in
             * FBOAttachment.buf.tex; without clearing them here the pointers
             * become dangling after free(tex) below, causing use-after-free
             * crashes in later draw calls that scan FBO attachments. */
            mglHashTableForEach(&STATE(framebuffer_table),
                                mglDetachTextureFromFramebuffers, tex);

            deleteHashElement(&STATE(texture_table), name);
            if (mglTextureViewUnlink(ctx, tex)) {
                tex->internalformat = store_format;
                continue;
            }
            /* One placement heap per sparse texture; nothing else holds it. */
            if (tex->sparse_heap) {
                mglRenderReleaseMetalObject(tex->sparse_heap);
                tex->sparse_heap = NULL;
                tex->sparse_heap_next_tile = 0;
            }
            free(tex->sparse_commits);
            tex->sparse_commits = NULL;
            mglRenderTextureReleaseBackend(ctx, tex);
            free(tex);
        }
    }
}

GLboolean mglIsTexture(GLMContext ctx, GLuint texture)
{
    return isTexture(ctx, texture);
}

void mglInvalidateTexImage(GLMContext ctx, GLuint texture, GLint level)
{
    Texture *tex = findTexture(ctx, texture);

    if (!tex) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (level < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (tex->num_levels > 0 && level >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    /* Invalidation is a hint; after validation it is legal to leave storage as-is. */
}

void mglInvalidateTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth)
{
    Texture *tex = findTexture(ctx, texture);

    if (!tex) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (level < 0 || xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (tex->num_levels > 0 && level >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (tex->faces[0].levels && level < (GLint)tex->num_levels) {
        TextureLevel *lvl = &tex->faces[0].levels[level];
        if (lvl->complete &&
            (xoffset > (GLint)lvl->width ||
             yoffset > (GLint)lvl->height ||
             zoffset > (GLint)lvl->depth ||
             width > (GLsizei)(lvl->width - (GLuint)xoffset) ||
             height > (GLsizei)(lvl->height - (GLuint)yoffset) ||
             depth > (GLsizei)(lvl->depth - (GLuint)zoffset))) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
    }

    /* Invalidation is a hint; after validation it is legal to leave storage as-is. */
}

void mglBindImageTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    if (!ctx) {
        return;
    }
    if (count < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (count == 0) {
        return;
    }
    {
        GLuint max_units = STATE(var).max_image_units;
        if (max_units == 0u) {
            max_units = 8u;
        }
        /* GL 4.6 §8.26: INVALID_OPERATION if first + count > MAX_IMAGE_UNITS. */
        if (first >= max_units || (GLuint)count > max_units - first) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    /* Values are checked per image unit. An invalid entry leaves that unit
     * unchanged and still generates an error; other units with valid values
     * are updated (GL 4.6 §8.26). */
    for (GLsizei i = 0; i < count; i++) {
        GLuint tex_name = textures ? textures[i] : 0u;
        if (tex_name == 0u) {
            mglResetImageUnit(&STATE(image_units)[first + i]);
            continue;
        }

        Texture *tex = findTexture(ctx, tex_name);
        if (!tex) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            continue;
        }

        GLenum format = tex->internalformat;
        GLuint w = tex->width;
        GLuint h = tex->height;
        GLuint d = tex->depth;
        if (tex->target != GL_TEXTURE_BUFFER &&
            tex->faces[0].levels && tex->num_levels > 0u) {
            w = tex->faces[0].levels[0].width;
            h = tex->faces[0].levels[0].height;
            d = tex->faces[0].levels[0].depth;
        }
        if (!mglIsLegalImageUnitFormat(format) || w == 0u || h == 0u || d == 0u) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            continue;
        }

        mglBindImageTexture(ctx,
                            first + i,
                            tex_name,
                            0,
                            GL_TRUE,
                            0,
                            GL_READ_WRITE,
                            format);
    }

    mglMarkStateDirtyBits(&ctx->state, DIRTY_IMAGE_UNIT_STATE);
}

void mglClientActiveTexture(GLMContext ctx, GLenum texture)
{
    if (texture < GL_TEXTURE0 ||
        (GLuint)(texture - GL_TEXTURE0) >= STATE_VAR(max_combined_texture_image_units)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    /* Legacy fixed-function client texture state is otherwise unused by MGL. */
}

void mglActiveTexture(GLMContext ctx, GLenum texture)
{
    GLuint unit;

    if (texture < GL_TEXTURE0)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    unit = (GLuint)(texture - GL_TEXTURE0);

    if (unit >= TEXTURE_UNITS || unit >= STATE_VAR(max_combined_texture_image_units))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (STATE(active_texture) == unit) {
        return;
    }

    STATE(active_texture) = unit;
    mglMarkRendererDirtyBits(&ctx->state, DIRTY_TEX_BINDING);
    mglTraceTextureUnitState(ctx, "ActiveTexture", unit, 0, 0, STATE(active_textures[unit]));
}

void mglBindTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    GLuint old_active_texture;

    if (!ctx) {
        return;
    }

    if (count < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (count == 0) {
        return;
    }

    if (first > TEXTURE_UNITS ||
        (GLuint)count > TEXTURE_UNITS - first ||
        first > STATE_VAR(max_combined_texture_image_units) ||
        (GLuint)count > STATE_VAR(max_combined_texture_image_units) - first) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    old_active_texture = STATE(active_texture);

    GLboolean any_changed = GL_FALSE;
    for (int i=0; i < count; i++)
    {
        GLuint texture;
        GLuint unit = first + (GLuint)i;

        if (textures == NULL)
        {
            texture = 0;
        }
        else
        {
            texture = textures[i];
        }

        if (texture != 0)
        {
            Texture *ptr;
            GLuint index;

            ptr = findTexture(ctx, texture);
            if (!ptr) {
                fprintf(stderr,
                        "MGL ERROR: mglBindTextures unknown texture=%u unit=%u first=%u count=%d\n",
                        texture,
                        unit,
                        first,
                        count);
                ERROR_RETURN(GL_INVALID_OPERATION);
                continue;
            }

            index = ptr->index;
            if (index >= _MAX_TEXTURE_TYPES) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                continue;
            }

            Texture *old_typed_ptr = STATE(texture_units[unit].textures[index]);
            Texture *old_active_ptr = STATE(active_textures[unit]);
            bool pending_write = mglPendingDrawsWriteTexture(ctx, ptr);
            bool binding_changed = (old_typed_ptr != ptr) || (old_active_ptr != ptr);
            if (!binding_changed && !pending_write) {
                continue;
            }
            if (pending_write) {
                MGL_PERF_INC(g_mglFlushReasonOtherSinceSwap);
                mglFlushPendingDrawsForTexture(ctx, ptr);
            } else if (binding_changed && !mglBindNoFlushEnabled()) {
                MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
                mglFlushPendingDraws(ctx);
            }

            STATE(texture_units[unit].textures[index]) = ptr;
            STATE(active_textures[unit]) = ptr;
            mglUpdateTextureUnitActiveMask(ctx, unit);
            mglTraceTextureUnitState(ctx, "BindTextures", unit, ptr->target, texture, ptr);
            any_changed = GL_TRUE;
        }
        else
        {
            GLboolean had_binding = GL_FALSE;
            for(GLuint index=0; index<_MAX_TEXTURE_TYPES; index++)
            {
                if (STATE(texture_units[unit].textures[index]) != NULL) {
                    had_binding = GL_TRUE;
                }
            }
            if (STATE(active_textures[unit]) != NULL) {
                had_binding = GL_TRUE;
            }
            if (had_binding) {
                if (!mglBindNoFlushEnabled()) {
                    MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
                    mglFlushPendingDraws(ctx);
                }
            } else {
                continue;
            }
            for(GLuint index=0; index<_MAX_TEXTURE_TYPES; index++)
            {
                STATE(texture_units[unit].textures[index]) = NULL;
            }
            STATE(active_textures[unit]) = NULL;
            mglUpdateTextureUnitActiveMask(ctx, unit);
            mglTraceTextureUnitState(ctx, "BindTextures.unbind", unit, 0, 0, NULL);
            any_changed = GL_TRUE;
        }
    }

    STATE(active_texture) = old_active_texture;
    if (any_changed) {
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);
    }
}

void mglBindTextureUnit(GLMContext ctx, GLuint unit, GLuint texture)
{
    Texture *ptr;
    GLuint index;

    if (!ctx) {
        return;
    }

    if (unit >= TEXTURE_UNITS) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (texture == 0) {
        GLboolean had_binding = GL_FALSE;
        for (index = 0; index < _MAX_TEXTURE_TYPES; index++) {
            if (STATE(texture_units[unit].textures[index]) != NULL) {
                had_binding = GL_TRUE;
            }
        }
        if (STATE(active_textures[unit]) != NULL) {
            had_binding = GL_TRUE;
        }
        if (had_binding) {
            if (!mglBindNoFlushEnabled()) {
                MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
                mglFlushPendingDraws(ctx);
            }
        }
        for (index = 0; index < _MAX_TEXTURE_TYPES; index++) {
            STATE(texture_units[unit].textures[index]) = NULL;
        }
        STATE(active_textures[unit]) = NULL;
        mglUpdateTextureUnitActiveMask(ctx, unit);
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);
        mglTraceTextureUnitState(ctx, "BindTextureUnit.unbind", unit, 0, 0, NULL);
        return;
    }

    ptr = findTexture(ctx, texture);
    if (!ptr) {
        fprintf(stderr,
                "MGL ERROR: mglBindTextureUnit unknown texture=%u unit=%u\n",
                texture,
                unit);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    index = ptr->index;
    if (index >= _MAX_TEXTURE_TYPES) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *old_typed_ptr = STATE(texture_units[unit].textures[index]);
    Texture *old_active_ptr = STATE(active_textures[unit]);
    bool pending_write = mglPendingDrawsWriteTexture(ctx, ptr);
    bool binding_changed = (old_typed_ptr != ptr) || (old_active_ptr != ptr);
    if (!binding_changed && !pending_write) {
        return;
    }
    if (pending_write) {
        MGL_PERF_INC(g_mglFlushReasonOtherSinceSwap);
        mglFlushPendingDrawsForTexture(ctx, ptr);
    } else if (binding_changed && !mglBindNoFlushEnabled()) {
        MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
        mglFlushPendingDraws(ctx);
    }

    STATE(texture_units[unit].textures[index]) = ptr;
    STATE(active_textures[unit]) = ptr;
    mglUpdateTextureUnitActiveMask(ctx, unit);
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);
    mglTraceTextureUnitState(ctx, "BindTextureUnit", unit, ptr->target, texture, ptr);
}

GLuint mglTextureTargetMaxLevels(GLenum target,
                                        GLuint width,
                                        GLuint height,
                                        GLuint depth)
{
    switch (target) {
        case GL_TEXTURE_1D:
        case GL_PROXY_TEXTURE_1D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_PROXY_TEXTURE_1D_ARRAY:
            return maxLevels(width, 1u, 1u);
        case GL_TEXTURE_3D:
        case GL_PROXY_TEXTURE_3D:
            return maxLevels(width, height, depth);
        case GL_TEXTURE_RECTANGLE:
        case GL_PROXY_TEXTURE_RECTANGLE:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return 1u;
        default:
            return maxLevels(width, height, 1u);
    }
}

void mglTextureTargetLevelDimensions(GLenum target,
                                            GLuint base_width,
                                            GLuint base_height,
                                            GLuint base_depth,
                                            GLuint level,
                                            GLuint *out_width,
                                            GLuint *out_height,
                                            GLuint *out_depth)
{
    GLuint width = (GLuint)mglRenderMetalTextureLevelDimension(base_width, level);
    GLuint height = (GLuint)mglRenderMetalTextureLevelDimension(base_height, level);
    GLuint depth = (GLuint)mglRenderMetalTextureLevelDimension(base_depth, level);

    switch (target) {
        case GL_TEXTURE_1D:
        case GL_PROXY_TEXTURE_1D:
            height = 1u;
            depth = 1u;
            break;
        case GL_TEXTURE_1D_ARRAY:
        case GL_PROXY_TEXTURE_1D_ARRAY:
            height = MAX(base_height, 1u);
            depth = 1u;
            break;
        case GL_TEXTURE_3D:
        case GL_PROXY_TEXTURE_3D:
            break;
        case GL_TEXTURE_2D_ARRAY:
        case GL_PROXY_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
            depth = MAX(base_depth, 1u);
            break;
        default:
            depth = 1u;
            break;
    }

    if (out_width) *out_width = width;
    if (out_height) *out_height = height;
    if (out_depth) *out_depth = depth;
}


/* F17/P31: GL 4.6 §8.17 defines mipmap completeness over the *set* of level
 * images level_base..q that were specified -- it never constrains the order in
 * which an application supplies them.  A level above base may therefore arrive
 * first, and the later call that supplies the real base must not throw it away.
 * Returns true when every level already recorded is still consistent with the
 * candidate base (width, height, depth), i.e. when the sequence base>>level
 * still reproduces the dimensions we already stored. */
bool mglTextureRecordedLevelsMatchBase(Texture *tex,
                                              GLsizei width,
                                              GLsizei height,
                                              GLsizei depth)
{
    GLuint i;
    GLuint levels;

    if (!tex || !tex->faces[0].levels) {
        return false;
    }
    levels = tex->mipmap_levels ? tex->mipmap_levels : tex->num_levels;
    if (levels == 0u || levels > 1024u) {
        return false;
    }

    for (i = 0u; i < levels; i++) {
        TextureLevel *lvl = &tex->faces[0].levels[i];
        GLuint ew = 1u, eh = 1u, ed = 1u;

        /* A level that was never specified carries no constraint. */
        if (!lvl->complete && !lvl->ever_written && !lvl->has_initialized_data) {
            continue;
        }
        mglTextureTargetLevelDimensions(tex->target,
                                        (GLuint)width, (GLuint)height, (GLuint)depth,
                                        i, &ew, &eh, &ed);
        if (lvl->width != ew || lvl->height != eh) {
            return false;
        }
    }
    return true;
}

