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

void generateMipmaps(GLMContext ctx, GLuint texture, GLenum target)
{
    Texture *ptr;
    GLboolean needs_storage_update;
    GLuint base_width;
    GLuint base_height;
    GLuint base_depth;
    GLuint level_count;
    GLuint face_count;
    size_t pixel_size;

    ptr = getTex(ctx, texture, target);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    // Level 0 must exist before mipmap generation can derive the chain.
    ERROR_CHECK_RETURN(ptr->num_levels > 0u &&
                       ptr->faces[0].levels != NULL &&
                       ptr->faces[0].levels[0].complete,
                       GL_INVALID_OPERATION);

    if (ptr->target == GL_TEXTURE_CUBE_MAP) {
        GLuint cube_width = ptr->faces[0].levels[0].width;
        GLuint cube_height = ptr->faces[0].levels[0].height;
        if (cube_width != cube_height) {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        for (GLuint face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
            if (!ptr->faces[face].levels ||
                !ptr->faces[face].levels[0].complete ||
                ptr->faces[face].levels[0].width != cube_width ||
                ptr->faces[face].levels[0].height != cube_height) {
                mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
                return;
            }
        }
    } else if (ptr->target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        TextureLevel *base = &ptr->faces[0].levels[0];
        if (base->width != base->height || base->depth == 0u || (base->depth % 6u) != 0u) {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
    }

    mglFlushPendingDraws(ctx);

    base_width = MAX(ptr->faces[0].levels[0].width, 1u);
    base_height = MAX(ptr->faces[0].levels[0].height, 1u);
    base_depth = MAX(ptr->faces[0].levels[0].depth, 1u);
    level_count = mglTextureTargetMaxLevels(ptr->target,
                                           base_width,
                                           base_height,
                                           base_depth);
    if (ptr->immutable_storage && ptr->mipmap_levels > 0u) {
        level_count = MIN(level_count, ptr->mipmap_levels);
    }
    face_count = ptr->target == GL_TEXTURE_CUBE_MAP ? _CUBE_MAP_MAX_FACE : 1u;
    pixel_size = (ptr->faces[0].levels[0].width > 0u)
        ? (ptr->faces[0].levels[0].pitch / ptr->faces[0].levels[0].width)
        : 0u;
    if (pixel_size == 0u)
        pixel_size = sizeForFormatType(GL_RGBA, GL_UNSIGNED_BYTE);

    ERROR_CHECK_RETURN(ensureTextureLevelCapacity(ctx, ptr, level_count), GL_OUT_OF_MEMORY);

    for (GLuint face = 0; face < face_count; face++) {
        if (!ptr->faces[face].levels)
            continue;
        for (GLuint level = 1; level < level_count; level++) {
            TextureLevel *lvl = &ptr->faces[face].levels[level];
            GLuint w = 1u;
            GLuint h = 1u;
            GLuint d = 1u;
            mglTextureTargetLevelDimensions(ptr->target,
                                            base_width,
                                            base_height,
                                            base_depth,
                                            level,
                                            &w,
                                            &h,
                                            &d);

            lvl->width = w;
            lvl->height = h;
            lvl->depth = d;
            lvl->pitch = pixel_size * lvl->width;
            lvl->pixel_format = ptr->faces[face].levels[0].pixel_format;
            lvl->data_size = lvl->pitch * MAX(lvl->height, 1u) * MAX(lvl->depth, 1u);
            lvl->has_initialized_data = GL_FALSE;
            lvl->ever_written = GL_FALSE;
            lvl->suspicious_zero_upload = GL_FALSE;
            lvl->last_init_source = kTexMetalFill;
            lvl->last_upload_size = 0u;
            lvl->last_src_ptr = NULL;
            lvl->last_src_hash = 0ull;
            lvl->complete = GL_TRUE;
        }
    }

    ptr->num_levels = MAX(ptr->num_levels, level_count);
    ptr->mipmap_levels = MAX(ptr->mipmap_levels, level_count);

    needs_storage_update = (!mglRenderTextureHandle(ptr) || !ptr->mipmapped || !ptr->genmipmaps);
    if (needs_storage_update && mglRenderTextureHandle(ptr) &&
        mglTextureHasGPUContent(ptr)) {
        /* A rendered texture whose Metal storage already holds the whole chain
         * has nothing to rebuild; rebuilding would drop its GPU content. */
        MGLRenderTextureInfo mtlInfo = {0};
        if (mglRenderGetTextureInfo(mglRenderTextureHandle(ptr), &mtlInfo) == 0 &&
            mtlInfo.mipmap_level_count >= level_count) {
            needs_storage_update = GL_FALSE;
        }
    }

    ptr->mipmapped = true;
    ptr->genmipmaps = true;

    if (needs_storage_update) {
        ptr->dirty_bits |= DIRTY_TEXTURE_LEVEL;
        /* A rendered texture's content lives on the GPU; the rebuild keeps it
         * with a GPU copy, so the (stale or empty) CPU level must not be
         * uploaded over it. */
        if (!mglTextureHasGPUContent(ptr)) {
            ptr->dirty_bits |= DIRTY_TEXTURE_DATA;
        }
    }

    mglRendererGenerateMipmaps(ctx, ptr);
}

void mglGenerateMipmap(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            break;

        default:
            fprintf(stderr, "MGL Error: mglTexImage2D invalid target 0x%x\n", target);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    generateMipmaps(ctx, 0, target);
}

void mglGenerateTextureMipmap(GLMContext ctx, GLuint texture)
{
    generateMipmaps(ctx, texture, 0);
}

void mglInvalidateTextureBaseLevelView(GLMContext ctx, Texture *tex)
{
    mglRenderTextureDropBaseLevelView(ctx, tex);
}

void invalidateTexture(GLMContext ctx, Texture *tex)
{
    if (!ctx || !tex)
        return;

    GLuint old_name = tex->name;
    GLenum old_target = tex->target;
    GLuint old_index = tex->index;
    GLuint level_count = tex->mipmap_levels ? tex->mipmap_levels : tex->num_levels;

    if (level_count > 1024u) {
        fprintf(stderr,
                "MGL WARNING: invalidateTexture suspicious level count tex=%u target=0x%x levels=%u mipmapLevels=%u numLevels=%u; clamping cleanup\n",
                tex->name,
                tex->target,
                level_count,
                tex->mipmap_levels,
                tex->num_levels);
        level_count = 1024u;
    }

    mglTraceLogExternal("MGL TRACE invalidateTexture tex=%u target=0x%x index=%u levels=%u mtl=%p",
                        tex->name,
                        tex->target,
                        tex->index,
                        level_count,
                        mglRenderTextureHandle(tex));

    if (mglRenderTextureHandle(tex))
    {
        mglRendererDeleteMetalObject(ctx, mglRenderTextureHandle(tex));
        mglRenderTextureForgetHandle(tex);
    }

    if (mglRenderTextureSamplerHandle(tex))
    {
        mglRendererDeleteMetalObject(ctx, mglRenderTextureSamplerHandle(tex));
        mglRenderTextureClearSamplerHandle(tex);
    }

    /* release the cached base-level view */
    mglRenderTextureDropBaseLevelView(ctx, tex);

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        TextureLevel *levels = tex->faces[face].levels;
        if (!levels)
            continue;

        for(GLuint i=0; i<level_count; i++)
        {
            TextureLevel *lvl = &levels[i];
            if (lvl->data)
            {
                size_t dealloc_size = lvl->data_size;
                if (dealloc_size == 0u && lvl->pitch > 0u && lvl->height > 0u && lvl->depth > 0u) {
                    size_t rows_size = lvl->pitch * (size_t)lvl->height;
                    dealloc_size = page_size_align(rows_size * (size_t)lvl->depth);
                }

                if (dealloc_size > 0u) {
                    kern_return_t kr = vm_deallocate((vm_map_t)mach_task_self(),
                                                     lvl->data,
                                                     dealloc_size);
                    if (kr != KERN_SUCCESS) {
                        fprintf(stderr,
                                "MGL WARNING: invalidateTexture vm_deallocate failed tex=%u face=%d level=%u data=%p size=%zu kr=%d\n",
                                old_name,
                                face,
                                i,
                                (void *)(uintptr_t)lvl->data,
                                dealloc_size,
                                kr);
                    }
                } else {
                    fprintf(stderr,
                            "MGL WARNING: invalidateTexture skipping CPU texture storage release with unknown size tex=%u face=%d level=%u data=%p\n",
                            old_name,
                            face,
                            i,
                            (void *)(uintptr_t)lvl->data);
                }

                lvl->data = 0;
                lvl->data_size = 0;
            }
        }
    }

    for(int i=0; i<_CUBE_MAP_MAX_FACE; i++)
    {
        free(tex->faces[i].levels);
        tex->faces[i].levels = NULL;
    }
    free(tex->stencil_shadow);
    tex->stencil_shadow = NULL;
    tex->stencil_shadow_width = 0u;
    tex->stencil_shadow_height = 0u;
    free(tex->depth_shadow);
    tex->depth_shadow = NULL;
    tex->depth_shadow_width = 0u;
    tex->depth_shadow_height = 0u;
    free(tex->rgb10a2_shadow);
    tex->rgb10a2_shadow = NULL;
    tex->rgb10a2_shadow_width = 0u;
    tex->rgb10a2_shadow_height = 0u;

    tex->dirty_bits = 0;
    tex->is_render_target = GL_FALSE;
    tex->immutable_storage = GL_FALSE;
    tex->mipmapped = 0;
    tex->genmipmaps = GL_FALSE;
    tex->internalformat = 0;
    tex->image_metal_format = 0;
    tex->compressed_internalformat = 0;
    tex->width = 0;
    tex->height = 0;
    tex->depth = 0;
    tex->is_array = GL_FALSE;
    tex->complete = GL_FALSE;
    tex->num_levels = 0;
    tex->mipmap_levels = 0;
    mglTextureSetWriteVersion(tex, 0);
    tex->metal_data_authoritative = GL_FALSE;
    tex->texture_buffer = NULL;
    tex->texture_buffer_offset = 0;
    tex->texture_buffer_size = 0;
    tex->debug_label[0] = '\0';

    /* Keep object identity intact for glTexImage redefinition paths. */
    tex->name = old_name;
    tex->target = old_target;
    tex->index = old_index;
}

void initBaseTexLevel(GLMContext ctx, Texture *tex, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    tex->mipmapped = 0;
    tex->mipmap_levels = mglTextureTargetMaxLevels(tex->target,
                                                   MAX(width, 1),
                                                   MAX(height, 1),
                                                   MAX(depth, 1));

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        // CRITICAL SECURITY FIX: Prevent integer overflow in mipmap allocation
        if ((size_t)tex->mipmap_levels > SIZE_MAX / sizeof(TextureLevel)) {
            fprintf(stderr, "MGL SECURITY ERROR: Mipmap levels %d would cause allocation overflow\n", tex->mipmap_levels);
            // CRITICAL FIX: Handle gracefully instead of crashing
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }

        tex->faces[face].levels = (TextureLevel *)calloc(tex->mipmap_levels, sizeof(TextureLevel));
        if (!tex->faces[face].levels) {
            fprintf(stderr, "MGL SECURITY ERROR: calloc failed for face %d with %d levels\n", face, tex->mipmap_levels);
            // CRITICAL FIX: Handle gracefully instead of crashing
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }
    }

    tex->internalformat = internalformat;
    tex->image_metal_format = 0;
    tex->width = width;
    tex->height = height;
    tex->depth = depth;
    tex->complete = false;
    tex->metal_data_authoritative = GL_FALSE;

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        for(int i=0; i<tex->mipmap_levels; i++)
        {
            tex->faces[face].levels[i].complete = false;
            tex->faces[face].levels[i].has_initialized_data = GL_FALSE;
            tex->faces[face].levels[i].ever_written = GL_FALSE;
            tex->faces[face].levels[i].suspicious_zero_upload = GL_FALSE;
        }
    }
}

bool ensureTextureLevelCapacity(GLMContext ctx, Texture *tex, GLuint required_levels)
{
    TextureLevel *new_levels[_CUBE_MAP_MAX_FACE] = {0};
    GLuint new_capacity;

    if (!ctx || !tex || required_levels == 0)
        return false;

    if (required_levels <= tex->mipmap_levels) {
        for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
            if (!tex->faces[face].levels) {
                break;
            }
            if (face == _CUBE_MAP_MAX_FACE - 1) {
                return true;
            }
        }
    }

    new_capacity = MAX(required_levels, tex->mipmap_levels);

    if ((size_t)new_capacity > SIZE_MAX / sizeof(TextureLevel)) {
        fprintf(stderr,
                "MGL ERROR: texture level grow overflow tex=%u required=%u old=%u\n",
                tex->name,
                new_capacity,
                tex->mipmap_levels);
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return false;
    }

    for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
        new_levels[face] = (TextureLevel *)calloc(new_capacity, sizeof(TextureLevel));
        if (!new_levels[face]) {
            for (int i = 0; i < face; i++) {
                free(new_levels[i]);
            }
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return false;
        }

        if (tex->faces[face].levels && tex->mipmap_levels > 0) {
            memcpy(new_levels[face],
                   tex->faces[face].levels,
                   tex->mipmap_levels * sizeof(TextureLevel));
        }
    }

    for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
        free(tex->faces[face].levels);
        tex->faces[face].levels = new_levels[face];
    }

    fprintf(stderr,
            "MGL TRACE texture level capacity grow tex=%u old=%u new=%u base=%ux%u target=0x%x\n",
            tex->name,
            tex->mipmap_levels,
            new_capacity,
            tex->width,
            tex->height,
            tex->target);

    tex->mipmap_levels = new_capacity;
    return true;
}

/* Device max for MS sample counts (MAX_SAMPLES / framebuffer samples). */
GLuint mglMultisampleCountLimit(GLMContext ctx)
{
    GLuint max_samples = 4u;
    if (ctx) {
        GLuint m = STATE(var).max_samples;
        if (m == 0u) {
            m = STATE(var).max_framebuffer_samples;
        }
        if (m != 0u) {
            max_samples = m;
        }
    }
    return max_samples ? max_samples : 1u;
}

/* GL 4.6 §8.8 / §8.19: samples > MAX_SAMPLES → INVALID_VALUE. */
GLenum mglMultisampleCountValidationError(GLMContext ctx, GLsizei samples)
{
    if (samples < 1) {
        return GL_INVALID_VALUE;
    }
    if ((GLuint)samples > mglMultisampleCountLimit(ctx)) {
        return GL_INVALID_VALUE;
    }
    return GL_NO_ERROR;
}

/* GL 4.6 §8.8: samples is a desired minimum; the actual count may be larger.
 * Round up to the next value in GetInternalformativ(GL_SAMPLES)
 * ({1,2,4,...,max_samples}) so texelFetch(sample = request) stays in range
 * when CTS requests max_samples-1 (see gl3cTextureSizePromotion).
 * Caller must have already rejected samples > MAX_SAMPLES (AUDIT-034). */
GLuint mglPromoteMultisampleCount(GLMContext ctx, GLsizei samples)
{
    static const GLuint candidates[] = {1u, 2u, 4u, 8u, 16u, 32u};
    GLuint max_samples = mglMultisampleCountLimit(ctx);
    if (samples <= 1) {
        return 1u;
    }
    GLuint req = (GLuint)samples;
    if (req > max_samples) {
        return max_samples;
    }
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (candidates[i] >= req && candidates[i] <= max_samples) {
            return candidates[i];
        }
    }
    return max_samples;
}

bool mglSparseTexStorageAllowed(GLMContext ctx, Texture *tex,
                                      GLenum internalformat,
                                      GLsizei levels, GLsizei width,
                                      GLsizei height, GLsizei depth);
void mglSparseZeroUncommitted(Texture *tex, GLuint face, GLint level,
                                     GLint xoffset, GLint yoffset, GLint zoffset,
                                     GLsizei width, GLsizei height, GLsizei depth);
void mglSparseScrubTextureLevel(Texture *tex, GLint level, int clear);

bool mglTextureStorageMultisampleMetadata(GLMContext ctx,
                                                 Texture *tex,
                                                 GLenum target,
                                                 GLsizei samples,
                                                 GLenum internalformat,
                                                 GLsizei width,
                                                 GLsizei height,
                                                 GLsizei depth,
                                                 GLboolean fixedsamplelocations,
                                                 GLboolean proxy)
{
    GLboolean is_array = GL_FALSE;

    if (!ctx || !tex)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    switch (target)
    {
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
            break;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
            is_array = GL_TRUE;
            break;
        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }

    if (samples < 1 || width <= 0 || height <= 0 || depth <= 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }
    /* AUDIT-034: refuse samples above the device limit (do not silently clamp). */
    {
        GLenum sample_err = mglMultisampleCountValidationError(ctx, samples);
        if (sample_err != GL_NO_ERROR) {
            ERROR_RETURN_VALUE(sample_err, false);
        }
    }
    samples = (GLsizei)mglPromoteMultisampleCount(ctx, samples);

    if (!checkInternalFormatForMetal(ctx, internalformat))
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (!proxy && tex->immutable_storage)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, height, depth, 0);
        return true;
    }

    if (tex->params.sparse &&
        !mglSparseTexStorageAllowed(ctx, tex, internalformat, 1, width, height, depth)) {
        return false;
    }

    if (tex->mipmap_levels != 0u)
    {
        invalidateTexture(ctx, tex);
    }

    initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
    if (STATE(error) != GL_NO_ERROR)
    {
        return false;
    }

    if (!ensureTextureLevelCapacity(ctx, tex, 1u))
    {
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    TextureLevel *lvl = &tex->faces[0].levels[0];
    lvl->width = (GLuint)width;
    lvl->height = (GLuint)height;
    lvl->depth = (GLuint)depth;
    lvl->pitch = 0;
    lvl->pixel_format = 0;
    lvl->data_size = 0;
    lvl->data = 0;
    lvl->has_initialized_data = GL_FALSE;
    lvl->ever_written = GL_FALSE;
    lvl->suspicious_zero_upload = GL_FALSE;
    lvl->last_init_source = kTexInitNone;
    lvl->last_upload_size = 0u;
    lvl->last_src_ptr = NULL;
    lvl->last_src_hash = 0ull;
    lvl->complete = GL_TRUE;

    tex->access = GL_READ_ONLY;
    tex->internalformat = internalformat;
    tex->image_metal_format = 0;
    tex->width = (GLuint)width;
    tex->height = (GLuint)height;
    tex->depth = (GLuint)depth;
    tex->is_array = is_array;
    tex->complete = GL_TRUE;
    tex->num_levels = 1u;
    tex->mipmap_levels = 1u;
    tex->samples = (GLuint)samples;
    tex->fixed_sample_locations = fixedsamplelocations ? GL_TRUE : GL_FALSE;
    tex->immutable_storage = BUFFER_IMMUTABLE_STORAGE_FLAG;
    if (!tex->view_root) {
        tex->immutable_levels = 1u;
        tex->view_num_levels = 1u;
        tex->view_num_layers = is_array ? (GLuint)depth : 1u;
    }
    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);

    /* Commitment looks up the placement heap, so a sparse multisample
     * texture has to exist before TexPageCommitment. */
    if (tex->params.sparse)
        mglRendererBindTexture(ctx, tex);

    return true;
}

bool checkTexLevelParams(GLMContext ctx, Texture *tex, GLint level, GLuint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type)
{
    if (!tex || tex->mipmap_levels == 0)
    {
        fprintf(stderr,
                "MGL ERROR: checkTexLevelParams before base level tex=%p level=%d size=%dx%dx%d\n",
                (void *)tex,
                level,
                width,
                height,
                depth);
        return false;
    }

    /* GL 4.6 §8.5 / §8.17: TexImage* may redefine a non-base level at any
     * size / internalformat. A mismatched mip chain is incomplete, not an
     * INVALID_OPERATION. Unsized requests (e.g. GL_RGB) that resolve to the
     * existing sized format (GL_RGB8) are treated as matching. */

    if (internalformat && internalformat != tex->internalformat)
    {
        GLuint derived = internalFormatForGLFormatType(format, type);
        if (derived != tex->internalformat)
        {
            static uint64_t s_mip_iformat_mismatch_logs = 0;
            uint64_t hit = ++s_mip_iformat_mismatch_logs;
            if (hit <= 8ull || (hit % 2048ull) == 0ull)
            {
                fprintf(stderr,
                        "MGL TRACE checkTexLevelParams internalformat mismatch (incomplete, not error) tex=%u level=%d got=0x%x expected=0x%x derived=0x%x hit=%llu\n",
                        tex->name,
                        level,
                        internalformat,
                        tex->internalformat,
                        derived,
                        (unsigned long long)hit);
            }
        }
    }

    if (checkInternalFormatForMetal(ctx, tex->internalformat) == false)
    {
        fprintf(stderr,
                "MGL ERROR: checkTexLevelParams unsupported internalformat=0x%x level=%d size=%dx%dx%d\n",
                tex->internalformat,
                level,
                width,
                height,
                depth);
        return false;
    }

    return true;
}


bool verifyInternalFormatAndFormatType(GLMContext ctx, GLint internalformat, GLenum format, GLenum type)
{
    /* GL spec: INVALID_OPERATION if one of (internalformat, format) is
     * DEPTH_COMPONENT or DEPTH_STENCIL and the other is neither.
     * The forward direction (internalformat is depth, format is not) is
     * handled in the switch below.  This handles the reverse direction:
     * format is depth/stencil but internalformat is not. */
    if (format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL)
    {
        bool internal_is_depth = (internalformat == GL_DEPTH_COMPONENT ||
                                  internalformat == GL_DEPTH_COMPONENT16 ||
                                  internalformat == GL_DEPTH_COMPONENT24 ||
                                  internalformat == GL_DEPTH_COMPONENT32 ||
                                  internalformat == GL_DEPTH_COMPONENT32F ||
                                  internalformat == GL_DEPTH_STENCIL ||
                                  internalformat == GL_DEPTH24_STENCIL8 ||
                                  internalformat == GL_DEPTH32F_STENCIL8);
        if (!internal_is_depth)
        {
            return false;
        }
    }

    switch(internalformat)
    {
        // unsized formats
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_RGBA:
            break;

        // sized formats
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_R3_G3_B2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGB16_SNORM:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGBA12:
        case GL_RGBA16:
        case GL_SRGB8:
        case GL_SRGB8_ALPHA8:
        case GL_R16F:
        case GL_RG16F:
        case GL_RGB16F:
        case GL_RGBA16F:
        case GL_R32F:
        case GL_RG32F:
        case GL_RGB32F:
        case GL_RGBA32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
        // Missing SNORM/UI formats used by virgl
        case 0x9014: // GL_ALPHA8_SNORM
        case 0x9016: // GL_LUMINANCE8_ALPHA8_SNORM
        case 0x9018: // GL_ALPHA16_SNORM
        case 0x901a: // GL_LUMINANCE16_ALPHA16_SNORM
        case 0x8d7e: // GL_ALPHA8UI_EXT
            break;

        // compressed types
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
            break;

        // Legacy alpha/luminance formats (deprecated but still used)
        case 0x803c: // GL_ALPHA8
        case 0x803e: // GL_ALPHA16
        case 0x8040: // GL_LUMINANCE8
        case 0x8042: // GL_LUMINANCE16
        case 0x8045: // GL_LUMINANCE8_ALPHA8
        case 0x8048: // GL_LUMINANCE16_ALPHA16
        case 0x8816: // GL_ALPHA16F_ARB
        case 0x8818: // GL_LUMINANCE16F_ARB
        case 0x8819: // GL_LUMINANCE_ALPHA16F_ARB
        case 0x881c: // GL_ALPHA32F_ARB
        case 0x881e: // GL_LUMINANCE32F_ARB
        case 0x881f: // GL_LUMINANCE_ALPHA32F_ARB
            break;

        // ASTC compressed formats
        case 0x93b0: // GL_COMPRESSED_RGBA_ASTC_4x4_KHR
        case 0x93b1: // GL_COMPRESSED_RGBA_ASTC_5x4_KHR
        case 0x93b2: // GL_COMPRESSED_RGBA_ASTC_5x5_KHR
        case 0x93b3: // GL_COMPRESSED_RGBA_ASTC_6x5_KHR
        case 0x93b4: // GL_COMPRESSED_RGBA_ASTC_6x6_KHR
        case 0x93b5: // GL_COMPRESSED_RGBA_ASTC_8x5_KHR
        case 0x93b6: // GL_COMPRESSED_RGBA_ASTC_8x6_KHR
        case 0x93b7: // GL_COMPRESSED_RGBA_ASTC_8x8_KHR
        case 0x93b8: // GL_COMPRESSED_RGBA_ASTC_10x5_KHR
        case 0x93b9: // GL_COMPRESSED_RGBA_ASTC_10x6_KHR
        case 0x93ba: // GL_COMPRESSED_RGBA_ASTC_10x8_KHR
        case 0x93bb: // GL_COMPRESSED_RGBA_ASTC_10x10_KHR
        case 0x93bc: // GL_COMPRESSED_RGBA_ASTC_12x10_KHR
        case 0x93bd: // GL_COMPRESSED_RGBA_ASTC_12x12_KHR
        case 0x93d0: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR
        case 0x93d1: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR
        case 0x93d2: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR
        case 0x93d3: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR
        case 0x93d4: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR
        case 0x93d5: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR
        case 0x93d6: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR
        case 0x93d7: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR
        case 0x93d8: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR
        case 0x93d9: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR
        case 0x93da: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR
        case 0x93db: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR
        case 0x93dc: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR
        case 0x93dd: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR
            break;

        // ETC2/EAC compressed formats
        case 0x9270: // GL_COMPRESSED_R11_EAC
        case 0x9271: // GL_COMPRESSED_SIGNED_R11_EAC
        case 0x9272: // GL_COMPRESSED_RG11_EAC
        case 0x9273: // GL_COMPRESSED_SIGNED_RG11_EAC
        case 0x9274: // GL_COMPRESSED_RGB8_ETC2
        case 0x9275: // GL_COMPRESSED_SRGB8_ETC2
        case 0x9276: // GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2
        case 0x9277: // GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2
        case 0x9278: // GL_COMPRESSED_RGBA8_ETC2_EAC
        case 0x9279: // GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC
            break;

        // S3TC/DXT compressed formats
        case 0x83f0: // GL_COMPRESSED_RGB_S3TC_DXT1_EXT
        case 0x83f1: // GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
        case 0x83f2: // GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
        case 0x83f3: // GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
        case 0x8c4c: // GL_COMPRESSED_SRGB_S3TC_DXT1_EXT
        case 0x8c4d: // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
        case 0x8c4e: // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
        case 0x8c4f: // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
            break;

        // Additional integer formats (alternate enum values used by some implementations)
        case 0x8d72: // alternate GL_RGBA8I
        case 0x8d75: // alternate GL_RGB8I
        case 0x8d78: // alternate GL_RGBA8UI
        case 0x8d7a: // alternate GL_RGB8UI
        case 0x8d7b: // GL_ALPHA8I_EXT
        // case 0x8d7e: // alternate GL_RGBA32UI - Duplicate of GL_ALPHA8UI_EXT
        case 0x8d80: // alternate GL_RGB32UI
        case 0x8d81: // GL_ALPHA32I_EXT
        case 0x8d84: // alternate GL_RGBA16I
        case 0x8d86: // alternate GL_RGB16I
        case 0x8d87: // GL_ALPHA16I_EXT
        case 0x8d8a: // alternate GL_RGBA32I
        case 0x8d8c: // alternate GL_RGB32I
        case 0x8d8d: // GL_ALPHA32I_EXT
        case 0x8d90: // alternate GL_RGBA16UI
        case 0x8d92: // alternate GL_RGB16UI
        case 0x8d93: // GL_ALPHA16UI_EXT
            break;

        // SNORM formats
        case 0x8f9b: // GL_SIGNED_NORMALIZED
        case 0x8fbd: // GL_RGB10_A2UI (alternate)
        case 0x8fbe: // GL_RGBA16_SNORM
            break;

        // Depth/stencil special formats
        // case 0x9014: // GL_DEPTH_COMPONENT16_NONLINEAR_NV - Duplicate of GL_ALPHA8_SNORM
        // case 0x9016: // GL_TEXTURE_2D_MULTISAMPLE - Duplicate of GL_LUMINANCE8_ALPHA8_SNORM
        // case 0x9018: // GL_TEXTURE_2D_MULTISAMPLE_ARRAY - Duplicate of GL_ALPHA16_SNORM
        // case 0x901a: // GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY - Duplicate of GL_LUMINANCE16_ALPHA16_SNORM
            break;

        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            /* CTS allows any depth/stencil format for depth internal formats. */
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL, GL_INVALID_OPERATION, false);
            break;

        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            /* CTS allows GL_DEPTH_COMPONENT or GL_DEPTH_STENCIL for depth/stencil formats. */
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL, GL_INVALID_OPERATION, false);
            break;
            
        case GL_STENCIL_INDEX8:
            ERROR_CHECK_RETURN_VALUE(format == GL_STENCIL_INDEX, GL_INVALID_OPERATION, false);
            break;
            
        case GL_RGB565:
            break;

        default:
            // Log warning but don't error - many formats work even if not explicitly listed
            fprintf(stderr, "MGL WARNING: verifyInternalFormat unknown internalformat 0x%x\n", internalformat);
            break;
    }

    /* Integer/non-integer compatibility check (OpenGL 4.6 Table 8.3):
     * Integer pixel formats require integer internal formats and vice versa. */
    bool format_is_integer = (format == GL_RED_INTEGER   ||
                              format == GL_RG_INTEGER    ||
                              format == GL_RGB_INTEGER   ||
                              format == GL_BGR_INTEGER   ||
                              format == GL_RGBA_INTEGER  ||
                              format == GL_BGRA_INTEGER  ||
                              format == 0x8d96 /*GL_BLUE_INTEGER*/  ||
                              format == 0x8d95 /*GL_GREEN_INTEGER*/ ||
                              format == 0x8d97 /*GL_ALPHA_INTEGER*/);

    bool internal_is_integer = (internalformat == GL_R8I       ||
                                internalformat == GL_R8UI      ||
                                internalformat == GL_R16I      ||
                                internalformat == GL_R16UI     ||
                                internalformat == GL_R32I      ||
                                internalformat == GL_R32UI     ||
                                internalformat == GL_RG8I      ||
                                internalformat == GL_RG8UI     ||
                                internalformat == GL_RG16I     ||
                                internalformat == GL_RG16UI    ||
                                internalformat == GL_RG32I     ||
                                internalformat == GL_RG32UI    ||
                                internalformat == GL_RGB8I     ||
                                internalformat == GL_RGB8UI    ||
                                internalformat == GL_RGB16I    ||
                                internalformat == GL_RGB16UI   ||
                                internalformat == GL_RGB32I    ||
                                internalformat == GL_RGB32UI   ||
                                internalformat == GL_RGBA8I    ||
                                internalformat == GL_RGBA8UI   ||
                                internalformat == GL_RGBA16I   ||
                                internalformat == GL_RGBA16UI  ||
                                internalformat == GL_RGBA32I   ||
                                internalformat == GL_RGBA32UI  ||
                                internalformat == GL_RGB10_A2UI);

    if (format_is_integer != internal_is_integer)
    {
        /* Allow depth/stencil formats to pass through - they are neither
         * integer nor non-integer in the traditional sense. */
        if (format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL ||
            format == GL_STENCIL_INDEX)
            /* ok */;
        else
            return false;
    }

    switch(format)
    {
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_BGR:
        case GL_RGBA:
        case GL_BGRA:
        case GL_RED_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        // Legacy formats (deprecated but still used by virglrenderer)
        case 0x1906: // GL_ALPHA
        case 0x1909: // GL_LUMINANCE
        case 0x190a: // GL_LUMINANCE_ALPHA
        case 0x8000: // GL_COLOR_INDEX (legacy)
        case 0x8d97: // GL_ALPHA_INTEGER
        case 0x8d96: // GL_BLUE_INTEGER
        case 0x8d95: // GL_GREEN_INTEGER
        case 0x8d9c: // GL_LUMINANCE_INTEGER_EXT
        case 0x8d9d: // GL_LUMINANCE_ALPHA_INTEGER_EXT
            break;

        default:
            // Allow unknown formats with warning - virglrenderer may use nonstandard values
            fprintf(stderr, "MGL WARNING: verifyFormat unknown format 0x%x, allowing\n", format);
            break;
    }

    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
            break;

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
            ERROR_CHECK_RETURN_VALUE(format == GL_RGB || format == GL_RGB_INTEGER,GL_INVALID_OPERATION, false);
            break;

        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            ERROR_CHECK_RETURN_VALUE((format == GL_RGBA || format == GL_BGRA ||
                                      format == GL_RGBA_INTEGER || format == GL_BGRA_INTEGER), GL_INVALID_OPERATION, false);
            break;
            
        case GL_UNSIGNED_INT_24_8:
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_STENCIL, GL_INVALID_OPERATION, false);
            break;
            
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_STENCIL, GL_INVALID_OPERATION, false);
            break;
        
        // Packed float types: each packs 3 channels, valid only with GL_RGB.
        case 0x8c3b: // GL_UNSIGNED_INT_10F_11F_11F_REV
        case 0x8c3e: // GL_UNSIGNED_INT_5_9_9_9_REV
            ERROR_CHECK_RETURN_VALUE(format == GL_RGB, GL_INVALID_OPERATION, false);
            break;
            
        default:
            fprintf(stderr, "MGL WARNING: verifyInternalFormat unknown type 0x%x\n", type);
            break;
    }

    /* Rule: GL_DEPTH_STENCIL format requires GL_UNSIGNED_INT_24_8 or
     * GL_FLOAT_32_UNSIGNED_INT_24_8_REV type. (CTS isFormatValid.) */
    if (format == GL_DEPTH_STENCIL &&
        type != GL_UNSIGNED_INT_24_8 &&
        type != GL_FLOAT_32_UNSIGNED_INT_24_8_REV) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    /* Rule: integer transfer format cannot be paired with float/half-float type. */
    if (mglExternalFormatIsInteger(format) &&
        (type == GL_FLOAT || type == GL_HALF_FLOAT)) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    /* Rule: depth/stencil transfer format must match a depth/stencil
     * internalformat, and vice versa. (CTS glcPackedPixelsTests isFormatValid.) */
    {
        bool fmt_ds = (format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL ||
                       format == GL_STENCIL_INDEX);
        bool ifmt_ds = mglInternalFormatIsDepthStencil(internalformat);
        if (fmt_ds != ifmt_ds) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
    }

    /* Rule: integer transfer format requires an integer internalformat, and
     * a non-integer transfer format requires a non-integer internalformat.
     * (CTS glcPackedPixelsTests isFormatValid: GL core path.) This is what
     * makes e.g. compressed_RED + BGR_INTEGER + UNSIGNED_BYTE invalid. */
    {
        bool fmt_int = mglExternalFormatIsInteger(format);
        bool ifmt_int = mglInternalFormatIsInteger(internalformat);
        if (fmt_int != ifmt_int) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
    }

    return true;
}



void unpackTexture(GLMContext ctx, Texture *tex, GLuint face, GLuint level, void *src_data, void *dst_data, size_t src_pitch, size_t src_image_pitch, size_t pixel_size, size_t xoffset, size_t yoffset, size_t zoffset, size_t width, size_t height, size_t depth)
{
    GLubyte *src, *dst;
    size_t dst_pitch;

    if (!tex || face >= _CUBE_MAP_MAX_FACE || !tex->faces[face].levels || level >= tex->mipmap_levels)
    {
        fprintf(stderr,
                "MGL WARNING: unpackTexture skipped invalid texture level tex=%p face=%u level=%u levels=%u\n",
                (void *)tex,
                face,
                level,
                tex ? tex->mipmap_levels : 0u);
        return;
    }

    if (!src_data || !dst_data || pixel_size == 0u || width == 0u || height == 0u || depth == 0u)
    {
        fprintf(stderr,
                "MGL WARNING: unpackTexture skipped invalid copy args tex=%u src=%p dst=%p pixelSize=%zu size=%zux%zux%zu\n",
                tex->name,
                src_data,
                dst_data,
                pixel_size,
                width,
                height,
                depth);
        return;
    }

    src = (GLubyte *)src_data;
    dst = (GLubyte *)dst_data;

    dst_pitch = tex->faces[face].levels[level].pitch;
    if (dst_pitch == 0u)
    {
        fprintf(stderr,
                "MGL WARNING: unpackTexture skipped level with zero destination pitch tex=%u face=%u level=%u\n",
                tex->name,
                face,
                level);
        return;
    }

    if (xoffset || yoffset || zoffset)
    {
        size_t xoffset_bytes = xoffset * pixel_size; // num pixels
        size_t yoffset_bytes = yoffset * dst_pitch; // num lines (rows * bytes_per_row)
        size_t level_height = tex->faces[face].levels[level].height;
        size_t zoffset_bytes = zoffset * dst_pitch * level_height; // num planes

        dst += xoffset_bytes;
        dst += yoffset_bytes;
        dst += zoffset_bytes;
    }

    if (depth > 1)
    {
        // 3d texture
        size_t copy_size = width * pixel_size;
        size_t dst_image_pitch = dst_pitch * tex->faces[face].levels[level].height;
        if (src_image_pitch == 0u) {
            src_image_pitch = src_pitch * height;
        }

        for(size_t z=0; z<depth; z++)
        {
            GLubyte *src_slice = src + z * src_image_pitch;
            GLubyte *dst_slice = dst + z * dst_image_pitch;
            for(size_t y=0; y<height; y++)
            {
                memcpy(dst_slice, src_slice, copy_size);
                src_slice += src_pitch;
                dst_slice += dst_pitch;
            }
        }
    }
    else if (height > 1)
    {
        // 2d texture
        size_t copy_size = width * pixel_size;
        
        for(int y=0; y<height; y++)
        {
            memcpy(dst, src, copy_size);
            src += src_pitch;
            dst += dst_pitch;
        }
    }
    else
    {
        // 1d texture
        memcpy(dst, src, width * pixel_size);
    }
}

#pragma mark texImage 1D/2D/3D
// Forward declaration
bool texSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels);

bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy)
{
    if (!tex)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (level < 0 || width < 0 || height < 0 || depth < 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (!proxy) {
        mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);
    }

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    // all the levels are created on a tex storage call.. if we get here we should just assert
    /* GL 4.6 §8.19: no TexImage* / CopyTexImage* on immutable storage. */
    if (tex->immutable_storage)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    /*
     * Minecraft/LWJGL can submit a zero-sized tail mip (for example 16x16
     * level 5 -> 0x0) while walking a mip chain. GL accepts non-negative
     * dimensions, but there is no Metal/CPU storage to create for a zero-sized
     * image. Treat this as a successful no-op instead of turning it into a
     * repeated INVALID_OPERATION from checkTexLevelParams().
     */
    if (level > 0 && (width == 0 || height == 0 || depth == 0))
    {
        static uint64_t s_zero_tail_mip_logs = 0;
        uint64_t hit = ++s_zero_tail_mip_logs;

        if (hit <= 8ull || (hit % 2048ull) == 0ull)
        {
            fprintf(stderr,
                    "MGL TRACE createTextureLevel skip zero-sized tail mip tex=%u target=0x%x face=%u level=%d size=%dx%dx%d base=%ux%ux%u numLevels=%u mipmapLevels=%u hit=%llu\n",
                    tex ? tex->name : 0u,
                    tex ? tex->target : 0u,
                    face,
                    level,
                    width,
                    height,
                    depth,
                    tex ? tex->width : 0u,
                    tex ? tex->height : 0u,
                    tex ? tex->depth : 0u,
                    tex ? tex->num_levels : 0u,
                    tex ? tex->mipmap_levels : 0u,
                    (unsigned long long)hit);
        }

        return true;
    }

    /* Compressed TexImage* policy (no Metal online encoder):
     * - Non-NULL uncompressed pixels + sized compressed IF → remap to a
     *   sized uncompressed IF and store the pixels (packed_pixels RGTC
     *   GetTexImage round-trip).
     * - NULL pixels → empty compressed storage (error-path / persistent
     *   map CTS that only needs a compressed object).
     * - Texture views (view_root): never allocate storage — fall through so
     *   mglRendererBindTexture builds a Metal view of the parent (§8.18).
     * - Generic COMPRESSED_* → same remap path. */
    if (mglTexLevelInternalFormatCompressed(internalformat) &&
        !mglIsGenericCompressedFormat((GLenum)internalformat) &&
        pixels == NULL)
    {
        if (!proxy && !tex->view_root)
        {
            GLuint bw = 0u, bh = 0u, bd = 1u, bs = 0u;
            GLsizei store_depth = depth > 0 ? depth : 1;
            GLsizei image_size;
            GLenum store_target = tex->target;

            if (!mglCompressedBlockInfoOf((GLenum)internalformat, &bw, &bh, &bd,
                                          &bs) ||
                bw == 0u || bh == 0u || bd == 0u || bs == 0u)
            {
                ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
            }
            image_size = (GLsizei)((((GLuint)width + bw - 1u) / bw) *
                                   (((GLuint)height + bh - 1u) / bh) *
                                   (((GLuint)store_depth + bd - 1u) / bd) * bs);
            if (store_target == GL_TEXTURE_CUBE_MAP)
            {
                store_target =
                    (GLenum)(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (GLint)face);
            }
            return mglStoreCompressedTextureImage(
                ctx, store_target, level, (GLenum)internalformat, width, height,
                store_depth, 0, image_size, NULL);
        }
        /* Proxy / view: keep compressed IF; fall through for metadata-only. */
        tex->compressed_internalformat = internalformat;
    }
    else if (mglTexLevelInternalFormatCompressed(internalformat))
    {
        /* Sized block formats + uncompressed transfer pixels:
         * - RGTC + formats/types mglEncodeRGTCRectToCPU supports: real online
         *   compression (GL 4.6 §8.5 Table 8.14). Keeps TEXTURE_COMPRESSED
         *   true so DSA textures_compressed_subimage's probe can download
         *   via GetCompressedTexImage.
         * - Other Copyable layouts (GREEN/BLUE, integer, EAC, …): remap to a
         *   sized uncompressed IF for GetTexImage round-trip (packed_pixels).
         *   No size-based accept/reject. */
        if (!mglIsGenericCompressedFormat((GLenum)internalformat)) {
            GLboolean rgtc_encode = GL_FALSE;
            switch ((GLenum)internalformat) {
                case GL_COMPRESSED_RED_RGTC1:
                case GL_COMPRESSED_SIGNED_RED_RGTC1:
                case GL_COMPRESSED_RG_RGTC2:
                case GL_COMPRESSED_SIGNED_RG_RGTC2:
                    rgtc_encode =
                        (type == GL_UNSIGNED_BYTE &&
                         (format == GL_RED || format == GL_RG ||
                          format == GL_RGB || format == GL_RGBA ||
                          format == GL_BGR || format == GL_BGRA));
                    break;
                default:
                    break;
            }
            if (rgtc_encode && !proxy) {
                GLuint bw = 0u, bh = 0u, bd = 1u, bs = 0u;
                GLsizei store_depth = depth > 0 ? depth : 1;
                GLsizei image_size;
                GLenum store_target = tex->target;
                if (!mglCompressedBlockInfoOf((GLenum)internalformat, &bw, &bh,
                                              &bd, &bs) ||
                    bw == 0u || bh == 0u || bd == 0u || bs == 0u) {
                    ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
                }
                image_size = (GLsizei)((((GLuint)width + bw - 1u) / bw) *
                                       (((GLuint)height + bh - 1u) / bh) *
                                       (((GLuint)store_depth + bd - 1u) / bd) *
                                       bs);
                if (store_target == GL_TEXTURE_CUBE_MAP) {
                    store_target =
                        (GLenum)(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (GLint)face);
                }
                if (!mglStoreCompressedTextureImage(
                        ctx, store_target, level, (GLenum)internalformat, width,
                        height, store_depth, 0, image_size, NULL)) {
                    return false;
                }
                return texSubImage(ctx, tex, face, level, 0, 0, 0, width, height,
                                   store_depth, format, type, pixels);
            }

            GLboolean format_ok = GL_FALSE;
            switch ((GLenum)internalformat) {
                case GL_COMPRESSED_RED_RGTC1:
                case GL_COMPRESSED_SIGNED_RED_RGTC1:
                case GL_COMPRESSED_RG_RGTC2:
                case GL_COMPRESSED_SIGNED_RG_RGTC2:
                    format_ok =
                        (format == GL_RED || format == GL_RED_INTEGER ||
                         format == GL_RG || format == GL_RG_INTEGER ||
                         format == GL_RGB || format == GL_RGB_INTEGER ||
                         format == GL_BGR || format == GL_BGR_INTEGER ||
                         format == GL_GREEN || format == GL_GREEN_INTEGER ||
                         format == GL_BLUE || format == GL_BLUE_INTEGER ||
                         format == GL_RGBA || format == GL_RGBA_INTEGER ||
                         format == GL_BGRA || format == GL_BGRA_INTEGER);
                    break;
                default:
                    /* EAC and other sized block formats: no online compression
                     * and no remap — INVALID_OPERATION (DSA falls back to
                     * CompressedTexImage static blobs). */
                    break;
            }
            if (!format_ok) {
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
        }
        tex->compressed_internalformat = internalformat;
        internalformat = mglCompressedInternalFormatToSizedUncompressed(internalformat);
    }

    if (level == 0)
    {
        if (internalformat == 0)
        {
            internalformat = internalFormatForGLFormatType(format, type);

            if (internalformat == 0)
            {
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
        }
        else if (pixels && internalformat == format)
        {
            GLuint temp_format;

            // check if format type can be copied directly to the internal format
            temp_format = internalFormatForGLFormatType(format, type);

            // MGL doesn't support pixel format conversion
            // If mismatch, use the format that matches the incoming data.
            // If temp_format is 0 (unknown mapping), keep the original
            // internalformat so unsized formats like GL_DEPTH_COMPONENT fall
            // back to mtlFormatForGLInternalFormat() which knows how to handle them.
            if (temp_format != 0 && temp_format != internalformat)
            {
                internalformat = temp_format;
            }
        }

        // see if we can actually use this internal format
        if (checkInternalFormatForMetal(ctx, internalformat) == false)
        {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        if (tex->mipmap_levels == 0)
        {
            // uninitialized tex
            initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
        }
        else if (width != tex->width || height != tex->height ||
                 depth != (GLsizei)tex->depth ||
                 internalformat != tex->internalformat)
        {
            /* F17/P31: if the dimensions we recorded earlier came from levels
             * above base, this "base changed" test fires even though every
             * stored level is still exactly base>>level -- the texture is
             * legal and the earlier levels must survive.  Only a genuine
             * inconsistency justifies discarding the whole texture. */
            if (internalformat == tex->internalformat &&
                mglTextureRecordedLevelsMatchBase(tex, width, height, depth))
            {
                tex->width = (GLuint)width;
                tex->height = (GLuint)height;
                tex->depth = (GLuint)depth;
                tex->complete = false;
                tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
                mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);

                /* The chain is now rooted at the real base, so the capacity
                 * must cover the full base-derived chain. */
                if (ensureTextureLevelCapacity(ctx, tex,
                        mglTextureTargetMaxLevels(tex->target,
                                                  (GLuint)MAX(width, 1),
                                                  (GLuint)MAX(height, 1),
                                                  (GLuint)MAX(depth, 1))) == false)
                {
                    ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
                }
            }
            else
            {
                // invalidate texture because the base level width / height / depth / internal format are being changed...
                invalidateTexture(ctx, tex);

                initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
            }
        }
    }
    else if (tex->mipmap_levels == 0)
    {
        if (internalformat == 0)
        {
            internalformat = internalFormatForGLFormatType(format, type);
            if (internalformat == 0)
            {
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
        }
        else if (pixels && internalformat == format)
        {
            GLuint temp_format = internalFormatForGLFormatType(format, type);
            if (temp_format != internalformat)
            {
                internalformat = temp_format;
            }
        }

        if (checkInternalFormatForMetal(ctx, internalformat) == false)
        {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        tex->internalformat = internalformat;
        tex->image_metal_format = 0;
        tex->width = width;
        tex->height = height;
        tex->depth = depth;
        tex->complete = false;
    }
    else
    {
        /* Same unsized→sized resolution as level 0 so GL_RGB after GL_RGB8
         * does not keep a divergent token for storage sizing. */
        if (internalformat == 0)
        {
            internalformat = internalFormatForGLFormatType(format, type);
            if (internalformat == 0)
            {
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
        }
        else if (internalformat == format)
        {
            GLuint temp_format = internalFormatForGLFormatType(format, type);
            if (temp_format != 0)
            {
                internalformat = temp_format;
            }
        }
        if (internalformat != tex->internalformat)
        {
            GLuint derived = internalFormatForGLFormatType(format, type);
            if (derived == tex->internalformat)
            {
                internalformat = tex->internalformat;
            }
        }

        if (checkTexLevelParams(ctx, tex, level, internalformat, width, height, depth, format, type) == false)
        {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
    }

    if (!ensureTextureLevelCapacity(ctx, tex, (GLuint)level + 1u))
    {
        fprintf(stderr,
                "MGL ERROR: createTextureLevel failed to grow levels tex=%u face=%u level=%d currentLevels=%u size=%dx%dx%d\n",
                tex ? tex->name : 0u,
                face,
                level,
                tex ? tex->mipmap_levels : 0u,
                width,
                height,
                depth);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    tex->num_levels = MAX(tex->num_levels, level + 1);
    tex->faces[face].levels[level].width = width;
    tex->faces[face].levels[level].height = height;
    tex->faces[face].levels[level].depth = depth;

    // Proxy textures are capability probes: validate and store metadata only.
    // Do not allocate backing storage or upload data.
    if (proxy)
    {
        tex->is_array = is_array;
        tex->faces[face].levels[level].pitch = 0;
        tex->faces[face].levels[level].data_size = 0;
        tex->faces[face].levels[level].data = 0;
        tex->faces[face].levels[level].has_initialized_data = GL_FALSE;
        tex->faces[face].levels[level].ever_written = GL_FALSE;
        tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
        tex->faces[face].levels[level].last_init_source = kTexInitNone;
        tex->faces[face].levels[level].last_upload_size = 0u;
        tex->faces[face].levels[level].last_src_ptr = NULL;
        tex->faces[face].levels[level].last_src_hash = 0ull;
        tex->faces[face].levels[level].complete = true;
        tex->complete = true;
        return true;
    }

    if (width == 0 || height == 0 || depth == 0)
    {
        fprintf(stderr,
                "MGL TRACE createTextureLevel zero-sized image tex=%u target=0x%x face=%u level=%d size=%dx%dx%d\n",
                tex->name,
                tex->target,
                face,
                level,
                width,
                height,
                depth);
        tex->faces[face].levels[level].pitch = 0;
        tex->faces[face].levels[level].data_size = 0;
        tex->faces[face].levels[level].data = 0;
        tex->faces[face].levels[level].has_initialized_data = GL_FALSE;
        tex->faces[face].levels[level].ever_written = GL_FALSE;
        tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
        tex->faces[face].levels[level].last_init_source = kTexInitNone;
        tex->faces[face].levels[level].last_upload_size = 0u;
        tex->faces[face].levels[level].last_src_ptr = NULL;
        tex->faces[face].levels[level].last_src_hash = 0ull;
        tex->faces[face].levels[level].complete = false;
        tex->complete = false;
        tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);
        return true;
    }

    /* Compressed internalformat was already remapped to uncompressed above.
     * If it's still compressed (e.g. a format not in our remap table), treat
     * as a no-op storage to avoid crashing on unknown compressed formats. */
    if (mglTexLevelInternalFormatCompressed(internalformat))
    {
        tex->faces[face].levels[level].pitch = 0;
        tex->faces[face].levels[level].data_size = 0;
        tex->faces[face].levels[level].data = 0;
        tex->faces[face].levels[level].has_initialized_data = pixels ? GL_TRUE : GL_FALSE;
        tex->faces[face].levels[level].ever_written = pixels ? GL_TRUE : GL_FALSE;
        tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
        tex->faces[face].levels[level].last_init_source = pixels ? kTexImageCopy : kTexImageNull;
        tex->faces[face].levels[level].last_upload_size = 0u;
        tex->faces[face].levels[level].last_src_ptr = pixels;
        tex->faces[face].levels[level].last_src_hash = 0ull;
        tex->faces[face].levels[level].complete = true;
        tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);
        return true;
    }

    kern_return_t err;
    vm_address_t texture_data;
    size_t pixel_size;
    size_t internal_size;
    size_t texture_size;
    size_t src_pitch;

    pixel_size = sizeForInternalFormat(internalformat, format, type);
    ERROR_CHECK_RETURN_VALUE(pixel_size, GL_INVALID_ENUM, false);

    tex->faces[face].levels[level].pitch = pixel_size * width;

    if (depth > 1)
    {
        // 3d texture
        internal_size = pixel_size * width * height * depth;
    }
    else if (height > 1)
    {
        // 2d texture
        internal_size = pixel_size * width * height;
    }
    else
    {
        // 1d texture
        internal_size = pixel_size * width;
    }

    texture_size = page_size_align(internal_size);
    if (texture_size == 0u)
    {
        fprintf(stderr,
                "MGL ERROR: createTextureLevel computed zero allocation tex=%u level=%d internal=0x%x size=%dx%dx%d pixelSize=%zu\n",
                tex->name,
                level,
                internalformat,
                width,
                height,
                depth,
                pixel_size);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    bool has_upload_source = (pixels != NULL) || (STATE(buffers[_PIXEL_UNPACK_BUFFER]) != NULL);

    // Keep a CPU shadow even for private Metal formats such as depth textures.
    // GL readback and glClearTexImage operate on texture contents regardless of
    // whether the Metal resource itself can be directly CPU-visible.
    err = vm_allocate((vm_map_t) mach_task_self(),
                      (vm_address_t*) &texture_data,
                      texture_size,
                      VM_FLAGS_ANYWHERE);
    if (err != 0 || !texture_data)
    {
        fprintf(stderr,
                "MGL ERROR: createTextureLevel vm_allocate failed err=%d bytes=%zu tex=%u level=%d\n",
                err,
                texture_size,
                tex->name,
                level);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    tex->faces[face].levels[level].data_size = texture_size;
    tex->faces[face].levels[level].data = (vm_address_t)texture_data;

    if (has_upload_source)
    {
        size_t source_pixel_size = sizeForFormatType(format, type);
        MGLTextureUnpackLayout unpack_layout;
        const uint8_t *resolved_src = NULL;
        Buffer *resolved_unpack_buf = NULL;

        ERROR_CHECK_RETURN_VALUE(source_pixel_size > 0u, GL_INVALID_ENUM, false);

        if (!mglComputeTextureUnpackLayout(ctx,
                                           width,
                                           height,
                                           depth,
                                           source_pixel_size,
                                           "createTextureLevel",
                                           &unpack_layout)) {
            return false;
        }

        if (!mglResolveTexSubImageSource(ctx,
                                         tex,
                                         face,
                                         level,
                                         0,
                                         0,
                                         0,
                                         width,
                                         height,
                                         depth,
                                         format,
                                         type,
                                         pixels,
                                         unpack_layout.skip_offset_bytes,
                                         unpack_layout.required_bytes,
                                         false,
                                         &resolved_src,
                                         &resolved_unpack_buf)) {
            return false;
        }

        if (!resolved_src) {
            fprintf(stderr,
                    "MGL createTextureLevel skip upload: resolved source is NULL tex=%u target=0x%x %dx%dx%d\n",
                    tex->name,
                    tex->target,
                    width,
                    height,
                    depth);
            has_upload_source = false;
        } else {
            src_pitch = unpack_layout.src_pitch;
            TextureLevel *lvl = &tex->faces[face].levels[level];
            if (!mglConvertTextureRectToCPU((GLenum)internalformat,
                                            lvl,
                                            0,
                                            0,
                                            0,
                                            width,
                                            height,
                                            depth,
                                            format,
                                            type,
                                            resolved_src,
                                            src_pitch,
                                            unpack_layout.src_image_size,
                                            STATE(unpack).swap_bytes == GL_TRUE)) {
                unpackTexture(ctx,
                              tex,
                              face,
                              level,
                              (void *)resolved_src,
                              (void *)texture_data,
                              src_pitch,
                              unpack_layout.src_image_size,
                              source_pixel_size,
                              0,
                              0,
                              0,
                              width,
                              height,
                              depth);
            }

            tex->faces[face].levels[level].last_upload_size = unpack_layout.required_bytes;
            tex->faces[face].levels[level].last_src_hash = mglHashBytesSampled(resolved_src, unpack_layout.required_bytes);
        }

        if (resolved_src) {
            tex->faces[face].levels[level].last_init_source = resolved_unpack_buf ? kTexImagePBO : kTexImageCopy;
            tex->faces[face].levels[level].last_src_ptr = resolved_src;
            tex->faces[face].levels[level].ever_written = GL_TRUE;
            tex->faces[face].levels[level].has_initialized_data = GL_TRUE;
            tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
        }

        if (unpack_layout.required_bytes > 0u && resolved_src) {
            if (mglLooksAllZeroSampled(resolved_src, unpack_layout.required_bytes)) {
                size_t first_nonzero = 0u;
                uint8_t first_value = 0u;
                bool has_nonzero = mglFindFirstNonZeroByte(resolved_src,
                                                           unpack_layout.required_bytes,
                                                           &first_nonzero,
                                                           &first_value);
                if (!has_nonzero) {
                    tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
                    tex->faces[face].levels[level].has_initialized_data = GL_TRUE;
                }
                fprintf(stderr,
                        "MGL WARNING: createTextureLevel upload sampled head/mid/tail all-zero tex=%u face=%u level=%d bytes=%zu src=%p fullZero=%d firstNonZero=0x%zx value=0x%02x\n",
                        tex->name,
                        face,
                        level,
                        unpack_layout.required_bytes,
                        resolved_src,
                        has_nonzero ? 0 : 1,
                        has_nonzero ? first_nonzero : 0u,
                        has_nonzero ? first_value : 0u);
            }
        }

        if (resolved_src) {
            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        }
    };

    tex->faces[face].levels[level].complete = true;
    if (!has_upload_source) {
        tex->faces[face].levels[level].has_initialized_data = GL_FALSE;
        tex->faces[face].levels[level].ever_written = GL_FALSE;
        tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
        tex->faces[face].levels[level].last_init_source = (format != 0 && type != 0) ? kTexImageNull : kTexInitNone;
        tex->faces[face].levels[level].last_upload_size = 0u;
        tex->faces[face].levels[level].last_src_ptr = NULL;
        tex->faces[face].levels[level].last_src_hash = 0ull;
    }

    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);

    /* Populate depth_shadow for depth internal formats so that
     * glReadPixels(GL_DEPTH_COMPONENT) can read back uploaded data. */
    if (has_upload_source && !proxy && face == 0u && level == 0 &&
        mglInternalFormatIsDepthStencil(internalformat) &&
        internalformat != GL_STENCIL_INDEX8 &&
        internalformat != GL_STENCIL_INDEX &&
        internalformat != GL_DEPTH_STENCIL) {
        if (mglInternalFormatCarriesDepth(internalformat)) {
            if (!tex->depth_shadow ||
                tex->depth_shadow_width != tex->width ||
                tex->depth_shadow_height != tex->height) {
                free(tex->depth_shadow);
                tex->depth_shadow = calloc((size_t)tex->width * tex->height, sizeof(GLfloat));
                tex->depth_shadow_width = tex->depth_shadow ? tex->width : 0u;
                tex->depth_shadow_height = tex->depth_shadow ? tex->height : 0u;
            }
            if (tex->depth_shadow) {
                uint8_t *src = (uint8_t *)tex->faces[face].levels[level].data;
                size_t src_pitch = tex->faces[face].levels[level].pitch;
                /* Use the GL internal format (not the Metal format) to determine
                 * the CPU shadow pixel layout, because multiple GL formats map
                 * to the same Metal format but have different CPU storage. */
                GLint canonical = mglTexLevelCanonicalInternalFormat(internalformat);
                size_t pixel_size = sizeForInternalFormat(canonical, 0, 0);
                if (src && src_pitch > 0u && pixel_size > 0u) {
                    for (GLuint row = 0u; row < tex->height; row++) {
                        uint8_t *p = src + (size_t)row * src_pitch;
                        for (GLuint col = 0u; col < tex->width; col++) {
                            uint8_t *pixel = p + (size_t)col * pixel_size;
                            GLfloat d = 0.0f;
                            switch (canonical) {
                                case GL_DEPTH_COMPONENT16:
                                    d = (GLfloat)((uint16_t *)pixel)[0] / 65535.0f;
                                    break;
                                case GL_DEPTH_COMPONENT24:
                                case GL_DEPTH_COMPONENT32:
                                case GL_DEPTH_COMPONENT32F:
                                    d = ((GLfloat *)pixel)[0];
                                    break;
                                case GL_DEPTH24_STENCIL8: {
                                    /* 4 bytes: depth high 24 [31:8], stencil low 8 [7:0]. */
                                    uint32_t v;
                                    memcpy(&v, pixel, sizeof(uint32_t));
                                    d = (GLfloat)(v >> 8) / 16777215.0f;
                                    break;
                                }
                                case GL_DEPTH32F_STENCIL8:
                                    /* 5 bytes: float depth + uint8 stencil. */
                                    d = ((GLfloat *)pixel)[0];
                                    break;
                                default:
                                    d = 0.0f;
                                    break;
                            }
                            tex->depth_shadow[(size_t)row * tex->width + col] = d;
                        }
                    }
                }
                if (canonical == GL_DEPTH24_STENCIL8 ||
                    canonical == GL_DEPTH32F_STENCIL8) {
                    if (!tex->stencil_shadow ||
                        tex->stencil_shadow_width != tex->width ||
                        tex->stencil_shadow_height != tex->height) {
                        free(tex->stencil_shadow);
                        tex->stencil_shadow = calloc((size_t)tex->width * tex->height, 1u);
                        tex->stencil_shadow_width = tex->stencil_shadow ? tex->width : 0u;
                        tex->stencil_shadow_height = tex->stencil_shadow ? tex->height : 0u;
                    }
                    if (tex->stencil_shadow && src && src_pitch > 0u && pixel_size > 0u) {
                        for (GLuint row = 0u; row < tex->height; row++) {
                            uint8_t *p = src + (size_t)row * src_pitch;
                            for (GLuint col = 0u; col < tex->width; col++) {
                                uint8_t *pixel = p + (size_t)col * pixel_size;
                                uint8_t s = 0u;
                                if (canonical == GL_DEPTH24_STENCIL8) {
                                    uint32_t v;
                                    memcpy(&v, pixel, sizeof(uint32_t));
                                    s = (uint8_t)(v & 0xffu);
                                } else if (pixel_size >= 5u) {
                                    s = pixel[4];
                                }
                                tex->stencil_shadow[(size_t)row * tex->width + col] = s;
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

