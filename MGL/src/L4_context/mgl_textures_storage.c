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

void mglTexStorage1D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
{
    Texture *tex;
    GLboolean proxy;

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

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(mglTexStorageInternalFormatValid(internalformat), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(checkMaxLevels(levels, width, 1, 1), GL_INVALID_OPERATION);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, 1, 1, 0);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);

    texStorage(ctx, tex, 1, levels, false, internalformat, width, 1, 1, proxy);
}

void mglTextureStorage1D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width)
{
    Texture *tex;

    if (!mglTexStorageInternalFormatValid(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(checkMaxLevels(levels, width, 1, 1), GL_INVALID_OPERATION);

    tex = getTex(ctx, texture, 0);
    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(tex->target == GL_TEXTURE_1D, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    texStorage(ctx, tex, 1, levels, false, internalformat, width, 1, 1, false);
}

void mglTexStorage2D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    Texture *tex;
    GLboolean is_array;
    GLboolean proxy;
    GLuint num_faces;

    is_array = false;
    proxy = false;
    num_faces = 1;

    switch(target)
    {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            break;

        case GL_PROXY_TEXTURE_2D:
        case GL_PROXY_TEXTURE_RECTANGLE:
            proxy = true;
            break;

        case GL_TEXTURE_CUBE_MAP:
            num_faces = 6;
            proxy = false;
            break;

        case GL_PROXY_TEXTURE_CUBE_MAP:
            num_faces = 6;
            proxy = true;
            break;

        case GL_TEXTURE_1D_ARRAY:
            is_array = true;
            break;

        case GL_PROXY_TEXTURE_1D_ARRAY:
            is_array = true;
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(mglTexStorageInternalFormatValid(internalformat), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);

    if (target == GL_TEXTURE_RECTANGLE || target == GL_PROXY_TEXTURE_RECTANGLE) {
        ERROR_CHECK_RETURN(levels == 1, GL_INVALID_OPERATION);
    } else if (target == GL_TEXTURE_1D_ARRAY || target == GL_PROXY_TEXTURE_1D_ARRAY) {
        ERROR_CHECK_RETURN(checkMaxLevels(levels, width, 1, 1), GL_INVALID_OPERATION);
    } else {
        ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, 1), GL_INVALID_OPERATION);
    }

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, height, 1, 0);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);

    texStorage(ctx, tex, num_faces, levels, is_array, internalformat, width, height, 1, proxy);
}


/* ARB_sparse_texture: commit or decommit pages of a sparse texture.  GL states
 * the region in texels and requires page alignment; the renderer converts to the
 * tile-space region Metal takes. */
void mglCommitSparsePages(GLMContext ctx, Texture *tex, GLint level,
                                 GLint xoffset, GLint yoffset, GLint zoffset,
                                 GLsizei width, GLsizei height, GLsizei depth,
                                 GLboolean commit)
{
    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(tex->params.sparse, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(tex->immutable_storage, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(tex->sparse_heap != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN((GLuint)level < tex->num_levels, GL_INVALID_OPERATION);

    /* Page granularity is GL_VIRTUAL_PAGE_SIZE_{X,Y} = 64, Z = 1.
     * Offsets must be page-aligned (INVALID_VALUE).  Sizes must be page
     * multiples unless they end exactly on the level edge, and the region
     * must lie inside the level (INVALID_OPERATION).  A cube map is a
     * six-layer array for commitment. */
    ERROR_CHECK_RETURN(tex->sparse_page_x > 0 && tex->sparse_page_y > 0, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN((xoffset % tex->sparse_page_x) == 0 &&
                       (yoffset % tex->sparse_page_y) == 0, GL_INVALID_VALUE);

    {
        GLuint level_w = tex->width;
        GLuint level_h = tex->height;
        GLuint level_d = tex->depth ? tex->depth : 1u;
        GLuint z_extent;
        TextureLevel *lvl = NULL;

        if (tex->faces[0].levels && (GLuint)level < tex->num_levels)
            lvl = &tex->faces[0].levels[level];
        if (lvl && lvl->width)
            level_w = lvl->width;
        if (lvl && lvl->height)
            level_h = lvl->height;
        if (lvl && lvl->depth)
            level_d = lvl->depth;

        if (tex->target == GL_TEXTURE_CUBE_MAP)
            z_extent = 6u;
        else if (tex->target == GL_TEXTURE_3D ||
                 tex->target == GL_TEXTURE_2D_ARRAY ||
                 tex->target == GL_TEXTURE_CUBE_MAP_ARRAY ||
                 tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
            z_extent = level_d ? level_d : 1u;
        else
            z_extent = 1u;

        if (((GLuint)width % (GLuint)tex->sparse_page_x) != 0u &&
            (GLuint)xoffset + (GLuint)width != level_w) {
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
        if (((GLuint)height % (GLuint)tex->sparse_page_y) != 0u &&
            (GLuint)yoffset + (GLuint)height != level_h) {
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
        /* Page Z is 1, so depth is always a multiple. */
        if ((uint64_t)xoffset + (uint64_t)width > (uint64_t)level_w ||
            (uint64_t)yoffset + (uint64_t)height > (uint64_t)level_h ||
            (uint64_t)zoffset + (uint64_t)depth > (uint64_t)z_extent) {
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
    }

    if (commit) {
        if (!mglSparseCommitAppend(tex, level, xoffset, yoffset, zoffset,
                                   width, height, depth)) {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
        }
    } else if (!mglSparseCommitSubtract(tex, level, xoffset, yoffset, zoffset,
                                        width, height, depth)) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
    }

    {
        /* Match mglRenderSparseTextureCommit tile walk: ceil(w/64)*ceil(h/64)
         * per slice (2D/array/cube) or *depth (3D).  MS is emulated as a
         * 2D array of sample planes (stride 8); GL commitment applies to
         * every sample, so expand depth to cover those Metal slices. */
        const uint32_t tile = 64u;
        const uint32_t ms_stride = 8u; /* MGL_TEXTURE_MS_PLANE_STRIDE */
        uint64_t tiles_x = ((uint64_t)width + tile - 1u) / tile;
        uint64_t tiles_y = ((uint64_t)height + tile - 1u) / tile;
        uint64_t slices;
        uint64_t heap_off = 0;
        uint32_t commit_z = (uint32_t)zoffset;
        uint32_t commit_d = (uint32_t)(depth > 0 ? depth : 1);
        if (tex->target == GL_TEXTURE_3D)
            slices = commit_d;
        else if (tex->target == GL_TEXTURE_2D_MULTISAMPLE) {
            commit_z = 0u;
            commit_d = ms_stride;
            slices = commit_d;
        } else if (tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
            commit_z = (uint32_t)zoffset * ms_stride;
            commit_d = (uint32_t)(depth > 0 ? depth : 1) * ms_stride;
            slices = commit_d;
        } else
            slices = commit_d;
        if (commit) {
            heap_off = tex->sparse_heap_next_tile;
            tex->sparse_heap_next_tile += tiles_x * tiles_y * slices;
        }
        (void)mglRenderSparseTextureCommit(
            tex->sparse_heap, mglRenderTextureHandle(tex),
            (uint32_t)xoffset, (uint32_t)yoffset, commit_z,
            (uint32_t)width, (uint32_t)height, commit_d,
            (uint32_t)level, commit ? 1 : 0, heap_off);
        /* Drain the renderer's CB so later compute imageStore / sample on
         * this texture cannot race the mapping update. */
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
        /* Unmap only — nothing written yet.  Clear+Unmap after stores. */
        mglSparseScrubTextureLevel(tex, level, 0);
    }
}

/* Drop stray mappings on pages that are not in the CPU commit list.  Metal
 * may keep tiles mapped after a write to an uncommitted region; Unmap makes
 * subsequent reads return zero and matches ARB_sparse_texture2. */
int mglSparsePageIntersectsCommit(const Texture *tex, GLint level,
                                         GLint x, GLint y, GLint z,
                                         GLint w, GLint h, GLint d)
{
    GLuint i;
    const int ms = tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                   tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    for (i = 0; i < tex->sparse_commit_count; i++) {
        const MGLSparseCommit *c = &tex->sparse_commits[i];
        GLint cz = c->z, cd = c->d;
        if (c->level != level)
            continue;
        /* MS commitment is recorded in GL layer space; scrub walks Metal
         * sample-plane slices.  Map plane → layer for the Z test. */
        if (ms) {
            const GLint stride = 8;
            if (tex->target == GL_TEXTURE_2D_MULTISAMPLE) {
                cz = 0;
                cd = stride;
            } else {
                cz = c->z * stride;
                cd = c->d * stride;
            }
        }
        if (x < c->x + c->w && x + w > c->x &&
            y < c->y + c->h && y + h > c->y &&
            z < cz + cd && z + d > cz)
            return 1;
    }
    return 0;
}

void mglSparseScrubTextureLevel(Texture *tex, GLint level, int clear)
{
    TextureLevel *lvl;
    GLint page_x, page_y, page_z;
    GLint width, height, depth;
    GLint x, y, z;
    MGLSparseUnmapRegion *regions = NULL;
    uint32_t region_count = 0u, region_cap = 0u;
    void *mtl;

    if (!tex || !tex->params.sparse || !tex->sparse_heap)
        return;
    if (level < 0 || (GLuint)level >= tex->num_levels || !tex->faces[0].levels)
        return;
    lvl = &tex->faces[0].levels[level];
    if (!lvl || lvl->width == 0)
        return;
    mtl = mglRenderTextureHandle(tex);
    if (!mtl)
        return;

    page_x = tex->sparse_page_x > 0 ? tex->sparse_page_x : 64;
    page_y = tex->sparse_page_y > 0 ? tex->sparse_page_y : 64;
    page_z = tex->sparse_page_z > 0 ? tex->sparse_page_z : 1;
    width = (GLint)lvl->width;
    height = (GLint)MAX(lvl->height, 1u);
    if (tex->target == GL_TEXTURE_CUBE_MAP)
        depth = 6;
    else if (tex->target == GL_TEXTURE_2D_MULTISAMPLE)
        depth = 8; /* MGL_TEXTURE_MS_PLANE_STRIDE sample planes */
    else if (tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
        depth = (GLint)MAX(lvl->depth, 1u) * 8;
    else
        depth = (GLint)MAX(lvl->depth, 1u);

    for (z = 0; z < depth; z += page_z) {
        GLint pd = page_z;
        if (z + pd > depth)
            pd = depth - z;
        for (y = 0; y < height; y += page_y) {
            GLint ph = page_y;
            if (y + ph > height)
                ph = height - y;
            for (x = 0; x < width; x += page_x) {
                GLint pw = page_x;
                if (x + pw > width)
                    pw = width - x;
                if (mglSparsePageIntersectsCommit(tex, level, x, y, z, pw, ph, pd))
                    continue;
                /* clear≠0 after imageStore/SubImage remaps; GPU-zero while
                 * mapped then Unmap.  Skip clear on commit-only scrub
                 * (unmapped pages — blit would fault). */
                if (clear)
                    (void)mglRenderSparseTextureClearRegion(
                        mtl, (uint32_t)x, (uint32_t)y, (uint32_t)z,
                        (uint32_t)pw, (uint32_t)ph, (uint32_t)pd,
                        (uint32_t)level);
                if (region_count == region_cap) {
                    uint32_t ncap = region_cap ? region_cap * 2u : 64u;
                    MGLSparseUnmapRegion *nregs = (MGLSparseUnmapRegion *)realloc(
                        regions, (size_t)ncap * sizeof(*nregs));
                    if (!nregs) {
                        free(regions);
                        return;
                    }
                    regions = nregs;
                    region_cap = ncap;
                }
                regions[region_count].x = (uint32_t)x;
                regions[region_count].y = (uint32_t)y;
                regions[region_count].z = (uint32_t)z;
                regions[region_count].width = (uint32_t)pw;
                regions[region_count].height = (uint32_t)ph;
                regions[region_count].depth = (uint32_t)pd;
                regions[region_count].level = (uint32_t)level;
                region_count++;
            }
        }
    }
    if (region_count > 0u)
        (void)mglRenderSparseTextureUnmapRegions(tex->sparse_heap, mtl, regions,
                                                 region_count);
    free(regions);
}

void mglSparseScrubBoundImages(GLMContext ctx)
{
    GLuint i, n;

    if (!ctx)
        return;
    n = STATE(var).max_image_units;
    if (n > (GLuint)TEXTURE_UNITS)
        n = (GLuint)TEXTURE_UNITS;
    for (i = 0; i < n; i++) {
        ImageUnit *iu = &STATE(image_units)[i];
        Texture *tex = iu->tex;
        if (!tex || !tex->params.sparse)
            continue;
        mglSparseScrubTextureLevel(tex, (GLint)iu->level, 1);
    }
}

void mglTexPageCommitmentARB(GLMContext ctx, GLenum target, GLint level,
                             GLint xoffset, GLint yoffset, GLint zoffset,
                             GLsizei width, GLsizei height, GLsizei depth,
                             GLboolean commit)
{
    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(xoffset >= 0 && yoffset >= 0 && zoffset >= 0, GL_INVALID_VALUE);
    /* depth may be 0 for targets whose minDepth is 0 (2D multisample).
     * A too-large width must still report INVALID_OPERATION, not VALUE. */
    ERROR_CHECK_RETURN(width > 0 && height > 0 && depth >= 0, GL_INVALID_VALUE);

    mglCommitSparsePages(ctx, getTex(ctx, 0, target), level, xoffset, yoffset,
                         zoffset, width, height, depth, commit);
}

void mglTexturePageCommitmentEXT(GLMContext ctx, GLuint texture, GLint level,
                                 GLint xoffset, GLint yoffset, GLint zoffset,
                                 GLsizei width, GLsizei height, GLsizei depth,
                                 GLboolean commit)
{
    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(xoffset >= 0 && yoffset >= 0 && zoffset >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(width > 0 && height > 0 && depth >= 0, GL_INVALID_VALUE);

    mglCommitSparsePages(ctx, findTexture(ctx, texture), level, xoffset, yoffset,
                         zoffset, width, height, depth, commit);
}

void mglTextureStorage2D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    Texture *tex;
    GLboolean is_array = GL_FALSE;
    GLuint faces = 1u;

    if (!mglTexStorageInternalFormatValid(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);

    tex = getTex(ctx, texture, 0);
    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);
    switch (tex->target) {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            break;
        case GL_TEXTURE_CUBE_MAP:
            faces = 6u;
            break;
        case GL_TEXTURE_1D_ARRAY:
            is_array = GL_TRUE;
            ERROR_CHECK_RETURN(checkMaxLevels(levels, width, 1, 1), GL_INVALID_OPERATION);
            break;
        default:
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
    }
    if (tex->target != GL_TEXTURE_1D_ARRAY) {
        if (tex->target == GL_TEXTURE_RECTANGLE) {
            ERROR_CHECK_RETURN(levels == 1, GL_INVALID_OPERATION);
        } else {
            ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, 1), GL_INVALID_OPERATION);
        }
    }
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    texStorage(ctx, tex, faces, levels, is_array, internalformat, width, height, 1, false);
}

void mglTextureStorage2DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    Texture *tex;

    if (!mglTexStorageInternalFormatValid(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width <= 0 || height <= 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if ((GLuint)width > STATE(var).max_texture_size ||
        (GLuint)height > STATE(var).max_texture_size) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    /* GL 4.6 §8.19 / AUDIT-034: samples > MAX_SAMPLES → INVALID_VALUE. */
    {
        GLenum sample_err = mglMultisampleCountValidationError(ctx, samples);
        if (sample_err != GL_NO_ERROR) {
            ERROR_RETURN(sample_err);
            return;
        }
    }

    tex = getTex(ctx, texture, 0);
    if (!tex || tex->target != GL_TEXTURE_2D_MULTISAMPLE) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (tex->immutable_storage) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    (void)mglTextureStorageMultisampleMetadata(ctx, tex, GL_TEXTURE_2D_MULTISAMPLE, samples, internalformat, width, height, 1, fixedsamplelocations, GL_FALSE);
}

void mglTexStorage3D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
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
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY: // keep proxy case explicit (no duplicate GL_TEXTURE_CUBE_MAP_ARRAY here)
            is_array = true;
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(mglTexStorageInternalFormatValid(internalformat), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(depth > 0, GL_INVALID_VALUE);
    if (target == GL_TEXTURE_3D || target == GL_PROXY_TEXTURE_3D) {
        ERROR_CHECK_RETURN(!mglInternalFormatIsRGTC(internalformat), GL_INVALID_OPERATION);
    }
    if (target == GL_TEXTURE_CUBE_MAP_ARRAY ||
        target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY) {
        ERROR_CHECK_RETURN(width == height, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN((depth % 6) == 0, GL_INVALID_VALUE);
    }

    if (target == GL_TEXTURE_2D_ARRAY ||
        target == GL_TEXTURE_CUBE_MAP_ARRAY ||
        target == GL_PROXY_TEXTURE_2D_ARRAY ||
        target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY) {
        ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, 1), GL_INVALID_OPERATION);
    } else {
        ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, depth), GL_INVALID_OPERATION);
    }

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, height, depth, 0);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);

    texStorage(ctx, tex, 1, levels, is_array, internalformat, width, height, depth, proxy);
}

void mglTextureStorage3D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    Texture *tex;
    GLboolean is_array = GL_FALSE;

    if (!mglTexStorageInternalFormatValid(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(depth > 0, GL_INVALID_VALUE);

    tex = getTex(ctx, texture, 0);
    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);
    switch (tex->target) {
        case GL_TEXTURE_3D:
            ERROR_CHECK_RETURN(!mglInternalFormatIsRGTC(internalformat), GL_INVALID_OPERATION);
            ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, depth), GL_INVALID_OPERATION);
            break;
        case GL_TEXTURE_2D_ARRAY:
            is_array = GL_TRUE;
            ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, 1), GL_INVALID_OPERATION);
            break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            is_array = GL_TRUE;
            ERROR_CHECK_RETURN(width == height, GL_INVALID_VALUE);
            ERROR_CHECK_RETURN((depth % 6) == 0, GL_INVALID_VALUE);
            ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, 1), GL_INVALID_OPERATION);
            break;
        default:
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
    }
    ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    texStorage(ctx, tex, 1, levels, is_array, internalformat, width, height, depth, false);
}

void mglTextureStorage3DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    Texture *tex;

    if (!mglTexStorageInternalFormatValid(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width <= 0 || height <= 0 || depth <= 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if ((GLuint)width > STATE(var).max_texture_size ||
        (GLuint)height > STATE(var).max_texture_size ||
        (GLuint)depth > STATE(var).max_array_texture_layers) {
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

    tex = getTex(ctx, texture, 0);
    if (!tex || tex->target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (tex->immutable_storage) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    mglTextureStorageMultisampleMetadata(ctx, tex, GL_TEXTURE_2D_MULTISAMPLE_ARRAY, samples, internalformat, width, height, depth, fixedsamplelocations, GL_FALSE);
}


#pragma mark clear tex image
void mglClearTexImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, const void *data)
{
    if (texture == 0u) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (level < 0 || level >= (GLint)tex->num_levels ||
        !tex->faces[0].levels ||
        !tex->faces[0].levels[level].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (sizeForFormatType(format, type) == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[level];
    if (mglTextureHasCompressedInternalFormat(tex)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    GLenum compatibility_error = mglClearTexFormatCompatibilityError(tex->internalformat, format);
    if (compatibility_error != GL_NO_ERROR) {
        ERROR_RETURN(compatibility_error);
        return;
    }

    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    GLsizei width = (GLsizei)lvl->width;
    GLsizei height = (GLsizei)MAX(lvl->height, 1u);
    GLsizei depth = (GLsizei)MAX(lvl->depth, 1u);
    if (mglClearTextureLevelCPU(lvl, tex->internalformat, 0, 0, 0, width, height, depth, format, type, data)) {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        return;
    }
    
    // For now, use texSubImage to clear - fill with the clear data
    width = tex->width >> level;
    height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    // If data is NULL, clear to zero
    if (data == NULL) {
        size_t pixel_size = sizeForFormatType(format, type);

        // CRITICAL SECURITY FIX: Prevent integer overflow in texture clear allocation
        if (width > SIZE_MAX / height / pixel_size) {
            fprintf(stderr, "MGL SECURITY ERROR: Texture clear allocation would overflow: %dx%dx%zu\n", width, height, pixel_size);
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }

        size_t size = width * height * pixel_size;
        void *clear_data = calloc(1, size);
        if (clear_data) {
            texSubImage(ctx, tex, 0, level, 0, 0, 0, width, height, 1, format, type, clear_data);
            free(clear_data);
        }
    } else {
        // Fill entire texture with the provided clear value
        size_t pixel_size = sizeForFormatType(format, type);

        // CRITICAL SECURITY FIX: Prevent integer overflow in texture fill allocation
        if (width > SIZE_MAX / height / pixel_size) {
            fprintf(stderr, "MGL SECURITY ERROR: Texture fill allocation would overflow: %dx%dx%zu\n", width, height, pixel_size);
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }

        size_t size = width * height * pixel_size;
        void *fill_data = malloc(size);
        if (fill_data) {
            // Replicate the clear value across the entire buffer
            for (size_t i = 0; i < width * height; i++) {
                memcpy((char*)fill_data + i * pixel_size, data, pixel_size);
            }
            texSubImage(ctx, tex, 0, level, 0, 0, 0, width, height, 1, format, type, fill_data);
            free(fill_data);
        }
    }
}

void mglClearTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data)
{
    if (texture == 0u) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (level < 0 || level >= (GLint)tex->num_levels ||
        !tex->faces[0].levels ||
        !tex->faces[0].levels[level].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (sizeForFormatType(format, type) == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[level];
    if (mglTextureHasCompressedInternalFormat(tex)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    GLenum compatibility_error = mglClearTexFormatCompatibilityError(tex->internalformat, format);
    if (compatibility_error != GL_NO_ERROR) {
        ERROR_RETURN(compatibility_error);
        return;
    }

    if (width == 0 || height == 0 || depth == 0) {
        return;
    }

    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    if (mglClearTextureLevelCPU(lvl, tex->internalformat, xoffset, yoffset, zoffset, width, height, depth, format, type, data)) {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        return;
    }
    
    size_t pixel_size = sizeForFormatType(format, type);

    // CRITICAL SECURITY FIX: Prevent integer overflow in texture subimage allocation
    if (width > SIZE_MAX / height / depth / pixel_size) {
        fprintf(stderr, "MGL SECURITY ERROR: Texture subimage allocation would overflow: %dx%dx%dx%zu\n", width, height, depth, pixel_size);
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }

    size_t size = width * height * depth * pixel_size;
    
    if (data == NULL) {
        void *clear_data = calloc(1, size);
        if (clear_data) {
            texSubImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, type, clear_data);
            free(clear_data);
        }
    } else {
        void *fill_data = malloc(size);
        if (fill_data) {
            for (size_t i = 0; i < width * height * depth; i++) {
                memcpy((char*)fill_data + i * pixel_size, data, pixel_size);
            }
            texSubImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, type, fill_data);
            free(fill_data);
        }
    }
}

#pragma mark compressed tex image


