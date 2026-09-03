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
#include "mgl_render.h"
#include "mgl_texture_debug.h"
#include "mgl_texture_transfer.h"

extern void *getBufferData(GLMContext ctx, Buffer *ptr);
extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern GLsizei mglSafeMaxTextureSize(GLMContext ctx);
extern GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
extern bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *iparam, GLfloat *fparam);
#include "mgl_trace_log.h"
extern GLint mglTexLevelCanonicalInternalFormat(GLint internalformat);
extern bool mglTexLevelInternalFormatCompressed(GLint internalformat);
extern GLint mglCompressedInternalFormatToSizedUncompressed(GLint internalformat);

/* Spec default image-unit state: name=0, level=0, layered=FALSE, layer=0,
 * access=GL_READ_ONLY, format=GL_R8. */
static void mglResetImageUnit(ImageUnit *iu)
{
    if (!iu) {
        return;
    }
    if (iu->mtl_image_view) {
        mglRenderReleaseMetalObject(iu->mtl_image_view);
        iu->mtl_image_view = NULL;
    }
    bzero(iu, sizeof(*iu));
    iu->access = GL_READ_ONLY;
    iu->internalformat = GL_R8;
}
extern GLint mglTexLevelComponentBits(GLint internalformat, GLenum pname);
extern GLint mglTexLevelComponentType(GLint internalformat, GLenum pname);
extern size_t mglPixelTypeDatumBytes(GLenum type);

#ifndef MGL_VERBOSE_TEXTURE_UPLOAD_LOGS
#define MGL_VERBOSE_TEXTURE_UPLOAD_LOGS 0
#endif

#ifndef MGL_VERBOSE_TEXTURE_BIND_LOGS
#define MGL_VERBOSE_TEXTURE_BIND_LOGS 0
#endif

bool texSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels);
void invalidateTexture(GLMContext ctx, Texture *tex);
bool ensureTextureLevelCapacity(GLMContext ctx, Texture *tex, GLuint required_levels);


GLuint textureIndexFromTarget(GLMContext ctx, GLenum target)
{
    (void)ctx;

    switch(target)
    {
        case GL_PROXY_TEXTURE_1D:
        case GL_TEXTURE_BUFFER: return _TEXTURE_BUFFER_TARGET;
        case GL_TEXTURE_1D: return _TEXTURE_1D;
        case GL_PROXY_TEXTURE_2D:
        case GL_TEXTURE_2D: return _TEXTURE_2D;
        case GL_PROXY_TEXTURE_3D:
        case GL_TEXTURE_3D: return _TEXTURE_3D;
        case GL_PROXY_TEXTURE_RECTANGLE:
        case GL_TEXTURE_RECTANGLE: return _TEXTURE_RECTANGLE;
        case GL_PROXY_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_1D_ARRAY: return _TEXTURE_1D_ARRAY;
        case GL_PROXY_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_2D_ARRAY: return _TEXTURE_2D_ARRAY;
        case GL_PROXY_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        case GL_TEXTURE_CUBE_MAP: return _TEXTURE_CUBE_MAP;
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY: return _TEXTURE_CUBE_MAP_ARRAY;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE: return _TEXTURE_2D_MULTISAMPLE;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return _TEXTURE_2D_MULTISAMPLE_ARRAY;
        case GL_RENDERBUFFER: return _RENDERBUFFER;

        default:
            return _MAX_TEXTURE_TYPES;
    }
}

static bool mglIsTextureObjectTarget(GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_TEXTURE_BUFFER:
            return true;
        default:
            return false;
    }
}

static GLenum mglCanonicalTextureObjectTarget(GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        case GL_PROXY_TEXTURE_CUBE_MAP:
            return GL_TEXTURE_CUBE_MAP;
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
            return GL_TEXTURE_CUBE_MAP_ARRAY;
        case GL_PROXY_TEXTURE_1D:
            return GL_TEXTURE_1D;
        case GL_PROXY_TEXTURE_2D:
            return GL_TEXTURE_2D;
        case GL_PROXY_TEXTURE_3D:
            return GL_TEXTURE_3D;
        case GL_PROXY_TEXTURE_RECTANGLE:
            return GL_TEXTURE_RECTANGLE;
        case GL_PROXY_TEXTURE_1D_ARRAY:
            return GL_TEXTURE_1D_ARRAY;
        case GL_PROXY_TEXTURE_2D_ARRAY:
            return GL_TEXTURE_2D_ARRAY;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
            return GL_TEXTURE_2D_MULTISAMPLE;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
        default:
            return target;
    }
}

Texture *currentTexture(GLMContext ctx, GLuint index)
{
    GLuint active_texture;

    active_texture = STATE(active_texture);

    return STATE(texture_units[active_texture].textures[index]);
}

Texture *newTexObj(GLMContext ctx, GLenum target)
{
    Texture *ptr;
    GLuint index;
    GLenum object_target;

    object_target = mglCanonicalTextureObjectTarget(target);
    index = textureIndexFromTarget(ctx, object_target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        STATE(error) = GL_INVALID_ENUM;
        return NULL;
    }

    ptr = (Texture *)malloc(sizeof(Texture));
    // CRITICAL SECURITY FIX: Check malloc result instead of using assert()
    if (!ptr) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate memory for texture\n");
        STATE(error) = GL_OUT_OF_MEMORY;
        return NULL;
    }

    bzero(ptr, sizeof(Texture));

    ptr->name = TEX_OBJ_RES_NAME;
    ptr->target = object_target;
    ptr->index = index;

    float black_color[] = {0,0,0,0};

    ptr->params.depth_stencil_mode = GL_DEPTH_COMPONENT;
    ptr->params.base_level = 0;
    memcpy(ptr->params.border_color, black_color, 4 * sizeof(float));
    ptr->params.compare_func = GL_LEQUAL;
    ptr->params.compare_mode = GL_NONE;
    ptr->params.lod_bias = 0.0;
    /* GL 4.6 spec §8.14: initial MIN_FILTER and MAG_FILTER are both NEAREST.
     * A prior change defaulted to NEAREST_MIPMAP_LINEAR/LINEAR, which enables
     * mip filtering on textures that never explicitly set MIN_FILTER —
     * sampling a non-mipmapped texture (mipmapLevelCount==1, e.g. MC's block
     * atlas) then reads uninitialized mip levels and produces stripes.
     * Restore the spec-correct NEAREST defaults (matches 59f4f7d). */
    ptr->params.min_filter = GL_NEAREST;
    ptr->params.mag_filter = GL_NEAREST;
    ptr->params.max_anisotropy = 1.0;
    ptr->params.min_lod = -1000;
    ptr->params.max_lod = 1000;
    ptr->params.max_level = 1000;
    ptr->params.swizzle_r = GL_RED;
    ptr->params.swizzle_g = GL_GREEN;
    ptr->params.swizzle_b = GL_BLUE;
    ptr->params.swizzle_a = GL_ALPHA;
    ptr->params.wrap_s = GL_REPEAT;
    ptr->params.wrap_t = GL_REPEAT;
    ptr->params.wrap_r = GL_REPEAT;

    return ptr;
}

Texture *newTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    Texture *ptr;
    GLuint index;

    if (!ctx || texture == 0)
    {
        if (ctx) {
            STATE(error) = GL_INVALID_VALUE;
        }
        fprintf(stderr,
                "MGL ERROR: newTexture refused invalid name=%u target=0x%x ctx=%p\n",
                texture,
                target,
                (void *)ctx);
        return NULL;
    }

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        STATE(error) = GL_INVALID_ENUM;
        return NULL;
    }

    ptr = newTexObj(ctx, target);
    if (!ptr)
        return NULL;

    ptr->name = texture;

    return ptr;
}

static Texture *getTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    Texture *ptr;

    if (!ctx || texture == 0)
        return NULL;

    ptr = (Texture *)searchHashTable(&STATE(texture_table), texture);

    if (!ptr)
    {
        ptr = newTexture(ctx, target, texture);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(texture_table), texture, ptr);
    }
    else if (ptr->target != target)
    {
        fprintf(stderr,
                "MGL ERROR: texture name %u was first bound/created as target 0x%x, cannot bind/create as 0x%x\n",
                texture,
                ptr->target,
                target);
        STATE(error) = GL_INVALID_OPERATION;
        return NULL;
    }

    return ptr;
}

static int isTexture(GLMContext ctx, GLuint texture)
{
    Texture *ptr;

    if (!ctx || texture == 0)
        return 0;

    ptr = (Texture *)searchHashTable(&STATE(texture_table), texture);

    if (ptr)
        return 1;

    return 0;
}

Texture *findTexture(GLMContext ctx, GLuint texture)
{
    Texture *ptr;

    if (!ctx || texture == 0)
        return NULL;

    ptr = (Texture *)searchHashTable(&STATE(texture_table), texture);

    return ptr;
}

void mglReleaseGLSampledTextureCopy(GLMContext ctx, Texture *tex, const char *reason)
{
    if (!tex) {
        return;
    }

    if (tex->mtl_gl_sampled_data) {
        if (ctx) {
            mglRendererDeleteMetalObject(ctx, tex->mtl_gl_sampled_data);
        }
        tex->mtl_gl_sampled_data = NULL;
    }
    tex->mtl_gl_sampled_width = 0u;
    tex->mtl_gl_sampled_height = 0u;
    tex->mtl_gl_sampled_format = 0u;
    tex->mtl_gl_sampled_write_version = 0u;

    if (reason && tex->is_render_target) {
        static uint64_t s_release_sampled_copy_logs = 0u;
        uint64_t hit = ++s_release_sampled_copy_logs;
        if (hit <= 32u || (hit % 512u) == 0u) {
            fprintf(stderr,
                    "MGL RT-SAMPLE-COPY release tex=%u reason=%s rtVersion=%u hit=%" PRIu64 "\n",
                    tex->name,
                    reason,
                    tex->mtl_render_target_write_version,
                    hit);
        }
    }
}

static bool mglTextureIsSampleableColor2D(Texture *tex)
{
    if (!tex ||
        tex->target != GL_TEXTURE_2D ||
        tex->index != _TEXTURE_2D ||
        tex->is_render_target ||
        tex->internalformat == 0 ||
        mglTextureFormatLooksDepthOrStencil(tex->internalformat) ||
        tex->num_levels == 0 ||
        !tex->faces[0].levels) {
        return false;
    }

    TextureLevel *level0 = &tex->faces[0].levels[0];
    return level0->complete &&
           (level0->ever_written || level0->has_initialized_data);
}

static bool mglTextureCanEnterRecentSampled2DHistory(Texture *tex)
{
    if (!tex ||
        tex->target != GL_TEXTURE_2D ||
        tex->index != _TEXTURE_2D) {
        return false;
    }

    /*
     * Render-target textures are often bound as sampler inputs before their
     * Metal backing is created. Keep them as candidates and validate the final
     * Metal format at draw time; known depth/stencil formats never qualify.
     */
    if (tex->internalformat != 0 &&
        mglTextureFormatLooksDepthOrStencil(tex->internalformat)) {
        return false;
    }

    return true;
}

static void mglPushRecentSampled2DTexture(GLMContext ctx, GLuint unit, Texture *tex)
{
    if (!ctx || unit >= TEXTURE_UNITS ||
        !mglTextureCanEnterRecentSampled2DHistory(tex)) {
        return;
    }

    Texture **history = STATE(recent_sampled_2d_textures[unit]);
    if (history[0] == tex) {
        return;
    }

    for (GLuint i = 1; i < MGL_RECENT_SAMPLED_2D_HISTORY; i++) {
        if (history[i] == tex) {
            memmove(&history[1],
                    &history[0],
                    sizeof(Texture *) * i);
            history[0] = tex;
            return;
        }
    }

    memmove(&history[1],
            &history[0],
            sizeof(Texture *) * (MGL_RECENT_SAMPLED_2D_HISTORY - 1u));
    history[0] = tex;
}

static void mglRecordLastSampled2DTexture(GLMContext ctx, GLuint unit, Texture *tex)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return;
    }

    if (mglTextureIsSampleableColor2D(tex)) {
        STATE(last_sampled_2d_textures[unit]) = tex;
    }
    mglPushRecentSampled2DTexture(ctx, unit, tex);
}

static void mglRecordBoundSampled2DTextureIfReady(GLMContext ctx, Texture *tex)
{
    if (!ctx || !tex || tex->index != _TEXTURE_2D) {
        return;
    }

    for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
        if (STATE(texture_units[unit].textures[_TEXTURE_2D]) == tex) {
            mglRecordLastSampled2DTexture(ctx, unit, tex);
        }
    }
}

void mglClearLastSampled2DTextureIfMatches(GLMContext ctx, Texture *tex)
{
    if (!ctx || !tex) {
        return;
    }

    for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
        if (STATE(last_sampled_2d_textures[unit]) == tex) {
            STATE(last_sampled_2d_textures[unit]) = NULL;
        }
        for (GLuint i = 0; i < MGL_RECENT_SAMPLED_2D_HISTORY; i++) {
            if (STATE(recent_sampled_2d_textures[unit][i]) == tex) {
                STATE(recent_sampled_2d_textures[unit][i]) = NULL;
            }
        }
    }
}

static GLboolean mglTextureUnitHasAnyBinding(GLMContext ctx, GLuint unit)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return GL_FALSE;
    }

    for (GLuint i = 0; i < _MAX_TEXTURE_TYPES; i++) {
        if (STATE(texture_units[unit].textures[i])) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static Texture *mglChooseTextureUnitActiveBinding(GLMContext ctx, GLuint unit)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return NULL;
    }

    Texture *active = STATE(active_textures[unit]);
    if (active) {
        for (GLuint i = 0; i < _MAX_TEXTURE_TYPES; i++) {
            if (STATE(texture_units[unit].textures[i]) == active) {
                return active;
            }
        }
    }

    static const GLuint fallback_order[] = {
        _TEXTURE_2D,
        _TEXTURE_2D_ARRAY,
        _TEXTURE_CUBE_MAP,
        _TEXTURE_3D,
        _TEXTURE_1D,
        _TEXTURE_1D_ARRAY,
        _TEXTURE_RECTANGLE,
        _TEXTURE_CUBE_MAP_ARRAY,
        _TEXTURE_BUFFER_TARGET,
        _TEXTURE_2D_MULTISAMPLE,
        _TEXTURE_2D_MULTISAMPLE_ARRAY,
        _RENDERBUFFER
    };

    for (size_t i = 0; i < sizeof(fallback_order) / sizeof(fallback_order[0]); i++) {
        Texture *tex = STATE(texture_units[unit].textures[fallback_order[i]]);
        if (tex) {
            return tex;
        }
    }

    return NULL;
}

static void mglUpdateTextureUnitActiveMask(GLMContext ctx, GLuint unit)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return;
    }

    Texture *active = mglChooseTextureUnitActiveBinding(ctx, unit);
    GLuint mask_index = unit / 32u;
    GLuint mask = 1u << (unit % 32u);

    STATE(active_textures[unit]) = active;
    if (mglTextureUnitHasAnyBinding(ctx, unit)) {
        STATE(active_texture_mask[mask_index]) |= mask;
    } else {
        STATE(active_texture_mask[mask_index]) &= ~mask;
    }
}

Texture *getTex(GLMContext ctx, GLuint name, GLenum target)
{
    GLuint index;
    Texture *ptr;

    if (!ctx) {
        return NULL;
    }

    if (name == 0)
    {
        index = textureIndexFromTarget(ctx, target);
        if (index == _MAX_TEXTURE_TYPES)
        {
            STATE(error) = GL_INVALID_ENUM;
            return NULL;
        }

        ptr = currentTexture(ctx, index);
        
        // Create default texture if none exists for this target
        if (!ptr) {
            GLuint active_texture = STATE(active_texture);
            ptr = newTexObj(ctx, target);
            if (!ptr) {
                fprintf(stderr,
                        "MGL ERROR: getTex failed to create default texture target=0x%x activeUnit=%u\n",
                        target,
                        active_texture);
                return NULL;
            }
            STATE(texture_units[active_texture].textures[index]) = ptr;
            mglUpdateTextureUnitActiveMask(ctx, active_texture);
            mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX_BINDING);
            fprintf(stderr, "MGL: Created default texture for target 0x%x\n", target);
        }
    }
    else
    {
        ptr = findTexture(ctx, name);
        if (!ptr) {
            fprintf(stderr,
                    "MGL ERROR: getTex failed to resolve texture name=%u target=0x%x\n",
                    name,
                    target);
            STATE(error) = GL_INVALID_OPERATION;
            return NULL;
        }
        
        target = ptr->target;

        index = textureIndexFromTarget(ctx, target);
        if (index == _MAX_TEXTURE_TYPES)
        {
            STATE(error) = GL_INVALID_ENUM;
            return NULL;
        }
    }

    return ptr;
}

bool checkInternalFormatForMetal(GLMContext ctx, GLuint internalformat)
{
    // see if we can actually use this internal format
    GLenum mtl_format;
    mtl_format = mtlFormatForGLInternalFormat(internalformat);

    if (mtl_format == MGLPixelFormatInvalid)
    {
        // Only warn once per format to reduce log spam during capability probing
        static unsigned warned_formats[64] = {0};
        static int warned_count = 0;
        int already_warned = 0;
        for (int i = 0; i < warned_count && i < 64; i++) {
            if (warned_formats[i] == internalformat) { already_warned = 1; break; }
        }
        if (!already_warned && warned_count < 64) {
            warned_formats[warned_count++] = internalformat;
            // Only warn for standard GL format ranges (not internal Mesa/Gallium enums)
            // Skip 0x2xxx (GL get parameters), 0x8Dxx-0x9xxx (internal enums)
            if (internalformat >= 0x8040 && internalformat < 0x8D70) {
                fprintf(stderr, "MGL: checkInternalFormatForMetal - internalformat 0x%x has no Metal equivalent\n", internalformat);
            }
        }
        return false;
    }

    return true;
}


#pragma mark basic tex calls bind / delete / gen...
void mglGenTextures(GLMContext ctx, GLsizei n, GLuint *textures)
{
    static uint64_t s_gen_textures_calls = 0u;
    uint64_t call_id = ++s_gen_textures_calls;

    if (!ctx)
        return;

    if (n < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n == 0)
        return;

    if (!textures) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (MGL_VERBOSE_TEXTURE_BIND_LOGS) {
        fprintf(stderr,
                "MGL TRACE GenTextures call=%llu ctx=%p n=%d textures=%p\n",
                (unsigned long long)call_id,
                (void *)ctx,
                (int)n,
                (void *)textures);
    }

    while(n--)
    {
        GLuint name = getNewName(&STATE(texture_table));
        *textures++ = name;
        if (MGL_VERBOSE_TEXTURE_BIND_LOGS) {
            fprintf(stderr,
                    "MGL TRACE GenTextures call=%llu generated=%u currentName=%u tableCount=%zu tableCap=%zu\n",
                    (unsigned long long)call_id,
                    name,
                    STATE(texture_table).current_name,
                    STATE(texture_table).count,
                    STATE(texture_table).size);
        } else if (mglTraceLogIsEnabled() &&
                   (call_id <= 32ull || (call_id % 4096ull) == 0ull)) {
            mglTraceLogExternal("MGL TRACE GenTextures call=%llu generated=%u currentName=%u tableCount=%zu tableCap=%zu",
                                (unsigned long long)call_id,
                                name,
                                STATE(texture_table).current_name,
                                STATE(texture_table).count,
                                STATE(texture_table).size);
        }

        // TEX_OBJ_RES_NAME has special name.. skip it
        if (STATE(texture_table.current_name) == TEX_OBJ_RES_NAME)
            getNewName(&STATE(texture_table));
    }
}

void mglCreateTextures(GLMContext ctx, GLenum target, GLsizei n, GLuint *textures)
{
    if (!ctx)
        return;

    if (!mglIsTextureObjectTarget(target))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (n < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n == 0)
        return;

    if (!textures)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglGenTextures(ctx, n, textures);

    while(n--)
    {
        // create a texture object
        GLuint name = *textures++;
        if (!getTexture(ctx, target, name))
        {
            fprintf(stderr, "MGL Error: mglCreateTextures: failed to create texture %u for target 0x%x\n",
                    (unsigned)name, (unsigned)target);
            STATE(error) = GL_INVALID_ENUM;
            return;
        }
    }
}

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
                ctx ? ctx->state.active_texture : 0u,
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
    if (pending_write || (binding_changed && !mglBindNoFlushEnabled())) {
        if (binding_changed && !pending_write) {
            MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
        } else if (pending_write) {
            MGL_PERF_INC(g_mglFlushReasonOtherSinceSwap);
        }
        mglFlushPendingDraws(ctx);
    }

    STATE(texture_units[active_texture].textures[index]) = ptr;
    if (ptr) {
        STATE(active_textures[active_texture]) = ptr;
    }
    mglRecordLastSampled2DTexture(ctx, active_texture, ptr);
    mglUpdateTextureUnitActiveMask(ctx, active_texture);
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);

    mglTraceTextureUnitState(ctx, ptr ? "BindTexture" : "BindTexture.unbindTarget", active_texture, target, texture, STATE(active_textures[active_texture]));
}

static GLboolean mglTextureTargetUsesImageLayerParameter(GLenum target)
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
static GLboolean mglIsLegalImageUnitFormat(GLenum format)
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
     * GL_MAX_IMAGE_UNITS == 8 (independent of TEXTURE_UNITS == 128). */
    if (unit >= ctx->state.var.max_image_units) {
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
        mglResetImageUnit(&ctx->state.image_units[unit]);
        mglMarkStateDirtyBits(&ctx->state, DIRTY_IMAGE_UNIT_STATE);
        return;
    }

    ptr = getTex(ctx, texture, 0);

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

    if (!layered && mglTextureTargetUsesImageLayerParameter(ptr->target) && layer < 0) {
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

    /* Spec: an incompatible <format> vs texture internalformat makes image
     * loads/stores undefined; it is not a BindImageTexture error. CTS
     * basic-api-bind intentionally rebinds one R32F texture with RGBA8/RG16/
     * R32I image formats. Keep the requested format on the image unit. */

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
        /* Immutable textures reject level past allocated mip count (GL 4.6
         * §8.26). Mutable textures may bind any level in [0, MAX_LEVEL] even
         * when that mip was never defined — image loads return 0 and stores
         * are ignored (CTS incomplete_textures). */
        if (ptr->immutable_storage) {
            if (level >= (GLint)ptr->num_levels) {
                fprintf(stderr, "MGL Error: mglBindImageTexture: level >= num_levels (%d >= %d)\n", level, ptr->num_levels);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
        } else {
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
    unit_params.mtl_image_view = NULL;

    if (ctx->state.image_units[unit].mtl_image_view) {
        mglRenderReleaseMetalObject(ctx->state.image_units[unit].mtl_image_view);
        ctx->state.image_units[unit].mtl_image_view = NULL;
    }
    ctx->state.image_units[unit] = unit_params;

    mglMarkStateDirtyBits(&ctx->state, DIRTY_IMAGE_UNIT_STATE);
}

/* Callback for mglHashTableForEach: detach a deleted texture from every FBO
 * attachment that still holds a raw pointer to it.  Runs before the texture
 * is freed so the pointer is still valid for comparison. */
static void mglDetachTextureFromFramebuffers(GLuint name, void *data, void *user)
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

                if(ctx->state.active_textures[i] == tex) {
                    ctx->state.active_textures[i] = NULL;
                    cleared_unit = GL_TRUE;
                }
                if(ctx->state.last_sampled_2d_textures[i] == tex) {
                    ctx->state.last_sampled_2d_textures[i] = NULL;
                    cleared_unit = GL_TRUE;
                }

                for (int target_index = 0; target_index < _MAX_TEXTURE_TYPES; target_index++) {
                    if (ctx->state.texture_units[i].textures[target_index] == tex) {
                        ctx->state.texture_units[i].textures[target_index] = NULL;
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
                if(ctx->state.image_units[i].texture == name)
                {
                    mglResetImageUnit(&ctx->state.image_units[i]);

                    mglMarkStateDirtyBits(&ctx->state, DIRTY_IMAGE_UNIT_STATE);
                }
            }

            invalidateTexture(ctx, tex);

            /* OpenGL spec: when a texture is deleted, it is detached from any
             * framebuffer it is bound to.  MGL stores raw Texture* pointers in
             * FBOAttachment.buf.tex; without clearing them here the pointers
             * become dangling after free(tex) below, causing use-after-free
             * crashes in later draw calls that scan FBO attachments (e.g.
             * mglFindFramebufferColorTexturePairedWithDepth). */
            mglHashTableForEach(&STATE(framebuffer_table),
                                mglDetachTextureFromFramebuffers, tex);

            deleteHashElement(&STATE(texture_table), name);
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
    if (first >= TEXTURE_UNITS || (GLuint)count > TEXTURE_UNITS - first) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < count; i++) {
        GLuint tex_name = textures ? textures[i] : 0u;
        if (tex_name == 0u) {
            continue;
        }

        Texture *tex = findTexture(ctx, tex_name);
        if (!tex || tex->num_levels == 0 || !tex->faces[0].levels ||
            !tex->faces[0].levels[0].complete) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    for (GLsizei i = 0; i < count; i++) {
        GLuint tex_name = textures ? textures[i] : 0u;
        if (tex_name == 0u) {
            mglResetImageUnit(&ctx->state.image_units[first + i]);
            continue;
        }

        Texture *tex = findTexture(ctx, tex_name);
        mglBindImageTexture(ctx,
                            first + i,
                            tex_name,
                            0,
                            GL_FALSE,
                            0,
                            tex->access ? tex->access : GL_READ_ONLY,
                            tex->internalformat);
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
            if (pending_write || (binding_changed && !mglBindNoFlushEnabled())) {
                if (binding_changed && !pending_write) {
                    MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
                } else if (pending_write) {
                    MGL_PERF_INC(g_mglFlushReasonOtherSinceSwap);
                }
                mglFlushPendingDraws(ctx);
            }

            STATE(texture_units[unit].textures[index]) = ptr;
            STATE(active_textures[unit]) = ptr;
            mglRecordLastSampled2DTexture(ctx, unit, ptr);
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
    if (pending_write || (binding_changed && !mglBindNoFlushEnabled())) {
        if (binding_changed && !pending_write) {
            MGL_PERF_INC(g_mglFlushReasonBindTextureSinceSwap);
        } else if (pending_write) {
            MGL_PERF_INC(g_mglFlushReasonOtherSinceSwap);
        }
        mglFlushPendingDraws(ctx);
    }

    STATE(texture_units[unit].textures[index]) = ptr;
    STATE(active_textures[unit]) = ptr;
    mglRecordLastSampled2DTexture(ctx, unit, ptr);
    mglUpdateTextureUnitActiveMask(ctx, unit);
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX | DIRTY_TEX_BINDING);
    mglTraceTextureUnitState(ctx, "BindTextureUnit", unit, ptr->target, texture, ptr);
}

static GLuint mglTextureTargetMaxLevels(GLenum target,
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

static void mglTextureTargetLevelDimensions(GLenum target,
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
            STATE(error) = GL_INVALID_OPERATION;
            return;
        }
        for (GLuint face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
            if (!ptr->faces[face].levels ||
                !ptr->faces[face].levels[0].complete ||
                ptr->faces[face].levels[0].width != cube_width ||
                ptr->faces[face].levels[0].height != cube_height) {
                STATE(error) = GL_INVALID_OPERATION;
                return;
            }
        }
    } else if (ptr->target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        TextureLevel *base = &ptr->faces[0].levels[0];
        if (base->width != base->height || base->depth == 0u || (base->depth % 6u) != 0u) {
            STATE(error) = GL_INVALID_OPERATION;
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
            lvl->mtl_format = ptr->faces[face].levels[0].mtl_format;
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

    needs_storage_update = (!ptr->mtl_data || !ptr->mipmapped || !ptr->genmipmaps);

    ptr->mipmapped = true;
    ptr->genmipmaps = true;

    if (needs_storage_update) {
        ptr->dirty_bits |= DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA;
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
    if (!tex || !tex->mtl_base_level_view)
        return;

    if (ctx)
        mglRendererDeleteMetalObject(ctx, tex->mtl_base_level_view);
    tex->mtl_base_level_view = NULL;
    tex->mtl_base_level_view_source = NULL;
    tex->mtl_base_level_view_base = 0u;
    tex->mtl_base_level_view_max = 0u;
    tex->mtl_base_level_view_swizzle_r = 0u;
    tex->mtl_base_level_view_swizzle_g = 0u;
    tex->mtl_base_level_view_swizzle_b = 0u;
    tex->mtl_base_level_view_swizzle_a = 0u;
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
                        tex->mtl_data);

    mglClearLastSampled2DTextureIfMatches(ctx, tex);

    if (tex->mtl_data)
    {
        mglRendererDeleteMetalObject(ctx, tex->mtl_data);
        tex->mtl_data = NULL;
    }

    if (tex->mtl_gl_sampled_data)
    {
        mglRendererDeleteMetalObject(ctx, tex->mtl_gl_sampled_data);
        tex->mtl_gl_sampled_data = NULL;
    }

    if (tex->params.mtl_data)
    {
        mglRendererDeleteMetalObject(ctx, tex->params.mtl_data);
        tex->params.mtl_data = NULL;
    }

    /* release cached base-level texture view */
    if (tex->mtl_base_level_view)
    {
        mglRendererDeleteMetalObject(ctx, tex->mtl_base_level_view);
        tex->mtl_base_level_view = NULL;
        tex->mtl_base_level_view_source = NULL;
        tex->mtl_base_level_view_base = 0u;
        tex->mtl_base_level_view_max = 0u;
        tex->mtl_base_level_view_swizzle_r = 0u;
        tex->mtl_base_level_view_swizzle_g = 0u;
        tex->mtl_base_level_view_swizzle_b = 0u;
        tex->mtl_base_level_view_swizzle_a = 0u;
    }

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
    tex->dirty_on_gpu = 0;
    tex->is_render_target = GL_FALSE;
    tex->immutable_storage = GL_FALSE;
    tex->mipmapped = 0;
    tex->genmipmaps = GL_FALSE;
    tex->mtl_requires_private_storage = GL_FALSE;
    tex->internalformat = 0;
    tex->compressed_internalformat = 0;
    tex->width = 0;
    tex->height = 0;
    tex->depth = 0;
    tex->is_array = GL_FALSE;
    tex->complete = GL_FALSE;
    tex->num_levels = 0;
    tex->mipmap_levels = 0;
    tex->mtl_gl_sampled_data = NULL;
    tex->mtl_gl_sampled_width = 0;
    tex->mtl_gl_sampled_height = 0;
    tex->mtl_gl_sampled_format = 0;
    tex->mtl_gl_sampled_write_version = 0;
    tex->mtl_render_target_write_version = 0;
    tex->mtl_gl_sampled_dirty_mip_mask = 0u;
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
            STATE(error) = GL_OUT_OF_MEMORY;
            return;
        }

        tex->faces[face].levels = (TextureLevel *)calloc(tex->mipmap_levels, sizeof(TextureLevel));
        if (!tex->faces[face].levels) {
            fprintf(stderr, "MGL SECURITY ERROR: calloc failed for face %d with %d levels\n", face, tex->mipmap_levels);
            // CRITICAL FIX: Handle gracefully instead of crashing
            STATE(error) = GL_OUT_OF_MEMORY;
            return;
        }
    }

    tex->internalformat = internalformat;
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
        STATE(error) = GL_OUT_OF_MEMORY;
        return false;
    }

    for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
        new_levels[face] = (TextureLevel *)calloc(new_capacity, sizeof(TextureLevel));
        if (!new_levels[face]) {
            for (int i = 0; i < face; i++) {
                free(new_levels[i]);
            }
            STATE(error) = GL_OUT_OF_MEMORY;
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

static bool mglTextureStorageMultisampleMetadata(GLMContext ctx,
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
    lvl->mtl_format = 0;
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
    tex->mtl_requires_private_storage = GL_TRUE;
    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);

    return true;
}

bool checkTexLevelParams(GLMContext ctx, Texture *tex, GLint level, GLuint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type)
{
    GLuint base_width, base_height;

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

    if (tex->target == GL_TEXTURE_2D)
    {
        if (level != 0)
        {
            GLint check_level = level;
            base_width = tex->width;
            base_height = tex->height;

            while(check_level--)
            {
                base_width = MAX(base_width >> 1, 1u);
                base_height = MAX(base_height >> 1, 1u);
            }

            if (width != base_width || height != base_height)
            {
                fprintf(stderr,
                        "MGL ERROR: checkTexLevelParams size mismatch tex=%u level=%d got=%dx%d expected=%ux%u base=%ux%u\n",
                        tex->name,
                        level,
                        width,
                        height,
                        base_width,
                        base_height,
                        tex->width,
                        tex->height);
                return false;
            }
        }
    }

    if (internalformat)
    {
        // internal formats don't jive
        if (internalformat != tex->internalformat)
        {
            fprintf(stderr,
                    "MGL ERROR: checkTexLevelParams internalformat mismatch tex=%u level=%d got=0x%x expected=0x%x\n",
                    tex->name,
                    level,
                    internalformat,
                    tex->internalformat);
            return false;
        }
    }
    else
    {
        GLuint temp_internalformat;

        // check if we are expected to convert data
        temp_internalformat = internalFormatForGLFormatType(format, type);

        if (temp_internalformat != tex->internalformat)
        {
            fprintf(stderr,
                    "MGL ERROR: checkTexLevelParams format/type mismatch tex=%u level=%d derived=0x%x expected=0x%x format=0x%x type=0x%x\n",
                    tex->name,
                    level,
                    temp_internalformat,
                    tex->internalformat,
                    format,
                    type);
            return false;
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
    if (tex->immutable_storage)
    {
        // Compatibility: Treat glTexImage* on immutable texture as glTexSubImage*
        // This allows guests to update content using glTexImage* which is common in some drivers
        // We pass 0 for offsets. texSubImage will handle validation.
        if (pixels == NULL) {
            // Allocation-only call against immutable storage: nothing to upload.
            return true;
        }
        return texSubImage(ctx, tex, face, level, 0, 0, 0, width, height, depth, format, type, pixels);
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

    /* Remap compressed internalformat to sized uncompressed equivalent early,
     * so that all downstream comparisons (level==0 base-level change check,
     * Metal format selection, storage allocation) see the actual uncompressed
     * storage format.  This prevents false-positive base-level invalidation
     * when glTexImage* is called multiple times for the same texture (e.g.
     * CubeMap face uploads) with a compressed internalformat: without this
     * early remap, the second call would see internalformat=compressed but
     * tex->internalformat=uncompressed and invalidate the whole texture. */
    if (mglTexLevelInternalFormatCompressed(internalformat))
    {
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
            // invalidate texture because the base level width / height / depth / internal format are being changed...
            invalidateTexture(ctx, tex);

            initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
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
        tex->width = width;
        tex->height = height;
        tex->depth = depth;
        tex->complete = false;
    }
    else if (checkTexLevelParams(ctx, tex, level, internalformat, width, height, depth, format, type) == false)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
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

    switch(mtlFormatForGLInternalFormat(internalformat))
    {
        case MGLPixelFormatDepth16Unorm:
        case MGLPixelFormatDepth32Float:
        case MGLPixelFormatDepth24Unorm_Stencil8:
        case MGLPixelFormatDepth32Float_Stencil8:
            tex->mtl_requires_private_storage = true;
            break;

        default:
            tex->mtl_requires_private_storage = false;
            break;
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
                                            ctx->state.unpack.swap_bytes == GL_TRUE)) {
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
    mglReleaseGLSampledTextureCopy(ctx, tex, "texImage");
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);

    /* Populate depth_shadow for depth internal formats so that
     * glReadPixels(GL_DEPTH_COMPONENT) can read back uploaded data. */
    if (has_upload_source && !proxy && face == 0u && level == 0 &&
        mglInternalFormatIsDepthStencil(internalformat) &&
        internalformat != GL_STENCIL_INDEX8 &&
        internalformat != GL_STENCIL_INDEX &&
        internalformat != GL_DEPTH_STENCIL) {
        GLuint mtl_fmt = mtlFormatForGLInternalFormat(internalformat);
        if (mtl_fmt == MGLPixelFormatDepth16Unorm ||
            mtl_fmt == MGLPixelFormatDepth32Float ||
            mtl_fmt == MGLPixelFormatDepth24Unorm_Stencil8 ||
            mtl_fmt == MGLPixelFormatDepth32Float_Stencil8) {
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
                                case GL_DEPTH_COMPONENT24: {
                                    /* 24-bit depth stored in 3 bytes (LE). */
                                    uint32_t v = (uint32_t)pixel[0] |
                                                 ((uint32_t)pixel[1] << 8) |
                                                 ((uint32_t)pixel[2] << 16);
                                    d = (GLfloat)v / 16777215.0f;
                                    break;
                                }
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
            }
        }
    }

    if (!proxy) {
        mglRecordBoundSampled2DTextureIfReady(ctx, tex);
    }

    return true;
}

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
            ERROR_CHECK_RETURN(level==0, GL_INVALID_OPERATION);
            break;

        case GL_TEXTURE_RECTANGLE:
            ERROR_CHECK_RETURN(level==0, GL_INVALID_OPERATION);
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
            STATE(error) = GL_INVALID_VALUE;
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
            tex->samples = (GLuint)samples;
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
        bool is_rgtc = (internalformat == GL_COMPRESSED_RED_RGTC1 ||
                        internalformat == GL_COMPRESSED_SIGNED_RED_RGTC1 ||
                        internalformat == GL_COMPRESSED_RG_RGTC2 ||
                        internalformat == GL_COMPRESSED_SIGNED_RG_RGTC2);
        ERROR_CHECK_RETURN(!is_rgtc, GL_INVALID_OPERATION);
        ERROR_CHECK_RETURN(!mglInternalFormatIsDepthStencil(internalformat), GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(depth >= 0, GL_INVALID_VALUE);
    if (target == GL_TEXTURE_CUBE_MAP_ARRAY ||
        target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY) {
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
            tex->samples = (GLuint)samples;
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
                                    ctx->state.unpack.swap_bytes == GL_TRUE)) {
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

        if (tex->mtl_data == NULL)
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
        mglReleaseGLSampledTextureCopy(ctx, tex, "texSubImage-PBO");
        mglRecordBoundSampled2DTextureIfReady(ctx, tex);

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

    lvl->ever_written = GL_TRUE;
    lvl->suspicious_zero_upload = GL_FALSE;
    lvl->metal_data_authoritative = GL_FALSE;
    lvl->has_initialized_data = GL_TRUE;
    lvl->last_init_source = resolved_unpack_buf ? kTexSubImagePBO : kTexSubImageCPU;
    lvl->last_upload_size = required_bytes;
    lvl->last_src_ptr = resolved_src;
    lvl->last_src_hash = dst_hash;

    bool uploaded_direct = false;
    bool had_pending_texture_data_before_direct = (tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0;
    if (!resolved_unpack_buf &&
        tex->mtl_data &&
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
    mglReleaseGLSampledTextureCopy(ctx, tex, resolved_unpack_buf ? "texSubImage-PBO" : "texSubImage-CPU");
    mglRecordBoundSampled2DTextureIfReady(ctx, tex);

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
                ctx->state.unpack.row_length,
                ctx->state.unpack.alignment,
                ctx->state.unpack.skip_pixels,
                ctx->state.unpack.skip_rows,
                ctx->state.unpack.skip_images);
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

    if (!texSubImage3D(ctx, tex, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels)) {
        return;
    }
}

#pragma mark TexStorage

void texStorage(GLMContext ctx, Texture *tex, GLuint faces, GLsizei levels, GLboolean is_array, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean proxy)
{
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

    // bind it to metal
    mglRendererBindTexture(ctx, tex);

    ERROR_CHECK_RETURN(tex->mtl_data, GL_OUT_OF_MEMORY);
}

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
    if ((GLuint)width > ctx->state.var.max_texture_size ||
        (GLuint)height > ctx->state.var.max_texture_size) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if ((GLuint)samples > MAX(ctx->state.var.max_framebuffer_samples, 1u)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
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
    if (target == GL_TEXTURE_CUBE_MAP_ARRAY ||
        target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY) {
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
            ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, depth), GL_INVALID_OPERATION);
            break;
        case GL_TEXTURE_2D_ARRAY:
            is_array = GL_TRUE;
            ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, 1), GL_INVALID_OPERATION);
            break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            is_array = GL_TRUE;
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
    if ((GLuint)width > ctx->state.var.max_texture_size ||
        (GLuint)height > ctx->state.var.max_texture_size ||
        (GLuint)depth > ctx->state.var.max_array_texture_layers) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if ((GLuint)samples > MAX(ctx->state.var.max_framebuffer_samples, 1u)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
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
        mglReleaseGLSampledTextureCopy(ctx, tex, "glClearTexImage-CPU");
        mglRecordBoundSampled2DTextureIfReady(ctx, tex);
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
            STATE(error) = GL_OUT_OF_MEMORY;
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
            STATE(error) = GL_OUT_OF_MEMORY;
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
        mglReleaseGLSampledTextureCopy(ctx, tex, "glClearTexSubImage-CPU");
        mglRecordBoundSampled2DTextureIfReady(ctx, tex);
        return;
    }
    
    size_t pixel_size = sizeForFormatType(format, type);

    // CRITICAL SECURITY FIX: Prevent integer overflow in texture subimage allocation
    if (width > SIZE_MAX / height / depth / pixel_size) {
        fprintf(stderr, "MGL SECURITY ERROR: Texture subimage allocation would overflow: %dx%dx%dx%zu\n", width, height, depth, pixel_size);
        STATE(error) = GL_OUT_OF_MEMORY;
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


void mglCopyTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
{
    // Stub - not commonly used
    fprintf(stderr, "MGL WARNING: glCopyTexImage1D called (stub)\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCopyTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    GLuint face = 0u;

    if (!mglCopyTex2DFaceForTarget(target, &face)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || width < 0 || height < 0 || border != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_TEXTURE_RECTANGLE && level != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
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
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            srcFormat = GL_DEPTH_COMPONENT;
            srcType   = GL_FLOAT;
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
    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    mglRendererCopyTexSubImage(ctx, tex, face, level, 0, 0, x, y, width, height);
}

void mglCopyTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    if (target != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (width == 0) {
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, 0, 0, width, 1)) {
        if (STATE(error) == GL_NO_ERROR) {
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
        return;
    }
    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

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

    // This copies from the current read framebuffer to the texture
    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

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
    if (width == 0 || height == 0) {
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, yoffset, zoffset, width, height)) {
        if (STATE(error) == GL_NO_ERROR) {
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
        return;
    }
    if ((GLuint)zoffset >= tex->faces[0].levels[level].depth) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

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
    if (width == 0) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex || tex->target != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, 0, 0, width, 1)) {
        if (STATE(error) == GL_NO_ERROR) {
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
        return;
    }
    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

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
    if (width == 0 || height == 0) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
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
        mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

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
    if (width == 0 || height == 0) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex ||
        (tex->target != GL_TEXTURE_3D &&
         tex->target != GL_TEXTURE_2D_ARRAY &&
         tex->target != GL_TEXTURE_CUBE_MAP_ARRAY)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!mglCopyTextureSubImageValidate(ctx, tex, level, xoffset, yoffset, zoffset, width, height)) {
        if (STATE(error) == GL_NO_ERROR) {
            ERROR_RETURN(GL_INVALID_OPERATION);
        }
        return;
    }
    if ((GLuint)zoffset >= tex->faces[0].levels[level].depth) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    mglFlushPendingDrawsBeforeTextureWrite(ctx, tex);

    /* MGL_SYNC_STRICT: force a full flush + commit + waitUntilCompleted for regression triage */
    if (ctx->sync_strict) {
        mglFlushCommandBuffer(ctx);
        mglRendererFlush(ctx, true);
    }

    mglRendererCopyTexSubImage(ctx, tex, (GLuint)zoffset, level, xoffset, yoffset, x, y, width, height);
}

#pragma mark get tex image

void mglGetTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, void *pixels)
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
    if (STATE(buffers[_PIXEL_PACK_BUFFER])) {
        Buffer *ptr = STATE(buffers[_PIXEL_PACK_BUFFER]);

        if (ptr->mapped) {
            GLboolean persistent_map =
                ((ptr->storage_flags & GL_MAP_PERSISTENT_BIT) != 0u) &&
                ((ptr->access_flags & GL_MAP_PERSISTENT_BIT) != 0u);
            if (!persistent_map) {
                fprintf(stderr, "MGL Error: glGetTexImage: pixel pack buffer is mapped non-persistently\n");
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
        }

        if (ptr->size < 0) {
            fprintf(stderr, "MGL Error: glGetTexImage: pixel pack buffer has negative size\n");
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        if (!ptr->data.buffer_data) {
            fprintf(stderr, "MGL Error: glGetTexImage: pixel pack buffer has no CPU storage\n");
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        pack_buffer = ptr;
    }
    if (!STATE(buffers[_PIXEL_PACK_BUFFER]) && !pixels) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* If a PIXEL_PACK_BUFFER is bound, the `pixels` argument is actually a
     * byte offset into that buffer. Redirect it to the real CPU backing
     * storage, mirroring the mglReadPixels PBO path. */
    if (pack_buffer) {
        uintptr_t offset = (uintptr_t)pixels;
        uint8_t *base = (uint8_t *)(uintptr_t)pack_buffer->data.buffer_data;
        if (offset > (uintptr_t)pack_buffer->size) {
            fprintf(stderr, "MGL Error: glGetTexImage: pixel pack offset overflow off=%" PRIuPTR " size=%lld\n",
                    offset, (long long)pack_buffer->size);
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        pixels = (void *)(base + offset);
    }

    Texture *tex = getTex(ctx, 0, target);
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
    if (!mglVerifyInternalFormatAndFormatTypeForCall(ctx, tex->internalformat, format, type)) {
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

    bool render_target_needs_readback =
        (tex->is_render_target && tex->mtl_render_target_write_version != 0u) ||
        tex->metal_data_authoritative ||
        lvl->metal_data_authoritative;

    if (!render_target_needs_readback &&
        mglCopyTextureLevelToPackBuffer(lvl, tex->internalformat, width, height, depth, format, type, &pack_layout, pixels, ctx->state.pack.swap_bytes == GL_TRUE)) {
        return;
    }

    if (target == GL_TEXTURE_3D ||
        target == GL_TEXTURE_1D_ARRAY ||
        target == GL_TEXTURE_2D_ARRAY ||
        target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        if (render_target_needs_readback) {
            if (!tex->mtl_data) {
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
            }
            return;
        }
        memset((uint8_t *)pixels + pack_layout.skip_offset_bytes, 0, pack_layout.write_span_bytes);
        return;
    }

    /* If the texture has no Metal data, try the CPU data fallback first.
     * If that fails, let mtlGetTexImage handle Metal texture creation via
     * bindMTLTexture. */
    if (!tex->mtl_data) {
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
}

void mglGetTextureImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (STATE(buffers[_PIXEL_PACK_BUFFER])) {
        fprintf(stderr, "MGL WARNING: glGetTextureImage with GL_PIXEL_PACK_BUFFER is unsupported\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!pixels && bufSize > 0) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    if (tex->target == GL_TEXTURE_3D ||
        tex->target == GL_TEXTURE_1D_ARRAY ||
        tex->target == GL_TEXTURE_2D_ARRAY ||
        tex->target == GL_TEXTURE_CUBE_MAP ||
        tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        fprintf(stderr, "MGL WARNING: glGetTextureImage layered/3D readback texture=%u target=0x%x is unsupported\n",
                texture,
                tex->target);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (level < 0 || level >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    
    GLsizei width = tex->width >> level;
    GLsizei height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    size_t pixel_size = sizeForFormatType(format, type);
    if (pixel_size == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    MGLTexturePackLayout pack_layout;
    if (!mglComputeTexturePackLayout(ctx,
                                     width,
                                     height,
                                     1,
                                     pixel_size,
                                     "glGetTextureImage",
                                     &pack_layout)) {
        return;
    }

    if (pack_layout.required_bytes > (size_t)bufSize ||
        pack_layout.dst_pitch > UINT_MAX ||
        pack_layout.dst_image_size > UINT_MAX) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[level];
    if (!lvl || !lvl->complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }


    if (mglCopyTextureLevelToPackBuffer(lvl, tex->internalformat, width, height, 1, format, type, &pack_layout, pixels, ctx->state.pack.swap_bytes == GL_TRUE)) {
        return;
    }

    if (!tex->mtl_data) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
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
                                  0);
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
    if (STATE(buffers[_PIXEL_PACK_BUFFER])) {
        fprintf(stderr, "MGL WARNING: glGetTextureSubImage with GL_PIXEL_PACK_BUFFER is unsupported\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!pixels) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    Texture *tex = findTexture(ctx, texture);
    if (!tex) {
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
    if (pack_layout.required_bytes > (size_t)bufSize ||
        pack_layout.dst_pitch > UINT_MAX ||
        pack_layout.dst_image_size > UINT_MAX) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (tex->target == GL_TEXTURE_CUBE_MAP) {
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
                                                   ctx->state.pack.swap_bytes == GL_TRUE)) {
                memset(dst_base + ((size_t)z * pack_layout.dst_image_size), 0, pack_layout.write_span_bytes);
            }
        }
        return;
    }

    if (mglCopyTextureSubRectToPackBuffer(lvl,
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
                                          ctx->state.pack.swap_bytes == GL_TRUE)) {
        return;
    }

    if (!tex->mtl_data) {
        memset((uint8_t *)pixels + pack_layout.skip_offset_bytes, 0, pack_layout.write_span_bytes);
        return;
    }

    mglFlushCommandBuffer(ctx);
    mglRendererGetTexImage(ctx,
                                  tex,
                                  (uint8_t *)pixels + pack_layout.skip_offset_bytes,
                                  (GLuint)pack_layout.dst_pitch,
                                  (GLuint)pack_layout.dst_image_size,
                                  xoffset,
                                  yoffset,
                                  width,
                                  height,
                                  format,
                                  type,
                                  level,
                                  level_zoffset);
}

void mglGetCompressedTexImage(GLMContext ctx, GLenum target, GLint level, void *img)
{
    if (!img || level < 0) {
        ERROR_RETURN(level < 0 ? GL_INVALID_VALUE : GL_INVALID_OPERATION);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex || !mglTexLevelInternalFormatCompressed(tex->internalformat) ||
        level >= (GLint)tex->num_levels ||
        !tex->faces[0].levels ||
        !tex->faces[0].levels[level].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[level];
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
    if (!tex || !mglTexLevelInternalFormatCompressed(tex->internalformat) ||
        lod >= (GLint)tex->num_levels ||
        !tex->faces[0].levels ||
        !tex->faces[0].levels[lod].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[lod];
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

void mglTextureView(GLMContext ctx, GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
{
    fprintf(stderr, "MGL WARNING: glTextureView called (stub) - texture views not supported\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

static void mglTextureBufferRangeImpl(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer,
                                      GLintptr offset, GLsizeiptr size, bool whole_buffer);

void mglTextureBuffer(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer)
{
    mglTextureBufferRangeImpl(ctx, texture, internalformat, buffer, 0, 0, true);
}

void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    mglTextureBufferRangeImpl(ctx, texture, internalformat, buffer, offset, size, false);
}

static void mglTextureBufferRangeImpl(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer,
                                      GLintptr offset, GLsizeiptr size, bool whole_buffer)
{
    Texture *tex;
    Buffer *buf = NULL;
    size_t bytes_per_texel;
    GLsizeiptr attach_size;

    ERROR_CHECK_RETURN(texture != 0, GL_INVALID_OPERATION);

    tex = findTexture(ctx, texture);
    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(tex->target == GL_TEXTURE_BUFFER, GL_INVALID_OPERATION);

    mglFlushPendingDraws(ctx);

    if (buffer == 0)
    {
        if (tex->mtl_data)
        {
            mglRendererDeleteMetalObject(ctx, tex->mtl_data);
        }

        tex->texture_buffer = NULL;
        tex->texture_buffer_offset = 0;
        tex->texture_buffer_size = 0;
        tex->internalformat = internalformat;
        tex->width = 0;
        tex->height = 1;
        tex->depth = 1;
        tex->complete = GL_FALSE;
        tex->mtl_data = NULL;
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

    if (tex->mtl_data)
    {
        mglRendererDeleteMetalObject(ctx, tex->mtl_data);
        tex->mtl_data = NULL;
    }

    tex->texture_buffer = buf;
    tex->texture_buffer_offset = offset;
    tex->texture_buffer_size = attach_size;
    tex->internalformat = internalformat;
    tex->width = (GLuint)((size_t)attach_size / bytes_per_texel);
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
    if (!pixels && bufSize > 0) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex || !mglTexLevelInternalFormatCompressed(tex->internalformat) ||
        level >= (GLint)tex->num_levels ||
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
        *params = 0;
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
            *params = internalformat;
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
            *params = mglTexLevelInternalFormatCompressed(internalformat) ? GL_TRUE : GL_FALSE;
            break;
        case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
            *params = mglTexLevelInternalFormatCompressed(internalformat)
                ? (GLint)tex_level->data_size
                : 0;
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

static bool mglTextureParameterGetiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *params)
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

static bool mglTextureParameterGetfv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLfloat *params)
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

static bool mglTextureParameterGetIiv(TextureParameter *tex_params, GLenum pname, GLint *params)
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

static bool mglTextureParameterGetIuiv(TextureParameter *tex_params, GLenum pname, GLuint *params)
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

static bool mglTextureParameterGetTarget(GLMContext ctx, Texture *tex, GLenum pname, GLfloat *fparams,
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
    if (pname == GL_TEXTURE_IMMUTABLE_LEVELS) {
        GLint val = (GLint)tex->num_levels;
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

void mglGetTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex)
        return;

    if (pname == GL_IMAGE_FORMAT_COMPATIBILITY_TYPE) {
        *params = tex->immutable_storage
            ? (GLint)GL_IMAGE_FORMAT_COMPATIBILITY_BY_CLASS
            : (GLint)GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE;
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

    Texture *tex = getTex(ctx, 0, target);
    if (!tex)
        return;

    if (pname == GL_IMAGE_FORMAT_COMPATIBILITY_TYPE) {
        *params = tex->immutable_storage
            ? (GLuint)GL_IMAGE_FORMAT_COMPATIBILITY_BY_CLASS
            : (GLuint)GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE;
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
