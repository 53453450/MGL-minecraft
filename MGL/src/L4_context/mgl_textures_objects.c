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
extern GLint mglTexLevelCanonicalInternalFormat(GLint internalformat);
extern bool mglTexLevelInternalFormatCompressed(GLint internalformat);
extern GLint mglCompressedInternalFormatToSizedUncompressed(GLint internalformat);
extern GLenum mglTextureQueryInternalFormat(const Texture *tex);
extern GLint mglTextureQueryCompressedImageSize(const Texture *tex, GLint level);
extern int mglTryTexBufferLevelParameter(const Texture *tex, GLint level, GLenum pname,
                                         GLint *params, GLenum *err);

/* Spec default image-unit state: name=0, level=0, layered=FALSE, layer=0,
 * access=GL_READ_ONLY, format=GL_R8. */
void mglResetImageUnit(ImageUnit *iu)
{
    if (!iu) {
        return;
    }
    mglImageUnitBackendRelease(iu);
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

bool mglIsTextureObjectTarget(GLenum target)
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

GLenum mglCanonicalTextureObjectTarget(GLenum target)
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_ENUM);
        return NULL;
    }

    ptr = (Texture *)malloc(sizeof(Texture));
    // CRITICAL SECURITY FIX: Check malloc result instead of using assert()
    if (!ptr) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate memory for texture\n");
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NULL;
    }

    bzero(ptr, sizeof(Texture));

    /* The backend record lives as long as the object rather than being created
     * on first write: binding and copy paths read backend bookkeeping and must
     * not depend on an earlier write having allocated the record.  Failure
     * leaves it NULL and mglTextureBackendEnsure still covers that. */
    ptr->backend = (struct MGLTextureBackend *)calloc(
        1, sizeof(struct MGLTextureBackend));

    ptr->name = TEX_OBJ_RES_NAME;
    if (ctx)
        ptr->identity_generation = ctx->state.texture_table.deletion_generation;
    ptr->target = object_target;
    ptr->index = index;

    float black_color[] = {0,0,0,0};

    ptr->params.depth_stencil_mode = GL_DEPTH_COMPONENT;
    ptr->params.base_level = 0;
    memcpy(ptr->params.border_color, black_color, 4 * sizeof(float));
    ptr->params.border_color_type = GL_FLOAT;
    ptr->params.compare_func = GL_LEQUAL;
    ptr->params.compare_mode = GL_NONE;
    ptr->params.lod_bias = 0.0;
    /* §8.22: rectangle textures start at LINEAR / CLAMP_TO_EDGE. */
    bool rect = object_target == GL_TEXTURE_RECTANGLE;
    GLenum wrap = rect ? GL_CLAMP_TO_EDGE : GL_REPEAT;
    ptr->params.min_filter = rect ? GL_LINEAR : GL_NEAREST_MIPMAP_LINEAR;
    ptr->params.mag_filter = GL_LINEAR;
    ptr->params.max_anisotropy = 1.0;
    ptr->params.min_lod = -1000;
    ptr->params.max_lod = 1000;
    ptr->params.max_level = 1000;
    ptr->params.swizzle_r = GL_RED;
    ptr->params.swizzle_g = GL_GREEN;
    ptr->params.swizzle_b = GL_BLUE;
    ptr->params.swizzle_a = GL_ALPHA;
    ptr->params.wrap_s = wrap;
    ptr->params.wrap_t = wrap;
    ptr->params.wrap_r = wrap;

    return ptr;
}

Texture *newTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    Texture *ptr;
    GLuint index;

    if (!ctx || texture == 0)
    {
        if (ctx) {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_ENUM);
        return NULL;
    }

    ptr = newTexObj(ctx, target);
    if (!ptr)
        return NULL;

    ptr->name = texture;

    return ptr;
}

Texture *getTexture(GLMContext ctx, GLenum target, GLuint texture)
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NULL;
    }

    return ptr;
}

int isTexture(GLMContext ctx, GLuint texture)
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

GLboolean mglTextureUnitHasAnyBinding(GLMContext ctx, GLuint unit)
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

Texture *mglChooseTextureUnitActiveBinding(GLMContext ctx, GLuint unit)
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

void mglUpdateTextureUnitActiveMask(GLMContext ctx, GLuint unit)
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
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_ENUM);
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
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NULL;
        }
        
        target = ptr->target;

        index = textureIndexFromTarget(ctx, target);
        if (index == _MAX_TEXTURE_TYPES)
        {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_ENUM);
            return NULL;
        }
    }

    return ptr;
}

bool checkInternalFormatForMetal(GLMContext ctx, GLuint internalformat)
{
    // see if we can actually use this internal format
    GLenum pixel_format;
    pixel_format = mtlFormatForGLInternalFormat(internalformat);

    if (pixel_format == MGLPixelFormatInvalid)
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
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_ENUM);
            return;
        }
    }
}

