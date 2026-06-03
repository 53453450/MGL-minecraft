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
 * tex_params.c
 * MGL
 *
 */

#include "glm_context.h"
#include "pixel_utils.h"

extern GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
extern Texture *currentTexture(GLMContext ctx, GLuint index);
Texture *getTex(GLMContext ctx, GLuint name, GLenum target);

static void mglMarkTextureParameterDirty(GLMContext ctx, Texture *tex)
{
    if (!tex) {
        return;
    }

    tex->dirty_bits |= DIRTY_TEXTURE_PARAM;
    if (ctx) {
        STATE(dirty_bits) |= DIRTY_TEX_PARAM;
    }
}

static GLint mglTexLevelCanonicalInternalFormat(GLint internalformat)
{
    switch (internalformat)
    {
        case GL_RED: return GL_R8;
        case GL_RG: return GL_RG8;
        case GL_RGB: return GL_RGB8;
        case GL_RGBA: return GL_RGBA8;
        case GL_SRGB: return GL_SRGB8;
        case GL_SRGB_ALPHA: return GL_SRGB8_ALPHA8;
        case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT24;
        case GL_DEPTH_STENCIL: return GL_DEPTH24_STENCIL8;
        default: return internalformat;
    }
}

static bool mglTexLevelInternalFormatCompressed(GLint internalformat)
{
    switch (mglTexLevelCanonicalInternalFormat(internalformat))
    {
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
        case GL_COMPRESSED_RGB8_ETC2:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_RGBA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
        case GL_COMPRESSED_R11_EAC:
        case GL_COMPRESSED_SIGNED_R11_EAC:
        case GL_COMPRESSED_RG11_EAC:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
            return true;
        default:
            return false;
    }
}

static bool mglTexLevelInternalFormatSignedInteger(GLint internalformat)
{
    switch (mglTexLevelCanonicalInternalFormat(internalformat))
    {
        case GL_R8I:
        case GL_R16I:
        case GL_R32I:
        case GL_RG8I:
        case GL_RG16I:
        case GL_RG32I:
        case GL_RGB8I:
        case GL_RGB16I:
        case GL_RGB32I:
        case GL_RGBA8I:
        case GL_RGBA16I:
        case GL_RGBA32I:
            return true;
        default:
            return false;
    }
}

static bool mglTexLevelInternalFormatUnsignedInteger(GLint internalformat)
{
    switch (mglTexLevelCanonicalInternalFormat(internalformat))
    {
        case GL_R8UI:
        case GL_R16UI:
        case GL_R32UI:
        case GL_RG8UI:
        case GL_RG16UI:
        case GL_RG32UI:
        case GL_RGB8UI:
        case GL_RGB16UI:
        case GL_RGB32UI:
        case GL_RGBA8UI:
        case GL_RGBA16UI:
        case GL_RGBA32UI:
        case GL_RGB10_A2UI:
            return true;
        default:
            return false;
    }
}

static bool mglTexLevelInternalFormatFloat(GLint internalformat)
{
    switch (mglTexLevelCanonicalInternalFormat(internalformat))
    {
        case GL_R16F:
        case GL_R32F:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH32F_STENCIL8:
            return true;
        default:
            return false;
    }
}

static bool mglTexLevelInternalFormatSignedNormalized(GLint internalformat)
{
    switch (mglTexLevelCanonicalInternalFormat(internalformat))
    {
        case GL_R8_SNORM:
        case GL_R16_SNORM:
        case GL_RG8_SNORM:
        case GL_RG16_SNORM:
        case GL_RGB8_SNORM:
        case GL_RGB16_SNORM:
        case GL_RGBA8_SNORM:
        case GL_RGBA16_SNORM:
            return true;
        default:
            return false;
    }
}

static GLint mglTexLevelComponentBits(GLint internalformat, GLenum pname)
{
    GLenum component = GL_NONE;
    GLint canonical = mglTexLevelCanonicalInternalFormat(internalformat);

    switch (pname)
    {
        case GL_TEXTURE_RED_SIZE: component = GL_RED; break;
        case GL_TEXTURE_GREEN_SIZE: component = GL_GREEN; break;
        case GL_TEXTURE_BLUE_SIZE: component = GL_BLUE; break;
        case GL_TEXTURE_ALPHA_SIZE: component = GL_ALPHA; break;
        case GL_TEXTURE_DEPTH_SIZE: component = GL_DEPTH; break;
        case GL_TEXTURE_STENCIL_SIZE: component = GL_STENCIL; break;
        default: return 0;
    }

    if (mglTexLevelInternalFormatCompressed(canonical))
        return 0;
    return (GLint)bitcountForInternalFormat(canonical, component);
}

static GLint mglTexLevelComponentType(GLint internalformat, GLenum pname)
{
    GLenum component = GL_NONE;
    GLint canonical = mglTexLevelCanonicalInternalFormat(internalformat);
    GLenum sizePname = GL_NONE;

    switch (pname)
    {
        case GL_TEXTURE_RED_TYPE: component = GL_RED; sizePname = GL_TEXTURE_RED_SIZE; break;
        case GL_TEXTURE_GREEN_TYPE: component = GL_GREEN; sizePname = GL_TEXTURE_GREEN_SIZE; break;
        case GL_TEXTURE_BLUE_TYPE: component = GL_BLUE; sizePname = GL_TEXTURE_BLUE_SIZE; break;
        case GL_TEXTURE_ALPHA_TYPE: component = GL_ALPHA; sizePname = GL_TEXTURE_ALPHA_SIZE; break;
        case GL_TEXTURE_DEPTH_TYPE: component = GL_DEPTH; sizePname = GL_TEXTURE_DEPTH_SIZE; break;
        default: return GL_NONE;
    }

    if (mglTexLevelComponentBits(canonical, sizePname) == 0)
        return GL_NONE;
    if (component == GL_STENCIL)
        return GL_UNSIGNED_INT;
    if (mglTexLevelInternalFormatSignedInteger(canonical))
        return GL_INT;
    if (mglTexLevelInternalFormatUnsignedInteger(canonical))
        return GL_UNSIGNED_INT;
    if (mglTexLevelInternalFormatFloat(canonical))
        return GL_FLOAT;
    if (mglTexLevelInternalFormatSignedNormalized(canonical))
        return GL_SIGNED_NORMALIZED;
    return GL_UNSIGNED_NORMALIZED;
}

static bool mglTexParameterTargetValid(GLenum target)
{
    switch (target)
    {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_BUFFER:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
        default:
            return false;
    }
}

static Texture *mglCurrentTextureForParameter(GLMContext ctx, GLenum target)
{
    if (!mglTexParameterTargetValid(target))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return NULL;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex && (!ctx || ctx->state.error == GL_NO_ERROR))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    return tex;
}

static Texture *mglNamedTextureForParameter(GLMContext ctx, GLuint texture)
{
    if (texture == 0)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return NULL;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex && (!ctx || ctx->state.error == GL_NO_ERROR))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    return tex;
}

static bool mglValidTextureSwizzle(GLint value)
{
    switch (value)
    {
        case GL_RED:
        case GL_GREEN:
        case GL_BLUE:
        case GL_ALPHA:
        case GL_ZERO:
        case GL_ONE:
            return true;
        default:
            return false;
    }
}

static bool mglValidTextureWrapMode(GLint value)
{
    switch (value)
    {
        case GL_CLAMP_TO_EDGE:
        case GL_CLAMP_TO_BORDER:
        case GL_MIRRORED_REPEAT:
        case GL_REPEAT:
        case GL_MIRROR_CLAMP_TO_EDGE:
            return true;
        default:
            return false;
    }
}

static bool mglTexParameterError(GLMContext ctx, GLenum error)
{
    ERROR_RETURN(error);
    return false;
}

static void mglTexParameterUnhandled(GLMContext ctx)
{
    if (!ctx || ctx->state.error == GL_NO_ERROR)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
    }
}

#pragma mark set params
bool setTexParmi(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLint *param)
{
    switch(pname)
    {
        case GL_DEPTH_STENCIL_TEXTURE_MODE:
            switch(*param)
            {
                case GL_DEPTH_COMPONENT:
                case GL_STENCIL_INDEX:
                    tex_params->depth_stencil_mode = *param;
                    break;

                default:
                    return mglTexParameterError(ctx, GL_INVALID_ENUM);
            }
            break;

        case GL_TEXTURE_BASE_LEVEL:
            if (*param < 0)
                return mglTexParameterError(ctx, GL_INVALID_VALUE);
            tex_params->base_level = *param;
            // need to compare this against something...
            break;

        case GL_TEXTURE_COMPARE_FUNC:
            switch(*param)
            {
                case GL_LEQUAL:
                case GL_GEQUAL:
                case GL_LESS:
                case GL_GREATER:
                case GL_EQUAL:
                case GL_NOTEQUAL:
                case GL_ALWAYS:
                case GL_NEVER:
                    tex_params->compare_func = *param;
                    break;

                default:
                    return mglTexParameterError(ctx, GL_INVALID_ENUM);
            }
            break;

        case GL_TEXTURE_COMPARE_MODE:
            switch(*param)
            {
                case GL_COMPARE_REF_TO_TEXTURE:
                case GL_NONE:
                    tex_params->compare_mode = *param;
                    break;

                default:
                    return mglTexParameterError(ctx, GL_INVALID_ENUM);
            }
            break;

        case GL_TEXTURE_MIN_FILTER:
            switch(*param)
            {
                case GL_NEAREST:
                case GL_LINEAR:
                case GL_NEAREST_MIPMAP_NEAREST:
                case GL_LINEAR_MIPMAP_NEAREST:
                case GL_NEAREST_MIPMAP_LINEAR:
                case GL_LINEAR_MIPMAP_LINEAR:
                    tex_params->min_filter = *param;
                    break;

                default:
                    return mglTexParameterError(ctx, GL_INVALID_ENUM);
            }
            break;

        case GL_TEXTURE_MAG_FILTER:
            switch(*param)
            {
                case GL_NEAREST:
                case GL_LINEAR:
                    tex_params->mag_filter = *param;
                    break;

                default:
                    return mglTexParameterError(ctx, GL_INVALID_ENUM);
            }
            break;

        case GL_TEXTURE_MIN_LOD:
            tex_params->min_lod = *param;
            break;

        case GL_TEXTURE_MAX_LOD:
            tex_params->max_lod = *param;
            break;

        case GL_TEXTURE_MAX_LEVEL:
            if (*param < 0)
                return mglTexParameterError(ctx, GL_INVALID_VALUE);
            tex_params->max_level = *param;
            break;

        case GL_TEXTURE_SWIZZLE_R:
            if (!mglValidTextureSwizzle(*param))
                return mglTexParameterError(ctx, GL_INVALID_ENUM);
            tex_params->swizzle_r = *param;
            break;

        case GL_TEXTURE_SWIZZLE_G:
            if (!mglValidTextureSwizzle(*param))
                return mglTexParameterError(ctx, GL_INVALID_ENUM);
            tex_params->swizzle_g = *param;
            break;

        case GL_TEXTURE_SWIZZLE_B:
            if (!mglValidTextureSwizzle(*param))
                return mglTexParameterError(ctx, GL_INVALID_ENUM);
            tex_params->swizzle_b = *param;
            break;

        case GL_TEXTURE_SWIZZLE_A:
            if (!mglValidTextureSwizzle(*param))
                return mglTexParameterError(ctx, GL_INVALID_ENUM);
            tex_params->swizzle_a = *param;
            break;

        case GL_TEXTURE_WRAP_S:
            if (!mglValidTextureWrapMode(*param))
                return mglTexParameterError(ctx, GL_INVALID_ENUM);
            tex_params->wrap_s = *param;
            break;

        case GL_TEXTURE_WRAP_T:
            if (!mglValidTextureWrapMode(*param))
                return mglTexParameterError(ctx, GL_INVALID_ENUM);
            tex_params->wrap_t = *param;
            break;

        case GL_TEXTURE_WRAP_R:
            if (!mglValidTextureWrapMode(*param))
                return mglTexParameterError(ctx, GL_INVALID_ENUM);
            tex_params->wrap_r = *param;
            break;

        default:
            return false;
            break;
    }

    return true;
}

bool setTexParamsi(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLint *params)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                tex_params->border_color[i] = (GLint)params[i];
            break;

        case GL_TEXTURE_SWIZZLE_RGBA:
            for (int i = 0; i < 4; i++)
            {
                if (!mglValidTextureSwizzle(params[i]))
                    return mglTexParameterError(ctx, GL_INVALID_ENUM);
            }

         if ((params[0] != GL_RED) ||
             (params[1] != GL_GREEN) ||
             (params[2] != GL_BLUE) ||
             (params[3] != GL_ALPHA))
            {
                tex_params->swizzled = true;
                tex_params->swizzle_r = params[0];
                tex_params->swizzle_g = params[1];
                tex_params->swizzle_b = params[2];
                tex_params->swizzle_a = params[3];
            }
            else
            {
                tex_params->swizzled = false;
            }
            break;

        default:
            return false;
            break;
    }

    return true;
}

bool setTexParamsIiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLint *params)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                tex_params->border_color_i[i] = params[i];
            break;

        default:
            return false;
            break;
    }

    return true;
}

bool setTexParamsIuiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLuint *params)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                tex_params->border_color_ui[i] = params[i];
            break;

        default:
            return false;
            break;
    }

    return true;
}

bool setTexParmf(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLfloat *param)
{
    switch(pname)
    {
        case GL_TEXTURE_LOD_BIAS:
            tex_params->lod_bias = *param;
            break;

        case GL_TEXTURE_MAX_ANISOTROPY:
            if (*param < 1.0f)
                return mglTexParameterError(ctx, GL_INVALID_VALUE);
            tex_params->max_anisotropy = *param;
            break;

        case GL_TEXTURE_MIN_LOD:
            tex_params->min_lod = *param;
            break;

        case GL_TEXTURE_MAX_LOD:
            tex_params->max_lod = *param;
            break;

        default:
            return false;
            break;
    }

    return true;
}

bool setTexParamsf(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLfloat *params)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                tex_params->border_color[i] = params[i];
            break;

        case GL_TEXTURE_SWIZZLE_RGBA:
            for (int i = 0; i < 4; i++)
            {
                if (!mglValidTextureSwizzle((GLint)params[i]))
                    return mglTexParameterError(ctx, GL_INVALID_ENUM);
            }
            tex_params->swizzle_r = (GLint)params[0];
            tex_params->swizzle_g = (GLint)params[1];
            tex_params->swizzle_b = (GLint)params[2];
            tex_params->swizzle_a = (GLint)params[3];
            break;

        default:
            return false;
            break;
    }

    return true;
}

#pragma mark get params
static bool getTexParmi(GLMContext ctx, TextureParameter *tex_params, const GLenum pname, GLint *ret)
{
    switch(pname)
    {
        case GL_DEPTH_STENCIL_TEXTURE_MODE:
            *ret = tex_params->depth_stencil_mode;
            break;

        case GL_TEXTURE_BASE_LEVEL:
            *ret = tex_params->base_level;
            // need to compare this against something...
            break;

        case GL_TEXTURE_COMPARE_FUNC:
            *ret = tex_params->compare_func;
                break;
            break;

        case GL_TEXTURE_COMPARE_MODE:
            *ret = tex_params->compare_mode;
            break;

        case GL_TEXTURE_MIN_FILTER:
            *ret = tex_params->min_filter;
            break;

        case GL_TEXTURE_MAG_FILTER:
            *ret = tex_params->mag_filter;
            break;

        case GL_TEXTURE_MIN_LOD:
            *ret = tex_params->min_lod;
            break;

        case GL_TEXTURE_MAX_LOD:
            *ret = tex_params->max_lod;
            break;

        case GL_TEXTURE_MAX_LEVEL:
            *ret = tex_params->max_level;
            break;

        case GL_TEXTURE_SWIZZLE_R:
            *ret = tex_params->swizzle_r;
            break;

        case GL_TEXTURE_SWIZZLE_G:
            *ret = tex_params->swizzle_g;
            break;

        case GL_TEXTURE_SWIZZLE_B:
            *ret = tex_params->swizzle_b;
            break;

        case GL_TEXTURE_SWIZZLE_A:
            *ret = tex_params->swizzle_a;
            break;

        case GL_TEXTURE_WRAP_S:
            *ret = tex_params->wrap_s;
            break;

        case GL_TEXTURE_WRAP_T:
            *ret = tex_params->wrap_t;
            break;

        case GL_TEXTURE_WRAP_R:
            *ret = tex_params->wrap_r;
            break;

        default:
            return false;
            break;
    }

    return true;
}

#if 0
static bool getTexParamsi(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *ret)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                ret[i] = tex_params->border_color[i];
            break;

        case GL_TEXTURE_SWIZZLE_RGBA:
            *ret++ = tex_params->swizzle_r;
            *ret++ = tex_params->swizzle_g;
            *ret++ = tex_params->swizzle_b;
            *ret++ = tex_params->swizzle_a;
            break;

        default:
            return false;
            break;
    }

    return true;
}

static bool getTexParamsIiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *ret)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                ret[i] = tex_params->border_color_i[i];
            break;

        default:
            return false;
            break;
    }

    return true;
}

static bool getTexParamsIuiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLuint *ret)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                ret[i] = tex_params->border_color_ui[i];
            break;

        default:
            return false;
            break;
    }

    return true;
}

static bool getTexParamsf(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLfloat *ret)
{
    switch(pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for(int i=0; i<4; i++)
                ret[i] = tex_params->border_color[i];
            break;

        case GL_TEXTURE_SWIZZLE_RGBA:
            *ret++ = tex_params->swizzle_r;
            *ret++ = tex_params->swizzle_g;
            *ret++ = tex_params->swizzle_b;
            *ret++ = tex_params->swizzle_a;
            break;

        default:
            return false;
            break;
    }

    return true;
}
#endif

static bool getTexParmf(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLfloat *ret)
{
    switch(pname)
    {
        case GL_TEXTURE_LOD_BIAS:
            *ret = tex_params->lod_bias;
            break;

        case GL_TEXTURE_MAX_ANISOTROPY:
            *ret = tex_params->max_anisotropy;
            break;

        default:
            return false;
            break;
    }

    return true;
}

bool setParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint iparam, GLfloat fparam)
{
    if (iparam)
    {
        if (setTexParmi(ctx, tex_params, pname, &iparam))
            return true;

        fparam = (float)iparam;
        if (setTexParmf(ctx, tex_params, pname, &fparam))
            return true;
    }
    else
    {
        if (setTexParmf(ctx, tex_params, pname, &fparam))
            return true;

        iparam = (GLint)fparam;
        if (setTexParmi(ctx, tex_params, pname, &iparam))
            return true;
    }

    return false;
}

bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *iparam, GLfloat *fparam)
{
    if (iparam)
    {
        if (getTexParmi(ctx, tex_params, pname, iparam))
            return true;

        if (getTexParmf(ctx, tex_params, pname, fparam))
            return true;
    }
    else
    {
        if (getTexParmf(ctx, tex_params, pname, fparam))
            return true;

        if (getTexParmi(ctx, tex_params, pname, iparam))
            return true;
    }

    return false;
}

#pragma mark tex param gl calls
void mglTexParameterf(GLMContext ctx, GLenum target, GLenum pname, GLfloat param)
{
    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if (setParam(ctx, &tex->params, pname, 0, param))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTexParameterfv(GLMContext ctx, GLenum target, GLenum pname, const GLfloat *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

   // more than one param... try setTexParamsf
    if (setTexParamsf(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    if (setParam(ctx, &tex->params, pname, 0, *params))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTexParameteri(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{
    GLfloat fparam = 0.0;

    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if (setParam(ctx, &tex->params, pname, param, fparam))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTexParameteriv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{
    GLfloat fparam = 0.0;

    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    // more than one param... try setTexParamsi
    if (setTexParamsi(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    if (setParam(ctx, &tex->params, pname, *params, fparam))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if (setTexParamsIiv(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    // more than one param... try setTexParamsi
    if (setTexParamsi(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    GLfloat fparam = 0.0;
    if (setParam(ctx, &tex->params, pname, *params, fparam))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, const GLuint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if (setTexParamsIuiv(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    // more than one param... try setTexParamsi
    if (setTexParamsi(ctx, &tex->params, pname, (GLint *)params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    GLfloat fparam = 0.0;
    if (setParam(ctx, &tex->params, pname, *params, fparam))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTextureParameterf(GLMContext ctx, GLuint texture, GLenum pname, GLfloat param)
{
    Texture *tex = mglNamedTextureForParameter(ctx, texture);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if(setParam(ctx, &tex->params, pname, 0, param))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, const GLfloat *param)
{
    if (!param)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglNamedTextureForParameter(ctx, texture);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if(setTexParamsf(ctx, &tex->params, pname, param))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }
    if(setTexParmf(ctx, &tex->params, pname, param))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    if (setParam(ctx, &tex->params, pname, 0, *param))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTextureParameteri(GLMContext ctx, GLuint texture, GLenum pname, GLint param)
{
    Texture *tex = mglNamedTextureForParameter(ctx, texture);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if(setParam(ctx, &tex->params, pname, param, 0.0f))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, const GLint *param)
{
    if (!param)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglNamedTextureForParameter(ctx, texture);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if(setTexParamsi(ctx, &tex->params, pname, param))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }
    if(setTexParmi(ctx, &tex->params, pname, param))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    if (setParam(ctx, &tex->params, pname, *param, 0.0f))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, const GLint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglNamedTextureForParameter(ctx, texture);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if (setTexParamsIiv(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    // more than one param... try setTexParamsi
    if (setTexParamsi(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    GLfloat fparam = 0.0;
    if (setParam(ctx, &tex->params, pname, *params, fparam))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, const GLuint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglNamedTextureForParameter(ctx, texture);
    if (!tex)
        return;
    mglFlushPendingDraws(ctx);

    if (setTexParamsIuiv(ctx, &tex->params, pname, params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    // more than one param... try setTexParamsi
    if (setTexParamsi(ctx, &tex->params, pname, (GLint *)params))
    {
        mglMarkTextureParameterDirty(ctx, tex);

        return;
    }

    GLfloat fparam = 0.0;
    if (setParam(ctx, &tex->params, pname, *params, fparam))
    {
        mglMarkTextureParameterDirty(ctx, tex);
        return;
    }

    mglTexParameterUnhandled(ctx);
}

#pragma mark get tex param gl calls
void mglGetTexParameterfv(GLMContext ctx, GLenum target, GLenum pname, GLfloat *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;

    GLint iparam;
    iparam = 0;

    if(getParam(ctx, &tex->params, pname, &iparam, params))
    {
        if (iparam)
        {
            *params = (float)iparam;
        }
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglGetTexParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = mglCurrentTextureForParameter(ctx, target);
    if (!tex)
        return;

    GLfloat fparam;
    fparam = 0.0;

    if(getParam(ctx, &tex->params, pname, params, &fparam))
    {
        if (fparam)
        {
            *params = (float)fparam;
        }
        return;
    }

    mglTexParameterUnhandled(ctx);
}

void mglGetTexLevelParameteriv(GLMContext ctx, GLenum target, GLint level, GLenum pname, GLint *params);

void mglGetTexLevelParameterfv(GLMContext ctx, GLenum target, GLint level, GLenum pname, GLfloat *params)
{
    GLint value = 0;
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    mglGetTexLevelParameteriv(ctx, target, level, pname, &value);
    *params = (GLfloat)value;
}

void mglGetTexLevelParameteriv(GLMContext ctx, GLenum target, GLint level, GLenum pname, GLint *params)
{
    GLuint index;
    Texture *tex;
    GLint internalformat;

    fprintf(stderr, "MGL GetTexLevelParameter ENTER target=0x%x level=%d pname=0x%x\n",
            target, level, pname);

    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (level < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (target == GL_PROXY_TEXTURE_2D ||
        target == GL_PROXY_TEXTURE_1D ||
        target == GL_PROXY_TEXTURE_3D ||
        target == GL_PROXY_TEXTURE_RECTANGLE ||
        target == GL_PROXY_TEXTURE_1D_ARRAY ||
        target == GL_PROXY_TEXTURE_2D_ARRAY ||
        target == GL_PROXY_TEXTURE_CUBE_MAP ||
        target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY ||
        target == GL_PROXY_TEXTURE_2D_MULTISAMPLE ||
        target == GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY)
    {
        GLuint proxy_index = textureIndexFromTarget(ctx, target);
        ProxyTextureQueryState *proxy_state = NULL;
        GLint proxy_internal = 0;

        if (proxy_index >= _MAX_TEXTURE_TYPES) {
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
        }

        proxy_state = &STATE(proxy_texture_query[proxy_index]);
        proxy_internal = (level == 0) ? proxy_state->internalformat : 0;
        switch (pname)
        {
            case GL_TEXTURE_WIDTH:
                *params = (level == 0) ? proxy_state->width : 0;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_HEIGHT:
                *params = (level == 0) ? proxy_state->height : 0;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_DEPTH:
                *params = (level == 0) ? proxy_state->depth : 0;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_INTERNAL_FORMAT:
                *params = proxy_internal;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_RED_SIZE:
            case GL_TEXTURE_GREEN_SIZE:
            case GL_TEXTURE_BLUE_SIZE:
            case GL_TEXTURE_ALPHA_SIZE:
            case GL_TEXTURE_DEPTH_SIZE:
            case GL_TEXTURE_STENCIL_SIZE:
                *params = (proxy_internal != 0) ? mglTexLevelComponentBits(proxy_internal, pname) : 0;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_COMPRESSED:
                *params = mglTexLevelInternalFormatCompressed(proxy_internal) ? GL_TRUE : GL_FALSE;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
                *params = 0;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_RED_TYPE:
            case GL_TEXTURE_GREEN_TYPE:
            case GL_TEXTURE_BLUE_TYPE:
            case GL_TEXTURE_ALPHA_TYPE:
            case GL_TEXTURE_DEPTH_TYPE:
                *params = (proxy_internal != 0) ? mglTexLevelComponentType(proxy_internal, pname) : GL_NONE;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_SAMPLES:
                *params = 0;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
                *params = GL_TRUE;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            case GL_TEXTURE_SHARED_SIZE:
                *params = (proxy_internal == GL_RGB9_E5) ? 5 : 0;
                fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
                return;
            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }
    }

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    tex = currentTexture(ctx, index);
    if (!tex)
    {
        *params = 0;
        fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
        return;
    }

    if (level >= (GLint)tex->num_levels || !tex->faces[0].levels)
    {
        *params = 0;
        fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
        return;
    }

    internalformat = tex->internalformat;

    switch (pname)
    {
        case GL_TEXTURE_WIDTH:
            *params = tex->faces[0].levels[level].width;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_HEIGHT:
            *params = tex->faces[0].levels[level].height;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_DEPTH:
            *params = tex->faces[0].levels[level].depth;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_INTERNAL_FORMAT:
            *params = internalformat;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_RED_SIZE:
        case GL_TEXTURE_GREEN_SIZE:
        case GL_TEXTURE_BLUE_SIZE:
        case GL_TEXTURE_ALPHA_SIZE:
        case GL_TEXTURE_DEPTH_SIZE:
        case GL_TEXTURE_STENCIL_SIZE:
            *params = mglTexLevelComponentBits(internalformat, pname);
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_COMPRESSED:
            *params = mglTexLevelInternalFormatCompressed(internalformat) ? GL_TRUE : GL_FALSE;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
            *params = mglTexLevelInternalFormatCompressed(internalformat)
                ? (GLint)tex->faces[0].levels[level].data_size
                : 0;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_RED_TYPE:
        case GL_TEXTURE_GREEN_TYPE:
        case GL_TEXTURE_BLUE_TYPE:
        case GL_TEXTURE_ALPHA_TYPE:
        case GL_TEXTURE_DEPTH_TYPE:
            *params = mglTexLevelComponentType(internalformat, pname);
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_SAMPLES:
            *params = 0;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
            *params = GL_TRUE;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        case GL_TEXTURE_SHARED_SIZE:
            *params = (mglTexLevelCanonicalInternalFormat(internalformat) == GL_RGB9_E5) ? 5 : 0;
            fprintf(stderr, "MGL GetTexLevelParameter target=0x%x level=%d pname=0x%x -> %d\n", target, level, pname, *params);
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}
