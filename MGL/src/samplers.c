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
 * samplers.c
 * MGL
 *
 */

#include <strings.h>
#include <stdio.h>
#include "glm_context.h"

bool setTexParmi(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLint *param);
bool setTexParamsi(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLint *params);
bool setTexParamsIiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLint *params);
bool setTexParamsIuiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLuint *params);
bool setTexParmf(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLfloat *param);
bool setTexParamsf(GLMContext ctx, TextureParameter *tex_params, GLenum pname, const GLfloat *params);
bool setParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint iparam, GLfloat fparam);


bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *iparam, GLfloat *fparam);

static void mglMarkSamplerParameterDirty(Sampler *sampler)
{
    if (sampler)
        sampler->dirty_bits |= DIRTY_SAMPLER_PARAM;
}

static void mglSamplerParameterUnhandled(GLMContext ctx)
{
    if (!ctx || ctx->state.error == GL_NO_ERROR)
        ERROR_RETURN(GL_INVALID_ENUM);
}

Sampler *newSampler(GLMContext ctx, GLuint sampler)
{
    Sampler *ptr;

    ptr = (Sampler *)malloc(sizeof(Sampler));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate sampler %u\n", sampler);
        return NULL;
    }

    bzero(ptr, sizeof(Sampler));

    ptr->name = sampler;

    float black_color[] = {0,0,0,0};

    ptr->params.depth_stencil_mode = GL_DEPTH_COMPONENT;
    ptr->params.base_level = 0;
    memcpy(ptr->params.border_color, black_color, 4 * sizeof(float));
    ptr->params.compare_func = GL_LEQUAL;
    ptr->params.compare_mode = GL_NONE;
    ptr->params.lod_bias = 0.0;
    ptr->params.min_filter = GL_NEAREST_MIPMAP_LINEAR;
    ptr->params.mag_filter = GL_LINEAR;
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

Sampler *getSampler(GLMContext ctx, GLuint sampler)
{
    Sampler *ptr;

    if (!ctx || sampler == 0)
        return NULL;

    ptr = (Sampler *)searchHashTable(&STATE(sampler_table), sampler);

    if (!ptr)
    {
        ptr = newSampler(ctx, sampler);

        insertHashElement(&STATE(sampler_table), sampler, ptr);
    }

    return ptr;
}

bool isSampler(GLMContext ctx, GLuint sampler)
{
    Sampler *ptr;

    if (!ctx || sampler == 0)
        return false;

    ptr = (Sampler *)searchHashTable(&STATE(sampler_table), sampler);

    if (ptr)
        return true;

    return false;
}

Sampler *findSampler(GLMContext ctx, GLuint sampler)
{
    Sampler *ptr;

    if (!ctx || sampler == 0)
        return NULL;

    ptr = (Sampler *)searchHashTable(&STATE(sampler_table), sampler);

    return ptr;
}

GLboolean mglIsSampler(GLMContext ctx, GLuint sampler)
{
    return isSampler(ctx, sampler);
}

void mglGenSamplers(GLMContext ctx, GLsizei count, GLuint *samplers)
{
    if (!ctx || count < 0 || !samplers)
    {
        if (ctx && count < 0)
            ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    while(count--)
    {
        *samplers++ = getNewName(&ctx->state.sampler_table);
    }
}

void mglBindSampler(GLMContext ctx, GLuint unit, GLuint sampler)
{
    Sampler *ptr;

    if (!ctx)
        return;

    // glBindSampler takes a zero-based texture unit index, not GL_TEXTURE0 + unit.
    if (unit >= STATE_VAR(max_combined_texture_image_units) || unit >= TEXTURE_UNITS)
    {
        ERROR_RETURN(GL_INVALID_INDEX);
        return;
    }

    if (sampler)
    {
        ptr = findSampler(ctx, sampler);

        if(ptr == NULL)
        {
            fprintf(stderr,
                    "MGL ERROR: mglBindSampler invalid sampler name unit=%u sampler=%u\n",
                    unit,
                    sampler);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }
    else
    {
        ptr = NULL;
    }
    
    ctx->state.texture_samplers[unit] = ptr;
    ctx->state.dirty_bits  |= DIRTY_SAMPLER;
}

void mglDeleteSamplers(GLMContext ctx, GLsizei count, const GLuint *samplers)
{
    if (!ctx || count < 0 || !samplers)
    {
        if (ctx && count < 0)
            ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglFlushPendingDraws(ctx);

    while(count--)
    {
        GLuint sampler;

        sampler = *samplers++;

        if (sampler == 0)
            continue;

        if (isSampler(ctx, sampler))
        {
            Sampler *ptr;

            ptr = findSampler(ctx, sampler);
            if (!ptr)
                continue;

            // remove any references to this sampler
            for(int i=0; i<TEXTURE_UNITS; i++)
            {
                if (ctx->state.texture_samplers[i] == ptr)
                {
                    ctx->state.texture_samplers[i] = NULL;
                }
            }

            deleteHashElement(&ctx->state.sampler_table, sampler);

            if (ptr->mtl_data)
            {
                ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data);
            }

            free(ptr);
        }
    }
}

void mglCreateSamplers(GLMContext ctx, GLsizei n, GLuint *samplers)
{
    if (!ctx || n < 0 || !samplers)
    {
        if (ctx && n < 0)
            ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglGenSamplers(ctx, n, samplers);

    while(n--)
    {
        GLuint name;

        name = *samplers++;

        if (!getSampler(ctx, name))
        {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
    }
}

void mglBindSamplers(GLMContext ctx, GLuint first, GLsizei count, const GLuint *samplers)
{
    if (!ctx || count < 0)
    {
        if (ctx && count < 0)
            ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    while(count--)
    {
        GLuint sampler = samplers ? *samplers++ : 0;
        mglBindSampler(ctx, first, sampler);
        first++;
    }
}

void mglSamplerParameterf(GLMContext ctx, GLuint sampler, GLenum pname, GLfloat param)
{
    Sampler *ptr = findSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);
    mglFlushPendingDraws(ctx);

    if (setParam(ctx, &ptr->params, pname, 0, param))
    {
        mglMarkSamplerParameterDirty(ptr);
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglSamplerParameterfv(GLMContext ctx, GLuint sampler, GLenum pname, const GLfloat *param)
{
    if (!param)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = findSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);
    mglFlushPendingDraws(ctx);

    if (setTexParamsf(ctx, &ptr->params, pname, param))
    {
        mglMarkSamplerParameterDirty(ptr);

        return;
    }

    if (setParam(ctx, &ptr->params, pname, 0, *param))
    {
        mglMarkSamplerParameterDirty(ptr);

        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglSamplerParameteri(GLMContext ctx, GLuint sampler, GLenum pname, GLint param)
{
    Sampler *ptr = getSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);
    mglFlushPendingDraws(ctx);

    if (setParam(ctx, &ptr->params, pname, param, 0.0f))
    {
        mglMarkSamplerParameterDirty(ptr);
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglSamplerParameteriv(GLMContext ctx, GLuint sampler, GLenum pname, const GLint *param)
{
    GLfloat fparam = 0.0;
    if (!param)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = getSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);
    mglFlushPendingDraws(ctx);

    if (setTexParamsi(ctx, &ptr->params, pname, param))
    {
        mglMarkSamplerParameterDirty(ptr);

        return;
    }

    if (setParam(ctx, &ptr->params, pname, *param, fparam))
    {
        mglMarkSamplerParameterDirty(ptr);
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglSamplerParameterIiv(GLMContext ctx, GLuint sampler, GLenum pname, const GLint *param)
{
    GLfloat fparam = 0.0;
    if (!param)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = getSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);
    mglFlushPendingDraws(ctx);

    if (setTexParamsIiv(ctx, &ptr->params, pname, param))
    {
        mglMarkSamplerParameterDirty(ptr);

        return;
    }

    if (setTexParamsi(ctx, &ptr->params, pname, param))
    {
        mglMarkSamplerParameterDirty(ptr);

        return;
    }

    if (setParam(ctx, &ptr->params, pname, *param, fparam))
    {
        mglMarkSamplerParameterDirty(ptr);
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglSamplerParameterIuiv(GLMContext ctx, GLuint sampler, GLenum pname, const GLuint *param)
{
    GLfloat fparam = 0.0;
    if (!param)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = getSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);
    mglFlushPendingDraws(ctx);

    if (setTexParamsIuiv(ctx, &ptr->params, pname, param))
    {
        mglMarkSamplerParameterDirty(ptr);

        return;
    }

    if (setTexParamsi(ctx, &ptr->params, pname, (GLint *)param))
    {
        mglMarkSamplerParameterDirty(ptr);

        return;
    }

    if (setParam(ctx, &ptr->params, pname, *param, fparam))
    {
        mglMarkSamplerParameterDirty(ptr);
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglGetSamplerParameterIiv(GLMContext ctx, GLuint sampler, GLenum pname, GLint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = findSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    if (pname == GL_TEXTURE_BORDER_COLOR)
    {
        for (int i = 0; i < 4; ++i)
            params[i] = ptr->params.border_color_i[i];
        return;
    }

    GLfloat fparam = 0.0f;
    if (getParam(ctx, &ptr->params, pname, params, &fparam))
    {
        if (fparam)
            *params = (GLint)fparam;
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglGetSamplerParameterIuiv(GLMContext ctx, GLuint sampler, GLenum pname, GLuint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = findSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    if (pname == GL_TEXTURE_BORDER_COLOR)
    {
        for (int i = 0; i < 4; ++i)
            params[i] = ptr->params.border_color_ui[i];
        return;
    }

    GLint iparam = 0;
    GLfloat fparam = 0.0f;
    if (getParam(ctx, &ptr->params, pname, &iparam, &fparam))
    {
        *params = fparam ? (GLuint)fparam : (GLuint)iparam;
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglGetSamplerParameterfv(GLMContext ctx, GLuint sampler, GLenum pname, GLfloat *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = findSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    GLint iparam;
    iparam = 0;

    if(getParam(ctx, &ptr->params, pname, &iparam, params))
    {
        if (iparam)
        {
            *params = (float)iparam;
        }
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglGetSamplerParameteriv(GLMContext ctx, GLuint sampler, GLenum pname, GLint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Sampler *ptr = findSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    GLfloat fparam;
    fparam = 0.0;

    if(getParam(ctx, &ptr->params, pname, params, &fparam))
    {
        if (fparam)
        {
            *params = (float)fparam;
        }
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}
