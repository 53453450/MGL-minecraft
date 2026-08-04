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
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include "glm_context.h"
#include "mgl_metal_ref.h"

#include "mgl_trace_log.h"

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

static bool mglSamplerParameterValuesEqual(const TextureParameter *a,
                                           const TextureParameter *b)
{
    return a && b &&
           memcmp(a, b, offsetof(TextureParameter, mtl_data)) == 0;
}

static void mglCommitSamplerParameter(GLMContext ctx,
                                      Sampler *sampler,
                                      GLenum pname,
                                      const TextureParameter *candidate)
{
    if (!sampler || !candidate ||
        mglSamplerParameterValuesEqual(&sampler->params, candidate)) {
        return;
    }

    if (!mglSamplerSnapshotCanDeferParameter(ctx, pname)) {
        mglFlushPendingDraws(ctx);
    }

    void *metal_sampler = sampler->params.mtl_data;
    sampler->params = *candidate;
    sampler->params.mtl_data = metal_sampler;
    mglMarkSamplerParameterDirty(sampler);
}

static void mglSamplerParameterUnhandled(GLMContext ctx)
{
    if (!ctx || ctx->state.error == GL_NO_ERROR)
        ERROR_RETURN(GL_INVALID_ENUM);
}
/* GL 4.6 spec: GL_TEXTURE_SWIZZLE_* are texture-object state, not sampler
 * state.  Sampler objects must reject these pnames with GL_INVALID_ENUM.
 * Without this guard, setParam → setTexParmi would accept them and silently
 * write into the sampler's dead swizzle fields. */
static GLboolean mglIsTextureOnlyParameter(GLenum pname)
{
    switch (pname) {
        case GL_TEXTURE_SWIZZLE_R:
        case GL_TEXTURE_SWIZZLE_G:
        case GL_TEXTURE_SWIZZLE_B:
        case GL_TEXTURE_SWIZZLE_A:
        case GL_TEXTURE_SWIZZLE_RGBA:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
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
    /* GL 4.6 spec §8.14: the initial MIN_FILTER/MAG_FILTER are both NEAREST.
     * A prior change defaulted these to NEAREST_MIPMAP_LINEAR/LINEAR, which
     * made samplers that never explicitly set MIN_FILTER enable mip filtering
     * — sampling a non-mipmapped texture (e.g. Minecraft's 256x256 block
     * atlas, which has mipmapLevelCount==1) with mipFilter=Linear reads
     * uninitialized/zero mip levels and produces horizontal/vertical stripes.
     * Restore the spec-correct NEAREST defaults. */
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
        GLuint name = getNewName(&ctx->state.sampler_table);
        if (!getSampler(ctx, name))
            return;
        *samplers++ = name;
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

    if (ctx->state.texture_samplers[unit] == ptr) {
        return;
    }

    mglTraceLogExternal("BIND_SAMPLER unit=%u sampler=%u resolved=%p minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x wrapR=0x%x minLod=%.3f maxLod=%.3f maxLevel=%d aniso=%.1f compareMode=0x%x compareFunc=0x%x",
                        (unsigned)unit,
                        (unsigned)sampler,
                        (void *)ptr,
                        ptr ? (unsigned)ptr->params.min_filter : 0u,
                        ptr ? (unsigned)ptr->params.mag_filter : 0u,
                        ptr ? (unsigned)ptr->params.wrap_s : 0u,
                        ptr ? (unsigned)ptr->params.wrap_t : 0u,
                        ptr ? (unsigned)ptr->params.wrap_r : 0u,
                        ptr ? (double)ptr->params.min_lod : 0.0,
                        ptr ? (double)ptr->params.max_lod : 0.0,
                        ptr ? (int)ptr->params.max_level : 0,
                        ptr ? (double)ptr->params.max_anisotropy : 1.0,
                        ptr ? (unsigned)ptr->params.compare_mode : 0u,
                        ptr ? (unsigned)ptr->params.compare_func : 0u);

    ctx->state.texture_samplers[unit] = ptr;
    mglMarkStateDirtyBits(&ctx->state, DIRTY_SAMPLER);
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
            GLboolean cleared_binding = GL_FALSE;
            for(int i=0; i<TEXTURE_UNITS; i++)
            {
                if (ctx->state.texture_samplers[i] == ptr)
                {
                    ctx->state.texture_samplers[i] = NULL;
                    cleared_binding = GL_TRUE;
                }
            }
            if (cleared_binding)
                mglMarkStateDirtyBits(&ctx->state, DIRTY_SAMPLER);

            deleteHashElement(&ctx->state.sampler_table, sampler);

            mglSafeReleaseMetalObj((void **)&ptr->mtl_data);

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
    if (!ctx) {
        return;
    }

    if (count < 0 ||
        first > STATE_VAR(max_combined_texture_image_units) ||
        (GLuint)count > STATE_VAR(max_combined_texture_image_units) - first ||
        first > TEXTURE_UNITS ||
        (GLuint)count > TEXTURE_UNITS - first) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    for (GLsizei i = 0; i < count; i++)
    {
        GLuint sampler = samplers ? samplers[i] : 0u;
        if (sampler && !isSampler(ctx, sampler)) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            continue;
        }
        mglBindSampler(ctx, first + (GLuint)i, sampler);
    }
}

void mglSamplerParameterf(GLMContext ctx, GLuint sampler, GLenum pname, GLfloat param)
{
    Sampler *ptr = findSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    if (mglIsTextureOnlyParameter(pname)) {
        mglSamplerParameterUnhandled(ctx);
        return;
    }

    mglTraceLogExternal("SAMPLER_PARAM_F sampler=%u pname=0x%x fparam=%.6f",
                        (unsigned)sampler,
                        (unsigned)pname,
                        (double)param);

    TextureParameter candidate = ptr->params;
    if (setParam(ctx, &candidate, pname, 0, param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
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
    TextureParameter candidate = ptr->params;

    if (mglIsTextureOnlyParameter(pname)) {
        mglSamplerParameterUnhandled(ctx);
        return;
    }

    if (setTexParamsf(ctx, &candidate, pname, param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
        return;
    }

    if (setParam(ctx, &candidate, pname, 0, *param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
        return;
    }

    mglSamplerParameterUnhandled(ctx);
}

void mglSamplerParameteri(GLMContext ctx, GLuint sampler, GLenum pname, GLint param)
{
    Sampler *ptr = getSampler(ctx, sampler);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    if (mglIsTextureOnlyParameter(pname)) {
        mglSamplerParameterUnhandled(ctx);
        return;
    }

    mglTraceLogExternal("SAMPLER_PARAM sampler=%u pname=0x%x iparam=%d",
                        (unsigned)sampler,
                        (unsigned)pname,
                        (int)param);

    TextureParameter candidate = ptr->params;
    if (setParam(ctx, &candidate, pname, param, 0.0f))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
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
    TextureParameter candidate = ptr->params;

    if (mglIsTextureOnlyParameter(pname)) {
        mglSamplerParameterUnhandled(ctx);
        return;
    }

    if (setTexParamsi(ctx, &candidate, pname, param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
        return;
    }

    if (setParam(ctx, &candidate, pname, *param, fparam))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
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
    TextureParameter candidate = ptr->params;

    if (mglIsTextureOnlyParameter(pname)) {
        mglSamplerParameterUnhandled(ctx);
        return;
    }

    if (setTexParamsIiv(ctx, &candidate, pname, param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
        return;
    }

    if (setTexParamsi(ctx, &candidate, pname, param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
        return;
    }

    if (setParam(ctx, &candidate, pname, *param, fparam))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
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
    TextureParameter candidate = ptr->params;

    if (mglIsTextureOnlyParameter(pname)) {
        mglSamplerParameterUnhandled(ctx);
        return;
    }

    if (setTexParamsIuiv(ctx, &candidate, pname, param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
        return;
    }

    if (setTexParamsi(ctx, &candidate, pname, (GLint *)param))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
        return;
    }

    if (setParam(ctx, &candidate, pname, *param, fparam))
    {
        mglCommitSamplerParameter(ctx, ptr, pname, &candidate);
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

    if (pname == GL_TEXTURE_BORDER_COLOR)
    {
        for (int i = 0; i < 4; ++i)
            params[i] = ptr->params.border_color[i];
        return;
    }

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

    if (pname == GL_TEXTURE_BORDER_COLOR)
    {
        for (int i = 0; i < 4; ++i)
            params[i] = (GLint)ptr->params.border_color[i];
        return;
    }

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
