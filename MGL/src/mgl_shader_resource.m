/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_shader_resource.m
 * MGL
 *
 * Implementation of the Shader Resource Helper Subsystem.
 * See mgl_shader_resource.h for the API contract.
 */

#import "mgl_shader_resource.h"
#include "mgl_render.h"

#import <Foundation/Foundation.h>

#include <string.h>

GLuint mglClientBufferBindingForResource(int resourceType, const MGLShaderResource *res)
{
    if (!res) {
        return 0u;
    }

    return mglRenderClientBufferBindingForResource(
        (uint32_t)resourceType, res->name, res->uniform_location, res->location,
        res->gl_binding);
}

GLuint mglMetalResourceSlot(const MGLShaderResource *res)
{
    return res ? res->binding : 0u;
}

GLuint mglStageBufferResourceElementCount(int resourceType, const MGLShaderResource *res)
{
    if ((resourceType == _UNIFORM_BUFFER_RES ||
         resourceType == _STORAGE_BUFFER_RES) &&
        res &&
        res->ubo_array_size > 1u) {
        return res->ubo_array_size;
    }
    if (resourceType == _UNIFORM_CONSTANT_RES &&
        res &&
        res->ubo_members &&
        res->gl_array_size > 1) {
        return (GLuint)res->gl_array_size;
    }
    if (resourceType == _STORAGE_BUFFER_RES &&
        res &&
        res->gl_array_size > 1) {
        return (GLuint)res->gl_array_size;
    }

    return 1u;
}

GLuint mglClientBufferBindingForResourceElement(int resourceType,
                                                const MGLShaderResource *res,
                                                GLuint element)
{
    GLuint baseBinding = mglClientBufferBindingForResource(resourceType, res);

    if ((resourceType == _UNIFORM_BUFFER_RES ||
         resourceType == _STORAGE_BUFFER_RES) &&
        res &&
        res->ubo_array_bindings &&
        element < res->ubo_array_size) {
        return res->ubo_array_bindings[element];
    }

    return baseBinding + element;
}

GLuint mglMetalResourceSlotForElement(const MGLShaderResource *res, GLuint element)
{
    return mglMetalResourceSlot(res) + element;
}

GLuint mglMetalCombinedSamplerSlot(const MGLShaderResource *res)
{
    if (!res || !res->has_combined_sampler) {
        return 0u;
    }
    return res->combined_sampler_binding;
}

GLuint mglMetalCombinedSamplerSlotForElement(const MGLShaderResource *res,
                                             GLuint element)
{
    return mglMetalCombinedSamplerSlot(res) + element;
}

bool mglPlainUniformAllowsGlobalFallback(const MGLShaderResource *res)
{
    if (!res || !res->name) {
        return true;
    }

    /*
     * Mojang/Iris' newer item/entity programs use u_* plain uniforms with the
     * same numeric locations as the old ShaderInstance uniforms, but the slots
     * do not mean the same thing. Falling back from u_RegionOffset or
     * u_TexCoordShrink to TextureMat/ColorModulator corrupts first-person items
     * and can make inventory icons disappear.
     */
    if (!strcmp(res->name, "u_ProjectionMatrix") ||
        !strcmp(res->name, "u_ModelViewMatrix") ||
        !strcmp(res->name, "u_RegionOffset") ||
        !strcmp(res->name, "u_TexCoordShrink") ||
        !strcmp(res->name, "u_FogColor") ||
        !strcmp(res->name, "u_EnvironmentFog") ||
        !strcmp(res->name, "u_RenderFog")) {
        return false;
    }

    return true;
}

const char *mglMGLShaderResourceTypeName(int type)
{
    return mglRenderShaderResourceTypeName((uint32_t)type);
}
