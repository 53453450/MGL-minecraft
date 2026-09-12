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

#include "mgl_shader_resource.h"
#include "mgl_render.h"

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
    return mglRenderResourceMetalSlot(res != NULL, res ? res->binding : 0u, 0u,
                                      0u);
}

GLuint mglStageBufferResourceElementCount(int resourceType, const MGLShaderResource *res)
{
    return mglRenderStageBufferResourceElementCount(
        (uint32_t)resourceType, res != NULL, res ? res->ubo_array_size : 0u,
        res && res->ubo_members != NULL,
        res ? res->gl_array_size : 0);
}

GLuint mglClientBufferBindingForResourceElement(int resourceType,
                                                const MGLShaderResource *res,
                                                GLuint element)
{
    return mglRenderClientBufferBindingForResourceElement(
        (uint32_t)resourceType,
        mglClientBufferBindingForResource(resourceType, res), element,
        res ? (const uint32_t *)res->ubo_array_bindings : NULL,
        res ? res->ubo_array_size : 0u);
}

GLuint mglMetalResourceSlotForElement(const MGLShaderResource *res, GLuint element)
{
    return mglRenderResourceMetalSlot(res != NULL, res ? res->binding : 0u,
                                      element, 0u);
}

GLuint mglMetalCombinedSamplerSlot(const MGLShaderResource *res)
{
    return mglRenderCombinedSamplerSlot(res != NULL,
                                        res && res->has_combined_sampler,
                                        res ? res->combined_sampler_binding : 0u);
}

GLuint mglMetalCombinedSamplerSlotForElement(const MGLShaderResource *res,
                                             GLuint element)
{
    return mglRenderCombinedSamplerSlotForElement(
        res != NULL, res && res->has_combined_sampler,
        res ? res->combined_sampler_binding : 0u, element);
}

bool mglPlainUniformAllowsGlobalFallback(const MGLShaderResource *res)
{
    if (!res) {
        return true;
    }
    return mglRenderPlainUniformAllowsGlobalFallback(res->name) != 0;
}

const char *mglMGLShaderResourceTypeName(int type)
{
    return mglRenderShaderResourceTypeName((uint32_t)type);
}
