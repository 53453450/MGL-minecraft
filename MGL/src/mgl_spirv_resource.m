/*
 * mgl_spirv_resource.m
 * MGL
 *
 * Implementation of the SPIR-V Resource Helper Subsystem.
 * See mgl_spirv_resource.h for the API contract.
 */

#import "mgl_spirv_resource.h"

#import <Foundation/Foundation.h>
#import "spirv_cross_c.h"

#include <string.h>

GLuint mglClientBufferBindingForResource(int resourceType, const SpirvResource *res)
{
    if (!res) {
        return 0u;
    }

    GLint knownPlainUniformBinding = -1;
    if (res->name) {
        if (!strcmp(res->name, "ModelViewMat")) {
            knownPlainUniformBinding = 0;
        } else if (!strcmp(res->name, "ProjMat")) {
            knownPlainUniformBinding = 1;
        } else if (!strcmp(res->name, "TextureMat")) {
            knownPlainUniformBinding = 2;
        } else if (!strcmp(res->name, "ColorModulator")) {
            knownPlainUniformBinding = 3;
        } else if (!strcmp(res->name, "FogStart")) {
            knownPlainUniformBinding = 4;
        } else if (!strcmp(res->name, "FogEnd")) {
            knownPlainUniformBinding = 5;
        } else if (!strcmp(res->name, "FogColor")) {
            knownPlainUniformBinding = 6;
        } else if (!strcmp(res->name, "FogShape")) {
            knownPlainUniformBinding = 7;
        } else if (!strcmp(res->name, "GameTime")) {
            knownPlainUniformBinding = 8;
        } else if (!strcmp(res->name, "ScreenSize")) {
            knownPlainUniformBinding = 9;
        } else if (!strcmp(res->name, "LineWidth")) {
            knownPlainUniformBinding = 10;
        } else if (!strcmp(res->name, "IViewRotMat")) {
            knownPlainUniformBinding = 11;
        } else if (!strcmp(res->name, "ChunkOffset")) {
            knownPlainUniformBinding = 12;
        } else if (!strcmp(res->name, "u_ProjectionMatrix")) {
            knownPlainUniformBinding = 0;
        } else if (!strcmp(res->name, "u_ModelViewMatrix")) {
            knownPlainUniformBinding = 1;
        } else if (!strcmp(res->name, "u_RegionOffset")) {
            knownPlainUniformBinding = 2;
        } else if (!strcmp(res->name, "u_TexCoordShrink")) {
            knownPlainUniformBinding = 3;
        } else if (!strcmp(res->name, "u_FogColor")) {
            knownPlainUniformBinding = 4;
        } else if (!strcmp(res->name, "u_EnvironmentFog")) {
            knownPlainUniformBinding = 5;
        } else if (!strcmp(res->name, "u_RenderFog")) {
            knownPlainUniformBinding = 6;
        /* 1.21.11 new plain uniforms */
        } else if (!strcmp(res->name, "CameraBlockPos")) {
            knownPlainUniformBinding = 13;
        } else if (!strcmp(res->name, "CameraOffset")) {
            knownPlainUniformBinding = 14;
        } else if (!strcmp(res->name, "UseRgss")) {
            knownPlainUniformBinding = 15;
        } else if (!strcmp(res->name, "ChunkVisibility")) {
            knownPlainUniformBinding = 16;
        }
    }

    /*
     * Plain uniforms are represented internally as one tiny GL buffer per
     * uniform location. SPIRV-Cross usually reports descriptor binding 0 for
     * all of them, while the generated MSL assigns distinct [[buffer(n)]]
     * slots. Use the GL uniform location to find the client-side buffer, then
     * map that location to the reflected Metal slot later.
     */
    if (resourceType == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) {
        if (knownPlainUniformBinding >= 0) {
            return (GLuint)knownPlainUniformBinding;
        }
        if (res->uniform_location >= 0 && res->uniform_location < MAX_BINDABLE_BUFFERS) {
            return (GLuint)res->uniform_location;
        }
        if (res->location < MAX_BINDABLE_BUFFERS) {
            return res->location;
        }
        if (res->gl_binding < MAX_BINDABLE_BUFFERS) {
            return res->gl_binding;
        }
    }

    return res->gl_binding;
}

GLuint mglMetalResourceSlot(const SpirvResource *res)
{
    return res ? res->binding : 0u;
}

GLuint mglStageBufferResourceElementCount(int resourceType, const SpirvResource *res)
{
    if (resourceType == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER &&
        res &&
        res->ubo_array_size > 1u) {
        return res->ubo_array_size;
    }
    if (resourceType == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
        res &&
        res->ubo_members &&
        res->gl_array_size > 1) {
        return (GLuint)res->gl_array_size;
    }
    if (resourceType == SPVC_RESOURCE_TYPE_STORAGE_BUFFER &&
        res &&
        res->gl_array_size > 1) {
        return (GLuint)res->gl_array_size;
    }

    return 1u;
}

GLuint mglClientBufferBindingForResourceElement(int resourceType,
                                                const SpirvResource *res,
                                                GLuint element)
{
    GLuint baseBinding = mglClientBufferBindingForResource(resourceType, res);

    if (resourceType == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER &&
        res &&
        res->ubo_array_bindings &&
        element < res->ubo_array_size) {
        return res->ubo_array_bindings[element];
    }

    return baseBinding + element;
}

GLuint mglMetalResourceSlotForElement(const SpirvResource *res, GLuint element)
{
    return mglMetalResourceSlot(res) + element;
}

GLuint mglMetalCombinedSamplerSlot(const SpirvResource *res)
{
    if (!res || !res->msl_has_combined_sampler) {
        return 0u;
    }
    return res->msl_combined_sampler_binding;
}

GLuint mglMetalCombinedSamplerSlotForElement(const SpirvResource *res,
                                             GLuint element)
{
    return mglMetalCombinedSamplerSlot(res) + element;
}

bool mglPlainUniformAllowsGlobalFallback(const SpirvResource *res)
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

const char *mglSpirvResourceTypeName(int type)
{
    switch (type) {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER: return "uniform_buffer";
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT: return "uniform_constant";
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER: return "storage_buffer";
        case SPVC_RESOURCE_TYPE_STAGE_INPUT: return "stage_input";
        case SPVC_RESOURCE_TYPE_STAGE_OUTPUT: return "stage_output";
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE: return "sampled_image";
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE: return "separate_image";
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS: return "separate_sampler";
        case SPVC_RESOURCE_TYPE_PUSH_CONSTANT: return "push_constant";
        default: return "resource";
    }
}
