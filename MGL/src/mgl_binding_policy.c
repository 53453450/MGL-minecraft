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
 * mgl_binding_policy.c — C1 / O3.3 strip from mgl_render.cpp.
 * Binding slot / sampler / stage / plain-uniform policy (CTS BindingState).
 * Pure C; resource-type and shader-stage use mgl_types_program numeric ABI.
 * Metal bind encode and +Binding.m ports remain thin — do not grow them.
 */

#include "mgl_binding_policy.h"

#include <stdint.h>
#include <string.h>

uint32_t mglRenderShaderResourceElementCount(uint32_t gl_array_size) {
    return gl_array_size > 1u ? gl_array_size : 1u;
}

int mglRenderImageUnitsInRange(uint32_t metal_slot, uint32_t gl_unit,
                               uint32_t max_units) {
    return metal_slot < max_units && gl_unit < max_units ? 1 : 0;
}

uint32_t mglRenderImageUnitFromResource(int explicit_by_slot,
                                        uint32_t explicit_unit,
                                        int32_t sampler_unit,
                                        uint32_t gl_binding, uint32_t element) {
    if (explicit_by_slot) {
        return explicit_unit;
    }
    uint32_t base = sampler_unit >= 0 ? (uint32_t)sampler_unit : gl_binding;
    return base + element;
}

int mglRenderComputeTextureBindKind(uint32_t spvc_type) {
    /* _STORAGE_IMAGE_RES=7, _SAMPLED_IMAGE_RES=8 */
    if (spvc_type == 8u) {
        return 0; /* sampled */
    }
    if (spvc_type == 7u) {
        return 1; /* storage */
    }
    return -1;
}

int mglRenderComputeTextureListExpandsByElement(uint32_t spvc_type) {
    return mglRenderComputeTextureBindKind(spvc_type) >= 0 ? 1 : 0;
}

int mglRenderComputeTextureBindIsStorage(uint32_t kind) {
    return kind == 1u ? 1 : 0;
}

int mglRenderComputeTextureBindNeedsSampler(uint32_t kind, int has_combined) {
    return kind == 0u && has_combined ? 1 : 0;
}

int mglRenderShaderResourceTypeIsSamplerImage(uint32_t res_type) {
    switch (res_type) {
    case 8u /* _SAMPLED_IMAGE_RES */:
    case 11u /* _SEPARATE_IMAGE_RES */:
    case 12u /* _SEPARATE_SAMPLERS_RES */:
    case 7u /* _STORAGE_IMAGE_RES */:
        return 1;
    default:
        return 0;
    }
}

const char *mglRenderShaderResourceTypeName(uint32_t res_type) {
    switch (res_type) {
    case 1u /* _UNIFORM_BUFFER_RES */:
        return "uniform_buffer";
    case 2u /* _UNIFORM_CONSTANT_RES */:
        return "uniform_constant";
    case 3u /* _STORAGE_BUFFER_RES */:
        return "storage_buffer";
    case 4u /* _STAGE_INPUT_RES */:
        return "stage_input";
    case 5u /* _STAGE_OUTPUT_RES */:
        return "stage_output";
    case 8u /* _SAMPLED_IMAGE_RES */:
        return "sampled_image";
    case 11u /* _SEPARATE_IMAGE_RES */:
        return "separate_image";
    case 12u /* _SEPARATE_SAMPLERS_RES */:
        return "separate_sampler";
    case 10u /* _PUSH_CONSTANT_RES */:
        return "push_constant";
    default:
        return "resource";
    }
}

int mglRenderPlainUniformBindingForName(const char *name) {
    if (!name) {
        return -1;
    }
    if (strcmp(name, "ModelViewMat") == 0) return 0;
    if (strcmp(name, "ProjMat") == 0) return 1;
    if (strcmp(name, "TextureMat") == 0) return 2;
    if (strcmp(name, "ColorModulator") == 0) return 3;
    if (strcmp(name, "FogStart") == 0) return 4;
    if (strcmp(name, "FogEnd") == 0) return 5;
    if (strcmp(name, "FogColor") == 0) return 6;
    if (strcmp(name, "FogShape") == 0) return 7;
    if (strcmp(name, "GameTime") == 0) return 8;
    if (strcmp(name, "ScreenSize") == 0) return 9;
    if (strcmp(name, "LineWidth") == 0) return 10;
    if (strcmp(name, "IViewRotMat") == 0) return 11;
    if (strcmp(name, "ChunkOffset") == 0) return 12;
    if (strcmp(name, "u_ProjectionMatrix") == 0) return 0;
    if (strcmp(name, "u_ModelViewMatrix") == 0) return 1;
    if (strcmp(name, "u_RegionOffset") == 0) return 2;
    if (strcmp(name, "u_TexCoordShrink") == 0) return 3;
    if (strcmp(name, "u_FogColor") == 0) return 4;
    if (strcmp(name, "u_EnvironmentFog") == 0) return 5;
    if (strcmp(name, "u_RenderFog") == 0) return 6;
    /* 1.21.11 new plain uniforms */
    if (strcmp(name, "CameraBlockPos") == 0) return 13;
    if (strcmp(name, "CameraOffset") == 0) return 14;
    if (strcmp(name, "UseRgss") == 0) return 15;
    if (strcmp(name, "ChunkVisibility") == 0) return 16;
    return -1;
}

uint32_t mglRenderClientBufferBindingForResource(uint32_t resource_type,
                                                 const char *name,
                                                 int32_t uniform_location,
                                                 uint32_t location,
                                                 uint32_t gl_binding) {
    if (resource_type == 2u /* _UNIFORM_CONSTANT_RES */) {
        int known = mglRenderPlainUniformBindingForName(name);
        if (known >= 0) {
            return (uint32_t)known;
        }
        if (uniform_location >= 0 &&
            (uint32_t)uniform_location < 84u /* MAX_BINDABLE_BUFFERS */) {
            return (uint32_t)uniform_location;
        }
        if (location < 84u /* MAX_BINDABLE_BUFFERS */) {
            return location;
        }
        if (gl_binding < 84u /* MAX_BINDABLE_BUFFERS */) {
            return gl_binding;
        }
    }
    return gl_binding;
}

int mglRenderPlainUniformAllowsGlobalFallback(const char *name) {
    if (!name) {
        return 1;
    }
    /*
     * Mojang/Iris' newer item/entity programs use u_* plain uniforms with the
     * same numeric locations as the old ShaderInstance uniforms, but the slots
     * do not mean the same thing. Falling back from u_RegionOffset or
     * u_TexCoordShrink to TextureMat/ColorModulator corrupts first-person items
     * and can make inventory icons disappear.
     */
    if (strcmp(name, "u_ProjectionMatrix") == 0 ||
        strcmp(name, "u_ModelViewMatrix") == 0 ||
        strcmp(name, "u_RegionOffset") == 0 ||
        strcmp(name, "u_TexCoordShrink") == 0 ||
        strcmp(name, "u_FogColor") == 0 ||
        strcmp(name, "u_EnvironmentFog") == 0 ||
        strcmp(name, "u_RenderFog") == 0) {
        return 0;
    }
    return 1;
}

uint32_t mglRenderStageBufferResourceElementCount(uint32_t resource_type,
                                                  int has_res,
                                                  uint32_t ubo_array_size,
                                                  int has_ubo_members,
                                                  int32_t gl_array_size) {
    if (!has_res) {
        return 1u;
    }
    if ((resource_type == 1u /* _UNIFORM_BUFFER_RES */ ||
         resource_type == 3u /* _STORAGE_BUFFER_RES */) &&
        ubo_array_size > 1u) {
        return ubo_array_size;
    }
    if (resource_type == 2u /* _UNIFORM_CONSTANT_RES */ && has_ubo_members &&
        gl_array_size > 1) {
        return (uint32_t)gl_array_size;
    }
    if (resource_type == 3u /* _STORAGE_BUFFER_RES */ && gl_array_size > 1) {
        return (uint32_t)gl_array_size;
    }
    return 1u;
}

uint32_t mglRenderClientBufferBindingForResourceElement(
    uint32_t resource_type, uint32_t base_binding, uint32_t element,
    const uint32_t *ubo_array_bindings, uint32_t ubo_array_size) {
    if ((resource_type == 1u /* _UNIFORM_BUFFER_RES */ ||
         resource_type == 3u /* _STORAGE_BUFFER_RES */) &&
        ubo_array_bindings && element < ubo_array_size) {
        return ubo_array_bindings[element];
    }
    return base_binding + element;
}

uint32_t mglRenderCombinedSamplerSlot(int has_res, int has_combined,
                                      uint32_t combined_binding) {
    return has_res && has_combined ? combined_binding : 0u;
}

uint32_t mglRenderCombinedSamplerSlotForElement(int has_res, int has_combined,
                                                uint32_t combined_binding,
                                                uint32_t element) {
    return mglRenderCombinedSamplerSlot(has_res, has_combined,
                                        combined_binding) +
           element;
}

int mglRenderResourceLooksSamplerLike(uint32_t res_type, uint32_t image_dim) {
    if (mglRenderShaderResourceTypeIsSamplerImage(res_type)) {
        return 1;
    }
    /* Plain-uniform resources are only sampler-like when the reflection gave
     * them a texture dim.  The SPIRV-era fallbacks that used to answer this --
     * a synthesized uniform location above MGL_SYNTHETIC_SAMPLER_LOCATION_BASE
     * and a "Sampler"/"CloudFaces" name heuristic -- are gone: sampler
     * declarations and the opaque leaves of plain uniform structs are
     * reflected as _SAMPLED_IMAGE_RES with image_dim set, and synthetic
     * locations are only ever assigned to sampled/storage image resources, so
     * neither fallback could fire for _UNIFORM_CONSTANT_RES. */
    if (res_type == 2u /* _UNIFORM_CONSTANT_RES */) {
        return image_dim != 0u ? 1 : 0;
    }
    return 0;
}

uint32_t mglRenderResourceMetalSlot(int has_resource, uint32_t binding,
                                    uint32_t element, uint32_t fallback) {
    return has_resource ? binding + element : fallback;
}

int mglRenderSamplerUnitValid(int32_t unit, uint32_t max_units) {
    return unit >= 0 && (uint32_t)unit < max_units ? 1 : 0;
}

int mglRenderShaderStageValid(int stage) {
    return stage >= 0 && stage < 6 /* _MAX_SHADER_TYPES */ ? 1 : 0;
}

int mglRenderStageMapsVertexAttribs(int stage) {
    return stage == 0 /* _VERTEX_SHADER */ ? 1 : 0;
}

int mglRenderVertexCaptureNeedsLoad(int stage, const void *bytes,
                                    const void *library, const void *function) {
    return mglRenderStageMapsVertexAttribs(stage) && bytes &&
                   (!library || !function)
               ? 1
               : 0;
}

int mglRenderStageUsesComputeBufferMap(int stage) {
    return stage == 5 /* _COMPUTE_SHADER */ ? 1 : 0;
}

uint32_t mglRenderTextureBindingStageForShader(int shader_stage) {
    return shader_stage == 0 /* _VERTEX_SHADER */
               ? 0u /* MGL_RENDER_BINDING_STAGE_VERTEX */
               : 1u /* MGL_RENDER_BINDING_STAGE_FRAGMENT */;
}

int mglRenderSamplerBindingStageForShader(int shader_stage,
                                          uint32_t *out_stage) {
    if (!out_stage) {
        return 0;
    }
    if (shader_stage == 0 /* _VERTEX_SHADER */) {
        *out_stage = 0u /* MGL_RENDER_BINDING_STAGE_VERTEX */;
        return 1;
    }
    if (shader_stage == 4 /* _FRAGMENT_SHADER */) {
        *out_stage = 1u /* MGL_RENDER_BINDING_STAGE_FRAGMENT */;
        return 1;
    }
    return 0;
}

uint32_t mglRenderSampledResourceUnit(int sampler_unit_explicit,
                                      int32_t sampler_unit,
                                      uint32_t metal_binding,
                                      uint32_t resource_binding,
                                      uint32_t max_units) {
    if (!sampler_unit_explicit ||
        !mglRenderSamplerUnitValid(sampler_unit, max_units)) {
        return UINT32_MAX;
    }
    uint32_t element = metal_binding >= resource_binding
                           ? metal_binding - resource_binding
                           : 0u;
    return (uint32_t)sampler_unit + element;
}

uint32_t mglRenderDefaultSamplerUnit(int32_t default_unit, uint32_t max_units) {
    return mglRenderSamplerUnitValid(default_unit, max_units)
               ? (uint32_t)default_unit
               : 0u;
}

int mglRenderMetalBindingPastUnits(uint32_t metal_binding, uint32_t max_units) {
    return metal_binding >= max_units ? 1 : 0;
}

int mglRenderExpectedTypeUnset(uint32_t expected_type) {
    return expected_type == 0u ? 1 : 0;
}
