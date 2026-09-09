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
 * mgl_binding_policy.h
 *
 * C1 / O3.3 domain strip from mgl_render.cpp — binding slot / sampler /
 * stage / plain-uniform policy (CTS BindingState alignment). Pure
 * classification + name tables; no Metal-cpp, no renderer instance.
 * Resource-type / shader-stage integers use the mgl_types_program.h ABI
 * (_UNIFORM_BUFFER_RES=1 … _SEPARATE_SAMPLERS_RES=12; _VERTEX_SHADER=0 …
 * _MAX_SHADER_TYPES=6). Binding stage 0/1 matches MGL_RENDER_BINDING_STAGE_*.
 *
 * Do not sink these helpers back into mgl_render.cpp.
 * Do not grow +Binding.m into a thick shell — ObjC stays a thin bind port.
 *
 * Callers historically went through mgl_render.h; that header includes
 * this one so BindingState / shader_resource / Texture keep call sites.
 */

#ifndef MGL_BINDING_POLICY_H
#define MGL_BINDING_POLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t mglRenderShaderResourceElementCount(uint32_t gl_array_size);
int mglRenderImageUnitsInRange(uint32_t metal_slot, uint32_t gl_unit,
                               uint32_t max_units);
uint32_t mglRenderImageUnitFromResource(int explicit_by_slot,
                                        uint32_t explicit_unit,
                                        int32_t sampler_unit,
                                        uint32_t gl_binding, uint32_t element);
int mglRenderComputeTextureBindKind(uint32_t spvc_type);
int mglRenderComputeTextureListExpandsByElement(uint32_t spvc_type);
int mglRenderComputeTextureBindIsStorage(uint32_t kind);
int mglRenderComputeTextureBindNeedsSampler(uint32_t kind, int has_combined);
int mglRenderShaderResourceTypeIsSamplerImage(uint32_t res_type);
const char *mglRenderShaderResourceTypeName(uint32_t res_type);
int mglRenderPlainUniformBindingForName(const char *name);
uint32_t mglRenderClientBufferBindingForResource(uint32_t resource_type,
                                                 const char *name,
                                                 int32_t uniform_location,
                                                 uint32_t location,
                                                 uint32_t gl_binding);
int mglRenderPlainUniformAllowsGlobalFallback(const char *name);
uint32_t mglRenderStageBufferResourceElementCount(uint32_t resource_type,
                                                  int has_res,
                                                  uint32_t ubo_array_size,
                                                  int has_ubo_members,
                                                  int32_t gl_array_size);
uint32_t mglRenderClientBufferBindingForResourceElement(
    uint32_t resource_type, uint32_t base_binding, uint32_t element,
    const uint32_t *ubo_array_bindings, uint32_t ubo_array_size);
uint32_t mglRenderCombinedSamplerSlot(int has_res, int has_combined,
                                      uint32_t combined_binding);
uint32_t mglRenderCombinedSamplerSlotForElement(int has_res, int has_combined,
                                                uint32_t combined_binding,
                                                uint32_t element);
int mglRenderSamplerNameLooksSamplerLike(const char *name);
int mglRenderResourceLooksSamplerLike(uint32_t res_type, uint32_t image_dim,
                                      int32_t uniform_location,
                                      const char *name);
uint32_t mglRenderResourceMetalSlot(int has_resource, uint32_t binding,
                                    uint32_t element, uint32_t fallback);
int mglRenderSamplerUnitValid(int32_t unit, uint32_t max_units);
int mglRenderShaderStageValid(int stage);
int mglRenderStageMapsVertexAttribs(int stage);
int mglRenderVertexCaptureNeedsLoad(int stage, const void *bytes,
                                    const void *library, const void *function);
int mglRenderStageUsesComputeBufferMap(int stage);
uint32_t mglRenderTextureBindingStageForShader(int shader_stage);
int mglRenderSamplerBindingStageForShader(int shader_stage,
                                          uint32_t *out_stage);
uint32_t mglRenderSampledResourceUnit(int sampler_unit_explicit,
                                      int32_t sampler_unit,
                                      uint32_t metal_binding,
                                      uint32_t resource_binding,
                                      uint32_t max_units);
uint32_t mglRenderDefaultSamplerUnit(int32_t default_unit, uint32_t max_units);
int mglRenderMetalBindingPastUnits(uint32_t metal_binding, uint32_t max_units);
int mglRenderExpectedTypeUnset(uint32_t expected_type);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BINDING_POLICY_H */
