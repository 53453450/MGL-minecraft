/*
 * mgl_spirv_resource.h
 * MGL
 *
 * SPIR-V Resource Helper Subsystem: pure helpers for mapping SPIR-V
 * reflected resources (SpirvResource) to GL client buffer bindings and
 * Metal argument slots.  These helpers encode the Minecraft-specific
 * plain-uniform binding table and the UBO/SSBO array element expansion
 * rules shared across the vertex/fragment/compute binding paths.
 *
 * All functions here are pure: they operate only on the SpirvResource
 * pointer passed in and have no dependency on the renderer instance,
 * command buffer, or encoder.
 */

#ifndef MGL_SPIRV_RESOURCE_H
#define MGL_SPIRV_RESOURCE_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve the GL client buffer binding (UBO/SSBO/plain-uniform index) for
 * a SPIR-V reflected resource.  For plain uniforms (SPVC_RESOURCE_TYPE_
 * UNIFORM_CONSTANT) this consults the Minecraft plain-uniform binding
 * table (ModelViewMat=0, ProjMat=1, ...) and falls back to
 * uniform_location / location / gl_binding.  For UBOs/SSBOs this returns
 * res->gl_binding. */
GLuint mglClientBufferBindingForResource(int resourceType,
                                         const SpirvResource *res);

/* Returns the Metal argument slot assigned to `res` (res->binding), or 0
 * if res is NULL. */
GLuint mglMetalResourceSlot(const SpirvResource *res);

/* Number of Metal buffer elements occupied by this resource.  UBOs with
 * ubo_array_size > 1 expand to that many elements; plain uniforms and
 * SSBOs with gl_array_size > 1 expand to gl_array_size elements.
 * Otherwise 1. */
GLuint mglStageBufferResourceElementCount(int resourceType,
                                          const SpirvResource *res);

/* Resolve the GL client buffer binding for a specific element of an
 * arrayed UBO.  For UBOs with ubo_array_bindings, returns the per-element
 * binding; otherwise returns base + element. */
GLuint mglClientBufferBindingForResourceElement(int resourceType,
                                                const SpirvResource *res,
                                                GLuint element);

/* Returns the Metal argument slot for `element` of `res`
 * (mglMetalResourceSlot(res) + element). */
GLuint mglMetalResourceSlotForElement(const SpirvResource *res,
                                      GLuint element);

/* Returns true if a plain uniform resource may fall back to the global
 * (Minecraft legacy) binding table.  Mojang/Iris u_* uniforms are
 * excluded because their numeric locations collide with legacy slots
 * but mean different things — falling back corrupts first-person items
 * and inventory icons. */
bool mglPlainUniformAllowsGlobalFallback(const SpirvResource *res);

/* Human-readable name for a SPVC_RESOURCE_TYPE_* constant, or "resource"
 * for unknown types.  Used by diagnostic/logging paths. */
const char *mglSpirvResourceTypeName(int type);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SPIRV_RESOURCE_H */
