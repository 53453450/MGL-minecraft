/*
 * mgl_air_reflect.h
 * MGL
 *
 * Reflection export for the self-hosted GLSL frontend (MGLIRModule ->
 * MGLShaderResourceList).  The AIR backend produces metallib bitcode with no
 * reflection payload, so the GL query layer and per-draw binding paths
 * get their resource tables from this exporter instead of the historical
 * SPIRV-Cross lowering.)
 */

#ifndef MGL_AIR_REFLECT_H
#define MGL_AIR_REFLECT_H

typedef struct GLMContextRec_t *GLMContext;

#include "mgl_glsl_sema.h"
#include "mgl_shader_abi.h"
#include "mgl_types_program.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Map a MGLIRType to its GL enum (GL_FLOAT, GL_FLOAT_VEC3, GL_SAMPLER_2D,
 * ...).  Arrays map to the element type. */
GLuint mglAirGLTypeFromIR(const MGLIRType *t);

/* GL array size of a MGLIRType: 1 for non-arrays, element count for
 * arrays (0 for runtime arrays). */
GLint mglAirGLArraySizeFromIR(const MGLIRType *t);

/* Export the symbols of `mod` into the per-type resource lists. `stage` uses
 * the public MGL_STAGE_* values from mgl_shader_abi.h. Lists
 * must point at zeroed MGLShaderResourceList arrays sized [MGL_MAX_SHADER_RESOURCES].
 * Returns 0 on success, -1 on failure (err filled when provided). */
int mglAirReflectModule(const MGLIRModule *mod, int stage,
                        const char *const *attrib_names,
                        MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES],
                        char *err, size_t errCap);

/* Free the resource lists produced by mglAirReflectModule. */
void mglAirReflectDestroy(MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES]);

#ifdef __cplusplus
}
#endif

#endif /* MGL_AIR_REFLECT_H */
