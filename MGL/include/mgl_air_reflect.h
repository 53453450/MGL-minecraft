/*
 * mgl_air_reflect.h
 * MGL
 *
 * Reflection export for the self-hosted GLSL frontend (MGLIRModule ->
 * SpirvResourceList).  The AIR backend produces metallib bitcode with no
 * reflection payload, so the GL query layer and per-draw binding paths
 * get their resource tables from this exporter instead of SPIRV-Cross.
 */

#ifndef MGL_AIR_REFLECT_H
#define MGL_AIR_REFLECT_H

typedef struct GLMContextRec_t *GLMContext;

#include "mgl_glsl_sema.h"
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

/* Export the symbols of `mod` into the per-type resource lists.  Lists
 * must point at zeroed SpirvResourceList arrays sized [_MAX_SPIRV_RES].
 * Returns 0 on success, -1 on failure (err filled when provided). */
int mglAirReflectModule(const MGLIRModule *mod, int stage,
                        SpirvResourceList lists[_MAX_SPIRV_RES],
                        char *err, size_t errCap);

/* Free the resource lists produced by mglAirReflectModule. */
void mglAirReflectDestroy(SpirvResourceList lists[_MAX_SPIRV_RES]);

#ifdef __cplusplus
}
#endif

#endif /* MGL_AIR_REFLECT_H */
