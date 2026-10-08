/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * SPIR-V → GLSL helper used by glShaderBinary / glSpecializeShader.
 */

#ifndef MGL_SPIRV_GLSL_H
#define MGL_SPIRV_GLSL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for mglSpirvToGLSL. */
enum {
    MGL_SPIRV_OK = 0,
    MGL_SPIRV_ERR_UNSUPPORTED = 1, /* built without spirv-cross */
    MGL_SPIRV_ERR_PARSE = 2,
    MGL_SPIRV_ERR_ENTRY = 3,       /* entry point missing for stage */
    MGL_SPIRV_ERR_SPEC = 4,        /* unknown specialization constant id */
    MGL_SPIRV_ERR_COMPILE = 5,
    MGL_SPIRV_ERR_OOM = 6
};

/* Map GL shader type (GL_VERTEX_SHADER, …) to SpvExecutionModel. Returns -1
 * if type is unknown. */
int mglSpirvExecutionModelForGLType(unsigned gl_shader_type);

/* True if the module contains OpName / OpMemberName. */
int mglSpirvModuleHasNames(const uint32_t *words, size_t word_count);

/*
 * Cross-compile a SPIR-V module to OpenGL-style GLSL for one entry point.
 * On success *glsl_out is a malloc'd NUL-terminated string the caller frees.
 * err/err_len receive a short diagnostic when non-NULL.
 */
int mglSpirvToGLSL(const uint32_t *words, size_t word_count,
                   unsigned gl_shader_type, const char *entry_point,
                   const unsigned *spec_ids, const unsigned *spec_values,
                   unsigned num_specs, char **glsl_out, char *err,
                   size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SPIRV_GLSL_H */
