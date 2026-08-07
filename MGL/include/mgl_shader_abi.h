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
 * mgl_shader_abi.h
 * MGL - pure C ABI boundary between the C/ObjC side and the C++ AIR
 * backend (see docs/AIR_SHADER_BACKEND_DESIGN.md).  C/ObjC code never
 * sees LLVM types; it hands GLSL source to mglShaderCompileGLSL and
 * receives a self-contained .metallib byte blob for newLibraryWithData.
 */

#ifndef MGL_SHADER_ABI_H
#define MGL_SHADER_ABI_H

#include <stddef.h>
#include <stdint.h>

typedef struct GLMContextRec_t *GLMContext;

#include "mgl_types_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MGLShaderStage {
    MGL_STAGE_VERTEX = 0,
    MGL_STAGE_FRAGMENT,
    MGL_STAGE_COMPUTE,
} MGLShaderStage;

/* Compile a GLSL source string for one stage into a .metallib byte blob.
 *
 * On success returns 0 and sets *metallib_out to malloc'd bytes
 * (caller frees) and *size_out to its length.
 * On failure returns -1 and writes a NUL-terminated message into err_buf
 * (err_cap bytes) if err_buf is non-NULL.
 */
int mglShaderCompileGLSL(const char *src, int stage,
                         unsigned char **metallib_out, size_t *size_out,
                         char *err_buf, size_t err_cap);

/* XFB capture variant of a vertex shader: the full output record
 * (position + varyings) is written to a device buffer at Metal buffer
 * index 29 with rasterization disabled.  Returns 0 on success. */
int mglShaderCompileGLSLCapture(const char *src, unsigned char **metallib_out,
                                size_t *size_out, char *err_buf,
                                size_t err_cap);

/* Compile one stage through the self-hosted frontend + AIR backend and
 * export its resource tables: metallib bytes + SpirvResourceList.
 * attrib_names is an optional MAX_ATTRIBS-sized array of glBindAttribLocation
 * names (index = desired location); pass NULL for no explicit bindings.
 * Returns 0 on success; lists may be NULL to skip reflection. */
int mglAirCompileGLSLWithReflect(const char *src, int stage,
                                 const char *const *attrib_names,
                                 unsigned char **metallib_out,
                                 size_t *size_out,
                                 SpirvResourceList lists[_MAX_SPIRV_RES],
                                 char *err_buf, size_t err_cap);

/* Free bytes returned by mglShaderCompileGLSL. */
void mglShaderFree(void *bytes);

/* Compare the vertex/fragment shader interfaces: varying names, types
 * and interface blocks must match across stages.  On success returns 0;
 * on mismatch or a parse/sema failure returns -1 and writes a
 * NUL-terminated message into err_buf (err_cap bytes) if non-NULL. */
int mglShaderInterfaceCheck(const char *vs_src, const char *fs_src,
                            char *err_buf, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SHADER_ABI_H */
