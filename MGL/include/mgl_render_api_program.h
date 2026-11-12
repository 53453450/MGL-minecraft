/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_render_api_program.h — FE-safe Program backend facade (ARCH-007 / G1).
 *
 * Must not include mgl_backend_objects.h / mgl_program_backend.h.
 */

#ifndef MGL_RENDER_API_PROGRAM_H
#define MGL_RENDER_API_PROGRAM_H

#include "mgl_render_fwd.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque blob large enough for one Program's per-stage module backends. */
enum { MGL_RENDER_PROGRAM_MODULE_BACKEND_BLOB = 4096 };

void mglRenderProgramReleaseBackend(Program *program);
uint64_t mglRenderProgramPipelineCacheInstanceId(const Program *program);
void mglRenderProgramAssignPipelineCacheInstanceId(Program *program,
                                                   uint64_t instance_id);
void mglRenderProgramBumpPipelineCacheGeneration(Program *program);

/* Steal / restore stage metallib backends into an opaque FE blob (A16). */
void mglRenderProgramCaptureModuleBackends(Program *program, void *blob,
                                           size_t blob_bytes);
void mglRenderProgramClearModuleBackends(Program *program);
void mglRenderProgramRestoreModuleBackends(Program *program, const void *blob,
                                           size_t blob_bytes);
void mglRenderProgramClearCapturedModuleBackend(void *blob, size_t blob_bytes,
                                                int stage);

/* Stage metallib accessors — FE must not touch MGLShaderModuleBackend. */
const unsigned char *mglRenderProgramModuleMetallibBytes(const Program *program,
                                                         int stage);
size_t mglRenderProgramModuleMetallibSize(const Program *program, int stage);
void mglRenderProgramSetModuleMetallib(Program *program, int stage,
                                       unsigned char *bytes, size_t size);
void mglRenderProgramSetModuleTessCaptureMetallib(Program *program, int stage,
                                                  unsigned char *bytes,
                                                  size_t size);
void mglRenderProgramSetModuleCullCaptureMetallib(Program *program, int stage,
                                                  unsigned char *bytes,
                                                  size_t size);
void mglRenderProgramSetModuleTesComputeMetallib(Program *program, int stage,
                                                 unsigned char *bytes,
                                                 size_t size);
void mglRenderProgramSetModuleSamplerReductionMetallib(Program *program,
                                                       int stage,
                                                       unsigned char *bytes,
                                                       size_t size);
void mglRenderProgramSetModulePrimitiveIdVaryingMetallib(
    Program *program, int stage, unsigned char *bytes, size_t size);
void mglRenderProgramSetModuleNeedsRuntimeArraySizeBuffer(Program *program,
                                                          int stage, int needs);
void mglRenderProgramSetModuleNeedsSparseResidency(Program *program,
                                                   int stage, int needs,
                                                   uint32_t buffer_index);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_API_PROGRAM_H */
