/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#ifndef MGL_DRAW_TESS_H
#define MGL_DRAW_TESS_H

#include "glcorearb.h"
#include "glm_context.h"
#include "mgl_air_tess_abi.h"
#include "mgl_types_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MGL_TESS_DRAW_NOT_APPLICABLE = 0,
    MGL_TESS_DRAW_NOOP_HANDLED = 1,
    MGL_TESS_DRAW_ACTIVE = 2,
} MGLTessDrawClass;

MGLTessDrawClass mglTessClassifyDraw(GLMContext ctx, GLenum mode, GLsizei count,
                                     GLsizei instanceCount, Program *tcs,
                                     Program *tes, const char *label);

void mglTessFillDrawContract(MGLAIRTessDrawContract *contract, GLMContext ctx,
                             Program *tcs, Program *tes, Program *vs,
                             GLint first, GLsizei count, GLenum indexType,
                             const void *indices, GLint baseVertex,
                             GLsizei instanceCount, GLuint baseInstance);

bool mglTessNativeInterfaceSupported(Program *tcs, Program *tes);
bool mglTessNativeBlockedByGeometry(Program *gs);

typedef struct MGLTessNativeEncodeState {
    void *encoder_owner;
    void *tcs_output_buffer;
    void *native_factors;
    void *control_point_index_buffer;
    void *tcs_patch_out_buffer;
    uint32_t patch_vertices;
    uint32_t patch_count;
    uint32_t instance_count;
    uint32_t base_instance;
    uint32_t tess_gen_mode;
    uint32_t tcs_out_vertices;
    uint64_t tcs_output_stride;
    uint64_t tess_vertex_capture_offset;
    uint64_t tess_instance_records;
    uint32_t tess_indexed_draw;
    uint32_t patch_out_stride;
} MGLTessNativeEncodeState;

void mglTessEncodeNativePatches(const MGLTessNativeEncodeState *state);

typedef struct MGLTessTCSCoreLayout {
    uint32_t patch_vertices;
    uint32_t patch_count;
    uint32_t instance_count;
    uint32_t tcs_out_vertices;
    uint32_t output_stride;
    uint32_t patch_stride;
    uint64_t output_bytes;
    uint64_t patch_out_bytes;
    uint64_t factor_bytes;
} MGLTessTCSCoreLayout;

typedef struct MGLRenderComputeExecutionPlan_t MGLRenderComputeExecutionPlan;

bool mglTessComputeTCSCoreLayout(Program *tcs,
                                 const MGLAIRTessDrawContract *contract,
                                 MGLTessTCSCoreLayout *out);

bool mglTessAppendTCSCoreBindings(MGLRenderComputeExecutionPlan *plan,
                                  void *output, void *patch_out,
                                  void *indirect, void *factors,
                                  void *stage_in, uint64_t stage_in_offset,
                                  const MGLTessTCSCoreLayout *layout);

uint32_t mglTessEvalItemsPerPatch(Program *tes, const void *factor_record);
uint64_t mglTessEvalItemsPerInstance(Program *tes, const void *factor_bytes,
                                     uint32_t patch_count);
bool mglTessFillEvalPatchItemBases(Program *tes, const void *factor_bytes,
                                   uint32_t patch_count, uint32_t *bases_out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_TESS_H */
