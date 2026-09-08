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

#include <stdbool.h>

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

uint32_t mglTessVerticesPerPrimitive(const Program *tes);
uint64_t mglTessPrimitivesFromItems(const Program *tes, uint64_t items);
uint64_t mglTessGeneratedPrimitiveCount(Program *tes, const void *factor_bytes,
                                        uint32_t patch_count,
                                        uint32_t instance_count);

/* Seed TES compute output with domain TessCoords for every live patch,
 * then replicate instance 0. Returns items per instance, or 0 on error. */
uint32_t mglTessSeedEvalOutputRecords(Program *tes, const void *factor_bytes,
                                      uint32_t patch_count,
                                      uint32_t instance_count, void *records,
                                      uint64_t records_bytes, uint32_t stride);

int mglTessResolveXFBSource(const Program *program, const char *name,
                            uint32_t *offset_out, uint32_t *gl_type_out,
                            uint32_t *bytes_out);
void mglTessPackXFBFieldFromCarrier(uint32_t gl_type, const void *src,
                                    void *dst, uint32_t field_bytes);
uint32_t mglTessPackXFBInterleaved(const Program *tes, const void *src,
                                   uint32_t src_stride, uint32_t vertex_count,
                                   void *dst, uint32_t dst_stride);
uint32_t mglTessPackXFBSeparate(const Program *tes, const char *name,
                                const void *src, uint32_t src_stride,
                                uint32_t vertex_count, void *dst);

typedef struct MGLTessEvalPerPatchDispatchSpec {
    void *gl_in_buffer;
    uint64_t gl_in_offset;
    uint64_t gl_in_instance_stride;
    void *gather_buffer;
    uint32_t gather_verts_per_instance;
    uint32_t gather_prims_per_instance;
    uint32_t gather_first_vertex;
    uint32_t indexed;
    uint32_t gl_in_vertices;
    uint32_t patch_count;
    uint32_t instance_count;
    uint32_t items_per_instance;
} MGLTessEvalPerPatchDispatchSpec;

/* Appends per-instance gl_in / gather binds and one dispatch per live patch.
 * Discarded patches (items==0) emit no dispatch. out_keep_alive is a malloc
 * blob of inline bytes the caller must keep until execute, then free. */
bool mglTessAppendEvalPerPatchDispatches(
    MGLRenderComputeExecutionPlan *plan, Program *tes, const void *factor_bytes,
    const MGLTessEvalPerPatchDispatchSpec *spec, void **out_keep_alive);

enum {
    MGL_TESS_BIND_STORAGE_IMAGE = 0u,
    MGL_TESS_BIND_SAMPLED_IMAGE = 1u,
};

typedef struct MGLTessTextureBind {
    uint32_t kind;
    uint32_t metal_slot;
    uint32_t gl_unit;
    uint32_t combined_sampler_slot; /* UINT32_MAX if none */
} MGLTessTextureBind;

uint32_t mglTessCollectTextureBinds(GLMContext ctx, Program *program, int stage,
                                    MGLTessTextureBind *out, uint32_t cap);

bool mglTessInitStageInDefaults(void *dst, uint64_t vertices, uint64_t stride);

bool mglTessCompactSparseCapture(const void *sparse, uint64_t sparse_offset,
                                 uint32_t sparse_records, uint32_t stride,
                                 const uint32_t *gather, uint32_t gather_count,
                                 uint32_t instance_count, void *continuous,
                                 uint64_t continuous_bytes);

enum {
    MGL_TESS_STAGE_IN_FLOAT = 0u,
    MGL_TESS_STAGE_IN_INT = 1u,
    MGL_TESS_STAGE_IN_UINT = 2u,
};

typedef struct MGLTessStageInMember {
    uint32_t attribute;
    uint32_t offset;
    uint32_t size;
    uint32_t component_bytes;
    uint32_t components;
    uint32_t base_type;
} MGLTessStageInMember;

typedef struct MGLTessStageInAttribSrc {
    const uint8_t *bytes;
    uint8_t current[16];
    uint32_t current_valid;
    uint32_t use_current;
    uint32_t type;
    uint32_t attrib_size;
    uint32_t normalized;
    uint32_t stride;
    uint32_t divisor;
    int64_t binding_offset;
    int64_t relativeoffset;
    uint64_t buffer_size;
} MGLTessStageInAttribSrc;

bool mglTessPackStageInRecords(
    void *dst, uint64_t vertices, uint64_t stride, GLint first, GLsizei count,
    const uint8_t *index_bytes, GLenum index_type, bool restart_enabled,
    uint32_t restart_index, GLint base_vertex, GLuint base_instance,
    const MGLTessStageInMember *members, uint32_t member_count,
    const MGLTessStageInAttribSrc *srcs);

bool mglTessSanitizeRestartIndices(void *dst, const void *src, uint32_t count,
                                   GLenum index_type, uint32_t restart_index);

void mglTessFillCaptureParams(uint32_t first, uint32_t records_per_instance,
                              uint32_t base_instance, uint32_t out[3]);
void mglTessBindCaptureSlots(void *encoder_owner, void *capture_buffer,
                             const uint32_t params[3]);
void mglTessEncodeCaptureArray(void *encoder_owner, uint32_t first,
                               uint32_t count, uint32_t instance_count,
                               uint32_t base_instance);
void mglTessEncodeCaptureIndexed(void *encoder_owner, void *index_buffer,
                                 uint32_t index_type, uint64_t index_offset,
                                 uint32_t count, int32_t base_vertex,
                                 uint32_t instance_count,
                                 uint32_t base_instance);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_TESS_H */
