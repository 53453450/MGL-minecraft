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
#include "glm_limits.h"
#include "mgl_air_tess_abi.h"
#include "mgl_types_program.h"

#include <stdbool.h>

enum {
    MGL_TESS_PRIMITIVE_POINT = 0u,
    MGL_TESS_PRIMITIVE_LINE = 1u,
    MGL_TESS_PRIMITIVE_TRIANGLE = 3u,
};

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

typedef enum {
    MGL_TESS_CAPTURE_NONE = 0,
    MGL_TESS_CAPTURE_ARRAY = 1,
    MGL_TESS_CAPTURE_INDEXED_COMPACT = 2,
    MGL_TESS_CAPTURE_INDEXED_GATHER = 3,
} MGLTessCaptureKind;

typedef enum {
    MGL_TESS_EXEC_NONE = 0,
    MGL_TESS_EXEC_NATIVE = 1,
    MGL_TESS_EXEC_TES_COMPUTE = 2,
    MGL_TESS_EXEC_TES_FALLBACK = 3,
    MGL_TESS_EXEC_UNSUPPORTED = 4,
} MGLTessExecKind;

typedef struct MGLTessDrawPathPlan {
    MGLTessDrawClass classify;
    uint32_t has_tcs;
    uint32_t has_tes;
    uint32_t air_tes;
    uint32_t native_ok;
    uint32_t indexed;
    MGLTessCaptureKind capture;
    MGLTessExecKind exec;
    uint32_t need_default_factors;
    uint32_t need_tcs;
} MGLTessDrawPathPlan;

bool mglTessPlanDrawPath(GLMContext ctx, GLenum mode, GLsizei count,
                         GLsizei instanceCount, Program *tcs, Program *tes,
                         Program *gs, GLenum indexType, const char *label,
                         MGLTessDrawPathPlan *out);

void mglTessApplyGatherToContract(MGLAIRTessDrawContract *contract,
                                  uint32_t gather_count,
                                  uint32_t gather_primitives);

typedef struct MGLTessEvalComputePlan {
    uint32_t empty;
    uint32_t items_per_instance;
    uint32_t instance_count;
    uint32_t out_stride;
    uint64_t instance_bytes;
    uint64_t out_size;
} MGLTessEvalComputePlan;

bool mglTessPlanEvalCompute(Program *tes, const void *factor_bytes,
                            uint64_t factor_byte_count, uint32_t patch_count,
                            uint32_t instance_count,
                            MGLTessEvalComputePlan *out);

enum {
    MGL_TESS_EVAL_XFB_SKIP = 0,
    MGL_TESS_EVAL_XFB_CAPTURE = 1,
    MGL_TESS_EVAL_XFB_DUMMY = 2,
};

int mglTessEvalInputsReady(int has_gl_in, int has_factors);
uint32_t mglTessPlanEvalXFBSlot(int xfb_active, int size_ok);
int mglTessEvalXFBDestReady(int has_metal, int has_buf, int dest_valid);
int mglTessKeepNativeTESOnly(int native_ok, int has_tcs, int has_capture,
                             int has_factors);
int mglTessPassthroughFailIsXFBSuccess(int xfb_active);
int mglXfbShouldAdvanceWriteOffset(int xfb_active, uint64_t written);
uint64_t mglTessPassthroughInstanceOffset(uint32_t instance, uint32_t items,
                                          uint32_t stride);

bool mglTessEvalOwnsXFB(GLMContext ctx, Program *gs);

bool mglXfbPrimitiveModeAccepts(GLenum xfb_mode, GLenum draw_mode);
bool mglXfbVsOnlyEligible(const Program *program);

typedef struct MGLXfbVsField {
    uint32_t buffer_index;
    uint32_t source_offset;
    uint32_t component_offset;
    uint32_t component_count;
    uint32_t gl_type;
    uint8_t has_source;
} MGLXfbVsField;

typedef struct MGLXfbVsPlan {
    uint32_t buffer_count;
    uint32_t field_count;
    uint32_t capture_stride;
    uint32_t buffer_stride[MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS];
    MGLXfbVsField fields[MAX_ATTRIBS];
} MGLXfbVsPlan;

bool mglXfbPlanVsCapture(const Program *program, MGLXfbVsPlan *out);

typedef struct MGLXfbVsBufferDest {
    uint8_t skip;
    uint32_t written_records;
    uint32_t written_bytes;
    uint32_t destination_offset;
} MGLXfbVsBufferDest;

bool mglXfbPlanVsBufferDest(uint32_t record_count, uint32_t stride,
                            int has_buffer, int64_t slot_offset,
                            uint64_t session_offset, uint64_t visible_bytes,
                            MGLXfbVsBufferDest *out);

uint32_t mglXfbPackVsRecords(const MGLXfbVsPlan *plan, uint32_t buffer,
                             const void *src, uint64_t src_offset,
                             uint32_t src_stride, uint32_t record_count,
                             void *dst, uint32_t dst_stride);

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
GLenum mglTessRasterGLMode(const Program *tes);
uint32_t mglTessRasterPrimitiveType(const Program *tes);

typedef struct MGLTessRasterQueryPlan {
    uint64_t prims;
    uint64_t written;
} MGLTessRasterQueryPlan;

void mglTessPlanRasterQuery(const Program *tes, uint64_t instance_count,
                            uint64_t items_per_instance, int xfb_active,
                            uint64_t xfb_written_bytes,
                            uint32_t xfb_compact_stride,
                            MGLTessRasterQueryPlan *out);

enum {
    MGL_TESS_AFTER_COMPUTE_GS = 0,
    MGL_TESS_AFTER_COMPUTE_DISCARD = 1,
    MGL_TESS_AFTER_COMPUTE_PASSTHROUGH = 2,
};

typedef struct MGLTessEvalAfterComputePlan {
    uint32_t action;
    uint32_t gs_vertex_count;
    uint8_t gs_empty;
} MGLTessEvalAfterComputePlan;

int mglTessPlanEvalAfterCompute(int has_gs, int rasterizer_discard,
                                uint32_t items_per_instance,
                                uint32_t instance_count,
                                MGLTessEvalAfterComputePlan *out);
int mglTessPassthroughRasterReady(int state_ready, int has_encoder,
                                  int raster_empty);
int mglTessNativePipelineReady(int state_ready, int has_encoder);
int mglTessNativeShouldDraw(int raster_empty, int fully_culled);
int mglTessTextureBindIsStorage(uint32_t kind);
int mglTessTextureBindNeedsSampler(uint32_t kind, uint32_t combined_slot);

bool mglXfbPlanVsBufferDestOrUnbacked(uint32_t record_count, uint32_t stride,
                                      int has_metal, int64_t slot_offset,
                                      uint64_t session_offset,
                                      uint64_t visible_bytes,
                                      MGLXfbVsBufferDest *out);

typedef struct MGLTessNativeAttribPlan {
    uint32_t index;
    uint32_t format;
    uint32_t offset;
} MGLTessNativeAttribPlan;

typedef struct MGLTessNativeVertexPlan {
    uint32_t attrib_count;
    uint32_t n_attribs;
    uint32_t stride;
    MGLTessNativeAttribPlan attribs[32];
} MGLTessNativeVertexPlan;

int mglTessPlanNativeVertexDescriptor(const Program *tes,
                                      uint32_t tcs_output_stride,
                                      MGLTessNativeVertexPlan *out);

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
void mglXfbDecodeIntCarriersInBytes(void *bytes, uint64_t nbytes,
                                    uint32_t stride, const Program *program,
                                    uint32_t buffer_index, int stage);
int mglXfbSeparateAttribs(uint32_t buffer_mode);
int mglTessXFBCopyBackReady(uint64_t written, int has_temp, int has_dest);
int mglXfbVaryingSlotValid(uint32_t varying);
int mglXfbCPUShadowFits(int has_cpu, int64_t buf_size, uint64_t dest_offset,
                        uint64_t written);
uint64_t mglXfbSessionOffsetOr(uint64_t write_offset, uint64_t fallback);
int mglXfbRecordCountFits(uint64_t records);
int mglTessStageInAttribInRange(uint32_t attrib);
int mglTessTCSStageInEmptyOK(uint32_t member_count);
uint32_t mglTessPackXFBInterleaved(const Program *tes, const void *src,
                                   uint32_t src_stride, uint32_t vertex_count,
                                   void *dst, uint32_t dst_stride);
uint32_t mglTessPackXFBSeparate(const Program *tes, const char *name,
                                const void *src, uint32_t src_stride,
                                uint32_t vertex_count, void *dst);

typedef struct MGLTessXFBDestPlan {
    uint32_t copied_vertices;
    uint32_t written_bytes;
    uint32_t destination_offset;
    uint8_t valid;
} MGLTessXFBDestPlan;

int mglTessPlanXFBDestination(uint32_t items_per_instance,
                              uint32_t instance_count,
                              uint32_t compact_stride,
                              uint32_t vertices_per_primitive,
                              uint64_t session_offset, int64_t slot_offset,
                              uint64_t visible_bytes,
                              MGLTessXFBDestPlan *out);

int mglTessPlanEvalXfbCapture(uint32_t items_per_instance,
                              uint32_t instance_count, uint32_t out_stride,
                              uint32_t compact_stride,
                              uint32_t *capture_vertices,
                              uint32_t *required_bytes);

void mglTessPlanEvalGather(int indexed, uint32_t instance_records,
                           uint32_t patch_vertices, uint32_t patch_count,
                           uint32_t *verts_per_instance,
                           uint32_t *prims_per_instance);

uint64_t mglTessDummyXfbBytes(uint64_t out_size);

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

void mglTessFillEvalPerPatchSpec(
    void *gl_in_buffer, uint64_t gl_in_offset, uint64_t gl_in_instance_stride,
    void *gather_buffer, uint32_t gather_verts, uint32_t gather_prims,
    int indexed, uint32_t gl_in_vertices, uint32_t patch_count,
    uint32_t instance_count, uint32_t items_per_instance,
    MGLTessEvalPerPatchDispatchSpec *out);

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

typedef struct MGLTessVertexCapturePlan {
    uint32_t records_per_instance;
    uint32_t capture_stride;
    uint64_t capture_size;
    uint64_t capture_offset;
    uint32_t params[3];
} MGLTessVertexCapturePlan;

bool mglTessPlanVertexCapture(Program *vs, uint32_t records_per_instance,
                              uint32_t instance_count, uint32_t first,
                              uint32_t base_instance,
                              MGLTessVertexCapturePlan *out);

typedef struct MGLTessEvalGlInPlan {
    uint32_t from_tcs;
    uint64_t gl_in_offset;
    uint64_t gl_in_stride;
    uint32_t gl_in_vertices;
    uint64_t gl_in_instance_stride;
} MGLTessEvalGlInPlan;

bool mglTessResolveEvalGlIn(const MGLAIRTessDrawContract *contract,
                            int has_tcs_output, uint64_t tcs_output_offset,
                            uint64_t tcs_output_stride,
                            uint32_t tcs_out_vertices, int has_capture,
                            uint64_t capture_offset, int indexed_draw,
                            uint32_t instance_records, uint32_t instance_count,
                            MGLTessEvalGlInPlan *out);

typedef struct MGLTessTCSStageInPlan {
    uint64_t vertices;
    uint64_t stride;
    uint64_t bytes;
    MGLTessStageInMember members[1];
    uint32_t member_count;
} MGLTessTCSStageInPlan;

bool mglTessPlanTCSStageIn(uint32_t patch_vertices, uint32_t patch_count,
                           GLsizei vertex_count, MGLTessTCSStageInPlan *out);

enum {
    MGL_TESS_INDEXED_STAGE_IN_ARRAY = 0,
    MGL_TESS_INDEXED_STAGE_IN_OK = 1,
    MGL_TESS_INDEXED_STAGE_IN_BAD = 2,
};

typedef struct MGLTessIndexedStageInPlan {
    uint32_t status;
    uint32_t index_stride;
    uint64_t bytes_needed;
} MGLTessIndexedStageInPlan;

int mglTessPlanIndexedStageIn(uint32_t index_type, uint64_t index_offset,
                              int32_t count, int64_t ebo_size,
                              MGLTessIndexedStageInPlan *out);
int mglTessCommandBufferNeedsNew(int has_state, uint32_t status);
int mglTessCommandBufferCanInitBlit(int has_state, uint32_t status);
int mglTessStageHasCompiledFunction(int has_shader, int has_mtl_function);
int mglTessMustEndRenderBeforeCompute(int has_render_encoder);
int mglTessComputePipelineReady(int create_ok, int has_handle);
int mglTessIsolatedNeedsCopyBack(int writable, int has_source,
                                 uint32_t init_length);

int mglTessStageInUseCurrentValue(uint32_t enabled_attribs, uint32_t attrib,
                                  int has_binding);

typedef struct MGLTessIsolatedBindingPlan {
    uint8_t isolated;
    uint8_t writable;
    uint32_t fallback_length;
    uint32_t init_length;
} MGLTessIsolatedBindingPlan;

bool mglTessPlanIsolatedBinding(int has_buffer, int64_t offset,
                                uint64_t buffer_length,
                                int64_t storage_remaining,
                                uint64_t available_bytes,
                                uint32_t required_bytes, int resource_type,
                                MGLTessIsolatedBindingPlan *out);

uint32_t mglTessRequiredBindingBytes(int resource_type,
                                     uint32_t required_bytes);
void mglTessFillRuntimeArraySizeConstants(const BufferMap *maps,
                                          uint32_t map_count,
                                          uint32_t size_buffer_index,
                                          uint32_t *out, uint32_t out_cap);
void mglTessFillPointSizeParams(float point_size, int program_point_size,
                                float out[2]);
int mglTessNativeBuffersReady(int has_factors, int has_tcs_out,
                              uint32_t tcs_stride);

enum {
    MGL_TESS_NATIVE_FACTOR_NONE = 0,
    MGL_TESS_NATIVE_FACTOR_REUSE = 1,
    MGL_TESS_NATIVE_FACTOR_REPACK_TRI = 2,
};

int mglTessPlanNativeFactor(uint32_t tess_gen_mode, uint64_t canonical_bytes,
                            uint32_t patch_count, uint32_t *out_bytes);
uint32_t mglTessNativePatchOutStride(int has_tcs, uint32_t tcs_patch_stride);
int mglTessMultiInstanceTCSReuseWarn(int from_tcs, int32_t instance_count);
int mglTessMultiInstanceTCSReuseIsError(int from_tcs, int32_t instance_count);
int mglTessEvalIndexedGatherReady(int indexed, int has_gather,
                                  uint32_t instance_records);
uint32_t mglTessTCSCaptureStageInStride(Program *tcs);

enum {
    MGL_TESS_TCS_STAGE_IN_CAPTURE = 0,
    MGL_TESS_TCS_STAGE_IN_PACK = 1,
};

typedef struct MGLTessTCSStageInSourcePlan {
    uint32_t kind;
    uint32_t stride;
} MGLTessTCSStageInSourcePlan;

int mglTessPlanTCSStageInSource(int has_capture, Program *tcs,
                                MGLTessTCSStageInSourcePlan *out);
void mglTessFillTCSIndirectParams(uint32_t patch_vertices,
                                  uint32_t instance_count, uint32_t out[2]);
void mglTessFillDefaultFactorLevels(const float outer[4], const float inner[2],
                                    float out[6]);
int mglTessPlanDefaultFactorBytes(uint32_t patch_count, uint64_t *out_bytes);
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
int mglTessGenModeIsIsolines(uint32_t tess_gen_mode);
int mglTessWritePointSize(uint32_t tess_gen_point_mode);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_TESS_H */
