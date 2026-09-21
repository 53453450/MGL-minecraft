/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_BINDING_H
#define MGL_RENDER_API_BINDING_H

/* Declarations for the binding slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifndef MGL_RENDER_PIPELINE_CACHE_KEY_WORDS_DEFINED
#define MGL_RENDER_PIPELINE_CACHE_KEY_WORDS_DEFINED
enum { MGL_RENDER_PIPELINE_CACHE_KEY_WORDS = 7 };
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderBindBuffer(GLMContext glm_ctx, Buffer *buffer);

void mglRenderBindProgram(GLMContext glm_ctx, Program *program);

/* Load every AIR-backed stage in a linked Program and install the resulting
 * +1 library/function references directly in its MGLShaderModule slots.  Programs that
 * still contain a legacy MSL stage are left untouched and return
 * NOT_APPLICABLE so the ObjC baseline can bind the whole program. */
int mglRenderBindAIRProgram(Program *program,
                               int *failed_stage_out,
                               char *err,
                               size_t errcap);

/* Materialize shared, copy-backed, client-storage, and persistent no-copy
 * Buffer storage in Metal-cpp. No-copy buffers transfer VM-range cleanup to
 * the retained Metal object and set data.mtl_owns_buffer_data. */
int mglRenderBindBufferStorage(Buffer *buffer,
                                  char *err,
                                  size_t errcap);

int mglRenderFillVertexConversionFromAttribKind(
    int attrib_kind, uint32_t size, uint32_t type, int normalized,
    int dst_signed, MGLRenderVertexConversion *out);

/* Validate every non-empty entry (bounds vs the Metal buffer lengths) and,
 * when blit_encoder is non-NULL, encode each copy via
 * mglRenderBlitCopyBuffer.  Returns 0 on success, -1 on the first
 * invalid entry / encode failure. */
int mglRenderEncodeStageBindingCopyBacks(
    const MGLRenderCopyBackEntry *entries,
    uint32_t count,
    void *blit_encoder);

int mglRenderPlanVertexAttribSpan(int64_t binding_offset, int64_t relativeoffset,
                                  uint32_t type, uint32_t size,
                                  int64_t *offset_out, int64_t *span_out,
                                  int64_t *end_out);

int mglRenderPlanAttribFetch(uint32_t gl_type, uint32_t size, uint32_t stride,
                             int64_t binding_offset, int64_t relativeoffset,
                             uint32_t divisor, uint64_t first_vertex,
                             uint64_t last_vertex, int64_t vbo_size,
                             MGLRenderAttribFetchPlan *out);

uint32_t mglRenderAttribFormatOrFallback(uint32_t planned, uint32_t type,
                                         uint32_t size, int normalized);

int mglRenderAttribNeedsConversion(int long_attr, uint32_t type, int integer);

int mglRenderAttribNeedsConvertedMetalStream(uint32_t type, int integer);

int mglRenderAttribColorUByteNeedsNormalize(uint32_t type, uint32_t size,
                                            int already_norm);

uint32_t mglRenderAttribEffectiveNormalized(uint32_t already, int needs);

double mglRenderDecodeVertexAttribComponent(const uint8_t *src, uint32_t type,
                                            int normalized, uint32_t component);

int mglRenderBufferMapIsBaseBinding(uint32_t attribute_mask);

int mglRenderClientBindingInRange(uint32_t binding, uint32_t max);

int mglRenderBufferBindingEmpty(int has_buf, uint32_t name);

int mglRenderBaseBindingTooSmall(int64_t range, uint64_t reflected);

int mglRenderAttribOffsetsValid(int64_t binding_offset,
                                int64_t relativeoffset);

uint32_t mglRenderBuildCurrentVertexAttribBytes(
    uint32_t type, uint32_t size, const int32_t current_i[4],
    const uint32_t current_u[4], const float current_f[4], uint8_t bytes[16]);

void mglRenderPlanVertexAttribFormat(uint32_t type, uint32_t size, int integer,
                                     int normalized, int is_color_input,
                                     uint32_t shader_gl_type,
                                     uint32_t *format_out,
                                     int *needs_conversion_out,
                                     int *normalized_out,
                                     int *conversion_kind_out);

uint32_t mglRenderPlanVertexAttribStride(
    uint32_t type, uint32_t size, int integer, int uses_current,
    int integer_converted, uint32_t resolved_stride,
    uint32_t existing_layout_stride);

uint32_t mglRenderPlanVertexAttribOffset(int uses_current, int needs_conversion,
                                         int absolute_offsets,
                                         uint32_t attrib_index,
                                         uint32_t pool_stride,
                                         uint32_t relativeoffset,
                                         uint32_t binding_offset);

/* ARB_vertex_attrib_binding resolve — the
 * binding-table override (offset/stride/divisor) vs the legacy per-attrib
 * values.  Pure decision shared by both gates; the GL buffer validation
 * stays on the ObjC side. */
int mglRenderResolveVertexAttribBinding(
    uint32_t binding_index,
    int binding_has_buffer,
    int64_t binding_offset,
    uint32_t binding_stride,
    int64_t attrib_binding_offset,
    uint32_t attrib_stride,
    uint32_t binding_divisor,
    uint32_t attrib_divisor,
    MGLRenderVertexAttribResolve *out);

/* Per-renderer pipeline ownership. The opaque owner retains active objects,
 * cached PSOs/functions/descriptors, and depth-stencil states. All returned
 * object pointers are borrowed for the lifetime of the owner/cache entry. */
int mglRenderCreatePipelineCacheOwner(int pso_dedup_enabled, int depth_stencil_cache_enabled, int binary_archive_enabled, MGLPipelineCacheOwner **owner_out);

void mglRenderDestroyPipelineCacheOwner(MGLPipelineCacheOwner **owner);

void mglRenderResetPipelineCacheOwner(MGLPipelineCacheOwner *owner);

/* Drop all C++ compute PSOs for a Program lifetime. Called before relink and
 * final Program destruction; safe before renderer initialization. */
void mglRenderInvalidateProgramPipelines(uint64_t program_instance);

int mglRenderGetPipelineCacheFlags(MGLPipelineCacheOwner *owner, int *pso_dedup_enabled_out, int *depth_stencil_cache_enabled_out, int *binary_archive_enabled_out);

void mglRenderDisablePipelineBinaryArchive(MGLPipelineCacheOwner *owner);

int mglRenderGetPipelineBinaryArchiveState(MGLPipelineCacheOwner *owner, int *enabled_out, int *present_out);

int mglRenderLoadPipelineBinaryArchive(MGLPipelineCacheOwner *owner, const char *cache_key, void *url, int archive_exists, int *reused_out, char *err, size_t errcap);

int mglRenderSerializePipelineBinaryArchive(MGLPipelineCacheOwner *owner, void *url, char *err, size_t errcap);

void mglRenderDiscardPipelineBinaryArchive(MGLPipelineCacheOwner *owner, const char *cache_key);

int mglRenderGetPipelineActiveState(MGLPipelineCacheOwner *owner, MGLRenderPipelineActiveState *state_out);

int mglRenderInvalidatePipelineActiveState(MGLPipelineCacheOwner *owner);

int mglRenderSetPipelineActiveObject(MGLPipelineCacheOwner *owner, void *pipeline_state);

int mglRenderActivatePipelineState(MGLPipelineCacheOwner *owner, const MGLRenderPipelineActiveState *state);

int mglRenderSetPipelineBlendState(MGLPipelineCacheOwner *owner, uint32_t attachment, const MGLRenderPipelineBlendState *state);

int mglRenderGetPipelineBlendState(MGLPipelineCacheOwner *owner, uint32_t attachment, MGLRenderPipelineBlendState *state_out);

int mglRenderLookupPipeline(MGLPipelineCacheOwner *owner, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], MGLRenderPipelineActiveState *state_out);

int mglRenderStorePipeline(MGLPipelineCacheOwner *owner, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], const MGLRenderPipelineActiveState *state, uint32_t *evicted_out);

/* Value-state descriptor cache. A hit returns the complete descriptor state. */
int mglRenderLookupPipelineDescriptorState(MGLPipelineCacheOwner *owner, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], MGLRenderPipelineDescriptorState *state_out);

int mglRenderStorePipelineDescriptorState(MGLPipelineCacheOwner *owner, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], const MGLRenderPipelineDescriptorState *state);

int mglRenderCreateRenderPipelineState(
    void *render_pipeline_descriptor,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Descriptor-based archive-aware creation used by the temporary ObjC
 * descriptor paths. Complete VS+FS pipelines query the archive first and are
 * added only on a miss. archive_hit_out is optional and receives 1 only when
 * the returned PSO came directly from the archive. */
int mglRenderCreateRenderPipelineStateWithArchive(
    void *render_pipeline_descriptor,
    void *binary_archive,
    void **pipeline_out,
    int *archive_hit_out,
    char *err,
    size_t errcap);

int mglRenderCreateRenderPipelineStateWithArchiveOwner(MGLPipelineCacheOwner *owner, void *render_pipeline_descriptor, void **pipeline_out, int *archive_hit_out, char *err, size_t errcap);

/* Creates a render PSO from value-state. Function and binary-archive pointers
 * are borrowed; binary_archive may be NULL. On success pipeline_out receives
 * an owned reference that must be released with mglAirRelease. */
int mglRenderCreateRenderPipelineFromState(
    void *vs_function,
    void *fs_function,
    const MGLRenderPipelineDescriptorState *state,
    void *binary_archive,
    void **pipeline_out,
    char *err,
    size_t errcap);

int mglRenderCreateRenderPipelineFromStateWithArchiveOwner(MGLPipelineCacheOwner *owner, void *vs_function, void *fs_function, const MGLRenderPipelineDescriptorState *state, void **pipeline_out, char *err, size_t errcap);

int mglRenderCreateComputePipelineState(void *function,
                                           void **pipeline_out,
                                           char *err,
                                           size_t errcap);

uint32_t mglRenderComputePipelineMaxTotalThreads(void *pipeline);

/* Create or reuse a compute PSO owned by the C++ renderer. function is the
 * actual MTLFunction selected by the caller, preserving AIR stage variants.
 * On success *pipeline_out is a +1 MTLComputePipelineState reference that the
 * ObjC bridge may consume with __bridge_transfer. */
int mglRenderGetOrCreateComputePipeline(
    void *function,
    uint64_t program_instance,
    uint64_t program_generation,
    uint32_t stage,
    int cache_enabled,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Lookup or create a renderer-lifetime auxiliary compute PSO. Passing a NULL
 * function performs lookup only and returns 1 on a cache miss. On success the
 * returned pipeline is an independent +1 reference. */
int mglRenderGetOrCreateAuxComputePipeline(
    void *function,
    uint32_t kind,
    uint64_t variant,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Aux compute PSO from the precompiled aux shader asset table. entry_name is
 * the metallib kernel name. On success *pipeline_out is a +1
 * MTL::ComputePipelineState reference. */
int mglRenderGetOrCreateAuxComputePipelineFromMetallib(
    const unsigned char *bytes,
    size_t size,
    uint64_t asset_hash,
    const char *entry_name,
    uint32_t kind,
    uint64_t variant,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Per-renderer-context binding dedup state. Metal objects stored in this
 * handle are retained by C++ and released on replacement, invalidation, or
 * destroy. Setter calls return 1 when encoded, 0 when deduplicated, and -1
 * for invalid arguments. */
void *mglRenderBindingCreate(uint32_t max_texture_slots);

void mglRenderBindingDestroy(MGLBindingState *binding_state);

void mglRenderBindingInvalidate(MGLBindingState *binding_state);

void mglRenderBindingSetValid(MGLBindingState *binding_state, int valid);

int mglRenderBindingGetValid(MGLBindingState *binding_state, uint32_t *valid_out);

int mglRenderBindingRecordVertexBuffer(MGLBindingState *binding_state, void *buffer, uint64_t offset, uint32_t index);

int mglRenderBindingRecordFragmentBuffer(MGLBindingState *binding_state, void *buffer, uint64_t offset, uint32_t index);

int mglRenderBindingInvalidateVertexBuffer(MGLBindingState *binding_state, uint32_t index);

int mglRenderBindingInvalidateFragmentBuffer(MGLBindingState *binding_state, uint32_t index);

int mglRenderBindingUpdateVertexBuffer(MGLBindingState *binding_state, void *buffer, uint64_t offset, uint32_t index);

int mglRenderBindingUpdateFragmentBuffer(MGLBindingState *binding_state, void *buffer, uint64_t offset, uint32_t index);

int mglRenderBindingGetBuffer(MGLBindingState *binding_state, uint32_t stage, uint32_t index, void **buffer_out, uint64_t *offset_out);

void mglRenderBindingOrVertexBufferMask(MGLBindingState *binding_state, uint32_t mask);

void mglRenderBindingOrFragmentBufferMask(MGLBindingState *binding_state, uint32_t mask);

void mglRenderBindingSetPipelineState(MGLBindingState *binding_state, void *pipeline_state);

void mglRenderBindingSetDepthStencilState(MGLBindingState *binding_state, void *depth_stencil_state);

int mglRenderBindingGetPipelineState(MGLBindingState *binding_state, void **pipeline_state_out);

int mglRenderBindingGetDepthStencilState(MGLBindingState *binding_state, void **depth_stencil_state_out);

void mglRenderBindingSetCullMode(MGLBindingState *binding_state, uint32_t mode);

void mglRenderBindingSetWinding(MGLBindingState *binding_state, uint32_t winding);

void mglRenderBindingSetDepthBias(MGLBindingState *binding_state, float bias, float clamp, float slope_scale);

void mglRenderBindingSetBlendColor(MGLBindingState *binding_state, float red, float green, float blue, float alpha);

int mglRenderBindingSetPipelineIfNeeded(MGLBindingState *binding_state, void *render_encoder, void *pipeline_state);

int mglRenderBindingSetDepthStencilIfNeeded(MGLBindingState *binding_state, void *render_encoder, void *depth_stencil_state);

int mglRenderBindingSetCullIfNeeded(MGLBindingState *binding_state, void *render_encoder, uint32_t mode);

int mglRenderBindingSetWindingIfNeeded(MGLBindingState *binding_state, void *render_encoder, uint32_t winding);

int mglRenderBindingSetDepthBiasIfNeeded(MGLBindingState *binding_state, void *render_encoder, float bias, float clamp, float slope_scale);

int mglRenderBindingSetBlendColorIfNeeded(MGLBindingState *binding_state, void *render_encoder, float red, float green, float blue, float alpha);

int mglRenderBindingSetPipelineIfNeededForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, void *pipeline_state);

int mglRenderBindingSetDepthStencilIfNeededForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, void *depth_stencil_state);

int mglRenderBindingSetCullIfNeededForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, uint32_t mode);

int mglRenderBindingSetWindingIfNeededForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, uint32_t winding);

int mglRenderBindingSetBlendColorIfNeededForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, float red, float green, float blue, float alpha);

int mglRenderBindingSetDepthBiasIfNeededForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, float depth_bias, float clamp, float slope_scale);

int mglRenderBindingSetViewport(MGLBindingState *binding_state, void *render_encoder, double origin_x, double origin_y, double width, double height, double znear, double zfar);

/* Array viewport binding (gl_ViewportIndex): viewports carries count
 * interleaved {x, y, w, h, znear, zfar} tuples, count <= 16. */
int mglRenderBindingSetViewports(MGLBindingState *binding_state, void *render_encoder, const double *viewports, uint64_t count);

int mglRenderBindingSetViewportsForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, const double *viewports, uint64_t count);

int mglRenderBindingSetScissor(MGLBindingState *binding_state, void *render_encoder, uint64_t x, uint64_t y, uint64_t width, uint64_t height);

int mglRenderBindingSetTriangleFill(MGLBindingState *binding_state, void *render_encoder, uint32_t mode);

int mglRenderBindingSetViewportForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, double origin_x, double origin_y, double width, double height, double znear, double zfar);

int mglRenderBindingSetScissorForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, uint64_t x, uint64_t y, uint64_t width, uint64_t height);

int mglRenderBindingSetTriangleFillForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, uint32_t mode);

int mglRenderBindingGetStats(MGLBindingState *binding_state, MGLRenderBindingStats *stats_out);

/* Compute encoder setter facade.  These entry points intentionally do not
 * retain resources: the command encoder owns the encoded references, matching
 * Objective-C Metal semantics.  Return 0 on success and -1 for bad inputs. */
int mglRenderSetComputePipelineState(void *compute_encoder,
                                        void *pipeline_state);

int mglRenderEncodeComputeBindingSnapshot(
    void *compute_encoder,
    const MGLRenderComputeBindingSnapshot *snapshot,
    char *err,
    size_t errcap);

int mglRenderAppendComputeBindingSnapshotToPlan(
    MGLRenderComputeExecutionPlan *plan,
    const MGLRenderComputeBindingSnapshot *snapshot,
    char *err,
    size_t errcap);

int mglRenderEncodeBindingSnapshot(
    void *render_encoder,
    const MGLRenderBindingSnapshot *snapshot,
    char *err,
    size_t errcap);

int mglRenderEncodeBindingSnapshotForRenderEncoderOwner(MGLRenderEncoderOwner *render_encoder_owner, const MGLRenderBindingSnapshot *snapshot, char *err, size_t errcap);

int mglRenderEncodeResourceBindingSnapshot(MGLBindingState *binding_state, void *render_encoder, const MGLRenderResourceBindingSnapshot *snapshot, char *err, size_t errcap);

int mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, const MGLRenderResourceBindingSnapshot *snapshot, char *err, size_t errcap);

bool mglRenderIsCullDistanceAttribName(const char *name);

const char *mglRenderVertexAttribName(const Program *program, uint32_t attrib);

void mglRenderAccumulateCullDistanceAttrib(MGLRenderCullDistanceLayout *layout,
                                           void *mtl_buffer,
                                           int64_t binding_offset,
                                           uint32_t stride,
                                           int64_t relativeoffset);

void mglRenderBindCullDistanceEmuSlots(MGLRenderEncoderOwner *encoder_owner, void *vertex_buffer, const MGLCullDistanceEmuParams *params);

int mglRenderSetRenderPipelineState(void *render_encoder,
                                       void *pipeline_state);

int mglRenderSetRenderPipelineStateForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *pipeline_state);

#ifdef __cplusplus
}
#endif

#endif
