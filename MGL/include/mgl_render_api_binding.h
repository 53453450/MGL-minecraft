/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_BINDING_H
#define MGL_RENDER_API_BINDING_H

/* Declarations for the binding slice of the renderer facade.
 * Value layouts are defined here; types of other subsystems that are only
 * passed by pointer are incomplete (mgl_render_fwd.h). */

#include <GL/glcorearb.h>
#include "mgl_render_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- renderer-side program / framebuffer-binding helpers ------------------
 *
 * These three are DEFINED IN C (mgl_renderer_entries.c, mgl_draw_support.c)
 * but their only declarations used to live in the ObjC private headers
 * (MGLRenderer+RenderPass_Private.h, MGLRenderer+Draw_Private.h), which C
 * translation units must not include.  C callers therefore either hand-declared
 * them locally (mglResolveProgramForStageFromState had four such local externs)
 * or relied on a duplicate declaration that this header used to carry.
 *
 * The declaration belongs here: this header is the C-facing renderer facade.
 * The duplicates in the private headers are the ones that must go, not these. */

/* Program bound to a stage by GL state: GL_CURRENT_PROGRAM when one is bound,
 * otherwise the separable pipeline's stage program. */
Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);

/* Identity of the program the current draw would use (restored/scheduled), as
 * the trace log and pipeline-cache keys see it. */
GLuint mglCurrentRenderProgramKey(GLMContext ctx);

/* Re-sync the framebuffer binding names after a hashtable swap. */
void mglRendererSyncFramebufferBindingNames(GLMContext ctx);

/* Restore the program/pipeline pair a state key names. */
/* single source: mglRestoreProgramPipelinePair (C TUs and the ObjC private headers both
 * resolve it from here; the private copies were duplicates) */
void mglRestoreProgramPipelinePair(GLMContext ctx, GLuint programName,
                                   GLuint pipelineName);

enum {
    MGL_RENDER_AIR_PROGRAM_BOUND = 0,
    MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE = 1,
    MGL_RENDER_AIR_PROGRAM_ERROR = -1,
};

/* C1: TextureRepackDepthPlanes -> mgl_readback_policy.h */

/* Expands legacy packed GL formats into RGBA8. Returns a malloc-owned buffer,
 * or NULL for invalid input, unsupported formats, or size overflow. */
/* stage-binding copy-back entry (C-ABI mirror of the
 * ObjC MGLStageBindingCopyBack — the ObjC side bridges the buffer refs). */
typedef struct MGLRenderCopyBackEntry_t {
    const void *temporary;        /* MTL::Buffer* */
    const void *destination;      /* MTL::Buffer* */
    const void *destination_buffer; /* GL Buffer* (CPU prefix sync) */
    uint64_t destination_offset;
    uint64_t length;
} MGLRenderCopyBackEntry;

uint32_t mglRenderCollectCopyBackEntries(const MGLRenderCopyBackEntry *slots,
                                         uint32_t slot_count,
                                         MGLRenderCopyBackEntry *out,
                                         uint32_t out_cap);

/* Synchronize the written CPU prefix of each entry's GL destination buffer
 * (guards + memmove; the Metal contents pointer is read via the
 * destination buffer).  Returns 0, or -1 with *failed_index_out set. */
int mglRenderCopyBackCPUPrefix(
    const MGLRenderCopyBackEntry *entries,
    uint32_t count,
    uint32_t *failed_index_out);

/* index-range scan ignoring primitive-restart markers
 * — computes min/max over the byte stream (BYTE/SHORT/INT width), skipping
 * the restart value.  Pure CPU; matches mglScanIndexRangeIgnoringRestart.
 * Returns 0 on success (with *out_valid = 1 if at least one non-restart
 * index was seen), -1 on bad args. */
/* Convert a restart-aware index span into the inclusive [first, first+count)
 * vertex range used by cull-distance capture. Returns 0 on success. */
/* prepared (Metal-side) byte offset for a GL element
 * buffer — GL_UNSIGNED_BYTE indices are expanded to UInt16 so the offset
 * doubles, other types pass through.  Matches mglComputePreparedIndexByteOffset.
 * Returns 0 on success, -1 on overflow / bad args. */
/* baseByteOffset + firstElement * indexStride with
 * overflow checks.  Matches mglComputeIndexByteOffset.  Returns 0 on success,
 * -1 on bad args / overflow. */
/* GL index element byte size (BYTE=1, SHORT=2, INT=4).
 * Matches mglGLIndexElementSize.  Returns 0 for unknown type. */

/* read a single GL index value from a byte buffer at
 * `element_index` (elem_width 1/2/4).  Matches mglReadGLIndexValue; returns 0
 * for NULL buffer or unknown width. */
/* GL vertex-attribute component size in bytes (1/2/4/8).
 * Matches mglVertexAttribComponentSize.  Returns 0 for unknown. */

/* total bytes for a vertex-attribute element (type x
 * size), with special handling for packed 10_10_10_2 formats.  Matches
 * mglVertexAttribElementBytes.  Returns 0 for unknown / zero size. */

enum {
    MGL_ATTRIB_SPAN_OK = 0,
    MGL_ATTRIB_SPAN_NEGATIVE_RELATIVE = -1,
    MGL_ATTRIB_SPAN_OVERFLOW = -2,
};

enum {
    MGL_ATTRIB_FETCH_OK = 0,
    MGL_ATTRIB_FETCH_BAD_FORMAT = 1,
    MGL_ATTRIB_FETCH_OVERFLOW = 2,
    MGL_ATTRIB_FETCH_OOB = 3,
};

typedef struct MGLRenderAttribFetchPlan {
    uint32_t status;
    uint64_t elem_bytes;
    uint64_t stride;
    uint64_t rel_offset;
    uint64_t byte_start;
    uint64_t byte_end;
} MGLRenderAttribFetchPlan;

int mglRenderRepairBlendSrcFactor(uint32_t *value);

int mglRenderRepairBlendDstFactor(uint32_t *value);

int mglRenderRepairBlendEquation(uint32_t *value);

uint32_t mglRenderRepairDepthFunc(uint32_t func);

uint32_t mglRenderRepairStencilFunc(uint32_t func);

/* does GL primitive mode produce polygonal primitives
 * (triangles/quads) subject to glPolygonMode point/line emulation?  Matches
 * mglDrawModeProducesPolygons.  Returns 1/0. */

/* does `mode` with `indexCount` vertices produce at
 * least one drawable segment (point/line/triangle/quad)?  Matches
 * mglPrimitiveModeHasDrawableSegment.  Returns 1/0. */
/* total triangle index count for `source_vertex_count`
 * vertices arranged as quads (4/quad -> 6 indices).  Matches
 * mglQuadTriangleIndexCount; returns 0 on overflow. */
/* Align vertex stride to 4; matches mglAlignVertexStrideForMetal. */
/* double-attrib size -> MTLVertexFormat value; matches mglDoubleVertexAttribFloatFormat. */
/* Integer attrib signedness mismatch -> Int/UInt MTLVertexFormat ABI value.
 * Returns MTLVertexFormatInvalid when no CPU conversion is required. */
enum {
    MGL_ATTRIB_CONV_NONE = 0,
    MGL_ATTRIB_CONV_DOUBLE = 1,
    MGL_ATTRIB_CONV_INT_TO_FLOAT = 2,
    MGL_ATTRIB_CONV_FIXED = 3,
    MGL_ATTRIB_CONV_UINT_1010102 = 4,
    MGL_ATTRIB_CONV_UINT_10F11F11F = 5,
    MGL_ATTRIB_CONV_INTEGER_SIGN = 6,
    MGL_ATTRIB_CONV_REV2101010 = 7,
};

uint32_t mglRenderGLTypeSizeToVertexFormat(uint32_t type, uint32_t size,
                                           int normalized);

typedef struct MGLRenderVertexAttribResolve_t {
    int use_binding_table;   /* bindingIndex < limit && binding has buffer */
    int64_t binding_offset;  /* table offset, or attrib binding_offset */
    uint32_t stride;         /* table stride, or attrib stride */
    uint32_t divisor;
} MGLRenderVertexAttribResolve;

enum {
    MGL_RENDER_PIPELINE_CACHE_KEY_WORDS = 7,
    MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS = 8,
};

#define MGL_RENDER_PIPELINE_CACHE_KEY_WORDS_DEFINED 1

/* 1 if the default Metal device is Apple Paravirtual (hosted CI VMs). */


typedef struct MGLRenderPipelineActiveState_t {
    void *pipeline_state;
    void *vertex_function;
    void *fragment_function;
    uint32_t color0_format;
    uint32_t depth_format;
    uint32_t stencil_format;
    uint32_t program_name;
} MGLRenderPipelineActiveState;

typedef struct MGLRenderPipelineBlendState_t {
    uint32_t source_rgb_factor;
    uint32_t destination_rgb_factor;
    uint32_t source_alpha_factor;
    uint32_t destination_alpha_factor;
    uint32_t rgb_operation;
    uint32_t alpha_operation;
    uint32_t color_write_mask;
} MGLRenderPipelineBlendState;

enum {
    MGL_RENDER_BINDING_STAGE_VERTEX = 0,
    MGL_RENDER_BINDING_STAGE_FRAGMENT = 1,
};

enum {
    MGL_RENDER_BINDING_VERTEX_TEXTURE = 0,
    MGL_RENDER_BINDING_FRAGMENT_TEXTURE = 1,
    MGL_RENDER_BINDING_VERTEX_SAMPLER = 2,
    MGL_RENDER_BINDING_FRAGMENT_SAMPLER = 3,
    MGL_RENDER_BINDING_VIEWPORT = 4,
    MGL_RENDER_BINDING_SCISSOR = 5,
    MGL_RENDER_BINDING_TRIANGLE_FILL = 6,
    MGL_RENDER_BINDING_SETTER_COUNT = 7,
};

typedef struct MGLRenderBindingStats {
    uint64_t emitted[MGL_RENDER_BINDING_SETTER_COUNT];
    uint64_t skipped[MGL_RENDER_BINDING_SETTER_COUNT];
} MGLRenderBindingStats;

/* Exact, program-based replacement for the retired shader-source scans: reads
 * the per-stage builtin mask published by the frontend. */
int mglRenderFragmentNeedsPerSampleMSValues(const Program *program);

void mglRenderFillFragCoordSlot(int use_fragcoord, int use_sample,
                                uint32_t pass_height, int lower_left,
                                uint32_t num_samples, uint32_t sample_buffers,
                                int ms_loop, uint32_t forced_sample_id,
                                int point_coord_lower_left, float out[5]);

void mglRenderClampLodBiasArray(float *bias, uint32_t count, float biasmax);

/* Per-draw binding snapshot. GL-side deduplication and accounting determine
 * which bindings are emitted; the backend replays the resulting operation list. */
#define MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS 32u

/* One per-draw binding op: kind 0 = set buffer (buffer == NULL clears the
 * slot, matching mglRenderSetRenderBuffer with a nil resource), kind 1 =
 * set bytes (bytes borrowed — valid until EncodeBindingSnapshot returns).
 * The op list keeps the exact per-stage emit order, including interleaved
 * buffer/bytes/clear ops on the same slot. */
typedef struct MGLRenderBindingOp_t {
    uint32_t kind;      /* 0 = buffer, 1 = bytes */
    uint32_t index;     /* Metal slot */
    uint64_t offset;    /* kind 0: byte offset */
    void *buffer;       /* kind 0: borrowed MTL::Buffer* (NULL = clear) */
    const void *bytes;  /* kind 1: borrowed byte pointer */
    uint32_t length;    /* kind 1: byte length */
} MGLRenderBindingOp;

typedef struct MGLRenderBindingSnapshot_t {
    uint32_t vertex_op_count;
    MGLRenderBindingOp
        vertex_ops[MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS];
    uint32_t fragment_op_count;
    MGLRenderBindingOp
        fragment_ops[MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderBindingSnapshot;

/* Texture/sampler bindings are collected separately from buffer/bytes ops.
 * Resource resolution may rotate the render encoder while uploading a
 * texture, so ObjC submits this snapshot in ordered segments at those
 * boundaries.  The C++ binding owner remains authoritative for dedup state
 * and retains every resource that becomes current. */
#define MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS 512u

enum {
    MGL_RENDER_RESOURCE_BINDING_TEXTURE = 0,
    MGL_RENDER_RESOURCE_BINDING_SAMPLER = 1,
};

typedef struct MGLRenderResourceBindingOp_t {
    uint32_t kind;
    uint32_t index;
    void *resource; /* borrowed MTL::Texture* or MTL::SamplerState* */
} MGLRenderResourceBindingOp;

typedef struct MGLRenderResourceBindingSnapshot_t {
    uint32_t vertex_op_count;
    MGLRenderResourceBindingOp
        vertex_ops[MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS];
    uint32_t fragment_op_count;
    MGLRenderResourceBindingOp
        fragment_ops[MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderResourceBindingSnapshot;

int mglRenderSetRenderBytes(void *render_encoder,
                               const void *bytes,
                               size_t length,
                               uint32_t stage,
                               uint32_t index);

int mglRenderSetRenderDepthStencilState(void *render_encoder,
                                           void *depth_stencil_state);

int mglRenderSetRenderViewport(void *render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar);

int mglRenderSetRenderScissor(void *render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height);

int mglRenderSetDepthClipMode(void *render_encoder, uint32_t mode);

int mglRenderSetStencilReferenceValues(void *render_encoder,
                                          uint32_t front_reference,
                                          uint32_t back_reference);

int mglRenderSetRenderBytesForOwner(MGLRenderEncoderOwner *render_encoder_owner, const void *bytes, size_t length, uint32_t stage, uint32_t index);

int mglRenderSetRenderDepthStencilStateForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *depth_stencil_state);

int mglRenderSetRenderViewportForOwner(MGLRenderEncoderOwner *render_encoder_owner, double origin_x, double origin_y, double width, double height, double znear, double zfar);

int mglRenderSetRenderScissorForOwner(MGLRenderEncoderOwner *render_encoder_owner, uint64_t x, uint64_t y, uint64_t width, uint64_t height);

int mglRenderSetDepthClipModeForOwner(MGLRenderEncoderOwner *render_encoder_owner, uint32_t mode);

int mglRenderSetStencilReferenceValuesForOwner(MGLRenderEncoderOwner *render_encoder_owner, uint32_t front_reference, uint32_t back_reference);

int mglRenderUseRenderResource(void *render_encoder,
                                  void *resource,
                                  uint32_t usage,
                                  uint32_t stages);

int mglRenderExecuteIndirectCommands(void *render_encoder,
                                        void *indirect_buffer,
                                        uint64_t location,
                                        uint64_t length);

int mglRenderReplayBatchDrawsForRenderEncoderOwner(MGLRenderEncoderOwner *render_encoder_owner, const MGLRenderReplayBatch *batch, char *err, size_t errcap);

int mglRenderUseRenderResourceForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *resource, uint32_t usage, uint32_t stages);

int mglRenderExecuteIndirectCommandsForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *indirect_buffer, uint64_t location, uint64_t length);


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

int mglRenderAttribNeedsConvertedMetalStream(uint32_t type, int integer,
                                             uint32_t size, int normalized);

int mglRenderAttribColorUByteNeedsNormalize(uint32_t type, uint32_t size,
                                            int already_norm);

uint32_t mglRenderAttribEffectiveNormalized(uint32_t already, int needs);

double mglRenderDecodeVertexAttribComponent(const uint8_t *src, uint32_t type,
                                            int normalized, uint32_t component);

int mglRenderBufferMapIsBaseBinding(uint32_t attribute_mask);

int mglRenderClientBindingInRange(uint32_t binding, uint32_t max);

int mglRenderBaseBindingTooSmall(int64_t range, uint64_t reflected);

int mglRenderAttribOffsetsValid(int64_t binding_offset,
                                int64_t relativeoffset);

void mglRenderPlanCurrentAttribFromShader(uint32_t shader_gl_type,
                                          uint32_t *type_out,
                                          uint32_t *size_out, int *integer_out);

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

/* Per-encoder sampled/image bind gate (see BindingState::sampledNeedsBind). */
void mglRenderBindingMarkSampledNeedsBind(MGLBindingState *binding_state);
void mglRenderBindingClearSampledNeedsBind(MGLBindingState *binding_state);
int mglRenderBindingSampledNeedsBind(MGLBindingState *binding_state);

/* Sampler warmup latch scoped to this binding state (not process-static). */
int mglRenderBindingWarmupUnchanged(MGLBindingState *binding_state,
                                    const void *vertex_program,
                                    const void *fragment_program,
                                    const uint32_t *vertex_mask,
                                    const uint32_t *fragment_mask);
void mglRenderBindingStoreWarmup(MGLBindingState *binding_state,
                                 const void *vertex_program,
                                 const void *fragment_program,
                                 const uint32_t *vertex_mask,
                                 const uint32_t *fragment_mask);

/* Borrowed MTL sampler cache keyed by (unit, tex, gl_sampler, params_sig). */
void *mglRenderBindingLookupSampledSampler(MGLBindingState *binding_state,
                                           uint32_t unit, const void *tex,
                                           const void *gl_sampler,
                                           uint64_t params_sig);
void mglRenderBindingStoreSampledSampler(MGLBindingState *binding_state,
                                         uint32_t unit, const void *tex,
                                         const void *gl_sampler,
                                         uint64_t params_sig,
                                         void *mtl_sampler);

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

/* Array scissor binding: rects carries count interleaved {x, y, w, h}.
 * Metal requires the scissor count to match the viewport count. */
int mglRenderBindingSetScissors(MGLBindingState *binding_state, void *render_encoder, const uint64_t *rects, uint64_t count);

int mglRenderBindingSetTriangleFill(MGLBindingState *binding_state, void *render_encoder, uint32_t mode);

int mglRenderBindingSetViewportForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, double origin_x, double origin_y, double width, double height, double znear, double zfar);

int mglRenderBindingSetScissorForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, uint64_t x, uint64_t y, uint64_t width, uint64_t height);

int mglRenderBindingSetScissorsForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, const uint64_t *rects, uint64_t count);

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
