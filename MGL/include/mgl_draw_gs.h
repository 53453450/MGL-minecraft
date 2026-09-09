/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#ifndef MGL_DRAW_GS_H
#define MGL_DRAW_GS_H

#include "glcorearb.h"
#include "glm_context.h"
#include "mgl_air_gs_abi.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool mglDrawGsInputModeAccepts(GLenum gsMode, GLenum drawMode);

void mglDrawGsNormalizeTopology(Program *gs, GLenum *in_mode, GLenum *out_mode,
                                uint32_t *out_primitive);

void mglDrawGsFillGatherParams(int indexed, uint32_t count, uint32_t first,
                               uint32_t gather_max_index, uint32_t primitives,
                               MGLAIRGSGatherParams *out);

enum {
    MGL_GS_INPUT_PENDING_TES = 0,
    MGL_GS_INPUT_CAPTURE_INDEXED = 1,
    MGL_GS_INPUT_CAPTURE_ARRAY = 2,
};

typedef struct MGLGsInputSourcePlan {
    uint32_t kind;
    uint32_t input_offset;
    uint32_t pending_stride;
} MGLGsInputSourcePlan;

void mglDrawGsPlanInputSource(int pending_active, int has_pending,
                              uint32_t pending_offset, uint32_t pending_stride,
                              int indexed, MGLGsInputSourcePlan *out);
uint32_t mglDrawGsMaxVerticesOut(uint32_t geometry_vertices_out);

uint32_t mglDrawGsResolveStageInStride(Program *vs, Program *tes,
                                       uint32_t pending_stride);

bool mglDrawGsGatherTopology(const uint8_t *indexBytes, GLenum indexType,
                             GLsizei count, GLint first, bool indexed,
                             bool restartEnabled, uint32_t restartIndex,
                             GLenum mode, uint32_t **outGather,
                             uint32_t *outGatherCount,
                             uint32_t *outPrimitiveCount, uint32_t *outMaxIndex);

typedef struct MGLGsPassthroughEncodeState {
    void *encoder_owner;
    void *output_buffer;
    void *counts_buffer;
    uint32_t output_primitive;
    uint32_t work_item_count;
    uint32_t records_per_primitive;
    uint32_t output_stride;
    uint32_t counts_record_bytes;
} MGLGsPassthroughEncodeState;

void mglDrawGsEncodePassthrough(const MGLGsPassthroughEncodeState *state);

typedef struct MGLRenderComputeExecutionPlan_t MGLRenderComputeExecutionPlan;
typedef struct MGLAIRGSXFBScatterParams MGLAIRGSXFBScatterParams;

void mglDrawGsFillLocationMap(Program *gs, Program *vs, Program *tes,
                              uint32_t loc_map[32]);

void mglDrawGsPresetCounts(void *counts, uint32_t work_item_count);

uint32_t mglDrawGsFillXFBScatterParams(Program *gs,
                                       MGLAIRGSXFBScatterParams *out);

bool mglDrawGsAppendCoreBindings(MGLRenderComputeExecutionPlan *plan,
                                 void *input, uint64_t input_offset,
                                 void *output, void *counts,
                                 void *gather_or_counts, void *xfb_capture,
                                 void *xfb_meta, void *xfb_vis_or_counts,
                                 const void *gparams, uint32_t gparams_bytes);

typedef struct MGLGsComputeLayout {
    uint32_t work_item_count;
    uint32_t records_per_primitive;
    uint32_t expanded_vertices;
    uint32_t output_stride;
    uint64_t output_bytes;
    uint64_t counts_bytes;
} MGLGsComputeLayout;

bool mglDrawGsComputeLayout(Program *gs, uint32_t primitive_count,
                            uint32_t instance_count, GLenum output_mode,
                            MGLGsComputeLayout *out);

void mglDrawGsExclusivePrefixSum(const uint32_t *vis, uint32_t *offsets,
                                 uint32_t work_item_count,
                                 uint32_t buffer_count);

bool mglDrawGsFillXFBScatterPlan(MGLRenderComputeExecutionPlan *plan,
                                 void *pipeline, const void *scatter_params,
                                 uint32_t params_bytes, void *vis, void *offsets,
                                 void *stage_out, void *xfb, void *written,
                                 uint32_t work_item_count);

typedef struct MGLGsXFBBufferBinding {
    uint8_t bound;
    int64_t slot_offset;
    uint64_t session_offset;
    uint64_t visible_bytes;
} MGLGsXFBBufferBinding;

typedef struct MGLGsXFBBufferDest {
    uint32_t remaining;
    uint32_t dst_offset;
    uint32_t cap_bytes;
    uint32_t phys_base;
    uint8_t valid;
} MGLGsXFBBufferDest;

typedef struct MGLGsXFBDestPlan {
    MGLGsXFBBufferDest buffers[MGL_AIR_GS_MAX_STREAMS];
    uint32_t phys_total;
} MGLGsXFBDestPlan;

/* GL 4.6 §13.2.4 store window from the bound offset. Capacity is the
 * remaining visible bytes, clipped to the expansion's max capture. */
void mglDrawGsPlanXFBDestinations(MGLAIRGSXFBScatterParams *params,
                                  uint32_t buffer_count,
                                  uint32_t work_item_count,
                                  uint32_t expanded_vertices,
                                  const MGLGsXFBBufferBinding bindings[MGL_AIR_GS_MAX_STREAMS],
                                  MGLGsXFBDestPlan *out);

void mglDrawGsFillXFBScatterRuntime(MGLAIRGSXFBScatterParams *params,
                                    uint32_t buffer_count,
                                    uint32_t work_item_count,
                                    uint32_t output_stride,
                                    uint32_t records_per_primitive,
                                    uint32_t output_primitive);
uint64_t mglDrawGsXFBVisBytes(uint32_t work_item_count);
int mglDrawGsXFBActive(int has_xfb, int active, int paused);
uint32_t mglDrawGsVerticesPerPrimitive(uint32_t output_primitive);
uint32_t mglDrawGsStreamCount(uint32_t geometry_stream_count);
GLenum mglDrawGsLastDrawMode(uint32_t output_primitive);
void mglDrawGsFillXFBDestForMeta(const uint32_t *cap_bytes,
                                 const uint32_t *phys_base, uint32_t count,
                                 MGLGsXFBDestPlan *out);
void mglDrawGsClearXFBMetaIfNoCapture(int has_capture, MGLAIRGSXFBMeta *meta);
int mglDrawGsNeedCPUVisibility(int xfb_active, int has_query);
uint64_t mglDrawGsQueryWritten(uint32_t output_primitive, uint32_t buffer0_stride,
                               uint64_t buffer0_written);
uint32_t mglDrawGsClampStreamCount(uint32_t stream_count);
uint64_t mglDrawGsIndexedStreamWritten(int xfb_active, uint64_t buffer_written,
                                       uint64_t stride);
uint64_t mglDrawGsStream0QueryWritten(int xfb_active, uint64_t written_stream0);
int mglDrawGsSkipRaster(int xfb_active, int rasterizer_discard);
int mglDrawGsPassthroughRasterReady(int state_ready, int has_encoder,
                                    int raster_empty, int fully_culled);
uint64_t mglDrawGsClampXFBCopy(uint64_t written, uint64_t remaining);
int mglDrawGsXFBCopyReady(int has_dst, uint32_t stride, uint64_t written);
void mglDrawGsInitDefaultTopology(GLenum *in_mode, GLenum *out_mode,
                                  uint32_t *out_prim);

void mglDrawGsFillXFBMetaFromDest(const MGLAIRGSXFBScatterParams *params,
                                  const MGLGsXFBDestPlan *dest,
                                  MGLAIRGSXFBMeta *out);

/* PRIMITIVES_GENERATED from kernel emit/meta, not the allocated expansion. */
uint64_t mglDrawGsReduceGeneratedPrimitives(GLenum output_mode,
                                            uint32_t work_item_count,
                                            uint32_t max_vertices,
                                            const uint32_t *counts,
                                            const MGLAIRGSXFBMeta *meta);

uint64_t mglDrawGsReduceBufferWritten(const uint32_t *written,
                                      uint32_t work_item_count,
                                      uint32_t buffer_index);
uint32_t mglDrawGsPassthroughDeclType(uint32_t output_type, uint32_t fs_type,
                                      int names_match);
int mglDrawGsStageShouldBlockDraw(int stage, uint32_t gs_route,
                                  const void *metallib_bytes,
                                  uint32_t metallib_size);


/* O1.4+/A1: GS draw host runner. Topology/gather/input + Metal expansion in C++;
 * ObjC supplies thin MTL HostOps (PSO materialize / bind / blit / encode).
 * A1: RunDraw calls ExecuteMetalExpansion directly via nested metal_ops. */
typedef struct MGLRenderCopyBackEntry_t MGLRenderCopyBackEntry;
typedef struct MGLRenderComputeExecutionResult_t MGLRenderComputeExecutionResult;

typedef struct MGLGsMetalExpansionHostOps {
    void *renderer;
    void *(*create_buffer)(void *renderer, uint64_t length);
    void *(*create_buffer_with_bytes)(void *renderer, const void *bytes,
                                      uint64_t length);
    void *(*buffer_contents)(void *buffer);
    uint64_t (*buffer_length)(void *buffer);
    void (*release)(void *obj);
    int (*ensure_command_buffer)(void *renderer);
    int (*bind_draw_textures)(void *renderer, GLMContext ctx);
    void *(*mtl_for_buffer)(void *renderer, Buffer *buf);
    /* Fills `plan` with *borrowed* MTL pointers.  Objects created purely to
     * fill the plan (isolated stage-binding buffers, runtime-array-size
     * constant buffers, fallback samplers, storage-image views) are kept alive
     * by a host-owned keep-alive set, which is handed back through
     * `temporaries_out` as a +1 CF reference (NULL when nothing was needed).
     * The caller MUST retain it until the plan has been encoded+dispatched and
     * then release it; otherwise the plan encodes dangling pointers. */
    int (*fill_compute_bindings)(void *renderer, GLMContext ctx,
                                 MGLRenderComputeExecutionPlan *plan,
                                 MGLRenderCopyBackEntry *copybacks,
                                 uint32_t copybacks_cap,
                                 uint32_t *copybacks_count,
                                 void **temporaries_out);
    void *(*command_buffer_owner)(void *renderer);
    void *(*recovery_owner)(void *renderer);
    void (*note_device_reset)(void *renderer);
    void (*set_expansion)(void *renderer, Program *program, int active,
                          GLenum last_draw_mode);
    void (*mark_cb_has_work)(void *renderer);
    void *(*begin_blit)(void *renderer);
    void (*blit_copy)(void *blit, void *src, uint64_t src_off, void *dst,
                      uint64_t dst_off, uint64_t bytes);
    void (*end_blit)(void *blit);
    int (*process_gl_state)(void *renderer);
    int (*encoder_has_current)(void *renderer);
    int (*raster_empty)(void *renderer);
    int (*fully_culled)(void *renderer, GLenum mode);
    void (*apply_polygon_offset)(void *renderer, GLenum mode);
    /* A1: clear loops live in C++; ObjC only rebinds fragment resources. */
    void *(*binding_state_owner)(void *renderer);
    int (*rebind_fragment_after_gs)(void *renderer, GLMContext ctx);
    void *(*encoder_owner)(void *renderer);
    void (*flush_command_buffer)(void *renderer, int wait);
    void (*record_queries)(GLMContext ctx, uint64_t generated, uint64_t written,
                           int xfb_active, const MGLAIRGSXFBMeta *meta,
                           uint32_t stream_count, const uint64_t *buffer_written,
                           const uint64_t *buffer_stride,
                           uint64_t geometry_invocations);
    void (*gpu_capture_start)(void *renderer);
    void (*gpu_capture_stop)(void *renderer);
    void (*set_vertex_buffer)(void *encoder_owner, void *buffer, uint64_t offset,
                              uint32_t index);
    void (*draw_primitives)(void *encoder_owner, uint32_t output_primitive,
                            uint32_t vertex_start, uint32_t vertex_count,
                            uint32_t instance_count, uint32_t base_instance);
    void (*draw_primitives_indirect)(void *encoder_owner,
                                     uint32_t output_primitive, void *counts,
                                     uint64_t offset);
    void (*log_diag)(const char *msg);
} MGLGsMetalExpansionHostOps;

/* C++ owns GS Metal expansion; ObjC fills nested metal_ops HostOps only. */
int mglDrawGsExecuteMetalExpansion(
    GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLenum indexType,
    const void *indices, GLint baseVertex, GLsizei instanceCount,
    GLuint baseInstance, const char *label, Program *program,
    GLenum gs_input_mode, GLenum gs_output_mode, uint32_t output_primitive,
    int indexed, void *gather_buf, const void *gparams, uint32_t gparams_bytes,
    const MGLGsComputeLayout *layout, void *input, uint64_t input_offset,
    Program *capture_vs, Program *capture_tes, uint32_t pending_stride,
    const MGLGsMetalExpansionHostOps *ops);

typedef struct MGLGsDrawHostOps {
    void *renderer;
    int (*bind_mtl_program)(void *renderer, Program *program);
    int (*ensure_passthrough)(void *renderer, Program *program,
                              uint32_t output_primitive);
    int (*process_buffer)(void *renderer, Buffer *buf);
    void *(*capture_array)(void *renderer, GLMContext ctx, GLint first,
                           GLsizei count, GLsizei instanceCount,
                           GLuint baseInstance, uint64_t *out_offset);
    void *(*capture_indexed)(void *renderer, GLMContext ctx, void *index_mtl,
                             GLenum indexType, uint64_t index_offset,
                             GLsizei count, GLint baseVertex,
                             GLsizei instanceCount, GLuint baseInstance,
                             uint32_t maxIndex, uint64_t *out_offset);
    void *(*create_buffer_with_bytes)(void *renderer, const void *bytes,
                                      uint64_t length);
    int (*pending_gs_input_active)(void *renderer);
    void *(*pending_gs_input)(void *renderer);
    uint32_t (*pending_gs_input_offset)(void *renderer);
    uint32_t (*pending_gs_input_stride)(void *renderer);
    /* A1: nested Metal expansion HostOps; C++ calls
     * mglDrawGsExecuteMetalExpansion directly (no ObjC middle-man). */
    const MGLGsMetalExpansionHostOps *metal_ops;
    void (*dispatch_error)(GLMContext ctx, const char *where, GLenum err);
    void (*log_diag)(const char *msg);
} MGLGsDrawHostOps;

/* Returns 1 if GS handled the draw, 0 if N/A. */
int mglDrawGsRunDraw(GLMContext ctx, GLenum mode, GLint first, GLsizei count,
                     GLenum indexType, const void *indices, GLint baseVertex,
                     GLsizei instanceCount, GLuint baseInstance,
                     const char *label, const MGLGsDrawHostOps *ops);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_GS_H */
