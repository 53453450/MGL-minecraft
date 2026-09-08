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
int mglDrawGsSkipRaster(int xfb_active, int rasterizer_discard);
int mglDrawGsPassthroughRasterReady(int state_ready, int has_encoder,
                                    int raster_empty, int fully_culled);
uint64_t mglDrawGsClampXFBCopy(uint64_t written, uint64_t remaining);
int mglDrawGsXFBCopyReady(int has_dst, uint32_t stride, uint64_t written);

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

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_GS_H */
