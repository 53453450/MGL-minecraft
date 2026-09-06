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

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool mglDrawGsInputModeAccepts(GLenum gsMode, GLenum drawMode);

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

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_GS_H */
