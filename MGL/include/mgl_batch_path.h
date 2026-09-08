/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_batch_path.h — pure-C batch path selection (O2.1).
 *
 * Decision tree for DIRECT / MDI / STREAM_MERGE / ICB.  No Metal.
 * Return codes match MGLBatchPath in draw_command.h.
 * ObjC scheduleDrawBatch fills inputs and casts the result.
 */

#ifndef MGL_BATCH_PATH_H
#define MGL_BATCH_PATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Keep in sync with MGLBatchPath / MGL_MDI_MIN_BATCH_SIZE (draw_command.h). */
enum {
    MGL_BATCH_SELECT_DIRECT = 0,
    MGL_BATCH_SELECT_MDI = 1,
    MGL_BATCH_SELECT_STREAM_MERGE = 2,
    MGL_BATCH_SELECT_ICB = 3,
    MGL_BATCH_SELECT_MDI_MIN_COMMANDS = 2
};

typedef struct MGLBatchSelectInputs {
    uint32_t command_count;
    uint8_t sampler_snapshots_mixed;
    uint8_t uses_cull_distance;
    uint8_t stream_merged;
    uint8_t has_dynamic_uniform_bindings;
    uint8_t has_dynamic_vertex_bindings;
    uint8_t has_dynamic_texture_bindings;
    uint8_t mdi_compatible;
    uint8_t uses_elements;
    uint8_t polygon_mode_point;
    uint8_t primitive_restart;
    uint8_t primitive_type; /* 0xFF = unset / invalid for ICB */
    uint8_t icb_os_supported;
    uint8_t enable_icb;  /* from mgl_batch_icb_config().enable */
    uint8_t disable_icb; /* from mgl_batch_icb_config().disable */
    uint8_t disable_mdi; /* MGL_DISABLE_MDI */
} MGLBatchSelectInputs;

int mgl_batch_select_path(const MGLBatchSelectInputs *in);

/*
 * O2.4: unified ICB gate for batch path selection AND
 * MTLRenderPipelineDescriptor.supportIndirectCommandBuffers.
 * Prefer MGL_ENABLE_ICB / MGL_DISABLE_ICB. Legacy ENABLE_ICB_BATCH or
 * ENABLE_ICB_PIPELINES either enable; DISABLE_ICB_BATCH (or DISABLE_ICB)
 * hard-disables. Avoids ENABLE_BATCH / ENABLE_PIPELINES split.
 */
typedef struct MGLBatchIcbConfig {
    uint8_t enable;  /* opt-in want ICB */
    uint8_t disable; /* hard off */
} MGLBatchIcbConfig;

MGLBatchIcbConfig mgl_batch_icb_config(void);

/* True when pipelines should set supportIndirectCommandBuffers /
 * when batch ICB path may be selected (still needs OS + other gates). */
int mgl_batch_icb_support_indirect_command_buffers(void);


/* Fill batch-derived select inputs (caller still sets cull/os/env/polygon). */
void mgl_batch_fill_select_inputs_from_batch_flags(
    uint32_t command_count, int sampler_snapshots_mixed, int stream_merged,
    int has_dynamic_uniform_bindings, int has_dynamic_vertex_bindings,
    int has_dynamic_texture_bindings, int mdi_compatible, int uses_elements,
    uint8_t primitive_type, MGLBatchSelectInputs *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_PATH_H */
