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
 * mgl_batch_issue.h — A3 / O2.5: stream-MDI / direct-arrays / dyn-bind plans.
 *
 * Pure C, no Metal. ObjC issue and applyDynamic ports fill POD inputs.
 */

#ifndef MGL_BATCH_ISSUE_H
#define MGL_BATCH_ISSUE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MGL_BATCH_STREAM_MDI_OK = 0,
    MGL_BATCH_STREAM_MDI_FAIL_EMPTY = 1,
    MGL_BATCH_STREAM_MDI_FAIL_DISABLED = 2,
    MGL_BATCH_STREAM_MDI_FAIL_BAD_PRIM = 3,
    MGL_BATCH_STREAM_MDI_FAIL_OVERFLOW = 4
};

typedef struct MGLBatchStreamMdiGateIn {
    uint8_t stream_merged;
    uint8_t has_encoder;
    uint8_t disable_mdi;
    uint8_t primitive_type; /* 0xFF = unsupported */
    uint32_t command_count;
    uint32_t stream_index_count;
    size_t arg_size; /* sizeof indexed indirect args */
} MGLBatchStreamMdiGateIn;

/* Gate stream-merged MDI before index materialize / scratch alloc. */
int mgl_batch_issue_stream_mdi_gate(const MGLBatchStreamMdiGateIn *in,
                                    size_t *needed_bytes);
const char *mgl_batch_issue_stream_mdi_gate_reason(int gate);

/* Derive instanceCount / baseInstance / trace reasons for array cmds.
 * cmd_type values match MGL_CMD_DRAW_ARRAYS* enums. */
enum {
    MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS = 0,
    MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED = 2,
    MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE = 6
};

void mgl_batch_issue_direct_arrays_params(uint32_t cmd_type,
                                          int32_t instance_count,
                                          uint32_t base_instance,
                                          int32_t *out_instance_count,
                                          uint32_t *out_base_instance,
                                          const char **out_reason,
                                          const char **out_cull_reason);

int mgl_batch_issue_dyn_cmd_has_bindings(uint8_t vertex_count,
                                         uint8_t uniform_count,
                                         uint8_t texture_count);

int mgl_batch_issue_dyn_needs_mapper_fallback(int vertex_ok, int uniform_ok);

enum {
    MGL_BATCH_CULL_CAPTURE_NONE = 0,
    MGL_BATCH_CULL_CAPTURE_ARRAYS = 1,
    MGL_BATCH_CULL_CAPTURE_ELEMENTS = 2
};
int mgl_batch_issue_cull_capture_path(int uses_cull_distance,
                                      uint32_t cmd_type);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_ISSUE_H */
