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
 * mgl_batch_restore.h — A3 / O2.5: same-key skip + dirty-key delta plans.
 *
 * Pure C, no Metal. ObjC flush/restore fills POD inputs and applies bits.
 */

#ifndef MGL_BATCH_RESTORE_H
#define MGL_BATCH_RESTORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Same-key restore skip (flushDrawBuffer loop) ---- */

typedef struct MGLBatchSameKeySkipIn {
    uint8_t skip_enabled;
    uint8_t last_key_valid;
    uint8_t last_execute_ok;
    uint8_t last_was_stream;
    uint8_t has_encoder;
    uint8_t bind_valid;
    uint8_t keys_equal;
    uint8_t absolute_offsets_match;
    uint8_t pass_matches;
} MGLBatchSameKeySkipIn;

enum {
    MGL_BATCH_SAME_KEY_SKIP = 0,
    MGL_BATCH_SAME_KEY_NO_SKIP = 1,
    MGL_BATCH_SAME_KEY_FAIL_NO_ENCODER = 2,
    MGL_BATCH_SAME_KEY_FAIL_BIND = 3,
    MGL_BATCH_SAME_KEY_FAIL_KEY = 4,
    /* Keys match but absolute-offset contract or pass mismatch. */
    MGL_BATCH_SAME_KEY_FAIL_PASS = 5
};

/* Returns SKIP, NO_SKIP, or FAIL_* (attribution when skip was otherwise
 * eligible: enabled + last_key_valid + last_execute_ok + !stream). */
int mgl_batch_same_key_skip_decision(const MGLBatchSameKeySkipIn *in);

/* ---- Dirty-key delta domains (restoreStateForBatch) ---- */

/* Fields needed from MGLStateKey for domain narrowing. ObjC copies. */
typedef struct MGLBatchStateKeyView {
    uint32_t program_name;
    uint32_t program_pipeline_name;
    uint32_t vertex_program_name;
    uint32_t fragment_program_name;
    uint32_t vao_name;
    uint64_t vertex_layout_hash;
    uint64_t texture_hash;
    uint64_t render_state_hash;
    uint64_t uniform_buffer_hash;
    uint32_t caps_flags;
    uint8_t scissor_enabled;
    uint8_t primitive_type;
    int32_t viewport[4];
    int32_t scissor[4];
} MGLBatchStateKeyView;

typedef struct MGLBatchDirtyDomainMasks {
    uint32_t program;      /* DIRTY_PROGRAM|DIRTY_BUFFER_BASE_STATE|DIRTY_BUFFER */
    uint32_t vao;          /* DIRTY_VAO|DIRTY_BUFFER */
    uint32_t texture;      /* DIRTY_TEX|BINDING|PARAM|SAMPLER|IMAGE_UNIT */
    uint32_t render_state; /* DIRTY_RENDER_STATE|DIRTY_ALPHA_STATE */
} MGLBatchDirtyDomainMasks;

typedef struct MGLBatchDirtyDeltaFlags {
    uint8_t domain_program;
    uint8_t domain_vao;
    uint8_t domain_texture;
    uint8_t domain_render_state;
    uint8_t domain_render_state_ubo_only;
    uint8_t narrowed;
} MGLBatchDirtyDeltaFlags;

/* When can_delta==0 returns full_bits and clears flags. Else ORs domain
 * masks for changed key fields; sets *flags_out for perf counters. */
uint32_t mgl_batch_compute_key_delta_dirty_bits(
    int can_delta, const MGLBatchStateKeyView *prev,
    const MGLBatchStateKeyView *cur, uint32_t full_bits,
    const MGLBatchDirtyDomainMasks *masks, MGLBatchDirtyDeltaFlags *flags_out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_RESTORE_H */
