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

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_GS_H */
