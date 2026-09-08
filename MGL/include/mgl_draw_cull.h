/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#ifndef MGL_DRAW_CULL_H
#define MGL_DRAW_CULL_H

#include "glcorearb.h"
#include "glm_context.h"
#include "mgl_types_program.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* O1.4 residual: cull-distance capture / encode orchestration in C++;
 * ObjC supplies thin MTL HostOps (buffer create / processGL / bind emu).
 * enc_ctx is the host MGLEncodeContext* (opaque here). */

typedef struct MGLCullDistanceHostOps {
    void *renderer;
    int (*bind_mtl_program)(void *renderer, Program *program);
    void *(*create_buffer)(void *renderer, uint64_t length);
    void (*set_ctx)(void *renderer, GLMContext ctx);
    void (*clear_cull_capture)(void *renderer);
    void (*set_cull_capture_active)(void *renderer, int active);
    void (*store_cull_capture)(void *renderer, void *buf, uint32_t first_instance,
                               uint32_t instance_stride);
    void *(*load_cull_capture)(void *renderer);
    void (*mark_dirty_all)(void *ctx);
    int (*process_gl_state)(void *renderer);
    int (*encoder_has_current)(void *renderer);
    void *(*encoder_owner)(void *renderer);
    void *(*device)(void *renderer);
    void (*mark_cb_has_work)(void *renderer);
    void (*end_render_encoding)(void *renderer);
    void (*bind_cull_emu)(void *renderer, GLenum mode, GLuint first_vertex,
                          const uint32_t *explicit_vertices,
                          uint32_t explicit_vertex_count, const void *enc_ctx);
    int (*try_array_split_encode)(void *renderer, void *device, void *encoder_owner,
                                  GLenum mode, GLint first, GLsizei count,
                                  uint64_t instance_count, uint64_t base_instance,
                                  const void *enc_ctx);
    void (*draw_indexed_primitives)(void *encoder_owner, uint32_t primitive_type,
                                    uint64_t index_count, void *index_buffer,
                                    uint64_t index_offset, uint64_t instance_count,
                                    int32_t base_vertex, uint64_t base_instance);
    int (*encode_context_active)(const void *enc_ctx);
    int (*primitive_restart)(GLMContext ctx, GLenum index_type,
                             uint32_t *out_restart_index);
} MGLCullDistanceHostOps;

/* 1 = captured, 0 = failed / not applicable. */
int mglDrawRunCullDistanceArrayCapture(GLMContext ctx, GLint first, GLsizei count,
                                       GLsizei instanceCount, GLuint baseInstance,
                                       const MGLCullDistanceHostOps *ops);

int mglDrawRunCullDistanceElementCapture(GLMContext ctx, const uint8_t *indexBytes,
                                         GLenum indexType, GLsizei count,
                                         GLint baseVertex, GLsizei instanceCount,
                                         GLuint baseInstance,
                                         const MGLCullDistanceHostOps *ops);

/* 1 = handled (encoded or early-ok), 0 = fall through to normal encode. */
int mglDrawEncodeCullDistanceArray(GLMContext ctx, GLenum mode, GLint first,
                                   GLsizei count, GLsizei instanceCount,
                                   GLuint baseInstance, const void *enc_ctx,
                                   const MGLCullDistanceHostOps *ops);

int mglDrawEncodeCullDistanceElement(GLMContext ctx, GLenum mode,
                                     const uint8_t *indexBytes, GLenum indexType,
                                     GLsizei count, GLint baseVertex,
                                     GLsizei instanceCount, GLuint baseInstance,
                                     int polygon_line_mode, const void *enc_ctx,
                                     const MGLCullDistanceHostOps *ops);

/* 1 = cull path handled draw (caller skips normal encode), 0 = N/A. */
int mglDrawPrepareAndEncodeCullDistanceElement(
    GLMContext ctx, GLenum mode, const uint8_t *indexBytes, GLenum indexType,
    GLsizei count, GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    int polygon_line_mode, const MGLCullDistanceHostOps *ops);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_CULL_H */
