/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#ifndef MGL_DRAW_ISSUE_H
#define MGL_DRAW_ISSUE_H

#include "glcorearb.h"
#include "glm_context.h"
#include "mgl_draw_validate.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C++ owns tess/GS/XFB → state → encode sequencing. renderer is MGLRenderer*. */

void mglIssueDrawArrays(GLMContext ctx, void *renderer, GLenum mode,
                        GLint first, GLsizei count, GLsizei instanceCount,
                        GLuint baseInstance, const char *label);

void mglIssueDrawElements(GLMContext ctx, void *renderer, GLenum mode,
                          GLsizei count, GLenum type, const void *indices,
                          GLsizei instanceCount, GLint baseVertex,
                          GLuint baseInstance, const char *label);

void mglIssueMultiDrawArrays(GLMContext ctx, void *renderer, GLenum mode,
                             const GLint *first, const GLsizei *count,
                             GLsizei drawcount, const char *label);

void mglIssueMultiDrawElements(GLMContext ctx, void *renderer, GLenum mode,
                               const GLsizei *count, GLenum type,
                               const void *const *indices, GLsizei drawcount,
                               const GLint *basevertex, const char *label);

void mglIssueDrawArraysIndirect(GLMContext ctx, void *renderer, GLenum mode,
                                const void *indirect, const char *label);

void mglIssueDrawElementsIndirect(GLMContext ctx, void *renderer, GLenum mode,
                                  GLenum type, const void *indirect,
                                  const char *label);

void mglIssueMultiDrawArraysIndirect(GLMContext ctx, void *renderer,
                                     GLenum mode, const void *indirect,
                                     GLsizei drawcount, GLsizei stride,
                                     const char *label);

void mglIssueMultiDrawElementsIndirect(GLMContext ctx, void *renderer,
                                       GLenum mode, GLenum type,
                                       const void *indirect, GLsizei drawcount,
                                       GLsizei stride, const char *label);

bool mglDrawHostBindContext(void *renderer, GLMContext ctx);
void mglDrawHostSetLastPrimitiveMode(void *renderer, GLenum mode);
bool mglDrawHostHandleTessellation(void *renderer, GLMContext ctx,
                                   GLenum *mode, GLint first, GLsizei count,
                                   GLenum indexType, const void *indices,
                                   GLint baseVertex, GLsizei instanceCount,
                                   GLuint baseInstance, const char *label);
bool mglDrawHostHandleGeometry(void *renderer, GLMContext ctx, GLenum mode,
                               GLint first, GLsizei count, GLenum indexType,
                               const void *indices, GLint baseVertex,
                               GLsizei instanceCount, GLuint baseInstance,
                               const char *label);
bool mglDrawHostHandleXFB(void *renderer, GLMContext ctx, GLenum mode,
                          GLint first, GLsizei count, GLsizei instanceCount,
                          GLuint baseInstance);
bool mglDrawHostCaptureCullDistanceArray(void *renderer, GLMContext ctx,
                                         GLint first, GLsizei count,
                                         GLsizei instanceCount,
                                         GLuint baseInstance);
bool mglDrawHostCaptureCullDistanceElement(void *renderer, GLMContext ctx,
                                           const uint8_t *indexBytes,
                                           GLenum indexType, GLsizei count,
                                           GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance);
bool mglDrawHostProcessGLStateLocked(void *renderer, bool draw_command);
bool mglDrawHostRasterizationIsEmpty(void *renderer);
bool mglDrawHostModeFullyCulled(void *renderer, GLenum mode);
void mglDrawHostApplyPolygonOffset(void *renderer, GLenum mode);
bool mglDrawHostEnsureRasterEncoder(void *renderer);
bool mglDrawHostValidateArrayVertexInputs(void *renderer, GLMContext ctx,
                                          GLenum mode, GLint first,
                                          GLsizei count);
bool mglDrawHostEncodeCullDistanceArray(void *renderer, GLenum mode,
                                        GLint first, GLsizei count,
                                        GLsizei instanceCount,
                                        GLuint baseInstance);
void *mglDrawHostEncoderOwner(void *renderer);
void *mglDrawHostDevice(void *renderer);
void mglDrawHostRecordArraySubmitted(void *renderer, GLenum mode,
                                     uint64_t vertexCount);
void mglDrawHostWatchdogArrays(void *renderer, GLMContext ctx);

bool mglDrawHostPrepareEncodeCullDistanceElement(
    void *renderer, GLenum mode, const uint8_t *indexBytes, GLenum type,
    GLsizei count, GLint baseVertex, GLsizei instanceCount,
    GLuint baseInstance, int polygon_line_mode);
bool mglDrawHostEncodeCullDistanceElementBytes(
    void *renderer, GLenum mode, const uint8_t *indexBytes, GLenum type,
    GLsizei count, GLint baseVertex, GLsizei instanceCount,
    GLuint baseInstance, int polygon_line_mode,
    const void *enc_ctx);
bool mglDrawHostEncodeCullDistanceElements(void *renderer, GLenum mode,
                                           GLenum type, const void *indices,
                                           GLsizei count, GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance);
/* The MGLCullDistanceBindFn the cull-distance split encoder calls back into;
 * defined next to the other draw host ports (mgl_draw_metal_port.m). */
void mglRendererBindCullDistanceEmu(void *renderer, const void *encode_context,
                                    GLenum mode, GLuint first_vertex,
                                    const uint32_t *explicit_vertices,
                                    uint32_t explicit_vertex_count);
bool mglDrawHostResolveElementBuffer(void *renderer, GLMContext ctx,
                                     const char *label, Buffer **glBufferOut,
                                     void **metalBufferOut);
void mglDrawHostRecordElementSubmitted(void *renderer, GLenum mode,
                                       uint64_t indexCount);
void mglDrawHostWatchdogElements(void *renderer, GLMContext ctx);

bool mglDrawHostResolveIndirectBuffer(void *renderer, GLMContext ctx,
                                      const char *label, Buffer **glBufferOut,
                                      void **metalBufferOut);
bool mglDrawHostPrepareIndirectCPURead(void *renderer, GLMContext ctx,
                                       const char *label);
bool mglDrawHostHasGeometry(GLMContext ctx);
bool mglDrawHostUsesCullDistance(GLMContext ctx);
void *mglDrawHostRunVertexCaptureArray(void *renderer, GLMContext ctx,
                                      GLint first, GLsizei count,
                                      GLsizei instanceCount,
                                      GLuint baseInstance,
                                      uint64_t *out_offset);
void *mglDrawHostRunVertexCaptureIndexed(
    void *renderer, GLMContext ctx, void *index_mtl, uint64_t index_type,
    uint64_t index_offset, GLsizei count, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, uint32_t maxIndex,
    uint64_t *out_offset);

/* O1.4 residual: drawArrays VBO-range validation HostOps runner. */
typedef struct MGLValidateArraysAttribInfo {
    uint32_t attrib_index;
    uint32_t buffer_name;
    int64_t binding_offset;
    int64_t relativeoffset;
    uint32_t stride;
    uint32_t divisor;
    uint32_t attrib_type;
    uint32_t attrib_size;
    int64_t vbo_size;
    int has_drawable;
    int64_t written_min;
    int64_t written_max;
    uint32_t last_init_source;
    uint32_t mapped;
    uint32_t access;
    uint32_t access_flags;
    uint32_t has_initialized_data;
    int64_t last_write_offset;
    int64_t last_write_size;
    const void *last_write_src_ptr;
    uint64_t last_write_src_hash;
    void *buffer_obj; /* Buffer* borrowed */
    void *mtl_data; /* borrowed */
} MGLValidateArraysAttribInfo;

typedef struct MGLValidateArraysHostOps {
    void *renderer;
    void *(*get_validated_vao)(GLMContext ctx, const char *where);
    int (*attrib_enabled)(void *vao, uint32_t attrib);
    int (*resolve_attrib)(GLMContext ctx, void *vao, uint32_t attrib,
                          const char *where, MGLValidateArraysAttribInfo *out);
    int (*ensure_mtl_buffer)(void *renderer, MGLValidateArraysAttribInfo *info);
    uint64_t (*mtl_buffer_length)(void *mtl_data);
    uint32_t (*max_attribs)(void);
    uint32_t (*current_program_key)(GLMContext ctx);
    int (*should_inspect)(uint64_t draw_call, uint32_t program_key);
    void (*log_line)(const char *msg);
} MGLValidateArraysHostOps;

/* 1 = inputs OK (or validation disabled), 0 = block draw. */
int mglDrawValidateArraysVertexInputs(GLMContext ctx, GLenum mode, GLint first,
                                      GLsizei count, uint64_t draw_call,
                                      int validation_enabled,
                                      const MGLValidateArraysHostOps *ops);

/* O1.5: ObjC mtlDrawArrays/Elements one-liners → lock/MS host then mglIssue*. */
void mglDrawHostGuardIssueArrays(void *renderer, GLMContext ctx, GLenum mode,
                                 GLint first, GLsizei count,
                                 GLsizei instanceCount, GLuint baseInstance,
                                 const char *label, int with_ms);
void mglDrawHostGuardIssueElements(void *renderer, GLMContext ctx, GLenum mode,
                                   GLsizei count, GLenum type,
                                   const void *indices, GLsizei instanceCount,
                                   GLint baseVertex, GLuint baseInstance,
                                   const char *label, int with_ms);


#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_ISSUE_H */
