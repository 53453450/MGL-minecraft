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

bool mglDrawHostEncodeCullDistanceElements(void *renderer, GLenum mode,
                                           GLenum type, const void *indices,
                                           GLsizei count, GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance);
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

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_ISSUE_H */
