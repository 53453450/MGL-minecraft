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
 * AIR-backed program reflection helpers.
 *
 * The legacy implementation mixed GL-facing reflection with backend
 * type handles and raw bytecode analysis.  AIR reflection now populates the
 * program resource tables directly, so this interface only exposes the
 * runtime queries and location reconciliation used by the GL API.
 */

#ifndef MGL_UNIFORM_REFLECTION_H
#define MGL_UNIFORM_REFLECTION_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

GLint mglActiveUniformBlockCount(Program *program);
GLint mglActiveAtomicCounterBufferCount(Program *program);
GLint mglActiveUniformBlockMaxNameLength(Program *program);
GLint mglProgramActiveAttribCount(Program *program);
MGLShaderResource *mglProgramActiveAttribAt(Program *program, GLuint index);
GLint mglProgramActiveAttribMaxNameLength(Program *program);
GLenum mglProgramActiveAttribType(const MGLShaderResource *res);

GLint mglSyntheticSamplerUniformLocation(int stage, int resource_type,
                                         GLuint index);
GLint mglSamplerUniformLocationFromReflection(GLuint reflected_location,
                                              int stage,
                                              int resource_type,
                                              GLuint index,
                                              const char *glsl_src,
                                              const char *resource_name);
bool mglUniformNameLooksSamplerLike(const char *name);
void mglUnifySamplerUniformLocations(Program *program);

void mglAssignPlainUniformLocations(Program *program);
void mglAssignAggregateMemberLocations(Program *program);
void mglFreeMGLShaderResourceOwnedFields(MGLShaderResource *res);

#ifdef __cplusplus
}
#endif

#endif /* MGL_UNIFORM_REFLECTION_H */
