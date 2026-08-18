/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#ifndef MGL_PROGRAM_REFLECTION_H
#define MGL_PROGRAM_REFLECTION_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void clearStageCompileState(Program *program, int stage);

GLboolean mglProgramPerVertexSignature(Program *program, int stage,
                                       unsigned *signature);
GLboolean mglProgramPipelinePerVertexCompatible(
    Program *const *stage_programs);
GLboolean mglLinkedProgramPerVertexCompatible(Program *program);

GLint mglDefaultAttribLocationForName(const char *name);
GLint mglProgramVertexInputOrdinal(Program *program, const char *name);
GLboolean mglProgramHasVertexInputNamed(Program *program, const char *name);
GLint mglContextualDefaultAttribLocationForName(Program *program,
                                                const char *name);
GLint mglDesiredAttribLocationForName(Program *program, const char *name);

void applyVertexInputLocations(Program *program);
void applyMultiDimArrayUniformNames(Program *program);
void applyFragmentOutputLocationIndices(Program *program);
void alignFragmentInputLocationsToVertexOutputs(Program *program);

GLboolean mglProgramVaryingTypesCompatible(const MGLShaderResource *a,
                                           const MGLShaderResource *b);
MGLShaderResource *mglFindVaryingByName(MGLShaderResourceList *list,
                                    const char *name,
                                    const MGLShaderResource *type_peer);
MGLShaderResource *mglFindVaryingByLocation(MGLShaderResourceList *list,
                                        GLuint location,
                                        const MGLShaderResource *type_peer);
void mglBridgeSkippedGeometryShaderVaryings(Program *program);
GLboolean mglProgramHasPassthroughGeometryShader(Program *program);

#ifdef __cplusplus
}
#endif

#endif
