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

#include <string.h>

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

/* Narrow Minecraft/CTS passthrough-GS detection (re-emits gl_in
 * unchanged, no layer/viewport/primitive-id).  Such a program normally
 * bypasses the GS compute expansion entirely (plain VS->FS draw).  A
 * program with transform-feedback varyings must NOT take the bypass:
 * the bypassed draw never runs the GS compute expansion and the CPU
 * feedback path rejects GS-attached programs, so the capture would be
 * silently lost.  static inline so the standalone Metal-cpp smoke
 * binary (which does not link mgl_program_reflection.c) shares the
 * single predicate. */
static inline GLboolean mglProgramHasPassthroughGeometryShader(
    Program *program)
{
    const char *src = program && program->shader_slots[_GEOMETRY_SHADER]
        ? program->shader_slots[_GEOMETRY_SHADER]->src : NULL;
    if (!src) return GL_FALSE;
    if (program->transform_feedback_varying_count > 0) return GL_FALSE;
    return strstr(src, "EmitVertex()") &&
           strstr(src, "EndPrimitive()") &&
           strstr(src, "gl_Position = gl_in[n_vertex_index].gl_Position") &&
           !strstr(src, "gl_PrimitiveID") &&
           !strstr(src, "gl_Layer") &&
           !strstr(src, "gl_ViewportIndex");
}

#ifdef __cplusplus
}
#endif

#endif
