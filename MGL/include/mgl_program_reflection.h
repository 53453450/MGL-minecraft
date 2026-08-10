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

GLboolean mglProgramVaryingTypesCompatible(const SpirvResource *a,
                                           const SpirvResource *b);
SpirvResource *mglFindVaryingByName(SpirvResourceList *list,
                                    const char *name,
                                    const SpirvResource *type_peer);
SpirvResource *mglFindVaryingByLocation(SpirvResourceList *list,
                                        GLuint location,
                                        const SpirvResource *type_peer);
void mglBridgeSkippedGeometryShaderVaryings(Program *program);
GLboolean mglProgramHasPassthroughGeometryShader(Program *program);

#ifdef __cplusplus
}
#endif

#endif
