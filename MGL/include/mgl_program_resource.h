/* AIR-reflected program resource helpers shared by C and Objective-C paths. */

#ifndef MGL_PROGRAM_RESOURCE_H
#define MGL_PROGRAM_RESOURCE_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *mglShaderStageName(int stage);

bool mglShouldSkipStageBufferResource(Program *program,
                                      int stage,
                                      int resource_type,
                                      const MGLShaderResource *resource);
bool mglShouldSkipStageTextureResource(Program *program,
                                       int stage,
                                       int resource_type,
                                       const MGLShaderResource *resource);
bool mglShouldSkipStageSamplerResource(Program *program,
                                       int stage,
                                       int resource_type,
                                       const MGLShaderResource *resource);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PROGRAM_RESOURCE_H */
