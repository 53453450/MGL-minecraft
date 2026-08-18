/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

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
