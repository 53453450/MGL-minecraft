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

/* Exact per-stage builtin usage (MGL_AIR_BUILTIN_* mask) as reflected by the
 * IR/TU at link.  This is the replacement for scanning shader->src for "gl_X":
 * the reflected resource lists filter builtins out, so the frontend publishes
 * the fact instead.  Returns 0 for an unknown stage or a program that predates
 * the flag. */
uint32_t mglProgramStageBuiltinMask(const Program *program, int stage);

/* Convenience predicate: does `stage` reference any of `mask`'s builtins? */
static inline int mglProgramStageUsesBuiltin(const Program *program, int stage,
                                             uint32_t mask)
{
    return (mglProgramStageBuiltinMask(program, stage) & mask) != 0u ? 1 : 0;
}

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
