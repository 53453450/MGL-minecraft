/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_program_backend.h -- access to the backend state of a Program.
 *
 * Program::backend (MGLProgramBackend, see mgl_backend_objects.h) holds what
 * the AIR backend built for each linked stage.  Readers use the view
 * accessors, which never return NULL: a program without backend state reads as
 * all zeros.  Writers use the ensure accessors, which allocate on first use.
 * If that allocation fails the writer gets a scratch sink instead of NULL, so
 * the link path stays straight-line; the program then simply has nothing
 * built and the link is reported as failed by the missing metallib.
 */

#ifndef MGL_PROGRAM_BACKEND_H
#define MGL_PROGRAM_BACKEND_H

#include <stdlib.h>
#include <string.h>

#include "mgl_backend_objects.h"
#include "mgl_types_program.h"

typedef char mgl_program_backend_stage_count_matches
    [(MGL_PROGRAM_BACKEND_STAGES == _MAX_SHADER_TYPES) ? 1 : -1];

#ifdef __cplusplus
extern "C" {
#endif

/* Free the metallib bytes and release the Metal objects of one stage. */
void mglShaderModuleBackendClear(struct MGLShaderModuleBackend *module);

/* Clear every stage and free the backend struct. */
void mglProgramBackendRelease(Program *program);

#ifdef __cplusplus
}
#endif

static inline const struct MGLProgramBackend *
mglProgramBackendView(const Program *program)
{
    static struct MGLProgramBackend none; /* zero, never written */
    return program && program->backend ? program->backend : &none;
}

static inline struct MGLProgramBackend *
mglProgramBackendEnsure(Program *program)
{
    static struct MGLProgramBackend sink;
    if (!program) {
        memset(&sink, 0, sizeof(sink));
        return &sink;
    }
    if (!program->backend) {
        program->backend = (struct MGLProgramBackend *)calloc(
            1, sizeof(struct MGLProgramBackend));
        if (!program->backend) {
            memset(&sink, 0, sizeof(sink));
            return &sink;
        }
    }
    return program->backend;
}

static inline const struct MGLShaderModuleBackend *
mglProgramModuleView(const Program *program, int stage)
{
    static struct MGLShaderModuleBackend none; /* zero, never written */
    if (stage < 0 || stage >= MGL_PROGRAM_BACKEND_STAGES) {
        return &none;
    }
    return &mglProgramBackendView(program)->modules[stage];
}

static inline struct MGLShaderModuleBackend *
mglProgramModuleBackend(Program *program, int stage)
{
    static struct MGLShaderModuleBackend sink;
    if (stage < 0 || stage >= MGL_PROGRAM_BACKEND_STAGES) {
        memset(&sink, 0, sizeof(sink));
        return &sink;
    }
    return &mglProgramBackendEnsure(program)->modules[stage];
}

#endif /* MGL_PROGRAM_BACKEND_H */
