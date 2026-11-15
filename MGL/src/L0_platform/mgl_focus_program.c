/*
 * mgl_focus_program.m
 * MGL
 *
 * Implementation of the Focus Program Observation Subsystem.
 * See mgl_focus_program.h for the API contract.
 *
 * Owns the two process-global focus-list symbols and the two operations that
 * mutate / observe them.  The inline predicate lives in the header so hot
 * paths can read the list without a function call.
 */

#include "mgl_focus_program.h"

#include "mgl_trace_log.h"

#include <stddef.h>

/* === Globals === */

GLuint g_mglFocusedLoadingPrograms[32] = {0};
uint32_t g_mglFocusedLoadingProgramCount = 0;

/* === Operations === */

void mglFocusLoadingProgram(GLuint programName, const char *reason, uint64_t detail)
{
    if (programName == 0) {
        return;
    }

    for (uint32_t i = 0; i < g_mglFocusedLoadingProgramCount; i++) {
        if (g_mglFocusedLoadingPrograms[i] == programName) {
            return;
        }
    }

    if (g_mglFocusedLoadingProgramCount >= (uint32_t)(sizeof(g_mglFocusedLoadingPrograms) / sizeof(g_mglFocusedLoadingPrograms[0]))) {
        return;
    }

    g_mglFocusedLoadingPrograms[g_mglFocusedLoadingProgramCount++] = programName;
    if (mglTraceLogIsEnabled()) {
        mglTraceLog("FOCUS_PROGRAM program=%u reason=%s detail=%llu",
                    (unsigned)programName,
                    reason ? reason : "(none)",
                    (unsigned long long)detail);
    }
}
