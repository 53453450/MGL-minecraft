/*
 * mgl_focus_program.h
 * MGL
 *
 * Focus Program Observation Subsystem.
 *
 * Tracks GL programs that exhibit "interesting" draw patterns (e.g. repeated
 * draws with the same vertex count + enabled-attrib mask) and marks them for
 * verbose trace logging.  The focus list is process-global and append-only.
 *
 * Once a program is focused, hot paths in MGLRenderer use the inline predicate
 * below to gate extra diagnostics (MSL dumps, resource-binding traces, etc.)
 * without paying a function-call cost on every draw.
 *
 * Dependencies: mgl_trace_log.h (for mglTraceLogIsEnabled / mglTraceLog) and
 * glcorearb.h (for GLuint / GLsizei).
 */

#ifndef MGL_FOCUS_PROGRAM_H
#define MGL_FOCUS_PROGRAM_H

#include "glcorearb.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Globals ===
 *
 * Exposed so the hot-path inline predicate below can read them without a
 * function call.  Do NOT mutate directly — use mglFocusLoadingProgram.
 * Array capacity is fixed at 32 entries. */
extern GLuint g_mglFocusedLoadingPrograms[32];
extern uint32_t g_mglFocusedLoadingProgramCount;

/* === Operations === */

/* Marks `programName` as focused, recording the trigger `reason` and `detail`
 * in the trace log.  No-op if programName is 0, already in the focus list,
 * or the list is full (32 entries). */
void mglFocusLoadingProgram(GLuint programName, const char *reason, uint64_t detail);

/* Observes a draw call against `programName`.  If the same program + count +
 * enabledAttribs pattern repeats 16 times, the program is auto-focused with
 * reason "repeated-draw-pattern".  No-op if programName is 0. */
void mglObserveProgramDrawForFocus(GLuint programName, GLsizei count, GLuint enabledAttribs);

/* === Hot-path predicate ===
 *
 * Returns true if `programName` is in the focus list.  Inlined because it is
 * called from draw-call hot paths (mglTraceShouldLogReplay, sampler-binding
 * traces, draw-element entry). */
static inline bool mglIsFocusedLoadingProgram(GLuint programName)
{
    for (uint32_t i = 0; i < g_mglFocusedLoadingProgramCount; i++) {
        if (g_mglFocusedLoadingPrograms[i] == programName) {
            return true;
        }
    }
    return false;
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_FOCUS_PROGRAM_H */
