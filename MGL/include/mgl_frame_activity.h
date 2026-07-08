/*
 * mgl_frame_activity.h
 * MGL
 *
 * Frame Activity Breadcrumb Subsystem.
 *
 * Cross-stage counters and "last draw" snapshots used for black-screen /
 * beachball diagnostics.  A watchdog in the draw path compares the current
 * swap-call index against these counters to detect stalled render loops; the
 * swap path snapshots + resets the per-frame counters and logs the last
 * draw-call metadata when a frame produced no work.
 *
 * All 19 globals are `volatile` because they are read from background
 * watchdog threads and written from the render thread.  They are exposed as
 * `extern` (not wrapped in getters) so hot-path write sites in MGLRenderer.m
 * incur no function-call overhead — the values are updated on every draw
 * call, dozens of times per frame.
 *
 * Dependencies: glcorearb.h (GLuint / GLsizei) + stdint.h (uint64_t).
 */

#ifndef MGL_FRAME_ACTIVITY_H
#define MGL_FRAME_ACTIVITY_H

#include "glcorearb.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Last draw-call metadata (written by draw path, read by swap path) === */

extern volatile uint64_t g_mglLastDrawArraysCall;
extern volatile double   g_mglLastDrawArraysSeconds;
extern volatile uint64_t g_mglLastDrawElementsCall;
extern volatile double   g_mglLastDrawElementsSeconds;
extern volatile GLuint   g_mglLastDrawArraysProgram;
extern volatile GLuint   g_mglLastDrawArraysMode;
extern volatile GLsizei  g_mglLastDrawArraysCount;
extern volatile GLuint   g_mglLastDrawElementsProgram;
extern volatile GLuint   g_mglLastDrawElementsMode;
extern volatile GLsizei  g_mglLastDrawElementsCount;

/* === Per-frame counters (reset to 0 by swap path after snapshot) === */

extern volatile uint64_t g_mglDrawArraysSinceSwap;
extern volatile uint64_t g_mglDrawElementsSinceSwap;
extern volatile uint64_t g_mglDrawArrayVerticesSinceSwap;
extern volatile uint64_t g_mglDrawElementIndicesSinceSwap;
extern volatile uint64_t g_mglDrawArraysSkippedSinceSwap;
extern volatile uint64_t g_mglDrawElementsSkippedSinceSwap;
extern volatile uint64_t g_mglProcessDrawCallsSinceSwap;

/* === Swap path metadata === */

extern volatile uint64_t g_mglSwapCallCount;
extern volatile double   g_mglLastSwapSeconds;

/* === Snapshot / reset helpers (inline — called once per swap) ===
 *
 * Snapshot reads the 7 per-frame counters into a struct; reset zeroes them.
 * Kept inline so the swap path can avoid a function call on the hot frame
 * boundary. */

typedef struct MGLSwapDrawCounters {
    uint64_t draw_arrays;
    uint64_t draw_elements;
    uint64_t array_vertices;
    uint64_t element_indices;
    uint64_t draw_arrays_skipped;
    uint64_t draw_elements_skipped;
    uint64_t process_draw_calls;
} MGLSwapDrawCounters;

static inline MGLSwapDrawCounters mglSnapshotSwapDrawCounters(void)
{
    MGLSwapDrawCounters counters;
    counters.draw_arrays            = g_mglDrawArraysSinceSwap;
    counters.draw_elements          = g_mglDrawElementsSinceSwap;
    counters.array_vertices         = g_mglDrawArrayVerticesSinceSwap;
    counters.element_indices        = g_mglDrawElementIndicesSinceSwap;
    counters.draw_arrays_skipped    = g_mglDrawArraysSkippedSinceSwap;
    counters.draw_elements_skipped  = g_mglDrawElementsSkippedSinceSwap;
    counters.process_draw_calls     = g_mglProcessDrawCallsSinceSwap;
    return counters;
}

static inline void mglResetSwapDrawCounters(void)
{
    g_mglDrawArraysSinceSwap           = 0;
    g_mglDrawElementsSinceSwap         = 0;
    g_mglDrawArrayVerticesSinceSwap    = 0;
    g_mglDrawElementIndicesSinceSwap   = 0;
    g_mglDrawArraysSkippedSinceSwap    = 0;
    g_mglDrawElementsSkippedSinceSwap  = 0;
    g_mglProcessDrawCallsSinceSwap     = 0;
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_FRAME_ACTIVITY_H */
