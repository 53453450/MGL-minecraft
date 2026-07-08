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
 * All 19 globals are `_Atomic` because they are read from background
 * watchdog threads and written from the render thread.  They are exposed as
 * `extern` (not wrapped in getters) so hot-path write sites in MGLRenderer.m
 * incur no function-call overhead — the values are updated on every draw
 * call, dozens of times per frame.  `volatile` only prevents compiler
 * reordering; `_Atomic` additionally guarantees CPU-level atomicity, which
 * is what cross-thread read/write of these counters actually requires.
 * Translation units that update these globals with the native `_Atomic`
 * operators (e.g. `counter++`, `counter = value`) also get atomic semantics
 * (sequential consistency by default); the inline helpers below use explicit
 * relaxed ordering since these are best-effort statistics.
 *
 * Dependencies: glcorearb.h (GLuint / GLsizei) + stdint.h (uint64_t)
 * + stdatomic.h (_Atomic / atomic_* operations).
 */

#ifndef MGL_FRAME_ACTIVITY_H
#define MGL_FRAME_ACTIVITY_H

#include "glcorearb.h"

#include <stdint.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Last draw-call metadata (written by draw path, read by swap path) === */

extern _Atomic uint64_t g_mglLastDrawArraysCall;
extern _Atomic double   g_mglLastDrawArraysSeconds;
extern _Atomic uint64_t g_mglLastDrawElementsCall;
extern _Atomic double   g_mglLastDrawElementsSeconds;
extern _Atomic GLuint   g_mglLastDrawArraysProgram;
extern _Atomic GLuint   g_mglLastDrawArraysMode;
extern _Atomic GLsizei  g_mglLastDrawArraysCount;
extern _Atomic GLuint   g_mglLastDrawElementsProgram;
extern _Atomic GLuint   g_mglLastDrawElementsMode;
extern _Atomic GLsizei  g_mglLastDrawElementsCount;

/* === Per-frame counters (reset to 0 by swap path after snapshot) === */

extern _Atomic uint64_t g_mglDrawArraysSinceSwap;
extern _Atomic uint64_t g_mglDrawElementsSinceSwap;
extern _Atomic uint64_t g_mglDrawArrayVerticesSinceSwap;
extern _Atomic uint64_t g_mglDrawElementIndicesSinceSwap;
extern _Atomic uint64_t g_mglDrawArraysSkippedSinceSwap;
extern _Atomic uint64_t g_mglDrawElementsSkippedSinceSwap;
extern _Atomic uint64_t g_mglProcessDrawCallsSinceSwap;

/* === Swap path metadata === */

extern _Atomic uint64_t g_mglSwapCallCount;
extern _Atomic double   g_mglLastSwapSeconds;

/* === Snapshot / reset helpers (inline — called once per swap) ===
 *
 * Snapshot reads the 7 per-frame counters into a struct; reset zeroes them.
 * Kept inline so the swap path can avoid a function call on the hot frame
 * boundary.  Each individual load/store is atomic (relaxed ordering); however
 * the snapshot as a whole is NOT a consistent point-in-time view because the
 * render thread may update counters between individual reads.  This is
 * acceptable for frame-activity statistics where approximate values suffice. */

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
    counters.draw_arrays            = atomic_load_explicit(&g_mglDrawArraysSinceSwap,          memory_order_relaxed);
    counters.draw_elements          = atomic_load_explicit(&g_mglDrawElementsSinceSwap,        memory_order_relaxed);
    counters.array_vertices         = atomic_load_explicit(&g_mglDrawArrayVerticesSinceSwap,   memory_order_relaxed);
    counters.element_indices        = atomic_load_explicit(&g_mglDrawElementIndicesSinceSwap,  memory_order_relaxed);
    counters.draw_arrays_skipped    = atomic_load_explicit(&g_mglDrawArraysSkippedSinceSwap,   memory_order_relaxed);
    counters.draw_elements_skipped  = atomic_load_explicit(&g_mglDrawElementsSkippedSinceSwap, memory_order_relaxed);
    counters.process_draw_calls     = atomic_load_explicit(&g_mglProcessDrawCallsSinceSwap,    memory_order_relaxed);
    return counters;
}

static inline void mglResetSwapDrawCounters(void)
{
    atomic_store_explicit(&g_mglDrawArraysSinceSwap,          0, memory_order_relaxed);
    atomic_store_explicit(&g_mglDrawElementsSinceSwap,        0, memory_order_relaxed);
    atomic_store_explicit(&g_mglDrawArrayVerticesSinceSwap,   0, memory_order_relaxed);
    atomic_store_explicit(&g_mglDrawElementIndicesSinceSwap,  0, memory_order_relaxed);
    atomic_store_explicit(&g_mglDrawArraysSkippedSinceSwap,   0, memory_order_relaxed);
    atomic_store_explicit(&g_mglDrawElementsSkippedSinceSwap, 0, memory_order_relaxed);
    atomic_store_explicit(&g_mglProcessDrawCallsSinceSwap,    0, memory_order_relaxed);
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_FRAME_ACTIVITY_H */
