/*
 * mgl_frame_activity.m
 * MGL
 *
 * Implementation of the Frame Activity Breadcrumb Subsystem.
 * See mgl_frame_activity.h for the API contract.
 *
 * Owns the 19 process-global atomic counters/snapshots used by the
 * black-screen / beachball watchdog.  The snapshot/reset helpers live in the
 * header (inline) so the swap path can read + clear the per-frame counters
 * without a function call.
 */

#import "mgl_frame_activity.h"

#include <stdatomic.h>

/* === Last draw-call metadata === */

_Atomic uint64_t g_mglLastDrawArraysCall       = 0;
_Atomic double   g_mglLastDrawArraysSeconds    = 0.0;
_Atomic uint64_t g_mglLastDrawElementsCall     = 0;
_Atomic double   g_mglLastDrawElementsSeconds  = 0.0;
_Atomic GLuint   g_mglLastDrawArraysProgram    = 0;
_Atomic GLuint   g_mglLastDrawArraysMode       = 0;
_Atomic GLsizei  g_mglLastDrawArraysCount      = 0;
_Atomic GLuint   g_mglLastDrawElementsProgram  = 0;
_Atomic GLuint   g_mglLastDrawElementsMode     = 0;
_Atomic GLsizei  g_mglLastDrawElementsCount    = 0;

/* === Per-frame counters === */

_Atomic uint64_t g_mglDrawArraysSinceSwap          = 0;
_Atomic uint64_t g_mglDrawElementsSinceSwap        = 0;
_Atomic uint64_t g_mglDrawArrayVerticesSinceSwap   = 0;
_Atomic uint64_t g_mglDrawElementIndicesSinceSwap  = 0;
_Atomic uint64_t g_mglDrawArraysSkippedSinceSwap   = 0;
_Atomic uint64_t g_mglDrawElementsSkippedSinceSwap = 0;
_Atomic uint64_t g_mglProcessDrawCallsSinceSwap    = 0;

/* === Swap path metadata === */

_Atomic uint64_t g_mglSwapCallCount   = 0;
_Atomic double   g_mglLastSwapSeconds = 0.0;
