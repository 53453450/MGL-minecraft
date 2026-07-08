/*
 * mgl_frame_activity.m
 * MGL
 *
 * Implementation of the Frame Activity Breadcrumb Subsystem.
 * See mgl_frame_activity.h for the API contract.
 *
 * Owns the 19 process-global volatile counters/snapshots used by the
 * black-screen / beachball watchdog.  The snapshot/reset helpers live in the
 * header (inline) so the swap path can read + clear the per-frame counters
 * without a function call.
 */

#import "mgl_frame_activity.h"

/* === Last draw-call metadata === */

volatile uint64_t g_mglLastDrawArraysCall       = 0;
volatile double   g_mglLastDrawArraysSeconds    = 0.0;
volatile uint64_t g_mglLastDrawElementsCall     = 0;
volatile double   g_mglLastDrawElementsSeconds  = 0.0;
volatile GLuint   g_mglLastDrawArraysProgram    = 0;
volatile GLuint   g_mglLastDrawArraysMode       = 0;
volatile GLsizei  g_mglLastDrawArraysCount      = 0;
volatile GLuint   g_mglLastDrawElementsProgram  = 0;
volatile GLuint   g_mglLastDrawElementsMode     = 0;
volatile GLsizei  g_mglLastDrawElementsCount    = 0;

/* === Per-frame counters === */

volatile uint64_t g_mglDrawArraysSinceSwap          = 0;
volatile uint64_t g_mglDrawElementsSinceSwap        = 0;
volatile uint64_t g_mglDrawArrayVerticesSinceSwap   = 0;
volatile uint64_t g_mglDrawElementIndicesSinceSwap  = 0;
volatile uint64_t g_mglDrawArraysSkippedSinceSwap   = 0;
volatile uint64_t g_mglDrawElementsSkippedSinceSwap = 0;
volatile uint64_t g_mglProcessDrawCallsSinceSwap    = 0;

/* === Swap path metadata === */

volatile uint64_t g_mglSwapCallCount   = 0;
volatile double   g_mglLastSwapSeconds = 0.0;
