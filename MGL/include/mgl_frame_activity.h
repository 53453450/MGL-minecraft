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
 * call, dozens of times per frame.
 *
 * Dependencies: glcorearb.h (GLuint / GLsizei) + stdint.h (uint64_t) +
 * stdatomic.h (_Atomic).
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

/* === Performance counters (reset to 0 by swap path after snapshot) ===
 *
 * Counters for MGL_PERF_SUMMARY.  All use _Atomic + memory_order_relaxed
 * for consistency with the watchdog counters above.  Single-threaded
 * assumption: these are written from the render thread only. */

/* Draw path classification */
extern _Atomic uint64_t g_mglDrawDirectSinceSwap;        /* direct (per-cmd) draws */
extern _Atomic uint64_t g_mglDrawMDISinceSwap;           /* MDI batch draws */
extern _Atomic uint64_t g_mglDrawStreamMergedSinceSwap;  /* stream-merged draws */
extern _Atomic uint64_t g_mglDrawSkippedSinceSwap;       /* skipped (no-op) draws */

/* Batch counts */
extern _Atomic uint64_t g_mglBatchesDirectSinceSwap;
extern _Atomic uint64_t g_mglBatchesMDISinceSwap;
extern _Atomic uint64_t g_mglBatchesStreamMergedSinceSwap;

/* Pipeline cache */
extern _Atomic uint64_t g_mglPipelineCacheHitsSinceSwap;
extern _Atomic uint64_t g_mglPipelineCacheMissesSinceSwap;
extern _Atomic uint64_t g_mglPipelineCacheEvictionsSinceSwap;

/* Shader compile */
extern _Atomic uint64_t g_mglShaderCompilesSinceSwap;
extern _Atomic double   g_mglShaderCompileTimeSinceSwap;  /* seconds */

/* Encoder state calls (total setXxx calls issued) */
extern _Atomic uint64_t g_mglSetVertexBufferCallsSinceSwap;
extern _Atomic uint64_t g_mglSetFragmentBufferCallsSinceSwap;
extern _Atomic uint64_t g_mglSetRenderPipelineStateCallsSinceSwap;

/* Encoder state skips (setXxx calls avoided by dedup) */
extern _Atomic uint64_t g_mglSetVertexBufferSkipsSinceSwap;
extern _Atomic uint64_t g_mglSetFragmentBufferSkipsSinceSwap;
extern _Atomic uint64_t g_mglSetRenderPipelineStateSkipsSinceSwap;

/* Render encoder lifecycle (Stage 4 — RenderPass Manager instrumentation) */
extern _Atomic uint64_t g_mglEncoderCreationsSinceSwap;   /* newRenderEncoderLocked calls */
extern _Atomic uint64_t g_mglEncoderFBORotationsSinceSwap; /* FBO-change driven rotations */

/* Parallel-group planning (Stage 5.1) */
extern _Atomic uint64_t g_mglParallelGroupsSinceSwap;          /* groups computed per swap */
extern _Atomic uint64_t g_mglParallelGroupBatchesSinceSwap;     /* batches inside groups */
extern _Atomic uint64_t g_mglLargestParallelGroupSinceSwap;     /* largest group batch count */
extern _Atomic uint64_t g_mglParallelEncodeEligibleBatchesSinceSwap; /* batches in groups with ≥2 members (parallel-encode candidates) */

/* Batch merge rejection reasons */
extern _Atomic uint64_t g_mglMergeRejectStateDiffersSinceSwap;
extern _Atomic uint64_t g_mglMergeRejectBufferHazardSinceSwap;
extern _Atomic uint64_t g_mglMergeRejectUnsafeBuiltinSinceSwap;
extern _Atomic uint64_t g_mglMergeRejectExcludedLayoutSinceSwap;
extern _Atomic uint64_t g_mglMergeRejectAppendFailedSinceSwap;

/* Lock timing (seconds) */
extern _Atomic double   g_mglLockWaitTimeSinceSwap;   /* time waiting to acquire lock */
extern _Atomic double   g_mglLockHoldTimeSinceSwap;   /* time holding lock */

int mglPerfSummaryEnabled(void);
int mglPerfLockTimingEnabled(void);
uint64_t mglPerfSummaryInterval(void);

#define MGL_FRAME_LOAD(var) \
    atomic_load_explicit(&(var), memory_order_relaxed)
#define MGL_FRAME_STORE(var, value) \
    atomic_store_explicit(&(var), (value), memory_order_relaxed)
#define MGL_FRAME_ADD(var, value) \
    ((void)atomic_fetch_add_explicit(&(var), (value), memory_order_relaxed))
#define MGL_FRAME_INC(var) \
    MGL_FRAME_ADD((var), 1)
#define MGL_PERF_ADD(var, value) \
    do { if (mglPerfSummaryEnabled()) MGL_FRAME_ADD((var), (value)); } while (0)
#define MGL_PERF_INC(var) \
    MGL_PERF_ADD((var), 1)

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

/* NOTE: each individual read below is atomic (memory_order_relaxed), but the
 * multi-counter snapshot is NOT a consistent point-in-time view — the render
 * thread may update counters between individual loads.  This is acceptable
 * for statistics/watchdog purposes where approximate values are sufficient. */
static inline MGLSwapDrawCounters mglSnapshotSwapDrawCounters(void)
{
    MGLSwapDrawCounters counters;
    counters.draw_arrays            = MGL_FRAME_LOAD(g_mglDrawArraysSinceSwap);
    counters.draw_elements          = MGL_FRAME_LOAD(g_mglDrawElementsSinceSwap);
    counters.array_vertices         = MGL_FRAME_LOAD(g_mglDrawArrayVerticesSinceSwap);
    counters.element_indices        = MGL_FRAME_LOAD(g_mglDrawElementIndicesSinceSwap);
    counters.draw_arrays_skipped    = MGL_FRAME_LOAD(g_mglDrawArraysSkippedSinceSwap);
    counters.draw_elements_skipped  = MGL_FRAME_LOAD(g_mglDrawElementsSkippedSinceSwap);
    counters.process_draw_calls     = MGL_FRAME_LOAD(g_mglProcessDrawCallsSinceSwap);
    return counters;
}

static inline void mglResetSwapDrawCounters(void)
{
    MGL_FRAME_STORE(g_mglDrawArraysSinceSwap,          0);
    MGL_FRAME_STORE(g_mglDrawElementsSinceSwap,        0);
    MGL_FRAME_STORE(g_mglDrawArrayVerticesSinceSwap,   0);
    MGL_FRAME_STORE(g_mglDrawElementIndicesSinceSwap,  0);
    MGL_FRAME_STORE(g_mglDrawArraysSkippedSinceSwap,   0);
    MGL_FRAME_STORE(g_mglDrawElementsSkippedSinceSwap, 0);
    MGL_FRAME_STORE(g_mglProcessDrawCallsSinceSwap,    0);
}

/* === Perf summary snapshot === */

typedef struct MGLPerfCounters {
    /* Draw path */
    uint64_t draw_direct;
    uint64_t draw_mdi;
    uint64_t draw_stream_merged;
    uint64_t draw_skipped;
    /* Batches */
    uint64_t batches_direct;
    uint64_t batches_mdi;
    uint64_t batches_stream_merged;
    /* Pipeline cache */
    uint64_t pipeline_cache_hits;
    uint64_t pipeline_cache_misses;
    uint64_t pipeline_cache_evictions;
    /* Shader compile */
    uint64_t shader_compiles;
    double   shader_compile_time;
    /* Encoder state calls */
    uint64_t set_vertex_buffer_calls;
    uint64_t set_fragment_buffer_calls;
    uint64_t set_render_pipeline_state_calls;
    /* Encoder state skips */
    uint64_t set_vertex_buffer_skips;
    uint64_t set_fragment_buffer_skips;
    uint64_t set_render_pipeline_state_skips;
    /* Render encoder lifecycle */
    uint64_t encoder_creations;
    uint64_t encoder_fbo_rotations;
    /* Parallel-group planning */
    uint64_t parallel_groups;
    uint64_t parallel_group_batches;
    uint64_t largest_parallel_group;
    uint64_t parallel_encode_eligible_batches;
    /* Merge rejections */
    uint64_t merge_reject_state_differs;
    uint64_t merge_reject_buffer_hazard;
    uint64_t merge_reject_unsafe_builtin;
    uint64_t merge_reject_excluded_layout;
    uint64_t merge_reject_append_failed;
    /* Lock timing */
    double   lock_wait_time;
    double   lock_hold_time;
} MGLPerfCounters;

static inline MGLPerfCounters mglSnapshotPerfCounters(void)
{
    MGLPerfCounters c;
    c.draw_direct           = MGL_FRAME_LOAD(g_mglDrawDirectSinceSwap);
    c.draw_mdi              = MGL_FRAME_LOAD(g_mglDrawMDISinceSwap);
    c.draw_stream_merged    = MGL_FRAME_LOAD(g_mglDrawStreamMergedSinceSwap);
    c.draw_skipped          = MGL_FRAME_LOAD(g_mglDrawSkippedSinceSwap);
    c.batches_direct        = MGL_FRAME_LOAD(g_mglBatchesDirectSinceSwap);
    c.batches_mdi           = MGL_FRAME_LOAD(g_mglBatchesMDISinceSwap);
    c.batches_stream_merged = MGL_FRAME_LOAD(g_mglBatchesStreamMergedSinceSwap);
    c.pipeline_cache_hits   = MGL_FRAME_LOAD(g_mglPipelineCacheHitsSinceSwap);
    c.pipeline_cache_misses = MGL_FRAME_LOAD(g_mglPipelineCacheMissesSinceSwap);
    c.pipeline_cache_evictions = MGL_FRAME_LOAD(g_mglPipelineCacheEvictionsSinceSwap);
    c.shader_compiles       = MGL_FRAME_LOAD(g_mglShaderCompilesSinceSwap);
    c.shader_compile_time   = MGL_FRAME_LOAD(g_mglShaderCompileTimeSinceSwap);
    c.set_vertex_buffer_calls    = MGL_FRAME_LOAD(g_mglSetVertexBufferCallsSinceSwap);
    c.set_fragment_buffer_calls  = MGL_FRAME_LOAD(g_mglSetFragmentBufferCallsSinceSwap);
    c.set_render_pipeline_state_calls = MGL_FRAME_LOAD(g_mglSetRenderPipelineStateCallsSinceSwap);
    c.set_vertex_buffer_skips    = MGL_FRAME_LOAD(g_mglSetVertexBufferSkipsSinceSwap);
    c.set_fragment_buffer_skips  = MGL_FRAME_LOAD(g_mglSetFragmentBufferSkipsSinceSwap);
    c.set_render_pipeline_state_skips = MGL_FRAME_LOAD(g_mglSetRenderPipelineStateSkipsSinceSwap);
    c.encoder_creations     = MGL_FRAME_LOAD(g_mglEncoderCreationsSinceSwap);
    c.encoder_fbo_rotations = MGL_FRAME_LOAD(g_mglEncoderFBORotationsSinceSwap);
    c.parallel_groups          = MGL_FRAME_LOAD(g_mglParallelGroupsSinceSwap);
    c.parallel_group_batches   = MGL_FRAME_LOAD(g_mglParallelGroupBatchesSinceSwap);
    c.largest_parallel_group   = MGL_FRAME_LOAD(g_mglLargestParallelGroupSinceSwap);
    c.parallel_encode_eligible_batches = MGL_FRAME_LOAD(g_mglParallelEncodeEligibleBatchesSinceSwap);
    c.merge_reject_state_differs   = MGL_FRAME_LOAD(g_mglMergeRejectStateDiffersSinceSwap);
    c.merge_reject_buffer_hazard   = MGL_FRAME_LOAD(g_mglMergeRejectBufferHazardSinceSwap);
    c.merge_reject_unsafe_builtin  = MGL_FRAME_LOAD(g_mglMergeRejectUnsafeBuiltinSinceSwap);
    c.merge_reject_excluded_layout = MGL_FRAME_LOAD(g_mglMergeRejectExcludedLayoutSinceSwap);
    c.merge_reject_append_failed   = MGL_FRAME_LOAD(g_mglMergeRejectAppendFailedSinceSwap);
    c.lock_wait_time          = MGL_FRAME_LOAD(g_mglLockWaitTimeSinceSwap);
    c.lock_hold_time          = MGL_FRAME_LOAD(g_mglLockHoldTimeSinceSwap);
    return c;
}

static inline void mglResetPerfCounters(void)
{
    MGL_FRAME_STORE(g_mglDrawDirectSinceSwap, 0);
    MGL_FRAME_STORE(g_mglDrawMDISinceSwap, 0);
    MGL_FRAME_STORE(g_mglDrawStreamMergedSinceSwap, 0);
    MGL_FRAME_STORE(g_mglDrawSkippedSinceSwap, 0);
    MGL_FRAME_STORE(g_mglBatchesDirectSinceSwap, 0);
    MGL_FRAME_STORE(g_mglBatchesMDISinceSwap, 0);
    MGL_FRAME_STORE(g_mglBatchesStreamMergedSinceSwap, 0);
    MGL_FRAME_STORE(g_mglPipelineCacheHitsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglPipelineCacheMissesSinceSwap, 0);
    MGL_FRAME_STORE(g_mglPipelineCacheEvictionsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglShaderCompilesSinceSwap, 0);
    MGL_FRAME_STORE(g_mglShaderCompileTimeSinceSwap, 0.0);
    MGL_FRAME_STORE(g_mglSetVertexBufferCallsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSetFragmentBufferCallsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSetRenderPipelineStateCallsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSetVertexBufferSkipsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSetFragmentBufferSkipsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSetRenderPipelineStateSkipsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglEncoderCreationsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglEncoderFBORotationsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglParallelGroupsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglParallelGroupBatchesSinceSwap, 0);
    MGL_FRAME_STORE(g_mglLargestParallelGroupSinceSwap, 0);
    MGL_FRAME_STORE(g_mglParallelEncodeEligibleBatchesSinceSwap, 0);
    MGL_FRAME_STORE(g_mglMergeRejectStateDiffersSinceSwap, 0);
    MGL_FRAME_STORE(g_mglMergeRejectBufferHazardSinceSwap, 0);
    MGL_FRAME_STORE(g_mglMergeRejectUnsafeBuiltinSinceSwap, 0);
    MGL_FRAME_STORE(g_mglMergeRejectExcludedLayoutSinceSwap, 0);
    MGL_FRAME_STORE(g_mglMergeRejectAppendFailedSinceSwap, 0);
    MGL_FRAME_STORE(g_mglLockWaitTimeSinceSwap, 0.0);
    MGL_FRAME_STORE(g_mglLockHoldTimeSinceSwap, 0.0);
}

/* Print per-frame perf summary if MGL_PERF_SUMMARY=1.  frame_interval_ms is
 * the CPU time between swaps.  No-op when env-var is not set. */
void mglPrintPerfSummary(double frame_interval_ms);

#ifdef __cplusplus
}
#endif

#endif /* MGL_FRAME_ACTIVITY_H */
