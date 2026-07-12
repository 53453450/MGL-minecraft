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
#include <os/signpost.h>

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

/* Depth/stencil state (Stage 1 — CPU audit gated opts) */
extern _Atomic uint64_t g_mglDepthStencilStateCreatesSinceSwap;  /* newDepthStencilStateWithDescriptor: calls */
extern _Atomic uint64_t g_mglDepthStencilStateSkipsSinceSwap;    /* setDepthStencilState: skipped by dedup */

/* Snapshot allocation (Stage 1) */
extern _Atomic uint64_t g_mglSnapshotBytesAllocatedSinceSwap;    /* bytes malloc'd for state+vao snapshots */
extern _Atomic uint64_t g_mglSnapshotAllocationCountSinceSwap;   /* snapshot malloc count */

/* State replay (Stage 1) */
extern _Atomic uint64_t g_mglReplayMemcpyCountSinceSwap;         /* memcpy calls in restoreStateForBatch: */

/* Hazard tracking (Stage 1) */
extern _Atomic uint64_t g_mglHazardActiveBindingsSinceSwap;      /* sampled active base-buffer binding count per draw */
extern _Atomic uint64_t g_mglHazardRangeCountSinceSwap;          /* sampled buffer_read_range_count per draw */
extern _Atomic uint64_t g_mglHazardOverflowFlushesSinceSwap;     /* overflow-triggered full flushes */

/* PSO dedup (Stage 1 — declared for future Task 5 instrumentation) */
extern _Atomic uint64_t g_mglPSODedupHitsSinceSwap;              /* PSO dedup fast path hits */
extern _Atomic uint64_t g_mglPSODedupMissesSinceSwap;            /* PSO dedup fast path misses */

/* Flush reasons + same-key restore instrumentation (100ms encoder kill path) */
extern _Atomic uint64_t g_mglFlushTotalSinceSwap;
extern _Atomic uint64_t g_mglFlushReasonBindTextureSinceSwap;
extern _Atomic uint64_t g_mglFlushReasonBindBufferSinceSwap;
extern _Atomic uint64_t g_mglFlushReasonTexWriteSinceSwap;
extern _Atomic uint64_t g_mglFlushReasonBufferRangeSinceSwap;
extern _Atomic uint64_t g_mglFlushReasonActiveTexWarSinceSwap;
extern _Atomic uint64_t g_mglFlushReasonCapacitySinceSwap;
extern _Atomic uint64_t g_mglFlushReasonOtherSinceSwap;
extern _Atomic uint64_t g_mglSameKeyRestoreSkipsSinceSwap;
extern _Atomic uint64_t g_mglSameKeyOracleWouldSkipSinceSwap;
extern _Atomic uint64_t g_mglDirtyKeyDeltaNarrowSinceSwap;
extern _Atomic uint64_t g_mglBatchesReplayedSinceSwap;

int mglPerfSummaryEnabled(void);
int mglPerfLockTimingEnabled(void);
uint64_t mglPerfSummaryInterval(void);

/* === os_signpost instrumentation (gated by MGL_SIGNPOST=1 env var) ===
 *
 * When MGL_SIGNPOST=1 is set, interval signposts are emitted to the default
 * os_log handle for visualization in Instruments.  When unset, the
 * mglSignpostEnabled() check in each macro evaluates to false and the branch
 * predictor learns to skip it, so per-call overhead is ~1 cycle.  The cache
 * is queried once and cached for the process lifetime.
 *
 * os_signpost_interval_begin/end are safe to call from any thread. */

extern os_log_t mglSignpostLog;

/* Returns 1 if MGL_SIGNPOST=1 is set, 0 otherwise.  Cached after first call. */
int mglSignpostEnabled(void);

/* Signpost interval macros.  These use a bare `if` (no braces), so they MUST
 * be used as standalone statements — never inside other expressions, and
 * never as the body of an if/else/for/while without enclosing braces.  When
 * MGL_SIGNPOST is not set the check evaluates to false and the os_signpost
 * call is skipped entirely. */
#define MGL_SIGNPOST_BEGIN(name) \
    if (mglSignpostEnabled()) os_signpost_interval_begin(mglSignpostLog, OS_SIGNPOST_ID_EXCLUSIVE, #name)

#define MGL_SIGNPOST_END(name) \
    if (mglSignpostEnabled()) os_signpost_interval_end(mglSignpostLog, OS_SIGNPOST_ID_EXCLUSIVE, #name)

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
    /* Depth/stencil state */
    uint64_t ds_state_creates;
    uint64_t ds_state_skips;
    /* Snapshot allocation */
    uint64_t snapshot_bytes_allocated;
    uint64_t snapshot_allocation_count;
    /* State replay */
    uint64_t replay_memcpy_count;
    /* Hazard tracking */
    uint64_t hazard_active_bindings;
    uint64_t hazard_range_count;
    uint64_t hazard_overflow_flushes;
    /* PSO dedup */
    uint64_t pso_dedup_hits;
    uint64_t pso_dedup_misses;
    /* Flush reasons + same-key restore */
    uint64_t flush_total;
    uint64_t flush_bind_texture;
    uint64_t flush_bind_buffer;
    uint64_t flush_tex_write;
    uint64_t flush_buffer_range;
    uint64_t flush_active_tex_war;
    uint64_t flush_capacity;
    uint64_t flush_other;
    uint64_t same_key_restore_skips;
    uint64_t same_key_oracle_would_skip;
    uint64_t dirty_key_delta_narrow;
    uint64_t batches_replayed;
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
    c.ds_state_creates        = MGL_FRAME_LOAD(g_mglDepthStencilStateCreatesSinceSwap);
    c.ds_state_skips          = MGL_FRAME_LOAD(g_mglDepthStencilStateSkipsSinceSwap);
    c.snapshot_bytes_allocated = MGL_FRAME_LOAD(g_mglSnapshotBytesAllocatedSinceSwap);
    c.snapshot_allocation_count = MGL_FRAME_LOAD(g_mglSnapshotAllocationCountSinceSwap);
    c.replay_memcpy_count     = MGL_FRAME_LOAD(g_mglReplayMemcpyCountSinceSwap);
    c.hazard_active_bindings  = MGL_FRAME_LOAD(g_mglHazardActiveBindingsSinceSwap);
    c.hazard_range_count      = MGL_FRAME_LOAD(g_mglHazardRangeCountSinceSwap);
    c.hazard_overflow_flushes = MGL_FRAME_LOAD(g_mglHazardOverflowFlushesSinceSwap);
    c.pso_dedup_hits          = MGL_FRAME_LOAD(g_mglPSODedupHitsSinceSwap);
    c.pso_dedup_misses        = MGL_FRAME_LOAD(g_mglPSODedupMissesSinceSwap);
    c.flush_total             = MGL_FRAME_LOAD(g_mglFlushTotalSinceSwap);
    c.flush_bind_texture      = MGL_FRAME_LOAD(g_mglFlushReasonBindTextureSinceSwap);
    c.flush_bind_buffer       = MGL_FRAME_LOAD(g_mglFlushReasonBindBufferSinceSwap);
    c.flush_tex_write         = MGL_FRAME_LOAD(g_mglFlushReasonTexWriteSinceSwap);
    c.flush_buffer_range      = MGL_FRAME_LOAD(g_mglFlushReasonBufferRangeSinceSwap);
    c.flush_active_tex_war    = MGL_FRAME_LOAD(g_mglFlushReasonActiveTexWarSinceSwap);
    c.flush_capacity          = MGL_FRAME_LOAD(g_mglFlushReasonCapacitySinceSwap);
    c.flush_other             = MGL_FRAME_LOAD(g_mglFlushReasonOtherSinceSwap);
    c.same_key_restore_skips  = MGL_FRAME_LOAD(g_mglSameKeyRestoreSkipsSinceSwap);
    c.same_key_oracle_would_skip = MGL_FRAME_LOAD(g_mglSameKeyOracleWouldSkipSinceSwap);
    c.dirty_key_delta_narrow  = MGL_FRAME_LOAD(g_mglDirtyKeyDeltaNarrowSinceSwap);
    c.batches_replayed        = MGL_FRAME_LOAD(g_mglBatchesReplayedSinceSwap);
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
    MGL_FRAME_STORE(g_mglDepthStencilStateCreatesSinceSwap, 0);
    MGL_FRAME_STORE(g_mglDepthStencilStateSkipsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSnapshotBytesAllocatedSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSnapshotAllocationCountSinceSwap, 0);
    MGL_FRAME_STORE(g_mglReplayMemcpyCountSinceSwap, 0);
    MGL_FRAME_STORE(g_mglHazardActiveBindingsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglHazardRangeCountSinceSwap, 0);
    MGL_FRAME_STORE(g_mglHazardOverflowFlushesSinceSwap, 0);
    MGL_FRAME_STORE(g_mglPSODedupHitsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglPSODedupMissesSinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushTotalSinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushReasonBindTextureSinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushReasonBindBufferSinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushReasonTexWriteSinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushReasonBufferRangeSinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushReasonActiveTexWarSinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushReasonCapacitySinceSwap, 0);
    MGL_FRAME_STORE(g_mglFlushReasonOtherSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSameKeyRestoreSkipsSinceSwap, 0);
    MGL_FRAME_STORE(g_mglSameKeyOracleWouldSkipSinceSwap, 0);
    MGL_FRAME_STORE(g_mglDirtyKeyDeltaNarrowSinceSwap, 0);
    MGL_FRAME_STORE(g_mglBatchesReplayedSinceSwap, 0);
}

/* Print per-frame perf summary if MGL_PERF_SUMMARY=1.  frame_interval_ms is
 * the CPU time between swaps.  No-op when env-var is not set.
 * NOTE: the env *name* must be exactly MGL_PERF_SUMMARY — a leading space
 * in the variable name (e.g. " MGL_PERF_SUMMARY") will not be found. */
void mglPrintPerfSummary(double frame_interval_ms);

#ifdef __cplusplus
}
#endif

#endif /* MGL_FRAME_ACTIVITY_H */
