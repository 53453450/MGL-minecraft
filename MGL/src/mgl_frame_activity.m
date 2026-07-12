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

#import <Foundation/Foundation.h>
#import "mgl_frame_activity.h"

#include <stdatomic.h>
#include <stdlib.h>

/* Env-var caches can be queried from render and shader-compile paths. */
static _Atomic int g_perf_summary_enabled = -1;
static _Atomic int g_perf_lock_timing_enabled = -1;
static _Atomic uint64_t g_perf_summary_interval = 0;
static const uint64_t kMGLDefaultPerfSummaryInterval = 60ull;
static const uint64_t kMGLMinSafePerfSummaryInterval = 30ull;

int mglPerfSummaryEnabled(void)
{
    int cached = atomic_load_explicit(&g_perf_summary_enabled, memory_order_acquire);
    if (cached < 0) {
        const char *env = getenv("MGL_PERF_SUMMARY");
        cached = (env && atoi(env) > 0) ? 1 : 0;
        atomic_store_explicit(&g_perf_summary_enabled, cached, memory_order_release);
    }
    return cached != 0;
}

int mglPerfLockTimingEnabled(void)
{
    int cached = atomic_load_explicit(&g_perf_lock_timing_enabled, memory_order_acquire);
    if (cached < 0) {
        const char *env = getenv("MGL_PERF_LOCK_TIMING");
        cached = (env && atoi(env) > 0) ? 1 : 0;
        atomic_store_explicit(&g_perf_lock_timing_enabled, cached, memory_order_release);
    }
    return cached != 0;
}

uint64_t mglPerfSummaryInterval(void)
{
    uint64_t cached = atomic_load_explicit(&g_perf_summary_interval, memory_order_acquire);
    if (cached == 0) {
        const char *env = getenv("MGL_PERF_SUMMARY_EVERY");
        long parsed = env ? strtol(env, NULL, 10) : 0;
        cached = parsed > 0 ? (uint64_t)parsed : kMGLDefaultPerfSummaryInterval;
        if (cached < kMGLMinSafePerfSummaryInterval) {
            const char *unsafeEveryFrame = getenv("MGL_PERF_SUMMARY_UNSAFE_EVERY_FRAME");
            BOOL allowUnsafeEveryFrame =
                unsafeEveryFrame && atoi(unsafeEveryFrame) > 0;
            cached = allowUnsafeEveryFrame ? cached : kMGLMinSafePerfSummaryInterval;
        }
        atomic_store_explicit(&g_perf_summary_interval, cached, memory_order_release);
    }
    return cached;
}

/* === os_signpost instrumentation === */

os_log_t mglSignpostLog = OS_LOG_DEFAULT;

static int _signpostEnabledCache = -1;

int mglSignpostEnabled(void)
{
    if (_signpostEnabledCache < 0) {
        const char *env = getenv("MGL_SIGNPOST");
        _signpostEnabledCache = (env && atoi(env) == 1) ? 1 : 0;
    }
    return _signpostEnabledCache;
}

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

/* === Performance counters === */

_Atomic uint64_t g_mglDrawDirectSinceSwap        = 0;
_Atomic uint64_t g_mglDrawMDISinceSwap           = 0;
_Atomic uint64_t g_mglDrawStreamMergedSinceSwap  = 0;
_Atomic uint64_t g_mglDrawSkippedSinceSwap       = 0;
_Atomic uint64_t g_mglBatchesDirectSinceSwap     = 0;
_Atomic uint64_t g_mglBatchesMDISinceSwap        = 0;
_Atomic uint64_t g_mglBatchesStreamMergedSinceSwap = 0;
_Atomic uint64_t g_mglPipelineCacheHitsSinceSwap    = 0;
_Atomic uint64_t g_mglPipelineCacheMissesSinceSwap  = 0;
_Atomic uint64_t g_mglPipelineCacheEvictionsSinceSwap = 0;
_Atomic uint64_t g_mglShaderCompilesSinceSwap      = 0;
_Atomic double   g_mglShaderCompileTimeSinceSwap   = 0.0;
_Atomic uint64_t g_mglSetVertexBufferCallsSinceSwap    = 0;
_Atomic uint64_t g_mglSetFragmentBufferCallsSinceSwap  = 0;
_Atomic uint64_t g_mglSetRenderPipelineStateCallsSinceSwap = 0;
_Atomic uint64_t g_mglSetVertexBufferSkipsSinceSwap    = 0;
_Atomic uint64_t g_mglSetFragmentBufferSkipsSinceSwap  = 0;
_Atomic uint64_t g_mglSetRenderPipelineStateSkipsSinceSwap = 0;
_Atomic uint64_t g_mglEncoderCreationsSinceSwap        = 0;
_Atomic uint64_t g_mglEncoderFBORotationsSinceSwap     = 0;
_Atomic uint64_t g_mglParallelGroupsSinceSwap          = 0;
_Atomic uint64_t g_mglParallelGroupBatchesSinceSwap     = 0;
_Atomic uint64_t g_mglLargestParallelGroupSinceSwap     = 0;
_Atomic uint64_t g_mglParallelEncodeEligibleBatchesSinceSwap = 0;
_Atomic uint64_t g_mglMergeRejectStateDiffersSinceSwap   = 0;
_Atomic uint64_t g_mglMergeRejectBufferHazardSinceSwap   = 0;
_Atomic uint64_t g_mglMergeRejectUnsafeBuiltinSinceSwap  = 0;
_Atomic uint64_t g_mglMergeRejectExcludedLayoutSinceSwap = 0;
_Atomic uint64_t g_mglMergeRejectAppendFailedSinceSwap   = 0;
_Atomic double   g_mglLockWaitTimeSinceSwap   = 0.0;
_Atomic double   g_mglLockHoldTimeSinceSwap   = 0.0;

/* === Stage 1 CPU audit counters === */

_Atomic uint64_t g_mglDepthStencilStateCreatesSinceSwap   = 0;
_Atomic uint64_t g_mglDepthStencilStateSkipsSinceSwap     = 0;
_Atomic uint64_t g_mglSnapshotBytesAllocatedSinceSwap     = 0;
_Atomic uint64_t g_mglSnapshotAllocationCountSinceSwap    = 0;
_Atomic uint64_t g_mglReplayMemcpyCountSinceSwap          = 0;
_Atomic uint64_t g_mglHazardActiveBindingsSinceSwap       = 0;
_Atomic uint64_t g_mglHazardRangeCountSinceSwap           = 0;
_Atomic uint64_t g_mglHazardOverflowFlushesSinceSwap      = 0;
_Atomic uint64_t g_mglPSODedupHitsSinceSwap               = 0;
_Atomic uint64_t g_mglPSODedupMissesSinceSwap             = 0;

#include <mach/mach_time.h>

void mglPrintPerfSummary(double frame_interval_ms)
{
    if (!mglPerfSummaryEnabled()) return;
    static _Atomic uint64_t s_perf_summary_print_count = 0;
    uint64_t hit = atomic_fetch_add_explicit(&s_perf_summary_print_count, 1, memory_order_relaxed) + 1;
    uint64_t interval = mglPerfSummaryInterval();
    if (interval > 1 && (hit % interval) != 0) {
        return;
    }

    MGLPerfCounters c = mglSnapshotPerfCounters();

    /* One concise line per frame */
    NSLog(@"MGL PERF: frame=%.1fms | draws: dir=%llu mdi=%llu sm=%llu skip=%llu | "
          @"batches: d=%llu m=%llu sm=%llu | pipe: hit=%llu miss=%llu evict=%llu | "
          @"shaders: %llu/%.1fms | enc: vb=%llu(%llu skip) fb=%llu(%llu skip) ps=%llu(%llu skip) | "
          @"encoder: new=%llu fboRot=%llu | "
          @"pgrp: g=%llu batches=%llu max=%llu elig=%llu | "
          @"merge rej: sd=%llu bh=%llu ub=%llu el=%llu af=%llu | "
          @"lock: wait=%.1fms hold=%.1fms | "
          @"ds: creates=%llu skips=%llu | "
          @"snap: bytes=%llu allocs=%llu | "
          @"replay: memcpy=%llu | "
          @"hazard: active=%llu ranges=%llu overflow=%llu | "
          @"pso_dedup: hits=%llu misses=%llu",
          frame_interval_ms,
          c.draw_direct, c.draw_mdi, c.draw_stream_merged, c.draw_skipped,
          c.batches_direct, c.batches_mdi, c.batches_stream_merged,
          c.pipeline_cache_hits, c.pipeline_cache_misses, c.pipeline_cache_evictions,
          c.shader_compiles, c.shader_compile_time * 1000.0,
          c.set_vertex_buffer_calls, c.set_vertex_buffer_skips,
          c.set_fragment_buffer_calls, c.set_fragment_buffer_skips,
          c.set_render_pipeline_state_calls, c.set_render_pipeline_state_skips,
          c.encoder_creations, c.encoder_fbo_rotations,
          c.parallel_groups, c.parallel_group_batches, c.largest_parallel_group,
          c.parallel_encode_eligible_batches,
          c.merge_reject_state_differs, c.merge_reject_buffer_hazard,
          c.merge_reject_unsafe_builtin, c.merge_reject_excluded_layout,
          c.merge_reject_append_failed,
          c.lock_wait_time * 1000.0, c.lock_hold_time * 1000.0,
          c.ds_state_creates, c.ds_state_skips,
          c.snapshot_bytes_allocated, c.snapshot_allocation_count,
          c.replay_memcpy_count,
          c.hazard_active_bindings, c.hazard_range_count, c.hazard_overflow_flushes,
          c.pso_dedup_hits, c.pso_dedup_misses);

    if (frame_interval_ms > 33.0) {
        NSLog(@"MGL PERF SLOW FRAME: %.1fms — see counters above for breakdown", frame_interval_ms);
    }
}
