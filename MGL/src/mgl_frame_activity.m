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

/* MGL_PERF_SUMMARY env-var cache.
 * Non-atomic read/write — consistent with MGL's single-threaded-per-context
 * assumption.  If multi-threaded shader compilation is ever supported, guard
 * with dispatch_once or atomic. */
static int g_perf_summary_enabled = -1;

int mglPerfSummaryEnabled(void)
{
    if (g_perf_summary_enabled < 0) {
        const char *env = getenv("MGL_PERF_SUMMARY");
        g_perf_summary_enabled = (env && atoi(env) > 0) ? 1 : 0;
    }
    return g_perf_summary_enabled != 0;
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
_Atomic uint64_t g_mglMergeRejectStateDiffersSinceSwap   = 0;
_Atomic uint64_t g_mglMergeRejectBufferHazardSinceSwap   = 0;
_Atomic uint64_t g_mglMergeRejectUnsafeBuiltinSinceSwap  = 0;
_Atomic uint64_t g_mglMergeRejectExcludedLayoutSinceSwap = 0;
_Atomic uint64_t g_mglMergeRejectAppendFailedSinceSwap   = 0;
_Atomic double   g_mglLockWaitTimeSinceSwap   = 0.0;
_Atomic double   g_mglLockHoldTimeSinceSwap   = 0.0;

#include <mach/mach_time.h>

void mglPrintPerfSummary(double frame_interval_ms)
{
    if (!mglPerfSummaryEnabled()) return;

    MGLPerfCounters c = mglSnapshotPerfCounters();

    /* One concise line per frame */
    NSLog(@"MGL PERF: frame=%.1fms | draws: dir=%llu mdi=%llu sm=%llu skip=%llu | "
          @"batches: d=%llu m=%llu sm=%llu | pipe: hit=%llu miss=%llu evict=%llu | "
          @"shaders: %llu/%.1fms | enc: vb=%llu(%llu skip) fb=%llu(%llu skip) ps=%llu(%llu skip) | "
          @"merge rej: sd=%llu bh=%llu ub=%llu el=%llu af=%llu | "
          @"lock: wait=%.1fms hold=%.1fms",
          frame_interval_ms,
          c.draw_direct, c.draw_mdi, c.draw_stream_merged, c.draw_skipped,
          c.batches_direct, c.batches_mdi, c.batches_stream_merged,
          c.pipeline_cache_hits, c.pipeline_cache_misses, c.pipeline_cache_evictions,
          c.shader_compiles, c.shader_compile_time * 1000.0,
          c.set_vertex_buffer_calls, c.set_vertex_buffer_skips,
          c.set_fragment_buffer_calls, c.set_fragment_buffer_skips,
          c.set_render_pipeline_state_calls, c.set_render_pipeline_state_skips,
          c.merge_reject_state_differs, c.merge_reject_buffer_hazard,
          c.merge_reject_unsafe_builtin, c.merge_reject_excluded_layout,
          c.merge_reject_append_failed,
          c.lock_wait_time * 1000.0, c.lock_hold_time * 1000.0);

    if (frame_interval_ms > 33.0) {
        NSLog(@"MGL PERF SLOW FRAME: %.1fms — see counters above for breakdown", frame_interval_ms);
    }
}
