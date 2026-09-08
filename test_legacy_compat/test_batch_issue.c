/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_batch_issue_* + mgl_batch_rt_* (A3 / O2.5). No Metal.
 */

#include "mgl_batch_issue.h"
#include "mgl_batch_rt_mark.h"
#include "mgl_batch_path.h"
#include "mgl_batch_restore.h"

#include <stdio.h>
#include <string.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static void test_rt_mark(void)
{
    expect(mgl_batch_rt_attachment_active(0x5u, 0, 8) == 1, "att0 active");
    expect(mgl_batch_rt_attachment_active(0x5u, 1, 8) == 0, "att1 inactive");
    expect(mgl_batch_rt_attachment_active(0x5u, 2, 8) == 1, "att2 active");
    expect(mgl_batch_rt_attachment_active(0x1u, 8, 8) == 0, "oob inactive");

    expect(mgl_batch_rt_yflip_authority(1, 1, 0, 0) == 1, "yflip ok");
    expect(mgl_batch_rt_yflip_authority(1, 1, 1, 0) == 0, "InSampler blocks");
    expect(mgl_batch_rt_yflip_authority(1, 1, 0, 1) == 0, "Diffuse blocks");
    expect(mgl_batch_rt_yflip_authority(0, 1, 0, 0) == 0, "no inject");
    expect(mgl_batch_rt_yflip_authority(1, 0, 0, 0) == 0, "not explicit");

    expect(mgl_batch_rt_should_trace_write_mark(1) == 1, "hit1");
    expect(mgl_batch_rt_should_trace_write_mark(128) == 1, "hit128");
    expect(mgl_batch_rt_should_trace_write_mark(129) == 0, "hit129");
    expect(mgl_batch_rt_should_trace_write_mark(256) == 1, "hit256");
}

static void test_stream_mdi_gate(void)
{
    MGLBatchStreamMdiGateIn in;
    size_t needed = 0;
    memset(&in, 0, sizeof(in));
    in.stream_merged = 1u;
    in.has_encoder = 1u;
    in.command_count = 4u;
    in.stream_index_count = 12u;
    in.arg_size = 20u;
    expect(mgl_batch_issue_stream_mdi_gate(&in, &needed) ==
               MGL_BATCH_STREAM_MDI_OK,
           "ok");
    expect(needed == 80u, "needed bytes");
    in.disable_mdi = 1u;
    expect(mgl_batch_issue_stream_mdi_gate(&in, &needed) ==
               MGL_BATCH_STREAM_MDI_FAIL_DISABLED,
           "disabled");
    in.disable_mdi = 0u;
    in.primitive_type = 0xFFu;
    expect(mgl_batch_issue_stream_mdi_gate(&in, &needed) ==
               MGL_BATCH_STREAM_MDI_FAIL_BAD_PRIM,
           "bad prim");
    in.primitive_type = 0u;
    in.has_encoder = 0u;
    expect(mgl_batch_issue_stream_mdi_gate(&in, &needed) ==
               MGL_BATCH_STREAM_MDI_FAIL_EMPTY,
           "no encoder");
}

static void test_direct_arrays_and_dyn(void)
{
    int32_t ic = 0;
    uint32_t bi = 0;
    const char *reason = NULL;
    const char *cull = NULL;
    mgl_batch_issue_direct_arrays_params(MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS, 9, 3,
                                         &ic, &bi, &reason, &cull);
    expect(ic == 1 && bi == 0 && reason && cull, "arrays params");
    mgl_batch_issue_direct_arrays_params(
        MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE, 9, 3, &ic,
        &bi, &reason, &cull);
    expect(ic == 9 && bi == 3u, "base instance params");

    expect(mgl_batch_issue_dyn_cmd_has_bindings(0, 0, 0) == 0, "no dyn");
    expect(mgl_batch_issue_dyn_cmd_has_bindings(1, 0, 0) == 1, "has vertex");
    expect(mgl_batch_issue_dyn_needs_mapper_fallback(1, 1) == 0, "no fallback");
    expect(mgl_batch_issue_dyn_needs_mapper_fallback(0, 1) == 1, "fallback");

    expect(mgl_batch_issue_cull_capture_path(0, 0) ==
               MGL_BATCH_CULL_CAPTURE_NONE,
           "cull none");
    expect(mgl_batch_issue_cull_capture_path(
               1, MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS) ==
               MGL_BATCH_CULL_CAPTURE_ARRAYS,
           "cull arrays");
    expect(mgl_batch_issue_cull_capture_path(1, 1) ==
               MGL_BATCH_CULL_CAPTURE_ELEMENTS,
           "cull elements");
}


static void test_encode_fold(void)
{
    MGLBatchFlushPathStats st;
    memset(&st, 0, sizeof(st));
    mgl_batch_flush_accum_path(&st, 2 /* STREAM */, 4);
    mgl_batch_flush_accum_path(&st, 1 /* MDI */, 3);
    mgl_batch_flush_accum_path(&st, 3 /* ICB */, 2);
    mgl_batch_flush_accum_path(&st, 0 /* DIRECT */, 1);
    expect(st.stream_batches == 1u && st.stream_commands == 4u, "stream stats");
    expect(st.mdi_batches == 1u && st.mdi_commands == 3u, "mdi stats");
    expect(st.icb_batches == 1u && st.icb_commands == 2u, "icb stats");
    expect(st.direct_batches == 1u && st.direct_commands == 1u, "direct stats");
    expect(strcmp(mgl_batch_flush_path_phase(2), "ISSUE_STREAM_MERGE") == 0,
           "phase stream");
    expect(mgl_batch_flush_should_trace_log(1, 10, 1, 0, 0) == 1, "trace hit1");
    expect(mgl_batch_flush_should_trace_log(100, 10, 1, 0, 0) == 0, "trace skip");
    expect(mgl_batch_flush_should_trace_log(100, 10, 0, 1, 0) == 1, "skipped");
    expect(mgl_batch_issue_scratch_range_ok(8, 16, 32) == 1, "scratch ok");
    expect(mgl_batch_issue_scratch_range_ok(20, 16, 32) == 0, "scratch oob");
    expect(mgl_batch_issue_should_apply_cmd_sampler(0, 1) == 1, "dyn tex samp");
    expect(mgl_batch_issue_should_apply_cmd_sampler(0, 0) == 0, "no samp");
    MGLBatchIcbArrayDrawParams ap;
    mgl_batch_issue_icb_array_draw_params(1, 2, 3, 4, &ap);
    expect(ap.vertex_start == 1u && ap.vertex_count == 2u &&
               ap.instance_count == 3u && ap.base_instance == 4u,
           "icb array params");
    expect(mgl_batch_issue_icb_command_types(1) == 2u, "icb indexed types");
    expect(mgl_batch_issue_icb_command_types(0) == 1u, "icb array types");
    MGLBatchCmdStatDelta d;
    mgl_batch_issue_cmd_stat_delta(MGL_BATCH_ISSUE_CMD_DRAW_ARRAYS, 9, 0, &d);
    expect(d.array_draws == 1u && d.array_vertices == 9ull, "array stats");
    mgl_batch_issue_cmd_stat_delta(1, 5, 1, &d);
    expect(d.element_draws == 1u && d.element_indices == 5ull, "elem stats");
    expect(mgl_batch_rt_should_cross_mark(0, 1) == 1, "cross mark");
    expect(mgl_batch_rt_should_cross_mark(1, 1) == 0, "already marked");
    expect(mgl_batch_rt_should_diag_attachment0(0, 1, 1) == 1, "diag0");
    expect(mgl_batch_rt_should_diag_attachment0(1, 1, 1) == 0, "diag1");
}

static void test_stream_index_and_sampler(void)
{
    expect(mgl_batch_issue_stream_index_ready(0, 0, 0) ==
               MGL_BATCH_STREAM_INDEX_NO_BUFFER,
           "no index buf");
    expect(mgl_batch_issue_stream_index_ready(1, 0, 1) ==
               MGL_BATCH_STREAM_INDEX_NO_BUFFER,
           "process fail");
    expect(mgl_batch_issue_stream_index_ready(1, 1, 0) ==
               MGL_BATCH_STREAM_INDEX_NO_MTL,
           "no mtl");
    expect(mgl_batch_issue_stream_index_ready(1, 1, 1) ==
               MGL_BATCH_STREAM_INDEX_OK,
           "index ok");
    expect(strcmp(mgl_batch_issue_stream_index_reason(
                      MGL_BATCH_STREAM_INDEX_NO_MTL),
                  "stream_no_mtl_index") == 0,
           "reason mtl");
    expect(mgl_batch_issue_should_apply_stable_sampler(0, 3u, 0xFFFFFFFFu) == 1,
           "apply stable");
    expect(mgl_batch_issue_should_apply_stable_sampler(1, 3u, 0xFFFFFFFFu) == 0,
           "mixed skip");
    expect(mgl_batch_issue_should_apply_stable_sampler(0, 0xFFFFFFFFu,
                                                       0xFFFFFFFFu) == 0,
           "invalid id skip");
}


static int g_dyn_steps;

static void dyn_refresh(void *ctx) { (void)ctx; g_dyn_steps += 1; }
static int dyn_has_enc(void *ctx) { (void)ctx; return 1; }
static int dyn_ok(void *ctx) { (void)ctx; return 1; }
static int dyn_fail(void *ctx) { (void)ctx; return 0; }


static uint32_t g_flush_batches;
static uint32_t g_flush_seen;
static int g_flush_issued;

static uint32_t flush_batch_count(void *ctx)
{
    (void)ctx;
    return g_flush_batches;
}
static uint32_t flush_cmd_count(void *ctx, uint32_t b)
{
    (void)ctx;
    return b == 0u ? 2u : 0u;
}
static void flush_fill_skip(void *ctx, uint32_t b, MGLBatchSameKeySkipIn *in,
                            int *want_abs)
{
    (void)ctx;
    (void)b;
    memset(in, 0, sizeof(*in));
    in->has_encoder = 1u;
    in->bind_valid = 1u;
    in->keys_equal = 0u;
    in->absolute_offsets_match = 1u;
    in->pass_matches = 1u;
    if (want_abs) *want_abs = 0;
}
static int flush_check(void *ctx, uint32_t b)
{
    (void)ctx;
    (void)b;
    g_flush_seen += 1u;
    return 1;
}
static void flush_mark(void *ctx, uint32_t b)
{
    (void)ctx;
    (void)b;
}
static int flush_sched(void *ctx, uint32_t b)
{
    (void)ctx;
    (void)b;
    return MGL_BATCH_SELECT_DIRECT;
}
static void flush_issue_direct(void *ctx, uint32_t b)
{
    (void)ctx;
    (void)b;
    g_flush_issued += 1;
}

static void test_flush_run(void)
{
    MGLBatchFlushLoopState st;
    memset(&st, 0, sizeof(st));
    g_flush_batches = 1u;
    g_flush_seen = 0u;
    g_flush_issued = 0;
    MGLBatchFlushLoopOps ops = {
        .ctx = NULL,
        .batch_count = flush_batch_count,
        .command_count = flush_cmd_count,
        .fill_skip_in = flush_fill_skip,
        .check_execute = flush_check,
        .mark_execute_ok = flush_mark,
        .schedule = flush_sched,
        .issue_direct = flush_issue_direct,
        .vao_buffer_dirty_mask = 0x3u,
    };
    mgl_batch_flush_run_batches(&st, &ops);
    expect(g_flush_seen == 1u, "checked one");
    expect(g_flush_issued == 1, "issued direct");
    expect(st.last_execute_ok == 1u, "exec ok");
    expect(st.path_stats.direct_batches == 1u, "direct path");
}

static int g_check_skip;
static int check_ok(void *ctx)
{
    (void)ctx;
    return 1;
}
static int check_no(void *ctx)
{
    (void)ctx;
    return 0;
}
static int check_skip(void *ctx, const char *phase, const char *reason)
{
    (void)ctx;
    (void)phase;
    (void)reason;
    g_check_skip += 1;
    return 0;
}

static void test_check_exec(void)
{
    g_check_skip = 0;
    MGLBatchCheckExecOps ops = {
        .ctx = NULL,
        .begin_trace = NULL,
        .prepare_fbo = check_ok,
        .process_gl_state = check_ok,
        .should_apply_sampler = check_no,
        .empty_raster = check_no,
        .fully_culled = check_no,
        .apply_polygon_offset = NULL,
        .trace_skip = check_skip,
    };
    expect(mgl_batch_check_should_execute(&ops) == 1, "execute ok");
    ops.prepare_fbo = check_no;
    expect(mgl_batch_check_should_execute(&ops) == 0, "fbo skip");
    expect(g_check_skip == 1, "skip called");
}

static void test_dyn_apply(void)
{
    g_dyn_steps = 0;
    expect(mgl_batch_issue_apply_dyn_bindings(0, 0, 0, NULL) == 0, "null ops");
    expect(mgl_batch_issue_apply_dyn_bindings(0, 0, 0, &(MGLBatchDynApplyOps){0}) ==
               1,
           "no bindings");
    MGLBatchDynApplyOps ops = {
        .ctx = NULL,
        .refresh_owner = dyn_refresh,
        .has_encoder = dyn_has_enc,
        .build_dyn_vao = dyn_ok,
        .apply_ubo = dyn_ok,
        .apply_tex = dyn_ok,
        .bind_tex_direct = dyn_ok,
        .bind_tex_mapper = dyn_fail,
        .restore_after_tex_upload = dyn_fail,
        .bind_vertex_direct = dyn_ok,
        .bind_uniform_direct = dyn_ok,
        .mapper_fallback = dyn_ok,
    };
    expect(mgl_batch_issue_apply_dyn_bindings(1, 1, 0, &ops) == 1, "vu ok");
    expect(g_dyn_steps > 0, "refreshed");
    ops.bind_vertex_direct = dyn_fail;
    ops.mapper_fallback = dyn_ok;
    expect(mgl_batch_issue_apply_dyn_bindings(1, 0, 0, &ops) == 1,
           "vertex fallback");
    ops.mapper_fallback = dyn_fail;
    expect(mgl_batch_issue_apply_dyn_bindings(1, 0, 0, &ops) == 0,
           "fallback fail");
}

int main(void)
{
    test_rt_mark();
    test_stream_mdi_gate();
    test_direct_arrays_and_dyn();
    test_stream_index_and_sampler();
    test_encode_fold();
    test_dyn_apply();
    test_flush_run();
    test_check_exec();

    expect(mgl_batch_flush_scheduled_path_perf_kind(2) ==
               MGL_BATCH_FLUSH_PERF_STREAM,
           "stream perf kind");
    expect(mgl_batch_flush_scheduled_path_perf_kind(0) ==
               MGL_BATCH_FLUSH_PERF_DIRECT,
           "direct perf kind");
    expect(mgl_batch_flush_scheduled_path_perf_kind(1) ==
               MGL_BATCH_FLUSH_PERF_NONE,
           "mdi perf kind none");
    {
        MGLBatchTraceFsSlot slots[4] = {{1u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 1u}};
        int has_rt = 0, used_copy = 0;
        mgl_batch_trace_fs_slot_flags(slots, 4, &has_rt, &used_copy);
        expect(has_rt == 1 && used_copy == 1, "fs slot flags");
    }

    if (g_fails) {
        fprintf(stderr, "test_batch_issue: %d fail(s)\n", g_fails);
        return 1;
    }
    printf("test_batch_issue: ok\n");
    return 0;
}
