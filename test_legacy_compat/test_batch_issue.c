/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_batch_issue_* + mgl_batch_rt_* (A3 / O2.5). No Metal.
 */

#include "mgl_batch_issue.h"
#include "mgl_batch_rt_mark.h"

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

int main(void)
{
    test_rt_mark();
    test_stream_mdi_gate();
    test_direct_arrays_and_dyn();
    if (g_fails) {
        fprintf(stderr, "test_batch_issue: %d fail(s)\n", g_fails);
        return 1;
    }
    printf("test_batch_issue: ok\n");
    return 0;
}
