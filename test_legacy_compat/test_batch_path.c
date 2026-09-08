/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_batch_select_path (O2.1). No Metal required.
 */

#include "mgl_batch_path.h"

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

static MGLBatchSelectInputs base_in(void)
{
    MGLBatchSelectInputs in;
    memset(&in, 0, sizeof(in));
    in.command_count = 4u;
    in.mdi_compatible = 1u;
    in.primitive_type = 0u;
    in.icb_os_supported = 1u;
    return in;
}

static void test_empty_and_mixed(void)
{
    MGLBatchSelectInputs in = {0};
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_DIRECT,
           "null/empty→DIRECT");
    in = base_in();
    in.sampler_snapshots_mixed = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_DIRECT,
           "mixed sampler→DIRECT");
}

static void test_cull_forces_direct(void)
{
    MGLBatchSelectInputs in = base_in();
    in.uses_cull_distance = 1u;
    in.stream_merged = 1u;
    in.enable_icb = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_DIRECT,
           "cull_distance beats stream/ICB");
}

static void test_stream_merge(void)
{
    MGLBatchSelectInputs in = base_in();
    in.stream_merged = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_STREAM_MERGE,
           "stream_merged→STREAM_MERGE");
}

static void test_icb(void)
{
    MGLBatchSelectInputs in = base_in();
    in.enable_icb = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_ICB, "ICB when enabled");
    in.disable_icb = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_MDI,
           "DISABLE_ICB falls through to MDI");
    in = base_in();
    in.enable_icb = 1u;
    in.icb_os_supported = 0u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_MDI, "no OS ICB → MDI");
    in = base_in();
    in.enable_icb = 1u;
    in.has_dynamic_uniform_bindings = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_MDI,
           "dynamic UBO blocks ICB");
}

static void test_mdi_and_restart(void)
{
    MGLBatchSelectInputs in = base_in();
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_MDI, "default→MDI");
    in.command_count = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_DIRECT,
           "below MDI min→DIRECT");
    in = base_in();
    in.polygon_mode_point = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_DIRECT,
           "polygon point→DIRECT");
    in = base_in();
    in.uses_elements = 1u;
    in.primitive_restart = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_DIRECT,
           "element restart→DIRECT");
    in = base_in();
    in.disable_mdi = 1u;
    expect(mgl_batch_select_path(&in) == MGL_BATCH_SELECT_DIRECT,
           "DISABLE_MDI→DIRECT");
}

int main(void)
{
    test_empty_and_mixed();
    test_cull_forces_direct();
    test_stream_merge();
    test_icb();
    test_mdi_and_restart();
    if (g_fails) {
        fprintf(stderr, "test_batch_path: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("test_batch_path: ok\n");
    return 0;
}
