/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_batch_icb_config / support gate (O2.4). No Metal.
 */

#include "mgl_batch_path.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static void clear_icb_env(void)
{
    unsetenv("MGL_ENABLE_ICB");
    unsetenv("MGL_ENABLE_ICB_BATCH");
    unsetenv("MGL_ENABLE_ICB_PIPELINES");
    unsetenv("MGL_DISABLE_ICB");
    unsetenv("MGL_DISABLE_ICB_BATCH");
}

static void test_default_off(void)
{
    clear_icb_env();
    MGLBatchIcbConfig cfg = mgl_batch_icb_config();
    expect(cfg.enable == 0u && cfg.disable == 0u, "default cfg clear");
    expect(!mgl_batch_icb_support_indirect_command_buffers(),
           "default support off");
}

static void test_unified_enable(void)
{
    clear_icb_env();
    setenv("MGL_ENABLE_ICB", "1", 1);
    expect(mgl_batch_icb_support_indirect_command_buffers(),
           "MGL_ENABLE_ICB enables");
    clear_icb_env();
    setenv("MGL_ENABLE_ICB", "true", 1);
    expect(mgl_batch_icb_support_indirect_command_buffers(),
           "MGL_ENABLE_ICB=true enables");
}

static void test_legacy_either_enable(void)
{
    clear_icb_env();
    setenv("MGL_ENABLE_ICB_BATCH", "1", 1);
    expect(mgl_batch_icb_support_indirect_command_buffers(),
           "legacy ENABLE_ICB_BATCH enables");
    clear_icb_env();
    setenv("MGL_ENABLE_ICB_PIPELINES", "1", 1);
    expect(mgl_batch_icb_support_indirect_command_buffers(),
           "legacy ENABLE_ICB_PIPELINES enables (same gate)");
    clear_icb_env();
    setenv("MGL_ENABLE_ICB_BATCH", "1", 1);
    setenv("MGL_ENABLE_ICB_PIPELINES", "1", 1);
    expect(mgl_batch_icb_support_indirect_command_buffers(),
           "both legacy enables still on");
}

static void test_disable_wins(void)
{
    clear_icb_env();
    setenv("MGL_ENABLE_ICB", "1", 1);
    setenv("MGL_DISABLE_ICB", "1", 1);
    expect(!mgl_batch_icb_support_indirect_command_buffers(),
           "DISABLE_ICB hard-off");
    clear_icb_env();
    setenv("MGL_ENABLE_ICB_PIPELINES", "1", 1);
    setenv("MGL_DISABLE_ICB_BATCH", "1", 1);
    expect(!mgl_batch_icb_support_indirect_command_buffers(),
           "DISABLE_ICB_BATCH hard-off over PIPELINES");
    clear_icb_env();
    setenv("MGL_ENABLE_ICB_BATCH", "1", 1);
    setenv("MGL_DISABLE_ICB", "1", 1);
    MGLBatchIcbConfig cfg = mgl_batch_icb_config();
    expect(cfg.enable == 1u && cfg.disable == 1u, "enable+disable both set");
    expect(!mgl_batch_icb_support_indirect_command_buffers(),
           "support requires enable && !disable");
}

static void test_falsey_enable(void)
{
    clear_icb_env();
    setenv("MGL_ENABLE_ICB", "0", 1);
    expect(!mgl_batch_icb_support_indirect_command_buffers(),
           "ENABLE_ICB=0 is off");
    setenv("MGL_ENABLE_ICB", "false", 1);
    expect(!mgl_batch_icb_support_indirect_command_buffers(),
           "ENABLE_ICB=false is off");
}

int main(void)
{
    test_default_off();
    test_unified_enable();
    test_legacy_either_enable();
    test_disable_wins();
    test_falsey_enable();
    clear_icb_env();
    if (g_fails) {
        fprintf(stderr, "test_batch_icb: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("test_batch_icb: ok\n");
    return 0;
}
