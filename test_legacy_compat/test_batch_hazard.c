/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_batch_hazard overflow policy (O2.2). No Metal required.
 */

#include "mgl_batch_hazard.h"

#include <stdio.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static void test_policy_select(void)
{
    expect(mgl_batch_hazard_overflow_policy(0) == MGL_HAZARD_OVERFLOW_STICKY,
           "default → sticky");
    expect(mgl_batch_hazard_overflow_policy(1) ==
               MGL_HAZARD_OVERFLOW_FLUSH_AND_CONTINUE,
           "flag → flush-and-continue");
}

static void test_sticky_capacity(void)
{
    expect(mgl_batch_hazard_on_capacity_full(MGL_HAZARD_OVERFLOW_STICKY, 1) ==
               MGL_HAZARD_TRACK_LATCH_OVERFLOW,
           "sticky+pending → latch");
    expect(mgl_batch_hazard_on_capacity_full(MGL_HAZARD_OVERFLOW_STICKY, 0) ==
               MGL_HAZARD_TRACK_LATCH_OVERFLOW,
           "sticky+empty → latch");
}

static void test_flush_and_continue(void)
{
    expect(mgl_batch_hazard_on_capacity_full(
               MGL_HAZARD_OVERFLOW_FLUSH_AND_CONTINUE, 1) ==
               MGL_HAZARD_TRACK_FLUSH_THEN_RETRY,
           "flush-continue+pending → flush then retry");
    expect(mgl_batch_hazard_on_capacity_full(
               MGL_HAZARD_OVERFLOW_FLUSH_AND_CONTINUE, 0) ==
               MGL_HAZARD_TRACK_LATCH_OVERFLOW,
           "flush-continue without pending degrades to latch");
}

static void test_query_degraded(void)
{
    expect(mgl_batch_hazard_query_degraded(0) == 0, "no latch → not degraded");
    expect(mgl_batch_hazard_query_degraded(1) == 1, "latch → degraded");
}

int main(void)
{
    test_policy_select();
    test_sticky_capacity();
    test_flush_and_continue();
    test_query_degraded();
    if (g_fails) {
        fprintf(stderr, "test_batch_hazard: %d FAIL(s)\n", g_fails);
        return 1;
    }
    printf("test_batch_hazard: PASS\n");
    return 0;
}
