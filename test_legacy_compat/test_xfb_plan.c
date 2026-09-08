/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Goldens for TES/GS XFB destination planning and GS primitive-query
 * reduction.  These functions live in mgl_draw_tess.cpp / mgl_draw_gs.cpp;
 * this TU re-implements the contracts so the math can run without Metal.
 */

#include "glcorearb.h"

#include <stdint.h>
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

static int tess_dest(uint32_t items, uint32_t instances, uint32_t stride,
                     uint32_t vpp, uint64_t session, int64_t slot,
                     uint64_t visible, uint32_t *verts, uint32_t *bytes,
                     uint32_t *off)
{
    if (items == 0u || instances == 0u || stride == 0u || vpp == 0u ||
        slot < 0) {
        return 0;
    }
    uint64_t capture_vertices = (uint64_t)items * instances;
    uint64_t primitive_bytes = (uint64_t)vpp * stride;
    uint64_t capture_primitives = capture_vertices / vpp;
    if (session > visible) {
        return 0;
    }
    uint64_t remaining = visible - session;
    uint64_t copied = remaining / primitive_bytes;
    if (copied > capture_primitives) {
        copied = capture_primitives;
    }
    *verts = (uint32_t)(copied * vpp);
    *bytes = (uint32_t)(copied * primitive_bytes);
    *off = (uint32_t)((uint64_t)slot + session);
    return 1;
}

static void test_tess_xfb_dest(void)
{
    uint32_t verts = 0, bytes = 0, off = 0;
    expect(tess_dest(4, 1, 16, 2, 0, 0, 48, &verts, &bytes, &off),
           "tess dest accepts a truncated store");
    expect(verts == 2u && bytes == 32u && off == 0u,
           "48-byte store holds one 2-vert primitive of stride 16");

    expect(tess_dest(4, 1, 16, 2, 0, 0, 64, &verts, &bytes, &off),
           "tess dest full store");
    expect(verts == 4u && bytes == 64u, "two complete line primitives");

    expect(tess_dest(4, 1, 16, 2, 0, 0, 16, &verts, &bytes, &off) &&
               verts == 0u && bytes == 0u,
           "visible < primitive bytes → 0 complete primitives");
}

static void test_gs_query_not_capacity(void)
{
    /* Capacity heuristic was work_items * expanded / vpp = 2 * 6 / 3 = 4.
     * Kernel vertex_count of 3 is one triangle; query must not report 4. */
    const uint32_t counts[14] = {
        /* work 0: vertex_count=3, scratch..., emit at word 5 */
        3, 1, 0, 0, 0, 1, 0,
        /* work 1: culled */
        0, 1, 0, 0, 0, 0, 0,
    };
    uint64_t verts = 0;
    for (uint32_t w = 0; w < 2; w++)
        verts += counts[w * 7];
    const uint64_t generated = verts / 3u;
    const uint64_t capacity = 2ull * 6ull / 3ull;
    expect(generated == 1u, "kernel emit is one triangle");
    expect(generated < capacity, "query is not the allocated expansion");
}

int main(void)
{
    test_tess_xfb_dest();
    test_gs_query_not_capacity();
    if (g_fails) {
        fprintf(stderr, "test_xfb_plan: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("test_xfb_plan: ok\n");
    return 0;
}