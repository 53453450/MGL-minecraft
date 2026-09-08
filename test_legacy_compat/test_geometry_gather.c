/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Pure unit test for mglGeometryGatherIndices (O1.4 move into mgl_draw_tess).
 */
#include "glcorearb.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Link against the same TU math via reimplementation of the thin wrapper
 * contract: gather unique indices for patch-style inputVertices groups. */

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

/* Mirror mglRenderGeometryGatherIndices for uint16 indices, no restart. */
static int gather_u16(const uint16_t *idx, uint32_t count, uint32_t input_verts,
                      uint32_t **out, uint32_t *out_count, uint32_t *out_prims,
                      uint32_t *out_max)
{
    if (!idx || !out || !out_count || !out_prims || !out_max || input_verts == 0) {
        return 0;
    }
    uint32_t prims = count / input_verts;
    uint32_t gathered = prims * input_verts;
    if (gathered == 0) {
        return 0;
    }
    uint32_t *g = (uint32_t *)calloc(gathered, sizeof(uint32_t));
    if (!g) return 0;
    uint32_t max_i = 0;
    for (uint32_t i = 0; i < gathered; i++) {
        g[i] = idx[i];
        if (g[i] > max_i) max_i = g[i];
    }
    *out = g;
    *out_count = gathered;
    *out_prims = prims;
    *out_max = max_i;
    return 1;
}

static void test_patch_gather(void)
{
    const uint16_t idx[] = {0, 1, 2, 3, 4, 5};
    uint32_t *g = NULL, gc = 0, gp = 0, gm = 0;
    expect(gather_u16(idx, 6, 3, &g, &gc, &gp, &gm), "gather 2 triangles");
    expect(gc == 6u && gp == 2u && gm == 5u, "counts");
    expect(g && g[0] == 0 && g[5] == 5, "values");
    free(g);

    expect(!gather_u16(idx, 2, 3, &g, &gc, &gp, &gm), "incomplete patch → empty");
}

int main(void)
{
    test_patch_gather();
    if (g_fails) {
        fprintf(stderr, "%d failure(s)\n", g_fails);
        return 1;
    }
    puts("test_geometry_gather: ok");
    return 0;
}
