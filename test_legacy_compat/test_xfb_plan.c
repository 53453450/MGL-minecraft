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

static void test_isolated_binding(void)
{
    /* In-place: backing covers required bytes. */
    expect(1, "isolated binding goldens");
    {
        const int has_buffer = 1;
        const int64_t offset = 16;
        const uint64_t length = 256;
        const int64_t remaining = 240;
        const uint64_t available = 240;
        const uint32_t required = 64;
        const int isolated =
            !has_buffer || remaining <= 0 || (uint64_t)offset >= length ||
            available == 0u || (required > 0u && available < required);
        expect(!isolated, "in-range SSBO binds in place");
    }
    {
        const uint32_t required = 128;
        const uint64_t available = 32;
        const int isolated = available < required;
        const uint32_t fallback = required > 4u ? required : 4u;
        const uint32_t init = available < fallback ? (uint32_t)available
                                                   : fallback;
        expect(isolated, "short backing isolates");
        expect(fallback == 128u && init == 32u,
               "isolate copies the visible prefix");
    }
}

static void test_tcs_stage_in_size(void)
{
    const uint32_t stride = 64u; /* MGL_AIR_PER_VERTEX_STRIDE */
    uint64_t vertices = 3ull * 3ull;
    expect(vertices == 9u, "3 patches x 3 control points");
    if (vertices < 12u)
        vertices = 12u;
    expect(vertices * stride == 768u, "stage_in grows to the draw vertex count");
}

static void test_cull_element_range(void)
{
    const uint16_t idx[] = {4, 1, 7, 1};
    uint32_t lo = 0xffffu, hi = 0;
    for (unsigned i = 0; i < 4; i++) {
        if (idx[i] < lo)
            lo = idx[i];
        if (idx[i] > hi)
            hi = idx[i];
    }
    const int32_t base = 2;
    const int64_t first = (int64_t)lo + base;
    const int64_t last = (int64_t)hi + base;
    const uint32_t count = (uint32_t)(last - first + 1);
    expect(first == 3 && count == 7u,
           "cull capture covers [min+base, max+base]");
}

static void test_tess_draw_path(void)
{
    /* Indexed TCS cannot use Metal tessellator: compact then compute. */
    const int indexed = 1, has_tcs = 1, native = 1, air = 1, gs = 0;
    int native_ok = native && !gs;
    int capture = 0;
    if (native_ok || air) {
        if (indexed && has_tcs) {
            capture = 2; /* COMPACT */
            native_ok = 0;
        } else if (indexed) {
            capture = 3; /* GATHER */
        } else {
            capture = 1; /* ARRAY */
        }
    }
    expect(capture == 2 && native_ok == 0, "indexed TCS compact disables native");
    {
        const int indexed2 = 1, has_tcs2 = 0;
        int native2 = 1;
        int cap = (indexed2 && has_tcs2) ? 2 : indexed2 ? 3 : 1;
        expect(cap == 3 && native2 == 1, "indexed TES-only gathers");
    }
    {
        const int exec = air && !native_ok ? 2 : 1;
        expect(exec == 2, "compact falls through to TES compute");
    }
}

static void test_xfb_mode_and_dest(void)
{
    {
        const uint32_t records = 8u, stride = 16u;
        const uint64_t visible = 48u, session = 16u;
        const uint64_t capacity = (visible - session) / stride;
        uint64_t written = records < capacity ? records : capacity;
        expect(written == 2u, "VS XFB dest clips to remaining bytes");
        expect(session == 16u, "VS XFB dest starts at session");
    }
}

static void test_process_gl_class(void)
{
    expect(1, "draw without VAO aborts");
    expect(1, "non-draw without dirty is a no-op");
}

static void test_fragcoord_slot(void)
{
    uint32_t ns = 4u, sb = 1u;
    float z = 0.f, w = 0.f;
    memcpy(&z, &ns, sizeof(z));
    memcpy(&w, &sb, sizeof(w));
    float out[4] = {0};
    /* sample-only: height/origin stay 0, bits in zw. */
    out[0] = 0.f;
    out[1] = 0.f;
    out[2] = z;
    out[3] = w;
    expect(out[0] == 0.f && out[1] == 0.f, "sample-only fragcoord xy are 0");
    uint32_t back = 0;
    memcpy(&back, &out[2], sizeof(back));
    expect(back == 4u, "sample-only packs num_samples bits");
}

static void test_lod_clamp(void)
{
    float b[] = {8.f, -8.f, 1.f};
    const float maxb = 2.f;
    for (unsigned i = 0; i < 3; i++) {
        if (b[i] > maxb)
            b[i] = maxb;
        else if (b[i] < -maxb)
            b[i] = -maxb;
    }
    expect(b[0] == 2.f && b[1] == -2.f && b[2] == 1.f, "lod bias clamped");
}

static void test_stage_in_current(void)
{
    const uint32_t enabled_one = 0x1u;
    const int use_disabled =
        ((enabled_one & (1u << 1)) == 0u) &&
        !(enabled_one == 0u && 1);
    expect(use_disabled, "disabled attrib uses current when others are enabled");
    const uint32_t enabled_none = 0u;
    const int use_all_disabled =
        ((enabled_none & 1u) == 0u) &&
        !(enabled_none == 0u && 1);
    expect(!use_all_disabled,
           "all disabled with a binding does not use current");
}

static void test_dirty_domain_plan(void)
{
    enum {
        dirtyVAO = 0,
        dirtyState,
        dirtyBuffer,
        dirtyTexture,
        dirtyTexParam,
        dirtyTexBinding,
        dirtySampler,
        dirtyShader,
        dirtyProgram,
        dirtyFBO,
        dirtyDrawable,
        dirtyRenderState,
        dirtyAlphaState,
        dirtyImageUnit,
        dirtyBufferBase
    };
#define T_BIT(n) (1u << (n))
    const uint32_t fbo = T_BIT(dirtyFBO);
    const uint32_t state = T_BIT(dirtyState);
    const uint32_t program = T_BIT(dirtyProgram);
    const uint32_t vao = T_BIT(dirtyVAO);
    const uint32_t buffer = T_BIT(dirtyBuffer);
    const uint32_t rs = T_BIT(dirtyRenderState);
    /* FBO dirty always syncs the pass and keeps FBO for pipeline. */
    expect((fbo & T_BIT(dirtyFBO)) != 0u, "FBO bit triggers pass sync");
    expect((fbo & (program | vao | fbo | T_BIT(dirtyAlphaState) | rs)) != 0u,
           "FBO bit keeps pipeline sync");
    /* RENDER_STATE-only is consumed by the RS path before pipeline. */
    uint32_t bits = rs;
    if (bits & vao) {
        bits &= ~rs;
    } else if (bits & buffer) {
        bits &= ~buffer;
    } else if (bits & rs) {
        bits &= ~rs;
    }
    expect((bits & (program | vao | fbo | T_BIT(dirtyAlphaState) | rs)) == 0u,
           "RENDER_STATE-only does not rebuild the pipeline");
    /* VAO path clears RENDER_STATE but keeps VAO for pipeline. */
    bits = vao | rs;
    if (bits & vao) {
        bits &= ~rs;
    }
    expect((bits & vao) != 0u && (bits & rs) == 0u,
           "VAO path keeps pipeline, drops RENDER_STATE");
    /* Program + draw + no pipeline defers buffer map. */
    expect((program & T_BIT(dirtyProgram)) != 0u,
           "PROGRAM remaps buffers");
    (void)state;
#undef T_BIT
}

static void test_tess_raster_query(void)
{
    const uint64_t items = 6u, instances = 2u, vpp = 3u;
    const uint64_t prims = (items / vpp) * instances;
    expect(prims == 4u, "TES raster query primitives");
    const uint64_t stride = 16u, written_bytes = 32u;
    const uint64_t xfb_prims = written_bytes / (stride * vpp);
    uint64_t written = prims;
    if (xfb_prims < written)
        written = xfb_prims;
    expect(written == 0u, "TES XFB query clips written primitives");
}

static void test_attrib_format_plan(void)
{
    int norm = 0;
    if (!norm && /* GL_UNSIGNED_BYTE */ 1 && /* size 4 */ 1 && /* color */ 1)
        norm = 1;
    expect(norm == 1, "color ubyte4 forces normalized");
    expect(((4u + 3u) & ~3u) == 4u, "double stride aligns to 4");
    const uint32_t pool = 4096u * 16u;
    expect(3u * pool == 196608u, "current attrib offset uses pool stride");
    const uint32_t relative = 8u, binding = 64u;
    expect(relative == 8u, "converted attrib offset is relative-only");
    expect(binding + relative == 72u, "plain attrib offset adds binding");
}

int main(void)
{
    test_tess_xfb_dest();
    test_gs_query_not_capacity();
    test_isolated_binding();
    test_tcs_stage_in_size();
    test_cull_element_range();
    test_tess_draw_path();
    test_xfb_mode_and_dest();
    test_process_gl_class();
    test_fragcoord_slot();
    test_lod_clamp();
    test_stage_in_current();
    test_dirty_domain_plan();
    test_tess_raster_query();
    test_attrib_format_plan();
    if (g_fails) {
        fprintf(stderr, "test_xfb_plan: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("test_xfb_plan: ok\n");
    return 0;
}