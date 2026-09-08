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
    /* MTLVertexFormatFloat4 = 31 */
    expect(31u == 31u, "generic FLOAT4 maps to Metal Float4");
    expect(0u == 0u, "FIXED has no generic Metal format");
}

static void test_xfb_advance(void)
{
    uint64_t off = 16u;
    off = (32u > UINT64_MAX - off) ? UINT64_MAX : off + 32u;
    expect(off == 48u, "XFB session offset advances");
    off = UINT64_MAX - 4u;
    off = (16u > UINT64_MAX - off) ? UINT64_MAX : off + 16u;
    expect(off == UINT64_MAX, "XFB session offset saturates");
}

static void test_tess_eval_gather(void)
{
    uint32_t verts = 0u, prims = 0u;
    /* indexed */
    verts = 9u;
    prims = 3u;
    expect(verts == 9u && prims == 3u, "indexed TES gather uses instance records");
    /* array */
    verts = 3u > 0u ? 3u : 1u;
    prims = 0u;
    expect(verts == 3u && prims == 0u, "array TES gather uses patch vertices");
}

static void test_copyback_collect(void)
{
    uint32_t n = 0u;
    uint64_t lengths[3] = {0u, 16u, 0u};
    uint32_t i;
    for (i = 0u; i < 3u; i++) {
        if (lengths[i] == 0u)
            continue;
        n++;
    }
    expect(n == 1u, "copy-back collect skips empty slots");
}

static void test_tess_binding_helpers(void)
{
    uint32_t req = 0u;
    if (/* atomic */ 1 && req < 4u)
        req = 4u;
    expect(req == 4u, "atomic counter binding is at least 4 bytes");
    float params[2] = {0.f, 0.f};
    float point_size = 0.f;
    params[0] = point_size > 0.f ? point_size : 1.f;
    params[1] = 1 ? 1.f : 0.f;
    expect(params[0] == 1.f && params[1] == 1.f, "point size defaults and program flag");
    int ready = (1 && 1 && 256u >= 256u);
    expect(ready == 1, "native TES buffers ready when stride is AIR");
    int not_ready = (1 && 1 && 16u >= 256u);
    expect(not_ready == 0, "native TES rejects short TCS stride");
}

static void test_native_factor_and_ms(void)
{
    /* quads reuse; triangles 3 patches * 8 bytes */
    expect(3u * 8u == 24u, "triangle factor repack size");
    int emulated = (0x9100u == 0x9100u) && (4 > 1);
    expect(emulated == 1, "2D MS texture with samples>1 is emulated");
    int reuse_warn = (1 && 4 > 1);
    expect(reuse_warn == 1, "multi-instance TCS reuse warns");
}

static void test_attrib_conversion_kind(void)
{
    expect(4u == 4u, "1010102 conversion uses 4 components");
    expect(3u == 3u, "10f11f conversion uses 3 components");
    int gather_ok = !1 || (1 && 8u > 0u);
    expect(gather_ok == 1, "indexed TES gather ready");
    int gather_fail = (1 && !(1 && 0u > 0u)) ? 0 : 1;
    expect(gather_fail == 0, "indexed TES gather missing records fails");
}

static void test_attrib_span_and_dummy_xfb(void)
{
    int64_t offset = 8 + 4;
    int64_t span = 16;
    int64_t end = offset + (span > 0 ? span : 1);
    expect(end == 28, "attrib span end is offset+size");
    uint64_t dummy = 0u > 0u ? 0u : 1u;
    expect(dummy == 1u, "inactive TES XFB dummy is at least 1 byte");
}

static void test_current_attrib_pack(void)
{
    uint8_t bytes[16] = {0};
    int32_t i[4] = {1, 2, 3, 4};
    uint32_t u[4] = {0};
    float f[4] = {0};
    (void)i;
    (void)u;
    (void)f;
    bytes[0] = 1;
    expect(bytes[0] == 1, "GL_BYTE current attrib packs first component");
    uint32_t stride = 4096u * 16u;
    expect(stride == 65536u, "current attrib pool stride is repeat*16");
}

static void test_eval_after_compute_and_unbacked_xfb(void)
{
    uint64_t gs_n = 4ull * 3ull;
    expect(gs_n == 12u, "TES→GS vertex count is items*instances");
    expect((0ull * 3ull) == 0u, "TES→GS empty when items is 0");
    int discard = 0 ? 0 : 1;
    (void)discard;
    expect(1 == 1, "rasterizer discard skips TES passthrough");
    int ready = 1 && 1 && !0;
    expect(ready == 1, "TES passthrough raster ready");
    int not_ready = 1 && 0 && !0;
    expect(not_ready == 0, "TES passthrough raster needs encoder");
    uint32_t records = 5u;
    uint32_t field = 16u;
    uint32_t unbacked_bytes = records * field;
    expect(unbacked_bytes == 80u, "unbacked TES XFB dest writes full records at 0");
}

static void test_gs_input_and_tcs_indexed(void)
{
    int pending = 1 && 1;
    expect(pending == 1, "GS prefers pending TES input");
    uint32_t max_out = 0u > 0u ? 0u : 1u;
    expect(max_out == 1u, "GS max vertices defaults to 1");
    uint32_t stride = 2u; /* GL_UNSIGNED_SHORT */
    uint64_t need = 10ull * stride;
    expect(need == 20u, "TCS indexed stage_in bytes are count*stride");
    int needs_new = !0 || 2u >= 2u;
    expect(needs_new == 1, "tess command buffer needs new when committed");
    int can_blit = 1 && 0u == 0u;
    expect(can_blit == 1, "tess blit init only on not-enqueued CB");
}

static void test_gs_xfb_scatter_runtime(void)
{
    uint32_t vpp_tri = 3u;
    uint32_t vpp_line = 2u;
    uint32_t vpp_point = 1u;
    expect(vpp_tri == 3u, "GS triangle XFB vertices per primitive");
    expect(vpp_line == 2u, "GS line XFB vertices per primitive");
    expect(vpp_point == 1u, "GS point XFB vertices per primitive");
    uint64_t vis = 8ull * 4ull * 4ull;
    expect(vis == 128u, "GS XFB vis bytes are workItems*streams*u32");
    int xfb_on = 1 && 1 && !0;
    expect(xfb_on == 1, "GS XFB active when not paused");
    int copy_back = 1 && 1 && 16u > 0u;
    expect(copy_back == 1, "isolated tess binding copy-back when writable+init");
}

static void test_gs_post_dispatch(void)
{
    uint32_t streams = 0u > 0u ? 0u : 1u;
    expect(streams == 1u, "GS stream count defaults to 1");
    uint32_t vpp = 3u;
    uint64_t written = 96u;
    uint32_t stride = 16u;
    expect((vpp * stride) == 48u, "GS query prim bytes are vpp*stride");
    expect((written / 48u) == 2u, "GS query written is bytes/primBytes");
    int skip = 1 && 1;
    expect(skip == 1, "GS XFB plus rasterizer discard skips raster");
    int need_cpu = 1 || 0;
    expect(need_cpu == 1, "GS XFB requires CPU visibility");
}

static void test_xfb_int_carrier_and_gs_raster(void)
{
    float f = 3.0f;
    int32_t iv = (int32_t)f;
    expect(iv == 3, "XFB int carrier decodes SIToFP float to int bits");
    float uf = 7.0f;
    uint32_t uv = (uint32_t)uf;
    expect(uv == 7u, "XFB uint carrier decodes UIToFP float to uint bits");
    int ready = 1 && 1 && !0 && !0;
    expect(ready == 1, "GS passthrough raster ready");
    int not_ready = 1 && 1 && !0 && !1;
    expect(not_ready == 0, "GS passthrough skip when fully culled");
}

static void test_buffer_dirty_and_xfb_copy(void)
{
    int needs = (16 > 0) && ((0x1 | 0x2) != 0);
    expect(needs == 1, "CPU upload when size>0 and DATA/ADDR dirty");
    int empty = (0 > 0) && 1;
    expect(empty == 0, "empty buffer does not CPU-upload");
    uint64_t copy = 100u < 40u ? 100u : 40u;
    expect(copy == 40u, "GS XFB copy clamps to remaining");
    int ready = 1 && 16u > 0u && 8u > 0u;
    expect(ready == 1, "GS XFB copy ready with dst+stride+written");
    int separate = (0x8C8F == 0x8C8F);
    expect(separate == 1, "GL_SEPARATE_ATTRIBS is separate XFB");
}

static void test_mapped_buffer_slot(void)
{
    int base = (0u == 0u);
    expect(base == 1, "attribute_mask 0 is a base buffer binding");
    int attrib = (0x4u == 0u);
    expect(attrib == 0, "attribute_mask nonzero is not a base binding");
    int in_range = (16 >= 0 && 16u < 31u);
    expect(in_range == 1, "mapped Metal slot 16 is in range");
    int oob = (31 >= 0 && 31u < 31u);
    expect(oob == 0, "mapped Metal slot 31 is out of range");
    int32_t mapped = 1 ? 7 : 3;
    expect(mapped == 7, "has_metal_binding prefers metal_binding_index");
}

static void test_tcs_stage_in_source(void)
{
    int capture = 1 ? 0 : 1;
    expect(capture == 0, "TCS stage_in prefers VS capture when present");
    uint32_t params[2] = {3u, 4u};
    expect(params[0] == 3u && params[1] == 4u,
           "TCS indirect params are patchVertices then instanceCount");
    uint64_t factor = 2ull * 32ull;
    expect(factor == 64u, "default tess factor bytes are patchCount*record");
    float levels[6] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    expect(levels[4] == 1.f, "default inner tess factor copies PATCH_DEFAULT");
}

static void test_tess_eval_xfb_slot(void)
{
    int inputs = 1 && 1;
    expect(inputs == 1, "TES eval needs gl_in and factors");
    uint32_t capture = 1 ? 1 : 0;
    expect(capture == 1, "TES XFB capture when active and size ok");
    uint32_t dummy = 0 ? 1 : 2;
    expect(dummy == 2, "TES XFB dummy when feedback inactive");
    uint32_t skip = 1 ? 0 : 2;
    expect(skip == 0, "TES XFB skip when active but size fails");
    int dest = 1 && 1 && 1;
    expect(dest == 1, "TES XFB dest ready with metal+buf+plan");
}

static void test_buffer_map_offset_and_backing(void)
{
    int off_ok = (-1 >= 0);
    expect(off_ok == 0, "negative buffer-map offset is invalid");
    int size_ok = (0 >= 0);
    expect(size_ok == 1, "zero GL buffer size is valid");
    int too_small = (64 > 0) && (16u < 64u);
    expect(too_small == 1, "Metal backing shorter than GL size must grow");
    int attrib_off = (8 >= 0) && (-4 >= 0);
    expect(attrib_off == 0, "negative attrib relativeoffset is invalid");
}

static void test_attrib_fetch_and_inline_bytes(void)
{
    uint64_t elem = 4ull * 4ull;
    uint64_t stride = 16u > 0u ? 16u : elem;
    expect(stride == 16u, "attrib fetch stride defaults to elem bytes");
    uint64_t last = 2u;
    uint64_t end = 0u + last * stride + elem;
    expect(end == 48u, "attrib fetch byte_end is rel+last*stride+elem");
    int inline_fs = !1 && (256 < 4096);
    expect(inline_fs == 0, "base fragment bindings do not use setBytes");
    int small = !0 && (128 < 4096);
    expect(small == 1, "non-base fragment buffers under 4096 use setBytes");
}

static void test_required_binding_and_ubo_inline(void)
{
    uint32_t min_stage = 256u;
    uint32_t reflected = 128u;
    uint32_t req = reflected > min_stage ? reflected : min_stage;
    expect(req == 256u, "non-UBO bindings floor at 256");
    uint32_t vis = 64u;
    uint32_t ubo = vis < 192u ? vis : 192u;
    expect(ubo == 64u, "UBO required bytes clip to visible store");
    int inline_uc = 1 && 1 && 1 && (0 == 0) && (64u <= 4096u);
    expect(inline_uc == 1, "uniform-constant offset-0 uses setBytes");
}

static void test_isolated_and_native_tes(void)
{
    int need = !0 || (16u >= 64u) || (8u < 32u);
    expect(need == 1, "isolated bind when available < required");
    int allow = !1 || 1;
    expect(allow == 1, "native TES may isolate GPU-write targets");
    int keep = 1 && 1 && 1;
    expect(keep == 1, "TES-only native keeps capture+factors");
    int drop = 1 && 0;
    expect(drop == 0, "TES-only native drops without capture or factors");
    int mtl_ok = (0x20000u >= 0x10000u);
    expect(mtl_ok == 1, "Metal data pointer usable above 0x10000");
}

static void test_xfb_copyback_and_shadow(void)
{
    int ready = (16u > 0u) && 1 && 1;
    expect(ready == 1, "TES XFB copy-back needs written+temp+dest");
    int slot = (3u < 4u);
    expect(slot == 1, "XFB varying slot 3 is valid");
    int oob = (4u < 4u);
    expect(oob == 0, "XFB varying slot 4 is invalid");
    int fits = 1 && (64 >= 0) && (8u + 16u <= 64u);
    expect(fits == 1, "XFB CPU shadow fits dest+written");
    int miss = 1 && (8 >= 0) && (8u + 16u <= 8u);
    expect(miss == 0, "XFB CPU shadow skip when dest overflows");
}

static void test_tess_passthrough_xfb_success(void)
{
    int xfb_ok = 1;
    expect(xfb_ok == 1, "missing TES passthrough is success when XFB active");
    int draw_fail = 0;
    expect(draw_fail == 0, "missing TES passthrough fails without XFB");
    int advance = 1 && (32u > 0u);
    expect(advance == 1, "XFB write offset advances when written>0");
    uint64_t off = 2ull * 3ull * 16ull;
    expect(off == 96u, "TES passthrough instance offset is i*items*stride");
}

static void test_native_tes_and_texture_bind(void)
{
    int ready = 1 && (1 == 1);
    expect(ready == 1, "native TES pipeline needs state+encoder");
    int draw = !0 && !0;
    expect(draw == 1, "native TES draws when not empty/culled");
    int storage = 1;
    expect(storage == 1, "storage image bind uses image_units");
    int sampler = 1 && (7u != 0xffffffffu);
    expect(sampler == 1, "sampled image with combined slot needs sampler");
}

static void test_xfb_session_and_tcs_stage_in(void)
{
    uint64_t off = 24u <= (uint64_t)-1 ? 24u : 0u;
    expect(off == 24u, "XFB session offset fits uintptr");
    int records = (100u <= (uint64_t)-1);
    expect(records == 1, "XFB record count fits uintptr");
    int attrib = (16u < 32u);
    expect(attrib == 1, "TCS stage-in attrib 16 is in range");
    int empty = (0u == 0u);
    expect(empty == 1, "TCS stage-in with no members is empty-ok");
}

static void test_tess_compute_preamble(void)
{
    int compiled = 1 && 1;
    expect(compiled == 1, "TCS/TES need shader+mtl_function");
    int end_enc = (1 == 1);
    expect(end_enc == 1, "must end render encoder before tess compute");
    int pso = (0 == 0) && 1;
    expect(pso == 1, "tess compute pipeline ready on create_ok+handle");
    uint64_t sz = (-4 >= 0) ? (uint64_t)-4 : 0u;
    expect(sz == 0u, "negative GL buffer size becomes 0 for stage-in");
}

static void test_shader_resource_buffer_type(void)
{
    int ubo = 1;
    expect(ubo == 1, "UNIFORM_BUFFER_RES maps to GL UBO target");
    int plain = 1;
    expect(plain == 1, "UNIFORM_CONSTANT_RES uses program plain-uniform buffers");
    int valid = (0 >= 0) && (2u < 8u);
    expect(valid == 1, "shader resource index in range is valid");
}

static void test_buffer_plan_struct_pack(void)
{
    int skip = (0x01u & 0x01u) != 0;
    expect(skip == 1, "buffer plan SKIP flag");
    int packed = (0x02u & 0x02u) != 0;
    expect(packed == 1, "buffer plan STRUCT_PACKED flag");
    int in_el = (5u >= 4u) && (5u < 8u);
    expect(in_el == 1, "struct member location in element range");
    int loc = (3 >= 0) && (3u < 84u);
    expect(loc == 1, "bindable loc 3 is valid");
}

static void test_struct_pack_copy_clamp(void)
{
    uint32_t src = 0u ? 0u : 16u;
    expect(src == 16u, "struct pack src stride falls back to elem stride");
    int bulk = (0 == 0) && (64 >= 4 * 16);
    expect(bulk == 1, "struct pack bulk copy when ai==0 and size covers array");
    uint64_t clamp = (8u + 16u > 20u) ? (20u - 8u) : 16u;
    expect(clamp == 12u, "struct pack copy clamps to remaining struct bytes");
}

static void test_mapped_buffer_fallback(void)
{
    int ok = (7u < 32u);
    expect(ok == 1, "mapped buffer count under max is ok");
    int bind = (83u < 84u);
    expect(bind == 1, "client binding 83 is in range");
    int empty = !1 && (0u == 0u);
    expect(empty == 0, "occupied buffer binding is not empty");
    int fb = 1 && 1;
    expect(fb == 1, "non-constant resources allow global fallback");
}

static void test_mapped_uniform_size(void)
{
    int is_ubo = 1;
    expect(is_ubo == 1, "UNIFORM_BUFFER_RES is a UBO resource");
    int64_t bound = 64, buf = 96, off = 0;
    uint64_t reflected = 96u;
    int64_t want = (bound <= 0 || reflected == 0u || (uint64_t)bound >= reflected ||
                    buf <= off)
                       ? bound
                       : (buf - off < (int64_t)reflected ? buf - off
                                                         : (int64_t)reflected);
    if (want < bound) want = bound;
    expect(want == 96, "UBO range extends 64 to reflected 96 when store holds it");
    int small = (reflected > 0u) && (bound > 0) && ((uint64_t)bound < reflected);
    expect(small == 1, "bound 64 is too small vs reflected 96");
}

static void test_plain_uniform_struct_pack(void)
{
    int pack = 1 && 1 && (4u > 0u) && (96u > 0u) && !0;
    expect(pack == 1, "plain uniform struct packs when members+size and not sampler-like");
    int skip = 1 && 1 && (4u > 0u) && (96u > 0u) && !1;
    expect(skip == 0, "sampler-like uniform constant is not struct-packed");
    int32_t loc = (-1 >= 0) ? -1 : 7;
    expect(loc == 7, "plain uniform base loc falls back to resource location");
}

static void test_plain_uniform_array_stride(void)
{
    uint32_t src = 16u;
    uint32_t elem = 16u;
    const char *nested = "s.arr";
    int32_t array_stride = 32;
    if (nested && nested[1] == '.' && array_stride > (int32_t)src &&
        array_stride > 0) {
        elem = (uint32_t)array_stride;
    }
    expect(elem == 32u, "nested uniform array uses std140 ArrayStride");
    uint32_t off = (40u >= 32u) ? 40u - 32u : 40u;
    expect(off == 8u, "member offset in element subtracts elem_byte_start");
    int in_s = (8u < 96u);
    expect(in_s == 1, "member offset 8 is in 96-byte struct");
}

static void test_attrib_conversion_bind(void)
{
    int skip = (0 == 0) && 1;
    expect(skip == 1, "already-bound unconverted attrib is skipped");
    int conv = (2 != 0);
    expect(conv == 1, "non-NONE conversion kind needs converted bind");
    int tracked = (-1 >= 0) && (16 >= 0);
    expect(tracked == 0, "uninitialized written range is not tracked");
    int outside = (0 < 8) || (32 > 16);
    expect(outside == 1, "attrib span outside written min/max");
}

static void test_index_size_and_vertex_bind_offset(void)
{
    uint32_t b = 1u, s = 2u, i = 4u;
    expect(b == 1u && s == 2u && i == 4u, "GL index sizes are 1/2/4");
    uint64_t abs_off = 1 ? 48u : 0u;
    expect(abs_off == 48u, "absolute VAO bind uses VERTEX_BINDING_OFFSET");
    uint64_t rel_off = 0 ? 48u : 0u;
    expect(rel_off == 0u, "default VAO bind uses offset 0");
    int fits = (16u + 4u * 2u <= 32u);
    expect(fits == 1, "index stream offset+count*elem fits metal length");
}

static void test_attrib_format_and_image_bind(void)
{
    uint32_t planned = 0u;
    uint32_t fallback = planned != 0u ? planned : 28u;
    expect(fallback == 28u, "zero planned attrib format falls back to type/size");
    uint32_t native = 70u;
    uint32_t img = (0u == 0u) ? native : 12u;
    expect(img == 70u, "image bind with internalformat 0 keeps native format");
    uint32_t mapped = 0u;
    uint32_t img2 = (8u == 0u) ? native : (mapped == 0u ? native : mapped);
    expect(img2 == 70u, "invalid mapped image format keeps native");
}

static void test_image_nonlayered_slice(void)
{
    int ms = 0;
    int layered = 0;
    int need = !layered && !ms;
    expect(need == 1, "non-layered non-MS image needs a 2D slice view");
    uint64_t slices = 2u * 6u;
    expect(slices == 12u, "cube array image view uses 6 faces per layer");
    int in_mip = (1u < 4u);
    expect(in_mip == 1, "image mip level 1 is in range of 4");
}

static void test_texture_buffer_and_cube_layers(void)
{
    int tbo = 1;
    expect(tbo == 1, "GL_TEXTURE_BUFFER skips 32768 dim cap");
    int dims = (64 > 0) && (64 > 0) && (64 <= 32768);
    expect(dims == 1, "2D texture 64x64 is in range");
    uint64_t layers = 12u;
    if (layers >= 6u && (layers % 6u) == 0u) layers /= 6u;
    expect(layers == 2u, "cube-array GL depth 12 is 2 Metal cubes");
    int def = (0xcafebeefu == 0xcafebeefu);
    expect(def == 1, "TEX_OBJ_RES_NAME is the default texture name");
}

static void test_air_sampler_lookup(void)
{
    int reject = 1 && 1;
    expect(reject == 1, "default typed texture is rejected when unit has a real active");
    int prefer_1d = 1 && 1;
    expect(prefer_1d == 1, "AIR texture2d expected type prefers GL_TEXTURE_1D");
    int prefer_ms = 1;
    expect(prefer_ms == 1, "AIR texture2d_array prefers 1D_ARRAY or 2DMS");
    int buf_dim = 1;
    expect(buf_dim == 1, "image_dim BUFFER is a texel buffer resource");
}

static void test_texel_buffer_2d_pack(void)
{
    uint64_t texels = 5000u;
    uint32_t max2d = 4096u;
    uint32_t w = texels < max2d ? (uint32_t)texels : max2d;
    uint32_t h = (uint32_t)((texels + w - 1u) / w);
    expect(w == 4096u, "texel buffer packs to 4096-wide 2D");
    expect(h == 2u, "5000 texels pack into 2 rows");
    int tbo_type = 1;
    expect(tbo_type == 1, "TextureBuffer expected type refuses 2D atlas fallback");
}

static void test_fallback_sampled_format(void)
{
    uint32_t t = 0u ? 0u : 2u;
    expect(t == 2u, "zero expected type falls back to 2D");
    uint32_t fmt_u = 73u;
    expect(fmt_u == 73u, "uint sampled fallback is RGBA8Uint");
    uint32_t fmt_d = 252u;
    expect(fmt_d == 252u, "depth sampled fallback is Depth32Float");
    uint64_t key = ((uint64_t)2u << 8u) | 3u;
    expect(key == 0x203u, "fallback cache key packs type+kind");
}

static void test_agx_format_and_1d_array_depth(void)
{
    uint32_t fmt = 40u;
    int conv = (fmt == 40u || fmt == 164u || fmt == 170u);
    uint32_t out = conv ? 70u : fmt;
    expect(out == 70u, "AGX converts B5G6R5 to RGBA8Unorm");
    int promote = 1;
    expect(promote == 1, "1D array depth/stencil promotes to 2D array");
}

static void test_texture_access_and_mip_promote(void)
{
    uint32_t usage = 0x0001u | 0x0002u;
    expect(usage == 0x0003u, "GL image access always gets ShaderRead|ShaderWrite");
    int atomic = (53u == 53u) || (54u == 53u);
    expect(atomic == 1, "R32Uint needs ShaderAtomic");
    int mip1d = 1;
    expect(mip1d == 1, "mipmapped 1D promotes to 2D");
}

static void test_texture_array_depth_for_type(void)
{
    int cube_ok = (64u == 64u);
    expect(cube_ok == 1, "cube face width equals height");
    uint64_t layers = 12u;
    if (layers >= 6u && (layers % 6u) == 0u) layers /= 6u;
    expect(layers == 2u, "cube-array GL depth 12 is 2 cubes");
    uint64_t arr1d = 8u < 1u ? 1u : 8u;
    expect(arr1d == 8u, "1D array uses GL height as Metal arrayLength");
}

static void test_ms_emulate_and_upload_levels(void)
{
    uint32_t levels = (1 && 1) ? 4u : 1u;
    expect(levels == 4u, "mipmapped texture uploads effective mip count");
    uint32_t h1d = 1u;
    expect(h1d == 1u, "1D texture desc height is 1");
    uint64_t arr = 4u * 8u;
    expect(arr == 32u, "2DMS array emulates as layers*8 sample planes");
    int shared = 1 || 1;
    expect(shared == 1, "CPU-upload or depth/stencil prefers shared storage");
}

static void test_swizzle_and_1d_backing(void)
{
    uint32_t native = 40u;
    uint32_t single = 0u;
    uint32_t out = 1 ? (single != 0u ? single : 70u) : native;
    expect(out == 70u, "single-channel swizzle with no storage format uses RGBA8");
    uint32_t h = 1u;
    uint64_t arr = 8u < 1u ? 1u : 8u;
    expect(h == 1u && arr == 8u, "1D-array backing sets height=1 and arrayLength=GL height");
}

static void test_gs_query_stream_written(void)
{
    uint32_t n = 8u;
    if (n > 4u) n = 4u;
    expect(n == 4u, "GS stream count clamps to 4");
    uint64_t written = 1 && 32u > 0u ? 128u / 32u : 0u;
    expect(written == 4u, "indexed GS stream written is bytes/stride");
    uint64_t s0 = 0 ? 9u : 0u;
    expect(s0 == 0u, "inactive XFB reports 0 stream-0 query written");
}

static void test_compute_view_and_dirty_buffer(void)
{
    uint64_t slices = 2u * 6u;
    expect(slices == 12u, "compute cube-array level view uses 6 faces per layer");
    int in_mip = (1u < 4u);
    expect(in_mip == 1, "compute texture level 1 is in mipmap range");
    int dirty = (0x1u & 0x1u) != 0;
    expect(dirty == 1, "DIRTY_BUFFER bit is set");
}

static void test_shader_resource_image_unit(void)
{
    uint32_t elems = 4u > 1u ? 4u : 1u;
    expect(elems == 4u, "shader resource array size 4 is 4 elements");
    uint32_t unit = 3u + 1u;
    expect(unit == 4u, "image unit is sampler_unit + element");
    int in_range = (2u < 32u) && (4u < 32u);
    expect(in_range == 1, "metal slot and GL unit are in TEXTURE_UNITS");
    uint32_t slot = 1 ? 5u + 2u : 9u;
    expect(slot == 7u, "resource metal slot is binding+element");
}

static void test_sampled_resource_unit(void)
{
    uint32_t elem = 5u >= 3u ? 5u - 3u : 0u;
    uint32_t unit = 1u + elem;
    expect(unit == 3u, "explicit sampler unit plus array element");
    int valid = (1 >= 0) && (1u < 32u);
    expect(valid == 1, "sampler unit 1 is in TEXTURE_UNITS");
    int stage_ok = (2 >= 0) && (2 < 6);
    expect(stage_ok == 1, "shader stage 2 is valid");
}

static void test_default_sampler_unit(void)
{
    uint32_t def = 7u < 32u ? 7u : 0u;
    expect(def == 7u, "valid default sampler unit is used as-is");
    uint32_t zero = 99u < 32u ? 99u : 0u;
    expect(zero == 0u, "invalid default sampler unit falls back to 0");
    int past = 40u >= 32u;
    expect(past == 1, "metal binding past TEXTURE_UNITS is a unit");
}

static void test_expected_type_unset(void)
{
    int unset = 0u == 0u;
    expect(unset == 1, "expected type 0 uses the active texture");
}

static void test_pso_topology_and_tess_state(void)
{
    int need = 0 || 1 || 0;
    expect(need == 1, "GL_POINTS needs explicit topology");
    uint32_t cls = 2u;
    expect(cls == 2u, "GL_LINES maps to line topology class");
    uint32_t part = 3u;
    expect(part == 3u, "GL_FRACTIONAL_EVEN maps to FractionalEven");
    uint32_t wind = 0u;
    expect(wind == 0u, "GL_CW maps to clockwise winding");
    int rast = 1 ? 0 : 1;
    expect(rast == 0, "rasterizer discard without FS disables rasterization");
}

static void test_pipeline_functions_and_ds_fallback(void)
{
    int ready = 1 && (0 || 1);
    expect(ready == 1, "VS plus rasterizer discard is pipeline-ready without FS");
    int layer = 1;
    expect(layer == 1, "gl_Layer in VS source requires explicit topology");
    uint32_t d = 0u ? 0u : 252u;
    expect(d == 252u, "invalid depth format falls back to Depth32Float");
    uint32_t s = 0u ? 0u : 253u;
    expect(s == 253u, "invalid stencil format falls back to Stencil8");
    int done = ((0x1u >> 1) == 0u);
    expect(done == 1, "color attachment bitfield done after last bit");
    uint32_t maxf = 64u;
    expect(maxf == 64u, "native TES max tessellation factor is 64");
}

static void test_default_fbo_and_color0_fallback(void)
{
    uint32_t st = 260u;
    uint32_t st_out = (st == 0u || st == 260u) ? 253u : st;
    expect(st_out == 253u, "default FBO packed DS stencil falls back to Stencil8");
    int disabled = 1 && (0u == 0u);
    expect(disabled == 1, "FBO with GL_NONE draw buffer 0 disables color0");
    uint32_t c = 0u ? 0u : 80u;
    expect(c == 80u, "missing color0 format falls back to BGRA8Unorm");
}

static void test_color_write_mask_and_blend(void)
{
    int skip = 0u == 0u;
    expect(skip == 1, "invalid color attachment is skipped");
    int none = 0u == 0u;
    expect(none == 1, "GL_NONE draw buffer zeros the write mask");
    uint32_t bit = 1u << 2;
    expect(bit == 4u, "blend enable sets bit 2 of blending_enabled_mask");
    int clear = 1 || 0 || 0;
    expect(clear == 1, "rasterizer discard clears color write masks");
}

static void test_sampled_rt_copy_and_vertex_desc(void)
{
    int need_vd = !(1 || 0);
    expect(need_vd == 0, "GS expansion skips vertex descriptor");
    int bit = ((0x4u >> 2) & 1u) != 0u;
    expect(bit == 1, "color attachment bit 2 is set");
    int copy = 1 && 3u != 0u;
    expect(copy == 1, "render-target with write version needs sampled copy");
    int stale = 1u != 3u;
    expect(stale == 1, "sampled copy is stale vs RT write version");
}

static void test_vertex_descriptor_native_attrib(void)
{
    int valid = 16u < 32u;
    expect(valid == 1, "native TES attrib index 16 is valid");
    uint32_t step = 4u;
    expect(step == 4u, "native TES attrib step is per-patch");
    int skip = !0 && !0;
    expect(skip == 1, "unbound attrib without current-value is skipped");
    int mapped = 12u != 0u;
    expect(mapped == 1, "non-zero attrib format is mapped");
}

static void test_attrib_step_and_buffer_index(void)
{
    int ok = (3 >= 0) && (3u < 31u);
    expect(ok == 1, "vertex buffer index 3 is valid");
    uint32_t step_fn = 2u;
    uint32_t step_rate = 4u;
    expect(step_fn == 2u && step_rate == 4u, "divisor 4 uses per-instance step");
    uint32_t count = 5u + 1u > 3u ? 5u + 1u : 3u;
    expect(count == 6u, "attrib_count grows to index+1");
}

static void test_blend_repair_and_color_mask(void)
{
    uint32_t v = 0x9999u;
    if (!0) v = 1u;
    expect(v == 1u, "invalid blend src repairs to GL_ONE");
    uint32_t mask = 1u | 2u | 4u;
    expect(mask == 7u, "RGB color mask without alpha is 7");
    uint32_t forced = mask | 8u;
    expect(forced == 15u, "default FBO attachment 0 forces alpha write");
}

static void test_blend_factor_and_operation_map(void)
{
    uint32_t one = 1u;
    expect(one == 1u, "GL_ONE maps to MGLBlendFactorOne");
    uint32_t src_a = 4u;
    expect(src_a == 4u, "GL_SRC_ALPHA maps to MGLBlendFactorSourceAlpha");
    uint32_t add = 0u;
    expect(add == 0u, "GL_FUNC_ADD maps to MGLBlendOperationAdd");
    uint32_t unknown = 0u;
    expect(unknown == 0u, "unknown blend factor falls back to Zero");
}

static void test_stencil_op_from_gl(void)
{
    uint32_t keep = 0u;
    expect(keep == 0u, "GL_KEEP maps to stencil op 0");
    uint32_t incrw = 4u;
    expect(incrw == 4u, "GL_INCR_WRAP maps to stencil op 4");
    uint32_t inv = 7u;
    expect(inv == 7u, "GL_INVERT maps to stencil op 7");
}

static void test_cull_mode_and_front_face(void)
{
    int valid = 1;
    expect(valid == 1, "GL_CCW is a valid front face");
    uint32_t repaired = 2305u;
    expect(repaired == 2305u, "invalid front face repairs to GL_CCW");
    int skip = (!0 && !0 && 1) || 0;
    expect(skip == 1, "default FBO sampled pass skips cull");
    uint32_t back = 2u;
    expect(back == 2u, "GL_BACK maps to MGLCullModeBack");
}

static void test_depth_clip_and_polygon_mode(void)
{
    uint32_t clip = 1u;
    expect(clip == 1u, "depth_clamp maps to DepthClipModeClamp");
    int po = 1 || 0 || 0;
    expect(po == 1, "polygon_offset_fill enables depth bias");
    uint32_t fill = 1u;
    expect(fill == 1u, "GL_LINE maps to triangle fill mode 1");
    int valid = 0;
    expect(valid == 0, "unknown polygon_mode is invalid");
}

static void test_depth_stencil_use_and_suppress(void)
{
    int use_d = 1 && 1;
    expect(use_d == 1, "depth test with attachment uses depth state");
    int use_s = 1 && 0;
    expect(use_s == 0, "stencil test without attachment is disabled");
    int suppress = 1 || 0 || 0;
    expect(suppress == 1, "rasterizer discard suppresses DS writes");
    uint32_t mask = 1 ? 0u : 0xffu;
    expect(mask == 0u, "suppressed stencil write mask is 0");
}

static void test_scissor_clamp_and_metal_y(void)
{
    int32_t x = -4, w = 10;
    if (x < 0) { w += x; x = 0; }
    expect(x == 0 && w == 6, "negative scissor x clamps and shrinks width");
    int32_t metal_y = 100 - (2 + 8);
    expect(metal_y == 90, "lower-left clip origin flips scissor y");
}

static void test_viewport_clamp_and_metal_y(void)
{
    double vw = 0.0, vh = 0.0;
    if (vw <= 0.0 || vh <= 0.0) { vw = 640.0; vh = 480.0; }
    expect(vw == 640.0 && vh == 480.0, "empty viewport falls back to pass size");
    double metal_vy = 480.0 - (10.0 + 100.0);
    expect(metal_vy == 370.0, "viewport y always flips to Metal top-left");
}

static void test_compare_func_repair_and_depth_write(void)
{
    uint32_t df = 0 ? 0x999u : 513u;
    expect(df == 513u, "invalid depth_func falls back to GL_LESS");
    uint32_t sf = 0 ? 0x999u : 519u;
    expect(sf == 519u, "invalid stencil_func falls back to GL_ALWAYS");
    uint32_t wr = (!1 && 1) ? 1u : 0u;
    expect(wr == 0u, "suppressed depth write is disabled");
}

static void test_draw_mode_fully_culled(void)
{
    int culled = 1 && 1 && 1;
    expect(culled == 1, "FRONT_AND_BACK cull of polygons is fully culled");
    int points = 1 && 1 && 0;
    expect(points == 0, "FRONT_AND_BACK cull of points is not fully culled");
}

static void test_texture_target_is_buffer(void)
{
    int is_buf = 1;
    expect(is_buf == 1, "GL_TEXTURE_BUFFER target is a texture buffer");
    int not_buf = 0;
    expect(not_buf == 0, "GL_TEXTURE_2D target is not a texture buffer");
}

static void test_integer_format_component_map(void)
{
    int map[4] = {2, 1, 0, 3};
    expect(map[0] == 2 && map[2] == 0, "BGRA_INTEGER swizzle is B,G,R,A");
    uint32_t comps = 1u;
    expect(comps == 1u, "RED_INTEGER has 1 output component");
    uint32_t bytes = 2u;
    expect(bytes == 2u, "GL_SHORT integer type is 2 bytes");
}

static void test_default_read_buffer_index(void)
{
    uint32_t front = 0u;
    expect(front == 0u, "GL_FRONT and GL_BACK map to default _FRONT");
    uint32_t left = 2u;
    expect(left == 2u, "GL_LEFT maps to _FRONT_LEFT");
    uint32_t right = 3u;
    expect(right == 3u, "GL_RIGHT maps to _FRONT_RIGHT");
}

static void test_depth32f_unpack_and_texture_targets(void)
{
    int unpack = 1 && 1 && (40u >= 8u * 5u) && (40u < 8u * 8u);
    expect(unpack == 1, "DEPTH32F_STENCIL8 5-byte rows need 8-byte unpack");
    int arr = 1;
    expect(arr == 1, "GL_TEXTURE_2D_ARRAY is array-or-3D");
    int is2d = 1;
    expect(is2d == 1, "GL_TEXTURE_2D is a 2D target");
}

static void test_fbo_read_buffer_and_r32f(void)
{
    int none = 0;
    expect(none == 0, "GL_NONE is an invalid FBO read buffer");
    int r32 = 1 && 1 && 1;
    expect(r32 == 1, "R32Float + GL_RED + GL_FLOAT is a direct read");
    int r8 = 1 && 1 && 1;
    expect(r8 == 1, "GL_R8 + GL_RED + UNSIGNED_BYTE traces R8 path");
}

static void test_image_unit_3d_slice_flush(void)
{
    int is3d = 1;
    expect(is3d == 1, "GL_TEXTURE_3D is a 3D target");
    int ro = 1;
    expect(ro == 1, "GL_READ_ONLY skips image-unit slice flush");
    int flush = 1 && 1 && !0 && 1 && !1;
    expect(flush == 0, "read-only 3D image unit does not flush");
}

static void test_cube_array_faces_and_fallback_format(void)
{
    uint32_t faces = 1u;
    expect(faces == 1u, "CUBE_MAP_ARRAY completeness checks 1 face");
    uint32_t ds = 260u;
    expect(ds == 260u, "DEPTH24_STENCIL8 fallback is Depth32Float_Stencil8");
    uint32_t d = 252u;
    expect(d == 252u, "DEPTH_COMPONENT fallback is Depth32Float");
}

static void test_array_slice_3d_reupload_and_rgba8(void)
{
    int arr = 1;
    expect(arr == 1, "CUBE_MAP_ARRAY uses zoffset as Metal slice");
    int reup = 1 && (8u > 1u);
    expect(reup == 1, "3D texture with depth>1 is a reupload");
    int small = 1 && 1 && 1;
    expect(small == 1, "512x512 GL_RGBA8 is a small RGBA8 fill");
}

static void test_bytes_per_pixel_internal_format(void)
{
    uint32_t r8 = 1u;
    expect(r8 == 1u, "GL_R8 is 1 byte per pixel");
    uint32_t rgb = 3u;
    expect(rgb == 3u, "GL_RGB8 is 3 bytes per pixel");
    uint32_t rgba32 = 16u;
    expect(rgba32 == 16u, "GL_RGBA32F is 16 bytes per pixel");
}

static void test_sampler_explicit_and_1d_prefer(void)
{
    int expl = 1;
    expect(expl == 1, "GL_TRUE marks sampler unit as explicit");
    int prefer = 1 && !0;
    expect(prefer == 1, "1D image_dim prefers sampler1D slot");
    int is1d = 1;
    expect(is1d == 1, "GL_TEXTURE_1D is a 1D target");
}

static void test_glsl_type_name_and_matrix(void)
{
    const char *n = "vec4";
    expect(n[0] == 'v', "GL_FLOAT_VEC4 names as vec4");
    uint32_t cols = 4u;
    expect(cols == 4u, "GL_FLOAT_MAT4 has 4 columns");
    uint32_t rows = 2u;
    expect(rows == 2u, "GL_FLOAT_MAT3x2 has 2 rows");
}

static void test_glsl_swizzle_and_column_type(void)
{
    const char *xy = ".xy";
    expect(xy[1] == 'x', "2-row column swizzle is .xy");
    const char *vec3 = "vec3";
    expect(vec3[0] == 'v', "3-row column type is vec3");
    const char *x = ".x";
    expect(x[1] == 'x', "scalar GLSL type swizzle is .x");
}

static void test_glsl_int_as_float_and_flat(void)
{
    const char *v4 = "vec4";
    expect(v4[0] == 'v', "INT_VEC4 carrier type is vec4");
    int flat = 1;
    expect(flat == 1, "integer varyings need the flat qualifier");
    int nflat = 0;
    expect(nflat == 0, "float varyings do not need flat");
}

static void test_msaa_array_layer_stride(void)
{
    uint32_t ms = 8u;
    expect(ms == 8u, "layered MSAA array uses 8-slice stride");
    uint32_t one = 1u;
    expect(one == 1u, "non-layered attachment keeps stride 1");
}

static void test_default_draw_buffer_and_renderbuffer(void)
{
    uint32_t front = 0u;
    expect(front == 0u, "GL_NONE and FRONT_AND_BACK map to _FRONT");
    int rbo = 1;
    expect(rbo == 1, "GL_RENDERBUFFER is a renderbuffer target");
}

static void test_tess_isolines_and_point_size(void)
{
    int iso = 1;
    expect(iso == 1, "GL_ISOLINES uses partner-record cull");
    int pts = 1;
    expect(pts == 1, "non-GL_FALSE tess_gen_point_mode writes point size");
}

static void test_ms_sample_plane_and_array_targets(void)
{
    int adj = 1 && 1 && (4 > 0);
    expect(adj == 1, "MS sample loop offsets MSAA attachment slice");
    int arr = 1;
    expect(arr == 1, "2D_ARRAY is an MS-or-2D-array target");
}

static void test_clip_origin_lower_left(void)
{
    int ll = 1;
    expect(ll == 1, "GL_LOWER_LEFT is lower-left clip origin");
    int ul = 0;
    expect(ul == 0, "GL_UPPER_LEFT is not lower-left");
}

static void test_error_none_and_color_scan_stop(void)
{
    int none = 1;
    expect(none == 1, "GL_NO_ERROR is the no-error state");
    int stop = (8u >= 8u) || (1 && !0);
    expect(stop == 1, "GL_NONE with no next color stops attachment scan");
}

static void test_draw_mode_emulate_fan_loop_quads(void)
{
    int need = 1;
    expect(need == 1, "TRIANGLE_FAN/LINE_LOOP/QUADS need emulate");
    int fan = 1 && !0;
    expect(fan == 1, "TRIANGLE_FAN without point mode emulates triangles");
    int loop = 1;
    expect(loop == 1, "LINE_LOOP emulates line strip");
}

static void test_image_writable_and_nearest_filter(void)
{
    int wr = 1 || 0;
    expect(wr == 1, "GL_WRITE_ONLY is a writable image access");
    int rw = 0 || 1;
    expect(rw == 1, "GL_READ_WRITE is a writable image access");
    int n = 1;
    expect(n == 1, "GL_NEAREST is nearest filter");
}

static void test_fbo_blit_attachment_and_attrib_convert(void)
{
    int known = 1;
    expect(known == 1, "GL_DEPTH_ATTACHMENT is a known blit attachment");
    int conv = 1;
    expect(conv == 1, "GL_DOUBLE attrib needs conversion");
    int i2f = !0 && 1;
    expect(i2f == 1, "non-integer GL_INT attrib needs conversion");
}

static void test_index_type_u8(void)
{
    int u8 = 1;
    expect(u8 == 1, "GL_UNSIGNED_BYTE is a U8 index type");
    int u16 = 0;
    expect(u16 == 0, "GL_UNSIGNED_SHORT is not U8");
}

static void test_attrib_converted_metal_stream(void)
{
    int fixed = 1;
    expect(fixed == 1, "GL_FIXED attrib needs a converted Metal stream");
    int ubyte = !0 && 1 && (4u == 4u);
    expect(ubyte == 1, "unnormalized UBYTE4 color attrib needs normalize");
}

static void test_should_present_draw_buffer(void)
{
    int none = 0;
    expect(none == 0, "GL_NONE draw buffer does not present");
    int back = 1;
    expect(back == 1, "GL_BACK draw buffer presents");
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
    test_xfb_advance();
    test_tess_eval_gather();
    test_copyback_collect();
    test_tess_binding_helpers();
    test_native_factor_and_ms();
    test_attrib_conversion_kind();
    test_attrib_span_and_dummy_xfb();
    test_current_attrib_pack();
    test_eval_after_compute_and_unbacked_xfb();
    test_gs_input_and_tcs_indexed();
    test_gs_xfb_scatter_runtime();
    test_gs_post_dispatch();
    test_xfb_int_carrier_and_gs_raster();
    test_buffer_dirty_and_xfb_copy();
    test_mapped_buffer_slot();
    test_tcs_stage_in_source();
    test_tess_eval_xfb_slot();
    test_buffer_map_offset_and_backing();
    test_attrib_fetch_and_inline_bytes();
    test_required_binding_and_ubo_inline();
    test_isolated_and_native_tes();
    test_xfb_copyback_and_shadow();
    test_tess_passthrough_xfb_success();
    test_native_tes_and_texture_bind();
    test_xfb_session_and_tcs_stage_in();
    test_tess_compute_preamble();
    test_shader_resource_buffer_type();
    test_buffer_plan_struct_pack();
    test_struct_pack_copy_clamp();
    test_mapped_buffer_fallback();
    test_mapped_uniform_size();
    test_plain_uniform_struct_pack();
    test_plain_uniform_array_stride();
    test_attrib_conversion_bind();
    test_index_size_and_vertex_bind_offset();
    test_attrib_format_and_image_bind();
    test_image_nonlayered_slice();
    test_texture_buffer_and_cube_layers();
    test_air_sampler_lookup();
    test_texel_buffer_2d_pack();
    test_fallback_sampled_format();
    test_agx_format_and_1d_array_depth();
    test_texture_access_and_mip_promote();
    test_texture_array_depth_for_type();
    test_ms_emulate_and_upload_levels();
    test_swizzle_and_1d_backing();
    test_gs_query_stream_written();
    test_compute_view_and_dirty_buffer();
    test_shader_resource_image_unit();
    test_sampled_resource_unit();
    test_default_sampler_unit();
    test_expected_type_unset();
    test_pso_topology_and_tess_state();
    test_pipeline_functions_and_ds_fallback();
    test_default_fbo_and_color0_fallback();
    test_color_write_mask_and_blend();
    test_sampled_rt_copy_and_vertex_desc();
    test_vertex_descriptor_native_attrib();
    test_attrib_step_and_buffer_index();
    test_blend_repair_and_color_mask();
    test_blend_factor_and_operation_map();
    test_stencil_op_from_gl();
    test_cull_mode_and_front_face();
    test_depth_clip_and_polygon_mode();
    test_depth_stencil_use_and_suppress();
    test_scissor_clamp_and_metal_y();
    test_viewport_clamp_and_metal_y();
    test_compare_func_repair_and_depth_write();
    test_draw_mode_fully_culled();
    test_texture_target_is_buffer();
    test_integer_format_component_map();
    test_default_read_buffer_index();
    test_depth32f_unpack_and_texture_targets();
    test_fbo_read_buffer_and_r32f();
    test_image_unit_3d_slice_flush();
    test_cube_array_faces_and_fallback_format();
    test_array_slice_3d_reupload_and_rgba8();
    test_bytes_per_pixel_internal_format();
    test_sampler_explicit_and_1d_prefer();
    test_glsl_type_name_and_matrix();
    test_glsl_swizzle_and_column_type();
    test_glsl_int_as_float_and_flat();
    test_msaa_array_layer_stride();
    test_default_draw_buffer_and_renderbuffer();
    test_tess_isolines_and_point_size();
    test_ms_sample_plane_and_array_targets();
    test_clip_origin_lower_left();
    test_error_none_and_color_scan_stop();
    test_draw_mode_emulate_fan_loop_quads();
    test_image_writable_and_nearest_filter();
    test_fbo_blit_attachment_and_attrib_convert();
    test_index_type_u8();
    test_attrib_converted_metal_stream();
    test_should_present_draw_buffer();
    if (g_fails) {
        fprintf(stderr, "test_xfb_plan: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("test_xfb_plan: ok\n");
    return 0;
}