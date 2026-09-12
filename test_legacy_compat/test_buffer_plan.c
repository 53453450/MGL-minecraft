/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Vertex-attribute buffer map harness (O5.1).
 *
 * Behavioural oracle for mglRenderPlanVertexAttribBuffers: the VAO attribute
 * buffers -> Metal vertex buffer slot decisions that used to live in
 * MGLRenderer+Buffer.m (mapVertexAttributeBuffersToBufferMap:).  Drives the
 * plan function directly with hand-populated VertexArray / Buffer fixtures and
 * a stub resolver, so it needs no Metal, no ObjC and no live GL context.
 *
 * The resolver seam is what makes this testable: the plan owns candidate
 * iteration, stream grouping, slot assignment, capacity / overflow guards and
 * diagnostics, while resolving one attribute against live GL state is supplied
 * by the caller (MGLRenderer+Buffer.m passes mglResolveVertexAttribForPlan).
 */

#include "mgl_vertex_attrib_plan.h"
#include "mgl_buffer_slots.h"

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

/* ---- fixtures ---------------------------------------------------- */

static Buffer g_buffers[4];
static MGLResolvedVertexAttribBinding g_resolved[MAX_ATTRIBS];
static int g_resolve_ok[MAX_ATTRIBS];
static int g_resolve_calls[MAX_ATTRIBS];

/* Stub resolver: succeeds for attributes marked in g_resolve_ok and copies the
 * prepared binding out, counting the calls so the harness can assert the plan
 * only resolves attributes it actually considers. */
static int stub_resolve(void *user, GLuint attribute,
                        MGLResolvedVertexAttribBinding *out)
{
    (void)user;
    if (attribute >= MAX_ATTRIBS || !out) {
        return 1;
    }
    g_resolve_calls[attribute]++;
    if (!g_resolve_ok[attribute]) {
        return 1;
    }
    *out = g_resolved[attribute];
    return 0;
}

static void reset_fixtures(void)
{
    memset(g_buffers, 0, sizeof(g_buffers));
    memset(g_resolved, 0, sizeof(g_resolved));
    memset(g_resolve_ok, 0, sizeof(g_resolve_ok));
    memset(g_resolve_calls, 0, sizeof(g_resolve_calls));
    g_buffers[0].name = 101u;
    g_buffers[0].target = 0x8892u; /* GL_ARRAY_BUFFER */
    g_buffers[1].name = 202u;
    g_buffers[1].target = 0x8892u;
    g_buffers[2].name = 303u;
    g_buffers[2].target = 0x8892u;
}

static void bind_attrib(GLuint attribute, Buffer *buffer, GLintptr offset,
                        GLuint stride, GLuint divisor)
{
    g_resolved[attribute].buffer = buffer;
    g_resolved[attribute].binding_offset = offset;
    g_resolved[attribute].stride = stride;
    g_resolved[attribute].divisor = divisor;
    g_resolve_ok[attribute] = 1;
}

static int run_plan(BufferMapList *map, uint32_t candidate_mask, int inputs)
{
    MGLVertexAttribBufferPlanInput in;
    memset(&in, 0, sizeof(in));
    in.candidate_mask = candidate_mask;
    in.stage_input_count = inputs;
    in.stage = 0; /* _VERTEX_SHADER */
    in.map_capacity = MAX_MAPPED_BUFFERS;
    in.resolve = stub_resolve;
    return mglRenderPlanVertexAttribBuffers(map, &in);
}

/* ---- cases ------------------------------------------------------- */

static void test_single_attrib(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 64, 16u, 0u);

    expect(run_plan(&map, 0x1u, 1) == 0, "single attrib plan succeeds");
    expect(map.count == 1, "single attrib maps exactly one entry");
    if (map.count == 1) {
        const BufferMap *e = &map.buffers[0];
        expect(e->attribute_mask == 0x1u, "single attrib mask bit 0");
        expect(e->buf == &g_buffers[0], "single attrib buffer identity");
        expect(e->offset == 64, "single attrib binding offset");
        expect(e->size == 0, "single attrib size left for the caller");
        expect(e->buffer_base_index == (GLuint)kMGLVertexAttribBufferBase,
               "single attrib takes the vertex attrib base slot");
        expect(e->has_metal_binding == (GLboolean)GL_FALSE,
               "single attrib starts without a metal binding");
        expect(e->metal_binding_index == 0u, "single attrib metal index unset");
    }
    expect(g_resolve_calls[0] == 1, "resolver called once for the candidate");
    expect(g_resolve_calls[1] == 0, "non-candidate attribute never resolved");
}

static void test_shared_stream_groups(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 0, 32u, 0u);
    bind_attrib(1u, &g_buffers[0], 16, 32u, 0u);

    expect(run_plan(&map, 0x3u, 2) == 0, "shared-stream plan succeeds");
    expect(map.count == 1, "same buffer + stride/divisor share one entry");
    if (map.count == 1) {
        expect(map.buffers[0].attribute_mask == 0x3u,
               "shared entry carries both attribute bits");
        expect(map.buffers[0].offset == 0,
               "shared entry keeps the first attribute's offset");
    }
    /* One resolve per candidate plus one re-resolve of the already-mapped
     * attribute to test stream compatibility. */
    expect(g_resolve_calls[0] == 2, "mapped attribute re-resolved for compat");
    expect(g_resolve_calls[1] == 1, "new attribute resolved once");
}

static void test_divergent_stride_splits_streams(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 0, 16u, 0u);
    bind_attrib(1u, &g_buffers[0], 32, 32u, 0u);

    expect(run_plan(&map, 0x3u, 2) == 0, "divergent-stride plan succeeds");
    expect(map.count == 2, "same buffer with different stride splits");
    if (map.count == 2) {
        expect(map.buffers[0].attribute_mask == 0x1u, "first stream mask");
        expect(map.buffers[1].attribute_mask == 0x2u, "second stream mask");
        expect(map.buffers[0].buffer_base_index ==
                   (GLuint)kMGLVertexAttribBufferBase,
               "first stream base slot");
        expect(map.buffers[1].buffer_base_index ==
                   (GLuint)kMGLVertexAttribBufferBase + 1u,
               "second stream takes the next slot");
        expect(map.buffers[1].offset == 32, "second stream offset");
        expect(map.buffers[1].buf == &g_buffers[0],
               "second stream keeps the same buffer object");
    }
}

static void test_divergent_divisor_splits_streams(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 0, 16u, 0u);
    bind_attrib(1u, &g_buffers[0], 0, 16u, 1u);

    expect(run_plan(&map, 0x3u, 2) == 0, "divergent-divisor plan succeeds");
    expect(map.count == 2, "same buffer with different divisor splits");
}

static void test_distinct_buffers(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 0, 16u, 0u);
    bind_attrib(1u, &g_buffers[1], 0, 16u, 0u);
    bind_attrib(2u, &g_buffers[2], 0, 16u, 0u);

    expect(run_plan(&map, 0x7u, 3) == 0, "distinct-buffer plan succeeds");
    expect(map.count == 3, "each distinct buffer takes an entry");
    if (map.count == 3) {
        for (int i = 0; i < 3; i++) {
            expect(map.buffers[i].attribute_mask == (GLuint)(0x1u << i),
                   "distinct entry mask follows the attribute");
            expect(map.buffers[i].buffer_base_index ==
                       (GLuint)kMGLVertexAttribBufferBase + (GLuint)i,
                   "distinct entries take consecutive slots");
        }
    }
}

static void test_unresolvable_attribute_is_skipped(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(1u, &g_buffers[1], 0, 16u, 0u);
    g_resolve_ok[0] = 0; /* candidate without a usable buffer */

    expect(run_plan(&map, 0x3u, 1) == 0,
           "a candidate without a buffer does not fail the plan");
    expect(map.count == 1, "only the resolvable attribute is mapped");
    if (map.count == 1) {
        expect(map.buffers[0].attribute_mask == 0x2u,
               "resolvable attribute keeps the right mask bit");
        expect(map.buffers[0].buffer_base_index ==
                   (GLuint)kMGLVertexAttribBufferBase,
               "skipping an attribute does not consume a slot");
    }
}

static void test_no_candidates_maps_nothing(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();

    expect(run_plan(&map, 0x0u, 0) == 0, "empty candidate mask succeeds");
    expect(map.count == 0, "empty candidate mask maps nothing");
    expect(map.buffers[0].buf == NULL, "placeholder entry stays empty");
    expect(map.buffers[0].buffer_base_index ==
               (GLuint)kMGLVertexAttribBufferBase,
           "placeholder entry still carries the vertex attrib base slot");
}

static void test_existing_entries_are_preserved(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    /* The shader-buffer pass fills the map first; attribute entries append. */
    map.count = 1;
    map.buffers[0].buf = &g_buffers[3];
    map.buffers[0].buffer_base_index = 5u;
    map.buffers[0].attribute_mask = 0u;
    bind_attrib(0u, &g_buffers[0], 0, 16u, 0u);

    expect(run_plan(&map, 0x1u, 1) == 0, "append plan succeeds");
    expect(map.count == 2, "attribute entry appends after shader buffers");
    expect(map.buffers[0].buf == &g_buffers[3],
           "pre-existing entry is untouched");
    expect(map.buffers[0].buffer_base_index == 5u,
           "pre-existing slot index is untouched");
    if (map.count == 2) {
        expect(map.buffers[1].buf == &g_buffers[0],
               "appended entry holds the attribute buffer");
        expect(map.buffers[1].buffer_base_index ==
                   (GLuint)kMGLVertexAttribBufferBase,
               "attribute slots start at the vertex attrib base");
    }
}

static void test_capacity_guard(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 0, 16u, 0u);

    /* A full map must be rejected before the placeholder entry is written. */
    map.count = MAX_MAPPED_BUFFERS;
    expect(run_plan(&map, 0x1u, 1) == -1, "full map is rejected");
    expect(map.count == MAX_MAPPED_BUFFERS, "rejection leaves count unchanged");

    memset(&map, 0, sizeof(map));
    MGLVertexAttribBufferPlanInput in;
    memset(&in, 0, sizeof(in));
    in.candidate_mask = 0x1u;
    in.stage_input_count = 1;
    in.map_capacity = 1u; /* caller-supplied capacity, smaller than the map */
    in.resolve = stub_resolve;
    map.count = 1;
    expect(mglRenderPlanVertexAttribBuffers(&map, &in) == -1,
           "capacity predicate is honoured before appending");
    expect(map.count == 1, "capacity rejection leaves count unchanged");
}

static void test_count_mismatch_still_succeeds(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 0, 16u, 0u);

    /* mapped (1) != program stage inputs (3): warn only, never fail. */
    expect(run_plan(&map, 0x1u, 3) == 0,
           "buffer/input count mismatch is a warning, not a failure");
    expect(map.count == 1, "mismatch still maps the resolvable attribute");
}

static void test_resolver_rejects_a_shared_candidate(void)
{
    BufferMapList map;
    memset(&map, 0, sizeof(map));
    reset_fixtures();
    bind_attrib(0u, &g_buffers[0], 0, 16u, 0u);
    bind_attrib(2u, &g_buffers[0], 16, 16u, 0u);
    g_resolve_ok[1] = 0; /* masked-in candidate with no buffer */

    expect(run_plan(&map, 0x7u, 2) == 0, "mixed candidates succeed");
    expect(map.count == 1, "unresolvable candidate does not split the stream");
    if (map.count == 1) {
        expect(map.buffers[0].attribute_mask == 0x5u,
               "only resolvable attributes join the stream mask");
    }
}

int main(void)
{
    test_single_attrib();
    test_shared_stream_groups();
    test_divergent_stride_splits_streams();
    test_divergent_divisor_splits_streams();
    test_distinct_buffers();
    test_unresolvable_attribute_is_skipped();
    test_no_candidates_maps_nothing();
    test_existing_entries_are_preserved();
    test_capacity_guard();
    test_count_mismatch_still_succeeds();
    test_resolver_rejects_a_shared_candidate();

    if (g_fails) {
        fprintf(stderr, "vertex-attrib buffer plan harness: %d failure(s)\n",
                g_fails);
        return 1;
    }
    printf("vertex-attrib buffer plan harness: all cases passed\n");
    return 0;
}
