/* SPDX-License-Identifier: LGPL-3.0-only
 * MGL domain generation from OpenGL 4.6 section 11.2 / ARB_tessellation_shader
 * sections 2.X.2.1 (triangular rings), 2.X.2.2 (quad grid and boundary strips).
 * Fractional short segments are placed at opposite ends of each edge.
 * Strip diagonals and stream ordering are implementation-defined choices.
 */
#include "mgl_tess_domain.h"
#include "glcorearb.h"
#include <stddef.h>
#include <string.h>

#define TESS_MAX_POINTS 4225u

typedef struct TessSink {
    void *output;
    uint32_t capacity, stride, count, target;
    int indexed;
} TessSink;

typedef struct TessMesh {
    MGLTessCoord points[TESS_MAX_POINTS];
    uint32_t point_count;
    uint32_t point_mode, winding;
    TessSink *sink;
} TessMesh;

static float edge_position(float factor, uint32_t segments, uint32_t spacing,
                           uint32_t index)
{
    if (!index) return 0.f;
    if (index == segments) return 1.f;
    if (index > segments / 2u)
        return 1.f - edge_position(factor, segments, spacing, segments - index);
    if (spacing != GL_FRACTIONAL_ODD && spacing != GL_FRACTIONAL_EVEN)
        return (float)index / (float)segments;
    /* n-2 regular segments of length 1/f; the remaining length is shared
     * by the two end segments. Compute in double before publishing float. */
    return (float)(((double)factor - (segments - 2u)) / (2.0 * factor) +
                   (double)(index - 1u) / factor);
}

static void emit(TessSink *sink, MGLTessCoord point)
{
    uint32_t index = sink->count++;
    if (sink->indexed) {
        if (index != sink->target) return;
        index = 0;
    }
    if (sink->output && index < sink->capacity)
        memcpy((char *)sink->output + (size_t)index * sink->stride, &point, sizeof point);
}

static uint32_t vertex(TessMesh *mesh, float u, float v, float w)
{
    const uint32_t index = mesh->point_count++;
    mesh->points[index] = (MGLTessCoord){u, v, w};
    if (mesh->point_mode) emit(mesh->sink, mesh->points[index]);
    return index;
}

static void triangle(TessMesh *mesh, uint32_t a, uint32_t b, uint32_t c)
{
    if (mesh->point_mode) return;
    emit(mesh->sink, mesh->points[a]);
    emit(mesh->sink, mesh->points[mesh->winding == GL_CW ? c : b]);
    emit(mesh->sink, mesh->points[mesh->winding == GL_CW ? b : c]);
}

/* Two monotone edge chains bound a strip. Advance one chain at a time,
 * producing non-overlapping triangles; a one-point inner edge makes a fan. */
static void join_edges(TessMesh *mesh, const uint32_t *outer, uint32_t n,
                       const uint32_t *inner, uint32_t m)
{
    uint32_t i = 0, j = 0;
    while (i < n || j < m) {
        if (j == m || (i < n && (i + 1u) * m <= (j + 1u) * n)) {
            triangle(mesh, outer[i], outer[i + 1], inner[j]);
            i++;
        } else {
            triangle(mesh, outer[i], inner[j + 1], inner[j]);
            j++;
        }
    }
}

static void generate_quads(const MGLTessFactorInput *in,
                           const MGLTessNormalizedFactors *n, TessMesh *mesh)
{
    /* Counter-clockwise boundary: bottom, right, top, left. */
    const unsigned levels[4] = {1, 2, 3, 0};
    uint32_t outer[4][65], lengths[4];
    uint32_t grid[63][63];
    const uint32_t nx = n->inner_eff[1], ny = n->inner_eff[0];
    for (unsigned edge = 0; edge < 4; edge++) {
        lengths[edge] = n->outer_eff[levels[edge]];
        for (uint32_t i = 0; i < lengths[edge]; i++) {
            const float t = edge_position(n->outer_clamped[levels[edge]],
                lengths[edge], in->spacing, i);
            const float u = edge == 0 ? t : edge == 1 ? 1.f : edge == 2 ? 1.f - t : 0.f;
            const float v = edge == 0 ? 0.f : edge == 1 ? t : edge == 2 ? 1.f : 1.f - t;
            outer[edge][i] = vertex(mesh, u, v, 0.f);
        }
    }
    for (unsigned edge = 0; edge < 4; edge++)
        outer[edge][lengths[edge]] = outer[(edge + 1u) % 4u][0];
    if (nx == 1 && ny == 1) {
        triangle(mesh, outer[0][0], outer[1][0], outer[2][0]);
        triangle(mesh, outer[0][0], outer[2][0], outer[3][0]);
        return;
    }
    for (uint32_t y = 1; y < ny; y++) {
        const float v = edge_position(n->inner_clamped[0], ny, in->spacing, y);
        for (uint32_t x = 1; x < nx; x++) {
            const float u = edge_position(n->inner_clamped[1], nx, in->spacing, x);
            grid[y - 1][x - 1] = vertex(mesh, u, v, 0.f);
        }
    }
    for (uint32_t y = 0; y + 2 < ny; y++)
        for (uint32_t x = 0; x + 2 < nx; x++) {
            triangle(mesh, grid[y][x], grid[y][x + 1], grid[y + 1][x + 1]);
            triangle(mesh, grid[y][x], grid[y + 1][x + 1], grid[y + 1][x]);
        }
    for (unsigned edge = 0; edge < 4; edge++) {
        uint32_t inner[65];
        const uint32_t length = (edge & 1u) ? ny - 2u : nx - 2u;
        for (uint32_t i = 0; i <= length; i++) {
            const uint32_t x = edge == 0 ? i : edge == 1 ? nx - 2u : edge == 2 ? nx - 2u - i : 0u;
            const uint32_t y = edge == 0 ? 0u : edge == 1 ? i : edge == 2 ? ny - 2u : ny - 2u - i;
            inner[i] = grid[y][x];
        }
        join_edges(mesh, outer[edge], lengths[edge], inner, length);
    }
}

static void generate_triangles(const MGLTessFactorInput *in,
                               const MGLTessNormalizedFactors *n, TessMesh *mesh)
{
    const unsigned levels[3] = {1, 2, 0};
    uint32_t outer[3][65], lengths[3];
    const uint32_t segments = n->inner_eff[0];
    for (unsigned edge = 0; edge < 3; edge++) {
        lengths[edge] = n->outer_eff[levels[edge]];
        for (uint32_t i = 0; i < lengths[edge]; i++) {
            const float t = edge_position(n->outer_clamped[levels[edge]],
                lengths[edge], in->spacing, i);
            const float u = edge == 0 ? t : edge == 1 ? 1.f - t : 0.f;
            const float v = edge == 0 ? 0.f : edge == 1 ? t : 1.f - t;
            outer[edge][i] = vertex(mesh, u, v, 1.f - u - v);
        }
    }
    for (unsigned edge = 0; edge < 3; edge++)
        outer[edge][lengths[edge]] = outer[(edge + 1u) % 3u][0];
    if (segments == 1) {
        triangle(mesh, outer[0][0], outer[1][0], outer[2][0]);
        return;
    }
    for (uint32_t ring = 1; ring <= segments / 2u; ring++) {
        uint32_t inner[3][65];
        const uint32_t length = segments - 2u * ring;
        if (!length) {
            const uint32_t center = vertex(mesh, 1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
            for (unsigned edge = 0; edge < 3; edge++) inner[edge][0] = center;
        } else {
            /* In the equilateral domain, perpendiculars through t on the
             * adjacent edges meet at barycentric (2t/3, 2t/3, 1-4t/3).
             * Project each remaining outer subdivision onto that ring. */
            const float a = (2.f / 3.f) * edge_position(n->inner_clamped[0],
                segments, in->spacing, ring);
            for (unsigned edge = 0; edge < 3; edge++) {
                for (uint32_t i = 0; i < length; i++) {
                    const float t = i == 0 ? a : edge_position(n->inner_clamped[0],
                        segments, in->spacing, ring + i) - 0.5f * a;
                    const float u = edge == 0 ? t : edge == 1 ? 1.f - a - t : a;
                    const float v = edge == 0 ? a : edge == 1 ? t : 1.f - a - t;
                    inner[edge][i] = vertex(mesh, u, v, 1.f - u - v);
                }
            }
            for (unsigned edge = 0; edge < 3; edge++)
                inner[edge][length] = inner[(edge + 1u) % 3u][0];
        }
        for (unsigned edge = 0; edge < 3; edge++) {
            join_edges(mesh, outer[edge], lengths[edge], inner[edge], length);
            memcpy(outer[edge], inner[edge], (length + 1u) * sizeof(uint32_t));
            lengths[edge] = length;
        }
        if (length == 1) triangle(mesh, inner[0][0], inner[1][0], inner[2][0]);
    }
}

static void generate(const MGLTessFactorInput *in, TessSink *sink)
{
    MGLTessNormalizedFactors n;
    if (!in) return;
    mglTessNormalizeFactors(in, &n);
    if (n.discard) return;
    if (in->gen_mode == GL_ISOLINES) {
        const uint32_t lines = n.outer_eff[0], segments = n.outer_eff[1];
        for (uint32_t line = 0; line < lines; line++) {
            for (uint32_t i = 0; i <= segments; i++) {
                const float u = edge_position(n.outer_clamped[1], segments, in->spacing, i);
                const MGLTessCoord point = {u, (float)line / lines, 0.f};
                if (in->point_mode || i > 0) emit(sink, point);
                if (!in->point_mode && i < segments) emit(sink, point);
            }
        }
        return;
    }
    TessMesh mesh;
    mesh.point_count = 0;
    mesh.point_mode = in->point_mode;
    mesh.winding = in->winding;
    mesh.sink = sink;
    if (in->gen_mode == GL_QUADS) generate_quads(in, &n, &mesh);
    else generate_triangles(in, &n, &mesh);
}

uint32_t mglTessDomainVertexCount(const MGLTessFactorInput *in)
{
    TessSink sink = {0};
    generate(in, &sink);
    return sink.count;
}

int mglTessDomainCoordAt(const MGLTessFactorInput *in, uint32_t index, MGLTessCoord *out)
{
    TessSink sink = {out, 1, sizeof(*out), 0, index, 1};
    if (!out) return -1;
    generate(in, &sink);
    return index < sink.count ? 0 : -1;
}

uint32_t mglTessGenerateDomainStrided(const MGLTessFactorInput *in, void *coords,
                                    uint32_t cap, uint32_t stride)
{
    TessSink sink = {coords, cap, stride, 0, 0, 0};
    if (!coords || stride < sizeof(MGLTessCoord)) return 0;
    generate(in, &sink);
    return sink.count <= cap ? sink.count : 0;
}

uint32_t mglTessGenerateDomain(const MGLTessFactorInput *in, MGLTessCoord *coords, uint32_t cap)
{
    return mglTessGenerateDomainStrided(in, coords, cap, sizeof(*coords));
}
