/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_draw_gs.h"

#include "mgl_air_gs_abi.h"
#include "mgl_draw_encode.h"
#include "mgl_render.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>

extern "C" bool mglDrawGsInputModeAccepts(GLenum gsMode, GLenum drawMode)
{
    switch (gsMode) {
        case GL_POINTS:
            return drawMode == GL_POINTS;
        case GL_LINES:
            return drawMode == GL_LINES || drawMode == GL_LINE_STRIP ||
                   drawMode == GL_LINE_LOOP;
        case GL_LINES_ADJACENCY:
            return drawMode == GL_LINES_ADJACENCY ||
                   drawMode == GL_LINE_STRIP_ADJACENCY;
        case GL_TRIANGLES:
            return drawMode == GL_TRIANGLES || drawMode == GL_TRIANGLE_STRIP ||
                   drawMode == GL_TRIANGLE_FAN;
        case GL_TRIANGLES_ADJACENCY:
            return drawMode == GL_TRIANGLES_ADJACENCY ||
                   drawMode == GL_TRIANGLE_STRIP_ADJACENCY;
        default:
            return false;
    }
}

extern "C" bool mglDrawGsGatherTopology(const uint8_t *indexBytes,
                                        GLenum indexType, GLsizei count,
                                        GLint first, bool indexed,
                                        bool restartEnabled,
                                        uint32_t restartIndex, GLenum mode,
                                        uint32_t **outGather,
                                        uint32_t *outGatherCount,
                                        uint32_t *outPrimitiveCount,
                                        uint32_t *outMaxIndex)
{
    if (!outGather || !outGatherCount || !outPrimitiveCount || !outMaxIndex ||
        count <= 0 || (indexed && !indexBytes) || (!indexed && first < 0)) {
        return false;
    }
    const uint32_t n = (uint32_t)count;
    const uint32_t elemBytes = indexType == GL_UNSIGNED_BYTE
                                   ? 1u
                                   : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    if ((size_t)n > SIZE_MAX / sizeof(uint32_t) ||
        (size_t)n > SIZE_MAX / (6u * sizeof(uint32_t))) {
        return false;
    }
    uint32_t *source = (uint32_t *)malloc((size_t)n * sizeof(*source));
    uint32_t *segment = (uint32_t *)malloc((size_t)n * sizeof(*segment));
    uint32_t *gather = (uint32_t *)malloc((size_t)n * 6u * sizeof(*gather));
    if (!source || !segment || !gather) {
        free(source);
        free(segment);
        free(gather);
        return false;
    }
    uint32_t maxIndex = 0u;
    for (uint32_t i = 0u; i < n; i++) {
        uint32_t value;
        if (!indexed) {
            const int64_t v = (int64_t)first + (int64_t)i;
            if (v < 0 || (uint64_t)v > UINT32_MAX) {
                free(source);
                free(segment);
                free(gather);
                return false;
            }
            value = (uint32_t)v;
        } else if (elemBytes == 1u) {
            value = indexBytes[i];
        } else if (elemBytes == 2u) {
            value = ((const uint16_t *)indexBytes)[i];
        } else {
            value = ((const uint32_t *)indexBytes)[i];
        }
        source[i] = value;
        if (!(indexed && restartEnabled && value == restartIndex) &&
            value > maxIndex) {
            maxIndex = value;
        }
    }

    uint32_t gathered = 0u;
    uint32_t primitives = 0u;
    uint32_t segmentCount = 0u;
    const bool restartMode = indexed && restartEnabled;
    const uint32_t primitiveWidth =
        (mode == GL_POINTS) ? 1u
        : (mode == GL_LINES || mode == GL_LINE_STRIP || mode == GL_LINE_LOOP)
            ? 2u
        : (mode == GL_LINES_ADJACENCY)          ? 4u
        : (mode == GL_TRIANGLES_ADJACENCY)      ? 6u
                                                : 3u;

#define EMIT(v)                                                                \
    do {                                                                       \
        gather[gathered++] = (v);                                              \
    } while (0)
#define EMIT_SEGMENT()                                                         \
    do {                                                                       \
        if (segmentCount > 0u) {                                               \
            if (mode == GL_POINTS) {                                           \
                for (uint32_t q = 0u; q < segmentCount; q++) {                 \
                    EMIT(segment[q]);                                          \
                    primitives++;                                              \
                }                                                              \
            } else if (mode == GL_LINES || mode == GL_TRIANGLES ||             \
                       mode == GL_LINES_ADJACENCY ||                           \
                       mode == GL_TRIANGLES_ADJACENCY) {                       \
                const uint32_t groups = segmentCount / primitiveWidth;         \
                for (uint32_t q = 0u; q < groups; q++) {                       \
                    for (uint32_t k = 0u; k < primitiveWidth; k++)             \
                        EMIT(segment[q * primitiveWidth + k]);                 \
                    primitives++;                                              \
                }                                                              \
            } else if (mode == GL_LINE_STRIP_ADJACENCY) {                      \
                for (uint32_t q = 0u; q + 3u < segmentCount; q++) {            \
                    for (uint32_t k = 0u; k < 4u; k++)                         \
                        EMIT(segment[q + k]);                                  \
                    primitives++;                                              \
                }                                                              \
            } else if (mode == GL_TRIANGLE_STRIP_ADJACENCY) {                  \
                uint32_t tri = 0u;                                             \
                for (uint32_t q = 0u; q + 5u < segmentCount; q += 2u, tri++) { \
                    uint32_t last = (q + 6u >= segmentCount);                  \
                    if (tri == 0u) {                                           \
                        EMIT(segment[q]);                                      \
                        EMIT(segment[q + 1u]);                                 \
                        EMIT(segment[q + 2u]);                                 \
                        EMIT(last ? segment[q + 5u] : segment[q + 6u]);        \
                        EMIT(segment[q + 4u]);                                 \
                        EMIT(segment[q + 3u]);                                 \
                    } else if (tri & 1u) {                                     \
                        EMIT(segment[q + 2u]);                                 \
                        EMIT(segment[q - 2u]);                                 \
                        EMIT(segment[q]);                                      \
                        EMIT(segment[q + 3u]);                                 \
                        EMIT(segment[q + 4u]);                                 \
                        EMIT(last ? segment[q + 5u] : segment[q + 6u]);        \
                    } else {                                                   \
                        EMIT(segment[q]);                                      \
                        EMIT(segment[q - 2u]);                                 \
                        EMIT(segment[q + 2u]);                                 \
                        EMIT(last ? segment[q + 5u] : segment[q + 6u]);        \
                        EMIT(segment[q + 4u]);                                 \
                        EMIT(segment[q + 3u]);                                 \
                    }                                                          \
                    primitives++;                                              \
                }                                                              \
            } else if (mode == GL_LINE_STRIP) {                                \
                for (uint32_t q = 0u; q + 1u < segmentCount; q++) {            \
                    EMIT(segment[q]);                                          \
                    EMIT(segment[q + 1u]);                                     \
                    primitives++;                                              \
                }                                                              \
            } else if (mode == GL_LINE_LOOP) {                                 \
                if (segmentCount >= 2u) {                                      \
                    for (uint32_t q = 0u; q + 1u < segmentCount; q++) {        \
                        EMIT(segment[q]);                                      \
                        EMIT(segment[q + 1u]);                                 \
                        primitives++;                                          \
                    }                                                          \
                    EMIT(segment[segmentCount - 1u]);                          \
                    EMIT(segment[0u]);                                         \
                    primitives++;                                              \
                }                                                              \
            } else if (mode == GL_TRIANGLE_STRIP) {                            \
                for (uint32_t q = 0u; q + 2u < segmentCount; q++) {            \
                    if (q & 1u) {                                              \
                        EMIT(segment[q + 1u]);                                 \
                        EMIT(segment[q]);                                      \
                    } else {                                                   \
                        EMIT(segment[q]);                                      \
                        EMIT(segment[q + 1u]);                                 \
                    }                                                          \
                    EMIT(segment[q + 2u]);                                     \
                    primitives++;                                              \
                }                                                              \
            } else if (mode == GL_TRIANGLE_FAN) {                              \
                for (uint32_t q = 1u; q + 1u < segmentCount; q++) {            \
                    EMIT(segment[0u]);                                         \
                    EMIT(segment[q]);                                          \
                    EMIT(segment[q + 1u]);                                     \
                    primitives++;                                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        segmentCount = 0u;                                                     \
    } while (0)

    for (uint32_t i = 0u; i < n; i++) {
        if (restartMode && source[i] == restartIndex) {
            EMIT_SEGMENT();
        } else {
            segment[segmentCount++] = source[i];
        }
    }
    EMIT_SEGMENT();
#undef EMIT_SEGMENT
#undef EMIT
    free(source);
    free(segment);
    if (gathered == 0u || primitives == 0u) {
        free(gather);
        return false;
    }
    *outGather = gather;
    *outGatherCount = gathered;
    *outPrimitiveCount = primitives;
    *outMaxIndex = maxIndex;
    return true;
}

extern "C" void mglDrawGsEncodePassthrough(const MGLGsPassthroughEncodeState *state)
{
    if (!state || !state->encoder_owner || !state->output_buffer ||
        !state->counts_buffer || state->work_item_count == 0u ||
        state->output_stride == 0u) {
        return;
    }
    const uint64_t headerBytes =
        (uint64_t)MGL_AIR_GS_HEADER_RECORDS * (uint64_t)state->output_stride;
    for (uint32_t primitive = 0u; primitive < state->work_item_count;
         primitive++) {
        const uint64_t offset =
            ((uint64_t)primitive * (uint64_t)state->records_per_primitive) *
                (uint64_t)state->output_stride +
            headerBytes;
        (void)mglRenderSetRenderBufferForOwner(
            state->encoder_owner, state->output_buffer, offset,
            MGL_RENDER_BINDING_STAGE_VERTEX, 0u);
        MGLRenderDrawPlan plan = {};
        plan.kind = MGL_RENDER_DRAW_ARRAY_INDIRECT;
        plan.primitive_type = state->output_primitive;
        plan.indirect_buffer = state->counts_buffer;
        plan.indirect_buffer_offset =
            (uint64_t)primitive * (uint64_t)state->counts_record_bytes;
        (void)mglRenderEncodeDrawForRenderEncoderOwner(
            state->encoder_owner, &plan, NULL, 0);
    }
}
