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
#include "mgl_shader_abi.h"

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

extern "C" void mglDrawGsNormalizeTopology(Program *gs, GLenum *in_mode,
                                           GLenum *out_mode,
                                           uint32_t *out_primitive)
{
    GLenum in = gs ? gs->geometry_input_type : GL_TRIANGLES;
    GLenum out = gs ? gs->geometry_output_type : GL_TRIANGLE_STRIP;
    if (in != GL_POINTS && in != GL_LINES && in != GL_LINES_ADJACENCY &&
        in != GL_TRIANGLES && in != GL_TRIANGLES_ADJACENCY) {
        in = GL_TRIANGLES;
    }
    if (out != GL_POINTS && out != GL_LINE_STRIP && out != GL_TRIANGLE_STRIP) {
        out = GL_TRIANGLE_STRIP;
    }
    if (in_mode) {
        *in_mode = in;
    }
    if (out_mode) {
        *out_mode = out;
    }
    if (out_primitive) {
        *out_primitive = out == GL_POINTS
                             ? MGL_DRAW_PRIMITIVE_POINT
                             : out == GL_LINE_STRIP ? MGL_DRAW_PRIMITIVE_LINE
                                                    : MGL_DRAW_PRIMITIVE_TRIANGLE;
    }
}

extern "C" void mglDrawGsFillGatherParams(int indexed, uint32_t count,
                                          uint32_t first,
                                          uint32_t gather_max_index,
                                          uint32_t primitives,
                                          MGLAIRGSGatherParams *out)
{
    if (!out) {
        return;
    }
    out->vertices_per_instance =
        indexed ? gather_max_index + 1u : count;
    out->primitives_per_instance = primitives;
    out->first_vertex = indexed ? 0u : first;
    out->gather_enabled = 1u;
}

extern "C" void mglDrawGsPlanInputSource(int pending_active, int has_pending,
                                         uint32_t pending_offset,
                                         uint32_t pending_stride, int indexed,
                                         MGLGsInputSourcePlan *out)
{
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (pending_active && has_pending) {
        out->kind = MGL_GS_INPUT_PENDING_TES;
        out->input_offset = pending_offset;
        out->pending_stride = pending_stride;
        return;
    }
    out->kind = indexed ? MGL_GS_INPUT_CAPTURE_INDEXED
                        : MGL_GS_INPUT_CAPTURE_ARRAY;
}

extern "C" uint32_t mglDrawGsMaxVerticesOut(uint32_t geometry_vertices_out)
{
    return geometry_vertices_out > 0u ? geometry_vertices_out : 1u;
}

extern "C" uint32_t mglDrawGsResolveStageInStride(Program *vs, Program *tes,
                                                  uint32_t pending_stride)
{
    if (pending_stride > 0u) {
        return pending_stride;
    }
    if (tes) {
        return mglAIRPerVertexStrideForResources(
            &tes->shader_resources_list[_TESS_EVALUATION_SHADER]
                                       [_STAGE_OUTPUT_RES]);
    }
    if (vs) {
        return mglAIRPerVertexStrideForResources(
            &vs->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    }
    return 0u;
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

static bool mglDrawGsPlanAppendBuffer(MGLRenderComputeExecutionPlan *plan,
                                      void *buffer, uint64_t offset,
                                      uint32_t index)
{
    if (!plan || !buffer) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        return false;
    }
    plan->binding_ops[plan->binding_op_count++] = {
        .kind = 0u,
        .index = index,
        .offset = offset,
        .buffer = buffer,
        .bytes = NULL,
        .length = 0u,
    };
    return true;
}

extern "C" void mglDrawGsFillLocationMap(Program *gs, Program *vs, Program *tes,
                                         uint32_t loc_map[32])
{
    if (!loc_map) {
        return;
    }
    memset(loc_map, 0, sizeof(uint32_t) * 32u);
    if (!gs) {
        return;
    }
    const MGLShaderResourceList *gsInputs =
        &gs->shader_resources_list[_GEOMETRY_SHADER][_STAGE_INPUT_RES];
    const MGLShaderResourceList *vsOutputs = NULL;
    if (tes) {
        vsOutputs = &tes->shader_resources_list[_TESS_EVALUATION_SHADER]
                                               [_STAGE_OUTPUT_RES];
    } else if (vs) {
        vsOutputs = &vs->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES];
    }
    if (!gsInputs || !vsOutputs || !gsInputs->list || !vsOutputs->list) {
        return;
    }
    for (GLuint gi = 0u; gi < gsInputs->count; gi++) {
        const MGLShaderResource *in = &gsInputs->list[gi];
        if (in->is_per_patch || !in->name || in->location >= 32u) {
            continue;
        }
        GLuint nameLen = (GLuint)strlen(in->name);
        const char *bracket = strchr(in->name, '[');
        if (bracket) {
            nameLen = (GLuint)(bracket - in->name);
        }
        for (GLuint vi = 0u; vi < vsOutputs->count; vi++) {
            const MGLShaderResource *out = &vsOutputs->list[vi];
            if (out->is_per_patch || !out->name) {
                continue;
            }
            GLuint outLen = (GLuint)strlen(out->name);
            const char *ob = strchr(out->name, '[');
            if (ob) {
                outLen = (GLuint)(ob - out->name);
            }
            if (nameLen == outLen && strncmp(in->name, out->name, nameLen) == 0) {
                loc_map[in->location] = out->location + 1u;
                break;
            }
        }
    }
}

extern "C" void mglDrawGsPresetCounts(void *counts, uint32_t work_item_count)
{
    if (!counts || work_item_count == 0u) {
        return;
    }
    uint32_t *countsWords = (uint32_t *)counts;
    for (uint32_t w = 0u; w < work_item_count; w++) {
        countsWords[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 1u] = 1u;
    }
}

extern "C" uint32_t mglDrawGsFillXFBScatterParams(Program *gs,
                                                 MGLAIRGSXFBScatterParams *out)
{
    if (!out) {
        return 0u;
    }
    memset(out, 0, sizeof(*out));
    for (uint32_t b = 0u; b < MGL_AIR_GS_MAX_STREAMS; b++) {
        out->buffer_stream[b] = MGL_AIR_GS_XFB_NO_STREAM;
    }
    if (!gs) {
        return 0u;
    }
    uint32_t fieldCount = 0u;
    uint32_t xfbBufferCount = 0u;
    for (uint32_t s = 0u; s < MGL_AIR_GS_MAX_STREAMS; s++) {
        for (GLsizei vi = 0;
             vi < gs->transform_feedback_varying_count &&
             fieldCount < MGL_AIR_GS_XFB_MAX_FIELDS;
             vi++) {
            const MGLTransformFeedbackVaryingPlan *plan =
                &gs->transform_feedback_layout[vi];
            if (plan->component_count == 0u) {
                continue;
            }
            if ((uint32_t)plan->stream != s) {
                continue;
            }
            if (plan->buffer_index >= MGL_AIR_GS_MAX_STREAMS) {
                continue;
            }
            const char *name = gs->transform_feedback_varying_names[vi];
            if (!name || !name[0]) {
                continue;
            }
            char baseName[96];
            strncpy(baseName, name, sizeof(baseName) - 1u);
            baseName[sizeof(baseName) - 1u] = '\0';
            char *bracket = strchr(baseName, '[');
            if (bracket) {
                *bracket = '\0';
            }
            GLuint location = UINT32_MAX;
            MGLShaderResource *gsOut = mglProgramFindStageOutputForXFBName(
                gs, _GEOMETRY_SHADER, name);
            if (gsOut) {
                location = gsOut->location;
            }
            uint32_t srcOffset;
            if (strcmp(baseName, "gl_Position") == 0 && plan->builtin) {
                srcOffset = MGL_AIR_PER_VERTEX_POSITION_OFFSET;
            } else if (strcmp(baseName, "gl_PointSize") == 0 && plan->builtin) {
                srcOffset = MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET;
            } else {
                if (location == UINT32_MAX) {
                    continue;
                }
                srcOffset = MGL_AIR_PER_VERTEX_STRIDE + location * 16u;
            }
            MGLAIRGSXFBFieldDesc *fd = &out->fields[fieldCount++];
            fd->buffer_index = plan->buffer_index;
            fd->src_offset = srcOffset;
            fd->dst_offset = plan->component_offset * 4u;
            fd->byte_count = plan->component_count * 4u;
            out->buffer_stream[plan->buffer_index] = s;
            if (plan->buffer_index + 1u > xfbBufferCount) {
                xfbBufferCount = plan->buffer_index + 1u;
            }
        }
    }
    out->field_count = fieldCount;
    for (uint32_t f = 0u; f < fieldCount; f++) {
        const MGLAIRGSXFBFieldDesc *fd = &out->fields[f];
        const uint32_t end = fd->dst_offset + fd->byte_count;
        if (end > out->buffers[fd->buffer_index].stride) {
            out->buffers[fd->buffer_index].stride = end;
        }
    }
    return xfbBufferCount;
}

extern "C" bool mglDrawGsAppendCoreBindings(
    MGLRenderComputeExecutionPlan *plan, void *input, uint64_t input_offset,
    void *output, void *counts, void *gather_or_counts, void *xfb_capture,
    void *xfb_meta, void *xfb_vis_or_counts, const void *gparams,
    uint32_t gparams_bytes)
{
    if (!plan || !input || !output || !counts || !gather_or_counts ||
        !xfb_meta || !xfb_vis_or_counts || !gparams || gparams_bytes == 0u) {
        return false;
    }
    if (!mglDrawGsPlanAppendBuffer(plan, input, input_offset,
                                   MGL_AIR_GS_SLOT_INPUT) ||
        !mglDrawGsPlanAppendBuffer(plan, output, 0u, MGL_AIR_GS_SLOT_OUTPUT) ||
        !mglDrawGsPlanAppendBuffer(plan, counts, 0u, MGL_AIR_GS_SLOT_COUNTS) ||
        !mglDrawGsPlanAppendBuffer(plan, gather_or_counts, 0u,
                                   MGL_AIR_GS_SLOT_GATHER)) {
        return false;
    }
    if (xfb_capture &&
        !mglDrawGsPlanAppendBuffer(plan, xfb_capture, 0u, MGL_AIR_GS_SLOT_XFB)) {
        return false;
    }
    if (!mglDrawGsPlanAppendBuffer(plan, xfb_meta, 0u, MGL_AIR_GS_SLOT_XFB_META) ||
        !mglDrawGsPlanAppendBuffer(plan, xfb_vis_or_counts, 0u,
                                   MGL_AIR_GS_SLOT_XFB_VIS)) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        return false;
    }
    plan->binding_ops[plan->binding_op_count++] = {
        .kind = 1u,
        .index = MGL_AIR_GS_SLOT_GATHER_PARAMS,
        .offset = 0u,
        .buffer = NULL,
        .bytes = gparams,
        .length = gparams_bytes,
    };
    return true;
}

static bool mglDrawGsPlanAppendBytes(MGLRenderComputeExecutionPlan *plan,
                                     const void *bytes, uint32_t length,
                                     uint32_t index)
{
    if (!plan || !bytes || length == 0u) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        return false;
    }
    plan->binding_ops[plan->binding_op_count++] = {
        .kind = 1u,
        .index = index,
        .offset = 0u,
        .buffer = NULL,
        .bytes = bytes,
        .length = length,
    };
    return true;
}

extern "C" bool mglDrawGsComputeLayout(Program *gs, uint32_t primitive_count,
                                       uint32_t instance_count,
                                       GLenum output_mode,
                                       MGLGsComputeLayout *out)
{
    if (!gs || !out || primitive_count == 0u || instance_count == 0u) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    const uint32_t invocations =
        gs->geometry_invocations > 0u ? gs->geometry_invocations : 1u;
    if (instance_count > UINT32_MAX / primitive_count) {
        return false;
    }
    const uint32_t draw_primitives = primitive_count * instance_count;
    if (draw_primitives > UINT32_MAX / invocations) {
        return false;
    }
    const uint32_t work_items = draw_primitives * invocations;
    const uint32_t output_stride = mglAIRPerVertexStrideForResources(
        &gs->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES]);
    if (output_stride == 0u) {
        return false;
    }
    const uint32_t max_vertices =
        gs->geometry_vertices_out > 0u ? gs->geometry_vertices_out : 1u;
    const MGLAIRGSOutputPrimitive air_out =
        output_mode == GL_POINTS
            ? MGL_AIR_GS_OUT_POINTS
            : output_mode == GL_LINE_STRIP ? MGL_AIR_GS_OUT_LINE_STRIP
                                           : MGL_AIR_GS_OUT_TRIANGLE_STRIP;
    const uint32_t expanded = mglAIRGSExpandedVertices(air_out, max_vertices);
    const uint32_t records = mglAIRGSRecordsPerPrimitive(air_out, max_vertices);
    uint64_t per_item = 0u;
    if (__builtin_mul_overflow((uint64_t)records, (uint64_t)output_stride,
                               &per_item) || per_item == 0u) {
        return false;
    }
    uint64_t output_bytes = 0u;
    uint64_t counts_bytes = 0u;
    if (__builtin_mul_overflow((uint64_t)work_items, per_item,
                               &output_bytes) ||
        __builtin_mul_overflow((uint64_t)work_items,
                               (uint64_t)MGL_AIR_GS_COUNTS_RECORD_BYTES,
                               &counts_bytes)) {
        return false;
    }
    out->work_item_count = work_items;
    out->records_per_primitive = records;
    out->expanded_vertices = expanded;
    out->output_stride = output_stride;
    out->output_bytes = output_bytes;
    out->counts_bytes = counts_bytes;
    return true;
}

extern "C" void mglDrawGsExclusivePrefixSum(const uint32_t *vis,
                                            uint32_t *offsets,
                                            uint32_t work_item_count,
                                            uint32_t buffer_count)
{
    if (!vis || !offsets || work_item_count == 0u || buffer_count == 0u) {
        return;
    }
    if (buffer_count > MGL_AIR_GS_MAX_STREAMS) {
        buffer_count = MGL_AIR_GS_MAX_STREAMS;
    }
    for (uint32_t b = 0u; b < buffer_count; b++) {
        uint32_t running = 0u;
        for (uint32_t w = 0u; w < work_item_count; w++) {
            const uint32_t idx = w * MGL_AIR_GS_MAX_STREAMS + b;
            offsets[idx] = running;
            running += vis[idx];
        }
    }
}

extern "C" bool mglDrawGsFillXFBScatterPlan(
    MGLRenderComputeExecutionPlan *plan, void *pipeline,
    const void *scatter_params, uint32_t params_bytes, void *vis, void *offsets,
    void *stage_out, void *xfb, void *written, uint32_t work_item_count)
{
    if (!plan || !pipeline || !scatter_params || params_bytes == 0u || !vis ||
        !offsets || !stage_out || !xfb || !written || work_item_count == 0u) {
        return false;
    }
    memset(plan, 0, sizeof(*plan));
    plan->pipeline = pipeline;
    if (!mglDrawGsPlanAppendBytes(plan, scatter_params, params_bytes,
                                  MGL_AIR_GS_XFB_SCATTER_PARAMS_SLOT) ||
        !mglDrawGsPlanAppendBuffer(plan, vis, 0u,
                                   MGL_AIR_GS_XFB_SCATTER_VIS_SLOT) ||
        !mglDrawGsPlanAppendBuffer(plan, offsets, 0u,
                                   MGL_AIR_GS_XFB_SCATTER_OFFSET_SLOT) ||
        !mglDrawGsPlanAppendBuffer(plan, stage_out, 0u,
                                   MGL_AIR_GS_XFB_SCATTER_STAGE_OUT_SLOT) ||
        !mglDrawGsPlanAppendBuffer(plan, xfb, 0u,
                                   MGL_AIR_GS_XFB_SCATTER_XFB_SLOT) ||
        !mglDrawGsPlanAppendBuffer(plan, written, 0u,
                                   MGL_AIR_GS_XFB_SCATTER_WRITTEN_SLOT)) {
        return false;
    }
    plan->dispatch = {
        .dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT,
        .groups_x = work_item_count,
        .groups_y = 1u,
        .groups_z = 1u,
        .local_x = 1u,
        .local_y = 1u,
        .local_z = 1u,
    };
    plan->barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
    return true;
}

extern "C" void mglDrawGsPlanXFBDestinations(
    MGLAIRGSXFBScatterParams *params, uint32_t buffer_count,
    uint32_t work_item_count, uint32_t expanded_vertices,
    const MGLGsXFBBufferBinding bindings[MGL_AIR_GS_MAX_STREAMS],
    MGLGsXFBDestPlan *out)
{
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    if (!params || !bindings || !out || buffer_count == 0u ||
        work_item_count == 0u) {
        return;
    }
    if (buffer_count > MGL_AIR_GS_MAX_STREAMS) {
        buffer_count = MGL_AIR_GS_MAX_STREAMS;
    }
    uint64_t phys_total = 0u;
    for (uint32_t b = 0u; b < buffer_count; b++) {
        if (params->buffers[b].stride == 0u || !bindings[b].bound) {
            continue;
        }
        if (bindings[b].session_offset > bindings[b].visible_bytes ||
            bindings[b].slot_offset < 0 ||
            (uint64_t)bindings[b].slot_offset >
                UINT64_MAX - bindings[b].session_offset) {
            continue;
        }
        const uint64_t remaining =
            bindings[b].visible_bytes - bindings[b].session_offset;
        uint64_t max_cap = 0u;
        if (__builtin_mul_overflow((uint64_t)work_item_count,
                                   (uint64_t)expanded_vertices, &max_cap) ||
            __builtin_mul_overflow(max_cap, (uint64_t)params->buffers[b].stride,
                                   &max_cap)) {
            max_cap = UINT32_MAX;
        }
        uint64_t cap = remaining < max_cap ? remaining : max_cap;
        if (cap > UINT32_MAX) {
            cap = UINT32_MAX;
        }
        out->buffers[b].remaining = remaining > UINT32_MAX
                                        ? UINT32_MAX
                                        : (uint32_t)remaining;
        out->buffers[b].dst_offset =
            (uint32_t)((uint64_t)bindings[b].slot_offset +
                       bindings[b].session_offset);
        out->buffers[b].cap_bytes = (uint32_t)cap;
        out->buffers[b].phys_base =
            phys_total > UINT32_MAX ? UINT32_MAX : (uint32_t)phys_total;
        out->buffers[b].valid = 1u;
        params->buffers[b].capacity_bytes = out->buffers[b].cap_bytes;
        params->buffers[b].capture_base = out->buffers[b].phys_base;
        phys_total += cap;
        if (phys_total > UINT32_MAX) {
            phys_total = UINT32_MAX;
        }
    }
    out->phys_total = (uint32_t)phys_total;
}

extern "C" void mglDrawGsFillXFBScatterRuntime(
    MGLAIRGSXFBScatterParams *params, uint32_t buffer_count,
    uint32_t work_item_count, uint32_t output_stride,
    uint32_t records_per_primitive, uint32_t output_primitive)
{
    if (!params) {
        return;
    }
    params->buffer_count = buffer_count;
    params->work_item_count = work_item_count;
    params->stage_out_stride = output_stride;
    params->records_per_primitive = records_per_primitive;
    params->vertices_per_primitive =
        mglDrawGsVerticesPerPrimitive(output_primitive);
    params->expanded_offset_records = MGL_AIR_GS_HEADER_RECORDS;
}

extern "C" uint32_t mglDrawGsVerticesPerPrimitive(uint32_t output_primitive)
{
    if (output_primitive == MGL_DRAW_PRIMITIVE_POINT) {
        return 1u;
    }
    if (output_primitive == MGL_DRAW_PRIMITIVE_LINE) {
        return 2u;
    }
    return 3u;
}

extern "C" uint32_t mglDrawGsStreamCount(uint32_t geometry_stream_count)
{
    return geometry_stream_count > 0u ? geometry_stream_count : 1u;
}

extern "C" GLenum mglDrawGsLastDrawMode(uint32_t output_primitive)
{
    if (output_primitive == MGL_DRAW_PRIMITIVE_POINT) {
        return GL_POINTS;
    }
    if (output_primitive == MGL_DRAW_PRIMITIVE_LINE) {
        return GL_LINES;
    }
    return GL_TRIANGLES;
}

extern "C" void mglDrawGsFillXFBDestForMeta(const uint32_t *cap_bytes,
                                            const uint32_t *phys_base,
                                            uint32_t count,
                                            MGLGsXFBDestPlan *out)
{
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (!cap_bytes || !phys_base) {
        return;
    }
    if (count > MGL_AIR_GS_MAX_STREAMS) {
        count = MGL_AIR_GS_MAX_STREAMS;
    }
    for (uint32_t b = 0u; b < count; b++) {
        out->buffers[b].cap_bytes = cap_bytes[b];
        out->buffers[b].phys_base = phys_base[b];
        out->buffers[b].valid = cap_bytes[b] > 0u ? 1u : 0u;
    }
}

extern "C" void mglDrawGsClearXFBMetaIfNoCapture(int has_capture,
                                                 MGLAIRGSXFBMeta *meta)
{
    if (has_capture || !meta) {
        return;
    }
    for (uint32_t s = 0u; s < MGL_AIR_GS_MAX_STREAMS; s++) {
        meta->stream[s].stride = 0u;
    }
}

extern "C" int mglDrawGsNeedCPUVisibility(int xfb_active, int has_query)
{
    return xfb_active || has_query ? 1 : 0;
}

extern "C" uint64_t mglDrawGsQueryWritten(uint32_t output_primitive,
                                          uint32_t buffer0_stride,
                                          uint64_t buffer0_written)
{
    const uint64_t prim_bytes =
        (uint64_t)mglDrawGsVerticesPerPrimitive(output_primitive) *
        (uint64_t)buffer0_stride;
    return prim_bytes > 0u ? buffer0_written / prim_bytes : 0u;
}

extern "C" int mglDrawGsSkipRaster(int xfb_active, int rasterizer_discard)
{
    return xfb_active && rasterizer_discard ? 1 : 0;
}

extern "C" uint64_t mglDrawGsXFBVisBytes(uint32_t work_item_count)
{
    uint64_t n = 0u;
    if (__builtin_mul_overflow((uint64_t)work_item_count,
                               (uint64_t)MGL_AIR_GS_MAX_STREAMS, &n) ||
        __builtin_mul_overflow(n, (uint64_t)sizeof(uint32_t), &n)) {
        return 0u;
    }
    return n;
}

extern "C" int mglDrawGsXFBActive(int has_xfb, int active, int paused)
{
    return has_xfb && active && !paused ? 1 : 0;
}

extern "C" void mglDrawGsFillXFBMetaFromDest(
    const MGLAIRGSXFBScatterParams *params, const MGLGsXFBDestPlan *dest,
    MGLAIRGSXFBMeta *out)
{
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (!params || !dest) {
        return;
    }
    for (uint32_t s = 0u; s < MGL_AIR_GS_MAX_STREAMS; s++) {
        const int capture = dest->buffers[s].valid &&
                            dest->buffers[s].cap_bytes > 0u &&
                            params->buffers[s].stride > 0u;
        out->stream[s].stride = capture ? params->buffers[s].stride : 0u;
        out->stream[s].capacity_bytes = dest->buffers[s].cap_bytes;
        out->stream[s].capture_base = dest->buffers[s].phys_base;
        out->buffer_stream[s] = params->buffer_stream[s];
    }
}

extern "C" uint64_t mglDrawGsReduceGeneratedPrimitives(
    GLenum output_mode, uint32_t work_item_count, uint32_t max_vertices,
    const uint32_t *counts, const MGLAIRGSXFBMeta *meta)
{
    if (work_item_count == 0u) {
        return 0u;
    }
    const uint32_t max_per =
        output_mode == GL_POINTS
            ? max_vertices
            : output_mode == GL_LINE_STRIP
                  ? (max_vertices > 1u ? max_vertices - 1u : 0u)
                  : (max_vertices > 2u ? max_vertices - 2u : 0u);
    const uint64_t max_generated =
        (uint64_t)work_item_count * (uint64_t)max_per;
    if (meta) {
        const uint64_t meta_gen = (uint64_t)meta->stream[0].generated;
        if (meta_gen <= max_generated) {
            return meta_gen;
        }
    }
    if (!counts) {
        return 0u;
    }
    const uint32_t emit_word =
        MGL_AIR_GS_COUNTS_ARGS_WORDS + (uint32_t)(MGL_AIR_GS_COUNT_EMITTED - 1u);
    const uint32_t vpp = output_mode == GL_POINTS
                             ? 1u
                             : output_mode == GL_LINE_STRIP ? 2u : 3u;
    uint64_t emit_sum = 0u;
    uint64_t vertex_sum = 0u;
    for (uint32_t w = 0u; w < work_item_count; w++) {
        const uint32_t *row = counts + w * MGL_AIR_GS_COUNTS_RECORD_WORDS;
        emit_sum += row[emit_word];
        vertex_sum += row[0];
    }
    if (output_mode == GL_POINTS) {
        return emit_sum <= max_generated ? emit_sum : 0u;
    }
    const uint64_t from_verts = vpp ? vertex_sum / vpp : 0u;
    return from_verts <= max_generated ? from_verts : 0u;
}

extern "C" uint64_t mglDrawGsReduceBufferWritten(const uint32_t *written,
                                                 uint32_t work_item_count,
                                                 uint32_t buffer_index)
{
    if (!written || work_item_count == 0u ||
        buffer_index >= MGL_AIR_GS_MAX_STREAMS) {
        return 0u;
    }
    uint64_t total = 0u;
    for (uint32_t w = 0u; w < work_item_count; w++) {
        total += written[w * MGL_AIR_GS_MAX_STREAMS + buffer_index];
    }
    return total;
}
