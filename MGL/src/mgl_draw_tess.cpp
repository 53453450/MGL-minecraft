/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_draw_tess.h"

#include "error.h"
#include "glcorearb.h"
#include "glm_limits.h"
#include "mgl_draw_encode.h"
#include "mgl_index_buffer.h"
#include "mgl_program_resource.h"
#include "mgl_render.h"
#include "mgl_renderer_backend.h"
#include "mgl_shader_abi.h"
#include "mgl_shader_resource.h"
#include "mgl_buffer_slots.h"
#include "mgl_env_flag.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

extern "C" MGLTessDrawClass mglTessClassifyDraw(GLMContext ctx, GLenum mode,
                                                GLsizei count,
                                                GLsizei instanceCount,
                                                Program *tcs, Program *tes,
                                                const char *label)
{
    if (mode != GL_PATCHES) {
        return MGL_TESS_DRAW_NOT_APPLICABLE;
    }
    if (!ctx || count <= 0) {
        return MGL_TESS_DRAW_NOOP_HANDLED;
    }
    if (tcs && !tcs->shader_slots[_TESS_CONTROL_SHADER]) {
        tcs = NULL;
    }
    if (tes && !tes->shader_slots[_TESS_EVALUATION_SHADER]) {
        tes = NULL;
    }
    if (!tcs && !tes) {
        return MGL_TESS_DRAW_NOT_APPLICABLE;
    }
    if (tcs && !tes) {
        mglDispatchError(ctx, label ? label : "tessellationDraw",
                         GL_INVALID_OPERATION);
        return MGL_TESS_DRAW_NOOP_HANDLED;
    }
    if (instanceCount <= 0) {
        return MGL_TESS_DRAW_NOOP_HANDLED;
    }
    if (!ctx->active_state) {
        return MGL_TESS_DRAW_NOOP_HANDLED;
    }
    const GLuint patchVertices =
        MAX(1u, (GLuint)ctx->active_state->var.patch_vertices);
    const GLuint patchCount = (GLuint)count / patchVertices;
    if (patchCount == 0u) {
        return MGL_TESS_DRAW_NOOP_HANDLED;
    }
    return MGL_TESS_DRAW_ACTIVE;
}

extern "C" void mglTessFillDrawContract(MGLAIRTessDrawContract *contract,
                                        GLMContext ctx, Program *tcs,
                                        Program *tes, Program *vs, GLint first,
                                        GLsizei count, GLenum indexType,
                                        const void *indices, GLint baseVertex,
                                        GLsizei instanceCount,
                                        GLuint baseInstance)
{
    if (!contract) {
        return;
    }
    memset(contract, 0, sizeof(*contract));
    if (!ctx || !ctx->active_state) {
        return;
    }
    if (tcs && !tcs->shader_slots[_TESS_CONTROL_SHADER]) {
        tcs = NULL;
    }
    if (tes && !tes->shader_slots[_TESS_EVALUATION_SHADER]) {
        tes = NULL;
    }

    const GLuint patchVertices =
        MAX(1u, (GLuint)ctx->active_state->var.patch_vertices);
    const GLuint patchCount = (GLuint)count / patchVertices;
    uint32_t restartIndex = 0u;
    const bool restartEnabled =
        indexType != 0u &&
        mglPrimitiveRestartIndexForType(ctx, indexType, &restartIndex);

    contract->patch_vertices = patchVertices;
    contract->vertex_count = (uint32_t)count;
    contract->patch_count = patchCount;
    contract->instance_count =
        instanceCount > 0 ? (uint32_t)instanceCount : 1u;
    contract->base_instance = baseInstance;
    contract->first = first;
    contract->index_type = indexType;
    contract->index_source = (uint64_t)(uintptr_t)indices;
    contract->index_count = indexType != 0u ? (uint64_t)count : 0u;
    contract->base_vertex = baseVertex;
    contract->primitive_restart = restartEnabled ? 1u : 0u;
    contract->restart_index = restartIndex;
    contract->tess_factor_bytes_per_patch = MGL_AIR_TESS_FACTOR_RECORD_BYTES;
    contract->tess_gen_mode =
        tes ? (uint32_t)tes->tess_gen_mode : (uint32_t)GL_TRIANGLES;
    contract->point_mode = tes ? (uint32_t)tes->tess_gen_point_mode : 0u;
    contract->tcs_out_vertices =
        tcs && tcs->tess_control_output_vertices > 0u
            ? tcs->tess_control_output_vertices
            : patchVertices;
    contract->per_vertex_out_stride =
        vs ? mglAIRPerVertexStrideForResources(
                 &vs->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES])
           : MGL_AIR_PER_VERTEX_STRIDE;
    contract->patch_out_stride = 16u;
}

extern "C" bool mglTessNativeInterfaceSupported(Program *tcs, Program *tes)
{
    if (!tes) {
        return false;
    }
    return mglRenderNativeTESInterfaceSupported(
               tes->modules[_TESS_EVALUATION_SHADER].mtl_function,
               (uint64_t)tes->modules[_TESS_EVALUATION_SHADER].metallib_bytes,
               (uint32_t)tes->tess_gen_point_mode,
               (uint32_t)tes->transform_feedback_varying_count,
               (uint32_t)tes->tess_gen_mode,
               tcs ? tcs->modules[_TESS_CONTROL_SHADER].mtl_function : NULL,
               tcs ? (uint64_t)tcs->modules[_TESS_CONTROL_SHADER].metallib_bytes
                   : 0u,
               tcs ? (uint32_t)tcs->tess_control_output_vertices : 0u) != 0;
}

extern "C" bool mglTessNativeBlockedByGeometry(Program *gs)
{
    return gs && (gs->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT) &&
           gs->shader_slots[_GEOMETRY_SHADER];
}

extern "C" bool mglTessPlanDrawPath(GLMContext ctx, GLenum mode, GLsizei count,
                                    GLsizei instanceCount, Program *tcs,
                                    Program *tes, Program *gs, GLenum indexType,
                                    const char *label, MGLTessDrawPathPlan *out)
{
    if (!out) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->classify =
        mglTessClassifyDraw(ctx, mode, count, instanceCount, tcs, tes, label);
    if (out->classify != MGL_TESS_DRAW_ACTIVE) {
        return true;
    }
    if (tcs && !tcs->shader_slots[_TESS_CONTROL_SHADER]) {
        tcs = NULL;
    }
    if (tes && !tes->shader_slots[_TESS_EVALUATION_SHADER]) {
        tes = NULL;
    }
    out->has_tcs = tcs ? 1u : 0u;
    out->has_tes = tes ? 1u : 0u;
    out->air_tes =
        tes && tes->modules[_TESS_EVALUATION_SHADER].metallib_bytes ? 1u : 0u;
    out->indexed = indexType != 0u ? 1u : 0u;
    out->native_ok =
        (mglTessNativeInterfaceSupported(tcs, tes) &&
         !mglTessNativeBlockedByGeometry(gs))
            ? 1u
            : 0u;
    if (out->native_ok || out->air_tes) {
        if (out->indexed && out->has_tcs) {
            out->capture = MGL_TESS_CAPTURE_INDEXED_COMPACT;
            out->native_ok = 0u;
        } else if (out->indexed) {
            out->capture = MGL_TESS_CAPTURE_INDEXED_GATHER;
        } else {
            out->capture = MGL_TESS_CAPTURE_ARRAY;
        }
    }
    out->need_default_factors =
        !out->has_tcs && (out->native_ok || out->air_tes) ? 1u : 0u;
    out->need_tcs = out->has_tcs;
    if (out->native_ok) {
        out->exec = MGL_TESS_EXEC_NATIVE;
    } else if (out->air_tes) {
        out->exec = tes && tes->tess_eval_compute
                        ? MGL_TESS_EXEC_TES_COMPUTE
                        : MGL_TESS_EXEC_UNSUPPORTED;
    } else if (out->has_tes) {
        out->exec = MGL_TESS_EXEC_TES_FALLBACK;
    }
    return true;
}

extern "C" void mglTessApplyGatherToContract(MGLAIRTessDrawContract *contract,
                                             uint32_t gather_count,
                                             uint32_t gather_primitives)
{
    if (!contract) {
        return;
    }
    contract->patch_count = gather_primitives;
    contract->vertex_count = gather_count;
}

extern "C" void mglTessEncodeNativePatches(const MGLTessNativeEncodeState *state)
{
    if (!state || !state->encoder_owner || !state->tcs_output_buffer ||
        !state->native_factors || state->patch_count == 0u ||
        state->instance_count == 0u) {
        return;
    }
    uint32_t tcsOut = state->tcs_out_vertices;
    if (tcsOut == 0u) {
        tcsOut = state->patch_vertices;
    }
    const uint64_t instanceStrideBytes =
        state->tess_instance_records * state->tcs_output_stride;
    const uint64_t nativeFactorStride =
        state->tess_gen_mode == GL_QUADS
            ? (uint64_t)MGL_AIR_TESS_FACTOR_RECORD_BYTES
            : (uint64_t)MGL_AIR_TESS_FACTOR_TRI_HALF_BYTES;
    const uint32_t factorInstanceStride =
        state->tess_gen_mode == GL_QUADS
            ? MGL_AIR_TESS_FACTOR_RECORD_BYTES
            : MGL_AIR_TESS_FACTOR_TRI_HALF_BYTES;
    (void)mglRenderSetTessellationFactorBufferForOwner(
        state->encoder_owner, state->native_factors, 0u, factorInstanceStride);

    for (uint32_t i = 0u; i < state->instance_count; i++) {
        const uint64_t instanceOffset =
            state->tess_vertex_capture_offset +
            (uint64_t)i * instanceStrideBytes;
        (void)mglRenderSetRenderBufferForOwner(
            state->encoder_owner, state->tcs_output_buffer, instanceOffset,
            MGL_RENDER_BINDING_STAGE_VERTEX, 0u);
        (void)mglRenderSetRenderBufferForOwner(
            state->encoder_owner, state->tcs_output_buffer, instanceOffset,
            MGL_RENDER_BINDING_STAGE_VERTEX, 30u);
        GLuint patchInfo[2] = {state->patch_vertices, tcsOut};
        (void)mglRenderSetRenderBytesForOwner(
            state->encoder_owner, patchInfo, sizeof(patchInfo),
            MGL_RENDER_BINDING_STAGE_VERTEX, 28u);
        const uint32_t perPatchNative =
            (state->tcs_patch_out_buffer != NULL) ? 1u : 0u;
        if (state->tess_indexed_draw) {
            MGLRenderDrawPlan plan = {};
            plan.kind = MGL_RENDER_DRAW_INDEXED_PATCHES;
            plan.primitive_type = (uint32_t)MGL_DRAW_PRIMITIVE_TRIANGLE;
            plan.control_point_count = tcsOut;
            plan.patch_start = 0u;
            plan.patch_count = state->patch_count;
            plan.control_point_index_buffer =
                state->control_point_index_buffer;
            plan.instance_count = 1u;
            plan.base_instance =
                (uint64_t)state->base_instance + (uint64_t)i;
            (void)mglRenderEncodeDrawForRenderEncoderOwner(
                state->encoder_owner, &plan, NULL, 0);
            continue;
        }
        const uint64_t cpcStride =
            (uint64_t)tcsOut * state->tcs_output_stride;
        for (uint32_t p = 0u; p < state->patch_count; p++) {
            const uint64_t patchOffset =
                instanceOffset + (uint64_t)p * cpcStride;
            (void)mglRenderSetRenderBufferForOwner(
                state->encoder_owner, state->tcs_output_buffer, patchOffset,
                MGL_RENDER_BINDING_STAGE_VERTEX, 0u);
            GLuint patchInfoWords[3] = {state->patch_vertices, tcsOut, p};
            (void)mglRenderSetRenderBytesForOwner(
                state->encoder_owner, patchInfoWords, sizeof(patchInfoWords),
                MGL_RENDER_BINDING_STAGE_VERTEX, 28u);
            if (perPatchNative) {
                (void)mglRenderSetRenderBufferForOwner(
                    state->encoder_owner, state->tcs_patch_out_buffer,
                    (uint64_t)p * (uint64_t)state->patch_out_stride,
                    MGL_RENDER_BINDING_STAGE_VERTEX, 27u);
                (void)mglRenderSetTessellationFactorBufferForOwner(
                    state->encoder_owner, state->native_factors,
                    (uint64_t)p * nativeFactorStride, 0u);
                MGLRenderDrawPlan plan = {};
                plan.kind = MGL_RENDER_DRAW_PATCHES;
                plan.primitive_type = (uint32_t)MGL_DRAW_PRIMITIVE_TRIANGLE;
                plan.control_point_count = tcsOut;
                plan.patch_start = 0u;
                plan.patch_count = 1u;
                plan.instance_count = 1u;
                plan.base_instance =
                    (uint64_t)state->base_instance + (uint64_t)i;
                (void)mglRenderEncodeDrawForRenderEncoderOwner(
                    state->encoder_owner, &plan, NULL, 0);
            } else {
                MGLRenderDrawPlan plan = {};
                plan.kind = MGL_RENDER_DRAW_PATCHES;
                plan.primitive_type = (uint32_t)MGL_DRAW_PRIMITIVE_TRIANGLE;
                plan.control_point_count = tcsOut;
                plan.patch_start = p;
                plan.patch_count = 1u;
                plan.instance_count = 1u;
                plan.base_instance =
                    (uint64_t)state->base_instance + (uint64_t)i;
                (void)mglRenderEncodeDrawForRenderEncoderOwner(
                    state->encoder_owner, &plan, NULL, 0);
            }
        }
    }
}

static bool mglTessPlanAppendBuffer(MGLRenderComputeExecutionPlan *plan,
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

extern "C" bool mglTessComputeTCSCoreLayout(
    Program *tcs, const MGLAIRTessDrawContract *contract,
    MGLTessTCSCoreLayout *out)
{
    if (!tcs || !contract || !out) {
        return false;
    }
    const uint32_t patchVertices = MAX(1u, contract->patch_vertices);
    uint32_t patchCountTC = contract->vertex_count / patchVertices;
    if (patchCountTC == 0u) {
        patchCountTC = 1u;
    }
    const uint32_t patchCount = MAX(1u, contract->patch_count);
    const uint32_t instanceCount = MAX(1u, contract->instance_count);
    uint32_t tcsOutVertices = tcs->tess_control_output_vertices;
    if (tcsOutVertices == 0u) {
        tcsOutVertices = patchVertices;
    }
    uint32_t outputStride = mglAIRPerVertexStrideForResources(
        &tcs->shader_resources_list[_TESS_CONTROL_SHADER][_STAGE_OUTPUT_RES]);
    uint32_t patchStride = mglAIRPatchVaryingStride(
        &tcs->shader_resources_list[_TESS_CONTROL_SHADER][_STAGE_OUTPUT_RES]);
    if (outputStride == 0u || patchStride == 0u) {
        return false;
    }
    if ((uint64_t)patchCountTC > UINT64_MAX / tcsOutVertices ||
        (uint64_t)patchCountTC * tcsOutVertices > UINT64_MAX / outputStride ||
        (uint64_t)patchCount > UINT64_MAX / patchStride ||
        (uint64_t)patchCount > UINT64_MAX / MGL_AIR_TESS_FACTOR_RECORD_BYTES) {
        return false;
    }
    *out = {};
    out->patch_vertices = patchVertices;
    out->patch_count = patchCount;
    out->instance_count = instanceCount;
    out->tcs_out_vertices = tcsOutVertices;
    out->output_stride = outputStride;
    out->patch_stride = patchStride;
    out->output_bytes =
        (uint64_t)patchCountTC * tcsOutVertices * outputStride;
    out->patch_out_bytes = (uint64_t)patchCountTC * patchStride;
    out->factor_bytes =
        (uint64_t)patchCount * MGL_AIR_TESS_FACTOR_RECORD_BYTES;
    return true;
}

extern "C" bool mglTessAppendTCSCoreBindings(
    MGLRenderComputeExecutionPlan *plan, void *output, void *patch_out,
    void *indirect, void *factors, void *stage_in, uint64_t stage_in_offset,
    const MGLTessTCSCoreLayout *layout)
{
    if (!plan || !output || !patch_out || !indirect || !factors || !stage_in ||
        !layout) {
        return false;
    }
    if (!mglTessPlanAppendBuffer(plan, output, 0u,
                                 MGL_AIR_TESS_SLOT_TCS_OUTPUT) ||
        !mglTessPlanAppendBuffer(plan, patch_out, 0u,
                                 MGL_AIR_TESS_SLOT_PATCH_OUT) ||
        !mglTessPlanAppendBuffer(plan, indirect, 0u,
                                 MGL_AIR_TESS_SLOT_INDIRECT) ||
        !mglTessPlanAppendBuffer(plan, factors, 0u,
                                 MGL_AIR_TESS_SLOT_TESS_FACTOR) ||
        !mglTessPlanAppendBuffer(plan, stage_in, stage_in_offset,
                                 MGL_AIR_TESS_SLOT_TCS_STAGE_IN)) {
        return false;
    }
    MGLRenderComputePlan dispatch = {};
    dispatch.dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT;
    dispatch.groups_x = layout->patch_count;
    dispatch.groups_y = 1u;
    dispatch.groups_z = 1u;
    dispatch.local_x = layout->tcs_out_vertices;
    dispatch.local_y = 1u;
    dispatch.local_z = 1u;
    return mglRenderAppendComputeDispatchToPlan(plan, &dispatch, NULL, 0) == 0;
}

extern "C" uint32_t mglTessEvalItemsPerPatch(Program *tes,
                                             const void *factor_record)
{
    return mglRenderTessEvalItemsPerPatch(
        factor_record,
        (uint32_t)(tes ? tes->tess_gen_mode : GL_TRIANGLES),
        (uint32_t)(tes ? tes->tess_gen_spacing : 0),
        (uint32_t)(tes ? tes->tess_gen_point_mode : 0));
}

extern "C" uint64_t mglTessEvalItemsPerInstance(Program *tes,
                                                const void *factor_bytes,
                                                uint32_t patch_count)
{
    if (!factor_bytes || patch_count == 0u) {
        return 0u;
    }
    uint64_t total = 0u;
    const uint8_t *base = (const uint8_t *)factor_bytes;
    for (uint32_t p = 0u; p < patch_count; p++) {
        total += mglTessEvalItemsPerPatch(
            tes, base + (uint64_t)p * MGL_AIR_TESS_FACTOR_RECORD_BYTES);
    }
    return total;
}

extern "C" bool mglTessPlanEvalCompute(Program *tes, const void *factor_bytes,
                                       uint64_t factor_byte_count,
                                       uint32_t patch_count,
                                       uint32_t instance_count,
                                       MGLTessEvalComputePlan *out)
{
    if (!out || !tes || !factor_bytes || patch_count == 0u ||
        instance_count == 0u) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    const uint64_t need =
        (uint64_t)patch_count * MGL_AIR_TESS_FACTOR_RECORD_BYTES;
    if (factor_byte_count < need) {
        return false;
    }
    const uint64_t items =
        mglTessEvalItemsPerInstance(tes, factor_bytes, patch_count);
    if (items == 0u) {
        out->empty = 1u;
        out->instance_count = instance_count;
        return true;
    }
    if (items > 0xffffffffull) {
        return false;
    }
    uint32_t stride = mglAIRPerVertexStrideForResources(
        &tes->shader_resources_list[_TESS_EVALUATION_SHADER]
                                   [_STAGE_OUTPUT_RES]);
    if (stride < MGL_AIR_PER_VERTEX_STRIDE) {
        stride = MGL_AIR_PER_VERTEX_STRIDE;
    }
    uint64_t instance_bytes = 0u;
    uint64_t out_size = 0u;
    if (__builtin_mul_overflow(items, (uint64_t)stride, &instance_bytes) ||
        __builtin_mul_overflow(instance_bytes, (uint64_t)instance_count,
                               &out_size)) {
        return false;
    }
    out->items_per_instance = (uint32_t)items;
    out->instance_count = instance_count;
    out->out_stride = stride;
    out->instance_bytes = instance_bytes;
    out->out_size = out_size;
    return true;
}

extern "C" bool mglTessEvalOwnsXFB(GLMContext ctx, Program *gs)
{
    if (!ctx || !ctx->active_state) {
        return false;
    }
    if (mglTessNativeBlockedByGeometry(gs)) {
        return false;
    }
    TransformFeedback *xfb = ctx->active_state->transform_feedback;
    return xfb && xfb->active && !xfb->paused;
}

extern "C" bool mglTessFillEvalPatchItemBases(Program *tes,
                                              const void *factor_bytes,
                                              uint32_t patch_count,
                                              uint32_t *bases_out)
{
    if (!factor_bytes || !bases_out || patch_count == 0u) {
        return false;
    }
    uint32_t base = 0u;
    const uint8_t *bytes = (const uint8_t *)factor_bytes;
    for (uint32_t p = 0u; p < patch_count; p++) {
        bases_out[p] = base;
        const uint32_t items = mglTessEvalItemsPerPatch(
            tes, bytes + (uint64_t)p * MGL_AIR_TESS_FACTOR_RECORD_BYTES);
        if (base > UINT32_MAX - items) {
            return false;
        }
        base += items;
    }
    bases_out[patch_count] = base;
    return true;
}

extern "C" uint32_t mglTessVerticesPerPrimitive(const Program *tes)
{
    if (!tes) {
        return 3u;
    }
    if (tes->tess_gen_point_mode) {
        return 1u;
    }
    if (tes->tess_gen_mode == GL_ISOLINES) {
        return 2u;
    }
    return 3u;
}

extern "C" uint64_t mglTessPrimitivesFromItems(const Program *tes,
                                               uint64_t items)
{
    const uint32_t vpp = mglTessVerticesPerPrimitive(tes);
    return vpp ? items / vpp : 0u;
}

extern "C" uint64_t mglTessGeneratedPrimitiveCount(Program *tes,
                                                   const void *factor_bytes,
                                                   uint32_t patch_count,
                                                   uint32_t instance_count)
{
    if (!tes || !factor_bytes || patch_count == 0u || instance_count == 0u) {
        return 0u;
    }
    const uint64_t items =
        mglTessEvalItemsPerInstance(tes, factor_bytes, patch_count);
    const uint64_t prims = mglTessPrimitivesFromItems(tes, items);
    if (instance_count && prims > UINT64_MAX / instance_count) {
        return UINT64_MAX;
    }
    return prims * (uint64_t)instance_count;
}

extern "C" GLenum mglTessRasterGLMode(const Program *tes)
{
    if (!tes) {
        return GL_TRIANGLES;
    }
    if (tes->tess_gen_point_mode) {
        return GL_POINTS;
    }
    if (tes->tess_gen_mode == GL_ISOLINES) {
        return GL_LINES;
    }
    return GL_TRIANGLES;
}

extern "C" uint32_t mglTessRasterPrimitiveType(const Program *tes)
{
    const GLenum mode = mglTessRasterGLMode(tes);
    if (mode == GL_POINTS) {
        return MGL_TESS_PRIMITIVE_POINT;
    }
    if (mode == GL_LINES) {
        return MGL_TESS_PRIMITIVE_LINE;
    }
    return MGL_TESS_PRIMITIVE_TRIANGLE;
}

extern "C" void mglTessPlanRasterQuery(const Program *tes,
                                       uint64_t instance_count,
                                       uint64_t items_per_instance,
                                       int xfb_active,
                                       uint64_t xfb_written_bytes,
                                       uint32_t xfb_compact_stride,
                                       MGLTessRasterQueryPlan *out)
{
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    const uint64_t prims_per =
        mglTessPrimitivesFromItems(tes, items_per_instance);
    uint64_t prims = prims_per;
    if (instance_count && prims_per > UINT64_MAX / instance_count) {
        prims = UINT64_MAX;
    } else {
        prims = prims_per * instance_count;
    }
    uint64_t written = prims;
    if (xfb_active) {
        const uint64_t vpp = mglTessVerticesPerPrimitive(tes);
        const uint64_t stride =
            xfb_compact_stride > 0u ? (uint64_t)xfb_compact_stride : 1u;
        const uint64_t denom = stride * (vpp ? vpp : 1u);
        const uint64_t xfb_prims = denom ? xfb_written_bytes / denom : 0u;
        if (xfb_prims < written) {
            written = xfb_prims;
        }
    }
    out->prims = prims;
    out->written = written;
}

extern "C" int mglTessPlanEvalAfterCompute(int has_gs, int rasterizer_discard,
                                           uint32_t items_per_instance,
                                           uint32_t instance_count,
                                           MGLTessEvalAfterComputePlan *out)
{
    if (!out) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    if (has_gs) {
        const uint64_t n =
            (uint64_t)items_per_instance * (uint64_t)instance_count;
        out->action = MGL_TESS_AFTER_COMPUTE_GS;
        if (n == 0u || n > (uint64_t)INT32_MAX) {
            out->gs_empty = 1u;
            return 1;
        }
        out->gs_vertex_count = (uint32_t)n;
        return 1;
    }
    if (rasterizer_discard) {
        out->action = MGL_TESS_AFTER_COMPUTE_DISCARD;
        return 1;
    }
    out->action = MGL_TESS_AFTER_COMPUTE_PASSTHROUGH;
    return 1;
}

extern "C" int mglTessPassthroughRasterReady(int state_ready, int has_encoder,
                                             int raster_empty)
{
    return state_ready && has_encoder == 1 && !raster_empty ? 1 : 0;
}

extern "C" int mglTessPlanNativeVertexDescriptor(
    const Program *tes, uint32_t tcs_output_stride,
    MGLTessNativeVertexPlan *out)
{
    if (!out || tcs_output_stride == 0u) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    out->stride = tcs_output_stride;
    out->attribs[0].index = 0u;
    out->attribs[0].format = mglRenderDoubleVertexAttribFloatFormat(4u);
    out->attribs[0].offset = 0u;
    out->n_attribs = 1u;
    out->attrib_count = 1u;
    const MGLShaderResourceList *inputs =
        tes ? &tes->shader_resources_list[_TESS_EVALUATION_SHADER]
                                        [_STAGE_INPUT_RES]
            : NULL;
    if (!inputs || !inputs->list) {
        return 1;
    }
    for (GLuint i = 0; i < inputs->count; i++) {
        const MGLShaderResource *input = &inputs->list[i];
        if (input->is_per_patch || input->location >= 30u) {
            continue;
        }
        const uint32_t format =
            mglRenderTessControlPointFormat((uint64_t)input->gl_type);
        if (format == 0u) {
            return 0;
        }
        const uint32_t attribute = (uint32_t)input->location + 1u;
        if (attribute >= 32u) {
            continue;
        }
        if (out->n_attribs >= 32u) {
            return 0;
        }
        out->attribs[out->n_attribs].index = attribute;
        out->attribs[out->n_attribs].format = format;
        out->attribs[out->n_attribs].offset =
            MGL_AIR_PER_VERTEX_STRIDE + (uint32_t)input->location * 16u;
        out->n_attribs++;
        if (attribute + 1u > out->attrib_count) {
            out->attrib_count = attribute + 1u;
        }
    }
    return 1;
}

extern "C" uint32_t mglTessSeedEvalOutputRecords(
    Program *tes, const void *factor_bytes, uint32_t patch_count,
    uint32_t instance_count, void *records, uint64_t records_bytes,
    uint32_t stride)
{
    if (!tes || !factor_bytes || !records || patch_count == 0u ||
        instance_count == 0u || stride < MGL_AIR_PER_VERTEX_STRIDE) {
        return 0u;
    }
    const uint64_t items =
        mglTessEvalItemsPerInstance(tes, factor_bytes, patch_count);
    if (items == 0u || items > UINT32_MAX) {
        return 0u;
    }
    uint64_t instance_bytes = 0u;
    uint64_t total_bytes = 0u;
    if (mglRenderCheckedProduct(items, stride, &instance_bytes) != 0 ||
        mglRenderCheckedProduct(instance_bytes, instance_count, &total_bytes) !=
            0 ||
        total_bytes > records_bytes) {
        return 0u;
    }
    uint8_t *base = (uint8_t *)records;
    memset(base, 0, (size_t)total_bytes);
    const uint8_t *factors = (const uint8_t *)factor_bytes;
    uint64_t item_base = 0u;
    for (uint32_t p = 0u; p < patch_count; p++) {
        const void *record =
            factors + (uint64_t)p * MGL_AIR_TESS_FACTOR_RECORD_BYTES;
        const uint32_t patch_items = mglTessEvalItemsPerPatch(tes, record);
        if (mglRenderSeedTessDomain(
                record, (uint32_t)tes->tess_gen_mode,
                (uint32_t)tes->tess_gen_spacing,
                (uint32_t)tes->tess_gen_point_mode,
                (uint32_t)tes->tess_gen_vertex_order,
                base + item_base * stride, patch_items, stride) != patch_items) {
            return 0u;
        }
        item_base += patch_items;
    }
    for (uint32_t inst = 1u; inst < instance_count; inst++) {
        memcpy(base + (uint64_t)inst * instance_bytes, base,
               (size_t)instance_bytes);
    }
    return (uint32_t)items;
}

extern "C" void mglTessPackXFBFieldFromCarrier(uint32_t gl_type, const void *src,
                                               void *dst, uint32_t field_bytes)
{
    if (!src || !dst || field_bytes == 0u) {
        return;
    }
    const uint8_t *in = (const uint8_t *)src;
    uint8_t *out = (uint8_t *)dst;
    const uint32_t comps = field_bytes / sizeof(uint32_t);
    if (gl_type == GL_INT || gl_type == GL_INT_VEC2 || gl_type == GL_INT_VEC3 ||
        gl_type == GL_INT_VEC4) {
        for (uint32_t c = 0u; c < comps && c < 4u; c++) {
            float f = 0.f;
            memcpy(&f, in + c * 4u, sizeof(f));
            const int32_t iv = (int32_t)f;
            memcpy(out + c * 4u, &iv, sizeof(iv));
        }
        return;
    }
    if (gl_type == GL_UNSIGNED_INT || gl_type == GL_UNSIGNED_INT_VEC2 ||
        gl_type == GL_UNSIGNED_INT_VEC3 || gl_type == GL_UNSIGNED_INT_VEC4) {
        for (uint32_t c = 0u; c < comps && c < 4u; c++) {
            float f = 0.f;
            memcpy(&f, in + c * 4u, sizeof(f));
            const uint32_t uv = (uint32_t)f;
            memcpy(out + c * 4u, &uv, sizeof(uv));
        }
        return;
    }
    if (gl_type == GL_DOUBLE || gl_type == GL_DOUBLE_VEC2 ||
        gl_type == GL_DOUBLE_VEC3 || gl_type == GL_DOUBLE_VEC4) {
        const uint32_t dcomps = field_bytes / (uint32_t)sizeof(double);
        for (uint32_t c = 0u; c < dcomps && c < 4u; c++) {
            float f = 0.f;
            memcpy(&f, in + c * 4u, sizeof(f));
            const double dv = (double)f;
            memcpy(out + c * sizeof(double), &dv, sizeof(dv));
        }
        return;
    }
    if (gl_type == GL_FLOAT_MAT2) {
        memcpy(out + 0u, in + 0u, 8u);
        memcpy(out + 8u, in + 16u, 8u);
        return;
    }
    if (gl_type == GL_FLOAT_MAT3) {
        memcpy(out + 0u, in + 0u, 12u);
        memcpy(out + 12u, in + 16u, 12u);
        memcpy(out + 24u, in + 32u, 12u);
        return;
    }
    if (gl_type == GL_FLOAT_MAT4) {
        memcpy(out + 0u, in + 0u, 16u);
        memcpy(out + 16u, in + 16u, 16u);
        memcpy(out + 32u, in + 32u, 16u);
        memcpy(out + 48u, in + 48u, 16u);
        return;
    }
    memcpy(out, in, field_bytes);
}

extern "C" int mglTessResolveXFBSource(const Program *program, const char *name,
                                       uint32_t *offset_out,
                                       uint32_t *gl_type_out,
                                       uint32_t *bytes_out)
{
    if (!program || !name || !offset_out || !gl_type_out || !bytes_out) {
        return 0;
    }
    if (strcmp(name, "gl_Position") == 0) {
        *offset_out = 0u;
        *gl_type_out = GL_FLOAT_VEC4;
        *bytes_out = 16u;
        return 1;
    }
    if (strcmp(name, "gl_PointSize") == 0) {
        *offset_out = 16u;
        *gl_type_out = GL_FLOAT;
        *bytes_out = 4u;
        return 1;
    }
    const MGLShaderResource *output = mglProgramFindStageOutputForXFBName(
        const_cast<Program *>(program), _TESS_EVALUATION_SHADER, name);
    if (!output) {
        return 0;
    }
    const uint32_t field_bytes =
        (uint32_t)mglRenderTESXFBFieldByteSize((uint64_t)output->gl_type);
    if (field_bytes == 0u) {
        return 0;
    }
    *offset_out = MGL_AIR_PER_VERTEX_STRIDE + (uint32_t)output->location * 16u;
    *gl_type_out = (uint32_t)output->gl_type;
    *bytes_out = field_bytes;
    return 1;
}

extern "C" uint32_t mglTessPackXFBInterleaved(const Program *tes, const void *src,
                                              uint32_t src_stride,
                                              uint32_t vertex_count, void *dst,
                                              uint32_t dst_stride)
{
    if (!tes || !src || !dst || src_stride == 0u || dst_stride == 0u) {
        return 0u;
    }
    const uint8_t *in = (const uint8_t *)src;
    uint8_t *out = (uint8_t *)dst;
    for (uint32_t vertex = 0u; vertex < vertex_count; vertex++) {
        uint32_t compact = 0u;
        for (GLsizei varying = 0;
             varying < tes->transform_feedback_varying_count; varying++) {
            const char *name = tes->transform_feedback_varying_names[varying];
            uint32_t record_offset = 0u, gl_type = 0u, field_bytes = 0u;
            if (!mglTessResolveXFBSource(tes, name, &record_offset, &gl_type,
                                         &field_bytes) ||
                compact > dst_stride || field_bytes > dst_stride - compact) {
                continue;
            }
            mglTessPackXFBFieldFromCarrier(
                gl_type, in + (uint64_t)vertex * src_stride + record_offset,
                out + (uint64_t)vertex * dst_stride + compact, field_bytes);
            compact += field_bytes;
        }
    }
    return vertex_count;
}

extern "C" uint32_t mglTessPackXFBSeparate(const Program *tes, const char *name,
                                           const void *src, uint32_t src_stride,
                                           uint32_t vertex_count, void *dst)
{
    uint32_t record_offset = 0u, gl_type = 0u, field_bytes = 0u;
    if (!tes || !name || !src || !dst || src_stride == 0u ||
        !mglTessResolveXFBSource(tes, name, &record_offset, &gl_type,
                                 &field_bytes) ||
        field_bytes == 0u) {
        return 0u;
    }
    const uint8_t *in = (const uint8_t *)src;
    uint8_t *out = (uint8_t *)dst;
    for (uint32_t vertex = 0u; vertex < vertex_count; vertex++) {
        mglTessPackXFBFieldFromCarrier(
            gl_type, in + (uint64_t)vertex * src_stride + record_offset,
            out + (uint64_t)vertex * field_bytes, field_bytes);
    }
    return vertex_count;
}

extern "C" int mglTessPlanXFBDestination(uint32_t items_per_instance,
                                         uint32_t instance_count,
                                         uint32_t compact_stride,
                                         uint32_t vertices_per_primitive,
                                         uint64_t session_offset,
                                         int64_t slot_offset,
                                         uint64_t visible_bytes,
                                         MGLTessXFBDestPlan *out)
{
    if (!out) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    if (items_per_instance == 0u || instance_count == 0u ||
        compact_stride == 0u || vertices_per_primitive == 0u ||
        slot_offset < 0) {
        return 0;
    }
    uint64_t capture_vertices = 0u;
    uint64_t primitive_bytes = 0u;
    if (__builtin_mul_overflow((uint64_t)items_per_instance,
                               (uint64_t)instance_count, &capture_vertices) ||
        __builtin_mul_overflow((uint64_t)vertices_per_primitive,
                               (uint64_t)compact_stride, &primitive_bytes) ||
        primitive_bytes == 0u) {
        return 0;
    }
    const uint64_t capture_primitives =
        capture_vertices / (uint64_t)vertices_per_primitive;
    if (session_offset > visible_bytes ||
        (uint64_t)slot_offset > UINT64_MAX - session_offset) {
        return 0;
    }
    const uint64_t remaining = visible_bytes - session_offset;
    uint64_t copied_primitives = remaining / primitive_bytes;
    if (copied_primitives > capture_primitives) {
        copied_primitives = capture_primitives;
    }
    uint64_t copied_vertices = 0u;
    uint64_t written_bytes = 0u;
    if (__builtin_mul_overflow(copied_primitives,
                               (uint64_t)vertices_per_primitive,
                               &copied_vertices) ||
        __builtin_mul_overflow(copied_primitives, primitive_bytes,
                               &written_bytes) ||
        copied_vertices > UINT32_MAX || written_bytes > UINT32_MAX) {
        return 0;
    }
    out->copied_vertices = (uint32_t)copied_vertices;
    out->written_bytes = (uint32_t)written_bytes;
    out->destination_offset = (uint32_t)((uint64_t)slot_offset + session_offset);
    out->valid = 1u;
    return 1;
}

extern "C" int mglTessPlanEvalXfbCapture(uint32_t items_per_instance,
                                         uint32_t instance_count,
                                         uint32_t out_stride,
                                         uint32_t compact_stride,
                                         uint32_t *capture_vertices,
                                         uint32_t *required_bytes)
{
    if (!capture_vertices || !required_bytes || compact_stride == 0u ||
        out_stride == 0u || items_per_instance == 0u || instance_count == 0u) {
        return 0;
    }
    uint64_t verts = 0u;
    uint64_t bytes = 0u;
    if (mglRenderCheckedProduct(items_per_instance, instance_count, &verts) !=
            0 ||
        mglRenderCheckedProduct(verts, out_stride, &bytes) != 0 || verts == 0u ||
        bytes == 0u || verts > UINT32_MAX || bytes > UINT32_MAX) {
        return 0;
    }
    *capture_vertices = (uint32_t)verts;
    *required_bytes = (uint32_t)bytes;
    return 1;
}

extern "C" void mglTessPlanEvalGather(int indexed, uint32_t instance_records,
                                      uint32_t patch_vertices,
                                      uint32_t patch_count,
                                      uint32_t *verts_per_instance,
                                      uint32_t *prims_per_instance)
{
    if (verts_per_instance) {
        if (indexed) {
            *verts_per_instance = instance_records;
        } else {
            *verts_per_instance = patch_vertices > 0u ? patch_vertices : 1u;
        }
    }
    if (prims_per_instance) {
        *prims_per_instance = indexed ? patch_count : 0u;
    }
}

extern "C" uint64_t mglTessDummyXfbBytes(uint64_t out_size)
{
    return out_size > 0u ? out_size : 1u;
}

extern "C" void mglTessFillEvalPerPatchSpec(
    void *gl_in_buffer, uint64_t gl_in_offset, uint64_t gl_in_instance_stride,
    void *gather_buffer, uint32_t gather_verts, uint32_t gather_prims,
    int indexed, uint32_t gl_in_vertices, uint32_t patch_count,
    uint32_t instance_count, uint32_t items_per_instance,
    MGLTessEvalPerPatchDispatchSpec *out)
{
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    out->gl_in_buffer = gl_in_buffer;
    out->gl_in_offset = gl_in_offset;
    out->gl_in_instance_stride = gl_in_instance_stride;
    out->gather_buffer = indexed ? gather_buffer : NULL;
    out->gather_verts_per_instance = gather_verts;
    out->gather_prims_per_instance = gather_prims;
    out->gather_first_vertex = 0u;
    out->indexed = indexed ? 1u : 0u;
    out->gl_in_vertices = gl_in_vertices;
    out->patch_count = patch_count;
    out->instance_count = instance_count;
    out->items_per_instance = items_per_instance;
}

static bool mglTessKeepAppend(uint8_t *keep, size_t *used, size_t cap,
                              const void *src, size_t len, const void **out_ptr)
{
    if (!keep || !used || !src || !out_ptr || len == 0u) {
        return false;
    }
    const size_t aligned = (*used + 3u) & ~size_t{3};
    if (aligned > cap || cap - aligned < len) {
        return false;
    }
    memcpy(keep + aligned, src, len);
    *out_ptr = keep + aligned;
    *used = aligned + len;
    return true;
}

static bool mglTessPlanAppendBytes(MGLRenderComputeExecutionPlan *plan,
                                   uint8_t *keep, size_t *used, size_t cap,
                                   const void *src, uint32_t len, uint32_t index)
{
    const void *stored = NULL;
    if (!plan || !mglTessKeepAppend(keep, used, cap, src, len, &stored)) {
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
        .bytes = stored,
        .length = len,
    };
    return true;
}

static bool mglTessPlanAppendDirectDispatch(MGLRenderComputeExecutionPlan *plan,
                                            uint32_t groups_x, uint32_t local_x)
{
    MGLRenderComputePlan dispatch = {};
    dispatch.dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT;
    dispatch.groups_x = groups_x;
    dispatch.groups_y = 1u;
    dispatch.groups_z = 1u;
    dispatch.local_x = local_x;
    dispatch.local_y = 1u;
    dispatch.local_z = 1u;
    return mglRenderAppendComputeDispatchToPlan(plan, &dispatch, NULL, 0) == 0;
}

extern "C" bool mglTessAppendEvalPerPatchDispatches(
    MGLRenderComputeExecutionPlan *plan, Program *tes, const void *factor_bytes,
    const MGLTessEvalPerPatchDispatchSpec *spec, void **out_keep_alive)
{
    if (out_keep_alive) {
        *out_keep_alive = NULL;
    }
    if (!plan || !tes || !factor_bytes || !spec || !out_keep_alive ||
        !spec->gl_in_buffer || spec->patch_count == 0u ||
        spec->instance_count == 0u || spec->items_per_instance == 0u) {
        return false;
    }
    if (spec->indexed && !spec->gather_buffer) {
        return false;
    }

    uint32_t *patchBases =
        (uint32_t *)malloc((size_t)(spec->patch_count + 1u) * sizeof(uint32_t));
    if (!patchBases) {
        return false;
    }
    if (!mglTessFillEvalPatchItemBases(tes, factor_bytes, spec->patch_count,
                                       patchBases)) {
        free(patchBases);
        return false;
    }

    const uint64_t cap64 =
        (uint64_t)spec->instance_count *
        (32u + (uint64_t)spec->patch_count * 16u);
    if (cap64 == 0u || cap64 > SIZE_MAX) {
        free(patchBases);
        return false;
    }
    uint8_t *keep = (uint8_t *)malloc((size_t)cap64);
    if (!keep) {
        free(patchBases);
        return false;
    }
    size_t used = 0u;

    const uint8_t *factors = (const uint8_t *)factor_bytes;
    for (uint32_t inst = 0u; inst < spec->instance_count; inst++) {
        const uint64_t instGlInOffset =
            spec->indexed
                ? 0u
                : spec->gl_in_offset +
                      (uint64_t)inst * spec->gl_in_instance_stride;
        if (!mglTessPlanAppendBuffer(plan, spec->gl_in_buffer, instGlInOffset,
                                     MGL_AIR_TESS_SLOT_TCS_STAGE_IN)) {
            free(patchBases);
            free(keep);
            return false;
        }
        if (spec->gather_buffer &&
            !mglTessPlanAppendBuffer(plan, spec->gather_buffer, 0u,
                                     MGL_AIR_TESS_SLOT_GATHER_INDEX)) {
            free(patchBases);
            free(keep);
            return false;
        }
        const uint32_t gatherParams[5] = {
            spec->gather_verts_per_instance, spec->gather_prims_per_instance,
            spec->gather_first_vertex, spec->indexed ? 1u : 0u, inst,
        };
        if (!mglTessPlanAppendBytes(plan, keep, &used, (size_t)cap64,
                                    gatherParams, sizeof(gatherParams),
                                    MGL_AIR_TESS_SLOT_GATHER_PARAMS)) {
            free(patchBases);
            free(keep);
            return false;
        }
        for (uint32_t p = 0u; p < spec->patch_count; p++) {
            const void *record =
                factors + (uint64_t)p * MGL_AIR_TESS_FACTOR_RECORD_BYTES;
            const uint32_t items = mglTessEvalItemsPerPatch(tes, record);
            if (items == 0u) {
                continue;
            }
            const uint32_t contractWords[4] = {
                p,
                spec->gl_in_vertices,
                items,
                inst * spec->items_per_instance + patchBases[p],
            };
            if (!mglTessPlanAppendBytes(plan, keep, &used, (size_t)cap64,
                                        contractWords, sizeof(contractWords),
                                        MGL_AIR_TESS_SLOT_INDIRECT) ||
                !mglTessPlanAppendDirectDispatch(plan, (items + 63u) / 64u,
                                                 64u)) {
                free(patchBases);
                free(keep);
                return false;
            }
        }
    }
    free(patchBases);
    *out_keep_alive = keep;
    return true;
}

static uint32_t mglTessGLUnitForResource(const MGLShaderResource *resource,
                                         uint32_t fallback)
{
    if (!resource) {
        return fallback;
    }
    if (resource->sampler_unit >= 0) {
        return (uint32_t)resource->sampler_unit;
    }
    return resource->gl_binding;
}

static uint32_t mglTessCollectTextureBindsOfType(
    GLMContext ctx, Program *program, int stage, int resource_type,
    uint32_t kind, MGLTessTextureBind *out, uint32_t cap, uint32_t filled)
{
    const int32_t count =
        ctx ? mglRendererGetProgramBindingCount(ctx, stage, resource_type)
            : (program && stage >= 0 && stage < _MAX_SHADER_TYPES &&
                       resource_type >= 0 &&
                       resource_type < MGL_MAX_SHADER_RESOURCES
                   ? (int32_t)program->shader_resources_list[stage][resource_type]
                         .count
                   : 0);
    for (int32_t i = 0; i < count; i++) {
        const MGLShaderResource *resource = NULL;
        if (program && stage >= 0 && stage < _MAX_SHADER_TYPES &&
            resource_type >= 0 && resource_type < MGL_MAX_SHADER_RESOURCES &&
            (uint32_t)i <
                program->shader_resources_list[stage][resource_type].count) {
            resource =
                &program->shader_resources_list[stage][resource_type].list[i];
        }
        if (mglShouldSkipStageTextureResource(program, stage, resource_type,
                                              resource)) {
            continue;
        }
        const uint32_t fallback_gl =
            ctx ? (uint32_t)mglRendererGetProgramGLBinding(ctx, stage,
                                                           resource_type, i)
                : 0u;
        const uint32_t metal_slot =
            resource ? mglMetalResourceSlot(resource)
                     : (ctx ? (uint32_t)mglRendererGetProgramBinding(
                                  ctx, stage, resource_type, i)
                            : 0u);
        const uint32_t gl_unit = mglTessGLUnitForResource(resource, fallback_gl);
        if (metal_slot >= TEXTURE_UNITS || gl_unit >= TEXTURE_UNITS) {
            continue;
        }
        if (filled >= cap) {
            return filled;
        }
        out[filled].kind = kind;
        out[filled].metal_slot = metal_slot;
        out[filled].gl_unit = gl_unit;
        out[filled].combined_sampler_slot = UINT32_MAX;
        if (kind == MGL_TESS_BIND_SAMPLED_IMAGE && resource &&
            resource->has_combined_sampler) {
            out[filled].combined_sampler_slot =
                mglMetalCombinedSamplerSlot(resource);
        }
        filled++;
    }
    return filled;
}

extern "C" uint32_t mglTessCollectTextureBinds(GLMContext ctx, Program *program,
                                               int stage, MGLTessTextureBind *out,
                                               uint32_t cap)
{
    if (!program || !out || cap == 0u) {
        return 0u;
    }
    uint32_t filled = mglTessCollectTextureBindsOfType(
        ctx, program, stage, _STORAGE_IMAGE_RES, MGL_TESS_BIND_STORAGE_IMAGE,
        out, cap, 0u);
    return mglTessCollectTextureBindsOfType(
        ctx, program, stage, _SAMPLED_IMAGE_RES, MGL_TESS_BIND_SAMPLED_IMAGE,
        out, cap, filled);
}

extern "C" bool mglTessInitStageInDefaults(void *dst, uint64_t vertices,
                                           uint64_t stride)
{
    if (!dst || vertices == 0u || stride < MGL_AIR_PER_VERTEX_STRIDE ||
        vertices > SIZE_MAX / stride) {
        return false;
    }
    memset(dst, 0, (size_t)(vertices * stride));
    const float one = 1.0f;
    uint8_t *base = (uint8_t *)dst;
    for (uint64_t v = 0u; v < vertices; v++) {
        uint8_t *record = base + v * stride;
        memcpy(record + MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET, &one, sizeof(one));
        for (uint32_t d = 0u; d < MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT; d++) {
            memcpy(record + MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET +
                       (uint64_t)d * sizeof(float),
                   &one, sizeof(one));
        }
    }
    return true;
}

extern "C" bool mglTessCompactSparseCapture(
    const void *sparse, uint64_t sparse_offset, uint32_t sparse_records,
    uint32_t stride, const uint32_t *gather, uint32_t gather_count,
    uint32_t instance_count, void *continuous, uint64_t continuous_bytes)
{
    if (!sparse || !gather || !continuous || stride == 0u ||
        gather_count == 0u || instance_count == 0u || sparse_records == 0u) {
        return false;
    }
    if ((uint64_t)gather_count > UINT64_MAX / stride ||
        (uint64_t)instance_count > UINT64_MAX / ((uint64_t)gather_count * stride)) {
        return false;
    }
    const uint64_t need =
        (uint64_t)instance_count * (uint64_t)gather_count * stride;
    if (need != continuous_bytes) {
        return false;
    }
    memset(continuous, 0, (size_t)continuous_bytes);
    const uint8_t *src = (const uint8_t *)sparse;
    uint8_t *dst = (uint8_t *)continuous;
    for (uint32_t inst = 0u; inst < instance_count; inst++) {
        const uint64_t sparseInstBase =
            sparse_offset + (uint64_t)inst * sparse_records * stride;
        const uint64_t contInstBase =
            (uint64_t)inst * gather_count * stride;
        for (uint32_t gi = 0u; gi < gather_count; gi++) {
            const uint32_t vid = gather[gi];
            if (vid >= sparse_records) {
                continue;
            }
            memcpy(dst + contInstBase + (uint64_t)gi * stride,
                   src + sparseInstBase + (uint64_t)vid * stride, stride);
        }
    }
    return true;
}

extern "C" double mglDecodeVertexAttribComponent(const uint8_t *src,
                                                 GLenum type,
                                                 GLboolean normalized,
                                                 unsigned long component);

static void mglTessWriteStageInComponent(uint8_t *destination,
                                         const MGLTessStageInMember *member,
                                         uint32_t component, double value)
{
    if (!destination || !member || component >= member->components ||
        member->component_bytes == 0u) {
        return;
    }
    uint8_t *component_destination =
        destination + member->offset +
        (uint64_t)component * member->component_bytes;
    const size_t copy_bytes = MIN(member->component_bytes, sizeof(int32_t));
    if (member->base_type == MGL_TESS_STAGE_IN_INT) {
        int32_t converted = (int32_t)value;
        memcpy(component_destination, &converted, copy_bytes);
    } else if (member->base_type == MGL_TESS_STAGE_IN_UINT) {
        uint32_t converted = value < 0.0 ? 0u : (uint32_t)value;
        memcpy(component_destination, &converted, copy_bytes);
    } else {
        float converted = (float)value;
        memcpy(component_destination, &converted, copy_bytes);
    }
}

extern "C" bool mglTessPackStageInRecords(
    void *dst, uint64_t vertices, uint64_t stride, GLint first, GLsizei count,
    const uint8_t *index_bytes, GLenum index_type, bool restart_enabled,
    uint32_t restart_index, GLint base_vertex, GLuint base_instance,
    const MGLTessStageInMember *members, uint32_t member_count,
    const MGLTessStageInAttribSrc *srcs)
{
    if (!dst || !members || !srcs || vertices == 0u || stride == 0u ||
        member_count == 0u || count <= 0) {
        return false;
    }
    if (vertices > SIZE_MAX / stride) {
        return false;
    }
    const uint32_t index_width =
        index_bytes ? mglRenderGLIndexElementSize((uint64_t)index_type) : 0u;
    if (index_bytes && index_width == 0u) {
        return false;
    }
    uint8_t *base = (uint8_t *)dst;
    const uint64_t live = MIN(vertices, (uint64_t)count);
    for (uint64_t v = 0u; v < live; v++) {
        int64_t vertexIndex64 = (int64_t)first + (int64_t)v;
        if (index_bytes) {
            const uint32_t rawIndex =
                mglRenderReadGLIndexValue(index_bytes, index_width, v);
            if (restart_enabled && rawIndex == restart_index) {
                continue;
            }
            vertexIndex64 = (int64_t)rawIndex + (int64_t)base_vertex;
        }
        if (vertexIndex64 < 0) {
            continue;
        }
        uint8_t *dstVertex = base + v * stride;
        for (uint32_t m = 0u; m < member_count; m++) {
            const MGLTessStageInMember *member = &members[m];
            const MGLTessStageInAttribSrc *src = &srcs[m];
            if (member->offset >= stride ||
                member->size > stride - member->offset) {
                continue;
            }
            double values[4] = {0.0, 0.0, 0.0, 1.0};
            if (src->use_current) {
                if (src->current_valid) {
                    const uint32_t comps = MIN(src->attrib_size, 4u);
                    for (uint32_t c = 0u; c < comps; c++) {
                        values[c] = mglDecodeVertexAttribComponent(
                            src->current, (GLenum)src->type,
                            (GLboolean)src->normalized, (unsigned long)c);
                    }
                }
            } else if (src->bytes) {
                const uint64_t element_bytes = mglRenderVertexAttribElementBytes(
                    (uint64_t)src->type, src->attrib_size);
                uint32_t stride_bytes = src->stride;
                if (stride_bytes == 0u) {
                    stride_bytes = (uint32_t)element_bytes;
                }
                uint64_t attribIndex = (uint64_t)vertexIndex64;
                if (src->divisor > 0u) {
                    attribIndex = (uint64_t)(base_instance / src->divisor);
                }
                if (element_bytes > 0u && stride_bytes > 0u &&
                    src->binding_offset >= 0 && src->relativeoffset >= 0) {
                    const uint64_t baseOffset =
                        (uint64_t)src->binding_offset +
                        (uint64_t)src->relativeoffset;
                    if (attribIndex <= (UINT64_MAX - baseOffset) / stride_bytes) {
                        const uint64_t vertexOffset =
                            baseOffset + attribIndex * stride_bytes;
                        if (vertexOffset <= src->buffer_size &&
                            (src->buffer_size - vertexOffset) >= element_bytes) {
                            const uint8_t *elem = src->bytes + vertexOffset;
                            const uint32_t comps = MIN(src->attrib_size, 4u);
                            for (uint32_t c = 0u; c < comps; c++) {
                                values[c] = mglDecodeVertexAttribComponent(
                                    elem, (GLenum)src->type,
                                    (GLboolean)src->normalized,
                                    (unsigned long)c);
                            }
                        }
                    }
                }
            }
            for (uint32_t c = 0u; c < member->components && c < 4u; c++) {
                mglTessWriteStageInComponent(dstVertex, member, c, values[c]);
            }
        }
    }
    return true;
}

extern "C" bool mglTessSanitizeRestartIndices(void *dst, const void *src,
                                              uint32_t count, GLenum index_type,
                                              uint32_t restart_index)
{
    if (!dst || !src || count == 0u) {
        return false;
    }
    const uint32_t width = mglRenderGLIndexElementSize((uint64_t)index_type);
    if (width == 0u) {
        return false;
    }
    if (dst != src) {
        memcpy(dst, src, (size_t)count * width);
    }
    if (width == 1u) {
        uint8_t *bytes = (uint8_t *)dst;
        const uint8_t marker = (uint8_t)restart_index;
        for (uint32_t i = 0u; i < count; i++) {
            if (bytes[i] == marker) {
                bytes[i] = 0u;
            }
        }
    } else if (width == 2u) {
        uint16_t *words = (uint16_t *)dst;
        const uint16_t marker = (uint16_t)restart_index;
        for (uint32_t i = 0u; i < count; i++) {
            if (words[i] == marker) {
                words[i] = 0u;
            }
        }
    } else {
        uint32_t *words = (uint32_t *)dst;
        for (uint32_t i = 0u; i < count; i++) {
            if (words[i] == restart_index) {
                words[i] = 0u;
            }
        }
    }
    return true;
}

extern "C" void mglTessFillCaptureParams(uint32_t first,
                                         uint32_t records_per_instance,
                                         uint32_t base_instance,
                                         uint32_t out[3]) {
    if (!out) {
        return;
    }
    out[0] = first;
    out[1] = records_per_instance;
    out[2] = base_instance;
}

extern "C" void mglTessBindCaptureSlots(void* encoder_owner,
                                        void* capture_buffer,
                                        const uint32_t params[3]) {
    if (!encoder_owner || !capture_buffer || !params) {
        return;
    }
    (void)mglRenderSetRenderBufferForOwner(
        encoder_owner, capture_buffer, 0u, MGL_RENDER_BINDING_STAGE_VERTEX,
        kMGLCullDistanceVertexBufferIndex);
    (void)mglRenderSetRenderBytesForOwner(
        encoder_owner, params, 3u * sizeof(uint32_t),
        MGL_RENDER_BINDING_STAGE_VERTEX, kMGLCullDistanceParamsBufferIndex);
}

extern "C" void mglTessEncodeCaptureArray(void *encoder_owner, uint32_t first,
                                          uint32_t count,
                                          uint32_t instance_count,
                                          uint32_t base_instance)
{
    if (!encoder_owner || count == 0u || instance_count == 0u) {
        return;
    }
    MGLRenderDrawPlan plan = {};
    plan.kind = MGL_RENDER_DRAW_ARRAY;
    plan.primitive_type = MGL_DRAW_PRIMITIVE_POINT;
    plan.vertex_start = first;
    plan.vertex_count = count;
    plan.instance_count = instance_count;
    plan.base_instance = base_instance;
    (void)mglRenderEncodeDrawForRenderEncoderOwner(encoder_owner, &plan, NULL,
                                                   0);
}

extern "C" void mglTessEncodeCaptureIndexed(
    void *encoder_owner, void *index_buffer, uint32_t index_type,
    uint64_t index_offset, uint32_t count, int32_t base_vertex,
    uint32_t instance_count, uint32_t base_instance)
{
    if (!encoder_owner || !index_buffer || count == 0u ||
        instance_count == 0u) {
        return;
    }
    MGLRenderDrawPlan plan = {};
    plan.kind = MGL_RENDER_DRAW_INDEXED;
    plan.primitive_type = MGL_DRAW_PRIMITIVE_POINT;
    plan.index_count = count;
    plan.index_type = index_type;
    plan.index_buffer = index_buffer;
    plan.index_buffer_offset = index_offset;
    plan.instance_count = instance_count;
    plan.base_vertex = base_vertex;
    plan.base_instance = base_instance;
    (void)mglRenderEncodeDrawForRenderEncoderOwner(encoder_owner, &plan, NULL,
                                                   0);
}

extern "C" bool mglTessPlanVertexCapture(Program *vs,
                                         uint32_t records_per_instance,
                                         uint32_t instance_count, uint32_t first,
                                         uint32_t base_instance,
                                         MGLTessVertexCapturePlan *out)
{
    if (!out || records_per_instance == 0u || instance_count == 0u) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    uint32_t stride = MGL_AIR_PER_VERTEX_STRIDE;
    if (vs) {
        stride = mglAIRPerVertexStrideForResources(
            &vs->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
        if (stride < MGL_AIR_PER_VERTEX_STRIDE) {
            stride = MGL_AIR_PER_VERTEX_STRIDE;
        }
    }
    uint64_t size = 0u;
    uint64_t offset = 0u;
    if (mglRenderCheckedTessCaptureSize(
            (int64_t)records_per_instance, (int64_t)instance_count,
            (uint64_t)stride, (uint64_t)MGL_AIR_PER_VERTEX_STRIDE, &size,
            &offset) != 0) {
        return false;
    }
    out->records_per_instance = records_per_instance;
    out->capture_stride = stride;
    out->capture_size = size;
    out->capture_offset = offset;
    mglTessFillCaptureParams(first, records_per_instance, base_instance,
                             out->params);
    return true;
}

extern "C" bool mglTessResolveEvalGlIn(
    const MGLAIRTessDrawContract *contract, int has_tcs_output,
    uint64_t tcs_output_offset, uint64_t tcs_output_stride,
    uint32_t tcs_out_vertices, int has_capture, uint64_t capture_offset,
    int indexed_draw, uint32_t instance_records, uint32_t instance_count,
    MGLTessEvalGlInPlan *out)
{
    if (!out || !contract) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    if (!has_tcs_output && !has_capture) {
        return false;
    }
    if (indexed_draw && instance_records == 0u) {
        return false;
    }
    if (has_tcs_output) {
        out->from_tcs = 1u;
        out->gl_in_offset = tcs_output_offset;
        out->gl_in_stride = tcs_output_stride;
        out->gl_in_vertices = tcs_out_vertices;
    } else {
        out->gl_in_offset = capture_offset;
        out->gl_in_stride = contract->per_vertex_out_stride;
        out->gl_in_vertices = contract->patch_vertices > 0u
                                  ? contract->patch_vertices
                                  : 1u;
    }
    if (out->gl_in_stride < MGL_AIR_PER_VERTEX_STRIDE) {
        out->gl_in_stride = MGL_AIR_PER_VERTEX_STRIDE;
    }
    if (out->gl_in_vertices == 0u) {
        out->gl_in_vertices =
            contract->patch_vertices > 0u ? contract->patch_vertices : 1u;
    }
    if (!out->from_tcs && !indexed_draw) {
        out->gl_in_instance_stride =
            (uint64_t)instance_records * out->gl_in_stride;
    }
    (void)instance_count;
    return true;
}

extern "C" bool mglTessPlanTCSStageIn(uint32_t patch_vertices,
                                     uint32_t patch_count, GLsizei vertex_count,
                                     MGLTessTCSStageInPlan *out)
{
    if (!out || patch_count == 0u) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    const uint32_t verts_per_patch = patch_vertices > 0u ? patch_vertices : 1u;
    uint64_t vertices = 0u;
    if (__builtin_mul_overflow((uint64_t)patch_count, (uint64_t)verts_per_patch,
                               &vertices)) {
        return false;
    }
    if (vertex_count > 0 && vertices < (uint64_t)vertex_count) {
        vertices = (uint64_t)vertex_count;
    }
    if (vertices == 0u ||
        vertices > UINT64_MAX / (uint64_t)MGL_AIR_PER_VERTEX_STRIDE) {
        return false;
    }
    out->vertices = vertices;
    out->stride = MGL_AIR_PER_VERTEX_STRIDE;
    out->bytes = vertices * out->stride;
    out->members[0].attribute = 0u;
    out->members[0].offset = 0u;
    out->members[0].size = 16u;
    out->members[0].component_bytes = 4u;
    out->members[0].components = 4u;
    out->members[0].base_type = MGL_TESS_STAGE_IN_FLOAT;
    out->member_count = 1u;
    return true;
}

extern "C" int mglTessStageInUseCurrentValue(uint32_t enabled_attribs,
                                             uint32_t attrib, int has_binding)
{
    if (attrib >= 32u) {
        return 0;
    }
    const uint32_t bit = 0x1u << attrib;
    if ((enabled_attribs & bit) != 0u) {
        return 0;
    }
    if (enabled_attribs == 0u && has_binding) {
        return 0;
    }
    return 1;
}

extern "C" bool mglTessPlanIsolatedBinding(
    int has_buffer, int64_t offset, uint64_t buffer_length,
    int64_t storage_remaining, uint64_t available_bytes,
    uint32_t required_bytes, int resource_type,
    MGLTessIsolatedBindingPlan *out)
{
    if (!out) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    if (offset < 0) {
        return false;
    }
    if (resource_type == _ATOMIC_COUNTER_RES &&
        required_bytes < sizeof(uint32_t)) {
        required_bytes = sizeof(uint32_t);
    }
    const int isolated =
        !has_buffer || storage_remaining <= 0 ||
        (uint64_t)offset >= buffer_length || available_bytes == 0u ||
        (required_bytes > 0u && available_bytes < required_bytes);
    const uint32_t fallback =
        required_bytes > sizeof(uint32_t) ? required_bytes : sizeof(uint32_t);
    out->isolated = isolated ? 1u : 0u;
    out->writable = (resource_type == _STORAGE_BUFFER_RES ||
                     resource_type == _ATOMIC_COUNTER_RES)
                        ? 1u
                        : 0u;
    out->fallback_length = isolated ? fallback : 0u;
    if (isolated && has_buffer && available_bytes > 0u) {
        out->init_length = available_bytes < fallback
                               ? (uint32_t)available_bytes
                               : fallback;
    }
    return true;
}

extern "C" uint32_t mglTessRequiredBindingBytes(int resource_type,
                                                uint32_t required_bytes)
{
    if (resource_type == _ATOMIC_COUNTER_RES &&
        required_bytes < sizeof(uint32_t)) {
        return sizeof(uint32_t);
    }
    return required_bytes;
}

extern "C" void mglTessFillRuntimeArraySizeConstants(
    const BufferMap *maps, uint32_t map_count, uint32_t size_buffer_index,
    uint32_t *out, uint32_t out_cap)
{
    if (!out || out_cap == 0u) {
        return;
    }
    memset(out, 0, sizeof(uint32_t) * out_cap);
    if (!maps) {
        return;
    }
    for (uint32_t i = 0u; i < map_count; i++) {
        const BufferMap *map = &maps[i];
        if (!map->buf) {
            continue;
        }
        const uint32_t slot = map->has_metal_binding
                                  ? map->metal_binding_index
                                  : map->buffer_base_index;
        if (slot >= out_cap || slot == size_buffer_index) {
            continue;
        }
        const GLsizeiptr visible = mglBufferMapVisibleSize(map);
        if (visible > 0) {
            out[slot] = visible > (GLsizeiptr)UINT32_MAX ? UINT32_MAX
                                                         : (uint32_t)visible;
        }
    }
}

extern "C" void mglTessFillPointSizeParams(float point_size,
                                           int program_point_size,
                                           float out[2])
{
    if (!out) {
        return;
    }
    out[0] = point_size > 0.0f ? point_size : 1.0f;
    out[1] = program_point_size ? 1.0f : 0.0f;
}

extern "C" int mglTessNativeBuffersReady(int has_factors, int has_tcs_out,
                                         uint32_t tcs_stride)
{
    return has_factors && has_tcs_out &&
           tcs_stride >= MGL_AIR_PER_VERTEX_STRIDE;
}

extern "C" int mglTessPlanNativeFactor(uint32_t tess_gen_mode,
                                       uint64_t canonical_bytes,
                                       uint32_t patch_count,
                                       uint32_t *out_bytes)
{
    if (out_bytes) {
        *out_bytes = 0u;
    }
    if (patch_count == 0u) {
        return MGL_TESS_NATIVE_FACTOR_NONE;
    }
    const uint64_t need =
        (uint64_t)patch_count * (uint64_t)MGL_AIR_TESS_FACTOR_RECORD_BYTES;
    if (canonical_bytes < need) {
        return MGL_TESS_NATIVE_FACTOR_NONE;
    }
    if (tess_gen_mode == GL_QUADS) {
        return MGL_TESS_NATIVE_FACTOR_REUSE;
    }
    if (tess_gen_mode == GL_TRIANGLES) {
        const uint64_t tri_bytes =
            (uint64_t)patch_count * (uint64_t)MGL_AIR_TESS_FACTOR_TRI_HALF_BYTES;
        if (tri_bytes > UINT32_MAX) {
            return MGL_TESS_NATIVE_FACTOR_NONE;
        }
        if (out_bytes) {
            *out_bytes = (uint32_t)tri_bytes;
        }
        return MGL_TESS_NATIVE_FACTOR_REPACK_TRI;
    }
    return MGL_TESS_NATIVE_FACTOR_NONE;
}

extern "C" uint32_t mglTessNativePatchOutStride(int has_tcs,
                                                uint32_t tcs_patch_stride)
{
    if (has_tcs && tcs_patch_stride > 0u) {
        return tcs_patch_stride;
    }
    return 16u;
}

extern "C" int mglTessMultiInstanceTCSReuseWarn(int from_tcs,
                                                int32_t instance_count)
{
    return from_tcs && instance_count > 1 ? 1 : 0;
}

extern "C" int mglTessMultiInstanceTCSReuseIsError(int from_tcs,
                                                   int32_t instance_count)
{
    return mglTessMultiInstanceTCSReuseWarn(from_tcs, instance_count) &&
           mglEnvFlagEnabled("MGL_TESS_MULTI_INSTANCE_ERROR");
}

extern "C" int mglTessEvalIndexedGatherReady(int indexed, int has_gather,
                                             uint32_t instance_records)
{
    if (!indexed) {
        return 1;
    }
    return has_gather && instance_records > 0u ? 1 : 0;
}

extern "C" uint32_t mglTessTCSCaptureStageInStride(Program *tcs)
{
    if (!tcs) {
        return 0u;
    }
    return mglAIRPerVertexStrideForResources(
        &tcs->shader_resources_list[_TESS_CONTROL_SHADER][_STAGE_INPUT_RES]);
}

extern "C" bool mglXfbPrimitiveModeAccepts(GLenum xfb_mode, GLenum draw_mode)
{
    if (xfb_mode == GL_POINTS) {
        return draw_mode == GL_POINTS;
    }
    if (xfb_mode == GL_LINES) {
        return draw_mode == GL_LINES || draw_mode == GL_LINE_LOOP ||
               draw_mode == GL_LINE_STRIP;
    }
    if (xfb_mode == GL_TRIANGLES) {
        return draw_mode == GL_TRIANGLES || draw_mode == GL_TRIANGLE_STRIP ||
               draw_mode == GL_TRIANGLE_FAN;
    }
    return false;
}

extern "C" bool mglXfbVsOnlyEligible(const Program *program)
{
    return program && !program->shader_slots[_GEOMETRY_SHADER] &&
           !program->shader_slots[_TESS_CONTROL_SHADER] &&
           !program->shader_slots[_TESS_EVALUATION_SHADER] &&
           program->transform_feedback_layout_valid &&
           program->transform_feedback_varying_count > 0;
}

extern "C" bool mglXfbPlanVsCapture(const Program *program, MGLXfbVsPlan *out)
{
    if (!out || !mglXfbVsOnlyEligible(program)) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    const uint32_t buffer_count = program->transform_feedback_layout_buffer_count;
    if (buffer_count == 0u ||
        buffer_count > MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS) {
        return false;
    }
    out->buffer_count = buffer_count;
    out->capture_stride = mglAIRPerVertexStrideForResources(
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    const GLsizei varying_count = program->transform_feedback_varying_count;
    if (varying_count < 0 || (uint32_t)varying_count > MAX_ATTRIBS) {
        return false;
    }
    out->field_count = (uint32_t)varying_count;
    for (GLsizei varying = 0; varying < varying_count; varying++) {
        const MGLTransformFeedbackVaryingPlan *layout =
            &program->transform_feedback_layout[varying];
        MGLXfbVsField *field = &out->fields[varying];
        field->buffer_index = layout->buffer_index;
        field->component_offset = layout->component_offset;
        field->component_count = layout->component_count;
        if (layout->buffer_index >= buffer_count || layout->stream > 0 ||
            layout->component_count > 4u) {
            return false;
        }
        const char *name = program->transform_feedback_varying_names[varying];
        if (layout->component_count == 0u || layout->stream < 0) {
            continue;
        }
        if (!name || !name[0]) {
            return false;
        }
        if (strcmp(name, "gl_Position") == 0 && layout->builtin) {
            field->source_offset = MGL_AIR_PER_VERTEX_POSITION_OFFSET;
            field->gl_type = GL_FLOAT_VEC4;
            field->has_source = 1u;
        } else if (strcmp(name, "gl_PointSize") == 0 && layout->builtin) {
            field->source_offset = MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET;
            field->gl_type = GL_FLOAT;
            field->has_source = 1u;
        } else {
            const char *bracket = strchr(name, '[');
            GLuint array_element = 0u;
            if (bracket) {
                char *end = NULL;
                const unsigned long parsed = strtoul(bracket + 1, &end, 10);
                if (!end || *end != ']' || end[1] != '\0') {
                    return false;
                }
                array_element = (GLuint)parsed;
            }
            const MGLShaderResource *output =
                mglProgramFindStageOutputForXFBName(
                    const_cast<Program *>(program), _VERTEX_SHADER, name);
            if (!output || output->location >= 0x0fffffffu) {
                return false;
            }
            uint32_t record_slot = (uint32_t)output->location;
            if (bracket) {
                const GLuint array_size =
                    output->gl_array_size > 0 ? (GLuint)output->gl_array_size
                                              : 1u;
                if (!output->is_array || array_element >= array_size) {
                    return false;
                }
                record_slot += array_element;
            } else if (output->is_array) {
                return false;
            }
            field->source_offset =
                MGL_AIR_PER_VERTEX_STRIDE + record_slot * 16u;
            field->gl_type = (uint32_t)output->gl_type;
            field->has_source = 1u;
        }
        const uint32_t comp_bytes =
            (field->gl_type == GL_DOUBLE || field->gl_type == GL_DOUBLE_VEC2 ||
             field->gl_type == GL_DOUBLE_VEC3 ||
             field->gl_type == GL_DOUBLE_VEC4)
                ? (uint32_t)sizeof(double)
                : (uint32_t)sizeof(uint32_t);
        const uint32_t end =
            (layout->component_offset + layout->component_count) * comp_bytes;
        if (end > out->buffer_stride[layout->buffer_index]) {
            out->buffer_stride[layout->buffer_index] = end;
        }
    }
    return true;
}

extern "C" bool mglXfbPlanVsBufferDest(uint32_t record_count, uint32_t stride,
                                       int has_buffer, int64_t slot_offset,
                                       uint64_t session_offset,
                                       uint64_t visible_bytes,
                                       MGLXfbVsBufferDest *out)
{
    if (!out) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    if (!has_buffer || slot_offset < 0 || stride == 0u) {
        out->skip = 1u;
        return true;
    }
    if (session_offset >= visible_bytes) {
        out->skip = 1u;
        return true;
    }
    const uint64_t capacity = (visible_bytes - session_offset) / stride;
    uint64_t written = record_count < capacity ? record_count : capacity;
    if (written == 0u || written > UINT32_MAX / stride) {
        out->skip = 1u;
        return true;
    }
    const uint64_t dest = (uint64_t)slot_offset + session_offset;
    if (dest > UINT32_MAX) {
        out->skip = 1u;
        return true;
    }
    out->written_records = (uint32_t)written;
    out->written_bytes = (uint32_t)written * stride;
    out->destination_offset = (uint32_t)dest;
    return true;
}

extern "C" bool mglXfbPlanVsBufferDestOrUnbacked(uint32_t record_count,
                                                uint32_t stride, int has_metal,
                                                int64_t slot_offset,
                                                uint64_t session_offset,
                                                uint64_t visible_bytes,
                                                MGLXfbVsBufferDest *out)
{
    if (has_metal && slot_offset >= 0) {
        return mglXfbPlanVsBufferDest(record_count, stride, 1, slot_offset,
                                      session_offset, visible_bytes, out);
    }
    if (!out) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    if (record_count == 0u || stride == 0u ||
        record_count > UINT32_MAX / stride) {
        out->skip = 1u;
        return true;
    }
    out->written_records = record_count;
    out->written_bytes = record_count * stride;
    out->destination_offset = 0u;
    return true;
}

extern "C" uint32_t mglXfbPackVsRecords(const MGLXfbVsPlan *plan, uint32_t buffer,
                                        const void *src, uint64_t src_offset,
                                        uint32_t src_stride,
                                        uint32_t record_count, void *dst,
                                        uint32_t dst_stride)
{
    if (!plan || !src || !dst || src_stride == 0u || dst_stride == 0u ||
        buffer >= plan->buffer_count) {
        return 0u;
    }
    const uint8_t *in = (const uint8_t *)src + src_offset;
    uint8_t *out = (uint8_t *)dst;
    for (uint32_t record = 0u; record < record_count; record++) {
        const uint8_t *src_record = in + (uint64_t)record * src_stride;
        uint8_t *dst_record = out + (uint64_t)record * dst_stride;
        for (uint32_t varying = 0u; varying < plan->field_count; varying++) {
            const MGLXfbVsField *field = &plan->fields[varying];
            if (field->buffer_index != buffer || !field->has_source) {
                continue;
            }
            const uint32_t dst_elem =
                (field->gl_type == GL_DOUBLE ||
                 field->gl_type == GL_DOUBLE_VEC2 ||
                 field->gl_type == GL_DOUBLE_VEC3 ||
                 field->gl_type == GL_DOUBLE_VEC4)
                    ? (uint32_t)sizeof(double)
                    : (uint32_t)sizeof(uint32_t);
            const uint32_t field_bytes = field->component_count * dst_elem;
            mglTessPackXFBFieldFromCarrier(
                field->gl_type, src_record + field->source_offset,
                dst_record + field->component_offset * dst_elem, field_bytes);
        }
    }
    return record_count;
}
