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
#include "mgl_draw_encode.h"
#include "mgl_index_buffer.h"
#include "mgl_render.h"
#include "mgl_shader_abi.h"

#include <cstdint>
#include <cstring>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
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
