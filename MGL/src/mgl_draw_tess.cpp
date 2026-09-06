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
#include "glm_limits.h"
#include "mgl_draw_encode.h"
#include "mgl_index_buffer.h"
#include "mgl_program_resource.h"
#include "mgl_render.h"
#include "mgl_renderer_backend.h"
#include "mgl_shader_abi.h"
#include "mgl_shader_resource.h"
#include "mgl_buffer_slots.h"

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
