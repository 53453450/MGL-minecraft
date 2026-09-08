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

#include "glm_limits.h"

#include "error.h"
#include "mgl_air_gs_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_draw_encode.h"
#include "mgl_draw_tess.h"
#include "mgl_render.h"
#include "mgl_shader_abi.h"
#include "mgl_types_buffer.h"

#include <CoreFoundation/CoreFoundation.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);
extern "C" GLboolean mglHasActiveIndexedPrimitiveQuery(GLMContext ctx);
extern "C" GLboolean mglHasActivePrimitiveQuery(GLMContext ctx);
extern "C" GLboolean mglHasActiveGeometryShaderQuery(GLMContext ctx);
extern "C" int mglTessComputePipelineReady(int create_ok, int has_handle);
extern "C" int mglTessCommandBufferNeedsNew(int has_state, uint32_t status);

namespace {

struct OwnedList {
    const MGLGsMetalExpansionHostOps *ops = nullptr;
    std::vector<void *> items;
    void track(void *p)
    {
        if (p) {
            items.push_back(p);
        }
    }
    ~OwnedList()
    {
        if (!ops || !ops->release) {
            for (void *p : items) {
                if (p) {
                    CFRelease(p);
                }
            }
            return;
        }
        for (void *p : items) {
            if (p) {
                ops->release(p);
            }
        }
    }
};

static void gs_log(const MGLGsMetalExpansionHostOps *ops, const char *msg)
{
    if (ops && ops->log_diag && msg) {
        ops->log_diag(msg);
    } else if (msg) {
        std::fputs(msg, stderr);
        std::fputc('\n', stderr);
    }
}

static void gs_logf(const MGLGsMetalExpansionHostOps *ops, const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    gs_log(ops, buf);
}

static int gs_ops_ready(const MGLGsMetalExpansionHostOps *ops)
{
    return ops && ops->renderer && ops->create_buffer &&
           ops->create_buffer_with_bytes && ops->buffer_contents &&
           ops->buffer_length && ops->ensure_command_buffer && ops->bind_draw_textures &&
           ops->mtl_for_buffer && ops->fill_compute_bindings &&
           ops->command_buffer_owner && ops->recovery_owner &&
           ops->set_expansion && ops->mark_cb_has_work && ops->begin_blit &&
           ops->blit_copy && ops->end_blit && ops->process_gl_state &&
           ops->encoder_has_current && ops->raster_empty && ops->fully_culled &&
           ops->apply_polygon_offset && ops->binding_state_owner &&
           ops->rebind_fragment_after_gs && ops->encoder_owner &&
           ops->flush_command_buffer &&
           ops->record_queries && ops->set_vertex_buffer &&
           ops->draw_primitives && ops->draw_primitives_indirect;
}

} // namespace

extern "C" int mglDrawGsExecuteMetalExpansion(
    GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLenum indexType,
    const void *indices, GLint baseVertex, GLsizei instanceCount,
    GLuint baseInstance, const char *label, Program *program,
    GLenum gs_input_mode, GLenum gs_output_mode, uint32_t output_primitive,
    int indexed, void *gather_buf, const void *gparams_in,
    uint32_t gparams_bytes, const MGLGsComputeLayout *layout, void *input,
    uint64_t input_offset, Program *capture_vs, Program *capture_tes,
    uint32_t pending_stride, const MGLGsMetalExpansionHostOps *ops)
{
    (void)mode;
    (void)first;
    (void)count;
    (void)indexType;
    (void)indices;
    (void)baseVertex;
    (void)baseInstance;
    (void)gs_input_mode;
    (void)indexed;
    (void)pending_stride;
    (void)gparams_bytes;
    (void)capture_vs;
    (void)capture_tes;

    if (!gs_ops_ready(ops) || !ctx || !program || !layout || !gparams_in) {
        return 1;
    }

    OwnedList owned;
    owned.ops = ops;
    int rc = 1;

    MGLAIRGSGatherParams gparams;
    std::memcpy(&gparams, gparams_in, sizeof(gparams));
    const MGLGsComputeLayout gsLayout = *layout;
    const GLuint workItemCount = gsLayout.work_item_count;
    const uint64_t outputStride = gsLayout.output_stride;
    const uint64_t expandedVertices = gsLayout.expanded_vertices;
    const uint64_t recordsPerPrimitive = gsLayout.records_per_primitive;
    const uint32_t maxVertices =
        mglDrawGsMaxVerticesOut(program->geometry_vertices_out);
    const GLuint primitiveCount =
        gsLayout.work_item_count /
        (uint32_t)(instanceCount > 0 ? instanceCount : 1);
    (void)primitiveCount;

    void *pipelineHandle = NULL;
    char pipelineError[2048] = {0};
    const int pipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _GEOMETRY_SHADER, &pipelineHandle, pipelineError,
        sizeof(pipelineError));
    void *pipeline =
        mglTessComputePipelineReady(pipelineResult, pipelineHandle ? 1 : 0)
            ? pipelineHandle
            : NULL;
    if (!pipeline) {
        gs_logf(ops, "MGL GS ERROR: compute PSO failed program=%u: %s",
                (unsigned)program->name,
                pipelineError[0] ? pipelineError : "unknown error");
        ctx->active_state->dirty_bits = DIRTY_ALL;
        return 1;
    }
    owned.track(pipeline);

    MGLRenderCommandBufferState commandState = {0};
    const int hasCommandState = mglRenderCommandBufferOwnerHasState(
        ops->command_buffer_owner(ops->renderer), &commandState);
    if (mglTessCommandBufferNeedsNew(hasCommandState, commandState.status)) {
        if (!ops->ensure_command_buffer(ops->renderer)) {
            ctx->active_state->dirty_bits = DIRTY_ALL;
            return 1;
        }
    }

    const uint64_t outputSize = (uint64_t)gsLayout.output_bytes;
    void *output = ops->create_buffer(ops->renderer, outputSize);
    owned.track(output);
    if (std::getenv("MGL_GS_DIAG")) {
        const uint64_t mtlLen =
            output && ops->buffer_length ? ops->buffer_length(output) : 0u;
        gs_logf(ops,
                "MGL GS DIAG outputSize=%llu stride=%llu recordsPerPrim=%llu "
                "workItems=%u mtlLen=%llu",
                (unsigned long long)outputSize, (unsigned long long)outputStride,
                (unsigned long long)recordsPerPrimitive, (unsigned)workItemCount,
                (unsigned long long)mtlLen);
    }

    const uint64_t countsRecordBytes = MGL_AIR_GS_COUNTS_RECORD_BYTES;
    void *counts =
        ops->create_buffer(ops->renderer, (uint64_t)gsLayout.counts_bytes);
    owned.track(counts);
    if (!output || !counts || !ops->buffer_contents(output) ||
        !ops->buffer_contents(counts)) {
        ctx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(ctx, label ? label : "geometryDraw",
                         (GLenum)mglRenderErrorOutOfMemory());
        return 1;
    }
    std::memset(ops->buffer_contents(counts), 0,
                (size_t)workItemCount * (size_t)countsRecordBytes);
    std::memset(ops->buffer_contents(output), 0, (size_t)outputSize);
    mglDrawGsPresetCounts(ops->buffer_contents(counts), workItemCount);

    if (!ops->bind_draw_textures(ops->renderer, ctx)) {
        ctx->active_state->dirty_bits = DIRTY_ALL;
        return 1;
    }

    TransformFeedback *xfbState = MGL_STATE(ctx)->transform_feedback;
    const bool xfbActive =
        mglDrawGsXFBActive(xfbState != NULL, xfbState && xfbState->active,
                           xfbState && xfbState->paused) != 0;
    const int xfbDiag = std::getenv("MGL_GS_XFB_DIAG") != NULL;

    const bool gsSeparate = mglXfbSeparateAttribs(
                                program->transform_feedback_buffer_mode) != 0;
    (void)gsSeparate;

    MGLAIRGSXFBScatterParams scatterParams;
    std::memset(&scatterParams, 0, sizeof(scatterParams));
    uint32_t xfbBufferCount =
        mglDrawGsFillXFBScatterParams(xfbActive ? program : NULL, &scatterParams);
    uint64_t bufferCapBytes[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint64_t bufferPhysBase[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint64_t bufferDstOffset[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint64_t bufferRemaining[MGL_AIR_GS_MAX_STREAMS] = {0};
    void *bufferDstMTL[MGL_AIR_GS_MAX_STREAMS] = {NULL};

    void *xfbTemporary = NULL;
    void *xfbCaptureBuffer = NULL;
    void *xfbVisBuffer = NULL;
    void *xfbOffsetBuffer = NULL;
    void *xfbWrittenBuffer = NULL;
    void *scatterPipeline = NULL;

    if (xfbActive) {
        const uint32_t fieldCount = scatterParams.field_count;
        if (xfbDiag) {
            gs_logf(ops,
                    "MGL GS XFB DIAG fields=%u buffers=%u varyings=%d mode=0x%x",
                    fieldCount, xfbBufferCount,
                    program->transform_feedback_varying_count,
                    program->transform_feedback_buffer_mode);
            for (uint32_t f = 0u; f < fieldCount; f++) {
                gs_logf(ops, "  field[%u] buf=%u src=%u dst=%u bytes=%u", f,
                        scatterParams.fields[f].buffer_index,
                        scatterParams.fields[f].src_offset,
                        scatterParams.fields[f].dst_offset,
                        scatterParams.fields[f].byte_count);
            }
        }

        MGLGsXFBBufferBinding xfbBindings[MGL_AIR_GS_MAX_STREAMS];
        std::memset(xfbBindings, 0, sizeof(xfbBindings));
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            if (scatterParams.buffers[b].stride == 0u) {
                continue;
            }
            BufferBaseTarget *slot =
                &MGL_STATE(ctx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[b];
            if (!slot->buf) {
                if (xfbDiag) {
                    gs_logf(ops, "MGL GS XFB DIAG buffer[%u] no bound GL buffer",
                            b);
                }
                continue;
            }
            void *mtl = ops->mtl_for_buffer(ops->renderer, slot->buf);
            if (!mtl) {
                if (xfbDiag) {
                    gs_logf(ops, "MGL GS XFB DIAG buffer[%u] no MTL backing", b);
                }
                continue;
            }
            BufferMap map = {0};
            map.buf = slot->buf;
            map.offset = slot->offset;
            map.size = slot->size;
            const uint64_t mtlLen =
                ops->buffer_length ? ops->buffer_length(mtl)
                                   : (uint64_t)slot->buf->size;
            uint64_t visible = (uint64_t)mglBufferMapVisibleBackingBytes(
                &map, (size_t)mtlLen);
            uint64_t sessionOffset = mglXfbSessionOffsetOr(
                (uint64_t)xfbState->buffer_write_offsets[b], 0u);
            xfbBindings[b].bound = 1u;
            xfbBindings[b].slot_offset = slot->offset;
            xfbBindings[b].session_offset = sessionOffset;
            xfbBindings[b].visible_bytes = visible;
            bufferDstMTL[b] = mtl;
        }
        MGLGsXFBDestPlan destPlan = {0};
        mglDrawGsPlanXFBDestinations(&scatterParams, xfbBufferCount,
                                     (uint32_t)workItemCount,
                                     (uint32_t)expandedVertices, xfbBindings,
                                     &destPlan);
        uint64_t physTotal = destPlan.phys_total;
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            if (!destPlan.buffers[b].valid) {
                continue;
            }
            bufferRemaining[b] = destPlan.buffers[b].remaining;
            bufferDstOffset[b] = destPlan.buffers[b].dst_offset;
            bufferCapBytes[b] = destPlan.buffers[b].cap_bytes;
            bufferPhysBase[b] = destPlan.buffers[b].phys_base;
        }
        if (xfbDiag) {
            for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                gs_logf(ops,
                        "  buffer[%u] stride=%u cap=%u base=%u physTotal=%llu "
                        "dstMTL=%p",
                        b, scatterParams.buffers[b].stride,
                        scatterParams.buffers[b].capacity_bytes,
                        scatterParams.buffers[b].capture_base,
                        (unsigned long long)physTotal, bufferDstMTL[b]);
            }
        }
        mglDrawGsFillXFBScatterRuntime(
            &scatterParams, xfbBufferCount, (uint32_t)workItemCount,
            (uint32_t)outputStride, (uint32_t)recordsPerPrimitive,
            output_primitive);

        if (physTotal > 0u && xfbBufferCount > 0u) {
            xfbTemporary = ops->create_buffer(ops->renderer, physTotal);
            owned.track(xfbTemporary);
            if (xfbTemporary && ops->buffer_contents(xfbTemporary)) {
                std::memset(ops->buffer_contents(xfbTemporary), 0,
                            (size_t)physTotal);
                xfbCaptureBuffer = xfbTemporary;
            }
            const uint64_t visBytes =
                mglDrawGsXFBVisBytes((uint32_t)workItemCount);
            xfbVisBuffer = ops->create_buffer(ops->renderer, visBytes);
            xfbOffsetBuffer = ops->create_buffer(ops->renderer, visBytes);
            xfbWrittenBuffer = ops->create_buffer(ops->renderer, visBytes);
            owned.track(xfbVisBuffer);
            owned.track(xfbOffsetBuffer);
            owned.track(xfbWrittenBuffer);
            if (xfbVisBuffer && ops->buffer_contents(xfbVisBuffer)) {
                std::memset(ops->buffer_contents(xfbVisBuffer), 0,
                            (size_t)visBytes);
            }
            if (xfbWrittenBuffer && ops->buffer_contents(xfbWrittenBuffer)) {
                std::memset(ops->buffer_contents(xfbWrittenBuffer), 0,
                            (size_t)visBytes);
            }
            const MGLAuxShaderAsset *scatterAsset =
                mglAuxShaderAssetFind("gs_xfb_scatter");
            if (scatterAsset && scatterAsset->data) {
                void *scatterHandle = NULL;
                char scatterError[256] = {0};
                if (mglRenderGetOrCreateAuxComputePipelineFromMetallib(
                        scatterAsset->data, scatterAsset->size,
                        scatterAsset->hash, "mgl_gs_xfb_scatter",
                        MGL_RENDER_AUX_COMPUTE_GS_XFB_SCATTER, 0u,
                        &scatterHandle, scatterError,
                        sizeof(scatterError)) == 0 &&
                    scatterHandle) {
                    scatterPipeline = scatterHandle;
                    owned.track(scatterPipeline);
                } else {
                    gs_logf(ops, "MGL GS XFB ERROR: scatter pipeline failed: %s",
                            scatterError[0] ? scatterError : "unknown");
                }
            }
            if (!xfbTemporary || !xfbVisBuffer || !xfbOffsetBuffer ||
                !xfbWrittenBuffer || !scatterPipeline) {
                ctx->active_state->dirty_bits = DIRTY_ALL;
                mglDispatchError(ctx, label ? label : "geometryDraw",
                                 (GLenum)mglRenderErrorOutOfMemory());
                return 1;
            }
        }
    }

    uint64_t streamStride[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint64_t bufferStride[MGL_AIR_GS_MAX_STREAMS] = {0};
    for (uint32_t b = 0u; b < MGL_AIR_GS_MAX_STREAMS; b++) {
        streamStride[b] = scatterParams.buffers[b].stride;
        bufferStride[b] = scatterParams.buffers[b].stride;
    }
    (void)streamStride;
    const uint32_t gsStreamCount =
        mglDrawGsStreamCount(program->geometry_stream_count);

    MGLGsXFBDestPlan destForMeta = {0};
    uint32_t capBytesU32[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint32_t physBaseU32[MGL_AIR_GS_MAX_STREAMS] = {0};
    for (uint32_t b = 0u; b < MGL_AIR_GS_MAX_STREAMS; b++) {
        capBytesU32[b] = (uint32_t)bufferCapBytes[b];
        physBaseU32[b] = (uint32_t)bufferPhysBase[b];
    }
    mglDrawGsFillXFBDestForMeta(capBytesU32, physBaseU32, MGL_AIR_GS_MAX_STREAMS,
                                &destForMeta);
    MGLAIRGSXFBMeta xfbMeta;
    mglDrawGsFillXFBMetaFromDest(&scatterParams, &destForMeta, &xfbMeta);
    mglDrawGsClearXFBMetaIfNoCapture(xfbCaptureBuffer ? 1 : 0, &xfbMeta);
    void *xfbMetaBuf = ops->create_buffer_with_bytes(ops->renderer, &xfbMeta,
                                                     sizeof(xfbMeta));
    owned.track(xfbMetaBuf);
    if (!xfbMetaBuf) {
        ctx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(ctx, label ? label : "geometryDraw",
                         (GLenum)mglRenderErrorOutOfMemory());
        return 1;
    }

    MGLRenderComputeExecutionResult executionResult = {0};
    int gsQueryCountersReady = 0;
    MGLRenderComputeExecutionPlan executionPlan = {0};
    executionPlan.pipeline = pipeline;
    if (!mglDrawGsAppendCoreBindings(
            &executionPlan, input, input_offset, output, counts,
            gather_buf ? gather_buf : counts,
            xfbCaptureBuffer ? xfbCaptureBuffer : NULL, xfbMetaBuf,
            xfbVisBuffer ? xfbVisBuffer : counts, &gparams,
            (uint32_t)sizeof(gparams))) {
        ctx->active_state->dirty_bits = DIRTY_ALL;
        mglDispatchError(ctx, label ? label : "geometryDraw",
                         (GLenum)mglRenderErrorOutOfMemory());
        return 1;
    }
    if (std::getenv("MGL_GS_DIAG")) {
        Program *gp =
            mglResolveProgramForStageFromState(ctx, _GEOMETRY_SHADER);
        gs_logf(ops, "MGL GS DIAG GS uniform-constant resources: %u",
                gp ? gp->shader_resources_list[_GEOMETRY_SHADER]
                         [_UNIFORM_CONSTANT_RES]
                             .count
                   : 0u);
    }
    if (ops->gpu_capture_start) {
        ops->gpu_capture_start(ops->renderer);
    }

    MGLRenderCopyBackEntry copyBackEntries[31] = {{0}};
    uint32_t copyBackEntryCount = 0u;
    if (!ops->fill_compute_bindings(ops->renderer, ctx, &executionPlan,
                                    copyBackEntries, 31u,
                                    &copyBackEntryCount)) {
        ctx->active_state->dirty_bits = DIRTY_ALL;
        return 1;
    }
    if (std::getenv("MGL_GS_DIAG")) {
        for (uint32_t bi = 0; bi < executionPlan.binding_op_count; bi++) {
            const MGLRenderComputeBindingOp *op = &executionPlan.binding_ops[bi];
            gs_logf(ops,
                    "MGL GS DIAG binding[%u] kind=%u slot=%u offset=%llu "
                    "buffer=%p",
                    (unsigned)bi, (unsigned)op->kind, (unsigned)op->index,
                    (unsigned long long)op->offset, op->buffer);
        }
    }

    executionPlan.dispatch = (MGLRenderComputePlan){
        .dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT,
        .groups_x = (uint32_t)workItemCount,
        .groups_y = 1u,
        .groups_z = 1u,
        .local_x = 1u,
        .local_y = 1u,
        .local_z = 1u,
    };
    executionPlan.barrier_scope =
        copyBackEntryCount ? MGL_RENDER_COMPUTE_BARRIER_BUFFERS
                           : MGL_RENDER_COMPUTE_BARRIER_NONE;
    const int requireCPUVisibility = mglDrawGsNeedCPUVisibility(
        xfbActive ? 1 : 0,
        (mglHasActiveIndexedPrimitiveQuery(ctx) ||
         mglHasActivePrimitiveQuery(ctx) ||
         mglHasActiveGeometryShaderQuery(ctx))
            ? 1
            : 0);
    const int gsDiagnostic = std::getenv("MGL_GS_DIAG") != NULL;
    char executionError[256] = {0};
    if (mglRenderExecuteComputeExecutionPlan(
            ops->command_buffer_owner(ops->renderer),
            ops->recovery_owner(ops->renderer), &executionPlan, copyBackEntries,
            copyBackEntryCount,
            (requireCPUVisibility || gsDiagnostic) ? 1u : 0u, &executionResult,
            executionError, sizeof(executionError)) != 0) {
        if (executionResult.transaction.device_reset_requested &&
            ops->note_device_reset) {
            ops->note_device_reset(ops->renderer);
        }
        gs_logf(ops, "MGL GS ERROR: C++ execution transaction failed: %s",
                executionError[0] ? executionError : "unknown error");
        ctx->active_state->dirty_bits = DIRTY_ALL;
        return 1;
    }
    gsQueryCountersReady = executionResult.transaction.waited != 0;

    ops->set_expansion(ops->renderer, program, 1,
                       mglDrawGsLastDrawMode(output_primitive));
    ctx->active_state->dirty_bits = DIRTY_ALL;

    uint64_t bufferWritten[MGL_AIR_GS_MAX_STREAMS] = {0};
    if (xfbActive && xfbVisBuffer && xfbOffsetBuffer && xfbWrittenBuffer &&
        scatterPipeline && xfbCaptureBuffer &&
        ops->buffer_contents(xfbVisBuffer) &&
        ops->buffer_contents(xfbOffsetBuffer) &&
        ops->buffer_contents(xfbWrittenBuffer)) {
        uint32_t *vis = (uint32_t *)ops->buffer_contents(xfbVisBuffer);
        uint32_t *offsets = (uint32_t *)ops->buffer_contents(xfbOffsetBuffer);
        if (xfbDiag && counts && ops->buffer_contents(counts) && output &&
            ops->buffer_contents(output)) {
            const uint32_t *cw =
                (const uint32_t *)ops->buffer_contents(counts);
            const float *outPos =
                (const float *)ops->buffer_contents(output);
            outPos += (MGL_AIR_GS_HEADER_RECORDS * outputStride) / sizeof(float);
            gs_logf(ops,
                    "MGL GS XFB DIAG pass1 vertex_count=%u emit=%u vis[0]=%u "
                    "out.pos={%g,%g,%g,%g}",
                    cw[0], cw[MGL_AIR_GS_COUNTS_ARGS_WORDS + 2u], vis[0],
                    outPos[0], outPos[1], outPos[2], outPos[3]);
        }
        mglDrawGsExclusivePrefixSum(vis, offsets, (uint32_t)workItemCount,
                                    xfbBufferCount);
        MGLRenderComputeExecutionPlan scatterPlan = {0};
        if (!mglDrawGsFillXFBScatterPlan(
                &scatterPlan, scatterPipeline, &scatterParams,
                (uint32_t)sizeof(scatterParams), xfbVisBuffer, xfbOffsetBuffer,
                output, xfbCaptureBuffer, xfbWrittenBuffer,
                (uint32_t)workItemCount)) {
            ctx->active_state->dirty_bits = DIRTY_ALL;
            mglDispatchError(ctx, label ? label : "geometryDraw",
                             (GLenum)mglRenderErrorOutOfMemory());
            ops->set_expansion(ops->renderer, NULL, 0, 0);
            return 1;
        }
        MGLRenderComputeExecutionResult scatterResult = {0};
        char scatterError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                ops->command_buffer_owner(ops->renderer),
                ops->recovery_owner(ops->renderer), &scatterPlan, NULL, 0u, 1u,
                &scatterResult, scatterError, sizeof(scatterError)) != 0) {
            if (scatterResult.transaction.device_reset_requested &&
                ops->note_device_reset) {
                ops->note_device_reset(ops->renderer);
            }
            gs_logf(ops, "MGL GS XFB ERROR: scatter transaction failed: %s",
                    scatterError[0] ? scatterError : "unknown error");
            ctx->active_state->dirty_bits = DIRTY_ALL;
            ops->set_expansion(ops->renderer, NULL, 0, 0);
            return 1;
        }
        const uint32_t *written =
            (const uint32_t *)ops->buffer_contents(xfbWrittenBuffer);
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            bufferWritten[b] = mglDrawGsReduceBufferWritten(
                written, (uint32_t)workItemCount, b);
        }
    }

    uint64_t queryGenerated = 0u;
    uint64_t queryWritten = 0u;
    const MGLAIRGSXFBMeta *queryMeta = NULL;
    if (xfbActive && xfbMetaBuf && ops->buffer_contents(xfbMetaBuf)) {
        const MGLAIRGSXFBMeta *meta =
            (const MGLAIRGSXFBMeta *)ops->buffer_contents(xfbMetaBuf);
        queryMeta = meta;
        if (xfbTemporary) {
            void *xfbBlit = NULL;
            uint8_t *xfbTempBytes =
                (uint8_t *)ops->buffer_contents(xfbTemporary);
            if (xfbTempBytes) {
                for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                    if (scatterParams.buffers[b].stride == 0u) {
                        continue;
                    }
                    uint64_t region = bufferWritten[b];
                    if (region == 0u) {
                        continue;
                    }
                    mglXfbDecodeIntCarriersInBytes(
                        xfbTempBytes + bufferPhysBase[b], region,
                        scatterParams.buffers[b].stride, program, b,
                        _GEOMETRY_SHADER);
                }
            }
            for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                if (!mglDrawGsXFBCopyReady(bufferDstMTL[b] ? 1 : 0,
                                           scatterParams.buffers[b].stride,
                                           bufferWritten[b])) {
                    continue;
                }
                uint64_t copyBytes = mglDrawGsClampXFBCopy(
                    bufferWritten[b], bufferRemaining[b]);
                if (copyBytes == 0u) {
                    continue;
                }
                if (!xfbBlit) {
                    xfbBlit = ops->begin_blit(ops->renderer);
                    if (!xfbBlit) {
                        ops->set_expansion(ops->renderer, NULL, 0, 0);
                        ctx->active_state->dirty_bits = DIRTY_ALL;
                        return 1;
                    }
                }
                ops->blit_copy(xfbBlit, xfbTemporary, bufferPhysBase[b],
                               bufferDstMTL[b], bufferDstOffset[b], copyBytes);
                BufferBaseTarget *slot =
                    &MGL_STATE(ctx)
                         ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER]
                         .buffers[b];
                if (slot->buf) {
                    slot->buf->ever_written =
                        (GLboolean)mglRenderGLBoolean(1);
                    if (xfbTempBytes &&
                        mglXfbCPUShadowFits(
                            slot->buf->data.buffer_data ? 1 : 0,
                            slot->buf->size, bufferDstOffset[b], copyBytes)) {
                        std::memcpy(
                            (uint8_t *)slot->buf->data.buffer_data +
                                bufferDstOffset[b],
                            xfbTempBytes + bufferPhysBase[b],
                            (size_t)copyBytes);
                        mglRenderMarkBufferCPUWrite(
                            slot->buf, (int64_t)bufferDstOffset[b],
                            (int64_t)copyBytes);
                    }
                    uint8_t *liveBase =
                        (uint8_t *)ops->buffer_contents(bufferDstMTL[b]);
                    if (xfbTempBytes && liveBase) {
                        MGLRenderBufferInfo liveInfo = {0};
                        if (mglRenderGetBufferInfo(bufferDstMTL[b],
                                                   &liveInfo) == 0 &&
                            bufferDstOffset[b] + copyBytes <= liveInfo.length) {
                            std::memcpy(liveBase + bufferDstOffset[b],
                                        xfbTempBytes + bufferPhysBase[b],
                                        (size_t)copyBytes);
                        }
                    }
                }
                xfbState->buffer_write_offsets[b] = mglXfbAdvanceWriteOffset(
                    xfbState->buffer_write_offsets[b], copyBytes);
            }
            if (xfbBlit) {
                ops->end_blit(xfbBlit);
            }
        }
        queryWritten = mglDrawGsQueryWritten(output_primitive,
                                             scatterParams.buffers[0].stride,
                                             bufferWritten[0]);
    }
    if (!queryMeta && xfbMetaBuf && ops->buffer_contents(xfbMetaBuf) &&
        (mglHasActiveIndexedPrimitiveQuery(ctx) ||
         mglHasActivePrimitiveQuery(ctx) ||
         mglHasActiveGeometryShaderQuery(ctx))) {
        queryMeta =
            (const MGLAIRGSXFBMeta *)ops->buffer_contents(xfbMetaBuf);
    }
    if (gsQueryCountersReady && counts && ops->buffer_contents(counts)) {
        queryGenerated = mglDrawGsReduceGeneratedPrimitives(
            gs_output_mode, (uint32_t)workItemCount, maxVertices,
            (const uint32_t *)ops->buffer_contents(counts), queryMeta);
    } else if (queryMeta) {
        queryGenerated = mglDrawGsReduceGeneratedPrimitives(
            gs_output_mode, (uint32_t)workItemCount, maxVertices, NULL,
            queryMeta);
    }

    auto finish_queries_and_clear = [&](void) {
        ops->record_queries(ctx, queryGenerated, queryWritten,
                            xfbActive ? 1 : 0, queryMeta, gsStreamCount,
                            bufferWritten, bufferStride, workItemCount);
        ops->set_expansion(ops->renderer, NULL, 0, 0);
        ctx->active_state->dirty_bits = DIRTY_ALL;
    };

    if (mglDrawGsSkipRaster(
            xfbActive ? 1 : 0,
            MGL_STATE(ctx)->caps.rasterizer_discard ? 1 : 0)) {
        ops->mark_cb_has_work(ops->renderer);
        finish_queries_and_clear();
        return 1;
    }
    if (std::getenv("MGL_GS_DIAG")) {
        gs_logf(ops, "MGL GS DIAG rasterize-check empty=%d culled=%d enc=%d",
                ops->raster_empty(ops->renderer),
                ops->fully_culled(ops->renderer, gs_output_mode),
                ops->encoder_has_current(ops->renderer));
    }
    if (!mglDrawGsPassthroughRasterReady(
            ops->process_gl_state(ops->renderer),
            ops->encoder_has_current(ops->renderer),
            ops->raster_empty(ops->renderer),
            ops->fully_culled(ops->renderer, gs_output_mode))) {
        if (xfbActive || mglHasActiveIndexedPrimitiveQuery(ctx) ||
            mglHasActivePrimitiveQuery(ctx) ||
            mglHasActiveGeometryShaderQuery(ctx)) {
            ops->mark_cb_has_work(ops->renderer);
            finish_queries_and_clear();
        } else {
            ops->set_expansion(ops->renderer, NULL, 0, 0);
            ctx->active_state->dirty_bits = DIRTY_ALL;
        }
        return 1;
    }

    if (!std::getenv("MGL_ABLATE_GS_REBIND")) {
        /* A1: clear stale fragment bindings in C++; ObjC only rebinds. */
        void *binding = ops->binding_state_owner(ops->renderer);
        if (binding) {
            for (uint32_t slot = 0u; slot < 31u; slot++) {
                (void)mglRenderBindingClearFragmentBuffer(binding, slot);
            }
            const uint32_t tex_slots = (uint32_t)TEXTURE_UNITS;
            for (uint32_t slot = 0u; slot < tex_slots; slot++) {
                (void)mglRenderBindingClearFragmentTexture(binding, slot);
            }
        }
        if (!ops->rebind_fragment_after_gs(ops->renderer, ctx)) {
            ops->set_expansion(ops->renderer, NULL, 0, 0);
            ctx->active_state->dirty_bits = DIRTY_ALL;
            return 1;
        }
    }
    ops->apply_polygon_offset(ops->renderer, gs_output_mode);
    if (std::getenv("MGL_SYNC_AFTER_GS")) {
        ops->flush_command_buffer(ops->renderer, 1);
        gs_log(ops, "MGL GS sync: flushed after compute");
    }
    if (std::getenv("MGL_GS_DIAG") && ops->buffer_contents(counts)) {
        const uint32_t *cw = (const uint32_t *)ops->buffer_contents(counts);
        gs_logf(ops,
                "MGL GS DIAG draw counts w0..6: %u %u %u %u %u %u %u "
                "outputBuf=%p",
                cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], output);
    }

    void *enc = ops->encoder_owner(ops->renderer);
    if (std::getenv("MGL_GS_SINGLE_DRAW")) {
        const uint32_t totalVerts =
            (uint32_t)(workItemCount * recordsPerPrimitive) -
            MGL_AIR_GS_HEADER_RECORDS;
        uint32_t *cw1 = (uint32_t *)ops->buffer_contents(counts);
        cw1[0] = totalVerts;
        cw1[1] = 1u;
        ops->set_vertex_buffer(enc, output,
                               MGL_AIR_GS_HEADER_RECORDS * outputStride, 0u);
        ops->draw_primitives(enc, output_primitive, 0u, totalVerts, 1u, 0u);
    } else if (std::getenv("MGL_GS_COPY_DRAW")) {
        const uint8_t *src = (const uint8_t *)ops->buffer_contents(output);
        for (GLuint w = 0u; w < workItemCount; w++) {
            uint64_t srcOff =
                ((uint64_t)w * recordsPerPrimitive + MGL_AIR_GS_HEADER_RECORDS) *
                outputStride;
            uint64_t bytes = 9u * outputStride;
            void *sub =
                ops->create_buffer_with_bytes(ops->renderer, src + srcOff, bytes);
            owned.track(sub);
            const uint32_t *cwv =
                (const uint32_t *)ops->buffer_contents(counts);
            ops->set_vertex_buffer(enc, sub, 0u, 0u);
            ops->draw_primitives(
                enc, output_primitive, 0u,
                cwv ? cwv[w * MGL_AIR_GS_COUNTS_RECORD_WORDS] : 0u, 1u, 0u);
        }
    } else {
        const bool gsDiagEncode =
            std::getenv("MGL_GS_ONLY_PRIM") || std::getenv("MGL_GS_DRAW_OFFSET") ||
            std::getenv("MGL_GS_VSTART_DRAW") || std::getenv("MGL_GS_BIND_INPUT") ||
            std::getenv("MGL_GS_DIRECT_DRAW") || std::getenv("MGL_GS_DRAW_VCOUNT") ||
            std::getenv("MGL_GS_REVERSE_DRAW") || std::getenv("MGL_GS_DIAG");
        if (!gsDiagEncode) {
            const MGLGsPassthroughEncodeState gsEnc = {
                .encoder_owner = enc,
                .output_buffer = output,
                .counts_buffer = counts,
                .output_primitive = output_primitive,
                .work_item_count = (uint32_t)workItemCount,
                .records_per_primitive = (uint32_t)recordsPerPrimitive,
                .output_stride = (uint32_t)outputStride,
                .counts_record_bytes = (uint32_t)countsRecordBytes,
            };
            mglDrawGsEncodePassthrough(&gsEnc);
        } else {
            const char *onlyPrim = std::getenv("MGL_GS_ONLY_PRIM");
            for (GLuint iter = 0u; iter < workItemCount; iter++) {
                GLuint primitive = std::getenv("MGL_GS_REVERSE_DRAW")
                                       ? (workItemCount - 1u - iter)
                                       : iter;
                if (onlyPrim && (GLint)primitive != std::atoi(onlyPrim)) {
                    continue;
                }
                const char *offOverride = std::getenv("MGL_GS_DRAW_OFFSET");
                uint64_t offset =
                    ((uint64_t)primitive * recordsPerPrimitive +
                     MGL_AIR_GS_HEADER_RECORDS) *
                    outputStride;
                if (offOverride) {
                    offset = (uint64_t)std::atol(offOverride);
                }
                if (std::getenv("MGL_GS_VSTART_DRAW")) {
                    uint32_t *cwStart =
                        (uint32_t *)ops->buffer_contents(counts);
                    cwStart[primitive * MGL_AIR_GS_COUNTS_RECORD_WORDS + 2] =
                        primitive * (uint32_t)recordsPerPrimitive + 2u;
                    offset = 0u;
                }
                void *ptvsSource = output;
                uint64_t ptvsOffset = offset;
                if (std::getenv("MGL_GS_BIND_INPUT")) {
                    ptvsSource = input;
                    if (!std::getenv("MGL_GS_DRAW_OFFSET")) {
                        ptvsOffset = input_offset;
                    }
                    offset = 0u;
                }
                ops->set_vertex_buffer(enc, ptvsSource, ptvsOffset, 0u);
                if (std::getenv("MGL_GS_DIRECT_DRAW")) {
                    const uint32_t *cw2 =
                        (const uint32_t *)ops->buffer_contents(counts);
                    ops->draw_primitives(
                        enc, output_primitive, 0u,
                        cw2 ? cw2[primitive * MGL_AIR_GS_COUNTS_RECORD_WORDS]
                            : 0u,
                        1u, 0u);
                } else if (std::getenv("MGL_GS_DRAW_VCOUNT")) {
                    ops->draw_primitives(
                        enc, output_primitive, 0u,
                        (uint32_t)std::atol(std::getenv("MGL_GS_DRAW_VCOUNT")),
                        1u, 0u);
                } else {
                    ops->draw_primitives_indirect(
                        enc, output_primitive, counts,
                        offOverride ? 0u
                                    : (uint64_t)primitive * countsRecordBytes);
                }
            }
        }
    }

    if (std::getenv("MGL_GS_POST_DIAG")) {
        ops->flush_command_buffer(ops->renderer, 1);
        gs_logf(ops, "MGL GS POST-DIAG outputStride=%llu recordsPerPrim=%llu",
                (unsigned long long)outputStride,
                (unsigned long long)recordsPerPrimitive);
    }
    ops->mark_cb_has_work(ops->renderer);
    if (ops->gpu_capture_stop && std::getenv("MGL_GPU_CAPTURE")) {
        ops->flush_command_buffer(ops->renderer, 1);
        ops->gpu_capture_stop(ops->renderer);
        gs_log(ops, "MGL GPU capture stopped");
    }
    finish_queries_and_clear();
    rc = 1;
    return rc;
}
