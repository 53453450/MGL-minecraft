/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Tessellation.m
// Tessellation compute path (TCS/TES dispatch) extracted from MGLRenderer.m.
// GL_PATCHES draws run as consecutive Metal compute encoders: the TCS kernel
// writes per-patch output plus tess factors, then the TES kernel consumes them.

#import "MGLRenderer_Private.h"
#include "mgl_texture_sampler.h"
#include "mgl_renderer_ports.h"
#include "mgl_buffer_map.h"  /* buffer mapping entries (was MGLRenderer+Buffer.m) */
#include "mgl_draw_support.h"  /* draw-support predicates */  /* mglRendererProcessBuffer */
#import "MGLRenderer+Tessellation_Private.h"
#import "mgl_sampler_compat.h"
#import "mgl_trace_log.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_shader_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"
#include "mgl_draw_tess.h"
#include "mgl_tess_texture.h"  /* mglTessEnsureTextureMetalData */
#include "mgl_tess_compute_ops.h"  /* mglTessBindPointSizeParamsToComputeEncoder */
/* The stage-binding plan and the texture binding plan are C now (log 125). */
#include "mgl_tess_stage_bind.h"
#include "mgl_draw_issue.h"

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

enum {
    MGL_TESS_RESOURCE_STORAGE_SHARED = 0u,
    MGL_TESS_COMMAND_STATUS_NOT_ENQUEUED = 0u,
    MGL_TESS_COMMAND_STATUS_COMMITTED = 2u,
    MGL_TESS_TEXTURE_TYPE_CUBE = 5u,
    MGL_TESS_TEXTURE_TYPE_CUBE_ARRAY = 6u,
};

static id mglTessCreateBuffer(id device,
                              size_t length,
                              uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return NULL;
}

static uint64_t mglTessBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo((__bridge void *)buffer, &info) == 0
        ? info.length : 0u;
}

static void *mglTessBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer &&
        mglRenderGetBufferContents((__bridge void *)buffer,
                                      &contents, &length) == 0
        ? contents : NULL;
}

static void mglTessSetRenderVertexBuffer(id encoder,
                                         void *renderEncoderOwner,
                                         id buffer,
                                         size_t offset,
                                         size_t index)
{
    (void)encoder;
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglTessDrawPrimitives(id encoder,
                                  void *renderEncoderOwner,
                                  uint32_t type,
                                  size_t vertexStart,
                                  size_t vertexCount,
                                  size_t instanceCount,
                                  size_t baseInstance)
{
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)type,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)encoder;
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static bool mglTessAppendComputeResourceOp(
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    uint32_t kind,
    id resource,
    size_t offset,
    size_t index)
{
    if (!plan || kind > 3u) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        fprintf(stderr, "MGL TESS ERROR: compute binding op overflow (%u)",
              (unsigned)plan->binding_op_count);
        return false;
    }
    plan->binding_ops[plan->binding_op_count++] =
        (MGLRenderComputeBindingOp){
            .kind = kind,
            .index = (uint32_t)index,
            .offset = (uint64_t)offset,
            .buffer = (__bridge void *)resource,
            .bytes = NULL,
            .length = 0u,
        };
    if (resource && temporaries) [temporaries addObject:resource];
    return true;
}


static bool mglTessPlanBufferOrBind(
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    id encoder,
    id buffer,
    size_t offset,
    size_t index)
{
    (void)encoder;
    return mglTessAppendComputeResourceOp(
        plan, temporaries, 0u, buffer, offset, index);
}

/* mglRendererReadableBufferBytes moved to mgl_tess_dispatch.c (log 126) with
 * its only remaining callers. */

@implementation MGLRenderer (Tessellation)

/* Isolines / point_mode TES as a render vertex function: the CPU domain
 * expansion seeds TessCoord records once, then the render encoder replays a
 * per-patch drawPrimitives with the TES compiled as the vertex stage.  This
 * removes the per-patch TES compute dispatch and the compute→render encoder
 * switch of the compute expansion path. */
static size_t mglTESXFBVertexStride(const Program *program)
{
    return (size_t)mglRenderTESXFBVertexStride((const void *)program);
}




/* Isolines / point-mode TES: expand one vertex record per work item with
 * the AIR TES compute kernel (backend ABI: stage_in(24) factors(26)
 * patchInputs(27) stageOut(28) indirect(29), one dispatch per patch), then
 * rasterize through the passthrough vertex stage as lines / points.
 * Each patch owns a contiguous item span; per-patch item counts differ,
 * so the runtime dispatches per patch with the patch id and output base in
 * the contract buffer (slot 29: {patch_id, gl_in_vertices, items,
 * output_item_base}). */
- (BOOL)dispatchAIRTessEvalCompute:(GLMContext)glm_ctx
                          program:(Program *)tesProgram
                         contract:(const MGLAIRTessDrawContract *)contract
                       patchCount:(GLuint)patchCount
                    instanceCount:(GLsizei)instanceCount
                     baseInstance:(GLuint)baseInstance
{
    if (!tesProgram || !glm_ctx || !contract || patchCount == 0u ||
        instanceCount <= 0) {
        return false;
    }

    /* This draw takes the compute expansion path: the TES stage needs
     * isolated bindings and copy-backs (the kernel writes its outputs), even
     * when the program also carries the render-vertex function. */
    _tessellation.tessVertexRenderActive = 0;

    Shader *tesShader = tesProgram->shader_slots[_TESS_EVALUATION_SHADER];
    if (!mglTessStageHasCompiledFunction(
            tesShader ? 1 : 0,
            tesProgram->modules[_TESS_EVALUATION_SHADER].mtl_function ? 1 : 0)) {
        fprintf(stderr, "MGL TESS WARNING: TES program %u has no compiled function",
              tesProgram->name);
        return false;
    }

    void *tesPipelineHandle = NULL;
    char tesPipelineError[512] = {0};
    int tesPipelineResult = mglGetOrCreateProgramComputePipeline(
        tesProgram, _TESS_EVALUATION_SHADER, &tesPipelineHandle,
        tesPipelineError, sizeof(tesPipelineError));
    id tesPipeline =
        mglTessComputePipelineReady(tesPipelineResult, tesPipelineHandle ? 1 : 0)
            ? (__bridge_transfer id)tesPipelineHandle
            : NULL;
    if (!tesPipeline) {
        fprintf(stderr, "MGL TESS ERROR: failed to create TES compute pipeline for program %u: %s",
              tesProgram->name,
              tesPipelineError[0] ? tesPipelineError : "unknown error");
        return false;
    }

    /* Inputs: gl_in is the post-TCS control point stream (or the VS capture
     * when there is no TCS, which the draw path already aliased into
     * tcsOutputBuffer).  Factors and per-patch inputs come from the TCS
     * dispatch (or defaults). */
    id tcsOutputBuffer = (__bridge id)
        mglRendererBackendGetTcsOutputBuffer(_backend);
    id tessFactorBuffer = (__bridge id)
        mglRendererBackendGetCurrentTessFactorBuffer(_backend);
    id captureBuffer = (__bridge id)
        mglRendererBackendGetTessVertexCaptureBuffer(_backend);
    MGLTessEvalGlInPlan glInPlan = {0};
    if (!mglTessResolveEvalGlIn(
            contract, tcsOutputBuffer != NULL,
            (uint64_t)_tessellation.tcsOutputOffset,
            (uint64_t)_tessellation.tcsOutputStride,
            _tessellation.tcsOutVertices, captureBuffer != NULL,
            (uint64_t)_tessellation.tessVertexCaptureOffset,
            _tessellation.tessIndexedDraw ? 1 : 0,
            (uint32_t)_tessellation.tessInstanceRecords,
            (uint32_t)instanceCount, &glInPlan)) {
        fprintf(stderr, "MGL TESS ERROR: missing TES compute inputs program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    id glInBuffer = glInPlan.from_tcs ? tcsOutputBuffer : captureBuffer;
    size_t glInOffset = (size_t)glInPlan.gl_in_offset;
    size_t glInStride = (size_t)glInPlan.gl_in_stride;
    GLuint glInVertices = glInPlan.gl_in_vertices;
    if (!mglTessEvalInputsReady(glInBuffer != NULL, tessFactorBuffer != NULL)) {
        fprintf(stderr, "MGL TESS ERROR: missing TES compute inputs program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    id controlPointIndexBuffer =
        (__bridge id)
            mglRendererBackendGetTessControlPointIndexBuffer(_backend);
    if (!mglTessEvalIndexedGatherReady(
            _tessellation.tessIndexedDraw ? 1 : 0,
            controlPointIndexBuffer != NULL,
            (uint32_t)_tessellation.tessInstanceRecords)) {
        fprintf(stderr, "MGL TESS ERROR: indexed TES compute missing gather "
              "program=%u", (unsigned)tesProgram->name);
        return false;
    }
    const BOOL glInFromTCS = glInPlan.from_tcs != 0u;
    /* TCS currently expands one instance of control points / factors.
     * TES still loops instances for XFB/output bases.  Reusing instance-0
     * TCS outs is wrong when VS outputs vary by gl_InstanceID.  Until
     * per-instance TCS re-dispatch exists: one-shot log, and hard-fail when
     * MGL_TESS_MULTI_INSTANCE_ERROR is set. */
    if (mglTessMultiInstanceTCSReuseWarn(glInFromTCS ? 1 : 0,
                                         (int32_t)instanceCount)) {
        static BOOL s_multiInstanceTCSLogged = 0;
        if (!s_multiInstanceTCSLogged) {
            fprintf(stderr, "MGL TESS ERROR: multi-instance TES with TCS reuses "
                  "instance-0 control points (program=%u instances=%d); "
                  "set MGL_TESS_MULTI_INSTANCE_ERROR=1 to fail the draw",
                  (unsigned)tesProgram->name, (int)instanceCount);
            s_multiInstanceTCSLogged = 1;
        }
        if (mglTessMultiInstanceTCSReuseIsError(glInFromTCS ? 1 : 0,
                                                (int32_t)instanceCount)) {
            return false;
        }
    }
    const size_t glInInstanceStride =
        (size_t)glInPlan.gl_in_instance_stride;

    /* Compute per-patch item counts and the per-instance total. */
    const uint16_t *factorBytes =
        (const uint16_t *)mglTessBufferContents(tessFactorBuffer);
    MGLTessEvalComputePlan evalPlan = {0};
    if (!mglTessPlanEvalCompute(tesProgram, factorBytes,
                                mglTessBufferLength(tessFactorBuffer),
                                patchCount, (uint32_t)instanceCount,
                                &evalPlan)) {
        fprintf(stderr, "MGL TESS ERROR: TES compute plan failed program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    if (evalPlan.empty) {
        /* Every patch discarded (outer ≤ 0, e.g. CTS isolines with
         * outer=-1).  Empty expansion is success — do not raise
         * GL_INVALID_OPERATION. */
        return true;
    }
    const GLuint instanceCountU = evalPlan.instance_count;
    const GLuint itemsPerInstanceU = evalPlan.items_per_instance;
    size_t outStride = evalPlan.out_stride;
    size_t outSize = (size_t)evalPlan.out_size;
    id outBuffer = mglTessCreateBuffer(
        _device, outSize, MGL_TESS_RESOURCE_STORAGE_SHARED);
    void *outContents = mglTessBufferContents(outBuffer);
    if (!outContents) {
        fprintf(stderr, "MGL TESS ERROR: failed to allocate TES compute output "
              "(%lu bytes) program=%u",
              (unsigned long)outSize, (unsigned)tesProgram->name);
        return false;
    }
    if (mglTessSeedEvalOutputRecords(tesProgram, factorBytes, patchCount,
                                     instanceCountU, outContents, outSize,
                                     (uint32_t)outStride) != itemsPerInstanceU) {
        fprintf(stderr, "MGL TESS ERROR: TES domain seed failed program=%u",
              (unsigned)tesProgram->name);
        return false;
    }

    /* PASS 1: pre-resolve textures before opening the compute encoder. */
    if (mglTessMustEndRenderBeforeCompute(mglRenderEncoderOwnerHasCurrent(
            _renderPassManager->state->currentRenderEncoderOwner))) {
        [self endRenderEncoding];
    }
    MGLRenderCommandBufferState commandState = {0};
    const int hasCommandState = mglRenderCommandBufferOwnerHasState(
        _renderPassManager->state->currentCommandBufferOwner, &commandState);
    if (mglTessCommandBufferNeedsNew(hasCommandState, commandState.status)) {
        if (![self newCommandBuffer]) {
            fprintf(stderr, "MGL TESS ERROR: failed to create command buffer for TES compute" "\n");
            return false;
        }
    }

    MGLTessTextureBind tesTextureBinds[TEXTURE_UNITS * 2u];
    const uint32_t tesTextureBindCount = mglTessCollectTextureBinds(
        glm_ctx, tesProgram, _TESS_EVALUATION_SHADER, tesTextureBinds,
        (uint32_t)(sizeof(tesTextureBinds) / sizeof(tesTextureBinds[0])));
    if (!mglTessEnsureTextureMetalData((__bridge void *)self, tesTextureBinds,
                                       tesTextureBindCount, glm_ctx)) {
        return false;
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (!mglTessPrepareStageBufferBindings((__bridge void *)self,
                                           &stageBufferBindings,
                                           _TESS_EVALUATION_SHADER,
                                           &stageCopyBacks)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    MGLRenderComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = [NSMutableArray array];
    id computeEncoder = NULL;
    executionPlan.pipeline = (__bridge void *)tesPipeline;
    id patchInputs = (__bridge id)
        mglRendererBackendGetTcsPatchOutBuffer(_backend);
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder,
            tessFactorBuffer, 0u,
            MGL_AIR_TESS_SLOT_TESS_FACTOR) ||
        !mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder,
            patchInputs ? patchInputs : outBuffer, 0u,
            MGL_AIR_TESS_SLOT_PATCH_OUT) ||
        !mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder, outBuffer, 0u,
            MGL_AIR_TESS_SLOT_TCS_OUTPUT)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    if (!mglTessPlanTextureBinds((__bridge void *)self, tesTextureBinds,
                                 tesTextureBindCount, glm_ctx, &executionPlan,
                                 (__bridge void *)executionTemporaries)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    if (!mglTessBindPreparedStageBufferBindings(
            &stageBufferBindings, (__bridge void *)computeEncoder,
            &executionPlan, (__bridge void *)executionTemporaries)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    mglTessBindPointSizeParamsToComputeEncoder((__bridge void *)self,
                                               tesProgram,
                                               _TESS_EVALUATION_SHADER,
                                               &executionPlan,
                                               (__bridge void *)executionTemporaries);


    /* Transform-feedback stream (slot 31): the kernel writes complete stage
     * records. The renderer gathers selected varyings into the compact GL XFB
     * layout and copies only the prefix containing complete primitives. */
    TransformFeedback *xfbState = MGL_STATE(glm_ctx)->transform_feedback;
    Program *gsProgram =
        mglResolveProgramForStageFromState(glm_ctx, _GEOMETRY_SHADER);
    /* A monolithic VS+TCS+TES program resolves to itself for the GS stage
     * even with no GS attached.  Guard on the shader slot (same pattern as
     * mglTessClassifyDraw for tcs/tes) so has_gs / the TES→GS handoff and
     * mglTessPlanEvalAfterCompute only see a real geometry stage. */
    if (gsProgram && !gsProgram->shader_slots[_GEOMETRY_SHADER]) {
        gsProgram = NULL;
    }
    const bool xfbActive = mglTessEvalOwnsXFB(glm_ctx, gsProgram);
    id xfbTemporary = NULL;
    id xfbCopyDestination = NULL;
    Buffer *xfbDestination = NULL;
    size_t xfbCopyDestinationOffset = 0u;
    size_t xfbCompactStride = 0u;
    size_t xfbCopiedVertices = 0u;
    size_t xfbWrittenBytes = 0u;
    int xfbSizeOK = 0;
    if (xfbActive) {
        BufferBaseTarget *xfbSlot =
            &MGL_STATE(glm_ctx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
        size_t captureVertices = 0u;
        size_t requiredBytes = 0u;
        const size_t xfbSessionOffset = (size_t)mglXfbSessionOffsetOr(
            (uint64_t)xfbState->buffer_write_offsets[0], 0u);
        xfbCompactStride = mglTESXFBVertexStride(tesProgram);
        uint32_t captureVertsU = 0u;
        uint32_t requiredBytesU = 0u;
        const bool sizeOK = mglTessPlanEvalXfbCapture(
                                itemsPerInstanceU, instanceCountU,
                                (uint32_t)outStride, (uint32_t)xfbCompactStride,
                                &captureVertsU, &requiredBytesU) != 0;
        xfbSizeOK = sizeOK ? 1 : 0;
        captureVertices = captureVertsU;
        requiredBytes = requiredBytesU;
        (void)captureVertices;

        id xfbMTL = NULL;
        size_t visibleBytes = 0u;
        if (xfbSlot->buf) {
            if (mglRenderBufferNeedsCPUUpload(
                    xfbSlot->buf->size, xfbSlot->buf->data.dirty_bits)) {
                /* Consume CPU initialization before the XFB blit writes the
                 * same backing. Otherwise a later map can upload the stale
                 * shadow over the captured GPU data. */
                if (!mglRendererUpdateDirtyBuffer((__bridge void *)self, xfbSlot->buf)) {
                    [self clearStageBindingCopyBacks:&stageCopyBacks];
                    return false;
                }
            } else if (xfbSlot->buf->size == 0) {
                mglRenderClearEmptyBufferDirty(xfbSlot->buf);
            }
            if (!xfbSlot->buf->data.mtl_data) {
                mglRendererBindMTLBuffer((__bridge void *)self, xfbSlot->buf);
            }
            xfbMTL = (__bridge id)(xfbSlot->buf->data.mtl_data);
            if (xfbMTL) {
                BufferMap xfbMap = {0};
                xfbMap.buf = xfbSlot->buf;
                xfbMap.offset = xfbSlot->offset;
                xfbMap.size = xfbSlot->size;
                visibleBytes =
                    mglBufferMapVisibleBackingBytes(
                        &xfbMap, (size_t)mglTessBufferLength(xfbMTL));
            }
        }

        if (mglTessPlanEvalXFBSlot(xfbActive ? 1 : 0, xfbSizeOK) ==
            MGL_TESS_EVAL_XFB_CAPTURE) {
            const GLuint verticesPerPrimitive =
                mglTessVerticesPerPrimitive(tesProgram);
            MGLTessXFBDestPlan destPlan = {0};
            const int destPlanOK =
                mglTessPlanXFBDestination(
                    itemsPerInstanceU, instanceCountU,
                    (uint32_t)xfbCompactStride, verticesPerPrimitive,
                    (uint64_t)xfbSessionOffset, (int64_t)xfbSlot->offset,
                    (uint64_t)visibleBytes, &destPlan) &&
                destPlan.valid;
            const int destOK = mglTessEvalXFBDestReady(
                xfbMTL != NULL, xfbSlot->buf != NULL, destPlanOK);
            /* The AIR kernel writes full stage records (built-ins followed by
             * location-based user outputs). GL XFB is a compact stream of only
             * the selected varyings, so it can never target the GL range
             * directly. Gather the selected fields after the dispatch. */
            xfbTemporary = mglTessCreateBuffer(
                _device, requiredBytes, MGL_TESS_RESOURCE_STORAGE_SHARED);
            if (!xfbTemporary) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            if (!mglTessPlanBufferOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder, xfbTemporary, 0u,
                    MGL_AIR_TESS_SLOT_XFB_OUT)) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            if (destOK) {
                xfbCopiedVertices = destPlan.copied_vertices;
                xfbWrittenBytes = destPlan.written_bytes;
                xfbCopyDestination = xfbMTL;
                xfbCopyDestinationOffset = destPlan.destination_offset;
                xfbDestination = xfbSlot->buf;
            }
        }
    }
    if (mglTessPlanEvalXFBSlot(xfbActive ? 1 : 0, xfbSizeOK) ==
        MGL_TESS_EVAL_XFB_DUMMY) {
        /* The TES compute kernel always declares and writes the XFB stream
         * slot (31); bind a 1-byte dummy so the slot is never dangling when
         * GL feedback is inactive. */
        const uint64_t dummyBytes = mglTessDummyXfbBytes((uint64_t)outSize);
        void *cachedDummy = NULL;
        id xfbDummy = NULL;
        if (mglRendererBackendGetTessXfbDummyBuffer(
                _backend, dummyBytes, &cachedDummy) == 1) {
            xfbDummy = (__bridge id)cachedDummy;
        }
        if (!xfbDummy) {
            xfbDummy = mglTessCreateBuffer(
                _device, (size_t)dummyBytes, MGL_TESS_RESOURCE_STORAGE_SHARED);
            if (xfbDummy) {
                (void)mglRendererBackendPutTessXfbDummyBuffer(
                    _backend, (__bridge void *)xfbDummy);
            }
        }
        if (xfbDummy) {
            if (!mglTessPlanBufferOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder,
                    xfbDummy, 0u,
                    MGL_AIR_TESS_SLOT_XFB_OUT)) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
        }
    }

    const BOOL indexed = _tessellation.tessIndexedDraw;
    uint32_t gatherVerts = 0u;
    uint32_t gatherPrims = 0u;
    mglTessPlanEvalGather(indexed ? 1 : 0,
                          (uint32_t)_tessellation.tessInstanceRecords,
                          contract->patch_vertices, patchCount, &gatherVerts,
                          &gatherPrims);
    MGLTessEvalPerPatchDispatchSpec patchSpec;
    mglTessFillEvalPerPatchSpec(
        (__bridge void *)glInBuffer, (uint64_t)glInOffset,
        (uint64_t)glInInstanceStride,
        indexed ? (__bridge void *)controlPointIndexBuffer : NULL, gatherVerts,
        gatherPrims, indexed ? 1 : 0, (uint32_t)glInVertices, patchCount,
        instanceCountU, itemsPerInstanceU, &patchSpec);
    void *patchKeepAlive = NULL;
    if (!mglTessAppendEvalPerPatchDispatches(&executionPlan, tesProgram,
                                             factorBytes, &patchSpec,
                                             &patchKeepAlive)) {
        free(patchKeepAlive);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    if (patchKeepAlive) {
        NSData *keep = [[NSData alloc] initWithBytesNoCopy:patchKeepAlive
                                                    length:1
                                               deallocator:^(void *bytes,
                                                             size_t length) {
            (void)length;
            free(bytes);
        }];
        if (!keep) {
            free(patchKeepAlive);
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [executionTemporaries addObject:keep];
    }
    {
        MGLRenderCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)stageCopyBacks.slots,
            kMGLMaxBufferSlots, copyBackEntries, kMGLMaxBufferSlots);
        executionPlan.barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
        MGLRenderComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                _renderPassManager->state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount, 1u,
                &executionResult, executionError,
                sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            fprintf(stderr, "MGL TESS ERROR: C++ TES execution failed: %s",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [self clearStageBindingCopyBacks:&stageCopyBacks];
    }

    if (mglTessXFBCopyBackReady((uint64_t)xfbWrittenBytes,
                                xfbTemporary ? 1 : 0,
                                xfbDestination ? 1 : 0)) {
        const uint8_t *srcBase =
            (const uint8_t *)mglTessBufferContents(xfbTemporary);
        if (!srcBase) {
            fprintf(stderr, "MGL TESS XFB: missing temporary contents" "\n");
            return false;
        }
        const bool separateAttribs =
            mglXfbSeparateAttribs(
                tesProgram->transform_feedback_buffer_mode) != 0;
        if (separateAttribs) {
            /* One GL buffer binding per varying (GL 4.6 §11.1.3.2). */
            for (GLsizei varying = 0;
                 varying < tesProgram->transform_feedback_varying_count;
                 varying++) {
                if (!mglXfbVaryingSlotValid((uint32_t)varying)) {
                    break;
                }
                const char *name =
                    tesProgram->transform_feedback_varying_names[varying];
                uint32_t recordOffset = 0u;
                uint32_t fieldType = 0u;
                uint32_t fieldBytes = 0u;
                if (!mglTessResolveXFBSource(tesProgram, name, &recordOffset,
                                             &fieldType, &fieldBytes)) {
                    continue;
                }
                (void)recordOffset;
                (void)fieldType;
                BufferBaseTarget *slot =
                    &MGL_STATE(glm_ctx)
                         ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER]
                         .buffers[varying];
                Buffer *destBuf = slot->buf;
                if (!destBuf) {
                    continue;
                }
                if (mglRenderBufferNeedsCPUUpload(
                        destBuf->size, destBuf->data.dirty_bits)) {
                    if (!mglRendererUpdateDirtyBuffer((__bridge void *)self, destBuf)) {
                        return false;
                    }
                }
                if (!destBuf->data.mtl_data) {
                    mglRendererBindMTLBuffer((__bridge void *)self, destBuf);
                }
                id destMTL = (__bridge id)(destBuf->data.mtl_data);
                const uint64_t sessionOffset = mglXfbSessionOffsetOr(
                    (uint64_t)xfbState->buffer_write_offsets[varying], 0u);
                uint64_t visible = 0u;
                if (destMTL && slot->offset >= 0) {
                    BufferMap xfbMap = {0};
                    xfbMap.buf = destBuf;
                    xfbMap.offset = slot->offset;
                    xfbMap.size = slot->size;
                    visible = (uint64_t)mglBufferMapVisibleBackingBytes(
                        &xfbMap, (size_t)mglTessBufferLength(destMTL));
                }
                MGLXfbVsBufferDest dest = {0};
                if (!mglXfbPlanVsBufferDestOrUnbacked(
                        (uint32_t)xfbCopiedVertices, fieldBytes,
                        destMTL ? 1 : 0, slot->offset, sessionOffset, visible,
                        &dest) ||
                    dest.skip) {
                    continue;
                }
                size_t destOffset = (size_t)dest.destination_offset;
                size_t maxVerts = dest.written_records;
                size_t written = dest.written_bytes;
                uint8_t *packed = (uint8_t *)calloc(1u, written);
                if (!packed) {
                    fprintf(stderr, "MGL TESS XFB: OOM packing separate attrib %d",
                          (int)varying);
                    return false;
                }
                mglTessPackXFBSeparate(tesProgram, name, srcBase,
                                       (uint32_t)outStride, (uint32_t)maxVerts,
                                       packed);
                mglRendererBufferSubData(glm_ctx, destBuf, (GLintptr)destOffset,
                                         (GLsizeiptr)written, packed);
                if (destMTL) {
                    uint8_t *live =
                        (uint8_t *)mglTessBufferContents(destMTL);
                    if (live) {
                        memcpy(live + destOffset, packed, written);
                    }
                }
                if (mglXfbCPUShadowFits(destBuf->data.buffer_data ? 1 : 0,
                                        destBuf->size, (uint64_t)destOffset,
                                        (uint64_t)written)) {
                    memcpy((uint8_t *)destBuf->data.buffer_data + destOffset,
                           packed, written);
                }
                mglRenderMarkBufferCPUWrite(destBuf, (int64_t)destOffset,
                                            (int64_t)written);
                free(packed);
            }
        } else {
        uint8_t *packed = (uint8_t *)calloc(1u, xfbWrittenBytes);
        if (!packed) {
            fprintf(stderr, "MGL TESS XFB: missing temporary contents or OOM" "\n");
            return false;
        }
        mglTessPackXFBInterleaved(tesProgram, srcBase, (uint32_t)outStride,
                                  (uint32_t)xfbCopiedVertices, packed,
                                  (uint32_t)xfbCompactStride);
        mglRendererBufferSubData(glm_ctx, xfbDestination,
                                 xfbCopyDestinationOffset, xfbWrittenBytes,
                                 packed);
        /* Mirror into the live Metal allocation: SubData may land in a
         * snapshot while glMapBufferRange serves the CPU shadow. */
        if (xfbCopyDestination) {
            uint8_t *live = (uint8_t *)mglTessBufferContents(xfbCopyDestination);
            if (live) {
                memcpy(live + xfbCopyDestinationOffset, packed,
                       xfbWrittenBytes);
            }
        }
        if (mglXfbCPUShadowFits(xfbDestination->data.buffer_data ? 1 : 0,
                                xfbDestination->size,
                                (uint64_t)xfbCopyDestinationOffset,
                                (uint64_t)xfbWrittenBytes)) {
            memcpy((uint8_t *)xfbDestination->data.buffer_data +
                       xfbCopyDestinationOffset,
                   packed, xfbWrittenBytes);
        }
        mglRenderMarkBufferCPUWrite(xfbDestination,
                                    (int64_t)xfbCopyDestinationOffset,
                                    (int64_t)xfbWrittenBytes);
        free(packed);
        }
    }
    if (mglXfbShouldAdvanceWriteOffset(xfbActive ? 1 : 0,
                                       (uint64_t)xfbWrittenBytes)) {
        xfbState->buffer_write_offsets[0] = mglXfbAdvanceWriteOffset(
            xfbState->buffer_write_offsets[0], (uint64_t)xfbWrittenBytes);
    }

    /* Rasterize through the passthrough vertex stage, or hand the expanded
     * records to a following geometry shader (coverage VS+TC+TE+GS path). */
    const GLenum tessRasterMode = mglTessRasterGLMode(tesProgram);
    MGLTessRasterQueryPlan query = {0};
    mglTessPlanRasterQuery(tesProgram, (uint64_t)instanceCount,
                           (uint64_t)itemsPerInstanceU, xfbActive ? 1 : 0,
                           (uint64_t)xfbWrittenBytes,
                           (uint32_t)xfbCompactStride, &query);
    MGLTessEvalAfterComputePlan after = {0};
    if (!mglTessPlanEvalAfterCompute(gsProgram ? 1 : 0,
                                     MGL_STATE(glm_ctx)->caps.rasterizer_discard
                                         ? 1
                                         : 0,
                                     itemsPerInstanceU, instanceCountU,
                                     &after)) {
        return false;
    }
    if (after.action == MGL_TESS_AFTER_COMPUTE_GS) {
        if (after.gs_empty) {
            fprintf(stderr, "MGL TESS ERROR: TES→GS empty expansion program=%u",
                  (unsigned)tesProgram->name);
            return false;
        }
        GLsizei gsCount = (GLsizei)after.gs_vertex_count;
        _tessellation.pendingGSInputActive = 1;
        _tessellation.pendingGSInput = (__bridge_retained void *)outBuffer;
        _tessellation.pendingGSInputOffset = 0u;
        _tessellation.pendingGSInputStride = outStride;
        _tessellation.pendingGSVertexCount = gsCount;
        /* O1.4: single mglIssue/host path — no ObjC dual call. */
        const BOOL gsOK = mglDrawHostHandleGeometry(
                              (__bridge void *)self, glm_ctx, tessRasterMode, 0,
                              gsCount, 0, NULL, 0, 1, baseInstance,
                              "tessEvalToGeometry")
                              ? 1
                              : 0;
        if (_tessellation.pendingGSInput) {
            (void)CFBridgingRelease(_tessellation.pendingGSInput);
            _tessellation.pendingGSInput = NULL;
        }
        _tessellation.pendingGSInputActive = 0;
        _tessellation.pendingGSInputOffset = 0u;
        _tessellation.pendingGSInputStride = 0u;
        _tessellation.pendingGSVertexCount = 0;
        return gsOK;
    }
    if (after.action == MGL_TESS_AFTER_COMPUTE_DISCARD) {
        /* GL_RASTERIZER_DISCARD: no pixels by definition, so skip the
         * passthrough draw entirely, but the compute expansion already ran
         * and the primitive query must still count the generated
         * primitives (persistent query semantics). */
        _batching.currentCommandBufferHasWork = 1;
        mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
        return 1;
    }
    if (![self ensureAIRTessEvalPassthroughFunctionForProgram:tesProgram]) {
        fprintf(stderr, "MGL TESS ERROR: TES passthrough vertex unavailable program=%u",
              (unsigned)tesProgram->name);
        /* XFB capture already completed above; do not fail the draw and
         * leave transform feedback active when the test only needed feedback. */
        if (mglTessPassthroughFailIsXFBSuccess(xfbActive ? 1 : 0)) {
            mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
            return 1;
        }
        return false;
    }
    uint32_t primType = mglTessRasterPrimitiveType(tesProgram);

    _tessellation.tessComputeActive = 1;
    _tessellation.tessComputeProgram = tesProgram;
    BOOL stateReady = [self processGLState:true];
    if (!mglTessPassthroughRasterReady(
            stateReady ? 1 : 0,
            mglRenderEncoderOwnerHasCurrent(
                _renderPassManager->state->currentRenderEncoderOwner),
            mglDrawRasterizationIsEmpty((__bridge void *)self) ? 1 : 0)) {
        fprintf(stderr, "MGL TESS ERROR: TES compute raster skip program=%u stateReady=%d encoder=%d empty=%d clip0=%d",
              (unsigned)tesProgram->name,
              (int)stateReady,
              mglRenderEncoderOwnerHasCurrent(
                  _renderPassManager->state->currentRenderEncoderOwner),
              (int)mglDrawRasterizationIsEmpty((__bridge void *)self),
              ctx && MGL_STATE(ctx)->caps.clip_distances[0] ? 1 : 0);
        _tessellation.tessComputeActive = 0;
        _tessellation.tessComputeProgram = NULL;
        if (mglTessPassthroughFailIsXFBSuccess(xfbActive ? 1 : 0)) {
            mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
            /* Feedback already landed; returning 0 would raise
             * INVALID_OPERATION and skip the test's EndTransformFeedback. */
            return 1;
        }
        return 0;
    }

    mglDrawApplyPolygonOffset((__bridge void *)self, tessRasterMode);
    id encoder = NULL;
    for (GLsizei i = 0; i < instanceCount; i++) {
        size_t instanceOffset = (size_t)mglTessPassthroughInstanceOffset(
            (uint32_t)i, itemsPerInstanceU, (uint32_t)outStride);
        mglTessSetRenderVertexBuffer(
            encoder, _renderPassManager->state->currentRenderEncoderOwner,
            outBuffer, instanceOffset, 0u);
        mglTessDrawPrimitives(
            encoder, _renderPassManager->state->currentRenderEncoderOwner,
            primType, 0u, (size_t)itemsPerInstanceU, 1u,
            (size_t)baseInstance + (size_t)i);
    }
    _batching.currentCommandBufferHasWork = 1;
    mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
    _tessellation.tessComputeActive = 0;
    _tessellation.tessComputeProgram = NULL;
    return 1;
}

@end
