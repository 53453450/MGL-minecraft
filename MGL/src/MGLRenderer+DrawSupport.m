/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+DrawSupport.m
// Draw validation, element-buffer resolution and rasterization helper
// methods extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+DrawSupportUtil.h"
#import "mgl_frame_activity.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_shader_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_program_reflection.h"
#include "mgl_draw_issue.h"
#include "mgl_draw_gs.h"
#include "mgl_draw_tess.h"
#include "mgl_index_buffer.h"
#include "mgl_draw_mode.h"
#include "mgl_draw_encode.h"

void *mglDrawSupportBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0;
    if (!buffer || mglRenderGetBufferContents(
            (__bridge void *)buffer, &contents, &length) != 0) {
        return NULL;
    }
    return contents;
}

/* AIR stage-out / GS scatter slots carry integers as SIToFP/UIToFP floats.
 * GL transform-feedback stores native int/uint bits — decode before the
 * compact XFB image is published to the GL buffer / CPU shadow.  Packing
 * lives in mglTessPackXFBFieldFromCarrier / mglXfbDecodeIntCarriersInBytes. */

uint64_t mglDrawSupportBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(
        (__bridge void *)buffer, &info) == 0 ? info.length : 0u;
}

MGLRenderTextureInfo mglDrawSupportTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

BOOL mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext)
{
    if (!encodeContext) return NO;
    return mglRenderEncoderOwnerHasCurrent(
        encodeContext->render_encoder_owner) == 1;
}


bool mglGeometryGatherIndices(const uint8_t *indexBytes,
                                     GLenum indexType,
                                     GLsizei count,
                                     int32_t baseVertex,
                                     bool restartEnabled,
                                     uint32_t restartIndex,
                                     uint32_t inputVertices,
                                     uint32_t **outGather,
                                     uint32_t *outGatherCount,
                                     uint32_t *outPrimitiveCount,
                                     uint32_t *outMaxIndex)
{

    (void)baseVertex; /* gather stores raw index values (vertex_id) */
    if (!outGather || !outGatherCount || !outPrimitiveCount || !outMaxIndex) {
        return false;
    }
    const uint32_t elemBytes = mglRenderGLIndexElementSize((uint64_t)indexType);
    MGLRenderGeometryGatherResult result = {0};
    if (mglRenderGeometryGatherIndices(
            indexBytes, elemBytes, (uint32_t)count,
            restartEnabled ? 1 : 0, restartIndex, inputVertices,
            &result) != 0) {
        return false;
    }
    *outGather = result.gather;
    *outGatherCount = result.gather_count;
    *outPrimitiveCount = result.primitive_count;
    *outMaxIndex = result.max_index;
    return true;
}

id mglDrawSupportCreateBuffer(
    id device,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

id mglDrawSupportCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

id mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

void mglDrawSupportBlitCopyBuffer(id encoder,
                                         id source,
                                         NSUInteger sourceOffset,
                                         id destination,
                                         NSUInteger destinationOffset,
                                         NSUInteger size)
{
    (void)mglRenderBlitCopyBuffer(
        (__bridge void *)encoder, (__bridge void *)source, sourceOffset,
        (__bridge void *)destination, destinationOffset, size);
}

void mglDrawSupportEndBlitEncoder(id encoder)
{
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

void mglDrawSupportDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger indexCount,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    (void)mglRenderEncodeDrawForRenderEncoderOwner(renderEncoderOwner,
        &(MGLRenderDrawPlan){
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)MGL_DRAW_INDEX_UINT32,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        }, NULL, 0);
}

void mglDrawSupportDrawPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger vertexStart,
    NSUInteger vertexCount,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

void mglDrawSupportDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    id indirectBuffer,
    NSUInteger indirectBufferOffset)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY_INDIRECT,
            .primitive_type = (uint32_t)primitiveType,
            .indirect_buffer = (__bridge void *)indirectBuffer,
            .indirect_buffer_offset = indirectBufferOffset,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

void mglRendererBindCullDistanceEmu(void *renderer, const void *encode_context,
                                    GLenum mode, GLuint first_vertex,
                                    const uint32_t *explicit_vertices,
                                    uint32_t explicit_vertex_count)
{
    if (!renderer || !encode_context) {
        return;
    }
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    [host bindCullDistanceEmulationBuffers:mode
                                firstVertex:first_vertex
                           explicitVertices:explicit_vertices
                         explicitVertexCount:explicit_vertex_count
                              encodeContext:(const MGLEncodeContext *)encode_context];
}

id mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateComputeEncoderBorrowed(
        commandBufferOwner);
}

void mglDrawSupportSetComputePipeline(
    id encoder,
    id pipeline)
{
    (void)mglRenderSetComputePipelineState((__bridge void *)encoder,
                                              (__bridge void *)pipeline);
}

void mglDrawSupportSetComputeBuffer(
    id encoder,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetComputeBuffer((__bridge void *)encoder,
                                       (__bridge void *)buffer, offset,
                                       (uint32_t)index);
}

void mglDrawSupportSetComputeBytes(
    id encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetComputeBytes((__bridge void *)encoder, bytes,
                                      length, (uint32_t)index);
}

void mglDrawSupportDispatchCompute(
    id encoder,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsX,
    uint32_t threadsY,
    uint32_t threadsZ)
{
    (void)mglRenderDispatchCompute(
        (__bridge void *)encoder, groupsX, groupsY, groupsZ,
        threadsX, threadsY, threadsZ);
}

void mglDrawSupportEndComputeEncoder(
    id encoder)
{
    (void)mglRenderEndComputeEncoder((__bridge void *)encoder);
}

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx,
                                               GLuint64 generated,
                                               GLuint64 written);
extern void mglRecordActivePrimitiveQueryDrawIndexed(GLMContext ctx,
                                                      GLuint index,
                                                      GLuint64 generated,
                                                      GLuint64 written);
extern void mglRecordActiveGeometryShaderQueryDraw(GLMContext ctx,
                                                    GLuint64 invocations,
                                                    GLuint64 primitives);
extern GLboolean mglHasActiveIndexedPrimitiveQuery(GLMContext ctx);
extern GLboolean mglHasActivePrimitiveQuery(GLMContext ctx);
extern GLboolean mglHasActiveGeometryShaderQuery(GLMContext ctx);

void mglRecordGeometryPrimitiveQueries(
    GLMContext ctx,
    GLuint64 generatedStream0,
    GLuint64 writtenStream0,
    BOOL xfbActive,
    const MGLAIRGSXFBMeta *meta,
    uint32_t streamCount,
    const NSUInteger *bufferWritten,
    const NSUInteger *bufferStride,
    GLuint64 geometryInvocations)
{
    mglRecordActiveGeometryShaderQueryDraw(
        ctx, geometryInvocations, generatedStream0);
    mglRecordActivePrimitiveQueryDraw(
        ctx, generatedStream0,
        mglDrawGsStream0QueryWritten(xfbActive ? 1 : 0, writtenStream0));
    if (!meta || !bufferWritten || !bufferStride) return;
    streamCount = mglDrawGsClampStreamCount(streamCount);
    for (uint32_t s = 1u; s < streamCount; s++) {
        /* Indexed stream s query: generated stays in the meta; written is
         * the ordered scatter's whole-primitive bytes for buffer s divided
         * by its per-record stride (streams > 0 are points, vpp = 1). */
        GLuint64 written = mglDrawGsIndexedStreamWritten(
            xfbActive ? 1 : 0, (uint64_t)bufferWritten[s],
            (uint64_t)bufferStride[s]);
        mglRecordActivePrimitiveQueryDrawIndexed(
            ctx, s, (GLuint64)meta->stream[s].generated, written);
    }
}

id mglDefaultTessFactorBuffer(id device,
                                                GLMState *state,
                                                GLuint patchCount)
{
    if (!device || !state || patchCount == 0u) return nil;
    uint64_t factorBytes = 0u;
    if (!mglTessPlanDefaultFactorBytes(patchCount, &factorBytes)) return nil;
    id buffer = mglDrawSupportCreateBuffer(
        device, (NSUInteger)factorBytes, 0u);
    if (!buffer || !mglDrawSupportBufferContents(buffer)) return nil;

    if (mglRenderFillDefaultTessFactorBuffer(
            (void *)mglDrawSupportBufferContents(buffer),
            factorBytes,
            state->var.patch_default_outer_level,
            state->var.patch_default_inner_level,
            patchCount) != 0) {
        return nil;
    }
    return buffer;
}

/* Cached variant of the default factor buffer for the TES-only path:
 * consecutive tess draws reuse one stable allocation unless the default
 * patch levels or patch count actually changed. */
id mglCachedDefaultTessFactorBuffer(
    id device, MGLRendererBackendHandle *backend, GLMState *state,
    GLuint patchCount)
{
    if (!device || !backend || !state || patchCount == 0u) return nil;
    float levels[6];
    mglTessFillDefaultFactorLevels(state->var.patch_default_outer_level,
                                   state->var.patch_default_inner_level,
                                   levels);
    void *cached = NULL;
    if (mglRendererBackendGetTessFactorBuffer(
            backend, patchCount, levels, &cached) == 1 && cached) {
        return (__bridge id)cached;
    }
    id fresh = mglDefaultTessFactorBuffer(device, state, patchCount);
    if (!fresh) return nil;
    if (mglRendererBackendPutTessFactorBuffer(
            backend, patchCount, levels, (__bridge void *)fresh) != 0) {
        return fresh;
    }
    return fresh;
}

id mglNativeTessFactorBuffer(id device,
                                                id canonical,
                                                GLenum mode,
                                                GLuint patchCount)
{
    if (!device || !canonical || !mglDrawSupportBufferContents(canonical) ||
        patchCount == 0u) {
        return nil;
    }
    uint32_t repackBytes = 0u;
    const int factorKind = mglTessPlanNativeFactor(
        (uint32_t)mode, (uint64_t)mglDrawSupportBufferLength(canonical),
        patchCount, &repackBytes);
    if (factorKind == MGL_TESS_NATIVE_FACTOR_REUSE) {
        return canonical;
    }
    if (factorKind != MGL_TESS_NATIVE_FACTOR_REPACK_TRI) {
        return nil;
    }

    id result = mglDrawSupportCreateBuffer(device, (NSUInteger)repackBytes, 0u);
    if (!result || !mglDrawSupportBufferContents(result)) {
        return nil;
    }

    if (mglRenderRepackTessFactorTriangles(
            (const void *)mglDrawSupportBufferContents(canonical), (uint64_t)mglDrawSupportBufferLength(canonical),
            (void *)mglDrawSupportBufferContents(result),
            (uint64_t)repackBytes,
            patchCount) != 0) {
        return nil;
    }
    return result;
}

GLuint64 mglNativeTessPrimitiveCount(id canonical,
                                             Program *tesProgram,
                                             GLuint patchCount,
                                             GLuint instanceCount)
{
    if (!canonical || !mglDrawSupportBufferContents(canonical) || !tesProgram || patchCount == 0u) {
        return 0u;
    }

    return mglTessGeneratedPrimitiveCount(
        tesProgram, (const void *)mglDrawSupportBufferContents(canonical),
        patchCount, instanceCount);
}


/* O1.2/O1.1: C ports for mglTessRunCaptureSession host ops. */
void mglDrawSupportCaptureMarkDirtyAll(void *ctx_ptr)
{
    GLMContext drawCtx = (GLMContext)ctx_ptr;
    if (drawCtx && drawCtx->active_state) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
    }
}

int mglDrawSupportCaptureProcessGL(void *renderer)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    return host && [host processGLState:true] ? 1 : 0;
}

int mglDrawSupportCaptureEncoderReady(void *renderer)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (!host) return 0;
    return mglRenderEncoderOwnerHasCurrent(
               host->_renderPassManager.state->currentRenderEncoderOwner) == 1
               ? 1
               : 0;
}

void mglDrawSupportCaptureBindSlots(void *renderer, void *capture,
                                           const uint32_t *params)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (!host || !capture || !params) return;
    mglTessBindCaptureSlots(
        host->_renderPassManager.state->currentRenderEncoderOwner, capture,
        params);
}

void mglDrawSupportCaptureSetActive(void *renderer, int active)
{
    MGLRenderer *host = (__bridge MGLRenderer *)renderer;
    if (host) {
        host->_tessellation.tessVertexCaptureActive = active ? YES : NO;
    }
}

@implementation MGLRenderer (Draw)

- (BOOL)resolveElementBufferForDraw:(const char *)label
                            context:(GLMContext)drawCtx
                           glBuffer:(Buffer **)glBufferOut
                          mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = getElementBuffer(drawCtx);
    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBufferForCommand:(const MGLDrawCommand *)cmd
                                  label:(const char *)label
                                context:(GLMContext)drawCtx
                               glBuffer:(Buffer **)glBufferOut
                              mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = NULL;
    if (cmd && cmd->element_buffer_name) {
        gl_element_buffer = mglRendererGetValidatedBuffer(drawCtx,
                                                          mglDrawCommandElementBuffer(drawCtx, cmd),
                                                          label ? label : "deferred indexed draw",
                                                          0);
        if (!gl_element_buffer) {
            return NO;
        }
    } else {
        gl_element_buffer = getElementBuffer(drawCtx);
    }

    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBuffer:(Buffer *)gl_element_buffer
                       label:(const char *)label
                     context:(GLMContext)drawCtx
                    glBuffer:(Buffer **)glBufferOut
                   mtlBuffer:(id *)mtlBufferOut
{
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no element array buffer is bound", label ? label : "indexed draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        }
        return NO;
    }

    if ([self processBuffer:gl_element_buffer] == false) {
        return NO;
    }

    id indexBuffer = (__bridge id)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: %s skipped because element buffer %u has no Metal buffer",
              label ? label : "indexed draw",
              gl_element_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_element_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indexBuffer;
    }
    return YES;
}

- (BOOL)resolveIndirectBufferForDraw:(const char *)label
                             context:(GLMContext)drawCtx
                            glBuffer:(Buffer **)glBufferOut
                           mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(drawCtx);
    if (!gl_indirect_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no draw indirect buffer is bound", label ? label : "indirect draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        }
        return NO;
    }

    if ([self processBuffer:gl_indirect_buffer] == false) {
        return NO;
    }

    id indirectBuffer = (__bridge id)(gl_indirect_buffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer",
              label ? label : "indirect draw",
              gl_indirect_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_indirect_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indirectBuffer;
    }
    return YES;
}

- (BOOL)prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label
{
    if (!drawCtx) {
        NSLog(@"MGL WARNING: %s skipped because context is NULL",
              label ? label : "indirect emulation");
        return NO;
    }

    /* The C draw-indirect frontends already flush pending command buffers before
     * dispatching into these Metal entry points. If processGLState has just
     * rebuilt a render encoder, keep it; a second flush can discard the fresh
     * pass and make state restoration fail for CPU-emulated indirect modes. */
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        NSLog(@"MGL WARNING: %s skipped because CPU-read synchronization left no render encoder",
              label ? label : "indirect emulation");
        return NO;
    }
    return YES;
}

- (BOOL)currentDrawRasterizationIsEmpty
{
    if (!ctx) {
        return NO;
    }

    GLint vx = MGL_STATE(ctx)->viewport[0];
    GLint vy = MGL_STATE(ctx)->viewport[1];
    GLint vw = MGL_STATE(ctx)->viewport[2];
    GLint vh = MGL_STATE(ctx)->viewport[3];

    NSUInteger passWidth = 0;
    NSUInteger passHeight = 0;
    mglRenderGetRenderTargetSizeOwner(
        _renderPassManager.state->renderPassStateOwner,
        (uint64_t *)&passWidth, (uint64_t *)&passHeight);
    if (passWidth == 0 || passHeight == 0) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id color = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i);
            if (color) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(color);
                passWidth = info.width;
                passHeight = info.height;
                break;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id depth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            if (depth) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(depth);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id stencil = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            if (stencil) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(stencil);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
    }


    return mglRenderRasterizationIsEmpty(
               vx, vy, vw, vh,
               (uint32_t)passWidth, (uint32_t)passHeight,
               MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
               MGL_STATE(ctx)->var.scissor_box[0],
               MGL_STATE(ctx)->var.scissor_box[1],
               MGL_STATE(ctx)->var.scissor_box[2],
               MGL_STATE(ctx)->var.scissor_box[3]) != 0;
}

- (void)applyPolygonOffsetForDrawMode:(GLenum)mode
{
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return;
    }


    MGLRenderPolygonOffsetDecision decision = {0};
    mglRenderPolygonOffsetDecision(
        (uint32_t)mode,
        ctx ? 1 : 0,
        mglDrawModeProducesPolygons(mode) ? 1 : 0,
        (uint32_t)(ctx ? MGL_STATE(ctx)->var.polygon_mode : 0u),
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_point) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_line) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_fill) ? 1 : 0,
        &decision);
    uint32_t triangleFillMode = decision.triangle_fill_mode ? 1u : 0u;
    if (decision.needs_polygon_mode_repair) {
        uint32_t repaired = mglRenderPolygonModeOrFill(
            (uint32_t)MGL_STATE(ctx)->var.polygon_mode);
        mglLogRenderStateRepair("polygon_mode", MGL_STATE(ctx)->var.polygon_mode,
                                (GLenum)repaired);
        MGL_STATE(ctx)->var.polygon_mode = (GLenum)repaired;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];

    BOOL enableDepthBias = decision.enable_depth_bias != 0;

    if (enableDepthBias) {
        float _bias = MGL_STATE(ctx)->var.polygon_offset_units;
        float _slope = MGL_STATE(ctx)->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    } else {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            0.0f, 0.0f, 0.0f);
    }
}

- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode
{
    return ctx &&
           mglRenderDrawModeFullyCulled(
               MGL_STATE(ctx)->caps.cull_face ? 1 : 0,
               (uint32_t)MGL_STATE(ctx)->var.cull_face_mode,
               mglDrawModeProducesPolygons(mode) ? 1 : 0) != 0;
}

- (BOOL)ensureRasterEncoderForDraw
{
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }
    [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_DRAW];
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return NO;
    }
    if (!_pipelineCache.state->pipelineState) {
        return NO;
    }

    uint32_t rpColor0Format = 0u;
    uint32_t rpDepthFormat = 0u;
    uint32_t rpStencilFormat = 0u;
    MGLRenderPassAttachmentState colorAttachment = {0};
    MGLRenderPassAttachmentState depthAttachment = {0};
    MGLRenderPassAttachmentState stencilAttachment = {0};
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, &colorAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, &depthAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, &stencilAttachment);
    id rpColor0 = (__bridge id)colorAttachment.texture;
    id rpDepth = (__bridge id)depthAttachment.texture;
    id rpStencil = (__bridge id)stencilAttachment.texture;
    MGLRenderTextureInfo textureInfo = {0};
    if (rpColor0 && mglRenderGetTextureInfo(
            (__bridge void *)rpColor0, &textureInfo) == 0) {
        rpColor0Format = textureInfo.pixel_format;
    }
    if (rpDepth && mglRenderGetTextureInfo(
            (__bridge void *)rpDepth, &textureInfo) == 0) {
        rpDepthFormat = textureInfo.pixel_format;
    }
    if (rpStencil && mglRenderGetTextureInfo(
            (__bridge void *)rpStencil, &textureInfo) == 0) {
        rpStencilFormat = textureInfo.pixel_format;
    }

    const BOOL colorMismatch =
        (_pipelineCache.state->pipelineColor0Format != 0u &&
         rpColor0Format != 0u &&
         _pipelineCache.state->pipelineColor0Format != rpColor0Format);
    const BOOL depthMismatch =
        (_pipelineCache.state->pipelineDepthFormat != rpDepthFormat);
    const BOOL stencilMismatch =
        (_pipelineCache.state->pipelineStencilFormat != rpStencilFormat);
    if (colorMismatch || depthMismatch || stencilMismatch) {
        return NO;
    }
    if (mglRenderSetRenderPipelineStateForOwner(
            _renderPassManager.state->currentRenderEncoderOwner,
            _pipelineCache.state->pipelineState) != 0) {
        return NO;
    }
    mglRenderBindingSetPipelineState(_bindingStateOwner,
                                     _pipelineCache.state->pipelineState);
    MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
    return YES;
}

- (void)bindCullDistanceEmulationBuffers:(GLenum)mode
                             firstVertex:(GLuint)firstVertex
                        explicitVertices:(const GLuint *)explicitVertices
                      explicitVertexCount:(GLuint)explicitVertexCount
                           encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!ctx || !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, "bindCullDistanceEmu");
    if (!vao) {
        return;
    }
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (!activeProgram) {
        return;
    }
    explicitVertexCount = MIN(explicitVertexCount, 4u);

    id captureBuffer = (__bridge id)
        mglRendererBackendGetCullDistanceCaptureBuffer(_backend);
    if (captureBuffer) {
        MGLCullDistanceEmuParams params;
        mglRenderFillCullDistanceEmuParams(
            mglRenderPrimitiveVertexCountForMode((uint32_t)mode), firstVertex,
            explicitVertices, explicitVertexCount, 0u, 32u,
            MIN(activeProgram->cull_distance_count, 8u),
            _tessellation.cullDistanceCaptureFirstInstance,
            _tessellation.cullDistanceCaptureInstanceStride, &params);
        mglRenderBindCullDistanceEmuSlots(encCtx->render_encoder_owner,
                                          (__bridge void *)captureBuffer,
                                          &params);
        [self recordLastBoundVertexBuffer:
                  captureBuffer
                                   offset:0
                                  atIndex:kMGLCullDistanceVertexBufferIndex];
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        [self invalidateLastBoundVertexBufferAtIndex:
                  kMGLCullDistanceParamsBufferIndex];
        return;
    }

    /* O1.3: ObjC fills VAO pointer ports; layout (+ dummy) in C++. */
    uint32_t attribs[MAX_ATTRIBS];
    const uint32_t attribCount = mglRenderCollectCullDistanceAttribs(
        activeProgram, attribs, MAX_ATTRIBS);
    MGLRenderCullDistanceAttribPort ports[MAX_ATTRIBS];
    memset(ports, 0, sizeof(ports));
    uint32_t portCount = 0u;
    for (uint32_t i = 0u; i < attribCount && portCount < MAX_ATTRIBS; i++) {
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(
                ctx, vao, attribs[i], "bindCullDistanceEmu", &resolved)) {
            continue;
        }
        if (!resolved.buffer || !resolved.buffer->data.mtl_data) {
            continue;
        }
        ports[portCount].mtl_buffer = resolved.buffer->data.mtl_data;
        ports[portCount].binding_offset = resolved.binding_offset;
        ports[portCount].stride = resolved.stride;
        ports[portCount].relativeoffset = resolved.relativeoffset;
        ports[portCount].valid = 1u;
        portCount++;
    }
    MGLRenderCullDistanceLayout layout;
    mglRenderBuildCullDistanceLayoutFromPorts(
        &layout, ports, portCount,
        mglRendererBackendGetCullDistanceDummyBuffer(_backend));
    void *cullMtlBuffer = layout.mtl_buffer;
    uint32_t cullStride = layout.stride;
    uint32_t cullDistSize = layout.culldist_size;

    MGLCullDistanceEmuParams params;
    mglRenderFillCullDistanceEmuParams(
        mglRenderPrimitiveVertexCountForMode((uint32_t)mode), firstVertex,
        explicitVertices, explicitVertexCount,
        mglRenderCullDistanceLayoutOffset(&layout), cullStride, cullDistSize,
        0u, 0u, &params);
    mglRenderBindCullDistanceEmuSlots(encCtx->render_encoder_owner,
                                      cullMtlBuffer, &params);
    [self recordLastBoundVertexBuffer:(__bridge id)cullMtlBuffer
                               offset:0
                              atIndex:kMGLCullDistanceVertexBufferIndex];
    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    [self invalidateLastBoundVertexBufferAtIndex:kMGLCullDistanceParamsBufferIndex];
}

- (Texture *)emulatedMSColor0TextureForContext:(GLMContext)glm_ctx
{
    if (!glm_ctx) return NULL;
    Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
    if (!fbo || (fbo->color_attachment_bitfield & 1u) == 0u) return NULL;
    FBOAttachment *att = &fbo->color_attachments[0];
    Texture *tex = [self framebufferAttachmentTexture:att];
    if (!tex) return NULL;
    if (!mglRenderIsEmulatedMSColorTexture((uint32_t)tex->target,
                                           (int32_t)tex->samples)) {
        return NULL;
    }
    /* Metal backing is created during processGLState; before the first draw
     * mtl_data may still be nil. All MS textures are emulated as array
     * sample planes, so the GL target/samples check is sufficient. */
    if (tex->mtl_data) {
        MGLRenderTextureInfo info = {0};
        (void)mglRenderGetTextureInfo(tex->mtl_data, &info);
        if (info.texture_type != MGLTextureType2DArray) return NULL;
    }
    return tex;
}

- (BOOL)fragmentNeedsPerSampleMSValuesForContext:(GLMContext)glm_ctx
{
    Program *fp = mglResolveProgramForStageFromState(glm_ctx, _FRAGMENT_SHADER);
    if (!fp) return NO;
    Shader *fs = fp->shader_slots[_FRAGMENT_SHADER];
    if (!fs || !fs->src) return NO;
    return mglRenderFragmentNeedsPerSampleMSValues(fs->src) != 0;
}

- (BOOL)runEmulatedMSSampleDrawLoopIfNeeded:(GLMContext)glm_ctx
                                   drawOnce:(void (^)(void))drawOnce
{
    if (_mglInMSSampleDrawLoop || !drawOnce) return NO;
    Texture *tex = [self emulatedMSColor0TextureForContext:glm_ctx];
    if (!tex) return NO;
    if (![self fragmentNeedsPerSampleMSValuesForContext:glm_ctx]) return NO;

    const GLint samples = (GLint)MAX(tex->samples, 1);
    _mglInMSSampleDrawLoop = YES;
    for (GLint s = 0; s < samples; s++) {
        _mglForcedMSSampleId = s;
        _mglMSSamplePlaneOffset = s;
        [self endRenderEncodingLocked];
        mglMarkStateDirtyBits(MGL_STATE(glm_ctx), DIRTY_FBO);
        Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
        if (fbo) {
            fbo->dirty_bits |= DIRTY_FBO_BINDING;
        }
        drawOnce();
    }
    _mglForcedMSSampleId = 0;
    _mglMSSamplePlaneOffset = 0;
    _mglInMSSampleDrawLoop = NO;
    return YES;
}

- (void)broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:(GLMContext)glm_ctx
{
    if (_mglInMSSampleDrawLoop) return;
    Texture *tex = [self emulatedMSColor0TextureForContext:glm_ctx];
    if (!tex || !tex->mtl_data) return;
    if ([self fragmentNeedsPerSampleMSValuesForContext:glm_ctx]) {
        /* Per-sample draws already filled each plane. */
        return;
    }

    const NSUInteger samples = MAX((NSUInteger)tex->samples, 1u);
    if (samples <= 1u) return;

    MGLMetalAttachmentSubresource sub =
        mglMetalAttachmentSubresourceForAttachment(
            &MGL_STATE(glm_ctx)->framebuffer->color_attachments[0]);
    const NSUInteger baseSlice = sub.slice;
    const NSUInteger level = sub.level;
    MGLRenderTextureInfo info = {0};
    (void)mglRenderGetTextureInfo(tex->mtl_data, &info);
    if (info.width == 0u || info.height == 0u) return;
    if (baseSlice + samples > info.array_length) return;

    [self endRenderEncodingLocked];
    if (!_renderPassManager.state->currentCommandBufferOwner &&
        ![self newCommandBufferLocked]) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(
        _renderPassManager.state->currentCommandBufferOwner);
    if (!blit) return;
    for (NSUInteger s = 1u; s < samples; s++) {
        (void)mglRenderBlitCopyTexture(
            blit, tex->mtl_data, baseSlice, level, 0u, 0u, 0u,
            info.width, info.height, 1u,
            tex->mtl_data, baseSlice + s, level, 0u, 0u, 0u);
    }
    (void)mglRenderEndBlitEncoder(blit);
}

static MGLRenderer *mglDrawHostSelf(void *renderer)
{
    return renderer ? (__bridge MGLRenderer *)renderer : nil;
}


void mglDrawHostGuardIssueArrays(void *renderer, GLMContext ctx, GLenum mode,
                                 GLint first, GLsizei count,
                                 GLsizei instanceCount, GLuint baseInstance,
                                 const char *label, int with_ms)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || !ctx) {
        return;
    }
    METAL_LOCK();
    host->_lastDrawPrimitiveMode = mode;
    if (with_ms &&
        [host runEmulatedMSSampleDrawLoopIfNeeded:ctx
                                         drawOnce:^{
                                             mglIssueDrawArrays(
                                                 ctx, renderer, mode, first,
                                                 count, instanceCount,
                                                 baseInstance, label);
                                         }]) {
        METAL_UNLOCK();
        return;
    }
    mglIssueDrawArrays(ctx, renderer, mode, first, count, instanceCount,
                       baseInstance, label);
    if (with_ms) {
        [host broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:ctx];
    }
    METAL_UNLOCK();
}

void mglDrawHostGuardIssueElements(void *renderer, GLMContext ctx, GLenum mode,
                                   GLsizei count, GLenum type,
                                   const void *indices, GLsizei instanceCount,
                                   GLint baseVertex, GLuint baseInstance,
                                   const char *label, int with_ms)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || !ctx) {
        return;
    }
    METAL_LOCK();
    host->_lastDrawPrimitiveMode = mode;
    if (with_ms &&
        [host runEmulatedMSSampleDrawLoopIfNeeded:ctx
                                         drawOnce:^{
                                             mglIssueDrawElements(
                                                 ctx, renderer, mode, count,
                                                 type, indices, instanceCount,
                                                 baseVertex, baseInstance,
                                                 label);
                                         }]) {
        METAL_UNLOCK();
        return;
    }
    mglIssueDrawElements(ctx, renderer, mode, count, type, indices,
                         instanceCount, baseVertex, baseInstance, label);
    if (with_ms) {
        [host broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:ctx];
    }
    METAL_UNLOCK();
}

bool mglDrawHostBindContext(void *renderer, GLMContext ctx)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    host->ctx = ctx;
    return true;
}

void mglDrawHostSetLastPrimitiveMode(void *renderer, GLenum mode)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        host->_lastDrawPrimitiveMode = mode;
    }
}

bool mglDrawHostHandleTessellation(void *renderer, GLMContext ctx,
                                   GLenum *mode, GLint first, GLsizei count,
                                   GLenum indexType, const void *indices,
                                   GLint baseVertex, GLsizei instanceCount,
                                   GLuint baseInstance, const char *label)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || !mode) {
        return false;
    }
    return [host handleTessellationPatchDrawIfNeeded:ctx
                                                mode:mode
                                               first:first
                                               count:count
                                           indexType:indexType
                                             indices:indices
                                          baseVertex:baseVertex
                                       instanceCount:instanceCount
                                        baseInstance:baseInstance
                                               label:label] ? true : false;
}

bool mglDrawHostHandleGeometry(void *renderer, GLMContext ctx, GLenum mode,
                               GLint first, GLsizei count, GLenum indexType,
                               const void *indices, GLint baseVertex,
                               GLsizei instanceCount, GLuint baseInstance,
                               const char *label)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host handleGeometryDrawIfNeeded:ctx
                                       mode:mode
                                      first:first
                                      count:count
                                  indexType:indexType
                                    indices:indices
                                 baseVertex:baseVertex
                              instanceCount:instanceCount
                               baseInstance:baseInstance
                                      label:label] ? true : false;
}

bool mglDrawHostHandleXFB(void *renderer, GLMContext ctx, GLenum mode,
                          GLint first, GLsizei count, GLsizei instanceCount,
                          GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host handleVertexTransformFeedbackDrawIfNeeded:ctx
                                                      mode:mode
                                                     first:first
                                                     count:count
                                             instanceCount:instanceCount
                                              baseInstance:baseInstance]
               ? true
               : false;
}

bool mglDrawHostCaptureCullDistanceArray(void *renderer, GLMContext ctx,
                                         GLint first, GLsizei count,
                                         GLsizei instanceCount,
                                         GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host captureAIRCullDistancesForArrayDraw:ctx
                                               first:first
                                               count:count
                                       instanceCount:instanceCount
                                        baseInstance:baseInstance]
               ? true
               : false;
}

bool mglDrawHostProcessGLStateLocked(void *renderer, bool draw_command)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host processGLStateLocked:draw_command] ? true : false;
}

bool mglDrawHostRasterizationIsEmpty(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host currentDrawRasterizationIsEmpty];
}

bool mglDrawHostModeFullyCulled(void *renderer, GLenum mode)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host currentDrawModeIsFullyCulled:mode];
}

void mglDrawHostApplyPolygonOffset(void *renderer, GLenum mode)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        [host applyPolygonOffsetForDrawMode:mode];
    }
}

bool mglDrawHostEnsureRasterEncoder(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host ensureRasterEncoderForDraw];
}

bool mglDrawHostValidateArrayVertexInputs(void *renderer, GLMContext ctx,
                                          GLenum mode, GLint first,
                                          GLsizei count)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    return [host validateDrawArraysVertexInputs:ctx
                                           mode:mode
                                          first:first
                                          count:count
                                       drawCall:0];
}

bool mglDrawHostEncodeCullDistanceArray(void *renderer, GLenum mode,
                                        GLint first, GLsizei count,
                                        GLsizei instanceCount,
                                        GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || mglPolygonModePointForDrawMode(host->ctx, mode)) {
        return false;
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner =
            host->_renderPassManager.state->currentRenderEncoderOwner,
    };
    return [host encodeCullDistanceArrayDraw:mode
                                       first:first
                                       count:count
                               instanceCount:instanceCount
                                baseInstance:baseInstance
                               encodeContext:&encCtx]
               ? true
               : false;
}

void *mglDrawHostEncoderOwner(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host ? host->_renderPassManager.state->currentRenderEncoderOwner
                : NULL;
}

void *mglDrawHostDevice(void *renderer)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host ? mglRendererBackendGetDevice(host->_backend) : NULL;
}

void mglDrawHostRecordArraySubmitted(void *renderer, GLenum mode,
                                     uint64_t vertexCount)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        [host recordArrayDrawSubmittedMode:mode vertexCount:vertexCount];
    }
}

void mglDrawHostWatchdogArrays(void *renderer, GLMContext ctx)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return;
    }
    mglLogDrawWithoutSwapWatchdog(
        "arrays", 0, ctx,
        host->_renderPassManager.state->currentCommandBufferOwner,
        host->_renderPassManager.state->currentRenderEncoderOwner,
        host->_renderPassManager.state->renderPassStateOwner);
}

bool mglDrawHostEncodeCullDistanceElements(void *renderer, GLenum mode,
                                           GLenum type, const void *indices,
                                           GLsizei count, GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host || mglPolygonModePointForDrawMode(host->ctx, mode)) {
        return false;
    }
    Buffer *glBuffer = NULL;
    id metalBuffer = nil;
    if (![host resolveElementBufferForDraw:"drawElements"
                                   context:host->ctx
                                  glBuffer:&glBuffer
                                 mtlBuffer:&metalBuffer]) {
        return false;
    }
    const NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    const uint8_t *cullIndexBytes = mglElementIndexSourceForDraw(
        glBuffer, metalBuffer, type, offset, count);
    return [host prepareAndEncodeDirectCullDistanceElementDraw:mode
                                                   indexBytes:cullIndexBytes
                                                    indexType:type
                                                        count:count
                                                   baseVertex:baseVertex
                                                instanceCount:instanceCount
                                                 baseInstance:baseInstance
                                              polygonLineMode:
                                                  mglPolygonModeLineForDrawMode(
                                                      host->ctx, mode)]
               ? true
               : false;
}

bool mglDrawHostResolveElementBuffer(void *renderer, GLMContext ctx,
                                     const char *label, Buffer **glBufferOut,
                                     void **metalBufferOut)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    id metalBuffer = nil;
    if (![host resolveElementBufferForDraw:label ? label : "drawElements"
                                   context:ctx
                                  glBuffer:glBufferOut
                                 mtlBuffer:&metalBuffer]) {
        return false;
    }
    if (metalBufferOut) {
        *metalBufferOut = (__bridge void *)metalBuffer;
    }
    return true;
}

void mglDrawHostRecordElementSubmitted(void *renderer, GLenum mode,
                                       uint64_t indexCount)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (host) {
        [host recordElementDrawSubmittedMode:mode indexCount:indexCount];
    }
}

void mglDrawHostWatchdogElements(void *renderer, GLMContext ctx)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return;
    }
    mglLogDrawWithoutSwapWatchdog(
        "elements", 0, ctx,
        host->_renderPassManager.state->currentCommandBufferOwner,
        host->_renderPassManager.state->currentRenderEncoderOwner,
        host->_renderPassManager.state->renderPassStateOwner);
}

bool mglDrawHostResolveIndirectBuffer(void *renderer, GLMContext ctx,
                                      const char *label, Buffer **glBufferOut,
                                      void **metalBufferOut)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    if (!host) {
        return false;
    }
    id metalBuffer = nil;
    if (![host resolveIndirectBufferForDraw:label ? label : "indirectDraw"
                                   context:ctx
                                  glBuffer:glBufferOut
                                 mtlBuffer:&metalBuffer]) {
        return false;
    }
    if (metalBufferOut) {
        *metalBufferOut = (__bridge void *)metalBuffer;
    }
    return true;
}

bool mglDrawHostPrepareIndirectCPURead(void *renderer, GLMContext ctx,
                                       const char *label)
{
    MGLRenderer *host = mglDrawHostSelf(renderer);
    return host && [host prepareEmulatedIndirectCPURead:ctx
                                                  label:label ? label : "indirectDraw"];
}

bool mglDrawHostHasGeometry(GLMContext ctx)
{
    Program *gsProgram = mglResolveProgramForStageFromState(ctx, _GEOMETRY_SHADER);
    return gsProgram && gsProgram->shader_slots[_GEOMETRY_SHADER];
}

bool mglDrawHostUsesCullDistance(GLMContext ctx)
{
    Program *vertexProgram =
        mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    return vertexProgram && vertexProgram->uses_cull_distance;
}

@end
