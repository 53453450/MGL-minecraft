/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+DrawStageHost.m
// A1 / O1.4 residual2: thin ObjC glue (kept; not empty). Capture/validate/cull
// encode → C++ (mgl_draw_tess / mgl_draw_cull / mgl_draw_issue). GS Metal
// expansion HostOps → mgl_draw_metal_port.m → mgl_draw_gs_metal.cpp. This
// category keeps bindCull VAO ports + MS sample loop + one-line wrappers.
// Do not open new thick ObjC categories here.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+DrawSupportUtil.h"
#include "mgl_draw_cull.h"
#include "mgl_draw_issue.h"
#include "mgl_draw_tess.h"
#include "mgl_draw_encode.h"
#include "mgl_index_buffer.h"
#include "mgl_render.h"
#include "glm_limits.h"
#include "mgl_buffer_query.h"

@implementation MGLRenderer (DrawStageHost)

- (BOOL)captureAIRCullDistancesForArrayDraw:(GLMContext)drawCtx
                                      first:(GLint)first
                                      count:(GLsizei)count
                              instanceCount:(GLsizei)instanceCount
                               baseInstance:(GLuint)baseInstance
{
    return mglDrawHostCaptureCullDistanceArray((__bridge void *)self, drawCtx,
                                               first, count, instanceCount,
                                               baseInstance)
               ? YES
               : NO;
}

- (BOOL)captureAIRCullDistancesForElementDraw:(GLMContext)drawCtx
                                    indexBytes:(const uint8_t *)indexBytes
                                     indexType:(GLenum)indexType
                                         count:(GLsizei)count
                                    baseVertex:(GLint)baseVertex
                                 instanceCount:(GLsizei)instanceCount
                                  baseInstance:(GLuint)baseInstance
{
    return mglDrawHostCaptureCullDistanceElement(
               (__bridge void *)self, drawCtx, indexBytes, indexType, count,
               baseVertex, instanceCount, baseInstance)
               ? YES
               : NO;
}

- (BOOL)prepareAndEncodeDirectCullDistanceElementDraw:(GLenum)mode
                                           indexBytes:(const uint8_t *)indexBytes
                                            indexType:(GLenum)indexType
                                                count:(GLsizei)count
                                           baseVertex:(GLint)baseVertex
                                        instanceCount:(GLsizei)instanceCount
                                         baseInstance:(GLuint)baseInstance
                                      polygonLineMode:(BOOL)polygonLineMode
{
    return mglDrawHostPrepareEncodeCullDistanceElement(
               (__bridge void *)self, mode, indexBytes, indexType, count,
               baseVertex, instanceCount, baseInstance,
               polygonLineMode ? 1 : 0)
               ? YES
               : NO;
}

- (BOOL)encodeCullDistanceArrayDraw:(GLenum)mode
                               first:(GLint)first
                               count:(GLsizei)count
                       instanceCount:(GLsizei)instanceCount
                        baseInstance:(GLuint)baseInstance
                       encodeContext:(const MGLEncodeContext *)encCtx
{
    (void)encCtx;
    return mglDrawHostEncodeCullDistanceArray((__bridge void *)self, mode, first,
                                              count, instanceCount, baseInstance)
               ? YES
               : NO;
}

- (BOOL)encodeCullDistanceElementDraw:(GLenum)mode
                            indexBytes:(const uint8_t *)indexBytes
                             indexType:(GLenum)indexType
                                 count:(GLsizei)count
                            baseVertex:(GLint)baseVertex
                         instanceCount:(GLsizei)instanceCount
                          baseInstance:(GLuint)baseInstance
                       polygonLineMode:(BOOL)polygonLineMode
                         encodeContext:(const MGLEncodeContext *)encCtx
{
    return mglDrawHostEncodeCullDistanceElementBytes(
               (__bridge void *)self, mode, indexBytes, indexType, count,
               baseVertex, instanceCount, baseInstance,
               polygonLineMode ? 1 : 0, encCtx)
               ? YES
               : NO;
}

- (BOOL)runVertexCaptureSession:(GLMContext)drawCtx
                        capture:(id)capture
                         params:(const uint32_t *)params
{
    if (!drawCtx || !capture || !params) return NO;
    self->ctx = drawCtx;
    MGLTessCaptureSessionHostOps ops = {
        .ctx = drawCtx,
        .renderer = (__bridge void *)self,
        .mark_dirty_all = mglDrawSupportCaptureMarkDirtyAll,
        .process_gl_state = mglDrawSupportCaptureProcessGL,
        .encoder_has_current = mglDrawSupportCaptureEncoderReady,
        .bind_capture_slots = mglDrawSupportCaptureBindSlots,
        .set_capture_active = mglDrawSupportCaptureSetActive,
    };
    return mglTessRunCaptureSession((__bridge void *)capture, params, &ops)
               ? YES
               : NO;
}

- (id)captureAIRVertexPositionsForTessellation:(GLMContext)drawCtx
                                                    first:(GLint)first
                                                    count:(GLsizei)count
                                            instanceCount:(GLsizei)instanceCount
                                             baseInstance:(GLuint)baseInstance
                                               outOffset:(NSUInteger *)outOffset
{
    uint64_t off = 0u;
    void *cap = mglDrawHostRunVertexCaptureArray(
        (__bridge void *)self, drawCtx, first, count, instanceCount,
        baseInstance, &off);
    if (outOffset) *outOffset = (NSUInteger)off;
    return (__bridge_transfer id)cap;
}

- (id)captureAIRVertexPositionsForGeometryIndexed:(GLMContext)drawCtx
                                                  indexBuffer:(id)indexBuffer
                                                    indexType:(uint64_t)indexType
                                                  indexOffset:(NSUInteger)indexOffset
                                                        count:(GLsizei)count
                                                    baseVertex:(GLint)baseVertex
                                                 instanceCount:(GLsizei)instanceCount
                                                  baseInstance:(GLuint)baseInstance
                                                     maxIndex:(uint32_t)maxIndex
                                                    outOffset:(NSUInteger *)outOffset
{
    uint64_t off = 0u;
    void *cap = mglDrawHostRunVertexCaptureIndexed(
        (__bridge void *)self, drawCtx, (__bridge void *)indexBuffer, indexType,
        (uint64_t)indexOffset, count, baseVertex, instanceCount, baseInstance,
        maxIndex, &off);
    if (outOffset) *outOffset = (NSUInteger)off;
    return (__bridge_transfer id)cap;
}

- (bool)validateDrawArraysVertexInputs:(GLMContext)drawCtx
                                    mode:(GLenum)mode
                                   first:(GLint)first
                                   count:(GLsizei)count
                                drawCall:(uint64_t)drawCall
{
    (void)drawCall;
    return mglDrawHostValidateArrayVertexInputs((__bridge void *)self, drawCtx,
                                                mode, first, count);
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
    return mglRenderFragmentNeedsPerSampleMSValues(fp) != 0;
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



@end
