/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_draw_support.c — bodies of -[MGLRenderer resolveIndirectBufferForDraw:],
 * -currentDrawRasterizationIsEmpty, -applyPolygonOffsetForDrawMode: and
 * -currentDrawModeIsFullyCulled:.
 *
 * Every step was already C (mglRenderRasterizationIsEmpty,
 * mglRenderPolygonOffsetDecision, mglRenderDrawModeFullyCulled,
 * mglRenderBinding*ForOwner, mglRendererProcessBuffer); the Objective-C state
 * they read arrives through the state areas now.  The NSLog diagnostics became
 * fprintf on the same stderr sink with the same fields.
 */

#include "mgl_draw_support.h"
#include "mgl_frame_activity.h"
#include "mgl_renderer_backend.h"
#include "mgl_vertex_attrib_binding.h"
#include "mgl_encode_context.h"
#include "mgl_binding_state_ops.h"
#include "mgl_draw_issue.h"
#include "mgl_draw_tess.h"    /* mglResolveProgramForStageFromState */   /* mglDrawHostRunVertexCaptureIndexed */
#include "mgl_renderer_ports.h"   /* state areas, mglRendererProcessBuffer */
#include "mgl_render.h"
#include "mgl_state_compat.h"     /* mglLogRenderStateRepair */
#include "error.h"                /* mglDispatchError */

#include <stdio.h>

/* Declared in the Objective-C MGLRenderer+DrawSupportUtil.h. */
extern int mglDrawSupportEncodeContextIsActive(const MGLEncodeContext *encCtx);
extern VertexArray *mglRendererGetValidatedVAO(GLMContext ctx, const char *where);
extern Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);

/* Defined in MGLRenderer.m; declared next to the definition in the
 * Objective-C MGLRenderer+Draw_Private.h, which C cannot include. */
extern Buffer *getIndirectBuffer(GLMContext ctx);

int mglDrawResolveIndirectBuffer(void *renderer, const char *label,
                                 GLMContext ctx, Buffer **gl_out,
                                 void **mtl_out)
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(ctx);
    if (!gl_indirect_buffer) {
        fprintf(stderr,
                "MGL WARNING: %s skipped because no draw indirect buffer is bound\n",
                label ? label : "indirect draw");
        if (ctx) {
            mglDispatchError(ctx, label ? label : "resolveIndirectBufferForDraw",
                             (GLenum)mglRenderErrorInvalidOperation());
        }
        return 0;
    }

    if (!mglRendererProcessBuffer(renderer, gl_indirect_buffer)) {
        return 0;
    }

    void *indirectBuffer = gl_indirect_buffer->data.mtl_data;
    if (!indirectBuffer) {
        fprintf(stderr,
                "MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer\n",
                label ? label : "indirect draw",
                gl_indirect_buffer->name);
        return 0;
    }

    if (gl_out) {
        *gl_out = gl_indirect_buffer;
    }
    if (mtl_out) {
        *mtl_out = indirectBuffer;
    }
    return 1;
}

int mglDrawRasterizationIsEmpty(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *cs = areas.command;
    if (!ctx || !cs) {
        return 0;
    }

    GLint vx = ctx->active_state->viewport[0];
    GLint vy = ctx->active_state->viewport[1];
    GLint vw = ctx->active_state->viewport[2];
    GLint vh = ctx->active_state->viewport[3];

    uint64_t passWidth = 0;
    uint64_t passHeight = 0;
    mglRenderGetRenderTargetSizeOwner(cs->renderPassStateOwner, &passWidth,
                                      &passHeight);
    if (passWidth == 0 || passHeight == 0) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            void *color = mglRenderGetRenderPassAttachmentTextureOwner(
                cs->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i);
            if (color) {
                MGLRenderTextureInfo info = {0};
                if (mglRenderGetTextureInfo(color, &info) != 0) {
                    continue;
                }
                passWidth = info.width;
                passHeight = info.height;
                break;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            void *depth = mglRenderGetRenderPassAttachmentTextureOwner(
                cs->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            if (depth) {
                MGLRenderTextureInfo info = {0};
                if (mglRenderGetTextureInfo(depth, &info) == 0) {
                    passWidth = info.width;
                    passHeight = info.height;
                }
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            void *stencil = mglRenderGetRenderPassAttachmentTextureOwner(
                cs->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            if (stencil) {
                MGLRenderTextureInfo info = {0};
                if (mglRenderGetTextureInfo(stencil, &info) == 0) {
                    passWidth = info.width;
                    passHeight = info.height;
                }
            }
        }
    }

    return mglRenderRasterizationIsEmpty(
               vx, vy, vw, vh,
               (uint32_t)passWidth, (uint32_t)passHeight,
               ctx->active_state->caps.scissor_test ? 1 : 0,
               ctx->active_state->var.scissor_box[0],
               ctx->active_state->var.scissor_box[1],
               ctx->active_state->var.scissor_box[2],
               ctx->active_state->var.scissor_box[3]) != 0;
}

void mglDrawApplyPolygonOffset(void *renderer, uint32_t mode)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *cs = areas.command;
    if (!ctx || !cs) {
        return;
    }
    if (mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }

    MGLRenderPolygonOffsetDecision decision = {0};
    mglRenderPolygonOffsetDecision(
        mode,
        ctx ? 1 : 0,
        mglRenderDrawModeProducesPolygons((uint64_t)mode) ? 1 : 0,
        (uint32_t)ctx->active_state->var.polygon_mode,
        ctx->active_state->caps.polygon_offset_point ? 1 : 0,
        ctx->active_state->caps.polygon_offset_line ? 1 : 0,
        ctx->active_state->caps.polygon_offset_fill ? 1 : 0,
        &decision);
    uint32_t triangleFillMode = decision.triangle_fill_mode ? 1u : 0u;
    if (decision.needs_polygon_mode_repair) {
        uint32_t repaired =
            mglRenderPolygonModeOrFill((uint32_t)ctx->active_state->var.polygon_mode);
        mglLogRenderStateRepair("polygon_mode", ctx->active_state->var.polygon_mode,
                                (GLenum)repaired);
        ctx->active_state->var.polygon_mode = (GLenum)repaired;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
    }

    /* -[MGLRenderer setTriangleFillModeIfNeeded:] is a two-line forward to the
     * owner-aware binding call. */
    if (areas.binding_state_owner) {
        mglRenderBindingSetTriangleFillForOwner(*areas.binding_state_owner,
                                                cs->currentRenderEncoderOwner,
                                                triangleFillMode);
    }

    int enableDepthBias = decision.enable_depth_bias != 0;

    if (enableDepthBias) {
        float bias = ctx->active_state->var.polygon_offset_units;
        float slope = ctx->active_state->var.polygon_offset_factor;
        float clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            areas.binding_state_owner ? *areas.binding_state_owner : NULL,
            cs->currentRenderEncoderOwner, bias, clamp, slope);
    } else {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            areas.binding_state_owner ? *areas.binding_state_owner : NULL,
            cs->currentRenderEncoderOwner, 0.0f, 0.0f, 0.0f);
    }
}

int mglDrawModeIsFullyCulled(void *renderer, uint32_t mode)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    if (!ctx) {
        return 0;
    }
    return mglRenderDrawModeFullyCulled(
               ctx->active_state->caps.cull_face ? 1 : 0,
               (uint32_t)ctx->active_state->var.cull_face_mode,
               mglRenderDrawModeProducesPolygons((uint64_t)mode) ? 1 : 0) != 0;
}

/* Body of the former -[MGLRenderer fragmentNeedsPerSampleMSValuesForContext:]. */
int mglDrawFragmentNeedsPerSampleMSValues(GLMContext ctx)
{
    Program *fp = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    if (!fp) {
        return 0;
    }
    return mglRenderFragmentNeedsPerSampleMSValues(fp) != 0;
}

/* Body of the former -[MGLRenderer emulatedMSColor0TextureForContext:].  The
 * attachment texture comes from the C port, so nothing here needs Objective-C. */
Texture *mglDrawEmulatedMSColor0Texture(GLMContext ctx)
{
    if (!ctx) {
        return NULL;
    }
    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!fbo || (fbo->color_attachment_bitfield & 1u) == 0u) {
        return NULL;
    }
    FBOAttachment *att = &fbo->color_attachments[0];
    Texture *tex = mglRendererAttachmentTextureFor(ctx, att);
    if (!tex) {
        return NULL;
    }
    if (!mglRenderIsEmulatedMSColorTexture((uint32_t)tex->target,
                                           (int32_t)tex->samples)) {
        return NULL;
    }
    /* Metal backing is created during processGLState; before the first draw
     * mtl_data may still be nil. All MS textures are emulated as array sample
     * planes, so the GL target/samples check is sufficient. */
    if (tex->mtl_data) {
        MGLRenderTextureInfo info = {0};
        (void)mglRenderGetTextureInfo(tex->mtl_data, &info);
        if (info.texture_type != MGLTextureType2DArray) {
            return NULL;
        }
    }
    return tex;
}

static uint32_t mglDrawSupportMinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

void mglDrawBindCullDistanceEmulationBuffers(void *renderer, uint32_t mode,
                                             uint32_t firstVertex,
                                             const uint32_t *explicitVertices,
                                             uint32_t explicitVertexCount,
                                             const MGLEncodeContext *encCtx)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    if (!areas.ctx || !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(areas.ctx, "bindCullDistanceEmu");
    if (!vao) {
        return;
    }
    Program *activeProgram = mglResolveProgramForStageFromState(areas.ctx, _VERTEX_SHADER);
    if (!activeProgram) {
        return;
    }
    explicitVertexCount = mglDrawSupportMinU32(explicitVertexCount, 4u);

    void *captureBuffer =
        mglRendererBackendGetCullDistanceCaptureBuffer(areas.backend);
    if (captureBuffer) {
        MGLCullDistanceEmuParams params;
        mglRenderFillCullDistanceEmuParams(
            mglRenderPrimitiveVertexCountForMode((uint32_t)mode), firstVertex,
            explicitVertices, explicitVertexCount, 0u, 32u,
            mglDrawSupportMinU32(activeProgram->cull_distance_count, 8u),
            areas.tess_cull_capture_first_instance,
            areas.tess_cull_capture_instance_stride, &params);
        mglRenderBindCullDistanceEmuSlots(encCtx->render_encoder_owner,
                                          captureBuffer,
                                          &params);
        mglBindingRecordLastBoundVertexBuffer(
            renderer, captureBuffer, 0,
            kMGLCullDistanceVertexBufferIndex);
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        mglBindingInvalidateLastBoundVertexBufferAtIndex(renderer, 
                  kMGLCullDistanceParamsBufferIndex);
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
                areas.ctx, vao, attribs[i], "bindCullDistanceEmu", &resolved)) {
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
        mglRendererBackendGetCullDistanceDummyBuffer(areas.backend));
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
    mglBindingRecordLastBoundVertexBuffer(renderer, cullMtlBuffer,
                                          0, kMGLCullDistanceVertexBufferIndex);
    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    mglBindingInvalidateLastBoundVertexBufferAtIndex(renderer, kMGLCullDistanceParamsBufferIndex);
}
