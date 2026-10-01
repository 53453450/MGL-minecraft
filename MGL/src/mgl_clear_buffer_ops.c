/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_clear_buffer_ops.c - C home of -mtlClearBuffer:type:mask: plus the two
 * default-draw-buffer creators it uses (P0-1, log 184).  The renderer entry
 * point mglRendererClearBuffer calls straight into this TU.
 */

#include "mgl_render_pass_sync_ops.h"
#include "mgl_clear_buffer_ops.h"

#include "mgl_attachment_binding.h" /* mglRendererBindFramebufferAttachmentTextures */
#include "mgl_renderer_binding_ops.h"  /* binding set / invalidate entries */
#include "mgl_blit_pipelines.h"     /* mglBlitClearRectPipeline / DepthState */
#include "mgl_draw_buffer.h"        /* mglMetal* draw-buffer mapping */
#include "mgl_renderer_backend.h"
#include "mgl_renderer_ports.h"
#include "mgl_sync.h"               /* MGLMetalAttachmentSubresource */
#include "mgl_texture_bind.h"       /* mglRendererBindMTLTexture */
#include "mgl_trace_log.h"
#include "mgl_frame_activity.h"       /* MGL_ENC_REASON_* */
#include "mgl_texture_compat.h"       /* mglMarkTextureLevelRenderTargetWritten */
#include "mgl_region_value.h"         /* MGLPdViewportValue */

#include "mgl_render.h"
#include "mgl_renderer_host.h"          /* mglMarkTextureLevelRenderTargetWritten */
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_encoder_ops.h" /* mglRenderPassNewCommandBufferLocked */

#include <limits.h>
#include <stdio.h>
#include <string.h>

/* The .m's file-local constants this TU needs (values copied verbatim from
 * MGLRenderer.m's enum). */
enum {
    MGL_PD_TEXTURE_USAGE_RENDER_TARGET = 4u,
    MGL_PD_STORAGE_PRIVATE = 2u,
    MGL_PD_PIXEL_FORMAT_INVALID = 0u,
    MGL_PD_DEPTH32_FLOAT = 252u,
    MGL_PD_LOAD_LOAD = 1u,
    MGL_PD_STORE_STORE = 1u,
    MGL_PD_PRIMITIVE_TRIANGLE_STRIP = 4u,
};

/* The .m's file-local structs this TU needs (declared in Objective-C private
 * headers, so they are restated here with the same layout). */
typedef struct MGLPdClearRectParams_t {
    vector_float4 color;
    float depth;
    vector_float3 _padding;
} MGLPdClearRectParams;

typedef struct MGLPdViewportValue_t {
    double origin_x;
    double origin_y;
    double width;
    double height;
    double znear;
    double zfar;
} MGLPdViewportValue;

typedef struct MGLPdScissorRectValue_t {
    uint64_t x;
    uint64_t y;
    uint64_t width;
    uint64_t height;
} MGLPdScissorRectValue;

/* Shell forwarder that applies the pending drawable size and returns it. */
extern MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);

/* MGLRenderer.m defines this; the only declaration is in an Objective-C
 * private header. */
void *mglBlitClearRectDepthState(void *renderer);

/* MGL_STATE(ctx) twin (same shape as mgl_render_pass_manager_ops.c). */
static GLMState *mglPdState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

static uint64_t mglPdMaxU64(uint64_t a, uint64_t b) { return a > b ? a : b; }

static MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo(texture, &info);
    return info;
}

/* Twin of the .m's mglRendererCreateTextureFromState (returns the +1 handle). */
static void *mglPdCreateTextureFromState(
    const MGLRenderTextureDescriptorState *state)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(state, NULL, &texture) == 0 && texture) {
        return texture;
    }
    return NULL;
}

/* Twins of the .m's encoder wrappers (they were file-local statics there). */
static void mglPdSetEncoderViewport(void *encoder, MGLPdViewportValue viewport)
{
    (void)mglRenderSetRenderViewport(encoder, viewport.origin_x,
                                     viewport.origin_y, viewport.width,
                                     viewport.height, viewport.znear,
                                     viewport.zfar);
}

static void mglPdSetEncoderScissor(void *encoder, MGLPdScissorRectValue scissor)
{
    (void)mglRenderSetRenderScissor(encoder, scissor.x, scissor.y, scissor.width,
                                    scissor.height);
}

static void mglPdSetEncoderPipeline(void *encoder, void *pipeline)
{
    (void)mglRenderSetRenderPipelineState(encoder, pipeline);
}

static void mglPdSetEncoderDepthStencil(void *encoder, void *state)
{
    (void)mglRenderSetRenderDepthStencilState(encoder, state);
}

static void mglPdSetEncoderBytes(void *encoder, const void *bytes,
                                 uint64_t length, uint32_t stage, uint64_t index)
{
    (void)mglRenderSetRenderBytes(encoder, bytes, length, stage,
                                  (uint32_t)index);
}

static void mglPdDrawPrimitives(void *encoder, uint32_t primitiveType,
                                uint64_t vertexStart, uint64_t vertexCount)
{
    (void)mglRenderEncodeDraw(encoder,
                              &(MGLRenderDrawPlan){
                                  .kind = MGL_RENDER_DRAW_ARRAY,
                                  .primitive_type = primitiveType,
                                  .vertex_start = vertexStart,
                                  .vertex_count = vertexCount,
                                  .instance_count = 1u,
                                  .base_instance = 0u,
                              },
                              NULL, 0);
}

/* -newDrawBuffer:isDepthStencil: */
void *mglRendererNewDrawBuffer(void *renderer, uint32_t pixelFormat,
                               int depthStencil)
{
    const MGLSizeValue drawableSize =
        mglPlatformShellApplyPendingDrawableSize(renderer);

    MGLRenderTextureDescriptorState state = {0};
    state.texture_type = 2u;
    state.pixel_format = pixelFormat;
    state.width = mglPdMaxU64(1u, drawableSize.width);
    state.height = mglPdMaxU64(1u, drawableSize.height);
    state.depth = 1u;
    state.mipmap_level_count = 1u;
    state.sample_count = 1u;
    state.array_length = 1u;
    state.usage = MGL_PD_TEXTURE_USAGE_RENDER_TARGET;
    state.storage_mode = depthStencil ? MGL_PD_STORAGE_PRIVATE : 0u;
    void *texture = mglPdCreateTextureFromState(&state);
    if (!texture) {
        fprintf(stderr,
                "MGL DRAWBUFFER ERROR: failed to create draw buffer texture format=%lu size=%lux%lu\n",
                (unsigned long)pixelFormat, (unsigned long)state.width,
                (unsigned long)state.height);
        return NULL;
    }

    return texture;
}

/* -newDrawBufferWithCustomSize:isDepthStencil:customSize: */
void *mglRendererNewDrawBufferWithCustomSize(uint32_t pixelFormat,
                                             int depthStencil, uint64_t width,
                                             uint64_t height)
{
    MGLRenderTextureDescriptorState state = {0};
    state.texture_type = 2u;
    state.pixel_format = pixelFormat;
    state.width = mglPdMaxU64(1u, width);
    state.height = mglPdMaxU64(1u, height);
    state.depth = 1u;
    state.mipmap_level_count = 1u;
    state.sample_count = 1u;
    state.array_length = 1u;
    state.usage = MGL_PD_TEXTURE_USAGE_RENDER_TARGET;
    state.storage_mode = depthStencil ? MGL_PD_STORAGE_PRIVATE : 0u;
    void *texture = mglPdCreateTextureFromState(&state);
    if (!texture) {
        fprintf(stderr,
                "MGL DRAWBUFFER ERROR: failed to create custom draw buffer texture format=%lu size=%lux%lu\n",
                (unsigned long)pixelFormat, (unsigned long)state.width,
                (unsigned long)state.height);
        return NULL;
    }

    return texture;
}

/* -mtlClearBuffer:type:mask: */
static int mglPdPassAttachmentIs(void *renderPassStateOwner, uint32_t kind,
                                 void *texture,
                                 MGLMetalAttachmentSubresource subresource)
{
    void *rpTexture = mglRenderGetRenderPassAttachmentTextureOwner(
        renderPassStateOwner, kind, 0);
    if (!texture) {
        return rpTexture == NULL;
    }
    uint64_t rpLevel = 0u, rpSlice = 0u, rpDepthPlane = 0u;
    mglRenderGetRenderPassAttachmentSubresourceOwner(
        renderPassStateOwner, kind, 0, &rpLevel, &rpSlice, &rpDepthPlane);
    return rpTexture == texture && rpLevel == subresource.level &&
           rpSlice == subresource.slice;
}

void mglRendererMTLClearBuffer(void *renderer, GLMContext glm_ctx,
                               unsigned int flags, unsigned int mask)
{
    if (!glm_ctx || !mglRenderClearMaskHasAny((uint32_t)mask)) {
        return;
    }

    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMState *glState = mglPdState(&areas);
    MGLCommandState *commandState = areas.command;
    mglPlatformShellSetContext(renderer, glm_ctx);

    const int scissored = glState->caps.scissor_test != 0;
    if (!scissored && !(flags & MGL_RENDERER_CLEAR_MASKED)) {
        mglRendererEndRenderEncodingLocked(renderer);

        MGLRenderCommandBufferState clearCommandState = {0};
        if (!mglRenderCommandBufferOwnerHasState(
                commandState->currentCommandBufferOwner,
                &clearCommandState) &&
            !mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: immediate clear failed to create command buffer\n");
            return;
        }

        Framebuffer *fbo = glState->framebuffer;
        if (fbo && (fbo->dirty_bits & DIRTY_FBO_BINDING)) {
            if (!mglRendererBindFramebufferAttachmentTextures(renderer)) {
                fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
                return;
            }
            fbo->dirty_bits &= ~DIRTY_FBO_BINDING;
        }

        if (!mglRenderPassNewRenderEncoderLockedWithReason(
                renderer, MGL_ENC_REASON_CLEAR)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return;
        }
        mglRendererEndRenderEncodingLocked(renderer);
        mglMarkRendererDirtyBits(glm_ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return;
    }

    const GLint rawX = scissored ? glState->var.scissor_box[0] : 0;
    const GLint rawY = scissored ? glState->var.scissor_box[1] : 0;
    const GLint rawW = scissored ? glState->var.scissor_box[2] : INT_MAX / 2;
    const GLint rawH = scissored ? glState->var.scissor_box[3] : INT_MAX / 2;
    if (rawW <= 0 || rawH <= 0) {
        return;
    }

    Framebuffer *fbo = glState->framebuffer;
    Texture *colorTexObj = NULL;
    Texture *depthTexObj = NULL;
    FBOAttachment *colorAttachment = NULL;
    FBOAttachment *depthAttachment = NULL;
    void *colorTexture = NULL;
    void *depthTexture = NULL;
    MGLMetalAttachmentSubresource colorSubresource = {0u, 0u, 0u};
    MGLMetalAttachmentSubresource depthSubresource = {0u, 0u, 0u};
    Texture *stencilTexObj = NULL;
    FBOAttachment *stencilAttachment = NULL;
    void *stencilTexture = NULL;
    MGLMetalAttachmentSubresource stencilSubresource = {0u, 0u, 0u};

    int wantsColor = mglRenderClearMaskHasColor((uint32_t)mask) != 0;
    int wantsDepth = mglRenderClearMaskHasDepth((uint32_t)mask) != 0 &&
                     glState->var.depth_writemask;
    /* Clears use the front stencil write mask (§17.4.2). */
    const uint32_t stencilWriteMask = glState->var.stencil_writemask & 0xffu;
    int wantsStencil = mglRenderClearMaskHasStencil((uint32_t)mask) != 0 &&
                       stencilWriteMask != 0u;

    const uint32_t colorWriteMask = mglRenderColorWriteMaskFromChannels(
        glState->caps.use_color_mask[0] ? 1 : 0,
        glState->var.color_writemask[0][0] ? 1 : 0,
        glState->var.color_writemask[0][1] ? 1 : 0,
        glState->var.color_writemask[0][2] ? 1 : 0,
        glState->var.color_writemask[0][3] ? 1 : 0);
    if (colorWriteMask == MGLColorWriteMaskNone) {
        wantsColor = 0;
    }

    if (fbo) {
        if (wantsColor) {
            const GLsizei drawBufferCount = mglMetalDrawBufferCount(glm_ctx);
            for (GLsizei slot = 0; slot < drawBufferCount; ++slot) {
                GLuint attachmentIndex = 0u;
                if (!mglMetalResolveFboDrawAttachmentIndex(
                        glm_ctx, mglMetalDrawBufferAt(glm_ctx, (GLuint)slot),
                        &attachmentIndex) ||
                    attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
                    ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) ==
                        0u) {
                    continue;
                }

                colorAttachment = &fbo->color_attachments[attachmentIndex];
                colorTexObj =
                    mglRendererAttachmentTextureFor(glm_ctx, colorAttachment);
                if (!colorTexObj) {
                    continue;
                }

                colorTexObj->is_render_target = 1;
                if (!mglRendererBindMTLTexture(renderer, colorTexObj) ||
                    !colorTexObj->mtl_data) {
                    colorTexObj = NULL;
                    colorAttachment = NULL;
                    continue;
                }

                colorTexture = colorTexObj->mtl_data;
                colorSubresource =
                    mglMetalAttachmentSubresourceForAttachment(colorAttachment);
                break;
            }

            if (!colorTexture) {
                wantsColor = 0;
            }
        }

        if (wantsDepth && fbo->depth.texture) {
            depthAttachment = &fbo->depth;
            depthTexObj = mglRendererAttachmentTextureFor(glm_ctx, depthAttachment);
            if (depthTexObj) {
                depthTexObj->is_render_target = 1;
                if (mglRendererBindMTLTexture(renderer, depthTexObj) &&
                    depthTexObj->mtl_data) {
                    depthTexture = depthTexObj->mtl_data;
                    depthSubresource =
                        mglMetalAttachmentSubresourceForAttachment(
                            depthAttachment);
                }
            }
        }
        if (wantsDepth && !depthTexture) {
            wantsDepth = 0;
        }

        if (wantsStencil && fbo->stencil.texture) {
            stencilAttachment = &fbo->stencil;
            stencilTexObj =
                mglRendererAttachmentTextureFor(glm_ctx, stencilAttachment);
            if (stencilTexObj) {
                stencilTexObj->is_render_target = 1;
                if (mglRendererBindMTLTexture(renderer, stencilTexObj) &&
                    stencilTexObj->mtl_data) {
                    stencilTexture = stencilTexObj->mtl_data;
                    stencilSubresource =
                        mglMetalAttachmentSubresourceForAttachment(
                            stencilAttachment);
                }
            }
        }
        /* Draw passes attach packed formats as depth too; matching that lets
         * the clear reuse the current encoder. Depth writes stay off unless
         * depth is cleared. */
        if (wantsStencil && stencilTexture && !depthTexture &&
            mglRenderPixelFormatIsPackedDepthStencil(
                mglPdTextureInfo(stencilTexture).pixel_format)) {
            depthTexture = stencilTexture;
            depthSubresource = stencilSubresource;
        }
    } else {
        const GLuint drawBufferIndex =
            mglDefaultDrawBufferIndexForGL(glState->draw_buffer);
        if (wantsColor) {
            if (mglRenderDefaultDrawBufferIsFront(drawBufferIndex)) {
                if (!areas.drawable && areas.layer) {
                    (void)mglPlatformShellApplyPendingDrawableSize(renderer);
                    (void)mglRendererNextDrawable(renderer);
                }
                colorTexture = mglRendererDrawableTexture(renderer);
            } else if (mglRenderDefaultDrawBufferIsOffscreen(
                           drawBufferIndex, _MAX_DRAW_BUFFERS)) {
                colorTexture = mglRendererBackendGetDefaultDrawBufferAttachment(
                    areas.backend, drawBufferIndex,
                    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR);
                if (!colorTexture) {
                    colorTexture = mglRendererNewDrawBuffer(
                        renderer, glm_ctx->pixel_format.mtl_pixel_format, 0);
                    (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                        areas.backend, drawBufferIndex,
                        MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR,
                        colorTexture);
                }
            }
            if (!colorTexture) {
                wantsColor = 0;
            }
        }

        if (wantsDepth && drawBufferIndex < _MAX_DRAW_BUFFERS) {
            depthTexture = mglRendererBackendGetDefaultDrawBufferAttachment(
                areas.backend, drawBufferIndex,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH);
            if (!depthTexture) {
                uint32_t depthFormat = glm_ctx->depth_format.mtl_pixel_format;
                if (depthFormat == MGL_PD_PIXEL_FORMAT_INVALID) {
                    depthFormat = MGL_PD_DEPTH32_FLOAT;
                }
                const uint64_t depthWidth =
                    colorTexture ? mglPdTextureInfo(colorTexture).width
                                 : mglPdMaxU64(glState->viewport[2], 1);
                const uint64_t depthHeight =
                    colorTexture ? mglPdTextureInfo(colorTexture).height
                                 : mglPdMaxU64(glState->viewport[3], 1);
                depthTexture = mglRendererNewDrawBufferWithCustomSize(
                    depthFormat, 1, depthWidth, depthHeight);
                (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                    areas.backend, drawBufferIndex,
                    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH,
                    depthTexture);
            }
            if (!depthTexture) {
                wantsDepth = 0;
            }
        }

        if (wantsStencil && drawBufferIndex < _MAX_DRAW_BUFFERS) {
            stencilTexture = mglRendererBackendGetDefaultDrawBufferAttachment(
                areas.backend, drawBufferIndex,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL);
            if (!stencilTexture && glm_ctx->stencil_format.format) {
                const uint32_t stencilFormat =
                    mglRenderRepairedDefaultStencilFormat(
                        glm_ctx->stencil_format.mtl_pixel_format);
                const uint64_t stencilWidth =
                    colorTexture ? mglPdTextureInfo(colorTexture).width
                                 : mglPdMaxU64(glState->viewport[2], 1);
                const uint64_t stencilHeight =
                    colorTexture ? mglPdTextureInfo(colorTexture).height
                                 : mglPdMaxU64(glState->viewport[3], 1);
                stencilTexture = mglRendererNewDrawBufferWithCustomSize(
                    stencilFormat, 1, stencilWidth, stencilHeight);
                (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                    areas.backend, drawBufferIndex,
                    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL,
                    stencilTexture);
            }
        }
    }

    if (!stencilTexture) {
        wantsStencil = 0;
    }
    if (!wantsColor && !wantsDepth && !wantsStencil) {
        return;
    }

    uint64_t passWidth = 0u;
    uint64_t passHeight = 0u;
    void *sizeTexture = colorTexture ? colorTexture
                                     : (depthTexture ? depthTexture
                                                     : stencilTexture);
    if (sizeTexture) {
        passWidth = mglPdTextureInfo(sizeTexture).width;
        passHeight = mglPdTextureInfo(sizeTexture).height;
    }
    if (passWidth == 0u || passHeight == 0u) {
        return;
    }

    GLint x0 = rawX;
    GLint y0 = rawY;
    GLint x1 = rawX + rawW;
    GLint y1 = rawY + rawH;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (GLint)passWidth) x1 = (GLint)passWidth;
    if (y1 > (GLint)passHeight) y1 = (GLint)passHeight;
    if (x1 <= x0 || y1 <= y0) {
        return;
    }

    const GLint clearW = x1 - x0;
    const GLint clearH = y1 - y0;
    GLint metalY = y0;
    if (mglRenderClipOriginIsLowerLeft((uint32_t)glState->var.clip_origin)) {
        metalY = (GLint)passHeight - y1;
        if (metalY < 0) {
            metalY = 0;
        }
    }

    const uint32_t colorFormat = colorTexture
                                     ? mglPdTextureInfo(colorTexture).pixel_format
                                     : MGL_PD_PIXEL_FORMAT_INVALID;
    const uint32_t depthFormat = depthTexture
                                     ? mglPdTextureInfo(depthTexture).pixel_format
                                     : MGL_PD_PIXEL_FORMAT_INVALID;
    const uint32_t stencilFormat =
        wantsStencil ? mglPdTextureInfo(stencilTexture).pixel_format
                     : MGL_PD_PIXEL_FORMAT_INVALID;
    void *pipeline = mglBlitClearRectPipeline(
        renderer, colorFormat, depthFormat, stencilFormat,
        wantsColor ? colorWriteMask : MGLColorWriteMaskNone, wantsDepth);
    if (!pipeline) {
        fprintf(stderr,
                "MGL ERROR: scissored clear missing pipeline color=%lu depth=%lu stencil=%lu wantsColor=%d wantsDepth=%d\n",
                (unsigned long)colorFormat, (unsigned long)depthFormat,
                (unsigned long)stencilFormat, wantsColor ? 1 : 0,
                wantsDepth ? 1 : 0);
        return;
    }

    void *depthStencilState =
        wantsDepth ? mglBlitClearRectDepthState(renderer) : NULL;
    if (wantsStencil) {
        MGLRenderStencilDescriptorState stencilDesc = {0};
        stencilDesc.present = 1u;
        stencilDesc.compare_function = MGLCompareFunctionAlways;
        stencilDesc.read_mask = 0xffu;
        stencilDesc.write_mask = stencilWriteMask;
        uint32_t replace = 0u;
        (void)mglRenderStencilOpFromGL(GL_REPLACE, &replace);
        stencilDesc.stencil_failure_operation = replace;
        stencilDesc.depth_failure_operation = replace;
        stencilDesc.depth_stencil_pass_operation = replace;
        MGLRenderDepthStencilDescriptorState dsDesc = {0};
        dsDesc.depth_compare_function = MGLCompareFunctionAlways;
        dsDesc.depth_write_enabled = wantsDepth ? 1u : 0u;
        dsDesc.front = stencilDesc;
        dsDesc.back = stencilDesc;
        depthStencilState = NULL;
        if (areas.pipeline_cache_depth_stencil_state_for_value_state) {
            areas.pipeline_cache_depth_stencil_state_for_value_state(
                areas.pipeline_cache_object, &dsDesc, &depthStencilState);
        }
    }
    const uint32_t stencilClearValue = glState->var.stencil_clear_value & 0xffu;

    MGLPdClearRectParams params;
    params.color = (vector_float4){
        glState->color_clear_value[0], glState->color_clear_value[1],
        glState->color_clear_value[2], glState->color_clear_value[3]};
    params.depth = (float)glState->var.depth_clear_value;
    params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

    const MGLPdViewportValue viewport = {
        .origin_x = 0.0,
        .origin_y = 0.0,
        .width = (double)passWidth,
        .height = (double)passHeight,
        .znear = 0.0,
        .zfar = 1.0};
    const MGLPdScissorRectValue scissor = {.x = (uint64_t)x0,
                                         .y = (uint64_t)metalY,
                                         .width = (uint64_t)clearW,
                                         .height = (uint64_t)clearH};

    /* Optimization: reuse the current render encoder when it targets the same
     * framebuffer attachments we're about to clear. This avoids ending the
     * current encoder and creating a dedicated encoder for every scissored clear
     * (3-8 times per frame in MC).
     *
     * Conditions: an encoder is active, the render pass matches the current FBO,
     * no visibility query is active (which would require an encoder rebuild to
     * attach the visibility buffer), and the render pass's color attachment 0 /
     * depth / stencil attachments are exactly the ones the clear pipeline
     * declares (Metal rejects a pipeline whose formats differ from the pass).
     * When any condition fails, fall back to the original endRenderEncoding +
     * new-encoder path. */
    int canReuseCurrentEncoder = 0;
    uint32_t sampleQueryActive = 0;
    if (areas.query_state_owner) {
        mglRenderIsSampleQueryActive(areas.query_state_owner, &sampleQueryActive);
    }
    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1 &&
        mglRenderPassMatchesCurrentFramebuffer(renderer) &&
        !sampleQueryActive) {
        if (commandState->renderPassStateOwner) {
            canReuseCurrentEncoder =
                mglPdPassAttachmentIs(commandState->renderPassStateOwner,
                                      MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                                      wantsColor ? colorTexture : NULL,
                                      colorSubresource) &&
                mglPdPassAttachmentIs(commandState->renderPassStateOwner,
                                      MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH,
                                      depthTexture, depthSubresource) &&
                mglPdPassAttachmentIs(commandState->renderPassStateOwner,
                                      MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL,
                                      wantsStencil ? stencilTexture : NULL,
                                      stencilSubresource);
        }
    }

    if (canReuseCurrentEncoder) {
        /* areas.binding_state_owner is the ADDRESS of the owner slot (see
         * mgl_renderer_ports.h); every binding call needs the handle.  The .m
         * this path came from passed the ivar itself. */
        void *bindingOwner =
            areas.binding_state_owner ? *areas.binding_state_owner : NULL;
        mglRenderBindingSetViewportForOwner(
            bindingOwner, commandState->currentRenderEncoderOwner,
            viewport.origin_x, viewport.origin_y, viewport.width, viewport.height,
            viewport.znear, viewport.zfar);
        mglRenderBindingSetScissorForOwner(
            bindingOwner, commandState->currentRenderEncoderOwner,
            scissor.x, scissor.y, scissor.width, scissor.height);
        mglRenderSetRenderPipelineStateForOwner(
            commandState->currentRenderEncoderOwner, pipeline);
        mglRenderBindingSetPipelineState(bindingOwner, pipeline);
        if (depthStencilState) {
            mglRenderSetRenderDepthStencilStateForOwner(
                commandState->currentRenderEncoderOwner, depthStencilState);
            mglRenderBindingSetDepthStencilState(bindingOwner,
                                                 depthStencilState);
        }
        if (wantsStencil) {
            mglRenderSetStencilReferenceValuesForOwner(
                commandState->currentRenderEncoderOwner, stencilClearValue,
                stencilClearValue);
        }
        mglRenderSetRenderBytesForOwner(
            commandState->currentRenderEncoderOwner, &params, sizeof(params),
            MGL_RENDER_BINDING_STAGE_VERTEX, 0);
        mglRenderBindingInvalidateVertexBuffer(bindingOwner, 0);
        if (wantsColor) {
            mglRenderSetRenderBytesForOwner(
                commandState->currentRenderEncoderOwner, &params, sizeof(params),
                MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
            mglRenderBindingInvalidateFragmentBuffer(bindingOwner, 0);
        }
        const MGLRenderDrawPlan clearDraw = {
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)MGL_PD_PRIMITIVE_TRIANGLE_STRIP,
            .vertex_start = 0u,
            .vertex_count = 4u,
            .instance_count = 1u,
        };
        (void)mglRenderEncodeDrawForRenderEncoderOwner(
            commandState->currentRenderEncoderOwner, &clearDraw, NULL, 0);

        if (wantsColor && colorTexObj && colorAttachment) {
            colorAttachment->clear_bitmask = (GLbitfield)mglRenderClearMaskClearColor(
                (uint32_t)colorAttachment->clear_bitmask);
            mglMarkTextureLevelRenderTargetWrittenImpl(
                colorTexObj, colorAttachment->level, __func__, __LINE__);
        }
        if (wantsDepth && depthTexObj && depthAttachment) {
            depthAttachment->clear_bitmask = (GLbitfield)mglRenderClearMaskClearDepth(
                (uint32_t)depthAttachment->clear_bitmask);
            mglMarkTextureLevelRenderTargetWrittenImpl(
                depthTexObj, depthAttachment->level, __func__, __LINE__);
        }
        if (wantsStencil && stencilTexObj && stencilAttachment) {
            stencilAttachment->clear_bitmask = (GLbitfield)mglRenderClearMaskClearStencil(
                (uint32_t)stencilAttachment->clear_bitmask);
            mglMarkTextureLevelRenderTargetWrittenImpl(
                stencilTexObj, stencilAttachment->level, __func__, __LINE__);
        }

        mglMarkRendererDirtyBits(glm_ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        return;
    }

    /* Fallback: end the current encoder and create a dedicated clear encoder.
     * Used when no encoder is active, the FBO doesn't match, a visibility query
     * is active, or the attachment textures don't match. */
    mglRendererEndRenderEncodingLocked(renderer);
    MGLRenderCommandBufferState clearCommandState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            commandState->currentCommandBufferOwner, &clearCommandState) &&
        !mglRenderPassNewCommandBufferLocked(renderer)) {
        fprintf(stderr,
                "MGL ERROR: scissored clear failed to create command buffer\n");
        return;
    }

    MGLRenderPassState clearState = {0};
    if (colorTexture) {
        clearState.color[0].attachment.texture = colorTexture;
        clearState.color[0].attachment.level = colorSubresource.level;
        clearState.color[0].attachment.slice = colorSubresource.slice;
        clearState.color[0].attachment.depth_plane = colorSubresource.depthPlane;
        clearState.color[0].attachment.load_action = MGL_PD_LOAD_LOAD;
        clearState.color[0].attachment.store_action = MGL_PD_STORE_STORE;
    }
    if (depthTexture) {
        clearState.depth.attachment.texture = depthTexture;
        clearState.depth.attachment.level = depthSubresource.level;
        clearState.depth.attachment.slice = depthSubresource.slice;
        clearState.depth.attachment.depth_plane = depthSubresource.depthPlane;
        clearState.depth.attachment.load_action = MGL_PD_LOAD_LOAD;
        clearState.depth.attachment.store_action = MGL_PD_STORE_STORE;
    }
    if (wantsStencil) {
        clearState.stencil.attachment.texture = stencilTexture;
        clearState.stencil.attachment.level = stencilSubresource.level;
        clearState.stencil.attachment.slice = stencilSubresource.slice;
        clearState.stencil.attachment.depth_plane = stencilSubresource.depthPlane;
        clearState.stencil.attachment.load_action = MGL_PD_LOAD_LOAD;
        clearState.stencil.attachment.store_action = MGL_PD_STORE_STORE;
    }
    clearState.render_target_width = passWidth;
    clearState.render_target_height = passHeight;

    void *clearEncoder = mglRenderCreateRenderEncoderBorrowed(
        commandState->currentCommandBufferOwner, &clearState);
    if (!clearEncoder) {
        fprintf(stderr,
                "MGL ERROR: scissored clear failed to create render encoder\n");
        return;
    }

    mglPdSetEncoderViewport(clearEncoder, viewport);
    mglPdSetEncoderScissor(clearEncoder, scissor);
    mglPdSetEncoderPipeline(clearEncoder, pipeline);
    if (depthStencilState) {
        mglPdSetEncoderDepthStencil(clearEncoder, depthStencilState);
    }
    if (wantsStencil) {
        (void)mglRenderSetStencilReferenceValues(
            clearEncoder, stencilClearValue, stencilClearValue);
    }
    mglPdSetEncoderBytes(clearEncoder, &params, sizeof(params),
                         MGL_RENDER_BINDING_STAGE_VERTEX, 0);
    if (wantsColor) {
        mglPdSetEncoderBytes(clearEncoder, &params, sizeof(params),
                             MGL_RENDER_BINDING_STAGE_FRAGMENT, 0);
    }
    mglPdDrawPrimitives(clearEncoder, MGL_PD_PRIMITIVE_TRIANGLE_STRIP, 0, 4);
    mglRenderEndRenderEncoder(clearEncoder);

    if (wantsColor && colorTexObj && colorAttachment) {
        colorAttachment->clear_bitmask = (GLbitfield)mglRenderClearMaskClearColor(
            (uint32_t)colorAttachment->clear_bitmask);
        mglMarkTextureLevelRenderTargetWrittenImpl(
            colorTexObj, colorAttachment->level, __func__, __LINE__);
    }
    if (wantsDepth && depthTexObj && depthAttachment) {
        depthAttachment->clear_bitmask = (GLbitfield)mglRenderClearMaskClearDepth(
            (uint32_t)depthAttachment->clear_bitmask);
        mglMarkTextureLevelRenderTargetWrittenImpl(
            depthTexObj, depthAttachment->level, __func__, __LINE__);
    }
    if (wantsStencil && stencilTexObj && stencilAttachment) {
        stencilAttachment->clear_bitmask = (GLbitfield)mglRenderClearMaskClearStencil(
            (uint32_t)stencilAttachment->clear_bitmask);
        mglMarkTextureLevelRenderTargetWrittenImpl(
            stencilTexObj, stencilAttachment->level, __func__, __LINE__);
    }

    mglMarkRendererDirtyBits(glm_ctx->active_state,
                             DIRTY_FBO | DIRTY_RENDER_STATE);
}
