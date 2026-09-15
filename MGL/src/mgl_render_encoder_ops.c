/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_encoder_ops.c - C homes of the render-encoder creation path: the
 * encoder entry point plus the attachment, load/store-action, transient-depth
 * and clear-resolve layers it drives (P0-1, log 186).
 */

#include "mgl_render_pass_sync_ops.h"
#include "mgl_render_encoder_ops.h"

#include "mgl_binding_state_ops.h"   /* mglBindingInvalidateLastBoundState */
#include "mgl_buffer_slots.h"
#include "mgl_draw_buffer.h"         /* mglMetal* draw-buffer mapping */
#include "mgl_frame_activity.h"      /* MGL_PERF_INC / MGL_ENC_REASON_* */
#include "mgl_gpu_recovery.h"        /* record GPU error / success */
#include "mgl_metal_ref.h"
#include "mgl_pso_format_class.h"
#include "mgl_render_pass_manager.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_clear_buffer_ops.h"     /* draw-buffer creators + encoder-free clear */
#include "mgl_render_pass_plan.h"    /* load/store planning */
#include "mgl_render_pass_clear.h"   /* mglRenderPassPlanClearValues */
#include "mgl_renderer_backend.h"
#include "mgl_renderer_ports.h"
#include "mgl_stage_encode_drivers.h" /* stage encode bind drivers */
#include "mgl_sync.h"                /* MGLMetalAttachmentSubresource */
#include "mgl_texture_compat.h"      /* mglMarkTextureLevelRenderTargetWrittenImpl */
#include "mgl_trace_log.h"

#include "mgl_render.h"

#include <stdio.h>
#include <string.h>

/* The .m's file-local constants this TU needs (values copied verbatim). */
enum {
    MGL_PD_TEXTURE_USAGE_RENDER_TARGET = 4u,
    MGL_PD_STORAGE_MODE_PRIVATE = 2u,
};

/* Objective-C private header declarations restated for C. */
extern GLuint mglRendererSafeFramebufferName(GLMContext ctx);
extern void *mglApplySRGBStateToRenderTarget(void *texture, GLMContext ctx);
extern int mglEnvFlagEnabled(const char *name);
extern int mglShouldTraceCallCompat(uint64_t count);

/* Shell forwarders. */
extern MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);
extern void *mglPlatformShellDrawablePointer(void *renderer);
extern int mglPlatformShellAutoreleasePoolCall(void *renderer,
                                               int (*body)(void *));

/* MGLRenderer+RenderPass_Private.h has this as a `static inline`. */
static bool mglPdShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

/* MGL_STATE(ctx) twin. */
static GLMState *mglPdStateOf(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

static MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo(texture, &info);
    return info;
}

/* === render-pass state twins ========================================== */

static const MGLRenderPassAttachmentState *mglPdAttachmentStateFromSnapshot(
    const MGLRenderPassState *state, uint32_t attachmentKind,
    size_t colorIndex);

static bool mglPdColorAttachmentIndexValid(uint32_t colorIndex, uint32_t limit)
{
    return colorIndex < limit;
}

static bool mglPdGetPersistentState(const MGLCommandState *commandState,
                                    MGLRenderPassState *stateOut)
{
    return commandState && stateOut && commandState->renderPassStateOwner &&
           mglRenderGetRenderPassStateOwner(commandState->renderPassStateOwner,
                                            stateOut) == 0;
}

static bool mglPdGetPersistentAttachmentState(
    const MGLCommandState *commandState, uint32_t attachmentKind,
    size_t colorIndex, MGLRenderPassAttachmentState *attachmentOut)
{
    if (!attachmentOut) return false;
    MGLRenderPassState state = {0};
    if (!mglPdGetPersistentState(commandState, &state)) return false;
    const MGLRenderPassAttachmentState *attachment =
        mglPdAttachmentStateFromSnapshot(&state, attachmentKind, colorIndex);
    if (!attachment) return false;
    *attachmentOut = *attachment;
    return true;
}

/* The .m's attachment-state helpers (mgl_render_pass_manager_ops.c has the same
 * statics; C hosts carry their own copy). */
static uint32_t mglPdAttachmentClass(uint32_t attachmentKind)
{
    switch (attachmentKind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        return 1u;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        return 2u;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        return 3u;
    default:
        return 0u;
    }
}

static const MGLRenderPassAttachmentState *mglPdAttachmentStateFromSnapshot(
    const MGLRenderPassState *state, uint32_t attachmentKind, size_t colorIndex)
{
    if (!state) return NULL;
    switch (mglPdAttachmentClass(attachmentKind)) {
    case 1:
        return mglPdColorAttachmentIndexValid((uint32_t)colorIndex,
                                                      MAX_COLOR_ATTACHMENTS)
                   ? &state->color[colorIndex].attachment
                   : NULL;
    case 2:
        return &state->depth.attachment;
    case 3:
        return &state->stencil.attachment;
    default:
        return NULL;
    }
}

static void *mglPdAttachmentTextureFor(const MGLCommandState *commandState,
                                       uint32_t attachmentKind,
                                       size_t colorIndex)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &attachment)) {
        return attachment.texture;
    }
    return NULL;
}

static void *mglPdColorTextureFor(const MGLCommandState *commandState,
                                  size_t colorIndex)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorIndex);
}

static void *mglPdDepthTextureFor(const MGLCommandState *commandState)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u);
}

static void *mglPdStencilTextureFor(const MGLCommandState *commandState)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u);
}

static bool mglPdRenderTargetSizeFor(const MGLCommandState *commandState,
                                     uint64_t *widthOut, uint64_t *heightOut)
{
    MGLRenderPassState state = {0};
    if (!mglPdGetPersistentState(commandState, &state)) return false;
    if (widthOut) *widthOut = state.render_target_width;
    if (heightOut) *heightOut = state.render_target_height;
    return true;
}

static uint64_t mglPdRenderTargetWidthFor(const MGLCommandState *commandState)
{
    uint64_t width = 0;
    if (mglPdRenderTargetSizeFor(commandState, &width, NULL)) return width;
    return 0;
}

static uint64_t mglPdRenderTargetHeightFor(const MGLCommandState *commandState)
{
    uint64_t height = 0;
    if (mglPdRenderTargetSizeFor(commandState, NULL, &height)) return height;
    return 0;
}

static void mglPdSetPersistentAttachment(const MGLCommandState *commandState,
                                         uint32_t attachmentKind,
                                         size_t colorIndex, void *texture,
                                         uint64_t level, uint64_t slice,
                                         uint64_t depthPlane, int layered)
{
    if (commandState && commandState->renderPassStateOwner) {
        (void)mglRenderSetRenderPassStateAttachmentTexture(
            commandState->renderPassStateOwner, attachmentKind,
            (uint32_t)colorIndex, texture, level, slice, depthPlane,
            layered ? 1u : 0u);
    }
}

static void mglPdSetPersistentDimensions(const MGLCommandState *commandState,
                                         uint64_t width, uint64_t height)
{
    if (commandState && commandState->renderPassStateOwner) {
        (void)mglRenderSetRenderPassStateDimensions(
            commandState->renderPassStateOwner, width, height);
    }
}

static void mglPdSetPersistentActions(const MGLCommandState *commandState,
                                      uint32_t attachmentKind,
                                      size_t colorIndex, uint32_t loadAction,
                                      uint32_t storeAction)
{
    if (!commandState) return;
    MGLRenderPassAttachmentState state = {0};
    if (!mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                           colorIndex, &state)) {
        return;
    }
    if (commandState->renderPassStateOwner) {
        (void)mglRenderSetRenderPassStateAttachmentActions(
            commandState->renderPassStateOwner, attachmentKind,
            (uint32_t)colorIndex, loadAction, storeAction,
            state.store_action_options);
    }
}

static void mglPdSetPersistentLoadAction(const MGLCommandState *commandState,
                                         uint32_t attachmentKind,
                                         size_t colorIndex,
                                         uint32_t loadAction)
{
    uint32_t storeAction = (uint32_t)MGLStoreActionDontCare;
    MGLRenderPassAttachmentState state = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &state)) {
        storeAction = (uint32_t)state.store_action;
    } else {
        return;
    }
    mglPdSetPersistentActions(commandState, attachmentKind, colorIndex,
                              loadAction, storeAction);
}

static void mglPdSetPersistentStoreAction(const MGLCommandState *commandState,
                                          uint32_t attachmentKind,
                                          size_t colorIndex,
                                          uint32_t storeAction)
{
    uint32_t loadAction = (uint32_t)MGLLoadActionDontCare;
    MGLRenderPassAttachmentState state = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &state)) {
        loadAction = (uint32_t)state.load_action;
    } else {
        return;
    }
    mglPdSetPersistentActions(commandState, attachmentKind, colorIndex,
                              loadAction, storeAction);
}

static uint32_t mglPdLoadActionFor(const MGLCommandState *commandState,
                                   uint32_t attachmentKind, size_t colorIndex,
                                   uint32_t fallback)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &attachment)) {
        return attachment.load_action;
    }
    return fallback;
}

static uint32_t mglPdStoreActionFor(const MGLCommandState *commandState,
                                    uint32_t attachmentKind, size_t colorIndex,
                                    uint32_t fallback)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &attachment)) {
        return attachment.store_action;
    }
    return fallback;
}

static bool mglPdClearValuesFor(const MGLCommandState *commandState,
                                uint32_t attachmentKind, size_t colorIndex,
                                double *clearColorOut, double *clearDepthOut,
                                uint32_t *clearStencilOut)
{
    MGLRenderPassState state = {0};
    if (!mglPdGetPersistentState(commandState, &state)) return false;
    return mglRenderPassPlanClearValues(&state, attachmentKind,
                                        (uint32_t)colorIndex, clearColorOut,
                                        clearDepthOut, clearStencilOut) != 0;
}

static void mglPdClearColorFor(const MGLCommandState *commandState,
                               uint32_t colorIndex, double fallback[4],
                               double out[4])
{
    double rgba[4] = {0.0, 0.0, 0.0, 0.0};
    if (mglPdClearValuesFor(commandState,
                            MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorIndex,
                            rgba, NULL, NULL)) {
        for (int i = 0; i < 4; i++) out[i] = rgba[i];
        return;
    }
    for (int i = 0; i < 4; i++) out[i] = fallback ? fallback[i] : 0.0;
}

static double mglPdClearDepthFor(const MGLCommandState *commandState,
                                 double fallback)
{
    double depth = 0.0;
    if (mglPdClearValuesFor(commandState,
                            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u, NULL,
                            &depth, NULL)) {
        return depth;
    }
    return fallback;
}

static uint32_t mglPdClearStencilFor(const MGLCommandState *commandState,
                                     uint32_t fallback)
{
    uint32_t stencil = 0u;
    if (mglPdClearValuesFor(commandState,
                            MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u, NULL,
                            NULL, &stencil)) {
        return stencil;
    }
    return fallback;
}

/* .m statics of the default-framebuffer / transient-depth path. */
static void *mglPdDefaultDrawBufferAttachment(
    MGLRendererBackendHandle *backend, GLuint drawBufferIndex,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind)
{
    return mglRendererBackendGetDefaultDrawBufferAttachment(backend,
                                                            drawBufferIndex, kind);
}

static MGLRendererBackendHandle *mglPdBackend(GLMContext ctx)
{
    return ctx ? (MGLRendererBackendHandle *)ctx->renderer_backend : NULL;
}

static void *mglPdTransientDepthTexture(GLMContext ctx, uint64_t *widthOut,
                                        uint64_t *heightOut)
{
    return mglRendererBackendGetTransientDepthTexture(mglPdBackend(ctx),
                                                      widthOut, heightOut);
}

static void *mglPdCreateTexture(const MGLRenderTextureDescriptorState *desc)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(desc, NULL, &texture) == 0 && texture) {
        return texture;
    }
    return NULL;
}

static uint64_t mglPdMaxU64(uint64_t a, uint64_t b) { return a > b ? a : b; }

/* -checkDrawBufferSize: */
int mglRenderPassCheckDrawBufferSize(void *renderer, unsigned int index)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    const MGLSizeValue drawableSize =
        mglPlatformShellApplyPendingDrawableSize(renderer);

    if ((uint32_t)drawableSize.width !=
        areas.core->drawBuffers[index].width) {
        return 0;
    }

    if ((uint32_t)drawableSize.height !=
        areas.core->drawBuffers[index].height) {
        return 0;
    }

    return 1;
}

/* -configureDefaultFramebufferAttachmentsLocked */
int mglRenderPassConfigureDefaultFramebufferAttachments(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *glState = mglPdStateOf(&areas);
    MGLCommandState *commandState = areas.command;

    void *texture = NULL;
    void *depth_texture = NULL;
    void *stencil_texture = NULL;

    uint32_t mappedDraw = 0u;
    if (!mglRenderDefaultDrawBufferIndex((uint32_t)glState->draw_buffer,
                                        &mappedDraw)) {
        DEBUG_PRINT("MGL: Unknown draw_buffer value: 0x%x, falling back to FRONT\n",
                    glState->draw_buffer);
        fprintf(stderr,
                "MGL WARNING: Unknown draw_buffer value 0x%x, using FRONT fallback\n",
                glState->draw_buffer);
    } else if (mglRenderDrawBufferIsNone((uint32_t)glState->draw_buffer)) {
        DEBUG_PRINT("MGL: draw_buffer is GL_NONE, falling back to FRONT\n");
    }
    const GLuint mgl_drawbuffer = (GLuint)mappedDraw;

    if (!mglRenderPassCheckDrawBufferSize(renderer, mgl_drawbuffer)) {
        (void)mglRendererBackendClearDefaultDrawBuffer(areas.backend,
                                                      mgl_drawbuffer);
        areas.core->drawBuffers[mgl_drawbuffer].width = 0;
        areas.core->drawBuffers[mgl_drawbuffer].height = 0;
    }

    /* attach color buffer */
    if (mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer)) {
        /* SAFETY: Ensure we have a valid drawable with texture */
        if (!areas.drawable) {
            fprintf(stderr,
                    "MGL ERROR: No drawable available for front buffer\n");
            return 0;
        }

        texture = mglRendererDrawableTexture(renderer);

        /* sleep mode will return a null texture - handle gracefully without
         * crashing */
        if (!texture) {
            fprintf(stderr,
                    "MGL WARNING: Drawable texture is NULL (sleep mode or window not visible), attempting to get new drawable\n");

            /* Try to get a new drawable */
            (void)mglRendererNextDrawable(renderer);
            if (mglPlatformShellDrawablePointer(renderer)) {
                texture = mglRendererDrawableTexture(renderer);
                fprintf(stderr,
                        "MGL INFO: Successfully obtained new drawable with texture\n");
            } else {
                fprintf(stderr,
                        "MGL ERROR: Still no drawable texture available\n");
                return 0;
            }
        }
    } else {
        texture = mglPdDefaultDrawBufferAttachment(
            areas.backend, mgl_drawbuffer,
            MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR);
        if (!texture) {
            texture = mglRendererNewDrawBuffer(
                renderer, ctx->pixel_format.mtl_pixel_format, 0);
            (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                areas.backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR, texture);
        }
    }

    /* attach depth. The default framebuffer must have a usable depth attachment
     * whenever GL depth testing is active, even if the legacy context format
     * fields were left unset by the window/bootstrap path. */
    void *cachedDepth = mglPdDefaultDrawBufferAttachment(
        areas.backend, mgl_drawbuffer,
        MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH);
    const int defaultPassNeedsDepth =
        glState->caps.depth_test || cachedDepth != NULL;
    if (defaultPassNeedsDepth) {
        const uint32_t depthFormat =
            mglRenderDepthFormatOrFallback(ctx->depth_format.mtl_pixel_format);

        if (cachedDepth) {
            depth_texture = cachedDepth;
        } else {
            const MGLRenderTextureInfo textureInfo = mglPdTextureInfo(texture);
            depth_texture = mglRendererNewDrawBufferWithCustomSize(
                depthFormat, 1, textureInfo.width, textureInfo.height);
            (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                areas.backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH, depth_texture);
            if (depth_texture) {
                static uint64_t s_defaultDepthCreateCount = 0;
                const uint64_t hit = ++s_defaultDepthCreateCount;
                if (kMGLDiagnosticStateLogs && hit <= 8) {
                    mglTraceLog(
                        "MGL DEFAULT FBO: created depth attachment fmt=%lu size=%lux%lu drawBuffer=%u",
                        (unsigned long)depthFormat,
                        (unsigned long)mglPdTextureInfo(depth_texture).width,
                        (unsigned long)mglPdTextureInfo(depth_texture).height,
                        mgl_drawbuffer);
                }
            }
        }
    }

    /* attach stencil */
    void *cachedStencil = mglPdDefaultDrawBufferAttachment(
        areas.backend, mgl_drawbuffer,
        MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL);
    const int defaultPassNeedsStencil =
        glState->caps.stencil_test || ctx->stencil_format.format ||
        cachedStencil != NULL;
    if (defaultPassNeedsStencil) {
        const uint32_t stencilFormat = mglRenderRepairedDefaultStencilFormat(
            ctx->stencil_format.mtl_pixel_format);

        if (cachedStencil) {
            stencil_texture = cachedStencil;
        } else {
            const MGLRenderTextureInfo textureInfo = mglPdTextureInfo(texture);
            stencil_texture = mglRendererNewDrawBufferWithCustomSize(
                stencilFormat, 1, textureInfo.width, textureInfo.height);
            (void)mglRendererBackendSetDefaultDrawBufferAttachment(
                areas.backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL,
                stencil_texture);
        }
    }

    mglPdSetPersistentAttachment(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
        mglApplySRGBStateToRenderTarget(texture, ctx), 0, 0, 0, 0);
    mglPdSetPersistentAttachment(commandState,
                                 MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                                 depth_texture, 0, 0, 0, 0);
    mglPdSetPersistentAttachment(commandState,
                                 MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                                 stencil_texture, 0, 0, 0, 0);

    mglPdSetPersistentDimensions(commandState, mglPdTextureInfo(texture).width,
                                 mglPdTextureInfo(texture).height);
    areas.core->drawBuffers[mgl_drawbuffer].width =
        (GLuint)mglPdTextureInfo(texture).width;
    areas.core->drawBuffers[mgl_drawbuffer].height =
        (GLuint)mglPdTextureInfo(texture).height;
    return 1;
}

/* -ensureTransientDepthForDefaultFramebufferLocked */
void mglRenderPassEnsureTransientDepthForDefaultFramebuffer(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *glState = mglPdStateOf(&areas);
    MGLCommandState *commandState = areas.command;

    if (!glState->framebuffer && glState->caps.depth_test &&
        !mglPdDepthTextureFor(commandState)) {
        uint64_t depthWidth = mglPdRenderTargetWidthFor(commandState);
        uint64_t depthHeight = mglPdRenderTargetHeightFor(commandState);

        if (depthWidth == 0 || depthHeight == 0) {
            void *color0 = mglPdColorTextureFor(commandState, 0);
            if (color0) {
                depthWidth = mglPdTextureInfo(color0).width;
                depthHeight = mglPdTextureInfo(color0).height;
            }
        }

        if (depthWidth > 0 && depthHeight > 0) {
            uint64_t cachedDepthWidth = 0;
            uint64_t cachedDepthHeight = 0;
            void *transientDepth =
                mglPdTransientDepthTexture(ctx, &cachedDepthWidth,
                                           &cachedDepthHeight);
            if (!transientDepth || cachedDepthWidth != depthWidth ||
                cachedDepthHeight != depthHeight) {
                MGLRenderTextureDescriptorState depthDesc = {0};
                depthDesc.texture_type = MGLTextureType2D;
                depthDesc.pixel_format = mglRenderDefaultDepthPixelFormat();
                depthDesc.width = depthWidth;
                depthDesc.height = depthHeight;
                depthDesc.depth = 1;
                depthDesc.mipmap_level_count = 1;
                depthDesc.sample_count = 1;
                depthDesc.array_length = 1;
                depthDesc.usage = MGL_PD_TEXTURE_USAGE_RENDER_TARGET;
                depthDesc.storage_mode = MGLStorageModePrivate;
                transientDepth = mglPdCreateTexture(&depthDesc);
                if (mglRendererBackendSetTransientDepthTexture(
                        mglPdBackend(ctx), transientDepth, depthWidth,
                        depthHeight) != 0) {
                    transientDepth = NULL;
                } else {
                    transientDepth =
                        mglPdTransientDepthTexture(ctx, NULL, NULL);
                }

                if (transientDepth) {
                    static uint64_t s_transientDepthCreateCount = 0;
                    const uint64_t hit = ++s_transientDepthCreateCount;
                    if (hit <= 16 || (hit % 128) == 0) {
                        fprintf(stderr,
                                "MGL TRANSIENT FBO: created depth attachment fmt=%lu size=%lux%lu fbo=%u\n",
                                (unsigned long)mglRenderDefaultDepthPixelFormat(),
                                (unsigned long)depthWidth,
                                (unsigned long)depthHeight,
                                (unsigned)(mglRendererSafeFramebufferName(ctx)));
                    }
                } else {
                    fprintf(stderr,
                            "MGL ERROR: failed to create transient depth attachment size=%lux%lu fbo=%u\n",
                            (unsigned long)depthWidth,
                            (unsigned long)depthHeight,
                            (unsigned)(mglRendererSafeFramebufferName(ctx)));
                }
            }

            if (transientDepth) {
                mglPdSetPersistentAttachment(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    transientDepth, 0, 0, 0, 0);
                mglPdSetPersistentActions(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    MGLLoadActionClear, MGLStoreActionDontCare);
                (void)mglRenderSetRenderPassStateDepthClear(
                    commandState->renderPassStateOwner,
                    glState->var.depth_clear_value);
            }
        }
    }
}

/* -configureUserFBOLoadStoreActionsLocked:fboColorClearMask:
 *  fboColorAttachment0ClearMask: */
void mglRenderPassConfigureUserFBOLoadStoreActions(
    void *renderer, unsigned int *outFboColorClearCount,
    unsigned int *outFboColorClearMask,
    unsigned int *outFboColorAttachment0ClearMask)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *glState = mglPdStateOf(&areas);
    MGLCommandState *commandState = areas.command;

    Framebuffer *fbo = glState->framebuffer;
    const GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);

    const int dontCareLoadEnabled = mglEnvFlagEnabled("MGL_ENABLE_DONTCARE_LOAD");
    for (int i = 0; i < drawBufferCount; ++i) {
        GLuint attachmentIndex = 0u;
        const GLuint colorSlot =
            mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (!mglMetalResolveFboDrawAttachmentIndex(
                ctx, mglMetalDrawBufferAt(ctx, (GLuint)i),
                &attachmentIndex) ||
            attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            mglPdSetPersistentLoadAction(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                colorSlot, MGLLoadActionLoad);
            continue;
        }

        FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
        if (attachmentIndex == 0) {
            *outFboColorAttachment0ClearMask = att->clear_bitmask;
        }

        Texture *attachmentTextureForClear =
            mglRendererAttachmentTextureFor(ctx, att);
        /* stamp this attachment's frame generation on EVERY render-target use
         * (clear/load/dontcare), capturing whether this is its first use this
         * frame BEFORE stamping. A clear-then-resume within one frame must
         * record the clear as a use so the resume is not mistaken for a first
         * use (which would wrongly DontCare and discard the cleared+drawn
         * content). */
        int colorFirstUseThisFrame = 0;
        if (dontCareLoadEnabled && attachmentTextureForClear) {
            colorFirstUseThisFrame =
                (attachmentTextureForClear->mtl_rt_frame_generation !=
                 commandState->dontCareFrameGeneration);
            attachmentTextureForClear->mtl_rt_frame_generation =
                commandState->dontCareFrameGeneration;
        }
        MGLRenderPassLoadStoreInput loadStore = {0};
        loadStore.attachment_kind = MGL_RP_ATTACHMENT_COLOR;
        loadStore.attachment_present = 1;
        loadStore.has_clear_pending =
            mglRenderClearMaskHasColor((uint32_t)att->clear_bitmask) ? 1 : 0;
        loadStore.texture_present =
            (attachmentTextureForClear && attachmentTextureForClear->mtl_data)
                ? 1
                : 0;
        loadStore.dontcare_enabled = dontCareLoadEnabled ? 1 : 0;
        loadStore.first_use_this_frame = colorFirstUseThisFrame ? 1 : 0;
        loadStore.blend_enabled =
            (ctx && glState->caps.blend) ? 1 : 0;
        MGLRenderPassLoadStorePlan loadStorePlan = {0};
        if (mglRenderPassPlanLoadStore(&loadStore, &loadStorePlan) != 0) {
            mglPdSetPersistentLoadAction(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                colorSlot, MGLLoadActionLoad);
            continue;
        }
        if (loadStorePlan.load_action == MGLLoadActionClear) {
            if (attachmentTextureForClear &&
                attachmentTextureForClear->name == 8u &&
                mglTraceLogIsEnabled()) {
                mglTraceLog(
                    "PENDING_COLOR_CLEAR_CONSUME tex=%u fbo=%u attachment=%u slot=%d program=%u clearMask=0x%x rgba=(%.3f,%.3f,%.3f,%.3f) drawBuf=0x%x readBuf=0x%x scissor(test=%d box=%d,%d,%d,%d) colorMask=%d%d%d%d depth(test=%d write=%d)",
                    (unsigned)attachmentTextureForClear->name,
                    (unsigned)fbo->name, (unsigned)attachmentIndex, i,
                    (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                    (unsigned)att->clear_bitmask, att->clear_color[0],
                    att->clear_color[1], att->clear_color[2],
                    att->clear_color[3],
                    (unsigned)(ctx ? glState->draw_buffer : 0u),
                    (unsigned)(ctx ? glState->read_buffer : 0u),
                    (ctx && glState->caps.scissor_test) ? 1 : 0,
                    (int)(ctx ? glState->var.scissor_box[0] : 0),
                    (int)(ctx ? glState->var.scissor_box[1] : 0),
                    (int)(ctx ? glState->var.scissor_box[2] : 0),
                    (int)(ctx ? glState->var.scissor_box[3] : 0),
                    (ctx && glState->var.color_writemask[0][0]) ? 1 : 0,
                    (ctx && glState->var.color_writemask[0][1]) ? 1 : 0,
                    (ctx && glState->var.color_writemask[0][2]) ? 1 : 0,
                    (ctx && glState->var.color_writemask[0][3]) ? 1 : 0,
                    (ctx && glState->caps.depth_test) ? 1 : 0,
                    (ctx && glState->var.depth_writemask) ? 1 : 0);
            }
            (void)mglRenderSetRenderPassStateColorClear(
                commandState->renderPassStateOwner, colorSlot,
                att->clear_color[0], att->clear_color[1], att->clear_color[2],
                att->clear_color[3]);
            mglPdSetPersistentActions(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                colorSlot, loadStorePlan.load_action,
                loadStorePlan.set_store_action ? loadStorePlan.store_action
                                               : MGLStoreActionStore);

            /* MS textures are texture2d_array sample planes; LoadActionClear
             * only hits the attached base slice. Clear the other sample planes
             * so imageLoad(sample) sees the same clear value. */
            if (attachmentTextureForClear &&
                attachmentTextureForClear->mtl_data &&
                mglRenderIsMultisampleTextureTarget(
                    (uint32_t)attachmentTextureForClear->target)) {
                const MGLMetalAttachmentSubresource sub =
                    mglMetalAttachmentSubresourceForAttachment(att);
                const uint64_t samples =
                    mglPdMaxU64((uint64_t)attachmentTextureForClear->samples, 1u);
                for (uint64_t s = 1u; s < samples; s++) {
                    (void)mglRenderEncodeColorClearForCommandBufferOwner(
                        commandState->currentCommandBufferOwner,
                        attachmentTextureForClear->mtl_data, sub.level,
                        sub.slice + s, sub.depthPlane, att->clear_color[0],
                        att->clear_color[1], att->clear_color[2],
                        att->clear_color[3]);
                }
            }

            att->clear_bitmask = (GLbitfield)mglRenderClearMaskClearColor(
                (uint32_t)att->clear_bitmask);
            mglMarkTextureLevelRenderTargetWrittenImpl(
                attachmentTextureForClear, att->level, __func__, __LINE__);

            (*outFboColorClearCount)++;
            *outFboColorClearMask |= (GLbitfield)(1u << attachmentIndex);
        } else {
            /* DontCare when the plan allows discarding, Load otherwise: the
             * predicate (flag, texture, first use this frame, blending) lives in
             * the plan. */
            mglPdSetPersistentLoadAction(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                colorSlot, loadStorePlan.load_action);
        }
    }

    for (GLuint ai = 0; ai < MAX_COLOR_ATTACHMENTS; ++ai) {
        /* A pending clear for an attachment this pass does not have cannot be
         * consumed; the rule lives in the plan. */
        if (mglRenderPassDropsStaleColorClear(
                (uint32_t)fbo->color_attachments[ai].clear_bitmask,
                (uint32_t)fbo->color_attachment_bitfield, (uint32_t)ai)) {
            fbo->color_attachments[ai].clear_bitmask =
                (GLbitfield)mglRenderClearMaskClearColor(
                    (uint32_t)fbo->color_attachments[ai].clear_bitmask);
        }
    }

    MGLRenderPassLoadStoreInput depthLoadStore = {0};
    depthLoadStore.attachment_kind = MGL_RP_ATTACHMENT_DEPTH;
    depthLoadStore.has_clear_pending =
        mglRenderClearMaskHasDepth((uint32_t)fbo->depth.clear_bitmask) ? 1 : 0;
    depthLoadStore.texture_present =
        mglPdDepthTextureFor(commandState) ? 1 : 0;
    MGLRenderPassLoadStorePlan depthPlan = {0};
    if (mglRenderPassPlanLoadStore(&depthLoadStore, &depthPlan) == 0 &&
        depthPlan.load_action == MGLLoadActionClear) {
        (void)mglRenderSetRenderPassStateDepthClear(
            commandState->renderPassStateOwner, fbo->depth.clear_color[0]);
        mglPdSetPersistentActions(commandState,
                                  MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                                  depthPlan.load_action, depthPlan.store_action);
        fbo->depth.clear_bitmask = (GLbitfield)mglRenderClearMaskClearDepth(
            (uint32_t)fbo->depth.clear_bitmask);
    } else {
        mglPdSetPersistentLoadAction(commandState,
                                     MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                                     depthPlan.load_action);
        if (depthPlan.set_store_action) {
            mglPdSetPersistentStoreAction(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                depthPlan.store_action);
        }
    }

    MGLRenderPassLoadStoreInput stencilLoadStore = {0};
    stencilLoadStore.attachment_kind = MGL_RP_ATTACHMENT_STENCIL;
    stencilLoadStore.has_clear_pending =
        mglRenderClearMaskHasStencil((uint32_t)fbo->stencil.clear_bitmask) ? 1
                                                                          : 0;
    stencilLoadStore.texture_present =
        mglPdStencilTextureFor(commandState) ? 1 : 0;
    MGLRenderPassLoadStorePlan stencilPlan = {0};
    if (mglRenderPassPlanLoadStore(&stencilLoadStore, &stencilPlan) == 0 &&
        stencilPlan.load_action == MGLLoadActionClear) {
        (void)mglRenderSetRenderPassStateStencilClear(
            commandState->renderPassStateOwner,
            (uint32_t)fbo->stencil.clear_color[0]);
        mglPdSetPersistentActions(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            stencilPlan.load_action, stencilPlan.store_action);
        fbo->stencil.clear_bitmask = (GLbitfield)mglRenderClearMaskClearStencil(
            (uint32_t)fbo->stencil.clear_bitmask);
    } else {
        mglPdSetPersistentLoadAction(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            stencilPlan.load_action);
        if (stencilPlan.set_store_action) {
            mglPdSetPersistentStoreAction(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                stencilPlan.store_action);
        }
    }
}

/* -configureDefaultFramebufferLoadStoreActionsLocked */
void mglRenderPassConfigureDefaultFramebufferLoadStoreActions(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMState *glState = mglPdStateOf(&areas);
    MGLCommandState *commandState = areas.command;

    Framebuffer *fbo = glState->framebuffer;
    const GLbitfield defaultClearMask = glState->default_fbo_clear_bitmask;
    if (mglRenderClearMaskHasColor((uint32_t)defaultClearMask)) {
        (void)mglRenderSetRenderPassStateColorClear(
            commandState->renderPassStateOwner, 0,
            glState->default_clear_color[0], glState->default_clear_color[1],
            glState->default_clear_color[2], glState->default_clear_color[3]);
        mglPdSetPersistentActions(commandState,
                                  MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                                  MGLLoadActionClear, MGLStoreActionStore);
        glState->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearColor(
                (uint32_t)glState->default_fbo_clear_bitmask);
    } else {
        mglPdSetPersistentLoadAction(commandState,
                                     MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                                     MGLLoadActionLoad);
        static uint64_t s_defaultFboLoadLogCount = 0;
        const uint64_t hit = ++s_defaultFboLoadLogCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            mglTraceLog(
                "MGL DEFAULT FBO: using Load (no clear mask) call=%llu drawBuf=0x%x fbo=%u",
                (unsigned long long)hit, glState->draw_buffer,
                fbo ? (unsigned)fbo->name : 0u);
        }
    }

    if (mglRenderClearMaskHasDepth((uint32_t)defaultClearMask)) {
        (void)mglRenderSetRenderPassStateDepthClear(
            commandState->renderPassStateOwner, glState->var.depth_clear_value);
        mglPdSetPersistentActions(commandState,
                                  MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                                  MGLLoadActionClear, MGLStoreActionStore);
        glState->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearDepth(
                (uint32_t)glState->default_fbo_clear_bitmask);
    } else {
        mglPdSetPersistentLoadAction(commandState,
                                     MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                                     MGLLoadActionLoad);
        if (mglPdDepthTextureFor(commandState)) {
            mglPdSetPersistentStoreAction(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MGLStoreActionStore);
        }
    }

    if (mglRenderClearMaskHasStencil((uint32_t)defaultClearMask)) {
        (void)mglRenderSetRenderPassStateStencilClear(
            commandState->renderPassStateOwner,
            glState->var.stencil_clear_value);
        mglPdSetPersistentActions(commandState,
                                  MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                                  MGLLoadActionClear, MGLStoreActionStore);
        glState->default_fbo_clear_bitmask =
            (GLbitfield)mglRenderClearMaskClearStencil(
                (uint32_t)glState->default_fbo_clear_bitmask);
    } else {
        mglPdSetPersistentLoadAction(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
            MGLLoadActionLoad);
        if (mglPdStencilTextureFor(commandState)) {
            mglPdSetPersistentStoreAction(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MGLStoreActionStore);
        }
    }
}

/* -logRenderPassClearResolveLocked:... */
void mglRenderPassLogClearResolve(
    void *renderer, uint64_t renderEncoderCall, int traceRenderEncoder,
    unsigned int fboColorClearCount, unsigned int fboColorClearMask,
    unsigned int fboColorAttachment0ClearMask,
    unsigned int fboDepthClearMaskBefore,
    unsigned int fboStencilClearMaskBefore, unsigned int defaultClearMask,
    struct Framebuffer_t *fboRaw)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *glState = mglPdStateOf(&areas);
    MGLCommandState *commandState = areas.command;
    Framebuffer *fbo = (Framebuffer *)fboRaw;

    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
        double c0[4] = {0.0, 0.0, 0.0, 0.0};
        const double fallback[4] = {0.0, 0.0, 0.0, 0.0};
        mglPdClearColorFor(commandState, 0, fallback, c0);
        mglTraceLog(
            "MGL TRACE clear.resolve call=%llu fbo=%u "
            "fboColorClears=%u fboColorMask=0x%x fboAtt0ClearMask=0x%x c0LA=%s depthLA=%s stencilLA=%s "
            "c0Clear=(%.3f,%.3f,%.3f,%.3f) depthClear=%.3f stencilClear=%u",
            (unsigned long long)renderEncoderCall,
            (unsigned)(mglRendererSafeFramebufferName(ctx)),
            (unsigned)fboColorClearCount, (unsigned)fboColorClearMask,
            (unsigned)fboColorAttachment0ClearMask,
            mglLoadActionName(mglPdLoadActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MGLLoadActionDontCare)),
            mglLoadActionName(mglPdLoadActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MGLLoadActionDontCare)),
            mglLoadActionName(mglPdLoadActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MGLLoadActionDontCare)),
            c0[0], c0[1], c0[2], c0[3], mglPdClearDepthFor(commandState, 0.0),
            (unsigned)mglPdClearStencilFor(commandState, 0));
    }

    const int clearResolveInteresting =
        (fboColorClearCount != 0) || (fboColorAttachment0ClearMask != 0) ||
        (mglPdLoadActionFor(commandState,
                            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                            MGLLoadActionDontCare) == MGLLoadActionClear) ||
        mglRenderClearMaskHasDepth((uint32_t)fboDepthClearMaskBefore) ||
        (!fbo && mglRenderClearMaskHasDepth((uint32_t)defaultClearMask));
    if (clearResolveInteresting) {
        static uint64_t s_clearResolveDetailLogCount = 0;
        const uint64_t hit = ++s_clearResolveDetailLogCount;
        if (mglTraceLogIsEnabled() && (hit <= 256ull || (hit % 512ull) == 0ull)) {
            double c0[4] = {0.0, 0.0, 0.0, 0.0};
            const double fallback[4] = {0.0, 0.0, 0.0, 0.0};
            mglPdClearColorFor(commandState, 0, fallback, c0);
            void *c0Tex = mglPdColorTextureFor(commandState, 0);
            void *dTex = mglPdDepthTextureFor(commandState);
            void *sTex = mglPdStencilTextureFor(commandState);
            mglTraceLog(
                "RENDERPASS_CLEAR call=%llu hit=%llu fbo=%u drawBuf=0x%x readBuf=0x%x "
                "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
                "fboColorClears=%u fboColorMask=0x%x fboAtt0Mask=0x%x pending(global=0x%x default=0x%x depth=0x%x stencil=0x%x) "
                "c0LA=%s c0SA=%s depthLA=%s depthSA=%s stencilLA=%s stencilSA=%s "
                "c0Tex=%p fmt=%lu size=%lux%lu depthTex=%p fmt=%lu size=%lux%lu stencilTex=%p "
                "clearRGBA=(%.6f,%.6f,%.6f,%.6f) depthClear=%.6f stencilClear=%u depthState(test=%d write=%d func=0x%x)",
                (unsigned long long)renderEncoderCall, (unsigned long long)hit,
                (unsigned)(mglRendererSafeFramebufferName(ctx)),
                (unsigned)glState->draw_buffer, (unsigned)glState->read_buffer,
                (int)glState->viewport[0], (int)glState->viewport[1],
                (int)glState->viewport[2], (int)glState->viewport[3],
                glState->caps.scissor_test ? 1 : 0,
                (int)glState->var.scissor_box[0],
                (int)glState->var.scissor_box[1],
                (int)glState->var.scissor_box[2],
                (int)glState->var.scissor_box[3],
                (unsigned)fboColorClearCount, (unsigned)fboColorClearMask,
                (unsigned)fboColorAttachment0ClearMask,
                (unsigned)glState->clear_bitmask,
                (unsigned)glState->default_fbo_clear_bitmask,
                (unsigned)fboDepthClearMaskBefore,
                (unsigned)fboStencilClearMaskBefore,
                mglLoadActionName(mglPdLoadActionFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    MGLLoadActionDontCare)),
                mglStoreActionName(mglPdStoreActionFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    MGLStoreActionDontCare)),
                mglLoadActionName(mglPdLoadActionFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    MGLLoadActionDontCare)),
                mglStoreActionName(mglPdStoreActionFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    MGLStoreActionDontCare)),
                mglLoadActionName(mglPdLoadActionFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                    MGLLoadActionDontCare)),
                mglStoreActionName(mglPdStoreActionFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                    MGLStoreActionDontCare)),
                c0Tex,
                (unsigned long)(c0Tex ? mglPdTextureInfo(c0Tex).pixel_format
                                      : mglRenderInvalidPixelFormat()),
                (unsigned long)(c0Tex ? mglPdTextureInfo(c0Tex).width : 0),
                (unsigned long)(c0Tex ? mglPdTextureInfo(c0Tex).height : 0),
                dTex,
                (unsigned long)(dTex ? mglPdTextureInfo(dTex).pixel_format
                                     : mglRenderInvalidPixelFormat()),
                (unsigned long)(dTex ? mglPdTextureInfo(dTex).width : 0),
                (unsigned long)(dTex ? mglPdTextureInfo(dTex).height : 0), sTex,
                c0[0], c0[1], c0[2], c0[3], mglPdClearDepthFor(commandState, 0.0),
                (unsigned)mglPdClearStencilFor(commandState, 0),
                glState->caps.depth_test ? 1 : 0,
                glState->var.depth_writemask ? 1 : 0,
                (unsigned)glState->var.depth_func);
        }
    }
}

/* -newRenderEncoderLockedWithReason: */
static int mglPdNewRenderEncoderBody(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *glState = mglPdStateOf(&areas);
    MGLCommandState *commandState = areas.command;

    mglBindingInvalidateLastBoundState(renderer);

    static uint64_t s_newRenderEncoderCallCount = 0;
    const uint64_t renderEncoderCall = ++s_newRenderEncoderCallCount;
    const bool traceRenderEncoder =
        mglPdShouldTraceCall(renderEncoderCall) ||
        (kMGLDiagnosticStateLogs && ((renderEncoderCall % 60ull) == 0ull));

    /* AGX ERROR THROTTLING: Check if we should skip render encoder creation BUT
     * allow limited render encoder creation for essential functionality */
    if (mglRendererShouldSkipGPUOperations(renderer)) {
        fprintf(stderr,
                "MGL AGX: Render encoder creation requested during GPU recovery - attempting essential creation\n");
    }

    /* CRITICAL SAFETY: Check command buffer before creating render encoder */
    if (mglRenderCommandBufferOwnerHasCurrent(
            commandState->currentCommandBufferOwner) != 1) {
        /* Attempt recovery: create a new command buffer instead of failing
         * immediately */
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: Cannot create render encoder - no command buffer available\n");
            mglRendererRecordGPUError(renderer);
            return 0;
        }
    }

    /* end encoding on current render encoder */
    mglRendererEndRenderEncodingLocked(renderer);

    /* grab the next drawable from CAMetalLayer */
    if (areas.drawable == NULL) {
        if (!areas.layer) {
            fprintf(stderr,
                    "MGL ERROR: Cannot get drawable - no CAMetalLayer available\n");
            return 0;
        }

        const MGLSizeValue expectedDrawableSize =
            mglPlatformShellApplyPendingDrawableSize(renderer);
        (void)mglRendererNextDrawable(renderer);

        /* late init of gl scissor box on attachment to window system */
        uint64_t drawableWidth = mglPdMaxU64(1u, expectedDrawableSize.width);
        uint64_t drawableHeight = mglPdMaxU64(1u, expectedDrawableSize.height);
        if (mglPlatformShellDrawablePointer(renderer) &&
            mglRendererDrawableTexture(renderer)) {
            drawableWidth =
                mglPdTextureInfo(mglRendererDrawableTexture(renderer)).width;
            drawableHeight =
                mglPdTextureInfo(mglRendererDrawableTexture(renderer)).height;
        }

        if (!glState->caps.scissor_test) {
            glState->var.scissor_box[0] = 0;
            glState->var.scissor_box[1] = 0;
        }
        glState->var.scissor_box[2] = (GLint)drawableWidth;
        glState->var.scissor_box[3] = (GLint)drawableHeight;
    }

    mglPassManagerInstallNewRenderPassDescriptor(areas.render_pass_manager);
    if (!commandState->renderPassStateOwner) {
        fprintf(stderr,
                "MGL RENDERPASS ERROR: failed to allocate render pass state owner\n");
        return 0;
    }

    /* Configure color/depth/stencil attachments based on FBO type */
    if (glState->framebuffer) {
        if (!mglRenderPassConfigureUserFBOAttachments(renderer)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    } else {
        if (!mglRenderPassConfigureDefaultFramebufferAttachments(renderer)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    }
    mglRenderPassEnsureTransientDepthForDefaultFramebuffer(renderer);

    /* Capture clear state before load/store resolution for diagnostic logging */
    unsigned int fboColorClearCount = 0;
    unsigned int fboColorClearMask = 0;
    unsigned int fboColorAttachment0ClearMask = 0;

    Framebuffer *fbo = glState->framebuffer;
    const unsigned int defaultClearMask = glState->default_fbo_clear_bitmask;
    const unsigned int fboDepthClearMaskBefore =
        fbo ? fbo->depth.clear_bitmask : 0u;
    const unsigned int fboStencilClearMaskBefore =
        fbo ? fbo->stencil.clear_bitmask : 0u;

    if (fbo) {
        mglRenderPassConfigureUserFBOLoadStoreActions(
            renderer, &fboColorClearCount, &fboColorClearMask,
            &fboColorAttachment0ClearMask);
    } else {
        mglRenderPassConfigureDefaultFramebufferLoadStoreActions(renderer);
    }

    mglRenderPassLogClearResolve(renderer, renderEncoderCall,
                                 traceRenderEncoder ? 1 : 0,
                                 fboColorClearCount, fboColorClearMask,
                                 fboColorAttachment0ClearMask,
                                 fboDepthClearMaskBefore,
                                 fboStencilClearMaskBefore, defaultClearMask,
                                 fbo);

    if (!mglRenderPassFinalizeRenderPassDescriptor(
            renderer, renderEncoderCall, traceRenderEncoder ? 1 : 0)) {
        fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
        return 0;
    }
    if (!mglRenderPassCreateRenderEncoderLocked(renderer, renderEncoderCall)) {
        fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
        return 0;
    }

    /* Apply dynamic state that is not part of the render-pass owner state. */
    mglRenderPassUpdateCurrentRenderEncoder(renderer);

    /* Only bind buffers when creating the encoder. Sampled textures depend on
     * the current GL program/MSL reflection and are rebound after the pipeline
     * state is selected for the draw. */
    if (glState->vao) {
        MGLEncodeContext encCtx = {
            .render_encoder_owner =
                commandState->currentRenderEncoderOwner,
        };
        if (mglStageEncodeBindVertexBuffers(renderer, &encCtx) == false) {
            DEBUG_PRINT("vertex buffer binding failed\n");
            mglRendererRecordGPUError(renderer);
            return 0;
        }

        if (mglStageEncodeBindFragmentBuffers(renderer, &encCtx) == false) {
            DEBUG_PRINT("fragment buffer binding failed\n");
            mglRendererRecordGPUError(renderer);
            return 0;
        }
    }

    /* Record successful render encoder creation (final success) */
    mglRendererRecordGPUSuccess(renderer);
    return 1;
}

int mglRenderPassNewRenderEncoderLockedWithReason(void *renderer,
                                                  uint32_t reason)
{
    /* instrumentation: count every render encoder (re)creation + reason. */
    MGL_PERF_INC(g_mglEncoderCreationsSinceSwap);
    if ((unsigned)reason >= (unsigned)MGL_ENC_REASON_COUNT) {
        reason = MGL_ENC_REASON_OTHER;
    }
    MGL_PERF_INC(g_mglEncoderCreateReasonSinceSwap[reason]);
    /* The .m wrapped the body in @autoreleasepool; the shell forwarder runs the
     * C body inside one so autoreleased temporaries still drain here. */
    return mglPlatformShellAutoreleasePoolCall(renderer, mglPdNewRenderEncoderBody);
}
