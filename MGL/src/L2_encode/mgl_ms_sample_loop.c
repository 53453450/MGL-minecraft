/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* Bodies of -[MGLRenderer runEmulatedMSSampleDrawLoopIfNeeded:drawOnce:] and
 * -broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded: (P0-1).  The block became a
 * function pointer plus a context, and the three _mgl* ivars travel through the
 * shell forwarders. */

#include "mgl_ms_sample_loop.h"
#include "mgl_ports_present.h"
#include "mgl_renderer_state_areas.h"
#include "mgl_render_pass_manager_ops.h"  /* mglRendererEndRenderEncodingLocked */
#include "mgl_draw_support.h"          /* mglDrawEmulatedMSColor0Texture, … */
#include "mgl_render_api_command.h"
#include "mgl_render_api_texture.h"
#include "mgl_metal_ref.h"
#include "mgl_sync.h"                  /* mglMetalAttachmentSubresourceForAttachment */
#include "mgl_texture_compat.h"        /* mglMarkTextureLevelRenderTargetWrittenImpl */

#include "mgl_render_api_readback.h"   /* pending clear masks */
#include "mgl_render_pass_sync_ops.h"   /* mglRenderPassProcessGLStateLocked */
#include "mgl_render_api_binding.h"    /* fragment texture/bytes binding */
#include "mgl_render_api_query.h"      /* GL compare enums */
#include "mgl_renderer_binding_ops.h"  /* fragment buffer cache invalidation */
#include "mgl_render_values.h"
#include "mgl_buffer_slots.h"          /* post-depth slots */
#include "mgl_program_resource.h"      /* fragment builtin mask */
#include "mgl_shader_abi.h"
#include "mgl_sampled_fallback.h"
#include "mgl_raster_orientation_state.h"

#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include "mgl_texture_backend.h"

extern int mglPlatformShellMSSampleInLoop(void *renderer);
extern void mglPlatformShellSetMSSampleState(void *renderer, int in_loop,
                                             int32_t forced, int32_t offset);
extern int mglPlatformShellNewCommandBuffer(void *renderer);

/* Copies of the depth/stencil sample planes taken before a post-depth
 * coverage draw loop.  The loop runs synchronously on the GL thread, and the
 * render pass of each plane binds the copies while it is set up. */
typedef struct MGLPostDepthSnapshot {
    void *depth;    /* +1, depth2d_array view, or NULL */
    void *stencil;  /* +1, uint texture2d_array view, or NULL */
    uint32_t samples;
} MGLPostDepthSnapshot;

static _Thread_local MGLPostDepthSnapshot s_postDepthSnapshot;

static Texture *mglDrawEmulatedMSTargetTexture(GLMContext ctx)
{
    Texture *tex = mglDrawEmulatedMSColor0Texture(ctx);
    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!tex && fbo) {
        tex = mglDrawEmulatedMSAttachmentTexture(ctx, &fbo->depth);
    }
    if (!tex && fbo) {
        tex = mglDrawEmulatedMSAttachmentTexture(ctx, &fbo->stencil);
    }
    return tex;
}

static int mglPostDepthCoverageProgram(GLMContext ctx)
{
    Program *fp = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    return fp && mglProgramStageUsesBuiltin(
                     fp, _FRAGMENT_SHADER, MGL_AIR_BUILTIN_POST_DEPTH_COVERAGE);
}

static void *mglMSPlaneMetalTexture(void *renderer, Texture *tex)
{
    if (!tex) {
        return NULL;
    }
    tex->is_render_target = 1;
    if (!mglRendererBindMTLTexture(renderer, tex)) {
        return NULL;
    }
    return mglTextureMtl(tex);
}

/* +1 copy of `tex`'s sample planes (array slice s = sample s), viewed with
 * `view_format`, or NULL. */
static void *mglCopyMSPlanes(void *renderer, void *blit, Texture *tex,
                             FBOAttachment *att, int stencil_view)
{
    void *metal = mglMSPlaneMetalTexture(renderer, tex);
    MGLRenderTextureInfo info = {0};
    if (!metal || mglRenderGetTextureInfo(metal, &info) != 0) {
        return NULL;
    }
    const MGLMetalAttachmentSubresource sub =
        mglMetalAttachmentSubresourceForAttachment(att);
    const uint64_t samples = (uint64_t)tex->samples;
    if (sub.slice + samples > info.array_length) {
        return NULL;
    }
    MGLRenderTextureDescriptorState desc = {0};
    desc.texture_type = MGLTextureType2DArray;
    desc.pixel_format = info.pixel_format;
    desc.width = mglMetalTextureLevelDimension((size_t)info.width, sub.level);
    desc.height = mglMetalTextureLevelDimension((size_t)info.height, sub.level);
    desc.depth = 1u;
    desc.mipmap_level_count = 1u;
    desc.sample_count = 1u;
    desc.array_length = samples;
    desc.usage = MGLTextureUsageShaderRead | MGLTextureUsagePixelFormatView;
    desc.storage_mode = MGLStorageModePrivate;
    void *copy = NULL;
    if (mglRenderCreateTextureFromState(&desc, NULL, &copy) != 0 || !copy) {
        return NULL;
    }
    for (uint64_t s = 0; s < samples; s++) {
        (void)mglRenderBlitCopyTexture(blit, metal, sub.slice + s, sub.level,
                                       0u, 0u, 0u, desc.width, desc.height, 1u,
                                       copy, s, 0u, 0u, 0u, 0u);
    }
    const uint32_t viewFormat = stencil_view
        ? mglRenderSampledViewPixelFormat(GL_STENCIL_INDEX, info.pixel_format)
        : info.pixel_format;
    if (viewFormat == info.pixel_format) {
        return copy;
    }
    void *view = NULL;
    (void)mglRenderCreateTextureView(copy, viewFormat, &view);
    CFRelease((CFTypeRef)copy);
    return view;
}

static void mglPostDepthSnapshotRelease(void)
{
    if (s_postDepthSnapshot.depth) {
        CFRelease((CFTypeRef)s_postDepthSnapshot.depth);
    }
    if (s_postDepthSnapshot.stencil) {
        CFRelease((CFTypeRef)s_postDepthSnapshot.stencil);
    }
    s_postDepthSnapshot = (MGLPostDepthSnapshot){0};
}

/* For a post_depth_coverage fragment program, snapshots the depth/stencil
 * sample planes the per-sample draws test against. */
static void mglSnapshotMSDepthStencilPlanes(void *renderer, GLMContext ctx,
                                            GLint samples)
{
    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!fbo || !mglPostDepthCoverageProgram(ctx)) {
        return;
    }
    Texture *depth = mglDrawEmulatedMSAttachmentTexture(ctx, &fbo->depth);
    Texture *stencil = mglDrawEmulatedMSAttachmentTexture(ctx, &fbo->stencil);
    if (!depth && !stencil) {
        return;
    }
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    /* A pass over the planes consumes pending depth/stencil clears first, so
     * the copies see cleared contents. */
    if ((depth && mglRenderClearMaskHasDepth((uint32_t)fbo->depth.clear_bitmask)) ||
        (stencil &&
         mglRenderClearMaskHasStencil((uint32_t)fbo->stencil.clear_bitmask))) {
        (void)mglRenderPassProcessGLStateLocked(renderer, 0);
    }
    mglRendererEndRenderEncodingLocked(renderer);
    if (!cs->currentCommandBufferOwner &&
        !mglPlatformShellNewCommandBuffer(renderer)) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(cs->currentCommandBufferOwner);
    if (!blit) {
        return;
    }
    mglPostDepthSnapshotRelease();
    if (depth) {
        s_postDepthSnapshot.depth =
            mglCopyMSPlanes(renderer, blit, depth, &fbo->depth, 0);
    }
    if (stencil) {
        s_postDepthSnapshot.stencil =
            mglCopyMSPlanes(renderer, blit, stencil, &fbo->stencil, 1);
    }
    (void)mglRenderEndBlitEncoder(blit);
    s_postDepthSnapshot.samples = (uint32_t)samples;
}

int mglRendererBindPostDepthCoverageInputs(void *renderer, GLMContext ctx,
                                           Program *fragment_program)
{
    if (!fragment_program ||
        !mglProgramStageUsesBuiltin(fragment_program, _FRAGMENT_SHADER,
                                    MGL_AIR_BUILTIN_POST_DEPTH_COVERAGE)) {
        return 1;
    }
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLRenderEncoderOwner *encoder =
        areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    if (!encoder) {
        return 0;
    }
    const GLMState *state = ctx->active_state;
    const int inLoop = mglPlatformShellMSSampleInLoop(renderer);
    void *depth = inLoop ? s_postDepthSnapshot.depth : NULL;
    void *stencil = inLoop ? s_postDepthSnapshot.stencil : NULL;
    const uint32_t depthOn = depth && state->caps.depth_test ? 1u : 0u;
    const uint32_t stencilOn = stencil && state->caps.stencil_test ? 1u : 0u;
    /* Word layout read by emitPostDepthCoverageMask; compare functions are
     * GL enums relative to GL_NEVER. */
    const uint32_t params[11] = {
        (depthOn || stencilOn) ? s_postDepthSnapshot.samples : 0u,
        depthOn,
        stencilOn,
        (uint32_t)state->var.depth_func - GL_NEVER,
        (uint32_t)state->var.stencil_func - GL_NEVER,
        (uint32_t)state->var.stencil_ref,
        (uint32_t)state->var.stencil_value_mask,
        (uint32_t)state->var.stencil_back_func - GL_NEVER,
        (uint32_t)state->var.stencil_back_ref,
        (uint32_t)state->var.stencil_back_value_mask,
        mglRasterTargetStoresGLOrder(state) ? 1u : 0u,
    };
    if (!depth) {
        depth = mglSampledFallbackTextureForExpectedType(
            renderer, MGLTextureType2DArray, MGLTextureDataKindDepth);
    }
    if (!stencil) {
        stencil = mglSampledFallbackTextureForExpectedType(
            renderer, MGLTextureType2DArray, MGLTextureDataKindUint);
    }
    if (!depth || !stencil ||
        mglRenderSetRenderTextureForOwner(encoder, depth,
                                          MGL_RENDER_BINDING_STAGE_FRAGMENT,
                                          kMGLPostDepthTextureIndex) != 0 ||
        mglRenderSetRenderTextureForOwner(encoder, stencil,
                                          MGL_RENDER_BINDING_STAGE_FRAGMENT,
                                          kMGLPostStencilTextureIndex) != 0 ||
        mglRenderSetRenderBytesForOwner(encoder, params, sizeof(params),
                                        MGL_RENDER_BINDING_STAGE_FRAGMENT,
                                        kMGLPostDepthParamsBufferIndex) != 0) {
        return 0;
    }
    mglRendererBindingInvalidateLastBoundFragmentBufferAtIndex(
        renderer, kMGLPostDepthParamsBufferIndex);
    return 1;
}

int mglRendererRunEmulatedMSSampleDrawLoopIfNeeded(
    void *renderer, GLMContext ctx, void (*draw_once)(void *), void *draw_ctx)
{
    if (mglPlatformShellMSSampleInLoop(renderer) || !draw_once) {
        return 0;
    }
    Texture *tex = mglDrawEmulatedMSTargetTexture(ctx);
    if (!tex) {
        return 0;
    }
    const GLint samples = tex->samples > 1 ? (GLint)tex->samples : 1;
    /* GL 4.6 §14.9.3: SAMPLE_MASK clears coverage of masked-out samples. */
    const int masked = mglDrawSampleMaskExcludesSamples(ctx, samples);
    if (!masked && !mglDrawFragmentNeedsPerSampleMSValues(ctx)) {
        return 0;
    }

    mglSnapshotMSDepthStencilPlanes(renderer, ctx, samples);
    mglPlatformShellSetMSSampleState(renderer, 1, 0, 0);
    for (GLint s = 0; s < samples; s++) {
        if (masked &&
            !(ctx->active_state->var.sample_mask_value & (1u << s))) {
            continue;
        }
        mglPlatformShellSetMSSampleState(renderer, 1, s, s);
        mglRendererEndRenderEncodingLocked(renderer);
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_FBO);
        Framebuffer *fbo = ctx->active_state->framebuffer;
        if (fbo) {
            fbo->dirty_bits |= DIRTY_FBO_BINDING;
        }
        draw_once(draw_ctx);
    }
    mglPlatformShellSetMSSampleState(renderer, 0, 0, 0);
    mglPostDepthSnapshotRelease();
    return 1;
}

static void mglBroadcastMSPlanes(void *blit, Texture *tex, FBOAttachment *att)
{
    const uint64_t samples = tex->samples > 1u ? (uint64_t)tex->samples : 1u;
    MGLMetalAttachmentSubresource sub =
        mglMetalAttachmentSubresourceForAttachment(att);
    const uint64_t baseSlice = sub.slice;
    const uint64_t level = sub.level;
    MGLRenderTextureInfo info = {0};
    (void)mglRenderGetTextureInfo(mglTextureMtl(tex), &info);
    if (samples <= 1u || info.width == 0u || info.height == 0u ||
        baseSlice + samples > info.array_length) {
        return;
    }
    for (uint64_t s = 1u; s < samples; s++) {
        (void)mglRenderBlitCopyTexture(
            blit, mglTextureMtl(tex), baseSlice, level, 0u, 0u, 0u, info.width,
            info.height, 1u, mglTextureMtl(tex), baseSlice + s, level, 0u, 0u, 0u);
    }
    mglMarkTextureLevelRenderTargetWrittenImpl(tex, (GLuint)level, __func__, __LINE__);
}

void mglRendererBroadcastEmulatedMSSamplePlanesAfterDrawIfNeeded(void *renderer,
                                                                 GLMContext ctx)
{
    if (mglPlatformShellMSSampleInLoop(renderer)) {
        return;
    }
    Framebuffer *fbo = ctx->active_state->framebuffer;
    if (!fbo) {
        return;
    }
    Texture *texs[3] = {
        mglDrawEmulatedMSColor0Texture(ctx),
        mglDrawEmulatedMSAttachmentTexture(ctx, &fbo->depth),
        mglDrawEmulatedMSAttachmentTexture(ctx, &fbo->stencil)};
    FBOAttachment *atts[3] = {&fbo->color_attachments[0], &fbo->depth,
                              &fbo->stencil};
    if (texs[2] == texs[1]) {
        texs[2] = NULL;
    }
    for (int i = 0; i < 3; i++) {
        if (texs[i] && !mglTextureMtl(texs[i])) {
            texs[i] = NULL;
        }
    }
    if (!texs[0] && !texs[1] && !texs[2]) {
        return;
    }
    /* Per-sample draws already filled each plane. */
    if (mglDrawFragmentNeedsPerSampleMSValues(ctx)) {
        return;
    }

    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    mglRendererEndRenderEncodingLocked(renderer);
    if (!cs->currentCommandBufferOwner &&
        !mglPlatformShellNewCommandBuffer(renderer)) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(cs->currentCommandBufferOwner);
    if (!blit) {
        return;
    }
    for (int i = 0; i < 3; i++) {
        if (texs[i]) {
            mglBroadcastMSPlanes(blit, texs[i], atts[i]);
        }
    }
    (void)mglRenderEndBlitEncoder(blit);
}
