/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Batch.m — thin Batch category shell (A3 / O2.5).
// Flush/restore/check/stream/schedule encode → mgl_batch_flush_restore_encode.m
// ICB/MDI → mgl_batch_icb_mdi_encode.m; RT-mark → mgl_batch_rt_mark_port.m;
// traces → mgl_batch_replay_trace.m. Keep: unlocked flush ABI, binding helpers.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#import "mgl_sampler_compat.h"
#include "mgl_render.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_restore.h"

@implementation MGLRenderer (Batch)

- (void)recordArrayDrawSubmittedMode:(GLenum)mode vertexCount:(uint64_t)vertexCount
{
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglTraceNowSeconds());
    MGL_FRAME_STORE(g_mglLastDrawArraysProgram, mglCurrentRenderProgramKey(ctx));
    MGL_FRAME_STORE(g_mglLastDrawArraysMode, mode);
    MGL_FRAME_STORE(g_mglLastDrawArraysCount,
                    (vertexCount > (uint64_t)INT_MAX) ? INT_MAX : (GLsizei)vertexCount);
    MGL_FRAME_INC(g_mglDrawArraysSinceSwap);
    MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap, vertexCount);
    [self markCurrentFramebufferDrawAttachmentsWritten];
}

- (void)recordElementDrawSubmittedMode:(GLenum)mode indexCount:(uint64_t)indexCount
{
    MGL_FRAME_STORE(g_mglLastDrawElementsSeconds, mglTraceNowSeconds());
    MGL_FRAME_STORE(g_mglLastDrawElementsProgram, mglCurrentRenderProgramKey(ctx));
    MGL_FRAME_STORE(g_mglLastDrawElementsMode, mode);
    MGL_FRAME_STORE(g_mglLastDrawElementsCount,
                    (indexCount > (uint64_t)INT_MAX) ? INT_MAX : (GLsizei)indexCount);
    MGL_FRAME_INC(g_mglDrawElementsSinceSwap);
    MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap, indexCount);
    [self markCurrentFramebufferDrawAttachmentsWritten];
}

typedef struct { __unsafe_unretained MGLRenderer *r; GLMContext glm; } ActTexCtx;
static int actBindUnit(void *v, uint32_t unit, int *stale_out)
{
    ActTexCtx *c = v; Texture *tex = MGL_STATE(c->glm)->active_textures[unit];
    if (!tex) { if (stale_out) *stale_out = 1; return 0; }
    if (stale_out) *stale_out = 0;
    return [c->r bindMTLTexture:tex] ? 1 : 0;
}
static void actClearStale(void *v, uint32_t word, uint32_t bit)
{
    ActTexCtx *c = v;
    MGL_STATE(c->glm)->active_texture_mask[word] &= ~(1u << bit);
    mglInvalidateStateHashCachesForDirtyBits(c->glm->active_state, DIRTY_TEX_BINDING);
}

- (bool)bindActiveTexturesToMTL
{
    ActTexCtx c = {.r = self, .glm = ctx};
    MGLBatchActiveTexBindOps ops = {
        .ctx = &c, .mask4 = MGL_STATE(ctx)->active_texture_mask,
        .bind_unit = actBindUnit, .clear_stale = actClearStale,
    };
    return mgl_batch_bind_active_textures(&ops) ? true : false;
}

- (void)invalidateLastBoundState
{
    mglRenderBindingInvalidate(_bindingStateOwner);
}

/* DUAL-PROXY INVARIANT HELPERS: see MGLRenderer_Private.h.
 *
 * These centralize all writes to _core.activeState and ctx->active_state so
 * that the invariant ("MGL_STATE(ctx) and MGL_STATE(ctx)->ctx) return the same
 * GLMState") cannot be broken by a caller forgetting to update one side.
 *
 * Valid invariant configurations:
 *   (A) _activeState == NULL  -> MGL_STATE falls through to ctx->active_state
 *                                (the "deactivated" / default mode)
 *   (B) _activeState != NULL  -> _activeState MUST equal ctx->active_state
 *                                (the "activated" mode used during batch replay)
 *
 * Configuration (A) is the teardown target; (B) is the batch-replay target. */
/* Configuration (A) is the teardown target; (B) is batch-replay via
 * replay_state (ARCHITECTURE_AUDIT R3). */
- (void)mglActivateReplayStateForContext:(GLMContext)glm_ctx
{
    memcpy(&glm_ctx->replay_state, &glm_ctx->state, sizeof(glm_ctx->replay_state));
    glm_ctx->active_state = &glm_ctx->replay_state;
    _core.activeState = &glm_ctx->replay_state;
}

- (void)mglRestoreLiveActiveStateForContext:(GLMContext)glm_ctx
{
    /* Configuration (A): ctx->active_state points to live embedded state,
     * _activeState is NULL so MGL_STATE() falls through. */
    glm_ctx->active_state = &glm_ctx->state;
    _core.activeState = NULL;
}

- (void)mglAssertDualProxyInSyncForContext:(GLMContext)glm_ctx
{
    /* Invariant checkpoint.  NSCAssert is compiled out in release builds,
     * so this is zero-cost in shipping binaries.  In debug builds it catches
     * desync at the earliest observation point (function entry/exit) instead
     * of letting it manifest as wrong binds/dirty bits later. */
    NSCAssert(_core.activeState == NULL || _core.activeState == glm_ctx->active_state,
              @"DUAL-PROXY DESYNC: _activeState != ctx->active_state — "
              @"MGL_STATE() and STATE() would read different GLMState objects");
}

- (void)recordLastBoundVertexBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    mglRenderBindingRecordVertexBuffer(
        _bindingStateOwner, (__bridge void *)buffer, offset, (uint32_t)index);
}

- (void)recordLastBoundFragmentBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    mglRenderBindingRecordFragmentBuffer(
        _bindingStateOwner, (__bridge void *)buffer, offset, (uint32_t)index);
}

- (void)invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index
{
    mglRenderBindingInvalidateVertexBuffer(
        _bindingStateOwner, (uint32_t)index);
}

- (void)invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index
{
    mglRenderBindingInvalidateFragmentBuffer(
        _bindingStateOwner, (uint32_t)index);
}

- (void)setViewportIfNeeded:(MGLViewportValue)viewport
{
    void *owner = _renderPassManager.state->currentRenderEncoderOwner;
    mglRenderBindingSetViewportForOwner(
        _bindingStateOwner, owner, viewport.origin_x, viewport.origin_y,
        viewport.width, viewport.height, viewport.znear, viewport.zfar);
}

- (void)setScissorRectIfNeeded:(MGLScissorRectValue)rect
{
    void *owner = _renderPassManager.state->currentRenderEncoderOwner;
    mglRenderBindingSetScissorForOwner(
        _bindingStateOwner, owner, rect.x, rect.y, rect.width, rect.height);
}

- (void)setTriangleFillModeIfNeeded:(uint32_t)mode
{
    void *owner = _renderPassManager.state->currentRenderEncoderOwner;
    mglRenderBindingSetTriangleFillForOwner(
        _bindingStateOwner, owner, (uint32_t)mode);
}

- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx
                           alreadyDone:(const MGLResourceSyncWork *)done
{
    GLMState *state = MGL_STATE(glm_ctx);
    if (!done || !done->mappedBuffers) {
        RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    }
    if (!done || !done->updatedBaseLists) {
        RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->vertex_buffer_map_list]);
        RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->fragment_buffer_map_list]);
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner,
    };
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindBufferSizeConstantsForRenderEncoder]);
    if (!done || !done->boundActiveTextures) {
        RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);
    }
    RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-active-texture-bind"]);
    if (![self bindTexturesToCurrentRenderEncoder:&encCtx]) {
        RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-sampled-texture-bind"]);
        RETURN_FALSE_ON_FAILURE([self bindTexturesToCurrentRenderEncoder:&encCtx]);
    }
    return true;
}

typedef struct { GLMContext glm; } KeyRestCtx;
static void keyRestProg(void *v, uint32_t prog, uint32_t pipe)
{ mglRestoreProgramPipelinePair(((KeyRestCtx *)v)->glm, prog, pipe); }
static void keyRestVao(void *v, uint32_t name)
{
    GLMContext glm = ((KeyRestCtx *)v)->glm;
    if (name == (MGL_STATE(glm)->vao ? MGL_STATE(glm)->vao->name : 0)) return;
    MGL_STATE(glm)->vao = name
        ? (VertexArray *)searchHashTable(&MGL_STATE(glm)->vao_table, name) : NULL;
}
static void keyRestFbo(void *v, uint32_t name)
{
    GLMContext glm = ((KeyRestCtx *)v)->glm;
    uint32_t cur = MGL_STATE(glm)->framebuffer ? MGL_STATE(glm)->framebuffer->name : 0;
    if (name == cur) return;
    MGL_STATE(glm)->framebuffer = name
        ? (Framebuffer *)searchHashTable(&MGL_STATE(glm)->framebuffer_table, name) : NULL;
}
static void keyRestSync(void *v) { mglRendererSyncFramebufferBindingNames(((KeyRestCtx *)v)->glm); }
static void keyRestVpSc(void *v, const int32_t vp[4], int sc_en, const int32_t sc[4])
{
    GLMContext glm = ((KeyRestCtx *)v)->glm;
    for (int i = 0; i < 4; i++) MGL_STATE(glm)->viewport[i] = vp[i];
    if (sc_en) {
        MGL_STATE(glm)->caps.scissor_test = true;
        for (int i = 0; i < 4; i++) MGL_STATE(glm)->var.scissor_box[i] = sc[i];
    } else {
        MGL_STATE(glm)->caps.scissor_test = false;
    }
}

- (void)restoreStateFromKey:(const MGLStateKey *)key context:(GLMContext)glm_ctx
{
    KeyRestCtx c = {.glm = glm_ctx};
    MGLBatchRestoreFromKeyOps ops = {
        .ctx = &c, .program_name = key->program_name,
        .program_pipeline_name = key->program_pipeline_name,
        .vao_name = key->vao_name, .fbo_name = key->fbo_name,
        .scissor_enabled = key->scissor_enabled,
        .restore_program = keyRestProg, .set_vao = keyRestVao, .set_fbo = keyRestFbo,
        .sync_fbo_names = keyRestSync, .apply_viewport_scissor = keyRestVpSc,
    };
    for (int i = 0; i < 4; i++) { ops.viewport[i] = key->viewport[i]; ops.scissor[i] = key->scissor[i]; }
    mgl_batch_restore_apply_from_key(&ops);
}

- (void)flushDrawBuffer:(GLMContext)glm_ctx
{
    /* Unlocked entry point: acquire METAL_LOCK and delegate to Locked variant.
     * Locked callers (mtlSwapBuffersLocked:, flushCommandBufferLocked:) call
     * flushDrawBufferLocked: directly to avoid recursive lock re-entry. */
    METAL_LOCK();
    @try {
        [self flushDrawBufferLocked:glm_ctx];
    } @finally {
        METAL_UNLOCK();
    }
}

void mglRendererFlushDrawBuffer(GLMContext glm_ctx)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        @autoreleasepool {
            @try {
                [renderer flushDrawBuffer:glm_ctx];
            } @catch (NSException *exception) {
                NSLog(@"MGL ERROR: callback flushDrawBuffer exception: %@", exception);
            }
        }
    }
    mglRendererBackendEnd(&_backend_lease);
}

@end
