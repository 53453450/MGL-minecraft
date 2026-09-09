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
// traces → mgl_batch_replay_trace.m. Keep: unlocked flush ABI, dual-proxy, restore-from-key, active-tex.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+BatchPorts_Private.h"
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
    ActTexCtx *c = v; Texture *tex = c->glm->active_state->active_textures[unit];
    if (!tex) { if (stale_out) *stale_out = 1; return 0; }
    if (stale_out) *stale_out = 0;
    return [c->r bindMTLTexture:tex] ? 1 : 0;
}
static void actClearStale(void *v, uint32_t word, uint32_t bit)
{
    ActTexCtx *c = v;
    c->glm->active_state->active_texture_mask[word] &= ~(1u << bit);
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

/* Dual-proxy: (A) _activeState NULL → live; (B) equals ctx->active_state (replay). */
- (void)mglActivateReplayStateForContext:(GLMContext)glm_ctx
{
    memcpy(&glm_ctx->replay_state, &glm_ctx->state, sizeof(glm_ctx->replay_state));
    glm_ctx->active_state = &glm_ctx->replay_state;
    _core.activeState = &glm_ctx->replay_state;
}

- (void)mglRestoreLiveActiveStateForContext:(GLMContext)glm_ctx
{
    glm_ctx->active_state = &glm_ctx->state;
    _core.activeState = NULL;
}

- (void)mglAssertDualProxyInSyncForContext:(GLMContext)glm_ctx
{
    NSCAssert(_core.activeState == NULL || _core.activeState == glm_ctx->active_state,
              @"DUAL-PROXY DESYNC: _activeState != ctx->active_state — "
              @"MGL_STATE() and STATE() would read different GLMState objects");
}

typedef struct { GLMContext glm; } KeyRestCtx;
static void keyRestProg(void *v, uint32_t prog, uint32_t pipe)
{ mglRestoreProgramPipelinePair(((KeyRestCtx *)v)->glm, prog, pipe); }
static void keyRestVao(void *v, uint32_t name)
{
    GLMContext glm = ((KeyRestCtx *)v)->glm;
    if (name == (glm->active_state->vao ? glm->active_state->vao->name : 0)) return;
    glm->active_state->vao = name
        ? (VertexArray *)searchHashTable(&glm->active_state->vao_table, name) : NULL;
}
static void keyRestFbo(void *v, uint32_t name)
{
    GLMContext glm = ((KeyRestCtx *)v)->glm;
    uint32_t cur = glm->active_state->framebuffer ? glm->active_state->framebuffer->name : 0;
    if (name == cur) return;
    glm->active_state->framebuffer = name
        ? (Framebuffer *)searchHashTable(&glm->active_state->framebuffer_table, name) : NULL;
}
static void keyRestSync(void *v) { mglRendererSyncFramebufferBindingNames(((KeyRestCtx *)v)->glm); }
static void keyRestVpSc(void *v, const int32_t vp[4], int sc_en, const int32_t sc[4])
{
    GLMContext glm = ((KeyRestCtx *)v)->glm;
    for (int i = 0; i < 4; i++) glm->active_state->viewport[i] = vp[i];
    if (sc_en) {
        glm->active_state->caps.scissor_test = true;
        for (int i = 0; i < 4; i++) glm->active_state->var.scissor_box[i] = sc[i];
    } else {
        glm->active_state->caps.scissor_test = false;
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
