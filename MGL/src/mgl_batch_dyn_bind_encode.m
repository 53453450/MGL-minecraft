/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: dyn-bind / sampler / simple-replay encode (Batch cluster).
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_byte_hash.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"
#include <string.h>

static const NSUInteger kMaxFragmentSamplerSlots = 16;

static BOOL mglBatchReplayHasActiveEncoder(const MGLEncodeContext *e)
{
    return e && mglRenderEncoderOwnerHasCurrent(e->render_encoder_owner) != 0;
}
static void mglBatchRefreshEncodeOwner(MGLEncodeContext *e, void *o)
{
    if (e) e->render_encoder_owner = o;
}
static uint64_t mglRendererSamplerSnapshotHash(const MGLSamplerSnapshotKey *key)
{
    return mglHashBytesFNV1a(key, sizeof(*key));
}

typedef struct {
    __unsafe_unretained MGLRenderer *r;
    VertexArray *vao;
    const MGLDrawCommand *cmd;
    GLMContext ctx;
    Program *prog;
} MGLDynVertexCtx;
static int mglDynVertexPlan(void *v, uint8_t bi, MGLBatchDynVertexStreamPlan *p)
{
    MGLDynVertexCtx *c = (MGLDynVertexCtx *)v;
    return mgl_batch_replay_plan_dyn_vertex_streams(
        c->ctx, c->vao, c->prog, &c->cmd->dynamic_vertex_bindings[bi], p);
}
static int mglDynVertexResolve(void *v, uint32_t attrib, int *slot)
{
    MGLDynVertexCtx *c = (MGLDynVertexCtx *)v;
    int s = mglRendererResolveVertexAttributeBufferIndex(c->ctx, c->vao, attrib,
                                                         __FUNCTION__);
    if (slot) *slot = s;
    return s >= 0;
}
static int mglDynVertexCanBind(void *v, const MGLBatchDynVertexStreamPlan *p,
                               uint32_t stream)
{
    MGLDynVertexCtx *c = (MGLDynVertexCtx *)v;
    return mgl_batch_replay_dyn_vertex_stream_can_bind_directly(c->prog, c->vao, p,
                                                                stream);
}
static int mglDynVertexEnsure(void *v, const MGLBatchDynVertexStreamPlan *p,
                              void **mtl, void **gl, uint64_t *dyn, uint64_t *len)
{
    MGLDynVertexCtx *c = (MGLDynVertexCtx *)v;
    Buffer *buf = p->buffer;
    if (!buf) return 0;
    if (buf->data.dirty_bits) {
        BufferMapList upload = {0};
        upload.count = 1;
        upload.buffers[0].buf = buf;
        if (![c->r updateDirtyBaseBufferList:&upload]) return 0;
    }
    if (!buf->data.mtl_data) [c->r bindMTLBuffer:buf];
    if (!mgl_batch_replay_mtl_ptr_ok(buf->data.mtl_data)) return 0;
    id mb = (__bridge id)buf->data.mtl_data;
    MGLRenderBufferInfo info = {0};
    if (mglRenderGetBufferInfo((__bridge void *)mb, &info) != 0) return 0;
    if (mtl) *mtl = (__bridge void *)mb;
    if (gl) *gl = buf;
    if (dyn) *dyn = (uint64_t)p->dynamic_offset;
    if (len) *len = info.length;
    return 1;
}
static uint64_t mglDynVertexBindOff(void *v, const MGLBatchDynVertexStreamPlan *p)
{
    return (uint64_t)((MGLDynVertexCtx *)v)->vao->bindings[p->binding_index].offset;
}

typedef struct {
    __unsafe_unretained MGLRenderer *r;
    const MGLDrawCommand *cmd;
    GLMContext ctx;
} MGLDynUniformCtx;
static int mglDynUniformGather(void *v, uint64_t *lens, uint32_t count)
{
    MGLDynUniformCtx *c = (MGLDynUniformCtx *)v;
    for (uint32_t i = 0; i < count; i++) {
        BufferBaseTarget *slot =
            &MGL_STATE(c->ctx)
                 ->buffer_base[_UNIFORM_BUFFER]
                 .buffers[c->cmd->dynamic_uniform_bindings[i].binding_index];
        if (!slot->buf || !mgl_batch_replay_mtl_ptr_ok(slot->buf->data.mtl_data))
            return 0;
        MGLRenderBufferInfo info = {0};
        if (mglRenderGetBufferInfo(slot->buf->data.mtl_data, &info) != 0) return 0;
        lens[i] = info.length;
    }
    return 1;
}
static int mglDynUniformResolve(void *v, const MGLBatchUniformBindOp *op, void **mtl,
                                void **gl)
{
    MGLDynUniformCtx *c = (MGLDynUniformCtx *)v;
    BufferBaseTarget *slot =
        &MGL_STATE(c->ctx)->buffer_base[_UNIFORM_BUFFER].buffers[op->binding_index];
    if (!slot->buf || !mgl_batch_replay_mtl_ptr_ok(slot->buf->data.mtl_data)) return 0;
    if (mtl) *mtl = slot->buf->data.mtl_data;
    if (gl) *gl = slot->buf;
    return 1;
}

typedef struct {
    __unsafe_unretained MGLRenderer *r;
    const MGLDrawCommand *cmd;
    GLMContext ctx;
    MGLEncodeContext *enc;
    VertexArray dynamic_vao;
    VertexArray *draw_vao;
    VertexArray *base_vao;
    VertexArray *saved_vao;
    bool touched[TEXTURE_UNITS];
} MGLDynApplyCtx;
static void mglDynApplyRefresh(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    if (c->enc)
        c->enc->render_encoder_owner =
            c->r->_renderPassManager.state->currentRenderEncoderOwner;
}
static int mglDynApplyHasEnc(void *v)
{
    return mglBatchReplayHasActiveEncoder(((MGLDynApplyCtx *)v)->enc) ? 1 : 0;
}
static int mglDynApplyBuildVao(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    if (!c->base_vao || c->base_vao->magic != MGL_VAO_MAGIC ||
        !mgl_batch_replay_build_dynamic_vertex_array(c->ctx, c->base_vao, c->cmd,
                                                     &c->dynamic_vao))
        return 0;
    c->draw_vao = &c->dynamic_vao;
    return 1;
}
static int mglDynApplyUbo(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    return mgl_batch_replay_apply_uniform_range_overrides(c->ctx, c->cmd) ? 1 : 0;
}
static int mglDynApplyTex(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    memset(c->touched, 0, sizeof(c->touched));
    return mgl_batch_replay_apply_texture_overrides(c->ctx, c->cmd, c->touched,
                                                    TEXTURE_UNITS)
               ? 1 : 0;
}
static int mglDynApplyBindTexDirect(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    return [c->r bindDynamicSampledTexturesDirectlyForTouchedUnits:c->touched
                                                           context:c->ctx
                                                     encodeContext:c->enc]
               ? 1 : 0;
}
static int mglDynApplyBindTexMapper(void *v)
{
    return [((MGLDynApplyCtx *)v)->r
               bindTexturesToCurrentRenderEncoder:((MGLDynApplyCtx *)v)->enc]
               ? 1 : 0;
}
static int mglDynApplyRestoreTex(void *v)
{
    return [((MGLDynApplyCtx *)v)->r
               restoreRenderEncoderAfterTextureUploadForDraw:
                   "dynamic-sampled-texture-bind"]
               ? 1 : 0;
}
static int mglDynApplyBindVertex(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    return [c->r bindDynamicVertexArrayBuffersDirectly:c->draw_vao command:c->cmd
                                               context:c->ctx
                                         encodeContext:c->enc]
               ? 1 : 0;
}
static int mglDynApplyBindUniform(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    return [c->r bindDynamicUniformRangesDirectly:c->cmd context:c->ctx
                                    encodeContext:c->enc]
               ? 1 : 0;
}
static int mglDynApplyMapperFallback(void *v)
{
    MGLDynApplyCtx *c = (MGLDynApplyCtx *)v;
    c->saved_vao = MGL_STATE(c->ctx)->vao;
    if (c->cmd->dynamic_vertex_binding_count > 0) MGL_STATE(c->ctx)->vao = c->draw_vao;
    mglDynApplyRefresh(v);
    int ok = ([c->r mapBuffersToMTL] &&
              [c->r bindVertexBuffersToCurrentRenderEncoder:c->enc])
                 ? 1 : 0;
    mglDynApplyRefresh(v);
    if (ok && c->cmd->dynamic_uniform_binding_count > 0) {
        ok = [c->r bindFragmentBuffersToCurrentRenderEncoder:c->enc] ? 1 : 0;
        mglDynApplyRefresh(v);
    }
    MGL_STATE(c->ctx)->vao = c->saved_vao;
    return ok;
}


typedef struct {
    __unsafe_unretained MGLRenderer *r;
    GLMContext ctx;
} MGLDynSampledCtx;
/* 1=append, 0=skip unit, -1=fail */
static int mglDynSampledResolve(void *v, const MGLBatchSampledTexCandidate *e,
                                const bool *touched, void **tex_out,
                                uint32_t *stage_out, int *needs_samp,
                                void **samp_out, uint32_t *samp_slot)
{
    MGLDynSampledCtx *c = (MGLDynSampledCtx *)v;
    MGLShaderResource *resource = e->resource;
    GLuint unit = [c->r textureUnitForSampledResource:resource
                                          metalBinding:e->metal_slot
                                                 stage:(int)e->stage];
    if (unit >= TEXTURE_UNITS || !touched[unit]) return 0;
    Texture *tex_obj =
        [c->r textureForSampledResource:resource
                           metalBinding:e->metal_slot
                                   stage:(int)e->stage
                            expectedType:(e->lookup_type ? e->lookup_type
                                                         : e->expected_type)];
    if (!mgl_batch_replay_sampled_tex_object_ok(
            tex_obj ? 1 : 0, tex_obj && tex_obj->mtl_data ? 1 : 0,
            tex_obj && tex_obj->dirty_bits ? 1 : 0,
            tex_obj && tex_obj->is_render_target ? 1 : 0))
        return -1;
    id texture = (__bridge id)mglSampledTextureViewForBaseLevel(
        tex_obj, tex_obj->mtl_data);
    MGLRenderTextureInfo info = {0};
    int info_ok =
        texture && mglRenderGetTextureInfo((__bridge void *)texture, &info) == 0;
    if (!mgl_batch_replay_sampled_tex_info_ok(
            info_ok, info.texture_type, e->expected_type,
            mglTexturePixelFormatCompatibleWithExpectedDataKind(
                info.pixel_format, (MGLTextureDataKind)e->expected_kind)))
        return -1;
    if (stage_out)
        *stage_out = mglRenderTextureBindingStageForShader((int)e->stage);
    if (tex_out) *tex_out = (__bridge void *)texture;
    if (!e->needs_combined_sampler) {
        if (needs_samp) *needs_samp = 0;
        return 1;
    }
    id sampler = nil;
    Sampler *bound = MGL_STATE(c->ctx)->texture_samplers[unit];
    if (bound) {
        if (bound->dirty_bits || !bound->mtl_data) return -1;
        sampler = (__bridge id)bound->mtl_data;
    } else if (tex_obj->params.mtl_data) {
        sampler = (__bridge id)tex_obj->params.mtl_data;
    } else {
        return -1;
    }
    if (needs_samp) *needs_samp = 1;
    if (samp_out) *samp_out = (__bridge void *)sampler;
    if (samp_slot)
        *samp_slot = resource ? mglMetalCombinedSamplerSlot(resource)
                              : e->metal_slot;
    return 1;
}

@implementation MGLRenderer (Draw)

- (bool)bindDynamicVertexArrayBuffersDirectly:(VertexArray *)vao
                                      command:(const MGLDrawCommand *)cmd
                                       context:(GLMContext)glm_ctx
                                 encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!vao || !cmd || !encCtx) return false;
    MGLDynVertexCtx c = {.r = self,
                         .vao = vao,
                         .cmd = cmd,
                         .ctx = glm_ctx,
                         .prog = mglResolveProgramForStageFromState(glm_ctx,
                                                                    _VERTEX_SHADER)};
    MGLBatchDynVertexBindOps ops = {
        .ctx = &c,
        .binding_count = cmd->dynamic_vertex_binding_count,
        .max_metal_slots = (int)kMGLMaxMetalVertexBufferCount,
        .binding_state_owner = _bindingStateOwner,
        .render_encoder_owner = encCtx->render_encoder_owner,
        .plan_binding = mglDynVertexPlan,
        .resolve_slot = mglDynVertexResolve,
        .stream_can_bind = mglDynVertexCanBind,
        .ensure_mtl = mglDynVertexEnsure,
        .vao_binding_offset = mglDynVertexBindOff,
    };
    return mgl_batch_mtl_bind_dyn_vertex(&ops) ? true : false;
}

- (bool)bindDynamicUniformRangesDirectly:(const MGLDrawCommand *)cmd
                                  context:(GLMContext)glm_ctx
                            encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!cmd || !glm_ctx || !encCtx) return false;
    MGLDynUniformCtx c = {.r = self, .cmd = cmd, .ctx = glm_ctx};
    MGLBatchDynUniformBindOps ops = {
        .ctx = &c,
        .binding_state_owner = _bindingStateOwner,
        .render_encoder_owner = encCtx->render_encoder_owner,
        .min_stage_binding_size = (uint64_t)kMGLMinimumStageBindingSize,
        .max_buffer_slots = (uint32_t)kMGLMaxBufferSlots,
        .gather_lengths = mglDynUniformGather,
        .cmd = cmd,
        .glm_ctx = glm_ctx,
        .resolve_op = mglDynUniformResolve,
    };
    return mgl_batch_mtl_bind_dyn_uniforms(&ops) ? true : false;
}

- (bool)bindDynamicSampledTexturesDirectlyForTouchedUnits:(const bool *)touched_units
                                                   context:(GLMContext)glm_ctx
                                             encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!touched_units || !glm_ctx || !mglBatchReplayHasActiveEncoder(encCtx))
        return false;
    MGLDynSampledCtx c = {.r = self, .ctx = glm_ctx};
    MGLBatchDynSampledBindOps ops = {
        .ctx = &c,
        .binding_state_owner = _bindingStateOwner,
        .render_encoder_owner = encCtx->render_encoder_owner,
        .max_sampler_slots = (uint32_t)kMaxFragmentSamplerSlots,
        .glm_ctx = glm_ctx,
        .resolve_candidate = mglDynSampledResolve,
        .touched_units = touched_units,
    };
    return mgl_batch_mtl_bind_dyn_sampled(&ops) ? true : false;
}

- (id)samplerStateForSnapshotKey:(const MGLSamplerSnapshotKey *)key
{
    if (!key) return nil;
    void *cachedState = NULL;
    int cacheResult = mglRendererBackendGetSamplerSnapshotState(
        _backend, key, &cachedState);
    if (cacheResult == 1) {
        return (__bridge id)cachedState;
    }
    if (cacheResult < 0) {
        return nil;
    }

    TextureParameter params;
    mgl_batch_replay_fill_sampler_params(key, &params);
    id state =
        [self createMTLSamplerForTexParam:&params target:key->target];
    if (!state) return nil;
    return mglRendererBackendPutSamplerSnapshotState(
        _backend, key, (__bridge void *)state) == 0 ? state : nil;
}

- (bool)applySamplerSnapshotForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!cmd || !glm_ctx) return false;
    if (cmd->sampler_snapshot_id == MGL_INVALID_SAMPLER_SNAPSHOT_ID) return true;
    if (!mglBatchReplayHasActiveEncoder(encCtx)) return false;
    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cmd->sampler_snapshot_id >= cb->sampler_snapshot_set_count) return false;
    const MGLSamplerSnapshotSet *set =
        &cb->sampler_snapshot_sets[cmd->sampler_snapshot_id];
    if (set->count > MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES) return false;
    MGLBatchResolvedSamplerBind items[MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES];
    uint32_t n = 0u;
    for (uint8_t i = 0; i < set->count; i++) {
        const MGLSamplerSnapshotEntry *entry = &set->entries[i];
        if (!mgl_batch_replay_sampler_slot_ok(entry->metal_slot, 16u)) return false;
        id sampler =
            (entry->key_index == MGL_FALLBACK_SAMPLER_KEY_INDEX)
                ? [self fallbackSamplerState]
                : (entry->key_index < cb->sampler_snapshot_key_count
                       ? [self samplerStateForSnapshotKey:
                              &cb->sampler_snapshot_keys[entry->key_index]]
                       : nil);
        if (!sampler) return false;
        if (mglMipDiagEnabled() && entry->stage == _FRAGMENT_SHADER &&
            entry->key_index != MGL_FALLBACK_SAMPLER_KEY_INDEX) {
            const MGLSamplerSnapshotKey *key =
                &cb->sampler_snapshot_keys[entry->key_index];
            static uint64_t s_snapshotState[16];
            if (mglMipDiagStateChanged(&s_snapshotState[entry->metal_slot],
                                       mglRendererSamplerSnapshotHash(key))) {
                NSLog(@"MGL MIP_DIAG snapshot slot=%u unit=%u target=0x%x "
                      @"minFilter=0x%x magFilter=0x%x minLod=%.1f maxLod=%.1f aniso=%.1f",
                      (unsigned)entry->metal_slot, (unsigned)entry->texture_unit,
                      (unsigned)key->target, (unsigned)key->min_filter,
                      (unsigned)key->mag_filter, (double)key->min_lod,
                      (double)key->max_lod, (double)key->max_anisotropy);
            }
        }
        items[n++] = (MGLBatchResolvedSamplerBind){
            .sampler = (__bridge void *)sampler,
            .metal_slot = entry->metal_slot,
            .shader_stage = (int)entry->stage,
        };
    }
    return mgl_batch_mtl_encode_resolved_samplers(
               _bindingStateOwner, encCtx->render_encoder_owner, items, n)
               ? true : false;
}

- (bool)applyDynamicBindingsForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(MGLEncodeContext *)encCtx
{
    if (!cmd) return true;
    if (!glm_ctx || !encCtx) return false;
    encCtx->render_encoder_owner =
        _renderPassManager.state->currentRenderEncoderOwner;
    MGLDynApplyCtx c = {.r = self,
                        .cmd = cmd,
                        .ctx = glm_ctx,
                        .enc = encCtx,
                        .base_vao = MGL_STATE(glm_ctx)->vao,
                        .draw_vao = MGL_STATE(glm_ctx)->vao};
    MGLBatchDynApplyOps ops = {
        .ctx = &c,
        .refresh_owner = mglDynApplyRefresh,
        .has_encoder = mglDynApplyHasEnc,
        .build_dyn_vao = mglDynApplyBuildVao,
        .apply_ubo = mglDynApplyUbo,
        .apply_tex = mglDynApplyTex,
        .bind_tex_direct = mglDynApplyBindTexDirect,
        .bind_tex_mapper = mglDynApplyBindTexMapper,
        .restore_after_tex_upload = mglDynApplyRestoreTex,
        .bind_vertex_direct = mglDynApplyBindVertex,
        .bind_uniform_direct = mglDynApplyBindUniform,
        .mapper_fallback = mglDynApplyMapperFallback,
    };
    return mgl_batch_issue_apply_dyn_bindings(cmd->dynamic_vertex_binding_count,
                                              cmd->dynamic_uniform_binding_count,
                                              cmd->dynamic_texture_binding_count,
                                              &ops)
               ? true : false;
}

typedef struct {
    __unsafe_unretained MGLRenderer *r;
    MGLDrawBatch *batch;
    GLMContext ctx;
} MGLSimpleReplayCtx;

static int mglSimpleResolve(void *v, uint32_t i, uint32_t gl_itype, void **mtl,
                            uint64_t *ioff, uint32_t *mtype)
{
    MGLSimpleReplayCtx *c = (MGLSimpleReplayCtx *)v;
    MGLDrawCommand *cmd = &c->batch->commands[i];
    Buffer *glBuf = NULL;
    id idxBuf = nil;
    if (![c->r resolveElementBufferForCommand:cmd label:"cppBatchReplay"
                                      context:c->ctx glBuffer:&glBuf
                                    mtlBuffer:&idxBuf])
        return 0;
    NSUInteger off = ioff ? (NSUInteger)*ioff : cmd->indexBufferOffset;
    uint64_t itype = mglIndexTypeForGLType((GLenum)gl_itype);
    id prepared = mglPreparedElementIndexBuffer(
        c->r->_device, glBuf, idxBuf, (GLenum)gl_itype, &off, &itype);
    if (ioff) *ioff = (uint64_t)off;
    if (mtype) *mtype = (uint32_t)itype;
    if (mtl) *mtl = (__bridge void *)prepared;
    return prepared ? 1 : 0;
}

- (BOOL)tryReplaySimpleBatch:(MGLDrawBatch *)batch
                            context:(GLMContext)glm_ctx
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!batch || batch->command_count == 0u) return NO;
    Program *batchProgram =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    const GLenum batchMode = batch->commands[0].mode;
    if (!mgl_batch_replay_simple_eligible(
            batch, MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS,
            mglBatchReplayHasActiveEncoder(encCtx) ? 1 : 0,
            (batchProgram && batchProgram->uses_cull_distance) ? 1 : 0,
            MGL_STATE(glm_ctx)->caps.primitive_restart ? 1 : 0,
            mglPolygonModePointForDrawMode(glm_ctx, batchMode) ? 1 : 0,
            mglRenderDrawModeNeedsEmulate((uint32_t)batchMode) ? 1 : 0)) {
        return NO;
    }
    MGLSimpleReplayCtx ctx = {self, batch, glm_ctx};
    MGLBatchSimpleReplayOps ops = {.ctx = &ctx, .resolve_index = mglSimpleResolve};
    return mgl_batch_mtl_issue_simple_replay(
               batch, encCtx->render_encoder_owner, &ops)
               ? YES
               : NO;
}

@end
