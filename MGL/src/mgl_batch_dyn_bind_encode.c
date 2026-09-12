/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: dyn-bind / sampler / simple-replay encode (Batch cluster).
 *
 * Formerly mgl_batch_dyn_bind_encode.m.  The six entry points are C drivers
 * now; every renderer operation they need goes through mgl_renderer_ports.h.
 */
#include "mgl_renderer_ports.h"       /* C port surface (T4) */
#include "mgl_draw_issue.h"           /* mglDrawHostDevice */
#include "mgl_vertex_attrib_query.h"  /* mglRendererResolveVertexAttributeBufferIndex */
#include "mgl_state_log.h"            /* mglMipDiag* */
#include "mgl_buffer_slots.h"         /* kmax slot constants */
#include "mgl_texture_compat.h"       /* mglSampledTextureViewForBaseLevel */
#include "mgl_shader_resource.h"      /* mglMetalCombinedSamplerSlot */
#include "mgl_binding_policy.h"       /* mglRenderTextureBindingStageForShader */
#include "mgl_byte_hash.h"
#include "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_encode.h"
#include "mgl_batch_replay.h"
#include "mgl_batch_issue.h"
#include "mgl_batch_mtl_encode.h"
#include <stdio.h>
#include <string.h>

static const uint32_t kMaxFragmentSamplerSlots = 16;

static int mglBatchReplayHasActiveEncoder(const MGLEncodeContext *e)
{ return e && mglRenderEncoderOwnerHasCurrent(e->render_encoder_owner) != 0; }
static uint64_t mglRendererSamplerSnapshotHash(const MGLSamplerSnapshotKey *key)
{ return mglHashBytesFNV1a(key, sizeof(*key)); }

typedef struct {
    void *r; VertexArray *vao; const MGLDrawCommand *cmd;
    GLMContext ctx; Program *prog;
} MGLDynVertexCtx;
static int mglDynVertexPlan(void *v, uint8_t bi, MGLBatchDynVertexStreamPlan *p)
{ MGLDynVertexCtx *c = v; return mgl_batch_replay_plan_dyn_vertex_streams(
      c->ctx, c->vao, c->prog, &c->cmd->dynamic_vertex_bindings[bi], p); }
static int mglDynVertexResolve(void *v, uint32_t attrib, int *slot)
{ MGLDynVertexCtx *c = v; int s = mglRendererResolveVertexAttributeBufferIndex(
      c->ctx, c->vao, attrib, __func__); if (slot) *slot = s; return s >= 0; }
static int mglDynVertexCanBind(void *v, const MGLBatchDynVertexStreamPlan *p, uint32_t stream)
{ MGLDynVertexCtx *c = v; return mgl_batch_replay_dyn_vertex_stream_can_bind_directly(
      c->prog, c->vao, p, stream); }
static int mglDynVertexEnsure(void *v, const MGLBatchDynVertexStreamPlan *p, void **mtl,
                              void **gl, uint64_t *dyn, uint64_t *len)
{
    MGLDynVertexCtx *c = v; Buffer *buf = p->buffer; if (!buf) return 0;
    if (buf->data.dirty_bits) {
        BufferMapList upload = {0}; upload.count = 1; upload.buffers[0].buf = buf;
        if (!mglRendererUpdateDirtyBaseBufferListPort(c->r, &upload)) return 0;
    }
    if (!buf->data.mtl_data) mglRendererBindMTLBufferPort(c->r, buf);
    if (!mgl_batch_replay_mtl_ptr_ok(buf->data.mtl_data)) return 0;
    void *mb = buf->data.mtl_data; MGLRenderBufferInfo info = {0};
    if (mglRenderGetBufferInfo(mb, &info) != 0) return 0;
    if (mtl) *mtl = mb; if (gl) *gl = buf;
    if (dyn) *dyn = (uint64_t)p->dynamic_offset; if (len) *len = info.length;
    return 1;
}
static uint64_t mglDynVertexBindOff(void *v, const MGLBatchDynVertexStreamPlan *p)
{ return (uint64_t)((MGLDynVertexCtx *)v)->vao->bindings[p->binding_index].offset; }

typedef struct {
    void *r; const MGLDrawCommand *cmd; GLMContext ctx;
} MGLDynUniformCtx;
static int mglDynUniformGather(void *v, uint64_t *lens, uint32_t count)
{
    MGLDynUniformCtx *c = v;
    for (uint32_t i = 0; i < count; i++) {
        BufferBaseTarget *slot = &c->ctx->active_state->buffer_base[_UNIFORM_BUFFER]
                                      .buffers[c->cmd->dynamic_uniform_bindings[i].binding_index];
        if (!slot->buf || !mgl_batch_replay_mtl_ptr_ok(slot->buf->data.mtl_data)) return 0;
        MGLRenderBufferInfo info = {0};
        if (mglRenderGetBufferInfo(slot->buf->data.mtl_data, &info) != 0) return 0;
        lens[i] = info.length;
    }
    return 1;
}
static int mglDynUniformResolve(void *v, const MGLBatchUniformBindOp *op, void **mtl, void **gl)
{
    MGLDynUniformCtx *c = v;
    BufferBaseTarget *slot =
        &c->ctx->active_state->buffer_base[_UNIFORM_BUFFER].buffers[op->binding_index];
    if (!slot->buf || !mgl_batch_replay_mtl_ptr_ok(slot->buf->data.mtl_data)) return 0;
    if (mtl) *mtl = slot->buf->data.mtl_data; if (gl) *gl = slot->buf; return 1;
}

typedef struct {
    void *r; const MGLDrawCommand *cmd; GLMContext ctx;
    MGLEncodeContext *enc; VertexArray dynamic_vao; VertexArray *draw_vao;
    VertexArray *base_vao; VertexArray *saved_vao; bool touched[TEXTURE_UNITS];
} MGLDynApplyCtx;

/* Defined below; the dyn-apply ops table wires them together. */
int mglBatchDynBindVertexDirect(void *renderer, VertexArray *vao,
                                const MGLDrawCommand *cmd, GLMContext glm_ctx,
                                const MGLEncodeContext *encCtx);
int mglBatchDynBindUniformDirect(void *renderer, const MGLDrawCommand *cmd,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encCtx);
int mglBatchDynBindSampledDirect(void *renderer, const bool *touched_units,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encCtx);

static void mglDynApplyRefresh(void *v)
{ MGLDynApplyCtx *c = v; if (c->enc)
      c->enc->render_encoder_owner = mglRendererCurrentRenderEncoderOwnerPort(c->r); }
static int mglDynApplyHasEnc(void *v)
{ return mglBatchReplayHasActiveEncoder(((MGLDynApplyCtx *)v)->enc) ? 1 : 0; }
static int mglDynApplyBuildVao(void *v)
{ MGLDynApplyCtx *c = v; if (!c->base_vao || c->base_vao->magic != MGL_VAO_MAGIC ||
      !mgl_batch_replay_build_dynamic_vertex_array(c->ctx, c->base_vao, c->cmd, &c->dynamic_vao))
      return 0; c->draw_vao = &c->dynamic_vao; return 1; }
static int mglDynApplyUbo(void *v)
{ MGLDynApplyCtx *c = v; return mgl_batch_replay_apply_uniform_range_overrides(c->ctx, c->cmd) ? 1 : 0; }
static int mglDynApplyTex(void *v)
{ MGLDynApplyCtx *c = v; memset(c->touched, 0, sizeof(c->touched));
  return mgl_batch_replay_apply_texture_overrides(c->ctx, c->cmd, c->touched, TEXTURE_UNITS) ? 1 : 0; }
static int mglDynApplyBindTexDirect(void *v)
{ MGLDynApplyCtx *c = v; return mglBatchDynBindSampledDirect(c->r, c->touched, c->ctx, c->enc) ? 1 : 0; }
static int mglDynApplyBindTexMapper(void *v)
{ MGLDynApplyCtx *c = v; return mglRendererBindTexturesToCurrentRenderEncoderPort(c->r, c->enc) ? 1 : 0; }
static int mglDynApplyRestoreTex(void *v)
{ return mglRendererRestoreRenderEncoderAfterTextureUploadPort(
      ((MGLDynApplyCtx *)v)->r, "dynamic-sampled-texture-bind") ? 1 : 0; }
static int mglDynApplyBindVertex(void *v)
{ MGLDynApplyCtx *c = v; return mglBatchDynBindVertexDirect(c->r, c->draw_vao, c->cmd,
      c->ctx, c->enc) ? 1 : 0; }
static int mglDynApplyBindUniform(void *v)
{ MGLDynApplyCtx *c = v; return mglBatchDynBindUniformDirect(c->r, c->cmd, c->ctx, c->enc) ? 1 : 0; }
static int mglDynApplyMapperFallback(void *v)
{
    MGLDynApplyCtx *c = v; c->saved_vao = c->ctx->active_state->vao;
    if (c->cmd->dynamic_vertex_binding_count > 0) c->ctx->active_state->vao = c->draw_vao;
    mglDynApplyRefresh(v);
    int ok = (mglRendererMapBuffersToMTLPort(c->r) &&
              mglRendererBindVertexBuffersToCurrentRenderEncoderPort(c->r, c->enc)) ? 1 : 0;
    mglDynApplyRefresh(v);
    if (ok && c->cmd->dynamic_uniform_binding_count > 0) {
        ok = mglRendererBindFragmentBuffersToCurrentRenderEncoderPort(c->r, c->enc) ? 1 : 0;
        mglDynApplyRefresh(v);
    }
    c->ctx->active_state->vao = c->saved_vao; return ok;
}

typedef struct {
    void *r; GLMContext ctx;
} MGLDynSampledCtx;
static int mglDynSampledResolve(void *v, const MGLBatchSampledTexCandidate *e,
                                const bool *touched, void **tex_out, uint32_t *stage_out,
                                int *needs_samp, void **samp_out, uint32_t *samp_slot)
{
    MGLDynSampledCtx *c = v; MGLShaderResource *resource = e->resource;
    GLuint unit = mglRendererTextureUnitForSampledResourcePort(
        c->r, resource, e->metal_slot, (int)e->stage);
    Texture *tex_obj = (unit < TEXTURE_UNITS && touched[unit])
        ? mglRendererTextureForSampledResourcePort(
              c->r, resource, e->metal_slot, (int)e->stage,
              e->lookup_type ? e->lookup_type : e->expected_type)
        : NULL;
    void *texture = (tex_obj && tex_obj->mtl_data)
        ? mglSampledTextureViewForBaseLevel(tex_obj, tex_obj->mtl_data) : NULL;
    MGLRenderTextureInfo info = {0};
    int info_ok = texture && mglRenderGetTextureInfo(texture, &info) == 0;
    Sampler *bound = (unit < TEXTURE_UNITS) ? c->ctx->active_state->texture_samplers[unit] : NULL;
    void *sampler = NULL;
    if (bound && !bound->dirty_bits && bound->mtl_data) sampler = bound->mtl_data;
    else if (tex_obj && tex_obj->params.mtl_data) sampler = tex_obj->params.mtl_data;
    MGLBatchSampledResolveGateIn gin = {
        .unit_ok = (unit < TEXTURE_UNITS && touched[unit]) ? 1 : 0,
        .has_tex = tex_obj ? 1 : 0, .has_mtl = (tex_obj && tex_obj->mtl_data) ? 1 : 0,
        .dirty = (tex_obj && tex_obj->dirty_bits) ? 1 : 0,
        .is_rt = (tex_obj && tex_obj->is_render_target) ? 1 : 0,
        .info_ok = info_ok, .texture_type = info.texture_type,
        .expected_type = e->expected_type,
        .format_compat = mglTexturePixelFormatCompatibleWithExpectedDataKind(
            info.pixel_format, (MGLTextureDataKind)e->expected_kind),
        .needs_combined_sampler = e->needs_combined_sampler ? 1 : 0,
        .has_sampler_mtl = sampler ? 1 : 0,
    };
    int gate = mgl_batch_replay_sampled_resolve_gate(&gin);
    if (gate <= 0) return gate;
    if (stage_out) *stage_out = mglRenderTextureBindingStageForShader((int)e->stage);
    if (tex_out) *tex_out = texture;
    if (gate == 1) { if (needs_samp) *needs_samp = 0; return 1; }
    if (needs_samp) *needs_samp = 1;
    if (samp_out) *samp_out = sampler;
    if (samp_slot) *samp_slot = resource ? mglMetalCombinedSamplerSlot(resource) : e->metal_slot;
    return 1;
}

/* Sampler state for a snapshot key (unretained; the backend cache owns it). */
static void *mglBatchSamplerStateForSnapshotKey(void *renderer,
                                                const MGLSamplerSnapshotKey *key)
{
    if (!key) return NULL;
    return mglRendererSamplerStateForSnapshotKeyPort(renderer, key);
}

typedef struct {
    void *r; GLMContext ctx; const MGLSamplerSnapshotSet *set;
    MGLCommandBuffer *cb;
} MGLSampSnapCtx;
static int mglSampResolve(void *v, uint32_t i, MGLBatchResolvedSamplerBind *out)
{
    MGLSampSnapCtx *c = v; const MGLSamplerSnapshotEntry *entry = &c->set->entries[i];
    void *sampler = (entry->key_index == MGL_FALLBACK_SAMPLER_KEY_INDEX)
                        ? mglRendererFallbackSamplerStatePort(c->r)
                        : (entry->key_index < c->cb->sampler_snapshot_key_count
                               ? mglBatchSamplerStateForSnapshotKey(
                                     c->r, &c->cb->sampler_snapshot_keys[entry->key_index])
                               : NULL);
    if (!sampler) return 0;
    out->sampler = sampler;
    out->metal_slot = entry->metal_slot;
    out->shader_stage = (int)entry->stage;
    return 1;
}
static void mglSampAfter(void *v, uint32_t i, const MGLBatchResolvedSamplerBind *bind)
{
    (void)bind; MGLSampSnapCtx *c = v; const MGLSamplerSnapshotEntry *entry = &c->set->entries[i];
    if (!mglMipDiagEnabled() || entry->stage != _FRAGMENT_SHADER ||
        entry->key_index == MGL_FALLBACK_SAMPLER_KEY_INDEX)
        return;
    const MGLSamplerSnapshotKey *key = &c->cb->sampler_snapshot_keys[entry->key_index];
    static uint64_t s_snapshotState[16];
    if (!mglMipDiagStateChanged(&s_snapshotState[entry->metal_slot],
                                mglRendererSamplerSnapshotHash(key)))
        return;
    /* Same sink and prefix as the NSLog this replaced; MGL_MIP_DIAG is
     * deliberately independent of the trace log, so this cannot go through
     * mglTraceLog. */
    fprintf(stderr,
            "MGL MIP_DIAG snapshot slot=%u unit=%u target=0x%x minFilter=0x%x "
            "magFilter=0x%x minLod=%.1f maxLod=%.1f aniso=%.1f\n",
            (unsigned)entry->metal_slot, (unsigned)entry->texture_unit, (unsigned)key->target,
            (unsigned)key->min_filter, (unsigned)key->mag_filter, (double)key->min_lod,
            (double)key->max_lod, (double)key->max_anisotropy);
}

typedef struct {
    void *r; MGLDrawBatch *batch; GLMContext ctx;
} MGLSimpleReplayCtx;
static int mglSimpleResolve(void *v, uint32_t i, uint32_t gl_itype, void **mtl,
                            uint64_t *ioff, uint32_t *mtype)
{
    MGLSimpleReplayCtx *c = v; MGLDrawCommand *cmd = &c->batch->commands[i];
    Buffer *glBuf = NULL; void *idxBuf = NULL;
    if (!mglRendererResolveElementBufferPort(c->r, cmd, "cppBatchReplay", c->ctx,
                                             &glBuf, &idxBuf))
        return 0;
    size_t off = ioff ? (size_t)*ioff : (size_t)cmd->indexBufferOffset;
    uint64_t itype = mglRenderMTLIndexTypeForGLType((uint32_t)gl_itype);
    void *prepared = mglPreparedElementIndexBuffer(mglDrawHostDevice(c->r), glBuf, idxBuf,
                                                   (GLenum)gl_itype, &off, &itype);
    if (ioff) *ioff = (uint64_t)off; if (mtype) *mtype = (uint32_t)itype;
    if (mtl) *mtl = prepared; return prepared ? 1 : 0;
}


int mglBatchDynBindVertexDirect(void *renderer, VertexArray *vao,
                                const MGLDrawCommand *cmd, GLMContext glm_ctx,
                                const MGLEncodeContext *encCtx)
{
    if (!vao || !cmd || !encCtx) return 0;
    MGLDynVertexCtx c = {.r = renderer, .vao = vao, .cmd = cmd, .ctx = glm_ctx,
                         .prog = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER)};
    MGLBatchDynVertexBindOps ops = {
        .ctx = &c, .binding_count = cmd->dynamic_vertex_binding_count,
        .max_metal_slots = (int)kMGLMaxMetalVertexBufferCount,
        .binding_state_owner = mglRendererBindingStateOwnerPort(renderer),
        .render_encoder_owner = encCtx->render_encoder_owner,
        .plan_binding = mglDynVertexPlan, .resolve_slot = mglDynVertexResolve,
        .stream_can_bind = mglDynVertexCanBind, .ensure_mtl = mglDynVertexEnsure,
        .vao_binding_offset = mglDynVertexBindOff,
    };
    return mgl_batch_mtl_bind_dyn_vertex(&ops) ? 1 : 0;
}

int mglBatchDynBindUniformDirect(void *renderer, const MGLDrawCommand *cmd,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encCtx)
{
    if (!cmd || !glm_ctx || !encCtx) return 0;
    MGLDynUniformCtx c = {.r = renderer, .cmd = cmd, .ctx = glm_ctx};
    MGLBatchDynUniformBindOps ops = {
        .ctx = &c, .binding_state_owner = mglRendererBindingStateOwnerPort(renderer),
        .render_encoder_owner = encCtx->render_encoder_owner,
        .min_stage_binding_size = (uint64_t)kMGLMinimumStageBindingSize,
        .max_buffer_slots = (uint32_t)kMGLMaxBufferSlots,
        .gather_lengths = mglDynUniformGather, .cmd = cmd, .glm_ctx = glm_ctx,
        .resolve_op = mglDynUniformResolve,
    };
    return mgl_batch_mtl_bind_dyn_uniforms(&ops) ? 1 : 0;
}

int mglBatchDynBindSampledDirect(void *renderer, const bool *touched_units,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encCtx)
{
    if (!touched_units || !glm_ctx || !mglBatchReplayHasActiveEncoder(encCtx)) return 0;
    MGLDynSampledCtx c = {.r = renderer, .ctx = glm_ctx};
    MGLBatchDynSampledBindOps ops = {
        .ctx = &c, .binding_state_owner = mglRendererBindingStateOwnerPort(renderer),
        .render_encoder_owner = encCtx->render_encoder_owner,
        .max_sampler_slots = kMaxFragmentSamplerSlots, .glm_ctx = glm_ctx,
        .resolve_candidate = mglDynSampledResolve, .touched_units = touched_units,
    };
    return mgl_batch_mtl_bind_dyn_sampled(&ops) ? 1 : 0;
}

int mglBatchApplySamplerSnapshot(void *renderer, const MGLDrawCommand *cmd,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encCtx)
{
    if (!cmd || !glm_ctx) return 0;
    if (cmd->sampler_snapshot_id == MGL_INVALID_SAMPLER_SNAPSHOT_ID) return 1;
    if (!mglBatchReplayHasActiveEncoder(encCtx)) return 0;
    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cmd->sampler_snapshot_id >= cb->sampler_snapshot_set_count) return 0;
    const MGLSamplerSnapshotSet *set = &cb->sampler_snapshot_sets[cmd->sampler_snapshot_id];
    if (set->count > MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES) return 0;
    MGLSampSnapCtx c = {.r = renderer, .ctx = glm_ctx, .set = set, .cb = cb};
    MGLBatchSamplerSnapshotApplyOps ops = {
        .ctx = &c, .binding_state_owner = mglRendererBindingStateOwnerPort(renderer),
        .render_encoder_owner = encCtx->render_encoder_owner,
        .entry_count = set->count, .max_sampler_slots = 16u,
        .resolve_entry = mglSampResolve, .after_resolved = mglSampAfter,
    };
    return mgl_batch_mtl_apply_sampler_snapshot(&ops) ? 1 : 0;
}

int mglBatchApplyDynamicBindings(void *renderer, const MGLDrawCommand *cmd,
                                 GLMContext glm_ctx, MGLEncodeContext *encCtx)
{
    if (!cmd) return 1;
    if (!glm_ctx || !encCtx) return 0;
    encCtx->render_encoder_owner = mglRendererCurrentRenderEncoderOwnerPort(renderer);
    MGLDynApplyCtx c = {.r = renderer, .cmd = cmd, .ctx = glm_ctx, .enc = encCtx,
                        .base_vao = glm_ctx->active_state->vao,
                        .draw_vao = glm_ctx->active_state->vao};
    MGLBatchDynApplyOps ops = {
        .ctx = &c, .refresh_owner = mglDynApplyRefresh, .has_encoder = mglDynApplyHasEnc,
        .build_dyn_vao = mglDynApplyBuildVao, .apply_ubo = mglDynApplyUbo,
        .apply_tex = mglDynApplyTex, .bind_tex_direct = mglDynApplyBindTexDirect,
        .bind_tex_mapper = mglDynApplyBindTexMapper,
        .restore_after_tex_upload = mglDynApplyRestoreTex,
        .bind_vertex_direct = mglDynApplyBindVertex,
        .bind_uniform_direct = mglDynApplyBindUniform,
        .mapper_fallback = mglDynApplyMapperFallback,
    };
    return mgl_batch_issue_apply_dyn_bindings(cmd->dynamic_vertex_binding_count,
                                              cmd->dynamic_uniform_binding_count,
                                              cmd->dynamic_texture_binding_count, &ops)
               ? 1 : 0;
}

int mglBatchTryReplaySimpleBatch(void *renderer, MGLDrawBatch *batch,
                                 GLMContext glm_ctx,
                                 const MGLEncodeContext *encCtx)
{
    if (!batch || batch->command_count == 0u) return 0;
    Program *batchProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    const GLenum batchMode = batch->commands[0].mode;
    if (!mgl_batch_replay_simple_eligible(
            batch, MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS,
            mglBatchReplayHasActiveEncoder(encCtx) ? 1 : 0,
            (batchProgram && batchProgram->uses_cull_distance) ? 1 : 0,
            glm_ctx->active_state->caps.primitive_restart ? 1 : 0,
            mglPolygonModePointForDrawMode(glm_ctx, batchMode) ? 1 : 0,
            mglRenderDrawModeNeedsEmulate((uint32_t)batchMode) ? 1 : 0))
        return 0;
    MGLSimpleReplayCtx ctx = {renderer, batch, glm_ctx};
    MGLBatchSimpleReplayOps ops = {.ctx = &ctx, .resolve_index = mglSimpleResolve};
    return mgl_batch_mtl_issue_simple_replay(batch, encCtx->render_encoder_owner, &ops) ? 1 : 0;
}
