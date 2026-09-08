/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * A3: Batch replay diagnostic traces (cluster metric). Same (Batch) category.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_render.h"
#include "mgl_batch_rt_mark.h"

#include <string.h>

static void *mglBatchEncoderTraceToken(void *owner) { return owner; }

static MGLRenderTextureInfo mglBatchTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture)
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    return info;
}

static void mglBatchFsFlags(MGLFragmentTextureTraceBinding *b0,
                            MGLFragmentTextureTraceBinding *b1,
                            MGLFragmentTextureTraceBinding *b2,
                            MGLFragmentTextureTraceBinding *b3,
                            int *has_rt, int *used_copy)
{
    const MGLBatchTraceFsSlot slots[4] = {
        {b0->rt_write_version, b0->used_sampled_copy ? 1u : 0u},
        {b1->rt_write_version, b1->used_sampled_copy ? 1u : 0u},
        {b2->rt_write_version, b2->used_sampled_copy ? 1u : 0u},
        {b3->rt_write_version, b3->used_sampled_copy ? 1u : 0u},
    };
    mgl_batch_trace_fs_slot_flags(slots, 4, has_rt, used_copy);
}

static void mglBatchFillStatePod(GLMContext glm_ctx, MGLBatchTraceStatePod *s)
{
    memset(s, 0, sizeof(*s));
    GLMState *st = MGL_STATE(glm_ctx);
    for (int i = 0; i < 4; i++) s->viewport[i] = (int32_t)st->viewport[i];
    s->scissor_test = st->caps.scissor_test ? 1 : 0;
    for (int i = 0; i < 4; i++) s->scissor[i] = (int32_t)st->var.scissor_box[i];
    s->draw_buf = (uint32_t)st->draw_buffer;
    s->read_buf = (uint32_t)st->read_buffer;
    for (int i = 0; i < 4; i++)
        s->color_mask[i] = st->var.color_writemask[0][i] ? 1 : 0;
    s->depth_test = st->caps.depth_test ? 1 : 0;
    s->depth_write = st->var.depth_writemask ? 1 : 0;
    s->depth_func = (uint32_t)st->var.depth_func;
    s->depth_clear = (double)st->var.depth_clear_value;
    s->blend = st->caps.blend ? 1 : 0;
    s->cull = st->caps.cull_face ? 1 : 0;
    s->cull_face = (uint32_t)st->var.cull_face_mode;
    s->front_face = (uint32_t)st->var.front_face;
    s->dirty = (uint32_t)st->dirty_bits;
}

static GLuint mglBatchSafeName(void *obj, size_t sz, GLuint name)
{
    if (obj && mglRendererObjectPointerLikelyValid(obj) &&
        mglPointerRangeIsReadable(obj, sz))
        return name;
    return 0u;
}

static MGLBatchTraceTexSlotView mglBatchTexSlotView(
    const MGLFragmentTextureTraceBinding *b)
{
    return (MGLBatchTraceTexSlotView){
        .gl_texture_name = b->gl_texture_name,
        .sampler_unit = b->sampler_unit,
        .program_name = b->program_name,
        .mtl_texture_ptr = b->mtl_texture_ptr,
        .direct_mtl_texture_ptr = b->direct_mtl_texture_ptr,
        .sampled_copy_ptr = b->sampled_copy_ptr,
        .used_sampled_copy = b->used_sampled_copy ? 1u : 0u,
        .used_fallback = b->used_fallback ? 1u : 0u,
        .rt_write_version = b->rt_write_version,
        .sampled_write_version = b->sampled_write_version,
        .width = b->width,
        .height = b->height,
        .pixel_format = b->pixel_format,
        .texture_type = b->texture_type,
    };
}

@implementation MGLRenderer (Batch)

- (void)traceReplayBatch:(MGLDrawBatch *)batch
                 context:(GLMContext)glm_ctx
                  flushId:(uint64_t)flushId
               batchIndex:(uint32_t)batchIndex
                    phase:(const char *)phase
{
    if (!batch || !glm_ctx || !mglTraceLogIsEnabled()) return;
    Program *drawProgram = mglTraceResolveDrawProgram(glm_ctx);
    MGLFragmentTextureTraceBinding *earlyFs =
        &_resourceFallback.fragmentTextureTraceBindings[0];
    int earlyFsSlotHasRT = 0, earlyFsSlotUsedCopy = 0;
    mglBatchFsFlags(earlyFs, earlyFs + 1, earlyFs + 2, earlyFs + 3,
                    &earlyFsSlotHasRT, &earlyFsSlotUsedCopy);
    if (!mglTraceShouldLogReplay(glm_ctx, drawProgram) && !earlyFsSlotHasRT &&
        !earlyFsSlotUsedCopy)
        return;

    VertexArray *vao = mglRendererGetValidatedVAO(glm_ctx, "replay.batch.trace");
    Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
    GLuint fboName = mglBatchSafeName(fbo, sizeof(*fbo), fbo ? fbo->name : 0u);
    id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    GLMState *snapshot =
        batch->state_snapshot ? (GLMState *)batch->state_snapshot : NULL;
    GLuint snapshotFBOName = 0u;
    if (snapshot && snapshot->framebuffer)
        snapshotFBOName = mglBatchSafeName(snapshot->framebuffer,
                                           sizeof(*snapshot->framebuffer),
                                           snapshot->framebuffer->name);
    Program *vertexProgram =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    Program *fragmentProgram =
        mglResolveProgramForStageFromState(glm_ctx, _FRAGMENT_SHADER);

    MGLBatchTraceBatchView v;
    memset(&v, 0, sizeof(v));
    v.phase = phase;
    v.flush_id = flushId;
    v.batch_index = batchIndex;
    v.command_count = batch->command_count;
    v.stream_merged = batch->stream_merged ? 1u : 0u;
    v.mdi_compatible = batch->mdi_compatible ? 1u : 0u;
    v.uses_elements = batch->uses_elements ? 1u : 0u;
    v.key_program = batch->key.program_name;
    v.key_pipeline = batch->key.program_pipeline_name;
    v.key_vs = batch->key.vertex_program_name;
    v.key_fs = batch->key.fragment_program_name;
    v.key_fbo = batch->key.fbo_name;
    v.key_vao = batch->key.vao_name;
    v.key_prim = batch->key.primitive_type;
    v.snap_program = snapshot ? snapshot->program_name : 0u;
    v.snap_pipeline = snapshot ? snapshot->var.program_pipeline_binding : 0u;
    v.snap_current = snapshot ? snapshot->program_name : 0u;
    v.snap_fbo = snapshotFBOName;
    v.snap_vao = snapshot ? snapshot->vao : NULL;
    v.restored_current = mglCurrentRenderProgramKey(glm_ctx);
    v.restored_program = MGL_STATE(glm_ctx)->program_name;
    v.restored_pipeline = MGL_STATE(glm_ctx)->var.program_pipeline_binding;
    v.restored_vs = vertexProgram ? vertexProgram->name : 0u;
    v.restored_fs = fragmentProgram ? fragmentProgram->name : 0u;
    v.restored_fbo = fboName;
    v.restored_vao = vao;
    v.enabled_attribs = vao ? (uint32_t)vao->enabled_attribs : 0u;
    MGLBatchTraceStatePod state;
    mglBatchFillStatePod(glm_ctx, &state);
    mgl_batch_trace_copy_state_to_batch(&v, &state);
    v.encoder = mglBatchEncoderTraceToken(
        _renderPassManager.state->currentRenderEncoderOwner);
    v.pipeline_state = _pipelineCache.state->pipelineState;
    v.rp_fbo = _renderPassManager.state->renderPassFramebufferName;
    v.rp_color = (__bridge void *)rpColor0;
    v.rp_depth = (__bridge void *)rpDepth;

    char line[2048];
    if (mgl_batch_trace_format_batch_line(line, sizeof(line), &v) > 0)
        mglTraceLog("%s", line);
}

- (void)traceReplayCommand:(MGLDrawBatch *)batch
                   command:(MGLDrawCommand *)cmd
                   context:(GLMContext)glm_ctx
                   flushId:(uint64_t)flushId
                batchIndex:(uint32_t)batchIndex
              commandIndex:(uint32_t)commandIndex
                     phase:(const char *)phase
                    reason:(const char *)reason
{
    if (!batch || !cmd || !glm_ctx || !mglTraceLogIsEnabled()) return;
    MGLFragmentTextureTraceBinding *fs0 =
        &_resourceFallback.fragmentTextureTraceBindings[0];
    int earlyFsSlotHasRT = 0, earlyFsSlotUsedCopy = 0;
    mglBatchFsFlags(fs0, fs0 + 1, fs0 + 2, fs0 + 3, &earlyFsSlotHasRT,
                    &earlyFsSlotUsedCopy);
    Program *drawProgram = mglTraceResolveDrawProgram(glm_ctx);
    if (!mglTraceShouldLogReplay(glm_ctx, drawProgram) && !earlyFsSlotHasRT &&
        !earlyFsSlotUsedCopy)
        return;

    Buffer *ebo =
        mglDrawCommandUsesElements(cmd) ? mglDrawCommandElementBuffer(glm_ctx, cmd)
                                        : NULL;
    GLuint eboName =
        mglBatchSafeName(ebo, sizeof(*ebo), ebo ? ebo->name : 0u);
    Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
    GLuint fboName = mglBatchSafeName(fbo, sizeof(*fbo), fbo ? fbo->name : 0u);
    id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    MGLRenderTextureInfo rpColorInfo = mglBatchTextureInfo(rpColor0);
    MGLRenderTextureInfo rpDepthInfo = mglBatchTextureInfo(rpDepth);
    Program *vertexProgram =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    Program *fragmentProgram =
        mglResolveProgramForStageFromState(glm_ctx, _FRAGMENT_SHADER);
    FBOAttachment *color0Attachment =
        (fbo && (fbo->color_attachment_bitfield & 1u)) ? &fbo->color_attachments[0]
                                                       : NULL;
    Texture *color0Texture =
        mglTraceFramebufferAttachmentTexture(glm_ctx, color0Attachment);
    Texture *depthTexture =
        fbo ? mglTraceFramebufferAttachmentTexture(glm_ctx, &fbo->depth) : NULL;
    GLuint cEver = 0u, cFull = 0u, cSource = 0u;
    GLuint dEver = 0u, dFull = 0u, dSource = 0u;
    mglTraceTextureLevelSummary(color0Texture,
                                color0Attachment ? color0Attachment->level : 0u,
                                &cEver, &cFull, &cSource);
    mglTraceTextureLevelSummary(depthTexture, fbo ? fbo->depth.level : 0u, &dEver,
                                &dFull, &dSource);
    int fsSlotHasRT = 0, fsSlotUsedCopy = 0;
    mglBatchFsFlags(fs0, fs0 + 1, fs0 + 2, fs0 + 3, &fsSlotHasRT, &fsSlotUsedCopy);

    MGLBatchTraceCmdView v;
    memset(&v, 0, sizeof(v));
    v.phase = phase;
    v.reason = reason;
    v.type_name = mglDrawCommandTypeName(cmd->type);
    v.flush_id = flushId;
    v.batch_index = batchIndex;
    v.command_index = commandIndex;
    v.program = mglCurrentRenderProgramKey(glm_ctx);
    v.vs = vertexProgram ? vertexProgram->name : 0u;
    v.fs = fragmentProgram ? fragmentProgram->name : 0u;
    v.mode = cmd->mode;
    v.count = cmd->count;
    v.first = cmd->first;
    v.index_type = cmd->indexType;
    v.index_offset = (uint32_t)cmd->indexBufferOffset;
    v.instances = cmd->instanceCount;
    v.base_vertex = cmd->baseVertex;
    v.base_instance = cmd->baseInstance;
    v.ebo_name = eboName;
    v.ebo = ebo;
    v.encoder = mglBatchEncoderTraceToken(
        _renderPassManager.state->currentRenderEncoderOwner);
    v.pipeline_state = _pipelineCache.state->pipelineState;
    v.fbo_name = fboName;
    v.rp_fbo = _renderPassManager.state->renderPassFramebufferName;
    v.rp_color = (__bridge void *)rpColor0;
    v.rp_depth = (__bridge void *)rpDepth;
    v.rp_color_w = rpColorInfo.width;
    v.rp_color_h = rpColorInfo.height;
    v.rp_depth_w = rpDepthInfo.width;
    v.rp_depth_h = rpDepthInfo.height;
    v.rp_la = mglLoadActionName((uint32_t)mglRenderPassLoadActionForTrace(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, 0u));
    v.rp_sa = mglStoreActionName((uint32_t)mglRenderPassStoreActionForTrace(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, 0u));
    v.depth_la = mglLoadActionName((uint32_t)mglRenderPassLoadActionForTrace(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, 0u));
    v.depth_sa = mglStoreActionName((uint32_t)mglRenderPassStoreActionForTrace(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, 0u));
    v.color0_tex = color0Attachment ? color0Attachment->texture : 0u;
    v.color0_target = color0Attachment ? color0Attachment->textarget : 0u;
    v.color0_level = color0Attachment ? color0Attachment->level : 0u;
    v.color0_ptr = color0Texture;
    v.color0_w = color0Texture ? color0Texture->width : 0u;
    v.color0_h = color0Texture ? color0Texture->height : 0u;
    v.color0_mtl = color0Texture ? color0Texture->mtl_data : NULL;
    v.color0_ever = cEver;
    v.color0_full = cFull;
    v.color0_source = cSource;
    v.color0_rt_ver =
        color0Texture ? color0Texture->mtl_render_target_write_version : 0u;
    v.color0_sampled_ver =
        color0Texture ? color0Texture->mtl_gl_sampled_write_version : 0u;
    v.depth_tex = fbo ? fbo->depth.texture : 0u;
    v.depth_target = fbo ? fbo->depth.textarget : 0u;
    v.depth_level = fbo ? fbo->depth.level : 0u;
    v.depth_ptr = depthTexture;
    v.depth_w = depthTexture ? depthTexture->width : 0u;
    v.depth_h = depthTexture ? depthTexture->height : 0u;
    v.depth_mtl = depthTexture ? depthTexture->mtl_data : NULL;
    v.depth_ever = dEver;
    v.depth_full = dFull;
    v.depth_source = dSource;
    v.depth_rt_ver =
        depthTexture ? depthTexture->mtl_render_target_write_version : 0u;
    v.depth_sampled_ver =
        depthTexture ? depthTexture->mtl_gl_sampled_write_version : 0u;
    Texture *u0a = MGL_STATE(glm_ctx)->active_textures[0];
    Texture *u0t = MGL_STATE(glm_ctx)->texture_units[0].textures[_TEXTURE_2D];
    Texture *u1a = MGL_STATE(glm_ctx)->active_textures[1];
    Texture *u1t = MGL_STATE(glm_ctx)->texture_units[1].textures[_TEXTURE_2D];
    Texture *u2a = MGL_STATE(glm_ctx)->active_textures[2];
    Texture *u2t = MGL_STATE(glm_ctx)->texture_units[2].textures[_TEXTURE_2D];
    v.u0_active = u0a ? u0a->name : 0u;
    v.u0_tex2d = u0t ? u0t->name : 0u;
    v.u1_active = u1a ? u1a->name : 0u;
    v.u1_tex2d = u1t ? u1t->name : 0u;
    v.u2_active = u2a ? u2a->name : 0u;
    v.u2_tex2d = u2t ? u2t->name : 0u;
    MGLBatchTraceStatePod state;
    mglBatchFillStatePod(glm_ctx, &state);
    mgl_batch_trace_copy_state_to_cmd(&v, &state);

    char line[3072];
    if (mgl_batch_trace_format_cmd_line(line, sizeof(line), &v) > 0)
        mglTraceLog("%s", line);

    BOOL submitPhase = phase && strcmp(phase, "SUBMIT") == 0;
    if (submitPhase &&
        (fsSlotHasRT || fsSlotUsedCopy ||
         mglProgramNeedsBindingTrace(fragmentProgram))) {
        MGLBatchTraceTexSlotView slots[4] = {
            mglBatchTexSlotView(fs0),
            mglBatchTexSlotView(fs0 + 1),
            mglBatchTexSlotView(fs0 + 2),
            mglBatchTexSlotView(fs0 + 3),
        };
        char texline[2048];
        if (mgl_batch_trace_format_texslots_line(
                texline, sizeof(texline), flushId, batchIndex, commandIndex,
                mglCurrentRenderProgramKey(glm_ctx),
                vertexProgram ? vertexProgram->name : 0u,
                fragmentProgram ? fragmentProgram->name : 0u,
                _pipelineCache.state->pipelineProgramName, slots) > 0)
            mglTraceLog("%s", texline);
        if ((fsSlotHasRT || fsSlotUsedCopy) && fragmentProgram) {
            mglWriteProgramMSLDump(
                fragmentProgram,
                [NSString stringWithFormat:@"texslot-submit-fs-%u-flush-%llu-cmd-%u",
                                           (unsigned)fragmentProgram->name,
                                           (unsigned long long)flushId,
                                           (unsigned)commandIndex]);
        } else if ((fsSlotHasRT || fsSlotUsedCopy) && drawProgram) {
            mglWriteProgramMSLDump(
                drawProgram,
                [NSString
                    stringWithFormat:@"texslot-submit-program-%u-flush-%llu-cmd-%u",
                                     (unsigned)drawProgram->name,
                                     (unsigned long long)flushId,
                                     (unsigned)commandIndex]);
        }
    }

    if (submitPhase && ebo) {
        Program *attribProgram = vertexProgram ? vertexProgram : drawProgram;
        mglTraceReplayCommandVertexAttribSamples(
            glm_ctx, attribProgram, cmd, ebo, flushId, batchIndex, commandIndex,
            fsSlotHasRT || fsSlotUsedCopy);
    }
}

@end
