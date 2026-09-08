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

static void *mglBatchEncoderTraceToken(void *owner)
{
    return owner;
}

static MGLRenderTextureInfo mglBatchTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
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

@implementation MGLRenderer (Batch)

- (void)traceReplayBatch:(MGLDrawBatch *)batch
                 context:(GLMContext)glm_ctx
                  flushId:(uint64_t)flushId
               batchIndex:(uint32_t)batchIndex
                    phase:(const char *)phase
{
    if (!batch || !glm_ctx) {
        return;
    }

    if (!mglTraceLogIsEnabled()) {
        return;
    }

    Program *drawProgram = mglTraceResolveDrawProgram(glm_ctx);
    MGLFragmentTextureTraceBinding *earlyFs =
        &_resourceFallback.fragmentTextureTraceBindings[0];
    int earlyFsSlotHasRT = 0, earlyFsSlotUsedCopy = 0;
    mglBatchFsFlags(earlyFs, earlyFs + 1, earlyFs + 2, earlyFs + 3,
                    &earlyFsSlotHasRT, &earlyFsSlotUsedCopy);
    if (!mglTraceShouldLogReplay(glm_ctx, drawProgram) && !earlyFsSlotHasRT &&
        !earlyFsSlotUsedCopy) {
        return;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(glm_ctx, "replay.batch.trace");
    Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
    GLuint fboName = 0u;
    if (fbo &&
        mglRendererObjectPointerLikelyValid(fbo) &&
        mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        fboName = fbo->name;
    }
    id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    GLMState *snapshot = batch->state_snapshot ? (GLMState *)batch->state_snapshot : NULL;
    GLuint snapshotFBOName = 0u;
    if (snapshot &&
        snapshot->framebuffer &&
        mglRendererObjectPointerLikelyValid(snapshot->framebuffer) &&
        mglPointerRangeIsReadable(snapshot->framebuffer, sizeof(*snapshot->framebuffer))) {
        snapshotFBOName = snapshot->framebuffer->name;
    }
    Program *vertexProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    Program *fragmentProgram = mglResolveProgramForStageFromState(glm_ctx, _FRAGMENT_SHADER);
    GLuint currentProgramKey = mglCurrentRenderProgramKey(glm_ctx);

    mglTraceLog("REPLAY_BATCH_%s flush=%llu batch=%u commands=%u stream=%d mdiCompat=%d usesElements=%d "
                "key(program=%u pipeline=%u vs=%u fs=%u fbo=%u vao=%u prim=%u) "
                "snapshot(program=%u pipeline=%u current=%u fbo=%u vao=%p) "
                "restored(program=%u current=%u pipeline=%u vs=%u fs=%u fbo=%u vao=%p enabled=0x%x) "
                "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
                "drawBuf=0x%x readBuf=0x%x colorMask=%d%d%d%d depth(test=%d write=%d func=0x%x) "
                "blend=%d cull=%d cullFace=0x%x frontFace=0x%x dirty=0x%x encoder=%p pipelineState=%p rpFbo=%u rpColor=%p rpDepth=%p",
                phase ? phase : "STATE",
                (unsigned long long)flushId,
                (unsigned)batchIndex,
                (unsigned)batch->command_count,
                batch->stream_merged ? 1 : 0,
                batch->mdi_compatible ? 1 : 0,
                batch->uses_elements ? 1 : 0,
                (unsigned)batch->key.program_name,
                (unsigned)batch->key.program_pipeline_name,
                (unsigned)batch->key.vertex_program_name,
                (unsigned)batch->key.fragment_program_name,
                (unsigned)batch->key.fbo_name,
                (unsigned)batch->key.vao_name,
                (unsigned)batch->key.primitive_type,
                snapshot ? (unsigned)snapshot->program_name : 0u,
                snapshot ? (unsigned)snapshot->var.program_pipeline_binding : 0u,
                snapshot ? (unsigned)snapshot->program_name : 0u,
                (unsigned)snapshotFBOName,
                snapshot ? snapshot->vao : NULL,
                (unsigned)currentProgramKey,
                (unsigned)MGL_STATE(glm_ctx)->program_name,
                (unsigned)MGL_STATE(glm_ctx)->var.program_pipeline_binding,
                vertexProgram ? (unsigned)vertexProgram->name : 0u,
                fragmentProgram ? (unsigned)fragmentProgram->name : 0u,
                (unsigned)fboName,
                vao,
                vao ? (unsigned)vao->enabled_attribs : 0u,
                (int)MGL_STATE(glm_ctx)->viewport[0],
                (int)MGL_STATE(glm_ctx)->viewport[1],
                (int)MGL_STATE(glm_ctx)->viewport[2],
                (int)MGL_STATE(glm_ctx)->viewport[3],
                MGL_STATE(glm_ctx)->caps.scissor_test ? 1 : 0,
                (int)MGL_STATE(glm_ctx)->var.scissor_box[0],
                (int)MGL_STATE(glm_ctx)->var.scissor_box[1],
                (int)MGL_STATE(glm_ctx)->var.scissor_box[2],
                (int)MGL_STATE(glm_ctx)->var.scissor_box[3],
                (unsigned)MGL_STATE(glm_ctx)->draw_buffer,
                (unsigned)MGL_STATE(glm_ctx)->read_buffer,
                MGL_STATE(glm_ctx)->var.color_writemask[0][0] ? 1 : 0,
                MGL_STATE(glm_ctx)->var.color_writemask[0][1] ? 1 : 0,
                MGL_STATE(glm_ctx)->var.color_writemask[0][2] ? 1 : 0,
                MGL_STATE(glm_ctx)->var.color_writemask[0][3] ? 1 : 0,
                MGL_STATE(glm_ctx)->caps.depth_test ? 1 : 0,
                MGL_STATE(glm_ctx)->var.depth_writemask ? 1 : 0,
                (unsigned)MGL_STATE(glm_ctx)->var.depth_func,
                MGL_STATE(glm_ctx)->caps.blend ? 1 : 0,
                MGL_STATE(glm_ctx)->caps.cull_face ? 1 : 0,
                (unsigned)MGL_STATE(glm_ctx)->var.cull_face_mode,
                (unsigned)MGL_STATE(glm_ctx)->var.front_face,
                (unsigned)MGL_STATE(glm_ctx)->dirty_bits,
                mglBatchEncoderTraceToken(_renderPassManager.state->currentRenderEncoderOwner),
                _pipelineCache.state->pipelineState,
                (unsigned)_renderPassManager.state->renderPassFramebufferName,
                rpColor0,
                rpDepth);
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
    if (!batch || !cmd || !glm_ctx) {
        return;
    }

    if (!mglTraceLogIsEnabled()) {
        return;
    }

    MGLFragmentTextureTraceBinding *fs0 =
        &_resourceFallback.fragmentTextureTraceBindings[0];
    MGLFragmentTextureTraceBinding *fs1 = fs0 + 1;
    MGLFragmentTextureTraceBinding *fs2 = fs0 + 2;
    MGLFragmentTextureTraceBinding *fs3 = fs0 + 3;
    int earlyFsSlotHasRT = 0, earlyFsSlotUsedCopy = 0;
    mglBatchFsFlags(fs0, fs1, fs2, fs3, &earlyFsSlotHasRT, &earlyFsSlotUsedCopy);
    Program *drawProgram = mglTraceResolveDrawProgram(glm_ctx);
    if (!mglTraceShouldLogReplay(glm_ctx, drawProgram) && !earlyFsSlotHasRT &&
        !earlyFsSlotUsedCopy) {
        return;
    }

    Buffer *ebo = NULL;
    if (mglDrawCommandUsesElements(cmd)) {
        ebo = mglDrawCommandElementBuffer(glm_ctx, cmd);
    }
    GLuint eboName = 0u;
    if (ebo &&
        mglRendererObjectPointerLikelyValid(ebo) &&
        mglPointerRangeIsReadable(ebo, sizeof(*ebo))) {
        eboName = ebo->name;
    }
    Framebuffer *fbo = MGL_STATE(glm_ctx)->framebuffer;
    GLuint fboName = 0u;
    if (fbo &&
        mglRendererObjectPointerLikelyValid(fbo) &&
        mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        fboName = fbo->name;
    }
    id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    MGLRenderTextureInfo rpColorInfo = mglBatchTextureInfo(rpColor0);
    MGLRenderTextureInfo rpDepthInfo = mglBatchTextureInfo(rpDepth);
    Program *vertexProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    Program *fragmentProgram = mglResolveProgramForStageFromState(glm_ctx, _FRAGMENT_SHADER);
    FBOAttachment *color0Attachment = (fbo && (fbo->color_attachment_bitfield & 1u))
        ? &fbo->color_attachments[0]
        : NULL;
    Texture *color0Texture = mglTraceFramebufferAttachmentTexture(glm_ctx, color0Attachment);
    Texture *depthTexture = fbo ? mglTraceFramebufferAttachmentTexture(glm_ctx, &fbo->depth) : NULL;
    Texture *unit0Active = MGL_STATE(glm_ctx)->active_textures[0];
    Texture *unit0Tex2D = MGL_STATE(glm_ctx)->texture_units[0].textures[_TEXTURE_2D];
    Texture *unit1Active = MGL_STATE(glm_ctx)->active_textures[1];
    Texture *unit1Tex2D = MGL_STATE(glm_ctx)->texture_units[1].textures[_TEXTURE_2D];
    Texture *unit2Active = MGL_STATE(glm_ctx)->active_textures[2];
    Texture *unit2Tex2D = MGL_STATE(glm_ctx)->texture_units[2].textures[_TEXTURE_2D];
    GLuint cEver = 0u, cFull = 0u, cSource = 0u;
    GLuint dEver = 0u, dFull = 0u, dSource = 0u;
    mglTraceTextureLevelSummary(color0Texture,
                                color0Attachment ? color0Attachment->level : 0u,
                                &cEver,
                                &cFull,
                                &cSource);
    mglTraceTextureLevelSummary(depthTexture,
                                fbo ? fbo->depth.level : 0u,
                                &dEver,
                                &dFull,
                                &dSource);
    BOOL submitPhase = phase && strcmp(phase, "SUBMIT") == 0;
    int fsSlotHasRT = 0, fsSlotUsedCopy = 0;
    mglBatchFsFlags(fs0, fs1, fs2, fs3, &fsSlotHasRT, &fsSlotUsedCopy);

    mglTraceLog("REPLAY_CMD_%s flush=%llu batch=%u cmd=%u type=%s reason=%s "
                "program=%u vs=%u fs=%u mode=0x%x count=%d first=%d indexType=0x%x indexOffset=%u "
                "instances=%d baseVertex=%d baseInstance=%u ebo=%u eboPtr=%p "
                "encoder=%p pipelineState=%p fbo=%u rpFbo=%u rpColor=%p rpDepth=%p "
                "rpColorSize=%lux%lu rpDepthSize=%lux%lu rpLA/SA=%s/%s depthLA/SA=%s/%s "
                "fboColor0(tex=%u target=0x%x level=%u ptr=%p size=%ux%u mtl=%p init=%u/%u/%u rtVer=%u sampledVer=%u) "
                "fboDepth(tex=%u target=0x%x level=%u ptr=%p size=%ux%u mtl=%p init=%u/%u/%u rtVer=%u sampledVer=%u) "
                "units(u0 active=%u tex2D=%u u1 active=%u tex2D=%u u2 active=%u tex2D=%u) "
                "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) drawBuf=0x%x readBuf=0x%x "
                "depth(test=%d write=%d func=0x%x clear=%.6f) blend=%d cull=%d colorMask=%d%d%d%d",
                phase ? phase : "STATE",
                (unsigned long long)flushId,
                (unsigned)batchIndex,
                (unsigned)commandIndex,
                mglDrawCommandTypeName(cmd->type),
                reason ? reason : "",
                (unsigned)mglCurrentRenderProgramKey(glm_ctx),
                vertexProgram ? (unsigned)vertexProgram->name : 0u,
                fragmentProgram ? (unsigned)fragmentProgram->name : 0u,
                (unsigned)cmd->mode,
                (int)cmd->count,
                (int)cmd->first,
                (unsigned)cmd->indexType,
                (unsigned)cmd->indexBufferOffset,
                (int)cmd->instanceCount,
                (int)cmd->baseVertex,
                (unsigned)cmd->baseInstance,
                (unsigned)eboName,
                ebo,
                mglBatchEncoderTraceToken(_renderPassManager.state->currentRenderEncoderOwner),
                _pipelineCache.state->pipelineState,
                (unsigned)fboName,
                (unsigned)_renderPassManager.state->renderPassFramebufferName,
                rpColor0,
                rpDepth,
                (unsigned long)rpColorInfo.width,
                (unsigned long)rpColorInfo.height,
                (unsigned long)rpDepthInfo.width,
                (unsigned long)rpDepthInfo.height,
                mglLoadActionName((uint32_t)mglRenderPassLoadActionForTrace(
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    0u)),
                mglStoreActionName((uint32_t)mglRenderPassStoreActionForTrace(
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    0u)),
                mglLoadActionName((uint32_t)mglRenderPassLoadActionForTrace(
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    0u)),
                mglStoreActionName((uint32_t)mglRenderPassStoreActionForTrace(
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                    0u)),
                color0Attachment ? (unsigned)color0Attachment->texture : 0u,
                color0Attachment ? (unsigned)color0Attachment->textarget : 0u,
                color0Attachment ? (unsigned)color0Attachment->level : 0u,
                color0Texture,
                color0Texture ? (unsigned)color0Texture->width : 0u,
                color0Texture ? (unsigned)color0Texture->height : 0u,
                color0Texture ? color0Texture->mtl_data : NULL,
                (unsigned)cEver,
                (unsigned)cFull,
                (unsigned)cSource,
                color0Texture ? (unsigned)color0Texture->mtl_render_target_write_version : 0u,
                color0Texture ? (unsigned)color0Texture->mtl_gl_sampled_write_version : 0u,
                fbo ? (unsigned)fbo->depth.texture : 0u,
                fbo ? (unsigned)fbo->depth.textarget : 0u,
                fbo ? (unsigned)fbo->depth.level : 0u,
                depthTexture,
                depthTexture ? (unsigned)depthTexture->width : 0u,
                depthTexture ? (unsigned)depthTexture->height : 0u,
                depthTexture ? depthTexture->mtl_data : NULL,
                (unsigned)dEver,
                (unsigned)dFull,
                (unsigned)dSource,
                depthTexture ? (unsigned)depthTexture->mtl_render_target_write_version : 0u,
                depthTexture ? (unsigned)depthTexture->mtl_gl_sampled_write_version : 0u,
                unit0Active ? (unsigned)unit0Active->name : 0u,
                unit0Tex2D ? (unsigned)unit0Tex2D->name : 0u,
                unit1Active ? (unsigned)unit1Active->name : 0u,
                unit1Tex2D ? (unsigned)unit1Tex2D->name : 0u,
                unit2Active ? (unsigned)unit2Active->name : 0u,
                unit2Tex2D ? (unsigned)unit2Tex2D->name : 0u,
                (int)MGL_STATE(glm_ctx)->viewport[0],
                (int)MGL_STATE(glm_ctx)->viewport[1],
                (int)MGL_STATE(glm_ctx)->viewport[2],
                (int)MGL_STATE(glm_ctx)->viewport[3],
                MGL_STATE(glm_ctx)->caps.scissor_test ? 1 : 0,
                (int)MGL_STATE(glm_ctx)->var.scissor_box[0],
                (int)MGL_STATE(glm_ctx)->var.scissor_box[1],
                (int)MGL_STATE(glm_ctx)->var.scissor_box[2],
                (int)MGL_STATE(glm_ctx)->var.scissor_box[3],
                (unsigned)MGL_STATE(glm_ctx)->draw_buffer,
                (unsigned)MGL_STATE(glm_ctx)->read_buffer,
                MGL_STATE(glm_ctx)->caps.depth_test ? 1 : 0,
                MGL_STATE(glm_ctx)->var.depth_writemask ? 1 : 0,
                (unsigned)MGL_STATE(glm_ctx)->var.depth_func,
                (double)MGL_STATE(glm_ctx)->var.depth_clear_value,
                MGL_STATE(glm_ctx)->caps.blend ? 1 : 0,
                MGL_STATE(glm_ctx)->caps.cull_face ? 1 : 0,
	                MGL_STATE(glm_ctx)->var.color_writemask[0][0] ? 1 : 0,
		                MGL_STATE(glm_ctx)->var.color_writemask[0][1] ? 1 : 0,
		                MGL_STATE(glm_ctx)->var.color_writemask[0][2] ? 1 : 0,
		                MGL_STATE(glm_ctx)->var.color_writemask[0][3] ? 1 : 0);

    if (submitPhase && (fsSlotHasRT || fsSlotUsedCopy || mglProgramNeedsBindingTrace(fragmentProgram))) {
        mglTraceLog("REPLAY_CMD_TEXSLOTS flush=%llu batch=%u cmd=%u program=%u vs=%u fs=%u pipelineProgram=%u "
                    "s0(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu) "
                    "s1(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu) "
                    "s2(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu) "
                    "s3(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu)",
                    (unsigned long long)flushId,
                    (unsigned)batchIndex,
                    (unsigned)commandIndex,
                    (unsigned)mglCurrentRenderProgramKey(glm_ctx),
                    vertexProgram ? (unsigned)vertexProgram->name : 0u,
                    fragmentProgram ? (unsigned)fragmentProgram->name : 0u,
                    (unsigned)_pipelineCache.state->pipelineProgramName,
                    (unsigned)fs0->gl_texture_name,
                    (unsigned)fs0->sampler_unit,
                    (unsigned)fs0->program_name,
                    fs0->mtl_texture_ptr,
                    fs0->direct_mtl_texture_ptr,
                    fs0->sampled_copy_ptr,
                    (unsigned)fs0->used_sampled_copy,
                    (unsigned)fs0->used_fallback,
                    (unsigned)fs0->rt_write_version,
                    (unsigned)fs0->sampled_write_version,
                    (unsigned long)fs0->width,
                    (unsigned long)fs0->height,
                    (unsigned long)fs0->pixel_format,
                    (unsigned long)fs0->texture_type,
                    (unsigned)fs1->gl_texture_name,
                    (unsigned)fs1->sampler_unit,
                    (unsigned)fs1->program_name,
                    fs1->mtl_texture_ptr,
                    fs1->direct_mtl_texture_ptr,
                    fs1->sampled_copy_ptr,
                    (unsigned)fs1->used_sampled_copy,
                    (unsigned)fs1->used_fallback,
                    (unsigned)fs1->rt_write_version,
                    (unsigned)fs1->sampled_write_version,
                    (unsigned long)fs1->width,
                    (unsigned long)fs1->height,
                    (unsigned long)fs1->pixel_format,
                    (unsigned long)fs1->texture_type,
                    (unsigned)fs2->gl_texture_name,
                    (unsigned)fs2->sampler_unit,
                    (unsigned)fs2->program_name,
                    fs2->mtl_texture_ptr,
                    fs2->direct_mtl_texture_ptr,
                    fs2->sampled_copy_ptr,
                    (unsigned)fs2->used_sampled_copy,
                    (unsigned)fs2->used_fallback,
                    (unsigned)fs2->rt_write_version,
                    (unsigned)fs2->sampled_write_version,
                    (unsigned long)fs2->width,
                    (unsigned long)fs2->height,
                    (unsigned long)fs2->pixel_format,
                    (unsigned long)fs2->texture_type,
                    (unsigned)fs3->gl_texture_name,
                    (unsigned)fs3->sampler_unit,
                    (unsigned)fs3->program_name,
                    fs3->mtl_texture_ptr,
                    fs3->direct_mtl_texture_ptr,
                    fs3->sampled_copy_ptr,
                    (unsigned)fs3->used_sampled_copy,
                    (unsigned)fs3->used_fallback,
                    (unsigned)fs3->rt_write_version,
                    (unsigned)fs3->sampled_write_version,
                    (unsigned long)fs3->width,
                    (unsigned long)fs3->height,
                    (unsigned long)fs3->pixel_format,
                    (unsigned long)fs3->texture_type);
        if ((fsSlotHasRT || fsSlotUsedCopy) && fragmentProgram) {
            mglWriteProgramMSLDump(fragmentProgram,
                                   [NSString stringWithFormat:@"texslot-submit-fs-%u-flush-%llu-cmd-%u",
                                                              (unsigned)fragmentProgram->name,
                                                              (unsigned long long)flushId,
                                                              (unsigned)commandIndex]);
        } else if ((fsSlotHasRT || fsSlotUsedCopy) && drawProgram) {
            mglWriteProgramMSLDump(drawProgram,
                                   [NSString stringWithFormat:@"texslot-submit-program-%u-flush-%llu-cmd-%u",
                                                              (unsigned)drawProgram->name,
                                                              (unsigned long long)flushId,
                                                              (unsigned)commandIndex]);
        }
    }

    if (phase && strcmp(phase, "SUBMIT") == 0 && ebo) {
        Program *attribProgram = vertexProgram ? vertexProgram : drawProgram;
        bool forceRTSampledCopyAttribTrace = fsSlotHasRT || fsSlotUsedCopy;
        mglTraceReplayCommandVertexAttribSamples(glm_ctx,
                                                 attribProgram,
                                                 cmd,
                                                 ebo,
                                                 flushId,
                                                 batchIndex,
                                                 commandIndex,
                                                 forceRTSampledCopyAttribTrace);
    }
}


@end
