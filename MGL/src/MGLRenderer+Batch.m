// MGLRenderer+Batch.m
// Batch scheduling and execution methods extracted from MGLRenderer+Draw.m.
// P2-1: Split from MGLRenderer+Draw.m to reduce file size (11722 -> ~9392 lines).
// These methods do not depend on any file-scope static functions in Draw.m.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#import "mgl_sampler_compat.h"

static BOOL mglTextureMayNeedUploadEncoderDuringReplay(Texture *tex)
{
    if (!tex) {
        return NO;
    }

    if (tex->target == GL_TEXTURE_BUFFER &&
        tex->texture_buffer &&
        tex->texture_buffer->data.dirty_bits) {
        return YES;
    }

    if (!tex->mtl_data) {
        return YES;
    }

    if ((tex->dirty_bits &
         (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_ACCESS)) != 0) {
        return YES;
    }

    if (tex->is_render_target) {
        id<MTLTexture> existingTexture = (__bridge id<MTLTexture>)(tex->mtl_data);
        if (!existingTexture) {
            return YES;
        }

        MTLTextureUsage requiredRenderTargetUsage =
            MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        NSUInteger requiredMipLevels =
            (tex->target == GL_RENDERBUFFER || tex->samples > 1u)
                ? 1u
                : ((tex->mipmap_levels > 1u) ? (NSUInteger)tex->mipmap_levels : 1u);
        if ((existingTexture.usage & requiredRenderTargetUsage) != requiredRenderTargetUsage ||
            requiredMipLevels > existingTexture.mipmapLevelCount) {
            return YES;
        }
    }

    return NO;
}

static BOOL mglProgramSetSamplesTextureUnit(Program *program,
                                            Program *vertexProgram,
                                            Program *fragmentProgram,
                                            GLuint unit)
{
    return mglProgramSamplesTextureUnit(program, unit) ||
           mglProgramSamplesTextureUnit(vertexProgram, unit) ||
           mglProgramSamplesTextureUnit(fragmentProgram, unit);
}

static BOOL mglBatchMayNeedTextureUploadEncoderDuringReplay(const MGLDrawBatch *batch)
{
    if (!batch || !batch->state_snapshot) {
        return NO;
    }

    const GLMState *snapshot = (const GLMState *)batch->state_snapshot;
    /* These are the only program slots retained for replay (see
     * mglRetainBatchProgramReferences): monolithic + pipeline VS/FS.  Resolving
     * any other stage here would touch an unretained, possibly-freed program. */
    Program *program = (Program *)batch->retained_program;
    Program *vertexProgram = (Program *)batch->retained_vertex_program;
    Program *fragmentProgram = (Program *)batch->retained_fragment_program;

    for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
        if (!mglProgramSetSamplesTextureUnit(program, vertexProgram, fragmentProgram, unit)) {
            continue;
        }

        Texture *active = snapshot->active_textures[unit];
        if (mglTextureMayNeedUploadEncoderDuringReplay(active)) {
            return YES;
        }

        const TextureUnit *textureUnit = &snapshot->texture_units[unit];
        for (GLuint target = 0; target < _MAX_TEXTURE_TYPES; target++) {
            Texture *bound = textureUnit->textures[target];
            if (bound != active &&
                mglTextureMayNeedUploadEncoderDuringReplay(bound)) {
                return YES;
            }
        }
    }

    return NO;
}

@implementation MGLRenderer (Batch)

- (void)markCurrentFramebufferColorAttachmentWrittenAtIndex:(GLuint)attachmentIndex
{
    Framebuffer *fbo = ctx ? ctx->active_state->framebuffer : NULL;
    if (!fbo || attachmentIndex >= MAX_COLOR_ATTACHMENTS) {
        return;
    }

    if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
        return;
    }

    FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
    Texture *tex = [self framebufferAttachmentTexture:attachment];
    mglMarkTextureLevelRenderTargetWritten(tex, attachment->level);

    /* Update original-sampling authority: if the current rendering program had
     * VS framebuffer Y-flip injection, the RT already holds GL-visible
     * orientation, so RT_SAMPLE_COPY must not flip it again.  Exclude only true
     * framebuffer input passes (InSampler); ordinary mesh/item shaders can
     * sample Sampler0 while still producing an authoritative RT.
     *
     * Do not infer this from scissored atlas writes. Minecraft 1.21.11's GUI
     * item atlas still samples with GL texture-origin semantics, so it needs the
     * refreshed Y-flipped sampled copy. */
    {
        Program *renderingProgram = mglResolveProgramFromState(ctx);
        BOOL framebufferYFlipWrite =
            renderingProgram &&
            renderingProgram->spirv[_VERTEX_SHADER].mgl_injected_framebuffer_yflip == GL_TRUE &&
            !mglRendererProgramHasSampledResourceNamed(renderingProgram, "InSampler");

        if (tex && framebufferYFlipWrite) {
            tex->mtl_render_yflip_authority |= 1u;
        }
    }

    if (attachmentIndex == 0u &&
        mglTraceLogIsEnabled() &&
        mglTextureCanUseGLSampledRenderTargetCopy(tex)) {
        static uint64_t s_guiRTWriteMarkCount = 0;
        uint64_t hit = ++s_guiRTWriteMarkCount;
        if (hit <= 128ull || (hit % 256ull) == 0ull) {
            Program *program = mglResolveProgramFromState(ctx);
            Texture *rtColor = NULL;
            Texture *rtDepth = NULL;
            (void)mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx, fbo, &rtColor, &rtDepth);
            id<MTLTexture> colorMTL = tex->mtl_data ? (__bridge id<MTLTexture>)(tex->mtl_data) : nil;
            id<MTLTexture> depthMTL = (rtDepth && rtDepth->mtl_data)
                ? (__bridge id<MTLTexture>)(rtDepth->mtl_data)
                : nil;
            id<MTLTexture> rpColor0 = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.colorAttachments[0].texture : nil;
            id<MTLTexture> rpDepth = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.depthAttachment.texture : nil;
            mglTraceLog("RT_SAMPLE_COPY_WRITE_MARK hit=%llu fbo=%u program=%u rtTex=%u label=\"%s\" depthTex=%u depthLabel=\"%s\" viewport=%d,%d,%d,%d scissor(en=%d box=%d,%d,%d,%d) depth(test=%d write=%d func=0x%x) blend=%d cull=%d colorMask=%d%d%d%d level=%u texInit(ever=%u full=%u source=%u) levels=%u mips=%u mipmapped=%u mtlColor=%p fmt=%lu size=%lux%lu rpColor=%p rpDepth=%p depthMTL=%p",
                        (unsigned long long)hit,
                        (unsigned)fbo->name,
                        program ? (unsigned)program->name : (unsigned)(ctx ? ctx->active_state->program_name : 0u),
                        (unsigned)mglTraceTextureName(tex),
                        mglTraceTextureLabel(tex),
                        (unsigned)mglTraceTextureName(rtDepth),
                        mglTraceTextureLabel(rtDepth),
                  (int)ctx->active_state->viewport[0],
                  (int)ctx->active_state->viewport[1],
                  (int)ctx->active_state->viewport[2],
                  (int)ctx->active_state->viewport[3],
                  ctx->active_state->caps.scissor_test ? 1 : 0,
                  (int)ctx->active_state->var.scissor_box[0],
                  (int)ctx->active_state->var.scissor_box[1],
                  (int)ctx->active_state->var.scissor_box[2],
                  (int)ctx->active_state->var.scissor_box[3],
                  ctx->active_state->caps.depth_test ? 1 : 0,
                  ctx->active_state->var.depth_writemask ? 1 : 0,
                  (unsigned)ctx->active_state->var.depth_func,
                  ctx->active_state->caps.blend ? 1 : 0,
                  ctx->active_state->caps.cull_face ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][3] ? 1 : 0,
                  (unsigned)attachment->level,
                  mglTextureAttachmentLevel(tex, attachment->level)
                      ? (unsigned)mglTextureAttachmentLevel(tex, attachment->level)->ever_written : 0u,
                  mglTextureAttachmentLevel(tex, attachment->level)
                      ? (unsigned)mglTextureAttachmentLevel(tex, attachment->level)->has_initialized_data : 0u,
                  mglTextureAttachmentLevel(tex, attachment->level)
                      ? (unsigned)mglTextureAttachmentLevel(tex, attachment->level)->last_init_source : 0u,
                  tex ? (unsigned)tex->num_levels : 0u,
                  tex ? (unsigned)tex->mipmap_levels : 0u,
                  tex ? (unsigned)tex->mipmapped : 0u,
                  colorMTL,
                  (unsigned long)(colorMTL ? colorMTL.pixelFormat : MTLPixelFormatInvalid),
                  (unsigned long)(colorMTL ? colorMTL.width : 0),
                  (unsigned long)(colorMTL ? colorMTL.height : 0),
                  rpColor0,
                  rpDepth,
                        depthMTL);
        }
    }
}

- (void)markCurrentFramebufferDrawAttachmentsWritten
{
    Framebuffer *fbo = ctx ? ctx->active_state->framebuffer : NULL;
    if (!fbo) {
        return;
    }

    /* Track which attachments the draw-buffer pass already marked, so the
     * render-pass-descriptor cross-check below doesn't double-bump
     * mtl_render_target_write_version (each bump invalidates the sampled
     * copy and forces an unnecessary Y-flip blit). */
    bool attachmentMarked[MAX_COLOR_ATTACHMENTS] = {false};
    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (GLsizei slot = 0; slot < drawBufferCount; ++slot) {
        GLuint attachmentIndex = 0u;
        if (mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, (GLuint)slot),
                                                  &attachmentIndex)) {
            [self markCurrentFramebufferColorAttachmentWrittenAtIndex:attachmentIndex];
            if (attachmentIndex < MAX_COLOR_ATTACHMENTS) {
                attachmentMarked[attachmentIndex] = true;
            }
        }
    }

    if (!_renderPassManager.state->renderPassDescriptor) {
        return;
    }

    /* Cross-check against the active Metal render-pass descriptor.
     *
     * MC 1.21.11's render abstraction creates transient FBOs (e.g. the GUI
     * item atlas) where the GL draw-buffer state can be incomplete or
     * partially resolved by the time the Metal encoder ends.  The previous
     * code only fell back to the render-pass descriptor when NO attachment
     * was marked (markedAnyAttachment == false), which left RTs unmarked
     * when draw-buffer resolution partially succeeded but missed the actual
     * Metal color attachment.  An unmarked RT skips its write-version bump,
     * leaving the sampled Y-flip copy stale — the next sampler bind falls
     * back to the un-flipped Metal texture and GUI items render upside-down.
     *
     * Now: always cross-check.  Any FBO color attachment whose Metal texture
     * appears in the render-pass descriptor but wasn't covered by the
     * draw-buffer pass gets marked here.  The attachmentMarked[] guard
     * prevents double-bumping the write version. */
    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (attachmentMarked[attachmentIndex]) {
            continue;
        }
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }
        Texture *tex = [self framebufferAttachmentTexture:&fbo->color_attachments[attachmentIndex]];
        id<MTLTexture> mtlTex = (tex && tex->mtl_data)
            ? (__bridge id<MTLTexture>)(tex->mtl_data)
            : nil;
        if (!mtlTex) {
            continue;
        }
        for (GLuint colorSlot = 0u; colorSlot < MAX_COLOR_ATTACHMENTS; colorSlot++) {
            if (_renderPassManager.state->renderPassDescriptor.colorAttachments[colorSlot].texture == mtlTex) {
                [self markCurrentFramebufferColorAttachmentWrittenAtIndex:attachmentIndex];
                break;
            }
        }
    }
}

- (void)recordArrayDrawSubmittedMode:(GLenum)mode vertexCount:(uint64_t)vertexCount
{
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglNowSeconds());
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
    MGL_FRAME_STORE(g_mglLastDrawElementsSeconds, mglNowSeconds());
    MGL_FRAME_STORE(g_mglLastDrawElementsProgram, mglCurrentRenderProgramKey(ctx));
    MGL_FRAME_STORE(g_mglLastDrawElementsMode, mode);
    MGL_FRAME_STORE(g_mglLastDrawElementsCount,
                    (indexCount > (uint64_t)INT_MAX) ? INT_MAX : (GLsizei)indexCount);
    MGL_FRAME_INC(g_mglDrawElementsSinceSwap);
    MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap, indexCount);
    [self markCurrentFramebufferDrawAttachmentsWritten];
}

- (bool)bindActiveTexturesToMTL
{
    // search through active_texture_mask for enabled bits
    // 128 bits long.. do it on 4 parts
    for(int i=0; i<4; i++)
    {
        unsigned mask = STATE(active_texture_mask[i]);

        if (mask)
        {
            for(int bitpos=0; bitpos<32; bitpos++)
            {
                if (mask & (0x1 << bitpos))
                {
                    Texture *tex;
                    int unit = i * 32 + bitpos;

                    tex = STATE(active_textures[unit]);
                    if (!tex)
                    {
                        // Stale active texture mask bit; clear it and continue.
                        STATE(active_texture_mask[i]) &= ~(0x1u << bitpos);
                        mglInvalidateStateHashCachesForDirtyBits(ctx->active_state,
                                                                DIRTY_TEX_BINDING);
                        continue;
                    }

                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);
                }

                // early out
                if ((mask >> (bitpos + 1)) == 0)
                    break;
            }
        }
    }

    return true;
}

- (void)invalidateLastBoundState
{
    [_bindingSync invalidate];
}

/* DUAL-PROXY INVARIANT HELPERS: see MGLRenderer_Private.h.
 *
 * These centralize all writes to _core.activeState and ctx->active_state so
 * that the invariant ("MGL_STATE(ctx) and STATE(ctx) return the same
 * GLMState") cannot be broken by a caller forgetting to update one side.
 *
 * Valid invariant configurations:
 *   (A) _activeState == NULL  -> MGL_STATE falls through to ctx->active_state
 *                                (the "deactivated" / default mode)
 *   (B) _activeState != NULL  -> _activeState MUST equal ctx->active_state
 *                                (the "activated" mode used during batch replay)
 *
 * Configuration (A) is the teardown target; (B) is the batch-replay target. */
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
              @"STATE() and MGL_STATE() would read different GLMState objects");
}

- (void)recordLastBoundVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    [_bindingSync recordVertexBuffer:buffer offset:offset atIndex:index];
}

- (void)recordLastBoundFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    [_bindingSync recordFragmentBuffer:buffer offset:offset atIndex:index];
}

- (void)invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index
{
    [_bindingSync invalidateVertexBufferAtIndex:index];
}

- (void)invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index
{
    [_bindingSync invalidateFragmentBufferAtIndex:index];
}

- (void)setVertexTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
    [_bindingSync setVertexTextureIfNeeded:texture
                                   atIndex:index
                                   encoder:_renderPassManager.state->currentRenderEncoder];
}

- (void)setFragmentTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
    [_bindingSync setFragmentTextureIfNeeded:texture
                                     atIndex:index
                                     encoder:_renderPassManager.state->currentRenderEncoder];
}

- (void)setVertexSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
    [_bindingSync setVertexSamplerIfNeeded:sampler
                                   atIndex:index
                                   encoder:_renderPassManager.state->currentRenderEncoder];
}

- (void)setFragmentSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
    [_bindingSync setFragmentSamplerIfNeeded:sampler
                                     atIndex:index
                                     encoder:_renderPassManager.state->currentRenderEncoder];
}

- (void)setViewportIfNeeded:(MTLViewport)viewport
{
    [_bindingSync setViewportIfNeeded:viewport encoder:_renderPassManager.state->currentRenderEncoder];
}

- (void)setScissorRectIfNeeded:(MTLScissorRect)rect
{
    [_bindingSync setScissorRectIfNeeded:rect encoder:_renderPassManager.state->currentRenderEncoder];
}

- (void)setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode
{
    [_bindingSync setTriangleFillModeIfNeeded:mode encoder:_renderPassManager.state->currentRenderEncoder];
}

- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx
{
    GLMState *state = MGL_STATE(glm_ctx);
    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->vertex_buffer_map_list]);
    RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->fragment_buffer_map_list]);
    MGLEncodeContext encCtx = { .encoder = _renderPassManager.state->currentRenderEncoder };
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindBufferSizeConstantsForRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);
    RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-active-texture-bind"]);
    encCtx.encoder = _renderPassManager.state->currentRenderEncoder;
    if (![self bindTexturesToCurrentRenderEncoder:&encCtx]) {
        RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-sampled-texture-bind"]);
        encCtx.encoder = _renderPassManager.state->currentRenderEncoder;
        RETURN_FALSE_ON_FAILURE([self bindTexturesToCurrentRenderEncoder:&encCtx]);
    }
    return true;
}

- (void)restoreStateFromKey:(const MGLStateKey *)key context:(GLMContext)glm_ctx
{
    /* Program */
    mglRestoreProgramPipelinePair(glm_ctx,
                                  key->program_name,
                                  key->program_pipeline_name);

    /* VAO */
    uint32_t vaoName = key->vao_name;
    if (vaoName != (glm_ctx->active_state->vao ? glm_ctx->active_state->vao->name : 0)) {
        VertexArray *vaoInst = NULL;
        if (vaoName != 0) {
            vaoInst = (VertexArray *)searchHashTable(&glm_ctx->active_state->vao_table, vaoName);
        }
        glm_ctx->active_state->vao = vaoInst;
    }

    /* FBO */
    uint32_t batchFBO = key->fbo_name;
    uint32_t currentFBO = glm_ctx->active_state->framebuffer ? glm_ctx->active_state->framebuffer->name : 0;
    if (batchFBO != currentFBO) {
        Framebuffer *fbo = NULL;
        if (batchFBO != 0) {
            fbo = (Framebuffer *)searchHashTable(&glm_ctx->active_state->framebuffer_table, batchFBO);
        }
        glm_ctx->active_state->framebuffer = fbo;
    }
    mglRendererSyncFramebufferBindingNames(glm_ctx);

    /* Viewport */
    glm_ctx->active_state->viewport[0] = key->viewport[0];
    glm_ctx->active_state->viewport[1] = key->viewport[1];
    glm_ctx->active_state->viewport[2] = key->viewport[2];
    glm_ctx->active_state->viewport[3] = key->viewport[3];

    /* Scissor */
    if (key->scissor_enabled) {
        glm_ctx->active_state->caps.scissor_test = true;
        glm_ctx->active_state->var.scissor_box[0] = key->scissor[0];
        glm_ctx->active_state->var.scissor_box[1] = key->scissor[1];
        glm_ctx->active_state->var.scissor_box[2] = key->scissor[2];
        glm_ctx->active_state->var.scissor_box[3] = key->scissor[3];
    } else {
        glm_ctx->active_state->caps.scissor_test = false;
    }
}

- (void)traceReplayBatch:(MGLDrawBatch *)batch
                 context:(GLMContext)glm_ctx
                  flushId:(uint64_t)flushId
               batchIndex:(uint32_t)batchIndex
                    phase:(const char *)phase
{
    if (!batch || !glm_ctx) {
        return;
    }

    Program *drawProgram = mglTraceResolveDrawProgram(glm_ctx);
    MGLFragmentTextureTraceBinding *earlyFs0 = &_resourceFallback.fragmentTextureTraceBindings[0];
    MGLFragmentTextureTraceBinding *earlyFs1 = &_resourceFallback.fragmentTextureTraceBindings[1];
    MGLFragmentTextureTraceBinding *earlyFs2 = &_resourceFallback.fragmentTextureTraceBindings[2];
    MGLFragmentTextureTraceBinding *earlyFs3 = &_resourceFallback.fragmentTextureTraceBindings[3];
    BOOL earlyFsSlotHasRT =
        earlyFs0->rt_write_version != 0u ||
        earlyFs1->rt_write_version != 0u ||
        earlyFs2->rt_write_version != 0u ||
        earlyFs3->rt_write_version != 0u;
    BOOL earlyFsSlotUsedCopy =
        earlyFs0->used_sampled_copy ||
        earlyFs1->used_sampled_copy ||
        earlyFs2->used_sampled_copy ||
        earlyFs3->used_sampled_copy;
    if (!mglTraceShouldLogReplay(glm_ctx, drawProgram) &&
        !earlyFsSlotHasRT &&
        !earlyFsSlotUsedCopy) {
        return;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(glm_ctx, "replay.batch.trace");
    Framebuffer *fbo = glm_ctx->active_state->framebuffer;
    GLuint fboName = 0u;
    if (fbo &&
        mglRendererObjectPointerLikelyValid(fbo) &&
        mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        fboName = fbo->name;
    }
    id<MTLTexture> rpColor0 = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> rpDepth = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.depthAttachment.texture : nil;
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
                (unsigned)glm_ctx->active_state->program_name,
                (unsigned)glm_ctx->active_state->var.program_pipeline_binding,
                vertexProgram ? (unsigned)vertexProgram->name : 0u,
                fragmentProgram ? (unsigned)fragmentProgram->name : 0u,
                (unsigned)fboName,
                vao,
                vao ? (unsigned)vao->enabled_attribs : 0u,
                (int)glm_ctx->active_state->viewport[0],
                (int)glm_ctx->active_state->viewport[1],
                (int)glm_ctx->active_state->viewport[2],
                (int)glm_ctx->active_state->viewport[3],
                glm_ctx->active_state->caps.scissor_test ? 1 : 0,
                (int)glm_ctx->active_state->var.scissor_box[0],
                (int)glm_ctx->active_state->var.scissor_box[1],
                (int)glm_ctx->active_state->var.scissor_box[2],
                (int)glm_ctx->active_state->var.scissor_box[3],
                (unsigned)glm_ctx->active_state->draw_buffer,
                (unsigned)glm_ctx->active_state->read_buffer,
                glm_ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
                glm_ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
                glm_ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
                glm_ctx->active_state->var.color_writemask[0][3] ? 1 : 0,
                glm_ctx->active_state->caps.depth_test ? 1 : 0,
                glm_ctx->active_state->var.depth_writemask ? 1 : 0,
                (unsigned)glm_ctx->active_state->var.depth_func,
                glm_ctx->active_state->caps.blend ? 1 : 0,
                glm_ctx->active_state->caps.cull_face ? 1 : 0,
                (unsigned)glm_ctx->active_state->var.cull_face_mode,
                (unsigned)glm_ctx->active_state->var.front_face,
                (unsigned)glm_ctx->active_state->dirty_bits,
                _renderPassManager.state->currentRenderEncoder,
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

    Program *drawProgram = mglTraceResolveDrawProgram(glm_ctx);
    MGLFragmentTextureTraceBinding *fs0 = &_resourceFallback.fragmentTextureTraceBindings[0];
    MGLFragmentTextureTraceBinding *fs1 = &_resourceFallback.fragmentTextureTraceBindings[1];
    MGLFragmentTextureTraceBinding *fs2 = &_resourceFallback.fragmentTextureTraceBindings[2];
    MGLFragmentTextureTraceBinding *fs3 = &_resourceFallback.fragmentTextureTraceBindings[3];
    BOOL earlyFsSlotHasRT =
        fs0->rt_write_version != 0u ||
        fs1->rt_write_version != 0u ||
        fs2->rt_write_version != 0u ||
        fs3->rt_write_version != 0u;
    BOOL earlyFsSlotUsedCopy =
        fs0->used_sampled_copy ||
        fs1->used_sampled_copy ||
        fs2->used_sampled_copy ||
        fs3->used_sampled_copy;
    if (!mglTraceShouldLogReplay(glm_ctx, drawProgram) &&
        !earlyFsSlotHasRT &&
        !earlyFsSlotUsedCopy) {
        return;
    }

    Buffer *ebo = mglDrawCommandUsesElements(cmd) ? (Buffer *)cmd->elementBuffer : NULL;
    GLuint eboName = 0u;
    if (ebo &&
        mglRendererObjectPointerLikelyValid(ebo) &&
        mglPointerRangeIsReadable(ebo, sizeof(*ebo))) {
        eboName = ebo->name;
    }
    Framebuffer *fbo = glm_ctx->active_state->framebuffer;
    GLuint fboName = 0u;
    if (fbo &&
        mglRendererObjectPointerLikelyValid(fbo) &&
        mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        fboName = fbo->name;
    }
    id<MTLTexture> rpColor0 = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> rpDepth = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.depthAttachment.texture : nil;
    Program *vertexProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    Program *fragmentProgram = mglResolveProgramForStageFromState(glm_ctx, _FRAGMENT_SHADER);
    FBOAttachment *color0Attachment = (fbo && (fbo->color_attachment_bitfield & 1u))
        ? &fbo->color_attachments[0]
        : NULL;
    Texture *color0Texture = mglTraceFramebufferAttachmentTexture(glm_ctx, color0Attachment);
    Texture *depthTexture = fbo ? mglTraceFramebufferAttachmentTexture(glm_ctx, &fbo->depth) : NULL;
    Texture *unit0Active = glm_ctx->active_state->active_textures[0];
    Texture *unit0Tex2D = glm_ctx->active_state->texture_units[0].textures[_TEXTURE_2D];
    Texture *unit1Active = glm_ctx->active_state->active_textures[1];
    Texture *unit1Tex2D = glm_ctx->active_state->texture_units[1].textures[_TEXTURE_2D];
    Texture *unit2Active = glm_ctx->active_state->active_textures[2];
    Texture *unit2Tex2D = glm_ctx->active_state->texture_units[2].textures[_TEXTURE_2D];
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
    BOOL fsSlotHasRT =
        fs0->rt_write_version != 0u ||
        fs1->rt_write_version != 0u ||
        fs2->rt_write_version != 0u ||
        fs3->rt_write_version != 0u;
    BOOL fsSlotUsedCopy =
        fs0->used_sampled_copy ||
        fs1->used_sampled_copy ||
        fs2->used_sampled_copy ||
        fs3->used_sampled_copy;

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
                _renderPassManager.state->currentRenderEncoder,
                _pipelineCache.state->pipelineState,
                (unsigned)fboName,
                (unsigned)_renderPassManager.state->renderPassFramebufferName,
                rpColor0,
                rpDepth,
                (unsigned long)(rpColor0 ? rpColor0.width : 0),
                (unsigned long)(rpColor0 ? rpColor0.height : 0),
                (unsigned long)(rpDepth ? rpDepth.width : 0),
                (unsigned long)(rpDepth ? rpDepth.height : 0),
                mglLoadActionName(_renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare),
                mglStoreActionName(_renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare),
                mglLoadActionName(_renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare),
                mglStoreActionName(_renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare),
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
                (int)glm_ctx->active_state->viewport[0],
                (int)glm_ctx->active_state->viewport[1],
                (int)glm_ctx->active_state->viewport[2],
                (int)glm_ctx->active_state->viewport[3],
                glm_ctx->active_state->caps.scissor_test ? 1 : 0,
                (int)glm_ctx->active_state->var.scissor_box[0],
                (int)glm_ctx->active_state->var.scissor_box[1],
                (int)glm_ctx->active_state->var.scissor_box[2],
                (int)glm_ctx->active_state->var.scissor_box[3],
                (unsigned)glm_ctx->active_state->draw_buffer,
                (unsigned)glm_ctx->active_state->read_buffer,
                glm_ctx->active_state->caps.depth_test ? 1 : 0,
                glm_ctx->active_state->var.depth_writemask ? 1 : 0,
                (unsigned)glm_ctx->active_state->var.depth_func,
                (double)glm_ctx->active_state->var.depth_clear_value,
                glm_ctx->active_state->caps.blend ? 1 : 0,
                glm_ctx->active_state->caps.cull_face ? 1 : 0,
	                glm_ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
		                glm_ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
		                glm_ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
		                glm_ctx->active_state->var.color_writemask[0][3] ? 1 : 0);

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

- (void)flushDrawBufferLocked:(GLMContext)glm_ctx
{
    ctx = glm_ctx;

    /* DUAL-PROXY INVARIANT checkpoint: entering flushDrawBuffer.  All
     * subsequent batch replay / teardown paths assume the proxies start in
     * sync.  NSCAssert compiled out in release. */
    [self mglAssertDualProxyInSyncForContext:glm_ctx];

    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cb->batch_count == 0) {
        return;
    }

    static uint64_t s_flushDrawBufferLogCount = 0;
    uint64_t flushHit = ++s_flushDrawBufferLogCount;
    BOOL traceFlush = kMGLDiagnosticStateLogs &&
                      (flushHit <= 16ull || (flushHit % 512ull) == 0ull ||
                       cb->total_commands >= 128ull);
    uint32_t mdiBatchCount = 0;
    uint32_t mdiCommandCount = 0;
    uint32_t icbBatchCount = 0;
    uint32_t icbCommandCount = 0;
    uint32_t directBatchCount = 0;
    uint32_t directCommandCount = 0;
    uint32_t streamMergedBatchCount = 0;
    uint32_t streamMergedCommandCount = 0;
    uint32_t skippedCommandCount = 0;

    GLMState savedState;
    memcpy(&savedState, glm_ctx->active_state, sizeof(savedState));
    GLenum savedError = savedState.error;
    GLenum replayError = GL_NO_ERROR;

    @try {

    /* Same-key restore skip: consecutive sequential batches that share an
     * MGLStateKey can reuse the already-bound encoder state without another
     * ~83KB GLMState memcpy + full processGLState.  Collision residual is
     * identical to batch merge (memcmp of hashed key fields).
     * Hold a stack copy of lastKey — do not keep pointers into batch array
     * past teardown. */
    MGLStateKey lastKey;
    BOOL lastKeyValid = NO;
    BOOL lastExecuteOk = NO;
    memset(&lastKey, 0, sizeof(lastKey));

    for (uint32_t b = 0; b < cb->batch_count; b++) {
        @autoreleasepool {
            MGLDrawBatch *batch = &cb->batches[b];
            if (batch->command_count == 0)
                continue;

            {
            /* Same-key skip: only when the previous sequential batch fully
             * executed, the same encoder is still open with valid bind
             * cache, and keys match. Never skip across FBO/pass changes. */
            BOOL canSkipRestore = NO;
            if (_batching.skipSameKeyRestoreEnabled &&
                lastKeyValid &&
                lastExecuteOk &&
                _renderPassManager.state->currentRenderEncoder != nil &&
                _bindingSync.state->lastBoundValid &&
                mglStateKeysEqual(&batch->key, &lastKey) &&
                [self currentRenderPassMatchesCurrentFramebuffer]) {
                canSkipRestore = YES;
            } else if (!_batching.skipSameKeyRestoreEnabled &&
                       lastKeyValid &&
                       lastExecuteOk &&
                       mglStateKeysEqual(&batch->key, &lastKey) &&
                       mglEnvFlagEnabled("MGL_SKIP_SAME_KEY_ORACLE")) {
                /* Oracle: measure skip opportunity without changing behavior. */
                MGL_PERF_INC(g_mglSameKeyOracleWouldSkipSinceSwap);
            }

            if (canSkipRestore) {
                /* DUAL-PROXY INVARIANT: both _activeState (ObjC ivar) and
                 * glm_ctx->active_state (C pointer) must point to the same GLMState.
                 * After skipping restore, they both still point to ctx->state from
                 * the previous batch, which is correct. Verify the invariant holds. */
                if (_activeState != glm_ctx->active_state) {
                    /* Defensive: sync _activeState to match ctx->active_state if they
                     * diverged (shouldn't happen, but fail gracefully). */
                    _activeState = glm_ctx->active_state;
                }
                glm_ctx->active_state->dirty_bits = 0;
                MGL_PERF_INC(g_mglSameKeyRestoreSkipsSinceSwap);
            } else {
                [self restoreStateForBatch:batch
                                   context:glm_ctx
                                savedState:&savedState
                                   prevKey:(lastKeyValid ? &lastKey : NULL)];
            }

            if (![self checkBatchShouldExecute:batch
                                       context:glm_ctx
                                       flushId:flushHit
                                    batchIndex:b
                                   replayError:&replayError
                               skippedCommands:&skippedCommandCount]) {
                lastExecuteOk = NO;
                continue;
            }

            lastExecuteOk = YES;
            lastKey = batch->key;
            lastKeyValid = YES;
            MGL_PERF_INC(g_mglBatchesReplayedSinceSwap);

            MGLBatchPath scheduledPath = [self scheduleDrawBatch:batch context:glm_ctx];
            MGLEncodeContext encCtx = { .encoder = _renderPassManager.state->currentRenderEncoder };
            switch (scheduledPath) {
                case MGL_BATCH_PATH_STREAM_MERGE:
                    streamMergedBatchCount++;
                    streamMergedCommandCount += batch->command_count;
                    MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap);
                    MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap,
                                 batch->command_count);
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_STREAM_MERGE"];
                    [self issueStreamMergedBatch:batch context:glm_ctx encodeContext:&encCtx];
                    /* Stream batches bind transient vertex storage that is not
                     * represented by MGLStateKey. Force the next batch to
                     * restore its real VAO even when the GL keys are equal. */
                    [self invalidateLastBoundState];
                    break;
                case MGL_BATCH_PATH_MDI:
                    mdiBatchCount++;
                    mdiCommandCount += batch->command_count;
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_MDI"];
                    [self issueMDIBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
                case MGL_BATCH_PATH_ICB:
                    icbBatchCount++;
                    icbCommandCount += batch->command_count;
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_ICB"];
                    [self issueIndirectCommandBufferBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
                default:
                    directBatchCount++;
                    directCommandCount += batch->command_count;
                    MGL_PERF_INC(g_mglBatchesDirectSinceSwap);
                    MGL_PERF_ADD(g_mglDrawDirectSinceSwap, batch->command_count);
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_DIRECT"];
                    [self issueDirectBatch:batch context:glm_ctx encodeContext:&encCtx];
                    break;
            }

            [self recordBatchCommandStats:batch context:glm_ctx];
        } /* sequentialBatch */
        }
    }
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglNowSeconds());
    if (traceFlush || skippedCommandCount > 0 || replayError != GL_NO_ERROR) {
        MGLTraceNSLog(@"MGL TRACE flushDrawBuffer hit=%llu batches=%u totalCommands=%u arrays=%u elements=%u streamMergedBatches=%u streamMergedCommands=%u mdiBatches=%u mdiCommands=%u icbBatches=%u icbCommands=%u directBatches=%u directCommands=%u skippedCommands=%u",
              (unsigned long long)flushHit,
              cb->batch_count, cb->total_commands,
              cb->array_cmd_count, cb->element_cmd_count,
              streamMergedBatchCount, streamMergedCommandCount,
              mdiBatchCount, mdiCommandCount,
              icbBatchCount, icbCommandCount,
              directBatchCount, directCommandCount,
              skippedCommandCount);
    }
    } @finally {
        [_renderPassManager setTraceReplayFlushId:0 batchIndex:0];
        [self teardownBatchReplayForContext:glm_ctx savedState:&savedState
                                savedError:savedError replayError:replayError];
    }
}

- (MGLBatchPath)scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    if (!batch || batch->command_count == 0) {
        return MGL_BATCH_PATH_DIRECT;
    }

    if (batch->sampler_snapshots_mixed) {
        return MGL_BATCH_PATH_DIRECT;
    }

    if (batch->stream_merged) {
        return MGL_BATCH_PATH_STREAM_MERGE;
    }

    if (!batch->has_dynamic_uniform_bindings &&
        !batch->has_dynamic_vertex_bindings &&
        !batch->has_dynamic_texture_bindings &&
        !batch->sampler_snapshots_mixed &&
        mglEnvFlagEnabled("MGL_ENABLE_ICB_BATCH") &&
        !mglEnvFlagEnabled("MGL_DISABLE_ICB_BATCH") &&
        batch->key.primitive_type != 0xFFu) {
        if (@available(macOS 10.14, *)) {
            return MGL_BATCH_PATH_ICB;
        }
    }

    if (!mglEnvFlagEnabled("MGL_DISABLE_MDI") &&
        batch->mdi_compatible &&
        batch->command_count >= MGL_MDI_MIN_BATCH_SIZE &&
        !mglPolygonModePointForDrawMode(glm_ctx, batch->commands[0].mode)) {
        bool primitiveRestart = false;
        if (batch->uses_elements) {
            uint32_t dummy;
            primitiveRestart = mglPrimitiveRestartIndexForType(glm_ctx,
                                                               batch->commands[0].indexType,
                                                               &dummy);
        }
        if (!primitiveRestart) {
            return MGL_BATCH_PATH_MDI;
        }
    }

    return MGL_BATCH_PATH_DIRECT;
}

- (void)restoreStateForBatch:(MGLDrawBatch *)batch
                     context:(GLMContext)glm_ctx
                  savedState:(const GLMState *)savedState
{
    [self restoreStateForBatch:batch context:glm_ctx savedState:savedState prevKey:NULL];
}

- (void)restoreStateForBatch:(MGLDrawBatch *)batch
                     context:(GLMContext)glm_ctx
                  savedState:(const GLMState *)savedState
                     prevKey:(const MGLStateKey *)prevKey
{
    MGL_SIGNPOST_BEGIN(RestoreStateForBatch);
    /* DUAL-PROXY INVARIANT checkpoint: entering batch replay state restore.
     * Caller is responsible for having ctx->active_state already pointing
     * to the desired target (&ctx->state for replay).  We sync _activeState
     * to match at the end of this function. */
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    if (batch->state_snapshot) {
        /* Selective restore: only copy hot fields (~51KB vs 82KB full).
         * Cold fields (HashTables + unused buffer_base types) are restored
         * from savedState below. */
        mglCopyHotStateFields(glm_ctx->active_state,
                              (const GLMState *)batch->state_snapshot);
        MGL_PERF_INC(g_mglReplayMemcpyCountSinceSwap);
        /* The snapshot shallow-copies the 10 embedded HashTables in GLMState.
         * Each HashTable owns a dynamically-allocated keys/states array that
         * may have been reallocated since the snapshot was taken, making the
         * snapshot's copies stale (use-after-free risk).  Preserve the live
         * HashTables from savedState so lookups during replay remain valid. */
        glm_ctx->active_state->vao_table                 = savedState->vao_table;
        glm_ctx->active_state->buffer_table              = savedState->buffer_table;
        glm_ctx->active_state->texture_table             = savedState->texture_table;
        glm_ctx->active_state->shader_table              = savedState->shader_table;
        glm_ctx->active_state->program_table             = savedState->program_table;
        glm_ctx->active_state->program_pipeline_table    = savedState->program_pipeline_table;
        glm_ctx->active_state->transform_feedback_table  = savedState->transform_feedback_table;
        glm_ctx->active_state->renderbuffer_table        = savedState->renderbuffer_table;
        glm_ctx->active_state->framebuffer_table         = savedState->framebuffer_table;
        glm_ctx->active_state->sampler_table             = savedState->sampler_table;
        /* Restore the 11 cold buffer_base types from savedState (the snapshot
         * only captured the 5 hot types read by the encoder). */
        mglRestoreColdBufferBase(glm_ctx->active_state, savedState);
        mglRestoreProgramPipelinePair(glm_ctx, glm_ctx->active_state->program_name,
                                     glm_ctx->active_state->var.program_pipeline_binding);
    } else {
        [self restoreStateFromKey:&batch->key context:glm_ctx];
    }
    /* Activate snapshot-based state access for sync functions.
     * _activeState points to ctx->state (which now holds the snapshot data). */
    _activeState = glm_ctx->active_state;
    glm_ctx->active_state->dirty_bits = 0;

    static const GLuint kMGLFullReplayDirtyBits =
        (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_RENDER_STATE |
         DIRTY_TEX_BINDING | DIRTY_TEX | DIRTY_TEX_PARAM |
         DIRTY_SAMPLER | DIRTY_ALPHA_STATE | DIRTY_BUFFER |
         DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE);

    GLuint replayDirtyBits = kMGLFullReplayDirtyBits;
    BOOL prevKeyValid = (prevKey != NULL);
    BOOL canDelta = _batching.dirtyKeyDeltaEnabled &&
                    prevKeyValid &&
                    _renderPassManager.state->currentRenderEncoder != nil &&
                    _bindingSync.state->lastBoundValid;

    if (canDelta) {
        const MGLStateKey *a = prevKey;
        const MGLStateKey *b = &batch->key;
        replayDirtyBits = 0;
        if (a->program_name != b->program_name ||
            a->program_pipeline_name != b->program_pipeline_name ||
            a->vertex_program_name != b->vertex_program_name ||
            a->fragment_program_name != b->fragment_program_name) {
            replayDirtyBits |= DIRTY_PROGRAM | DIRTY_BUFFER_BASE_STATE | DIRTY_BUFFER;
        }
        if (a->vao_name != b->vao_name ||
            a->vertex_layout_hash != b->vertex_layout_hash) {
            replayDirtyBits |= DIRTY_VAO | DIRTY_BUFFER;
        }
        if (a->texture_hash != b->texture_hash) {
            replayDirtyBits |= DIRTY_TEX | DIRTY_TEX_BINDING | DIRTY_TEX_PARAM | DIRTY_SAMPLER;
        }
        if (a->render_state_hash != b->render_state_hash ||
            a->caps_flags != b->caps_flags ||
            a->scissor_enabled != b->scissor_enabled ||
            a->primitive_type != b->primitive_type ||
            memcmp(a->viewport, b->viewport, sizeof(a->viewport)) != 0 ||
            memcmp(a->scissor, b->scissor, sizeof(a->scissor)) != 0) {
            replayDirtyBits |= DIRTY_RENDER_STATE | DIRTY_ALPHA_STATE;
        }
        if (replayDirtyBits != kMGLFullReplayDirtyBits) {
            MGL_PERF_INC(g_mglDirtyKeyDeltaNarrowSinceSwap);
        }
    }

    Framebuffer *replayFBO = glm_ctx->active_state->framebuffer;
    if ((replayFBO && (replayFBO->dirty_bits & DIRTY_FBO_BINDING)) ||
        (prevKeyValid && prevKey->fbo_name != batch->key.fbo_name) ||
        (_renderPassManager.state->currentRenderEncoder &&
         ![self currentRenderPassMatchesCurrentFramebuffer])) {
        replayDirtyBits |= DIRTY_FBO;
    }
    /* Empty encoder cannot delta-bind — force full domains. */
    if (_renderPassManager.state->currentRenderEncoder == nil || !_bindingSync.state->lastBoundValid) {
        replayDirtyBits = kMGLFullReplayDirtyBits |
                          ((replayDirtyBits & DIRTY_FBO) ? DIRTY_FBO : 0);
        if ((replayFBO && (replayFBO->dirty_bits & DIRTY_FBO_BINDING)) ||
            (_renderPassManager.state->currentRenderEncoder &&
             ![self currentRenderPassMatchesCurrentFramebuffer])) {
            replayDirtyBits |= DIRTY_FBO;
        } else if (prevKeyValid && prevKey->fbo_name != batch->key.fbo_name) {
            replayDirtyBits |= DIRTY_FBO;
        }
    }
    mglMarkRendererDirtyBits(glm_ctx->active_state, replayDirtyBits);
    MGL_SIGNPOST_END(RestoreStateForBatch);
}

- (void)teardownBatchReplayForContext:(GLMContext)glm_ctx
                           savedState:(const GLMState *)savedState
                           savedError:(GLenum)savedError
                          replayError:(GLenum)replayError
{
    /* DUAL-PROXY INVARIANT checkpoint: entering batch replay teardown.
     * Pre-teardown, both proxies may be in either config:
     *   (A) default: _activeState=NULL, ctx->active_state=&ctx->state
     *   (B) redirected: both point at &ctx->state (snapshot-based replay)
     * The assert verifies whichever config holds is internally consistent. */
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    /* Deactivate snapshot-based state access — revert to live ctx->state.
     * DUAL-PROXY INVARIANT: use the helper so both proxies revert atomically
     * (previously two separate statements: _activeState=nil then ctx reset). */
    [self mglRestoreLiveActiveStateForContext:glm_ctx];
    /* DUAL-PROXY INVARIANT checkpoint: post-teardown, both proxies must be
     * in default config (A).  MGL_STATE now falls through to &ctx->state. */
    [self mglAssertDualProxyInSyncForContext:glm_ctx];
    mglResetCommandBufferForContext(glm_ctx, &glm_ctx->draw_command_buffer);
    /* Task 4: Reset the snapshot arena now that all batch replay is complete
     * and mglResetCommandBufferForContext has cleared all batch references.
     * This is the safe point — no worker/encoder is accessing snapshot data. */
    if (_batching.arenaSnapshotEnabled) {
        mglResetBatchArena(&_batching.batchArena);
    }
    memcpy(glm_ctx->active_state, savedState, sizeof(GLMState));
    /* savedState carries the independent hash flags latched by live mutations.
     * Clear only renderer-consumed legacy bits; deriving flags from those bits
     * would force an unnecessary hash recompute after every non-empty flush. */
    mglClearStateDirtyBitsPreservingHashInvalidation(glm_ctx->active_state);
    mglRestoreProgramPipelinePair(glm_ctx, glm_ctx->active_state->program_name,
                                  glm_ctx->active_state->var.program_pipeline_binding);
    if (savedError == GL_NO_ERROR && replayError != GL_NO_ERROR) {
        glm_ctx->active_state->error = replayError;
    }
}

- (BOOL)checkBatchShouldExecute:(MGLDrawBatch *)batch
                        context:(GLMContext)glm_ctx
                        flushId:(uint64_t)flushId
                     batchIndex:(uint32_t)batchIndex
                    replayError:(GLenum *)replayError
                skippedCommands:(uint32_t *)skippedCommands
{
    [_renderPassManager setTraceReplayFlushId:flushId batchIndex:batchIndex];
    [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                batchIndex:batchIndex phase:"RESTORE"];

    if (![self prepareRenderPassIfFBOChanged:batch context:glm_ctx replayError:replayError]) {
        [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                    batchIndex:batchIndex phase:"SKIP_FBO_ROTATION"];
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch command:&batch->commands[i]
                             context:glm_ctx flushId:flushId
                          batchIndex:batchIndex commandIndex:i
                               phase:"SKIP" reason:"fbo_rotation"];
        }
        *skippedCommands += batch->command_count;
        MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
        return NO;
    }

    if ([self processGLState:true] == false) {
        if (glm_ctx->active_state->error != GL_NO_ERROR) {
            *replayError = glm_ctx->active_state->error;
        }
        [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                    batchIndex:batchIndex phase:"SKIP_PROCESS_STATE"];
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch command:&batch->commands[i]
                             context:glm_ctx flushId:flushId
                          batchIndex:batchIndex commandIndex:i
                               phase:"SKIP" reason:"processGLState"];
        }
        *skippedCommands += batch->command_count;
        MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
        return NO;
    }

    /* A stable sampler snapshot is batch state, not per-draw state. Apply it
     * once after texture binding so stream-merge, MDI and ICB paths remain
     * available. Only genuinely mixed batches rebind per command. */
    MGLEncodeContext samplerEncCtx = { .encoder = _renderPassManager.state->currentRenderEncoder };
    if (!batch->sampler_snapshots_mixed &&
        batch->sampler_snapshot_id != MGL_INVALID_SAMPLER_SNAPSHOT_ID &&
        ![self applySamplerSnapshotForCommand:&batch->commands[0]
                                      context:glm_ctx
                                encodeContext:&samplerEncCtx]) {
        [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                    batchIndex:batchIndex phase:"SKIP_SAMPLER_SNAPSHOT"];
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch command:&batch->commands[i]
                             context:glm_ctx flushId:flushId
                          batchIndex:batchIndex commandIndex:i
                               phase:"SKIP" reason:"sampler_snapshot"];
        }
        *skippedCommands += batch->command_count;
        MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
        return NO;
    }

    [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                batchIndex:batchIndex phase:"READY"];

    if ([self currentDrawRasterizationIsEmpty]) {
        [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                    batchIndex:batchIndex phase:"SKIP_EMPTY_RASTER"];
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch command:&batch->commands[i]
                             context:glm_ctx flushId:flushId
                          batchIndex:batchIndex commandIndex:i
                               phase:"SKIP" reason:"empty_rasterization"];
        }
        *skippedCommands += batch->command_count;
        MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
        return NO;
    }

    GLenum mode = batch->commands[0].mode;
    if ([self currentDrawModeIsFullyCulled:mode]) {
        [self traceReplayBatch:batch context:glm_ctx flushId:flushId
                    batchIndex:batchIndex phase:"SKIP_FULLY_CULLED"];
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch command:&batch->commands[i]
                             context:glm_ctx flushId:flushId
                          batchIndex:batchIndex commandIndex:i
                               phase:"SKIP" reason:"front_and_back_culled"];
        }
        *skippedCommands += batch->command_count;
        MGL_PERF_INC(g_mglDrawSkippedSinceSwap);
        return NO;
    }

    [self applyPolygonOffsetForDrawMode:mode];
    return YES;
}

- (void)recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        switch (cmd->type) {
        case MGL_CMD_DRAW_ARRAYS:
        case MGL_CMD_DRAW_ARRAYS_INSTANCED:
        case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
            MGL_FRAME_INC(g_mglDrawArraysSinceSwap);
            MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap,
                          (uint64_t)(cmd->count > 0 ? cmd->count : 0));
            break;
        case MGL_CMD_DRAW_ELEMENTS:
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED:
        case MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX:
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX:
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE:
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE:
            MGL_FRAME_INC(g_mglDrawElementsSinceSwap);
            MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap,
                          (uint64_t)(cmd->count > 0 ? cmd->count : 0));
            break;
        default:
            break;
        }
    }
    [self markCurrentFramebufferDrawAttachmentsWritten];
    (void)glm_ctx;
}

- (void)issueStreamMergedBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                 encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!batch || !batch->stream_merged || batch->stream_index_count == 0) {
        if (batch && batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"SKIP"
                              reason:"stream_empty"];
        }
        return;
    }

    if (batch->key.primitive_type == 0xFFu) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_unsupported_primitive"];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    if (!mglEnvFlagEnabled("MGL_DISABLE_MDI")) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"ISSUE"
                              reason:"stream_merge_to_mdi"];
        }
        if ([self issueStreamMergedMDIBatch:batch context:glm_ctx encodeContext:encCtx]) {
            return;
        }
    }

    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;
    if (!indexBuffer || ![self processBuffer:indexBuffer]) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_index_buffer"];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    id<MTLBuffer> mtlIndexBuffer = (__bridge id<MTLBuffer>)(indexBuffer->data.mtl_data);
    if (!mtlIndexBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_no_mtl_index"];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    MGLDrawCommand *firstCmd = &batch->commands[0];
    MTLPrimitiveType primType = (MTLPrimitiveType)batch->key.primitive_type;

    [encCtx->encoder drawIndexedPrimitives:primType
                                      indexCount:(NSUInteger)batch->stream_index_count
                                       indexType:MTLIndexTypeUInt32
                                     indexBuffer:mtlIndexBuffer
                               indexBufferOffset:0
                                   instanceCount:1
                                      baseVertex:0
                                    baseInstance:firstCmd->baseInstance];
    [self traceReplayCommand:batch
                     command:firstCmd
                     context:glm_ctx
                     flushId:_renderPassManager.state->traceReplayFlushId
                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                commandIndex:0
                       phase:"SUBMIT"
                      reason:"stream_direct_merged"];
}

- (BOOL)issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                    encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!batch || !batch->stream_merged || batch->command_count == 0 ||
        batch->stream_index_count == 0 || !encCtx->encoder) {
        return NO;
    }
    if (mglEnvFlagEnabled("MGL_DISABLE_MDI") ||
        batch->key.primitive_type == 0xFFu) {
        return NO;
    }

    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;
    if (!indexBuffer || ![self processBuffer:indexBuffer]) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_mdi_index_buffer"];
        }
        return NO;
    }

    id<MTLBuffer> mtlIndexBuffer = (__bridge id<MTLBuffer>)(indexBuffer->data.mtl_data);
    if (!mtlIndexBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_mdi_no_mtl_index"];
        }
        return NO;
    }

    size_t argSize = sizeof(MTLDrawIndexedPrimitivesIndirectArguments);
    if (batch->command_count > (UINT32_MAX / argSize)) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_mdi_args_overflow"];
        }
        return NO;
    }

    NSUInteger neededBytes = (NSUInteger)argSize * (NSUInteger)batch->command_count;
    NSUInteger indirectArgsOffset = 0;
    id<MTLBuffer> indirectArgsBuffer =
        [self mdiArgumentScratchBufferWithLength:neededBytes
                                          offset:&indirectArgsOffset];
    if (!indirectArgsBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_mdi_args_alloc"];
        }
        return NO;
    }

    MTLDrawIndexedPrimitivesIndirectArguments *args =
        (MTLDrawIndexedPrimitivesIndirectArguments *)((uint8_t *)indirectArgsBuffer.contents + indirectArgsOffset);
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        args[i].indexCount = (uint32_t)cmd->count;
        args[i].instanceCount = (uint32_t)(cmd->instanceCount > 0 ? cmd->instanceCount : 1);
        args[i].indexStart = 0u;
        args[i].baseVertex = 0;
        args[i].baseInstance = cmd->baseInstance;
    }

    MTLPrimitiveType primType = (MTLPrimitiveType)batch->key.primitive_type;
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        [encCtx->encoder drawIndexedPrimitives:primType
                                           indexType:MTLIndexTypeUInt32
                                         indexBuffer:mtlIndexBuffer
                                   indexBufferOffset:(NSUInteger)cmd->indexBufferOffset
                                      indirectBuffer:indirectArgsBuffer
                                indirectBufferOffset:indirectArgsOffset + (i * argSize)];
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"stream_mdi_indexed"];
    }

    return YES;
}

- (BOOL)issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!batch || batch->command_count == 0 || !_device || !encCtx->encoder) {
        if (batch && batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"icb_unavailable"];
        }
        return NO;
    }
    if (batch->key.primitive_type == 0xFFu) {
        [self traceReplayCommand:batch
                         command:&batch->commands[0]
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:0
                           phase:"FALLBACK"
                          reason:"icb_unsupported_primitive"];
        return NO;
    }
    if (!mglEnvFlagEnabled("MGL_ENABLE_ICB_BATCH") ||
        mglEnvFlagEnabled("MGL_DISABLE_ICB_BATCH")) {
        return NO;
    }

    if (@available(macOS 10.14, *)) {
        BOOL indexed = batch->uses_elements ? YES : NO;
        MTLIndirectCommandBufferDescriptor *descriptor = [[MTLIndirectCommandBufferDescriptor alloc] init];
        descriptor.commandTypes = indexed ? MTLIndirectCommandTypeDrawIndexed : MTLIndirectCommandTypeDraw;
        descriptor.inheritPipelineState = YES;
        descriptor.inheritBuffers = YES;
        descriptor.maxVertexBufferBindCount = 0;
        descriptor.maxFragmentBufferBindCount = 0;

        id<MTLIndirectCommandBuffer> icb = nil;
        @try {
            icb = [_device newIndirectCommandBufferWithDescriptor:descriptor
                                                  maxCommandCount:(NSUInteger)batch->command_count
                                                          options:MTLResourceStorageModePrivate];
        } @catch (NSException *exception) {
            static uint64_t s_icbCreateExceptionCount = 0;
            uint64_t hit = ++s_icbCreateExceptionCount;
            if (hit <= 8ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: ICB creation failed, falling back to indirect draw loop: %@", exception);
            }
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"icb_create_exception"];
            return NO;
        }
        if (!icb) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"icb_create_nil"];
            return NO;
        }

        [icb resetWithRange:NSMakeRange(0, (NSUInteger)batch->command_count)];

        MTLPrimitiveType primType = (MTLPrimitiveType)batch->key.primitive_type;
        if (indexed) {
            for (uint32_t i = 0; i < batch->command_count; i++) {
                MGLDrawCommand *cmd = &batch->commands[i];
                if (cmd->indexType == GL_UNSIGNED_BYTE) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_u8_index"];
                    return NO;
                }

                Buffer *glBuf = NULL;
                id<MTLBuffer> idxBuf = nil;
                if (![self resolveElementBufferForCommand:cmd
                                                    label:"icbBatch"
                                                  context:glm_ctx
                                                 glBuffer:&glBuf
                                                mtlBuffer:&idxBuf]) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_resolve_element"];
                    return NO;
                }

                NSUInteger drawIndexOffset = cmd->indexBufferOffset;
                MTLIndexType drawIndexType = getMTLIndexType(cmd->indexType);
                id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                              glBuf,
                                                                              idxBuf,
                                                                              cmd->indexType,
                                                                              &drawIndexOffset,
                                                                              &drawIndexType);
                if (!drawIndexBuffer || (GLuint)drawIndexType == 0xFFFFFFFF) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_prepared_index"];
                    return NO;
                }

                id<MTLIndirectRenderCommand> indirectCommand = [icb indirectRenderCommandAtIndex:(NSUInteger)i];
                if (!indirectCommand) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_command_nil"];
                    return NO;
                }

                [indirectCommand drawIndexedPrimitives:primType
                                            indexCount:(NSUInteger)cmd->count
                                             indexType:drawIndexType
                                           indexBuffer:drawIndexBuffer
                                     indexBufferOffset:drawIndexOffset
                                         instanceCount:(NSUInteger)cmd->instanceCount
                                            baseVertex:(NSInteger)cmd->baseVertex
                                          baseInstance:(NSUInteger)cmd->baseInstance];
                [encCtx->encoder useResource:drawIndexBuffer
                                             usage:MTLResourceUsageRead
                                            stages:MTLRenderStageVertex];
            }
        } else {
            for (uint32_t i = 0; i < batch->command_count; i++) {
                MGLDrawCommand *cmd = &batch->commands[i];
                id<MTLIndirectRenderCommand> indirectCommand = [icb indirectRenderCommandAtIndex:(NSUInteger)i];
                if (!indirectCommand) {
                    [self traceReplayCommand:batch
                                     command:cmd
                                     context:glm_ctx
                                     flushId:_renderPassManager.state->traceReplayFlushId
                                  batchIndex:_renderPassManager.state->traceReplayBatchIndex
                                commandIndex:i
                                       phase:"FALLBACK"
                                      reason:"icb_command_nil"];
                    return NO;
                }
                [indirectCommand drawPrimitives:primType
                                    vertexStart:(NSUInteger)cmd->first
                                    vertexCount:(NSUInteger)cmd->count
                                  instanceCount:(NSUInteger)cmd->instanceCount
                                   baseInstance:(NSUInteger)cmd->baseInstance];
            }
        }

        [encCtx->encoder useResource:icb
                                     usage:MTLResourceUsageRead
                                    stages:MTLRenderStageVertex];
        [encCtx->encoder executeCommandsInBuffer:icb
                                             withRange:NSMakeRange(0, (NSUInteger)batch->command_count)];
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch
                             command:&batch->commands[i]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SUBMIT"
                              reason:"icb"];
        }
        return YES;
    }

    return NO;
}

- (id<MTLBuffer>)mdiArgumentScratchBufferWithLength:(NSUInteger)length
                                             offset:(NSUInteger *)offsetOut
{
    return [_renderPassManager mdiArgumentScratchBufferWithDevice:_device
                                                            length:length
                                                            offset:offsetOut];
}

@end
