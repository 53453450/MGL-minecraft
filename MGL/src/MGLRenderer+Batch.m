// MGLRenderer+Batch.m
// Batch scheduling and execution methods extracted from MGLRenderer+Draw.m.
// P2-1: Split from MGLRenderer+Draw.m to reduce file size (11722 -> ~9392 lines).
// These methods do not depend on any file-scope static functions in Draw.m.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"

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
            id<MTLTexture> rpColor0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
            id<MTLTexture> rpDepth = _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.texture : nil;
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

    if (!_renderPassDescriptor) {
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
            if (_renderPassDescriptor.colorAttachments[colorSlot].texture == mtlTex) {
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
    for (int i = 0; i < kMGLMaxBufferSlots; i++) {
        _lastBoundVertexBuffers[i].buffer = nil;
        _lastBoundVertexBuffers[i].offset = 0;
        _lastBoundFragmentBuffers[i].buffer = nil;
        _lastBoundFragmentBuffers[i].offset = 0;
    }
    for (int i = 0; i < TEXTURE_UNITS; i++) {
        _lastBoundVertexTextures[i] = nil;
        _lastBoundFragmentTextures[i] = nil;
        _lastBoundVertexSamplers[i] = nil;
        _lastBoundFragmentSamplers[i] = nil;
    }
    _lastPipelineState = nil;
    _lastDepthStencilState = nil;
    _lastViewport = (MTLViewport){0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    _lastScissorRect = (MTLScissorRect){0, 0, 0, 0};
    _lastCullMode = MTLCullModeNone;
    _lastFrontFacingWinding = MTLWindingClockwise;
    _lastTriangleFillMode = MTLTriangleFillModeFill;
    _lastDepthBias = 0;
    _lastDepthBiasClamp = 0;
    _lastDepthSlopeScale = 0;
    _lastBoundValid = NO;
}

- (void)saveDedupStateToWorker:(MGLWorkerContext *)worker
{
    if (!worker) return;

    worker->encoder = _currentRenderEncoder;

    for (int i = 0; i < kMGLMaxBufferSlots; i++) {
        worker->lastBoundVertexBuffers[i] = _lastBoundVertexBuffers[i];
        worker->lastBoundFragmentBuffers[i] = _lastBoundFragmentBuffers[i];
    }
    for (int i = 0; i < TEXTURE_UNITS; i++) {
        worker->lastBoundVertexTextures[i] = _lastBoundVertexTextures[i];
        worker->lastBoundFragmentTextures[i] = _lastBoundFragmentTextures[i];
        worker->lastBoundVertexSamplers[i] = _lastBoundVertexSamplers[i];
        worker->lastBoundFragmentSamplers[i] = _lastBoundFragmentSamplers[i];
    }
    worker->lastPipelineState = _lastPipelineState;
    worker->lastDepthStencilState = _lastDepthStencilState;
    worker->lastViewport = _lastViewport;
    worker->lastScissorRect = _lastScissorRect;
    worker->lastCullMode = _lastCullMode;
    worker->lastFrontFacingWinding = _lastFrontFacingWinding;
    worker->lastTriangleFillMode = _lastTriangleFillMode;
    worker->lastDepthBias = _lastDepthBias;
    worker->lastDepthBiasClamp = _lastDepthBiasClamp;
    worker->lastDepthSlopeScale = _lastDepthSlopeScale;
    worker->lastBoundValid = _lastBoundValid;

    worker->pipelineState = _pipelineState;
    worker->pipelineColor0Format = _pipelineColor0Format;
    worker->pipelineDepthFormat = _pipelineDepthFormat;
    worker->pipelineStencilFormat = _pipelineStencilFormat;
    worker->pipelineProgramName = _pipelineProgramName;

    worker->mdiArgsScratchOffset = _mdiArgsScratchOffset;

    worker->traceReplayFlushId = _traceReplayFlushId;
    worker->traceReplayBatchIndex = _traceReplayBatchIndex;
}

- (void)loadDedupStateFromWorker:(const MGLWorkerContext *)worker
{
    if (!worker) return;

    _currentRenderEncoder = worker->encoder;

    for (int i = 0; i < kMGLMaxBufferSlots; i++) {
        _lastBoundVertexBuffers[i] = worker->lastBoundVertexBuffers[i];
        _lastBoundFragmentBuffers[i] = worker->lastBoundFragmentBuffers[i];
    }
    for (int i = 0; i < TEXTURE_UNITS; i++) {
        _lastBoundVertexTextures[i] = worker->lastBoundVertexTextures[i];
        _lastBoundFragmentTextures[i] = worker->lastBoundFragmentTextures[i];
        _lastBoundVertexSamplers[i] = worker->lastBoundVertexSamplers[i];
        _lastBoundFragmentSamplers[i] = worker->lastBoundFragmentSamplers[i];
    }
    _lastPipelineState = worker->lastPipelineState;
    _lastDepthStencilState = worker->lastDepthStencilState;
    _lastViewport = worker->lastViewport;
    _lastScissorRect = worker->lastScissorRect;
    _lastCullMode = worker->lastCullMode;
    _lastFrontFacingWinding = worker->lastFrontFacingWinding;
    _lastTriangleFillMode = worker->lastTriangleFillMode;
    _lastDepthBias = worker->lastDepthBias;
    _lastDepthBiasClamp = worker->lastDepthBiasClamp;
    _lastDepthSlopeScale = worker->lastDepthSlopeScale;
    _lastBoundValid = worker->lastBoundValid;

    _pipelineState = worker->pipelineState;
    _pipelineColor0Format = worker->pipelineColor0Format;
    _pipelineDepthFormat = worker->pipelineDepthFormat;
    _pipelineStencilFormat = worker->pipelineStencilFormat;
    _pipelineProgramName = worker->pipelineProgramName;

    _mdiArgsScratchOffset = worker->mdiArgsScratchOffset;

    _traceReplayFlushId = worker->traceReplayFlushId;
    _traceReplayBatchIndex = worker->traceReplayBatchIndex;
}

- (BOOL)parallelEncodeEnabled
{
    static BOOL s_checked = NO;
    static BOOL s_enabled = NO;
    if (!s_checked) {
        const char *env = getenv("MGL_PARALLEL_ENCODE");
        s_enabled = (env && env[0] == '1');
        s_checked = YES;
    }
    return s_enabled;
}

- (MGLBatchPath)encodeBatchForParallelWorker:(MGLWorkerContext *)worker
                                       batch:(MGLDrawBatch *)batch
                                     context:(GLMContext)glm_ctx
                                     flushId:(uint64_t)flushId
                                  batchIndex:(uint32_t)batchIndex
                                  savedState:(const GLMState *)savedState
                                    executed:(BOOL *)executedOut
{
    if (executedOut) *executedOut = NO;

    /* Bug 4: Re-entrancy guard.  If active_state is already redirected
     * (not pointing at &ctx->state), a previous encodeBatchForParallelWorker
     * did not restore it — either an exception escaped the @finally, or
     * true multi-thread parallel encode was attempted on one GLMContext.
     * Either way, proceeding would corrupt the already-redirected state.
     * NSCAssert is compiled out in release but catches bugs in debug. */
    NSCAssert(glm_ctx->active_state == &glm_ctx->state,
              @"encodeBatchForParallelWorker entered with active_state already "
              @"redirected — nested/parallel call on one GLMContext would race");

    /* Install worker's dedup state (simulates the worker's encoder having
     * certain state already bound from a previous batch on the same worker). */
    [self loadDedupStateFromWorker:worker];

    /* Invalidate dedup — the sub-encoder is fresh and has nothing bound.
     * This forces processGLStateLocked to re-issue all binds.  In sequential
     * mode this causes redundant Metal calls but produces identical output;
     * in parallel mode (Step 4) each sub-encoder genuinely starts empty. */
    [self invalidateLastBoundState];

    /* Bug D7: Heap-allocate the per-worker GLMState to avoid ~83KB stack
     * frame.  Three stack copies (preGroupState + worker0 + worker1) under
     * MGL_PARALLEL_ENCODE can exceed small-thread stacks (e.g. Forge
     * EarlyDisplay ~2MB), causing ___chkstk_darwin SIGBUS.
     *
     * Bug 2: If malloc fails, fall back to encoding on the sub-encoder
     * using live ctx->state (no worker isolation).  This loses parallel-
     * safety but prevents silently dropping the batch geometry.  The
     * sub-encoder is already created by the caller; we just encode
     * without the active_state redirect. */
    bool workerStateAllocated = false;
    if (!worker->workerState) {
        worker->workerState = (GLMState *)malloc(sizeof(GLMState));
        if (!worker->workerState) {
            /* Fallback: encode using live state, no redirect, no free. */
            worker->workerState = NULL;
        } else {
            workerStateAllocated = true;
        }
    }

    /* Redirect state access to the worker's per-worker GLMState copy so
     * that restoreStateForBatch memcpy's the snapshot into workerState,
     * not the shared ctx->state.  This is the key Stage 5.3 change: each
     * worker's encoding reads from its own isolated state, making true
     * dispatch_async parallel encoding safe.  In sequential mode both
     * pointers ultimately reference the same data, so behaviour is
     * identical.
     *
     * Bug D10 CONCURRENCY NOTE: glm_ctx->active_state is a shared pointer
     * mutated here without locking.  Today this is safe because parallel
     * sub-encoders are created/encoded/ended SEQUENTIALLY (create sub0 →
     * encode → endEncoding → create sub1 → encode → endEncoding).  True
     * multi-thread parallel encode on one GLMContext (e.g. dispatch_async
     * to concurrent queues) would race active_state/_activeState/
     * _currentRenderEncoder between workers.  Enabling true concurrency
     * requires per-worker GLMContext or a copy-on-write state model. */
    GLMState *savedActiveState = glm_ctx->active_state;
    /* Bug B4: Save _activeState too — restoreStateForBatch sets
     * _activeState = glm_ctx->active_state (= worker->workerState), which
     * would dangle into freed heap after this function returns. */
    GLMState *savedIvarActiveState = _activeState;

    /* Bug 2: Only redirect if we have a valid workerState; otherwise
     * encode on live ctx->state (sequential fallback). */
    if (worker->workerState) {
        glm_ctx->active_state = worker->workerState;
    }

    MGLBatchPath scheduledPath = MGL_BATCH_PATH_DIRECT;

    /* Bug B8: @try/@finally guarantees active_state and _activeState
     * restoration even if Metal validation or issue* throws an NSException.
     * Without this, mtlFlushDrawBuffer catches/logs and returns with
     * active_state still pointing at the worker's (soon-freed) state. */
    @try {
        [self restoreStateForBatch:batch context:glm_ctx savedState:savedState];

        /* Parallel group batches share the same FBO — clear DIRTY_FBO and
         * DIRTY_STATE to prevent checkBatchShouldExecute from triggering
         * rotateRenderEncoderForCurrentFramebufferLocked or newRenderEncoderLocked,
         * which would destroy the parallel sub-encoder and assert. */
        glm_ctx->active_state->dirty_bits &= ~(DIRTY_FBO | DIRTY_STATE);

        /* Sync render-pass metadata ivars to match the batch's restored FBO.
         * endRenderEncodingLocked (called before creating the parallel encoder)
         * nulls these ivars, but the parallel sub-encoder reuses the saved
         * _renderPassDescriptor which was built for this FBO.  Without this
         * sync, ensureCurrentRenderPassMatchesFramebufferForDraw sees a NULL
         * _renderPassFramebuffer vs the batch's FBO, falsely reports a
         * mismatch, and calls newRenderEncoder — which asserts on AGX because
         * the parallel sub-encoder is still active on the command buffer.
         * This mirrors the ivar sync in newRenderEncoderLocked. */
        _renderPassFramebuffer = glm_ctx->active_state->framebuffer;
        _renderPassFramebufferName = _renderPassFramebuffer ? _renderPassFramebuffer->name : 0u;
        _renderPassDrawBuffer = glm_ctx->active_state->draw_buffer;
        _renderPassDrawBufferCount = mglMetalDrawBufferCount(glm_ctx);
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            _renderPassDrawBuffers[i] = (i < _renderPassDrawBufferCount)
                ? mglMetalDrawBufferAt(glm_ctx, (GLuint)i)
                : GL_NONE;
        }

        /* checkBatchShouldExecute calls processGLStateLocked which syncs GL state
         * to _currentRenderEncoder + dedup ivars.  It also handles FBO rotation
         * and rasterization-empty culling. */
        GLenum replayError = GL_NO_ERROR;
        uint32_t skippedCommands = 0;
        if (![self checkBatchShouldExecute:batch
                                   context:glm_ctx
                                   flushId:flushId
                                batchIndex:batchIndex
                               replayError:&replayError
                           skippedCommands:&skippedCommands]) {
            [self saveDedupStateToWorker:worker];
            return MGL_BATCH_PATH_DIRECT;
        }

        scheduledPath = [self scheduleDrawBatch:batch context:glm_ctx];
        switch (scheduledPath) {
            case MGL_BATCH_PATH_STREAM_MERGE:
                [self traceReplayBatch:batch
                               context:glm_ctx
                               flushId:flushId
                            batchIndex:batchIndex
                                 phase:"PARALLEL_ISSUE_STREAM_MERGE"];
                [self issueStreamMergedBatch:batch context:glm_ctx];
                break;
            case MGL_BATCH_PATH_MDI:
                [self traceReplayBatch:batch
                               context:glm_ctx
                               flushId:flushId
                            batchIndex:batchIndex
                                 phase:"PARALLEL_ISSUE_MDI"];
                [self issueMDIBatch:batch context:glm_ctx];
                break;
            case MGL_BATCH_PATH_ICB:
                [self traceReplayBatch:batch
                               context:glm_ctx
                               flushId:flushId
                            batchIndex:batchIndex
                                 phase:"PARALLEL_ISSUE_ICB"];
                [self issueIndirectCommandBufferBatch:batch context:glm_ctx];
                break;
            default:
                [self traceReplayBatch:batch
                               context:glm_ctx
                               flushId:flushId
                            batchIndex:batchIndex
                                 phase:"PARALLEL_ISSUE_DIRECT"];
                [self issueDirectBatch:batch context:glm_ctx];
                break;
        }

        [self recordBatchCommandStats:batch context:glm_ctx];

        /* Capture post-batch dedup state back to the worker. */
        [self saveDedupStateToWorker:worker];

        if (executedOut) *executedOut = YES;
    } @finally {
        /* Restore both state proxies to their pre-redirect values. */
        glm_ctx->active_state = savedActiveState;
        _activeState = savedIvarActiveState;
        /* Bug D7: Free heap-allocated workerState.
         * Bug 2: Only free if we allocated it (malloc-fail fallback leaves
         * workerState NULL and uses live state instead). */
        if (workerStateAllocated && worker->workerState) {
            free(worker->workerState);
            worker->workerState = NULL;
        }
    }

    return scheduledPath;
}

- (void)recordLastBoundVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) {
        return;
    }
    _lastBoundVertexBuffers[index].buffer = buffer;
    _lastBoundVertexBuffers[index].offset = offset;
}

- (void)recordLastBoundFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) {
        return;
    }
    _lastBoundFragmentBuffers[index].buffer = buffer;
    _lastBoundFragmentBuffers[index].offset = offset;
}

- (void)invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) {
        return;
    }
    _lastBoundVertexBuffers[index].buffer = nil;
    _lastBoundVertexBuffers[index].offset = (NSUInteger)-1;
}

- (void)invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) {
        return;
    }
    _lastBoundFragmentBuffers[index].buffer = nil;
    _lastBoundFragmentBuffers[index].offset = (NSUInteger)-1;
}

- (void)setVertexTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
    if (!_currentRenderEncoder || index >= TEXTURE_UNITS) {
        return;
    }
    if (!_lastBoundValid || _lastBoundVertexTextures[index] != texture) {
        [_currentRenderEncoder setVertexTexture:texture atIndex:index];
        _lastBoundVertexTextures[index] = texture;
    }
}

- (void)setFragmentTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
    if (!_currentRenderEncoder || index >= TEXTURE_UNITS) {
        return;
    }
    if (!_lastBoundValid || _lastBoundFragmentTextures[index] != texture) {
        [_currentRenderEncoder setFragmentTexture:texture atIndex:index];
        _lastBoundFragmentTextures[index] = texture;
    }
}

- (void)setVertexSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
    if (!_currentRenderEncoder || index >= TEXTURE_UNITS) {
        return;
    }
    if (!_lastBoundValid || _lastBoundVertexSamplers[index] != sampler) {
        [_currentRenderEncoder setVertexSamplerState:sampler atIndex:index];
        _lastBoundVertexSamplers[index] = sampler;
    }
}

- (void)setFragmentSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
    if (!_currentRenderEncoder || index >= TEXTURE_UNITS) {
        return;
    }
    if (!_lastBoundValid || _lastBoundFragmentSamplers[index] != sampler) {
        [_currentRenderEncoder setFragmentSamplerState:sampler atIndex:index];
        _lastBoundFragmentSamplers[index] = sampler;
    }
}

- (void)setViewportIfNeeded:(MTLViewport)viewport
{
    if (!_currentRenderEncoder) {
        return;
    }
    if (!_lastBoundValid ||
        _lastViewport.originX != viewport.originX ||
        _lastViewport.originY != viewport.originY ||
        _lastViewport.width != viewport.width ||
        _lastViewport.height != viewport.height ||
        _lastViewport.znear != viewport.znear ||
        _lastViewport.zfar != viewport.zfar) {
        [_currentRenderEncoder setViewport:viewport];
        _lastViewport = viewport;
    }
}

- (void)setScissorRectIfNeeded:(MTLScissorRect)rect
{
    if (!_currentRenderEncoder) {
        return;
    }
    if (!_lastBoundValid ||
        _lastScissorRect.x != rect.x ||
        _lastScissorRect.y != rect.y ||
        _lastScissorRect.width != rect.width ||
        _lastScissorRect.height != rect.height) {
        [_currentRenderEncoder setScissorRect:rect];
        _lastScissorRect = rect;
    }
}

- (void)setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode
{
    if (!_currentRenderEncoder) {
        return;
    }
    if (!_lastBoundValid || _lastTriangleFillMode != mode) {
        [_currentRenderEncoder setTriangleFillMode:mode];
        _lastTriangleFillMode = mode;
    }
}

- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx
{
    GLMState *state = MGL_STATE(glm_ctx);
    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->vertex_buffer_map_list]);
    RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->fragment_buffer_map_list]);
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self bindBufferSizeConstantsForRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);
    RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-active-texture-bind"]);
    if (![self bindTexturesToCurrentRenderEncoder]) {
        RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-sampled-texture-bind"]);
        RETURN_FALSE_ON_FAILURE([self bindTexturesToCurrentRenderEncoder]);
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
    MGLFragmentTextureTraceBinding *earlyFs0 = &_fragmentTextureTraceBindings[0];
    MGLFragmentTextureTraceBinding *earlyFs1 = &_fragmentTextureTraceBindings[1];
    MGLFragmentTextureTraceBinding *earlyFs2 = &_fragmentTextureTraceBindings[2];
    MGLFragmentTextureTraceBinding *earlyFs3 = &_fragmentTextureTraceBindings[3];
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
    id<MTLTexture> rpColor0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> rpDepth = _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.texture : nil;
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
                snapshot ? (unsigned)snapshot->var.current_program : 0u,
                (unsigned)snapshotFBOName,
                snapshot ? snapshot->vao : NULL,
                (unsigned)currentProgramKey,
                (unsigned)glm_ctx->active_state->var.current_program,
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
                _currentRenderEncoder,
                _pipelineState,
                (unsigned)_renderPassFramebufferName,
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
    MGLFragmentTextureTraceBinding *fs0 = &_fragmentTextureTraceBindings[0];
    MGLFragmentTextureTraceBinding *fs1 = &_fragmentTextureTraceBindings[1];
    MGLFragmentTextureTraceBinding *fs2 = &_fragmentTextureTraceBindings[2];
    MGLFragmentTextureTraceBinding *fs3 = &_fragmentTextureTraceBindings[3];
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
    id<MTLTexture> rpColor0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> rpDepth = _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.texture : nil;
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
                _currentRenderEncoder,
                _pipelineState,
                (unsigned)fboName,
                (unsigned)_renderPassFramebufferName,
                rpColor0,
                rpDepth,
                (unsigned long)(rpColor0 ? rpColor0.width : 0),
                (unsigned long)(rpColor0 ? rpColor0.height : 0),
                (unsigned long)(rpDepth ? rpDepth.width : 0),
                (unsigned long)(rpDepth ? rpDepth.height : 0),
                mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare),
                mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare),
                mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare),
                mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare),
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
                    (unsigned)_pipelineProgramName,
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
    ctx = glm_ctx;

    /* P2-2: This method is reachable from both locked callers
     * (mtlSwapBuffersLocked:, flushCommandBufferLocked:) and unlocked callers
     * (mtlFlushDrawBuffer via mgl_metal_bridge.m, mtlInvalidateRenderPass:).
     * It calls Locked methods (endRenderEncodingLocked, newCommandBufferLocked,
     * newRenderEncoderLocked) which assume the Metal lock is held.  Acquire
     * METAL_LOCK here unconditionally — NSRecursiveLock makes re-entrant
     * acquisition from already-locked callers a no-op. */
    METAL_LOCK();

    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cb->batch_count == 0) {
        METAL_UNLOCK();
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

    /* Stage 5.1: compute parallel groups (runs of consecutive, non-empty
     * batches sharing the same FBO). The replay loop still runs sequentially;
     * this only instruments the grouping so it can be observed in
     * MGL_PERF_SUMMARY. A later Stage 5.3 will actually parallelize within
     * these groups. Groups are pure metadata over the command buffer. */
    MGLParallelGroup parallelGroups[MGL_MAX_PARALLEL_GROUPS];
    uint32_t parallelGroupCount = mglComputeParallelGroups(cb, parallelGroups,
                                                            MGL_MAX_PARALLEL_GROUPS);
    uint32_t parallelGroupBatches = 0u;
    uint32_t largestParallelGroup = 0u;
    for (uint32_t g = 0u; g < parallelGroupCount; g++) {
        parallelGroupBatches += parallelGroups[g].batch_count;
        if (parallelGroups[g].batch_count > largestParallelGroup) {
            largestParallelGroup = parallelGroups[g].batch_count;
        }
    }
    if (parallelGroupCount > 0u) {
        MGL_PERF_ADD(g_mglParallelGroupsSinceSwap, parallelGroupCount);
        MGL_PERF_ADD(g_mglParallelGroupBatchesSinceSwap, parallelGroupBatches);
        if (largestParallelGroup > MGL_FRAME_LOAD(g_mglLargestParallelGroupSinceSwap)) {
            MGL_FRAME_STORE(g_mglLargestParallelGroupSinceSwap, largestParallelGroup);
        }
        /* Stage 5.3: count batches in groups with ≥2 members — these are
         * parallel-encode candidates.  When MGL_PARALLEL_ENCODE=1 and the
         * processGLStateLocked parameterization is complete, these batches
         * will be encoded on separate sub-encoders. */
        uint32_t eligibleBatches = 0u;
        for (uint32_t g = 0u; g < parallelGroupCount; g++) {
            if (parallelGroups[g].batch_count >= 2u) {
                eligibleBatches += parallelGroups[g].batch_count;
            }
        }
        if (eligibleBatches > 0u) {
            MGL_PERF_ADD(g_mglParallelEncodeEligibleBatchesSinceSwap, eligibleBatches);
        }
    }

    BOOL useParallelEncode = [self parallelEncodeEnabled] && (largestParallelGroup >= 2u);
    if (useParallelEncode && traceFlush) {
        MGLTraceNSLog(@"MGL TRACE parallelEncode ENABLED groups=%u eligibleBatches=%u",
                      parallelGroupCount, largestParallelGroup);
    }

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

            /* Stage 5.3 Step 4: Parallel encode via MTLParallelRenderCommandEncoder.
             *
             * When MGL_PARALLEL_ENCODE=1 and the current batch starts a
             * parallel group with ≥2 members, create a parallel render
             * encoder with 2 sub-encoders.  Each sub-encoder gets its own
             * batch, encoded sequentially on the calling thread (Step 4
             * validates the parallel encoder API and execution order;
             * multi-threaded dispatch is a future enhancement).
             *
             * Sub-encoder execution order = creation order (Apple docs),
             * which matches GL submission order: batch[b] before batch[b+1].
             *
             * The current render encoder must be ended first because the
             * parallel encoder needs a fresh render pass start (load/store
             * actions are owned by the parallel encoder, not sub-encoders). */
            if (useParallelEncode) {
                int groupIdx = -1;
                for (uint32_t g = 0; g < parallelGroupCount; g++) {
                    if (parallelGroups[g].start_batch == b &&
                        parallelGroups[g].batch_count >= 2) {
                        groupIdx = (int)g;
                        break;
                    }
                }
                if (groupIdx >= 0 && b + 1 < cb->batch_count &&
                    cb->batches[b + 1].command_count > 0) {
                    MGLDrawBatch *batch1 = &cb->batches[b + 1];

                    /* Save the current render pass descriptor before ending
                     * the encoder — the parallel encoder reuses it. */
                    MTLRenderPassDescriptor *parallelDesc = _renderPassDescriptor;

                    /* End the current render encoder and commit the command
                     * buffer.  The parallel encoder needs a fresh render pass
                     * start (load/store actions are owned by the parallel
                     * encoder, not sub-encoders).  endRenderEncodingLocked
                     * also performs GL sampled-RT Y-flip copies that may be
                     * pending for the ended pass. */
                    [self endRenderEncodingLocked];

                    /* Save dedup state before entering parallel mode.
                     * Zero-init first: workerState is a pointer field that
                     * saveDedupStateToWorker does not touch — garbage would
                     * cause encodeBatchForParallelWorker to skip malloc and
                     * treat the garbage as a live GLMState*. */
                    MGLWorkerContext preGroupState;
                    memset(&preGroupState, 0, sizeof(preGroupState));
                    [self saveDedupStateToWorker:&preGroupState];

                    if (_currentCommandBuffer) {
                        [self commitCommandBufferWithAGXRecovery:_currentCommandBuffer];
                        _currentCommandBuffer = nil;
                    }
                    if (![self newCommandBufferLocked]) {
                        NSLog(@"MGL WARNING: newCommandBufferLocked failed before parallel encode, "
                              "falling back to sequential for batches %u-%u", b, b + 1);
                        [self loadDedupStateFromWorker:&preGroupState];
                        if (![self newRenderEncoderLocked]) {
                            continue;
                        }
                        [self restoreStateForBatch:batch context:glm_ctx savedState:&savedState];
                        goto sequentialBatch;
                    }

                    id<MTLParallelRenderCommandEncoder> parallelEncoder =
                        [_currentCommandBuffer parallelRenderCommandEncoderWithDescriptor:parallelDesc];
                    if (!parallelEncoder) {
                        NSLog(@"MGL WARNING: parallelRenderCommandEncoder failed, "
                              "falling back to sequential for batches %u-%u", b, b + 1);
                        [self loadDedupStateFromWorker:&preGroupState];
                        if (![self newRenderEncoderLocked]) {
                            continue;
                        }
                        [self restoreStateForBatch:batch context:glm_ctx savedState:&savedState];
                        goto sequentialBatch;
                    }
                    parallelEncoder.label = @"MGL Parallel Render Encoder";

                    /* Stage 5.3 Step 5: Activate parallel-encode mode so
                     * processGLStateLocked (called inside
                     * encodeBatchForParallelWorker → checkBatchShouldExecute)
                     * skips encoder reconstruction paths that would destroy
                     * the sub-encoder. */
                    _parallelEncodeActive = YES;

                    /* Sub-encoder lifecycle: Metal requires that each
                     * sub-encoder created from a parallel encoder be
                     * endEncoding'd BEFORE the next sub-encoder is created.
                     * Creating two sub-encoders concurrently triggers
                     * "A command encoder is already encoding to this command
                     * buffer".  The correct pattern is:
                     *   create sub0 → encode batch[b] → endEncoding
                     *   create sub1 → encode batch[b+1] → endEncoding
                     *   parallelEncoder endEncoding */

                    /* --- Worker 0: create subEncoder0, encode batch[b], end --- */
                    MGL_PERF_INC(g_mglParallelEncodeEligibleBatchesSinceSwap);
                    id<MTLRenderCommandEncoder> subEncoder0 =
                        [parallelEncoder renderCommandEncoder];
                    if (!subEncoder0) {
                        NSLog(@"MGL WARNING: sub-encoder 0 creation failed");
                        [parallelEncoder endEncoding];
                        [self loadDedupStateFromWorker:&preGroupState];
                        if (![self newRenderEncoderLocked]) {
                            continue;
                        }
                        [self restoreStateForBatch:batch context:glm_ctx savedState:&savedState];
                        goto sequentialBatch;
                    }
                    subEncoder0.label = @"MGL Sub-encoder 0";

                    _currentRenderEncoder = subEncoder0;
                    MGLWorkerContext worker0 = preGroupState;
                    /* loadDedupStateFromWorker (called inside
                     * encodeBatchForParallelWorker) overwrites
                     * _currentRenderEncoder with worker->encoder.  preGroupState
                     * was saved after endRenderEncodingLocked, so its encoder
                     * is nil.  Set worker0.encoder to the sub-encoder so the
                     * load restores the correct encoder. */
                    worker0.encoder = subEncoder0;
                    BOOL exec0 = NO;
                    MGLBatchPath path0 =
                        [self encodeBatchForParallelWorker:&worker0
                                                    batch:batch
                                                  context:glm_ctx
                                                  flushId:flushHit
                                               batchIndex:b
                                               savedState:&savedState
                                                 executed:&exec0];
                    [subEncoder0 endEncoding];

                    if (exec0) {
                        switch (path0) {
                            case MGL_BATCH_PATH_STREAM_MERGE:
                                streamMergedBatchCount++;
                                streamMergedCommandCount += batch->command_count;
                                MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap);
                                MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap,
                                             batch->command_count);
                                break;
                            case MGL_BATCH_PATH_MDI:
                                mdiBatchCount++;
                                mdiCommandCount += batch->command_count;
                                break;
                            case MGL_BATCH_PATH_ICB:
                                icbBatchCount++;
                                icbCommandCount += batch->command_count;
                                break;
                            default:
                                directBatchCount++;
                                directCommandCount += batch->command_count;
                                MGL_PERF_INC(g_mglBatchesDirectSinceSwap);
                                MGL_PERF_ADD(g_mglDrawDirectSinceSwap,
                                             batch->command_count);
                                break;
                        }
                    }

                    /* --- Worker 1: create subEncoder1, encode batch[b+1], end --- */
                    id<MTLRenderCommandEncoder> subEncoder1 =
                        [parallelEncoder renderCommandEncoder];
                    if (!subEncoder1) {
                        NSLog(@"MGL WARNING: sub-encoder 1 creation failed");
                        [parallelEncoder endEncoding];
                        [self loadDedupStateFromWorker:&preGroupState];
                        if (![self newRenderEncoderLocked]) {
                            continue;
                        }
                        [self restoreStateForBatch:batch context:glm_ctx savedState:&savedState];
                        goto sequentialBatch;
                    }
                    subEncoder1.label = @"MGL Sub-encoder 1";

                    _currentRenderEncoder = subEncoder1;
                    MGLWorkerContext worker1 = preGroupState;
                    worker1.encoder = subEncoder1;
                    BOOL exec1 = NO;
                    MGLBatchPath path1 =
                        [self encodeBatchForParallelWorker:&worker1
                                                    batch:batch1
                                                  context:glm_ctx
                                                  flushId:flushHit
                                               batchIndex:b + 1
                                               savedState:&savedState
                                                 executed:&exec1];
                    [subEncoder1 endEncoding];

                    if (exec1) {
                        switch (path1) {
                            case MGL_BATCH_PATH_STREAM_MERGE:
                                streamMergedBatchCount++;
                                streamMergedCommandCount += batch1->command_count;
                                MGL_PERF_INC(g_mglBatchesStreamMergedSinceSwap);
                                MGL_PERF_ADD(g_mglDrawStreamMergedSinceSwap,
                                             batch1->command_count);
                                break;
                            case MGL_BATCH_PATH_MDI:
                                mdiBatchCount++;
                                mdiCommandCount += batch1->command_count;
                                break;
                            case MGL_BATCH_PATH_ICB:
                                icbBatchCount++;
                                icbCommandCount += batch1->command_count;
                                break;
                            default:
                                directBatchCount++;
                                directCommandCount += batch1->command_count;
                                MGL_PERF_INC(g_mglBatchesDirectSinceSwap);
                                MGL_PERF_ADD(g_mglDrawDirectSinceSwap,
                                             batch1->command_count);
                                break;
                        }
                    }

                    /* End the parallel encoder — this finalizes the pass
                     * and triggers load/store actions. */
                    [parallelEncoder endEncoding];

                    /* Stage 5.3 Step 5: Deactivate parallel-encode mode —
                     * subsequent sequential batches resume normal
                     * processGLStateLocked encoder management. */
                    _parallelEncodeActive = NO;

                    /* Restore dedup state from worker1 (last encoded batch).
                     * The encoder is now ended, so invalidate to force
                     * re-bind on the next sequential batch. */
                    [self loadDedupStateFromWorker:&worker1];
                    [self invalidateLastBoundState];
                    _currentRenderEncoder = nil;

                    if (traceFlush) {
                        MGLTraceNSLog(@"MGL TRACE parallelEncode "
                                      "batch[%u]+batch[%u] exec0=%d path0=%d "
                                      "exec1=%d path1=%d",
                                      b, b + 1, exec0, (int)path0,
                                      exec1, (int)path1);
                    }

                    b++; /* Skip batch[b+1] — already processed. */
                    /* Parallel branch ends the encoder and invalidates dedup;
                     * force a full restore on the next sequential batch. */
                    lastKeyValid = NO;
                    lastExecuteOk = NO;
                    continue;
                }
            }

        sequentialBatch: {
            /* Same-key skip: only when the previous sequential batch fully
             * executed, the same encoder is still open with valid bind
             * cache, and keys match. Never skip across FBO/pass changes. */
            BOOL canSkipRestore = NO;
            if (_skipSameKeyRestoreEnabled &&
                lastKeyValid &&
                lastExecuteOk &&
                _currentRenderEncoder != nil &&
                _lastBoundValid &&
                !_parallelEncodeActive &&
                mglStateKeysEqual(&batch->key, &lastKey) &&
                [self currentRenderPassMatchesCurrentFramebuffer]) {
                canSkipRestore = YES;
            } else if (!_skipSameKeyRestoreEnabled &&
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
                    [self issueStreamMergedBatch:batch context:glm_ctx];
                    break;
                case MGL_BATCH_PATH_MDI:
                    mdiBatchCount++;
                    mdiCommandCount += batch->command_count;
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_MDI"];
                    [self issueMDIBatch:batch context:glm_ctx];
                    break;
                case MGL_BATCH_PATH_ICB:
                    icbBatchCount++;
                    icbCommandCount += batch->command_count;
                    [self traceReplayBatch:batch
                                   context:glm_ctx
                                   flushId:flushHit
                                batchIndex:b
                                     phase:"ISSUE_ICB"];
                    [self issueIndirectCommandBufferBatch:batch context:glm_ctx];
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
                    [self issueDirectBatch:batch context:glm_ctx];
                    break;
            }

            [self recordBatchCommandStats:batch context:glm_ctx];
        } /* sequentialBatch */
        }
    }
    _traceReplayFlushId = 0;
    _traceReplayBatchIndex = 0;

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
    [self teardownBatchReplayForContext:glm_ctx savedState:&savedState
                            savedError:savedError replayError:replayError];
    METAL_UNLOCK();
}

- (MGLBatchPath)scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    if (!batch || batch->command_count == 0) {
        return MGL_BATCH_PATH_DIRECT;
    }

    if (batch->stream_merged) {
        return MGL_BATCH_PATH_STREAM_MERGE;
    }

    if (mglEnvFlagEnabled("MGL_ENABLE_ICB_BATCH") &&
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
     * _activeState points to ctx->state (which now holds the snapshot data).
     * In Stage 5.3 this will point to a per-worker GLMState copy instead. */
    _activeState = glm_ctx->active_state;
    glm_ctx->active_state->dirty_bits = 0;

    static const GLuint kMGLFullReplayDirtyBits =
        (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_RENDER_STATE |
         DIRTY_TEX_BINDING | DIRTY_TEX | DIRTY_TEX_PARAM |
         DIRTY_SAMPLER | DIRTY_ALPHA_STATE | DIRTY_BUFFER |
         DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE);

    GLuint replayDirtyBits = kMGLFullReplayDirtyBits;
    BOOL prevKeyValid = (prevKey != NULL);
    BOOL canDelta = _dirtyKeyDeltaEnabled &&
                    prevKeyValid &&
                    _currentRenderEncoder != nil &&
                    _lastBoundValid;

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
        (_currentRenderEncoder &&
         ![self currentRenderPassMatchesCurrentFramebuffer])) {
        replayDirtyBits |= DIRTY_FBO;
    }
    /* Empty encoder cannot delta-bind — force full domains. */
    if (_currentRenderEncoder == nil || !_lastBoundValid) {
        replayDirtyBits = kMGLFullReplayDirtyBits |
                          ((replayDirtyBits & DIRTY_FBO) ? DIRTY_FBO : 0);
        if ((replayFBO && (replayFBO->dirty_bits & DIRTY_FBO_BINDING)) ||
            (_currentRenderEncoder &&
             ![self currentRenderPassMatchesCurrentFramebuffer])) {
            replayDirtyBits |= DIRTY_FBO;
        } else if (prevKeyValid && prevKey->fbo_name != batch->key.fbo_name) {
            replayDirtyBits |= DIRTY_FBO;
        }
    }
    glm_ctx->active_state->dirty_bits |= replayDirtyBits;
    MGL_SIGNPOST_END(RestoreStateForBatch);
}

- (void)teardownBatchReplayForContext:(GLMContext)glm_ctx
                           savedState:(const GLMState *)savedState
                           savedError:(GLenum)savedError
                          replayError:(GLenum)replayError
{
    /* Deactivate snapshot-based state access — revert to live ctx->state. */
    _activeState = nil;
    /* Ensure active_state points back to the embedded state (not a worker
     * copy) before restoring the saved live state. */
    glm_ctx->active_state = &glm_ctx->state;
    mglResetCommandBufferForContext(glm_ctx, &glm_ctx->draw_command_buffer);
    /* Task 4: Reset the snapshot arena now that all batch replay is complete
     * and mglResetCommandBufferForContext has cleared all batch references.
     * This is the safe point — no worker/encoder is accessing snapshot data. */
    if (_arenaSnapshotEnabled) {
        mglResetBatchArena(&_batchArena);
    }
    memcpy(glm_ctx->active_state, savedState, sizeof(GLMState));
    /* Replay has fully applied all pending state to Metal encoders.
     * Clear dirty bits so the next defer-path draw starts clean instead of
     * inheriting the stale DIRTY_ALL from savedState. */
    glm_ctx->active_state->dirty_bits = 0;
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
    _traceReplayFlushId  = flushId;
    _traceReplayBatchIndex = batchIndex;
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
{
    if (!batch || !batch->stream_merged || batch->stream_index_count == 0) {
        if (batch && batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_unsupported_primitive"];
        }
        [self issueDirectBatch:batch context:glm_ctx];
        return;
    }

    if (!mglEnvFlagEnabled("MGL_DISABLE_MDI")) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
                        commandIndex:0
                               phase:"ISSUE"
                              reason:"stream_merge_to_mdi"];
        }
        if ([self issueStreamMergedMDIBatch:batch context:glm_ctx]) {
            return;
        }
    }

    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;
    if (!indexBuffer || ![self processBuffer:indexBuffer]) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_index_buffer"];
        }
        [self issueDirectBatch:batch context:glm_ctx];
        return;
    }

    id<MTLBuffer> mtlIndexBuffer = (__bridge id<MTLBuffer>)(indexBuffer->data.mtl_data);
    if (!mtlIndexBuffer) {
        if (batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"stream_no_mtl_index"];
        }
        [self issueDirectBatch:batch context:glm_ctx];
        return;
    }

    MGLDrawCommand *firstCmd = &batch->commands[0];
    MTLPrimitiveType primType = (MTLPrimitiveType)batch->key.primitive_type;

    [_currentRenderEncoder drawIndexedPrimitives:primType
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
                     flushId:_traceReplayFlushId
                  batchIndex:_traceReplayBatchIndex
                commandIndex:0
                       phase:"SUBMIT"
                      reason:"stream_direct_merged"];
}

- (BOOL)issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    if (!batch || !batch->stream_merged || batch->command_count == 0 ||
        batch->stream_index_count == 0 || !_currentRenderEncoder) {
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
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
        [_currentRenderEncoder drawIndexedPrimitives:primType
                                           indexType:MTLIndexTypeUInt32
                                         indexBuffer:mtlIndexBuffer
                                   indexBufferOffset:(NSUInteger)cmd->indexBufferOffset
                                      indirectBuffer:indirectArgsBuffer
                                indirectBufferOffset:indirectArgsOffset + (i * argSize)];
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_traceReplayFlushId
                      batchIndex:_traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"stream_mdi_indexed"];
    }

    return YES;
}

- (BOOL)issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
{
    if (!batch || batch->command_count == 0 || !_device || !_currentRenderEncoder) {
        if (batch && batch->command_count > 0) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
                         flushId:_traceReplayFlushId
                      batchIndex:_traceReplayBatchIndex
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
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
                        commandIndex:0
                               phase:"FALLBACK"
                              reason:"icb_create_exception"];
            return NO;
        }
        if (!icb) {
            [self traceReplayCommand:batch
                             command:&batch->commands[0]
                             context:glm_ctx
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
                                     flushId:_traceReplayFlushId
                                  batchIndex:_traceReplayBatchIndex
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
                                     flushId:_traceReplayFlushId
                                  batchIndex:_traceReplayBatchIndex
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
                                     flushId:_traceReplayFlushId
                                  batchIndex:_traceReplayBatchIndex
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
                                     flushId:_traceReplayFlushId
                                  batchIndex:_traceReplayBatchIndex
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
                [_currentRenderEncoder useResource:drawIndexBuffer
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
                                     flushId:_traceReplayFlushId
                                  batchIndex:_traceReplayBatchIndex
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

        [_currentRenderEncoder useResource:icb
                                     usage:MTLResourceUsageRead
                                    stages:MTLRenderStageVertex];
        [_currentRenderEncoder executeCommandsInBuffer:icb
                                             withRange:NSMakeRange(0, (NSUInteger)batch->command_count)];
        for (uint32_t i = 0; i < batch->command_count; i++) {
            [self traceReplayCommand:batch
                             command:&batch->commands[i]
                             context:glm_ctx
                             flushId:_traceReplayFlushId
                          batchIndex:_traceReplayBatchIndex
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
    if (offsetOut) {
        *offsetOut = 0;
    }
    if (!_device || !_currentCommandBuffer || length == 0) {
        return nil;
    }

    const NSUInteger alignment = 256u;
    NSUInteger alignedOffset = (_mdiArgsScratchOffset + (alignment - 1u)) & ~(alignment - 1u);
    if (alignedOffset < _mdiArgsScratchOffset ||
        length > (NSUIntegerMax - alignedOffset)) {
        return nil;
    }

    NSUInteger requiredBytes = alignedOffset + length;
    if (!_mdiArgsScratchBuffer || requiredBytes > _mdiArgsScratchCapacity) {
        NSUInteger newCapacity = _mdiArgsScratchCapacity ? (_mdiArgsScratchCapacity * 2u) : (64u * 1024u);
        if (newCapacity < length) {
            newCapacity = length;
        }
        if (newCapacity < requiredBytes) {
            newCapacity = requiredBytes;
        }
        if (newCapacity < _mdiArgsScratchCapacity) {
            return nil;
        }

        id<MTLBuffer> newBuffer = [_device newBufferWithLength:newCapacity
                                                       options:MTLResourceStorageModeShared];
        if (!newBuffer) {
            return nil;
        }
        _mdiArgsScratchBuffer = newBuffer;
        _mdiArgsScratchCapacity = newCapacity;
        alignedOffset = 0;
        requiredBytes = length;
    }

    _mdiArgsScratchOffset = requiredBytes;
    if (offsetOut) {
        *offsetOut = alignedOffset;
    }
    return _mdiArgsScratchBuffer;
}

@end
