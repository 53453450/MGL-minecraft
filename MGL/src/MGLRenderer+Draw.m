// MGLRenderer+Draw.m
// Draw command encoding methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"

/* === C helpers used by Draw and Batch methods === */
/* mglRendererProgramHasSampledResourceNamed is non-static so
 * MGLRenderer+Batch.m can also call it.  Declared in MGLRenderer+Draw_Private.h. */

bool mglRendererProgramHasSampledResourceNamed(Program *program, const char *name)
{
    if (!program || !name) {
        return false;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (int resType = 0; resType < _MAX_SPIRV_RES; resType++) {
            SpirvResourceList *resources = &program->spirv_resources_list[stage][resType];
            for (GLuint i = 0; resources->list && i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (res->name &&
                    strcmp(res->name, name) == 0 &&
                    mglRendererResourceLooksSamplerLike(res, resType)) {
                    return true;
                }
            }
        }
    }

    return false;
}


@implementation MGLRenderer (Draw)

-(void) mtlDrawArrays: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count
{
    METAL_LOCK();
    [self mtlDrawArraysLocked:ctx mode:mode first:first count:count];
    METAL_UNLOCK();
}

-(void) mtlDrawArraysLocked: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count
{
    self->ctx = ctx;

    static uint64_t s_drawArraysCallCount = 0;
    static double s_drawArraysLastCallTime = 0.0;
    static uint64_t s_drawArraysLastCallCount = 0;
    uint64_t drawCall = ++s_drawArraysCallCount;
    double drawStartSeconds = mglNowSeconds();
    bool traceDraw = mglShouldTraceCall(drawCall);
    mglLogLoopHeartbeat("drawArrays.loop",
                        drawCall,
                        drawStartSeconds,
                        &s_drawArraysLastCallTime,
                        &s_drawArraysLastCallCount,
                        0.25);

    MTLPrimitiveType primitiveType;
    static uint64_t process_state_fail_count = 0;
    static uint64_t no_render_encoder_count = 0;

    // AGGRESSIVE MEMORY SAFETY: Immediate validation before any Metal operations
    if (!ctx || ((uintptr_t)ctx < 0x1000)) {
        NSLog(@"MGL ERROR: mtlDrawArrays - Invalid context detected, aborting");
        return; // Early return to prevent crash
    }

    /* GL_PATCHES early handling: TCS/TES are dispatched as Metal compute
     * kernels because the render pipeline only handles VS+FS stages. */
    if ([self handleTessellationPatchDrawIfNeeded:ctx
                                             mode:&mode
                                            first:first
                                            count:count
                                        indexType:0
                                          indices:NULL
                                       baseVertex:0
                                    instanceCount:1
                                     baseInstance:0
                                            label:"drawArrays"]) {
        return;
    }

    if ([self processGLStateLocked: true] == false) {
        process_state_fail_count++;
        MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
        if (process_state_fail_count <= 8 || (process_state_fail_count % 1000) == 0) {
            NSLog(@"MGL ERROR: mtlDrawArrays - processGLState failed, aborting (occurrence=%llu)",
                  (unsigned long long)process_state_fail_count);
        }
        return; // Early return instead of continuing with invalid state
    }
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    BOOL traceLogDraw = mglProgramNeedsTraceLog(activeProgram);
    if (traceLogDraw) {
        VertexArray *drawVAO = mglRendererGetValidatedVAO(ctx, "drawArrays.trace");
        mglTraceLog("DRAW_ARRAYS_BEGIN call=%llu program=%u mode=0x%x first=%d count=%d fbo=%u vao=%p enabled=0x%x viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) drawBuf=0x%x readBuf=0x%x colorMask=%d%d%d%d depth(test=%d write=%d func=0x%x) blend=%d cull=%d cullFace=0x%x frontFace=0x%x",
                    (unsigned long long)drawCall,
                    activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx),
                    (unsigned)mode,
                    (int)first,
                    (int)count,
                    ctx && MGL_STATE(ctx)->framebuffer ? (unsigned)MGL_STATE(ctx)->framebuffer->name : 0u,
                    drawVAO,
                    drawVAO ? (unsigned)drawVAO->enabled_attribs : 0u,
                    (int)MGL_STATE(ctx)->viewport[0],
                    (int)MGL_STATE(ctx)->viewport[1],
                    (int)MGL_STATE(ctx)->viewport[2],
                    (int)MGL_STATE(ctx)->viewport[3],
                    MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
                    (int)MGL_STATE(ctx)->var.scissor_box[0],
                    (int)MGL_STATE(ctx)->var.scissor_box[1],
                    (int)MGL_STATE(ctx)->var.scissor_box[2],
                    (int)MGL_STATE(ctx)->var.scissor_box[3],
                    (unsigned)MGL_STATE(ctx)->draw_buffer,
                    (unsigned)MGL_STATE(ctx)->read_buffer,
                    MGL_STATE(ctx)->var.color_writemask[0][0] ? 1 : 0,
                    MGL_STATE(ctx)->var.color_writemask[0][1] ? 1 : 0,
                    MGL_STATE(ctx)->var.color_writemask[0][2] ? 1 : 0,
                    MGL_STATE(ctx)->var.color_writemask[0][3] ? 1 : 0,
                    MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
                    MGL_STATE(ctx)->var.depth_writemask ? 1 : 0,
                    (unsigned)MGL_STATE(ctx)->var.depth_func,
                    MGL_STATE(ctx)->caps.blend ? 1 : 0,
                    MGL_STATE(ctx)->caps.cull_face ? 1 : 0,
                    (unsigned)MGL_STATE(ctx)->var.cull_face_mode,
                    (unsigned)MGL_STATE(ctx)->var.front_face);
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        if (traceLogDraw) {
            mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=empty_rasterization",
                        (unsigned long long)drawCall,
                        activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
        }
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        if (traceLogDraw) {
            mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=front_and_back_culled",
                        (unsigned long long)drawCall,
                        activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
        }
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    // Additional safety check after processGLState
    if (!_renderPassManager.state->currentRenderEncoder) {
        // One recovery attempt to avoid persistent "No current render encoder" failure loops.
        [self newRenderEncoderLockedWithReason:MGL_ENC_REASON_DRAW];
        if (!_renderPassManager.state->currentRenderEncoder) {
            no_render_encoder_count++;
            if (no_render_encoder_count <= 8 || (no_render_encoder_count % 1000) == 0) {
                NSLog(@"MGL ERROR: mtlDrawArrays - No current render encoder, aborting (occurrence=%llu)",
                      (unsigned long long)no_render_encoder_count);
            }
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=no_render_encoder",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }

        if (!_pipelineCache.state->pipelineState) {
            NSLog(@"MGL ERROR: mtlDrawArrays - No pipeline state after render encoder recovery, aborting draw");
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=no_pipeline_state",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }

        // Guard against Metal validation aborts when emergency-rebinding pipeline after
        // encoder recovery. Only bind when pass attachment formats are compatible.
        MTLPixelFormat rpColor0Format = MTLPixelFormatInvalid;
        MTLPixelFormat rpDepthFormat = MTLPixelFormatInvalid;
        MTLPixelFormat rpStencilFormat = MTLPixelFormatInvalid;
        if (_renderPassManager.state->renderPassDescriptor) {
            id<MTLTexture> rpColor0 = _renderPassManager.state->renderPassDescriptor.colorAttachments[0].texture;
            id<MTLTexture> rpDepth = _renderPassManager.state->renderPassDescriptor.depthAttachment.texture;
            id<MTLTexture> rpStencil = _renderPassManager.state->renderPassDescriptor.stencilAttachment.texture;
            if (rpColor0) rpColor0Format = rpColor0.pixelFormat;
            if (rpDepth) rpDepthFormat = rpDepth.pixelFormat;
            if (rpStencil) rpStencilFormat = rpStencil.pixelFormat;
        }

        BOOL colorMismatch = (_pipelineCache.state->pipelineColor0Format != MTLPixelFormatInvalid &&
                              rpColor0Format != MTLPixelFormatInvalid &&
                              _pipelineCache.state->pipelineColor0Format != rpColor0Format);
        BOOL depthMismatch = (_pipelineCache.state->pipelineDepthFormat != rpDepthFormat);
        BOOL stencilMismatch = (_pipelineCache.state->pipelineStencilFormat != rpStencilFormat);
        if (colorMismatch || depthMismatch || stencilMismatch) {
            NSLog(@"MGL WARNING: mtlDrawArrays recovery skipped pipeline bind due to pass mismatch "
                  "(pipeline c/d/s=%lu/%lu/%lu, pass c/d/s=%lu/%lu/%lu)",
                  (unsigned long)_pipelineCache.state->pipelineColor0Format,
                  (unsigned long)_pipelineCache.state->pipelineDepthFormat,
                  (unsigned long)_pipelineCache.state->pipelineStencilFormat,
                  (unsigned long)rpColor0Format,
                  (unsigned long)rpDepthFormat,
                  (unsigned long)rpStencilFormat);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=pipeline_pass_mismatch pipeline=%lu/%lu/%lu pass=%lu/%lu/%lu",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx),
                            (unsigned long)_pipelineCache.state->pipelineColor0Format,
                            (unsigned long)_pipelineCache.state->pipelineDepthFormat,
                            (unsigned long)_pipelineCache.state->pipelineStencilFormat,
                            (unsigned long)rpColor0Format,
                            (unsigned long)rpDepthFormat,
                            (unsigned long)rpStencilFormat);
            }
            return;
        }

        @try {
            [_renderPassManager.state->currentRenderEncoder setRenderPipelineState:_pipelineCache.state->pipelineState];
            [_bindingSync setLastPipelineState:_pipelineCache.state->pipelineState];
            MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: mtlDrawArrays - setRenderPipelineState failed after recovery: %@", exception);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=set_pipeline_exception",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }
    }

    if (![self validateDrawArraysVertexInputs:ctx
                                         mode:mode
                                        first:first
                                        count:count
                                     drawCall:drawCall]) {
        MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=vertex_validation",
                        (unsigned long long)drawCall,
                        activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
        }
        return;
    }

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    if (polygonModePoint) {
        if (!mglEncodeArrayPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                        _device,
                                        mode,
                                        first,
                                        count,
                                        1u,
                                        0u,
                                        "drawArrays")) {
            MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=polygon_point_encode",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }
    } else if (emulateTriangleFan) {
        if (count < 3) {
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=triangle_fan_too_small",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }

        NSUInteger fanIndexCount = 0u;
        id<MTLBuffer> fanIndexBuffer = mglNewTriangleFanArrayIndexBuffer(_device,
                                                                         (NSUInteger)count,
                                                                         &fanIndexCount);
        if (!fanIndexBuffer || fanIndexCount == 0u) {
            NSLog(@"MGL WARNING: drawArrays triangle fan emulation failed count=%d first=%d",
                  (int)count,
                  (int)first);
            MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=triangle_fan_emulation_failed",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }

        @try {
            [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                              indexCount:fanIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:fanIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1
                                              baseVertex:first
                                            baseInstance:0];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: mtlDrawArrays triangle fan drawIndexedPrimitives failed: %@", exception);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=triangle_fan_exception",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }
    } else if (emulateLineLoop) {
        if (count < 2) {
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=line_loop_too_small",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }

        NSUInteger loopIndexCount = 0u;
        id<MTLBuffer> loopIndexBuffer = mglNewLineLoopArrayIndexBuffer(_device,
                                                                       (NSUInteger)first,
                                                                       (NSUInteger)count,
                                                                       &loopIndexCount);
        if (!loopIndexBuffer || loopIndexCount == 0u) {
            NSLog(@"MGL WARNING: drawArrays line loop emulation failed count=%d first=%d",
                  (int)count,
                  (int)first);
            MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=line_loop_emulation_failed",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }

        @try {
            [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                                              indexCount:loopIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:loopIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1
	                                              baseVertex:0
                                            baseInstance:0];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: mtlDrawArrays line loop drawIndexedPrimitives failed: %@", exception);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=line_loop_exception",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }
    } else if (emulateQuads) {
        if (!mglEncodeArrayQuads(_renderPassManager.state->currentRenderEncoder,
                                 _device,
                                 count,
                                 first,
                                 1u,
                                 0u,
                                 mglPolygonModeLineForDrawMode(ctx, mode),
                                 "drawArrays")) {
            MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=quad_emulation_failed",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
            }
            return;
        }
    } else {
        primitiveType = getMTLPrimitiveType(mode);
        if ((GLuint)primitiveType == 0xFFFFFFFF) {
            NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=unsupported_mode mode=0x%x",
                            (unsigned long long)drawCall,
                            activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx),
                            (unsigned)mode);
            }
            return;
        }

    /* Cull distance emulation: if the active vertex shader uses
     * mgl_CullDistance, bind the vertex buffer and params so the injected
     * shader code can read sibling-vertex cull distance values. */
    if (activeProgram && (activeProgram->mslCacheValid
            ? activeProgram->uses_cull_distance
            : (activeProgram->spirv[_VERTEX_SHADER].msl_str &&
               strstr(activeProgram->spirv[_VERTEX_SHADER].msl_str, "mgl_CullDistance")))) {
        MGLEncodeContext encCtx = { .encoder = _renderPassManager.state->currentRenderEncoder };
        [self bindCullDistanceEmulationBuffers:mode encodeContext:&encCtx];
    }

    @try {
        mglTraceLog("DRAW_ARRAYS_OBJ_SUBMIT call=%llu program=%u mode=0x%x first=%d count=%d encoder=%p pipeline=%p",
                    (unsigned long long)drawCall,
                    activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx),
                    (unsigned)mode,
                    (int)first,
                    (int)count,
                    _renderPassManager.state->currentRenderEncoder,
                    _pipelineCache.state->pipelineState);
        [_renderPassManager.state->currentRenderEncoder drawPrimitives: primitiveType
                                 vertexStart: first
                                 vertexCount: count];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: mtlDrawArrays - drawPrimitives failed: %@", exception);
        // Don't crash, just return gracefully
        if (traceLogDraw) {
            mglTraceLog("DRAW_ARRAYS_SKIP call=%llu program=%u reason=draw_exception",
                        (unsigned long long)drawCall,
                        activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx));
        }
        return;
    }
    }

    if (traceLogDraw) {
        mglTraceLog("DRAW_ARRAYS_SUBMIT call=%llu program=%u mode=0x%x first=%d count=%d encoder=%p pipeline=%p",
                    (unsigned long long)drawCall,
                    activeProgram ? (unsigned)activeProgram->name : (unsigned)mglCurrentRenderProgramKey(ctx),
                    (unsigned)mode,
                    (int)first,
                    (int)count,
                    _renderPassManager.state->currentRenderEncoder,
                    _pipelineCache.state->pipelineState);
    }

    MGL_FRAME_STORE(g_mglLastDrawArraysCall, drawCall);
    [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0)];
    mglLogDrawWithoutSwapWatchdog("arrays",
                                  drawCall,
                                  ctx,
                                  _renderPassManager.state->currentCommandBuffer,
                                  _renderPassManager.state->currentRenderEncoder,
                                  _renderPassManager.state->renderPassDescriptor);

    double drawElapsedMs = (mglNowSeconds() - drawStartSeconds) * 1000.0;
    if (traceDraw || drawElapsedMs >= 16.0) {
        MGLTraceNSLog(@"MGL TRACE drawArrays.end call=%llu mode=0x%x first=%d count=%d elapsed=%.3fms encoder=%p",
              (unsigned long long)drawCall,
              (unsigned)mode,
              (int)first,
              (int)count,
              drawElapsedMs,
              _renderPassManager.state->currentRenderEncoder);
    }
}

-(void) mtlDrawElements: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    METAL_LOCK();
    [self mtlDrawElementsLocked:glm_ctx mode:mode count:count type:type indices:indices];
    METAL_UNLOCK();
}

-(void) mtlDrawElementsLocked: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    ctx = glm_ctx;

    static uint64_t s_drawElementsCallCount = 0;
    static double s_drawElementsLastCallTime = 0.0;
    static uint64_t s_drawElementsLastCallCount = 0;
    static uint64_t s_drawElementsProcessStateFailCount = 0;
    uint64_t drawCall = ++s_drawElementsCallCount;
    double drawStartSeconds = mglNowSeconds();
    bool traceDraw = mglShouldTraceCall(drawCall);
    mglLogLoopHeartbeat("drawElements.loop",
                        drawCall,
                        drawStartSeconds,
                        &s_drawElementsLastCallTime,
                        &s_drawElementsLastCallCount,
                        0.25);

    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;
    GLuint activeProgramName = ctx ? mglCurrentRenderProgramKey(ctx) : 0u;
    Program *drawProgram = NULL;
    Program *drawVertexProgram = NULL;
    Program *drawFragmentProgram = NULL;
    bool drawProgramUsesCloudFaces = false;

    if (traceDraw) {
        MGLTraceNSLog(@"MGL TRACE drawElements.begin call=%llu mode=0x%x count=%d type=0x%x indices=%p program=%u vao=%p fbo=%p",
              (unsigned long long)drawCall,
              (unsigned)mode,
              (int)count,
              (unsigned)type,
              indices,
              activeProgramName,
              ctx ? MGL_STATE(ctx)->vao : NULL,
              ctx ? MGL_STATE(ctx)->framebuffer : NULL);
    }

    if (count <= 0) {
        if (traceDraw) {
            MGLTraceNSLog(@"MGL TRACE drawElements.skip.invalidCount call=%llu count=%d",
                  (unsigned long long)drawCall,
                  (int)count);
        }
        return;
    }

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:count
                                        indexType:type
                                          indices:indices
                                       baseVertex:0
                                    instanceCount:1
                                     baseInstance:0
                                            label:"drawElements"]) {
        return;
    }

    if ([self processGLStateLocked: true] == false) {
        s_drawElementsProcessStateFailCount++;
        MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
        if (traceDraw || s_drawElementsProcessStateFailCount <= 16 || (s_drawElementsProcessStateFailCount % 500) == 0) {
            MGLTraceNSLog(@"MGL TRACE drawElements.skip.processGLState call=%llu failCount=%llu",
                  (unsigned long long)drawCall,
                  (unsigned long long)s_drawElementsProcessStateFailCount);
        }
        return;
    }
    Program *activeProgram = ctx ? mglResolveProgramFromState(ctx) : NULL;
    drawVertexProgram = ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : NULL;
    drawFragmentProgram = ctx ? mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER) : NULL;
    activeProgramName = ctx ? mglCurrentRenderProgramKey(ctx) : 0u;
    drawProgram = activeProgram ? activeProgram : (drawFragmentProgram ? drawFragmentProgram : drawVertexProgram);
    BOOL traceLogDraw = mglProgramNeedsTraceLog(drawProgram);
    if (traceLogDraw && ctx) {
        VertexArray *drawVAO = mglRendererGetValidatedVAO(ctx, "drawElements.trace");
        mglTraceLog("DRAW_ELEMENTS_BEGIN call=%llu program=%u mode=0x%x count=%d type=0x%x indices=%p fbo=%u vao=%p enabled=0x%x viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) drawBuf=0x%x readBuf=0x%x colorMask=%d%d%d%d depth(test=%d write=%d func=0x%x) blend=%d cull=%d cullFace=0x%x frontFace=0x%x",
                    (unsigned long long)drawCall,
                    (unsigned)activeProgramName,
                    (unsigned)mode,
                    (int)count,
                    (unsigned)type,
                    indices,
                    MGL_STATE(ctx)->framebuffer ? (unsigned)MGL_STATE(ctx)->framebuffer->name : 0u,
                    drawVAO,
                    drawVAO ? (unsigned)drawVAO->enabled_attribs : 0u,
                    (int)MGL_STATE(ctx)->viewport[0],
                    (int)MGL_STATE(ctx)->viewport[1],
                    (int)MGL_STATE(ctx)->viewport[2],
                    (int)MGL_STATE(ctx)->viewport[3],
                    MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
                    (int)MGL_STATE(ctx)->var.scissor_box[0],
                    (int)MGL_STATE(ctx)->var.scissor_box[1],
                    (int)MGL_STATE(ctx)->var.scissor_box[2],
                    (int)MGL_STATE(ctx)->var.scissor_box[3],
                    (unsigned)MGL_STATE(ctx)->draw_buffer,
                    (unsigned)MGL_STATE(ctx)->read_buffer,
                    MGL_STATE(ctx)->var.color_writemask[0][0] ? 1 : 0,
                    MGL_STATE(ctx)->var.color_writemask[0][1] ? 1 : 0,
                    MGL_STATE(ctx)->var.color_writemask[0][2] ? 1 : 0,
                    MGL_STATE(ctx)->var.color_writemask[0][3] ? 1 : 0,
                    MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
                    MGL_STATE(ctx)->var.depth_writemask ? 1 : 0,
                    (unsigned)MGL_STATE(ctx)->var.depth_func,
                    MGL_STATE(ctx)->caps.blend ? 1 : 0,
                    MGL_STATE(ctx)->caps.cull_face ? 1 : 0,
                    (unsigned)MGL_STATE(ctx)->var.cull_face_mode,
                    (unsigned)MGL_STATE(ctx)->var.front_face);
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=empty_rasterization",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName);
        }
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=front_and_back_culled",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName);
        }
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    if (ctx && activeProgramName != 0u) {
        VertexArray *validVAO = mglRendererGetValidatedVAO(ctx, "drawElements.enabledMask");
        GLuint enabledAttribMask = validVAO ? validVAO->enabled_attribs : 0u;
        drawProgramUsesCloudFaces =
            mglProgramHasResourceNamed(drawVertexProgram, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "CloudFaces") ||
            mglProgramHasResourceNamed(drawVertexProgram, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "CloudFaces") ||
            mglProgramHasResourceNamed(drawFragmentProgram, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "CloudFaces") ||
            mglProgramHasResourceNamed(drawFragmentProgram, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "CloudFaces");

        mglObserveProgramDrawForFocus(activeProgramName, count, enabledAttribMask);

        // SPIR-V image dimension enum values: Cube is 3. Keep this literal here to avoid
        // depending on which SPIR-V enum header variant is pulled through spirv_cross_c.h.
        if (mglProgramHasImageDim(drawProgram, 3u)) {
            mglFocusLoadingProgram(activeProgramName, "cube-sampled-image", drawCall);
        }
    }

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=unsupported_mode mode=0x%x",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)mode);
        }
        return;
    }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) {
        NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=unsupported_index_type type=0x%x",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)type);
        }
        return;
    }

    Buffer *gl_element_buffer = getElementBuffer(ctx);
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: drawElements call=%llu missing element buffer mode=0x%x count=%d type=0x%x",
              (unsigned long long)drawCall,
              (unsigned)mode,
              (int)count,
              (unsigned)type);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=missing_element_buffer",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName);
        }
        return;
    }

    if ([self processBuffer: gl_element_buffer] == false)
    {
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=process_element_buffer_failed ebo=%u",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)gl_element_buffer->name);
        }
        return;
    }

    if (!gl_element_buffer->data.mtl_data) {
        NSLog(@"MGL WARNING: drawElements call=%llu element buffer %u has no Metal backing",
              (unsigned long long)drawCall, gl_element_buffer->name);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=element_no_mtl ebo=%u",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)gl_element_buffer->name);
        }
        return;
    }

    id <MTLBuffer>indexBuffer = (__bridge id<MTLBuffer>)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: drawElements call=%llu element buffer bridge failed for gl=%u",
              (unsigned long long)drawCall, gl_element_buffer->name);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=element_bridge_nil ebo=%u",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)gl_element_buffer->name);
        }
        return;
    }

    NSUInteger indexStride = mglGLIndexElementSize(type);
    if (indexStride == 0u) {
        NSLog(@"MGL WARNING: drawElements call=%llu unsupported GL index type=0x%x",
              (unsigned long long)drawCall,
              (unsigned)type);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=unsupported_gl_index_type type=0x%x",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)type);
        }
        return;
    }
    NSUInteger indexOffset = (NSUInteger)(uintptr_t)indices;
    if ((indexOffset % indexStride) != 0u) {
        NSLog(@"MGL DRAW_ELEMENTS BLOCK: call=%llu unaligned indices offset=%lu stride=%lu mode=0x%x count=%d type=0x%x ebo=%u len=%lu program=%u",
              (unsigned long long)drawCall,
              (unsigned long)indexOffset,
              (unsigned long)indexStride,
              (unsigned)mode,
              (int)count,
              (unsigned)type,
              gl_element_buffer->name,
              (unsigned long)indexBuffer.length,
              (unsigned)activeProgramName);
        MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=unaligned_index_offset ebo=%u offset=%lu stride=%lu",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)gl_element_buffer->name,
                        (unsigned long)indexOffset,
                        (unsigned long)indexStride);
        }
        return;
    }
    NSUInteger indexBytesNeeded = 0u;
    if ((NSUInteger)count > (NSUInteger)(NSUIntegerMax / indexStride)) {
        NSLog(@"MGL ERROR: drawElements call=%llu overflow computing index bytes count=%d stride=%lu",
              (unsigned long long)drawCall,
              (int)count,
              (unsigned long)indexStride);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=index_byte_overflow count=%d stride=%lu",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (int)count,
                        (unsigned long)indexStride);
        }
        return;
    }
    indexBytesNeeded = (NSUInteger)count * indexStride;
    if (indexOffset > indexBuffer.length || (indexBuffer.length - indexOffset) < indexBytesNeeded) {
        NSLog(@"MGL ERROR: drawElements call=%llu index range OOB gl=%u offset=%lu needed=%lu len=%lu type=0x%x count=%d",
              (unsigned long long)drawCall,
              gl_element_buffer->name,
              (unsigned long)indexOffset,
              (unsigned long)indexBytesNeeded,
              (unsigned long)indexBuffer.length,
              (unsigned)type,
              (int)count);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=index_oob ebo=%u offset=%lu needed=%lu len=%lu",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName,
                        (unsigned)gl_element_buffer->name,
                        (unsigned long)indexOffset,
                        (unsigned long)indexBytesNeeded,
                        (unsigned long)indexBuffer.length);
        }
        return;
    }

    const uint8_t *indexBytesForValidation = NULL;
    if (gl_element_buffer->data.buffer_data &&
        ((uintptr_t)gl_element_buffer->data.buffer_data >= 0x1000ull)) {
        indexBytesForValidation = (const uint8_t *)gl_element_buffer->data.buffer_data;
    } else if (indexBuffer.contents) {
        indexBytesForValidation = (const uint8_t *)indexBuffer.contents;
    }

    uint32_t minIndexForDraw = 0u;
    uint32_t maxIndexForDraw = 0u;
    bool haveIndexRange = false;
    uint32_t restartIndexForDraw = 0u;
    bool primitiveRestartForDraw = mglPrimitiveRestartIndexForType(ctx, type, &restartIndexForDraw);
    /* The scan only feeds the VBO-range guard and draw tracing — skip it
     * entirely when neither consumer is active. */
    bool needIndexRange = traceLogDraw || mglVboRangeValidationEnabled();
    if (indexBytesForValidation && needIndexRange) {
        /* Cache the O(count) index-range scan on the element buffer keyed by
         * last_write_src_hash (+ offset/count/type/restart).  Unchanged EBOs
         * skip the scan on every draw.  Eligible only when the bytes come from
         * CPU-side buffer_data (GPU-owned buffers and persistent maps may be
         * modified without a GL call, so their hash would not update). */
        bool scanCacheEligible =
            (indexBytesForValidation == (const uint8_t *)gl_element_buffer->data.buffer_data) &&
            (gl_element_buffer->storage_flags & GL_MAP_PERSISTENT_BIT) == 0u;
        if (scanCacheEligible &&
            gl_element_buffer->scan_cache_valid != 0u &&
            gl_element_buffer->scan_cache_src_hash == gl_element_buffer->last_write_src_hash &&
            gl_element_buffer->scan_cache_offset == (uint64_t)indexOffset &&
            gl_element_buffer->scan_cache_count == (uint32_t)count &&
            gl_element_buffer->scan_cache_type == type &&
            gl_element_buffer->scan_cache_restart_index == restartIndexForDraw &&
            (gl_element_buffer->scan_cache_restart_enabled != 0u) == primitiveRestartForDraw) {
            minIndexForDraw = gl_element_buffer->scan_cache_min_index;
            maxIndexForDraw = gl_element_buffer->scan_cache_max_index;
            haveIndexRange = true;
        } else {
            haveIndexRange = mglScanIndexRangeIgnoringRestart(indexBytesForValidation + indexOffset,
                                                              type,
                                                              count,
                                                              primitiveRestartForDraw,
                                                              restartIndexForDraw,
                                                              &minIndexForDraw,
                                                              &maxIndexForDraw);
            if (haveIndexRange && scanCacheEligible) {
                gl_element_buffer->scan_cache_min_index = minIndexForDraw;
                gl_element_buffer->scan_cache_max_index = maxIndexForDraw;
                gl_element_buffer->scan_cache_src_hash = gl_element_buffer->last_write_src_hash;
                gl_element_buffer->scan_cache_offset = (uint64_t)indexOffset;
                gl_element_buffer->scan_cache_count = (uint32_t)count;
                gl_element_buffer->scan_cache_type = type;
                gl_element_buffer->scan_cache_restart_index = restartIndexForDraw;
                gl_element_buffer->scan_cache_restart_enabled = primitiveRestartForDraw ? 1u : 0u;
                gl_element_buffer->scan_cache_valid = 1u;
            }
        }
    }
    if (traceLogDraw) {
        mglTraceLog("DRAW_ELEMENTS_INDEX_RANGE call=%llu program=%u ebo=%u offset=%lu stride=%lu needed=%lu len=%lu haveRange=%d range=[%u,%u] restart=%d",
                    (unsigned long long)drawCall,
                    (unsigned)activeProgramName,
                    (unsigned)gl_element_buffer->name,
                    (unsigned long)indexOffset,
                    (unsigned long)indexStride,
                    (unsigned long)indexBytesNeeded,
                    (unsigned long)indexBuffer.length,
                    haveIndexRange ? 1 : 0,
                    (unsigned)minIndexForDraw,
                    (unsigned)maxIndexForDraw,
                    primitiveRestartForDraw ? 1 : 0);
    }

    if (mglVboRangeValidationEnabled() && haveIndexRange && ctx) {
        if (![self validateDrawElementsVboRange:drawCall
                               activeProgramName:activeProgramName
                                        minIndex:minIndexForDraw
                                        maxIndex:maxIndexForDraw]) {
            return;
        }
    }

    [self inspectDrawElementsTrace:drawCall
                   activeProgramName:activeProgramName
               drawProgramUsesCloudFaces:drawProgramUsesCloudFaces
                         drawProgram:drawProgram
                  drawVertexProgram:drawVertexProgram
                drawFragmentProgram:drawFragmentProgram
                   glElementBuffer:gl_element_buffer
                        indexBuffer:indexBuffer
                        indexOffset:indexOffset
                        indexStride:indexStride
                    indexBytesNeeded:indexBytesNeeded
                     haveIndexRange:haveIndexRange
                      minIndexForDraw:minIndexForDraw
                      maxIndexForDraw:maxIndexForDraw
                             mode:mode
                           count:count
                           type:type
                         traceDraw:traceDraw
                      traceLogDraw:traceLogDraw];

    if (![self encodeDrawElementsPrimitive:drawCall
                          activeProgramName:activeProgramName
                          glElementBuffer:gl_element_buffer
                               indexBuffer:indexBuffer
                                     mode:mode
                            primitiveType:primitiveType
                                    type:type
                               indexType:indexType
                             indexOffset:indexOffset
                                   count:count
                   indexBytesForValidation:indexBytesForValidation
                           traceLogDraw:traceLogDraw
                       polygonModePoint:polygonModePoint
                      emulateTriangleFan:emulateTriangleFan
                         emulateLineLoop:emulateLineLoop
                           emulateQuads:emulateQuads]) {
        return;
    }


    if (traceLogDraw) {
        mglTraceLog("DRAW_ELEMENTS_SUBMIT call=%llu program=%u mode=0x%x count=%d type=0x%x ebo=%u offset=%lu haveRange=%d range=[%u,%u] encoder=%p pipeline=%p",
                    (unsigned long long)drawCall,
                    (unsigned)activeProgramName,
                    (unsigned)mode,
                    (int)count,
                    (unsigned)type,
                    (unsigned)gl_element_buffer->name,
                    (unsigned long)indexOffset,
                    haveIndexRange ? 1 : 0,
                    (unsigned)minIndexForDraw,
                    (unsigned)maxIndexForDraw,
                    _renderPassManager.state->currentRenderEncoder,
                    _pipelineCache.state->pipelineState);
    }

    MGL_FRAME_STORE(g_mglLastDrawElementsCall, drawCall);
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
    mglLogDrawWithoutSwapWatchdog("elements",
                                  drawCall,
                                  ctx,
                                  _renderPassManager.state->currentCommandBuffer,
                                  _renderPassManager.state->currentRenderEncoder,
                                  _renderPassManager.state->renderPassDescriptor);

    double drawElapsedMs = (mglNowSeconds() - drawStartSeconds) * 1000.0;
    if (traceDraw || drawElapsedMs >= 16.0) {
        MGLTraceNSLog(@"MGL TRACE drawElements.end call=%llu elapsed=%.3fms indexBuffer=%u len=%lu encoder=%p",
              (unsigned long long)drawCall,
              drawElapsedMs,
              gl_element_buffer->name,
              (unsigned long)indexBuffer.length,
              _renderPassManager.state->currentRenderEncoder);
    }
}

- (BOOL)validateDrawElementsVboRange:(uint64_t)drawCall
                      activeProgramName:(GLuint)activeProgramName
                               minIndex:(uint32_t)minIndexForDraw
                               maxIndex:(uint32_t)maxIndexForDraw
{
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    if (vao) {
        GLuint maxAttribs = MAX_ATTRIBS;

        for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
            if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
                continue;
            }

            MGLResolvedVertexAttribBinding resolved = {0};
            if (!mglRendererResolveVertexAttribBinding(ctx,
                                                       vao,
                                                       attrib,
                                                       "drawElements.vboRange",
                                                       &resolved)) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u invalid buffer",
                      (unsigned long long)drawCall,
                      (unsigned)attrib);
                return NO;
            }
            const VertexAttrib *a = resolved.attrib;
            Buffer *vbo = resolved.buffer;

            if (!mglRendererBufferHasDrawableContents(vbo)) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u never written",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name);
                return NO;
            }

            if (resolved.binding_offset < 0 || resolved.relativeoffset < 0) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u negative attrib offset bindingOffset=%lld relativeOffset=%lld",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (long long)resolved.binding_offset,
                      (long long)resolved.relativeoffset);
                return NO;
            }

            size_t compSize = mglVertexAttribComponentSize(a->type);
            size_t compCount = (size_t)a->size;
            if (compSize == 0u || compCount == 0u) {
                continue;
            }

            if (compCount > (SIZE_MAX / compSize)) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u component span overflow type=0x%x size=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned)a->type,
                      (unsigned)a->size);
                return NO;
            }

            uint64_t elemBytes = (uint64_t)compSize * (uint64_t)compCount;
            if (elemBytes == 0u) {
                continue;
            }

            uint64_t stride = (resolved.stride > 0u) ? (uint64_t)resolved.stride : elemBytes;
            uint64_t bindingOffset = (uint64_t)resolved.binding_offset;
            uint64_t attrRelativeOffset = (uint64_t)(uintptr_t)resolved.relativeoffset;
            if (bindingOffset > UINT64_MAX - attrRelativeOffset) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u offset overflow bindingOffset=%llu relativeOffset=%llu",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned long long)bindingOffset,
                      (unsigned long long)attrRelativeOffset);
                return NO;
            }
            uint64_t relOffset = bindingOffset + attrRelativeOffset;
            if (stride == 0u) {
                continue;
            }

            uint32_t attribMinIndex = (resolved.divisor != 0u) ? 0u : minIndexForDraw;
            uint32_t attribMaxIndex = (resolved.divisor != 0u) ? 0u : maxIndexForDraw;

            if (relOffset > UINT64_MAX - elemBytes) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u overflow computing vertex range bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned long long)bindingOffset,
                      (unsigned long long)relOffset,
                      (unsigned long long)elemBytes,
                      (unsigned)resolved.divisor);
                return NO;
            }

            if ((uint64_t)attribMaxIndex > (UINT64_MAX - relOffset - elemBytes) / stride) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u overflow computing vertex range maxIndex=%u stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)attribMaxIndex,
                      (unsigned long long)stride,
                      (unsigned long long)bindingOffset,
                      (unsigned long long)relOffset,
                      (unsigned long long)elemBytes,
                      (unsigned)resolved.divisor);
                return NO;
            }

            if ((uint64_t)attribMinIndex > (UINT64_MAX - relOffset) / stride) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u overflow computing min range minIndex=%u stride=%llu bindingOffset=%llu relOffset=%llu divisor=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)attribMinIndex,
                      (unsigned long long)stride,
                      (unsigned long long)bindingOffset,
                      (unsigned long long)relOffset,
                      (unsigned)resolved.divisor);
                return NO;
            }

            uint64_t minStart = relOffset + ((uint64_t)attribMinIndex * stride);
            uint64_t maxEnd = relOffset + ((uint64_t)attribMaxIndex * stride) + elemBytes;
            uint64_t vboSize = (vbo->size > 0) ? (uint64_t)vbo->size : 0u;

            if (maxEnd > vboSize) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u indexRange=[%u,%u] byteRange=[%llu,%llu) exceeds vboSize=%llu (stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u)",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned)attribMinIndex,
                      (unsigned)attribMaxIndex,
                      (unsigned long long)minStart,
                      (unsigned long long)maxEnd,
                      (unsigned long long)vboSize,
                      (unsigned long long)stride,
                      (unsigned long long)bindingOffset,
                      (unsigned long long)relOffset,
                      (unsigned long long)elemBytes,
                      (unsigned)resolved.divisor);
                return NO;
            }

            if (!vbo->data.mtl_data) {
                [self bindMTLBufferLocked:vbo];
            }
            if (!vbo->data.mtl_data) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u has no Metal backing byteRange=[%llu,%llu)",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned long long)minStart,
                      (unsigned long long)maxEnd);
                return NO;
            }
            id<MTLBuffer> attribMetalBuffer = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
            if (!attribMetalBuffer) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u Metal bridge failed",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name);
                return NO;
            }

            uint64_t metalLen = (uint64_t)attribMetalBuffer.length;
            if (maxEnd > metalLen) {
                NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u indexRange=[%u,%u] byteRange=[%llu,%llu) exceeds metalLen=%llu vboSize=%llu stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned)attribMinIndex,
                      (unsigned)attribMaxIndex,
                      (unsigned long long)minStart,
                      (unsigned long long)maxEnd,
                      (unsigned long long)metalLen,
                      (unsigned long long)vboSize,
                      (unsigned long long)stride,
                      (unsigned long long)bindingOffset,
                      (unsigned long long)relOffset,
                      (unsigned long long)elemBytes,
                      (unsigned)resolved.divisor);
                return NO;
            }

            if (vbo->written_min >= 0 && vbo->written_max >= 0) {
                uint64_t writtenMin = (uint64_t)vbo->written_min;
                uint64_t writtenMax = (uint64_t)vbo->written_max;
                if (minStart < writtenMin || maxEnd > writtenMax) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u indexRange=[%u,%u] byteRange=[%llu,%llu) outside written=[%llu,%llu) (source=%u divisor=%u)",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name,
                          (unsigned)attribMinIndex,
                          (unsigned)attribMaxIndex,
                          (unsigned long long)minStart,
                          (unsigned long long)maxEnd,
                          (unsigned long long)writtenMin,
                          (unsigned long long)writtenMax,
                          (unsigned)vbo->last_init_source,
                          (unsigned)resolved.divisor);
                    return NO;
                }
            }
        }
    }
    return YES;
}

- (void)inspectDrawElementsTrace:(uint64_t)drawCall
                   activeProgramName:(GLuint)activeProgramName
               drawProgramUsesCloudFaces:(BOOL)drawProgramUsesCloudFaces
                         drawProgram:(Program *)drawProgram
                  drawVertexProgram:(Program *)drawVertexProgram
                drawFragmentProgram:(Program *)drawFragmentProgram
                   glElementBuffer:(Buffer *)gl_element_buffer
                        indexBuffer:(id<MTLBuffer>)indexBuffer
                        indexOffset:(NSUInteger)indexOffset
                        indexStride:(NSUInteger)indexStride
                    indexBytesNeeded:(NSUInteger)indexBytesNeeded
                     haveIndexRange:(bool)haveIndexRange
                      minIndexForDraw:(uint32_t)minIndexForDraw
                      maxIndexForDraw:(uint32_t)maxIndexForDraw
                             mode:(GLenum)mode
                           count:(GLsizei)count
                           type:(GLenum)type
                         traceDraw:(bool)traceDraw
                      traceLogDraw:(BOOL)traceLogDraw
{
    if (traceDraw || indexOffset != 0u) {
        MGLTraceNSLog(@"MGL TRACE drawElements.indices call=%llu gl=%u offset=%lu stride=%lu needed=%lu len=%lu",
              (unsigned long long)drawCall,
              gl_element_buffer->name,
              (unsigned long)indexOffset,
              (unsigned long)indexStride,
              (unsigned long)indexBytesNeeded,
              (unsigned long)indexBuffer.length);
    }

    if (mglShouldInspectDrawCall(drawCall, activeProgramName) || drawProgramUsesCloudFaces) {
        if (ctx && mglIsFocusedLoadingProgram(activeProgramName)) {
            if (drawVertexProgram) {
                mglWriteProgramMSLDump(drawVertexProgram,
                                       [NSString stringWithFormat:@"drawElements hot program %u call %llu",
                                                                  (unsigned)activeProgramName,
                                                                  (unsigned long long)drawCall]);
            }
            if (drawFragmentProgram && drawFragmentProgram != drawVertexProgram) {
                mglWriteProgramMSLDump(drawFragmentProgram,
                                       [NSString stringWithFormat:@"drawElements hot program %u call %llu",
                                                                  (unsigned)activeProgramName,
                                                                  (unsigned long long)drawCall]);
            }
        }
        if (drawProgramUsesCloudFaces && drawProgram) {
            mglWriteProgramMSLDump(drawProgram,
                                   [NSString stringWithFormat:@"CloudFaces texel buffer drawElements call %llu",
                                                              (unsigned long long)drawCall]);
        }

        if (ctx) {
            MTLTriangleFillMode loggedTriangleFillMode =
                (mglDrawModeProducesPolygons(mode) && MGL_STATE(ctx)->var.polygon_mode == GL_LINE)
                    ? MTLTriangleFillModeLines
                    : MTLTriangleFillModeFill;
            MGLTraceNSLog(@"MGL TRACE drawElements.state call=%llu program=%u mode=0x%x polygonMode=0x%x triFill=%lu colorMask(use=%d rgba=%d%d%d%d) depth(write=%d test=%d) blend=%d cull=%d viewport=%d,%d,%d,%d",
                  (unsigned long long)drawCall,
                  (unsigned)activeProgramName,
                  (unsigned)mode,
                  (unsigned)MGL_STATE(ctx)->var.polygon_mode,
                  (unsigned long)loggedTriangleFillMode,
                  MGL_STATE(ctx)->caps.use_color_mask[0] ? 1 : 0,
                  MGL_STATE(ctx)->var.color_writemask[0][0] ? 1 : 0,
                  MGL_STATE(ctx)->var.color_writemask[0][1] ? 1 : 0,
                  MGL_STATE(ctx)->var.color_writemask[0][2] ? 1 : 0,
                  MGL_STATE(ctx)->var.color_writemask[0][3] ? 1 : 0,
                  MGL_STATE(ctx)->var.depth_writemask ? 1 : 0,
                  MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
                  MGL_STATE(ctx)->caps.blend ? 1 : 0,
                  MGL_STATE(ctx)->caps.cull_face ? 1 : 0,
                  (int)MGL_STATE(ctx)->viewport[0],
                  (int)MGL_STATE(ctx)->viewport[1],
                  (int)MGL_STATE(ctx)->viewport[2],
                  (int)MGL_STATE(ctx)->viewport[3]);
        }

        const uint8_t *indexBytes = NULL;
        if (gl_element_buffer->data.buffer_data &&
            ((uintptr_t)gl_element_buffer->data.buffer_data >= 0x1000ull)) {
            indexBytes = (const uint8_t *)gl_element_buffer->data.buffer_data;
        } else if (indexBuffer.contents) {
            indexBytes = (const uint8_t *)indexBuffer.contents;
        }

        if (indexBytes) {
            const uint8_t *start = indexBytes + indexOffset;
            NSUInteger previewCount = MIN((NSUInteger)count, (NSUInteger)12);
            char preview[256];
            preview[0] = '\0';
            uint32_t minIndex = UINT32_MAX;
            uint32_t maxIndex = 0u;

            for (NSUInteger ii = 0; ii < previewCount; ii++) {
                uint32_t idxValue = mglReadGLIndexValue(start, type, ii);
                if (idxValue < minIndex) {
                    minIndex = idxValue;
                }
                if (idxValue > maxIndex) {
                    maxIndex = idxValue;
                }

                size_t used = strlen(preview);
                if (used < sizeof(preview) - 1u) {
                    snprintf(preview + used,
                             sizeof(preview) - used,
                             "%s%u",
                             (ii == 0u ? "" : ","),
                             idxValue);
                }
            }

            MGLTraceNSLog(@"MGL TRACE drawElements.preview call=%llu program=%u ebo=%u count=%d type=0x%x offset=%lu first[%lu]={%s} min=%u max=%u",
                  (unsigned long long)drawCall,
                  (unsigned)activeProgramName,
                  (unsigned)gl_element_buffer->name,
                  (int)count,
                  (unsigned)type,
                  (unsigned long)indexOffset,
                  (unsigned long)previewCount,
                  preview,
                  minIndex == UINT32_MAX ? 0u : minIndex,
                  maxIndex);

            VertexArray *vao = ctx ? MGL_STATE(ctx)->vao : NULL;
            if (vao) {
                GLuint traceAttribLimit = MIN((GLuint)4u, ctx ? MGL_STATE(ctx)->max_vertex_attribs : (GLuint)4u);
                for (GLuint attrib = 0; attrib < traceAttribLimit; attrib++) {
	                    mglTraceDrawElementsAttrib(ctx,
	                                               vao,
	                                               drawCall,
		                                               activeProgramName,
		                                               start,
		                                               type,
		                                               0,
		                                               0,
		                                               attrib,
		                                               traceLogDraw);
                }
            }
            if (vao && (vao->enabled_attribs & 0x1u)) {
                MGLResolvedVertexAttribBinding resolved0 = {0};
                if (mglRendererResolveVertexAttribBinding(ctx,
                                                           vao,
                                                           0u,
                                                           "drawElements.attrib0",
                                                           &resolved0)) {
                    const VertexAttrib *a0 = resolved0.attrib;
                    Buffer *vbo = resolved0.buffer;
                    const uint8_t *vboBytes = NULL;
                    if (vbo->data.buffer_data && ((uintptr_t)vbo->data.buffer_data >= 0x1000ull)) {
                        vboBytes = (const uint8_t *)vbo->data.buffer_data;
                    } else if (vbo->data.mtl_data) {
                        id<MTLBuffer> vb = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
                        vboBytes = (const uint8_t *)vb.contents;
                    }

                    if (vboBytes &&
                        a0->type == GL_FLOAT &&
                        (a0->size >= 2u && a0->size <= 4u) &&
                        resolved0.stride >= (sizeof(float) * a0->size)) {
                        uint32_t firstIndex = mglReadGLIndexValue(start, type, 0u);
                        NSUInteger bindingOffset = (resolved0.binding_offset > 0) ? (NSUInteger)resolved0.binding_offset : 0u;
                        NSUInteger relativeOffset = (resolved0.relativeoffset > 0) ? (NSUInteger)resolved0.relativeoffset : 0u;
                        NSUInteger stride = (resolved0.stride > 0u) ? (NSUInteger)resolved0.stride : 0u;
                        NSUInteger vertexOffset = bindingOffset +
                                                  relativeOffset +
                                                  ((NSUInteger)firstIndex * stride);
                        NSUInteger needed = sizeof(float) * a0->size;
                        if (vertexOffset <= (NSUInteger)vbo->size &&
                            ((NSUInteger)vbo->size - vertexOffset) >= needed) {
                            float comps[4] = {0.f, 0.f, 0.f, 0.f};
                            memcpy(comps, vboBytes + vertexOffset, needed);
                            MGLTraceNSLog(@"MGL TRACE drawElements.attrib0 call=%llu program=%u vbo=%u firstIndex=%u bindingOffset=%lu relOffset=%u stride=%u size=%u vec=(%.4f,%.4f,%.4f,%.4f) vboSize=%lld init(ever=%u full=%u source=%u off=%lld size=%lld src=%p hash=0x%016llx)",
                                  (unsigned long long)drawCall,
                                  (unsigned)activeProgramName,
                                  (unsigned)vbo->name,
                                  (unsigned)firstIndex,
                                  (unsigned long)bindingOffset,
                                  (unsigned)relativeOffset,
                                  (unsigned)stride,
                                  (unsigned)a0->size,
                                  comps[0], comps[1], comps[2], comps[3],
                                  (long long)vbo->size,
                                  (unsigned)vbo->ever_written,
                                  (unsigned)vbo->has_initialized_data,
                                  (unsigned)vbo->last_init_source,
                                  (long long)vbo->last_write_offset,
                                  (long long)vbo->last_write_size,
                                  vbo->last_write_src_ptr,
                                  (unsigned long long)vbo->last_write_src_hash);

                            typedef struct MGLAttrib0DumpKey {
                                GLuint program;
                                GLuint vbo;
                            } MGLAttrib0DumpKey;
                            static MGLAttrib0DumpKey s_dumpedAttrib0RawBuffers[24] = {{0, 0}};
                            static uint32_t s_dumpedAttrib0RawBufferCount = 0;
                            BOOL alreadyDumpedAttrib0 = NO;
                            for (uint32_t dumpIndex = 0; dumpIndex < s_dumpedAttrib0RawBufferCount; dumpIndex++) {
                                if (s_dumpedAttrib0RawBuffers[dumpIndex].program == activeProgramName &&
                                    s_dumpedAttrib0RawBuffers[dumpIndex].vbo == vbo->name) {
                                    alreadyDumpedAttrib0 = YES;
                                    break;
                                }
                            }

                            if (!alreadyDumpedAttrib0 &&
                                s_dumpedAttrib0RawBufferCount < (uint32_t)(sizeof(s_dumpedAttrib0RawBuffers) / sizeof(s_dumpedAttrib0RawBuffers[0])) &&
                                vbo->size > 0) {
                                size_t totalSize = (size_t)vbo->size;
                                size_t headLen = MIN((size_t)256, totalSize);
                                size_t windowOffset = (size_t)vertexOffset;
                                if (windowOffset > totalSize) {
                                    windowOffset = totalSize;
                                }
                                size_t windowLen = 0;
                                if (windowOffset < totalSize) {
                                    windowLen = MIN((size_t)128, totalSize - windowOffset);
                                }

                                MGLTraceNSLog(@"MGL DUMP attrib0.raw.begin call=%llu program=%u vbo=%u size=%zu firstIndex=%u vertexOffset=%zu stride=%u bindingOffset=%lu relOffset=%u",
                                              (unsigned long long)drawCall,
                                              (unsigned)activeProgramName,
                                              (unsigned)vbo->name,
                                              totalSize,
                                              (unsigned)firstIndex,
                                              (size_t)vertexOffset,
                                              (unsigned)stride,
                                              (unsigned long)bindingOffset,
                                              (unsigned)relativeOffset);
                                mglDumpBytesToLog(@"attrib0.vbo.head", vboBytes, headLen, 0u);
                                if (windowLen > 0) {
                                    mglDumpBytesToLog(@"attrib0.vbo.vertexWindow",
                                                      vboBytes + windowOffset,
                                                      windowLen,
                                                      windowOffset);
                                }
                                MGLTraceNSLog(@"MGL DUMP attrib0.raw.end vbo=%u", (unsigned)vbo->name);
                                s_dumpedAttrib0RawBuffers[s_dumpedAttrib0RawBufferCount].program = activeProgramName;
                                s_dumpedAttrib0RawBuffers[s_dumpedAttrib0RawBufferCount].vbo = vbo->name;
                                s_dumpedAttrib0RawBufferCount++;
                            }
                        } else {
                            NSLog(@"MGL WARNING: drawElements.attrib0 call=%llu OOB firstIndex=%u bindingOffset=%lu relOffset=%u stride=%u size=%u vboSize=%lld",
                                  (unsigned long long)drawCall,
                                  (unsigned)firstIndex,
                                  (unsigned long)bindingOffset,
                                  (unsigned)relativeOffset,
                                  (unsigned)stride,
                                  (unsigned)a0->size,
                                  (long long)vbo->size);
                        }
                    } else {
                        MGLTraceNSLog(@"MGL TRACE drawElements.attrib0 call=%llu skipped(vboBytes=%p type=0x%x size=%u stride=%u)",
                              (unsigned long long)drawCall,
                              vboBytes,
                              (unsigned)a0->type,
                              (unsigned)a0->size,
                              (unsigned)resolved0.stride);
                    }
                }
            }
        } else {
            NSLog(@"MGL WARNING: drawElements.preview call=%llu unavailable(index bytes nil) ebo=%u",
                  (unsigned long long)drawCall,
                  (unsigned)gl_element_buffer->name);
        }
    }

    if (mglShouldInspectDrawCall(drawCall, activeProgramName) || drawProgramUsesCloudFaces) {
        VertexArray *submitVAO = ctx ? MGL_STATE(ctx)->vao : NULL;
        MGLTraceNSLog(@"MGL TRACE drawElements.submit call=%llu program=%u mode=0x%x count=%d type=0x%x ebo=%u offset=%lu stride=%lu needed=%lu len=%lu haveRange=%d range=[%u,%u] vao=%p enabled=0x%x encoder=%p cloudFaces=%d",
              (unsigned long long)drawCall,
              (unsigned)activeProgramName,
              (unsigned)mode,
              (int)count,
              (unsigned)type,
              (unsigned)gl_element_buffer->name,
              (unsigned long)indexOffset,
              (unsigned long)indexStride,
              (unsigned long)indexBytesNeeded,
              (unsigned long)indexBuffer.length,
              haveIndexRange ? 1 : 0,
              (unsigned)minIndexForDraw,
              (unsigned)maxIndexForDraw,
              submitVAO,
              submitVAO ? (unsigned)submitVAO->enabled_attribs : 0u,
              _renderPassManager.state->currentRenderEncoder,
              drawProgramUsesCloudFaces ? 1 : 0);
    }
}

- (BOOL)encodeDrawElementsPrimitive:(uint64_t)drawCall
                          activeProgramName:(GLuint)activeProgramName
                          glElementBuffer:(Buffer *)gl_element_buffer
                               indexBuffer:(id<MTLBuffer>)indexBuffer
                                     mode:(GLenum)mode
                            primitiveType:(MTLPrimitiveType)primitiveType
                                    type:(GLenum)type
                               indexType:(MTLIndexType)indexType
                             indexOffset:(NSUInteger)indexOffset
                                   count:(GLsizei)count
                   indexBytesForValidation:(const uint8_t *)indexBytesForValidation
                           traceLogDraw:(BOOL)traceLogDraw
                       polygonModePoint:(BOOL)polygonModePoint
                      emulateTriangleFan:(BOOL)emulateTriangleFan
                         emulateLineLoop:(BOOL)emulateLineLoop
                           emulateQuads:(BOOL)emulateQuads
{
    @try {
        MGLPrimitiveRestartEncodeResult restartResult =
            mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                                   _device,
                                                   ctx,
                                                   gl_element_buffer,
                                                   indexBuffer,
                                                   mode,
                                                   primitiveType,
                                                   type,
                                                   indexType,
                                                   indexOffset,
                                                   count,
                                                   1u,
                                                   0,
                                                   0u,
                                                   "drawElements");
        if (restartResult == MGLPrimitiveRestartEncodeFailed) {
            MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
            if (traceLogDraw) {
                mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=primitive_restart_encode_failed",
                            (unsigned long long)drawCall,
                            (unsigned)activeProgramName);
            }
            return NO;
        }
        BOOL restartHandled = (restartResult == MGLPrimitiveRestartEncodeHandled);

        if (restartHandled) {
            // Already emitted as restart-separated Metal draws.
        } else if (polygonModePoint) {
            if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                              _device,
                                              gl_element_buffer,
                                              indexBuffer,
                                              mode,
                                              type,
                                              indexType,
                                              indexOffset,
                                              count,
                                              1u,
                                              0,
                                              0u,
                                              "drawElements")) {
                MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
                if (traceLogDraw) {
                    mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=polygon_point_encode",
                                (unsigned long long)drawCall,
                                (unsigned)activeProgramName);
                }
                return NO;
            }
        } else if (emulateTriangleFan) {
            if (count < 3) {
                if (traceLogDraw) {
                    mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=triangle_fan_too_small",
                                (unsigned long long)drawCall,
                                (unsigned)activeProgramName);
                }
                return NO;
            }

            const uint8_t *fanSource = indexBytesForValidation ? (indexBytesForValidation + indexOffset) : NULL;
            NSUInteger fanIndexCount = 0u;
            id<MTLBuffer> fanIndexBuffer = mglNewTriangleFanElementIndexBuffer(_device,
                                                                               fanSource,
                                                                               type,
                                                                               (NSUInteger)count,
                                                                               &fanIndexCount);
            if (!fanIndexBuffer || fanIndexCount == 0u) {
                NSLog(@"MGL WARNING: drawElements call=%llu triangle fan emulation failed ebo=%u count=%d offset=%lu source=%p",
                      (unsigned long long)drawCall,
                      (unsigned)gl_element_buffer->name,
                      (int)count,
                      (unsigned long)indexOffset,
                      fanSource);
                MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
                if (traceLogDraw) {
                    mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=triangle_fan_emulation_failed ebo=%u",
                                (unsigned long long)drawCall,
                                (unsigned)activeProgramName,
                                (unsigned)gl_element_buffer->name);
                }
                return NO;
            }

            [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                              indexCount:fanIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:fanIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1];
        } else if (emulateLineLoop) {
            if (count < 2) {
                if (traceLogDraw) {
                    mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=line_loop_too_small",
                                (unsigned long long)drawCall,
                                (unsigned)activeProgramName);
                }
                return NO;
            }

            const uint8_t *loopSource = indexBytesForValidation ? (indexBytesForValidation + indexOffset) : NULL;
            NSUInteger loopIndexCount = 0u;
            id<MTLBuffer> loopIndexBuffer = mglNewLineLoopElementIndexBuffer(_device,
                                                                             loopSource,
                                                                             type,
                                                                             (NSUInteger)count,
                                                                             &loopIndexCount);
            if (!loopIndexBuffer || loopIndexCount == 0u) {
                NSLog(@"MGL WARNING: drawElements call=%llu line loop emulation failed ebo=%u count=%d offset=%lu source=%p",
                      (unsigned long long)drawCall,
                      (unsigned)gl_element_buffer->name,
                      (int)count,
                      (unsigned long)indexOffset,
                      loopSource);
                MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
                if (traceLogDraw) {
                    mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=line_loop_emulation_failed ebo=%u",
                                (unsigned long long)drawCall,
                                (unsigned)activeProgramName,
                                (unsigned)gl_element_buffer->name);
                }
                return NO;
            }

            [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                                              indexCount:loopIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:loopIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1];
        } else if (emulateQuads) {
            if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                       _device,
                                       gl_element_buffer,
                                       indexBuffer,
                                       type,
                                       indexOffset,
                                       count,
                                       1u,
                                       0,
                                       0u,
                                       mglPolygonModeLineForDrawMode(ctx, mode),
                                       "drawElements")) {
                MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
                if (traceLogDraw) {
                    mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=quad_emulation_failed ebo=%u",
                                (unsigned long long)drawCall,
                                (unsigned)activeProgramName,
                                (unsigned)gl_element_buffer->name);
                }
                return NO;
            }
        } else {
            NSUInteger drawIndexOffset = indexOffset;
            MTLIndexType drawIndexType = indexType;
            id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                          gl_element_buffer,
                                                                          indexBuffer,
                                                                          type,
                                                                          &drawIndexOffset,
                                                                          &drawIndexType);
            if (!drawIndexBuffer) {
                MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
                if (traceLogDraw) {
                    mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=prepared_index_buffer_failed ebo=%u",
                                (unsigned long long)drawCall,
                                (unsigned)activeProgramName,
                                (unsigned)gl_element_buffer->name);
                }
                return NO;
            }
            [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType
                                             indexBuffer:drawIndexBuffer indexBufferOffset:drawIndexOffset instanceCount:1];
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: drawElements call=%llu drawIndexedPrimitives exception: %@",
              (unsigned long long)drawCall, exception);
        if (traceLogDraw) {
            mglTraceLog("DRAW_ELEMENTS_SKIP call=%llu program=%u reason=draw_exception",
                        (unsigned long long)drawCall,
                        (unsigned)activeProgramName);
        }
        return NO;
    }
    return YES;
}


-(void) mtlDrawRangeElements: (GLMContext) glm_ctx mode:(GLenum) mode start:(GLuint) start end:(GLuint) end count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;
    (void)start;
    (void)end;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:count
                                        indexType:type
                                          indices:indices
                                       baseVertex:0
                                    instanceCount:1
                                     baseInstance:0
                                            label:"drawRangeElements"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawRangeElements" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               1u,
                                               0,
                                               0u,
                                               "drawRangeElements");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        if (restartResult == MGLPrimitiveRestartEncodeHandled) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        }
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          1u,
                                          0,
                                          0u,
                                          "drawRangeElements")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         1u,
                                         0,
                                         0u,
                                         "drawRangeElements")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      1u,
                                      0,
                                      0u,
                                      "drawRangeElements")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }
    if (emulateQuads) {
        if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                   _device,
                                   gl_element_buffer,
                                   indexBuffer,
                                   type,
                                   offset,
                                   count,
                                   1u,
                                   0,
                                   0u,
                                   mglPolygonModeLineForDrawMode(ctx, mode),
                                   "drawRangeElements")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1];
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
}

-(void) mtlDrawArraysInstanced: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount
{
    MTLPrimitiveType primitiveType;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:first
                                            count:count
                                        indexType:0
                                          indices:NULL
                                       baseVertex:0
                                    instanceCount:instancecount
                                     baseInstance:0
                                            label:"drawArraysInstanced"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint) {
        if (mglEncodeArrayPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                       _device,
                                       mode,
                                       first,
                                       count,
                                       (NSUInteger)instancecount,
                                       0u,
                                       "drawArraysInstanced")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    if (mode == GL_TRIANGLE_FAN) {
        if (mglEncodeArrayTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      count,
                                      first,
                                      (NSUInteger)instancecount,
                                      0u,
                                      "drawArraysInstanced")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }
    if (mode == GL_LINE_LOOP) {
        if (mglEncodeArrayLineLoop(_renderPassManager.state->currentRenderEncoder,
                                   glm_ctx,
                                   _device,
                                   count,
                                   first,
                                   (NSUInteger)instancecount,
                                   0u,
                                   "drawArraysInstanced")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }
    if (mode == GL_QUADS) {
        if (mglEncodeArrayQuads(_renderPassManager.state->currentRenderEncoder,
                                _device,
                                count,
                                first,
                                (NSUInteger)instancecount,
                                0u,
                                mglPolygonModeLineForDrawMode(ctx, mode),
                                "drawArraysInstanced")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    [_renderPassManager.state->currentRenderEncoder drawPrimitives:primitiveType vertexStart:first vertexCount:count instanceCount:instancecount];
    [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
}

-(void) mtlDrawElementsInstanced: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:count
                                        indexType:type
                                          indices:indices
                                       baseVertex:0
                                    instanceCount:instancecount
                                     baseInstance:0
                                            label:"drawElementsInstanced"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstanced" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               (NSUInteger)instancecount,
                                               0,
                                               0u,
                                               "drawElementsInstanced");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        if (restartResult == MGLPrimitiveRestartEncodeHandled) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          (NSUInteger)instancecount,
                                          0,
                                          0u,
                                          "drawElementsInstanced")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         0,
                                         0u,
                                         "drawElementsInstanced")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      0,
                                      0u,
                                      "drawElementsInstanced")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateQuads) {
        if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                   _device,
                                   gl_element_buffer,
                                   indexBuffer,
                                   type,
                                   offset,
                                   count,
                                   (NSUInteger)instancecount,
                                   0,
                                   0u,
                                   mglPolygonModeLineForDrawMode(ctx, mode),
                                   "drawElementsInstanced")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    // for now lets just ignore the range data and use drawIndexedPrimitives
    //
    // in the future it would be an idea to use temp buffers for large buffers that would wire
    // to much memory down.. like a million point galaxy drawing
    //
    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount];
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
}

-(void) mtlDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:count
                                        indexType:type
                                          indices:indices
                                       baseVertex:basevertex
                                    instanceCount:1
                                     baseInstance:0
                                            label:"drawElementsBaseVertex"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               1u,
                                               basevertex,
                                               0u,
                                               "drawElementsBaseVertex");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        if (restartResult == MGLPrimitiveRestartEncodeHandled) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        }
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          1u,
                                          basevertex,
                                          0u,
                                          "drawElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         1u,
                                         basevertex,
                                         0u,
                                         "drawElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      1u,
                                      basevertex,
                                      0u,
                                      "drawElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }
    if (emulateQuads) {
        if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                   _device,
                                   gl_element_buffer,
                                   indexBuffer,
                                   type,
                                   offset,
                                   count,
                                   1u,
                                   basevertex,
                                   0u,
                                   mglPolygonModeLineForDrawMode(ctx, mode),
                                   "drawElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives: primitiveType indexCount:count indexType: drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1 baseVertex:basevertex baseInstance:0];
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
}

-(void) mtlDrawRangeElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode start: (GLuint) start end: (GLuint) end count:(GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;
    (void)start;
    (void)end;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:count
                                        indexType:type
                                          indices:indices
                                       baseVertex:basevertex
                                    instanceCount:1
                                     baseInstance:0
                                            label:"drawRangeElementsBaseVertex"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawRangeElementsBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               1u,
                                               basevertex,
                                               0u,
                                               "drawRangeElementsBaseVertex");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        if (restartResult == MGLPrimitiveRestartEncodeHandled) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        }
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          1u,
                                          basevertex,
                                          0u,
                                          "drawRangeElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         1u,
                                         basevertex,
                                         0u,
                                         "drawRangeElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      1u,
                                      basevertex,
                                      0u,
                                      "drawRangeElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }
    if (emulateQuads) {
        if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                   _device,
                                   gl_element_buffer,
                                   indexBuffer,
                                   type,
                                   offset,
                                   count,
                                   1u,
                                   basevertex,
                                   0u,
                                   mglPolygonModeLineForDrawMode(ctx, mode),
                                   "drawRangeElementsBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives: primitiveType indexCount:count indexType: drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1 baseVertex:basevertex baseInstance:0];
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0)];
}

-(void) mtlDrawElementsInstancedBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count:(GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    if (count <= (GLuint)INT_MAX &&
        [self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:(GLsizei)count
                                        indexType:type
                                          indices:indices
                                       baseVertex:basevertex
                                    instanceCount:instancecount
                                     baseInstance:0
                                            label:"drawElementsInstancedBaseVertex"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstancedBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               (GLsizei)count,
                                               (NSUInteger)instancecount,
                                               basevertex,
                                               0u,
                                               "drawElementsInstancedBaseVertex");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        if (restartResult == MGLPrimitiveRestartEncodeHandled) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)count * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          (GLsizei)count,
                                          (NSUInteger)instancecount,
                                          basevertex,
                                          0u,
                                          "drawElementsInstancedBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)count * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         basevertex,
                                         0u,
                                         "drawElementsInstancedBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)count * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      basevertex,
                                      0u,
                                      "drawElementsInstancedBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)count * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateQuads) {
        if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                   _device,
                                   gl_element_buffer,
                                   indexBuffer,
                                   type,
                                   offset,
                                   (GLsizei)count,
                                   (NSUInteger)instancecount,
                                   basevertex,
                                   0u,
                                   mglPolygonModeLineForDrawMode(ctx, mode),
                                   "drawElementsInstancedBaseVertex")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)count * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount baseVertex:basevertex baseInstance:0];
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)count * (uint64_t)MAX(instancecount, 0)];
}

-(void) mtlDrawArraysIndirect: (GLMContext) glm_ctx mode:(GLenum) mode indirect: (const void *) indirect
{
    MTLPrimitiveType primitiveType;

    mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_ENTRY mode=0x%x indirect=%p program=%u",
                (unsigned)mode, indirect,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "drawArraysIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=fully_culled mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "drawArraysIndirect")) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"drawArraysIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    if (mode == GL_QUADS || mode == GL_LINE_LOOP) {
        if (![self prepareEmulatedIndirectCPURead:ctx label:"drawArraysIndirect.quads"]) {
            return;
        }

        DrawArraysIndirectCommand cmd = {0};
        NSUInteger indirectOffset = (NSUInteger)(uintptr_t)indirect;
        if (!mglReadBufferBytes(gl_indirect_buffer,
                                indirectBuffer,
                                indirectOffset,
                                &cmd,
                                sizeof(cmd),
                                mode == GL_LINE_LOOP ? "drawArraysIndirect.lineLoop" : "drawArraysIndirect.quads")) {
            return;
        }
        if (cmd.count == 0u || cmd.instanceCount == 0u) {
            return;
        }
        if (cmd.count > (unsigned int)INT_MAX || cmd.first > (unsigned int)INT_MAX) {
            NSLog(@"MGL WARNING: drawArraysIndirect emulated command exceeds range mode=0x%x count=%u first=%u",
                  (unsigned)mode,
                  cmd.count,
                  cmd.first);
            return;
        }

        BOOL ok = NO;
        if (mode == GL_LINE_LOOP) {
            ok = mglEncodeArrayLineLoop(_renderPassManager.state->currentRenderEncoder,
                                        glm_ctx,
                                        _device,
                                        (GLsizei)cmd.count,
                                        (GLint)cmd.first,
                                        (NSUInteger)cmd.instanceCount,
                                        (NSUInteger)cmd.baseInstance,
                                        "drawArraysIndirect");
        } else if (polygonModePoint) {
            ok = mglEncodeArrayPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                            _device,
                                            mode,
                                            (GLint)cmd.first,
                                            (GLsizei)cmd.count,
                                            (NSUInteger)cmd.instanceCount,
                                            (NSUInteger)cmd.baseInstance,
                                            "drawArraysIndirect");
        } else {
            ok = mglEncodeArrayQuads(_renderPassManager.state->currentRenderEncoder,
                                     _device,
                                     (GLsizei)cmd.count,
                                     (GLint)cmd.first,
                                     (NSUInteger)cmd.instanceCount,
                                     (NSUInteger)cmd.baseInstance,
                                     mglPolygonModeLineForDrawMode(ctx, mode),
                                     "drawArraysIndirect");
        }
        if (ok) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)cmd.count * (uint64_t)cmd.instanceCount];
            mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SUBMIT path=emulated mode=0x%x count=%u instances=%u first=%u baseInstance=%u program=%u",
                        (unsigned)mode, cmd.count, cmd.instanceCount, cmd.first, cmd.baseInstance,
                        (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        }
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: drawArraysIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode);
        return;
    }

    [_renderPassManager.state->currentRenderEncoder drawPrimitives:primitiveType
                           indirectBuffer:indirectBuffer
                     indirectBufferOffset:(NSUInteger)(uintptr_t)indirect];
    [self recordArrayDrawSubmittedMode:mode vertexCount:0u];
    mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SUBMIT path=native mode=0x%x indirect=%p offset=%lu program=%u",
                (unsigned)mode, indirect, (unsigned long)(NSUInteger)(uintptr_t)indirect,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
}

-(void) mtlDrawElementsIndirect: (GLMContext) glm_ctx mode:(GLenum) mode type:(GLenum) type indirect: (const void *) indirect
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_ENTRY mode=0x%x type=0x%x indirect=%p program=%u",
                (unsigned)mode, (unsigned)type, indirect,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "drawElementsIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=fully_culled mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "drawElementsIndirect")) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    // get element buffer
    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_index_type type=0x%x program=%u",
                    (unsigned)type,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type);
        return;
    }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, "drawElementsIndirect")) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=primitive_restart program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsIndirect" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_element_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    // get indirect buffer
    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"drawElementsIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    if (mode == GL_QUADS || mode == GL_LINE_LOOP) {
        if (![self prepareEmulatedIndirectCPURead:ctx label:"drawElementsIndirect.quads"]) {
            return;
        }

        DrawElementsIndirectCommand cmd = {0};
        NSUInteger indirectOffset = (NSUInteger)(uintptr_t)indirect;
        if (!mglReadBufferBytes(gl_indirect_buffer,
                                indirectBuffer,
                                indirectOffset,
                                &cmd,
                                sizeof(cmd),
                                mode == GL_LINE_LOOP ? "drawElementsIndirect.lineLoop" : "drawElementsIndirect.quads")) {
            return;
        }
        if (cmd.count == 0u || cmd.instanceCount == 0u) {
            return;
        }
        if (cmd.count > (unsigned int)INT_MAX) {
            NSLog(@"MGL WARNING: drawElementsIndirect emulated command exceeds range mode=0x%x count=%u",
                  (unsigned)mode,
                  cmd.count);
            return;
        }

        NSUInteger indexStride = mglGLIndexElementSize(type);
        if (indexStride == 0u || (NSUInteger)cmd.first > (NSUIntegerMax / indexStride)) {
            NSLog(@"MGL WARNING: drawElementsIndirect emulated invalid firstIndex=%u stride=%lu",
                  cmd.first,
                  (unsigned long)indexStride);
            return;
        }

        NSUInteger elementOffset = (NSUInteger)cmd.first * indexStride;
        BOOL ok = NO;
        if (mode == GL_LINE_LOOP) {
            ok = mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          type,
                                          elementOffset,
                                          (GLsizei)cmd.count,
                                          (NSUInteger)cmd.instanceCount,
                                          cmd.baseVertex,
                                          (NSUInteger)cmd.baseInstance,
                                          "drawElementsIndirect");
        } else if (polygonModePoint) {
            ok = mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                              _device,
                                              gl_element_buffer,
                                              indexBuffer,
                                              mode,
                                              type,
                                              indexType,
                                              elementOffset,
                                              (GLsizei)cmd.count,
                                              (NSUInteger)cmd.instanceCount,
                                              cmd.baseVertex,
                                              (NSUInteger)cmd.baseInstance,
                                              "drawElementsIndirect");
        } else {
            ok = mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                       _device,
                                       gl_element_buffer,
                                       indexBuffer,
                                       type,
                                       elementOffset,
                                       (GLsizei)cmd.count,
                                       (NSUInteger)cmd.instanceCount,
                                       cmd.baseVertex,
                                       (NSUInteger)cmd.baseInstance,
                                       mglPolygonModeLineForDrawMode(ctx, mode),
                                       "drawElementsIndirect");
        }
        if (ok) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)cmd.count * (uint64_t)cmd.instanceCount];
            mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SUBMIT path=emulated mode=0x%x type=0x%x count=%u instances=%u first=%u baseVertex=%d baseInstance=%u program=%u",
                        (unsigned)mode, (unsigned)type, cmd.count, cmd.instanceCount, cmd.first,
                        cmd.baseVertex, cmd.baseInstance,
                        (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        }
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: drawElementsIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode);
        return;
    }

    NSUInteger indexBufferOffset = 0u;
    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &indexBufferOffset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=prepare_index_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    // draw indexed primitive
    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType
                                       indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer
                               indexBufferOffset:indexBufferOffset
                                  indirectBuffer:indirectBuffer
                            indirectBufferOffset:(NSUInteger)(uintptr_t)indirect];
    [self recordElementDrawSubmittedMode:mode indexCount:0u];
    mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SUBMIT path=native mode=0x%x type=0x%x indirect=%p offset=%lu program=%u",
                (unsigned)mode, (unsigned)type, indirect,
                (unsigned long)(NSUInteger)(uintptr_t)indirect,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
}

-(void) mtlDrawArraysInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance
{
    MTLPrimitiveType primitiveType;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:first
                                            count:count
                                        indexType:0
                                          indices:NULL
                                       baseVertex:0
                                    instanceCount:instancecount
                                     baseInstance:baseinstance
                                            label:"drawArraysInstancedBaseInstance"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint) {
        if (mglEncodeArrayPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                       _device,
                                       mode,
                                       first,
                                       count,
                                       (NSUInteger)instancecount,
                                       (NSUInteger)baseinstance,
                                       "drawArraysInstancedBaseInstance")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    if (mode == GL_TRIANGLE_FAN) {
        if (mglEncodeArrayTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      count,
                                      first,
                                      (NSUInteger)instancecount,
                                      (NSUInteger)baseinstance,
                                      "drawArraysInstancedBaseInstance")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }
    if (mode == GL_LINE_LOOP) {
        if (mglEncodeArrayLineLoop(_renderPassManager.state->currentRenderEncoder,
                                   glm_ctx,
                                   _device,
                                   count,
                                   first,
                                   (NSUInteger)instancecount,
                                   (NSUInteger)baseinstance,
                                   "drawArraysInstancedBaseInstance")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }
    if (mode == GL_QUADS) {
        if (mglEncodeArrayQuads(_renderPassManager.state->currentRenderEncoder,
                                _device,
                                count,
                                first,
                                (NSUInteger)instancecount,
                                (NSUInteger)baseinstance,
                                mglPolygonModeLineForDrawMode(ctx, mode),
                                "drawArraysInstancedBaseInstance")) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    [_renderPassManager.state->currentRenderEncoder drawPrimitives:primitiveType vertexStart:first vertexCount:count instanceCount:instancecount baseInstance:baseinstance];
    [self recordArrayDrawSubmittedMode:mode vertexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
}

-(void) mtlDrawElementsInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode  count: (GLsizei) count type:(GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:count
                                        indexType:type
                                          indices:indices
                                       baseVertex:0
                                    instanceCount:instancecount
                                     baseInstance:baseinstance
                                            label:"drawElementsInstancedBaseInstance"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstancedBaseInstance" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               (NSUInteger)instancecount,
                                               0,
                                               (NSUInteger)baseinstance,
                                               "drawElementsInstancedBaseInstance");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        if (restartResult == MGLPrimitiveRestartEncodeHandled) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          (NSUInteger)instancecount,
                                          0,
                                          (NSUInteger)baseinstance,
                                          "drawElementsInstancedBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         0,
                                         (NSUInteger)baseinstance,
                                         "drawElementsInstancedBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      0,
                                      (NSUInteger)baseinstance,
                                      "drawElementsInstancedBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateQuads) {
        if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                   _device,
                                   gl_element_buffer,
                                   indexBuffer,
                                   type,
                                   offset,
                                   count,
                                   (NSUInteger)instancecount,
                                   0,
                                   (NSUInteger)baseinstance,
                                   mglPolygonModeLineForDrawMode(ctx, mode),
                                   "drawElementsInstancedBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    // for now lets just ignore the range data and use drawIndexedPrimitives
    //
    // in the future it would be an idea to use temp buffers for large buffers that would wire
    // to much memory down.. like a million point galaxy drawing
    //
    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount baseVertex:0 baseInstance:baseinstance];
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
}

-(void) mtlDrawElementsInstancedBaseVertexBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type:(GLenum) type indices:(const void *)indices
                                                        instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex baseinstance:(GLuint) baseinstance
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    if ([self handleTessellationPatchDrawIfNeeded:glm_ctx
                                             mode:&mode
                                            first:0
                                            count:count
                                        indexType:type
                                          indices:indices
                                       baseVertex:basevertex
                                    instanceCount:instancecount
                                     baseInstance:baseinstance
                                            label:"drawElementsInstancedBaseVertexBaseInstance"]) {
        return;
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstancedBaseVertexBaseInstance" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               (NSUInteger)instancecount,
                                               basevertex,
                                               (NSUInteger)baseinstance,
                                               "drawElementsInstancedBaseVertexBaseInstance");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        if (restartResult == MGLPrimitiveRestartEncodeHandled) {
            [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        }
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          (NSUInteger)instancecount,
                                          basevertex,
                                          (NSUInteger)baseinstance,
                                          "drawElementsInstancedBaseVertexBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         basevertex,
                                         (NSUInteger)baseinstance,
                                         "drawElementsInstancedBaseVertexBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      basevertex,
                                      (NSUInteger)baseinstance,
                                      "drawElementsInstancedBaseVertexBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }
    if (emulateQuads) {
        if (!mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                   _device,
                                   gl_element_buffer,
                                   indexBuffer,
                                   type,
                                   offset,
                                   count,
                                   (NSUInteger)instancecount,
                                   basevertex,
                                   (NSUInteger)baseinstance,
                                   mglPolygonModeLineForDrawMode(ctx, mode),
                                   "drawElementsInstancedBaseVertexBaseInstance")) {
            return;
        }
        [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    // for now lets just ignore the range data and use drawIndexedPrimitives
    //
    // in the future it would be an idea to use temp buffers for large buffers that would wire
    // to much memory down.. like a million point galaxy drawing
    //
    [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount baseVertex:basevertex baseInstance:baseinstance];
    [self recordElementDrawSubmittedMode:mode indexCount:(uint64_t)MAX(count, 0) * (uint64_t)MAX(instancecount, 0)];
}

-(void) mtlMultiDrawArrays: (GLMContext)glm_ctx mode:(GLenum) mode first:(const GLint *)first count:(const GLsizei *)count drawcount:(GLsizei) drawcount
{
    MTLPrimitiveType primitiveType;

    if (mode == GL_PATCHES) {
        BOOL handled = NO;
        BOOL sawPositiveCount = NO;
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] <= 0) {
                continue;
            }
            sawPositiveCount = YES;
            if (![self handleTessellationPatchDrawIfNeeded:glm_ctx
                                                       mode:&mode
                                                      first:first[i]
                                                      count:count[i]
                                                 indexType:0
                                                   indices:NULL
                                                baseVertex:0
                                             instanceCount:1
                                              baseInstance:0
                                                      label:"multiDrawArrays"]) {
                handled = NO;
                break;
            }
            handled = YES;
        }
        /* Program state is stable across one multi-draw, so helper success
         * cannot turn into helper failure on a later positive-count draw. */
        if (handled || !sawPositiveCount) {
            return;
        }
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint) {
        uint64_t submittedVertices = 0u;
        for (int i = 0; i < drawcount; i++) {
            if (mglEncodeArrayPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                           _device,
                                           mode,
                                           first[i],
                                           count[i],
                                           1u,
                                           0u,
                                           "multiDrawArrays")) {
                submittedVertices += (uint64_t)MAX(count[i], 0);
            }
        }
        if (submittedVertices > 0u) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:submittedVertices];
        }
        return;
    }

    if (mode == GL_TRIANGLE_FAN) {
        uint64_t submittedVertices = 0u;
        for (int i = 0; i < drawcount; i++) {
            if (mglEncodeArrayTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                          _device,
                                          count[i],
                                          first[i],
                                          1u,
                                          0u,
                                          "multiDrawArrays")) {
                submittedVertices += (uint64_t)MAX(count[i], 0);
            }
        }
        if (submittedVertices > 0u) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:submittedVertices];
        }
        return;
    }
    if (mode == GL_LINE_LOOP) {
        uint64_t submittedVertices = 0u;
        for (int i = 0; i < drawcount; i++) {
            if (mglEncodeArrayLineLoop(_renderPassManager.state->currentRenderEncoder,
                                       glm_ctx,
                                       _device,
                                       count[i],
                                       first[i],
                                       1u,
                                       0u,
                                       "multiDrawArrays")) {
                submittedVertices += (uint64_t)MAX(count[i], 0);
            }
        }
        if (submittedVertices > 0u) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:submittedVertices];
        }
        return;
    }
    if (mode == GL_QUADS) {
        uint64_t submittedVertices = 0u;
        for (int i = 0; i < drawcount; i++) {
            if (mglEncodeArrayQuads(_renderPassManager.state->currentRenderEncoder,
                                    _device,
                                    count[i],
                                    first[i],
                                    1u,
                                    0u,
                                    mglPolygonModeLineForDrawMode(ctx, mode),
                                    "multiDrawArrays")) {
                submittedVertices += (uint64_t)MAX(count[i], 0);
            }
        }
        if (submittedVertices > 0u) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:submittedVertices];
        }
        return;
    }

    primitiveType = getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    uint64_t submittedVertices = 0u;
    for(int i=0; i<drawcount; i++)
    {
         [_renderPassManager.state->currentRenderEncoder drawPrimitives: primitiveType
                                  vertexStart: first[i]
                                  vertexCount: count[i]];
         submittedVertices += (uint64_t)MAX(count[i], 0);
    }
    if (submittedVertices > 0u) {
        [self recordArrayDrawSubmittedMode:mode vertexCount:submittedVertices];
    }
}

-(void) mtlMultiDrawElements: (GLMContext)glm_ctx mode:(GLenum) mode count:(const GLsizei *)count type:(GLenum)type indices:(const void *const*)indices drawcount:(GLsizei) drawcount
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    if (mode == GL_PATCHES) {
        BOOL handled = NO;
        BOOL sawPositiveCount = NO;
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] <= 0) {
                continue;
            }
            sawPositiveCount = YES;
            if (![self handleTessellationPatchDrawIfNeeded:glm_ctx
                                                       mode:&mode
                                                      first:0
                                                      count:count[i]
                                                 indexType:type
                                                   indices:indices ? indices[i] : NULL
                                                baseVertex:0
                                             instanceCount:1
                                              baseInstance:0
                                                      label:"multiDrawElements"]) {
                handled = NO;
                break;
            }
            handled = YES;
        }
        /* Program state is stable across one multi-draw, so helper success
         * cannot turn into helper failure on a later positive-count draw. */
        if (handled || !sawPositiveCount) {
            return;
        }
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"multiDrawElements" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    uint64_t submittedIndices = 0u;
    for(int i=0; i<drawcount; i++)
    {
        NSUInteger offset = (NSUInteger)(uintptr_t)indices[i];
        MGLPrimitiveRestartEncodeResult restartResult =
            mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                                   _device,
                                                   ctx,
                                                   gl_element_buffer,
                                                   indexBuffer,
                                                   mode,
                                                   primitiveType,
                                                   type,
                                                   indexType,
                                                   offset,
                                                   count[i],
                                                   1u,
                                                   0,
                                                   0u,
                                                   "multiDrawElements");
        if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
            if (restartResult == MGLPrimitiveRestartEncodeHandled) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }

        if (polygonModePoint) {
            if (mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                             _device,
                                             gl_element_buffer,
                                             indexBuffer,
                                             mode,
                                             type,
                                             indexType,
                                             offset,
                                             count[i],
                                             1u,
                                             0,
                                             0u,
                                             "multiDrawElements")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }

        if (emulateTriangleFan) {
            if (mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                            _device,
                                            gl_element_buffer,
                                            indexBuffer,
                                            type,
                                            offset,
                                            count[i],
                                            1u,
                                            0,
                                            0u,
                                            "multiDrawElements")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }
        if (emulateLineLoop) {
            if (mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count[i],
                                         1u,
                                         0,
                                         0u,
                                         "multiDrawElements")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }
        if (emulateQuads) {
            if (mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count[i],
                                      1u,
                                      0,
                                      0u,
                                      mglPolygonModeLineForDrawMode(ctx, mode),
                                      "multiDrawElements")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }

        MTLIndexType drawIndexType = indexType;
        id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                      gl_element_buffer,
                                                                      indexBuffer,
                                                                      type,
                                                                      &offset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer) {
            continue;
        }

        [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count[i] indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1];
        submittedIndices += (uint64_t)MAX(count[i], 0);
    }
    if (submittedIndices > 0u) {
        [self recordElementDrawSubmittedMode:mode indexCount:submittedIndices];
    }
}

-(void) mtlMultiDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (const GLsizei *) count type: (GLenum) type indices:(const void *const *)indices drawcount:(GLsizei) drawcount basevertex:(const GLint *) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    if (mode == GL_PATCHES) {
        BOOL handled = NO;
        BOOL sawPositiveCount = NO;
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] <= 0) {
                continue;
            }
            sawPositiveCount = YES;
            if (![self handleTessellationPatchDrawIfNeeded:glm_ctx
                                                       mode:&mode
                                                      first:0
                                                      count:count[i]
                                                 indexType:type
                                                   indices:indices ? indices[i] : NULL
                                                baseVertex:basevertex ? basevertex[i] : 0
                                             instanceCount:1
                                              baseInstance:0
                                                      label:"multiDrawElementsBaseVertex"]) {
                handled = NO;
                break;
            }
            handled = YES;
        }
        /* Program state is stable across one multi-draw, so helper success
         * cannot turn into helper failure on a later positive-count draw. */
        if (handled || !sawPositiveCount) {
            return;
        }
    }

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : (emulateQuads ? MTLPrimitiveTypeTriangle : getMTLPrimitiveType(mode))));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    // element buffer
    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"multiDrawElementsBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;


    uint64_t submittedIndices = 0u;
    for(int i=0; i<drawcount; i++)
    {
        NSUInteger offset = (NSUInteger)(uintptr_t)indices[i];
        MGLPrimitiveRestartEncodeResult restartResult =
            mglEncodePrimitiveRestartedElementDraw(_renderPassManager.state->currentRenderEncoder,
                                                   _device,
                                                   ctx,
                                                   gl_element_buffer,
                                                   indexBuffer,
                                                   mode,
                                                   primitiveType,
                                                   type,
                                                   indexType,
                                                   offset,
                                                   count[i],
                                                   1u,
                                                   basevertex[i],
                                                   0u,
                                                   "multiDrawElementsBaseVertex");
        if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
            if (restartResult == MGLPrimitiveRestartEncodeHandled) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }

        if (polygonModePoint) {
            if (mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                             _device,
                                             gl_element_buffer,
                                             indexBuffer,
                                             mode,
                                             type,
                                             indexType,
                                             offset,
                                             count[i],
                                             1u,
                                             basevertex[i],
                                             0u,
                                             "multiDrawElementsBaseVertex")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }

        if (emulateTriangleFan) {
            if (mglEncodeElementTriangleFan(_renderPassManager.state->currentRenderEncoder,
                                            _device,
                                            gl_element_buffer,
                                            indexBuffer,
                                            type,
                                            offset,
                                            count[i],
                                            1u,
                                            basevertex[i],
                                            0u,
                                            "multiDrawElementsBaseVertex")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }
        if (emulateLineLoop) {
            if (mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count[i],
                                         1u,
                                         basevertex[i],
                                         0u,
                                         "multiDrawElementsBaseVertex")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }
        if (emulateQuads) {
            if (mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count[i],
                                      1u,
                                      basevertex[i],
                                      0u,
                                      mglPolygonModeLineForDrawMode(ctx, mode),
                                      "multiDrawElementsBaseVertex")) {
                submittedIndices += (uint64_t)MAX(count[i], 0);
            }
            continue;
        }

        MTLIndexType drawIndexType = indexType;
        id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                      gl_element_buffer,
                                                                      indexBuffer,
                                                                      type,
                                                                      &offset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer) {
            continue;
        }

        [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count[i] indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1 baseVertex:basevertex[i] baseInstance:0];
        submittedIndices += (uint64_t)MAX(count[i], 0);
    }
    if (submittedIndices > 0u) {
        [self recordElementDrawSubmittedMode:mode indexCount:submittedIndices];
    }
}

-(void) mtlMultiDrawArraysIndirect: (GLMContext)glm_ctx mode:(GLenum) mode indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride
{
    MTLPrimitiveType primitiveType;

    mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_ENTRY mode=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, indirect, (int)drawcount, (int)stride,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "multiDrawArraysIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=fully_culled mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "multiDrawArraysIndirect")) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"multiDrawArraysIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    if (mode == GL_QUADS || mode == GL_LINE_LOOP) {
        if (stride < 0) {
            NSLog(@"MGL WARNING: multiDrawArraysIndirect emulated invalid negative stride=%d",
                  (int)stride);
            return;
        }
        if (![self prepareEmulatedIndirectCPURead:ctx label:mode == GL_LINE_LOOP ? "multiDrawArraysIndirect.lineLoop" : "multiDrawArraysIndirect.quads"]) {
            return;
        }

        NSUInteger commandStride = stride ? (NSUInteger)stride : sizeof(DrawArraysIndirectCommand);
        NSUInteger baseOffset = (NSUInteger)(uintptr_t)indirect;
        uint64_t submittedVertices = 0u;
        for(int i=0; i<drawcount; i++)
        {
            if ((NSUInteger)i > (NSUIntegerMax - baseOffset) / commandStride) {
                NSLog(@"MGL WARNING: multiDrawArraysIndirect GL_QUADS command offset overflow draw=%d stride=%lu",
                      i,
                      (unsigned long)commandStride);
                break;
            }

            DrawArraysIndirectCommand cmd = {0};
            NSUInteger offset = baseOffset + ((NSUInteger)i * commandStride);
            if (!mglReadBufferBytes(gl_indirect_buffer,
                                    indirectBuffer,
                                    offset,
                                    &cmd,
                                    sizeof(cmd),
                                    mode == GL_LINE_LOOP ? "multiDrawArraysIndirect.lineLoop" : "multiDrawArraysIndirect.quads")) {
                break;
            }
            if (cmd.count == 0u || cmd.instanceCount == 0u) {
                continue;
            }
            if (cmd.count > (unsigned int)INT_MAX || cmd.first > (unsigned int)INT_MAX) {
                NSLog(@"MGL WARNING: multiDrawArraysIndirect emulated command exceeds range mode=0x%x draw=%d count=%u first=%u",
                      (unsigned)mode,
                      i,
                      cmd.count,
                      cmd.first);
                continue;
            }

            BOOL ok = NO;
            if (mode == GL_LINE_LOOP) {
                ok = mglEncodeArrayLineLoop(_renderPassManager.state->currentRenderEncoder,
                                            glm_ctx,
                                            _device,
                                            (GLsizei)cmd.count,
                                            (GLint)cmd.first,
                                            (NSUInteger)cmd.instanceCount,
                                            (NSUInteger)cmd.baseInstance,
                                            "multiDrawArraysIndirect");
            } else if (polygonModePoint) {
                ok = mglEncodeArrayPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                                _device,
                                                mode,
                                                (GLint)cmd.first,
                                                (GLsizei)cmd.count,
                                                (NSUInteger)cmd.instanceCount,
                                                (NSUInteger)cmd.baseInstance,
                                                "multiDrawArraysIndirect");
            } else {
                ok = mglEncodeArrayQuads(_renderPassManager.state->currentRenderEncoder,
                                         _device,
                                         (GLsizei)cmd.count,
                                         (GLint)cmd.first,
                                         (NSUInteger)cmd.instanceCount,
                                         (NSUInteger)cmd.baseInstance,
                                         mglPolygonModeLineForDrawMode(ctx, mode),
                                         "multiDrawArraysIndirect");
            }
            if (ok) {
                submittedVertices += (uint64_t)cmd.count * (uint64_t)cmd.instanceCount;
            }
        }
        if (submittedVertices > 0u) {
            [self recordArrayDrawSubmittedMode:mode vertexCount:submittedVertices];
        }
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SUBMIT path=emulated mode=0x%x drawcount=%d submittedVertices=%llu program=%u",
                    (unsigned)mode, (int)drawcount,
                    (unsigned long long)submittedVertices,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: multiDrawArraysIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode);
        return;
    }

    for(int i=0; i<drawcount; i++)
    {
        size_t offset;

        if (stride)
        {
            offset = (char *)((char *)indirect + i * stride) - (char *)NULL;
        }
        else
        {
            offset = (char *)((char *)indirect + i * sizeof(DrawArraysIndirectCommand)) - (char *)NULL;
        }

        [_renderPassManager.state->currentRenderEncoder drawPrimitives:primitiveType indirectBuffer:indirectBuffer indirectBufferOffset:offset];
    }
    if (drawcount > 0) {
        [self recordArrayDrawSubmittedMode:mode vertexCount:0u];
    }
    mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SUBMIT path=native mode=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, indirect, (int)drawcount, (int)stride,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
}

-(void) mtlMultiDrawElementsIndirect: (GLMContext)glm_ctx mode:(GLenum) mode type:(GLenum)type indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_ENTRY mode=0x%x type=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, (unsigned)type, indirect, (int)drawcount, (int)stride,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "multiDrawElementsIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=fully_culled program=%u mode=0x%x",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u), (unsigned)mode);
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "multiDrawElementsIndirect")) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    // get element buffer
    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_index_type type=0x%x program=%u",
                    (unsigned)type,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type);
        return;
    }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, "multiDrawElementsIndirect")) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=primitive_restart program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"multiDrawElementsIndirect" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_element_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    // get indirect buffer
    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"multiDrawElementsIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    if (mode == GL_QUADS || mode == GL_LINE_LOOP) {
        if (stride < 0) {
            NSLog(@"MGL WARNING: multiDrawElementsIndirect emulated invalid negative stride=%d",
                  (int)stride);
            return;
        }
        if (![self prepareEmulatedIndirectCPURead:ctx label:mode == GL_LINE_LOOP ? "multiDrawElementsIndirect.lineLoop" : "multiDrawElementsIndirect.quads"]) {
            return;
        }

        NSUInteger indexStride = mglGLIndexElementSize(type);
        if (indexStride == 0u) {
            return;
        }

        NSUInteger commandStride = stride ? (NSUInteger)stride : sizeof(DrawElementsIndirectCommand);
        NSUInteger baseOffset = (NSUInteger)(uintptr_t)indirect;
        uint64_t submittedIndices = 0u;
        for(int i=0; i<drawcount; i++)
        {
            if ((NSUInteger)i > (NSUIntegerMax - baseOffset) / commandStride) {
                NSLog(@"MGL WARNING: multiDrawElementsIndirect GL_QUADS command offset overflow draw=%d stride=%lu",
                      i,
                      (unsigned long)commandStride);
                break;
            }

            DrawElementsIndirectCommand cmd = {0};
            NSUInteger offset = baseOffset + ((NSUInteger)i * commandStride);
            if (!mglReadBufferBytes(gl_indirect_buffer,
                                    indirectBuffer,
                                    offset,
                                    &cmd,
                                    sizeof(cmd),
                                    mode == GL_LINE_LOOP ? "multiDrawElementsIndirect.lineLoop" : "multiDrawElementsIndirect.quads")) {
                break;
            }
            if (cmd.count == 0u || cmd.instanceCount == 0u) {
                continue;
            }
            if (cmd.count > (unsigned int)INT_MAX) {
                NSLog(@"MGL WARNING: multiDrawElementsIndirect emulated command exceeds range mode=0x%x draw=%d count=%u",
                      (unsigned)mode,
                      i,
                      cmd.count);
                continue;
            }
            if ((NSUInteger)cmd.first > (NSUIntegerMax / indexStride)) {
                NSLog(@"MGL WARNING: multiDrawElementsIndirect emulated firstIndex overflow draw=%d first=%u stride=%lu",
                      i,
                      cmd.first,
                      (unsigned long)indexStride);
                continue;
            }

            NSUInteger elementOffset = (NSUInteger)cmd.first * indexStride;
            BOOL ok = NO;
            if (mode == GL_LINE_LOOP) {
                ok = mglEncodeElementLineLoop(_renderPassManager.state->currentRenderEncoder,
                                              _device,
                                              gl_element_buffer,
                                              indexBuffer,
                                              type,
                                              elementOffset,
                                              (GLsizei)cmd.count,
                                              (NSUInteger)cmd.instanceCount,
                                              cmd.baseVertex,
                                              (NSUInteger)cmd.baseInstance,
                                              "multiDrawElementsIndirect");
            } else if (polygonModePoint) {
                ok = mglEncodeElementPolygonPoint(_renderPassManager.state->currentRenderEncoder,
                                                  _device,
                                                  gl_element_buffer,
                                                  indexBuffer,
                                                  mode,
                                                  type,
                                                  indexType,
                                                  elementOffset,
                                                  (GLsizei)cmd.count,
                                                  (NSUInteger)cmd.instanceCount,
                                                  cmd.baseVertex,
                                                  (NSUInteger)cmd.baseInstance,
                                                  "multiDrawElementsIndirect");
            } else {
                ok = mglEncodeElementQuads(_renderPassManager.state->currentRenderEncoder,
                                           _device,
                                           gl_element_buffer,
                                           indexBuffer,
                                           type,
                                           elementOffset,
                                           (GLsizei)cmd.count,
                                           (NSUInteger)cmd.instanceCount,
                                           cmd.baseVertex,
                                           (NSUInteger)cmd.baseInstance,
                                           mglPolygonModeLineForDrawMode(ctx, mode),
                                           "multiDrawElementsIndirect");
            }
            if (ok) {
                submittedIndices += (uint64_t)cmd.count * (uint64_t)cmd.instanceCount;
            }
        }
        if (submittedIndices > 0u) {
            [self recordElementDrawSubmittedMode:mode indexCount:submittedIndices];
        }
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SUBMIT path=emulated mode=0x%x type=0x%x drawcount=%d submittedIndices=%llu program=%u",
                    (unsigned)mode, (unsigned)type, (int)drawcount,
                    (unsigned long long)submittedIndices,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: multiDrawElementsIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode);
        return;
    }

    NSUInteger indexBufferOffset = 0u;
    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &indexBufferOffset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=prepare_index_buffer program=%u",
                    (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
        return;
    }

    for(int i=0; i<drawcount; i++)
    {
        size_t offset;

        if (stride)
        {
            offset = (char *)((char *)indirect + i * stride) - (char *)NULL;
        }
        else
        {
            offset = (char *)((char *)indirect + i * sizeof(DrawElementsIndirectCommand)) - (char *)NULL;
        }

        // draw indexed primitive
        [_renderPassManager.state->currentRenderEncoder drawIndexedPrimitives:primitiveType indexType:drawIndexType indexBuffer: drawIndexBuffer indexBufferOffset:indexBufferOffset indirectBuffer:indirectBuffer indirectBufferOffset:offset];
    }
    if (drawcount > 0) {
        [self recordElementDrawSubmittedMode:mode indexCount:0u];
    }
    mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SUBMIT path=native mode=0x%x type=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, (unsigned)type, indirect, (int)drawcount, (int)stride,
                (unsigned)(glm_ctx ? MGL_STATE(glm_ctx)->program_name : 0u));
}

@end
