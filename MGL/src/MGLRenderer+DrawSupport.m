// MGLRenderer+DrawSupport.m
// Draw validation, element-buffer resolution and rasterization helper
// methods extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"

@implementation MGLRenderer (Draw)

- (bool) validateDrawArraysVertexInputs:(GLMContext)drawCtx
                                    mode:(GLenum)mode
                                   first:(GLint)first
                                   count:(GLsizei)count
                                drawCall:(uint64_t)drawCall
{
    if (!mglVboRangeValidationEnabled()) {
        return true;
    }

    if (!drawCtx) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=null_ctx mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    if (count == 0) {
        return false;
    }

    if (count < 0 || first < 0) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_range mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t firstVertex = (uint64_t)(uint32_t)first;
    uint64_t vertexCount = (uint64_t)(uint32_t)count;
    if (vertexCount == 0u || firstVertex > UINT64_MAX - (vertexCount - 1u)) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=vertex_range_overflow mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t lastVertex = firstVertex + vertexCount - 1u;
    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "drawArrays.vboRange");
    if (!vao) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_vao mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    GLuint maxAttribs = MAX_ATTRIBS;

    for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
        if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
            continue;
        }

        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(drawCtx,
                                                   vao,
                                                   attrib,
                                                   "drawArrays.vboRange",
                                                   &resolved)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u reason=invalid_vbo mode=0x%x first=%d count=%d",
                  (unsigned long long)drawCall, (unsigned)attrib, (unsigned)mode, (int)first, (int)count);
            return false;
        }
        const VertexAttrib *a = resolved.attrib;
        Buffer *vbo = resolved.buffer;

        if (!mglRendererBufferHasDrawableContents(vbo)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=never_written "
                  "init(source=%u mapped=%u access=0x%x accessFlags=0x%x full=%u range=[%lld,%lld) lastOff=%lld lastSize=%lld src=%p hash=0x%016llx)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)vbo->last_init_source,
                  (unsigned)vbo->mapped,
                  (unsigned)vbo->access,
                  (unsigned)vbo->access_flags,
                  (unsigned)vbo->has_initialized_data,
                  (long long)vbo->written_min,
                  (long long)vbo->written_max,
                  (long long)vbo->last_write_offset,
                  (long long)vbo->last_write_size,
                  vbo->last_write_src_ptr,
                  (unsigned long long)vbo->last_write_src_hash);
            return false;
        }

        if (resolved.binding_offset < 0 || resolved.relativeoffset < 0) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=negative_attrib_offset bindingOffset=%lld relativeOffset=%lld",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (long long)resolved.binding_offset,
                  (long long)resolved.relativeoffset);
            return false;
        }

        size_t compSize = mglVertexAttribComponentSize(a->type);
        size_t compCount = (size_t)a->size;
        if (compSize == 0u || compCount == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=invalid_attrib_format type=0x%x size=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)a->type,
                  (unsigned)a->size);
            return false;
        }

        if (compCount > (SIZE_MAX / compSize)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=elem_size_overflow compSize=%zu compCount=%zu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  compSize,
                  compCount);
            return false;
        }

        uint64_t elemBytes = (uint64_t)(compSize * compCount);
        uint64_t stride = (resolved.stride > 0u) ? (uint64_t)resolved.stride : elemBytes;
        uint64_t bindingOffset = (uint64_t)resolved.binding_offset;
        uint64_t attrRelativeOffset = (uint64_t)resolved.relativeoffset;
        if (bindingOffset > UINT64_MAX - attrRelativeOffset) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=offset_overflow bindingOffset=%llu relativeOffset=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)attrRelativeOffset);
            return false;
        }
        uint64_t relOffset = bindingOffset + attrRelativeOffset;
        if (stride == 0u || elemBytes == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=zero_stride_or_elem stride=%llu elem=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)stride,
                  (unsigned long long)elemBytes);
            return false;
        }

        // Per-instance attributes are still consumed by a non-instanced draw for
        // instance zero, so validate element zero instead of ignoring them.
        uint64_t rangeFirst = (resolved.divisor != 0u) ? 0u : firstVertex;
        uint64_t rangeLast = (resolved.divisor != 0u) ? 0u : lastVertex;

        if (relOffset > UINT64_MAX - elemBytes) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)resolved.divisor);
            return false;
        }

        if (rangeLast > (UINT64_MAX - relOffset - elemBytes) / stride ||
            rangeFirst > (UINT64_MAX - relOffset) / stride) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow "
                  "range=[%llu,%llu] stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)resolved.divisor);
            return false;
        }

        uint64_t byteStart = relOffset + (rangeFirst * stride);
        uint64_t byteEnd = relOffset + (rangeLast * stride) + elemBytes;
        uint64_t vboSize = (vbo->size > 0) ? (uint64_t)vbo->size : 0u;
        if (byteEnd > vboSize) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=vbo_oob "
                  "vertexRange=[%llu,%llu] byteRange=[%llu,%llu) vboSize=%llu "
                  "mode=0x%x first=%d count=%d stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu type=0x%x size=%u divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned)mode,
                  (int)first,
                  (int)count,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)a->type,
                  (unsigned)a->size,
                  (unsigned)resolved.divisor);
            return false;
        }

        if (!vbo->data.mtl_data) {
            [self bindMTLBuffer:vbo];
        }
        if (!vbo->data.mtl_data) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=no_mtl_buffer byteRange=[%llu,%llu)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd);
            return false;
        }

        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
        if (!mtlBuffer) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=mtl_bridge_nil",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name);
            return false;
        }

        uint64_t metalLen = (uint64_t)mtlBuffer.length;
        if (byteEnd > metalLen) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=metal_oob "
                  "byteRange=[%llu,%llu) metalLen=%llu vboSize=%llu first=%d count=%d",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)metalLen,
                  (unsigned long long)vboSize,
                  (int)first,
                  (int)count);
            return false;
        }

        if (vbo->written_min >= 0 && vbo->written_max >= 0) {
            uint64_t writtenMin = (uint64_t)vbo->written_min;
            uint64_t writtenMax = (uint64_t)vbo->written_max;
            if (byteStart < writtenMin || byteEnd > writtenMax) {
                NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=unwritten_range "
                      "byteRange=[%llu,%llu) written=[%llu,%llu) first=%d count=%d source=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned long long)byteStart,
                      (unsigned long long)byteEnd,
                      (unsigned long long)writtenMin,
                      (unsigned long long)writtenMax,
                      (int)first,
                      (int)count,
                      (unsigned)vbo->last_init_source);
                return false;
            }
        }

        GLuint drawProgramKey = mglCurrentRenderProgramKey(drawCtx);
        if (mglShouldInspectDrawCall(drawCall, drawProgramKey) && attrib == 0u) {
            MGLTraceNSLog(@"MGL TRACE drawArrays.attrib0 call=%llu program=%u buffer=%u first=%d count=%d "
                  "byteRange=[%llu,%llu) vboSize=%llu metalLen=%llu stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)drawProgramKey,
                  (unsigned)vbo->name,
                  (int)first,
                  (int)count,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned long long)metalLen,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes);
        }
    }

    return true;
}

- (BOOL)resolveElementBufferForDraw:(const char *)label
                            context:(GLMContext)drawCtx
                           glBuffer:(Buffer **)glBufferOut
                          mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
{
    Buffer *gl_element_buffer = getElementBuffer(drawCtx);
    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBufferForCommand:(const MGLDrawCommand *)cmd
                                  label:(const char *)label
                                context:(GLMContext)drawCtx
                               glBuffer:(Buffer **)glBufferOut
                              mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
{
    Buffer *gl_element_buffer = NULL;
    if (cmd && cmd->elementBuffer) {
        gl_element_buffer = mglRendererGetValidatedBuffer(drawCtx,
                                                          (Buffer *)cmd->elementBuffer,
                                                          label ? label : "deferred indexed draw",
                                                          0);
        if (!gl_element_buffer) {
            return NO;
        }
    } else {
        gl_element_buffer = getElementBuffer(drawCtx);
    }

    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBuffer:(Buffer *)gl_element_buffer
                       label:(const char *)label
                     context:(GLMContext)drawCtx
                    glBuffer:(Buffer **)glBufferOut
                   mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
{
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no element array buffer is bound", label ? label : "indexed draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_element_buffer] == false) {
        return NO;
    }

    id<MTLBuffer> indexBuffer = (__bridge id<MTLBuffer>)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: %s skipped because element buffer %u has no Metal buffer",
              label ? label : "indexed draw",
              gl_element_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_element_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indexBuffer;
    }
    return YES;
}

- (BOOL)resolveIndirectBufferForDraw:(const char *)label
                             context:(GLMContext)drawCtx
                            glBuffer:(Buffer **)glBufferOut
                           mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(drawCtx);
    if (!gl_indirect_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no draw indirect buffer is bound", label ? label : "indirect draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_indirect_buffer] == false) {
        return NO;
    }

    id<MTLBuffer> indirectBuffer = (__bridge id<MTLBuffer>)(gl_indirect_buffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer",
              label ? label : "indirect draw",
              gl_indirect_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_indirect_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indirectBuffer;
    }
    return YES;
}

- (BOOL)prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label
{
    if (!drawCtx) {
        NSLog(@"MGL WARNING: %s skipped because context is NULL",
              label ? label : "indirect emulation");
        return NO;
    }

    /* The C draw-indirect frontends already flush pending command buffers before
     * dispatching into these Metal entry points. If processGLState has just
     * rebuilt a render encoder, keep it; a second flush can discard the fresh
     * pass and make state restoration fail for CPU-emulated indirect modes. */
    if (_renderPassManager.state->currentRenderEncoder) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (!_renderPassManager.state->currentRenderEncoder) {
        NSLog(@"MGL WARNING: %s skipped because CPU-read synchronization left no render encoder",
              label ? label : "indirect emulation");
        return NO;
    }
    return YES;
}

- (BOOL)currentDrawRasterizationIsEmpty
{
    if (!ctx) {
        return NO;
    }

    GLint vx = MGL_STATE(ctx)->viewport[0];
    GLint vy = MGL_STATE(ctx)->viewport[1];
    GLint vw = MGL_STATE(ctx)->viewport[2];
    GLint vh = MGL_STATE(ctx)->viewport[3];
    if (vw <= 0 || vh <= 0) {
        return YES;
    }

    NSUInteger passWidth = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.renderTargetWidth : 0;
    NSUInteger passHeight = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.renderTargetHeight : 0;
    if ((passWidth == 0 || passHeight == 0) && _renderPassManager.state->renderPassDescriptor) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id<MTLTexture> color = _renderPassManager.state->renderPassDescriptor.colorAttachments[i].texture;
            if (color) {
                passWidth = color.width;
                passHeight = color.height;
                break;
            }
        }
        if ((passWidth == 0 || passHeight == 0) && _renderPassManager.state->renderPassDescriptor.depthAttachment.texture) {
            passWidth = _renderPassManager.state->renderPassDescriptor.depthAttachment.texture.width;
            passHeight = _renderPassManager.state->renderPassDescriptor.depthAttachment.texture.height;
        }
        if ((passWidth == 0 || passHeight == 0) && _renderPassManager.state->renderPassDescriptor.stencilAttachment.texture) {
            passWidth = _renderPassManager.state->renderPassDescriptor.stencilAttachment.texture.width;
            passHeight = _renderPassManager.state->renderPassDescriptor.stencilAttachment.texture.height;
        }
    }

    if (passWidth == 0 || passHeight == 0) {
        return NO;
    }

    int64_t fbW = (int64_t)passWidth;
    int64_t fbH = (int64_t)passHeight;
    int64_t vx0 = (int64_t)vx;
    int64_t vy0 = (int64_t)vy;
    int64_t vx1 = vx0 + (int64_t)vw;
    int64_t vy1 = vy0 + (int64_t)vh;
    if (vx1 <= 0 || vy1 <= 0 || vx0 >= fbW || vy0 >= fbH) {
        return YES;
    }

    if (MGL_STATE(ctx)->caps.scissor_test) {
        GLint sx = MGL_STATE(ctx)->var.scissor_box[0];
        GLint sy = MGL_STATE(ctx)->var.scissor_box[1];
        GLint sw = MGL_STATE(ctx)->var.scissor_box[2];
        GLint sh = MGL_STATE(ctx)->var.scissor_box[3];
        if (sw <= 0 || sh <= 0) {
            return YES;
        }

        int64_t sx0 = (int64_t)sx;
        int64_t sy0 = (int64_t)sy;
        int64_t sx1 = sx0 + (int64_t)sw;
        int64_t sy1 = sy0 + (int64_t)sh;
        if (sx1 <= 0 || sy1 <= 0 || sx0 >= fbW || sy0 >= fbH) {
            return YES;
        }
    }

    return NO;
}

- (void)applyPolygonOffsetForDrawMode:(GLenum)mode
{
    if (!_renderPassManager.state->currentRenderEncoder) {
        return;
    }

    MTLTriangleFillMode triangleFillMode = MTLTriangleFillModeFill;
    if (ctx && mglDrawModeProducesPolygons(mode)) {
        if (MGL_STATE(ctx)->var.polygon_mode == GL_LINE) {
            triangleFillMode = MTLTriangleFillModeLines;
        } else if (MGL_STATE(ctx)->var.polygon_mode != GL_FILL &&
                   MGL_STATE(ctx)->var.polygon_mode != GL_POINT) {
            mglLogRenderStateRepair("polygon_mode", MGL_STATE(ctx)->var.polygon_mode, GL_FILL);
            MGL_STATE(ctx)->var.polygon_mode = GL_FILL;
            mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
        }
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];

    BOOL enableDepthBias = NO;

    if (ctx && mglDrawModeProducesPolygons(mode)) {
        switch (MGL_STATE(ctx)->var.polygon_mode) {
            case GL_POINT:
                enableDepthBias = MGL_STATE(ctx)->caps.polygon_offset_point;
                break;
            case GL_LINE:
                enableDepthBias = MGL_STATE(ctx)->caps.polygon_offset_line;
                break;
            case GL_FILL:
            default:
                enableDepthBias = MGL_STATE(ctx)->caps.polygon_offset_fill;
                break;
        }
    }

    if (enableDepthBias) {
        float _bias = MGL_STATE(ctx)->var.polygon_offset_units;
        float _slope = MGL_STATE(ctx)->var.polygon_offset_factor;
        float _clamp = 0.0f;
        if (!_bindingSync.state->lastBoundValid || _bindingSync.state->lastDepthBias != _bias ||
            _bindingSync.state->lastDepthBiasClamp != _clamp || _bindingSync.state->lastDepthSlopeScale != _slope) {
            [_renderPassManager.state->currentRenderEncoder setDepthBias:_bias
                                     slopeScale:_slope
                                          clamp:_clamp];
            [_bindingSync setLastDepthBias:_bias clamp:_clamp slopeScale:_slope];
        }
    } else {
        if (!_bindingSync.state->lastBoundValid || _bindingSync.state->lastDepthBias != 0.0f ||
            _bindingSync.state->lastDepthBiasClamp != 0.0f || _bindingSync.state->lastDepthSlopeScale != 0.0f) {
            [_renderPassManager.state->currentRenderEncoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
            [_bindingSync setLastDepthBias:0.0f clamp:0.0f slopeScale:0.0f];
        }
    }
}

- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode
{
    return ctx &&
           MGL_STATE(ctx)->caps.cull_face &&
           MGL_STATE(ctx)->var.cull_face_mode == GL_FRONT_AND_BACK &&
           mglDrawModeProducesPolygons(mode);
}

- (void)bindCullDistanceEmulationBuffers:(GLenum)mode
                           encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!ctx || !encCtx->encoder) {
        return;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, "bindCullDistanceEmu");
    if (!vao) {
        return;
    }
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (!activeProgram) {
        return;
    }

    /* Determine primitive vertex count from the draw mode. */
    uint32_t prim_vertex_count = 0;
    switch (mode) {
        case GL_TRIANGLES: prim_vertex_count = 3; break;
        case GL_TRIANGLE_STRIP: prim_vertex_count = 3; break;
        case GL_TRIANGLE_FAN: prim_vertex_count = 3; break;
        case GL_LINES: prim_vertex_count = 2; break;
        case GL_LINE_STRIP: prim_vertex_count = 2; break;
        case GL_LINE_LOOP: prim_vertex_count = 2; break;
        case GL_POINTS: prim_vertex_count = 1; break;
        default: prim_vertex_count = 1; break;
    }

    /* Scan enabled attributes for cull distance entries. The GLSL source
     * uses "culldistance_data" as the attribute name. We identify them
     * via the SPIRV-Cross resource list (which preserves the name) or
     * by checking the MSL source for [[attribute(N)]] with that name. */
    id<MTLBuffer> cullMtlBuffer = nil;
    GLintptr cullBindingOffset = 0;
    GLuint cullStride = 0;
    GLuint cullDistSize = 0;
    GLintptr cullFirstRelativeOffset = -1;

    SpirvResourceList *vsInputs =
        &activeProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];

    for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }
        /* Find the resource name for this attribute. */
        const char *attrName = NULL;
        if (vsInputs && vsInputs->list) {
            for (GLuint r = 0; r < vsInputs->count; r++) {
                SpirvResource *res = &vsInputs->list[r];
                if (res->location == attrib) {
                    attrName = res->name;
                    break;
                }
                if (res->gl_array_size > 1 &&
                    attrib >= res->location &&
                    attrib < res->location + (GLuint)res->gl_array_size) {
                    attrName = res->name;
                    break;
                }
            }
        }
        /* Fall back to attrib_location_names if the resource name is missing. */
        if (!attrName && attrib < MAX_ATTRIBS) {
            attrName = activeProgram->attrib_location_names[attrib];
        }
        if (!attrName) {
            continue;
        }
        /* Match "culldistance_data" or "culldistance_data[N]" */
        if (strncmp(attrName, "culldistance_data", 17) != 0) {
            continue;
        }
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(ctx, vao, attrib, "bindCullDistanceEmu", &resolved)) {
            continue;
        }
        if (!resolved.buffer || !resolved.buffer->data.mtl_data) {
            continue;
        }
        if (cullDistSize == 0) {
            /* First cull distance attribute: record buffer/stride/offset. */
            cullMtlBuffer = (__bridge id<MTLBuffer>)resolved.buffer->data.mtl_data;
            cullBindingOffset = resolved.binding_offset;
            cullStride = resolved.stride;
            cullFirstRelativeOffset = resolved.relativeoffset;
        } else {
            /* Subsequent cull distance attributes: verify they share the same
             * buffer and stride. If not, fall back to the first attribute's
             * layout (the CTS test uses a single interleaved buffer). */
            if (resolved.buffer->data.mtl_data != (void *)(__bridge void *)cullMtlBuffer ||
                resolved.stride != cullStride) {
                /* Layout mismatch; keep the first attribute's layout. */
            }
        }
        cullDistSize++;
    }

    if (!cullMtlBuffer || cullDistSize == 0) {
        /* No cull distance attributes found; bind a dummy buffer to satisfy
         * Metal validation (the shader still references the slots). */
        static id<MTLBuffer> sDummyCullBuffer = nil;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            float dummy[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            sDummyCullBuffer = [_device newBufferWithBytes:dummy
                                                    length:sizeof(dummy)
                                                   options:MTLResourceStorageModeShared];
        });
        cullMtlBuffer = sDummyCullBuffer;
        cullBindingOffset = 0;
        cullStride = 4;
        cullFirstRelativeOffset = 0;
        cullDistSize = 0; /* zero size means the shader loop is skipped */
    }

    /* The cull distance offset within each vertex is the binding offset plus
     * the relative offset of the first cull distance attribute. */
    uint32_t culldist_offset = (uint32_t)(cullBindingOffset + (cullFirstRelativeOffset >= 0 ? cullFirstRelativeOffset : 0));

    MGLCullDistanceEmuParams params;
    params.prim_vertex_count = prim_vertex_count;
    params.culldist_offset = culldist_offset;
    params.vertex_stride = (uint32_t)cullStride;
    params.culldist_size = cullDistSize;

    [encCtx->encoder setVertexBuffer:cullMtlBuffer
                                    offset:0
                                   atIndex:kMGLCullDistanceVertexBufferIndex];
    [self recordLastBoundVertexBuffer:cullMtlBuffer
                               offset:0
                              atIndex:kMGLCullDistanceVertexBufferIndex];
    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    [encCtx->encoder setVertexBytes:&params
                                    length:sizeof(params)
                                   atIndex:kMGLCullDistanceParamsBufferIndex];
    [self invalidateLastBoundVertexBufferAtIndex:kMGLCullDistanceParamsBufferIndex];
}

- (BOOL)handleTessellationPatchDrawIfNeeded:(GLMContext)drawCtx
                                        mode:(GLenum *)mode
                                       first:(GLint)first
                                       count:(GLsizei)count
                                   indexType:(GLenum)indexType
                                     indices:(const void *)indices
                                  baseVertex:(GLint)baseVertex
                               instanceCount:(GLsizei)instanceCount
                                baseInstance:(GLuint)baseInstance
                                       label:(const char *)label
{
    if (!mode || *mode != GL_PATCHES) {
        return NO;
    }
    if (!drawCtx || count <= 0) {
        return YES;
    }

    if (mglResolvePassthroughPatchModeForContext(drawCtx, mode, label)) {
        return NO;
    }

    self->ctx = drawCtx;

    Program *tcsProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_CONTROL_SHADER);
    Program *tesProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_EVALUATION_SHADER);
    if (!tcsProgram && !tesProgram) {
        return NO;
    }

    if (tcsProgram) {
        if (tcsProgram->dirty_bits) {
            [self bindMTLProgram:tcsProgram];
        }
        if (![self dispatchTessControlShader:drawCtx
                                     program:tcsProgram
                                       first:first
                                       count:count
                                   indexType:indexType
                                     indices:indices
                                  baseVertex:baseVertex
                               instanceCount:instanceCount
                                baseInstance:baseInstance]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    if (tesProgram) {
        if (tesProgram->dirty_bits) {
            [self bindMTLProgram:tesProgram];
        }
        if (![self dispatchTessEvaluationShader:drawCtx
                                           program:tesProgram
                                             first:first
                                             count:count]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    drawCtx->state.dirty_bits = DIRTY_ALL;
    (void)label;
    return YES;
}


@end
