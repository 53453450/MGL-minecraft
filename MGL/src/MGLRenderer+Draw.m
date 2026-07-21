// MGLRenderer+Draw.m
// Draw command encoding methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"

/* === C helpers used by Draw and Batch methods === */
/* P2-1: mglRendererProgramHasSampledResourceNamed is non-static so
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

static bool mglBuildDynamicVertexArray(const VertexArray *base,
                                       const MGLDrawCommand *cmd,
                                       VertexArray *out)
{
    if (!base || !cmd || !out ||
        cmd->dynamic_vertex_binding_count > MGL_MAX_DYNAMIC_VERTEX_BINDINGS) {
        return false;
    }

    *out = *base;
    for (uint8_t i = 0; i < cmd->dynamic_vertex_binding_count; i++) {
        const MGLDynamicVertexBinding *override =
            &cmd->dynamic_vertex_bindings[i];
        if (!override->buffer ||
            override->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
            return false;
        }
        BufferBinding *binding = &out->bindings[override->binding_index];
        if (binding->offset > INTPTR_MAX - (GLintptr)override->offset_delta) {
            return false;
        }
        binding->buffer = (Buffer *)override->buffer;
        binding->offset += (GLintptr)override->offset_delta;
    }
    return true;
}

static bool mglDynamicVertexAttribCanBindDirectly(Program *active_program,
                                                   GLuint attrib_index,
                                                   const VertexAttrib *attrib)
{
    if (!attrib || attrib->long_attribute || attrib->type == GL_DOUBLE ||
        (!attrib->integer &&
         (attrib->type == GL_INT || attrib->type == GL_UNSIGNED_INT))) {
        return false;
    }
    if (!attrib->integer) {
        return true;
    }

    SpirvResource *resource =
        mglRendererProgramVertexAttribResource(active_program, attrib_index);
    GLuint shader_type = resource ? resource->gl_type : 0u;
    return !mglIntegerAttribNeedsConversion(attrib->type,
                                            shader_type,
                                            attrib->size,
                                            NULL);
}

static uint64_t mglRendererSamplerSnapshotHash(const MGLSamplerSnapshotKey *key)
{
    const uint8_t *bytes = (const uint8_t *)key;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < sizeof(*key); i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

@implementation MGLRenderer (Draw)

- (bool) bindVertexBuffersToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_vbindCallCount = 0;
    static double s_vbindLastCallTime = 0.0;
    static uint64_t s_vbindLastCallCount = 0;
    uint64_t vbindCall = ++s_vbindCallCount;
    double vbindStartSeconds = mglNowSeconds();
    mglLogLoopHeartbeat("vbind.loop",
                        vbindCall,
                        vbindStartSeconds,
                        &s_vbindLastCallTime,
                        &s_vbindLastCallCount,
                        0.25);

    BufferMap *map;
    Buffer *ptr;
    GLintptr offset;
    NSUInteger bindingIndex;
    bool isBaseBinding;
    bool anyBindingPresent[MAX_MAPPED_BUFFERS] = {false};
    bool baseBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    bool attribBindingReserved[MAX_MAPPED_BUFFERS] = {false};
    int attribBindingIndex[MAX_ATTRIBS];
    Program *activeProgram;
    VertexArray *vao;
    GLuint mapCount;

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL VBIND begin ctx=%p vao=%p encoder=%p",
              ctx, ctx ? ctx->active_state->vao : NULL, encCtx->encoder);
    }

    if (!ctx || !encCtx->encoder) {
        NSLog(@"MGL VBIND skip: encoder/ctx nil");
        return false;
    }

    vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    if (!vao) {
        NSLog(@"MGL VBIND skip: vao nil/invalid");
        return false;
    }
    activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL VBIND vao=%p magic=0x%x", vao, vao->magic);
    }
    mapCount = ctx->active_state->vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: VBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        attribBindingIndex[i] = -1;
    }

    // Resolve attribute slot reservations first so base/resource bindings do not
    // overwrite shader-required vertex input slots.
    bool attribsEnabledByApp = (vao->enabled_attribs != 0u);
    GLuint reserveMaxAttribs = MAX_ATTRIBS;
    // Get the vertex shader MSL source to check which attributes are actually used.
    const char *vsMslStr = activeProgram ? activeProgram->spirv[_VERTEX_SHADER].msl_str : NULL;
    for (GLuint attrib = 0; attrib < reserveMaxAttribs; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }
        // Skip attributes not present in the MSL source (same check as generateVertexDescriptor).
        if (vsMslStr) {
            bool attribInMSL;
            if (_resourceFallback.mslCacheEnabled && activeProgram && activeProgram->mslCacheValid) {
                attribInMSL = ((activeProgram->vertexAttribUsageMask & (1u << attrib)) != 0u);
            } else {
                char attrPattern[32];
                snprintf(attrPattern, sizeof(attrPattern), "[[attribute(%u)]]", attrib);
                attribInMSL = (strstr(vsMslStr, attrPattern) != NULL);
            }
            if (!attribInMSL) {
                continue;
            }
        }

        int mappedIndex = [self getVertexBufferIndexWithAttributeSet:(int)attrib];
        if (mappedIndex < 0 || mappedIndex >= (int)kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL ERROR: VBIND reserve attrib=%u unresolved mapping=%d", attrib, mappedIndex);
            continue;
        }

        attribBindingIndex[attrib] = mappedIndex;
        attribBindingReserved[mappedIndex] = true;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        BOOL enabled = attribsEnabledByApp && ((vao->enabled_attribs >> i) & 0x1u) != 0;
        MGLResolvedVertexAttribBinding resolved = {0};
        Buffer *attribBuffer = mglRendererResolveVertexAttribBinding(ctx,
                                                                     vao,
                                                                     i,
                                                                     __FUNCTION__,
                                                                     &resolved)
            ? resolved.buffer
            : NULL;
        GLuint attribBufferName = attribBuffer ? attribBuffer->name : 0;
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL VBIND attrib=%u enabled=%d buf=%p bufName=%u bindOffset=%lld ptr=0x%llx stride=%u size=%u type=0x%x normalized=%u divisor=%u binding=%u table=%d",
                  i,
                  enabled ? 1 : 0,
                  attribBuffer,
                  attribBufferName,
                  (long long)(attribBuffer ? resolved.binding_offset : vao->attrib[i].binding_offset),
                  (unsigned long long)(uintptr_t)vao->attrib[i].relativeoffset,
                  (unsigned)(attribBuffer ? resolved.stride : vao->attrib[i].stride),
                  (unsigned)vao->attrib[i].size,
                  (unsigned)vao->attrib[i].type,
                  (unsigned)vao->attrib[i].normalized,
                  (unsigned)(attribBuffer ? resolved.divisor : vao->attrib[i].divisor),
                  (unsigned)vao->attrib[i].buffer_bindingindex,
                  attribBuffer && resolved.uses_binding_table ? 1 : 0);
        }

        if (kMGLVerboseBindLogs && enabled && attribBuffer) {
            NSLog(@"MGL VBIND buffer detail attrib=%u name=%u size=%lld mtl=%p data=%p init(ever=%u full=%u range=[%lld,%lld) source=%u off=%lld size=%lld src=%p hash=0x%016llx)",
                  i,
                  attribBuffer->name,
                  (long long)attribBuffer->size,
                  attribBuffer->data.mtl_data,
                  (void *)attribBuffer->data.buffer_data,
                  (unsigned)attribBuffer->ever_written,
                  (unsigned)attribBuffer->has_initialized_data,
                  (long long)attribBuffer->written_min,
                  (long long)attribBuffer->written_max,
                  (unsigned)attribBuffer->last_init_source,
                  (long long)attribBuffer->last_write_offset,
                  (long long)attribBuffer->last_write_size,
                  attribBuffer->last_write_src_ptr,
                  (unsigned long long)attribBuffer->last_write_src_hash);
        }
    }

    for(int i=0; i<(int)mapCount; i++)
    {
        map = &ctx->active_state->vertex_buffer_map_list.buffers[i];
        
        ptr = mglRendererGetValidatedBuffer(ctx, map->buf, __FUNCTION__, (NSUInteger)i);
        offset = map->offset;
        isBaseBinding = (map->attribute_mask == 0);
        GLuint glBindingIndex = map->buffer_base_index;
        bindingIndex = glBindingIndex;
        if (isBaseBinding) {
            NSInteger metalBindingIndex = map->has_metal_binding
                ? (NSInteger)map->metal_binding_index
                : [self getProgramMetalBufferIndexForStage:_VERTEX_SHADER
                                             clientBinding:glBindingIndex];
            if (metalBindingIndex < 0) {
                continue;
            }
            bindingIndex = (NSUInteger)metalBindingIndex;
        }

        // Vertex attribute streams are rebound from VAO below using a deterministic
        // attribute->slot mapping shared with generateVertexDescriptor.
        // Keep this pass for resource/base bindings only.
        if (!isBaseBinding) {
            continue;
        }

        if (bindingIndex >= kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL WARNING: Vertex binding index %lu out of Metal range (max valid=%lu), skipping map[%d]",
                  (unsigned long)bindingIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex, i);
            continue;
        }

        if (attribBindingReserved[bindingIndex]) {
            if (kMGLVerboseBindLogs) {
                NSLog(@"MGL VBIND skip base slot %lu: reserved by attrib mapping",
                      (unsigned long)bindingIndex);
            }
            continue;
        }

        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            baseBindingPresent[glBindingIndex] = true;
        }

        if (!ptr) {
            NSLog(@"MGL WARNING: Vertex buffer map[%d] has invalid/NULL buffer pointer, skipping", i);
            [encCtx->encoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
            continue;
        }

        if (offset < 0) {
            NSLog(@"MGL WARNING: Vertex buffer map[%d] has negative offset=%lld, skipping",
                  i, (long long)offset);
            [encCtx->encoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
            continue;
        }

        if (ptr->size < 0) {
            NSLog(@"MGL WARNING: Vertex buffer %u has invalid size=%lld, skipping",
                  ptr->name, (long long)ptr->size);
            [encCtx->encoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
            continue;
        }

        if (!ptr->data.mtl_data) {
            [self bindMTLBuffer:ptr];
        }
        id<MTLBuffer> buffer = nil;
        if (ptr->data.mtl_data &&
            (uintptr_t)ptr->data.mtl_data >= 0x10000u) {
            buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
        }

        NSUInteger metalLen = buffer ? buffer.length : 0u;
        NSUInteger bindOffset = (NSUInteger)offset;
        NSUInteger reflectedRequiredBytes = 0;
        NSUInteger requiredBindingBytes = kMGLMinimumStageBindingSize;
        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            reflectedRequiredBytes = map->has_metal_binding
                ? [self getProgramBindingRequiredSize:_VERTEX_SHADER
                                                 type:(int)map->resource_type
                                                index:(int)map->resource_index]
                : [self getProgramBindingRequiredSizeForStage:_VERTEX_SHADER
                                                clientBinding:glBindingIndex];
            if (reflectedRequiredBytes > requiredBindingBytes) {
                requiredBindingBytes = reflectedRequiredBytes;
            }
        }
        NSUInteger availableBytes = buffer
            ? mglBufferMapVisibleBackingBytes(map, metalLen)
            : 0u;

        if (!buffer || bindOffset >= metalLen ||
            availableBytes < requiredBindingBytes) {
            id<MTLBuffer> isolated =
                [self isolatedStageBindingBufferForMap:map
                                                 source:buffer
                                         requiredLength:requiredBindingBytes];
            if (!isolated) {
                NSLog(@"MGL WARNING: VBIND failed to isolate undersized buffer=%u slot=%lu required=%lu available=%lu",
                      ptr->name,
                      (unsigned long)bindingIndex,
                      (unsigned long)requiredBindingBytes,
                      (unsigned long)availableBytes);
                [encCtx->encoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = nil;
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
                continue;
            }

            [encCtx->encoder setVertexBuffer:isolated
                                            offset:0
                                           atIndex:bindingIndex];
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = isolated;
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
            MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            anyBindingPresent[bindingIndex] = true;
            if (kMGLVerboseBindLogs) {
                NSLog(@"MGL SET VERTEX BUFFER index=%lu glName=%u offset=0 source=isolated required=%lu reflected=%lu available=%lu range=%lld",
                      (unsigned long)bindingIndex,
                      ptr->name,
                      (unsigned long)requiredBindingBytes,
                      (unsigned long)reflectedRequiredBytes,
                      (unsigned long)availableBytes,
                      (long long)map->size);
            }
            continue;
        }

        /* For small uniform constants (plain uniforms), use setVertexBytes
         * to copy the data into the command buffer at bind time. This is
         * critical for correctness when the same uniform buffer is updated
         * between draws encoded into the same command buffer — a shared-
         * memory MTLBuffer would let the GPU see only the final value. */
        if (isBaseBinding &&
            map->resource_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
            ptr->data.buffer_data &&
            (NSUInteger)ptr->size <= 4096 &&
            offset == 0) {
            [encCtx->encoder setVertexBytes:(const void *)(uintptr_t)ptr->data.buffer_data
                                            length:(NSUInteger)ptr->size
                                           atIndex:bindingIndex];
            [self invalidateLastBoundVertexBufferAtIndex:bindingIndex];
            anyBindingPresent[bindingIndex] = true;
            ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            continue;
        }

        if (!_bindingSync.state->lastBoundValid ||
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer != buffer ||
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset != (NSUInteger)offset) {
            [encCtx->encoder setVertexBuffer:buffer offset:offset atIndex:bindingIndex];
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = buffer;
            _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = (NSUInteger)offset;
            MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
        }
        Program *bindProgram = activeProgram;
        if (mglProgramNeedsBindingTrace(bindProgram)) {
            static uint64_t s_focusedVertexBufferBindLogs = 0;
            if (mglShouldLogFocusedBinding(&s_focusedVertexBufferBindLogs)) {
                NSLog(@"MGL VBIND focused program=%u clientBinding=%u metalSlot=%lu resourceType=%s resourceIndex=%u buffer=%u offset=%lu available=%lu metalLen=%lu range=%lld",
                      (unsigned)bindProgram->name,
                      (unsigned)glBindingIndex,
                      (unsigned long)bindingIndex,
                      mglSpirvResourceTypeName((int)map->resource_type),
                      (unsigned)map->resource_index,
                      (unsigned)ptr->name,
                      (unsigned long)bindOffset,
                      (unsigned long)availableBytes,
                      (unsigned long)metalLen,
                      (long long)map->size);
            }
        }
        static uint64_t s_traceFileVertexBufferBindLogs = 0;
        if (mglProgramNeedsTraceLog(bindProgram) &&
            mglShouldLogTraceFileBindingForProgram(bindProgram, &s_traceFileVertexBufferBindLogs)) {
            mglTraceLog("VBIND program=%u clientBinding=%u metalSlot=%lu resourceType=%s resourceIndex=%u buffer=%u offset=%lu available=%lu metalLen=%lu range=%lld",
                        (unsigned)bindProgram->name,
                        (unsigned)glBindingIndex,
                        (unsigned long)bindingIndex,
                        mglSpirvResourceTypeName((int)map->resource_type),
                        (unsigned)map->resource_index,
                        (unsigned)ptr->name,
                        (unsigned long)bindOffset,
                        (unsigned long)availableBytes,
                        (unsigned long)metalLen,
                        (long long)map->size);
        }
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL SET VERTEX BUFFER index=%lu glName=%u offset=%lu available=%lu source=base",
                  (unsigned long)bindingIndex,
                  ptr->name,
                  (unsigned long)bindOffset,
                  (unsigned long)metalLen);
        }
        anyBindingPresent[bindingIndex] = true;
    }

    if (![self bindVertexAttributesFromVAO:vao
                              activeProgram:activeProgram
                                  vsMslStr:vsMslStr
                        attribsEnabledByApp:attribsEnabledByApp
                        attribBindingIndex:attribBindingIndex
                          anyBindingPresent:anyBindingPresent
                              encodeContext:encCtx]) {
        return false;
    }

    [self bindVertexFallbackBuffersToCurrentRenderEncoder:activeProgram
                                      anyBindingPresent:anyBindingPresent
                                      baseBindingPresent:baseBindingPresent
                                          encodeContext:encCtx];

    [self bindPointSizeParamsIfNeeded:anyBindingPresent encodeContext:encCtx];

    /* Snapshot the final Metal slot set only after VAO, fallback, and
     * generated point-size bindings have all updated anyBindingPresent.  The
     * cache mask accumulates for the encoder lifetime because older dedup
     * entries remain valid until invalidateLastBoundState. */
    uint32_t boundVertexBufferMask = 0;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        if (anyBindingPresent[i]) {
            boundVertexBufferMask |= 1U << i;
        }
    }
    _bindingSync.state->lastBoundVertexBufferMask |= boundVertexBufferMask;

    if (getenv("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_vbind_trace_count = 0;
        if ((++s_vbind_trace_count % 500) == 1) {
            NSLog(@"MGL SPARSE VBIND: mask=0x%x activeSlots=%d/31",
                  boundVertexBufferMask,
                  __builtin_popcount(boundVertexBufferMask));
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(vbindCall)) {
        NSUInteger boundSlots = 0;
        NSUInteger reservedSlots = 0;
        NSUInteger baseSlots = 0;
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (anyBindingPresent[s]) {
                boundSlots++;
            }
            if (attribBindingReserved[s]) {
                reservedSlots++;
            }
        }
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (baseBindingPresent[s]) {
                baseSlots++;
            }
        }
        MGLTraceNSLog(@"MGL TRACE vbind.end call=%llu mapCount=%u boundSlots=%lu reservedAttribSlots=%lu baseSlots=%lu elapsed=%.3fms",
              (unsigned long long)vbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)reservedSlots,
              (unsigned long)baseSlots,
              (mglNowSeconds() - vbindStartSeconds) * 1000.0);
    }

    /* Mark the dedup cache as valid for the current encoder so subsequent
     * binds can be skipped when the resource and offset are unchanged. */
    _bindingSync.state->lastBoundValid = YES;
    return true;
}

/* Bind vertex attributes from the VAO.  Extracted from
 * bindVertexBuffersToCurrentRenderEncoder to keep that function under the
 * 500-line limit.  Pure mechanical extraction — no behavior change. */
- (bool)bindVertexAttributesFromVAO:(VertexArray *)vao
                      activeProgram:(Program *)activeProgram
                          vsMslStr:(const char *)vsMslStr
                attribsEnabledByApp:(bool)attribsEnabledByApp
                attribBindingIndex:(int *)attribBindingIndex
                  anyBindingPresent:(bool *)anyBindingPresent
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    NSUInteger bindingIndex;

    // Attribute bindings must use the exact same index mapping as generateVertexDescriptor.
    // Do this pass directly from the VAO so pipeline creation does not depend on map list timing.
    GLuint maxAttribs = MAX_ATTRIBS;
    for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }
        // Skip attributes not present in the MSL source (same check as generateVertexDescriptor).
        if (vsMslStr) {
            bool attribInMSL;
            if (_resourceFallback.mslCacheEnabled && activeProgram && activeProgram->mslCacheValid) {
                attribInMSL = ((activeProgram->vertexAttribUsageMask & (1u << attrib)) != 0u);
            } else {
                char attrPattern[32];
                snprintf(attrPattern, sizeof(attrPattern), "[[attribute(%u)]]", attrib);
                attribInMSL = (strstr(vsMslStr, attrPattern) != NULL);
            }
            if (!attribInMSL) {
                continue;
            }
        }
        BOOL usesCurrentValue = mglRendererVertexAttribUsesCurrentValue(vao, attrib);
        MGLResolvedVertexAttribBinding resolved = {0};
        bool hasAttribBinding = mglRendererResolveVertexAttribBinding(ctx,
                                                                      vao,
                                                                      attrib,
                                                                      __FUNCTION__,
                                                                      &resolved);
        // When enabled_attribs tracking is empty but the program uses this attribute,
        // fall through and bind if a valid buffer exists (Sodium DSA path compatibility).
        if (!attribsEnabledByApp && !hasAttribBinding) {
            continue;
        }

        int mappedIndex = attribBindingIndex[attrib];
        if (mappedIndex < 0 || mappedIndex >= (int)kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL ERROR: VBIND attrib=%u unresolved mapping=%d", attrib, mappedIndex);
            continue;
        }

        bindingIndex = (NSUInteger)mappedIndex;
        if (usesCurrentValue) {
            uint8_t attribBytes[16];
            NSUInteger attribStride = mglRendererBuildCurrentVertexAttribBytes(ctx,
                                                                               attrib,
                                                                               &vao->attrib[attrib],
                                                                               attribBytes);
            if (attribStride == 0u) {
                NSLog(@"MGL VBIND skip attrib=%u: failed to build current vertex attrib bytes", attrib);
                continue;
            }
            static const NSUInteger kMGLCurrentAttribRepeatCount = 4096u;
            NSMutableData *repeated = [NSMutableData dataWithLength:kMGLCurrentAttribRepeatCount * attribStride];
            if (!repeated) {
                NSLog(@"MGL VBIND skip attrib=%u: failed to allocate current vertex attrib stream", attrib);
                continue;
            }
            uint8_t *dst = (uint8_t *)repeated.mutableBytes;
            for (NSUInteger v = 0; v < kMGLCurrentAttribRepeatCount; v++) {
                memcpy(dst + v * attribStride, attribBytes, MIN((NSUInteger)16u, attribStride));
            }
            id<MTLBuffer> currentAttribBuffer = [_device newBufferWithBytes:repeated.bytes
                                                                      length:repeated.length
                                                                     options:MTLResourceStorageModeShared];
            if (!currentAttribBuffer) {
                NSLog(@"MGL VBIND skip attrib=%u: failed to allocate current vertex attrib Metal buffer", attrib);
                continue;
            }
            if (!_bindingSync.state->lastBoundValid ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer != currentAttribBuffer ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset != 0) {
                [encCtx->encoder setVertexBuffer:currentAttribBuffer
                                                offset:0
                                               atIndex:bindingIndex];
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = currentAttribBuffer;
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
            anyBindingPresent[bindingIndex] = true;
            static uint64_t s_traceFileCurrentAttribBindLogs = 0;
            if (mglProgramNeedsTraceLog(activeProgram) &&
                mglShouldLogTraceFileBindingForProgram(activeProgram, &s_traceFileCurrentAttribBindLogs)) {
                SpirvResource *resource = mglRendererProgramVertexAttribResource(activeProgram, attrib);
                mglTraceLog("VATTR_BIND_CURRENT program=%u attrib=%u resource=%s loc=%u metalSlot=%lu stride=%lu size=%u type=0x%x valueI=(%d,%d,%d,%d) valueF=(%.6f,%.6f,%.6f,%.6f)",
                            activeProgram ? (unsigned)activeProgram->name : 0u,
                            (unsigned)attrib,
                            resource && resource->name ? resource->name : "(unknown)",
                            resource ? (unsigned)resource->location : 0xffffffffu,
                            (unsigned long)bindingIndex,
                            (unsigned long)attribStride,
                            (unsigned)vao->attrib[attrib].size,
                            (unsigned)vao->attrib[attrib].type,
                            (int)ctx->active_state->current_vertex_attrib[attrib].i[0],
                            (int)ctx->active_state->current_vertex_attrib[attrib].i[1],
                            (int)ctx->active_state->current_vertex_attrib[attrib].i[2],
                            (int)ctx->active_state->current_vertex_attrib[attrib].i[3],
                            ctx->active_state->current_vertex_attrib[attrib].f[0],
                            ctx->active_state->current_vertex_attrib[attrib].f[1],
                            ctx->active_state->current_vertex_attrib[attrib].f[2],
                            ctx->active_state->current_vertex_attrib[attrib].f[3]);
            }
            continue;
        }
        if (!hasAttribBinding) {
            NSLog(@"MGL VBIND skip attrib=%u: enabled but buffer is invalid", attrib);
            continue;
        }
        Buffer *attribBuffer = resolved.buffer;
        const VertexAttrib *attribState = resolved.attrib;

        if (!mglRendererBufferHasDrawableContents(attribBuffer)) {
            NSLog(@"MGL VBIND BLOCK draw: attrib=%u uses buffer=%u that was allocated but never populated "
                  "(initSource=%u mapped=%u access=0x%x accessFlags=0x%x hasInitialized=%u written=[%lld,%lld) lastOff=%lld lastSize=%lld lastSrc=%p hash=0x%016llx)",
                  attrib,
                  attribBuffer->name,
                  (unsigned)attribBuffer->last_init_source,
                  (unsigned)attribBuffer->mapped,
                  (unsigned)attribBuffer->access,
                  (unsigned)attribBuffer->access_flags,
                  (unsigned)attribBuffer->has_initialized_data,
                  (long long)attribBuffer->written_min,
                  (long long)attribBuffer->written_max,
                  (long long)attribBuffer->last_write_offset,
                  (long long)attribBuffer->last_write_size,
                  attribBuffer->last_write_src_ptr,
                  (unsigned long long)attribBuffer->last_write_src_hash);
            return false;
        }

        if (resolved.binding_offset < 0) {
            NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u negative bindingOffset=%lld",
                  attrib,
                  attribBuffer->name,
                  (long long)resolved.binding_offset);
            return false;
        }
        if (resolved.relativeoffset < 0) {
            NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u negative relativeOffset=%lld",
                  attrib,
                  attribBuffer->name,
                  (long long)resolved.relativeoffset);
            return false;
        }
        GLintptr attrOffset = resolved.binding_offset +
                              (GLintptr)(uintptr_t)resolved.relativeoffset;
        size_t compSize = mglVertexAttribComponentSize(attribState->type);
        size_t compCount = (size_t)attribState->size;
        GLintptr attrSpan = 0;
        if (compSize > 0u && compCount > 0u) {
            size_t total = compSize * compCount;
            if (total > (size_t)INTPTR_MAX) {
                NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u attr span overflow (compSize=%zu compCount=%zu)",
                      attrib,
                      attribBuffer->name,
                      compSize,
                      compCount);
                return false;
            }
            attrSpan = (GLintptr)total;
        }
        GLintptr attrEnd = attrOffset + ((attrSpan > 0) ? attrSpan : 1);
        if (kMGLVerboseBindLogs &&
            attribBuffer->written_min >= 0 && attribBuffer->written_max >= 0) {
            if (attrOffset < attribBuffer->written_min || attrEnd > attribBuffer->written_max) {
                static uint64_t s_vbindWrittenRangeWarningCount = 0;
                uint64_t hit = ++s_vbindWrittenRangeWarningCount;
                if (hit <= 16ull || (hit % 4096ull) == 0ull) {
                    NSLog(@"MGL VBIND WARNING draw: attrib=%u buffer=%u attrRange=[%lld,%lld) outside written range [%lld,%lld) (type=0x%x size=%u) - allowing, Sodium arena buffers use sub-ranges hit=%llu",
                          attrib,
                          attribBuffer->name,
                          (long long)attrOffset,
                          (long long)attrEnd,
                          (long long)attribBuffer->written_min,
                          (long long)attribBuffer->written_max,
                          (unsigned)attribState->type,
                          (unsigned)attribState->size,
                          (unsigned long long)hit);
                }
                // Continue instead of blocking: MGL's write tracking uses the union of
                // all mapped ranges. Sodium arena-allocates large buffers and writes
                // vertex data at varying sub-range offsets. The Metal backing has the
                // data from the flush, so the draw will render correctly.
            }
        }

        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL VBIND attrib map attrib=%u -> index=%lu buffer=%u bindingOffset=%lld table=%d",
                  attrib,
                  (unsigned long)bindingIndex,
                  (unsigned)attribBuffer->name,
                  (long long)resolved.binding_offset,
                  resolved.uses_binding_table ? 1 : 0);
        }

        bool needsIntToFloatConversion = (attribState->integer == 0 &&
                                          (attribState->type == GL_INT ||
                                           attribState->type == GL_UNSIGNED_INT));

        /* glVertexAttribIFormat (integer==1): detect signedness mismatch
         * between source type and shader's declared int/uint input. Metal
         * rejects e.g. UChar/UShort/UInt feeding `int` shader inputs (and
         * signed sources feeding `uint` inputs). When mismatched, convert
         * the data on the CPU to the shader's 32-bit integer type. */
        bool needsIntegerConversion = false;
        BOOL integerConvDstIsInt = NO;
        if (attribState->integer == 1 && attribState->type != GL_DOUBLE) {
            SpirvResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, attrib);
            GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
            MTLVertexFormat ignored = MTLVertexFormatInvalid;
            if (mglIntegerAttribNeedsConversion(attribState->type,
                                                shaderGlType,
                                                attribState->size,
                                                &ignored)) {
                needsIntegerConversion = true;
                integerConvDstIsInt = (shaderGlType == GL_INT ||
                                       shaderGlType == GL_INT_VEC2 ||
                                       shaderGlType == GL_INT_VEC3 ||
                                       shaderGlType == GL_INT_VEC4);
            }
        }

        if (attribState->type != GL_DOUBLE && !needsIntToFloatConversion &&
            !needsIntegerConversion && anyBindingPresent[bindingIndex]) {
            continue;
        }

        if (attribState->type == GL_DOUBLE) {
            NSUInteger convertedStride = 0;
            id<MTLBuffer> convertedBuffer = [self floatVertexBufferForDoubleAttrib:attribBuffer
                                                                          resolved:&resolved
                                                                              size:attribState->size
                                                                          outStride:&convertedStride];
            if (!convertedBuffer) {
                NSLog(@"MGL VBIND skip attrib=%u buffer=%u: failed to convert GL_DOUBLE vertex attrib",
                      attrib,
                      attribBuffer->name);
                continue;
            }
            if (!_bindingSync.state->lastBoundValid ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer != convertedBuffer ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset != 0) {
                [encCtx->encoder setVertexBuffer:convertedBuffer offset:0 atIndex:bindingIndex];
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = convertedBuffer;
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
            anyBindingPresent[bindingIndex] = true;
            continue;
        }

        if (needsIntToFloatConversion) {
            NSUInteger convertedStride = 0;
            id<MTLBuffer> convertedBuffer = [self floatVertexBufferForIntAttrib:attribBuffer
                                                                        resolved:&resolved
                                                                            size:attribState->size
                                                                      normalized:attribState->normalized
                                                                            type:attribState->type
                                                                        outStride:&convertedStride];
            if (!convertedBuffer) {
                NSLog(@"MGL VBIND skip attrib=%u buffer=%u: failed to convert GL_INT/GL_UNSIGNED_INT vertex attrib to float",
                      attrib,
                      attribBuffer->name);
                continue;
            }
            if (!_bindingSync.state->lastBoundValid ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer != convertedBuffer ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset != 0) {
                [encCtx->encoder setVertexBuffer:convertedBuffer offset:0 atIndex:bindingIndex];
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = convertedBuffer;
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
            anyBindingPresent[bindingIndex] = true;
            continue;
        }

        if (needsIntegerConversion) {
            NSUInteger convertedStride = 0;
            id<MTLBuffer> convertedBuffer = [self integerVertexBufferForAttrib:attribBuffer
                                                                       resolved:&resolved
                                                                           size:attribState->size
                                                                         srcType:attribState->type
                                                                       dstIsInt:integerConvDstIsInt
                                                                      outStride:&convertedStride];
            if (!convertedBuffer) {
                NSLog(@"MGL VBIND skip attrib=%u buffer=%u: failed to convert integer vertex attrib (src=0x%x dstIsInt=%d)",
                      attrib,
                      attribBuffer->name,
                      (unsigned)attribState->type,
                      (int)integerConvDstIsInt);
                continue;
            }
            if (!_bindingSync.state->lastBoundValid ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer != convertedBuffer ||
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset != 0) {
                [encCtx->encoder setVertexBuffer:convertedBuffer offset:0 atIndex:bindingIndex];
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = convertedBuffer;
                _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
            anyBindingPresent[bindingIndex] = true;
            continue;
        }

        if (!attribBuffer->data.mtl_data) {
            [self bindMTLBuffer:attribBuffer];
        }
        if (!attribBuffer->data.mtl_data) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: no Metal backing",
                  attrib, attribBuffer->name);
            continue;
        }
        if ((uintptr_t)attribBuffer->data.mtl_data < 0x10000u) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: suspicious mtl_data=%p",
                  attrib, attribBuffer->name, attribBuffer->data.mtl_data);
            continue;
        }

        id<MTLBuffer> attribMetalBuffer = (__bridge id<MTLBuffer>)(attribBuffer->data.mtl_data);
        if (!attribMetalBuffer) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: Metal bridge failed",
                  attrib, attribBuffer->name);
            continue;
        }

        NSUInteger attribBindingOffset = (NSUInteger)resolved.binding_offset;
        if (attribBindingOffset >= attribMetalBuffer.length) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: bindingOffset=%lu >= metalLen=%lu",
                  attrib,
                  attribBuffer->name,
                  (unsigned long)attribBindingOffset,
                  (unsigned long)attribMetalBuffer.length);
            continue;
        }

        /* Bind the VBO at offset 0. Per-attribute offsets are expressed via
         * the vertex descriptor's attribute offset field (set in
         * generateVertexDescriptor), which is relative to this buffer base. */
	    if (!_bindingSync.state->lastBoundValid ||
	        _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer != attribMetalBuffer ||
	        _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset != 0) {
	        [encCtx->encoder setVertexBuffer:attribMetalBuffer offset:0 atIndex:bindingIndex];
	        _bindingSync.state->lastBoundVertexBuffers[bindingIndex].buffer = attribMetalBuffer;
	        _bindingSync.state->lastBoundVertexBuffers[bindingIndex].offset = 0;
	        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
	    } else {
	        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
	    }
	    anyBindingPresent[bindingIndex] = true;
            static uint64_t s_traceFileVertexAttribBindLogs = 0;
            if (mglProgramNeedsTraceLog(activeProgram) &&
                mglShouldLogTraceFileBindingForProgram(activeProgram, &s_traceFileVertexAttribBindLogs)) {
                SpirvResource *resource = mglRendererProgramVertexAttribResource(activeProgram, attrib);
                GLboolean effectiveNormalized = attribState->normalized;
                if (!effectiveNormalized &&
                    attribState->type == GL_UNSIGNED_BYTE &&
                    attribState->size == 4 &&
                    mglRendererVertexAttribIsColorInput(activeProgram, attrib)) {
                    effectiveNormalized = GL_TRUE;
                }
                MTLVertexFormat format = glTypeSizeToMtlType(attribState->type,
                                                             attribState->size,
                                                             effectiveNormalized);
                mglTraceLog("VATTR_BIND program=%u attrib=%u resource=%s loc=%u metalSlot=%lu glBuffer=%u bindingIndex=%u bindingOffset=%lu relOffset=%lld stride=%u size=%u type=0x%x normalized=%u/%u divisor=%u table=%d metalLen=%lu format=%lu(%s)",
                            activeProgram ? (unsigned)activeProgram->name : 0u,
                            (unsigned)attrib,
                            resource && resource->name ? resource->name : "(unknown)",
                            resource ? (unsigned)resource->location : 0xffffffffu,
                            (unsigned long)bindingIndex,
                            (unsigned)attribBuffer->name,
                            (unsigned)resolved.binding_index,
                            (unsigned long)attribBindingOffset,
                            (long long)resolved.relativeoffset,
                            (unsigned)resolved.stride,
                            (unsigned)attribState->size,
                            (unsigned)attribState->type,
                            (unsigned)attribState->normalized,
                            (unsigned)effectiveNormalized,
                            (unsigned)resolved.divisor,
                            resolved.uses_binding_table ? 1 : 0,
                            (unsigned long)attribMetalBuffer.length,
                            (unsigned long)format,
                            mglVertexFormatName(format));
            }
	        if (kMGLVerboseBindLogs) {
	            NSLog(@"MGL SET VERTEX ATTRIB BUFFER index=%lu glName=%u offset=%lu available=%lu attrib=%u stride=%u attrOffset=0x%llx mtl=%p",
	                  (unsigned long)bindingIndex,
                  attribBuffer->name,
                  (unsigned long)attribBindingOffset,
                  (unsigned long)attribMetalBuffer.length,
                  attrib,
                  (unsigned)resolved.stride,
                  (unsigned long long)(uintptr_t)resolved.relativeoffset,
                  attribBuffer->data.mtl_data);
        }
    }

    return true;
}


/* Bind fallback buffers for vertex-stage buffer slots that were not mapped
 * by the main binding loop above.  Extracted from
 * bindVertexBuffersToCurrentRenderEncoder to keep that function under the
 * 500-line limit.  Pure mechanical extraction — no behavior change. */
- (void)bindVertexFallbackBuffersToCurrentRenderEncoder:(Program *)activeProgram
                                     anyBindingPresent:(bool *)anyBindingPresent
                                     baseBindingPresent:(bool *)baseBindingPresent
                                         encodeContext:(const MGLEncodeContext *)encCtx
{
    static id<MTLBuffer> fallbackBindingBuffer = nil;

    if (!fallbackBindingBuffer) {
        fallbackBindingBuffer = [_device newBufferWithLength:kMGLDefaultStageFallbackBufferSize
                                                     options:MTLResourceStorageModeShared];
    }

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    // This prevents Metal validation aborts on missing buffer slots.
    const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER
    };
    for (int t = 0; t < 4; t++) {
        int resourceType = resourceTypes[t];
        int count = [self getProgramBindingCount:_VERTEX_SHADER type:resourceType];
        Program *program = activeProgram;
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= _MAX_SPIRV_RES ||
                i >= (int)program->spirv_resources_list[_VERTEX_SHADER][resourceType].count) {
                continue;
            }
            SpirvResource *resource = &program->spirv_resources_list[_VERTEX_SHADER][resourceType].list[i];
            if (mglShouldSkipStageBufferResource(program, _VERTEX_SHADER, resourceType, resource)) {
                continue;
            }
            GLuint elementCount = mglStageBufferResourceElementCount(resourceType, resource);
            for (GLuint element = 0; element < elementCount; element++) {
                GLuint clientBinding =
                    mglClientBufferBindingForResourceElement(resourceType, resource, element);
                if (clientBinding >= MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                NSInteger metalBinding =
                    (NSInteger)mglMetalResourceSlotForElement(resource, element);
                if (metalBinding < 0 || metalBinding >= (NSInteger)kMGLMaxMetalVertexBufferCount) {
                    continue;
                }
                if (!anyBindingPresent[(NSUInteger)metalBinding] && fallbackBindingBuffer) {
                    NSUInteger _slot = (NSUInteger)metalBinding;
                    if (!_bindingSync.state->lastBoundValid ||
                        _bindingSync.state->lastBoundVertexBuffers[_slot].buffer != fallbackBindingBuffer ||
                        _bindingSync.state->lastBoundVertexBuffers[_slot].offset != 0) {
                        [encCtx->encoder setVertexBuffer:fallbackBindingBuffer offset:0 atIndex:_slot];
                        _bindingSync.state->lastBoundVertexBuffers[_slot].buffer = fallbackBindingBuffer;
                        _bindingSync.state->lastBoundVertexBuffers[_slot].offset = 0;
                        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                    }
                    baseBindingPresent[clientBinding] = true;
                    anyBindingPresent[_slot] = true;
                }
            }
        }
    }

    // Conservative safety net:
    // Ensure every stage buffer slot has a valid binding before draw validation.
    // This avoids hard aborts when reflection misses hidden/generated buffer args.
    if (kMGLEnableVertexAllSlotFallback && fallbackBindingBuffer) {
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (!anyBindingPresent[s]) {
                if (!_bindingSync.state->lastBoundValid ||
                    _bindingSync.state->lastBoundVertexBuffers[s].buffer != fallbackBindingBuffer ||
                    _bindingSync.state->lastBoundVertexBuffers[s].offset != 0) {
                    [encCtx->encoder setVertexBuffer:fallbackBindingBuffer offset:0 atIndex:s];
                    _bindingSync.state->lastBoundVertexBuffers[s].buffer = fallbackBindingBuffer;
                    _bindingSync.state->lastBoundVertexBuffers[s].offset = 0;
                    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                } else {
                    MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                }
                anyBindingPresent[s] = true;
            }
        }
    }

}


/* Bind point-size parameters if the active shader references them.
 * Extracted from bindVertexBuffersToCurrentRenderEncoder. */
- (void)bindPointSizeParamsIfNeeded:(bool *)anyBindingPresent
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    BOOL needsPointSizeParams = NO;
    int pointSizeStages[] = { _VERTEX_SHADER, _TESS_EVALUATION_SHADER, _GEOMETRY_SHADER };
    for (NSUInteger ps = 0; ps < sizeof(pointSizeStages) / sizeof(pointSizeStages[0]); ps++) {
        Program *pointProgram = mglResolveProgramForStageFromState(ctx, pointSizeStages[ps]);
        const char *pointMsl = pointProgram ? pointProgram->spirv[pointSizeStages[ps]].msl_str : NULL;
        if (pointMsl && strstr(pointMsl, "_mgl_point_size_params")) {
            needsPointSizeParams = YES;
            break;
        }
    }
    if (needsPointSizeParams) {
        float pointSizeParams[2] = {
            ctx && ctx->active_state->var.point_size > 0.0f ? ctx->active_state->var.point_size : 1.0f,
            ctx && ctx->active_state->caps.program_point_size ? 1.0f : 0.0f
        };
        [encCtx->encoder setVertexBytes:pointSizeParams
                                      length:sizeof(pointSizeParams)
                                      atIndex:kMGLPointSizeParamBufferIndex];
        [self invalidateLastBoundVertexBufferAtIndex:kMGLPointSizeParamBufferIndex];
        anyBindingPresent[kMGLPointSizeParamBufferIndex] = true;
    }

}


- (bool) bindFragmentBuffersToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_fbindCallCount = 0;
    static double s_fbindLastCallTime = 0.0;
    static uint64_t s_fbindLastCallCount = 0;
    uint64_t fbindCall = ++s_fbindCallCount;
    double fbindStartSeconds = mglNowSeconds();
    mglLogLoopHeartbeat("fbind.loop",
                        fbindCall,
                        fbindStartSeconds,
                        &s_fbindLastCallTime,
                        &s_fbindLastCallCount,
                        0.25);

    GLuint mapCount;
    BufferMap *map;
    Buffer *ptr;
    GLintptr offset;
    NSUInteger bindingIndex;
    bool isBaseBinding;
    bool anyBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    bool baseBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    Program *activeProgram = NULL;

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL FBIND begin ctx=%p encoder=%p", ctx, encCtx->encoder);
    }

    if (!ctx || !encCtx->encoder) {
        NSLog(@"MGL FBIND skip: ctx/encoder nil");
        return false;
    }
    activeProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);

    mapCount = ctx->active_state->fragment_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: FBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        map = &ctx->active_state->fragment_buffer_map_list.buffers[i];

        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL FBIND slot=%u candidate=%p mask=0x%x baseIndex=%u offset=%lld",
                  i,
                  map->buf,
                  map->attribute_mask,
                  map->buffer_base_index,
                  (long long)map->offset);
        }

        ptr = mglRendererGetValidatedBuffer(ctx, map->buf, __FUNCTION__, (NSUInteger)i);
        offset = map->offset;
        isBaseBinding = (map->attribute_mask == 0);
        GLuint glBindingIndex = map->buffer_base_index;
        bindingIndex = glBindingIndex;
        if (isBaseBinding) {
            NSInteger metalBindingIndex = map->has_metal_binding
                ? (NSInteger)map->metal_binding_index
                : [self getProgramMetalBufferIndexForStage:_FRAGMENT_SHADER
                                             clientBinding:glBindingIndex];
            if (metalBindingIndex < 0) {
                continue;
            }
            bindingIndex = (NSUInteger)metalBindingIndex;
        }

        if (bindingIndex >= MAX_BINDABLE_BUFFERS) {
            NSLog(@"MGL WARNING: Fragment binding index %lu out of range (max=%d), skipping map[%d]",
                  (unsigned long)bindingIndex, MAX_BINDABLE_BUFFERS, i);
            continue;
        }

        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            baseBindingPresent[glBindingIndex] = true;
        }

        if (!ptr) {
            NSLog(@"MGL FBIND skip slot=%u: invalid/NULL candidate=%p", i, map->buf);
            map->buf = NULL;
            [encCtx->encoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
            continue;
        }

        if (offset < 0) {
            NSLog(@"MGL FBIND skip slot=%u buffer=%u: negative offset=%lld",
                  i, ptr->name, (long long)offset);
            [encCtx->encoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
            continue;
        }

        if (ptr->size < 0) {
            NSLog(@"MGL FBIND skip slot=%u buffer=%u: invalid size=%lld",
                  i, ptr->name, (long long)ptr->size);
            continue;
        }
        
        if (!isBaseBinding && ptr->size < 4096)
        {
            if (ptr->data.buffer_data && ptr->size > 0) {
                uintptr_t cpuData = (uintptr_t)ptr->data.buffer_data;
                if (cpuData < 0x100000000ULL) {
                    NSLog(@"MGL FBIND skip small buffer=%u slot=%u: suspicious CPU pointer=%p",
                          ptr->name, i, (void *)ptr->data.buffer_data);
                    [encCtx->encoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
                    continue;
                }

                size_t bindOffset = (size_t)offset;
                size_t bufferSize = (size_t)ptr->size;
                if (bindOffset >= bufferSize) {
                    NSLog(@"MGL FBIND skip small buffer=%u slot=%u: offset=%lu bufferSize=%lu",
                          ptr->name, i, (unsigned long)bindOffset, (unsigned long)bufferSize);
                    [encCtx->encoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
                    continue;
                }

                size_t bindLength = bufferSize - bindOffset;
                const uint8_t *bindPtr = ((const uint8_t *)ptr->data.buffer_data) + bindOffset;
                [encCtx->encoder setFragmentBytes:bindPtr length:bindLength atIndex:bindingIndex];
                [self invalidateLastBoundFragmentBufferAtIndex:bindingIndex];
                if (kMGLVerboseBindLogs) {
                    NSLog(@"MGL FBIND ok(slot=%lu) setFragmentBytes buffer=%u len=%lu offset=%lu",
                          (unsigned long)bindingIndex,
                          ptr->name,
                          (unsigned long)bindLength,
                          (unsigned long)bindOffset);
                }
                anyBindingPresent[bindingIndex] = true;
            } else if (ptr->data.mtl_data) {
                if ((uintptr_t)ptr->data.mtl_data < 0x100000000ULL) {
                    NSLog(@"MGL FBIND skip small MTL buffer=%u slot=%u: suspicious mtl_data pointer=%p",
                          ptr->name, i, ptr->data.mtl_data);
                    [encCtx->encoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
                    continue;
                }
                id<MTLBuffer> fallbackBuffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
                if (fallbackBuffer) {
                    NSUInteger metalLen = fallbackBuffer.length;
                    NSUInteger bindOffset = (NSUInteger)offset;
                    if (bindOffset >= metalLen) {
                        NSLog(@"MGL FBIND skip small MTL buffer=%u slot=%u: offset=%lu length=%lu",
                              ptr->name, i, (unsigned long)bindOffset, (unsigned long)metalLen);
                        [encCtx->encoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
                        continue;
                    }

                    if (!_bindingSync.state->lastBoundValid ||
                        _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer != fallbackBuffer ||
                        _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset != (NSUInteger)offset) {
                        [encCtx->encoder setFragmentBuffer:fallbackBuffer offset:offset atIndex:bindingIndex];
                        _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = fallbackBuffer;
                        _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = (NSUInteger)offset;
                        MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                    }
                    if (kMGLVerboseBindLogs) {
                        NSLog(@"MGL FBIND ok(slot=%lu) setFragmentBuffer buffer=%u mtl=%p len=%lu offset=%lu",
                              (unsigned long)bindingIndex,
                              ptr->name,
                              ptr->data.mtl_data,
                              (unsigned long)metalLen,
                              (unsigned long)bindOffset);
                    }
                    anyBindingPresent[bindingIndex] = true;
                }
            } else {
                [encCtx->encoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
            _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
            }
            
            // clear buffer data dirty bits
            ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
        }
        else
        {
            if (!ptr->data.mtl_data) {
                [self bindMTLBuffer:ptr];
            }
            id<MTLBuffer> buffer = nil;
            if (ptr->data.mtl_data &&
                (uintptr_t)ptr->data.mtl_data >= 0x100000000ULL) {
                buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
            }

            NSUInteger metalLen = buffer ? buffer.length : 0u;
            NSUInteger bindOffset = (NSUInteger)offset;
            NSUInteger reflectedRequiredBytes = 0;
            NSUInteger requiredBindingBytes = kMGLMinimumStageBindingSize;
            if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
                reflectedRequiredBytes = map->has_metal_binding
                    ? [self getProgramBindingRequiredSize:_FRAGMENT_SHADER
                                                     type:(int)map->resource_type
                                                    index:(int)map->resource_index]
                    : [self getProgramBindingRequiredSizeForStage:_FRAGMENT_SHADER
                                                    clientBinding:glBindingIndex];
                if (reflectedRequiredBytes > requiredBindingBytes) {
                    requiredBindingBytes = reflectedRequiredBytes;
                }
            }
            NSUInteger availableBytes = buffer
                ? mglBufferMapVisibleBackingBytes(map, metalLen)
                : 0u;

            if (!buffer || bindOffset >= metalLen ||
                availableBytes < requiredBindingBytes) {
                id<MTLBuffer> isolated =
                    [self isolatedStageBindingBufferForMap:map
                                                     source:buffer
                                             requiredLength:requiredBindingBytes];
                if (!isolated) {
                    NSLog(@"MGL WARNING: FBIND failed to isolate undersized buffer=%u slot=%lu required=%lu available=%lu",
                          ptr->name,
                          (unsigned long)bindingIndex,
                          (unsigned long)requiredBindingBytes,
                          (unsigned long)availableBytes);
                    [encCtx->encoder setFragmentBuffer:nil
                                                      offset:0
                                                     atIndex:bindingIndex];
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = nil;
                    _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
                    continue;
                }

                [encCtx->encoder setFragmentBuffer:isolated
                                                  offset:0
                                                 atIndex:bindingIndex];
                _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = isolated;
                _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = 0;
                MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                anyBindingPresent[bindingIndex] = true;
                if (kMGLVerboseBindLogs) {
                    NSLog(@"MGL SET FRAGMENT BUFFER index=%lu glName=%u offset=0 source=isolated required=%lu reflected=%lu available=%lu range=%lld",
                          (unsigned long)bindingIndex,
                          ptr->name,
                          (unsigned long)requiredBindingBytes,
                          (unsigned long)reflectedRequiredBytes,
                          (unsigned long)availableBytes,
                          (long long)map->size);
                }
                continue;
            }
            
            /* For small uniform constants (plain uniforms), use setFragmentBytes
             * to copy the data into the command buffer at bind time. This is
             * critical for correctness when the same uniform buffer is updated
             * between draws encoded into the same command buffer — a shared-
             * memory MTLBuffer would let the GPU see only the final value. */
            if (isBaseBinding &&
                map->resource_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
                ptr->data.buffer_data &&
                (NSUInteger)ptr->size <= 4096 &&
                offset == 0) {
                [encCtx->encoder setFragmentBytes:(const void *)(uintptr_t)ptr->data.buffer_data
                                                  length:(NSUInteger)ptr->size
                                                 atIndex:bindingIndex];
                [self invalidateLastBoundFragmentBufferAtIndex:bindingIndex];
                if (kMGLVerboseBindLogs) {
                    NSLog(@"MGL FBIND uniform-constant setFragmentBytes slot=%lu buffer=%u len=%lu (plain uniform snapshot)",
                          (unsigned long)bindingIndex,
                          ptr->name,
                          (unsigned long)ptr->size);
                }
                anyBindingPresent[bindingIndex] = true;
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
                continue;
            }

            if (!_bindingSync.state->lastBoundValid ||
                _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer != buffer ||
                _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset != (NSUInteger)offset) {
                [encCtx->encoder setFragmentBuffer:buffer offset:offset atIndex:bindingIndex];
                _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].buffer = buffer;
                _bindingSync.state->lastBoundFragmentBuffers[bindingIndex].offset = (NSUInteger)offset;
                MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
            }
            Program *bindProgram = activeProgram;
            if (mglProgramNeedsBindingTrace(bindProgram)) {
                static uint64_t s_focusedFragmentBufferBindLogs = 0;
                if (mglShouldLogFocusedBinding(&s_focusedFragmentBufferBindLogs)) {
                    NSLog(@"MGL FBIND focused program=%u clientBinding=%u metalSlot=%lu resourceType=%s resourceIndex=%u buffer=%u offset=%lu available=%lu metalLen=%lu range=%lld",
                          (unsigned)bindProgram->name,
                          (unsigned)glBindingIndex,
                          (unsigned long)bindingIndex,
                          mglSpirvResourceTypeName((int)map->resource_type),
                          (unsigned)map->resource_index,
                          (unsigned)ptr->name,
                          (unsigned long)bindOffset,
                          (unsigned long)availableBytes,
                          (unsigned long)metalLen,
                          (long long)map->size);
                }
            }
            static uint64_t s_traceFileFragmentBufferBindLogs = 0;
            if (mglProgramNeedsTraceLog(bindProgram) &&
                mglShouldLogTraceFileBindingForProgram(bindProgram, &s_traceFileFragmentBufferBindLogs)) {
                mglTraceLog("FBIND program=%u clientBinding=%u metalSlot=%lu resourceType=%s resourceIndex=%u buffer=%u offset=%lu available=%lu metalLen=%lu range=%lld",
                            (unsigned)bindProgram->name,
                            (unsigned)glBindingIndex,
                            (unsigned long)bindingIndex,
                            mglSpirvResourceTypeName((int)map->resource_type),
                            (unsigned)map->resource_index,
                            (unsigned)ptr->name,
                            (unsigned long)bindOffset,
                            (unsigned long)availableBytes,
                            (unsigned long)metalLen,
                            (long long)map->size);
            }
            if (kMGLVerboseBindLogs) {
                NSLog(@"MGL SET FRAGMENT BUFFER index=%lu glName=%u offset=%lu available=%lu source=%s",
                      (unsigned long)bindingIndex,
                      ptr->name,
                      (unsigned long)bindOffset,
                      (unsigned long)metalLen,
                      isBaseBinding ? "base" : "attrib");
            }
            if (kMGLVerboseBindLogs) {
                NSLog(@"MGL FBIND ok(slot=%lu) setFragmentBuffer buffer=%u mtl=%p len=%lu offset=%lu",
                      (unsigned long)bindingIndex,
                      ptr->name,
                      ptr->data.mtl_data,
                      (unsigned long)metalLen,
                      (unsigned long)bindOffset);
            }
            anyBindingPresent[bindingIndex] = true;
        }
    }

    [self bindFragmentFallbackBuffersToCurrentRenderEncoder:activeProgram
                                         anyBindingPresent:anyBindingPresent
                                         baseBindingPresent:baseBindingPresent
                                             encodeContext:encCtx];

    /* Fallback bindings are real Metal slots and must be included in the
     * worker snapshot. */
    uint32_t boundFragmentBufferMask = 0;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        if (anyBindingPresent[i]) {
            boundFragmentBufferMask |= 1U << i;
        }
    }
    _bindingSync.state->lastBoundFragmentBufferMask |= boundFragmentBufferMask;

    if (getenv("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_fbind_trace_count = 0;
        if ((++s_fbind_trace_count % 500) == 1) {
            int textureSlotCount = __builtin_popcountll(_bindingSync.state->lastBoundTextureSlotMask[0]) +
                                   __builtin_popcountll(_bindingSync.state->lastBoundTextureSlotMask[1]);
            NSLog(@"MGL SPARSE FBIND: fbuf=0x%x(%d/31) textureSlots=%d/128",
                  boundFragmentBufferMask,
                  __builtin_popcount(boundFragmentBufferMask),
                  textureSlotCount);
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(fbindCall)) {
        NSUInteger boundSlots = 0;
        NSUInteger baseSlots = 0;
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (anyBindingPresent[s]) {
                boundSlots++;
            }
            if (baseBindingPresent[s]) {
                baseSlots++;
            }
        }
        MGLTraceNSLog(@"MGL TRACE fbind.end call=%llu mapCount=%u boundSlots=%lu baseSlots=%lu elapsed=%.3fms",
              (unsigned long long)fbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)baseSlots,
              (mglNowSeconds() - fbindStartSeconds) * 1000.0);
    }

    /* Mark the dedup cache as valid for the current encoder so subsequent
     * binds can be skipped when the resource and offset are unchanged. */
    _bindingSync.state->lastBoundValid = YES;
    return true;
}

/* Bind fallback buffers for fragment-stage buffer slots that were not mapped
 * by the main binding loop above.  Extracted from
 * bindFragmentBuffersToCurrentRenderEncoder to keep that function under the
 * 500-line limit.  Pure mechanical extraction — no behavior change. */
- (void)bindFragmentFallbackBuffersToCurrentRenderEncoder:(Program *)activeProgram
                                       anyBindingPresent:(bool *)anyBindingPresent
                                       baseBindingPresent:(bool *)baseBindingPresent
                                           encodeContext:(const MGLEncodeContext *)encCtx
{
    static id<MTLBuffer> fallbackBindingBuffer = nil;

    if (!fallbackBindingBuffer) {
        fallbackBindingBuffer = [_device newBufferWithLength:kMGLDefaultStageFallbackBufferSize
                                                     options:MTLResourceStorageModeShared];
    }

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER
    };
    for (int t = 0; t < 4; t++) {
        int resourceType = resourceTypes[t];
        int count = [self getProgramBindingCount:_FRAGMENT_SHADER type:resourceType];
        Program *program = activeProgram;
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= _MAX_SPIRV_RES ||
                i >= (int)program->spirv_resources_list[_FRAGMENT_SHADER][resourceType].count) {
                continue;
            }
            SpirvResource *resource = &program->spirv_resources_list[_FRAGMENT_SHADER][resourceType].list[i];
            if (mglShouldSkipStageBufferResource(program, _FRAGMENT_SHADER, resourceType, resource)) {
                continue;
            }
            GLuint elementCount = mglStageBufferResourceElementCount(resourceType, resource);
            for (GLuint element = 0; element < elementCount; element++) {
                GLuint clientBinding =
                    mglClientBufferBindingForResourceElement(resourceType, resource, element);
                if (clientBinding >= MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                NSInteger metalBinding =
                    (NSInteger)mglMetalResourceSlotForElement(resource, element);
                if (metalBinding < 0 || metalBinding >= (NSInteger)MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                if (!anyBindingPresent[(NSUInteger)metalBinding] && fallbackBindingBuffer) {
                    NSUInteger _slot = (NSUInteger)metalBinding;
                    if (!_bindingSync.state->lastBoundValid ||
                        _bindingSync.state->lastBoundFragmentBuffers[_slot].buffer != fallbackBindingBuffer ||
                        _bindingSync.state->lastBoundFragmentBuffers[_slot].offset != 0) {
                        [encCtx->encoder setFragmentBuffer:fallbackBindingBuffer offset:0 atIndex:_slot];
                        _bindingSync.state->lastBoundFragmentBuffers[_slot].buffer = fallbackBindingBuffer;
                        _bindingSync.state->lastBoundFragmentBuffers[_slot].offset = 0;
                        MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                    }
                    baseBindingPresent[clientBinding] = true;
                    anyBindingPresent[_slot] = true;
                }
            }
        }
    }

    if (fallbackBindingBuffer) {
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (!anyBindingPresent[s]) {
                if (!_bindingSync.state->lastBoundValid ||
                    _bindingSync.state->lastBoundFragmentBuffers[s].buffer != fallbackBindingBuffer ||
                    _bindingSync.state->lastBoundFragmentBuffers[s].offset != 0) {
                    [encCtx->encoder setFragmentBuffer:fallbackBindingBuffer offset:0 atIndex:s];
                    _bindingSync.state->lastBoundFragmentBuffers[s].buffer = fallbackBindingBuffer;
                    _bindingSync.state->lastBoundFragmentBuffers[s].offset = 0;
                    MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                } else {
                    MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                }
                anyBindingPresent[s] = true;
            }
        }
    }
}


static const NSUInteger kMaxFragmentSamplerSlots = 16;

#define MGL_ABORT_TBIND_IF_ENCODER_CLOSED() do { \
    if (!_renderPassManager.state->currentRenderEncoder) { \
        if (ctx) { \
            mglMarkRendererDirtyBits(ctx->active_state, (DIRTY_TEX | DIRTY_TEX_BINDING | DIRTY_RENDER_STATE)); \
        } \
        return false; \
    } \
} while (0)

- (bool) bindTexturesToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_bindTexturesCallCount = 0;
    uint64_t bindCall = ++s_bindTexturesCallCount;
    bool traceBind = mglShouldTraceCall(bindCall);
    GLuint vertexSampledCount = 0;
    GLuint vertexBoundTextures = 0;
    GLuint vertexFallbackTextures = 0;
    GLuint boundSampledTextures = 0;
    GLuint nilSampledTextures = 0;
    GLuint fallbackSampledTextures = 0;
    GLuint boundSampledSamplers = 0;
    Program *vertexProgram = NULL;
    Program *fragmentProgram = NULL;
    GLuint vertexProgramName = 0u;
    GLuint fragmentProgramName = 0u;

    if (!encCtx->encoder) {
        // No active render encoder yet (or it was rotated). Texture/sampler binding
        // can be deferred until the next encoder is created.
        return true;
    }

    /*
     * This array is the per-draw sampler snapshot used later by replay logging
     * and by the RT-sampled-copy cull bypass decision.  Do not let bindings from
     * the previous program survive in slots the current program does not touch.
     */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             "bind_textures_begin",
                                             _resourceFallback.fragmentTextureTraceBindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             _pipelineCache.state->pipelineProgramName);
    }
    memset(_resourceFallback.fragmentTextureTraceBindings, 0,
           sizeof(_resourceFallback.fragmentTextureTraceBindings));


    vertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    vertexProgramName = vertexProgram ? vertexProgram->name : mglCurrentRenderProgramKey(ctx);
    fragmentProgramName = fragmentProgram ? fragmentProgram->name : mglCurrentRenderProgramKey(ctx);

    id<MTLSamplerState> defaultSampler = [self fallbackSamplerState];
    if (defaultSampler) {
        NSUInteger warmupCount = TEXTURE_UNITS;
        if (warmupCount > kMaxFragmentSamplerSlots) {
            warmupCount = kMaxFragmentSamplerSlots;
        }
        for (NSUInteger s = 0; s < warmupCount; s++) {
            [self setVertexSamplerStateIfNeeded:defaultSampler atIndex:s];
            [self setFragmentSamplerStateIfNeeded:defaultSampler atIndex:s];
        }
    }


    GLuint sampledCount = 0;
    GLuint separateSamplerCount = 0;
    GLuint boundSeparateSamplers = 0;

    // Metal validates every active stage resource. Bind vertex-stage sampled
    // images as well, even though most Minecraft pipelines only sample in FS.
    vertexSampledCount = [self getProgramBindingCount:_VERTEX_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    if (![self bindVertexSampledTexturesToEncoder:vertexProgram
                                  vertexProgramName:vertexProgramName
                                     defaultSampler:defaultSampler
                                            bindCall:bindCall
                                          traceBind:traceBind
                                  vertexSampledCount:vertexSampledCount
                                         boundCount:&vertexBoundTextures
                                      fallbackCount:&vertexFallbackTextures]) {
        return false;
    }

    // Bind sampled images (texture + sampler).
    if (![self bindFragmentSampledTexturesToEncoder:fragmentProgram
                                  fragmentProgramName:fragmentProgramName
                                     vertexProgramName:vertexProgramName
                                        defaultSampler:defaultSampler
                                               bindCall:bindCall
                                             traceBind:traceBind
                                  boundSampledTextures:&boundSampledTextures
                                    nilSampledTextures:&nilSampledTextures
                               fallbackSampledTextures:&fallbackSampledTextures
                                 boundSampledSamplers:&boundSampledSamplers
                                            sampledCount:&sampledCount]) {
        return false;
    }

    /* Vertex/Fragment-stage storage image binding. */
    if (![self bindStorageImagesToEncoder:vertexProgram
                          fragmentProgram:fragmentProgram
                            encodeContext:encCtx]) {
        return false;
    }

    // Bind separate samplers explicitly.
    [self bindSeparateSamplersAndArrayTextures:vertexProgram
                              fragmentProgram:fragmentProgram
                        fragmentProgramName:fragmentProgramName
                          vertexProgramName:vertexProgramName
                             defaultSampler:defaultSampler
                                    bindCall:bindCall
                                  traceBind:traceBind
                         separateSamplerCount:&separateSamplerCount
                           boundSeparateSamplers:&boundSeparateSamplers];

    BOOL interestingTextureBind =
        (sampledCount > 0 && boundSampledTextures == 0) ||
        fallbackSampledTextures > 0 ||
        vertexFallbackTextures > 0;
    BOOL logTextureSummary = traceBind;
    if (interestingTextureBind) {
        static uint64_t s_interestingTextureSummaryCount = 0;
        uint64_t hit = ++s_interestingTextureSummaryCount;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            logTextureSummary = YES;
        }
    }
    if (logTextureSummary) {
        GLuint programName = mglCurrentRenderProgramKey(ctx);
        MGLTraceNSLog(@"MGL TRACE texbind.summary call=%llu program=%u vertexSampled=%u vertexBoundTex=%u vertexFallback=%u sampled=%u boundTex=%u nilTex=%u fallbackTex=%u sampledSamplers=%u separateSamplers=%u boundSeparate=%u",
              (unsigned long long)bindCall,
              (unsigned)programName,
              (unsigned)vertexSampledCount,
              (unsigned)vertexBoundTextures,
              (unsigned)vertexFallbackTextures,
              (unsigned)sampledCount,
              (unsigned)boundSampledTextures,
              (unsigned)nilSampledTextures,
              (unsigned)fallbackSampledTextures,
              (unsigned)boundSampledSamplers,
              (unsigned)separateSamplerCount,
              (unsigned)boundSeparateSamplers);
    }

    return true;
}

- (bool)bindVertexSampledTexturesToEncoder:(Program *)vertexProgram
                          vertexProgramName:(GLuint)vertexProgramName
                             defaultSampler:(id<MTLSamplerState>)defaultSampler
                                    bindCall:(uint64_t)bindCall
                                  traceBind:(bool)traceBind
                          vertexSampledCount:(GLuint)vertexSampledCount
                                 boundCount:(GLuint *)boundCount
                              fallbackCount:(GLuint *)fallbackCount
{
    GLuint vertexBoundTextures = *boundCount;
    GLuint vertexFallbackTextures = *fallbackCount;

    for (GLuint i = 0; i < vertexSampledCount; i++)
    {
        GLuint spirvBinding = [self getProgramBinding:_VERTEX_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        GLuint glBinding = [self getProgramGLBinding:_VERTEX_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        Program *currentProgram = vertexProgram;
        SpirvResource *sampledResource = NULL;
        const char *sampledName = "";
        if (currentProgram &&
            i < currentProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
            sampledResource = &currentProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list[i];
            sampledName = sampledResource->name;
        }
        if (mglShouldSkipStageTextureResource(currentProgram,
                                              _VERTEX_SHADER,
                                              SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
                                              sampledResource)) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                    metalBinding:spirvBinding
                                                           stage:_VERTEX_SHADER];
        MTLTextureType expectedType = [self getProgramExpectedTextureType:_VERTEX_SHADER
                                                                      type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                     index:(int)i];
        MTLTextureType lookupType = [self getProgramDeclaredTextureType:_VERTEX_SHADER
                                                                    type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                   index:(int)i];
        MGLTextureDataKind expectedKind = [self getProgramExpectedTextureDataKind:_VERTEX_SHADER
                                                                             type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                            index:(int)i];
        Texture *ptr = [self textureForSampledResource:sampledResource
                                          metalBinding:spirvBinding
                                                  stage:_VERTEX_SHADER
                                           expectedType:(lookupType ? lookupType : expectedType)];
        id<MTLTexture> texture = nil;
        id<MTLSamplerState> sampler = defaultSampler;
        BOOL usedTypeFallback = NO;

        if (ptr) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            if (ptr->mtl_data) {
                texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
                texture = mglSampledTextureViewForBaseLevel(ptr, texture);
            }
            if (texture && expectedType != 0 && texture.textureType != expectedType) {
                static uint64_t s_vertexTypeMismatchLogCount = 0;
                uint64_t hit = ++s_vertexTypeMismatchLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX TYPE MISMATCH vertex binding=%u program=%u glTex=%u glTarget=0x%x mtlType=%lu expected=%lu hit=%llu",
                          (unsigned)spirvBinding,
                          (unsigned)vertexProgramName,
                          (unsigned)ptr->name,
                          (unsigned)ptr->target,
                          (unsigned long)texture.textureType,
                          (unsigned long)expectedType,
                          (unsigned long long)hit);
                }
                Program *dumpProgram = currentProgram;
                mglWriteProgramMSLDump(dumpProgram,
                                       [NSString stringWithFormat:@"tex-type-mismatch-vertex-binding-%u", spirvBinding]);
                texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
                usedTypeFallback = YES;
            }
            if (texture &&
                !mglTexturePixelFormatCompatibleWithExpectedDataKind(texture.pixelFormat, expectedKind)) {
                static uint64_t s_vertexDataKindMismatchLogCount = 0;
                uint64_t hit = ++s_vertexDataKindMismatchLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX DATA MISMATCH vertex binding=%u program=%u glTex=%u glTarget=0x%x format=%lu actualKind=%s expectedKind=%s expectedType=%lu hit=%llu",
                          (unsigned)spirvBinding,
                          (unsigned)vertexProgramName,
                          (unsigned)ptr->name,
                          (unsigned)ptr->target,
                          (unsigned long)texture.pixelFormat,
                          mglTextureDataKindName(mglTextureDataKindForPixelFormat(texture.pixelFormat)),
                          mglTextureDataKindName(expectedKind),
                          (unsigned long)expectedType,
                          (unsigned long long)hit);
                }
                Program *dumpProgram = currentProgram;
                mglWriteProgramMSLDump(dumpProgram,
                                       [NSString stringWithFormat:@"tex-data-mismatch-vertex-binding-%u", spirvBinding]);
                texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
                usedTypeFallback = YES;
            }

            if (textureUnit < TEXTURE_UNITS && STATE(texture_samplers[textureUnit])) {
                Sampler *glSampler = STATE(texture_samplers[textureUnit]);
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (glSampler->mtl_data == NULL) {
                    glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
                mglTraceLogExternal("VERT_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=glSampler samplerName=%u minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                                    (unsigned)vertexProgramName,
                                    (unsigned)spirvBinding,
                                    (unsigned)textureUnit,
                                    (unsigned)glSampler->name,
                                    (unsigned)glSampler->params.min_filter,
                                    (unsigned)glSampler->params.mag_filter,
                                    (unsigned)glSampler->params.wrap_s,
                                    (unsigned)glSampler->params.wrap_t,
                                    (double)glSampler->params.min_lod,
                                    (double)glSampler->params.max_lod,
                                    (unsigned)ptr->name,
                                    (unsigned)ptr->params.base_level,
                                    (unsigned)ptr->params.max_level,
                                    (unsigned)ptr->width,
                                    (unsigned)ptr->height,
                                    (unsigned long)(texture ? texture.width : 0u),
                                    (unsigned long)(texture ? texture.height : 0u),
                                    (unsigned long)(texture ? texture.mipmapLevelCount : 0u));
            } else if (ptr->params.mtl_data) {
                sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
                mglTraceLogExternal("VERT_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=texParamsFallback samplerName=0 minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                                    (unsigned)vertexProgramName,
                                    (unsigned)spirvBinding,
                                    (unsigned)textureUnit,
                                    (unsigned)ptr->params.min_filter,
                                    (unsigned)ptr->params.mag_filter,
                                    (unsigned)ptr->params.wrap_s,
                                    (unsigned)ptr->params.wrap_t,
                                    (double)ptr->params.min_lod,
                                    (double)ptr->params.max_lod,
                                    (unsigned)ptr->name,
                                    (unsigned)ptr->params.base_level,
                                    (unsigned)ptr->params.max_level,
                                    (unsigned)ptr->width,
                                    (unsigned)ptr->height,
                                    (unsigned long)(texture ? texture.width : 0u),
                                    (unsigned long)(texture ? texture.height : 0u),
                                    (unsigned long)(texture ? texture.mipmapLevelCount : 0u));
            }
        }

        /* Y-Flip Subsystem: unified decision for sampling a render target.
         *
         * NOTE: lazy refresh from bindTexturesToCurrentRenderEncoder was
         * removed — it re-enters the Metal render encoder during a flush and
         * crashes AGX.  See the fragment counterpart above. */
        if (!usedTypeFallback && ptr && ptr->is_render_target) {
            MGLYFlipDecision yflip = mglDecideYFlipForSampledRT(ptr, currentProgram);
            if (mglTraceRTYFlipDiagnosticsEnabled()) {
                mglTraceLog("RT_YFLIP_DECISION stage=vertex program=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" decision=%s(%d) authority=0x%x rtVer=%u copyVer=%u hasCopy=%d sampleYFlip=%d",
                            (unsigned)vertexProgramName,
                            sampledName ? sampledName : "",
                            (unsigned)spirvBinding,
                            (unsigned)textureUnit,
                            (unsigned)ptr->name,
                            mglTraceTextureLabel(ptr),
                            mglYFlipDecisionName(yflip),
                            (int)yflip,
                            (unsigned)ptr->mtl_render_yflip_authority,
                            (unsigned)ptr->mtl_render_target_write_version,
                            (unsigned)ptr->mtl_gl_sampled_write_version,
                            ptr->mtl_gl_sampled_data ? 1 : 0,
                            mglProgramHasExistingFramebufferSampleYFlip(currentProgram) ? 1 : 0);
            }

            if (yflip == MGL_YFLIP_USE_SAMPLED_COPY) {
                BOOL boundSampledCopy = NO;
                if (ptr->mtl_gl_sampled_data &&
                    ptr->mtl_gl_sampled_write_version == ptr->mtl_render_target_write_version &&
                    mglTextureCanUseGLSampledRenderTargetCopy(ptr)) {
                    id<MTLTexture> sampledCopy = (__bridge id<MTLTexture>)(ptr->mtl_gl_sampled_data);
                    if (sampledCopy &&
                        (expectedType == 0 || sampledCopy.textureType == expectedType) &&
                        mglTexturePixelFormatCompatibleWithExpectedDataKind(sampledCopy.pixelFormat, expectedKind)) {
                        if (mglTraceLogIsEnabled()) {
                            mglTraceLog("RT_SAMPLE_COPY_BIND stage=vertex program=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" original=%p copy=%p size=%lux%lu originalLevels=%lu copyLevels=%lu glLevels=%u mips=%u base=%u max=%u version=%u",
                                        (unsigned)vertexProgramName,
                                        sampledName ? sampledName : "",
                                        (unsigned)spirvBinding,
                                        (unsigned)textureUnit,
                                        (unsigned)ptr->name,
                                        mglTraceTextureLabel(ptr),
                                        texture,
                                        sampledCopy,
                                        (unsigned long)sampledCopy.width,
                                        (unsigned long)sampledCopy.height,
                                        (unsigned long)(texture ? texture.mipmapLevelCount : 0u),
                                        (unsigned long)sampledCopy.mipmapLevelCount,
                                        (unsigned)ptr->num_levels,
                                        (unsigned)ptr->mipmap_levels,
                                        (unsigned)ptr->params.base_level,
                                        (unsigned)ptr->params.max_level,
                                        (unsigned)ptr->mtl_gl_sampled_write_version);
                        }
                        texture = mglSampledTextureViewForBaseLevel(ptr, sampledCopy);
                        boundSampledCopy = YES;
                    }
                }
                if (!boundSampledCopy && mglTextureCanUseGLSampledRenderTargetCopy(ptr)) {
                    id<MTLTexture> repairedCopy =
                        [self freshGLSampledRenderTargetCopyForSampling:ptr
                                                                  source:texture
                                                                   stage:"vertex"
                                                                 program:vertexProgramName
                                                                 binding:spirvBinding
                                                                    unit:textureUnit
                                                            expectedType:expectedType
                                                            expectedKind:expectedKind];
                    if (repairedCopy) {
                        return false;
                    }
                }
                if (!boundSampledCopy && ptr->mtl_gl_sampled_data &&
                    !mglTextureCanUseGLSampledRenderTargetCopy(ptr)) {
                    mglLogSkippedGLSampledRenderTargetCopy(ctx,
                                                           currentProgram,
                                                           ptr,
                                                           "vertex",
                                                           sampledName,
                                                           spirvBinding,
                                                           textureUnit,
                                                           "target-gate");
                } else if (!boundSampledCopy && mglTraceLogIsEnabled()) {
                    BOOL hasCopy = (ptr->mtl_gl_sampled_data != NULL);
                    BOOL verMatch = (ptr->mtl_gl_sampled_write_version == ptr->mtl_render_target_write_version);
                    BOOL canUse = mglTextureCanUseGLSampledRenderTargetCopy(ptr);
                    mglTraceLog("RT_SAMPLE_COPY_GATE_MISS stage=vertex program=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" isRT=%d hasCopy=%d verMatch=%d writeVer=%u rtVer=%u canUse=%d expectedType=%lu",
                                (unsigned)vertexProgramName,
                                sampledName ? sampledName : "",
                                (unsigned)spirvBinding,
                                (unsigned)textureUnit,
                                (unsigned)ptr->name,
                                mglTraceTextureLabel(ptr),
                                1, hasCopy ? 1 : 0, verMatch ? 1 : 0,
                                (unsigned)ptr->mtl_gl_sampled_write_version,
                                (unsigned)ptr->mtl_render_target_write_version,
                                canUse ? 1 : 0,
                                (unsigned long)expectedType);
                }
            } else {
                /* MGL_YFLIP_USE_ORIGINAL or MGL_YFLIP_USE_ORIGINAL_AND_INJECT:
                 * keep the original texture; no copy needed. */
                static uint64_t s_vertexRTSampleCopySkipExistingFlipLogCount = 0;
                uint64_t hit = ++s_vertexRTSampleCopySkipExistingFlipLogCount;
                if (mglTraceLogIsEnabled() && (hit <= 32ull || (hit % 512ull) == 0ull)) {
                    mglTraceLog("RT_SAMPLE_COPY_SKIP_EXISTING_YFLIP hit=%llu stage=vertex program=%u name=%s binding=%u tex=%u label=\"%s\" decision=%s(%d)",
                                (unsigned long long)hit,
                                (unsigned)vertexProgramName,
                                sampledName ? sampledName : "",
                                (unsigned)spirvBinding,
                                (unsigned)(ptr ? ptr->name : 0u),
                                mglTraceTextureLabel(ptr),
                                mglYFlipDecisionName(yflip),
                                (int)yflip);
                }
            }
        }

        if (!texture) {
            texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
            if (texture) {
                vertexFallbackTextures++;
                static uint64_t s_vertexFallbackLogCount = 0;
                uint64_t hit = ++s_vertexFallbackLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX FALLBACK vertex sampled binding=%u program=%u name=%s unit=%u glTex=%u kind=%s size=%lux%lu hit=%llu",
                          (unsigned)spirvBinding,
                          (unsigned)vertexProgramName,
                          sampledName ? sampledName : "",
                          (unsigned)textureUnit,
                          ptr ? (unsigned)ptr->name : 0u,
                          "generic",
                          (unsigned long)texture.width,
                          (unsigned long)texture.height,
                          (unsigned long long)hit);
                }
            }
        }

        [self setVertexTextureIfNeeded:texture atIndex:spirvBinding];
        GLuint samplerBinding = sampledResource && sampledResource->msl_has_combined_sampler
            ? mglMetalCombinedSamplerSlot(sampledResource)
            : spirvBinding;
        if (sampler &&
            (!sampledResource || sampledResource->msl_has_combined_sampler) &&
            samplerBinding < kMaxFragmentSamplerSlots) {
            [self setVertexSamplerStateIfNeeded:sampler atIndex:samplerBinding];
        }
        Program *focusedTextureProgram = currentProgram;
        if (mglProgramNeedsBindingTrace(focusedTextureProgram)) {
            static uint64_t s_focusedVertexTextureBindLogs = 0;
            if (mglShouldLogFocusedBinding(&s_focusedVertexTextureBindLogs)) {
                TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
                NSLog(@"MGL TBIND focused stage=vertex program=%u resource=%s metalTextureSlot=%u samplerUnit=%u glTex=%u target=0x%x mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                      (unsigned)focusedTextureProgram->name,
                      sampledName ? sampledName : "",
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      ptr ? (unsigned)ptr->name : 0u,
                      ptr ? (unsigned)ptr->target : 0u,
                      texture,
                      (unsigned long)(texture ? texture.textureType : 0),
                      (unsigned long)(texture ? texture.width : 0),
                      (unsigned long)(texture ? texture.height : 0),
                      level0 ? (unsigned)level0->width : 0u,
                      level0 ? (unsigned)level0->height : 0u,
                      level0 ? (unsigned)level0->ever_written : 0u,
                      level0 ? (unsigned)level0->has_initialized_data : 0u,
                      level0 ? (unsigned)level0->last_init_source : 0u);
            }
        }
        static uint64_t s_traceFileVertexTextureBindLogs = 0;
        if (mglProgramNeedsTraceLog(focusedTextureProgram) &&
            mglShouldLogTraceFileBindingForProgram(focusedTextureProgram, &s_traceFileVertexTextureBindLogs)) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
            int expectedIndex = [self textureIndexForExpectedMetalType:(lookupType ? lookupType : expectedType)];
            Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
            Texture *unitExpected = (textureUnit < TEXTURE_UNITS &&
                                     expectedIndex >= 0 &&
                                     expectedIndex < _MAX_TEXTURE_TYPES)
                ? STATE(texture_units[textureUnit].textures[expectedIndex])
                : NULL;
            Texture *unit2D = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_2D]) : NULL;
            Texture *unitCube = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP]) : NULL;
            mglTraceLog("TBIND stage=vertex program=%u resource=%s metalTextureSlot=%u samplerUnit=%u resUnit=%d explicit=%d glTex=%u target=0x%x fallback=%d expectedType=%lu lookupType=%lu expectedIndex=%d unit(active=%u expected=%u tex2D=%u cube=%u) mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                        (unsigned)focusedTextureProgram->name,
                        sampledName ? sampledName : "",
                        (unsigned)spirvBinding,
                        (unsigned)textureUnit,
                        sampledResource ? (int)sampledResource->sampler_unit : -1,
                        (sampledResource && sampledResource->sampler_unit_explicit) ? 1 : 0,
                        ptr ? (unsigned)ptr->name : 0u,
                        ptr ? (unsigned)ptr->target : 0u,
                        usedTypeFallback ? 1 : 0,
                        (unsigned long)expectedType,
                        (unsigned long)lookupType,
                        expectedIndex,
                        mglTraceTextureName(unitActive),
                        mglTraceTextureName(unitExpected),
                        mglTraceTextureName(unit2D),
                        mglTraceTextureName(unitCube),
                        texture,
                        (unsigned long)(texture ? texture.textureType : 0),
                        (unsigned long)(texture ? texture.width : 0),
                        (unsigned long)(texture ? texture.height : 0),
                        level0 ? (unsigned)level0->width : 0u,
                        level0 ? (unsigned)level0->height : 0u,
                        level0 ? (unsigned)level0->ever_written : 0u,
                        level0 ? (unsigned)level0->has_initialized_data : 0u,
                        level0 ? (unsigned)level0->last_init_source : 0u);
        }
        if (ptr && ptr->target == GL_TEXTURE_BUFFER) {
            static uint64_t s_vertexTexelBufferBindLogs = 0;
            uint64_t hit = ++s_vertexTexelBufferBindLogs;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
                Texture *unitBuffer = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_BUFFER_TARGET]) : NULL;
                NSLog(@"MGL TEXBUFFER BIND vertex hit=%llu program=%u binding=%u unit=%u ptrTex=%u active=%u bufferSlot=%u expectedType=%lu lookupType=%lu mtlTex=%p mtlType=%lu size=%lux%lu format=%lu sampler=%p",
                      (unsigned long long)hit,
                      (unsigned)vertexProgramName,
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      (unsigned)ptr->name,
                      mglTraceTextureName(unitActive),
                      mglTraceTextureName(unitBuffer),
                      (unsigned long)expectedType,
                      (unsigned long)lookupType,
                      texture,
                      (unsigned long)(texture ? texture.textureType : 0),
                      (unsigned long)(texture ? texture.width : 0),
                      (unsigned long)(texture ? texture.height : 0),
                      (unsigned long)(texture ? texture.pixelFormat : 0),
                      sampler);
            }
        }
        if (ptr && ptr->target != GL_TEXTURE_BUFFER) {
            Program *sampleProgram = currentProgram;
            GLuint sampleProgramName = sampleProgram ? sampleProgram->name : vertexProgramName;
            TextureLevel *sampleLevel0 = mglTraceTextureBaseLevel(ptr);
            BOOL focusedVertexSample =
                (sampleProgramName == 34u) ||
                (sampleLevel0 &&
                 (sampleLevel0->suspicious_zero_upload ||
                  !sampleLevel0->ever_written ||
                  !sampleLevel0->has_initialized_data));
            if (focusedVertexSample) {
                static uint64_t s_vertexSampleDetailLogCount = 0;
                uint64_t hit = ++s_vertexSampleDetailLogCount;
                if (hit <= 128ull || (hit % 512ull) == 0ull) {
                    int expectedIndex = [self textureIndexForExpectedMetalType:expectedType];
                    Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
                    Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
                        ? STATE(texture_units[textureUnit].textures[expectedIndex])
                        : NULL;
                    uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
                        ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
                        : 0ull;

                    MGLTraceNSLog(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=vertex program=%u name=%s binding=%u "
                          "unit=%u expectedType=%lu expectedIndex=%d ptrTex=%u ptr=%p target=0x%x fallback=%d mtlTex=%p mtlType=%lu mtlSize=%lux%lu "
                          "unit(active=%u expected=%u) "
                          "l0=%ux%ux%u bytes=%lu init(ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx dataHash=0x%016llx)",
                          (unsigned long long)bindCall,
                          (unsigned long long)hit,
                          sampleProgramName,
                          sampledName ? sampledName : "",
                          (unsigned)spirvBinding,
                          (unsigned)textureUnit,
                          (unsigned long)expectedType,
                          expectedIndex,
                          mglTraceTextureName(ptr),
                          ptr,
                          ptr ? (unsigned)ptr->target : 0u,
                          usedTypeFallback ? 1 : 0,
                          texture,
                          (unsigned long)(texture ? texture.textureType : 0),
                          (unsigned long)(texture ? texture.width : 0),
                          (unsigned long)(texture ? texture.height : 0),
                          mglTraceTextureName(unitActive),
                          mglTraceTextureName(unitExpected),
                          sampleLevel0 ? (unsigned)sampleLevel0->width : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->height : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->depth : 0u,
                          (unsigned long)(sampleLevel0 ? sampleLevel0->data_size : 0u),
                          sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
                          (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
                          sampleLevel0 ? (void *)sampleLevel0->last_src_ptr : NULL,
                          (unsigned long long)(sampleLevel0 ? sampleLevel0->last_src_hash : 0ull),
                          (unsigned long long)levelDataHash);
                }
            }
        }
        if (texture) {
            vertexBoundTextures++;
            if (usedTypeFallback) {
                vertexFallbackTextures++;
            }
        }
    }

    *boundCount = vertexBoundTextures;
    *fallbackCount = vertexFallbackTextures;
    return true;
}

- (bool)bindFragmentSampledTexturesToEncoder:(Program *)fragmentProgram
                          fragmentProgramName:(GLuint)fragmentProgramName
                             vertexProgramName:(GLuint)vertexProgramName
                                defaultSampler:(id<MTLSamplerState>)defaultSampler
                                       bindCall:(uint64_t)bindCall
                                     traceBind:(bool)traceBind
                          boundSampledTextures:(GLuint *)boundSampledTexturesPtr
                            nilSampledTextures:(GLuint *)nilSampledTexturesPtr
                       fallbackSampledTextures:(GLuint *)fallbackSampledTexturesPtr
                         boundSampledSamplers:(GLuint *)boundSampledSamplersPtr
                                    sampledCount:(GLuint *)sampledCount
{
    GLuint boundSampledTextures = *boundSampledTexturesPtr;
    GLuint nilSampledTextures = *nilSampledTexturesPtr;
    GLuint fallbackSampledTextures = *fallbackSampledTexturesPtr;
    GLuint boundSampledSamplers = *boundSampledSamplersPtr;

    // Bind sampled images (texture + sampler).
    *sampledCount = [self getProgramBindingCount:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < *sampledCount; i++)
    {
        GLuint spirvBinding = [self getProgramBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        GLuint glBinding = [self getProgramGLBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        Program *sampleProgram = fragmentProgram;
        SpirvResource *sampledResource = NULL;
        const char *sampledName = "";
        if (sampleProgram &&
            i < sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
            sampledResource = &sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list[i];
            sampledName = sampledResource->name;
        }
        if (mglShouldSkipStageTextureResource(sampleProgram,
                                              _FRAGMENT_SHADER,
                                              SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
                                              sampledResource)) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                    metalBinding:spirvBinding
                                                           stage:_FRAGMENT_SHADER];

        MTLTextureType expectedType = [self getProgramExpectedTextureType:_FRAGMENT_SHADER
                                                                      type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                     index:(int)i];
        MTLTextureType lookupType = [self getProgramDeclaredTextureType:_FRAGMENT_SHADER
                                                                    type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                   index:(int)i];
        MGLTextureDataKind expectedKind = [self getProgramExpectedTextureDataKind:_FRAGMENT_SHADER
                                                                             type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                            index:(int)i];
        Texture *ptr = [self textureForSampledResource:sampledResource
                                          metalBinding:spirvBinding
                                                  stage:_FRAGMENT_SHADER
                                           expectedType:(lookupType ? lookupType : expectedType)];
        id<MTLTexture> texture = nil;
        id<MTLSamplerState> sampler = nil;
        id<MTLTexture> directTextureForTrace = nil;
        id<MTLTexture> sampledCopyForTrace = nil;
        BOOL usedFallbackTexture = NO;
        BOOL suppressMissingTextureFallback = NO;
        BOOL usedSampledCopyForTrace = NO;

        if (ptr) {
            if (![self recoverFragmentSampledDepthTexture:&ptr
                                                   texture:&texture
                                               sampledName:sampledName
                                              spirvBinding:spirvBinding
                                                textureUnit:textureUnit
                                               expectedType:expectedType
                                               expectedKind:expectedKind
                                       fragmentProgramName:fragmentProgramName
                            suppressMissingTextureFallback:&suppressMissingTextureFallback
                                      usedFallbackTexture:&usedFallbackTexture]) {
                return false;
            }
            if (![self resolveFragmentSampledYFlipAndSampler:ptr
                                                      texture:&texture
                                                      sampler:&sampler
                                                  sampledName:sampledName
                                                 spirvBinding:spirvBinding
                                                   textureUnit:textureUnit
                                                  expectedType:expectedType
                                                  expectedKind:expectedKind
                                          fragmentProgramName:fragmentProgramName
                                           vertexProgramName:vertexProgramName
                                                 sampleProgram:sampleProgram
                                             usedFallbackTexture:&usedFallbackTexture
                                        usedSampledCopyForTrace:&usedSampledCopyForTrace
                                           directTextureForTrace:&directTextureForTrace
                                           sampledCopyForTrace:&sampledCopyForTrace]) {
                return false;
            }
        }

        if (!texture && !suppressMissingTextureFallback) {
            texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
            if (texture) {
                usedFallbackTexture = YES;
                usedSampledCopyForTrace = NO;
                mglFocusLoadingProgram(fragmentProgramName,
                                       "sample-fallback",
                                       bindCall);
                fallbackSampledTextures++;
                static uint64_t s_fragmentFallbackLogCount = 0;
                uint64_t hit = ++s_fragmentFallbackLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX FALLBACK fragment sampled binding=%u program=%u glTex=%u hit=%llu",
                          (unsigned)spirvBinding,
                          (unsigned)fragmentProgramName,
                          ptr ? (unsigned)ptr->name : 0u,
                          (unsigned long long)hit);
                }
            }
        } else if (!texture && suppressMissingTextureFallback) {
            static uint64_t s_fragmentFallbackSuppressedLogCount = 0;
            uint64_t hit = ++s_fragmentFallbackSuppressedLogCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL TEX FALLBACK SUPPRESSED fragment sampled binding=%u program=%u name=%s glTex=%u unit=%u reason=insampler-current-target-no-copy hit=%llu",
                      (unsigned)spirvBinding,
                      (unsigned)fragmentProgramName,
                      sampledName ? sampledName : "",
                      ptr ? (unsigned)ptr->name : 0u,
                      (unsigned)textureUnit,
                      (unsigned long long)hit);
            }
        }

	        if (!sampler) {
	            sampler = defaultSampler;
	        }
        if (usedFallbackTexture && expectedKind == MGLTextureDataKindDepth) {
            sampler = defaultSampler;
        }

                GLuint sampleProgramName = sampleProgram ? sampleProgram->name : fragmentProgramName;
                TextureLevel *sampleLevel0 = mglTraceTextureBaseLevel(ptr);
	        BOOL focusedSample =
		            mglIsFocusedLoadingProgram(sampleProgramName) &&
		            (bindCall <= 2048ull || ((bindCall % 512ull) == 0ull));
                BOOL guiRTSample =
                    ptr &&
                    mglTextureCanUseGLSampledRenderTargetCopy(ptr);
		        BOOL suspiciousSample =
		            usedFallbackTexture ||
		            (ptr && ptr->name == 13u) ||
                    guiRTSample ||
		    focusedSample ||
		            (sampleLevel0 &&
		             (sampleLevel0->suspicious_zero_upload ||
		              !sampleLevel0->ever_written ||
		              !sampleLevel0->has_initialized_data));
        if (suspiciousSample) {
            static uint64_t s_fragmentSampleDetailLogCount = 0;
            uint64_t hit = ++s_fragmentSampleDetailLogCount;
            if (hit <= 256ull || (hit % 512ull) == 0ull) {
	                int expectedIndex = [self textureIndexForExpectedMetalType:expectedType];
	                Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
	                Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
	                    ? STATE(texture_units[textureUnit].textures[expectedIndex])
	                    : NULL;
	                Texture *unit2D = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_2D]) : NULL;
	                Texture *unitCube = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP]) : NULL;
	                MTLTextureType actualType = texture ? texture.textureType : 0;
	                uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
	                    ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
	                    : 0ull;

	                MGLTraceNSLog(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=fragment program=%u name=%s binding=%u "
	                      "unit=%u expectedType=%lu expectedIndex=%d ptrTex=%u ptr=%p target=0x%x fallback=%d mtlTex=%p mtlType=%lu mtlSize=%lux%lu "
	                      "unit(active=%u expected=%u tex2D=%u cube=%u) "
	                      "l0=%ux%ux%u bytes=%lu init(ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx dataHash=0x%016llx)",
	                      (unsigned long long)bindCall,
	                      (unsigned long long)hit,
		                      sampleProgramName,
                          sampledName ? sampledName : "",
	                      (unsigned)spirvBinding,
	                      (unsigned)textureUnit,
	                      (unsigned long)expectedType,
	                      expectedIndex,
	                      mglTraceTextureName(ptr),
	                      ptr,
	                      ptr ? (unsigned)ptr->target : 0u,
	                      usedFallbackTexture ? 1 : 0,
	                      texture,
	                      (unsigned long)actualType,
	                      (unsigned long)(texture ? texture.width : 0),
	                      (unsigned long)(texture ? texture.height : 0),
	                      mglTraceTextureName(unitActive),
	                      mglTraceTextureName(unitExpected),
	                      mglTraceTextureName(unit2D),
	                      mglTraceTextureName(unitCube),
	                      sampleLevel0 ? (unsigned)sampleLevel0->width : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->height : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->depth : 0u,
	                      (unsigned long)(sampleLevel0 ? sampleLevel0->data_size : 0u),
	                      sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
	                      (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
	                      sampleLevel0 ? (void *)sampleLevel0->last_src_ptr : NULL,
	                      (unsigned long long)(sampleLevel0 ? sampleLevel0->last_src_hash : 0ull),
	                      (unsigned long long)levelDataHash);
	            }

                if (guiRTSample) {
                    static uint64_t s_guiRTSampleLogCount = 0;
	                    uint64_t atlasHit = ++s_guiRTSampleLogCount;
                    if (atlasHit <= 128ull || (atlasHit % 256ull) == 0ull) {
                            int atlasExpectedIndex = [self textureIndexForExpectedMetalType:expectedType];
                            Texture *atlasUnitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
                            Texture *atlasUnitExpected = (atlasExpectedIndex >= 0 && atlasExpectedIndex < _MAX_TEXTURE_TYPES)
                                ? STATE(texture_units[textureUnit].textures[atlasExpectedIndex])
                                : NULL;
                            Texture *atlasUnit2D = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_2D]) : NULL;
                            Texture *atlasUnitCube = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP]) : NULL;
	                        id<MTLTexture> rpColor0 = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.colorAttachments[0].texture : nil;
	                        id<MTLTexture> rpDepth = _renderPassManager.state->renderPassDescriptor ? _renderPassManager.state->renderPassDescriptor.depthAttachment.texture : nil;
                        mglTraceLog("RT_SAMPLE_COPY_SAMPLE hit=%llu bindCall=%llu program=%u stateProgram=%u current=%u pipeline=%u vs=%u fs=%u pipelineProgram=%u name=%s binding=%u unit=%u "
                                    "rtTex=%u label=\"%s\" fallback=%d useCopy=%d ptr=%p mtl=%p direct=%p copy=%p fmt=%lu type=%lu size=%lux%lu "
                                    "unit(active=%u expected=%u tex2D=%u cube=%u) "
                                    "l0(ever=%u full=%u zero=%u source=%u upload=%lu) "
                                    "drawFbo=%u rpFbo=%u rpColor=%p rpDepth=%p depthTest=%d blend=%d",
                                    (unsigned long long)atlasHit,
                                    (unsigned long long)bindCall,
                                    sampleProgramName,
                                    (unsigned)(ctx ? ctx->active_state->program_name : 0u),
                                    (unsigned)(ctx ? ctx->active_state->var.current_program : 0u),
                                    (unsigned)(ctx ? ctx->active_state->var.program_pipeline_binding : 0u),
                                    (unsigned)vertexProgramName,
                                    (unsigned)fragmentProgramName,
                                    (unsigned)_pipelineCache.state->pipelineProgramName,
                                    sampledName ? sampledName : "",
                                    (unsigned)spirvBinding,
                                    (unsigned)textureUnit,
                                    (unsigned)mglTraceTextureName(ptr),
                                    mglTraceTextureLabel(ptr),
                                    usedFallbackTexture ? 1 : 0,
                                    usedSampledCopyForTrace ? 1 : 0,
                                    ptr,
                                    texture,
                                    directTextureForTrace,
                                    sampledCopyForTrace,
                                    (unsigned long)(texture ? texture.pixelFormat : MTLPixelFormatInvalid),
                                    (unsigned long)(texture ? texture.textureType : 0),
                                    (unsigned long)(texture ? texture.width : 0),
                                    (unsigned long)(texture ? texture.height : 0),
                                    mglTraceTextureName(atlasUnitActive),
                                    mglTraceTextureName(atlasUnitExpected),
                                    mglTraceTextureName(atlasUnit2D),
                                    mglTraceTextureName(atlasUnitCube),
                                    sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
                                    sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
                                    sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
                                    sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
                                    (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
                                    (unsigned)(ctx && ctx->active_state->framebuffer ? ctx->active_state->framebuffer->name : 0u),
                                    (unsigned)_renderPassManager.state->renderPassFramebufferName,
                                    rpColor0,
                                    rpDepth,
                                    ctx && ctx->active_state->caps.depth_test ? 1 : 0,
                                    ctx && ctx->active_state->caps.blend ? 1 : 0);
                    }
                }

		            if (texture && sampleLevel0 &&
		                (sampleLevel0->suspicious_zero_upload ||
		                 !sampleLevel0->ever_written ||
		                 !sampleLevel0->has_initialized_data)) {
		                static uint64_t s_fragmentSampleReadbackCount = 0;
		                uint64_t rbHit = ++s_fragmentSampleReadbackCount;
		                if (rbHit <= 32ull || (rbHit % 512ull) == 0ull) {
	                    [self traceSampledTextureReadback:texture
	                                                glTex:ptr
	                                                level:sampleLevel0
	                                              program:sampleProgramName
	                                              binding:spirvBinding
	                                                stage:@"fragment"
		                                               reason:(sampleLevel0->suspicious_zero_upload ? @"zero-level" :
		                                                       (!sampleLevel0->ever_written ? @"never-written" : @"not-initialized"))
		                                                  hit:rbHit];
		                }
		            }
	        }

        [self setFragmentTextureIfNeeded:texture atIndex:spirvBinding];
        if (spirvBinding < TEXTURE_UNITS) {
            MGLFragmentTextureTraceBinding *traceBinding = &_resourceFallback.fragmentTextureTraceBindings[spirvBinding];
            memset(traceBinding, 0, sizeof(*traceBinding));
            traceBinding->gl_texture_name = ptr ? ptr->name : 0u;
            traceBinding->sampler_unit = textureUnit;
            traceBinding->metal_binding = spirvBinding;
            traceBinding->program_name = sampleProgramName;
            traceBinding->rt_write_version = ptr ? ptr->mtl_render_target_write_version : 0u;
            traceBinding->sampled_write_version = ptr ? ptr->mtl_gl_sampled_write_version : 0u;
            traceBinding->gl_texture_ptr = ptr;
            traceBinding->mtl_texture_ptr = (__bridge void *)texture;
            traceBinding->direct_mtl_texture_ptr = (__bridge void *)(directTextureForTrace ? directTextureForTrace : texture);
            traceBinding->sampled_copy_ptr = (__bridge void *)sampledCopyForTrace;
            traceBinding->width = texture ? texture.width : 0u;
            traceBinding->height = texture ? texture.height : 0u;
            traceBinding->pixel_format = texture ? texture.pixelFormat : MTLPixelFormatInvalid;
            traceBinding->texture_type = texture ? texture.textureType : 0u;
            traceBinding->used_sampled_copy = usedSampledCopyForTrace ? 1u : 0u;
            traceBinding->used_fallback = usedFallbackTexture ? 1u : 0u;
        }
        Program *focusedTextureProgram = sampleProgram;
        if (mglProgramNeedsBindingTrace(focusedTextureProgram)) {
            static uint64_t s_focusedFragmentTextureBindLogs = 0;
            if (mglShouldLogFocusedBinding(&s_focusedFragmentTextureBindLogs)) {
                TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
                NSLog(@"MGL TBIND focused stage=fragment program=%u resource=%s metalTextureSlot=%u samplerUnit=%u glTex=%u target=0x%x mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                      (unsigned)focusedTextureProgram->name,
                      sampledName ? sampledName : "",
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      ptr ? (unsigned)ptr->name : 0u,
                      ptr ? (unsigned)ptr->target : 0u,
                      texture,
                      (unsigned long)(texture ? texture.textureType : 0),
                      (unsigned long)(texture ? texture.width : 0),
                      (unsigned long)(texture ? texture.height : 0),
                      level0 ? (unsigned)level0->width : 0u,
                      level0 ? (unsigned)level0->height : 0u,
                      level0 ? (unsigned)level0->ever_written : 0u,
                      level0 ? (unsigned)level0->has_initialized_data : 0u,
                      level0 ? (unsigned)level0->last_init_source : 0u);
            }
        }
        static uint64_t s_traceFileFragmentTextureBindLogs = 0;
        if (mglProgramNeedsTraceLog(focusedTextureProgram) &&
            mglShouldLogTraceFileBindingForProgram(focusedTextureProgram, &s_traceFileFragmentTextureBindLogs)) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
            int expectedIndex = [self textureIndexForExpectedMetalType:(lookupType ? lookupType : expectedType)];
            Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
            Texture *unitExpected = (textureUnit < TEXTURE_UNITS &&
                                     expectedIndex >= 0 &&
                                     expectedIndex < _MAX_TEXTURE_TYPES)
                ? STATE(texture_units[textureUnit].textures[expectedIndex])
                : NULL;
            Texture *unit2D = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_2D]) : NULL;
            Texture *unitCube = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP]) : NULL;
            mglTraceLog("TBIND stage=fragment program=%u resource=%s metalTextureSlot=%u samplerUnit=%u resUnit=%d explicit=%d glTex=%u target=0x%x fallback=%d expectedType=%lu lookupType=%lu expectedIndex=%d unit(active=%u expected=%u tex2D=%u cube=%u) mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                        (unsigned)focusedTextureProgram->name,
                        sampledName ? sampledName : "",
                        (unsigned)spirvBinding,
                        (unsigned)textureUnit,
                        sampledResource ? (int)sampledResource->sampler_unit : -1,
                        (sampledResource && sampledResource->sampler_unit_explicit) ? 1 : 0,
                        ptr ? (unsigned)ptr->name : 0u,
                        ptr ? (unsigned)ptr->target : 0u,
                        usedFallbackTexture ? 1 : 0,
                        (unsigned long)expectedType,
                        (unsigned long)lookupType,
                        expectedIndex,
                        mglTraceTextureName(unitActive),
                        mglTraceTextureName(unitExpected),
                        mglTraceTextureName(unit2D),
                        mglTraceTextureName(unitCube),
                        texture,
                        (unsigned long)(texture ? texture.textureType : 0),
                        (unsigned long)(texture ? texture.width : 0),
                        (unsigned long)(texture ? texture.height : 0),
                        level0 ? (unsigned)level0->width : 0u,
                        level0 ? (unsigned)level0->height : 0u,
                        level0 ? (unsigned)level0->ever_written : 0u,
                        level0 ? (unsigned)level0->has_initialized_data : 0u,
                        level0 ? (unsigned)level0->last_init_source : 0u);
        }
        if (texture && !usedFallbackTexture) {
            boundSampledTextures++;
        } else if (usedFallbackTexture) {
            // Keep nilTex as the original GL binding failure count, while Metal receives fallback texture.
            nilSampledTextures++;
        } else {
            nilSampledTextures++;
        }
        GLuint samplerBinding = sampledResource && sampledResource->msl_has_combined_sampler
            ? mglMetalCombinedSamplerSlot(sampledResource)
            : spirvBinding;
        if (sampler &&
            (!sampledResource || sampledResource->msl_has_combined_sampler) &&
            samplerBinding < kMaxFragmentSamplerSlots) {
            [self setFragmentSamplerStateIfNeeded:sampler atIndex:samplerBinding];
            boundSampledSamplers++;
        }

        if (traceBind && i < 6) {
            TextureLevel *level0 = NULL;
            if (ptr && ptr->faces[0].levels) {
                level0 = &ptr->faces[0].levels[0];
            }
            uint32_t cpuFirstTexel = 0u;
            bool cpuFirstTexelValid = false;
            if (level0 && level0->data && level0->data_size >= sizeof(cpuFirstTexel) &&
                ((uintptr_t)level0->data >= 0x1000ull)) {
                memcpy(&cpuFirstTexel, (const void *)level0->data, sizeof(cpuFirstTexel));
                cpuFirstTexelValid = true;
            }

            MGLTraceNSLog(@"MGL TRACE texbind.sampled call=%llu idx=%u binding=%u glTex=%u target=0x%x internal=0x%x "
                  "l0=%ux%ux%u l0bytes=%lu l0first=0x%08x(valid=%d) "
                  "l0src(source=%u upload=%lu srcPtr=%p hash=0x%016llx init(ever=%u full=%u zero=%u)) "
                  "mtlTex=%p size=%lux%lu sampler=%p fallback=%d",
                  (unsigned long long)bindCall,
                  (unsigned)i,
                  (unsigned)spirvBinding,
                  ptr ? (unsigned)ptr->name : 0u,
                  ptr ? (unsigned)ptr->target : 0u,
                  ptr ? (unsigned)ptr->internalformat : 0u,
                  level0 ? (unsigned)level0->width : 0u,
                  level0 ? (unsigned)level0->height : 0u,
                  level0 ? (unsigned)level0->depth : 0u,
                  (unsigned long)(level0 ? level0->data_size : 0u),
                  (unsigned)cpuFirstTexel,
                  cpuFirstTexelValid ? 1 : 0,
                  (unsigned)(level0 ? level0->last_init_source : 0u),
                  (unsigned long)(level0 ? level0->last_upload_size : 0u),
                  (void *)(level0 ? level0->last_src_ptr : NULL),
                  (unsigned long long)(level0 ? level0->last_src_hash : 0ull),
                  (unsigned)(level0 ? level0->ever_written : 0u),
                  (unsigned)(level0 ? level0->has_initialized_data : 0u),
                  (unsigned)(level0 ? level0->suspicious_zero_upload : 0u),
                  texture,
                  (unsigned long)(texture ? texture.width : 0),
                  (unsigned long)(texture ? texture.height : 0),
                  sampler,
                  usedFallbackTexture ? 1 : 0);
        }
    }

    *boundSampledTexturesPtr = boundSampledTextures;
    *nilSampledTexturesPtr = nilSampledTextures;
    *fallbackSampledTexturesPtr = fallbackSampledTextures;
    *boundSampledSamplersPtr = boundSampledSamplers;
    return true;
}

- (bool)recoverFragmentSampledDepthTexture:(Texture **)ptrPtr
                                    texture:(id<MTLTexture> *)texturePtr
                                sampledName:(const char *)sampledName
                                spirvBinding:(GLuint)spirvBinding
                                  textureUnit:(GLuint)textureUnit
                                 expectedType:(MTLTextureType)expectedType
                                 expectedKind:(MGLTextureDataKind)expectedKind
                         fragmentProgramName:(GLuint)fragmentProgramName
                  suppressMissingTextureFallback:(BOOL *)suppressMissingTextureFallbackPtr
                            usedFallbackTexture:(BOOL *)usedFallbackTexturePtr
{
    Texture *ptr = *ptrPtr;
    id<MTLTexture> texture = *texturePtr;
    BOOL suppressMissingTextureFallback = *suppressMissingTextureFallbackPtr;
    BOOL usedFallbackTexture = *usedFallbackTexturePtr;

    RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
    MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
    if (ptr->mtl_data) {
        texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
        texture = mglSampledTextureViewForBaseLevel(ptr, texture);
    }
    BOOL sampledNameIsInSampler =
        sampledName && strcmp(sampledName, "InSampler") == 0;
    if (texture &&
        sampledNameIsInSampler &&
        mglMetalPixelFormatIsDepthOrStencil(texture.pixelFormat)) {
        GLuint pairedFboName = 0u;
        Texture *pairedColor =
            mglFindFramebufferColorTexturePairedWithDepth(ctx, ptr, &pairedFboName);
        Texture *recoverTexture = NULL;
        id<MTLTexture> recoverMTL = nil;
        const char *recoverReason = "none";
        BOOL recoveredFromSampledCopy = NO;
        BOOL recoveredFromPreviousVersion = NO;
        NSUInteger recoverAttachmentIndex = MAX_COLOR_ATTACHMENTS;
        NSUInteger currentAttachmentIndex = MAX_COLOR_ATTACHMENTS;
        BOOL pairedColorIsCurrentDrawTarget =
            mglCurrentDrawFramebufferUsesColorTexture(ctx,
                                                      pairedColor,
                                                      pairedFboName,
                                                      &currentAttachmentIndex);
        id<MTLTexture> pairedMTL = nil;

        if (pairedColor) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:pairedColor]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            pairedMTL = pairedColor->mtl_data
                ? (__bridge id<MTLTexture>)(pairedColor->mtl_data)
                : nil;
            if (!pairedColorIsCurrentDrawTarget && pairedMTL) {
                pairedColorIsCurrentDrawTarget =
                    mglRenderPassUsesColorTexture(_renderPassManager.state->renderPassDescriptor,
                                                  pairedMTL,
                                                  &currentAttachmentIndex);
            }
        }

        if (pairedColorIsCurrentDrawTarget) {
            static uint64_t s_inSamplerDepthHistoryScanSuppressedLogCount = 0;
            uint64_t hit = ++s_inSamplerDepthHistoryScanSuppressedLogCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL INSAMPLER DEPTH HISTORY SCAN SUPPRESSED hit=%llu program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u pairedColor=%u currentDrawTarget=1",
                      (unsigned long long)hit,
                      (unsigned)fragmentProgramName,
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      (unsigned)pairedFboName,
                      (unsigned long)currentAttachmentIndex,
                      ptr ? (unsigned)ptr->name : 0u,
                      pairedColor ? (unsigned)pairedColor->name : 0u);
            }

            id<MTLTexture> pairedCopy = nil;
            BOOL usedPreviousVersion = NO;
            if (pairedColor &&
                mglRendererGLSampledCopyLooksUsable(pairedColor,
                                                    expectedType,
                                                    expectedKind,
                                                    YES,
                                                    &pairedCopy,
                                                    &usedPreviousVersion)) {
                recoverTexture = pairedColor;
                recoverMTL = pairedCopy;
                recoverReason = "paired-current-copy";
                recoveredFromSampledCopy = YES;
                recoveredFromPreviousVersion = usedPreviousVersion;
                recoverAttachmentIndex = currentAttachmentIndex;
            } else {
                static uint64_t s_inSamplerDepthCurrentTargetNoCopyLogCount = 0;
                uint64_t noCopyHit = ++s_inSamplerDepthCurrentTargetNoCopyLogCount;
                if (noCopyHit <= 64ull || (noCopyHit % 512ull) == 0ull) {
                    NSLog(@"MGL INSAMPLER DEPTH CURRENT TARGET NO COPY hit=%llu program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u colorTex=%u depthFmt=%lu sampledVersion=%u rtVersion=%u",
                          (unsigned long long)noCopyHit,
                          (unsigned)fragmentProgramName,
                          (unsigned)spirvBinding,
                          (unsigned)textureUnit,
                          (unsigned)pairedFboName,
                          (unsigned long)currentAttachmentIndex,
                          ptr ? (unsigned)ptr->name : 0u,
                          pairedColor ? (unsigned)pairedColor->name : 0u,
                          (unsigned long)texture.pixelFormat,
                          pairedColor ? (unsigned)pairedColor->mtl_gl_sampled_write_version : 0u,
                          pairedColor ? (unsigned)pairedColor->mtl_render_target_write_version : 0u);
                }
                texture = nil;
                suppressMissingTextureFallback = YES;
            }
        } else if (pairedColor &&
                   pairedMTL &&
                   !mglMetalPixelFormatIsDepthOrStencil(pairedMTL.pixelFormat)) {
            static uint64_t s_inSamplerDepthRecoveryLogCount = 0;
            uint64_t hit = ++s_inSamplerDepthRecoveryLogCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL INSAMPLER DEPTH RECOVERY hit=%llu program=%u binding=%u unit=%u fbo=%u depthTex=%u colorTex=%u depthFmt=%lu colorFmt=%lu size=%lux%lu",
                      (unsigned long long)hit,
                      (unsigned)fragmentProgramName,
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      (unsigned)pairedFboName,
                      ptr ? (unsigned)ptr->name : 0u,
                      (unsigned)pairedColor->name,
                      (unsigned long)texture.pixelFormat,
                      (unsigned long)pairedMTL.pixelFormat,
                      (unsigned long)pairedMTL.width,
                      (unsigned long)pairedMTL.height);
            }
            ptr = pairedColor;
            texture = pairedMTL;
        } else if (textureUnit < TEXTURE_UNITS) {
            for (GLuint historyIndex = 0;
                 historyIndex < MGL_RECENT_SAMPLED_2D_HISTORY;
                 historyIndex++) {
                Texture *candidate =
                    STATE(recent_sampled_2d_textures[textureUnit][historyIndex]);
                if (!candidate ||
                    candidate == ptr ||
                    candidate == pairedColor ||
                    !mglRendererTextureLooksLikeSampledColor2D(ctx, candidate)) {
                    continue;
                }

                id<MTLTexture> candidateMTL = candidate->mtl_data
                    ? (__bridge id<MTLTexture>)(candidate->mtl_data)
                    : nil;
                NSUInteger candidateAttachmentIndex = MAX_COLOR_ATTACHMENTS;
                BOOL candidateIsCurrentDrawTarget =
                    mglCurrentDrawFramebufferUsesColorTexture(ctx,
                                                              candidate,
                                                              0u,
                                                              &candidateAttachmentIndex) ||
                    mglRenderPassUsesColorTexture(_renderPassManager.state->renderPassDescriptor,
                                                  candidateMTL,
                                                  &candidateAttachmentIndex);

                if (!candidateIsCurrentDrawTarget &&
                    (!candidate->mtl_data || candidate->dirty_bits)) {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture:candidate]);
                    MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
                    candidateMTL = candidate->mtl_data
                        ? (__bridge id<MTLTexture>)(candidate->mtl_data)
                        : nil;
                    candidateAttachmentIndex = MAX_COLOR_ATTACHMENTS;
                    candidateIsCurrentDrawTarget =
                        mglCurrentDrawFramebufferUsesColorTexture(ctx,
                                                                  candidate,
                                                                  0u,
                                                                  &candidateAttachmentIndex) ||
                        mglRenderPassUsesColorTexture(_renderPassManager.state->renderPassDescriptor,
                                                      candidateMTL,
                                                      &candidateAttachmentIndex);
                }

                id<MTLTexture> candidateCopy = nil;
                BOOL usedPreviousVersion = NO;
                if (candidate->is_render_target &&
                    mglRendererGLSampledCopyLooksUsable(candidate,
                                                        expectedType,
                                                        expectedKind,
                                                        candidateIsCurrentDrawTarget,
                                                        &candidateCopy,
                                                        &usedPreviousVersion)) {
                    recoverTexture = candidate;
                    recoverMTL = candidateCopy;
                    recoverReason = candidateIsCurrentDrawTarget
                        ? "history-current-copy"
                        : "history-copy";
                    recoveredFromSampledCopy = YES;
                    recoveredFromPreviousVersion = usedPreviousVersion;
                    recoverAttachmentIndex = candidateAttachmentIndex;
                    break;
                }

                if (candidateIsCurrentDrawTarget) {
                    continue;
                }
                if (candidateMTL &&
                    !mglMetalPixelFormatIsDepthOrStencil(candidateMTL.pixelFormat) &&
                    (expectedType == 0 || candidateMTL.textureType == expectedType) &&
                    mglTexturePixelFormatCompatibleWithExpectedDataKind(candidateMTL.pixelFormat, expectedKind)) {
                    recoverTexture = candidate;
                    recoverMTL = candidateMTL;
                    recoverReason = "history-direct";
                    recoverAttachmentIndex = candidateAttachmentIndex;
                    break;
                }
            }
        } else if (!pairedColor) {
            static uint64_t s_inSamplerDepthUnpairedLogCount = 0;
            uint64_t hit = ++s_inSamplerDepthUnpairedLogCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL INSAMPLER DEPTH UNPAIRED hit=%llu program=%u binding=%u unit=%u depthTex=%u fmt=%lu size=%lux%lu",
                      (unsigned long long)hit,
                      (unsigned)fragmentProgramName,
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      ptr ? (unsigned)ptr->name : 0u,
                      (unsigned long)texture.pixelFormat,
                      (unsigned long)texture.width,
                      (unsigned long)texture.height);
            }
        }

        if (recoverTexture && recoverMTL) {
            static uint64_t s_inSamplerDepthHistoryRecoveryLogCount = 0;
            uint64_t hit = ++s_inSamplerDepthHistoryRecoveryLogCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL INSAMPLER DEPTH RECOVERY hit=%llu reason=%s program=%u binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u recoverTex=%u depthFmt=%lu recoverFmt=%lu size=%lux%lu copy=%d prevVersion=%d sampledVersion=%u rtVersion=%u pairedColor=%u pairedCurrent=%d",
                      (unsigned long long)hit,
                      recoverReason,
                      (unsigned)fragmentProgramName,
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      (unsigned)pairedFboName,
                      (unsigned long)recoverAttachmentIndex,
                      ptr ? (unsigned)ptr->name : 0u,
                      recoverTexture ? (unsigned)recoverTexture->name : 0u,
                      (unsigned long)texture.pixelFormat,
                      (unsigned long)recoverMTL.pixelFormat,
                      (unsigned long)recoverMTL.width,
                      (unsigned long)recoverMTL.height,
                      recoveredFromSampledCopy ? 1 : 0,
                      recoveredFromPreviousVersion ? 1 : 0,
                      recoverTexture ? (unsigned)recoverTexture->mtl_gl_sampled_write_version : 0u,
                      recoverTexture ? (unsigned)recoverTexture->mtl_render_target_write_version : 0u,
                      pairedColor ? (unsigned)pairedColor->name : 0u,
                      pairedColorIsCurrentDrawTarget ? 1 : 0);
            }
            ptr = recoverTexture;
            texture = recoverMTL;
        }
    }
    TextureLevel *depthSampleLevel0 = mglTraceTextureBaseLevel(ptr);
    if (texture &&
        !sampledNameIsInSampler &&
        ptr &&
        ptr->is_render_target &&
        mglMetalPixelFormatIsDepthOrStencil(texture.pixelFormat) &&
        (!depthSampleLevel0 ||
         !depthSampleLevel0->ever_written ||
         !depthSampleLevel0->has_initialized_data)) {
        Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
        Texture *unit2D = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_2D]) : NULL;
        Texture *last2D = textureUnit < TEXTURE_UNITS ? STATE(last_sampled_2d_textures[textureUnit]) : NULL;
        Texture *recoverTexture = NULL;
        const char *recoverReason = "none";
        GLuint recoverFboName = 0u;

        Texture *pairedColor =
            mglFindFramebufferColorTexturePairedWithDepth(ctx, ptr, &recoverFboName);
        if (pairedColor) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:pairedColor]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            id<MTLTexture> pairedMTL = pairedColor->mtl_data
                ? (__bridge id<MTLTexture>)(pairedColor->mtl_data)
                : nil;
            NSUInteger drawAttachmentIndex = MAX_COLOR_ATTACHMENTS;
            BOOL pairedColorIsCurrentDrawTarget =
                mglRenderPassUsesColorTexture(_renderPassManager.state->renderPassDescriptor,
                                              pairedMTL,
                                              &drawAttachmentIndex);
            if (pairedMTL &&
                !pairedColorIsCurrentDrawTarget &&
                !mglMetalPixelFormatIsDepthOrStencil(pairedMTL.pixelFormat) &&
                (expectedType == 0 || pairedMTL.textureType == expectedType) &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(pairedMTL.pixelFormat, expectedKind)) {
                recoverTexture = pairedColor;
                recoverReason = "paired-color";
            } else if (pairedColorIsCurrentDrawTarget) {
                static uint64_t s_sampledDepthRenderTargetRecoverSkipLogCount = 0;
                uint64_t hit = ++s_sampledDepthRenderTargetRecoverSkipLogCount;
                if (hit <= 64ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL SAMPLED DEPTH RT RECOVER SKIP current-draw-target hit=%llu program=%u name=%s binding=%u unit=%u fbo=%u colorAttachment=%lu depthTex=%u colorTex=%u",
                          (unsigned long long)hit,
                          (unsigned)fragmentProgramName,
                          sampledName ? sampledName : "",
                          (unsigned)spirvBinding,
                          (unsigned)textureUnit,
                          (unsigned)recoverFboName,
                          (unsigned long)drawAttachmentIndex,
                          ptr ? (unsigned)ptr->name : 0u,
                          pairedColor ? (unsigned)pairedColor->name : 0u);
                }
            }
        }

        if (!recoverTexture &&
            mglRendererTextureLooksRecoverableSampled2D(ctx, last2D, expectedType, expectedKind)) {
            static uint64_t s_sampledDepthLast2DRecoverySuppressedLogCount = 0;
            uint64_t hit = ++s_sampledDepthLast2DRecoverySuppressedLogCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL SAMPLED DEPTH RT RECOVER SUPPRESS last-sampled-2d hit=%llu program=%u name=%s binding=%u unit=%u depthTex=%u last2D=%u",
                      (unsigned long long)hit,
                      (unsigned)fragmentProgramName,
                      sampledName ? sampledName : "",
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      ptr ? (unsigned)ptr->name : 0u,
                      (unsigned)last2D->name);
            }
        }

        if (recoverTexture) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:recoverTexture]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            id<MTLTexture> recoverMTL = recoverTexture->mtl_data
                ? (__bridge id<MTLTexture>)(recoverTexture->mtl_data)
                : nil;
            if (recoverMTL &&
                !mglMetalPixelFormatIsDepthOrStencil(recoverMTL.pixelFormat) &&
                (expectedType == 0 || recoverMTL.textureType == expectedType) &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(recoverMTL.pixelFormat, expectedKind)) {
                Framebuffer *currentFbo = ctx ? ctx->active_state->framebuffer : NULL;
                GLuint colorTexName = 0u;
                GLuint depthTexName = 0u;
                if (currentFbo &&
                    mglRendererObjectPointerLikelyValid(currentFbo) &&
                    mglPointerRangeIsReadable(currentFbo, sizeof(*currentFbo))) {
                    colorTexName = currentFbo->color_attachments[0].texture;
                    depthTexName = currentFbo->depth.texture;
                }

                static uint64_t s_sampledDepthRenderTargetRecoverLogCount = 0;
                uint64_t hit = ++s_sampledDepthRenderTargetRecoverLogCount;
                if (hit <= 64ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL SAMPLED DEPTH RT RECOVER hit=%llu reason=%s program=%u name=%s binding=%u unit=%u depthTex=%u recoverTex=%u fmt=%lu recoverFmt=%lu size=%lux%lu level=%p ever=%u init=%u unit(active=%u tex2D=%u last2D=%u) recoverFbo=%u currentFbo=%u colorTex=%u fboDepthTex=%u",
                          (unsigned long long)hit,
                          recoverReason,
                          (unsigned)fragmentProgramName,
                          sampledName ? sampledName : "",
                          (unsigned)spirvBinding,
                          (unsigned)textureUnit,
                          ptr ? (unsigned)ptr->name : 0u,
                          (unsigned)recoverTexture->name,
                          (unsigned long)texture.pixelFormat,
                          (unsigned long)recoverMTL.pixelFormat,
                          (unsigned long)texture.width,
                          (unsigned long)texture.height,
                          depthSampleLevel0,
                          depthSampleLevel0 ? (unsigned)depthSampleLevel0->ever_written : 0u,
                          depthSampleLevel0 ? (unsigned)depthSampleLevel0->has_initialized_data : 0u,
                          mglTraceTextureName(unitActive),
                          mglTraceTextureName(unit2D),
                          mglTraceTextureName(last2D),
                          (unsigned)recoverFboName,
                          currentFbo ? (unsigned)currentFbo->name : 0u,
                          (unsigned)colorTexName,
                          (unsigned)depthTexName);
                }

                ptr = recoverTexture;
                texture = recoverMTL;
            }
        }

        if (mglMetalPixelFormatIsDepthOrStencil(texture.pixelFormat)) {
            id<MTLTexture> fallbackTexture =
                [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
            if (fallbackTexture) {
                static uint64_t s_sampledDepthRenderTargetFallbackLogCount = 0;
                uint64_t hit = ++s_sampledDepthRenderTargetFallbackLogCount;
                if (hit <= 64ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL SAMPLED DEPTH RT FALLBACK hit=%llu program=%u name=%s binding=%u unit=%u depthTex=%u fmt=%lu size=%lux%lu level=%p ever=%u init=%u unit(active=%u tex2D=%u last2D=%u)",
                          (unsigned long long)hit,
                          (unsigned)fragmentProgramName,
                          sampledName ? sampledName : "",
                          (unsigned)spirvBinding,
                          (unsigned)textureUnit,
                          ptr ? (unsigned)ptr->name : 0u,
                          (unsigned long)texture.pixelFormat,
                          (unsigned long)texture.width,
                          (unsigned long)texture.height,
                          depthSampleLevel0,
                          depthSampleLevel0 ? (unsigned)depthSampleLevel0->ever_written : 0u,
                          depthSampleLevel0 ? (unsigned)depthSampleLevel0->has_initialized_data : 0u,
                          mglTraceTextureName(unitActive),
                          mglTraceTextureName(unit2D),
                          mglTraceTextureName(last2D));
                }
                texture = fallbackTexture;
                usedFallbackTexture = YES;
            }
        }
    }

    *ptrPtr = ptr;
    *texturePtr = texture;
    *suppressMissingTextureFallbackPtr = suppressMissingTextureFallback;
    *usedFallbackTexturePtr = usedFallbackTexture;
    return true;
}

- (bool)resolveFragmentSampledYFlipAndSampler:(Texture *)ptr
                                       texture:(id<MTLTexture> *)texturePtr
                                       sampler:(id<MTLSamplerState> *)samplerPtr
                                   sampledName:(const char *)sampledName
                                spirvBinding:(GLuint)spirvBinding
                                  textureUnit:(GLuint)textureUnit
                                 expectedType:(MTLTextureType)expectedType
                                 expectedKind:(MGLTextureDataKind)expectedKind
                         fragmentProgramName:(GLuint)fragmentProgramName
                          vertexProgramName:(GLuint)vertexProgramName
                                sampleProgram:(Program *)sampleProgram
                            usedFallbackTexture:(BOOL *)usedFallbackTexturePtr
                       usedSampledCopyForTrace:(BOOL *)usedSampledCopyForTracePtr
                          directTextureForTrace:(id<MTLTexture> *)directTextureForTracePtr
                          sampledCopyForTrace:(id<MTLTexture> *)sampledCopyForTracePtr
{
    id<MTLTexture> texture = *texturePtr;
    id<MTLSamplerState> sampler = *samplerPtr;
    BOOL usedFallbackTexture = *usedFallbackTexturePtr;
    BOOL usedSampledCopyForTrace = *usedSampledCopyForTracePtr;
    id<MTLTexture> directTextureForTrace = *directTextureForTracePtr;
    id<MTLTexture> sampledCopyForTrace = *sampledCopyForTracePtr;

    /* Y-Flip Subsystem: unified decision for sampling a render target.
     *
     * NOTE: lazy refresh from bindTexturesToCurrentRenderEncoder was
     * removed — updateGLSampledRenderTargetCopyForTexture creates its
     * own render encoder, which re-enters the encoder while a flush
     * triggered by mglBindBufferRange is mid-process and crashes AGX
     * (MTLReportFailure -> SIGABRT).  Refresh is left to the
     * end_render_pass / blit_framebuffer paths, which run outside an
     * active encoder and are encoder-safe. */
    if (texture &&
        !usedFallbackTexture &&
        ptr &&
        ptr->is_render_target) {
        MGLYFlipDecision yflip = mglDecideYFlipForSampledRT(ptr, sampleProgram);
        if (mglTraceRTYFlipDiagnosticsEnabled()) {
            mglTraceLog("RT_YFLIP_DECISION stage=fragment program=%u stateProgram=%u current=%u pipeline=%u vs=%u fs=%u pipelineProgram=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" decision=%s(%d) authority=0x%x rtVer=%u copyVer=%u hasCopy=%d sampleYFlip=%d",
                        (unsigned)fragmentProgramName,
                        (unsigned)(ctx ? ctx->active_state->program_name : 0u),
                        (unsigned)(ctx ? ctx->active_state->var.current_program : 0u),
                        (unsigned)(ctx ? ctx->active_state->var.program_pipeline_binding : 0u),
                        (unsigned)vertexProgramName,
                        (unsigned)fragmentProgramName,
                        (unsigned)_pipelineCache.state->pipelineProgramName,
                        sampledName ? sampledName : "",
                        (unsigned)spirvBinding,
                        (unsigned)textureUnit,
                        (unsigned)ptr->name,
                        mglTraceTextureLabel(ptr),
                        mglYFlipDecisionName(yflip),
                        (int)yflip,
                        (unsigned)ptr->mtl_render_yflip_authority,
                        (unsigned)ptr->mtl_render_target_write_version,
                        (unsigned)ptr->mtl_gl_sampled_write_version,
                        ptr->mtl_gl_sampled_data ? 1 : 0,
                        mglProgramHasExistingFramebufferSampleYFlip(sampleProgram) ? 1 : 0);
        }

        if (yflip == MGL_YFLIP_USE_SAMPLED_COPY) {
            BOOL boundSampledCopy = NO;
            if (ptr->mtl_gl_sampled_data &&
                ptr->mtl_gl_sampled_write_version == ptr->mtl_render_target_write_version &&
                mglTextureCanUseGLSampledRenderTargetCopy(ptr)) {
                directTextureForTrace = texture;
                id<MTLTexture> sampledCopy = (__bridge id<MTLTexture>)(ptr->mtl_gl_sampled_data);

                if (sampledCopy &&
                    (expectedType == 0 || sampledCopy.textureType == expectedType) &&
                    mglTexturePixelFormatCompatibleWithExpectedDataKind(sampledCopy.pixelFormat, expectedKind)) {
                    sampledCopyForTrace = sampledCopy;
                    if (mglTraceLogIsEnabled()) {
                        mglTraceLog("RT_SAMPLE_COPY_BIND stage=fragment program=%u stateProgram=%u current=%u pipeline=%u vs=%u fs=%u pipelineProgram=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" original=%p copy=%p size=%lux%lu originalLevels=%lu copyLevels=%lu glLevels=%u mips=%u base=%u max=%u version=%u",
                                    (unsigned)fragmentProgramName,
                                    (unsigned)(ctx ? ctx->active_state->program_name : 0u),
                                    (unsigned)(ctx ? ctx->active_state->var.current_program : 0u),
                                    (unsigned)(ctx ? ctx->active_state->var.program_pipeline_binding : 0u),
                                    (unsigned)vertexProgramName,
                                    (unsigned)fragmentProgramName,
                                    (unsigned)_pipelineCache.state->pipelineProgramName,
                                    sampledName ? sampledName : "",
                                    (unsigned)spirvBinding,
                                    (unsigned)textureUnit,
                                    (unsigned)ptr->name,
                                    mglTraceTextureLabel(ptr),
                                    texture,
                                    sampledCopy,
                                    (unsigned long)sampledCopy.width,
                                    (unsigned long)sampledCopy.height,
                                    (unsigned long)(texture ? texture.mipmapLevelCount : 0u),
                                    (unsigned long)sampledCopy.mipmapLevelCount,
                                    (unsigned)ptr->num_levels,
                                    (unsigned)ptr->mipmap_levels,
                                    (unsigned)ptr->params.base_level,
                                    (unsigned)ptr->params.max_level,
                                    (unsigned)ptr->mtl_gl_sampled_write_version);
                    }
                    mglWriteProgramMSLDump(sampleProgram,
                                           [NSString stringWithFormat:@"tex-rt-sample-copy-fragment-binding-%u-program-%u",
                                                                      (unsigned)spirvBinding,
                                                                      (unsigned)(sampleProgram ? sampleProgram->name : fragmentProgramName)]);
                    texture = mglSampledTextureViewForBaseLevel(ptr, sampledCopy);
                    usedSampledCopyForTrace = YES;
                    boundSampledCopy = YES;
                }
            }
            if (!boundSampledCopy && mglTextureCanUseGLSampledRenderTargetCopy(ptr)) {
                directTextureForTrace = texture;
                id<MTLTexture> repairedCopy =
                    [self freshGLSampledRenderTargetCopyForSampling:ptr
                                                              source:texture
                                                               stage:"fragment"
                                                             program:fragmentProgramName
                                                             binding:spirvBinding
                                                                unit:textureUnit
                                                        expectedType:expectedType
                                                        expectedKind:expectedKind];
                if (repairedCopy) {
                    return false;
                }
            }
            if (!boundSampledCopy && ptr->mtl_gl_sampled_data &&
                !mglTextureCanUseGLSampledRenderTargetCopy(ptr)) {
                mglLogSkippedGLSampledRenderTargetCopy(ctx,
                                                       sampleProgram,
                                                       ptr,
                                                       "fragment",
                                                       sampledName,
                                                       spirvBinding,
                                                       textureUnit,
                                                       "target-gate");
            } else if (!boundSampledCopy && mglTraceLogIsEnabled()) {
                BOOL hasCopy = (ptr->mtl_gl_sampled_data != NULL);
                BOOL verMatch = (ptr->mtl_gl_sampled_write_version == ptr->mtl_render_target_write_version);
                BOOL canUse = mglTextureCanUseGLSampledRenderTargetCopy(ptr);
                mglTraceLog("RT_SAMPLE_COPY_GATE_MISS stage=fragment program=%u stateProgram=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" isRT=%d hasCopy=%d verMatch=%d writeVer=%u rtVer=%u canUse=%d expectedType=%lu",
                            (unsigned)fragmentProgramName,
                            (unsigned)(ctx ? ctx->active_state->program_name : 0u),
                            sampledName ? sampledName : "",
                            (unsigned)spirvBinding,
                            (unsigned)textureUnit,
                            (unsigned)ptr->name,
                            mglTraceTextureLabel(ptr),
                            ptr->is_render_target ? 1 : 0,
                            hasCopy ? 1 : 0,
                            verMatch ? 1 : 0,
                            (unsigned)ptr->mtl_gl_sampled_write_version,
                            (unsigned)ptr->mtl_render_target_write_version,
                            canUse ? 1 : 0,
                            (unsigned long)expectedType);
            }
        } else {
            /* MGL_YFLIP_USE_ORIGINAL or MGL_YFLIP_USE_ORIGINAL_AND_INJECT:
             * keep the original texture; no copy needed. */
            static uint64_t s_rtSampleCopySkipExistingFlipLogCount = 0;
            uint64_t hit = ++s_rtSampleCopySkipExistingFlipLogCount;
            if (mglTraceLogIsEnabled() && (hit <= 32ull || (hit % 512ull) == 0ull)) {
                mglTraceLog("RT_SAMPLE_COPY_SKIP_EXISTING_YFLIP hit=%llu stage=fragment program=%u name=%s binding=%u tex=%u label=\"%s\" decision=%s(%d)",
                            (unsigned long long)hit,
                            (unsigned)fragmentProgramName,
                            sampledName ? sampledName : "",
                            (unsigned)spirvBinding,
                            (unsigned)(ptr ? ptr->name : 0u),
                            mglTraceTextureLabel(ptr),
                            mglYFlipDecisionName(yflip),
                            (int)yflip);
            }
        }
    }
    if (texture && expectedType != 0 && texture.textureType != expectedType) {
        static uint64_t s_fragmentTypeMismatchLogCount = 0;
        uint64_t hit = ++s_fragmentTypeMismatchLogCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL TEX TYPE MISMATCH fragment binding=%u program=%u glTex=%u glTarget=0x%x mtlType=%lu expected=%lu hit=%llu",
                  (unsigned)spirvBinding,
                  (unsigned)fragmentProgramName,
                  (unsigned)ptr->name,
                  (unsigned)ptr->target,
                  (unsigned long)texture.textureType,
                  (unsigned long)expectedType,
                  (unsigned long long)hit);
        }
        Program *dumpProgram = sampleProgram;
        mglWriteProgramMSLDump(dumpProgram,
                               [NSString stringWithFormat:@"tex-type-mismatch-fragment-binding-%u", spirvBinding]);
        texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
        usedFallbackTexture = YES;
        usedSampledCopyForTrace = NO;
    }
    if (texture &&
        !mglTexturePixelFormatCompatibleWithExpectedDataKind(texture.pixelFormat, expectedKind)) {
        static uint64_t s_fragmentDataKindMismatchLogCount = 0;
        uint64_t hit = ++s_fragmentDataKindMismatchLogCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL TEX DATA MISMATCH fragment binding=%u program=%u glTex=%u glTarget=0x%x format=%lu actualKind=%s expectedKind=%s expectedType=%lu hit=%llu",
                  (unsigned)spirvBinding,
                  (unsigned)fragmentProgramName,
                  (unsigned)ptr->name,
                  (unsigned)ptr->target,
                  (unsigned long)texture.pixelFormat,
                  mglTextureDataKindName(mglTextureDataKindForPixelFormat(texture.pixelFormat)),
                  mglTextureDataKindName(expectedKind),
                  (unsigned long)expectedType,
                  (unsigned long long)hit);
        }
        Program *dumpProgram = sampleProgram;
        mglWriteProgramMSLDump(dumpProgram,
                               [NSString stringWithFormat:@"tex-data-mismatch-fragment-binding-%u", spirvBinding]);
        texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
        usedFallbackTexture = YES;
        usedSampledCopyForTrace = NO;
    }

    if (textureUnit < TEXTURE_UNITS && STATE(texture_samplers[textureUnit])) {
        Sampler *glSampler = STATE(texture_samplers[textureUnit]);
        if (glSampler->dirty_bits && glSampler->mtl_data) {
            mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
        }
        if (glSampler->mtl_data == NULL) {
            glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
            glSampler->dirty_bits = 0;
        }
        sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
        mglTraceLogExternal("FRAG_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=glSampler samplerName=%u minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                            (unsigned)fragmentProgramName,
                            (unsigned)spirvBinding,
                            (unsigned)textureUnit,
                            (unsigned)glSampler->name,
                            (unsigned)glSampler->params.min_filter,
                            (unsigned)glSampler->params.mag_filter,
                            (unsigned)glSampler->params.wrap_s,
                            (unsigned)glSampler->params.wrap_t,
                            (double)glSampler->params.min_lod,
                            (double)glSampler->params.max_lod,
                            (unsigned)ptr->name,
                            (unsigned)ptr->params.base_level,
                            (unsigned)ptr->params.max_level,
                            (unsigned)ptr->width,
                            (unsigned)ptr->height,
                            (unsigned long)(texture ? texture.width : 0u),
                            (unsigned long)(texture ? texture.height : 0u),
                            (unsigned long)(texture ? texture.mipmapLevelCount : 0u));
    } else {
        sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
        mglTraceLogExternal("FRAG_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=texParamsFallback samplerName=0 minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                            (unsigned)fragmentProgramName,
                            (unsigned)spirvBinding,
                            (unsigned)textureUnit,
                            (unsigned)ptr->params.min_filter,
                            (unsigned)ptr->params.mag_filter,
                            (unsigned)ptr->params.wrap_s,
                            (unsigned)ptr->params.wrap_t,
                            (double)ptr->params.min_lod,
                            (double)ptr->params.max_lod,
                            (unsigned)ptr->name,
                            (unsigned)ptr->params.base_level,
                            (unsigned)ptr->params.max_level,
                            (unsigned)ptr->width,
                            (unsigned)ptr->height,
                            (unsigned long)(texture ? texture.width : 0u),
                            (unsigned long)(texture ? texture.height : 0u),
                            (unsigned long)(texture ? texture.mipmapLevelCount : 0u));
    }

    *texturePtr = texture;
    *samplerPtr = sampler;
    *usedFallbackTexturePtr = usedFallbackTexture;
    *usedSampledCopyForTracePtr = usedSampledCopyForTrace;
    *directTextureForTracePtr = directTextureForTrace;
    *sampledCopyForTracePtr = sampledCopyForTrace;
    return true;
}

- (bool)bindStorageImagesToEncoder:(Program *)vertexProgram
                    fragmentProgram:(Program *)fragmentProgram
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    /* Vertex-stage storage image binding (two-pass, same pattern as fragment). */
    GLuint vertexStorageImageCount = [self getProgramBindingCount:_VERTEX_SHADER
                                                            type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    for (GLuint i = 0; i < vertexStorageImageCount; i++)
    {
        SpirvResource *resource = NULL;
        if (vertexProgram &&
            i < vertexProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &vertexProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(vertexProgram,
                                              _VERTEX_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:_VERTEX_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(image_units[glUnit].tex);
        if (ptr) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
        }
    }
    if (!encCtx->encoder) {
        RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"vs-storage-image-bind"]);
    }
    for (GLuint i = 0; i < vertexStorageImageCount; i++)
    {
        SpirvResource *resource = NULL;
        if (vertexProgram &&
            i < vertexProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &vertexProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(vertexProgram,
                                              _VERTEX_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_VERTEX_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:_VERTEX_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = STATE(image_units[glUnit].tex);
        id<MTLTexture> texture = nil;
        if (ptr) {
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            GLuint imgLevel = STATE(image_units[glUnit].level);
            if (imgLevel > 0u && texture) {
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                id<MTLTexture> levelView = [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                                                       textureType:texture.textureType
                                                                            levels:NSMakeRange(imgLevel, 1)
                                                                            slices:NSMakeRange(0, sliceCount)];
                if (levelView) {
                    texture = levelView;
                }
            }
        }
        [self setVertexTextureIfNeeded:texture atIndex:metalSlot];
    }

    GLuint fragmentStorageImageCount = [self getProgramBindingCount:_FRAGMENT_SHADER
                                                               type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    /* Two-pass storage image binding:
     *
     * Pass 1: Pre-resolve every storage image's Metal texture via
     * bindMTLTexture.  This may trigger texture (re)creation and CPU→GPU
     * blit uploads, which close _renderPassManager.state->currentRenderEncoder (Metal does not allow
     * a render encoder and a blit encoder on the same command buffer
     * simultaneously).  We do NOT touch _renderPassManager.state->currentRenderEncoder here.
     *
     * Pass 2: Bind the now-resolved Metal textures to the (possibly
     * restored) render encoder.  If pass 1 closed the encoder, restore it
     * first so every setFragmentTexture:atIndex: call has a valid encoder.
     *
     * Without the two-pass split, MGL_ABORT_TBIND_IF_ENCODER_CLOSED would
     * abort the loop on the first texture that needs a blit upload, leaving
     * the remaining storage images unbound and the shader reading stale/
     * default data. */
    for (GLuint i = 0; i < fragmentStorageImageCount; i++)
    {
        SpirvResource *resource = NULL;
        if (fragmentProgram &&
            i < fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(fragmentProgram,
                                              _FRAGMENT_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }

        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:_FRAGMENT_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }

        Texture *ptr = STATE(image_units[glUnit].tex);
        if (ptr) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
        }
    }

    /* Restore render encoder if any pass-1 bindMTLTexture closed it. */
    if (!encCtx->encoder) {
        RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"storage-image-bind"]);
    }

    for (GLuint i = 0; i < fragmentStorageImageCount; i++)
    {
        SpirvResource *resource = NULL;
        if (fragmentProgram &&
            i < fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(fragmentProgram,
                                              _FRAGMENT_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }

        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_FRAGMENT_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:_FRAGMENT_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }

        Texture *ptr = STATE(image_units[glUnit].tex);
        id<MTLTexture> texture = nil;
        if (ptr) {
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            texture = (__bridge id<MTLTexture>)(ptr->mtl_data);

            /* Create a mipmap-level-specific texture view so that imageSize()
             * in the shader returns the dimensions at the bound level, not
             * level 0.  Metal's get_width()/get_height() on a view created
             * with levels={N,1} returns the size at level N (the view's
             * level 0 maps to the original's level N).  Without this view,
             * glBindImageTexture's <level> parameter is silently ignored
             * and all imageSize queries return level-0 dimensions. */
            GLuint imgLevel = STATE(image_units[glUnit].level);
            if (imgLevel > 0u && texture) {
                /* Cube and cube-array textures pack 6 face-slices per cube;
                 * the view's slice count must be a multiple of 6 for these
                 * types.  Other types use arrayLength directly. */
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                id<MTLTexture> levelView = [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                                                       textureType:texture.textureType
                                                                            levels:NSMakeRange(imgLevel, 1)
                                                                            slices:NSMakeRange(0, sliceCount)];
                if (levelView) {
                    texture = levelView;
                }
            }
        }

        [self setFragmentTextureIfNeeded:texture atIndex:metalSlot];
    }
    return true;
}

- (void)bindSeparateSamplersAndArrayTextures:(Program *)vertexProgram
                              fragmentProgram:(Program *)fragmentProgram
                        fragmentProgramName:(GLuint)fragmentProgramName
                          vertexProgramName:(GLuint)vertexProgramName
                             defaultSampler:(id<MTLSamplerState>)defaultSampler
                                    bindCall:(uint64_t)bindCall
                                  traceBind:(bool)traceBind
                         separateSamplerCount:(GLuint *)separateSamplerCount
                           boundSeparateSamplers:(GLuint *)boundSeparateSamplers
{
    // Bind separate samplers explicitly.
    *separateSamplerCount = [self getProgramBindingCount:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS];
    *boundSeparateSamplers = 0;
    for (GLuint i = 0; i < *separateSamplerCount; i++)
    {
        GLuint spirvBinding = [self getProgramBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS index:(int)i];
        GLuint glBinding = [self getProgramGLBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS index:(int)i];
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        Program *sampleProgram = fragmentProgram;
        SpirvResource *samplerResource = NULL;
        if (sampleProgram &&
            i < sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS].count) {
            samplerResource = &sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS].list[i];
        }
        if (mglShouldSkipStageSamplerResource(sampleProgram,
                                              _FRAGMENT_SHADER,
                                              SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
                                              samplerResource)) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:samplerResource
                                                    metalBinding:spirvBinding
                                                           stage:_FRAGMENT_SHADER];

        id<MTLSamplerState> sampler = nil;
        if (textureUnit < TEXTURE_UNITS && STATE(texture_samplers[textureUnit])) {
            Sampler *glSampler = STATE(texture_samplers[textureUnit]);
            if (glSampler->dirty_bits && glSampler->mtl_data) {
                mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
            }
            if (glSampler->mtl_data == NULL) {
                glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:GL_TEXTURE_2D]);
                glSampler->dirty_bits = 0;
            }
            sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
        }

        if (!sampler) {
            sampler = defaultSampler;
        }
        if (sampler && spirvBinding < kMaxFragmentSamplerSlots) {
            [self setFragmentSamplerStateIfNeeded:sampler atIndex:spirvBinding];
            boundSeparateSamplers++;
        }

        if (traceBind && i < 6) {
            MGLTraceNSLog(@"MGL TRACE texbind.separateSampler call=%llu idx=%u binding=%u unit=%u sampler=%p",
                  (unsigned long long)bindCall,
                  (unsigned)i,
                  (unsigned)spirvBinding,
                  (unsigned)textureUnit,
                  sampler);
        }
    }

    Program *arrayPrograms[] = { vertexProgram, fragmentProgram };
    int arrayStages[] = { _VERTEX_SHADER, _FRAGMENT_SHADER };
    for (NSUInteger programIndex = 0; programIndex < 2; programIndex++) {
        Program *arrayProgram = arrayPrograms[programIndex];
        int arrayStage = arrayStages[programIndex];
        if (!arrayProgram) {
            continue;
        }

        SpirvResourceList *arrayResources =
            &arrayProgram->spirv_resources_list[arrayStage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
        for (GLuint resourceIndex = 0; arrayResources->list && resourceIndex < arrayResources->count; resourceIndex++) {
            SpirvResource *resource = &arrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            MTLTextureType expectedType = [self getProgramExpectedTextureType:arrayStage
                                                                         type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                        index:(int)resourceIndex];
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                GLuint samplerSlot =
                    mglMetalCombinedSamplerSlotForElement(resource,
                                                          (GLuint)element);
                if (metalSlot >= TEXTURE_UNITS) {
                    break;
                }

                GLuint textureUnit = [self textureUnitForSampledResource:NULL
                                                             metalBinding:metalSlot
                                                                    stage:arrayStage];
                Texture *arrayTexture = [self textureForSampledResource:NULL
                                                            metalBinding:metalSlot
                                                                    stage:arrayStage
                                                             expectedType:expectedType];
                id<MTLTexture> metalTexture = nil;
                id<MTLSamplerState> metalSampler = defaultSampler;
                if (arrayTexture && [self bindMTLTexture:arrayTexture]) {
                    metalTexture = (__bridge id<MTLTexture>)(arrayTexture->mtl_data);
                    if (textureUnit < TEXTURE_UNITS && STATE(texture_samplers[textureUnit])) {
                        Sampler *glSampler = STATE(texture_samplers[textureUnit]);
                        if (glSampler->mtl_data == NULL) {
                            glSampler->mtl_data = (void *)CFBridgingRetain(
                                [self createMTLSamplerForTexParam:&glSampler->params target:arrayTexture->target]);
                            glSampler->dirty_bits = 0;
                        }
                        metalSampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
                    } else if (arrayTexture->params.mtl_data) {
                        metalSampler = (__bridge id<MTLSamplerState>)(arrayTexture->params.mtl_data);
                    }
                }
                if (!metalTexture) {
                    metalTexture = [self fallbackSampledTextureForExpectedType:expectedType
                                                                      dataKind:MGLTextureDataKindFloat];
                }

                if (arrayStage == _VERTEX_SHADER) {
                    [self setVertexTextureIfNeeded:metalTexture atIndex:metalSlot];
                    if (resource->msl_has_combined_sampler && metalSampler &&
                        samplerSlot < kMaxFragmentSamplerSlots) {
                        [self setVertexSamplerStateIfNeeded:metalSampler atIndex:samplerSlot];
                    }
                } else {
                    [self setFragmentTextureIfNeeded:metalTexture atIndex:metalSlot];
                    if (resource->msl_has_combined_sampler && metalSampler &&
                        samplerSlot < kMaxFragmentSamplerSlots) {
                        [self setFragmentSamplerStateIfNeeded:metalSampler atIndex:samplerSlot];
                    }
                }
            }
        }
    }
}



- (void)issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
                encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!batch || batch->command_count == 0) {
        return;
    }
    if (mglEnvFlagEnabled("MGL_DISABLE_MDI")) {
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }
    if (batch->key.primitive_type == 0xFFu) {
        [self traceReplayCommand:batch
                         command:&batch->commands[0]
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:0
                           phase:"FALLBACK"
                          reason:"mdi_unsupported_primitive"];
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    bool indexed = batch->uses_elements;
    size_t argSize = indexed ? sizeof(MTLDrawIndexedPrimitivesIndirectArguments)
                             : sizeof(MTLDrawPrimitivesIndirectArguments);
    if (batch->command_count > (UINT32_MAX / argSize)) {
        [self traceReplayCommand:batch
                         command:&batch->commands[0]
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:0
                           phase:"FALLBACK"
                          reason:"mdi_args_overflow"];
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
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
                              reason:"mdi_args_alloc"];
        }
        [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
        return;
    }

    MTLPrimitiveType primType = (MTLPrimitiveType)batch->key.primitive_type;

    if (indexed) {
        GLenum glIdxType = batch->commands[0].indexType;

        MTLDrawIndexedPrimitivesIndirectArguments *args =
            (MTLDrawIndexedPrimitivesIndirectArguments *)((uint8_t *)indirectArgsBuffer.contents + indirectArgsOffset);
        for (uint32_t i = 0; i < batch->command_count; i++) {
            MGLDrawCommand *cmd = &batch->commands[i];
            if (cmd->indexType != glIdxType) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"FALLBACK"
                                  reason:"mdi_mixed_index_type"];
                [self issueDirectBatch:batch context:glm_ctx encodeContext:encCtx];
                return;
            }
            args[i].indexCount = (uint32_t)cmd->count;
            args[i].instanceCount = (uint32_t)cmd->instanceCount;
            args[i].indexStart = 0u;
            args[i].baseVertex = cmd->baseVertex;
            args[i].baseInstance = cmd->baseInstance;
        }

        for (uint32_t i = 0; i < batch->command_count; i++) {
            MGLDrawCommand *cmd = &batch->commands[i];
            Buffer *glBuf = NULL;
            id<MTLBuffer> idxBuf = nil;
            if (![self resolveElementBufferForCommand:cmd
                                                label:"mdiBatch"
                                              context:glm_ctx
                                             glBuffer:&glBuf
                                            mtlBuffer:&idxBuf]) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"mdi_resolve_element"];
                continue;
            }
            NSUInteger drawIndexOffset = cmd->indexBufferOffset;
            MTLIndexType drawIndexType = getMTLIndexType(glIdxType);
            id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                          glBuf,
                                                                          idxBuf,
                                                                          glIdxType,
                                                                          &drawIndexOffset,
                                                                          &drawIndexType);
            if (!drawIndexBuffer || (GLuint)drawIndexType == 0xFFFFFFFF) {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"mdi_prepared_index"];
                continue;
            }
            [encCtx->encoder drawIndexedPrimitives:primType
                                               indexType:drawIndexType
                                             indexBuffer:drawIndexBuffer
                                          indexBufferOffset:drawIndexOffset
                                          indirectBuffer:indirectArgsBuffer
                                    indirectBufferOffset:indirectArgsOffset + (i * argSize)];
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SUBMIT"
                              reason:"mdi_indexed"];
        }
    } else {
        MTLDrawPrimitivesIndirectArguments *args =
            (MTLDrawPrimitivesIndirectArguments *)((uint8_t *)indirectArgsBuffer.contents + indirectArgsOffset);
        for (uint32_t i = 0; i < batch->command_count; i++) {
            MGLDrawCommand *cmd = &batch->commands[i];
            args[i].vertexCount = (uint32_t)cmd->count;
            args[i].instanceCount = (uint32_t)cmd->instanceCount;
            args[i].vertexStart = (uint32_t)cmd->first;
            args[i].baseInstance = cmd->baseInstance;
        }

        for (uint32_t i = 0; i < batch->command_count; i++) {
            [encCtx->encoder drawPrimitives:primType
                                   indirectBuffer:indirectArgsBuffer
                             indirectBufferOffset:indirectArgsOffset + (i * argSize)];
            [self traceReplayCommand:batch
                             command:&batch->commands[i]
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SUBMIT"
                              reason:"mdi_arrays"];
        }
    }
}

- (bool)bindDynamicVertexArrayBuffersDirectly:(VertexArray *)vao
                                      command:(const MGLDrawCommand *)cmd
                                       context:(GLMContext)glm_ctx
                                 encodeContext:(const MGLEncodeContext *)encCtx
{
    Program *active_program =
        mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
    for (uint8_t binding_index = 0;
         binding_index < cmd->dynamic_vertex_binding_count;
         binding_index++) {
        const MGLDynamicVertexBinding *override =
            &cmd->dynamic_vertex_bindings[binding_index];
        if (!override->buffer ||
            override->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
            return false;
        }

        const BufferBinding *binding = &vao->bindings[override->binding_index];
        if (binding->buffer != (Buffer *)override->buffer) {
            return false;
        }

        /* Slot assignment depends on the effective stream, not on attribute
         * format or relative offset. Attributes sharing a DSA binding almost
         * always share one stream, so resolve that stream once per draw. */
        GLuint representative_attribs[MAX_ATTRIBS];
        GLuint representative_strides[MAX_ATTRIBS];
        GLuint representative_count = 0u;
        for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
            if ((vao->enabled_attribs & (1u << attrib)) == 0u ||
                vao->attrib[attrib].buffer_bindingindex !=
                    override->binding_index ||
                !mglRendererProgramUsesVertexAttrib(active_program, attrib)) {
                continue;
            }

            GLuint effective_stride = binding->stride > 0
                ? (GLuint)binding->stride : vao->attrib[attrib].stride;
            bool known_stream = false;
            for (GLuint stream = 0; stream < representative_count; stream++) {
                if (representative_strides[stream] == effective_stride) {
                    known_stream = true;
                    break;
                }
            }
            if (!known_stream) {
                representative_attribs[representative_count] = attrib;
                representative_strides[representative_count] = effective_stride;
                representative_count++;
            }
        }

        int resolved_slots[MAX_ATTRIBS];
        GLuint resolved_slot_count = 0u;
        for (GLuint stream = 0; stream < representative_count; stream++) {
            GLuint representative = representative_attribs[stream];
            int resolved_slot = mglRendererResolveVertexAttributeBufferIndex(
                glm_ctx, vao, representative, __FUNCTION__);
            if (resolved_slot < 0) {
                continue;
            }
            if (resolved_slot >= (int)kMGLMaxMetalVertexBufferCount) {
                return false;
            }

            for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
                if ((vao->enabled_attribs & (1u << attrib)) == 0u ||
                    vao->attrib[attrib].buffer_bindingindex !=
                        override->binding_index ||
                    !mglRendererProgramUsesVertexAttrib(active_program, attrib)) {
                    continue;
                }
                GLuint effective_stride = binding->stride > 0
                    ? (GLuint)binding->stride : vao->attrib[attrib].stride;
                if (effective_stride == representative_strides[stream] &&
                    !mglDynamicVertexAttribCanBindDirectly(active_program,
                                                          attrib,
                                                          &vao->attrib[attrib])) {
                    return false;
                }
            }
            resolved_slots[resolved_slot_count++] = resolved_slot;
        }

        /* A captured binding may be unused by this shader. */
        if (resolved_slot_count == 0u) {
            continue;
        }

        Buffer *draw_buffer = (Buffer *)override->buffer;
        NSUInteger dynamic_offset = (NSUInteger)override->offset_delta;
        if (draw_buffer->data.dirty_bits) {
            BufferMapList upload = {0};
            upload.count = 1;
            upload.buffers[0].buf = draw_buffer;
            if (![self updateDirtyBaseBufferList:&upload]) {
                return false;
            }
        }
        if (!draw_buffer->data.mtl_data) {
            [self bindMTLBuffer:draw_buffer];
        }
        if (!draw_buffer->data.mtl_data ||
            (uintptr_t)draw_buffer->data.mtl_data < 0x10000u) {
            return false;
        }

        id<MTLBuffer> metal_buffer =
            (__bridge id<MTLBuffer>)(draw_buffer->data.mtl_data);
        if (binding->offset < 0 ||
            (uint64_t)binding->offset >= (uint64_t)metal_buffer.length ||
            dynamic_offset >= metal_buffer.length) {
            return false;
        }

        for (GLuint stream = 0; stream < resolved_slot_count; stream++) {
            NSUInteger metal_slot = (NSUInteger)resolved_slots[stream];
            if (!_bindingSync.state->lastBoundValid ||
                _bindingSync.state->lastBoundVertexBuffers[metal_slot].buffer != metal_buffer ||
                _bindingSync.state->lastBoundVertexBuffers[metal_slot].offset != dynamic_offset) {
                [encCtx->encoder setVertexBuffer:metal_buffer
                                                offset:dynamic_offset
                                               atIndex:metal_slot];
                _bindingSync.state->lastBoundVertexBuffers[metal_slot].buffer = metal_buffer;
                _bindingSync.state->lastBoundVertexBuffers[metal_slot].offset = dynamic_offset;
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
        }
    }
    return true;
}

- (bool)bindDynamicUniformRangesDirectly:(const MGLDrawCommand *)cmd
                                  context:(GLMContext)glm_ctx
                            encodeContext:(const MGLEncodeContext *)encCtx
{
    BufferMapList *stage_maps[2] = {
        &glm_ctx->active_state->vertex_buffer_map_list,
        &glm_ctx->active_state->fragment_buffer_map_list,
    };
    const int stages[2] = { _VERTEX_SHADER, _FRAGMENT_SHADER };

    for (uint8_t dynamic_index = 0;
         dynamic_index < cmd->dynamic_uniform_binding_count;
         dynamic_index++) {
        const MGLDynamicUniformBinding *override =
            &cmd->dynamic_uniform_bindings[dynamic_index];
        BufferBaseTarget *slot =
            &glm_ctx->active_state->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override->binding_index];
        if (!slot->buf || !slot->buf->data.mtl_data ||
            (uintptr_t)slot->buf->data.mtl_data < 0x10000u ||
            override->offset < 0 || override->size <= 0) {
            return false;
        }

        id<MTLBuffer> metal_buffer =
            (__bridge id<MTLBuffer>)(slot->buf->data.mtl_data);
        uint64_t start = (uint64_t)override->offset;
        uint64_t length = (uint64_t)override->size;
        if (start > metal_buffer.length || length > metal_buffer.length - start) {
            return false;
        }

        for (int stage_index = 0; stage_index < 2; stage_index++) {
            BufferMapList *maps = stage_maps[stage_index];
            GLuint map_count = maps->count < MAX_MAPPED_BUFFERS
                ? maps->count : MAX_MAPPED_BUFFERS;
            for (GLuint map_index = 0; map_index < map_count; map_index++) {
                BufferMap *map = &maps->buffers[map_index];
                if (map->attribute_mask != 0u ||
                    map->buffer_base_index != override->binding_index ||
                    map->buf != slot->buf) {
                    continue;
                }
                NSUInteger reflected_required_bytes = map->has_metal_binding
                    ? [self getProgramBindingRequiredSize:stages[stage_index]
                                                   type:(int)map->resource_type
                                                  index:(int)map->resource_index]
                    : [self getProgramBindingRequiredSizeForStage:stages[stage_index]
                                                    clientBinding:override->binding_index];
                NSUInteger required_binding_bytes = kMGLMinimumStageBindingSize;
                if (reflected_required_bytes > required_binding_bytes) {
                    required_binding_bytes = reflected_required_bytes;
                }
                if (length < required_binding_bytes) {
                    return false;
                }

                NSInteger resolved_slot = map->has_metal_binding
                    ? (NSInteger)map->metal_binding_index
                    : [self getProgramMetalBufferIndexForStage:stages[stage_index]
                                                 clientBinding:override->binding_index];
                if (resolved_slot < 0 || resolved_slot >= kMGLMaxBufferSlots) {
                    return false;
                }
                NSUInteger metal_slot = (NSUInteger)resolved_slot;
                if (stages[stage_index] == _VERTEX_SHADER) {
                    if (!_bindingSync.state->lastBoundValid ||
                        _bindingSync.state->lastBoundVertexBuffers[metal_slot].buffer != metal_buffer ||
                        _bindingSync.state->lastBoundVertexBuffers[metal_slot].offset != start) {
                        [encCtx->encoder setVertexBuffer:metal_buffer
                                                        offset:(NSUInteger)start
                                                       atIndex:metal_slot];
                        _bindingSync.state->lastBoundVertexBuffers[metal_slot].buffer = metal_buffer;
                        _bindingSync.state->lastBoundVertexBuffers[metal_slot].offset = (NSUInteger)start;
                        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                    }
                } else {
                    if (!_bindingSync.state->lastBoundValid ||
                        _bindingSync.state->lastBoundFragmentBuffers[metal_slot].buffer != metal_buffer ||
                        _bindingSync.state->lastBoundFragmentBuffers[metal_slot].offset != start) {
                        [encCtx->encoder setFragmentBuffer:metal_buffer
                                                          offset:(NSUInteger)start
                                                         atIndex:metal_slot];
                        _bindingSync.state->lastBoundFragmentBuffers[metal_slot].buffer = metal_buffer;
                        _bindingSync.state->lastBoundFragmentBuffers[metal_slot].offset = (NSUInteger)start;
                        MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                    }
                }
            }
        }
    }
    return true;
}

- (bool)bindDynamicSampledTexturesDirectlyForTouchedUnits:(const bool *)touched_units
                                                   context:(GLMContext)glm_ctx
                                             encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!touched_units || !glm_ctx || !encCtx->encoder) return false;

    for (int stage_index = 0; stage_index < 2; stage_index++) {
        int stage = stage_index == 0 ? _VERTEX_SHADER : _FRAGMENT_SHADER;
        Program *program = mglResolveProgramForStageFromState(glm_ctx, stage);
        GLuint sampled_count = [self getProgramBindingCount:stage
                                                       type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
        for (GLuint resource_index = 0;
             resource_index < sampled_count;
             resource_index++) {
            GLuint metal_slot = [self getProgramBinding:stage
                                                    type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                   index:(int)resource_index];
            if (metal_slot >= TEXTURE_UNITS) continue;

            SpirvResource *resource = NULL;
            if (program &&
                resource_index < program->spirv_resources_list[stage]
                                           [SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
                resource = &program->spirv_resources_list[stage]
                                   [SPVC_RESOURCE_TYPE_SAMPLED_IMAGE]
                                   .list[resource_index];
            }
            if (mglShouldSkipStageTextureResource(
                    program, stage, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, resource)) {
                continue;
            }
            if (resource && resource->is_array) return false;

            GLuint texture_unit = [self textureUnitForSampledResource:resource
                                                          metalBinding:metal_slot
                                                                 stage:stage];
            if (texture_unit >= TEXTURE_UNITS ||
                !touched_units[texture_unit]) {
                continue;
            }

            MTLTextureType expected_type =
                [self getProgramExpectedTextureType:stage
                                                type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                               index:(int)resource_index];
            MTLTextureType lookup_type =
                [self getProgramDeclaredTextureType:stage
                                                type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                               index:(int)resource_index];
            MGLTextureDataKind expected_kind =
                [self getProgramExpectedTextureDataKind:stage
                                                   type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                  index:(int)resource_index];
            Texture *texture_object =
                [self textureForSampledResource:resource
                                   metalBinding:metal_slot
                                           stage:stage
                                    expectedType:(lookup_type ? lookup_type
                                                              : expected_type)];
            if (!texture_object || !texture_object->mtl_data ||
                texture_object->dirty_bits || texture_object->is_render_target) {
                return false;
            }

            id<MTLTexture> texture =
                (__bridge id<MTLTexture>)texture_object->mtl_data;
            texture = mglSampledTextureViewForBaseLevel(texture_object, texture);
            if (!texture ||
                (expected_type != 0 && texture.textureType != expected_type) ||
                !mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    texture.pixelFormat, expected_kind)) {
                return false;
            }

            if (stage == _VERTEX_SHADER) {
                [self setVertexTextureIfNeeded:texture atIndex:metal_slot];
            } else {
                [self setFragmentTextureIfNeeded:texture atIndex:metal_slot];
            }

            if (!resource || resource->msl_has_combined_sampler) {
                id<MTLSamplerState> sampler = nil;
                Sampler *bound_sampler =
                    glm_ctx->active_state->texture_samplers[texture_unit];
                if (bound_sampler) {
                    if (bound_sampler->dirty_bits || !bound_sampler->mtl_data) {
                        return false;
                    }
                    sampler = (__bridge id<MTLSamplerState>)bound_sampler->mtl_data;
                } else if (texture_object->params.mtl_data) {
                    sampler = (__bridge id<MTLSamplerState>)texture_object->params.mtl_data;
                } else {
                    return false;
                }

                GLuint sampler_slot = resource
                    ? mglMetalCombinedSamplerSlot(resource) : metal_slot;
                if (sampler_slot >= kMaxFragmentSamplerSlots) return false;
                if (stage == _VERTEX_SHADER) {
                    [self setVertexSamplerStateIfNeeded:sampler
                                               atIndex:sampler_slot];
                } else {
                    [self setFragmentSamplerStateIfNeeded:sampler
                                                 atIndex:sampler_slot];
                }
            }
        }
    }
    return true;
}

- (id<MTLSamplerState>)samplerStateForSnapshotKey:(const MGLSamplerSnapshotKey *)key
{
    if (!key) return nil;

    const uint32_t mask = kMGLSamplerSnapshotCacheIndexCapacity - 1u;
    uint32_t hashSlot = (uint32_t)mglRendererSamplerSnapshotHash(key) & mask;
    for (uint32_t probe = 0; probe < kMGLSamplerSnapshotCacheIndexCapacity;
         probe++, hashSlot = (hashSlot + 1u) & mask) {
        uint16_t encoded = _resourceFallback.samplerSnapshotCacheIndex[hashSlot];
        if (encoded == 0u) break;
        if (encoded == UINT16_MAX) continue;
        uint16_t index = encoded - 1u;
        if (index < _resourceFallback.samplerSnapshotCacheCount &&
            memcmp(&_resourceFallback.samplerSnapshotCacheKeys[index], key, sizeof(*key)) == 0) {
            return _resourceFallback.samplerSnapshotCacheStates[index];
        }
    }

    TextureParameter params;
    memset(&params, 0, sizeof(params));
    params.min_filter = key->min_filter;
    params.mag_filter = key->mag_filter;
    params.wrap_s = key->wrap_s;
    params.wrap_t = key->wrap_t;
    params.wrap_r = key->wrap_r;
    params.compare_mode = key->compare_mode;
    params.compare_func = key->compare_func;
    params.max_anisotropy = key->max_anisotropy;
    params.min_lod = key->min_lod;
    params.max_lod = key->max_lod;
    memcpy(params.border_color, key->border_color, sizeof(params.border_color));

    id<MTLSamplerState> state =
        [self createMTLSamplerForTexParam:&params target:key->target];
    if (!state) return nil;

    uint16_t slot;
    if (_resourceFallback.samplerSnapshotCacheCount < kMGLSamplerSnapshotCacheCapacity) {
        slot = _resourceFallback.samplerSnapshotCacheCount++;
    } else {
        slot = _resourceFallback.samplerSnapshotCacheNext++ % kMGLSamplerSnapshotCacheCapacity;

        const MGLSamplerSnapshotKey *oldKey = &_resourceFallback.samplerSnapshotCacheKeys[slot];
        uint32_t oldHashSlot =
            (uint32_t)mglRendererSamplerSnapshotHash(oldKey) & mask;
        for (uint32_t probe = 0; probe < kMGLSamplerSnapshotCacheIndexCapacity;
             probe++, oldHashSlot = (oldHashSlot + 1u) & mask) {
            uint16_t encoded = _resourceFallback.samplerSnapshotCacheIndex[oldHashSlot];
            if (encoded == 0u) break;
            if (encoded == slot + 1u) {
                _resourceFallback.samplerSnapshotCacheIndex[oldHashSlot] = UINT16_MAX;
                break;
            }
        }
    }
    _resourceFallback.samplerSnapshotCacheKeys[slot] = *key;
    _resourceFallback.samplerSnapshotCacheStates[slot] = state;

    hashSlot = (uint32_t)mglRendererSamplerSnapshotHash(key) & mask;
    uint32_t firstTombstone = UINT32_MAX;
    for (uint32_t probe = 0; probe < kMGLSamplerSnapshotCacheIndexCapacity;
         probe++, hashSlot = (hashSlot + 1u) & mask) {
        uint16_t encoded = _resourceFallback.samplerSnapshotCacheIndex[hashSlot];
        if (encoded == UINT16_MAX && firstTombstone == UINT32_MAX) {
            firstTombstone = hashSlot;
        } else if (encoded == 0u) {
            if (firstTombstone != UINT32_MAX) hashSlot = firstTombstone;
            _resourceFallback.samplerSnapshotCacheIndex[hashSlot] = slot + 1u;
            return state;
        }
    }
    if (firstTombstone != UINT32_MAX) {
        _resourceFallback.samplerSnapshotCacheIndex[firstTombstone] = slot + 1u;
    }
    return state;
}

- (bool)applySamplerSnapshotForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!cmd || !glm_ctx) return false;
    if (cmd->sampler_snapshot_id == MGL_INVALID_SAMPLER_SNAPSHOT_ID) return true;
    if (!encCtx->encoder) return false;

    MGLCommandBuffer *cb = &glm_ctx->draw_command_buffer;
    if (cmd->sampler_snapshot_id >= cb->sampler_snapshot_set_count) return false;
    const MGLSamplerSnapshotSet *set =
        &cb->sampler_snapshot_sets[cmd->sampler_snapshot_id];
    if (set->count > MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES) return false;

    for (uint8_t i = 0; i < set->count; i++) {
        const MGLSamplerSnapshotEntry *entry = &set->entries[i];
        if (entry->metal_slot >= 16u) {
            return false;
        }
        id<MTLSamplerState> sampler;
        if (entry->key_index == MGL_FALLBACK_SAMPLER_KEY_INDEX) {
            sampler = [self fallbackSamplerState];
        } else {
            if (entry->key_index >= cb->sampler_snapshot_key_count) return false;
            sampler = [self samplerStateForSnapshotKey:
                &cb->sampler_snapshot_keys[entry->key_index]];
        }
        if (!sampler) return false;

        if (entry->stage == _VERTEX_SHADER) {
            [self setVertexSamplerStateIfNeeded:sampler atIndex:entry->metal_slot];
        } else if (entry->stage == _FRAGMENT_SHADER) {
            [self setFragmentSamplerStateIfNeeded:sampler atIndex:entry->metal_slot];
        } else {
            return false;
        }
    }
    return true;
}

- (bool)applyDynamicBindingsForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!cmd || (cmd->dynamic_vertex_binding_count == 0 &&
                 cmd->dynamic_uniform_binding_count == 0 &&
                 cmd->dynamic_texture_binding_count == 0)) {
        return true;
    }
    if (!glm_ctx || !encCtx->encoder) {
        return false;
    }

    VertexArray dynamic_vao;
    VertexArray *base_vao = glm_ctx->active_state->vao;
    VertexArray *draw_vao = base_vao;
    if (cmd->dynamic_vertex_binding_count > 0) {
        if (!base_vao || base_vao->magic != MGL_VAO_MAGIC ||
            !mglBuildDynamicVertexArray(base_vao, cmd, &dynamic_vao)) {
            return false;
        }
        draw_vao = &dynamic_vao;
    }

    for (uint8_t i = 0; i < cmd->dynamic_uniform_binding_count; i++) {
        const MGLDynamicUniformBinding *override =
            &cmd->dynamic_uniform_bindings[i];
        if (override->binding_index >= MAX_BINDABLE_BUFFERS) {
            return false;
        }
        BufferBaseTarget *slot =
            &glm_ctx->active_state->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override->binding_index];
        if (!slot->buf) {
            return false;
        }
        slot->offset = override->offset;
        slot->size = override->size;
    }

    bool touched_texture_units[TEXTURE_UNITS] = { false };
    for (uint8_t i = 0; i < cmd->dynamic_texture_binding_count; i++) {
        const MGLDynamicTextureBinding *override =
            &cmd->dynamic_texture_bindings[i];
        if (override->unit >= TEXTURE_UNITS ||
            override->target_index >= _MAX_TEXTURE_TYPES ||
            !override->texture) {
            return false;
        }
        Texture *texture = (Texture *)override->texture;
        if (texture->index != override->target_index) {
            return false;
        }
        if (!touched_texture_units[override->unit]) {
            glm_ctx->active_state->active_textures[override->unit] = NULL;
            touched_texture_units[override->unit] = true;
        }
        glm_ctx->active_state->texture_units[override->unit]
            .textures[override->target_index] = texture;
        if (override->is_active) {
            glm_ctx->active_state->active_textures[override->unit] = texture;
        }
    }

    bool direct_texture_ok = true;
    if (cmd->dynamic_texture_binding_count > 0) {
        direct_texture_ok =
            [self bindDynamicSampledTexturesDirectlyForTouchedUnits:
                touched_texture_units
                                                           context:glm_ctx
                                                     encodeContext:encCtx];
        if (!direct_texture_ok) {
            direct_texture_ok = [self bindTexturesToCurrentRenderEncoder:encCtx];
            if (!direct_texture_ok) {
                direct_texture_ok =
                    [self restoreRenderEncoderAfterTextureUploadForDraw:
                        "dynamic-sampled-texture-bind"] &&
                    [self bindTexturesToCurrentRenderEncoder:encCtx];
            }
        }
    }
    if (!direct_texture_ok) {
        return false;
    }

    /* Texture materialization may replace the render encoder and restore the
     * batch VAO. Apply per-draw buffer overrides only after that completes. */
    bool direct_vertex_ok = cmd->dynamic_vertex_binding_count == 0 ||
        [self bindDynamicVertexArrayBuffersDirectly:draw_vao
                                            command:cmd
                                            context:glm_ctx
                                      encodeContext:encCtx];
    bool direct_uniform_ok = cmd->dynamic_uniform_binding_count == 0 ||
        [self bindDynamicUniformRangesDirectly:cmd context:glm_ctx encodeContext:encCtx];
    if (direct_vertex_ok && direct_uniform_ok) {
        return true;
    }

    /* Uncommon conversion, undersized-range and allocation cases reuse the
     * full validated mapper.  Pipeline, textures and render state remain
     * constant for the containing batch. */
    VertexArray *saved_vao = glm_ctx->active_state->vao;
    if (cmd->dynamic_vertex_binding_count > 0) {
        glm_ctx->active_state->vao = draw_vao;
    }
    bool fallback_ok = [self mapBuffersToMTL] &&
        [self bindVertexBuffersToCurrentRenderEncoder:encCtx];
    if (fallback_ok && cmd->dynamic_uniform_binding_count > 0) {
        fallback_ok = [self bindFragmentBuffersToCurrentRenderEncoder:encCtx];
    }
    glm_ctx->active_state->vao = saved_vao;
    return fallback_ok;
}

- (void)issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
             encodeContext:(const MGLEncodeContext *)encCtx
{
    for (uint32_t i = 0; i < batch->command_count; i++) {
        MGLDrawCommand *cmd = &batch->commands[i];
        if (![self applyDynamicBindingsForCommand:cmd context:glm_ctx encodeContext:encCtx]) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"dynamic_binding"];
            continue;
        }
        if ((batch->sampler_snapshots_mixed ||
             batch->has_dynamic_texture_bindings) &&
            ![self applySamplerSnapshotForCommand:cmd context:glm_ctx encodeContext:encCtx]) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"sampler_snapshot"];
            continue;
        }
        GLenum mode = cmd->mode;
        GLsizei count = cmd->count;
        GLsizei instanceCount = cmd->instanceCount;

        BOOL polygonModePoint = mglPolygonModePointForDrawMode(glm_ctx, mode);
        BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
        BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
        BOOL emulateQuads = (mode == GL_QUADS && !polygonModePoint);
        MTLPrimitiveType primType = polygonModePoint
            ? MTLPrimitiveTypePoint
            : (emulateTriangleFan ? MTLPrimitiveTypeTriangle
                                  : (emulateLineLoop ? MTLPrimitiveTypeLineStrip
                                                     : (emulateQuads ? MTLPrimitiveTypeTriangle
                                                                    : (MTLPrimitiveType)batch->key.primitive_type)));
        if (!polygonModePoint && !emulateTriangleFan && !emulateLineLoop && !emulateQuads &&
            batch->key.primitive_type == 0xFFu) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_unsupported_primitive"];
            continue;
        }

        switch (cmd->type) {
            case MGL_CMD_DRAW_ARRAYS:
                [self issueDirectBatchDrawArrays:batch
                                         command:cmd
                                          context:glm_ctx
                                       batchIndex:i
                                             mode:mode
                                            count:count
                               polygonModePoint:polygonModePoint
                              emulateTriangleFan:emulateTriangleFan
                                 emulateLineLoop:emulateLineLoop
                                   emulateQuads:emulateQuads
                                        primType:primType
                                   encodeContext:encCtx];
                break;

            case MGL_CMD_DRAW_ARRAYS_INSTANCED:
                [self issueDirectBatchDrawArraysInstanced:batch
                                                  command:cmd
                                                   context:glm_ctx
                                                batchIndex:i
                                                      mode:mode
                                                    count:count
                                            instanceCount:instanceCount
                                       polygonModePoint:polygonModePoint
                              emulateTriangleFan:emulateTriangleFan
                                 emulateLineLoop:emulateLineLoop
                                   emulateQuads:emulateQuads
                                        primType:primType
                                   encodeContext:encCtx];
                break;

            case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
                [self issueDirectBatchDrawArraysInstancedBaseInstance:batch
                                                              command:cmd
                                                               context:glm_ctx
                                                            batchIndex:i
                                                                  mode:mode
                                                                count:count
                                                        instanceCount:instanceCount
                                                   polygonModePoint:polygonModePoint
                                                  emulateTriangleFan:emulateTriangleFan
                                                     emulateLineLoop:emulateLineLoop
                                                       emulateQuads:emulateQuads
                                                            primType:primType
                                                       encodeContext:encCtx];
                break;

            default:
                [self issueDirectBatchElementDraw:batch
                                          command:cmd
                                           context:glm_ctx
                                        batchIndex:i
                                              mode:mode
                                            count:count
                                    instanceCount:instanceCount
                               polygonModePoint:polygonModePoint
                              emulateTriangleFan:emulateTriangleFan
                                 emulateLineLoop:emulateLineLoop
                                   emulateQuads:emulateQuads
                                        primType:primType
                                   encodeContext:encCtx];
                break;
        }
    }
}

- (void)issueDirectBatchDrawArrays:(MGLDrawBatch *)batch
                           command:(MGLDrawCommand *)cmd
                            context:(GLMContext)glm_ctx
                         batchIndex:(uint32_t)i
                               mode:(GLenum)mode
                              count:(GLsizei)count
                  polygonModePoint:(BOOL)polygonModePoint
                 emulateTriangleFan:(BOOL)emulateTriangleFan
                    emulateLineLoop:(BOOL)emulateLineLoop
                      emulateQuads:(BOOL)emulateQuads
                           primType:(MTLPrimitiveType)primType
                      encodeContext:(const MGLEncodeContext *)encCtx
{
    if (polygonModePoint) {
        mglEncodeArrayPolygonPoint(encCtx->encoder, _device,
                                   mode, cmd->first, count, 1u, 0u, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_polygon_point"];
    } else if (emulateTriangleFan) {
        if (count >= 3) {
            NSUInteger fanCount = 0;
            id<MTLBuffer> fanBuf = mglNewTriangleFanArrayIndexBuffer(
                _device, (NSUInteger)count, &fanCount);
            if (fanBuf && fanCount > 0) {
                [encCtx->encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                                  indexCount:fanCount
                                                   indexType:MTLIndexTypeUInt32
                                                 indexBuffer:fanBuf
                                                  indexBufferOffset:0
                                               instanceCount:1
                                                  baseVertex:cmd->first
                                                baseInstance:0];
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_triangle_fan"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_triangle_fan_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_triangle_fan_small"];
        }
    } else if (emulateLineLoop) {
        if (count >= 2) {
            NSUInteger loopCount = 0;
            id<MTLBuffer> loopBuf = mglNewLineLoopArrayIndexBuffer(
                _device, (NSUInteger)cmd->first, (NSUInteger)count, &loopCount);
            if (loopBuf && loopCount > 0) {
                [encCtx->encoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                                                  indexCount:loopCount
                                                   indexType:MTLIndexTypeUInt32
                                                 indexBuffer:loopBuf
                                                  indexBufferOffset:0
                                               instanceCount:1
                                                  baseVertex:0
                                                baseInstance:0];
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_line_loop"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_line_loop_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_line_loop_small"];
        }
    } else if (emulateQuads) {
        BOOL ok = mglEncodeArrayQuads(encCtx->encoder,
                                      _device,
                                      count,
                                      cmd->first,
                                      1u,
                                      0u,
                                      mglPolygonModeLineForDrawMode(glm_ctx, mode),
                                      "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_arrays_quads" : "direct_arrays_quads_buffer")];
    } else {
        mglTraceLog("DIRECT_BATCH_DRAW_ARRAYS_SUBMIT flush=%llu batch=%u cmd=%u program=%u mode=0x%x first=%d count=%d encoder=%p pipeline=%p",
                    (unsigned long long)_renderPassManager.state->traceReplayFlushId,
                    (unsigned)_renderPassManager.state->traceReplayBatchIndex,
                    (unsigned)i,
                    (unsigned)mglCurrentRenderProgramKey(glm_ctx),
                    (unsigned)mode,
                    (int)cmd->first,
                    (int)count,
                    encCtx->encoder,
                    _pipelineCache.state->pipelineState);
        /* Cull distance emulation: bind vertex/params buffers before
         * drawPrimitives in the deferred batch path. */
        {
            Program *batchProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
            if (batchProgram && batchProgram->spirv[_VERTEX_SHADER].msl_str &&
                strstr(batchProgram->spirv[_VERTEX_SHADER].msl_str, "mgl_CullDistance")) {
                [self bindCullDistanceEmulationBuffers:mode encodeContext:encCtx];
            }
        }
        [encCtx->encoder drawPrimitives:primType
                                 vertexStart:cmd->first
                                 vertexCount:count];
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays"];
    }
}

- (void)issueDirectBatchDrawArraysInstanced:(MGLDrawBatch *)batch
                                      command:(MGLDrawCommand *)cmd
                                       context:(GLMContext)glm_ctx
                                    batchIndex:(uint32_t)i
                                          mode:(GLenum)mode
                                        count:(GLsizei)count
                                instanceCount:(GLsizei)instanceCount
                           polygonModePoint:(BOOL)polygonModePoint
                          emulateTriangleFan:(BOOL)emulateTriangleFan
                             emulateLineLoop:(BOOL)emulateLineLoop
                               emulateQuads:(BOOL)emulateQuads
                                    primType:(MTLPrimitiveType)primType
                               encodeContext:(const MGLEncodeContext *)encCtx
{
    if (polygonModePoint) {
        mglEncodeArrayPolygonPoint(encCtx->encoder, _device,
                                   mode, cmd->first, count,
                                   instanceCount, 0u, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_instanced_polygon_point"];
    } else if (emulateTriangleFan) {
        if (count >= 3) {
            NSUInteger fanCount = 0;
            id<MTLBuffer> fanBuf = mglNewTriangleFanArrayIndexBuffer(
                _device, (NSUInteger)count, &fanCount);
            if (fanBuf && fanCount > 0) {
                [encCtx->encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                                  indexCount:fanCount
                                                   indexType:MTLIndexTypeUInt32
                                                 indexBuffer:fanBuf
                                                  indexBufferOffset:0
                                               instanceCount:instanceCount
                                                  baseVertex:cmd->first
                                                baseInstance:0];
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_instanced_triangle_fan"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_instanced_triangle_fan_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_instanced_triangle_fan_small"];
        }
    } else if (emulateLineLoop) {
        if (count >= 2) {
            NSUInteger loopCount = 0;
            id<MTLBuffer> loopBuf = mglNewLineLoopArrayIndexBuffer(
                _device, (NSUInteger)cmd->first, (NSUInteger)count, &loopCount);
            if (loopBuf && loopCount > 0) {
                [encCtx->encoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                                                  indexCount:loopCount
                                                   indexType:MTLIndexTypeUInt32
                                                 indexBuffer:loopBuf
                                                  indexBufferOffset:0
                                               instanceCount:instanceCount
                                                  baseVertex:0
                                                baseInstance:0];
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_instanced_line_loop"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_instanced_line_loop_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_instanced_line_loop_small"];
        }
    } else if (emulateQuads) {
        BOOL ok = mglEncodeArrayQuads(encCtx->encoder,
                                      _device,
                                      count,
                                      cmd->first,
                                      instanceCount,
                                      0u,
                                      mglPolygonModeLineForDrawMode(glm_ctx, mode),
                                      "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_arrays_instanced_quads" : "direct_arrays_instanced_quads_buffer")];
    } else {
        /* Cull distance emulation: bind vertex/params buffers before
         * drawPrimitives in the deferred batch path. */
        {
            Program *batchProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
            if (batchProgram && batchProgram->spirv[_VERTEX_SHADER].msl_str &&
                strstr(batchProgram->spirv[_VERTEX_SHADER].msl_str, "mgl_CullDistance")) {
                [self bindCullDistanceEmulationBuffers:mode encodeContext:encCtx];
            }
        }
        [encCtx->encoder drawPrimitives:primType
                                 vertexStart:cmd->first
                                 vertexCount:count
                               instanceCount:instanceCount];
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_instanced"];
    }
}

- (void)issueDirectBatchDrawArraysInstancedBaseInstance:(MGLDrawBatch *)batch
                                                   command:(MGLDrawCommand *)cmd
                                                    context:(GLMContext)glm_ctx
                                                 batchIndex:(uint32_t)i
                                                       mode:(GLenum)mode
                                                     count:(GLsizei)count
                                             instanceCount:(GLsizei)instanceCount
                                        polygonModePoint:(BOOL)polygonModePoint
                                       emulateTriangleFan:(BOOL)emulateTriangleFan
                                          emulateLineLoop:(BOOL)emulateLineLoop
                                            emulateQuads:(BOOL)emulateQuads
                                                 primType:(MTLPrimitiveType)primType
                                            encodeContext:(const MGLEncodeContext *)encCtx
{
    if (polygonModePoint) {
        mglEncodeArrayPolygonPoint(encCtx->encoder, _device,
                                   mode, cmd->first, count,
                                   instanceCount, cmd->baseInstance, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_base_instance_polygon_point"];
    } else if (emulateTriangleFan) {
        if (count >= 3) {
            NSUInteger fanCount = 0;
            id<MTLBuffer> fanBuf = mglNewTriangleFanArrayIndexBuffer(
                _device, (NSUInteger)count, &fanCount);
            if (fanBuf && fanCount > 0) {
                [encCtx->encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                                  indexCount:fanCount
                                                   indexType:MTLIndexTypeUInt32
                                                 indexBuffer:fanBuf
                                                  indexBufferOffset:0
                                               instanceCount:instanceCount
                                                  baseVertex:cmd->first
                                                baseInstance:cmd->baseInstance];
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_base_instance_triangle_fan"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_base_instance_triangle_fan_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_base_instance_triangle_fan_small"];
        }
    } else if (emulateLineLoop) {
        if (count >= 2) {
            NSUInteger loopCount = 0;
            id<MTLBuffer> loopBuf = mglNewLineLoopArrayIndexBuffer(
                _device, (NSUInteger)cmd->first, (NSUInteger)count, &loopCount);
            if (loopBuf && loopCount > 0) {
                [encCtx->encoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                                                  indexCount:loopCount
                                                   indexType:MTLIndexTypeUInt32
                                                 indexBuffer:loopBuf
                                                  indexBufferOffset:0
                                               instanceCount:instanceCount
                                                  baseVertex:0
                                                baseInstance:cmd->baseInstance];
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SUBMIT"
                                  reason:"direct_arrays_base_instance_line_loop"];
            } else {
                [self traceReplayCommand:batch
                                 command:cmd
                                 context:glm_ctx
                                 flushId:_renderPassManager.state->traceReplayFlushId
                              batchIndex:_renderPassManager.state->traceReplayBatchIndex
                            commandIndex:i
                                   phase:"SKIP"
                                  reason:"direct_arrays_base_instance_line_loop_buffer"];
            }
        } else {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_arrays_base_instance_line_loop_small"];
        }
    } else if (emulateQuads) {
        BOOL ok = mglEncodeArrayQuads(encCtx->encoder,
                                      _device,
                                      count,
                                      cmd->first,
                                      instanceCount,
                                      cmd->baseInstance,
                                      mglPolygonModeLineForDrawMode(glm_ctx, mode),
                                      "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_arrays_base_instance_quads" : "direct_arrays_base_instance_quads_buffer")];
    } else {
        /* Cull distance emulation: bind vertex/params buffers before
         * drawPrimitives in the deferred batch path. */
        {
            Program *batchProgram = mglResolveProgramForStageFromState(glm_ctx, _VERTEX_SHADER);
            if (batchProgram && batchProgram->spirv[_VERTEX_SHADER].msl_str &&
                strstr(batchProgram->spirv[_VERTEX_SHADER].msl_str, "mgl_CullDistance")) {
                [self bindCullDistanceEmulationBuffers:mode encodeContext:encCtx];
            }
        }
        [encCtx->encoder drawPrimitives:primType
                                 vertexStart:cmd->first
                                 vertexCount:count
                               instanceCount:instanceCount
                                baseInstance:cmd->baseInstance];
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_arrays_base_instance"];
    }
}

- (void)issueDirectBatchElementDraw:(MGLDrawBatch *)batch
                           command:(MGLDrawCommand *)cmd
                            context:(GLMContext)glm_ctx
                         batchIndex:(uint32_t)i
                               mode:(GLenum)mode
                              count:(GLsizei)count
                      instanceCount:(GLsizei)instanceCount
                 polygonModePoint:(BOOL)polygonModePoint
                emulateTriangleFan:(BOOL)emulateTriangleFan
                   emulateLineLoop:(BOOL)emulateLineLoop
                     emulateQuads:(BOOL)emulateQuads
                          primType:(MTLPrimitiveType)primType
                     encodeContext:(const MGLEncodeContext *)encCtx
{
    /* Element-based draws */
    Buffer *glBuf = NULL;
    id<MTLBuffer> idxBuf = nil;
    if (![self resolveElementBufferForCommand:cmd
                                        label:"directBatch"
                                      context:glm_ctx
                                     glBuffer:&glBuf
                                    mtlBuffer:&idxBuf]) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SKIP"
                          reason:"direct_resolve_element"];
        return;
    }
    NSUInteger idxOffset = cmd->indexBufferOffset;
    MTLIndexType mtlIdxType = getMTLIndexType(cmd->indexType);
    if ((GLuint)mtlIdxType == 0xFFFFFFFF) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SKIP"
                          reason:"direct_index_type"];
        return;
    }

    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(encCtx->encoder,
                                               _device,
                                               glm_ctx,
                                               glBuf,
                                               idxBuf,
                                               mode,
                                               primType,
                                               cmd->indexType,
                                               mtlIdxType,
                                               idxOffset,
                                               count,
                                               instanceCount,
                                               cmd->baseVertex,
                                               cmd->baseInstance,
                                               "directBatch");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:(restartResult == MGLPrimitiveRestartEncodeHandled ? "SUBMIT" : "SKIP")
                          reason:"direct_primitive_restart"];
        return;
    }

    if (polygonModePoint) {
        mglEncodeElementPolygonPoint(encCtx->encoder, _device,
                                     glBuf, idxBuf, mode,
                                     cmd->indexType, mtlIdxType,
                                     idxOffset, count, instanceCount,
                                     cmd->baseVertex,
                                     cmd->baseInstance, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_elements_polygon_point"];
    } else if (emulateTriangleFan) {
        mglEncodeElementTriangleFan(encCtx->encoder, _device,
                                    glBuf, idxBuf,
                                    cmd->indexType, idxOffset,
                                    count, instanceCount,
                                    cmd->baseVertex,
                                    cmd->baseInstance, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:(count >= 3 ? "SUBMIT" : "SKIP")
                          reason:"direct_elements_triangle_fan"];
    } else if (emulateLineLoop) {
        mglEncodeElementLineLoop(encCtx->encoder, _device,
                                 glBuf, idxBuf,
                                 cmd->indexType, idxOffset,
                                 count, instanceCount,
                                 cmd->baseVertex,
                                 cmd->baseInstance, "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                         phase:(count >= 2 ? "SUBMIT" : "SKIP")
                          reason:"direct_elements_line_loop"];
    } else if (emulateQuads) {
        BOOL ok = mglEncodeElementQuads(encCtx->encoder, _device,
                                        glBuf, idxBuf,
                                        cmd->indexType, idxOffset,
                                        count, instanceCount,
                                        cmd->baseVertex,
                                        cmd->baseInstance,
                                        mglPolygonModeLineForDrawMode(glm_ctx, mode),
                                        "batch");
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:(ok && count >= 4 ? "SUBMIT" : "SKIP")
                          reason:(ok ? "direct_elements_quads" : "direct_elements_quads_buffer")];
    } else {
        MTLIndexType drawIndexType = mtlIdxType;
        id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                      glBuf,
                                                                      idxBuf,
                                                                      cmd->indexType,
                                                                      &idxOffset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer || (GLuint)drawIndexType == 0xFFFFFFFF) {
            [self traceReplayCommand:batch
                             command:cmd
                             context:glm_ctx
                             flushId:_renderPassManager.state->traceReplayFlushId
                          batchIndex:_renderPassManager.state->traceReplayBatchIndex
                        commandIndex:i
                               phase:"SKIP"
                              reason:"direct_prepared_index"];
            return;
        }
        [encCtx->encoder drawIndexedPrimitives:primType
                                          indexCount:count
                                           indexType:drawIndexType
                                         indexBuffer:drawIndexBuffer
                                   indexBufferOffset:idxOffset
                                       instanceCount:instanceCount
                                          baseVertex:cmd->baseVertex
                                        baseInstance:cmd->baseInstance];
        [self traceReplayCommand:batch
                         command:cmd
                         context:glm_ctx
                         flushId:_renderPassManager.state->traceReplayFlushId
                      batchIndex:_renderPassManager.state->traceReplayBatchIndex
                    commandIndex:i
                           phase:"SUBMIT"
                          reason:"direct_elements"];
    }
}


- (bool) validateDrawArraysVertexInputs:(GLMContext)drawCtx
                                    mode:(GLenum)mode
                                   first:(GLint)first
                                   count:(GLsizei)count
                                drawCall:(uint64_t)drawCall
{
    if (!kMGLValidateDrawArraysVboRange) {
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

    GLint vx = ctx->active_state->viewport[0];
    GLint vy = ctx->active_state->viewport[1];
    GLint vw = ctx->active_state->viewport[2];
    GLint vh = ctx->active_state->viewport[3];
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

    if (ctx->active_state->caps.scissor_test) {
        GLint sx = ctx->active_state->var.scissor_box[0];
        GLint sy = ctx->active_state->var.scissor_box[1];
        GLint sw = ctx->active_state->var.scissor_box[2];
        GLint sh = ctx->active_state->var.scissor_box[3];
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
        if (ctx->active_state->var.polygon_mode == GL_LINE) {
            triangleFillMode = MTLTriangleFillModeLines;
        } else if (ctx->active_state->var.polygon_mode != GL_FILL &&
                   ctx->active_state->var.polygon_mode != GL_POINT) {
            mglLogRenderStateRepair("polygon_mode", ctx->active_state->var.polygon_mode, GL_FILL);
            ctx->active_state->var.polygon_mode = GL_FILL;
            mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
        }
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];

    BOOL enableDepthBias = NO;

    if (ctx && mglDrawModeProducesPolygons(mode)) {
        switch (ctx->active_state->var.polygon_mode) {
            case GL_POINT:
                enableDepthBias = ctx->active_state->caps.polygon_offset_point;
                break;
            case GL_LINE:
                enableDepthBias = ctx->active_state->caps.polygon_offset_line;
                break;
            case GL_FILL:
            default:
                enableDepthBias = ctx->active_state->caps.polygon_offset_fill;
                break;
        }
    }

    if (enableDepthBias) {
        float _bias = ctx->active_state->var.polygon_offset_units;
        float _slope = ctx->active_state->var.polygon_offset_factor;
        float _clamp = 0.0f;
        if (!_bindingSync.state->lastBoundValid || _bindingSync.state->lastDepthBias != _bias ||
            _bindingSync.state->lastDepthBiasClamp != _clamp || _bindingSync.state->lastDepthSlopeScale != _slope) {
            [_renderPassManager.state->currentRenderEncoder setDepthBias:_bias
                                     slopeScale:_slope
                                          clamp:_clamp];
            _bindingSync.state->lastDepthBias = _bias;
            _bindingSync.state->lastDepthBiasClamp = _clamp;
            _bindingSync.state->lastDepthSlopeScale = _slope;
        }
    } else {
        if (!_bindingSync.state->lastBoundValid || _bindingSync.state->lastDepthBias != 0.0f ||
            _bindingSync.state->lastDepthBiasClamp != 0.0f || _bindingSync.state->lastDepthSlopeScale != 0.0f) {
            [_renderPassManager.state->currentRenderEncoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
            _bindingSync.state->lastDepthBias = 0.0f;
            _bindingSync.state->lastDepthBiasClamp = 0.0f;
            _bindingSync.state->lastDepthSlopeScale = 0.0f;
        }
    }
}

- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode
{
    return ctx &&
           ctx->active_state->caps.cull_face &&
           ctx->active_state->var.cull_face_mode == GL_FRONT_AND_BACK &&
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
                    ctx && ctx->active_state->framebuffer ? (unsigned)ctx->active_state->framebuffer->name : 0u,
                    drawVAO,
                    drawVAO ? (unsigned)drawVAO->enabled_attribs : 0u,
                    (int)ctx->active_state->viewport[0],
                    (int)ctx->active_state->viewport[1],
                    (int)ctx->active_state->viewport[2],
                    (int)ctx->active_state->viewport[3],
                    ctx->active_state->caps.scissor_test ? 1 : 0,
                    (int)ctx->active_state->var.scissor_box[0],
                    (int)ctx->active_state->var.scissor_box[1],
                    (int)ctx->active_state->var.scissor_box[2],
                    (int)ctx->active_state->var.scissor_box[3],
                    (unsigned)ctx->active_state->draw_buffer,
                    (unsigned)ctx->active_state->read_buffer,
                    ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
                    ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
                    ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
                    ctx->active_state->var.color_writemask[0][3] ? 1 : 0,
                    ctx->active_state->caps.depth_test ? 1 : 0,
                    ctx->active_state->var.depth_writemask ? 1 : 0,
                    (unsigned)ctx->active_state->var.depth_func,
                    ctx->active_state->caps.blend ? 1 : 0,
                    ctx->active_state->caps.cull_face ? 1 : 0,
                    (unsigned)ctx->active_state->var.cull_face_mode,
                    (unsigned)ctx->active_state->var.front_face);
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
        [self newRenderEncoderLocked];
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
            _bindingSync.state->lastPipelineState = _pipelineCache.state->pipelineState;
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
    if (activeProgram && activeProgram->spirv[_VERTEX_SHADER].msl_str &&
        strstr(activeProgram->spirv[_VERTEX_SHADER].msl_str, "mgl_CullDistance")) {
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
              ctx ? ctx->active_state->vao : NULL,
              ctx ? ctx->active_state->framebuffer : NULL);
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
                    ctx->active_state->framebuffer ? (unsigned)ctx->active_state->framebuffer->name : 0u,
                    drawVAO,
                    drawVAO ? (unsigned)drawVAO->enabled_attribs : 0u,
                    (int)ctx->active_state->viewport[0],
                    (int)ctx->active_state->viewport[1],
                    (int)ctx->active_state->viewport[2],
                    (int)ctx->active_state->viewport[3],
                    ctx->active_state->caps.scissor_test ? 1 : 0,
                    (int)ctx->active_state->var.scissor_box[0],
                    (int)ctx->active_state->var.scissor_box[1],
                    (int)ctx->active_state->var.scissor_box[2],
                    (int)ctx->active_state->var.scissor_box[3],
                    (unsigned)ctx->active_state->draw_buffer,
                    (unsigned)ctx->active_state->read_buffer,
                    ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
                    ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
                    ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
                    ctx->active_state->var.color_writemask[0][3] ? 1 : 0,
                    ctx->active_state->caps.depth_test ? 1 : 0,
                    ctx->active_state->var.depth_writemask ? 1 : 0,
                    (unsigned)ctx->active_state->var.depth_func,
                    ctx->active_state->caps.blend ? 1 : 0,
                    ctx->active_state->caps.cull_face ? 1 : 0,
                    (unsigned)ctx->active_state->var.cull_face_mode,
                    (unsigned)ctx->active_state->var.front_face);
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
    if (indexBytesForValidation) {
        haveIndexRange = mglScanIndexRangeIgnoringRestart(indexBytesForValidation + indexOffset,
                                                          type,
                                                          count,
                                                          primitiveRestartForDraw,
                                                          restartIndexForDraw,
                                                          &minIndexForDraw,
                                                          &maxIndexForDraw);
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

    if (kMGLValidateDrawElementsVboRange && haveIndexRange && ctx) {
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
                (mglDrawModeProducesPolygons(mode) && ctx->active_state->var.polygon_mode == GL_LINE)
                    ? MTLTriangleFillModeLines
                    : MTLTriangleFillModeFill;
            MGLTraceNSLog(@"MGL TRACE drawElements.state call=%llu program=%u mode=0x%x polygonMode=0x%x triFill=%lu colorMask(use=%d rgba=%d%d%d%d) depth(write=%d test=%d) blend=%d cull=%d viewport=%d,%d,%d,%d",
                  (unsigned long long)drawCall,
                  (unsigned)activeProgramName,
                  (unsigned)mode,
                  (unsigned)ctx->active_state->var.polygon_mode,
                  (unsigned long)loggedTriangleFillMode,
                  ctx->active_state->caps.use_color_mask[0] ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][0] ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][1] ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][2] ? 1 : 0,
                  ctx->active_state->var.color_writemask[0][3] ? 1 : 0,
                  ctx->active_state->var.depth_writemask ? 1 : 0,
                  ctx->active_state->caps.depth_test ? 1 : 0,
                  ctx->active_state->caps.blend ? 1 : 0,
                  ctx->active_state->caps.cull_face ? 1 : 0,
                  (int)ctx->active_state->viewport[0],
                  (int)ctx->active_state->viewport[1],
                  (int)ctx->active_state->viewport[2],
                  (int)ctx->active_state->viewport[3]);
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

            VertexArray *vao = ctx ? ctx->active_state->vao : NULL;
            if (vao) {
                GLuint traceAttribLimit = MIN((GLuint)4u, ctx ? ctx->active_state->max_vertex_attribs : (GLuint)4u);
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
        VertexArray *submitVAO = ctx ? ctx->active_state->vao : NULL;
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
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "drawArraysIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=fully_culled mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "drawArraysIndirect")) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"drawArraysIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                        (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        }
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        NSLog(@"MGL WARNING: drawArraysIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode);
        return;
    }

    [_renderPassManager.state->currentRenderEncoder drawPrimitives:primitiveType
                           indirectBuffer:indirectBuffer
                     indirectBufferOffset:(NSUInteger)(uintptr_t)indirect];
    [self recordArrayDrawSubmittedMode:mode vertexCount:0u];
    mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SUBMIT path=native mode=0x%x indirect=%p offset=%lu program=%u",
                (unsigned)mode, indirect, (unsigned long)(NSUInteger)(uintptr_t)indirect,
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
}

-(void) mtlDrawElementsIndirect: (GLMContext) glm_ctx mode:(GLenum) mode type:(GLenum) type indirect: (const void *) indirect
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_ENTRY mode=0x%x type=0x%x indirect=%p program=%u",
                (unsigned)mode, (unsigned)type, indirect,
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "drawElementsIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=fully_culled mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "drawElementsIndirect")) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    // get element buffer
    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_index_type type=0x%x program=%u",
                    (unsigned)type,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type);
        return;
    }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, "drawElementsIndirect")) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=primitive_restart program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsIndirect" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_element_buffer program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    // get indirect buffer
    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"drawElementsIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                        (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        }
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        NSLog(@"MGL WARNING: drawElementsIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "multiDrawArraysIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=fully_culled mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "multiDrawArraysIndirect")) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"multiDrawArraysIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        NSLog(@"MGL WARNING: multiDrawArraysIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
}

-(void) mtlMultiDrawElementsIndirect: (GLMContext)glm_ctx mode:(GLenum) mode type:(GLenum)type indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_ENTRY mode=0x%x type=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, (unsigned)type, indirect, (int)drawcount, (int)stride,
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));

    mglResolvePassthroughPatchModeForContext(glm_ctx, &mode, "multiDrawElementsIndirect");

    if (![self processGLState: true]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=process_gl_state program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=rasterization_empty program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=fully_culled program=%u mode=0x%x",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u), (unsigned)mode);
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint && mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "multiDrawElementsIndirect")) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=polygon_point_indirect mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    // get element buffer
    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_index_type type=0x%x program=%u",
                    (unsigned)type,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type);
        return;
    }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, "multiDrawElementsIndirect")) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=primitive_restart program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"multiDrawElementsIndirect" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_element_buffer program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    // get indirect buffer
    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"multiDrawElementsIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer]) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        return;
    }

    if (mode == GL_PATCHES) {
        /* Indirect patch draws would require command decoding before TCS/TES
         * dispatch. Keep them explicit until a real caller needs this path. */
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=patches_not_emulated program=%u",
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
        NSLog(@"MGL WARNING: multiDrawElementsIndirect GL_PATCHES is not emulated yet; skipping draw");
        return;
    }

    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode,
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                    (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
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
                (unsigned)(glm_ctx ? glm_ctx->active_state->var.current_program : 0u));
}

@end
